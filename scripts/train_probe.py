"""Linear-probe classifier for LITHOS mineral identification.

Frozen ResNet18 (ImageNet) features + logistic regression, following the
paper's whole-patch classification setup (PolarViT classifies the whole
256x256 patch; we do the same -- the classifier does NOT depend on SAM
mask quality, SAM is evaluated separately as a segmentation task).

Two feature variants, trained on the Fold-1 subset (2,812 grains) and
tested on the Fold-2 subset (1,000 grains):
  * ppl:      ResNet18 features of the PPL patch only (512-d)
  * ppl+xpl:  concatenated PPL + XPL features (1024-d)

Outputs (runs/probe/):
  features.npz, results.json, classification_report.txt,
  confusion_matrix.png (25-class), confusion_matrix_binary.png

Usage:
    <demo1-venv>/bin/python scripts/train_probe.py
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

ROOT = Path(__file__).resolve().parent.parent
INDEX = ROOT / "data" / "index.csv"
SUBSET = ROOT / "data" / "subset"
OUT = ROOT / "runs" / "probe"

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class PatchSet(Dataset):
    def __init__(self, rows, split, pol):
        self.rows = rows
        self.dir = SUBSET / split
        self.pol = pol

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(self.dir / r["grain_id"] / f"{self.pol}-0.png")
        img = img.convert("RGB").resize((224, 224), Image.BILINEAR)
        x = np.asarray(img, dtype=np.float32) / 255.0
        x = (x - MEAN) / STD
        return torch.from_numpy(x.transpose(2, 0, 1)), int(r["label25"])


@torch.no_grad()
def extract(rows, split, pol, model, device, batch=64):
    ds = PatchSet(rows, split, pol)
    dl = DataLoader(ds, batch_size=batch, num_workers=0)
    feats, labels = [], []
    for x, y in dl:
        f = model(x.to(device)).cpu().numpy()
        feats.append(f)
        labels.append(y.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def main() -> None:
    torch.set_num_threads(4)
    device = "cpu"
    rows = list(csv.DictReader(open(INDEX)))
    train_rows = [r for r in rows if r["split"] == "train"]
    eval_rows = [r for r in rows if r["split"] == "eval"]
    print(f"train {len(train_rows)}, eval {len(eval_rows)}", flush=True)

    weights = ResNet18_Weights.IMAGENET1K_V1
    backbone = resnet18(weights=weights)
    model = nn.Sequential(*list(backbone.children())[:-1],
                          nn.Flatten()).to(device).eval()

    t0 = time.perf_counter()
    feats = {}
    for split, rs in (("train", train_rows), ("eval", eval_rows)):
        for pol in ("ppl", "xpl"):
            f, y = extract(rs, split, pol, model, device)
            feats[f"{split}_{pol}"] = f
            print(f"{split}/{pol}: {f.shape} in "
                  f"{time.perf_counter() - t0:.0f}s", flush=True)
    feats["train_y"] = np.array([int(r["label25"]) for r in train_rows])
    feats["eval_y"] = np.array([int(r["label25"]) for r in eval_rows])
    feats["train_bin"] = np.array([int(r["binary"]) for r in train_rows])
    feats["eval_bin"] = np.array([int(r["binary"]) for r in eval_rows])
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "features.npz", **feats)

    results = {}
    for variant, (tr_key, ev_key) in {
        "ppl": (["train_ppl"], ["eval_ppl"]),
        "ppl+xpl": (["train_ppl", "train_xpl"], ["eval_ppl", "eval_xpl"]),
    }.items():
        Xtr = np.concatenate([feats[k] for k in tr_key], axis=1)
        Xev = np.concatenate([feats[k] for k in ev_key], axis=1)
        ytr, yev = feats["train_y"], feats["eval_y"]
        scaler = StandardScaler().fit(Xtr)
        clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        clf.fit(scaler.transform(Xtr), ytr)
        pred = clf.predict(scaler.transform(Xev))
        proba = clf.predict_proba(scaler.transform(Xev))
        # binary Quartz vs Non-Quartz from the 25-class prediction:
        # Quartz is label25 4 in our scheme
        yev_bin = feats["eval_bin"]
        pred_bin = (pred == 4).astype(int)
        # note: binary label 1 == Quartz
        results[variant] = {
            "acc_25": round(float(accuracy_score(yev, pred)), 4),
            "macro_f1_25": round(float(f1_score(yev, pred, average="macro")), 4),
            "acc_binary_quartz": round(float(accuracy_score(yev_bin, pred_bin)), 4),
            "n_train": len(ytr), "n_eval": len(yev),
        }
        print(f"[{variant}] acc25={results[variant]['acc_25']} "
              f"macroF1={results[variant]['macro_f1_25']} "
              f"acc_bin={results[variant]['acc_binary_quartz']}", flush=True)
        if variant == "ppl+xpl":
            from sklearn.metrics import ConfusionMatrixDisplay
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            names = [r for r in
                     sorted(set((int(r["label25"]), r["label25_name"])
                                  for r in rows))]
            class_names = [n for _, n in names]
            cm = confusion_matrix(yev, pred)
            fig, ax = plt.subplots(figsize=(11, 10))
            disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
            disp.plot(ax=ax, cmap="Blues", xticks_rotation=90,
                      colorbar=False)
            ax.set_title("LITHOS 25-class confusion matrix (eval, Fold 2)\n"
                         "ResNet18 frozen features + logistic regression, "
                         "PPL+XPL")
            fig.tight_layout()
            fig.savefig(OUT / "confusion_matrix.png", dpi=130)
            plt.close(fig)
            # binary confusion matrix
            cm_b = confusion_matrix(yev_bin, pred_bin)
            fig, ax = plt.subplots(figsize=(5, 4.5))
            ConfusionMatrixDisplay(
                cm_b, display_labels=["Non-Quartz", "Quartz"]).plot(
                    ax=ax, cmap="Blues", colorbar=False)
            ax.set_title("Quartz vs Non-Quartz (eval, Fold 2)")
            fig.tight_layout()
            fig.savefig(OUT / "confusion_matrix_binary.png", dpi=130)
            plt.close(fig)
            report = classification_report(yev, pred,
                                           target_names=class_names,
                                           digits=3, zero_division=0)
            (OUT / "classification_report.txt").write_text(report)
            np.save(OUT / "eval_pred.npy", pred)
            np.save(OUT / "eval_proba.npy", proba)
            # artifact for app.py: scaler + classifier + class names
            import pickle
            with open(OUT / "probe_model.pkl", "wb") as fh:
                pickle.dump({"scaler": scaler, "clf": clf,
                             "classes": class_names,
                             "variant": "ppl+xpl"}, fh)

    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"total {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
