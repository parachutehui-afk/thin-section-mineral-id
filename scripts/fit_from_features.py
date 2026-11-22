"""Fit the linear probe from already-extracted features.

Used when features.npz exists (e.g. after a train_probe.py run that
failed late, or to re-fit quickly). Replicates the training part of
train_probe.py: logistic regression on StandardScaler-standardised
ResNet18 features, PPL-only and PPL+XPL variants, plus the
classification report, confusion matrices and the pickled model
artifacts app.py needs.

Usage:
    <demo1-venv>/bin/python scripts/fit_from_features.py
"""
from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "probe"


def main() -> None:
    feats = np.load(OUT / "features.npz")
    rows = list(csv.DictReader(open(ROOT / "data" / "index.csv")))
    names = sorted(set((int(r["label25"]), r["label25_name"]) for r in rows))
    class_names = [n for _, n in names]

    results = {}
    for variant, (tr_keys, ev_keys) in {
        "ppl": (["train_ppl"], ["eval_ppl"]),
        "ppl+xpl": (["train_ppl", "train_xpl"], ["eval_ppl", "eval_xpl"]),
    }.items():
        Xtr = np.concatenate([feats[k] for k in tr_keys], axis=1)
        Xev = np.concatenate([feats[k] for k in ev_keys], axis=1)
        ytr, yev = feats["train_y"], feats["eval_y"]
        scaler = StandardScaler().fit(Xtr)
        clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        clf.fit(scaler.transform(Xtr), ytr)
        pred = clf.predict(scaler.transform(Xev))
        proba = clf.predict_proba(scaler.transform(Xev))
        yev_bin = feats["eval_bin"]
        pred_bin = (pred == 4).astype(int)  # label25 4 == Quartz
        results[variant] = {
            "acc_25": round(float(accuracy_score(yev, pred)), 4),
            "macro_f1_25": round(float(f1_score(yev, pred, average="macro")), 4),
            "acc_binary_quartz": round(float(accuracy_score(yev_bin, pred_bin)), 4),
            "n_train": len(ytr), "n_eval": len(yev),
        }
        print(f"[{variant}] acc25={results[variant]['acc_25']} "
              f"macroF1={results[variant]['macro_f1_25']} "
              f"acc_bin={results[variant]['acc_binary_quartz']}", flush=True)
        pkl_name = ("probe_model.pkl" if variant == "ppl+xpl"
                    else "probe_model_ppl.pkl")
        with open(OUT / pkl_name, "wb") as fh:
            pickle.dump({"scaler": scaler, "clf": clf,
                         "classes": class_names,
                         "variant": variant}, fh)
        if variant == "ppl+xpl":
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from sklearn.metrics import ConfusionMatrixDisplay
            cm = confusion_matrix(yev, pred)
            fig, ax = plt.subplots(figsize=(11, 10))
            ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
                ax=ax, cmap="Blues", xticks_rotation=90, colorbar=False)
            ax.set_title("LITHOS 25-class confusion matrix (eval, Fold 2)\n"
                         "ResNet18 frozen features + logistic regression, "
                         "PPL+XPL")
            fig.tight_layout()
            fig.savefig(OUT / "confusion_matrix.png", dpi=130)
            plt.close(fig)
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

    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
