"""Summarise Demo 2 evaluation: SAM2 segmentation + probe classification.

Reads runs/sam_eval/results.csv and runs/probe/results.json (+ the
classification report) and prints the tables that go into the README.
Writes runs/summary.json.

Usage:
    <demo1-venv>/bin/python scripts/evaluate.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SAM_CSV = ROOT / "runs" / "sam_eval" / "results.csv"
PROBE_JSON = ROOT / "runs" / "probe" / "results.json"


def main() -> None:
    rows = list(csv.DictReader(open(SAM_CSV)))
    print(f"SAM eval grains: {len(rows)}")
    iou_a = np.array([float(r["iou_argmax"]) for r in rows])
    iou_o = np.array([float(r["iou_oracle"]) for r in rows])
    whole = np.array([int(r["whole_patch"]) for r in rows]) == 1
    sec = np.array([float(r["sec_per_image"]) for r in rows])

    def summ(name, arr):
        print(f"  {name}: mean {arr.mean():.3f} median {np.median(arr):.3f} "
              f"IoU>0.5: {(arr > 0.5).mean() * 100:.1f}% "
              f"IoU>0.75: {(arr > 0.75).mean() * 100:.1f}%")

    print("All grains:")
    summ("argmax-score mask IoU", iou_a)
    summ("oracle best-of-3 IoU", iou_o)
    print(f"Whole-patch grains (GT ellipse >95% of patch): {whole.sum()}")
    if (~whole).sum():
        print("Excluding whole-patch grains:")
        summ("argmax-score mask IoU", iou_a[~whole])
        summ("oracle best-of-3 IoU", iou_o[~whole])
    print(f"SAM time: mean {sec.mean():.1f} s/image "
          f"(min {sec.min():.1f}, max {sec.max():.1f})")

    # per-class IoU (argmax), excluding whole-patch grains
    print("\nPer-class mean IoU (argmax, excl. whole-patch):")
    per_class = {}
    for r, a, o, w in zip(rows, iou_a, iou_o, whole):
        if w:
            continue
        per_class.setdefault(r["label25_name"], []).append((a, o))
    for name in sorted(per_class, key=lambda n: -np.mean([x[0] for x in per_class[n]])):
        vals = per_class[name]
        per_class[name] = {
            "n": len(vals),
            "iou_argmax": round(float(np.mean([x[0] for x in vals])), 3),
            "iou_oracle": round(float(np.mean([x[1] for x in vals])), 3),
        }
        print(f"  {name:20s} n={len(vals):3d} "
              f"argmax {per_class[name]['iou_argmax']:.3f} "
              f"oracle {per_class[name]['iou_oracle']:.3f}")

    summary = {
        "sam": {
            "n": len(rows),
            "n_whole_patch": int(whole.sum()),
            "iou_argmax_mean": round(float(iou_a.mean()), 3),
            "iou_argmax_median": round(float(np.median(iou_a)), 3),
            "iou_argmax_gt05_pct": round(float((iou_a > 0.5).mean() * 100), 1),
            "iou_oracle_mean": round(float(iou_o.mean()), 3),
            "iou_oracle_median": round(float(np.median(iou_o)), 3),
            "iou_oracle_gt05_pct": round(float((iou_o > 0.5).mean() * 100), 1),
            "iou_argmax_mean_excl_whole": round(float(iou_a[~whole].mean()), 3) if (~whole).sum() else None,
            "iou_oracle_mean_excl_whole": round(float(iou_o[~whole].mean()), 3) if (~whole).sum() else None,
            "sec_per_image_mean": round(float(sec.mean()), 1),
            "per_class": per_class,
        },
    }
    if PROBE_JSON.exists():
        probe = json.loads(PROBE_JSON.read_text())
        summary["probe"] = probe
        print("\nProbe (linear, ResNet18 frozen features):")
        for variant, m in probe.items():
            print(f"  {variant}: acc25 {m['acc_25']} "
                  f"macro-F1 {m['macro_f1_25']} "
                  f"Quartz-vs-rest acc {m['acc_binary_quartz']}")
    (ROOT / "runs" / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\nwrote runs/summary.json")


if __name__ == "__main__":
    main()
