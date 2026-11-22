"""Batch SAM2 centre-point segmentation eval on the LITHOS eval subset.

Runs SAM2.1-hiera-tiny (official sam2 library, CPU) on the 300 sam_eval
grains: one centre-point prompt (128, 128) per 256x256 PPL patch, and
scores all 3 returned masks against the GT ellipse mask
(scripts/build_gt_masks.py) with IoU.

Run from the sam2 source dir so the config path resolves:
    cd smoke-test/sam2-main
    <demo1-venv>/bin/python ../../scripts/segment.py
Expected ~11 s/image on this VM -> ~55 min for 300 patches.
Results are written incrementally (runs/sam_eval/results.csv) so a restart
resumes where it stopped.

Notes:
* predict() returns 3 masks whose predicted-IoU scores are NOT sorted and
  -- measured on thin sections -- do NOT rank masks correctly (the top
  score can be a tiny speck). We therefore report BOTH:
    - iou_argmax: the mask with the highest predicted score (standard SAM
      usage; this is the honest headline), and
    - iou_oracle: the best of the 3 masks (upper bound: SAM did propose a
      usable mask, it just did not rank it first).
  The gap between them is itself a finding about zero-shot SAM on
  polarised-light microscopy.
* GT masks are ellipses fitted from the annotated grain axes, not
  hand-drawn boundaries (see build_gt_masks.py). Mean IoU therefore
  measures agreement with that approximation; irregular grains cap it
  below 1.0 even for a perfect segmenter. Grains whose GT ellipse covers
  >95% of the patch (grain bigger than the patch) are flagged
  (whole_patch=1) and reported separately.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
MASK_DIR = ROOT / "data" / "masks_eval"
SUBSET = ROOT / "data" / "subset" / "eval"
RUNS = ROOT / "runs" / "sam_eval"
CKPT = ROOT / "smoke-test" / "checkpoints" / "sam2.1_hiera_tiny.pt"


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--limit", type=int, default=None,
                    help="only run the first N sam_eval grains (smoke run)")
    args = ap.parse_args()

    torch.set_num_threads(4)
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    t0 = time.perf_counter()
    model = build_sam2("configs/sam2.1/sam2.1_hiera_t.yaml", str(CKPT),
                       device="cpu")
    pred = SAM2ImagePredictor(model)
    print(f"model loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    rows = [r for r in csv.DictReader(open(ROOT / "data" / "index.csv"))
            if r["split"] == "eval" and int(r["sam_eval"]) == 1]
    if args.limit:
        rows = rows[: args.limit]
    print(f"{len(rows)} grains to segment", flush=True)

    RUNS.mkdir(parents=True, exist_ok=True)
    out_csv = RUNS / "results.csv"
    done = set()
    if out_csv.exists():
        with open(out_csv) as fh:
            done = {r["grain_id"] for r in csv.DictReader(fh)}
        print(f"resuming: {len(done)} already done", flush=True)

    fields = ["grain_id", "label25", "label25_name", "binary",
              "sam_scores", "sam_areas", "ious", "iou_argmax",
              "iou_oracle", "gt_area_frac", "whole_patch",
              "sec_per_image"]
    write_header = not out_csv.exists()
    with open(out_csv, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        if write_header:
            w.writeheader()
        for r in rows:
            gid = r["grain_id"]
            if gid in done:
                continue
            img = np.array(Image.open(SUBSET / gid / "ppl-0.png")
                           .convert("RGB"))
            gt = np.load(MASK_DIR / f"{gid}_mask.npy")
            t1 = time.perf_counter()
            pred.set_image(img)
            masks, scores, _ = pred.predict(
                point_coords=np.array([[128, 128]]),
                point_labels=np.array([1]))
            dt = time.perf_counter() - t1
            masks_b = [m.astype(bool) for m in masks]
            ious = [iou(m, gt) for m in masks_b]
            best = int(np.argmax(scores))
            oracle = int(np.argmax(ious))
            # save the argmax mask and the oracle mask
            np.save(RUNS / f"{gid}_sam_argmax.npy", masks_b[best])
            if oracle != best:
                np.save(RUNS / f"{gid}_sam_oracle.npy", masks_b[oracle])
            gt_frac = float(gt.mean())
            w.writerow({
                "grain_id": gid,
                "label25": r["label25"],
                "label25_name": r["label25_name"],
                "binary": r["binary"],
                "sam_scores": ";".join(f"{float(s):.3f}" for s in scores),
                "sam_areas": ";".join(f"{float(m.mean()):.4f}"
                                      for m in masks_b),
                "ious": ";".join(f"{v:.4f}" for v in ious),
                "iou_argmax": round(ious[best], 4),
                "iou_oracle": round(ious[oracle], 4),
                "gt_area_frac": round(gt_frac, 4),
                "whole_patch": 1 if gt_frac > 0.95 else 0,
                "sec_per_image": round(dt, 2),
            })
            fh.flush()
            done.add(gid)
            if len(done) % 25 == 0:
                print(f"{len(done)}/{len(rows)} done, "
                      f"last {dt:.1f}s", flush=True)
    print(f"finished {len(done)} grains in "
          f"{(time.perf_counter() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
