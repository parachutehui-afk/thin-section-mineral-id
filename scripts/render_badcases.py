"""Render bad-case / good-case overlays for the README.

Each image: the PPL patch with
  * SAM argmax mask in translucent red,
  * SAM oracle (best-of-3) mask in translucent blue, when it differs,
  * GT ellipse outline in green (a green frame if the GT is the whole patch).

Run after scripts/segment.py.

Usage:
    <demo1-venv>/bin/python scripts/render_badcases.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent

CASES = {
    # ranking failure: oracle ~1.0, argmax picked a speck
    "badcase_ranking_speck": "GMAS_LOTE__216__9",
    # top mask is the whole patch, grain is a thin sliver
    "badcase_whole_patch": "GMAS_LOTE2__Muestra029__352",
    # SAM genuinely struggles: even the oracle mask is mediocre
    "badcase_thin_grain": "GMAS_LOTE2__Muestra038__241",
    # good case for contrast: medium grain SAM actually segmented
    "goodcase_feldspar": "GMAS_LOTE__256__567",
}


def up(mask: np.ndarray, size: int = 512) -> np.ndarray:
    return np.array(Image.fromarray((mask.astype(np.uint8) * 255))
                    .resize((size, size), Image.NEAREST)) > 0


def edge_of(mask: np.ndarray) -> np.ndarray:
    return ((mask ^ np.roll(mask, 1, axis=0))
            | (mask ^ np.roll(mask, 1, axis=1)))


def blend(arr: np.ndarray, m: np.ndarray, color, alpha=0.45):
    for c, val in enumerate(color):
        arr[:, :, c] = np.where(m, (arr[:, :, c] * (1 - alpha)
                                   + val * alpha).astype(np.int32),
                                arr[:, :, c])


def main() -> None:
    rows = {r["grain_id"]: r
            for r in csv.DictReader(open(ROOT / "runs" / "sam_eval" / "results.csv"))}
    out_dir = ROOT / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, gid in CASES.items():
        r = rows[gid]
        ppl = Image.open(ROOT / "data" / "subset" / "eval" / gid / "ppl-0.png")
        ppl = ppl.convert("RGB").resize((512, 512), Image.NEAREST)
        arr = np.array(ppl).astype(np.int32)
        sam = np.load(ROOT / "runs" / "sam_eval" / f"{gid}_sam_argmax.npy")
        blend(arr, up(sam), (255, 0, 0))
        oracle_path = ROOT / "runs" / "sam_eval" / f"{gid}_sam_oracle.npy"
        if oracle_path.exists():
            blend(arr, up(np.load(oracle_path)), (0, 0, 255), alpha=0.35)
        gt = np.load(ROOT / "data" / "masks_eval" / f"{gid}_mask.npy")
        if gt.mean() > 0.95:
            # whole-patch GT: draw a green frame instead of an outline
            arr[:6, :] = [0, 255, 0]
            arr[-6:, :] = [0, 255, 0]
            arr[:, :6] = [0, 255, 0]
            arr[:, -6:] = [0, 255, 0]
        else:
            arr[up(edge_of(gt))] = [0, 255, 0]
        Image.fromarray(arr.astype(np.uint8)).save(out_dir / f"{name}.png")
        print(f"{name}: {gid} {r['label25_name']} "
              f"argmax IoU {r['iou_argmax']} oracle {r['iou_oracle']} "
              f"-> assets/{name}.png")


if __name__ == "__main__":
    main()
