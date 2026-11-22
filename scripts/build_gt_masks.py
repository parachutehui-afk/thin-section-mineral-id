"""Build ground-truth masks for LITHOS patches from the annotated grain axes.

Key finding (verified 2026-11-22 on a 2,000-grain sample, see README/STATUS):
``path_1`` / ``path_2`` in ``center_component.csv`` are NOT a closed grain
boundary. They are the grain's major/minor AXIS segments:

* each path is a single straight segment, "M x1,y1 L x2,y2", in coordinates
  normalised by the mosaic width (24786 px);
* the two segments are exactly perpendicular (median angle 90.00 deg);
* their length ratio matches the CSV major-axis / minor-axis ratio
  (median relative error 0.16%);
* both segments pass through (x_inter, y_inter), the grain centre.

So the honest GT mask derivable from the annotation is the ELLIPSE fitted
through the four axis endpoints: centre (128, 128) in patch coordinates,
semi-axes = half the segment lengths, rotation = direction of path_1.
It is an elliptical approximation of the central grain, not a hand-drawn
boundary — the README must say so.

Conversion (NeurIPS 2025 supplement, Listing 1):
    patch_coord = path_coord * 24786 + dx,  dx = 128 - x_inter
    patch_coord = path_coord * 24786 + dy,  dy = 128 - y_inter
i.e. the 256x256 patch is a 1:1 crop of the 24,786-px-wide mosaic,
centred on (x_inter, y_inter). The mysterious 24786 is the mosaic width.

Usage:
    python scripts/build_gt_masks.py --index data/index.csv --split eval
        --out data/subset/eval --overlays assets/verify --n-overlay 10
Reads the patch list + annotation paths from the index, writes one
``<grain_id>_mask.npy`` (bool, 256x256) per patch, and renders overlay
images (PPL + ellipse outline + axis lines) for visual verification.
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

MOSAIC_W = 24786  # width of the original thin-section mosaic, in pixels
PATCH = 256

_PATH_RE = re.compile(
    r"^M\s*([-\d.eE+]+),([-\d.eE+]+)\s*L\s*([-\d.eE+]+),([-\d.eE+]+)\s*$"
)


def parse_path(s: str) -> tuple[tuple[float, float], tuple[float, float]]:
    """Parse 'Mx1,y1Lx2,y2' -> ((x1, y1), (x2, y2)) in normalised coords."""
    m = _PATH_RE.match(s.strip())
    if not m:
        raise ValueError(f"unparsable path: {s!r}")
    x1, y1, x2, y2 = (float(v) for v in m.groups())
    return (x1, y1), (x2, y2)


def axes_to_patch(path_1: str, path_2: str, x_inter: float, y_inter: float):
    """Convert the two axis segments to patch coordinates.

    Returns (cx, cy, a, b, angle_deg, p1a, p1b, p2a, p2b) where (cx, cy) is
    the ellipse centre in patch px, a/b are the semi-axis lengths in patch
    px, angle_deg is the direction of path_1, and p?a/p?b are the segment
    endpoints in patch px.
    """
    dx = PATCH / 2 - x_inter
    dy = PATCH / 2 - y_inter

    def conv(p):
        return (p[0] * MOSAIC_W + dx, p[1] * MOSAIC_W + dy)

    (a1, a2) = parse_path(path_1)
    (b1, b2) = parse_path(path_2)
    p1a, p1b = conv(a1), conv(a2)
    p2a, p2b = conv(b1), conv(b2)
    cx, cy = PATCH / 2, PATCH / 2
    v1 = (p1b[0] - p1a[0], p1b[1] - p1a[1])
    v2 = (p2b[0] - p2a[0], p2b[1] - p2a[1])
    a = math.hypot(*v1) / 2
    b = math.hypot(*v2) / 2
    angle = math.degrees(math.atan2(v1[1], v1[0]))
    return cx, cy, a, b, angle, p1a, p1b, p2a, p2b


def ellipse_mask(cx: float, cy: float, a: float, b: float, angle_deg: float,
                 size: int = PATCH) -> np.ndarray:
    """Boolean mask of a rotated ellipse, clipped to the patch."""
    theta = math.radians(angle_deg)
    ux, uy = math.cos(theta), math.sin(theta)   # along major axis
    vx, vy = -uy, ux                            # along minor axis
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    dx = xx - cx
    dy = yy - cy
    u = (dx * ux + dy * uy) / max(a, 1e-6)
    v = (dx * vx + dy * vy) / max(b, 1e-6)
    return (u * u + v * v) <= 1.0


def render_overlay(ppl_path: Path, mask: np.ndarray, axes, out_path: Path,
                   scale: int = 2) -> None:
    """PPL image + green ellipse outline + red axis lines + centre cross."""
    img = Image.open(ppl_path).convert("RGB").resize(
        (PATCH * scale, PATCH * scale), Image.NEAREST)
    # ellipse outline: sample the boundary densely
    cx, cy, a, b, angle, p1a, p1b, p2a, p2b = axes
    theta = math.radians(angle)
    ux, uy = math.cos(theta), math.sin(theta)
    vx, vy = -uy, ux
    pts = []
    for t in np.linspace(0, 2 * math.pi, 361):
        x = cx + a * math.cos(t) * ux + b * math.sin(t) * vx
        y = cy + a * math.cos(t) * uy + b * math.sin(t) * vy
        pts.append((x * scale, y * scale))
    draw = ImageDraw.Draw(img)
    draw.line(pts, fill=(0, 255, 0), width=2)
    for (x1, y1), (x2, y2) in ((p1a, p1b), (p2a, p2b)):
        draw.line([(x1 * scale, y1 * scale), (x2 * scale, y2 * scale)],
                  fill=(255, 0, 0), width=2)
    c = PATCH / 2 * scale
    draw.line([(c - 8, c), (c + 8, c)], fill=(255, 255, 0), width=2)
    draw.line([(c, c - 8), (c, c + 8)], fill=(255, 255, 0), width=2)
    img.save(out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, help="data/index.csv")
    ap.add_argument("--split", required=True, choices=["train", "eval"])
    ap.add_argument("--subset-root", required=True,
                    help="data/subset/<split> with <grain_id>/ppl-0.png")
    ap.add_argument("--out", required=True, help="dir for *_mask.npy")
    ap.add_argument("--overlays", default=None,
                    help="dir for verification overlay PNGs")
    ap.add_argument("--n-overlay", type=int, default=10)
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(args.index))
            if r["split"] == args.split]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ov_dir = Path(args.overlays) if args.overlays else None
    if ov_dir:
        ov_dir.mkdir(parents=True, exist_ok=True)

    n_overlay = 0
    areas = []
    for r in rows:
        gid = r["grain_id"]
        grain_dir = Path(args.subset_root) / gid
        ppl = grain_dir / "ppl-0.png"
        axes = axes_to_patch(r["path_1"], r["path_2"],
                             float(r["x_inter"]), float(r["y_inter"]))
        cx, cy, a, b, angle, *_ = axes
        mask = ellipse_mask(cx, cy, a, b, angle)
        np.save(out_dir / f"{gid}_mask.npy", mask)
        areas.append(float(mask.mean()))
        if ov_dir is not None and n_overlay < args.n_overlay and ppl.exists():
            render_overlay(ppl, mask, axes,
                           ov_dir / f"{gid}_overlay.png")
            n_overlay += 1
    areas = np.array(areas)
    print(f"{args.split}: {len(rows)} masks -> {out_dir}")
    print(f"mask area fraction of patch: mean {areas.mean():.3f} "
          f"median {np.median(areas):.3f} min {areas.min():.3f} "
          f"max {areas.max():.3f}")
    print(f"overlays rendered: {n_overlay} -> {ov_dir}")


if __name__ == "__main__":
    main()
