"""Build the Demo 2 grain index and extract a stratified subset.

Reads the extracted center_component.csv files (data/annotations/...) and
the Fold CSVs, writes data/index.csv, then extracts the selected grains'
ppl-0.png / xpl-0.png from the zip into data/subset/<split>/<grain_id>/.

Class scheme: the official LITHOS code (data_loader.py) maps the raw
0..28 labels down to 25 classes: raw 4 (Orthoclase), 21 (Biotite),
25 (Olivine) and 26 (Intraclast) are folded into "Other" (new index 24);
the remaining raw labels keep their sorted order. This script reproduces
that mapping exactly (see LABEL25_MAP below, derived from the full data).

Subsets (seeded, deterministic):
  eval  (Fold 2): stratified by 25-class label, cap 40/class  -> ~800 grains
          (probe test set; first 300 flagged sam_eval=1 for the SAM run)
  train (Fold 1): stratified by 25-class label, cap 120/class -> ~2,500 grains
          (probe training set)

Usage:
    python scripts/prepare_data.py
"""
from __future__ import annotations

import csv
import random
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ZIP_PATH = DATA / "lithos-dataset.zip"
ANNOT_ROOT = DATA / "annotations" / "LITHOS_DATASET_P" / "LITHOS_DATASET"
TMP = DATA / "tmp"

# raw label -> (25-class index, 25-class name); built from the official
# LITHOS remap: {4: 24, 21: 24, 25: 24, 26: 24} over sorted unique raws.
_KEEP = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
         19, 20, 22, 23, 24, 27, 28]
_RAW_NAMES = {
    0: "Monocrystalline", 1: "Rock fragment", 2: "Polycrystalline",
    3: "Feldspar", 4: "Orthoclase", 5: "Quartz", 6: "Plagioclase",
    7: "Foraminifer", 8: "Fossil fragment", 9: "Opaque", 10: "Amphibole",
    11: "Calcareous fossil", 12: "Red algae", 13: "Mica", 14: "Calcite",
    15: "Muscovite", 16: "Heavy mineral", 17: "Coral", 18: "Hornblende",
    19: "Microcline", 20: "Pyroxene", 21: "Biotite", 22: "Dolomite",
    23: "Echinoderm", 24: "Sanidine", 25: "Olivine", 26: "Intraclast",
    27: "Crystalline mosaic", 28: "Other",
}
LABEL25_MAP = {raw: i for i, raw in enumerate(_KEEP)}
for raw in (4, 21, 25, 26):
    LABEL25_MAP[raw] = 24
LABEL25_NAMES = [_RAW_NAMES[r] for r in _KEEP]  # index -> name

SEED = 20261122
EVAL_CAP = 40
TRAIN_CAP = 120
SAM_EVAL_N = 300


def grain_id(sec_patch: str) -> str:
    return sec_patch.rsplit("/", 1)[0].replace("/", "__")


def main() -> None:
    # fold membership from the Fold CSVs
    fold_of: dict[str, str] = {}
    for fold, fname in (("Fold1", "Fold1_complete_info_allclasses.csv"),
                        ("Fold2", "Fold2_complete_info_allclasses.csv")):
        with open(TMP / fname, newline="") as fh:
            for row in csv.DictReader(fh):
                fold_of[row["Sec_patch"]] = fold

    rows = []
    label_seen: dict[int, str] = {}
    annot_files = sorted(ANNOT_ROOT.rglob("center_component.csv"))
    print(f"annotation files: {len(annot_files)}")
    for af in annot_files:
        # grain dir relative to ANNOT_ROOT, e.g. GMAS_LOTE/10/1
        rel = af.parent.relative_to(ANNOT_ROOT)
        sec_patch = f"{rel.as_posix()}/ppl-0.png"
        fold = fold_of.get(sec_patch)
        if fold is None:
            print(f"WARNING: {sec_patch} not in any Fold CSV")
            continue
        with open(af, newline="") as fh:
            r = next(csv.DictReader(fh))
        raw = int(r["Multiclass_label"])
        label_seen[raw] = r["Multiclass_label_name"]
        if _RAW_NAMES.get(raw, r["Multiclass_label_name"]) != r["Multiclass_label_name"]:
            print(f"WARNING: raw {raw} name mismatch: "
                  f"{_RAW_NAMES.get(raw)} vs {r['Multiclass_label_name']}")
        rows.append({
            "grain_id": grain_id(sec_patch),
            "split": "train" if fold == "Fold1" else "eval",
            "sec_patch": sec_patch,
            "label_raw": raw,
            "label_name": r["Multiclass_label_name"],
            "label25": LABEL25_MAP[raw],
            "label25_name": LABEL25_NAMES[LABEL25_MAP[raw]],
            "binary": int(r["Binary_label"]),
            "binary_name": r["Binary_label_name"],
            "major_mm": r["major-axis"],
            "minor_mm": r["minor-axis"],
            "x_inter": r["x_inter"],
            "y_inter": r["y_inter"],
            "path_1": r["path_1"],
            "path_2": r["path_2"],
        })
    print("raw labels seen:", {k: label_seen[k] for k in sorted(label_seen)})

    rng = random.Random(SEED)
    by_split_class: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in rows:
        by_split_class[(r["split"], r["label25"])].append(r)

    selected: list[dict] = []
    for (split, cls), members in sorted(by_split_class.items()):
        cap = EVAL_CAP if split == "eval" else TRAIN_CAP
        rng.shuffle(members)
        take = members[:cap]
        selected.extend(take)
        print(f"{split} class {cls:2d} {LABEL25_NAMES[cls]:20s}: "
              f"{len(members):6d} available -> {len(take)}")
    # flag the SAM eval subset: 300 random eval grains
    eval_rows = [r for r in selected if r["split"] == "eval"]
    rng.shuffle(eval_rows)
    sam_ids = {r["grain_id"] for r in eval_rows[:SAM_EVAL_N]}
    for r in selected:
        r["sam_eval"] = 1 if r["grain_id"] in sam_ids else 0

    index_path = DATA / "index.csv"
    fields = ["grain_id", "split", "sam_eval", "sec_patch", "label_raw",
              "label_name", "label25", "label25_name", "binary",
              "binary_name", "major_mm", "minor_mm", "x_inter", "y_inter",
              "path_1", "path_2"]
    with open(index_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in sorted(selected, key=lambda r: (r["split"], r["grain_id"])):
            w.writerow({k: r[k] for k in fields})
    n_train = sum(1 for r in selected if r["split"] == "train")
    n_eval = sum(1 for r in selected if r["split"] == "eval")
    print(f"index: {len(selected)} grains "
          f"(train {n_train}, eval {n_eval}, sam_eval {SAM_EVAL_N}) "
          f"-> {index_path}")

    # extract the selected images from the zip
    z = zipfile.ZipFile(ZIP_PATH)
    subset_root = DATA / "subset"
    n_files = 0
    for r in selected:
        gid = r["grain_id"]
        folder = r["sec_patch"].rsplit("/", 1)[0]
        for suffix in ("ppl-0.png", "xpl-0.png"):
            zpath = f"LITHOS_DATASET_P/LITHOS_DATASET/{folder}/{suffix}"
            dest = subset_root / r["split"] / gid / suffix
            if dest.exists():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(z.read(zpath))
            n_files += 1
    print(f"extracted {n_files} images -> {subset_root}")

    # subset definition record
    (DATA / "lithos.yaml").write_text(
        "# LITHOS subset definition (Demo 2)\n"
        "# Source: Kaggle paolaruizpuentes/lithos-dataset (CC BY-NC-SA 4.0),\n"
        "# full zip kept at data/lithos-dataset.zip; only this subset is unzipped.\n"
        f"# train: Fold 1, stratified by 25-class label, cap {TRAIN_CAP}/class, "
        f"n={n_train}\n"
        f"# eval:  Fold 2, stratified by 25-class label, cap {EVAL_CAP}/class, "
        f"n={n_eval}\n"
        f"# sam_eval: {SAM_EVAL_N} of the eval grains, flagged in data/index.csv\n"
        "# Classes: official LITHOS 25-class scheme (raw 4/21/25/26 -> Other).\n"
        "# GT masks: ellipses fitted from the annotated grain axes\n"
        "# (see scripts/build_gt_masks.py) -- NOT hand-drawn boundaries.\n"
    )
    print("wrote data/lithos.yaml")


if __name__ == "__main__":
    main()
