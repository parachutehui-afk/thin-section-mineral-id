# Demo 2 — SAM2 CPU smoke test: PASSED (2026-11-22, verified on this VM)

Status: SAM2 runs on this machine. The only remaining blocker for Demo 2 is the
user's Kaggle token (LITHOS download). No Kaggle data was used or needed for this test.

## What was tested
- Machine: same VM as Demo 1 — 4 vCPU, ~8 GB RAM, no GPU, torch 2.6.0+cpu.
- Model: SAM2.1-hiera-tiny (38.9M params), centre-point prompt, CPU only.
- Proxy image: Demo 1 steel photo `good_scratches_138.jpg` (600×600) — NOT a thin
  section. This test proves install + inference + timing only, not mineral accuracy.
  A real thin-section mask check still needs LITHOS data.

## Results (both routes work)
| Route | Load | Centre-point inference | Total |
|---|---|---|---|
| Official sam2 library (`SAM2ImagePredictor`, `sam2.1_hiera_tiny.pt`) | 2.8 s | 11.1 s (set_image + predict) | 14.0 s |
| HF transformers (`Sam2Model`, `facebook/sam2.1-hiera-tiny`) | 0.5 s | 10.5–10.8 s | ~11 s |

- Official route, 3 masks, IoU scores [0.945, 0.002, 0.986], mask sizes
  [331,670 / 1,141 / 248,827] px of 360,000 — sensible segments, best mask ~69% of frame.
- **Budget correction vs the DRAFT brief (§4):** the brief's "1–4 s/image" was a
  third-party estimate. Measured here is **~11 s/image on this CPU**, so a
  200-patch eval ≈ 37 min, 500 patches ≈ 90 min. Still feasible as a batch/overnight
  run; no GPU rental needed for zero-shot inference. (GPU is still the route if
  fine-tuning is added later.)

## Download workarounds found (record these — the naive commands fail here)
- `huggingface_hub` download of `facebook/sam2.1-hiera-tiny` fails: 403 on the
  xet-read-token endpoint (`policy_denied`). Workaround: `curl` the files from
  `https://huggingface.co/facebook/sam2.1-hiera-tiny/resolve/main/<file>` —
  `config.json`, `preprocessor_config.json`, `model.safetensors` (149 MB) all
  download fine that way. Saved in `smoke-test/hf-sam2-tiny/`.
- `git clone https://github.com/facebookresearch/sam2.git` fails: HTTP 403.
  Workaround: `curl -L https://codeload.github.com/facebookresearch/sam2/zip/refs/heads/main`
  (54 MB zip, unzipped to `smoke-test/sam2-main/`).
- `pip install -e ./sam2-main` fails in build isolation (tries to fetch torch 2.15
  from PyPI). Workaround: `SAM2_BUILD_CUDA=0 pip install --no-build-isolation -e ./sam2-main`.
  Installed into the Demo 1 venv (torch 2.6.0+cpu already there); `transformers 5.24.0`
  was also added to that venv for the HF route.
- Direct checkpoint works fine: `https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt`
  — 156,008,466 bytes, sha256 `7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69`.
  Saved in `smoke-test/checkpoints/sam2.1_hiera_tiny.pt`.

## Kaggle dataset — verified live 2026-11-22 08:37 UTC
- Page `https://www.kaggle.com/datasets/paolaruizpuentes/lithos-dataset` opens and
  matches the DRAFT brief: LITHOS, 211,604 patches / 105,802 grains / 25 classes,
  **19.98 GB, 317k files**, CC BY-NC-SA 4.0, Usability 8.24, 10.2K downloads,
  contents `LITHOS_DATASET_P/` + `Fold1_complete_info_allclasses.csv` +
  `Fold2_complete_info_allclasses.csv`.
- Exact download command once the token is in place:
  `kaggle datasets download paolaruizpuentes/lithos-dataset`
  (kaggle CLI already installed in `demo-2-sam-mineral/venv/`).
- Download strategy from the brief still stands: download once, unzip only a
  2,000–5,000-grain subset (Fold 2 first), don't unzip all 20 GB.

## Files in this folder
- `smoke-test/sam2_smoke.py` — transformers route (note: 4-level point nesting
  `[[[[x, y]]]]`; `post_process_masks` key `reshaped_input_sizes` does not exist
  in transformers 5.24 — script reads `pred_masks` directly and picks the
  best-IoU mask).
- `smoke-test/sam2_official_smoke.py` — official-library route (recommended for Demo 2).
- `smoke-test/checkpoints/sam2.1_hiera_tiny.pt`, `smoke-test/hf-sam2-tiny/` — weights.
- `smoke-test/sam2-main/` — SAM2 source (zip kept as `sam2-main.zip`).

## Build progress (2026-11-22 evening, after the LITHOS download landed)

- **Data**: `data/lithos-dataset.zip` verified (20,029,690,253 bytes, 317,408
  entries). All 105,802 `center_component.csv` extracted to
  `data/annotations/` (414 MB). Full grain index built by
  `scripts/prepare_data.py` -> `data/index.csv`: **3,812 grains**
  (train Fold 1: 2,812, cap 120/class; eval Fold 2: 1,000, cap 40/class;
  sam_eval: 300 of the eval grains). 7,624 PPL+XPL images extracted to
  `data/subset/`. Raw labels 0-28 confirmed; official 25-class scheme
  reproduced (raw 4 Orthoclase, 21 Biotite, 25 Olivine, 26 Intraclast
  folded into "Other"). `data/lithos.yaml` records the subset definition.

- **GT masks — the 24786 factor is SOLVED, and the paths are NOT a
  boundary** (verified on a 2,000-grain sample + the NeurIPS supplement):
  `path_1`/`path_2` are the grain's major/minor AXIS segments (single
  straight lines, exactly perpendicular, length ratio matches
  major/minor with median error 0.16%). 24786 is the width in pixels of
  the original thin-section mosaic; the 256x256 patch is a 1:1 crop of
  it centred on (x_inter, y_inter). **GT masks are therefore ELLIPSES**
  fitted through the four axis endpoints (see
  `scripts/build_gt_masks.py`) — an approximation, not a hand-drawn
  outline. 10-patch visual check (`assets/verify/*_overlay.png`):
  7 good fits, 2 fair (irregular/angular grains where the ellipse
  overshoots ~10-20%), 2 whole-patch (grain bigger than the patch --
  correct). Mask area: mean 45%, median 30% of patch.
  Consequence: mean IoU measures agreement with the elliptical
  approximation; irregular grains cap it below 1.0 even for a perfect
  segmenter. The README says this.

- **SAM2 batch eval DONE** (300 grains, 70.8 min): centre point
  (128,128), SAM2.1-hiera-tiny, CPU. Mean IoU **0.434 top-scored /
  0.609 oracle**; excluding the 93 whole-patch grains (31%):
  **0.287 / 0.445**. Only 17.9% of non-whole-patch grains reach IoU
  > 0.5 top-scored. Main finding: SAM often proposes a usable mask but
  ranks a speck first (see `assets/badcase_ranking_speck.png`).
  Per-class (excl. whole): best Calcite 0.74 (n=4), worst Foraminifer
  0.05 (n=2), Quartz 0.15. Timing: batch mean 14.1 s/image (shared CPU
  with probe training); unloaded ~9-11 s. Full table in
  `runs/summary.json` and README.

- **Probe DONE** (frozen ResNet18 + logistic regression, train Fold 1
  2,812 -> test Fold 2 1,000): **PPL-only acc 0.229 / macro-F1 0.228**;
  **PPL+XPL acc 0.319 / macro-F1 0.319**; Quartz-vs-rest acc 0.865 /
  0.872. XPL adds ~9 points. Best classes Dolomite (F1 0.68),
  Sanidine (0.60); worst Other (0.07), Fossil fragment (0.10).
  Quartz recall 0.325 (paper's trained ViT baseline ~0.49-0.55).
  Note: the first train_probe.py run failed late on sklearn 1.9
  removing the `multi_class` kwarg (fixed in the script); features
  were re-fit from `runs/probe/features.npz` via
  `scripts/fit_from_features.py`, which also saved `probe_model.pkl`
  (PPL+XPL) and `probe_model_ppl.pkl` for the app.

- **App VERIFIED** (Streamlit AppTest, headless): page loads, 1,000
  eval grains in the picker, clicking "分割 + 识别" runs SAM +
  classifier with no exceptions. One real bug caught by the test and
  fixed: `build_sam2` needs cwd inside sam2-main (Hydra rejects
  absolute config paths) — `load_sam()` now chdirs. Also replaced
  deprecated `use_container_width` with `width="stretch"`.

- **README.md complete** with all final numbers, the GT-mask honesty
  section, 3 bad cases + 1 good case (assets/*.png), confusion
  matrices (copied to assets/), SAM2-vs-SAM3 rationale, and
  reproduction steps. Repo scaffolding ready: requirements.txt,
  .gitignore (data/, runs/, weights ignored; index.csv + lithos.yaml
  kept), STATUS.md.

## Still to do
1. Create the public repo (suggested name `thin-section-mineral-id`,
   NOT decided by the user) via the browser, then push the text files
   (push-files) + images (web upload), same pattern as Demo 1. Do NOT
   commit data/, runs/, weights, or the LITHOS images in bulk.
2. Optional stretch: box-prompt SAM eval on ~100 grains; DINOv2-small
   probe variant; SAM3 text-prompt follow-up (needs GPU).
