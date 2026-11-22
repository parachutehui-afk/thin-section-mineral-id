"""Streamlit demo: SAM2 segmentation + mineral identification (LITHOS).

Pick a grain-centred patch from the eval subset (or upload your own thin
section crop), SAM2 cuts out the central grain with a centre-point prompt
(CPU, ~11 s), and a frozen-ResNet18 linear probe predicts the mineral.

This is a public-dataset reproduction (LITHOS, CC BY-NC-SA 4.0) of the
kind of work in the author's portfolio -- NOT private project data.
See README.md for the honest numbers and the limits of the GT masks
(ellipses fitted from the annotated grain axes, not hand-drawn outlines).
"""
import csv
import pickle
import time
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18

ROOT = Path(__file__).resolve().parent
INDEX = ROOT / "data" / "index.csv"
SUBSET = ROOT / "data" / "subset" / "eval"
MODEL_PKL = ROOT / "runs" / "probe" / "probe_model.pkl"
MODEL_PKL_PPL = ROOT / "runs" / "probe" / "probe_model_ppl.pkl"
CKPT = ROOT / "smoke-test" / "checkpoints" / "sam2.1_hiera_tiny.pt"
SAM2_CFG = None  # build_sam2 needs cwd inside sam2-main; see load_sam()

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@st.cache_resource
def load_sam():
    # build_sam2 resolves its config through Hydra relative to the sam2
    # source tree, so it must run with cwd inside sam2-main (an absolute
    # config path does NOT work -- Hydra strips the leading slash).
    import os
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    torch.set_num_threads(4)
    sam2_dir = ROOT / "smoke-test" / "sam2-main"
    prev = os.getcwd()
    try:
        os.chdir(sam2_dir)
        model = build_sam2("configs/sam2.1/sam2.1_hiera_t.yaml",
                           str(CKPT), device="cpu")
    finally:
        os.chdir(prev)
    return SAM2ImagePredictor(model)


@st.cache_resource
def load_backbone():
    backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model = nn.Sequential(*list(backbone.children())[:-1],
                          nn.Flatten()).eval()
    return model


@torch.no_grad()
def extract_features(img_pil, backbone):
    img = img_pil.convert("RGB").resize((224, 224), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - MEAN) / STD
    t = torch.from_numpy(x.transpose(2, 0, 1))[None]
    return backbone(t).numpy()[0]


def load_pickle(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def predict_mineral(feat, model_bundle):
    scaler, clf = model_bundle["scaler"], model_bundle["clf"]
    proba = clf.predict_proba(scaler.transform(feat[None]))[0]
    order = np.argsort(proba)[::-1]
    return [(model_bundle["classes"][i], float(proba[i])) for i in order]


def overlay_mask(img_rgb, mask, color=(255, 0, 0), alpha=0.45):
    out = img_rgb.astype(np.float32).copy()
    for c in range(3):
        out[:, :, c] = np.where(mask, out[:, :, c] * (1 - alpha)
                                + color[c] * alpha, out[:, :, c])
    return out.astype(np.uint8)


st.set_page_config(page_title="薄片矿物识别 · SAM2 + LITHOS", layout="centered")
st.title("岩石薄片矿物识别 · SAM2 + 线性探针")
st.caption("SAM2 切出中心颗粒（CPU），冻结 ResNet18 特征 + 逻辑回归认矿物。"
           "公开数据集 LITHOS 复现，非作者私有项目数据。")

if not MODEL_PKL.exists():
    st.error("未找到探针模型 runs/probe/probe_model.pkl，"
             "请先运行 scripts/prepare_data.py 与 scripts/train_probe.py。")
    st.stop()
if not CKPT.exists():
    st.error("未找到 SAM2 权重 smoke-test/checkpoints/sam2.1_hiera_tiny.pt。")
    st.stop()

mode = st.radio("图片来源", ["数据集示例 patch（Fold 2）", "上传自己的薄片图"],
                horizontal=True)
pol = st.radio("偏振光", ["PPL（平面偏振光）", "XPL（正交偏振光）"],
               horizontal=True)
pol_key = "ppl" if pol.startswith("PPL") else "xpl"

img_ppl = img_xpl = None
gt_label = None
if mode.startswith("数据集"):
    rows = [r for r in csv.DictReader(open(INDEX)) if r["split"] == "eval"]
    gid = st.selectbox("选择一颗颗粒（Fold 2 评估集，共 1,000 颗）",
                       [r["grain_id"] for r in rows])
    row = next(r for r in rows if r["grain_id"] == gid)
    gt_label = row["label25_name"]
    img_ppl = Image.open(SUBSET / gid / "ppl-0.png").convert("RGB")
    img_xpl = Image.open(SUBSET / gid / "xpl-0.png").convert("RGB")
else:
    up = st.file_uploader("选择图片（jpg / png，最好是已裁到单颗颗粒居中的薄片图）",
                          type=["jpg", "jpeg", "png"])
    if up is not None:
        img_ppl = Image.open(up).convert("RGB")
        st.info("上传图按 PPL 特征识别；SAM 只做中心点分割，不参与分类打分。")

img_show = img_ppl if pol_key == "ppl" else (img_xpl or img_ppl)
if img_show is not None:
    st.image(img_show, caption=f"{pol} · {img_show.size[0]}×{img_show.size[1]}",
             width="stretch")

if st.button("分割 + 识别", disabled=img_ppl is None):
    backbone = load_backbone()
    t0 = time.perf_counter()
    # --- SAM2 centre-point segmentation (on the displayed polarisation) ---
    pred = load_sam()
    arr = np.array(img_show)
    pred.set_image(arr)
    masks, scores, _ = pred.predict(point_coords=np.array([[arr.shape[1] // 2,
                                                            arr.shape[0] // 2]]),
                                    point_labels=np.array([1]))
    best = int(np.argmax(scores))
    sam_mask = masks[best].astype(bool)
    sam_s = time.perf_counter() - t0
    ov = overlay_mask(arr, sam_mask)
    st.image(ov, caption=f"SAM2 中心点分割 · mask {sam_mask.mean() * 100:.0f}% "
             f"of patch · SAM 置信度 {scores[best]:.2f} · {sam_s:.1f} s（CPU）",
             width="stretch")
    # --- mineral classification (whole-patch features, like the paper) ---
    t1 = time.perf_counter()
    if mode.startswith("数据集"):
        feat = np.concatenate([extract_features(img_ppl, backbone),
                               extract_features(img_xpl, backbone)])
        bundle = load_pickle(MODEL_PKL)
    else:
        feat = extract_features(img_ppl, backbone)
        bundle = load_pickle(MODEL_PKL_PPL) if MODEL_PKL_PPL.exists() \
            else load_pickle(MODEL_PKL)
        if not MODEL_PKL_PPL.exists():
            feat = np.concatenate([feat, feat])  # fallback: PPL twice
    top = predict_mineral(feat, bundle)
    clf_s = time.perf_counter() - t1
    st.subheader(f"预测：{top[0][0]}（置信度 {top[0][1]:.2f}）")
    st.table([{"矿物": n, "概率": f"{p:.3f}"} for n, p in top[:3]])
    if gt_label:
        mark = "✅ 一致" if top[0][0] == gt_label else "❌ 与标注不同"
        st.info(f"数据集标注（仅供对照）：{gt_label} · {mark}")
    st.caption(f"分类耗时 {clf_s:.1f} s · SAM {sam_s:.1f} s · "
               f"模型：SAM2.1-hiera-tiny + ResNet18 冻结特征逻辑回归"
               f"（{bundle.get('variant', 'ppl+xpl')}）")
    st.caption("提醒：SAM2 是零样本分割，薄片上常切不准；"
               "分类用的是整张 patch 特征，不依赖 SAM mask。")
