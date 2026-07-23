#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把16次IDSW事件的ReID识别效果可视化成"胶片条"：
每行一个事件: [old锚点] [丢失期间6帧采样] [new重现] [decoy同帧另一辆车]
每张图下面标注和old锚点的余弦相似度，绿框=>=0.65(ReID认为是同一辆车)，红框=<0.65(认不准)，
decoy统一蓝框(负样本参照)。
"""
import sys
sys.path.insert(0, r"D:\deep-person-reid-master")
import os
import json
import numpy as np
import torch
import torchreid
import cv2

REID_MODEL_PATH = r"D:\deep-person-reid-master\scripts\log\osnet_ain_x1_0_market1501_softmax_cosinelr\model\model.pth.tar-100"
OLD_CROP_DIR = r"D:\yolo26-tracking\outputs\idsw_reid_check"
GAP_DIR = r"D:\yolo26-tracking\outputs\idsw_reid_check\gap_frames"
OUT_PATH = r"D:\yolo26-tracking\outputs\idsw_reid_check\reid_filmstrip.jpg"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESH = 0.65
CELL = 150

print(f"加载ReID模型 (device={DEVICE}) ...")
extractor = torchreid.utils.FeatureExtractor(
    model_name="osnet_ain_x1_0", model_path=REID_MODEL_PATH, device=DEVICE,
)


def cosine_sim(f1, f2):
    f1 = f1 / (np.linalg.norm(f1) + 1e-6)
    f2 = f2 / (np.linalg.norm(f2) + 1e-6)
    return float(np.dot(f1, f2))


def load_feat(path):
    if not os.path.exists(path):
        return None
    return extractor([path]).cpu().numpy()[0]


def make_tile(img_path, label, color, sim_text=None):
    tile = np.full((CELL, CELL, 3), 255, dtype=np.uint8)
    if img_path is not None and os.path.exists(img_path):
        img = cv2.imread(img_path)
        if img is not None:
            h, w = img.shape[:2]
            scale = min((CELL - 30) / w, (CELL - 45) / h)
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
            resized = cv2.resize(img, (nw, nh))
            y0 = 5 + ((CELL - 45) - nh) // 2
            x0 = 15 + ((CELL - 30) - nw) // 2
            tile[y0:y0 + nh, x0:x0 + nw] = resized
    cv2.rectangle(tile, (2, 2), (CELL - 3, CELL - 3), color, 3)
    cv2.putText(tile, label, (5, CELL - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    if sim_text is not None:
        cv2.putText(tile, sim_text, (5, CELL - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA)
    return tile


with open(os.path.join(GAP_DIR, "gap_manifest.json"), encoding="utf-8") as f:
    gap_manifest = {m["evt"]: m for m in json.load(f)}
with open(os.path.join(OLD_CROP_DIR, "manifest.json"), encoding="utf-8") as f:
    old_manifest = {m["evt"]: m for m in json.load(f)}

GREEN, RED, BLUE, GRAY = (0, 170, 0), (0, 0, 220), (200, 100, 0), (120, 120, 120)

rows = []
for evt in sorted(old_manifest.keys()):
    om = old_manifest[evt]
    gid = om["gt_id"]
    old_p = os.path.join(OLD_CROP_DIR, f"evt{evt:02d}_old_gt{gid}.jpg")
    new_p = os.path.join(OLD_CROP_DIR, f"evt{evt:02d}_new_gt{gid}.jpg")
    decoy_p = os.path.join(OLD_CROP_DIR, f"evt{evt:02d}_decoy_gt{gid}.jpg")

    f_old = load_feat(old_p)
    if f_old is None:
        continue

    tiles = [make_tile(old_p, f"evt{evt:02d} gt{gid} OLD", GRAY, "anchor")]

    gm = gap_manifest.get(evt)
    gap_samples = gm["samples"] if gm else []
    for s in gap_samples:
        f_g = load_feat(s["path"])
        sim = cosine_sim(f_old, f_g) if f_g is not None else None
        color = GREEN if (sim is not None and sim >= THRESH) else RED
        label = f"gap f{s['frame']} ({s['pos_in_gap']*100:.0f}%)"
        tiles.append(make_tile(s["path"], label, color, f"{sim:.3f}" if sim is not None else "N/A"))

    f_new = load_feat(new_p)
    sim_new = cosine_sim(f_old, f_new) if f_new is not None else None
    color_new = GREEN if (sim_new is not None and sim_new >= THRESH) else RED
    tiles.append(make_tile(new_p, "NEW(reappear)", color_new, f"{sim_new:.3f}" if sim_new is not None else "N/A"))

    f_decoy = load_feat(decoy_p) if os.path.exists(decoy_p) else None
    sim_decoy = cosine_sim(f_old, f_decoy) if f_decoy is not None else None
    tiles.append(make_tile(decoy_p, "DECOY(别的车)", BLUE, f"{sim_decoy:.3f}" if sim_decoy is not None else "N/A"))

    row = np.concatenate(tiles, axis=1)
    rows.append(row)
    print(f"evt{evt:02d} gt#{gid}: 完成一行 ({len(tiles)}格)")

max_w = max(r.shape[1] for r in rows)
padded_rows = []
for r in rows:
    if r.shape[1] < max_w:
        pad = np.full((r.shape[0], max_w - r.shape[1], 3), 255, dtype=np.uint8)
        r = np.concatenate([r, pad], axis=1)
    padded_rows.append(r)

grid = np.concatenate(padded_rows, axis=0)
cv2.imwrite(OUT_PATH, grid)
print(f"\n已保存: {OUT_PATH}  (每行: OLD锚点 -> 丢失期间6帧采样 -> NEW重现 -> DECOY负样本；绿框>=阈值{THRESH}，红框<阈值)")
