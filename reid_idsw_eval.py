#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用OSNet ReID模型提取16次IDSW事件的old/new/decoy裁剪特征，验证:
  1. sim(old, new) 正样本相似度是否明显高
  2. sim(old, decoy) 负样本相似度是否明显低
  3. 正负样本之间有没有可用的分界阈值
"""
import sys
sys.path.insert(0, r"D:\deep-person-reid-master")
import os
import json
import numpy as np
import torch
import torchreid
from PIL import Image

REID_MODEL_PATH = r"D:\deep-person-reid-master\scripts\log\osnet_ain_x1_0_market1501_softmax_cosinelr\model\model.pth.tar-100"
CROP_DIR = r"D:\yolo26-tracking\outputs\idsw_reid_check"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"加载ReID模型 (device={DEVICE}) ...")
extractor = torchreid.utils.FeatureExtractor(
    model_name="osnet_ain_x1_0",
    model_path=REID_MODEL_PATH,
    device=DEVICE,
)


def cosine_sim(f1, f2):
    f1 = f1 / (np.linalg.norm(f1) + 1e-6)
    f2 = f2 / (np.linalg.norm(f2) + 1e-6)
    return float(np.dot(f1, f2))


def load_feat(path):
    if not os.path.exists(path):
        return None
    feat = extractor([path]).cpu().numpy()[0]
    return feat


with open(os.path.join(CROP_DIR, "manifest.json"), encoding="utf-8") as f:
    manifest = json.load(f)

pos_sims, neg_sims = [], []
print(f"\n{'evt':>4} {'gt_id':>6} {'sim(old,new)正样本':>20} {'sim(old,decoy)负样本':>22} {'区分度':>8}")
print("-" * 70)
for m in manifest:
    evt, gid = m["evt"], m["gt_id"]
    old_p = os.path.join(CROP_DIR, f"evt{evt:02d}_old_gt{gid}.jpg")
    new_p = os.path.join(CROP_DIR, f"evt{evt:02d}_new_gt{gid}.jpg")
    decoy_p = os.path.join(CROP_DIR, f"evt{evt:02d}_decoy_gt{gid}.jpg")

    f_old = load_feat(old_p)
    f_new = load_feat(new_p)
    f_decoy = load_feat(decoy_p)
    if f_old is None or f_new is None:
        print(f"{evt:>4} {gid:>6}  数据缺失，跳过")
        continue

    sim_pos = cosine_sim(f_old, f_new)
    pos_sims.append(sim_pos)
    sim_neg = None
    if f_decoy is not None:
        sim_neg = cosine_sim(f_old, f_decoy)
        neg_sims.append(sim_neg)

    gap_str = f"{sim_pos - sim_neg:+.3f}" if sim_neg is not None else "N/A"
    print(f"{evt:>4} {gid:>6} {sim_pos:>20.3f} {(sim_neg if sim_neg is not None else float('nan')):>22.3f} {gap_str:>8}")

print("\n=== 汇总 ===")
pos_sims = np.array(pos_sims)
neg_sims = np.array(neg_sims)
print(f"正样本 sim(old,new)  : mean={pos_sims.mean():.3f}  std={pos_sims.std():.3f}  min={pos_sims.min():.3f}  max={pos_sims.max():.3f}")
if len(neg_sims):
    print(f"负样本 sim(old,decoy): mean={neg_sims.mean():.3f}  std={neg_sims.std():.3f}  min={neg_sims.min():.3f}  max={neg_sims.max():.3f}")
    overlap = (pos_sims.min() <= neg_sims.max())
    print(f"正负样本相似度区间是否重叠: {'是(区分度差)' if overlap else '否(有清晰分界)'}")
    print(f"  正样本最小值={pos_sims.min():.3f}  负样本最大值={neg_sims.max():.3f}")
