#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对丢失区间内(检测器认不出来的那些帧)的GT裁剪，逐帧计算与"丢失前最后一帧(old锚点)"
的ReID相似度，看这些帧车辆外观本身是否还能被ReID正确识别为同一辆车，
并与之前算出的negative(decoy)相似度分布对比，判断能不能用一个统一阈值把它们分开。
"""
import sys
sys.path.insert(0, r"D:\deep-person-reid-master")
import os
import json
import numpy as np
import torch
import torchreid

REID_MODEL_PATH = r"D:\deep-person-reid-master\scripts\log\osnet_ain_x1_0_market1501_softmax_cosinelr\model\model.pth.tar-100"
OLD_CROP_DIR = r"D:\yolo26-tracking\outputs\idsw_reid_check"
GAP_DIR = r"D:\yolo26-tracking\outputs\idsw_reid_check\gap_frames"
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
    return extractor([path]).cpu().numpy()[0]


with open(os.path.join(GAP_DIR, "gap_manifest.json"), encoding="utf-8") as f:
    manifest = json.load(f)
with open(os.path.join(OLD_CROP_DIR, "manifest.json"), encoding="utf-8") as f:
    old_manifest = {m["evt"]: m for m in json.load(f)}

# 之前算出的负样本(old vs decoy)相似度分布，作为对照基准
NEG_MEAN, NEG_MAX = 0.468, 0.587
POS_MIN = 0.650  # 之前 old-vs-new 正样本的最小值

all_gap_sims = []
low_conf_worrying = []

for m in manifest:
    evt, gid = m["evt"], m["gt_id"]
    old_p = os.path.join(OLD_CROP_DIR, f"evt{evt:02d}_old_gt{gid}.jpg")
    f_old = load_feat(old_p)
    if f_old is None:
        print(f"evt{evt:02d}: old锚点缺失，跳过")
        continue

    print(f"\nevt{evt:02d} gt#{gid}  丢失区间[{m['lost_start']},{m['lost_end']}] (共{m['gap']}帧)")
    for s in m["samples"]:
        f_gap = load_feat(s["path"])
        if f_gap is None:
            continue
        sim = cosine_sim(f_old, f_gap)
        all_gap_sims.append(sim)
        flag = ""
        if sim < POS_MIN:
            flag = "  <-- 低于正样本下限，ReID也认不准了"
            low_conf_worrying.append((evt, gid, s["frame"], sim))
        print(f"    frame={s['frame']:>5}  丢失进度={s['pos_in_gap']*100:5.1f}%  sim(old,该帧)={sim:.3f}{flag}")

all_gap_sims = np.array(all_gap_sims)
print("\n=== 汇总 ===")
print(f"丢失期间所有采样帧 sim(old, 该帧): mean={all_gap_sims.mean():.3f}  std={all_gap_sims.std():.3f}  "
      f"min={all_gap_sims.min():.3f}  max={all_gap_sims.max():.3f}")
print(f"对照：之前 old-vs-new(正样本) min={POS_MIN:.3f}；old-vs-decoy(负样本) mean={NEG_MEAN:.3f} max={NEG_MAX:.3f}")
print(f"\n丢失期间相似度低于正样本下限({POS_MIN})的采样帧数: {len(low_conf_worrying)}/{len(all_gap_sims)}")
for evt, gid, frame, sim in low_conf_worrying:
    print(f"    evt{evt:02d} gt#{gid} frame={frame} sim={sim:.3f}")
