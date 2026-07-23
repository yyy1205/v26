#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""核对：走ultralytics现成的ReID/AutoBackend路径(ONNX)算出来的相似度，
和之前torchreid直接算出来的相似度是否一致，确认导出/预处理没出问题。
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
import numpy as np
import cv2
import torch
from ultralytics.trackers.utils.reid import ReID

ONNX_PATH = r"D:\yolo26-tracking\reid_model\osnet_ain_x1_0_vehicle.onnx"
CROP_DIR = r"D:\yolo26-tracking\outputs\idsw_reid_check"

reid = ReID(ONNX_PATH, device="cpu")
print(f"imgsz={reid.imgsz}  batch_size={reid.batch_size}")


def cosine_sim(f1, f2):
    f1 = f1 / (np.linalg.norm(f1) + 1e-6)
    f2 = f2 / (np.linalg.norm(f2) + 1e-6)
    return float(np.dot(f1, f2))


def load_feat(path):
    img = cv2.imread(path)
    if img is None:
        return None
    h, w = img.shape[:2]
    # ReID.__call__ 需要 dets(xywh) 去裁剪，这里图已经是裁好的crop，直接构造一个覆盖全图的框
    dets = np.array([[w / 2, h / 2, w, h]], dtype=np.float32)
    feats = reid(img, dets)
    return feats[0]


with open(os.path.join(CROP_DIR, "manifest.json"), encoding="utf-8") as f:
    manifest = json.load(f)

print(f"\n{'evt':>4} {'gt_id':>6} {'sim(old,new) ONNX':>18} {'sim(old,decoy) ONNX':>20}")
print("-" * 55)
onnx_pos, onnx_neg = [], []
for m in manifest:
    evt, gid = m["evt"], m["gt_id"]
    old_p = os.path.join(CROP_DIR, f"evt{evt:02d}_old_gt{gid}.jpg")
    new_p = os.path.join(CROP_DIR, f"evt{evt:02d}_new_gt{gid}.jpg")
    decoy_p = os.path.join(CROP_DIR, f"evt{evt:02d}_decoy_gt{gid}.jpg")

    f_old, f_new, f_decoy = load_feat(old_p), load_feat(new_p), load_feat(decoy_p)
    if f_old is None or f_new is None:
        continue
    sim_pos = cosine_sim(f_old, f_new)
    onnx_pos.append(sim_pos)
    sim_neg = cosine_sim(f_old, f_decoy) if f_decoy is not None else None
    if sim_neg is not None:
        onnx_neg.append(sim_neg)
    print(f"{evt:>4} {gid:>6} {sim_pos:>18.3f} {(sim_neg if sim_neg is not None else float('nan')):>20.3f}")

onnx_pos, onnx_neg = np.array(onnx_pos), np.array(onnx_neg)
print("\n=== ONNX路径 汇总 ===")
print(f"正样本: mean={onnx_pos.mean():.3f} min={onnx_pos.min():.3f} max={onnx_pos.max():.3f}")
print(f"负样本: mean={onnx_neg.mean():.3f} min={onnx_neg.min():.3f} max={onnx_neg.max():.3f}")
print(f"正负样本是否重叠: {'是' if onnx_pos.min() <= onnx_neg.max() else '否(有清晰分界)'}")
print("\n对照(torchreid直接跑，之前的结果): 正样本 mean=0.851 min=0.700 max=0.970；负样本 mean=0.454 min=0.351 max=0.571")
