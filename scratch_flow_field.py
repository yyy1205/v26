#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证"局部平均流向场"思路：把瓶颈弯道区域切网格，每个格子用经过它的干净轨迹的
逐帧位移方向做平均，画成箭头场，看是否连贯地贴合弯道走向、且不同车道之间不会互相打架。
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from collections import defaultdict
import cv2
import numpy as np

TRACK_RESULT = r"D:\yolo26-tracking\outputs\yolo26-obb_20260722_155902\track_result.txt"
BG = r"D:\yolo26-tracking\outputs\yolo26-obb_20260722_155902\roi_selected.jpg"
RX0, RX1, RY0, RY1 = 0, 1400, 0, 400
CELL = 60

by_track = defaultdict(list)
with open(TRACK_RESULT, encoding="utf-8") as f:
    for line in f:
        p = line.strip().split(",")
        if len(p) < 6:
            continue
        frame, tid, cx, cy = int(p[0]), int(p[1]), float(p[2]), float(p[3])
        by_track[tid].append((frame, cx, cy))
for tid in by_track:
    by_track[tid].sort(key=lambda r: r[0])

# 每个格子累积 (sum_vx, sum_vy, count)，只用相邻帧位移当方向向量(因果式，逐帧可在线累加)
cell_sum = defaultdict(lambda: [0.0, 0.0, 0])
for tid, recs in by_track.items():
    for (f1, x1, y1), (f2, x2, y2) in zip(recs[:-1], recs[1:]):
        if f2 - f1 != 1:
            continue  # 跳帧(丢失期间)的不算，只用连续两帧的真实位移
        if not (RX0 <= x1 <= RX1 and RY0 <= y1 <= RY1):
            continue
        vx, vy = x2 - x1, y2 - y1
        speed = np.hypot(vx, vy)
        if speed < 0.5:  # 几乎静止的不参与(噪声/红绿灯停车)
            continue
        key = (int(x1 // CELL), int(y1 // CELL))
        s = cell_sum[key]
        s[0] += vx; s[1] += vy; s[2] += 1

bg = cv2.imread(BG)
canvas = bg.copy()
n_cells = 0
for (gx, gy), (svx, svy, cnt) in cell_sum.items():
    if cnt < 3:
        continue
    n_cells += 1
    cx, cy = gx * CELL + CELL // 2, gy * CELL + CELL // 2
    avx, avy = svx / cnt, svy / cnt
    norm = np.hypot(avx, avy)
    if norm < 1e-3:
        continue
    scale = 25.0
    ex, ey = int(cx + avx / norm * scale), int(cy + avy / norm * scale)
    cv2.arrowedLine(canvas, (cx, cy), (ex, ey), (0, 0, 255), 3, cv2.LINE_AA, tipLength=0.4)

print(f"有效格子数(样本>=3): {n_cells}")
out_path = r"D:\yolo26-tracking\outputs\yolo26-obb_20260722_155902\flow_field_check.jpg"
cv2.imwrite(out_path, canvas)
print(f"已保存: {out_path}")
