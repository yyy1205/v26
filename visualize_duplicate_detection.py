#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化验证：GT id=146 在帧1179-1196附近，track_id=154 与 394 是否是
同一辆车被同时判成 car / truck 的两个几乎重叠的重复框。

从视频里抠出这几帧对应车辆附近的区域，把两个预测框画上去（不同颜色），
标注 track_id / label / score，拼成一张对比图。
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import numpy as np
import pandas as pd

VIDEO_PATH = r"E:\DJI\cvat\0111_D_stabilized.avi"
TRACK_TXT = r"D:\yolo26-tracking\outputs\yolo26-obb_20260721_105049\track_result.txt"
OUT_PATH = r"D:\yolo26-tracking\outputs\yolo26-obb_20260721_105049\idsw_diagnosis\dup_detection_check.jpg"

FRAMES = [1179, 1183, 1188, 1191, 1194, 1196]
TRACK_A, TRACK_B = 154, 394
CROP_HALF = 110  # 裁剪区域半径（像素）

df = pd.read_csv(TRACK_TXT, header=None,
                  names=["frame_id", "track_id", "xc", "yc", "w", "h", "angle_rad", "score", "label", "v_kmh"])

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"无法打开视频: {VIDEO_PATH}")

tiles = []
for f in FRAMES:
    cap.set(cv2.CAP_PROP_POS_FRAMES, f - 1)  # frame_id 从1开始，对应视频第0帧起的第 f-1 帧
    ret, frame = cap.read()
    if not ret:
        print(f"警告: 读取帧 {f} 失败，跳过")
        continue

    row_a = df[(df.frame_id == f) & (df.track_id == TRACK_A)]
    row_b = df[(df.frame_id == f) & (df.track_id == TRACK_B)]

    # 用两个框中心的均值作为裁剪中心
    centers = []
    for row in (row_a, row_b):
        if len(row):
            centers.append((float(row.iloc[0].xc), float(row.iloc[0].yc)))
    if not centers:
        print(f"警告: 帧 {f} 两个 track 都不存在，跳过")
        continue
    ccx = float(np.mean([c[0] for c in centers]))
    ccy = float(np.mean([c[1] for c in centers]))

    x1 = max(0, int(ccx - CROP_HALF))
    y1 = max(0, int(ccy - CROP_HALF))
    x2 = min(frame.shape[1], int(ccx + CROP_HALF))
    y2 = min(frame.shape[0], int(ccy + CROP_HALF))

    vis = frame.copy()
    specs = (
        (row_a, (0, 255, 255), f"{TRACK_A}", -1),  # 青色，标签放上方
        (row_b, (255, 0, 255), f"{TRACK_B}", 1),   # 品红，标签放下方
    )
    for row, color, name, text_dir in specs:
        if not len(row):
            continue
        r = row.iloc[0]
        rect = ((float(r.xc), float(r.yc)), (float(r.w), float(r.h)), float(np.rad2deg(r.angle_rad)))
        pts = np.intp(cv2.boxPoints(rect))
        thickness = 4 if text_dir < 0 else 2
        cv2.polylines(vis, [pts], True, color, thickness)
        txt = f"id{name}:{r.label} {r.score:.2f}"
        ty = int(r.yc + text_dir * (float(r.h) / 2 + 25))
        cv2.putText(vis, txt, (int(r.xc - 70), ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
        cv2.putText(vis, txt, (int(r.xc - 70), ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1, cv2.LINE_AA)

    crop = vis[y1:y2, x1:x2]
    crop = cv2.resize(crop, (440, 440))
    cv2.rectangle(crop, (0, 0), (439, 439), (255, 255, 255), 2)
    cv2.putText(crop, f"frame {f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
    tiles.append(crop)

cap.release()

if not tiles:
    raise RuntimeError("没有成功生成任何一帧的对比图")

# 拼成 2 行 x 3 列（或按数量自适应）
cols = 3
rows = (len(tiles) + cols - 1) // cols
grid = np.zeros((rows * 440, cols * 440, 3), dtype=np.uint8)
for i, tile in enumerate(tiles):
    r, c = divmod(i, cols)
    grid[r * 440:(r + 1) * 440, c * 440:(c + 1) * 440] = tile

cv2.imwrite(OUT_PATH, grid)
print(f"已保存对比图: {OUT_PATH}  (共 {len(tiles)} 帧, 黄框=track{TRACK_A}, 品红框=track{TRACK_B})")
