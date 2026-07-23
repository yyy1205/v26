#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用该区域内所有干净轨迹点的并集(不是单条track)，通过网格分箱+最近邻链式连接，
拟合出一条参考路径骨架，对比"只用截止到frame163的数据"vs"全量1824帧数据"两种情况下拟合出的曲线形状。
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from collections import defaultdict
import cv2
import numpy as np

TRACK_RESULT = r"D:\yolo26-tracking\outputs\yolo26-obb_20260722_155902\track_result.txt"
BG = r"D:\yolo26-tracking\outputs\yolo26-obb_20260722_155902\roi_selected.jpg"
RX0, RX1, RY0, RY1 = 0, 1400, 0, 400
CELL = 40

by_track = defaultdict(list)
with open(TRACK_RESULT, encoding="utf-8") as f:
    for line in f:
        p = line.strip().split(",")
        if len(p) < 6:
            continue
        frame, tid, cx, cy = int(p[0]), int(p[1]), float(p[2]), float(p[3])
        if RX0 <= cx <= RX1 and RY0 <= cy <= RY1:
            by_track[tid].append((frame, cx, cy))
clean = {tid: pts for tid, pts in by_track.items() if len(pts) >= 5}


def fit_skeleton(max_frame):
    pts = []
    for tid, recs in clean.items():
        pts.extend([(cx, cy) for f, cx, cy in recs if f <= max_frame])
    if len(pts) < 20:
        return None, 0
    pts = np.array(pts, dtype=np.float32)

    # 网格分箱求质心
    cells = defaultdict(list)
    for x, y in pts:
        cells[(int(x // CELL), int(y // CELL))].append((x, y))
    centroids = np.array([np.mean(v, axis=0) for v in cells.values()], dtype=np.float32)

    # 从"最靠上"(y最小，也就是离摄像头最远、弯道最上端)的点开始，最近邻链式连接
    remaining = list(range(len(centroids)))
    start = min(remaining, key=lambda i: centroids[i][1])
    chain = [start]
    remaining.remove(start)
    while remaining:
        last = centroids[chain[-1]]
        dists = [np.hypot(centroids[i][0] - last[0], centroids[i][1] - last[1]) for i in remaining]
        j = int(np.argmin(dists))
        if dists[j] > CELL * 4:  # 断得太远就停止(避免跳到不相关的分支)
            break
        chain.append(remaining[j])
        remaining.pop(j)

    path = centroids[chain]
    return path, len(pts)


bg = cv2.imread(BG)
canvas = bg.copy()

for max_frame, color, label in [(163, (0, 0, 255), "RED: fit using data up to frame 163"),
                                  (1824, (0, 220, 0), "GREEN: fit using all 1824 frames")]:
    path, n_pts = fit_skeleton(max_frame)
    if path is None:
        print(f"frame<={max_frame}: 数据不足")
        continue
    print(f"frame<={max_frame}: 用了{n_pts}个点，骨架节点数={len(path)}, "
          f"x范围=({path[:,0].min():.0f},{path[:,0].max():.0f}) y范围=({path[:,1].min():.0f},{path[:,1].max():.0f})")
    poly = path.astype(np.int32)
    cv2.polylines(canvas, [poly], False, color, 4, cv2.LINE_AA)
    for p in poly:
        cv2.circle(canvas, tuple(p), 4, color, -1)

cv2.putText(canvas, "RED = skeleton fit using only data up to frame 163", (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3, cv2.LINE_AA)
cv2.putText(canvas, "GREEN = skeleton fit using all 1824 frames", (30, 110),
            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 220, 0), 3, cv2.LINE_AA)
out_path = r"D:\yolo26-tracking\outputs\yolo26-obb_20260722_155902\road_path_coldstart_compare.jpg"
cv2.imwrite(out_path, canvas)
print(f"已保存: {out_path}")
