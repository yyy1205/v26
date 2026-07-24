# -*- coding: utf-8 -*-
"""
统计当前24次IDSW事件中，"new_pred_id"是否属于"长时间丢失后被复活"的僵尸track，
以及其复活前最后一次出现的位置是否在画面边缘（大概率已驶出画面）。
"""
import numpy as np
from collections import defaultdict

PRED_TXT = r"D:\yolo26-tracking\outputs\yolo26-obb_20260723_213702\cvat_eval.txt"

EVENTS = [
    (146, 157, 128, 31),
    (163, 138, 137, 158),
    (212, 169, 62, 167),
    (301, 182, 36, 174),
    (466, 136, 135, 22),
    (480, 136, 22, 189),
    (547, 136, 189, 135),
    (563, 136, 135, 189),
    (570, 150, 45, 113),
    (576, 150, 113, 197),
    (627, 223, 80, 201),
    (660, 227, 40, 23),
    (673, 150, 197, 113),
    (965, 268, 75, 16),
    (967, 268, 16, 221),
    (1120, 218, 110, 233),
    (1141, 218, 233, 110),
    (1444, 275, 226, 258),
    (1521, 337, 12, 261),
    (1535, 275, 258, 226),
    (1648, 293, 235, 97),
    (1651, 293, 97, 235),
    (1771, 365, 95, 279),
    (1784, 313, 67, 281),
]

# 解析 pred 轨迹：track_id -> sorted [(frame, cx, cy)]
tracks = defaultdict(list)
all_x, all_y = [], []
with open(PRED_TXT, "r") as f:
    for line in f:
        parts = line.strip().split(",")
        if len(parts) < 10:
            continue
        frame_id = int(parts[0])
        track_id = int(parts[1])
        xs = [float(parts[2]), float(parts[4]), float(parts[6]), float(parts[8])]
        ys = [float(parts[3]), float(parts[5]), float(parts[7]), float(parts[9])]
        cx, cy = sum(xs) / 4, sum(ys) / 4
        tracks[track_id].append((frame_id, cx, cy))
        all_x.append(cx); all_y.append(cy)

for tid in tracks:
    tracks[tid].sort(key=lambda r: r[0])

# 用所有点的范围估计画面边界（比读取视频文件简单可靠）
xmin, xmax = min(all_x), max(all_x)
ymin, ymax = min(all_y), max(all_y)
print(f"坐标范围估计画面边界: x[{xmin:.0f},{xmax:.0f}]  y[{ymin:.0f},{ymax:.0f}]")

EDGE_MARGIN = 100  # px，距边界多近算"边缘"


def is_edge(cx, cy):
    return (cx - xmin < EDGE_MARGIN or xmax - cx < EDGE_MARGIN or
            cy - ymin < EDGE_MARGIN or ymax - cy < EDGE_MARGIN)


print(f"\n{'GT帧':>6} {'GTid':>5} {'old':>5} {'new':>5} | {'gap(帧)':>8} {'复活前位置':>18} {'是否边缘':>8}")
print("-" * 70)

n_recovery = 0
n_edge_recovery = 0
n_continuous = 0

for gt_frame, gt_id, old_pid, new_pid in EVENTS:
    pred_frame = gt_frame + 1
    hist = tracks.get(new_pid, [])
    # 找到 pred_frame 之前 new_pid 最近一次出现的帧
    prev = [r for r in hist if r[0] < pred_frame]
    if not prev:
        print(f"{gt_frame:>6} {gt_id:>5} {old_pid:>5} {new_pid:>5} | new_pid 无更早历史（首次出现即匹配上）")
        continue
    last_frame, lcx, lcy = prev[-1]
    gap = pred_frame - last_frame
    edge = is_edge(lcx, lcy)
    tag = "continuous" if gap <= 1 else "RECOVERY"
    if gap <= 1:
        n_continuous += 1
    else:
        n_recovery += 1
        if edge:
            n_edge_recovery += 1
    print(f"{gt_frame:>6} {gt_id:>5} {old_pid:>5} {new_pid:>5} | {gap:>8} ({lcx:>6.0f},{lcy:>6.0f})   {'边缘' if edge else '否':>6}  [{tag}]")

print("-" * 70)
print(f"总事件数: {len(EVENTS)}")
print(f"new_pred_id 与前一帧连续(gap<=1，非复活): {n_continuous}")
print(f"new_pred_id 存在丢失间隔后复活(gap>1):     {n_recovery}")
print(f"  其中复活前最后位置在画面边缘的:           {n_edge_recovery}")
