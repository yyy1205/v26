# -*- coding: utf-8 -*-
"""
检查所有track_id自身内部的长间隔(gap>=30帧)恢复(即同一个id消失又重新出现，说明大概率是第三轮ReID正确
恢复了同一辆车，没有产生IDSW，属于"合法恢复")，统计这些合法恢复消失前的位置是否在画面边缘。
如果合法恢复也大量集中在边缘，那么直接"排除边缘候选"这个方案会有副作用，需要更精细的判据。
"""
from collections import defaultdict

PRED_TXT = r"D:\yolo26-tracking\outputs\yolo26-obb_20260723_213702\cvat_eval.txt"

tracks = defaultdict(list)
with open(PRED_TXT, "r") as f:
    for line in f:
        p = line.strip().split(",")
        if len(p) < 10:
            continue
        frame_id = int(p[0]); tid = int(p[1])
        xs = [float(p[2]), float(p[4]), float(p[6]), float(p[8])]
        ys = [float(p[3]), float(p[5]), float(p[7]), float(p[9])]
        tracks[tid].append((frame_id, sum(xs) / 4, sum(ys) / 4))
for t in tracks:
    tracks[t].sort()

EDGE_XMIN, EDGE_XMAX = 0, 3534
EDGE_YMIN, EDGE_YMAX = 0, 1988
EDGE_MARGIN = 100


def is_edge(x, y):
    return (x - EDGE_XMIN < EDGE_MARGIN or EDGE_XMAX - x < EDGE_MARGIN or
            y - EDGE_YMIN < EDGE_MARGIN or EDGE_YMAX - y < EDGE_MARGIN)


internal_recoveries = []  # (track_id, gap, pre_gap_frame, pre_x, pre_y, edge)
for tid, hist in tracks.items():
    for i in range(1, len(hist)):
        prev_frame, px, py = hist[i - 1]
        cur_frame, cx, cy = hist[i]
        gap = cur_frame - prev_frame
        if gap >= 30:
            internal_recoveries.append((tid, gap, prev_frame, px, py, is_edge(px, py)))

internal_recoveries.sort(key=lambda r: -r[1])
print(f"track内部长间隔(>=30帧)恢复总数: {len(internal_recoveries)}")
n_edge = sum(1 for r in internal_recoveries if r[5])
n_nonedge = len(internal_recoveries) - n_edge
print(f"  消失前位置在边缘: {n_edge}")
print(f"  消失前位置不在边缘: {n_nonedge}")
print()
print(f"{'track_id':>8} {'gap':>5} {'消失帧':>8} {'消失前位置':>18} {'边缘':>5}")
for tid, gap, pf, px, py, edge in internal_recoveries:
    print(f"{tid:>8} {gap:>5} {pf:>8} ({px:>7.1f},{py:>7.1f})   {'是' if edge else '否'}")
