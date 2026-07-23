#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证担忧：就算不让建新id，重新匹配时会不会同时有好几条Lost track都在附近，导致选错？
对每个IDSW事件，在切换帧统计：还有哪些track_id最近(<=120帧, 即track_buffer)丢失过，
且最后位置离这次重新出现的位置较近(<=400px)——这些就是"候选竞争者"。
"""
from collections import defaultdict

TRACK_RESULT = r"D:\yolo26-tracking\outputs\yolo26-obb_20260722_155902\track_result.txt"
TRACK_BUFFER = 120
SEARCH_RADIUS = 400

IDSW_EVENTS = [
    (163, 138, 137, 165), (420, 46, 114, 210), (467, 136, 135, 216), (489, 46, 210, 221),
    (564, 136, 216, 235), (576, 150, 150, 238), (628, 136, 235, 248), (688, 150, 238, 256),
    (893, 198, 202, 286), (999, 198, 286, 307), (1120, 218, 234, 329), (1156, 218, 329, 338),
    (1444, 275, 310, 382), (1538, 275, 382, 390), (1649, 293, 334, 407), (1784, 313, 358, 426),
]

by_track = defaultdict(list)
with open(TRACK_RESULT, encoding="utf-8") as f:
    for line in f:
        p = line.strip().split(",")
        if len(p) < 6:
            continue
        by_track[int(p[1])].append((int(p[0]), float(p[2]), float(p[3]), p[8]))
for tid in by_track:
    by_track[tid].sort(key=lambda r: r[0])

# 每个track的"最后一次真实出现"记录，按最后帧排序，方便按帧窗口查询
last_seen = []  # (last_frame, tid, cx, cy, cls)
for tid, recs in by_track.items():
    f, cx, cy, cls = recs[-1]
    last_seen.append((f, tid, cx, cy, cls))

for gt_frame, gt_id, old_pid, new_pid in IDSW_EVENTS:
    pred_frame = gt_frame + 1
    new_rec = next((r for r in by_track.get(new_pid, []) if r[0] >= pred_frame), None)
    if new_rec is None:
        continue
    _, ncx, ncy, ncls = new_rec

    candidates = []
    for last_frame, tid, cx, cy, cls in last_seen:
        if tid == new_pid:
            continue
        if not (last_frame < pred_frame <= last_frame + TRACK_BUFFER):
            continue  # 该track在这一帧是否处于"仍未超时的Lost状态"
        # 用它最后已知位置判断是否在候选搜索半径内(注：这是"最后已知"而非当帧预测，
        # 只是粗略估计——真正预测位置会因为丢失时长走得更远，这里是保守/宽松估计)
        dist = ((cx - ncx) ** 2 + (cy - ncy) ** 2) ** 0.5
        if dist <= SEARCH_RADIUS:
            candidates.append((tid, last_frame, cls, dist))

    candidates.sort(key=lambda c: c[3])
    correct_is_nearest = candidates and candidates[0][0] == old_pid
    print(f"GT#{gt_id} f{gt_frame} (真正应匹配 old#{old_pid} -> new#{new_pid}): "
          f"半径{SEARCH_RADIUS}px内共有 {len(candidates)} 个候选track " +
          (f"[最近的是old#{candidates[0][0]}, dist={candidates[0][3]:.0f}px, "
           f"{'正确' if correct_is_nearest else '不是old_pid!'}]" if candidates else "[无候选]"))
    for tid, lf, cls, dist in candidates[:5]:
        mark = " <-- 应该匹配这个" if tid == old_pid else ""
        print(f"      候选 track#{tid} cls={cls} 最后出现帧={lf} 距离={dist:.0f}px{mark}")
