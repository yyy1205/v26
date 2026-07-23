#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对16次IDSW事件做根因定位：用track_result.txt(每帧每条track的完整输出，不限于GT匹配)
逐个事件回答:
  1. new_pred_id 在切换帧之前是否已经存在(是"已存在的另一条track把身份抢走"，
     还是"切换帧当场诞生的全新track")
  2. old_pred_id 消失(丢失)到 new_pred_id 出现/接管之间隔了多少帧(丢失时长)
  3. old_pred_id 最后一次出现的位置，与 new_pred_id 在切换帧的位置，中心点距离多大
     (间接反映：如果距离很小，说明就是同一物理车辆的框，纯粹是身份没接上；
      如果距离较大，可能是关联到了别的车辆)
"""
import sys
sys.path.insert(0, r"D:\yolo26-tracking")
from collections import defaultdict

TRACK_RESULT = r"D:\yolo26-tracking\outputs\yolo26-obb_20260722_155902\track_result.txt"

# (gt_frame, gt_id, old_pred_id, new_pred_id) —— 取自 eval_mot.py 最近一次(20260722_155902)的输出
IDSW_EVENTS = [
    (163, 138, 137, 165),
    (420, 46, 114, 210),
    (467, 136, 135, 216),
    (489, 46, 210, 221),
    (564, 136, 216, 235),
    (576, 150, 150, 238),
    (628, 136, 235, 248),
    (688, 150, 238, 256),
    (893, 198, 202, 286),
    (999, 198, 286, 307),
    (1120, 218, 234, 329),
    (1156, 218, 329, 338),
    (1444, 275, 310, 382),
    (1538, 275, 382, 390),
    (1649, 293, 334, 407),
    (1784, 313, 358, 426),
]

# track_id -> sorted list of (frame, cx, cy, w, h)
by_track = defaultdict(list)
with open(TRACK_RESULT, encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split(",")
        if len(parts) < 6:
            continue
        frame = int(parts[0])
        tid = int(parts[1])
        cx, cy, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
        by_track[tid].append((frame, cx, cy, w, h))

for tid in by_track:
    by_track[tid].sort(key=lambda r: r[0])

def last_before(tid, frame):
    """tid 在 < frame 的最后一次出现记录，没有则 None。"""
    recs = by_track.get(tid, [])
    prev = None
    for r in recs:
        if r[0] < frame:
            prev = r
        else:
            break
    return prev

def first_record(tid):
    recs = by_track.get(tid, [])
    return recs[0] if recs else None

def at_or_after(tid, frame):
    """tid 在 >= frame 的第一次出现记录。"""
    for r in by_track.get(tid, []):
        if r[0] >= frame:
            return r
    return None

print(f"{'GT帧':>6} {'GT id':>6} {'old_pred':>9} {'new_pred':>9} {'old最后帧':>9} {'丢失帧数':>8} "
      f"{'new首次出现帧':>12} {'new是否已存在':>14} {'中心距离px':>10}")
print("-" * 110)

for gt_frame, gt_id, old_pid, new_pid in IDSW_EVENTS:
    pred_frame = gt_frame + 1  # GT frame N = tracker frame N+1

    old_last = last_before(old_pid, pred_frame + 1)  # old track 在切换帧附近最后一次出现
    new_first = first_record(new_pid)
    new_at_switch = at_or_after(new_pid, pred_frame)

    old_last_frame = old_last[0] if old_last else None
    gap = (pred_frame - old_last_frame) if old_last_frame else None
    new_first_frame = new_first[0] if new_first else None
    pre_existing = (new_first_frame is not None and new_first_frame < pred_frame - 1)

    dist = None
    if old_last and new_at_switch:
        dist = ((old_last[1] - new_at_switch[1]) ** 2 + (old_last[2] - new_at_switch[2]) ** 2) ** 0.5

    print(f"{gt_frame:>6} {gt_id:>6} {old_pid:>9} {new_pid:>9} "
          f"{str(old_last_frame):>9} {str(gap):>8} {str(new_first_frame):>12} "
          f"{'是(早于切换帧)' if pre_existing else '否(当场新建)':>14} "
          f"{f'{dist:.1f}' if dist is not None else 'N/A':>10}")
