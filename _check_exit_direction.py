# -*- coding: utf-8 -*-
"""
修正版边缘判据：不是"最后位置在边缘"就算数，而是要看丢失前的运动方向是"朝边缘驶出"还是
"刚从边缘驶入/在边缘附近平行运动"。只有朝最近边界方向运动(速度矢量在出边界法线方向上的投影为正)
才算真正"驶离画面"，反之(驶入或背离边界)不算。

对两组数据都重算：
  A) 24次IDSW里12次"边缘僵尸复活"事件的候选track
  B) 全视频192次track内部长间隔(>=30帧)自恢复(大概率合法)
"""
from collections import defaultdict
import numpy as np

PRED_TXT = r"D:\yolo26-tracking\outputs\yolo26-obb_20260723_213702\cvat_eval.txt"

EDGE_XMIN, EDGE_XMAX = 0, 3534
EDGE_YMIN, EDGE_YMAX = 0, 1988
EDGE_MARGIN = 100

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


def nearest_edge_outward(x, y):
    """返回(是否在边缘带内, 该边界的外法线方向单位向量)"""
    d_left = x - EDGE_XMIN
    d_right = EDGE_XMAX - x
    d_top = y - EDGE_YMIN
    d_bottom = EDGE_YMAX - y
    d_min = min(d_left, d_right, d_top, d_bottom)
    if d_min >= EDGE_MARGIN:
        return False, (0, 0)
    if d_min == d_left:
        return True, (-1, 0)
    if d_min == d_right:
        return True, (1, 0)
    if d_min == d_top:
        return True, (0, -1)
    return True, (0, 1)


def classify(hist_before_gap):
    """hist_before_gap: 丢失前按帧排序的[(frame,x,y),...]，取最后若干帧估计速度方向。
    返回 'EXITING'(朝边缘驶出，应排除) / 'ENTERING_OR_PASSING'(驶入或平行，应保留) / 'NOT_EDGE' / 'NO_VELOCITY'
    """
    last_frame, lx, ly = hist_before_gap[-1]
    is_edge, outward = nearest_edge_outward(lx, ly)
    if not is_edge:
        return "NOT_EDGE", None
    # 用最后最多5个点做一个简单的首尾差分速度估计(比相邻两帧抗噪一点)
    window = hist_before_gap[-5:] if len(hist_before_gap) >= 2 else hist_before_gap
    if len(window) < 2:
        return "NO_VELOCITY", None
    f0, x0, y0 = window[0]
    f1, x1, y1 = window[-1]
    dt = f1 - f0
    if dt <= 0:
        return "NO_VELOCITY", None
    vx, vy = (x1 - x0) / dt, (y1 - y0) / dt
    dot = vx * outward[0] + vy * outward[1]
    speed = (vx ** 2 + vy ** 2) ** 0.5
    if speed < 0.3:  # 基本静止，方向不可靠
        return "ENTERING_OR_PASSING", dot
    return ("EXITING" if dot > 0.5 else "ENTERING_OR_PASSING"), dot


# ── A) 12次边缘僵尸复活事件的"被选中"候选 ──────────────────────────────
# (revival_frame, chosen_track_id) 来自上一步日志里 [被选中] 且真正对应24-IDSW明细的那个
ZOMBIE_EVENTS = [
    (147, 31), (467, 22), (548, 135), (564, 189), (571, 113),
    (661, 23), (674, 113), (966, 16), (1142, 110), (1536, 226),
    (1649, 97), (1652, 235),
]

print("=" * 90)
print("A) 12次边缘僵尸复活事件 —— 按运动方向重新判定")
print("=" * 90)
for revival_frame, tid in ZOMBIE_EVENTS:
    hist = tracks.get(tid, [])
    before = [r for r in hist if r[0] < revival_frame]
    if not before:
        print(f"revival={revival_frame} track#{tid}: 无历史数据，跳过")
        continue
    label, dot = classify(before)
    lf, lx, ly = before[-1]
    print(f"revival={revival_frame:>5} track#{tid:>4} last_pos=({lx:>7.1f},{ly:>7.1f}) frame={lf:>5} "
          f"-> {label} (dot={dot})")

# ── B) 全视频192次内部自恢复 ──────────────────────────────────────────
internal = []
for tid, hist in tracks.items():
    for i in range(1, len(hist)):
        prev_frame, px, py = hist[i - 1]
        cur_frame, cx, cy = hist[i]
        gap = cur_frame - prev_frame
        if gap >= 30:
            before = hist[:i]  # 到丢失前为止的完整历史
            label, dot = classify(before)
            internal.append((tid, gap, prev_frame, px, py, label, dot))

print()
print("=" * 90)
print("B) 全视频192次track内部长间隔自恢复 —— 按运动方向重新判定统计")
print("=" * 90)
from collections import Counter
counts = Counter(r[5] for r in internal)
print(f"总数: {len(internal)}")
for k, v in counts.items():
    print(f"  {k}: {v}")

print()
print("详情(按EXITING排在前面，方便检查):")
internal.sort(key=lambda r: (r[5] != "EXITING", -r[1]))
print(f"{'track_id':>8} {'gap':>5} {'消失帧':>8} {'消失前位置':>18} {'判定':>22} {'dot':>8}")
for tid, gap, pf, px, py, label, dot in internal:
    dot_s = f"{dot:.2f}" if dot is not None else "  -"
    print(f"{tid:>8} {gap:>5} {pf:>8} ({px:>7.1f},{py:>7.1f})   {label:>22} {dot_s:>8}")
