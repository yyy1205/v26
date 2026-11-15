# -*- coding: utf-8 -*-
"""
离线全局轨迹拼接后处理：跑完整段视频之后，对输出的track片段做一次事后拼接，
把"因为短暂遮挡/检测退化而被拆成两段的同一辆车"重新接回同一个id。

跟在线跟踪(tracktworoi.py)完全独立——只读已经生成的cvat_eval.txt，输出一份
拼接后的新cvat_eval.txt，不改动任何实时跟踪逻辑。

思路（专门针对"一个消失、一个诞生，没有第三方候选同时竞争"这种低歧义场景，
不是给ReID第三轮那种多对多竞争场景用的）：
1. 把每个track_id的死亡端(最后几帧)用恒速模型往后外推位置
2. 找间隔期内诞生的track，看外推位置/尺寸/类别是否吻合
3. 全局匈牙利匹配(不是贪心)，只接受同时满足位置/尺寸双重阈值、且是全局最优的唯一配对
4. 拼接后重新编号输出
"""
import numpy as np
import cv2
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

PRED_TXT = r"D:\yolo26-tracking\outputs\yolo26-obb_20260804_175700\cvat_eval.txt"
OUT_TXT = r"D:\yolo26-tracking\outputs\yolo26-obb_20260804_175700\cvat_eval_stitched.txt"

MAX_GAP_FRAMES = 45          # 收紧：只覆盖已验证的短间隔场景(6/16/20/37帧)，
                              # 长间隔(90-120帧)车流量大、候选太多，外推可信度不够，不冒险处理
VELOCITY_WINDOW = 10         # 用死亡/诞生端最后/最前多少帧估计速度
POS_TOL_BASE = 30.0          # 外推位置容忍度基础值(像素)
POS_TOL_PER_FRAME = 4.0      # 每多一帧间隔，容忍度增加的像素数(留出车速非严格恒定的余量)
MIN_SIZE_IOU = 0.5           # 尺寸一致性门槛，跟第三轮ReID恢复用的同一个阈值


def _size_iou(w1, h1, w2, h2):
    """旋转无关的尺寸一致性：把(w,h)统一成(长边,短边)再比。"""
    l1, s1 = max(w1, h1), min(w1, h1)
    l2, s2 = max(w2, h2), min(w2, h2)
    if l1 <= 0 or l2 <= 0 or s1 <= 0 or s2 <= 0:
        return 0.0
    l_iou = min(l1, l2) / max(l1, l2)
    s_iou = min(s1, s2) / max(s1, s2)
    return l_iou * s_iou


def parse_pred(path):
    """{frame: [(track_id, pts(4,2), cls)]}"""
    frames = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) < 13:
                continue
            frame = int(p[0])
            tid = int(p[1])
            pts = np.array([[float(p[2 + 2 * i]), float(p[3 + 2 * i])] for i in range(4)], dtype=np.float32)
            cls = int(p[11])
            frames[frame].append((tid, pts, cls))
    return frames


def build_tracks(frames):
    """{track_id: [(frame, cx, cy, w, h, rot, cls), ...]} 按frame排序"""
    tracks = defaultdict(list)
    for frame, boxes in frames.items():
        for tid, pts, cls in boxes:
            (cx, cy), (w, h), rot = cv2.minAreaRect(pts)
            tracks[tid].append((frame, cx, cy, w, h, rot, cls))
    for tid in tracks:
        tracks[tid].sort(key=lambda r: r[0])
    return tracks


def estimate_velocity(rows, n=VELOCITY_WINDOW, from_end=True):
    """用track末端(或首端)最近n个观测点做最小二乘线性拟合，返回(vx, vy)。"""
    sub = rows[-n:] if from_end else rows[:n]
    if len(sub) < 2:
        return 0.0, 0.0
    frames = np.array([r[0] for r in sub], dtype=np.float64)
    xs = np.array([r[1] for r in sub], dtype=np.float64)
    ys = np.array([r[2] for r in sub], dtype=np.float64)
    vx = np.polyfit(frames, xs, 1)[0]
    vy = np.polyfit(frames, ys, 1)[0]
    return vx, vy


def find_stitch_pairs(tracks):
    """返回接受拼接的 (old_tid, new_tid, info) 列表，全局匈牙利最优匹配。"""
    dying, newborn = [], []
    for tid, rows in tracks.items():
        dying.append((tid, rows))
        newborn.append((tid, rows))

    n_d, n_b = len(dying), len(newborn)
    cost = np.full((n_d, n_b), 1e6, dtype=np.float64)
    info_map = {}

    for i, (old_tid, old_rows) in enumerate(dying):
        last_frame, lcx, lcy, lw, lh, lrot, lcls = old_rows[-1]
        vx, vy = estimate_velocity(old_rows, from_end=True)

        for j, (new_tid, new_rows) in enumerate(newborn):
            if new_tid == old_tid:
                continue
            first_frame, fcx, fcy, fw, fh, frot, fcls = new_rows[0]
            gap = first_frame - last_frame
            if gap <= 0 or gap > MAX_GAP_FRAMES:
                continue
            if fcls != lcls:
                continue

            pred_x = lcx + vx * gap
            pred_y = lcy + vy * gap
            dist = float(np.hypot(pred_x - fcx, pred_y - fcy))
            tol = POS_TOL_BASE + POS_TOL_PER_FRAME * gap
            if dist > tol:
                continue

            size_iou = _size_iou(lw, lh, fw, fh)
            if size_iou < MIN_SIZE_IOU:
                continue

            # 代价 = 位置残差(归一化到容忍度) + 尺寸不一致惩罚，越小越好
            c = (dist / tol) + (1.0 - size_iou)
            cost[i, j] = c
            info_map[(i, j)] = (gap, dist, size_iou)

    row_ind, col_ind = linear_sum_assignment(cost)
    accepted = []
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] >= 1e6:
            continue
        old_tid = dying[r][0]
        new_tid = newborn[c][0]
        gap, dist, size_iou = info_map[(r, c)]
        accepted.append((old_tid, new_tid, gap, dist, size_iou))
    return accepted


def apply_stitch(frames, accepted):
    """用并查集把链式拼接(A->B->C)压平成同一个根id，重写输出。"""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra  # 保留更早出现(old_tid)的根

    for old_tid, new_tid, *_ in accepted:
        union(old_tid, new_tid)

    remap = {}
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        for frame in sorted(frames.keys()):
            for tid, pts, cls in frames[frame]:
                root = find(tid) if tid in parent else tid
                remap[tid] = root
                f.write(
                    f"{frame},{root},{pts[0][0]:.2f},{pts[0][1]:.2f},{pts[1][0]:.2f},{pts[1][1]:.2f},"
                    f"{pts[2][0]:.2f},{pts[2][1]:.2f},{pts[3][0]:.2f},{pts[3][1]:.2f},1,{cls},1.0\n"
                )
    return remap


def main():
    frames = parse_pred(PRED_TXT)
    tracks = build_tracks(frames)
    print(f"共 {len(tracks)} 条轨迹，开始寻找可拼接的片段对...")

    accepted = find_stitch_pairs(tracks)
    print(f"\n接受拼接 {len(accepted)} 对：")
    for old_tid, new_tid, gap, dist, size_iou in accepted:
        print(f"  id{old_tid} -> id{new_tid}  间隔{gap}帧  外推误差={dist:.1f}px  尺寸一致性={size_iou:.3f}")

    remap = apply_stitch(frames, accepted)
    n_merged_groups = len(set(remap.values()) | (set(tracks.keys()) - set(remap.keys())))
    print(f"\n拼接后轨迹数: {len(tracks)} -> {n_merged_groups}")
    print(f"输出: {OUT_TXT}")


if __name__ == "__main__":
    main()
