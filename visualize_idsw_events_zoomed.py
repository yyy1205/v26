#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对16次IDSW事件逐个放大可视化，用于验证"长丢失后纯匀速外推预测漂移导致关联失败"这一假设。

每个事件画一张子图，展示：
  - 蓝色框: old_pred_id 最后一次真实被跟踪到的框(丢失前最后位置)
  - 绿色轨迹点+连线: 丢失期间 GT 的真实运动轨迹(从old track消失到重新出现)
  - 红色 X: 用 old track 消失前的速度做纯匀速外推，得到的"如果Kalman纯预测会落在哪"
  - 黄色框: new_pred_id 在重新出现帧的框(=新建track的初始位置，约等于真实检测位置)
  - 标题标注: 丢失帧数、朴素匀速外推位置与真实重现位置的距离(距离越大，越支持"纯匀速假设本身就不成立/车辆做了非匀速运动"这个解释)

数据来源：GT XML + track_result.txt(不需要重新跑视频)
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import xml.etree.ElementTree as ET
import numpy as np
import cv2
from collections import defaultdict

GT_XML = r"E:\DJI\cvat\GT(1).xml"
TRACK_RESULT = r"D:\yolo26-tracking\outputs\yolo26-obb_20260722_155902\track_result.txt"
BACKGROUND = r"D:\yolo26-tracking\outputs\yolo26-obb_20260722_155902\roi_selected.jpg"
OUT_PATH = r"D:\yolo26-tracking\outputs\yolo26-obb_20260722_155902\idsw_root_cause_grid.jpg"

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

CELL = 520  # 每个子图渲染尺寸(正方形)，最后再统一缩放拼接
GRID_COLS = 4


def parse_gt(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    gt = defaultdict(dict)  # gt[gt_id][frame] = (cx, cy, w, h, rot)
    for track in root.findall("track"):
        gt_id = int(track.get("id"))
        for box in track.findall("box"):
            if box.get("outside") == "1":
                continue
            frame = int(box.get("frame"))
            xtl, ytl = float(box.get("xtl")), float(box.get("ytl"))
            xbr, ybr = float(box.get("xbr")), float(box.get("ybr"))
            rot = float(box.get("rotation", 0))
            cx, cy = (xtl + xbr) / 2, (ytl + ybr) / 2
            w, h = xbr - xtl, ybr - ytl
            gt[gt_id][frame] = (cx, cy, w, h, rot)
        for poly in track.findall("polygon"):
            if poly.get("outside") == "1":
                continue
            frame = int(poly.get("frame"))
            pts = np.array(
                [[float(v) for v in pair.split(",")] for pair in poly.get("points").split(";")],
                dtype=np.float32,
            )
            (cx, cy), (w, h), rot = cv2.minAreaRect(pts)
            gt[gt_id][frame] = (cx, cy, w, h, rot)
    return gt


def parse_track_result(path):
    by_track = defaultdict(list)  # track_id -> sorted [(frame, cx, cy, w, h, angle), ...]
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 7:
                continue
            frame = int(parts[0]); tid = int(parts[1])
            cx, cy, w, h, ang = (float(parts[2]), float(parts[3]), float(parts[4]),
                                  float(parts[5]), float(parts[6]))
            by_track[tid].append((frame, cx, cy, w, h, ang))
    for tid in by_track:
        by_track[tid].sort(key=lambda r: r[0])
    return by_track


def draw_rot_box(img, cx, cy, w, h, rot_deg, color, thickness=2, label=None):
    rect = ((cx, cy), (max(w, 4), max(h, 4)), rot_deg)
    box = cv2.boxPoints(rect).astype(int)
    cv2.drawContours(img, [box], 0, color, thickness, cv2.LINE_AA)
    if label:
        cv2.putText(img, label, (int(cx - w / 2), int(cy - h / 2 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def render_event(bg, gt, by_track, gt_frame, gt_id, old_pid, new_pid):
    pred_frame = gt_frame + 1

    old_hist = [r for r in by_track.get(old_pid, []) if r[0] <= pred_frame]
    if not old_hist:
        return None
    old_last = old_hist[-1]
    old_last_frame, old_lcx, old_lcy, old_lw, old_lh, old_lang = old_last

    # 用 old track 消失前最后两帧估计速度(px/frame)
    if len(old_hist) >= 2:
        prev = old_hist[-2]
        dt = max(old_last_frame - prev[0], 1)
        vx = (old_last[1] - prev[1]) / dt
        vy = (old_last[2] - prev[2]) / dt
    else:
        vx = vy = 0.0

    gap = pred_frame - old_last_frame
    naive_x = old_lcx + vx * gap
    naive_y = old_lcy + vy * gap

    new_hist = [r for r in by_track.get(new_pid, []) if r[0] >= pred_frame]
    if not new_hist:
        return None
    new_first = new_hist[0]
    _, new_cx, new_cy, new_w, new_h, new_ang = new_first

    dist_naive_to_true = float(np.hypot(naive_x - new_cx, naive_y - new_cy))

    # GT 真实轨迹：从 old_last_frame-1 (GT帧号=pred帧号-1) 到 gt_frame
    gt_track = gt.get(gt_id, {})
    gt_pts = []
    for f in range(max(old_last_frame - 1, 0), gt_frame + 1):
        if f in gt_track:
            gt_pts.append((f, gt_track[f][0], gt_track[f][1]))

    # 裁剪区域：覆盖 old位置、naive预测位置、new位置、GT轨迹 的包围盒 + padding
    xs = [old_lcx, naive_x, new_cx] + [p[1] for p in gt_pts]
    ys = [old_lcy, naive_y, new_cy] + [p[2] for p in gt_pts]
    pad = 150
    x0, x1 = max(min(xs) - pad, 0), max(xs) + pad
    y0, y1 = max(min(ys) - pad, 0), max(ys) + pad
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    x1 = min(x1, bg.shape[1]); y1 = min(y1, bg.shape[0])
    if x1 <= x0 or y1 <= y0:
        return None
    crop = bg[y0:y1, x0:x1].copy()

    def to_crop(px, py):
        return int(px - x0), int(py - y0)

    # GT 真实轨迹(绿色)
    gt_crop_pts = [to_crop(p[1], p[2]) for p in gt_pts]
    for p1, p2 in zip(gt_crop_pts[:-1], gt_crop_pts[1:]):
        cv2.line(crop, p1, p2, (0, 200, 0), 2, cv2.LINE_AA)
    for p in gt_crop_pts:
        cv2.circle(crop, p, 3, (0, 200, 0), -1)

    # old track 最后位置(蓝色框)
    ocx, ocy = to_crop(old_lcx, old_lcy)
    draw_rot_box(crop, ocx, ocy, old_lw, old_lh, np.rad2deg(old_lang), (255, 140, 0), 2,
                 f"old#{old_pid}@f{old_last_frame}")

    # 朴素匀速外推位置(红色X)
    nx, ny = to_crop(naive_x, naive_y)
    cv2.drawMarker(crop, (nx, ny), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 22, 3)
    cv2.putText(crop, "naive predict", (nx + 10, ny), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 255), 2, cv2.LINE_AA)

    # new track 首次出现位置(黄色框，约等于真实检测位置)
    ycx, ycy = to_crop(new_cx, new_cy)
    draw_rot_box(crop, ycx, ycy, new_w, new_h, np.rad2deg(new_ang), (0, 220, 255), 2,
                 f"new#{new_pid}@f{pred_frame}")

    crop = cv2.resize(crop, (CELL, CELL - 60), interpolation=cv2.INTER_AREA)
    canvas = np.full((CELL, CELL, 3), 255, dtype=np.uint8)
    canvas[60:, :] = crop
    title1 = f"GT#{gt_id}  gap={gap}f  old#{old_pid}->new#{new_pid}"
    title2 = f"naive_predict-to-true dist = {dist_naive_to_true:.0f}px"
    cv2.putText(canvas, title1, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, title2, (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 150), 1, cv2.LINE_AA)
    return canvas


def main():
    print(f"解析 GT: {GT_XML}")
    gt = parse_gt(GT_XML)
    print(f"解析 track_result: {TRACK_RESULT}")
    by_track = parse_track_result(TRACK_RESULT)
    bg = cv2.imread(BACKGROUND)
    if bg is None:
        raise RuntimeError(f"无法读取背景图: {BACKGROUND}")

    cells = []
    for gt_frame, gt_id, old_pid, new_pid in IDSW_EVENTS:
        cell = render_event(bg, gt, by_track, gt_frame, gt_id, old_pid, new_pid)
        if cell is None:
            cell = np.full((CELL, CELL, 3), 200, dtype=np.uint8)
            cv2.putText(cell, f"GT#{gt_id} f{gt_frame}: 数据不足", (10, CELL // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        cells.append(cell)

    n = len(cells)
    rows = (n + GRID_COLS - 1) // GRID_COLS
    grid = np.full((rows * CELL, GRID_COLS * CELL, 3), 255, dtype=np.uint8)
    for i, cell in enumerate(cells):
        r, c = divmod(i, GRID_COLS)
        grid[r * CELL:(r + 1) * CELL, c * CELL:(c + 1) * CELL] = cell

    cv2.imwrite(OUT_PATH, grid)
    print(f"已保存: {OUT_PATH}")


if __name__ == "__main__":
    main()
