#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""分析 IDSW 事件发生的位置是否靠近切片边界（960x960 / 1280x1280，overlap=0.3）。"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import xml.etree.ElementTree as ET
import numpy as np
import cv2
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

GT_XML = r"E:\DJI\cvat\GT(1).xml"
PRED_TXT = r"D:\yolo26-tracking\outputs\yolo26-obb_20260722_003112\cvat_eval.txt"
IOU_THRESH = 0.5
CENTER_DIST_FILTER = 300
FRAME_W, FRAME_H = 3534, 1988
NEAR_THR = 40  # 像素，判定“靠近切片边界”的距离阈值


def get_slice_bboxes(H, W, sh=640, sw=640, oh=0.3, ow=0.3):
    y_overlap = int(oh * sh)
    x_overlap = int(ow * sw)
    bboxes = []
    y_min = 0
    while y_min < H:
        y_max = min(H, y_min + sh)
        y1 = max(0, y_max - sh)
        x_min = 0
        while x_min < W:
            x_max = min(W, x_min + sw)
            x1 = max(0, x_max - sw)
            bboxes.append((x1, y1, x_max, y_max))
            if x_max == W:
                break
            x_min = x_max - x_overlap
        if y_max == H:
            break
        y_min = y_max - y_overlap
    return bboxes


def boundary_lines(H, W, sh, sw, oh=0.3, ow=0.3):
    slices = get_slice_bboxes(H, W, sh, sw, oh, ow)
    xs = sorted({b[0] for b in slices} | {b[2] for b in slices})
    ys = sorted({b[1] for b in slices} | {b[3] for b in slices})
    xs = [x for x in xs if 0 < x < W]
    ys = [y for y in ys if 0 < y < H]
    return xs, ys


xs_960, ys_960 = boundary_lines(FRAME_H, FRAME_W, 960, 960)
xs_1280, ys_1280 = boundary_lines(FRAME_H, FRAME_W, 1280, 1280)
print(f"960x960  切片边界: x={xs_960}  y={ys_960}")
print(f"1280x1280 切片边界: x={xs_1280}  y={ys_1280}")


def min_dist_to_lines(x, y, xs, ys):
    d = float("inf")
    for lx in xs:
        d = min(d, abs(x - lx))
    for ly in ys:
        d = min(d, abs(y - ly))
    return d


def parse_gt(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    gt = defaultdict(list)
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
            gt[frame].append((gt_id, cx, cy, w, h, rot))
        for poly in track.findall("polygon"):
            if poly.get("outside") == "1":
                continue
            frame = int(poly.get("frame"))
            pts = np.array(
                [[float(v) for v in pair.split(",")] for pair in poly.get("points").split(";")],
                dtype=np.float32,
            )
            (cx, cy), (w, h), rot = cv2.minAreaRect(pts)
            gt[frame].append((gt_id, cx, cy, w, h, rot))
    return gt


def parse_pred(txt_path):
    pred = defaultdict(list)
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 10:
                continue
            frame_id = int(parts[0])
            track_id = int(parts[1])
            pts = np.array([
                [float(parts[2]), float(parts[3])],
                [float(parts[4]), float(parts[5])],
                [float(parts[6]), float(parts[7])],
                [float(parts[8]), float(parts[9])],
            ], dtype=np.float32)
            pred[frame_id].append((track_id, pts))
    return pred


def _iou(rr1, rr2):
    ret, inter_pts = cv2.rotatedRectangleIntersection(rr1, rr2)
    if ret == cv2.INTERSECT_NONE or inter_pts is None or len(inter_pts) == 0:
        return 0.0
    inter_area = cv2.contourArea(inter_pts)
    if inter_area <= 0:
        return 0.0
    area1 = rr1[1][0] * rr1[1][1]
    area2 = rr2[1][0] * rr2[1][1]
    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0.0


gt = parse_gt(GT_XML)
pred = parse_pred(PRED_TXT)
gt_frames = sorted(gt.keys())

gt_last_pred = {}
idsw_events = []  # (frame, gt_id, old_pid, new_pid, cx, cy)

for gt_frame in gt_frames:
    gt_list = gt[gt_frame]
    pred_frame = gt_frame + 1
    pred_list = pred.get(pred_frame, [])
    if not gt_list or not pred_list:
        continue

    gt_rects = [(gid, ((float(cx), float(cy)), (float(w), float(h)), float(rot)), cx, cy)
                for gid, cx, cy, w, h, rot in gt_list]
    pred_rects = []
    for track_id, pts in pred_list:
        rr = cv2.minAreaRect(pts)
        pcx, pcy = rr[0]
        pred_rects.append((track_id, rr, pcx, pcy))

    n_gt, n_pred = len(gt_rects), len(pred_rects)
    iou_mat = np.zeros((n_gt, n_pred), dtype=np.float32)
    for i, (_, rr_g, gcx, gcy) in enumerate(gt_rects):
        for j, (_, rr_p, pcx, pcy) in enumerate(pred_rects):
            if abs(gcx - pcx) > CENTER_DIST_FILTER or abs(gcy - pcy) > CENTER_DIST_FILTER:
                continue
            iou_mat[i, j] = _iou(rr_g, rr_p)

    row_ind, col_ind = linear_sum_assignment(1.0 - iou_mat)
    for r, c in zip(row_ind, col_ind):
        if iou_mat[r, c] < IOU_THRESH:
            continue
        gid = gt_rects[r][0]
        pred_id = pred_rects[c][0]
        gcx, gcy = gt_rects[r][2], gt_rects[r][3]
        if gid in gt_last_pred and gt_last_pred[gid] != pred_id:
            idsw_events.append((gt_frame, gid, gt_last_pred[gid], pred_id, gcx, gcy))
        gt_last_pred[gid] = pred_id

print(f"\n共 {len(idsw_events)} 次 IDSW，判定阈值: 距最近切片边界 <= {NEAR_THR}px 视为“靠近边界”\n")
print(f"{'帧号':>6} {'GTid':>6} {'旧pid':>6} {'新pid':>6} {'x':>8} {'y':>8} {'到960边界':>10} {'到1280边界':>10}  靠近?")

near_960 = near_1280 = near_either = 0
for frame, gid, old_pid, new_pid, x, y in idsw_events:
    d960 = min_dist_to_lines(x, y, xs_960, ys_960)
    d1280 = min_dist_to_lines(x, y, xs_1280, ys_1280)
    is_near_960 = d960 <= NEAR_THR
    is_near_1280 = d1280 <= NEAR_THR
    if is_near_960:
        near_960 += 1
    if is_near_1280:
        near_1280 += 1
    if is_near_960 or is_near_1280:
        near_either += 1
    tag = []
    if is_near_960:
        tag.append("960边界")
    if is_near_1280:
        tag.append("1280边界")
    tag_str = ",".join(tag) if tag else "-"
    print(f"{frame:>6} {gid:>6} {old_pid:>6} {new_pid:>6} {x:>8.1f} {y:>8.1f} {d960:>10.1f} {d1280:>10.1f}  {tag_str}")

n = len(idsw_events)
print(f"\n靠近960切片边界: {near_960}/{n} ({near_960/n*100:.1f}%)")
print(f"靠近1280切片边界: {near_1280}/{n} ({near_1280/n*100:.1f}%)")
print(f"靠近任一边界(960或1280): {near_either}/{n} ({near_either/n*100:.1f}%)")
