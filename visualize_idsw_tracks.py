#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把发生过 IDSW 的 GT 轨迹画在一张图上：
- 每条 GT 轨迹一种颜色，细线连接逐帧 GT 中心点（完整轨迹）
- 绿色小圆点 : 正确跟上 (TP)
- 红色叉 (x) : 漏检 (FN，GT 存在但没有匹配到任何预测框)
- 黑色五角星 : 发生 IDSW 的帧（GT 的匹配 pred_id 与上一帧不同）
- 橙色三角  : 框突变（同一 pred_id 连续两帧之间 w 或 h 变化超过 40%）
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import xml.etree.ElementTree as ET
import numpy as np
import cv2
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

GT_XML = r"E:\DJI\cvat\GT(1).xml"
PRED_TXT = r"D:\yolo26-tracking\outputs\yolo26-obb_20260722_155902\cvat_eval.txt"
BACKGROUND = r"D:\yolo26-tracking\outputs\yolo26-obb_20260722_155902\roi_selected.jpg"
OUT_PATH = r"D:\yolo26-tracking\outputs\yolo26-obb_20260722_155902\idsw_tracks_visualization.jpg"

IOU_THRESH = 0.5
CENTER_DIST_FILTER = 300
BOX_JUMP_RATIO = 0.4  # 同一 pred_id 连续帧 w/h 变化超过这个比例视为“框突变”

# 用了新的检测/NMS逻辑后，重新跑全视频剩下的16次IDSW，涉及这9条GT轨迹（id=146已经不在里面了）
TARGET_IDS = [46, 136, 138, 150, 198, 218, 275, 293, 313]


def parse_gt(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    gt = defaultdict(list)
    labels = {}
    for track in root.findall("track"):
        gt_id = int(track.get("id"))
        labels[gt_id] = track.get("label")
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
    return gt, labels


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


def main():
    print(f"解析 GT  : {GT_XML}")
    gt, labels = parse_gt(GT_XML)
    print(f"解析 Pred: {PRED_TXT}")
    pred = parse_pred(PRED_TXT)
    gt_frames = sorted(gt.keys())

    target_set = set(TARGET_IDS)
    # records[gt_id] = list of dict(frame, cx, cy, status, pred_w, pred_h, pred_id)
    records = defaultdict(list)
    gt_last_pred = {}

    for gt_frame in gt_frames:
        gt_list = gt[gt_frame]
        pred_frame = gt_frame + 1
        pred_list = pred.get(pred_frame, [])

        relevant_gt = [g for g in gt_list if g[0] in target_set]
        if not relevant_gt:
            continue

        if not pred_list:
            for gid, cx, cy, w, h, rot in relevant_gt:
                records[gid].append(dict(frame=gt_frame, cx=cx, cy=cy, status="FN"))
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
        matched = {}
        for r, c in zip(row_ind, col_ind):
            if iou_mat[r, c] < IOU_THRESH:
                continue
            matched[r] = c

        for i, (gid, rr_g, gcx, gcy) in enumerate(gt_rects):
            if gid not in target_set:
                continue
            if i in matched:
                c = matched[i]
                pred_id, rr_p, pcx, pcy = pred_rects[c]
                (pw, ph) = rr_p[1]
                is_idsw = gid in gt_last_pred and gt_last_pred[gid] != pred_id
                records[gid].append(dict(frame=gt_frame, cx=gcx, cy=gcy, status="TP",
                                          pred_id=pred_id, pw=pw, ph=ph, idsw=is_idsw))
                gt_last_pred[gid] = pred_id
            else:
                records[gid].append(dict(frame=gt_frame, cx=gcx, cy=gcy, status="FN"))

    # ── 标记框突变：同一 pred_id 连续帧之间 w/h 变化超过阈值 ──────────
    for gid, recs in records.items():
        recs.sort(key=lambda r: r["frame"])
        prev = None
        for r in recs:
            if r["status"] != "TP":
                prev = None
                continue
            if prev is not None and prev["pred_id"] == r["pred_id"]:
                dw = abs(r["pw"] - prev["pw"]) / max(prev["pw"], 1e-6)
                dh = abs(r["ph"] - prev["ph"]) / max(prev["ph"], 1e-6)
                r["box_jump"] = dw > BOX_JUMP_RATIO or dh > BOX_JUMP_RATIO
            else:
                r["box_jump"] = False
            prev = r

    # ── 画图 ──────────────────────────────────────────────────────
    bg = cv2.imread(BACKGROUND)
    if bg is None:
        raise RuntimeError(f"无法读取背景图: {BACKGROUND}")
    canvas = bg.copy()

    rng = np.random.RandomState(0)
    colors = {}
    for gid in TARGET_IDS:
        rng2 = np.random.RandomState(gid * 97 + 13)
        colors[gid] = tuple(int(x) for x in rng2.randint(60, 256, size=3))

    n_fn = n_idsw = n_jump = 0
    for gid in TARGET_IDS:
        recs = records.get(gid, [])
        if not recs:
            continue
        color = colors[gid]
        pts = [(int(r["cx"]), int(r["cy"])) for r in recs]
        for p1, p2 in zip(pts[:-1], pts[1:]):
            cv2.line(canvas, p1, p2, color, 2, cv2.LINE_AA)

        for r in recs:
            p = (int(r["cx"]), int(r["cy"]))
            if r["status"] == "FN":
                n_fn += 1
                cv2.drawMarker(canvas, p, (0, 0, 255), cv2.MARKER_TILTED_CROSS, 14, 3)
            else:
                cv2.circle(canvas, p, 3, color, -1)
                if r.get("idsw"):
                    n_idsw += 1
                    cv2.drawMarker(canvas, p, (0, 0, 0), cv2.MARKER_STAR, 22, 2)
                if r.get("box_jump"):
                    n_jump += 1
                    cv2.drawMarker(canvas, p, (0, 140, 255), cv2.MARKER_TRIANGLE_UP, 18, 3)

        # 在轨迹起点标注 GT id + 类别
        label = f"{gid}:{labels.get(gid,'?')}"
        cv2.putText(canvas, label, pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
        cv2.putText(canvas, label, pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 1, cv2.LINE_AA)

    # 图例
    legend_x, legend_y = 20, 40
    items = [
        ("FN 漏检 (红色叉)", (0, 0, 255)),
        ("IDSW (黑色星)", (0, 0, 0)),
        ("框突变 (橙色三角)", (0, 140, 255)),
    ]
    for i, (text, c) in enumerate(items):
        y = legend_y + i * 40
        cv2.rectangle(canvas, (legend_x - 10, y - 25), (legend_x + 420, y + 10), (255, 255, 255), -1)
        cv2.putText(canvas, text, (legend_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, c, 2, cv2.LINE_AA)

    cv2.imwrite(OUT_PATH, canvas)
    print(f"\n共 {len(TARGET_IDS)} 条轨迹，FN 标记 {n_fn} 处，IDSW 标记 {n_idsw} 处，框突变标记 {n_jump} 处")
    print(f"已保存到: {OUT_PATH}")


if __name__ == "__main__":
    main()
