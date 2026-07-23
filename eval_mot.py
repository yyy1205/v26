#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MOT 评估脚本：MOTA / MOTP / Precision / Recall / IDSW / IDF1
GT  格式: CVAT XML  (xtl, ytl, xbr, ybr, rotation_deg, frame 从 0 开始)
Pred格式: cvat_eval.txt (frame, track_id, p1x,p1y, p2x,p2y, p3x,p3y, p4x,p4y, ...)
帧号偏移: GT frame N = Tracker frame N+1
"""

import os, sys, time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import xml.etree.ElementTree as ET
import numpy as np
import cv2
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

# ─── 配置 ────────────────────────────────────────────────────────────────────
GT_XML   = r"E:\DJI\cvat\GT(1).xml"
PRED_TXT = r"D:\yolo26-tracking\outputs\yolo26-obb_20260723_213702\cvat_eval.txt"
IOU_THRESH = 0.5          # TP 判定阈值
CENTER_DIST_FILTER = 300  # 像素，超过此距离直接跳过 IoU 计算（加速）
# ─────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# 1. 解析
# ══════════════════════════════════════════════════════════════════════════════

def parse_gt(xml_path):
    """返回 {gt_frame: [(gt_id, cx, cy, w, h, rot_deg)]}，跳过 outside=1 的框。

    兼容两种 CVAT for video 导出形式：
    - <box xtl ytl xbr ybr rotation>：轴对齐框 + 旋转角
    - <polygon points="x1,y1;x2,y2;x3,y3;x4,y4">：4 点旋转矩形（用 cv2.minAreaRect 还原 cx,cy,w,h,rot）
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    gt = defaultdict(list)
    for track in root.findall("track"):
        gt_id = int(track.get("id"))
        for box in track.findall("box"):
            if box.get("outside") == "1":
                continue
            frame = int(box.get("frame"))
            xtl = float(box.get("xtl"));  ytl = float(box.get("ytl"))
            xbr = float(box.get("xbr"));  ybr = float(box.get("ybr"))
            rot = float(box.get("rotation", 0))
            cx = (xtl + xbr) / 2;  cy = (ytl + ybr) / 2
            w  = xbr - xtl;         h  = ybr - ytl
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
    """返回 {pred_frame: [(track_id, pts_4x2)]}"""
    pred = defaultdict(list)
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 10:
                continue
            frame_id  = int(parts[0])
            track_id  = int(parts[1])
            pts = np.array([
                [float(parts[2]), float(parts[3])],
                [float(parts[4]), float(parts[5])],
                [float(parts[6]), float(parts[7])],
                [float(parts[8]), float(parts[9])],
            ], dtype=np.float32)
            pred[frame_id].append((track_id, pts))
    return pred


# ══════════════════════════════════════════════════════════════════════════════
# 2. OBB IoU（cv2，比 shapely 快 ~10x）
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# 3. 逐帧评估
# ══════════════════════════════════════════════════════════════════════════════

def evaluate():
    t0 = time.time()
    print(f"解析 GT  : {GT_XML}")
    gt   = parse_gt(GT_XML)
    print(f"解析 Pred: {PRED_TXT}")
    pred = parse_pred(PRED_TXT)

    gt_frames = sorted(gt.keys())
    print(f"GT 帧数={len(gt_frames)}  GT 总框={sum(len(v) for v in gt.values())}")
    print(f"Pred 帧数={len(pred)}  Pred 总框={sum(len(v) for v in pred.values())}\n")

    total_gt   = 0
    total_tp   = 0
    total_fp   = 0
    total_fn   = 0
    total_idsw = 0
    iou_sum    = 0.0

    # ID Switch 追踪：gt_id → 上一次匹配到的 pred_id
    gt_last_pred: dict[int, int] = {}
    idsw_events = []  # (gt_frame, gt_id, old_pred_id, new_pred_id)

    # IDF1 所需：每条 GT 轨迹 / Pred 轨迹的帧级 TP 统计
    # gt_id  → 被正确关联的帧数（IDTP 贡献）
    # pred_id→ 被错误关联的帧数（IDFP 贡献，即 FP 帧）
    gt_idtp:   dict[int, int] = defaultdict(int)
    gt_idfn:   dict[int, int] = defaultdict(int)
    pred_idfp: dict[int, int] = defaultdict(int)

    unique_pred_ids: set[int] = set()

    for fi, gt_frame in enumerate(gt_frames):
        pred_frame = gt_frame + 1           # 帧号偏移

        gt_list   = gt[gt_frame]            # [(gt_id, cx, cy, w, h, rot)]
        pred_list = pred.get(pred_frame, []) # [(track_id, pts)]

        total_gt += len(gt_list)
        for tid, _ in pred_list:
            unique_pred_ids.add(tid)

        # ── 快速路径 ──────────────────────────────────────────────
        if not gt_list and not pred_list:
            continue
        if not gt_list:
            total_fp += len(pred_list)
            for tid, _ in pred_list:
                pred_idfp[tid] += 1
            continue
        if not pred_list:
            total_fn += len(gt_list)
            for gid, *_ in gt_list:
                gt_idfn[gid] += 1
            continue

        # ── 构建 RotatedRect ─────────────────────────────────────
        gt_rects   = []   # (gt_id, rr, cx, cy)
        pred_rects = []   # (track_id, rr, cx, cy)

        for gt_id, cx, cy, w, h, rot in gt_list:
            rr = ((float(cx), float(cy)), (float(w), float(h)), float(rot))
            gt_rects.append((gt_id, rr, cx, cy))

        for track_id, pts in pred_list:
            rr = cv2.minAreaRect(pts)
            pcx, pcy = rr[0]
            pred_rects.append((track_id, rr, pcx, pcy))

        n_gt   = len(gt_rects)
        n_pred = len(pred_rects)

        # ── IoU 矩阵（带中心距离预过滤）────────────────────────
        iou_mat = np.zeros((n_gt, n_pred), dtype=np.float32)
        for i, (_, rr_g, gcx, gcy) in enumerate(gt_rects):
            for j, (_, rr_p, pcx, pcy) in enumerate(pred_rects):
                if abs(gcx - pcx) > CENTER_DIST_FILTER or abs(gcy - pcy) > CENTER_DIST_FILTER:
                    continue
                iou_mat[i, j] = _iou(rr_g, rr_p)

        # ── 匈牙利匹配 ──────────────────────────────────────────
        row_ind, col_ind = linear_sum_assignment(1.0 - iou_mat)

        matched_gt   = set()
        matched_pred = set()

        for r, c in zip(row_ind, col_ind):
            if iou_mat[r, c] < IOU_THRESH:
                continue
            gt_id   = gt_rects[r][0]
            pred_id = pred_rects[c][0]
            iou_val = float(iou_mat[r, c])

            total_tp += 1
            iou_sum  += iou_val
            matched_gt.add(r)
            matched_pred.add(c)
            gt_idtp[gt_id] += 1

            # ID Switch
            if gt_id in gt_last_pred and gt_last_pred[gt_id] != pred_id:
                total_idsw += 1
                idsw_events.append((gt_frame, gt_id, gt_last_pred[gt_id], pred_id))
            gt_last_pred[gt_id] = pred_id

        # 未匹配 GT → FN
        for i in range(n_gt):
            if i not in matched_gt:
                gid = gt_rects[i][0]
                total_fn += 1
                gt_idfn[gid] += 1

        # 未匹配 Pred → FP
        for j in range(n_pred):
            if j not in matched_pred:
                tid = pred_rects[j][0]
                total_fp += 1
                pred_idfp[tid] += 1

        if (fi + 1) % 500 == 0:
            elapsed = time.time() - t0
            pct = (fi + 1) / len(gt_frames) * 100
            print(f"  {fi+1}/{len(gt_frames)} 帧 ({pct:.1f}%)  已用时 {elapsed:.0f}s ...")

    # ══════════════════════════════════════════════════════════════════════════
    # 4. 指标计算
    # ══════════════════════════════════════════════════════════════════════════

    mota      = 1.0 - (total_fn + total_fp + total_idsw) / max(total_gt, 1)
    motp      = iou_sum / max(total_tp, 1)
    precision = total_tp / max(total_tp + total_fp, 1)
    recall    = total_tp / max(total_tp + total_fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-9)

    # IDF1（帧级简化版）
    # IDTP = sum of gt_idtp values (每 GT 轨迹与其最常匹配的 pred 轨迹的一致帧数)
    # 简化：IDTP = total_tp（已用匈牙利匹配，最优帧级分配）
    # IDFP = FP 帧数；IDFN = FN 帧数
    idtp = total_tp
    idfp = total_fp
    idfn = total_fn
    idf1 = 2 * idtp / max(2 * idtp + idfp + idfn, 1)

    # 轨迹级统计
    mostly_tracked = sum(1 for gid in gt_idtp
                         if gt_idtp[gid] / max(gt_idtp[gid] + gt_idfn[gid], 1) >= 0.8)
    mostly_lost    = sum(1 for gid in gt_idfn
                         if gt_idfn[gid] / max(gt_idtp.get(gid,0) + gt_idfn[gid], 1) >= 0.8)
    n_gt_tracks    = len(set(gid for boxes in gt.values() for gid, *_ in boxes))

    elapsed = time.time() - t0

    print("\n" + "=" * 55)
    print("           MOT 评估结果")
    print("=" * 55)
    print(f"  GT 总帧数       : {len(gt_frames)}")
    print(f"  GT 总目标数     : {total_gt}")
    print(f"  GT 轨迹数       : {n_gt_tracks}")
    print(f"  IoU 阈值        : {IOU_THRESH}")
    print("-" * 55)
    print(f"  TP              : {total_tp:>8}")
    print(f"  FP (误检)       : {total_fp:>8}")
    print(f"  FN (漏检)       : {total_fn:>8}")
    print(f"  ID Switches     : {total_idsw:>8}")
    print(f"  唯一 Track ID   : {len(unique_pred_ids):>8}")
    print("-" * 55)
    print(f"  MOTA            : {mota*100:>7.2f}%")
    print(f"  MOTP (均值 IoU) : {motp*100:>7.2f}%")
    print(f"  IDF1 (简化)     : {idf1*100:>7.2f}%")
    print(f"  Precision       : {precision*100:>7.2f}%")
    print(f"  Recall          : {recall*100:>7.2f}%")
    print(f"  F1              : {f1*100:>7.2f}%")
    print("-" * 55)
    print(f"  主要跟踪 (≥80%) : {mostly_tracked:>8} / {n_gt_tracks}")
    print(f"  主要丢失 (≥80%) : {mostly_lost:>8} / {n_gt_tracks}")
    print("=" * 55)
    print(f"  评估耗时: {elapsed:.1f}s")

    # ── ID Switch 明细 ────────────────────────────────────────────
    affected_gt_ids = sorted({e[1] for e in idsw_events})
    print("\n" + "=" * 55)
    print(f"  ID Switch 明细（共 {total_idsw} 次，涉及 {len(affected_gt_ids)} 条 GT 轨迹）")
    print("=" * 55)
    print(f"  涉及的 GT id: {affected_gt_ids}")
    print("-" * 55)
    print(f"  {'GT帧号':>8}  {'GT id':>8}  {'旧 pred_id':>10}  {'新 pred_id':>10}")
    for gt_frame, gt_id, old_pid, new_pid in idsw_events:
        print(f"  {gt_frame:>8}  {gt_id:>8}  {old_pid:>10}  {new_pid:>10}")


if __name__ == "__main__":
    evaluate()
