#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对16次IDSW事件，用GT真值位置(不依赖检测器，因为丢失期间很多帧检测器压根没输出框)
把"丢失区间"内均匀采样的若干帧车辆区域裁出来，供ReID判断：即使检测器认不出来，
这些帧车辆本身的外观是否还能被ReID识别成"同一辆车"。
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import xml.etree.ElementTree as ET
import json
import cv2
import numpy as np
from collections import defaultdict

VIDEO_PATH = r"E:\DJI\cvat\0111_D_stabilized.avi"
GT_XML = r"E:\DJI\cvat\GT(1).xml"
TRACK_RESULT = r"D:\yolo26-tracking\outputs\yolo26-obb_20260722_155902\track_result.txt"
OUT_DIR = r"D:\yolo26-tracking\outputs\idsw_reid_check\gap_frames"
os.makedirs(OUT_DIR, exist_ok=True)

IDSW_EVENTS = [
    (163, 138, 137, 165), (420, 46, 114, 210), (467, 136, 135, 216), (489, 46, 210, 221),
    (564, 136, 216, 235), (576, 150, 150, 238), (628, 136, 235, 248), (688, 150, 238, 256),
    (893, 198, 202, 286), (999, 198, 286, 307), (1120, 218, 234, 329), (1156, 218, 329, 338),
    (1444, 275, 310, 382), (1538, 275, 382, 390), (1649, 293, 334, 407), (1784, 313, 358, 426),
]
MAX_SAMPLES_PER_EVENT = 6


def parse_gt(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    gt = defaultdict(dict)
    for track in root.findall("track"):
        gid = int(track.get("id"))
        for box in track.findall("box"):
            if box.get("outside") == "1":
                continue
            frame = int(box.get("frame"))
            xtl, ytl = float(box.get("xtl")), float(box.get("ytl"))
            xbr, ybr = float(box.get("xbr")), float(box.get("ybr"))
            rot = float(box.get("rotation", 0))
            cx, cy = (xtl + xbr) / 2, (ytl + ybr) / 2
            gt[gid][frame] = (cx, cy, xbr - xtl, ybr - ytl, rot)
        for poly in track.findall("polygon"):
            if poly.get("outside") == "1":
                continue
            frame = int(poly.get("frame"))
            pts = np.array([[float(v) for v in p.split(",")] for p in poly.get("points").split(";")], dtype=np.float32)
            (cx, cy), (w, h), rot = cv2.minAreaRect(pts)
            gt[gid][frame] = (cx, cy, w, h, rot)
    return gt


def gt_pos_near(gt_track, gt_frame, max_search=20):
    if gt_frame in gt_track:
        return gt_track[gt_frame]
    for d in range(1, max_search + 1):
        if gt_frame - d in gt_track:
            return gt_track[gt_frame - d]
        if gt_frame + d in gt_track:
            return gt_track[gt_frame + d]
    return None


by_track = defaultdict(list)
with open(TRACK_RESULT, encoding="utf-8") as f:
    for line in f:
        p = line.strip().split(",")
        if len(p) < 6:
            continue
        frame, tid, cx, cy = int(p[0]), int(p[1]), float(p[2]), float(p[3])
        by_track[tid].append((frame, cx, cy))
for tid in by_track:
    by_track[tid].sort(key=lambda r: r[0])

print(f"解析GT: {GT_XML}")
gt = parse_gt(GT_XML)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"无法打开视频: {VIDEO_PATH}")


def crop_box(frame_img, cx, cy, w, h, angle_deg=0.0, pad=1.15):
    """按旋转角度先转正再裁剪，避免斜着的车被轴对齐裁剪切掉或混入背景/护栏。"""
    diag = float(np.hypot(w, h)) * pad
    r = int(diag / 2) + 2
    x1, y1 = max(0, int(cx - r)), max(0, int(cy - r))
    x2, y2 = min(frame_img.shape[1], int(cx + r)), min(frame_img.shape[0], int(cy + r))
    if x2 <= x1 or y2 <= y1:
        return None
    patch = frame_img[y1:y2, x1:x2]
    pcx, pcy = cx - x1, cy - y1
    M = cv2.getRotationMatrix2D((pcx, pcy), angle_deg, 1.0)
    rotated = cv2.warpAffine(patch, M, (patch.shape[1], patch.shape[0]), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
    cw, ch = w * pad, h * pad
    cx1, cy1 = max(0, int(pcx - cw / 2)), max(0, int(pcy - ch / 2))
    cx2, cy2 = min(rotated.shape[1], int(pcx + cw / 2)), min(rotated.shape[0], int(pcy + ch / 2))
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    return rotated[cy1:cy2, cx1:cx2]


manifest = []
for evt_idx, (gt_frame_switch, gt_id, old_pid, new_pid) in enumerate(IDSW_EVENTS):
    pred_frame = gt_frame_switch + 1
    old_hist = [r for r in by_track.get(old_pid, []) if r[0] <= pred_frame]
    old_last_frame = old_hist[-1][0] if old_hist else pred_frame - 1
    lost_start, lost_end = old_last_frame + 1, pred_frame - 1
    gap = lost_end - lost_start + 1
    if gap <= 0:
        print(f"evt{evt_idx:02d} gt#{gt_id}: 无丢失区间，跳过")
        continue

    n_samples = min(gap, MAX_SAMPLES_PER_EVENT)
    sample_frames = sorted(set(int(round(lost_start + k * (gap - 1) / max(n_samples - 1, 1)))
                                for k in range(n_samples)))

    gt_track = gt.get(gt_id, {})
    saved = []
    for sf in sample_frames:
        gt_frame_for_sf = sf - 1
        pos = gt_pos_near(gt_track, gt_frame_for_sf)
        if pos is None:
            continue
        gcx, gcy, gw, gh, grot = pos
        cap.set(cv2.CAP_PROP_POS_FRAMES, sf - 1)
        ret, frame_img = cap.read()
        if not ret:
            continue
        crop = crop_box(frame_img, gcx, gcy, gw, gh, grot)
        if crop is None:
            continue
        out_path = os.path.join(OUT_DIR, f"evt{evt_idx:02d}_gap_f{sf}_gt{gt_id}.jpg")
        cv2.imwrite(out_path, crop)
        saved.append(dict(frame=sf, path=out_path, pos_in_gap=(sf - lost_start) / max(gap - 1, 1)))

    manifest.append(dict(evt=evt_idx, gt_id=gt_id, old_pid=old_pid, new_pid=new_pid,
                          lost_start=lost_start, lost_end=lost_end, gap=gap, samples=saved))
    print(f"evt{evt_idx:02d} gt#{gt_id}: 丢失区间[{lost_start},{lost_end}] (共{gap}帧), "
          f"成功裁剪 {len(saved)}/{len(sample_frames)} 帧")

cap.release()

with open(os.path.join(OUT_DIR, "gap_manifest.json"), "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)
print(f"\n完成，裁剪图和gap_manifest.json保存在: {OUT_DIR}")
