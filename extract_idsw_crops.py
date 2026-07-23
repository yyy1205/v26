#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为16次IDSW事件抽取三类裁剪图，供ReID模型验证:
  old : old_pid 丢失前最后一帧的车辆裁剪(正样本锚点)
  new : new_pid 重新出现那帧的车辆裁剪(正样本，理论上应该和old是同一辆车)
  decoy: 同一帧里，另一辆不相关车辆的裁剪(负样本，用于检验ReID能不能把它和old分开)
输出到 outputs/idsw_reid_check/ 下，文件名形如 evt00_old_gt138.jpg / evt00_new_gt138.jpg / evt00_decoy_gt138.jpg
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from collections import defaultdict
import cv2
import numpy as np

VIDEO_PATH = r"E:\DJI\cvat\0111_D_stabilized.avi"
TRACK_RESULT = r"D:\yolo26-tracking\outputs\yolo26-obb_20260722_155902\track_result.txt"
OUT_DIR = r"D:\yolo26-tracking\outputs\idsw_reid_check"
os.makedirs(OUT_DIR, exist_ok=True)

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
        if len(p) < 7:
            continue
        frame, tid = int(p[0]), int(p[1])
        cx, cy, w, h, ang = float(p[2]), float(p[3]), float(p[4]), float(p[5]), float(p[6])
        by_track[tid].append((frame, cx, cy, w, h, ang))
for tid in by_track:
    by_track[tid].sort(key=lambda r: r[0])

# frame -> [(tid, cx,cy,w,h,ang)]
by_frame = defaultdict(list)
for tid, recs in by_track.items():
    for f, cx, cy, w, h, ang in recs:
        by_frame[f].append((tid, cx, cy, w, h, ang))


def crop_box(frame_img, cx, cy, w, h, angle_rad=0.0, pad=1.3):
    """按旋转角度先转正再裁剪，避免斜着的车被轴对齐裁剪切掉或混入背景。"""
    diag = float(np.hypot(w, h)) * pad
    r = int(diag / 2) + 2
    x1, y1 = max(0, int(cx - r)), max(0, int(cy - r))
    x2, y2 = min(frame_img.shape[1], int(cx + r)), min(frame_img.shape[0], int(cy + r))
    if x2 <= x1 or y2 <= y1:
        return None
    patch = frame_img[y1:y2, x1:x2]
    pcx, pcy = cx - x1, cy - y1
    angle_deg = np.rad2deg(angle_rad)
    M = cv2.getRotationMatrix2D((pcx, pcy), angle_deg, 1.0)
    rotated = cv2.warpAffine(patch, M, (patch.shape[1], patch.shape[0]), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
    cw, ch = w * pad, h * pad
    cx1, cy1 = max(0, int(pcx - cw / 2)), max(0, int(pcy - ch / 2))
    cx2, cy2 = min(rotated.shape[1], int(pcx + cw / 2)), min(rotated.shape[0], int(pcy + ch / 2))
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    return rotated[cy1:cy2, cx1:cx2]


def read_frame(cap, frame_no):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no - 1)
    ret, frame = cap.read()
    return frame if ret else None


cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"无法打开视频: {VIDEO_PATH}")

manifest = []
for evt_idx, (gt_frame, gt_id, old_pid, new_pid) in enumerate(IDSW_EVENTS):
    pred_frame = gt_frame + 1
    old_hist = [r for r in by_track.get(old_pid, []) if r[0] <= pred_frame]
    new_hist = [r for r in by_track.get(new_pid, []) if r[0] >= pred_frame]
    if not old_hist or not new_hist:
        print(f"evt{evt_idx:02d} gt#{gt_id}: 数据不足，跳过")
        continue
    old_f, old_cx, old_cy, old_w, old_h, old_ang = old_hist[-1]
    new_f, new_cx, new_cy, new_w, new_h, new_ang = new_hist[0]

    old_frame_img = read_frame(cap, old_f)
    new_frame_img = read_frame(cap, new_f)
    if old_frame_img is None or new_frame_img is None:
        print(f"evt{evt_idx:02d} gt#{gt_id}: 读帧失败，跳过")
        continue

    old_crop = crop_box(old_frame_img, old_cx, old_cy, old_w, old_h, old_ang)
    new_crop = crop_box(new_frame_img, new_cx, new_cy, new_w, new_h, new_ang)

    # decoy: old_f这一帧里，离old_pid最近的另一辆车(排除自己)
    decoy_crop = None
    candidates = [(tid, cx, cy, w, h, ang) for tid, cx, cy, w, h, ang in by_frame.get(old_f, []) if tid != old_pid]
    if candidates:
        candidates.sort(key=lambda c: (c[1] - old_cx) ** 2 + (c[2] - old_cy) ** 2)
        dtid, dcx, dcy, dw, dh, dang = candidates[0]
        decoy_crop = crop_box(old_frame_img, dcx, dcy, dw, dh, dang)

    if old_crop is not None:
        p = os.path.join(OUT_DIR, f"evt{evt_idx:02d}_old_gt{gt_id}.jpg")
        cv2.imwrite(p, old_crop)
    if new_crop is not None:
        p2 = os.path.join(OUT_DIR, f"evt{evt_idx:02d}_new_gt{gt_id}.jpg")
        cv2.imwrite(p2, new_crop)
    if decoy_crop is not None:
        p3 = os.path.join(OUT_DIR, f"evt{evt_idx:02d}_decoy_gt{gt_id}.jpg")
        cv2.imwrite(p3, decoy_crop)

    manifest.append(dict(evt=evt_idx, gt_id=gt_id, old_pid=old_pid, new_pid=new_pid,
                          old_frame=old_f, new_frame=new_f,
                          has_old=old_crop is not None, has_new=new_crop is not None,
                          has_decoy=decoy_crop is not None))
    print(f"evt{evt_idx:02d} gt#{gt_id}: old_f={old_f} new_f={new_f} "
          f"old_crop={'ok' if old_crop is not None else 'MISS'} "
          f"new_crop={'ok' if new_crop is not None else 'MISS'} "
          f"decoy={'ok' if decoy_crop is not None else 'none'}")

cap.release()

import json
with open(os.path.join(OUT_DIR, "manifest.json"), "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)
print(f"\n完成，裁剪图和manifest.json保存在: {OUT_DIR}")
