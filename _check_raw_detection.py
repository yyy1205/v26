# -*- coding: utf-8 -*-
"""
用生产管线里完全一样的检测逻辑(sliced_predict_obb_direct + _full_detection_pipeline)，
单独对几个问题帧跑检测(不经过跟踪器)，直接看原始检测框，判断:
  - 卡点A(GT241等)：框宽度异常放大是不是在检测这一步就已经发生了
  - 卡点B(GT202/339)：检测器是不是真的完全没输出框
"""
import runpy
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

ns = runpy.run_path(r"D:\yolo26-tracking\_repro_base.py", run_name="repro_base")

import cv2
import numpy as np

VIDEO_PATH = ns["VIDEO_PATH"]
ROI_FILE = ns["ROI_FILE"]
load_roi_file = ns["load_roi_file"]
multi_roi_mask = ns["multi_roi_mask"]
apply_clahe = ns["apply_clahe"]
gamma_transform = ns["gamma_transform"]
apply_blurred_background_fill = ns["apply_blurred_background_fill"]
sliced_predict_obb_direct = ns["sliced_predict_obb_direct"]
_full_detection_pipeline = ns["_full_detection_pipeline"]
PRE_MASK_DETECTION = ns["PRE_MASK_DETECTION"]
PRE_MASK_MODE = ns["PRE_MASK_MODE"]
PRE_MASK_FEATHER = ns["PRE_MASK_FEATHER"]
PRE_MASK_BLUR_KSIZE = ns["PRE_MASK_BLUR_KSIZE"]
IOU_NMS_THR = 0.2
yolo_model = ns["yolo_model"]

cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
ret, first_frame = cap.read()
roi_polys_modes, pixel_length, real_length = load_roi_file(ROI_FILE)
roi_mask = multi_roi_mask(roi_polys_modes, first_frame.shape)

# 这次跑不做画面增强(跟当前的"关闭增强"实验保持一致，因为要查的是这个实验里的漏检)
USE_ENHANCE = False

targets = [
    ("locA_GT241_f776", 777, 1653.7, 1187.9),   # (label, video_frame(1-based), cx, cy)  用pred_frame对应video_frame
    ("locA_GT175_f253", 254, 1640.5, 1204.0),
    ("locA_GT214_f528", 529, 1638.5, 1198.2),
    ("locB_GT202_f422", 423, 2804.8, 818.2),
    ("locB_GT339_f1540", 1541, 2651.0, 818.2),
]

for label, video_frame, cx, cy in targets:
    cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame - 1)
    ret, frame = cap.read()
    if not ret:
        print(label, "读帧失败"); continue

    proc_frame = frame
    if USE_ENHANCE:
        proc_frame = apply_clahe(proc_frame)
        proc_frame = gamma_transform(proc_frame, gamma=1.2)

    proc_frame_for_sahi = proc_frame
    if PRE_MASK_DETECTION and roi_mask is not None and PRE_MASK_MODE == "blur":
        proc_frame_for_sahi = apply_blurred_background_fill(
            proc_frame, roi_mask, feather=PRE_MASK_FEATHER, blur_ksize=PRE_MASK_BLUR_KSIZE
        )

    raw_dets = sliced_predict_obb_direct(
        proc_frame_for_sahi, yolo_model, roi_mask, slice_h=960, slice_w=960, overlap_h=0.3, overlap_w=0.3, conf_thr=0.3,
    )
    raw_dets += sliced_predict_obb_direct(
        proc_frame_for_sahi, yolo_model, roi_mask, slice_h=1280, slice_w=1280, overlap_h=0.3, overlap_w=0.3, conf_thr=0.3,
    )
    dets_before_nms = list(raw_dets)
    dets = _full_detection_pipeline(raw_dets, nms_iou_thr=IOU_NMS_THR)

    print(f"\n=== {label} (video_frame={video_frame}, GT附近pos=({cx:.0f},{cy:.0f})) ===")
    print(f"  NMS前原始检测总数: {len(dets_before_nms)}, NMS后: {len(dets)}")
    near_before = [d for d in dets_before_nms if abs(d["xywhr"][0]-cx) < 150 and abs(d["xywhr"][1]-cy) < 150]
    near_after = [d for d in dets if abs(d["xywhr"][0]-cx) < 150 and abs(d["xywhr"][1]-cy) < 150]
    print(f"  附近150px内 NMS前: {len(near_before)}个, NMS后: {len(near_after)}个")
    for d in near_before:
        x, y, w, h, r = d["xywhr"]
        print(f"    [NMS前] pos=({x:.1f},{y:.1f}) w={w:.1f} h={h:.1f} rot={np.rad2deg(r):.1f} conf={d['conf']:.3f} cls={d['cls']}")
    for d in near_after:
        x, y, w, h, r = d["xywhr"]
        print(f"    [NMS后] pos=({x:.1f},{y:.1f}) w={w:.1f} h={h:.1f} rot={np.rad2deg(r):.1f} conf={d['conf']:.3f} cls={d['cls']}")

cap.release()
print("\n完成")
