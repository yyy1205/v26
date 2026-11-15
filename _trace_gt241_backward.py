# -*- coding: utf-8 -*-
"""
把GT241对应车辆的链式追踪继续往前推，看这辆车最早能追溯到哪一帧，
以此验证"GT标注是不是起晚了/车辆早就存在于画面中"这个猜测。
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
sliced_predict_obb_direct = ns["sliced_predict_obb_direct"]
_full_detection_pipeline = ns["_full_detection_pipeline"]
PRE_MASK_DETECTION = ns["PRE_MASK_DETECTION"]
PRE_MASK_MODE = ns["PRE_MASK_MODE"]
PRE_MASK_FEATHER = ns["PRE_MASK_FEATHER"]
PRE_MASK_BLUR_KSIZE = ns["PRE_MASK_BLUR_KSIZE"]
apply_blurred_background_fill = ns["apply_blurred_background_fill"]
IOU_NMS_THR = 0.2
yolo_model = ns["yolo_model"]

cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
ret, first_frame = cap.read()
roi_polys_modes, pixel_length, real_length = load_roi_file(ROI_FILE)
roi_mask = multi_roi_mask(roi_polys_modes, first_frame.shape)
H_IMG, W_IMG = first_frame.shape[:2]

MAX_STEP = 50
START_FRAME = 740
START_POS = (1645.4, 1194.5)  # 帧740链式追踪到的位置(取自上次结果附近，重新算一次锚点)


def detect(video_frame):
    cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame - 1)
    ret, frame = cap.read()
    if not ret:
        return None
    proc_frame_for_sahi = frame
    if PRE_MASK_DETECTION and roi_mask is not None and PRE_MASK_MODE == "blur":
        proc_frame_for_sahi = apply_blurred_background_fill(
            frame, roi_mask, feather=PRE_MASK_FEATHER, blur_ksize=PRE_MASK_BLUR_KSIZE
        )
    raw_dets = sliced_predict_obb_direct(
        proc_frame_for_sahi, yolo_model, roi_mask, slice_h=960, slice_w=960, overlap_h=0.3, overlap_w=0.3, conf_thr=0.3,
    )
    raw_dets += sliced_predict_obb_direct(
        proc_frame_for_sahi, yolo_model, roi_mask, slice_h=1280, slice_w=1280, overlap_h=0.3, overlap_w=0.3, conf_thr=0.3,
    )
    return _full_detection_pipeline(raw_dets, nms_iou_thr=IOU_NMS_THR)


pos = START_POS
video_frame = START_FRAME
trace = [(video_frame, pos)]

while True:
    video_frame -= 1
    dets = detect(video_frame)
    if not dets:
        print(f"帧{video_frame}: 无检测，停止")
        break
    ranked = sorted(
        ((((d["xywhr"][0]-pos[0])**2+(d["xywhr"][1]-pos[1])**2)**0.5, d["xywhr"][0], d["xywhr"][1]) for d in dets),
        key=lambda r: r[0]
    )
    best = ranked[0]
    if best[0] > MAX_STEP:
        print(f"帧{video_frame}: 最近候选距离{best[0]:.1f}px 超过阈值，追踪中断（这就是车辆真正开始出现的时间点附近）")
        break
    pos = (best[1], best[2])
    trace.append((video_frame, pos))
    d_left, d_right, d_top, d_bottom = pos[0], W_IMG - pos[0], pos[1], H_IMG - pos[1]
    if video_frame % 20 == 0 or min(d_left, d_right, d_top, d_bottom) < 100:
        print(f"帧{video_frame}: pos=({pos[0]:.1f},{pos[1]:.1f})  距边缘: left={d_left:.0f} right={d_right:.0f} top={d_top:.0f} bottom={d_bottom:.0f}")
    if min(d_left, d_right, d_top, d_bottom) < 30:
        print(f"帧{video_frame}: 已经贴近画面边缘了，这应该就是车辆真正入画的地方")
        break
    if video_frame < 400:
        print("已经追到帧400还没到边缘，停止(避免无限跑下去)")
        break

cap.release()
print(f"\n最终追溯范围: 帧{trace[-1][0]} ~ 帧{START_FRAME}，共{len(trace)}帧连续存在")
print(f"而GT241标注是从帧757才开始的——中间差了{757-trace[-1][0]}帧车辆已经存在但没有GT标注")
