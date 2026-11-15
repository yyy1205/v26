# -*- coding: utf-8 -*-
"""
检查video2上GT283(id580<->id594交替)的问题帧：低置信度阈值+不做NMM/NMS合并，
直接看原始检测框，判断"长框"在"短框"那几帧是否也被检测出来了(只是被置信度/NMS抑制)，
还是检测器这一帧压根就只输出了短框(没有长框候选)。
"""
import runpy
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

ns = runpy.run_path(r"D:\yolo26-tracking\_repro_base.py", run_name="repro_base")

import cv2
import numpy as np

load_roi_file = ns["load_roi_file"]
multi_roi_mask = ns["multi_roi_mask"]
sliced_predict_obb_direct = ns["sliced_predict_obb_direct"]
yolo_model = ns["yolo_model"]

VIDEO_PATH = r"E:\DJI\sgybw\video2\DJI_20250302041036_0037_D.MP4"
ROI_FILE = r"E:\DJI\sgybw\video2\DJI_20250302041036_0037_D_roi.json"

cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
ret, first_frame = cap.read()
roi_polys_modes, pixel_length, real_length = load_roi_file(ROI_FILE)
roi_mask = multi_roi_mask(roi_polys_modes, first_frame.shape)

CX, CY = 4993, 1717  # id580/594的位置(取id580最后一帧的中心)
LOW_CONF = 0.05        # 尽量放低阈值，看有没有被压掉的候选

# pred_frame范围: 2782(580,长框)~2790(594,短框)
for video_frame in range(2782, 2792):
    cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame - 1)
    ret, frame = cap.read()
    if not ret:
        print(video_frame, "读帧失败")
        continue

    raw_dets = sliced_predict_obb_direct(frame, yolo_model, roi_mask=roi_mask, conf_thr=LOW_CONF)

    nearby = []
    for d in raw_dets:
        cx, cy, w, h, rot_rad = d['xywhr']
        if abs(cx - CX) < 120 and abs(cy - CY) < 120:
            nearby.append((cx, cy, w, h, np.rad2deg(rot_rad), d['conf'], d['cls'], d['slice']))

    nearby.sort(key=lambda x: -x[5])
    print(f"video_frame={video_frame}: 附近原始候选框(conf>={LOW_CONF})共{len(nearby)}个")
    for cx, cy, w, h, rot, conf, cls, sl in nearby:
        print(f"    ({cx:.0f},{cy:.0f}) {w:.0f}x{h:.0f} rot={rot:.0f} conf={conf:.3f} cls={cls} slice={sl}")
