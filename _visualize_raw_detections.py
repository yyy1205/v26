# -*- coding: utf-8 -*-
"""
不看GT，只跑生产管线的检测逻辑，把每个问题位置多帧的实际检测框画出来存图。
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

OUT_DIR = r"D:\yolo26-tracking\visualizations_missing_gt"
os.makedirs(OUT_DIR, exist_ok=True)

# (location label, 中心大致位置cx,cy, 要看的一批video_frame(1-based))
LOCATIONS = [
    ("locA", 1645, 1195, [253, 255, 257, 528, 530, 532, 756, 762, 768, 774, 779]),
    ("locB", 2750, 815, [422, 425, 428, 431, 433, 1540, 1543, 1546, 1549, 1550]),
]

PAD = 300  # 裁剪窗口半径

for loc_label, cx, cy, frames in LOCATIONS:
    for video_frame in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame - 1)
        ret, frame = cap.read()
        if not ret:
            print(f"{loc_label} 帧{video_frame} 读取失败")
            continue

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
        dets = _full_detection_pipeline(raw_dets, nms_iou_thr=IOU_NMS_THR)

        x1, y1 = max(0, cx - PAD), max(0, cy - PAD)
        x2, y2 = cx + PAD, cy + PAD
        crop = frame[y1:y2, x1:x2].copy()

        n_drawn = 0
        for d in dets:
            dx, dy, dw, dh, dr = d["xywhr"]
            if not (x1 - 50 <= dx <= x2 + 50 and y1 - 50 <= dy <= y2 + 50):
                continue
            rr = ((dx - x1, dy - y1), (dw, dh), np.rad2deg(dr))
            box = cv2.boxPoints(rr).astype(int)
            cv2.drawContours(crop, [box], 0, (0, 255, 0), 2)
            label = f"{d['conf']:.2f}"
            cv2.putText(crop, label, (int(dx - x1 - dw / 2), int(dy - y1 - dh / 2) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            n_drawn += 1

        out_path = os.path.join(OUT_DIR, f"{loc_label}_rawdet_f{video_frame}.jpg")
        cv2.imwrite(out_path, crop)
        print(f"{loc_label} 帧{video_frame}: 窗口内画出{n_drawn}个检测框 -> {out_path}")

cap.release()
print("完成")
