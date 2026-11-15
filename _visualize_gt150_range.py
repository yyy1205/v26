# -*- coding: utf-8 -*-
"""
GT150 IDSW事件(630帧, 151->230)前后50帧(gt_frame 580~680)，逐帧画GT真值框(蓝)+全部检测框(绿)+
离GT最近的检测(红)，裁剪窗口跟随GT位置动态移动。
"""
import runpy
import os
import json

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

OUT_DIR = r"D:\yolo26-tracking\visualizations_missing_gt\GT150_580_680"
os.makedirs(OUT_DIR, exist_ok=True)
PAD = 220

with open(r"D:\yolo26-tracking\_gt150_580_680.json") as f:
    gt_rows = json.load(f)

t0 = __import__("time").time()
for gt_frame, gcx, gcy, gw, gh, grot in gt_rows:
    video_frame = gt_frame + 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame - 1)
    ret, frame = cap.read()
    if not ret:
        print(gt_frame, "读取失败")
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

    x1, y1 = max(0, int(gcx - PAD)), max(0, int(gcy - PAD))
    x2, y2 = int(gcx + PAD), int(gcy + PAD)
    crop = frame[y1:y2, x1:x2].copy()

    ranked = sorted(
        ((((d["xywhr"][0]-gcx)**2+(d["xywhr"][1]-gcy)**2)**0.5, d) for d in dets),
        key=lambda r: r[0]
    )
    nearest_d = ranked[0][0] if ranked else None
    nearest_det = ranked[0][1] if ranked else None

    for d in dets:
        dx, dy, dw, dh, dr = d["xywhr"]
        if not (x1 - 50 <= dx <= x2 + 50 and y1 - 50 <= dy <= y2 + 50):
            continue
        is_nearest = d is nearest_det
        color = (0, 0, 255) if is_nearest else (0, 255, 0)
        rr = ((dx - x1, dy - y1), (dw, dh), np.rad2deg(dr))
        box = cv2.boxPoints(rr).astype(int)
        cv2.drawContours(crop, [box], 0, color, 2)
        cv2.putText(crop, f"{d['conf']:.2f}", (int(dx-x1-dw/2), int(dy-y1-dh/2)-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    grr = ((gcx - x1, gcy - y1), (gw, gh), grot)
    gbox = cv2.boxPoints(grr).astype(int)
    cv2.drawContours(crop, [gbox], 0, (255, 0, 0), 2)

    marker = " <== IDSW事件帧" if gt_frame == 630 else (" <== 检测缺失帧" if gt_frame == 626 else "")
    tag = f"gt_frame={gt_frame} nearest={nearest_d:.0f}px{marker}" if nearest_d is not None else f"gt_frame={gt_frame} 无检测{marker}"
    cv2.putText(crop, tag, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    out_path = os.path.join(OUT_DIR, f"gtf{gt_frame:04d}.jpg")
    cv2.imwrite(out_path, crop)
    if gt_frame % 10 == 0:
        print(f"gt_frame={gt_frame} 完成, nearest={nearest_d}, 耗时{__import__('time').time()-t0:.0f}s")

cap.release()
print("完成，全部保存到", OUT_DIR)
