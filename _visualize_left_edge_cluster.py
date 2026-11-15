# -*- coding: utf-8 -*-
"""
可视化左边缘一批低覆盖率GT(40/124/127/131/374) + GT150(IDSW那次)，
每个目标裁剪窗口跟随GT位置动态移动，画GT真值框(蓝)+所有检测框(绿)+离GT最近的检测(红)。
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

OUT_DIR = r"D:\yolo26-tracking\visualizations_missing_gt\left_edge_cluster"
os.makedirs(OUT_DIR, exist_ok=True)
PAD = 220


def detect(video_frame):
    cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame - 1)
    ret, frame = cap.read()
    if not ret:
        return None, None
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
    return frame, dets


# (label, gt_frame, cx, cy, w, h, rot)  -- gt_frame转成video_frame=gt_frame+1
TARGETS = [
    ("GT40", 0, 75.75, 782.5, 66.6, 33.3, -27.7),
    ("GT40", 30, 45.75, 798.0, 58.4, 33.8, -28.3),
    ("GT40", 60, 24.0, 810.0, 58.4, 33.8, -27.7),
    ("GT40", 90, 15.4, 815.0, 51.3, 33.8, -26.6),
    ("GT40", 100, 12.5, 816.4, 47.9, 33.8, -29.5),
    ("GT40", 101, 16.7, 821.2, 33.8, 40.6, -79.1),
    ("GT40", 102, 8.8, 819.5, 33.8, 58.4, -38.9),
    ("GT40", 110, 4.7, 820.5, 33.8, 58.4, -27.4),
    ("GT40", 130, 6.0, 827.9, 33.8, 58.4, -27.4),
    ("GT40", 142, 4.5, 831.5, 33.8, 58.4, -27.4),
    ("GT124", 0, 25.7, 978.5, 28.0, 30.6, -11.0),
    ("GT124", 15, 18.2, 980.5, 28.0, 30.6, -11.0),
    ("GT124", 30, 12.5, 982.5, 28.0, 30.6, -13.6),
    ("GT124", 45, 8.3, 982.5, 22.9, 30.6, -11.2),
    ("GT124", 61, 6.6, 986.7, 19.6, 30.6, -11.2),
    ("GT127", 0, 16.8, 1033.0, 32.6, 22.6, -12.7),
    ("GT127", 6, 13.5, 1034.7, 32.6, 22.6, -12.7),
    ("GT127", 11, 10.5, 1035.8, 26.5, 22.6, -12.7),
    ("GT127", 16, 8.7, 1036.9, 22.9, 22.6, -12.7),
    ("GT127", 20, 7.9, 1038.8, 21.4, 22.6, -12.7),
    ("GT131", 0, 22.3, 1077.0, 27.2, 34.1, -2.9),
    ("GT131", 2, 15.3, 1079.0, 27.2, 34.1, -2.9),
    ("GT131", 4, 10.6, 1080.9, 23.0, 34.1, -2.9),
    ("GT131", 6, 6.4, 1084.6, 14.6, 33.5, -2.9),
    ("GT374", 1818, 4.8, 1191.3, 17.4, 32.6, -13.2),
    ("GT374", 1820, 7.0, 1190.3, 21.6, 32.6, -13.2),
    ("GT374", 1823, 12.8, 1188.5, 21.6, 32.6, -13.2),
    ("GT150", 626, 702.4, 74.5, 60.5, 33.4, -18.7),
    ("GT150", 628, 692.3, 77.7, 61.9, 32.4, -20.0),
    ("GT150", 630, 680.6, 82.0, 63.2, 33.8, -23.2),
    ("GT150", 632, 668.4, 85.0, 67.3, 34.9, -23.6),
    ("GT150", 634, 655.3, 91.4, 64.9, 36.7, -22.3),
]

for label, gt_frame, gcx, gcy, gw, gh, grot in TARGETS:
    video_frame = gt_frame + 1
    frame, dets = detect(video_frame)
    if frame is None:
        print(label, gt_frame, "读取失败")
        continue

    x1, y1 = max(0, int(gcx - PAD)), max(0, int(gcy - PAD))
    x2, y2 = int(gcx + PAD), int(gcy + PAD)
    crop = frame[y1:y2, x1:x2].copy()

    ranked = sorted(
        ((((d["xywhr"][0]-gcx)**2+(d["xywhr"][1]-gcy)**2)**0.5, d) for d in dets),
        key=lambda r: r[0]
    )
    nearest_d = ranked[0][0] if ranked else None
    second_d = ranked[1][0] if len(ranked) > 1 else None
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

    # GT真值框(蓝)
    grr = ((gcx - x1, gcy - y1), (gw, gh), grot)
    gbox = cv2.boxPoints(grr).astype(int)
    cv2.drawContours(crop, [gbox], 0, (255, 0, 0), 2)

    tag = f"{label} gt_frame={gt_frame} nearest={nearest_d:.0f}px" if nearest_d is not None else f"{label} gt_frame={gt_frame} 无检测"
    cv2.putText(crop, tag, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    out_path = os.path.join(OUT_DIR, f"{label}_gtf{gt_frame}.jpg")
    cv2.imwrite(out_path, crop)
    print(f"{label} gt_frame={gt_frame}: 最近={nearest_d} 次近={second_d} -> {out_path}")

cap.release()
print("完成")
