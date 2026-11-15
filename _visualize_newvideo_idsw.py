# -*- coding: utf-8 -*-
"""
新视频(DJI_20250302041036_0037_D.MP4)上GT283/GT346两次反复震荡IDSW的可视化。
不用重新跑检测——直接用已有的GT(mota.xml)和预测(cvat_eval.txt)画框对比。
"""
import os
import cv2
import numpy as np
from eval_mot import parse_gt, parse_pred

VIDEO_PATH = r"E:\DJI\sgybw\video2\DJI_20250302041036_0037_D.MP4"
GT_XML = r"D:\ultralytics-8.1.41\outputs\v8e53-2condence_20260521_150313\mota.xml"
PRED_TXT = r"D:\yolo26-tracking\outputs\yolo26-obb_20260724_212206\cvat_eval.txt"
OUT_DIR = r"D:\yolo26-tracking\visualizations_missing_gt\newvideo_idsw"
os.makedirs(OUT_DIR, exist_ok=True)

gt = parse_gt(GT_XML)
pred = parse_pred(PRED_TXT)

cap = cv2.VideoCapture(VIDEO_PATH)
PAD = 260

TARGETS = [
    ("GT283", 283, range(2778, 2818), [580, 594]),
    ("GT346", 346, range(3568, 3630), [458, 755, 799]),
]

for label, gid, gt_frame_range, highlight_ids in TARGETS:
    sub_dir = os.path.join(OUT_DIR, label)
    os.makedirs(sub_dir, exist_ok=True)
    for gt_frame in gt_frame_range:
        gt_boxes = [b for b in gt.get(gt_frame, []) if b[0] == gid]
        if not gt_boxes:
            continue
        _, gcx, gcy, gw, gh, grot = gt_boxes[0]

        pred_frame = gt_frame + 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, pred_frame - 1)
        ret, frame = cap.read()
        if not ret:
            continue

        x1, y1 = max(0, int(gcx - PAD)), max(0, int(gcy - PAD))
        x2, y2 = int(gcx + PAD), int(gcy + PAD)
        crop = frame[y1:y2, x1:x2].copy()

        pred_list = pred.get(pred_frame, [])
        for tid, pts in pred_list:
            (pcx, pcy), (pw, ph), prot = cv2.minAreaRect(pts)
            if not (x1 - 50 <= pcx <= x2 + 50 and y1 - 50 <= pcy <= y2 + 50):
                continue
            is_highlight = tid in highlight_ids
            color = (0, 0, 255) if is_highlight else (0, 255, 0)
            box = (pts - np.array([x1, y1])).astype(int)
            cv2.drawContours(crop, [box], 0, color, 2)
            cv2.putText(crop, f"id{tid}", (int(pcx - x1 - pw / 2), int(pcy - y1 - ph / 2) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        grr = ((gcx - x1, gcy - y1), (gw, gh), grot)
        gbox = cv2.boxPoints(grr).astype(int)
        cv2.drawContours(crop, [gbox], 0, (255, 0, 0), 2)
        cv2.putText(crop, f"{label} gt_frame={gt_frame}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imwrite(os.path.join(sub_dir, f"gtf{gt_frame:04d}.jpg"), crop)
    print(f"{label} 完成，保存到 {sub_dir}")

cap.release()
print("完成")
