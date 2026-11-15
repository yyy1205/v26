# -*- coding: utf-8 -*-
"""
第二个视频(DJI_20250302041036_0037_D.MP4)全部26次IDSW的可视化。
不重新跑检测，直接用已有的GT(mota.xml)和预测(cvat_eval.txt)画框对比。
每次切换事件输出切换帧前后各2帧(共5帧)，裁剪到GT框附近，GT画蓝色，
两个涉事pred_id分别画红/黄，其余pred框画绿色做背景参考。
"""
import os
import cv2
import numpy as np
import eval_mot
from eval_mot import parse_gt, parse_pred

VIDEO_PATH = r"E:\DJI\sgybw\video2\DJI_20250302041036_0037_D.MP4"
GT_XML = r"D:\ultralytics-8.1.41\outputs\v8e53-2condence_20260521_150313\mota.xml"
PRED_TXT = r"D:\yolo26-tracking\outputs\yolo26-obb_20260804_175700\cvat_eval.txt"
OUT_DIR = r"D:\yolo26-tracking\visualizations_missing_gt\video2_all_idsw_after_nmmfix"
os.makedirs(OUT_DIR, exist_ok=True)

eval_mot.GT_XML = GT_XML
eval_mot.PRED_TXT = PRED_TXT
eval_mot.FRAME_W, eval_mot.FRAME_H = 5120, 2700

idsw_events = eval_mot.evaluate()
print(f"\n共 {len(idsw_events)} 次 IDSW，开始可视化...")

gt = parse_gt(GT_XML)
pred = parse_pred(PRED_TXT)

cap = cv2.VideoCapture(VIDEO_PATH)
PAD = 300

for k, (gt_frame, gt_id, old_pid, new_pid) in enumerate(idsw_events):
    sub_dir = os.path.join(OUT_DIR, f"{k:02d}_GT{gt_id}_f{gt_frame}_{old_pid}to{new_pid}")
    os.makedirs(sub_dir, exist_ok=True)

    for off in range(-2, 3):
        gtf = gt_frame + off
        gt_boxes = [b for b in gt.get(gtf, []) if b[0] == gt_id]
        if not gt_boxes:
            continue
        _, gcx, gcy, gw, gh, grot = gt_boxes[0]

        pred_frame = gtf + 1
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
            if tid == old_pid:
                color = (0, 0, 255)      # 红：旧id
            elif tid == new_pid:
                color = (0, 255, 255)    # 黄：新id
            else:
                color = (0, 255, 0)      # 绿：其他背景车辆
            box = (pts - np.array([x1, y1])).astype(int)
            cv2.drawContours(crop, [box], 0, color, 2)
            cv2.putText(crop, f"id{tid}", (int(pcx - x1 - pw / 2), int(pcy - y1 - ph / 2) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        grr = ((gcx - x1, gcy - y1), (gw, gh), grot)
        gbox = cv2.boxPoints(grr).astype(int)
        cv2.drawContours(crop, [gbox], 0, (255, 0, 0), 2)
        tag = "SWITCH" if off == 0 else f"{off:+d}"
        cv2.putText(crop, f"GT{gt_id} gt_frame={gtf} [{tag}] {old_pid}->{new_pid}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imwrite(os.path.join(sub_dir, f"gtf{gtf:04d}_{tag}.jpg"), crop)

    print(f"[{k+1}/{len(idsw_events)}] GT{gt_id} f{gt_frame} {old_pid}->{new_pid} 完成")

cap.release()
print(f"\n全部完成，输出目录: {OUT_DIR}")
