# -*- coding: utf-8 -*-
"""
对10条低覆盖率GT(175/202/214/241/339/374/124/127/40/131)逐个:
  1. 从GT标注的第一帧/最后一帧为锚点，用链式帧间最近邻分别向前/向后外推，
     找到这辆车"真正开始出现"和"真正消失"的帧(不受GT标注范围限制)
  2. 把外推覆盖到的全部帧都画出来(GT真值蓝框，全部检测绿框，链式识别出的目标红框)
"""
import runpy
import os
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

ns = runpy.run_path(r"D:\yolo26-tracking\_repro_base.py", run_name="repro_base")

import cv2
import numpy as np
from eval_mot import parse_gt

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

OUT_ROOT = r"D:\yolo26-tracking\visualizations_missing_gt\all_ids"
os.makedirs(OUT_ROOT, exist_ok=True)

GT_XML = r"E:\DJI\cvat\GT(1).xml"
gt = parse_gt(GT_XML)

TARGET_IDS = [175, 202, 214, 241, 339, 374, 124, 127, 40, 131]
MAX_STEP = 60          # 帧间链式匹配最大允许位移(px)
MAX_EXTEND = 120        # 每个方向最多外推多少帧(避免跑飞/跑太久)
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


results_summary = []
t_start = time.time()

for gid in TARGET_IDS:
    gt_rows = sorted((frame, cx, cy, w, h, rot) for frame, boxes in gt.items() for g2, cx, cy, w, h, rot in boxes if g2 == gid)
    if not gt_rows:
        print(f"GT{gid}: 没有GT数据")
        continue
    gt_by_frame = {r[0]: r[1:] for r in gt_rows}
    first_gt_frame = gt_rows[0][0]
    last_gt_frame = gt_rows[-1][0]

    out_dir = os.path.join(OUT_ROOT, f"GT{gid}")
    os.makedirs(out_dir, exist_ok=True)

    # 缓存检测结果，避免同一帧算两次
    det_cache = {}

    def get_dets(video_frame):
        if video_frame not in det_cache:
            det_cache[video_frame] = detect(video_frame)
        return det_cache[video_frame]

    # 锚点：GT标注第一帧的真实位置(video_frame = gt_frame+1)
    anchor_frame = first_gt_frame + 1
    anchor_cx, anchor_cy = gt_by_frame[first_gt_frame][0], gt_by_frame[first_gt_frame][1]
    frame_img, dets = get_dets(anchor_frame)
    identified = {anchor_frame: (anchor_cx, anchor_cy)}

    # 向后外推(从GT最后一帧继续往后找，直到真的消失)
    last_anchor_frame = last_gt_frame + 1
    last_anchor_pos = (gt_by_frame[last_gt_frame][0], gt_by_frame[last_gt_frame][1])
    identified[last_anchor_frame] = last_anchor_pos

    vf = last_anchor_frame
    end_reason = "达到最大外推帧数"
    for _ in range(MAX_EXTEND):
        vf += 1
        _, dets = get_dets(vf)
        if not dets:
            end_reason = f"帧{vf}读取失败"
            break
        prev_pos = identified[vf - 1]
        ranked = sorted(((((d["xywhr"][0]-prev_pos[0])**2+(d["xywhr"][1]-prev_pos[1])**2)**0.5, d["xywhr"][0], d["xywhr"][1]) for d in dets), key=lambda r: r[0])
        best = ranked[0]
        if best[0] > MAX_STEP:
            end_reason = f"帧{vf}: 最近候选{best[0]:.0f}px超出阈值，判定车辆在这附近真正消失"
            break
        identified[vf] = (best[1], best[2])
    else:
        pass
    true_end_frame = max(identified.keys())

    # 向前外推(从GT第一帧继续往前找，直到真正入画)
    vf = anchor_frame
    start_reason = "达到最大外推帧数"
    for _ in range(MAX_EXTEND):
        vf -= 1
        if vf < 1:
            start_reason = "已到视频开头"
            break
        _, dets = get_dets(vf)
        if not dets:
            start_reason = f"帧{vf}读取失败"
            break
        prev_pos = identified[vf + 1]
        ranked = sorted(((((d["xywhr"][0]-prev_pos[0])**2+(d["xywhr"][1]-prev_pos[1])**2)**0.5, d["xywhr"][0], d["xywhr"][1]) for d in dets), key=lambda r: r[0])
        best = ranked[0]
        if best[0] > MAX_STEP:
            start_reason = f"帧{vf}: 最近候选{best[0]:.0f}px超出阈值，判定车辆在这附近真正入画"
            break
        identified[vf] = (best[1], best[2])
    true_start_frame = min(identified.keys())

    print(f"\n=== GT{gid} ===  GT标注范围: video_frame[{anchor_frame},{last_anchor_frame}] (gt_frame[{first_gt_frame},{last_gt_frame}])")
    print(f"  真实存在范围(链式追踪): video_frame[{true_start_frame},{true_end_frame}]  (共{true_end_frame-true_start_frame+1}帧)")
    print(f"  起点判定: {start_reason}")
    print(f"  终点判定: {end_reason}")
    results_summary.append((gid, anchor_frame, last_anchor_frame, true_start_frame, true_end_frame, start_reason, end_reason))

    # 画图：整个true_start_frame ~ true_end_frame范围
    for vf in range(true_start_frame, true_end_frame + 1):
        frame_img, dets = get_dets(vf)
        if frame_img is None:
            continue
        pos = identified.get(vf)
        cx, cy = pos if pos else (anchor_cx, anchor_cy)
        x1, y1 = max(0, int(cx - PAD)), max(0, int(cy - PAD))
        x2, y2 = int(cx + PAD), int(cy + PAD)
        crop = frame_img[y1:y2, x1:x2].copy()

        for d in dets:
            dx, dy, dw, dh, dr = d["xywhr"]
            if not (x1 - 50 <= dx <= x2 + 50 and y1 - 50 <= dy <= y2 + 50):
                continue
            is_id = pos is not None and abs(dx - pos[0]) < 1 and abs(dy - pos[1]) < 1
            color = (0, 0, 255) if is_id else (0, 255, 0)
            rr = ((dx - x1, dy - y1), (dw, dh), np.rad2deg(dr))
            box = cv2.boxPoints(rr).astype(int)
            cv2.drawContours(crop, [box], 0, color, 2)
            cv2.putText(crop, f"{d['conf']:.2f}", (int(dx-x1-dw/2), int(dy-y1-dh/2)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        gt_frame = vf - 1
        if gt_frame in gt_by_frame:
            gcx, gcy, gw, gh, grot = gt_by_frame[gt_frame]
            grr = ((gcx - x1, gcy - y1), (gw, gh), grot)
            gbox = cv2.boxPoints(grr).astype(int)
            cv2.drawContours(crop, [gbox], 0, (255, 0, 0), 2)

        tag = f"GT{gid} video_frame={vf}" + (" [GT标注范围内]" if first_gt_frame <= gt_frame <= last_gt_frame else " [GT范围外/外推]")
        cv2.putText(crop, tag, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(out_dir, f"f{vf:04d}.jpg"), crop)

    print(f"  已保存到 {out_dir}  总耗时{time.time()-t_start:.0f}s")

cap.release()

print("\n\n========== 汇总 ==========")
for gid, af, laf, tsf, tef, sr, er in results_summary:
    print(f"GT{gid}: GT标注video_frame[{af},{laf}]  真实存在video_frame[{tsf},{tef}]")
    print(f"   开始: {sr}")
    print(f"   结束: {er}")
print(f"\n全部完成，总耗时{time.time()-t_start:.0f}s")
