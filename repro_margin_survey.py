#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
排查"Bug C修复后，第三轮ReID恢复的最终判据仍然只是相似度最高者胜出"这件事有多risky。
不只看已知失败的事件，而是记录全视频里【每一次】第三轮成功匹配的：
  赢家相似度 vs (通过了类别+距离+方向门槛的)第二名相似度 的差距(margin)。
margin小 -> 说明是"险胜"，换个场景容易翻车；margin普遍很大 -> 说明目前的相似度分布本身就有安全边际，
不是靠运气赢的。
"""
import runpy
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

ns = runpy.run_path(r"D:\yolo26-tracking\_repro_base.py", run_name="repro_base")

import numpy as np
import cv2
import json
import time

VIDEO_PATH = ns["VIDEO_PATH"]
ROI_FILE = ns["ROI_FILE"]
load_roi_file = ns["load_roi_file"]
multi_roi_mask = ns["multi_roi_mask"]
apply_clahe = ns["apply_clahe"]
gamma_transform = ns["gamma_transform"]
apply_blurred_background_fill = ns["apply_blurred_background_fill"]
sliced_predict_obb_direct = ns["sliced_predict_obb_direct"]
_full_detection_pipeline = ns["_full_detection_pipeline"]
OBBResultsAdapter = ns["OBBResultsAdapter"]
PRE_MASK_DETECTION = ns["PRE_MASK_DETECTION"]
PRE_MASK_MODE = ns["PRE_MASK_MODE"]
PRE_MASK_FEATHER = ns["PRE_MASK_FEATHER"]
PRE_MASK_BLUR_KSIZE = ns["PRE_MASK_BLUR_KSIZE"]
IOU_NMS_THR = 0.2
yolo_model = ns["yolo_model"]
ReIDEncoderObj = ns["ReIDEncoderObj"]
REID_AVAILABLE = ns["REID_AVAILABLE"]
args = ns["args"]

from ultralytics.trackers.bot_sort import BOTSORT, ReidRecoveryConfig
from ultralytics.trackers.basetrack import TrackState
from ultralytics.trackers.utils import matching

print(f"REID_AVAILABLE={REID_AVAILABLE}")

COLD_START_FRAME = 1

cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
ret, first_frame = cap.read()
assert ret, "无法读取视频首帧"

roi_polys_modes, pixel_length, real_length = load_roi_file(ROI_FILE)
roi_mask = multi_roi_mask(roi_polys_modes, first_frame.shape)
print(f"ROI加载完成，{len(roi_polys_modes)}个区域")

cap.set(cv2.CAP_PROP_POS_FRAMES, COLD_START_FRAME - 1)

tracker = BOTSORT(args=args)
if REID_AVAILABLE and ReIDEncoderObj is not None:
    tracker.encoder = ReIDEncoderObj
    tracker.args.with_reid = True
    print("[Tracker] ReID encoder injected.")
else:
    raise RuntimeError("ReID没有成功加载，复现没有意义")

results_log = []  # (frame, winner_tid, winner_sim, runnerup_tid, runnerup_sim, margin, n_geo_valid_candidates)

cfg = ReidRecoveryConfig
_orig_post_first = tracker._post_first_association


def logging_post_first(strack_pool, detections, u_track, u_detection, activated, refind):
    lost_idx = [
        i for i in u_track
        if strack_pool[i].state == TrackState.Lost
        and strack_pool[i].last_observed_frame is not None
        and (tracker.frame_id - strack_pool[i].last_observed_frame) >= cfg.MIN_GAP_FOR_RECOVERY
        and strack_pool[i].smooth_feat is not None
        and not tracker._is_exiting_frame(strack_pool[i])
    ]
    det_idx = [j for j in u_detection if detections[j].curr_feat is not None]

    if lost_idx and det_idx:
        lost_tracks = [strack_pool[i] for i in lost_idx]
        cand_dets = [detections[j] for j in det_idx]
        dist = matching.embedding_distance(lost_tracks, cand_dets)

        # 对每个检测j，收集所有"通过类别+距离门槛"(几何合理)的候选，按相似度降序
        for b, dj in enumerate(det_idx):
            det = detections[dj]
            dx, dy = det.tlwh[0] + det.tlwh[2] / 2, det.tlwh[1] + det.tlwh[3] / 2
            geo_valid = []
            for a, ti in enumerate(lost_idx):
                track = strack_pool[ti]
                if track.cls != det.cls:
                    continue
                gap = tracker.frame_id - track.last_observed_frame
                max_dist = gap * cfg.MAX_SPEED_PX_PER_FRAME
                lx, ly = track.last_observed_mean[0], track.last_observed_mean[1]
                pdist = float(np.hypot(dx - lx, dy - ly))
                if pdist > max_dist:
                    continue
                sim = 1.0 - float(dist[a, b])
                geo_valid.append((sim, track.track_id))
            if len(geo_valid) >= 2:
                geo_valid.sort(key=lambda r: -r[0])
                top_sim, top_tid = geo_valid[0]
                if top_sim >= cfg.ACCEPT_SIM_THRESH:
                    second_sim, second_tid = geo_valid[1]
                    results_log.append((tracker.frame_id, top_tid, top_sim, second_tid, second_sim,
                                         top_sim - second_sim, len(geo_valid)))

    return _orig_post_first(strack_pool, detections, u_track, u_detection, activated, refind)


tracker._post_first_association = logging_post_first

frame_id = COLD_START_FRAME
t0 = time.time()
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = apply_clahe(frame)
    frame = gamma_transform(frame, gamma=1.2)

    proc_frame_for_sahi = frame
    if PRE_MASK_DETECTION and roi_mask is not None:
        if PRE_MASK_MODE == "blur":
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

    if dets:
        results_adapter = OBBResultsAdapter(dets)
        tracker.update(results_adapter, img=frame)

    if frame_id % 100 == 0:
        print(f"已处理 {frame_id} 帧, 耗时{time.time()-t0:.0f}s, 目前记录到{len(results_log)}条竞争候选记录", flush=True)

    frame_id += 1

cap.release()
print(f"完成，总耗时{time.time()-t0:.0f}s，共记录{len(results_log)}条'>=2个几何合理候选'的竞争记录")

with open(r"D:\yolo26-tracking\margin_survey_results.json", "w", encoding="utf-8") as f:
    json.dump(results_log, f)
print("已保存到 margin_survey_results.json")

if results_log:
    margins = np.array([r[5] for r in results_log])
    print(f"\nmargin统计: mean={margins.mean():.4f} std={margins.std():.4f} min={margins.min():.4f} max={margins.max():.4f}")
    print(f"margin < 0.05 (险胜) 的次数: {(margins < 0.05).sum()} / {len(margins)}")
    print(f"margin < 0.10 的次数: {(margins < 0.10).sum()} / {len(margins)}")
