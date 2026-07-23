#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""复现脚本：跑真实tracktworoi.py的完整生产管线(检测流水线+真实torchreid ReID+BOTSORT)，
非交互式(复用已保存的ROI，不做比例标定)，对frame625~636这个已知会churn的入场车辆区域，
在关联的每一步打日志，找到到底是哪一步把它的id搞飞的。
"""
import runpy
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 把_repro_base.py的顶层代码(模型/ReID加载、检测流水线函数定义)跑一遍，拿到它的全局命名空间
ns = runpy.run_path(r"D:\yolo26-tracking\_repro_base.py", run_name="repro_base")

import numpy as np
import cv2
import json

cv2_ = cv2
np_ = np

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

from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.trackers.basetrack import TrackState

print(f"REID_AVAILABLE={REID_AVAILABLE}")

# ── 非交互式加载已保存的ROI(复用已有json，不做比例标定/绘制) ──────────────
COLD_START_FRAME = 400  # 冷启动帧号(1-based)，留够200+帧让别处的track能攒到>=30帧丢失门槛

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

# ── 目标监控区域: 帧625~636, 位置x<100, y in[1180,1300] ──────────────────
TARGET_FRAME_LO, TARGET_FRAME_HI = 618, 640
TARGET_X_MAX = 100
TARGET_Y_LO, TARGET_Y_HI = 1180, 1300


LOG_FROM_FRAME = 610  # 只在临近目标事件的窗口打印详细日志，减少无关车辆的噪音


def in_target(cx, cy):
    return frame_id >= LOG_FROM_FRAME and cx < TARGET_X_MAX and TARGET_Y_LO <= cy <= TARGET_Y_HI


def log(msg):
    print(f"    [LOG] {msg}", flush=True)


# ── монки-patch get_dists 打印目标区域相关的代价矩阵 ─────────────────────
_orig_get_dists = tracker.get_dists


def logging_get_dists(tracks, detections):
    dists = _orig_get_dists(tracks, detections)
    relevant_t = [i for i, t in enumerate(tracks) if t.mean is not None and in_target(t.mean[0], t.mean[1])]
    relevant_d = [j for j, d in enumerate(detections) if in_target(d.tlwh[0] + d.tlwh[2] / 2, d.tlwh[1] + d.tlwh[3] / 2)]
    if relevant_t or relevant_d:
        log(f"get_dists调用 (真实video帧={frame_id}, tracker.frame_id={tracker.frame_id}): "
            f"tracks数={len(tracks)} detections数={len(detections)}")
        for i in relevant_t:
            t = tracks[i]
            log(f"    候选track#{t.track_id} state={t.state} pos=({t.mean[0]:.1f},{t.mean[1]:.1f}) "
                f"last_obs_frame={t.last_observed_frame}")
        for j in relevant_d:
            d = detections[j]
            cx, cy = d.tlwh[0] + d.tlwh[2] / 2, d.tlwh[1] + d.tlwh[3] / 2
            log(f"    候选detection[{j}] pos=({cx:.1f},{cy:.1f}) score={d.score:.3f} cls={d.cls}")
        for i in relevant_t:
            for j in range(len(detections)):
                if dists[i, j] < 1.0:
                    log(f"    dist[track#{tracks[i].track_id}, det{j}] = {dists[i, j]:.4f}")
    return dists


tracker.get_dists = logging_get_dists

# ── monkey-patch _post_first_association 打印我的第三轮ReID恢复逻辑的候选和决策 ──
_orig_post_first = tracker._post_first_association


def logging_post_first(strack_pool, detections, u_track, u_detection, activated, refind):
    relevant_d = [j for j in u_detection if in_target(detections[j].tlwh[0] + detections[j].tlwh[2] / 2,
                                                        detections[j].tlwh[1] + detections[j].tlwh[3] / 2)]
    if relevant_d:
        log(f"_post_first_association 调用 (真实video帧={frame_id}): "
            f"u_track候选数={len(u_track)} u_detection候选数={len(u_detection)}")
        for j in relevant_d:
            d = detections[j]
            cx, cy = d.tlwh[0] + d.tlwh[2] / 2, d.tlwh[1] + d.tlwh[3] / 2
            log(f"    目标区域检测[{j}] pos=({cx:.1f},{cy:.1f}) cls={d.cls} curr_feat是否为空={d.curr_feat is None}")
        lost_candidates = [i for i in u_track if strack_pool[i].state == TrackState.Lost
                            and strack_pool[i].last_observed_frame is not None
                            and strack_pool[i].smooth_feat is not None]
        log(f"    满足Lost+有smooth_feat的候选track数: {len(lost_candidates)}")
        for i in lost_candidates:
            t = strack_pool[i]
            gap = tracker.frame_id - t.last_observed_frame
            if gap >= 30:
                log(f"    候选track#{t.track_id} gap={gap}帧 cls={t.cls} last_pos=({t.last_observed_mean[0]:.1f},{t.last_observed_mean[1]:.1f})")
    before_refind_ids = set(id(t) for t in refind)
    result = _orig_post_first(strack_pool, detections, u_track, u_detection, activated, refind)
    for t in refind:
        if id(t) not in before_refind_ids and t.mean is not None and in_target(t.mean[0], t.mean[1]):
            log(f"    !!! 第三轮ReID恢复命中: track#{t.track_id} 被复活到目标区域位置({t.mean[0]:.1f},{t.mean[1]:.1f})")
    return result


tracker._post_first_association = logging_post_first

# ── monkey-patch _apply_match 打印匹配结果 ────────────────────────────────
_orig_apply_match = tracker._apply_match


def logging_apply_match(track, det, activated, refind):
    cx, cy = det.tlwh[0] + det.tlwh[2] / 2, det.tlwh[1] + det.tlwh[3] / 2
    if in_target(cx, cy):
        log(f"_apply_match (真实video帧={frame_id}): track#{track.track_id}(state={track.state}) <- det pos=({cx:.1f},{cy:.1f})")
    return _orig_apply_match(track, det, activated, refind)


tracker._apply_match = logging_apply_match

# ── monkey-patch _init_new_tracks 打印新建track ──────────────────────────
_orig_init_new = tracker._init_new_tracks


def logging_init_new(u_detection, detections, activated, refind=None):
    for j in u_detection:
        d = detections[j]
        cx, cy = d.tlwh[0] + d.tlwh[2] / 2, d.tlwh[1] + d.tlwh[3] / 2
        if in_target(cx, cy):
            log(f"_init_new_tracks (真实video帧={frame_id}): 检测pos=({cx:.1f},{cy:.1f}) score={d.score:.3f} "
                f"是否达到new_track_thresh({tracker.args.new_track_thresh}): {d.score >= tracker.args.new_track_thresh}")
    before_ids = set(id(t) for t in activated)
    result = _orig_init_new(u_detection, detections, activated, refind)
    for t in activated:
        if id(t) not in before_ids and t.mean is not None and in_target(t.mean[0], t.mean[1]):
            log(f"    -> 新建track#{t.track_id}")
    return result


tracker._init_new_tracks = logging_init_new

# ── monkey-patch _unconfirmed_association 打印细节 ────────────────────────
_orig_unconf = tracker._unconfirmed_association


def logging_unconf(unconfirmed, u_detection, detections, activated, removed):
    relevant_u = [t for t in unconfirmed if t.mean is not None and in_target(t.mean[0], t.mean[1])]
    for t in relevant_u:
        log(f"_unconfirmed_association 输入 (真实video帧={frame_id}): unconfirmed track#{t.track_id} pos=({t.mean[0]:.1f},{t.mean[1]:.1f})")
    result = _orig_unconf(unconfirmed, u_detection, detections, activated, removed)
    for t in removed:
        if t.mean is not None and in_target(t.mean[0], t.mean[1]):
            log(f"    unconfirmed track#{t.track_id} 被removed(没配上)")
    for t in activated:
        if t in relevant_u and t.mean is not None and in_target(t.mean[0], t.mean[1]):
            log(f"    unconfirmed track#{t.track_id} 被confirm成功")
    return result


tracker._unconfirmed_association = logging_unconf

# ── 主循环 ────────────────────────────────────────────────────────────
frame_id = COLD_START_FRAME
import time
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

    if TARGET_FRAME_LO <= frame_id <= TARGET_FRAME_HI:
        target_dets = [d for d in dets if in_target(d["xywhr"][0], d["xywhr"][1])]
        for d in target_dets:
            log(f"帧{frame_id} 原始检测: pos=({d['xywhr'][0]:.1f},{d['xywhr'][1]:.1f}) "
                f"w={d['xywhr'][2]:.1f} h={d['xywhr'][3]:.1f} conf={d['conf']:.3f} cls={d['cls']}")

    if dets:
        results_adapter = OBBResultsAdapter(dets)
        tracked_boxes = tracker.update(results_adapter, img=frame)
        if TARGET_FRAME_LO <= frame_id <= TARGET_FRAME_HI:
            for track in tracked_boxes:
                if len(track) < 9:
                    continue
                xc, yc, w, h, angle, track_id, score, cls, idx = track
                if in_target(xc, yc):
                    log(f"帧{frame_id} 最终输出: track#{int(track_id)} pos=({xc:.1f},{yc:.1f})")

    if frame_id % 5 == 0:
        print(f"已处理 {frame_id} 帧, 耗时{time.time()-t0:.0f}s", flush=True)

    frame_id += 1
    if frame_id > TARGET_FRAME_HI + 5:
        break

cap.release()
print("完成")
