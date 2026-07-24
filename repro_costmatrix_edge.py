#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对当前24次IDSW里12次"边缘僵尸复活"事件，打印_post_first_association在该次调用时的完整候选代价矩阵
(每个候选track的相似度sim/物理距离/是否边缘退出)，并模拟"排除边缘僵尸候选"后是否还能正确匹配
(即剩余候选里有没有别的track会顶上，还是应该正常落到_init_new_tracks建新id)。

跑全量视频(冷启动帧1)，因为12次事件横跨全片(帧147~1652)。只在目标事件的±2帧窗口内打印，其余帧静默跑。
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

# (revival_frame, x, y) —— 12个"边缘僵尸复活"事件，来自 outputs/yolo26-obb_20260723_213702 的24次IDSW明细
TARGETS = [
    (147, 11.5, 1188.6),
    (467, 1141.0, 41.2),
    (548, 681.0, 70.6),
    (564, 600.6, 101.9),
    (571, 1040.2, 28.0),
    (661, 10.3, 1188.4),
    (674, 437.0, 208.8),
    (966, 9.0, 1241.0),
    (1142, 492.1, 177.1),
    (1536, 623.8, 111.6),
    (1649, 1103.6, 35.2),
    (1652, 1093.2, 30.7),
]
FRAME_TOL = 2
POS_TOL = 80

EDGE_XMIN, EDGE_XMAX = 1, 3525
EDGE_YMIN, EDGE_YMAX = 10, 1986
EDGE_MARGIN = 100


def is_edge(x, y):
    return (x - EDGE_XMIN < EDGE_MARGIN or EDGE_XMAX - x < EDGE_MARGIN or
            y - EDGE_YMIN < EDGE_MARGIN or EDGE_YMAX - y < EDGE_MARGIN)


def matched_target(cx, cy):
    for tf, tx, ty in TARGETS:
        if abs(frame_id - tf) <= FRAME_TOL and abs(cx - tx) < POS_TOL and abs(cy - ty) < POS_TOL:
            return tf
    return None


def log(msg):
    print(f"    [LOG] {msg}", flush=True)


cfg = ReidRecoveryConfig
_orig_post_first = tracker._post_first_association


def logging_post_first(strack_pool, detections, u_track, u_detection, activated, refind):
    relevant_d = [(j, matched_target(detections[j].tlwh[0] + detections[j].tlwh[2] / 2,
                                      detections[j].tlwh[1] + detections[j].tlwh[3] / 2))
                  for j in u_detection]
    relevant_d = [(j, tf) for j, tf in relevant_d if tf is not None]

    if relevant_d and tracker.args.with_reid and tracker.encoder is not None:
        lost_idx = [
            i for i in u_track
            if strack_pool[i].state == TrackState.Lost
            and strack_pool[i].last_observed_frame is not None
            and (tracker.frame_id - strack_pool[i].last_observed_frame) >= cfg.MIN_GAP_FOR_RECOVERY
            and strack_pool[i].smooth_feat is not None
        ]
        det_idx = [j for j in u_detection if detections[j].curr_feat is not None]

        if lost_idx and det_idx:
            lost_tracks = [strack_pool[i] for i in lost_idx]
            cand_dets = [detections[j] for j in det_idx]
            dist = matching.embedding_distance(lost_tracks, cand_dets)

            for j, tf in relevant_d:
                if j not in det_idx:
                    continue
                b = det_idx.index(j)
                det = detections[j]
                dx, dy = det.tlwh[0] + det.tlwh[2] / 2, det.tlwh[1] + det.tlwh[3] / 2
                log(f"===== 事件 revival_frame={tf} 真实video帧={frame_id} 检测pos=({dx:.1f},{dy:.1f}) cls={det.cls} =====")
                rows = []
                for a, ti in enumerate(lost_idx):
                    track = strack_pool[ti]
                    if track.cls != det.cls:
                        continue
                    gap = tracker.frame_id - track.last_observed_frame
                    max_dist = gap * cfg.MAX_SPEED_PX_PER_FRAME
                    lx, ly = track.last_observed_mean[0], track.last_observed_mean[1]
                    sim = 1.0 - float(dist[a, b])
                    pdist = float(np.hypot(dx - lx, dy - ly))
                    edge = is_edge(lx, ly)
                    accepted = sim >= cfg.ACCEPT_SIM_THRESH and pdist <= max_dist
                    rows.append((sim, track.track_id, gap, lx, ly, pdist, max_dist, edge, accepted))
                rows.sort(key=lambda r: -r[0])
                log(f"    候选数(同类)={len(rows)}")
                for sim, tid, gap, lx, ly, pdist, max_dist, edge, accepted in rows:
                    log(f"    track#{tid:>4} sim={sim:.4f} gap={gap:>4} last_pos=({lx:>7.1f},{ly:>7.1f}) "
                        f"phys_dist={pdist:>7.1f} max_allow={max_dist:>7.1f} 边缘={'是' if edge else '否'} "
                        f"{'[被选中]' if accepted and sim >= cfg.ACCEPT_SIM_THRESH and pdist <= max_dist else ''}")
                # 模拟：排除边缘候选后，剩余候选里谁会赢
                non_edge = [r for r in rows if not r[7] and r[0] >= cfg.ACCEPT_SIM_THRESH and r[5] <= r[6]]
                if non_edge:
                    best = non_edge[0]
                    log(f"    [排除边缘候选后] 仍有非边缘候选达标 -> 会误判给 track#{best[1]} (sim={best[0]:.4f})")
                else:
                    log(f"    [排除边缘候选后] 无候选达标 -> 会正确走向_init_new_tracks建新id")

    result = _orig_post_first(strack_pool, detections, u_track, u_detection, activated, refind)
    return result


tracker._post_first_association = logging_post_first

frame_id = COLD_START_FRAME
t0 = time.time()
LAST_TARGET_FRAME = max(tf for tf, _, _ in TARGETS)
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

    if frame_id % 50 == 0:
        print(f"已处理 {frame_id} 帧, 耗时{time.time()-t0:.0f}s", flush=True)

    frame_id += 1
    if frame_id > LAST_TARGET_FRAME + 5:
        break

cap.release()
print(f"完成，总耗时{time.time()-t0:.0f}s")
