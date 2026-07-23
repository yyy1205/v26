#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""跑前500帧真实视频(YOLO检测 + BOTSORT+ReID恢复关联)，输出cvat_eval.txt格式结果，
方便直接复用eval_mot.py评估这段范围内的IDSW，看ReID第三轮恢复关联的实际效果
(不需要等交互式tracktworoi.py跑完整个视频)。
"""
import sys
sys.path.insert(0, r"D:\yolo26-tracking")
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import tracemalloc
import psutil
import numpy as np
import cv2
from types import SimpleNamespace
from ultralytics import YOLO
from ultralytics.trackers.bot_sort import BOTSORT

VIDEO_PATH = r"E:\DJI\cvat\0111_D_stabilized.avi"
MODEL_PATH = r"D:\yolo26-tracking\runs\obb\exp_yolo26-3\weights\best.pt"
N_FRAMES = 230
CONF_THR = 0.3
IMGSZ = 1280
OUT_PATH = r"D:\yolo26-tracking\outputs\reid_500frame_test\cvat_eval.txt"
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

tracker_args = {
    'track_high_thresh': 0.7,
    'track_low_thresh': 0.3,
    'new_track_thresh': 0.7,
    'track_buffer': 120,
    'match_thresh': 0.6,
    'fuse_score': True,
    'proximity_thresh': 0.5,
    'appearance_thresh': 0.5,
    'with_reid': True,
    'model': r"D:\yolo26-tracking\reid_model\osnet_ain_x1_0_vehicle.onnx",
    'device': 'cuda:0',
    'gmc_method': 'sparseOptFlow',
}
args = SimpleNamespace(**tracker_args)

model = YOLO(MODEL_PATH)
model.to("cuda:0")
tracker = BOTSORT(args=args)
print(f"encoder类型: {type(tracker.encoder).__name__}, with_reid={tracker.args.with_reid}")

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"无法打开视频: {VIDEO_PATH}")

out_f = open(OUT_PATH, "w")
frame_id = 0
t_start = time.time()
reid_recovery_count = [0]
_orig_post_first = tracker._post_first_association


def _counting_post_first(strack_pool, detections, u_track, u_detection, activated, refind):
    before = len(refind)
    result = _orig_post_first(strack_pool, detections, u_track, u_detection, activated, refind)
    reid_recovery_count[0] += len(refind) - before
    return result


tracker._post_first_association = _counting_post_first

_orig_inject = tracker._maybe_inject_virtual_trajectory


def _logging_inject(track, det):
    t1 = track.last_observed_frame
    t2 = tracker.frame_id
    gap = (t2 - t1 - 1) if t1 is not None else None
    print(f"    [virtual_trajectory] track#{track.track_id} gap={gap}帧 t1={t1} t2={t2}", flush=True)
    return _orig_inject(track, det)


tracker._maybe_inject_virtual_trajectory = _logging_inject

tracemalloc.start()
last_snapshot = tracemalloc.take_snapshot()

while frame_id < N_FRAMES:
    ret, frame = cap.read()
    if not ret:
        print(f"视频在第{frame_id}帧结束")
        break
    frame_id += 1

    res = model.predict(frame, imgsz=IMGSZ, conf=CONF_THR, verbose=False)
    obb = res[0].obb.cpu() if res[0].obb is not None else None

    try:
        tracked_boxes = tracker.update(obb, img=frame) if obb is not None else np.empty((0, 9))
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"帧{frame_id}: tracker.update() 抛出异常: {e}")
        break

    for track in tracked_boxes:
        if len(track) < 9:
            continue
        xc, yc, w, h, angle, track_id, score, cls, idx = track
        rect = ((float(xc), float(yc)), (float(w), float(h)), float(np.rad2deg(angle)))
        p = cv2.boxPoints(rect)
        out_f.write(f"{frame_id},{int(track_id)},{p[0][0]:.2f},{p[0][1]:.2f},{p[1][0]:.2f},{p[1][1]:.2f},"
                    f"{p[2][0]:.2f},{p[2][1]:.2f},{p[3][0]:.2f},{p[3][1]:.2f},1,{int(cls)},1.0\n")

    if frame_id % 10 == 0:
        elapsed = time.time() - t_start
        rss_mb = psutil.Process().memory_info().rss / 1024 / 1024
        traced_mb = tracemalloc.get_traced_memory()[0] / 1024 / 1024
        print(f"已处理 {frame_id}/{N_FRAMES} 帧，耗时{elapsed:.0f}s，活跃track数={len(tracked_boxes)}，"
              f"lost_stracks数={len(tracker.lost_stracks)}，removed_stracks数={len(tracker.removed_stracks)}，"
              f"ReID恢复触发次数={reid_recovery_count[0]}，RSS内存={rss_mb:.0f}MB，tracemalloc追踪={traced_mb:.0f}MB", flush=True)
        new_snapshot = tracemalloc.take_snapshot()
        top_stats = new_snapshot.compare_to(last_snapshot, "lineno")
        print("    [内存增量Top5]")
        for stat in top_stats[:5]:
            print(f"      {stat}")
        last_snapshot = new_snapshot

out_f.close()
cap.release()
print(f"\n完成，共{frame_id}帧，总耗时{time.time()-t_start:.0f}s")
print(f"ReID第三轮恢复关联总触发次数: {reid_recovery_count[0]}")
print(f"结果保存在: {OUT_PATH}")
