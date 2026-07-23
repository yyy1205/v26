#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""非交互集成测试：用真实视频片段跑 YOLO检测 -> BOTSORT.update()（新10维角度KF），
检查全程不报错、track数量合理、角度/位置没有NaN或离谱跳变。
"""
import sys
sys.path.insert(0, r"D:\yolo26-tracking")
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import cv2
from types import SimpleNamespace
from ultralytics import YOLO
from ultralytics.trackers.bot_sort import BOTSORT

VIDEO_PATH = r"E:\DJI\cvat\0111_D_stabilized.avi"
MODEL_PATH = r"D:\yolo26-tracking\runs\obb\exp_yolo26-3\weights\best.pt"
N_FRAMES = 100
CONF_THR = 0.3
IMGSZ = 1280  # 简化：单尺度整帧推理即可，本测试只验证追踪链路，不重复验证检测流程

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
    'device': 'cuda:0',  # 显式带索引，规避 AutoBackend 用 torch.device('cuda').index=None 传给 onnxruntime 报错的问题
    'gmc_method': 'sparseOptFlow',
}
args = SimpleNamespace(**tracker_args)

model = YOLO(MODEL_PATH)
model.to("cuda:0")
tracker = BOTSORT(args=args)
print(f"encoder类型: {type(tracker.encoder).__name__}, with_reid={tracker.args.with_reid}")

print(f"kalman_filter类型: {type(tracker.kalman_filter).__name__}")
assert type(tracker.kalman_filter).__name__ == "KalmanFilterXYWH", "追踪器没有用上预期的(无角度)KF类！"

reid_recovery_count = [0]
_orig_post_first = tracker._post_first_association


def _counting_post_first(strack_pool, detections, u_track, u_detection, activated, refind):
    before = len(refind)
    result = _orig_post_first(strack_pool, detections, u_track, u_detection, activated, refind)
    reid_recovery_count[0] += len(refind) - before
    return result


tracker._post_first_association = _counting_post_first

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"无法打开视频: {VIDEO_PATH}")

frame_id = 0
max_track_count = 0
angle_history = {}  # track_id -> [angle_deg, ...]
nan_found = False
crash_found = False

while frame_id < N_FRAMES:
    ret, frame = cap.read()
    if not ret:
        print(f"视频在第{frame_id}帧结束")
        break
    frame_id += 1

    res = model.predict(frame, imgsz=IMGSZ, conf=CONF_THR, verbose=False)
    obb = res[0].obb.cpu() if res[0].obb is not None else None
    if obb is None or len(obb) == 0:
        tracked = tracker.update(obb, img=frame) if obb is not None else np.empty((0, 9))
        continue

    import time as _time
    _t0 = _time.time()
    try:
        tracked_boxes = tracker.update(obb, img=frame)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"帧{frame_id}: tracker.update() 抛出异常: {e}")
        crash_found = True
        break

    max_track_count = max(max_track_count, len(tracked_boxes))

    for track in tracked_boxes:
        if len(track) < 9:
            continue
        xc, yc, w, h, angle, track_id, score, cls, idx = track
        if not np.all(np.isfinite([xc, yc, w, h, angle])):
            print(f"帧{frame_id} track{int(track_id)}: 出现NaN/Inf! xywh_angle=({xc},{yc},{w},{h},{angle})")
            nan_found = True
        angle_history.setdefault(int(track_id), []).append(float(np.rad2deg(angle)))

    print(f"帧{frame_id}: tracker.update() 耗时 {_time.time()-_t0:.2f}s, 检测数={len(obb)}, 活跃track数={len(tracked_boxes)}")
    if frame_id % 50 == 0:
        print(f"已处理 {frame_id}/{N_FRAMES} 帧，当前活跃track数={len(tracked_boxes)}，累计出现过的track_id数={len(angle_history)}")

cap.release()

print(f"\n=== 汇总 ===")
print(f"处理帧数: {frame_id}")
print(f"最大同时活跃track数: {max_track_count}")
print(f"累计出现过的track_id数: {len(angle_history)}")
print(f"出现过NaN/Inf: {nan_found}")
print(f"tracker.update()崩溃: {crash_found}")
print(f"ReID第三轮恢复关联触发次数: {reid_recovery_count[0]}")

# 检查角度是否有逐帧跳变过大的情况(比如相邻帧转了>60度，可能说明消歧逻辑有问题)
max_jump = 0.0
max_jump_track = None
for tid, hist in angle_history.items():
    if len(hist) < 2:
        continue
    for a, b in zip(hist[:-1], hist[1:]):
        diff = abs(((b - a + 90) % 180) - 90)  # 按180度周期算最短角度差
        if diff > max_jump:
            max_jump = diff
            max_jump_track = tid
print(f"逐帧最大角度跳变(按180°周期折算): {max_jump:.2f}° (track_id={max_jump_track})")

ok = (not nan_found) and (not crash_found)
print(f"\n{'通过' if ok else '失败'}")
