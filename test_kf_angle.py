#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""冒烟测试：扩展后的10维KF(含角度)能否正确跑通 activate/update/lost/re_activate 全流程。"""
import sys
sys.path.insert(0, r"D:\yolo26-tracking")
import numpy as np
from ultralytics.trackers.bot_sort import BOTrack
from ultralytics.trackers.basetrack import TrackState
from ultralytics.trackers.utils.kalman_filter import KalmanFilterXYWHA, resolve_angle


def make_track(cx, cy, w, h, angle_deg, score=0.9, cls=0, idx=0):
    xywh = np.array([cx, cy, w, h, np.deg2rad(angle_deg), idx], dtype=np.float32)
    return BOTrack(xywh, score, cls)


print("=== 1. resolve_angle 单元测试：跨周期边界不该跳变 ===")
period = np.pi  # 180度周期
# 场景：预测角度 179度(接近周期上边界)，新观测原始给的是 -179度(其实和179度物理上很接近，只差2度)
pred = np.deg2rad(179)
raw = np.deg2rad(-179)
resolved = resolve_angle(raw, ref_angle=pred, vx=0.0, vy=0.0, period=period, motion_thresh=2.0)
print(f"  预测角度=179°, 原始观测=-179°(静止,退化用ref_angle) -> 消歧后={np.rad2deg(resolved):.2f}° (期望接近181°,不是-179°)")
assert abs(np.rad2deg(resolved) - 181.0) < 1e-6, "静止分支消歧结果不对"

# 场景：车辆正在向右运动(vx>0,vy=0)，航向应该是0度；原始角度给的是180度(和0度物理上等价，周期pi)
resolved2 = resolve_angle(np.deg2rad(180), ref_angle=999.0, vx=5.0, vy=0.0, period=period, motion_thresh=2.0)
print(f"  航向=atan2(0,5)=0°, 原始观测=180°(物理等价) -> 消歧后={np.rad2deg(resolved2):.2f}° (期望接近0°，用了航向而不是ref_angle)")
assert abs(np.rad2deg(resolved2) - 0.0) < 1e-6, "运动分支消歧结果不对，没有用航向"
print("  通过\n")


print("=== 2. BOTrack 完整生命周期：activate -> update x3 -> lost x5(纯predict) -> re_activate ===")
kf = KalmanFilterXYWHA()

t0 = make_track(100, 100, 80, 40, angle_deg=10)
t0.activate(kf, frame_id=1)
print(f"  activate后 mean.shape={t0.mean.shape} (期望(10,))  angle={np.rad2deg(t0.angle):.2f}°")
assert t0.mean.shape == (10,)
assert t0.state == TrackState.Tracked

# 正常更新3帧，角度缓慢变化 10->12->14度，位置匀速右移
for i, (dx, ang) in enumerate([(5, 12), (5, 14), (5, 16)], start=2):
    det = make_track(100 + dx * (i - 1), 100, 80, 40, angle_deg=ang)
    t0.predict()
    t0.update(det, frame_id=i)
    print(f"  frame{i}: mean=(x={t0.mean[0]:.1f},y={t0.mean[1]:.1f},w={t0.mean[2]:.1f},h={t0.mean[3]:.1f},"
          f"angle={np.rad2deg(t0.mean[4]):.2f}°,vx={t0.mean[5]:.2f},vy={t0.mean[6]:.2f},vangle={np.rad2deg(t0.mean[9]):.2f}°/f)")

assert t0.mean.shape == (10,)
print(f"  tlwh属性: {t0.tlwh} (应为长度4)")
assert t0.tlwh.shape == (4,)
print(f"  xywha属性: {t0.xywha} (应为长度5)")
assert t0.xywha.shape == (5,)

# 模拟丢失5帧：只做predict，不update，state切成Lost
t0.mark_lost()
print(f"\n  标记为Lost，做5帧纯predict（角速度/宽高速度应该被冻结，位置仍按vx,vy外推）...")
angle_before_lost = float(t0.mean[4])
for _ in range(5):
    t0.predict()
angle_after_lost = float(t0.mean[4])
print(f"  丢失前角度={np.rad2deg(angle_before_lost):.2f}°, 5帧纯predict后角度={np.rad2deg(angle_after_lost):.2f}° (期望完全不变，因为vangle被冻结为0)")
assert abs(angle_before_lost - angle_after_lost) < 1e-9, "Lost期间角度不该漂移(vangle应被冻结)"
print(f"  丢失期间x坐标变化: {t0.mean[0]:.2f} (期望仍在按vx外推移动，不是冻结)")

# 重新匹配：给一个角度经过了周期边界翻转的新检测（比如16度写成了16-180=-164度，模拟检测器角度歧义）
det_reactivate = make_track(100 + 5 * 4, 100, 80, 40, angle_deg=16 - 180)  # 物理等价，但数值差了180度
print(f"\n  重新匹配检测: 原始角度值={16-180}° (和真实的16°物理等价，差了180°周期)")
t0.re_activate(det_reactivate, frame_id=8, new_id=False)
print(f"  re_activate后角度={np.rad2deg(t0.mean[4]):.2f}° (期望接近16°附近，不是被错误当成-164°来修正)")
assert t0.state == TrackState.Tracked
assert t0.is_activated
# 消歧后角度应该落在合理范围（离预测角度16度左右不远，不应该是-164度这种离谱跳变）
diff = abs(((np.rad2deg(t0.mean[4]) - 16 + 90) % 180) - 90)
print(f"  与真实16°的最短角度差: {diff:.2f}° (期望很小)")
assert diff < 10, "重新匹配后角度消歧失败，产生了错误跳变"

print("\n=== 3. BOTrack.multi_predict 向量化路径 ===")
tracks = [make_track(200, 200, 70, 35, angle_deg=a) for a in (0, 45, 90)]
for tr in tracks:
    tr.activate(kf, frame_id=1)
BOTrack.multi_predict(tracks)
for tr in tracks:
    assert tr.mean.shape == (10,)
print("  3条轨迹批量predict通过，mean形状都正确\n")

print("全部测试通过。")
