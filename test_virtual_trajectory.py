#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""虚拟轨迹补偿(LI+GPR)验收测试：对应 spec 的三条验收要求。
1. 几何平滑性：中间帧状态位置平滑过渡、角度无异常跳变
2. 角度周期边界消歧：跨越周期边界时选出物理合理的角度值
3. 集成对比：真实 BOTrack/BOTSORT 上，丢失-重新匹配场景下，
   启用虚拟轨迹补偿 vs. 现有纯预测方式，轨迹平滑度/位置误差对比
"""
import sys
sys.path.insert(0, r"D:\yolo26-tracking")
import numpy as np
from types import SimpleNamespace

from ultralytics.trackers.utils.virtual_trajectory import build_virtual_trajectory, VirtualTrajectoryConfig
from ultralytics.trackers.bot_sort import BOTrack, BOTSORT
from ultralytics.trackers.basetrack import TrackState


def make_det(cx, cy, w, h, angle_deg, score=0.9, cls=0, idx=0):
    xywh = np.array([cx, cy, w, h, np.deg2rad(angle_deg), idx], dtype=np.float32)
    return BOTrack(xywh, score, cls)


print("=== 1. 几何平滑性：中间帧位置平滑过渡、角度无异常跳变 ===")
z_t1 = np.array([100.0, 100.0, 80.0, 40.0, np.deg2rad(10.0)])
z_t2 = np.array([150.0, 100.0, 80.0, 40.0, np.deg2rad(15.0)])
t1, t2 = 10, 15
lost_frame_indices = [11, 12, 13, 14]
virtual = build_virtual_trajectory(
    z_t1=z_t1, z_t2=z_t2, t1=t1, t2=t2, lost_frame_indices=lost_frame_indices,
    angle_dim_index=4, angle_period=np.pi,
    gpr_length_scale=VirtualTrajectoryConfig.GPR_LENGTH_SCALE,
    gpr_noise_level=VirtualTrajectoryConfig.GPR_NOISE_LEVEL,
    vx=10.0, vy=0.0,
)
print(f"  x序列: {virtual[:, 0]}")
print(f"  angle序列(度): {np.rad2deg(virtual[:, 4])}")
assert virtual.shape == (4, 5)
# x应单调递增(在100~150之间小幅超调都可接受，但不能整体乱序)
diffs = np.diff(virtual[:, 0])
assert np.all(diffs > 0), "x序列应单调递增，平滑过渡"
assert 95 <= virtual[0, 0] <= 155 and 95 <= virtual[-1, 0] <= 155
# angle应在10~15度附近平滑变化，不应有大跳变
angle_deg = np.rad2deg(virtual[:, 4])
assert np.all(np.abs(np.diff(angle_deg)) < 5), "角度逐帧变化不应有异常跳变"
assert np.all((angle_deg > 5) & (angle_deg < 20)), "角度应在起止值附近，不应跑偏"
# w, h应基本保持不变(两端相同)
assert np.allclose(virtual[:, 2], 80.0, atol=2.0)
assert np.allclose(virtual[:, 3], 40.0, atol=2.0)
print("  通过\n")


print("=== 2. 角度周期边界消歧：跨越180度边界不应产生跳变 ===")
# 车辆朝向接近179度，沿着接近-x方向(略带+y分量)运动，heading约等于179度
# 观测t2给出的原始角度是-179度的等价表示(周期pi=180度物理等价,但数值上差了180)
z_t1b = np.array([200.0, 200.0, 90.0, 40.0, np.deg2rad(179.0)])
z_t2b = np.array([150.0, 205.0, 90.0, 40.0, np.deg2rad(-179.0)])
vx, vy = -10.0, 0.3  # heading = atan2(0.3,-10) ≈ 178.3度，与179度同一分支
virtual2 = build_virtual_trajectory(
    z_t1=z_t1b, z_t2=z_t2b, t1=10, t2=15, lost_frame_indices=[11, 12, 13, 14],
    angle_dim_index=4, angle_period=np.pi, vx=vx, vy=vy,
)
angle_deg2 = np.rad2deg(virtual2[:, 4])
print(f"  angle序列(度,规范分支): {angle_deg2}")
# cos/sin周期编码会把输出规范化到主分支(-90°,90°]内，179°和-179°在这个分支下
# 都等价于约 -1°；只要序列本身连续平滑(不出现真正的跳变)，就说明周期边界处理正确
period_dist = lambda a, b: abs(((a - b + 90) % 180) - 90)
assert period_dist(angle_deg2[0], -1.0) < 5, f"起点应落在179度所在的等价分支(约-1度)附近，实际={angle_deg2[0]}"
assert period_dist(angle_deg2[-1], 1.0) < 5, f"终点应落在181度所在的等价分支(约1度)附近，实际={angle_deg2[-1]}"
frame_jumps = [period_dist(a, b) for a, b in zip(angle_deg2[:-1], angle_deg2[1:])]
print(f"  逐帧周期最短角度差: {frame_jumps}")
assert all(j < 5 for j in frame_jumps), "角度序列逐帧不应出现跳变"
print("  通过\n")


print("=== 3. 集成对比：真实 BOTrack + BOTSORT._maybe_inject_virtual_trajectory ===")
args = SimpleNamespace(
    track_high_thresh=0.7, track_low_thresh=0.3, new_track_thresh=0.7,
    track_buffer=120, match_thresh=0.6, fuse_score=True,
    proximity_thresh=0.5, appearance_thresh=0.5, with_reid=False,
    model=None, gmc_method='sparseOptFlow',
)
bot = BOTSORT(args=args)
bot.frame_id = 1

# 轨迹：车辆沿x轴匀速运动，dx=10/帧，先正常跑5帧建立稳定速度估计
track = make_det(100, 100, 80, 40, angle_deg=0)
track.activate(bot.kalman_filter, frame_id=1)
for f in range(2, 6):
    det = make_det(100 + 10 * (f - 1), 100, 80, 40, angle_deg=0)
    track.predict()
    track.update(det, frame_id=f)

# 丢失5帧(6~10)，真实轨迹本应继续在x=140+10*k处，但检测器没检测到
# 第11帧重新匹配到，真实位置应为 x=100+10*10=200 (若一直匀速)
true_x_at_11 = 100 + 10 * 10
det_reactivate = make_det(true_x_at_11, 100, 80, 40, angle_deg=0)

# --- 方案A: 现有默认方式(纯预测跨越丢失帧，无虚拟观测) ---
import copy
track_baseline = copy.deepcopy(track)
track_baseline.mark_lost()
for f in range(6, 11):
    track_baseline.predict()
bot.frame_id = 11
track_baseline.re_activate(det_reactivate, frame_id=11, new_id=False)
baseline_cov_trace = np.trace(track_baseline.covariance)

# --- 方案B: 启用虚拟轨迹补偿 ---
track_virtual = copy.deepcopy(track)
track_virtual.mark_lost()
for f in range(6, 11):
    track_virtual.predict()  # 模拟tracker每帧仍会调用multi_predict(strack_pool)
bot.frame_id = 11
bot._maybe_inject_virtual_trajectory(track_virtual, det_reactivate)
track_virtual.re_activate(det_reactivate, frame_id=11, new_id=False)
virtual_cov_trace = np.trace(track_virtual.covariance)

print(f"  重新匹配后 mean.shape: baseline={track_baseline.mean.shape}, virtual={track_virtual.mean.shape}")
print(f"  重新匹配后位置: baseline=({track_baseline.mean[0]:.1f},{track_baseline.mean[1]:.1f}), "
      f"virtual=({track_virtual.mean[0]:.1f},{track_virtual.mean[1]:.1f})  (真实观测x={true_x_at_11})")
print(f"  重新匹配后协方差迹(不确定度): baseline={baseline_cov_trace:.2f}, virtual={virtual_cov_trace:.2f}")
assert np.all(np.isfinite(track_virtual.mean)) and np.all(np.isfinite(track_virtual.covariance))
assert track_virtual.state == TrackState.Tracked and track_virtual.is_activated
# 核心预期：走过LI+GPR虚拟观测桥接的轨迹，协方差应显著小于纯预测方案
# (因为中间每一帧都有伪观测做修正，不确定度没有像纯预测那样单调累积)
assert virtual_cov_trace < baseline_cov_trace, \
    f"虚拟轨迹补偿后的不确定度应低于纯预测方案，但 {virtual_cov_trace} >= {baseline_cov_trace}"
print(f"  不确定度降低比例: {(1 - virtual_cov_trace / baseline_cov_trace) * 100:.1f}%")
print("  通过\n")

print("全部测试通过。")
