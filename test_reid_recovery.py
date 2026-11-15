#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验收测试：BOTSORT第三轮"ReID引导恢复关联"。
用合成的外观特征向量(不依赖真实ReID模型)验证核心逻辑：
  1. 丢失帧数不够(<MIN_GAP_FOR_RECOVERY)时不触发，交给正常IoU关联
  2. 丢失够久 + 外观相似度过阈值 -> 成功恢复，保住原track_id，不产生新track
  3. 外观相似度不够(不同的车) -> 不强行匹配，检测正常流入后续阶段(不会被错误合并)
  4. 类别不一致 -> 硬性拒绝，即使外观向量凑巧很像
"""
import sys
sys.path.insert(0, r"D:\yolo26-tracking")
import numpy as np
from types import SimpleNamespace

from ultralytics.trackers.bot_sort import BOTrack, BOTSORT, ReidRecoveryConfig
from ultralytics.trackers.basetrack import TrackState


def make_det(cx, cy, w=80, h=40, score=0.9, cls=0, idx=0, feat=None):
    xywh = np.array([cx, cy, w, h, idx], dtype=np.float32)
    return BOTrack(xywh, score, cls, feat=feat)


def rand_unit_vec(dim=512, seed=0):
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


args = SimpleNamespace(
    track_high_thresh=0.7, track_low_thresh=0.3, new_track_thresh=0.7,
    track_buffer=120, match_thresh=0.6, fuse_score=True,
    proximity_thresh=0.5, appearance_thresh=0.5, with_reid=True,
    model="auto", gmc_method='sparseOptFlow',
)


def fresh_bot():
    return BOTSORT(args=args)


print("=== 1. 丢失帧数不够，不触发第三轮 ===")
bot = fresh_bot()
feat_a = rand_unit_vec(seed=1)
track = make_det(100, 100, feat=feat_a)
track.activate(bot.kalman_filter, frame_id=1)
track.mark_lost()
bot.frame_id = 1 + (ReidRecoveryConfig.MIN_GAP_FOR_RECOVERY - 5)  # 故意不够门槛

det_same_car = make_det(100, 100, idx=0, feat=feat_a)  # 外观完全一样(同一辆车)
strack_pool = [track]
detections = [det_same_car]
u_track, u_detection = [0], [0]
activated, refind = [], []
new_u_track, new_u_detection = bot._post_first_association(strack_pool, detections, u_track, u_detection, activated, refind)
print(f"  gap不够时: refind={len(refind)}, 剩余u_track={new_u_track}, 剩余u_detection={new_u_detection}")
assert len(refind) == 0 and new_u_track == [0] and new_u_detection == [0], "gap不够却触发了恢复，门槛没生效"
print("  通过\n")


print("=== 2. 丢失够久 + 外观相似 -> 成功恢复，保住原track_id ===")
bot = fresh_bot()
track = make_det(100, 100, feat=feat_a)
track.activate(bot.kalman_filter, frame_id=1)
old_id = track.track_id
track.mark_lost()
bot.frame_id = 1 + ReidRecoveryConfig.MIN_GAP_FOR_RECOVERY + 10

det_same_car = make_det(400, 400, idx=0, feat=feat_a)  # 位置差很远(模拟弯道漂移)，但外观一致
strack_pool = [track]
detections = [det_same_car]
u_track, u_detection = [0], [0]
activated, refind = [], []
new_u_track, new_u_detection = bot._post_first_association(strack_pool, detections, u_track, u_detection, activated, refind)
print(f"  外观匹配时: refind track_id={[t.track_id for t in refind]}, 剩余u_track={new_u_track}, 剩余u_detection={new_u_detection}")
assert len(refind) == 1 and refind[0].track_id == old_id, "应该用原track_id恢复，不应该建新id"
assert new_u_track == [] and new_u_detection == []
assert refind[0].state == TrackState.Tracked
print("  通过（即使位置差很远，纯靠外观也能正确恢复，不依赖几何距离）\n")


print("=== 3. 外观不像(不同的车) -> 不强行匹配 ===")
bot = fresh_bot()
track = make_det(100, 100, feat=feat_a)
track.activate(bot.kalman_filter, frame_id=1)
track.mark_lost()
bot.frame_id = 1 + ReidRecoveryConfig.MIN_GAP_FOR_RECOVERY + 10

feat_b = rand_unit_vec(seed=999)  # 完全不同的随机向量，模拟另一辆车
det_diff_car = make_det(105, 105, idx=0, feat=feat_b)  # 位置很近(模拟"瓶颈弯道旁边恰好有另一辆车")
strack_pool = [track]
detections = [det_diff_car]
u_track, u_detection = [0], [0]
activated, refind = [], []
new_u_track, new_u_detection = bot._post_first_association(strack_pool, detections, u_track, u_detection, activated, refind)
print(f"  外观不匹配时: refind={len(refind)}, 剩余u_track={new_u_track}, 剩余u_detection={new_u_detection}")
assert len(refind) == 0 and new_u_track == [0] and new_u_detection == [0], \
    "外观明显不像却被强行合并了，这正是之前流场方案翻车的问题"
print("  通过（即使位置很近，外观不像也不会被错误合并）\n")


print("=== 4. 类别不一致 -> 硬性拒绝 ===")
bot = fresh_bot()
track = make_det(100, 100, cls=0, feat=feat_a)  # car
track.activate(bot.kalman_filter, frame_id=1)
track.mark_lost()
bot.frame_id = 1 + ReidRecoveryConfig.MIN_GAP_FOR_RECOVERY + 10

det_truck = make_det(100, 100, cls=1, idx=0, feat=feat_a)  # truck，外观向量特意设成完全一样
strack_pool = [track]
detections = [det_truck]
u_track, u_detection = [0], [0]
activated, refind = [], []
new_u_track, new_u_detection = bot._post_first_association(strack_pool, detections, u_track, u_detection, activated, refind)
print(f"  类别不一致时: refind={len(refind)}")
assert len(refind) == 0, "类别都不一致(car vs truck)不应该被匹配，哪怕外观向量一样"
print("  通过\n")


print("=== 5. 外观相似但位置离谱(复现真实bug场景：僵尸track在画面另一头) -> 不应恢复 ===")
bot = fresh_bot()
track = make_det(662.9, 1977.4, feat=feat_a)  # 复现实际抓到的bug: track#63最后位置(662.9,1977.4)
track.activate(bot.kalman_filter, frame_id=1)
track.mark_lost()
gap = ReidRecoveryConfig.MIN_GAP_FOR_RECOVERY  # 刚好卡在门槛，最容易被误伤的情况
bot.frame_id = 1 + gap

det_far_away = make_det(7.9, 1246.6, idx=0, feat=feat_a)  # 真实bug里新入场车辆的位置，外观向量故意设成一样
dist_px = ((662.9 - 7.9) ** 2 + (1977.4 - 1246.6) ** 2) ** 0.5
print(f"  距离={dist_px:.0f}px, gap={gap}帧, 允许的最大距离={gap * ReidRecoveryConfig.MAX_SPEED_PX_PER_FRAME:.0f}px")
strack_pool = [track]
detections = [det_far_away]
u_track, u_detection = [0], [0]
activated, refind = [], []
new_u_track, new_u_detection = bot._post_first_association(strack_pool, detections, u_track, u_detection, activated, refind)
print(f"  外观相似但距离离谱时: refind={len(refind)}, 剩余u_track={new_u_track}, 剩余u_detection={new_u_detection}")
assert len(refind) == 0 and new_u_track == [0] and new_u_detection == [0], \
    "外观相似但物理距离远超合理范围，不应该被恢复——这正是实测抓到的真实bug"
print("  通过（即使外观向量完全一样，物理距离不合理也会被拒绝）\n")

print("=== 6. 靠近画面边缘且速度朝外(真的在驶出画面) -> 不应恢复 ===")
bot = fresh_bot()
bot.frame_shape = (1988, 3534)  # (h, w)，模拟真实视频分辨率
track = make_det(50, 1000, feat=feat_a)  # 靠近左边缘(margin=100)
track.activate(bot.kalman_filter, frame_id=1)
track.mark_lost()
track.last_observed_mean[4] = -5.0  # vx 朝左(朝最近的左边界方向)，明显超过EDGE_EXIT_MIN_SPEED
track.last_observed_mean[5] = 0.0
bot.frame_id = 1 + ReidRecoveryConfig.MIN_GAP_FOR_RECOVERY + 10

det_same_car = make_det(60, 1005, idx=0, feat=feat_a)  # 外观一样，位置也很近，仅方向不同
strack_pool = [track]
detections = [det_same_car]
u_track, u_detection = [0], [0]
activated, refind = [], []
new_u_track, new_u_detection = bot._post_first_association(strack_pool, detections, u_track, u_detection, activated, refind)
print(f"  靠边缘+朝外驶出时: refind={len(refind)}, 剩余u_track={new_u_track}, 剩余u_detection={new_u_detection}")
assert len(refind) == 0 and new_u_track == [0] and new_u_detection == [0], \
    "track丢失前正朝画面边缘驶出，大概率已经开走了，不应该被当成僵尸恢复候选"
print("  通过（即使外观和距离都合理，朝画面外驶出的track也会被排除）\n")


print("=== 7. 靠近画面边缘但速度朝内/平行(刚驶入或原地丢失) -> 仍应正常恢复 ===")
bot = fresh_bot()
bot.frame_shape = (1988, 3534)
track = make_det(50, 1000, feat=feat_a)  # 同样靠近左边缘
track.activate(bot.kalman_filter, frame_id=1)
track.mark_lost()
track.last_observed_mean[4] = 5.0  # vx 朝右(朝画面内部)，说明是刚驶入或在往内走
track.last_observed_mean[5] = 0.0
bot.frame_id = 1 + ReidRecoveryConfig.MIN_GAP_FOR_RECOVERY + 10

det_same_car = make_det(60, 1005, idx=0, feat=feat_a)
strack_pool = [track]
detections = [det_same_car]
u_track, u_detection = [0], [0]
activated, refind = [], []
new_u_track, new_u_detection = bot._post_first_association(strack_pool, detections, u_track, u_detection, activated, refind)
print(f"  靠边缘+朝内运动时: refind track_id={[t.track_id for t in refind]}, 剩余u_track={new_u_track}, 剩余u_detection={new_u_detection}")
assert len(refind) == 1 and new_u_track == [] and new_u_detection == [], \
    "track虽然靠边缘丢失，但丢失前是朝画面内部运动的(不是在驶出)，应该正常按外观恢复"
print("  通过（同样靠边缘，但方向朝内时不受影响，正常恢复）\n")

print("=== 8. 外观/距离都合理，但尺寸差异很大(不同大小的车) -> 不应恢复 ===")
bot = fresh_bot()
track = make_det(400, 400, w=80, h=40, feat=feat_a)  # 正常轿车尺寸
track.activate(bot.kalman_filter, frame_id=1)
track.mark_lost()
bot.frame_id = 1 + ReidRecoveryConfig.MIN_GAP_FOR_RECOVERY + 10

det_big_car = make_det(405, 405, w=160, h=80, idx=0, feat=feat_a)  # 尺寸整整大一倍，外观向量故意设成一样
strack_pool = [track]
detections = [det_big_car]
u_track, u_detection = [0], [0]
activated, refind = [], []
new_u_track, new_u_detection = bot._post_first_association(strack_pool, detections, u_track, u_detection, activated, refind)
print(f"  尺寸差异大时: refind={len(refind)}, 剩余u_track={new_u_track}, 剩余u_detection={new_u_detection}")
assert len(refind) == 0 and new_u_track == [0] and new_u_detection == [0], \
    "外观和距离都合理，但车辆物理尺寸差异很大，大概率不是同一辆车，不应该被恢复"
print("  通过（即使外观向量完全一样，尺寸对不上也会被拒绝）\n")


print("=== 9. 外观/距离合理，尺寸也基本一致(仅有正常检测噪声) -> 仍应正常恢复 ===")
bot = fresh_bot()
track = make_det(400, 400, w=80, h=40, feat=feat_a)
track.activate(bot.kalman_filter, frame_id=1)
old_id = track.track_id
track.mark_lost()
bot.frame_id = 1 + ReidRecoveryConfig.MIN_GAP_FOR_RECOVERY + 10

det_same_size_car = make_det(405, 405, w=90, h=45, idx=0, feat=feat_a)  # 尺寸略有出入，正常检测噪声范围
strack_pool = [track]
detections = [det_same_size_car]
u_track, u_detection = [0], [0]
activated, refind = [], []
new_u_track, new_u_detection = bot._post_first_association(strack_pool, detections, u_track, u_detection, activated, refind)
print(f"  尺寸基本一致时: refind track_id={[t.track_id for t in refind]}, 剩余u_track={new_u_track}, 剩余u_detection={new_u_detection}")
assert len(refind) == 1 and refind[0].track_id == old_id and new_u_track == [] and new_u_detection == [], \
    "尺寸只是正常检测噪声范围内的出入，不应该被尺寸门槛误伤"
print("  通过（正常的尺寸检测噪声不受影响，仍正常恢复）\n")

print("全部测试通过。")
