#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
obb_tracking_with_roi_full_integration.py

在原始跟踪脚本基础上，集成了两种 PRE-MASK 策略：
- black: ROI 外像素置黑（原始行为）
- blur : 使用整图高斯模糊作为背景，然后用羽化 alpha 在 ROI 边缘做平滑过渡（推荐）

你可以通过修改脚本顶部的 PRE_MASK_DETECTION（是否启用 pre-mask）和 PRE_MASK_MODE
（'black' 或 'blur'）来切换行为。同时提供 PRE_MASK_FEATHER & PRE_MASK_BLUR_KSIZE 两个参数用于调节羽化和模糊核大小。

其余逻辑（ReID 加载、BOTSORT、速度/加速度计算、CSV 输出等）均保持不变。
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import time
import numpy as np
from ultralytics import YOLO
from types import SimpleNamespace
import csv
import re
import json

# Optional imports for ReID (will handle absence gracefully)
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import models
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# --------- 图像预处理函数 ---------
def apply_clahe(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    # clipLimit 2.0→3.5：更强的局部对比度，黑色车顶与路面区分更明显
    # tileGridSize (8,8)→(4,4)：更细粒度的局部均衡，适合小目标
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced

def gamma_transform(image, gamma=1.2):
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    return cv2.LUT(image, lookUpTable)

# --------- PRE-MASK 辅助函数：高斯模糊背景填充与羽化 ---------
def feather_alpha_from_mask(mask, feather_pixels=31):
    """从二值 mask 生成羽化后的 alpha（浮点 0.0-1.0）"""
    if feather_pixels <= 0:
        return (mask.astype(np.float32) / 255.0)
    alpha = (mask.astype(np.float32) / 255.0)
    k = feather_pixels
    if k % 2 == 0:
        k += 1
    alpha_blur = cv2.GaussianBlur(alpha, (k, k), 0)
    alpha_blur = np.clip(alpha_blur, 0.0, 1.0)
    return alpha_blur

def apply_blurred_background_fill(frame, roi_mask, feather=51, blur_ksize=101):
    """
    用整图的高斯模糊版本在 ROI 外填充，并对边缘做羽化融合（不使用均值填充）。
    参数：
      - frame: 原始 BGR 图像 uint8
      - roi_mask: 单通道 0/255 mask
      - feather: 羽化核大小（奇数）
      - blur_ksize: 生成模糊背景的高斯核大小（应为奇数）
    返回：合成后的 BGR 图像 uint8
    """
    assert frame.ndim == 3
    h, w = roi_mask.shape[:2]
    if frame.shape[0] != h or frame.shape[1] != w:
        raise ValueError("frame 和 roi_mask 尺寸不匹配")

    # 保证核为合理奇数并且不超过图像最小边
    min_dim = min(h, w)
    bk = int(blur_ksize)
    if bk % 2 == 0:
        bk += 1
    if bk > min_dim - 1:
        bk = min_dim - 1
        if bk % 2 == 0:
            bk -= 1
        bk = max(3, bk)

    # 1) 生成模糊背景（对整图进行大核高斯模糊）
    blurred = cv2.GaussianBlur(frame, (bk, bk), 0)

    # 2) 生成羽化 alpha（mask 的软边）
    fk = int(feather)
    if fk % 2 == 0:
        fk += 1
    if fk > min_dim - 1:
        fk = min_dim - 1
        if fk % 2 == 0:
            fk -= 1
        fk = max(3, fk)
    alpha = (roi_mask.astype(np.float32) / 255.0)
    alpha = cv2.GaussianBlur(alpha, (fk, fk), 0)[..., None]  # HxWx1

    # 3) 合成：ROI 内趋向原图，外部趋向模糊图；边缘平滑过渡
    out = (frame.astype(np.float32) * alpha + blurred.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
    return out

# --------- 配置（按需修改） ---------
VIDEO_PATH = r"E:\DJI\cvat\0111_D_stabilized.avi"
MODEL_PATH = r"D:\yolo26-tracking\runs\obb\exp_yolo26-3\weights\best.pt"   # YOLO26 OBB 自训练权重(data17数据集)
# ROI 持久化文件路径（与视频同目录，同文件名 + _roi.json）
ROI_FILE = os.path.splitext(VIDEO_PATH)[0] + "_roi.json"
# OSNet AIN ReID 模型路径
REID_MODEL_PATH = r"D:\deep-person-reid-master\scripts\log\osnet_ain_x1_0_market1501_softmax_cosinelr\model\model.pth.tar-100"  # 置空以关闭 ReID

timestamp = time.strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f"outputs/yolo26-obb_{timestamp}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ROI / mask behavior configuration:
# 如果希望把 ROI 之外区域在送入检测器前做预处理，设置 PRE_MASK_DETECTION = True
PRE_MASK_DETECTION = True

# PRE_MASK_MODE: 'black' 或 'blur'
# - 'black' : ROI 外置黑（原始行为）
# - 'blur'  : 使用整图高斯模糊作为背景，并在边缘做羽化融合（推荐）
PRE_MASK_MODE = 'blur'  # 可选 'black' 或 'blur'

# 如果 PRE_MASK_MODE == 'blur'，可以调节下面两个参数：
PRE_MASK_FEATHER = 51      # 羽化半径（像素，奇数推荐）
PRE_MASK_BLUR_KSIZE = 101   # 用于生成模糊背景的大核高斯（奇数，越大越模糊）

# 如果在 post-filter 判定阶段遇到无法判断的检测（如没有 mask/无法解析 polygon/bbox），是否严格丢弃
# True -> 丢弃（更严格，可能丢掉一些无法解析的真目标）
# False -> 保留（宽松，与你原始脚本默认行为更接近）
ROI_STRICT_FILTER = False

# tracker 参数·
# [改进1-recall] track_low_thresh 0.5→0.3：扩大 BYTE 第二次关联的低置信区间，减少漏检
# [改进2-frag]  track_buffer 60→120：lost 轨迹保留 4 秒（原 2 秒），给遮挡车辆更多重匹配机会
# [改进2-frag]  match_thresh 0.5→0.6：对 lost 轨迹重激活更宽容，减少轨迹
tracker_args = {
    'track_high_thresh': 0.7,
    'track_low_thresh': 0.3,
    'new_track_thresh': 0.7,
    'track_buffer': 120,
    'match_thresh': 0.6,
    'fuse_score': True,
    'proximity_thresh': 0.5,
    'appearance_thresh': 0.5,
    'with_reid': False,
    'model': None,
    'gmc_method': 'sparseOptFlow',
}
args = SimpleNamespace(**tracker_args)

from ultralytics.trackers.bot_sort import BOTSORT

CLASS_NAMES = ["car", "truck", "bus"]

# ── 截断感知 NMM / 全局跨类别 NMS 的参数 ──────────────────────────────────
# 车辆长边中位数先验（来自本项目 GT 统计：car=72.5, truck=90.6, bus=192.0px，
# 换了新视频/新数据集建议重新统计一遍再改这里）
CLASS_MEDIAN_LONG = {0: 72.5, 1: 90.6, 2: 192.0}
EDGE_MARGIN = 3            # 贴边判定容差（像素）
EDGE_CONF_PENALTY = 0.15   # 贴边+尺寸偏小时，从置信度里扣掉的惩罚（用于NMM排序）
SIZE_RATIO_THR = 0.7       # 长边 < 该类别中位数 * 此比例，视为"明显偏小"
REJECT_CONF_THR_INTERNAL = 0.6  # 贴切片内部切线（航拍无遮挡，必然不完整）：调整后置信度低于此值直接丢弃
REJECT_CONF_THR_BORDER = 0.4    # 贴原图真实边界（可能本来就只露出一部分）：阈值放宽一些

# 碎片拼接：只对贴"内部切线"的碎片生效
STITCH_IOS_MAX = 0.5          # 两个碎片的IOS必须小于这个值（否则NMM本来就会合并，不需要拼接）
STITCH_ANGLE_TOL_DEG = 15     # 两个碎片角度差容忍度
STITCH_PERP_OVERLAP_MIN = 0.6  # 垂直于切割方向的重叠比例下限
STITCH_GAP_TOL_PX = 30         # 沿切割方向，两碎片"切口"之间允许的最大缝隙/重叠（像素）
STITCH_SIZE_RATIO_LO = 0.7    # 拼接后长边 / 该类别中位数，允许范围下限
STITCH_SIZE_RATIO_HI = 1.4    # 拼接后长边 / 该类别中位数，允许范围上限

GLOBAL_NMS_IOS_THR = 0.5   # 全局(跨类别)NMS的IOS阈值：航拍俯视下车辆框不应重合，超过此值判定为同一物体

OPPOSITE_EDGE = {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}

# ── 切片坐标生成：直接用 SAHI 官方实现（原先是手写的等价版本，现在改为真依赖）──
from sahi.slicing import get_slice_bboxes as _sahi_get_slice_bboxes

def get_slice_bboxes(H, W, sh=640, sw=640, oh=0.3, ow=0.3):
    return _sahi_get_slice_bboxes(
        image_height=H, image_width=W,
        slice_height=sh, slice_width=sw,
        auto_slice_resolution=False,  # sh/sw 都已显式传入，不需要自动推断
        overlap_height_ratio=oh, overlap_width_ratio=ow,
    )

# ── 截断检测：区分"贴原图真实边界"还是"贴切片内部人为切线" ──────────────
def _touched_edges(xywhr, slice_box, margin=EDGE_MARGIN):
    x1, y1, x2, y2 = slice_box
    pts = cv2.boxPoints(((float(xywhr[0]), float(xywhr[1])), (float(xywhr[2]), float(xywhr[3])), float(np.rad2deg(xywhr[4]))))
    edges = set()
    if pts[:, 0].min() <= x1 + margin:
        edges.add("left")
    if pts[:, 0].max() >= x2 - margin:
        edges.add("right")
    if pts[:, 1].min() <= y1 + margin:
        edges.add("top")
    if pts[:, 1].max() >= y2 - margin:
        edges.add("bottom")
    return edges

def _classify_edges(slice_box, edges, margin=EDGE_MARGIN):
    x1, y1, x2, y2 = slice_box
    border, internal = set(), set()
    for e in edges:
        is_border = (
            (e == "left" and x1 <= margin) or
            (e == "right" and x2 >= frame_width - margin) or
            (e == "top" and y1 <= margin) or
            (e == "bottom" and y2 >= frame_height - margin)
        )
        (border if is_border else internal).add(e)
    return border, internal

def _truncation_penalty(d):
    edges = _touched_edges(d['xywhr'], d['slice'])
    border_edges, internal_edges = _classify_edges(d['slice'], edges)
    long_side = max(d['xywhr'][2], d['xywhr'][3])
    median = CLASS_MEDIAN_LONG.get(d['cls'], 80.0)
    size_ratio = long_side / median
    undersized = size_ratio < SIZE_RATIO_THR

    is_trunc_internal = bool(internal_edges) and undersized
    is_trunc_border = bool(border_edges) and not internal_edges and undersized
    is_trunc = is_trunc_internal or is_trunc_border

    d['_border_edges'], d['_internal_edges'] = border_edges, internal_edges
    return (EDGE_CONF_PENALTY if is_trunc else 0.0), bool(edges), size_ratio, is_trunc, is_trunc_internal

def _annotate_truncation(dets):
    out = []
    for d in dets:
        d = dict(d)
        pen, et, sr, is_trunc, is_trunc_internal = _truncation_penalty(d)
        d['_penalty'], d['_edge_touch'], d['_size_ratio'] = pen, et, sr
        d['_is_trunc'], d['_is_trunc_internal'] = is_trunc, is_trunc_internal
        out.append(d)
    return out

def _reject_fragments(dets):
    """截断碎片硬性丢弃：贴内部切线的（航拍无遮挡，必然不完整）用更严阈值；贴真实边界的用更宽松阈值。"""
    out = []
    for d in dets:
        if d['_is_trunc']:
            thr = REJECT_CONF_THR_INTERNAL if d['_is_trunc_internal'] else REJECT_CONF_THR_BORDER
            if (d['conf'] - d['_penalty']) < thr:
                continue
        out.append(d)
    return out

# ── 碎片拼接：把同一条内部切线两侧、几何上严丝合缝的两个碎片拼回一个完整框 ──
def _to_poly(xywhr):
    cx, cy, w, h, r = xywhr
    pts = cv2.boxPoints(((float(cx), float(cy)), (float(w), float(h)), float(np.rad2deg(r))))
    from shapely.geometry import Polygon
    return Polygon(pts)

def _merge_by_corner_union(xywhr1, xywhr2):
    pts1 = cv2.boxPoints(((float(xywhr1[0]), float(xywhr1[1])), (float(xywhr1[2]), float(xywhr1[3])), float(np.rad2deg(xywhr1[4]))))
    pts2 = cv2.boxPoints(((float(xywhr2[0]), float(xywhr2[1])), (float(xywhr2[2]), float(xywhr2[3])), float(np.rad2deg(xywhr2[4]))))
    all_pts = np.vstack([pts1, pts2]).astype(np.float32)
    (cx, cy), (w, h), r = cv2.minAreaRect(all_pts)
    return np.array([cx, cy, w, h, np.deg2rad(r)], dtype=np.float32)

def _find_seam_axis(det_i, det_j):
    for e_i in det_i['_internal_edges']:
        e_j_wanted = OPPOSITE_EDGE[e_i]
        if e_j_wanted not in det_j['_internal_edges']:
            continue
        if e_i in ("left", "right"):
            if e_i == "right" and det_j['xywhr'][0] > det_i['xywhr'][0]:
                return "x"
            if e_i == "left" and det_j['xywhr'][0] < det_i['xywhr'][0]:
                return "x"
        else:
            if e_i == "bottom" and det_j['xywhr'][1] > det_i['xywhr'][1]:
                return "y"
            if e_i == "top" and det_j['xywhr'][1] < det_i['xywhr'][1]:
                return "y"
    return None

def _perpendicular_overlap_ratio(pts_i, pts_j, axis):
    dim = 1 if axis == "x" else 0
    lo = max(pts_i[:, dim].min(), pts_j[:, dim].min())
    hi = min(pts_i[:, dim].max(), pts_j[:, dim].max())
    overlap = max(0.0, hi - lo)
    span = min(pts_i[:, dim].max() - pts_i[:, dim].min(), pts_j[:, dim].max() - pts_j[:, dim].min())
    return overlap / span if span > 0 else 0.0

def _split_gap(pts_i, pts_j, axis):
    dim = 0 if axis == "x" else 1
    i_lo, i_hi = pts_i[:, dim].min(), pts_i[:, dim].max()
    j_lo, j_hi = pts_j[:, dim].min(), pts_j[:, dim].max()
    if i_hi <= j_lo:
        return j_lo - i_hi
    if j_hi <= i_lo:
        return i_lo - j_hi
    return -min(i_hi - j_lo, j_hi - i_lo)

def _stitch_fragments(dets):
    n = len(dets)
    used = [False] * n
    stitched = []
    for i in range(n):
        if used[i] or not dets[i]['_internal_edges']:
            continue
        for j in range(i + 1, n):
            if used[j] or dets[j]['cls'] != dets[i]['cls'] or not dets[j]['_internal_edges']:
                continue
            axis = _find_seam_axis(dets[i], dets[j])
            if axis is None:
                continue
            angle_diff = abs(((dets[i]['xywhr'][4] - dets[j]['xywhr'][4] + np.pi) % (2 * np.pi)) - np.pi)
            if np.rad2deg(angle_diff) > STITCH_ANGLE_TOL_DEG:
                continue
            pts_i = cv2.boxPoints(((float(dets[i]['xywhr'][0]), float(dets[i]['xywhr'][1])),
                                    (float(dets[i]['xywhr'][2]), float(dets[i]['xywhr'][3])),
                                    float(np.rad2deg(dets[i]['xywhr'][4]))))
            pts_j = cv2.boxPoints(((float(dets[j]['xywhr'][0]), float(dets[j]['xywhr'][1])),
                                    (float(dets[j]['xywhr'][2]), float(dets[j]['xywhr'][3])),
                                    float(np.rad2deg(dets[j]['xywhr'][4]))))
            if _perpendicular_overlap_ratio(pts_i, pts_j, axis) < STITCH_PERP_OVERLAP_MIN:
                continue
            if abs(_split_gap(pts_i, pts_j, axis)) > STITCH_GAP_TOL_PX:
                continue
            merged_xywhr = _merge_by_corner_union(dets[i]['xywhr'], dets[j]['xywhr'])
            long_side = max(merged_xywhr[2], merged_xywhr[3])
            median = CLASS_MEDIAN_LONG.get(dets[i]['cls'], 80.0)
            ratio = long_side / median
            if not (STITCH_SIZE_RATIO_LO <= ratio <= STITCH_SIZE_RATIO_HI):
                continue
            used[i] = used[j] = True
            new_det = dict(dets[i])
            new_det['xywhr'] = merged_xywhr
            new_det['conf'] = max(dets[i]['conf'], dets[j]['conf'])
            new_det['slice'] = (-1e9, -1e9, 1e9, 1e9)  # 拼接后视为完整框，不会再被误判成截断
            stitched.append(new_det)
            break
    remaining = [d for k, d in enumerate(dets) if not used[k]]
    return remaining, stitched

# ── 第一阶段：GREEDYNMM ──────────────────────────────────────────────────
# 匹配：OBB polygon IOS（交集/较小框面积）> ios_thr
# class_agnostic=True 时不管类别是否相同都参与合并——用于全局跨类别去重
def _obb_nmm(dets, ios_thr=0.5, truncation_aware=False, class_agnostic=False):
    if not dets:
        return []
    dets = [dict(d) for d in dets]
    if truncation_aware:
        for d in dets:
            pen, et, sr, is_trunc, is_trunc_internal = _truncation_penalty(d)
            d['_penalty'], d['_edge_touch'], d['_size_ratio'] = pen, et, sr
            d['_is_trunc'], d['_is_trunc_internal'] = is_trunc, is_trunc_internal
        dets.sort(key=lambda d: -(d['conf'] - d['_penalty']))
    else:
        dets.sort(key=lambda d: -d['conf'])

    polys = [_to_poly(d['xywhr']) for d in dets]
    merged = [False] * len(dets)
    result = []

    for i in range(len(dets)):
        if merged[i]:
            continue
        cur_cls     = dets[i]['cls']
        anchor_poly = polys[i]
        group       = [i]   # 同一物体的重复候选组，用于之后挑"最完整"的框

        for j in range(i + 1, len(dets)):
            if merged[j] or (not class_agnostic and dets[j]['cls'] != cur_cls):
                continue
            inter    = anchor_poly.intersection(polys[j]).area
            min_area = min(anchor_poly.area, polys[j].area)
            if min_area > 0 and inter / min_area > ios_thr:
                merged[j] = True   # 抑制重叠框，标记为组内成员
                group.append(j)

        # 组内保留面积最大（最完整）的框，而不是置信度最高的：
        # 车身被干扰（车道标线/箭头等）导致的碎片检测，置信度有时反而高于完整检测，
        # 单纯按置信度选锚点会让残缺框顶掉完整框（video2 GT283 id594 案例已实测验证）。
        best = max(group, key=lambda k: polys[k].area)
        result.append({
            'xywhr': dets[best]['xywhr'].copy(),
            'conf':  float(dets[best]['conf']),
            'cls':   cur_cls,
        })
    return result

# ── 第二阶段：NMS（清除 NMM 后残余重复，与 SAHI 第二级一致）──────────────
def _rotated_nms(dets, iou_thr=0.2, class_agnostic=False):
    if not dets:
        return []
    from ultralytics.utils.metrics import batch_probiou
    import torch
    dets = sorted(dets, key=lambda d: -d['conf'])
    xywhr_arr = np.array([d['xywhr'] for d in dets], dtype=np.float32)
    t = torch.from_numpy(xywhr_arr)
    iou_mat = batch_probiou(t, t).numpy()
    suppressed = [False] * len(dets)
    keep = []
    for i, d in enumerate(dets):
        if suppressed[i]:
            continue
        keep.append(d)
        for j in range(i + 1, len(dets)):
            if suppressed[j] or (not class_agnostic and dets[j]['cls'] != d['cls']):
                continue
            if iou_mat[i, j] > iou_thr:
                suppressed[j] = True
    return keep

# ── 第三阶段：全局跨类别 NMS（航拍俯视车辆框不应重合，抓 car/truck/bus 反复摇摆产生的重复框）──
def _global_cross_class_nms(dets):
    return _obb_nmm(dets, ios_thr=GLOBAL_NMS_IOS_THR, truncation_aware=False, class_agnostic=True)

def _full_detection_pipeline(raw_dets, nms_iou_thr=0.2):
    """截断感知完整流程：标注截断 -> 拼接碎片对 -> 硬性丢弃低质量碎片
    -> NMM(截断感知，同类别) -> NMS(同类别) -> 全局NMS(跨类别)。
    """
    annotated = _annotate_truncation(raw_dets)
    remaining, stitched = _stitch_fragments(annotated)
    cleaned = _reject_fragments(remaining) + stitched
    merged = _obb_nmm(cleaned, ios_thr=0.5, truncation_aware=True)
    nms_result = _rotated_nms(merged, iou_thr=nms_iou_thr)
    return _global_cross_class_nms(nms_result)

# ── 自定义切片推理：直接读 YOLO 原始 xywhr，返回原始检测列表（不含 NMM/NMS）──
def sliced_predict_obb_direct(frame, yolo_model, roi_mask=None,
                               slice_h=640, slice_w=640,
                               overlap_h=0.3, overlap_w=0.3,
                               conf_thr=0.3):
    H, W = frame.shape[:2]
    slices = get_slice_bboxes(H, W, slice_h, slice_w, overlap_h, overlap_w)
    all_dets = []
    for (x1, y1, x2, y2) in slices:
        # ROI 感知：跳过与 ROI 完全无交的切片
        if roi_mask is not None and roi_mask[y1:y2, x1:x2].max() == 0:
            continue
        crop = frame[y1:y2, x1:x2]
        # imgsz=slice_h：显式指定原生分辨率推理，避免被隐式压缩到默认640导致细节丢失
        res = yolo_model.predict(crop, imgsz=slice_h, conf=conf_thr, verbose=False)
        if not res or res[0].obb is None or len(res[0].obb) == 0:
            continue
        obb = res[0].obb
        xywhr = obb.xywhr.cpu().numpy().copy()   # (N,5) cx cy w h angle_rad
        xywhr[:, 0] += x1                        # 切片坐标 → 全图坐标
        xywhr[:, 1] += y1
        confs = obb.conf.cpu().numpy()
        clses = obb.cls.cpu().numpy().astype(int)
        for i in range(len(xywhr)):
            w_box, h_box = float(xywhr[i, 2]), float(xywhr[i, 3])
            if min(w_box, h_box) < 20:          # 真实车辆min(w,h)≈37-50px，20px以下均为噪声ghost
                continue
            cx, cy = float(xywhr[i, 0]), float(xywhr[i, 1])
            if roi_mask is not None:
                ix = int(np.clip(cx, 0, W - 1))
                iy = int(np.clip(cy, 0, H - 1))
                if roi_mask[iy, ix] == 0:
                    continue
            all_dets.append({
                'xywhr': xywhr[i], 'conf': float(confs[i]), 'cls': int(clses[i]),
                'slice': (x1, y1, x2, y2),
            })
    return all_dets

class OBBResultsAdapter:
    def __init__(self, dets):
        if not dets:
            self.xywhr = np.zeros((0, 5), dtype=np.float32)
            self.conf  = np.zeros(0,      dtype=np.float32)
            self.cls   = np.zeros(0,      dtype=np.int32)
        else:
            self.xywhr = np.array([d['xywhr'] for d in dets], dtype=np.float32)
            self.conf  = np.array([d['conf']  for d in dets], dtype=np.float32)
            self.cls   = np.array([d['cls']   for d in dets], dtype=np.int32)

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, idx):
        sub = OBBResultsAdapter.__new__(OBBResultsAdapter)
        sub.xywhr = self.xywhr[idx]
        sub.conf = self.conf[idx]
        sub.cls = self.cls[idx]
        return sub

    @property
    def xyxy(self):
        """轴对齐外接矩形（供 GMC 等只需要 xyxy 的下游逻辑使用）。"""
        if len(self.xywhr) == 0:
            return np.zeros((0, 4), dtype=np.float32)
        boxes = np.zeros((len(self.xywhr), 4), dtype=np.float32)
        for i, (cx, cy, w, h, r) in enumerate(self.xywhr):
            pts = cv2.boxPoints(((float(cx), float(cy)), (float(w), float(h)), float(np.rad2deg(r))))
            boxes[i] = [pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()]
        return boxes


def load_roi_file(path):
    """从 JSON 文件加载 ROI，返回 (roi_polys_modes, pixel_length, real_length) 或 None。

    兼容两种格式：
    - 完整格式（本脚本 save_roi_file 写出）：{"pixel_length":.., "real_length":.., "roi_polys_modes":[{"pts":.., "mode":..}, ...]}
    - 纯 ROI 格式（外部工具导出，没有比例信息）：[{"pts":.., "mode":..}, ...]，此时 pixel_length/real_length 返回 None，
      调用方需要另外让用户输入比例。
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            items = data
            pixel_length = None
            real_length = None
        else:
            items = data["roi_polys_modes"]
            pixel_length = float(data["pixel_length"])
            real_length = float(data["real_length"])

        roi_polys_modes = [
            ([tuple(p) for p in item["pts"]], item["mode"])
            for item in items
        ]

        if pixel_length is not None:
            print(f"[ROI] 已从文件加载 ROI（{len(roi_polys_modes)} 个区域），比例 {pixel_length:.2f}px = {real_length} 米")
        else:
            print(f"[ROI] 已从文件加载 ROI（{len(roi_polys_modes)} 个区域），文件中没有比例信息，需要手动输入")
        return roi_polys_modes, pixel_length, real_length
    except Exception as e:
        print(f"[ROI] 加载失败: {e}")
        return None


def multi_roi_mask(polys_with_mode, shape):
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for pts, mode in polys_with_mode:
        arr = np.array(pts, dtype=np.int32)
        if mode == 1:
            cv2.fillPoly(mask, [arr], 255)
    # 对mode==2的区域直接挖空/去掉
    for pts, mode in polys_with_mode:
        arr = np.array(pts, dtype=np.int32)
        if mode == 2:
            cv2.fillPoly(mask, [arr], 0)
    return mask
# --------- OSNet AIN ReID 加载 ---------
REID_AVAILABLE = False
ReIDModelObj   = None
ReIDEncoderObj = None

if TORCH_AVAILABLE and REID_MODEL_PATH and os.path.isfile(REID_MODEL_PATH):
    try:
        import sys as _sys, types as _types
        import numpy as _np
        import numpy.core as _np_core
        import numpy.core.multiarray as _np_ma

        # numpy 2.x 把 numpy._core 作为内部模块，1.x 没有；
        # 用 fake module 做兼容垫片，把 2.x checkpoint 里的引用重定向到 1.x 对应位置
        _fake = _types.ModuleType('numpy._core')
        _fake.__dict__.update(_np_core.__dict__)
        _fake.scalar   = _np_ma.scalar        # numpy._core.scalar  → numpy.core.multiarray.scalar
        _fake.ndarray  = _np.ndarray
        _fake.dtype    = _np.dtype
        _sys.modules['numpy._core']            = _fake
        _sys.modules['numpy._core.multiarray'] = _np_ma
        _sys.modules.setdefault('numpy._core.numeric', _np_core)
        _sys.modules.setdefault('numpy._core.umath',   _np_core)

        _torchreid_root = r"D:\deep-person-reid-master"
        if _torchreid_root not in _sys.path:
            _sys.path.insert(0, _torchreid_root)
        import torchreid

        DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'
        REID_IMG_H = 256
        REID_IMG_W = 128
        REID_BATCH = 32
        FEAT_DIM   = 512   # OSNet AIN x1.0 输出维度

        _reid_model = torchreid.models.build_model(
            name='osnet_ain_x1_0',
            num_classes=751,   # Market-1501
            loss='softmax',
            pretrained=False,
        )
        torchreid.utils.load_pretrained_weights(_reid_model, REID_MODEL_PATH)
        _reid_model = _reid_model.to(DEVICE).eval()
        print(f"[ReID] OSNet AIN x1.0 加载成功，设备: {DEVICE}")

        class ReIDEncoder:
            def __init__(self, model):
                self.model = model
                self.mean  = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                self.std   = np.array([0.229, 0.224, 0.225], dtype=np.float32)

            def _crop(self, img_bgr, box):
                H_img, W_img = img_bgr.shape[:2]
                if len(box) >= 5:
                    xc, yc, w, h, angle_rad = (float(box[0]), float(box[1]),
                                                float(box[2]), float(box[3]),
                                                float(box[4]))
                    # 只裁取车辆周围的局部小图，在局部图上旋转，避免旋转整张大图
                    pad   = int(max(w, h) * 0.8 + 4)
                    x1p   = max(0, int(xc) - pad)
                    y1p   = max(0, int(yc) - pad)
                    x2p   = min(W_img, int(xc) + pad)
                    y2p   = min(H_img, int(yc) + pad)
                    patch = img_bgr[y1p:y2p, x1p:x2p]
                    if patch.size == 0:
                        return None
                    # 车辆中心在 patch 内的局部坐标
                    lxc = xc - x1p
                    lyc = yc - y1p
                    ph, pw = patch.shape[:2]
                    M = cv2.getRotationMatrix2D((lxc, lyc), -np.rad2deg(angle_rad), 1.0)
                    rotated = cv2.warpAffine(patch, M, (pw, ph),
                                             flags=cv2.INTER_LINEAR,
                                             borderMode=cv2.BORDER_REFLECT_101)
                    rx1 = max(0, int(lxc - w / 2))
                    ry1 = max(0, int(lyc - h / 2))
                    rx2 = min(pw, int(lxc + w / 2))
                    ry2 = min(ph, int(lyc + h / 2))
                    if rx2 <= rx1 or ry2 <= ry1:
                        return None
                    return rotated[ry1:ry2, rx1:rx2]
                else:
                    x1, y1, x2, y2 = map(int, map(round, box[:4]))
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(W_img, x2), min(H_img, y2)
                    if x2 <= x1 or y2 <= y1:
                        return None
                    return img_bgr[y1:y2, x1:x2]

            def _preprocess(self, crop):
                im = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                im = cv2.resize(im, (REID_IMG_W, REID_IMG_H))
                im = im.astype(np.float32) / 255.0
                im = (im - self.mean) / self.std
                return np.transpose(im, (2, 0, 1))

            @torch.no_grad()
            def inference(self, img_bgr, dets):
                if dets is None or len(dets) == 0:
                    return np.zeros((0, FEAT_DIM), dtype=np.float32)
                crops = []
                for box in dets:
                    crop = self._crop(img_bgr, box)
                    if crop is None or crop.size == 0:
                        crop = np.zeros((REID_IMG_H, REID_IMG_W, 3), dtype=np.uint8)
                    crops.append(self._preprocess(crop))
                tensor = torch.from_numpy(np.stack(crops)).float().to(DEVICE)
                feats  = []
                for i in range(0, len(tensor), REID_BATCH):
                    out = self.model(tensor[i:i + REID_BATCH])
                    if isinstance(out, (tuple, list)):
                        out = out[0]
                    out = F.normalize(out, p=2, dim=1)
                    feats.append(out.cpu().numpy())
                return np.concatenate(feats, axis=0)

            def __call__(self, img_bgr, dets):
                """BOTSORT/init_track 按 self.encoder(img, bboxes) 的约定直接调用这个对象。"""
                return self.inference(img_bgr, dets)

        ReIDEncoderObj = ReIDEncoder(_reid_model)
        ReIDModelObj   = _reid_model
        REID_AVAILABLE = True

    except Exception as e:
        print(f"[ReID] 加载失败: {e}，继续不使用 ReID")
        REID_AVAILABLE = False
else:
    print("[ReID] 模型文件未找到或 Torch 不可用，跳过 ReID")

# --------- 加载检测模型 ---------
yolo_model = YOLO(MODEL_PATH)
yolo_model.to("cuda:0")

cap = cv2.VideoCapture(VIDEO_PATH)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_path = os.path.join(OUTPUT_DIR, "obb_track_result.mp4")
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

csv_save_path = os.path.join(OUTPUT_DIR, "track_output.csv")
csv_file = open(csv_save_path, "w", newline='', encoding='utf-8')
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    "frame_id", "track_id", "class", "center_x", "center_y",
    "width", "height", "angle_deg", "confidence",
    "pt1_x", "pt1_y", "pt2_x", "pt2_y", "pt3_x", "pt3_y", "pt4_x", "pt4_y",
    "velocity_kmh", "acceleration_m_per_s2",
])

track_save_path = os.path.join(OUTPUT_DIR, "track_result.txt")
track_file = open(track_save_path, "w")

cvat_eval_path = os.path.join(OUTPUT_DIR, "cvat_eval.txt")
cvat_eval_file = open(cvat_eval_path, "w")

