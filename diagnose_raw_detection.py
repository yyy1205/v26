#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从原视频抽取指定帧，套用和 tracktworoi.py 完全一致的预处理 + 双尺度切片检测逻辑，
但只看 NMM/NMS 合并之前的原始检测框，并标注每个框来自哪个尺度（960/1280）的哪个切片。
用于判断 car/truck 重复检测是发生在“模型检测”这一步，还是后续合并/追踪阶段引入的。
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
import cv2
import numpy as np
import torch
from shapely.geometry import Polygon
from ultralytics import YOLO
from ultralytics.utils.metrics import batch_probiou
import pandas as pd

# ── 截断感知 NMM 的参数（先验来自 GT 统计的车辆长边中位数：car/truck/bus）──
CLASS_MEDIAN_LONG = {0: 72.5, 1: 90.6, 2: 192.0}
EDGE_MARGIN = 3          # 贴边判定容差（像素）
EDGE_CONF_PENALTY = 0.15  # 贴边+尺寸偏小时，从置信度里扣掉的惩罚
SIZE_RATIO_THR = 0.7     # 长边 < 该类别中位数 * 此比例，视为"明显偏小"

# 原图尺寸（用来区分"贴的是原图真实边界"还是"贴的是切片人为切出来的内部切线"）
FRAME_W, FRAME_H = 3534, 1988

# 贴切片内部切线：航拍俯视无遮挡，物理上车辆必然完整，只是被切片人为切开——判定应更严格
REJECT_CONF_THR_INTERNAL = 0.6
# 贴原图真实边界：车辆可能本来就只在画面里露出一部分（开进/开出拍摄范围），不强求完整
REJECT_CONF_THR_BORDER = 0.4

# ── 碎片拼接参数（只对贴"内部切线"的碎片生效，贴真实边界的不拼——没有另一半可拼）──
STITCH_IOS_MAX = 0.5          # 两个碎片的IOS必须小于这个值（否则NMM本来就会合并，不需要拼接）
STITCH_ANGLE_TOL_DEG = 15     # 两个碎片角度差容忍度
STITCH_PERP_OVERLAP_MIN = 0.6  # 垂直于切割方向上的重叠比例下限（确认两者在同一"高度/位置"上）
STITCH_GAP_TOL_PX = 30         # 沿切割方向，两碎片"切口"之间允许的最大缝隙/重叠（像素）
STITCH_SIZE_RATIO_LO = 0.7    # 拼接后长边 / 该类别中位数，允许范围下限
STITCH_SIZE_RATIO_HI = 1.4    # 拼接后长边 / 该类别中位数，允许范围上限

VIDEO_PATH = r"E:\DJI\cvat\0111_D_stabilized.avi"
MODEL_PATH = r"D:\yolo26-tracking\runs\obb\exp_yolo26-3\weights\best.pt"
ROI_FILE = r"E:\DJI\cvat\0111_D_stabilized_roi.json"
TRACK_TXT = r"D:\yolo26-tracking\outputs\yolo26-obb_20260722_003112\track_result.txt"
OUT_DIR = r"D:\yolo26-tracking\outputs\yolo26-obb_20260722_003112\idsw_diagnosis"

# 新逻辑跑完整视频后剩下的全部16次IDSW事件（gid, label, pred_frame=GT帧+1, x, y）
# 位置(x,y)直接用GT在该帧的中心坐标
_BOUNDARY_EVENTS = [
    (138, "car", 164, 417.1, 215.6),
    (46,  "car", 421, 1017.5, 14.2),
    (136, "car", 468, 1142.4, 36.4),
    (46,  "car", 490, 605.9, 104.9),
    (136, "car", 565, 596.3, 103.0),
    (150, "car", 577, 1009.4, 25.2),
    (136, "car", 629, 314.5, 280.2),
    (150, "car", 689, 357.7, 259.8),
    (198, "car", 894, 1067.6, 27.9),
    (198, "car", 1000, 469.4, 185.4),
    (218, "car", 1121, 608.0, 114.5),
    (218, "car", 1157, 408.7, 227.4),
    (275, "car", 1445, 1114.3, 35.8),
    (275, "car", 1539, 604.1, 119.1),
    (293, "car", 1650, 1105.3, 37.4),
    (313, "car", 1785, 1078.2, 38.1),
]
TARGET_FRAMES = [e[2] for e in _BOUNDARY_EVENTS]
_POS_BY_FRAME = {e[2]: (e[3], e[4]) for e in _BOUNDARY_EVENTS}
_GID_BY_FRAME = {e[2]: (e[0], e[1]) for e in _BOUNDARY_EVENTS}
FILTER_RADIUS = 150

_track_df = pd.read_csv(TRACK_TXT, header=None,
                         names=["frame_id", "track_id", "xc", "yc", "w", "h", "angle_rad", "score", "label", "v_kmh"])


def vehicle_xy_at(frame_id):
    if frame_id in _POS_BY_FRAME:
        return _POS_BY_FRAME[frame_id]
    sub = _track_df[_track_df.track_id == 154]
    exact = sub[sub.frame_id == frame_id]
    if len(exact):
        return float(exact.iloc[0].xc), float(exact.iloc[0].yc)
    sub = sub.copy()
    sub["d"] = (sub.frame_id - frame_id).abs()
    nearest = sub.sort_values("d").iloc[0]
    return float(nearest.xc), float(nearest.yc)

PRE_MASK_MODE = "blur"
PRE_MASK_FEATHER = 51
PRE_MASK_BLUR_KSIZE = 101


def apply_clahe(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


def gamma_transform(image, gamma=1.2):
    lut = np.empty((1, 256), np.uint8)
    for i in range(256):
        lut[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    return cv2.LUT(image, lut)


def apply_blurred_background_fill(frame, roi_mask, feather=51, blur_ksize=101):
    h, w = roi_mask.shape[:2]
    min_dim = min(h, w)
    bk = int(blur_ksize)
    if bk % 2 == 0:
        bk += 1
    bk = min(bk, min_dim - 1 if (min_dim - 1) % 2 == 1 else min_dim - 2)
    bk = max(3, bk)
    blurred = cv2.GaussianBlur(frame, (bk, bk), 0)
    fk = int(feather)
    if fk % 2 == 0:
        fk += 1
    fk = min(fk, min_dim - 1 if (min_dim - 1) % 2 == 1 else min_dim - 2)
    fk = max(3, fk)
    alpha = (roi_mask.astype(np.float32) / 255.0)
    alpha = cv2.GaussianBlur(alpha, (fk, fk), 0)[..., None]
    out = (frame.astype(np.float32) * alpha + blurred.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
    return out


def multi_roi_mask(polys_with_mode, shape):
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for pts, mode in polys_with_mode:
        arr = np.array(pts, dtype=np.int32)
        if mode == 1:
            cv2.fillPoly(mask, [arr], 255)
    for pts, mode in polys_with_mode:
        arr = np.array(pts, dtype=np.int32)
        if mode == 2:
            cv2.fillPoly(mask, [arr], 0)
    return mask


def get_slice_bboxes(H, W, sh=640, sw=640, oh=0.3, ow=0.3):
    y_overlap = int(oh * sh)
    x_overlap = int(ow * sw)
    bboxes = []
    y_min = 0
    while y_min < H:
        y_max = min(H, y_min + sh)
        y1 = max(0, y_max - sh)
        x_min = 0
        while x_min < W:
            x_max = min(W, x_min + sw)
            x1 = max(0, x_max - sw)
            bboxes.append((x1, y1, x_max, y_max))
            if x_max == W:
                break
            x_min = x_max - x_overlap
        if y_max == H:
            break
        y_min = y_max - y_overlap
    return bboxes


def sliced_predict_raw(frame, model, roi_mask, slice_h, slice_w, tag, overlap=0.3, conf_thr=0.3):
    H, W = frame.shape[:2]
    slices = get_slice_bboxes(H, W, slice_h, slice_w, overlap, overlap)
    dets = []
    for (x1, y1, x2, y2) in slices:
        if roi_mask is not None and roi_mask[y1:y2, x1:x2].max() == 0:
            continue
        crop = frame[y1:y2, x1:x2]
        res = model.predict(crop, imgsz=slice_h, conf=conf_thr, verbose=False)
        if not res or res[0].obb is None or len(res[0].obb) == 0:
            continue
        obb = res[0].obb
        xywhr = obb.xywhr.cpu().numpy().copy()
        xywhr[:, 0] += x1
        xywhr[:, 1] += y1
        confs = obb.conf.cpu().numpy()
        clses = obb.cls.cpu().numpy().astype(int)
        for i in range(len(xywhr)):
            dets.append({
                "xywhr": xywhr[i], "conf": float(confs[i]), "cls": int(clses[i]),
                "scale": tag, "slice": (x1, y1, x2, y2),
            })
    return dets


OPPOSITE_EDGE = {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}


def touched_edges(xywhr, slice_box, margin=EDGE_MARGIN):
    """返回这个框贴到了它所属切片crop的哪几条边（left/right/top/bottom 的子集）。"""
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


def classify_edges(slice_box, edges, margin=EDGE_MARGIN):
    """把贴到的边分成：贴原图真实边界的 vs 贴切片内部人为切线的。"""
    x1, y1, x2, y2 = slice_box
    border, internal = set(), set()
    for e in edges:
        is_border = (
            (e == "left" and x1 <= margin) or
            (e == "right" and x2 >= FRAME_W - margin) or
            (e == "top" and y1 <= margin) or
            (e == "bottom" and y2 >= FRAME_H - margin)
        )
        (border if is_border else internal).add(e)
    return border, internal


def truncation_penalty(d):
    edges = touched_edges(d["xywhr"], d["slice"])
    border_edges, internal_edges = classify_edges(d["slice"], edges)
    long_side = max(d["xywhr"][2], d["xywhr"][3])
    median = CLASS_MEDIAN_LONG.get(d["cls"], 80.0)
    size_ratio = long_side / median
    undersized = size_ratio < SIZE_RATIO_THR

    # 贴内部切线：航拍无遮挡，必然不完整，判定更严格；贴真实边界：可能本来就只露出一部分
    is_trunc_internal = bool(internal_edges) and undersized
    is_trunc_border = bool(border_edges) and not internal_edges and undersized
    is_trunc = is_trunc_internal or is_trunc_border

    d["_border_edges"], d["_internal_edges"] = border_edges, internal_edges
    return (EDGE_CONF_PENALTY if is_trunc else 0.0), bool(edges), size_ratio, is_trunc, is_trunc_internal


def _to_poly(xywhr):
    cx, cy, w, h, r = xywhr
    pts = cv2.boxPoints(((float(cx), float(cy)), (float(w), float(h)), float(np.rad2deg(r))))
    return Polygon(pts)


def obb_nmm(dets, ios_thr=0.5, truncation_aware=False, class_agnostic=False):
    """SAHI 风格 GreedyNMM：按(调整后)置信度排序，IOS>阈值的框合并进最高分锚框，保留锚框原始xywhr。
    class_agnostic=True 时不管类别是否相同都参与合并——用于"航拍俯视车辆框不应重合"这条物理先验，
    抓那种同一辆车被同时判成car/truck/bus、产生多个几乎重合但类别不同的框的情况。
    """
    if not dets:
        return []
    dets = [dict(d) for d in dets]
    if truncation_aware:
        for d in dets:
            pen, et, sr, is_trunc, is_trunc_internal = truncation_penalty(d)
            d["_penalty"], d["_edge_touch"], d["_size_ratio"] = pen, et, sr
            d["_is_trunc"], d["_is_trunc_internal"] = is_trunc, is_trunc_internal
        dets.sort(key=lambda d: -(d["conf"] - d["_penalty"]))
    else:
        for d in dets:
            d["_penalty"], d["_edge_touch"], d["_size_ratio"] = 0.0, False, 1.0
            d["_is_trunc"], d["_is_trunc_internal"] = False, False
        dets.sort(key=lambda d: -d["conf"])

    polys = [_to_poly(d["xywhr"]) for d in dets]
    merged = [False] * len(dets)
    result = []
    for i in range(len(dets)):
        if merged[i]:
            continue
        cur_cls = dets[i]["cls"]
        anchor_poly = polys[i]
        for j in range(i + 1, len(dets)):
            if merged[j] or (not class_agnostic and dets[j]["cls"] != cur_cls):
                continue
            inter = anchor_poly.intersection(polys[j]).area
            min_area = min(anchor_poly.area, polys[j].area)
            if min_area > 0 and inter / min_area > ios_thr:
                merged[j] = True
        result.append(dets[i])
    return result


def rotated_nms(dets, iou_thr=0.2, class_agnostic=False):
    if not dets:
        return []
    dets = sorted(dets, key=lambda d: -d["conf"])
    xywhr_arr = np.array([d["xywhr"] for d in dets], dtype=np.float32)
    t = torch.from_numpy(xywhr_arr)
    iou_mat = batch_probiou(t, t).numpy()
    suppressed = [False] * len(dets)
    keep = []
    for i, d in enumerate(dets):
        if suppressed[i]:
            continue
        keep.append(d)
        for j in range(i + 1, len(dets)):
            if suppressed[j] or (not class_agnostic and dets[j]["cls"] != d["cls"]):
                continue
            if iou_mat[i, j] > iou_thr:
                suppressed[j] = True
    return keep


GLOBAL_NMS_IOS_THR = 0.5  # 全局(跨类别)NMS的IOS阈值：航拍俯视下车辆框不应重合，超过此值判定为同一物体


def global_cross_class_nms(dets):
    """最后一道全局去重：不管类别，只要IOS够高就判定是同一物体，只留置信度最高的那个（类别也跟它走）。
    专门解决"同一辆车被同时判成car/truck/bus，产生多个几乎重合但类别不同的框"的问题。
    """
    return obb_nmm(dets, ios_thr=GLOBAL_NMS_IOS_THR, truncation_aware=False, class_agnostic=True)


def annotate_truncation(dets):
    """给每个det标注 _penalty/_edge_touch/_size_ratio/_is_trunc/_is_trunc_internal/_internal_edges/_border_edges（不修改原dets）。"""
    out = []
    for d in dets:
        d = dict(d)
        pen, et, sr, is_trunc, is_trunc_internal = truncation_penalty(d)
        d["_penalty"], d["_edge_touch"], d["_size_ratio"] = pen, et, sr
        d["_is_trunc"], d["_is_trunc_internal"] = is_trunc, is_trunc_internal
        out.append(d)
    return out


def _merge_by_corner_union(xywhr1, xywhr2):
    """把两个碎片框的8个角点合并，重新拟合一个旋转矩形，作为拼接后的完整框估计。"""
    pts1 = cv2.boxPoints(((float(xywhr1[0]), float(xywhr1[1])), (float(xywhr1[2]), float(xywhr1[3])), float(np.rad2deg(xywhr1[4]))))
    pts2 = cv2.boxPoints(((float(xywhr2[0]), float(xywhr2[1])), (float(xywhr2[2]), float(xywhr2[3])), float(np.rad2deg(xywhr2[4]))))
    all_pts = np.vstack([pts1, pts2]).astype(np.float32)
    (cx, cy), (w, h), r = cv2.minAreaRect(all_pts)
    return np.array([cx, cy, w, h, np.deg2rad(r)], dtype=np.float32)


def find_seam_axis(det_i, det_j):
    """检查两个框是否贴在同一条内部切线上、且方位相对（i在切线一侧，j在另一侧）。
    返回 'x'（左右切开）或 'y'（上下切开），都不满足则返回 None。
    """
    for e_i in det_i["_internal_edges"]:
        e_j_wanted = OPPOSITE_EDGE[e_i]
        if e_j_wanted not in det_j["_internal_edges"]:
            continue
        if e_i in ("left", "right"):
            if e_i == "right" and det_j["xywhr"][0] > det_i["xywhr"][0]:
                return "x"
            if e_i == "left" and det_j["xywhr"][0] < det_i["xywhr"][0]:
                return "x"
        else:
            if e_i == "bottom" and det_j["xywhr"][1] > det_i["xywhr"][1]:
                return "y"
            if e_i == "top" and det_j["xywhr"][1] < det_i["xywhr"][1]:
                return "y"
    return None


def perpendicular_overlap_ratio(pts_i, pts_j, axis):
    """垂直于切割方向的重叠比例：axis='x'（左右切开）就比较y方向重叠，反之比较x方向。"""
    dim = 1 if axis == "x" else 0
    lo = max(pts_i[:, dim].min(), pts_j[:, dim].min())
    hi = min(pts_i[:, dim].max(), pts_j[:, dim].max())
    overlap = max(0.0, hi - lo)
    span = min(pts_i[:, dim].max() - pts_i[:, dim].min(), pts_j[:, dim].max() - pts_j[:, dim].min())
    return overlap / span if span > 0 else 0.0


def split_gap(pts_i, pts_j, axis):
    """沿切割方向，两碎片"切口"之间的缝隙（正=有缝隙，负=有重叠，绝对值越小说明拼接越严丝合缝）。"""
    dim = 0 if axis == "x" else 1
    i_lo, i_hi = pts_i[:, dim].min(), pts_i[:, dim].max()
    j_lo, j_hi = pts_j[:, dim].min(), pts_j[:, dim].max()
    if i_hi <= j_lo:
        return j_lo - i_hi
    if j_hi <= i_lo:
        return i_lo - j_hi
    return -min(i_hi - j_lo, j_hi - i_lo)


def stitch_fragments(dets):
    """只在"贴的是切片内部切线（不是原图真实边界）"的碎片里找拼接对象——因为航拍俯视无遮挡，
    这种情况下车辆物理上必然完整，只是被切片人为切开。拼接前用四道几何校验确认两个碎片
    真的是"同一条切线切开的两半"：
      1) 同类别
      2) 分别贴着同一条内部切线的两侧（find_seam_axis）
      3) 角度接近
      4) 垂直于切割方向的跨度要大幅重叠（同一"高度/位置"）
      5) 沿切割方向的缝隙要小（严丝合缝，不是隔着一段距离的两个不相关物体）
    最后再用先验尺寸兜底：拼接后长边必须落在该类别合理范围内。
    返回 (未被拼接的dets, 新拼出的dets)。
    """
    n = len(dets)
    used = [False] * n
    stitched = []
    for i in range(n):
        if used[i] or not dets[i]["_internal_edges"]:
            continue
        for j in range(i + 1, n):
            if used[j] or dets[j]["cls"] != dets[i]["cls"] or not dets[j]["_internal_edges"]:
                continue

            axis = find_seam_axis(dets[i], dets[j])
            if axis is None:
                continue

            angle_diff = abs(((dets[i]["xywhr"][4] - dets[j]["xywhr"][4] + np.pi) % (2 * np.pi)) - np.pi)
            if np.rad2deg(angle_diff) > STITCH_ANGLE_TOL_DEG:
                continue

            pts_i = cv2.boxPoints(((float(dets[i]["xywhr"][0]), float(dets[i]["xywhr"][1])),
                                    (float(dets[i]["xywhr"][2]), float(dets[i]["xywhr"][3])),
                                    float(np.rad2deg(dets[i]["xywhr"][4]))))
            pts_j = cv2.boxPoints(((float(dets[j]["xywhr"][0]), float(dets[j]["xywhr"][1])),
                                    (float(dets[j]["xywhr"][2]), float(dets[j]["xywhr"][3])),
                                    float(np.rad2deg(dets[j]["xywhr"][4]))))

            if perpendicular_overlap_ratio(pts_i, pts_j, axis) < STITCH_PERP_OVERLAP_MIN:
                continue
            if abs(split_gap(pts_i, pts_j, axis)) > STITCH_GAP_TOL_PX:
                continue

            merged_xywhr = _merge_by_corner_union(dets[i]["xywhr"], dets[j]["xywhr"])
            long_side = max(merged_xywhr[2], merged_xywhr[3])
            median = CLASS_MEDIAN_LONG.get(dets[i]["cls"], 80.0)
            ratio = long_side / median
            if not (STITCH_SIZE_RATIO_LO <= ratio <= STITCH_SIZE_RATIO_HI):
                continue  # 拼完尺寸不合理（可能根本不是同一辆车），不采信

            used[i] = used[j] = True
            new_det = dict(dets[i])
            new_det["xywhr"] = merged_xywhr
            new_det["conf"] = max(dets[i]["conf"], dets[j]["conf"])
            new_det["scale"] = f"{dets[i]['scale']}+{dets[j]['scale']}(拼接)"
            # 拼接后视为完整框，给一个不可能贴边的伪切片框，避免后续再被误判成截断
            new_det["slice"] = (-1e9, -1e9, 1e9, 1e9)
            stitched.append(new_det)
            break
    remaining = [d for k, d in enumerate(dets) if not used[k]]
    return remaining, stitched


def reject_fragments(dets):
    """截断碎片的硬性丢弃：贴内部切线的（必然不完整）用更严的阈值；贴真实边界的（可能本来就
    是部分视野）用更宽松的阈值。不管有没有同类别竞争对手，够不到阈值直接丢弃。
    """
    out = []
    for d in dets:
        if d["_is_trunc"]:
            thr = REJECT_CONF_THR_INTERNAL if d["_is_trunc_internal"] else REJECT_CONF_THR_BORDER
            if (d["conf"] - d["_penalty"]) < thr:
                continue
        out.append(d)
    return out


def full_pipeline(near_dets):
    """截断感知完整流程：标注截断 -> 尝试拼接碎片对 -> 硬性丢弃低质量碎片 -> NMM(截断感知，同类别)
    -> NMS(同类别) -> 全局NMS(跨类别，航拍俯视车辆框不应重合)。
    """
    annotated = annotate_truncation(near_dets)
    remaining, stitched = stitch_fragments(annotated)
    cleaned = reject_fragments(remaining) + stitched
    merged = obb_nmm(cleaned, truncation_aware=True)
    nms_result = rotated_nms(merged, iou_thr=0.2)
    return global_cross_class_nms(nms_result)


with open(ROI_FILE, "r", encoding="utf-8") as f:
    roi_data = json.load(f)
_items = roi_data if isinstance(roi_data, list) else roi_data["roi_polys_modes"]
roi_polys_modes = [([tuple(p) for p in item["pts"]], item["mode"]) for item in _items]

model = YOLO(MODEL_PATH)
model.to("cuda:0")

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"无法打开视频: {VIDEO_PATH}")

CLASS_NAMES = ["car", "truck", "bus"]
tiles = []
roi_mask = None

for f in TARGET_FRAMES:
    cap.set(cv2.CAP_PROP_POS_FRAMES, f - 1)
    ret, frame = cap.read()
    if not ret:
        print(f"帧 {f} 读取失败，跳过")
        continue

    if roi_mask is None and roi_polys_modes:
        roi_mask = multi_roi_mask(roi_polys_modes, frame.shape)

    proc = apply_clahe(frame)
    proc = gamma_transform(proc, gamma=1.2)
    proc = apply_blurred_background_fill(proc, roi_mask, feather=PRE_MASK_FEATHER, blur_ksize=PRE_MASK_BLUR_KSIZE)

    dets = sliced_predict_raw(proc, model, roi_mask, 960, 960, "960")
    dets += sliced_predict_raw(proc, model, roi_mask, 1280, 1280, "1280")

    vx, vy = vehicle_xy_at(f)
    near = [d for d in dets if abs(d["xywhr"][0] - vx) < FILTER_RADIUS and abs(d["xywhr"][1] - vy) < FILTER_RADIUS]

    old_final = rotated_nms(obb_nmm(near, truncation_aware=False), iou_thr=0.2)
    new_final = full_pipeline(near)

    gid, glabel = _GID_BY_FRAME[f]
    print(f"\n=== 帧 {f}（GT id={gid}:{glabel}）：该车附近原始检测 {len(near)} 个 -> 旧NMM合并后 {len(old_final)} 个 / 新NMM合并后 {len(new_final)} 个 ===")
    print("  [合并前的原始检测，标注来源切片]")
    for d in near:
        cx, cy, w, h, r = d["xywhr"]
        print(f"    scale={d['scale']:>4} slice={d['slice']} cls={CLASS_NAMES[d['cls']]:<5} conf={d['conf']:.3f} center=({cx:.1f},{cy:.1f}) w={w:.1f} h={h:.1f}")
    print("  [旧NMM，纯置信度排序]")
    for d in old_final:
        cx, cy, w, h, r = d["xywhr"]
        print(f"    scale={d['scale']:>4} cls={CLASS_NAMES[d['cls']]:<5} conf={d['conf']:.3f} center=({cx:.1f},{cy:.1f}) w={w:.1f} h={h:.1f}")
    print("  [新NMM，截断感知]")
    for d in new_final:
        cx, cy, w, h, r = d["xywhr"]
        if "拼接" in str(d["scale"]):
            flag = "[碎片拼接]"
        elif d["_is_trunc"]:
            kind = "内部切线" if d["_is_trunc_internal"] else "原图边界"
            flag = f"trunc({kind},size_ratio={d['_size_ratio']:.2f},penalty={d['_penalty']:.2f})"
        else:
            flag = ""
        print(f"    scale={d['scale']:>4} cls={CLASS_NAMES[d['cls']]:<5} conf={d['conf']:.3f} center=({cx:.1f},{cy:.1f}) w={w:.1f} h={h:.1f} {flag}")

    # 画图：黄色=旧NMM合并结果，绿色=新NMM(截断感知)合并结果
    x1 = max(0, int(vx - FILTER_RADIUS))
    y1 = max(0, int(vy - FILTER_RADIUS))
    x2 = min(frame.shape[1], int(vx + FILTER_RADIUS))
    y2 = min(frame.shape[0], int(vy + FILTER_RADIUS))
    vis = frame.copy()
    for i, d in enumerate(old_final):
        cx, cy, w, h, r = d["xywhr"]
        color = (0, 255, 255)  # 黄
        rect = ((float(cx), float(cy)), (float(w), float(h)), float(np.rad2deg(r)))
        pts = np.intp(cv2.boxPoints(rect))
        cv2.polylines(vis, [pts], True, color, 3)
        txt = f"old:{CLASS_NAMES[d['cls']]} {d['conf']:.2f}"
        ty = int(cy - float(h) / 2 - 12 - 18 * i)
        cv2.putText(vis, txt, (int(cx - 70), ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        cv2.putText(vis, txt, (int(cx - 70), ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    for i, d in enumerate(new_final):
        cx, cy, w, h, r = d["xywhr"]
        color = (0, 200, 0)  # 绿
        rect = ((float(cx), float(cy)), (float(w), float(h)), float(np.rad2deg(r)))
        pts = np.intp(cv2.boxPoints(rect))
        cv2.polylines(vis, [pts], True, color, 2)
        txt = f"new:{CLASS_NAMES[d['cls']]} {d['conf']:.2f}"
        ty = int(cy + float(h) / 2 + 20 + 18 * i)
        cv2.putText(vis, txt, (int(cx - 70), ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        cv2.putText(vis, txt, (int(cx - 70), ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

    crop = vis[y1:y2, x1:x2]
    crop = cv2.resize(crop, (440, 440))
    cv2.rectangle(crop, (0, 0), (439, 439), (255, 255, 255), 2)
    cv2.putText(crop, f"frame {f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
    tiles.append(crop)

cap.release()

cols = 5
rows = (len(tiles) + cols - 1) // cols
grid = np.zeros((rows * 440, cols * 440, 3), dtype=np.uint8)
for i, tile in enumerate(tiles):
    r, c = divmod(i, cols)
    grid[r * 440:(r + 1) * 440, c * 440:(c + 1) * 440] = tile

out_path = os.path.join(OUT_DIR, "remaining_16idsw_detection_check.jpg")
cv2.imwrite(out_path, grid)
print(f"\n已保存: {out_path}  (黄色=旧NMM合并结果, 绿色=截断感知新NMM合并结果)")
