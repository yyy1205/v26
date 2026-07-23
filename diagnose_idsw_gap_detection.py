#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对16次IDSW事件，检查"丢失期间"(old track最后一帧 ~ 重新出现帧之间)的若干采样帧，
套用和 tracktworoi.py 完全一致的检测流水线(双尺度切片 + 截断感知NMM + NMS + 全局跨类别NMS)，
但用极低的原始置信度阈值(0.05)先捞出所有候选框，再标注它们相对生产阈值(conf_thr=0.3)的
状态，用于判断：这些帧车辆真的完全没被模型看见，还是被模型看见了但置信度不够/被我们自己
的截断逻辑滤掉了——如果是后者，说明丢失时长是可以通过调整流水线本身来缩短的。
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import torch
from shapely.geometry import Polygon
from ultralytics import YOLO
from ultralytics.utils.metrics import batch_probiou
from collections import defaultdict

# ── 与 tracktworoi.py 完全一致的常量 ──────────────────────────────────
CLASS_MEDIAN_LONG = {0: 72.5, 1: 90.6, 2: 192.0}
EDGE_MARGIN = 3
EDGE_CONF_PENALTY = 0.15
SIZE_RATIO_THR = 0.7
FRAME_W, FRAME_H = 3534, 1988
REJECT_CONF_THR_INTERNAL = 0.6
REJECT_CONF_THR_BORDER = 0.4
STITCH_ANGLE_TOL_DEG = 15
STITCH_PERP_OVERLAP_MIN = 0.6
STITCH_GAP_TOL_PX = 30
STITCH_SIZE_RATIO_LO = 0.7
STITCH_SIZE_RATIO_HI = 1.4
GLOBAL_NMS_IOS_THR = 0.5
PRODUCTION_CONF_THR = 0.3   # 生产流水线真正使用的置信度阈值
RAW_PROBE_CONF_THR = 0.05   # 诊断用：放到很低，看看模型到底"看没看见"

VIDEO_PATH = r"E:\DJI\cvat\0111_D_stabilized.avi"
MODEL_PATH = r"D:\yolo26-tracking\runs\obb\exp_yolo26-3\weights\best.pt"
ROI_FILE = r"E:\DJI\cvat\0111_D_stabilized_roi.json"
GT_XML = r"E:\DJI\cvat\GT(1).xml"
TRACK_TXT = r"D:\yolo26-tracking\outputs\yolo26-obb_20260722_155902\track_result.txt"
OUT_DIR = r"D:\yolo26-tracking\outputs\yolo26-obb_20260722_155902\idsw_gap_detection_check"
os.makedirs(OUT_DIR, exist_ok=True)

IDSW_EVENTS = [
    (163, 138, 137, 165),
    (420, 46, 114, 210),
    (467, 136, 135, 216),
    (489, 46, 210, 221),
    (564, 136, 216, 235),
    (576, 150, 150, 238),
    (628, 136, 235, 248),
    (688, 150, 238, 256),
    (893, 198, 202, 286),
    (999, 198, 286, 307),
    (1120, 218, 234, 329),
    (1156, 218, 329, 338),
    (1444, 275, 310, 382),
    (1538, 275, 382, 390),
    (1649, 293, 334, 407),
    (1784, 313, 358, 426),
]
SAMPLES_PER_EVENT = 3
FILTER_RADIUS = 150
CLASS_NAMES = ["car", "truck", "bus"]


def parse_gt(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    gt = defaultdict(dict)
    labels = {}
    for track in root.findall("track"):
        gt_id = int(track.get("id"))
        labels[gt_id] = track.get("label")
        for box in track.findall("box"):
            if box.get("outside") == "1":
                continue
            frame = int(box.get("frame"))
            xtl, ytl = float(box.get("xtl")), float(box.get("ytl"))
            xbr, ybr = float(box.get("xbr")), float(box.get("ybr"))
            rot = float(box.get("rotation", 0))
            cx, cy = (xtl + xbr) / 2, (ytl + ybr) / 2
            gt[gt_id][frame] = (cx, cy, xbr - xtl, ybr - ytl, rot)
        for poly in track.findall("polygon"):
            if poly.get("outside") == "1":
                continue
            frame = int(poly.get("frame"))
            pts = np.array([[float(v) for v in p.split(",")] for p in poly.get("points").split(";")], dtype=np.float32)
            (cx, cy), (w, h), rot = cv2.minAreaRect(pts)
            gt[gt_id][frame] = (cx, cy, w, h, rot)
    return gt, labels


def gt_pos_near(gt_track, gt_frame, max_search=20):
    """精确帧没有就在±max_search范围内找最近的已知GT帧。"""
    if gt_frame in gt_track:
        return gt_track[gt_frame]
    for d in range(1, max_search + 1):
        if gt_frame - d in gt_track:
            return gt_track[gt_frame - d]
        if gt_frame + d in gt_track:
            return gt_track[gt_frame + d]
    return None


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
        if mode == 1:
            cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 255)
    for pts, mode in polys_with_mode:
        if mode == 2:
            cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 0)
    return mask


def get_slice_bboxes(H, W, sh=640, sw=640, oh=0.3, ow=0.3):
    y_overlap, x_overlap = int(oh * sh), int(ow * sw)
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


def sliced_predict_raw(frame, model, roi_mask, slice_h, slice_w, tag, overlap=0.3, conf_thr=RAW_PROBE_CONF_THR):
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
            dets.append({"xywhr": xywhr[i], "conf": float(confs[i]), "cls": int(clses[i]), "scale": tag, "slice": (x1, y1, x2, y2)})
    return dets


OPPOSITE_EDGE = {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}


def touched_edges(xywhr, slice_box, margin=EDGE_MARGIN):
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
    x1, y1, x2, y2 = slice_box
    border, internal = set(), set()
    for e in edges:
        is_border = ((e == "left" and x1 <= margin) or (e == "right" and x2 >= FRAME_W - margin) or
                     (e == "top" and y1 <= margin) or (e == "bottom" and y2 >= FRAME_H - margin))
        (border if is_border else internal).add(e)
    return border, internal


def truncation_penalty(d):
    edges = touched_edges(d["xywhr"], d["slice"])
    border_edges, internal_edges = classify_edges(d["slice"], edges)
    long_side = max(d["xywhr"][2], d["xywhr"][3])
    median = CLASS_MEDIAN_LONG.get(d["cls"], 80.0)
    size_ratio = long_side / median
    undersized = size_ratio < SIZE_RATIO_THR
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


def global_cross_class_nms(dets):
    return obb_nmm(dets, ios_thr=GLOBAL_NMS_IOS_THR, truncation_aware=False, class_agnostic=True)


def annotate_truncation(dets):
    out = []
    for d in dets:
        d = dict(d)
        pen, et, sr, is_trunc, is_trunc_internal = truncation_penalty(d)
        d["_penalty"], d["_edge_touch"], d["_size_ratio"] = pen, et, sr
        d["_is_trunc"], d["_is_trunc_internal"] = is_trunc, is_trunc_internal
        out.append(d)
    return out


def reject_fragments(dets):
    out = []
    for d in dets:
        if d["_is_trunc"]:
            thr = REJECT_CONF_THR_INTERNAL if d["_is_trunc_internal"] else REJECT_CONF_THR_BORDER
            if (d["conf"] - d["_penalty"]) < thr:
                continue
        out.append(d)
    return out


def full_pipeline(near_dets):
    """完整生产流水线，但输入已经是 RAW_PROBE_CONF_THR(0.05) 捞出来的候选，
    所以这里再手动卡一道 PRODUCTION_CONF_THR，模拟生产环境里 conf_thr=0.3 直接从源头过滤掉的效果。
    """
    prod_dets = [d for d in near_dets if d["conf"] >= PRODUCTION_CONF_THR]
    annotated = annotate_truncation(prod_dets)
    cleaned = reject_fragments(annotated)
    merged = obb_nmm(cleaned, truncation_aware=True)
    nms_result = rotated_nms(merged, iou_thr=0.2)
    return global_cross_class_nms(nms_result)


# ── 主流程 ──────────────────────────────────────────────────────────
print(f"解析 GT: {GT_XML}")
gt, gt_labels = parse_gt(GT_XML)

print(f"解析 track_result: {TRACK_TXT}")
by_track = defaultdict(list)
with open(TRACK_TXT, encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split(",")
        if len(parts) < 2:
            continue
        by_track[int(parts[1])].append(int(parts[0]))
for tid in by_track:
    by_track[tid].sort()

with open(ROI_FILE, "r", encoding="utf-8") as f:
    roi_data = json.load(f)
_items = roi_data if isinstance(roi_data, list) else roi_data["roi_polys_modes"]
roi_polys_modes = [([tuple(p) for p in item["pts"]], item["mode"]) for item in _items]

model = YOLO(MODEL_PATH)
model.to("cuda:0")

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"无法打开视频: {VIDEO_PATH}")

roi_mask = None
tiles = []

for gt_frame, gt_id, old_pid, new_pid in IDSW_EVENTS:
    pred_frame = gt_frame + 1
    old_frames = [fr for fr in by_track.get(old_pid, []) if fr <= pred_frame]
    old_last_frame = old_frames[-1] if old_frames else pred_frame - 1
    lost_start, lost_end = old_last_frame + 1, pred_frame - 1
    gap = lost_end - lost_start + 1

    if gap <= 0:
        sample_frames = [pred_frame - 1]
    elif gap <= SAMPLES_PER_EVENT:
        sample_frames = list(range(lost_start, lost_end + 1))
    else:
        sample_frames = sorted(set(int(round(lost_start + k * (gap - 1) / (SAMPLES_PER_EVENT - 1)))
                                    for k in range(SAMPLES_PER_EVENT)))

    gt_track = gt.get(gt_id, {})
    print(f"\n=== GT#{gt_id} 丢失区间 tracker帧[{lost_start},{lost_end}] (共{gap}帧)，采样帧: {sample_frames} ===")

    for sf in sample_frames:
        gt_frame_for_sf = sf - 1  # tracker frame -> GT frame
        pos = gt_pos_near(gt_track, gt_frame_for_sf)
        tile = np.full((360, 360, 3), 230, dtype=np.uint8)
        if pos is None:
            cv2.putText(tile, f"GT#{gt_id} f{sf}: 无GT位置数据", (8, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            tiles.append((gt_id, sf, tile))
            continue
        gcx, gcy, gw, gh, grot = pos

        cap.set(cv2.CAP_PROP_POS_FRAMES, sf - 1)
        ret, frame = cap.read()
        if not ret:
            cv2.putText(tile, f"GT#{gt_id} f{sf}: 读帧失败", (8, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            tiles.append((gt_id, sf, tile))
            continue

        if roi_mask is None and roi_polys_modes:
            roi_mask = multi_roi_mask(roi_polys_modes, frame.shape)

        proc = apply_clahe(frame)
        proc = gamma_transform(proc, gamma=1.2)
        proc = apply_blurred_background_fill(proc, roi_mask, feather=51, blur_ksize=101)

        raw_dets = sliced_predict_raw(proc, model, roi_mask, 960, 960, "960")
        raw_dets += sliced_predict_raw(proc, model, roi_mask, 1280, 1280, "1280")
        near = [d for d in raw_dets if abs(d["xywhr"][0] - gcx) < FILTER_RADIUS and abs(d["xywhr"][1] - gcy) < FILTER_RADIUS]
        prod_result = full_pipeline(near)
        prod_boxes = {id(d) for d in prod_result}  # full_pipeline内部deep-copy了，用conf+center近似匹配即可
        prod_keys = {(round(d["xywhr"][0], 1), round(d["xywhr"][1], 1)) for d in prod_result}

        best_conf = max((d["conf"] for d in near), default=0.0)
        print(f"  f{sf}: 附近原始候选 {len(near)} 个 (含sub-threshold), 最高conf={best_conf:.3f}, "
              f"通过生产流水线 {len(prod_result)} 个")
        for d in sorted(near, key=lambda d: -d["conf"]):
            cx, cy, w, h, r = d["xywhr"]
            passed = (round(cx, 1), round(cy, 1)) in prod_keys
            print(f"      scale={d['scale']:>4} cls={CLASS_NAMES[d['cls']]:<5} conf={d['conf']:.3f} "
                  f"center=({cx:.1f},{cy:.1f}) {'[通过生产流水线]' if passed else ''}")

        x1, y1 = max(0, int(gcx - FILTER_RADIUS)), max(0, int(gcy - FILTER_RADIUS))
        x2, y2 = min(frame.shape[1], int(gcx + FILTER_RADIUS)), min(frame.shape[0], int(gcy + FILTER_RADIUS))
        vis = frame.copy()
        # GT 真实框：青色虚线效果(用短线段模拟)
        gt_rect = ((float(gcx), float(gcy)), (float(gw), float(gh)), float(grot))
        gt_pts = np.intp(cv2.boxPoints(gt_rect))
        cv2.polylines(vis, [gt_pts], True, (255, 255, 0), 2)
        for d in near:
            cx, cy, w, h, r = d["xywhr"]
            passed = (round(cx, 1), round(cy, 1)) in prod_keys
            color = (0, 200, 0) if passed else ((0, 140, 255) if d["conf"] >= PRODUCTION_CONF_THR else (0, 0, 255))
            rect = ((float(cx), float(cy)), (float(w), float(h)), float(np.rad2deg(r)))
            pts = np.intp(cv2.boxPoints(rect))
            cv2.polylines(vis, [pts], True, color, 2)
            cv2.putText(vis, f"{d['conf']:.2f}", (int(cx - 20), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        crop = vis[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        crop = cv2.resize(crop, (330, 330))
        tile[:330, :330] = crop
        cv2.putText(tile, f"GT#{gt_id} f{sf} (gap{gap})", (5, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append((gt_id, sf, tile))

cap.release()

cols = SAMPLES_PER_EVENT
rows = len(IDSW_EVENTS)
grid = np.full((rows * 360, cols * 360, 3), 255, dtype=np.uint8)
row_map = {}
next_row = 0
for gid, sf, tile in tiles:
    if gid not in row_map:
        row_map[gid] = next_row
        next_row += 1
    r = row_map[gid]
    # 在该行里按出现顺序放列（同一event最多SAMPLES_PER_EVENT个）
    used_cols = sum(1 for g2, _, _ in tiles[:tiles.index((gid, sf, tile))] if g2 == gid)
    c = min(used_cols, cols - 1)
    grid[r * 360:(r + 1) * 360, c * 360:(c + 1) * 360] = tile

out_path = os.path.join(OUT_DIR, "idsw_gap_detection_grid.jpg")
cv2.imwrite(out_path, grid)
print(f"\n已保存: {out_path}")
print("图例: 青色=GT真值框  绿色=通过生产流水线(conf>=0.3且未被截断逻辑滤掉)  "
      "橙色=conf>=0.3但被自定义流水线滤掉  红色=conf<0.3的极弱信号")
