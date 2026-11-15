# -*- coding: utf-8 -*-
"""
用GT283所在片段(frame 2670-2820)分别跑一遍"旧NMM(按置信度选锚点)"和"新NMM(组内选面积最大)"，
对比该片段实际跑出来的track_id是否还会像原来一样在id580/id594之间反复横跳。
"""
import runpy, os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

ns = runpy.run_path(r"D:\yolo26-tracking\_repro_base.py", run_name="repro_base")

import cv2
import numpy as np
from shapely.geometry import Polygon

load_roi_file = ns['load_roi_file']
multi_roi_mask = ns['multi_roi_mask']
sliced_predict_obb_direct = ns['sliced_predict_obb_direct']
apply_blurred_background_fill = ns['apply_blurred_background_fill']
_rotated_nms = ns['_rotated_nms']
_to_poly = ns['_to_poly']
GLOBAL_NMS_IOS_THR = ns['GLOBAL_NMS_IOS_THR']
yolo_model = ns['yolo_model']
BOTSORT = ns['BOTSORT']
ReIDEncoderObj = ns['ReIDEncoderObj']
OBBResultsAdapter = ns['OBBResultsAdapter']
tracker_args = dict(ns['tracker_args'])
from types import SimpleNamespace

import sys
sys.path.insert(0, r"D:\yolo26-tracking")
from eval_mot import parse_gt

GT_XML = r"D:\ultralytics-8.1.41\outputs\v8e53-2condence_20260521_150313\mota.xml"
gt = parse_gt(GT_XML)

VIDEO_PATH = r"E:\DJI\sgybw\video2\DJI_20250302041036_0037_D.MP4"
ROI_FILE = r"E:\DJI\sgybw\video2\DJI_20250302041036_0037_D_roi.json"

cap0 = cv2.VideoCapture(VIDEO_PATH)
cap0.set(cv2.CAP_PROP_POS_FRAMES, 0)
ret, first_frame = cap0.read()
roi_polys_modes, pixel_length, real_length = load_roi_file(ROI_FILE)
roi_mask = multi_roi_mask(roi_polys_modes, first_frame.shape)
cap0.release()

FIXED_NMM = ns['_obb_nmm']          # 当前文件里的版本(已修复：组内选面积最大)
_full_detection_pipeline = ns['_full_detection_pipeline']  # 真实的完整流程(含截断标注/碎片拼接)


def _old_obb_nmm(dets, ios_thr=0.5, truncation_aware=False, class_agnostic=False):
    """旧版本：组内始终保留置信度最高的锚点(修复前的原始逻辑，这里手写一份做对比，不改文件)。"""
    if not dets:
        return []
    dets = [dict(d) for d in dets]
    if truncation_aware:
        _truncation_penalty = ns['_truncation_penalty']
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
        cur_cls = dets[i]['cls']
        anchor_poly = polys[i]
        for j in range(i + 1, len(dets)):
            if merged[j] or (not class_agnostic and dets[j]['cls'] != cur_cls):
                continue
            inter = anchor_poly.intersection(polys[j]).area
            min_area = min(anchor_poly.area, polys[j].area)
            if min_area > 0 and inter / min_area > ios_thr:
                merged[j] = True
        result.append({'xywhr': dets[i]['xywhr'].copy(), 'conf': float(dets[i]['conf']), 'cls': cur_cls})
    return result


def full_pipeline(raw_dets, nmm_fn, nms_iou_thr=0.2):
    """通过临时替换命名空间里的 _obb_nmm，复用真实的 _full_detection_pipeline
    (包含截断标注/碎片拼接/拒绝低质量碎片等真实步骤，不再手写简化版)。"""
    ns['_obb_nmm'] = nmm_fn
    try:
        return _full_detection_pipeline(raw_dets, nms_iou_thr=nms_iou_thr)
    finally:
        ns['_obb_nmm'] = FIXED_NMM


GT_ID = 283


def run_segment(nmm_fn, label, f_start, f_end):
    cap = cv2.VideoCapture(VIDEO_PATH)

    # 每次都重新建一套干净的 args：先 with_reid=False 初始化(避免 build_encoder 用错误的
    # 内置模型路径去构造)，构造完 tracker 后再注入 encoder、打开 with_reid —— 跟
    # tracktworoi.py 的真实顺序完全一致。
    fresh_args = SimpleNamespace(**tracker_args)
    fresh_args.with_reid = False
    tracker = BOTSORT(args=fresh_args)
    tracker.encoder = ReIDEncoderObj
    tracker.args.with_reid = True

    id_at_watch = []  # (video_frame, matched_track_id, dist_px)

    for video_frame in range(f_start, f_end + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame - 1)
        ret, frame = cap.read()
        if not ret:
            break
        proc = apply_blurred_background_fill(frame, roi_mask, feather=51, blur_ksize=101)
        raw = sliced_predict_obb_direct(proc, yolo_model, roi_mask, slice_h=960, slice_w=960,
                                         overlap_h=0.3, overlap_w=0.3, conf_thr=0.3)
        raw += sliced_predict_obb_direct(proc, yolo_model, roi_mask, slice_h=1280, slice_w=1280,
                                          overlap_h=0.3, overlap_w=0.3, conf_thr=0.3)
        dets = full_pipeline(raw, nmm_fn, nms_iou_thr=0.2)
        results = OBBResultsAdapter(dets)
        online = tracker.update(results, img=frame)

        # 用GT283这一帧的真实标注位置，找online输出里离它最近的那个track，
        # 而不是用固定区域(固定区域会把画面上络绎不绝经过的其他车也算进来)。
        gt_frame = video_frame - 1
        gt_boxes = [b for b in gt.get(gt_frame, []) if b[0] == GT_ID]
        if not gt_boxes:
            continue
        _, gcx, gcy, gw, gh, grot = gt_boxes[0]

        best = None
        for row in online:
            xc, yc = float(row[0]), float(row[1])
            d = ((xc - gcx) ** 2 + (yc - gcy) ** 2) ** 0.5
            if best is None or d < best[1]:
                best = (int(row[5]), d)
        if best is not None and best[1] < 80:   # 80px内才算真的匹配上了GT283
            id_at_watch.append((video_frame, best[0], round(best[1], 1)))

    cap.release()
    print(f"\n=== {label} ===")
    seen_ids = sorted(set(t[1] for t in id_at_watch))
    print(f"匹配到GT283的track_id: {seen_ids}")
    switches = 0
    last_id = None
    for f, tid, d in id_at_watch:
        if last_id is not None and tid != last_id:
            switches += 1
            print(f"  切换: frame={f} {last_id}->{tid}  (匹配距离={d}px)")
        last_id = tid
    print(f"  共{len(id_at_watch)}帧匹配上GT283，{switches}次id切换")
    return id_at_watch


F_START, F_END = 2670, 2815

old_result = run_segment(_old_obb_nmm, "修复前(旧NMM，按置信度选锚点)", F_START, F_END)
new_result = run_segment(FIXED_NMM, "修复后(新NMM，组内选面积最大)", F_START, F_END)
