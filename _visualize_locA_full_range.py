# -*- coding: utf-8 -*-
"""
卡点A：740~790帧全部跑一遍检测，画出所有检测框，并且用红框特别标出GT241对应的那辆车。

标注方法改为【链式帧间最近邻】而不是"离外推位置最近"：
从已知的真实GT位置(757帧, 1636.5,1194.2)开始，逐帧往前/往后走，每一步只在【上一帧已确认的
GT241位置】附近一个很小的半径内(<=15px，真实车辆帧间位移不可能突然跳到隔壁车上)找最近检测，
找不到就停止外推(而不是继续瞎猜)。这样能避免两车贴近时误跳到旁边那辆车上。
"""
import runpy
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

ns = runpy.run_path(r"D:\yolo26-tracking\_repro_base.py", run_name="repro_base")

import cv2
import numpy as np

VIDEO_PATH = ns["VIDEO_PATH"]
ROI_FILE = ns["ROI_FILE"]
load_roi_file = ns["load_roi_file"]
multi_roi_mask = ns["multi_roi_mask"]
sliced_predict_obb_direct = ns["sliced_predict_obb_direct"]
_full_detection_pipeline = ns["_full_detection_pipeline"]
PRE_MASK_DETECTION = ns["PRE_MASK_DETECTION"]
PRE_MASK_MODE = ns["PRE_MASK_MODE"]
PRE_MASK_FEATHER = ns["PRE_MASK_FEATHER"]
PRE_MASK_BLUR_KSIZE = ns["PRE_MASK_BLUR_KSIZE"]
apply_blurred_background_fill = ns["apply_blurred_background_fill"]
IOU_NMS_THR = 0.2
yolo_model = ns["yolo_model"]

cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
ret, first_frame = cap.read()
roi_polys_modes, pixel_length, real_length = load_roi_file(ROI_FILE)
roi_mask = multi_roi_mask(roi_polys_modes, first_frame.shape)

OUT_DIR = r"D:\yolo26-tracking\visualizations_missing_gt\locA_740_790"
os.makedirs(OUT_DIR, exist_ok=True)

CX, CY, PAD = 1645, 1195, 300
MAX_STEP = 50  # 帧间最大允许位移(px)，超过这个距离就认为找不到了，不再强行外推

# 已知的GT241真实位置(从GT标注拿到的756~779帧数据，转成video_frame=gt_frame+1)
KNOWN_GT241 = {
    757: (1636.5, 1194.2), 758: (1637.0, 1194.0), 759: (1637.7, 1194.0), 760: (1638.3, 1194.0),
    761: (1639.0, 1194.0), 762: (1639.2, 1192.8), 763: (1640.0, 1192.5), 764: (1641.0, 1192.8),
    765: (1642.0, 1192.2), 766: (1643.8, 1192.0), 767: (1645.0, 1191.7), 768: (1646.0, 1191.3),
    769: (1647.0, 1190.8), 770: (1647.7, 1190.3), 771: (1648.3, 1189.9), 772: (1649.0, 1189.5),
    773: (1650.5, 1189.8), 774: (1651.3, 1189.3), 775: (1652.1, 1188.8), 776: (1652.9, 1188.4),
    777: (1653.7, 1187.9), 778: (1654.5, 1187.5), 779: (1656.0, 1187.5), 780: (1657.0, 1187.3),
}

# 完整GT241框(w,h,rot_deg)，跟上面的KNOWN_GT241位置一一对应，用来画蓝色GT框
GT241_BOXES = {
    757: (39.8, 33.2, -19.0), 758: (33.2, 39.8, -20.9), 759: (33.2, 39.8, -22.3), 760: (33.2, 39.8, -23.6),
    761: (33.2, 39.8, -25.0), 762: (33.2, 39.8, -25.1), 763: (33.2, 39.8, -24.3), 764: (39.8, 33.2, -25.9),
    765: (39.8, 33.2, -25.9), 766: (39.8, 33.2, -22.6), 767: (39.8, 33.2, -22.1), 768: (39.8, 33.2, -22.3),
    769: (39.8, 33.2, -22.5), 770: (39.8, 33.2, -22.6), 771: (39.8, 33.2, -22.8), 772: (39.8, 33.2, -22.9),
    773: (39.8, 33.2, -22.0), 774: (39.8, 33.2, -21.8), 775: (39.8, 33.2, -21.6), 776: (39.8, 33.2, -21.5),
    777: (39.8, 33.2, -21.3), 778: (39.8, 33.2, -21.1), 779: (39.8, 33.2, -20.8), 780: (39.8, 33.2, -20.5),
}

# 先跑完740~790每一帧的检测，缓存下来
all_dets = {}
for video_frame in range(740, 791):
    cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame - 1)
    ret, frame = cap.read()
    if not ret:
        print(f"帧{video_frame} 读取失败")
        continue
    proc_frame_for_sahi = frame
    if PRE_MASK_DETECTION and roi_mask is not None and PRE_MASK_MODE == "blur":
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
    all_dets[video_frame] = (frame, dets)
    if video_frame % 10 == 0:
        print(f"检测完成到帧{video_frame}")

# 链式识别：以757为锚点(用真实GT位置)，分别向前(758..790)和向后(756..740)逐帧最近邻传播
identified = {757: KNOWN_GT241[757]}

for video_frame in range(758, 791):
    prev_pos = identified.get(video_frame - 1)
    if prev_pos is None:
        break
    _, dets = all_dets[video_frame]
    ranked = sorted(
        ((((d["xywhr"][0]-prev_pos[0])**2+(d["xywhr"][1]-prev_pos[1])**2)**0.5, d["xywhr"][0], d["xywhr"][1]) for d in dets),
        key=lambda r: r[0]
    )
    top2 = ranked[:2]
    print(f"帧{video_frame}(向前): 最近={top2[0][0]:.1f}px  次近={(top2[1][0] if len(top2)>1 else float('nan')):.1f}px")
    best_d, best_pos = (top2[0][0], (top2[0][1], top2[0][2])) if top2 else (None, None)
    if best_d is not None and best_d <= MAX_STEP:
        identified[video_frame] = best_pos
    else:
        print(f"帧{video_frame}: 找不到帧间距离<={MAX_STEP}px的候选(最近{best_d}), 停止向前外推")
        break

for video_frame in range(756, 739, -1):
    prev_pos = identified.get(video_frame + 1)
    if prev_pos is None:
        break
    _, dets = all_dets[video_frame]
    ranked = sorted(
        ((((d["xywhr"][0]-prev_pos[0])**2+(d["xywhr"][1]-prev_pos[1])**2)**0.5, d["xywhr"][0], d["xywhr"][1]) for d in dets),
        key=lambda r: r[0]
    )
    top2 = ranked[:2]
    print(f"帧{video_frame}(向后): 最近={top2[0][0]:.1f}px  次近={(top2[1][0] if len(top2)>1 else float('nan')):.1f}px")
    best_d, best_pos = (top2[0][0], (top2[0][1], top2[0][2])) if top2 else (None, None)
    if best_d is not None and best_d <= MAX_STEP:
        identified[video_frame] = best_pos
    else:
        print(f"帧{video_frame}: 找不到帧间距离<={MAX_STEP}px的候选(最近{best_d}), 停止向后外推")
        break

print(f"链式识别成功覆盖的帧范围: {min(identified)}~{max(identified)} (共{len(identified)}帧)")

# 画图
for video_frame in range(740, 791):
    if video_frame not in all_dets:
        continue
    frame, dets = all_dets[video_frame]
    x1, y1 = max(0, CX - PAD), max(0, CY - PAD)
    x2, y2 = CX + PAD, CY + PAD
    crop = frame[y1:y2, x1:x2].copy()

    gt241_pos = identified.get(video_frame)

    for d in dets:
        dx, dy, dw, dh, dr = d["xywhr"]
        if not (x1 - 50 <= dx <= x2 + 50 and y1 - 50 <= dy <= y2 + 50):
            continue
        is_gt241 = gt241_pos is not None and abs(dx - gt241_pos[0]) < 1 and abs(dy - gt241_pos[1]) < 1
        color = (0, 0, 255) if is_gt241 else (0, 255, 0)
        rr = ((dx - x1, dy - y1), (dw, dh), np.rad2deg(dr))
        box = cv2.boxPoints(rr).astype(int)
        cv2.drawContours(crop, [box], 0, color, 2)
        label = f"{'GT241 ' if is_gt241 else ''}{d['conf']:.2f}"
        cv2.putText(crop, label, (int(dx - x1 - dw / 2), int(dy - y1 - dh / 2) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # 画GT241真实标注框(蓝色)，只有756~779这24帧(video_frame757~780)有真实GT数据
    if video_frame in GT241_BOXES:
        gcx, gcy = KNOWN_GT241[video_frame]
        gw, gh, grot = GT241_BOXES[video_frame]
        grr = ((gcx - x1, gcy - y1), (gw, gh), grot)
        gbox = cv2.boxPoints(grr).astype(int)
        cv2.drawContours(crop, [gbox], 0, (255, 0, 0), 2)
        cv2.putText(crop, "GT241真值", (int(gcx - x1 - gw / 2), int(gcy - y1 + gh / 2) + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1, cv2.LINE_AA)

    tag = "frame={} {}".format(video_frame, "" if gt241_pos is not None else "(GT241链式识别中断)")
    cv2.putText(crop, tag, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    out_path = os.path.join(OUT_DIR, f"f{video_frame:04d}.jpg")
    cv2.imwrite(out_path, crop)

cap.release()
print("完成，全部保存到", OUT_DIR)
