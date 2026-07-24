# -*- coding: utf-8 -*-
"""
验证画面增强(CLAHE+gamma)对ReID相似度区分度是好是坏。
用margin_survey实测抓到的3个"险胜"真实案例，同一批检测框，分别用
(a) 生产环境实际用的增强后画面 (b) 原始未增强画面
截图提特征，对比正负样本相似度差距(margin)有没有变化。
"""
import sys
sys.path.insert(0, r"D:\deep-person-reid-master")
import numpy as np
import cv2
import torch
import torchreid

VIDEO_PATH = r"E:\DJI\cvat\0111_D_stabilized.avi"
REID_MODEL_PATH = r"D:\deep-person-reid-master\scripts\log\osnet_ain_x1_0_market1501_softmax_cosinelr\model\model.pth.tar-100"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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


def crop_rotated(img_bgr, pts):
    """pts: 4x2 corner points -> minAreaRect -> 旋转裁剪(逻辑对齐生产ReIDEncoder._crop)"""
    pts = np.array(pts, dtype=np.float32)
    (xc, yc), (w, h), angle_deg = cv2.minAreaRect(pts)
    H_img, W_img = img_bgr.shape[:2]
    pad = int(max(w, h) * 0.8 + 4)
    x1p, y1p = max(0, int(xc) - pad), max(0, int(yc) - pad)
    x2p, y2p = min(W_img, int(xc) + pad), min(H_img, int(yc) + pad)
    patch = img_bgr[y1p:y2p, x1p:x2p]
    if patch.size == 0:
        return None
    lxc, lyc = xc - x1p, yc - y1p
    ph, pw = patch.shape[:2]
    M = cv2.getRotationMatrix2D((lxc, lyc), -angle_deg, 1.0)
    rotated = cv2.warpAffine(patch, M, (pw, ph), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    rx1, ry1 = max(0, int(lxc - w / 2)), max(0, int(lyc - h / 2))
    rx2, ry2 = min(pw, int(lxc + w / 2)), min(ph, int(lyc + h / 2))
    if rx2 <= rx1 or ry2 <= ry1:
        return None
    return rotated[ry1:ry2, rx1:rx2]


# (event_name, det_frame, det_pts, pos_frame(真实同一辆车的old), pos_pts, neg_frame(次名/干扰车), neg_pts)
EVENTS = [
    ("evt440(69vs54)",
     440, [(0.3, 1209.7), (-8.02, 1181.42), (11.82, 1175.59), (20.14, 1203.87)],
     402, [(0.16, 1054.73), (-7.54, 1024.62), (12.56, 1019.48), (20.26, 1049.6)],
     382, [(0.28, 1006.2), (-6.71, 977.25), (13.29, 972.42), (20.29, 1001.36)]),
    ("evt1601(240vs272)",
     1601, [(922.3, 1628.52), (922.1, 1599.51), (963.88, 1599.23), (964.08, 1628.24)],
     1519, [(1682.78, 751.76), (1651.79, 695.5), (1676.82, 681.72), (1707.8, 737.98)],
     1526, [(335.12, 520.57), (374.73, 473.3), (399.39, 493.96), (359.78, 541.23)]),
    ("evt1726(272vs240)",
     1726, [(1676.46, 753.48), (1652.76, 699.68), (1678.54, 688.32), (1702.24, 742.13)],
     1638, [(1667.08, 748.5), (1640.32, 698.0), (1665.79, 684.51), (1692.54, 735.01)],
     1642, [(1265.5, 1442.47), (1253.81, 1415.66), (1279.38, 1404.5), (1291.08, 1431.31)]),
]

cap = cv2.VideoCapture(VIDEO_PATH)


def read_frame(frame_id):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id - 1)
    ret, frame = cap.read()
    assert ret, f"无法读取帧{frame_id}"
    return frame


print("加载ReID模型...")
extractor = torchreid.utils.FeatureExtractor(model_name="osnet_ain_x1_0", model_path=REID_MODEL_PATH, device=DEVICE)


def feat_from_crop(crop_bgr):
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    f = extractor([crop_rgb]).cpu().numpy()[0]
    return f / (np.linalg.norm(f) + 1e-6)


def cos_sim(a, b):
    return float(np.dot(a, b))


print(f"\n{'event':>20} {'pipeline':>10} {'sim(det,pos)':>13} {'sim(det,neg)':>13} {'margin':>8}")
print("-" * 70)
for name, df, dpts, pf, ppts, nf, npts in EVENTS:
    for enhanced in (False, True):
        det_raw = read_frame(df)
        pos_raw = read_frame(pf)
        neg_raw = read_frame(nf)
        if enhanced:
            det_raw = gamma_transform(apply_clahe(det_raw), gamma=1.2)
            pos_raw = gamma_transform(apply_clahe(pos_raw), gamma=1.2)
            neg_raw = gamma_transform(apply_clahe(neg_raw), gamma=1.2)
        det_crop = crop_rotated(det_raw, dpts)
        pos_crop = crop_rotated(pos_raw, ppts)
        neg_crop = crop_rotated(neg_raw, npts)
        if det_crop is None or pos_crop is None or neg_crop is None or det_crop.size == 0 or pos_crop.size == 0 or neg_crop.size == 0:
            print(f"{name:>20} {'增强' if enhanced else '原始':>10}  裁剪失败，跳过")
            continue
        f_det, f_pos, f_neg = feat_from_crop(det_crop), feat_from_crop(pos_crop), feat_from_crop(neg_crop)
        sim_pos = cos_sim(f_det, f_pos)
        sim_neg = cos_sim(f_det, f_neg)
        pipeline = "增强(生产实际)" if enhanced else "原始未增强"
        print(f"{name:>20} {pipeline:>10} {sim_pos:>13.4f} {sim_neg:>13.4f} {sim_pos - sim_neg:>8.4f}")
    print()

cap.release()
