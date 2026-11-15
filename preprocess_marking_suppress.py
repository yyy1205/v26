# -*- coding: utf-8 -*-
"""
独立视频预处理脚本：只模糊标志标线和地面文字，不动其他背景(路面本身/绿化带/桥梁结构等保持原样)。

思路：
1. 对整段视频采样多帧算中位数背景模型(车辆被自然滤掉，只剩静止内容)。
2. 在背景模型上用top-hat形态学变换找出"局部比周围路面更亮的细小结构"——标线/文字通常是白色油漆，
   跟旁边的沥青路面对比度很高，这个变换能把它们跟大片均匀的路面区分开。
3. 逐帧处理时，只在"标线掩码命中 且 这一帧没有车辆经过"的像素上做模糊替换，其余背景原样保留。
"""
import cv2
import numpy as np
import time

VIDEO_PATH = r"E:\DJI\cvat\0111_D_stabilized.avi"
OUTPUT_PATH = r"E:\DJI\cvat\0111_D_stabilized_marksuppress.mp4"

BG_MODEL_N_SAMPLES = 120
TOPHAT_KSIZE = 21   # 9→21：核太小只能抓到粗笔画文字的边缘(空心)，抓不到中间实心部分
MARKING_THRESH = 55
MARKING_DILATE = 3
FG_THRESH = 25
FG_MIN_AREA = 150   # 连通域最小面积(像素)，小于此值当噪声过滤掉；真实车辆远大于此值
FG_DILATE = 15
BG_BLUR_KSIZE = 31
ALPHA_FEATHER = 9


def build_background_model(video_path, n_samples=120):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = max(1, total // n_samples)
    frames = []
    idx = 0
    while idx < total and len(frames) < n_samples:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        idx += stride
    cap.release()
    if not frames:
        raise RuntimeError("背景采样失败，读不到任何帧")
    return np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8)


def build_marking_mask(bg_model, tophat_ksize=9, thresh=55, dilate_ksize=3):
    gray = cv2.cvtColor(bg_model, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tophat_ksize, tophat_ksize))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    _, mask = cv2.threshold(tophat, thresh, 255, cv2.THRESH_BINARY)
    if dilate_ksize > 0:
        mask = cv2.dilate(mask, np.ones((dilate_ksize, dilate_ksize), np.uint8))
    return mask


def _clean_foreground_mask(fg_mask, min_area=150):
    """连通域面积过滤：车辆是几十像素见方的大色块，噪声/光照抖动造成的假前景通常是零散小点。
    只保留面积>=min_area的连通域，能大幅减少"没车却被误判成有车"导致的逐帧闪烁，
    而真实车辆(面积远大于min_area)完全不受影响。
    """
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg_mask, connectivity=8)
    cleaned = np.zeros_like(fg_mask)
    for label in range(1, n_labels):  # label 0 是背景
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label] = 255
    return cleaned


def apply_marking_suppression(frame, bg_model, marking_mask, fg_thresh=25, fg_min_area=150,
                               fg_dilate=15, blur_ksize=31, alpha_feather=9):
    diff = cv2.absdiff(frame, bg_model)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, fg_mask = cv2.threshold(gray_diff, fg_thresh, 255, cv2.THRESH_BINARY)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    fg_mask = _clean_foreground_mask(fg_mask, fg_min_area)  # 过滤噪声碎块，只留车辆大小的前景
    if fg_dilate > 0:
        fg_mask = cv2.dilate(fg_mask, np.ones((fg_dilate, fg_dilate), np.uint8))

    # 只模糊"标线区域 且 这一帧没有车辆经过"的地方，车辆经过时该处保持原样清晰
    suppress_mask = cv2.bitwise_and(marking_mask, cv2.bitwise_not(fg_mask))

    bk = blur_ksize | 1
    blurred_bg = cv2.GaussianBlur(bg_model, (bk, bk), 0)

    af = alpha_feather | 1
    alpha = cv2.GaussianBlur(suppress_mask.astype(np.float32) / 255.0, (af, af), 0)[..., None]
    return (frame.astype(np.float32) * (1 - alpha) + blurred_bg.astype(np.float32) * alpha).astype(np.uint8)


def main():
    print(f"[标线压平] 正在采样{BG_MODEL_N_SAMPLES}帧计算背景模型...")
    t0 = time.time()
    bg_model = build_background_model(VIDEO_PATH, BG_MODEL_N_SAMPLES)
    print(f"[标线压平] 背景模型计算完成，耗时{time.time()-t0:.0f}s")

    marking_mask = build_marking_mask(bg_model, TOPHAT_KSIZE, MARKING_THRESH, MARKING_DILATE)
    coverage = (marking_mask > 0).mean() * 100
    print(f"[标线压平] 标线掩码计算完成，覆盖率={coverage:.1f}%")
    cv2.imwrite(r"E:\DJI\cvat\0111_D_stabilized_marksuppress_mask.jpg", marking_mask)

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (W, H))

    frame_id = 0
    t0 = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        processed = apply_marking_suppression(
            frame, bg_model, marking_mask,
            fg_thresh=FG_THRESH, fg_min_area=FG_MIN_AREA, fg_dilate=FG_DILATE,
            blur_ksize=BG_BLUR_KSIZE, alpha_feather=ALPHA_FEATHER,
        )
        out.write(processed)
        frame_id += 1
        if frame_id % 100 == 0:
            print(f"已处理 {frame_id}/{total} 帧, 耗时{time.time()-t0:.0f}s", flush=True)

    cap.release()
    out.release()
    print(f"完成，总耗时{time.time()-t0:.0f}s")
    print(f"输出视频: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
