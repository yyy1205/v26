# -*- coding: utf-8 -*-
"""
独立的视频预处理脚本：背景压平。
先对整段视频采样多帧算中位数背景模型（车道文字/导流线/斑马线等静止内容会完整保留在背景模型里，
车辆几乎被中位数滤掉），然后逐帧做背景差分，前景(车辆)保持清晰，背景(含文字/标线)替换成
"干净背景模型"的模糊版本，减少这些静止内容对后续检测的干扰。

跟tracktworoi.py完全独立，只负责输出一个新的视频文件，不改动任何跟踪代码。
输出视频路径确定后，把tracktworoi.py的VIDEO_PATH指向它即可。
"""
import cv2
import numpy as np
import time
import os

VIDEO_PATH = r"E:\DJI\cvat\0111_D_stabilized.avi"
OUTPUT_PATH = r"E:\DJI\cvat\0111_D_stabilized_bgsuppress.mp4"

BG_MODEL_N_SAMPLES = 120     # 采样多少帧算中位数背景模型
BG_FG_THRESH = 25            # 前景/背景判定的像素差阈值
BG_FG_DILATE = 15            # 前景mask膨胀核大小，避免车辆边缘被压平
BG_BLUR_KSIZE = 51           # 背景模糊核大小


def build_background_model(video_path, n_samples=120):
    """对整段视频等间隔采样n_samples帧，逐像素取中位数，得到不含车辆的静态背景模型。"""
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
    stack = np.stack(frames, axis=0)
    return np.median(stack, axis=0).astype(np.uint8)


def apply_background_suppression(frame, bg_model, fg_thresh=25, blur_ksize=51, dilate_ksize=15):
    """前景(运动车辆)保持清晰，背景(含车道线/文字/标线等静止内容)替换成"干净背景模型"的模糊版本。"""
    diff = cv2.absdiff(frame, bg_model)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, fg_mask = cv2.threshold(gray_diff, fg_thresh, 255, cv2.THRESH_BINARY)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    if dilate_ksize > 0:
        fg_mask = cv2.dilate(fg_mask, np.ones((dilate_ksize, dilate_ksize), np.uint8))

    bk = int(blur_ksize)
    if bk % 2 == 0:
        bk += 1
    blurred_bg = cv2.GaussianBlur(bg_model, (bk, bk), 0)

    alpha = cv2.GaussianBlur(fg_mask.astype(np.float32) / 255.0, (21, 21), 0)[..., None]
    return (frame.astype(np.float32) * alpha + blurred_bg.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)


def main():
    print(f"[背景压平] 正在采样{BG_MODEL_N_SAMPLES}帧计算背景模型...")
    t0 = time.time()
    bg_model = build_background_model(VIDEO_PATH, BG_MODEL_N_SAMPLES)
    print(f"[背景压平] 背景模型计算完成，耗时{time.time()-t0:.0f}s")
    cv2.imwrite(os.path.splitext(OUTPUT_PATH)[0] + "_bgmodel.jpg", bg_model)
    print("[背景压平] 背景模型预览图已保存")

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
        processed = apply_background_suppression(
            frame, bg_model, fg_thresh=BG_FG_THRESH,
            blur_ksize=BG_BLUR_KSIZE, dilate_ksize=BG_FG_DILATE,
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
