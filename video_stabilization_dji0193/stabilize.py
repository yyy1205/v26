from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


@dataclass
class Config:
    analysis_width: int = 960
    features: int = 1800
    ratio: float = 0.72
    ransac_threshold: float = 2.5
    min_matches: int = 12
    min_inlier_ratio: float = 0.35
    smooth_window: int = 41
    camera_lock: float = 0.85
    crop: float = 0.04
    border: str = "crop"
    codec: str = "libx264"
    crf: int = 18
    correction_strength: float = 0.15


def ffmpeg_path() -> str:
    for name in ("ffmpeg", r"C:\Tools\ffmpeg\ffmpeg-9.0.1-essentials_build\bin\ffmpeg.exe"):
        p = shutil.which(name) if name == "ffmpeg" else name
        if p and Path(p).exists():
            return p
    raise FileNotFoundError("找不到 ffmpeg，请将 ffmpeg 加入 PATH 或修改 ffmpeg_path()。")


def similarity_matrix(tx: float, ty: float, angle_deg: float, scale: float = 1.0) -> np.ndarray:
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[scale * c, -scale * s, tx], [scale * s, scale * c, ty], [0, 0, 1]], dtype=np.float64)


def center_conjugate(m: np.ndarray, width: int, height: int) -> np.ndarray:
    c = np.array([[1, 0, width / 2], [0, 1, height / 2], [0, 0, 1]], dtype=np.float64)
    return c @ m @ np.linalg.inv(c)


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n < 3:
        return values.copy()
    w = max(3, int(window))
    if w % 2 == 0:
        w += 1
    w = min(w, n if n % 2 else n - 1)
    if w < 3:
        return values.copy()
    # 先用短中值窗去除单帧匹配跳变，再用高斯窗抑制低频漂移。
    median_w = 9 if n >= 9 else (n if n % 2 else n - 1)
    if median_w >= 3:
        mpad = median_w // 2
        med = np.empty_like(values)
        for j in range(values.shape[1]):
            padded = np.pad(values[:, j], (mpad, mpad), mode="edge")
            med[:, j] = np.array([np.median(padded[k:k + median_w]) for k in range(n)])
    else:
        med = values.copy()
    sigma = max(1.0, w / 6.0)
    x = np.arange(w) - w // 2
    kernel = np.exp(-(x * x) / (2 * sigma * sigma))
    kernel /= kernel.sum()
    out = np.empty_like(values)
    pad = w // 2
    for j in range(values.shape[1]):
        padded = np.pad(med[:, j], (pad, pad), mode="edge")
        out[:, j] = np.convolve(padded, kernel, mode="valid")
    return out


def frame_gray(frame: np.ndarray, width: int) -> np.ndarray:
    height = round(frame.shape[0] * width / frame.shape[1])
    return cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (width, height), interpolation=cv2.INTER_AREA)


def estimate_pair(prev: np.ndarray, curr: np.ndarray, orb: cv2.ORB, cfg: Config):
    k1, d1 = orb.detectAndCompute(prev, None)
    k2, d2 = orb.detectAndCompute(curr, None)
    info = {"keypoints": len(k2), "matches": 0, "inliers": 0, "inlier_ratio": 0.0, "valid": False}
    identity = np.eye(2, 3, dtype=np.float32)
    if d1 is None or d2 is None:
        return identity, info
    matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1, d2, k=2)
    good = [m for m, n in matches if m.distance < cfg.ratio * n.distance]
    info["matches"] = len(good)
    if len(good) < cfg.min_matches:
        return identity, info
    p = np.float32([k1[m.queryIdx].pt for m in good])
    q = np.float32([k2[m.trainIdx].pt for m in good])
    matrix, mask = cv2.estimateAffinePartial2D(p, q, method=cv2.RANSAC, ransacReprojThreshold=cfg.ransac_threshold)
    if matrix is None or mask is None:
        return identity, info
    info["inliers"] = int(mask.sum())
    info["inlier_ratio"] = float(mask.mean())
    if info["inlier_ratio"] < cfg.min_inlier_ratio:
        return identity, info
    info["valid"] = True
    return matrix.astype(np.float32), info


def analyze(input_path: str, cfg: Config, report_path: Optional[str] = None):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{input_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orb = cv2.ORB_create(cfg.features, 1.2, 8, fastThreshold=12)
    ok, first = cap.read()
    if not ok:
        raise RuntimeError("视频没有可读取的画面")
    prev = frame_gray(first, cfg.analysis_width)
    trajectory = [np.array([0.0, 0.0, 0.0, 1.0])]
    logs = [{"frame": 0, "valid": True, "tx": 0.0, "ty": 0.0, "rotation_deg": 0.0, "scale": 1.0}]
    # 固定视角视频直接与首帧匹配，避免逐帧累积造成长期漂移。
    reference = prev.copy()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        curr = frame_gray(frame, cfg.analysis_width)
        m, info = estimate_pair(reference, curr, orb, cfg)
        a, b = m[0, 0], m[1, 0]
        pose = np.array([m[0, 2], m[1, 2], math.degrees(math.atan2(b, a)), math.hypot(a, b)])
        trajectory.append(pose)
        logs.append({"frame": len(logs), "time_s": len(logs) / fps, **info, "tx": float(pose[0]), "ty": float(pose[1]), "rotation_deg": float(pose[2]), "scale": float(pose[3])})
        prev = curr
        if len(logs) % 500 == 0:
            print(f"分析 {len(logs)}/{count} 帧", flush=True)
    cap.release()
    arr = np.asarray(trajectory)
    increments = np.diff(arr[:, :3], axis=0) if len(arr) > 1 else np.zeros((0, 3))
    report = {
        "input": str(input_path), "width": width, "height": height, "fps": fps,
        "frames": len(logs), "duration_s": len(logs) / fps,
        "config": asdict(cfg), "frames_detail": logs,
        "summary": {
            "valid_ratio": float(np.mean([x.get("valid", False) for x in logs[1:]]) if len(logs) > 1 else 1.0),
            "mean_inlier_ratio": float(np.mean([x.get("inlier_ratio", 0.0) for x in logs[1:]]) if len(logs) > 1 else 1.0),
            "increment_translation_p95_px": [float(np.percentile(np.abs(increments[:, j]), 95)) for j in (0, 1)] if len(increments) else [0, 0],
            "increment_rotation_p95_deg": float(np.percentile(np.abs(increments[:, 2]), 95)) if len(increments) else 0.0,
            "trajectory_translation_range_px": [float(np.ptp(arr[:, 0])), float(np.ptp(arr[:, 1]))],
            "trajectory_rotation_range_deg": float(np.ptp(arr[:, 2])),
        },
    }
    if report_path:
        Path(report_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report, arr


def stabilize(input_path: str, output_path: str, report_path: Optional[str], cfg: Config):
    report, poses = analyze(input_path, cfg, report_path)
    smoothed = smooth(poses, cfg.smooth_window)
    if cfg.camera_lock > 0:
        anchor = np.median(smoothed, axis=0)
        smoothed = smoothed * (1.0 - cfg.camera_lock) + anchor * cfg.camera_lock
    cap = cv2.VideoCapture(input_path)
    ok, first = cap.read()
    if not ok:
        raise RuntimeError("无法重新打开输入视频")
    h, w = first.shape[:2]
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    scale_x = w / cfg.analysis_width
    scale_y = h / round(h * cfg.analysis_width / w)
    crop_scale = 1.0 / max(1e-6, 1.0 - 2.0 * cfg.crop) if cfg.border == "crop" else 1.0
    ffmpeg = ffmpeg_path()
    cmd = [ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-i", input_path, "-map", "0:v:0", "-map", "1:a?", "-c:v", cfg.codec, "-preset", "medium", "-crf", str(cfg.crf), "-pix_fmt", "yuv420p", "-c:a", "copy", "-shortest", output_path]
    stderr_path = str(Path(output_path).with_suffix(".ffmpeg.log"))
    with open(stderr_path, "wb") as err:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=err)
        i = 0
        try:
            while ok and i < len(smoothed):
                observed = similarity_matrix(*poses[i])
                effective = poses[i] + cfg.correction_strength * (smoothed[i] - poses[i])
                target = similarity_matrix(*effective)
                correction = target @ np.linalg.inv(observed)
                # Motion was estimated on the analysis image; map it back to 4K pixels.
                scale_map = np.diag([scale_x, scale_y, 1.0])
                correction = scale_map @ correction @ np.linalg.inv(scale_map)
                if crop_scale != 1.0:
                    crop_matrix = center_conjugate(np.diag([crop_scale, crop_scale, 1.0]), w, h)
                    correction = crop_matrix @ center_conjugate(correction, w, h)
                else:
                    correction = center_conjugate(correction, w, h)
                border = cv2.BORDER_REFLECT if cfg.border == "reflect" else cv2.BORDER_CONSTANT
                warped = cv2.warpAffine(first, correction[:2].astype(np.float32), (w, h), borderMode=border)
                proc.stdin.write(np.ascontiguousarray(warped).tobytes())
                i += 1
                ok, first = cap.read()
                if i % 500 == 0:
                    print(f"输出 {i}/{len(smoothed)} 帧", flush=True)
            proc.stdin.close()
            rc = proc.wait()
        except Exception:
            proc.kill()
            raise
    cap.release()
    if rc != 0:
        raise RuntimeError(f"ffmpeg 输出失败，详见 {stderr_path}")
    report["output"] = str(output_path)
    report["output_frames"] = i
    report["output_crop_scale"] = crop_scale
    if report_path:
        Path(report_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    ap = argparse.ArgumentParser(description="固定视角无人机视频稳像")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output")
    ap.add_argument("--report", default="dji0193_report.json")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--analysis-width", type=int, default=960)
    ap.add_argument("--smooth-window", type=int, default=41)
    ap.add_argument("--camera-lock", type=float, default=0.85)
    ap.add_argument("--crop", type=float, default=0.04)
    ap.add_argument("--border", choices=["crop", "reflect"], default="crop")
    ap.add_argument("--correction-strength", type=float, default=0.15)
    args = ap.parse_args()
    cfg = Config(analysis_width=args.analysis_width, smooth_window=args.smooth_window, camera_lock=args.camera_lock, crop=args.crop, border=args.border, correction_strength=args.correction_strength)
    if args.analyze_only:
        report, _ = analyze(args.input, cfg, args.report)
        print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    else:
        output = args.output or str(Path(args.input).with_name(Path(args.input).stem + "_stabilized.mp4"))
        report = stabilize(args.input, output, args.report, cfg)
        print(json.dumps({"output": output, "frames": report["output_frames"], "summary": report["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
