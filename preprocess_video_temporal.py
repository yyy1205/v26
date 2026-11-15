# -*- coding: utf-8 -*-
"""Offline temporal-residual video preprocessing for small-object detection.

The program never changes frame geometry, frame order, or FPS.  It estimates
frame-to-frame camera motion, warps a robust running background into the
current frame, and softly enhances only pixels whose temporal residual is
large.  The generated video can be used as a drop-in source for the existing
detector/tracker.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np


DEFAULT_INPUT = Path(r"E:\DJI\sgybw\video2\DJI_20250302041036_0037_D.MP4")
DEFAULT_OUTPUT_DIR = Path("outputs/temporal_preprocess_video2")


def odd(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 else value + 1


class TemporalResidualEnhancer:
    """Motion-aligned robust background and soft residual enhancement."""

    def __init__(
        self,
        width: int,
        height: int,
        align_scale: float = 0.25,
        background_alpha: float = 0.035,
        residual_low: float = 28.0,
        residual_high: float = 55.0,
        gain: float = 0.5,
        dilate: int = 9,
        feather: int = 9,
        max_mask_ratio: float = 0.30,
    ) -> None:
        self.width = width
        self.height = height
        self.align_scale = align_scale
        self.background_alpha = background_alpha
        self.residual_low = residual_low
        self.residual_high = max(residual_low + 1.0, residual_high)
        self.gain = gain
        self.dilate = odd(dilate)
        self.feather = odd(feather)
        self.max_mask_ratio = max_mask_ratio
        self.background: np.ndarray | None = None
        self.prev_small: np.ndarray | None = None
        self.prev_points: np.ndarray | None = None

    def _small_gray(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, None, fx=self.align_scale, fy=self.align_scale, interpolation=cv2.INTER_AREA)

    def _estimate_warp(self, frame: np.ndarray) -> tuple[np.ndarray, int, bool]:
        """Return affine transform mapping previous coordinates to current."""
        current = self._small_gray(frame)
        identity = np.eye(2, 3, dtype=np.float32)
        if self.prev_small is None:
            self.prev_small = current
            self.prev_points = cv2.goodFeaturesToTrack(
                current, maxCorners=1200, qualityLevel=0.01, minDistance=4, blockSize=5
            )
            return identity, 0, True

        points = self.prev_points
        if points is None or len(points) < 20:
            points = cv2.goodFeaturesToTrack(
                self.prev_small, maxCorners=1200, qualityLevel=0.01, minDistance=4, blockSize=5
            )
        matched = 0
        valid = False
        warp = identity
        if points is not None and len(points) >= 6:
            next_points, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_small,
                current,
                points,
                None,
                winSize=(21, 21),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            )
            good_prev = points[status.ravel() == 1].reshape(-1, 2)
            good_curr = next_points[status.ravel() == 1].reshape(-1, 2)
            matched = len(good_prev)
            if matched >= 6:
                estimated, inliers = cv2.estimateAffinePartial2D(
                    good_prev, good_curr, method=cv2.RANSAC, ransacReprojThreshold=2.0
                )
                if estimated is not None and np.isfinite(estimated).all():
                    estimated = estimated.astype(np.float32)
                    estimated[:, 2] /= self.align_scale
                    # Reject catastrophic estimates rather than corrupting the background.
                    scale = float(np.hypot(estimated[0, 0], estimated[0, 1]))
                    tx, ty = map(float, estimated[:, 2])
                    inlier_ratio = float(inliers.mean()) if inliers is not None else 0.0
                    if 0.97 <= scale <= 1.03 and abs(tx) <= 80 and abs(ty) <= 80 and inlier_ratio >= 0.35:
                        warp = estimated
                        valid = True

        self.prev_small = current
        self.prev_points = cv2.goodFeaturesToTrack(
            current, maxCorners=1200, qualityLevel=0.01, minDistance=4, blockSize=5
        )
        return warp, matched, valid

    def apply(self, frame: np.ndarray) -> tuple[np.ndarray, dict[str, float | int | bool]]:
        warp, matches, alignment_valid = self._estimate_warp(frame)
        frame_f = frame.astype(np.float32)
        if self.background is None:
            self.background = frame_f.copy()
            return frame.copy(), {
                "alignment_valid": True,
                "feature_matches": matches,
                "mask_ratio": 0.0,
                "mean_residual": 0.0,
                "tx": 0.0,
                "ty": 0.0,
            }

        aligned_bg = cv2.warpAffine(
            self.background,
            warp,
            (self.width, self.height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray_bg = cv2.cvtColor(aligned_bg.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        residual = cv2.absdiff(gray_frame, gray_bg)

        # Build regions from a binary seed first. Dilating a low-valued floating
        # residual directly lets compression noise flood most of a 5K frame.
        seed = (residual >= self.residual_low).astype(np.uint8)
        seed = cv2.morphologyEx(seed, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        seed = cv2.dilate(seed, np.ones((self.dilate, self.dilate), np.uint8))
        strength = np.clip(
            (residual - self.residual_low) / (self.residual_high - self.residual_low), 0.0, 1.0
        )
        mask = seed.astype(np.float32) * np.maximum(strength, 0.20)
        mask = cv2.GaussianBlur(mask, (self.feather, self.feather), 0)
        mask_ratio = float((mask >= 0.25).mean())

        # A large residual field usually means failed alignment or a scene cut.
        use_enhancement = alignment_valid and mask_ratio <= self.max_mask_ratio
        if use_enhancement:
            temporal_detail = frame_f - aligned_bg
            processed = np.clip(frame_f + self.gain * temporal_detail * mask[..., None], 0, 255).astype(np.uint8)
        else:
            processed = frame.copy()
            mask.fill(0)

        # Update only stable pixels, so moving vehicles are not quickly absorbed.
        stable_weight = self.background_alpha * (1.0 - mask[..., None])
        self.background = aligned_bg * (1.0 - stable_weight) + frame_f * stable_weight
        return processed, {
            "alignment_valid": alignment_valid,
            "feature_matches": matches,
            "mask_ratio": mask_ratio,
            "mean_residual": float(residual.mean()),
            "tx": float(warp[0, 2]),
            "ty": float(warp[1, 2]),
        }


def parse_ranges(value: str | None, total: int) -> list[tuple[int, int]]:
    if not value:
        return [(0, total - 1)]
    result = []
    for part in value.split(","):
        bounds = part.strip().split(":")
        if len(bounds) != 2:
            raise ValueError(f"Invalid range {part!r}; expected START:END")
        start, end = map(int, bounds)
        if start < 0 or end < start or end >= total:
            raise ValueError(f"Range {part!r} is outside 0:{total - 1}")
        result.append((start, end))
    return result


def open_writer(path: Path, fps: float, size: tuple[int, int], codec: str) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer: {path} (codec={codec})")
    return writer


def process_range(args: argparse.Namespace, metadata: dict, start: int, end: int) -> Path:
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = ".avi" if args.codec.upper() == "FFV1" else ".mp4"
    stem = f"{input_path.stem}_temporal_f{start:04d}-{end:04d}"
    output_path = output_dir / f"{stem}{suffix}"
    log_path = output_dir / f"{stem}.csv"

    cap = cv2.VideoCapture(str(input_path))
    warm_start = max(0, start - args.warmup)
    cap.set(cv2.CAP_PROP_POS_FRAMES, warm_start)
    enhancer = TemporalResidualEnhancer(
        metadata["width"], metadata["height"], args.align_scale, args.background_alpha,
        args.residual_low, args.residual_high, args.gain, args.dilate, args.feather, args.max_mask_ratio,
    )
    writer = open_writer(output_path, metadata["fps"], (metadata["width"], metadata["height"]), args.codec)
    rows = []
    begin = time.time()
    frame_index = warm_start
    written = 0
    while frame_index <= end:
        ok, frame = cap.read()
        if not ok:
            break
        processed, stats = enhancer.apply(frame)
        if frame_index >= start:
            writer.write(processed)
            rows.append({"source_frame": frame_index, **stats})
            written += 1
        frame_index += 1
        if written and written % 25 == 0:
            print(f"[{start}:{end}] written {written}/{end-start+1}", flush=True)
    cap.release()
    writer.release()
    if written != end - start + 1:
        raise RuntimeError(f"Expected {end-start+1} frames, wrote {written}")

    with log_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer_csv = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer_csv.writeheader()
        writer_csv.writerows(rows)
    print(f"Created {output_path} in {time.time()-begin:.1f}s")
    return output_path


def verify_video(path: Path, expected_frames: int, metadata: dict) -> dict:
    cap = cv2.VideoCapture(str(path))
    readable = cap.isOpened()
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if readable else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if readable else 0
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if readable else 0
    fps = float(cap.get(cv2.CAP_PROP_FPS)) if readable else 0.0
    decoded = 0
    while readable:
        ok, _ = cap.read()
        if not ok:
            break
        decoded += 1
    cap.release()
    result = {
        "path": str(path.resolve()), "readable": readable, "header_frames": frames, "decoded_frames": decoded,
        "expected_frames": expected_frames, "width": width, "height": height, "fps": fps,
        "passed": readable and decoded == expected_frames and width == metadata["width"]
        and height == metadata["height"] and abs(fps - metadata["fps"]) < 0.01,
    }
    if not result["passed"]:
        raise RuntimeError(f"Output verification failed: {result}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--ranges", help="Inclusive zero-based ranges, e.g. 250:290,290:320")
    parser.add_argument("--warmup", type=int, default=45)
    parser.add_argument("--codec", default="FFV1", choices=("FFV1", "mp4v"))
    parser.add_argument("--align-scale", type=float, default=0.25)
    parser.add_argument("--background-alpha", type=float, default=0.035)
    parser.add_argument("--residual-low", type=float, default=28.0)
    parser.add_argument("--residual-high", type=float, default=55.0)
    parser.add_argument("--gain", type=float, default=0.5)
    parser.add_argument("--dilate", type=int, default=9)
    parser.add_argument("--feather", type=int, default=9)
    parser.add_argument("--max-mask-ratio", type=float, default=0.30)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open input video: {args.input}")
    metadata = {
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    ranges = parse_ranges(args.ranges, metadata["frames"])
    outputs = []
    for start, end in ranges:
        path = process_range(args, metadata, start, end)
        outputs.append(verify_video(path, end - start + 1, metadata))

    report = {"input": str(Path(args.input).resolve()), "source": metadata, "parameters": vars(args), "outputs": outputs}
    report_path = Path(args.output_dir) / "verification.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"All outputs verified. Report: {report_path}")


if __name__ == "__main__":
    main()
