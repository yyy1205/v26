#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate evidence-first visual diagnostics for the four latest video2 ID switches.

The script never reruns detection or tracking. It combines the saved production
outputs, the GT XML, and the source video. Each event receives:
  1) seven annotated source-frame crops around the switch;
  2) timelines for assigned ID, confidence/IoU, box size, and OBB angle;
  3) a CSV containing every plotted value and the nearest competing prediction.

Observable evidence is kept separate from interpretation. In particular, a missing
tracker output is not described as a missing raw detector output because raw detector
proposals were not saved by the production run.
"""

from __future__ import annotations

import csv
import math
import os
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import eval_mot


VIDEO_PATH = Path(r"E:\DJI\sgybw\video2\DJI_20250302041036_0037_D.MP4")
GT_XML = Path(r"D:\ultralytics-8.1.41\outputs\v8e53-2condence_20260521_150313\mota.xml")
RUN_DIR = Path(r"D:\yolo26-tracking\outputs\yolo26-obb_20260805_192534")
PRED_TXT = RUN_DIR / "cvat_eval.txt"
TRACK_RESULT = RUN_DIR / "track_result.txt"
OUT_DIR = Path(r"D:\yolo26-tracking\outputs\idsw_evidence_latest")

EVENTS = [
    # (GT frame, GT id, old prediction id, new prediction id)
    (278, 122, 183, 188),
    (312, 30, 100, 193),
    (1856, 505, 421, 425),
    (2654, 524, 558, 564),
]
WINDOW = 30
IMAGE_OFFSETS = (-15, -5, -1, 0, 1, 5, 15)
PAD = 180

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
})

COLORS = {
    "gt": "#2F6BFF",
    "old": "#D84A4A",
    "new": "#E6A700",
    "other": "#51A36D",
    "missing": "#8A8A8A",
}


def load_track_result(path: Path):
    """Return per-frame/per-id pixel metrics saved for every tracker output."""
    rows = defaultdict(dict)
    with path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 8:
                continue
            frame, tid = int(parts[0]), int(parts[1])
            rows[frame][tid] = {
                "conf": float(parts[7]),
                "width": float(parts[4]),
                "height": float(parts[5]),
                "angle_deg": math.degrees(float(parts[6])),
            }
    return rows


def canonical_box(w: float, h: float, angle_deg: float):
    """Rotation-invariant long/short edges and a 180-degree-periodic angle."""
    long_side, short_side = max(w, h), min(w, h)
    angle = ((angle_deg + 90.0) % 180.0) - 90.0
    return long_side, short_side, angle


def match_frame(gt_list, pred_list, iou_thresh=0.5):
    """Reproduce the evaluator's Hungarian assignment for one frame."""
    if not gt_list or not pred_list:
        return {}, {}
    cost = np.ones((len(gt_list), len(pred_list)), dtype=np.float64)
    ious = np.zeros_like(cost)
    for i, (_gid, gcx, gcy, gw, gh, grot) in enumerate(gt_list):
        for j, (_tid, pts) in enumerate(pred_list):
            (pcx, pcy), _, _ = cv2.minAreaRect(pts)
            if math.hypot(gcx - pcx, gcy - pcy) > eval_mot.CENTER_DIST_FILTER:
                continue
            gt_rr = ((float(gcx), float(gcy)), (float(gw), float(gh)), float(grot))
            pred_rr = cv2.minAreaRect(pts.astype(np.float32))
            iou = eval_mot._iou(gt_rr, pred_rr)
            ious[i, j] = iou
            cost[i, j] = 1.0 - iou
    from scipy.optimize import linear_sum_assignment

    rr, cc = linear_sum_assignment(cost)
    matches = {}
    matched_ious = {}
    for i, j in zip(rr, cc):
        if ious[i, j] >= iou_thresh:
            matches[gt_list[i][0]] = pred_list[j][0]
            matched_ious[gt_list[i][0]] = float(ious[i, j])
    return matches, matched_ious


def build_event_rows(gt, pred, track_metrics, event):
    switch_gt_frame, gt_id, old_pid, new_pid = event
    rows = []
    for gtf in range(switch_gt_frame - WINDOW, switch_gt_frame + WINDOW + 1):
        gt_box = next((b for b in gt.get(gtf, []) if b[0] == gt_id), None)
        if gt_box is None:
            continue
        pred_frame = gtf + 1
        pred_list = pred.get(pred_frame, [])
        matches, matched_ious = match_frame(gt.get(gtf, []), pred_list)
        assigned = matches.get(gt_id)
        assigned_iou = matched_ious.get(gt_id, 0.0)

        # Best-overlap candidate is recorded independently as a control against
        # assignment competition from other GT objects.
        candidate_rows = []
        for tid, pts in pred_list:
            gt_rr = ((float(gt_box[1]), float(gt_box[2])),
                     (float(gt_box[3]), float(gt_box[4])), float(gt_box[5]))
            iou = eval_mot._iou(gt_rr, cv2.minAreaRect(pts.astype(np.float32)))
            (cx, cy), (w, h), angle = cv2.minAreaRect(pts)
            candidate_rows.append((iou, tid, cx, cy, w, h, angle))
        candidate_rows.sort(reverse=True)
        best = candidate_rows[0] if candidate_rows else (0.0, None, np.nan, np.nan, np.nan, np.nan, np.nan)
        second = candidate_rows[1] if len(candidate_rows) > 1 else (0.0, None, np.nan, np.nan, np.nan, np.nan, np.nan)

        metric = track_metrics.get(pred_frame, {}).get(assigned, {}) if assigned is not None else {}
        w = metric.get("width", best[4] if assigned == best[1] else np.nan)
        h = metric.get("height", best[5] if assigned == best[1] else np.nan)
        angle = metric.get("angle_deg", best[6] if assigned == best[1] else np.nan)
        long_side, short_side, canon_angle = canonical_box(w, h, angle) if np.isfinite(w) else (np.nan, np.nan, np.nan)
        rows.append({
            "gt_frame": gtf,
            "pred_frame": pred_frame,
            "relative_frame": gtf - switch_gt_frame,
            "gt_id": gt_id,
            "old_pid": old_pid,
            "new_pid": new_pid,
            "assigned_pid": assigned,
            "assigned_iou": assigned_iou,
            "confidence": metric.get("conf", np.nan),
            "long_side": long_side,
            "short_side": short_side,
            "angle_deg": canon_angle,
            "best_pid": best[1],
            "best_iou": best[0],
            "second_pid": second[1],
            "second_iou": second[0],
            "gt_cx": gt_box[1],
            "gt_cy": gt_box[2],
            "gt_w": gt_box[3],
            "gt_h": gt_box[4],
            "gt_angle": gt_box[5],
        })
    return rows


def draw_box(ax, pts, color, label, lw=1.5):
    pts = np.vstack([pts, pts[0]])
    ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw, label=label)


def read_crop(cap, video_frame, gt_box):
    cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame - 1)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Cannot read video frame {video_frame}")
    _, cx, cy, _, _, _ = gt_box
    x1, y1 = max(0, int(cx - PAD)), max(0, int(cy - PAD))
    x2, y2 = min(frame.shape[1], int(cx + PAD)), min(frame.shape[0], int(cy + PAD))
    return cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB), x1, y1


def make_event_figure(gt, pred, rows, event, cap):
    switch_gt_frame, gt_id, old_pid, new_pid = event
    fig = plt.figure(figsize=(12.0, 7.2), constrained_layout=True)
    gs = fig.add_gridspec(3, 7, height_ratios=[1.35, 1.0, 1.0])

    for col, off in enumerate(IMAGE_OFFSETS):
        gtf = switch_gt_frame + off
        pred_frame = gtf + 1
        gt_box = next((b for b in gt.get(gtf, []) if b[0] == gt_id), None)
        ax = fig.add_subplot(gs[0, col])
        if gt_box is None:
            ax.text(0.5, 0.5, "GT absent", ha="center", va="center")
            ax.axis("off")
            continue
        crop, x1, y1 = read_crop(cap, pred_frame, gt_box)
        ax.imshow(crop)
        gt_pts = cv2.boxPoints(((gt_box[1] - x1, gt_box[2] - y1), (gt_box[3], gt_box[4]), gt_box[5]))
        draw_box(ax, gt_pts, COLORS["gt"], "GT")
        for tid, pts in pred.get(pred_frame, []):
            shifted = pts - np.array([x1, y1], dtype=np.float32)
            (pcx, pcy), _, _ = cv2.minAreaRect(pts)
            if not (x1 <= pcx <= x1 + crop.shape[1] and y1 <= pcy <= y1 + crop.shape[0]):
                continue
            color = COLORS["old"] if tid == old_pid else COLORS["new"] if tid == new_pid else COLORS["other"]
            draw_box(ax, shifted, color, f"ID {tid}", lw=1.8 if tid in (old_pid, new_pid) else 0.8)
            if tid in (old_pid, new_pid):
                ax.text(pcx - x1, pcy - y1, str(tid), color="white", fontsize=7,
                        ha="center", va="center", bbox=dict(facecolor=color, edgecolor="none", pad=1.2))
        ax.set_title(f"{off:+d}  (GT f{gtf})", color="#B21F2D" if off == 0 else "black")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#B21F2D" if off == 0 else "#C8C8C8")

    x = np.array([r["relative_frame"] for r in rows])
    assigned = [r["assigned_pid"] for r in rows]
    y_id = np.array([0 if p is None else 1 if p == old_pid else 2 if p == new_pid else 3 for p in assigned])

    ax_id = fig.add_subplot(gs[1, :2])
    ax_id.step(x, y_id, where="mid", color="#4B5563", lw=1.3)
    ax_id.scatter(x, y_id, c=[COLORS["missing"] if p is None else COLORS["old"] if p == old_pid
                              else COLORS["new"] if p == new_pid else COLORS["other"] for p in assigned], s=12)
    ax_id.set_yticks([0, 1, 2, 3], ["no match", f"old {old_pid}", f"new {new_pid}", "other"])
    ax_id.set_ylabel("GT assignment")

    ax_q = fig.add_subplot(gs[1, 2:5])
    ax_q.plot(x, [r["assigned_iou"] for r in rows], color="#3B6FB6", lw=1.4, label="assigned IoU")
    ax_q.plot(x, [r["best_iou"] for r in rows], color="#8DB3DF", lw=1.0, ls="--", label="best candidate IoU")
    ax_q.plot(x, [r["confidence"] for r in rows], color="#D58B24", lw=1.2, label="track confidence")
    ax_q.axhline(0.5, color="#777777", lw=0.8, ls=":")
    ax_q.set_ylim(0, 1.03); ax_q.set_ylabel("score")
    ax_q.legend(ncol=3, loc="lower center", fontsize=6)

    ax_s = fig.add_subplot(gs[1, 5:])
    ax_s.plot(x, [r["long_side"] for r in rows], color="#4C78A8", label="long edge")
    ax_s.plot(x, [r["short_side"] for r in rows], color="#72B7B2", label="short edge")
    ax_s.set_ylabel("box edge (px)"); ax_s.legend(fontsize=6)

    ax_a = fig.add_subplot(gs[2, :3])
    ax_a.plot(x, [r["angle_deg"] for r in rows], color="#7A5195", lw=1.2)
    ax_a.set_ylabel("canonical OBB angle (deg)"); ax_a.set_xlabel("frames relative to switch")

    ax_c = fig.add_subplot(gs[2, 3:])
    margin = np.array([r["best_iou"] - r["second_iou"] for r in rows])
    ax_c.plot(x, margin, color="#2E8B57", lw=1.2)
    ax_c.axhline(0, color="#777777", lw=0.8, ls=":")
    ax_c.set_ylabel("best minus 2nd IoU")
    ax_c.set_xlabel("frames relative to switch")

    for ax in (ax_id, ax_q, ax_s, ax_a, ax_c):
        ax.axvline(0, color="#B21F2D", lw=1.0, ls="--")
        ax.grid(axis="y", color="#E8E8E8", lw=0.5)

    fig.suptitle(f"GT {gt_id}: prediction ID {old_pid} → {new_pid} at GT frame {switch_gt_frame}",
                 fontsize=11, fontweight="bold")
    return fig


def write_rows(path: Path, rows):
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    eval_mot.FRAME_W, eval_mot.FRAME_H = 5120, 2700
    gt = eval_mot.parse_gt(str(GT_XML))
    pred = eval_mot.parse_pred(str(PRED_TXT))
    track_metrics = load_track_result(TRACK_RESULT)
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {VIDEO_PATH}")

    all_rows = []
    for event in EVENTS:
        rows = build_event_rows(gt, pred, track_metrics, event)
        all_rows.extend(rows)
        gtf, gid, old_pid, new_pid = event
        stem = OUT_DIR / f"GT{gid}_f{gtf}_{old_pid}to{new_pid}"
        write_rows(stem.with_suffix(".csv"), rows)
        fig = make_event_figure(gt, pred, rows, event, cap)
        fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight", facecolor="white")
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"wrote {stem.name}")

    write_rows(OUT_DIR / "all_events_evidence.csv", all_rows)
    cap.release()
    print(f"Evidence bundle: {OUT_DIR}")


if __name__ == "__main__":
    main()
