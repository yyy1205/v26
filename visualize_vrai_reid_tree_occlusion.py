# -*- coding: utf-8 -*-
"""Visualize VRAI ReID evidence before/after the three tree occlusions."""

from pathlib import Path

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon as MplPolygon

from evaluate_vrai_reid_tree_occlusion import GT_XML, OUT_DIR, VIDEOS, oriented_crop, parse_gt


mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 8,
    "axes.spines.right": False,
    "axes.spines.top": False,
})

EVENTS = {
    122: {"pre": 262, "post": 278, "gap": 15},
    505: {"pre": 1834, "post": 1854, "gap": 19},
    524: {"pre": 2621, "post": 2654, "gap": 36},
}
COLORS = {"true": "#2A6FBB", "wrong": "#D55E00", "threshold": "#555555"}


def read_frame(cap, frame_id):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Cannot read frame {frame_id}")
    return frame


def rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def context_crop(image, rect, radius=245):
    (cx, cy), _, _ = rect
    x1, y1 = max(0, int(cx) - radius), max(0, int(cy) - radius)
    x2, y2 = min(image.shape[1], int(cx) + radius), min(image.shape[0], int(cy) + radius)
    crop = image[y1:y2, x1:x2].copy()
    points = cv2.boxPoints(rect) - np.array([x1, y1], dtype=np.float32)
    return crop, points


def show_context(ax, image, rect, label, color):
    crop, points = context_crop(image, rect)
    ax.imshow(rgb(crop))
    ax.add_patch(MplPolygon(points, closed=True, fill=False, edgecolor=color, linewidth=2.0))
    ax.text(0.02, 0.96, label, transform=ax.transAxes, va="top", color="white", fontsize=8,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "black", "alpha": 0.70, "edgecolor": "none"})
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def show_chip_pair(ax, true_chip, wrong_chip, wrong_gid):
    canvas_h = 150
    chips = []
    for chip in (true_chip, wrong_chip):
        scale = min(220 / chip.shape[1], 95 / chip.shape[0])
        resized = cv2.resize(chip, (max(1, int(chip.shape[1] * scale)), max(1, int(chip.shape[0] * scale))))
        chips.append(resized)
    canvas = np.full((canvas_h, 480, 3), 245, dtype=np.uint8)
    for i, chip in enumerate(chips):
        x0 = 10 + i * 240 + (220 - chip.shape[1]) // 2
        y0 = 15 + (95 - chip.shape[0]) // 2
        canvas[y0:y0 + chip.shape[0], x0:x0 + chip.shape[1]] = chip
    ax.imshow(rgb(canvas))
    ax.text(0.25, 0.04, "true vehicle", transform=ax.transAxes, ha="center", color=COLORS["true"], weight="bold")
    ax.text(0.75, 0.04, f"hardest wrong (GT {wrong_gid})", transform=ax.transAxes, ha="center",
            color=COLORS["wrong"], weight="bold")
    ax.set_axis_off()


def main():
    results = pd.read_csv(OUT_DIR / "frame_results.csv")
    gt = parse_gt(GT_XML)
    cap = cv2.VideoCapture(str(VIDEOS["original"]))
    fig = plt.figure(figsize=(13.6, 8.4), constrained_layout=True)
    grid = fig.add_gridspec(3, 4, width_ratios=[1.15, 1.15, 1.15, 0.95])

    for row, (gid, event) in enumerate(EVENTS.items()):
        pre, post = event["pre"], event["post"]
        pre_frame, post_frame = read_frame(cap, pre), read_frame(cap, post)
        true_rect = gt[post][gid][1]
        row_original = results[(results.video == "original") & (results.gt_id == gid) & (results.post_frame == post)].iloc[0]
        wrong_gid = int(row_original.best_wrong_gid)
        wrong_rect = gt[post][wrong_gid][1]

        ax = fig.add_subplot(grid[row, 0])
        show_context(ax, pre_frame, gt[pre][gid][1], f"before canopy | frame {pre}", COLORS["true"])
        ax.set_ylabel(f"GT {gid}\n{event['gap']}-frame gap", fontsize=10, weight="bold")

        ax = fig.add_subplot(grid[row, 1])
        show_context(ax, post_frame, true_rect, f"reappearance | frame {post}", COLORS["true"])

        true_chip = oriented_crop(post_frame, true_rect)
        wrong_chip = oriented_crop(post_frame, wrong_rect)
        ax = fig.add_subplot(grid[row, 2])
        show_chip_pair(ax, true_chip, wrong_chip, wrong_gid)

        ax = fig.add_subplot(grid[row, 3])
        subset = results[(results.gt_id == gid) & (results.post_frame == post)].set_index("video")
        x = np.array([0, 1])
        width = 0.34
        true_vals = [subset.loc[v, "true_similarity"] for v in ("original", "temporal")]
        wrong_vals = [subset.loc[v, "best_wrong_similarity"] for v in ("original", "temporal")]
        ax.bar(x - width / 2, true_vals, width, color=COLORS["true"], label="true")
        ax.bar(x + width / 2, wrong_vals, width, color=COLORS["wrong"], label="hardest wrong")
        ax.axhline(0.65, color=COLORS["threshold"], linestyle="--", linewidth=1, label="current threshold")
        ax.set_xticks(x, ["original", "temporal"])
        ax.set_ylim(0.50, 0.90)
        ax.set_ylabel("cosine similarity")
        for xi, tv, wv in zip(x, true_vals, wrong_vals):
            ax.text(xi - width / 2, tv + 0.008, f"{tv:.3f}", ha="center", fontsize=7)
            ax.text(xi + width / 2, wv + 0.008, f"{wv:.3f}", ha="center", fontsize=7)
        if row == 0:
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.26), ncol=2, fontsize=7)

    cap.release()
    fig.suptitle(
        "VRAI OSNet identity evidence across tree-canopy occlusion\n"
        "Appearance usually retrieves the correct vehicle, but margins can be unsafe",
        fontsize=13, weight="bold",
    )
    fig.text(0.5, -0.012,
             "Blue boxes/bars: the same GT vehicle. Orange: the most similar different vehicle among all visible GT candidates. "
             "Dashed line: current ReID recovery threshold (0.65).",
             ha="center", fontsize=8)
    base = OUT_DIR / "vrai_reid_tree_occlusion_evidence"
    fig.savefig(base.with_suffix(".png"), dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(base)


if __name__ == "__main__":
    main()
