# -*- coding: utf-8 -*-
"""Evidence test for VRAI OSNet ReID across the three video2 tree occlusions."""

from __future__ import annotations

import csv
import json
import os
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, r"D:\deep-person-reid-master")

import cv2
import numpy as np
import torch
import torchreid


MODEL = Path(r"D:\deep-person-reid-master\scripts\log\osnet_ain_x1_0_vrai_softmax_cosinelr\model\model.pth.tar-100")
GT_XML = Path(r"D:\ultralytics-8.1.41\outputs\v8e53-2condence_20260521_150313\mota.xml")
VIDEOS = {
    "original": Path(r"E:\DJI\sgybw\video2\DJI_20250302041036_0037_D.MP4"),
    "temporal": Path(r"D:\yolo26-tracking\outputs\temporal_preprocess_video2_full\DJI_20250302041036_0037_D_temporal_f0000-3822.mp4"),
}
OUT_DIR = Path(r"D:\yolo26-tracking\outputs\vrai_reid_tree_occlusion")

# Frames use the native zero-based GT/video index. Multiple anchors reduce one-frame noise.
EVENTS = {
    122: {"pre": [260, 261, 262], "post": [278, 279, 280]},
    505: {"pre": [1832, 1833, 1834], "post": [1854, 1856, 1858]},
    524: {"pre": [2619, 2620, 2621], "post": [2643, 2654, 2656]},
}


def parse_gt(path: Path):
    gt = defaultdict(dict)
    root = ET.parse(path).getroot()
    for track in root.findall("track"):
        gid = int(track.get("id"))
        label = track.get("label", "unknown")
        for shape in list(track.findall("box")) + list(track.findall("polygon")):
            if shape.get("outside") == "1":
                continue
            frame = int(shape.get("frame"))
            if shape.tag == "polygon":
                pts = np.array([[float(v) for v in p.split(",")] for p in shape.get("points").split(";")], np.float32)
                rect = cv2.minAreaRect(pts)
            else:
                x1, y1 = float(shape.get("xtl")), float(shape.get("ytl"))
                x2, y2 = float(shape.get("xbr")), float(shape.get("ybr"))
                rect = (((x1 + x2) / 2, (y1 + y2) / 2), (x2 - x1, y2 - y1), float(shape.get("rotation", 0)))
            gt[frame][gid] = (label, rect)
    return gt


def oriented_crop(image: np.ndarray, rect, pad: float = 1.18) -> np.ndarray | None:
    (cx, cy), (w, h), angle = rect
    if w < 2 or h < 2:
        return None
    # Normalize so output width follows the vehicle long axis.
    if h > w:
        w, h = h, w
        angle += 90.0
    radius = int(np.hypot(w, h) * pad / 2) + 4
    x1, y1 = max(0, int(cx) - radius), max(0, int(cy) - radius)
    x2, y2 = min(image.shape[1], int(cx) + radius + 1), min(image.shape[0], int(cy) + radius + 1)
    if x2 <= x1 or y2 <= y1:
        return None
    patch = image[y1:y2, x1:x2]
    local = (cx - x1, cy - y1)
    matrix = cv2.getRotationMatrix2D(local, angle, 1.0)
    rotated = cv2.warpAffine(patch, matrix, (patch.shape[1], patch.shape[0]), borderMode=cv2.BORDER_REPLICATE)
    size = (max(2, int(round(w * pad))), max(2, int(round(h * pad))))
    return cv2.getRectSubPix(rotated, size, local)


def read_frames(path: Path, wanted: set[int]):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {path}")
    result = {}
    for frame_id in sorted(wanted):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, image = cap.read()
        if not ok:
            raise RuntimeError(f"Cannot read frame {frame_id} from {path}")
        result[frame_id] = image
    cap.release()
    return result


def normalize(features: np.ndarray) -> np.ndarray:
    return features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gt = parse_gt(GT_XML)
    wanted = {f for e in EVENTS.values() for phase in e.values() for f in phase}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL} on {device}")
    extractor = torchreid.utils.FeatureExtractor(
        model_name="osnet_ain_x1_0", model_path=str(MODEL), device=device, image_size=(256, 128)
    )

    all_rows = []
    summary = []
    for video_name, video_path in VIDEOS.items():
        frames = read_frames(video_path, wanted)
        for gid, event in EVENTS.items():
            anchor_paths = []
            for frame_id in event["pre"]:
                item = gt.get(frame_id, {}).get(gid)
                if item is None:
                    continue
                crop = oriented_crop(frames[frame_id], item[1])
                path = OUT_DIR / f"{video_name}_gt{gid}_anchor_f{frame_id}.png"
                cv2.imwrite(str(path), crop)
                anchor_paths.append(str(path))
            anchor_feats = normalize(extractor(anchor_paths).cpu().numpy())
            anchor = normalize(anchor_feats.mean(axis=0, keepdims=True))[0]

            for frame_id in event["post"]:
                candidate_paths, candidate_ids = [], []
                for candidate_gid, (_, rect) in gt.get(frame_id, {}).items():
                    crop = oriented_crop(frames[frame_id], rect)
                    if crop is None:
                        continue
                    path = OUT_DIR / f"{video_name}_eventgt{gid}_f{frame_id}_cand{candidate_gid}.png"
                    cv2.imwrite(str(path), crop)
                    candidate_paths.append(str(path))
                    candidate_ids.append(candidate_gid)
                feats = normalize(extractor(candidate_paths).cpu().numpy())
                sims = feats @ anchor
                order = np.argsort(-sims)
                true_index = candidate_ids.index(gid)
                true_sim = float(sims[true_index])
                true_rank = int(np.where(order == true_index)[0][0]) + 1
                best_wrong_index = next(int(i) for i in order if candidate_ids[int(i)] != gid)
                wrong_sim = float(sims[best_wrong_index])
                wrong_gid = int(candidate_ids[best_wrong_index])
                margin = true_sim - wrong_sim
                row = {
                    "video": video_name, "gt_id": gid, "post_frame": frame_id,
                    "candidate_count": len(candidate_ids), "true_similarity": true_sim,
                    "true_rank": true_rank, "best_wrong_gid": wrong_gid,
                    "best_wrong_similarity": wrong_sim, "margin": margin,
                }
                all_rows.append(row)
                print(row)

            event_rows = [r for r in all_rows if r["video"] == video_name and r["gt_id"] == gid]
            summary.append({
                "video": video_name, "gt_id": gid,
                "all_rank1": all(r["true_rank"] == 1 for r in event_rows),
                "min_true_similarity": min(r["true_similarity"] for r in event_rows),
                "min_margin": min(r["margin"] for r in event_rows),
            })

    with (OUT_DIR / "frame_results.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    (OUT_DIR / "summary.json").write_text(
        json.dumps({"model": str(MODEL), "device": device, "events": summary}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved results to {OUT_DIR}")


if __name__ == "__main__":
    main()
