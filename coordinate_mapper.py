# -*- coding: utf-8 -*-
"""Pixel/GPS mapping from a CSV of ground-control points.

The public :class:`CoordinateMapper` uses all GCPs, fits a RANSAC homography
and an affine model, and reports residuals in metres.  With no pyproj
installation it uses a local equirectangular ENU approximation, which is
accurate to sub-metre scale over this small footprint.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

REQUIRED = ("id", "pixel_x", "pixel_y", "longitude", "latitude", "altitude")
EARTH_RADIUS = 6378137.0


def read_gcps(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        # The supplied file has an empty first header cell; id is its row value.
        if reader.fieldnames and reader.fieldnames[0] == "":
            reader.fieldnames[0] = "id"
        missing = [x for x in REQUIRED if x not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"GCP CSV missing fields: {missing}")
        for line, row in enumerate(reader, 2):
            if any(row.get(k, "").strip() == "" for k in REQUIRED):
                raise ValueError(f"Missing GCP value at CSV line {line}")
            try:
                rows.append({"id": int(row["id"]), "pixel_x": float(row["pixel_x"]),
                             "pixel_y": float(row["pixel_y"]), "longitude": float(row["longitude"]),
                             "latitude": float(row["latitude"]), "altitude": float(row["altitude"])})
            except ValueError as e:
                raise ValueError(f"Invalid numeric GCP value at CSV line {line}") from e
    if len(rows) != 17:
        raise ValueError(f"Expected 17 GCPs, found {len(rows)}")
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("GCP ids are not unique")
    if set(ids) != set(range(1, 18)):
        raise ValueError("GCP ids must be exactly 1..17")
    return rows


def video_metadata(video: str | Path) -> tuple[int, int, float]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video}")
    result = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
              float(cap.get(cv2.CAP_PROP_FPS)))
    cap.release()
    return result


def validate_pixels(gcps: Iterable[dict], width: int, height: int) -> None:
    bad = [r["id"] for r in gcps if not (0 <= r["pixel_x"] < width and 0 <= r["pixel_y"] < height)]
    if bad:
        raise ValueError(f"Pixel coordinates outside {width}x{height}: ids {bad}")


def lonlat_to_enu(lon, lat, lon0, lat0):
    lat0r = np.deg2rad(lat0)
    return (np.deg2rad(np.asarray(lon) - lon0) * EARTH_RADIUS * np.cos(lat0r),
            np.deg2rad(np.asarray(lat) - lat0) * EARTH_RADIUS)


def enu_to_lonlat(e, n, lon0, lat0):
    lat0r = np.deg2rad(lat0)
    return (lon0 + np.rad2deg(np.asarray(e) / (EARTH_RADIUS * np.cos(lat0r))),
            lat0 + np.rad2deg(np.asarray(n) / EARTH_RADIUS))


class CoordinateMapper:
    def __init__(self, gcps: list[dict], width: int | None = None, height: int | None = None,
                 ransac_threshold_m: float = 5.0):
        if width is not None and height is not None:
            validate_pixels(gcps, width, height)
        self.gcps = gcps
        self.lon0, self.lat0 = gcps[0]["longitude"], gcps[0]["latitude"]
        lon = np.array([r["longitude"] for r in gcps]); lat = np.array([r["latitude"] for r in gcps])
        east, north = lonlat_to_enu(lon, lat, self.lon0, self.lat0)
        self.world = np.column_stack([east, north]).astype(np.float64)
        self.pixels = np.array([[r["pixel_x"], r["pixel_y"]] for r in gcps], dtype=np.float64)
        self.H, self.inliers = cv2.findHomography(self.pixels, self.world, cv2.RANSAC, ransac_threshold_m)
        if self.H is None:
            raise RuntimeError("Homography fitting failed")
        self.affine, self.affine_inliers = cv2.estimateAffine2D(self.pixels, self.world, method=cv2.RANSAC,
                                                                  ransacReprojThreshold=ransac_threshold_m)
        if self.affine is None:
            raise RuntimeError("Affine fitting failed")
        self.homography_errors = self._errors(self.H, homography=True)
        self.affine_errors = self._errors(self.affine, homography=False)

    def _errors(self, matrix, homography):
        pred = self._transform(self.pixels, matrix, homography)
        return np.linalg.norm(pred - self.world, axis=1)

    @staticmethod
    def _transform(points, matrix, homography):
        points = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
        if homography:
            return cv2.perspectiveTransform(points.astype(np.float64), matrix).reshape(-1, 2)
        return cv2.transform(points.astype(np.float64), matrix).reshape(-1, 2)

    def pixel_to_enu(self, x: float, y: float) -> tuple[float, float]:
        return tuple(self._transform([[x, y]], self.H, True)[0])

    def enu_to_pixel(self, east: float, north: float) -> tuple[float, float]:
        return tuple(self._transform([[east, north]], np.linalg.inv(self.H), True)[0])

    def pixel_to_gps(self, x: float, y: float) -> dict[str, float]:
        e, n = self.pixel_to_enu(x, y)
        lon, lat = enu_to_lonlat(e, n, self.lon0, self.lat0)
        return {"longitude": float(lon), "latitude": float(lat), "east": float(e), "north": float(n)}

    def gps_to_pixel(self, longitude: float, latitude: float) -> tuple[float, float]:
        e, n = lonlat_to_enu(longitude, latitude, self.lon0, self.lat0)
        return self.enu_to_pixel(float(e), float(n))

    def report(self) -> dict:
        def stats(err):
            return {"mean_m": float(np.mean(err)), "rmse_m": float(np.sqrt(np.mean(err ** 2))),
                    "max_m": float(np.max(err)), "outlier_ids": [self.gcps[i]["id"] for i, v in enumerate(err) if v > 5.0]}
        return {"homography": stats(self.homography_errors), "affine": stats(self.affine_errors),
                "homography_inliers": [self.gcps[i]["id"] for i, v in enumerate(self.inliers.ravel()) if v],
                "affine_inliers": [self.gcps[i]["id"] for i, v in enumerate(self.affine_inliers.ravel()) if v]}


def draw_gcp_overlay(video: str | Path, gcps: list[dict], output: str | Path) -> None:
    cap = cv2.VideoCapture(str(video)); ok, frame = cap.read(); cap.release()
    if not ok:
        raise RuntimeError("Cannot read video first frame")
    for r in gcps:
        p = (int(round(r["pixel_x"])), int(round(r["pixel_y"])))
        cv2.circle(frame, p, 12, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, p, 15, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, str(r["id"]), (p[0] + 16, p[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 0, 255), 3, cv2.LINE_AA)
    cv2.imwrite(str(output), frame)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gcps", required=True); ap.add_argument("--video", required=True); ap.add_argument("--output-dir", required=True)
    args = ap.parse_args(); out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    gcps = read_gcps(args.gcps); width, height, fps = video_metadata(args.video); validate_pixels(gcps, width, height)
    mapper = CoordinateMapper(gcps, width, height)
    draw_gcp_overlay(args.video, gcps, out / "gcp_overlay_first_frame.jpg")
    final = [{"id": r["id"], "x": r["pixel_x"], "y": r["pixel_y"], "longitude": r["longitude"], "latitude": r["latitude"]} for r in gcps]
    (out / "pixel_gps_mapping.json").write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {"video_width": width, "video_height": height, "video_fps": fps, "gcp_count": len(gcps),
              "coordinate_system": "local ENU (first GCP origin)", "fit_method": "RANSAC homography (comparison: affine)",
              "fit": mapper.report()}
    (out / "fit_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
