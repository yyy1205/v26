import csv
from pathlib import Path

import numpy as np
import unittest

from coordinate_mapper import CoordinateMapper, read_gcps, validate_pixels


GCP = Path(r"F:\ybw\gps\gcps.csv")


class CoordinateMapperTests(unittest.TestCase):
 def test_csv_read_and_ids(self):
    rows = read_gcps(GCP)
    assert len(rows) == 17
    assert [r["id"] for r in rows] == list(range(1, 18))
    assert all(all(np.isfinite(r[k]) for k in ("pixel_x", "pixel_y", "longitude", "latitude", "altitude")) for r in rows)


 def test_count_and_duplicate_validation(self):
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "bad.csv"
        p.write_text("id,pixel_x,pixel_y,longitude,latitude,altitude\n1,1,1,1,1,1\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Expected 17"):
            read_gcps(p)


 def test_pixel_range(self):
    rows = read_gcps(GCP)
    validate_pixels(rows, 5120, 2700)
    rows[0]["pixel_x"] = 5120
    with self.assertRaisesRegex(ValueError, "outside"):
        validate_pixels(rows, 5120, 2700)


 def test_pixel_gps_and_inverse(self):
    rows = read_gcps(GCP)
    mapper = CoordinateMapper(rows, 5120, 2700)
    r = rows[3]
    gps = mapper.pixel_to_gps(r["pixel_x"], r["pixel_y"])
    assert abs(gps["longitude"] - r["longitude"]) < 1e-4
    assert abs(gps["latitude"] - r["latitude"]) < 1e-4
    x, y = mapper.gps_to_pixel(r["longitude"], r["latitude"])
    # GCP fit residual is sub-metre; at this oblique view that can correspond
    # to several tens of pixels, so use a conservative inverse consistency bound.
    assert np.hypot(x - r["pixel_x"], y - r["pixel_y"]) < 100


 def test_reprojection_errors_and_report(self):
    mapper = CoordinateMapper(read_gcps(GCP), 5120, 2700)
    assert len(mapper.homography_errors) == 17
    assert np.all(np.isfinite(mapper.homography_errors))
    report = mapper.report()
    assert "rmse_m" in report["homography"]
    assert report["homography"]["rmse_m"] >= 0
    assert report["affine"]["rmse_m"] >= 0


if __name__ == "__main__":
    unittest.main()
