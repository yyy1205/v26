# Pixel/GPS coordinate mapper

`coordinate_mapper.py` fits a robust 2-D pixel-to-world mapping from the 17
GCPs. It uses a local ENU plane centred at GCP 1 because `pyproj` is not
installed in the project environment. Over this small area the
equirectangular ENU approximation is suitable for the planar fit. OpenCV's
RANSAC homography is the final mapping; a RANSAC affine fit is reported for
comparison.

Run:

```powershell
python coordinate_mapper.py `
  --gcps F:\ybw\gps\gcps.csv `
  --video E:\DJI\sgybw\video2\DJI_20250302041036_0037_D.MP4 `
  --output-dir outputs\coordinate_mapper
```

Outputs are `pixel_gps_mapping.json` (the requested minimal mapping list),
`fit_report.json` (validation and error statistics), and
`gcp_overlay_first_frame.jpg` (red numbered GCP overlay). The module can also
be imported and used through `CoordinateMapper.pixel_to_gps(x, y)` and
`CoordinateMapper.gps_to_pixel(longitude, latitude)`.

Tests:

```powershell
pytest -q test_coordinate_mapper.py
```

Dependencies: Python, NumPy and OpenCV. `pyproj` is optional; if installed,
it can be added later for a UTM implementation, but is not required for the
current local ENU mapping.
