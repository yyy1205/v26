# -*- coding: utf-8 -*-
"""Fit the pixel/GPS mapping using only GCPs inside the tracking ROI."""
import argparse, json
from pathlib import Path
import cv2
import numpy as np
from coordinate_mapper import read_gcps, video_metadata, validate_pixels, CoordinateMapper, draw_gcp_overlay

def roi_gcp_ids(gcps, roi_json):
    data=json.loads(Path(roi_json).read_text(encoding='utf-8'))
    include=[np.asarray(x['pts'],np.float32) for x in data['roi_polys_modes'] if x['mode']==1]
    exclude=[np.asarray(x['pts'],np.float32) for x in data['roi_polys_modes'] if x['mode']==2]
    selected=[]
    for r in gcps:
        p=(r['pixel_x'],r['pixel_y'])
        inside=any(cv2.pointPolygonTest(poly,p,False)>=0 for poly in include)
        blocked=any(cv2.pointPolygonTest(poly,p,False)>=0 for poly in exclude)
        if inside and not blocked: selected.append(r)
    return selected

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--gcps',required=True); ap.add_argument('--roi',required=True); ap.add_argument('--video',required=True); ap.add_argument('--output-dir',required=True); ap.add_argument('--exclude-ids',default='',help='Comma-separated GCP ids excluded from fitting'); ap.add_argument('--include-ids',default='',help='Comma-separated GCP ids used for fitting'); a=ap.parse_args()
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True); all_gcps=read_gcps(a.gcps); W,H,fps=video_metadata(a.video);validate_pixels(all_gcps,W,H); selected=roi_gcp_ids(all_gcps,a.roi)
    # All measured GCPs define the transform. ROI is only the valid application
    # domain; points outside it are retained as global geometric constraints.
    excluded={int(x) for x in a.exclude_ids.split(',') if x.strip()}; included={int(x) for x in a.include_ids.split(',') if x.strip()}; fit_gcps=[r for r in all_gcps if (not included or r['id'] in included) and r['id'] not in excluded]
    if len(fit_gcps)<4: raise ValueError('At least 4 GCPs are required after exclusion')
    mapper=CoordinateMapper(fit_gcps,W,H); draw_gcp_overlay(a.video,all_gcps,out/'roi_gcp_overlay_first_frame.jpg')
    minimal=[{'id':r['id'],'x':r['pixel_x'],'y':r['pixel_y'],'longitude':r['longitude'],'latitude':r['latitude']} for r in fit_gcps]
    (out/'roi_pixel_gps_mapping.json').write_text(json.dumps(minimal,ensure_ascii=False,indent=2),encoding='utf-8')
    report={'video_width':W,'video_height':H,'video_fps':fps,'fit_gcp_ids':[r['id'] for r in fit_gcps], 'excluded_fit_gcp_ids':sorted({r['id'] for r in all_gcps if r not in fit_gcps}),
            'roi_valid_gcp_ids':[r['id'] for r in selected],'roi_only_application_domain':True,
            'fit_method':'RANSAC homography (comparison: affine)','fit':mapper.report()}
    (out/'roi_fit_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
