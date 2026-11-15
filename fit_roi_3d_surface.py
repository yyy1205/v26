"""Fit one unified 3-D ROI ground surface from GCPs (no road grouping)."""
import argparse, csv, json, math
from pathlib import Path
import numpy as np

R=6378137.0
def read_gcps(p):
    rows=list(csv.DictReader(open(p,encoding='utf-8-sig')))
    out=[]
    for r in rows:
        out.append(dict(id=int(r[''] or r.get('id')), x=float(r['pixel_x']), y=float(r['pixel_y']), lon=float(r['longitude']), lat=float(r['latitude']), z=float(r['altitude'])))
    return out
def enu(g, g0):
    la0=math.radians(g0['lat']);
    return np.array([R*math.cos(la0)*math.radians(g['lon']-g0['lon']), R*math.radians(g['lat']-g0['lat']), g['z']])
def feat(x,y):
    return np.c_[np.ones(len(x)),x,y,x*x,x*y,y*y]
def fit(A,b): return np.linalg.lstsq(A,b,rcond=None)[0]
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--gcps',required=True); ap.add_argument('--output-dir',required=True); ap.add_argument('--fit-ids',default='1,3,5,9,10,12,13,15,17'); a=ap.parse_args(); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    allg=read_gcps(a.gcps); ids=[int(x) for x in a.fit_ids.split(',')]; g=[q for q in allg if q['id'] in ids]; val=[q for q in allg if q['id'] not in ids]; g0=allg[0]
    px=np.array([q['x'] for q in g]); py=np.array([q['y'] for q in g]); W=np.array([enu(q,g0) for q in g])
    A=feat(px,py); C=fit(A,W)
    pred=A@C; err=np.linalg.norm(pred-W,axis=1)
    V=np.array([enu(q,g0) for q in val]); vp=feat(np.array([q['x'] for q in val]),np.array([q['y'] for q in val]))@C; ve=np.linalg.norm(vp-V,axis=1)
    report={'method':'global quadratic pixel -> local ENU (E,N,Z)','fit_ids':ids,'validation_ids':[q['id'] for q in val],'train_mean_m':float(err.mean()),'train_rmse_m':float(np.sqrt(np.mean(err**2))),'train_max_m':float(err.max()),'validation_mean_m':float(ve.mean()),'validation_rmse_m':float(np.sqrt(np.mean(ve**2))),'validation_max_m':float(ve.max()),'validation_per_point':[{'id':q['id'],'error_m':float(e)} for q,e in zip(val,ve)]}
    # final control-point mapping output (predicted world coordinates)
    mapping=[{'id':q['id'],'x':q['x'],'y':q['y'],'longitude':q['lon'],'latitude':q['lat'],'altitude':q['z'],'fitted_east_m':float(pred[i,0]),'fitted_north_m':float(pred[i,1]),'fitted_altitude_m':float(pred[i,2]),'fit_error_m':float(err[i])} for i,q in enumerate(g)]
    json.dump(report,open(out/'fit_report.json','w',encoding='utf-8'),ensure_ascii=False,indent=2); json.dump(mapping,open(out/'mapping.json','w',encoding='utf-8'),ensure_ascii=False,indent=2)
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
