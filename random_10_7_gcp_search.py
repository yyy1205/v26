# -*- coding: utf-8 -*-
"""Random 10-GCP fit / 7-GCP validation search."""
import argparse, json
from pathlib import Path
import cv2, numpy as np
from coordinate_mapper import read_gcps, lonlat_to_enu

def fit_affine(p,w):
    c,s=p.mean(0),p.std(0); A=np.c_[np.ones(len(p)),(p-c)/s]; B=np.linalg.lstsq(A,w,rcond=None)[0]
    return lambda z: np.c_[np.ones(len(z)),(np.asarray(z)-c)/s]@B
def fit_h(p,w):
    H,_=cv2.findHomography(p,w,0)
    return lambda z: cv2.perspectiveTransform(np.asarray(z,float).reshape(-1,1,2),H).reshape(-1,2)
def evaluate(fitter,p,w,tr,va):
    f=fitter(p[tr],w[tr]); et=np.linalg.norm(f(p[tr])-w[tr],axis=1); ev=np.linalg.norm(f(p[va])-w[va],axis=1)
    return {'train_rmse_m':float(np.sqrt(np.mean(et**2))),'train_max_m':float(et.max()),'val_rmse_m':float(np.sqrt(np.mean(ev**2))), 'val_mean_m':float(ev.mean()),'val_max_m':float(ev.max()),'val_errors_m':ev.tolist()}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--gcps',required=True); ap.add_argument('--output',required=True); ap.add_argument('--iterations',type=int,default=10000); ap.add_argument('--seed',type=int,default=20260904); a=ap.parse_args()
    g=read_gcps(a.gcps); p=np.array([[r['pixel_x'],r['pixel_y']] for r in g]); e,n=lonlat_to_enu([r['longitude'] for r in g],[r['latitude'] for r in g],g[0]['longitude'],g[0]['latitude']); w=np.c_[e,n]; rng=np.random.default_rng(a.seed); seen=set(); best={}
    for _ in range(a.iterations):
        tr=tuple(sorted(rng.choice(17,10,replace=False))); 
        if tr in seen: continue
        seen.add(tr); va=tuple(i for i in range(17) if i not in tr)
        for name,fn in [('affine',fit_affine),('homography',fit_h)]:
            try: r=evaluate(fn,p,w,np.array(tr),np.array(va))
            except Exception: continue
            if name not in best or r['val_rmse_m']<best[name]['metrics']['val_rmse_m']: best[name]={'train_ids':[int(i)+1 for i in tr],'val_ids':[int(i)+1 for i in va],'metrics':r}
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps({'iterations':a.iterations,'unique_subsets':len(seen),'seed':a.seed,'best':best},ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps({'unique_subsets':len(seen),'best':best},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
