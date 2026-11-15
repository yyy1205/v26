# -*- coding: utf-8 -*-
"""Infer road-level GCP groups from tracks and compare per-road coordinate fits."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator
from coordinate_mapper import read_gcps, lonlat_to_enu


def affine_fit(p,w):
    q0,qscale=p.mean(0),p.std(0); A=np.c_[np.ones(len(p)),(p-q0)/qscale]
    B=np.linalg.lstsq(A,w,rcond=None)[0]
    return lambda z: np.c_[np.ones(len(z)),(np.asarray(z)-q0)/qscale]@B

def homography_fit(p,w):
    H,_=cv2.findHomography(np.asarray(p,float),np.asarray(w,float),0)
    return lambda z: cv2.perspectiveTransform(np.asarray(z,float).reshape(-1,1,2),H).reshape(-1,2)

def poly2_fit(p,w):
    c,s=p.mean(0),p.std(0)
    def D(z):
        x,y=((np.asarray(z)-c)/s).T; return np.c_[np.ones(len(x)),x,y,x*y,x*x,y*y]
    B=np.linalg.lstsq(D(p),w,rcond=None)[0]
    return lambda z:D(z)@B

def tps_fit(p,w):
    c,s=p.mean(0),p.std(0); m=RBFInterpolator((p-c)/s,w,kernel='thin_plate_spline')
    return lambda z:m((np.asarray(z)-c)/s)

FITTERS={'affine':affine_fit,'homography':homography_fit,'polynomial_2':poly2_fit,'tps':tps_fit}

def evaluate(fitter,p,w):
    model=fitter(p,w); train=np.linalg.norm(model(p)-w,axis=1); loo=[]
    for i in range(len(p)):
        model_i=fitter(np.delete(p,i,0),np.delete(w,i,0)); loo.append(np.linalg.norm(model_i(p[i:i+1])[0]-w[i]))
    loo=np.asarray(loo)
    return {'train_rmse_m':float(np.sqrt(np.mean(train**2))),'train_max_m':float(train.max()),
            'loo_rmse_m':float(np.sqrt(np.mean(loo**2))),'loo_mean_m':float(loo.mean()),'loo_max_m':float(loo.max()),
            'loo_errors_m':loo.tolist()}

def components(adj):
    seen=set(); out=[]
    for start in range(len(adj)):
        if start in seen: continue
        stack=[start]; seen.add(start); comp=[]
        while stack:
            i=stack.pop(); comp.append(i)
            for j in np.flatnonzero(adj[i]):
                if int(j) not in seen: seen.add(int(j)); stack.append(int(j))
        out.append(sorted(comp))
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--tracks',required=True); ap.add_argument('--gcps',required=True)
    ap.add_argument('--video',required=True); ap.add_argument('--output-dir',required=True); ap.add_argument('--radius',type=float,default=220)
    ap.add_argument('--min-common-tracks',type=int,default=3)
    ap.add_argument('--num-roads',type=int,default=0,help='Force spectral clustering to this road count; 0 uses graph components')
    a=ap.parse_args(); out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    gcps=read_gcps(a.gcps); G=np.array([[r['pixel_x'],r['pixel_y']] for r in gcps]);
    d=pd.read_csv(a.tracks,header=None,usecols=[0,1,2,3],names=['frame','track_id','x','y'])
    count=np.zeros((17,17),int); trajectories={}
    for tid,t in d.groupby('track_id'):
        if len(t)<20: continue
        q=t[['x','y']].to_numpy()[::5]; md=np.sqrt(((q[:,None,:]-G[None,:,:])**2).sum(2)).min(0); hit=np.flatnonzero(md<a.radius)
        if len(hit): trajectories[int(tid)]=(q,hit.tolist())
        for i in hit:
            for j in hit: count[i,j]+=1
    adj=(count>=a.min_common_tracks); np.fill_diagonal(adj,True)
    if a.num_roads:
        from sklearn.cluster import SpectralClustering
        diag=np.maximum(np.diag(count).astype(float),1.0)
        similarity=count/np.sqrt(diag[:,None]*diag[None,:])
        np.fill_diagonal(similarity,1.0)
        labels=SpectralClustering(n_clusters=a.num_roads,affinity='precomputed',assign_labels='cluster_qr',
                                  random_state=0).fit_predict(similarity)
        groups=[sorted(np.flatnonzero(labels==k).tolist()) for k in range(a.num_roads)]
        groups=sorted(groups,key=lambda z:min(z))
    else:
        groups=[x for x in components(adj) if len(x)>=4]
    lon0,lat0=gcps[0]['longitude'],gcps[0]['latitude']; e,n=lonlat_to_enu([r['longitude'] for r in gcps],[r['latitude'] for r in gcps],lon0,lat0); W=np.c_[e,n]
    group_rows=[]
    for k,idx in enumerate(groups,1):
        fits={}
        if len(idx)>=4:
            eligible={'affine':len(idx)>=4,'homography':len(idx)>=5,
                      'polynomial_2':len(idx)>=7,'tps':len(idx)>=4}
            for name,fn in FITTERS.items():
                if eligible[name]:
                    try: fits[name]=evaluate(fn,G[idx],W[idx])
                    except Exception as exc: fits[name]={'error':str(exc)}
            fits={k:v for k,v in fits.items() if 'loo_rmse_m' in v}
        best=min(fits,key=lambda x:fits[x]['loo_rmse_m']) if fits else None
        group_rows.append({'road_id':f'road_{k}','gcp_ids':[i+1 for i in idx],
                           'supporting_track_count':sum(1 for _,h in trajectories.values() if len(set(h)&set(idx))>=2),
                           'best_method_by_loo':best,'fit_status':'ok' if fits else 'insufficient_gcps','methods':fits})
    result={'method':'track co-visitation graph','radius_px':a.radius,'min_common_tracks':a.min_common_tracks,
            'groups':group_rows,'common_track_matrix':count.tolist()}
    (out/'road_grouping.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')

    cap=cv2.VideoCapture(a.video);ok,frame=cap.read();cap.release()
    if not ok: raise RuntimeError('Cannot read first video frame')
    colors=[(0,70,255),(0,200,80),(255,100,0),(220,0,180)]
    overlay=frame.copy()
    for k,idx in enumerate(groups):
        color=colors[k%len(colors)]; selected=0
        for q,h in trajectories.values():
            if len(set(h)&set(idx))>=2 and selected<100:
                cv2.polylines(overlay,[np.int32(q.reshape(-1,1,2))],False,color,1,cv2.LINE_AA); selected+=1
        for i in idx:
            p=tuple(np.int32(np.round(G[i])));cv2.circle(overlay,p,15,(255,255,255),-1,cv2.LINE_AA);cv2.circle(overlay,p,11,color,-1,cv2.LINE_AA)
            cv2.putText(overlay,str(i+1),(p[0]+15,p[1]-12),cv2.FONT_HERSHEY_SIMPLEX,1,color,3,cv2.LINE_AA)
    vis=cv2.addWeighted(frame,.55,overlay,.45,0);cv2.imwrite(str(out/'road_grouping_overlay.jpg'),vis)
    print(json.dumps([{'road_id':r['road_id'],'gcp_ids':r['gcp_ids'],'supporting_tracks':r['supporting_track_count'],
                       'best_method':r['best_method_by_loo'],
                       'best_loo_rmse_m':r['methods'][r['best_method_by_loo']]['loo_rmse_m'] if r['best_method_by_loo'] else None}
                      for r in group_rows],ensure_ascii=False,indent=2))

if __name__=='__main__':main()
