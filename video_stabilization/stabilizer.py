from __future__ import annotations

import argparse, csv, json
from dataclasses import dataclass, asdict
from pathlib import Path
import cv2
import numpy as np
import subprocess

def encode_ffmpeg(frames, output_path, width, height, fps, ffmpeg=r'C:\Tools\ffmpeg\ffmpeg-9.0.1-essentials_build\bin\ffmpeg.exe', stderr_path=None, codec='libx264'):
    """Encode BGR uint8 frames through FFmpeg rawvideo stdin; raises on failure."""
    cmd=[ffmpeg,'-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'{width}x{height}','-r',str(fps),'-i','pipe:0','-an','-c:v',codec,'-preset','medium','-crf','18','-pix_fmt','yuv420p',str(output_path)]
    p=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        for frame in frames: p.stdin.write(np.ascontiguousarray(frame,dtype=np.uint8).tobytes())
        p.stdin.close()
    except Exception:
        p.kill(); raise
    err=p.stderr.read(); rc=p.wait()
    if stderr_path: Path(stderr_path).write_bytes(err)
    if rc: raise RuntimeError(f'FFmpeg failed ({rc}); see {stderr_path}')
    return rc

def load_roi_mask(roi_path, shape):
    """Load the project's ``roi_polys_modes`` JSON into an OpenCV mask."""
    if not roi_path or not Path(roi_path).exists():
        return None
    data = json.loads(Path(roi_path).read_text(encoding='utf-8'))
    items = data.get('roi_polys_modes', data if isinstance(data, list) else [])
    mask = np.zeros(shape[:2], np.uint8)
    for item in items:
        pts = item.get('pts', [])
        if len(pts) >= 3 and item.get('mode', 1) == 1:
            cv2.fillPoly(mask, [np.asarray(pts, np.int32)], 255)
    return mask

@dataclass
class StabilizationConfig:
    estimate_width: int = 1600
    model: str = "similarity"
    nfeatures: int = 3000
    scale_factor: float = 1.2
    nlevels: int = 8
    fast_threshold: int = 15
    ratio: float = 0.7
    min_matches: int = 8
    ransac_threshold: float = 3.0
    smooth_window: int = 30
    smooth_method: str = "gaussian"
    crop: float = 0.02
    camera_lock: float = 0.0
    exclude_border: float = 0.04
    max_rotation_deg: float = 3.0
    max_translation_fraction: float = 0.05

def _smooth(x, window, method):
    x=np.asarray(x,float); n=len(x)
    if n < 2: return x.copy()
    w=max(1,int(window)); w=min(w,n if n%2 else n-1)
    if w<3:return x.copy()
    if method == "savgol":
        try:
            from scipy.signal import savgol_filter
            return savgol_filter(x,w,min(2,w-1),mode="interp")
        except Exception: pass
    sigma=max(1.,w/6); k=np.arange(w)-w//2; g=np.exp(-(k*k)/(2*sigma*sigma)); g/=g.sum()
    return np.convolve(np.pad(x,(w//2,w//2),mode="edge"),g,mode="valid")

def estimate_motion(prev, curr, cfg, orb=None, mask=None, previous=None):
    if orb is None: orb=cv2.ORB_create(cfg.nfeatures,cfg.scale_factor,cfg.nlevels,fastThreshold=cfg.fast_threshold)
    k1,d1=orb.detectAndCompute(prev,mask); k2,d2=orb.detectAndCompute(curr,mask)
    info=dict(keypoint_count=len(k2),match_count=0,inlier_count=0,inlier_ratio=0.,valid=False)
    if d1 is None or d2 is None: return np.eye(2,3,dtype=np.float32) if previous is None else previous, info
    knn=cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1,d2,k=2)
    good=[m for m,n in knn if m.distance < cfg.ratio*n.distance]; info['match_count']=len(good)
    if len(good)<cfg.min_matches:return (np.eye(2,3,dtype=np.float32) if previous is None else previous),info
    p=np.float32([k1[m.queryIdx].pt for m in good]); q=np.float32([k2[m.trainIdx].pt for m in good])
    fn=cv2.estimateAffinePartial2D if cfg.model=="similarity" else cv2.estimateAffine2D
    M,inl=fn(p,q,method=cv2.RANSAC,ransacReprojThreshold=cfg.ransac_threshold)
    if M is None or inl is None:return (np.eye(2,3,dtype=np.float32) if previous is None else previous),info
    info['inlier_count']=int(inl.sum()); info['inlier_ratio']=info['inlier_count']/len(good)
    a,b=M[0,0],M[1,0]; rot=np.degrees(np.arctan2(b,a)); tx,ty=M[0,2],M[1,2]
    h,w=prev.shape[:2]
    if abs(rot)>cfg.max_rotation_deg or max(abs(tx),abs(ty))>cfg.max_translation_fraction*max(w,h):
        return (np.eye(2,3,dtype=np.float32) if previous is None else previous),info
    info['valid']=True; return M.astype(np.float32),info

def stabilize_frames(frames, cfg=None, roi_mask=None):
    cfg=cfg or StabilizationConfig(); frames=list(frames)
    if not frames:return [],[]
    orb=cv2.ORB_create(cfg.nfeatures,cfg.scale_factor,cfg.nlevels,fastThreshold=cfg.fast_threshold); transforms=[np.eye(2,3,dtype=np.float32)]; logs=[]
    ew=cfg.estimate_width; eh=round(frames[0].shape[0]*ew/frames[0].shape[1]); mask=cv2.resize(roi_mask,(ew,eh),interpolation=cv2.INTER_NEAREST) if roi_mask is not None else None
    prev=cv2.resize(cv2.cvtColor(frames[0],cv2.COLOR_BGR2GRAY),(ew,eh))
    for frame in frames[1:]:
        cur=cv2.resize(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),(cfg.estimate_width,round(frame.shape[0]*cfg.estimate_width/frame.shape[1])))
        M,info=estimate_motion(prev,cur,cfg,orb,mask=mask,previous=np.eye(2,3,dtype=np.float32));
        transforms.append((np.vstack([transforms[-1],[0,0,1]]) @ np.vstack([M,[0,0,1]]))[:2].astype(np.float32));
        a,b=transforms[-1][0,0],transforms[-1][1,0]; info.update(frame_id=len(logs)+1,tx=float(transforms[-1][0,2]),ty=float(transforms[-1][1,2]),rotation=float(np.degrees(np.arctan2(b,a))),scale=float(np.hypot(a,b))); logs.append(info); prev=cur
    logs.insert(0,dict(frame_id=0,keypoint_count=0,match_count=0,inlier_count=0,inlier_ratio=0.,tx=0.,ty=0.,rotation=0.,scale=1.,valid=True))
    vals=np.array([[x['tx'],x['ty'],x['rotation'],x['scale']] for x in logs]); smooth=np.column_stack([_smooth(vals[:,i],cfg.smooth_window,cfg.smooth_method) for i in range(4)])
    out=[]; h,w=frames[0].shape[:2]
    sx=w/cfg.estimate_width; sy=h/round(h*cfg.estimate_width/w)
    for i,f in enumerate(frames):
        # Apply only the residual correction, not the smoothed camera trajectory.
        a=np.deg2rad(smooth[i,2]-vals[i,2]); s=smooth[i,3]/max(vals[i,3],1e-6)
        tx=(smooth[i,0]-vals[i,0])*sx; ty=(smooth[i,1]-vals[i,1])*sy
        corr=np.array([[s*np.cos(a),-s*np.sin(a),tx],[s*np.sin(a),s*np.cos(a),ty]],np.float32)
        out.append(cv2.warpAffine(f,corr,(w,h),borderMode=cv2.BORDER_REFLECT))
    return out,logs

def stabilize_video(input_path, output_path, log_path=None, cfg=None):
    cfg=cfg or StabilizationConfig(); cap=cv2.VideoCapture(str(input_path)); fps=cap.get(cv2.CAP_PROP_FPS); n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); first=cap.read()
    if not first[0]: raise ValueError("No frames in input video")
    prev=cv2.resize(cv2.cvtColor(first[1],cv2.COLOR_BGR2GRAY),(cfg.estimate_width,round(first[1].shape[0]*cfg.estimate_width/first[1].shape[1]))); orb=cv2.ORB_create(cfg.nfeatures,cfg.scale_factor,cfg.nlevels,fastThreshold=cfg.fast_threshold); transforms=[np.eye(2,3,dtype=np.float32)]; logs=[dict(frame_id=0,keypoint_count=0,match_count=0,inlier_count=0,inlier_ratio=0.,tx=0.,ty=0.,rotation=0.,scale=1.,valid=True)]
    while True:
        ok,f=cap.read()
        if not ok: break
        cur=cv2.resize(cv2.cvtColor(f,cv2.COLOR_BGR2GRAY),(cfg.estimate_width,round(f.shape[0]*cfg.estimate_width/f.shape[1]))); M,info=estimate_motion(prev,cur,cfg,orb,previous=np.eye(2,3,dtype=np.float32)); transforms.append((np.vstack([transforms[-1],[0,0,1]])@np.vstack([M,[0,0,1]]))[:2].astype(np.float32)); a,b=transforms[-1][0,0],transforms[-1][1,0]; info.update(frame_id=len(logs),tx=float(transforms[-1][0,2]),ty=float(transforms[-1][1,2]),rotation=float(np.degrees(np.arctan2(b,a))),scale=float(np.hypot(a,b))); logs.append(info); prev=cur
    cap.release()
    vals=np.array([[x['tx'],x['ty'],x['rotation'],x['scale']] for x in logs]); smooth=np.column_stack([_smooth(vals[:,i],cfg.smooth_window,cfg.smooth_method) for i in range(4)])
    cap=cv2.VideoCapture(str(input_path)); ok,f=cap.read(); h,w=f.shape[:2]
    codec='MJPG' if Path(output_path).suffix.lower()=='.avi' else 'mp4v'
    writer=cv2.VideoWriter(str(output_path),cv2.VideoWriter_fourcc(*codec),fps,(w,h))
    if not writer.isOpened() and codec!='MJPG':
        output_path=str(Path(output_path).with_suffix('.avi')); writer=cv2.VideoWriter(output_path,cv2.VideoWriter_fourcc(*'MJPG'),fps,(w,h))
    i=0
    while ok:
        a=np.deg2rad(smooth[i,2]); s=smooth[i,3]; corr=np.array([[s*np.cos(a),-s*np.sin(a),smooth[i,0]],[s*np.sin(a),s*np.cos(a),smooth[i,1]]],np.float32); writer.write(cv2.warpAffine(f,corr,(w,h),borderMode=cv2.BORDER_REFLECT)); i+=1; ok,f=cap.read()
    cap.release(); writer.release()
    writer.release()
    if log_path:
        with open(log_path,'w',newline='',encoding='utf-8') as fh:
            json.dump({'config':asdict(cfg),'frames':logs},fh,ensure_ascii=False,indent=2)
    return {'frames':len(logs),'fps':fps,'width':w,'height':h,'logs':logs}

def main():
    p=argparse.ArgumentParser(); p.add_argument('input'); p.add_argument('output'); p.add_argument('--log'); p.add_argument('--model',choices=['similarity','affine'],default='similarity'); p.add_argument('--smooth-window',type=int,default=30); p.add_argument('--smooth-method',choices=['gaussian','savgol'],default='gaussian'); p.add_argument('--estimate-width',type=int,default=1600); p.add_argument('--ratio',type=float,default=.7); p.add_argument('--camera-lock',type=float,default=0.)
    a=p.parse_args(); stabilize_video(a.input,a.output,a.log,StabilizationConfig(model=a.model,smooth_window=a.smooth_window,smooth_method=a.smooth_method,estimate_width=a.estimate_width,ratio=a.ratio,camera_lock=a.camera_lock))
if __name__=='__main__': main()
