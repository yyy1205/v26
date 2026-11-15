"""Two-pass full-video ROI stabilizer with FFmpeg rawvideo output."""
import argparse,json,subprocess
from pathlib import Path
import cv2,numpy as np
from .stabilizer import StabilizationConfig,load_roi_mask,estimate_motion,_smooth
from .nadir_conservative import estimate_pair
from .sift_flann import estimate as estimate_sift
from .template_stabilizer import conservative_corrections,center_conjugate

def run(inp,out,roi_path=None,log_path=None,ffmpeg=r'C:\Tools\ffmpeg\ffmpeg-9.0.1-essentials_build\bin\ffmpeg.exe',window=30,checkpoint_path=None,reference_interval=60,drift_blend=0.08,max_frames=None,mode='orb_baseline',start_frame=0):
    cap=cv2.VideoCapture(str(inp)); fps=cap.get(cv2.CAP_PROP_FPS); n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.set(cv2.CAP_PROP_POS_FRAMES,start_frame); ok,first=cap.read()
    if not ok: raise RuntimeError('empty input')
    h,w=first.shape[:2]; cfg=StabilizationConfig(smooth_window=window); ew=cfg.estimate_width; eh=round(h*ew/w); roi=load_roi_mask(roi_path or str(Path(inp).with_name(Path(inp).stem+'_roi.json')),first.shape); mask=cv2.resize(roi,(ew,eh),interpolation=cv2.INTER_NEAREST) if roi is not None else None
    orb=cv2.ORB_create(cfg.nfeatures,cfg.scale_factor,cfg.nlevels,fastThreshold=cfg.fast_threshold); prev=cv2.resize(cv2.cvtColor(first,cv2.COLOR_BGR2GRAY),(ew,eh)); traj=[[0.,0.,0.,1.]]; logs=[{'frame_id':0,'valid_estimation':True,'tx':0.,'ty':0.,'rotation':0.,'scale':1.}]
    while True:
        if max_frames and len(logs)>=max_frames: break
        ok,f=cap.read()
        if not ok: break
        cur=cv2.resize(cv2.cvtColor(f,cv2.COLOR_BGR2GRAY),(ew,eh))
        if mode=='nadir_conservative_drift_correction':
            M,response=estimate_pair(prev,cur,mask,'translation'); info={'keypoint_count':0,'match_count':0,'inlier_count':0,'inlier_ratio':response,'valid':response>0.1}
        elif mode=='sift_flann':
            M,si=estimate_sift(prev,cur); info={'keypoint_count':si['keypoints'],'match_count':si['matches'],'inlier_count':si['inliers'],'inlier_ratio':si['ratio'],'valid':si['valid']}
        else: M,info=estimate_motion(prev,cur,cfg,orb,mask=mask,previous=np.eye(2,3,dtype=np.float32))
        a,b=M[0,0],M[1,0]; r=np.degrees(np.arctan2(b,a)); s=np.hypot(a,b); last=np.array(traj[-1]); t=np.array([last[0]+M[0,2],last[1]+M[1,2],last[2]+r,last[3]*s]);
        # Fixed-camera drift constraint: gently pull reliable trajectory toward its local reference.
        if reference_interval and len(logs)%reference_interval==0 and info['valid'] and info['inlier_ratio']>=0.5: t[:3] *= (1.0-drift_blend)
        t=t.tolist(); traj.append(t); logs.append({'frame_id':len(logs),'timestamp':len(logs)/fps,'keypoint_count':info['keypoint_count'],'raw_match_count':info['match_count'],'ransac_inlier_count':info['inlier_count'],'inlier_ratio':info['inlier_ratio'],'incremental_tx':float(M[0,2]),'incremental_ty':float(M[1,2]),'incremental_rotation_deg':float(r),'cumulative_tx':t[0],'cumulative_ty':t[1],'cumulative_rotation_deg':t[2],'valid_estimation':info['valid']}); prev=cur
        if len(logs)%10==0: print(f'analysis frame {len(logs)}/{n}',flush=True)
        if checkpoint_path and len(logs)%100==0: Path(checkpoint_path).write_text(json.dumps({'frame_id':len(logs)-1,'trajectory':traj[-1],'log_count':len(logs)}),encoding='utf-8')
    cap.release(); arr=np.asarray(traj); corrections=conservative_corrections(arr,window=window)
    cmd=[ffmpeg,'-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'{w}x{h}','-r',str(fps),'-i','pipe:0','-i',str(inp),'-map','0:v:0','-map','1:a?','-c:v','libx264','-preset','medium','-crf','18','-pix_fmt','yuv420p','-c:a','copy','-shortest',str(out)]
    err_file=Path(str(out)+'.ffmpeg.stderr.log').open('wb')
    p=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=err_file)
    cap=cv2.VideoCapture(str(inp)); cap.set(cv2.CAP_PROP_POS_FRAMES,start_frame); i=0
    try:
        while True:
            if max_frames and i>=max_frames: break
            ok,f=cap.read()
            if not ok: break
            S=np.diag([w/ew,h/eh,1.0]); C=S@corrections[i]@np.linalg.inv(S); C=center_conjugate(C,w,h); p.stdin.write(cv2.warpAffine(f,C[:2].astype(np.float32),(w,h),borderMode=cv2.BORDER_REFLECT).tobytes()); i+=1
        p.stdin.close(); rc=p.wait()
    finally:
        cap.release(); err_file.close()
    if rc: raise RuntimeError(f'ffmpeg failed {rc}')
    if log_path: Path(log_path).write_text(json.dumps({'frames':logs,'width':w,'height':h,'fps':fps},ensure_ascii=False),encoding='utf-8')
    return {'frames':i,'width':w,'height':h,'fps':fps}

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('input'); p.add_argument('output'); p.add_argument('--roi'); p.add_argument('--log'); p.add_argument('--checkpoint',default='video_stabilization/test_outputs/checkpoint.json'); p.add_argument('--window',type=int,default=30); p.add_argument('--max-frames',type=int); p.add_argument('--start-frame',type=int,default=0); p.add_argument('--mode',choices=['orb_baseline','nadir_conservative_drift_correction','sift_flann'],default='orb_baseline'); a=p.parse_args(); print(run(a.input,a.output,a.roi,a.log,window=a.window,checkpoint_path=a.checkpoint,max_frames=a.max_frames,mode=a.mode,start_frame=a.start_frame))
