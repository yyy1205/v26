import cv2, json, os, math, time
import numpy as np

INPUT = r'E:\DJI\9\DJI_20240310135754_0193_D.MP4'
BASE = r'D:\yolo26-tracking\video_stabilization_dji0193'
MASK_PATH = os.path.join(BASE, 'static_mask.png')
OUTPUT = os.path.join(BASE, 'DJI_0193_paper_orb_flann_homography.mp4')
REPORT = os.path.join(BASE, 'paper_orb_flann_homography_report.json')
ANALYSIS_W = 1280
RATIO = 0.60
RANSAC_THRESH = 3.0
MIN_MATCHES = 4
MAX_FEATURES = 3000

cap=cv2.VideoCapture(INPUT)
if not cap.isOpened(): raise RuntimeError('无法打开输入视频')
W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); fps=cap.get(cv2.CAP_PROP_FPS) or 25; n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
ok, first=cap.read()
if not ok: raise RuntimeError('无法读取首帧')
analysis_h=round(H*ANALYSIS_W/W)
scale_x=ANALYSIS_W/W; scale_y=analysis_h/H
S=np.array([[scale_x,0,0],[0,scale_y,0],[0,0,1]],np.float64)
Sinv=np.linalg.inv(S)
mask=cv2.imread(MASK_PATH,cv2.IMREAD_GRAYSCALE)
if mask is None: raise RuntimeError('找不到静态掩膜: '+MASK_PATH)
mask=cv2.resize(mask,(ANALYSIS_W,analysis_h),interpolation=cv2.INTER_NEAREST)
mask=(mask>0).astype(np.uint8)*255
orb=cv2.ORB_create(nfeatures=MAX_FEATURES, scaleFactor=1.2, nlevels=8, fastThreshold=12)
ref_gray=cv2.resize(cv2.cvtColor(first,cv2.COLOR_BGR2GRAY),(ANALYSIS_W,analysis_h),interpolation=cv2.INTER_AREA)
k1,d1=orb.detectAndCompute(ref_gray,mask)
if d1 is None or len(k1)<MIN_MATCHES: raise RuntimeError(f'首帧特征不足: {0 if d1 is None else len(k1)}')
flann=cv2.FlannBasedMatcher(dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1), dict(checks=50))
fourcc=cv2.VideoWriter_fourcc(*'mp4v'); writer=cv2.VideoWriter(OUTPUT,fourcc,fps,(W,H))
if not writer.isOpened(): raise RuntimeError('无法创建输出视频')

stats={'input':INPUT,'output':OUTPUT,'width':W,'height':H,'fps':fps,'frames':n,'analysis_size':[ANALYSIS_W,analysis_h], 'mask_fraction':float((mask>0).mean()), 'method':'ORB + FLANN-LSH + Lowe ratio + RANSAC findHomography + warpPerspective', 'ratio':RATIO,'ransac_threshold_analysis_px':RANSAC_THRESH,'min_matches':MIN_MATCHES,'reference_keypoints':len(k1),'valid':0,'identity':0,'matches':[],'inliers':[],'reprojection_rmse':[],'det_h':[]}

def apply_h(frame,Hs):
    Hf=Sinv@Hs@S
    return cv2.warpPerspective(frame,Hf,(W,H),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)

writer.write(first)
prev_t=time.time()
for idx in range(1,n):
    ok,frame=cap.read()
    if not ok: break
    gray=cv2.resize(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),(ANALYSIS_W,analysis_h),interpolation=cv2.INTER_AREA)
    k2,d2=orb.detectAndCompute(gray,mask)
    Hs=np.eye(3,dtype=np.float64); mcount=0; icount=0; rmse=None
    if d2 is not None and len(k2)>=MIN_MATCHES:
        try:
            pairs=flann.knnMatch(d1,d2,k=2)
            good=[m for m,nm in pairs if m.distance < RATIO*nm.distance]
            mcount=len(good)
            if mcount>=MIN_MATCHES:
                # current -> reference
                src=np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
                dst=np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
                Hcand,inlier=cv2.findHomography(src,dst,cv2.RANSAC,RANSAC_THRESH)
                if Hcand is not None and inlier is not None and int(inlier.sum())>=MIN_MATCHES:
                    Hs=Hcand.astype(np.float64); icount=int(inlier.sum())
                    pred=cv2.perspectiveTransform(src,Hs)
                    err=np.linalg.norm(pred-dst,axis=2).ravel()[inlier.ravel().astype(bool)]
                    rmse=float(np.sqrt(np.mean(err*err))) if len(err) else None
        except cv2.error:
            pass
    if icount<MIN_MATCHES:
        Hs=np.eye(3,dtype=np.float64)
        stats['identity']+=1
    else:
        stats['valid']+=1
    out=apply_h(frame,Hs)
    writer.write(out)
    stats['matches'].append(mcount); stats['inliers'].append(icount); stats['reprojection_rmse'].append(rmse); stats['det_h'].append(float(np.linalg.det(Hs)))
    if idx%500==0:
        print(f'{idx}/{n} valid={stats["valid"]} last_matches={mcount} inliers={icount} elapsed={time.time()-prev_t:.1f}s',flush=True); prev_t=time.time()
cap.release(); writer.release()
for key in ['matches','inliers','reprojection_rmse','det_h']:
    vals=[v for v in stats[key] if v is not None]
    if vals: stats[key+'_summary']={'mean':float(np.mean(vals)),'p05':float(np.percentile(vals,5)),'p50':float(np.percentile(vals,50)),'p95':float(np.percentile(vals,95)),'min':float(np.min(vals)),'max':float(np.max(vals))}
with open(REPORT,'w',encoding='utf-8') as f: json.dump(stats,f,ensure_ascii=False,indent=2)
print(json.dumps({k:stats[k] for k in ['output','frames','reference_keypoints','valid','identity','matches_summary','inliers_summary','reprojection_rmse_summary'] if k in stats},ensure_ascii=False,indent=2))
