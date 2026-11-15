import cv2, numpy as np, json, os, time
INPUT=r'E:\DJI\9\DJI_20240310135754_0193_D.MP4'; BASE=r'D:\yolo26-tracking\video_stabilization_dji0193'; MASK_PATH=os.path.join(BASE,'static_mask.png'); OUT=os.path.join(BASE,'DJI_0193_opencv_videostab_gaussian.mp4'); REPORT=os.path.join(BASE,'opencv_videostab_gaussian_report.json')
AW=960; MAXC=300; RADIUS=50; SIGMA=0
cap=cv2.VideoCapture(INPUT); W=int(cap.get(3)); H=int(cap.get(4)); fps=cap.get(5) or 25; n=int(cap.get(7)); cap.release(); ah=round(H*AW/W)
mask=cv2.resize(cv2.imread(MASK_PATH,0),(AW,ah),interpolation=cv2.INTER_NEAREST)
mask=(mask>0).astype(np.uint8)*255

def gray(f): return cv2.resize(cv2.cvtColor(f,cv2.COLOR_BGR2GRAY),(AW,ah),interpolation=cv2.INTER_AREA)
def smooth(a):
  if len(a)<7:return a.copy()
  k=int(2*RADIUS+1); k=k if k%2 else k+1
  # GaussianMotionFilter-like temporal filter on each parameter
  return cv2.GaussianBlur(a.astype(np.float32),(1,k),0,borderType=cv2.BORDER_REPLICATE)
def params(M):
  A=M[:,:2]; scale=float(np.sqrt(A[0,0]**2+A[1,0]**2)); theta=float(np.arctan2(A[1,0],A[0,0])); return np.array([M[0,2],M[1,2],theta,scale],np.float64)
def makeM(p):
  tx,ty,th,sc=p; c=np.cos(th)*sc; s=np.sin(th)*sc; return np.array([[c,-s,tx],[s,c,ty]],np.float32)
# pass 1
cap=cv2.VideoCapture(INPUT); ok,first=cap.read(); prev=gray(first); prev_pts=None; Ts=[]; counts=[]; scores=[]; t0=time.time();
for i in range(1,n):
 ok,frame=cap.read()
 if not ok: break
 cur=gray(frame)
 pts=cv2.goodFeaturesToTrack(prev,maxCorners=MAXC,qualityLevel=0.01,minDistance=20,blockSize=3,mask=mask)
 M=np.eye(2,3,dtype=np.float32); cnt=0; valid=False
 if pts is not None and len(pts)>=3:
  nxt,st,err=cv2.calcOpticalFlowPyrLK(prev,cur,pts,None,winSize=(21,21),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,0.01))
  if nxt is not None and st is not None:
   good=st.reshape(-1).astype(bool); p0=pts[good].reshape(-1,2); p1=nxt[good].reshape(-1,2); cnt=len(p0)
   if cnt>=3:
    M,inl=cv2.estimateAffinePartial2D(p0,p1,method=cv2.RANSAC,ransacReprojThreshold=2.0)
    if M is not None and inl is not None and int(inl.sum())>=3:
     M=M.astype(np.float32); valid=True; scores.append(float(inl.mean()))
 if not valid: M=np.eye(2,3,dtype=np.float32); scores.append(0.0)
 Ts.append(params(M)); counts.append(cnt); prev=cur
 if i%500==0: print('estimate',i,'/',n,flush=True)
cap.release(); Ts=np.asarray(Ts); trajectory=np.cumsum(Ts,axis=0); smooth_tr=smooth(trajectory); corrected=Ts+smooth_tr-trajectory; corrected[:,3]=np.clip(corrected[:,3],0.9995,1.0005)
# pass 2
cap=cv2.VideoCapture(INPUT); writer=cv2.VideoWriter(OUT,cv2.VideoWriter_fourcc(*'mp4v'),fps,(W,H)); ok,f=cap.read(); writer.write(f)
for i in range(len(corrected)):
 ok,f=cap.read()
 if not ok: break
 p=corrected[i].copy(); p[0:2]*=(W/AW); writer.write(cv2.warpAffine(f,makeM(p),(W,H),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT))
 if (i+1)%500==0: print('write',i+1,'/',n,flush=True)
cap.release(); writer.release()
rep={'input':INPUT,'output':OUT,'frames':n,'fps':fps,'analysis_width':AW,'model':'OpenCV videostab equivalent: Shi-Tomasi + LK + RANSAC affine partial + cumulative trajectory + GaussianMotionFilter','radius_frames':RADIUS,'max_corners':MAXC,'valid_inlier_ratio_mean':float(np.mean(scores)),'tracked_points_mean':float(np.mean(counts)),'trajectory_range':{'tx':float(np.ptp(trajectory[:,0])),'ty':float(np.ptp(trajectory[:,1])),'rotation_deg':float(np.degrees(np.ptp(trajectory[:,2])))},'corrected_motion_range':{'tx':float(np.ptp(corrected[:,0])),'ty':float(np.ptp(corrected[:,1])),'rotation_deg':float(np.degrees(np.ptp(corrected[:,2])))}}
json.dump(rep,open(REPORT,'w',encoding='utf-8'),ensure_ascii=False,indent=2); print(json.dumps(rep,ensure_ascii=False,indent=2)); print('elapsed',time.time()-t0)
