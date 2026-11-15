import cv2, numpy as np, json, os, time
INPUT=r'E:\DJI\9\DJI_20240310135754_0193_D.MP4'; BASE=r'D:\yolo26-tracking\video_stabilization_dji0193'; MASK_PATH=os.path.join(BASE,'static_mask.png'); OUT=os.path.join(BASE,'DJI_0193_multiscale_adaptive_confidence_ecc.mp4'); REPORT=os.path.join(BASE,'multiscale_adaptive_confidence_ecc_report.json')
AW=640; LEVELS=[0.5,1.0]; RADIUS=62; SCORE_THR=0.985; ROT_DEADBAND=np.deg2rad(0.003); MAX_ROT=np.deg2rad(0.02)
cap=cv2.VideoCapture(INPUT); W=int(cap.get(3)); H=int(cap.get(4)); fps=cap.get(5) or 25; n=int(cap.get(7)); cap.release(); ah=round(H*AW/W)
base_mask=(cv2.resize(cv2.imread(MASK_PATH,0),(AW,ah),interpolation=cv2.INTER_NEAREST)>0).astype(np.uint8)*255
# build scaled masks; use full static-mask bounding rectangle to speed ECC
masks=[]
for s in LEVELS:
 w=max(32,round(AW*s)); h=max(32,round(ah*s)); masks.append(cv2.resize(base_mask,(w,h),interpolation=cv2.INTER_NEAREST))
def gray(f,s):
 w=max(32,round(AW*s)); h=max(32,round(ah*s)); return cv2.resize(cv2.cvtColor(f,cv2.COLOR_BGR2GRAY),(w,h),interpolation=cv2.INTER_AREA).astype(np.float32)/255.0
def smooth(a):
 k=2*RADIUS+1; return cv2.GaussianBlur(a.astype(np.float32),(1,k),0,borderType=cv2.BORDER_REPLICATE)
def phase_init(a,b,mask):
 try:
  mm=(mask>0).astype(np.float32); ys,xs=np.where(mask>0); y0,y1,x0,x1=ys.min(),ys.max()+1,xs.min(),xs.max()+1
  win=cv2.createHanningWindow((x1-x0,y1-y0),cv2.CV_32F); sh,_=cv2.phaseCorrelate(a[y0:y1,x0:x1]*mm[y0:y1,x0:x1],b[y0:y1,x0:x1]*mm[y0:y1,x0:x1],win); return np.array([-sh[0],-sh[1]],np.float32)
 except Exception:return np.zeros(2,np.float32)
def estimate(prevs,curs):
 # phase init at coarsest level
 v=phase_init(prevs[0],curs[0],masks[0]); theta=0.; score=0.; used='translation'
 for li,(a,b,mask) in enumerate(zip(prevs,curs,masks)):
  s=LEVELS[li]; w=np.array([[np.cos(theta),-np.sin(theta),v[0]*s],[np.sin(theta),np.cos(theta),v[1]*s]],np.float32)
  crit=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,35,1e-6)
  try:
   sc,w=cv2.findTransformECC(a,b,w,cv2.MOTION_EUCLIDEAN,crit,mask,5); score=float(sc); theta=float(np.arctan2(w[1,0],w[0,0])); v=np.array([w[0,2]/s,w[1,2]/s],np.float32)
  except cv2.error: pass
 # adaptive: ignore rotation if it is below noise/deadband; retain Euclidean only when confident
 if score<SCORE_THR or not np.isfinite(score): return np.array([0.,0.,0.],np.float64),0.,'rejected'
 if abs(theta)<ROT_DEADBAND: theta=0.; used='translation'
 theta=float(np.clip(theta,-MAX_ROT,MAX_ROT)); return np.array([v[0],v[1],theta],np.float64),score,used
# pass 1
cap=cv2.VideoCapture(INPUT); ok,first=cap.read(); prevs=[gray(first,s) for s in LEVELS]; motions=[]; scores=[]; models=[]; rejects=0; t0=time.time()
for i in range(1,n):
 ok,frame=cap.read()
 if not ok: break
 curs=[gray(frame,s) for s in LEVELS]; p,sc,model=estimate(prevs,curs)
 if model=='rejected': rejects+=1; p=np.array([0.,0.,0.],np.float64)
 motions.append(p); scores.append(sc); models.append(model); prevs=curs
 if i%500==0: print('estimate',i,'/',n,'score',sc,'model',model,flush=True)
cap.release(); motions=np.asarray(motions); traj=np.cumsum(motions,axis=0); sm=smooth(traj); corrected=motions+sm-traj; corrected[:,2]=np.clip(corrected[:,2],-MAX_ROT,MAX_ROT); corrected[np.abs(corrected[:,2])<ROT_DEADBAND]=0
# pass 2
cap=cv2.VideoCapture(INPUT); writer=cv2.VideoWriter(OUT,cv2.VideoWriter_fourcc(*'mp4v'),fps,(W,H)); ok,f=cap.read(); writer.write(f)
for i in range(len(corrected)):
 ok,f=cap.read()
 if not ok:break
 tx,ty,th=corrected[i]; tx*=W/AW; ty*=W/AW; c=np.cos(th); s=np.sin(th); M=np.array([[c,-s,tx],[s,c,ty]],np.float32); writer.write(cv2.warpAffine(f,M,(W,H),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT))
 if (i+1)%500==0: print('write',i+1,'/',n,flush=True)
cap.release(); writer.release()
valid=np.array(scores)>0
rep={'input':INPUT,'output':OUT,'frames':n,'fps':fps,'analysis_width':AW,'pyramid_levels':LEVELS,'smooth_radius_frames':RADIUS,'score_threshold':SCORE_THR,'rotation_deadband_deg':float(np.degrees(ROT_DEADBAND)),'max_rotation_deg':float(np.degrees(MAX_ROT)),'method':'static-mask multiscale phase-initialized adaptive ECC with confidence gating and trajectory smoothing','rejected_frames':int(rejects),'valid_ratio':float(valid.mean()),'score_summary':{'mean':float(np.mean(scores)),'p05':float(np.percentile(scores,5)),'p95':float(np.percentile(scores,95)),'min':float(np.min(scores))},'model_counts':{k:int(models.count(k)) for k in sorted(set(models))},'trajectory_range':{'tx':float(np.ptp(traj[:,0])),'ty':float(np.ptp(traj[:,1])),'rotation_deg':float(np.degrees(np.ptp(traj[:,2])))},'corrected_range':{'tx':float(np.ptp(corrected[:,0])),'ty':float(np.ptp(corrected[:,1])),'rotation_deg':float(np.degrees(np.ptp(corrected[:,2])))}}
json.dump(rep,open(REPORT,'w',encoding='utf-8'),ensure_ascii=False,indent=2); print(json.dumps(rep,ensure_ascii=False,indent=2)); print('elapsed',time.time()-t0)

