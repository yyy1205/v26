import cv2,json,os,time
import numpy as np
INPUT=r'E:\DJI\9\DJI_20240310135754_0193_D.MP4'; BASE=r'D:\yolo26-tracking\video_stabilization_dji0193'; MASK_PATH=os.path.join(BASE,'static_mask.png'); OUT=os.path.join(BASE,'DJI_0193_segment_multiref_phase_ecc.mp4'); REPORT=os.path.join(BASE,'segment_multiref_phase_ecc_report.json')
AW=960; SEG=250; OV=50; NREF=5
cap=cv2.VideoCapture(INPUT); W=int(cap.get(3)); H=int(cap.get(4)); fps=cap.get(5) or 25; n=int(cap.get(7)); cap.release(); ah=round(H*AW/W)
mask=(cv2.resize(cv2.imread(MASK_PATH,0),(AW,ah),interpolation=cv2.INTER_NEAREST)>0).astype(np.uint8)*255
ys,xs=np.where(mask>0); y0,y1,x0,x1=map(int,[ys.min(),ys.max()+1,xs.min(),xs.max()+1]); win=cv2.createHanningWindow((x1-x0,y1-y0),cv2.CV_32F); mm=(mask[y0:y1,x0:x1]>0).astype(np.float32)
def readseg(start,end):
 c=cv2.VideoCapture(INPUT); c.set(cv2.CAP_PROP_POS_FRAMES,start); arr=[]
 for _ in range(end-start):
  ok,f=c.read()
  if not ok: break
  arr.append(cv2.resize(cv2.cvtColor(f,cv2.COLOR_BGR2GRAY),(AW,ah),interpolation=cv2.INTER_AREA).astype(np.float32)/255.)
 c.release(); return arr
def phase(ref,cur):
 try: sh,_=cv2.phaseCorrelate(ref[y0:y1,x0:x1]*mm,cur[y0:y1,x0:x1]*mm,win); return np.array([-sh[0],-sh[1]],np.float32)
 except cv2.error:return np.zeros(2,np.float32)
def ecc(ref,cur,init):
 try:
  w=np.array([[1,0,float(init[0])],[0,1,float(init[1])]],np.float32); crit=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,1e-6)
  score,w=cv2.findTransformECC(ref[y0:y1,x0:x1],cur[y0:y1,x0:x1],w,cv2.MOTION_TRANSLATION,crit,mask[y0:y1,x0:x1],5)
  return np.array([w[0,2],w[1,2]],np.float32),float(score)
 except cv2.error:return init,0.0
def smooth(a):
 if len(a)<7:return a
 p=np.pad(a,((3,3),(0,0)),mode='edge'); z=np.stack([np.median(p[i:i+7],axis=0) for i in range(len(a))]); return cv2.GaussianBlur(z.astype(np.float32),(1,31),8,borderType=cv2.BORDER_REPLICATE)
segments=[]; t0=time.time(); starts=list(range(0,n,SEG-OV))
for si,start in enumerate(starts):
 end=min(n,start+SEG); frames=readseg(start,end); L=len(frames); end=start+L; idxs=np.linspace(0,L-1,NREF,dtype=int).tolist(); refs=[frames[i] for i in idxs]; center=NREF//2; obs=np.zeros((L,2),np.float32); scores=[]
 for j,cur in enumerate(frames):
  pv=np.median(np.stack([phase(r,cur) for r in refs]),axis=0)
  v,s=ecc(refs[center],cur,pv); obs[j]=v; scores.append(s)
 sm=smooth(obs); segments.append({'start':start,'end':end,'corr':-sm*(W/AW),'scores':scores})
 print(f'segment {si+1}/{len(starts)} [{start},{end}) score={np.mean(scores):.5f}',flush=True)
 if end==n:break
corrs=np.zeros((n,2),np.float32); weights=np.zeros(n,np.float32); prev=None
for seg in segments:
 c=seg['corr'].copy(); st,en=seg['start'],seg['end']
 if prev is not None:
  os_=max(st,prev['start']); oe=min(en,prev['end'])
  if oe>os_: c+=np.median(prev['corr'][os_-prev['start']:oe-prev['start']]-c[os_-st:oe-st],axis=0)
 seg['corr_aligned']=c; prev={'start':st,'end':en,'corr':c}
 for j in range(en-st):
  i=st+j; w=1.
  if st>0 and j<OV:w=min(w,(j+1)/(OV+1))
  if en<n and j>=en-st-OV:w=min(w,(en-st-j)/(OV+1))
  corrs[i]+=c[j]*w; weights[i]+=w
corrs/=np.maximum(weights[:,None],1e-6); corrs[np.abs(corrs)<0.05]=0; np.save(os.path.join(BASE,'segment_multiref_phase_ecc_corrections.npy'),corrs)
cap=cv2.VideoCapture(INPUT); writer=cv2.VideoWriter(OUT,cv2.VideoWriter_fourcc(*'mp4v'),fps,(W,H))
for i in range(n):
 ok,f=cap.read()
 if not ok:break
 tx,ty=map(float,corrs[i]); writer.write(cv2.warpAffine(f,np.array([[1,0,tx],[0,1,ty]],np.float32),(W,H),borderMode=cv2.BORDER_REFLECT))
 if i and i%500==0:print('write',i,'/',n,flush=True)
cap.release();writer.release(); report={'input':INPUT,'output':OUT,'frames':n,'fps':fps,'analysis_width':AW,'segment_frames':SEG,'overlap_frames':OV,'references_per_segment':NREF,'method':'multi-reference phase correlation + one ECC translation refinement + median/Gaussian smoothing + overlap blending','corr_stats':{k:{'mean_abs':float(np.mean(np.abs(corrs[:,j]))),'p95_abs':float(np.percentile(np.abs(corrs[:,j]),95)),'max_abs':float(np.max(np.abs(corrs[:,j])))} for j,k in enumerate(['tx_px','ty_px'])}}
json.dump(report,open(REPORT,'w',encoding='utf-8'),ensure_ascii=False,indent=2); print(json.dumps(report,ensure_ascii=False,indent=2)); print('elapsed',time.time()-t0)

