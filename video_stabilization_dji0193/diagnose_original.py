import argparse,cv2,numpy as np,json
from pathlib import Path

def run(inp,mask_path,out,aw=960):
 cap=cv2.VideoCapture(inp); fps=cap.get(cv2.CAP_PROP_FPS) or 25.; n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); ok,first=cap.read(); ah=round(first.shape[0]*aw/first.shape[1]); mask=cv2.resize(cv2.imread(mask_path,0),(aw,ah),interpolation=cv2.INTER_NEAREST); prev=cv2.resize(cv2.cvtColor(first,cv2.COLOR_BGR2GRAY),(aw,ah)).astype(np.float32)/255.; rows=[]; i=1
 while True:
  ok,f=cap.read()
  if not ok: break
  cur=cv2.resize(cv2.cvtColor(f,cv2.COLOR_BGR2GRAY),(aw,ah)).astype(np.float32)/255.; wm=np.eye(2,3,dtype=np.float32)
  try:
   score,wm=cv2.findTransformECC(prev,cur,wm,cv2.MOTION_EUCLIDEAN,(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,50,1e-5),mask,5); a,b=wm[0,0],wm[1,0]; rows.append([i,float(wm[0,2]),float(wm[1,2]),float(np.degrees(np.arctan2(b,a))),float(score),1])
  except cv2.error: rows.append([i,0.,0.,0.,0.,0])
  prev=cur
  if i%500==0: print(i,'/',n,flush=True)
  i+=1
 cap.release(); a=np.asarray(rows); valid=a[:,5]>0; x=a[valid];
 def stats(v): return {'mean':float(np.mean(v)),'std':float(np.std(v)),'p50_abs':float(np.percentile(np.abs(v),50)),'p95_abs':float(np.percentile(np.abs(v),95)),'p99_abs':float(np.percentile(np.abs(v),99)),'max_abs':float(np.max(np.abs(v)))}
 # low-frequency trend over 125 frames and residual random component
 w=125; drift=np.empty_like(x[:,1:4]);
 for j in range(3): drift[:,j]=np.convolve(np.pad(x[:,j+1],(w//2,w//2),mode='edge'),np.ones(w)/w,mode='valid')
 residual=x[:,1:4]-drift
 report={'input':inp,'fps':fps,'frames':len(rows),'analysis_width':aw,'valid_ratio':float(valid.mean()),'score':stats(a[:,4]),'frame_to_frame':{'tx':stats(a[:,1]),'ty':stats(a[:,2]),'rotation_deg':stats(a[:,3])},'random_residual_after_125f_lowpass':{'tx':stats(residual[:,0]),'ty':stats(residual[:,1]),'rotation_deg':stats(residual[:,2])},'low_frequency_drift_range':{'tx':float(np.ptp(drift[:,0])),'ty':float(np.ptp(drift[:,1])),'rotation_deg':float(np.ptp(drift[:,2]))},'outliers':{'score_below_0.9':int((a[:,4]<.9).sum()),'translation_over_1px':int((np.linalg.norm(a[:,1:3],axis=1)>1).sum()),'rotation_over_0.05deg':int((np.abs(a[:,3])>.05).sum())},'frames_detail':rows}
 Path(out).write_text(json.dumps(report,indent=2),encoding='utf-8'); print(json.dumps({k:v for k,v in report.items() if k!='frames_detail'},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--mask',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.input,a.mask,a.output)

