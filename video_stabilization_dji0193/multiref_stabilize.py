import argparse, cv2, numpy as np
from pathlib import Path

def pose(M):
 a,b=M[0,0],M[1,0]; return np.array([M[0,2],M[1,2],np.degrees(np.arctan2(b,a)),np.hypot(a,b)],float)
def mat(p):
 tx,ty,r,s=p; a=np.radians(r); return np.array([[s*np.cos(a),-s*np.sin(a),tx],[s*np.sin(a),s*np.cos(a),ty],[0,0,1]],np.float32)
def estimate(ref,cur,orb):
 k1,d1=orb.detectAndCompute(ref,None); k2,d2=orb.detectAndCompute(cur,None)
 if d1 is None or d2 is None:return None
 ms=cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1,d2,k=2); good=[m for m,n in ms if m.distance<.72*n.distance]
 if len(good)<12:return None
 p=np.float32([k1[m.queryIdx].pt for m in good]); q=np.float32([k2[m.trainIdx].pt for m in good]); M,inl=cv2.estimateAffinePartial2D(p,q,method=cv2.RANSAC,ransacReprojThreshold=2.5)
 if M is None or inl is None or inl.mean()<.45:return None
 return pose(M)
def smooth(a,w=31):
 if len(a)<3:return a
 w=min(w if w%2 else w-1,len(a) if len(a)%2 else len(a)-1); pad=w//2; out=[]
 for i in range(len(a)):
  lo=max(0,i-pad); hi=min(len(a),i+pad+1); out.append(np.median(a[lo:hi],axis=0))
 return np.asarray(out)
def run(inp,out,start=3000,count=50,ref_step=750):
 cap=cv2.VideoCapture(inp); fps=cap.get(cv2.CAP_PROP_FPS) or 25; n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); aw=960; ah=540
 refs=[]
 for idx in range(0,n,ref_step):
  cap.set(cv2.CAP_PROP_POS_FRAMES,idx); ok,f=cap.read()
  if ok: refs.append((idx,cv2.resize(cv2.cvtColor(f,cv2.COLOR_BGR2GRAY),(aw,ah))))
 cap.set(cv2.CAP_PROP_POS_FRAMES,start); ok,first=cap.read();
 if not ok: raise RuntimeError('cannot read')
 h,w=first.shape[:2]; orb=cv2.ORB_create(2200,1.2,8,fastThreshold=12); poses=[]; frames=[]
 while ok and len(frames)<count:
  frames.append(first.copy()); cur=cv2.resize(cv2.cvtColor(first,cv2.COLOR_BGR2GRAY),(aw,ah)); ps=[]
  for _,r in refs: 
   z=estimate(r,cur,orb)
   if z is not None: ps.append(z)
  poses.append(np.median(np.asarray(ps),axis=0) if ps else np.zeros(4))
  ok,first=cap.read()
 cap.release(); poses=smooth(np.asarray(poses),31); writer=cv2.VideoWriter(out,cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h)); sx=w/aw; sy=h/ah
 for f,p in zip(frames,poses):
  M=mat(p); S=np.diag([sx,sy,1.]); C=S@np.linalg.inv(M)@np.linalg.inv(S); C[:2,2]=C[:2,2] # inverse ref->current
  writer.write(cv2.warpAffine(f,C[:2],(w,h),borderMode=cv2.BORDER_REFLECT))
 writer.release(); return len(frames),len(refs)
if __name__=='__main__':
 ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--output',required=True); ap.add_argument('--start',type=int,default=3000); ap.add_argument('--count',type=int,default=50); a=ap.parse_args(); print(run(a.input,a.output,a.start,a.count))
