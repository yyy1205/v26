import argparse,cv2,numpy as np

def run(inp,out,start=3000,count=50,mask_path=None):
 c=cv2.VideoCapture(inp); fps=c.get(cv2.CAP_PROP_FPS) or 25; c.set(cv2.CAP_PROP_POS_FRAMES,start); ok,first=c.read();
 if not ok: raise RuntimeError('read failed')
 h,w=first.shape[:2]; aw,ah=960,540; ref=cv2.resize(cv2.cvtColor(first,cv2.COLOR_BGR2GRAY),(aw,ah)); mask=cv2.resize(cv2.imread(mask_path,0),(aw,ah),interpolation=cv2.INTER_NEAREST) if mask_path else None; sift=cv2.SIFT_create(nfeatures=1800,contrastThreshold=.02); k1,d1=sift.detectAndCompute(ref,mask); flann=cv2.FlannBasedMatcher(dict(algorithm=1,trees=5),dict(checks=50)); frames=[]; poses=[]
 while ok and len(frames)<count:
  frames.append(first.copy()); cur=cv2.resize(cv2.cvtColor(first,cv2.COLOR_BGR2GRAY),(aw,ah)); k2,d2=sift.detectAndCompute(cur,mask); M=None
  if d2 is not None:
   ms=flann.knnMatch(d1,d2,k=2); good=[m for m,n in ms if m.distance<.7*n.distance]
   if len(good)>=12:
    p=np.float32([k1[m.queryIdx].pt for m in good]); q=np.float32([k2[m.trainIdx].pt for m in good]); M,inl=cv2.estimateAffinePartial2D(p,q,method=cv2.RANSAC,ransacReprojThreshold=2.5)
  if M is None: poses.append([0.,0.,0.])
  else: poses.append([float(M[0,2]),float(M[1,2]),float(np.degrees(np.arctan2(M[1,0],M[0,0])))])
  ok,first=c.read()
 c.release(); a=np.asarray(poses); outp=np.empty_like(a)
 for j in range(3): outp[:,j]=np.convolve(np.pad(a[:,j],(3,3),mode='edge'),np.ones(7)/7,mode='valid')
 writer=cv2.VideoWriter(out,cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h)); sx,sy=w/aw,h/ah
 for f,(tx,ty,r) in zip(frames,outp):
  ang=np.radians(r); M=np.array([[np.cos(ang),-np.sin(ang),tx],[np.sin(ang),np.cos(ang),ty]],np.float32); M[0,2]*=-sx; M[1,2]*=-sy; writer.write(cv2.warpAffine(f,M,(w,h),borderMode=cv2.BORDER_REFLECT))
 writer.release(); print('frames',len(frames),'refkeypoints',len(k1),'pose_std',a.std(0))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--start',type=int,default=3000);p.add_argument('--count',type=int,default=50);p.add_argument('--mask');a=p.parse_args();run(a.input,a.output,a.start,a.count,a.mask)
