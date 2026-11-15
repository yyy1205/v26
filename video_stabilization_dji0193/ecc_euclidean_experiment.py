import argparse,cv2,numpy as np
def run(inp,out,start=3000,count=50,mask_path=None):
 c=cv2.VideoCapture(inp); fps=c.get(cv2.CAP_PROP_FPS) or 25.; c.set(cv2.CAP_PROP_POS_FRAMES,start); ok,first=c.read(); h,w=first.shape[:2]; aw,ah=960,540; ref=cv2.resize(cv2.cvtColor(first,cv2.COLOR_BGR2GRAY),(aw,ah)).astype(np.float32)/255.; mask=cv2.resize(cv2.imread(mask_path,0),(aw,ah),interpolation=cv2.INTER_NEAREST); frames=[]; poses=[]
 while ok and len(frames)<count:
  frames.append(first.copy()); cur=cv2.resize(cv2.cvtColor(first,cv2.COLOR_BGR2GRAY),(aw,ah)).astype(np.float32)/255.; warp=np.eye(2,3,dtype=np.float32)
  try:
   score,wm=cv2.findTransformECC(ref,cur,warp,cv2.MOTION_EUCLIDEAN,(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,80,1e-6),mask,5); poses.append([wm[0,2],wm[1,2],np.degrees(np.arctan2(wm[1,0],wm[0,0]))])
  except cv2.error: poses.append(poses[-1] if poses else [0,0,0])
  ok,first=c.read()
 c.release(); a=np.asarray(poses); b=np.empty_like(a)
 for j in range(3): b[:,j]=np.convolve(np.pad(a[:,j],(3,3),mode='edge'),np.ones(7)/7,mode='valid')
 wr=cv2.VideoWriter(out,cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h)); sx,sy=w/aw,h/ah
 for f,(tx,ty,r) in zip(frames,b):
  t=np.radians(r); M=np.array([[np.cos(t),-np.sin(t),-tx*sx],[np.sin(t),np.cos(t),-ty*sy]],np.float32); wr.write(cv2.warpAffine(f,M,(w,h),borderMode=cv2.BORDER_REFLECT))
 wr.release(); print('frames',len(frames),'pose_std',a.std(0))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--mask',required=True);p.add_argument('--start',type=int,default=3000);p.add_argument('--count',type=int,default=50);a=p.parse_args();run(a.input,a.output,a.start,a.count,a.mask)
