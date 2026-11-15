import argparse,cv2,numpy as np

def run(inp,out,start=3000,count=50,mask_path=None,model=cv2.MOTION_EUCLIDEAN):
 c=cv2.VideoCapture(inp); fps=c.get(cv2.CAP_PROP_FPS) or 25; c.set(cv2.CAP_PROP_POS_FRAMES,start); ok,first=c.read()
 if not ok: raise RuntimeError('read failed')
 h,w=first.shape[:2]; aw,ah=960,540; ref=cv2.resize(cv2.cvtColor(first,cv2.COLOR_BGR2GRAY),(aw,ah)).astype(np.float32)/255.; mask=cv2.resize(cv2.imread(mask_path,0),(aw,ah),interpolation=cv2.INTER_NEAREST) if mask_path else np.ones((ah,aw),np.uint8)*255
 frames=[]; shifts=[]; ok=True
 while ok and len(frames)<count:
  frames.append(first.copy()); cur=cv2.resize(cv2.cvtColor(first,cv2.COLOR_BGR2GRAY),(aw,ah)).astype(np.float32)/255.
  try:
   warp=np.array([[1,0,0],[0,1,0]],np.float32); score,cc=cv2.findTransformECC(ref,cur,warp,model,(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,80,1e-6),mask,5); shifts.append([float(cc[0,2]),float(cc[1,2]),0.,float(score)])
  except cv2.error: shifts.append([0.,0.,0.,0.])
  ok,first=c.read()
 c.release(); a=np.asarray(shifts); # robust temporal smoothing
 for j in (0,1): a[:,j]=np.convolve(np.pad(a[:,j],(3,3),mode='edge'),np.ones(7)/7,mode='valid')
 writer=cv2.VideoWriter(out,cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h)); sx,sy=w/aw,h/ah
 for f,(tx,ty,_,_) in zip(frames,a): writer.write(cv2.warpAffine(f,np.array([[1,0,-tx*sx],[0,1,-ty*sy]],np.float32),(w,h),borderMode=cv2.BORDER_REFLECT))
 writer.release(); print('frames',len(frames),'mean response',a[:,2].mean(),'mean ecc',a[:,3].mean(),'shift std',a[:,:2].std(0))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--start',type=int,default=3000);p.add_argument('--count',type=int,default=50);p.add_argument('--mask');a=p.parse_args();run(a.input,a.output,a.start,a.count,a.mask)

