import argparse,cv2,numpy as np,subprocess,shutil
from pathlib import Path
def smooth(a,w):
 w=min(w,len(a) if len(a)%2 else len(a)-1); w=max(3,w); pad=w//2; out=np.empty_like(a)
 k=np.ones(w)/w
 for j in range(a.shape[1]): out[:,j]=np.convolve(np.pad(a[:,j],(pad,pad),mode='edge'),k,'valid')
 return out
def run(inp,out,mask_path,aw=960,window=1001,deadband=0.02):
 cap=cv2.VideoCapture(inp); fps=cap.get(cv2.CAP_PROP_FPS) or 25.; n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); ok,first=cap.read(); h,w=first.shape[:2]; ah=round(h*aw/w); ref=cv2.resize(cv2.cvtColor(first,cv2.COLOR_BGR2GRAY),(aw,ah)).astype(np.float32)/255.; mask=cv2.resize(cv2.imread(mask_path,0),(aw,ah),interpolation=cv2.INTER_NEAREST); poses=[[0.,0.,0.]]; i=1
 while True:
  ok,f=cap.read()
  if not ok: break
  cur=cv2.resize(cv2.cvtColor(f,cv2.COLOR_BGR2GRAY),(aw,ah)).astype(np.float32)/255.; wm=np.eye(2,3,dtype=np.float32)
  try:
   _,wm=cv2.findTransformECC(ref,cur,wm,cv2.MOTION_EUCLIDEAN,(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,60,1e-5),mask,5); poses.append([float(wm[0,2]),float(wm[1,2]),float(np.degrees(np.arctan2(wm[1,0],wm[0,0])))])
  except cv2.error: poses.append(poses[-1])
  if i%500==0: print('analysis',i,'/',n,flush=True)
  i+=1
 cap.release(); p=np.asarray(poses); drift=smooth(p,window); drift-=drift[0]; drift[:,2]=np.where(np.abs(drift[:,2])<deadband,0,drift[:,2]); drift[:,0]=np.where(np.abs(drift[:,0])<.15,0,drift[:,0]); drift[:,1]=np.where(np.abs(drift[:,1])<.15,0,drift[:,1]); np.save(str(Path(out).with_suffix('.drift.npy')),drift)
 cap=cv2.VideoCapture(inp); ok,f=cap.read(); ff=shutil.which('ffmpeg') or r'C:\Tools\ffmpeg\ffmpeg-9.0.1-essentials_build\bin\ffmpeg.exe'; cmd=[ff,'-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'{w}x{h}','-r',str(fps),'-i','-','-i',inp,'-map','0:v:0','-map','1:a?','-c:v','libx264','-preset','medium','-crf','18','-pix_fmt','yuv420p','-c:a','copy','-shortest',out]; proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.DEVNULL); i=0; sx,sy=w/aw,h/ah
 while ok and i<len(drift):
  tx,ty,r=drift[i]; a=np.radians(r); M=np.array([[np.cos(a),-np.sin(a),-tx*sx],[np.sin(a),np.cos(a),-ty*sy]],np.float32); proc.stdin.write(cv2.warpAffine(f,M,(w,h),borderMode=cv2.BORDER_REFLECT).tobytes()); i+=1; ok,f=cap.read()
 proc.stdin.close(); proc.wait(); cap.release(); print('output',i)
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--input',required=True); p.add_argument('--output',required=True); p.add_argument('--mask',required=True); p.add_argument('--window',type=int,default=1001); a=p.parse_args(); run(a.input,a.output,a.mask,window=a.window)
