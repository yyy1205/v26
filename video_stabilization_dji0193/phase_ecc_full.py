import argparse, subprocess, shutil
from pathlib import Path
import cv2, numpy as np

def smooth(a,w=7):
 a=np.asarray(a,float); out=np.empty_like(a); pad=w//2
 for j in range(a.shape[1]): out[:,j]=np.convolve(np.pad(a[:,j],(pad,pad),mode='edge'),np.ones(w)/w,mode='valid')
 return out

def ffmpeg():
 return shutil.which('ffmpeg') or r'C:\Tools\ffmpeg\ffmpeg-9.0.1-essentials_build\bin\ffmpeg.exe'
def run(inp,out,analysis_width=960,window=7,mask_path=None):
 cap=cv2.VideoCapture(inp); fps=cap.get(cv2.CAP_PROP_FPS) or 25.; n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); ok,first=cap.read()
 if not ok: raise RuntimeError('input unreadable')
 h,w=first.shape[:2]; ah=round(h*analysis_width/w); ref=cv2.resize(cv2.cvtColor(first,cv2.COLOR_BGR2GRAY),(analysis_width,ah)).astype(np.float32)/255.; mask=cv2.resize(cv2.imread(mask_path,0),(analysis_width,ah),interpolation=cv2.INTER_NEAREST) if mask_path else np.ones((ah,analysis_width),np.uint8)*255
 shifts=[[0.,0.]]; idx=1; prev=None
 while True:
  ok,frame=cap.read()
  if not ok: break
  cur=cv2.resize(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),(analysis_width,ah)).astype(np.float32)/255.; warp=np.eye(2,3,dtype=np.float32)
  try:
   score,wm=cv2.findTransformECC(ref,cur,warp,cv2.MOTION_TRANSLATION,(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,60,1e-5),mask,5); shifts.append([float(wm[0,2]),float(wm[1,2])])
  except cv2.error: shifts.append(shifts[-1])
  if idx%500==0: print('analysis',idx,'/',n,flush=True)
  idx+=1
 cap.release(); shifts=smooth(np.asarray(shifts),window)
 cap=cv2.VideoCapture(inp); ok,frame=cap.read(); cmd=[ffmpeg(),'-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'{w}x{h}','-r',str(fps),'-i','-','-i',inp,'-map','0:v:0','-map','1:a?','-c:v','libx264','-preset','medium','-crf','18','-pix_fmt','yuv420p','-c:a','copy','-shortest',out]; log=str(Path(out).with_suffix('.ffmpeg.log')); err=open(log,'wb'); proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=err); i=0; sx=w/analysis_width; sy=h/ah
 while ok and i<len(shifts):
  tx,ty=shifts[i]; M=np.array([[1,0,-tx*sx],[0,1,-ty*sy]],np.float32); proc.stdin.write(cv2.warpAffine(frame,M,(w,h),borderMode=cv2.BORDER_REFLECT).tobytes()); i+=1; ok,frame=cap.read()
 proc.stdin.close(); rc=proc.wait(); err.close(); cap.release()
 if rc: raise RuntimeError(rc)
 print('output',i,'frames')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--window',type=int,default=7);p.add_argument('--mask');a=p.parse_args();run(a.input,a.output,window=a.window,mask_path=a.mask)
