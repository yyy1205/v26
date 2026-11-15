"""Conservative fixed-camera mode: no warp unless low-frequency drift exceeds deadband."""
import cv2,numpy as np,subprocess,json
from pathlib import Path
from .nadir_conservative import estimate_pair
def run(inp,out,max_frames=300,proxy_width=960,lowpass=125,deadband=.5,ffmpeg=r'C:\Tools\ffmpeg\ffmpeg-9.0.1-essentials_build\bin\ffmpeg.exe'):
 c=cv2.VideoCapture(inp); fps=c.get(cv2.CAP_PROP_FPS); ok,f=c.read();
 if not ok: raise RuntimeError('empty')
 h,w=f.shape[:2]; eh=round(h*proxy_width/w); prev=cv2.resize(cv2.cvtColor(f,cv2.COLOR_BGR2GRAY),(proxy_width,eh)); shifts=[]
 for i in range(1,max_frames):
  ok,f=c.read()
  if not ok: break
  cur=cv2.resize(cv2.cvtColor(f,cv2.COLOR_BGR2GRAY),(proxy_width,eh)); M,r=estimate_pair(prev,cur,None,'translation'); shifts.append((M[0,2],M[1,2],r)); prev=cur
 c.release(); n=len(shifts)+1; tr=np.vstack([[0,0],np.cumsum(np.asarray(shifts)[:,:2],axis=0)]); k=np.ones(min(lowpass,n)); k/=k.sum(); sm=np.column_stack([np.convolve(np.pad(tr[:,j],(len(k)//2,)*2,mode='edge'),k,'valid') for j in range(2)]); corr=sm-tr; corr[np.linalg.norm(corr,axis=1)<deadband]=0; corr=np.clip(corr,-1.0,1.0); Path(str(out)+'.metrics.json').write_text(json.dumps({'frames':n,'proxy_resolution_px':[proxy_width,eh],'deadband_proxy_px':deadband,'mean_phase_response':float(np.mean([x[2] for x in shifts]))}),encoding='utf-8')
 # second pass: apply only tiny low-frequency correction, then encode with FFmpeg
 cmd=[ffmpeg,'-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'{w}x{h}','-r',str(fps),'-i','pipe:0','-c:v','libx264','-preset','ultrafast','-crf','23','-pix_fmt','yuv420p','-an',out]; p=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.DEVNULL); c=cv2.VideoCapture(inp); i=0
 while i<n:
  ok,f=c.read()
  if not ok: break
  C=np.float32([[1,0,corr[i,0]*w/proxy_width],[0,1,corr[i,1]*h/eh]]); p.stdin.write(cv2.warpAffine(f,C,(w,h),borderMode=cv2.BORDER_REFLECT).tobytes()); i+=1
 c.release(); p.stdin.close(); return p.wait()
