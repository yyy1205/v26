import argparse
from collections import deque
from pathlib import Path
import cv2
import numpy as np


def stabilize_segment(input_path, output_path, start_frame=3000, count=50, analysis_width=480, temporal_window=7, strength=0.8):
    cap=cv2.VideoCapture(str(input_path)); fps=cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES,start_frame); ok,first=cap.read()
    if not ok: raise RuntimeError("cannot read segment")
    h,w=first.shape[:2]; ah=round(h*analysis_width/w)
    prev=cv2.cvtColor(cv2.resize(first,(analysis_width,ah)),cv2.COLOR_BGR2GRAY)
    flow_history=deque(maxlen=temporal_window)
    writer=cv2.VideoWriter(str(output_path),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
    writer.write(first); frames=1
    yy,xx=np.mgrid[0:ah,0:analysis_width].astype(np.float32)
    while frames<count:
        ok,frame=cap.read()
        if not ok: break
        small=cv2.cvtColor(cv2.resize(frame,(analysis_width,ah)),cv2.COLOR_BGR2GRAY)
        flow=cv2.calcOpticalFlowFarneback(prev,small,None,0.5,3,25,3,5,1.2,0)
        # robust spatial smoothing suppresses moving cars and isolated flow spikes
        fx=cv2.medianBlur(flow[...,0].astype(np.float32),5); fy=cv2.medianBlur(flow[...,1].astype(np.float32),5)
        flow_history.append(np.dstack([fx,fy]))
        robust=np.median(np.stack(flow_history,axis=0),axis=0).astype(np.float32)
        robust=cv2.GaussianBlur(robust,(0,0),1.2)
        # Apply the opposite of the robust background motion.
        mapx=xx-strength*robust[...,0]; mapy=yy-strength*robust[...,1]
        mapx=cv2.resize(mapx,(w,h),interpolation=cv2.INTER_LINEAR); mapy=cv2.resize(mapy,(w,h),interpolation=cv2.INTER_LINEAR)
        out=cv2.remap(frame,mapx,mapy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
        writer.write(out); prev=small; frames+=1
    writer.release(); cap.release(); return frames

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--output',required=True); ap.add_argument('--start',type=int,default=3000); ap.add_argument('--count',type=int,default=50); ap.add_argument('--strength',type=float,default=.8); a=ap.parse_args(); print(stabilize_segment(a.input,a.output,a.start,a.count,strength=a.strength))
