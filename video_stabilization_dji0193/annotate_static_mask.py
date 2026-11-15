"""Interactive static-background mask annotation.
Left click: add polygon vertex
Enter: fill current polygon as static (white)
Right click: undo last vertex
R: clear current polygon
S: save static_mask.png
Q/Esc: quit
"""
import argparse
from pathlib import Path
import cv2
import numpy as np

points=[]
mask=None
image=None
scale=1.0
window='Static background annotation'

def redraw():
    canvas=image.copy()
    if mask is not None:
        overlay=canvas.copy(); overlay[mask>0]=(0,255,0)
        canvas=cv2.addWeighted(canvas,0.72,overlay,0.28,0)
    if points:
        pts=np.array(points,np.int32)
        cv2.polylines(canvas,[pts],False,(0,0,255),3)
        for i,(x,y) in enumerate(points): cv2.circle(canvas,(x,y),6,(0,0,255),-1); cv2.putText(canvas,str(i+1),(x+8,y-8),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,255),2)
    view=cv2.resize(canvas,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
    cv2.putText(view,'Left: points | Enter: fill | Right: undo | R: reset | S: save | Q: quit',(15,30),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,255,255),2)
    cv2.imshow(window,view)

def mouse(event,x,y,flags,param):
    global points
    if event==cv2.EVENT_LBUTTONDOWN:
        points.append((int(x/scale),int(y/scale))); redraw()
    elif event==cv2.EVENT_RBUTTONDOWN and points:
        points.pop(); redraw()

def main():
    global image,mask,scale
    ap=argparse.ArgumentParser(); ap.add_argument('--image',default=str(Path(__file__).with_name('first_frame.png'))); ap.add_argument('--output',default=str(Path(__file__).with_name('static_mask.png'))); args=ap.parse_args()
    image=cv2.imread(args.image)
    if image is None: raise SystemExit(f'Cannot read {args.image}')
    h,w=image.shape[:2]; max_w,max_h=1600,900; scale=min(1.0,max_w/w,max_h/h); mask=np.zeros((h,w),np.uint8)
    cv2.namedWindow(window,cv2.WINDOW_AUTOSIZE); cv2.setMouseCallback(window,mouse); redraw()
    while True:
        key=cv2.waitKey(30)&0xff
        if key in (13,10):
            if len(points)>=3: cv2.fillPoly(mask,[np.array(points,np.int32)],255); points.clear(); redraw()
        elif key in (ord('r'),ord('R')): points.clear(); redraw()
        elif key in (ord('s'),ord('S')):
            cv2.imwrite(args.output,mask); print(f'Saved {args.output}')
        elif key in (ord('q'),ord('Q'),27): break
    cv2.destroyAllWindows()
if __name__=='__main__': main()
