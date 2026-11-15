import os, cv2, numpy as np

IMG = r"F:\vis\frame_000001.png"
GT_TXT = r"D:\yolo26-tracking\outputs\tuvis-txt\labels\1.txt"
DIRECT_TXT = r"D:\yolo26-tracking\outputs\v8_v26_direct_sizes\exp63_960\labels\frame_000001_exp63_direct960.txt"
OUT = r"D:\yolo26-tracking\outputs\publication\exp63_960_sahi_advantage"
os.makedirs(OUT, exist_ok=True)

def read_txt(path):
    out=[]
    with open(path, encoding="utf-8") as f:
        for line in f:
            p=line.rstrip("\n").split("\t")
            if len(p) < 10: continue
            try:
                pts=np.array([float(x) for x in p[2:10]], np.float32).reshape(4,2)
                out.append({"cls":int(p[0]), "conf":float(p[1]), "pts":pts, "name":p[10] if len(p)>10 else str(p[0])})
            except ValueError: pass
    return out

def iou(a,b):
    a=cv2.convexHull(a); b=cv2.convexHull(b)
    aa,bb=abs(cv2.contourArea(a)),abs(cv2.contourArea(b))
    if aa<=0 or bb<=0:return 0.0
    inter,_=cv2.intersectConvexConvex(a,b); u=aa+bb-inter
    return float(inter/u) if u>0 else 0.0

def draw_poly(im, pts, color, label=None, thickness=5, font=1.0):
    pts=np.round(pts).astype(np.int32).reshape(-1,1,2)
    cv2.polylines(im,[pts],True,color,thickness,cv2.LINE_AA)
    if label:
        x,y=pts[:,0,0].min(),pts[:,0,1].min()
        cv2.putText(im,label,(int(x),max(25,int(y)-8)),cv2.FONT_HERSHEY_SIMPLEX,font,color,3,cv2.LINE_AA)

gt=read_txt(GT_TXT); direct=read_txt(DIRECT_TXT)
missed=[]; matched=0
for g in gt:
    best=max((iou(g["pts"],d["pts"]) for d in direct if d["cls"]==g["cls"]), default=0.0)
    if best >= 0.30: matched += 1
    else: missed.append(g)

image=cv2.imread(IMG); h,w=image.shape[:2]
cell=640; counts=[]
for gy in range(0,h,cell):
    for gx in range(0,w,cell):
        items=[m for m in missed if gx <= float(m["pts"][:,0].mean()) < gx+cell and gy <= float(m["pts"][:,1].mean()) < gy+cell]
        if len(items)>=4: counts.append((len(items),gx,gy))
counts.sort(reverse=True)
regions=[]
for n,gx,gy in counts:
    r=(max(0,gx-80),max(0,gy-80),min(w,gx+cell+80),min(h,gy+cell+80))
    if all(abs((r[0]+r[2])-(q[0]+q[2]))>700 or abs((r[1]+r[3])-(q[1]+q[3]))>700 for q in regions): regions.append(r)
    if len(regions)>=4: break

overview=image.copy()
for d in direct: draw_poly(overview,d["pts"],(255,120,0),None,3,0.7)
for m in missed: draw_poly(overview,m["pts"],(0,0,255),None,5,0.8)
for i,(x1,y1,x2,y2) in enumerate(regions,1):
    cv2.rectangle(overview,(x1,y1),(x2,y2),(0,215,255),10)
    cv2.putText(overview,f"Region {i}",(x1+12,y1+48),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,215,255),4,cv2.LINE_AA)
cv2.putText(overview,f"SAHI reference: {len(gt)} | Direct exp63-960: {len(direct)} | Missed: {len(missed)}",(35,65),cv2.FONT_HERSHEY_SIMPLEX,1.35,(0,0,255),4,cv2.LINE_AA)
cv2.imwrite(os.path.join(OUT,"overview.png"),overview,[cv2.IMWRITE_PNG_COMPRESSION,1])

for i,(x1,y1,x2,y2) in enumerate(regions,1):
    crop=image[y1:y2,x1:x2].copy()
    for d in direct:
        cx,cy=d["pts"][:,0].mean(),d["pts"][:,1].mean()
        if x1<=cx<x2 and y1<=cy<y2: draw_poly(crop,d["pts"]-[x1,y1],(255,120,0),f"D {d['conf']:.2f}",6,1.05)
    for m in missed:
        cx,cy=m["pts"][:,0].mean(),m["pts"][:,1].mean()
        if x1<=cx<x2 and y1<=cy<y2: draw_poly(crop,m["pts"]-[x1,y1],(0,0,255),f"M {m['conf']:.2f}",7,1.1)
    cv2.rectangle(crop,(8,8),(min(crop.shape[1]-8,900),62),(255,255,255),-1)
    cv2.putText(crop,f"Region {i}: red = missed by direct exp63-960",(22,48),cv2.FONT_HERSHEY_SIMPLEX,0.95,(0,0,180),3,cv2.LINE_AA)
    crop=cv2.resize(crop,None,fx=1.6,fy=1.6,interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(OUT,f"region_{i}_zoom.png"),crop,[cv2.IMWRITE_PNG_COMPRESSION,1])

    # Side-by-side publication panel: SAHI reference vs direct detection.
    left=image[y1:y2,x1:x2].copy(); right=left.copy()
    for g in gt:
        cx,cy=g["pts"][:,0].mean(),g["pts"][:,1].mean()
        if x1<=cx<x2 and y1<=cy<y2:
            draw_poly(left,g["pts"]-[x1,y1],(0,170,0),f"S {g['conf']:.2f}",5,0.85)
    for d in direct:
        cx,cy=d["pts"][:,0].mean(),d["pts"][:,1].mean()
        if x1<=cx<x2 and y1<=cy<y2:
            draw_poly(right,d["pts"]-[x1,y1],(255,120,0),f"D {d['conf']:.2f}",5,0.85)
    for m in missed:
        cx,cy=m["pts"][:,0].mean(),m["pts"][:,1].mean()
        if x1<=cx<x2 and y1<=cy<y2:
            draw_poly(right,m["pts"]-[x1,y1],(0,0,255),None,6,0.8)
    cv2.putText(left,"SAHI reference",(20,42),cv2.FONT_HERSHEY_SIMPLEX,1.15,(0,120,0),3,cv2.LINE_AA)
    cv2.putText(right,"Direct exp63-960 | red = missed",(20,42),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,200),3,cv2.LINE_AA)
    panel=cv2.hconcat([left,right])
    panel=cv2.resize(panel,None,fx=1.25,fy=1.25,interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(OUT,f"region_{i}_side_by_side.png"),panel,[cv2.IMWRITE_PNG_COMPRESSION,1])

print(f"reference={len(gt)}, direct={len(direct)}, matched={matched}, missed={len(missed)}")
print("regions", regions)
print(OUT)
