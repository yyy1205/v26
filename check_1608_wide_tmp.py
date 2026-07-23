import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import numpy as np
import json
from ultralytics import YOLO

VIDEO_PATH = r"E:\DJI\cvat\0111_D_stabilized.avi"
MODEL_PATH = r"D:\yolo26-tracking\runs\obb\exp_yolo26-3\weights\best.pt"
ROI_FILE = r"E:\DJI\cvat\0111_D_stabilized_roi.json"

def apply_clahe(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl,a,b)), cv2.COLOR_LAB2BGR)

def gamma_transform(image, gamma=1.2):
    lut = np.empty((1,256), np.uint8)
    for i in range(256): lut[0,i]=np.clip(pow(i/255.0,gamma)*255.0,0,255)
    return cv2.LUT(image, lut)

def apply_blurred_background_fill(frame, roi_mask, feather=51, blur_ksize=101):
    h,w = roi_mask.shape[:2]; min_dim=min(h,w)
    bk=int(blur_ksize); bk += (bk%2==0); bk=min(bk, min_dim-1 if (min_dim-1)%2==1 else min_dim-2); bk=max(3,bk)
    blurred = cv2.GaussianBlur(frame,(bk,bk),0)
    fk=int(feather); fk += (fk%2==0); fk=min(fk, min_dim-1 if (min_dim-1)%2==1 else min_dim-2); fk=max(3,fk)
    alpha=(roi_mask.astype(np.float32)/255.0); alpha=cv2.GaussianBlur(alpha,(fk,fk),0)[...,None]
    return (frame.astype(np.float32)*alpha + blurred.astype(np.float32)*(1.0-alpha)).astype(np.uint8)

def multi_roi_mask(polys_with_mode, shape):
    h,w = shape[:2]; mask=np.zeros((h,w),dtype=np.uint8)
    for pts,mode in polys_with_mode:
        if mode==1: cv2.fillPoly(mask,[np.array(pts,dtype=np.int32)],255)
    for pts,mode in polys_with_mode:
        if mode==2: cv2.fillPoly(mask,[np.array(pts,dtype=np.int32)],0)
    return mask

def get_slice_bboxes(H,W,sh,sw,oh=0.3,ow=0.3):
    y_overlap=int(oh*sh); x_overlap=int(ow*sw); bboxes=[]; y_min=0
    while y_min<H:
        y_max=min(H,y_min+sh); x_min=0
        while x_min<W:
            x_max=min(W,x_min+sw); bboxes.append((max(0,x_max-sw),max(0,y_max-sh),x_max,y_max))
            if x_max==W: break
            x_min=x_max-x_overlap
        if y_max==H: break
        y_min=y_max-y_overlap
    return bboxes

with open(ROI_FILE,encoding="utf-8") as f: roi_data=json.load(f)
items = roi_data if isinstance(roi_data,list) else roi_data["roi_polys_modes"]
roi_polys_modes=[([tuple(p) for p in it["pts"]], it["mode"]) for it in items]

model = YOLO(MODEL_PATH); model.to("cuda:0")
cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_POS_FRAMES, 1608-1)
ret, frame = cap.read()
roi_mask = multi_roi_mask(roi_polys_modes, frame.shape)
proc = apply_clahe(frame); proc = gamma_transform(proc,1.2)
proc = apply_blurred_background_fill(proc, roi_mask, 51, 101)

CLASS_NAMES=["car","truck","bus"]
BUS_X, BUS_Y = 952.5, 1623.0
WIDE_RADIUS = 400  # 扩大搜索半径，以bus本身位置为中心，不再依赖之前的car事件位置

all_dets = []
for sh in (960,1280):
    slices = get_slice_bboxes(frame.shape[0], frame.shape[1], sh, sh, 0.3, 0.3)
    for (x1,y1,x2,y2) in slices:
        if roi_mask[y1:y2,x1:x2].max()==0: continue
        crop = proc[y1:y2,x1:x2]
        res = model.predict(crop, imgsz=sh, conf=0.3, verbose=False)
        if not res or res[0].obb is None or len(res[0].obb)==0: continue
        obb = res[0].obb
        xywhr = obb.xywhr.cpu().numpy().copy()
        xywhr[:,0]+=x1; xywhr[:,1]+=y1
        confs = obb.conf.cpu().numpy(); clses = obb.cls.cpu().numpy().astype(int)
        for i in range(len(xywhr)):
            all_dets.append({"xywhr":xywhr[i],"conf":float(confs[i]),"cls":int(clses[i]),"scale":sh,"slice":(x1,y1,x2,y2)})

near_bus = [d for d in all_dets if d["cls"]==2 and abs(d["xywhr"][0]-BUS_X)<WIDE_RADIUS and abs(d["xywhr"][1]-BUS_Y)<WIDE_RADIUS]
print(f"以bus本身位置({BUS_X},{BUS_Y})为中心，半径{WIDE_RADIUS}px内，所有bus类检测(不管切片) 共{len(near_bus)}个:")
for d in near_bus:
    cx,cy,w,h,r = d["xywhr"]
    print(f"  scale={d['scale']} slice={d['slice']} conf={d['conf']:.3f} center=({cx:.1f},{cy:.1f}) w={w:.1f} h={h:.1f}")

# 同时打印所有class的检测，看bus附近有没有被误分类成别的类的疑似"另一半"
near_any = [d for d in all_dets if abs(d["xywhr"][0]-BUS_X)<WIDE_RADIUS and abs(d["xywhr"][1]-BUS_Y)<WIDE_RADIUS]
print(f"\n同半径内所有类别检测 共{len(near_any)}个:")
for d in near_any:
    cx,cy,w,h,r = d["xywhr"]
    print(f"  scale={d['scale']} cls={CLASS_NAMES[d['cls']]:<5} slice={d['slice']} conf={d['conf']:.3f} center=({cx:.1f},{cy:.1f}) w={w:.1f} h={h:.1f}")
