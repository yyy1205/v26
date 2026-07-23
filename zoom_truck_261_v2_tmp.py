import sys
sys.path.insert(0, r"D:\yolo26-tracking")
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
import cv2
import numpy as np
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
cap.set(cv2.CAP_PROP_POS_FRAMES, 261-1)
ret, frame = cap.read()
roi_mask = multi_roi_mask(roi_polys_modes, frame.shape)
proc = apply_clahe(frame); proc = gamma_transform(proc,1.2)
proc = apply_blurred_background_fill(proc, roi_mask, 51, 101)

CLASS_NAMES=["car","truck","bus"]
found = []
for sh, target_slice in [(960,(672,0,1632,960)), (1280,(896,0,2176,1280)), (1280,(0,0,1280,1280))]:
    x1,y1,x2,y2 = target_slice
    if roi_mask[y1:y2,x1:x2].max()==0: continue
    crop = proc[y1:y2,x1:x2]
    res = model.predict(crop, imgsz=sh, conf=0.3, verbose=False)
    if not res or res[0].obb is None: continue
    obb = res[0].obb
    xywhr = obb.xywhr.cpu().numpy().copy()
    xywhr[:,0]+=x1; xywhr[:,1]+=y1
    confs = obb.conf.cpu().numpy(); clses = obb.cls.cpu().numpy().astype(int)
    for i in range(len(xywhr)):
        cx,cy,w,h,r = xywhr[i]
        if clses[i]==1 and abs(cx-1290)<60 and abs(cy-387)<60:
            found.append((sh, target_slice, CLASS_NAMES[clses[i]], float(confs[i]), xywhr[i].copy()))
            print(f"scale={sh} slice={target_slice} conf={confs[i]:.3f} xywhr={xywhr[i]} angle_deg={np.rad2deg(r):.1f}")

# 画图：用真实角度
cx0, cy0 = 1290, 385
half = 120
x1,y1,x2,y2 = int(cx0-half), int(cy0-half), int(cx0+half), int(cy0+half)
crop = frame[y1:y2, x1:x2].copy()
scale = 6
crop_big = cv2.resize(crop, (crop.shape[1]*scale, crop.shape[0]*scale), interpolation=cv2.INTER_NEAREST)

colors = [(0,255,255),(0,255,0),(0,0,255)]
for idx, (sh, sl, cname, conf, xywhr) in enumerate(found):
    cx,cy,w,h,r = xywhr
    ccx, ccy = (cx - x1)*scale, (cy - y1)*scale
    rect = ((ccx,ccy),(w*scale,h*scale), float(np.rad2deg(r)))
    pts = np.intp(cv2.boxPoints(rect))
    color = colors[idx % len(colors)]
    cv2.polylines(crop_big, [pts], True, color, 3)
    cv2.putText(crop_big, f"{sh}:{conf:.3f}", (int(ccx-40), int(ccy-h*scale/2-15-idx*30)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

cv2.imwrite(r"D:\yolo26-tracking\outputs\yolo26-obb_20260721_105049\idsw_diagnosis\frame261_truck_zoom_v2.jpg", crop_big)
print("saved")
