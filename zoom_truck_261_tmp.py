import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import numpy as np

VIDEO_PATH = r"E:\DJI\cvat\0111_D_stabilized.avi"
cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_POS_FRAMES, 261-1)
ret, frame = cap.read()

# 三个候选框：两个"完整"(0.902/0.907) + 一个截断碎片(0.517)
boxes = [
    ("960:truck 0.902", (1290.0,387.3), (89.8,39.8), 0, (0,255,255)),
    ("1280:truck 0.907(最终保留)", (1289.9,387.2), (89.8,39.7), 0, (0,255,0)),
    ("1280:truck 0.517(碎片)", (1267.1,378.7), (40.8,38.3), 0, (0,0,255)),
]

cx0, cy0 = 1290, 385
half = 120
x1,y1,x2,y2 = int(cx0-half), int(cy0-half), int(cx0+half), int(cy0+half)
crop = frame[y1:y2, x1:x2].copy()
scale = 6
crop_big = cv2.resize(crop, (crop.shape[1]*scale, crop.shape[0]*scale), interpolation=cv2.INTER_NEAREST)

for name, center, wh, ang, color in boxes:
    cx, cy = center
    ccx, ccy = (cx - x1)*scale, (cy - y1)*scale
    w, h = wh[0]*scale, wh[1]*scale
    rect = ((ccx,ccy),(w,h),ang)
    pts = np.intp(cv2.boxPoints(rect))
    cv2.polylines(crop_big, [pts], True, color, 3)

cv2.imwrite(r"D:\yolo26-tracking\outputs\yolo26-obb_20260721_105049\idsw_diagnosis\frame261_truck_zoom.jpg", crop_big)
print("saved")
