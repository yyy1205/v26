import os
import subprocess
import shutil

ROOT = r"D:\yolo26-tracking"
PYTHON = r"D:\anaconda\envs\YOLOv8-Deepsort\python.exe"
IMG = r"F:\vis\1.jpg"
MODEL = os.path.join(ROOT, "runs", "obb", "exp_yolo26-3", "weights", "best.pt")
OUT = os.path.join(ROOT, "outputs", "compare_sahi_direct")
SAHI_OUT = os.path.join(OUT, "sahi_nms")
DIRECT_OUT = os.path.join(OUT, "direct")
os.makedirs(SAHI_OUT, exist_ok=True)
os.makedirs(DIRECT_OUT, exist_ok=True)

subprocess.check_call([PYTHON, os.path.join(ROOT, "cheshi", "run_one_sahi_nms.py"), IMG, MODEL, SAHI_OUT])
subprocess.check_call([PYTHON, os.path.join(ROOT, "cheshi", "run_one_filtered.py"), IMG, MODEL, DIRECT_OUT])

# 直接检测脚本的默认文件名改为带有明确方法名，避免与 SAHI 结果混淆。
direct_src = os.path.join(DIRECT_OUT, "1.jpg")
direct_dst = os.path.join(DIRECT_OUT, "1_direct.jpg")
if os.path.exists(direct_src):
    os.replace(direct_src, direct_dst)
sahi_src = os.path.join(SAHI_OUT, "1_sahi_nms.jpg")
root_sahi = os.path.join(OUT, "1_sahi_nms.jpg")
root_direct = os.path.join(OUT, "1_direct.jpg")
if os.path.exists(sahi_src):
    shutil.copy2(sahi_src, root_sahi)
if os.path.exists(direct_dst):
    shutil.copy2(direct_dst, root_direct)
print(f"SAHI结果: {root_sahi}")
print(f"直接检测结果: {root_direct}")
