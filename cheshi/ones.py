import glob
import os
import subprocess
import sys

INPUT_DIR = r"F:\vis"
OUTPUT_DIR = r"D:\yolo26-tracking\outputs\tuvis-txt"
MODEL_PATH = r"D:\yolo26-tracking\runs\obb\exp_yolo26-3\weights\best.pt"

os.makedirs(OUTPUT_DIR, exist_ok=True)
RUNNER = os.path.join(os.path.dirname(__file__), "run_one_filtered.py")
img_files = []
for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
    img_files.extend(glob.glob(os.path.join(INPUT_DIR, ext)))
img_files.sort()

print(f"本次共需处理 {len(img_files)} 张图片。")

for idx, img_path in enumerate(img_files):
    print(f"\n====== 处理 {idx+1}/{len(img_files)}: {img_path}")
    # 调用run_one.py并传递必要参数
    python_exe = r"D:\anaconda\envs\YOLOv8-Deepsort\python.exe"
    if not os.path.exists(python_exe):
        python_exe = sys.executable
    ret = subprocess.call([python_exe, RUNNER, img_path, MODEL_PATH, OUTPUT_DIR])
    if ret != 0:
        print(f"处理失败（返回码{ret}），请检查: {img_path}")

print("\n全部处理结束！输出目录为：", OUTPUT_DIR)
