import cv2
import numpy as np
import glob
import os

# ================= 配置区 =================
# 你的文件夹路径
folder_path = r'D:\yolo26-tracking\biaoding\frames_output10.22.4k3-1\4k3'
# 棋盘格内角点数 (横向11，纵向8)
CHECKERBOARD = (11, 8)
# ==========================================

# 检查路径是否存在
if not os.path.exists(folder_path):
    print(f"❌ 严重错误：电脑上找不到这个文件夹：{folder_path}")
    exit()

# 搜刮所有格式的图片
image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG', '*.bmp']
images = []
for ext in image_extensions:
    images.extend(glob.glob(os.path.join(folder_path, ext)))

print(f"📂 在文件夹中找到了 {len(images)} 张图片。")

if len(images) == 0:
    print("❌ 文件夹里没有找到图片！")
    exit()

# 参数设置
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

objpoints = []
imgpoints = []
image_names = []

# 提取角点
print("\n开始提取角点...")
for fname in images:
    img = cv2.imdecode(np.fromfile(fname, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        image_names.append(fname)
        print(f"✅ 成功: {os.path.basename(fname)}")
    else:
        print(f"⚠️ 失败(未找到角点): {os.path.basename(fname)}")

if len(objpoints) < 5:
    print("\n❌ 成功识别的图片少于 5 张，无法标定！")
    exit()

# 计算内参
print("\n正在计算相机内参...")
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

print("\n================ 标定结果 ================")
print(f"全局 RMS 误差: {ret:.4f} pixels")
print("内参矩阵 K:\n", mtx)
print("畸变系数 dist:\n", dist.ravel())

# ================= 修复报错的核心部分 =================
print("\n==== 正在计算每张图片的误差 ====")
errors = []
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)

    # 【修复】：使用 numpy 计算误差，并将数组展平(ravel)避免形状不匹配
    error = np.linalg.norm(imgpoints[i].ravel() - imgpoints2.ravel()) / len(imgpoints2)
    errors.append(error)

# 排序并输出结果
results = sorted(zip(errors, image_names), reverse=True)

print("\n==== 误差排名前 5 的图片 (误差越大越差) ====")
for err, name in results[:5]:
    print(f"误差: {err:.4f} 像素 | 图片: {os.path.basename(name)}")

# 自动生成剔除建议
threshold = 1.0  # 如果单张误差大于 1.0，说明这张图有问题
bad_images = [name for err, name in results if err > threshold]

print(f"\n==== 建议剔除的图片 (单张误差 > {threshold} 像素) ====")
if len(bad_images) == 0:
    print("所有图片的个体误差都在阈值内，没有特别烂的图！")
else:
    for name in bad_images:
        print(os.path.basename(name))