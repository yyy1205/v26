import sys
import os
import cv2
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import visualize_object_predictions

if len(sys.argv) != 4:
    print("用法: python run_one.py <img_path> <model_path> <output_dir>")
    sys.exit(1)

img_path = sys.argv[1]
model_path = sys.argv[2]
output_dir = sys.argv[3]

os.makedirs(output_dir, exist_ok=True)

try:
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=model_path,
        confidence_threshold=0.7,
        device="cuda:0"  # 可以改成 "cuda:0" 如显卡环境无异常
    )
    img = cv2.imread(img_path)
    if img is None:
        print(f"无法读取图片: {img_path}")
        sys.exit(2)
    sahi_result = get_sliced_prediction(
        img, detection_model,
        slice_height=640, slice_width=640,
        overlap_height_ratio=0.2, overlap_width_ratio=0.2
    )
    vis_img = visualize_object_predictions(
        image=img,
        object_prediction_list=sahi_result.object_prediction_list,
        rect_th=2,
        text_size=0.6,
        text_th=2,
        color=None,
    )["image"]
    out_path = os.path.join(output_dir, os.path.basename(img_path))
    cv2.imwrite(out_path, vis_img)
    print(f"已保存: {out_path}")
except Exception as e:
    print(f"处理图片出错: {img_path}，错误: {e}")
    sys.exit(3)