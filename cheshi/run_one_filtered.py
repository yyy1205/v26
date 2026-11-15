import os
import sys

# Allow this script to use the repository-local ultralytics package.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(PROJECT_ROOT, ".tmp-ultralytics-config"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
from sahi.prediction import ObjectPrediction
from sahi.utils.cv import visualize_object_predictions
from ultralytics import YOLO

IOU_THRESHOLD = 0.5


def polygon_iou(points_a, points_b):
    """计算两个 OBB 四边形的 IoU。points shape=(4,2)。"""
    a = cv2.convexHull(points_a.astype("float32"))
    b = cv2.convexHull(points_b.astype("float32"))
    area_a = abs(cv2.contourArea(a))
    area_b = abs(cv2.contourArea(b))
    if area_a <= 0 or area_b <= 0:
        return 0.0
    inter_area, _ = cv2.intersectConvexConvex(a, b)
    union = area_a + area_b - inter_area
    return float(inter_area / union) if union > 0 else 0.0


def write_predictions(path, predictions):
    """TXT 格式：class_id confidence x1 y1 x2 y2 x3 y3 x4 y4 class_name。"""
    with open(path, "w", encoding="utf-8") as f:
        for pred in predictions:
            coords = pred["points"].reshape(-1).tolist()
            coord_text = "\t".join(f"{v:.2f}" for v in coords)
            f.write(f"{pred['class_id']}\t{pred['confidence']:.6f}\t{coord_text}\t{pred['class_name']}\n")


def read_predictions(path):
    predictions = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 11:
                print(f"警告：跳过格式错误的 OBB TXT 行 {path}:{line_no}")
                continue
            try:
                predictions.append({
                    "class_id": int(parts[0]),
                    "confidence": float(parts[1]),
                    "points": __import__("numpy").array([float(v) for v in parts[2:10]], dtype="float32").reshape(4, 2),
                    "class_name": "\t".join(parts[10:]),
                })
            except ValueError:
                print(f"警告：跳过无法解析的 OBB TXT 行 {path}:{line_no}")
    return predictions


def suppress_overlaps(predictions, iou_threshold=IOU_THRESHOLD):
    """类无关 OBB NMS：旋转框重叠时保留置信度较大的结果。"""
    ordered = sorted(predictions, key=lambda item: item["confidence"], reverse=True)
    kept = []
    for candidate in ordered:
        if all(polygon_iou(candidate["points"], selected["points"]) <= iou_threshold for selected in kept):
            kept.append(candidate)
    return kept


def to_sahi_prediction(pred, image_shape):
    """将 TXT 中的 OBB 恢复为 SAHI ObjectPrediction，以便使用 SAHI OBB 可视化。"""
    points = pred["points"]
    x1, y1 = points.min(axis=0)
    x2, y2 = points.max(axis=0)
    return ObjectPrediction(
        bbox=[int(x1), int(y1), int(x2), int(y2)],
        category_id=pred["class_id"],
        category_name=pred["class_name"],
        score=pred["confidence"],
        segmentation=[points.reshape(-1).tolist()],
        full_shape=list(image_shape[:2]),
    )


def main():
    if len(sys.argv) < 4 or len(sys.argv) > 7:
        print("用法: python run_one_filtered.py <img_path> <model_path> <output_dir> [imgsz] [suffix] [no_nms]")
        return 1

    img_path, model_path, output_dir = sys.argv[1:4]
    imgsz = int(sys.argv[4]) if len(sys.argv) >= 5 else 2048
    suffix = sys.argv[5] if len(sys.argv) >= 6 else ""
    disable_nms = len(sys.argv) >= 7 and sys.argv[6].lower() == "no_nms"
    os.makedirs(output_dir, exist_ok=True)
    raw_txt_dir = os.path.join(output_dir, "labels_raw")
    filtered_txt_dir = os.path.join(output_dir, "labels")
    visualized_dir = os.path.join(output_dir, "visualized")
    os.makedirs(raw_txt_dir, exist_ok=True)
    os.makedirs(filtered_txt_dir, exist_ok=True)
    os.makedirs(visualized_dir, exist_ok=True)

    try:
        image = cv2.imread(img_path)
        if image is None:
            print(f"无法读取图片: {img_path}")
            return 2

        # 直接使用 Ultralytics 对整张原图推理；SAHI 仅用于后面的可视化。
        model = YOLO(model_path)
        # 原图中目标较小，使用较大的整图输入尺寸提升直接推理的召回率。
        result = model.predict(source=image, conf=0.7, imgsz=imgsz, device=0, verbose=False)[0]
        predictions = []
        obb = getattr(result, "obb", None)
        if obb is None or obb.xyxyxyxy is None:
            print("警告：模型没有返回 OBB 结果")
            obb_points = []
        else:
            obb_points = obb.xyxyxyxy.cpu().numpy()
        names = result.names
        for idx, points_array in enumerate(obb_points):
            points = __import__("numpy").array(points_array, dtype="float32").reshape(4, 2)
            class_id = int(obb.cls[idx].item())
            predictions.append({
                "class_id": class_id,
                "class_name": str(names[class_id]),
                "confidence": float(obb.conf[idx].item()),
                "points": points,
            })

        stem = os.path.splitext(os.path.basename(img_path))[0]
        output_stem = stem + suffix
        raw_txt_path = os.path.join(raw_txt_dir, output_stem + ".txt")
        filtered_txt_path = os.path.join(filtered_txt_dir, output_stem + ".txt")
        write_predictions(raw_txt_path, predictions)
        filtered_predictions = read_predictions(raw_txt_path)
        if not disable_nms:
            filtered_predictions = suppress_overlaps(filtered_predictions)
        write_predictions(filtered_txt_path, filtered_predictions)

        # 用 SAHI 的 OBB 可视化接口绘制筛选后的四点旋转框。
        sahi_predictions = [to_sahi_prediction(pred, image.shape) for pred in filtered_predictions]
        vis_img = visualize_object_predictions(
            image=image,
            object_prediction_list=sahi_predictions,
            rect_th=2,
            text_size=0.6,
            text_th=2,
            color=None,
        )["image"]
        out_path = os.path.join(output_dir, output_stem + ".png")
        visualized_path = os.path.join(visualized_dir, output_stem + ".png")
        cv2.imwrite(out_path, vis_img)
        cv2.imwrite(visualized_path, vis_img)
        print(f"完成: {os.path.basename(img_path)} | 原始框 {len(predictions)} -> 保留框 {len(filtered_predictions)} | TXT: {filtered_txt_path} | 图片: {visualized_path}")
        return 0
    except Exception as e:
        print(f"处理图片出错: {img_path} | {type(e).__name__}: {e}")
        return 3


if __name__ == "__main__":
    sys.exit(main())



