import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(PROJECT_ROOT, ".tmp-ultralytics-config"))
ALT_ULTRALYTICS_ROOT = os.environ.get("ULTRALYTICS_ROOT")
if ALT_ULTRALYTICS_ROOT and ALT_ULTRALYTICS_ROOT not in sys.path:
    sys.path.insert(0, ALT_ULTRALYTICS_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.prediction import ObjectPrediction
from sahi.utils.cv import visualize_object_predictions


def polygon_iou(a, b):
    a = cv2.convexHull(a.astype("float32"))
    b = cv2.convexHull(b.astype("float32"))
    aa, ab = abs(cv2.contourArea(a)), abs(cv2.contourArea(b))
    if aa <= 0 or ab <= 0:
        return 0.0
    inter, _ = cv2.intersectConvexConvex(a, b)
    return float(inter / (aa + ab - inter)) if aa + ab - inter > 0 else 0.0


def nms_cross_class(predictions, threshold=0.5):
    kept = []
    for item in sorted(predictions, key=lambda x: x["confidence"], reverse=True):
        if all(polygon_iou(item["points"], old["points"]) <= threshold for old in kept):
            kept.append(item)
    return kept


def main():
    if len(sys.argv) != 4:
        print("用法: python run_one_sahi_nms.py <img> <model> <output_dir>")
        return 1
    img_path, model_path, output_dir = sys.argv[1:]
    os.makedirs(output_dir, exist_ok=True)
    image = cv2.imread(img_path)
    if image is None:
        return 2

    detector = AutoDetectionModel.from_pretrained(
        model_type="ultralytics", model_path=model_path,
        confidence_threshold=0.7, device="cuda:0"
    )
    result = get_sliced_prediction(
        image, detector, slice_height=640, slice_width=640,
        overlap_height_ratio=0.2, overlap_width_ratio=0.2
    )
    raw = []
    for obj in result.object_prediction_list:
        if obj.mask is None or not obj.mask.segmentation:
            continue
        points = np.asarray(obj.mask.segmentation[0], dtype="float32").reshape(4, 2)
        cid = int(obj.category.id)
        raw.append({"class_id": cid, "class_name": str(obj.category.name),
                    "confidence": float(obj.score.value), "points": points})
    kept = nms_cross_class(raw)
    sahi_predictions = []
    for p in kept:
        x1, y1 = p["points"].min(axis=0)
        x2, y2 = p["points"].max(axis=0)
        sahi_predictions.append(ObjectPrediction(
            bbox=[int(x1), int(y1), int(x2), int(y2)],
            category_id=p["class_id"], category_name=p["class_name"],
            score=p["confidence"], segmentation=[p["points"].reshape(-1).tolist()],
            full_shape=list(image.shape[:2])))
    vis = visualize_object_predictions(
        image=image, object_prediction_list=sahi_predictions,
        rect_th=2, text_size=0.6, text_th=2, color=None)["image"]
    out_path = os.path.join(output_dir, os.path.splitext(os.path.basename(img_path))[0] + "_sahi_nms.png")
    cv2.imwrite(out_path, vis)
    print(f"SAHI检测: 原始框 {len(raw)} -> 跨类别NMS后 {len(kept)} | {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
