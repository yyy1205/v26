import os

os.environ.setdefault("YOLO_CONFIG_DIR", r"D:\yolo26-tracking\.tmp-ultralytics-config")

from ultralytics import YOLO


if __name__ == "__main__":
    model = YOLO(r"ultralytics/cfg/models/26/yolo26-mobilenetv4-obb.yaml")
    model.train(
        data=r"D:\yolo26-tracking\obb.yaml",
        epochs=200,
        batch=16,
        imgsz=640,
        optimizer="SGD",
        lr0=0.01,
        patience=200,
        close_mosaic=15,
        mixup=0.1,
        project=r"D:\yolo26-tracking\runs\obb",
        name="exp_yolo26_mobilenetv4_200",
        device=0,
        workers=4,
        amp=True,
        seed=0,
    )
