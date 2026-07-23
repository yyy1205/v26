from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo26s-obb.yaml")  # YOLO26 small OBB 架构
    model.train(
        data=r"D:\yolo26-tracking\obb.yaml",
        epochs=200,
        batch=16,
        imgsz=640,
        optimizer="SGD",
        lr0=0.01,
        patience=30,
        close_mosaic=15,
        mixup=0.1,
        name="exp_yolo26",
    )
