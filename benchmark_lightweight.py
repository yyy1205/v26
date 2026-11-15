import os
import time

os.environ.setdefault("YOLO_CONFIG_DIR", r"D:\yolo26-tracking\.tmp-ultralytics-config")

import numpy as np
import torch
from ultralytics import YOLO

MODELS = {
    "baseline_yolo26s": r"D:\yolo26-tracking\runs\obb\exp_yolo26-3\weights\best.pt",
    "mobilenetv4": r"D:\yolo26-tracking\runs\obb\exp_yolo26_mobilenetv4\weights\best.pt",
}
WARMUP = 20
ITERS = 50


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    for imgsz in (640, 960, 1280):
        image = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        print(f"imgsz={imgsz}")
        for name, path in MODELS.items():
            model = YOLO(path)
            model.model.info(imgsz=imgsz, verbose=True)
            for _ in range(WARMUP):
                model.predict(image, imgsz=imgsz, device=device, verbose=False)
            sync()
            start = time.perf_counter()
            for _ in range(ITERS):
                model.predict(image, imgsz=imgsz, device=device, verbose=False)
            sync()
            elapsed = time.perf_counter() - start
            ms = elapsed * 1000 / ITERS
            print(f"{name}: params={sum(p.numel() for p in model.model.parameters()):,} latency={ms:.2f} ms inference_fps={1000/ms:.2f}")


if __name__ == "__main__":
    main()
