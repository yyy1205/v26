"""Batch-one FP32 predict latency on identical validation images; no training."""
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / '.tmp-ultralytics-config'))

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO
from compare_trained_models import RUNS


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this GPU comparison')
    cfg = yaml.safe_load((ROOT / 'obb.yaml').read_text(encoding='utf-8'))
    source = Path(cfg['path']) / cfg['val']
    paths = sorted(p for p in source.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})[:8]
    images = [cv2.imread(str(p)) for p in paths]
    if not images or any(im is None for im in images):
        raise RuntimeError('Unable to read validation images')
    models = {name: YOLO(str(ROOT / 'runs' / 'obb' / run / 'weights' / 'best.pt')) for name, run in RUNS.items()}
    result = dict(device=torch.cuda.get_device_name(0), precision='FP32', batch=1,
                  sources=[str(p) for p in paths], warmup=20, rounds=3, iterations_per_round=40,
                  note='Images resized to square to fix actual tensor size. Includes predict preprocessing, inference, postprocessing and Python overhead; excludes file decoding and tracking.', results=[])
    for size in (640, 960, 1280):
        inputs = [cv2.resize(im, (size, size)) for im in images]
        kwargs = dict(imgsz=size, device=0, half=False, conf=0.3, verbose=False)
        for model in models.values():
            for i in range(20):
                model.predict(inputs[i % len(inputs)], **kwargs)
        samples = {name: [] for name in models}
        core = {name: [] for name in models}
        for round_index in range(3):
            # Rotate ordering to reduce fixed-order bias.
            names = list(models)
            names = names[round_index:] + names[:round_index]
            for name in names:
                model = models[name]
                for i in range(40):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    output = model.predict(inputs[i % len(inputs)], **kwargs)
                    torch.cuda.synchronize()
                    samples[name].append((time.perf_counter() - start) * 1000)
                    core[name].append(output[0].speed['inference'])
        for name in models:
            values = samples[name]
            row = dict(model=name, imgsz=size, mean_ms=float(np.mean(values)),
                       median_ms=float(np.median(values)), p95_ms=float(np.percentile(values, 95)),
                       inference_only_mean_ms=float(np.mean(core[name])), fps=1000 / float(np.mean(values)))
            result['results'].append(row)
            print(json.dumps(row), flush=True)
    out = ROOT / 'outputs' / 'lightweight_comparison'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'speed.json').write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
