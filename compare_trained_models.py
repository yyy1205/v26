"""Compare completed runs without starting training or changing checkpoints."""
import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / '.tmp-ultralytics-config'))

from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops

RUNS = {
    'YOLO26s-OBB': 'exp_yolo26s_baseline_200-2',
    'MobileNetV4-OBB': 'exp_yolo26_mobilenetv4_200',
    'MobileNetV4-DWConv-OBB': 'exp_yolo26_mobilenetv4_dwdown',
}


def main():
    results = []
    for name, run in RUNS.items():
        directory = ROOT / 'runs' / 'obb' / run
        with (directory / 'results.csv').open(encoding='utf-8-sig', newline='') as handle:
            rows = [{k.strip(): v for k, v in row.items()} for row in csv.DictReader(handle)]
        best = max(rows, key=lambda row: float(row['metrics/mAP50-95(B)']))
        weights = directory / 'weights' / 'best.pt'
        model = YOLO(str(weights)).model
        record = dict(
            name=name, run=run, weights=str(weights),
            completed_epoch=int(rows[-1]['epoch']), best_epoch=int(best['epoch']),
            precision=float(best['metrics/precision(B)']),
            recall=float(best['metrics/recall(B)']),
            mAP50=float(best['metrics/mAP50(B)']),
            mAP50_95=float(best['metrics/mAP50-95(B)']),
            parameters=sum(p.numel() for p in model.parameters()),
            GFLOPs_640=get_flops(model, imgsz=640),
            checkpoint_MB=weights.stat().st_size / 1e6,
        )
        results.append(record)
        print(json.dumps(record, ensure_ascii=False))
    output = ROOT / 'outputs' / 'lightweight_comparison'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'comparison.json').write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
    lines = ['# Trained model comparison', '',
             'Metrics are from the single epoch with highest validation mAP50-95 in each results.csv; no new validation was run.',
             'All three runs completed 200 epochs. GFLOPs use 640x640 input and unfused loaded checkpoints.', '',
             '| Model | Best epoch | Precision % | Recall % | mAP50 % | mAP50-95 % | Parameters M | GFLOPs | File MB |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in results:
        lines.append(f"| {r['name']} | {r['best_epoch']} | {r['precision']*100:.3f} | {r['recall']*100:.3f} | {r['mAP50']*100:.3f} | {r['mAP50_95']*100:.3f} | {r['parameters']/1e6:.3f} | {r['GFLOPs_640']:.3f} | {r['checkpoint_MB']:.3f} |")
    lines += ['', 'Checkpoint size includes serialized metadata; it is not runtime memory usage.',
              'Training configs reference the same data YAML, but historical dataset identity is not verified.',
              'The newly configured scratch baseline is not included: no completed run was found for it.']
    (output / 'comparison.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
