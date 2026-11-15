"""从视频抽帧并保存原图。不用命令行，直接改下方【配置区】后运行:
    python video_to_frames.py
结果自动保存在本脚本所在目录下新建的文件夹里。
"""
import csv
from pathlib import Path

import cv2

# ==================== 配置区（改这里） ====================
# 输入：可以是单个视频文件，也可以是包含视频的文件夹
VIDEO_INPUT = r"F:\ybw\jibian\4k3.MP4"

# 输出：在“本脚本所在目录”下新建的文件夹名
OUTPUT_FOLDER_NAME = "frames_output10.22.4k3-1"

# 抽帧方式：三选一
#   "every_n"  每 N 帧保存一张
#   "interval" 每 X 秒保存一张
#   "num"      整个视频均匀抽 N 张
SAMPLE_MODE = "every_n"
SAMPLE_VALUE = 15          # 对应上面的数值

START_SEC = 0.0             # 起始时间（秒）
END_SEC = None              # 结束时间（秒），None = 到视频结尾
MAX_PER_VIDEO = 0           # 每个视频最多保存张数，0 = 不限

IMAGE_EXT = "png"           # "png" 无损（推荐） / "jpg" 有损
JPG_QUALITY = 100           # 仅 jpg 有效，1~100
MIN_SHARPNESS = 0.0         # 拉普拉斯方差阈值，低于此值视为模糊丢弃；0 = 不过滤
SAVE_CSV = True             # 是否输出索引 CSV
# =========================================================

VIDEO_EXT = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.m4v',
             '.mpg', '.mpeg', '.ts', '.webm', '.mts', '.3gp'}
BASE_DIR = Path(__file__).resolve().parent


def save_image(path: Path, image, quality=100):
    """支持中文路径的图片保存。"""
    ext = path.suffix.lower()
    if ext in ('.jpg', '.jpeg'):
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif ext == '.png':
        params = [cv2.IMWRITE_PNG_COMPRESSION, 1]  # 无损，速度优先
    else:
        params = []
    ok, buf = cv2.imencode(ext, image, params)
    if not ok:
        raise IOError(f'编码失败: {path}')
    buf.tofile(str(path))


def collect_videos(input_path: Path):
    if input_path.is_dir():
        return [p for p in sorted(input_path.iterdir())
                if p.is_file() and p.suffix.lower() in VIDEO_EXT]
    if input_path.is_file():
        return [input_path]
    raise FileNotFoundError(f'输入路径不存在: {input_path}')


def process_video(video: Path, out_root: Path):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f'[失败] 无法打开: {video}')
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or fps > 1000:
        fps = 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = max(0, int(round(START_SEC * fps)))
    if END_SEC is not None:
        end_frame = int(round(END_SEC * fps))
        if total > 0:
            end_frame = min(end_frame, total)
    else:
        end_frame = total if total > 0 else None

    # 计算采样步长（单位：帧）
    if SAMPLE_MODE == "num":
        span = (end_frame - start_frame) if end_frame else max(total - start_frame, 1)
        step = max(1.0, span / max(1, SAMPLE_VALUE))
    elif SAMPLE_MODE == "every_n":
        step = float(SAMPLE_VALUE)
    else:  # interval
        step = max(1e-6, SAMPLE_VALUE * fps)
    next_target = float(start_frame)

    out_dir = out_root / video.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = '.' + IMAGE_EXT

    rows, saved, idx = [], 0, 0
    while True:
        if not cap.grab():          # grab 只解复用不解码，跳帧极快
            break
        if idx < start_frame:
            idx += 1
            continue
        if end_frame is not None and idx >= end_frame:
            break
        if MAX_PER_VIDEO and saved >= MAX_PER_VIDEO:
            break

        if idx + 1e-9 >= next_target:
            while next_target <= idx + 1e-9:
                next_target += step
            ok, frame = cap.retrieve()   # 只对要保存的帧解码
            if ok and frame is not None:
                sharp = None
                if MIN_SHARPNESS > 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                if sharp is None or sharp >= MIN_SHARPNESS:
                    name = f'{video.stem}_{idx:06d}{ext}'
                    save_image(out_dir / name, frame, JPG_QUALITY)
                    rows.append([video.name, idx, round(idx / fps, 4),
                                 frame.shape[1], frame.shape[0], name,
                                 '' if sharp is None else round(sharp, 2)])
                    saved += 1
                    print(f'  保存 {name}', flush=True)
        idx += 1

    cap.release()
    print(f'[完成] {video.name}: 共 {saved} 张  '
          f'(fps={fps:.3f}, 分辨率={width}x{height}, 总帧={total or "未知"})')
    return rows


def main():
    input_path = Path(VIDEO_INPUT)
    out_root = BASE_DIR / OUTPUT_FOLDER_NAME
    out_root.mkdir(parents=True, exist_ok=True)

    videos = collect_videos(input_path)
    if not videos:
        raise SystemExit('未找到任何视频文件')

    if SAMPLE_MODE not in ('every_n', 'interval', 'num'):
        raise ValueError('SAMPLE_MODE 只能是 "every_n" / "interval" / "num"')
    if SAMPLE_VALUE <= 0:
        raise ValueError('SAMPLE_VALUE 必须为正数')

    print(f'输出目录: {out_root}')
    all_rows = []
    for v in videos:
        print(f'处理: {v}')
        all_rows += process_video(v, out_root)

    if SAVE_CSV and all_rows:
        csv_path = out_root / 'index.csv'
        with csv_path.open('w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(['video', 'frame_index', 'timestamp_sec',
                        'width', 'height', 'filename', 'sharpness'])
            w.writerows(all_rows)
        print(f'[索引] {csv_path}')

    print(f'[总计] {len(videos)} 个视频，共保存 {len(all_rows)} 张到 {out_root}')


if __name__ == '__main__':
    main()