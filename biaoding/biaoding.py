"""相机内参标定（张正友棋盘格法），无需命令行。

直接修改下方【配置区】的 IMAGE_DIR 为标定图片文件夹路径，然后运行：
    python calibrate_camera.py

结果自动保存在 IMAGE_DIR 的“上一级目录”下新建的 calibration_results 文件夹。
迭代剔除过程中，每剔除一张图都会把该轮的标定结果单独输出到 iterations/ 下。
"""
import csv
import json
from pathlib import Path

import cv2
import numpy as np

# ==================== 配置区（改这里） ====================
# 标定图片所在文件夹（直接写路径即可）
IMAGE_DIR = r"D:\yolo26-tracking\biaoding\frames_output10.22.4k3\4k3"

# 结果文件夹名，会建在 IMAGE_DIR 的上一级目录下
RESULT_FOLDER_NAME = "calibration_results4k3"

# 棋盘格内角点数（不是格子数），按顺序尝试
BOARDS = ['11x8', '6x8']

# 手动排除的图片文件名，例如 ['bad1.jpg', 'bad2.jpg']
EXCLUDE = []

# 迭代剔除：每轮剔除单张重投影 RMS 最差的一张，直到剩余 TRIM_TO 张。
# None = 不剔除，直接用全部检测成功的图。
# 若 TRIM_TO >= 检测成功数，会自动跳过剔除（不报错）。
TRIM_TO = None
# ======================================================

IMAGE_EXT = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']


def save_image(path, image):
    """支持中文路径的图片保存。"""
    cv2.imencode(path.suffix, image)[1].tofile(str(path))


def fit(views, size, flags=0):
    result = cv2.calibrateCameraExtended(
        [v['object'] for v in views], [v['corners'] for v in views], size,
        None, None, flags=flags,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 150, 1e-10))
    rms, K, dist, rotations, translations, std, _, errors = result
    return dict(rms_pixels=float(rms), camera_matrix=K.tolist(),
                distortion_order=['k1', 'k2', 'p1', 'p2', 'k3'],
                distortion=dist.ravel().tolist(), intrinsic_std=std.ravel().tolist(),
                per_image=[dict(file=v['name'], board=v['board'], rms_pixels=float(e),
                                rotation_vector=r.ravel().tolist(),
                                translation_in_square_units=t.ravel().tolist())
                           for v, e, r, t in zip(views, errors.ravel(), rotations, translations)])


def save_params(fit_result, out: Path):
    """把一组标定参数写到指定目录。"""
    out.mkdir(parents=True, exist_ok=True)
    K = np.array(fit_result['camera_matrix'])
    dist = np.array(fit_result['distortion'])
    np.savetxt(out / 'camera_matrix.txt', K, fmt='%.10f')
    np.savetxt(out / 'distortion.txt', dist[None, :], fmt='%.10f', header='k1 k2 p1 p2 k3')
    return K, dist


def save_iteration(iteration, out: Path, iter_idx, retained, removed_file, removed_rms):
    """把单轮迭代结果完整输出到 iterations/ 下。"""
    snap_dir = out / 'iterations' / f'iter_{iter_idx:02d}_views_{retained}'
    snap_dir.mkdir(parents=True, exist_ok=True)
    snapshot = dict(
        iteration=iter_idx,
        images_retained=retained,
        overall_rms_pixels=iteration['rms_pixels'],
        removed_file=removed_file,
        removed_image_rms_pixels=removed_rms,
        camera_matrix=iteration['camera_matrix'],
        distortion=iteration['distortion'],
        per_image=iteration['per_image'],
    )
    (snap_dir / 'iteration.json').write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False), encoding='utf-8')
    save_params(iteration, snap_dir)
    return snap_dir


def main():
    image_dir = Path(IMAGE_DIR)
    if not image_dir.is_dir():
        raise SystemExit(f'图片目录不存在: {image_dir}')

    out = image_dir.parent / RESULT_FOLDER_NAME      # 结果放在照片上一级目录
    (out / 'corners').mkdir(parents=True, exist_ok=True)
    (out / 'iterations').mkdir(parents=True, exist_ok=True)
    print(f'图片目录: {image_dir}')
    print(f'结果目录: {out}')

    cv2.setNumThreads(4)
    boards = [tuple(map(int, b.lower().split('x'))) for b in BOARDS]
    excluded = set(EXCLUDE)
    views, failed, size = [], [], None

    files = sorted((p for p in image_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_EXT
                    and p.name not in excluded), key=lambda p: p.name)
    if not files:
        raise SystemExit('图片目录中没有可用图片')

    # ---------- 1. 检测角点 ----------
    for path in files:
        gray = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            failed.append(dict(file=path.name, reason='unreadable')); continue
        current_size = (gray.shape[1], gray.shape[0])
        if size is None:
            size = current_size
        if size != current_size:
            failed.append(dict(file=path.name, reason='different resolution')); continue
        scale = min(1., 1600 / gray.shape[1])
        small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        found = False
        for board in boards:
            found, corners = cv2.findChessboardCornersSB(
                small, board, flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
            if found:
                # 不同 OpenCV 版本返回 (N,2) 或 (N,1,2)，统一成 (N,1,2)
                corners = corners.reshape(-1, 1, 2).astype(np.float32)
                break
        if not found:
            failed.append(dict(file=path.name, reason='no complete board'))
            print('FAILED', path.name, flush=True)
            continue
        # 把缩小图上的角点映射回原图坐标，再在原分辨率做亚像素精化
        corners[:, :, 0] = (corners[:, :, 0] + .5) * gray.shape[1] / small.shape[1] - .5
        corners[:, :, 1] = (corners[:, :, 1] + .5) * gray.shape[0] / small.shape[0] - .5
        cv2.cornerSubPix(gray, corners, (9, 9), (-1, -1),
                         (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 80, .001))
        obj = np.zeros((board[0] * board[1], 3), np.float32)
        obj[:, :2] = np.mgrid[0:board[0], 0:board[1]].T.reshape(-1, 2)
        views.append(dict(name=path.name, board=board, object=obj, corners=corners))

        preview = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
        display = corners.copy()
        display[:, :, 0] *= small.shape[1] / gray.shape[1]
        display[:, :, 1] *= small.shape[0] / gray.shape[0]
        cv2.drawChessboardCorners(preview, board, display, True)
        save_image(out / 'corners' / (path.stem + '.jpg'), preview)
        print('OK', path.name, board, flush=True)

    if len(views) < 5:
        raise RuntimeError('检测成功少于 5 张，无法标定；请补充更多不同姿态的照片。')

    initially_detected = len(views)

    # ---------- 2. 迭代剔除，每轮输出一次结果 ----------
    trim_to = TRIM_TO
    if trim_to is not None:
        if trim_to < 5:
            raise ValueError('TRIM_TO 必须 >= 5')
        if trim_to >= len(views):
            print(f'[INFO] TRIM_TO={trim_to} >= 检测成功数 {len(views)}，跳过迭代剔除')
            trim_to = None

    trim_history = []
    while trim_to is not None and len(views) > trim_to:
        iteration = fit(views, size)
        worst = max(iteration['per_image'], key=lambda row: row['rms_pixels'])
        iter_idx = len(trim_history) + 1

        # 每轮输出一次结果
        snap_dir = save_iteration(iteration, out, iter_idx,
                                  retained=len(views),
                                  removed_file=worst['file'],
                                  removed_rms=worst['rms_pixels'])
        K = np.array(iteration['camera_matrix'])
        dist = np.array(iteration['distortion'])
        print(f'[ITER {iter_idx:02d}] views={len(views)} '
              f'RMS={iteration["rms_pixels"]:.4f}px  '
              f'fx={K[0,0]:.3f} fy={K[1,1]:.3f} cx={K[0,2]:.3f} cy={K[1,2]:.3f}  '
              f'dist=[{", ".join(f"{d:.6f}" for d in dist)}]  '
              f'-> 剔除 {worst["file"]} ({worst["rms_pixels"]:.4f}px)', flush=True)
        print(f'        本轮结果已保存: {snap_dir}')

        trim_history.append(dict(images_before=len(views),
                                 overall_rms_pixels=iteration['rms_pixels'],
                                 removed_file=worst['file'],
                                 removed_image_rms_pixels=worst['rms_pixels'],
                                 saved_to=str(snap_dir.relative_to(out))))
        views = [v for v in views if v['name'] != worst['file']]

    # ---------- 3. 最终结果 ----------
    result = dict(image_size=list(size), square_size=1, square_unit='one grid square',
                  excluded_images=sorted(excluded),
                  initially_detected=initially_detected, detected=len(views), failed=failed,
                  iterative_rejections=trim_history,
                  all_images=fit(views, size),
                  fixed_k3=fit(views, size, cv2.CALIB_FIX_K3),
                  board_groups={})
    for board in boards:
        group = [v for v in views if v['board'] == board]
        if len(group) >= 5:
            result['board_groups'][str(board)] = fit(group, size)

    points = np.concatenate([v['corners'].reshape(-1, 2) for v in views])
    result['corner_extent_pixels'] = [points.min(axis=0).tolist(), points.max(axis=0).tolist()]

    (out / 'calibration.json').write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    K, dist = save_params(result['all_images'], out)
    np.savez(out / 'calibration.npz', camera_matrix=K, distortion=dist, image_size=size)

    with (out / 'per_image_errors.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['filename', 'inner_corners', 'rms_pixels'])
        for row in result['all_images']['per_image']:
            w.writerow([row['file'], str(row['board']), row['rms_pixels']])

    np.savez(out / 'detected_points.npz',
             **{f'image_{i}': v['corners'] for i, v in enumerate(views)})

    print('\n===== 最终标定结果 =====')
    print(json.dumps({k: result['all_images'][k]
                      for k in ['rms_pixels', 'camera_matrix', 'distortion']}, indent=2))
    print(f'全部结果已保存到: {out}')


if __name__ == '__main__':
    main()