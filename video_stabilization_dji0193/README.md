# DJI_20240310135754_0193 固定视角视频稳像

## 检测结论

输入视频是 3840×2160、25 fps、约 227.88 s 的固定视角无人机视频。抽样检测得到：

- ORB 特征匹配数：约 870–980 / 帧对
- RANSAC 内点率：约 0.83–0.96
- 相邻帧平移：通常小于 0.25 px（960 px 分析宽度）
- 相邻帧旋转：通常小于 0.02°，抽样长间隔最大约 0.12°
- 缩放变化：约 0.1% 以内

因此使用全局相似变换即可，不需要光流网格或透视单应性。视频中的车辆、行人等局部运动会通过 RANSAC 内点筛选被排除。

## 稳像方案

`stabilize.py` 采用两遍处理：

1. 第一遍以 960 px 宽度读取视频，使用 ORB + 汉明距离匹配 + RANSAC 估计帧间相似变换。
2. 累积平移、旋转和缩放轨迹，用高斯窗口平滑；固定视角模式再将轨迹轻微拉回全局中位姿态，抑制慢漂移。
3. 第二遍按 `平滑轨迹 × 原始轨迹的逆` 计算每帧校正矩阵，以画面中心为旋转中心进行仿射变换。
4. 默认裁掉四周 4%，再放大回 3840×2160，避免边缘黑边；音频从原视频复制。

## 运行

先只检测并生成报告：

```powershell
D:\anaconda\envs\drone-tools\python.exe video_stabilization_dji0193\stabilize.py `
  --input "E:\DJI\9\DJI_20240310135754_0193_D.MP4" `
  --report video_stabilization_dji0193\dji0193_report.json `
  --analyze-only
```

生成稳像视频：

```powershell
D:\anaconda\envs\drone-tools\python.exe video_stabilization_dji0193\stabilize.py `
  --input "E:\DJI\9\DJI_20240310135754_0193_D.MP4" `
  --output video_stabilization_dji0193\DJI_0193_stabilized.mp4 `
  --report video_stabilization_dji0193\dji0193_report.json
```

调参建议：

- 抖动仍明显：将 `--smooth-window` 从 41 增大到 61。
- 画面边缘被裁得过多：将 `--crop` 降到 0.03。
- 需要保留更多边缘：使用 `--border reflect`，但边缘可能出现反射填充。

默认参数针对本视频：`smooth_window=41`、`crop=0.04`、`camera_lock=0.85`。
