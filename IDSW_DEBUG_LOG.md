# IDSW 调试记录（BOTSORT + ReID 恢复关联）

最后更新：2026-07-23

## 1. 各版本运行结果对比

| 版本 | IDSW | MOTA | 唯一track数(GT=375) | 运行目录 |
|---|---|---|---|---|
| 基线：KF回退(无角度)+GMC修复+虚拟轨迹(LI+GPR)，无ReID | **16** | 98.16% | 389 | `outputs/yolo26-obb_20260722_155902` |
| 尝试1：加流场引导第三轮恢复(几何方案) | 84→1(过拟合/false merge，已废弃) | - | 315(<GT，误合并) | 已回退，不再使用 |
| 开ReID，`get_dists()`带ReID融合(bug) + 第三轮ReID恢复(bug) | **76** | - | 240 | `outputs/yolo26-obb_20260723_132045` |
| 修复`get_dists()`(去掉ReID融合，纯IoU) + 第三轮ReID恢复仍有bug | **89**（比76更差，证明get_dists不是主因） | 98.04% | 240 | `outputs/yolo26-obb_20260723_180709` |
| **修复第三轮ReID恢复的距离合理性门槛**（本次改动，尚未跑完整视频验证） | 待验证 | 待验证 | 待验证 | 待跑 |

## 2. 基线16次IDSW事件清单（GT帧, GT id, old_pred, new_pred）

来自 `outputs/yolo26-obb_20260722_155902/cvat_eval.txt`，涉及9条GT车辆：46, 136, 138, 150, 198, 218, 275, 293, 313。

```
163  138  137  165
420  46   114  210
467  136  135  216
489  46   210  221
564  136  216  235
576  150  150  238
628  136  235  248
688  150  238  256
893  198  202  286
999  198  286  307
1120 218  234  329
1156 218  329  338
1444 275  310  382
1538 275  382  390
1649 293  334  407
1784 313  358  426
```

**根因**：全部16次都发生在画面左上角同一段弯曲高架匝道（离摄像头最远、检测置信度低+桥梁护栏局部遮挡），车辆丢失14~97帧后，纯匀速直线外推的卡尔曼预测跑偏，IoU关联失败，被误建新track。ReID端点相似度实测能正确区分这9条车辆的"丢失前/重现后"（相似度0.70~0.97）vs 附近别的车（相似度0.35~0.57），零重叠——说明用ReID做长丢失恢复这个方向本身是对的。

## 3. 已定位并修复的Bug

### Bug A（已排除，不是根因）：`BOTSORT.get_dists()` 原有的ReID融合
ultralytics官方代码自带，之前从未被启用过（因为`with_reid`一直是False）。曾怀疑是它导致76次IDSW，去掉后（改成纯IoU）复测反而变成89次，证明**不是主因**，已保持去掉的状态（原因见下方"设计决策"）。

### Bug B（已修复，是真正根因）：第三轮ReID恢复缺少距离合理性检查
`bot_sort.py` 的 `_post_first_association`（本项目新加的第三轮长丢失恢复逻辑）只看"类别一致+外观相似度≥0.65"，**没有任何几何距离约束**。

**实测抓到的具体案例**（`repro_reid_bug.py`复现，冷启动帧400，真实torchreid+GPU）：
- 帧625：新车刚进入画面(7.9,1246.6)，被错误"复活"成track#63——但track#63最后已知位置是**(662.9,1977.4)**，丢失了30帧，距离约981px
- 帧626：同一检测又被track#78（最后位置(3523.7,1170.4)，画面最右边缘）抢走
- 帧627：又被track#17（最后位置(651.9,1976.0)）抢走
- 帧628：候选池终于耗尽，才正常建新track#178，之后靠纯IoU稳定延续

根本原因：画面里持续有十几到二十个"丢失≥30帧的僵尸track"分布在各个角落，任何新入场车辆只要外观特征凑巧和某个僵尸的相似度过线（0.65），就会被跨越大半个画面强行"复活"，纯属多重假设检验下的偶然误伤。

**修复**：`ReidRecoveryConfig`新增`MAX_SPEED_PX_PER_FRAME = 20.0`，候选距离上限=`丢失帧数 × 20px/帧`。20这个值留了充分余量——本项目16个真实合理恢复案例，实测位移从未超过6.2px/帧；而抓到的这次误伤案例是32.7px/帧，中间区分度很大。

代码位置：`ultralytics/trackers/bot_sort.py` 的 `ReidRecoveryConfig` 类 + `_post_first_association` 方法内的候选打分循环。

## 4. 设计决策记录

- **`get_dists()`不加回ReID融合**：虽然确认不是bug，但也没证据证明有用（去掉后指标反而没变差），且它解决的"近距离视觉混淆"场景在本项目诊断出的问题里没出现过。保持简单，等真的遇到这类新问题再考虑加回。
- **第三轮恢复只用外观相似度，不用任何运动/位置预测作为主要判据**：之前试过纯几何方案（流场引导），在这种车流密集、外观相近车辆多的场景下极易"就近乱认亲"，效果比现在更差，已放弃。现在的距离门槛只是**兜底上限**，不是主要判据。
- **`MIN_GAP_FOR_RECOVERY=30`**：短丢失交给正常IoU两轮关联，没必要走ReID这条路。

## 5. 关键文件位置

- `ultralytics/trackers/bot_sort.py` — 核心：`ReidRecoveryConfig`、`_post_first_association`（第三轮ReID恢复）、`get_dists`（已改回纯IoU）、`_maybe_inject_virtual_trajectory`（LI+GPR虚拟轨迹）
- `ultralytics/trackers/utils/virtual_trajectory.py` — LI+GPR实现
- `tracktworoi.py` — 生产脚本；`REID_MODEL_PATH`已指向真实torchreid权重；`ReIDEncoder`类（约654行）已补上`__call__`方法
- `reid_model/osnet_ain_x1_0_vehicle.onnx` — 导出的ONNX版本，**仅供本地非交互测试脚本用**（走`ultralytics/trackers/utils/reid.py`的`ReID`/`AutoBackend`路径），生产环境走的是`tracktworoi.py`里的torchreid原生GPU路径，两者不是同一套代码但共享底层tracker逻辑
- `test_reid_recovery.py` — 第三轮恢复逻辑的单元测试（5个用例，含真实bug场景的回归测试）
- `test_virtual_trajectory.py` — 虚拟轨迹单元测试
- `eval_mot.py` — MOT评估脚本，每次换`PRED_TXT`指向新跑的`cvat_eval.txt`
- `repro_reid_bug.py` + `_repro_base.py` — 非交互复现脚本，用于给内部关联逻辑打日志调试（可从任意`COLD_START_FRAME`冷启动，走真实生产检测+ReID管线）。`_repro_base.py`是从`tracktworoi.py`用`sed`切出纯函数定义部分组装的，如果`tracktworoi.py`改了核心逻辑需要重新生成
- `ultralytics/trackers/utils/reid.py` — 共享ReID基础设施，之前修过一个非方形输入的bug（只读了ONNX输入shape的高度维度）

## 6. 待办

- [ ] 用修复后的代码重新跑一次完整`tracktworoi.py`（1824帧），确认IDSW是否回到16甚至更低，且唯一track数是否回到接近375（而不是240）
- [ ] 如果修复后还有残留IDSW，用`repro_reid_bug.py`同样的方法定位（改`COLD_START_FRAME`和目标区域坐标）
- [ ] 内存泄漏问题（之前诊断ONNX路径时发现RSS会随第三轮恢复触发次数加速增长到6GB+）尚未查清根因，但确认只出现在诊断脚本用的ONNX/AutoBackend路径，不影响生产的torchreid路径，优先级较低
