# IDSW 调试记录（BOTSORT + ReID 恢复关联）

最后更新：2026-07-24

## 1. 各版本运行结果对比

| 版本 | IDSW | MOTA | 唯一track数(GT=375) | 运行目录 |
|---|---|---|---|---|
| 基线：KF回退(无角度)+GMC修复+虚拟轨迹(LI+GPR)，无ReID | **16** | 98.16% | 389 | `outputs/yolo26-obb_20260722_155902` |
| 尝试1：加流场引导第三轮恢复(几何方案) | 84→1(过拟合/false merge，已废弃) | - | 315(<GT，误合并) | 已回退，不再使用 |
| 开ReID，`get_dists()`带ReID融合(bug) + 第三轮ReID恢复(bug) | **76** | - | 240 | `outputs/yolo26-obb_20260723_132045` |
| 修复`get_dists()`(去掉ReID融合，纯IoU) + 第三轮ReID恢复仍有bug | **89**（比76更差，证明get_dists不是主因） | 98.04% | 240 | `outputs/yolo26-obb_20260723_180709` |
| 修复第三轮ReID恢复的距离合理性门槛（Bug B fix） | **24** | 98.13% | 249 | `outputs/yolo26-obb_20260723_213702` |
| **加"驶出画面方向感知"排除（Bug C fix）** | **10**（比16基线还低） | **98.18%**（全版本最优） | **367**（最接近GT375） | `outputs/yolo26-obb_20260724_120344` |

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

## 3.5 24次残留IDSW的分析（Bug B修复后，`outputs/yolo26-obb_20260723_213702`）

修复Bug B后IDSW从89降到24，涉及15条GT车辆：`[136,138,150,157,169,182,218,223,227,268,275,293,313,337,365]`。其中136/150/218/275/293/313六条是原16基线也出问题的车（还没根治，只是换了个错法），157/169/182/223/227/268/337/365八条是ReID新引入的问题。

按`new_pred_id`是否在这次switch之前有过更早历史，分两类：
- 12次：`new_pred_id`是全新id（首次出现），非"复活"
- **12次：`new_pred_id`是丢失后被复活的旧id（gap 30~117帧，平均约70帧）**——即真正的"僵尸复活"模式

### 尝试并否决：单纯"最后位置在画面边缘"判据

最初假设：这12次僵尸复活的候选，丢失前最后位置是否都在画面边缘（车已经真的驶出画面）？实测结果**12/12（100%）都在边缘**（左边界x≈0~8，或上边界y≈25~75，对应那条弯曲匝道的位置）。

但补充验证时发现这个判据**不能直接拿来当排除条件**：统计全视频里所有track**内部**发生的长间隔(≥30帧)自恢复（同一个id丢失又复活，因为id没变所以不会被计成IDSW，大概率是合法恢复），共192次，其中**175次(91%)恢复前最后位置同样在画面边缘**。原因是那条弯曲匝道（低置信度+护栏遮挡的问题根源）本身就紧贴画面左上角边界，"车真的开走了"和"车还在匝道上只是没检测到"这两种情况在这个场景里长得几乎一样——单纯按位置判断的话，把边缘候选全部排除会连带打掉91%的所有恢复（包括已经把GT46、GT198完全修好的合法恢复），**没有采用**。

### Bug C（已修复）：僵尸复活判据改为"位置+运动方向"

改进判据：只有当track丢失前**最后位置在边缘（`EDGE_MARGIN_PX=100`）且卡尔曼速度朝向最近边界的分量超过`EDGE_EXIT_MIN_SPEED=0.5px/帧`**（即真的朝画面外驶出，不是刚驶入/原地丢失/沿边界平行运动）才排除。

用真实torchreid+GPU复现脚本(`repro_costmatrix_edge.py`，冷启动帧1，全量1824帧跑了约47分钟)对12次僵尸事件重新分类：**10/12(83%)判定为EXITING（应排除）**，只有2次（revival_frame=564的track189、revival_frame=1142的track110，都是朝画面内部/平行运动）不会被新判据挡住。对192次内部自恢复重新分类：148 EXITING / 27 ENTERING_OR_PASSING / 17 NOT_EDGE——即"应该保留"的比例从9%(17/192)提升到23%(44/192)，且很多被判EXITING的案例本身速度就很大（如朝画面下边界飙出，dot值10~15px/帧），本身更像是真·驶离而非"合法恢复被误伤"。

**实现**：`ReidRecoveryConfig`新增`EDGE_MARGIN_PX=100.0`、`EDGE_EXIT_MIN_SPEED=0.5`；`BOTSORT`新增`self.frame_shape`（在`_pre_first_associate`里从`img.shape[:2]`获取）和`_is_exiting_frame(track)`方法（用`track.last_observed_mean`里的位置`[0:2]`+卡尔曼速度状态`[4:6]`，不是原始历史位置差分——避免了之前发现的"历史被id切换污染"问题）。`_post_first_association`的`lost_idx`候选池过滤新增`and not self._is_exiting_frame(strack_pool[i])`。

单元测试新增2个用例（test_reid_recovery.py的第6、7个）：靠边缘+朝外驶出→不恢复；靠边缘+朝内/平行→仍正常恢复。全部7个测试通过，且验证了旧的5个测试因为`bot.frame_shape`默认是`None`（`_is_exiting_frame`直接短路返回False）完全不受影响，向后兼容。

代码位置：`ultralytics/trackers/bot_sort.py` 的 `ReidRecoveryConfig`、`BOTSORT.__init__`、`_pre_first_associate`、`_is_exiting_frame`（新方法）、`_post_first_association`的`lost_idx`过滤条件。

诊断脚本：`_analyze_edge_zombies.py`（初版，只按静态位置）、`_check_legit_recoveries.py`（192次内部恢复的边缘占比统计，否决了静态位置方案）、`_check_exit_direction.py`（方向感知版重新分类，两组数据）、`repro_costmatrix_edge.py`（真实管线复现12次僵尸事件的完整候选代价矩阵）——都在项目根目录，非production代码，仅供调试参考。

## 4. 设计决策记录

- **`get_dists()`不加回ReID融合**：虽然确认不是bug，但也没证据证明有用（去掉后指标反而没变差），且它解决的"近距离视觉混淆"场景在本项目诊断出的问题里没出现过。保持简单，等真的遇到这类新问题再考虑加回。
- **第三轮恢复只用外观相似度，不用任何运动/位置预测作为主要判据**：之前试过纯几何方案（流场引导），在这种车流密集、外观相近车辆多的场景下极易"就近乱认亲"，效果比现在更差，已放弃。现在的距离门槛只是**兜底上限**，不是主要判据。
- **`MIN_GAP_FOR_RECOVERY=30`**：短丢失交给正常IoU两轮关联，没必要走ReID这条路。

## 5. 关键文件位置

- `ultralytics/trackers/bot_sort.py` — 核心：`ReidRecoveryConfig`、`_post_first_association`（第三轮ReID恢复）、`get_dists`（已改回纯IoU）、`_maybe_inject_virtual_trajectory`（LI+GPR虚拟轨迹）
- `ultralytics/trackers/utils/virtual_trajectory.py` — LI+GPR实现
- `tracktworoi.py` — 生产脚本；`REID_MODEL_PATH`已指向真实torchreid权重；`ReIDEncoder`类（约654行）已补上`__call__`方法
- `reid_model/osnet_ain_x1_0_vehicle.onnx` — 导出的ONNX版本，**仅供本地非交互测试脚本用**（走`ultralytics/trackers/utils/reid.py`的`ReID`/`AutoBackend`路径），生产环境走的是`tracktworoi.py`里的torchreid原生GPU路径，两者不是同一套代码但共享底层tracker逻辑
- `test_reid_recovery.py` — 第三轮恢复逻辑的单元测试（7个用例：原5个+Bug C修复的方向感知测试2个，含真实bug场景的回归测试）
- `test_virtual_trajectory.py` — 虚拟轨迹单元测试
- `eval_mot.py` — MOT评估脚本，每次换`PRED_TXT`指向新跑的`cvat_eval.txt`
- `repro_reid_bug.py` + `_repro_base.py` — 非交互复现脚本，用于给内部关联逻辑打日志调试（可从任意`COLD_START_FRAME`冷启动，走真实生产检测+ReID管线）。`_repro_base.py`是从`tracktworoi.py`用`sed`切出纯函数定义部分组装的，如果`tracktworoi.py`改了核心逻辑需要重新生成。**注意`_repro_base.py`里的`REID_MODEL_PATH`是独立硬编码的，不随`tracktworoi.py`当前值变化**
- `repro_costmatrix_edge.py` — 基于`_repro_base.py`的另一个复现脚本，全量冷启动(帧1)跑完整视频，在指定的(帧,位置)事件点打印第三轮ReID恢复的完整候选代价矩阵，用于Bug C的验证
- `ultralytics/trackers/utils/reid.py` — 共享ReID基础设施，之前修过一个非方形输入的bug（只读了ONNX输入shape的高度维度）
- 运行环境：生产管线(`tracktworoi.py`/`_repro_base.py`及其衍生复现脚本，含真实torchreid+CUDA)必须用 `F:\ybw\ultralytics-8.1.41\python.exe`（其他conda环境要么没torch/cv2，要么没torchreid——torchreid是通过`sys.path.insert`指向`D:\deep-person-reid-master`源码目录使用的，不是pip装的）；`eval_mot.py`等纯CPU评估脚本可以用`D:\anaconda\envs\YOLOv8-Deepsort\python.exe`（有cv2+scipy）
- `tracktworoi.py`非交互跑法：ROI文件(`E:\DJI\cvat\0111_D_stabilized_roi.json`)已存在时，只需要对"是否重新绘制ROI"这一个prompt回车，后续比例标定/绘制都会被跳过，不会有其它prompt。**注意：`echo "" | python tracktworoi.py`这种管道形式在后台(`run_in_background`)运行时会挂死**（进程起来了但`input()`一直收不到那一行，CPU几乎不涨，很容易误判成"在正常处理只是没flush"）；必须用文件重定向 `python tracktworoi.py < 一个只含换行符的文件 > log 2>&1` 才可靠

## 3.6 Bug C修复后实测结果（`outputs/yolo26-obb_20260724_120344`）

IDSW从24降到**10**，比原始16基线还低；MOTA 98.18%、IDF1 99.09%、唯一track数367，三项都是目前所有版本里最好的。之前ReID新引入的8条GT车辆问题（157,169,182,223,227,268,337,365）**全部消失**，说明这次不是"挡住了部分僵尸、换个方式又冒出来"，而是干净地解决了。

剩余10次事件，涉及的9条GT车辆**正好是原始16基线的那9条**（46,136,138,150,198,218,275,293,313），一个新面孔都没有：

```
163  138  137  164
420  46   114  207
467  136  135  212
576  150  150  230
893  198  200  271
1120 218  227  309
1141 218  309  227
1444 275  290  354
1649 293  313  376
1784 313  332  393
```

和16基线逐车对比：GT46(2→1)、GT136(3→1)、GT150(2→1)、GT198(2→1)、GT275(2→1)各修好了一部分；GT218仍是2次（227↔309来回震荡，同一种"两个都合理的候选来回摇摆"模式，没解决）；GT138/293/313这3条**完全没变**（同一帧、同样的old_pred，new_pred只是编号不同）——大概率是这几个case的丢失间隔本来就不到`MIN_GAP_FOR_RECOVERY=30`帧，第三轮ReID恢复根本没触发，跟纯KF基线走的是完全一样的失败路径，不是ReID出错，是ReID压根没参与。

## 6. 待办

- [x] 用修复Bug B后的代码跑完整`tracktworoi.py`，IDSW从89→24，但仍有12次是"边缘僵尸复活"模式（见3.5节）
- [x] 定位12次残留边缘僵尸复活的根因，否决了"静态边缘位置"判据，改用"位置+运动方向"（Bug C）
- [x] 跑完Bug C修复后的完整`tracktworoi.py`，IDSW 24→**10**，见3.6节
- [ ] 剩余10次里，GT138/293/313这3次疑似"丢失间隔<30帧、ReID根本没触发"，需要确认具体gap帧数，如果确实<30可以考虑是否要把`MIN_GAP_FOR_RECOVERY`调低（但要重新评估会不会引入新的短间隔误判）
- [ ] GT218仍有2次来回震荡(227↔309)，两个候选大概率都物理/外观合理，是否需要额外的tie-break逻辑待定
- [ ] 内存泄漏问题（之前诊断ONNX路径时发现RSS会随第三轮恢复触发次数加速增长到6GB+）尚未查清根因，但确认只出现在诊断脚本用的ONNX/AutoBackend路径，不影响生产的torchreid路径，优先级较低
