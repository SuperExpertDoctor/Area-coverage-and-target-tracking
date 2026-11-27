# Area-coverage-and-target-tracking

单固定翼UAV、单艘科考船、300 km × 300 km二维海域；两个独立GRU模仿学习模型。云只限制飞机，岛只限制船舶；覆盖用右侧扇形，跟踪用可偏转前视扇形。

详细算法、等式约束、初始化和接口见[设计文档](docs/superpowers/specs/2026-11-27-maritime-imitation-design.md)，实施步骤和问题修复见[实施方案](docs/superpowers/plans/2026-11-27-maritime-imitation-implementation.md)。

本分支 `feat/local-potential-field` 在 `c56e40c` 上增加局部引斥场。新接口、指数曲线、优化评分与实现步骤见[局部场设计](docs/superpowers/specs/2026-11-27-local-potential-field-design.md)及[实施计划](docs/superpowers/plans/2026-11-27-local-potential-field.md)。`run.sh` 默认启用 field-v2，独立输出到 `outputs/field-v2`、`tf-logs/field-v2`；旧 v1 数据和权重保持独立。

新场值为：刚扫描 −1、按归一化指数在信息价值失效周期 T 恢复至 +1；从未扫描 +1；ROI 外 −1；雷云及安全膨胀区固定 −2。覆盖正引力只在当前子任务 ROI 的海水格中存在，数值和绘图均遵守边界。只生成 UAV 周围半径内的33×33场，历史仅存扫描时间戳；T 控制信息老化，不是重访期限。两个任务分别使用带运动几何参考的残差 CNN+GRU 模型，共同输入 `vector + potential + field_context`，输出世界 XY 单位速度方向与归一化雷达 pan；恒速、有限滚转与雷达转速仍由环境执行。侧视/前视保留各自安装方向。前端“局部引斥场”开关与 MP4 使用同一固定色标。

field-v2 专家当前24/24固定场景通过，0排除、0失败：16例覆盖各10800秒全部达到100%，8例跟踪各1800秒全程可见。跟踪网络在6个验证回合和4个新的独立1800秒测试回合均全程可见；覆盖网络尚在闭环修复中。新版本的验收范围、失败证据与本地路径见[局部场验证记录](docs/local-potential-field-validation.md)。旧 v1 的64例专家、17例跟踪网络及覆盖网络局限见[原验证状态](docs/validation-status.md)，这些旧数字不计入 field-v2。

覆盖候选第31轮虽通过固定验证和4个新测试，实际7100演示仍在37秒碰云，已撤下部署别名并保留失败回放。后续使用该场景进行训练诊断，新独立测试集另行冻结；不会以低损失或此前成功案例替代这次失败。

共享云编码修复模型已通过同一7100及16/16已知初始化回归，但新32100–32103测试为3/4：32103在5481.1秒碰云。因此该候选仍未发布。场逻辑独立审计覆盖46份回放、29,335个快照，正引力全部局限在ROI海面；这项通过不代表网络飞行安全。详细权重SHA、指标及控制分歧见上面的验证记录。

## 安装与运行

新训练默认采用任务分别配置的云编码（`--field-cloud-geometry-encoder auto`）：覆盖使用共享几何编码，跟踪保留slots。已保存模型、warm-start和DAgger始终继承checkpoint实际架构。跨架构fresh重训可用 `--split-reference-checkpoint` 固定历史场景划分，权重和归一化重新初始化。

Python 3.10+：

```bash
python -m pip install -r requirements.txt
bash run.sh all
```

Windows请在Git Bash运行。PowerShell可显式调用：

```powershell
& 'C:/Program Files/Git/bin/bash.exe' ./run.sh all
```

`run.sh`集中列出仿真、专家、训练、路径和录像参数；修改其中数值，或在命令末尾覆盖对应CLI参数。`--device auto`优先可用CUDA。首次完整流程包括长专家试跑、两套采集、两次训练和闭环录像，耗时取决于硬件。

```bash
bash run.sh expert                         # 先验证两位专家，开启端口界面并导出影片
bash run.sh expert-validate --no-visualize --no-open-browser --no-keep-server
# 固定64案例：30随机覆盖、3个ROI各8个边界起点、10随机跟踪；耗时较长，可续跑
bash run.sh collect                        # 各自采集；失败轨迹不进入成功清单
bash run.sh train                          # 根据成功清单分别训练
bash run.sh evaluate                       # 从已保存专家场景恢复，再运行两网络
bash run.sh serve                          # 只启动历史回放界面
bash run.sh expert --no-potential-field --output-dir outputs/legacy-check  # 显式使用保留的 v1 专家
bash run.sh train --task tracking          # 单任务单独训练
bash run.sh dagger --task coverage --dagger-beta 0.8 --run-name coverage-dagger-1
# 继续训练时可用 --warm-start --checkpoint-path ...；只补采用 --dagger-collect-only
```

默认界面为[http://127.0.0.1:8787](http://127.0.0.1:8787)，任务完成后保持服务直到Ctrl+C。自动运行可加`--no-open-browser --no-keep-server`；不需要端口界面时加`--no-visualize`。独立回放也可直接：

```bash
python main.py --mode serve --potential-field --output-dir outputs/field-v2 --tf-log-dir tf-logs/field-v2 --port 8787
tensorboard --logdir tf-logs --port 6006
```

入口参数帮助：`python main.py --help`。仿真/专家全部参数位于`run.sh`的`SIM_CONFIG`，通过`--config-json`整体传入Python；不是隐藏在另一份配置文件中。

覆盖模型可用 `--coverage-mask-previous-command` 做独立消融：仅在模型内部预处理将上一拍命令 `vector[8:10]` 置零，原始共同观测和世界XY单位方向动作接口保持不变，tracking不受影响。`run.sh`显式默认 `--no-coverage-mask-previous-command`；旧checkpoint缺少 `model_config.mask_previous_command` 时按False解释，保留原行为。此开关只决定fresh覆盖训练的架构；warm-start与DAgger明确继承checkpoint配置，不用该fresh开关重解释旧权重。要启用该消融，应从零训练，或继续训练已保存True标志的模型。该选项本身不代表覆盖闭环已经达标。

## 产物

| 路径 | 内容 |
|---|---|
| src/ | 物理、环境、专家、两网络、采集、训练与服务源码 |
| visual/ | 原生JS海图、实时SSE、历史回放 |
| figs/ | 下载的彩虹CH-4、科考船、海洋照片；加载式二维SVG模型及来源许可 |
| outputs/field-v2/datasets/{coverage,tracking}/ | 默认采集的完整JSON、派生NPZ、accepted-manifest |
| outputs/field-v2/models/{coverage,tracking}/ | best.pt、last.pt、通过闭环选择后的 selected.pt |
| outputs/field-v2/{coverage,tracking}-experiment/ | 本地实跑数据、分组划分、候选权重、失败及测试回放 |
| outputs/field-v2/replays/、outputs/field-v2/final-demo-current/ | 专家/网络JSON、H.264 MP4与PNG |
| outputs/field-v2/training/、outputs/field-v2/reports/ | 训练报告和阶段报告 |
| outputs/field-v2/validation-report.json | 默认入口的两任务合并结果 |
| outputs/diagnostics/、outputs/cache/ | 诊断记录与缓存 |
| tf-logs/ | 分任务/分实验的TensorBoard事件文件 |

JSON训练数组使用可无损解码的zlib+base64，不依赖NPZ才能恢复。可用`src.rollouts.restore_training_cache(json_path)`重建训练缓存。演示JSON的frames是完整真实状态轨迹，前端可导入文件；选择策略、拖动进度条、切换任务区/全域/跟随视角。

全世界按250m栅格记录飞机实际飞越位置，最近飞越的格子最深，随仿真时间渐淡；雷达扫描年龄图独立显示。默认显示60秒流星尾迹，可关闭或切为完整轨迹。历史拖动会重建截至当前时刻的记录，不显示未来足迹。远景网格线自动抽稀，底层访问格大小不变。

雷达始终按真实3km绘制。图标按参考长宽缩放，同时保留最小像素以便辨认；因此全域视角的图标是有最小可见尺寸的符号，并非严格等比例的实机轮廓。`SIM_CONFIG`中的16个`visual_*`参数可控制网格、渐淡、尾迹、参考尺寸、最小像素及默认开关。服务通过`/api/visual-config`把当前入口的显示参数提供给旧回放，不改变其物理数据。

两模型不共享参数。field-v2 动作为二维期望世界速度方向加雷达 pan 指令，v1 保留原二维动作；真实机船位置由有限倾斜/舵机响应模型推进，不能直接跳点。图片/二维模型只影响显示，按真实速度旋转。

## 训练指标与检查

仅记录确定性连续策略适用的6条曲线：Train/Validation的Loss、Action_MSE、Action_MAE。field-v2 默认Loss是每步三个动作分量的平方误差和，MSE为逐分量均值；可选纠正权重启用时Train/Loss记录实际加权目标，MSE/MAE与Validation始终不加权；不虚构离散准确率或概率分布指标。验证集按完整场景划分，同一场景的相邻时间片不会泄漏。`best.pt` 按验证原始Action_MSE保存，部署 `selected.pt` 只按验证场景闭环结果生成；测试场景不参与选择。

可选 `--field-steering-loss-weight` 默认0；大于0时Loss额外包含根据相同观测计算的执行感知excess BCE，以监督实际左右转分支。原动作MSE/MAE、输入输出和推理控制保持原义。该项尚须独立闭环验证，不能据此声称安全；公式和局限见局部场设计。

`run.sh`还显式设置`--dagger-recovery-min-steps 7200`：短DAgger失败前缀按完整回合重复采样，每次重置GRU，不能拼接成虚构连续轨迹。原始场景划分、归一化和验证不重复；checkpoint记录采样预算及各回合实际曝光。设为0可关闭。无验证集时，重复采样模型按唯一训练回合的独立推理误差选取，避免与第0轮比较不同分布。

```bash
python -m pytest -q
python -m examples.plot_kinematics --output outputs/kinematics/kinematics.png
```

低验证误差不能替代闭环测试。应同时查看覆盖率/重扫/扫描年龄、跟踪真实可见性/距离和碰撞越界结果。随机场景的物理可行性和有限时域专家可行性不构成全局保证。

全域螺线约2000 km，科考船按12节走完需80–90小时；演示保留真实速度，倍速只是回放。当前未模拟燃油，没有承诺所有海格固定时限重访，也未穷举整条多小时船路的动力学状态。照片和二维示意模型的具体来源见[素材说明](figs/ATTRIBUTION.md)。

Git仅跟踪源码、文档、测试和必要授权素材；数据、权重、影片、TensorBoard日志及缓存全部忽略。克隆后需重新采集/训练，仓库不包含已验证的本地模型权重。JSON旁的`.replay-index.json`是可丢弃目录索引，大型旧回放在后台建立索引，避免首次历史列表阻塞界面。

覆盖训练诊断及实际局限见 `outputs/diagnostics/coverage-learning-diagnosis.md`；DAgger训练混合数据在回放中有明确标注，不能当作纯网络部署效果。
