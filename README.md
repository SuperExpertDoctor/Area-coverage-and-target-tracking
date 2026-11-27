# Area-coverage-and-target-tracking

单固定翼UAV、单艘科考船、300 km × 300 km二维海域；两个独立GRU模仿学习模型。云只限制飞机，岛只限制船舶；覆盖用右侧扇形，跟踪用可偏转前视扇形。

详细算法、等式约束、初始化和接口见[设计文档](docs/superpowers/specs/2026-11-27-maritime-imitation-design.md)，实施步骤和问题修复见[实施方案](docs/superpowers/plans/2026-11-27-maritime-imitation-implementation.md)。

当前跟踪网络已在7个独立1800秒回合保持目标全程可见。覆盖专家能完成持续扫描；覆盖网络仍处于实验阶段，现有小批量BC/DAgger权重尚不能稳定完成闭环覆盖，不能仅凭验证损失下降判为达标。具体实验结果见[验证状态](docs/validation-status.md)。

## 安装与运行

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
bash run.sh train --task tracking          # 单任务单独训练
bash run.sh dagger --task coverage --dagger-beta 0.8 --run-name coverage-dagger-1
# 继续训练时可用 --warm-start --checkpoint-path ...；只补采用 --dagger-collect-only
```

默认界面为[http://127.0.0.1:8787](http://127.0.0.1:8787)，任务完成后保持服务直到Ctrl+C。自动运行可加`--no-open-browser --no-keep-server`；不需要端口界面时加`--no-visualize`。独立回放也可直接：

```bash
python main.py --mode serve --output-dir outputs --tf-log-dir tf-logs --port 8787
tensorboard --logdir tf-logs --port 6006
```

入口参数帮助：`python main.py --help`。仿真/专家全部参数位于`run.sh`的`SIM_CONFIG`，通过`--config-json`整体传入Python；不是隐藏在另一份配置文件中。

## 产物

| 路径 | 内容 |
|---|---|
| src/ | 物理、环境、专家、两网络、采集、训练与服务源码 |
| visual/ | 原生JS海图、实时SSE、历史回放 |
| figs/ | 下载的彩虹CH-4、科考船、海洋照片；加载式二维SVG模型及来源许可 |
| outputs/datasets/{coverage,tracking}/ | 完整JSON示范、派生NPZ、权威accepted-manifest |
| outputs/models/{coverage,tracking}/ | best.pt、last.pt、任务名导出模型、场景划分和训练历史 |
| outputs/replays/ | 同场景专家/网络JSON，前端可直接回放 |
| outputs/videos/ | H.264 MP4及最终帧PNG |
| outputs/training/、outputs/reports/ | 训练报告和阶段报告 |
| outputs/validation-report.json | 两任务专家、训练、网络闭环的合并实际结果 |
| outputs/diagnostics/、outputs/cache/ | 诊断记录与缓存 |
| tf-logs/ | 分任务/分实验的TensorBoard事件文件 |

JSON训练数组使用可无损解码的zlib+base64，不依赖NPZ才能恢复。可用`src.rollouts.restore_training_cache(json_path)`重建训练缓存。演示JSON的frames是完整真实状态轨迹，前端可导入文件；选择策略、拖动进度条、切换任务区/全域/跟随视角。

全世界按250m栅格记录飞机实际飞越位置，最近飞越的格子最深，随仿真时间渐淡；雷达扫描年龄图独立显示。默认显示60秒流星尾迹，可关闭或切为完整轨迹。历史拖动会重建截至当前时刻的记录，不显示未来足迹。远景网格线自动抽稀，底层访问格大小不变。

雷达始终按真实3km绘制。图标按参考长宽缩放，同时保留最小像素以便辨认；因此全域视角的图标是有最小可见尺寸的符号，并非严格等比例的实机轮廓。`SIM_CONFIG`中的16个`visual_*`参数可控制网格、渐淡、尾迹、参考尺寸、最小像素及默认开关。服务通过`/api/visual-config`把当前入口的显示参数提供给旧回放，不改变其物理数据。

两模型不共享参数。原始动作是二维期望世界速度方向；真实机船位置由有限倾斜/舵机响应模型推进，不能直接跳点。图片/二维模型只影响显示，按真实速度旋转。

## 训练指标与检查

仅记录确定性连续策略适用的6条曲线：Train/Validation的Loss、Action_MSE、Action_MAE。默认Loss是每步XY平方误差和，MSE是每分量均方误差；可选纠正权重启用时Train/Loss记录实际加权目标，MSE/MAE与Validation始终不加权；不虚构离散准确率或概率分布指标。验证集按完整场景划分，同一场景的相邻时间片不会泄漏。

```bash
python -m pytest -q
python -m examples.plot_kinematics --output outputs/kinematics/kinematics.png
```

低验证误差不能替代闭环测试。应同时查看覆盖率/重扫/扫描年龄、跟踪真实可见性/距离和碰撞越界结果。随机场景的物理可行性和有限时域专家可行性不构成全局保证。

全域螺线约2000 km，科考船按12节走完需80–90小时；演示保留真实速度，倍速只是回放。当前未模拟燃油，没有承诺所有海格固定时限重访，也未穷举整条多小时船路的动力学状态。照片和二维示意模型的具体来源见[素材说明](figs/ATTRIBUTION.md)。

Git仅跟踪源码、文档、测试和必要授权素材；数据、权重、影片、TensorBoard日志及缓存全部忽略。克隆后需重新采集/训练，仓库不包含已验证的本地模型权重。JSON旁的`.replay-index.json`是可丢弃目录索引，大型旧回放在后台建立索引，避免首次历史列表阻塞界面。

覆盖训练诊断及实际局限见 `outputs/diagnostics/coverage-learning-diagnosis.md`；DAgger训练混合数据在回放中有明确标注，不能当作纯网络部署效果。
