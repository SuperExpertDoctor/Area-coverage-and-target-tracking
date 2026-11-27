# 海上模仿学习实施方案与执行记录

2026-11-27，版本2.0。对应[设计规范](../specs/2026-11-27-maritime-imitation-design.md)。用户已授权实施、运行和修复；本目录没有Git仓库，原地实施。旧计划已归档outputs/docs-baseline/implementation-v1.md。

## 1. 总体依赖

    运动学 → 几何/A*/场景 → 两Gym环境 → 两专家闭环
                                         ↓
                         独立JSON采集/成功清单/场景划分
                                         ↓
                         独立CNN/MLP/GRU训练与TensorBoard
                                         ↓
                              同场景网络闭环与专家对比

JS与端口服务消费同一render帧，不维护第二套运动方程。MP4消费同一JSON记录，避免显示和评估用不同轨迹。

## 2. 工程布局

| 文件 | 职责 |
|---|---|
| main.py / run.sh | CLI编排 / 全部可调超参数与输出路径 |
| src/config.py | 可发现平面Config、范围校验、物理配置、JSON/hash |
| src/dynamics.py | 已保留的有限响应UAV与Nomoto船舶模型 |
| src/geometry.py | 距离、圆障碍、线段与扇形、海面mask |
| src/navigation.py | A*、螺线、绕岛、平滑、前向船路跟随 |
| src/scenario.py | 两套随机reset、筛选、SceneSpec和指定场景回放 |
| src/observations.py | 111/134维共同输入、3×64×64地图、spaces/schema |
| src/env_base.py | Gym公共时序、物理推进、雷达、安全、render和失败信息 |
| src/env_coverage.py / env_tracking.py | 独立任务统计与环境类 |
| src/expert_coverage.py / expert_tracking.py | 独立启发式，复用有限物理候选检查 |
| src/policy_coverage.py / policy_tracking.py | 独立CNN/MLP/GRU与状态化推理包装 |
| src/rollouts.py | 同步标签、完整JSON、无损数组、NPZ派生缓存 |
| src/datasets.py | 成功清单、整场景划分、回合序列和padding |
| src/train.py | 连续TBPTT、真实指标、best/last和版本化权重 |
| src/collect_dagger.py / validation.py | 训练布局再标注 / 独立闭环场景套件 |
| src/rendering.py | 真实帧Matplotlib MP4/PNG |
| src/visual_server.py | 只读HTTP、SSE、JSON历史目录 |
| visual/* | JS海图、加载式模型、实时/回放与指标界面 |
| figs/* | 真实下载照片、二维SVG、许可与来源 |

20个src业务模块和1个根入口；不为满足原计划文件数添加空文件。采集和评估由main的独立task分支编排，共用循环但不混合任务数据。

## 3. 实施步骤及实际验证

### A. 保留物理基线

- [x] 原dynamics移到src，保持状态方程和原28项解析测试。
- [x] 真实速率、曲率、bank/舵速、跨π、非法输入和分步一致性检查。
- [x] 原诊断产物迁入outputs/kinematics；示例默认输出也修正。

### B. 几何、船路和场景

- [x] 圆障碍、海面mask和扇形。
- [x] 禁对角穿缝的A*、转向安全膨胀、局部绕岛和平滑。
- [x] 全域螺线与真实Nomoto跟随，强制绕岛物理回归。
- [x] 两套随机初始化、规则筛选、有上限重采样、精确SceneSpec回放。
- [x] 每任务10随机种子检查；全参考路线几何验证。

### C. 两Gym环境及共同接口

- [x] Gym reset/step/terminated/truncated及结束后调用检查。
- [x] 同一真实模型逐物理步推进、雷达有限转速、扫描年龄与安全事件。
- [x] 观测维度、历史快照、岛/云角色和前视/侧视区分。
- [x] 两环境通过Gymnasium checker与有限JSON检查。

### D. 两启发式专家

- [x] 右侧扇形对齐的顺时针扫描环、漏扫补扫与全区重新扫描。
- [x] 恒速相对运动绕飞，真实雷达可见性验证。
- [x] 修复远云覆盖近云避让、预测激活太晚和局部补扫驻留问题。
- [x] 覆盖0/7/7100各10800s全覆盖、安全；跟踪多个1800s回合全程可见。
- [x] 失败轨迹留档，不以修改物理限制或隐藏专家回退掩盖失败。

### E. JSON采集和训练数据

- [x] 分任务目录，T+1观测对T个同拍专家原始命令。
- [x] 实际执行方向单独记录；JSON完整保存、NPZ可无损重建。
- [x] 采集前作废旧缓存；失败/丢目标示范默认拒绝。
- [x] 成功manifest权威限定数据，拒绝未完成、重复和缺失条目。
- [x] scene_group_id整场景训练/验证，不按时间步随机泄漏。
- [x] 小样本真实采集；覆盖延长到5400s以包括重访阶段。

### F. 两模型与序列训练

- [x] 两独立网络、GRU真实回合reset、因果推理和checkpoint回放。
- [x] 连续TBPTT、hidden.detach、padding掩码、只编码当前地图chunk。
- [x] 机体系方向头可表达任意方向；修复世界方向相关性和浮点Box溢出。
- [x] 训练统计不使用验证集；标准差下限避免小方差放大分布外状态。
- [x] 只记录确定性连续BC的Loss/MSE/MAE，best与last都保存。
- [x] 真实跟踪训练及独立闭环回归；17个诊断测试1800s全程可见。
- [ ] 最终覆盖训练和两个生产权重闭环验收，以最终验收报告更新。

### G. 端口JS、素材和影片

- [x] SSE现场状态、只读历史目录、路径穿越保护与慢客户端队列。
- [x] 真实照片下载，二维SVG模型加载并按真实速度旋转。
- [x] 任务区/全域/跟随，回放倍速、进度、任务/策略切换和JSON导入。
- [x] 桌面/移动端与浏览器零错误检查，修复时间轴被面板挤出屏幕。
- [x] Matplotlib使用真实frames导出H.264 MP4/PNG。
- [x] 专家与已验证跟踪模型影片、最新版JS界面实录；覆盖网络影片按真实未达标结果保留。

## 4. 审查发现与处置

| 发现 | 原因/修复 |
|---|---|
| 覆盖补扫2750s后无安全候选 | 多云顺序和有限转弯预留不足；近障碍最后处理，提前真实模型预测 |
| 100%后平均age继续极高 | 局部补扫驻留；完成后重新全区扫描，保留长尾指标 |
| 低MSE跟踪仍丢目标 | 世界动作相位和输入分布放大；机体系动作/特征及归一化下限 |
| 网络输出1.000000119 | 浮点旋转末端归一化与Box限制；训练/推理一致 |
| 同名失败遗留旧NPZ | reset前作废派生缓存 |
| 重复采集混入旧批次 | 权威accepted-manifest代替无界目录发现 |
| 训练/验证相邻样本泄漏风险 | 按完整scene_group_id划分 |
| standalone evaluate场景不一致 | 读取保存的专家SceneSpec与配置，并比较scene_id |
| 目标曾丢失但未终止也收为专家 | 根据first_loss_time拒绝，不只检查terminated |
| 首拍规划失败导出视频掩盖原因 | 少于两帧不导出MP4，仍保留失败JSON |
| 前端列入pytest合成回放 | 遍历前剪枝cache/logs/隐藏目录 |
| 继续训练改变seed导致历史训练场景进入验证 | 检查checkpoint分组固定归属，禁止交叉 |
| 继续训练覆盖自己的输入best.pt | 不可变初始checkpoint副本和SHA256来源 |
| 合并manifest相对路径换了基准目录 | 按原清单目录解析后写绝对路径 |
| 显式checkpoint缺失静默换模型 | 显式输入缺失报错，仅默认路径保留发现兼容 |
| 转换后的模型覆盖原始三通道源 | 转换器也保存原始不可变SHA256副本，分别保留原始与转换后来源 |
| 实时训练混合控制误标为网络推理 | SSE与前端同时保留behavior/beta |
| 原文HDF5/根源码/仅Matplotlib过时 | 改为JSON+NPZ、src与JS+Matplotlib双显示消费端 |

## 5. 命令和验收口径

在Git Bash中：

    bash run.sh expert
    bash run.sh collect
    bash run.sh train
    bash run.sh evaluate
    bash run.sh all
    bash run.sh serve

自动验证加 --no-open-browser --no-keep-server；不需前端时加 --no-visualize。每一阶段可用--task coverage或tracking独立运行。TensorBoard：tensorboard --logdir tf-logs --port 6006。

验收必须同时看数据与演示：覆盖首次完整时间、最终覆盖率、扫描年龄和重扫率；跟踪可见比例、首次丢失、实际距离；共同碰撞/越界、速度/曲率。没有人工数据不报告人工对比。小批次只证明已验证范围，不推导任意场景长期能力。

没有硬最大重访上界、没有全80小时船路逐时域动力验证、没有近目标复杂云区穷举，这些是明确保留的物理/算法边界，不伪装为已经解决。

## 6. 最新需求迭代

- [x] 全世界250m稀疏飞越格，按最后访问仿真时间渐淡；与雷达扫描年龄分层。
- [x] 默认60秒流星尾迹，支持完整历史及关闭；倒拖重建，未显示未来。
- [x] 真实雷达距离保持3km，参考尺寸和最小像素缩小机船图标，并按真实速度旋转。
- [x] Config全部133字段及每个CLI选项列在run.sh，静态清单和实际功能均检查；跟踪角度/距离余量参与初始化。
- [x] expert-validate：预注册30随机覆盖、24边界起点、10跟踪；source/config哈希绑定，失败与剔除单独记录。
- [x] 修复覆盖2000候选遗漏；54个覆盖案例初版全部3小时100%首扫、安全。
- [x] 保留跟踪5003反例；仅收紧额外软引导缓冲，真实闭环红→绿，硬安全边界和30秒物理预测不变。
- [x] 最终版本同64案例全部重新复验通过，0剔除、0失败；结果见outputs/diagnostics/expert-randomized/suite-summary.json。
- [x] 端口目录改为单后台索引大型旧JSON，sidecar可丢弃；字段顺序不影响回放识别。
- [x] JS与Matplotlib同样遵从轨迹/年龄开关，双实体流星窗口精确插值；独立审查3项问题全部修复。
- [x] .gitignore排除数据/权重/影片/日志/缓存，保留必要授权素材；初始化main和指定origin。
- [x] 按用户明确授权推送GitHub，首提交84e225c与远端main一致；最终验证记录随后补充提交。

覆盖网络额外审查：12回合64800动作由相同完整观测历史逐位复现，adapter与整段输出误差仅浮点级；没有以“最近256步reset”冒充完整教师历史。当前网络对前一命令的依赖远强于扫描历史，属于学习与数据覆盖不足的已观察证据；未把低Loss或训练混合轨迹作为闭环成功。后续中途fork DAgger需要完整prefix warm和分支loss mask，尚未实现，详见验证状态。
