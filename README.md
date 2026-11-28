# Area-coverage-and-target-tracking

单架固定翼无人机、单艘科考船、300 km × 300 km二维海域，使用**两套独立启发式专家、两套独立数据集、两套独立CNN+GRU策略网络**实现持续区域覆盖与移动目标跟踪。基于Gymnasium，原生JavaScript海图支持实时显示、JSON历史回放和真实MP4录屏，Matplotlib提供离线绘图/视频导出。

## 功能与任务边界

- 覆盖：随机选取10–20 km尺度的目标子区域，在其边界随机初始化UAV并筛选物理可行性；右侧安装的3 km扇形雷达扫描海面，维护首扫、重扫和扫描年龄。
- 跟踪：船舶沿全域螺线参考航路恒速前进，遇岛屿以A*规划绕行，有限舵角/舵速模型实际推进。UAV在距目标1.6–2.4 km的可行位置初始化，以可偏转前视扇形持续跟踪。策略始终获得船舶真实状态，视场内可见性另行计算。
- 雷云只限制UAV；岛屿只限制船舶，UAV可以飞越岛屿。岛屿不计入海面覆盖面积。
- 覆盖的正引力**只存在于目标子区域海面**；区外为−1，雷云及飞行安全膨胀区为−2。局部场以外的无效查询填0并提供有效掩码，不表示吸引力。
- 全域飞越栅格随访问时间渐淡，雷达扫描年龄单独展示；默认60秒流星尾迹，可关闭或切换完整轨迹。实体模型按运动方向旋转，雷达距离保持真实比例，图标保留最小可见像素。
- 正式流程自动执行专家预检、并行采集、分组划分、两网络训练、独立闭环测试及最新权重前端录屏；状态/错误/失败轨迹均落盘。

有限时域仿真验收不构成任意场景或无限时间的安全证明。历史测试中覆盖纯网络出现过碰云，因此纯网络与显式物理约束系统分别测试和标记。训练误差下降不能替代闭环任务验收。

## 安装与入口

Python3.10+，Windows使用Git Bash；GPU训练需安装与本机驱动匹配的PyTorch CUDA版本。

```bash
python -m pip install -r requirements.txt
npm install --no-save playwright
npx playwright install chromium
bash run.sh formal
```

自动录像优先复用已安装Chrome/Edge，也支持Playwright Chromium。没有浏览器时，可加`--no-browser-recording`运行仿真训练；自动前端录屏需要补齐浏览器。所有Python仿真、专家、训练、路径、端口及录像超参数集中于`run.sh`，经CLI传给`main.py`；末尾参数可以覆盖入口默认值。

```bash
# 默认正式配置：每任务1000个成功回合，按场景组8:2划分，另做独立闭环测试
bash run.sh
# 后台/自动运行：关闭自动打开浏览器和完成后常驻服务
bash run.sh formal --no-open-browser --no-keep-server
# 单任务；同一协议中断后直接重复命令可续采集、从最近完整epoch恢复
bash run.sh formal --task tracking --no-open-browser
# 下一次运行使用独立实验目录（也可设置环境变量 EXPERIMENT_NAME）
bash run.sh formal --experiment-name comparison-01 --no-open-browser --no-keep-server
# 保留的分阶段接口
bash run.sh collect --task coverage
bash run.sh train --task tracking
bash run.sh expert-validate --no-open-browser --no-keep-server
bash run.sh serve
python main.py --help
python -m tensorboard.main --logdir tf-logs --host 127.0.0.1 --port 6006
```

默认界面http://127.0.0.1:8787，TensorBoard http://127.0.0.1:6006。默认输出`outputs/formal`，指标`tf-logs/formal/{coverage,tracking}/formal-1000`。正式默认：覆盖10800秒、跟踪1800秒；采集8进程、展示帧间隔60步；20epochs、batch16、TBPTT长度128、学习率0.0003、验证比例0.2、GRU128、训练线程4、批内加载线程4。预检种子7100/7101；独立测试覆盖960000–960007、跟踪970000–970007。

`--experiment-name comparison-01`将本次产物与缓存保存到`outputs/comparison-01/`，TensorBoard保存到`tf-logs/comparison-01/{coverage,tracking}/formal-1000/`；名称必须是单个合法目录名，支持中文，禁止路径分隔符、`.`、`..`及Windows保留名称。`--output-dir`与`--tf-log-dir`指定各自的父目录，`--run-name`仍控制任务内的TensorBoard运行子目录。`run.sh`默认传入实验名`formal`；直接调用`main.py`时省略实验名会保留原有路径规则。该参数仅影响下一次启动，不迁移已有产物，也不改变正在运行的任务。

`--episodes`是**每任务成功回合数**。每回合完整观测、专家标签和实际动作均保留；稀疏展示帧不降低物理积分或训练采样频率。无法初始化、规划失败、物理失败和任务未完成分别保留，只有成功清单中的NPZ进入训练。正式流程不使用旧数据或warm start；同源代码/配置/预算/种子协议可续跑，协议不一致必须使用新输出目录。

数据按`scene_group_id`作8:2分组；1000个独立场景组对应800训练/200验证回合，实际划分写入`dataset_split.json`。每轮用于计算损失的子集称为验证集，独立测试场景不参与归一化、训练或权重选择。最终影片**仅加载本轮最后一轮`last.pt`**，按SHA256冻结，不回退`best.pt`或专家动作。覆盖影片默认公开启用`forecast`物理约束，跟踪使用纯网络；覆盖纯网络结果同时保留。可通过`--formal-coverage-safety-layer none`明确改为纯网络验收。

## 数学模型

所有计算使用米、秒、弧度，x向东、y向北，航向从+x逆时针。飞机和船均恒速，动作只改变方向，不能原地停止或跳点。

### 固定翼和船舶

固定翼状态为$(x,y,\psi,\phi)$，协同转弯与有限滚转响应为：

$$
\dot x=V_a\cos\psi,\quad \dot y=V_a\sin\psi,\quad
\dot\psi=\frac{g}{V_a}\tan\phi,\quad
\dot\phi=\operatorname{clip}\left(\frac{\phi_c-\phi}{\tau_\phi},-\dot\phi_{max},\dot\phi_{max}\right).
$$

$$
|\phi|\le\min\left(\phi_{max},\arctan\frac{V_a^2}{gR_{min}}\right),\quad
|\dot\psi|\le V_a/R_{min},\quad V_a=220/3.6\;\mathrm{m/s},\quad R_{min}=1000\;\mathrm m.
$$

船舶状态$(x_s,y_s,\psi_s,r,\delta)$采用一阶Nomoto模型：

$$
\dot x_s=V_s\cos\psi_s,\quad \dot y_s=V_s\sin\psi_s,\quad
\dot\psi_s=r,\quad T_r\dot r+r=K\delta,\quad
\dot\delta=\operatorname{clip}((\delta_c-\delta)/\tau_\delta,-\dot\delta_{max},\dot\delta_{max}).
$$

$V_s=12$节；舵角、舵速和转弯半径均有限制。A*输出航路，路径跟随器输出期望航向，再由上述模型积分。决策步长1秒、物理检查0.1秒，内部RK4最大步长0.05秒。速度参考彩虹系列和科考船公开资料；转弯/舵机系数为明确列出的仿真参数，未声称完成真实机船辨识。详见[运动学说明](docs/kinematics.md)。

雷达指令$\alpha_c$先按任务机械偏角限幅，每个物理子步$h=0.1$s执行转速限制：

$$
\alpha_{n+1}=\alpha_n+\operatorname{clip}(\alpha_c-\alpha_n,-\omega_{radar}h,+\omega_{radar}h).
$$

点$q$命中的条件为$\|q-p_a\|\le3000$且$|\operatorname{wrap}(\operatorname{atan2}(q_y-y_a,q_x-x_a)-\psi-\beta-\alpha)|\le\pi/6$。覆盖安装角$\beta=-\pi/2$、偏角限幅±60°；跟踪$\beta=0$、限幅±120°，最大偏转速度30°/s。按实际扇形与物理子步判定命中。

### 局部引力–斥力场

覆盖格距上次扫描的时间为$a$，信息失效周期$T=3600$s，衰减形状$k=3$。令$z=\operatorname{clip}(a/T,0,1)$：

$$
F(a)=1-2\frac{e^{-kz}-e^{-k}}{1-e^{-k}}.
$$

刚扫描$F=-1$，经过$T$恢复$+1$，未扫描也为$+1$；$T$是信息价值失效周期，**不是重访保证**。覆盖子区域海面以外固定−1；雷云、安全膨胀区和全域外为硬斥力−2，优先级最高。只计算UAV周围6 km半径内的33×33局部场，全域历史仅保留扫描时间戳。覆盖质量另用ROI64×64海面格统计，二者用途和分辨率不同。

跟踪场按船舶距离构造局部吸引值，并叠加相同硬障碍；不沿用覆盖任务的时间老化或把全域海面当成覆盖目标。EGO-Planner启发了按需构造局部信息的思路，**上述有符号指数曲线和固定翼启发式为本项目工程设计，并非复现EGO-Planner算法**。

### 专家目标与约束

覆盖专家从有限航向候选和雷达偏角候选中最大化：

$$
J_c=w_g\,d\cdot d_{goal}+w_s U_{sector}+w_p F_{path}
-w_b C_{outside}-w_o C_{hard}-w_t|\operatorname{wrap}(\psi_c-\psi)|.
$$

$U_{sector}$汇总实际可达扇形内的场值，并含目标对准收益、雷达转动惩罚；$F_{path}$为预测采样点$\min(F,0)$的均值，$C_{outside}$为到ROI的归一化距离。候选预测遵守最大转率，选中动作再进行有限滚转的精确避障预测。全局公开扫描目标依据未扫优先、年龄和距离产生，扫描完成或超时后重新评选；有实质接近时延长服务期限。专家和网络都能从相同obs获得该目标。该机制不能证明完全消除局部环路。目标区外−1为软偏好，固定翼转弯可以短暂使用区外缓冲。

跟踪专家在移动目标参考系中维持期望绕飞半径$r_d=2000$m。设$e=(p_a-p_s)/\|p_a-p_s\|$，$t$为顺时针切向，

$$
 n=\frac{t-k_r(\|p_a-p_s\|-r_d)e/r_d}{\|t-k_r(\|p_a-p_s\|-r_d)e/r_d\|},\quad
 \lambda=-v_s\cdot n+\sqrt{(v_s\cdot n)^2+V_a^2-\|v_s\|^2},\quad
 d=\frac{v_s+\lambda n}{V_a}.
$$

雷达偏角对准船舶并补偿相对视线角速度。运动学等式、转弯/舵机/雷达速率、雷云和边界为约束；距离/视场为跟踪目标和失败判据。实现对滚转延迟作预测补偿，近障碍使用受限斥力和几何避让。有限候选启发式不是求得全局最优的非线性规划器。

设全域为$\mathcal W$，雷云中心/半径为$(c_j,r_j)$，岛屿为$(i_k,\rho_k)$，实体安全膨胀量为$m_a,m_s$。飞行与航行硬约束分别为：

$$
p_a(t),p_s(t)\in\mathcal W,\qquad
\|p_a(t)-c_j\|\ge r_j+m_a,\qquad
\|p_s(t)-i_k\|\ge\rho_k+m_s.
$$

岛屿不约束UAV，雷云不约束船舶；覆盖ROI边界不等同全域硬边界。避障的提前引导软缓冲与上述硬膨胀量是两个参数，缩小软缓冲不能放宽硬约束。实际实现逐物理子步检查轨迹段，并在前向预测中额外考虑圆弧与弦的偏差。

场覆盖专家的软缓冲由`field_coverage_obstacle_buffer=200`米单独配置，上限仍为`expert_obstacle_buffer`，以免提前绕飞距离超出3 km扫描包络；跟踪仍使用其自身距离预留。通用安全引导原候选全部失败时，才补查当前航向及原请求航向附近的显式配置候选；仍无解才使用`expert_safety_fallback_heading_samples=145`个全圆方向，按与通用修正方向的角差逐条执行相同物理预测。网格为$\psi_m=\psi-\pi+2\pi m/N$（$m=0,\ldots,N-1$），以当前航向为参考，保持与机体系网络输入的旋转一致性。有限离散搜索仍可能无解，失败如实保留。真实云簇回归见[实施记录](docs/formal-training-validation.md)。

显式`forecast`层独立预测网络动作未来30秒的物理轨迹：安全时原样透传，冲突时公开记录几何修正，再检查修正轨迹。它不调用覆盖/跟踪任务专家；通用修正不可行时，额外搜索配置中的转向候选及上一条安全指令，每条都从当前状态重新进行物理预测；记录`constraint_version`及`candidate_repairs`。原始网络动作、约束后动作、实际动作及干预标志分别保存。它有有限预测视野，不构成全局安全证明。

## 观测、网络与损失

专家`act(obs)`与网络`act(obs)`接收同一字典，返回同一float32动作。仿真直接提供真实位置/速度、最近障碍几何、雷达状态和公共目标，无需感知模型。

| 字段 | 覆盖 | 跟踪 | 内容 |
|---|---:|---:|---|
| `vector` |111|134|自身运动、雷达/边界/云/岛槽位及有效位、目标相对位置/速度等 |
| `potential` |4×33×33|4×33×33|有符号场、任务海面、硬障碍、有效查询掩码 |
| `field_context` |16|16|场半径/周期/形状、安装角/偏角限制/转速、目标等 |
| `action` |3|3|世界XY单位期望方向 + 归一化雷达偏角 |

环境将非零XY指令归一化为期望速度$V_a\,a_{0:2}/\|a_{0:2}\|$，近零指令保持当前航向；实际速度为$V_a(\cos\psi,\sin\psi)$，雷达指令为限幅角乘$a_2$。环境统一执行运动学限制。默认在网络内部旋转至机体系，屏蔽上一拍命令以减少复制捷径，公共接口保持世界坐标。

每个网络独立拥有：局部场CNN(4→16→32→32→128)、运动矢量MLP、场上下文MLP(16→64→64)、单层GRU128、方向和雷达残差头。覆盖默认对云集合共享编码并池化，跟踪保留固定槽位编码；方向/雷达几何参考只使用公开观测。GRU保留历史，但不能凭空恢复从未观测的信息。

序列BC采用真实回合、因果TBPTT：每回合reset，chunk间传递并detach隐藏状态，padding不计损失，$T+1$观测对应$T$动作，禁止错一拍。默认优化：

$$
\mathcal L_{BC}=\frac{1}{N}\sum_{t=1}^N\|\pi_\theta(o_{\le t})-a_t^E\|_2^2,\quad
\operatorname{MSE}=\mathcal L_{BC}/3,\quad
\operatorname{MAE}=\frac{1}{3N}\sum_t\|\pi_\theta(o_{\le t})-a_t^E\|_1.
$$

可选纠正加权/执行感知损失默认关闭；前者改变Loss权重，后者增加辅助项，原始MSE/MAE始终按共同动作不加权计算。归一化仅拟合训练集，同一`scene_group_id`不能跨训练/验证。只上报适用的6条TF曲线：`Train/Loss`、`Validation/Loss`、两子集的`Action_MSE`和`Action_MAE`。当前确定性连续策略不伪造离散准确率、NLL、KL或熵。epoch0提供未更新权重的基线；之后Train累计更新过程中各chunk的误差，Validation在当轮更新完成后用固定权重计算。

`best.pt`按验证MSE保存，候选包含epoch0；`last.pt`保存最新完成epoch。formal最终测试和影片固定使用last；分阶段训练导出的`<task>_policy.pt`采用best，普通evaluate按显式checkpoint或selected/best规则执行。完整epoch边界续训恢复Adam、RNG、划分、归一化和历史，批内加载有界；普通warm-start恢复模型及归一化缓冲区、重新初始化优化器，不等同续训。普通CUDA可能有浮点非确定性，严格可复现需一致环境及确定性设置。按配置的chunk数或时间间隔、批加载及阶段边界更新`training-status.json`（默认50chunk或30秒），每个完整epoch原子更新checkpoint和`history.json`。

## Python与HTTP接口

```python
# 项目内部采用src目录导入；独立调用时加入该路径
import sys
sys.path.insert(0, 'src')
from config import Config
from env_coverage import CoverageEnv
from expert_field import FieldCoverageExpert
cfg = Config(potential_enabled=True)
env = CoverageEnv(cfg, render_mode='json')
obs, info = env.reset(seed=2000)
expert = FieldCoverageExpert(cfg)
expert.reset()
action = expert.act(obs)
obs, reward, terminated, truncated, info = env.step(action)
frame = env.render()  # 真实状态字典；通过VisualServer.publish发送
scene = env.scene    # 可通过reset(options={'scene': scene})精确回放初始化
loaded = env.summary()
env.close()
```

`TrackingEnv`/`FieldTrackingExpert`使用相同接口；`NeuralPolicy.from_checkpoint(path, device='cpu')`得到网络actor。`collect_expert_episodes(task, config, output_dir, episodes=..., seed=..., workers=..., attempt_multiplier=..., replay_every=..., progress_every=..., require_task_success=True)`返回权威manifest。`train_policy(task, manifest, model_dir, tf_log_dir, ...)`训练单任务；`record_policy_replay(server_url, replay_path, output_dir, fps=24, width=1440, height=900, duration_seconds=30)`自动调用与UI相同录屏功能。

| 路由 | 用途 |
|---|---|
| `/events` |SSE场景、真实帧、进度和正式运行状态 |
| `/api/replays` |JSON历史目录（大文件后台索引） |
| `/api/visual-config` |入口显示配置 |
| `/api/recording-config` |尺寸、FPS、码率、时长、上传限额、当前保存目录 |
| `/api/run-status` |当前采集数量、训练阶段/epoch/损失 |
| `POST /api/recordings` |有界原始视频上传，真实转码为H.264 MP4并保存来源元数据 |
| `/outputs/...` |当前输出根目录内的回放/影片下载 |

前端操作集中在顶部的数据、视图、回放与录制菜单。所有可视化开关位于“视图”，包括经过网格、雷达覆盖年龄、引斥场色块、相邻栅格差分方向箭头和轨迹；支持局部引斥场近景。默认白色主题，可切换深海蓝并记住选择，两套主题均区分无人机、船舶、云区、岛屿及各图层。图中的大虚线圈表示局部引斥场采样范围（默认半径6km），雷达视场则使用单独的物理距离（默认3km）。

在顶部“录制”菜单打开“录制演示”开关，可保存完整历史回放或当前实时场景；关闭开关可提前停止，影片真实记录停止时刻。到时自动保存到当前实验的`outputs/<experiment-name>/recordings/`，界面显示保存路径与下载链接。浏览器支持时直接录MP4，否则先捕获WebM再由FFmpeg转码，不改扩展名冒充MP4。自动录像含任务、模型SHA、控制模式、仿真时间和真实任务结果；转换失败会在UI显示错误。默认30秒、1440×900、24FPS，回放倍速只影响显示时间，不改变仿真速度。全域螺线路程约2000km，12节完成约80–90小时；30分钟跟踪演示展示其中一段。

## 项目结构与产物

```text
main.py                     CLI、阶段编排、服务启动
run.sh                      全部显式超参数
requirements.txt
src/
  config.py dynamics.py     配置与机船运动学
  scenario.py navigation.py 随机筛选、SceneSpec、船舶A*航路
  env_base.py               Gym公共基类/物理推进/render
  env_coverage.py env_tracking.py
  observations.py potential_field.py
  expert_field.py           场感知覆盖/跟踪专家
  expert_coverage.py expert_tracking.py
  policy_field.py policy_coverage.py policy_tracking.py
  policy_constraints.py     显式可选物理预测约束
  rollouts.py collection.py JSON/NPZ轨迹、并行续采集
  datasets.py train.py      分组、序列、BC、TF、续训
  formal_run.py             正式协议/状态/最新权重测试与录像
  visual_server.py browser_recording.py rendering.py
visual/                     JS海图、实时/历史、录屏
figs/                       本地彩虹UAV/科考船/海洋素材及许可
examples/ tests/ docs/       示例、回归测试、设计与验证记录
outputs/formal/
  formal-protocol.json      运行前冻结的训练/测试协议
  formal-status.json        实时阶段/数量/损失/错误
  formal-report.json        本轮实际专家/训练/测试/影片证据
  datasets/{coverage,tracking}/
    accepted-manifest.json  成功清单、拒绝/失败/未使用记录、源SHA
    episode-*.json/.npz     完整无损训练数据与稀疏展示帧
  models/{coverage,tracking}/
    best.pt last.pt history.json training-status.json
  frozen-latest/            最后权重不可变SHA快照
  closed-loop-tests/        独立场景，纯网络/约束模式分别保存
  preflight/ replays/       专家预检及最新网络实际回放
  recordings/              MP4及来源/解码证据JSON
  cache/                    有界临时缓存
tf-logs/formal/{coverage,tracking}/formal-1000/
```

JSON中的训练数组以zlib+base64无损编码，可用`src.rollouts.restore_training_cache`恢复NPZ。展示帧通过`frame_action_indices`对应原始动作，不能把稀疏帧误当训练抽样。Git只上传源码、文档、测试、必要素材；数据、模型、影片、TF和缓存全部忽略。克隆仓库后需重新生成，不能依赖本机旧路径。

## 验证与参考

```bash
python -m pytest -q
python -m examples.plot_kinematics --output outputs/kinematics/kinematics.png
```

正式本轮结果以运行后生成的`outputs/formal/formal-report.json`为准，未完成前不将开发结果当成本轮验收。当前修复版专家的42场景开发验证全部通过：34个覆盖场景安全完成10800秒、栅格覆盖率1（重扫比例0.9060–1），8个跟踪场景完成1800秒且全程可见；全量373项测试通过、1项跳过。1000+1000回合存储估计约67 GB，不含失败与测试。详细性能、真实失败、修复及已知边界见[正式实施记录](docs/formal-training-validation.md)。之前的纯覆盖网络失败与显式约束验证保留在[历史验证](docs/local-potential-field-validation.md)。本机旧缓存因自动执行审查拒绝递归删除，已移入被Git忽略的`.run-archive/20261128-preformal/`；正式流程在新建的空输出目录启动，不读取隔离数据。

数学/初始化/文献详见[算法设计](docs/superpowers/specs/2026-11-27-maritime-imitation-design.md)、[局部场设计](docs/superpowers/specs/2026-11-27-local-potential-field-design.md)、[正式方案](docs/superpowers/specs/2026-11-28-formal-training-and-recording-design.md)。参考[Dubins](https://msl.cs.illinois.edu/~lavalle/planning/node821.html)、[EGO-Planner](https://arxiv.org/abs/2008.08835)、[持续覆盖规划](https://arxiv.org/abs/1908.09236)、[Fossen船舶模型](https://www.fossen.biz/html/marineCraftModel.html)；具体方法与论文贡献的区别在设计文档明确说明。图片及二维模型许可见[素材说明](figs/ATTRIBUTION.md)。
