# 单固定翼海上持续覆盖与移动船舶跟踪：详细设计

版本2.0，2026-11-27。本文替代原实施前设计，区分数学目标、实际启发式和实测结果。旧版归档于outputs/docs-baseline；验收结果以outputs/validation-report.json及验收报告为准。

## 1. 已确认需求

交付两个独立模型：coverage_policy.pt负责持续覆盖，tracking_policy.pt负责移动船舶跟踪。两任务各自初始化、采集、划分、训练、保存权重和维护GRU隐藏状态。复用工具代码不代表共享策略。

| 讨论项 | 最终约定 |
|---|---|
| 数量/空间 | 单架固定翼、单艘科考船，二维质点，不考虑高度 |
| 世界/任务区 | 世界300km×300km；每回合随机10–20km矩形扫描区 |
| 初始点 | 覆盖从扫描区边界任意一点开始；跟踪从已捕获目标的可行绕飞状态开始 |
| 已知信息 | 仿真直接提供位置、速度、距离、边界、操纵状态和障碍物；无感知或估计模块 |
| 云/岛 | 云只限制UAV，岛只限制船舶；UAV可以飞越岛；陆地从覆盖分母排除 |
| 雷达 | 覆盖固定右侧扇形；跟踪前部安装、可左右偏转扇形，有限转速 |
| 运动 | 恒定速率，UAV有限倾斜响应与Dubins曲率约束，船舶Nomoto及舵机响应 |
| 船路 | 全域长螺线参考路线，遇岛屿A*绕行，实际位置只能由物理方程推进 |
| 网络 | 覆盖CNN+MLP+GRU；跟踪MLP+GRU；两者独立 |
| 数据 | 专家和网络同一obs/action协议，JSON历史可回放，NPZ仅为训练缓存 |
| 可视化 | Gym render状态经端口SSE到JS；Matplotlib导出相同真实轨迹的MP4 |
| 目录 | 源码src，前端visual，素材figs，输出/缓存outputs，TensorBoard为tf-logs |
| 入口 | run.sh集中列出全部可调配置和路径，调用根目录main.py |

不模拟风、海流、升阻力、侧滑、复杂雷达信号和遮挡；云下海面仍是扫描目标。GRU用于共同观测历史和专家阶段记忆，不人为删除已有真值制造感知困难。完整船路存入SceneSpec用于环境复现，但UAV专家和网络的act都不能访问未来船路或环境句柄。

两个任务分别验证。“发现目标后切换跟踪”是上层调度，可在两个独立策略之上实现，不将两类标签混成一个网络。没有提供独立人类操作者数据，因此当前比较对象是启发式专家，不能称为人工实测示范。

## 2. 论文依据与采用边界

| 来源 | 采用内容 | 本项目实现 |
|---|---|---|
| Choset/Pignon区域分解 | 分区组织扫描 | 矩形扫描环及8×8目标优先级，不宣称完整复现分解算法 |
| Paull固定翼传感器覆盖 | 真实轨迹与传感器足迹反馈 | 右侧二维扇形、逐物理步更新 |
| Smith持续监测 | 未重访区域代价随时间累积 | 扫描年龄和重复扫描，不保证最大年龄硬上界 |
| Dubins/LaValle、FAA | 恒速曲率约束与协调转弯 | 增加有限倾斜响应，参考几何不能直接当作执行轨迹 |
| 目标跟踪与避障向量场 | 相对运动、绕飞与避障 | 加入真实有限雷达和物理预测，不声称原论文逐项复现 |
| Hart等A* | 图搜索绕障 | 岛屿膨胀、禁对角穿缝、平滑及真实船舶跟随 |
| Fossen/Nomoto | 船舶转向惯性 | 系数为可解释仿真值，未做实船辨识 |
| Ross等DAgger | 网络访问状态上的专家再标注 | 已实现训练场景内再标注；实际补采规模与结果以验收报告为准 |

权重、参数和组合方式为本项目设计。论文思想不自动保证本项目全局最优、无限时域安全或网络性能。

## 3. 坐标、时间和物理等式约束

统一m/s/rad：x东、y北，航向从+x逆时针计正，正bank/rudder/pan表示左侧。屏幕y向下需反转。图标按真实vx/vy旋转，不以期望动作替代朝向。

### 固定翼

状态s_U=(x,y,ψ,φ)：

    x_dot = V_U cosψ
    y_dot = V_U sinψ
    ψ_dot = g tanφ / V_U
    φ_dot = clip((φ_cmd−φ)/τ_φ, −p_max, p_max)
    φ_lim = min(φ_max, atan(V_U²/(g R_min)))

等式约束为上述状态转移和速度模长V_U；不等式为|φ|≤φ_lim、|φ_dot|≤p_max、|ψ_dot|/V_U≤1/R_min。默认220km/h、R_min=1000m、配置bank25°，实际限幅约20.85°，最大转弯率约3.50°/s。

step_velocity仅取二维指令方向，航向误差转换为角速度、倾斜目标，再经响应与RK4推进；零向量保持当前航向，不能悬停。决策1s，物理/安全/雷达更新0.1s，内部RK4≤0.05s。恒定倾斜时为圆弧，变化时保持有限曲率与执行器响应，不把Dubins拼接点瞬时转弯当作物理动作。

220km/h参考彩虹CH-3公开巡航信息；显示采用CH-4照片。不能因此声称已辨识CH-4真实操纵特性。

### 科考船

状态s_S=(x,y,ψ,r,δ)：

    x_dot = V_S cosψ
    y_dot = V_S sinψ
    ψ_dot = r
    T r_dot + r = K δ
    δ_dot = clip((δ_cmd−δ)/τ_δ, −δ_rate_max, δ_rate_max)

默认12节=6.1733m/s、最小半径300m、T=20s、最大舵35°、舵速3°/s；r_max=V_S/R_min，K=r_max/δ_max。step_heading经过角速度反馈和舵机，不直接改航向。匀速表示速率恒定，速度方向可变；参数是仿真假设，不是实船校准。

### 安全约束

UAV对每个云圆：distance≥radius+200m；船对每个岛：distance≥radius+100m；两平台不越世界边界。每物理子步检查线段，并按曲率为真实弧线偏离预留裕度。岛不限制UAV，云不限制船。扫描区边界是目标范围，不是飞行硬边界。

## 4. 实际雷达与覆盖地图

覆盖：R=3000m，半角30°，固定offset=−90°。跟踪：R=3000m，半角30°，前部安装，pan±120°、最大转速30°/s。

    visible = (d≤R) and (abs(wrap(目标方位−机头−offset−实际pan))≤半角)

跟踪pan目标为clip(wrap(目标方位−机头),±pan_limit)，实际偏角按最大转速推进；不能将目标角瞬移为实际角，也不能跨±π跳转。

覆盖地图3×64×64：sea_mask、ever_seen、age/3600。只统计实际扇形命中的格中心，逐0.1s更新；岛格排除；未扫格age=t，已扫格age=t−last_seen；reset在t=0真实扫描一次。第0行在南侧。100%指有效格中心全部命中，不是连续海面完备性证明。

第一次100%后继续运行。重扫需脱离扫描至少60s后重新进入，避免将连续照射当重复重访。JS热图下采样16×16，训练保留完整64×64。

## 5. 随机初始化与SceneSpec

SceneSpec包含schema/task/seed、世界、ROI、带稳定id的云/岛、机船完整初始状态、实际初始pan、完整参考船路、时长和配置快照。reset(options={'scene':scene})验证后原样回放，不偷偷重采样；随机尝试次数有上限。

完整scene_id用于精确复现。scene_group_id去掉seed/初始姿态/时长，保留布局和航路，同布局重复轨迹同组，避免训练/验证泄漏。

### 覆盖初始化

1. 随机矩形宽高10–20km，距总边界留6km缓冲。
2. 周边随机1–4岛、2–6云，筛选海面占比≥70%、合法起点、世界边界和有效航路；各最多8个完整进入obs。
3. 在矩形周长均匀抽弧长映射边界点，不等概率选边造成长短边偏置。
4. 初始顺时针切向使右侧雷达朝内，bank=0。首版不额外叠加航向噪声，位置和场景仍随机。
5. 船作为移动背景沿独立全域螺线运动，不切换覆盖标签。
6. 构造有效海面及扫描历史，应用初始扇形后返回观测。

### 跟踪初始化

1. 随机螺线相位和船起点、岛/云，A*绕岛；船沿首段方向，r=δ=0。
2. 相对方位均匀，初距1600–2400m；目标已被捕获，不混入搜索发现过程。
3. 根据当前船速构造恒速相对顺时针绕飞初速度和bank，并检查曲率可行性，不要求瞬时建立轨道。
4. 初始雷达对准船，检查真实距离、扇形、pan限制及安全距离。
5. 默认对测试时长内的船路保留安全云走廊，避免随机生成不可维持视场的目标穿云场景。这不代表已验证所有近目标复杂云障碍。

### 全域螺线和A*

参考阿基米德螺线4圈，半径约30–125km，位于300km世界内，点距≤25m。访问大范围海域，不强迫船到达方形每个角落。约2000km路线按12节需80–90小时；短演示只是真实航行片段，加速显示不改变运动速度。

受阻段用250m、8邻接A*；岛按转弯/响应与安全距离膨胀，禁对角穿缝，视线简化、圆弧平滑、重采样和连续障碍检查。ShipNavigator在当前与前方局部点推进，700m前视输出航向，避免跨圈跳进度。

reset检查完整参考路线几何、边界和间距；已有强制绕岛的真实Nomoto回归和任务回合验证。未在每次reset穷举仿真整条80小时航路，几何合法不等于全航程动力学保证。

## 6. 专家/网络共同输入和动作

外部obs为{'vector':float32[N]}，覆盖另有coverage_map(3,64,64)。策略统一reset()/act(obs)→float32[2]。固定专家算法参数为公共配置，逐步变化的物理和任务信息均在共同obs中；不允许专家额外读未来状态。

| 切片 | 内容 |
|---|---|
| ego[0:10] | x/y、cos/sin航向、真实vx/vy、bank、角速度、上步命令 |
| boundary[10:14] | 总边界左/右/下/上距离 |
| clouds[14:54] | 8×(相对dx/dy、半径、表面距离、valid) |
| islands[54:94] | 同上，飞机在岛上方合法 |
| coverage[94:98]/[98:111] | ROI相对边界/13个物理雷达参数 |
| tracking[94:103] | 船相对位置、距离、速度、航向、r和δ |
| tracking[103:106]/[106:134] | 实际pan/指向误差/可见性，以及28个机船/雷达/距离参数 |

向量覆盖111维、跟踪134维。详细字段顺序见observations.py的observation_metadata与构造函数。固定尺度：世界300000m、雷达3000m、ROI20000m、速度220/3.6m/s、年龄3600s、角度π。障碍按稳定id排序和valid补零，不按距离逐步换槽。

动作是单位世界XY期望方向，Box[-1,1]^2。原始actions为训练标签，executed_actions为步末真实速度/V_U，两者不混淆。非有限或越界指令拒绝；网络近零向量回退当前航向并限制浮点Box边界。

## 7. 覆盖目标与实际启发式

可定义长期目标：

    min J_C = sum_k [w_u×未扫格数 + w_a×平均age + w_m×最大age
                     + w_l×路程 + w_s×转弯变化平方]

等式约束是第3节完整物理转移和第4节真实扇形扫描更新；硬约束是恒速、曲率/倾斜/速率、云和总边界。ROI边界允许飞出。

实际专家对该目标做可解释近似，不求解大规模最优控制或完整论文MPC：

1. 顺时针圆角矩形扫描环，逐环内缩约2200m，中心用可飞小圆；右侧足迹优先朝内。
2. 根据当前位置和历史前向进度，取约650m前视点产生名义方向。
3. 漏扫/陈旧区域按8×8块选目标，优先级约为(20×未扫格数+平均归一化年龄)/(1+距离/15000)，近期目标降权。
4. 对补扫目标顺时针绕飞，右侧雷达照向目标；驻留有超时。
5. 首次全覆盖后重新全区扫描，持续重访，不无限停留在一个局部块。

### 共用物理避障

切向绕云场保持绕行侧，覆盖优先顺时针以保持传感器朝向。多个云先远后近，避免远障碍覆盖近障碍决策；世界边缘增加向内引导。进入保守预测带时，对名义方向、偏转和紧急转向候选调用同一FixedWingModel预测默认30s，检查连续段云/总边界。不可行丢弃，按方向偏差和最小余量选候选；全部失败抛ExpertPlanningError并留存失败。

阈值/增益/时长/权重均由Config/run.sh配置。有限预测不是无限安全证明。最大扫描年龄仍可能有长尾，未承诺全区域固定重访上界。

## 8. 跟踪目标与实际启发式

可定义目标：

    J_T = sum_k [w_d((d−d*)/d*)² + w_e(指向误差/雷达半角)²
                 + w_p(pan/pan_limit)^4 + w_s×转弯变化平方] + 终端距离代价

约束包括机船物理、相对位置及有限雷达的等式，和曲率、执行器、云/边界不等式。d*=2000m，近距参考1300m及雷达余量用于专家/初始化；环境不会替网络强行投影理想轨道，偏离必须实测报告。

实际相对运动场：e_r=(p_U−p_S)/d，e_tau=(e_ry,−e_rx)，

    n = normalize(e_tau − k_r(d−d*)/d* × e_r)
    c = −v_S·n + sqrt((v_S·n)² + V_U² − ||v_S||²)
    v_ref = v_S + c n

所以||v_ref||=V_U，并考虑船舶平移。用当前相对速度和执行器延迟预瞄，再经共用安全引导。雷达由环境共同规则控制，不额外给专家独占动作。任意障碍/船路下始终可见并非公式保证。

## 9. 两个独立GRU网络

覆盖：CNN(3→16,k5,s2；16→32,k3,s2；32→32,k3,s2)→2048→128；111维MLP128→128；拼接256→单向GRU128→MLP64→二维动作。

跟踪：134维→MLP128→128→单向GRU128→MLP64→二维动作。没有双向未来信息，每回合重置hidden。

默认动作头预测机体系单位方向，再旋转到世界坐标：

    b = normalize(tanh(head + (atanh(0.5),0)))
    a = R(真实航向) b

零修正精确保持航向，并可表达任意方向。最终动作与专家世界XY一致。保留world模式兼容旧checkpoint。

跟踪网络内部可把相对几何/速度/航向/前动作旋转到机体系，绝对位置屏蔽为平移不变量；动作转换仍使用原始航向。覆盖保留地图世界轴和真实航向，同时将相对云/速度转到机体系。feature_frame=auto默认两任务均body，旧模型缺字段按world解释。公共obs不变。

可选ego_map_channels在CNN内部由共同obs的ROI相对边界和真实航向生成3个通道：飞机位置高斯、该高斯×cos航向、该高斯×sin航向；原3个公开地图通道不变。这明确告诉CNN飞机位于覆盖地图何处，不读专家计划或未来信息。启用后CNN输入6通道，屏蔽冗余绝对XY及不限制飞行的岛矢量，陆地仍由sea_mask表示。另一可选模式body_coordinates提供每个格中心相对飞机的机体系x/20000、y/20000以及位置高斯三个稠密定位通道，避免仅在少量像素出现的定位信号；其设计借鉴[CoordConv的显式坐标输入](https://arxiv.org/abs/1807.03247)，这是本项目针对当前观测的推演，不声称该论文证明海上覆盖性能。world_coordinates则提供相对世界x/y与位置高斯，保持地图坐标轴一致；该实验也未单独解决覆盖闭环。模式和通道语义写入checkpoint，不能让旧权重默默改变语义。旧3通道权重需显式转换：新卷积通道权重初始为0、屏蔽项均值0/标准差1/首层权重0，保留原文件并重新训练，不能声称变换前后功能完全相同。

均值/标准差只由训练场景拟合；精确常量std=1，非恒定小std默认下限0.1，避免合法操纵状态被放大几十倍。实际闭环表明这种放大会导致短训练分布之外的跟踪漂移。预处理/下限/结构随checkpoint保存。

新增可选mask_previous_command：仅在覆盖模型内部预处理将vector[8:10]置零，原始共同观测和动作协议不变，bank/yaw物理状态仍保留。该消融针对已观察到的“复制上一命令”捷径，不是性能保证。CLI为--coverage-mask-previous-command，默认关闭；fresh训练使用当前开关，warm/DAgger继承checkpoint设置并打印。旧checkpoint缺字段按False，训练归一化、前向与在线部署共用同一预处理，tracking模型不增加此字段。

## 10. 独立采集、JSON和验证划分

main按task单独创建环境/专家。run_episode只复用交互代码。保存obs_0，随后obs_t→a_t→obs_(t+1)，T+1观测对应T动作；地图和状态是独立快照。

JSON包含schema、task、actor、seed、SceneSpec、metadata、frames、actions、executed_actions、summary。训练回合observations记录数组shape/dtype和zlib+base64编码原始float32字节；可仅从JSON无损恢复NPZ，NPZ不是唯一真值。演示可只保存完整状态/动作，不额外存所有训练地图张量。

frames保存真实时刻、机船状态/速度、真实雷达、覆盖热图、统计与失败原因。专家/网络同一格式，支持前端回放和本地导入。

同名采集在reset前作废旧NPZ；失败保留JSON但不产生训练缓存。跟踪专家出现首次失去视场也判示范无效，即使环境允许继续恢复。accepted-manifest列出本次成功数据，complete=true后才可训练，不递归吸收目录残留旧样本。

按scene_group_id整场景75%训练/25%验证，相邻片段不跨集合。继续训练还检查历史分组，禁止已参与训练的布局进入验证集；保存不可变的初始权重副本和SHA256，避免best.pt覆盖输入来源。覆盖默认5400s/回合，包含扫描及重访；只收1800s会漏掉后续阶段。跟踪也应覆盖实际船路转向，不能用短时直航示范宣称长时稳定。

## 11. 序列行为克隆与TensorBoard

    Loss = sum_valid ||a_network − a_expert||² / 有效时间步数

标签是原始专家指令。可选纠正权重仅用于训练优化：若||专家动作−obs[8:10]上一拍真实行为命令||>correction_threshold，则该步权重为correction_weight_max，否则1；padding权重0，按有效权重总和归一化。默认max=1完全关闭。它强调离开持续错误转弯等纠正示范，不按左右转简单平衡。启用后Train/Loss是实际加权目标，而Train/Action_MSE、MAE和全部Validation指标仍为未加权误差；epoch0只作未加权基线。因此启用时Train/Loss不再等于2×Train/Action_MSE。

标签是原始专家指令。batch是完整回合的并行槽位，连续64步TBPTT，chunk间hidden.detach，真实回合开始清零；只打乱回合顺序；padding不计损失。CNN仅编码当前chunk，避免整回合反向图撑满显存。

默认Adam，lr3e−4、weight_decay1e−5、梯度clip1、batch4、epoch30、GRU128。device/线程/参数均从run.sh传入；训练auto优先CUDA，在线单步推理设备独立配置，默认CPU。

| TF标签 | 定义 |
|---|---|
| Train/Loss、Validation/Loss | 训练为实际优化损失；验证为每有效步XY平方误差和，默认关闭纠正加权时定义一致 |
| Train/Action_MSE、Validation/Action_MSE | 始终未加权的每分量均方误差；未加权Loss/2 |
| Train/Action_MAE、Validation/Action_MAE | 每分量平均绝对误差 |

确定性连续策略不产生离散Accuracy、概率NLL/KL/Entropy，不能虚构。每epoch真实值可波动，只要求总体改善并保留验证最优；不美化为单调曲线。日志tf-logs/task/run-name，避免不同实验重叠同一步数。

checkpoint包含结构、obs版本、预处理、归一化、权重、优化器、场景划分、seed和物理/专家配置。网络闭环不调用专家作隐藏回退。离线误差下降不等于长时任务成功；需要同场景闭环检查，必要时仅在训练场景补充专家标签。

### 训练阶段DAgger

collect_dagger只从checkpoint中训练集合选择布局，保持验证布局隔离。在网络访问的当前obs上查询专家，原始行为actions、policy_actions、expert_actions和真实执行方向分别保存；NPZ标签与JSON恢复统一使用expert_actions。β表示每步采用专家控制的概率，只在训练补采生效，JSON和界面明确标记dagger_training。部署与最终网络评估β不存在，不调用专家。DAgger保留成功取得标签的非空前缀：若学生后续碰撞、失去目标或专家随后无法规划，最终失败观测不作为训练输入，保留此前obs_t→有效expert_action_t；metadata明确label_valid/accepted_prefix/behavior_failed，summary仍保留真实失败。原始失败专家示范不因此变成成功示范，意外程序错误仍抛出。合并manifest先以原清单目录解析所有相对路径。

继续训练继承结构、预处理和归一化，保留历史划分；旧标准化不能因新数据而被悄悄重拟合。DAgger增加访问状态覆盖，不保证小批次一定学会长时覆盖，仍必须运行纯网络验证。

## 12. Gym、端口JS和MP4

reset返回(obs,info)，step返回(obs,0.0,terminated,truncated,info)。碰撞/越总边界/非法物理状态终止；达到时长截断；首次100%或暂时丢失目标不自动终止。结束后step拒绝。

render_mode=json/human产生真实状态字典，main调用render并发布SSE。VisualServer提供只读静态资源、/events、/api/replays和/api/visual-config，慢客户端只跳显示旧帧，不影响仿真。缓存和pytest记录不列历史，路径穿越受限。

JS支持任务/专家或网络选择、JSON导入、播放/暂停/倍速/进度、全域/任务区/跟随、年龄热图、真实速度朝向的加载式SVG机船模型和实拍照片。SVG是原创二维示意模型，不冒称下载三维模型；照片是真实下载，来源/许可/hash见figs/sources.json与ATTRIBUTION.md。

最新显示采用全世界250m访问栅格，真实飞越格按最后访问时间渐淡至最小透明度，与雷达扫描年龄分开。默认60秒流星尾迹，可切换完整轨迹或关闭；历史seek不泄漏未来足迹。图标按参考长宽及最小可见像素绘制，雷达保留3km真实距离。16个visual_*字段在run.sh显式传入，旧回放通过服务端显示配置适配；不改变原轨迹物理。大型JSON目录索引由单后台线程生成可丢弃sidecar，新回放直接生成索引。

Matplotlib MP4采用同一真实frames，抽帧显示长过程、标明真实仿真时刻；倍速不更改物理方程。少于两帧的失败回合保留JSON，不强行导出电影。可补充前端实录演示。

## 13. 工程接口与实施顺序

src有21个业务模块（含新增expert_validation批量验证）：config/dynamics/geometry/navigation/scenario/observations、env_base/env_coverage/env_tracking、expert_coverage/expert_tracking、policy_coverage/policy_tracking、datasets/rollouts/train/rendering/visual_server、collect_dagger/validation。根main.py统一CLI，run.sh列出全部可调参数。文件数量不作为算法效果指标。

outputs含datasets/models/replays/videos/training/reports/diagnostics/cache；tf-logs独立。输出与日志路径通过CLI传入，不写用户桌面。训练/演示配置快照随数据保存。

顺序：保留物理基线→场景/导航/Gym→专家闭环和失败修复→独立采集→场景划分→两网络训练→同场景闭环→JS/MP4检查→代码审查/实测验收。单测不是长时间演示的替代，低Loss不是稳定闭环的替代。

## 14. 覆盖面积合理性

不存在不依赖雷达宽度、航时、重访要求和障碍的统一“固定翼最大覆盖面积”。3km单侧理论扫描宽度×220km/h约660km²/h；按12h理想7920km²，转弯/重叠/绕云会降低。这是公开速度/航时的推算，不是厂家公布的搜救面积。

90000km²世界即使理想也约136h，所以将300km世界作为背景、每回合100–400km²局部任务合理。年龄3600s是归一化尺度，不是已经保证的一小时重访时限。首版不建模燃油，时长由任务参数决定。

## 15. 参考链接

- [区域分解覆盖](https://publications.ri.cmu.edu/coverage-path-planning-the-boustrophedon-decomposition)
- [Paull传感器覆盖](https://people.csail.mit.edu/lpaull/publications/Paull_CYB_2013.pdf)
- [持续监测](https://dspace.mit.edu/entities/publication/b1ed9ada-d912-4767-b072-6a3bc194ad27)
- [Dubins](https://msl.cs.illinois.edu/~lavalle/planning/node821.html)
- [目标绕飞避障向量场](https://www.jstage.jst.go.jp/article/jaciii/19/2/19_277/_article/-char/en)
- [A*](https://ai.stanford.edu/~nilsson/OnlinePubs-Nils/PublishedPapers/astar.pdf)
- [FAA协调转弯](https://www.faa.gov/sites/faa.gov/files/regulations_policies/handbooks_manuals/aviation/00-80T-80.pdf)
- [Fossen模型](https://www.fossen.biz/html/marineCraftModel.html)
- [Nomoto操纵仿真](https://www.mdpi.com/2077-1312/7/4/90)
- [DAgger](https://proceedings.mlr.press/v15/ross11a/ross11a.pdf)
- [彩虹公开参数](https://m.spacechina.com/n146/n238/n15357/c18332/content.html)
- [海洋一所科考船](https://www.fio.org.cn/news/tzgg-detail-10287.htm)
- [Gymnasium](https://gymnasium.farama.org/api/env/)
- [PyTorch GRU](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html)
- [TensorBoard](https://docs.pytorch.org/docs/stable/tensorboard.html)

## 16. 最新增补与验收状态

新增expert-validate入口，预先固定64个随机/边界案例；只排除物理上非法初始状态，不按专家试跑结果筛场景。结果与源码和物理配置哈希绑定，断点续跑不混用已修改算法的旧结果。有限验证不能证明任意几何与初态必然可行。

run.sh完整列出133个Config字段，布尔显示字段和所有CLI选项有一致性检查；跟踪角度/距离余量实际约束初始化，显示配置不改变物理观测或运动。

GitHub发布遵照用户明确授权，忽略outputs、tf-logs、模型、数据和缓存，保留必要授权素材。当前跟踪网络通过独立闭环；覆盖神经网络仍未达标，详见../../validation-status.md和本地验收报告，不能宣称两个模型均已学会任务。
