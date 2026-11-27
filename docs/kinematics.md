# 固定翼与科考船的二维运动模型

实现位于 `src/dynamics.py`，独立于Gym、启发式策略和神经网络。环境和专家预测应复用同一个模型。

## 基本约定

- 位置只有x/y，单位m；x向东、y向北。速度单位m/s，时间单位s。
- 航向从正x轴起算，逆时针为正；内部所有角度用rad。
- 正倾斜/正舵角表示左转。这是本项目约定，与一些航空/航海NED坐标的符号不同。
- 无风、无海流、无侧滑；自动速度控制维持恒定速率。速度方向随运动连续变化。
- UAV内部倾斜角只用于确定二维转弯，不引入高度状态。船舶只建模平面转向。
- 状态不可变，每次推进产生新状态，适合保存轨迹和专家候选预测。

## 固定翼UAV

状态为 \(s_U=(x,y,\psi,\phi)\)，其中\(\phi\)是内部倾斜角：

\[
\dot x=V_U\cos\psi,\quad \dot y=V_U\sin\psi,\quad
\dot\psi=\frac{g\tan\phi}{V_U}.
\]

协调转弯的半径为\(R=V_U^2/(g|\tan\phi|)\)。因此有效倾斜限幅取

\[
\phi_{lim}=\min\left(\phi_{max},\arctan\frac{V_U^2}{gR_{min}}\right).
\]

倾斜角具有有限响应：

\[
\dot\phi=\operatorname{clip}\left(
\frac{\phi_{cmd}-\phi}{\tau_\phi},-p_{max},p_{max}\right).
\]

`step_velocity`接收期望世界坐标速度方向，由`atan2(vy,vx)`得到期望航向。航向误差用最短角差，除以固定的指令时间尺度后得到期望角速度，再由协调转弯公式得到倾斜目标。控制映射不依赖物理调用步长，因此1秒推进与10次0.1秒推进具有相同物理含义。速度向量的模长不参与速度控制，零向量表示保持调用开始时的航向，不表示悬停。

该指令仍表示下一决策时刻的期望速度方向，但响应延迟意味着实际方向未必在一拍内达到。训练应保存专家原始指令与实际速度两个字段，专家规划必须通过本模型预测动作效果。

默认值：V=61.1111 m/s、R_min=1000 m；有效倾斜上限约20.8478°，最大转弯角速度约3.5014°/s；倾斜响应时间常数1 s、最大倾斜变化率15°/s。

当倾斜角恒定时轨迹为直线或圆弧，满足Dubins曲率约束；倾斜变化时曲率平滑变化。模型在Dubins最大曲率约束上增加了操纵响应，不把经典Dubins连接处的瞬时曲率跳变当成真实飞机动作。

## 科考船

状态为\(s_S=(x,y,\psi,r,\delta)\)，r为转弯角速度、delta为实际舵角：

\[
\dot x=V_S\cos\psi,\quad \dot y=V_S\sin\psi,\quad
\dot\psi=r,\quad T\dot r+r=K\delta.
\]

舵机响应为

\[
\dot\delta=\operatorname{clip}\left(
\frac{\delta_{cmd}-\delta}{\tau_\delta},
-\dot\delta_{max},\dot\delta_{max}\right),\quad
|\delta_{cmd}|\le\delta_{max}.
\]

本项目用\(K=V_S/(R_{min}\delta_{max})\)将所选定常最小半径参数化。若初始\(|r|\le V_S/R_{min}\)，有限舵角下Nomoto响应保持此角速度约束，且转向有惯性。

`step_heading`将期望航向交给共同航向控制器：

\[
r_{cmd}=\operatorname{clip}\left(
\frac{\operatorname{wrap}(\psi_{cmd}-\psi)}{\tau_\psi}
-k_r r,-r_{max},r_{max}\right),\quad
\delta_{cmd}=r_{cmd}/K.
\]

默认值：V=6.17333 m/s（12节）、R_min=300 m、T=20 s、最大舵角35°、最大舵速3°/s、舵机时间常数1 s、航向时间尺度30 s、角速度反馈系数0.8。对应最大定常转弯角速度约1.1790°/s。

A*与实际运动的关系为：`A*海面航路 → 平滑及可航行性校验 → 航路跟随器给出期望航向 → ShipModel.step_heading`。A*折线不能直接成为船舶的瞬时运动。有限转向响应下，即使几何航路避岛，实际航迹仍须由环境检查；本运动模块本身不实现避障或A*。

## 环境和策略接入

```python
from dynamics import FixedWingModel, FixedWingState, ShipModel, ShipState

uav_model, ship_model = FixedWingModel(), ShipModel()
uav = FixedWingState(x=0, y=0, heading=0, bank=0)
ship = ShipState(x=2000, y=0, heading=0, yaw_rate=0, rudder=0)

# action是专家或网络输出的二维期望速度方向，环境内保持相同低层控制。
action = (0.8, 0.2)
ship_heading_command = 0.3  # 由船舶航路跟随器产生
for _ in range(10):
    uav = uav_model.step_velocity(uav, action, dt=0.1)
    ship = ship_model.step_heading(ship, ship_heading_command, dt=0.1)
    # 在环境中更新雷达并检查本子步轨迹与雷云/岛屿/总边界的关系。
actual_uav_velocity = uav_model.velocity(uav)
actual_ship_velocity = ship_model.velocity(ship)
```

后续共同观测中增加UAV的bank，以及船舶的yaw_rate/rudder；这些真实运动状态对专家和网络同时可见。GRU可以保留历史，但不能让专家额外使用网络看不到的操纵状态。

推进采用RK4，内部步长不超过0.05 s，并根据响应时间常数自动缩小。初始超限操纵状态、NaN、无穷大和负时间会被拒绝。位置不会因为速度指令发生跳变。碰撞和越界处理属于环境层，不能通过坐标裁剪或瞬移修正。

## 运行验证

```powershell
python -m pytest -q
python -m examples.plot_kinematics --output outputs/kinematics/kinematics.png
```

示例输出PNG、两份时序CSV和包含参数及运动统计的JSON。图只演示运动模型，不代表覆盖/跟踪策略已经完成。

## 依据与适用范围

- [航天科技集团CH-3参数](https://m.spacechina.com/n146/n238/n15357/c18332/content.html)：巡航速率220 km/h。
- [海洋一所向阳红01资料](https://www.fio.org.cn/news/tzgg-detail-10287.htm)：服务航速12节。
- [FAA协调转弯公式](https://www.faa.gov/sites/faa.gov/files/regulations_policies/handbooks_manuals/aviation/00-80T-80.pdf)：速率、倾斜角、转弯半径与转弯角速度的关系。
- [Fossen船舶运动模型](https://www.fossen.biz/html/marineCraftModel.html)：平面船体速度到位置/航向的运动学关系。
- [Ni等，2019](https://www.mdpi.com/2077-1312/7/4/90)：船舶操纵仿真中的Nomoto转向响应。

除参考航速外，操纵参数均为仿真设定，尚未通过真实机船试验辨识。模型适合本项目的二维规划、数据采集和模仿学习；不模拟升阻力、失速、推进功率、船舶侧滑或完整水动力。
