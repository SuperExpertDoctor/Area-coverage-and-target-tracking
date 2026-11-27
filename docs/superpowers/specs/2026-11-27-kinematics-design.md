# 二维机船运动模型

本次实现独立运动模块，供后续环境、专家预测与轨迹回放共同调用。位置只有x/y，采用东向x、北向y、逆时针为正的局部坐标；航向从正x轴起算，内部角度为弧度。无风、无流，速度控制器维持恒定速率。倾斜角、舵角属于内部操纵状态，不增加高度维度。

UAV状态为(x,y,heading,bank)，模型为x_dot=V cos(heading)、y_dot=V sin(heading)、heading_dot=g tan(bank)/V。bank具有一阶响应和最大变化率。有效倾斜上限同时满足配置的倾斜上限与最小转弯半径：min(bank_max,atan(V²/(g R_min)))。正bank表示左转。默认V=220/3.6 m/s、R_min=1000 m、bank_max=25°、倾斜时间常数1 s、最大倾斜变化率15°/s。step_velocity接收期望世界坐标速度方向，使用固定1 s指令时间尺度映射到倾斜目标，内部随当前航向反馈更新；零向量表示保持调用开始时的航向，不表示悬停。step_bank供底层操纵与定常转弯验证使用。

船舶状态为(x,y,heading,yaw_rate,rudder)。忽略侧滑，位置导数同上。采用一阶Nomoto响应：T yaw_rate_dot+yaw_rate=K rudder；实际舵角具有一阶舵机响应和最大变化率。默认V=12*1852/3600 m/s、R_min=300 m、T=20 s、舵角上限35°、最大舵速3°/s、舵机时间常数1 s。K=V/(R_min*rudder_max)，用于按所选定常最小转弯半径参数化简化模型。step_heading接收航路跟随器给定的期望航向，以30 s航向时间尺度和0.8角速度反馈生成舵角。step_rudder供底层操纵与响应验证使用。正rudder表示左转。

速度引用：CH-3 https://m.spacechina.com/n146/n238/n15357/c18332/content.html ；向阳红01 https://www.fio.org.cn/news/tzgg-detail-10287.htm 。协调转弯参考FAA https://www.faa.gov/sites/faa.gov/files/regulations_policies/handbooks_manuals/aviation/00-80T-80.pdf 。Nomoto模型参考 https://www.mdpi.com/2077-1312/7/4/90 。除航速外，默认操纵参数都是本项目仿真假设，并未按上述真实机船辨识。该模型不声称为完整气动/水动力仿真。

状态不可变，推进返回新状态，不修改输入。采用RK4及不大于0.05 s的内部子步；快速响应配置自动缩小子步。step不包含碰撞规避，雷云/岛屿/边界由环境检查。环境与专家必须采用相同的控制映射、配置及完整运动状态，专家不能用瞬时Dubins转向替代有响应过程的真实推进。船舶A*只是导航层，其航路跟随器调用step_heading；本次不实现A*或完整Gym环境。

验证：解析直行；定常圆弧位置；UAV最大曲率、倾斜变化率及转向响应；船舶Nomoto阶跃解析解、舵角/舵速/曲率；指令速度大小不改变速率；跨±pi的最短转向；整步与分步推进一致性；非有限输入与非法物理状态拒绝。独立生成Matplotlib轨迹与状态响应图，检查二维运动行为。
