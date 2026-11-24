# 从趴姿起身到四足小跑：BPX 的 MPC + WBC 教程

先运行 [README](../README.md) 里的无窗口前进示例，再打开生成的 CSV。取三个时间段对照代码：0–4 s 看起身，4–5 s 看站稳与交接，6–11 s 看小跑。工程用三个模型处理三个问题：解析腿运动学把起身足位转换为关节角，单刚体模型预测机身整体运动，全身动力学把足端与机身任务转换为电机力矩。

## 1. 先认识控制量和坐标系

以下腿序贯穿项目：`FL, FR, HL, HR` 分别是左前、右前、左后、右后。每条腿的关节序为 `hip_roll, hip_pitch, knee`。受控关节是 12 个，浮动基座还有 6 个速度自由度。根姿态由四元数表示，所以 `qpos` 长度 19、`qvel` 长度 18。记号 $q$、$v$、$a=\dot v$ 分别表示广义位置、速度和加速度。对浮动基座，$v$ 与逐元素改变 $q$ 的方式不同，因此机器人适配层用 MuJoCo 雅可比计算直观的笛卡尔速度。

世界坐标 $z$ 朝上，机体 $x$ 朝前、$y$ 朝左。$R$ 将机体系向量旋转到世界系。手柄期望速度在机体系定义，控制器以目标航向 $R_z$ 将前向和横向分量转换到世界系。WBC 实测机身位置是 torso 原点，MPC 的平动状态是整机质心。转动参考经机身坐标下的质心偏移把这两个位置联系起来。

| 对象 | 维度与含义 |
|---|---|
| `state.position` | 世界系 torso 原点，m |
| `state.com` | 世界系整机质心，m |
| `state.omega` | 世界系角速度，rad/s |
| `state.feet` | 四只脚的足垫中心，形状 `(4, 3)`，m |
| `targets.contact` | 调度的四足接触布尔值，供两个 QP 使用 |
| `force_ref` | MPC 当前步期望的地面对脚的力，形状 `(4, 3)`，N |
| `torque` | 关节力矩，形状 `(12,)`，N·m |

`--terrain rough` 在无限平地上布置分散的丘包，保留出生点和丘包之间的平地通道。仿真按世界位置更新附近的高度场。跨区时，MuJoCo 的水平原点与控制参考、脚锚点一起平移。表里的位置量使用同一个局部原点，坐标轴方向保持一致。CSV 的 `x`、`y` 加回原点偏移，记录连续的世界轨迹。[地形生成](../src/bpx_mpc/terrain.py) · [控制参考平移](../src/bpx_mpc/controller.py)

读 [robot.py](../src/bpx_mpc/robot.py) 中的 `qadr`、`vadr` 和 `aids`。同一个关节在 `qpos`、`qvel` 和 `ctrl` 中的下标由各自的 MuJoCo 地址得到。检查足端运动学时，可以使用 $v_i=J_i v$ 对比两帧足位的差分。跨区的两帧须先加回各自的原点偏移。起身 IK 用髋坐标下的 toe link 原点，行走控制使用世界坐标轴下的足垫中心，两者分别来自腿几何与 MJCF 的 `*_foot` site。

## 2. 趴姿起身与交接

对一段时长 $T$ 的运动，取 $s=\operatorname{clip}(t/T,0,1)$。希望起点和终点速度、加速度都为零，六个边界条件给出五次标度：

$$
h(s)=10s^3-15s^4+6s^5,\qquad
p(s)=p_0+h(s)(p_1-p_0).
$$

$h(0)=0$，$h(1)=1$，$h'$ 和 $h''$ 在两端都为零。先由实测关节角做正运动学得到起点，再在髋坐标内分两段规划足位。第 1 段把四脚收到较近的支撑点，默认深度 0.19 m，用时 1 s。第 2 段逐渐加深到 0.36 m，默认再用 3 s。足端接触地面后，继续伸腿使机身升高。这里的深度是从髋关节到 toe link 原点的相对距离，机身高度由物理仿真演化得到。若跟踪误差过大，或者到支撑点后没有四脚受力，轨迹计时可以暂停。

摘自 [init_motion.py](../src/bpx_mpc/init_motion.py) 的第 2 段与 IK 接口：

```python
alpha = quintic((self.progress - self.config.support_duration)
                / (self.config.duration - self.config.support_duration))
feet = self.support_feet + alpha * (self.final_feet - self.support_feet)
# 在两段共用的末尾，解析 IK → 关节目标，并限制每步角度变化。
desired = self.kinematics.inverse(feet)
self.target += np.clip(desired - self.target, -self.config.joint_speed_limit * dt,
                       self.config.joint_speed_limit * dt)
```

`LegKinematics.inverse()` 先由足位的 $y,z$ 解髋 roll，移除横向连杆偏置。再在矢状面中用余弦定理求向后弯的膝关节分支和髋 pitch。验证时可将一组合法关节角先 `forward` 后 `inverse`，检查角度闭环，再逐步改变深度观察站姿。关节 PD 在每个控制更新时执行 $\tau=K_p(q_d-q)-K_d v_{joint}$，最终经过执行器力矩限幅。

轨迹完成后仍保持 PD 终点。状态机连续检查四脚各有超过 2 N 的法向力、机身高度超过 0.33 m、倾角小于 0.20 rad、线速度小于 0.08 m/s、角速度小于 0.22 rad/s。持续满足 0.5 s 后，锁定当前脚锚点和参考航向，进入交接。交接期也用五次标度，系数从 0 到 1：

```python
pd = robot.pd(self.rise.final_target, rise["kp"], rise["kd"])
torque = (1.-alpha)*pd + alpha*torque
```

这里右端的 `torque` 是当拍 WBC 输出，混合后的力矩才写入电机。行走指令在交接完成后开始参与控制。可在 [controller.py](../src/bpx_mpc/controller.py) 查找 `Mode.HANDOFF`，对照 CSV 的 `t=4.5s` 和 `t=5.0s`。

## 3. 步态提供接触时序与落足目标

[gait.py](../src/bpx_mpc/gait.py) 为 FL+HR 与 FR+HL 分配半个周期的相位差。单腿相位

$$
\phi_i(t)=\left((t-t_0)/T+o_i\right)\bmod 1,
\qquad
o=(0,0.5,0.5,0).
$$

当相位小于占空比 $d=0.60$，该腿处于支撑期。$T=0.60$ s，所以一只脚的支撑时间为 0.36 s、摆动时间为 0.24 s。$d>0.5$ 留出了四足同时支撑的窗口。起步与停止阈值做了迟滞，停止时还要等限速后的指令降下来，并进入一个四足支撑窗口。

```python
velocity = desired_v + np.cross([0., 0., command[2]], hip)
delta = (Ts+.5*Tstance)*velocity + c["velocity_feedback"]*(state.velocity-desired_v)
delta[:2] = np.clip(delta[:2], -c["max_step"], c["max_step"])
self.end[leg] = state.position + hip + delta
```

`hip` 是站稳时捕获的名义足位在当前航向下的水平偏置。第一项用期望平移与转向速度预测摆腿结束附近的落脚点，第二项根据实际和期望速度差修正步幅。转弯时 $\omega_z\times r_i$ 让四条腿得到不同目标。摆腿自抬脚时的实测足位出发，终点的足垫中心高度取落点处的地表高度加足垫半径 0.027 m。轨迹使用五次插值，另加 $64 H s^3(1-s)^3$ 的竖直隆起，$H$ 默认 0.06 m。位置、速度和加速度通过解析式直接交给 WBC。

控制器会先用各轴速度上限归一化输入。如果归一化向量的二范数大于 1，就同比缩小各轴，再执行逐轴速率限制。这让单轴仍可达到上限，同时抑制多轴满量程叠加时的过大步幅。

预测时域中的接触掩码按相位向未来查询。已在摆动的脚用已规划的落点，其他脚用当前支撑锚点。MPC 用足垫中心减去足垫半径得到近似地面接触点。这个冻结近似使力优化保持凸性，也意味着较远处的落脚几何有误差。改变步态切换规则时，要同步验证足端轨迹、MPC 掩码和 WBC 掩码三处。

## 4. 单刚体 MPC 从力推到速度

将整机近似为质量 $m$、绕质心的惯量 $I$ 的单刚体。令 $p$ 为整机质心，$r_i$ 为从质心到第 $i$ 只脚地面力作用点的力臂，$f_i$ 为地面对该脚的力，$g$ 为世界系重力加速度。

$$
m\ddot p=\sum_i f_i + mg,\qquad
I\dot \omega \approx \sum_i r_i\times f_i .
$$

第二式忽略了陀螺项和关节相对运动造成的动量变化。每次 MPC 更新从 MuJoCo 读取连杆质心、姿态和惯量，用平行轴定理得到 $I$，在该次求解内部冻结它。预测中每步的 $r_i$ 使用预测足位减当前质心，转动从世界角速度近似映射到小 roll/pitch 的欧拉角速度。将动力学按预测步长离散化：

$$
x_k=[\phi,\theta,\psi,p_x,p_y,p_z,\omega_x,\omega_y,\omega_z,v_x,v_y,v_z]^T,
\qquad u_k=[f_{FL}^T,f_{FR}^T,f_{HL}^T,f_{HR}^T]^T.
$$

$$
x_{k+1}=Ax_k+B_k u_k+g_d.
$$

[mpc.py](../src/bpx_mpc/mpc.py) 里的 $A$ 把现有速度积分到位置与角度。$B_k$ 的每三列是一条腿的变化率映射。设 $[r_i]_\times f_i=r_i\times f_i$，在单步假定加速度恒定，对位置积分出现 $\Delta t^2/2$，对速度积分出现 $\Delta t$：

```python
angular = np.linalg.solve(inertia, skew(r))
sl = slice(3*leg, 3*leg+3)
B[:3, sl] = .5*dt**2*T @ angular
B[3:6, sl] = .5*dt**2/mass*np.eye(3)
B[6:9, sl] = dt*angular
B[9:12, sl] = dt/mass*np.eye(3)
```

对应代码中 `g[5]` 是重力对高度的单步影响，`g[11]` 是重力对竖直速度的单步影响。有个可手算的检查：$m=15.157$ kg 时，平地对称站立只考虑平动，每个脚分到约 37.2 N 的向上力，四脚总和约 148.7 N。实测四脚力会随载荷分布而不同。

## 5. 把 MPC 写成 OSQP 能求解的 QP

每次用当前测量 $x_0$ 作为初值。预测参考包含滚俯角零值、平面位置与速度、机身高度、航向和转向角速度。航向误差在 $\pm\pi$ 附近展开，避免机器人跨过角度边界时出现接近 $2\pi$ 的假误差。每轮优化以下代价：

$$
\min \sum_{k=0}^{N-1}
  (x_{k+1}-x_{k+1}^{ref})^T Q(x_{k+1}-x_{k+1}^{ref})
  + \lambda \lVert u_k\rVert^2 .
$$

$Q$ 的 12 个对角权重和力正则 $\lambda$ 位于 [default.toml](../src/bpx_mpc/data/default.toml)。提高高度权重会优先抵抗高度偏差，提高速度权重会优先跟踪速度。力正则惩罚过大的力，绝对可行范围仍由约束决定。

机身原点的目标高度是当前机身正下方的地表高度加 `robot.body_height`，默认离地 0.40 m。机身高度参考每秒最多调整 0.12 m。MPC 根据站姿的质心偏移把机身参考换算为质心参考，WBC 跟踪机身原点参考。一次 MPC 求解中，整个预测域沿用当前参考高度。[参考计算](../src/bpx_mpc/controller.py)

约束包括动力学等式和单脚摩擦棱锥：

$$
0\leq f_z\leq f_{z,max},
\qquad |f_x|+|f_y|\leq \mu f_z .
$$

这个棱锥是圆形库仑锥的保守内近似。将两个绝对值展开为四个线性不等式后，OSQP 可直接处理。摆腿的法向力上界置零，因此三维接触力都为零。

代码保存全部预测状态和控制输入作为决策变量，采用 OSQP 形式 $\tfrac12 z^TPz+q^Tz$，约束写作 $l\leq A_{qp}z\leq u$。展开 $\lVert x-x_{ref}\rVert_Q^2$ 后，二次项系数为 $2Q$，线性项为 $-2Qx_{ref}$。默认 $N=12$ 时，一共有 $12N+12N=288$ 个变量。每步 12 条动力学等式和 20 条力约束，共 384 行。第一行块消去已知的 $x_0$：

```python
# z=[x1,...,xN,u0,...,u(N-1)]。以下摘自逐步组装循环。
row[:, k*nx:(k+1)*nx] = np.eye(nx)
if k > 0:
    row[:, (k-1)*nx:k*nx] = -A
row[:, N*nx+k*nu:N*nx+(k+1)*nu] = -B
values.append(g+A@x0 if k == 0 else g)
```

因此第一步等式是 $x_1-B_0 u_0=g_d+Ax_0$，第二步开始写作 $x_{k+1}-Ax_k-B_k u_k=g_d$。求完 $N$ 步，只把首拍 4×3 力交给 WBC。下一次用新测量重建 $x_0$ 后再滚动求解。它能使用未来换腿信息，同时用反馈修正模型误差。

[qp.py](../src/bpx_mpc/qp.py) 用 SciPy CSC 稀疏矩阵传给 OSQP。结构相同时更新数值并沿用求解器热启动，结构改变时重新建立问题。求解器返回 `solved` 或 `solved inaccurate` 后，还要检查 $A_{qp}z$ 相对上下界的最大违反量。这个诊断混合了力与动力学等式等不同行的量纲，用于发现数值失败。程序只接受有限解，且最大违反量须小于或等于 0.02。

## 6. WBC 把期望足端力变为关节力矩

MPC 的近似模型只有机身与接触力。WBC 每次使用完整浮动基座动力学和每条腿的雅可比：

$$
M(q)a+h(q,v)=S^T\tau+J_c(q)^Tf.
$$

$M$ 是 18×18 的关节空间质量矩阵。$h$ 来自 `qfrc_bias - qfrc_passive`，包含重力、速度相关项和移到左侧的被动力。$S^T$ 把 12 路电机力矩散布到对应关节行，浮动基座的 6 行没有直接电机力矩。$J_c^Tf$ 由虚功关系得到：$f^T(J_c v)=(J_c^Tf)^T v$。这也解释了代码等式中力项的负号。

```python
dynamics = np.hstack([M, -jac.reshape(12, 18).T, np.zeros((18, 12))])
dynamics[robot.vadr, 30+np.arange(12)] = -1.
```

决策变量是 $z=[a(18), f(12), \tau(12)]$。WBC 首次求解时，把支撑脚的 $J_i a+\dot J_i v$ 约束到由锚点误差和足端速度计算的目标加速度 $a_i^{ref}$。摆腿把同一加速度表达式作为软目标，追踪预先生成的足端位置、速度和加速度。姿态任务用 $R_{ref}R^T$ 对应的旋转向量做误差，避免欧拉角直接相减。

其余软任务包括基座位置与姿态、MPC 接触力跟踪、关节姿态正则。一项任务 $w\lVert Tz-y\rVert^2$ 是这样累加的：

```python
P += 2*weight*matrix.T @ matrix
q -= 2*weight*matrix.T @ target
```

动力学、摩擦棱锥、摆动脚零接触力、电机力矩上下限，以及按一个 WBC 周期预测的关节位置与速度，都是硬约束。急换向或接近关节限位时，支撑脚加速度目标可能和这些约束冲突。首次求解失败且存在计划支撑脚时，WBC 将支撑脚等式移入代价函数，增加 $w_{\mathrm{stance}}\lVert J_i a+\dot J_i v-a_i^{ref}\rVert^2$，再求解一次。`wbc.stance_weight` 默认是 30，只用于这次重试。重试保留其余硬约束并检查约束残差。重试成功就执行本帧力矩，下一帧重新从支撑脚硬约束开始。[WBC 求解](../src/bpx_mpc/wbc.py) · [默认权重](../src/bpx_mpc/data/default.toml)

力矩和摩擦受到约束时，实际速度或步幅可能低于目标。MuJoCo 根据实际碰撞求解接触，再由 `mj_step` 积分。日志提供 `mpc`、`wbc`、`actual` 三类力，分别对应计划值、逆动力学计算值和物理引擎计算值。WBC 采用足垫中心的点雅可比，MPC 使用近似位于地面的力臂端点，MuJoCo 计算圆柱足垫上的分布接触力。

## 7. 建议的代码练习和排查顺序

1. **重算起身曲线。** 给 $s=0,0.5,1$ 代值，再求两阶导数。改变 `init.duration`，用 CSV 看 `init_progress` 和机身高度的先后关系。若卡在支撑点，比较四路实测 `fz` 与足端深度。
2. **看懂静站的力。** 将 `vx/vy/wz` 都设为零，比较四路 `actual_fz` 之和与 $mg$。用 MPC 的首拍力代入质心的平动方程，理解力不均分时的结果。
3. **画出对角换腿。** 对前进日志的 `*_planned_contact` 画阶梯图，把 `actual_fz` 和力矩叠在同一时间轴上。起落地附近的短时差来自名义相位与实际接触。
4. **调整一个代价权重。** 将 MPC 的世界系 `vx` 权重提高一倍，对比全周期平均速度、摆腿相位附近的力峰值和 WBC 力矩。然后恢复基线，只修改 `wbc.force_weight`，观察 WBC 力与 MPC 力的差异。
5. **读一次停步。** 观察 `raw_vx` 归零、`cmd_vx` 逐步下降、`STOPPING` 进入、最后一组脚落地后的 `STAND`。若调大指令限速后机身抖动，先调整限速，再检查脚锚点和求解诊断。

6. **对照丘包与重试日志。** 用 README 的丘包示例生成 `logs/rough.csv`，对照 `x` 和 `height` 看上坡、下坡与回到平地。用手柄排查时，加上 `--log` 保存摇杆指令。`wbc_stance_retries` 每增加 1 表示一次重试成功，即使重试发生在两个 CSV 采样点之间也能看出。`wbc_status` 和 `wbc_violation` 记录最近一次求解的状态与约束违反量。[运行示例](../README.md#无限平地与丘包)

MPC 求解失败、WBC 重试后仍失败或检测到跌倒时，终端会给出仿真时间和状态。`maximum iterations reached` 表示该轮已用完 6000 次迭代，`violation=inf` 是程序在未取得可评估解时保留的诊断值。遇到 WBC 停机，可回看故障前的摇杆指令、计划接触和实测竖直足力，再核对足端目标、关节预测限位与可用力矩。排查 MPC 时先看接触掩码、摩擦和状态参考。该工程适合研究低速平地和丘包场景，机器人状态读取 MuJoCo 真值。复杂地形或真实硬件需要单独处理接触检测、状态估计和扰动恢复。

- [MuJoCo 函数文档](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html)
- [OSQP 的 Python 接口](https://osqp.org/docs/interfaces/python.html)
- [Kim 等人关于四足 MPC 与全身控制的论文](https://arxiv.org/abs/1909.06586)
