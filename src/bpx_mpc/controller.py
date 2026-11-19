"""控制器编排：趴姿等待 → 五次轨迹起身 → 站稳 → 力矩交接 → 站立/小跑。"""
from enum import StrEnum

import numpy as np

from .gait import GaitPlanner
from .init_motion import FootTrajectory, InitControllerConfig, LegKinematics
from .math_utils import quintic, yaw_rotation
from .mpc import ConvexMPC
from .robot import JOINTS, Robot
from .wbc import WholeBodyController


class Mode(StrEnum):
    WAIT = "WAIT"
    RISE = "RISE"
    SETTLE = "SETTLE"
    HANDOFF = "HANDOFF"
    STAND = "STAND"
    TROT = "TROT"
    STOPPING = "STOPPING"


class Controller:
    def __init__(self, cfg):
        self.cfg = cfg
        self.robot = Robot(cfg)
        self.gait = GaitPlanner(cfg)
        self.mpc = ConvexMPC(cfg, self.robot.mass)
        self.wbc = WholeBodyController(cfg, self.robot)
        rise = cfg["init"]
        kinematics = LegKinematics(JOINTS, [.11522, -.11522, .11522, -.11522],
                                   .23, .2315648, self.robot.model.jnt_range[self.robot.jids])
        self.rise = FootTrajectory(InitControllerConfig(
            duration=rise["duration"], support_duration=rise["support_duration"],
            support_depth=rise["support_depth"], standing_depth=rise["standing_depth"],
            timeout=rise["timeout"],
        ), kinematics)
        self.mode = Mode.WAIT
        self.command = np.zeros(3)
        self.last_control = None
        self.next_mpc = 0.
        self.stable_since = None
        self.handoff_started = None
        self.base_ref = None
        self.yaw_ref = 0.
        self.force_ref = np.zeros((4, 3))
        self.targets = None

    def update(self, now, state, raw_command, trigger=False):
        robot, rise = self.robot, self.cfg["init"]
        dt = self.cfg["simulation"]["control_dt"] if self.last_control is None else now-self.last_control
        self.last_control = now
        q = robot.data.qpos[robot.qadr]
        tilt = np.arccos(np.clip(state.rotation[2, 2], -1., 1.))

        if self.mode == Mode.WAIT:
            if not trigger:
                target = np.tile(self.cfg["robot"]["lying_angles"], 4)
                return robot.pd(target, rise["kp"], rise["kd"])
            self.rise.start(now, q)
            self.mode = Mode.RISE

        if self.mode in (Mode.RISE, Mode.SETTLE):
            if self.rise.timed_out(now):
                raise RuntimeError(f"起身超时，轨迹进度 {self.rise.progress:.2f}s，机身高度 {state.position[2]:.3f}m")
            if tilt > .55:
                raise RuntimeError(f"起身倾斜过大：{tilt:.2f} rad")
            support = np.count_nonzero(robot.contacts()[:, 2] > 2.) == 4
            target = self.rise.step(now, q, support)
            if self.rise.complete:
                self.mode = Mode.SETTLE
                stable = (support and state.position[2] > .33 and tilt < .20
                          and np.linalg.norm(state.velocity) < .08 and np.linalg.norm(state.omega) < .22)
                self.stable_since = (now if self.stable_since is None else self.stable_since) if stable else None
                if self.stable_since is not None and now-self.stable_since >= rise["settle_time"]:
                    self.mode = Mode.HANDOFF
                    self.handoff_started = now
                    self.gait.reset(state)
                    self.base_ref = state.position.copy()
                    self.yaw_ref = state.rpy[2]
                    # 保存站立构型下质心相对于 torso 的偏移，作为两个控制层之间的转换。
                    self.com_offset = state.rotation.T @ (state.com-state.position)
                else:
                    return robot.pd(target, rise["kp"], rise["kd"])
            else:
                return robot.pd(target, rise["kp"], rise["kd"])

        if state.position[2] < .22 or tilt > .75:
            raise RuntimeError(f"检测到跌倒：高度 {state.position[2]:.3f}m，倾角 {tilt:.2f}rad")
        alpha = 1.
        if self.mode == Mode.HANDOFF:
            alpha, _, _ = quintic((now-self.handoff_started)/rise["blend_time"])
            if alpha >= 1.:
                self.mode = Mode.STAND

        gait_cfg = self.cfg["gait"]
        limit = np.asarray(gait_cfg["command_limit"])
        goal = np.clip(raw_command, -limit, limit) if self.mode != Mode.HANDOFF else np.zeros(3)
        # 多轴满量程同时叠加会使某些角落的步幅过大。归一化保持指令方向，
        # 并让单轴仍然可以达到配置中的各自上限。
        goal /= max(1., np.linalg.norm(goal/limit))
        slew = np.asarray(gait_cfg["command_slew"]) * dt
        self.command += np.clip(goal-self.command, -slew, slew)
        previous = self.gait.last_contact.copy()
        self.targets = self.gait.step(now, state, self.command, goal)
        if self.mode != Mode.HANDOFF:
            if self.gait.walking:
                self.mode = Mode.STOPPING if self.gait.stopping else Mode.TROT
            else:
                self.mode = Mode.STAND

        # 以目标航向把机体速度旋转到世界系，限制长时间误差积分导致的参考漂移。
        cmd = self.command if self.gait.walking else np.zeros(3)
        velocity_world = yaw_rotation(self.yaw_ref) @ np.r_[cmd[:2], 0.]
        self.base_ref[:2] += dt*velocity_world[:2]
        self.base_ref[:2] = state.position[:2] + np.clip(self.base_ref[:2]-state.position[:2], -.12, .12)
        self.base_ref[2] += np.clip(self.cfg["robot"]["body_height"]-self.base_ref[2], -.12*dt, .12*dt)
        yaw_error = (self.yaw_ref-state.rpy[2]+np.pi) % (2*np.pi)-np.pi
        self.yaw_ref = state.rpy[2]+np.clip(yaw_error+dt*cmd[2], -.35, .35)

        # 接触切换会立刻刷新力计划。否则使用上一轮 MPC 的首拍力作为 WBC 的参考。
        if now >= self.next_mpc-1e-8 or np.any(previous != self.targets.contact):
            self.next_mpc = now+self.cfg["mpc"]["dt"]
            contact, feet = self.gait.forecast(now)
            contact[0] = self.targets.contact
            reference = self.reference(velocity_world, cmd[2])
            self.force_ref = self.mpc.solve(state, robot.inertia_world(), reference, contact, feet)

        torque = self.wbc.solve(state, self.targets, self.base_ref, velocity_world, self.yaw_ref,
                                cmd[2], self.force_ref)
        if self.mode == Mode.HANDOFF:
            pd = robot.pd(self.rise.final_target, rise["kp"], rise["kd"])
            torque = (1.-alpha)*pd + alpha*torque
        return torque

    def reference(self, velocity_world, yaw_rate):
        """预测每个 x[k+1] 参考值，x[0] 则始终取仿真测量。"""
        mpc = self.cfg["mpc"]
        times = mpc["dt"] * np.arange(1, mpc["horizon"]+1)
        reference = np.zeros((mpc["horizon"], 12))
        reference[:, 2] = self.yaw_ref + times*yaw_rate
        reference[:, 3:6] = self.base_ref + times[:, None]*velocity_world
        for row in reference:
            row[3:6] += yaw_rotation(row[2]) @ self.com_offset
        reference[:, 8] = yaw_rate
        reference[:, 9:12] = velocity_world
        return reference
