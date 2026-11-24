"""MuJoCo 适配层：名称映射、真值状态、质量矩阵、雅可比与力矩接口。

不依赖旧工程的 RL 观测/动作格式。所有控制模块采用逐腿排列：
FL(roll,pitch,knee), FR(...), HL(...), HR(...)。
"""
from dataclasses import dataclass
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from .config import DATA
from .terrain import RoughTerrain

LEGS = ("fl", "fr", "hl", "hr")
JOINTS = tuple(f"{leg}_{joint}_joint" for leg in LEGS for joint in ("hip_roll", "hip_pitch", "knee"))
SCENES = {"flat": "bpx.xml", "rough": "rough.xml"}


@dataclass
class State:
    position: np.ndarray       # torso 原点，世界系
    rotation: np.ndarray       # 世界 <- 机体
    rpy: np.ndarray
    omega: np.ndarray          # 世界系角速度
    velocity: np.ndarray       # torso 原点世界系线速度
    com: np.ndarray            # 整机质心，世界系
    com_velocity: np.ndarray
    feet: np.ndarray           # 4x3，足端碰撞体中心
    foot_velocity: np.ndarray


class Robot:
    def __init__(self, cfg, terrain="flat"):
        self.cfg = cfg
        self.model = mujoco.MjModel.from_xml_path(str(DATA / "bpx" / SCENES[terrain]))
        self.data = mujoco.MjData(self.model)
        m = self.model
        m.opt.timestep = cfg["simulation"]["timestep"]
        m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        m.geom_friction[:, 0] = cfg["robot"]["friction"]
        self.base = self.id(mujoco.mjtObj.mjOBJ_BODY, "torso")
        self.jids = np.array([self.id(mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINTS])
        self.qadr = m.jnt_qposadr[self.jids]
        self.vadr = m.jnt_dofadr[self.jids]
        self.aids = np.array([self.id(mujoco.mjtObj.mjOBJ_ACTUATOR, n + "_motor") for n in JOINTS])
        self.sites = [self.id(mujoco.mjtObj.mjOBJ_SITE, leg + "_foot") for leg in LEGS]
        self.toes = [self.id(mujoco.mjtObj.mjOBJ_BODY, leg + "_toe_link") for leg in LEGS]
        self.floor = self.id(mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.rough_ground = None
        self.terrain = None
        self.ground_geoms = {self.floor}
        if terrain == "rough":
            self.rough_ground = self.id(mujoco.mjtObj.mjOBJ_GEOM, "rough_ground")
            self.terrain = RoughTerrain(m, self.rough_ground)
            self.ground_geoms.add(self.rough_ground)
        if m.nv != 18 or m.nq != 19 or m.nu != 12:
            raise ValueError("要求 6 自由度浮动基座 + 12 个单自由度关节")
        self.mass = float(mujoco.mj_getTotalmass(m))
        self.nominal_q = np.tile(cfg["robot"]["standing_angles"], 4)
        self.torque_limit = min(cfg["robot"]["torque_limit"], float(np.min(m.actuator_ctrlrange[:, 1])))
        self.reset()

    def id(self, kind, name):
        index = mujoco.mj_name2id(self.model, kind, name)
        if index < 0:
            raise ValueError(f"模型缺少 {name}")
        return index

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        if self.terrain is not None:
            self.terrain.tile.fill(0)
            self.terrain.update()
        self.data.qpos[:3] = [0., 0., self.cfg["robot"]["lying_height"]]
        self.data.qpos[3:7] = [1., 0., 0., 0.]
        self.data.qpos[self.qadr] = np.tile(self.cfg["robot"]["lying_angles"], 4)
        self.forward()

    def forward(self):
        # mj_step 积分后的派生量可能仍对应上一个状态；控制前显式更新。
        mujoco.mj_forward(self.model, self.data)

    def ground_height(self, x, y):
        """沿竖直方向查询足底下方的可碰撞地表高度。"""
        if self.rough_ground is None:
            return 0.
        origin = np.array([x, y, 1.])
        down = np.array([0., 0., -1.])
        distance = mujoco.mj_rayHfield(self.model, self.data, self.rough_ground, origin, down)
        return max(0., origin[2]-distance) if distance >= 0. else 0.

    def world_position(self, position):
        if self.terrain is None:
            return position
        return position + np.r_[self.terrain.origin, 0.]

    def jacobians(self):
        jac, dots = [], []
        for site, body in zip(self.sites, self.toes):
            jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, jp, jr, site)
            dp, dr = np.zeros_like(jp), np.zeros_like(jr)
            mujoco.mj_jacDot(self.model, self.data, dp, dr, self.data.site_xpos[site], body)
            jac.append(jp)
            dots.append(dp @ self.data.qvel)
        return np.array(jac), np.array(dots)

    def base_jacobians(self):
        jp, jr = np.zeros((3, 18)), np.zeros((3, 18))
        mujoco.mj_jacBody(self.model, self.data, jp, jr, self.base)
        return jp, jr

    def base_bias_acceleration(self):
        """机身位置与姿态任务中的 Jdot(q,v) v，使用世界坐标。"""
        dp, dr = np.zeros((3, 18)), np.zeros((3, 18))
        mujoco.mj_jacDot(self.model, self.data, dp, dr, self.data.xpos[self.base], self.base)
        return dp @ self.data.qvel, dr @ self.data.qvel

    def state(self):
        d = self.data
        R = d.xmat[self.base].reshape(3, 3).copy()
        jp, jr = self.base_jacobians()
        jc = np.zeros((3, 18))
        mujoco.mj_jacSubtreeCom(self.model, d, jc, self.base)
        jf, _ = self.jacobians()
        return State(d.xpos[self.base].copy(), R, Rotation.from_matrix(R).as_euler("xyz"),
                     jr @ d.qvel, jp @ d.qvel, d.subtree_com[self.base].copy(),
                     jc @ d.qvel, d.site_xpos[self.sites].copy(), jf @ d.qvel)

    def inertia_world(self):
        """冻结当前关节构型，将每个连杆惯量平移到整机质心后求和。

        MPC 的单刚体近似使用该惯量；WBC 仍使用完整 18x18 质量矩阵。
        """
        I = np.zeros((3, 3))
        com = self.data.subtree_com[self.base]
        for body in range(1, self.model.nbody):
            R = self.data.ximat[body].reshape(3, 3)
            r = self.data.xipos[body] - com
            I += R @ np.diag(self.model.body_inertia[body]) @ R.T
            I += self.model.body_mass[body] * ((r @ r)*np.eye(3) - np.outer(r, r))
        return I

    def dynamics(self):
        M = np.zeros((18, 18))
        mujoco.mj_fullM(self.model, self.data, M)
        # 被动力（如阻尼）在方程右侧，移到左侧应取负号。
        h = self.data.qfrc_bias - self.data.qfrc_passive
        return M, h.copy()

    def contacts(self):
        forces = np.zeros((4, 3))
        for idx, c in enumerate(self.data.contact):
            g1, g2 = int(c.geom1), int(c.geom2)
            ground_first = g1 in self.ground_geoms
            if not ground_first and g2 not in self.ground_geoms:
                continue
            other = g2 if ground_first else g1
            body = int(self.model.geom_bodyid[other])
            if body not in self.toes:
                continue
            f = np.zeros(6)
            # contact 的局部第一轴是法线；力作用在 geom2，需根据顺序翻转。
            mujoco.mj_contactForce(self.model, self.data, idx, f)
            fw = c.frame.reshape(3, 3).T @ f[:3]
            forces[self.toes.index(body)] += fw if ground_first else -fw
        return forces

    def apply(self, torque):
        if np.shape(torque) != (12,) or not np.isfinite(torque).all():
            raise FloatingPointError("关节力矩必须为 12 维有限向量")
        self.data.ctrl[self.aids] = np.clip(torque, -self.torque_limit, self.torque_limit)

    def pd(self, target, kp, kd):
        return np.clip(kp*(target-self.data.qpos[self.qadr]) - kd*self.data.qvel[self.vadr],
                       -self.torque_limit, self.torque_limit)
