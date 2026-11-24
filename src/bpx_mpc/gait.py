"""固定接触时序、Raibert 风格落足点与 C² 摆腿轨迹。

接触时序由步态调度器给定；MPC 优化的是力，不会自动发现步态。
松杆后完成正在进行的摆腿，四足支撑时切回 STAND，避免悬空锁腿。
"""
from dataclasses import dataclass
import numpy as np
from .math_utils import quintic, yaw_rotation


@dataclass
class FootTargets:
    contact: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray


def swing_trajectory(start, end, phase, duration, height):
    """水平五次插值；竖直抬脚为 64*s³(1-s)³，峰值恰为 height。

    端点的位置、速度、加速度均连续。直接解析求导，避免数值差分噪声。
    """
    s = float(np.clip(phase, 0, 1))
    a, da, dda = quintic(s)
    bump = 64*(s**3-3*s**4+3*s**5-s**6)
    dbump = 64*(3*s**2-12*s**3+15*s**4-6*s**5)
    ddbump = 64*(6*s-36*s**2+60*s**3-30*s**4)
    p = start + a*(end-start) + np.array([0., 0., height*bump])
    v = (da*(end-start)+[0., 0., height*dbump])/duration
    acc = (dda*(end-start)+[0., 0., height*ddbump])/duration**2
    return p, v, acc


class GaitPlanner:
    offsets = np.array([0., .5, .5, 0.])  # FL+HR 与 FR+HL 两组对角腿

    def __init__(self, cfg, ground_height=None):
        self.cfg = cfg
        self.ground_height = ground_height
        self.walking = False
        self.stopping = False
        self.started = 0.
        self.last_contact = np.ones(4, dtype=bool)
        self.anchors = None
        self.start = np.zeros((4, 3))
        self.end = np.zeros((4, 3))

    def reset(self, state):
        self.walking = self.stopping = False
        self.last_contact[:] = True
        self.anchors = state.feet.copy()
        # 站姿足端相对机身的水平偏置，用于左右/前后对称的落足参考。
        self.nominal = (state.feet-state.position) @ state.rotation
        self.nominal[:, 2] = 0.

    def shift_origin(self, delta):
        self.anchors[:, :2] -= delta
        self.start[:, :2] -= delta
        self.end[:, :2] -= delta

    def phase(self, now):
        return ((now-self.started)/self.cfg["gait"]["period"] + self.offsets) % 1.

    def contacts_at(self, now):
        if not self.walking:
            return np.ones(4, dtype=bool)
        return self.phase(now) < self.cfg["gait"]["duty_factor"]

    @staticmethod
    def activity(command):
        return max(np.linalg.norm(command[:2]), abs(command[2])*.3)

    def step(self, now, state, command, goal):
        c = self.cfg["gait"]
        activity = self.activity(command)
        if not self.walking and activity > c["start_threshold"]:
            self.walking, self.started = True, now
        if self.walking:
            self.stopping = self.activity(goal) < c["stop_threshold"]
        contact = self.contacts_at(now)
        # 只在全支撑窗口接纳停止请求。当前已有摆腿必先落地。
        if self.stopping and activity < c["stop_threshold"] and contact.all():
            self.walking = self.stopping = False
            contact[:] = True
        phase = self.phase(now) if self.walking else np.zeros(4)
        p, v, a = self.anchors.copy(), np.zeros((4, 3)), np.zeros((4, 3))
        Rz = yaw_rotation(state.rpy[2])
        desired_v = Rz @ np.r_[command[:2], 0.]
        for leg in range(4):
            if self.last_contact[leg] and not contact[leg]:
                self.start[leg] = state.feet[leg]
                hip = Rz @ self.nominal[leg]
                # 预测摆腿结束时的机身位置，加半个支撑周期前馈与速度误差反馈。
                Ts = c["period"]*(1-c["duty_factor"])
                Tstance = c["period"]*c["duty_factor"]
                velocity = desired_v + np.cross([0., 0., command[2]], hip)
                delta = (Ts+.5*Tstance)*velocity + c["velocity_feedback"]*(state.velocity-desired_v)
                delta[:2] = np.clip(delta[:2], -c["max_step"], c["max_step"])
                self.end[leg] = state.position + hip + delta
                self.end[leg, 2] = self.cfg["robot"]["foot_radius"]
                if self.ground_height is not None:
                    self.end[leg, 2] += self.ground_height(*self.end[leg, :2])
            elif not self.last_contact[leg] and contact[leg]:
                self.anchors[leg] = self.end[leg]
            if contact[leg]:
                p[leg] = self.anchors[leg]
            else:
                s = (phase[leg]-c["duty_factor"])/(1-c["duty_factor"])
                p[leg], v[leg], a[leg] = swing_trajectory(self.start[leg], self.end[leg], s,
                    c["period"]*(1-c["duty_factor"]), c["swing_height"])
        self.last_contact = contact.copy()
        return FootTargets(contact, p, v, a)

    def forecast(self, now):
        N, dt = self.cfg["mpc"]["horizon"], self.cfg["mpc"]["dt"]
        contacts = np.array([self.contacts_at(now+k*dt) for k in range(N)])
        feet = np.tile(self.anchors, (N, 1, 1))
        # 尚未实际落地的脚使用已规划的落足点。远期接触位置采用冻结近似。
        for leg in range(4):
            if not self.last_contact[leg]:
                feet[:, leg] = self.end[leg]
        feet[:, :, 2] -= self.cfg["robot"]["foot_radius"]
        return contacts, feet
