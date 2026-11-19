"""单刚体凸 MPC：在固定接触时序下优化未来接触力。

状态 x=[roll,pitch,yaw, px,py,pz, wx,wy,wz, vx,vy,vz]，全部速度为世界系。
控制 u=[f_FL(3), f_FR(3), f_HL(3), f_HR(3)]，表示地面对机器人的力。
仅在当前滚动窗口内冻结惯量与足端力臂，QP 不包含神经网络或训练过程。
"""
import numpy as np
from scipy import sparse
from .math_utils import skew, yaw_rotation
from .qp import QPSolver, force_constraints


def discrete_dynamics(dt, mass, inertia, yaw, lever_arms):
    """小 roll/pitch 线性模型；加入常加速度的 dt²/2 项。

    欧拉角速度近似 Rz(yaw).T @ omega_world；此近似不适用于翻滚运动。
    """
    T = yaw_rotation(yaw).T
    A = np.eye(12)
    A[:3, 6:9] = dt*T
    A[3:6, 9:12] = dt*np.eye(3)
    B = np.zeros((12, 12))
    for leg, r in enumerate(lever_arms):
        angular = np.linalg.solve(inertia, skew(r))
        sl = slice(3*leg, 3*leg+3)
        B[:3, sl] = .5*dt**2*T @ angular
        B[3:6, sl] = .5*dt**2/mass*np.eye(3)
        B[6:9, sl] = dt*angular
        B[9:12, sl] = dt/mass*np.eye(3)
    g = np.zeros(12)
    g[5], g[11] = -.5*9.81*dt**2, -9.81*dt
    return A, B, g


class ConvexMPC:
    def __init__(self, cfg, mass):
        self.cfg, self.mass = cfg, mass
        self.qp = QPSolver("MPC")
        self.predicted_states = None
        self.predicted_forces = None

    def solve(self, state, inertia, reference, contacts, predicted_feet):
        cfg = self.cfg["mpc"]
        N, dt = cfg["horizon"], cfg["dt"]
        nx, nu = 12, 12
        # 决策变量 z=[x1,...,xN,u0,...,u(N-1)]；x0 是测量值，不是自由变量。
        x0 = np.r_[state.rpy, state.com, state.omega, state.com_velocity]
        # yaw 在 ±pi 附近展开到参考附近，防止 QP 追踪多转一圈。
        x0[2] = reference[0, 2] + (x0[2]-reference[0, 2]+np.pi) % (2*np.pi)-np.pi
        Q = np.asarray(cfg["state_weights"])
        P = sparse.diags(np.r_[np.tile(2*Q, N), np.full(N*nu, 2*cfg["force_weight"])], format="csc")
        linear = np.r_[(-2*reference*Q).ravel(), np.zeros(N*nu)]
        rows, values = [], []
        cones, lows, ups = [], [], []
        for k in range(N):
            A, B, g = discrete_dynamics(dt, self.mass, inertia, state.rpy[2],
                                         predicted_feet[k]-state.com)
            row = sparse.lil_matrix((nx, N*(nx+nu)))
            row[:, k*nx:(k+1)*nx] = np.eye(nx)
            if k > 0:
                row[:, (k-1)*nx:k*nx] = -A
            row[:, N*nx+k*nu:N*nx+(k+1)*nu] = -B
            rows.append(row.tocsc())
            values.append(g+A@x0 if k == 0 else g)
            C, lo, up = force_constraints(contacts[k], self.cfg["robot"]["friction"], cfg["max_normal_force"])
            cones.append(C)
            lows.append(lo)
            ups.append(up)
        equality = sparse.vstack(rows, format="csc")
        inequality = sparse.hstack([sparse.csc_matrix((20*N, nx*N)), sparse.block_diag(cones)], format="csc")
        Aqp = sparse.vstack([equality, inequality], format="csc")
        rhs = np.concatenate(values)
        z = self.qp.solve(P, linear, Aqp, np.r_[rhs, np.concatenate(lows)],
                         np.r_[rhs, np.concatenate(ups)])
        self.predicted_states = z[:nx*N].reshape(N, nx)
        self.predicted_forces = z[nx*N:].reshape(N, 4, 3)
        return self.predicted_forces[0].copy()
