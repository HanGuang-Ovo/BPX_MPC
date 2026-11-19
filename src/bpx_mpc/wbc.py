"""全身逆动力学 QP（加权任务形式）。

决策变量 z=[dv(18), f(12), tau(12)]，总计 42 维。
硬约束：M dv + h = S.T tau + J.T f；支撑脚加速度约束；
摩擦棱锥；摆腿接触力为零；关节力矩/预测速度/预测关节位置限制。
软目标：机身姿态/位置、摆动足轨迹、MPC 接触力、关节姿态正则。
"""
import numpy as np
from scipy import sparse
from .math_utils import orientation_error, yaw_rotation
from .qp import QPSolver, force_constraints


class WholeBodyController:
    def __init__(self, cfg, robot):
        self.cfg, self.robot = cfg, robot
        self.qp = QPSolver("WBC")
        self.force = np.zeros((4, 3))
        self.acceleration = np.zeros(18)

    def solve(self, state, targets, base_ref, velocity_ref, yaw_ref, yaw_rate, forces):
        robot, c = self.robot, self.cfg["wbc"]
        M, h = robot.dynamics()
        jac, jdotv = robot.jacobians()
        jp, jr = robot.base_jacobians()
        linear_bias, angular_bias = robot.base_bias_acceleration()
        P, q = 1e-7*np.eye(42), np.zeros(42)

        def task(matrix, target, weight):
            # w ||T z - y||² -> P += 2w T.T T，q -= 2w T.T y。
            nonlocal P, q
            P += 2*weight*matrix.T @ matrix
            q -= 2*weight*matrix.T @ target

        T = np.zeros((3, 42))
        T[:, :18] = jp
        desired = np.asarray(c["body_kp"])*(base_ref-state.position)
        desired += np.asarray(c["body_kd"])*(velocity_ref-state.velocity)
        task(T, desired-linear_bias, c["body_weight"])
        T[:, :18] = jr
        desired = np.asarray(c["orientation_kp"])*orientation_error(yaw_rotation(yaw_ref), state.rotation)
        desired += np.asarray(c["orientation_kd"])*([0., 0., yaw_rate]-state.omega)
        task(T, desired-angular_bias, c["orientation_weight"])
        equality = [np.hstack([M, -jac.reshape(12, 18).T, np.zeros((18, 12))])]
        # S 从广义速度中选择受控关节，基座对应的 6 行没有电机力矩。
        equality[0][robot.vadr, 30+np.arange(12)] = -1.
        rhs = [-h]
        for leg in range(4):
            T = np.zeros((3, 42))
            T[:, :18] = jac[leg]
            if targets.contact[leg]:
                # 位置/速度反馈抑制有限精度和 MuJoCo 软接触带来的漂移。
                value = c["stance_kp"]*(targets.position[leg]-state.feet[leg])
                value -= c["stance_kd"]*state.foot_velocity[leg] + jdotv[leg]
                equality.append(T)
                rhs.append(value)
            else:
                value = targets.acceleration[leg] + c["swing_kp"]*(targets.position[leg]-state.feet[leg])
                value += c["swing_kd"]*(targets.velocity[leg]-state.foot_velocity[leg])-jdotv[leg]
                task(T, value, c["swing_weight"])
        T = np.zeros((12, 42))
        T[:, 18:30] = np.eye(12)
        task(T, forces.ravel(), c["force_weight"])
        T = np.zeros((12, 42))
        T[np.arange(12), robot.vadr] = 1.
        postural = 10*(robot.nominal_q-robot.data.qpos[robot.qadr])-3*robot.data.qvel[robot.vadr]
        task(T, postural, c["posture_weight"])
        C, lo, up = force_constraints(targets.contact, self.cfg["robot"]["friction"],
                                      self.cfg["mpc"]["max_normal_force"])
        cone = sparse.hstack([sparse.csc_matrix((20, 18)), C, sparse.csc_matrix((20, 12))])
        # 一步预测限制提供基本关节保护；它不是全预测域碰撞/可达性约束。
        limit = np.zeros((24, 42))
        limit[:12, 30:] = np.eye(12)
        limit[12+np.arange(12), robot.vadr] = 1.
        dt = self.cfg["simulation"]["control_dt"]
        pos, vel = robot.data.qpos[robot.qadr], robot.data.qvel[robot.vadr]
        qrange = robot.model.jnt_range[robot.jids]
        amin = np.maximum((-20.-vel)/dt, 2*(qrange[:, 0]-pos-dt*vel)/dt**2)
        amax = np.minimum((20.-vel)/dt, 2*(qrange[:, 1]-pos-dt*vel)/dt**2)
        eq = np.vstack(equality)
        b = np.concatenate(rhs)
        A = sparse.vstack([sparse.csc_matrix(eq), cone, sparse.csc_matrix(limit)], format="csc")
        lower = np.r_[b, lo, np.full(12, -robot.torque_limit), amin]
        upper = np.r_[b, up, np.full(12, robot.torque_limit), amax]
        z = self.qp.solve(P, q, A, lower, upper)
        self.acceleration = z[:18]
        self.force = z[18:30].reshape(4, 3)
        return z[30:42]
