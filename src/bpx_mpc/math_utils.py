"""坐标工具：R 将机体系向量变换到世界系；角速度统一在调用处标明。"""
import numpy as np
from scipy.spatial.transform import Rotation


def skew(v):
    """[v]× @ w = v × w，用于把接触力变为质心力矩。"""
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def yaw_rotation(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def orientation_error(target, current):
    """世界系旋转向量误差，比直接相减欧拉角更适合 WBC 姿态反馈。"""
    return Rotation.from_matrix(target @ current.T).as_rotvec()


def quintic(s):
    """五次标度及其对归一化时间 s 的一、二阶导数。"""
    s = np.clip(s, 0., 1.)
    return (10*s**3 - 15*s**4 + 6*s**5,
            30*s**2 - 60*s**3 + 30*s**4,
            60*s - 180*s**2 + 120*s**3)
