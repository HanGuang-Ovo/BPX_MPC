"""读取配置，并在启动物理仿真之前检查基本时序与物理范围。"""
from pathlib import Path
import tomllib
import numpy as np

DATA = Path(__file__).resolve().parent / "data"


def load_config(path: str | Path | None = None) -> dict:
    with open(path or DATA / "default.toml", "rb") as stream:
        cfg = tomllib.load(stream)
    sim, gait, robot, mpc = (cfg[k] for k in ("simulation", "gait", "robot", "mpc"))
    for group, values in cfg.items():
        for key, value in values.items():
            if not np.isfinite(np.asarray(value, dtype=float)).all():
                raise ValueError(f"配置必须是有限数：{group}.{key}")
    for dt in (sim["timestep"], sim["control_dt"], sim["log_dt"], mpc["dt"]):
        if dt <= 0:
            raise ValueError("时间间隔必须大于零")
        ratio = dt / sim["timestep"]
        if abs(ratio - round(ratio)) > 1e-8:
            raise ValueError("控制、MPC 和日志周期必须是物理步长的整数倍")
    if sim["control_dt"] < sim["timestep"] or mpc["dt"] < sim["control_dt"]:
        raise ValueError("要求物理步长 <= WBC 周期 <= MPC 周期")
    if not 0.5 < gait["duty_factor"] < 1 or gait["period"] <= 0:
        raise ValueError("教学 trot 要求支撑占空比在 (0.5,1) 内且周期为正")
    if not isinstance(mpc["horizon"], int) or mpc["horizon"] < 2:
        raise ValueError("MPC horizon 必须是至少 2 的整数")
    if len(mpc["state_weights"]) != 12 or min(mpc["state_weights"]) <= 0:
        raise ValueError("MPC 需要 12 个正状态权重")
    for key in ("friction", "torque_limit", "body_height", "foot_radius"):
        if robot[key] <= 0:
            raise ValueError(f"robot.{key} 必须为正")
    for group, key in (("gait", "command_limit"), ("gait", "command_slew"),
                       ("robot", "standing_angles"), ("robot", "lying_angles"),
                       ("wbc", "body_kp"), ("wbc", "body_kd"),
                       ("wbc", "orientation_kp"), ("wbc", "orientation_kd")):
        if np.shape(cfg[group][key]) != (3,):
            raise ValueError(f"{group}.{key} 必须有 3 项")
    if (min(gait["command_limit"]) <= 0 or min(gait["command_slew"]) <= 0
            or not 0 <= gait["stop_threshold"] < gait["start_threshold"]
            or cfg["init"]["duration"] <= cfg["init"]["support_duration"]
            or cfg["init"]["blend_time"] <= 0 or mpc["force_weight"] <= 0
            or mpc["max_normal_force"] <= 0):
        raise ValueError("速度限制、起停阈值、起身时序或 MPC 力参数无效")
    return cfg
