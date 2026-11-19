"""运行入口。命令源只生成速度和起身事件，控制器与输入设备解耦。"""
import argparse
from contextlib import ExitStack
import csv
from pathlib import Path
import sys
import time

import mujoco
import numpy as np

from .config import load_config
from .controller import Controller, Mode
from .gamepad import BTN_TL, BTN_TR, GamepadCommandSource, GamepadMapping, enumerate_gamepads, select_gamepad
from .qp import QPFailure
from .robot import JOINTS, LEGS


def parser():
    p = argparse.ArgumentParser(description="BPX MuJoCo 足式控制仿真")
    p.add_argument("--config", type=Path)
    p.add_argument("--duration", type=float, default=16.)
    p.add_argument("--viewer", action="store_true", help="打开 MuJoCo 窗口并按墙钟播放")
    p.add_argument("--gamepad", action="store_true", help="用手柄控制，RB 触发起身")
    p.add_argument("--list-gamepads", action="store_true")
    p.add_argument("--gamepad-device", type=Path)
    p.add_argument("--gamepad-index", type=int)
    p.add_argument("--gamepad-axes", nargs=3, type=int, default=(1, 0, 3), metavar=("VX", "VY", "WZ"))
    p.add_argument("--gamepad-signs", nargs=3, type=float, default=(-1., -1., -1.), metavar=("VX", "VY", "WZ"))
    p.add_argument("--gamepad-deadzone", type=float, default=.1)
    p.add_argument("--gamepad-deadman", action="store_true", help="按住 LB 才接受速度指令")
    p.add_argument("--auto-init", action="store_true", help="手柄模式也自动起身")
    p.add_argument("--vx", type=float, default=0.)
    p.add_argument("--vy", type=float, default=0.)
    p.add_argument("--wz", type=float, default=0.)
    p.add_argument("--move-at", type=float, default=6., help="固定速度指令的开始仿真时间")
    p.add_argument("--stop-at", type=float, default=None, help="固定速度指令归零的仿真时间")
    p.add_argument("--log", type=Path, help="写入 CSV，包括求解状态、接触力和电机力矩")
    return p


def log_row(controller, now, state, raw):
    c = controller
    body_v = state.rotation.T @ state.velocity
    row = {"time": now, "mode": c.mode.value, "init_progress": c.rise.progress,
           **dict(zip(("x", "y", "height"), state.position)),
           **dict(zip(("roll", "pitch", "yaw"), state.rpy)),
           "vx": body_v[0], "vy": body_v[1], "wz": state.omega[2]}
    for i, name in enumerate(("vx", "vy", "wz")):
        row["raw_"+name] = raw[i]
        row["cmd_"+name] = c.command[i]
    actual_force = c.robot.contacts()
    for i, leg in enumerate(LEGS):
        row[f"{leg}_planned_contact"] = int(c.targets.contact[i]) if c.targets is not None else -1
        for j, axis in enumerate("xyz"):
            row[f"{leg}_actual_f{axis}"] = actual_force[i, j]
            row[f"{leg}_mpc_f{axis}"] = c.force_ref[i, j]
            row[f"{leg}_wbc_f{axis}"] = c.wbc.force[i, j]
    for name, aid in zip(JOINTS, c.robot.aids):
        row["tau_"+name] = c.robot.data.ctrl[aid]
    for name, qp in (("mpc", c.mpc.qp), ("wbc", c.wbc.qp)):
        row[name+"_status"] = qp.info.status
        row[name+"_ms"] = qp.info.seconds*1000
        row[name+"_violation"] = qp.info.violation
    return row


def run(cfg, args):
    controller = Controller(cfg)
    robot = controller.robot
    dt = cfg["simulation"]["timestep"]
    control_stride = round(cfg["simulation"]["control_dt"]/dt)
    log_stride = round(cfg["simulation"]["log_dt"]/dt)
    fixed = np.array([args.vx, args.vy, args.wz])
    mode = None
    writer = None
    raw = np.zeros(3)
    with ExitStack() as stack:
        gamepad = None
        if args.gamepad:
            device = select_gamepad(args.gamepad_device, args.gamepad_index)
            mapping = GamepadMapping(*args.gamepad_axes, *args.gamepad_signs)
            gamepad = stack.enter_context(GamepadCommandSource(
                device, tuple(cfg["gait"]["command_limit"]), deadzone=args.gamepad_deadzone, mapping=mapping,
                deadman_button=BTN_TL if args.gamepad_deadman else None, init_button=BTN_TR))
            print(f"手柄 {device.label}。按 RB 开始起身。", flush=True)
        viewer = None
        if args.viewer:
            from mujoco.viewer import launch_passive
            viewer = stack.enter_context(launch_passive(robot.model, robot.data))
            with viewer.lock():
                viewer.cam.trackbodyid = robot.base
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                viewer.cam.distance = 2.
                viewer.opt.geomgroup[3] = 0
        stream = None
        if args.log:
            args.log.parent.mkdir(parents=True, exist_ok=True)
            stream = stack.enter_context(args.log.open("w", newline="", encoding="utf-8"))
        start_wall = time.monotonic()
        nsteps = round(args.duration/dt)
        for step in range(nsteps):
            now = step*dt
            if viewer and not viewer.is_running():
                break
            if step % control_stride == 0 or (stream and step % log_stride == 0):
                robot.forward()
                state = robot.state()
            if step % control_stride == 0:
                if gamepad:
                    try:
                        raw = gamepad.read()
                        trigger = gamepad.consume_init_request() or (args.auto_init and step == 0)
                    except OSError as exc:
                        print(f"手柄输入中断，速度指令归零：{exc}", file=sys.stderr)
                        gamepad.close()
                        gamepad = None
                        raw, trigger = np.zeros(3), False
                else:
                    moving = not args.gamepad and now >= args.move_at and (args.stop_at is None or now < args.stop_at)
                    raw = fixed if moving else np.zeros(3)
                    trigger = (not args.gamepad or args.auto_init) and step == 0
                try:
                    torque = controller.update(now, state, raw, trigger)
                    robot.apply(torque)
                except (QPFailure, RuntimeError, ValueError, FloatingPointError) as exc:
                    robot.apply(np.zeros(12))
                    raise RuntimeError(f"t={now:.3f}s，{controller.mode}：{exc}") from exc
                if controller.mode != mode:
                    mode = controller.mode
                    print(f"t={now:6.3f}s  {mode:8}  z={state.position[2]:.3f} m", flush=True)
            if stream and step % log_stride == 0:
                row = log_row(controller, now, state, raw)
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=row.keys())
                    writer.writeheader()
                writer.writerow(row)
            mujoco.mj_step(robot.model, robot.data)
            if viewer and step % max(1, round(1/(60*dt))) == 0:
                viewer.sync()
            if viewer or args.gamepad:
                delay = start_wall+(step+1)*dt-time.monotonic()
                if delay > 0:
                    time.sleep(delay)
        robot.forward()
        print(f"仿真 {robot.data.time:.2f}s，机身位置 {np.round(robot.data.qpos[:3], 3)}", flush=True)
    return controller


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.list_gamepads:
        devices = enumerate_gamepads()
        for i, device in enumerate(devices):
            print(f"[{i}] {device.label}")
        if not devices:
            print("未发现 Linux 游戏手柄")
        return 0
    numbers = [args.duration, args.move_at, args.vx, args.vy, args.wz]
    if args.stop_at is not None:
        numbers.append(args.stop_at)
    if (not np.isfinite(numbers).all() or args.duration <= 0 or args.move_at < 0
            or (args.stop_at is not None and args.stop_at <= args.move_at)):
        p.error("仿真时长、速度及起停时间必须为有效数值，且 stop-at 必须晚于 move-at")
    try:
        run(load_config(args.config), args)
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"仿真停止：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
