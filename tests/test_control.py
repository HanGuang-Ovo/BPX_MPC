"""无窗口验证起身、全身动力学、跟踪与手柄映射。"""
import csv
import os
from pathlib import Path

import numpy as np
import pytest

from bpx_mpc.cli import parser, run
from bpx_mpc.config import load_config
from bpx_mpc.controller import Controller, Mode
from bpx_mpc.gait import swing_trajectory
from bpx_mpc.gamepad import (
    ABS_RX, ABS_X, ABS_Y, BTN_TL, BTN_TR, EVENT_STRUCT, EV_ABS, EV_KEY,
    AxisCalibration, GamepadCommandSource, InputDevice,
)
from bpx_mpc.mpc import discrete_dynamics
from bpx_mpc.robot import JOINTS, LEGS


def test_wait_and_stand_dynamics():
    c = Controller(load_config())
    robot = c.robot
    torque = c.update(0., robot.state(), np.array([.2, 0., 0.]))
    assert c.mode == Mode.WAIT
    np.testing.assert_allclose(torque, 0.)
    np.testing.assert_allclose(c.command, 0.)
    # 对称足位下，均分重力应令单刚体维持静止。
    A, B, gravity = discrete_dynamics(.04, robot.mass, np.eye(3), 0.,
                                      [[.2, .1, -.4], [.2, -.1, -.4], [-.2, .1, -.4], [-.2, -.1, -.4]])
    x = np.zeros(12)
    x[5] = .4
    forces = np.tile([0., 0., robot.mass*9.81/4], 4)
    np.testing.assert_allclose(A @ x + B @ forces + gravity, x, atol=1e-12)


def test_swing_endpoints():
    start = np.array([0., 0., .027])
    finish = np.array([.1, .05, .027])
    for phase, expected in ((0., start), (1., finish)):
        position, velocity, acceleration = swing_trajectory(start, finish, phase, .24, .06)
        np.testing.assert_allclose(position, expected)
        np.testing.assert_allclose(velocity, 0.)
        np.testing.assert_allclose(acceleration, 0.)
    middle, _, _ = swing_trajectory(start, finish, .5, .24, .06)
    assert middle[2] == pytest.approx(.087)


def test_linux_gamepad_axes_and_button_edges():
    # 用 Linux input_event 字节流验证原手柄代码，无需物理设备。
    device = InputDevice(Path("/dev/input/mock"), "test", 0, 0, "-")
    gamepad = GamepadCommandSource(device, (.4, .25, .7), deadman_button=BTN_TL, init_button=BTN_TR)
    rd, wr = os.pipe2(os.O_NONBLOCK)
    gamepad._file_descriptor = rd
    for axis in (ABS_X, ABS_Y, ABS_RX):
        gamepad._values[axis] = 0
        gamepad._calibrations[axis] = AxisCalibration(-100, 100, 0)

    def event(kind, code, value):
        os.write(wr, EVENT_STRUCT.pack(0, 0, kind, code, value))

    try:
        event(EV_ABS, ABS_Y, -100)
        event(EV_ABS, ABS_X, 100)
        event(EV_ABS, ABS_RX, -100)
        event(EV_KEY, BTN_TR, 1)
        np.testing.assert_allclose(gamepad.read(), 0.)
        assert gamepad.consume_init_request()
        assert not gamepad.consume_init_request()
        event(EV_KEY, BTN_TL, 1)
        np.testing.assert_allclose(gamepad.read(), [.4, -.25, .7])
        event(EV_KEY, BTN_TL, 0)
        np.testing.assert_allclose(gamepad.read(), 0.)
    finally:
        os.close(wr)
        gamepad.close()


@pytest.mark.parametrize("command", [(0., 0., 0.), (.25, 0., 0.), (-.22, 0., 0.), (0., .16, 0.),
                                     (0., 0., .45), (.22, -.12, -.35), (.4, .25, .7), (-.4, -.25, -.7)])
def test_closed_loop(command, tmp_path):
    log = tmp_path / "sim.csv"
    args = parser().parse_args(["--duration", "15", "--stop-at", "11", "--log", str(log),
                               "--vx", str(command[0]), "--vy", str(command[1]), "--wz", str(command[2])])
    cfg = load_config()
    c = run(cfg, args)
    with log.open() as stream:
        rows = list(csv.DictReader(stream))
    modes = [row["mode"] for row in rows]
    assert modes[:1] == ["RISE"]
    assert modes.index("SETTLE") < modes.index("HANDOFF") < modes.index("STAND")
    assert all(row["mpc_status"] == "idle" for row in rows[:modes.index("HANDOFF")])
    assert c.mode == Mode.STAND
    assert 0.38 < float(rows[-1]["height"]) < 0.42
    assert max(abs(float(rows[-1][k])) for k in ("vx", "vy", "wz")) < .02
    if any(command):
        assert modes.index("STAND") < modes.index("TROT") < modes.index("STOPPING")
        settled_walk = [r for r in rows if 7. <= float(r["time"]) < 10.8]
        actual = np.array([[float(r[k]) for k in ("vx", "vy", "wz")] for r in settled_walk])
        expected = np.asarray(command)/max(1., np.linalg.norm(np.asarray(command)/cfg["gait"]["command_limit"]))
        np.testing.assert_allclose(actual.mean(axis=0), expected, atol=.06)
        patterns = {tuple(int(r[f"{leg}_planned_contact"]) for leg in LEGS) for r in settled_walk}
        assert {(1, 0, 0, 1), (0, 1, 1, 0), (1, 1, 1, 1)} <= patterns
    for row in rows:
        assert all(abs(float(row["tau_"+j])) <= cfg["robot"]["torque_limit"] for j in JOINTS)
        for name in ("mpc", "wbc"):
            assert row[f"{name}_status"] in ("idle", "solved", "solved inaccurate")
            assert float(row[f"{name}_violation"]) < .02
            for leg in LEGS:
                fx, fy, fz = [float(row[f"{leg}_{name}_f{ax}"]) for ax in "xyz"]
                bound = cfg["mpc"]["max_normal_force"] if int(row[f"{leg}_planned_contact"]) == 1 else 0.
                assert -.02 <= fz <= bound+.02
                assert abs(fx)+abs(fy) <= cfg["robot"]["friction"]*fz+.02
    # 检查末帧的真实支撑力和电机力矩方向能平衡重力。
    normal = sum(float(rows[-1][f"{leg}_actual_fz"]) for leg in LEGS)
    assert normal == pytest.approx(c.robot.mass*9.81, rel=.05)
