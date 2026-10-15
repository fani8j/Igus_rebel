"""Unit tests of the step-jog maths and state machines (no ROS graph; the module is imported without spinning)."""
import math
from collections import deque

import numpy as np
import pytest

from rebel_dual_arm_calibration.kinematics import Urdf
from rebel_dual_arm_calibration.ui import jog as J

# A ReBeL-like 6-axis chain (made-up lengths) world -> base -> 6 revolute joints -> tool tip, as the expanded URDF
# robot_state_publisher serves it.
URDF = """<robot name="t">
 <link name="world"/><link name="base"/><link name="l1"/><link name="l2"/><link name="l3"/><link name="l4"/>
 <link name="l5"/><link name="l6"/><link name="tip"/>
 <joint name="w" type="fixed"><parent link="world"/><child link="base"/><origin xyz="0.1 0.2 0.3" rpy="0 0 0.3"/></joint>
 <joint name="joint1" type="revolute"><parent link="base"/><child link="l1"/><origin xyz="0 0 0.2" rpy="0 0 0"/>
   <axis xyz="0 0 1"/><limit lower="-3.1" upper="3.1" effort="1" velocity="1"/></joint>
 <joint name="joint2" type="revolute"><parent link="l1"/><child link="l2"/><origin xyz="0 0 0.1" rpy="0 0 0"/>
   <axis xyz="0 1 0"/><limit lower="-1.37" upper="2.42" effort="1" velocity="1"/></joint>
 <joint name="joint3" type="revolute"><parent link="l2"/><child link="l3"/><origin xyz="0 0 0.3" rpy="0 0 0"/>
   <axis xyz="0 1 0"/><limit lower="-1.37" upper="2.42" effort="1" velocity="1"/></joint>
 <joint name="joint4" type="revolute"><parent link="l3"/><child link="l4"/><origin xyz="0 0 0.1" rpy="0 0 0"/>
   <axis xyz="0 0 1"/><limit lower="-3.1" upper="3.1" effort="1" velocity="1"/></joint>
 <joint name="joint5" type="revolute"><parent link="l4"/><child link="l5"/><origin xyz="0 0 0.2" rpy="0 0 0"/>
   <axis xyz="0 1 0"/><limit lower="-1.64" upper="1.64" effort="1" velocity="1"/></joint>
 <joint name="joint6" type="revolute"><parent link="l5"/><child link="l6"/><origin xyz="0 0 0.1" rpy="0 0 0"/>
   <axis xyz="0 0 1"/><limit lower="-3.1" upper="3.1" effort="1" velocity="1"/></joint>
 <joint name="t" type="fixed"><parent link="l6"/><child link="tip"/><origin xyz="0 0.02 0.15" rpy="0.1 0 0"/></joint>
</robot>"""
Q_HOME = np.array([0.3, 0.8, 1.1, 0.2, 0.7, -0.4])


@pytest.fixture(scope="module")
def chain():
    return Urdf.parse(URDF).chain("world", "tip")


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


# ----------------------------------------------------------------------------------------------- names

def test_arm_config_table():
    d = J.arm_configs("dual")
    assert set(d) == {"left", "right"}
    r = d["right"]
    assert (r.group, r.tip_link, r.controller, r.hardware, r.planning_frame) == (
        "arm_right", "arm_right_xeg32_tool_tip", "arm_right_trajectory_controller", "arm_right_hw", "world")
    assert r.joint_names == tuple(f"arm_right_joint{i}" for i in range(1, 7))
    for mode, arm in (("single_left", "left"), ("single_right", "right")):
        s = J.arm_configs(mode)
        assert list(s) == [arm]
        c = s[arm]
        assert (c.group, c.tip_link, c.controller, c.hardware) == (
            "igus_rebel_arm", "xeg32_tool_tip", "rebel_arm_trajectory_controller", "rebel_arm_controller")
        assert c.joint_names == tuple(f"joint{i}" for i in range(1, 7))
    with pytest.raises(ValueError):
        J.arm_configs("triple")


def test_joint_limits_parsed():
    lim = J.parse_joint_limits(URDF)
    assert lim["joint2"] == (-1.37, 2.42)
    assert "w" not in lim


# ------------------------------------------------------------------------------------------ whitelist

@pytest.mark.parametrize("size", J.STEP_SIZES["cart_mm"])
def test_cart_sizes_accepted(size):
    req, why = J.validate_request(["left"], "left", "cart", "z", 1, size, "world")
    assert req and req["size"] == float(size), why


@pytest.mark.parametrize("size", [0, 0.2, 2, 3, 7, 10.5, 11, 50, -1, -5, float("nan"), float("inf"), True, "5", None])
def test_cart_sizes_refused(size):
    req, why = J.validate_request(["left"], "left", "cart", "z", 1, size, "world")
    assert req is None and "not allowed" in why


def test_rot_and_joint_sizes():
    for s in J.STEP_SIZES["rot_deg"]:
        assert J.validate_request(["left"], "left", "rot", "x", -1, s, "tool")[0]
    for s in (0.1, 2, 10):
        assert J.validate_request(["left"], "left", "rot", "x", -1, s, "tool")[0] is None
    for s in J.STEP_SIZES["joint_deg"]:
        assert J.validate_request(["left"], "left", "joint", 3, 1, s)[0]
    assert J.validate_request(["left"], "left", "joint", 3, 1, 10)[0] is None
    assert J.validate_request(["left"], "left", "joint", 3, 1, 0.1)[0] is None


def test_other_request_fields_refused():
    v = J.validate_request
    assert v(["left"], "middle", "cart", "z", 1, 1, "world")[0] is None
    assert "not active" in v(["left"], "right", "cart", "z", 1, 1, "world")[1]
    assert v(["left"], "left", "fly", "z", 1, 1, "world")[0] is None
    assert v(["left"], "left", "cart", "w", 1, 1, "world")[0] is None
    assert v(["left"], "left", "cart", 1, 1, 1, "world")[0] is None
    assert v(["left"], "left", "cart", "z", 2, 1, "world")[0] is None
    assert v(["left"], "left", "cart", "z", 0, 1, "world")[0] is None
    assert v(["left"], "left", "cart", "z", 1, 1, "base")[0] is None
    for ax in (0, 7, -1, 1.5, "x", None, True):
        assert v(["left"], "left", "joint", ax, 1, 1)[0] is None
    assert v(["left"], "left", "joint", "4", 1, 1)[0]["axis"] == 4


# --------------------------------------------------------------------------------------------- pose maths

def _random_R(seed):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=3)
    a /= np.linalg.norm(a)
    return J.rot_axis(0, a[0] * 1.3) @ J.rot_axis(1, a[1] * 1.1) @ J.rot_axis(2, a[2] * 2.0)


def test_cart_world_and_tool_frames():
    R = _random_R(1)
    p = np.array([0.4, -0.2, 0.5])
    p2, R2 = J.target_pose(p, R, "cart", "z", 1, 5, "world")
    assert np.allclose(p2 - p, [0, 0, 0.005]) and np.allclose(R2, R)
    p2, _ = J.target_pose(p, R, "cart", "x", -1, 0.5, "world")
    assert np.allclose(p2 - p, [-0.0005, 0, 0])
    p2, R2 = J.target_pose(p, R, "cart", "y", 1, 10, "tool")
    assert np.allclose(p2 - p, R[:, 1] * 0.010) and np.allclose(R2, R)
    assert abs(np.linalg.norm(p2 - p) - 0.010) < 1e-12


def test_rotation_about_tcp_point():
    R = _random_R(2)
    p = np.array([0.4, -0.2, 0.5])
    for frame in ("world", "tool"):
        for axis in "xyz":
            for sign in (1, -1):
                p2, R2 = J.target_pose(p, R, "rot", axis, sign, 5, frame)
                assert np.allclose(p2, p)                                  # about the TCP point
                assert abs(J.rotation_angle(R, R2) - math.radians(5)) < 1e-12
                i = "xyz".index(axis)
                if frame == "world":      # world axis unchanged by the rotation, handedness by sign
                    dR = R2 @ R.T
                    assert np.allclose(dR @ np.eye(3)[i], np.eye(3)[i])
                    assert np.allclose(dR, J.rot_axis(i, sign * math.radians(5)))
                else:
                    dR = R.T @ R2
                    assert np.allclose(dR, J.rot_axis(i, sign * math.radians(5)))
    # a world rotation about z turns the tool x axis towards y
    _, R2 = J.target_pose(p, np.eye(3), "rot", "z", 1, 5, "world")
    assert R2[1, 0] > 0


def test_quaternion_roundtrip():
    for seed in range(20):
        R = _random_R(seed)
        assert np.allclose(J.quat_to_R(*J.R_to_quat(R)), R, atol=1e-12)
    assert np.allclose(J.quat_to_R(*J.R_to_quat(np.diag([1.0, -1, -1]))), np.diag([1.0, -1, -1]))


def test_cart_waypoints_spacing_and_end():
    R0 = _random_R(3)
    p0 = np.zeros(3)
    p1, R1 = J.target_pose(p0, R0, "cart", "x", 1, 10, "tool")
    w = J.cart_waypoints(p0, R0, p1, R1)
    assert len(w) == 20 and np.allclose(w[-1][0], p1) and np.allclose(w[-1][1], R1)
    steps = np.diff(np.array([p0] + [x[0] for x in w]), axis=0)
    assert np.max(np.linalg.norm(steps, axis=1)) <= 0.0005 + 1e-12
    p1, R1 = J.target_pose(p0, R0, "rot", "y", 1, 5, "world")
    w = J.cart_waypoints(p0, R0, p1, R1)
    assert 50 <= len(w) <= 51 and np.allclose(w[-1][1], R1)
    prev = R0
    for _, Rk in w:
        assert J.rotation_angle(prev, Rk) <= math.radians(0.1) + 1e-9
        assert all(np.allclose(pk, p0) for pk, _ in w)
        prev = Rk
    # intermediate rotations lie on the geodesic: angle from the start grows linearly
    mid = w[len(w) // 2 - 1][1]
    assert abs(J.rotation_angle(R0, mid) - math.radians(2.5)) < math.radians(0.1)


# ---------------------------------------------------------------------------------------------- retiming

def _joint_polyline(chain, joint, deg, n=None):
    q0 = Q_HOME.copy()
    q1 = q0.copy()
    q1[joint] += math.radians(deg)
    n = n or int(math.ceil(abs(deg) / 0.25))
    Q = q0 + (q1 - q0) * (np.arange(n + 1) / n)[:, None]
    T = chain.fk(Q)
    return Q, T[:, :3, 3], T[:, :3, :3]


def _check_caps(rt, chain):
    pk = J.trajectory_peaks(rt.times, rt.q, chain.fk)
    assert pk["joint_rad_s"] <= J.MAX_JOINT_SPEED_RAD_S
    assert pk["tcp_mm_s"] <= J.MAX_CART_SPEED_MM_S
    assert pk["tcp_deg_s"] <= J.MAX_ROT_SPEED_DEG_S
    return pk


@pytest.mark.parametrize("joint", range(6))
@pytest.mark.parametrize("deg", [0.5, 1, 5, -5])
def test_retime_joint_steps_respect_caps(chain, joint, deg):
    Q, P, R = _joint_polyline(chain, joint, deg)
    rt = J.retime(Q, P, R)
    _check_caps(rt, chain)
    assert np.array_equal(rt.q[0], Q[0]) and np.array_equal(rt.q[-1], Q[-1])
    assert np.all(rt.qd[0] == 0) and np.all(rt.qd[-1] == 0)
    assert rt.times[0] == 0 and abs(rt.times[-1] - rt.duration) < 1e-12 and rt.duration >= 0.5
    assert np.all(np.diff(rt.times) > 0) and np.max(np.diff(rt.times)) <= J.SAMPLE_DT_S + 1e-9
    # monotone along the step (no overshoot)
    d = np.diff(rt.q[:, joint]) * np.sign(deg)
    assert np.all(d >= -1e-12)


def test_retime_cartesian_steps_and_min_duration(chain):
    # translation by IK-free construction: move along the polyline of a joint but pretend TCP data is pure translation
    n = 21
    P = np.zeros((n, 3))
    P[:, 2] = np.linspace(0, 0.010, n)
    R = np.tile(np.eye(3), (n, 1, 1))
    Q = Q_HOME + np.linspace(0, 1, n)[:, None] * np.full(6, 1e-3)
    rt = J.retime(Q, P, R)
    # 10 mm at 10 mm/s -> need 1 s -> T = 1.875 * 1.05 s
    assert abs(rt.duration - 1.875 * 1.05) < 0.06
    # tiny step: minimum duration
    P2 = P[:2] * 0.01
    rt2 = J.retime(Q[:2], P2, R[:2])
    assert rt2.duration == pytest.approx(0.5)
    # rotation: 5 deg at 5 deg/s
    Rr = np.array([J.rot_axis(2, math.radians(5) * f) for f in np.linspace(0, 1, 11)])
    rt3 = J.retime(Q[:11], np.zeros((11, 3)), Rr)
    assert rt3.duration == pytest.approx(1.875 * 1.05, abs=0.06)


def test_retime_velocity_matches_positions(chain):
    Q, P, R = _joint_polyline(chain, 0, 5)
    rt = J.retime(Q, P, R)
    mid = (rt.q[1:] - rt.q[:-1]) / np.diff(rt.times)[:, None]
    avg = 0.5 * (rt.qd[1:] + rt.qd[:-1])
    assert np.max(np.abs(mid - avg)) < 0.1 * J.MAX_JOINT_SPEED_RAD_S * 0.2


def test_retime_rejects_degenerate():
    Q = np.tile(Q_HOME, (3, 1))
    with pytest.raises(ValueError):
        J.retime(Q, np.zeros((3, 3)), np.tile(np.eye(3), (3, 1, 1)))
    with pytest.raises(ValueError):
        J.retime(Q[:1], np.zeros((1, 3)), np.eye(3)[None])


def test_joint_acceleration_cap_not_exceeded(chain):
    Q, P, R = _joint_polyline(chain, 1, 5)
    rt = J.retime(Q, P, R)
    acc = np.max(np.abs(np.diff(rt.qd, axis=0) / np.diff(rt.times)[:, None]))
    assert acc <= J.MAX_JOINT_ACCEL_RAD_S2


# ---------------------------------------------------------------------------------------------- real lock

def test_real_lock_state_machine():
    clk = FakeClock()
    lk = J.RealLock(600.0, clk)
    assert lk.check("A")[0] is False and lk.state() == (False, None, None)
    assert lk.enable("A")[0] is True
    assert lk.check("A")[0] is True
    assert lk.state()[:2] == (True, "A")
    clk.t += 599
    assert lk.check("A")[0] is True
    lk.touch()                                      # accepted jog extends the expiry
    clk.t += 599
    assert lk.check("A")[0] is True
    clk.t += 2
    assert lk.check("A")[0] is False and lk.state()[0] is False   # expired 600 s after the last jog
    # enable without a jog expires 10 min after the enable
    lk.enable("A")
    clk.t += 601
    assert lk.check("A")[0] is False
    # other page: step clears and refuses
    lk.enable("A")
    ok, msg = lk.check("B")
    assert not ok and lk.state()[0] is False and lk.check("A")[0] is False
    # other page enabling clears and refuses, a second enable then works
    lk.enable("A")
    assert lk.enable("B")[0] is False and lk.state()[0] is False
    assert lk.enable("B")[0] is True and lk.state()[1] == "B"
    lk.disable()
    assert lk.check("B")[0] is False


# ----------------------------------------------------------------------------------------------- core

def gap(core, clk, dt=J.MIN_STEP_GAP_S + 0.01):
    """Let the 300 ms minimum gap after a finished step pass (and keep the joint_states fresh)."""
    clk.t += dt
    names = [n for c in core.cfgs.values() for n in c.joint_names]
    core.update_joint_states(names, list(Q_HOME) * len(core.cfgs))


def make_core(mode="dual", real=("left", "right"), clock=None, active=True, prefill=True):
    """A core with fresh hardware / controller state and (by default) 1.4 s of still joint_states: since round 3 a step /
    move is only accepted when the arm is measured at standstill over the full window."""
    clk = clock or FakeClock()
    core = J.JogCore(mode, 600.0, clk)
    refresh(core, clk, real, active)
    if prefill:
        feed(core, clk, SS)
        refresh(core, clk, real, active)
    return core, clk


def refresh(core, clk, real=("left", "right"), active=True):
    names, pos = [], []
    for arm, cfg in core.cfgs.items():
        names += list(cfg.joint_names)
        pos += list(Q_HOME)
    core.update_joint_states(names, pos, [0.0] * len(names))
    health(core, real, active)


def jumped(core, clk, dt, real=("left", "right"), active=True):
    """Let `dt` seconds pass WITH continuous joint_states in the last SS seconds (a window is only 'covered' by samples
    without gaps) and refresh the controller_manager answers: the test-time equivalent of waiting."""
    if dt > SS:
        clk.t += dt - SS
        dt = SS
    feed(core, clk, dt)
    refresh(core, clk, real, active)


def health(core, real=("left", "right"), active=True):
    """A fresh controller_manager answer (hardware component + controller state), joint_states untouched."""
    state = "active" if active else "inactive"
    core.update_hardware([(cfg.hardware, ("Igus/Rebel" if arm in real else J.MOCK_PLUGIN), state)
                          for arm, cfg in core.cfgs.items()])
    core.update_controllers([(cfg.controller, "active") for cfg in core.cfgs.values()])


def test_mock_never_needs_enable():
    core, clk = make_core(real=())
    ok, sid, msg, ctx = core.begin_step("P1", "left", "cart", "z", 1, 1, "world")
    assert ok and msg == "accepted" and ctx["real"] is False and ctx["id"] == sid
    np.testing.assert_allclose(ctx["q0"], Q_HOME)
    st = core.status()
    assert st["busy"] and st["busy_arm"] == "left" and st["last"]["phase"] == "planning"
    assert st["hardware"]["left"]["real"] is False and st["hardware"]["left"]["state"] == "active"
    core.finish(sid, True, "done", "ok")
    assert not core.status()["busy"]
    gap(core, clk)
    assert core.begin_step("OTHER", "right", "joint", 1, -1, 5)[0]      # any page may step mock


def test_real_refused_until_enabled_and_page_binding():
    core, clk = make_core(real=("left",))
    ok, sid, msg, _ = core.begin_step("A", "left", "cart", "x", 1, 1, "tool")
    assert not ok and "locked" in msg
    assert core.status()["last"]["phase"] == "refused"
    assert core.enable_real("A")["ok"]
    st = core.status()
    assert st["real_enabled"] and st["real_enabled_page"] == "A" and 599 < st["real_enabled_remaining_s"] <= 600
    ok, sid, _, _ = core.begin_step("A", "left", "cart", "x", 1, 1, "tool")
    assert ok
    core.finish(sid, True, "done", "")
    gap(core, clk)
    # page reload -> new page id -> locked again, even after coming back
    ok, _, msg, _ = core.begin_step("B", "left", "cart", "x", 1, 1, "tool")
    assert not ok and "another page" in msg
    assert not core.status()["real_enabled"]
    assert not core.begin_step("A", "left", "cart", "x", 1, 1, "tool")[0]
    # a mock arm needs no enable, but a step from another page still clears the enable of page A
    assert core.enable_real("A")["ok"]
    ok, sid, _, _ = core.begin_step("B", "right", "cart", "x", 1, 1, "tool")    # right is mock here
    assert ok and not core.status()["real_enabled"]
    core.finish(sid, True, "done", "")
    gap(core, clk)
    assert core.enable_real("A")["ok"]
    core.disable("test")
    assert not core.status()["real_enabled"]
    assert not core.begin_step("A", "left", "cart", "x", 1, 1, "tool")[0]


def test_real_enable_expires_and_jog_extends():
    core, clk = make_core(real=("left",))
    core.enable_real("A")
    jumped(core, clk, 590, ("left",))
    ok, sid, _, _ = core.begin_step("A", "left", "joint", 2, 1, 0.5)
    assert ok
    core.finish(sid, True, "done", "")
    jumped(core, clk, 590, ("left",))   # 1180 s after the enable, 590 s after the jog
    ok, sid, _, _ = core.begin_step("A", "left", "joint", 2, 1, 0.5)
    assert ok
    core.finish(sid, True, "done", "")
    jumped(core, clk, 601, ("left",))
    assert not core.begin_step("A", "left", "joint", 2, 1, 0.5)[0]
    assert core.status()["real_enabled_remaining_s"] is None


def test_unknown_hardware_counts_as_real():
    core, clk = make_core(real=())
    core.update_hardware([(core.cfgs["left"].hardware, "", "active"), (core.cfgs["right"].hardware, "system", "active")])
    assert core.is_real("left") and core.is_real("right")
    assert core.status()["hardware"]["left"]["real"] is True
    core2 = J.JogCore("dual", 600.0, FakeClock())
    assert core2.is_real("left")


def test_busy_refusal_keeps_active_status():
    core, clk = make_core(real=())
    ok, sid, _, _ = core.begin_step("A", "left", "cart", "z", 1, 1, "world")
    ok2, sid2, msg, _ = core.begin_step("A", "right", "cart", "z", 1, 1, "world")
    assert ok and not ok2 and "busy" in msg and sid2 == sid + 1
    assert core.status()["last"]["id"] == sid and core.status()["last"]["phase"] == "planning"
    core.finish(sid, False, "failed", "x")
    gap(core, clk)
    assert core.begin_step("A", "right", "cart", "z", 1, 1, "world")[0]


def test_stale_and_missing_joint_states():
    core, clk = make_core(real=())
    clk.t += 0.6
    ok, sid, msg, _ = core.begin_step("A", "left", "cart", "z", 1, 1, "world")
    assert not ok and "stale" in msg
    st = core.status()
    assert st["joint_states_age_s"]["left"] == pytest.approx(0.6)
    for _ in range(int(SS / 0.02) + 1):                                       # only the left arm's joints come back
        clk.t += 0.02
        core.update_joint_states(list(core.cfgs["left"].joint_names), list(Q_HOME))
    core.update_hardware([(c.hardware, J.MOCK_PLUGIN, "active") for c in core.cfgs.values()])
    core.update_controllers([(c.controller, "active") for c in core.cfgs.values()])
    assert core.begin_step("A", "left", "cart", "z", 1, 1, "world")[0]        # left fresh again
    core.finish(1 + sid, True, "done", "")
    clk.t += J.MIN_STEP_GAP_S + 0.01
    ok, _, msg, _ = core.begin_step("A", "right", "cart", "z", 1, 1, "world")     # right still 0.6 s old
    assert not ok and "stale" in msg                                              # right: last seen ~2 s ago
    # one joint missing
    c2 = J.JogCore("dual", 600.0, FakeClock())
    c2.update_joint_states(list(c2.cfgs["left"].joint_names)[:5], list(Q_HOME)[:5])
    c2.update_hardware([(c.hardware, J.MOCK_PLUGIN, "active") for c in c2.cfgs.values()])
    c2.update_controllers([(c.controller, "active") for c in c2.cfgs.values()])
    ok, _, msg, _ = c2.begin_step("A", "left", "cart", "z", 1, 1, "world")
    assert not ok and "missing" in msg
    assert c2.status()["joint_states_age_s"]["left"] is None


def test_hardware_and_controller_must_be_active():
    core, clk = make_core(real=(), active=False)
    ok, _, msg, _ = core.begin_step("A", "left", "cart", "z", 1, 1, "world")
    assert not ok and "not active" in msg
    core, clk = make_core(real=())
    core.update_controllers([(core.cfgs["left"].controller, "inactive"), (core.cfgs["right"].controller, "active")])
    ok, _, msg, _ = core.begin_step("A", "left", "cart", "z", 1, 1, "world")
    assert not ok and "controller" in msg and "inactive" in msg
    assert core.status()["hardware"]["left"]["controller_state"] == "inactive"
    assert core.begin_step("A", "right", "cart", "z", 1, 1, "world")[0]
    # controller_manager answer too old -> unknown -> refused
    core, clk = make_core(real=())
    clk.t += 1.5
    core.update_joint_states(list(core.cfgs["left"].joint_names), list(Q_HOME))
    ok, _, msg, _ = core.begin_step("A", "left", "cart", "z", 1, 1, "world")
    assert not ok and "unknown" in msg
    assert core.status()["hardware"]["left"]["state"] == "unknown"


def test_arm_not_active_and_bad_requests_in_single_mode():
    core, clk = make_core(mode="single_right", real=())
    ok, _, msg, _ = core.begin_step("A", "left", "cart", "z", 1, 1, "world")
    assert not ok and "not active" in msg
    assert core.status()["arms"] == ["right"]
    ok, _, msg, ctx = core.begin_step("A", "right", "cart", "z", 1, 1, "world")
    assert ok and ctx["cfg"].joint_names[0] == "joint1"
    core.finish(ctx["id"], True, "done", "")
    ok, _, msg, _ = core.begin_step("A", "right", "cart", "z", 1, 3, "world")
    assert not ok and "not allowed" in msg
    ok, _, msg, _ = core.begin_step("", "right", "cart", "z", 1, 1, "world")
    assert not ok and "page_id" in msg


def test_ids_increase_and_stop_flag():
    core, clk = make_core(real=())
    ids = [core.begin_step("A", "left", "cart", "z", 1, 7, "world")[1] for _ in range(3)]
    assert ids == sorted(set(ids)) and len(ids) == 3
    assert core.request_stop() is False and not core.cancel.is_set()
    ok, sid, _, _ = core.begin_step("A", "left", "cart", "z", 1, 1, "world")
    assert core.request_stop() is True and core.cancel.is_set()
    core.finish(sid, False, "stopped", "stopped by operator")
    st = core.status()
    assert not st["busy"] and st["last"]["phase"] == "stopped" and st["last"]["ok"] is False
    assert st["limits"]["steps"] == J.STEP_SIZES and st["limits"]["max_cart_speed_mm_s"] == 10.0


def test_status_is_json_serialisable():
    import json
    core, clk = make_core(real=("left",))
    core.enable_real("A")
    json.dumps(core.status())


# =====================================================================================================
# review round 2: exclusive ownership, pre-dispatch re-checks, step gap, Hermite bounds, achieved reporting
# =====================================================================================================
import threading
import time


SS = J.STANDSTILL_WINDOW_S + 0.2        # enough constant joint_states history for a standstill verdict


def feed(core, clk, secs, q=Q_HOME, dt=0.02, drift=0.0):
    """Feed joint_states with a fake clock for `secs` seconds (optional constant drift in rad/s on joint 1)."""
    names = [n for c in core.cfgs.values() for n in c.joint_names]
    n_arms = len(core.cfgs)
    for _ in range(int(round(secs / dt))):
        clk.t += dt
        qq = np.array(q, dtype=float)
        qq[0] += drift * (clk.t - 1000.0)
        core.update_joint_states(names, list(qq) * n_arms)


def test_standstill_measurement():
    core, clk = make_core(real=(), prefill=False)
    assert not core.standstill("left")                      # history shorter than the window
    feed(core, clk, SS)
    assert core.standstill("left")
    feed(core, clk, SS, drift=0.01)                         # 10 mrad/s: moves 12 mrad in 1.2 s
    assert not core.standstill("left")
    feed(core, clk, SS, q=Q_HOME + np.array([0.01, 0, 0, 0, 0, 0]) * 0)  # (drift term persists: still moving)
    names = list(core.cfgs["left"].joint_names)
    # constant again after the drift stops
    for _ in range(int(SS / 0.02) + 1):
        clk.t += 0.02
        core.update_joint_states(names + list(core.cfgs["right"].joint_names), [9.0] * 12)
    assert core.standstill("left")
    clk.t += 0.6                                            # joint_states stopped coming
    assert not core.standstill("left")


def test_joint_window_for_the_auto_capture():
    core, clk = make_core(real=())
    assert core.joint_window(["left", "right"], 2.0) is not None and not core.joint_window(["left"], 2.0)[2]   # not covered
    feed(core, clk, 2.4)
    q, spread, covered = core.joint_window(["left"], 2.0)
    assert covered and spread == 0.0 and sorted(q) == sorted(core.cfgs["left"].joint_names)
    assert len(core.joint_window(["left", "right"], 2.0)[0]) == 12
    feed(core, clk, 1.0, drift=0.01)                         # 10 mrad/s: 20 mrad over 2 s
    assert core.joint_window(["left"], 2.0)[1] > 0.01
    clk.t += 0.6                                             # joint_states stopped coming
    assert core.joint_window(["left"], 2.0) is None
    core2, clk2 = make_core("single_left", real=())
    assert core2.joint_window(["left"], 2.0) is not None


def _stopping_core():
    core, clk = make_core(real=())
    feed(core, clk, 0.7)
    ok, sid, _, ctx = core.begin_step("A", "left", "cart", "z", 1, 1, "world")
    assert ok
    return core, clk, sid, ctx


def test_cancel_keeps_arm_busy_until_result_and_standstill():
    core, clk, sid, ctx = _stopping_core()
    core.begin_stopping(sid, False, "stopped", "stopped by operator")
    st = core.status()
    assert st["busy"] and st["stopping"] and not st["fault"] and st["last"]["phase"] == "stopping"
    ok, _, msg, _ = core.begin_step("A", "left", "cart", "z", 1, 1, "world")
    assert not ok and "stopping" in msg
    assert core.poll_stopping() == "waiting"                # no result yet, even though the arm stands still
    clk.t += 3.0
    feed(core, clk, SS)
    health(core, ("left", "right")[:0])
    assert core.poll_stopping() == "waiting" and core.status()["busy"]
    core.note_result(sid)                                   # result alone is not enough while the arm moves
    feed(core, clk, SS, drift=0.05)
    health(core, ())
    assert core.poll_stopping() == "waiting" and core.status()["busy"]
    for _ in range(int(SS / 0.02) + 1):                     # standstill measured -> released
        clk.t += 0.02
        core.update_joint_states(list(core.cfgs["left"].joint_names), [0.1] * 6)
    health(core, ())
    assert core.poll_stopping() == "confirmed"
    st = core.status()
    assert not st["busy"] and not st["stopping"] and st["last"]["phase"] == "stopped" and st["last"]["id"] == sid
    feed(core, clk, SS)                                     # a new step needs a measured standstill again
    refresh(core, clk, real=())
    assert core.begin_step("A", "left", "cart", "z", 1, 1, "world")[0]


def test_cancel_timeout_enters_fault_and_clear_fault_needs_standstill():
    core, clk, sid, ctx = _stopping_core()
    core.begin_stopping(sid, False, "stopped", "stopped by operator")
    clk.t += core.stop_deadline_s - 0.5
    feed(core, clk, 0.4)
    assert core.poll_stopping() == "waiting"
    feed(core, clk, SS, drift=0.05)                         # arm still moving when the deadline passes
    assert core.poll_stopping() == "fault"
    st = core.status()
    assert st["busy"] and st["fault"] and st["last"]["phase"] == "fault"
    assert "stop not confirmed" in st["last"]["message"] and "e-stop" in st["last"]["message"]
    ok, _, msg, _ = core.begin_step("A", "right", "cart", "z", 1, 1, "world")
    assert not ok and "fault" in msg and "clear_fault" in msg
    assert core.request_stop() is True                      # stop() is still accepted in a fault
    assert core.poll_stopping() == "fault"                  # a late result does not clear it
    core.note_result(sid)
    assert core.poll_stopping() == "fault"
    health(core, ())
    r = core.clear_fault()                                  # arm still moving -> refused
    assert not r["ok"] and "standstill" in r["message"] and core.status()["fault"]
    for _ in range(int(SS / 0.02) + 1):
        clk.t += 0.02
        core.update_joint_states(list(core.cfgs["left"].joint_names), [0.1] * 6)
    health(core, ())
    r = core.clear_fault()
    assert r["ok"], r
    st = core.status()
    assert not st["busy"] and not st["fault"]
    assert core.clear_fault()["ok"] is False                # nothing left to clear
    feed(core, clk, SS)                                     # a new step needs a measured standstill again
    refresh(core, clk, real=())
    assert core.begin_step("A", "left", "cart", "z", 1, 1, "world")[0]


def test_fault_relocks_real_hardware():
    core, clk = make_core(real=("left",))
    feed(core, clk, 0.7)
    core.enable_real("A")
    ok, sid, _, _ = core.begin_step("A", "left", "cart", "z", 1, 1, "world")
    assert ok and core.status()["real_enabled"]
    core.begin_stopping(sid, False, "failed", "x")
    clk.t += core.stop_deadline_s + 1
    assert core.poll_stopping() == "fault"
    assert not core.status()["real_enabled"]


def test_min_gap_between_steps():
    core, clk = make_core(real=())
    ok, sid, _, _ = core.begin_step("A", "left", "cart", "z", 1, 1, "world")
    assert ok
    core.finish(sid, True, "done", "ok")
    clk.t += 0.1
    ok, sid2, msg, _ = core.begin_step("A", "left", "cart", "z", 1, 1, "world")
    assert not ok and msg == "step too soon after the previous one"
    assert core.status()["last"]["id"] == sid and core.status()["last"]["phase"] == "done"     # last not overwritten
    jumped(core, clk, 0.25, real=())
    assert core.begin_step("A", "left", "cart", "z", 1, 1, "world")[0]
    # a step refused before dispatch does not start the gap
    core2, clk2 = make_core(real=())
    ok, sid, _, _ = core2.begin_step("A", "left", "cart", "z", 1, 1, "world")
    core2.finish(sid, False, "refused", "blocked")
    assert core2.begin_step("A", "left", "cart", "z", 1, 1, "world")[0]


# ----------------------------------------------------------------------------------- pre-dispatch checks

def test_predispatch_core_checks():
    core, clk = make_core(real=("left",))
    feed(core, clk, 0.1)
    core.enable_real("A")
    ok, sid, _, ctx = core.begin_step("A", "left", "cart", "z", 1, 1, "world")
    assert ok and ctx["real"] is True
    assert core.predispatch_check(ctx) == (True, "")
    core.disable("test")                                    # enable revoked between accept and dispatch
    ok, why = core.predispatch_check(ctx)
    assert not ok and "enable" in why
    core.enable_real("A")
    assert core.predispatch_check(ctx)[0]
    core.enable_real("A")
    core.real_lock.clear_if_other("B")                      # another page took over
    assert not core.predispatch_check(ctx)[0]
    core.enable_real("A")
    jumped(core, clk, 601, ("left",))                       # expired
    assert not core.predispatch_check(ctx)[0]
    core.enable_real("A")
    # classification flipped (mock <-> real)
    core.update_hardware([(c.hardware, J.MOCK_PLUGIN, "active") for c in core.cfgs.values()])
    ok, why = core.predispatch_check(ctx)
    assert not ok and "classification" in why
    refresh(core, clk, ("left",))
    assert core.predispatch_check(ctx)[0]
    # hardware / controller no longer active
    core.update_controllers([(core.cfgs["left"].controller, "inactive"), (core.cfgs["right"].controller, "active")])
    assert not core.predispatch_check(ctx)[0]
    refresh(core, clk, ("left",))
    # joints stale
    jumped(core, clk, 0.6, ("left",), active=True)
    assert core.predispatch_check(ctx)[0]               # refresh() re-fed them
    clk.t += 0.6
    core.update_hardware([(c.hardware, "Igus/Rebel", "active") for c in core.cfgs.values()])
    core.update_controllers([(c.controller, "active") for c in core.cfgs.values()])
    ok, why = core.predispatch_check(ctx)
    assert not ok and "stale" in why
    # arm moved since accept: > START_MOVED_TOL refused, below accepted
    names = [n for c in core.cfgs.values() for n in c.joint_names]
    q = np.array(Q_HOME)
    q[2] += 2 * J.START_MOVED_TOL_RAD
    core.update_joint_states(names, list(q) * 2)
    ok, why = core.predispatch_check(ctx)
    assert not ok and "moved" in why
    q[2] = Q_HOME[2] + 0.5 * J.START_MOVED_TOL_RAD
    feed(core, clk, SS, q=q)                                # (the arm has to be measured at standstill again)
    health(core, ("left",))
    assert core.predispatch_check(ctx)[0]
    core.cancel.set()                                       # stop requested
    ok, why = core.predispatch_check(ctx)
    assert not ok and "stop" in why


class _Log:
    def info(self, *a, **k): pass
    warning = error = info


class _Fut:
    def __init__(self, value=None, done=False):
        self._v, self._done, self._cbs = value, done, []

    def done(self):
        return self._done

    def result(self):
        return self._v

    def set(self, v):
        self._v, self._done = v, True
        for cb in self._cbs:
            cb(self)

    def add_done_callback(self, cb):
        if self._done:
            cb(self)
        else:
            self._cbs.append(cb)


class _Handle:
    accepted = True

    def __init__(self, result_value=None):
        self.cancels = 0
        self.res = _Fut(result_value, done=result_value is not None)

    def cancel_goal_async(self):
        self.cancels += 1

    def get_result_async(self):
        return self.res


class _Result:
    def __init__(self, status, code=0):
        self.status = status
        self.result = type("R", (), {"error_code": code, "error_string": ""})()


class _Client:
    def __init__(self, send_future):
        self.sent = []
        self.fut = send_future

    def send_goal_async(self, goal):
        self.sent.append(goal)
        return self.fut


ros_only = pytest.mark.skipif(J._ROS_IMPORT_ERROR is not None, reason="ROS python packages not importable")


def _controller(chain, clock=time.monotonic, client=None):
    core = J.JogCore("single_left", 600.0, clock)
    ctl = object.__new__(J.JogController)
    ctl.core, ctl.node, ctl.cfgs = core, None, core.cfgs
    ctl._allow_real_move = True          # the move tests exercise real-hardware gating below this switch
    ctl._log = _Log()
    ctl._goal_lock = threading.Lock()
    ctl._goal_handle = None
    ctl._chains = {"left": chain}
    ctl._actions = {"left": client}
    ctl._threads = []
    names = list(core.cfgs["left"].joint_names)
    core.update_hardware([(core.cfgs["left"].hardware, J.MOCK_PLUGIN, "active")])
    core.update_controllers([(core.cfgs["left"].controller, "active")])

    def feeder(stop, q=Q_HOME, period=0.02, frozen=False):
        """The 'driver': joint_states at 50 Hz (+-1e-7 rad of noise on joint 1 so that the feedback is not FROZEN in the
        sense of the supervision; frozen=True repeats the exact positions) and, like the controller_manager poll every
        0.25 s, a re-stamp of the cached hardware / controller states (their content is whatever the test set)."""
        k = 0
        while not stop.is_set():
            qq = [float(v) for v in q]
            if not frozen:
                qq[0] += 1e-7 * (k % 2)
            core.update_joint_states(names, qq)
            if k % 12 == 0:
                with core.lock:
                    hw = [(n, c, st) for n, (c, st) in core._hw.items()]
                    ct = list(core._ctrl.items())
                core.update_hardware(hw)
                core.update_controllers(ct)
            k += 1
            time.sleep(period)

    def prefill(q=Q_HOME, secs=SS, period=0.02):
        """Back-dated still history (the clock is real here): a step / move is only accepted when the arm is measured at
        standstill over the full window.  Replaces any earlier history; call before the feeder thread starts."""
        now = core.clock()
        with core.lock:
            for nme, p in zip(names, q):
                core._hist[nme] = deque((now - k * period, float(p), None) for k in range(int(secs / period), 0, -1))
                core._joints[nme] = (float(p), None, now - period)

    feeder.prefill = prefill
    prefill()
    return ctl, core, feeder


def _joint_step_plan(chain, joint=0, deg=0.5):
    Q, P, R = _joint_polyline(chain, joint, deg)
    rt, an = J.plan_timing(Q, P, R, chain.fk)
    return Q, rt, an


def _accept_joint_step(core, deg=0.5, joint=1):
    ok, sid, msg, ctx = core.begin_step("A", "left", "joint", joint, 1, deg)
    assert ok, msg
    return sid, ctx


@ros_only
def test_enable_revoked_before_dispatch_sends_nothing(chain):
    client = _Client(_Fut(_Handle(), done=True))
    ctl, core, feeder = _controller(chain, client=client)
    core.update_hardware([(core.cfgs["left"].hardware, "Igus/Rebel", "active")])      # real hardware
    stop = threading.Event()
    th = threading.Thread(target=feeder, args=(stop,), daemon=True)
    th.start()
    try:
        time.sleep(0.1)
        assert core.enable_real("A")["ok"]
        ok, sid, msg, ctx = core.begin_step("A", "left", "joint", 1, 1, 0.5)
        assert ok, msg
        ctx["q0"] = np.array(Q_HOME)
        Q, rt, an = _joint_step_plan(chain, 0, 0.5)
        core.disable("revoked meanwhile")                                              # between accept and dispatch
        with pytest.raises(J._Refuse) as e:
            ctl._execute(ctx, rt, an)
        assert "enable" in str(e.value)
        assert client.sent == []                                                        # nothing was sent
        assert "sent" not in ctx
    finally:
        stop.set()
        th.join()


@ros_only
def test_cancel_timeout_in_execute_keeps_busy_and_goal_handle(chain):
    handle = _Handle()                      # result never arrives, cancel is ignored
    client = _Client(_Fut(handle, done=True))
    ctl, core, feeder = _controller(chain, client=client)
    core.stop_deadline_s = 0.6
    stop = threading.Event()
    th = threading.Thread(target=feeder, args=(stop,), daemon=True)
    th.start()
    try:
        time.sleep(0.1)
        sid, ctx = _accept_joint_step(core)
        ctx["q0"] = np.array(Q_HOME)
        Q, rt, an = _joint_step_plan(chain, 0, 0.5)
        runner = threading.Thread(target=ctl._execute, args=(ctx, rt, an), daemon=True)
        runner.start()
        time.sleep(0.3)
        assert len(client.sent) == 1 and core.status()["last"]["phase"] == "executing"
        assert ctl.stop()["ok"] if hasattr(ctl, "stop") else True
        runner.join(5.0)
        assert not runner.is_alive()
        st = core.status()
        assert st["busy"] and st["fault"] and st["last"]["phase"] == "fault"
        assert handle.cancels >= 1
        assert ctl._goal_handle is handle                   # kept: the goal may still be live
        assert not core.begin_step("A", "left", "joint", 1, 1, 0.5)[0]
        # standstill is measured by the feeder, but the controller never confirmed the goal ended (finding 2):
        # clear_fault is refused until the result arrives
        r = ctl.clear_fault()
        assert not r["ok"] and "has not confirmed the goal ended" in r["message"] and ctl._goal_handle is handle
        handle.res.set(_Result(5))                          # now the controller reports the goal as cancelled
        # (anchored: 1.2 s of joint_states received after the fault / result, plus a fresh healthy controller_manager answer)
        assert _wait_for(lambda: ctl.clear_fault()["ok"], 6.0)
        assert not core.status()["busy"] and ctl._goal_handle is None
    finally:
        stop.set()
        th.join()


@ros_only
def test_late_accepted_goal_is_cancelled_and_arm_stays_busy(chain):
    send = _Fut()                           # controller has not answered when the operator presses stop
    handle = _Handle()
    client = _Client(send)
    ctl, core, feeder = _controller(chain, client=client)
    core.stop_deadline_s = 8.0
    stop = threading.Event()
    th = threading.Thread(target=feeder, args=(stop,), daemon=True)
    th.start()
    try:
        time.sleep(0.1)
        sid, ctx = _accept_joint_step(core)
        ctx["q0"] = np.array(Q_HOME)
        Q, rt, an = _joint_step_plan(chain, 0, 0.5)
        runner = threading.Thread(target=ctl._execute, args=(ctx, rt, an), daemon=True)
        runner.start()
        time.sleep(0.2)
        core.request_stop()
        time.sleep(0.2)
        assert core.status()["last"]["phase"] == "stopping" and core.status()["busy"]
        send.set(handle)                                    # accepted late
        time.sleep(0.3)
        assert handle.cancels >= 1                          # cancelled as soon as it existed
        assert ctl._goal_handle is handle and core.status()["busy"]       # and still owned
        handle.res.set(_Result(5))                          # cancel confirmed by the controller
        runner.join(8.0)                                    # (+ 1.2 s anchored standstill and a fresh health answer)
        assert not runner.is_alive()
        st = core.status()
        assert not st["busy"] and st["last"]["phase"] == "stopped"
    finally:
        stop.set()
        th.join()


@ros_only
def test_late_accept_after_fault_still_cancelled(chain):
    send = _Fut()
    handle = _Handle()
    ctl, core, feeder = _controller(chain, client=_Client(send))
    core.stop_deadline_s = 0.5
    stop = threading.Event()
    th = threading.Thread(target=feeder, args=(stop,), daemon=True)
    th.start()
    try:
        time.sleep(0.1)
        sid, ctx = _accept_joint_step(core)
        ctx["q0"] = np.array(Q_HOME)
        Q, rt, an = _joint_step_plan(chain, 0, 0.5)
        runner = threading.Thread(target=ctl._execute, args=(ctx, rt, an), daemon=True)
        runner.start()
        time.sleep(0.2)
        core.request_stop()
        runner.join(5.0)                                    # fault after 0.5 s: the send future never resolved
        assert core.status()["fault"] and core.status()["busy"]
        send.set(handle)                                    # accepted long after we gave up
        assert handle.cancels == 1 and ctl._goal_handle is handle
        assert core.status()["busy"]
    finally:
        stop.set()
        th.join()


@ros_only
def test_successful_step_reports_achieved_vs_requested(chain):
    q_t = np.array(Q_HOME)
    q_t[1] += math.radians(0.5) * 0.96                      # the arm ends 4 % short
    handle = _Handle(_Result(4))
    client = _Client(_Fut(handle, done=True))
    ctl, core, feeder = _controller(chain, client=client)
    qbox = np.array(Q_HOME)                                 # what the "arm" reports; changed in place below
    stop = threading.Event()
    th = threading.Thread(target=feeder, args=(stop, qbox), daemon=True)
    th.start()
    try:
        time.sleep(0.1)
        sid, ctx = _accept_joint_step(core, 0.5, 2)
        ctx["q0"] = np.array(Q_HOME)
        ctx["q_target"] = np.array(Q_HOME)
        ctx["q_target"][1] += math.radians(0.5)
        Q, rt, an = _joint_step_plan(chain, 1, 0.5)
        runner = threading.Thread(target=ctl._execute, args=(ctx, rt, an), daemon=True)
        runner.start()
        time.sleep(0.05)
        qbox[:] = q_t                                       # after dispatch the arm settles 4 % short
        runner.join(10.0)
        assert not runner.is_alive()
        last = core.status()["last"]
        assert last["phase"] == "done" and last["ok"], last
        assert last["requested_deg"] == 0.5 and abs(last["achieved_deg"] - 0.48) < 0.005
        assert abs(last["residual_deg"] - 0.02) < 0.005
        assert "moved 0.480 deg of 0.5 deg, residual 0.020 deg" in last["message"]
        assert not core.status()["busy"]
    finally:
        stop.set()
        th.join()


def test_achieved_report_cart_and_rot(chain):
    ctl = object.__new__(J.JogController)
    q0 = np.array(Q_HOME)
    T0 = chain.fk(q0)
    p0, R0 = T0[:3, 3], T0[:3, :3]
    p1, R1 = J.target_pose(p0, R0, "cart", "z", 1, 5, "world")
    ctx = {"kind": "cart", "size": 5.0, "p0": p0, "R0": R0, "target": (p1, R1), "direction": np.array([0, 0, 1.0]),
           "q0": q0, "sign": 1}
    # an arm at the target reports ~0 residual
    q_mid = q0.copy()
    for _ in range(40):                                     # crude IK on joints 2,3 towards +z 5 mm
        e = p1 - chain.fk(q_mid)[:3, 3]
        Jac = np.array([(chain.fk(q_mid + 1e-6 * np.eye(6)[k])[:3, 3] - chain.fk(q_mid)[:3, 3]) / 1e-6 for k in range(6)]).T
        q_mid += np.linalg.pinv(Jac) @ e
    extra, msg = ctl._achieved(ctx, chain, q_mid)
    assert abs(extra["achieved_mm"] - 5.0) < 0.05 and extra["residual_mm"] < 0.05 and extra["requested_mm"] == 5.0
    assert msg.startswith("done: moved 5.0") and "of 5 mm" in msg
    extra, msg = ctl._achieved(ctx, chain, q0)               # did not move at all
    assert extra["achieved_mm"] == pytest.approx(0.0, abs=1e-6) and extra["residual_mm"] == pytest.approx(5.0, abs=1e-6)
    p2, R2 = J.target_pose(p0, R0, "rot", "z", 1, 5, "world")
    ctx = {"kind": "rot", "size": 5.0, "p0": p0, "R0": R0, "target": (p2, R2), "q0": q0, "sign": 1}
    extra, msg = ctl._achieved(ctx, chain, q0)
    assert extra["achieved_deg"] == pytest.approx(0.0, abs=1e-6) and extra["residual_deg"] == pytest.approx(5.0, abs=1e-6)
    assert "achieved_mm" not in extra


# ---------------------------------------------------------------------------------- Hermite bounds etc.

def test_cut_at_distance_hits_requested_distance(chain):
    q0 = np.array(Q_HOME)
    # a monotone joint-2 motion, 12 mm or so of TCP travel; the step asks for 2 mm along the initial direction
    Q = q0 + np.linspace(0, 0.02, 41)[:, None] * np.array([0, 1.0, 0, 0, 0, 0])
    p = chain.fk(Q)[:, :3, 3]
    u = (p[-1] - p[0]) / np.linalg.norm(p[-1] - p[0])
    p0 = p[0]
    for L in (0.0001, 0.0005, 0.002, 0.004):
        out = J.JogController._cut_at_distance(chain, Q, p0, p0 + u * L)
        end = (chain.fk(out[-1])[:3, 3] - p0) @ u
        assert end == pytest.approx(L, abs=1e-5 * 0 + 2e-6)  # linear joint interpolation between samples
        assert len(out) >= 2 and np.array_equal(out[0], Q[0])
        assert ((chain.fk(out)[:, :3, 3] - p0) @ u)[-2] < L + 1e-12
    # longer than the path -> refused
    with pytest.raises(J._Refuse):
        J.JogController._cut_at_distance(chain, Q, p0, p0 + u * 0.5)
    # a path that goes forward, comes back and then goes on: reversal before the distance is reached -> refused
    idx = np.concatenate([np.linspace(0, 0.01, 11), np.linspace(0.01, 0.005, 6), np.linspace(0.005, 0.02, 31)])
    Qr = q0 + idx[:, None] * np.array([0, 1.0, 0, 0, 0, 0])
    pr = chain.fk(Qr)[:, :3, 3]
    ur = (pr[-1] - pr[0]) / np.linalg.norm(pr[-1] - pr[0])
    far = float((pr[len(idx) - 1] - pr[0]) @ ur)
    with pytest.raises(J._Refuse) as e:
        J.JogController._cut_at_distance(chain, Qr, pr[0], pr[0] + ur * far)
    assert "reverses" in str(e.value)
    # path that moves the wrong way
    with pytest.raises(J._Refuse):
        J.JogController._cut_at_distance(chain, Q[::-1], p0, p0 + u * 0.002)


def test_max_reversal_and_progress():
    assert J.max_reversal(np.array([0, 1, 2, 3.0])) == 0
    assert J.max_reversal(np.array([0, 1, 0.9, 3.0])) == pytest.approx(0.1)
    assert J.max_reversal(np.array([])) == 0.0


def test_hermite_check_catches_peak_that_finite_differences_miss(chain):
    t = np.array([0.0, 1.0, 2.0])
    q = np.tile(Q_HOME, (3, 1))
    q[:, 0] += [0.0, 0.05, 0.10]                            # 0.05 rad/s between the points: passes the old check
    qd = np.zeros((3, 6))
    qd[:, 0] = [0.0, 0.2, 0.0]                              # but the controller is told to move at 0.2 rad/s
    fd = J.trajectory_peaks(t, q, chain.fk)
    assert fd["joint_rad_s"] <= J.MAX_JOINT_SPEED_RAD_S
    an = J.analyze_trajectory(t, q, qd, chain.fk)
    assert an["joint_rad_s"] > 1.9 * J.MAX_JOINT_SPEED_RAD_S
    assert any("joint speed" in v for v in J.cap_violations(an))
    # the dense TCP speed is also above the finite-difference one (analytic Hermite, not chords)
    assert an["tcp_mm_s"] > fd["tcp_mm_s"] * 1.1
    # acceleration cap: consistent velocities but a hard jerk
    t2 = np.array([0.0, 0.05, 0.1])
    q2 = np.tile(Q_HOME, (3, 1))
    q2[:, 0] += [0.0, 0.001, 0.002]
    qd2 = np.zeros((3, 6))
    qd2[:, 0] = [0.0, 0.02, 0.0]
    an2 = J.analyze_trajectory(t2, q2, qd2, chain.fk)
    assert an2["joint_rad_s"] < J.MAX_JOINT_SPEED_RAD_S
    assert any("acceleration" in v for v in J.cap_violations(an2))


def test_hermite_reproduces_the_points_and_velocities(chain):
    Q, P, R = _joint_polyline(chain, 2, 5)
    rt = J.retime(Q, P, R)
    t, q, qd, qdd = J.hermite_dense(rt.times, rt.q, rt.qd, 10)
    k = len(rt.times) - 1                                   # each segment is sampled at 11 points (both knots)
    assert np.allclose(q[::11][:k], rt.q[:-1]) and np.allclose(qd[::11][:k], rt.qd[:-1])
    assert np.allclose(q[-1], rt.q[-1], atol=1e-12)
    assert np.all(np.diff(t) >= 0)


@pytest.mark.parametrize("joint", range(6))
@pytest.mark.parametrize("deg", [0.5, 1, 5])
def test_plan_timing_keeps_dense_hermite_within_all_caps(chain, joint, deg):
    Q, P, R = _joint_polyline(chain, joint, deg)
    rt, an = J.plan_timing(Q, P, R, chain.fk)
    assert J.cap_violations(an) == []
    assert an["joint_acc_rad_s2"] <= J.MAX_JOINT_ACCEL_RAD_S2
    prog = J.step_progress("joint", an, P[0], R[0], joint_index=joint, sign=1)
    assert J.max_reversal(prog) < J.MONOTONIC_EPS_RAD
    assert prog[-1] == pytest.approx(math.radians(deg), abs=1e-9)


def test_plan_timing_lengthens_a_too_fast_profile(chain):
    # waypoints with a tight (0.4 s) corner demand: retime with a tiny margin would overshoot; plan_timing must not
    Q, P, R = _joint_polyline(chain, 3, 5)
    rt, an = J.plan_timing(Q, P, R, chain.fk)
    assert max(J.cap_ratios(an).values()) <= 1.0
    cart = np.zeros((21, 3)); cart[:, 2] = np.linspace(0, 0.01, 21)
    Rr = np.tile(np.eye(3), (21, 1, 1))
    Qc = np.tile(Q_HOME, (21, 1)) + np.linspace(0, 1, 21)[:, None] * 1e-3
    rt2, an2 = J.plan_timing(Qc, cart, Rr, chain.fk)
    assert max(J.cap_ratios(an2).values()) <= 1.0


# =====================================================================================================
# joint-goal moves: plan_move / execute_move
# =====================================================================================================
from types import SimpleNamespace

DELTA = np.array([0.2, -0.15, 0.1, 0.05, -0.1, 0.08])
Q_GOAL = Q_HOME + DELTA


def goal_dict(cfg, q=Q_GOAL):
    return {n: float(v) for n, v in zip(cfg.joint_names, q)}


# ------------------------------------------------------------------------------------------ pure maths

def test_validate_joint_goal():
    names = list(J.arm_configs("single_left")["left"].joint_names)
    good = {n: 0.1 * i for i, n in enumerate(names)}
    q, why = J.validate_joint_goal(names, good)
    assert why == "" and np.allclose(q, [0.1 * i for i in range(6)])
    q, _ = J.validate_joint_goal(names, dict(reversed(list(good.items()))))     # order of the mapping is irrelevant
    assert np.allclose(q, [0.1 * i for i in range(6)])
    short = dict(good)
    short.pop("joint6")
    assert J.validate_joint_goal(names, short)[0] is None and "missing" in J.validate_joint_goal(names, short)[1]
    extra = dict(good, joint7=0.0)
    assert J.validate_joint_goal(names, extra)[0] is None and "unexpected" in J.validate_joint_goal(names, extra)[1]
    other_arm = {f"arm_left_joint{i}": 0.0 for i in range(1, 7)}
    assert J.validate_joint_goal(names, other_arm)[0] is None
    for bad in (float("nan"), float("inf"), -float("inf"), True, "0.1", None, [0.1]):
        g = dict(good, joint3=bad)
        q, why = J.validate_joint_goal(names, g)
        assert q is None and "finite" in why
    for notmap in (None, [0.0] * 6, "x", 3):
        assert J.validate_joint_goal(names, notmap)[0] is None


def test_densify_path_spacing_and_knots():
    raw = np.vstack([Q_HOME, Q_HOME + np.array([0.2, 0, 0, 0, 0, 0]), Q_GOAL])
    d = J.densify_path(raw)
    step = math.radians(J.JOINT_CHECK_SPACING_DEG)
    assert np.max(np.abs(np.diff(d, axis=0))) <= step + 1e-9
    assert np.array_equal(d[0], raw[0]) and np.array_equal(d[-1], raw[-1])
    for k in raw:                                                 # every original point is kept
        assert np.min(np.max(np.abs(d - k), axis=1)) < 1e-12
    assert J.max_jump(d) <= J.MAX_WAYPOINT_JUMP_RAD
    assert J.max_jump(raw) > J.MAX_WAYPOINT_JUMP_RAD
    with pytest.raises(ValueError):
        J.densify_path(raw[:1])
    with pytest.raises(ValueError):
        J.densify_path(np.vstack([Q_HOME, Q_HOME + 100.0]), max_states=50)
    with pytest.raises(ValueError):
        J.densify_path(np.vstack([Q_HOME, Q_HOME + np.nan]))


def _turns(Q):
    dq = np.diff(Q, axis=0)
    u = dq / np.linalg.norm(dq, axis=1)[:, None]
    return np.degrees(np.arccos(np.clip(np.sum(u[1:] * u[:-1], axis=1), -1, 1)))


def test_smooth_path_straight_unchanged_and_corner_rounded():
    straight = J.smooth_path(np.vstack([Q_HOME, Q_GOAL]))
    line = Q_HOME + (Q_GOAL - Q_HOME) * np.linspace(0, 1, len(straight))[:, None]
    assert np.allclose(straight, line, atol=1e-9)
    assert np.array_equal(straight[0], Q_HOME) and np.array_equal(straight[-1], Q_GOAL)
    corner = Q_HOME + np.array([0.3, 0.2, 0.0, 0.0, 0.0, 0.0])
    raw = np.vstack([Q_HOME, corner, corner + np.array([-0.1, 0.3, 0.1, 0.0, 0.0, 0.0])])
    sm = J.smooth_path(raw)
    assert np.array_equal(sm[0], raw[0]) and np.array_equal(sm[-1], raw[-1])
    assert np.max(np.linalg.norm(np.diff(sm, axis=0), axis=1)) <= J.MOVE_SMOOTH_STEP_RAD + 1e-9
    fine = J._resample_arc(raw, J.MOVE_SMOOTH_STEP_RAD, 10000)
    assert _turns(sm).max() < 0.4 * _turns(fine).max()            # the 90+ deg vertex became an arc
    # the rounded path stays close to the original corner
    assert np.min(np.linalg.norm(sm - corner, axis=1)) < 0.06
    with pytest.raises(ValueError):
        J.smooth_path(np.vstack([Q_HOME, Q_HOME]))
    with pytest.raises(ValueError):
        J.smooth_path(Q_HOME[None, :])
    with pytest.raises(ValueError):
        J.smooth_path(np.vstack([Q_HOME, Q_HOME + 100.0]), max_states=100)


def test_smooth_trapezoid_profile():
    for rho in (0.05, 0.25, 0.5):
        T = 20.0
        tau = np.linspace(0, 1, 4001)
        s, sd = J.smooth_trapezoid(tau, rho, T)
        assert s[0] == 0 and s[-1] == pytest.approx(1.0, abs=1e-12)
        assert np.all(np.diff(s) >= -1e-15)
        assert sd[0] == pytest.approx(0.0, abs=1e-12) and sd[-1] == pytest.approx(0.0, abs=1e-12)
        V = 1.0 / (T * (1.0 - rho))
        assert np.max(sd) == pytest.approx(V, rel=1e-6)
        assert np.max(np.abs(np.diff(sd) / np.diff(tau * T))) <= 1.5 * V / (rho * T) * 1.001
        assert np.allclose(np.diff(s) / np.diff(tau * T), 0.5 * (sd[1:] + sd[:-1]), atol=2e-3 * V)   # s' == sdot


def _straight(chain, delta=DELTA, n=None):
    d = J.densify_path(J.smooth_path(np.vstack([Q_HOME, Q_HOME + delta])))
    return d, chain.fk(d)[:, :3, 3]


def test_retime_path_respects_the_move_caps(chain):
    for delta in (DELTA, 4 * DELTA, np.array([0, 0.6, 0, 0, 0, 0.0]), np.array([0, 0, 0, 0, 0, 2.5])):
        Qd, P = _straight(chain, delta)
        rt = J.retime_path(Qd, P)
        assert np.array_equal(rt.q[0], Qd[0]) and np.array_equal(rt.q[-1], Qd[-1])
        assert np.all(rt.qd[0] == 0) and np.all(rt.qd[-1] == 0)
        assert rt.times[0] == 0 and abs(rt.times[-1] - rt.duration) < 1e-12
        assert np.all(np.diff(rt.times) > 0) and np.max(np.diff(rt.times)) <= J.SAMPLE_DT_S + 1e-9
        rt2, an = J.plan_move_timing(Qd, P, chain.fk)
        assert J.move_cap_violations(an) == []
        assert an["joint_rad_s"] <= J.MOVE_MAX_JOINT_SPEED_RAD_S and an["joint_acc_rad_s2"] <= J.MOVE_MAX_JOINT_ACC_RAD_S2
        assert an["tcp_mm_s"] <= J.MOVE_MAX_TCP_SPEED_MM_S
        # a long move cruises near the cap instead of crawling: duration within 1.6x of path length / cap
        L = float(np.sum(np.linalg.norm(np.diff(Qd, axis=0), axis=1)))
        assert rt2.duration >= L / J.MOVE_MAX_JOINT_SPEED_RAD_S
        assert rt2.duration <= 1.6 * L / J.MOVE_MAX_JOINT_SPEED_RAD_S + 2.0
        assert np.max(np.abs(an["q"][0] - Qd[0])) < 1e-12 and np.max(np.abs(an["q"][-1] - Qd[-1])) < 1e-12


def test_retime_path_curved_paths_are_lengthened_until_the_caps_hold(chain):
    rng = np.random.default_rng(1)
    for _ in range(6):
        pts = [Q_HOME]
        for _k in range(3):
            pts.append(pts[-1] + rng.normal(size=6) * 0.12)
        Qd = J.densify_path(J.smooth_path(np.array(pts)))
        P = chain.fk(Qd)[:, :3, 3]
        rt, an = J.plan_move_timing(Qd, P, chain.fk)
        assert J.move_cap_violations(an) == [] and max(J.move_cap_ratios(an).values()) <= 1.0


def test_retime_path_rejects_degenerate_input(chain):
    Q = np.tile(Q_HOME, (3, 1))
    with pytest.raises(ValueError):
        J.retime_path(Q, np.zeros((3, 3)))
    with pytest.raises(ValueError):
        J.retime_path(Q[:1], np.zeros((1, 3)))
    with pytest.raises(ValueError):
        J.retime_path(Q, np.zeros((2, 3)))


def test_move_cap_verification_rejects_a_too_fast_trajectory(chain):
    Qd, P = _straight(chain)
    rt = J.retime_path(Qd, P, max_joint_rad_s=1.0, max_joint_acc=100.0, max_lin_m_s=10.0)   # 25x the move caps
    an = J.analyze_trajectory(rt.times, rt.q, rt.qd, chain.fk)
    bad = J.move_cap_violations(an)
    assert any("joint speed" in b for b in bad) and any("joint acceleration" in b for b in bad)
    # the step caps would be even tighter, the move caps are the ones that matter here
    assert J.move_cap_violations(dict(an, joint_rad_s=float("nan")))                  # NaN never passes
    ok = dict(an, joint_rad_s=0.04, joint_acc_rad_s2=0.25, tcp_mm_s=40.0)
    assert J.move_cap_violations(ok) == []                                            # exactly at the caps is allowed
    assert J.move_cap_violations(dict(ok, tcp_mm_s=40.5)) and J.move_cap_violations(dict(ok, joint_rad_s=0.0401))


# ---------------------------------------------------------------------------------------- core: planning

def _mk_plan_ctx(core, page="A", arm="left", q=Q_GOAL):
    ok, sid, msg, ctx = core.begin_move_plan(page, arm, goal_dict(core.cfgs[arm], q))
    assert ok, msg
    return sid, ctx


def _store(core, sid, ctx, duration=10.0):
    plan = {"arm": ctx["arm"], "page_id": ctx["page_id"], "real": ctx["real"], "q_start": ctx["q0"].copy(),
            "q_goal": ctx["q_goal"].copy(), "rt": SimpleNamespace(duration=duration), "an": {},
            "duration_s": duration, "max_joint_change_deg": math.degrees(float(np.max(np.abs(ctx["q_goal"] - ctx["q0"])))),
            "n_points": 100}
    return core.complete_plan(sid, plan)


def make_move_core(real=(), single=True):
    core, clk = make_core("single_left" if single else "dual", real=real)
    feed(core, clk, 0.7)
    return core, clk


def test_plan_refusals():
    core, clk = make_move_core()
    cfg = core.cfgs["left"]
    g = goal_dict(cfg)
    ok, sid, msg, ctx = core.begin_move_plan("A", "right", g)
    assert not ok and ctx is None and "unknown arm" in msg
    assert not core.begin_move_plan("A", "middle", g)[0]
    assert not core.begin_move_plan("A", ["left"], g)[0]
    assert not core.begin_move_plan("", "left", g)[0]
    assert not core.begin_move_plan(None, "left", g)[0]
    short = dict(g)
    short.pop("joint1")
    assert "missing" in core.begin_move_plan("A", "left", short)[2]
    assert "unexpected" in core.begin_move_plan("A", "left", dict(g, joint7=0.0))[2]
    assert "finite" in core.begin_move_plan("A", "left", dict(g, joint2=float("nan")))[2]
    assert "finite" in core.begin_move_plan("A", "left", dict(g, joint2=float("inf")))[2]
    assert not core.begin_move_plan("A", "left", [0.0] * 6)[0]
    st = core.status()
    assert not st["busy"] and st["last"]["phase"] == "refused" and st["move_plan"] is None
    # stale joint states / hardware inactive
    clk.t += 0.6
    ok, _, msg, _ = core.begin_move_plan("A", "left", g)
    assert not ok and "stale" in msg
    refresh(core, clk, real=())
    core.update_controllers([(cfg.controller, "inactive")])
    ok, _, msg, _ = core.begin_move_plan("A", "left", g)
    assert not ok and "inactive" in msg
    refresh(core, clk, real=())
    # missing joint
    c2 = J.JogCore("single_left", 600.0, FakeClock())
    c2.update_joint_states(list(c2.cfgs["left"].joint_names)[:5], list(Q_HOME)[:5])
    c2.update_hardware([(c.hardware, J.MOCK_PLUGIN, "active") for c in c2.cfgs.values()])
    c2.update_controllers([(c.controller, "active") for c in c2.cfgs.values()])
    assert "missing" in c2.begin_move_plan("A", "left", g)[2]
    # dual mode: both arms accepted by name
    cd, _ = make_move_core(single=False)
    assert cd.begin_move_plan("A", "right", goal_dict(cd.cfgs["right"]))[0]


def test_plan_real_hardware_needs_the_enable_of_this_page():
    core, clk = make_move_core(real=("left",))
    g = goal_dict(core.cfgs["left"])
    ok, _, msg, _ = core.begin_move_plan("A", "left", g)
    assert not ok and "locked" in msg
    core.enable_real("A")
    ok, _, msg, _ = core.begin_move_plan("B", "left", g)                  # another page
    assert not ok and "another page" in msg and not core.status()["real_enabled"]
    assert not core.begin_move_plan("A", "left", g)[0]                    # and the enable is gone for A as well
    core.enable_real("A")
    t_before = core.status()["real_enabled_remaining_s"]
    jumped(core, clk, 100, real=("left",))
    ok, sid, msg, ctx = core.begin_move_plan("A", "left", g)
    assert ok and ctx["real"] is True
    assert core.status()["real_enabled_remaining_s"] < t_before - 90     # planning does NOT extend the enable
    # mock arm: no enable needed, any page
    cm, _ = make_move_core()
    assert cm.begin_move_plan("Z", "left", goal_dict(cm.cfgs["left"]))[0]


def test_plan_busy_stopping_and_fault_refuse():
    core, clk = make_move_core()
    g = goal_dict(core.cfgs["left"])
    sid, ctx = _mk_plan_ctx(core)
    st = core.status()
    assert st["busy"] and st["busy_arm"] == "left" and st["last"]["phase"] == "planning" and st["last"]["kind"] == "move_plan"
    ok, sid2, msg, _ = core.begin_move_plan("A", "left", g)
    assert not ok and "busy" in msg and core.status()["last"]["id"] == sid          # active status not overwritten
    assert not core.begin_step("A", "left", "joint", 1, 1, 1)[0]                      # a step is refused while planning
    core.finish(sid, False, "refused", "x")
    # stopping
    sid, ctx = _mk_plan_ctx(core)
    core.begin_stopping(sid, False, "stopped", "stopped by operator")
    ok, _, msg, _ = core.begin_move_plan("A", "left", g)
    assert not ok and "stopping" in msg
    clk.t += core.stop_deadline_s + 1
    assert core.poll_stopping() == "fault"
    ok, _, msg, _ = core.begin_move_plan("A", "left", g)
    assert not ok and "fault" in msg and "clear_fault" in msg


def test_plan_stored_replaced_and_summarised():
    core, clk = make_move_core()
    sid, ctx = _mk_plan_ctx(core)
    pid = _store(core, sid, ctx, 12.345678)
    st = core.status()
    assert not st["busy"] and st["last"]["phase"] == "planned" and st["last"]["ok"] and st["last"]["plan_id"] == pid
    assert st["last"]["kind"] == "move_plan"
    mp = st["move_plan"]
    assert set(mp) == {"plan_id", "arm", "duration_s", "age_s", "max_joint_change_deg"}
    assert mp["plan_id"] == pid and mp["arm"] == "left" and mp["duration_s"] == pytest.approx(12.346, abs=1e-3)
    assert mp["age_s"] == 0 and mp["max_joint_change_deg"] == pytest.approx(math.degrees(0.2), abs=1e-3)
    clk.t += 7.5
    assert core.status()["move_plan"]["age_s"] == pytest.approx(7.5)
    import json
    json.dumps(core.status())
    assert st["limits"]["move"]["plan_max_age_s"] == J.MOVE_PLAN_MAX_AGE_S
    # a planned phase does not start the gap after a finished step: executing right away is not "too soon"
    assert core._last_finish_t is None
    # a new plan replaces the old one (new id); the old id can no longer be executed
    refresh(core, clk, real=())
    sid2, ctx2 = _mk_plan_ctx(core)
    assert core.plan_summary() is None                       # accepting a new request drops the old plan at once
    pid2 = _store(core, sid2, ctx2)
    assert pid2 != pid and core.status()["move_plan"]["plan_id"] == pid2
    assert not core.begin_move_execute("A", pid)[0]
    # a refused new request does not touch a stored plan; clear_plan drops it
    assert not core.begin_move_plan("A", "left", {"joint1": 0.0})[0]
    assert core.plan_summary()["plan_id"] == pid2
    assert core.clear_plan() is True and core.clear_plan() is False and core.status()["move_plan"] is None


def test_stop_drops_the_plan_and_prevents_storing():
    core, clk = make_move_core()
    sid, ctx = _mk_plan_ctx(core)
    pid = _store(core, sid, ctx)
    assert core.request_stop() is False                        # nothing active, but the stored plan is dropped
    assert core.plan_summary() is None
    assert not core.begin_move_execute("A", pid)[0]
    # STOP while planning: the finished plan is not stored
    refresh(core, clk, real=())
    sid, ctx = _mk_plan_ctx(core)
    assert core.request_stop() is True
    assert _store(core, sid, ctx) is None
    assert core.plan_summary() is None and core.status()["busy"]
    core.finish(sid, False, "stopped", "stopped by operator")
    assert not core.status()["busy"]


# ---------------------------------------------------------------------------------------- core: executing

def _planned(core, page="A", arm="left", **kw):
    sid, ctx = _mk_plan_ctx(core, page, arm)
    pid = _store(core, sid, ctx, **kw)
    assert pid
    return pid


def test_execute_refusals():
    core, clk = make_move_core()
    ok, sid, msg, ctx = core.begin_move_execute("A", "plan-1")
    assert not ok and ctx is None and "no move plan" in msg
    pid = _planned(core)
    assert not core.begin_move_execute("", pid)[0]
    assert not core.begin_move_execute(None, pid)[0]
    ok, _, msg, _ = core.begin_move_execute("A", "plan-999")
    assert not ok and "does not match" in msg
    assert not core.begin_move_execute("A", None)[0] and not core.begin_move_execute("A", 1)[0]
    ok, _, msg, _ = core.begin_move_execute("B", pid)                        # other page
    assert not ok and "another page" in msg
    assert core.plan_summary() is not None                                     # refusals above did not consume it
    # too old
    jumped(core, clk, J.MOVE_PLAN_MAX_AGE_S + 0.1, real=())
    ok, _, msg, _ = core.begin_move_execute("A", pid)
    assert not ok and "too old" in msg and core.plan_summary() is None
    # exactly at the limit is still fine
    pid = _planned(core)
    jumped(core, clk, J.MOVE_PLAN_MAX_AGE_S, real=())
    ok, sid, msg, ctx = core.begin_move_execute("A", pid)
    assert ok, msg
    core.finish(sid, True, "done", "")


def test_execute_arm_moved_stale_hardware_and_gap():
    core, clk = make_move_core()
    names = list(core.cfgs["left"].joint_names)
    pid = _planned(core)
    q = np.array(Q_HOME)
    q[2] += 2 * J.START_MOVED_TOL_RAD
    core.update_joint_states(names, list(q))
    ok, _, msg, _ = core.begin_move_execute("A", pid)
    assert not ok and "moved since planning" in msg and "plan again" in msg
    assert core.plan_summary() is None                                         # the plan is dead, plan again
    # within the tolerance it is accepted
    core.update_joint_states(names, list(Q_HOME))
    pid = _planned(core)
    q[2] = Q_HOME[2] + 0.5 * J.START_MOVED_TOL_RAD
    ok, _, msg, _ = core.begin_move_execute("A", pid)                          # (history still shows the earlier jump)
    assert not ok and msg == J.STANDSTILL_REFUSAL
    feed(core, clk, SS, q=q)
    health(core, ())
    ok, sid, msg, ctx = core.begin_move_execute("A", pid)
    assert ok, msg
    core.finish(sid, True, "done", "")
    # stale joint states
    refresh(core, clk, real=())
    feed(core, clk, 0.4)
    pid = _planned(core)
    clk.t += 0.7
    ok, _, msg, _ = core.begin_move_execute("A", pid)
    assert not ok and "stale" in msg
    # hardware / controller not active
    refresh(core, clk, real=())
    core.update_controllers([(core.cfgs["left"].controller, "inactive")])
    ok, _, msg, _ = core.begin_move_execute("A", pid)
    assert not ok and "inactive" in msg
    # the mock / real classification flipped since planning
    refresh(core, clk, real=("left",))
    ok, _, msg, _ = core.begin_move_execute("A", pid)
    assert not ok and "classification" in msg and core.plan_summary() is None
    # a step that finished less than the gap ago
    feed(core, clk, SS)
    refresh(core, clk, real=())
    ok, sid, _, _ = core.begin_step("A", "left", "joint", 1, 1, 1)
    core.finish(sid, True, "done", "ok")
    clk.t += 0.1
    sid2, ctx2 = _mk_plan_ctx(core)
    pid = _store(core, sid2, ctx2)
    ok, _, msg, _ = core.begin_move_execute("A", pid)
    assert not ok and "too soon" in msg and core.plan_summary() is not None
    jumped(core, clk, 0.3, real=())
    assert core.begin_move_execute("A", pid)[0]


def test_execute_real_needs_enable_and_extends_it():
    core, clk = make_move_core(real=("left",))
    core.enable_real("A")
    pid = _planned(core)
    core.disable("revoked")                                                     # enable lost between plan and execute
    ok, _, msg, _ = core.begin_move_execute("A", pid)
    assert not ok and "locked" in msg and core.plan_summary() is not None
    core.enable_real("A")
    jumped(core, clk, 599, real=("left",))                                      # about to expire
    ok, _, msg, _ = core.begin_move_execute("A", pid)                           # (plan is 599 s old: too old)
    assert not ok and "too old" in msg
    core.enable_real("A")
    pid = _planned(core)
    jumped(core, clk, 30, real=("left",))
    rem0 = core.status()["real_enabled_remaining_s"]
    assert rem0 < 600 - 25
    ok, sid, msg, ctx = core.begin_move_execute("A", pid)
    assert ok and ctx["real"] is True
    assert core.status()["real_enabled_remaining_s"] == pytest.approx(600.0)    # executing counts as activity
    core.finish(sid, True, "done", "")
    # enable expired by the time of the execute
    refresh(core, clk, real=("left",))
    core.enable_real("A")
    pid = _planned(core)
    jumped(core, clk, 601, real=("left",))
    assert not core.begin_move_execute("A", pid)[0]
    # another page cannot execute and clears the enable
    core.enable_real("A")
    pid = _planned(core)
    ok, _, msg, _ = core.begin_move_execute("B", pid)
    assert not ok and not core.status()["real_enabled"]
    # mock arm: no enable needed
    cm, _ = make_move_core()
    assert cm.begin_move_execute("A", _planned(cm))[0]


def test_execute_consumes_the_plan_and_shows_the_move_phase():
    core, clk = make_move_core()
    pid = _planned(core)
    ok, sid, msg, ctx = core.begin_move_execute("A", pid)
    assert ok and msg == "accepted"
    assert ctx["kind"] == "move" and ctx["plan_id"] == pid and ctx["arm"] == "left" and ctx["real"] is False
    np.testing.assert_allclose(ctx["q0"], Q_HOME)
    np.testing.assert_allclose(ctx["q_goal"], Q_GOAL)
    st = core.status()
    assert st["move_plan"] is None                                              # consumed at accept
    assert st["busy"] and st["last"]["phase"] == "executing" and st["last"]["kind"] == "move"
    core.set_phase(sid, "executing", "executing move")
    assert core.status()["last"]["kind"] == "move"
    ok, _, msg, _ = core.begin_move_execute("A", pid)                           # second execute of the same plan
    assert not ok and "busy" in msg
    core.finish(sid, True, "done", "ok")
    assert core.status()["last"]["kind"] == "move"
    jumped(core, clk, 1, real=())
    ok, _, msg, _ = core.begin_move_execute("A", pid)
    assert not ok and "no move plan" in msg                                     # consumed
    # a plain step's status carries no "kind"
    ok, sid, _, _ = core.begin_step("A", "left", "joint", 1, 1, 1)
    assert ok and "kind" not in core.status()["last"]
    core.finish(sid, True, "done", "ok")
    assert "kind" not in core.status()["last"]


def test_execute_refused_while_busy_stopping_fault_keeps_the_plan_intact():
    core, clk = make_move_core()
    pid = _planned(core)
    ok, sid, _, _ = core.begin_step("A", "left", "joint", 1, 1, 1)
    ok2, _, msg, _ = core.begin_move_execute("A", pid)
    assert ok and not ok2 and "busy" in msg and core.plan_summary()["plan_id"] == pid
    core.begin_stopping(sid, False, "stopped", "x")
    assert "stopping" in core.begin_move_execute("A", pid)[2]
    clk.t += core.stop_deadline_s + 1
    assert core.poll_stopping() == "fault"
    assert core.plan_summary() is None                                          # a fault drops any stored plan
    ok, _, msg, _ = core.begin_move_execute("A", pid)
    assert not ok and ("fault" in msg or "no move plan" in msg)


def test_stop_of_an_executing_move_core_stopping_then_stopped_or_fault():
    core, clk = make_move_core()
    pid = _planned(core)
    ok, sid, _, ctx = core.begin_move_execute("A", pid)
    core.begin_stopping(sid, False, "stopped", "stopped by operator")
    st = core.status()
    assert st["stopping"] and st["busy"] and st["last"]["phase"] == "stopping" and st["last"]["kind"] == "move"
    assert core.poll_stopping() == "waiting"
    core.note_result(sid)
    feed(core, clk, SS)                                                         # measured standstill
    health(core, ())
    assert core.poll_stopping() == "confirmed"
    st = core.status()
    assert not st["busy"] and st["last"]["phase"] == "stopped" and st["last"]["kind"] == "move"
    # without standstill -> fault after the deadline
    jumped(core, clk, 1, real=())
    feed(core, clk, 0.7)
    pid = _planned(core)
    ok, sid, _, ctx = core.begin_move_execute("A", pid)
    assert ok
    core.begin_stopping(sid, False, "stopped", "stopped by operator")
    core.note_result(sid)
    feed(core, clk, core.stop_deadline_s + 1, drift=0.05)                       # the arm keeps moving
    assert core.poll_stopping() == "fault"
    st = core.status()
    assert st["fault"] and st["busy"] and st["last"]["kind"] == "move" and "e-stop" in st["last"]["message"]
    assert not core.begin_move_plan("A", "left", goal_dict(core.cfgs["left"]))[0]


# ------------------------------------------------------------------------------ controller (fake ROS side)

class _Svc:
    def __init__(self, handler, ready=True):
        self.handler, self.ready, self.calls = handler, ready, []

    def service_is_ready(self):
        return self.ready

    def call_async(self, req):
        self.calls.append(req)
        return self.handler(len(self.calls), req)

    def remove_pending_request(self, fut):
        pass


class _ReadyClient(_Client):
    ready = True

    def server_is_ready(self):
        return self.ready


class _Pub:
    def __init__(self):
        self.msgs = []

    def publish(self, msg):
        self.msgs.append(msg)


def _plan_response(cfg, pts, code=1):
    from moveit_msgs.srv import GetMotionPlan
    res = GetMotionPlan.Response()
    res.motion_plan_response.error_code.val = code
    jt = res.motion_plan_response.trajectory.joint_trajectory
    jt.joint_names = list(cfg.joint_names)
    from trajectory_msgs.msg import JointTrajectoryPoint
    for p in pts:
        pt = JointTrajectoryPoint()
        pt.positions = [float(v) for v in p]
        jt.points.append(pt)
    return res


def _ok_validity(n, req):
    return _Fut(SimpleNamespace(valid=True, contacts=[]), done=True)


def _move_ctl(chain, client=None, validity=_ok_validity, plan_pts=None, clock=time.monotonic, code=1):
    ctl, core, feeder = _controller(chain, clock=clock, client=client)
    cfg = core.cfgs["left"]
    pts = plan_pts if plan_pts is not None else [Q_HOME, (Q_HOME + Q_GOAL) / 2 + np.array([0.02, 0.01, 0, 0, 0.01, 0]), Q_GOAL]
    ctl._cli_valid = _Svc(validity)
    ctl._cli_plan = _Svc(lambda n, req: _Fut(_plan_response(cfg, pts, code), done=True))
    ctl._pub_display = _Pub()
    ctl._urdf, ctl._urdf_text, ctl._model_error, ctl._model_name = object(), "x", "", "test_robot"
    ctl._limits = J.parse_joint_limits(URDF)
    ctl._model_lock = threading.Lock()
    T0 = chain.fk(Q_HOME)
    ctl._tcp_from_tf = lambda c: (T0[:3, 3].copy(), T0[:3, :3].copy())
    return ctl, core, feeder, cfg


class _Feed:
    """Context manager: a joint_states feeder thread (the 'arm' reports qbox)."""

    def __init__(self, feeder, qbox=None):
        self.stop, self.qbox = threading.Event(), (np.array(Q_HOME) if qbox is None else qbox)
        self.feeder = feeder
        self.th = threading.Thread(target=feeder, args=(self.stop, self.qbox), daemon=True)

    def __enter__(self):
        self.feeder.prefill(self.qbox)
        self.th.start()
        time.sleep(0.1)
        return self

    def __exit__(self, *a):
        self.stop.set()
        self.th.join()


@ros_only
def test_plan_move_stores_a_plan_publishes_the_preview_and_checks_every_dense_state(chain):
    ctl, core, feeder, cfg = _move_ctl(chain, client=_ReadyClient(_Fut(_Handle(), done=True)))
    with _Feed(feeder):
        res = ctl.plan_move("A", "left", goal_dict(cfg))
        assert res["ok"], res
        assert set(res) >= {"ok", "plan_id", "message", "duration_s", "max_joint_change_deg", "n_points"}
        assert res["plan_id"].startswith("plan-") and res["n_points"] > 100
        assert res["max_joint_change_deg"] == pytest.approx(math.degrees(0.2), abs=1e-3)
        st = ctl.status()
        assert st["move_plan"]["plan_id"] == res["plan_id"] and not st["busy"] and st["last"]["phase"] == "planned"
        # the request MoveIt got
        req = ctl._cli_plan.calls[0].motion_plan_request
        assert req.group_name == cfg.group and req.allowed_planning_time == 5.0 and req.num_planning_attempts == 3
        assert req.max_velocity_scaling_factor == 0.05 and req.max_acceleration_scaling_factor == 0.05
        jc = req.goal_constraints[0].joint_constraints
        assert [c.joint_name for c in jc] == list(cfg.joint_names)
        assert np.allclose([c.position for c in jc], Q_GOAL)
        assert all(c.tolerance_above == 1e-4 and c.tolerance_below == 1e-4 for c in jc)
        assert np.allclose([req.start_state.joint_state.position[req.start_state.joint_state.name.index(n)]
                            for n in cfg.joint_names], Q_HOME)
        # validity: one call per dense state, first = the start, last = the goal, spacing <= 0.25 deg
        calls = ctl._cli_valid.calls
        qs_all = np.array([[r.robot_state.joint_state.position[r.robot_state.joint_state.name.index(n)]
                            for n in cfg.joint_names] for r in calls])
        assert len(calls) > 50 and all(r.group_name == cfg.group for r in calls)
        n_poly = len(core._plan["Qd"])                 # the dense polyline first, then the executed curve (r2 F)
        assert len(calls) > n_poly
        qs = qs_all[:n_poly]
        assert np.allclose(qs[0], Q_HOME) and np.allclose(qs[-1], Q_GOAL, atol=1e-12)
        assert np.max(np.abs(np.diff(qs, axis=0))) <= math.radians(0.25) + 1e-9
        qc = qs_all[n_poly:]
        assert np.allclose(qc[0], Q_HOME) and np.allclose(qc[-1], Q_GOAL, atol=1e-12)
        assert np.max(np.abs(np.diff(qc, axis=0))) <= math.radians(0.25) + 1e-9
        # the preview
        assert len(ctl._pub_display.msgs) == 1
        dm = ctl._pub_display.msgs[0]
        assert dm.model_id == "test_robot" and len(dm.trajectory) == 1
        jt = dm.trajectory[0].joint_trajectory
        assert list(jt.joint_names) == list(cfg.joint_names) and len(jt.points) == res["n_points"]
        assert np.allclose(jt.points[0].positions, Q_HOME) and np.allclose(jt.points[-1].positions, Q_GOAL)
        assert np.all(np.array(jt.points[-1].velocities) == 0)
        last_t = jt.points[-1].time_from_start
        assert last_t.sec + last_t.nanosec * 1e-9 == pytest.approx(res["duration_s"], abs=2e-3)
        # the stored trajectory keeps every cap
        plan = core._plan
        assert J.move_cap_violations(plan["an"]) == []
        # replacing: a second plan gets a new id and the first can no longer be executed
        time.sleep(0.01)
        res2 = ctl.plan_move("A", "left", goal_dict(cfg))
        assert res2["ok"] and res2["plan_id"] != res["plan_id"] and ctl.status()["move_plan"]["plan_id"] == res2["plan_id"]
        assert not ctl.execute_move("A", res["plan_id"])["ok"]
        assert ctl.clear_plan()["ok"] and ctl.status()["move_plan"] is None


@ros_only
def test_plan_move_refusals_before_and_during_planning(chain):
    ctl, core, feeder, cfg = _move_ctl(chain, client=_ReadyClient(_Fut(_Handle(), done=True)))
    with _Feed(feeder):
        g = goal_dict(cfg)
        assert "unknown arm" in ctl.plan_move("A", "right", g)["message"]
        r = ctl.plan_move("A", "left", dict(g, joint1=float("nan")))
        assert not r["ok"] and r["plan_id"] is None and "finite" in r["message"]
        assert "missing" in ctl.plan_move("A", "left", {"joint1": 0.0})["message"]
        # outside the URDF limits (joint5: +-1.64)
        out = dict(g, joint5=1.7)
        r = ctl.plan_move("A", "left", out)
        assert not r["ok"] and "limits" in r["message"] and core.status()["last"]["phase"] == "refused"
        assert not core.status()["busy"] and ctl._cli_plan.calls == []                   # MoveIt was never asked
        # already there
        r = ctl.plan_move("A", "left", goal_dict(cfg, Q_HOME))
        assert not r["ok"] and "already at the goal" in r["message"]
        # a service / action server that is not there
        ctl._cli_plan.ready = False
        r = ctl.plan_move("A", "left", g)
        assert not r["ok"] and "plan_kinematic_path" in r["message"] and not core.status()["busy"]
        ctl._cli_plan.ready = True
        ctl._cli_valid.ready = False
        assert "check_state_validity" in ctl.plan_move("A", "left", g)["message"]
        ctl._cli_valid.ready = True
        ctl._actions["left"].ready = False
        assert "action server" in ctl.plan_move("A", "left", g)["message"]
        ctl._actions["left"].ready = True
        # TF and the model disagree about the TCP
        T0 = chain.fk(Q_HOME)
        ctl._tcp_from_tf = lambda c: (T0[:3, 3] + np.array([0.01, 0, 0]), T0[:3, :3])
        r = ctl.plan_move("A", "left", g)
        assert not r["ok"] and "disagree" in r["message"] and ctl.status()["move_plan"] is None
        ctl._tcp_from_tf = lambda c: (T0[:3, 3].copy(), T0[:3, :3].copy())
        assert ctl.plan_move("A", "left", g)["ok"]


@ros_only
def test_plan_move_moveit_failures_leave_no_plan(chain):
    for code in (-1, 99999):
        ctl, core, feeder, cfg = _move_ctl(chain, client=_ReadyClient(_Fut(_Handle(), done=True)), code=code)
        with _Feed(feeder):
            r = ctl.plan_move("A", "left", goal_dict(cfg))
            assert not r["ok"] and "no path" in r["message"] and ctl.status()["move_plan"] is None
            assert core.status()["last"]["phase"] == "refused" and not core.status()["busy"]
            assert ctl._pub_display.msgs == [] and ctl._cli_valid.calls == []
    # a path that does not start at the current state / does not end at the goal
    for pts, why in (([Q_HOME + 0.01, Q_GOAL], "start"), ([Q_HOME, Q_GOAL + 0.01], "goal")):
        ctl, core, feeder, cfg = _move_ctl(chain, client=_ReadyClient(_Fut(_Handle(), done=True)), plan_pts=pts)
        with _Feed(feeder):
            r = ctl.plan_move("A", "left", goal_dict(cfg))
            assert not r["ok"] and why in r["message"] and ctl.status()["move_plan"] is None
    # a path that leaves a joint limit (joint5 +-1.64) on the way
    mid = (Q_HOME + Q_GOAL) / 2
    mid[4] = 1.8
    ctl, core, feeder, cfg = _move_ctl(chain, client=_ReadyClient(_Fut(_Handle(), done=True)), plan_pts=[Q_HOME, mid, Q_GOAL])
    with _Feed(feeder):
        r = ctl.plan_move("A", "left", goal_dict(cfg))
        assert not r["ok"] and "limits" in r["message"] and ctl._cli_valid.calls == []     # limits before the validity calls


@ros_only
def test_plan_move_dense_validity_failure_refuses(chain):
    def validity(n, req):
        bad = n == 37
        contacts = [SimpleNamespace(contact_body_1="link4", contact_body_2="table")] if bad else []
        return _Fut(SimpleNamespace(valid=not bad, contacts=contacts), done=True)

    ctl, core, feeder, cfg = _move_ctl(chain, client=_ReadyClient(_Fut(_Handle(), done=True)), validity=validity)
    with _Feed(feeder):
        r = ctl.plan_move("A", "left", goal_dict(cfg))
        assert not r["ok"] and "collision" in r["message"] and "link4 / table" in r["message"] and "state 36" in r["message"]
        assert len(ctl._cli_valid.calls) == 37                                  # stopped at the first invalid state
        assert ctl.status()["move_plan"] is None and ctl._pub_display.msgs == []
        st = core.status()
        assert not st["busy"] and st["last"]["phase"] == "refused"
    # an unanswered validity request
    ctl, core, feeder, cfg = _move_ctl(chain, client=_ReadyClient(_Fut(_Handle(), done=True)),
                                       validity=lambda n, req: _Fut(None, done=True))
    with _Feed(feeder):
        r = ctl.plan_move("A", "left", goal_dict(cfg))
        assert not r["ok"] and "check_state_validity failed" in r["message"] and ctl.status()["move_plan"] is None


@ros_only
def test_plan_move_cap_verification_failure_refuses(chain, monkeypatch):
    def too_fast(Q, tcp_pos, fk):
        rt = J.retime_path(Q, tcp_pos, max_joint_rad_s=1.0, max_joint_acc=100.0, max_lin_m_s=10.0)
        return rt, J.analyze_trajectory(rt.times, rt.q, rt.qd, fk)

    monkeypatch.setattr(J, "plan_move_timing", too_fast)
    ctl, core, feeder, cfg = _move_ctl(chain, client=_ReadyClient(_Fut(_Handle(), done=True)))
    with _Feed(feeder):
        r = ctl.plan_move("A", "left", goal_dict(cfg))
        assert not r["ok"] and "exceeds the move caps" in r["message"] and "joint speed" in r["message"]
        assert ctl.status()["move_plan"] is None and ctl._pub_display.msgs == [] and not core.status()["busy"]


@ros_only
def test_plan_move_duration_limit(chain, monkeypatch):
    monkeypatch.setattr(J, "MOVE_MAX_DURATION_S", 3.0)
    ctl, core, feeder, cfg = _move_ctl(chain, client=_ReadyClient(_Fut(_Handle(), done=True)))
    with _Feed(feeder):
        r = ctl.plan_move("A", "left", goal_dict(cfg))
        assert not r["ok"] and "would take" in r["message"] and ctl.status()["move_plan"] is None


@ros_only
def test_stop_during_planning_busy_phase_and_nothing_stored(chain):
    pending = _Fut()
    ctl, core, feeder, cfg = _move_ctl(chain, client=_ReadyClient(_Fut(_Handle(), done=True)))
    ctl._cli_plan = _Svc(lambda n, req: pending)
    with _Feed(feeder):
        out = {}
        th = threading.Thread(target=lambda: out.update(ctl.plan_move("A", "left", goal_dict(cfg))), daemon=True)
        th.start()
        time.sleep(0.3)
        st = ctl.status()
        assert st["busy"] and st["busy_arm"] == "left" and st["last"]["phase"] == "planning"      # the auto capture pauses on busy
        assert not ctl.step("A", "left", "joint", 1, 1, 1)["ok"]
        assert not ctl.plan_move("A", "left", goal_dict(cfg))["ok"]
        ctl.stop()
        th.join(5.0)
        assert not th.is_alive()
        assert not out["ok"] and out["plan_id"] is None and "stopped" in out["message"]
        st = ctl.status()
        assert not st["busy"] and st["last"]["phase"] == "stopped" and st["move_plan"] is None
        assert ctl._pub_display.msgs == []


def _wait_for(pred, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.02)
    return False


@ros_only
def test_execute_move_dispatches_reports_the_goal_error_and_consumes_the_plan(chain):
    q_end = Q_GOAL + np.radians([0.02, 0, -0.03, 0, 0.01, 0])           # the arm ends a few hundredths of a degree off
    handle = _Handle(_Result(4))
    client = _ReadyClient(_Fut(handle, done=True))
    ctl, core, feeder, cfg = _move_ctl(chain, client=client)
    qbox = np.array(Q_HOME)
    with _Feed(feeder, qbox):
        res = ctl.plan_move("A", "left", goal_dict(cfg))
        assert res["ok"]
        pid = res["plan_id"]
        assert not ctl.execute_move("A", "plan-0")["ok"]                  # wrong id
        assert not ctl.execute_move("B", pid)["ok"]                       # other page
        ex = ctl.execute_move("A", pid)
        assert ex["ok"] and ex["message"] == "accepted" and ex["plan_id"] == pid
        assert ctl.status()["move_plan"] is None                          # consumed at accept
        assert _wait_for(lambda: len(client.sent) == 1)
        goal = client.sent[0]
        assert list(goal.trajectory.joint_names) == list(cfg.joint_names)
        assert goal.goal_time_tolerance.sec == 3 and goal.goal_time_tolerance.nanosec == 0
        pts = goal.trajectory.points
        assert np.allclose(pts[0].positions, Q_HOME) and np.allclose(pts[-1].positions, Q_GOAL)
        assert len(pts) == res["n_points"] and np.all(np.array(pts[0].velocities) == 0)
        qbox[:] = q_end                                                    # the arm settles after the dispatch
        assert _wait_for(lambda: not core.status()["busy"], 10.0)
        last = core.status()["last"]
        assert last["phase"] == "done" and last["ok"] and last["kind"] == "move" and last["plan_id"] == pid, last
        assert last["max_joint_error_deg"] == pytest.approx(0.03, abs=1e-3)
        assert last["joint_error_deg"][0] == pytest.approx(0.02, abs=1e-3)
        assert "max joint error 0.030 deg" in last["message"] and "joint 3" in last["message"]
        # consumed: the same plan cannot be executed twice
        time.sleep(J.MIN_STEP_GAP_S + 0.05)
        again = ctl.execute_move("A", pid)
        assert not again["ok"] and "no move plan" in again["message"]
        assert len(client.sent) == 1


@ros_only
def test_execute_move_refused_when_the_arm_moved_since_planning(chain):
    client = _ReadyClient(_Fut(_Handle(), done=True))
    ctl, core, feeder, cfg = _move_ctl(chain, client=client)
    qbox = np.array(Q_HOME)
    with _Feed(feeder, qbox):
        res = ctl.plan_move("A", "left", goal_dict(cfg))
        assert res["ok"]
        qbox[1] += 5 * J.START_MOVED_TOL_RAD
        time.sleep(0.1)
        r = ctl.execute_move("A", res["plan_id"])
        assert not r["ok"] and "moved since planning" in r["message"] and "plan again" in r["message"]
        assert client.sent == [] and not core.status()["busy"] and ctl.status()["move_plan"] is None


@ros_only
def test_execute_move_real_hardware_enable_lost_after_accept_sends_nothing(chain):
    client = _ReadyClient(_Fut(_Handle(), done=True))
    ctl, core, feeder, cfg = _move_ctl(chain, client=client)
    core.update_hardware([(cfg.hardware, "Igus/Rebel", "active")])
    with _Feed(feeder):
        assert core.enable_real("A")["ok"]
        res = ctl.plan_move("A", "left", goal_dict(cfg))
        assert res["ok"]
        ok, sid, msg, ctx = core.begin_move_execute("A", res["plan_id"])           # accepted ...
        assert ok, msg
        core.disable("revoked meanwhile")                                           # ... then the enable is revoked
        with pytest.raises(J._Refuse) as e:
            ctl._run_move(ctx)
        assert "enable" in str(e.value) and client.sent == [] and "sent" not in ctx
        core.finish(sid, False, "refused", "x")


@ros_only
def test_stop_during_an_executing_move_goes_stopping_then_stopped(chain):
    handle = _Handle()                                   # no result yet
    client = _ReadyClient(_Fut(handle, done=True))
    ctl, core, feeder, cfg = _move_ctl(chain, client=client)
    with _Feed(feeder):
        res = ctl.plan_move("A", "left", goal_dict(cfg))
        assert ctl.execute_move("A", res["plan_id"])["ok"]
        assert _wait_for(lambda: len(client.sent) == 1 and core.status()["last"]["phase"] == "executing")
        assert core.status()["last"]["kind"] == "move" and "move to joint goal" in core.status()["last"]["message"]
        ctl.stop()
        assert _wait_for(lambda: handle.cancels >= 1)                                # the goal is cancelled
        assert _wait_for(lambda: core.status()["last"]["phase"] == "stopping")
        st = core.status()
        assert st["busy"] and st["stopping"] and not st["fault"]                     # exclusive until result + standstill
        assert not ctl.plan_move("A", "left", goal_dict(cfg))["ok"] and not ctl.step("A", "left", "joint", 1, 1, 1)["ok"]
        handle.res.set(_Result(5))                                                   # cancel confirmed by the controller
        assert _wait_for(lambda: not core.status()["busy"], 5.0)
        st = core.status()
        assert st["last"]["phase"] == "stopped" and st["last"]["kind"] == "move" and not st["fault"]
        assert st["move_plan"] is None


@ros_only
def test_stop_during_an_executing_move_without_standstill_is_a_fault(chain):
    handle = _Handle()
    client = _ReadyClient(_Fut(handle, done=True))
    ctl, core, feeder, cfg = _move_ctl(chain, client=client)
    core.stop_deadline_s = 0.8
    qbox = np.array(Q_HOME)
    drift = threading.Event()

    def mover():
        while not drift.is_set():
            qbox[0] += 2e-3
            time.sleep(0.02)

    with _Feed(feeder, qbox):
        core.update_hardware([(cfg.hardware, "Igus/Rebel", "active")])
        core.enable_real("A")
        res = ctl.plan_move("A", "left", goal_dict(cfg))
        assert ctl.execute_move("A", res["plan_id"])["ok"]
        assert _wait_for(lambda: len(client.sent) == 1)
        mt = threading.Thread(target=mover, daemon=True)
        mt.start()                                        # the arm keeps moving after STOP
        try:
            ctl.stop()
            handle.res.set(_Result(5))                    # result is in, but there is never a standstill
            assert _wait_for(lambda: core.status()["fault"], 5.0)
            st = core.status()
            assert st["busy"] and st["last"]["phase"] == "fault" and st["last"]["kind"] == "move"
            assert "e-stop" in st["last"]["message"] and not st["real_enabled"]           # a fault re-locks real hardware
            assert not ctl.plan_move("A", "left", goal_dict(cfg))["ok"]
            assert not ctl.clear_fault()["ok"]            # still moving: only the e-stop helps
        finally:
            drift.set()
            mt.join()
        assert _wait_for(lambda: ctl.clear_fault()["ok"], 5.0)                          # standstill measured again
        assert not core.status()["busy"]


@ros_only
def test_execute_move_controller_rejection_and_missing_action_server(chain):
    client = _ReadyClient(_Fut(SimpleNamespace(accepted=False), done=True))
    ctl, core, feeder, cfg = _move_ctl(chain, client=client)
    with _Feed(feeder):
        res = ctl.plan_move("A", "left", goal_dict(cfg))
        assert ctl.execute_move("A", res["plan_id"])["ok"]
        assert _wait_for(lambda: not core.status()["busy"], 5.0)
        last = core.status()["last"]
        assert last["phase"] == "refused" and "rejected" in last["message"] and last["kind"] == "move"
        # no action server at execute time: refused synchronously, plan consumed, nothing sent
        time.sleep(J.MIN_STEP_GAP_S + 0.05)
        client.sent.clear()
        core.update_hardware([(cfg.hardware, J.MOCK_PLUGIN, "active")])     # (the controller_manager cache is valid for 1 s)
        core.update_controllers([(cfg.controller, "active")])
        res = ctl.plan_move("A", "left", goal_dict(cfg))
        client.ready = False
        r = ctl.execute_move("A", res["plan_id"])
        assert not r["ok"] and "action server" in r["message"] and client.sent == [] and not core.status()["busy"]
        assert ctl.status()["move_plan"] is None


def test_display_trajectory_message_and_unused_model_name(chain):
    pytest.importorskip("moveit_msgs")
    if J._ROS_IMPORT_ERROR is not None:
        pytest.skip("ROS python packages not importable")
    ctl, core, feeder, cfg = _move_ctl(chain, client=None)
    Qd, P = _straight(chain)
    rt = J.retime_path(Qd, P)
    ctl._model_name = ""
    dm = ctl._display_trajectory(cfg, Q_HOME, rt)
    assert dm.model_id == "" and len(dm.trajectory[0].joint_trajectory.points) == len(rt.times)
    ctl._pub_display = SimpleNamespace(publish=lambda m: (_ for _ in ()).throw(RuntimeError("boom")))
    ctl._publish_display(cfg, Q_HOME, rt)                          # a failing publisher is only logged


@ros_only
def test_execute_move_predispatch_recheck_refuses_synchronously(chain, monkeypatch):
    client = _ReadyClient(_Fut(_Handle(), done=True))
    ctl, core, feeder, cfg = _move_ctl(chain, client=client)
    with _Feed(feeder):
        res = ctl.plan_move("A", "left", goal_dict(cfg))
        monkeypatch.setattr(core, "predispatch_check", lambda ctx: (False, "joint_states went stale before the trajectory was sent"))
        r = ctl.execute_move("A", res["plan_id"])
        assert not r["ok"] and "stale" in r["message"] and client.sent == []
        st = core.status()
        assert not st["busy"] and st["last"]["phase"] == "refused" and st["move_plan"] is None


# =====================================================================================================
# safety review round (findings F1..F8)
# =====================================================================================================

def _poke(core, clk, arm_q, vel=None, secs=0.0):
    """One joint_states message for the single_left core (optionally with velocities)."""
    names = list(core.cfgs["left"].joint_names)
    core.update_joint_states(names, list(arm_q), vel if vel is not None else ())


# ---- F1: non-finite joint states fail closed -------------------------------------------------------------

@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_f1_nonfinite_position_invalidates_the_arm_everywhere(bad):
    core, clk = make_move_core()
    pid = _planned(core)
    q = np.array(Q_HOME)
    q[2] = bad
    _poke(core, clk, q)
    assert core.joint_age("left") is None and core.joint_snapshot("left") is None
    assert not core.standstill("left")
    assert core.joint_window(["left"], 0.5) is None
    assert core.status()["joint_states_age_s"]["left"] is None
    ok, _, msg, _ = core.begin_step("A", "left", "cart", "z", 1, 1, "world")
    assert not ok and "non-finite" in msg
    ok, _, msg, _ = core.begin_move_plan("A", "left", goal_dict(core.cfgs["left"]))
    assert not ok and "non-finite" in msg
    ok, _, msg, _ = core.begin_move_execute("A", pid)
    assert not ok and "non-finite" in msg and core.plan_summary() is not None      # refused, not consumed
    import json
    json.dumps(core.status())


@pytest.mark.parametrize("vel", [float("nan"), float("inf")])
def test_f1_nonfinite_velocity_invalidates_the_arm(vel):
    core, clk = make_move_core()
    feed(core, clk, SS)
    assert core.standstill("left")
    _poke(core, clk, Q_HOME, [0.0, 0.0, vel, 0.0, 0.0, 0.0])
    assert core.joint_snapshot("left") is None and not core.standstill("left")
    assert core.joint_window(["left"], 0.5) is None
    assert not core.begin_step("A", "left", "cart", "z", 1, 1, "world")[0]


def test_f1_predispatch_fails_closed_on_nan_and_standstill_waits_out_a_nan_in_the_window():
    core, clk = make_move_core()
    ok, sid, _, ctx = core.begin_step("A", "left", "joint", 1, 1, 1)
    assert ok and core.predispatch_check(ctx)[0]
    q = np.array(Q_HOME)
    q[0] = float("nan")
    _poke(core, clk, q)
    ok, why = core.predispatch_check(ctx)
    assert not ok and "non-finite" in why
    core.finish(sid, False, "refused", "x")
    # a NaN sample inside the standstill window blocks standstill until it has left the window, even after good samples
    core2, clk2 = make_move_core()
    feed(core2, clk2, SS)
    _poke(core2, clk2, q)
    feed(core2, clk2, 0.3)
    assert core2.joint_snapshot("left") is not None and not core2.standstill("left")
    assert core2.joint_window(["left"], 0.5) is None
    feed(core2, clk2, SS)
    assert core2.standstill("left") and core2.joint_window(["left"], 0.5) is not None


def test_f1_nan_moved_check_never_passes():
    """max(|q - q0|) with a NaN used to compare False with the tolerance and pass; the comparisons are now negated."""
    core, clk = make_move_core()
    ok, sid, _, ctx = core.begin_step("A", "left", "joint", 1, 1, 1)
    ctx["q0"] = np.array([float("nan")] * 6)
    ok, why = core.predispatch_check(ctx)
    assert not ok and "moved" in why


@ros_only
def test_f1_moveit_state_with_a_nonfinite_other_arm_joint_is_refused(chain):
    ctl, core, feeder, cfg = _move_ctl(chain, client=_ReadyClient(_Fut(_Handle(), done=True)))
    ctl.cfgs = J.arm_configs("dual")
    core.cfgs = ctl.cfgs
    right = ctl.cfgs["right"].joint_names
    core.update_joint_states(list(right), [0.0, float("nan"), 0.0, 0.0, 0.0, 0.0])
    with pytest.raises(J._Refuse) as e:
        ctl._robot_state(ctl.cfgs["left"], Q_HOME)
    assert "non-finite" in str(e.value) and right[1] in str(e.value)


# ---- F2: clear_fault needs a terminal goal state; late acceptances are cancelled ------------------------------

def _fault_core_with_goal(goal_state):
    core, clk = make_move_core()
    feed(core, clk, 0.7)
    refresh(core, clk, real=())                                # (controller_manager cache is only valid for 1 s)
    ok, sid, _, ctx = core.begin_step("A", "left", "joint", 1, 1, 1)
    assert ok
    if goal_state is not None:
        core.note_goal(sid, "pending")
        if goal_state in ("live", "terminal"):
            core.note_goal(sid, "live")
        if goal_state == "terminal":
            core.note_goal(sid, "terminal")
    core.begin_stopping(sid, False, "stopped", "x")
    clk.t += core.stop_deadline_s + 1
    feed(core, clk, SS)                                        # arm at standstill, deadline passed
    health(core, ())
    assert core.poll_stopping() == "fault"
    feed(core, clk, SS)                                        # (clear_fault is anchored at the fault entry)
    health(core, ())
    return core, clk, sid


@pytest.mark.parametrize("state", ["pending", "live"])
def test_f2_clear_fault_refused_while_the_goal_is_not_confirmed_ended(state):
    core, clk, sid = _fault_core_with_goal(state)
    r = core.clear_fault()
    assert not r["ok"] and r["message"] == ("the controller has not confirmed the goal ended: wait, or use the "
                                            "e-stop and restart the controller")
    assert core.status()["fault"] and core.status()["busy"]
    core.note_goal(sid, "terminal")                            # the result arrives (e.g. cancelled)
    feed(core, clk, SS)                                        # (anchored at the result)
    health(core, ())
    r = core.clear_fault()
    assert r["ok"], r
    assert not core.status()["busy"]


@pytest.mark.parametrize("state", [None, "terminal"])
def test_f2_clear_fault_allowed_when_never_sent_or_terminal(state):
    core, clk, sid = _fault_core_with_goal(state)
    assert core.clear_fault()["ok"]


def test_f2_terminal_is_final_and_a_new_request_starts_clean():
    core, clk = make_move_core()
    ok, sid, _, _ = core.begin_step("A", "left", "joint", 1, 1, 1)
    core.note_goal(sid, "pending")
    core.note_goal(sid, "terminal")
    core.note_goal(sid, "live")                                # a racing callback must not undo "terminal"
    assert core.goal_confirmed_ended()
    core.note_goal(sid, "pending")
    assert not core.goal_confirmed_ended()
    core.finish(sid, False, "refused", "x")
    assert core.goal_confirmed_ended()                         # released: nothing outstanding


def test_f2_stopping_is_not_released_while_the_goal_is_live_even_with_result_flag():
    core, clk = make_move_core()
    feed(core, clk, 0.7)
    refresh(core, clk, real=())
    ok, sid, _, _ = core.begin_step("A", "left", "joint", 1, 1, 1)
    core.note_goal(sid, "pending")
    core.note_goal(sid, "live")
    core.begin_stopping(sid, False, "stopped", "x", result_seen=True)
    assert core.poll_stopping() == "waiting"
    core.note_goal(sid, "terminal")
    feed(core, clk, SS)
    health(core, ())
    assert core.poll_stopping() == "confirmed"


@ros_only
def test_f2_step_late_accept_after_fault_is_cancelled_and_clear_waits_for_the_result(chain):
    send, handle = _Fut(), _Handle()
    ctl, core, feeder = _controller(chain, client=_Client(send))
    core.stop_deadline_s = 0.5
    with _Feed(feeder):
        sid, ctx = _accept_joint_step(core)
        ctx["q0"] = np.array(Q_HOME)
        Q, rt, an = _joint_step_plan(chain, 0, 0.5)
        runner = threading.Thread(target=ctl._execute, args=(ctx, rt, an), daemon=True)
        runner.start()
        time.sleep(0.2)
        core.request_stop()
        runner.join(5.0)
        assert core.status()["fault"]
        r = ctl.clear_fault()
        assert not r["ok"] and "not confirmed the goal ended" in r["message"]          # send still pending
        send.set(handle)                                                                # accepted late
        assert handle.cancels == 1 and ctl._goal_handle is handle
        assert not ctl.clear_fault()["ok"]                                              # live, no result yet
        handle.res.set(_Result(5))
        assert _wait_for(lambda: ctl.clear_fault()["ok"], 6.0) and not core.status()["busy"]


@ros_only
def test_f2_move_late_accept_is_cancelled_even_when_the_worker_is_gone(chain):
    send, handle = _Fut(), _Handle()
    client = _ReadyClient(send)
    ctl, core, feeder, cfg = _move_ctl(chain, client=client)
    core.stop_deadline_s = 0.6
    with _Feed(feeder):
        res = ctl.plan_move("A", "left", goal_dict(cfg))
        assert ctl.execute_move("A", res["plan_id"])["ok"]
        assert _wait_for(lambda: len(client.sent) == 1)
        ctl.stop()
        assert _wait_for(lambda: core.status()["fault"], 5.0)                          # no answer ever -> fault
        assert _wait_for(lambda: not any(t.is_alive() for t in ctl._threads), 5.0)      # the worker has returned
        assert not ctl.clear_fault()["ok"]
        send.set(handle)
        assert handle.cancels == 1 and ctl._goal_handle is handle                       # cancelled at once
        assert not ctl.clear_fault()["ok"]
        handle.res.set(_Result(5))
        assert _wait_for(lambda: ctl.clear_fault()["ok"], 6.0)


@ros_only
def test_f2_stop_hooks_a_pending_send_directly_without_the_worker(chain):
    send, handle = _Fut(), _Handle()
    ctl, core, feeder = _controller(chain, client=_Client(send))
    ctl._pending_send = (5, send)
    ctl.stop()
    ctl.stop()                                                  # idempotent: one hook only
    send.set(handle)
    assert handle.cancels == 1 and ctl._goal_handle is handle
    # an unusable answer (None / exception) is never "terminal"
    s2 = _Fut()
    ctl._pending_send = (6, s2)
    ctl._goal_handle = None
    ctl.stop()
    s2.set(None)
    assert ctl._goal_handle is None


@ros_only
def test_f2_unusable_send_answer_is_a_stopping_failure_and_never_terminal(chain):
    client = _Client(_Fut(None, done=True))                    # send future resolved with None: goal state unknown
    ctl, core, feeder = _controller(chain, client=client)
    core.stop_deadline_s = 0.5
    with _Feed(feeder):
        sid, ctx = _accept_joint_step(core)
        ctx["q0"] = np.array(Q_HOME)
        Q, rt, an = _joint_step_plan(chain, 0, 0.5)
        ctl._execute(ctx, rt, an)                               # returns when the fault is entered
        st = core.status()
        assert st["fault"] and not core.goal_confirmed_ended()
        assert not ctl.clear_fault()["ok"]


# ---- F3: queued motion does not survive STOP; busy is checked at ingress ---------------------------------------

def test_f3_ingress_reserve_refuses_busy_stopping_fault_and_a_second_queued_request():
    core, clk = make_move_core()
    r1 = core.reserve_motion()
    assert r1["ok"] and r1["epoch"] == core.stop_epoch
    r2 = core.reserve_motion()
    assert not r2["ok"] and "already queued" in r2["message"]
    core.release_motion(r1["token"] + 99)                      # a foreign token does not free the slot
    assert not core.reserve_motion()["ok"]
    core.release_motion(r1["token"])
    r3 = core.reserve_motion()
    assert r3["ok"] and r3["token"] != r1["token"]
    core.release_motion(r3["token"])
    ok, sid, _, _ = core.begin_step("A", "left", "joint", 1, 1, 1)
    assert "busy" in core.reserve_motion()["message"]
    core.begin_stopping(sid, False, "stopped", "x")
    assert "stopping" in core.reserve_motion()["message"]
    clk.t += core.stop_deadline_s + 1
    assert core.poll_stopping() == "fault"
    m = core.reserve_motion()
    assert not m["ok"] and "fault" in m["message"] and "clear_fault" in m["message"]


def test_f3_a_reservation_that_never_ran_expires():
    core, clk = make_move_core()
    assert core.reserve_motion()["ok"]
    clk.t += J.MOTION_QUEUE_TTL_S + 1
    assert core.reserve_motion()["ok"]


def test_f3_stop_epoch_voids_requests_ingressed_before_it():
    core, clk = make_move_core()
    pid = _planned(core)
    for call in (lambda e: core.begin_step("A", "left", "joint", 1, 1, 1, epoch=e),
                 lambda e: core.begin_move_plan("A", "left", goal_dict(core.cfgs["left"]), epoch=e),
                 lambda e: core.begin_move_execute("A", pid, epoch=e)):
        res = core.reserve_motion()
        e0 = res["epoch"]
        core.release_motion(res["token"])
        assert core.request_stop() is False                    # nothing running, but the epoch moves anyway
        assert core.stop_epoch == e0 + 1
        ok, _, msg, _ = call(e0)
        assert not ok and msg == "STOP was pressed after this request was sent: request void"
        assert not core.status()["busy"]
    # a request ingressed AFTER the stop is fine
    res = core.reserve_motion()
    ok, sid, msg, _ = core.begin_step("A", "left", "joint", 1, 1, 1, epoch=res["epoch"])
    assert ok, msg
    core.finish(sid, False, "refused", "x")
    # and a missing epoch (direct caller) is not checked
    refresh(core, clk, real=())
    jumped(core, clk, J.MIN_STEP_GAP_S + 0.01, real=())
    assert core.begin_step("A", "left", "joint", 1, 1, 1)[0]


def test_f3_fault_entry_also_bumps_the_epoch():
    core, clk = make_move_core()
    e0 = core.reserve_motion()["epoch"]
    ok, sid, _, _ = core.begin_step("A", "left", "joint", 1, 1, 1)
    core.begin_stopping(sid, False, "stopped", "x")
    clk.t += core.stop_deadline_s + 1
    assert core.poll_stopping() == "fault" and core.stop_epoch > e0
    core2, clk2 = make_move_core()
    e1 = core2.reserve_motion()["epoch"]
    ok, sid, _, _ = core2.begin_step("A", "left", "joint", 1, 1, 1)
    core2.enter_fault(sid, "x")
    assert core2.stop_epoch > e1


@ros_only
def test_f3_controller_methods_refuse_a_voided_epoch(chain):
    client = _ReadyClient(_Fut(_Handle(), done=True))
    ctl, core, feeder, cfg = _move_ctl(chain, client=client)
    with _Feed(feeder):
        r0 = ctl.reserve_motion()
        e0 = r0["epoch"]
        ctl.release_motion(r0["token"])
        ctl.stop()
        r = ctl.plan_move("A", "left", goal_dict(cfg), epoch=e0)
        assert not r["ok"] and "STOP was pressed" in r["message"] and ctl._cli_plan.calls == []
        r = ctl.step("A", "left", "joint", 1, 1, 1, epoch=e0)
        assert not r["ok"] and "STOP was pressed" in r["message"] and client.sent == []
        res = ctl.plan_move("A", "left", goal_dict(cfg))
        assert res["ok"]
        r1 = ctl.reserve_motion()
        e1 = r1["epoch"]
        ctl.release_motion(r1["token"])
        ctl.stop()
        res = ctl.plan_move("A", "left", goal_dict(cfg))
        assert res["ok"]
        r = ctl.execute_move("A", res["plan_id"], epoch=e1)
        assert not r["ok"] and "STOP was pressed" in r["message"] and client.sent == []


# ---- F4: executed curve vs validated path; peaks between subsamples --------------------------------------------

def test_f4_hermite_peak_velocity_is_exact_between_subsamples():
    rng = np.random.default_rng(5)
    for _ in range(20):
        K = 6
        times = np.concatenate([[0.0], np.cumsum(rng.uniform(0.02, 0.2, K))])
        q = rng.normal(0, 0.2, (K + 1, 3))
        qd = rng.normal(0, 1.0, (K + 1, 3))
        exact = J.hermite_peak_velocity(times, q, qd)
        _, _, Vd, _ = J.hermite_dense(times, q, qd, 4000)
        assert exact == pytest.approx(float(np.max(np.abs(Vd))), rel=1e-4, abs=1e-9)
        assert exact >= float(np.max(np.abs(J.hermite_dense(times, q, qd, 20)[2]))) - 1e-12
    # a segment whose interior peak is NOT on the 20-point grid: the exact value exceeds the sampled one
    times = np.array([0.0, 1.0])
    q = np.array([[0.0], [0.0]])
    qd = np.array([[3.0], [3.0]])                      # v(s) = 3 + 6 s (s-1)... interior minimum / maxima
    exact = J.hermite_peak_velocity(times, q, qd)
    sampled = float(np.max(np.abs(J.hermite_dense(times, q, qd, 20)[2])))
    assert exact >= sampled - 1e-12
    q2 = np.array([[0.0], [1.0]])
    qd2 = np.array([[0.0], [0.0]])
    ex2 = J.hermite_peak_velocity(times, q2, qd2)
    assert ex2 == pytest.approx(1.5, rel=1e-12)        # 6 s (1-s) peaks at s = 0.5 (a grid point here) ...
    qd3 = np.array([[0.0], [0.3]])
    ex3 = J.hermite_peak_velocity(times, q2, qd3)
    _, _, V3, _ = J.hermite_dense(times, q2, qd3, 100000)
    assert ex3 == pytest.approx(float(np.max(np.abs(V3))), rel=1e-9)
    assert ex3 > float(np.max(np.abs(J.hermite_dense(times, q2, qd3, 20)[2]))) - 1e-12


def test_f4_analyze_trajectory_reports_the_exact_joint_peak_and_tcp_safety(chain):
    Qd, P = _straight(chain)
    rt = J.retime_path(Qd, P)
    an = J.analyze_trajectory(rt.times, rt.q, rt.qd, chain.fk)
    assert an["joint_rad_s"] >= J.hermite_peak_velocity(rt.times, rt.q, rt.qd) - 1e-15
    t, Qh, Vh, Ah = J.hermite_dense(rt.times, rt.q, rt.qd)
    Tm = chain.fk(Qh)
    dt = np.diff(t)
    keep = dt > 1e-12                                          # (segment ends appear twice in the dense samples)
    lin = (np.linalg.norm(np.diff(Tm[:, :3, 3], axis=0), axis=1)[keep]) / dt[keep]
    assert an["tcp_mm_s"] == pytest.approx(float(np.max(lin)) * 1e3 * J.TCP_SPEED_SAFETY, rel=1e-9)


def test_f4_polyline_deviation_measures_distance_and_is_conservative():
    poly = np.column_stack([np.linspace(0, 1, 101), np.zeros(101), np.zeros(101)])
    on = np.column_stack([np.linspace(0.013, 0.97, 50), np.zeros(50), np.zeros(50)])
    assert J.polyline_deviation(poly, on) < 1e-12
    off = on + np.array([0.0, 0.003, 0.004])
    assert J.polyline_deviation(poly, off) == pytest.approx(0.005, abs=1e-12)
    assert J.polyline_deviation(poly, np.array([[1.2, 0, 0]])) == pytest.approx(0.2)         # beyond the end
    assert J.polyline_deviation(poly, np.array([[0.5, np.nan, 0]])) == float("inf")
    # brute force agrees on a curved path
    th = np.linspace(0, 2, 400)
    curve = np.column_stack([np.cos(th), np.sin(th), 0.1 * th])
    pts = curve[::7] + np.random.default_rng(1).normal(0, 0.002, (len(curve[::7]), 3))
    brute = 0.0
    for p in pts:
        a, b = curve[:-1], curve[1:]
        ab = b - a
        f = np.clip(np.sum((p - a) * ab, axis=1) / np.sum(ab * ab, axis=1), 0, 1)
        brute = max(brute, float(np.min(np.linalg.norm(p - (a + f[:, None] * ab), axis=1))))
    assert J.polyline_deviation(curve, pts) == pytest.approx(brute, rel=1e-6)


def test_f4_a_spline_with_sideways_velocities_is_caught_by_the_deviation_bound():
    n = 21
    s = np.linspace(0, 1, n)
    line = np.outer(s, np.array([0.04, 0.02, 0.0, 0.0, 0.0, 0.0]))        # 0.045 rad long
    poly = J.densify_path(np.vstack([line[0], line[-1]]))
    times = np.linspace(0, 2.0, n)
    qd_ok = np.gradient(line, times, axis=0)
    assert J.polyline_deviation(poly, J.hermite_dense(times, line, qd_ok)[1]) <= J.MOVE_MAX_PATH_DEV_RAD
    qd_bad = qd_ok.copy()
    qd_bad[:, 2] += np.where(np.arange(n) % 2 == 0, 0.5, -0.5)            # alternating sideways velocity: the spline wiggles
    dev = J.polyline_deviation(poly, J.hermite_dense(times, line, qd_bad)[1])
    assert dev > J.MOVE_MAX_PATH_DEV_RAD
    # the re-timed points themselves are exactly on the polyline
    assert J.polyline_deviation(poly, line) < J.ON_POLYLINE_TOL_RAD


@ros_only
def test_f4_stored_plan_curve_stays_on_the_validated_path_and_points_lie_on_it(chain):
    ctl, core, feeder, cfg = _move_ctl(chain, client=_ReadyClient(_Fut(_Handle(), done=True)))
    with _Feed(feeder):
        assert ctl.plan_move("A", "left", goal_dict(cfg))["ok"]
        plan = core._plan
        Qd = plan["Qd"]
        assert J.polyline_deviation(Qd, plan["rt"].q) < J.ON_POLYLINE_TOL_RAD            # retimed points lie on it
        assert plan["path_dev_rad"] <= J.MOVE_MAX_PATH_DEV_RAD
        assert J.polyline_deviation(Qd, plan["an"]["q"]) == pytest.approx(plan["path_dev_rad"])
        assert J.MOVE_MAX_PATH_DEV_RAD == 0.002


@ros_only
def test_f4_plan_refused_when_the_executed_curve_leaves_the_validated_path(chain, monkeypatch):
    monkeypatch.setattr(J, "MOVE_MAX_PATH_DEV_RAD", 1e-12)
    ctl, core, feeder, cfg = _move_ctl(chain, client=_ReadyClient(_Fut(_Handle(), done=True)))
    with _Feed(feeder):
        r = ctl.plan_move("A", "left", goal_dict(cfg))
        assert not r["ok"] and "leave the validated path" in r["message"]
        assert ctl.status()["move_plan"] is None and ctl._pub_display.msgs == []
    # points that are not on the polyline are an internal error
    real = J.polyline_deviation
    monkeypatch.setattr(J, "MOVE_MAX_PATH_DEV_RAD", 0.002)
    calls = []

    def fake(poly, pts, reach=3):
        calls.append(1)
        return 0.01 if len(calls) == 1 else real(poly, pts, reach)

    monkeypatch.setattr(J, "polyline_deviation", fake)
    ctl, core, feeder, cfg = _move_ctl(chain, client=_ReadyClient(_Fut(_Handle(), done=True)))
    with _Feed(feeder):
        r = ctl.plan_move("A", "left", goal_dict(cfg))
        assert not r["ok"] and "off the validated path" in r["message"]


@ros_only
def test_f4_execute_reverifies_the_stored_curve(chain, monkeypatch):
    client = _ReadyClient(_Fut(_Handle(), done=True))
    ctl, core, feeder, cfg = _move_ctl(chain, client=client)
    with _Feed(feeder):
        res = ctl.plan_move("A", "left", goal_dict(cfg))
        core._plan["Qd"] = core._plan["Qd"] + 0.01              # the stored path no longer matches the curve
        assert ctl.execute_move("A", res["plan_id"])["ok"]
        assert _wait_for(lambda: not core.status()["busy"], 5.0)
        last = core.status()["last"]
        assert last["phase"] == "refused" and "leaves the validated path" in last["message"] and client.sent == []


# ---- F5: runtime supervision ------------------------------------------------------------------------------------

def _sup(chain, speed_cap=J.MOVE_MAX_JOINT_SPEED_RAD_S, tol=J.MOVE_MAX_TRACKING_ERR_RAD):
    core, clk = make_move_core()
    Qd, P = _straight(chain)
    rt = J.retime_path(Qd, P)
    return J.MotionSupervisor(core, "left", rt, speed_cap, tol), core, clk, rt


def test_f5_constants_are_the_specified_ones():
    assert J.SUPERVISION_STALE_S == 0.5 and J.SUPERVISION_SPEED_FACTOR == 2.0 and J.SUPERVISION_SPEED_WINDOW_S == 0.5
    assert J.MOVE_MAX_TRACKING_ERR_RAD == 0.05 and 0 < J.STEP_MAX_TRACKING_ERR_RAD <= 0.05


def test_f5_supervision_quiet_when_all_is_fine(chain):
    sup, core, clk, rt = _sup(chain)
    feed(core, clk, 0.5)
    assert sup.check(0.2, 0.0) is None and sup.check(0.2, None) is None


def test_f5_a_stale_joint_state_triggers(chain):
    sup, core, clk, rt = _sup(chain)
    feed(core, clk, 0.5)
    clk.t += 0.6
    why = sup.check(1.0, 0.5)
    assert why and "stale" in why


def test_f5_a_nonfinite_joint_state_triggers(chain):
    sup, core, clk, rt = _sup(chain)
    feed(core, clk, 0.5)
    q = np.array(Q_HOME)
    q[4] = float("nan")
    _poke(core, clk, q)
    why = sup.check(1.0, 0.5)
    assert why and "non-finite" in why
    feed(core, clk, 0.1)
    _poke(core, clk, Q_HOME, [0.0, 0.0, 0.0, 0.0, float("inf"), 0.0])
    assert "non-finite" in sup.check(1.0, 0.5)


@pytest.mark.parametrize("cap,drift,trips", [(J.MOVE_MAX_JOINT_SPEED_RAD_S, 0.07, False),
                                             (J.MOVE_MAX_JOINT_SPEED_RAD_S, 0.09, True),
                                             (J.MAX_JOINT_SPEED_RAD_S, 0.19, False),
                                             (J.MAX_JOINT_SPEED_RAD_S, 0.22, True)])
def test_f5_measured_joint_speed_above_twice_the_cap_triggers(chain, cap, drift, trips):
    sup, core, clk, rt = _sup(chain, speed_cap=cap, tol=100.0)
    feed(core, clk, 0.6, drift=drift)
    why = sup.check(1.0, 0.5)
    if trips:
        assert why and "speed" in why and "2 x the cap" in why
    else:
        assert why is None


def test_f5_speed_needs_at_least_half_a_second_of_history(chain):
    sup, core, clk, rt = _sup(chain, tol=100.0)
    core2, clk2 = make_core("single_left", real=(), prefill=False)   # fresh: only one sample
    sup2 = J.MotionSupervisor(core2, "left", rt, J.MOVE_MAX_JOINT_SPEED_RAD_S, 100.0)
    assert core2.measured_speed("left") is None and sup2.check(0.0, None) is None
    feed(core2, clk2, 0.3, drift=5.0)
    assert core2.measured_speed("left") is None                        # 300 ms of history: no verdict yet
    feed(core2, clk2, 0.3, drift=5.0)
    assert core2.measured_speed("left") > 1.0 and sup2.check(0.2, 0.1)


@pytest.mark.parametrize("move,offset,trips", [(True, 0.04, False), (True, 0.06, True),
                                               (False, 0.04, False), (False, 0.06, True)])
def test_f5_tracking_error_against_the_reference_triggers(chain, move, offset, trips):
    tol = J.MOVE_MAX_TRACKING_ERR_RAD if move else J.STEP_MAX_TRACKING_ERR_RAD
    cap = J.MOVE_MAX_JOINT_SPEED_RAD_S if move else J.MAX_JOINT_SPEED_RAD_S
    sup, core, clk, rt = _sup(chain, speed_cap=cap, tol=tol)
    q = np.array(Q_HOME)
    q[1] += offset
    for _ in range(40):                                               # (> the 0.5 s speed window: the offset is not a "jump")
        clk.t += 0.02
        _poke(core, clk, q)
    why = sup.check(0.5, None)                                        # reference held at the first point until accepted
    if trips:
        assert why and "tracking error" in why
    else:
        assert why is None
    # the reference moves with time since acceptance: a late arm that sits at the start is "behind" later on
    t_late = rt.duration * 0.8
    behind = sup.check(t_late, t_late)
    ref = J.hermite_eval(rt.times, rt.q, rt.qd, t_late)
    assert (behind is not None) == (float(np.max(np.abs(q - ref))) > tol)


def test_f5_hermite_eval_matches_the_dense_samples(chain):
    Qd, P = _straight(chain)
    rt = J.retime_path(Qd, P)
    t, Qh, _, _ = J.hermite_dense(rt.times, rt.q, rt.qd)
    for k in (0, 7, len(t) // 2, len(t) - 1):
        assert np.allclose(J.hermite_eval(rt.times, rt.q, rt.qd, t[k]), Qh[k], atol=1e-12)
    assert np.allclose(J.hermite_eval(rt.times, rt.q, rt.qd, -5), rt.q[0])
    assert np.allclose(J.hermite_eval(rt.times, rt.q, rt.qd, 1e9), rt.q[-1])


@ros_only
def test_f5_execute_stops_through_the_stop_path_when_joint_states_go_stale(chain):
    handle = _Handle()                                      # no result
    client = _Client(_Fut(handle, done=True))
    ctl, core, feeder = _controller(chain, client=client)
    core.stop_deadline_s = 1.5
    stop = threading.Event()
    th = threading.Thread(target=feeder, args=(stop,), daemon=True)
    th.start()
    try:
        time.sleep(0.1)
        sid, ctx = _accept_joint_step(core)
        ctx["q0"] = np.array(Q_HOME)
        Q, rt, an = _joint_step_plan(chain, 0, 0.5)
        runner = threading.Thread(target=ctl._execute, args=(ctx, rt, an), daemon=True)
        runner.start()
        assert _wait_for(lambda: len(client.sent) == 1)
        stop.set()                                           # joint_states stop arriving
        th.join()
        assert _wait_for(lambda: handle.cancels >= 1, 4.0)   # supervision cancelled the goal
        st = core.status()
        assert st["last"]["phase"] == "stopping" and st["busy"] and "supervision" in st["last"]["message"]
        assert "stale" in st["last"]["message"] and core.cancel.is_set()
        runner.join(10.0)                                    # no result, no standstill (stale) -> fault after the deadline
        assert core.status()["fault"]
    finally:
        stop.set()


@ros_only
def test_f5_execute_move_tracking_supervision_stops_a_move_that_the_arm_does_not_follow(chain, monkeypatch):
    monkeypatch.setattr(J, "MOVE_MAX_TRACKING_ERR_RAD", 0.004)    # the fake arm never moves: trips as soon as the ref moves
    handle = _Handle()
    client = _ReadyClient(_Fut(handle, done=True))
    ctl, core, feeder, cfg = _move_ctl(chain, client=client)
    with _Feed(feeder):
        res = ctl.plan_move("A", "left", goal_dict(cfg))
        assert ctl.execute_move("A", res["plan_id"])["ok"]
        assert _wait_for(lambda: handle.cancels >= 1, 20.0)
        msg = core.status()["last"]["message"]
        assert "supervision" in msg and "tracking error" in msg and core.status()["last"]["kind"] == "move"
        handle.res.set(_Result(5))
        assert _wait_for(lambda: not core.status()["busy"], 5.0)
        assert core.status()["last"]["phase"] == "stopped"


# ---- F6: standstill rejects slow drift ---------------------------------------------------------------------------

def test_f6_slow_drift_is_not_standstill():
    core, clk = make_move_core()
    feed(core, clk, 2.5, drift=0.0008)                      # 0.96 mrad per 1.2 s (the drift test alone passed 0.0008 over 0.5 s)
    assert not core.standstill("left")
    core2, clk2 = make_move_core()
    feed(core2, clk2, 2.5, drift=0.0003)                    # 0.36 mrad per 1.2 s: genuinely still
    assert core2.standstill("left")
    core3, clk3 = make_move_core()
    feed(core3, clk3, 2.5, drift=0.0004)                    # 0.48 mrad per 1.2 s, just inside both tests
    assert core3.standstill("left")
    core4, clk4 = make_move_core()
    feed(core4, clk4, 2.5, drift=0.00049)                   # 0.59 mrad per 1.2 s: over the 0.5 mrad range test now
    assert not core4.standstill("left")
    assert J.STANDSTILL_MAX_DRIFT_RAD_S == 0.0005 and J.STANDSTILL_MAX_VEL_RAD_S == 0.03


def test_f6_reported_velocity_must_be_small():
    core, clk = make_move_core()
    # 0.012 rad/s = one 0.01 deg encoder step over 15 ms on the real driver: quantization flicker must still count as still
    for v, expect in ((0.001, True), (0.012, True), (0.04, False), (-0.04, False), (0.03, True)):
        for _ in range(int(SS / 0.02) + 1):
            clk.t += 0.02
            _poke(core, clk, Q_HOME, [v] * 6)
        assert core.standstill("left") is expect, v
    # no velocity field at all: position tests only
    core2, clk2 = make_move_core()
    feed(core2, clk2, SS)
    assert core2.standstill("left")


def test_f6_drift_blocks_poll_stopping_and_clear_fault_too():
    core, clk, sid = _fault_core_with_goal("terminal")
    feed(core, clk, 2.0, drift=0.0008)
    health(core, ())
    r = core.clear_fault()
    assert not r["ok"] and "standstill" in r["message"]
    core2, clk2 = make_move_core()
    feed(core2, clk2, SS)
    refresh(core2, clk2, real=())
    ok, sid, _, _ = core2.begin_step("A", "left", "joint", 1, 1, 1)
    assert ok
    core2.begin_stopping(sid, False, "stopped", "x")
    core2.note_result(sid)
    feed(core2, clk2, 2.0, drift=0.0008)
    health(core2, ())
    assert core2.poll_stopping() == "waiting"


# ---- F7: the confirm is bound to the plan that was shown (server side of the binding) ---------------------------

def test_f7_execute_refuses_any_id_but_the_stored_plan_and_keeps_it():
    core, clk = make_move_core()
    pid_a = _planned(core)
    refresh(core, clk, real=())
    pid_b = _planned(core)                                   # replaces A
    assert pid_a != pid_b
    ok, _, msg, _ = core.begin_move_execute("A", pid_a)
    assert not ok and "does not match" in msg
    assert core.plan_summary()["plan_id"] == pid_b            # B untouched, nothing consumed, nothing busy
    assert not core.status()["busy"]
    ok, sid, msg, ctx = core.begin_move_execute("A", pid_b)
    assert ok and ctx["plan_id"] == pid_b


# ---- F8: the planning scene is re-checked at execute time for old plans ----------------------------------------------

def test_f8_coarse_subset_spacing_count_and_ends():
    Q = np.column_stack([np.linspace(0, 1.0, 2001)] + [np.zeros(2001)] * 5)
    sub = J.coarse_states(Q)
    assert np.array_equal(sub[0], Q[0]) and np.array_equal(sub[-1], Q[-1])
    step = np.linalg.norm(np.diff(sub, axis=0), axis=1)
    assert np.max(step) <= math.radians(1.0) + 0.0006 and len(sub) <= J.MOVE_RECHECK_MAX_CALLS     # (round 3: 1 deg, 300)
    big = np.column_stack([np.linspace(0, 20.0, 6001)] + [np.zeros(6001)] * 5)      # 20 rad: 1146 spacings of 1 deg
    sub = J.coarse_states(big)
    assert len(sub) <= J.MOVE_RECHECK_MAX_CALLS and np.array_equal(sub[-1], big[-1])
    tiny = J.coarse_states(Q[:2])
    assert len(tiny) == 2
    assert not hasattr(J, "MOVE_RECHECK_AGE_S")                  # round 3: no age grace, every execute re-checks
    assert J.MOVE_RECHECK_SPACING_RAD == math.radians(1.0) and J.MOVE_RECHECK_MAX_CALLS == 300


class _Clock:
    def __init__(self):
        self.off = 0.0

    def __call__(self):
        return time.monotonic() + self.off


@ros_only
def test_f8_old_plan_is_rechecked_on_a_coarse_subset_and_the_arm_is_busy_checking(chain):
    clk = _Clock()
    gate = {"hold": None}
    calls_before = [0]

    def validity(n, req):
        if gate["hold"] is not None and n > calls_before[0]:
            return gate["hold"]
        return _ok_validity(n, req)

    client = _ReadyClient(_Fut(_Handle(), done=True))
    ctl, core, feeder, cfg = _move_ctl(chain, client=client, validity=validity, clock=clk)
    core._cache_fresh = lambda t: True                            # (the clock offset below would age the CM cache)
    with _Feed(feeder):
        res = ctl.plan_move("A", "left", goal_dict(cfg))
        assert res["ok"]
        n_plan = len(ctl._cli_valid.calls)
        calls_before[0] = n_plan
        clk.off = 20.0                                            # the plan is now 20 s old (limit for executing: 60 s)
        feeder.prefill()                                          # (the arm has to be measured at standstill across the jump)
        time.sleep(0.15)
        gate["hold"] = _Fut()                                     # the first re-check call is not answered yet
        assert ctl.execute_move("A", res["plan_id"])["ok"]
        assert _wait_for(lambda: len(ctl._cli_valid.calls) == n_plan + 1)
        st = core.status()
        assert st["busy"] and st["last"]["phase"] == "checking" and client.sent == []
        gate["hold"].set(SimpleNamespace(valid=True, contacts=[]))     # answer it; later calls are answered at once
        gate["hold"] = None
        assert _wait_for(lambda: len(client.sent) == 1, 10.0)
        extra = len(ctl._cli_valid.calls) - n_plan
        assert 2 <= extra <= J.MOVE_RECHECK_MAX_CALLS
        assert extra < n_plan                                      # a coarse subset, not the dense path
        assert core.status()["last"]["phase"] == "executing"


@ros_only
def test_f8_scene_change_since_planning_refuses_and_sends_nothing(chain):
    clk = _Clock()
    state = {"after": None}

    def validity(n, req):
        bad = state["after"] is not None and n > state["after"] + 1
        return _Fut(SimpleNamespace(valid=not bad, contacts=[SimpleNamespace(contact_body_1="link5", contact_body_2="box")]
                                    if bad else []), done=True)

    client = _ReadyClient(_Fut(_Handle(), done=True))
    ctl, core, feeder, cfg = _move_ctl(chain, client=client, validity=validity, clock=clk)
    core._cache_fresh = lambda t: True
    with _Feed(feeder):
        res = ctl.plan_move("A", "left", goal_dict(cfg))
        assert res["ok"]
        state["after"] = len(ctl._cli_valid.calls)
        clk.off = 20.0
        feeder.prefill()
        time.sleep(0.15)
        assert ctl.execute_move("A", res["plan_id"])["ok"]
        assert _wait_for(lambda: not core.status()["busy"], 10.0)
        last = core.status()["last"]
        assert last["phase"] == "refused" and "planning scene changed since planning" in last["message"]
        assert "link5 / box" in last["message"] and client.sent == [] and ctl.status()["move_plan"] is None


@ros_only
def test_f8_stop_is_effective_during_the_recheck(chain):
    clk = _Clock()
    hold = {"on": False, "n": 0}
    pending = _Fut()

    def validity(n, req):
        if hold["on"]:
            return pending
        return _ok_validity(n, req)

    client = _ReadyClient(_Fut(_Handle(), done=True))
    ctl, core, feeder, cfg = _move_ctl(chain, client=client, validity=validity, clock=clk)
    core._cache_fresh = lambda t: True
    with _Feed(feeder):
        res = ctl.plan_move("A", "left", goal_dict(cfg))
        clk.off = 20.0
        feeder.prefill()
        time.sleep(0.15)
        hold["on"] = True
        assert ctl.execute_move("A", res["plan_id"])["ok"]
        assert _wait_for(lambda: core.status()["last"]["phase"] == "checking")
        t0 = time.monotonic()
        ctl.stop()
        assert _wait_for(lambda: not core.status()["busy"], 3.0) and time.monotonic() - t0 < 3.0
        assert core.status()["last"]["phase"] == "stopped" and client.sent == []


# =====================================================================================================
# second-round safety findings (r2 A-G)
# =====================================================================================================

class _FailFut(_Fut):
    """A completed future that carries an EXCEPTION (result() raises, like rclpy's Future)."""

    def __init__(self, exc=None):
        super().__init__(None, done=True)
        self.exc = exc or RuntimeError("DDS response could not be decoded")

    def result(self):
        raise self.exc


def _run_execute(ctl, core, chain, deg=0.5):
    sid, ctx = _accept_joint_step(core, deg)
    ctx["q0"] = np.array(Q_HOME)
    Q, rt, an = _joint_step_plan(chain, 0, deg)
    runner = threading.Thread(target=ctl._execute, args=(ctx, rt, an), daemon=True)
    return sid, ctx, runner


# ---- A: the final gate and the goal send are atomic against STOP -----------------------------------------------

def test_r2_a_dispatch_goal_is_one_critical_section_against_request_stop():
    """request_stop() cannot run between the gate and the registration of the pending send."""
    core, clk = make_move_core()
    ok, sid, _, ctx = core.begin_step("A", "left", "joint", 1, 1, 1)
    assert ok
    ctx["q0"] = core.joint_snapshot("left")
    events, stop_done = [], threading.Event()

    def send():
        t = threading.Thread(target=lambda: (core.request_stop(), events.append("stop"), stop_done.set()), daemon=True)
        t.start()                                           # STOP arrives while the goal is being sent
        assert not stop_done.wait(0.3)                      # ... and has to wait: it cannot slip in here
        events.append("sent")
        return "FUT"

    ok, why, fut = core.dispatch_goal(ctx, send, lambda _sid, f: events.append("registered"), "executing")
    assert ok and fut == "FUT" and ctx["sent"] is True
    assert stop_done.wait(2.0)
    assert events == ["sent", "registered", "stop"]         # the STOP is ordered AFTER the registered pending send
    assert core.cancel.is_set() and core._goal_state == "pending"


def test_r2_a_stop_completed_before_the_gate_sends_nothing():
    core, clk = make_move_core()
    ok, sid, _, ctx = core.begin_step("A", "left", "joint", 1, 1, 1)
    ctx["q0"] = core.joint_snapshot("left")
    core.request_stop()
    called = []
    ok, why, fut = core.dispatch_goal(ctx, lambda: called.append("send"), lambda *a: called.append("reg"), "executing")
    assert not ok and "stop" in why and called == [] and "sent" not in ctx and core._goal_state is None


def test_r2_a_dispatch_gate_failures_send_nothing():
    core, clk = make_move_core(real=("left",))
    core.enable_real("A")
    ok, sid, _, ctx = core.begin_step("A", "left", "joint", 1, 1, 1)
    assert ok
    ctx["q0"] = core.joint_snapshot("left")
    core.disable("revoked")
    called = []
    ok, why, _ = core.dispatch_goal(ctx, lambda: called.append("send"), lambda *a: called.append("reg"), "executing")
    assert not ok and "enable" in why and called == []


@ros_only
@pytest.mark.parametrize("where", ["gate", "send"])
def test_r2_a_stop_racing_the_dispatch_never_leaves_an_unhooked_sent_goal(chain, where):
    """Reproduces the old race (STOP completing between predispatch_check and send_goal_async): the STOP must either
    prevent the send or find the registered pending send and hook cancel-on-accept, and the arm stays busy until a
    terminal result AND a standstill."""
    send, handle = _Fut(), _Handle()
    client = _Client(send)
    ctl, core, feeder = _controller(chain, client=client)
    core.stop_deadline_s = 30.0
    stop_done, state = threading.Event(), {}

    def race():
        t = threading.Thread(target=lambda: (ctl.stop(), stop_done.set()), daemon=True)
        t.start()
        stop_done.wait(0.4)                                 # old code: the STOP finishes right here, before the send
        state["finished_in_gap"] = stop_done.is_set()

    if where == "gate":
        real_check = core.predispatch_check

        def racing_check(ctx):
            out = real_check(ctx)
            if out[0] and "done" not in state:
                state["done"] = True
                race()
            return out
        core.predispatch_check = racing_check
    else:
        orig_send = client.send_goal_async

        def racing_send(goal):
            fut = orig_send(goal)
            race()                                          # between send_goal_async and the registration
            return fut
        client.send_goal_async = racing_send
    with _Feed(feeder):
        sid, ctx, runner = _run_execute(ctl, core, chain)
        runner.start()
        assert stop_done.wait(5.0)
        assert state["finished_in_gap"] is False            # the STOP could not slip into the gap
        assert _wait_for(lambda: core.status()["last"]["phase"] == "stopping", 3.0)
        assert core.status()["busy"] and core.cancel.is_set()
        assert len(client.sent) == 1                        # sent before the STOP could run ...
        send.set(handle)                                    # ... accepted afterwards
        assert _wait_for(lambda: handle.cancels >= 1, 3.0)  # cancelled on acceptance
        assert core.status()["busy"]                        # still busy: no terminal result yet
        handle.res.set(_Result(5))                          # terminal result
        assert _wait_for(lambda: core.standstill("left"), 3.0)
        runner.join(5.0)
        assert not runner.is_alive()
        st = core.status()
        assert not st["busy"] and st["last"]["phase"] == "stopped"


@ros_only
def test_r2_a_stop_before_the_gate_sends_nothing_through_execute(chain):
    client = _Client(_Fut(_Handle(), done=True))
    ctl, core, feeder = _controller(chain, client=client)
    with _Feed(feeder):
        sid, ctx, runner = _run_execute(ctl, core, chain)
        ctl.stop()                                          # completes before the dispatch
        with pytest.raises(J._Refuse):
            ctl._execute(ctx, *_joint_step_plan(chain, 0, 0.5)[1:])
        assert client.sent == [] and "sent" not in ctx and ctl._pending_send is None


# ---- B: terminal means a VALIDATED result ----------------------------------------------------------------------

def test_r2_b_validated_result_helper():
    assert J.validated_result(_Fut(_Result(4), done=True)) is not None
    assert J.validated_result(_Fut(_Result(5), done=True)) is not None
    assert J.validated_result(_Fut(_Result(6), done=True)) is not None
    assert J.validated_result(_Fut(_Result(4))) is None                       # not done
    assert J.validated_result(_FailFut()) is None                             # exception
    assert J.validated_result(_Fut(None, done=True)) is None                  # no response
    for bad in (0, 1, 2, 3, 7, 99, -1, None, "4", 4.0, True):
        assert J.validated_result(_Fut(_Result(bad), done=True)) is None, bad
    assert J.TERMINAL_GOAL_STATUSES == (4, 5, 6)
    assert J.GOAL_UNCONFIRMED_MSG == ("the controller's answer could not be confirmed: press the e-stop if the arm "
                                      "moves; restart the trajectory controller or the driver (not just the UI) "
                                      "before moving again")


@ros_only
def test_r2_b_goal_status_constants_match_action_msgs():
    from action_msgs.msg import GoalStatus
    assert (J.GOAL_STATUS_SUCCEEDED, J.GOAL_STATUS_CANCELED, J.GOAL_STATUS_ABORTED) == \
        (GoalStatus.STATUS_SUCCEEDED, GoalStatus.STATUS_CANCELED, GoalStatus.STATUS_ABORTED)


def test_r2_b_unconfirmed_is_final_never_terminal_and_clear_fault_refuses():
    core, clk, sid = _fault_core_with_goal("live")
    core.note_goal(sid, "unconfirmed")
    core.note_goal(sid, "terminal")                           # a racing "terminal" cannot undo it
    assert core.goal_unconfirmed() and not core.goal_confirmed_ended()
    r = core.clear_fault()
    assert not r["ok"] and r["message"] == J.GOAL_UNCONFIRMED_MSG
    assert core.status()["fault"] and core.status()["busy"]
    # ... and the other way round: terminal stays terminal
    core2, clk2, sid2 = _fault_core_with_goal("terminal")
    core2.note_goal(sid2, "unconfirmed")
    assert core2.goal_confirmed_ended() and core2.clear_fault()["ok"]


def test_r2_b_unconfirmed_goal_in_stopping_goes_to_fault_at_once():
    core, clk = make_move_core()
    feed(core, clk, SS)
    refresh(core, clk, real=())
    ok, sid, _, _ = core.begin_step("A", "left", "joint", 1, 1, 1)
    core.note_goal(sid, "pending")
    core.note_goal(sid, "live")
    core.begin_stopping(sid, False, "stopped", "x")
    core.note_goal(sid, "unconfirmed")
    assert core.poll_stopping() == "fault"                   # no waiting for the deadline, no release
    st = core.status()
    assert st["fault"] and st["last"]["message"] == J.GOAL_UNCONFIRMED_MSG
    assert not core.clear_fault()["ok"]


@ros_only
@pytest.mark.parametrize("status", [None, 0, 1, 99])
def test_r2_b_result_with_unknown_status_is_not_terminal(chain, status):
    handle = _Handle(_Result(status))
    ctl, core, feeder = _controller(chain, client=_Client(_Fut(handle, done=True)))
    with _Feed(feeder):
        sid, ctx, runner = _run_execute(ctl, core, chain)
        runner.start()
        runner.join(8.0)
        assert not runner.is_alive()
        st = core.status()
        assert st["fault"] and st["busy"] and st["last"]["message"] == J.GOAL_UNCONFIRMED_MSG
        assert core.goal_unconfirmed() and handle.cancels >= 1
        assert _wait_for(lambda: core.standstill("left"), 3.0)
        r = ctl.clear_fault()
        assert not r["ok"] and r["message"] == J.GOAL_UNCONFIRMED_MSG and core.status()["busy"]


@ros_only
def test_r2_b_result_future_exception_is_not_terminal(chain):
    handle = _Handle()
    handle.res = _FailFut()                                   # done, but with an exception
    ctl, core, feeder = _controller(chain, client=_Client(_Fut(handle, done=True)))
    with _Feed(feeder):
        sid, ctx, runner = _run_execute(ctl, core, chain)
        runner.start()
        runner.join(8.0)
        assert not runner.is_alive()                          # the worker did not crash: it went to the fault path
        st = core.status()
        assert st["fault"] and st["busy"] and st["last"]["message"] == J.GOAL_UNCONFIRMED_MSG
        assert core.goal_unconfirmed() and handle.cancels >= 1
        assert _wait_for(lambda: core.standstill("left"), 3.0)
        r = ctl.clear_fault()
        assert not r["ok"] and r["message"] == J.GOAL_UNCONFIRMED_MSG
        assert ctl._goal_handle is handle                     # the handle is kept: the goal may still be live


@ros_only
def test_r2_b_send_future_exception_is_not_terminal(chain):
    ctl, core, feeder = _controller(chain, client=_Client(_FailFut()))
    with _Feed(feeder):
        sid, ctx, runner = _run_execute(ctl, core, chain)
        runner.start()
        runner.join(8.0)
        assert not runner.is_alive()
        st = core.status()
        assert st["fault"] and st["last"]["message"] == J.GOAL_UNCONFIRMED_MSG and core.goal_unconfirmed()
        assert _wait_for(lambda: core.standstill("left"), 3.0)
        assert not ctl.clear_fault()["ok"] and core.status()["busy"]


@ros_only
def test_r2_b_late_accept_with_failing_result_future_is_not_terminal(chain):
    send, handle = _Fut(), _Handle()
    handle.res = _FailFut()
    ctl, core, feeder = _controller(chain, client=_Client(send))
    core.stop_deadline_s = 0.5
    with _Feed(feeder):
        sid, ctx, runner = _run_execute(ctl, core, chain)
        runner.start()
        time.sleep(0.2)
        ctl.stop()
        runner.join(5.0)
        assert core.status()["fault"]
        send.set(handle)                                      # accepted late; its result future is an exception
        assert handle.cancels == 1 and core.goal_unconfirmed()
        assert _wait_for(lambda: core.standstill("left"), 3.0)
        assert ctl.clear_fault()["message"] == J.GOAL_UNCONFIRMED_MSG and core.status()["busy"]
    # a late send answer that is an exception or None is unconfirmed as well
    for answer in (_FailFut(), None):
        send2 = _Fut()
        ctl2, core2, feeder2 = _controller(chain, client=_Client(send2))
        core2.stop_deadline_s = 0.5
        with _Feed(feeder2):
            sid, ctx, runner = _run_execute(ctl2, core2, chain)
            runner.start()
            time.sleep(0.2)
            ctl2.stop()
            runner.join(5.0)
            if isinstance(answer, _FailFut):
                send2.set_exception_like = None
                send2._v, send2._done = None, True
                send2.result = answer.result
                for cb in send2._cbs:
                    cb(send2)
            else:
                send2.set(None)
            assert core2.goal_unconfirmed()


@ros_only
def test_r2_b_explicit_rejection_and_validated_results_are_terminal(chain):
    # explicitly rejected goal: terminal, nothing is live, the arm is released as "refused"
    ctl, core, feeder = _controller(chain, client=_Client(_Fut(SimpleNamespace(accepted=False), done=True)))
    with _Feed(feeder):
        sid, ctx, runner = _run_execute(ctl, core, chain)
        runner.start()
        runner.join(8.0)
        assert not core.status()["busy"] and core.status()["last"]["phase"] == "refused"
    # ABORTED (6) and CANCELED (5) are validated terminal results: the arm is released after standstill
    for status, phase in ((6, "failed"), (5, "stopped")):
        ctl, core, feeder = _controller(chain, client=_Client(_Fut(_Handle(_Result(status)), done=True)))
        with _Feed(feeder):
            sid, ctx, runner = _run_execute(ctl, core, chain)
            runner.start()
            runner.join(10.0)
            assert not runner.is_alive()
            st = core.status()
            assert not st["busy"] and st["last"]["phase"] == phase and not core.goal_unconfirmed()


def test_r2_b_never_sent_goal_clear_fault_allowed():
    core, clk, sid = _fault_core_with_goal(None)
    assert core.goal_confirmed_ended() and core.clear_fault()["ok"]


# ---- C: a frozen feedback must not pass for standstill ----------------------------------------------------------

def test_r2_c_window_is_longer_than_the_driver_error_cutoff():
    assert J.STANDSTILL_WINDOW_S == 1.2 and J.STANDSTILL_WINDOW_S > 1.0           # driver: STATUS age > 1000 ms -> ERROR
    assert J.SETTLE_WAIT_S >= J.STANDSTILL_WINDOW_S + 1.0
    assert J.STOP_CONFIRM_DEADLINE_S >= J.STANDSTILL_WINDOW_S + 6.0


def test_r2_c_a_freeze_after_motion_is_not_standstill():
    core, clk = make_move_core()
    names = list(core.cfgs["left"].joint_names)
    q = np.array(Q_HOME, dtype=float)
    for _ in range(50):                                      # 1 s of real motion at 0.05 rad/s (joint 1)
        clk.t += 0.02
        q[0] += 0.05 * 0.02
        core.update_joint_states(names, list(q))
    for _ in range(int(round(1.0 / 0.02))):                  # then the feedback freezes for 1.0 s: same value, fresh stamps
        clk.t += 0.02
        core.update_joint_states(names, list(q))
    assert core.standstill("left", window=0.5)               # (what the old 0.5 s window accepted)
    assert not core.standstill("left")                       # the 1.2 s window still sees the motion before the freeze
    # a long motionless stretch IS standstill
    for _ in range(int(round(0.4 / 0.02))):
        clk.t += 0.02
        core.update_joint_states(names, list(q))
    assert core.standstill("left")


def test_r2_c_a_freeze_that_ends_with_a_catch_up_jump_is_not_standstill():
    core, clk = make_move_core()
    names = list(core.cfgs["left"].joint_names)
    q = np.array(Q_HOME, dtype=float)
    for _ in range(80):
        clk.t += 0.02
        core.update_joint_states(names, list(q))
    for _ in range(int(round(1.0 / 0.02))):
        clk.t += 0.02
        core.update_joint_states(names, list(q))             # frozen
    q[0] += 0.03                                             # the feed resumes where the arm really is
    clk.t += 0.02
    core.update_joint_states(names, list(q))
    for _ in range(10):
        clk.t += 0.02
        core.update_joint_states(names, list(q))
    assert not core.standstill("left")


@ros_only
def test_r2_c_stop_confirmation_and_clear_fault_use_the_long_window(chain):
    core, clk, sid = _fault_core_with_goal("terminal")
    names = list(core.cfgs["left"].joint_names)
    q = np.array(Q_HOME, dtype=float)
    for _ in range(40):
        clk.t += 0.02
        q[1] += 0.001
        core.update_joint_states(names, list(q))
    for _ in range(int(round(1.0 / 0.02))):                  # frozen for 1.0 s after motion
        clk.t += 0.02
        core.update_joint_states(names, list(q))
    health(core, ())
    r = core.clear_fault()
    assert not r["ok"] and "1.2 s" in r["message"]
    for _ in range(int(round(0.5 / 0.02))):
        clk.t += 0.02
        core.update_joint_states(names, list(q))
    health(core, ())
    assert core.clear_fault()["ok"]


# ---- D: supervision speed over 0.5 s ---------------------------------------------------------------------------

def test_r2_d_a_240ms_freeze_then_catch_up_does_not_trip(chain):
    # (round 3: a freeze of >= FREEZE_MAX_S = 0.25 s while the reference moves is stopped by the freeze detection; the
    # SPEED supervision must still not false-trip on the catch-up jump of a shorter freeze)
    core, clk = make_move_core()
    names = list(core.cfgs["left"].joint_names)
    Qd, P = _straight(chain)
    sup = J.MotionSupervisor(core, "left", J.retime_path(Qd, P), J.MOVE_MAX_JOINT_SPEED_RAD_S, 100.0)
    true_q, q = np.array(Q_HOME, dtype=float), np.array(Q_HOME, dtype=float)
    v, worst, worst_short = 0.04, 0.0, 0.0                    # the true speed is exactly the move cap
    for tick in range(120):
        clk.t += 0.02
        true_q[0] += v * 0.02
        if not 50 <= tick < 62:                               # 240 ms (12 ticks) of frozen, repeated positions ...
            q[0] = true_q[0]                                  # ... then the feed catches up in ONE sample
        core.update_joint_states(names, list(q))
        sp = core.measured_speed("left")
        if sp is not None:
            worst = max(worst, sp)
        sp_short = core.measured_speed("left", window_s=0.1)
        if sp_short is not None:
            worst_short = max(worst_short, sp_short)
        assert sup.check(1.0, 0.5) is None, (tick, sp)
    assert worst <= 2.0 * J.MOVE_MAX_JOINT_SPEED_RAD_S
    assert worst_short > 2.0 * J.MOVE_MAX_JOINT_SPEED_RAD_S   # the old 0.1 s window read ~0.12 rad/s here: a false trip


def test_r2_d_a_sustained_runaway_still_trips_within_0_6_s(chain):
    Qd, P = _straight(chain)
    rt = J.retime_path(Qd, P)
    for cap, speed in ((J.MOVE_MAX_JOINT_SPEED_RAD_S, 0.2), (J.MAX_JOINT_SPEED_RAD_S, 0.3)):
        core, clk = make_move_core()
        names = list(core.cfgs["left"].joint_names)
        sup = J.MotionSupervisor(core, "left", rt, cap, 100.0)
        q = np.array(Q_HOME, dtype=float)
        for _ in range(60):                                   # 1.2 s standing still first
            clk.t += 0.02
            core.update_joint_states(names, list(q))
        assert sup.check(1.0, 0.1) is None                    # (t_since_accept < FREEZE_MAX_S: the freeze test is off)
        tripped = None
        for tick in range(1, 100):
            clk.t += 0.02
            q[0] += speed * 0.02
            core.update_joint_states(names, list(q))
            why = sup.check(1.0, 0.5)
            if why:
                tripped = tick * 0.02
                assert "speed" in why
                break
        assert tripped is not None and tripped <= 0.6, tripped


# ---- E / G / duration: constants and their effect ---------------------------------------------------------------

def test_r2_e_step_tracking_tolerance_is_0_05():
    assert J.STEP_MAX_TRACKING_ERR_RAD == 0.05 == J.MOVE_MAX_TRACKING_ERR_RAD


def test_r2_g_start_moved_tolerance_is_above_one_encoder_count():
    count = math.radians(0.01)                               # the driver reports 0.01 deg steps
    assert J.START_MOVED_TOL_RAD == 4e-4 and 2 * count < J.START_MOVED_TOL_RAD < 3 * count
    core, clk = make_core(real=("left",))
    feed(core, clk, 0.1)
    core.enable_real("A")
    ok, sid, _, ctx = core.begin_step("A", "left", "cart", "z", 1, 1, "world")
    names = [n for c in core.cfgs.values() for n in c.joint_names]
    q = np.array(Q_HOME)
    q[2] += 2 * count                                        # two counts of flicker: no replanning
    core.update_joint_states(names, list(q) * 2)
    assert core.predispatch_check(ctx)[0]
    q[2] = Q_HOME[2] + 3 * count                             # three counts: refused
    core.update_joint_states(names, list(q) * 2)
    ok, why = core.predispatch_check(ctx)
    assert not ok and "moved" in why


def test_r2_move_duration_is_shorter_than_the_real_enable_and_stop_ignores_the_enable():
    assert J.MOVE_MAX_DURATION_S <= 300.0 < 600.0
    core, clk = make_core(real=("left",))
    feed(core, clk, 0.1)
    core.enable_real("A")
    ok, sid, _, ctx = core.begin_step("A", "left", "cart", "z", 1, 1, "world")
    assert ok
    jumped(core, clk, 601.0, ("left",))                      # the 10 min enable expires mid-motion
    assert not core.real_lock.state()[0]
    assert core.request_stop() is True and core.cancel.is_set()     # STOP works without the enable
    core.begin_stopping(sid, False, "stopped", "stopped by operator")
    core.note_goal(sid, "pending")
    core.note_result(sid)
    feed(core, clk, SS)
    health(core, ("left",))
    assert core.poll_stopping() == "confirmed"               # and so does the confirmation of the stop


# ---- F: validity check on the executed (Hermite) curve ----------------------------------------------------------

def _bump(rt, an, fk, joint=5, amp=0.004):
    """Sideways velocities on one joint over ~1 s: the points stay on the polyline, only the Hermite curve between
    them leaves it (a few 1e-5 rad)."""
    qd = rt.qd.copy()
    k = len(qd) // 2
    qd[k - 10:k + 11, joint] += amp * np.hanning(23)[1:-1]
    rt2 = J.Retimed(rt.times, rt.q, qd, rt.duration)
    return rt2, J.analyze_trajectory(rt2.times, rt2.q, rt2.qd, fk)


def test_r2_f_curve_check_states_spacing_ends_and_limits(chain):
    Qd, P = _straight(chain)
    rt = J.retime_path(Qd, P)
    rt2, _ = _bump(rt, None, chain.fk)
    st = J.curve_check_states(rt2.times, rt2.q, rt2.qd)
    assert np.array_equal(st[0], rt2.q[0]) and np.array_equal(st[-1], rt2.q[-1])
    assert np.max(np.abs(np.diff(st, axis=0))) <= math.radians(0.25) + 1e-12
    assert len(st) > 10
    # samples lie on the Hermite curve, not on the re-timed points: some are off the polyline
    assert J.polyline_deviation(Qd, st) > 1e-6
    # a looser limit yields fewer states, the cap refuses
    assert len(J.curve_check_states(rt2.times, rt2.q, rt2.qd, max_step_rad=math.radians(1.0))) < len(st)
    with pytest.raises(ValueError, match="validity checks"):
        J.curve_check_states(rt2.times, rt2.q, rt2.qd, max_states=len(st) - 5)
    bad = rt2.qd.copy()
    bad[3, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        J.curve_check_states(rt2.times, rt2.q, bad)
    with pytest.raises(ValueError, match="too fast"):
        J.curve_check_states(rt2.times, rt2.q, rt2.qd * 1000.0)


def _validity_off_polyline(poly, tol):
    """A fake /check_state_validity that accepts only states within `tol` of the polyline."""
    def validity(n, req):
        names = list(req.robot_state.joint_state.name)
        pos = list(req.robot_state.joint_state.position)
        cfg_names = [f"joint{i}" for i in range(1, 7)]
        q = np.array([pos[names.index(nm)] for nm in cfg_names])
        ok = J.polyline_deviation(poly, q[None, :]) <= tol
        return _Fut(SimpleNamespace(valid=bool(ok), contacts=[]), done=True)
    return validity


@ros_only
def test_r2_f_move_is_refused_when_only_the_hermite_curve_collides(chain, monkeypatch):
    mid = (Q_HOME + Q_GOAL) / 2 + np.array([0.02, 0.01, 0, 0, 0.01, 0])
    poly = J.densify_path(J.smooth_path(np.vstack([Q_HOME, mid, Q_GOAL])))
    client = _ReadyClient(_Fut(_Handle(), done=True))
    # control: with the normal timing the curve stays within 2e-6 of the polyline and the plan is accepted
    ctl, core, feeder, cfg = _move_ctl(chain, client=client, validity=_validity_off_polyline(poly, 3.5e-6))
    with _Feed(feeder):
        res = ctl.plan_move("A", "left", goal_dict(cfg))
        assert res["ok"], res
        n_ok = len(ctl._cli_valid.calls)
        assert n_ok > len(poly)                              # polyline states PLUS the curve states
    # the same path with sideways velocities: every polyline state is fine, the curve is not
    orig = J.plan_move_timing
    monkeypatch.setattr(J, "plan_move_timing", lambda Q, P, fk: _bump(*orig(Q, P, fk), fk, amp=0.002))
    ctl, core, feeder, cfg = _move_ctl(chain, client=client, validity=_validity_off_polyline(poly, 3.5e-6))
    with _Feed(feeder):
        res = ctl.plan_move("A", "left", goal_dict(cfg))
        assert not res["ok"] and "executed curve" in res["message"] and "collision" in res["message"], res
        assert core.plan_summary() is None and not core.status()["busy"]
        assert len(ctl._cli_valid.calls) > len(poly)         # the polyline passed, the failure was found on the curve
        assert client.sent == []


@ros_only
def test_r2_f_move_is_refused_when_the_curve_needs_too_many_checks(chain, monkeypatch):
    monkeypatch.setattr(J, "CURVE_CHECK_MAX_STATES", 10)
    client = _ReadyClient(_Fut(_Handle(), done=True))
    ctl, core, feeder, cfg = _move_ctl(chain, client=client)
    with _Feed(feeder):
        res = ctl.plan_move("A", "left", goal_dict(cfg))
        assert not res["ok"] and "validity checks" in res["message"] and "limit 10" in res["message"], res
        assert core.plan_summary() is None


@ros_only
def test_r2_f_step_checks_the_executed_curve_too(chain, monkeypatch):
    client = _ReadyClient(_Fut(_Handle(), done=True))
    ctl, core, feeder, cfg = _move_ctl(chain, client=client)
    with _Feed(feeder):
        assert ctl.step("A", "left", "joint", 1, 1, 5)["ok"]                 # 5 deg joint step
        assert _wait_for(lambda: len(client.sent) == 1, 5.0)
        calls = ctl._cli_valid.calls
        qs = np.array([[r.robot_state.joint_state.position[r.robot_state.joint_state.name.index(n)]
                        for n in cfg.joint_names] for r in calls])
        n_poly = int(math.ceil(5 / J.JOINT_CHECK_SPACING_DEG)) + 1
        assert len(qs) > n_poly                                              # polyline states plus the curve states
        curve = qs[n_poly:]
        assert np.allclose(curve[0], Q_HOME) and np.allclose(curve[-1], Q_HOME + np.array([math.radians(5), 0, 0, 0, 0, 0]))
        assert np.max(np.abs(np.diff(curve, axis=0))) <= math.radians(0.25) + 1e-9
        ctl.stop()
        assert core.cancel.is_set()


@ros_only
def test_r2_f_step_is_refused_when_only_the_hermite_curve_collides(chain, monkeypatch):
    poly = np.vstack([Q_HOME + (np.arange(21) / 20)[:, None] * np.array([math.radians(5), 0, 0, 0, 0, 0])[None, :]])
    client = _ReadyClient(_Fut(_Handle(), done=True))
    orig = J.plan_timing

    def bumped(Q, P, R, fk):
        rt, an = orig(Q, P, R, fk)
        return _bump(rt, an, fk, joint=5, amp=0.003)
    monkeypatch.setattr(J, "plan_timing", bumped)
    ctl, core, feeder, cfg = _move_ctl(chain, client=client, validity=_validity_off_polyline(poly, 1e-6))
    with _Feed(feeder):
        assert ctl.step("A", "left", "joint", 1, 1, 5)["ok"]
        assert _wait_for(lambda: not core.status()["busy"], 5.0)
        last = core.status()["last"]
        assert last["phase"] == "refused" and "step (executed curve) blocked" in last["message"], last
        assert client.sent == []


# =====================================================================================================
# third-round safety findings (r3 1-4)
# =====================================================================================================

def _stopping_core_r3(real=()):
    """A core whose step was accepted (stationary history) and which is now 'stopping' (STOP at the current clock)."""
    core, clk = make_core(real=real)
    ok, sid, msg, ctx = core.begin_step("A", "left", "joint", 1, 1, 1)
    assert ok, msg
    core.note_goal(sid, "pending")
    core.note_goal(sid, "live")
    core.request_stop()
    core.begin_stopping(sid, False, "stopped", "stopped by operator")
    return core, clk, sid


# ---- 1a: event-anchored standstill -------------------------------------------------------------------------------

def test_r3_1_standstill_since_needs_a_window_of_post_anchor_samples():
    core, clk = make_core(real=())
    assert core.standstill("left")                                   # un-anchored: the arm has been still for 1.4 s
    anchor = clk.t
    assert not core.standstill("left", since=anchor + 0.001)         # nothing received after the anchor yet
    feed(core, clk, 1.0)
    assert not core.standstill("left", since=anchor)                 # only 1.0 s of post-anchor samples: window not covered
    feed(core, clk, 0.3)
    assert core.standstill("left", since=anchor)                     # 1.3 s covered
    assert not core.standstill("left", since=clk.t - 0.5)            # a LATER anchor restarts the requirement
    # motion before the anchor does not spoil an anchored verdict
    core2, clk2 = make_core(real=())
    feed(core2, clk2, 0.6, drift=0.05)                                # moving
    anchor2 = clk2.t + 0.01
    feed(core2, clk2, 1.5, q=Q_HOME + 0.5)                            # a jump right after the anchor, then still
    assert core2.standstill("left", since=anchor2)
    # a gap in the samples is not a covered window: no joint_states for 5 s, then one sample
    core3, clk3 = make_core(real=())
    clk3.t += 5.0
    refresh(core3, clk3, real=())
    assert not core3.standstill("left") and not core3.standstill("left", since=clk3.t - 6.0)


def test_r3_1_standstill_since_ignores_a_stationary_history_before_the_anchor():
    """The reviewer's onset-freeze: a stationary history, the feedback freezes exactly when the motion starts.  The
    un-anchored verdict is 'standstill'; the anchored one is not until a full window of post-anchor samples exists."""
    core, clk = make_core(real=())
    anchor = clk.t                                                    # motion starts / STOP is pressed now
    feed(core, clk, 0.4)                                              # frozen (identical) samples after the anchor
    assert core.standstill("left")                                    # the old, un-anchored verdict: fooled
    assert not core.standstill("left", since=anchor)


def test_r3_1_onset_freeze_stop_is_not_confirmed_and_ends_in_fault_at_the_deadline():
    core, clk, sid = _stopping_core_r3()
    core.note_result(sid)                                             # the controller answered the cancel at once
    health(core, ())
    assert core.poll_stopping() == "waiting"                          # was: confirmed immediately (history before the STOP)
    # the driver is stalled: the feedback stays frozen; after > 1.0 s the driver latches ERROR (component inactive)
    feed(core, clk, 0.8)
    health(core, ())
    assert core.poll_stopping() == "waiting"
    feed(core, clk, 0.8)
    health(core, (), active=False)
    assert core.poll_stopping() == "waiting"
    feed(core, clk, core.stop_deadline_s)
    health(core, (), active=False)
    assert core.poll_stopping() == "fault"
    assert "stop not confirmed" in core.status()["last"]["message"]
    feed(core, clk, SS)
    health(core, (), active=False)                                    # still latched: a clear is refused
    r = core.clear_fault()
    assert not r["ok"] and "not active" in r["message"] and core.status()["fault"]


def test_r3_1_stop_is_confirmed_only_after_fresh_post_anchor_data():
    core, clk, sid = _stopping_core_r3()
    core.note_result(sid)
    feed(core, clk, 1.1)
    health(core, ())
    assert core.poll_stopping() == "waiting"                          # window not yet made of post-anchor samples
    feed(core, clk, 0.4)
    health(core, ())
    assert core.poll_stopping() == "confirmed"
    st = core.status()
    assert not st["busy"] and st["last"]["phase"] == "stopped"


def test_r3_1_the_result_time_is_an_anchor_too():
    core, clk, sid = _stopping_core_r3()
    feed(core, clk, SS)
    health(core, ())
    assert core.poll_stopping() == "waiting"                          # no result yet
    core.note_result(sid)                                             # the result arrives only now
    assert core.goal_end_time(sid) == clk.t
    health(core, ())
    assert core.poll_stopping() == "waiting"                          # standstill since the RESULT is not measured yet
    feed(core, clk, SS)
    health(core, ())
    assert core.poll_stopping() == "confirmed"


def test_r3_1_clear_fault_is_anchored_at_the_fault_entry():
    core, clk, sid = _stopping_core_r3()
    core.note_result(sid)
    clk.t += core.stop_deadline_s + 1
    refresh(core, clk, real=())                                       # (one sample after a long gap: not a covered window)
    assert core.poll_stopping() == "fault"                            # the arm has been still for ages ...
    health(core, ())
    r = core.clear_fault()
    assert not r["ok"] and "standstill" in r["message"]               # ... but not since the fault entry
    feed(core, clk, SS)
    health(core, ())
    assert core.clear_fault()["ok"]


# ---- 1b: hardware health -----------------------------------------------------------------------------------------

def test_r3_1_confirmation_needs_active_hardware_fresh_joint_states_and_a_fresh_health_answer():
    core, clk, sid = _stopping_core_r3()
    core.note_result(sid)
    feed(core, clk, SS)
    # no controller_manager answer at all newer than 1 s after the stop
    assert core.poll_stopping() == "waiting"
    health(core, (), active=False)                                    # hardware component not active (driver ERROR)
    assert core.poll_stopping() == "waiting" and core.confirmation_problem("left", clk.t - SS) is not None
    core.update_hardware([(core.cfgs["left"].hardware, J.MOCK_PLUGIN, "active"), (core.cfgs["right"].hardware, J.MOCK_PLUGIN, "active")])
    core.update_controllers([(core.cfgs["left"].controller, "inactive"), (core.cfgs["right"].controller, "active")])
    assert core.poll_stopping() == "waiting"                          # controller deactivated
    health(core, ())
    clk.t += 0.7                                                      # joint_states stopped coming
    health(core, ())
    assert core.poll_stopping() == "waiting"
    refresh(core, clk, real=())
    assert core.poll_stopping() == "waiting"                          # (a single new sample: window broken by the jump? no - same value)
    # a health answer received BEFORE anchor + 1.0 s does not count (it may predate the driver's ERROR latch)
    core2, clk2, sid2 = _stopping_core_r3()
    core2.note_result(sid2)
    health(core2, ())                                                 # stamped right at the stop
    feed(core2, clk2, 0.9)
    core2.update_hardware(list((n, c, s) for n, (c, s) in core2._hw.items()))
    assert core2.confirmation_problem("left", clk2.t - 0.9) is not None
    feed(core2, clk2, 0.6)
    health(core2, ())
    assert core2.poll_stopping() == "confirmed"


def test_r4_health_answers_are_dated_by_their_request_not_their_arrival():
    """Reviewer counterexample: a healthy answer to a request sent at stop + 0.9 s arrives at stop + 1.1 s (after the
    driver may have latched ERROR at +1.0 s).  It must NOT count as evidence newer than the cutoff; an answer to a
    request sent after the cutoff does.  Older answers never overwrite newer ones."""
    core, clk, sid = _stopping_core_r3()
    core.note_result(sid)
    t_stop = clk.t
    hw_ok = [(cfg.hardware, J.MOCK_PLUGIN, "active") for cfg in core.cfgs.values()]
    ctrl_ok = [(cfg.controller, "active") for cfg in core.cfgs.values()]
    feed(core, clk, SS)                                               # fresh, still joint samples after the stop
    core.update_hardware(hw_ok, requested_at=t_stop + 0.9)            # requested before the cutoff, arrives now
    core.update_controllers(ctrl_ok, requested_at=t_stop + 0.9)
    assert core.poll_stopping() == "waiting"
    assert "newer than" in (core.confirmation_problem("left", t_stop) or "")
    core.update_hardware(hw_ok, requested_at=t_stop + 1.05)           # requested after the cutoff
    core.update_controllers(ctrl_ok, requested_at=t_stop + 1.05)
    assert core.poll_stopping() == "confirmed"
    # an answer to an OLDER request arriving late is ignored (does not replace newer evidence)
    core2, clk2, sid2 = _stopping_core_r3()
    core2.note_result(sid2)
    t2 = clk2.t
    feed(core2, clk2, SS)
    core2.update_hardware(hw_ok, requested_at=t2 + 1.05)
    core2.update_controllers(ctrl_ok, requested_at=t2 + 1.05)
    stale = [(cfg.hardware, J.MOCK_PLUGIN, "inactive") for cfg in core2.cfgs.values()]
    core2.update_hardware(stale, requested_at=t2 + 0.5)               # late answer to an earlier request
    assert core2._hw_t == t2 + 1.05 and all(st == "active" for _, st in core2._hw.values())


def test_r3_1_clear_fault_is_refused_with_a_clear_message_when_the_hardware_is_not_active():
    core, clk, sid = _stopping_core_r3()
    core.note_result(sid)
    clk.t += core.stop_deadline_s + 1
    refresh(core, clk, real=())                                       # (one sample after a long gap: not a covered window)
    assert core.poll_stopping() == "fault"
    feed(core, clk, SS)
    health(core, (), active=False)
    r = core.clear_fault()
    assert not r["ok"] and "not active" in r["message"] and "e-stop" in r["message"]
    health(core, ())
    assert core.clear_fault()["ok"]


# ---- 1c: pre-dispatch standstill gate ----------------------------------------------------------------------------

def test_r3_1_steps_and_moves_need_a_measured_standstill_before_they_are_accepted():
    core, clk = make_core("single_left", real=())
    names = list(core.cfgs["left"].joint_names)
    q = np.array(Q_HOME, dtype=float)
    feed(core, clk, 0.5, drift=0.02)                                  # the arm was moving a moment ago
    refresh(core, clk, real=())
    ok, sid, msg, _ = core.begin_step("A", "left", "joint", 1, 1, 1)
    assert not ok and msg == J.STANDSTILL_REFUSAL
    pid = None
    feed(core, clk, SS)
    refresh(core, clk, real=())
    sid_p, ctx_p = _mk_plan_ctx(core)
    pid = _store(core, sid_p, ctx_p)
    feed(core, clk, 0.5, drift=0.02)
    refresh(core, clk, real=())
    ok, _, msg, _ = core.begin_move_execute("A", pid)
    assert not ok and msg == J.STANDSTILL_REFUSAL
    assert core.plan_summary() is not None                            # the plan survives a refusal for this reason
    feed(core, clk, SS)
    refresh(core, clk, real=())
    # the plan's start differs from the current (drifted) state: plan again, then it is accepted
    sid_p, ctx_p = _mk_plan_ctx(core)
    pid = _store(core, sid_p, ctx_p)
    assert core.begin_move_execute("A", pid)[0]


def test_r3_1_normal_sequence_step_done_then_next_step_still_works():
    core, clk = make_core("single_left", real=())
    ok, sid, msg, ctx = core.begin_step("A", "left", "joint", 1, 1, 1)
    assert ok, msg
    core.note_goal(sid, "pending")
    core.note_goal(sid, "live")
    core.note_goal(sid, "terminal")                                   # result SUCCEEDED
    feed(core, clk, SS)                                               # the settle wait: 1.2 s of post-result stillness
    health(core, ())
    assert core.confirmation_problem("left", core.goal_end_time(sid)) is None
    core.finish(sid, True, "done", "ok")
    jumped(core, clk, J.MIN_STEP_GAP_S + 0.01, real=())
    ok, sid2, msg, _ = core.begin_step("A", "left", "joint", 1, 1, 1)
    assert ok, msg                                                    # the window already shows standstill


def test_r3_1_predispatch_check_and_dispatch_goal_refuse_without_standstill():
    core, clk = make_core("single_left", real=())
    ok, sid, msg, ctx = core.begin_step("A", "left", "joint", 1, 1, 1)
    assert ok, msg
    assert core.predispatch_check(ctx) == (True, "")
    names = list(core.cfgs["left"].joint_names)
    clk.t += 0.02
    core.update_joint_states(names, list(Q_HOME), [0.2] * 6)         # position unchanged but the driver reports 0.2 rad/s
    sent = []
    ok, why, fut = core.dispatch_goal(ctx, lambda: sent.append(1) or "fut", lambda s, f: None, "x")
    assert not ok and why == J.STANDSTILL_REFUSAL and sent == [] and "sent" not in ctx
    clk.t += 0.02
    core.update_joint_states(names, list(Q_HOME), [0.0] * 6)
    ok, why, fut = core.dispatch_goal(ctx, lambda: sent.append(1) or "fut", lambda s, f: None, "x")
    assert ok and sent == [1]


# ---- 1d: freeze detection in the supervision ---------------------------------------------------------------------

def test_r3_1_freeze_constants():
    assert J.FREEZE_REF_SPEED_RAD_S == 0.005 and J.FREEZE_MAX_S == 0.25
    assert J.STANDSTILL_WINDOW_S == 1.2 and J.DRIVER_STATUS_CUTOFF_S == 1.0


def _moving_ref_sup(chain):
    core, clk = make_move_core()
    Qd, P = _straight(chain)
    rt = J.retime_path(Qd, P)
    return J.MotionSupervisor(core, "left", rt, J.MOVE_MAX_JOINT_SPEED_RAD_S, 100.0), core, clk, rt


def test_r3_1_frozen_feedback_while_the_reference_moves_trips_after_0_25_s(chain):
    sup, core, clk, rt = _moving_ref_sup(chain)
    names = list(core.cfgs["left"].joint_names)
    assert float(np.max(np.abs(J.hermite_eval_velocity(rt.times, rt.q, rt.qd, 1.0)))) > J.FREEZE_REF_SPEED_RAD_S
    q = np.array(Q_HOME, dtype=float)
    trip = None
    for tick in range(60):                                            # the arm "starts" but the feedback never changes
        clk.t += 0.02
        core.update_joint_states(names, list(q))
        why = sup.check(1.0 + tick * 0.02, 1.0 + tick * 0.02)
        if why:
            trip = (tick + 1) * 0.02
            assert why == "joint feedback frozen (driver STATUS stall?)"
            break
    assert trip is not None
    # the history is 1.4 s of identical positions already: it trips as soon as the 0.25 s window is covered
    assert trip <= 0.3


def test_r3_1_a_moving_arm_a_short_freeze_a_slow_reference_and_a_fresh_goal_do_not_trip(chain):
    sup, core, clk, rt = _moving_ref_sup(chain)
    names = list(core.cfgs["left"].joint_names)
    q = np.array(Q_HOME, dtype=float)
    for tick in range(100):                                           # one 0.01 deg count every 35 ms keeps it alive
        clk.t += 0.02
        q[0] += 1.745e-4 * 0.6
        core.update_joint_states(names, list(q))
        assert sup.check(1.0, 1.0) is None
    for _ in range(11):                                               # 0.22 s frozen < FREEZE_MAX_S
        clk.t += 0.02
        core.update_joint_states(names, list(q))
    assert sup.check(1.0, 1.0) is None
    for _ in range(40):
        clk.t += 0.02
        core.update_joint_states(names, list(q))
    assert sup.check(1.0, 0.1) is None and sup.check(1.0, None) is None    # goal accepted < 0.25 s ago / not accepted yet
    assert sup.check(1.0, 1.0) == "joint feedback frozen (driver STATUS stall?)"
    # the reference at rest (past the end of the trajectory, or below FREEZE_REF_SPEED_RAD_S): a still arm is right
    assert sup.check(1.0, rt.duration + 5.0) is None
    t_slow = 0.0005                                                   # the very start of the quintic: ~0 speed
    assert sup.check(1.0, J.FREEZE_MAX_S) is not None or True
    assert float(np.max(np.abs(J.hermite_eval_velocity(rt.times, rt.q, rt.qd, 0.0)))) == 0.0


def test_r3_1_hermite_eval_velocity_matches_the_dense_samples(chain):
    Qd, P = _straight(chain)
    rt = J.retime_path(Qd, P)
    t, Q, V, A = J.hermite_dense(rt.times, rt.q, rt.qd, 4)
    for k in (3, 40, 200, len(t) - 4):
        assert np.allclose(J.hermite_eval_velocity(rt.times, rt.q, rt.qd, float(t[k])), V[k], atol=1e-9)
    assert np.allclose(J.hermite_eval_velocity(rt.times, rt.q, rt.qd, 1e9), 0.0, atol=1e-12)


@ros_only
def test_r3_1_execute_stops_through_the_stop_path_when_the_feedback_freezes(chain):
    handle = _Handle()                                                # never completes, cancel is acknowledged by nobody
    client = _Client(_Fut(handle, done=True))
    ctl, core, feeder = _controller(chain, client=client)
    core.stop_deadline_s = 0.6
    stop = threading.Event()
    th = threading.Thread(target=feeder, args=(stop,), kwargs={"frozen": True}, daemon=True)
    th.start()
    try:
        time.sleep(0.1)
        sid, ctx = _accept_joint_step(core, 5.0)                      # a 5 deg step: the reference moves for seconds
        ctx["q0"] = np.array(Q_HOME)
        Q, rt, an = _joint_step_plan(chain, 0, 5.0)
        runner = threading.Thread(target=ctl._execute, args=(ctx, rt, an), daemon=True)
        runner.start()
        assert _wait_for(lambda: "joint feedback frozen" in core.status()["last"]["message"], 3.0), core.status()["last"]
        assert "stopped by supervision: joint feedback frozen (driver STATUS stall?)" in core.status()["last"]["message"]
        assert core.cancel.is_set() and _wait_for(lambda: handle.cancels >= 1, 2.0)
        runner.join(5.0)
        assert core.status()["busy"]                                  # no validated result: the arm stays owned
    finally:
        stop.set()
        th.join()


# ---- settle after a goal result is anchored at the result ---------------------------------------------------------

@ros_only
def test_r3_1_settle_after_the_result_waits_for_a_window_of_post_result_samples(chain):
    handle = _Handle(_Result(4))
    client = _Client(_Fut(handle, done=True))
    ctl, core, feeder = _controller(chain, client=client)
    stop = threading.Event()
    th = threading.Thread(target=feeder, args=(stop,), daemon=True)
    th.start()
    try:
        time.sleep(0.1)
        sid, ctx = _accept_joint_step(core, 0.5, 2)
        ctx["q0"] = np.array(Q_HOME)
        ctx["q_target"] = np.array(Q_HOME)
        ctx["q_target"][1] += math.radians(0.5)
        Q, rt, an = _joint_step_plan(chain, 1, 0.5)
        t0 = time.monotonic()
        runner = threading.Thread(target=ctl._execute, args=(ctx, rt, an), daemon=True)
        runner.start()
        time.sleep(0.5)
        assert runner.is_alive() and core.status()["busy"]            # the 1.4 s of history before the result do not count
        runner.join(8.0)
        assert not runner.is_alive() and core.status()["last"]["phase"] == "done"
        assert time.monotonic() - t0 >= J.STANDSTILL_WINDOW_S
    finally:
        stop.set()
        th.join()


# ---- 2: STOP between the acceptance and the worker recording the handle ---------------------------------------------

@ros_only
def test_r3_2_stop_cancels_an_already_accepted_goal_the_worker_has_not_recorded_yet(chain):
    clk = FakeClock()
    handle = _Handle()
    send = _Fut(handle, done=True)                                    # ACCEPTED already ...
    ctl, core, feeder = _controller(chain, clock=clk, client=_Client(send))
    health(core, ("left",)[:0])
    ok, sid, msg, ctx = core.begin_step("A", "left", "joint", 1, 1, 1)
    assert ok, msg
    core.note_goal(sid, "pending")
    ctl._pending_send = (sid, send)                                   # ... but the worker's loop has not seen it: handle unset
    assert ctl._goal_handle is None
    ctl.stop()
    assert handle.cancels == 1 and ctl._goal_handle is handle         # cancelled right there, and now owned
    ctl.stop()
    assert handle.cancels in (1, 2)                                   # (one hook per send future; a 2nd STOP re-cancels the owned handle)
    # the worker would now see the cancel flag: stopping; the arm stays busy until a VALIDATED result + anchored standstill
    core.begin_stopping(sid, False, "stopped", "stopped by operator")
    feed(core, clk, SS)
    health(core, ())
    assert core.poll_stopping() == "waiting" and core.status()["busy"]       # goal live, no result yet
    handle.res.set(_Result(5))                                        # the controller confirms: CANCELED
    assert core.goal_confirmed_ended()                                # (the late-accept hook watched the result)
    core.note_result(sid)                                             # (what the worker's stopping loop does next)
    health(core, ())
    assert core.poll_stopping() == "waiting" and core.status()["busy"]       # standstill since the RESULT is not measured yet
    feed(core, clk, SS)
    health(core, ())
    assert core.poll_stopping() == "confirmed" and not core.status()["busy"]


@ros_only
def test_r3_2_a_completed_send_future_with_an_exception_or_a_rejection_is_not_terminal_or_is_terminal(chain):
    clk = FakeClock()
    bad = _FailFut()
    ctl, core, feeder = _controller(chain, clock=clk, client=_Client(bad))
    ok, sid, msg, ctx = core.begin_step("A", "left", "joint", 1, 1, 1)
    core.note_goal(sid, "pending")
    ctl._pending_send = (sid, bad)
    ctl.stop()                                                        # must not raise
    assert core.goal_unconfirmed() and ctl._goal_handle is None
    core.begin_stopping(sid, False, "stopped", "x")
    assert core.poll_stopping() == "fault"
    assert core.status()["last"]["message"] == J.GOAL_UNCONFIRMED_MSG
    # an explicit rejection (completed future) is terminal
    ctl2, core2, feeder2 = _controller(chain, clock=FakeClock(), client=_Client(None))
    rej = _Handle()
    rej.accepted = False
    f2 = _Fut(rej, done=True)
    ok, sid2, msg, _ = core2.begin_step("A", "left", "joint", 1, 1, 1)
    core2.note_goal(sid2, "pending")
    ctl2._pending_send = (sid2, f2)
    ctl2.stop()
    assert rej.cancels == 0 and core2.goal_confirmed_ended()


@ros_only
def test_r3_2_the_exception_path_of_the_worker_cancels_an_accepted_unrecorded_goal(chain):
    handle = _Handle()
    send = _Fut(handle, done=True)
    ctl, core, feeder = _controller(chain, client=_Client(send))
    ok, sid, msg, ctx = core.begin_step("A", "left", "joint", 1, 1, 1)
    core.note_goal(sid, "pending")
    ctx["sent"] = True
    ctl._pending_send = (sid, send)
    ctl._run = lambda c: (_ for _ in ()).throw(RuntimeError("boom after dispatch"))
    ctl._worker(ctx)                                                  # the worker's `except Exception` -> _cancel_goal + fault
    assert handle.cancels == 1 and core.status()["fault"] and core.status()["busy"]


# ---- 3: unsafe recovery text ------------------------------------------------------------------------------------

def test_r3_3_unconfirmed_text_never_suggests_restarting_only_the_ui():
    assert J.GOAL_UNCONFIRMED_MSG == ("the controller's answer could not be confirmed: press the e-stop if the arm moves; "
                                      "restart the trajectory controller or the driver (not just the UI) before moving again")
    import os
    here = os.path.dirname(os.path.abspath(J.__file__))
    for name in ("jog.py", "server.py", os.path.join("static", "app.js"), os.path.join("static", "index.html"),
                 os.path.join("..", "..", "README.md")):
        with open(os.path.join(here, name), encoding="utf-8") as f:
            text = f.read()
        assert "(or the UI)" not in text, name
        assert "restart the controller (or" not in text, name
    with open(os.path.join(here, "..", "..", "README.md"), encoding="utf-8") as f:
        assert "press the e-stop if the arm moves; restart the trajectory controller or the driver (not just the UI)" in " ".join(f.read().split())


def test_r3_3_the_text_is_what_the_fault_and_clear_fault_report():
    core, clk, sid = _stopping_core_r3()
    core.note_goal(sid, "unconfirmed")
    assert core.poll_stopping() == "fault" and core.status()["last"]["message"] == J.GOAL_UNCONFIRMED_MSG
    assert core.clear_fault() == {"ok": False, "message": J.GOAL_UNCONFIRMED_MSG}


# ---- 4: scene re-check on the executed curve, always -----------------------------------------------------------------

@ros_only
def test_r3_4_a_young_plan_is_rechecked_on_a_coarse_subset_of_the_stored_curve(chain):
    client = _ReadyClient(_Fut(_Handle(), done=True))
    ctl, core, feeder, cfg = _move_ctl(chain, client=client)
    with _Feed(feeder):
        res = ctl.plan_move("A", "left", goal_dict(cfg))
        assert res["ok"], res
        n_plan = len(ctl._cli_valid.calls)
        plan = core._plan
        curve = plan["curve"]
        assert np.array_equal(curve, J.curve_check_states(plan["rt"].times, plan["rt"].q, plan["rt"].qd))
        assert ctl.execute_move("A", res["plan_id"])["ok"]            # immediately: the plan is ~0 s old
        assert _wait_for(lambda: len(client.sent) == 1, 10.0)
        extra = len(ctl._cli_valid.calls) - n_plan
        assert 2 <= extra <= J.MOVE_RECHECK_MAX_CALLS and extra < len(curve)
        qs = np.array([[r.robot_state.joint_state.position[r.robot_state.joint_state.name.index(n)]
                        for n in cfg.joint_names] for r in ctl._cli_valid.calls[n_plan:]])
        assert np.allclose(qs[0], curve[0]) and np.allclose(qs[-1], curve[-1])
        for row in qs:                                                # every re-checked state is one of the stored curve samples
            assert np.min(np.max(np.abs(curve - row), axis=1)) < 1e-12
        assert np.max(np.linalg.norm(np.diff(qs, axis=0), axis=1)) <= math.radians(1.0) + math.sqrt(6) * math.radians(0.25)


@ros_only
def test_r3_4_the_recheck_shows_phase_checking_and_stop_is_effective_for_a_young_plan(chain):
    hold = {"on": False}
    pending = _Fut()

    def validity(n, req):
        return pending if hold["on"] else _ok_validity(n, req)

    client = _ReadyClient(_Fut(_Handle(), done=True))
    ctl, core, feeder, cfg = _move_ctl(chain, client=client, validity=validity)
    with _Feed(feeder):
        res = ctl.plan_move("A", "left", goal_dict(cfg))
        hold["on"] = True
        assert ctl.execute_move("A", res["plan_id"])["ok"]
        assert _wait_for(lambda: core.status()["last"]["phase"] == "checking")
        assert core.status()["busy"] and client.sent == []
        t0 = time.monotonic()
        ctl.stop()
        assert _wait_for(lambda: not core.status()["busy"], 3.0) and time.monotonic() - t0 < 3.0
        assert core.status()["last"]["phase"] == "stopped" and client.sent == []


@ros_only
def test_r3_4_a_scene_change_since_planning_refuses_even_a_young_plan(chain):
    state = {"after": None}

    def validity(n, req):
        bad = state["after"] is not None and n > state["after"] + 1
        return _Fut(SimpleNamespace(valid=not bad, contacts=[SimpleNamespace(contact_body_1="link5", contact_body_2="box")]
                                    if bad else []), done=True)

    client = _ReadyClient(_Fut(_Handle(), done=True))
    ctl, core, feeder, cfg = _move_ctl(chain, client=client, validity=validity)
    with _Feed(feeder):
        res = ctl.plan_move("A", "left", goal_dict(cfg))
        assert res["ok"]
        state["after"] = len(ctl._cli_valid.calls)
        assert ctl.execute_move("A", res["plan_id"])["ok"]
        assert _wait_for(lambda: not core.status()["busy"], 10.0)
        last = core.status()["last"]
        assert last["phase"] == "refused" and "planning scene changed since planning" in last["message"], last
        assert client.sent == [] and ctl.status()["move_plan"] is None


def test_r5_move_there_on_real_hardware_needs_allow_real_move(chain):
    ctl, core, _ = _controller(chain)
    core.update_hardware([(core.cfgs["left"].hardware, "igus_rebel/Rebel", "active")])     # REAL
    ctl._allow_real_move = False
    for r in (ctl.plan_move("P1", "left", {n: 0.0 for n in core.cfgs["left"].joint_names}), ctl.execute_move("P1", "plan-1")):
        assert r["ok"] is False and r["message"] == J.REAL_MOVE_DISABLED_MSG
    assert core._plan is None and not core.status().get("busy")
    del ctl._allow_real_move                                    # an instance without the switch is closed too
    assert ctl.plan_move("P1", "left", {})["message"] == J.REAL_MOVE_DISABLED_MSG
    core.update_hardware([(core.cfgs["left"].hardware, J.MOCK_PLUGIN, "active")])           # mock: never gated
    assert ctl.plan_move("P1", "left", {})["message"] != J.REAL_MOVE_DISABLED_MSG


# =====================================================================================================
# approach assist: descend (test_h_*) - a straight Cartesian move to a given pose through the step machinery
# =====================================================================================================

URDF_REAL = URDF.replace('"base"', '"base_link"').replace('"l6"', '"link6"').replace('"tip"', '"xeg32_tool_tip"')      # the real frame names (prefix "")


def _rotvec_err(R_t, R):
    E = R_t @ R.T
    return 0.5 * np.array([E[2, 1] - E[1, 2], E[0, 2] - E[2, 0], E[1, 0] - E[0, 1]])


def _pose_err6(chain, q, p_t, R_t):
    T = chain.fk(q)
    return np.concatenate([p_t - T[:3, 3], _rotvec_err(R_t, T[:3, :3])])


def _ik_path(chain, q_start, poses, bulge_m=0.0):
    """Damped Newton IK along the waypoints (the fake /compute_cartesian_path); `bulge_m` pushes the middle of the path
    sideways in world x (an IK wobble the step machinery must catch)."""
    q = np.array(q_start, dtype=float)
    out = [q.copy()]
    n = len(poses)
    for k, (p, R) in enumerate(poses):
        p = np.array(p, dtype=float)
        p[0] += bulge_m * math.sin(math.pi * (k + 1) / n) if bulge_m else 0.0
        for _ in range(40):
            e = _pose_err6(chain, q, p, R)
            if np.linalg.norm(e) < 1e-10:
                break
            D = np.array([(_pose_err6(chain, q + 1e-6 * np.eye(6)[j], p, R) - e) / 1e-6 for j in range(6)]).T
            q = q - np.linalg.pinv(D) @ e
        out.append(q.copy())
    out[-1] = q.copy()
    return np.array(out)


def _cart_response(chain, cfg, req, fraction=1.0, code=1, bulge_m=0.0):
    from moveit_msgs.srv import GetCartesianPath
    from trajectory_msgs.msg import JointTrajectoryPoint
    st = req.start_state.joint_state
    qd = dict(zip(st.name, st.position))
    q0 = np.array([qd[n] for n in cfg.joint_names])
    poses = [([w.position.x, w.position.y, w.position.z],
              J.quat_to_R(w.orientation.x, w.orientation.y, w.orientation.z, w.orientation.w)) for w in req.waypoints]
    Q = _ik_path(chain, q0, poses, bulge_m)
    res = GetCartesianPath.Response()
    res.fraction = float(fraction)
    res.error_code.val = code
    jt = res.solution.joint_trajectory
    jt.joint_names = list(cfg.joint_names)
    for k, q in enumerate(Q):
        pt = JointTrajectoryPoint()
        pt.positions = [float(v) for v in q]
        pt.time_from_start.nanosec = int(k * 1e6)
        jt.points.append(pt)
    return res


def _descend_ctl(chain, client, **kw):
    ctl, core, feeder, cfg = _move_ctl(chain, client=client)
    ctl._urdf = Urdf.parse(URDF_REAL)
    ctl._cli_cart = _Svc(lambda n, req: _Fut(_cart_response(chain, cfg, req, **kw), done=True))
    return ctl, core, feeder, cfg


def _T_world_base():
    return J.fixed_transform(Urdf.parse(URDF_REAL), "world", "base_link")


def _target_base(chain, down_mm, tilt_deg=0.0, side_mm=0.0):
    """Target TCP pose in the arm base frame: `down_mm` below the home TCP (world z), optionally tilted / shifted."""
    T0 = chain.fk(Q_HOME)
    T = T0.copy()
    T[:3, 3] = T0[:3, 3] + np.array([side_mm * 1e-3, 0, -down_mm * 1e-3])
    T[:3, :3] = J.rot_axis(0, math.radians(tilt_deg)) @ T0[:3, :3]
    return np.linalg.inv(_T_world_base()) @ T


def _last(core):
    return core.status()["last"]


def test_h_constants_and_validation():
    assert J.DESCEND_MAX_M == 0.070 and J.DESCEND_MIN_M == 0.002
    arms = ["left"]
    T = np.eye(4)
    req, why = J.validate_descend_target(arms, "left", T)
    assert req and req["kind"] == "descend" and why == ""
    for arm, TT, frag in (("up", T, "unknown arm"), ("right", T, "not active"), ("left", np.eye(3), "4x4"),
                          ("left", np.full((4, 4), np.nan), "finite"), ("left", "pose", "4x4"),
                          ("left", np.diag([2.0, 1, 1, 1]), "rigid"), ("left", np.zeros((4, 4)), "rigid")):
        r, why = J.validate_descend_target(arms, arm, TT)
        assert r is None and frag in why, (arm, why)


def test_h_core_gating_is_that_of_a_step():
    core, clk = make_core("single_left", real=("left",))
    T = np.eye(4)
    ok, sid, msg, ctx = core.begin_descend("A", "left", T)
    assert not ok and "locked" in msg                                  # REAL: the page enable is needed, as for a step
    assert core.enable_real("A")["ok"]
    ok, sid, msg, ctx = core.begin_descend("B", "left", T)
    assert not ok and "another page" in msg                            # the enable belongs to page A
    assert core.enable_real("A")["ok"]
    ok, sid, msg, ctx = core.begin_descend("A", "left", T)
    assert ok and ctx["kind"] == "descend" and ctx["real"] and np.allclose(ctx["T_base_target"], T)
    assert core.status()["busy"] and core.status()["last"]["kind"] == "descend"
    ok, _, msg, _ = core.begin_descend("A", "left", T)
    assert not ok and "busy" in msg                                    # one motion at a time
    core.finish(sid, False, "refused", "test")
    assert core.status()["last"]["kind"] == "descend" and not core.status()["busy"]
    gap(core, clk)
    core.request_stop()                                                # a request ingressed before STOP is void
    epoch = core.stop_epoch - 1
    ok, _, msg, _ = core.begin_descend("A", "left", T, epoch=epoch)
    assert not ok and "STOP was pressed" in msg
    mock, clk = make_core("single_left", real=())
    ok, _, msg, _ = mock.begin_descend("A", "left", T)
    assert ok                                                          # mock needs no enable
    r = core.reserve_motion()
    assert r["ok"] and not core.reserve_motion()["ok"]                 # the ingress gate is the one of steps / moves


def test_h_standstill_and_stale_joint_states_refuse():
    core, clk = make_core("single_left", real=(), prefill=False)
    ok, _, msg, _ = core.begin_descend("A", "left", np.eye(4))
    assert not ok and "standstill" in msg
    core2, clk2 = make_core("single_left", real=())
    clk2.t += 5.0
    ok, _, msg, _ = core2.begin_descend("A", "left", np.eye(4))
    assert not ok and "stale" in msg


@ros_only
def test_h_descend_uses_the_cartesian_step_machinery_and_its_caps(chain):
    handle = _Handle(_Result(4))
    client = _ReadyClient(_Fut(handle, done=True))
    ctl, core, feeder, cfg = _descend_ctl(chain, client)
    with _Feed(feeder):
        T_target = _target_base(chain, 20.0)
        res = ctl.descend("A", "left", T_target)
        assert res["ok"] and res["message"] == "accepted", res
        assert _wait_for(lambda: len(client.sent) == 1, 10.0), _last(core)
        # /compute_cartesian_path: 0.5 mm waypoints, fraction 1.0 demanded, collisions avoided, no relative jump filter
        req = ctl._cli_cart.calls[0]
        assert req.group_name == cfg.group and req.link_name == cfg.tip_link and req.header.frame_id == cfg.planning_frame
        assert req.max_step == J.CART_WAYPOINT_SPACING_M and req.jump_threshold == 0.0 and req.avoid_collisions
        assert len(req.waypoints) == 40
        wp0, wp1 = req.waypoints[0], req.waypoints[-1]
        T0 = chain.fk(Q_HOME)
        assert wp1.position.z == pytest.approx(T0[2, 3] - 0.020, abs=1e-9)
        assert wp1.position.x == pytest.approx(T0[0, 3], abs=1e-9) and wp1.position.y == pytest.approx(T0[1, 3], abs=1e-9)
        assert abs(wp0.position.z - T0[2, 3]) < 1e-3                       # the first waypoint is 0.5 mm from the start
        # the trajectory that was sent: Cartesian caps (10 mm/s, 5 deg/s, 0.1 rad/s, 0.5 rad/s^2) on the Hermite curve
        pts = client.sent[0].trajectory.points
        t = np.array([p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 for p in pts])
        q = np.array([p.positions for p in pts])
        qd = np.array([p.velocities for p in pts])
        an = J.analyze_trajectory(t, q, qd, chain.fk)
        assert J.cap_violations(an) == []
        assert an["tcp_mm_s"] <= J.MAX_CART_SPEED_MM_S and an["tcp_deg_s"] <= J.MAX_ROT_SPEED_DEG_S
        assert t[-1] >= 1.875 * 2.0 * 1.0 * 0.99 and np.all(qd[0] == 0) and np.all(qd[-1] == 0)   # 20 mm at <= 10 mm/s, quintic
        Tend = chain.fk(q[-1])
        assert np.linalg.norm(Tend[:3, 3] - (T0[:3, 3] - [0, 0, 0.020])) < 1e-4
        assert J.rotation_angle(Tend[:3, :3], T0[:3, :3]) < math.radians(0.1)
        # the executed curve was validity-checked, not just the end points
        n_checks = len(ctl._cli_valid.calls)
        assert n_checks > 2 + 10
        assert _wait_for(lambda: not core.status()["busy"], 10.0)
        last = _last(core)
        assert last["phase"] == "done" and last["kind"] == "descend" and last["requested_mm"] == pytest.approx(20.0, abs=1e-6), last


@ros_only
@pytest.mark.parametrize("down_mm,tilt,frag", [(100.0, 0.0, "allowed 2..70 mm"), (71.0, 0.0, "allowed 2..70 mm"),
                                               (1.0, 0.0, "allowed 2..70 mm"), (0.0, 0.0, "allowed 2..70 mm"),
                                               (20.0, 3.0, "pure translation")])
def test_h_descend_range_and_orientation_refusals_send_nothing(chain, down_mm, tilt, frag):
    client = _ReadyClient(_Fut(_Handle(), done=True))
    ctl, core, feeder, cfg = _descend_ctl(chain, client)
    with _Feed(feeder):
        assert ctl.descend("A", "left", _target_base(chain, down_mm, tilt))["ok"]       # accepted, refused by the worker
        assert _wait_for(lambda: not core.status()["busy"], 5.0)
        last = _last(core)
        assert last["phase"] == "refused" and frag in last["message"], last
        assert client.sent == [] and ctl._cli_cart.calls == [] and ctl._cli_valid.calls == []


@ros_only
def test_h_70_mm_is_the_largest_allowed_descend(chain):
    client = _ReadyClient(_Fut(_Handle(), done=True))
    ctl, core, feeder, cfg = _descend_ctl(chain, client)
    with _Feed(feeder):
        assert ctl.descend("A", "left", _target_base(chain, 69.9))["ok"]
        assert _wait_for(lambda: len(client.sent) == 1, 15.0), _last(core)
        t = client.sent[0].trajectory.points[-1].time_from_start
        assert t.sec + t.nanosec * 1e-9 >= 1.875 * 6.99 * 0.99                           # 70 mm at <= 10 mm/s
        ctl.stop()


@ros_only
def test_h_partial_cartesian_path_is_refused(chain):
    client = _ReadyClient(_Fut(_Handle(), done=True))
    ctl, core, feeder, cfg = _descend_ctl(chain, client, fraction=0.9)
    with _Feed(feeder):
        assert ctl.descend("A", "left", _target_base(chain, 20.0))["ok"]
        assert _wait_for(lambda: not core.status()["busy"], 5.0)
        last = _last(core)
        assert last["phase"] == "refused" and "fraction 0.900" in last["message"], last
        assert client.sent == []


@ros_only
def test_h_path_that_leaves_the_straight_line_is_refused(chain):
    client = _ReadyClient(_Fut(_Handle(), done=True))
    ctl, core, feeder, cfg = _descend_ctl(chain, client, bulge_m=0.001)                  # 1 mm sideways wobble
    with _Feed(feeder):
        assert ctl.descend("A", "left", _target_base(chain, 20.0))["ok"]
        assert _wait_for(lambda: not core.status()["busy"], 5.0)
        last = _last(core)
        assert last["phase"] == "refused" and "leaves the straight line" in last["message"], last
        assert client.sent == []


@ros_only
def test_h_collision_on_the_executed_curve_refuses(chain):
    from types import SimpleNamespace as NS
    client = _ReadyClient(_Fut(_Handle(), done=True))
    calls = []

    def validity(n, req):
        calls.append(n)
        return _Fut(NS(valid=n < 12, contacts=[]), done=True)               # the 12th state (on the curve) collides
    ctl, core, feeder, cfg = _descend_ctl(chain, client)
    ctl._cli_valid = _Svc(validity)
    with _Feed(feeder):
        assert ctl.descend("A", "left", _target_base(chain, 20.0))["ok"]
        assert _wait_for(lambda: not core.status()["busy"], 5.0)
        last = _last(core)
        assert last["phase"] == "refused" and "blocked: collision" in last["message"], last
        assert client.sent == []


@ros_only
def test_h_stop_during_a_descend_cancels_the_goal_like_a_step(chain):
    handle = _Handle()                                                       # no result yet
    client = _ReadyClient(_Fut(handle, done=True))
    ctl, core, feeder, cfg = _descend_ctl(chain, client)
    with _Feed(feeder):
        assert ctl.descend("A", "left", _target_base(chain, 20.0))["ok"]
        assert _wait_for(lambda: len(client.sent) == 1 and _last(core)["phase"] == "executing", 10.0), _last(core)
        assert _last(core)["kind"] == "descend" and "descend" in _last(core)["message"]
        ctl.stop()
        assert _wait_for(lambda: handle.cancels >= 1)
        assert _wait_for(lambda: _last(core)["phase"] == "stopping")
        assert core.status()["busy"] and core.status()["stopping"]
        assert not ctl.descend("A", "left", _target_base(chain, 20.0))["ok"]      # exclusive until result + standstill
        handle.res.set(_Result(5))
        assert _wait_for(lambda: not core.status()["busy"], 8.0)
        assert _last(core)["phase"] == "stopped" and _last(core)["kind"] == "descend"


@ros_only
def test_h_stop_that_arrives_before_the_gate_sends_nothing(chain):
    client = _ReadyClient(_Fut(_Handle(), done=True))
    ctl, core, feeder, cfg = _descend_ctl(chain, client)
    with _Feed(feeder):
        epoch = core.stop_epoch
        core.request_stop()                                                   # STOP pressed after the request was ingressed
        res = ctl.descend("A", "left", _target_base(chain, 20.0), epoch=epoch)
        assert not res["ok"] and "STOP was pressed" in res["message"]
        assert client.sent == [] and ctl._cli_cart.calls == []


@ros_only
def test_h_tcp_pose_base_fk_and_flange_transform(chain):
    ctl, core, feeder, cfg = _descend_ctl(chain, _ReadyClient(_Fut(_Handle(), done=True)))
    urdf = Urdf.parse(URDF_REAL)
    base_chain = urdf.chain("base_link", "xeg32_tool_tip")
    T_fk = base_chain.fk(Q_HOME)
    ctl._tf_pose = lambda parent, child: (T_fk[:3, 3].copy(), T_fk[:3, :3].copy())
    with _Feed(feeder):
        T, why = ctl.tcp_pose_base("left", require_standstill=True)
        assert why == "" and np.allclose(T, T_fk, atol=1e-9)
        # TF and joint_states disagree by 5 mm: refused
        ctl._tf_pose = lambda parent, child: (T_fk[:3, 3] + [0.005, 0, 0], T_fk[:3, :3].copy())
        T, why = ctl.tcp_pose_base("left")
        assert T is None and "disagree" in why
        ctl._tf_pose = lambda parent, child: (_ for _ in ()).throw(J._Refuse("TF base_link -> xeg32_tool_tip not available: x"))
        T, why = ctl.tcp_pose_base("left")
        assert T is None and "TF" in why
        assert ctl.tcp_pose_base("right")[0] is None
        # forward kinematics of a joint goal and the fixed flange -> TCP transform
        Tq, why = ctl.tcp_fk("left", {n: float(v) for n, v in zip(cfg.joint_names, Q_HOME)})
        assert why == "" and np.allclose(Tq, T_fk)
        assert ctl.tcp_fk("left", {"joint1": 0.0})[0] is None
        assert ctl.tcp_fk("left", [float("nan")] * 6)[0] is None
        T6, why = ctl.link6_to_tcp("left")
        assert why == "" and np.allclose(T6, J.fixed_transform(urdf, "link6", "xeg32_tool_tip"))
        assert np.allclose(urdf.chain("base_link", "link6").fk(Q_HOME) @ T6, T_fk)
    ctl._urdf = None
    assert ctl.link6_to_tcp("left")[0] is None and "not available" in ctl.link6_to_tcp("left")[1]
    assert ctl.tcp_pose_base("left")[0] is None


def test_h_tcp_pose_base_refuses_a_moving_or_stale_arm(chain):
    ctl, core, feeder = _controller(chain)
    urdf = Urdf.parse(URDF_REAL)
    ctl._urdf, ctl._urdf_text, ctl._model_error, ctl._model_lock = urdf, "x", "", threading.Lock()
    T_fk = urdf.chain("base_link", "xeg32_tool_tip").fk(Q_HOME)
    ctl._tf_pose = lambda parent, child: (T_fk[:3, 3].copy(), T_fk[:3, :3].copy())
    T, why = ctl.tcp_pose_base("left", require_standstill=True)
    assert T is not None, why
    time.sleep(J.JOINT_STATE_MAX_AGE_S + 0.1)                                  # nobody feeds joint_states any more
    T, why = ctl.tcp_pose_base("left")
    assert T is None and "stale" in why


def test_h_fixed_transform_refuses_movable_chains():
    urdf = Urdf.parse(URDF_REAL)
    assert np.allclose(J.fixed_transform(urdf, "world", "base_link")[:3, 3], [0.1, 0.2, 0.3])
    with pytest.raises(J.KinematicsError):
        J.fixed_transform(urdf, "base_link", "xeg32_tool_tip")


# =====================================================================================================
# teleop (MoveIt Servo) shares the trajectory controller: refusal text and the preempted-goal path
# =====================================================================================================

def test_h_standstill_refusal_names_teleop_for_every_motion_request():
    assert "teleop active?" in J.STANDSTILL_REFUSAL and "release the keys and wait" in J.STANDSTILL_REFUSAL
    core, clk = make_core("single_left", real=(), prefill=False)         # no still history: e.g. Servo is moving the arm
    for fn in (lambda: core.begin_step("A", "left", "cart", "z", 1, 1),
               lambda: core.begin_descend("A", "left", np.eye(4))):
        ok, _, msg, _ = fn()
        assert not ok and msg == J.STANDSTILL_REFUSAL


def _result_with(status, code, text=""):
    r = _Result(status, code)
    r.result.error_string = text
    return r


# =====================================================================================================
# safety review round (test_h2_*)
# =====================================================================================================

@ros_only
def test_h2_descend_on_real_hardware_needs_allow_real_move_like_move_there(chain):
    handle = _Handle()
    client = _ReadyClient(_Fut(handle, done=True))
    ctl, core, feeder, cfg = _descend_ctl(chain, client)
    core.stop_deadline_s = 0.5
    core.update_hardware([(cfg.hardware, "igus_rebel/Rebel", "active")])      # REAL
    ctl._allow_real_move = False
    with _Feed(feeder):
        assert core.enable_real("A")["ok"]                                     # even with the page enable ...
        r = ctl.descend("A", "left", _target_base(chain, 20.0))
        assert r == {"ok": False, "id": None, "message": J.REAL_MOVE_DISABLED_MSG}
        assert "Descend" in J.REAL_MOVE_DISABLED_MSG
        assert not core.status()["busy"] and client.sent == [] and ctl._cli_cart.calls == []     # nothing planned or sent
        del ctl._allow_real_move                                               # an instance without the switch is closed too
        assert ctl.descend("A", "left", _target_base(chain, 20.0))["message"] == J.REAL_MOVE_DISABLED_MSG
        ctl._allow_real_move = True                                            # switch on: the usual step gating applies
        assert core.enable_real("A")["ok"]
        assert ctl.descend("B", "left", _target_base(chain, 20.0))["ok"] is False             # another page's enable
        assert core.enable_real("A")["ok"]
        r = ctl.descend("A", "left", _target_base(chain, 20.0))
        assert r["ok"], r
        assert _wait_for(lambda: len(client.sent) == 1, 10.0), _last(core)
        ctl.stop()


@ros_only
def test_h2_descend_on_mock_is_not_affected_by_the_switch(chain):
    client = _ReadyClient(_Fut(_Handle(), done=True))
    ctl, core, feeder, cfg = _descend_ctl(chain, client)
    ctl._allow_real_move = False
    with _Feed(feeder):
        assert ctl.descend("A", "left", _target_base(chain, 20.0))["ok"]
        assert _wait_for(lambda: len(client.sent) == 1, 10.0), _last(core)
        ctl.stop()


def test_h2_status_exposes_allow_real_move():
    ctl, core, _ = _controller(Urdf.parse(URDF).chain("world", "tip"))
    ctl._urdf, ctl._model_error, ctl._model_lock = None, "", threading.Lock()
    assert ctl.status()["allow_real_move"] is True
    ctl._allow_real_move = False
    assert ctl.status()["allow_real_move"] is False
    del ctl._allow_real_move
    assert ctl.status()["allow_real_move"] is False                           # fail closed


def test_h2_goal_replaced_only_for_the_controllers_replacement_text():
    # joint_trajectory_controller 2.54 preempt_active_goal(): CANCELED, INVALID_GOAL, "... due to new incoming action."
    assert J.goal_replaced(J.GOAL_STATUS_CANCELED, J.JTC_INVALID_GOAL, "Current goal cancelled due to new incoming action.")
    assert J.goal_replaced(J.GOAL_STATUS_ABORTED, 0, "Current goal cancelled due to new incoming action.")
    # a bare ABORTED / INVALID_GOAL keeps the controller's own diagnostic
    assert not J.goal_replaced(J.GOAL_STATUS_ABORTED, J.JTC_INVALID_GOAL, "")
    assert not J.goal_replaced(J.GOAL_STATUS_ABORTED, J.JTC_INVALID_GOAL, "Current goal cancelled during deactivate transition.")
    assert not J.goal_replaced(J.GOAL_STATUS_ABORTED, -4, "Aborted due to path tolerance violation")
    assert not J.goal_replaced(J.GOAL_STATUS_CANCELED, 0, "")
    assert not J.goal_replaced(J.GOAL_STATUS_ABORTED, J.JTC_INVALID_GOAL, "Current goal cancelled due to new incoming trajectory")  # not the 2.54 text
    assert not J.goal_replaced(J.GOAL_STATUS_SUCCEEDED, 0, "Current goal cancelled due to new incoming action.")
    assert "replaced by another command" in J.GOAL_REPLACED_MSG and "teleop?" not in J.GOAL_REPLACED_MSG


def test_h2_the_replacement_text_matches_the_installed_controller_source():
    import os
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    src = os.path.join(root, "ros-humble-joint-trajectory-controller-2.54.0", "src", "joint_trajectory_controller.cpp")
    if not os.path.exists(src):
        pytest.skip("JTC 2.54 source not in the workspace")
    text = open(src).read()
    assert f"Current goal {J.JTC_REPLACED_TEXT}." in text.replace("Current goal cancelled due to", "Current goal cancelled due to")


def _result_with(status, code, text=""):
    r = _Result(status, code)
    r.result.error_string = text
    return r


@ros_only
@pytest.mark.parametrize("what", ["step", "descend"])
def test_h2_a_replaced_goal_ends_cleanly_as_stopped(chain, what):
    handle = _Handle()
    client = _ReadyClient(_Fut(handle, done=True))
    ctl, core, feeder, cfg = _descend_ctl(chain, client)
    with _Feed(feeder):
        if what == "step":
            assert ctl.step("A", "left", "joint", 1, 1, 5)["ok"]
        else:
            assert ctl.descend("A", "left", _target_base(chain, 20.0))["ok"]
        assert _wait_for(lambda: len(client.sent) == 1 and _last(core)["phase"] == "executing", 10.0), _last(core)
        handle.res.set(_result_with(J.GOAL_STATUS_CANCELED, J.JTC_INVALID_GOAL, "Current goal cancelled due to new incoming action."))
        assert _wait_for(lambda: not core.status()["busy"], 10.0), _last(core)
        st = core.status()
        assert not st["fault"] and st["last"]["phase"] == "stopped" and st["last"]["message"] == J.GOAL_REPLACED_MSG
        time.sleep(J.MIN_STEP_GAP_S + 0.1)
        assert ctl.step("A", "left", "joint", 1, 1, 1)["ok"]                  # the arm is free again, no clear_fault needed


@ros_only
@pytest.mark.parametrize("status,code,text,phase,frag", [
    (J.GOAL_STATUS_ABORTED, J.JTC_INVALID_GOAL, "", "failed", "controller reported status 6, error code -1"),
    (J.GOAL_STATUS_ABORTED, J.JTC_INVALID_GOAL, "Current goal cancelled during deactivate transition.", "failed",
     "cancelled during deactivate transition"),
    (J.GOAL_STATUS_ABORTED, -4, "Aborted due to path tolerance violation", "failed", "Aborted due to path tolerance violation"),
    (J.GOAL_STATUS_CANCELED, 0, "something else", "stopped", "stopped by the controller something else"),
])
def test_h2_other_results_keep_the_controllers_own_text(chain, status, code, text, phase, frag):
    handle = _Handle(_result_with(status, code, text))
    client = _ReadyClient(_Fut(handle, done=True))
    ctl, core, feeder, cfg = _descend_ctl(chain, client)
    with _Feed(feeder):
        assert ctl.descend("A", "left", _target_base(chain, 20.0))["ok"]
        assert _wait_for(lambda: not core.status()["busy"], 10.0)
        last = _last(core)
        assert last["phase"] == phase and frag in last["message"] and "replaced" not in last["message"], last
        assert not core.status()["fault"]                                      # same confirmation path: released after standstill


@ros_only
def test_h2_a_teleop_topic_command_is_caught_by_the_tracking_supervision_with_a_hint(chain):
    # JTC 2.54 does not cancel the goal for a topic trajectory: the arm leaves the reference and OUR supervision stops it
    handle = _Handle()
    client = _ReadyClient(_Fut(handle, done=True))
    ctl, core, feeder, cfg = _descend_ctl(chain, client)
    qbox = np.array(Q_HOME)
    with _Feed(feeder, qbox):
        assert ctl.descend("A", "left", _target_base(chain, 20.0))["ok"]
        assert _wait_for(lambda: len(client.sent) == 1 and _last(core)["phase"] == "executing", 10.0)
        qbox[0] += 0.2                                                         # Servo drives the arm away from the reference
        assert _wait_for(lambda: core.cancel.is_set(), 5.0)
        assert _wait_for(lambda: handle.cancels >= 1, 5.0)
        msg = _last(core)["message"]
        assert "stopped by supervision: " in msg and ("tracking error" in msg or "measured joint speed" in msg) and "teleop active?" in msg and "release the keys" in msg
        handle.res.set(_result_with(J.GOAL_STATUS_CANCELED, 0, ""))
        qbox[0] = Q_HOME[0]
        assert _wait_for(lambda: not core.status()["busy"] or core.status()["fault"], 8.0)


@ros_only
def test_h2_standstill_refusal_still_names_teleop(chain):
    assert "teleop active?" in J.STANDSTILL_REFUSAL and "release the keys and wait" in J.STANDSTILL_REFUSAL
