import math

import pytest

from rebel_mocap_follower.follower_core import (
    CATCHUP, DISABLED, ENABLED, HOLD, REFUSED, FollowerCore, Params)

NAMES = [f"joint{i}" for i in range(1, 7)]
LO = [-3.0] * 6
HI = [3.0] * 6
DT = 0.01


def make(**kw):
    return FollowerCore(NAMES, LO, HI, Params(**kw))


class Sim:
    """Core + ideal double-integrator-free plant: position integrates commanded velocity."""

    def __init__(self, core, start=None):
        self.core = core
        self.t = 100.0
        self.q = list(start or [0.0] * 6)
        self.enable = True
        self.last_v = [0.0] * 6
        self.status = None

    def send(self, target, names=None, offset=0.0):
        return self.core.on_input(self.t, names or NAMES, target, offset)

    def step(self, target=None, blocked=None):
        if target is not None:
            self.send(target)
        v, self.status = self.core.update(self.t, self.q, 0.0, self.enable, blocked)
        self.q = [q + vi * DT for q, vi in zip(self.q, v)]
        self.last_v = v
        self.t += DT
        return v

    def run(self, seconds, target=None, refresh=True):
        for _ in range(int(seconds / DT)):
            self.step(target if refresh else None)


def test_param_caps_refused():
    for kw in ({"max_velocity": 0.61}, {"max_acceleration": 1.6}, {"stop_deceleration": 2.0},
               {"max_velocity": 0.0}, {"kp": -1.0}, {"input_timeout_s": float("nan")}):
        with pytest.raises(ValueError):
            make(**kw)
    make(max_velocity=0.6, max_acceleration=1.5, stop_deceleration=1.5)


def test_limit_margin_reject_when_clamp_disabled():
    sim = Sim(make(clamp_to_limits=False))
    tgt = [0.0] * 6
    tgt[2] = 2.96  # inside 3.0 but within 0.05 margin
    assert "joint3" in sim.send(tgt)
    sim.step()
    sim.step()
    assert sim.status.state != ENABLED
    assert all(v == 0.0 for v in sim.last_v)


def test_target_beyond_limit_is_clamped_and_followed():
    core = make()
    sim = Sim(core, start=[0.0, -2.9, 0.0, 0.0, 0.0, 0.0])
    tgt = [0.0] * 6
    tgt[1] = -3.02  # beyond the hard limit -3.0
    assert sim.send(tgt) is None
    assert core.clamped == ["joint2"]
    assert core.target[1] == pytest.approx(-2.95)
    for _ in range(600):
        sim.send(tgt)
        sim.step()
    assert sim.status.state == ENABLED
    assert sim.q[1] == pytest.approx(-2.95, abs=0.01)
    assert sim.q[1] >= -2.95 - 1e-6
    tgt[1] = -1.0  # back in range
    sim.send(tgt)
    assert core.clamped == []


def test_clamped_wrap_around_is_not_followed():
    core = make(max_velocity=0.3, catchup_velocity=0.3)
    sim = Sim(core, start=[0.0, 0.0, 0.0, 0.0, 0.0, 2.95])
    tgt = [0.0, 0.0, 0.0, 0.0, 0.0, 3.1]   # clamped to 2.95
    for _ in range(20):
        sim.send(tgt)
        sim.step()
    assert sim.status.state == ENABLED
    tgt[5] = -3.1                            # mocap wraps at pi -> clamped to -2.95: 5.9 rad away
    for _ in range(50):
        sim.send(tgt)
        sim.step()
    assert sim.status.state == HOLD and sim.status.latched
    assert abs(sim.q[5] - 2.95) < 0.05


def test_message_validation():
    core = make()
    good = [0.0] * 6
    assert core.on_input(1.0, NAMES, good) is None
    assert core.on_input(1.0, NAMES[:5], good[:5])  # missing
    assert core.on_input(1.0, NAMES[:5] + ["joint9"], good)  # unknown
    assert core.on_input(1.0, NAMES[:5] + ["joint1"], good)  # duplicate
    assert core.on_input(1.0, NAMES, good[:5])  # length mismatch
    bad = list(good)
    bad[3] = float("nan")
    assert core.on_input(1.0, NAMES, bad)
    bad[3] = float("inf")
    assert core.on_input(1.0, NAMES, bad)


def test_names_any_order():
    sim = Sim(make())
    order = [5, 3, 1, 0, 2, 4]
    names = [NAMES[i] for i in order]
    pos = [0.1 * (i + 1) for i in order]
    assert sim.send(pos, names) is None
    assert sim.core.target == pytest.approx([0.1 * (i + 1) for i in range(6)])


def test_stamp_check():
    core = make()
    assert core.on_input(1.0, NAMES, [0.0] * 6, 0.5) is None
    assert "stamp" in core.on_input(1.0, NAMES, [0.0] * 6, 1.5)
    assert "stamp" in core.on_input(1.0, NAMES, [0.0] * 6, -2.0)
    assert "stamp" in core.on_input(1.0, NAMES, [0.0] * 6, float("nan"))
    core2 = make(check_stamp=False)
    assert core2.on_input(1.0, NAMES, [0.0] * 6, 500.0) is None


def test_engage_refused_when_far_and_latched_until_reenable():
    sim = Sim(make())
    far = [0.0] * 6
    far[0] = 1.8
    sim.run(0.1, far)
    assert sim.status.state == REFUSED
    assert sim.q == [0.0] * 6
    # even if target becomes close it stays refused while enable is held
    sim.run(0.2, [0.0] * 6)
    assert sim.status.state == REFUSED and sim.q == [0.0] * 6
    sim.enable = False
    sim.run(0.1, [0.0] * 6)
    assert sim.status.state == DISABLED
    sim.enable = True
    sim.run(0.1, [0.0] * 6)
    assert sim.status.state == ENABLED


def test_engage_ok_within_limit_and_follows():
    sim = Sim(make())
    tgt = [0.3, 0, 0, 0, 0, -0.3]
    sim.run(0.05, tgt)
    assert sim.status.state == ENABLED
    sim.run(6.0, tgt)
    assert sim.q == pytest.approx(tgt, abs=0.003)


def test_waits_for_input_before_engaging():
    sim = Sim(make())
    sim.run(0.1)  # no input at all
    assert sim.status.state == HOLD and "waiting" in sim.status.reason
    sim.run(0.1, [0.0] * 6)
    assert sim.status.state == ENABLED


def test_step_enters_catchup_not_latch():
    sim = Sim(make())
    sim.run(0.3, [0.0] * 6)
    assert sim.status.state == ENABLED
    sim.step([0.5, 0, 0, 0, 0, 0])
    sim.step([0.5, 0, 0, 0, 0, 0])
    assert sim.status.state == CATCHUP and not sim.status.latched


def test_step_just_below_threshold_is_ok():
    sim = Sim(make())
    sim.run(0.1, [0.0] * 6)
    sim.step([0.19, 0, 0, 0, 0, 0])
    sim.step([0.19, 0, 0, 0, 0, 0])
    assert not sim.status.latched


def test_nan_holds_then_recovers_without_jump():
    sim = Sim(make())
    sim.run(0.3, [0.0] * 6)
    bad = [float("nan")] + [0.0] * 5
    sim.step(bad)
    assert sim.status.state == HOLD
    sim.step([0.0] * 6)
    assert sim.status.state == ENABLED


def test_nan_then_jump_enters_catchup():
    sim = Sim(make())
    sim.run(0.3, [0.0] * 6)
    sim.step([float("nan")] + [0.0] * 5)
    sim.step([0.5, 0, 0, 0, 0, 0])
    sim.step([0.5, 0, 0, 0, 0, 0])
    assert sim.status.state == CATCHUP and not sim.status.latched


def test_input_timeout_holds_and_stops():
    sim = Sim(make())
    tgt = [0.45, 0, 0, 0, 0, 0]
    sim.run(1.0, tgt)
    assert sim.status.state == ENABLED and max(map(abs, sim.last_v)) > 0.1
    sim.run(0.25, refresh=False)  # input stops
    assert sim.status.state == HOLD and "stale" in sim.status.reason
    sim.run(0.5, refresh=False)
    assert sim.last_v == [0.0] * 6


def test_enable_release_stops_within_decel():
    sim = Sim(make())
    sim.run(1.0, [0.45, 0, 0, 0, 0, 0])
    v0 = abs(sim.last_v[0])
    assert v0 == pytest.approx(0.2, abs=1e-6)
    sim.enable = False
    n = 0
    while sim.last_v[0] != 0.0 and n < 1000:
        sim.step([0.45, 0, 0, 0, 0, 0])
        n += 1
    assert n * DT == pytest.approx(v0 / 1.0, abs=0.03)
    assert sim.status.state == DISABLED


def test_joint_states_missing_holds():
    core = make()
    core.on_input(1.0, NAMES, [0.0] * 6)
    v, st = core.update(1.0, None, 0.0, True)
    assert st.state == HOLD
    v, st = core.update(1.01, [0.0] * 6, 5.0, True)  # stale joint_states
    assert st.state == HOLD
    v, st = core.update(1.02, [float("nan")] + [0.0] * 5, 0.0, True)
    assert st.state == HOLD


def test_blocked_holds_and_resumes():
    sim = Sim(make())
    sim.run(0.2, [0.0] * 6)
    sim.step([0.0] * 6, blocked="other publisher")
    assert sim.status.state == HOLD
    sim.step([0.0] * 6)
    assert sim.status.state == ENABLED


def test_velocity_and_accel_limits_respected():
    core = make()
    sim = Sim(core)
    prev = 0.0
    peak_v = peak_a = 0.0
    tgt = [0.4] * 6
    for _ in range(int(8 / DT)):
        sim.step(tgt)
        v = sim.last_v[0]
        peak_v = max(peak_v, abs(v))
        peak_a = max(peak_a, abs(v - prev) / DT)
        prev = v
    assert peak_v <= 0.2 + 1e-9
    assert peak_a <= 0.5 + 1e-6


@pytest.mark.parametrize("kp,vmax,amax,tau,tol", [
    (2.0, 0.2, 0.5, 0.1, 0.005),    # defaults, sluggish plant (100 ms lag)
    (2.0, 0.2, 0.5, 0.02, 0.001),
    (10.0, 0.6, 1.5, 0.02, 0.003),  # hard-cap settings, near-ideal plant
    (5.0, 0.3, 0.3, 0.02, 0.003),
])
def test_no_overshoot_double_integrator(kp, vmax, amax, tau, tol):
    """Plant: commanded velocity is realised through a first-order lag (tau)."""
    sim = Sim(make(kp=kp, max_velocity=vmax, max_acceleration=amax, stop_deceleration=amax))
    sim.q = [0.0] * 6
    target = [0.45, -0.45, 0.2, 0, 0, 0.45]
    v_real = [0.0] * 6
    max_overshoot = 0.0
    for _ in range(int(25 / DT)):
        sim.send(target)
        v, _ = sim.core.update(sim.t, sim.q, 0.0, True)
        v_real = [vr + (vc - vr) * DT / tau for vr, vc in zip(v_real, v)]
        sim.q = [q + vr * DT for q, vr in zip(sim.q, v_real)]
        sim.t += DT
        for q, tg in zip(sim.q, target):
            if tg != 0:
                max_overshoot = max(max_overshoot, (q - tg) * math.copysign(1, tg))
    assert max_overshoot < tol
    assert sim.q == pytest.approx(target, abs=0.004)


def test_reversing_target_respects_accel():
    sim = Sim(make())
    prev = 0.0
    peak_a = 0.0
    for k in range(int(10 / DT)):
        tgt = [0.45 * math.sin(2 * math.pi * 0.25 * k * DT)] + [0.0] * 5
        sim.step(tgt)
        peak_a = max(peak_a, abs(sim.last_v[0] - prev) / DT)
        prev = sim.last_v[0]
    assert peak_a <= 0.5 + 1e-6


def test_dual_arm_message_maps_prefixed_names_and_ignores_others():
    names = ["joint%d" % i for i in range(1, 7)]
    core = FollowerCore(names, [-3.0] * 6, [3.0] * 6, Params(check_stamp=False),
                        ["arm_left_joint%d" % i for i in range(1, 7)])
    msg_names = (["arm_left_joint%d" % i for i in range(1, 7)]
                 + ["arm_right_xeg32_left_carriage_joint", "arm_left_xeg32_left_carriage_joint"]
                 + ["arm_right_joint%d" % i for i in range(1, 7)])
    msg_pos = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -0.01, -0.01, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
    assert core.on_input(0.0, msg_names, msg_pos) is None
    assert core.target == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    # a NaN in the ignored arm does not matter, a NaN in ours does
    msg_pos[10] = float("nan")
    assert core.on_input(0.01, msg_names, msg_pos) is None
    msg_pos[2] = float("nan")
    assert core.on_input(0.02, msg_names, msg_pos) is not None
    # missing one of our joints is rejected
    assert "missing" in core.on_input(0.03, msg_names[1:], msg_pos[1:])


# ------------------------------------------------------------ slow catch-up
def run_until(sim, target, state, max_s=60.0):
    for _ in range(int(max_s / DT)):
        sim.step(target)
        if sim.status.state == state:
            return True
    return False


def test_catchup_engage_limits_sync_and_converges():
    p = Params()
    sim = Sim(make())
    tgt = [1.5, -0.75, 0.3, 0.0, 1.0, -1.2]
    sim.step(tgt)
    sim.step(tgt)
    assert sim.status.state == CATCHUP and not sim.status.latched
    seen_events = []
    seen_catchup = False
    for _ in range(int(120 / DT)):
        v = sim.step(tgt)
        seen_events += sim.status.events
        if sim.status.state == CATCHUP:
            seen_catchup = True
            assert max(map(abs, v)) <= p.catchup_velocity + 1e-9
            vm = max(map(abs, v))
            if vm > 0.02:
                emax = max(abs(t - q) for t, q in zip(tgt, sim.q))
                for vi, t, q in zip(v, tgt, sim.q):
                    # direction ratio ~ error ratio (command lags plant by one step)
                    assert vi / vm == pytest.approx((t - q) / emax, abs=0.02)
        if sim.status.state == ENABLED:
            break
    assert seen_catchup and sim.status.state == ENABLED
    assert "caught up" in seen_events
    sim.run(8.0, tgt)
    assert sim.q == pytest.approx(tgt, abs=0.003)


def test_catchup_acceleration_limit():
    sim = Sim(make())
    tgt = [1.5, 0, 0, 0, -1.0, 0]
    prev = [0.0] * 6
    peak = 0.0
    for _ in range(int(60 / DT)):
        v = sim.step(tgt)
        if sim.status.state == CATCHUP:
            peak = max(peak, max(abs(a - b) for a, b in zip(v, prev)))
        prev = v
    assert peak > 0.0
    assert peak <= 0.3 * DT + 1e-9


def test_engage_beyond_max_catchup_refused():
    sim = Sim(make())
    far = [1.76, 0, 0, 0, 0, 0]
    sim.run(0.1, far)
    assert sim.status.state == REFUSED and sim.status.latched
    assert sim.q == [0.0] * 6


def test_step_while_engaged_catches_up_then_enabled():
    sim = Sim(make())
    sim.run(0.3, [0.0] * 6)
    jump = [0.262, 0, 0, 0, 0, 0]
    sim.step(jump)
    sim.step(jump)
    assert sim.status.state == CATCHUP and not sim.status.latched
    assert run_until(sim, jump, ENABLED)
    assert max(abs(q - t) for q, t in zip(sim.q, jump)) < 0.03


def test_stale_input_during_catchup_holds_then_resumes_catchup():
    sim = Sim(make())
    tgt = [1.5, 0, 0, 0, 0, 0]
    sim.run(2.0, tgt)
    assert sim.status.state == CATCHUP
    sim.run(0.3, refresh=False)
    assert sim.status.state == HOLD and "stale" in sim.status.reason
    sim.run(0.5, refresh=False)
    assert sim.last_v == [0.0] * 6
    sim.step(tgt)
    assert sim.status.state == CATCHUP
    assert max(map(abs, sim.last_v)) <= 0.3 * DT + 1e-9  # ramps from rest, slow
    assert not sim.status.latched


def test_error_beyond_max_catchup_while_engaged_latches_hold():
    sim = Sim(make())
    sim.run(0.3, [0.0] * 6)
    sim.step([0.1, 0, 0, 0, 0, 0])
    sim.step([0.3, 0, 0, 0, 0, 0])
    far = [0.0] * 6
    far[1] = 1.9
    for k in (0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 1.9):
        far[1] = k
        sim.step(far)
        sim.step(far)
    sim.step(far)
    # error 1.9 rad > 1.75 -> latched HOLD
    assert sim.status.state == HOLD and sim.status.latched
    assert "max_catchup" not in sim.status.reason and "1.75" in sim.status.reason
    assert sim.last_v == pytest.approx([0.0] * 6, abs=0.03)
    sim.run(0.5, [0.0] * 6)
    assert sim.status.state == HOLD and sim.status.latched
    sim.enable = False
    sim.run(0.1, [0.0] * 6)
    assert not sim.status.latched and sim.status.state == DISABLED
    sim.enable = True
    sim.run(0.2, [0.0] * 6)
    assert sim.status.state == ENABLED


def test_error_beyond_engage_while_following_enters_catchup():
    sim = Sim(make())
    sim.run(0.3, [0.0] * 6)
    # ramp the target slowly (steps < 0.2) far away faster than the robot follows
    for k in range(1, 12):
        sim.step([0.1 * k, 0, 0, 0, 0, 0])
        sim.step([0.1 * k, 0, 0, 0, 0, 0])
    assert sim.status.state == CATCHUP


@pytest.mark.parametrize("kw", [
    {"catchup_velocity": 0.3},                    # > max_velocity 0.2
    {"catchup_velocity": 0.0},
    {"catchup_acceleration": 0.6},                # > max_acceleration 0.5
    {"catchup_done_rad": 0.2},                    # >= max_input_step_rad
    {"catchup_done_rad": 0.5, "max_input_step_rad": 0.9},   # >= max_engage_error_rad
    {"max_catchup_error_rad": 3.2},               # > pi
    {"max_catchup_error_rad": 0.5},               # not > max_engage_error_rad
])
def test_catchup_params_rejected(kw):
    with pytest.raises(ValueError):
        make(**kw)
