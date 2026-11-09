"""Tests for tools/mocap_dataset/replay_episode.py (trajectory build, limits, CLI; no ROS needed).

    cd tools/mocap_dataset && python3 -m pytest -p no:anyio test/test_replay_episode.py
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import replay_episode as R  # noqa: E402
from test_lerobot_to_plan import FPS, make_dataset  # noqa: E402

BASE = np.array([0.0, 0.2, 1.0, 0.0, 0.5, 0.0])
AMP = np.array([0.4, 0.3, 0.5, 0.6, 0.3, 0.8])


def rest_to_rest(seconds=30.0, noise=0.0, seed=0):
    """Smooth raised-cosine sweep, at rest at both ends (peak ~0.1 rad/s)."""
    t = np.arange(int(seconds * FPS) + 1) / FPS
    s = (1 - np.cos(2 * math.pi * t / seconds)) / 2
    q = BASE + AMP * s[:, None]
    if noise:
        q = q + np.random.default_rng(seed).normal(0, noise, q.shape)
        q[0], q[-1] = BASE, BASE
    return t, q


def mid_motion(seconds=25.0):
    """sin sweep: starts and ends while moving."""
    t = np.arange(int(seconds * FPS) + 1) / FPS
    return t, BASE + AMP * np.sin(2 * math.pi * t / 40.0)[:, None]


def args_for(ds, *extra):
    return [str(ds), "--offline", *extra]


# ---------------------------------------------------------------- build
def test_time_monotonic_zero_end_velocity_and_pinned_ends():
    t, q = rest_to_rest(noise=0.0005)
    tr = R.build_trajectory(t, q)
    assert np.all(np.diff(tr.tau) > 0) and tr.tau[0] == 0.0
    assert np.all(tr.v[0] == 0) and np.all(tr.v[-1] == 0)
    assert np.allclose(tr.q[0], q[0], atol=1e-9) and np.allclose(tr.q[-1], q[-1], atol=1e-9)
    assert tr.duration == pytest.approx(t[-1] + (tr.ramp_in + tr.ramp_out) / 2, abs=1e-9)
    assert np.max(np.diff(tr.tau)) <= 1 / tr.rate_hz + 1e-9
    assert len(tr.tau) == len(tr.q) == len(tr.v)


def test_deviation_bound_and_noise_is_smoothed():
    t, q = rest_to_rest(noise=0.0008)
    tr = R.build_trajectory(t, q)
    assert math.degrees(tr.dev_time_rad) < 1.0 and math.degrees(tr.dev_path_rad) < 1.0
    # smoother than the input: second difference of the sampled trajectory is far below the noise-driven one
    raw_a = np.abs(np.diff(q, 2, axis=0)).max() * FPS**2
    assert R.dynamics(tr, 1.0)[1].max() < 0.3 * raw_a


def test_velocities_consistent_with_positions():
    t, q = rest_to_rest(noise=0.0003)
    tr = R.build_trajectory(t, q)
    tt = np.linspace(2.0, tr.duration - 2.0, 400)
    num = (R.hermite_eval(tr.tau, tr.q, tr.v, tt + 1e-4) - R.hermite_eval(tr.tau, tr.q, tr.v, tt - 1e-4)) / 2e-4
    assert np.allclose(R.hermite_eval(tr.tau, tr.q, tr.v, tt, deriv=1), num, atol=1e-4)
    # knot velocities equal the spline derivative there and the data derivative within the smoothing error
    kv = R.hermite_eval(tr.tau, tr.q, tr.v, tr.tau, deriv=1)
    assert np.allclose(kv, tr.v, atol=1e-6)
    # accelerations: analytic knot-end formula agrees with the evaluator
    a = R.knot_accelerations(tr.tau, tr.q, tr.v)
    k = len(tr.tau) - 1
    assert a.shape == (2 * k, 6)
    # right limit of the spline acceleration at knot 3 = first row of segment 3; left limit at knot 4 = second block
    eps = 1e-7
    right = R.hermite_eval(tr.tau, tr.q, tr.v, np.array([tr.tau[3] + eps]), deriv=2)[0]
    left = R.hermite_eval(tr.tau, tr.q, tr.v, np.array([tr.tau[4] - eps]), deriv=2)[0]
    assert np.allclose(right, a[3], atol=1e-4) and np.allclose(left, a[k + 3], atol=1e-4)


def test_mid_motion_ends_get_ease_and_keep_the_path():
    t, q = mid_motion()
    tr = R.build_trajectory(t, q, ramp_acc=0.55)
    assert tr.ramp_in > 0.2 and tr.ramp_out > 0.2                 # still moving at both ends: longer ease
    assert np.all(tr.v[0] == 0) and np.all(tr.v[-1] == 0)
    assert tr.duration > t[-1]                                   # eased ends take longer, never shorter
    assert math.degrees(tr.dev_path_rad) < 1.0                    # same geometry
    assert np.allclose(tr.q[-1], q[-1], atol=1e-9)                # still ends at the recorded pose


def test_ease_ramp_acceleration_bounded_by_max_acc():
    t, q = mid_motion()
    tr = R.build_trajectory(t, q, ramp_acc=0.55)
    assert R.dynamics(tr, 1.0)[1].max() < 0.55


def test_short_slice_refused():
    t, q = rest_to_rest(1.0)
    with pytest.raises(R.ReplayError):
        R.build_trajectory(t, q)


# ---------------------------------------------------------------- speed scaling, limits
def test_speed_scaling_of_time_velocity_acceleration():
    t, q = rest_to_rest()
    tr = R.build_trajectory(t, q)
    pv1, pa1 = R.dynamics(tr, 1.0)
    pv, pa = R.dynamics(tr, 0.5)
    assert np.allclose(pv, 0.5 * pv1) and np.allclose(pa, 0.25 * pa1)
    tt, pos, vel = R.scaled_points(tr, 0.5)
    assert tt[-1] == pytest.approx(2 * tr.duration) and np.allclose(vel, 0.5 * tr.v) and np.allclose(pos, tr.q)
    assert np.all(np.diff(tt) > 0) and np.all(vel[0] == 0) and np.all(vel[-1] == 0)


def test_limit_refusal_suggests_a_passing_speed():
    t, q = rest_to_rest(10.0)                                      # fast: ~0.3 rad/s peak, accel ~0.3
    tr = R.build_trajectory(t, q)
    pv1, pa1 = R.dynamics(tr, 1.0)
    # refuse on velocity
    with pytest.raises(R.ReplayError) as e:
        R.check_dynamics(tr, 1.0, pv1.max() * 0.7, 10.0)
    s = R.suggest_speed(tr, pv1.max() * 0.7, 10.0)
    assert f"--speed {s:g} would pass" in str(e.value) and "re-timed" in str(e.value)
    R.check_dynamics(tr, s, pv1.max() * 0.7, 10.0)                  # the suggestion passes
    with pytest.raises(R.ReplayError):
        R.check_dynamics(tr, s + 0.02, pv1.max() * 0.7, 10.0)       # a bit more does not
    # refuse on acceleration (quadratic in speed)
    amax = pa1.max() * 0.5
    with pytest.raises(R.ReplayError) as e2:
        R.check_dynamics(tr, 1.0, 10.0, amax)
    s2 = R.suggest_speed(tr, 10.0, amax)
    assert s2 == pytest.approx(math.sqrt(0.5), abs=0.011) and "peak acceleration" in str(e2.value)
    R.check_dynamics(tr, s2, 10.0, amax)
    # both limits ok: returns the peaks
    pv, pa = R.check_dynamics(tr, 0.5, 10.0, 10.0)
    assert pv.shape == (6,) and pa.shape == (6,)


def test_trajectory_limit_check():
    t, q = rest_to_rest()
    tr = R.build_trajectory(t, q)
    lim = np.array([[-5, 5]] * 6, float)
    R.trajectory_within_limits(tr, lim, math.radians(2))
    lim[2] = [-5, 1.2]                                             # joint3 reaches 1.5
    with pytest.raises(R.ReplayError):
        R.trajectory_within_limits(tr, lim, math.radians(2))


# ---------------------------------------------------------------- approach
def test_approach_zero_end_velocity_and_peak_velocity():
    q0 = np.zeros(6)
    q1 = np.array([0.5, -0.3, 0.2, 0.0, 0.1, 0.0])
    tt, pos, vel, acc = R.approach_points(q0, q1, 0.10)
    assert np.all(vel[0] == 0) and np.all(np.abs(vel[-1]) < 1e-12) and np.allclose(pos[-1], q1) and np.allclose(pos[0], q0)
    assert np.all(np.diff(tt) > 0)
    assert np.abs(vel).max() == pytest.approx(0.10, rel=0.02)       # sampled peak (exact peak at the middle)
    assert acc == pytest.approx(6 * 0.5 / tt[-1] ** 2)
    # tiny move: minimum duration, still smooth
    tt2, _, vel2, _ = R.approach_points(q0, q0 + 0.01, 0.10)
    assert tt2[-1] >= 2.0 and np.abs(vel2).max() < 0.02


# ---------------------------------------------------------------- CLI on a synthetic dataset
def test_cli_dry_run_ok_and_refusals(tmp_path, capsys):
    t, q = rest_to_rest(30.0)
    ds = make_dataset(tmp_path / "ds", q)
    assert R.main(args_for(ds)) == 0
    out = capsys.readouterr().out
    assert "dry-run: nothing sent" in out and "verdict   trajectory OK" in out and "approach  unknown" in out
    assert R.main(args_for(ds, "--speed", "1.2")) == 2                         # faster needs --allow-faster
    assert "needs --allow-faster" in capsys.readouterr().err
    assert R.main(args_for(ds, "--speed", "1.6", "--allow-faster")) == 2       # cap 1.5
    assert R.main(args_for(ds, "--speed", "0")) == 2
    assert R.main(args_for(ds, "--max-vel", "0.01", "--approach-vel", "0.005")) == 2   # velocity limit: names a speed
    assert "would pass" in capsys.readouterr().err
    assert R.main(args_for(ds, "--arm", "arm_right")) == 2                     # right arm all zeros
    assert "all zeros" in capsys.readouterr().err


def test_cli_refuses_limit_violation_and_jump(tmp_path, capsys):
    t, q = rest_to_rest(30.0)
    bad = q.copy()
    bad[100:, 4] = 3.0                                                         # joint5 beyond +-95 deg, and a jump
    assert R.main(args_for(make_dataset(tmp_path / "a", bad))) == 2
    assert "outside the allowed" in capsys.readouterr().err
    bad = q.copy()
    bad[200:, 0] += 0.5
    assert R.main(args_for(make_dataset(tmp_path / "b", bad))) == 2
    assert "jump" in capsys.readouterr().err


def test_cli_slice_ending_mid_motion_gets_ease_out(tmp_path, capsys):
    t, q = rest_to_rest(30.0)
    ds = make_dataset(tmp_path / "ds", q, action=q + 0.01)
    assert R.main(args_for(ds, "--start-s", "0", "--end-s", "10")) == 0         # ends at 1/3 of the sweep: moving
    out = capsys.readouterr().out
    assert "slice 0.0-10.0 s" in out and "ease-out 0.40 s" in out and "ease-in 0.20 s" in out
    assert R.main(args_for(ds, "--end-s", "1")) == 2                           # < 2 s slice refused
    assert "at least" in capsys.readouterr().err
    # action source differs from state by 0.01 rad: runs and reports the same shape of summary
    assert R.main(args_for(ds, "--source", "action")) == 0


def test_execute_needs_tty(tmp_path, capsys):
    t, q = rest_to_rest(30.0)
    ds = make_dataset(tmp_path / "ds", q)
    assert R.main([str(ds), "--execute"]) == 2                                 # pytest stdin is not a TTY: refused before ROS
    assert "interactive terminal" in capsys.readouterr().err


def test_execute_refuses_foreign_action(tmp_path, capsys):
    t, q = rest_to_rest(30.0)
    ds = make_dataset(tmp_path / "ds", q)
    assert R.main([str(ds), "--execute", "--action", "/arm_right_trajectory_controller/follow_joint_trajectory"]) == 2
    assert "never command the other arm" in capsys.readouterr().err
