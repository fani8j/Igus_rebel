#!/usr/bin/env python3
"""Replay one episode of a robot_remote LeRobot v3.0 dataset (mocap/teleop recording of ONE ReBeL arm) on the real arm at
the RECORDED timing (time-scaled) as ONE smooth FollowJointTrajectory goal.

    python3 replay_episode.py DATASET_DIR [--arm arm_left] [--source state|action] [--start-s S] [--end-s E]
                              [--speed 1.0] [--max-vel 0.35] [--max-acc 0.55] [--execute]

Dry-run by default: load, check, build the trajectory and print a summary.  Nothing is sent without --execute.

Why this instead of lerobot_to_plan.py + publish_plan.py: that player re-times every point with a constant --max-speed
(stop-and-go chunks).  Here the recorded timestamps are kept: time_from_start = recorded time / --speed.

Trajectory build (pure numpy, see build_trajectory):
  1. the slice of the recorded joints is checked (limits minus margin, jumps, NaN, all-zero arm: lerobot_to_plan.check_series);
  2. linear upsample to 200 Hz, Gaussian low-pass (--smooth-s, sigma 0.2 s; odd reflection at the slice ends so the end
     slope is kept), velocity = derivative of the SMOOTHED signal, so positions and velocities belong together;
  3. zero velocity at both ends by a smooth time-ease (smoothstep) of the progression along the SAME path: where the
     recording is still moving at the slice end (4 of 12 episodes end mid-motion) the arm slows down over a ramp of
     max(0.2 s, 3 * v_edge / --max-acc) instead of stopping dead.  The geometry is unchanged, only the last/first part
     takes a little longer (printed);
  4. sampled at --rate-hz (10) with positions AND velocities: the controller interpolates cubic splines between points;
  5. the result is compared with the recorded data (time-aligned and geometric path deviation, must stay <= --max-dev-deg).
Then everything is scaled: time / speed, velocity * speed, acceleration * speed^2.  The peak velocity must stay <=
--max-vel and the peak acceleration <= --max-acc, otherwise the script REFUSES and prints the --speed that would pass.
It never silently re-times.

Default limits.  --max-vel 0.35 rad/s: the recording teleop was capped at 0.3 rad/s (smoothed data peaks at 0.26-0.28), the
robot override is 100 %, so 0.35 leaves ~20 % headroom for the smoothed peak and rejects --speed > ~1.25.  --max-acc 0.55
rad/s^2: the controllers.yaml fit of the real plant gives a robot-side acceleration limit of ~0.52 rad/s^2 (planned
accelerations above ~0.5 can not be followed); the recorded motion itself has 0.43-0.51 rad/s^2 after the 0.2 s smoothing
(onsets of the teleop catch-up), measured on a real arm, so 0.55 passes --speed 1.0 for the recorded data with a little
headroom and refuses anything faster (acceleration scales with speed^2).

Execute flow (needs a TTY): fresh /joint_states -> live pose and start pose printed -> typed `yes` -> APPROACH (separate
smooth goal at --approach-vel, own Enter, arrival check <= 1 deg) if the arm is > 0.5 deg from the start pose -> Enter ->
REPLAY as one goal with progress, tracking-error and stale-state watchdogs.  Ctrl-C cancels the goal.  No park move.
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
from lerobot_episode import load_episode  # noqa: E402
from lerobot_to_plan import DEFAULT_URDF, PlanError, check_series, urdf_limits  # noqa: E402

PUBLISH_PLAN = REPO / "tools" / "lid_open_friend" / "scripts" / "lid_open" / "publish_plan.py"
FINE_HZ = 200.0                  # internal grid for smoothing / dense evaluation
MIN_SLICE_S = 2.0
APPROACH_TOL_DEG = 0.5           # approach only if the live pose is further than this from the start pose
ARRIVAL_TOL_DEG = 1.0            # required after the approach (and before the replay)
EXIT_REFUSED, EXIT_FAILED, EXIT_ABORTED, EXIT_INTERRUPTED = 2, 3, 4, 130


class ReplayError(PlanError):
    """Input or precondition refused (nothing sent)."""


# ---------------------------------------------------------------- cubic Hermite (what the controller interpolates)
def hermite_eval(tau: np.ndarray, q: np.ndarray, v: np.ndarray, tq: np.ndarray, deriv: int = 0) -> np.ndarray:
    """Cubic Hermite spline through knots (tau (K,), q (K,6), v (K,6)) at times tq (M,) -> (M,6).
    deriv 0/1/2 = position/velocity/acceleration.  tq outside [tau[0], tau[-1]] is clipped."""
    tq = np.clip(tq, tau[0], tau[-1])
    i = np.clip(np.searchsorted(tau, tq, side="right") - 1, 0, len(tau) - 2)
    h = (tau[i + 1] - tau[i])[:, None]
    s = ((tq - tau[i])[:, None]) / h
    q0, q1, v0, v1 = q[i], q[i + 1], v[i], v[i + 1]
    if deriv == 0:
        return ((2 * s**3 - 3 * s**2 + 1) * q0 + (s**3 - 2 * s**2 + s) * h * v0
                + (-2 * s**3 + 3 * s**2) * q1 + (s**3 - s**2) * h * v1)
    if deriv == 1:
        return ((6 * s**2 - 6 * s) * q0 + (-6 * s**2 + 6 * s) * q1) / h + (3 * s**2 - 4 * s + 1) * v0 + (3 * s**2 - 2 * s) * v1
    return ((12 * s - 6) * q0 + (-12 * s + 6) * q1) / h**2 + ((6 * s - 4) * v0 + (6 * s - 2) * v1) / h


def knot_accelerations(tau: np.ndarray, q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """(2*(K-1), 6) accelerations at both ends of every segment (acceleration is linear inside a segment, so this is the
    exact maximum of |a| of the spline)."""
    h = np.diff(tau)[:, None]
    q0, q1, v0, v1 = q[:-1], q[1:], v[:-1], v[1:]
    a0 = (6 * (q1 - q0)) / h**2 + (-4 * v0 - 2 * v1) / h
    a1 = (-6 * (q1 - q0)) / h**2 + (2 * v0 + 4 * v1) / h
    return np.vstack([a0, a1])


# ---------------------------------------------------------------- trajectory build
@dataclass
class Trajectory:
    """Speed-independent (data-rate) trajectory; time is `tau` [s] at speed 1.0."""
    tau: np.ndarray                  # (K,) knot times
    q: np.ndarray                    # (K, 6) rad
    v: np.ndarray                    # (K, 6) rad/s at speed 1.0
    data_duration: float             # recorded slice duration [s]
    ramp_in: float                   # ease-in / ease-out durations [s]
    ramp_out: float
    sigma_s: float
    rate_hz: float
    dev_time_rad: float = 0.0        # max |trajectory(t_i) - recorded(t_i)| over joints and frames (time-aligned)
    dev_time_at: tuple = (0.0, 0)    # (data time, joint index 0-based)
    dev_path_rad: float = 0.0        # max over frames of the max-norm distance to the dense trajectory (geometric)
    notes: list = field(default_factory=list)

    @property
    def duration(self) -> float:
        return float(self.tau[-1])


def gaussian_smooth(qf: np.ndarray, sigma_s: float, fine_hz: float = FINE_HZ) -> np.ndarray:
    """Gaussian low-pass of a uniformly sampled (N, 6) signal; odd (point) reflection at both ends keeps the end value
    and slope."""
    k = int(math.ceil(4 * sigma_s * fine_hz))
    if len(qf) <= k + 1:
        raise ReplayError(f"slice too short for --smooth-s {sigma_s:g} ({len(qf)} samples at {fine_hz:g} Hz)")
    x = np.arange(-k, k + 1) / fine_hz
    w = np.exp(-0.5 * (x / sigma_s) ** 2)
    w /= w.sum()
    pad = np.concatenate([2 * qf[0] - qf[1:k + 1][::-1], qf, 2 * qf[-1] - qf[-k - 1:-1][::-1]])
    return np.column_stack([np.convolve(pad[:, j], w, mode="valid") for j in range(qf.shape[1])])


def _ramp_time(x: np.ndarray, r: float) -> np.ndarray:
    """Data time consumed after x seconds of an ease ramp of length r (progress rate = smoothstep(x / r))."""
    y = np.clip(x / r, 0.0, 1.0)
    return r * (y**3 - 0.5 * y**4)


def build_trajectory(t: np.ndarray, q: np.ndarray, *, sigma_s: float = 0.2, rate_hz: float = 10.0,
                     ramp_acc: float = 0.55, min_ramp_s: float = 0.2) -> Trajectory:
    """Recorded joints (t (N,) s, q (N, 6) rad) -> smooth trajectory with positions + velocities and zero end
    velocities.  ramp_acc: the acceleration the end ease is dimensioned for (use --max-acc)."""
    t = np.asarray(t, float) - float(t[0])
    T = float(t[-1])
    if T < MIN_SLICE_S:
        raise ReplayError(f"slice is {T:.2f} s long, need at least {MIN_SLICE_S:g} s")
    if sigma_s <= 0 or rate_hz <= 0 or ramp_acc <= 0:
        raise ReplayError("--smooth-s, --rate-hz and --max-acc must be > 0")
    n_fine = int(round(T * FINE_HZ)) + 1
    tf = np.linspace(0.0, T, n_fine)
    qf = np.column_stack([np.interp(tf, t, q[:, j]) for j in range(q.shape[1])])
    qs = gaussian_smooth(qf, sigma_s)
    vs = np.gradient(qs, tf, axis=0)

    v_start, v_end = float(np.abs(vs[0]).max()), float(np.abs(vs[-1]).max())
    r_in = max(min_ramp_s, 3.0 * v_start / ramp_acc)
    r_out = max(min_ramp_s, 3.0 * v_end / ramp_acc)
    if (r_in + r_out) / 2 > 0.4 * T:
        raise ReplayError(f"slice of {T:.1f} s is too short for the end ease ({r_in:.1f} s + {r_out:.1f} s)")
    L = T + (r_in + r_out) / 2                                # replay duration at speed 1.0

    def warp(tau: np.ndarray):
        """tau (trajectory time) -> (data time, progress rate m = dt/dtau)."""
        y_in = np.clip(tau / r_in, 0.0, 1.0)
        y_out = np.clip((L - tau) / r_out, 0.0, 1.0)
        t_in = _ramp_time(tau, r_in)
        t_out = T - _ramp_time(L - tau, r_out)
        t_mid = r_in / 2 + (tau - r_in)
        td = np.where(tau < r_in, t_in, np.where(tau > L - r_out, t_out, t_mid))
        m = np.minimum(3 * y_in**2 - 2 * y_in**3, 3 * y_out**2 - 2 * y_out**3)
        return np.clip(td, 0.0, T), m

    n_seg = max(2, int(math.ceil(L * rate_hz)))
    tau_k = np.linspace(0.0, L, n_seg + 1)
    td_k, m_k = warp(tau_k)
    q_k = np.column_stack([np.interp(td_k, tf, qs[:, j]) for j in range(6)])
    v_k = np.column_stack([np.interp(td_k, tf, vs[:, j]) for j in range(6)]) * m_k[:, None]
    v_k[0] = 0.0
    v_k[-1] = 0.0
    traj = Trajectory(tau_k, q_k, v_k, T, r_in, r_out, sigma_s, rate_hz)

    # --- deviation from the recorded data
    tau_f = np.linspace(0.0, L, int(round(L * FINE_HZ)) + 1)
    td_f, _ = warp(tau_f)
    tau_of_t = np.interp(t, td_f, tau_f)                      # trajectory time at which recorded frame i is replayed
    err = np.abs(hermite_eval(tau_k, q_k, v_k, tau_of_t) - q)
    i, j = np.unravel_index(int(np.argmax(err)), err.shape)
    traj.dev_time_rad, traj.dev_time_at = float(err[i, j]), (float(t[i]), int(j))
    dense = hermite_eval(tau_k, q_k, v_k, tau_f)
    worst = 0.0
    for a in range(0, len(q), 32):                            # chunked max-norm nearest-point search (bounded memory)
        d = np.abs(q[a:a + 32, None, :] - dense[None, :, :]).max(2).min(1)
        worst = max(worst, float(d.max()))
    traj.dev_path_rad = worst
    return traj


def dynamics(traj: Trajectory, speed: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-joint peak |velocity| (rad/s) and |acceleration| (rad/s^2) at the given speed (velocity: dense evaluation of
    the cubic spline; acceleration: exact maximum at the segment ends)."""
    dense_t = np.linspace(0.0, traj.duration, int(round(traj.duration * FINE_HZ)) + 1)
    pv = np.abs(hermite_eval(traj.tau, traj.q, traj.v, dense_t, deriv=1)).max(0)
    pa = np.abs(knot_accelerations(traj.tau, traj.q, traj.v)).max(0)
    return pv * speed, pa * speed**2


def suggest_speed(traj: Trajectory, max_vel: float, max_acc: float) -> float:
    """Largest --speed (floored to 0.01) at which both limits hold."""
    pv, pa = dynamics(traj, 1.0)
    s = min(max_vel / max(float(pv.max()), 1e-9), math.sqrt(max_acc / max(float(pa.max()), 1e-9)))
    return math.floor(s * 100.0 + 1e-9) / 100.0


def check_dynamics(traj: Trajectory, speed: float, max_vel: float, max_acc: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (peak vel, peak acc) at `speed`, or raise ReplayError with the speed that would pass."""
    pv, pa = dynamics(traj, speed)
    bad = []
    if pv.max() > max_vel:
        bad.append(f"peak velocity {pv.max():.3f} rad/s (joint{int(pv.argmax()) + 1}) > --max-vel {max_vel:g}")
    if pa.max() > max_acc:
        bad.append(f"peak acceleration {pa.max():.3f} rad/s^2 (joint{int(pa.argmax()) + 1}) > --max-acc {max_acc:g}")
    if bad:
        s = suggest_speed(traj, max_vel, max_acc)
        hint = f"--speed {s:g} would pass" if s >= 0.01 else "no --speed >= 0.01 passes"
        raise ReplayError(f"at --speed {speed:g}: " + "; ".join(bad) + f".  {hint} (nothing was re-timed).")
    return pv, pa


def scaled_points(traj: Trajectory, speed: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(time_from_start [s], positions, velocities) of the knots at the chosen speed."""
    return traj.tau / speed, traj.q.copy(), traj.v * speed


def approach_points(q0: np.ndarray, q1: np.ndarray, vel: float, min_dur: float = 2.0, step_s: float = 0.25):
    """Smoothstep (cubic, zero end velocity) joint-space move q0 -> q1 whose peak joint velocity is `vel`.
    Returns (time_from_start, positions, velocities, peak_acc)."""
    d = np.asarray(q1, float) - np.asarray(q0, float)
    dmax = float(np.abs(d).max())
    T = max(min_dur, 1.5 * dmax / vel)
    n = max(2, int(math.ceil(T / step_s)))
    s = np.linspace(0.0, 1.0, n + 1)[:, None]
    tt = (s[:, 0] * T)
    pos = np.asarray(q0, float) + d * (3 * s**2 - 2 * s**3)
    vv = d * (6 * s - 6 * s**2) / T
    return tt, pos, vv, float(6 * dmax / T**2)


# ---------------------------------------------------------------- loading
def load_slice(args) -> tuple[object, np.ndarray, np.ndarray, str]:
    """(episode, t (N,), q (N, 6), info text) of the requested arm / source / slice, all checks applied."""
    ep = load_episode(args.dataset, args.episode)
    arm = args.arm
    if f"{arm}_joint1" not in ep.names:
        raise ReplayError(f"{arm} is not in this dataset (names: {ep.names})")
    q_all = ep.arm(arm, args.source)
    t_all = ep.timestamp - ep.timestamp[0]
    if float(np.abs(q_all).max()) < 1e-6:
        raise ReplayError(f"{arm} {args.source} data is all zeros in the whole episode: this arm was not recorded")
    start_s = max(0.0, args.start_s)
    end_s = float(t_all[-1]) if args.end_s is None else min(float(args.end_s), float(t_all[-1]))
    sel = (t_all >= start_s - 1e-9) & (t_all <= end_s + 1e-9)
    t, q = t_all[sel], q_all[sel]
    limits = urdf_limits(Path(args.urdf_limits))
    margin = math.radians(args.limit_margin_deg)
    check_series(t, q, limits, margin, math.radians(args.max_jump_deg))
    args._limits, args._margin = limits, margin
    info = (f"dataset   {ep.root.name} ep{ep.episode_index}  '{ep.task}'  {len(ep.timestamp)} frames @ {ep.fps:g} fps "
            f"({t_all[-1]:.1f} s)\n"
            f"arm       {arm}  source={args.source}  slice {start_s:.1f}-{end_s:.1f} s ({len(q)} frames, {t[-1] - t[0]:.1f} s)\n"
            f"limits    OK: inside URDF limits - {args.limit_margin_deg:g} deg; max frame jump "
            f"{math.degrees(float(np.abs(np.diff(q, axis=0)).max())):.2f} deg (refuse > {args.max_jump_deg:g})")
    return ep, t, q, info


def trajectory_within_limits(traj: Trajectory, limits: np.ndarray, margin: float) -> None:
    dense_t = np.linspace(0.0, traj.duration, int(round(traj.duration * FINE_HZ)) + 1)
    p = hermite_eval(traj.tau, traj.q, traj.v, dense_t)
    lo, hi = limits[:, 0] + margin / 2, limits[:, 1] - margin / 2
    bad = np.argwhere((p < lo) | (p > hi))
    if len(bad):
        i, j = bad[0]
        raise ReplayError(f"the smoothed trajectory leaves the joint{j + 1} limits at t={dense_t[i]:.2f} s "
                          f"({p[i, j]:+.4f} rad); limit - {math.degrees(margin / 2):g} deg")


def d2(x: float) -> float:
    return math.degrees(float(x))


def summary_lines(args, traj: Trajectory, speed: float, pv: np.ndarray, pa: np.ndarray, live: np.ndarray | None,
                  live_note: str) -> list[str]:
    arm = args.arm
    L = []
    L.append(f"build     smooth sigma {args.smooth_s:g} s, {args.rate_hz:g} Hz knots with velocities (cubic splines), "
             f"{len(traj.tau)} points; ease-in {traj.ramp_in:.2f} s, ease-out {traj.ramp_out:.2f} s (zero velocity at both ends)")
    L.append(f"deviation vs recording: time-aligned max {d2(traj.dev_time_rad):.2f} deg (joint{traj.dev_time_at[1] + 1} at data "
             f"t={traj.dev_time_at[0]:.1f} s), path (geometric, max-norm) max {d2(traj.dev_path_rad):.2f} deg  "
             f"(refuse > {args.max_dev_deg:g})")
    L.append(f"timing    recorded {traj.data_duration:.1f} s -> replay {traj.duration / speed:.1f} s at --speed {speed:g} "
             f"(+{(traj.duration - traj.data_duration) / speed:.2f} s from the end eases, x{1 / speed:.2f} time scale)")
    L.append(f"dynamics at --speed {speed:g}: limits --max-vel {args.max_vel:g} rad/s, --max-acc {args.max_acc:g} rad/s^2")
    L.append("          joint        " + "".join(f"{j + 1:>8d}" for j in range(6)))
    L.append("          peak vel     " + "".join(f"{x:8.3f}" for x in pv) + "  rad/s")
    L.append("          peak acc     " + "".join(f"{x:8.3f}" for x in pa) + "  rad/s^2")
    L.append(f"start pose [deg] {[round(d2(x), 1) for x in traj.q[0]]}")
    L.append(f"end pose   [deg] {[round(d2(x), 1) for x in traj.q[-1]]}")
    if live is None:
        L.append(f"approach  unknown ({live_note})")
    else:
        dq = np.abs(live - traj.q[0])
        T = approach_points(live, traj.q[0], args.approach_vel)[0][-1]
        if d2(dq.max()) > APPROACH_TOL_DEG:
            L.append(f"approach  live pose {[round(d2(x), 1) for x in live]} deg is {d2(dq.max()):.1f} deg (max joint) from the "
                     f"start pose: approach goal at {args.approach_vel:g} rad/s, {T:.1f} s")
        else:
            L.append(f"approach  live pose is {d2(dq.max()):.2f} deg from the start pose (<= {APPROACH_TOL_DEG:g}): no approach needed")
    return L


# ---------------------------------------------------------------- main (pure part)
def prepare(args):
    """Load, check, build; returns (ep, t, q, traj, pv, pa, info).  Raises ReplayError."""
    if not math.isfinite(args.speed) or args.speed < 0.05:
        raise ReplayError("--speed must be >= 0.05")
    if args.speed > 1.5:
        raise ReplayError("--speed above 1.5 is not allowed")
    if args.speed > 1.0 and not args.allow_faster:
        raise ReplayError("--speed above 1.0 (faster than recorded) needs --allow-faster")
    ep, t, q, info = load_slice(args)
    traj = build_trajectory(t, q, sigma_s=args.smooth_s, rate_hz=args.rate_hz, ramp_acc=args.max_acc)
    trajectory_within_limits(traj, args._limits, args._margin)
    if math.degrees(traj.dev_time_rad) > args.max_dev_deg or math.degrees(traj.dev_path_rad) > args.max_dev_deg:
        raise ReplayError(f"deviation from the recorded path too large: time-aligned {d2(traj.dev_time_rad):.2f} deg, path "
                          f"{d2(traj.dev_path_rad):.2f} deg > --max-dev-deg {args.max_dev_deg:g}; lower --smooth-s "
                          "or choose a slice that starts/ends at rest (--start-s/--end-s)")
    pv, pa = check_dynamics(traj, args.speed, args.max_vel, args.max_acc)
    return ep, t, q, traj, pv, pa, info


# ---------------------------------------------------------------- ROS part (rclpy imported lazily)
class Abort(Exception):
    """Operator said no / precondition failed (exit code attached)."""

    def __init__(self, msg: str, code: int = EXIT_REFUSED):
        super().__init__(msg)
        self.code = code


class Runner:
    def __init__(self, args, joint_names: list[str], action: str):
        import rclpy
        from control_msgs.action import FollowJointTrajectory
        from rclpy.action import ActionClient
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import qos_profile_sensor_data
        from rclpy.signals import SignalHandlerOptions
        from sensor_msgs.msg import JointState

        self.rclpy, self.FJT = rclpy, FollowJointTrajectory
        self.args, self.names, self.action = args, joint_names, action
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        self.node = rclpy.create_node("replay_episode")
        self.exec = SingleThreadedExecutor()
        self.exec.add_node(self.node)
        self.pos: dict[str, float] = {}
        self.rx: dict[str, float] = {}
        # best effort: connects to the broadcaster (reliable) and to a best-effort publisher alike
        self.node.create_subscription(JointState, args.state_topic, self._on_state, qos_profile_sensor_data)
        self.client = ActionClient(self.node, FollowJointTrajectory, action)
        self.handle = None
        self.result_future = None
        self.done = False
        self.sigint = 0
        self.goal_pending = False
        self.fb = None                     # (monotonic, max |error| rad, joint index)
        for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(s, self._on_signal)

    # --- signals: with a goal in flight only set a counter (the loop cancels); otherwise behave like Ctrl-C
    def _on_signal(self, signum, frame):
        self.sigint += 1
        if self.handle is None and not self.goal_pending:
            raise KeyboardInterrupt

    def _on_state(self, msg):
        now = time.monotonic()
        for n, p in zip(msg.name, msg.position):
            if n in self.names:
                self.pos[n] = float(p)
                self.rx[n] = now

    def spin(self, dt: float = 0.02):
        self.exec.spin_once(timeout_sec=dt)

    def state_age(self) -> float:
        if any(n not in self.rx for n in self.names):
            return float("inf")
        return time.monotonic() - min(self.rx[n] for n in self.names)

    def q(self) -> np.ndarray:
        return np.array([self.pos[n] for n in self.names])

    def wait_state(self, timeout: float) -> np.ndarray:
        t0 = time.monotonic()
        while self.state_age() > 0.5:
            if time.monotonic() - t0 > timeout:
                miss = [n for n in self.names if n not in self.pos]
                from_pp = _load_publish_plan()
                msg = (f"no fresh /joint_states with all of {self.names} within {timeout:g} s"
                       + (f" (missing: {miss})" if miss else f" (last update {self.state_age():.1f} s ago)") + "\n"
                       + from_pp.describe_state_topics(self.node, self.args.state_topic))
                raise Abort(msg)
            self.spin()
        return self.q()

    # --- goal handling
    def _feedback(self, fb_msg):
        e = list(fb_msg.feedback.error.positions)
        if len(e) == len(self.names):
            a = np.abs(np.asarray(e))
            self.fb = (time.monotonic(), float(a.max()), int(a.argmax()))

    def _build_goal(self, tt, pos, vel):
        from builtin_interfaces.msg import Duration
        from trajectory_msgs.msg import JointTrajectoryPoint
        goal = self.FJT.Goal()
        goal.trajectory.joint_names = list(self.names)
        for t, p, v in zip(tt, pos, vel):
            pt = JointTrajectoryPoint()
            pt.positions = [float(x) for x in p]
            pt.velocities = [float(x) for x in v]
            sec = int(math.floor(t))
            ns = int(round((t - sec) * 1e9))
            if ns >= 1_000_000_000:
                sec, ns = sec + 1, ns - 1_000_000_000
            pt.time_from_start = Duration(sec=sec, nanosec=ns)
            goal.trajectory.points.append(pt)
        return goal

    def run_goal(self, label: str, tt, pos, vel) -> dict:
        """Send one goal and supervise it.  Returns dict(status, code, text, reason).  reason = None when the controller
        finished it on its own, else why this script cancelled it."""
        args = self.args
        total = float(tt[-1])
        goal = self._build_goal(tt, pos, vel)
        self.fb = None
        self.goal_pending = True
        self.done = False
        reason = None
        try:
            fut = self.client.send_goal_async(goal, feedback_callback=self._feedback)
            t0 = time.monotonic()
            while not fut.done() and time.monotonic() - t0 < 5.0:
                self.spin()
            if not fut.done() or not fut.result().accepted:
                raise Abort(f"{label}: goal not accepted by {self.action} "
                            "(controller inactive? `ros2 control list_controllers`)", EXIT_FAILED)
            self.handle = fut.result()
            self.result_future = self.handle.get_result_async()
            t_acc = time.monotonic()
            last_print, worst, seen = 0.0, 0.0, self.sigint
            print(f"[{label}] goal accepted, {total:.1f} s.  Ctrl-C cancels (controller holds position).")
            while not self.result_future.done():
                self.spin()
                el = time.monotonic() - t_acc
                if self.fb is not None:
                    worst = max(worst, self.fb[1])
                if self.sigint > seen:
                    reason = "Ctrl-C / signal"
                elif self.state_age() > args.stale_cancel_s:
                    reason = f"/joint_states stale {self.state_age():.2f} s > {args.stale_cancel_s:g} s"
                elif self.fb is not None and math.degrees(self.fb[1]) > args.max_track_err_deg:
                    reason = (f"tracking error {d2(self.fb[1]):.1f} deg (joint{self.fb[2] + 1}) > --max-track-err-deg "
                              f"{args.max_track_err_deg:g}")
                elif el > 1.5 and (self.fb is None or time.monotonic() - self.fb[0] > 1.0):
                    reason = "no action feedback for > 1 s (tracking can not be supervised)"
                elif el > total + 3.0 + 10.0:
                    reason = f"no result {el - total:.0f} s after the end of the trajectory"
                if reason:
                    self.cancel(reason)
                    break
                if el - last_print >= 1.0:
                    cur = "-" if self.fb is None else f"{d2(self.fb[1]):5.2f}"
                    print(f"[{label}] t={min(el, total):6.1f}/{total:.1f} s  tracking err now {cur} deg  worst {d2(worst):5.2f} deg"
                          f"  state age {self.state_age() * 1000:3.0f} ms")
                    last_print = el
            res = self.result_future.result() if self.result_future.done() else None
            out = {"reason": reason, "worst_err_deg": d2(worst)}
            if res is None:
                out.update(status=None, code=None, text="no result received")
            else:
                out.update(status=res.status, code=res.result.error_code, text=res.result.error_string)
            self.done = True
            return out
        finally:
            if self.handle is not None and not self.done:
                self.cancel("exit path (exception)")          # any exception path must not leave the goal running
            self.goal_pending = False
            self.handle = None

    def cancel(self, reason: str):
        h = self.handle
        if h is None:
            return
        print(f"\n*** CANCEL: {reason}.  The controller now holds the current position.")
        seen = self.sigint
        fut = h.cancel_goal_async()
        t0 = time.monotonic()
        while time.monotonic() - t0 < 3.0 and not (fut.done() and self.result_future.done()):
            self.spin()
            if self.sigint > seen:                           # another Ctrl-C: send again, do not wait as long
                seen = self.sigint
                h.cancel_goal_async()
                t0 = max(t0, time.monotonic() - 2.0)
        if fut.done():
            print(f"    cancel response return_code={fut.result().return_code} (0 = accepted)")
        else:
            print("    NO cancel response within 3 s: hit the E-STOP if the arm is still moving")
        self.done = True

    def watch_hold(self, secs: float = 1.5):
        """After a cancel: print how far the joints still move (hold evidence)."""
        t0 = time.monotonic()
        samples = []
        while time.monotonic() - t0 < secs:
            self.spin()
            if self.state_age() < 0.5:
                samples.append((time.monotonic() - t0, self.q()))
        if len(samples) < 4:
            print("    (no joint states to verify the hold)")
            return
        late = [q for t, q in samples if t > secs - 0.5]
        mv = float(np.abs(late[-1] - late[0]).max()) if len(late) > 1 else float("nan")
        print(f"    joints [deg] {[round(d2(x), 2) for x in samples[-1][1]]};  movement during the last 0.5 s: {d2(mv):.3f} deg")

    def shutdown(self):
        try:
            self.exec.shutdown()
            self.node.destroy_node()
            self.rclpy.shutdown()
        except Exception:                                    # noqa: BLE001
            pass


def _load_publish_plan():
    spec = importlib.util.spec_from_file_location("lid_open_publish_plan", PUBLISH_PLAN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)                              # top level: std imports and constants only
    return mod


def read_live_state(args, names: list[str], wait: float) -> tuple[np.ndarray | None, str]:
    """Dry-run helper: one passive look at /joint_states.  (None, reason) if unavailable."""
    try:
        import rclpy  # noqa: F401
    except ImportError:
        return None, "rclpy not available: source the ROS environment"
    r = None
    try:
        r = Runner(args, names, "unused")
        t0 = time.monotonic()
        while r.state_age() > 0.5 and time.monotonic() - t0 < wait:
            r.spin()
        if r.state_age() > 0.5:
            return None, f"no /joint_states with {names[0]}.. within {wait:g} s on ROS_DOMAIN_ID={os.environ.get('ROS_DOMAIN_ID', '(unset)')}"
        return r.q(), "live"
    except Exception as e:                                    # noqa: BLE001
        return None, f"live state unavailable ({e})"
    finally:
        if r is not None:
            r.shutdown()


def confirm_text(prompt: str, want: str) -> None:
    try:
        ans = input(prompt).strip().lower()
    except EOFError:
        raise Abort("end of input: aborted, nothing sent")
    if ans != want:
        raise Abort("not confirmed: aborted, nothing sent")


def execute(args, traj: Trajectory, pv: np.ndarray, pa: np.ndarray) -> int:
    names = [f"{args.arm}_joint{i}" for i in range(1, 7)]
    action = args.action or f"/{args.arm}_trajectory_controller/follow_joint_trajectory"
    if args.arm not in action:
        raise Abort(f"refusing: --action {action} does not belong to {args.arm} (never command the other arm)")
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        raise Abort("refusing to run --execute without an interactive terminal (typed confirmation required)")
    tt, pos, vel = scaled_points(traj, args.speed)
    run = Runner(args, names, action)
    code = 0
    try:
        print(f"ROS_DOMAIN_ID={os.environ.get('ROS_DOMAIN_ID', '(unset)')}  action {action}  state {args.state_topic}")
        live = run.wait_state(args.state_wait)
        if not run.client.wait_for_server(timeout_sec=5.0):
            raise Abort(f"action server {action} not available (controller not active? `ros2 control list_controllers`)")
        topic = action.rsplit("/", 1)[0] + "/joint_trajectory"
        pubs = [p for p in run.node.get_publishers_info_by_topic(topic)]
        limits, margin = args._limits, args._margin
        lo_i, hi_i = limits[:, 0], limits[:, 1]
        if np.any(live < lo_i) or np.any(live > hi_i):
            raise Abort(f"live pose {[round(d2(x), 1) for x in live]} deg is outside the URDF limits: not moving")
        print("\n=== LIVE ===")
        print(f"live joints  [deg] {[round(d2(x), 2) for x in live]}   (state age {run.state_age() * 1000:.0f} ms)")
        print(f"start pose   [deg] {[round(d2(x), 2) for x in traj.q[0]]}")
        dq = np.abs(live - traj.q[0])
        print(f"difference   [deg] {[round(d2(x), 2) for x in dq]}  (max {d2(dq.max()):.2f})")
        if pubs:
            print(f"!! WARNING: other publisher(s) on {topic}: "
                  + ", ".join(f"{p.node_name}" for p in pubs) + "\n   A message on that topic REPLACES an action goal: stop teleop / servo / executors first.")
        else:
            print(f"OK: no other publisher on {topic}")
        print(f"\nReplay: {len(tt)} points, {tt[-1]:.1f} s at --speed {args.speed:g} (recorded {traj.data_duration:.1f} s), "
              f"peak {pv.max():.2f} rad/s / {pa.max():.2f} rad/s^2.  Only {args.arm} is commanded.  No park move at the end.")
        print("SAFETY: Ctrl-C CANCELS the goal: the Humble joint_trajectory_controller then holds the CURRENT position "
              "(set_hold_position): the arm decelerates and stops within a fraction of a second, not instantly.\n"
              "        If this script crashes / is killed (SIGKILL, terminal gone), a goal that was already sent keeps running "
              "to its end: the E-STOP is the only hard stop.  Keep a hand on it.")
        confirm_text("E-stop in hand, workspace clear, box placed as in the recording, no other controller user?  Type 'yes': ", "yes")

        # ---- a. approach
        if dq.max() > math.radians(APPROACH_TOL_DEG):
            atime, apos, avel, aacc = approach_points(live, traj.q[0], args.approach_vel)
            print(f"\n=== APPROACH: joint-space smooth move, max joint travel {d2(dq.max()):.1f} deg, {atime[-1]:.1f} s "
                  f"(peak {args.approach_vel:g} rad/s, {aacc:.2f} rad/s^2), straight line in joint space, no collision check ===")
            confirm_enter("Enter = start the APPROACH, Ctrl-C = quit: ")
            live = run.wait_state(2.0)
            if np.abs(live - apos[0]).max() > math.radians(1.0):
                raise Abort("the arm moved since the last reading (> 1 deg): aborted before the approach")
            res = run.run_goal("approach", atime, apos, avel)
            code = finish_goal(run, res, "approach")
            if code:
                return code
            t0 = time.monotonic()
            err = float("inf")
            while time.monotonic() - t0 < args.arrival_wait_s:      # settle, up to the controller's goal_time
                run.spin()
                if run.state_age() < 0.5:
                    err = float(np.abs(run.q() - traj.q[0]).max())
                    if err <= math.radians(ARRIVAL_TOL_DEG):
                        break
            print(f"approach arrival error {d2(err):.2f} deg (required <= {ARRIVAL_TOL_DEG:g})")
            if not err <= math.radians(ARRIVAL_TOL_DEG):
                raise Abort("approach did not reach the start pose: aborted, replay NOT started", EXIT_FAILED)
        else:
            print("\nlive pose is within 0.5 deg of the start pose: no approach")

        # ---- b. replay
        live = run.wait_state(2.0)
        e0 = float(np.abs(live - traj.q[0]).max())
        if e0 > math.radians(ARRIVAL_TOL_DEG):
            raise Abort(f"the arm is {d2(e0):.2f} deg from the start pose (> {ARRIVAL_TOL_DEG:g}): replay NOT started", EXIT_FAILED)
        print(f"\n=== REPLAY: {tt[-1]:.1f} s, one goal; tracking watchdog {args.max_track_err_deg:g} deg, stale cancel "
              f"{args.stale_cancel_s:g} s ===")
        confirm_enter("Enter = start the REPLAY, Ctrl-C = quit: ")
        res = run.run_goal("replay", tt, pos, vel)
        code = finish_goal(run, res, "replay")
        t0 = time.monotonic()
        while time.monotonic() - t0 < 0.5:
            run.spin()
        if code == 0 and run.state_age() < 0.5:
            fe = np.abs(run.q() - traj.q[-1])
            print(f"final error vs last point [deg] {[round(d2(x), 2) for x in fe]}  (max {d2(fe.max()):.2f})")
        return code
    except KeyboardInterrupt:
        print("\ninterrupted before any goal was running: nothing to cancel")
        return EXIT_INTERRUPTED
    finally:
        run.shutdown()


def confirm_enter(prompt: str) -> None:
    try:
        input(prompt)
    except EOFError:
        raise Abort("end of input: aborted, nothing sent")


GOAL_STATUS = {4: "SUCCEEDED", 5: "CANCELED", 6: "ABORTED"}


def finish_goal(run: Runner, res: dict, label: str) -> int:
    """Print the outcome of a goal, return an exit code (0 = ok)."""
    st = GOAL_STATUS.get(res["status"], str(res["status"]))
    print(f"[{label}] result: status={st} error_code={res['code']} error_string={res['text']!r}; worst tracking error "
          f"{res['worst_err_deg']:.2f} deg")
    if res["reason"] is not None:
        run.watch_hold()
        return EXIT_INTERRUPTED if res["reason"].startswith("Ctrl-C") else EXIT_ABORTED
    if res["status"] != 4 or res["code"] != 0:
        return EXIT_FAILED
    return 0


# ---------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", help="LeRobot v3.0 dataset directory (never run with it as the working directory)")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--arm", default="arm_left", choices=["arm_left", "arm_right"])
    ap.add_argument("--source", choices=["state", "action"], default="state",
                    help="state = what the arm did (default); action = the commanded target, ~0.28 s ahead of state")
    ap.add_argument("--start-s", type=float, default=0.0)
    ap.add_argument("--end-s", type=float, default=None)
    ap.add_argument("--speed", type=float, default=1.0,
                    help="time scale: 0.5 = half speed, 1.0 = recorded; > 1.0 needs --allow-faster, cap 1.5")
    ap.add_argument("--allow-faster", action="store_true")
    ap.add_argument("--max-vel", type=float, default=0.35, help="refuse above this peak joint velocity at --speed [rad/s]")
    ap.add_argument("--max-acc", type=float, default=0.55, help="refuse above this peak joint acceleration at --speed [rad/s^2]")
    ap.add_argument("--smooth-s", type=float, default=0.2, help="Gaussian smoothing sigma [s] (mocap jitter)")
    ap.add_argument("--rate-hz", type=float, default=10.0, help="trajectory point rate (positions + velocities)")
    ap.add_argument("--max-dev-deg", type=float, default=1.0, help="refuse if smoothing deviates more than this from the recording")
    ap.add_argument("--limit-margin-deg", type=float, default=2.0)
    ap.add_argument("--max-jump-deg", type=float, default=5.7)
    ap.add_argument("--urdf-limits", default=str(DEFAULT_URDF))
    ap.add_argument("--action", default=None, help="FollowJointTrajectory action (default /<arm>_trajectory_controller/follow_joint_trajectory)")
    ap.add_argument("--state-topic", default="/joint_states")
    ap.add_argument("--state-wait", type=float, default=10.0, help="--execute: wait this long for the first /joint_states [s]")
    ap.add_argument("--approach-vel", type=float, default=0.10, help="peak joint velocity of the approach move [rad/s]")
    ap.add_argument("--arrival-wait-s", type=float, default=3.0, help="settle time after the approach (the controller's goal_time)")
    ap.add_argument("--max-track-err-deg", type=float, default=8.0, help="cancel when the tracking error exceeds this")
    ap.add_argument("--stale-cancel-s", type=float, default=0.3, help="cancel when /joint_states is older than this")
    ap.add_argument("--offline", action="store_true", help="dry-run: do not look at /joint_states for the approach distance")
    ap.add_argument("--execute", action="store_true", help="really move the arm (TTY, typed yes, Enter per goal)")
    args = ap.parse_args(argv)

    try:
        if args.approach_vel <= 0 or args.approach_vel > args.max_vel:
            raise ReplayError(f"--approach-vel must be in (0, --max-vel {args.max_vel:g}]")
        ep, t, q, traj, pv, pa, info = prepare(args)
    except (PlanError, ValueError, OSError, KeyError) as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return EXIT_REFUSED
    names = [f"{args.arm}_joint{i}" for i in range(1, 7)]
    if args.execute:
        live, note = None, "checked against the live /joint_states in the execute flow below"
    elif args.offline:
        live, note = None, "--offline"
    else:
        live, note = read_live_state(args, names, 2.0)
    print(info)
    print("\n".join(summary_lines(args, traj, args.speed, pv, pa, live, note)))
    print(f"verdict   trajectory OK at --speed {args.speed:g} (inside --max-vel / --max-acc / --max-dev-deg)")
    if not args.execute:
        print("dry-run: nothing sent.  Add --execute (real arm: operator only, E-stop in hand).")
        return 0
    try:
        return execute(args, traj, pv, pa)
    except Abort as e:
        print(f"\nABORTED: {e}", file=sys.stderr)
        return e.code
    except KeyboardInterrupt:
        print("\ninterrupted")
        return EXIT_INTERRUPTED


if __name__ == "__main__":
    sys.exit(main())
