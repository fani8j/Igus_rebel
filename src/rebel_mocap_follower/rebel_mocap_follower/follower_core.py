"""ROS-free safety/velocity logic of the mocap follower.

All time values are seconds on a monotonic clock supplied by the caller, so the
logic is deterministic and unit-testable.
"""
import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

DISABLED = "DISABLED"
ENABLED = "ENABLED"
HOLD = "HOLD"
REFUSED = "REFUSED"
CATCHUP = "CATCHUP"

MAX_VELOCITY_CAP = 0.6
MAX_ACCELERATION_CAP = 1.5
MAX_DT = 0.05


@dataclass
class Params:
    input_timeout_s: float = 0.2
    joint_states_timeout_s: float = 0.2
    limit_margin_rad: float = 0.05
    max_input_step_rad: float = 0.2
    max_engage_error_rad: float = 0.5
    catchup_velocity: float = 0.1
    catchup_acceleration: float = 0.3
    catchup_done_rad: float = 0.03
    max_catchup_error_rad: float = 1.75
    max_velocity: float = 0.2
    max_acceleration: float = 0.5
    stop_deceleration: float = 1.0
    kp: float = 2.0
    deadband_rad: float = 0.002
    check_stamp: bool = True
    max_stamp_skew_s: float = 1.0
    clamp_to_limits: bool = True

    def validate(self):
        """Raise ValueError for unsafe or nonsensical values (hard caps)."""
        if not (0.0 < self.max_velocity <= MAX_VELOCITY_CAP):
            raise ValueError(f"max_velocity {self.max_velocity} outside (0, {MAX_VELOCITY_CAP}]")
        if not (0.0 < self.max_acceleration <= MAX_ACCELERATION_CAP):
            raise ValueError(
                f"max_acceleration {self.max_acceleration} outside (0, {MAX_ACCELERATION_CAP}]")
        if not (0.0 < self.stop_deceleration <= MAX_ACCELERATION_CAP):
            raise ValueError(
                f"stop_deceleration {self.stop_deceleration} outside (0, {MAX_ACCELERATION_CAP}]")
        if not (0.0 < self.kp <= 10.0):
            raise ValueError(f"kp {self.kp} outside (0, 10]")
        for name in ("input_timeout_s", "joint_states_timeout_s", "max_input_step_rad",
                     "max_engage_error_rad", "max_stamp_skew_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a positive finite number")
        if not math.isfinite(self.limit_margin_rad) or self.limit_margin_rad < 0.0:
            raise ValueError("limit_margin_rad must be >= 0")
        if not math.isfinite(self.deadband_rad) or self.deadband_rad < 0.0:
            raise ValueError("deadband_rad must be >= 0")
        for name in ("catchup_velocity", "catchup_acceleration", "catchup_done_rad",
                     "max_catchup_error_rad"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a positive finite number")
        if self.catchup_velocity > self.max_velocity:
            raise ValueError(
                f"catchup_velocity {self.catchup_velocity} > max_velocity {self.max_velocity}")
        if self.catchup_acceleration > self.max_acceleration:
            raise ValueError(f"catchup_acceleration {self.catchup_acceleration} > "
                             f"max_acceleration {self.max_acceleration}")
        if not (self.catchup_done_rad < self.max_engage_error_rad
                and self.catchup_done_rad < self.max_input_step_rad):
            raise ValueError("catchup_done_rad must be < max_engage_error_rad and "
                             "< max_input_step_rad")
        if not (self.max_engage_error_rad < self.max_catchup_error_rad <= math.pi):
            raise ValueError(
                f"max_catchup_error_rad {self.max_catchup_error_rad} must be in "
                f"(max_engage_error_rad, pi]")


@dataclass
class Status:
    state: str = DISABLED
    reason: str = "startup"
    max_error: Optional[float] = None
    input_age: Optional[float] = None
    latched: bool = False
    max_command: float = 0.0
    events: List[str] = field(default_factory=list)


def _finite(values: Sequence[float]) -> bool:
    return all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)


class FollowerCore:
    def __init__(self, joint_names: Sequence[str], lower: Sequence[float],
                 upper: Sequence[float], params: Params,
                 input_joint_names: Optional[Sequence[str]] = None):
        """input_joint_names[i] is the name in the mocap message that drives joint_names[i]
        (default: the same names). Other names in the message (e.g. the other arm of a dual-arm
        message, gripper joints) are ignored."""
        params.validate()
        n = len(joint_names)
        if len(set(joint_names)) != n or len(lower) != n or len(upper) != n:
            raise ValueError("joint_names/limits mismatch")
        self.input_names = list(input_joint_names) if input_joint_names else list(joint_names)
        if len(self.input_names) != n or len(set(self.input_names)) != n:
            raise ValueError("input_joint_names must be %d distinct names" % n)
        if not (_finite(lower) and _finite(upper)):
            raise ValueError("joint limits must be finite")
        self.names = list(joint_names)
        self.n = n
        self.p = params
        self.lo = [lo + params.limit_margin_rad for lo in lower]
        self.hi = [hi - params.limit_margin_rad for hi in upper]
        if any(l >= h for l, h in zip(self.lo, self.hi)):
            raise ValueError("limit margin leaves no usable range")
        self.vel = [0.0] * n  # last commanded velocity
        self._last_t: Optional[float] = None
        # input
        self.target: Optional[List[float]] = None
        self.target_valid = False
        self.invalid_reason = "no input"
        self.last_rx: Optional[float] = None
        self._prev_msg: Optional[List[float]] = None
        self.clamped: List[str] = []  # joints whose last target was clamped into the limits
        # mode
        self.enable_prev = False
        self.pending = False
        self.engaged = False
        self.latch: Optional[Tuple[str, str]] = None  # (state, reason)
        self.catchup = False          # slow catch-up in progress (kept across holds)
        self._step_flag: Optional[str] = None  # input step seen; next update enters catch-up
        self._events: List[str] = []

    # -------------------------------------------------------------- input
    def on_input(self, t_recv: float, names: Sequence[str], positions: Sequence[float],
                 stamp_offset_s: Optional[float] = None) -> Optional[str]:
        """Process one mocap message. Returns a rejection reason or None if accepted."""
        self.last_rx = t_recv
        reason = self._check_message(names, positions, stamp_offset_s)
        if reason is not None:
            self.target_valid = False
            self.invalid_reason = reason
            return reason
        by_name = dict(zip(names, positions))
        pos = [float(by_name[n]) for n in self.input_names]
        self.clamped = []
        if self.p.clamp_to_limits:
            # The mocap model's limits are a little wider than the real arm's: follow up to the
            # limit (minus margin) instead of holding. A wrap at +/-pi becomes a big step and is
            # caught by catch-up / max_catchup_error_rad.
            for i, v in enumerate(pos):
                c = min(max(v, self.lo[i]), self.hi[i])
                if c != v:
                    self.clamped.append(self.names[i])
                    pos[i] = c
        prev = self._prev_msg
        self._prev_msg = pos
        if prev is not None and self.enable_prev and self.engaged:
            step = max(abs(a - b) for a, b in zip(pos, prev))
            if step > self.p.max_input_step_rad and self._step_flag is None:
                self._step_flag = (f"input step {step:.3f} rad > "
                                   f"{self.p.max_input_step_rad}")
        self.target = pos
        for i, v in enumerate(pos):
            if v < self.lo[i] or v > self.hi[i]:
                self.target_valid = False
                self.invalid_reason = (f"{self.names[i]} target {v:.3f} outside "
                                       f"[{self.lo[i]:.3f}, {self.hi[i]:.3f}]")
                return self.invalid_reason
        self.target_valid = True
        self.invalid_reason = ""
        return None

    def _check_message(self, names, positions, stamp_offset_s) -> Optional[str]:
        if len(names) != len(positions):
            return "name/position length mismatch"
        if len(set(names)) != len(names):
            return "duplicate joint names"
        missing = [n for n in self.input_names if n not in names]
        if missing:
            return "missing joints " + ",".join(missing)
        by_name = dict(zip(names, positions))
        if not _finite([by_name[n] for n in self.input_names]):
            return "non-finite position"
        if self.p.check_stamp and stamp_offset_s is not None:
            if not math.isfinite(stamp_offset_s) or abs(stamp_offset_s) > self.p.max_stamp_skew_s:
                return f"header.stamp {stamp_offset_s:+.2f} s off local clock"
        return None

    # ------------------------------------------------------------- update
    def update(self, t: float, current: Optional[Sequence[float]], current_age: float,
               enable: bool, blocked: Optional[str] = None) -> Tuple[List[float], Status]:
        dt = 0.0 if self._last_t is None else min(max(t - self._last_t, 0.0), MAX_DT)
        self._last_t = t
        events, self._events = self._events, []

        cur_ok = (current is not None and len(current) == self.n and _finite(current)
                  and current_age <= self.p.joint_states_timeout_s)
        input_age = None if self.last_rx is None else t - self.last_rx
        fresh = input_age is not None and input_age <= self.p.input_timeout_s
        max_err = None
        if cur_ok and self.target is not None:
            max_err = max(abs(a - b) for a, b in zip(self.target, current))

        state, reason = DISABLED, "enable false"
        follow = False
        if not enable:
            if self.enable_prev or self.latch is not None:
                events.append("enable released")
            self.pending = self.engaged = False
            self.latch = None
            self.catchup = False
            self._step_flag = None
        else:
            if not self.enable_prev:
                self.pending, self.engaged = True, False
                self.catchup = False
                self._step_flag = None
                events.append("enable rising edge")
            if self.latch is not None:
                state, reason = self.latch
                self.engaged = self.pending = False
            elif self.pending:
                state, reason = HOLD, "waiting to engage"
                problem = self._condition_problem(cur_ok, fresh, blocked)
                if problem:
                    reason = f"waiting to engage: {problem}"
                elif max_err > self.p.max_catchup_error_rad:
                    self.latch = (REFUSED, f"engage error {max_err:.3f} rad > "
                                           f"{self.p.max_catchup_error_rad}")
                    events.append(self.latch[1])
                    state, reason = self.latch
                    self.pending = False
                elif max_err > self.p.max_engage_error_rad:
                    self.pending, self.engaged = False, True
                    self.catchup = True
                    self._step_flag = None
                    events.append(f"engaged in catch-up (error {max_err:.3f} rad)")
                else:
                    self.pending, self.engaged = False, True
                    self._step_flag = None
                    events.append(f"engaged (error {max_err:.3f} rad)")
            if self.engaged:
                problem = self._condition_problem(cur_ok, fresh, blocked)
                if problem:
                    state, reason = HOLD, problem
                elif max_err > self.p.max_catchup_error_rad:
                    self.latch = (HOLD, f"error {max_err:.3f} rad > "
                                        f"{self.p.max_catchup_error_rad}")
                    events.append(f"LATCH {self.latch[1]}")
                    state, reason = self.latch
                    self.engaged = False
                    self.catchup = False
                    self._step_flag = None
                else:
                    step_reason, self._step_flag = self._step_flag, None
                    if self.catchup and max_err < self.p.catchup_done_rad:
                        self.catchup = False
                        events.append("caught up")
                    elif not self.catchup:
                        if step_reason is not None:
                            self.catchup = True
                            events.append(f"catch-up: {step_reason}")
                        elif max_err > self.p.max_engage_error_rad:
                            self.catchup = True
                            events.append(f"catch-up: error {max_err:.3f} rad > "
                                          f"{self.p.max_engage_error_rad}")
                    follow = True
                    if self.catchup:
                        state, reason = CATCHUP, f"catching up, error {max_err:.3f} rad"
                    else:
                        state, reason = ENABLED, "following"
        self.enable_prev = enable

        desired = [0.0] * self.n
        if follow and self.catchup:
            desired = self._catchup_velocities(
                [self.target[i] - current[i] for i in range(self.n)])
            accel = self.p.catchup_acceleration
        elif follow:
            for i in range(self.n):
                desired[i] = self._desired_velocity(self.target[i] - current[i])
            accel = self.p.max_acceleration
        else:
            accel = self.p.stop_deceleration
        self._apply_accel(desired, accel, dt, sync=follow and self.catchup)
        status = Status(state, reason, max_err, input_age, self.latch is not None,
                        max(abs(v) for v in self.vel), events)
        return list(self.vel), status

    def _condition_problem(self, cur_ok, fresh, blocked) -> Optional[str]:
        if not cur_ok:
            return "joint_states missing/stale/non-finite"
        if blocked:
            return blocked
        if self.target is None:
            return "no mocap input"
        if not fresh:
            return "mocap input stale"
        if not self.target_valid:
            return f"invalid mocap input: {self.invalid_reason}"
        return None

    def _desired_velocity(self, error: float) -> float:
        a = abs(error)
        if a < self.p.deadband_rad:
            return 0.0
        v = min(self.p.kp * a, self.p.max_velocity,
                math.sqrt(2.0 * 0.8 * self.p.max_acceleration * a))
        return math.copysign(v, error)

    def _catchup_velocities(self, errors):
        """Synchronized straight-line joint-space motion: all joints arrive together."""
        emax = max(abs(e) for e in errors)
        if emax < self.p.deadband_rad:
            return [0.0] * self.n
        s = min(self.p.catchup_velocity, self.p.kp * emax,
                math.sqrt(2.0 * 0.8 * self.p.catchup_acceleration * emax))
        return [s * e / emax for e in errors]

    def _apply_accel(self, desired, accel, dt, sync=False):
        step = accel * dt
        if sync:
            # scale the whole velocity change uniformly so the joints stay on the
            # straight line (per-joint |dv| <= step still holds)
            dvs = [d - v for d, v in zip(desired, self.vel)]
            biggest = max(abs(d) for d in dvs)
            k = 1.0 if biggest <= step else step / biggest
            for i in range(self.n):
                self.vel[i] += dvs[i] * k
                if abs(self.vel[i]) < 1e-9:
                    self.vel[i] = 0.0
            return
        for i in range(self.n):
            dv = desired[i] - self.vel[i]
            dv = max(-step, min(step, dv))
            self.vel[i] += dv
            if abs(self.vel[i]) < 1e-9:
                self.vel[i] = 0.0
