"""ROS-free logic of the dual-arm teleop selector (tested in test/test_selector_core.py).

The selector sits between the single-arm keyboard / gamepad nodes (which publish unprefixed
``base_link`` / ``xeg32_tool_tip`` twists and ``joint1..6`` jogs) and the two MoveIt Servo instances of
the dual cell. This module decides, for every incoming command and every clock tick:

* which arm (if any) gets the command, translated to that arm's frames and joint names and scaled/clamped;
* which Servo instances should be unpaused (only the selected arm, only while nothing blocks teleop);
* which zero commands must be sent (to the old arm when the selection changes, and to an arm whose teleop
  gets blocked), so an arm never keeps coasting on its last non-zero command.

TOUCH MODE (explicit, off by default): for the SELECTED arm only, the knife links may touch the carton and the table
(ACM entries in the scene fed to that arm's Servo); everything else stays checked. Speed is reduced and it turns
itself off on arm switch, teleop disable, any coordinator state other than IDLE/STOPPED/unknown, an inactive
controller, a stale scene, or a timeout. See ``allow_touch_pairs`` and ``SelectorCore.set_touch``.

Interlocks (``block_reasons``): teleop disabled (start state), coordinator executing, the selected arm's
controller not active, the selected arm's Servo not confirmed running. The coordinator state "unknown"
(no CoordinatorStatus at all, or a stale one that was not blocking) is NOT a block: it is reported as a
warning. A stale status whose last known state was blocking stays blocked (the coordinator may have died
mid-phase) until ``ignore_stale_coordinator`` is set.
"""

import math
from dataclasses import dataclass, field, replace
from typing import List, Optional, Tuple

ARMS = ("left", "right")

STATE_IDLE = 0
STATE_PLANNING = 1
STATE_PLANNED = 2
STATE_WAITING_BARRIER = 3
STATE_EXECUTING = 4
STATE_STOPPED = 5
STATE_NAMES = {0: "IDLE", 1: "PLANNING", 2: "PLANNED", 3: "WAITING_BARRIER", 4: "EXECUTING", 5: "STOPPED"}

JOINT_SUFFIXES = tuple(f"joint{n}" for n in range(1, 7))
SINGLE_ARM_BASE_FRAME = "base_link"
SINGLE_ARM_TOOL_FRAME = "xeg32_tool_tip"


def arm_of(link_name):
    """'left' / 'right' for a link named arm_left_* / arm_right_*, else None (table, world objects)."""
    for arm in ARMS:
        if link_name.startswith(f"arm_{arm}_"):
            return arm
    return None


def allow_same_arm_pairs(names, rows):
    """ACM rows (list of lists of bool) with every pair of links of the SAME arm allowed.

    MoveIt Servo's proximity check treats any non-ACM pair as a hazard, including pairs that are close by
    construction (wrist camera / holder vs link5: 19 mm at home) -- the arm then crawls everywhere. Servo's job
    here is the other arm, the table and the scene objects; same-arm pairs stay covered by MoveIt planning.
    Cross-arm, arm-vs-table and arm-vs-world-object entries are left exactly as move_group has them.
    """
    out = [list(r) for r in rows]
    arms = [arm_of(n) for n in names]
    for i, ai in enumerate(arms):
        for j, aj in enumerate(arms):
            if ai is not None and ai == aj:
                out[i][j] = True
    return out


# The knife assembly of the dual model. knife_c_holder is included although the task listed only blade / handpiece /
# tool tip: the holder sits ~6 mm behind the blade tip, and Servo's exponential proximity slow-down on the holder alone
# stalls the blade ~20 mm above the carton (mock gate A, results/gateA_carton.json, first run). The wrist camera and
# link6 and the rest of the arm stay checked.
KNIFE_LINK_SUFFIXES = ("knife_c_blade", "knife_c_handpiece", "knife_c_holder", "xeg32_tool_tip")
# Additionally allowed during the RETREAT window only: the rest of the tool end. At knife contact link6 and the wrist
# camera are ~25 mm from the carton, and their (direction-blind, exponential) slow-down alone lets the arm back off at
# only ~0.4 mm/s (mock gate A, first retreat run). link1..link5 and the motors stay checked.
RETREAT_EXTRA_SUFFIXES = ("link6", "knife_c_wrist_camera")
TABLE_LINK = "table"


def knife_links(arm):
    return [f"arm_{arm}_{s}" for s in KNIFE_LINK_SUFFIXES]


def allow_touch_pairs(names, rows, arm, carton_ids, retreat=False):
    """ACM (names, rows) with the knife links of ``arm`` allowed against the carton objects and the table.

    Exactly these pairs and nothing else: not the rest of the arm (link1..6, motors, holder, wrist camera), not the
    other arm's knife, not the camera pole / head. ``carton_ids`` that are not in the matrix yet are appended
    (a world object only has an ACM row once something allowed it). The table is a link of the dual model.
    Inputs are not modified. ``retreat`` (window after touch mode) also allows link6 and the wrist camera.
    """
    names = list(names)
    rows = [list(r) for r in rows]
    for extra in list(carton_ids):
        if extra not in names:
            names.append(extra)
            for r in rows:
                r.append(False)
            rows.append([False] * len(names))
    index = {n: i for i, n in enumerate(names)}
    links = knife_links(arm) + ([f"arm_{arm}_{x}" for x in RETREAT_EXTRA_SUFFIXES] if retreat else [])
    for knife in links:
        if knife not in index:
            continue  # link without an ACM row (e.g. no such link in this model)
        for target in [*carton_ids, TABLE_LINK]:
            if target in index:
                rows[index[knife]][index[target]] = True
                rows[index[target]][index[knife]] = True
    return names, rows


def other(arm):
    return "right" if arm == "left" else "left"


@dataclass
class Command:
    """A teleop command: kind 'twist' (frame + linear + angular) or 'joint' (names + velocities)."""

    kind: str
    frame: str = ""
    linear: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    joint_names: List[str] = field(default_factory=list)
    velocities: List[float] = field(default_factory=list)
    arm: Optional[str] = None  # set on outputs


def is_zero(cmd):
    if cmd.kind == "twist":
        return all(v == 0.0 for v in (*cmd.linear, *cmd.angular))
    return all(v == 0.0 for v in cmd.velocities)


def is_finite(cmd):
    values = (*cmd.linear, *cmd.angular) if cmd.kind == "twist" else tuple(cmd.velocities)
    return all(math.isfinite(v) for v in values)


@dataclass
class Config:
    speed_scale: float = 0.5
    max_linear_mps: float = 0.05
    max_angular_rps: float = 0.3
    max_joint_rps: float = 0.1
    base_command_frame: str = "world"  # target of "base_link": "world" or "arm_base" (arm_X_base_link)
    block_states: Tuple[int, ...] = (STATE_WAITING_BARRIER, STATE_EXECUTING)
    coordinator_timeout_s: float = 5.0
    controller_timeout_s: float = 3.0
    neutral_gap_s: float = 0.3
    pause_delay_s: float = 0.3
    scene_timeout_s: float = 5.0
    touch_speed_scale: float = 0.1667  # about 1/3 of the normal speed_scale; never above it (see effective_scale)
    touch_mode_timeout_s: float = 120.0  # 0 = no timeout
    touch_ok_states: Tuple[int, ...] = (STATE_IDLE, STATE_STOPPED)
    # RETREAT assist after touch mode is turned off by the operator / the timeout / a planning coordinator: Servo's
    # slow-down is exponential (x0.001 at contact), so with the normal ACM an arm left touching the carton can only
    # back off at ~0.02 mm/s. For the retreat window the knife pairs stay allowed (everything else stays checked),
    # at touch speed, until the TCP is retreat_distance_m from where it was at OFF, or retreat_window_s passed.
    retreat_distance_m: float = 0.08
    retreat_window_s: float = 60.0
    require_scene: bool = True  # Servo's collision scene comes from this node; stale -> blocked


def base_frame_for(arm, mode):
    return f"arm_{arm}_base_link" if mode == "arm_base" else "world"


class SelectorCore:
    def __init__(self, config=None, default_arm="left", enabled=False):
        if default_arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}, got {default_arm!r}")
        self.cfg = config or Config()
        self.selected = default_arm
        self.enabled = bool(enabled)
        # coordinator
        self.coord_state = None  # last CoordinatorStatus.state
        self.coord_time = None  # time of the last status
        self.ignore_stale = False
        # per arm
        self.controller_active = {a: None for a in ARMS}  # None = unknown
        self.controller_time = {a: None for a in ARMS}
        self.servo_running = {a: False for a in ARMS}  # confirmed unpaused
        self.scene_time = None  # last Servo planning scene published from move_group's
        # command stream
        self.last_input_time = None
        self.needs_neutral = True
        self._was_open = {a: False for a in ARMS}
        self.closed_since = {a: None for a in ARMS}
        self.last_drop_reason = ""
        # touch mode
        self.touch_arm = None  # arm that has touch mode ON, else None
        self.touch_since = None
        self.touch_off_reason = ""
        self.retreat_arm = None
        self.retreat_since = None
        self.retreat_origin = None
        self.tcp = {a: None for a in ARMS}  # latest world position of each TCP (from TF), for the retreat window

    # ----- inputs from the node -----
    def update_tcp(self, arm, xyz):
        self.tcp[arm] = tuple(xyz)

    def update_coordinator(self, state, now):
        self.coord_state = int(state)
        self.coord_time = now

    def update_controller(self, arm, active, now):
        self.controller_active[arm] = bool(active)
        self.controller_time[arm] = now

    def update_scene(self, now):
        self.scene_time = now

    def set_servo_running(self, arm, running):
        self.servo_running[arm] = bool(running)

    # ----- views -----
    def coordinator_view(self, now):
        """(label, blocking, warning) for the coordinator interlock."""
        if self.coord_time is None:
            return "UNKNOWN (no CoordinatorStatus received; coordinator not running?)", False, True
        age = now - self.coord_time
        blocking_state = self.coord_state in self.cfg.block_states
        name = STATE_NAMES.get(self.coord_state, str(self.coord_state))
        if age <= self.cfg.coordinator_timeout_s:
            return name, blocking_state, False
        if blocking_state and not self.ignore_stale:
            return f"STALE {age:.0f} s, last state {name}", True, False
        return f"UNKNOWN (status stale {age:.0f} s, last state {name})", False, True

    def controller_view(self, arm, now):
        active = self.controller_active[arm]
        if active is None or self.controller_time[arm] is None:
            return False
        if now - self.controller_time[arm] > self.cfg.controller_timeout_s:
            return False  # poll not answering: fail closed
        return active

    def global_block_reasons(self, now):
        reasons = []
        if not self.enabled:
            reasons.append("teleop disabled (start state; enable it explicitly)")
        label, blocking, _ = self.coordinator_view(now)
        if blocking:
            reasons.append(f"coordinator {label}: teleop of BOTH arms blocked")
        if self.cfg.require_scene and (self.scene_time is None or now - self.scene_time > self.cfg.scene_timeout_s):
            reasons.append("no fresh planning scene for Servo collision checking (is move_group running?)")
        return reasons

    def arm_block_reasons(self, arm, now):
        reasons = self.global_block_reasons(now)
        if not self.controller_view(arm, now):
            reasons.append(f"arm_{arm}_trajectory_controller not active (or state unknown)")
        if not self.servo_running[arm]:
            reasons.append(f"arm_{arm} Servo not running (stopped or not yet settled)")
        return reasons

    def warnings(self, now):
        _, _, warn = self.coordinator_view(now)
        return ["coordinator status unknown: teleop is NOT interlocked against a running cut"] if warn else []

    def is_open(self, arm, now):
        return arm == self.selected and not self.arm_block_reasons(arm, now)

    def desired_unpaused(self, arm, now):
        """Servo of ``arm`` should run: selected, enabled, not blocked by coordinator/controller."""
        if arm != self.selected or self.global_block_reasons(now):
            return False
        return self.controller_view(arm, now)

    def pause_due(self, arm, now):
        """True when ``arm``'s Servo should be paused now (zeros had pause_delay_s to take effect)."""
        if self.desired_unpaused(arm, now):
            return False
        since = self.closed_since[arm]
        return since is None or now - since >= self.cfg.pause_delay_s

    # ----- transitions -----
    def zero_outputs(self, arm):
        base = base_frame_for(arm, self.cfg.base_command_frame)
        return [
            Command("twist", frame=base, arm=arm),
            Command("joint", joint_names=[f"arm_{arm}_{s}" for s in JOINT_SUFFIXES],
                    velocities=[0.0] * 6, arm=arm),
        ]

    def select(self, arm, now):
        """Change the selected arm; returns zero commands for the old arm (sent before anything else)."""
        if arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
        out = []
        if arm != self.selected:
            out = self.zero_outputs(self.selected)
            self._touch_off("arm switched", now)
            self._end_retreat()
            self.selected = arm
            self.needs_neutral = True
            self._was_open[other(arm)] = False
            self.closed_since[other(arm)] = now
        return out

    def set_enabled(self, enabled, now):
        enabled = bool(enabled)
        out = []
        if self.enabled and not enabled:
            out = self.zero_outputs(self.selected)
            self._touch_off("teleop disabled", now)
            self._end_retreat()
            self._was_open[self.selected] = False
            self.closed_since[self.selected] = now
        if enabled and not self.enabled:
            self.needs_neutral = True
        self.enabled = enabled
        return out

    def refresh(self, now):
        """Track open/closed transitions; returns zero outputs for an arm that just closed."""
        out = []
        if self.touch_arm is not None:
            why = self.touch_violation(now)
            if why:
                arm = self.touch_arm
                soft = why.startswith("timeout") or why.startswith("coordinator PLAN")
                self._touch_off(why, now, retreat=soft)
                out += self.zero_outputs(arm)
                self.needs_neutral = True
        self._update_retreat(now)
        for arm in ARMS:
            is_open = self.is_open(arm, now)
            if is_open and not self._was_open[arm]:
                self.needs_neutral = True  # a command held through the block must not resume
                self.closed_since[arm] = None
            elif not is_open and self._was_open[arm]:
                out += self.zero_outputs(arm)
                self.closed_since[arm] = now
            elif not is_open and self.closed_since[arm] is None:
                self.closed_since[arm] = now
            self._was_open[arm] = is_open
        return out

    # ----- touch mode -----
    @property
    def touch_on(self):
        return self.touch_arm is not None

    def hard_violation(self, arm, now):
        """Conditions that end touch mode AND its retreat window at once (the arm is blocked / the cell is active)."""
        if not self.enabled:
            return "teleop disabled"
        if arm != self.selected:
            return "arm switched"
        label, blocking, _ = self.coordinator_view(now)
        if blocking:
            return f"coordinator {label}"
        for a in ARMS:
            if not self.controller_view(a, now):
                return f"arm_{a}_trajectory_controller not active"
        if self.cfg.require_scene and (self.scene_time is None or now - self.scene_time > self.cfg.scene_timeout_s):
            return "no fresh planning scene"
        return ""

    def touch_violation(self, now, entering=False):
        """Reason touch mode may not be (or stay) ON, or '' when it is allowed."""
        arm = self.touch_arm if self.touch_arm is not None else self.selected
        why = self.hard_violation(arm, now)
        if why:
            return why
        fresh = self.coord_time is not None and now - self.coord_time <= self.cfg.coordinator_timeout_s
        if fresh and self.coord_state not in self.cfg.touch_ok_states:
            return f"coordinator {STATE_NAMES.get(self.coord_state, self.coord_state)}"
        if (not entering and self.cfg.touch_mode_timeout_s > 0 and self.touch_since is not None
                and now - self.touch_since > self.cfg.touch_mode_timeout_s):
            return f"timeout ({self.cfg.touch_mode_timeout_s:.0f} s)"
        return ""

    def _touch_off(self, reason, now, retreat=False):
        arm = self.touch_arm
        if arm is not None:
            self.touch_off_reason = f"{reason} (touch mode was ON for {arm})"
        self.touch_arm = None
        self.touch_since = None
        self._end_retreat()
        if arm is not None and retreat and self.hard_violation(arm, now) == "":
            self.retreat_arm, self.retreat_since, self.retreat_origin = arm, now, self.tcp.get(arm)

    def _end_retreat(self):
        self.retreat_arm = self.retreat_since = self.retreat_origin = None

    @property
    def acm_touch_arm(self):
        """Arm whose Servo scene gets the knife allowances: touch mode ON or its retreat window."""
        return self.touch_arm if self.touch_arm is not None else self.retreat_arm

    def _update_retreat(self, now):
        arm = self.retreat_arm
        if arm is None:
            return
        why = self.hard_violation(arm, now)
        if not why and now - self.retreat_since > self.cfg.retreat_window_s:
            why = "retreat window over"
        if not why and self.retreat_origin is not None and self.tcp.get(arm) is not None:
            if math.dist(self.tcp[arm], self.retreat_origin) >= self.cfg.retreat_distance_m:
                why = "retreated far enough"
        if why:
            self.touch_off_reason = f"retreat ended: {why}"
            self._end_retreat()

    def set_touch(self, on, now):
        """Turn touch mode ON for the selected arm / OFF. Returns (ok, message, zero outputs)."""
        if not on:
            if self.touch_arm is None:
                return True, "touch mode already OFF", []
            arm = self.touch_arm
            self._touch_off("turned off by the operator", now, retreat=True)
            self.needs_neutral = True
            return True, "touch mode OFF", self.zero_outputs(arm)
        why = self.touch_violation(now, entering=True)
        if why:
            return False, f"touch mode refused: {why}", []
        if self.touch_arm == self.selected:
            return True, "touch mode already ON", []
        self._end_retreat()
        self.touch_arm = self.selected
        self.touch_since = now
        self.touch_off_reason = ""
        self.needs_neutral = True
        return True, f"touch mode ON for {self.selected}", self.zero_outputs(self.selected)

    def effective_scale(self, arm=None):
        """Speed scale in force for ``arm``: touch_speed_scale (never above speed_scale) while touch mode is ON for it."""
        if self.acm_touch_arm is not None and (arm is None or arm == self.acm_touch_arm):
            return min(self.cfg.touch_speed_scale, self.cfg.speed_scale)
        return self.cfg.speed_scale

    # ----- command path -----
    def handle_command(self, cmd, now):
        """Process one incoming command. Returns (outputs, drop_reason); drop_reason '' when forwarded."""
        outputs = self.refresh(now)
        gap = None if self.last_input_time is None else now - self.last_input_time
        self.last_input_time = now

        def drop(reason):
            self.last_drop_reason = reason
            return outputs, reason

        if not is_finite(cmd):
            return drop("non-finite command")
        arm = self.selected
        reasons = self.arm_block_reasons(arm, now)
        if reasons:
            return drop("blocked: " + "; ".join(reasons))
        if self.needs_neutral:
            if is_zero(cmd) or (gap is not None and gap >= self.cfg.neutral_gap_s):
                self.needs_neutral = False
            else:
                return drop("waiting for a neutral (zero or new) command after select/enable/unblock")
        mapped = self.translate(cmd, arm)
        if isinstance(mapped, str):
            return drop(mapped)
        outputs.append(self.scale(mapped))
        self.last_drop_reason = ""
        return outputs, ""

    def translate(self, cmd, arm):
        """Command in single-arm names -> the arm's frames/joint names, or a drop-reason string."""
        out = replace(cmd, arm=arm)
        if cmd.kind == "twist":
            frame = cmd.frame
            if frame in ("", SINGLE_ARM_BASE_FRAME):
                out.frame = base_frame_for(arm, self.cfg.base_command_frame)
            elif frame == SINGLE_ARM_TOOL_FRAME:
                out.frame = f"arm_{arm}_{SINGLE_ARM_TOOL_FRAME}"
            elif frame in ("world", f"arm_{arm}_{SINGLE_ARM_TOOL_FRAME}", f"arm_{arm}_base_link"):
                out.frame = frame
            else:
                return f"unknown or foreign command frame {frame!r}"
            return out
        if len(cmd.joint_names) != len(cmd.velocities):
            return "joint jog with mismatched names/velocities"
        names = []
        for name in cmd.joint_names:
            if name in JOINT_SUFFIXES:
                names.append(f"arm_{arm}_{name}")
            elif name.startswith(f"arm_{arm}_") and name[len(f"arm_{arm}_"):] in JOINT_SUFFIXES:
                names.append(name)
            else:
                return f"joint {name!r} does not belong to arm_{arm}"
        out.joint_names = names
        return out

    def scale(self, cmd):
        """Apply speed_scale, then clamp the linear / angular norm and every joint velocity."""
        s = self.effective_scale(cmd.arm)
        if cmd.kind == "twist":
            lin = [v * s for v in cmd.linear]
            ang = [v * s for v in cmd.angular]
            for vec, limit in ((lin, self.cfg.max_linear_mps), (ang, self.cfg.max_angular_rps)):
                norm = math.sqrt(sum(v * v for v in vec))
                if norm > limit > 0.0:
                    k = limit / norm
                    vec[:] = [v * k for v in vec]
            cmd.linear, cmd.angular = tuple(lin), tuple(ang)
        else:
            lim = self.cfg.max_joint_rps
            cmd.velocities = [max(-lim, min(lim, v * s)) for v in cmd.velocities]
        return cmd

    def tick(self, now):
        return self.refresh(now)

    # ----- status for the screen -----
    def status(self, now):
        label, blocking, _ = self.coordinator_view(now)
        reasons = self.arm_block_reasons(self.selected, now)
        if reasons and not self.enabled:
            state = "PAUSED"
        elif reasons:
            state = "BLOCKED"
        else:
            state = "ACTIVE"
        return {
            "selected": self.selected,
            "enabled": self.enabled,
            "state": state,
            "reasons": reasons,
            "warnings": self.warnings(now),
            "coordinator": label,
            "coordinator_blocking": blocking,
            "controllers": {a: self.controller_view(a, now) for a in ARMS},
            "servo_running": dict(self.servo_running),
            "speed_scale": self.cfg.speed_scale,
            "touch": {
                "on": self.touch_on,
                "arm": self.touch_arm,
                "speed_scale": self.effective_scale(self.acm_touch_arm) if self.acm_touch_arm else None,
                "remaining_s": (round(max(0.0, self.cfg.touch_mode_timeout_s - (now - self.touch_since)), 1)
                                if self.touch_on and self.cfg.touch_mode_timeout_s > 0 else None),
                "retreat": self.retreat_arm is not None,
                "retreat_arm": self.retreat_arm,
                "last_off_reason": self.touch_off_reason,
            },
            "last_drop": self.last_drop_reason,
        }
