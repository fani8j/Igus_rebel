#!/usr/bin/env python3
"""Operator CLI for the dual ReBeL cell: status, fault recovery and controller activation.

    ros2 run dual_cut_bringup dual_arm_recover.py status
    ros2 run dual_cut_bringup dual_arm_recover.py recover [--root left|right|auto] [--log-file FILE] [--dry-run]
    ros2 run dual_cut_bringup dual_arm_recover.py activate [--yes]

Subcommands
-----------
status    Hardware components, controllers, the fault/latch information the driver exposes, joint-state
          freshness and stillness, and the coordinator status. Read only.
recover   Recover the arms whose hardware component is not ACTIVE (a latched fault group stops both arms) or
          is ACTIVE with disabled drives. Phase 1, per arm, the ROOT-CAUSE arm first, then the other:
          - hardware NOT active (latched): [reset_and_enable (best effort), set_hardware_component_state inactive,
            then active (a peer retries for ~6 s while the root clears)], THEN deactivate its trajectory controller
            if it is still active. Humble's controller_manager does not deactivate controllers on a hardware error
            and REJECTS a deactivation while the hardware is not active ("Not available: .../velocity"), so the
            controller can only be deactivated after the hardware is active again. Until the controller is
            restarted the driver keeps the transmitted jog at ZERO (Rebel.cpp controllerRestartHold_), so the stale
            controller cannot move the arm in between, and reset_and_enable does not release that hold.
          - hardware ACTIVE with disabled drives: deactivate the controller FIRST (it is allowed), then re-enable.
          Then ensure the drives are ENABLED (reset_and_enable until the driver replies enabled, up to ~5 s), and for
          the root arm wait until the driver reports its root-cause fault cleared. Phase 2, only after ALL hardware
          is recovered: every arm stands still, ONE identity confirmation, then the trajectory controllers are
          ACTIVATED (the activation is what releases the driver's jog hold). Every step is verified; the first
          failure stops with a message and no controller is activated on an arm whose drives are not confirmed
          enabled. The coordinator is NEVER rearmed here: the operator does that on purpose.
activate  Activate both trajectory controllers after printing both arms' joint positions and the operator
          typing ``yes`` (identity check). ``--yes`` skips the prompt in mock only; it is refused when any
          hardware component is not a mock component.

Signals used (src/igus_rebel Rebel.cpp):
  drives enabled   the reply of /igus_rebel_<arm>/reset_and_enable: the driver re-sends SetActive/Reset/Enable (jog
                   held at zero) and waits up to 3 s inside the call for STATUS 'Motor not enabled' to clear;
                   success=True means enabled. Synchronous, independent of /rosout timing. Needed because after an
                   e-stop the Enable sent by the lifecycle activation can be lost (motor supply not back yet).
  root cleared     only the /rosout line "...'arm_X_hw' is enabled with clean STATUS again; its root-cause fault is
                   cleared" (500 ms of continuous enabled, clean STATUS; kFaultArmSettleTime). The buffer position is
                   taken BEFORE the arm is re-activated so the line is not missed.
  drives disabled  (status / planning only, advisory) the driver's 10 s throttled /rosout warning "Robot drives
                   disabled ... /igus_rebel_<arm>/reset_and_enable"; a hardware-ACTIVE arm with that as latest
                   drive-state line is flagged and re-enabled by recover.

What the driver exposes about a fault (src/igus_rebel Rebel.cpp): NO topic and NO service. A latched component
goes to a non-ACTIVE lifecycle state; the root cause is only in the driver log text (the node's /rosout
messages, which the middleware keeps for about 10 s, and the launch / journal output). ``--root auto`` reads
/rosout and an optional ``--log-file``; when the root cannot be determined and both arms need recovery,
``--root left|right`` is required.

Exit codes: 0 ok, 1 a step failed or the cell is unhealthy (status), 2 refused / bad usage.
"""
import argparse
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field

ARMS = ("left", "right")
OTHER = {"left": "right", "right": "left"}
JOINT_STATE_BROADCASTER = "joint_state_broadcaster"
CONTROLLER_MANAGER = "/controller_manager"
COORDINATOR_STATUS_TOPIC = "/dual_cut_coordinator/status"
COORDINATOR_REARM_SERVICE = "/dual_cut_coordinator/rearm"
# /rosout substrings worth keeping (driver fault-group, latch and drive-state lines).
ROSOUT_KEYS = ("Fault group", "fault-group", "latched error", "Robot drives disabled", "drives enabled")

# Lifecycle primary-state ids (lifecycle_msgs/State) as the controller_manager reports them.
LIFECYCLE = {0: "unknown", 1: "unconfigured", 2: "inactive", 3: "active", 4: "finalized",
             10: "configuring", 11: "cleaning-up", 12: "shutting-down", 13: "activating",
             14: "deactivating", 15: "error-processing"}
STATE_ID = {name: ident for ident, name in LIFECYCLE.items() if ident <= 4}
COORDINATOR_STATES = {0: "IDLE", 1: "PLANNING", 2: "PLANNED", 3: "WAITING_BARRIER", 4: "EXECUTING", 5: "STOPPED"}

# Standing-still criteria (the coordinator's start barrier uses 0.01 rad / 0.03 rad/s / 0.5 s).
STILL_WINDOW_S = 0.8
STILL_MIN_SAMPLES = 5
STILL_MAX_DISPLACEMENT_RAD = 0.005
STILL_MAX_VELOCITY = 0.03
STILL_MAX_AGE_S = 0.5

# The driver accepts a peer only after the root cause has had 500 ms of clean STATUS (kFaultArmSettleTime).
PEER_SETTLE_S = 0.8
# Peer activation is retried for this long: root re-enable (~1 s) + 500 ms clean run + margin.
PEER_ACTIVATE_WINDOW_S = 6.0
# After a hardware component is ACTIVE its drives must report enabled: reset_and_enable is re-tried this long
# (each call itself waits up to 3 s inside the driver for 'Motor not enabled' to disappear).
ENABLE_TIMEOUT_S = 5.0
ENABLE_RETRY_PAUSE_S = 0.5
# The root-cause arm must see 500 ms of continuous enabled, clean STATUS before its fault is cleared; the
# clock restarts with every reset_and_enable, so allow generous margin.
ROOT_CLEAR_TIMEOUT_S = 1.5

# Injectable for tests.
_now = time.monotonic
_sleep = time.sleep


def hw_name(arm):
    return f"arm_{arm}_hw"


def controller_name(arm):
    return f"arm_{arm}_trajectory_controller"


def reset_service(arm):
    return f"/igus_rebel_{arm}/reset_and_enable"


def joint_names(arm):
    return [f"arm_{arm}_joint{n}" for n in range(1, 7)]


def arm_of_component(name):
    for arm in ARMS:
        if name == hw_name(arm):
            return arm
    return None


class Refused(Exception):
    """The command is refused by a safety rule (exit code 2)."""


class StepFailed(Exception):
    """A verified step failed (exit code 1)."""


# ---------------------------------------------------------------------------------------------------------
# pure logic (tested by test/test_dual_arm_recover.py)
# ---------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class FaultEvent:
    kind: str          # ROOT | PEER | BLOCKING | RECOVERED | CLEARED | DRIVES_DISABLED | DRIVES_ENABLED
    component: str     # the component the message is about
    root: str          # the component named as root cause ('' for RECOVERED / CLEARED)
    text: str


_RE_ROOT = re.compile(r"Fault group '([^']*)': component '([^']*)' faulted \(")
_RE_PEER = re.compile(r"Hardware component '([^']*)' stopped because fault-group member '([^']*)' faulted")
_RE_BLOCKING = re.compile(
    r"Refusing to activate hardware component '([^']*)': fault-group '[^']*' member '([^']*)' still holds")
_RE_RECOVERED = re.compile(r"Hardware component '([^']*)' recovered from a latched error")
_RE_CLEARED = re.compile(r"Fault group '[^']*': '([^']*)' is enabled with clean STATUS again")
# Driver read(): "Robot drives disabled (...) ... Call 'ros2 service call /igus_rebel_right/reset_and_enable ...'"
_RE_DRIVES_DISABLED = re.compile(r"Robot drives disabled .*?/igus_rebel_(left|right)/reset_and_enable")
# These two lines carry no component name; the logger name ("igus_rebel.arm_right_hw") identifies it.
_RE_DRIVES_ENABLED = re.compile(
    r"igus_rebel\.(arm_[a-z]+_hw).*(?:drives enabled and STATUS clean; monitoring armed|"
    r"start-up sequence re-sent; drives enabled)")


def parse_fault_events(lines):
    """Driver fault-group log lines (any surrounding text is ignored) -> [FaultEvent] in order of appearance."""
    events = []
    for line in lines:
        text = line.strip()
        match = _RE_ROOT.search(text)
        if match:
            events.append(FaultEvent("ROOT", match.group(2), match.group(2), text))
            continue
        match = _RE_PEER.search(text)
        if match:
            events.append(FaultEvent("PEER", match.group(1), match.group(2), text))
            continue
        match = _RE_BLOCKING.search(text)
        if match:
            events.append(FaultEvent("BLOCKING", match.group(1), match.group(2), text))
            continue
        match = _RE_RECOVERED.search(text)
        if match:
            events.append(FaultEvent("RECOVERED", match.group(1), "", text))
            continue
        match = _RE_CLEARED.search(text)
        if match:
            events.append(FaultEvent("CLEARED", match.group(1), "", text))
            continue
        match = _RE_DRIVES_DISABLED.search(text)
        if match:
            events.append(FaultEvent("DRIVES_DISABLED", hw_name(match.group(1)), "", text))
            continue
        match = _RE_DRIVES_ENABLED.search(text)
        if match:
            events.append(FaultEvent("DRIVES_ENABLED", match.group(1), "", text))
    return events


def active_root_causes(events):
    """Components currently named as an un-recovered root cause, oldest first."""
    roots = []
    for event in events:
        if event.kind in ("ROOT", "PEER", "BLOCKING"):
            if event.root in roots:
                roots.remove(event.root)   # keep the most recent position
            roots.append(event.root)
        elif event.kind in ("RECOVERED", "CLEARED") and event.component in roots:
            roots.remove(event.component)
    return roots


def detected_root_arms(events):
    """Arms (left/right) whose component is an un-recovered root cause per the log events."""
    arms = []
    for component in active_root_causes(events):
        arm = arm_of_component(component)
        if arm and arm not in arms:
            arms.append(arm)
    return arms


def root_cleared(events, component):
    """True when the driver reported the root-cause fault of ``component`` cleared (and not faulted again since)."""
    cleared = False
    for event in events:
        if event.kind == "CLEARED" and event.component == component:
            cleared = True
        elif event.kind in ("ROOT", "PEER") and event.root == component:
            cleared = False
    return cleared


def drives_disabled_arms(events):
    """Arms whose latest drive-state line in the log says 'Robot drives disabled' (advisory: /rosout only).

    The driver warns every 10 s while the drives are disabled; it logs an 'enabled' line only after a successful
    reset_and_enable or once the fault monitor armed. A line up to ~10 s old can therefore be stale right after a
    recovery; ``recover`` verifies with the reset_and_enable reply, which is the authoritative signal.
    """
    state = {}
    for event in events:
        if event.kind == "DRIVES_DISABLED":
            state[event.component] = True
        elif event.kind in ("DRIVES_ENABLED", "CLEARED"):
            state[event.component] = False
    return [arm for arm in ARMS if state.get(hw_name(arm))]


@dataclass
class ArmView:
    arm: str
    hw_state: str = None            # lifecycle name from LIFECYCLE, None = component missing
    hw_class: str = None            # plugin class, e.g. Igus/Rebel or mock_components/GenericSystem
    controller_state: str = None    # controller_manager state string, None = controller missing
    drives_disabled: bool = False   # driver log says 'Robot drives disabled' (hardware may still be ACTIVE)


def is_mock_class(class_type):
    return bool(class_type) and class_type.startswith("mock_components/")


def yes_allowed(views):
    """(ok, reason): --yes (skip the operator prompt) is allowed only when every hardware component is mock."""
    classes = {view.arm: view.hw_class for view in views}
    for arm in ARMS:
        if arm not in classes or not classes[arm]:
            return False, f"hardware class of arm_{arm} unknown; cannot prove it is mock"
    bad = [f"arm_{arm}: {classes[arm]}" for arm in ARMS if not is_mock_class(classes[arm])]
    if bad:
        return False, ("--yes refused: not a mock component (" + ", ".join(bad) + "); on real hardware the "
                       "operator must type 'yes' after checking the arm identity")
    return True, ""


@dataclass
class RecoveryPlan:
    order: list = field(default_factory=list)     # arms to recover, root first
    root: str = None
    notes: list = field(default_factory=list)


def controller_active_without_hardware(view):
    """True for the Humble post-error state: trajectory controller ACTIVE while its hardware is not ACTIVE.

    The controller_manager does not deactivate controllers on a hardware error and refuses to deactivate one while
    the hardware is not active. After the hardware is recovered the driver keeps jog at zero until that controller
    is restarted.
    """
    return view.controller_state == "active" and view.hw_state not in (None, "active")


def needs_enable_check(view):
    """Hardware ACTIVE on a real driver but the drives are reported disabled."""
    return view.hw_state == "active" and view.drives_disabled and not is_mock_class(view.hw_class)


def plan_recovery(views, requested="auto", detected=()):
    """Decide which arms to recover and in which order.

    views      {arm: ArmView} for both arms
    requested  'auto' | 'left' | 'right'  (operator's --root)
    detected   arms named as un-recovered root cause by the driver log
    An arm needs recovery when its hardware component is not ACTIVE, or when it is ACTIVE on a real driver but its
    drives are reported disabled (then only the enable step is needed). Raises Refused when it cannot be done safely.
    """
    if requested not in ("auto", "left", "right"):
        raise Refused(f"--root must be auto, left or right, got {requested!r}")
    for arm in ARMS:
        view = views.get(arm)
        if view is None or view.hw_state is None:
            raise Refused(f"hardware component {hw_name(arm)} not found in the controller_manager")
        if view.hw_state in ("finalized", "unknown"):
            raise Refused(f"{hw_name(arm)} is {view.hw_state.upper()}: it cannot be recovered with the lifecycle "
                          f"service; relaunch the driver stack")
        if view.hw_state not in ("active", "inactive", "unconfigured"):
            raise Refused(f"{hw_name(arm)} is in the transition state {view.hw_state}; wait and retry")
    needing = [arm for arm in ARMS if views[arm].hw_state != "active" or needs_enable_check(views[arm])]
    plan = RecoveryPlan()
    if not needing:
        plan.notes.append("both hardware components are ACTIVE with enabled drives: nothing to recover (an inactive "
                          "controller of an active arm is activated with the 'activate' subcommand, which asks for "
                          "the identity check)")
        return plan
    detected = [arm for arm in detected if arm in ARMS]
    if requested != "auto":
        root = requested
        if detected and root not in detected:
            plan.notes.append(f"WARNING: --root {root} but the driver log names {', '.join(detected)} as the root "
                              f"cause; the driver will refuse the peer's activation if this is wrong")
        if root not in needing and not is_mock_class(views[root].hw_class):
            # The named root is ACTIVE with no drive-disabled evidence in the (10 s) log: it may still hold its
            # root-cause fault (peers are refused until it is cleared), so its drives are re-checked.
            needing.append(root)
            plan.notes.append(f"arm_{root} (--root) is ACTIVE; its drives are re-checked so that its root-cause "
                              f"fault can clear before the peer is recovered")
    elif len(needing) == 1:
        root = needing[0]
        plan.notes.append(f"only arm_{root} needs recovery")
    else:
        candidates = [arm for arm in detected if arm in needing]
        if len(candidates) != 1:
            raise Refused(
                "cannot determine the root-cause arm (the driver exposes no fault topic/service; the log shows "
                + (f"{len(detected)} candidates: {', '.join(detected)}" if detected else "none within reach")
                + "). Pass --root left or --root right (the arm that faulted FIRST), or --log-file FILE with the "
                  "driver output.")
        root = candidates[0]
        plan.notes.append(f"root cause arm_{root} from the driver log")
    plan.root = root
    plan.order = [arm for arm in (root, OTHER[root]) if arm in needing]
    for arm in plan.order:
        state = views[arm].controller_state
        if state is None:
            raise Refused(f"controller {controller_name(arm)} not found in the controller_manager")
        if state not in ("active", "inactive"):
            raise Refused(f"controller {controller_name(arm)} is {state!r}; only active or inactive controllers "
                          f"can be re-activated here (relaunch the driver stack)")
    return plan


def steps_for_arm(view, is_root=False, peer_pending=False):
    """Ordered hardware step names for one arm (the controllers are activated at the very end, for all arms).

    A latched arm is cycled inactive -> active and only then has its (still active) controller deactivated; an ACTIVE
    arm with disabled drives deactivates its controller first and skips the cycle. Every arm on a real
    driver then gets ``ensure_enabled`` (reset_and_enable until the drives report enabled). The root arm waits for
    its root-cause fault to clear before the peer is touched.
    """
    steps = []
    if view.hw_state != "active":
        # Humble rejects deactivating a controller while the hardware is not active: recover the hardware first,
        # then deactivate the still-active controller (the driver keeps jog at zero meanwhile).
        steps += ["reset_and_enable", "hw_inactive", "hw_active"]
        if view.controller_state == "active":
            steps.append("controller_deactivate")
    elif view.controller_state == "active":
        steps.append("controller_deactivate")
    if not is_mock_class(view.hw_class):
        steps.append("ensure_enabled")
        if is_root and peer_pending:
            steps.append("wait_root_cleared")
    return steps


def final_steps(views, plan):
    """Steps after ALL hardware is recovered: stillness of every arm, one identity check, controller activation."""
    steps = ["stillness"]
    if any(not is_mock_class(views[arm].hw_class) for arm in plan.order):
        steps.append("identity")
    steps.append("controllers_activate")
    return steps


def recovery_steps(views, plan):
    """Flat [(arm, step)] for a plan (ordering tests and --dry-run); final steps use the pseudo-arm 'all'."""
    steps = []
    for arm in plan.order:
        peer_pending = arm == plan.root and len(plan.order) > 1
        steps += [(arm, step) for step in steps_for_arm(views[arm], arm == plan.root, peer_pending)]
    steps += [("all", step) for step in final_steps(views, plan)]
    return steps


def assess_stillness(samples, arm, now, window_s=STILL_WINDOW_S, min_samples=STILL_MIN_SAMPLES,
                     max_displacement=STILL_MAX_DISPLACEMENT_RAD, max_velocity=STILL_MAX_VELOCITY,
                     max_age_s=STILL_MAX_AGE_S):
    """(ok, reason). samples: [{'t': monotonic receive time, 'pos': {joint: rad}, 'vel': {joint: rad/s}}]."""
    joints = joint_names(arm)
    usable = []
    for sample in samples:
        positions = [sample["pos"].get(joint) for joint in joints]
        if all(p is not None and math.isfinite(p) for p in positions):
            usable.append(sample)
    if len(usable) < min_samples:
        return False, f"only {len(usable)} complete joint_states samples for arm_{arm} (need {min_samples})"
    usable.sort(key=lambda s: s["t"])
    age = now - usable[-1]["t"]
    if age > max_age_s:
        return False, f"latest joint_states sample of arm_{arm} is {age:.2f} s old (limit {max_age_s} s)"
    span = usable[-1]["t"] - usable[0]["t"]
    if span < 0.6 * window_s:
        return False, f"joint_states of arm_{arm} cover only {span:.2f} s (need {0.6 * window_s:.2f} s)"
    for joint in joints:
        series = [s["pos"][joint] for s in usable]
        moved = max(series) - min(series)
        if moved > max_displacement:
            return False, f"{joint} moved {moved:.4f} rad within {span:.2f} s (limit {max_displacement} rad)"
        for s in usable:
            velocity = s["vel"].get(joint)
            if velocity is not None and math.isfinite(velocity) and abs(velocity) > max_velocity:
                return False, f"{joint} velocity {velocity:.3f} rad/s (limit {max_velocity} rad/s)"
    return True, f"arm_{arm} stood still for {span:.2f} s"


def interpret_reset_and_enable(success, message):
    """('ok'|'skipped'|'failed', text) for a reset_and_enable reply.

    The driver refuses the call while the hardware error is latched (Rebel.cpp ResetAndEnable): in a latched
    state the lifecycle re-activation performs the same start-up sequence, so that refusal is expected.
    """
    if success:
        return "ok", message
    if "latched" in message:
        return "skipped", "refused because the hardware error is latched (expected; the lifecycle re-activation "\
                          "re-sends the same start-up sequence): " + message
    return "failed", message


def confirmed(answer):
    return answer is not None and answer.strip() == "yes"


def format_positions(arm, positions):
    rows = []
    for joint in joint_names(arm):
        value = positions.get(joint)
        if value is None or not math.isfinite(value):
            rows.append(f"    {joint:<18} (no value)")
        else:
            rows.append(f"    {joint:<18} {value:+9.4f} rad  {math.degrees(value):+9.3f} deg")
    return "\n".join(rows)


# ---------------------------------------------------------------------------------------------------------
# ROS layer
# ---------------------------------------------------------------------------------------------------------

def ros_client_class():
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
    from controller_manager_msgs.srv import ListControllers, ListHardwareComponents, SetHardwareComponentState
    from controller_manager_msgs.srv import SwitchController
    from lifecycle_msgs.msg import State
    from rcl_interfaces.msg import Log
    from sensor_msgs.msg import JointState
    from std_srvs.srv import Trigger

    class Cell(Node):
        def __init__(self):
            super().__init__(f"dual_arm_recover_{os.getpid()}")
            self.samples = []
            self.rosout = []
            self.coordinator = None
            self.create_subscription(JointState, "/joint_states", self._on_joint_state, qos_profile_sensor_data)
            # /rosout is transient-local with a 10 s lifespan: a late subscriber still gets the last 10 s.
            self.create_subscription(
                Log, "/rosout", self._on_rosout,
                QoSProfile(depth=1000, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.RELIABLE,
                           durability=DurabilityPolicy.TRANSIENT_LOCAL))
            self._status_sub = None
            self.list_hw_cli = self.create_client(ListHardwareComponents, f"{CONTROLLER_MANAGER}/list_hardware_components")
            self.list_ctrl_cli = self.create_client(ListControllers, f"{CONTROLLER_MANAGER}/list_controllers")
            self.set_hw_cli = self.create_client(SetHardwareComponentState,
                                                 f"{CONTROLLER_MANAGER}/set_hardware_component_state")
            self.switch_cli = self.create_client(SwitchController, f"{CONTROLLER_MANAGER}/switch_controller")

        # -- plumbing
        def _on_joint_state(self, msg):
            names = list(msg.name)
            pos = {n: msg.position[i] for i, n in enumerate(names) if i < len(msg.position)}
            vel = {n: msg.velocity[i] for i, n in enumerate(names) if i < len(msg.velocity)}
            self.samples.append({"t": time.monotonic(), "pos": pos, "vel": vel})
            del self.samples[:-400]

        def _on_rosout(self, msg):
            text = msg.msg
            if any(key in text for key in ROSOUT_KEYS):
                # The logger name ("igus_rebel.arm_right_hw") is the only component tag of some driver lines.
                self.rosout.append(f"[{msg.name}]: {text}")

        def rosout_mark(self):
            """Position in the driver-log buffer; lines appended later are 'new' (subscribed before triggering)."""
            self.spin_for(0.05)
            return len(self.rosout)

        def rosout_since(self, mark):
            return list(self.rosout[mark:])

        def spin_for(self, seconds):
            end = time.monotonic() + seconds
            while time.monotonic() < end:
                rclpy.spin_once(self, timeout_sec=0.05)

        def call(self, client, request, timeout=10.0, what="service"):
            if not client.wait_for_service(timeout_sec=min(timeout, 5.0)):
                raise StepFailed(f"{what}: service {client.srv_name} not available (is the driver stack running "
                                 f"in this ROS domain? ROS_DOMAIN_ID={os.environ.get('ROS_DOMAIN_ID', '(unset)')})")
            future = client.call_async(request)
            end = time.monotonic() + timeout
            while not future.done() and time.monotonic() < end:
                rclpy.spin_once(self, timeout_sec=0.05)
            if not future.done():
                future.cancel()
                raise StepFailed(f"{what}: no answer from {client.srv_name} within {timeout:.0f} s")
            return future.result()

        # -- queries
        def views(self):
            hw = self.call(self.list_hw_cli, ListHardwareComponents.Request(), what="list_hardware_components")
            ctrl = self.call(self.list_ctrl_cli, ListControllers.Request(), what="list_controllers")
            components = {c.name: c for c in hw.component}
            controllers = {c.name: c.state for c in ctrl.controller}
            result = {}
            for arm in ARMS:
                component = components.get(hw_name(arm))
                result[arm] = ArmView(
                    arm=arm,
                    hw_state=None if component is None else LIFECYCLE.get(component.state.id, "unknown"),
                    hw_class=None if component is None else component.class_type,
                    controller_state=controllers.get(controller_name(arm)))
            return result, controllers

        def sample_joint_states(self, seconds=STILL_WINDOW_S):
            self.samples.clear()
            self.spin_for(seconds)
            return list(self.samples)

        def positions(self, seconds=0.5):
            samples = self.sample_joint_states(seconds)
            merged = {}
            for sample in samples:
                merged.update(sample["pos"])
            return merged, samples

        def stillness(self, arm):
            samples = self.sample_joint_states()
            return assess_stillness(samples, arm, time.monotonic())

        def fault_events(self, log_file=None):
            lines = list(self.rosout)
            if log_file:
                try:
                    with open(os.path.expanduser(log_file), errors="replace") as handle:
                        lines = handle.read().splitlines() + lines
                except OSError as error:
                    raise Refused(f"cannot read --log-file {log_file}: {error}")
            return parse_fault_events(lines)

        def coordinator_status(self, timeout=3.0):
            from dual_cut_msgs.msg import CoordinatorStatus
            if self._status_sub is None:
                self._status_sub = self.create_subscription(
                    CoordinatorStatus, COORDINATOR_STATUS_TOPIC,
                    lambda m: setattr(self, "coordinator", m), QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE))
            self.coordinator = None
            end = time.monotonic() + timeout
            while self.coordinator is None and time.monotonic() < end:
                rclpy.spin_once(self, timeout_sec=0.05)
            return self.coordinator

        # -- actions
        def set_hw_state(self, arm, target, timeout=20.0):
            request = SetHardwareComponentState.Request()
            request.name = hw_name(arm)
            request.target_state = State(id=STATE_ID[target], label=target)
            reply = self.call(self.set_hw_cli, request, timeout=timeout, what=f"set {hw_name(arm)} {target}")
            return reply.ok, LIFECYCLE.get(reply.state.id, str(reply.state.id))

        def switch(self, activate=(), deactivate=(), timeout=15.0):
            request = SwitchController.Request()
            request.activate_controllers = list(activate)
            request.deactivate_controllers = list(deactivate)
            request.strictness = SwitchController.Request.STRICT
            request.activate_asap = True
            request.timeout.sec = 5
            return self.call(self.switch_cli, request, timeout=timeout, what="switch_controller").ok

        def reset_and_enable(self, arm, timeout=10.0, discover_s=2.0):
            client = self.create_client(Trigger, reset_service(arm))
            try:
                if not client.wait_for_service(timeout_sec=discover_s):
                    return None
                future = client.call_async(Trigger.Request())
                end = time.monotonic() + timeout
                while not future.done() and time.monotonic() < end:
                    rclpy.spin_once(self, timeout_sec=0.05)
                if not future.done():
                    future.cancel()
                    raise StepFailed(f"no answer from {reset_service(arm)} within {timeout:.0f} s")
                return future.result().success, future.result().message
            finally:
                self.destroy_client(client)

        def has_service(self, name):
            return any(n == name for n, _ in self.get_service_names_and_types())

    return rclpy, Cell


class Session:
    def __init__(self):
        self.rclpy, cell_class = ros_client_class()
        self.rclpy.init()
        self.cell = cell_class()

    def close(self):
        self.cell.destroy_node()
        if self.rclpy.ok():
            self.rclpy.shutdown()


def say(message=""):
    print(message, flush=True)


def view_line(view):
    return (f"  {hw_name(view.arm):<14} class={view.hw_class or '-':<32} state={view.hw_state or 'MISSING':<13}"
            f" | {controller_name(view.arm)}: {view.controller_state or 'MISSING'}"
            + (" | DRIVES DISABLED (per driver log)" if view.drives_disabled else "")
            + (" | WARNING: controller ACTIVE while hardware not active" if controller_active_without_hardware(view)
               else ""))


def mark_drives(views, events):
    """Copy the drive-disabled evidence of the driver log onto the arm views."""
    disabled = drives_disabled_arms(events)
    for arm in ARMS:
        if views.get(arm) is not None:
            views[arm].drives_disabled = arm in disabled


# ---------------------------------------------------------------------------------------------------------
# subcommands
# ---------------------------------------------------------------------------------------------------------

def cmd_status(args, session):
    cell = session.cell
    say(f"ROS_DOMAIN_ID={os.environ.get('ROS_DOMAIN_ID', '(unset)')} "
        f"ROS_LOCALHOST_ONLY={os.environ.get('ROS_LOCALHOST_ONLY', '(unset)')}")
    cell.spin_for(1.0)   # collect the /rosout backlog and joint states
    views, controllers = cell.views()
    events = cell.fault_events(args.log_file)
    mark_drives(views, events)
    healthy = True
    say("Hardware components and trajectory controllers:")
    for arm in ARMS:
        say(view_line(views[arm]))
        healthy &= views[arm].hw_state == "active" and views[arm].controller_state == "active"
        healthy &= not views[arm].drives_disabled
    say(f"  {JOINT_STATE_BROADCASTER}: {controllers.get(JOINT_STATE_BROADCASTER, 'MISSING')}")
    healthy &= controllers.get(JOINT_STATE_BROADCASTER) == "active"

    say("Fault / latch information (the driver has no fault topic or service; this is inferred):")
    for arm in ARMS:
        state = views[arm].hw_state
        if state == "active" and views[arm].drives_disabled:
            say(f"  arm_{arm}: hardware ACTIVE but drives DISABLED ('Robot drives disabled' in the driver log; the "
                f"arm will not move) -> run 'recover' (it re-enables via {reset_service(arm)}; if the line is only "
                f"a few seconds old and the arm just recovered, recover verifies it)")
        elif state == "active":
            say(f"  arm_{arm}: hardware ACTIVE (no latch visible; drive state is only visible in /rosout, where the "
                f"driver warns every 10 s while the drives are disabled)")
        else:
            say(f"  arm_{arm}: hardware is {state} -> stopped or latched by the driver (jog zero); "
                f"recover with the 'recover' subcommand")
        if controller_active_without_hardware(views[arm]):
            say(f"  arm_{arm}: WARNING {controller_name(arm)} is ACTIVE while the hardware is {state} (Humble does "
                f"not deactivate controllers on a hardware error, and refuses to while the hardware is not active). "
                f"'recover' deactivates it right after the hardware is active again; the driver keeps jog at zero "
                f"until the controller is restarted, so it cannot move the arm meanwhile")
    roots = active_root_causes(events)
    if events:
        say(f"  driver log events found ({len(events)}); un-recovered root cause per the log: "
            f"{', '.join(roots) if roots else 'none'}")
        for event in events[-4:]:
            say(f"    [{event.kind}] {event.text[:200]}")
    else:
        say("  no fault-group lines in /rosout (the middleware keeps only the last ~10 s; use --log-file with the "
            "launch log / journal to search further back)")
    for arm in ARMS:
        present = cell.has_service(reset_service(arm))
        say(f"  {reset_service(arm)}: {'present' if present else 'absent (mock hardware has no Igus driver)'}")

    say("Joint states:")
    samples = cell.sample_joint_states()
    if not samples:
        say("  /joint_states: NO messages received")
        healthy = False
    else:
        for arm in ARMS:
            ok, reason = assess_stillness(samples, arm, time.monotonic())
            say(f"  arm_{arm}: {'standing still' if ok else 'NOT confirmed still'} ({reason})")

    say("Coordinator:")
    status = cell.coordinator_status()
    if status is None:
        say(f"  no message on {COORDINATOR_STATUS_TOPIC} within 3 s (coordinator not running?)")
        healthy = False
    else:
        say(f"  state={COORDINATOR_STATES.get(status.state, status.state)} phase={status.phase or '-'} "
            f"plan_seq={status.plan_seq} retained={status.plan_retained} validated={status.phase_validated}")
        say(f"  stop_latched={status.stop_latched}" + (f" reason={status.stop_reason!r}" if status.stop_reason else "")
            + f" min_cross_distance={status.min_cross_distance_m:.3f} m joint_states_age={status.joint_states_age_s:.2f} s")
        if status.barrier_message:
            say(f"  barrier_ok={status.barrier_ok}: {status.barrier_message}")
        if status.stop_latched:
            say(f"  The coordinator is STOPPED and latched. After the arms are recovered, rearm on purpose:\n"
                f"    ros2 service call {COORDINATOR_REARM_SERVICE} std_srvs/srv/Trigger '{{}}'")
            healthy = False
    say("OVERALL: " + ("healthy" if healthy else "NOT healthy (see above)"))
    return 0 if healthy else 1


def confirm_identity(cell, views, args, what):
    """Print both arms' joint positions and ask for 'yes'. --yes skips it in mock only. Returns True if confirmed."""
    allowed, reason = yes_allowed(list(views.values()))
    if args.yes and not allowed:
        raise Refused(reason)
    positions, _ = cell.positions()
    say("Current joint positions (compare with the pendant of each arm):")
    for arm in ARMS:
        say(f"  arm_{arm}:")
        say(format_positions(arm, positions))
    if args.yes:
        say(f"--yes given and every hardware component is a mock component: skipping the prompt for {what}.")
        return True
    say(f"Before {what}: confirm that the arms are standing still and that the left arm is the one on the left-IP "
        f"controller (arm identity checked by pendant jog).")
    try:
        answer = input("Type 'yes' to continue: ")
    except EOFError:
        answer = None
    return confirmed(answer)


def verify_hw(cell, arm, expected):
    views, _ = cell.views()
    state = views[arm].hw_state
    if state != expected:
        raise StepFailed(f"{hw_name(arm)} is {state}, expected {expected}")


def wait_still_or_fail(cell, arm):
    ok, reason = cell.stillness(arm)
    if not ok:
        raise StepFailed(f"arm_{arm} is not confirmed standing still, controller {controller_name(arm)} NOT "
                         f"activated: {reason}")
    say(f"    arm_{arm} standing still: {reason}")


def ensure_drives_enabled(cell, arm, view):
    """Block until the driver confirms the drives of ``arm`` are enabled (call after the component is ACTIVE).

    Signal: the reply of <driver>/reset_and_enable. The driver re-sends SetActive/Reset/Enable with jog held at zero
    and waits up to 3 s INSIDE the service call for 'Motor not enabled' to disappear from STATUS; success=True means
    mask 0, i.e. enabled. This does not depend on /rosout timing. The lifecycle activation alone does not guarantee
    it: after an e-stop the Enable can be sent before the motor supply is back and is silently lost. The call is
    permitted once the component is no longer latched, and is repeated for up to ENABLE_TIMEOUT_S.
    """
    mock = is_mock_class(view.hw_class)
    deadline = _now() + ENABLE_TIMEOUT_S
    attempt = 0
    last = "no reply"
    while True:
        attempt += 1
        reply = cell.reset_and_enable(arm, discover_s=2.0 if mock else 5.0)
        if reply is None:
            if mock:
                say(f"    reset_and_enable: service {reset_service(arm)} not present (mock), skipped")
                return
            raise StepFailed(f"{reset_service(arm)} is not available, so the drives of arm_{arm} cannot be "
                             f"confirmed enabled; controllers NOT activated")
        success, message = reply
        if success:
            say(f"    arm_{arm} drives enabled (attempt {attempt}): {message}")
            return
        last = message
        if "latched" in message:
            raise StepFailed(f"{hw_name(arm)} is latched again after activation: {message}")
        if _now() >= deadline:
            break
        say(f"    arm_{arm} drives not enabled yet (attempt {attempt}): {message}; retrying")
        _sleep(ENABLE_RETRY_PAUSE_S)
    raise StepFailed(
        f"drives of arm_{arm} did not report enabled within {ENABLE_TIMEOUT_S:.0f} s (last reply: {last}). Is the "
        f"e-stop released and the motor supply on (pendant)? No trajectory controller was activated.")


def wait_root_cleared(cell, arm, mark, require):
    """Wait for the driver's 'enabled with clean STATUS again; its root-cause fault is cleared' line for ``arm``.

    The line is the only observable of the root-cause clearing (needs 500 ms of continuous enabled, clean STATUS;
    every reset_and_enable restarts that clock). ``mark`` is the /rosout buffer position taken BEFORE the arm was
    re-activated, so a line emitted early is not missed. require=False (arm was already ACTIVE, so its fault may
    have been cleared long ago and no new line will come): a missing line is not an error here; the peer's
    activation is then the arbiter.
    """
    component = hw_name(arm)
    deadline = _now() + ROOT_CLEAR_TIMEOUT_S
    while True:
        if root_cleared(parse_fault_events(cell.rosout_since(mark)), component):
            say(f"    {component}: driver reports its root-cause fault cleared")
            return True
        if _now() >= deadline:
            break
        cell.spin_for(0.1)
    if not require:
        say(f"    no 'root-cause fault is cleared' line for {component} within {ROOT_CLEAR_TIMEOUT_S:.0f} s "
            f"({component} was already ACTIVE, so it may have been cleared earlier); the peer activation decides")
        return False
    raise StepFailed(
        f"{component} is active with enabled drives but the driver did not report its root-cause fault cleared "
        f"within {ROOT_CLEAR_TIMEOUT_S:.0f} s (it needs 500 ms of continuous clean STATUS: a joint error, e-stop or "
        f"supply fault may still be present; read the driver log). The peer arm_{OTHER[arm]} was NOT touched and no "
        f"trajectory controller was activated. If arm_{arm} was not the arm that faulted FIRST, re-run with "
        f"--root {OTHER[arm]}.")


def activate_hardware(cell, arm, is_root, step):
    """Latched/inactive hardware component -> inactive -> active (peers retry while the root clears)."""
    ok, current = cell.set_hw_state(arm, "inactive")
    if not ok or current != "inactive":
        raise StepFailed(f"set {hw_name(arm)} inactive failed (ok={ok}, state={current})")
    verify_hw(cell, arm, "inactive")
    say(f"  [{step()}] {hw_name(arm)} inactive")
    deadline = _now() + PEER_ACTIVATE_WINDOW_S
    attempt = 0
    while True:
        attempt += 1
        ok, current = cell.set_hw_state(arm, "active")
        if ok and current == "active":
            break
        if is_root or _now() >= deadline:
            raise StepFailed(
                f"set {hw_name(arm)} active failed (ok={ok}, state={current}). The driver refuses activation "
                f"unless STATUS is fresh, and for a peer while the root cause is un-recovered; read the "
                f"driver log line 'Refusing to activate hardware component' for the reason. If arm_{OTHER[arm]} "
                f"is the arm that faulted FIRST (even with its hardware ACTIVE), re-run with --root {OTHER[arm]}"
                + ("" if is_root else f" (tried {attempt}x over {PEER_ACTIVATE_WINDOW_S:.0f} s)"))
        say(f"      activation refused (state={current}); retrying in {PEER_SETTLE_S:.1f} s "
            f"(the root cause needs 500 ms of clean STATUS)")
        _sleep(PEER_SETTLE_S)
        if current != "inactive":
            ok2, current2 = cell.set_hw_state(arm, "inactive")
            if not ok2 or current2 != "inactive":
                raise StepFailed(f"cannot return {hw_name(arm)} to inactive for the retry (state={current2})")
    verify_hw(cell, arm, "active")
    say(f"  [{step()}] {hw_name(arm)} active")


def recover_arm_hardware(cell, arm, view, is_root, peer_pending):
    """Hardware recovery of one arm; NEVER activates a trajectory controller (that happens at the very end)."""
    latched = view.hw_state != "active"
    say(f"== arm_{arm} ({'ROOT CAUSE' if is_root else 'peer'}; "
        f"{'hardware ' + view.hw_state if latched else 'hardware ACTIVE but drives disabled / re-check'}) ==")
    counter = [0]

    def step():
        counter[0] += 1
        return counter[0]

    mark = cell.rosout_mark()   # subscribe-before-trigger: lines after this point belong to this recovery
    controller = controller_name(arm)

    def deactivate_controller():
        if not cell.switch(deactivate=[controller]):
            raise StepFailed(f"deactivating {controller} failed" + (
                " (the hardware is active again, so Humble should accept this; the driver keeps jog at zero "
                "until the controller is restarted, so nothing can move)" if latched else ""))
        say(f"  [{step()}] {controller} deactivated")

    if not latched and view.controller_state == "active":
        # Hardware ACTIVE (drives disabled): the deactivation is accepted, and it must come BEFORE the drives are
        # re-enabled so that a stale trajectory cannot move the arm the moment they are enabled.
        deactivate_controller()
    controller_still_active = latched and view.controller_state == "active"
    if controller_still_active:
        # Always try to deactivate FIRST. It succeeds when the interfaces are still available (e.g. hardware set
        # INACTIVE by hand: no driver latch, so no driver jog hold either, and reset_and_enable would work). Humble
        # rejects it only after an error (component UNCONFIGURED); only then is the driver latched, so its
        # controller-restart hold covers the hardware-first order below.
        if cell.switch(deactivate=[controller]):
            say(f"  [{step()}] {controller} deactivated")
            controller_still_active = False
        elif view.hw_state != "unconfigured":
            raise StepFailed(
                f"{controller} is ACTIVE and could not be deactivated while {hw_name(arm)} is {view.hw_state} "
                f"(expected only for 'unconfigured' after a hardware error). Not touching this arm; restart the "
                f"dual driver launch to get a clean state.")
    if latched:
        reply = cell.reset_and_enable(arm)
        if reply is None:
            say(f"  [{step()}] reset_and_enable: service {reset_service(arm)} not present, skipped")
        else:
            verdict, text = interpret_reset_and_enable(*reply)
            if verdict == "failed":
                raise StepFailed(f"reset_and_enable of arm_{arm} failed: {text}")
            say(f"  [{step()}] reset_and_enable {verdict}: {text}")
        activate_hardware(cell, arm, is_root, step)
        if controller_still_active:
            # Humble's controller_manager leaves the controller ACTIVE after a hardware error and rejects its
            # deactivation while the hardware is not active, so it can only be deactivated now. The driver holds
            # the transmitted jog at zero until the controller is re-started, so its stale setpoint cannot move
            # the arm in the meantime (and enabling the drives does not release that hold).
            say(f"      note: {controller} was still ACTIVE (it cannot be deactivated while the hardware is not "
                f"active). The driver keeps jog at zero until the controller is re-activated (the last step).")
            deactivate_controller()
        elif not is_mock_class(view.hw_class):
            say(f"      note: the driver keeps jog at zero until {controller} is activated (the last step).")
    if not is_mock_class(view.hw_class):
        say(f"  [{step()}] making sure the drives of arm_{arm} are enabled")
        ensure_drives_enabled(cell, arm, view)
        if is_root and peer_pending:
            say(f"  [{step()}] waiting for the root-cause fault of {hw_name(arm)} to clear")
            # Informational only: the driver logs through a non-node logger ('igus_rebel.<component>'),
            # which does not reach /rosout on the real stack (2026-10-15: the line was in the launch log
            # but never on /rosout). The driver itself refuses the peer until the root is cleared, so the
            # peer activation window below is the arbiter.
            wait_root_cleared(cell, arm, mark, require=False)


def run_recovery(cell, plan, views, args):
    """Recover all hardware (root first), THEN stillness, one identity check, THEN activate the controllers."""
    original = {arm: views[arm].drives_disabled for arm in ARMS}
    for arm in plan.order:
        fresh, _ = cell.views()
        for other in ARMS:
            fresh[other].drives_disabled = original[other]
        recover_arm_hardware(cell, arm, fresh[arm], arm == plan.root, arm == plan.root and len(plan.order) > 1)
    say("== all hardware recovered; checking the arms before any controller is activated ==")
    after, _ = cell.views()
    for arm in plan.order:
        if after[arm].hw_state != "active":
            raise StepFailed(f"{hw_name(arm)} is {after[arm].hw_state}, expected active; controllers NOT activated")
    for arm in plan.order:
        wait_still_or_fail(cell, arm)
    if any(not is_mock_class(after[arm].hw_class) for arm in plan.order):
        if not confirm_identity(cell, after, args, "re-activating the trajectory controllers"):
            raise StepFailed("operator did not confirm; controllers NOT activated")
    targets = [controller_name(arm) for arm in plan.order]
    if not cell.switch(activate=targets):
        raise StepFailed("activating " + ", ".join(targets) + " failed")
    final, _ = cell.views()
    bad = [controller_name(arm) for arm in plan.order if final[arm].controller_state != "active"]
    if bad:
        raise StepFailed("not active after the switch: " + ", ".join(bad))
    say("  controllers active: " + ", ".join(targets))
    if any(not is_mock_class(after[arm].hw_class) for arm in plan.order):
        say("  (the drivers keep jog at zero after a recovery until their controller is re-activated; that "
            "activation released it)")


def cmd_recover(args, session):
    cell = session.cell
    cell.spin_for(1.0)
    views, _ = cell.views()
    events = cell.fault_events(args.log_file)
    mark_drives(views, events)
    plan = plan_recovery(views, args.root, detected_root_arms(events))
    for arm in ARMS:
        say(view_line(views[arm]))
    for note in plan.notes:
        say(f"  note: {note}")
    if not plan.order:
        return 0
    say("Recovery order: " + " then ".join(f"arm_{arm}" for arm in plan.order) + f" (root cause: arm_{plan.root})")
    if args.dry_run:
        for arm, step in recovery_steps(views, plan):
            say(f"  would run: {'arm_' + arm if arm != 'all' else 'all arms'}: {step}")
        say("  (dry run: nothing was changed)")
        return 0
    run_recovery(cell, plan, views, args)
    say("Recovered: " + ", ".join(f"arm_{arm}" for arm in plan.order))
    say(f"The coordinator was NOT rearmed. When both arms are verified (status), rearm it yourself:\n"
        f"  ros2 service call {COORDINATOR_REARM_SERVICE} std_srvs/srv/Trigger '{{}}'")
    return 0


def cmd_activate(args, session):
    cell = session.cell
    cell.spin_for(1.0)
    views, _ = cell.views()
    mark_drives(views, cell.fault_events(None))
    for arm in ARMS:
        say(view_line(views[arm]))
    if args.yes:
        allowed, reason = yes_allowed(list(views.values()))
        if not allowed:
            raise Refused(reason)
    for arm in ARMS:
        view = views[arm]
        if view.hw_state != "active":
            raise Refused(f"{hw_name(arm)} is {view.hw_state}; run 'recover' first")
        if view.drives_disabled:
            raise Refused(f"arm_{arm} hardware is active but its drives are disabled (driver log); run 'recover' first")
        if view.controller_state not in ("active", "inactive"):
            raise Refused(f"{controller_name(arm)} is {view.controller_state}; cannot activate it")
    targets = [controller_name(arm) for arm in ARMS if views[arm].controller_state == "inactive"]
    if not targets:
        say("Both trajectory controllers are already active.")
        say("Note: if a hardware component was recovered by hand after a fault, its driver keeps jog at zero until "
            "the controller is restarted (ros2 control switch_controllers --deactivate <controller>, then a "
            "separate --activate <controller>); a single call with both does not restart a controller in Humble.")
        return 0
    for arm in ARMS:
        ok, reason = cell.stillness(arm)
        if not ok:
            raise Refused(f"arm_{arm} is not confirmed standing still, nothing activated: {reason}")
        say(f"  {reason}")
    if not confirm_identity(cell, views, args, "activating the trajectory controllers"):
        say("Not confirmed (anything but exactly 'yes'): nothing activated.")
        return 2
    if not cell.switch(activate=targets):
        raise StepFailed("switch_controller refused the activation (strict); nothing was activated")
    after, _ = cell.views()
    bad = [controller_name(arm) for arm in ARMS if after[arm].controller_state != "active"]
    if bad:
        raise StepFailed("not active after the switch: " + ", ".join(bad))
    say("Trajectory controllers active: " + ", ".join(targets))
    say("Execution is still disabled in the coordinator (execution_enabled and the phase enables are false).")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(prog="dual_arm_recover.py", description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    status = sub.add_parser("status", help="show hardware, controllers, fault info and coordinator status")
    status.add_argument("--log-file", help="driver/launch log to search for fault-group lines")
    recover = sub.add_parser("recover", help="recover the latched arms, root-cause arm first")
    recover.add_argument("--root", choices=("left", "right", "auto"), default="auto",
                         help="arm that faulted FIRST (default auto: from the driver log; required if unknown)")
    recover.add_argument("--log-file", help="driver/launch log to search for the root cause")
    recover.add_argument("--dry-run", action="store_true", help="print the plan, change nothing")
    recover.add_argument("--yes", action="store_true",
                         help="skip the operator prompt before re-activating controllers (mock only)")
    activate = sub.add_parser("activate", help="activate both trajectory controllers after the identity check")
    activate.add_argument("--yes", action="store_true", help="skip the prompt (mock only; refused on real hardware)")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    handlers = {"status": cmd_status, "recover": cmd_recover, "activate": cmd_activate}
    session = Session()
    try:
        return handlers[args.command](args, session)
    except Refused as error:
        say(f"REFUSED: {error}")
        return 2
    except StepFailed as error:
        say(f"FAILED: {error}")
        say("Stopped at the first failure; nothing further was done.")
        return 1
    finally:
        session.close()


if __name__ == "__main__":
    sys.exit(main())
