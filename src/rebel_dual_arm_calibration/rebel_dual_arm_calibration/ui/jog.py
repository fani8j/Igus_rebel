"""Step jog for the calibration UI: ONE bounded step per request, then the arm stops.

There is no continuous / hold-to-move mode, so a lost browser can never leave an arm moving.  A step is

    cart   translate the TCP by 0.1 / 0.5 / 1 / 5 / 10 mm along x|y|z of the world (planning) or tool frame
    rot    rotate the TCP by 0.5 / 1 / 5 deg about x|y|z of the world or tool axes, about the TCP point
    joint  move one joint (1..6) by 0.5 / 1 / 5 deg

and is executed as one FollowJointTrajectory goal that this module re-times itself (quintic time scaling,
zero velocity at both ends, peak TCP speed 10 mm/s, peak TCP angular speed 5 deg/s, every joint 0.1 rad/s,
at least 0.5 s).  The returned MoveIt timing is never used.

Besides steps there is ONE "move to a joint goal" (plan_move -> RViz preview -> execute_move): MoveIt plans a
collision-free joint path, this module densifies (<= 0.25 deg), validity-checks and re-times it itself under the
MOVE_* caps (0.04 rad/s, 0.25 rad/s^2, 40 mm/s) and stores it as the single move plan; execute_move runs the
stored plan through the same FollowJointTrajectory / STOP / standstill / fault machinery as a step.

Besides steps and moves there is ONE "descend" (approach assist / hover): a straight Cartesian move of the TCP by
2..70 mm at constant orientation, to a pose given by the caller.  It is exactly a Cartesian step (same
/compute_cartesian_path with fraction 1.0, END / PATH deviation checks, Cartesian caps 10 mm/s and 5 deg/s, Hermite cap
verification, executed-curve validity check, MotionSupervisor, dispatch_goal, event-anchored settle, STOP / fault) with
the target as a pose and a length limit instead of the step-size whitelist.  Gating (safety review): on REAL hardware a
descend needs the UI to be started with allow_real_move:=true exactly like plan_move / execute_move (otherwise it returns
REAL_MOVE_DISABLED_MSG; the operator fine-tunes with teleop instead) AND the page's real-hardware enable like a step.

Safety review additions: invalid (non-finite) joint states fail closed, standstill includes a drift / velocity test,
clear_fault needs the controller's confirmation that the goal ended, motion requests are gated at HTTP ingress and carry a
STOP epoch, the executed Hermite curve is bounded against the validated polyline, a MotionSupervisor watches every
executing goal (stale / speed / tracking -> STOP) and every move plan gets a coarse scene re-check at execute time.

Second-round safety additions: the final gate and the goal send are one critical section against STOP
(JogCore.dispatch_goal), "terminal" needs a VALIDATED goal result (validated_result; an exception / unknown status is
"unconfirmed" and keeps the fault), standstill needs 1.2 s (longer than the driver's 1.0 s STATUS ERROR cutoff, so a
frozen feedback cannot pass), the speed supervision looks at 0.5 s, and /check_state_validity also runs on dense samples of
the executed Hermite curve (curve_check_states).

Third-round safety additions: every confirmation (stop, settle after a goal, clear_fault) is EVENT-ANCHORED
(standstill(arm, since=...): only samples received after the STOP / result / fault count and must cover the whole 1.2 s
window) and needs healthy hardware (component + controller active, answer to a request sent after the driver's 1.0 s STATUS cutoff);
a step / move is only accepted and dispatched when the arm is measured at standstill over the full window; the supervision
stops a goal whose feedback is frozen while the reference moves; a STOP between the controller's acceptance and the worker
recording the handle cancels the goal; the move plan's final curve is re-checked at EVERY execute.

Layout
    pure maths (no ROS):  STEP_SIZES, arm_configs, validate_request, target_pose, retime, trajectory_peaks,
                          validate_joint_goal, densify_path, retime_path, plan_move_timing (joint-goal moves),
                          RealLock (real-hardware enable state machine with an injectable clock),
                          JogCore (caches, pre-step refusals, busy flag, status, the stored move plan)
    ROS glue:             JogController (the interface the backend uses; owns a JogCore plus worker threads)

Threading: JogController methods are called from a non-ROS thread (Tornado) and never block for more than a
few ms, except plan_move() (planning + validity checks, seconds; the backend runs it in a thread pool).  Planning and execution run in one worker thread per step; the node is spun by the backend's
MultiThreadedExecutor (this module never spins) and service / action results arrive through done callbacks.

Mock vs real hardware is decided per arm from controller_manager's /list_hardware_components: the arm's
component is MOCK only if its ``class_type`` is exactly ``mock_components/GenericSystem``; anything else
(including a missing component or an unreachable controller_manager) is treated as REAL.
"""
from __future__ import annotations

import bisect
import math
import threading
import time
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..kinematics import KinematicsError, Urdf

try:  # the pure part of this module (and its unit tests) must import without a sourced ROS
    import rclpy
    from builtin_interfaces.msg import Duration
    from control_msgs.action import FollowJointTrajectory
    from controller_manager_msgs.srv import ListControllers, ListHardwareComponents
    from geometry_msgs.msg import Pose
    from moveit_msgs.msg import Constraints, DisplayTrajectory, JointConstraint, RobotState, RobotTrajectory
    from moveit_msgs.srv import GetCartesianPath, GetMotionPlan, GetStateValidity
    from rcl_interfaces.srv import GetParameters
    from rclpy.action import ActionClient
    from rclpy.callback_groups import ReentrantCallbackGroup
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from rclpy.time import Time
    from sensor_msgs.msg import JointState
    from std_msgs.msg import String
    from tf2_ros import Buffer, TransformListener
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    _ROS_IMPORT_ERROR: Optional[Exception] = None
except ImportError as _e:  # pragma: no cover - only without ROS
    _ROS_IMPORT_ERROR = _e

# ---------------------------------------------------------------------------------------------------
# constants

STEP_SIZES: Dict[str, list] = {"cart_mm": [0.1, 0.5, 1, 5, 10], "rot_deg": [0.5, 1, 5], "joint_deg": [0.5, 1, 5]}

MAX_CART_SPEED_MM_S = 10.0
MAX_ROT_SPEED_DEG_S = 5.0
MAX_JOINT_SPEED_RAD_S = 0.1
MAX_JOINT_ACCEL_RAD_S2 = 0.5       # = max_acceleration of the dual-table joint_limits.yaml bring-up caps
MIN_DURATION_S = 0.5
DURATION_MARGIN = 1.05             # peak speeds end up at <= 1/1.05 of the caps
SAMPLE_DT_S = 0.05                 # spacing of the trajectory points sent to the controller
QUINTIC_PEAK = 1.875               # max of d/dtau of the quintic smootherstep u(tau)
QUINTIC_ACCEL_PEAK = 5.773502691896258   # max of d2/dtau2 of u

JOINT_STATE_MAX_AGE_S = 0.5
CM_CACHE_MAX_AGE_S = 1.0
CM_POLL_PERIOD_S = 0.5

CART_WAYPOINT_SPACING_M = 0.0005   # our own waypoint spacing handed to /compute_cartesian_path (and its max_step)
ROT_WAYPOINT_SPACING_RAD = math.radians(0.1)
JOINT_CHECK_SPACING_DEG = 0.25     # /check_state_validity spacing for joint steps
MAX_WAYPOINT_JUMP_RAD = 0.02       # absolute joint jump between consecutive path points (IK branch flip guard)
END_POS_TOL_M = 0.0001             # returned path must end on the target within 0.1 mm / 0.1 deg
END_ROT_TOL_RAD = math.radians(0.1)
PATH_POS_DEV_M = 0.0002            # every path point within 0.2 mm / 0.2 deg of the ideal straight / geodesic path
PATH_ROT_DEV_RAD = math.radians(0.2)
TF_FK_POS_TOL_M = 0.001            # TF pose vs forward kinematics of the same joint state
TF_FK_ROT_TOL_RAD = math.radians(0.1)
# The first trajectory point is the joint state read at accept time and the controller starts a trajectory from
# its own measured state, so any difference between the two is commanded as an instantaneous corrective jump.
# The real driver reports positions in 0.01 deg steps (1.745e-4 rad): a tolerance below ONE encoder count (the old
# 1e-4 rad) is crossed by the single-count flicker of a stationary arm and forced endless replanning.  4e-4 rad is
# ~2.3 counts (0.023 deg), ~0.36 mm at a 0.9 m lever: more than flicker (+-1 count), still far below what a moving
# arm covers in a 50 ms control cycle at the caps (0.1 rad/s -> 5e-3 rad), so a real start offset is still refused.
START_MOVED_TOL_RAD = 4e-4
MIN_STEP_GAP_S = 0.3               # a step arriving sooner than this after the previous one finished is refused
# standstill = all six joints within STANDSTILL_TOL_RAD over this window.  The window MUST be longer than the real
# driver's 1.0 s STATUS-age ERROR cutoff (igus_rebel RebelSafety.hpp): during a STATUS freeze (> 200 ms coast, > 400 ms
# hold) /joint_states keeps publishing the FROZEN position with fresh receive stamps, so a freeze of up to 1.0 s looks
# like a perfectly still arm.  With a window of 1.2 s a freeze can never cover the whole window: the motion before the
# freeze (or the position jump when the feed resumes) is always inside it, and a freeze of >= 1.0 s ends in a driver
# ERROR (controller deactivated, hardware_ok fails) anyway.  (Was 0.5 s: a 550 ms freeze passed.)
STANDSTILL_WINDOW_S = 1.2
STANDSTILL_TOL_RAD = 5e-4
# EVENT-ANCHORED confirmation (round 3).  A plain standstill() looks at the last 1.2 s whatever happened in it, so a
# feedback that freezes exactly when the motion starts (STATUS stall right after a stationary period) looks like a
# stationary arm and a STOP / fault clear would be "confirmed" at once.  Every confirmation decision (stop
# confirmation, settle after a goal, clear_fault) therefore uses standstill(arm, since=anchor): only samples received at
# or after the anchor (STOP request / goal result / fault entry / begin of the stop) count and the window must be fully
# covered by them.  It also needs a controller_manager health answer (hardware + controller active) whose REQUEST was sent at
# least DRIVER_STATUS_CUTOFF_S after the anchor: a freeze that lasts through the whole anchored window is >= 1.0 s long,
# i.e. the driver has latched ERROR (component / controller leave "active") by then, and this answer shows it.
STANDSTILL_REFUSAL = ("the arm is moving (teleop active?): release the keys and wait - it is not measured at standstill "
                      "(1.2 s without motion)")
REAL_MOVE_DISABLED_MSG = ("'Move there' / 'Descend' are disabled on the REAL robot (start the UI with allow_real_move:=true after the "
                          "supervised first test); use 'Send to RViz' and Plan / Execute in RViz")
DRIVER_STATUS_CUTOFF_S = 1.0       # igus_rebel RebelSafety.hpp: STATUS older than this -> read() ERROR, latched
# Standstill also needs: |last - first| over the window / window <= this (the max-min test alone accepted a steady
# 0.0008 rad/s drift: 0.4 mrad < 0.5 mrad), and, if joint_states carry a finite velocity, |velocity| <= STANDSTILL_MAX_VEL.
STANDSTILL_MAX_DRIFT_RAD_S = 0.0005
STANDSTILL_MAX_VEL_RAD_S = 0.03   # gross check only: the real driver's velocity is a finite difference over >= 15 ms of
                                  # 0.01 deg (1.75e-4 rad) encoder steps, so ONE step of flicker at rest reads ~0.009-0.012
                                  # rad/s; slow real motion is caught by the window range and the drift check above
# After "goal succeeded": wait this long for standstill before reading the result.  Standstill needs STANDSTILL_WINDOW_S
# (1.2 s) of history without motion, so the earliest possible verdict is ~1.2 s after the last motion; 3.0 s leaves the
# 1.8 s the old 1.0 s limit left on top of the old 0.5 s window (the JTC end tolerance is GOAL_TIME_TOLERANCE_S).
SETTLE_WAIT_S = 3.0
# cancel/failed -> "fault" if result + standstill are not both seen by then.  JTC 2.54 holds at the measured position
# at once on cancel, the driver's jog decelerates within a few 100 ms; the result round trip is < 1 s.  The deadline
# must cover the 1.2 s standstill window plus that, and keeps the old ~7.5 s margin on top of the window: 9.0 s.
STOP_CONFIRM_DEADLINE_S = 9.0
# goal_time_tolerance: same 3.0 s as `goal_time` in igus_rebel/config/controller{,_mock}.yaml and
# rebel_dual_table_description/config/controllers{,_mock}.yaml (JTC uses the goal's value when it is non-zero).
GOAL_TIME_TOLERANCE_S = 3.0
HERMITE_SUBSAMPLES = 20            # dense samples per trajectory segment for the cap / monotonicity checks
MONOTONIC_EPS_M = 2e-5             # tolerated reversal along a Cartesian step direction (0.02 mm)
MONOTONIC_EPS_RAD = math.radians(0.01)

# "descend" (approach assist): a straight Cartesian move of the TCP by DESCEND_MIN_M .. DESCEND_MAX_M at constant orientation.
# The 70 mm ceiling is the one limit that replaces the step-size whitelist (a step is <= 10 mm); the 2 mm floor keeps
# the move above the sub-2 mm trouble of the time parameterisation (see _plan_cartesian) and above the encoder noise.
DESCEND_MIN_M = 0.002
DESCEND_MAX_M = 0.070
DESCEND_MAX_ROT_RAD = math.radians(0.5)   # the target orientation must be the arm's current one (a pure translation)

MOCK_PLUGIN = "mock_components/GenericSystem"
AXES = ("x", "y", "z")

# action_msgs/GoalStatus values (fixed by the message definition; asserted against GoalStatus in the tests).  A goal
# result is TERMINAL only if it arrived as a normal response with one of these statuses.
GOAL_STATUS_SUCCEEDED = 4
GOAL_STATUS_CANCELED = 5
GOAL_STATUS_ABORTED = 6
TERMINAL_GOAL_STATUSES = (GOAL_STATUS_SUCCEEDED, GOAL_STATUS_CANCELED, GOAL_STATUS_ABORTED)
# joint_trajectory_controller 2.54 (src/joint_trajectory_controller.cpp, preempt_active_goal): a NEW ACTION GOAL on the same
# controller cancels the active one with status CANCELED, error_code INVALID_GOAL and the error_string below.  A trajectory
# on the controller's topic (MoveIt Servo teleop) does NOT touch the active goal: it silently replaces the trajectory, and
# our goal then ends by the tolerances (ABORTED, "Aborted due to path tolerance violation") or is stopped earlier by our own
# tracking supervision.  Only the controller's own replacement text may be labelled "replaced by another command".
JTC_INVALID_GOAL = -1
JTC_REPLACED_TEXT = "cancelled due to new incoming action"
GOAL_REPLACED_MSG = ("stopped: the trajectory was replaced by another command (the controller received a new action goal) - "
                     "wait for the arm to stand still")
GOAL_UNCONFIRMED_MSG = ("the controller's answer could not be confirmed: press the e-stop if the arm moves; "
                        "restart the trajectory controller or the driver (not just the UI) before moving again")


def goal_replaced(status, error_code, error_string="") -> bool:
    """True only if a VALIDATED goal result carries the controller's own replacement text ("Current goal cancelled due to new
    incoming action."; JTC 2.54: status CANCELED, error_code INVALID_GOAL).  A bare ABORTED / INVALID_GOAL (e.g. "Current
    goal cancelled during deactivate transition.", tolerance violations) is NOT labelled: it keeps the controller's text."""
    if status not in (GOAL_STATUS_CANCELED, GOAL_STATUS_ABORTED):
        return False
    return JTC_REPLACED_TEXT in str(error_string or "").lower()


def validated_result(fut):
    """The goal result response if the result future is done, completed WITHOUT an exception and carries a terminal
    status (SUCCEEDED / CANCELED / ABORTED); None for anything else (not done, exception, no response, unknown
    status).  Only a validated result may count as 'the controller confirmed the goal ended'."""
    try:
        if not fut.done():
            return None
        res = fut.result()
    except Exception:  # noqa: BLE001 - an exceptional future is an unconfirmed answer
        return None
    status = getattr(res, "status", None)
    if res is None or isinstance(status, bool) or not isinstance(status, (int, np.integer)):
        return None
    return res if int(status) in TERMINAL_GOAL_STATUSES else None

# --- "move to a joint goal" (plan_move / execute_move) ---------------------------------------------------------
# Caps of a move.  A move can be tens of degrees long, so it runs at about RViz scaling 0.05 of the 0.785 rad/s
# joint_limits.yaml speed (0.039 rad/s -> 0.04) and at 1/2 of the step acceleration cap.  The TCP cap (40 mm/s) is
# 4x the step cap but only a backstop: at 0.04 rad/s a joint moves a 0.7 m lever at 28 mm/s.  There is no TCP angular cap for moves (joint-space motion; bounded by the six
# joint speeds, <= 6 * 0.04 rad/s = 13.8 deg/s in the worst case).
MOVE_MAX_JOINT_SPEED_RAD_S = 0.04
MOVE_MAX_JOINT_ACC_RAD_S2 = 0.25
MOVE_MAX_TCP_SPEED_MM_S = 40.0
MOVE_PLAN_MAX_AGE_S = 60.0         # a stored plan older than this can no longer be executed
MOVE_PLANNING_TIME_S = 5.0         # allowed_planning_time of the MoveIt request
MOVE_PLANNING_ATTEMPTS = 3
MOVE_PLAN_SCALING = 0.05           # max_velocity / max_acceleration_scaling_factor of the MoveIt request (timing is ours)
MOVE_GOAL_TOL_RAD = 1e-4           # JointConstraint tolerance; also: a goal closer than this to the start is "already there"
MOVE_PLAN_WAIT_S = 30.0            # wait for /plan_kinematic_path (planning time + time-parameterisation adapters)
MOVE_MAX_DENSE_STATES = 6000       # /check_state_validity calls per plan (0.25 deg spacing => ~25 rad of joint travel)
MOVE_SMOOTH_STEP_RAD = 0.004      # smooth_path resampling step (Euclidean in joint space; 0.23 deg, so <= 0.25 deg per joint)
MOVE_SMOOTH_WINDOW_RAD = 0.2       # corner rounding window along the path (a TOTG-blended MoveIt path is already smoother)
# A longer move is refused (something is off: a huge detour).  300 s: a calibration move is a few radians at most
# (~75 s at 0.04 rad/s), and the real-hardware enable (RealLock, 600 s) is checked only when a motion is accepted and
# right before its goal is sent, NOT while it executes - so a move must always be much shorter than the enable.  STOP,
# cancel, fault handling and supervision never depend on the enable (see JogCore.request_stop).
MOVE_MAX_DURATION_S = 300.0
MOVE_RAMP_FRACTIONS = tuple(np.linspace(0.02, 0.5, 49))   # acceleration-ramp share of the duration tried by retime_path
# Executed curve vs validated path.  Validity (/check_state_validity) is checked on the smoothed dense polyline, but
# the controller interpolates the re-timed points as a cubic Hermite spline.  The dense Hermite samples of the final
# trajectory must stay within MOVE_MAX_PATH_DEV_RAD (joint-space Euclidean distance) of that polyline.  TCP bound: a
# deviation of 0.002 rad in ONE joint moves the TCP by at most 0.002 * 0.9 m (reach of the ReBeL with the XEG-32 and
# knife) = 1.8 mm; if all six joints deviate by the maximum at once the Euclidean bound allows sqrt(6) * 1.8 mm = 4.4 mm
# at the very worst (the distal joints have much shorter levers, so real deviations are far below that).  The path
# points are 0.004 rad apart and the retimed points are 0.002 rad apart at most, i.e. the real deviation (sagitta of
# the spline between two points on the polyline) is a few micro-radian: this bound is a tripwire, not a tolerance
# that is expected to be used.
MOVE_MAX_PATH_DEV_RAD = 0.002
# Collision check of the EXECUTED curve: /check_state_validity is called on dense samples of the cubic Hermite curve
# through the final re-timed points (what the JTC really interpolates), consecutive samples at most
# CURVE_CHECK_SPACING_DEG apart in every joint, in addition to the check of the planned polyline.  More than
# CURVE_CHECK_MAX_STATES samples (serial service calls) -> the motion is refused.
CURVE_CHECK_SPACING_DEG = 0.25
CURVE_CHECK_MAX_STATES = 6000
ON_POLYLINE_TOL_RAD = 1e-9         # the re-timed POINTS are interpolated on the polyline: they must lie on it exactly
# Peaks between the Hermite subsamples: joint speed is a quadratic per segment, so its maximum is computed analytically
# (hermite_peak_velocity) and joint acceleration is linear per segment (maximum at the segment ends, sampled exactly).
# The TCP speed is a finite difference of the dense FK samples; the peak could hide between two subsamples (spacing
# dt/20): the speed varies smoothly over a 50 ms segment, the sampling error of a smooth peak is O(spacing^2) (<0.1 % for
# the quadratic profiles seen here), and a factor of 1.01 covers it.
TCP_SPEED_SAFETY = 1.01
# Runtime supervision while a goal executes (steps and moves)
SUPERVISION_STALE_S = JOINT_STATE_MAX_AGE_S       # joint_states older than this (or non-finite) -> STOP
SUPERVISION_SPEED_FACTOR = 2.0                    # measured joint speed above FACTOR * cap of the motion -> STOP
# Finite difference over at least this long.  0.1 s was too short for the real driver's feedback: a 300 ms freeze of
# repeated positions followed by a catch-up jump at a true 0.04 rad/s reads 0.12 rad/s over 0.1 s (a false trip, factor
# 3).  Over 0.5 s the same event reads at most 0.04 * 0.5 / 0.5 = 0.04 rad/s plus the quantisation (1.75e-4 / 0.5 s),
# while a sustained runaway still trips within ~0.6 s (check() runs every 50 ms).
SUPERVISION_SPEED_WINDOW_S = 0.5
MOVE_MAX_TRACKING_ERR_RAD = 0.05                  # |measured - reference(t)| of any joint, moves (cap 0.04 rad/s: 1.25 s lag)
# Steps: with the current controller gains (p 1.0 / ff 0.4) the steady following lag is ~0.47 x the joint velocity, i.e.
# ~0.045 rad at the 0.1 rad/s step cap, plus the latency of the controller start (some tens of ms, <= 0.005 rad at
# that speed).  0.02 rad would trip on a healthy arm at the cap; 0.05 rad (2.9 deg, ~45 mm at a 0.9 m lever) is the
# same bound as for moves and still stops an arm that has run away from its reference.
STEP_MAX_TRACKING_ERR_RAD = 0.05
# Re-check of the planning scene at EVERY execute (no age grace any more): a coarse subset of the stored final Hermite
# curve samples (the ones curve_check_states produced at planning time), about every 1 deg of joint travel, at most
# MOVE_RECHECK_MAX_CALLS serial /check_state_validity calls.
MOVE_RECHECK_SPACING_RAD = math.radians(1.0)
MOVE_RECHECK_MAX_CALLS = 300
# Freeze detection while a goal executes: the reference trajectory commands some joint faster than
# FREEZE_REF_SPEED_RAD_S (one 0.01 deg driver count, 1.745e-4 rad, per ~35 ms) but NO joint position changed for
# FREEZE_MAX_S -> the feedback is frozen (driver STATUS stall: positions are repeated with fresh stamps) -> STOP.
FREEZE_REF_SPEED_RAD_S = 0.005
FREEZE_MAX_S = 0.25
MOTION_QUEUE_TTL_S = 30.0          # a reserved (queued) motion request that never ran is forgotten after this

PHASES = ("planning", "planned", "checking", "executing", "stopping", "done", "refused", "failed", "stopped", "fault")


# ---------------------------------------------------------------------------------------------------
# arm configuration

@dataclass(frozen=True)
class ArmConfig:
    key: str                    # "left" | "right"
    group: str                  # MoveIt planning group
    tip_link: str               # TCP link (xeg32_tool_tip)
    controller: str             # FollowJointTrajectory controller
    joint_prefix: str
    hardware: str               # ros2_control hardware component name
    planning_frame: str
    joint_names: Tuple[str, ...]


def arm_configs(mode: str) -> Dict[str, ArmConfig]:
    """Per arm key the MoveIt / ros2_control names for a recorder mode (verified against
    rebel_dual_table_moveit_config/{rebel_dual_table.srdf,moveit_controllers.yaml},
    rebel_dual_table_description/urdf/rebel_dual_table.ros2_control.xacro and, for the single-arm stack,
    igus_rebel_moveit_config/config/{igus_rebel2.srdf,moveit_controllers.yaml} + igus_rebel2.control.xacro)."""
    if mode == "dual":
        out = {}
        for arm in ("left", "right"):
            prefix = f"arm_{arm}_"
            out[arm] = ArmConfig(arm, f"arm_{arm}", f"{prefix}xeg32_tool_tip", f"arm_{arm}_trajectory_controller",
                                 prefix, f"arm_{arm}_hw", "world",
                                 tuple(f"{prefix}joint{i}" for i in range(1, 7)))
        return out
    if mode in ("single_left", "single_right"):
        arm = mode.split("_")[1]
        # srdf: group igus_rebel_arm, chain world -> link6 (the TCP link xeg32_tool_tip is rigidly attached);
        # ros2_control block "<arm_id>_controller" with arm_id "rebel_arm"
        return {arm: ArmConfig(arm, "igus_rebel_arm", "xeg32_tool_tip", "rebel_arm_trajectory_controller", "",
                               "rebel_arm_controller", "world", tuple(f"joint{i}" for i in range(1, 7)))}
    raise ValueError(f"unknown mode '{mode}', expected dual, single_left or single_right")


# ---------------------------------------------------------------------------------------------------
# request validation

def _is_number(v) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool) and math.isfinite(float(v))


def _canonical_size(size, allowed: Sequence[float]) -> Optional[float]:
    if not _is_number(size):
        return None
    for a in allowed:
        if abs(float(size) - float(a)) < 1e-9:
            return float(a)
    return None


def validate_request(arms: Sequence[str], arm, kind, axis, sign, size, frame="world") -> Tuple[Optional[dict], str]:
    """Normalise a step request.  Returns (request, "") or (None, reason).  Nothing but the whitelisted values
    passes: unknown arm / mode / axis / size / frame are refused."""
    if arm not in ("left", "right"):
        return None, f"unknown arm {arm!r} (expected 'left' or 'right')"
    if arm not in arms:
        return None, f"arm '{arm}' is not active in this mode (active arms: {', '.join(arms)})"
    if kind not in ("cart", "rot", "joint"):
        return None, f"unknown mode {kind!r} (expected 'cart', 'rot' or 'joint')"
    if not _is_number(sign) or float(sign) not in (1.0, -1.0):
        return None, f"sign must be +1 or -1, got {sign!r}"
    sign = int(sign)
    if kind == "joint":
        ax = axis
        if isinstance(ax, str) and ax.strip().isdigit():
            ax = int(ax.strip())
        if isinstance(ax, (float, np.floating)) and float(ax).is_integer():
            ax = int(ax)
        if isinstance(ax, bool) or not isinstance(ax, (int, np.integer)) or not 1 <= int(ax) <= 6:
            return None, f"joint axis must be 1..6, got {axis!r}"
        ax = int(ax)
        s = _canonical_size(size, STEP_SIZES["joint_deg"])
        if s is None:
            return None, f"joint step size {size!r} deg is not allowed (allowed: {STEP_SIZES['joint_deg']})"
        return {"arm": arm, "kind": kind, "axis": ax, "sign": sign, "size": s, "frame": "joint"}, ""
    if not isinstance(axis, str) or axis.lower() not in AXES:
        return None, f"axis must be 'x', 'y' or 'z' for {kind}, got {axis!r}"
    if frame not in ("world", "tool"):
        return None, f"frame must be 'world' or 'tool', got {frame!r}"
    key = "cart_mm" if kind == "cart" else "rot_deg"
    s = _canonical_size(size, STEP_SIZES[key])
    if s is None:
        unit = "mm" if kind == "cart" else "deg"
        return None, f"{kind} step size {size!r} {unit} is not allowed (allowed: {STEP_SIZES[key]})"
    return {"arm": arm, "kind": kind, "axis": axis.lower(), "sign": sign, "size": s, "frame": frame}, ""


def validate_descend_target(arms: Sequence[str], arm, T) -> Tuple[Optional[dict], str]:
    """Normalise a descend request: ``T`` is the target TCP pose (4x4, finite, proper rotation) in the arm's base frame.
    Returns (request, "") or (None, reason).  The length / orientation limits are checked against the measured TCP in
    the worker (the pose is not known here)."""
    if arm not in ("left", "right"):
        return None, f"unknown arm {arm!r} (expected 'left' or 'right')"
    if arm not in arms:
        return None, f"arm '{arm}' is not active in this mode (active arms: {', '.join(arms)})"
    try:
        M = np.asarray(T, dtype=float)
    except (TypeError, ValueError):
        return None, "descend target must be a 4x4 pose"
    if M.shape != (4, 4) or not np.all(np.isfinite(M)):
        return None, "descend target must be a finite 4x4 pose"
    R = M[:3, :3]
    if not (np.allclose(R @ R.T, np.eye(3), atol=1e-6) and abs(float(np.linalg.det(R)) - 1.0) < 1e-6
            and np.allclose(M[3], [0, 0, 0, 1], atol=1e-9)):
        return None, "descend target is not a rigid transform"
    return {"arm": arm, "kind": "descend", "axis": "n", "sign": 1, "size": float("nan"), "frame": "base",
            "T_base_target": M.copy()}, ""


def fixed_transform(urdf, parent: str, child: str) -> np.ndarray:
    """T_parent_child of a chain of FIXED joints (the product of the joint origins); raises KinematicsError if the chain
    has a movable joint."""
    ch = urdf.chain(parent, child)
    if ch.n_dof:
        raise KinematicsError(f"{parent} -> {child} is not a fixed transform (movable joints {ch.joint_names})")
    T = np.eye(4)
    for T0 in ch.T0:
        T = T @ T0
    return T


# ---------------------------------------------------------------------------------------------------
# rotation helpers (numpy only)

def rot_axis(index: int, angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    if index == 0:
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)
    if index == 1:
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def quat_to_R(x: float, y: float, z: float, w: float) -> np.ndarray:
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        raise ValueError("zero quaternion")
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]], dtype=float)


def R_to_quat(R: np.ndarray) -> Tuple[float, float, float, float]:
    """Rotation matrix -> unit quaternion (x, y, z, w), w >= 0."""
    m = np.asarray(R, dtype=float)
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w])
    q /= np.linalg.norm(q)
    if q[3] < 0:
        q = -q
    return float(q[0]), float(q[1]), float(q[2]), float(q[3])


def rotation_angle(R1: np.ndarray, R2: np.ndarray) -> float:
    """Geodesic angle [rad] between two rotation matrices."""
    M = np.asarray(R1).T @ np.asarray(R2)
    v = np.array([M[2, 1] - M[1, 2], M[0, 2] - M[2, 0], M[1, 0] - M[0, 1]])
    return float(math.atan2(0.5 * np.linalg.norm(v), 0.5 * (np.trace(M) - 1.0)))


def target_pose(p: np.ndarray, R: np.ndarray, kind: str, axis, sign: int, size: float,
                frame: str) -> Tuple[np.ndarray, np.ndarray]:
    """TCP pose (position p, rotation R in the planning frame) after one cart / rot step.

    cart: p + sign*size_mm*1e-3 along the world axis, or along R's axis for frame 'tool'; orientation unchanged.
    rot:  rotation of sign*size_deg about the TCP point; 'world' pre-multiplies (R' = Rw R), 'tool' post-multiplies
          (R' = R Rt).  The position is unchanged."""
    p = np.asarray(p, dtype=float).copy()
    R = np.asarray(R, dtype=float).copy()
    i = AXES.index(str(axis).lower())
    if kind == "cart":
        d = np.zeros(3)
        d[i] = sign * size * 1e-3
        if frame == "tool":
            d = R @ d
        return p + d, R
    if kind == "rot":
        Rs = rot_axis(i, sign * math.radians(size))
        return p, (Rs @ R if frame == "world" else R @ Rs)
    raise ValueError(f"target_pose: kind must be cart or rot, got {kind!r}")


def cart_waypoints(p0, R0, p1, R1) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Intermediate TCP poses (excluding the start, including the target) at <= 0.5 mm / 0.1 deg spacing: straight
    line in position and a constant-axis rotation (geodesic) in orientation."""
    dist = float(np.linalg.norm(p1 - p0))
    ang = rotation_angle(R0, R1)
    n = max(1, int(math.ceil(max(dist / CART_WAYPOINT_SPACING_M, ang / ROT_WAYPOINT_SPACING_RAD) - 1e-9)))
    M = R0.T @ R1
    v = np.array([M[2, 1] - M[1, 2], M[0, 2] - M[2, 0], M[1, 0] - M[0, 1]])
    out = []
    for k in range(1, n + 1):
        f = k / n
        if ang < 1e-12:
            Rk = R1.copy()
        elif abs(math.pi - ang) < 1e-6:  # not reachable with <= 5 deg steps
            raise ValueError("rotation of ~180 deg is not supported")
        else:
            u = v / np.linalg.norm(v)
            K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
            a = f * ang
            Rk = R0 @ (np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * (K @ K))
        out.append((p0 + f * (p1 - p0), Rk))
    out[-1] = (p1.copy(), R1.copy())
    return out


# ---------------------------------------------------------------------------------------------------
# re-timing

def smootherstep(tau: np.ndarray) -> np.ndarray:
    tau = np.clip(tau, 0.0, 1.0)
    return tau ** 3 * (10.0 - 15.0 * tau + 6.0 * tau ** 2)


def smootherstep_rate(tau: np.ndarray) -> np.ndarray:
    tau = np.clip(tau, 0.0, 1.0)
    return 30.0 * tau ** 2 * (1.0 - tau) ** 2


@dataclass
class Retimed:
    times: np.ndarray          # (M,) s, times[0] = 0
    q: np.ndarray              # (M, n) rad, q[0] == the given start, q[-1] == the given end
    qd: np.ndarray             # (M, n) rad/s, zero at both ends
    duration: float


def retime(Q: np.ndarray, tcp_pos: np.ndarray, tcp_rot: np.ndarray,
           max_lin_m_s: float = MAX_CART_SPEED_MM_S * 1e-3, max_ang_rad_s: float = math.radians(MAX_ROT_SPEED_DEG_S),
           max_joint_rad_s: float = MAX_JOINT_SPEED_RAD_S, max_joint_acc: float = MAX_JOINT_ACCEL_RAD_S2,
           min_duration: float = MIN_DURATION_S, dt: float = SAMPLE_DT_S,
           margin: float = DURATION_MARGIN) -> Retimed:
    """Time-scale the joint polyline Q (N, n) whose TCP poses are tcp_pos (N,3) / tcp_rot (N,3,3).

    Each segment gets the time it would need at exactly the speed caps; the path parameter s is the cumulative
    share of that time, so ds/dt = (total need R) * (cap fraction).  With the quintic s(t) = u(t/T) (zero
    velocity and acceleration at both ends, peak du/dtau = 1.875) every segment runs at most R*1.875/T of its
    caps, hence T = max(min_duration, 1.875*R*margin, acceleration bound)."""
    Q = np.asarray(Q, dtype=float)
    tcp_pos = np.asarray(tcp_pos, dtype=float)
    tcp_rot = np.asarray(tcp_rot, dtype=float)
    if Q.ndim != 2 or len(Q) < 2 or len(tcp_pos) != len(Q) or len(tcp_rot) != len(Q):
        raise ValueError("retime needs at least two waypoints with matching TCP poses")
    keep = [0]
    for i in range(1, len(Q)):
        if np.max(np.abs(Q[i] - Q[keep[-1]])) > 0.0 or i == len(Q) - 1 and len(keep) == 1:
            keep.append(i)
    Q, tcp_pos, tcp_rot = Q[keep], tcp_pos[keep], tcp_rot[keep]
    K = len(Q) - 1
    dq = np.diff(Q, axis=0)
    lin = np.linalg.norm(np.diff(tcp_pos, axis=0), axis=1)
    ang = np.array([rotation_angle(tcp_rot[i], tcp_rot[i + 1]) for i in range(K)])
    need = np.maximum.reduce([lin / max_lin_m_s, ang / max_ang_rad_s, np.max(np.abs(dq), axis=1) / max_joint_rad_s])
    total = float(np.sum(need))
    if total <= 0.0:
        raise ValueError("zero-length step")
    path_len = np.sum(np.abs(dq), axis=0)                 # per-joint path length (>= net travel)
    t_acc = margin * math.sqrt(QUINTIC_ACCEL_PEAK * float(np.max(path_len)) / max_joint_acc)
    T = max(min_duration, QUINTIC_PEAK * total * margin, t_acc)
    s_grid = np.concatenate([[0.0], np.cumsum(need) / total])
    s_grid[-1] = 1.0
    ds = np.diff(s_grid)
    ok = ds > 0
    if not np.all(ok):  # segments with zero need (cannot happen after the duplicate drop, but stay safe)
        s_grid = np.concatenate([[0.0], np.cumsum(np.where(ok, ds, 1e-12))])
        s_grid /= s_grid[-1]
        ds = np.diff(s_grid)
    n = max(2, int(math.ceil(T / dt)) + 1)
    times = np.linspace(0.0, T, n)
    tau = times / T
    s = smootherstep(tau)
    sdot = smootherstep_rate(tau) / T
    k = np.clip(np.searchsorted(s_grid, s, side="right") - 1, 0, K - 1)
    frac = (s - s_grid[k]) / ds[k]
    q = Q[k] + frac[:, None] * dq[k]
    qd = dq[k] / ds[k][:, None] * sdot[:, None]
    q[0], q[-1] = Q[0], Q[-1]
    qd[0] = 0.0
    qd[-1] = 0.0
    return Retimed(times, q, qd, float(T))


def trajectory_peaks(times: np.ndarray, q: np.ndarray, fk: Callable[[np.ndarray], np.ndarray]) -> Dict[str, float]:
    """Peak joint speed [rad/s], TCP speed [mm/s] and TCP angular speed [deg/s] of a sampled trajectory (finite
    differences over the sample intervals); ``fk(q (M,n)) -> (M,4,4)`` is the TCP forward kinematics."""
    times = np.asarray(times, dtype=float)
    q = np.asarray(q, dtype=float)
    T = fk(q)
    dt = np.diff(times)
    joint = np.max(np.abs(np.diff(q, axis=0)) / dt[:, None])
    lin = np.linalg.norm(np.diff(T[:, :3, 3], axis=0), axis=1) / dt
    ang = np.array([rotation_angle(T[i, :3, :3], T[i + 1, :3, :3]) for i in range(len(T) - 1)]) / dt
    return {"joint_rad_s": float(joint), "tcp_mm_s": float(np.max(lin) * 1e3), "tcp_deg_s": float(math.degrees(np.max(ang)))}


def rotation_angles(Ra: np.ndarray, Rb: np.ndarray) -> np.ndarray:
    """Batched geodesic angles between rotation stacks (N,3,3) and (N,3,3) or one (3,3)."""
    Ra = np.asarray(Ra, dtype=float)
    Rb = np.asarray(Rb, dtype=float)
    M = np.einsum("...ji,...jk->...ik", np.broadcast_to(Ra, np.broadcast(Ra, Rb).shape), np.broadcast_to(Rb, np.broadcast(Ra, Rb).shape))
    v = np.stack([M[..., 2, 1] - M[..., 1, 2], M[..., 0, 2] - M[..., 2, 0], M[..., 1, 0] - M[..., 0, 1]], axis=-1)
    return np.arctan2(0.5 * np.linalg.norm(v, axis=-1), 0.5 * (np.trace(M, axis1=-2, axis2=-1) - 1.0))


def hermite_dense(times: np.ndarray, q: np.ndarray, qd: np.ndarray,
                  n_sub: int = HERMITE_SUBSAMPLES) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cubic Hermite interpolation of the trajectory points, exactly what joint_trajectory_controller samples
    when a goal carries positions and velocities (no accelerations): (t, q, qd, qdd), each segment sampled at
    n_sub+1 points (segment ends included once)."""
    times = np.asarray(times, dtype=float)
    q = np.asarray(q, dtype=float)
    qd = np.asarray(qd, dtype=float)
    s = np.linspace(0.0, 1.0, n_sub + 1)[None, :, None]            # (1, S, 1)
    h = np.diff(times)[:, None, None]                              # (K, 1, 1)
    q0, q1 = q[:-1, None, :], q[1:, None, :]
    v0, v1 = qd[:-1, None, :] * h, qd[1:, None, :] * h
    h00, h10, h01, h11 = 2 * s ** 3 - 3 * s ** 2 + 1, s ** 3 - 2 * s ** 2 + s, -2 * s ** 3 + 3 * s ** 2, s ** 3 - s ** 2
    d00, d10, d01, d11 = 6 * s ** 2 - 6 * s, 3 * s ** 2 - 4 * s + 1, -6 * s ** 2 + 6 * s, 3 * s ** 2 - 2 * s
    a00, a10, a01, a11 = 12 * s - 6, 6 * s - 4, -12 * s + 6, 6 * s - 2
    Q = h00 * q0 + h10 * v0 + h01 * q1 + h11 * v1
    V = (d00 * q0 + d10 * v0 + d01 * q1 + d11 * v1) / h
    A = (a00 * q0 + a10 * v0 + a01 * q1 + a11 * v1) / h ** 2
    T = times[:-1, None] + s[0, :, 0][None, :] * h[:, :, 0]
    n = q.shape[1]
    return T.reshape(-1), Q.reshape(-1, n), V.reshape(-1, n), A.reshape(-1, n)


def hermite_eval(times: np.ndarray, q: np.ndarray, qd: np.ndarray, t: float) -> np.ndarray:
    """Position of the cubic Hermite trajectory at time t (clamped to the first / last point)."""
    times = np.asarray(times, dtype=float)
    q = np.asarray(q, dtype=float)
    qd = np.asarray(qd, dtype=float)
    t = float(min(max(t, times[0]), times[-1]))
    i = int(np.clip(np.searchsorted(times, t, side="right") - 1, 0, len(times) - 2))
    h = float(times[i + 1] - times[i])
    s = (t - float(times[i])) / h
    h00, h10, h01, h11 = 2 * s ** 3 - 3 * s ** 2 + 1, s ** 3 - 2 * s ** 2 + s, -2 * s ** 3 + 3 * s ** 2, s ** 3 - s ** 2
    return h00 * q[i] + h10 * h * qd[i] + h01 * q[i + 1] + h11 * h * qd[i + 1]


def hermite_eval_velocity(times: np.ndarray, q: np.ndarray, qd: np.ndarray, t: float) -> np.ndarray:
    """Joint velocity of the cubic Hermite trajectory at time t (clamped to the first / last point; the re-timed
    trajectories start and end at rest)."""
    times = np.asarray(times, dtype=float)
    q = np.asarray(q, dtype=float)
    qd = np.asarray(qd, dtype=float)
    t = float(min(max(t, times[0]), times[-1]))
    i = int(np.clip(np.searchsorted(times, t, side="right") - 1, 0, len(times) - 2))
    h = float(times[i + 1] - times[i])
    s = (t - float(times[i])) / h
    d00, d10, d01, d11 = 6 * s ** 2 - 6 * s, 3 * s ** 2 - 4 * s + 1, -6 * s ** 2 + 6 * s, 3 * s ** 2 - 2 * s
    return (d00 * q[i] + d10 * h * qd[i] + d01 * q[i + 1] + d11 * h * qd[i + 1]) / h


def hermite_peak_velocity(times: np.ndarray, q: np.ndarray, qd: np.ndarray) -> float:
    """EXACT maximum |joint velocity| of the cubic Hermite trajectory (all joints, all segments).  Per segment the
    velocity is the quadratic (A s^2 + B s + C) / h in the normalised time s, so the maximum is at s = 0, s = 1 or at
    the stationary point s = -B / 2A; no subsampling can miss an interior peak."""
    times = np.asarray(times, dtype=float)
    q = np.asarray(q, dtype=float)
    qd = np.asarray(qd, dtype=float)
    h = np.diff(times)[:, None]
    q0, q1 = q[:-1], q[1:]
    v0, v1 = qd[:-1] * h, qd[1:] * h
    A = 6 * q0 + 3 * v0 - 6 * q1 + 3 * v1
    B = -6 * q0 - 4 * v0 + 6 * q1 - 2 * v1
    C = v0
    cand = [np.abs(C), np.abs(A + B + C)]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        sstar = -B / (2 * A)
        inside = np.isfinite(sstar) & (sstar > 0.0) & (sstar < 1.0)
        sin = np.where(inside, sstar, 0.0)
    cand.append(np.where(inside, np.abs(A * sin ** 2 + B * sin + C), 0.0))
    return float(np.max(np.max(cand, axis=0) / h))


def curve_check_states(times: np.ndarray, q: np.ndarray, qd: np.ndarray,
                       max_step_rad: float = math.radians(CURVE_CHECK_SPACING_DEG),
                       max_states: int = CURVE_CHECK_MAX_STATES) -> np.ndarray:
    """Dense samples of the cubic Hermite curve through the re-timed points (what the JTC executes), for the validity
    check of the executed curve: first and last point included, consecutive samples at most `max_step_rad` apart in
    EVERY joint (the curve is sampled at dt/HERMITE_SUBSAMPLES and thinned by the cumulative max-norm arc length,
    which bounds the max-norm distance of any two samples).  Raises ValueError for a non-finite curve or more than
    `max_states` samples."""
    _, Qh, _, _ = hermite_dense(times, q, qd, HERMITE_SUBSAMPLES)
    if not np.all(np.isfinite(Qh)):
        raise ValueError("the executed curve contains non-finite values")
    d = np.max(np.abs(np.diff(Qh, axis=0)), axis=1)
    if len(d) == 0:
        return Qh.copy()
    # the fine sampling must be much finer than the check spacing, else the thinning below could overshoot it
    if float(np.max(d)) > 0.1 * max_step_rad:
        raise ValueError(f"the executed curve moves {math.degrees(float(np.max(d))):.3f} deg between fine samples "
                         f"(too fast to check at {math.degrees(max_step_rad):.2f} deg spacing)")
    c = np.concatenate([[0.0], np.cumsum(d)])
    total = float(c[-1])
    n = max(1, int(math.ceil(total / (0.9 * max_step_rad))))
    if n + 1 > max_states:
        raise ValueError(f"the executed curve needs {n + 1} validity checks at {math.degrees(max_step_rad):.2f} deg "
                         f"spacing (limit {max_states})")
    idx = np.searchsorted(c, np.linspace(0.0, total, n + 1), side="left")
    idx = np.unique(np.concatenate([[0], np.clip(idx, 0, len(Qh) - 1), [len(Qh) - 1]]))
    return Qh[idx]


def polyline_deviation(poly: np.ndarray, pts: np.ndarray, reach: int = 3) -> float:
    """Largest Euclidean joint-space distance of any point of `pts` (M, n) from the polyline `poly` (N, n).  The nearest
    polyline vertex is found with a KD-tree and only the segments within `reach` vertices of it are measured: the
    result can only be LARGER than the true distance (conservative), never smaller."""
    from scipy.spatial import cKDTree
    poly = np.asarray(poly, dtype=float)
    pts = np.asarray(pts, dtype=float)
    if len(poly) == 0 or len(pts) == 0:
        return 0.0
    if not (np.all(np.isfinite(poly)) and np.all(np.isfinite(pts))):
        return float("inf")
    if len(poly) == 1:
        return float(np.max(np.linalg.norm(pts - poly[0], axis=1)))
    _, idx = cKDTree(poly).query(pts)
    best = np.full(len(pts), np.inf)
    last = len(poly) - 1
    for off in range(-reach, reach):
        ia = np.clip(idx + off, 0, last)
        ib = np.clip(idx + off + 1, 0, last)
        a, b = poly[ia], poly[ib]
        ab = b - a
        denom = np.sum(ab * ab, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            f = np.where(denom > 0, np.sum((pts - a) * ab, axis=1) / denom, 0.0)
        f = np.clip(f, 0.0, 1.0)
        best = np.minimum(best, np.linalg.norm(pts - (a + f[:, None] * ab), axis=1))
    return float(np.max(best))


def coarse_states(Q: np.ndarray, spacing_rad: float = MOVE_RECHECK_SPACING_RAD,
                  max_calls: int = MOVE_RECHECK_MAX_CALLS) -> np.ndarray:
    """A coarse subset of the dense path Q (N, n) for a validity re-check: first and last point plus a point about every
    `spacing_rad` of joint-space arc length (Euclidean), widened so that at most `max_calls` states result."""
    Q = np.asarray(Q, dtype=float)
    if len(Q) <= 2:
        return Q.copy()
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))])
    total = float(arc[-1])
    if total <= 0.0:
        return Q[[0, -1]].copy()
    spacing = max(float(spacing_rad), total / max(1, max_calls - 1))
    marks = np.arange(0.0, total, spacing)
    idx = np.searchsorted(arc, marks, side="left")
    idx = np.unique(np.concatenate([[0], np.clip(idx, 0, len(Q) - 1), [len(Q) - 1]]))
    return Q[idx]


def analyze_trajectory(times: np.ndarray, q: np.ndarray, qd: np.ndarray,
                       fk: Callable[[np.ndarray], np.ndarray], n_sub: int = HERMITE_SUBSAMPLES) -> dict:
    """Peaks of what the controller will actually execute: the densely sampled cubic Hermite interpolation of the
    points (not finite differences of the points).  Joint speed / acceleration are analytic; the TCP speeds are
    differences of the dense FK samples (spacing dt/n_sub).  Returns the peaks plus the dense arrays."""
    t, Qd, Vd, Ad = hermite_dense(times, q, qd, n_sub)
    T = fk(Qd)
    dt = np.diff(t)
    keep = dt > 1e-12
    lin = np.linalg.norm(np.diff(T[:, :3, 3], axis=0), axis=1)[keep] / dt[keep]
    ang = rotation_angles(T[:-1, :3, :3], T[1:, :3, :3])[keep] / dt[keep]
    # joint speed: exact (analytic) maximum, never smaller than the sampled one; acceleration is linear per segment so
    # its maximum is at the (sampled) segment ends; TCP speeds carry TCP_SPEED_SAFETY (see its comment)
    vmax = max(float(np.max(np.abs(Vd))), hermite_peak_velocity(times, q, qd))
    return {"joint_rad_s": vmax, "joint_acc_rad_s2": float(np.max(np.abs(Ad))),
            "tcp_mm_s": float(np.max(lin) * 1e3 * TCP_SPEED_SAFETY),
            "tcp_deg_s": float(math.degrees(np.max(ang)) * TCP_SPEED_SAFETY),
            "t": t, "q": Qd, "T": T}


def cap_ratios(an: dict) -> Dict[str, float]:
    return {"joint speed": an["joint_rad_s"] / MAX_JOINT_SPEED_RAD_S,
            "joint acceleration": an["joint_acc_rad_s2"] / MAX_JOINT_ACCEL_RAD_S2,
            "TCP speed": an["tcp_mm_s"] / MAX_CART_SPEED_MM_S,
            "TCP angular speed": an["tcp_deg_s"] / MAX_ROT_SPEED_DEG_S}


def cap_violations(an: dict) -> List[str]:
    """Which caps the dense Hermite trajectory exceeds (empty = within all caps)."""
    caps = {"joint speed": ("joint_rad_s", MAX_JOINT_SPEED_RAD_S, "rad/s"),
            "joint acceleration": ("joint_acc_rad_s2", MAX_JOINT_ACCEL_RAD_S2, "rad/s^2"),
            "TCP speed": ("tcp_mm_s", MAX_CART_SPEED_MM_S, "mm/s"),
            "TCP angular speed": ("tcp_deg_s", MAX_ROT_SPEED_DEG_S, "deg/s")}
    return [f"{name} {an[key]:.4g} > {cap:g} {unit}" for name, (key, cap, unit) in caps.items() if an[key] > cap]


def step_progress(kind: str, an: dict, p0: np.ndarray, R0: np.ndarray, direction=None,
                  joint_index: int = 0, sign: int = 1) -> np.ndarray:
    """Signed progress along the step at every dense sample: distance along `direction` (cart, metres), rotation
    angle from the start orientation (rot, rad) or signed joint travel (joint, rad)."""
    if kind == "cart":
        return (an["T"][:, :3, 3] - np.asarray(p0)) @ np.asarray(direction)
    if kind == "rot":
        return rotation_angles(np.asarray(R0), an["T"][:, :3, :3])
    return sign * (an["q"][:, joint_index] - an["q"][0, joint_index])


def max_reversal(progress: np.ndarray) -> float:
    """Largest drop of the progress below its running maximum (0 for a monotone path)."""
    progress = np.asarray(progress, dtype=float)
    return float(np.max(np.maximum.accumulate(progress) - progress)) if len(progress) else 0.0


def plan_timing(Q: np.ndarray, tcp_pos: np.ndarray, tcp_rot: np.ndarray,
                fk: Callable[[np.ndarray], np.ndarray]) -> Tuple[Retimed, dict]:
    """retime() until the cubic Hermite interpolation of the points stays inside every cap; the duration is
    lengthened by the worst overshoot ratio.  Returns (trajectory, analysis); the caller must still check
    cap_violations() (it fails only if six lengthenings did not converge)."""
    min_duration = MIN_DURATION_S
    for _ in range(6):
        rt = retime(Q, tcp_pos, tcp_rot, min_duration=min_duration)
        an = analyze_trajectory(rt.times, rt.q, rt.qd, fk)
        worst = max(cap_ratios(an).values())
        if worst <= 1.0:
            break
        min_duration = rt.duration * worst * 1.02
    return rt, an


# ---------------------------------------------------------------------------------------------------
# joint-goal moves: goal validation, densifying, re-timing of an arbitrary joint path

def validate_joint_goal(joint_names: Sequence[str], q_goal) -> Tuple[Optional[np.ndarray], str]:
    """q_goal must be a mapping with exactly the arm's joint names, every value a finite real number.  Returns the
    goal as an array in joint_names order, or (None, reason)."""
    if not isinstance(q_goal, Mapping):
        return None, "q_goal must be a mapping joint name -> radians"
    names = list(joint_names)
    missing = [n for n in names if n not in q_goal]
    extra = [n for n in q_goal if n not in names]
    if missing or extra:
        return None, ("q_goal must contain exactly the six joints of the arm"
                      + (f"; missing {missing}" if missing else "") + (f"; unexpected {extra}" if extra else ""))
    bad = [n for n in names if not _is_number(q_goal[n])]
    if bad:
        return None, f"q_goal values must be finite numbers (radians); bad: {bad}"
    return np.array([float(q_goal[n]) for n in names], dtype=float), ""


def densify_path(Q: np.ndarray, max_step_rad: float = math.radians(JOINT_CHECK_SPACING_DEG),
                 max_states: int = MOVE_MAX_DENSE_STATES) -> np.ndarray:
    """Linear joint-space interpolation of the polyline Q (N, n) so that no two consecutive points differ by more
    than max_step_rad in any joint.  The original points are kept (first and last exactly).  Raises ValueError for
    fewer than two points, a non-finite point or more than max_states points."""
    Q = np.asarray(Q, dtype=float)
    if Q.ndim != 2 or len(Q) < 2:
        raise ValueError("a path needs at least two points")
    if not np.all(np.isfinite(Q)):
        raise ValueError("path contains non-finite values")
    d = np.max(np.abs(np.diff(Q, axis=0)), axis=1)
    n = np.maximum(1, np.ceil(d / max_step_rad - 1e-9).astype(int))
    if int(np.sum(n)) + 1 > max_states:
        raise ValueError(f"path needs {int(np.sum(n)) + 1} states at {math.degrees(max_step_rad):.2f} deg spacing "
                         f"(limit {max_states})")
    out = [Q[0:1]]
    for a, b, k in zip(Q[:-1], Q[1:], n):
        f = (np.arange(1, k + 1) / k)[:, None]
        seg = a[None, :] + (b - a)[None, :] * f
        seg[-1] = b
        out.append(seg)
    return np.vstack(out)


def _resample_arc(Q: np.ndarray, step_rad: float, max_states: int) -> np.ndarray:
    """Uniform arc-length (Euclidean, joint space) resampling of the polyline Q, step <= step_rad, ends exact."""
    seg = np.linalg.norm(np.diff(Q, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    L = float(arc[-1])
    if L <= 0.0:
        raise ValueError("zero-length path")
    n = max(1, int(math.ceil(L / step_rad - 1e-9)))
    if n + 1 > max_states:
        raise ValueError(f"path needs {n + 1} states at {step_rad:g} rad spacing (limit {max_states})")
    u = np.linspace(0.0, L, n + 1)
    R = np.column_stack([np.interp(u, arc, Q[:, j]) for j in range(Q.shape[1])])
    R[0], R[-1] = Q[0], Q[-1]
    return R


def smooth_path(Q: np.ndarray, step_rad: float = MOVE_SMOOTH_STEP_RAD, window_rad: float = MOVE_SMOOTH_WINDOW_RAD,
                max_states: int = MOVE_MAX_DENSE_STATES) -> np.ndarray:
    """Round the corners of a joint polyline Q (N, n) and resample it at uniform arc length (Euclidean in joint
    space, step <= step_rad): resample, moving average over window_rad of path length with the ends pinned (odd
    reflection at both ends, so a straight path is unchanged and the first / last points stay exact), resample again
    (the average bunches the samples up at a turn).

    Why: the path is re-timed with ONE rate profile, so a velocity-direction jump at a polyline vertex would be an
    instantaneous velocity step (acceleration spike over one 50 ms sample) and force the whole move to crawl.  With
    a 0.2 rad window a sharp corner (measured on random 6-D polylines with 90-140 deg turns) moves the path by about
    0.02 rad (1.2 deg) and costs at most ~3x in duration (plan_move_timing lengthens until the caps hold).  The
    rounded path leaves MoveIt's, so the caller MUST validity-check THIS path."""
    Q = np.asarray(Q, dtype=float)
    if Q.ndim != 2 or len(Q) < 2:
        raise ValueError("a path needs at least two points")
    if not np.all(np.isfinite(Q)):
        raise ValueError("path contains non-finite values")
    R = _resample_arc(Q, step_rad, max_states)
    n = len(R) - 1
    L = float(np.sum(np.linalg.norm(np.diff(R, axis=0), axis=1)))
    m = min(int(round(0.5 * window_rad / (L / n))), n // 2)
    if m < 1:
        return R
    left = 2.0 * R[0][None, :] - R[np.arange(m, 0, -1)]
    right = 2.0 * R[-1][None, :] - R[n - np.arange(1, m + 1)]
    ext = np.vstack([left, R, right])
    csum = np.vstack([np.zeros((1, ext.shape[1])), np.cumsum(ext, axis=0)])
    out = (csum[2 * m + 1:] - csum[:-(2 * m + 1)]) / (2 * m + 1)
    out[0], out[-1] = Q[0], Q[-1]
    return _resample_arc(out, step_rad, max_states)


def max_jump(Q: np.ndarray) -> float:
    """Largest absolute joint difference between consecutive points of Q."""
    Q = np.asarray(Q, dtype=float)
    return float(np.max(np.abs(np.diff(Q, axis=0)))) if len(Q) > 1 else 0.0


def smooth_trapezoid(tau: np.ndarray, rho: float, T: float) -> Tuple[np.ndarray, np.ndarray]:
    """Path parameter s(tau) in [0, 1] and its rate ds/dt [1/s] for a velocity profile that ramps up over the first
    `rho` of the duration T with a smoothstep (zero acceleration at both ends of the ramp), cruises, and mirrors the
    ramp at the end.  Cruise rate V = 1 / (T (1 - rho)); peak ds/dt = V, peak d2s/dt2 = 1.5 V / (rho T).
    rho = 0.5 has no cruise."""
    tau = np.clip(np.asarray(tau, dtype=float), 0.0, 1.0)
    V = 1.0 / (T * (1.0 - rho))

    def up(t):          # distance covered in the up ramp at fraction t (0..rho) of T, and the rate
        x = np.clip(t / rho, 0.0, 1.0)
        return V * T * rho * (x ** 3 - x ** 4 / 2.0), V * (3 * x ** 2 - 2 * x ** 3)

    s = np.empty_like(tau)
    sd = np.empty_like(tau)
    a = tau < rho
    c = (tau >= rho) & (tau <= 1.0 - rho)
    d = tau > 1.0 - rho
    s[a], sd[a] = up(tau[a])
    s[c] = V * T * rho / 2.0 + V * T * (tau[c] - rho)
    sd[c] = V
    sdn, sdd = up(1.0 - tau[d])
    s[d], sd[d] = 1.0 - sdn, sdd
    return np.clip(s, 0.0, 1.0), sd


def retime_path(Q: np.ndarray, tcp_pos: np.ndarray,
                max_lin_m_s: float = MOVE_MAX_TCP_SPEED_MM_S * 1e-3,
                max_joint_rad_s: float = MOVE_MAX_JOINT_SPEED_RAD_S,
                max_joint_acc: float = MOVE_MAX_JOINT_ACC_RAD_S2,
                min_duration: float = MIN_DURATION_S, dt: float = SAMPLE_DT_S,
                margin: float = DURATION_MARGIN) -> Retimed:
    """Time-scale an arbitrary joint polyline Q (N, n) with TCP positions tcp_pos (N, 3): sibling of retime() for
    paths of any length.  Same idea (the path parameter s is the cumulative share of the time each segment needs at
    exactly the speed caps, so every segment runs at most R*ds/dt of its caps) but with a smooth-trapezoid rate
    profile (smooth_trapezoid) instead of one quintic, so a long path cruises near the cap instead of at 53 % of
    its peak.  First-order bounds, with R the total time needed at the caps and g = max |dq/ds|:
        joint / TCP speed  <= R / (T (1 - rho)) / margin of the caps            ->  T >= margin R / (1 - rho)
        joint acceleration ~  g * 1.5 / ((1 - rho) rho T^2)                     ->  T^2 >= margin 1.5 g / (a rho (1 - rho))
    rho is chosen to minimise T.  These ignore path curvature, so the caller must verify the result with
    analyze_trajectory() (plan_move_timing() does and lengthens until the cubic Hermite interpolation complies)."""
    Q = np.asarray(Q, dtype=float)
    tcp_pos = np.asarray(tcp_pos, dtype=float)
    if Q.ndim != 2 or len(Q) < 2 or len(tcp_pos) != len(Q):
        raise ValueError("retime_path needs at least two waypoints with matching TCP positions")
    keep = [0]
    for i in range(1, len(Q)):
        if np.max(np.abs(Q[i] - Q[keep[-1]])) > 0.0 or i == len(Q) - 1 and len(keep) == 1:
            keep.append(i)
    Q, tcp_pos = Q[keep], tcp_pos[keep]
    K = len(Q) - 1
    dq = np.diff(Q, axis=0)
    lin = np.linalg.norm(np.diff(tcp_pos, axis=0), axis=1)
    # The joint cap is applied to the EUCLIDEAN joint speed |dq/dt|_2 (>= every single joint's speed, so each joint
    # stays under the cap; conservative by up to sqrt(6) for a move of all six joints).  The max-norm would be tighter
    # but is not smooth where the dominant joint changes, which turns into acceleration spikes.
    need = np.maximum(lin / max_lin_m_s, np.linalg.norm(dq, axis=1) / max_joint_rad_s)
    R = float(np.sum(need))
    if R <= 0.0:
        raise ValueError("zero-length move")
    ok = need > 0
    if not np.all(ok):                       # two points with identical joints but a (noisy) TCP move cannot happen; stay safe
        need = np.where(ok, need, 1e-12)
        R = float(np.sum(need))
    g = float(np.max(np.max(np.abs(dq), axis=1) * R / need))        # max |dq_j/ds| over segments and joints
    best = None
    for rho in MOVE_RAMP_FRACTIONS:
        rho = float(rho)
        T = max(min_duration, margin * R / (1.0 - rho),
                math.sqrt(margin * 1.5 * g / (max_joint_acc * rho * (1.0 - rho))))
        if best is None or T < best[0] - 1e-9:
            best = (T, rho)
    T, rho = best
    s_grid = np.concatenate([[0.0], np.cumsum(need) / R])
    s_grid[-1] = 1.0
    ds = np.diff(s_grid)
    n = max(2, int(math.ceil(T / dt)) + 1)
    times = np.linspace(0.0, T, n)
    s, sdot = smooth_trapezoid(times / T, rho, T)
    s[0], s[-1] = 0.0, 1.0
    k = np.clip(np.searchsorted(s_grid, s, side="right") - 1, 0, K - 1)
    frac = (s - s_grid[k]) / ds[k]
    q = Q[k] + frac[:, None] * dq[k]
    qd = dq[k] / ds[k][:, None] * sdot[:, None]
    q[0], q[-1] = Q[0], Q[-1]
    qd[0] = 0.0
    qd[-1] = 0.0
    return Retimed(times, q, qd, float(T))


def move_cap_ratios(an: dict) -> Dict[str, float]:
    return {"joint speed": an["joint_rad_s"] / MOVE_MAX_JOINT_SPEED_RAD_S,
            "joint acceleration": an["joint_acc_rad_s2"] / MOVE_MAX_JOINT_ACC_RAD_S2,
            "TCP speed": an["tcp_mm_s"] / MOVE_MAX_TCP_SPEED_MM_S}


def move_cap_violations(an: dict) -> List[str]:
    """Which move caps the dense cubic Hermite trajectory exceeds (empty = within all of them)."""
    caps = {"joint speed": ("joint_rad_s", MOVE_MAX_JOINT_SPEED_RAD_S, "rad/s"),
            "joint acceleration": ("joint_acc_rad_s2", MOVE_MAX_JOINT_ACC_RAD_S2, "rad/s^2"),
            "TCP speed": ("tcp_mm_s", MOVE_MAX_TCP_SPEED_MM_S, "mm/s")}
    return [f"{name} {an[key]:.4g} > {cap:g} {unit}" for name, (key, cap, unit) in caps.items() if not an[key] <= cap]


def plan_move_timing(Q: np.ndarray, tcp_pos: np.ndarray,
                     fk: Callable[[np.ndarray], np.ndarray]) -> Tuple[Retimed, dict]:
    """retime_path() until the cubic Hermite interpolation of the points stays inside every move cap; the duration
    is lengthened by the worst overshoot ratio.  The caller must still check move_cap_violations()."""
    min_duration = MIN_DURATION_S
    for _ in range(6):
        rt = retime_path(Q, tcp_pos, min_duration=min_duration)
        an = analyze_trajectory(rt.times, rt.q, rt.qd, fk)
        worst = max(move_cap_ratios(an).values())
        if worst <= 1.0 or not math.isfinite(worst):    # a non-finite ratio is reported by move_cap_violations()
            break
        min_duration = rt.duration * worst * 1.02
    return rt, an


# ---------------------------------------------------------------------------------------------------
# URDF joint limits

def parse_joint_limits(urdf_text: str) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    """joint name -> (lower, upper) in rad for revolute joints; (None, None) for continuous / unlimited."""
    data = urdf_text.encode("utf-8") if isinstance(urdf_text, str) else bytes(urdf_text)
    root = ET.fromstring(data)
    out: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    for j in root.findall("joint"):
        typ = j.get("type", "")
        lim = j.find("limit")
        if typ in ("revolute", "prismatic") and lim is not None and lim.get("lower") is not None:
            out[j.get("name")] = (float(lim.get("lower")), float(lim.get("upper")))
        elif typ in ("revolute", "continuous", "prismatic"):
            out[j.get("name")] = (None, None)
    return out


# ---------------------------------------------------------------------------------------------------
# real-hardware enable (state machine, injectable clock)

class RealLock:
    """Real hardware is locked unless the page that called enable() steps within ``timeout_s`` of the last
    accepted jog (or of the enable).  A step or enable from another page clears the enable and is refused
    (page reload = new page_id = locked again)."""

    def __init__(self, timeout_s: float = 600.0, clock: Callable[[], float] = time.monotonic):
        self.timeout_s = float(timeout_s)
        self.clock = clock
        self._page: Optional[str] = None
        self._expires = 0.0

    def _expire(self) -> None:
        if self._page is not None and self.clock() >= self._expires:
            self._page = None

    def enable(self, page_id: str) -> Tuple[bool, str]:
        self._expire()
        if self._page is not None and self._page != page_id:
            self._page = None
            return False, "real-hardware enable was held by another page; it is cleared, enable again from this page"
        self._page = page_id
        self._expires = self.clock() + self.timeout_s
        return True, f"real-hardware jog enabled for {self.timeout_s / 60.0:.0f} min after the last step"

    def disable(self) -> None:
        self._page = None

    def clear_if_other(self, page_id: str) -> bool:
        self._expire()
        if self._page is not None and self._page != page_id:
            self._page = None
            return True
        return False

    def check(self, page_id: str) -> Tuple[bool, str]:
        """May this page step a REAL arm now?  (Does not extend the expiry; call touch() when accepted.)"""
        self._expire()
        if self._page is None:
            return False, "real hardware is locked: press 'enable real jog' first (the enable expires after 10 min)"
        if self._page != page_id:
            self._page = None
            return False, "real-hardware enable belonged to another page and is cleared; enable again"
        return True, ""

    def touch(self) -> None:
        if self._page is not None:
            self._expires = self.clock() + self.timeout_s

    def state(self) -> Tuple[bool, Optional[str], Optional[float]]:
        self._expire()
        if self._page is None:
            return False, None, None
        return True, self._page, max(0.0, self._expires - self.clock())


# ---------------------------------------------------------------------------------------------------
# core: caches, pre-step refusals, busy flag, status (no ROS)

class JogCore:
    def __init__(self, mode: str, real_enable_timeout_s: float = 600.0,
                 clock: Callable[[], float] = time.monotonic):
        self.mode = mode
        self.cfgs = arm_configs(mode)
        self.arms = list(self.cfgs)
        self.clock = clock
        self.lock = threading.RLock()
        self.real_lock = RealLock(real_enable_timeout_s, clock)
        self._joints: Dict[str, Tuple[float, float, float]] = {}   # name -> (position, velocity, t_received)
        self._hw: Dict[str, Tuple[str, str]] = {}                   # component -> (class_type, state label)
        self._hw_t: Optional[float] = None
        self._ctrl: Dict[str, str] = {}                             # controller -> state
        self._ctrl_t: Optional[float] = None
        self._counter = 0
        self._busy_arm: Optional[str] = None
        self._active_id: Optional[int] = None
        self._last: Optional[dict] = None
        self.cancel = threading.Event()
        self.stop_deadline_s = STOP_CONFIRM_DEADLINE_S
        self._hist: Dict[str, deque] = {}                           # joint -> deque[(t, position)], last ~3 s
        self._last_finish_t: Optional[float] = None
        # exclusive ownership: after a cancel / failure the arm stays busy ("stopping") until the goal result AND
        # a measured standstill are both seen; otherwise "fault" (needs clear_fault() + standstill)
        self._stop_state: Optional[str] = None                      # None | "stopping" | "fault"
        self._stop_since = 0.0
        self._result_seen = False
        self._final: Optional[Tuple[bool, str, str, dict]] = None
        # joint-goal moves: at most one stored plan; _active_kind is None for steps, "move_plan" while planning and
        # "move" while a stored plan executes (it is added to `last` as "kind")
        self._plan: Optional[dict] = None
        self._plan_counter = 0
        self._active_kind: Optional[str] = None
        # STOP epoch: bumped by every request_stop() and every fault entry.  A motion request remembers the epoch it
        # was ingressed under and begin_* refuses it if the epoch changed ("STOP was pressed after this request was sent").
        self.stop_epoch = 0
        # HTTP-ingress reservation: at most ONE motion request may be queued behind the single jog worker
        self._queued: Optional[Tuple[int, float]] = None           # (token, t_reserved)
        self._queue_counter = 0
        # goal life cycle of the active motion: None (never sent) | "pending" (send_goal_async, no answer yet) |
        # "live" (accepted, no terminal result) | "terminal" (result received or goal rejected)
        self._goal_sid: Optional[int] = None
        self._goal_state: Optional[str] = None
        # event times for the anchored confirmations (see standstill(since=...)): the first STOP request of the active
        # motion, the moment a VALIDATED goal result / rejection was received, the entry into FAULT
        self._stop_req_t: Optional[float] = None
        self._goal_end_t: Optional[float] = None
        self._fault_t: Optional[float] = None

    # -- cache updates (called from ROS callbacks) --------------------------------------------------
    def update_joint_states(self, names: Sequence[str], positions: Sequence[float],
                            velocities: Sequence[float] = ()) -> None:
        t = self.clock()
        with self.lock:
            for i, n in enumerate(names):
                if i < len(positions):
                    # velocity is None when the message carries none; a non-finite value is kept (and makes the arm's
                    # joint state invalid, see _arm_finite) - nothing is silently replaced by 0
                    v = float(velocities[i]) if i < len(velocities) else None
                    p = float(positions[i])
                    self._joints[n] = (p, v, t)
                    h = self._hist.setdefault(n, deque())
                    h.append((t, p, v))
                    while len(h) > 2 and h[1][0] < t - 3.0:
                        h.popleft()

    def update_hardware(self, entries: Sequence[Tuple[str, str, str]], requested_at: Optional[float] = None) -> None:
        """entries: (component name, class_type, lifecycle state label).  ``requested_at`` (core clock) = when the
        controller_manager REQUEST was sent: the answer describes the state at or after that moment, never later, so
        it is the time stamp used as evidence (a slow answer to an early request must not count as recent).  An answer
        to an older request than the stored one is ignored.  None = now (tests / direct updates)."""
        t = self.clock() if requested_at is None else float(requested_at)
        with self.lock:
            if self._hw_t is not None and t < self._hw_t:
                return
            self._hw = {n: (c, s) for n, c, s in entries}
            self._hw_t = t

    def update_controllers(self, entries: Sequence[Tuple[str, str]], requested_at: Optional[float] = None) -> None:
        """As update_hardware(): stamped with the request time, older answers ignored."""
        t = self.clock() if requested_at is None else float(requested_at)
        with self.lock:
            if self._ctrl_t is not None and t < self._ctrl_t:
                return
            self._ctrl = {n: s for n, s in entries}
            self._ctrl_t = t

    # -- queries --------------------------------------------------------------------------------------
    def _arm_finite(self, arm: str) -> bool:
        """True if all six joints are present and their latest position (and velocity, if the message had one) is
        finite.  Caller holds the lock."""
        for n in self.cfgs[arm].joint_names:
            e = self._joints.get(n)
            if e is None or not math.isfinite(e[0]) or (e[1] is not None and not math.isfinite(e[1])):
                return False
        return True

    def joint_age(self, arm: str) -> Optional[float]:
        """Seconds since the oldest of the arm's six joints was last received; None if any is missing OR NON-FINITE
        (a NaN / inf position or velocity makes the arm's joint state invalid: every consumer refuses)."""
        with self.lock:
            now = self.clock()
            ages = []
            for n in self.cfgs[arm].joint_names:
                e = self._joints.get(n)
                if e is None:
                    return None
                ages.append(now - e[2])
            if not self._arm_finite(arm):
                return None
            return max(ages)

    def joint_gate(self, arm: str) -> Optional[str]:
        """Refusal text if the arm's joint_states cannot be used right now (missing, non-finite, stale), else None."""
        with self.lock:
            if any(n not in self._joints for n in self.cfgs[arm].joint_names):
                return f"joint_states are missing some of the six joints of arm '{arm}'"
            if not self._arm_finite(arm):
                return f"joint_states of arm '{arm}' contain non-finite values (NaN / inf): invalid"
            age = self.joint_age(arm)
            if age is None:
                return f"joint_states are missing some of the six joints of arm '{arm}'"
            if age > JOINT_STATE_MAX_AGE_S:
                return f"joint_states of arm '{arm}' are stale ({age:.2f} s old, limit {JOINT_STATE_MAX_AGE_S} s)"
            return None

    def standstill(self, arm: str, window: float = STANDSTILL_WINDOW_S, tol: float = STANDSTILL_TOL_RAD,
                   since: Optional[float] = None) -> bool:
        """True if all six joints of the arm have been received within the last JOINT_STATE_MAX_AGE_S, every position / velocity in
        the window is finite, no joint moved more than `tol` rad (max-min) during the last `window` s, none drifted
        faster than STANDSTILL_MAX_DRIFT_RAD_S (|last - first| / window) and the latest velocity (if there is one) is
        within STANDSTILL_MAX_VEL_RAD_S.  Needs history reaching back `window`.
        `since` (a clock() time) EVENT-ANCHORS the verdict: only samples received at or after it count (older samples
        are ignored altogether, including the sample just before the window) and the window must be fully covered by
        them, i.e. the answer is False until `since + window` has passed with fresh samples all the way.  Without
        `since` the arm must merely be still (the pre-dispatch gate)."""
        with self.lock:
            now = self.clock()
            for n in self.cfgs[arm].joint_names:
                h = self._hist.get(n)
                if since is not None and h:
                    h = [smp for smp in h if smp[0] >= since]
                if not h or now - h[-1][0] > JOINT_STATE_MAX_AGE_S or h[0][0] > now - window:
                    return False
                inwin = [(t, pos, vel) for (t, pos, vel) in h if t >= now - window]
                anchor = [(t, pos, vel) for (t, pos, vel) in h if t <= now - window][-1:]
                seq = anchor + inwin
                # the window must be COVERED by samples: a gap longer than JOINT_STATE_MAX_AGE_S (no joint_states at all
                # for a while, then a sample or two) proves nothing about the time in between
                if len(seq) > 1 and max(b[0] - a[0] for a, b in zip(seq, seq[1:])) > JOINT_STATE_MAX_AGE_S:
                    return False
                allv = np.array([pos for (_, pos, _) in seq], dtype=float)
                if not np.all(np.isfinite(allv)):
                    return False
                if float(np.max(allv) - np.min(allv)) >= tol:
                    return False
                span = max(window, float(seq[-1][0] - seq[0][0]))          # the anchor sample is a little older than the window
                if abs(float(allv[-1] - allv[0])) / span > STANDSTILL_MAX_DRIFT_RAD_S:
                    return False
                vel = h[-1][2]
                if vel is not None and not (math.isfinite(vel) and abs(vel) <= STANDSTILL_MAX_VEL_RAD_S):
                    return False
            return True

    def feedback_frozen(self, arm: str, window_s: float = FREEZE_MAX_S) -> bool:
        """True if the history of the arm reaches back `window_s` (from its newest sample) and NONE of its six joints
        reported a different position during that time (a stationary arm reports the same position too: the caller
        decides whether the arm was commanded to move)."""
        with self.lock:
            for n in self.cfgs[arm].joint_names:
                h = self._hist.get(n)
                if not h:
                    return False
                t1, p1, _ = h[-1]
                older = [t for (t, _, _) in h if t <= t1 - window_s]
                if not older:
                    return False
                t0 = older[-1]
                if any(pos != p1 for (t, pos, _) in h if t >= t0):      # NaN != NaN: never "frozen"
                    return False
            return True

    def confirmation_problem(self, arm: str, since: float) -> Optional[str]:
        """None if a stop / settle / fault clear may be CONFIRMED for the event at `since`, else why not: the hardware
        component and the controller must be active (a controller_manager answer to a request SENT at least
        DRIVER_STATUS_CUTOFF_S after the event - a feedback freeze that spans the whole anchored window has made the
        driver latch ERROR by then and that answer shows it), joint_states fresh, and the arm measured at standstill
        over a window made of samples received after the event only."""
        with self.lock:
            ok, why = self.hardware_ok(arm)
            if not ok:
                return f"cannot confirm the stop: {why}"
            why = self.joint_gate(arm)
            if why:
                return f"cannot confirm the stop: {why}"
            if not self.standstill(arm, since=since):
                return (f"arm '{arm}' is not measured at standstill since the stop / result (all joints within "
                        f"{STANDSTILL_TOL_RAD} rad over {STANDSTILL_WINDOW_S} s of fresh samples and drifting less than "
                        f"{STANDSTILL_MAX_DRIFT_RAD_S} rad/s)")
            floor = since + DRIVER_STATUS_CUTOFF_S
            if self._hw_t is None or self._ctrl_t is None or self._hw_t < floor or self._ctrl_t < floor:
                return ("cannot confirm the stop yet: waiting for a controller_manager answer that is newer than "
                        f"{DRIVER_STATUS_CUTOFF_S:g} s after the stop / result")
            return None

    def measured_speed(self, arm: str, window_s: float = SUPERVISION_SPEED_WINDOW_S) -> Optional[float]:
        """Largest measured joint speed [rad/s] of the arm as a finite difference between the newest sample and the
        newest sample at least `window_s` older.  None if the history does not reach back that far (or a joint is
        missing); inf if a position involved is non-finite."""
        with self.lock:
            worst = 0.0
            for n in self.cfgs[arm].joint_names:
                h = self._hist.get(n)
                if not h:
                    return None
                t1, p1, _ = h[-1]
                older = [(t, p) for (t, p, _) in h if t <= t1 - window_s]
                if not older:
                    return None
                t0, p0 = older[-1]
                if not (math.isfinite(p0) and math.isfinite(p1)):
                    return float("inf")
                worst = max(worst, abs(p1 - p0) / (t1 - t0))
            return worst

    def joint_window(self, arms: Sequence[str], window_s: float = STANDSTILL_WINDOW_S):
        """(latest position per joint name, largest max-min of any joint over the last `window_s` s, history reaches back
        that far) for all joints of `arms`; None if a joint is missing, non-finite (latest value or anywhere in the
        window) or was last received > JOINT_STATE_MAX_AGE_S ago.
        Used by the UI auto capture (it applies its own stationary tolerance)."""
        with self.lock:
            now = self.clock()
            q, spread, covered = {}, 0.0, True
            for arm in arms:
                if not self._arm_finite(arm):
                    return None
                for n in self.cfgs[arm].joint_names:
                    e, h = self._joints.get(n), self._hist.get(n)
                    if e is None or not h or now - e[2] > JOINT_STATE_MAX_AGE_S:
                        return None
                    q[n] = e[0]
                    covered = covered and h[0][0] <= now - window_s
                    vals = [pos for (t, pos, _) in h if t >= now - window_s] + \
                        [pos for (t, pos, _) in h if t <= now - window_s][-1:]
                    if not np.all(np.isfinite(vals)):
                        return None
                    spread = max(spread, max(vals) - min(vals))
            return q, spread, covered

    def joint_snapshot(self, arm: str) -> Optional[np.ndarray]:
        """The arm's six latest positions; None if one is missing or non-finite."""
        with self.lock:
            try:
                q = np.array([self._joints[n][0] for n in self.cfgs[arm].joint_names], dtype=float)
            except KeyError:
                return None
            return q if self._arm_finite(arm) else None

    def all_joint_positions(self) -> Dict[str, float]:
        with self.lock:
            return {n: e[0] for n, e in self._joints.items()}

    def _cache_fresh(self, t: Optional[float]) -> bool:
        return t is not None and self.clock() - t <= CM_CACHE_MAX_AGE_S

    def is_real(self, arm: str) -> bool:
        """True unless controller_manager reported the arm's component with class_type mock_components/GenericSystem."""
        with self.lock:
            e = self._hw.get(self.cfgs[arm].hardware)
            return not (e is not None and e[0] == MOCK_PLUGIN)

    def hardware_ok(self, arm: str) -> Tuple[bool, str]:
        cfg = self.cfgs[arm]
        with self.lock:
            if not self._cache_fresh(self._hw_t) or not self._cache_fresh(self._ctrl_t):
                return False, "controller_manager state unknown (no fresh answer from list_hardware_components / list_controllers)"
            e = self._hw.get(cfg.hardware)
            if e is None:
                return False, f"hardware component '{cfg.hardware}' not found in controller_manager"
            if e[1] != "active":
                return False, f"hardware component '{cfg.hardware}' is '{e[1]}', not active"
            c = self._ctrl.get(cfg.controller)
            if c is None:
                return False, f"controller '{cfg.controller}' not found in controller_manager"
            if c != "active":
                return False, f"controller '{cfg.controller}' is '{c}', not active"
        return True, ""

    def status(self) -> dict:
        with self.lock:
            hardware, ages = {}, {}
            hw_fresh = self._cache_fresh(self._hw_t)
            ctrl_fresh = self._cache_fresh(self._ctrl_t)
            for arm, cfg in self.cfgs.items():
                e = self._hw.get(cfg.hardware) if hw_fresh else None
                hardware[arm] = {
                    "real": self.is_real(arm) if hw_fresh else True,
                    "component": cfg.hardware,
                    "state": e[1] if e else "unknown",
                    "controller": cfg.controller,
                    "controller_state": (self._ctrl.get(cfg.controller, "unknown") if ctrl_fresh else "unknown"),
                }
                ages[arm] = self.joint_age(arm)
            enabled, page, remaining = self.real_lock.state()
            return {
                "mode": self.mode, "arms": list(self.arms), "hardware": hardware, "joint_states_age_s": ages,
                "real_enabled": enabled, "real_enabled_page": page, "real_enabled_remaining_s": remaining,
                "busy": self._busy_arm is not None, "busy_arm": self._busy_arm,
                "stopping": self._stop_state == "stopping", "fault": self._stop_state == "fault",
                "last": dict(self._last) if self._last else None,
                "move_plan": self.plan_summary(),
                "limits": {"steps": STEP_SIZES, "max_cart_speed_mm_s": MAX_CART_SPEED_MM_S,
                           "max_rot_speed_deg_s": MAX_ROT_SPEED_DEG_S, "max_joint_speed_rad_s": MAX_JOINT_SPEED_RAD_S,
                           "move": {"max_joint_speed_rad_s": MOVE_MAX_JOINT_SPEED_RAD_S,
                                    "max_joint_acc_rad_s2": MOVE_MAX_JOINT_ACC_RAD_S2,
                                    "max_tcp_speed_mm_s": MOVE_MAX_TCP_SPEED_MM_S,
                                    "plan_max_age_s": MOVE_PLAN_MAX_AGE_S}},
            }

    # -- real enable ----------------------------------------------------------------------------------
    def enable_real(self, page_id: str) -> dict:
        if not isinstance(page_id, str) or not page_id:
            return {"ok": False, "message": "page_id is required"}
        with self.lock:
            ok, msg = self.real_lock.enable(page_id)
        return {"ok": ok, "message": msg}

    def disable(self, reason: str = "") -> None:
        with self.lock:
            self.real_lock.disable()

    # -- step life cycle --------------------------------------------------------------------------------
    def _refuse(self, sid: int, msg: str, record: bool = True) -> Tuple[bool, int, str, None]:
        if record:
            self._last = {"id": sid, "ok": False, "phase": "refused", "message": msg}
        return False, sid, msg, None

    def begin_step(self, page_id, arm, kind, axis, sign, size, frame="world", epoch=None):
        """All synchronous checks.  Returns (ok, id, message, ctx); ctx (on ok) carries the request and the start
        joint snapshot and the arm is marked busy until finish().  `epoch` (from reserve_motion) voids the request if
        STOP was pressed after it was ingressed."""
        return self._begin_cartesian(page_id, epoch, lambda: validate_request(self.arms, arm, kind, axis, sign, size, frame))

    def begin_descend(self, page_id, arm, T_base_target, epoch=None):
        """Same gating as begin_step() (page enable on REAL hardware, healthy hardware, fresh joint_states, standstill,
        busy / stopping / fault, STOP epoch) for a descend; ``allow_real_move`` is NOT consulted (see the module doc).
        ctx carries the target pose in the arm's base frame; the length / orientation limits are checked in the worker
        against the measured TCP."""
        return self._begin_cartesian(page_id, epoch, lambda: validate_descend_target(self.arms, arm, T_base_target),
                                     active_kind="descend")

    def _begin_cartesian(self, page_id, epoch, validate, active_kind=None):
        with self.lock:
            self._counter += 1
            sid = self._counter
            if not isinstance(page_id, str) or not page_id:
                return self._refuse(sid, "page_id is required", record=self._busy_arm is None)
            why = self._epoch_refusal(epoch)
            if why:
                return self._refuse(sid, why, record=self._busy_arm is None)
            cleared = self.real_lock.clear_if_other(page_id)
            busy = self._busy_arm is not None
            req, why = validate()
            if req is None:
                return self._refuse(sid, why, record=not busy)
            arm = req["arm"]
            if busy:
                if self._stop_state == "fault":
                    why = ("fault: the last stop was not confirmed - check the arm, use the e-stop; "
                           "clear_fault() is required before any new step")
                elif self._stop_state == "stopping":
                    why = f"busy: arm '{self._busy_arm}' is stopping (waiting for the controller result and standstill)"
                else:
                    why = f"busy: a step on arm '{self._busy_arm}' is still running"
                return self._refuse(sid, why, record=False)
            if self._last_finish_t is not None and self.clock() - self._last_finish_t < MIN_STEP_GAP_S:
                return self._refuse(sid, "step too soon after the previous one", record=False)
            why = self.joint_gate(arm)
            if why:
                return self._refuse(sid, why)
            q0 = self.joint_snapshot(arm)
            if q0 is None or not np.all(np.isfinite(q0)):
                return self._refuse(sid, f"joint_states of arm '{arm}' are not finite")
            ok, why = self.hardware_ok(arm)
            if not ok:
                return self._refuse(sid, why)
            real = self.is_real(arm)
            if real:
                ok, why = self.real_lock.check(page_id)
                if not ok:
                    if cleared:
                        why = "real-hardware enable belonged to another page and is cleared; enable again from this page"
                    return self._refuse(sid, why)
            if not self.standstill(arm):
                return self._refuse(sid, STANDSTILL_REFUSAL)
            if real:
                self.real_lock.touch()
            self._busy_arm = arm
            self._active_id = sid
            self._stop_req_t = None
            self._active_kind = active_kind
            self.cancel.clear()
            self._last = dict({"id": sid, "ok": True, "phase": "planning", "message": "planning step"},
                              **({"kind": active_kind} if active_kind else {}))
            ctx = dict(req, id=sid, q0=q0, real=real, page_id=page_id, cfg=self.cfgs[arm])
            return True, sid, "accepted", ctx

    def _kind_extra(self, sid: Optional[int]) -> dict:
        return {"kind": self._active_kind} if self._active_kind is not None and self._active_id == sid else {}

    def set_phase(self, sid: int, phase: str, message: str) -> None:
        with self.lock:
            if self._active_id == sid and self._stop_state is None:
                self._last = dict({"id": sid, "ok": True, "phase": phase, "message": message}, **self._kind_extra(sid))

    def _release(self, sid: int, ok: bool, phase: str, message: str, extra: Optional[dict]) -> None:
        self._last = dict({"id": sid, "ok": bool(ok), "phase": phase, "message": message}, **(extra or {}),
                          **self._kind_extra(sid))
        if self._active_id == sid:
            self._busy_arm = None
            self._active_id = None
            self._active_kind = None
            self._stop_state = None
            self._final = None
            self._goal_sid, self._goal_state = None, None
            self._stop_req_t = self._goal_end_t = self._fault_t = None
            if phase not in ("refused", "planned"):      # a stored plan must be executable right away
                self._last_finish_t = self.clock()

    def finish(self, sid: int, ok: bool, phase: str, message: str, extra: Optional[dict] = None) -> None:
        """Release the arm.  Only for outcomes where no trajectory goal can be live (refusals before dispatch,
        a confirmed finish); a cancel or failure after dispatch goes through begin_stopping()."""
        with self.lock:
            self._release(sid, ok, phase, message, extra)

    def in_fault(self) -> bool:
        with self.lock:
            return self._stop_state == "fault"

    def request_stop(self) -> bool:
        """Set the cancel flag; True if a step was active (planning, executing, stopping or fault).  A stored move
        plan is dropped too: pressing STOP means nothing may start from an earlier preview.  STOP, cancel, fault
        handling and the supervision never consult the real-hardware enable (RealLock): an expired or revoked enable
        stops nothing that is already moving, and it never blocks a STOP.  Takes the core lock, i.e. it is ordered
        against dispatch_goal(): the goal send and the registration of the pending send happen inside one hold."""
        with self.lock:
            self.stop_epoch += 1           # every request ingressed before this moment is void (begin_* checks it)
            self._plan = None
            active = self._busy_arm is not None
            if active:
                if self._stop_req_t is None:
                    self._stop_req_t = self.clock()
                self.cancel.set()
            return active

    # -- HTTP ingress: one queued motion request at most, void after a STOP ---------------------------------
    def reserve_motion(self) -> dict:
        """Called at HTTP ingress BEFORE a plan / step / move request is queued behind the single jog worker.  Refuses
        at once (ok False) while the jog is busy / stopping / in fault or another motion request is already queued;
        otherwise reserves the single queue slot and returns {ok, epoch, token}.  The caller passes `epoch` to
        begin_* (a STOP in between voids the request) and calls release_motion(token) when the request is done."""
        with self.lock:
            now = self.clock()
            if self._busy_arm is not None:
                return {"ok": False, "message": self._busy_message()}
            if self._queued is not None and now - self._queued[1] <= MOTION_QUEUE_TTL_S:
                return {"ok": False, "message": "busy: another motion request is already queued"}
            self._queue_counter += 1
            self._queued = (self._queue_counter, now)
            return {"ok": True, "message": "reserved", "epoch": self.stop_epoch, "token": self._queue_counter}

    def release_motion(self, token) -> None:
        with self.lock:
            if self._queued is not None and self._queued[0] == token:
                self._queued = None

    def _epoch_refusal(self, epoch) -> Optional[str]:
        if epoch is not None and epoch != self.stop_epoch:
            return "STOP was pressed after this request was sent: request void"
        return None

    # -- goal life cycle (steps and moves): clear_fault needs a terminal state ---------------------------------
    def note_goal(self, sid: int, state: str) -> None:
        """state: "pending" (goal sent, no answer), "live" (accepted), "terminal" (a VALIDATED result was received -
        validated_result() - or the goal was rejected) or "unconfirmed" (the answer was an exception / unusable /
        carried an unknown status: whether the goal ended is unknown).  "terminal" and "unconfirmed" are final for
        that goal (whichever comes first); a later "live" / "pending" for the same goal is ignored (callbacks may
        race).  Only "terminal" (or no goal at all) lets clear_fault() release the arm."""
        with self.lock:
            if state == "pending" or self._goal_sid != sid:
                if state == "pending":
                    self._goal_sid, self._goal_state = sid, "pending"
                    self._goal_end_t = None
                return
            if self._goal_state in ("terminal", "unconfirmed"):
                return
            if state == "live" and self._goal_state == "pending":
                self._goal_state = "live"
            elif state in ("terminal", "unconfirmed"):
                self._goal_state = state
                if state == "terminal":
                    self._goal_end_t = self.clock()      # anchor of the settle / stop confirmation

    def goal_end_time(self, sid: int) -> Optional[float]:
        """clock() time at which the controller's validated result (or rejection) of the active goal `sid` was seen."""
        with self.lock:
            return self._goal_end_t if self._goal_sid == sid else None

    def _event_anchor(self, *times: Optional[float]) -> float:
        known = [t for t in times if t is not None]
        return max(known) if known else self.clock()

    def goal_confirmed_ended(self) -> bool:
        with self.lock:
            return self._goal_state in (None, "terminal")

    def goal_unconfirmed(self) -> bool:
        """True if the controller's answer for the active goal could not be confirmed (final for that goal: a UI restart
        does NOT help - it would drop the goal / fault ownership without stopping anything)."""
        with self.lock:
            return self._goal_state == "unconfirmed"

    # -- exclusive ownership until the stop is confirmed ------------------------------------------------
    def begin_stopping(self, sid: int, ok: bool, phase: str, message: str, extra: Optional[dict] = None,
                       result_seen: bool = False) -> None:
        """A goal may be live: keep the arm busy ("stopping") until the goal result is in (note_result) AND
        standstill is measured, then release it with the given final outcome (poll_stopping)."""
        with self.lock:
            if self._active_id != sid or self._stop_state is not None:
                return
            self._stop_state = "stopping"
            self._stop_since = self.clock()
            self._result_seen = bool(result_seen)
            self._final = (ok, phase, message, extra or {})
            self._last = dict({"id": sid, "ok": False, "phase": "stopping",
                               "message": f"stopping ({message}): waiting for the controller result and a measured standstill"},
                              **self._kind_extra(sid))

    def note_result(self, sid: int) -> None:
        with self.lock:
            if self._active_id == sid:
                self._result_seen = True
        self.note_goal(sid, "terminal")

    def poll_stopping(self) -> str:
        """"waiting" | "confirmed" (arm released with the final outcome) | "fault" (deadline passed) | "none"."""
        with self.lock:
            if self._stop_state is None:
                return "none"
            if self._stop_state == "fault":
                return "fault"
            sid, arm = self._active_id, self._busy_arm
            if self._goal_state == "unconfirmed":
                # the goal's end can never be confirmed any more: do not wait for the deadline
                self._enter_fault_locked(sid, GOAL_UNCONFIRMED_MSG)
                return "fault"
            if self._result_seen and self.goal_confirmed_ended():
                # event-anchored: only samples received after the stop began / the result arrived count, and the
                # hardware must be healthy (a frozen feedback must not "confirm" anything)
                anchor = self._event_anchor(self._stop_since, self._stop_req_t, self._goal_end_t)
                if self.confirmation_problem(arm, anchor) is None:
                    ok, phase, message, extra = self._final
                    self._release(sid, ok, phase, message, extra)
                    return "confirmed"
            if self.clock() - self._stop_since > self.stop_deadline_s:
                self._enter_fault_locked(sid, "stop not confirmed - check the arm, use the e-stop")
                return "fault"
            return "waiting"

    def _enter_fault_locked(self, sid: Optional[int], message: str) -> None:
        self._stop_state = "fault"
        self._fault_t = self.clock()
        self.stop_epoch += 1               # a fault voids every queued request as well
        self._plan = None                  # the arm state is uncertain: no stored plan survives a fault
        self.real_lock.disable()           # a fault always re-locks real hardware
        self._last = dict({"id": sid, "ok": False, "phase": "fault", "message": message}, **self._kind_extra(sid))

    def enter_fault(self, sid: int, message: str) -> None:
        with self.lock:
            if self._active_id == sid:
                self._enter_fault_locked(sid, message)

    def clear_fault(self) -> dict:
        """Release the arm after a fault.  Only allowed when the controller confirmed the goal ended (result received,
        goal rejected, or never sent) AND the arm is measured to stand still."""
        with self.lock:
            if self._stop_state != "fault":
                return {"ok": False, "message": "no fault to clear" if self._stop_state is None
                        else "the arm is still stopping (no fault yet)"}
            arm = self._busy_arm
            if self._goal_state == "unconfirmed":
                return {"ok": False, "message": GOAL_UNCONFIRMED_MSG}
            if not self.goal_confirmed_ended():
                # a late goal acceptance could still arrive after the clear and start moving the released arm
                return {"ok": False, "message": "the controller has not confirmed the goal ended: wait, or use the "
                                                "e-stop and restart the controller"}
            # event-anchored: the window must consist of samples received after the fault began / the result arrived,
            # and the hardware component + controller must be active (a frozen feedback proves nothing)
            anchor = self._event_anchor(self._fault_t, self._goal_end_t, self._stop_req_t)
            why = self.confirmation_problem(arm, anchor)
            if why:
                return {"ok": False, "message": f"{why}: use the e-stop"}
            sid = self._active_id
            self._release(sid, False, "stopped", "fault cleared by the operator after standstill was measured", None)
            return {"ok": True, "message": "fault cleared; standstill was measured"}

    # -- checks right before the trajectory goal is sent ---------------------------------------------------
    def predispatch_check(self, ctx: dict) -> Tuple[bool, str]:
        """Everything that may have changed since begin_step(): stop request, real-hardware enable (same page, not
        expired), hardware classification mock/real, hardware + controller active, joint_states fresh and the arm
        unmoved (START_MOVED_TOL_RAD)."""
        arm = ctx["arm"]
        with self.lock:
            if self.cancel.is_set():
                return False, "stop requested before the trajectory was sent"
            if self._stop_state is not None or self._active_id != ctx["id"]:
                return False, "step is no longer the active one"
            ok, why = self.hardware_ok(arm)
            if not ok:
                return False, why
            if self.is_real(arm) != ctx["real"]:
                return False, "hardware classification (mock / real) changed since the step was accepted"
            if ctx["real"]:
                ok, why = self.real_lock.check(ctx["page_id"])
                if not ok:
                    return False, f"real-hardware enable no longer valid: {why}"
            age = self.joint_age(arm)
            q = self.joint_snapshot(arm)
            if age is None or q is None or age > JOINT_STATE_MAX_AGE_S:
                return False, "joint_states went stale (or missing / non-finite) before the trajectory was sent"
            moved = float(np.max(np.abs(q - ctx["q0"])))
            if not moved <= START_MOVED_TOL_RAD:           # NaN never passes
                return False, (f"the arm moved {moved:.5f} rad since the step was accepted "
                               f"(limit {START_MOVED_TOL_RAD} rad)")
            if not self.standstill(arm):                   # full window, not anchored: the arm must simply be still
                return False, STANDSTILL_REFUSAL
        return True, ""

    def dispatch_goal(self, ctx: dict, send_fn: Callable[[], Any], register_fn: Callable[[int, Any], None],
                      phase_message: str) -> Tuple[bool, str, Any]:
        """The final gate and the goal send as ONE step with respect to request_stop().  Under the core lock:
        (1) predispatch_check (cancel flag, active id, hardware, enable, joints, unmoved arm), (2) phase "executing" and
        goal state "pending", (3) send_fn() (send_goal_async: it only queues the request and returns a future, it
        waits for nothing and no callback it triggers needs this lock synchronously; the executor thread that delivers
        the answer merely waits for the lock), (4) register_fn(sid, future) records the pending send so that a STOP can
        hook cancel-on-accept.  request_stop() takes the same lock, so a STOP either completes BEFORE the gate (the gate
        refuses, nothing is sent) or finds the registered pending send AFTER it (and hooks it).  Returns (ok, reason,
        send future); an exception of send_fn propagates with ctx['sent'] already True and the goal state "pending"
        (a goal may exist: the caller must fail closed)."""
        sid = ctx["id"]
        with self.lock:
            ok, why = self.predispatch_check(ctx)
            if not ok:
                return False, why, None
            self.set_phase(sid, "executing", phase_message)
            ctx["sent"] = True              # from here on a goal may be live: only a confirmed stop releases the arm
            self.note_goal(sid, "pending")
            fut = send_fn()
            register_fn(sid, fut)
            return True, "", fut

    # -- joint-goal moves: plan (preview) and execute -----------------------------------------------------
    def _busy_message(self) -> str:
        if self._stop_state == "fault":
            return ("fault: the last stop was not confirmed - check the arm, use the e-stop; "
                    "clear_fault() is required before any new step or move")
        if self._stop_state == "stopping":
            return f"busy: arm '{self._busy_arm}' is stopping (waiting for the controller result and standstill)"
        return f"busy: a step or move on arm '{self._busy_arm}' is still running"

    def plan_summary(self) -> Optional[dict]:
        """The stored plan as shown in status(): {plan_id, arm, duration_s, age_s, max_joint_change_deg} or None."""
        with self.lock:
            p = self._plan
            if p is None:
                return None
            return {"plan_id": p["plan_id"], "arm": p["arm"], "duration_s": round(p["duration_s"], 3),
                    "age_s": round(max(0.0, self.clock() - p["created"]), 3),
                    "max_joint_change_deg": round(p["max_joint_change_deg"], 4)}

    def clear_plan(self) -> bool:
        """Drop the stored move plan; True if there was one."""
        with self.lock:
            had = self._plan is not None
            self._plan = None
            return had

    def begin_move_plan(self, page_id, arm, q_goal, epoch=None):
        """All synchronous checks of plan_move().  Same gating as begin_step() (arm, busy / stopping / fault, fresh
        joint_states, hardware + controller active, real-hardware enable of THIS page; mock needs none) plus the
        goal check (exactly the arm's six joints, finite).  Returns (ok, id, message, ctx); on ok the arm is busy
        ("planning") until finish() / complete_plan() and ANY earlier stored plan is dropped (a new request always
        invalidates the old preview, whatever its outcome).  The enable is not extended: only executing counts."""
        with self.lock:
            self._counter += 1
            sid = self._counter
            busy = self._busy_arm is not None
            if not isinstance(page_id, str) or not page_id:
                return self._refuse(sid, "page_id is required", record=not busy)
            why = self._epoch_refusal(epoch)
            if why:
                return self._refuse(sid, why, record=not busy)
            cleared = self.real_lock.clear_if_other(page_id)
            if arm not in ("left", "right") or arm not in self.arms:
                return self._refuse(sid, f"unknown arm {arm!r} (active arms: {', '.join(self.arms)})", record=not busy)
            goal, why = validate_joint_goal(self.cfgs[arm].joint_names, q_goal)
            if goal is None:
                return self._refuse(sid, why, record=not busy)
            if busy:
                return self._refuse(sid, self._busy_message(), record=False)
            why = self.joint_gate(arm)
            if why:
                return self._refuse(sid, why)
            q0 = self.joint_snapshot(arm)
            if q0 is None or not np.all(np.isfinite(q0)):
                return self._refuse(sid, f"joint_states of arm '{arm}' are not finite")
            ok, why = self.hardware_ok(arm)
            if not ok:
                return self._refuse(sid, why)
            real = self.is_real(arm)
            if real:
                ok, why = self.real_lock.check(page_id)
                if not ok:
                    if cleared:
                        why = "real-hardware enable belonged to another page and is cleared; enable again from this page"
                    return self._refuse(sid, why)
            # (no standstill gate here: planning moves nothing; begin_move_execute and the pre-dispatch check require the
            # measured standstill AND that the arm is still where the plan started, so a plan made while teleop moved
            # the arm can never be executed)
            self._plan = None
            self._busy_arm = arm
            self._active_id = sid
            self._stop_req_t = None
            self._active_kind = "move_plan"
            self.cancel.clear()
            self._last = {"id": sid, "ok": True, "phase": "planning", "message": "planning move", "kind": "move_plan"}
            ctx = {"kind": "move_plan", "arm": arm, "id": sid, "q0": q0, "q_goal": goal, "real": real,
                   "page_id": page_id, "cfg": self.cfgs[arm]}
            return True, sid, "accepted", ctx

    def complete_plan(self, sid: int, plan: dict) -> Optional[str]:
        """Store a finished plan (replacing any older one) and release the arm with phase "planned".  Atomic with the
        cancel flag: returns None (nothing stored) if STOP was pressed meanwhile or the planning is no longer the
        active operation.  `plan` needs arm, page_id, real, q_start, q_goal, rt, an, duration_s, max_joint_change_deg,
        n_points; plan_id and created are set here."""
        with self.lock:
            if self._active_id != sid or self._stop_state is not None or self.cancel.is_set():
                return None
            self._plan_counter += 1
            plan = dict(plan, plan_id=f"plan-{self._plan_counter}", created=self.clock())
            self._plan = plan
            msg = (f"move planned ({plan['plan_id']}): {plan['max_joint_change_deg']:.2f} deg max joint change, "
                   f"{plan['duration_s']:.1f} s; execute within {MOVE_PLAN_MAX_AGE_S:g} s")
            self._release(sid, True, "planned", msg, {"plan_id": plan["plan_id"]})
            return plan["plan_id"]

    def begin_move_execute(self, page_id, plan_id, epoch=None):
        """All synchronous checks of execute_move(); on ok the stored plan is CONSUMED (it can never be executed
        twice, whatever happens next), the arm is busy and ctx carries everything _execute() needs.  Refused unless:
        the page matches, a plan with this id is stored, it is <= MOVE_PLAN_MAX_AGE_S old, the arm is not busy /
        stopping / in fault, joint_states are fresh, hardware + controller are active, the mock / real
        classification is unchanged since planning, on REAL hardware this page still holds the enable, and the
        joints are within START_MOVED_TOL_RAD of the plan's start.  Executing extends the real enable (activity)."""
        with self.lock:
            self._counter += 1
            sid = self._counter
            busy = self._busy_arm is not None
            if not isinstance(page_id, str) or not page_id:
                return self._refuse(sid, "page_id is required", record=not busy)
            why = self._epoch_refusal(epoch)
            if why:
                return self._refuse(sid, why, record=not busy)
            cleared = self.real_lock.clear_if_other(page_id)
            if busy:
                return self._refuse(sid, self._busy_message(), record=False)
            plan = self._plan
            if plan is None:
                return self._refuse(sid, "no move plan is stored (already executed, cancelled or replaced): plan again")
            if not isinstance(plan_id, str) or plan_id != plan["plan_id"]:
                return self._refuse(sid, f"plan id {plan_id!r} does not match the stored plan (replaced?): plan again")
            if plan["page_id"] != page_id:
                return self._refuse(sid, "the stored plan was made by another page: plan again from this page")
            age = self.clock() - plan["created"]
            if age > MOVE_PLAN_MAX_AGE_S:
                self._plan = None
                return self._refuse(sid, f"the plan is too old ({age:.0f} s, limit {MOVE_PLAN_MAX_AGE_S:g} s): plan again")
            if self._last_finish_t is not None and self.clock() - self._last_finish_t < MIN_STEP_GAP_S:
                return self._refuse(sid, "move too soon after the previous step", record=False)
            arm = plan["arm"]
            why = self.joint_gate(arm)
            if why:
                return self._refuse(sid, why)
            q = self.joint_snapshot(arm)
            if q is None or not np.all(np.isfinite(q)):
                return self._refuse(sid, f"joint_states of arm '{arm}' are not finite")
            ok, why = self.hardware_ok(arm)
            if not ok:
                return self._refuse(sid, why)
            real = self.is_real(arm)
            if real != plan["real"]:
                self._plan = None
                return self._refuse(sid, "hardware classification (mock / real) changed since planning: plan again")
            if real:
                ok, why = self.real_lock.check(page_id)
                if not ok:
                    if cleared:
                        why = "real-hardware enable belonged to another page and is cleared; enable again from this page"
                    return self._refuse(sid, why)
            moved = float(np.max(np.abs(q - plan["q_start"])))
            if not moved <= START_MOVED_TOL_RAD:           # NaN never passes
                self._plan = None
                return self._refuse(sid, f"the arm moved since planning ({moved:.5f} rad, limit {START_MOVED_TOL_RAD} rad): "
                                         f"plan again")
            if not self.standstill(arm):
                return self._refuse(sid, STANDSTILL_REFUSAL)
            if real:
                self.real_lock.touch()
            self._plan = None                                   # consumed
            self._busy_arm = arm
            self._active_id = sid
            self._stop_req_t = None
            self._active_kind = "move"
            self.cancel.clear()
            self._last = {"id": sid, "ok": True, "phase": "executing", "message": "preparing move dispatch",
                          "kind": "move"}
            ctx = {"kind": "move", "arm": arm, "id": sid, "q0": plan["q_start"].copy(), "q_goal": plan["q_goal"].copy(),
                   "real": real, "page_id": page_id, "cfg": self.cfgs[arm], "plan_id": plan["plan_id"],
                   "rt": plan["rt"], "an": plan["an"], "max_joint_change_deg": plan["max_joint_change_deg"],
                   "Qd": plan.get("Qd"), "curve": plan.get("curve"), "plan_created": plan["created"]}
            return True, sid, "accepted", ctx


# ---------------------------------------------------------------------------------------------------
# runtime supervision of an executing goal (steps and moves); pure, driven by JogController._execute

class MotionSupervisor:
    """While a goal executes: (a) the arm's joint_states must stay fresh (<= SUPERVISION_STALE_S) and finite, (b) the
    measured joint speed (finite difference over >= SUPERVISION_SPEED_WINDOW_S) must stay <= SUPERVISION_SPEED_FACTOR
    x `speed_cap`, (c) while the reference commands more than FREEZE_REF_SPEED_RAD_S some position must have changed
    within the last FREEZE_MAX_S (frozen feedback), (d) every joint must stay within `track_tol` of the reference
    trajectory.  check() returns the text
    of the first violated condition, or None.  The reference is evaluated at the time since the goal was ACCEPTED
    (before that the arm must still be at the start: the reference is held at the first point), so a late goal
    acceptance does not look like a tracking error."""

    def __init__(self, core: "JogCore", arm: str, rt: Retimed, speed_cap: float, track_tol: float):
        self.core, self.arm, self.rt = core, arm, rt
        self.speed_cap, self.track_tol = float(speed_cap), float(track_tol)

    def check(self, t_since_dispatch: float, t_since_accept: Optional[float]) -> Optional[str]:
        core, arm = self.core, self.arm
        age = core.joint_age(arm)
        if age is None:
            return "joint_states of the arm are missing or non-finite"
        if not age <= SUPERVISION_STALE_S:
            return f"joint_states are stale ({age:.2f} s old, limit {SUPERVISION_STALE_S} s)"
        limit = SUPERVISION_SPEED_FACTOR * self.speed_cap
        v = core.measured_speed(arm)
        if v is not None and not v <= limit:
            return (f"measured joint speed {v:.3f} rad/s exceeds {SUPERVISION_SPEED_FACTOR:g} x the cap "
                    f"({limit:.3f} rad/s)")
        if t_since_accept is not None and t_since_accept >= FREEZE_MAX_S:
            # FROZEN FEEDBACK: the driver repeats the last positions with fresh stamps while its STATUS stalls (coast /
            # hold), so staleness and speed look fine.  The goal is accepted, the reference commands motion, yet no
            # joint position changed for FREEZE_MAX_S: the feedback cannot be trusted -> STOP.
            vref = float(np.max(np.abs(hermite_eval_velocity(self.rt.times, self.rt.q, self.rt.qd, t_since_accept))))
            if vref > FREEZE_REF_SPEED_RAD_S and core.feedback_frozen(arm, FREEZE_MAX_S):
                return "joint feedback frozen (driver STATUS stall?)"
        q = core.joint_snapshot(arm)
        if q is None:
            return "joint_states of the arm are missing or non-finite"
        ref = hermite_eval(self.rt.times, self.rt.q, self.rt.qd, 0.0 if t_since_accept is None else t_since_accept)
        err = float(np.max(np.abs(q - ref)))
        if not err <= self.track_tol:
            return f"tracking error {err:.4f} rad exceeds {self.track_tol:g} rad"
        return None


# ---------------------------------------------------------------------------------------------------
# ROS glue

class _Refuse(Exception):
    pass


class _Stopped(Exception):
    pass


def _step_text(ctx: dict) -> str:
    if ctx["kind"] == "move":
        return f"move to joint goal (max change {ctx['max_joint_change_deg']:.2f} deg, {ctx['plan_id']})"
    if ctx["kind"] == "descend":
        return "descend along the surface normal" + (f" {ctx['size']:g} mm" if math.isfinite(ctx.get("size", float("nan"))) else "")
    if ctx["kind"] == "joint":
        return f"joint{ctx['axis']} {'+' if ctx['sign'] > 0 else '-'}{ctx['size']:g} deg"
    unit = "mm" if ctx["kind"] == "cart" else "deg"
    what = "move" if ctx["kind"] == "cart" else "rotate"
    return f"{what} {ctx['frame']} {'+' if ctx['sign'] > 0 else '-'}{ctx['axis']} {ctx['size']:g} {unit}"


class JogController:
    """Interface used by the backend (see module docstring).  ``node`` is spun by the backend's executor."""

    _pending_send = None        # (sid, send future) of a goal whose acceptance has not been seen yet
    _hooked_sends = None        # id -> send future that already carries the "cancel when accepted" hook (the future is
                                # kept so that its id can never be reused by a later one)

    def __init__(self, node, mode: str, real_enable_timeout_s: float = 600.0, *,
                 clock: Callable[[], float] = time.monotonic, allow_real_move: bool = False):
        if _ROS_IMPORT_ERROR is not None:
            raise RuntimeError(f"ROS python packages are not importable: {_ROS_IMPORT_ERROR}")
        self.node = node
        # "Move there" (plan_move / execute_move) on REAL hardware is off unless the UI is started with
        # allow_real_move:=true (operator decision after a supervised first test).  Steps are not affected.
        self._allow_real_move = bool(allow_real_move)
        self.core = JogCore(mode, real_enable_timeout_s, clock)
        self.cfgs = self.core.cfgs
        self._log = node.get_logger()
        self._threads: List[threading.Thread] = []
        self._goal_handle = None
        self._goal_lock = threading.Lock()
        self._urdf_text: Optional[str] = None
        self._urdf: Optional[Urdf] = None
        self._limits: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
        self._chains: Dict[str, Any] = {}
        self._model_error = ""
        self._model_name = ""
        self._model_lock = threading.Lock()
        self._pending: Dict[str, Optional[Tuple[Any, float]]] = {"hw": None, "ctrl": None}

        cb = ReentrantCallbackGroup()
        self._cb = cb
        node.create_subscription(JointState, "/joint_states", self._on_joint_states,
                                 QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT), callback_group=cb)
        node.create_subscription(String, "/robot_description", self._on_description,
                                 QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                            durability=DurabilityPolicy.TRANSIENT_LOCAL), callback_group=cb)
        self._cli_hw = node.create_client(ListHardwareComponents, "/controller_manager/list_hardware_components",
                                          callback_group=cb)
        self._cli_ctrl = node.create_client(ListControllers, "/controller_manager/list_controllers", callback_group=cb)
        self._cli_cart = node.create_client(GetCartesianPath, "/compute_cartesian_path", callback_group=cb)
        self._cli_valid = node.create_client(GetStateValidity, "/check_state_validity", callback_group=cb)
        self._cli_plan = node.create_client(GetMotionPlan, "/plan_kinematic_path", callback_group=cb)
        self._pub_display = node.create_publisher(DisplayTrajectory, "/display_planned_path", 10)
        self._cli_param = node.create_client(GetParameters, "/robot_state_publisher/get_parameters", callback_group=cb)
        self._actions = {arm: ActionClient(node, FollowJointTrajectory, f"/{cfg.controller}/follow_joint_trajectory",
                                           callback_group=cb) for arm, cfg in self.cfgs.items()}
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, node, spin_thread=False)
        self._timer = node.create_timer(CM_POLL_PERIOD_S, self._poll_controller_manager, callback_group=cb)

    # -- subscriptions / polling ----------------------------------------------------------------------
    def _on_joint_states(self, msg) -> None:
        self.core.update_joint_states(msg.name, msg.position, msg.velocity)

    def _on_description(self, msg) -> None:
        self._set_model(msg.data)

    def _set_model(self, text: str) -> None:
        if not text or text == self._urdf_text:
            return
        try:
            urdf = Urdf.parse(text)
            limits = parse_joint_limits(text)
            chains = {}
            for arm, cfg in self.cfgs.items():
                ch = urdf.chain(cfg.planning_frame, cfg.tip_link)
                if tuple(ch.joint_names) != cfg.joint_names:
                    raise KinematicsError(f"chain {cfg.planning_frame} -> {cfg.tip_link} has joints {ch.joint_names}, "
                                          f"expected {list(cfg.joint_names)}")
                chains[arm] = ch
        except Exception as e:  # noqa: BLE001 - reported on every refusal
            with self._model_lock:
                self._model_error = f"robot_description unusable for mode '{self.core.mode}': {e}"
            self._log.error(self._model_error)
            return
        try:
            name = ET.fromstring(text.encode("utf-8")).get("name") or ""
        except ET.ParseError:       # pragma: no cover - Urdf.parse succeeded above
            name = ""
        with self._model_lock:
            self._urdf_text, self._urdf, self._limits, self._chains, self._model_error = text, urdf, limits, chains, ""
            self._model_name = name

    def _poll_controller_manager(self) -> None:
        now = time.monotonic()
        for key, cli, req_cls, handler in (
                ("hw", self._cli_hw, ListHardwareComponents.Request, self._on_hw),
                ("ctrl", self._cli_ctrl, ListControllers.Request, self._on_ctrl)):
            pend = self._pending[key]
            if pend is not None and now - pend[1] < 3.0:
                continue
            if pend is not None:
                try:
                    cli.remove_pending_request(pend[0])
                except Exception:  # noqa: BLE001
                    pass
            if not cli.service_is_ready():
                self._pending[key] = None
                continue
            requested_at = self.core.clock()           # evidence time of the answer (see JogCore.update_hardware)
            fut = cli.call_async(req_cls())
            self._pending[key] = (fut, now)
            fut.add_done_callback(lambda f, h=handler, t=requested_at: h(f, t))

    def _on_hw(self, fut, requested_at: Optional[float] = None) -> None:
        self._pending["hw"] = None
        try:
            res = fut.result()
        except Exception:  # noqa: BLE001
            return
        if res is not None:
            self.core.update_hardware([(c.name, c.class_type, c.state.label) for c in res.component], requested_at)

    def _on_ctrl(self, fut, requested_at: Optional[float] = None) -> None:
        self._pending["ctrl"] = None
        try:
            res = fut.result()
        except Exception:  # noqa: BLE001
            return
        if res is not None:
            self.core.update_controllers([(c.name, c.state) for c in res.controller], requested_at)

    # -- public interface -------------------------------------------------------------------------------
    def status(self) -> dict:
        st = self.core.status()
        st["allow_real_move"] = bool(getattr(self, "_allow_real_move", False))
        with self._model_lock:
            st["model_ready"] = self._urdf is not None
            st["model_error"] = self._model_error
        return st

    def joint_window(self, arms: Sequence[str], window_s: float = STANDSTILL_WINDOW_S):
        return self.core.joint_window(arms, window_s)

    def enable_real(self, page_id: str) -> dict:
        res = self.core.enable_real(page_id)
        self._log.warning(f"jog: real-hardware enable by page {page_id!r}: {res['message']}")
        return res

    def disable(self, reason: str = "") -> None:
        self.core.disable(reason)
        self._log.info(f"jog: real-hardware enable cleared ({reason or 'no reason given'})")

    def reserve_motion(self) -> dict:
        """HTTP ingress gate (see JogCore.reserve_motion)."""
        return self.core.reserve_motion()

    def release_motion(self, token) -> None:
        self.core.release_motion(token)

    def step(self, page_id: str, arm: str, mode: str, axis, sign: int, size: float, frame: str = "world",
             epoch=None) -> dict:
        ok, sid, msg, ctx = self.core.begin_step(page_id, arm, mode, axis, sign, size, frame, epoch=epoch)
        if not ok:
            self._log.info(f"jog #{sid}: refused ({msg})")
            return {"ok": False, "id": sid, "message": msg}
        if not self._actions[arm].server_is_ready():
            self.core.finish(sid, False, "refused", f"controller action server for arm '{arm}' is not available")
            return {"ok": False, "id": sid, "message": f"controller action server for arm '{arm}' is not available"}
        if mode != "joint" and not self._cli_cart.service_is_ready():
            self.core.finish(sid, False, "refused", "/compute_cartesian_path (move_group) is not available")
            return {"ok": False, "id": sid, "message": "/compute_cartesian_path (move_group) is not available"}
        if not self._cli_valid.service_is_ready():
            self.core.finish(sid, False, "refused", "/check_state_validity (move_group) is not available")
            return {"ok": False, "id": sid, "message": "/check_state_validity (move_group) is not available"}
        self._log.info(f"jog #{sid}: accepted {arm} {_step_text(ctx)}{' [REAL hardware]' if ctx['real'] else ' [mock]'}")
        th = threading.Thread(target=self._worker, args=(ctx,), name=f"jog-{sid}", daemon=True)
        self._threads = [t for t in self._threads if t.is_alive()] + [th]
        th.start()
        return {"ok": True, "id": sid, "message": "accepted"}

    def descend(self, page_id: str, arm: str, T_base_target, epoch=None) -> dict:
        """Approach assist: move the TCP in a straight line (constant orientation, 2..70 mm) to ``T_base_target`` (4x4 TCP
        pose in the arm's base frame).  Exactly a Cartesian step otherwise (see the module doc): the same planning,
        caps, checks, supervision, STOP and fault semantics, and the same gating - PLUS, on REAL hardware, the
        ``allow_real_move`` switch exactly like plan_move / execute_move (safety review: the target comes from a camera
        prediction, not from a bounded step): REAL_MOVE_DISABLED_MSG otherwise, mock is unaffected.  Returns at once
        ("accepted" = dispatch accepted, not completed)."""
        if not getattr(self, "_allow_real_move", False) and any(self.core.is_real(a) for a in self.cfgs):
            return {"ok": False, "id": None, "message": REAL_MOVE_DISABLED_MSG}
        ok, sid, msg, ctx = self.core.begin_descend(page_id, arm, T_base_target, epoch=epoch)
        if not ok:
            self._log.info(f"jog #{sid}: descend refused ({msg})")
            return {"ok": False, "id": sid, "message": msg}
        for ready, text in ((self._actions[arm].server_is_ready(), f"controller action server for arm '{arm}' is not available"),
                            (self._cli_cart.service_is_ready(), "/compute_cartesian_path (move_group) is not available"),
                            (self._cli_valid.service_is_ready(), "/check_state_validity (move_group) is not available")):
            if not ready:
                self.core.finish(sid, False, "refused", text)
                return {"ok": False, "id": sid, "message": text}
        self._log.info(f"jog #{sid}: accepted {arm} {_step_text(ctx)}{' [REAL hardware]' if ctx['real'] else ' [mock]'}")
        th = threading.Thread(target=self._worker, args=(ctx,), name=f"jog-{sid}", daemon=True)
        self._threads = [t for t in self._threads if t.is_alive()] + [th]
        th.start()
        return {"ok": True, "id": sid, "message": "accepted"}

    # -- kinematics queries for the approach assist (no motion; refuse with a text instead of raising) -----------
    @staticmethod
    def _base_frame(cfg: ArmConfig) -> str:
        """The arm's base frame: the frame the calibration (and the recorder's touches) express the TCP in."""
        return f"{cfg.joint_prefix}base_link"

    def _model(self):
        with self._model_lock:
            urdf, err = self._urdf, self._model_error
        if urdf is None:
            raise _Refuse(err or "robot model (robot_description) not available yet")
        return urdf

    def _base_chain(self, arm: str):
        urdf = self._model()
        cache = getattr(self, "_base_chain_cache", None)
        if cache is None or cache[0] is not urdf:
            cache = (urdf, {})
            self._base_chain_cache = cache
        if arm not in cache[1]:
            cfg = self.cfgs[arm]
            try:
                ch = urdf.chain(self._base_frame(cfg), cfg.tip_link)
            except KinematicsError as e:
                raise _Refuse(f"robot model: {e}") from e
            if tuple(ch.joint_names) != cfg.joint_names:
                raise _Refuse(f"robot model: chain {self._base_frame(cfg)} -> {cfg.tip_link} has joints {ch.joint_names}")
            cache[1][arm] = ch
        return cache[1][arm]

    def tcp_pose_base(self, arm: str, require_standstill: bool = False) -> Tuple[Optional[np.ndarray], str]:
        """(T_base_tcp 4x4, "") of the arm's TCP in its base frame from forward kinematics of the latest joint snapshot,
        cross-checked against TF (TF_FK_POS_TOL_M / TF_FK_ROT_TOL_RAD); (None, reason) if the joint states are missing /
        stale / non-finite, the arm is not at standstill (when asked), the model or TF is missing or they disagree."""
        cfg = self.cfgs.get(arm)
        if cfg is None:
            return None, f"unknown arm {arm!r}"
        why = self.core.joint_gate(arm)
        if why:
            return None, why
        if require_standstill and not self.core.standstill(arm):
            return None, STANDSTILL_REFUSAL
        q = self.core.joint_snapshot(arm)
        if q is None:
            return None, f"joint_states of arm '{arm}' are not finite"
        try:
            T = self._base_chain(arm).fk(q)
            p_tf, R_tf = self._tf_pose(self._base_frame(cfg), cfg.tip_link)
        except (_Refuse, KinematicsError) as e:
            return None, str(e)
        if np.linalg.norm(p_tf - T[:3, 3]) > TF_FK_POS_TOL_M or rotation_angle(R_tf, T[:3, :3]) > TF_FK_ROT_TOL_RAD:
            return None, (f"TF and joint_states disagree about the TCP ({np.linalg.norm(p_tf - T[:3, 3]) * 1e3:.2f} mm, "
                          f"{math.degrees(rotation_angle(R_tf, T[:3, :3])):.2f} deg): refusing")
        return T, ""

    def tcp_fk(self, arm: str, q) -> Tuple[Optional[np.ndarray], str]:
        """(T_base_tcp, "") for joint values ``q`` (mapping joint name -> rad, or six values in joint order)."""
        cfg = self.cfgs.get(arm)
        if cfg is None:
            return None, f"unknown arm {arm!r}"
        try:
            if isinstance(q, Mapping):
                qa = np.array([float(q[n]) for n in cfg.joint_names], dtype=float)
            else:
                qa = np.asarray(q, dtype=float).reshape(6)
        except (KeyError, TypeError, ValueError):
            return None, "joint goal is incomplete"
        if not np.all(np.isfinite(qa)):
            return None, "joint goal is not finite"
        try:
            return self._base_chain(arm).fk(qa), ""
        except (_Refuse, KinematicsError) as e:
            return None, str(e)

    def link6_to_tcp(self, arm: str) -> Tuple[Optional[np.ndarray], str]:
        """(T_link6_tcp, "") - the fixed transform from the flange link to the TCP link, from the robot model."""
        cfg = self.cfgs.get(arm)
        if cfg is None:
            return None, f"unknown arm {arm!r}"
        try:
            return fixed_transform(self._model(), f"{cfg.joint_prefix}link6", cfg.tip_link), ""
        except (_Refuse, KinematicsError) as e:
            return None, str(e)

    def stop(self) -> dict:
        active = self.core.request_stop()
        self._cancel_goal()
        self._log.info("jog: stop requested" + ("" if active else " (no step active)"))
        return {"ok": True, "message": "stop requested: step cancelled" if active else "no step active"}

    def clear_fault(self) -> dict:
        """After a "fault" (stop not confirmed): release the arm.  Refused unless standstill is measured."""
        res = self.core.clear_fault()
        if res["ok"]:
            with self._goal_lock:
                self._goal_handle = None
                self._pending_send = None
        self._log.warning(f"jog: clear_fault: {res['message']}")
        return res

    # -- joint-goal moves (plan preview, then execute) ---------------------------------------------------
    def plan_move(self, page_id: str, arm: str, q_goal: Dict[str, float], epoch=None) -> dict:
        """Plan a collision-free joint-space path from the current joints to q_goal ({joint name: rad}, exactly the
        arm's six joints) and store it as THE move plan (replacing any older one).  BLOCKS (planning + one
        /check_state_validity call per 0.25 deg): the backend calls it from a thread pool.  While it runs the arm is
        busy with phase "planning" (STOP cancels it).  Nothing is sent to the controller.  Returns
        {ok, plan_id, message, duration_s, max_joint_change_deg, n_points, id}; plan_id is None unless ok."""
        if not getattr(self, "_allow_real_move", False) and any(self.core.is_real(a) for a in self.cfgs):
            return {"ok": False, "id": None, "plan_id": None, "message": REAL_MOVE_DISABLED_MSG}
        ok, sid, msg, ctx = self.core.begin_move_plan(page_id, arm, q_goal, epoch=epoch)
        if not ok:
            self._log.info(f"jog #{sid}: move planning refused ({msg})")
            return self._plan_reply(False, msg, sid)
        for ready, text in ((self._actions[arm].server_is_ready(), f"controller action server for arm '{arm}' is not available"),
                            (self._cli_plan.service_is_ready(), "/plan_kinematic_path (move_group) is not available"),
                            (self._cli_valid.service_is_ready(), "/check_state_validity (move_group) is not available")):
            if not ready:
                self.core.finish(sid, False, "refused", text)
                return self._plan_reply(False, text, sid)
        self._log.info(f"jog #{sid}: planning move of {arm}{' [REAL hardware]' if ctx['real'] else ' [mock]'}")
        try:
            return self._plan_move(ctx)
        except _Refuse as e:
            self.core.finish(sid, False, "refused", str(e))
            self._log.info(f"jog #{sid}: move planning refused ({e})")
            return self._plan_reply(False, str(e), sid)
        except _Stopped:
            self.core.finish(sid, False, "stopped", "stopped by operator")
            return self._plan_reply(False, "stopped by operator", sid)
        except Exception as e:  # noqa: BLE001
            self._log.error(f"jog #{sid}: move planning failed: {type(e).__name__}: {e}")
            self.core.finish(sid, False, "failed", f"internal error: {type(e).__name__}: {e}")
            return self._plan_reply(False, f"internal error: {type(e).__name__}: {e}", sid)

    @staticmethod
    def _plan_reply(ok: bool, message: str, sid: int, plan: Optional[dict] = None) -> dict:
        out = {"ok": ok, "plan_id": None, "message": message, "id": sid}
        if plan is not None:
            out.update(plan_id=plan["plan_id"], duration_s=round(plan["duration_s"], 3),
                       max_joint_change_deg=round(plan["max_joint_change_deg"], 4), n_points=plan["n_points"])
        return out

    def execute_move(self, page_id: str, plan_id: str, epoch=None) -> dict:
        """Execute the stored move plan through the same FollowJointTrajectory / result / STOP / standstill / fault
        machinery as a step.  Returns at once ("accepted" = dispatch accepted, not completed; progress and the
        achieved goal error are in status()["last"]).  The plan is consumed by an accepted request."""
        if not getattr(self, "_allow_real_move", False) and any(self.core.is_real(a) for a in self.cfgs):
            return {"ok": False, "id": None, "plan_id": None, "message": REAL_MOVE_DISABLED_MSG}
        ok, sid, msg, ctx = self.core.begin_move_execute(page_id, plan_id, epoch=epoch)
        if not ok:
            self._log.info(f"jog #{sid}: move refused ({msg})")
            return {"ok": False, "id": sid, "message": msg}
        arm = ctx["arm"]
        if not self._actions[arm].server_is_ready():
            self.core.finish(sid, False, "refused", f"controller action server for arm '{arm}' is not available")
            return {"ok": False, "id": sid, "message": f"controller action server for arm '{arm}' is not available"}
        ok, why = self.core.predispatch_check(ctx)          # the same recheck the worker repeats right before the send
        if not ok:
            self.core.finish(sid, False, "refused", why)
            return {"ok": False, "id": sid, "message": why}
        self._log.info(f"jog #{sid}: accepted {arm} {_step_text(ctx)}{' [REAL hardware]' if ctx['real'] else ' [mock]'}")
        th = threading.Thread(target=self._worker, args=(ctx,), name=f"jog-{sid}", daemon=True)
        self._threads = [t for t in self._threads if t.is_alive()] + [th]
        th.start()
        return {"ok": True, "id": sid, "plan_id": ctx["plan_id"], "message": "accepted"}

    def clear_plan(self) -> dict:
        had = self.core.clear_plan()
        return {"ok": True, "had_plan": had, "message": "stored move plan dropped" if had else "no move plan stored"}

    def shutdown(self) -> None:
        self.core.request_stop()
        self.core.cancel.set()
        self._cancel_goal()
        for t in list(self._threads):
            t.join(timeout=3.0)
        try:
            self.node.destroy_timer(self._timer)
        except Exception:  # noqa: BLE001
            pass

    def _cancel_when_accepted(self, sid: int, send_future) -> None:
        """A goal that is accepted after we gave up on it must not run: keep its handle (the arm stays busy until
        the cancel is confirmed or the fault is cleared) and cancel it as soon as it exists.  Idempotent per future.
        Also records the goal life cycle (live -> terminal when its result arrives; a rejection is terminal) so that
        clear_fault() waits for the controller's confirmation."""
        with self._goal_lock:
            if self._hooked_sends is None:
                self._hooked_sends = {}
            if id(send_future) in self._hooked_sends:
                return
            self._hooked_sends[id(send_future)] = send_future

        def late(f):
            try:
                gh = f.result()
            except Exception as e:  # noqa: BLE001
                # the send future failed: whether the goal exists is unknown - never "terminal"
                self._log.error(f"jog: late-accept answer is an exception: {type(e).__name__}: {e}")
                self.core.note_goal(sid, "unconfirmed")
                return
            if gh is None:
                self.core.note_goal(sid, "unconfirmed")
                return
            try:
                if not gh.accepted:
                    self.core.note_goal(sid, "terminal")        # explicitly rejected: nothing is live
                    return
                with self._goal_lock:
                    self._goal_handle = gh
                self.core.note_goal(sid, "live")
                try:
                    gh.get_result_async().add_done_callback(lambda rf: self._on_result_done(sid, rf))
                except Exception as e:  # noqa: BLE001
                    self._log.error(f"jog: could not watch the late goal's result: {e}")
                gh.cancel_goal_async()
            except Exception as e:  # noqa: BLE001
                self._log.error(f"jog: late-accept cancel failed: {type(e).__name__}: {e}")
        send_future.add_done_callback(late)

    def _on_result_done(self, sid: int, res_fut) -> None:
        """Result future of a goal completed.  "terminal" only for a VALIDATED result (completed without exception,
        status SUCCEEDED / CANCELED / ABORTED); anything else leaves the goal "unconfirmed" (fail closed)."""
        if validated_result(res_fut) is not None:
            self.core.note_goal(sid, "terminal")
        else:
            self._log.error(f"jog #{sid}: the goal's result future completed without a usable result")
            self.core.note_goal(sid, "unconfirmed")

    def _cancel_handle(self, gh) -> None:
        try:
            gh.cancel_goal_async()
        except Exception as e:  # noqa: BLE001
            self._log.error(f"jog: cancel_goal_async failed: {e}")

    def _cancel_goal(self) -> None:
        """Cancel the live goal; if the controller has not answered the send yet, arrange for the goal to be cancelled
        the moment it is accepted (STOP / fault must not depend on the worker thread still looping)."""
        with self._goal_lock:
            gh = self._goal_handle
            pend = self._pending_send
        if gh is not None:
            try:
                gh.cancel_goal_async()
            except Exception as e:  # noqa: BLE001
                self._log.error(f"jog: cancel_goal_async failed: {e}")
        if pend is not None:
            # not answered yet: cancel the moment it is accepted.  ALREADY answered (accepted) but the worker has not
            # recorded the handle yet: the same path cancels it right here (on a completed future the done callback runs
            # at once: handle recorded, result watched, cancel_goal_async called; idempotent per future and with the
            # worker's own later cancel).  A completed future that carries an exception is "unconfirmed" there.
            self._cancel_when_accepted(pend[0], pend[1])

    # -- worker ---------------------------------------------------------------------------------------------
    def _worker(self, ctx: dict) -> None:
        sid = ctx["id"]
        try:
            self._run(ctx)
        except _Refuse as e:
            self.core.finish(sid, False, "refused", str(e))
        except _Stopped:
            self.core.finish(sid, False, "stopped", "stopped by operator")      # nothing had been sent
        except Exception as e:  # noqa: BLE001
            self._log.error(f"jog #{sid}: failed: {type(e).__name__}: {e}")
            if ctx.get("sent"):
                # a goal may be live and nobody is watching it any more: keep the arm busy, ask for a stop
                self._cancel_goal()
                self.core.enter_fault(sid, f"internal error after dispatch ({type(e).__name__}: {e}) - "
                                           f"stop not confirmed, check the arm, use the e-stop")
            else:
                self.core.finish(sid, False, "failed", f"internal error: {type(e).__name__}: {e}")
        finally:
            if not self.core.in_fault():
                with self._goal_lock:
                    self._goal_handle = None
                    self._pending_send = None

    def _check_cancel(self) -> None:
        if self.core.cancel.is_set():
            raise _Stopped()

    def _wait(self, futs: Sequence[Any], timeout: float) -> bool:
        """Wait for all futures (done callbacks, no spinning).  True if all done; raises _Stopped on cancel."""
        remaining = [len(futs)]
        lock = threading.Lock()
        ev = threading.Event()

        def done(_f):
            with lock:
                remaining[0] -= 1
                if remaining[0] <= 0:
                    ev.set()

        for f in futs:
            f.add_done_callback(done)
        if not futs:
            return True
        deadline = time.monotonic() + timeout
        while not ev.wait(0.05):
            self._check_cancel()
            if time.monotonic() > deadline:
                return False
        return True

    def _ensure_model(self) -> None:
        with self._model_lock:
            if self._urdf is not None:
                return
            err = self._model_error
        if err:
            raise _Refuse(err)
        if self._cli_param.service_is_ready():
            req = GetParameters.Request()
            req.names = ["robot_description"]
            fut = self._cli_param.call_async(req)
            if self._wait([fut], 5.0) and fut.result() is not None and fut.result().values:
                self._set_model(fut.result().values[0].string_value)
        with self._model_lock:
            if self._urdf is None:
                raise _Refuse(self._model_error or "robot model (robot_description) not available yet")

    def _tcp_from_tf(self, cfg: ArmConfig) -> Tuple[np.ndarray, np.ndarray]:
        return self._tf_pose(cfg.planning_frame, cfg.tip_link)

    def _tf_pose(self, parent: str, child: str) -> Tuple[np.ndarray, np.ndarray]:
        try:
            tf = self._tf_buffer.lookup_transform(parent, child, Time())
        except Exception as e:  # noqa: BLE001
            raise _Refuse(f"TF {parent} -> {child} not available: {e}") from e
        t, r = tf.transform.translation, tf.transform.rotation
        return np.array([t.x, t.y, t.z]), quat_to_R(r.x, r.y, r.z, r.w)

    def _descend_target(self, ctx: dict, cfg: ArmConfig, p0: np.ndarray, R0: np.ndarray) -> None:
        """Worker side of a descend: the requested pose (arm base frame) -> the planning frame, and the limits that need
        the measured TCP: a straight move of DESCEND_MIN_M .. DESCEND_MAX_M, the same orientation as now (a pure
        translation; the target orientation is replaced by the measured one)."""
        try:
            T_pb = fixed_transform(self._model(), cfg.planning_frame, self._base_frame(cfg))
        except KinematicsError as e:
            raise _Refuse(f"descend blocked: {e}") from e
        T1 = T_pb @ np.asarray(ctx["T_base_target"], dtype=float)
        p1, R1 = T1[:3, 3].copy(), T1[:3, :3].copy()
        length = float(np.linalg.norm(p1 - p0))
        if not DESCEND_MIN_M <= length <= DESCEND_MAX_M:
            raise _Refuse(f"descend blocked: the target is {length * 1e3:.1f} mm from the TCP (allowed "
                          f"{DESCEND_MIN_M * 1e3:g}..{DESCEND_MAX_M * 1e3:g} mm)")
        ang = rotation_angle(R0, R1)
        if not ang <= DESCEND_MAX_ROT_RAD:
            raise _Refuse(f"descend blocked: the target orientation differs from the TCP's by {math.degrees(ang):.2f} deg "
                          f"(a descend is a pure translation, limit {math.degrees(DESCEND_MAX_ROT_RAD):g} deg)")
        ctx["target_p"], ctx["target_R"], ctx["size"] = p1, R0.copy(), length * 1e3

    def _robot_state(self, cfg: ArmConfig, q: np.ndarray):
        known = {n for c in self.cfgs.values() for n in c.joint_names}
        positions = {n: v for n, v in self.core.all_joint_positions().items() if n in known}
        positions.update({n: float(v) for n, v in zip(cfg.joint_names, q)})
        bad = [n for n, v in positions.items() if not math.isfinite(v)]
        if bad:
            raise _Refuse(f"joint_states contain non-finite values ({', '.join(bad)}): refusing to hand them to MoveIt")
        rs = RobotState()
        rs.joint_state.name = list(positions)
        rs.joint_state.position = [float(positions[n]) for n in rs.joint_state.name]
        return rs

    def _check_validity(self, ctx: dict, states: Sequence[np.ndarray], label: str = "step",
                        scene_changed: bool = False) -> None:
        """Ask /check_state_validity about every state, one request at a time (a burst of >~10 concurrent service
        requests overflows the DDS request queue and silently drops requests)."""
        cfg = ctx["cfg"]
        for i, q in enumerate(states):
            req = GetStateValidity.Request()
            req.robot_state = self._robot_state(cfg, q)
            req.group_name = cfg.group
            fut = self._cli_valid.call_async(req)
            try:
                answered = self._wait([fut], 5.0)
            finally:
                if not fut.done():
                    try:
                        self._cli_valid.remove_pending_request(fut)
                    except Exception:  # noqa: BLE001
                        pass
            if not answered:
                raise _Refuse("/check_state_validity did not answer within 5 s")
            res = fut.result()
            if res is None:
                raise _Refuse("/check_state_validity failed")
            if not res.valid:
                what = ""
                if res.contacts:
                    c = res.contacts[0]
                    what = f" ({c.contact_body_1} / {c.contact_body_2})"
                where = "start state" if i == 0 else f"state {i} of {len(states) - 1}"
                if scene_changed:
                    raise _Refuse(f"the planning scene changed since planning: {label} blocked, collision / invalid "
                                  f"state at {where}{what}")
                raise _Refuse(f"{label} blocked: collision / invalid state at {where}{what}")

    def _check_curve(self, ctx: dict, rt: Retimed, label: str) -> np.ndarray:
        """/check_state_validity on dense samples of the FINAL timed trajectory, i.e. of the cubic Hermite curve the
        controller interpolates through the re-timed points (<= CURVE_CHECK_SPACING_DEG between samples in every joint),
        in addition to the check of the planned polyline.  Refuses if the curve needs more than CURVE_CHECK_MAX_STATES
        calls.  Returns the checked samples (a move stores them for the scene re-check at execute time)."""
        try:
            states = curve_check_states(rt.times, rt.q, rt.qd, max_states=CURVE_CHECK_MAX_STATES)
        except ValueError as e:
            raise _Refuse(f"{label} blocked: {e}") from e
        self._check_validity(ctx, list(states), f"{label} (executed curve)")
        return states

    def _limits_ok(self, cfg: ArmConfig, Q: np.ndarray, label: str = "step") -> None:
        with self._model_lock:
            limits = dict(self._limits)
        for j, name in enumerate(cfg.joint_names):
            lo, hi = limits.get(name, (None, None))
            if lo is None:
                continue
            if np.min(Q[:, j]) < lo - 1e-9 or np.max(Q[:, j]) > hi + 1e-9:
                raise _Refuse(f"{label} blocked: {name} would leave its limits [{math.degrees(lo):.1f}, "
                              f"{math.degrees(hi):.1f}] deg (needs {math.degrees(np.min(Q[:, j])):.2f}.."
                              f"{math.degrees(np.max(Q[:, j])):.2f} deg)")

    def _plan_cartesian(self, ctx: dict, chain, p0, R0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cfg = ctx["cfg"]
        if ctx["kind"] == "descend":
            p1, R1 = ctx["target_p"], ctx["target_R"]        # set (and range-checked) by _descend_target
        else:
            p1, R1 = target_pose(p0, R0, ctx["kind"], ctx["axis"], ctx["sign"], ctx["size"], ctx["frame"])
        ctx["target"] = (p1, R1)
        if ctx["kind"] in ("cart", "descend"):
            ctx["direction"] = (p1 - p0) / float(np.linalg.norm(p1 - p0))
        # TOTG, which move_group runs on the result, merges path points closer than ~1e-3 rad in joint space, so a
        # sub-millimetre step comes back as ONE point (its start) and the end point is lost.  Short translations are
        # therefore planned 2 mm long (same line, same orientation) and cut at the requested distance below.
        ext = 2.0 / ctx["size"] if ctx["kind"] == "cart" and ctx["size"] < 2.0 else 1.0     # (a descend is >= 2 mm)
        wps = cart_waypoints(p0, R0, p0 + (p1 - p0) * ext, R1)
        # move_group time-parameterizes the path (TOTG, a point every 0.1 s at the joint velocity limits scaled by
        # max_velocity_scaling_factor) and the result is only a sampling of the path: at full speed a 5 deg or
        # 0.1 mm step comes back as 1-12 points (a very short one even without its end point).  Only the geometry
        # is used here, so the scaling factor is tuned until the sampling is dense (20..400 points); the timing
        # that comes back is discarded.
        factor = 0.05
        res = None
        for attempt in range(4):
            req = GetCartesianPath.Request()
            req.header.frame_id = cfg.planning_frame
            req.start_state = self._robot_state(cfg, ctx["q0"])
            req.group_name = cfg.group
            req.link_name = cfg.tip_link
            for p, R in wps:
                pose = Pose()
                pose.position.x, pose.position.y, pose.position.z = (float(v) for v in p)
                pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = R_to_quat(R)
                req.waypoints.append(pose)
            req.max_step = CART_WAYPOINT_SPACING_M
            # jump_threshold 0.0 disables MoveIt's *relative* jump filter (a ratio to the average joint step, which
            # misfires on these tiny uniform paths); the absolute per-point jump limit is enforced below.
            req.jump_threshold = 0.0
            req.avoid_collisions = True
            req.max_velocity_scaling_factor = float(factor)
            req.max_acceleration_scaling_factor = 1.0
            fut = self._cli_cart.call_async(req)
            try:
                answered = self._wait([fut], 10.0)
            finally:
                if not fut.done():
                    try:
                        self._cli_cart.remove_pending_request(fut)
                    except Exception:  # noqa: BLE001
                        pass
            if not answered:
                raise _Refuse("/compute_cartesian_path did not answer within 10 s")
            res = fut.result()
            if res is None:
                raise _Refuse("/compute_cartesian_path failed")
            if res.error_code.val != 1 or res.fraction < 1.0 - 1e-9:
                raise _Refuse(f"step blocked: collision/IK, fraction {res.fraction:.3f} "
                              f"(MoveIt error code {res.error_code.val})")
            pts = res.solution.joint_trajectory.points
            if 20 <= len(pts) <= 400 or attempt == 3:
                break
            last = pts[-1].time_from_start if pts else None
            dur = (last.sec + last.nanosec * 1e-9) if last is not None else 0.0
            factor = min(1.0, max(1e-3, factor * dur / 8.0)) if dur > 0.05 else max(1e-3, factor / 20.0)
        jt = res.solution.joint_trajectory
        idx = []
        for n in cfg.joint_names:
            if n not in jt.joint_names:
                raise _Refuse(f"Cartesian path lacks joint {n}")
            idx.append(list(jt.joint_names).index(n))
        Q = np.array([[pt.positions[i] for i in idx] for pt in jt.points], dtype=float)
        if len(Q) < 2:
            raise _Refuse("Cartesian path has fewer than two points")
        if np.max(np.abs(Q[0] - ctx["q0"])) > 1e-3:
            raise _Refuse("Cartesian path does not start at the current joint state")
        Q[0] = ctx["q0"]
        if ext > 1.0:
            Q = self._cut_at_distance(chain, Q, p0, p1)
        if np.max(np.abs(np.diff(Q, axis=0))) > MAX_WAYPOINT_JUMP_RAD:
            raise _Refuse(f"step blocked: IK jump of {np.max(np.abs(np.diff(Q, axis=0))):.3f} rad between path points")
        T = chain.fk(Q)
        P, Rm = T[:, :3, 3], T[:, :3, :3]
        if np.linalg.norm(P[-1] - p1) > END_POS_TOL_M or rotation_angle(Rm[-1], R1) > END_ROT_TOL_RAD:
            raise _Refuse(f"step blocked: path ends {np.linalg.norm(P[-1] - p1) * 1e3:.3f} mm / "
                          f"{math.degrees(rotation_angle(Rm[-1], R1)):.3f} deg off the target")
        # every path point must stay on the ideal line / rotation (no IK wobble)
        line = p1 - p0
        L = float(np.linalg.norm(line))
        for pt, Rk in zip(P, Rm):
            if L > 1e-12:
                u = line / L
                dev = np.linalg.norm((pt - p0) - u * float(np.dot(pt - p0, u)))
            else:
                dev = np.linalg.norm(pt - p0)
            if dev > PATH_POS_DEV_M:
                raise _Refuse(f"step blocked: path leaves the straight line by {dev * 1e3:.3f} mm")
            if rotation_angle(Rk, R0) > rotation_angle(R0, R1) + PATH_ROT_DEV_RAD:
                raise _Refuse("step blocked: path rotates beyond the requested angle")
        self._limits_ok(cfg, Q)
        self._check_validity(ctx, [Q[0], Q[-1]])
        return Q, P, Rm, T

    @staticmethod
    def _cut_at_distance(chain, Q: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
        """Keep the part of the joint path whose TCP has travelled |p1 - p0| along that direction; the last point is
        interpolated between the two samples around it (they are <= ~0.1 mm apart)."""
        line = p1 - p0
        L = float(np.linalg.norm(line))
        u = line / L
        prog = (chain.fk(Q)[:, :3, 3] - p0) @ u
        idx = np.nonzero(prog >= L - 1e-9)[0]
        if len(idx) == 0:
            raise _Refuse("step blocked: planned path is shorter than the requested step")
        i = int(idx[0])
        if i == 0:
            raise _Refuse("Cartesian path does not move along the requested direction")
        if max_reversal(prog[: i + 1]) > MONOTONIC_EPS_M:
            raise _Refuse(f"step blocked: planned path reverses by {max_reversal(prog[: i + 1]) * 1e3:.3f} mm "
                          f"before reaching the requested distance")
        f = (L - prog[i - 1]) / (prog[i] - prog[i - 1])
        return np.vstack([Q[:i], Q[i - 1] + f * (Q[i] - Q[i - 1])])

    def _plan_joint(self, ctx: dict, chain) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        cfg = ctx["cfg"]
        q0 = ctx["q0"]
        q1 = q0.copy()
        q1[ctx["axis"] - 1] += ctx["sign"] * math.radians(ctx["size"])
        n = max(1, int(math.ceil(ctx["size"] / JOINT_CHECK_SPACING_DEG - 1e-9)))
        Q = q0[None, :] + (q1 - q0)[None, :] * (np.arange(n + 1) / n)[:, None]
        Q[0], Q[-1] = q0, q1
        self._limits_ok(cfg, Q)
        self._check_validity(ctx, list(Q))
        T = chain.fk(Q)
        return Q, T[:, :3, 3], T[:, :3, :3]

    def _motion_plan_request(self, cfg: ArmConfig, q0: np.ndarray, q_goal: np.ndarray):
        """/plan_kinematic_path request: start = the joint snapshot, goal = the six joints (tolerance MOVE_GOAL_TOL_RAD).
        The scaling factors only keep MoveIt's own time-parameterisation slow; the timing used is ours."""
        req = GetMotionPlan.Request()
        mp = req.motion_plan_request
        mp.group_name = cfg.group
        mp.start_state = self._robot_state(cfg, q0)
        goal = Constraints()
        for name, v in zip(cfg.joint_names, q_goal):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = float(v)
            jc.tolerance_above = MOVE_GOAL_TOL_RAD
            jc.tolerance_below = MOVE_GOAL_TOL_RAD
            jc.weight = 1.0
            goal.joint_constraints.append(jc)
        mp.goal_constraints = [goal]
        mp.allowed_planning_time = MOVE_PLANNING_TIME_S
        mp.num_planning_attempts = MOVE_PLANNING_ATTEMPTS
        mp.max_velocity_scaling_factor = MOVE_PLAN_SCALING
        mp.max_acceleration_scaling_factor = MOVE_PLAN_SCALING
        return req

    def _call_moveit_plan(self, ctx: dict) -> np.ndarray:
        """Joint path (N, 6) from MoveIt: first point = the start snapshot, last point = the goal (both exact)."""
        cfg, q0, qg = ctx["cfg"], ctx["q0"], ctx["q_goal"]
        fut = self._cli_plan.call_async(self._motion_plan_request(cfg, q0, qg))
        try:
            answered = self._wait([fut], MOVE_PLAN_WAIT_S)
        finally:
            if not fut.done():
                try:
                    self._cli_plan.remove_pending_request(fut)
                except Exception:  # noqa: BLE001
                    pass
        if not answered:
            raise _Refuse(f"/plan_kinematic_path did not answer within {MOVE_PLAN_WAIT_S:g} s")
        res = fut.result()
        if res is None:
            raise _Refuse("/plan_kinematic_path failed")
        resp = res.motion_plan_response
        if resp.error_code.val != 1:
            raise _Refuse(f"move blocked: MoveIt found no path (error code {resp.error_code.val})")
        jt = resp.trajectory.joint_trajectory
        idx = []
        for n in cfg.joint_names:
            if n not in jt.joint_names:
                raise _Refuse(f"planned path lacks joint {n}")
            idx.append(list(jt.joint_names).index(n))
        Q = np.array([[pt.positions[i] for i in idx] for pt in jt.points], dtype=float).reshape(-1, len(cfg.joint_names))
        if len(Q) == 0 or not np.all(np.isfinite(Q)):
            raise _Refuse("planned path is empty or not finite")
        if np.max(np.abs(Q[0] - q0)) > 1e-3:
            raise _Refuse("planned path does not start at the current joint state")
        if np.max(np.abs(Q[-1] - qg)) > 2 * MOVE_GOAL_TOL_RAD:
            raise _Refuse(f"planned path ends {math.degrees(float(np.max(np.abs(Q[-1] - qg)))):.4f} deg away from the goal")
        return np.vstack([q0, Q[1:-1], qg]) if len(Q) > 2 else np.vstack([q0, qg])

    def _plan_move(self, ctx: dict) -> dict:
        sid, cfg, arm = ctx["id"], ctx["cfg"], ctx["arm"]
        q0, qg = ctx["q0"], ctx["q_goal"]
        self._ensure_model()
        chain = self._chains[arm]
        self._check_cancel()
        change = float(np.max(np.abs(qg - q0)))
        if change < MOVE_GOAL_TOL_RAD:
            raise _Refuse(f"the arm is already at the goal (max joint difference {math.degrees(change):.4f} deg)")
        self._limits_ok(cfg, np.vstack([q0, qg]), "move")
        # the model that gives the TCP speeds must agree with the live TF (same cross-check as a step)
        T0 = chain.fk(q0)
        p_tf, R_tf = self._tcp_from_tf(cfg)
        if np.linalg.norm(p_tf - T0[:3, 3]) > TF_FK_POS_TOL_M or rotation_angle(R_tf, T0[:3, :3]) > TF_FK_ROT_TOL_RAD:
            raise _Refuse(f"TF and joint_states disagree about the TCP ({np.linalg.norm(p_tf - T0[:3, 3]) * 1e3:.2f} mm, "
                          f"{math.degrees(rotation_angle(R_tf, T0[:3, :3])):.2f} deg): refusing")
        raw = self._call_moveit_plan(ctx)
        self._check_cancel()
        try:
            Qd = densify_path(smooth_path(raw))      # rounded corners, then the <= 0.25 deg guarantee
        except ValueError as e:
            raise _Refuse(f"move blocked: {e}") from e
        jump = max_jump(Qd)
        if jump > MAX_WAYPOINT_JUMP_RAD:
            raise _Refuse(f"move blocked: joint jump of {jump:.3f} rad between path points")
        self._limits_ok(cfg, Qd, "move")
        self._check_validity(ctx, list(Qd), "move")
        self._check_cancel()
        P = chain.fk(Qd)[:, :3, 3]
        rt, an = plan_move_timing(Qd, P, chain.fk)
        bad = move_cap_violations(an)
        if bad:
            raise _Refuse("move blocked: the re-timed trajectory exceeds the move caps (" + "; ".join(bad) + ")")
        if rt.duration > MOVE_MAX_DURATION_S:
            raise _Refuse(f"move blocked: the planned move would take {rt.duration:.0f} s (limit {MOVE_MAX_DURATION_S:g} s)")
        self._limits_ok(cfg, an["q"], "move")
        if np.max(np.abs(an["q"][-1] - qg)) > 1e-9 or np.max(np.abs(an["q"][0] - q0)) > 1e-9:
            raise _Refuse("internal error: the re-timed trajectory does not run from the start to the goal")
        # What the controller will execute (the cubic Hermite curve through the re-timed points) must stay on the
        # path that was validity-checked (the dense polyline Qd), and the re-timed points themselves must lie on it.
        off = polyline_deviation(Qd, rt.q)
        if not off <= ON_POLYLINE_TOL_RAD:
            raise _Refuse(f"internal error: the re-timed points are {off:.2e} rad off the validated path")
        dev = polyline_deviation(Qd, an["q"])
        if not dev <= MOVE_MAX_PATH_DEV_RAD:
            raise _Refuse(f"move blocked: the executed curve would leave the validated path by {dev:.4f} rad "
                          f"(limit {MOVE_MAX_PATH_DEV_RAD} rad)")
        self._check_cancel()
        curve = self._check_curve(ctx, rt, "move")          # collision check of the curve the controller will really run
        self._check_cancel()
        plan = {"arm": arm, "page_id": ctx["page_id"], "real": ctx["real"], "q_start": q0.copy(), "q_goal": qg.copy(),
                "Qd": Qd, "curve": curve, "path_dev_rad": dev,
                "rt": rt, "an": {k: an[k] for k in ("joint_rad_s", "joint_acc_rad_s2", "tcp_mm_s", "tcp_deg_s", "q")},
                "duration_s": float(rt.duration), "max_joint_change_deg": math.degrees(change), "n_points": len(rt.times)}
        plan_id = self.core.complete_plan(sid, plan)
        if plan_id is None:
            raise _Stopped()
        plan["plan_id"] = plan_id
        peaks = {k: round(an[k], 4) for k in ("joint_rad_s", "joint_acc_rad_s2", "tcp_mm_s", "tcp_deg_s")}
        self._log.info(f"jog #{sid}: {plan_id} stored: {rt.duration:.1f} s, {len(rt.times)} points, Hermite peaks {peaks}")
        self._publish_display(cfg, q0, rt)
        reply = self._plan_reply(True, f"move planned ({plan_id}): {plan['max_joint_change_deg']:.2f} deg max joint "
                                       f"change, {rt.duration:.1f} s; execute within {MOVE_PLAN_MAX_AGE_S:g} s",
                                 sid, plan)
        return reply

    def _display_trajectory(self, cfg: ArmConfig, q0: np.ndarray, rt: Retimed):
        msg = DisplayTrajectory()
        with self._model_lock:
            msg.model_id = getattr(self, "_model_name", "") or ""
        msg.trajectory_start = self._robot_state(cfg, q0)
        rtj = RobotTrajectory()
        rtj.joint_trajectory = self._joint_trajectory(cfg, rt)
        msg.trajectory = [rtj]
        return msg

    def _publish_display(self, cfg: ArmConfig, q0: np.ndarray, rt: Retimed) -> None:
        """Preview in RViz (/display_planned_path); the plan is already stored, so a failure here is only logged."""
        try:
            self._pub_display.publish(self._display_trajectory(cfg, q0, rt))
        except Exception as e:  # noqa: BLE001
            self._log.warning(f"jog: could not publish the move preview: {type(e).__name__}: {e}")

    def _run_move(self, ctx: dict) -> None:
        """Worker of an accepted execute_move(): last verification of the stored trajectory, then the same dispatch /
        result / STOP / standstill machinery as a step.  goal_time_tolerance stays GOAL_TIME_TOLERANCE_S (3 s, the
        controller's own value): it is the allowance for settling after the LAST point, and the trajectory is
        re-timed by us, so it does not need to scale with the (up to MOVE_MAX_DURATION_S) duration."""
        cfg, rt, an = ctx["cfg"], ctx["rt"], ctx["an"]
        self._check_cancel()
        self._ensure_model()
        bad = move_cap_violations(an)
        if bad:
            raise _Refuse("internal error: the stored move trajectory exceeds the move caps (" + "; ".join(bad) + ")")
        self._limits_ok(cfg, an["q"], "move")
        if np.max(np.abs(rt.q[0] - ctx["q0"])) > 1e-12 or np.max(np.abs(rt.q[-1] - ctx["q_goal"])) > 1e-12:
            raise _Refuse("internal error: the stored move trajectory does not run from the plan start to the goal")
        Qd = ctx.get("Qd")
        if Qd is not None:
            dev = polyline_deviation(Qd, an["q"])
            if not dev <= MOVE_MAX_PATH_DEV_RAD:
                raise _Refuse(f"internal error: the stored move curve leaves the validated path by {dev:.4f} rad")
            self._check_cancel()
        age = self.core.clock() - ctx["plan_created"] if ctx.get("plan_created") is not None else 0.0
        # The scene may have changed since the dense check at planning time (however recently): ALWAYS re-check a
        # coarse subset (about every 1 deg of joint travel, at most MOVE_RECHECK_MAX_CALLS) of the stored final Hermite
        # curve samples - the curve the controller will really run.  The arm stays busy (phase "checking") and STOP
        # stays effective (_wait raises _Stopped).
        self.core.set_phase(ctx["id"], "checking", f"re-checking the planning scene (plan is {age:.0f} s old)")
        curve = ctx.get("curve")
        if curve is None:
            try:
                curve = curve_check_states(rt.times, rt.q, rt.qd, max_states=CURVE_CHECK_MAX_STATES)
            except ValueError as e:
                raise _Refuse(f"move blocked: {e}") from e
        self._check_validity(ctx, list(coarse_states(curve)), "move", scene_changed=True)
        self._check_cancel()
        self._execute(ctx, rt, an)

    def _run(self, ctx: dict) -> None:
        if ctx["kind"] == "move":
            self._run_move(ctx)
            return
        sid, cfg, arm = ctx["id"], ctx["cfg"], ctx["arm"]
        self._ensure_model()
        chain = self._chains[arm]
        q0 = ctx["q0"]
        self._check_cancel()
        # TCP pose: forward kinematics of the SAME joint snapshot MoveIt gets as start state (no race with TF),
        # cross-checked against TF.
        T0 = chain.fk(q0)
        p0, R0 = T0[:3, 3], T0[:3, :3]
        p_tf, R_tf = self._tcp_from_tf(cfg)
        if np.linalg.norm(p_tf - p0) > TF_FK_POS_TOL_M or rotation_angle(R_tf, R0) > TF_FK_ROT_TOL_RAD:
            raise _Refuse(f"TF and joint_states disagree about the TCP ({np.linalg.norm(p_tf - p0) * 1e3:.2f} mm, "
                          f"{math.degrees(rotation_angle(R_tf, R0)):.2f} deg): refusing")
        if ctx["kind"] == "descend":
            self._descend_target(ctx, cfg, p0, R0)
        if ctx["kind"] == "joint":
            Q, P, Rm = self._plan_joint(ctx, chain)
        else:
            Q, P, Rm, _ = self._plan_cartesian(ctx, chain, p0, R0)
        self._check_cancel()
        rt, an = plan_timing(Q, P, Rm, chain.fk)
        bad = cap_violations(an)
        if bad:
            raise _Refuse("internal error: the cubic Hermite interpolation of the re-timed points exceeds the caps ("
                          + "; ".join(bad) + ")")
        kind = "cart" if ctx["kind"] == "descend" else ctx["kind"]       # a descend is progress along a line, like a cart step
        prog = step_progress(kind, an, p0, R0, ctx.get("direction"), joint_index=ctx["axis"] - 1 if kind == "joint" else 0,
                             sign=ctx["sign"])
        rev = max_reversal(prog)
        if rev > (MONOTONIC_EPS_M if kind == "cart" else MONOTONIC_EPS_RAD):
            raise _Refuse(f"internal error: the interpolated step reverses by {rev:.2e} ({'m' if kind == 'cart' else 'rad'})")
        self._limits_ok(cfg, an["q"])
        self._check_curve(ctx, rt, "step")          # collision check of the curve the controller will really run
        self._check_cancel()
        ctx["p0"], ctx["R0"], ctx["q_target"] = p0, R0, Q[-1]
        self._execute(ctx, rt, an)

    def _achieved(self, ctx: dict, chain, q_act: np.ndarray) -> Tuple[dict, str]:
        """Achieved vs requested step (or vs the joint goal of a move) from the settled joint snapshot."""
        if ctx["kind"] == "move":
            err = np.degrees(np.abs(np.asarray(q_act, dtype=float) - ctx["q_goal"]))
            j = int(np.argmax(err))
            return ({"max_joint_error_deg": round(float(err[j]), 4), "joint_error_deg": [round(float(e), 4) for e in err],
                     "plan_id": ctx["plan_id"]},
                    f"done: reached the joint goal, max joint error {float(err[j]):.3f} deg (joint {j + 1}); "
                    f"goal error per joint {', '.join(f'{e:.3f}' for e in err)} deg")
        kind, size = ("cart" if ctx["kind"] == "descend" else ctx["kind"]), ctx["size"]
        T = chain.fk(q_act)
        if kind == "cart":
            p1, _ = ctx["target"]
            ach = float((T[:3, 3] - ctx["p0"]) @ ctx["direction"]) * 1e3
            res = float(np.linalg.norm(T[:3, 3] - p1)) * 1e3
            return ({"requested_mm": size, "achieved_mm": round(ach, 4), "residual_mm": round(res, 4)},
                    f"done: moved {ach:.3f} mm of {size:g} mm, residual {res:.3f} mm")
        if kind == "rot":
            _, R1 = ctx["target"]
            ach = math.degrees(rotation_angle(ctx["R0"], T[:3, :3]))
            res = math.degrees(rotation_angle(R1, T[:3, :3]))
        else:
            j = ctx["axis"] - 1
            ach = math.degrees(ctx["sign"] * (q_act[j] - ctx["q0"][j]))
            res = math.degrees(float(np.max(np.abs(q_act - ctx["q_target"]))))
        return ({"requested_deg": size, "achieved_deg": round(ach, 4), "residual_deg": round(res, 4)},
                f"done: moved {ach:.3f} deg of {size:g} deg, residual {res:.3f} deg")

    @staticmethod
    def _joint_trajectory(cfg: ArmConfig, rt: Retimed):
        traj = JointTrajectory()
        traj.joint_names = list(cfg.joint_names)
        for t, q, qd in zip(rt.times, rt.q, rt.qd):
            pt = JointTrajectoryPoint()
            pt.positions = [float(v) for v in q]
            pt.velocities = [float(v) for v in qd]
            sec = int(t)
            pt.time_from_start = Duration(sec=sec, nanosec=int(round((t - sec) * 1e9)))
            traj.points.append(pt)
        return traj

    def _execute(self, ctx: dict, rt: Retimed, an: dict) -> None:
        sid, cfg, arm = ctx["id"], ctx["cfg"], ctx["arm"]
        traj = self._joint_trajectory(cfg, rt)
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj
        goal.goal_time_tolerance = Duration(sec=int(GOAL_TIME_TOLERANCE_S),
                                            nanosec=int(round((GOAL_TIME_TOLERANCE_S % 1) * 1e9)))
        # last look + send + registration of the pending send as ONE step against STOP (JogCore.dispatch_goal): a STOP
        # either wins before the gate (nothing is sent) or finds the registered pending send and hooks cancel-on-accept.
        def register(_sid, fut):
            with self._goal_lock:
                self._pending_send = (_sid, fut)

        ok, why, send = self.core.dispatch_goal(ctx, lambda: self._actions[arm].send_goal_async(goal), register,
                                                f"executing {_step_text(ctx)} ({rt.duration:.1f} s)")
        if not ok:
            raise _Refuse(why)
        moving = ctx["kind"] == "move"
        sup = MotionSupervisor(self.core, arm, rt, MOVE_MAX_JOINT_SPEED_RAD_S if moving else MAX_JOINT_SPEED_RAD_S,
                               MOVE_MAX_TRACKING_ERR_RAD if moving else STEP_MAX_TRACKING_ERR_RAD)
        peaks = {k: round(an[k], 4) for k in ("joint_rad_s", "joint_acc_rad_s2", "tcp_mm_s", "tcp_deg_s")}
        self._log.info(f"jog #{sid}: dispatched {rt.duration:.1f} s trajectory, Hermite peaks {peaks}")
        t_send = time.monotonic()
        exec_deadline = t_send + rt.duration + GOAL_TIME_TOLERANCE_S + 5.0
        gh = None
        res_fut = None
        send_resolved = rejected = send_failed = cancel_sent = late_cb = stopping = False
        t_accept = None
        while True:
            now = time.monotonic()
            if not send_resolved and send.done():
                send_resolved = True
                try:
                    gh = send.result()
                except Exception:  # noqa: BLE001
                    gh = None
                if gh is not None and gh.accepted:
                    with self._goal_lock:
                        self._goal_handle = gh
                    self.core.note_goal(sid, "live")
                    res_fut = gh.get_result_async()
                    res_fut.add_done_callback(lambda _f, _sid=sid: self._on_result_done(_sid, _f))
                    t_accept = now
                elif gh is not None:
                    gh, rejected = None, True                     # the controller said no: nothing is live
                    self.core.note_goal(sid, "terminal")
                else:
                    # no usable answer (exception / None): whether the goal exists is UNKNOWN - never "terminal"
                    gh, send_failed = None, True
                    self.core.note_goal(sid, "unconfirmed")
            if not stopping:
                final = None
                if self.core.cancel.is_set():
                    final = (False, "stopped", "stopped by operator")
                elif rejected:
                    final = (False, "refused", "controller rejected the trajectory goal")
                elif send_failed:
                    final = (False, "failed", GOAL_UNCONFIRMED_MSG)
                elif not send_resolved and now > t_send + 5.0:
                    final = (False, "failed", "controller did not answer the trajectory goal within 5 s")
                elif now > exec_deadline:
                    final = (False, "failed", "trajectory did not finish in time")
                elif res_fut is not None and res_fut.done():
                    res = validated_result(res_fut)
                    if res is None:
                        # exception / no response / unknown status: NOT a confirmed end of the goal (fail closed)
                        self.core.note_goal(sid, "unconfirmed")
                        final = (False, "failed", GOAL_UNCONFIRMED_MSG)
                    else:
                        self.core.note_goal(sid, "terminal")       # (idempotent) stamps the result time = the anchor
                        status = int(res.status)
                        code = res.result.error_code
                        if status == GOAL_STATUS_SUCCEEDED and code == 0:
                            if self._settle_and_finish(ctx, rt):
                                return
                            final = (False, "failed", "the arm did not settle after the trajectory")
                        elif goal_replaced(status, code, getattr(res.result, "error_string", "")):
                            self._log.warning(f"jog #{sid}: the controller aborted the goal: replaced by another command")
                            final = (False, "stopped", GOAL_REPLACED_MSG)
                        else:
                            err = getattr(res.result, "error_string", "")
                            final = (False, "stopped" if status == GOAL_STATUS_CANCELED else "failed",
                                     f"stopped by the controller {err}".strip() if status == GOAL_STATUS_CANCELED else
                                     f"controller reported status {status}, error code {code} {err}".strip())
                else:
                    why = sup.check(now - t_send, None if t_accept is None else now - t_accept)
                    if why:
                        self._log.error(f"jog #{sid}: supervision STOP: {why}")
                        self.core.request_stop()            # same path as the STOP button (cancel flag, plan dropped)
                        hint = (" - teleop active? (a command on the controller's topic replaces the running trajectory without "
                                "cancelling the goal): release the keys") if why.startswith(("tracking error", "measured joint speed")) else ""
                        final = (False, "stopped", f"stopped by supervision: {why}{hint}")
                if final is not None:
                    self.core.begin_stopping(sid, *final, result_seen=rejected)
                    stopping = True
            if stopping:
                if gh is not None and not cancel_sent:
                    self._cancel_handle(gh)
                    cancel_sent = True
                if not send_resolved and not late_cb:
                    self._cancel_when_accepted(sid, send)  # a late-accepted goal must be cancelled, arm stays busy
                    late_cb = True
                if rejected:
                    self.core.note_result(sid)
                elif res_fut is not None and res_fut.done():
                    if validated_result(res_fut) is not None:
                        self.core.note_result(sid)
                    else:
                        self.core.note_goal(sid, "unconfirmed")      # a failed result future is not a confirmation
                state = self.core.poll_stopping()
                if state == "confirmed":
                    return
                if state == "fault":
                    self._log.error(f"jog #{sid}: FAULT - stop not confirmed (goal handle kept, arm stays busy)")
                    return
            time.sleep(0.05)

    def _settle_and_finish(self, ctx: dict, rt: Retimed) -> bool:
        """After 'goal succeeded': wait up to SETTLE_WAIT_S for a measured, EVENT-ANCHORED standstill (only samples
        received after the goal result count; hardware healthy), then report achieved vs requested."""
        sid, arm = ctx["id"], ctx["arm"]
        end = time.monotonic() + SETTLE_WAIT_S
        anchor = self.core.goal_end_time(sid)
        if anchor is None:
            anchor = self.core.clock()
        while self.core.confirmation_problem(arm, anchor) is not None:
            if time.monotonic() > end:
                return False
            time.sleep(0.05)
        q_act = self.core.joint_snapshot(arm)
        extra, msg = self._achieved(ctx, self._chains[arm], q_act)
        self._log.info(f"jog #{sid}: {msg} (trajectory {rt.duration:.1f} s)")
        self.core.finish(sid, True, "done", msg, extra)
        return True
