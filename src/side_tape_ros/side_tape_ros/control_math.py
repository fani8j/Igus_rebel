"""Pure math for side_tape_controller: no rclpy, no ROS message types.

Everything here takes/returns plain numbers and ``numpy`` arrays (or the
pure-Python core's dataclasses) so it can be unit-tested without spinning a
node. See docs/side_seam_architecture.md S3/S4/S4b and
precision_box_detector/side_tape/{types,config,supervisor}.py for the
conventions this implements:

- Cut frame axes (``cut``, ``lateral``, ``depth``) are unit vectors given in
  ``tool_frame`` (``RobotCfg``); they are "frozen" into ``base_frame`` at the
  start of ALIGN/CUT by rotating with the tool's orientation *at that
  instant* (:func:`freeze_axes`), then held fixed in the base frame for the
  rest of the phase (the tool itself may rotate slightly under Servo).
- ``TcpState`` (mm) is the tool's position relative to the frozen start pose,
  projected onto the frozen axes (:func:`tcp_state`).
- A :class:`~precision_box_detector.side_tape.types.ControlOutput` (cut frame,
  m/s) is composed into a base-frame linear velocity by
  :func:`compose_linear_velocity`.
- MoveIt Servo's ``~/status`` (std_msgs/Int8) is mapped to the supervisor's
  three-level :class:`ServoStatus` by :func:`map_servo_status_int`; "no
  status received yet/in a while" is a fourth, ROS-only label ("UNKNOWN")
  collapsed onto WARN (non-dry-run) or OK (dry-run, where Servo may not even
  be running) by :func:`servo_status_for_supervisor`.
- :func:`trajectory_goal_active` reads ``action_msgs/GoalStatusArray``-style
  integer status codes (STATUS_ACCEPTED=1, STATUS_EXECUTING=2,
  STATUS_CANCELING=3) to detect MoveIt owning the arm controller.
- :func:`arm_owned` is the interlock (S3): TF fresh, no active trajectory
  goal, and (dry_run, or we started+unpaused Servo ourselves and its status
  is still fresh).
- :func:`sanitize_linear_velocity` is the node's own absolute clamp (defence
  in depth on top of the supervisor's own limits) plus NaN detection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import core_path  # noqa: F401  (adds the pure-Python core to sys.path)
from precision_box_detector.side_tape.supervisor import ServoStatus  # noqa: E402
from precision_box_detector.side_tape.types import ControlOutput  # noqa: E402

__all__ = [
    "quat_to_rotation_matrix",
    "freeze_axes",
    "tcp_state_mm",
    "compose_linear_velocity",
    "map_servo_status_int",
    "servo_status_for_supervisor",
    "trajectory_goal_active",
    "arm_owned",
    "sanitize_linear_velocity",
    "TfState",
    "TcpVectors",
    "acm_add_allowed_pairs",
    "acm_matrices_equal",
    "parse_get_parameters_result",
    "servo_params_ok",
    "servo_incoming_timeout_warning",
]

# action_msgs/msg/GoalStatus status codes that mean "MoveIt is driving (or
# about to drive, or trying to stop driving) the arm controller right now".
_ACTIVE_GOAL_STATUSES = (1, 2, 3)  # STATUS_ACCEPTED, STATUS_EXECUTING, STATUS_CANCELING

# Servo ~/status (std_msgs/Int8) levels, per docs S4b / moveit_servo/servo_node.h.
_SERVO_OK = (0,)
_SERVO_WARN = (1, 3, 6)  # DECEL_SINGULARITY, DECEL_COLLISION, DECEL_LEAVING_SINGULARITY
_SERVO_HALT = (2, 4, 5)  # HALT_SINGULARITY, HALT_COLLISION, JOINT_BOUND


@dataclass(frozen=True)
class TfState:
    """One TF lookup result, already reduced to plain numbers."""

    position_m: np.ndarray  # (3,) in base_frame
    quat_xyzw: np.ndarray  # (4,)
    age_s: float  # now - lookup stamp
    ok: bool  # lookup succeeded at all


@dataclass(frozen=True)
class TcpVectors:
    """Frozen cut-frame axes, expressed in base_frame, plus the frozen origin."""

    cut: np.ndarray
    lateral: np.ndarray
    depth: np.ndarray
    p0_m: np.ndarray


def quat_to_rotation_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    """Standard xyzw quaternion -> 3x3 rotation matrix (assumes unit quaternion)."""
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        raise ValueError("degenerate quaternion")
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def freeze_axes(R_base_tool: np.ndarray, cut_axis_tool, lateral_axis_tool, depth_axis_tool) -> tuple:
    """Rotate the (unit, tool-frame) cut-frame axes into base_frame, once."""
    cut = R_base_tool @ np.asarray(cut_axis_tool, dtype=np.float64)
    lat = R_base_tool @ np.asarray(lateral_axis_tool, dtype=np.float64)
    depth = R_base_tool @ np.asarray(depth_axis_tool, dtype=np.float64)
    return cut, lat, depth


def tcp_state_mm(p_base_m: np.ndarray, vecs: TcpVectors):
    """Tool pose vs. the frozen start pose, projected onto the frozen axes, in mm.

    Returns ``(along_mm, lat_dev_mm, depth_dev_mm)`` (see
    ``supervisor.TcpState`` for the meaning of each).
    """
    d_mm = (np.asarray(p_base_m, dtype=np.float64) - vecs.p0_m) * 1000.0
    along_mm = float(d_mm @ vecs.cut)
    lat_dev_mm = float(d_mm @ vecs.lateral)
    depth_dev_mm = float(d_mm @ vecs.depth)
    return along_mm, lat_dev_mm, depth_dev_mm


def compose_linear_velocity(control: ControlOutput, vecs: TcpVectors) -> np.ndarray:
    """Cut-frame (m/s) -> base-frame linear velocity (m/s), angular is always 0."""
    return (control.v_cut_mps * vecs.cut
            + control.v_lat_mps * vecs.lateral
            + control.v_depth_mps * vecs.depth)


def map_servo_status_int(value: int | None) -> str:
    """Servo ``~/status`` (Int8) -> one of "OK"/"WARN"/"HALT"/"UNKNOWN".

    ``None`` = no status message received yet (or too stale to trust,
    decided by the caller via :func:`servo_status_for_supervisor`'s
    ``status_age_s``).
    """
    if value is None:
        return "UNKNOWN"
    if value in _SERVO_OK:
        return "OK"
    if value in _SERVO_WARN:
        return "WARN"
    if value in _SERVO_HALT:
        return "HALT"
    return "UNKNOWN"  # an undocumented code: treat like "never got a real status"


def servo_status_for_supervisor(status_label: str, dry_run: bool) -> ServoStatus:
    """Collapse the ROS-only "UNKNOWN" label onto what the supervisor understands.

    UNKNOWN (no status ever received, or stale beyond 1 s) is treated as WARN
    when Servo should be driving the arm (not dry_run: we can't tell whether
    it is safe, so don't proceed as if it were fine), and as OK in dry_run
    (Servo may not even be started, and no twist is ever sent to it anyway).
    """
    if status_label == "OK":
        return ServoStatus.OK
    if status_label == "HALT":
        return ServoStatus.HALT
    if status_label == "WARN":
        return ServoStatus.WARN
    return ServoStatus.OK if dry_run else ServoStatus.WARN


def trajectory_goal_active(goal_statuses) -> bool:
    """``True`` if any action_msgs/GoalStatus.status in ``goal_statuses`` means
    MoveIt's trajectory-execution action is accepted/executing/canceling."""
    return any(int(s) in _ACTIVE_GOAL_STATUSES for s in goal_statuses)


def arm_owned(tf_fresh: bool, trajectory_active: bool, dry_run: bool,
              servo_unpaused_by_us: bool, servo_status_age_s: float | None) -> bool:
    """Interlock (docs S3): exactly one of {MoveIt, us} may drive the arm.

    ``servo_status_age_s`` is ``now - last status stamp``, or ``None`` if no
    status has ever arrived; "received within 1 s" per the task spec.
    """
    if not tf_fresh or trajectory_active:
        return False
    if dry_run:
        return True
    servo_status_fresh = servo_status_age_s is not None and servo_status_age_s <= 1.0
    return bool(servo_unpaused_by_us) and servo_status_fresh


def sanitize_linear_velocity(v: np.ndarray, v_cut_mps: float, v_lat_max_mps: float,
                              v_retract_mps: float) -> tuple[np.ndarray, bool]:
    """Absolute defence-in-depth clamp + NaN guard on the composed velocity.

    Returns ``(v_safe, nan_detected)``. ``nan_detected`` -> caller must zero
    and ABORT (the supervisor's own math should never produce a NaN; this is
    a last-resort guard against a TF glitch or similar).
    """
    v = np.asarray(v, dtype=np.float64)
    if not np.all(np.isfinite(v)):
        return np.zeros(3), True
    limit = max(v_cut_mps, v_lat_max_mps, v_retract_mps) * 1.2
    norm = float(np.linalg.norm(v))
    if norm > limit and norm > 1e-12:
        v = v * (limit / norm)
    return v, False


# -- Allowed Collision Matrix editing (finding 8) ----------------------------
#
# In Humble, applying a PlanningScene diff with an AllowedCollisionMatrix
# REPLACES the whole matrix (it is not merged like most other diff fields).
# So the caller must: fetch the *current* full ACM (moveit_msgs/GetPlanningScene,
# components=ALLOWED_COLLISION_MATRIX), pass its (entry_names, enabled) here
# to compute the new full matrix, apply that via ApplyPlanningScene, and later
# restore by re-applying the exact (entry_names, enabled) it fetched originally
# (a plain, unmodified round-trip -- no math needed for "restore" beyond
# keeping that original copy). ``enabled`` is a plain square list-of-lists
# mirroring ``AllowedCollisionMatrix.entry_values[i].enabled``: row i, col j
# says whether entry_names[i] is allowed to collide with entry_names[j].

def acm_add_allowed_pairs(
    entry_names: list, enabled: list,
    object_id: str, allowed_links: list,
) -> tuple[list, list]:
    """Return a NEW ``(entry_names, enabled)`` with ``object_id`` allowed to
    collide with each of ``allowed_links``. Any name not already present is
    added as a new all-``False`` row/column first, so pre-existing pairs
    (including ones already ``True`` for unrelated reasons) are left exactly
    as they were. Does not mutate its inputs.
    """
    names = list(entry_names)
    rows = [list(row) for row in enabled]
    for row in rows:
        if len(row) != len(names):
            raise ValueError("enabled must be a square matrix matching entry_names")

    def _ensure(name: str) -> int:
        if name in names:
            return names.index(name)
        idx = len(names)
        names.append(name)
        for row in rows:
            row.append(False)
        rows.append([False] * (idx + 1))
        return idx

    oi = _ensure(object_id)
    for link in allowed_links:
        li = _ensure(link)
        rows[oi][li] = True
        rows[li][oi] = True
    return names, rows


def acm_matrices_equal(names_a: list, enabled_a: list, names_b: list, enabled_b: list) -> bool:
    """True if the two ACMs describe the same set of allowed pairs, ignoring
    row/column order (useful in tests, and to sanity-check a restore)."""
    def _as_pairs(names, enabled):
        pairs = set()
        for i, ni in enumerate(names):
            for j, nj in enumerate(names):
                if enabled[i][j]:
                    pairs.add(frozenset((ni, nj)) if ni != nj else (ni, nj))
        return pairs
    return _as_pairs(names_a, enabled_a) == _as_pairs(names_b, enabled_b)


# -- MoveIt Servo parameter contract check (finding 7d) ----------------------

def parse_get_parameters_result(names: list, values: list) -> dict:
    """rcl_interfaces/srv/GetParameters response -> ``{name: python_value}``.

    ``values[i]`` is a duck-typed ``rcl_interfaces/msg/ParameterValue``: only
    ``.type`` and the one populated ``*_value`` field are read. Unset/unknown
    parameters (``type == PARAMETER_NOT_SET == 0``) map to ``None``.
    """
    # rcl_interfaces/msg/ParameterType values.
    _STRING, _BOOL, _INT, _DOUBLE = 4, 1, 2, 3
    out = {}
    for name, value in zip(names, values):
        t = int(value.type)
        if t == _STRING:
            out[name] = str(value.string_value)
        elif t == _BOOL:
            out[name] = bool(value.bool_value)
        elif t == _INT:
            out[name] = int(value.integer_value)
        elif t == _DOUBLE:
            out[name] = float(value.double_value)
        else:
            out[name] = None
    return out


def servo_params_ok(
    command_frame, command_in_type, incoming_timeout_s,
    cfg_command_frame: str, cfg_command_in_type: str, cfg_max_incoming_timeout_s: float,
) -> tuple[bool, str]:
    """Check the live Servo node's parameters against ``RobotCfg`` (finding 7d).

    Any of the three being ``None`` (parameter missing/unreadable) refuses.
    Returns ``(ok, message)``; ``message`` is empty when ``ok``, otherwise a
    recommendation is appended to a slow ``incoming_timeout_s`` even when it
    is within bounds (kept separate from ``ok`` since it is only a warning).
    """
    if command_frame is None or command_in_type is None or incoming_timeout_s is None:
        return False, "servo_params_unavailable"
    if str(command_frame) != cfg_command_frame:
        return False, (f"servo robot_link_command_frame={command_frame!r} "
                        f"!= config.robot.servo_command_frame={cfg_command_frame!r}")
    if str(command_in_type) != cfg_command_in_type:
        return False, (f"servo command_in_type={command_in_type!r} "
                        f"!= config.robot.servo_command_in_type={cfg_command_in_type!r}")
    if float(incoming_timeout_s) > cfg_max_incoming_timeout_s:
        return False, (f"servo incoming_command_timeout={incoming_timeout_s} "
                        f"> config.robot.servo_max_incoming_timeout_s={cfg_max_incoming_timeout_s}")
    return True, ""


def servo_incoming_timeout_warning(incoming_timeout_s: float) -> str | None:
    """A non-blocking recommendation (finding 7d): "log a warning
    recommending 0.1s if > 0.1", independent of whether it still passes
    ``servo_params_ok`` against a looser configured max."""
    if incoming_timeout_s is not None and float(incoming_timeout_s) > 0.1:
        return (f"servo incoming_command_timeout={incoming_timeout_s} > 0.1s "
                f"recommended for a 100 Hz publisher")
    return None
