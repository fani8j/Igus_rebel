"""The ROS-facing interface the testable core talks to.  ``ros_bridge.RosBridge`` implements it with rclpy; the tests use
a fake.  All times are seconds on one monotonic clock shared with the core (``clock``).

Blocking methods (``trigger``, ``set_gates``, ``cancel_all``) are only called from worker threads, never from the HTTP
loop or from a ROS callback; they always return (ok, message) and never raise for a missing service or a timeout.
The status listener, ``subscribe_camera`` callbacks run on the ROS executor thread and must stay cheap.
"""
import threading
from typing import Callable, Dict, List, Optional, Tuple

STATE_NAMES = {0: "IDLE", 1: "PLANNING", 2: "PLANNED", 3: "WAITING_BARRIER", 4: "EXECUTING", 5: "STOPPED"}
STATE_IDLE, STATE_PLANNING, STATE_PLANNED, STATE_WAITING_BARRIER, STATE_EXECUTING, STATE_STOPPED = range(6)
GATE_PARAMS = ("execution_enabled", "execute_phase_a_enabled", "execute_phase_b_enabled")      # every one is written false on any OFF path
ENABLE_PARAMS = ("execution_enabled", "execute_phase_a_enabled")          # the only ones "Enable execution" writes true (phase B stays untouched)
DISPLAY_PARAMS = ("phase_order", "schedule_mode", "top_split_mode", "external_snapshot_max_age_s")
# Runtime bool on the coordinator (default false): while false, executing a plan of a detection whose snapshot cut_permission.status is
# "blocked" is refused by the coordinator.  The UI switch writes it; it is NOT a gate parameter (never part of set_gates), so a
# coordinator without this parameter cannot make the gate closing fail.
ALLOW_BLOCKED_PARAM = "allow_blocked_detection"
ARMS = ("left", "right")
SCHEDULE_MODES = ("phases", "sequence")
# The DetectBox goal exactly as the operator sends it by hand:
#   ros2 action send_goal --feedback /box_seam_detector/detect_box compal_box_msgs/action/DetectBox "{dry_run: false}"
DETECT_GOAL = {"max_attempts": 0, "allow_needs_review": False, "allow_cut_blocked": False, "dry_run": False}
DETECT_SOURCE_HINT = "source /home/farhan/orca/workspaces/igus_rebel_ros2/box-seam-detection/install/setup.bash"


def gate_params_for(enabled: bool) -> Tuple[str, ...]:
    """The parameters ``set_gates(enabled)`` writes: ON writes only ENABLE_PARAMS, OFF writes all of GATE_PARAMS (an off path
    never writes fewer parameters than before)."""
    return ENABLE_PARAMS if enabled else GATE_PARAMS


class Bridge:
    # ---- latest cached data: (value, rx_time) or None when nothing was received yet --------------------------------
    def status(self) -> Optional[Tuple[dict, float]]:
        """Latest CoordinatorStatus as a dict (fields: state, phase, plan_seq, plan_retained, phase_validated,
        min_cross_distance_m, closest_link_a/b, joint_states_age_s, stop_reason, stop_latched, barrier_ok,
        barrier_message)."""
        raise NotImplementedError

    def add_status_listener(self, fn: Callable[[dict, float], None]) -> None:
        """fn(status_dict, rx_time) is called for EVERY status message (ROS thread; must not block)."""
        raise NotImplementedError

    def plan_report(self) -> Optional[Tuple[str, float]]:
        raise NotImplementedError

    def exec_reports(self) -> List[Tuple[str, float]]:
        """Newest last."""
        raise NotImplementedError

    def health(self) -> Optional[Tuple[Dict[str, Dict[str, bool]], float]]:
        """({"controllers": {"left": bool, "right": bool}, "hardware": {...}}, rx_time) from the polled
        list_controllers / list_hardware_components (None until both polls have answered; rx_time is the older of the two)."""
        raise NotImplementedError

    def gates(self) -> Optional[Tuple[dict, float]]:
        """Polled coordinator parameters: GATE_PARAMS (bool) and DISPLAY_PARAMS, missing ones None."""
        raise NotImplementedError

    def allow_blocked(self) -> Optional[Tuple[Optional[bool], float]]:
        """(value, rx_time) of the polled coordinator parameter ``allow_blocked_detection``; value None when the coordinator
        does not have the parameter (or it is not a bool); None until the first poll answered."""
        return None

    # ---- blocking calls -------------------------------------------------------------------------------------------
    def trigger(self, name: str, timeout_s: float) -> Tuple[bool, str]:
        """Call /dual_cut_coordinator/<name> (std_srvs/Trigger); name in plan, execute, stop, rearm."""
        raise NotImplementedError

    def set_gates(self, enabled: bool, timeout_s: float) -> Tuple[bool, str]:
        """enabled=True: set ENABLE_PARAMS (execution_enabled, execute_phase_a_enabled) true on the coordinator.
        enabled=False: set ALL of GATE_PARAMS (also execute_phase_b_enabled) false.  See ``gate_params_for``."""
        raise NotImplementedError

    def set_schedule_mode(self, mode: str, timeout_s: float) -> Tuple[bool, str]:
        """Set the coordinator parameter ``schedule_mode`` (one of SCHEDULE_MODES) like ``ros2 param set``."""
        raise NotImplementedError

    def set_allow_blocked(self, enabled: bool, timeout_s: float) -> Tuple[bool, str]:
        """Set the coordinator parameter ``allow_blocked_detection`` (bool) like ``ros2 param set``.  Never part of the
        execution gates."""
        raise NotImplementedError

    def detect(self, action: str, goal: dict, on_feedback: Callable[[str, int, str], None], cancel: threading.Event,
               timeout_s: float) -> Tuple[bool, dict]:
        """Send a compal_box_msgs/action/DetectBox goal (fields of ``goal``: max_attempts, allow_needs_review,
        allow_cut_blocked, dry_run) to ``action`` and block (worker thread) until the result, calling
        ``on_feedback(stage, attempt, message)`` for every feedback.  Setting ``cancel`` (from any thread) cancels the goal.
        Returns (ok, info): ok is the detector's ``success``.  ``info`` has the result fields (success, status, reason,
        reasons, cut_permission, snapshot_id, snapshot_path, capture_id, seam_status, attempts, elapsed_s) when the detector
        answered; ``{"error": text}`` when it could not be reached / rejected the goal / the interface is missing / no
        answer in ``timeout_s``; ``{"cancelled": True}`` (plus result fields if known) after a cancel."""
        raise NotImplementedError

    def cancel_all(self, arm: str, timeout_s: float) -> Tuple[bool, str]:
        """Cancel every goal of that arm's FollowJointTrajectory action (zero goal id + zero stamp)."""
        raise NotImplementedError

    # ---- cameras --------------------------------------------------------------------------------------------------
    def subscribe_camera(self, name: str, topic: str, callback: Callable[[bytes], None]) -> None:
        raise NotImplementedError

    def unsubscribe_camera(self, name: str) -> None:
        raise NotImplementedError


# action_msgs/srv/CancelGoal return codes
CANCEL_ERROR_NONE, CANCEL_ERROR_REJECTED, CANCEL_ERROR_UNKNOWN_GOAL_ID, CANCEL_ERROR_GOAL_TERMINATED = 0, 1, 2, 3


def interpret_cancel(return_code: int, n_canceling: int) -> Tuple[bool, str]:
    """(ok, message) for a cancel-all answer.  Goals cancelling -> ok whatever the code.  NONE with no goal -> ok, nothing to
    cancel.  REJECTED with no goal -> NOT ok (the controller refused: the arm may still be moving).  UNKNOWN_GOAL_ID /
    GOAL_TERMINATED with no goal -> ok, nothing active.  Anything else -> not ok.  The code is always in the message."""
    if n_canceling > 0:
        return True, f"{n_canceling} goal(s) cancelling (code {return_code})"
    if return_code == CANCEL_ERROR_NONE:
        return True, "no active goal (code 0)"
    if return_code == CANCEL_ERROR_REJECTED:
        return False, "cancel REJECTED by the controller (code 1)"
    if return_code in (CANCEL_ERROR_UNKNOWN_GOAL_ID, CANCEL_ERROR_GOAL_TERMINATED):
        return True, f"no active goal (code {return_code})"
    return False, f"unexpected cancel return code {return_code}"
