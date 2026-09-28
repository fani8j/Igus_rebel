"""`side_tape_controller`: the only publisher to MoveIt Servo's ``delta_twist_cmds``.

Runs the pure-Python :class:`~precision_box_detector.side_tape.supervisor.Supervisor`
against TF, the perception node's ``Observation`` and MoveIt Servo's own status,
and turns its :class:`ControlOutput` into a base-frame twist. See
docs/side_seam_architecture.md S3/S4/S4b and side_tape_ros/control_math.py
(all the non-ROS math lives there and is unit-tested there).

Safety defaults: ``dry_run:=true`` (the default) never touches Servo at all --
no service calls, no ``delta_twist_cmds`` publish -- only the always-on
``/side_tape/twist_preview``. Only pass ``dry_run:=false`` once the whole
chain (TF, observation, axis signs) has been checked in preview.
"""

from __future__ import annotations

import time

import signal
import threading

import numpy as np
import rclpy
from action_msgs.msg import GoalStatusArray
from geometry_msgs.msg import TwistStamped
from moveit_msgs.msg import AllowedCollisionEntry, PlanningScene, PlanningSceneComponents
from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene
from rcl_interfaces.srv import GetParameters
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.signals import SignalHandlerOptions
from rclpy.time import Time
from std_msgs.msg import Int8
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener
from tf2_ros import TransformException as Tf2Exception

from . import control_math, conversions, core_path  # noqa: F401
from precision_box_detector.side_tape.config import ConfigError, load_config  # noqa: E402
from precision_box_detector.side_tape.supervisor import Supervisor, TcpState  # noqa: E402
from precision_box_detector.side_tape.types import ControlOutput, Phase  # noqa: E402
from side_tape_interfaces.msg import Observation as ObservationMsg  # noqa: E402
from side_tape_interfaces.msg import SupervisorStatus as SupervisorStatusMsg  # noqa: E402
from side_tape_interfaces.srv import StartCut  # noqa: E402

# Parameter names read from the live Servo node at the start of any live
# motion (finding 7d); see docs S4b / launch/move_group.launch.py's
# ``moveit_servo`` namespace for how they are declared.
_SERVO_PARAM_NAMES = (
    "moveit_servo.robot_link_command_frame",
    "moveit_servo.command_in_type",
    "moveit_servo.incoming_command_timeout",
)

_TRAJ_STATUS_QOS = QoSProfile(
    depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


class SideTapeController(Node):
    def __init__(self):
        super().__init__("side_tape_controller")

        self.declare_parameter("config_path", core_path.DEFAULT_CONFIG)
        self.declare_parameter("dry_run", True)
        config_path = self.get_parameter("config_path").get_parameter_value().string_value
        self.dry_run = bool(self.get_parameter("dry_run").get_parameter_value().bool_value)

        try:
            self.cfg = load_config(config_path)
        except (ConfigError, OSError) as exc:
            self.get_logger().fatal(f"failed to load side-tape config {config_path!r}: {exc}")
            raise

        self.get_logger().info(
            f"side_tape_controller: config={config_path} dry_run={self.dry_run} "
            f"calibrated={self.cfg.mount.calibrated} base_frame={self.cfg.robot.base_frame} "
            f"tool_frame={self.cfg.robot.tool_frame}"
        )

        self.supervisor = Supervisor(self.cfg)
        self._lock = threading.RLock()

        # -- state filled in by subscriptions / freezing --------------------
        self._latest_obs = None  # precision_box_detector.side_tape.types.Observation
        self._last_delivered_frame_seq: int | None = None
        self._last_servo_status_value: int | None = None
        self._last_servo_status_time_s: float | None = None
        self._last_goal_statuses: list[int] = []
        self._vecs: control_math.TcpVectors | None = None
        self._servo_unpaused_by_us = False
        self._last_arm_owned = False
        self._last_servo_label = "UNKNOWN"
        self._last_tcp = TcpState()
        self._last_control = ControlOutput.zero(Phase.IDLE, "startup")
        self._prev_should_drive = False
        self._prev_terminal = True  # IDLE at startup

        # -- Servo lifecycle / ACM bookkeeping (findings 7c, 7d, 8) -----------
        self._servo_started_once = False  # start_servo restarts Servo's loop: call at most once
        self._acm_applied = False
        self._acm_original: tuple[list, list] | None = None

        # -- TF ---------------------------------------------------------------
        self.tf_buffer = Buffer(cache_time=Duration(seconds=2.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self._shutdown_executor = None
        cbg = ReentrantCallbackGroup()
        cbg_clients = ReentrantCallbackGroup()

        # -- subs ---------------------------------------------------------------
        self.create_subscription(ObservationMsg, "/side_tape/observation",
                                  self._on_observation, 1, callback_group=cbg)
        self.create_subscription(Int8, self.cfg.robot.servo_status_topic,
                                  self._on_servo_status, 10, callback_group=cbg)
        self.create_subscription(GoalStatusArray, f"{self.cfg.robot.trajectory_action}/_action/status",
                                  self._on_traj_status, _TRAJ_STATUS_QOS, callback_group=cbg)

        # -- pubs -----------------------------------------------------------
        self.pub_preview = self.create_publisher(TwistStamped, "/side_tape/twist_preview", 10)
        self.pub_twist = self.create_publisher(TwistStamped, self.cfg.robot.servo_twist_topic, 10)
        self.pub_status = self.create_publisher(SupervisorStatusMsg, "/side_tape/status", 10)

        # -- servo service clients (only ever called when dry_run is False) --
        self._servo_start_client = self.create_client(Trigger, f"{self.cfg.robot.servo_ns}/start_servo",
                                                        callback_group=cbg_clients)
        self._servo_pause_client = self.create_client(Trigger, f"{self.cfg.robot.servo_ns}/pause_servo",
                                                        callback_group=cbg_clients)
        self._servo_unpause_client = self.create_client(Trigger, f"{self.cfg.robot.servo_ns}/unpause_servo",
                                                          callback_group=cbg_clients)
        self._servo_param_client = self.create_client(
            GetParameters, f"{self.cfg.robot.servo_ns}/get_parameters", callback_group=cbg_clients)
        self._get_scene_client = self.create_client(
            GetPlanningScene, "/get_planning_scene", callback_group=cbg_clients)
        self._apply_scene_client = self.create_client(
            ApplyPlanningScene, "/apply_planning_scene", callback_group=cbg_clients)

        # -- our services -----------------------------------------------------
        self.create_service(Trigger, "/side_tape/start_align", self._on_start_align, callback_group=cbg)
        self.create_service(StartCut, "/side_tape/start_cut", self._on_start_cut, callback_group=cbg)
        self.create_service(Trigger, "/side_tape/abort", self._on_abort, callback_group=cbg)
        self.create_service(Trigger, "/side_tape/acknowledge_abort", self._on_acknowledge_abort, callback_group=cbg)
        self.create_service(Trigger, "/side_tape/resume", self._on_resume, callback_group=cbg)

        # -- timers -----------------------------------------------------------
        # Each timer in its own mutually-exclusive group: at most one control tick and
        # one status tick in flight, so blocked ticks can never occupy every executor
        # thread and starve the Servo client responses (deadlock seen in live mode).
        self.create_timer(1.0 / self.cfg.control.rate_hz, self._on_control_timer,
                          callback_group=MutuallyExclusiveCallbackGroup())
        self.create_timer(1.0 / 20.0, self._on_status_timer,
                          callback_group=MutuallyExclusiveCallbackGroup())

        # absolute clamp inputs (defence in depth, task spec)
        self._v_clamp_args = (self.cfg.control.v_cut_mps, self.cfg.control.v_lat_max_mps,
                               self.cfg.control.v_retract_mps)

    # -- time ---------------------------------------------------------------

    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    # -- subscription callbacks ----------------------------------------------

    def _on_observation(self, msg: ObservationMsg) -> None:
        try:
            obs = conversions.msg_to_observation(msg)
        except Exception as exc:  # noqa: BLE001 - never let a bad message kill the node
            self.get_logger().warn(f"bad Observation message: {exc}")
            return
        with self._lock:
            self._latest_obs = obs

    def _on_servo_status(self, msg: Int8) -> None:
        with self._lock:
            self._last_servo_status_value = int(msg.data)
            self._last_servo_status_time_s = self._now_s()

    def _on_traj_status(self, msg: GoalStatusArray) -> None:
        with self._lock:
            self._last_goal_statuses = [int(s.status) for s in msg.status_list]

    # -- TF helpers -----------------------------------------------------------

    def _lookup_tf_now(self):
        """Latest available base_frame<-tool_frame transform, or None. Non-blocking."""
        try:
            return self.tf_buffer.lookup_transform(
                self.cfg.robot.base_frame, self.cfg.robot.tool_frame, Time())
        except Tf2Exception:
            return None

    def _freeze_at_current_pose(self) -> control_math.TcpVectors | None:
        tf = self._lookup_tf_now()
        if tf is None:
            return None
        q = tf.transform.rotation
        p = tf.transform.translation
        R = control_math.quat_to_rotation_matrix(q.x, q.y, q.z, q.w)
        cut, lat, depth = control_math.freeze_axes(
            R, self.cfg.robot.cut_axis_tool, self.cfg.robot.lateral_axis_tool, self.cfg.robot.depth_axis_tool)
        p0 = np.array([p.x, p.y, p.z], dtype=np.float64)
        return control_math.TcpVectors(cut=cut, lateral=lat, depth=depth, p0_m=p0)

    # -- control step (shared by the timer and service handlers) --------------

    def _control_step(self, now_s: float) -> ControlOutput:
        """One supervisor tick: TF -> arm_owned/tcp -> tick -> twist -> publish.

        Called by the periodic control timer, and also once by the
        start_align/start_cut handlers right after any Servo service call, so
        `arm_owned` reflects the just-changed Servo state before the
        supervisor's own entry checks run (avoids a start-of-align race where
        the last tick's cached arm_owned still predates our own unpause).
        """
        with self._lock:
            tf = self._lookup_tf_now()
            tf_fresh = False
            p_base_m = None
            if tf is not None:
                stamp_s = tf.header.stamp.sec + tf.header.stamp.nanosec * 1e-9
                age = now_s - stamp_s
                tf_fresh = -0.05 <= age < 0.2  # small clock skew allowed; ~2 missed TF samples
                p_base_m = np.array([tf.transform.translation.x, tf.transform.translation.y,
                                      tf.transform.translation.z], dtype=np.float64)

            trajectory_active = control_math.trajectory_goal_active(self._last_goal_statuses)

            servo_status_age_s = None
            if self._last_servo_status_time_s is not None:
                servo_status_age_s = now_s - self._last_servo_status_time_s
            servo_label = "UNKNOWN"
            if servo_status_age_s is not None and 0.0 <= servo_status_age_s <= 1.0:
                servo_label = control_math.map_servo_status_int(self._last_servo_status_value)
            servo_status = control_math.servo_status_for_supervisor(servo_label, self.dry_run)

            arm_owned = control_math.arm_owned(
                tf_fresh, trajectory_active, self.dry_run, self._servo_unpaused_by_us, servo_status_age_s)

            if self._vecs is not None and p_base_m is not None:
                along, lat, depth = control_math.tcp_state_mm(p_base_m, self._vecs)
                tcp = TcpState(along_mm=along, lat_dev_mm=lat, depth_dev_mm=depth)
            else:
                tcp = TcpState()

            obs_to_feed = None
            if self._latest_obs is not None and self._latest_obs.frame_seq != self._last_delivered_frame_seq:
                obs_to_feed = self._latest_obs
                self._last_delivered_frame_seq = self._latest_obs.frame_seq

            control = self.supervisor.tick(now_s, obs_to_feed, tcp, servo_status, arm_owned)

            if self._vecs is not None:
                v_base = control_math.compose_linear_velocity(control, self._vecs)
            else:
                v_base = np.zeros(3)
            v_safe, nan_detected = control_math.sanitize_linear_velocity(v_base, *self._v_clamp_args)
            if nan_detected:
                self.get_logger().error("NaN/inf in commanded velocity: zeroing and aborting")
                self.supervisor.abort(now_s, "nan_velocity")
                control = ControlOutput.zero(self.supervisor.phase, "nan_velocity")

            self._publish_twists(now_s, v_safe, control, arm_owned)

            self._last_arm_owned = arm_owned
            self._last_servo_label = servo_label
            self._last_tcp = tcp
            self._last_control = control
            return control

    def _publish_twists(self, now_s: float, v_safe: np.ndarray, control: ControlOutput,
                          arm_owned: bool) -> None:
        stamp = self.get_clock().now().to_msg()
        preview = TwistStamped()
        preview.header.stamp = stamp
        preview.header.frame_id = self.cfg.robot.base_frame
        preview.twist.linear.x, preview.twist.linear.y, preview.twist.linear.z = (float(v) for v in v_safe)
        self.pub_preview.publish(preview)

        # Computed regardless of arm_owned, so _prev_terminal/_prev_should_drive
        # always reflect the real phase -- an ownership gap must not corrupt
        # the one-shot "just entered terminal" edge detection used below.
        is_abort_stop = "operator_required" in control.reasons
        should_drive = control.phase in (Phase.ALIGN, Phase.CUT, Phase.HOLD) or (
            control.phase == Phase.ABORT and not is_abort_stop)
        entering_terminal = control.phase in (Phase.DONE, Phase.IDLE) or (
            control.phase == Phase.ABORT and is_abort_stop)

        if not arm_owned:
            # Finding 7a: when we don't own the arm, publish NOTHING to Servo
            # -- not even a zero twist -- and pause Servo (async: this runs
            # under self._lock inside the control tick) if we had unpaused it.
            if not self.dry_run and self._servo_unpaused_by_us:
                self._pause_servo_async("arm_ownership_lost")
                self._servo_unpaused_by_us = False
            self._prev_should_drive = should_drive
            self._prev_terminal = entering_terminal
            return

        if self.dry_run:
            self._prev_should_drive = should_drive
            self._prev_terminal = entering_terminal
            return

        if should_drive:
            cmd = TwistStamped()
            cmd.header.stamp = stamp
            cmd.header.frame_id = self.cfg.robot.base_frame
            cmd.twist.linear.x, cmd.twist.linear.y, cmd.twist.linear.z = (float(v) for v in v_safe)
            self.pub_twist.publish(cmd)
        elif entering_terminal and not self._prev_terminal:
            # freshly IDLE/DONE/abort_stop: one zero twist, then pause Servo
            # and restore the ACM entries we (may have) widened (finding 8).
            zero = TwistStamped()
            zero.header.stamp = stamp
            zero.header.frame_id = self.cfg.robot.base_frame
            self.pub_twist.publish(zero)
            # Non-blocking: this runs under self._lock inside the control tick.
            self._pause_servo_async("phase exit")
            self._servo_unpaused_by_us = False
            self._restore_carton_acm_async("phase exit")

        self._prev_should_drive = should_drive
        self._prev_terminal = entering_terminal

    # -- timers ---------------------------------------------------------------

    def _on_control_timer(self) -> None:
        self._control_step(self._now_s())

    def _on_status_timer(self) -> None:
        with self._lock:
            control = self._last_control
            tcp = self._last_tcp
            obs = self._latest_obs
            arm_owned = self._last_arm_owned
            servo_label = self._last_servo_label
        msg = SupervisorStatusMsg()
        msg.stamp = self.get_clock().now().to_msg()
        msg.phase = control.phase.value
        msg.reasons = [str(r) for r in control.reasons]
        msg.dry_run = self.dry_run
        msg.arm_owned = bool(arm_owned)
        msg.calibrated = bool(self.cfg.mount.calibrated)
        msg.servo_status = servo_label
        msg.has_e_lat = obs is not None and obs.e_lat_mm is not None
        msg.e_lat_mm = float(obs.e_lat_mm) if (obs is not None and obs.e_lat_mm is not None) else float("nan")
        msg.along_mm = tcp.along_mm
        msg.lat_dev_mm = tcp.lat_dev_mm
        msg.depth_dev_mm = tcp.depth_dev_mm
        msg.v_cut_mps = control.v_cut_mps
        msg.v_lat_mps = control.v_lat_mps
        msg.v_depth_mps = control.v_depth_mps
        self.pub_status.publish(msg)

    # -- Servo service helper ---------------------------------------------------

    def _wait_future(self, future, timeout_sec: float):
        # Do NOT use rclpy.spin_until_future_complete(self, ...) here: in Humble it
        # moves this node onto the global executor and then removes it, detaching
        # it from the MultiThreadedExecutor (the control timer would stop). The
        # clients live in a reentrant group, so another executor thread completes
        # the future while this one waits.
        deadline = time.monotonic() + timeout_sec
        while not future.done() and time.monotonic() < deadline:
            if self._shutdown_executor is not None:  # main spin loop has exited
                self._shutdown_executor.spin_once(timeout_sec=0.01)
            else:
                time.sleep(0.01)
        return future.done()

    def _call_trigger(self, client, name: str, timeout_sec: float = 2.0) -> tuple[bool, str]:
        if not client.wait_for_service(timeout_sec=1.0):
            return False, f"{name}_unavailable"
        future = client.call_async(Trigger.Request())
        if not self._wait_future(future, timeout_sec):
            return False, f"{name}_timeout"
        res = future.result()
        if res is None:
            return False, f"{name}_no_response"
        return bool(res.success), str(res.message)

    def _pause_servo_async(self, why: str) -> None:
        if not self._servo_pause_client.service_is_ready():
            self.get_logger().warn(f"pause_servo unavailable ({why})")
            return
        future = self._servo_pause_client.call_async(Trigger.Request())

        def _done(f):
            res = f.result()
            if res is None or not res.success:
                self.get_logger().warn(f"pause_servo failed ({why}): {getattr(res, 'message', 'no response')}")
        future.add_done_callback(_done)

    # -- Servo parameter contract (finding 7d) -----------------------------

    def _check_servo_params(self) -> tuple[bool, str]:
        """Live mode only: validate the running Servo node's parameters
        against ``RobotCfg`` before the first live motion. Refuses if the
        parameter service is unavailable, per the task spec."""
        if not self._servo_param_client.wait_for_service(timeout_sec=1.0):
            return False, "servo_params_unavailable:get_parameters_service"
        req = GetParameters.Request()
        req.names = list(_SERVO_PARAM_NAMES)
        future = self._servo_param_client.call_async(req)
        if not self._wait_future(future, timeout_sec=2.0):
            return False, "servo_params_unavailable:get_parameters_timeout"
        res = future.result()
        if res is None:
            return False, "servo_params_unavailable:get_parameters_no_response"
        parsed = control_math.parse_get_parameters_result(_SERVO_PARAM_NAMES, res.values)
        ok, msg = control_math.servo_params_ok(
            parsed.get(_SERVO_PARAM_NAMES[0]), parsed.get(_SERVO_PARAM_NAMES[1]),
            parsed.get(_SERVO_PARAM_NAMES[2]), self.cfg.robot.servo_command_frame,
            self.cfg.robot.servo_command_in_type, self.cfg.robot.servo_max_incoming_timeout_s)
        if not ok:
            return False, msg
        warn = control_math.servo_incoming_timeout_warning(parsed.get(_SERVO_PARAM_NAMES[2]))
        if warn:
            self.get_logger().warn(warn)
        return True, ""

    # -- ACM: allow blade<->carton collisions during live Servo motion (finding 8) --

    def _apply_carton_acm(self) -> tuple[bool, str]:
        """Fetch the current full ACM, widen it to allow
        ``robot.carton_allowed_links`` vs. ``robot.carton_collision_id``, and
        apply the new full matrix (Humble replaces the whole ACM on a diff,
        so a partial diff would silently drop every other entry). Remembers
        the original for :meth:`_restore_carton_acm_async`. Idempotent."""
        if self._acm_applied:
            return True, ""
        if not self._get_scene_client.wait_for_service(timeout_sec=1.0):
            return False, "get_planning_scene_unavailable"
        req = GetPlanningScene.Request()
        req.components.components = PlanningSceneComponents.ALLOWED_COLLISION_MATRIX
        future = self._get_scene_client.call_async(req)
        if not self._wait_future(future, timeout_sec=2.0) or future.result() is None:
            return False, "get_planning_scene_failed"
        acm = future.result().scene.allowed_collision_matrix
        names = list(acm.entry_names)
        enabled = [list(row.enabled) for row in acm.entry_values]
        self._acm_original = (names, enabled)
        if self.cfg.robot.carton_collision_id not in names:
            # Normal case on a real move_group: a freshly-added CollisionObject gets
            # no ACM row until something explicitly sets one (verified against a
            # real planning scene -- the fake-Servo harness pre-seeded the carton
            # into its fake ACM's entry_names from the start, which hid this).
            # acm_add_allowed_pairs() adds a new all-False row/column for any name
            # not already present, so this is not an error: fall through and let it
            # create the carton's row along with the allowed pairs.
            self.get_logger().info(
                f"carton '{self.cfg.robot.carton_collision_id}' not yet in the planning scene ACM "
                "(no explicit entry for it yet); adding it with the allowed pairs")
        new_names, new_enabled = control_math.acm_add_allowed_pairs(
            names, enabled, self.cfg.robot.carton_collision_id, self.cfg.robot.carton_allowed_links)
        ok, msg = self._apply_acm_blocking(new_names, new_enabled)
        if not ok:
            self._acm_original = None
            return False, msg
        self._acm_applied = True
        return True, ""

    def _apply_acm_blocking(self, names: list, enabled: list) -> tuple[bool, str]:
        if not self._apply_scene_client.wait_for_service(timeout_sec=1.0):
            return False, "apply_planning_scene_unavailable"
        scene = PlanningScene()
        scene.is_diff = True
        scene.allowed_collision_matrix.entry_names = list(names)
        scene.allowed_collision_matrix.entry_values = [
            AllowedCollisionEntry(enabled=[bool(v) for v in row]) for row in enabled
        ]
        req = ApplyPlanningScene.Request()
        req.scene = scene
        future = self._apply_scene_client.call_async(req)
        if not self._wait_future(future, timeout_sec=2.0):
            return False, "apply_planning_scene_timeout"
        res = future.result()
        if res is None or not res.success:
            return False, "apply_planning_scene_failed"
        return True, ""

    def _restore_carton_acm_async(self, why: str) -> None:
        """Best-effort, non-blocking restore of the ACM entries fetched by
        :meth:`_apply_carton_acm` -- called from the control-tick context
        (DONE/IDLE/abort_stop), so it must not block the 100 Hz loop."""
        if not self._acm_applied or self._acm_original is None:
            self._acm_applied = False
            return
        names, enabled = self._acm_original
        self._acm_applied = False
        self._acm_original = None
        if not self._apply_scene_client.service_is_ready():
            self.get_logger().warn(f"apply_planning_scene unavailable, cannot restore ACM ({why})")
            return
        scene = PlanningScene()
        scene.is_diff = True
        scene.allowed_collision_matrix.entry_names = list(names)
        scene.allowed_collision_matrix.entry_values = [
            AllowedCollisionEntry(enabled=[bool(v) for v in row]) for row in enabled
        ]
        req = ApplyPlanningScene.Request()
        req.scene = scene
        future = self._apply_scene_client.call_async(req)

        def _done(f):
            res = f.result()
            if res is None or not res.success:
                self.get_logger().error(f"failed to restore ACM ({why}): "
                                          f"{getattr(res, 'message', 'no response')}")
        future.add_done_callback(_done)

    def _prepare_live_servo(self) -> tuple[bool, str]:
        """Live-mode-only preparation shared by start_align and (in-material)
        start_cut, run AFTER the cheap, non-mutating supervisor preconditions
        already passed (finding 7b): check Servo's parameter contract, start
        Servo's loop at most once per node lifetime (finding 7c), unpause it,
        then allow the carton<->blade collision pair (finding 8). Any failure
        undoes whatever it already did and refuses. ``dry_run``: no-op."""
        if self.dry_run:
            return True, ""
        ok, msg = self._check_servo_params()
        if not ok:
            return False, f"servo_params_refused:{msg}"
        if not self._servo_started_once:
            ok, msg = self._call_trigger(self._servo_start_client, "start_servo")
            if not ok:
                return False, f"start_servo_failed:{msg}"
            self._servo_started_once = True
        ok, msg = self._call_trigger(self._servo_unpause_client, "unpause_servo")
        if not ok:
            return False, f"unpause_servo_failed:{msg}"
        with self._lock:
            self._servo_unpaused_by_us = True
        ok, msg = self._apply_carton_acm()
        if not ok:
            self._pause_servo_async("acm_apply_failed")
            with self._lock:
                self._servo_unpaused_by_us = False
            return False, f"acm_apply_failed:{msg}"
        return True, ""

    # -- services -----------------------------------------------------------

    def _on_start_align(self, request, response):
        # Finding 7b: check every cheap precondition (phase, calibrated, lock
        # streak, edge angle) BEFORE touching Servo at all. arm_owned itself
        # can't be checked here yet -- in live mode it only becomes true once
        # Servo is actually unpaused below -- so it is re-checked for real
        # once side effects have happened.
        with self._lock:
            now_s = self._now_s()
            ok, reason = self.supervisor.can_start_align(now_s)
            trajectory_active = control_math.trajectory_goal_active(self._last_goal_statuses)
        if not ok:
            response.success = False
            response.message = reason
            return response
        if trajectory_active:
            response.success = False
            response.message = "start_align_refused:trajectory_active"
            return response
        if self._freeze_at_current_pose() is None:
            response.success = False
            response.message = "start_align_refused:tf_unavailable"
            return response

        # Servo calls happen WITHOUT holding self._lock (the control tick needs it).
        ok, msg = self._prepare_live_servo()
        if not ok:
            response.success = False
            response.message = f"start_align_refused:{msg}"
            return response

        with self._lock:
            now_s = self._now_s()
            vecs = self._freeze_at_current_pose()
            if vecs is None:
                response.success = False
                response.message = "start_align_refused:tf_unavailable"
                self._undo_live_servo_prep("tf_unavailable")
                return response
            self._vecs = vecs
            # refresh arm_owned/servo status with the just-changed Servo state
            # before the supervisor's own entry checks run.
            self._control_step(now_s)
            self.supervisor.start_align(self._now_s())
            accepted = self.supervisor.phase == Phase.ALIGN
            if not accepted:
                self._undo_live_servo_prep("start_align refused")
            response.success = accepted
            response.message = "align_started" if accepted else ";".join(self.supervisor._reason)
            return response

    def _on_start_cut(self, request, response):
        with self._lock:
            phase = self.supervisor.phase
        if phase == Phase.READY:
            # air_align: Servo is already running/unpaused from start_align;
            # no further Servo side effects here.
            with self._lock:
                now_s = self._now_s()
                if self.supervisor.phase != Phase.READY:
                    response.accepted = False
                    response.message = f"start_cut_refused:wrong_phase:{self.supervisor.phase.value}"
                    return response
                vecs = self._freeze_at_current_pose()
                if vecs is None:
                    response.accepted = False
                    response.message = "start_cut_refused:tf_unavailable"
                    return response
                self._vecs = vecs
                self._control_step(now_s)
                self.supervisor.start_cut(self._now_s(), float(request.cut_length_mm))
                accepted = self.supervisor.phase == Phase.CUT
                response.accepted = accepted
                response.message = "cut_started" if accepted else ";".join(self.supervisor._reason)
                return response

        # in_material: IDLE -> CUT directly. MoveIt has already put the blade
        # in the cardboard; check the cheap preconditions before touching
        # Servo (finding 7b), same shape as _on_start_align.
        with self._lock:
            now_s = self._now_s()
            ok, reason = self.supervisor.can_start_cut_in_material(now_s, float(request.cut_length_mm))
            trajectory_active = control_math.trajectory_goal_active(self._last_goal_statuses)
        if not ok:
            response.accepted = False
            response.message = reason
            return response
        if trajectory_active:
            response.accepted = False
            response.message = "start_cut_refused:trajectory_active"
            return response
        if self._freeze_at_current_pose() is None:
            response.accepted = False
            response.message = "start_cut_refused:tf_unavailable"
            return response

        ok, msg = self._prepare_live_servo()
        if not ok:
            response.accepted = False
            response.message = f"start_cut_refused:{msg}"
            return response

        with self._lock:
            now_s = self._now_s()
            vecs = self._freeze_at_current_pose()
            if vecs is None:
                response.accepted = False
                response.message = "start_cut_refused:tf_unavailable"
                self._undo_live_servo_prep("tf_unavailable")
                return response
            self._vecs = vecs
            self._control_step(now_s)
            self.supervisor.start_cut(self._now_s(), float(request.cut_length_mm))
            accepted = self.supervisor.phase == Phase.CUT
            if not accepted:
                self._undo_live_servo_prep("start_cut refused")
            response.accepted = accepted
            response.message = "cut_started" if accepted else ";".join(self.supervisor._reason)
            return response

    def _undo_live_servo_prep(self, why: str) -> None:
        """Compensating action when a refusal happens right after
        :meth:`_prepare_live_servo` already unpaused Servo and widened the
        ACM: pause again and restore the ACM (finding 7b/8)."""
        if not self.dry_run and self._servo_unpaused_by_us:
            self._pause_servo_async(why)
            self._servo_unpaused_by_us = False
        self._restore_carton_acm_async(why)

    def _on_abort(self, request, response):
        with self._lock:
            self.supervisor.abort(self._now_s(), "operator_abort")
            response.success = True
            response.message = f"phase={self.supervisor.phase.value}"
            return response

    def _on_acknowledge_abort(self, request, response):
        with self._lock:
            self.supervisor.acknowledge_abort(self._now_s())
            response.success = self.supervisor.phase == Phase.IDLE
            response.message = f"phase={self.supervisor.phase.value}"
            return response

    def _on_resume(self, request, response):
        with self._lock:
            before = self.supervisor.phase
            self.supervisor.resume(self._now_s())
            response.success = before == Phase.HOLD and self.supervisor.phase != Phase.HOLD
            response.message = f"phase={self.supervisor.phase.value}"
            return response

    # -- shutdown -------------------------------------------------------------

    def shutdown_safe(self, executor=None) -> None:
        """Best-effort: zero twist, pause Servo, restore the ACM (finding 8).
        Called on normal exit or exception."""
        self._shutdown_executor = executor
        try:
            stamp = self.get_clock().now().to_msg()
            zero = TwistStamped()
            zero.header.stamp = stamp
            zero.header.frame_id = self.cfg.robot.base_frame
            self.pub_preview.publish(zero)
            if not self.dry_run:
                self.pub_twist.publish(zero)
                if self._servo_pause_client.service_is_ready():
                    self._call_trigger(self._servo_pause_client, "pause_servo", timeout_sec=1.0)
                if self._acm_applied and self._acm_original is not None:
                    names, enabled = self._acm_original
                    ok, msg = self._apply_acm_blocking(names, enabled)
                    if not ok:
                        self.get_logger().error(f"shutdown: failed to restore ACM: {msg}")
                    self._acm_applied = False
                    self._acm_original = None
        except Exception as exc:  # noqa: BLE001 - shutdown must not raise
            self.get_logger().error(f"shutdown_safe failed: {exc}")


def main(args=None):
    # Disable rclpy's own SIGINT/SIGTERM handler (which shuts the context down
    # asynchronously the instant the signal arrives): otherwise shutdown_safe's
    # "zero twist, pause_servo" publish/service calls below can lose the race
    # against context teardown and silently fail right when they matter most.
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = SideTapeController()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    stop = threading.Event()

    def _request_stop(signum, frame):  # noqa: ARG001
        stop.set()

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    try:
        while rclpy.ok() and not stop.is_set():
            executor.spin_once(timeout_sec=0.2)
    except Exception:
        node.get_logger().error("side_tape_controller crashed", exc_info=True)
    finally:
        node.shutdown_safe(executor)  # context is still valid here
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
