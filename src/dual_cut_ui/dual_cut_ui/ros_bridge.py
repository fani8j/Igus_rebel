"""rclpy implementation of bridge.Bridge.  The node is spun by a MultiThreadedExecutor on its own thread; every
callback group is reentrant and nothing here spins or blocks inside a callback.  Blocking methods wait on an Event set
by the future's done-callback, with a timeout, and are called from worker threads only.

The UI never publishes any motion command: it calls the coordinator's Trigger services and parameters, and cancels
trajectory goals through the action's ``_action/cancel_goal`` service.  The detector is reached through the DetectBox
action (compal_box_msgs/action/DetectBox, imported lazily: it only exists in the box-seam-detection install).
"""
import collections
import threading
import time
from typing import Callable, Deque, Dict, List, Optional, Tuple

from action_msgs.srv import CancelGoal
from controller_manager_msgs.srv import ListControllers, ListHardwareComponents
from dual_cut_msgs.msg import CoordinatorStatus
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import GetParameters, SetParameters
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from std_srvs.srv import Trigger

from . import bridge as B
from .domains import localhost_only_env

POLL_PERIOD_S = 1.0
HW_ACTIVE_ID = 3          # lifecycle_msgs/State PRIMARY_STATE_ACTIVE
TRIGGERS = ("plan", "execute", "stop", "rearm")
GOAL_STATUS_CANCELED = 5       # action_msgs/GoalStatus
DETECT_RESULT_FIELDS = ("success", "status", "reason", "reasons", "cut_permission", "snapshot_id", "snapshot_path",
                        "capture_id", "seam_status", "attempts", "elapsed_s")
DETECT_SERVER_WAIT_S = 5.0
DETECT_ACCEPT_WAIT_S = 10.0
DETECT_CANCEL_WAIT_S = 5.0
STATUS_FIELDS = ("state", "phase", "plan_seq", "plan_retained", "phase_validated", "min_cross_distance_m",
                 "closest_link_a", "closest_link_b", "joint_states_age_s", "stop_reason", "stop_latched",
                 "barrier_ok", "barrier_message")


def _decode(v: ParameterValue):
    t = v.type
    if t == ParameterType.PARAMETER_BOOL:
        return bool(v.bool_value)
    if t == ParameterType.PARAMETER_INTEGER:
        return int(v.integer_value)
    if t == ParameterType.PARAMETER_DOUBLE:
        return float(v.double_value)
    if t == ParameterType.PARAMETER_STRING:
        return str(v.string_value)
    return None


def _plain(v):
    """A ROS message field as a plain python value (arrays -> list)."""
    if isinstance(v, (str, bool, int, float)):
        return v
    try:
        return [str(x) for x in v]
    except TypeError:
        return v


class RosBridge(B.Bridge):
    def __init__(self, node, coordinator: str, controllers: Dict[str, str],
                 controller_manager: str = "/controller_manager"):
        self.node = node
        self.ns = "/" + coordinator.strip("/")
        self.controllers = {arm: "/" + name.strip("/") for arm, name in controllers.items()}
        self.hw_names = {arm: f"arm_{arm}_hw" for arm in B.ARMS}
        self._lock = threading.Lock()
        self._status: Optional[Tuple[dict, float]] = None
        self._plan_report: Optional[Tuple[str, float]] = None
        self._exec_reports: Deque[Tuple[str, float]] = collections.deque(maxlen=20)
        self._listeners: List[Callable[[dict, float], None]] = []
        self._ctrl: Optional[Tuple[Dict[str, bool], float]] = None
        self._hw: Optional[Tuple[Dict[str, bool], float]] = None
        self._gates: Optional[Tuple[dict, float]] = None
        self._allow_blocked: Optional[Tuple[Optional[bool], float]] = None
        self._pending: Dict[str, float] = {}
        self._detect_type = None                          # compal_box_msgs.action.DetectBox, imported lazily
        self._detect_import_error: Optional[str] = None
        self._detect_clients: Dict[str, object] = {}
        group = self.group = ReentrantCallbackGroup()
        self.cameras = CameraSource(node)                 # default: cameras in the node's own domain
        reliable = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        latched = lambda depth: QoSProfile(depth=depth, reliability=ReliabilityPolicy.RELIABLE,  # noqa: E731
                                           durability=DurabilityPolicy.TRANSIENT_LOCAL)
        node.create_subscription(CoordinatorStatus, f"{self.ns}/status", self._on_status, reliable, callback_group=group)
        node.create_subscription(String, f"{self.ns}/plan_report", self._on_plan_report, latched(1), callback_group=group)
        node.create_subscription(String, f"{self.ns}/exec_report", self._on_exec_report, latched(20), callback_group=group)
        self._triggers = {n: node.create_client(Trigger, f"{self.ns}/{n}", callback_group=group) for n in TRIGGERS}
        self._set_params = node.create_client(SetParameters, f"{self.ns}/set_parameters", callback_group=group)
        self._get_params = node.create_client(GetParameters, f"{self.ns}/get_parameters", callback_group=group)
        cm = "/" + controller_manager.strip("/")
        self._list_controllers = node.create_client(ListControllers, f"{cm}/list_controllers", callback_group=group)
        self._list_hw = node.create_client(ListHardwareComponents, f"{cm}/list_hardware_components", callback_group=group)
        self._cancel = {arm: node.create_client(CancelGoal, f"{name}/follow_joint_trajectory/_action/cancel_goal",
                                                callback_group=group) for arm, name in self.controllers.items()}
        node.create_timer(POLL_PERIOD_S, self._poll, callback_group=group)

    # ---- subscriptions --------------------------------------------------------------------------------------------
    def _on_status(self, msg: CoordinatorStatus) -> None:
        fields = {n: getattr(msg, n) for n in STATUS_FIELDS}
        for k in ("state", "plan_seq"):
            fields[k] = int(fields[k])
        now = time.monotonic()
        with self._lock:
            self._status = (fields, now)
            listeners = list(self._listeners)
        for fn in listeners:
            try:
                fn(fields, now)
            except Exception as e:  # noqa: BLE001
                self.node.get_logger().error(f"status listener failed: {e}")

    def _on_plan_report(self, msg: String) -> None:
        with self._lock:
            self._plan_report = (str(msg.data), time.monotonic())

    def _on_exec_report(self, msg: String) -> None:
        with self._lock:
            self._exec_reports.append((str(msg.data), time.monotonic()))

    def status(self):
        with self._lock:
            return self._status

    def add_status_listener(self, fn) -> None:
        with self._lock:
            self._listeners.append(fn)

    def plan_report(self):
        with self._lock:
            return self._plan_report

    def exec_reports(self):
        with self._lock:
            return list(self._exec_reports)

    def health(self):
        with self._lock:
            if self._ctrl is None or self._hw is None:
                return None
            return ({"controllers": dict(self._ctrl[0]), "hardware": dict(self._hw[0])}, min(self._ctrl[1], self._hw[1]))

    def gates(self):
        with self._lock:
            return self._gates

    def allow_blocked(self):
        with self._lock:
            return self._allow_blocked

    # ---- polling (timer callback: only call_async, never wait) ---------------------------------------------------------
    def _begin(self, key: str, client) -> bool:
        now = time.monotonic()
        with self._lock:
            if now - self._pending.get(key, -1e9) < 5.0:
                return False
            if not client.service_is_ready():
                return False
            self._pending[key] = now
        return True

    def _end(self, key: str) -> None:
        with self._lock:
            self._pending.pop(key, None)

    def _poll(self) -> None:
        self._poll_gates()
        self._poll_allow_blocked()
        if self._begin("ctrl", self._list_controllers):
            fut = self._list_controllers.call_async(ListControllers.Request())
            fut.add_done_callback(self._on_controllers)
        if self._begin("hw", self._list_hw):
            fut = self._list_hw.call_async(ListHardwareComponents.Request())
            fut.add_done_callback(self._on_hw)

    def _on_controllers(self, fut) -> None:
        self._end("ctrl")
        try:
            r = fut.result()
            states = {c.name: c.state for c in r.controller}
            val = {arm: states.get(name.strip("/")) == "active" for arm, name in self.controllers.items()}
            with self._lock:
                self._ctrl = (val, time.monotonic())
        except Exception as e:  # noqa: BLE001
            self.node.get_logger().warn(f"list_controllers: {e}", throttle_duration_sec=10.0)

    def _on_hw(self, fut) -> None:
        self._end("hw")
        try:
            r = fut.result()
            states = {c.name: c.state.id for c in r.component}
            val = {arm: states.get(name) == HW_ACTIVE_ID for arm, name in self.hw_names.items()}
            with self._lock:
                self._hw = (val, time.monotonic())
        except Exception as e:  # noqa: BLE001
            self.node.get_logger().warn(f"list_hardware_components: {e}", throttle_duration_sec=10.0)

    def _poll_gates(self) -> None:
        if not self._begin("gates", self._get_params):
            return
        names = list(B.GATE_PARAMS + B.DISPLAY_PARAMS)
        req = GetParameters.Request()
        req.names = names
        fut = self._get_params.call_async(req)

        def done(f):
            self._end("gates")
            try:
                r = f.result()
                vals = {n: _decode(v) for n, v in zip(names, r.values)}
                with self._lock:
                    self._gates = (vals, time.monotonic())
            except Exception as e:  # noqa: BLE001
                self.node.get_logger().warn(f"get_parameters: {e}", throttle_duration_sec=10.0)
        fut.add_done_callback(done)

    def _poll_allow_blocked(self) -> None:
        """Its own get_parameters request: a coordinator without this parameter answers with an empty / short reply (or an
        error) and that must never disturb the gate read-back above."""
        if not self._begin("allow_blocked", self._get_params):
            return
        req = GetParameters.Request()
        req.names = [B.ALLOW_BLOCKED_PARAM]
        fut = self._get_params.call_async(req)

        def done(f):
            self._end("allow_blocked")
            try:
                r = f.result()
                v = _decode(r.values[0]) if len(r.values) == 1 else None
                with self._lock:
                    self._allow_blocked = (v if isinstance(v, bool) else None, time.monotonic())
            except Exception as e:  # noqa: BLE001
                self.node.get_logger().warn(f"get_parameters {B.ALLOW_BLOCKED_PARAM}: {e}", throttle_duration_sec=30.0)
        fut.add_done_callback(done)

    # ---- blocking calls (worker threads) ---------------------------------------------------------------------------------
    def _call(self, client, request, timeout_s: float, require_ready: bool = True):
        """(response, error).  Stop-type calls pass require_ready=False: the request is sent even if discovery is late."""
        if require_ready and not client.service_is_ready() and not client.wait_for_service(timeout_sec=min(1.0, timeout_s)):
            return None, "service not available"
        done = threading.Event()
        fut = client.call_async(request)
        fut.add_done_callback(lambda _f: done.set())
        if not done.wait(timeout_s):
            try:
                client.remove_pending_request(fut)
            except Exception:  # noqa: BLE001
                pass
            return None, f"no answer within {timeout_s:.1f} s (the request was sent)"
        try:
            return fut.result(), ""
        except Exception as e:  # noqa: BLE001
            return None, f"service call failed: {e}"

    def trigger(self, name: str, timeout_s: float) -> Tuple[bool, str]:
        if name not in self._triggers:
            raise ValueError(f"unknown coordinator service '{name}'")
        resp, err = self._call(self._triggers[name], Trigger.Request(), timeout_s, require_ready=(name != "stop"))
        if resp is None:
            return False, f"{name}: {err}"
        return bool(resp.success), str(resp.message)

    def set_gates(self, enabled: bool, timeout_s: float) -> Tuple[bool, str]:
        req = SetParameters.Request()
        for n in B.gate_params_for(enabled):          # ON: only ENABLE_PARAMS; OFF: all of GATE_PARAMS
            req.parameters.append(Parameter(name=n, value=ParameterValue(type=ParameterType.PARAMETER_BOOL,
                                                                          bool_value=bool(enabled))))
        resp, err = self._call(self._set_params, req, timeout_s, require_ready=False)
        if resp is None:
            return False, f"execution gates: {err}"
        bad = [f"{p.name}: {r.reason}" for p, r in zip(req.parameters, resp.results) if not r.successful]
        self._poll_gates()                          # refresh the read-back now rather than at the next 1 Hz tick
        if bad or len(resp.results) != len(req.parameters):
            return False, "execution gates not set: " + "; ".join(bad or ["short answer"])
        return True, f"execution gates set {'true' if enabled else 'false'} ({', '.join(B.gate_params_for(enabled))})"

    def set_schedule_mode(self, mode: str, timeout_s: float) -> Tuple[bool, str]:
        if mode not in B.SCHEDULE_MODES:
            return False, f"schedule_mode must be one of {', '.join(B.SCHEDULE_MODES)}, got {mode!r}"
        req = SetParameters.Request()
        req.parameters.append(Parameter(name="schedule_mode", value=ParameterValue(type=ParameterType.PARAMETER_STRING,
                                                                                    string_value=mode)))
        resp, err = self._call(self._set_params, req, timeout_s)
        if resp is None:
            return False, f"schedule_mode: {err}"
        if len(resp.results) != 1 or not resp.results[0].successful:
            return False, "schedule_mode not set: " + (resp.results[0].reason if resp.results else "short answer")
        self._poll_gates()                          # the page shows schedule_mode from this read-back
        return True, f"schedule_mode set {mode}"

    def set_allow_blocked(self, enabled: bool, timeout_s: float) -> Tuple[bool, str]:
        req = SetParameters.Request()
        req.parameters.append(Parameter(name=B.ALLOW_BLOCKED_PARAM, value=ParameterValue(type=ParameterType.PARAMETER_BOOL,
                                                                                         bool_value=bool(enabled))))
        resp, err = self._call(self._set_params, req, timeout_s, require_ready=bool(enabled))      # an OFF write is sent even if discovery is late
        if resp is None:
            return False, f"{B.ALLOW_BLOCKED_PARAM}: {err}"
        if len(resp.results) != 1 or not resp.results[0].successful:
            return False, f"{B.ALLOW_BLOCKED_PARAM} not set: " + (resp.results[0].reason if resp.results else "short answer")
        self._poll_allow_blocked()                  # refresh the read-back now rather than at the next 1 Hz tick
        return True, f"{B.ALLOW_BLOCKED_PARAM} set {'true' if enabled else 'false'}"

    def cancel_all(self, arm: str, timeout_s: float) -> Tuple[bool, str]:
        client = self._cancel[arm]
        resp, err = self._call(client, CancelGoal.Request(), timeout_s, require_ready=False)   # zero id + zero stamp = all
        if resp is None:
            return False, f"cancel {arm}: {err}"
        ok, msg = B.interpret_cancel(int(resp.return_code), len(resp.goals_canceling))
        return ok, f"cancel {arm}: {msg}"

    # ---- detector action (worker thread) ------------------------------------------------------------------------------------
    def detect_import_error(self) -> Optional[str]:
        """None if compal_box_msgs.action.DetectBox can be imported (cached), else the operator-facing reason."""
        with self._lock:
            if self._detect_type is not None:
                return None
            if self._detect_import_error is not None:
                return self._detect_import_error
        try:
            from compal_box_msgs.action import DetectBox
        except Exception as e:  # noqa: BLE001 - ImportError, or a broken half-sourced install
            err = (f"the DetectBox action interface is not available ({e}); start the UI after sourcing the box-seam-detection "
                   f"install:  {B.DETECT_SOURCE_HINT}")
            with self._lock:
                self._detect_import_error = err
            return err
        with self._lock:
            self._detect_type = DetectBox
        return None

    def _detect_client(self, action: str):
        with self._lock:
            client = self._detect_clients.get(action)
        if client is None:
            client = ActionClient(self.node, self._detect_type, action, callback_group=self.group)
            with self._lock:
                client = self._detect_clients.setdefault(action, client)
        return client

    def detect(self, action: str, goal: dict, on_feedback, cancel: threading.Event, timeout_s: float):
        err = self.detect_import_error()
        if err:
            return False, {"error": err}
        client = self._detect_client(action)
        t_end = time.monotonic() + max(1.0, float(timeout_s))
        server_deadline = time.monotonic() + DETECT_SERVER_WAIT_S
        while not client.server_is_ready():
            if cancel.is_set():
                return False, {"cancelled": True}
            if time.monotonic() >= server_deadline:
                return False, {"error": "detector action server not available (is box_seam_detector running?)"}
            cancel.wait(0.1)

        def feedback(msg):
            try:
                fb = msg.feedback
                on_feedback(str(fb.stage), int(fb.attempt), str(fb.message))
            except Exception as e:  # noqa: BLE001 - a feedback problem must not break the goal
                self.node.get_logger().warn(f"detect feedback: {e}", throttle_duration_sec=10.0)
        msg = self._detect_type.Goal()
        for k, v in goal.items():
            setattr(msg, k, v)
        accepted = threading.Event()
        fut = client.send_goal_async(msg, feedback_callback=feedback)
        fut.add_done_callback(lambda _f: accepted.set())
        end = min(t_end, time.monotonic() + DETECT_ACCEPT_WAIT_S)
        while not accepted.wait(0.05):
            if cancel.is_set():
                fut.add_done_callback(lambda f: self._cancel_late(f))
                return False, {"cancelled": True}
            if time.monotonic() >= end:
                fut.add_done_callback(lambda f: self._cancel_late(f))      # cancel the goal if it is accepted after all
                return False, {"error": "detector did not answer the goal (no accept / reject in time)"}
        try:
            handle = fut.result()
        except Exception as e:  # noqa: BLE001
            return False, {"error": f"detector goal failed: {e}"}
        if handle is None or not handle.accepted:
            return False, {"error": "detector busy / goal rejected"}
        done = threading.Event()
        rfut = handle.get_result_async()
        rfut.add_done_callback(lambda _f: done.set())
        cancel_deadline = None
        timed_out = False
        while not done.wait(0.05):
            now = time.monotonic()
            if cancel_deadline is None and (cancel.is_set() or now >= t_end):
                timed_out = not cancel.is_set()
                cancel_deadline = now + DETECT_CANCEL_WAIT_S
                if timed_out:
                    self.node.get_logger().warn(f"detect: no result within {timeout_s:.0f} s, cancelling the goal")
                handle.cancel_goal_async()
            elif cancel_deadline is not None and now >= cancel_deadline:
                if timed_out:
                    return False, {"error": f"no result within {timeout_s:.0f} s; the goal was cancelled"}
                return False, {"cancelled": True, "error": "cancel requested; the detector did not confirm in time"}
        try:
            res = rfut.result()
        except Exception as e:  # noqa: BLE001
            return False, {"error": f"detector result failed: {e}"}
        info = {k: _plain(getattr(res.result, k)) for k in DETECT_RESULT_FIELDS}
        if timed_out and not info["success"]:
            info["error"] = f"no result within {timeout_s:.0f} s; the goal was cancelled"
        elif res.status == GOAL_STATUS_CANCELED or (cancel.is_set() and not info["success"]):
            info["cancelled"] = True
        return bool(info["success"]), info

    def _cancel_late(self, fut) -> None:
        try:
            h = fut.result()
            if h is not None and h.accepted:
                h.cancel_goal_async()
        except Exception:  # noqa: BLE001
            pass

    # ---- cameras (delegated: they may live in another ROS domain, see CameraSource / CameraContext) -------------------------
    def subscribe_camera(self, name: str, topic: str, callback: Callable[[bytes], None]) -> None:
        self.cameras.subscribe_camera(name, topic, callback)

    def unsubscribe_camera(self, name: str) -> None:
        self.cameras.unsubscribe_camera(name)


class CameraSource:
    """Lazy CompressedImage subscriptions on a node.  The ONLY thing that ever runs on the camera node: no services, no
    publishers (beyond what rclpy itself creates for a node), no clients, no parameters."""

    def __init__(self, node):
        self.node = node
        self._lock = threading.Lock()
        self._subs: Dict[str, object] = {}
        self.group = ReentrantCallbackGroup()

    def subscribe_camera(self, name: str, topic: str, callback: Callable[[bytes], None]) -> None:
        with self._lock:
            if name in self._subs:
                return
        sub = self.node.create_subscription(CompressedImage, topic, lambda msg: callback(bytes(msg.data)),
                                            qos_profile_sensor_data, callback_group=self.group)
        with self._lock:
            self._subs[name] = sub

    def unsubscribe_camera(self, name: str) -> None:
        with self._lock:
            sub = self._subs.pop(name, None)
        if sub is not None:
            self.node.destroy_subscription(sub)


class CameraContext:
    """A second rclpy context in ``domain_id`` with its own node ``dual_cut_ui_cameras`` and executor thread, used ONLY to
    read camera topics.  The node has no parameter services and no rosout publisher.  ``localhost_only`` (0/1; None =
    inherit) sets ROS_LOCALHOST_ONLY for this context only, e.g. when the cell runs with 0 and the UI's own domain with 1."""

    def __init__(self, domain_id: int, args=None, localhost_only=None):
        import rclpy
        from rclpy.context import Context
        from rclpy.executors import MultiThreadedExecutor
        from rclpy.node import Node
        from rclpy.signals import SignalHandlerOptions
        self.domain_id = domain_id
        self.context = Context()
        with localhost_only_env(localhost_only):     # read by rcl/rmw when the context and its participant are created
            rclpy.init(args=args, context=self.context, domain_id=domain_id, signal_handler_options=SignalHandlerOptions.NO)
            self.node = Node("dual_cut_ui_cameras", context=self.context, enable_rosout=False, start_parameter_services=False)
        self.source = CameraSource(self.node)
        self.executor = MultiThreadedExecutor(num_threads=2, context=self.context)
        self.executor.add_node(self.node)
        self.thread = threading.Thread(target=self.executor.spin, name="ros-camera-executor", daemon=True)
        self.thread.start()

    def teardown(self) -> None:
        shutdown_camera_context(self.executor, self.thread, self.node, self.context)


def shutdown_camera_context(executor, thread, node, context) -> None:
    """Executor first, then the thread, the node, finally the context (duck-typed so the order can be tested)."""
    executor.shutdown()
    thread.join(timeout=1.0)
    node.destroy_node()
    context.shutdown()
