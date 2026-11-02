"""ROS 2 plumbing for wrist_handeye.py: services, actions, topics, keyboard.  Imported lazily (offline `solve` needs no ROS)."""
from __future__ import annotations

import collections
import contextlib
import math
import select
import sys
import termios
import threading
import time
import tty
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import rclpy
from builtin_interfaces.msg import Duration as DurationMsg
from controller_manager_msgs.srv import ListHardwareComponents
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import (CollisionObject, Constraints, JointConstraint, MotionPlanRequest, PlanningScene,
                             PlanningSceneComponents, RobotState, RobotTrajectory)
from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene, GetPositionIK, GetMotionPlan, GetStateValidity
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import CameraInfo, Image, JointState
from shape_msgs.msg import SolidPrimitive
import tf2_ros

JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
BASE = "base_link"
LINK = "link6"
PREFIX = "whe_"


class UserStop(Exception):
    """Ctrl-C / SIGTERM / 'q' / EOF at a prompt."""


class PlanError(Exception):
    pass


class Keys:
    """Non-blocking single-key reader for the motion waits (only when stdin is a terminal)."""

    def __init__(self):
        self.tty = sys.stdin.isatty()
        self._saved = None

    @contextlib.contextmanager
    def raw(self):
        if not self.tty:
            yield
            return
        self._saved = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            yield
        finally:
            self.restore()

    def restore(self):
        if self._saved is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._saved)
            self._saved = None

    def poll(self) -> Optional[str]:
        if not self.tty or self._saved is None:
            return None
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None


class IO:
    def __init__(self, group: str = "igus_rebel_arm"):
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        self.group = group
        self.node = Node("wrist_handeye")
        self.log = self.node.get_logger()
        cb = ReentrantCallbackGroup()
        self.cb = cb
        self.bridge = CvBridge()
        self.stop_event = threading.Event()
        self.in_prompt = False
        self.cleaning = False  # True while removing the safety scene: stop requests no longer raise
        self._goal_lock = threading.Lock()
        self._active_goal = None
        self.keys = Keys()
        # joint states
        self._lock = threading.Lock()
        self._js: Dict[str, float] = {}
        self._js_hist: collections.deque = collections.deque(maxlen=4000)
        self._js_msg: Optional[JointState] = None
        self._js_recv = 0.0
        self.node.create_subscription(JointState, "/joint_states", self._on_js, 50, callback_group=cb)
        # cameras
        self._wrist_imgs: collections.deque = collections.deque(maxlen=12)
        self._scene_imgs: collections.deque = collections.deque(maxlen=12)
        self.wrist_info: Optional[CameraInfo] = None
        self.scene_info: Optional[CameraInfo] = None
        self.node.create_subscription(Image, "/arm_left/wrist_camera/color/image_raw", lambda m: self._wrist_imgs.append((time.monotonic(), m)),
                                      qos_profile_sensor_data, callback_group=cb)
        self.node.create_subscription(Image, "/camera/color/image_raw", lambda m: self._scene_imgs.append((time.monotonic(), m)),
                                      qos_profile_sensor_data, callback_group=cb)
        self.node.create_subscription(CameraInfo, "/arm_left/wrist_camera/color/camera_info", self._on_wi, qos_profile_sensor_data, callback_group=cb)
        self.node.create_subscription(CameraInfo, "/camera/color/camera_info", self._on_si, qos_profile_sensor_data, callback_group=cb)
        # services / actions
        self.cli_hw = self.node.create_client(ListHardwareComponents, "/controller_manager/list_hardware_components", callback_group=cb)
        self.cli_ik = self.node.create_client(GetPositionIK, "/compute_ik", callback_group=cb)
        self.cli_plan = self.node.create_client(GetMotionPlan, "/plan_kinematic_path", callback_group=cb)
        self.cli_apply = self.node.create_client(ApplyPlanningScene, "/apply_planning_scene", callback_group=cb)
        self.cli_get = self.node.create_client(GetPlanningScene, "/get_planning_scene", callback_group=cb)
        self.cli_valid = self.node.create_client(GetStateValidity, "/check_state_validity", callback_group=cb)
        self.act_exec = ActionClient(self.node, ExecuteTrajectory, "/execute_trajectory", callback_group=cb)
        self.tf_buf = tf2_ros.Buffer(cache_time=rclpy.duration.Duration(seconds=30))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buf, self.node, spin_thread=False)
        self.executor = MultiThreadedExecutor(num_threads=6)
        self.executor.add_node(self.node)
        self._spin = threading.Thread(target=self.executor.spin, daemon=True)
        self._spin.start()

    # ---------------------------------------------------------------- callbacks
    def _on_js(self, m: JointState):
        with self._lock:
            for n, p in zip(m.name, m.position):
                self._js[n] = p
            self._js_msg = m
            self._js_recv = time.monotonic()
            if all(j in self._js for j in JOINTS):
                self._js_hist.append((self._js_recv, np.array([self._js[j] for j in JOINTS])))

    def _on_wi(self, m):
        self.wrist_info = m

    def _on_si(self, m):
        self.scene_info = m

    # ---------------------------------------------------------------- stop handling
    def request_stop(self, why: str = ""):
        if self.cleaning:
            return
        if not self.stop_event.is_set():
            print("\n[STOP] %s - cancelling the active goal" % why, flush=True)
        self.stop_event.set()
        self.cancel_active()
        if self.in_prompt:
            raise UserStop(why)

    def cancel_active(self):
        with self._goal_lock:
            gh = self._active_goal
        if gh is not None:
            try:
                gh.cancel_goal_async()
            except Exception as e:  # noqa: BLE001
                print("cancel failed:", e, flush=True)

    def check_stop(self):
        if self.cleaning:
            return
        if self.stop_event.is_set():
            raise UserStop("stop requested")
        k = self.keys.poll()
        if k and k.lower() == "q":
            self.stop_event.set()
            self.cancel_active()
            raise UserStop("'q' pressed")

    def prompt(self, text: str) -> str:
        """Blocking line prompt; Ctrl-C / EOF raise UserStop.  'q' answers also stop."""
        self.in_prompt = True
        try:
            try:
                ans = input(text)
            except EOFError:
                raise UserStop("stdin closed at a prompt")
            except KeyboardInterrupt:
                raise UserStop("Ctrl-C at a prompt")
        finally:
            self.in_prompt = False
        if ans.strip().lower() == "q":
            self.stop_event.set()
            raise UserStop("'q' answered")
        return ans

    def shutdown(self):
        self.keys.restore()
        try:
            self.executor.shutdown(timeout_sec=1.0)
            self.node.destroy_node()
            rclpy.shutdown()
        except Exception:  # noqa: BLE001
            pass

    # ---------------------------------------------------------------- futures
    def wait(self, fut, timeout: float, what: str, allow_stop: bool = True):
        t0 = time.monotonic()
        while not fut.done():
            if allow_stop:
                self.check_stop()
            if time.monotonic() - t0 > timeout:
                raise TimeoutError("%s timed out after %.0f s" % (what, timeout))
            time.sleep(0.01)
        return fut.result()

    def wait_for(self, client, name: str, timeout: float = 10.0):
        t0 = time.monotonic()
        while not client.wait_for_service(timeout_sec=0.2):
            self.check_stop()
            if time.monotonic() - t0 > timeout:
                raise TimeoutError("service %s not available" % name)

    # ---------------------------------------------------------------- hardware
    def hardware_components(self) -> List[Tuple[str, str, str]]:
        self.wait_for(self.cli_hw, "/controller_manager/list_hardware_components")
        resp = self.wait(self.cli_hw.call_async(ListHardwareComponents.Request()), 5.0, "list_hardware_components")
        return [(c.name, c.class_type, c.state.label) for c in resp.component]

    # ---------------------------------------------------------------- joint states
    def wait_joint_states(self, timeout=10.0) -> np.ndarray:
        t0 = time.monotonic()
        while True:
            q = self.current_q(max_age=1.0)
            if q is not None:
                return q
            self.check_stop()
            if time.monotonic() - t0 > timeout:
                raise TimeoutError("no fresh /joint_states")
            time.sleep(0.05)

    def current_q(self, max_age: float = 1.0) -> Optional[np.ndarray]:
        with self._lock:
            if not all(j in self._js for j in JOINTS) or time.monotonic() - self._js_recv > max_age:
                return None
            return np.array([self._js[j] for j in JOINTS])

    def joint_state_msg(self, q: Sequence[float]) -> JointState:
        js = JointState()
        js.name = list(JOINTS)
        js.position = [float(v) for v in q]
        return js

    def wait_standstill(self, window=1.0, tol=0.002, timeout=40.0):
        t0 = time.monotonic()
        while True:
            self.check_stop()
            now = time.monotonic()
            with self._lock:
                hist = [(t, q) for t, q in self._js_hist if now - t <= window]
                oldest_ok = bool(hist) and (now - hist[0][0]) >= window - 0.15 and (now - self._js_recv) < 0.5
            if oldest_ok:
                arr = np.array([q for _, q in hist])
                if np.max(np.ptp(arr, axis=0)) < tol:
                    return
            if now - t0 > timeout:
                raise TimeoutError("arm did not come to a standstill within %.0f s" % timeout)
            time.sleep(0.05)

    # ---------------------------------------------------------------- TF
    def lookup(self, target: str, source: str, stamp_ns: Optional[int] = None, timeout: float = 2.0) -> Tuple[np.ndarray, bool]:
        """T_target_source; (T, exact_stamp)."""
        from handeye_math import T_from_xyz_quat

        def to_T(tf):
            t, q = tf.transform.translation, tf.transform.rotation
            return T_from_xyz_quat(t.x, t.y, t.z, q.x, q.y, q.z, q.w)

        if stamp_ns:
            try:
                tf = self.tf_buf.lookup_transform(target, source, rclpy.time.Time(nanoseconds=int(stamp_ns)),
                                                  rclpy.duration.Duration(seconds=timeout))
                return to_T(tf), True
            except Exception:  # noqa: BLE001
                pass
        t0 = time.monotonic()
        while True:
            try:
                tf = self.tf_buf.lookup_transform(target, source, rclpy.time.Time(), rclpy.duration.Duration(seconds=0.5))
                return to_T(tf), False
            except Exception as e:  # noqa: BLE001
                self.check_stop()
                if time.monotonic() - t0 > timeout:
                    raise RuntimeError("TF %s <- %s unavailable: %s" % (target, source, e))
                time.sleep(0.05)

    # ---------------------------------------------------------------- images
    def latest_image(self, which: str, newer_than_ns: int = 0, timeout: float = 5.0):
        """Newest (stamp_ns, bgr image) of 'wrist'/'scene' with stamp > newer_than_ns."""
        buf = self._wrist_imgs if which == "wrist" else self._scene_imgs
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            self.check_stop()
            if buf:
                _, m = buf[-1]
                st = m.header.stamp.sec * 10 ** 9 + m.header.stamp.nanosec
                if st > newer_than_ns:
                    return st, self.bridge.imgmsg_to_cv2(m, "bgr8")
            time.sleep(0.03)
        return None

    def grab_frames(self, which: str, n: int, newer_than_ns: int, timeout: float = 8.0) -> List[Tuple[int, np.ndarray]]:
        out, last = [], newer_than_ns
        t0 = time.monotonic()
        while len(out) < n and time.monotonic() - t0 < timeout:
            r = self.latest_image(which, last, timeout=timeout - (time.monotonic() - t0))
            if r is None:
                break
            out.append(r)
            last = r[0]
        return out

    def wait_camera_info(self, which: str, timeout=3.0) -> Optional[CameraInfo]:
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            ci = self.wrist_info if which == "wrist" else self.scene_info
            if ci is not None:
                return ci
            self.check_stop()
            time.sleep(0.05)
        return None

    def now_ns(self) -> int:
        return self.node.get_clock().now().nanoseconds

    # ---------------------------------------------------------------- planning scene
    def world_objects(self) -> Dict[str, CollisionObject]:
        self.wait_for(self.cli_get, "/get_planning_scene")
        req = GetPlanningScene.Request()
        req.components.components = PlanningSceneComponents.WORLD_OBJECT_GEOMETRY | PlanningSceneComponents.WORLD_OBJECT_NAMES
        resp = self.wait(self.cli_get.call_async(req), 10.0, "get_planning_scene")
        return {o.id: o for o in resp.scene.world.collision_objects}

    def _apply(self, objs: List[CollisionObject]):
        self.wait_for(self.cli_apply, "/apply_planning_scene")
        scene = PlanningScene()
        scene.is_diff = True
        scene.robot_state.is_diff = True
        scene.world.collision_objects = objs
        req = ApplyPlanningScene.Request()
        req.scene = scene
        resp = self.wait(self.cli_apply.call_async(req), 10.0, "apply_planning_scene", allow_stop=False)
        if not resp.success:
            raise RuntimeError("apply_planning_scene refused")

    def add_boxes(self, boxes: Dict[str, dict]):
        objs = []
        for name, b in boxes.items():
            assert name.startswith(PREFIX), name
            co = CollisionObject()
            co.id = name
            co.header.frame_id = BASE
            co.operation = CollisionObject.ADD
            prim = SolidPrimitive()
            prim.type = SolidPrimitive.BOX
            prim.dimensions = [float(v) for v in b["size"]]
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = [float(v) for v in b["center"]]
            qx, qy, qz, qw = b.get("quat", [0.0, 0.0, 0.0, 1.0])
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = qx, qy, qz, qw
            co.primitives = [prim]
            co.primitive_poses = [pose]
            objs.append(co)
        self._apply(objs)

    def remove_objects(self, names: Sequence[str]):
        objs = []
        for n in names:
            assert n.startswith(PREFIX), n
            co = CollisionObject()
            co.id = n
            co.header.frame_id = BASE
            co.operation = CollisionObject.REMOVE
            objs.append(co)
        if objs:
            self._apply(objs)

    def whe_objects_present(self) -> List[str]:
        return sorted(n for n in self.world_objects() if n.startswith(PREFIX))

    # ---------------------------------------------------------------- IK / planning / validity
    def compute_ik(self, T_base_link6: np.ndarray, seed_q: Sequence[float], avoid_collisions=True, timeout=0.25) -> Optional[np.ndarray]:
        from handeye_math import T_to_xyz_quat

        self.wait_for(self.cli_ik, "/compute_ik")
        req = GetPositionIK.Request()
        r = req.ik_request
        r.group_name = self.group
        r.robot_state.joint_state = self.joint_state_msg(seed_q)
        r.avoid_collisions = bool(avoid_collisions)
        r.ik_link_name = LINK
        ps = PoseStamped()
        ps.header.frame_id = BASE
        xyz, q = T_to_xyz_quat(T_base_link6)
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = xyz
        ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w = q
        r.pose_stamped = ps
        r.timeout = DurationMsg(sec=int(timeout), nanosec=int((timeout % 1) * 1e9))
        resp = self.wait(self.cli_ik.call_async(req), timeout + 5.0, "compute_ik")
        if resp.error_code.val != 1:
            return None
        pos = dict(zip(resp.solution.joint_state.name, resp.solution.joint_state.position))
        if not all(j in pos for j in JOINTS):
            return None
        return np.array([pos[j] for j in JOINTS])

    def plan(self, q_start: Sequence[float], q_goal: Sequence[float], scale: float, planning_time=6.0, attempts=4):
        """Joint-space plan.  Returns (times Nx, positions Nx6, joint_trajectory msg, planning_seconds)."""
        self.wait_for(self.cli_plan, "/plan_kinematic_path")
        req = GetMotionPlan.Request()
        mp: MotionPlanRequest = req.motion_plan_request
        mp.group_name = self.group
        mp.start_state.joint_state = self.joint_state_msg(q_start)
        mp.start_state.is_diff = False
        c = Constraints()
        for j, v in zip(JOINTS, q_goal):
            jc = JointConstraint()
            jc.joint_name, jc.position = j, float(v)
            jc.tolerance_above = jc.tolerance_below = 1e-3
            jc.weight = 1.0
            c.joint_constraints.append(jc)
        mp.goal_constraints = [c]
        mp.allowed_planning_time = float(planning_time)
        mp.num_planning_attempts = int(attempts)
        mp.max_velocity_scaling_factor = float(scale)
        mp.max_acceleration_scaling_factor = float(scale)
        t0 = time.monotonic()
        resp = self.wait(self.cli_plan.call_async(req), planning_time * attempts + 15.0, "plan_kinematic_path")
        dt = time.monotonic() - t0
        res = resp.motion_plan_response
        if res.error_code.val != 1:
            raise PlanError("planning failed (MoveIt error %d)" % res.error_code.val)
        jt = res.trajectory.joint_trajectory
        idx = [jt.joint_names.index(j) for j in JOINTS]
        pos = np.array([[p.positions[i] for i in idx] for p in jt.points])
        times = np.array([p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 for p in jt.points])
        if len(pos) < 2 or times[-1] <= 0 or np.any(np.diff(times) < 0):
            raise PlanError("trajectory has no valid time parameterisation")
        return times, pos, jt, dt

    def state_validity(self, q: Sequence[float]) -> Tuple[bool, List[Tuple[str, str]]]:
        """Is a joint state valid in the CURRENT planning scene?  Returns (valid, [(body1, body2) contacts])."""
        self.wait_for(self.cli_valid, "/check_state_validity")
        req = GetStateValidity.Request()
        req.group_name = self.group
        req.robot_state.joint_state = self.joint_state_msg(q)
        resp = self.wait(self.cli_valid.call_async(req), 5.0, "check_state_validity")
        return bool(resp.valid), [(c.contact_body_1, c.contact_body_2) for c in resp.contacts]

    def path_collision_free(self, positions: np.ndarray, max_checks=80) -> Tuple[bool, int]:
        """Independent per-waypoint /check_state_validity (the planner already validated; this is a cross-check)."""
        self.wait_for(self.cli_valid, "/check_state_validity")
        stride = max(1, int(math.ceil(len(positions) / max_checks)))
        idx = list(range(0, len(positions), stride))
        if idx[-1] != len(positions) - 1:
            idx.append(len(positions) - 1)
        for i in idx:
            req = GetStateValidity.Request()
            req.group_name = self.group
            req.robot_state.joint_state = self.joint_state_msg(positions[i])
            resp = self.wait(self.cli_valid.call_async(req), 5.0, "check_state_validity")
            if not resp.valid:
                return False, len(idx)
        return True, len(idx)

    def execute(self, jt) -> None:
        """Send a trajectory to /execute_trajectory; cancels at once on stop; raises on any non-success."""
        if not self.act_exec.wait_for_server(timeout_sec=5.0):
            raise RuntimeError("/execute_trajectory action server not available")
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = RobotTrajectory()
        goal.trajectory.joint_trajectory = jt
        with self.keys.raw():
            gh = self.wait(self.act_exec.send_goal_async(goal), 10.0, "send_goal")
            if not gh.accepted:
                raise RuntimeError("ExecuteTrajectory goal rejected")
            with self._goal_lock:
                self._active_goal = gh
            try:
                res_fut = gh.get_result_async()
                try:
                    res = self.wait(res_fut, 600.0, "ExecuteTrajectory")
                except UserStop:
                    self.cancel_active()
                    t0 = time.monotonic()
                    while not res_fut.done() and time.monotonic() - t0 < 3.0:
                        time.sleep(0.02)
                    raise
            finally:
                with self._goal_lock:
                    self._active_goal = None
        if res.result.error_code.val != 1:
            raise RuntimeError("ExecuteTrajectory failed, MoveIt error code %d" % res.result.error_code.val)


def publish_static_tf(parent: str, child: str, T: np.ndarray):
    """Publish a static TF until Ctrl-C (live check of a result; nothing in src/ is touched)."""
    from geometry_msgs.msg import TransformStamped
    from handeye_math import T_to_xyz_quat

    if not rclpy.ok():
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = Node("wrist_handeye_static_tf")
    br = tf2_ros.StaticTransformBroadcaster(node)
    msg = TransformStamped()
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.header.frame_id, msg.child_frame_id = parent, child
    xyz, q = T_to_xyz_quat(T)
    msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z = xyz
    msg.transform.rotation.x, msg.transform.rotation.y, msg.transform.rotation.z, msg.transform.rotation.w = q
    br.sendTransform(msg)
    print("publishing static TF %s -> %s (Ctrl-C to stop)" % (parent, child), flush=True)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
