"""Capture recorder for the dual-arm eye-to-hand calibration.

Never commands the robot: it only subscribes (image, camera_info, joint_states, TF)
and offers std_srvs/Trigger services:

  ~/capture        take the newest image, require the active arms stationary, detect the targets and record
                   one sample per arm whose target is detected; if the fixed table board is detected, its
                   corners are stored too (table_obs.jsonl, not tied to an arm)
  ~/capture_touch  the operator has put the knife tip (TCP) of the arm named by parameter ``touch_arm`` on the
                   printed touch point ``touch_point_id`` of the table board and keeps it still: record
                   T_base_tcp from TF plus the joints (touches.jsonl).  Set the parameters first:
                   ``ros2 param set /dual_arm_calibration_recorder touch_point_id 2``.
  ~/capture_intrinsics  a (handheld or rigidly mounted) board of the target ``intrinsics_target`` (default the table
                   board spec: use a second print of it) is held still anywhere in the image: save the full image and
                   its detections for ``calibrate_intrinsics`` (refuses blurred frames: two-frame corner check)
  ~/status         counts per arm, stationary state, image / TF health, URDF saved or not, session info
  ~/status_json    the same (and more: touched point ids, intrinsics view coverage, touch list) as one JSON object in
                   ``message``, for the calibration web UI; read-only (does not touch session.json)

Capture quality: ``capture`` / ``capture_touch`` need the arms stationary for ``stationary_window_s`` (default 2.0 s)
and compare two images >= 0.4 s apart (both after the stationary start); a common corner that moved more than
``max_corner_motion_px`` (0.2, coherent motion of a fitted 2D similarity transform, so that detection noise does not
count) or a single corner farther than ``max_corner_single_px`` (1.0) rejects the capture ("board still moving").
Encoder stillness does not prove that the printed structure is still.

Session output (param ``output_dir``): images/cap_NNNN.png, samples.jsonl, table_obs.jsonl, touches.jsonl,
robot_description.urdf (saved once, for forward kinematics / joint offsets), session.json.

Parameter ``active_arms`` (default both) restricts the recorder to one arm for the single-arm stack
(``launch/record.launch.py mode:=single_left|single_right`` sets it together with the frame names and an
empty joint prefix).  Only active arms are required to be present in TF / joint_states, stationary-checked
and recorded; the other arm's target, if visible, is ignored.
"""
from __future__ import annotations

import collections
import datetime
import json
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import rclpy
import yaml
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, JointState
from rcl_interfaces.srv import GetParameters
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener

from .capture_quality import corner_motion_stats, judge_motion, pick_previous_frame
from .modes import active_arms_of
from .kinematics import ArmKinematics, KinematicsError, Urdf
from .session import (append_intrinsics_view, append_sample, append_table_obs, append_touch, init_session,
                      load_intrinsics_views, load_table_obs, load_touches, save_urdf_once, update_session_info,
                      urdf_path)
from .solver import Sample, TableObservation, Touch, pnp_candidates
from .targets import (TargetDetector, default_targets_path, load_targets_all, specs_to_config, stickers_from_config,
                      table_to_config)
from .transforms import make_T, quat_to_R

ARMS = ("left", "right")


def image_msg_to_bgr(msg: Image) -> np.ndarray:
    """Decode a sensor_msgs/Image without cv_bridge (avoids numpy/cv2 ABI mismatches)."""
    enc = msg.encoding.lower()
    channels = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "mono8": 1, "8uc1": 1, "8uc3": 3}.get(enc)
    if channels is None:
        raise ValueError(f"unsupported image encoding '{msg.encoding}'")
    buf = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    img = buf[:, : msg.width * channels].reshape(msg.height, msg.width, channels)
    if enc == "rgb8":
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if enc == "rgba8":
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    if enc == "bgra8":
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if channels == 1:
        return cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)
    return img.copy()


def _centre_px(pix) -> List[float]:
    a = np.asarray(pix, dtype=float).reshape(-1, 2)
    return [float(a[:, 0].mean()), float(a[:, 1].mean())]


def _bbox_px(pix) -> List[float]:
    a = np.asarray(pix, dtype=float).reshape(-1, 2)
    return [float(a[:, 0].min()), float(a[:, 1].min()), float(a[:, 0].max()), float(a[:, 1].max())]


def tf_msg_to_T(tf) -> np.ndarray:
    t, q = tf.transform.translation, tf.transform.rotation
    return make_T(quat_to_R(q.x, q.y, q.z, q.w), [t.x, t.y, t.z])


class CalibrationRecorder(Node):
    def __init__(self, parameter_overrides=None):
        super().__init__("dual_arm_calibration_recorder", parameter_overrides=parameter_overrides)
        P = self.declare_parameter
        P("output_dir", "")
        P("targets_file", "")
        P("active_arms", ["left", "right"])   # subset of ["left", "right"]; single-arm stacks record one arm
        P("image_topic", "/camera/color/image_raw")
        P("camera_info_topic", "/camera/color/camera_info")
        P("joint_states_topic", "/joint_states")
        P("left_joint_prefix", "arm_left_")
        P("right_joint_prefix", "arm_right_")
        P("left_base_frame", "arm_left_base_link")
        P("left_tip_frame", "arm_left_link6")
        P("right_base_frame", "arm_right_base_link")
        P("right_tip_frame", "arm_right_link6")
        P("camera_link_frame", "camera_link")
        P("optical_frame", "camera_color_optical_frame")
        P("min_corners", 8)            # ChArUco: minimum detected points per arm
        P("min_corners_aruco", 4)      # single-ArUco fallback: all 4 corners
        P("max_pnp_rms_px", 2.0)
        P("stationary_window_s", 2.0)       # arms must have been still this long (was 0.5 s)
        P("max_joint_velocity", 0.01)       # rad/s, from the message velocity field if present
        P("max_joint_motion_rad", 0.002)    # max-min of every joint over the stationary window
        P("max_image_age_s", 1.0)
        P("joint_state_max_age_s", 0.5)
        P("tf_max_age_s", 1.0)
        P("min_pose_separation_rad", 0.01)  # skip a sample whose joints match an existing sample of that arm
        P("min_marker_perimeter_rate", 0.02)
        P("left_tcp_frame", "")        # default: <tip frame without 'link6'> + 'xeg32_tool_tip'
        P("right_tcp_frame", "")
        P("min_corners_table", 16)     # table board: minimum detected points (4 markers)
        P("max_table_obs", 60)         # the board and camera are static: more images add little
        P("touch_arm", "")             # "left" | "right"; empty = the only active arm
        P("touch_point_id", 0)         # id of the table-board touch point being touched (set before capture_touch)
        P("max_corner_motion_px", 0.2)       # two-frame check: coherent (similarity-fit) corner motion above this rejects
        P("max_corner_single_px", 1.0)       # ... and so does any single corner moving more (shaking / bad corner)
        P("min_image_gap_s", 0.4)            # ... the two images must be at least this far apart
        P("corner_check", True)
        P("min_common_corners_check", 4)     # fewer common corners: stillness cannot be verified
        P("intrinsics_file", "")             # YAML from calibrate_intrinsics: replaces camera_info for K / distortion
        P("intrinsics_target", "table")      # board used by capture_intrinsics (name in the targets file)
        P("min_corners_intrinsics", 16)
        P("min_intrinsics_view_shift_px", 15.0)   # a new view must differ from the previous one by this much
        P("ambient_temp_c", float("nan"))    # session metadata (stored in session.json): ambient temperature,
        P("ambient_rh_pct", float("nan"))    # relative humidity, minutes since the stack/camera were switched on,
        P("warmup_min", float("nan"))
        P("operator_note", "")               # free text
        P("robot_description_file", "")      # URDF file to save into the session (else topic / parameter)
        P("robot_description_node", "robot_state_publisher")
        P("robot_description_topic", "robot_description")
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.p = {n: g(n) for n in [
            "image_topic", "camera_info_topic", "joint_states_topic", "left_joint_prefix", "right_joint_prefix",
            "left_base_frame", "left_tip_frame", "right_base_frame", "right_tip_frame", "camera_link_frame",
            "optical_frame", "min_corners", "min_corners_aruco", "max_pnp_rms_px", "stationary_window_s", "max_joint_velocity",
            "max_joint_motion_rad", "max_image_age_s", "joint_state_max_age_s", "tf_max_age_s",
            "min_pose_separation_rad", "min_corners_table", "max_table_obs", "max_corner_motion_px", "max_corner_single_px",
            "min_image_gap_s",
            "corner_check", "min_common_corners_check", "intrinsics_target", "min_corners_intrinsics",
            "min_intrinsics_view_shift_px"]}
        self.active_arms = active_arms_of(g("active_arms"))
        self.p["active_arms"] = list(self.active_arms)
        for arm in ARMS:
            tcp = g(f"{arm}_tcp_frame")
            tip = self.p[f"{arm}_tip_frame"]
            if not tcp:
                tcp = (tip[: -len("link6")] if tip.endswith("link6") else "") + "xeg32_tool_tip"
            self.p[f"{arm}_tcp_frame"] = tcp

        targets_file = g("targets_file") or default_targets_path()
        self.specs, self.table_spec = load_targets_all(targets_file)
        with open(targets_file) as fh:
            self.stickers = stickers_from_config(yaml.safe_load(fh), self.table_spec)
        self.intrinsics_K, self.intrinsics_dist = None, None
        intr_file = os.path.expanduser(g("intrinsics_file") or "")
        if intr_file:
            from .intrinsics import load_intrinsics_file
            self.intrinsics_K, self.intrinsics_dist, _ = load_intrinsics_file(intr_file)
            self.get_logger().info(f"intrinsics from {intr_file} override camera_info")
        all_specs = dict(self.specs)
        if self.table_spec is not None:
            all_specs["table"] = self.table_spec
        self.all_specs = all_specs
        if self.p["intrinsics_target"] not in all_specs:
            raise ValueError(f"intrinsics_target '{self.p['intrinsics_target']}' is not a target of the targets file "
                             f"({sorted(all_specs)})")
        self.detector = TargetDetector(all_specs, float(g("min_marker_perimeter_rate")))
        for s in all_specs.values():
            self.get_logger().info(s.describe())

        out = g("output_dir") or os.path.expanduser(
            "~/dual_arm_calibration/session_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.session_dir = os.path.abspath(os.path.expanduser(out))
        meta = {"created": datetime.datetime.now().isoformat(), "targets_file": os.path.abspath(targets_file),
                "targets": {**specs_to_config(self.specs),
                            **({"table_board": table_to_config(self.table_spec)} if self.table_spec else {}),
                            "touch_stickers": self.stickers.to_config()},
                "parameters": self.p}
        self.n_existing = init_session(self.session_dir, meta)
        self._update_session_info()
        self.capture_count = len([f for f in os.listdir(os.path.join(self.session_dir, "images"))
                                  if f.startswith("cap_")])
        self.counts = {a: 0 for a in ARMS}
        self.touch_counts = {a: 0 for a in ARMS}
        self.n_table_obs = 0
        self.n_intrinsics = 0
        self.touched: Dict[str, Dict[int, int]] = {a: {} for a in ARMS}     # arm -> {touch point id: times touched}
        self.intrinsics_views: List[Tuple[List[float], List[float]]] = []   # (centre px, bbox px) per captured view
        self.sample_centres: Dict[str, List[List[float]]] = {a: [] for a in ARMS}   # board centre px per arm sample
        self.last_joints: Dict[str, List[Dict[str, float]]] = {a: [] for a in ARMS}
        self._load_existing()
        self.get_logger().info(f"session dir: {self.session_dir} ({self.n_existing} existing samples)")

        self.image: Optional[Image] = None
        self.image_hist = collections.deque(maxlen=14)   # (stamp_s, Image), >= 0.15 s apart: for the two-frame check
        self.image_rx: Optional[Time] = None
        self.info: Optional[CameraInfo] = None
        self.joint_hist = collections.deque()  # (t_sec, {name: pos}, {name: vel})
        self.tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(Image, g("image_topic"), self._on_image, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, g("camera_info_topic"), self._on_info, qos_profile_sensor_data)
        self.create_subscription(JointState, g("joint_states_topic"), self._on_joints, qos_profile_sensor_data)
        self.create_service(Trigger, "~/capture", self._srv_capture)
        self.create_service(Trigger, "~/capture_touch", self._srv_capture_touch)
        self.create_service(Trigger, "~/capture_intrinsics", self._srv_capture_intrinsics)
        self.create_service(Trigger, "~/status", self._srv_status)
        self.create_service(Trigger, "~/status_json", self._srv_status_json)
        self._setup_urdf_capture()

    # ---- bookkeeping -----------------------------------------------------
    def _load_existing(self):
        from .session import load_samples, samples_path
        if os.path.exists(samples_path(self.session_dir)):
            for s in load_samples(self.session_dir):
                self.counts[s.arm] += 1
                self.last_joints[s.arm].append(s.joints)
                self.sample_centres[s.arm].append(_centre_px(s.pix))
        self.n_table_obs = len(load_table_obs(self.session_dir))
        for t in load_touches(self.session_dir):
            self.touch_counts[t.arm] += 1
            self.touched[t.arm][t.point_id] = self.touched[t.arm].get(t.point_id, 0) + 1
        views = load_intrinsics_views(self.session_dir)
        self.n_intrinsics = len(views)
        for v in views:
            if v.get("pix"):
                self.intrinsics_views.append((_centre_px(v["pix"]), _bbox_px(v["pix"])))

    # ---- robot description (URDF for forward kinematics) ------------------------------------
    def _setup_urdf_capture(self):
        self.urdf_saved = os.path.exists(urdf_path(self.session_dir))
        self._urdf_tries = 0
        if self.urdf_saved:
            return
        f = self.get_parameter("robot_description_file").value
        if f:
            try:
                with open(os.path.expanduser(f), encoding="utf-8") as fh:
                    self._save_urdf(fh.read(), f"file {f}")
            except OSError as e:
                self.get_logger().error(f"robot_description_file: {e}")
            return
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(String, self.get_parameter("robot_description_topic").value,
                                 lambda m: self._save_urdf(m.data, "topic"), qos)
        node = self.get_parameter("robot_description_node").value.strip("/")
        self._urdf_client = self.create_client(GetParameters, f"/{node}/get_parameters")
        self._urdf_timer = self.create_timer(3.0, self._urdf_poll)

    def _urdf_poll(self):
        if self.urdf_saved:
            self._urdf_timer.cancel()
            return
        self._urdf_tries += 1
        if self._urdf_tries > 20:
            self.get_logger().warn("robot_description not received: joint offsets will not be available for this "
                                   "session (set robot_description_file:=<urdf> to save one)")
            self._urdf_timer.cancel()
            return
        if not self._urdf_client.service_is_ready():
            return
        req = GetParameters.Request()
        req.names = ["robot_description"]
        fut = self._urdf_client.call_async(req)

        def done(f):
            try:
                v = f.result().values
                if v and v[0].string_value:
                    self._save_urdf(v[0].string_value, "robot_state_publisher parameter")
            except Exception as e:  # noqa: BLE001
                self.get_logger().warn(f"robot_description parameter query failed: {e}")
        fut.add_done_callback(done)

    def _save_urdf(self, text: str, source: str) -> None:
        if self.urdf_saved or not text.strip():
            return
        try:
            urdf = Urdf.parse(text)
        except KinematicsError as e:
            self.get_logger().error(f"robot_description from {source} is not usable: {e}")
            return
        for arm in self.active_arms:
            try:
                ArmKinematics.from_urdf(urdf, self.p[f"{arm}_base_frame"], self.p[f"{arm}_tip_frame"],
                                        self.p[f"{arm}_tcp_frame"])
            except KinematicsError as e:
                self.get_logger().warn(f"robot_description ({source}): arm {arm}: {e}")
        self.urdf_saved = save_urdf_once(self.session_dir, text)
        self.get_logger().info(f"saved robot_description.urdf ({source}) into the session")

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    # ---- callbacks -------------------------------------------------------
    def _on_image(self, msg: Image):
        self.image = msg
        self.image_rx = self.get_clock().now()
        st = self._image_stamp(msg)
        if not self.image_hist or st - self.image_hist[-1][0] >= 0.15 or st < self.image_hist[-1][0]:
            self.image_hist.append((st, msg))

    def _image_stamp(self, msg: Image) -> float:
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if stamp <= 0:
            stamp = (self.image_rx or self.get_clock().now()).nanoseconds * 1e-9
        return stamp

    def _session_info(self) -> dict:
        def num(name):
            v = float(self.get_parameter(name).value)
            return None if v != v else v
        return {"ambient_temp_c": num("ambient_temp_c"), "ambient_rh_pct": num("ambient_rh_pct"),
                "warmup_min": num("warmup_min"), "operator_note": str(self.get_parameter("operator_note").value)}

    def _update_session_info(self):
        try:
            update_session_info(self.session_dir, self._session_info())
        except OSError as e:
            self.get_logger().warn(f"cannot update session.json: {e}")

    def _camera(self, info: CameraInfo):
        """(K, dist): the --intrinsics-file values if configured, else camera_info."""
        if self.intrinsics_K is not None:
            return self.intrinsics_K.copy(), self.intrinsics_dist.copy()
        return np.array(info.k, dtype=float).reshape(3, 3), np.array(info.d, dtype=float)

    def _previous_frame(self, stamp: float, since: Optional[float]):
        """(prev_stamp, prev_Image) >= min_image_gap_s older than ``stamp`` and not before the stationary start."""
        return pick_previous_frame(self.image_hist, stamp, -1e18 if since is None else since,
                                   float(self.p["min_image_gap_s"]))

    def _on_info(self, msg: CameraInfo):
        self.info = msg

    def _on_joints(self, msg: JointState):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        t = stamp if stamp > 0 else self._now()
        pos = dict(zip(msg.name, msg.position))
        vel = dict(zip(msg.name, msg.velocity)) if len(msg.velocity) == len(msg.name) else {}
        self.joint_hist.append((t, pos, vel, self._now()))
        while self.joint_hist and self.joint_hist[0][3] < self._now() - 10.0:
            self.joint_hist.popleft()

    # ---- checks ----------------------------------------------------------
    def stationary_since(self) -> Tuple[Optional[float], str]:
        """Return (time since which all arm joints have been still, '' ) or (None, reason)."""
        if not self.joint_hist:
            return None, "no /joint_states received"
        now = self._now()
        age = now - self.joint_hist[-1][3]
        if age > self.p["joint_state_max_age_s"]:
            return None, f"joint_states stale ({age:.2f} s old)"
        prefixes = tuple(self.p[f"{arm}_joint_prefix"] for arm in self.active_arms)
        last = self.joint_hist[-1][1]
        for pre in prefixes:
            if not any(n.startswith(pre) for n in last):
                return None, f"no joints with prefix '{pre}' in joint_states"
        names = [n for n in last if n.startswith(prefixes)]
        lo = {n: last[n] for n in names}
        hi = dict(lo)
        since = self.joint_hist[-1][0]
        full_history = True
        for t, pos, vel, _ in reversed(self.joint_hist):
            moving = False
            for n in names:
                if n not in pos:
                    continue
                lo[n], hi[n] = min(lo[n], pos[n]), max(hi[n], pos[n])
                if hi[n] - lo[n] > self.p["max_joint_motion_rad"]:
                    moving = True
                if n in vel and abs(vel[n]) > self.p["max_joint_velocity"]:
                    moving = True
            if moving:
                full_history = False
                break
            since = t
        if full_history and now - self.joint_hist[0][3] < self.p["stationary_window_s"]:
            return None, "not enough joint_states history yet"
        if now - since < self.p["stationary_window_s"]:
            return None, (f"arms not stationary yet (still for {max(now - since, 0.0):.2f} s, "
                          f"need {self.p['stationary_window_s']:.2f} s)")
        return since, ""

    def _lookup(self, target: str, source: str, dynamic: bool):
        try:
            tf = self.tf_buffer.lookup_transform(target, source, Time(), timeout=Duration(seconds=0.0))
        except Exception as e:  # noqa: BLE001 - tf2 raises several exception types
            return None, f"TF {target}->{source} unavailable: {e}"
        if dynamic:
            st = Time.from_msg(tf.header.stamp)
            if st.nanoseconds > 0:
                age = (self.get_clock().now() - st).nanoseconds * 1e-9
                if age > self.p["tf_max_age_s"]:
                    return None, f"TF {target}->{source} is stale ({age:.2f} s old)"
        return tf_msg_to_T(tf), ""

    def _stillness(self, dets_new: dict, stamp: float, since: Optional[float], names):
        """Two-frame check.  Returns ({name: (ok, text, coherent_px, single_max_px)}, refusal) where ``refusal`` is a
        message if no earlier frame is available (nothing can be verified yet) and None otherwise."""
        if not self.p["corner_check"]:
            return {n: (True, "corner check off", None, None) for n in names}, None
        prev = self._previous_frame(stamp, since)
        if prev is None:
            return None, (f"no earlier image >= {self.p['min_image_gap_s']:.1f} s older than the newest one after the "
                          "stationary start (two-frame stillness check); hold still and call again")
        try:
            dets_prev = self.detector.detect(image_msg_to_bgr(prev[1]))
        except ValueError as e:
            return None, str(e)
        out = {}
        for n in names:
            d_new, d_prev = dets_new.get(n), dets_prev.get(n)
            if d_new is None or d_prev is None:
                out[n] = (False, "cannot verify stillness: not detected in the earlier frame", None, None)
                continue
            m, ms, k = corner_motion_stats(d_prev.points, d_new.points)
            ok, txt = judge_motion(m, ms, k, float(self.p["max_corner_motion_px"]), float(self.p["max_corner_single_px"]),
                                   int(self.p["min_common_corners_check"]))
            out[n] = (ok, txt, m, ms)
        return out, None

    # ---- services --------------------------------------------------------
    def _srv_status(self, req, resp):
        lines = [f"session {self.session_dir}",
                 f"active arms {','.join(self.active_arms)}",
                 f"samples left {self.counts['left']} right {self.counts['right']}, captures {self.capture_count}",
                 f"table-board images {self.n_table_obs}, touches left {self.touch_counts['left']} "
                 f"right {self.touch_counts['right']}"
                 + ("" if self.table_spec else " (no table_board in the targets file)"),
                 "robot_description.urdf: " + ("saved" if self.urdf_saved else "not saved (no joint offsets possible)"),
                 f"stationary window {self.p['stationary_window_s']:.1f} s, two-frame corner check "
                 + (f"on (common <= {self.p['max_corner_motion_px']:.2f} px, single <= {self.p['max_corner_single_px']:.2f} px, "
                    f"images >= {self.p['min_image_gap_s']:.1f} s apart)"
                    if self.p["corner_check"] else "OFF"),
                 f"intrinsics views {self.n_intrinsics} (target '{self.p['intrinsics_target']}')"
                 + ("; intrinsics file overrides camera_info" if self.intrinsics_K is not None else "")]
        info_now = {k: v for k, v in self._session_info().items() if v not in (None, "")}
        lines.append("session info: " + (", ".join(f"{k} {v}" for k, v in info_now.items()) if info_now else
                                         "none (set ambient_temp_c, ambient_rh_pct, warmup_min, operator_note)"))
        self._update_session_info()
        if self.image is None:
            lines.append("image: none received")
        else:
            lines.append(f"image: {self.image.width}x{self.image.height} {self.image.encoding}, "
                         f"age {(self.get_clock().now() - self.image_rx).nanoseconds * 1e-9:.2f} s")
        lines.append("camera_info: " + ("ok" if self.info else "none received"))
        since, why = self.stationary_since()
        lines.append("active arms stationary: " + (f"yes, for {self._now() - since:.1f} s" if since is not None else f"no ({why})"))
        for arm in self.active_arms:
            T, why = self._lookup(self.p[f"{arm}_base_frame"], self.p[f"{arm}_tip_frame"], True)
            lines.append(f"TF {arm}: " + ("ok" if T is not None else why))
        T, why = self._lookup(self.p["camera_link_frame"], self.p["optical_frame"], False)
        lines.append("TF camera_link->optical: " + ("ok" if T is not None else why))
        resp.success = True
        resp.message = "\n".join(lines)
        return resp

    def touch_list(self) -> List[dict]:
        """All touch points the operator can record: printed board crosshairs, then fit and check stickers."""
        out = []
        if self.table_spec is not None:
            out += [{"id": int(i), "kind": "board"} for i in sorted(self.table_spec.touch_points)]
        out += [{"id": int(i), "kind": "fit"} for i in sorted(self.stickers.fit_points)]
        out += [{"id": int(i), "kind": "check"} for i in sorted(self.stickers.check_points)]
        return out

    def status_dict(self) -> dict:
        """Machine-readable status for the web UI (JSON-serialisable; no side effects)."""
        now = self._now()
        image = {"ok": False, "age_s": None, "size": None}
        if self.image is not None and self.image_rx is not None:
            age = (self.get_clock().now() - self.image_rx).nanoseconds * 1e-9
            image = {"ok": age <= float(self.p["max_image_age_s"]), "age_s": age,
                     "size": [int(self.image.width), int(self.image.height)]}
        since, why = self.stationary_since()
        tf = {}
        for arm in self.active_arms:
            T, w = self._lookup(self.p[f"{arm}_base_frame"], self.p[f"{arm}_tip_frame"], True)
            tf[arm] = {"ok": T is not None, "reason": w}
        T, w = self._lookup(self.p["camera_link_frame"], self.p["optical_frame"], False)
        tf["camera"] = {"ok": T is not None, "reason": w}
        min_corners = {a: int(self.p["min_corners_aruco"] if self.specs[a].type == "aruco" else self.p["min_corners"])
                       for a in self.active_arms}
        if self.table_spec is not None:
            min_corners["table"] = int(self.p["min_corners_table"])
        return {
            "session_dir": self.session_dir,
            "active_arms": list(self.active_arms),
            "counts": dict(self.counts),
            "capture_count": int(self.capture_count),
            "n_table_obs": int(self.n_table_obs),
            "touch_counts": dict(self.touch_counts),
            "touched_ids": {a: sorted(self.touched[a]) for a in ARMS},
            "touch_point_counts": {a: {str(k): v for k, v in sorted(self.touched[a].items())} for a in ARMS},
            "touch_list": self.touch_list(),
            "n_intrinsics": int(self.n_intrinsics),
            "intrinsics_target": self.p["intrinsics_target"],
            "intrinsics_centres": [c for c, _ in self.intrinsics_views],
            "intrinsics_bboxes": [b for _, b in self.intrinsics_views],
            "intrinsics_file_override": self.intrinsics_K is not None,
            "min_corners": min_corners,
            "stationary": {"ok": since is not None, "for_s": None if since is None else now - since,
                           "reason": why, "window_s": float(self.p["stationary_window_s"])},
            "image": image,
            "camera_info": self.info is not None,
            "tf": tf,
            "urdf_saved": bool(self.urdf_saved),
            "session_info": self._session_info(),
            "samples": {a: {"count": self.counts[a], "last_centre_px": (self.sample_centres[a] or [None])[-1],
                            "centres_px": self.sample_centres[a][-100:]} for a in self.active_arms},
        }

    def _srv_status_json(self, req, resp):
        resp.success = True
        resp.message = json.dumps(self.status_dict(), allow_nan=False)
        return resp

    def _fail(self, resp, msg):
        self.get_logger().warn(f"capture refused: {msg}")
        resp.success = False
        resp.message = msg
        return resp

    def _srv_capture(self, req, resp):
        if self.image is None or self.info is None:
            return self._fail(resp, "no image / camera_info received yet")
        msg, info = self.image, self.info
        if (info.width, info.height) != (msg.width, msg.height):
            return self._fail(resp, f"image {msg.width}x{msg.height} != camera_info {info.width}x{info.height}")
        now_t = self.get_clock().now()
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if stamp <= 0:
            stamp = self.image_rx.nanoseconds * 1e-9
        age = now_t.nanoseconds * 1e-9 - stamp
        if age > self.p["max_image_age_s"]:
            return self._fail(resp, f"newest image is {age:.2f} s old (max {self.p['max_image_age_s']:.1f} s)")
        since, why = self.stationary_since()
        if since is None:
            return self._fail(resp, why)
        if stamp < since:
            return self._fail(resp, "newest image predates the stationary window; retry in a moment")
        Ts = {}
        for arm in self.active_arms:
            T, why = self._lookup(self.p[f"{arm}_base_frame"], self.p[f"{arm}_tip_frame"], True)
            if T is None:
                return self._fail(resp, why)
            Ts[arm] = T
        T_cl_o, why = self._lookup(self.p["camera_link_frame"], self.p["optical_frame"], False)
        if T_cl_o is None:
            return self._fail(resp, why)
        try:
            bgr = image_msg_to_bgr(msg)
        except ValueError as e:
            return self._fail(resp, str(e))

        K, dist = self._camera(info)
        dets = self.detector.detect(bgr)
        last = self.joint_hist[-1][1]
        parts, recorded = [], 0
        cap_id = self.capture_count + 1
        img_name = f"images/cap_{cap_id:04d}.png"
        names = [a for a in self.active_arms if dets.get(a) is not None]
        if self.table_spec is not None and dets.get("table") is not None:
            names.append("table")
        still, refusal = self._stillness(dets, stamp, since, names) if names else ({}, None)
        if refusal:
            return self._fail(resp, refusal)
        self._update_session_info()
        for arm in self.active_arms:
            det = dets.get(arm)
            if det is None:
                parts.append(f"{arm}: not detected")
                continue
            ids, obj, pix = det.arrays()
            need = self.p["min_corners_aruco"] if self.specs[arm].type == "aruco" else self.p["min_corners"]
            if len(ids) < need:
                parts.append(f"{arm}: only {len(ids)} corners (< {need})")
                continue
            if not still[arm][0]:
                parts.append(f"{arm}: {still[arm][1]}, rejected")
                continue
            prefix = self.p[f"{arm}_joint_prefix"]
            joints = {n: float(v) for n, v in last.items() if n.startswith(prefix)}
            sample = Sample(arm=arm, T_base_link6=Ts[arm], ids=ids, obj=obj, pix=pix, K=K, dist=dist,
                            image_size=(msg.width, msg.height), T_camlink_optical=T_cl_o, joints=joints,
                            stamp=stamp, image=img_name)
            cands = pnp_candidates(sample)
            if not cands:
                parts.append(f"{arm}: PnP failed")
                continue
            rms = cands[0][1]
            if rms > self.p["max_pnp_rms_px"]:
                parts.append(f"{arm}: PnP reprojection {rms:.2f} px > {self.p['max_pnp_rms_px']:.1f}, rejected")
                continue
            dup = any(set(joints) == set(q) and max(abs(joints[k] - q[k]) for k in joints)
                      < self.p["min_pose_separation_rad"] for q in self.last_joints[arm] if q)
            if dup:
                parts.append(f"{arm}: same pose as an existing sample, skipped ({len(ids)} corners)")
                continue
            if recorded == 0:
                cv2.imwrite(os.path.join(self.session_dir, img_name), bgr)
            append_sample(self.session_dir, sample,
                          {"capture": cap_id, "pnp_rms_px": rms, "n_markers": det.n_markers,
                           "n_chess": det.n_chess, "corner_motion_px": still[arm][2],
                           "corner_motion_max_px": still[arm][3]})
            self.counts[arm] += 1
            self.sample_centres[arm].append(_centre_px(pix))
            self.last_joints[arm].append(joints)
            recorded += 1
            parts.append(f"{arm}: {len(ids)} corners ({det.n_markers} markers, {det.n_chess} chess), "
                         f"PnP {rms:.2f} px, {still[arm][1]}, sample #{self.counts[arm]}")
        n_arm_samples = recorded
        tdet = dets.get("table") if self.table_spec is not None else None
        if tdet is not None:
            ids, obj, pix = tdet.arrays()
            tobs = TableObservation(ids=ids, obj=obj, pix=pix, K=K, dist=dist, image_size=(msg.width, msg.height),
                                    T_camlink_optical=T_cl_o, stamp=stamp, image=img_name)
            if len(ids) < self.p["min_corners_table"]:
                parts.append(f"table board: only {len(ids)} corners (< {self.p['min_corners_table']})")
            elif not still["table"][0]:
                parts.append(f"table board: {still['table'][1]}, not recorded")
            elif self.n_table_obs >= self.p["max_table_obs"]:
                parts.append(f"table board: {len(ids)} corners, enough images already ({self.n_table_obs})")
            else:
                c = pnp_candidates(tobs)
                if not c or c[0][1] > self.p["max_pnp_rms_px"]:
                    parts.append("table board: PnP failed or reprojection too large, not recorded")
                else:
                    if recorded == 0:
                        cv2.imwrite(os.path.join(self.session_dir, img_name), bgr)
                    append_table_obs(self.session_dir, tobs, {"capture": cap_id, "pnp_rms_px": c[0][1],
                                                              "n_markers": tdet.n_markers})
                    self.n_table_obs += 1
                    recorded += 1
                    parts.append(f"table board: {len(ids)} corners, PnP {c[0][1]:.2f} px, image #{self.n_table_obs}")
        if recorded:
            self.capture_count += 1
        resp.success = recorded > 0
        resp.message = f"cap {cap_id}: " + "; ".join(parts) if recorded else "nothing recorded: " + "; ".join(parts)
        if recorded and n_arm_samples == 0:
            # success=True only means "something was stored": the table-board image alone must not look like a good
            # capture of the arm target (found in the mock end-to-end run: arm board occluded / duplicate pose)
            resp.message = ("NO ARM SAMPLE RECORDED (table-board image only; the arm target was not stored, see below) - "
                            + resp.message)
        self.get_logger().info(resp.message)
        return resp

    def _touch_target(self, pid: int):
        """(kind, p_board xyz or None, description) of a touch point id: a printed board crosshair (known board
        coordinates) or a free sticker (fit / check; position unknown)."""
        if self.table_spec is not None and pid in self.table_spec.touch_points:
            x, y = self.table_spec.touch_points[pid]
            return "board", np.array([x, y, 0.0]), f"point {pid} (x {x * 1e3:+.1f}, y {y * 1e3:+.1f} mm on the board)"
        kind = self.stickers.kind(pid)
        if kind:
            return kind, np.zeros(3), f"{kind} sticker {pid}" + (" (held-out check point)" if kind == "check" else "")
        return None, None, ""

    def _srv_capture_touch(self, req, resp):
        """Record the TCP of the touching arm resting on a table-board touch point or a free sticker (arm stationary)."""
        board_ids = sorted(self.table_spec.touch_points) if self.table_spec is not None else []
        if not board_ids and not self.stickers.all_ids():
            return self._fail(resp, "the targets file has no table_board with touch_points and no touch_stickers")
        arm = self.get_parameter("touch_arm").value or (self.active_arms[0] if len(self.active_arms) == 1 else "")
        if arm not in self.active_arms:
            return self._fail(resp, f"set parameter touch_arm to one of {self.active_arms} (got '{arm}')")
        pid = int(self.get_parameter("touch_point_id").value)
        kind, p_board, desc = self._touch_target(pid)
        if kind is None:
            return self._fail(resp, f"set parameter touch_point_id to one of board points {board_ids} or sticker ids "
                                    f"{self.stickers.all_ids()} (got {pid})")
        since, why = self.stationary_since()
        if since is None:
            return self._fail(resp, why)
        T, why = self._lookup(self.p[f"{arm}_base_frame"], self.p[f"{arm}_tcp_frame"], True)
        if T is None:
            return self._fail(resp, why)
        # the contemporaneous base -> link6 (same stamp policy) and the link6 -> TCP transform in use: a later TCP
        # change can then be re-based exactly (solve --tcp-file); T_link6_tcp is the static TF, else computed
        T6, why = self._lookup(self.p[f"{arm}_base_frame"], self.p[f"{arm}_tip_frame"], True)
        if T6 is None:
            return self._fail(resp, why)
        T_l6_tcp, _ = self._lookup(self.p[f"{arm}_tip_frame"], self.p[f"{arm}_tcp_frame"], False)
        if T_l6_tcp is None:
            T_l6_tcp = np.linalg.inv(T6) @ T
        prefix = self.p[f"{arm}_joint_prefix"]
        joints = {n: float(v) for n, v in self.joint_hist[-1][1].items() if n.startswith(prefix)}
        if not joints:
            return self._fail(resp, f"no joint values with prefix '{prefix}'")
        # two-frame image check on the table board, if it is visible (the touch itself needs no camera)
        check_note = "no camera image: stillness image check skipped"
        if self.p["corner_check"] and self.image is not None and self.table_spec is not None:
            stamp = self._image_stamp(self.image)
            age = self.get_clock().now().nanoseconds * 1e-9 - stamp
            if age <= self.p["max_image_age_s"] and stamp >= since:
                try:
                    dets = self.detector.detect(image_msg_to_bgr(self.image))
                except ValueError:
                    dets = {}
                td = dets.get("table")
                n_td = 0 if td is None else len(td.points)
                expected = len(self.table_spec.all_points_obj()) - len(self.table_spec.touch_corner_ids())
                if n_td < self.p["min_corners_table"]:
                    check_note = "table board not visible: stillness image check skipped"
                elif n_td < 0.9 * expected:
                    # the corner detection of a partly hidden board (the arm in front of it) jumps by about a pixel
                    check_note = (f"table board partly hidden ({n_td} of {expected} corners): stillness image check "
                                  "skipped")
                else:
                    still, refusal = self._stillness(dets, stamp, since, ["table"])
                    if refusal:
                        return self._fail(resp, refusal)
                    if not still["table"][0]:
                        return self._fail(resp, "table board in the image: " + still["table"][1])
                    check_note = "image check: " + still["table"][1]
            else:
                check_note = "newest image too old / before the stationary start: stillness image check skipped"
        touch = Touch(arm=arm, point_id=pid, p_board=p_board, T_base_tcp=T, joints=joints,
                      stamp=self._now(), tcp_frame=self.p[f"{arm}_tcp_frame"], kind=kind,
                      T_base_link6=T6, T_link6_tcp=T_l6_tcp)
        append_touch(self.session_dir, touch, {"since": since})
        self._update_session_info()
        self.touch_counts[arm] += 1
        self.touched[arm][pid] = self.touched[arm].get(pid, 0) + 1
        n_pt = sum(1 for t in load_touches(self.session_dir) if t.arm == arm and t.point_id == pid)
        resp.success = True
        resp.message = (f"touch recorded: arm {arm}, {desc}, "
                        f"tcp in base xyz [{T[0, 3]:.4f}, {T[1, 3]:.4f}, {T[2, 3]:.4f}] m; point touched {n_pt}x, "
                        f"{self.touch_counts[arm]} touches of {arm} in total; {check_note}")
        self.get_logger().info(resp.message)
        return resp

    def _srv_capture_intrinsics(self, req, resp):
        """Save the full image and the detections of the intrinsics board (handheld, or a second rigidly mounted
        print) for ``calibrate_intrinsics``.  Needs two sharp images >= min_image_gap_s apart: a board that moved
        (blur) is refused.  No arm is involved."""
        if self.image is None or self.info is None:
            return self._fail(resp, "no image / camera_info received yet")
        msg, info = self.image, self.info
        if (info.width, info.height) != (msg.width, msg.height):
            return self._fail(resp, f"image {msg.width}x{msg.height} != camera_info {info.width}x{info.height}")
        stamp = self._image_stamp(msg)
        age = self.get_clock().now().nanoseconds * 1e-9 - stamp
        if age > self.p["max_image_age_s"]:
            return self._fail(resp, f"newest image is {age:.2f} s old (max {self.p['max_image_age_s']:.1f} s)")
        name = self.p["intrinsics_target"]
        try:
            bgr = image_msg_to_bgr(msg)
        except ValueError as e:
            return self._fail(resp, str(e))
        dets = self.detector.detect(bgr)
        det = dets.get(name)
        if det is None:
            return self._fail(resp, f"intrinsics board '{name}' not detected")
        ids, obj, pix = det.arrays()
        if len(ids) < self.p["min_corners_intrinsics"]:
            return self._fail(resp, f"intrinsics board: only {len(ids)} corners (< {self.p['min_corners_intrinsics']})")
        still, refusal = self._stillness(dets, stamp, None, [name])
        if refusal:
            return self._fail(resp, refusal)
        if not still[name][0]:
            return self._fail(resp, f"{name}: {still[name][1]}, rejected (blurred / moving)")
        K = np.array(info.k, dtype=float).reshape(3, 3)       # always camera_info here: this view calibrates it
        dist = np.array(info.d, dtype=float)
        sample = Sample(arm=name, T_base_link6=np.eye(4), ids=ids, obj=obj, pix=pix, K=K, dist=dist,
                        image_size=(msg.width, msg.height), stamp=stamp)
        cands = pnp_candidates(sample)
        if not cands:
            return self._fail(resp, "PnP failed")
        rms, depth = cands[0][1], float(cands[0][0][2, 3])
        if rms > self.p["max_pnp_rms_px"]:
            return self._fail(resp, f"PnP reprojection {rms:.2f} px > {self.p['max_pnp_rms_px']:.1f}, rejected")
        prev_views = load_intrinsics_views(self.session_dir)
        if prev_views:
            last = prev_views[-1]
            a = {int(i): np.array(uv) for i, uv in zip(ids, pix)}
            b = {int(i): np.array(uv) for i, uv in zip(last["ids"], last["pix"])}
            common = set(a) & set(b)
            if len(common) >= 4:
                mean_shift = float(np.mean([np.linalg.norm(a[i] - b[i]) for i in common]))
                if mean_shift < self.p["min_intrinsics_view_shift_px"]:
                    return self._fail(resp, f"same view as the previous one (moved {mean_shift:.1f} px < "
                                            f"{self.p['min_intrinsics_view_shift_px']:.0f}); move the board")
        self.n_intrinsics += 1
        self.intrinsics_views.append((_centre_px(pix), _bbox_px(pix)))
        img_name = f"intrinsics/intr_{self.n_intrinsics:04d}.png"
        os.makedirs(os.path.join(self.session_dir, "intrinsics"), exist_ok=True)
        cv2.imwrite(os.path.join(self.session_dir, img_name), bgr)
        append_intrinsics_view(self.session_dir, {
            "image": img_name, "stamp": stamp, "target": name, "ids": ids.tolist(), "obj": obj.tolist(),
            "pix": pix.tolist(), "K": K.tolist(), "dist": dist.tolist(), "image_size": [msg.width, msg.height],
            "pnp_rms_px": rms, "board_distance_m": depth, "corner_motion_px": still[name][2],
            "corner_motion_max_px": still[name][3]})
        self._update_session_info()
        resp.success = True
        resp.message = (f"intrinsics view #{self.n_intrinsics}: {len(ids)} corners, board at {depth:.2f} m, PnP "
                        f"{rms:.2f} px, {still[name][1]}; need >= 25 views covering the image corners/borders and a "
                        "depth range")
        self.get_logger().info(resp.message)
        return resp


def main(args=None):
    rclpy.init(args=args)
    node = CalibrationRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
