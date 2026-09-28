"""CLI: measure the 2x2 image Jacobian (px per mm) by jogging the robot via
MoveIt Servo while tracking a printed point target, and write it into the
mount YAML.

    ros2 run side_tape_ros calib_jacobian [--step-mm 5] [--speed-mps 0.002] \\
        [--dry-run] [--raw] [--force-save] [--point U V] [--config PATH]

PRECONDITION: the robot is at a safe pose viewing a printed high-contrast dot
target on a FLAT surface, at the same cutting standoff, and the blade is
NOT touching anything. `--point` gives the target pixel in DE-ROTATED image
coordinates; without it, the strongest high-contrast blob near `mount.tip_uv`
(or the image center, if not yet calibrated) is used.

For each of +cut, -cut, +lateral, -lateral (base-frame axes frozen from TF at
the start, same convention as side_tape_controller / control_math.freeze_axes):
jog `step_mm` at `speed_mps` via Servo, settle, average N tracked frames, jog
back to the start (checked against TF, abort on > 1 mm deviation), then move
on to the next axis. Aborts on an active MoveIt trajectory goal or stale Servo
status. Always leaves Servo paused on exit (success or error). `--dry-run`:
no motion at all, just prints the plan.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import rclpy
import yaml
from action_msgs.msg import GoalStatusArray
from geometry_msgs.msg import TwistStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Int8
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener
from tf2_ros import TransformException as Tf2Exception

from . import control_math, core_path  # noqa: F401
from precision_box_detector.side_tape import geometry as geo  # noqa: E402
from precision_box_detector.side_tape.config import load_config  # noqa: E402

_RATE_HZ = 30.0
_SETTLE_S = 0.4
_AVG_FRAMES = 8
_TF_TOLERANCE_MM = 1.0
_TEMPLATE_HALF_PX = 18
_SEARCH_HALF_PX = 80


def _decode_compressed_gray(data: bytes, rotate_180: bool) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("could not decode compressed image")
    return geo.derotate_image(image, rotate_180)


class CalibNode(Node):
    def __init__(self, cfg, raw: bool = False):
        super().__init__("side_tape_calib_jacobian")
        self.cfg = cfg
        self.latest_gray: np.ndarray | None = None
        self.latest_stamp_s: float | None = None
        self.last_servo_status: int | None = None
        self.last_servo_status_at: float | None = None
        self.last_goal_statuses: list[int] | None = None

        self.image_topic = (cfg.camera.image_topic if raw else
                            cfg.camera.image_topic.rstrip("/") + "/compressed")
        self._bridge = None
        if raw:
            try:
                from cv_bridge import CvBridge
                self._bridge = CvBridge()
            except ImportError:
                pass

        image_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                               reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Image if raw else CompressedImage, self.image_topic,
                                 self._on_image if raw else self._on_compressed, image_qos)
        self.create_subscription(Int8, cfg.robot.servo_status_topic, self._on_servo_status, 10)
        traj_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                              durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(GoalStatusArray, f"{cfg.robot.trajectory_action}/_action/status",
                                 self._on_traj_status, traj_qos)
        self.pub_twist = self.create_publisher(TwistStamped, cfg.robot.servo_twist_topic, 10)
        self.cli_start = self.create_client(Trigger, f"{cfg.robot.servo_ns}/start_servo")
        self.cli_pause = self.create_client(Trigger, f"{cfg.robot.servo_ns}/pause_servo")
        self.cli_unpause = self.create_client(Trigger, f"{cfg.robot.servo_ns}/unpause_servo")

        self.tf_buffer = Buffer(cache_time=Duration(seconds=2.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def _on_image(self, msg: Image) -> None:
        try:
            if self._bridge is not None:
                img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
                gray = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_BGR2GRAY)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"bad image: {exc}")
            return
        gray = geo.derotate_image(gray, self.cfg.camera.rotate_180)
        self.latest_gray = gray
        self.latest_stamp_s = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def _on_compressed(self, msg: CompressedImage) -> None:
        try:
            gray = _decode_compressed_gray(msg.data, self.cfg.camera.rotate_180)
        except ValueError as exc:
            self.get_logger().warn(f"bad compressed image: {exc}")
            return
        self.latest_gray = gray
        self.latest_stamp_s = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def _on_servo_status(self, msg: Int8) -> None:
        self.last_servo_status = int(msg.data)
        self.last_servo_status_at = time.monotonic()

    def _on_traj_status(self, msg: GoalStatusArray) -> None:
        self.last_goal_statuses = [int(s.status) for s in msg.status_list]

    def now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def lookup_tf_m(self) -> np.ndarray | None:
        try:
            tf = self.tf_buffer.lookup_transform(self.cfg.robot.base_frame, self.cfg.robot.tool_frame, Time())
        except Tf2Exception:
            return None
        t = tf.transform.translation
        return np.array([t.x, t.y, t.z], dtype=np.float64)

    def lookup_rotation(self):
        tf = self.tf_buffer.lookup_transform(self.cfg.robot.base_frame, self.cfg.robot.tool_frame, Time())
        q = tf.transform.rotation
        return control_math.quat_to_rotation_matrix(q.x, q.y, q.z, q.w)

    def call_trigger(self, client, name: str, timeout_sec: float = 3.0) -> tuple[bool, str]:
        if not client.wait_for_service(timeout_sec=2.0):
            return False, f"{name}_unavailable"
        future = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_sec)
        if not future.done():
            return False, f"{name}_timeout"
        res = future.result()
        if res is None:
            return False, f"{name}_no_response"
        return bool(res.success), str(res.message)


def _wait_for_image(node: CalibNode, timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while rclpy.ok() and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        if node.latest_gray is not None:
            return True
    return False


def _pick_target_point(node: CalibNode, args) -> tuple[float, float]:
    if args.point is not None:
        return float(args.point[0]), float(args.point[1])
    gray = node.latest_gray
    h, w = gray.shape
    if node.cfg.mount.tip_uv is not None:
        cu, cv_ = node.cfg.mount.tip_uv
    else:
        cu, cv_ = w / 2.0, h / 2.0
    half = min(200, w // 2, h // 2)
    x0, y0 = int(max(0, cu - half)), int(max(0, cv_ - half))
    x1, y1 = int(min(w, cu + half)), int(min(h, cv_ + half))
    roi = gray[y0:y1, x0:x1]
    blur = cv2.GaussianBlur(roi, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("no high-contrast blob found near the search center; pass --point u v")
    best = max(contours, key=cv2.contourArea)
    m = cv2.moments(best)
    if m["m00"] == 0:
        raise RuntimeError("degenerate blob moments; pass --point u v")
    u = x0 + m["m10"] / m["m00"]
    v = y0 + m["m01"] / m["m00"]
    return float(u), float(v)


def _extract_template(gray: np.ndarray, u: float, v: float, half: int = _TEMPLATE_HALF_PX) -> np.ndarray:
    x0, y0 = int(round(u - half)), int(round(v - half))
    x1, y1 = x0 + 2 * half + 1, y0 + 2 * half + 1
    if x0 < 0 or y0 < 0 or x1 > gray.shape[1] or y1 > gray.shape[0]:
        raise RuntimeError("target point too close to the image edge for a template")
    return gray[y0:y1, x0:x1].copy()


def _match_subpixel(gray: np.ndarray, template: np.ndarray, center: tuple, half: int = _SEARCH_HALF_PX):
    """Search for `template`'s center within `half` px of `center`.

    matchTemplate only slides the template's top-left corner within the ROI
    (no padding), so for the template's *center* to reach `half` px away from
    `center` in any direction, the ROI must extend `half + template_half`
    past `center` on every side -- not just `half` (an ROI padded by only
    `half` silently clips off exactly the large-displacement matches this is
    for, and reports a degenerate near-zero score instead of failing loudly).
    """
    th, tw = template.shape
    cu, cv_ = center
    pad_x = half + tw // 2 + 1
    pad_y = half + th // 2 + 1
    x0 = int(max(0, cu - pad_x))
    y0 = int(max(0, cv_ - pad_y))
    x1 = int(min(gray.shape[1], cu + pad_x))
    y1 = int(min(gray.shape[0], cv_ + pad_y))
    roi = np.ascontiguousarray(gray[y0:y1, x0:x1])
    if roi.shape[0] < th or roi.shape[1] < tw:
        return None
    res = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    mx, my = max_loc

    # TM_CCOEFF_NORMED normalizes by the local window's std; over a flat
    # (near zero-variance) patch that division is near-singular and can
    # report a spuriously high (even ~1.0) score for a location that does not
    # actually resemble the template at all (most of a small bright-blob
    # template is background-colored, so a plain intensity-difference check
    # against a flat patch is fooled the same way). Reject any match whose
    # own local contrast is far below the template's -- a flat/textureless
    # patch cannot really contain the templated feature.
    patch = roi[my:my + th, mx:mx + tw]
    if patch.shape != template.shape:
        return None
    patch_std = float(np.std(patch))
    template_std = float(np.std(template))
    if template_std > 1e-6 and patch_std < 0.5 * template_std:
        return None

    def _parab(a: float, b: float, c: float) -> float:
        denom = a - 2.0 * b + c
        return 0.0 if abs(denom) < 1e-9 else 0.5 * (a - c) / denom

    dx = _parab(res[my, mx - 1], res[my, mx], res[my, mx + 1]) if 0 < mx < res.shape[1] - 1 else 0.0
    dy = _parab(res[my - 1, mx], res[my, mx], res[my + 1, mx]) if 0 < my < res.shape[0] - 1 else 0.0
    peak_u = x0 + mx + dx + tw / 2.0
    peak_v = y0 + my + dy + th / 2.0
    return peak_u, peak_v, float(max_val)


def _average_tracked_position(node: CalibNode, template: np.ndarray, search_center, n: int = _AVG_FRAMES):
    samples = []
    last_stamp = None
    deadline = time.monotonic() + 5.0
    while len(samples) < n and rclpy.ok() and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        _check_jog_safety(node)
        if node.latest_gray is None or node.latest_stamp_s == last_stamp:
            continue
        last_stamp = node.latest_stamp_s
        m = _match_subpixel(node.latest_gray, template, search_center)
        if m is None:
            continue
        u, v, score = m
        if score < 0.5:
            continue
        samples.append((u, v))
    if len(samples) < max(3, n // 2):
        raise RuntimeError(f"lost the template (only {len(samples)}/{n} good frames)")
    arr = np.asarray(samples)
    return arr.mean(axis=0), arr.std(axis=0)


def _servo_status_is_halt(value: int | None) -> bool:
    return control_math.map_servo_status_int(value) == "HALT"


def _jog_safety_reason(goal_statuses: list[int] | None, servo_status: int | None,
                       status_age_s: float | None) -> str | None:
    if goal_statuses is None:
        return "no MoveIt trajectory status received; ownership is unknown"
    if control_math.trajectory_goal_active(goal_statuses):
        return "MoveIt trajectory goal became active during calibration"
    if status_age_s is None or status_age_s > 1.0:
        return "no Servo status received or last status older than 1 s during calibration"
    if _servo_status_is_halt(servo_status):
        return "Servo reported HALT during calibration"
    return None


def _check_jog_safety(node: CalibNode) -> None:
    """Fail closed before every motion command and at each jog boundary."""
    age = None if node.last_servo_status_at is None else time.monotonic() - node.last_servo_status_at
    reason = _jog_safety_reason(node.last_goal_statuses, node.last_servo_status, age)
    if reason is not None:
        _publish_zero(node)
        ok, msg = node.call_trigger(node.cli_pause, "pause_servo")
        raise RuntimeError(reason + ("" if ok else f"; pause_servo failed: {msg}"))


def _jog(node: CalibNode, axis_base: np.ndarray, distance_mm: float, speed_mps: float,
         p_ref_m: np.ndarray) -> None:
    """Closed-loop (by TF) jog of `distance_mm` (signed) along `axis_base`, from
    wherever the tool currently is. Raises if ownership, Servo status, or TF is unsafe."""
    sign = 1.0 if distance_mm >= 0 else -1.0
    target_mm = abs(distance_mm)
    v_cmd = axis_base * sign * speed_mps
    rclpy.spin_once(node, timeout_sec=0)
    _check_jog_safety(node)
    p_start = node.lookup_tf_m()
    if p_start is None:
        raise RuntimeError("TF lookup failed at the start of a jog")
    period = 1.0 / _RATE_HZ
    # A tight-ish bound on top of the TF-distance stopping check: if this
    # process (or TF/Servo) is so delayed that the jog takes much longer than
    # nominal, stop trusting it and abort loudly rather than silently letting
    # the arm keep moving on an ever-larger overshoot margin.
    max_duration_s = (target_mm / 1000.0) / max(speed_mps, 1e-6) * 1.5 + 0.5
    t_deadline = time.monotonic() + max_duration_s
    while True:
        _check_jog_safety(node)
        p_now = node.lookup_tf_m()
        if p_now is None:
            _publish_zero(node)
            raise RuntimeError("TF lookup failed during a jog")
        travelled_mm = float((p_now - p_start) @ axis_base) * sign * 1000.0
        # Stop ~one loop period + TF latency early; the measured displacement
        # (not the commanded one) is what the J fit uses.
        if travelled_mm >= target_mm - speed_mps * 1000.0 * 0.08 or time.monotonic() > t_deadline:
            break
        stamp = node.get_clock().now().to_msg()
        msg = TwistStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = node.cfg.robot.base_frame
        msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z = (float(x) for x in v_cmd)
        node.pub_twist.publish(msg)
        rclpy.spin_once(node, timeout_sec=period)
    _publish_zero(node)
    _check_jog_safety(node)
    if time.monotonic() > t_deadline:
        raise RuntimeError("jog exceeded its time budget without reaching the target distance")

    p_end = node.lookup_tf_m()
    if p_end is not None:
        drift_from_ref_mm = float(np.linalg.norm(p_end - p_ref_m)) * 1000.0
        node.get_logger().debug(f"jog end: {drift_from_ref_mm:.3f} mm from reference")


def _publish_zero(node: CalibNode) -> None:
    msg = TwistStamped()
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.header.frame_id = node.cfg.robot.base_frame
    node.pub_twist.publish(msg)


def _check_back_at_start(node: CalibNode, p0_m: np.ndarray, label: str) -> None:
    p_now = node.lookup_tf_m()
    if p_now is None:
        raise RuntimeError(f"TF lookup failed while verifying return-to-start after {label}")
    dev_mm = float(np.linalg.norm(p_now - p0_m)) * 1000.0
    if dev_mm > _TF_TOLERANCE_MM:
        raise RuntimeError(f"return-to-start after {label} deviated {dev_mm:.2f} mm "
                            f"(> {_TF_TOLERANCE_MM} mm tolerance)")


def _update_mount_yaml(config_path: str, J: np.ndarray, set_calibrated: bool,
                       residual_px: float, calibrated_at: datetime | None = None) -> Path:
    calibrated_at = calibrated_at or datetime.now(timezone.utc)
    path = Path(config_path)
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    mount = data.setdefault("mount", {})
    mount["J"] = [[float(v) for v in row] for row in J]
    if set_calibrated:
        mount["calibrated"] = True
    backup = path.with_name(path.name + ".bak-" + calibrated_at.strftime("%Y%m%dT%H%M%S%fZ"))
    shutil.copy2(path, backup)
    print(f"backup: {backup}")
    with path.open("w", encoding="utf-8") as fh:
        fh.write(f"# J calibrated {calibrated_at.isoformat()} residual {residual_px:.3f} px\n")
        yaml.safe_dump(data, fh, sort_keys=False)
    return backup


def _fit_axis_column(samples: list[tuple[float, float]]) -> tuple[np.ndarray, float]:
    """Least-squares slope (px per mm) through the origin, over
    ``samples = [(dx_mm, du_px, dv_px), ...]``. Returns (column, rms_residual_px)."""
    dx = np.array([s[0] for s in samples])
    du = np.array([s[1] for s in samples])
    dv = np.array([s[2] for s in samples])
    denom = float(dx @ dx)
    m_u = float(dx @ du) / denom
    m_v = float(dx @ dv) / denom
    resid_u = du - m_u * dx
    resid_v = dv - m_v * dx
    rms = float(np.sqrt(np.mean(np.concatenate([resid_u, resid_v]) ** 2)))
    return np.array([m_u, m_v]), rms


def _save_rejection_reasons(J: np.ndarray, cut_rms: float, lat_rms: float) -> list[str]:
    reasons = []
    for label, rms in (("cut", cut_rms), ("lateral", lat_rms)):
        if not np.isfinite(rms) or rms > 1.0:
            reasons.append(f"{label} RMS residual {rms:.3f} px exceeds 1.0 px")
    for label, col in (("cut", J[:, 0]), ("lateral", J[:, 1])):
        magnitude = float(np.linalg.norm(col))
        if not np.isfinite(magnitude) or magnitude < 0.5:
            reasons.append(f"{label} |J| {magnitude:.3f} px/mm is below 0.5 px/mm")
    return reasons


def _should_save(J: np.ndarray, cut_rms: float, lat_rms: float, force_save: bool) -> bool:
    if not np.all(np.isfinite(J)) or not np.all(np.isfinite((cut_rms, lat_rms))):
        return False
    return force_save or not _save_rejection_reasons(J, cut_rms, lat_rms)


def _run(node: CalibNode, args) -> int:
    if node.last_goal_statuses is not None and control_math.trajectory_goal_active(node.last_goal_statuses):
        print("REFUSED: a MoveIt trajectory goal is active; wait for it to finish first.")
        return 1

    if not _wait_for_image(node, timeout_s=5.0):
        if args.point is None:
            print("REFUSED: no image received on "
                  f"{node.image_topic} within 5 s (needed to pick/track the target).")
            return 1
        print(f"WARNING: no image received yet on {node.image_topic}; "
              "continuing with --point only for the dry-run plan.")

    if node.latest_gray is not None:
        u0, v0 = _pick_target_point(node, args)
        template = _extract_template(node.latest_gray, u0, v0)
        print(f"target pixel (de-rotated): ({u0:.1f}, {v0:.1f}), template {template.shape[::-1]} px")
    else:
        u0 = v0 = None
        template = None

    try:
        R = node.lookup_rotation()
    except Tf2Exception as exc:
        print(f"REFUSED: TF lookup failed for {node.cfg.robot.base_frame}->{node.cfg.robot.tool_frame}: {exc}")
        return 1
    cut_b, lat_b, _depth_b = control_math.freeze_axes(
        R, node.cfg.robot.cut_axis_tool, node.cfg.robot.lateral_axis_tool, node.cfg.robot.depth_axis_tool)
    p0 = node.lookup_tf_m()

    plan = [("+cut", cut_b, args.step_mm), ("-cut", cut_b, -args.step_mm),
            ("+lateral", lat_b, args.step_mm), ("-lateral", lat_b, -args.step_mm)]
    print("J is valid only at the same camera-to-surface distance as cutting.")
    print(f"plan: step={args.step_mm} mm, speed={args.speed_mps} m/s, base_frame axes "
          f"cut={cut_b.round(3).tolist()} lateral={lat_b.round(3).tolist()}")
    for name, axis, dist in plan:
        print(f"  jog {name}: {dist:+.1f} mm along {axis.round(3).tolist()}, "
              f"track {_AVG_FRAMES} frames, return to start (<= {_TF_TOLERANCE_MM} mm)")

    if args.dry_run:
        print("--dry-run: no motion performed.")
        return 0

    if template is None or u0 is None:
        print("REFUSED: no image/target available; cannot calibrate without --dry-run.")
        return 1

    deadline = time.monotonic() + 5.0
    while node.last_goal_statuses is None and rclpy.ok() and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    if node.last_goal_statuses is None:
        print("REFUSED: no MoveIt trajectory status received; ownership is unknown.")
        return 1
    if control_math.trajectory_goal_active(node.last_goal_statuses):
        print("REFUSED: a MoveIt trajectory goal is active; cannot start Servo.")
        return 1
    ok, msg = node.call_trigger(node.cli_start, "start_servo")
    if not ok:
        print(f"REFUSED: start_servo failed: {msg}")
        _publish_zero(node)
        node.call_trigger(node.cli_pause, "pause_servo")
        return 1
    if control_math.trajectory_goal_active(node.last_goal_statuses):
        _publish_zero(node)
        node.call_trigger(node.cli_pause, "pause_servo")
        print("ABORTED: MoveIt trajectory goal became active before Servo unpause.")
        return 1
    ok, msg = node.call_trigger(node.cli_unpause, "unpause_servo")
    if not ok:
        print(f"REFUSED: unpause_servo failed: {msg}")
        node.call_trigger(node.cli_pause, "pause_servo")
        return 1

    exit_code = 0
    cut_samples: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)]
    lat_samples: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)]
    try:
        for name, axis, dist in plan:
            print(f"jogging {name} ({dist:+.1f} mm)...")
            _jog(node, axis, dist, args.speed_mps, p0)
            time.sleep(_SETTLE_S)
            p_meas = node.lookup_tf_m()
            if p_meas is None:
                raise RuntimeError("TF lookup failed after a jog")
            # Fit on the MEASURED displacement along the axis, not the commanded one:
            # any overshoot/undershoot would otherwise scale J directly.
            dist_meas = float((p_meas - p0) @ axis) * 1000.0
            print(f"  commanded {dist:+.2f} mm, measured {dist_meas:+.2f} mm along axis")
            (u, v), (su, sv) = _average_tracked_position(node, template, (u0, v0))
            _check_jog_safety(node)
            print(f"  tracked at ({u:.2f}, {v:.2f}) px, jitter std ({su:.2f}, {sv:.2f}) px")
            du, dv = u - u0, v - v0
            if name.endswith("cut"):
                cut_samples.append((dist_meas, du, dv))
            else:
                lat_samples.append((dist_meas, du, dv))
            print(f"  returning to start...")
            _jog(node, axis, -dist, args.speed_mps, p0)
            time.sleep(_SETTLE_S)
            _check_back_at_start(node, p0, name)

        cut_col, cut_rms = _fit_axis_column(cut_samples)
        lat_col, lat_rms = _fit_axis_column(lat_samples)
        J = np.column_stack([cut_col, lat_col])
        print(f"fitted J =\n{J}")
        print(f"residuals (px RMS): cut={cut_rms:.3f} lateral={lat_rms:.3f}")
        if not np.all(np.isfinite(J)) or not np.all(np.isfinite((cut_rms, lat_rms))):
            raise RuntimeError("non-finite Jacobian or residual; refusing to write YAML even with --force-save")
        reasons = _save_rejection_reasons(J, cut_rms, lat_rms)
        if reasons:
            for reason in reasons:
                print(f"UNSAFE FIT: {reason}")
        if not _should_save(J, cut_rms, lat_rms, args.force_save):
            raise RuntimeError("refusing to write YAML; --force-save overrides the fit gate")
        if reasons:
            print("WARNING: --force-save overriding the fit gate")
        set_calibrated = node.cfg.mount.tip_uv is not None
        _update_mount_yaml(args.config, J, set_calibrated, max(cut_rms, lat_rms))
        print(f"wrote mount.J to {args.config} (calibrated={set_calibrated}"
              + ("" if set_calibrated else "; tip_uv not set, calibrated left as-is") + ")")
    except Exception as exc:  # noqa: BLE001
        print(f"ABORTED: {exc}")
        _publish_zero(node)
        exit_code = 1
    finally:
        _publish_zero(node)
        ok, msg = node.call_trigger(node.cli_pause, "pause_servo")
        if not ok:
            print(f"WARNING: pause_servo failed on exit: {msg}")
            exit_code = 1
    return exit_code


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--step-mm", type=float, default=5.0)
    parser.add_argument("--speed-mps", type=float, default=0.002)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--raw", action="store_true", help="subscribe to raw sensor_msgs/Image instead of JPEG")
    parser.add_argument("--force-save", action="store_true", help="write J despite unsafe fit residual/scale")
    parser.add_argument("--point", type=float, nargs=2, metavar=("U", "V"), default=None)
    parser.add_argument("--config", type=str, default=core_path.DEFAULT_CONFIG)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    rclpy.init(args=None)
    node = CalibNode(cfg, raw=args.raw)
    try:
        exit_code = _run(node, args)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
