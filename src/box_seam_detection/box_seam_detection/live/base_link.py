#!/usr/bin/env python3
# Provenance: moved into box_seam_detection (2026-10-12) from seam_handoff_20260908/scripts/camera_point_to_base_link.py
# at commit 08d1041 (working tree identical to HEAD, unmodified).
# Changes: manifest driver_extrinsic support; library functions never init/shutdown rclpy (LiveTf only in the CLI).
# -*- coding: utf-8 -*-
"""camera_point_to_base_link.py

驗證用的橋接腳本（第一版，範圍刻意縮小）：把 query_depth_at_pixel.py 算出來、
在「相機座標系」下的 3D 點，透過 TF2 轉換到機械手臂的 base_link 座標系，
並印出轉換結果，方便人工核對數值是否合理。

明確不做的事（之後決定夾爪接近姿態、串 IK/MoveIt 執行時才會做）：
    - 不計算夾爪的目標姿態（orientation），只轉換位置（position）。
    - 不呼叫 MoveIt、不反解 IK、不送出任何動作指令。
這支程式單純是「①③ 之間」的驗證步驟：先確認座標轉換本身正確，再決定怎麼用這個
轉換後的座標去規劃動作。

為什麼來源座標系（--source-frame）預設是 camera_color_optical_frame，不是
camera_link？
----------------------------------------------------------------------
query_depth_at_pixel.py 查出來的 X/Y/Z 是直接從 PCD 點雲的 x/y/z 欄位讀出來的，
而這份點雲是用 rgbd_recorder.py 錄自 /camera/depth_registered/points（見
scripts/README.md「啟動相機」那節）。依 Orbbec 官方文件，這個 topic 是
「registered to the color image frame」，也就是點雲座標系跟彩色相機的
「光學座標系」（optical frame，X 向右、Y 向下、Z 朝前，符合針孔投影模型）
是同一個——這一點 query_depth_at_pixel.py 的反投影公式（proj_u = fx*x/z+cx）
也是照這個假設寫的（見該檔案 query_by_reprojection() 的說明）。在目前這台機器
實際跑起來的 TF 樹裡，符合這個座標慣例、且真的存在的 frame 名稱是
camera_color_optical_frame（不是 camera_link，camera_link 是 driver 自己
定義的機構安裝點，X 向前、Z 向上，跟點雲座標系不是同一個慣例）。

如果你之後改了相機驅動設定（例如換成用 depth 對齊到 color 以外的方式），
或改用別的方式取得點雲，務必重新確認一次點雲實際座標系是哪一個 frame，
再用 --source-frame 覆蓋。

前置條件
--------
執行本程式前，以下兩條 launch 必須都還在跑（提供完整的 TF 樹）：
    1) ros2 launch igus_rebel rebel.launch.py hardware_protocol:=mock_hardware
       （或真實硬體 hardware_protocol:=rebel）—— 提供 world/base_link/... 這條鏈
    2) 相機 driver（gemini_330_series.launch.py，見 scripts/README.md）
       —— 提供 camera_link → camera_color_optical_frame 這條鏈
兩者透過 igus_rebel_robot2.urdf.xacro 裡的 camera_mount_joint 接在一起。

用法
----
    # 轉換 query_depth_at_pixel.py 用 --output-json 存出的整批結果
    python3 camera_point_to_base_link.py --json seam_3d_full.json

    # 另存轉換後的結果
    python3 camera_point_to_base_link.py --json seam_3d_full.json \\
        --output-json seam_3d_base_link.json

輸入 JSON 格式：跟 query_depth_at_pixel.py 的 --output-json 輸出格式相同，
即 {"點名稱": {"x": ..., "y": ..., "z": ..., "valid": true/false, ...}, ...}。
valid 為 false 的點會被跳過（沿用來源的無效判定，不會硬轉換一個本來就不可信的深度值）。

如何判斷轉換結果「看起來對不對」（沒有自動化 ground truth，需要人工核對）：
    - Z 應該接近箱子實際的高度（相對 base_link 所在的地面/桌面）。
    - X/Y 的方向與大致距離，應該與箱子相對機械手臂底座的實際擺放位置一致
      （建議拿捲尺量一次箱子中心大概在手臂正前方多遠、左右偏多少，跟轉換
      結果的數量級和正負號對一下，不需要精確到毫米，但方向不能反）。
    - 同一個箱子上的幾個點（left_mid/right_mid/back_left/...）彼此的相對
      位置關係（例如 Z 應該都差不多高，X/Y 的相對前後左右關係）在 base_link
      座標系下也應該維持合理，跟在相機座標系下看到的關係一致。

2026-09-28 change: camera -> base_link from the hand-eye calibration file
-----------------------------------------------------------------------
USER DECISION: base_link <- camera_link ALWAYS comes from the easy_handeye2
calibration file (--calib-file, default rebel_handeye_h_v2.calib, eye_on_base).
camera_link <- camera_color_optical_frame is the Orbbec driver extrinsic:
    --driver-extrinsic tf        (default) live TF2 lookup camera_link <- optical
    --driver-extrinsic recorded  module constant RECORDED_DRIVER_EXTRINSIC,
                                 offline replay without ROS (rclpy is never
                                 imported in this mode)
T_base_optical = T_calib @ T_driver. If a live base_link <- optical TF is also
available (tf mode) it is only compared against the composed transform and a
WARNING is printed when they differ by > 1 mm / 0.1 deg; it is never used.
The applied 4x4, its provenance (calib file + sha256 + name, driver source) and
frames are written to seam_base_geometry/v1 under "transform".
"""

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# rclpy / tf2_ros are imported lazily (only for --driver-extrinsic tf) so the
# offline replay mode works on machines without ROS 2.

DEFAULT_CALIB_FILE = "/home/farhan/.ros2/easy_handeye2/calibrations/rebel_handeye_h_v2.calib"

# Orbbec driver extrinsic camera_link <- camera_color_optical_frame (static TF
# published by the camera driver). Provenance: recovered 2026-09-28 offline by
# a pooled Kabsch fit of the camera_xyz -> target_xyz pairs of 185 stored
# base_geometry.json runs (captures 2026-09-10 01:44:48 .. 2026-09-17 00:57:43,
# /home/farhan/Igus_rebel/outputs/rgbd_captures/pipeline), whose live TF was
# base_link <- camera_link = rebel_handeye_h_v2.calib; T_driver =
# inv(T_calib) @ T_fit. Fit residual ~1e-15 m, per-run spread ~1e-15 m; i.e.
# t = (1.634, -13.983, -0.043) mm, rpy(xyz) = (-89.919, -0.193, -90.219) deg.
RECORDED_DRIVER_EXTRINSIC = {
    "parent_frame": "camera_link",
    "child_frame": "camera_color_optical_frame",
    "translation": (0.0016341905593874606, -0.013982509613037264, -4.271093383417801e-05),
    "quaternion_xyzw": (-0.499537503542144, 0.4997559393201414, -0.5021479272076848, 0.49855164513834604),
    "provenance": "kabsch fit of 185 stored runs 2026-09-10T01:44..2026-09-17 vs rebel_handeye_h_v2.calib",
}

LIVE_TF_MISMATCH_TRANSLATION_M = 0.001
LIVE_TF_MISMATCH_ROTATION_DEG = 0.1


def quaternion_to_rotation_matrix(x, y, z, w):
    """標準四元數 → 3x3 旋轉矩陣公式（跟寫入 URDF camera_mount_joint 時用的
    是同一套慣例，這裡直接吃 TF 回傳的四元數，不需要再轉成 rpy）。"""
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        raise ValueError("四元數長度接近零，無法正規化")
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array([
        [1.0 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1.0 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1.0 - (xx + yy)],
    ])


def apply_transform(translation, quaternion, point_xyz):
    """point_new = R(q) @ point + t"""
    r = quaternion_to_rotation_matrix(*quaternion)
    t = np.array(translation, dtype=np.float64)
    p = np.array(point_xyz, dtype=np.float64)
    return (r @ p) + t


def transform_matrix(translation, quaternion_xyzw):
    matrix = np.eye(4)
    matrix[:3, :3] = quaternion_to_rotation_matrix(*quaternion_xyzw)
    matrix[:3, 3] = np.asarray(translation, dtype=np.float64)
    return matrix


def rotation_matrix_to_quaternion(rotation):
    """3x3 rotation -> (x, y, z, w) with w >= 0."""
    m = np.asarray(rotation, dtype=np.float64)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = 2.0 * math.sqrt(trace + 1.0)
        q = ((m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s)
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = (0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s)
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = ((m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s)
    else:
        s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = ((m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s)
    q = np.asarray(q, dtype=np.float64)
    q /= np.linalg.norm(q)
    if q[3] < 0.0:
        q = -q
    return tuple(float(value) for value in q)


def transform_difference(first, second):
    """(translation difference m, rotation difference deg) between two 4x4 transforms."""
    translation = float(np.linalg.norm(first[:3, 3] - second[:3, 3]))
    relative = first[:3, :3].T @ second[:3, :3]
    cosine = max(-1.0, min(1.0, (float(np.trace(relative)) - 1.0) / 2.0))
    return translation, math.degrees(math.acos(cosine))


def load_handeye_calibration(path):
    """Parse an easy_handeye2 .calib (YAML) file -> dict with the 4x4
    robot_base_frame <- tracking_base_frame transform and provenance."""
    import yaml  # PyYAML; imported lazily to keep module import light

    path = Path(path)
    raw = path.read_bytes()
    try:
        data = yaml.safe_load(raw.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise ValueError(f"cannot parse calibration file {path}: {error}") from error
    if not isinstance(data, dict):
        raise ValueError(f"calibration file {path} is not a mapping")
    parameters = data.get("parameters")
    transform = data.get("transform")
    if not isinstance(parameters, dict) or not isinstance(transform, dict):
        raise ValueError(f"calibration file {path} lacks parameters/transform sections")
    if parameters.get("calibration_type") != "eye_on_base":
        raise ValueError(
            f"calibration {path} is {parameters.get('calibration_type')!r}; only eye_on_base is supported")
    base_frame = parameters.get("robot_base_frame")
    tracking_frame = parameters.get("tracking_base_frame")
    if not isinstance(base_frame, str) or not base_frame or not isinstance(tracking_frame, str) or not tracking_frame:
        raise ValueError(f"calibration {path} lacks robot_base_frame/tracking_base_frame")
    try:
        translation = tuple(float(transform["translation"][axis]) for axis in ("x", "y", "z"))
        quaternion = tuple(float(transform["rotation"][axis]) for axis in ("x", "y", "z", "w"))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"calibration {path} has an invalid transform: {error}") from error
    if not all(math.isfinite(value) for value in translation + quaternion):
        raise ValueError(f"calibration {path} transform is not finite")
    norm = math.sqrt(sum(value * value for value in quaternion))
    if abs(norm - 1.0) > 1e-3:
        raise ValueError(f"calibration {path} quaternion norm {norm:.6f} is not ~1")
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "name": parameters.get("name"),
        "calibration_type": parameters["calibration_type"],
        "robot_base_frame": base_frame,
        "tracking_base_frame": tracking_frame,
        "translation": translation,
        "quaternion_xyzw": quaternion,
        "matrix": transform_matrix(translation, quaternion),
    }


class LiveTf:
    """Minimal live TF2 access; imports rclpy/tf2_ros only when constructed."""

    def __init__(self, node_name="camera_point_to_base_link_verifier"):
        import rclpy
        from rclpy.node import Node
        import tf2_ros

        self._rclpy = rclpy
        # Only the CLI (own process) owns the rclpy context: never re-init / shut down one
        # that somebody else (e.g. a ROS node hosting this library) already initialised.
        self._owns_context = not rclpy.ok()
        if self._owns_context:
            rclpy.init()
        self._node = Node(node_name)
        self._buffer = tf2_ros.Buffer()
        self._listener = tf2_ros.TransformListener(self._buffer, self._node)

    def lookup(self, target_frame, source_frame, stamp_ns, timeout_sec):
        """4x4 target_frame <- source_frame at the capture stamp; never substitutes latest TF."""
        from rclpy.time import Time as RclpyTime
        from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

        if not isinstance(stamp_ns, int) or stamp_ns <= 0:
            raise ValueError("capture timestamp must be a positive integer nanosecond value")
        stamp = RclpyTime(nanoseconds=stamp_ns)
        deadline = time.monotonic() + timeout_sec
        last_exc = None
        while time.monotonic() < deadline:
            self._rclpy.spin_once(self._node, timeout_sec=0.2)
            try:
                message = self._buffer.lookup_transform(target_frame, source_frame, stamp)
            except (LookupException, ConnectivityException, ExtrapolationException) as exc:
                last_exc = exc
                continue
            t = message.transform.translation
            r = message.transform.rotation
            return transform_matrix((t.x, t.y, t.z), (r.x, r.y, r.z, r.w))
        raise RuntimeError(
            f"cannot resolve capture-time TF {source_frame} -> {target_frame} within "
            f"{timeout_sec:.1f} s (last error: {last_exc})"
        )

    def close(self):
        try:
            self._node.destroy_node()
        finally:
            if self._owns_context and self._rclpy.ok():
                self._rclpy.shutdown()


def _matrix_block(matrix):
    return {
        "matrix": [[float(value) for value in row] for row in matrix],
        "translation": [float(value) for value in matrix[:3, 3]],
        "quaternion_xyzw": list(rotation_matrix_to_quaternion(matrix[:3, :3])),
    }


def driver_extrinsic_from_manifest(manifest_path, expected_parent, expected_child):
    """The ``driver_extrinsic`` block of a capture manifest (written by the ROS capture node:
    same keys as RECORDED_DRIVER_EXTRINSIC + ``source``/``stamp_ns``), or None when the manifest
    has none (today's recorder captures -> the module constant is used). A block with other
    frames or a non-finite / zero quaternion raises ValueError (fail closed)."""
    if not manifest_path:
        return None
    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            block = (json.load(handle) or {}).get("driver_extrinsic")
    except (OSError, ValueError, AttributeError):
        return None
    if block is None:
        return None
    if not isinstance(block, dict):
        raise ValueError("manifest driver_extrinsic is not an object")
    if block.get("parent_frame") != expected_parent or block.get("child_frame") != expected_child:
        raise ValueError(
            f"manifest driver_extrinsic is {block.get('parent_frame')} <- {block.get('child_frame')}, "
            f"need {expected_parent} <- {expected_child}")
    try:
        translation = tuple(float(v) for v in block["translation"])
        quaternion = tuple(float(v) for v in block["quaternion_xyzw"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"manifest driver_extrinsic is invalid: {error}") from error
    if (len(translation) != 3 or len(quaternion) != 4
            or not all(math.isfinite(v) for v in translation + quaternion)):
        raise ValueError("manifest driver_extrinsic is not finite / wrong size")
    norm = math.sqrt(sum(v * v for v in quaternion))
    if abs(norm - 1.0) > 1e-3:
        raise ValueError(f"manifest driver_extrinsic quaternion norm {norm:.6f} is not ~1")
    # q and -q are the same rotation; quaternion_to_rotation_matrix normalises, so the sign
    # of the recorded quaternion cannot matter.
    return {**block, "translation": translation, "quaternion_xyzw": quaternion}


DRIVER_EXTRINSIC_POLICIES = ("refuse", "warn", "use_manifest")


def compare_driver_block_to_constant(driver_block):
    """(translation diff m, rotation diff deg) between a manifest driver_extrinsic block and the
    recorded constant (rotation compared as rotations, so q and -q agree)."""
    rec = RECORDED_DRIVER_EXTRINSIC
    return transform_difference(
        transform_matrix(rec["translation"], rec["quaternion_xyzw"]),
        transform_matrix(driver_block["translation"], driver_block["quaternion_xyzw"]))


def driver_extrinsic_check(driver_block, policy="refuse"):
    """Decide what to do with a manifest driver_extrinsic block.

    Returns the comparison dict {translation_diff_m, rotation_diff_deg, tolerance_m, tolerance_deg,
    exceeds_tolerance, policy, used}. Raises ValueError("driver_extrinsic_mismatch(...)") when the block
    differs from the recorded constant by > 1 mm / 0.1 deg and ``policy`` is ``refuse`` (fail closed).
    ``warn`` and ``use_manifest`` use the manifest block (``warn`` also logs)."""
    if policy not in DRIVER_EXTRINSIC_POLICIES:
        raise ValueError(f"unknown driver_extrinsic_policy {policy!r}")
    dt, dr = compare_driver_block_to_constant(driver_block)
    exceeds = dt > LIVE_TF_MISMATCH_TRANSLATION_M or dr > LIVE_TF_MISMATCH_ROTATION_DEG
    info = {"translation_diff_m": dt, "rotation_diff_deg": dr,
            "tolerance_m": LIVE_TF_MISMATCH_TRANSLATION_M, "tolerance_deg": LIVE_TF_MISMATCH_ROTATION_DEG,
            "exceeds_tolerance": bool(exceeds), "policy": policy, "used": "manifest"}
    if exceeds and policy == "refuse":
        raise ValueError(
            f"driver_extrinsic_mismatch(translation={dt * 1000:.3f}mm, rotation={dr:.4f}deg; manifest "
            f"{driver_block.get('source')} vs recorded constant, limit {LIVE_TF_MISMATCH_TRANSLATION_M * 1000:g}mm/"
            f"{LIVE_TF_MISMATCH_ROTATION_DEG:g}deg; policy=refuse)")
    if exceeds and policy == "warn":
        print(f"WARNING: driver_extrinsic_mismatch(translation={dt * 1000:.3f}mm, rotation={dr:.4f}deg) "
              f"- using the manifest block (policy=warn)", file=sys.stderr)
    return info


def resolve_camera_to_base(calib_file, driver_extrinsic, target_frame, source_frame, stamp_ns,
                           timeout_sec=5.0, tf_provider=None, log=None, driver_block=None,
                           driver_policy="refuse"):
    """Compose target_frame <- source_frame = T_calib @ T_driver.

    driver_extrinsic: "tf" (live lookup tracking_frame <- source_frame through
    tf_provider) or "recorded" (``driver_block`` = the capture manifest's driver_extrinsic
    block when given, else RECORDED_DRIVER_EXTRINSIC; no ROS).
    Returns (matrix_4x4, transform_block) where transform_block is written to
    seam_base_geometry/v1["transform"].
    """
    log = log or (lambda message: print(message, file=sys.stderr))
    calibration = load_handeye_calibration(calib_file)
    if calibration["robot_base_frame"] != target_frame:
        raise ValueError(
            f"calibration robot_base_frame {calibration['robot_base_frame']!r} != target frame {target_frame!r}")
    tracking_frame = calibration["tracking_base_frame"]
    live_full = None
    driver_comparison = None
    if driver_extrinsic == "recorded":
        recorded = driver_block if driver_block is not None else RECORDED_DRIVER_EXTRINSIC
        if driver_block is not None:
            driver_comparison = driver_extrinsic_check(driver_block, driver_policy)   # may refuse
        if recorded["parent_frame"] != tracking_frame or recorded["child_frame"] != source_frame:
            raise ValueError(
                f"recorded driver extrinsic is {recorded['parent_frame']} <- {recorded['child_frame']}, "
                f"need {tracking_frame} <- {source_frame}")
        driver = transform_matrix(recorded["translation"], recorded["quaternion_xyzw"])
        driver_source = "recorded"
    elif driver_extrinsic == "tf":
        if tf_provider is None:
            raise ValueError("--driver-extrinsic tf requires a live TF provider")
        driver = tf_provider.lookup(tracking_frame, source_frame, stamp_ns, timeout_sec)
        driver_source = "tf_live"
        recorded = RECORDED_DRIVER_EXTRINSIC
        if recorded["parent_frame"] == tracking_frame and recorded["child_frame"] == source_frame:
            dt, dr = transform_difference(
                driver, transform_matrix(recorded["translation"], recorded["quaternion_xyzw"]))
            if dt > LIVE_TF_MISMATCH_TRANSLATION_M or dr > LIVE_TF_MISMATCH_ROTATION_DEG:
                log(f"WARNING: live driver extrinsic {tracking_frame} <- {source_frame} differs from the "
                    f"recorded constant by {dt * 1000:.3f} mm / {dr:.4f} deg (using the live one)")
    else:
        raise ValueError(f"unknown driver extrinsic source {driver_extrinsic!r}")
    composed = calibration["matrix"] @ driver

    comparison = {"available": False}
    if tf_provider is not None:
        try:
            live_full = tf_provider.lookup(target_frame, source_frame, stamp_ns, min(timeout_sec, 1.0))
        except (RuntimeError, ValueError) as error:
            comparison = {"available": False, "reason": str(error)}
    if live_full is not None:
        dt, dr = transform_difference(composed, live_full)
        exceeds = dt > LIVE_TF_MISMATCH_TRANSLATION_M or dr > LIVE_TF_MISMATCH_ROTATION_DEG
        comparison = {
            "available": True, "translation_diff_m": dt, "rotation_diff_deg": dr,
            "tolerance_m": LIVE_TF_MISMATCH_TRANSLATION_M, "tolerance_deg": LIVE_TF_MISMATCH_ROTATION_DEG,
            "exceeds_tolerance": bool(exceeds), "used": False,
        }
        if exceeds:
            log(f"WARNING: live TF {target_frame} <- {source_frame} differs from the calibration-composed "
                f"transform by {dt * 1000:.3f} mm / {dr:.4f} deg; using the calibration-composed transform "
                f"({calibration['path']})")

    block = {
        **_matrix_block(composed),
        "frames": {
            "parent_frame": target_frame,
            "child_frame": source_frame,
            "calibration_tracking_frame": tracking_frame,
        },
        "source": {
            "method": "calib_file_composed_with_driver_extrinsic",
            "calib_file": calibration["path"],
            "calib_sha256": calibration["sha256"],
            "calibration_name": calibration["name"],
            "calibration_type": calibration["calibration_type"],
            "calibration_transform": _matrix_block(calibration["matrix"]),
            "driver_extrinsic_source": driver_source,
            "driver_extrinsic": _matrix_block(driver),
            "live_full_tf_comparison": comparison,
        },
    }
    if driver_extrinsic == "recorded" and driver_block is not None:
        block["source"]["driver_extrinsic_origin"] = "capture_manifest"
        block["source"]["driver_extrinsic_manifest_source"] = driver_block.get("source")
        block["source"]["driver_extrinsic_comparison"] = driver_comparison
    return composed, block


PLANNER_POINTS = (
    "left_mid", "right_mid", "back_left", "back_right", "front_left", "front_right",
)


def _as_positive_int(value, label):
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def load_camera_geometry(path):
    with open(path, "r", encoding="utf-8") as input_file:
        data = json.load(input_file)
    if data.get("schema") != "seam_camera_geometry/v1":
        raise ValueError("planner conversion requires seam_camera_geometry/v1 input")
    capture = data.get("capture")
    surface = data.get("surface")
    results = data.get("results")
    if not isinstance(capture, dict) or not isinstance(surface, dict) or not isinstance(results, dict):
        raise ValueError("camera geometry must contain capture, surface, and results objects")
    capture_id = capture.get("capture_id")
    source_frame = capture.get("source_frame")
    if not isinstance(capture_id, str) or not capture_id or not isinstance(source_frame, str) or not source_frame:
        raise ValueError("camera geometry is missing capture_id or source_frame")
    image_stamp_ns = _as_positive_int(capture.get("image_stamp_ns"), "image_stamp_ns")
    pointcloud_stamp_ns = _as_positive_int(capture.get("pointcloud_stamp_ns"), "pointcloud_stamp_ns")
    if abs(image_stamp_ns - pointcloud_stamp_ns) > 50_000_000:
        raise ValueError("camera geometry image and pointcloud timestamps differ by more than 50 ms")
    normal = np.asarray(surface.get("plane_normal"), dtype=np.float64)
    rms = surface.get("plane_rms_m")
    if normal.shape != (3,) or not np.all(np.isfinite(normal)) or np.linalg.norm(normal) <= 1e-9:
        raise ValueError("camera geometry has an invalid measured plane normal")
    if not isinstance(rms, (int, float)) or not np.isfinite(rms) or rms < 0.0:
        raise ValueError("camera geometry has an invalid measured plane RMS")
    return data, results, capture, normal / np.linalg.norm(normal), float(rms)


def apply_rotation(quaternion, vector_xyz):
    return quaternion_to_rotation_matrix(*quaternion) @ np.asarray(vector_xyz, dtype=np.float64)


def _target_point(results, name, target_frame):
    entry = results.get(name)
    if not isinstance(entry, dict) or not entry.get("valid") or entry.get("target_frame") != target_frame:
        raise ValueError(f"missing independently valid {name} in {target_frame}")
    point = np.asarray(entry.get("target_xyz"), dtype=np.float64)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError(f"{name} target_xyz is invalid")
    return point

def build_planner_snapshot(results, input_path, target_frame, capture, target_normal, plane_rms_m):
    """Build the strict, fresh base_link snapshot consumed by box_h_cut_executor."""
    if target_frame != "base_link":
        raise ValueError("planner snapshots must use target frame base_link")
    points = {name: _target_point(results, name, target_frame) for name in PLANNER_POINTS}
    normal = np.asarray(target_normal, dtype=np.float64)
    if normal.shape != (3,) or not np.all(np.isfinite(normal)) or np.linalg.norm(normal) <= 1e-9:
        raise ValueError("transformed measured plane normal is invalid")
    normal /= np.linalg.norm(normal)
    if normal[2] < 0.0:
        normal = -normal
    seam_start = points["left_mid"]
    seam_end = points["right_mid"]
    return {
        "schema": "box_h_cut_snapshot/v1",
        "snapshot_id": f"capture_{capture['capture_id']}",
        "capture_id": capture["capture_id"],
        "acquisition_stamp_ns": capture["image_stamp_ns"],
        "frame_id": target_frame,
        "source_frame": capture["source_frame"],
        "source_file": str(Path(input_path).resolve()),
        "seam_start": dict(zip(("x", "y", "z"), map(float, seam_start))),
        "seam_end": dict(zip(("x", "y", "z"), map(float, seam_end))),
        "seam_center": dict(zip(("x", "y", "z"), map(float, (seam_start + seam_end) / 2.0))),
        "back_left": dict(zip(("x", "y", "z"), map(float, points["back_left"]))),
        "back_right": dict(zip(("x", "y", "z"), map(float, points["back_right"]))),
        "front_left": dict(zip(("x", "y", "z"), map(float, points["front_left"]))),
        "front_right": dict(zip(("x", "y", "z"), map(float, points["front_right"]))),
        "surface_normal": dict(zip(("x", "y", "z"), map(float, normal))),
        "plane_rms_m": float(plane_rms_m),
    }


def transform_results(source_results, source_frame, target_frame, matrix, camera_normal):
    """Apply the 4x4 target <- source transform to every valid camera point and the plane normal."""
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    results = {}
    for name, entry in source_results.items():
        if not isinstance(entry, dict) or not entry.get("valid"):
            continue
        point = np.asarray([entry.get("x"), entry.get("y"), entry.get("z")], dtype=np.float64)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise ValueError(f"{name} has invalid camera XYZ")
        transformed = rotation @ point + translation
        results[name] = {
            "valid": True,
            "camera_frame": source_frame,
            "camera_xyz": [float(value) for value in point],
            "target_frame": target_frame,
            "target_xyz": [float(value) for value in transformed],
        }
    target_normal = rotation @ np.asarray(camera_normal, dtype=np.float64)
    return results, target_normal


def write_json_atomically(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", suffix=".tmp", delete=False) as output:
        json.dump(value, output, ensure_ascii=False, indent=2)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
        temporary_path = output.name
    try:
        os.replace(temporary_path, path)
    except Exception:
        Path(temporary_path).unlink(missing_ok=True)
        raise
def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="camera_point_to_base_link.py",
        description="Transform a validated camera geometry capture into a canonical base_link snapshot."
    )
    parser.add_argument("--json", required=True, help="seam_camera_geometry/v1 input JSON")
    parser.add_argument(
        "--source-frame", default=None,
        help="Optional assertion of the source frame; the capture metadata remains authoritative.",
    )
    parser.add_argument("--target-frame", default="base_link")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument(
        "--calib-file", default=DEFAULT_CALIB_FILE,
        help="easy_handeye2 eye_on_base .calib providing base_link <- camera_link (always used)",
    )
    parser.add_argument(
        "--driver-extrinsic", choices=("tf", "recorded"), default="tf",
        help="camera_link <- optical source: live TF lookup (default) or the recorded constant "
             "(offline replay, no ROS required)",
    )
    parser.add_argument(
        "--driver-extrinsic-policy", choices=DRIVER_EXTRINSIC_POLICIES, default="refuse",
        help="--driver-extrinsic recorded with a driver_extrinsic block in the capture manifest: refuse "
             "(default) when it differs from the recorded constant by > 1 mm / 0.1 deg, warn = use the "
             "manifest block and log, use_manifest = use it silently",
    )
    parser.add_argument("--output-json", help="Optional transformed base-frame geometry JSON")
    parser.add_argument(
        "--planner-output-json",
        help="Atomic canonical base_link snapshot for box_h_cut_executor; requires all six valid points.",
    )
    args = parser.parse_args(argv)

    input_path = Path(args.json)
    if not input_path.is_file():
        parser.error(f"input does not exist: {input_path}")
    try:
        geometry, source_results, capture, camera_normal, plane_rms_m = load_camera_geometry(input_path)
        source_frame = capture["source_frame"]
        if args.source_frame and args.source_frame != source_frame:
            raise ValueError(
                f"requested source frame {args.source_frame} does not match capture source frame {source_frame}"
            )
        if args.planner_output_json and args.target_frame != "base_link":
            raise ValueError("planner output must use target frame base_link")
        if not Path(args.calib_file).is_file():
            raise ValueError(f"calibration file does not exist: {args.calib_file}")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"camera geometry validation failed: {error}", file=sys.stderr)
        sys.exit(1)

    tf_provider = None
    try:
        if args.driver_extrinsic == "tf":
            tf_provider = LiveTf()
        driver_block = None
        if args.driver_extrinsic == "recorded":
            calibration_frames = load_handeye_calibration(args.calib_file)
            driver_block = driver_extrinsic_from_manifest(
                capture.get("manifest_path"), calibration_frames["tracking_base_frame"], source_frame)
        matrix, transform_block = resolve_camera_to_base(
            args.calib_file, args.driver_extrinsic, args.target_frame, source_frame,
            capture["image_stamp_ns"], timeout_sec=args.timeout, tf_provider=tf_provider,
            driver_block=driver_block, driver_policy=args.driver_extrinsic_policy,
        )
        results, target_normal = transform_results(source_results, source_frame, args.target_frame,
                                                   matrix, camera_normal)
    except (RuntimeError, ValueError, OSError, ImportError) as error:
        print(f"base-frame conversion failed: {error}", file=sys.stderr)
        if tf_provider is not None:
            tf_provider.close()
        sys.exit(1)
    if tf_provider is not None:
        tf_provider.close()
    translation = transform_block["translation"]
    print(f"camera->{args.target_frame}: calib {transform_block['source']['calib_file']} "
          f"(sha256 {transform_block['source']['calib_sha256'][:12]}), driver extrinsic "
          f"{transform_block['source']['driver_extrinsic_source']}, t = "
          f"({translation[0]:.6f}, {translation[1]:.6f}, {translation[2]:.6f}) m")

    transformed_output = {
        "schema": "seam_base_geometry/v1",
        "capture": capture,
        "surface": {
            "plane_normal": [float(value) for value in target_normal / np.linalg.norm(target_normal)],
            "plane_rms_m": plane_rms_m,
        },
        "results": results,
        "source_file": str(input_path.resolve()),
        "transform": transform_block,
    }
    if args.output_json:
        write_json_atomically(args.output_json, transformed_output)
        print(f"wrote base-frame geometry: {args.output_json}")
    if args.planner_output_json:
        try:
            snapshot = build_planner_snapshot(
                results, input_path, args.target_frame, capture, target_normal, plane_rms_m
            )
        except ValueError as error:
            print(f"planner snapshot rejected: {error}", file=sys.stderr)
            sys.exit(2)
        write_json_atomically(args.planner_output_json, snapshot)
        print(f"wrote canonical planner snapshot: {args.planner_output_json}")


if __name__ == "__main__":
    main()
