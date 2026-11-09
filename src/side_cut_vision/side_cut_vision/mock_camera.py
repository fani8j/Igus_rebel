#!/usr/bin/env python3
"""MOCK ONLY - simulated knife-holder Gemini 305 for the side-cut probe. NOT FOR REAL ROBOTS.

Renders a synthetic carton (the snapshot carton displaced by a HIDDEN offset) from the LIVE mock TF
(world -> arm_<arm>_link6 composed with the hand-eye) and publishes under the real wrist topic names:

  /arm_<arm>/wrist_camera/color/image_raw             sensor_msgs/Image rgb8 1280x800
  /arm_<arm>/wrist_camera/color/image_raw/compressed  sensor_msgs/CompressedImage (what cut_recorder reads)
  /arm_<arm>/wrist_camera/depth/image_raw             sensor_msgs/Image 16UC1 mm, aligned to colour
  /arm_<arm>/wrist_camera/color/camera_info           the REAL intrinsics (8 rational coefficients)

Refuses to start unless /controller_manager reports mock_components/GenericSystem for every hardware
component, ROS_LOCALHOST_ONLY=1 and the domain is not 11/12 (the real cell). Never run it next to the
real camera driver in the same domain (same topics).
"""
from __future__ import annotations

import os as _os

# <= 1 core in total (the cell's cameras starve under load): one BLAS/OpenMP thread, one OpenCV thread
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_var, "1")

import argparse
import array
import os
import sys
import time

import cv2
import numpy as np

from .calib import load_hand_eye
from .camera import CameraModel
from .geom import make_T, quat_xyzw_to_matrix
from .model import build_arm_model, load_snapshot
from .synthcam import SceneSpec, render

RESERVED_DOMAINS = (11, 12)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MOCK ONLY synthetic wrist RGB-D camera for side_cut_probe")
    p.add_argument("--arm", choices=["left", "right"], default="left")
    p.add_argument("--snapshot-file", required=True)
    p.add_argument("--calib", required=True, help="wrist_handeye/v1 YAML used as the TRUE mount")
    p.add_argument("--camera-info", required=True, help="camera_info.json (K, D with 8 coefficients)")
    p.add_argument("--world-frame", default="world")
    p.add_argument("--hidden-offset-mm", type=float, nargs=3, default=[8.0, -6.0, 3.0],
                   metavar=("ALONG", "ACROSS", "NORMAL"))
    p.add_argument("--hidden-yaw-deg", type=float, default=2.0)
    p.add_argument("--rate-hz", type=float, default=5.0)
    p.add_argument("--stride", type=int, default=2, help="render grid stride (2 = 640x400 then upsample)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--no-hw-check", action="store_true", help="skip the mock_components check (tests only)")
    return p


def _u8(a) -> array.array:
    out = array.array("B")
    out.frombytes(np.ascontiguousarray(a).tobytes())
    return out


def safety_refusal(env=os.environ) -> str:
    """Reason to refuse from the environment alone ('' = fine)."""
    try:
        dom = int(env.get("ROS_DOMAIN_ID", "0"))
    except ValueError:
        return "ROS_DOMAIN_ID is not an integer"
    if dom in RESERVED_DOMAINS:
        return f"ROS_DOMAIN_ID={dom} is a real-cell domain"
    if env.get("ROS_LOCALHOST_ONLY", "") != "1":
        return "ROS_LOCALHOST_ONLY must be 1"
    return ""


def main(argv=None) -> int:
    cv2.setNumThreads(1)
    args = build_parser().parse_args(argv)
    why = safety_refusal()
    if why:
        print(f"mock_side_cut_camera: refusing to start: {why}", file=sys.stderr)
        return 2
    import rclpy
    from controller_manager_msgs.srv import ListHardwareComponents
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.time import Time
    from sensor_msgs.msg import CameraInfo, CompressedImage, Image
    from tf2_ros import Buffer, TransformListener

    rclpy.init(args=[])
    node = Node("mock_side_cut_camera")
    log = node.get_logger()
    if not args.no_hw_check:
        cli = node.create_client(ListHardwareComponents, "/controller_manager/list_hardware_components")
        ok, seen = False, []
        for _attempt in range(4):
            if not cli.wait_for_service(10.0):
                seen = ["service not available"]
                continue
            fut = cli.call_async(ListHardwareComponents.Request())
            t0 = time.time()
            while not fut.done() and time.time() - t0 < 15.0:
                rclpy.spin_once(node, timeout_sec=0.1)
            if fut.done():
                seen = [c.class_type for c in fut.result().component]
                ok = bool(seen) and all(c == "mock_components/GenericSystem" for c in seen)
                break
        if not ok:
            log.error(f"hardware is not (confirmed to be) mock_components/GenericSystem (saw {seen}): "
                      "refusing to start")
            node.destroy_node()
            rclpy.shutdown()
            return 2
    he = load_hand_eye(args.calib)
    cam = CameraModel.from_json(args.camera_info)
    model = build_arm_model(load_snapshot(args.snapshot_file), args.arm)
    along, across, normal = args.hidden_offset_mm
    spec = SceneSpec(d_along_mm=along, d_across_mm=across, d_normal_mm=normal, yaw_deg=args.hidden_yaw_deg,
                     seed=args.seed)
    prefix = f"/arm_{args.arm}/wrist_camera"
    pub_c = node.create_publisher(Image, f"{prefix}/color/image_raw", qos_profile_sensor_data)
    pub_cc = node.create_publisher(CompressedImage, f"{prefix}/color/image_raw/compressed", qos_profile_sensor_data)
    pub_d = node.create_publisher(Image, f"{prefix}/depth/image_raw", qos_profile_sensor_data)
    pub_i = node.create_publisher(CameraInfo, f"{prefix}/color/camera_info", qos_profile_sensor_data)
    buf = Buffer()
    TransformListener(buf, node, spin_thread=True)
    link6 = f"arm_{args.arm}_link6"
    frame_id = "wrist_camera_color_optical_frame" if args.arm == "left" else "right_wrist_camera_color_optical_frame"
    info = CameraInfo()
    info.height, info.width = cam.height, cam.width
    info.k = [float(v) for v in cam.K.ravel()]
    info.d = [float(v) for v in cam.D]
    info.distortion_model = "plumb_bob"          # as the real driver labels its 8 coefficients
    info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    info.p = [cam.K[0, 0], 0.0, cam.K[0, 2], 0.0, 0.0, cam.K[1, 1], cam.K[1, 2], 0.0, 0.0, 0.0, 1.0, 0.0]
    state = {"n": 0, "last_stamp": None}

    def tick():
        try:
            tf = buf.lookup_transform(args.world_frame, link6, Time())
        except Exception:                      # noqa: BLE001
            return
        # robot_state_publisher publishes TF only when joints move (<= 20 Hz): at rest the newest TF is old.
        # Stamp the image with that TF's stamp while it is fresh (so the probe finds the pose at the stamp),
        # else with the clock (the arm is at rest: the old TF is the current pose).
        now_msg = node.get_clock().now()
        st = tf.header.stamp
        stamp_ns = st.sec * 10 ** 9 + st.nanosec
        if now_msg.nanoseconds - stamp_ns > 60_000_000:
            st = now_msg.to_msg()
        out_ns = st.sec * 10 ** 9 + st.nanosec
        if state["last_stamp"] == out_ns:
            return
        state["last_stamp"] = out_ns
        t, q = tf.transform.translation, tf.transform.rotation
        T_wl6 = make_T(quat_xyzw_to_matrix([q.x, q.y, q.z, q.w]), [t.x, t.y, t.z])
        T_wo = T_wl6 @ he.T_link6_optical
        color, depth, _ = render(cam, T_wo, model, spec, stride=args.stride)
        hdr_stamp = st
        m = Image()
        m.header.stamp, m.header.frame_id = hdr_stamp, frame_id
        m.height, m.width, m.encoding, m.step = cam.height, cam.width, "rgb8", cam.width * 3
        m.data = _u8(color)                                        # array.array('B'): rclpy's fast path
        pub_c.publish(m)
        ok, jpg = cv2.imencode(".jpg", color[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            mc = CompressedImage()
            mc.header, mc.format, mc.data = m.header, "jpeg", _u8(jpg)
            pub_cc.publish(mc)
        md = Image()
        md.header = m.header
        md.height, md.width, md.encoding, md.step = cam.height, cam.width, "16UC1", cam.width * 2
        md.data = _u8(depth)
        pub_d.publish(md)
        info.header = m.header
        pub_i.publish(info)
        state["n"] += 1

    node.create_timer(1.0 / args.rate_hz, tick)
    log.warning(f"MOCK wrist camera {args.arm}: SYNTHETIC images, hidden carton offset (along, across, normal) mm = "
                f"{args.hidden_offset_mm}, yaw {args.hidden_yaw_deg} deg, rendering at {args.rate_hz} Hz (stride {args.stride})")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
