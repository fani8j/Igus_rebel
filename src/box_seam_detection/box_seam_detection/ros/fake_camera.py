"""Test/dev tool: replay a saved capture as colour + aligned-depth + camera_info topics (no camera driver).

    ROS_DOMAIN_ID=87 ROS_LOCALHOST_ONLY=1 python3 -m box_seam_detection.ros.fake_camera \
        /home/farhan/Igus_rebel/outputs/rgbd_captures/captures/<id>.json [--rate 15] [--prefix /camera]

The saved PCD (colour-registered cloud, row-major valid pixels) is re-projected with the manifest K into a 16UC1
aligned depth image, so the node's depth -> cloud path can be compared with the original capture. Every published
frame is stamped with the current time (like the driver's host-clock stamps). Also latches camera_link -> colour
optical frame on /tf_static (the recorded driver extrinsic) unless --no-tf.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

RECORDED_DRIVER_EXTRINSIC = {  # as camera_point_to_base_link.py (seam_handoff)
    "parent_frame": "camera_link", "child_frame": "camera_color_optical_frame",
    "translation": (0.0016341905593874606, -0.013982509613037264, -4.271093383417801e-05),
    "quaternion_xyzw": (-0.499537503542144, 0.4997559393201414, -0.5021479272076848, 0.49855164513834604),
}


def read_pcd_xyzrgb(path) -> tuple[np.ndarray, np.ndarray]:
    """(xyz float32 (N,3), rgb uint32 (N,)) of a binary 'x y z rgb' PCD (recorder format; rgb at byte 12 or 16)."""
    with open(path, "rb") as fh:
        hdr = {}
        while True:
            line = fh.readline().decode("ascii", "replace").strip()
            if not line or line.startswith("#"):
                continue
            key, _, val = line.partition(" ")
            hdr[key] = val
            if key == "DATA":
                break
        data = fh.read()
    if hdr["DATA"] != "binary":
        raise ValueError("binary PCD only")
    n = int(hdr["POINTS"])
    sizes = [int(s) * int(c) for s, c in zip(hdr["SIZE"].split(), hdr["COUNT"].split())]
    stride = sum(sizes)
    buf = np.frombuffer(data[: n * stride], dtype=np.uint8).reshape(n, stride)
    xyz = np.ascontiguousarray(buf[:, :12]).view("<f4").reshape(n, 3)
    names = hdr["FIELDS"].split()
    rgb = None
    if "rgb" in names:
        off = sum(sizes[: names.index("rgb")])
        rgb = np.ascontiguousarray(buf[:, off:off + 4]).view("<u4").reshape(n)
    return xyz, rgb


def pcd_to_depth(xyz: np.ndarray, k, width: int, height: int, scale: float = 0.001) -> np.ndarray:
    """Aligned depth image (uint16, units of `scale` m) from a registered cloud (nearest point wins)."""
    fx, cx, fy, cy = k[0], k[2], k[4], k[5]
    z = xyz[:, 2].astype(np.float64)
    u = np.floor(xyz[:, 0] * fx / z + cx + 0.5).astype(np.int64)
    v = np.floor(xyz[:, 1] * fy / z + cy + 0.5).astype(np.int64)
    ok = (z > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    depth = np.zeros((height, width), dtype=np.uint16)
    order = np.argsort(-z[ok], kind="stable")  # far first, near overwrites
    depth[v[ok][order], u[ok][order]] = np.round(z[ok][order] / scale).astype(np.uint16)
    return depth


def load_capture(manifest_path):
    """(rgb HxWx3 uint8, depth uint16 HxW, manifest dict) of a saved capture."""
    import cv2
    m = json.loads(Path(manifest_path).read_text())
    bgr = cv2.imread(m["image"]["path"], cv2.IMREAD_COLOR)
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    xyz, _ = read_pcd_xyzrgb(m["pointcloud"]["path"])
    ci = m["camera_info"]
    depth = pcd_to_depth(xyz, ci["k"], ci["width"], ci["height"])
    return rgb, depth, m


def main(argv=None):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, Image

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("manifest")
    ap.add_argument("--rate", type=float, default=15.0)
    ap.add_argument("--prefix", default="/camera")
    ap.add_argument("--no-tf", action="store_true")
    ap.add_argument("--depth-offset-ms", type=float, default=0.0, help="depth stamp minus colour stamp")
    ns, ros_args = ap.parse_known_args(argv)

    rclpy.init(args=ros_args)
    rgb, depth, m = load_capture(ns.manifest)
    node = Node("fake_camera")
    frame = m["image"]["header"]["frame_id"]
    ci = m["camera_info"]
    cpub = node.create_publisher(Image, f"{ns.prefix}/color/image_raw", qos_profile_sensor_data)
    dpub = node.create_publisher(Image, f"{ns.prefix}/depth/image_raw", qos_profile_sensor_data)
    ipub = node.create_publisher(CameraInfo, f"{ns.prefix}/color/camera_info", qos_profile_sensor_data)
    if not ns.no_tf:
        from geometry_msgs.msg import TransformStamped
        from tf2_ros import StaticTransformBroadcaster
        e = RECORDED_DRIVER_EXTRINSIC
        t = TransformStamped()
        t.header.frame_id, t.child_frame_id = e["parent_frame"], frame
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = e["translation"]
        (t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w) = \
            e["quaternion_xyzw"]
        node._tfb = StaticTransformBroadcaster(node)
        node._tfb.sendTransform(t)

    h, w = rgb.shape[:2]

    def tick():
        now = node.get_clock().now()
        stamp = now.to_msg()
        c = Image(height=h, width=w, encoding="rgb8", step=w * 3, data=rgb.tobytes())
        c.header.stamp, c.header.frame_id = stamp, frame
        d = Image(height=h, width=w, encoding="16UC1", step=w * 2, data=depth.tobytes())
        d.header.stamp = (now + rclpy.duration.Duration(nanoseconds=int(ns.depth_offset_ms * 1e6))).to_msg()
        d.header.frame_id = frame
        i = CameraInfo(width=ci["width"], height=ci["height"], distortion_model=ci["distortion_model"],
                       d=ci["d"], k=ci["k"], r=[float(v) for v in ci["r"]], p=[float(v) for v in ci["p"]])
        i.header.stamp, i.header.frame_id = stamp, frame
        cpub.publish(c)
        dpub.publish(d)
        ipub.publish(i)

    node.create_timer(1.0 / ns.rate, tick)
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
