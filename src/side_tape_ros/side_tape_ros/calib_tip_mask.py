"""Calibrate knife tip and blade/holder occlusion in de-rotated image pixels."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from . import core_path  # noqa: F401
from precision_box_detector.side_tape.config import load_config
from precision_box_detector.side_tape.geometry import derotate_image


def _grab(topic):
    import rclpy
    from cv_bridge import CvBridge
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Image

    rclpy.init()
    node = rclpy.create_node("side_tape_calib_tip_mask")
    frames = []
    qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                     reliability=ReliabilityPolicy.BEST_EFFORT)
    subscription = node.create_subscription(Image, topic, frames.append, qos)
    try:
        deadline = time.monotonic() + 10.0
        while not frames and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.25)
        if not frames:
            raise TimeoutError(f"no image on {topic} within 10 s")
        return CvBridge().imgmsg_to_cv2(frames[-1], desired_encoding="bgr8")
    finally:
        node.destroy_subscription(subscription)
        node.destroy_node()
        rclpy.shutdown()


def _collect_points(image):
    title = "Tip (first click), then holder polygon; Enter saves, Esc cancels"
    tip = []
    polygon = []

    def click(event, x, y, _flags, _userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            (tip if not tip else polygon).append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and polygon:
            polygon.pop()

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(title, click)
    try:
        while True:
            preview = image.copy()
            if tip:
                cv2.drawMarker(preview, tip[0], (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
            for vertex in polygon:
                cv2.circle(preview, vertex, 4, (0, 255, 255), -1)
            if len(polygon) > 1:
                cv2.polylines(preview, [np.asarray(polygon, np.int32)], False, (0, 255, 255), 2)
            cv2.imshow(title, preview)
            key = cv2.waitKey(30) & 0xff
            if key in (10, 13) and tip and len(polygon) >= 3:
                return tip[0], polygon
            if key == 27:
                raise KeyboardInterrupt("calibration cancelled")
    finally:
        cv2.destroyWindow(title)


def save_calibration(config_path, tip, polygon, shape):
    """Update only mount tip/mask keys and write a binary full-frame PNG."""
    config_path = Path(config_path)
    cfg = load_config(config_path)
    h, w = shape[:2]
    if (h, w) != (cfg.camera.height, cfg.camera.width):
        raise ValueError(f"image {w}x{h} != configured {cfg.camera.width}x{cfg.camera.height}")
    if not (0 <= tip[0] < w and 0 <= tip[1] < h):
        raise ValueError("tip is outside the image")
    vertices = np.asarray(polygon, dtype=np.float64)
    if (vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 3
            or not np.all(np.isfinite(vertices)) or np.any(vertices < 0)
            or np.any(vertices[:, 0] >= w) or np.any(vertices[:, 1] >= h)):
        raise ValueError("polygon needs at least three in-image vertices")
    if cv2.contourArea(vertices.astype(np.float32)) <= 0:
        raise ValueError("polygon has zero area")
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [np.rint(vertices).astype(np.int32)], 255)
    mask_path = config_path.with_name(f"{cfg.mount.mount_id}_occluder.png")
    with config_path.open(encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    data.setdefault("mount", {})["tip_uv"] = [float(tip[0]), float(tip[1])]
    data["mount"]["occluder_mask_path"] = mask_path.name
    # Validate the updated configuration before changing either output on disk.
    from precision_box_detector.side_tape.config import config_from_dict
    config_from_dict(data)
    if not cv2.imwrite(str(mask_path), mask):
        raise OSError(f"cannot write mask {mask_path}")
    config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return mask_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", help="still image, otherwise grab one Image from camera topic")
    parser.add_argument("--config", default=core_path.DEFAULT_CONFIG)
    parser.add_argument("--tip", nargs=2, type=float, metavar=("U", "V"))
    parser.add_argument("--mask-poly", help='polygon as "u,v;u,v;u,v;..."')
    parser.add_argument("--no-gui", action="store_true")
    args = parser.parse_args(argv)
    if args.no_gui and (args.tip is None or args.mask_poly is None):
        parser.error("--no-gui requires --tip U V and --mask-poly")
    cfg = load_config(args.config)
    image = cv2.imread(args.image, cv2.IMREAD_COLOR) if args.image else _grab(cfg.camera.image_topic)
    if image is None:
        parser.error(f"cannot decode image {args.image}")
    if image.shape[:2] != (cfg.camera.height, cfg.camera.width):
        parser.error(f"image shape {image.shape[:2]} != configured "
                     f"{(cfg.camera.height, cfg.camera.width)}")
    image = derotate_image(image, cfg.camera.rotate_180)
    if args.no_gui:
        try:
            polygon = [[float(v) for v in pair.split(",")] for pair in args.mask_poly.split(";")]
        except ValueError as exc:
            parser.error(f"invalid --mask-poly: {exc}")
        tip = args.tip
    else:
        tip, polygon = _collect_points(image)
    print(f"mask: {save_calibration(args.config, tip, polygon, image.shape)}; tip: {tip}")


if __name__ == "__main__":
    main()
