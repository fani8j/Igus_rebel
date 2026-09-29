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
    """Click UI (matplotlib; the pip OpenCV here is a headless build without windows).

    Left click: first the knife tip, then the blade/holder polygon vertices.
    Right click: undo the last polygon vertex (or the tip if no vertices).
    Enter: save (needs the tip and >= 3 vertices). Esc: cancel.
    Use the toolbar zoom/pan to place the tip precisely; clicks are ignored while a
    toolbar tool is active (click the tool again to turn it off).
    """
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    tip = []
    polygon = []
    result = {"done": False, "cancel": False}
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(image[..., ::-1] if image.ndim == 3 else image, cmap=None if image.ndim == 3 else "gray")
    ax.set_title("LEFT: tip first, then blade/holder polygon | RIGHT: undo | ENTER: save | ESC: cancel")
    tip_art, = ax.plot([], [], "r+", markersize=20, markeredgewidth=2)
    poly_art, = ax.plot([], [], "y.-", linewidth=2, markersize=8)

    def redraw():
        tip_art.set_data([p[0] for p in tip], [p[1] for p in tip])
        pts = polygon + polygon[:1] if len(polygon) >= 3 else polygon
        poly_art.set_data([p[0] for p in pts], [p[1] for p in pts])
        fig.canvas.draw_idle()

    def on_click(event):
        toolbar = getattr(fig.canvas, "toolbar", None)
        if event.inaxes is not ax or (toolbar is not None and getattr(toolbar, "mode", "")):
            return
        if event.button == 1:
            (tip if not tip else polygon).append((float(event.xdata), float(event.ydata)))
        elif event.button == 3:
            if polygon:
                polygon.pop()
            elif tip:
                tip.pop()
        redraw()

    def on_key(event):
        if event.key == "enter" and tip:
            result["done"] = True
            plt.close(fig)
        elif event.key == "escape":
            result["cancel"] = True
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    print("Window open: LEFT click the knife tip, then the corners around blade+holder; "
          "press ENTER in the window to save (ESC cancels).", flush=True)
    plt.show()
    if not result["done"] and not result["cancel"] and tip:
        answer = input(f"Window closed without Enter. Save tip {tip[0][0]:.1f},{tip[0][1]:.1f}"
                       f" and {len(polygon)} outline point(s)? [y/N] ").strip().lower()
        result["done"] = answer in ("y", "yes")
    if not result["done"]:
        raise KeyboardInterrupt("calibration cancelled, nothing saved")
    if 0 < len(polygon) < 3:
        print(f"WARNING: only {len(polygon)} outline point(s); saving the tip without a blade mask.")
    outline = [(int(round(u)), int(round(v))) for u, v in polygon] if len(polygon) >= 3 else None
    return tip[0], outline


def save_calibration(config_path, tip, polygon, shape):
    """Update only mount tip/mask keys and write a binary full-frame PNG."""
    config_path = Path(config_path)
    cfg = load_config(config_path)
    h, w = shape[:2]
    if (h, w) != (cfg.camera.height, cfg.camera.width):
        raise ValueError(f"image {w}x{h} != configured {cfg.camera.width}x{cfg.camera.height}")
    if not (0 <= tip[0] < w and 0 <= tip[1] < h):
        raise ValueError("tip is outside the image")
    if polygon is None:
        with config_path.open(encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}
        data.setdefault("mount", {})["tip_uv"] = [float(tip[0]), float(tip[1])]
        from precision_box_detector.side_tape.config import config_from_dict
        config_from_dict(data)
        config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        return None
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
    if args.no_gui and args.tip is None:
        parser.error("--no-gui requires --tip U V (--mask-poly optional)")
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
            polygon = ([[float(v) for v in pair.split(",")] for pair in args.mask_poly.split(";")]
                       if args.mask_poly else None)
        except ValueError as exc:
            parser.error(f"invalid --mask-poly: {exc}")
        tip = args.tip
    else:
        tip, polygon = _collect_points(image)
    mask = save_calibration(args.config, tip, polygon, image.shape)
    print(f"SAVED to {args.config}: tip_uv = [{tip[0]:.1f}, {tip[1]:.1f}]"
          + (f", blade mask = {mask}" if mask else ", no blade mask"))


if __name__ == "__main__":
    main()
