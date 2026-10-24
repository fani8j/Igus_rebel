"""Hands-free capture session: saves one RGB-D capture each time the scene has
changed (the operator moved the box) and then stayed still.

Same files as ``recorder.py`` (rgb/<id>.png, pointcloud/<id>.pcd,
captures/<id>.json under the output directory) because it reuses
:class:`~box_seam_detection.live.recorder.RgbdRecorder`'s writers; only the
"when to save" rule differs. Read-only on the camera topics; never commands
the robot.

    ros2 run box_seam_detection box_seam_auto_capture --count 20 \
        --output-dir /home/farhan/Igus_rebel/outputs/rgbd_captures_dual_arm_20261012

Rule (``MotionGate``): the colour image is shrunk and blurred; a frame is
"still" when < ``still_frac`` of its pixels changed by > ``still_px`` levels
(largest colour channel) since the previous check; a capture is taken after ``still_s`` seconds
of stillness if the scene differs from the last capture (> ``change_frac`` of
the pixels changed by > ``change_px``) — the first capture only needs
stillness. ``--roi`` limits both tests to the region where the box stands
(so people elsewhere in the image don't matter).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

SMALL_W = 320  # analysis width (px); height keeps the aspect ratio


def small_gray(bgr: np.ndarray, roi: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """Shrunk, blurred COLOUR image (int16, H x W x 3) of the image or a full-resolution ROI
    x0,y0,x1,y1. Colour, not grey: a brown box on a grey floor can have the same grey level."""
    if roi is not None:
        x0, y0, x1, y1 = roi
        bgr = bgr[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]
    h, w = bgr.shape[:2]
    scale = SMALL_W / float(max(w, 1))
    small = cv2.resize(bgr, (SMALL_W, max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(small, (5, 5), 0).astype(np.int16)


def changed_fraction(a: np.ndarray, b: np.ndarray, px: float) -> float:
    """Fraction of pixels whose largest per-channel difference exceeds ``px``."""
    if a.shape != b.shape:
        return 1.0
    d = np.abs(a - b)
    if d.ndim == 3:
        d = d.max(axis=2)
    return float(np.mean(d > px))


class MotionGate:
    """Pure decision logic (no ROS): feed frames with timestamps, get 'capture now'."""

    def __init__(self, still_px=12.0, still_frac=0.002, still_s=1.5,
                 change_px=20.0, change_frac=0.003, min_interval_s=3.0):
        self.still_px, self.still_frac, self.still_s = still_px, still_frac, still_s
        self.change_px, self.change_frac = change_px, change_frac
        self.min_interval_s = min_interval_s
        self.prev = None
        self.ref = None              # frame of the last capture
        self.still_since = None
        self.last_capture_t = None
        self.state = "waiting"       # waiting | settling | ready

    def update(self, frame: np.ndarray, t: float) -> bool:
        """True when a capture should be taken now (caller then calls captured())."""
        moving = self.prev is not None and changed_fraction(frame, self.prev, self.still_px) > self.still_frac
        self.prev = frame
        if moving or self.still_since is None:
            self.still_since = t if not moving else None
            if moving:
                self.state = "moving"
                return False
        if self.still_since is None:
            self.still_since = t
        still_for = t - self.still_since
        changed = self.ref is None or changed_fraction(frame, self.ref, self.change_px) > self.change_frac
        if not changed:
            self.state = "waiting"   # same scene as the last capture: move the box
            return False
        if still_for < self.still_s:
            self.state = "settling"
            return False
        if self.last_capture_t is not None and t - self.last_capture_t < self.min_interval_s:
            self.state = "settling"
            return False
        self.state = "ready"
        return True

    def captured(self, frame: np.ndarray, t: float) -> None:
        self.ref = frame
        self.last_capture_t = t
        self.still_since = None


def _parse_roi(text: str | None):
    if not text:
        return None
    vals = [int(float(v)) for v in text.split(",")]
    if len(vals) != 4 or vals[2] <= vals[0] or vals[3] <= vals[1]:
        raise argparse.ArgumentTypeError("--roi must be x0,y0,x1,y1 with x1>x0, y1>y0")
    return tuple(vals)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--output-dir", default="/home/farhan/Igus_rebel/outputs/rgbd_captures_dual_arm_20261012")
    ap.add_argument("--count", type=int, default=20, help="captures to take, then exit")
    ap.add_argument("--roi", type=_parse_roi, default=None,
                    help="x0,y0,x1,y1 (full-resolution px) where the box stands; default whole image")
    ap.add_argument("--still-s", type=float, default=1.5, help="seconds the scene must be still")
    ap.add_argument("--rgb-topic", default="/camera/color/image_raw")
    ap.add_argument("--pointcloud-topic", default="/camera/depth_registered/points")
    ap.add_argument("--camera-info-topic", default="/camera/color/camera_info")
    ap.add_argument("--rate-hz", type=float, default=4.0, help="motion checks per second")
    ns, ros_args = ap.parse_known_args(argv)

    try:
        import rclpy
        from .recorder import RgbdRecorder
    except Exception as exc:  # noqa: BLE001
        print(f"box_seam_auto_capture: ROS 2 python packages unavailable: {exc}", file=sys.stderr)
        return 2

    out = Path(ns.output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    rclpy.init(args=["box_seam_auto_capture", "--ros-args",
                     "-p", f"rgb_topic:={ns.rgb_topic}",
                     "-p", f"pointcloud_topic:={ns.pointcloud_topic}",
                     "-p", f"camera_info_topic:={ns.camera_info_topic}",
                     *[a for a in ros_args if a != "--ros-args"]])
    node = RgbdRecorder(cli_output_dir=str(out), cli_max_frames=ns.count, cli_timeout_sec=0.0)
    node.timer.cancel()              # no fixed-rate saving: MotionGate decides
    node.save_immediately = False
    gate = MotionGate(still_s=ns.still_s)
    ids_file = out / "captures.txt"
    log = node.get_logger()
    log.info(f"auto capture: {ns.count} captures into {out} — move the box, step back, "
             f"keep the scene still ~{ns.still_s:g} s; ROI={ns.roi or 'whole image'}")
    last_hint = 0.0
    last_state = None

    def tick():
        nonlocal last_hint, last_state
        with node._lock:
            pair = node._latest_pair
        if pair is None:
            if time.monotonic() - last_hint > 5.0:
                log.warn("no synced RGB + point cloud yet (camera running? topics?)")
                last_hint = time.monotonic()
            return
        try:
            bgr = node.bridge.imgmsg_to_cv2(pair[0], desired_encoding="bgr8")
        except Exception as exc:  # noqa: BLE001
            log.error(f"image decode failed: {exc}")
            return
        frame = small_gray(bgr, ns.roi)
        now = time.monotonic()
        if gate.update(frame, now):
            n_before = node._frame_count
            if node._save_pending_pair() and node._frame_count > n_before:
                gate.captured(frame, now)
                cid = f"{node.prefix}_{node._frame_count:06d}"
                # the recorder logged the exact paths; record the newest manifest id
                manifests = sorted((out / "captures").glob("*.json"), key=lambda p: p.stat().st_mtime)
                if manifests:
                    cid = manifests[-1].stem
                    with open(ids_file, "a", encoding="utf-8") as fh:
                        fh.write(cid + "\n")
                print(f"\a\n>>> CAPTURED {node._frame_count}/{ns.count}: {cid}  — now move the box\n",
                      flush=True)
        if gate.state != last_state or now - last_hint > 6.0:
            msg = {"moving": "moving…", "settling": "hold still…",
                   "waiting": "same scene as the last capture — move/rotate the box",
                   "ready": "capturing"}.get(gate.state, gate.state)
            log.info(f"[{node._frame_count}/{ns.count}] {msg}")
            last_state, last_hint = gate.state, now

    node.create_timer(1.0 / max(ns.rate_hz, 0.5), tick)
    code = 0
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.2)
    except KeyboardInterrupt:
        code = 130
    finally:
        node.destroy_node()
        rclpy.shutdown()
    print(f"auto capture finished: {node._frame_count} captures in {out} (ids in {ids_file})", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
