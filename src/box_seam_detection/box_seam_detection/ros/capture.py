"""Scene-camera capture for the DetectBox node: colour + aligned depth + CameraInfo -> rgb png / pcd / manifest.

Subscribes (sensor-data QoS, its own callback group) to the colour image, the depth image registered to the colour
camera and the colour CameraInfo, synchronised with an ApproximateTimeSynchronizer. It never subscribes to the
driver's coloured point cloud (34 MB, ~1 core of driver CPU, ~0.6 s topic lag while subscribed).

Freshness: ``grab(after_ns, timeout_s)`` returns the first synced (colour, depth) set whose BOTH image stamps are
> after_ns + margin (stamps are host-clock, same clock as ``now_ns()``); older sets are never returned, a missing
fresh set raises ``CaptureTimeout``. The consumed set (and every older one) is dropped, so a frame is used once.
Sets whose depth size differs from the colour size are skipped (the Orbbec driver publishes the raw, unregistered
depth when software alignment finds no colour frame in the frameset, e.g. when auto exposure drops the colour rate);
``write_capture`` still validates the registration of the set that is returned.
"""

from __future__ import annotations

import collections
import dataclasses
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import cloud


class CaptureTimeout(cloud.CaptureError):
    pass


class CaptureAborted(cloud.CaptureError):
    pass


@dataclasses.dataclass
class FrameSet:
    color: Any
    depth: Any
    recv_ns: int

    @property
    def color_stamp_ns(self) -> int:
        return cloud.stamp_ns(self.color.header)

    @property
    def depth_stamp_ns(self) -> int:
        return cloud.stamp_ns(self.depth.header)

    @property
    def depth_aligned(self) -> bool:
        return (self.depth.width, self.depth.height) == (self.color.width, self.color.height)


@dataclasses.dataclass
class LocalCaptureFiles:
    """Fallback with the live.orchestrator.CaptureFiles fields (used when no files_cls is given)."""
    capture_id: str
    image_path: Path
    pcd_path: Path
    manifest_path: Path
    stamp_ns: int


class SceneCapture:
    def __init__(self, node, *, color_topic: str = "/camera/color/image_raw",
                 depth_topic: str = "/camera/depth/image_raw",
                 info_topic: str = "/camera/color/camera_info",
                 output_root: str | Path = "~/Igus_rebel/outputs/rgbd_captures",
                 sync_slop_s: float = 0.04, queue_size: int = 10, margin_s: float = 0.0,
                 prefix: str = "frame", driver_parent_frame: str = "camera_link",
                 use_tf: bool = True, extrinsic_required: bool = False, tf_timeout_s: float = 2.0,
                 files_cls: Callable[..., Any] | None = None, callback_group=None, tf_buffer=None,
                 staged: bool = False):
        import message_filters
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo, Image

        self.node = node
        self.topics = {"rgb": color_topic, "depth": depth_topic, "camera_info": info_topic}
        self.margin_ns = int(max(margin_s, 0.0) * 1e9)
        self.prefix = prefix
        self.driver_parent_frame = driver_parent_frame
        self.extrinsic_required = extrinsic_required
        self.tf_timeout_s = tf_timeout_s
        self.files_cls = files_cls or LocalCaptureFiles
        self.staged = staged    # output_root is a RAM staging dir: a failed write must not leave partial files
        root = Path(str(output_root)).expanduser()
        self.dirs = {n: root / n for n in ("rgb", "pointcloud", "captures")}
        for d in self.dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        self._seq = 0
        self._cond = threading.Condition()
        self._sets: collections.deque[FrameSet] = collections.deque(maxlen=4)
        self._info = None
        self._counts = {"rgb": 0, "depth": 0, "camera_info": 0, "synced": 0, "skipped_unaligned": 0}
        self._last_stamp_ns = {"rgb": 0, "depth": 0}

        self._color_sub = message_filters.Subscriber(
            node, Image, color_topic, qos_profile=qos_profile_sensor_data, callback_group=callback_group)
        self._depth_sub = message_filters.Subscriber(
            node, Image, depth_topic, qos_profile=qos_profile_sensor_data, callback_group=callback_group)
        self._color_sub.registerCallback(lambda m: self._count("rgb", m))
        self._depth_sub.registerCallback(lambda m: self._count("depth", m))
        self._info_sub = node.create_subscription(
            CameraInfo, info_topic, self._on_info, qos_profile_sensor_data, callback_group=callback_group)
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [self._color_sub, self._depth_sub], queue_size=queue_size, slop=sync_slop_s)
        self._sync.registerCallback(self._on_synced)

        self._tf = tf_buffer
        self._tf_listener = None
        if self._tf is None and use_tf:
            import tf2_ros
            self._tf = tf2_ros.Buffer()
            self._tf_listener = tf2_ros.TransformListener(self._tf, node, spin_thread=False)

    # ------------------------------------------------------------------ callbacks (cheap: store references)
    def now_ns(self) -> int:
        return int(self.node.get_clock().now().nanoseconds)

    def _count(self, key, msg):
        with self._cond:
            self._counts[key] += 1
            self._last_stamp_ns[key] = cloud.stamp_ns(msg.header)

    def _on_info(self, msg):
        with self._cond:
            self._info = msg
            self._counts["camera_info"] += 1

    def _on_synced(self, color, depth):
        with self._cond:
            self._counts["synced"] += 1
            self._sets.append(FrameSet(color, depth, self.now_ns()))
            self._cond.notify_all()

    # ------------------------------------------------------------------ freshness
    def diagnostics(self) -> dict:
        with self._cond:
            return {"counts": dict(self._counts), "last_stamp_ns": dict(self._last_stamp_ns),
                    "have_camera_info": self._info is not None}

    def grab(self, after_ns: int, timeout_s: float, abort: Callable[[], bool] | None = None) -> tuple[FrameSet, Any]:
        """First synced set with colour AND depth stamp > after_ns + margin, plus the latest CameraInfo."""
        threshold = int(after_ns) + self.margin_ns
        deadline = time.monotonic() + max(float(timeout_s), 0.0)
        with self._cond:
            while True:
                if abort is not None and abort():
                    raise CaptureAborted("capture wait aborted (cancel / shutdown)")
                aligned = [fs for fs in self._sets if fs.depth_aligned]
                if len(aligned) != len(self._sets):
                    self._counts["skipped_unaligned"] += len(self._sets) - len(aligned)
                    self._sets = collections.deque(aligned, maxlen=self._sets.maxlen)
                for i, fs in enumerate(self._sets):
                    if fs.color_stamp_ns > threshold and fs.depth_stamp_ns > threshold and self._info is not None:
                        for _ in range(i + 1):
                            self._sets.popleft()
                        return fs, self._info
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._cond.wait(min(remaining, 0.1))
            d = {"counts": dict(self._counts), "last_stamp_ns": dict(self._last_stamp_ns),
                 "have_camera_info": self._info is not None}
        unaligned = (f"; {d['counts']['skipped_unaligned']} synced set(s) skipped because the depth is not aligned "
                     "to the colour image (driver alignment failing - colour frame rate / exposure?)"
                     if d["counts"]["skipped_unaligned"] else "")
        raise CaptureTimeout(
            f"no fresh synced colour+depth frame (stamps > {threshold}) within {timeout_s:g} s{unaligned}; "
            f"received {d['counts']}, camera_info={'yes' if d['have_camera_info'] else 'NO'}, "
            f"newest stamps {d['last_stamp_ns']}")

    # ------------------------------------------------------------------ extrinsic
    def driver_extrinsic(self, child_frame: str, stamp_ns: int) -> dict | None:
        """camera_link <- colour optical frame from tf2 at the image stamp (static in practice)."""
        err: Exception | None = None
        if self._tf is None:
            err = RuntimeError("tf disabled")
        else:
            from rclpy.time import Time
            deadline = time.monotonic() + self.tf_timeout_s
            while True:
                try:
                    t = self._tf.lookup_transform(self.driver_parent_frame, child_frame,
                                                  Time(nanoseconds=int(stamp_ns)))
                    tr, q = t.transform.translation, t.transform.rotation
                    return {"parent_frame": self.driver_parent_frame, "child_frame": child_frame,
                            "translation": [tr.x, tr.y, tr.z], "quaternion_xyzw": [q.x, q.y, q.z, q.w],
                            "source": "tf_live", "stamp_ns": int(stamp_ns)}
                except Exception as exc:  # noqa: BLE001 - tf2 raises several exception types
                    err = exc
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.05)
        if self.extrinsic_required:
            raise cloud.CaptureError(f"driver extrinsic {self.driver_parent_frame} <- {child_frame}: {err}")
        self.node.get_logger().warn(
            f"driver extrinsic {self.driver_parent_frame} <- {child_frame} unavailable ({err}); "
            "manifest has none (driver_extrinsic 'recorded' still works)")
        return None

    # ------------------------------------------------------------------ capture
    def write_capture(self, fs: FrameSet, info) -> Any:
        scale, k, width, height = cloud.validate_registration(fs.color, fs.depth, info)
        scaled_from = None if (info.width, info.height) == (width, height) else (info.width, info.height)
        rgb = cloud.rgb_from_color_msg(fs.color)
        depth = cloud.image_array(fs.depth)
        points = cloud.backproject_depth(depth, rgb, k, scale)
        if points.shape[0] == 0:
            raise cloud.CaptureError("aligned depth image has no valid pixel")
        self._seq += 1
        cid = f"{self.prefix}_{self._seq:06d}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        try:
            image_path = cloud.write_png(str(self.dirs["rgb"] / f"{cid}.png"), rgb)
            pcd_path = cloud.write_pcd(str(self.dirs["pointcloud"] / f"{cid}.pcd"), points)
            ext = self.driver_extrinsic(fs.color.header.frame_id, fs.color_stamp_ns)
            manifest = cloud.build_manifest(cid, image_path, pcd_path, fs.color, fs.depth, info, k,
                                            int(points.shape[0]), self.topics, ext, k_scaled_from=scaled_from)
            mpath = cloud.write_manifest(str(self.dirs["captures"] / f"{cid}.json"), manifest)
        except BaseException:
            if self.staged:     # RAM staging: drop what this capture wrote (nothing else would ever archive it)
                for sub, ext_ in (("rgb", ".png"), ("pointcloud", ".pcd"), ("captures", ".json")):
                    (self.dirs[sub] / f"{cid}{ext_}").unlink(missing_ok=True)
            raise
        return self.files_cls(capture_id=cid, image_path=Path(image_path), pcd_path=Path(pcd_path),
                              manifest_path=Path(mpath), stamp_ns=fs.color_stamp_ns)

    def capture(self, after_ns: int | None = None, timeout_s: float = 10.0,
                abort: Callable[[], bool] | None = None) -> Any:
        """grab a fresh set (after_ns defaults to now) and write rgb/pcd/manifest; returns files_cls(...)."""
        if after_ns is None:
            after_ns = self.now_ns()
        fs, info = self.grab(after_ns, timeout_s, abort)
        return self.write_capture(fs, info)
