"""Scene-camera capture for the DetectBox node: colour + depth + CameraInfo -> rgb png / pcd / manifest.

Subscribes (sensor-data QoS, its own callback group) to the colour image, the depth image, the colour CameraInfo and,
for the node-side registration path, the raw (unaligned) depth image and the depth CameraInfo. It never subscribes to
the driver's coloured point cloud (34 MB, ~1 core of driver CPU, ~0.6 s topic lag while subscribed).

Two kinds of frame sets feed ``grab``:

* driver-aligned (``source: driver``): [colour, depth] synchronised with an ApproximateTimeSynchronizer
  (``sync_slop_s``); the depth must already be registered to the colour image (same size and frame). Sets whose depth
  size differs from the colour size are skipped (the Orbbec driver publishes the raw, unregistered depth when its
  software alignment finds no colour frame in the frameset, e.g. when the colour rate drops); ``write_capture``
  still validates the registration of the set that is returned.
* node-registered (``source: node``, ``node_depth_registration``): [colour, raw depth] synchronised with a second
  ApproximateTimeSynchronizer (``raw_sync_slop_s``, below one frame period: the stamps share the colour grid, so
  only same-frame pairs match; the base_link step rejects pairs > 50 ms apart). The raw depth (``raw_depth_topic``,
  e.g. /camera/depth/image_unaligned: own optical frame, undistorted) is re-projected into the raw colour pixel grid
  in ``write_capture`` (outside the lock) with ``cloud.register_depth_to_color``, using the depth intrinsics from
  ``depth_info_topic`` (the latest CameraInfo whose size equals the raw depth size; the driver also publishes the
  wrong-frame aligned info there) and the colour<-depth extrinsic from tf2 (colour header frame_id <- raw depth
  header frame_id, cached after the first success). The path fails closed - the set is not a candidate and the
  reason is reported in the ``CaptureTimeout`` message - when the intrinsics or the extrinsic are missing, the frame
  ids are equal, or the raw depth has the colour size (then aligned and raw CameraInfo are indistinguishable, the
  node path refuses such sets and logs once). Everything after the registration (backprojection, PNG, PCD,
  manifest) is shared with the driver path; the manifest records the path in ``depth_registration``.
  ``raw_depth_intrinsics`` ([w, h, fx, fy, cx, cy], zero distortion) is a configured fallback for the depth
  intrinsics: the driver publishes /camera/depth/camera_info only at the aligned size while its alignment works, so
  there is then no raw-size info. A live CameraInfo of the raw depth size always wins; the fallback is used only when
  its size equals the raw depth image size (``intrinsics_source`` in the manifest: camera_info / config_fallback).

Freshness: ``grab(after_ns, timeout_s)`` returns the first fresh set of either kind (smallest colour stamp, driver
first on a tie) whose BOTH image stamps are > after_ns + margin (stamps are host-clock, same clock as ``now_ns()``);
older sets are never returned, a missing fresh set raises ``CaptureTimeout``. The consumed set (and every older
one of either kind) is dropped, so a frame is used once.
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


def parse_raw_depth_intrinsics(values) -> tuple[int, int, float, float, float, float] | None:
    """[width, height, fx, fy, cx, cy] -> (w, h, fx, fy, cx, cy); empty/None -> None. Raises ValueError if invalid."""
    try:
        v = [float(x) for x in values] if values is not None else []
    except (TypeError, ValueError) as exc:
        raise ValueError(f"raw_depth_intrinsics must be numbers [width, height, fx, fy, cx, cy]: {exc}") from exc
    if not v:
        return None
    if len(v) != 6 or not all(np.isfinite(v)):
        raise ValueError(f"raw_depth_intrinsics must be exactly 6 finite numbers [width, height, fx, fy, cx, cy], "
                         f"got {list(values)}")
    w, h, fx, fy, cx, cy = v
    if w < 1 or h < 1 or w != int(w) or h != int(h):
        raise ValueError(f"raw_depth_intrinsics width/height must be positive integers, got {w:g}x{h:g}")
    if fx <= 0 or fy <= 0:
        raise ValueError(f"raw_depth_intrinsics fx/fy must be > 0, got {fx:g}/{fy:g}")
    if not (0 < cx < w and 0 < cy < h):
        raise ValueError(f"raw_depth_intrinsics principal point ({cx:g}, {cy:g}) must lie inside the {w:g}x{h:g} image")
    return int(w), int(h), fx, fy, cx, cy


@dataclasses.dataclass
class FrameSet:
    color: Any
    depth: Any
    recv_ns: int
    source: str = "driver"          # "driver": depth already registered to colour; "node": raw depth, to be registered
    depth_info: Any = None          # node sets: the depth CameraInfo (size == raw depth size) at grab time
    extrinsic: dict | None = None   # node sets: T_color_depth {"matrix", "translation", "quaternion_xyzw", ...}
    depth_info_source: str = ""     # node sets: "camera_info" (live, raw size) or "config_fallback"

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
                 staged: bool = False, node_depth_registration: bool = True,
                 raw_depth_topic: str = "/camera/depth/image_unaligned",
                 depth_info_topic: str = "/camera/depth/camera_info", raw_sync_slop_s: float = 0.03,
                 raw_depth_intrinsics=None):
        import message_filters
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo, Image

        self.node = node
        self.topics = {"rgb": color_topic, "depth": depth_topic, "camera_info": info_topic}
        self.node_registration = bool(node_depth_registration)
        self.raw_depth_topic, self.depth_info_topic = raw_depth_topic, depth_info_topic
        self.sync_slop_s, self.raw_sync_slop_s = float(sync_slop_s), float(raw_sync_slop_s)
        self.raw_intrinsics = parse_raw_depth_intrinsics(raw_depth_intrinsics)     # configured fallback or None
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
        self._counts = {"rgb": 0, "depth": 0, "camera_info": 0, "synced": 0, "skipped_unaligned": 0,
                        "raw_depth": 0, "depth_info": 0, "raw_synced": 0, "captures_driver": 0, "captures_node": 0}
        self._last_stamp_ns = {"rgb": 0, "depth": 0, "raw_depth": 0}
        # node registration state (all under self._cond)
        self._raw_sets: collections.deque[FrameSet] = collections.deque(maxlen=4)
        self._depth_infos: dict[tuple[int, int], Any] = {}      # (width, height) -> latest depth CameraInfo
        self._extrinsics: dict[tuple[str, str], dict] = {}      # (colour frame, depth frame) -> T_color_depth
        self._raw_blocker = ""                                  # why the newest fresh raw set was not usable
        self._last_intrinsics_source = ""                       # of the last node capture written
        self._logged: set[str] = set()

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
        if self.node_registration:
            self._raw_sub = message_filters.Subscriber(
                node, Image, raw_depth_topic, qos_profile=qos_profile_sensor_data, callback_group=callback_group)
            self._raw_sub.registerCallback(lambda m: self._count("raw_depth", m))
            self._depth_info_sub = node.create_subscription(
                CameraInfo, depth_info_topic, self._on_depth_info, qos_profile_sensor_data,
                callback_group=callback_group)
            self._raw_sync = message_filters.ApproximateTimeSynchronizer(
                [self._color_sub, self._raw_sub], queue_size=queue_size, slop=float(raw_sync_slop_s))
            self._raw_sync.registerCallback(self._on_raw_synced)

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

    def _on_depth_info(self, msg):
        with self._cond:
            self._depth_infos[(int(msg.width), int(msg.height))] = msg
            self._counts["depth_info"] += 1
        diff = self._fallback_difference(msg)
        if diff:
            self._log_once("fallback_mismatch", (
                f"live depth camera_info {msg.width}x{msg.height} on {self.depth_info_topic} differs from the "
                f"configured raw_depth_intrinsics ({diff}); the live camera_info is used - update the parameter if "
                "the camera or depth resolution changed"))

    def _fallback_difference(self, msg) -> str:
        """'' if the configured fallback is absent / of another size / equal to the live info within tolerance."""
        fb = self.raw_intrinsics
        if fb is None or (int(msg.width), int(msg.height)) != fb[:2] or len(msg.k) < 6:
            return ""
        fx, fy, cx, cy = fb[2:]
        lfx, lfy, lcx, lcy = (float(msg.k[i]) for i in (0, 4, 2, 5))
        bad = (abs(lfx - fx) > 0.005 * fx or abs(lfy - fy) > 0.005 * fy
               or abs(lcx - cx) > 2.0 or abs(lcy - cy) > 2.0)
        return (f"live fx/fy/cx/cy {lfx:g}/{lfy:g}/{lcx:g}/{lcy:g} vs configured {fx:g}/{fy:g}/{cx:g}/{cy:g}"
                if bad else "")

    def _fallback_info(self, frame_id: str):
        """The configured raw_depth_intrinsics as a CameraInfo (plumb_bob, zero distortion)."""
        from sensor_msgs.msg import CameraInfo
        w, h, fx, fy, cx, cy = self.raw_intrinsics
        info = CameraInfo(width=w, height=h, distortion_model="plumb_bob", d=[0.0] * 5,
                          k=[fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0],
                          p=[fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0])
        info.header.frame_id = frame_id
        return info

    def _log_once(self, key: str, text: str):
        with self._cond:
            if key in self._logged:
                return
            self._logged.add(key)
        self.node.get_logger().warn(text)

    def _on_raw_synced(self, color, depth):
        """A [colour, raw depth] pair: permanent refusals are counted here, everything else waits for grab()."""
        reason = None
        if (depth.width, depth.height) == (color.width, color.height):
            reason = "same_size"
            self._log_once("same_size", (
                f"node depth registration disabled for this stream: raw depth {depth.width}x{depth.height} on "
                f"{self.raw_depth_topic} has the colour size, so aligned and raw depth CameraInfo cannot be told "
                "apart (the raw depth topic must be the unaligned stream)"))
        elif depth.header.frame_id == color.header.frame_id:
            reason = "same_frame"
            self._log_once("same_frame", (
                f"node depth registration refused: raw depth frame_id {depth.header.frame_id!r} equals the colour "
                "frame_id (the raw depth must be in its own optical frame)"))
        elif depth.encoding not in cloud.DEPTH_ENCODING_SCALES:
            reason = "depth_encoding"
        elif color.encoding not in cloud.COLOR_ENCODINGS:
            reason = "color_encoding"
        with self._cond:
            self._counts["raw_synced"] += 1
            if reason is not None:
                key = f"raw_rejected_{reason}"
                self._counts[key] = self._counts.get(key, 0) + 1
                self._raw_blocker = {
                    "same_size": f"raw depth {depth.width}x{depth.height} has the colour size (ambiguous CameraInfo)",
                    "same_frame": f"raw depth frame_id equals the colour frame_id {color.header.frame_id!r}",
                    "depth_encoding": f"raw depth encoding {depth.encoding!r} unsupported",
                    "color_encoding": f"colour encoding {color.encoding!r} unsupported"}[reason]
                return
            self._raw_sets.append(FrameSet(color, depth, self.now_ns(), source="node"))
            self._cond.notify_all()

    # ------------------------------------------------------------------ freshness
    def _diag_locked(self) -> dict:
        return {"counts": dict(self._counts), "last_stamp_ns": dict(self._last_stamp_ns),
                "have_camera_info": self._info is not None,
                "node_registration": {
                    "enabled": self.node_registration,
                    "have_extrinsic": bool(self._extrinsics),
                    "depth_info_sizes": sorted(f"{w}x{h}" for w, h in self._depth_infos),
                    "fallback_intrinsics": self.raw_intrinsics is not None,
                    "fallback_size": f"{self.raw_intrinsics[0]}x{self.raw_intrinsics[1]}" if self.raw_intrinsics else "",
                    "last_intrinsics_source": self._last_intrinsics_source,
                    "blocker": self._raw_blocker}}

    def diagnostics(self) -> dict:
        with self._cond:
            return self._diag_locked()

    def _node_extrinsic(self, color_frame: str, depth_frame: str) -> dict | None:
        """Cached colour <- depth transform (tf2 lookup_transform(colour, depth), latest). None + blocker if absent."""
        key = (color_frame, depth_frame)
        ext = self._extrinsics.get(key)
        if ext is not None:
            return ext
        if self._tf is None:
            self._raw_blocker = "tf disabled (use_tf false and no tf_buffer): no extrinsic colour <- raw depth frame"
            return None
        try:
            from rclpy.time import Time
            t = self._tf.lookup_transform(color_frame, depth_frame, Time())
            tr, q = t.transform.translation, t.transform.rotation
            translation, quat = [tr.x, tr.y, tr.z], [q.x, q.y, q.z, q.w]
            matrix = cloud.transform_matrix(translation, quat)
            if float(np.linalg.norm(translation)) > cloud.EXTRINSIC_MAX_TRANSLATION_M:
                raise cloud.CaptureError(f"implausible translation {translation}")
        except Exception as exc:  # noqa: BLE001 - tf2 raises several exception types
            self._raw_blocker = f"no tf {color_frame} <- {depth_frame}: {exc}"
            return None
        ext = {"parent_frame": color_frame, "child_frame": depth_frame, "translation": translation,
               "quaternion_xyzw": quat, "matrix": matrix.tolist(), "source": "tf_static_lookup"}
        self._extrinsics[key] = ext
        return ext

    def _ready_raw(self, fs: FrameSet) -> FrameSet | None:
        """A copy of the raw set with its depth info and extrinsic filled in, or None (blocker recorded)."""
        if self._info is None:
            self._raw_blocker = "no colour camera_info yet"
            return None
        size = (int(fs.depth.width), int(fs.depth.height))
        di, source = self._depth_infos.get(size), "camera_info"
        if di is None and self.raw_intrinsics is not None and self.raw_intrinsics[:2] == size:
            di, source = self._fallback_info(fs.depth.header.frame_id), "config_fallback"
        if di is None:
            have = ", ".join(f"{w}x{h}" for w, h in sorted(self._depth_infos)) or "none"
            fb = self.raw_intrinsics
            fallback = (f"; configured raw_depth_intrinsics fallback is for {fb[0]}x{fb[1]}, not the raw depth size "
                        f"(fallback size mismatch)" if fb is not None else "")
            self._raw_blocker = (f"no depth camera_info of the raw depth size {size[0]}x{size[1]} on "
                                 f"{self.depth_info_topic} (have: {have}){fallback}")
            return None
        ext = self._node_extrinsic(fs.color.header.frame_id, fs.depth.header.frame_id)
        if ext is None:
            return None
        self._raw_blocker = ""
        return dataclasses.replace(fs, depth_info=di, extrinsic=ext, depth_info_source=source)

    def grab(self, after_ns: int, timeout_s: float, abort: Callable[[], bool] | None = None) -> tuple[FrameSet, Any]:
        """First fresh set (colour AND depth stamp > after_ns + margin) of either kind, plus the latest CameraInfo."""
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
                best: FrameSet | None = None
                for fs in self._sets:
                    if fs.color_stamp_ns > threshold and fs.depth_stamp_ns > threshold and self._info is not None:
                        best = fs
                        break
                for fs in self._raw_sets:
                    if best is not None and fs.color_stamp_ns >= best.color_stamp_ns:
                        break       # the driver set is older (or the same frame): it wins
                    if fs.color_stamp_ns > threshold and fs.depth_stamp_ns > threshold:
                        ready = self._ready_raw(fs)
                        if ready is not None:
                            best = ready
                            break
                if best is not None:
                    cut = best.color_stamp_ns       # a frame is used once: drop it and everything older, both kinds
                    for dq in (self._sets, self._raw_sets):
                        while dq and dq[0].color_stamp_ns <= cut:
                            dq.popleft()
                    return best, self._info
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._cond.wait(min(remaining, 0.1))
            d = self._diag_locked()
        c = d["counts"]
        unaligned = (f"; {c['skipped_unaligned']} synced set(s) skipped because the depth is not aligned "
                     "to the colour image (driver alignment failing - colour frame rate / exposure?)"
                     if c["skipped_unaligned"] else "")
        if self.node_registration:
            rejected = {k[len("raw_rejected_"):]: v for k, v in c.items() if k.startswith("raw_rejected_")}
            nr = d["node_registration"]
            node = (f"; node registration: {c['raw_depth']} raw depth frame(s) on {self.raw_depth_topic}, "
                    f"{c['raw_synced']} synced with colour (slop {self.raw_sync_slop_s:g} s), "
                    f"captures from driver/node {c['captures_driver']}/{c['captures_node']}"
                    f"{', rejected ' + str(rejected) if rejected else ''}"
                    f"{', not usable: ' + nr['blocker'] if nr['blocker'] else ''}"
                    f"{', depth camera_info sizes ' + str(nr['depth_info_sizes']) if not nr['depth_info_sizes'] else ''}")
        else:
            node = "; node registration disabled"
        counts = {k: v for k, v in c.items() if v or k in ("rgb", "depth", "synced", "raw_synced")}
        raise CaptureTimeout(
            f"no fresh synced colour+depth frame (stamps > {threshold}) within {timeout_s:g} s{unaligned}{node}; "
            f"received {counts}, camera_info={'yes' if d['have_camera_info'] else 'NO'}, "
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
    def _register_node_set(self, fs: FrameSet, info) -> tuple[np.ndarray, float, tuple, dict, dict]:
        """Register the raw depth of a node set into the colour grid. Returns (depth, scale, K, header, block).

        ``header`` is the pointcloud header for the manifest (colour frame, raw depth stamp: the registered depth
        lives in the colour optical frame); ``block`` the manifest ``depth_registration`` entry."""
        color, raw, di = fs.color, fs.depth, fs.depth_info
        scale = cloud.DEPTH_ENCODING_SCALES.get(raw.encoding)
        if scale is None:
            raise cloud.CaptureError(f"raw depth encoding must be 16UC1 (mm) or 32FC1 (m), got {raw.encoding!r}")
        if color.encoding not in cloud.COLOR_ENCODINGS:
            raise cloud.CaptureError(f"colour image encoding must be rgb8/bgr8, got {color.encoding!r}")
        if info is None:
            raise cloud.CaptureError("no colour CameraInfo received")
        if di is None or fs.extrinsic is None:
            raise cloud.CaptureError("node set without depth CameraInfo / extrinsic (internal error)")
        if (info.width, info.height) != (color.width, color.height):
            raise cloud.CaptureError(
                f"colour CameraInfo {info.width}x{info.height} != colour image {color.width}x{color.height}: the "
                "node registration needs the exact colour pixel grid")
        if info.header.frame_id != color.header.frame_id:
            raise cloud.CaptureError(
                f"colour CameraInfo frame {info.header.frame_id!r} != colour image frame {color.header.frame_id!r}")
        if (raw.width, raw.height) != (di.width, di.height):
            raise cloud.CaptureError(
                f"depth CameraInfo {di.width}x{di.height} != raw depth image {raw.width}x{raw.height}")
        if (raw.width, raw.height) == (color.width, color.height):
            raise cloud.CaptureError("raw depth has the colour size: ambiguous CameraInfo, node registration refused")
        if raw.header.frame_id == color.header.frame_id:
            raise cloud.CaptureError("raw depth and colour share a frame_id: node registration refused")
        k = tuple(float(v) for v in info.k)
        if k[0] <= 0 or k[4] <= 0:
            raise cloud.CaptureError(f"invalid colour CameraInfo K {k}")
        depth_raw = cloud.image_array(raw)
        stats: dict = {}
        t0 = time.monotonic()
        registered = cloud.register_depth_to_color(
            depth_raw, scale, di.k, di.d, info.k, info.d, info.distortion_model,
            np.asarray(fs.extrinsic["matrix"], dtype=np.float64), color.width, color.height, stats=stats)
        elapsed = time.monotonic() - t0
        header = {"frame_id": color.header.frame_id, "stamp_ns": fs.depth_stamp_ns}
        block = {
            "source": "node", "raw_depth_topic": self.raw_depth_topic, "depth_info_topic": self.depth_info_topic,
            "intrinsics_source": fs.depth_info_source,
            "raw_depth_header": cloud.header_metadata(raw.header),
            "raw_depth": {"width": raw.width, "height": raw.height, "encoding": raw.encoding},
            "color_depth_dt_ms": (fs.color_stamp_ns - fs.depth_stamp_ns) / 1e6,
            "raw_sync_slop_s": self.raw_sync_slop_s,
            "depth_k": [float(v) for v in di.k], "depth_d": [float(v) for v in di.d],
            "depth_info_header": cloud.header_metadata(di.header),
            "t_color_depth": {key: fs.extrinsic[key] for key in
                              ("parent_frame", "child_frame", "translation", "quaternion_xyzw", "source")},
            "method": "numpy footprint splat (4 corners, grown by "
                      f"{cloud.FOOTPRINT_MARGIN_PX} px), z-buffer nearest wins, colour distortion "
                      f"{info.distortion_model}",
            "stats": {**stats, "seconds": round(elapsed, 3)},
        }
        return registered, scale, k, header, block

    def write_capture(self, fs: FrameSet, info) -> Any:
        depth_header = None
        topics = self.topics
        if fs.source == "node":
            depth, scale, k, depth_header, reg_block = self._register_node_set(fs, info)
            width, height = fs.color.width, fs.color.height
            scaled_from = None
            topics = {**self.topics, "depth": self.raw_depth_topic}
            depth_encoding = fs.depth.encoding
        else:
            scale, k, width, height = cloud.validate_registration(fs.color, fs.depth, info)
            scaled_from = None if (info.width, info.height) == (width, height) else (info.width, info.height)
            depth = cloud.image_array(fs.depth)
            depth_encoding = fs.depth.encoding
            reg_block = {"source": "driver", "depth_topic": self.topics["depth"],
                         "color_depth_dt_ms": (fs.color_stamp_ns - fs.depth_stamp_ns) / 1e6,
                         "sync_slop_s": self.sync_slop_s}
        rgb = cloud.rgb_from_color_msg(fs.color)
        points = cloud.backproject_depth(depth, rgb, k, scale)
        if points.shape[0] == 0:
            raise cloud.CaptureError("registered depth image has no valid pixel" if fs.source == "node"
                                     else "aligned depth image has no valid pixel")
        self._seq += 1
        cid = f"{self.prefix}_{self._seq:06d}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        try:
            image_path = cloud.write_png(str(self.dirs["rgb"] / f"{cid}.png"), rgb)
            pcd_path = cloud.write_pcd(str(self.dirs["pointcloud"] / f"{cid}.pcd"), points)
            ext = self.driver_extrinsic(fs.color.header.frame_id, fs.color_stamp_ns)
            manifest = cloud.build_manifest(cid, image_path, pcd_path, fs.color, fs.depth, info, k,
                                            int(points.shape[0]), topics, ext, k_scaled_from=scaled_from,
                                            depth_header=depth_header, depth_registration=reg_block)
            mpath = cloud.write_manifest(str(self.dirs["captures"] / f"{cid}.json"), manifest)
        except BaseException:
            if self.staged:     # RAM staging: drop what this capture wrote (nothing else would ever archive it)
                for sub, ext_ in (("rgb", ".png"), ("pointcloud", ".pcd"), ("captures", ".json")):
                    (self.dirs[sub] / f"{cid}{ext_}").unlink(missing_ok=True)
            raise
        with self._cond:
            self._counts[f"captures_{fs.source}"] += 1
            if fs.source == "node":
                self._last_intrinsics_source = fs.depth_info_source
        return self.files_cls(capture_id=cid, image_path=Path(image_path), pcd_path=Path(pcd_path),
                              manifest_path=Path(mpath), stamp_ns=fs.color_stamp_ns)

    def capture(self, after_ns: int | None = None, timeout_s: float = 10.0,
                abort: Callable[[], bool] | None = None) -> Any:
        """grab a fresh set (after_ns defaults to now) and write rgb/pcd/manifest; returns files_cls(...)."""
        if after_ns is None:
            after_ns = self.now_ns()
        fs, info = self.grab(after_ns, timeout_s, abort)
        return self.write_capture(fs, info)
