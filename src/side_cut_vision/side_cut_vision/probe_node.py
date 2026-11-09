#!/usr/bin/env python3
"""side_cut_probe: STRICTLY PASSIVE side-cut camera probe (Stage A, measure only).

Subscribes to TF, /joint_states, the coordinator status and the knife-holder Gemini 305 colour/depth/
camera_info of each enabled arm; publishes ONLY a JPEG overlay and a JSON measurement per arm
(/side_cut_vision/<arm>/overlay/compressed, /side_cut_vision/<arm>/measurement). It has no service
clients, no action clients, no services, no parameters (arguments come from the command line) and no
motion interface (a test enforces this on the source). The robot domain is the process environment's
(ROS_DOMAIN_ID); the cameras may be in another domain (--camera-domain-id), read in a second rclpy context.

Run (real cell, probe in the robot environment): see README.md.
"""
from __future__ import annotations

import os as _os

# <= 1 core in total (the cell's cameras starve under load): one BLAS/OpenMP thread, one OpenCV thread
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_var, "1")

import argparse
import os
import signal
import sys
import threading
import time
from typing import Dict, List, Optional

import numpy as np

from .calib import CalibError, load_hand_eye
from .camera import CameraError, CameraModel
from .domains import MAX_DOMAIN_ID, plan_domains, startup_message
from .probe_core import (ArmConfig, ArmProbe, Env, LiveGate, SnapshotCache, Stamped, matrix_from_tf)
from .trigger import default_out_dir

DEFAULT_SNAPSHOT = "/home/farhan/Igus_rebel/outputs/seam_3d_base_link.json"
DEFAULT_LEFT_CALIB = "/home/farhan/igus_rebel_ros2/scan_results/wrist_handeye_20261013_112818/result.yaml"
COORDINATOR_STATUS = "/dual_cut_coordinator/status"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Passive side-cut camera probe (measure only; subscribers + overlay publisher).")
    p.add_argument("--camera-domain-id", type=int, required=True,
                   help=f"ROS domain of the cameras (0..{MAX_DOMAIN_ID}); -1 = the robot domain (mock only)")
    p.add_argument("--arms", default="left,right", help="comma list of left,right (default both)")
    p.add_argument("--left-calib", default=DEFAULT_LEFT_CALIB, help="wrist_handeye/v1 YAML of the left arm")
    p.add_argument("--right-calib", default="", help="wrist_handeye/v1 YAML of the right arm; empty = SEED from the "
                   "left calibration (flagged seed_from_left_UNCALIBRATED everywhere)")
    p.add_argument("--snapshot-file", default=DEFAULT_SNAPSHOT)
    p.add_argument("--out-dir", default=None, help="cut_recorder output dir (default: its default)")
    p.add_argument("--live-rate-hz", type=float, default=3.0)
    p.add_argument("--overlay-context", choices=["robot", "camera"], default="robot",
                   help="ROS context that publishes the overlay/measurement (default robot: the recorder's domain)")
    p.add_argument("--always-live", action="store_true", help="process also while the coordinator is idle")
    p.add_argument("--world-frame", default="world")
    p.add_argument("--fit-stride", type=int, default=2)
    p.add_argument("--status-topic", default=COORDINATOR_STATUS)
    p.add_argument("--camera-topic-prefix", default="/arm_{arm}/wrist_camera",
                   help="per-arm camera topic prefix ({arm} = left|right)")
    p.add_argument("--color-topic", default="color/image_raw",
                   help="colour topic below the prefix; a name ending in /compressed is read as JPEG")
    p.add_argument("--depth-topic", default="depth/image_raw")
    p.add_argument("--info-topic", default="color/camera_info")
    return p


def stamp_s(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def decode_color(msg) -> Optional[np.ndarray]:
    """sensor_msgs/Image or CompressedImage -> RGB uint8 (H, W, 3); None if unusable."""
    import cv2
    if hasattr(msg, "format"):                                   # CompressedImage
        bgr = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        return None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if msg.encoding not in ("rgb8", "bgr8") or msg.step < msg.width * 3:
        return None
    a = np.frombuffer(msg.data, np.uint8)
    if a.size < msg.step * msg.height:
        return None
    a = a[:msg.step * msg.height].reshape(msg.height, msg.step)[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
    return a.copy() if msg.encoding == "rgb8" else np.ascontiguousarray(a[:, :, ::-1])


def decode_depth(msg) -> Optional[np.ndarray]:
    if msg.encoding not in ("16UC1", "mono16") or msg.step < msg.width * 2:
        return None
    a = np.frombuffer(msg.data, np.uint16)
    if a.size * 2 < msg.step * msg.height:
        return None
    return a[:(msg.step // 2) * msg.height].reshape(msg.height, msg.step // 2)[:, :msg.width].copy()


class RosEnv(Env):
    def __init__(self, buffer, world_unused=""):
        self.buffer = buffer

    def lookup(self, target: str, source: str, stamp_s_: Optional[float]):
        from rclpy.duration import Duration
        from rclpy.time import Time
        try:
            t = Time() if stamp_s_ is None else Time(nanoseconds=int(stamp_s_ * 1e9))
            tf = self.buffer.lookup_transform(target, source, t, timeout=Duration(seconds=0.0))
        except Exception:                          # noqa: BLE001 - tf2 raises many types; any failure = unavailable
            return None
        tr, q = tf.transform.translation, tf.transform.rotation
        try:
            T = matrix_from_tf([tr.x, tr.y, tr.z], [q.x, q.y, q.z, q.w])
        except ValueError:
            return None
        return T if np.all(np.isfinite(T)) else None

    def lookup_latest(self, target: str, source: str):
        from rclpy.duration import Duration
        from rclpy.time import Time
        try:
            tf = self.buffer.lookup_transform(target, source, Time(), timeout=Duration(seconds=0.0))
        except Exception:                          # noqa: BLE001
            return None
        tr, q = tf.transform.translation, tf.transform.rotation
        try:
            T = matrix_from_tf([tr.x, tr.y, tr.z], [q.x, q.y, q.z, q.w])
        except ValueError:
            return None
        return (T, stamp_s(tf.header.stamp)) if np.all(np.isfinite(T)) else None

    def now_s(self) -> float:
        return time.time()


class ProbeNode:
    def __init__(self, args):
        import rclpy
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, HistoryPolicy
        from rclpy.signals import SignalHandlerOptions
        from sensor_msgs.msg import CameraInfo, CompressedImage, Image, JointState
        from std_msgs.msg import String
        from tf2_ros import Buffer, TransformListener
        self.rclpy = rclpy
        self.CompressedImage, self.String, self.Image = CompressedImage, String, Image
        self.args = args
        self.log = lambda m: print(f"[side_cut_probe] {m}", flush=True)
        rclpy.init(args=[], signal_handler_options=SignalHandlerOptions.NO)
        node_domain = rclpy.get_default_context().get_domain_id()
        self.plan = plan_domains(node_domain, args.camera_domain_id, -1)
        self.log(startup_message(self.plan, node_domain))
        self.node = Node("side_cut_probe", cli_args=[], use_global_arguments=False,
                         start_parameter_services=False, enable_rosout=False)
        self._drop_parameter_event_publisher(self.node)
        self.contexts = [rclpy.get_default_context()]
        self.executors = []
        self.cam_ctx = None
        if self.plan.separate_camera_context:
            self.cam_ctx = Context()
            rclpy.init(args=[], context=self.cam_ctx, domain_id=self.plan.camera_domain,
                       signal_handler_options=SignalHandlerOptions.NO)
            self.cam_node = Node("side_cut_probe_cameras", context=self.cam_ctx, cli_args=[],
                                 use_global_arguments=False, enable_rosout=False,
                                 start_parameter_services=False)
            self._drop_parameter_event_publisher(self.cam_node)
        else:
            self.cam_node = self.node
        self.pub_node = self.cam_node if args.overlay_context == "camera" else self.node

        self.buffer = Buffer()
        self.tfl = TransformListener(self.buffer, self.node, spin_thread=False)
        self.env = RosEnv(self.buffer)
        self.gate = LiveGate(args.always_live)
        out_dir = args.out_dir or default_out_dir()
        self.snapshots = SnapshotCache(args.snapshot_file)
        self.probes: Dict[str, ArmProbe] = {}
        self.stream_cfg: Dict[str, tuple] = {}
        self.stream_subs: Dict[str, list] = {}
        self.stream_idle_since: Dict[str, Optional[float]] = {}
        self.qos_sensor = qos_profile_sensor_data
        self.pubs: Dict[str, tuple] = {}
        self.color_compressed = args.color_topic.endswith("/compressed")
        left_he = None
        for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
            if arm not in ("left", "right"):
                raise ValueError(f"unknown arm {arm!r}")
            if arm == "left" or args.right_calib:
                path = args.left_calib if arm == "left" else args.right_calib
                he = load_hand_eye(path, seed_from_left=False)
            else:
                he = load_hand_eye(args.left_calib, seed_from_left=True)
            cfg = ArmConfig.default(arm, args.world_frame)
            cfg.calib_status = he.calib_status
            self.probes[arm] = ArmProbe(cfg, he, self.snapshots, out_dir, args.snapshot_file,
                                        live_rate_hz=args.live_rate_hz, gate=self.gate)
            self.probes[arm].params.fit_stride = args.fit_stride
            self.log(f"arm {arm}: calib {he.calib_status} ({he.path}, sha256 {he.sha256[:12]})")
            pre = args.camera_topic_prefix.format(arm=arm)
            # colour/depth subscriptions are created only while the probe is live (see _sync_subscriptions):
            # 5 MB frames at 15 Hz cost 20-25 % of a core in DDS reassembly alone, even if every frame is dropped.
            self.stream_cfg[arm] = (f"{pre}/{args.color_topic}", f"{pre}/{args.depth_topic}")
            self.cam_node.create_subscription(CameraInfo, f"{pre}/{args.info_topic}", self._info_cb(arm),
                                              qos_profile_sensor_data)
            self.pubs[arm] = (
                self.pub_node.create_publisher(CompressedImage, f"/side_cut_vision/arm_{arm}/overlay/compressed", 2),
                self.pub_node.create_publisher(String, f"/side_cut_vision/arm_{arm}/measurement", 5))
            self.log(f"arm {arm}: {pre}/{args.color_topic} + {args.depth_topic} (while live) + {args.info_topic}; "
                     f"publishing /side_cut_vision/arm_{arm}/overlay/compressed (+/measurement)")
        self.node.create_subscription(JointState, "/joint_states", self._joint_cb, 20)
        from dual_cut_msgs.msg import CoordinatorStatus
        self.node.create_subscription(CoordinatorStatus, args.status_topic, self._status_cb, 10)
        self.stop = threading.Event()
        self.threads: List[threading.Thread] = []
        for ctx, nodes in ((rclpy.get_default_context(), [self.node]),
                           *([(self.cam_ctx, [self.cam_node])] if self.cam_ctx is not None else [])):
            ex = SingleThreadedExecutor(context=ctx)
            for n in nodes:
                ex.add_node(n)
            self.executors.append(ex)
            self.threads.append(threading.Thread(target=ex.spin, name="ros-executor", daemon=True))
        self.worker = threading.Thread(target=self._work, name="probe-worker", daemon=True)
        self.out_dir = out_dir
        self.log(f"record out dir: {out_dir}; snapshot: {args.snapshot_file}; live rate {args.live_rate_hz} Hz; "
                 f"gate: {'always' if args.always_live else 'coordinator WAITING_BARRIER/EXECUTING'}")

    @staticmethod
    def _drop_parameter_event_publisher(node) -> None:
        try:
            node.destroy_publisher(node._parameter_event_publisher)
            node._parameter_event_publisher = None
        except Exception:                          # noqa: BLE001
            pass

    # ------------------------------------------------------------------ callbacks (cheap)
    def _sync_subscriptions(self) -> None:
        """Subscribe to colour+depth of an arm while the gate is active or a LOOK is pending; drop the
        subscriptions after the gate has been idle for 5 s (the gate itself already holds 3 s)."""
        now = time.time()
        live = self.gate.active(now)
        for arm, probe in self.probes.items():
            want = live or probe.look is not None
            have = arm in self.stream_subs
            if want:
                self.stream_idle_since[arm] = None
                if not have:
                    ctopic, dtopic = self.stream_cfg[arm]
                    self.stream_subs[arm] = [
                        self.cam_node.create_subscription(
                            self.CompressedImage if self.color_compressed else self.Image, ctopic,
                            self._color_cb(arm), self.qos_sensor, raw=True),
                        self.cam_node.create_subscription(self.Image, dtopic, self._depth_cb(arm),
                                                          self.qos_sensor, raw=True)]
                    self.log(f"arm {arm}: live, subscribed to {ctopic} + {dtopic}")
            elif have:
                since = self.stream_idle_since.get(arm)
                if since is None:
                    self.stream_idle_since[arm] = now
                elif now - since > 5.0:
                    for sub in self.stream_subs.pop(arm):
                        self.cam_node.destroy_subscription(sub)
                    probe.pairs.color.clear()
                    probe.pairs.depth.clear()
                    self.log(f"arm {arm}: idle, colour/depth subscriptions released")

    def _accept(self, probe, data: bytes, key: tuple) -> bool:
        """Gate only: frames are used while the probe is live or a LOOK is pending. (A stamp-based decimation of
        the two streams was tried and dropped: one lost message shifts the phase of one stream and every later
        colour/depth pair is then 'stale'. Pairing by nearest stamp is robust; the cost of decoding every frame
        is ~10 % of a core and is paid only while live, because the subscriptions exist only then.)"""
        return probe.look is not None or self.gate.active(time.time())

    def _color_cb(self, arm):
        probe = self.probes[arm]
        from rclpy.serialization import deserialize_message
        msg_type = self.CompressedImage if self.color_compressed else self.Image

        def cb(data):
            if not self._accept(probe, data, (arm, "color")):
                return
            msg = deserialize_message(data, msg_type)
            img = decode_color(msg)
            if img is not None:
                probe.pairs.add_color(Stamped(stamp_s(msg.header.stamp), img, msg.header.frame_id))
        return cb

    def _depth_cb(self, arm):
        probe = self.probes[arm]
        from rclpy.serialization import deserialize_message

        def cb(data):
            if not self._accept(probe, data, (arm, "depth")):
                return
            msg = deserialize_message(data, self.Image)
            d = decode_depth(msg)
            if d is not None:
                probe.pairs.add_depth(Stamped(stamp_s(msg.header.stamp), d, msg.header.frame_id))
        return cb

    def _info_cb(self, arm):
        probe = self.probes[arm]

        def cb(msg):
            if probe.cam is not None and probe.cam.size == (msg.width, msg.height):
                return
            try:
                probe.set_camera(CameraModel(np.array(msg.k).reshape(3, 3), list(msg.d), msg.width, msg.height,
                                             frame_id=msg.header.frame_id))
            except CameraError as exc:
                self.log(f"arm {arm}: bad camera_info: {exc}")
        return cb

    def _joint_cb(self, msg) -> None:
        t = stamp_s(msg.header.stamp) or time.time()
        for p in self.probes.values():
            p.on_joint_state(t, msg.name, msg.position, msg.velocity)

    def _status_cb(self, msg) -> None:
        self.gate.on_status(msg.state, time.time())

    # ------------------------------------------------------------------ worker
    def _work(self) -> None:
        last_report = time.time()
        while not self.stop.is_set():
            try:
                self._sync_subscriptions()
            except Exception as exc:                 # noqa: BLE001 - never kill the worker
                self.log(f"subscription sync error: {exc}")
            for arm, probe in self.probes.items():
                for o in probe.step(self.env):
                    for line in o.log:
                        self.log(line)
                    if o.overlay_jpeg is not None:
                        self._publish(arm, o)
            if time.time() - last_report > 30:
                last_report = time.time()
                for arm, p in self.probes.items():
                    self.log(f"{arm}: {p.counters} snapshot reloads {self.snapshots.reloads} "
                             f"err={p.last_error} snap_err={self.snapshots.error}")
            self.stop.wait(0.02)

    def _publish(self, arm: str, o) -> None:
        pub_img, pub_meas = self.pubs[arm]
        m = self.CompressedImage()
        sec = int(o.overlay_stamp)
        m.header.stamp.sec, m.header.stamp.nanosec = sec, int((o.overlay_stamp - sec) * 1e9)
        m.header.frame_id = f"side_cut_vision_{arm}"
        m.format = "jpeg"
        m.data = o.overlay_jpeg
        pub_img.publish(m)
        s = self.String()
        s.data = o.measurement_json
        pub_meas.publish(s)

    def run(self) -> int:
        for t in self.threads:
            t.start()
        self.worker.start()
        try:
            while not self.stop.is_set():
                self.stop.wait(0.5)
        finally:
            self.shutdown()
        return 0

    def shutdown(self) -> None:
        self.stop.set()
        self.worker.join(timeout=3.0)
        for ex in self.executors:
            ex.shutdown()
        for t in self.threads:
            t.join(timeout=1.0)
        try:
            self.node.destroy_node()
            if self.cam_ctx is not None:
                self.cam_node.destroy_node()
                self.cam_ctx.shutdown()
        finally:
            if self.rclpy.ok():
                self.rclpy.shutdown()
        self.log("stopped")


def main(argv=None) -> int:
    import cv2
    cv2.setNumThreads(1)
    args = build_parser().parse_args(argv)
    if not (-1 <= args.camera_domain_id <= MAX_DOMAIN_ID):
        print(f"side_cut_probe: --camera-domain-id must be -1..{MAX_DOMAIN_ID}", file=sys.stderr)
        return 2
    if args.live_rate_hz <= 0 or args.live_rate_hz > 15:
        print("side_cut_probe: --live-rate-hz must be in (0, 15]", file=sys.stderr)
        return 2
    try:
        node = ProbeNode(args)
    except (CalibError, ValueError) as exc:
        print(f"side_cut_probe: {exc}", file=sys.stderr)
        return 2
    signal.signal(signal.SIGINT, lambda *_: node.stop.set())
    signal.signal(signal.SIGTERM, lambda *_: node.stop.set())
    return node.run()


if __name__ == "__main__":
    sys.exit(main())
