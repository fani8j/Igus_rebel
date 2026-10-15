#!/usr/bin/env python3
"""Passive camera recorder for igus ReBeL carton cuts.

Records the knife, scene and wrist cameras to mp4 while a trajectory is
executing on the arm, and writes one folder per execution.

STRICTLY PASSIVE: this tool only subscribes.  It has no publishers, no service
clients, no action clients and never sets a parameter, so it cannot influence
the robot or the rest of the stack.  (rclpy itself creates an internal
parameter-event channel for every node; this tool tears it down right after
creating the node and never declares or sets a parameter.)

Trigger
  Subscribes to the action status topic of the arm's FollowJointTrajectory
  action server.  A clip opens when any goal is ACCEPTED/EXECUTING and closes
  when no goal has been ACCEPTED/EXECUTING for --post-roll-s seconds, so
  back-to-back trajectories of one execution end up in one clip.
  --max-clip-s hard-closes a clip.  --now records immediately until Ctrl-C.

Output (per clip, folder named by local time YYYYmmdd_HHMMSS)
  <camera>.mp4   one file per camera that delivered frames
  clip.json      timing, trigger, goal ids and per-camera frame statistics
  seam_3d_base_link.json   copy of the current snapshot (best effort)

Camera spec syntax (--camera, repeatable; overrides a default of that name):
  name=topic[:type[:fps[:scale]]]     type = raw (sensor_msgs/Image) or
                                      compressed (sensor_msgs/CompressedImage)
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import queue
import shutil
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

try:  # cv2 is only needed to decode/encode; keep import errors readable
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

DEFAULT_STATUS_TOPIC = (
    '/rebel_arm_trajectory_controller/follow_joint_trajectory/_action/status')
DEFAULT_SNAPSHOT = '/home/farhan/Igus_rebel/outputs/seam_3d_base_link.json'
EXTERNAL_DRIVE = '/media/farhan/My Passport'
EXTERNAL_OUT_DIR = EXTERNAL_DRIVE + '/igus_nvidia/cut_videos'
FALLBACK_OUT_DIR = '/home/farhan/Igus_rebel/outputs/cut_videos'

STATUS_ACCEPTED = 1
STATUS_EXECUTING = 2
ACTIVE_STATUSES = (STATUS_ACCEPTED, STATUS_EXECUTING)

# A header stamp further than this from the receipt time is not trusted.
MAX_STAMP_SKEW_S = 5.0
# Queue length per camera (frames); a full queue drops and counts frames.
DEFAULT_QUEUE_LEN = 90


# --------------------------------------------------------------------------
# Camera specs
# --------------------------------------------------------------------------
@dataclass
class CameraSpec:
    name: str
    topic: str
    kind: str = 'raw'          # 'raw' | 'compressed'
    fps: float = 15.0
    scale: float = 1.0


DEFAULT_CAMERAS = {
    'knife': CameraSpec('knife', '/knife_camera/image_raw/compressed',
                        'compressed', 30.0, 1.0),
    # Orbbec topics via image_transport's compressed plugin (JPEG): far less DDS
    # traffic than raw rgb8 frames, which arrived with drops in the mock test.
    'scene': CameraSpec('scene', '/camera/color/image_raw/compressed',
                        'compressed', 15.0, 0.5),
    # Gemini 305 on the left arm's knife (rebel_cameras wrist_camera.yaml).
    'wrist': CameraSpec('wrist', '/wrist_camera/color/image_raw/compressed',
                        'compressed', 15.0, 1.0),
    # Gemini 305 on the right arm's knife (rebel_cameras right_wrist_camera.yaml).
    'right_wrist': CameraSpec('right_wrist',
                              '/right_wrist_camera/color/image_raw/compressed',
                              'compressed', 15.0, 1.0),
}


def parse_camera_spec(text: str) -> CameraSpec:
    """Parse ``name=topic[:type[:fps[:scale]]]``."""
    if '=' not in text:
        raise ValueError(f'camera spec {text!r}: expected name=topic[:type[:fps[:scale]]]')
    name, rest = text.split('=', 1)
    name = name.strip()
    if not name or not all(c.isalnum() or c in '_-' for c in name):
        raise ValueError(f'camera spec {text!r}: bad camera name {name!r}')
    parts = rest.split(':')
    topic = parts[0].strip()
    if not topic:
        raise ValueError(f'camera spec {text!r}: empty topic')
    default_kind = 'compressed' if topic.endswith('/compressed') else 'raw'
    kind = parts[1].strip() if len(parts) > 1 and parts[1].strip() else default_kind
    if kind not in ('raw', 'compressed'):
        raise ValueError(f'camera spec {text!r}: type must be raw or compressed')
    base = DEFAULT_CAMERAS.get(name)
    fps = float(parts[2]) if len(parts) > 2 and parts[2].strip() else (
        base.fps if base else 15.0)
    scale = float(parts[3]) if len(parts) > 3 and parts[3].strip() else (
        base.scale if base else 1.0)
    if len(parts) > 4:
        raise ValueError(f'camera spec {text!r}: too many fields')
    if not (0 < fps <= 240):
        raise ValueError(f'camera spec {text!r}: fps out of range')
    if not (0 < scale <= 4):
        raise ValueError(f'camera spec {text!r}: scale out of range')
    return CameraSpec(name, topic, kind, fps, scale)


# --------------------------------------------------------------------------
# Image decoding (no cv_bridge)
# --------------------------------------------------------------------------
_CHANNELS = {'rgb8': 3, 'bgr8': 3, 'mono8': 1, '8UC1': 1, '8UC3': 3}


def image_to_bgr(height: int, width: int, step: int, encoding: str,
                 data) -> np.ndarray:
    """Convert raw sensor_msgs/Image fields to a contiguous BGR uint8 array."""
    ch = _CHANNELS.get(encoding)
    if ch is None:
        raise ValueError(f'unsupported image encoding {encoding!r}')
    row = width * ch
    if step < row:
        raise ValueError(f'step {step} < width*channels {row}')
    buf = np.frombuffer(data, dtype=np.uint8)
    if buf.size < step * height:
        raise ValueError(f'image data too short: {buf.size} < {step * height}')
    img = buf[:step * height].reshape(height, step)[:, :row]
    if ch == 1:
        return cv2.cvtColor(img.reshape(height, width), cv2.COLOR_GRAY2BGR)
    img = img.reshape(height, width, 3)
    if encoding in ('rgb8', '8UC3'):
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return np.ascontiguousarray(img)


def compressed_to_bgr(data) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError('cv2.imdecode failed')
    return img


def stamp_to_s(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def make_msg_decoder(spec: CameraSpec) -> Callable[[bytes], Tuple[float, np.ndarray]]:
    """Return decode(serialized_bytes) -> (header_stamp_s, bgr). Needs rclpy."""
    from rclpy.serialization import deserialize_message
    if spec.kind == 'compressed':
        from sensor_msgs.msg import CompressedImage as MsgT

        def decode(raw: bytes):
            msg = deserialize_message(raw, MsgT)
            return stamp_to_s(msg.header.stamp), compressed_to_bgr(msg.data)
    else:
        from sensor_msgs.msg import Image as MsgT

        def decode(raw: bytes):
            msg = deserialize_message(raw, MsgT)
            return (stamp_to_s(msg.header.stamp),
                    image_to_bgr(msg.height, msg.width, msg.step,
                                 msg.encoding, msg.data))
    return decode


# --------------------------------------------------------------------------
# Pure logic: frame index mapping
# --------------------------------------------------------------------------
@dataclass
class FramePlan:
    write_new: bool = False     # write the new frame at all?
    repeat_prev: int = 0        # copies of the previous frame written first
    repeat_new: int = 0         # copies of the new frame written
    index: int = -1             # frame index of (last copy of) the new frame

    @property
    def duplicated(self) -> int:
        """Extra frames written beyond one per accepted new frame."""
        if not self.write_new:
            return 0
        return self.repeat_prev + max(0, self.repeat_new - 1)


class FrameMapper:
    """Maps frame times (s since clip start) to constant-fps frame indices.

    index = round(t * fps).  A frame whose index was already written is
    dropped; a gap is filled by repeating the previous frame.  The very first
    frame is repeated to fill the lead-in so every camera starts at clip
    start (keeps cameras aligned in wall time).
    """

    def __init__(self, fps: float):
        if fps <= 0:
            raise ValueError('fps must be > 0')
        self.fps = float(fps)
        self.next_index = 0   # number of frames written so far

    def plan(self, t_rel_s: float) -> FramePlan:
        idx = max(0, int(math.floor(t_rel_s * self.fps + 0.5)))
        if self.next_index == 0:
            # lead-in padding with the first frame itself
            self.next_index = idx + 1
            return FramePlan(True, 0, idx + 1, idx)
        if idx < self.next_index:
            return FramePlan(False, 0, 0, idx)
        gap = idx - self.next_index
        self.next_index = idx + 1
        return FramePlan(True, gap, 1, idx)


# --------------------------------------------------------------------------
# Pure logic: clip state machine
# --------------------------------------------------------------------------
class ClipStateMachine:
    """Opens a clip on activity, closes it after post-roll idle time."""

    def __init__(self, post_roll_s: float = 3.0, max_clip_s: float = 900.0):
        self.post_roll_s = float(post_roll_s)
        self.max_clip_s = float(max_clip_s)
        self.is_open = False
        self.start_t = 0.0
        self.last_active_t = 0.0

    def update(self, now: float, active: bool) -> Optional[Tuple[str, str]]:
        """Returns ('start'|'stop', reason) or None."""
        if self.is_open:
            if active:
                self.last_active_t = now
            if now - self.start_t >= self.max_clip_s:
                self.is_open = False
                return ('stop', 'max_clip')
            if not active and now - self.last_active_t >= self.post_roll_s:
                self.is_open = False
                return ('stop', 'post_roll')
            return None
        if active:
            self.is_open = True
            self.start_t = now
            self.last_active_t = now
            return ('start', 'status')
        return None

    def force_open(self, now: float) -> None:
        self.is_open = True
        self.start_t = now
        self.last_active_t = now

    def force_close(self) -> None:
        self.is_open = False


# --------------------------------------------------------------------------
# Per-camera recorder (worker thread)
# --------------------------------------------------------------------------
def _cv_writer_factory(path: str, fps: float, size: Tuple[int, int]):
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    if not w.isOpened():
        raise RuntimeError(f'cv2.VideoWriter failed to open {path}')
    return w


class CameraRecorder:
    """Decodes, scales and writes frames of one camera on its own thread."""

    def __init__(self, spec: CameraSpec, path: str, clip_start_s: float,
                 decode: Callable[[bytes], Tuple[float, np.ndarray]],
                 writer_factory=_cv_writer_factory,
                 queue_len: int = DEFAULT_QUEUE_LEN, threaded: bool = True):
        self.spec = spec
        self.path = path
        self.clip_start_s = clip_start_s
        self._decode = decode
        self._writer_factory = writer_factory
        self._writer = None
        self._size: Optional[Tuple[int, int]] = None
        self._mapper = FrameMapper(spec.fps)
        self._last_frame: Optional[np.ndarray] = None
        self._q: queue.Queue = queue.Queue(maxsize=queue_len)
        self._lock = threading.Lock()
        self.received = 0
        self.written = 0
        self.duplicated = 0
        self.dropped_queue_full = 0
        self.dropped_same_index = 0
        self.decode_errors = 0
        self.stamp_fallbacks = 0
        self.write_error: Optional[str] = None
        self.tail_padded = 0
        self.first_frame_offset_s: Optional[float] = None
        self._thread = None
        if threaded:
            self._thread = threading.Thread(
                target=self._run, name=f'rec_{spec.name}', daemon=True)
            self._thread.start()

    # called from the ROS callback: never blocks
    def submit(self, receipt_s: float, payload: bytes) -> None:
        self.received += 1
        try:
            self._q.put_nowait((receipt_s, payload))
        except queue.Full:
            self.dropped_queue_full += 1

    def _run(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            try:
                self.process(*item)
            except Exception as exc:  # keep the worker alive
                self.write_error = f'{type(exc).__name__}: {exc}'

    def process(self, receipt_s: float, payload: bytes) -> None:
        try:
            stamp_s, frame = self._decode(payload)
        except Exception:
            self.decode_errors += 1
            return
        if stamp_s <= 0.0 or abs(stamp_s - receipt_s) > MAX_STAMP_SKEW_S:
            if stamp_s > 0.0:
                self.stamp_fallbacks += 1
            stamp_s = receipt_s
        self.add_frame(stamp_s, frame)

    def add_frame(self, stamp_s: float, frame: np.ndarray) -> None:
        if self.spec.scale != 1.0:
            frame = cv2.resize(frame, None, fx=self.spec.scale,
                               fy=self.spec.scale, interpolation=cv2.INTER_AREA)
        if self._writer is None:
            h, w = frame.shape[:2]
            self._size = (w - (w % 2), h - (h % 2))   # even dims for codecs
            self._writer = self._writer_factory(self.path, self.spec.fps,
                                                self._size)
        if (frame.shape[1], frame.shape[0]) != self._size:
            frame = cv2.resize(frame, self._size, interpolation=cv2.INTER_AREA)
        t_rel = stamp_s - self.clip_start_s
        plan = self._mapper.plan(t_rel)
        if not plan.write_new:
            self.dropped_same_index += 1
            return
        if self.first_frame_offset_s is None:
            self.first_frame_offset_s = round(max(0.0, t_rel), 3)
        for _ in range(plan.repeat_prev):
            self._writer.write(self._last_frame)
        for _ in range(plan.repeat_new):
            self._writer.write(frame)
        self._last_frame = frame
        self.written += 1
        self.duplicated += plan.duplicated

    def close(self, join_timeout_s: float = 30.0,
              stop_rel_s: Optional[float] = None) -> dict:
        """Flush, optionally pad the tail up to ``stop_rel_s`` (seconds since
        clip start) by repeating the last frame, and release the writer."""
        if self._thread is not None:
            # a full queue must not block shutdown: make room for the sentinel
            while True:
                try:
                    self._q.put(None, timeout=0.2)
                    break
                except queue.Full:
                    if not self._thread.is_alive():
                        break
            self._thread.join(join_timeout_s)
        if (self._writer is not None and self._last_frame is not None
                and stop_rel_s is not None):
            total = int(math.floor(stop_rel_s * self.spec.fps + 0.5))
            pad = total - self._mapper.next_index
            try:
                for _ in range(max(0, pad)):
                    self._writer.write(self._last_frame)
            except Exception as exc:  # pragma: no cover
                self.write_error = f'tail pad: {exc}'
            else:
                if pad > 0:
                    self._mapper.next_index += pad
                    self.duplicated += pad
                    self.tail_padded = pad
        if self._writer is not None:
            try:
                self._writer.release()
            except Exception as exc:  # pragma: no cover
                self.write_error = f'release: {exc}'
        return self.stats()

    def stats(self) -> dict:
        if self.written == 0:
            status = 'no frames'
        else:
            status = 'ok'
        d = {
            'status': status,
            'topic': self.spec.topic,
            'type': self.spec.kind,
            'fps': self.spec.fps,
            'scale': self.spec.scale,
            'size': list(self._size) if self._size else None,
            'file': os.path.basename(self.path) if self.written else None,
            'frames_received': self.received,
            'frames_written': self.written,
            'frames_duplicated': self.duplicated,
            'frames_dropped': self.dropped_queue_full + self.dropped_same_index,
            'dropped_queue_full': self.dropped_queue_full,
            'dropped_same_index': self.dropped_same_index,
            'decode_errors': self.decode_errors,
            'stamp_fallbacks': self.stamp_fallbacks,
            'frames_in_video': self._mapper.next_index,
            'video_duration_s': round(self._mapper.next_index / self.spec.fps, 3),
            'first_frame_offset_s': self.first_frame_offset_s,
            'tail_padded_frames': self.tail_padded,
        }
        if self.write_error:
            d['error'] = self.write_error
        return d


# --------------------------------------------------------------------------
# Output helpers
# --------------------------------------------------------------------------
def default_out_dir() -> str:
    if os.path.ismount(EXTERNAL_DRIVE) and os.access(EXTERNAL_DRIVE, os.W_OK | os.X_OK):
        return EXTERNAL_OUT_DIR
    return FALLBACK_OUT_DIR


def local_iso(ts: Optional[float] = None) -> str:
    d = _dt.datetime.fromtimestamp(ts if ts is not None else time.time())
    return d.astimezone().isoformat(timespec='milliseconds')


def make_clip_dir(out_dir: str, when: Optional[float] = None) -> str:
    name = _dt.datetime.fromtimestamp(
        when if when is not None else time.time()).strftime('%Y%m%d_%H%M%S')
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    n = 1
    while True:
        try:
            os.mkdir(path)
            return path
        except FileExistsError:
            n += 1
            path = os.path.join(out_dir, f'{name}_{n}')


def write_json_atomic(path: str, obj: dict) -> None:
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


# --------------------------------------------------------------------------
# Clip
# --------------------------------------------------------------------------
class Clip:
    def __init__(self, out_dir: str, specs: List[CameraSpec], trigger: str,
                 start_ros_s: float, snapshot: Optional[str],
                 decoder_factory=make_msg_decoder, writer_factory=_cv_writer_factory):
        self.trigger = trigger
        self.start_ros_s = start_ros_s
        self.start_wall = time.time()
        self.start_mono = time.monotonic()
        self.dir = make_clip_dir(out_dir, self.start_wall)
        self.goal_ids: Dict[str, int] = {}
        self.stop_reason: Optional[str] = None
        self.snapshot_copied: Optional[str] = None
        self.snapshot_error: Optional[str] = None
        if snapshot:
            try:
                if os.path.isfile(snapshot):
                    dst = os.path.join(self.dir, os.path.basename(snapshot))
                    shutil.copy2(snapshot, dst)
                    self.snapshot_copied = os.path.basename(dst)
                else:
                    self.snapshot_error = 'snapshot file not found'
            except Exception as exc:
                self.snapshot_error = f'{type(exc).__name__}: {exc}'
        self.recorders: Dict[str, CameraRecorder] = {}
        for s in specs:
            self.recorders[s.name] = CameraRecorder(
                s, os.path.join(self.dir, f'{s.name}.mp4'), start_ros_s,
                decoder_factory(s), writer_factory)
        self._write_meta(None, None, recording=True)

    def note_goals(self, goals: Dict[str, int]) -> None:
        self.goal_ids.update(goals)

    def update_goal_status(self, every: Dict[str, int]) -> None:
        """Track the latest status of goals already seen in this clip."""
        for gid, st in every.items():
            if gid in self.goal_ids:
                self.goal_ids[gid] = st

    def submit(self, name: str, receipt_s: float, payload: bytes) -> None:
        rec = self.recorders.get(name)
        if rec is not None:
            rec.submit(receipt_s, payload)

    def _write_meta(self, stop_ros_s, stop_wall, recording: bool,
                    cams: Optional[dict] = None) -> dict:
        meta = {
            'recording': recording,
            'trigger': self.trigger,
            'start_local': local_iso(self.start_wall),
            'start_ros_s': round(self.start_ros_s, 6),
            'stop_local': local_iso(stop_wall) if stop_wall else None,
            'stop_ros_s': round(stop_ros_s, 6) if stop_ros_s is not None else None,
            'duration_s': (round(stop_wall - self.start_wall, 3)
                           if stop_wall else None),
            'stop_reason': self.stop_reason,
            'goal_ids': sorted(self.goal_ids),
            'goal_last_status': dict(sorted(self.goal_ids.items())),  # 2=executing 4=succeeded
            'snapshot': self.snapshot_copied,
            'snapshot_error': self.snapshot_error,
            'cameras': cams if cams is not None else {
                n: {'status': 'recording', 'topic': r.spec.topic}
                for n, r in self.recorders.items()},
        }
        try:
            write_json_atomic(os.path.join(self.dir, 'clip.json'), meta)
        except Exception as exc:  # pragma: no cover
            print(f'[cut_recorder] WARNING: could not write clip.json: {exc}',
                  flush=True)
        return meta

    def finish(self, stop_ros_s: float, reason: str) -> dict:
        self.stop_reason = reason
        stop_wall = time.time()
        rel = stop_ros_s - self.start_ros_s
        cams = {n: r.close(stop_rel_s=rel) for n, r in self.recorders.items()}
        return self._write_meta(stop_ros_s, stop_wall, recording=False, cams=cams)


# --------------------------------------------------------------------------
# ROS node (composition, lazy rclpy import so the pure logic is importable)
# --------------------------------------------------------------------------
class CutRecorder:
    def __init__(self, args, specs: List[CameraSpec]):
        import rclpy
        from rclpy.qos import (QoSProfile, ReliabilityPolicy, HistoryPolicy,
                               qos_profile_action_status_default)
        from action_msgs.msg import GoalStatusArray
        self._rclpy = rclpy
        self.args = args
        self.specs = specs
        self.node = rclpy.create_node(
            'cut_recorder', cli_args=[], use_global_arguments=False,
            start_parameter_services=False, enable_rosout=False)
        # rclpy creates an idle parameter-event channel; this tool never uses
        # it, so drop it to stay purely subscription-only.
        try:
            self.node.destroy_publisher(self.node._parameter_event_publisher)
            self.node._parameter_event_publisher = None
        except Exception:
            pass
        self.clock = self.node.get_clock()
        self.sm = ClipStateMachine(args.post_roll_s, args.max_clip_s)
        self.clip: Optional[Clip] = None
        self._active_goals: Dict[str, int] = {}
        self._lock = threading.Lock()
        sensor_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_LAST, depth=2)
        self._subs = []
        if args.now:
            self.log('--now: recording immediately; status topic ignored')
        else:
            self._subs.append(self.node.create_subscription(
                GoalStatusArray, args.status_topic, self._on_status,
                qos_profile_action_status_default))
        for s in specs:
            msg_t = self._msg_type(s)
            self._subs.append(self.node.create_subscription(
                msg_t, s.topic, self._make_cam_cb(s.name), sensor_qos, raw=True))
        self.timer = self.node.create_timer(0.1, self._tick)
        self.log('cameras: ' + ', '.join(
            f'{s.name}={s.topic} ({s.kind}, {s.fps:g} fps, scale {s.scale:g})'
            for s in specs))
        if not args.now:
            self.log(f'waiting for trajectory goals on {args.status_topic}')

    @staticmethod
    def _msg_type(spec: CameraSpec):
        if spec.kind == 'compressed':
            from sensor_msgs.msg import CompressedImage
            return CompressedImage
        from sensor_msgs.msg import Image
        return Image

    def now_s(self) -> float:
        return self.clock.now().nanoseconds * 1e-9

    @staticmethod
    def log(msg: str) -> None:
        print(f'[cut_recorder] {msg}', flush=True)

    # -- callbacks ---------------------------------------------------------
    def _on_status(self, msg) -> None:
        active, every = {}, {}
        for st in msg.status_list:
            gid = bytes(st.goal_info.goal_id.uuid).hex()
            every[gid] = int(st.status)
            if st.status in ACTIVE_STATUSES:
                active[gid] = int(st.status)
        self._active_goals = active
        if self.clip is not None:
            self.clip.note_goals(active)
            self.clip.update_goal_status(every)
        self._tick()

    def _make_cam_cb(self, name: str):
        def cb(data: bytes) -> None:
            clip = self.clip
            if clip is not None:
                clip.submit(name, self.now_s(), data)
        return cb

    def _tick(self) -> None:
        if self.args.now:
            return
        ev = self.sm.update(time.monotonic(), bool(self._active_goals))
        if ev is None:
            return
        if ev[0] == 'start':
            self._start_clip('status')
            if self.clip is not None:
                self.clip.note_goals(self._active_goals)
                self.clip._write_meta(None, None, recording=True)
        else:
            self._stop_clip(ev[1])

    # -- clip lifecycle ----------------------------------------------------
    def _start_clip(self, trigger: str) -> None:
        a = self.args
        try:
            self.clip = Clip(a.out_dir, self.specs, trigger, self.now_s(),
                             None if a.no_snapshot else a.snapshot)
        except Exception as exc:
            self.clip = None
            self.sm.force_close()
            self.log(f'ERROR: could not start clip: {type(exc).__name__}: {exc}')
            return
        self.log(f'clip started ({trigger}): {self.clip.dir}')

    def _stop_clip(self, reason: str) -> None:
        clip, self.clip = self.clip, None
        if clip is None:
            return
        try:
            stop_ros = self.now_s()
        except Exception:
            stop_ros = time.time()
        meta = clip.finish(stop_ros, reason)
        cams = ', '.join(f"{n}:{c['frames_written']}f"
                         if c['status'] == 'ok' else f'{n}:no frames'
                         for n, c in meta['cameras'].items())
        self.log(f"clip saved ({reason}): {clip.dir} duration "
                 f"{meta['duration_s']:.1f}s [{cams}]")

    # -- main loop ---------------------------------------------------------
    def run(self, stop_event: threading.Event) -> None:
        from rclpy.executors import SingleThreadedExecutor
        ex = SingleThreadedExecutor()
        ex.add_node(self.node)
        if self.args.now:
            self.sm.force_open(time.monotonic())
            self._start_clip('now')
        try:
            while not stop_event.is_set() and self._rclpy.ok():
                ex.spin_once(timeout_sec=0.1)
        finally:
            self._stop_clip('shutdown')
            ex.shutdown()
            self.node.destroy_node()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Passive recorder: saves camera video while the arm executes '
                    'a trajectory (subscriptions only).')
    p.add_argument('--status-topic', default=DEFAULT_STATUS_TOPIC,
                   help='FollowJointTrajectory action status topic (default: %(default)s)')
    p.add_argument('--post-roll-s', type=float, default=3.0,
                   help='keep recording this long after the last active goal (default 3.0)')
    p.add_argument('--max-clip-s', type=float, default=900.0,
                   help='hard-close a clip after this many seconds (default 900)')
    p.add_argument('--now', action='store_true',
                   help='record immediately until Ctrl-C (ignore the status topic)')
    p.add_argument('--cameras', default='knife,scene,wrist,right_wrist',
                   help='comma-separated camera names to record '
                        '(default: knife,scene,wrist,right_wrist)')
    p.add_argument('--camera', action='append', default=[], metavar='SPEC',
                   help='define/override a camera: name=topic[:raw|compressed[:fps[:scale]]]; repeatable')
    p.add_argument('--scene-scale', type=float, default=None,
                   help='downscale factor of the scene camera (default 0.5)')
    p.add_argument('--out-dir', default=None,
                   help=f'output directory (default: {EXTERNAL_OUT_DIR} if the '
                        f'drive is mounted and writable, else {FALLBACK_OUT_DIR})')
    p.add_argument('--snapshot', default=DEFAULT_SNAPSHOT,
                   help='snapshot JSON copied into each clip folder (default: %(default)s)')
    p.add_argument('--no-snapshot', action='store_true', help='do not copy a snapshot')
    return p


def resolve_cameras(args) -> List[CameraSpec]:
    table = {k: CameraSpec(**vars(v)) for k, v in DEFAULT_CAMERAS.items()}
    for text in args.camera:
        spec = parse_camera_spec(text)
        table[spec.name] = spec
    if args.scene_scale is not None and 'scene' in table:
        if not (0 < args.scene_scale <= 4):
            raise ValueError('--scene-scale out of range')
        # an explicit --camera scene=... scale wins only if it set one
        if not any(t.startswith('scene=') and len(t.split(':')) > 3 for t in args.camera):
            table['scene'].scale = args.scene_scale
    names = [n.strip() for n in args.cameras.split(',') if n.strip()]
    if not names:
        raise ValueError('--cameras is empty')
    unknown = [n for n in names if n not in table]
    if unknown:
        raise ValueError(f'unknown camera(s) {unknown}; known: {sorted(table)}')
    return [table[n] for n in names]


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        specs = resolve_cameras(args)
    except ValueError as exc:
        print(f'cut_recorder: {exc}', file=sys.stderr)
        return 2
    if args.out_dir is None:
        args.out_dir = default_out_dir()
    if args.post_roll_s < 0 or args.max_clip_s <= 0:
        print('cut_recorder: bad --post-roll-s/--max-clip-s', file=sys.stderr)
        return 2
    try:
        os.makedirs(args.out_dir, exist_ok=True)
        if not os.access(args.out_dir, os.W_OK):
            raise PermissionError(args.out_dir)
    except Exception as exc:
        print(f'cut_recorder: output dir not writable: {exc}', file=sys.stderr)
        return 2

    import rclpy
    try:
        from rclpy.signals import SignalHandlerOptions
        rclpy.init(args=[], signal_handler_options=SignalHandlerOptions.NO)
    except ImportError:  # pragma: no cover
        rclpy.init(args=[])
    stop = threading.Event()

    def _handler(signum, _frame):
        stop.set()
    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)

    rec = CutRecorder(args, specs)
    rec.log(f'output dir: {args.out_dir}  (Ctrl-C to stop)')
    try:
        rec.run(stop)
    finally:
        if rclpy.ok():
            rclpy.shutdown()
    rec.log('stopped')
    return 0


if __name__ == '__main__':
    sys.exit(main())
