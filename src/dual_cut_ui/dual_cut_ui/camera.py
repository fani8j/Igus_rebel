"""Lazy camera streaming for the page: CPU matters (the cameras and perception share this PC).

* A camera topic is subscribed only while a client asked for it within the last ``idle_s`` seconds; afterwards the
  subscription is destroyed (the camera driver then does no JPEG encoding for us).
* The ROS callback only stores the newest message bytes and their receive time (``on_frame``); nothing is decoded there.
* On a request the newest frame is decoded with cv2 using JPEG reduced-size decoding where possible, resized to the
  target width, re-encoded as JPEG and cached per (camera, size, frame seq); re-encodes are rate limited per
  (camera, size).
No ROS imports: the subscription is made through the Bridge (``subscribe_camera`` / ``unsubscribe_camera``).
"""
import collections
import threading
import time
from typing import Callable, Dict, Optional, Tuple

import cv2
import numpy as np

DEFAULT_CAMERAS = collections.OrderedDict([
    ("scene", "/camera/color/image_raw/compressed"),
    ("knife", "/knife_camera/image_raw/compressed"),
    ("left_wrist", "/arm_left/wrist_camera/color/image_raw/compressed"),
    ("right_wrist", "/arm_right/wrist_camera/color/image_raw/compressed"),
    ("left_gripper", "/arm_left/gripper_camera/image_raw/compressed"),
    ("right_gripper", "/arm_right/gripper_camera/image_raw/compressed"),
])


DEFAULT_CAMERAS_OFF = "knife"       # shown unchecked in the page's camera menu until the viewer turns it on


def parse_camera_off(spec: str, cameras) -> list:
    """"name,name" -> the names (in camera order) that start switched off in the page.  Unknown names are ignored, so the
    default still works with a custom ``cameras`` list; each browser then remembers its own choice."""
    wanted = {n.strip() for n in (spec or "").split(",") if n.strip()}
    return [n for n in cameras if n in wanted]


def parse_camera_spec(spec: str) -> "collections.OrderedDict[str, str]":
    """"name=topic,name=topic" -> OrderedDict (empty string -> the defaults).  ValueError on malformed items."""
    if not spec or not spec.strip():
        return collections.OrderedDict(DEFAULT_CAMERAS)
    out = collections.OrderedDict()
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        name, sep, topic = item.partition("=")
        name, topic = name.strip(), topic.strip()
        if not sep or not name or not topic.startswith("/") or not name.replace("_", "").isalnum():
            raise ValueError(f"bad camera spec item '{item}' (want name=/topic)")
        out[name] = topic
    if not out:
        raise ValueError("camera spec is empty")
    return out


def jpeg_width(buf: bytes) -> Optional[int]:
    """Width from the JPEG SOF marker without decoding; None if it is not a (parsable) JPEG."""
    n = len(buf)
    if n < 4 or buf[0] != 0xFF or buf[1] != 0xD8:
        return None
    i = 2
    while i + 9 < n:
        if buf[i] != 0xFF:
            i += 1
            continue
        marker = buf[i + 1]
        if marker == 0xFF:
            i += 1
            continue
        if marker in (0x01,) or 0xD0 <= marker <= 0xD9:
            i += 2
            continue
        seg_len = (buf[i + 2] << 8) | buf[i + 3]
        if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            return (buf[i + 7] << 8) | buf[i + 8]
        i += 2 + seg_len
    return None


_REDUCED = {2: cv2.IMREAD_REDUCED_COLOR_2, 4: cv2.IMREAD_REDUCED_COLOR_4, 8: cv2.IMREAD_REDUCED_COLOR_8}


PASSTHROUGH_RATIO = 1.35


def reencode(data: bytes, target_w: int, quality: int = 70, passthrough: float = PASSTHROUGH_RATIO) -> Optional[bytes]:
    """Shrink a compressed image to ``target_w`` pixels wide (never enlarge), return JPEG bytes.
    * A JPEG that is at most ``passthrough`` x ``target_w`` wide is returned UNCHANGED (no decode, no encode: the
      browser scales it); this is what makes the large tile nearly free for a 1280 px camera.
    * Otherwise the reduction factor (1, 2, 4, 8) is the largest one that leaves the decoded width >= target_w.
    Returns None if the data cannot be decoded."""
    buf = np.frombuffer(data, dtype=np.uint8)
    src_w = jpeg_width(data)
    if src_w and passthrough and src_w <= target_w * passthrough:
        return bytes(data)
    factor = 1
    if src_w:
        for f in (8, 4, 2):
            if src_w // f >= target_w:
                factor = f
                break
    flag = _REDUCED.get(factor, cv2.IMREAD_COLOR)
    img = cv2.imdecode(buf, flag)
    if img is None:
        return None
    h, w = img.shape[:2]
    if w > target_w:
        img = cv2.resize(img, (target_w, max(1, int(round(h * target_w / w)))), interpolation=cv2.INTER_AREA)
    ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return enc.tobytes() if ok else None


class CameraHub:
    """Frame store + lazy subscription manager + JPEG cache.  ``bridge`` provides subscribe_camera(name, topic, cb) and
    unsubscribe_camera(name); cb(data: bytes) is called from the ROS thread."""

    SIZES = ("small", "large")

    def __init__(self, bridge, cameras: Dict[str, str], small_width: int = 320, large_width: int = 960,
                 quality: int = 70, idle_s: float = 10.0, min_interval_s: float = 0.125, no_signal_s: float = 3.0,
                 clock: Callable[[], float] = time.monotonic, small_interval_s: float = 0.4,
                 passthrough: float = PASSTHROUGH_RATIO, default_off=()):
        self.bridge, self.cameras = bridge, dict(cameras)
        self.default_off = [n for n in self.cameras if n in set(default_off)]     # the page's starting choice only
        self.widths = {"small": int(small_width), "large": int(large_width)}
        self.quality, self.idle_s, self.no_signal_s, self.passthrough = int(quality), idle_s, no_signal_s, passthrough
        self.min_interval = {"small": small_interval_s, "large": min_interval_s}      # re-encode rate cap per size
        self._clock = clock
        self._lock = threading.Lock()
        self._sub_lock = threading.Lock()          # serialises subscribe / unsubscribe so a sweep cannot undo a new request
        self._subscribed: Dict[str, bool] = {}
        self._last_request: Dict[str, float] = {}
        self._frame: Dict[str, Tuple[bytes, int, float]] = {}          # name -> (bytes, seq, rx)
        self._cache: Dict[Tuple[str, str], Tuple[int, float, bytes]] = {}   # (name, size) -> (seq, encoded_at, jpeg)

    def names(self):
        return list(self.cameras)

    # ---- ROS thread ------------------------------------------------------------------------
    def _on_frame(self, name: str, data: bytes) -> None:
        with self._lock:
            if not self._subscribed.get(name):
                return
            seq = self._frame.get(name, (b"", 0, 0.0))[1] + 1
            self._frame[name] = (data, seq, self._clock())

    # ---- request path (worker thread) ----------------------------------------------------------
    def _ensure_subscribed(self, name: str) -> None:
        with self._lock:
            self._last_request[name] = self._clock()
            if self._subscribed.get(name):
                return
        with self._sub_lock:
            with self._lock:
                if self._subscribed.get(name):
                    return
                self._subscribed[name] = True
            try:
                self.bridge.subscribe_camera(name, self.cameras[name], lambda data, _n=name: self._on_frame(_n, data))
            except Exception:  # noqa: BLE001
                with self._lock:
                    self._subscribed[name] = False
                raise

    def request(self, name: str, size: str) -> Tuple[Optional[bytes], str]:
        """(jpeg or None, reason).  reason: "ok", "unknown camera", "bad size", "no signal"."""
        if name not in self.cameras:
            return None, "unknown camera"
        if size not in self.widths:
            return None, "bad size"
        self._ensure_subscribed(name)
        now = self._clock()
        with self._lock:
            frame = self._frame.get(name)
            cached = self._cache.get((name, size))
        if frame is None or now - frame[2] > self.no_signal_s:
            return None, "no signal"
        data, seq, _rx = frame
        if cached is not None:
            c_seq, c_at, c_jpeg = cached
            if c_seq == seq or now - c_at < self.min_interval[size]:
                return c_jpeg, "ok"
        jpeg = reencode(data, self.widths[size], self.quality, self.passthrough)
        if jpeg is None:
            return (cached[2], "ok") if cached is not None else (None, "undecodable frame")
        with self._lock:
            self._cache[(name, size)] = (seq, now, jpeg)
        return jpeg, "ok"

    def sweep(self) -> None:
        """Destroy the subscriptions nobody asked for during the last ``idle_s`` seconds (periodic)."""
        with self._sub_lock:
            now = self._clock()
            with self._lock:
                idle = [n for n, on in self._subscribed.items() if on and now - self._last_request.get(n, 0.0) > self.idle_s]
                for n in idle:
                    self._subscribed[n] = False
                    self._frame.pop(n, None)
                    for size in self.widths:
                        self._cache.pop((n, size), None)
            for n in idle:
                try:
                    self.bridge.unsubscribe_camera(n)
                except Exception:  # noqa: BLE001
                    pass

    def shutdown(self) -> None:
        with self._lock:
            names = [n for n, on in self._subscribed.items() if on]
            self._subscribed.clear()
        for n in names:
            try:
                self.bridge.unsubscribe_camera(n)
            except Exception:  # noqa: BLE001
                pass

    def status(self) -> Dict[str, dict]:
        now = self._clock()
        with self._lock:
            return {n: {"topic": t, "subscribed": bool(self._subscribed.get(n)),
                        "signal": n in self._frame and now - self._frame[n][2] <= self.no_signal_s}
                    for n, t in self.cameras.items()}


def overlay_jpeg(path: str, width: int, quality: int = 80) -> Optional[bytes]:
    """The detection overlay PNG shrunk to ``width`` pixels as JPEG (None if unreadable)."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    if w > width:
        img = cv2.resize(img, (width, max(1, int(round(h * width / w)))), interpolation=cv2.INTER_AREA)
    ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return enc.tobytes() if ok else None
