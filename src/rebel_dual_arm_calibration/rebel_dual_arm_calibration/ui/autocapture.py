"""Auto capture for the Record arm tab: capture when the arm stands still at a pose not handled yet (no ROS imports).

A pure state machine; the clock, the joint source, the capture call, the jog-busy predicate and the recorder check are
injected, so it is unit-testable without ROS.  It never moves the robot and needs no "Enable motion": it only calls the
same ``capture`` trigger the manual button uses.  ``tick()`` is called about twice a second from one worker thread and
may block for the duration of one capture call; ``start`` / ``stop`` / ``heartbeat`` / ``snapshot`` are called from the
request handlers and never wait for it.

Per pose (all joints of the active arms; a pose is new when it differs by >= ``min_change_deg`` in some joint from EVERY
pose handled in this run, including poses that were skipped or given up):

  stationary (every joint max-min < 2 mrad over 2.0 s, window restarted after a jog step)  ->  capture
    recorded an arm sample                         -> handled, "captured (n this run)"
    "same pose as an existing sample"              -> handled, "same pose, skipped"
    timing refusals (board still moving, ...)      -> retry every 0.5 s, give up after 15 s from the first attempt
    pose problems (no arm sample, not detected)    -> one retry after 0.5 s, then "board not usable here: ..."
    anything else (service down, TF stale, ...)    -> shown, retried every 2 s, the pose is NOT handled

The run belongs to the page that started it (``X-Page-Id``) and stops by itself when that page stops polling
/api/state for 5 s, when the owner stops it, or when the recorder disappears."""
from __future__ import annotations

import math
import re
import threading
from collections import deque
from typing import Callable, Deque, Dict, List, NamedTuple, Optional, Tuple

STATIONARY_WINDOW_S = 2.0          # mirrors the recorder's stationary_window_s
MAX_JOINT_MOTION_RAD = 0.002       # ... and max_joint_motion_rad
RETRY_TIMING_S = 0.5
GIVE_UP_S = 15.0
RETRY_POSE_S = 0.5
RETRY_OTHER_S = 2.0
HEARTBEAT_TIMEOUT_S = 5.0
CHANGE_CHOICES_DEG = (1.0, 2.0, 5.0)
DEFAULT_CHANGE_DEG = 2.0
EVENTS_KEPT = 30

# reply classification (texts of recorder_node._srv_capture / _stillness / stationary_since)
TIMING_RE = re.compile(r"board still moving|not stationary|no earlier image|predates the stationary window|"
                       r"newest image is [\d.]+ s old|cannot verify stillness|not enough joint_states history", re.I)
SAME_POSE_RE = re.compile(r"same pose as an existing sample", re.I)
POSE_RE = re.compile(r"NO ARM SAMPLE RECORDED|not detected|only \d+ corners|PnP", re.I)


class JointReading(NamedTuple):
    q: Dict[str, float]            # latest position per joint name (all joints of the active arms)
    spread: float                  # largest max-min of any joint over the stationary window (rad)
    covered: bool                  # the joint history reaches back over the whole window


def classify_reply(ok: bool, message: str) -> Tuple[str, str]:
    """(kind, reason): kind is captured | same | timing | pose | other."""
    msg = " ".join(str(message).split())
    if ok and not msg.startswith("NO ARM SAMPLE RECORDED"):
        return "captured", msg
    if TIMING_RE.search(msg):
        return "timing", msg
    if SAME_POSE_RE.search(msg):
        return "same", msg
    if POSE_RE.search(msg):
        return "pose", msg
    return "other", msg


def _short(text: str, n: int = 160) -> str:
    return text if len(text) <= n else text[:n - 1] + "…"


def max_delta(a: Dict[str, float], b: Dict[str, float]) -> float:
    keys = set(a) & set(b)
    return max((abs(a[k] - b[k]) for k in keys), default=float("inf"))


class AutoCapture:
    """joints(window_s) -> JointReading | None (None: joint states missing / stale); capture() -> (ok, message), or None if
    a manual action holds the recorder (try again next tick); jog_busy() -> bool; recorder_ok() -> bool."""

    def __init__(self, clock: Callable[[], float], joints: Optional[Callable[[float], Optional[JointReading]]],
                 capture: Callable[[], Optional[Tuple[bool, str]]], jog_busy: Callable[[], bool] = lambda: False,
                 recorder_ok: Callable[[], bool] = lambda: True, unavailable: str = "joint states are not available"):
        self.clock, self._joints, self._capture = clock, joints, capture
        self._jog_busy, self._recorder_ok, self._unavailable = jog_busy, recorder_ok, unavailable
        self.lock = threading.RLock()
        self.on = False
        self.owner: Optional[str] = None
        self.min_change_deg = DEFAULT_CHANGE_DEG
        self.state, self.message = "off", ""
        self.attempts = self.captured = self.skipped = self.gave_up = 0
        self.events: Deque[dict] = deque(maxlen=EVENTS_KEPT)
        self._seq = 0
        self._run = 0                                   # run id: a capture reply of a stopped run is only logged
        self._handled: List[Dict[str, float]] = []
        self._pending: Optional[dict] = None
        self._outcome: Optional[Tuple[str, str]] = None
        self._resume_t: Optional[float] = None          # the stationary window counts from here (after a jog step)
        self._last_hb = 0.0
        self._last_other: Optional[str] = None

    # ---- control (request threads) ------------------------------------------------------------------
    def start(self, page_id: str, min_change_deg: float = DEFAULT_CHANGE_DEG) -> Tuple[bool, str]:
        with self.lock:
            if float(min_change_deg) not in CHANGE_CHOICES_DEG:
                return False, f"min_change_deg must be one of {list(CHANGE_CHOICES_DEG)}"
            if self.on and self.owner != page_id:
                return False, "auto capture is running on another page"
            if self._joints is None:
                return False, self._unavailable
            if not self._recorder_ok():
                return False, "recorder not available"
            self.min_change_deg = float(min_change_deg)
            if self.on:
                return True, "auto capture is already running"
            self.on, self.owner, self._run = True, page_id, self._run + 1
            self.attempts = self.captured = self.skipped = self.gave_up = 0
            self._handled, self._pending, self._outcome = [], None, None
            self._resume_t, self._last_other = float("-inf"), None     # the arm may already have been still for long
            self._last_hb = self.clock()
            self._set("waiting", "waiting for a new pose")
            return True, "auto capture started"

    def stop(self, page_id: Optional[str] = None, reason: str = "auto capture stopped") -> Tuple[bool, str]:
        """The owner stops its run (``page_id`` None: stop regardless, used internally)."""
        with self.lock:
            if not self.on:
                return True, "auto capture is not running"
            if page_id is not None and page_id != self.owner:
                return False, "auto capture is running on another page"
            self._stop_locked(reason)
            return True, reason

    def heartbeat(self, page_id: Optional[str]) -> None:
        with self.lock:
            if self.on and page_id is not None and page_id == self.owner:
                self._last_hb = self.clock()

    def _stop_locked(self, reason: str) -> None:
        self.on, self.owner, self._pending, self._outcome = False, None, None, None
        self._set("off", reason)

    def _set(self, state: str, message: str) -> None:
        self.state, self.message = state, message

    def _event(self, level: str, kind: str, message: str) -> None:
        self._seq += 1
        self.events.append({"seq": self._seq, "level": level, "kind": kind, "message": message})

    def snapshot(self, page_id: Optional[str] = None) -> dict:
        with self.lock:
            return {"on": self.on, "owner_is_you": bool(self.on and page_id is not None and page_id == self.owner),
                    "state": self.state, "message": self.message, "attempts": self.attempts,
                    "captured": self.captured, "skipped": self.skipped, "gave_up": self.gave_up,
                    "min_change_deg": self.min_change_deg, "events": list(self.events)}

    # ---- the periodic step (one worker thread) ---------------------------------------------------------
    def tick(self) -> None:
        with self.lock:
            if not self.on:
                return
            now = self.clock()
            if now - self._last_hb > HEARTBEAT_TIMEOUT_S:
                return self._stop_locked("auto capture stopped: the page stopped polling")
            if not self._recorder_ok():
                return self._stop_locked("auto capture stopped: recorder not available")
            if self._joints is None:                     # start() refuses this; never tick without a joint source
                return self._stop_locked("auto capture stopped: " + self._unavailable)
            if self._jog_busy():
                self._pending, self._outcome, self._resume_t = None, None, None
                return self._set("paused", "paused: a jog step is running")
            reading = self._joints(STATIONARY_WINDOW_S)
            if reading is None:
                self._pending, self._outcome, self._resume_t = None, None, None
                return self._set("no_joints", "waiting for fresh joint states")
            if self._resume_t is None:
                self._resume_t = now
            if reading.spread >= MAX_JOINT_MOTION_RAD:
                self._pending, self._outcome = None, None
                return self._set("moving", "arm moving")
            if not reading.covered or now - self._resume_t < STATIONARY_WINDOW_S:
                self._pending = None
                return self._set("settling", "settling")
            tol = math.radians(self.min_change_deg)
            q = reading.q
            if self._pending is not None and max_delta(q, self._pending["q"]) >= tol:
                self._pending = None
            if self._pending is None:
                if any(max_delta(q, h) < tol for h in self._handled):
                    if self._outcome is not None:
                        return self._set(*self._outcome)
                    return self._set("waiting", "waiting for a new pose")
                self._outcome = None
                self._pending = {"q": dict(q), "first_t": None, "next_t": now, "tries": 0, "pose_fails": 0,
                                 "reason": ""}
            p = self._pending
            if now < p["next_t"]:
                return
            run = self._run
            if p["first_t"] is None:
                p["first_t"] = now
            self._set("capturing", "capturing…")
        # the capture call blocks (up to its own timeout): outside the lock, so start / stop / state stay responsive
        reply = self._capture()
        with self.lock:
            now = self.clock()
            if reply is None:                                   # a manual action holds the recorder: next tick
                if self._run == run and self._pending is p:
                    p["next_t"] = now + RETRY_TIMING_S
                    self._set("retrying", f"waiting for the recorder ({p['tries']} tries)" if p["tries"] else
                              "waiting for the recorder")
                return
            ok, msg = bool(reply[0]), str(reply[1])
            kind, reason = classify_reply(ok, msg)
            if self._run != run or self._pending is not p:      # stopped meanwhile: a stored sample is still logged
                if kind == "captured":
                    self._event("ok", kind, msg)
                return
            self.attempts += 1
            p["tries"] += 1
            p["reason"] = reason
            if kind == "captured":
                self.captured += 1
                return self._handle(p, "captured", f"captured ({self.captured} this run)", "ok", msg)
            if kind == "same":
                self.skipped += 1
                return self._handle(p, "skipped", "same pose, skipped", "warn", msg)
            if kind == "timing":
                if now - p["first_t"] >= GIVE_UP_S:
                    self.gave_up += 1
                    return self._handle(p, "gave_up", "gave up: " + _short(reason), "warn", None)
                p["next_t"] = now + RETRY_TIMING_S
                return self._set("retrying", f"retrying ({p['tries']}): " + _short(reason))
            if kind == "pose":
                p["pose_fails"] += 1
                if p["pose_fails"] >= 2:
                    self.gave_up += 1
                    return self._handle(p, "unusable", "board not usable here: " + _short(reason), "warn", None)
                p["next_t"] = now + RETRY_POSE_S
                return self._set("retrying", f"retrying ({p['tries']}): " + _short(reason))
            p["next_t"] = now + RETRY_OTHER_S
            if msg != self._last_other:                         # a persistent failure is logged once
                self._last_other = msg
                self._event("bad", "other", msg)
            self._set("retrying", f"retrying ({p['tries']}): " + _short(reason))

    def _handle(self, p: dict, state: str, message: str, level: str, reply: Optional[str]) -> None:
        """The pose is done (captured, skipped, gave up, unusable): remember it and log the recorder reply (or, without
        one, the summary) for the result line."""
        self._handled.append(p["q"])
        self._pending, self._last_other = None, None
        self._outcome = (state, message)
        self._set(state, message)
        self._event(level, state, reply if reply is not None else message)
