"""Pure logic of the passive probe: still-period detection, LOOK trigger, record-dir choice.

No ROS and no clocks inside: callers pass stamps, so the tests can feed synthetic sequences.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .model import ArmModel

# Same locations as tools/cut_recorder/cut_recorder.py (kept in sync by hand; both are passive).
EXTERNAL_DRIVE = "/media/farhan/My Passport"
EXTERNAL_OUT_DIR = EXTERNAL_DRIVE + "/igus_nvidia/cut_videos"
FALLBACK_OUT_DIR = "/home/farhan/Igus_rebel/outputs/cut_videos"


# ---------------------------------------------------------------------------- still period
@dataclass
class StillState:
    still: bool
    still_for_s: float        # how long the arm has been continuously still (0 when moving/unknown)
    max_speed: float          # max |dq|/dt over the speed window (rad/s), nan when unknown
    known: bool


class StillTracker:
    """Arm stillness from /joint_states position differences.

    speed = max over the arm's joints of |q(t) - q(t - w)| / w, with w the speed window (default
    0.1 s: single-sample differences of a 100 Hz encoder-quantised stream are too noisy for
    0.01 rad/s). The arm is still while speed < thresh; unknown (too little history, a gap, a joint
    missing or a non-finite value) counts as NOT still and resets the timer.
    """

    def __init__(self, joint_names: Sequence[str], thresh_rad_s: float = 0.01, window_s: float = 0.1,
                 max_gap_s: float = 0.5):
        self.names = list(joint_names)
        self.thresh = thresh_rad_s
        self.window = window_s
        self.max_gap = max_gap_s
        self.hist: Deque[Tuple[float, np.ndarray]] = deque()
        self.since: Optional[float] = None

    def reset(self) -> None:
        self.hist.clear()
        self.since = None

    def update(self, t: float, names: Sequence[str], positions: Sequence[float]) -> StillState:
        mp = dict(zip(names, positions))
        try:
            q = np.array([float(mp[n]) for n in self.names])
        except (KeyError, TypeError, ValueError):
            self.reset()
            return StillState(False, 0.0, float("nan"), False)
        if not np.all(np.isfinite(q)) or not np.isfinite(t):
            self.reset()
            return StillState(False, 0.0, float("nan"), False)
        if self.hist and (t <= self.hist[-1][0] or t - self.hist[-1][0] > self.max_gap):
            self.reset()
        self.hist.append((t, q))
        while len(self.hist) > 2 and self.hist[1][0] <= t - self.window:
            self.hist.popleft()
        t0, q0 = self.hist[0]
        if t - t0 < 0.8 * self.window:
            self.since = None
            return StillState(False, 0.0, float("nan"), False)
        speed = float(np.max(np.abs(q - q0)) / (t - t0))
        if speed < self.thresh:
            if self.since is None:
                self.since = t0
            return StillState(True, t - self.since, speed, True)
        self.since = None
        return StillState(False, 0.0, speed, True)


class LookTrigger:
    """Fires ONCE per still period when the tool tip is at the cut-side edge, outside the carton.

    Conditions: still for >= min_still_s AND tip within near_edge_m of the model cut-side edge line
    (perpendicular distance in the across/normal plane, along within the edge extent +- near_edge_m)
    AND tip across >= 0 (outside the carton). Re-arms when the arm moves again.
    """

    def __init__(self, arm: ArmModel, min_still_s: float = 0.3, near_edge_m: float = 0.040,
                 max_still_s: float = 2.0):
        self.arm = arm
        self.min_still = min_still_s
        self.max_still = max_still_s
        self.near = near_edge_m
        self.armed = True

    def tip_near_edge(self, tip_world) -> bool:
        tip = np.asarray(tip_world, dtype=float).reshape(3)
        if not np.all(np.isfinite(tip)):
            return False
        a, c, h = self.arm.to_local(tip)
        lo, hi = self.arm.edge_extent_a()
        return bool((lo - self.near) <= a <= (hi + self.near) and c >= 0.0
                    and np.hypot(c, h) <= self.near)

    def update(self, still: StillState, tip_world) -> bool:
        if not still.still:
            self.armed = True
            return False
        if still.still_for_s > self.max_still:
            # a dwell lasts ~1 s: an arm that has been standing there for longer (e.g. at rest between two
            # phases, when the coordinator becomes active) is not at a dwell; wait for it to move
            self.armed = False
            return False
        if not self.armed or still.still_for_s < self.min_still:
            return False
        if not self.tip_near_edge(tip_world):
            return False
        self.armed = False
        return True


# ---------------------------------------------------------------------------- record dir
def default_out_dir(ismount=os.path.ismount, access=os.access) -> str:
    if ismount(EXTERNAL_DRIVE) and access(EXTERNAL_DRIVE, os.W_OK | os.X_OK):
        return EXTERNAL_OUT_DIR
    return FALLBACK_OUT_DIR


def open_clip_dir(out_dir: str, now_s: Optional[float] = None, max_age_s: float = 1800.0) -> Optional[str]:
    """Newest clip folder under out_dir whose clip.json has "recording": true.

    cut_recorder writes clip.json with recording true at clip start and false at stop. A crashed
    recorder leaves true forever, so clip.json must also have been modified within max_age_s
    (it is rewritten on goal changes while recording; a clip lasts ~1.5 min) to count as open.
    """
    if not os.path.isdir(out_dir):
        return None
    best: Optional[Tuple[float, str]] = None
    for name in os.listdir(out_dir):
        cj = os.path.join(out_dir, name, "clip.json")
        if not os.path.isfile(cj):
            continue
        try:
            with open(cj) as f:
                rec = json.load(f).get("recording") is True
            mt = os.path.getmtime(cj)
        except (OSError, json.JSONDecodeError, AttributeError):
            continue
        if not rec:
            continue
        if now_s is not None and now_s - mt > max_age_s:
            continue
        if best is None or mt > best[0]:
            best = (mt, os.path.join(out_dir, name))
    return best[1] if best else None


def select_record_dir(out_dir: str, now_s: float) -> str:
    """Directory that holds the `side_cut_vision/` folder for this moment."""
    clip = open_clip_dir(out_dir, now_s)
    if clip:
        return clip
    day = _dt.datetime.fromtimestamp(now_s).strftime("%Y%m%d")
    return os.path.join(out_dir, f"side_cut_vision_{day}")
