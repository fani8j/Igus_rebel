"""Approach assist / hover for the calibration UI: ROS-free state machine (the geometry is in ``hover.py``).

    HoverManager   detect the printed touch marks in the camera image and predict them in the arm base frame with the
                   arm's own calibration fit (``compute``), choose an IK pose 60 mm above a mark (``prepare``), preview /
                   move there through the jog controller (``move``), descend straight along the paper normal to 10 mm
                   (``descend``), and log how far the predicted point was from where the knife tip really touched
                   (``after_touch`` for the Touches tab, ``record_offset`` for the Hover test tab).

Safety: this module never talks to a controller.  The only things that move the robot are the jog controller's
``execute_move`` (the pre-hover joint move, gated by ``allow_real_move`` + the page enable + the confirm dialog, exactly
as "Move there") and its ``descend`` (a straight Cartesian move of 2..70 mm at constant orientation, same gating as a jog
step: page enable on REAL hardware, no ``allow_real_move``).  ``send_rviz`` publishes one ``moveit_msgs/RobotState`` on
RViz's "update custom goal state" topic and nothing else.  Every motion request passes the HTTP ingress gate and carries
the STOP epoch (see server.py).  A mark the camera did not see, a prediction older than ``PRED_MAX_AGE_S`` or a TCP that
is not on the approach line refuses the request with a text.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .jog import DESCEND_MAX_M, DESCEND_MIN_M, REAL_MOVE_DISABLED_MSG
from .suggest import RVIZ_HINT, SampleCounter, SuggestRefused, clean

PREHOVER_HEIGHT_M = 0.06           # stage "prehover": TCP this far above the paper along the normal
DESCEND_HEIGHTS_MM = (10, 20, 30)  # selectable height of the final hover pose above the mark
DEFAULT_DESCEND_MM = 20            # (the operator fine-tunes the last part by teleop / the jog panel)
STAGES = {"prehover": PREHOVER_HEIGHT_M}
YAW_STEP_DEG = 30.0                # hover_tcp_poses: candidate spin of the tool about the normal
MAX_IK_CHANGE_DEG = 120.0          # larger joint change from the current pose: refused ("jog closer first")
IK_VERIFY_POS_M = 0.0005           # FK of the IK solution must reach the requested TCP pose within 0.5 mm / 0.5 deg
IK_VERIFY_ROT_RAD = math.radians(0.5)
DESCEND_MAX_LATERAL_M = 0.005      # the TCP must be within 5 mm of the line through the mark along the normal
DESCEND_MAX_TILT_DEG = 5.0         # and its z axis within 5 deg of -n
DESCEND_MIN_REMAINING_M = DESCEND_MIN_M    # same limits as JogController.descend (which checks them again)
DESCEND_MAX_REMAINING_M = DESCEND_MAX_M
N_FRAMES = 5                       # newest frames handed to the detection
MIN_FRAMES = 3
FRAME_POLL_S = 0.2
FRAME_WAIT_S = 5.0
PRED_MAX_AGE_S = 120.0             # age limit of _pred_for(check_age=True); motions use the fresh observation instead
FRESH_FRAME_MAX_AGE_S = 1.0        # a camera frame received longer ago than this is not used for a detection
OBSERVE_MAX_SHIFT_M = 0.002        # a fresh detection must agree with the stored prediction within 2.0 mm ...
OBSERVE_MAX_TILT_RAD = math.radians(2.0)   # ... and 2 deg of the surface normal, else "Detect points again"
INTERLOCK_POS_M = 0.002            # descend only from within 2 mm / 2 deg of the pre-hover pose that was moved to
INTERLOCK_ROT_RAD = math.radians(2.0)
MIN_CLEARANCE_M = 0.010            # the descend target is never closer than this to the predicted surface
assert min(DESCEND_HEIGHTS_MM) * 1e-3 >= MIN_CLEARANCE_M - 1e-12
RECENT_RECORDS = 40
GO_PREHOVER_FIRST = "go to the pre-hover pose first (Plan hover -> Move)"
MOVED_MSG = "the board moved or the detection changed: Detect points again"

CORRECTION_MAX_MM = 20.0           # a hover prediction correction (per arm, base frame) longer than this is refused at startup
CORRECTION_ARMS = ("left", "right")
ZERO_CORRECTION_MM = (0.0, 0.0, 0.0)

LOG_FILE = "hover_log.jsonl"       # one line per touch of a mark that has a prediction (Touches tab)
TEST_FILE = "hover_test.jsonl"     # one line per "Record offset" (Hover test tab)


# ---- opt-in base-frame correction of the predictions (stopgap, always explicit) ---------------------------
def parse_correction_mm(value) -> Tuple[float, float, float]:
    """A correction vector [x, y, z] in mm from a launch / parameter value: None or "" -> zero; a string like
    "-5.3,8.1,0", "[-5.3, 8.1, 0]" or "-5.3 8.1 0"; or a sequence of 3 numbers.  Raises ValueError if it is not exactly
    three numbers (finiteness and size: ``validate_correction_mm``)."""
    if value is None:
        return ZERO_CORRECTION_MM
    if isinstance(value, str):
        text = value.strip().strip("[]()").strip()
        if not text:
            return ZERO_CORRECTION_MM
        parts = [t for t in re.split(r"[,;\s]+", text) if t]
        try:
            vals = [float(t) for t in parts]
        except ValueError:
            raise ValueError(f"cannot read {value!r} as three numbers x,y,z (mm), e.g. '-5.3,8.1,0'")
    else:
        try:
            seq = list(value)
            if any(isinstance(v, bool) for v in seq):
                raise TypeError("bool")
            vals = [float(v) for v in seq]
        except (TypeError, ValueError):
            raise ValueError(f"cannot read {value!r} as three numbers x,y,z (mm)")
    if len(vals) != 3:
        raise ValueError(f"need exactly three numbers x,y,z (mm), got {len(vals)} in {value!r}")
    return (vals[0], vals[1], vals[2])


def validate_correction_mm(vec, name: str = "hover correction") -> Tuple[float, float, float]:
    """``vec`` (3 numbers, mm) as a tuple of floats; every component finite and the norm <= CORRECTION_MAX_MM,
    else ValueError naming ``name``."""
    try:
        v = parse_correction_mm(vec)
    except ValueError as e:
        raise ValueError(f"{name}: {e}")
    if not all(math.isfinite(c) for c in v):
        raise ValueError(f"{name}: every component must be finite, got {list(v)}")
    norm = math.sqrt(sum(c * c for c in v))
    if norm > CORRECTION_MAX_MM:
        raise ValueError(f"{name}: |{list(v)}| = {norm:.1f} mm exceeds the {CORRECTION_MAX_MM:g} mm limit "
                         "(this is a stopgap for a small constant base-frame shift; refusing to start)")
    return v


def corrections_from_params(left=None, right=None) -> Dict[str, Tuple[float, float, float]]:
    """{'left': vec, 'right': vec} (mm, validated) from the two parameter / launch values; ValueError if either is bad."""
    return {"left": validate_correction_mm(left, "hover_correction_left_mm"),
            "right": validate_correction_mm(right, "hover_correction_right_mm")}


def correction_text(vec) -> str:
    """'-5.3 / +8.1 / +0.0 mm (base frame)' (the same wording the page shows)."""
    return " / ".join(f"{float(c):+.1f}" for c in vec) + " mm (base frame)"


def warn_corrections(corrections: Dict[str, Sequence[float]], mode: str, warn: Callable[[str], None]) -> None:
    """One WARNING per non-zero correction, naming the arm and the vector (``warn(text)``); a correction for an arm the
    mode does not use is reported as ignored."""
    used = mode.split("_")[1] if mode in ("single_left", "single_right") else None
    for arm in CORRECTION_ARMS:
        vec = corrections.get(arm) or ZERO_CORRECTION_MM
        if not any(vec):
            continue
        if arm == used:
            warn(f"hover prediction correction ACTIVE ({arm}): {correction_text(vec)} - added to every approach-assist "
                 "prediction (stopgap; logged as correction_mm in hover_log.jsonl / hover_test.jsonl)")
        else:
            warn(f"hover_correction_{arm}_mm {list(vec)} IGNORED: the approach assist is not active for the {arm} arm in "
                 f"mode {mode}")


class HoverRefused(SuggestRefused):
    """A request that is refused with a clear message (HTTP ``status``)."""


class NotObserved(HoverRefused):
    """The fresh FULL detection did not find the point (e.g. the arm hides part of the board): the partial-occlusion
    'has it moved?' test (hover.unmoved) may still be able to confirm the paper is where it was."""


def offset_text(off: dict) -> str:
    """'predicted vs touched: 2.3 mm (x +1.9, y -1.2, above paper +0.1)' from a compare() result."""
    return (f"predicted vs touched: {off['lateral_mm']:.1f} mm lateral "
            f"(x {off['dx_mm']:+.1f}, y {off['dy_mm']:+.1f}, above paper {off['dn_mm']:+.1f}; 3D {off['dist_mm']:.1f} mm)")


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(v))
    if not math.isfinite(n) or n < 1e-9:
        raise ValueError("zero or non-finite vector")
    return v / n


def rotation_angle(R1: np.ndarray, R2: np.ndarray) -> float:
    M = np.asarray(R1).T @ np.asarray(R2)
    c = 0.5 * (float(np.trace(M)) - 1.0)
    return float(math.acos(min(1.0, max(-1.0, c))))


def _rot_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """A rotation taking unit vector a to unit vector b (only its angle is used)."""
    v = np.cross(a, b)
    c = float(a @ b)
    s_ = float(np.linalg.norm(v))
    if s_ < 1e-12:
        return np.eye(3) if c > 0 else -np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / (s_ * s_))


def sheet_list(sheet_spec) -> list:
    """The hover targets of a scene as a list: None -> [], a HoverTilesSpec (has ``.tiles``) -> its tiles, a list / tuple
    -> as is, one spec -> [spec]."""
    if sheet_spec is None:
        return []
    if hasattr(sheet_spec, "tiles"):
        sheet_spec = sheet_spec.tiles
    return list(sheet_spec) if isinstance(sheet_spec, (list, tuple)) else [sheet_spec]


def status_text(status: Dict[str, str]) -> str:
    """'table: ...; tile A: ...' from the per-source status texts; a text that already starts with 'source:' is not
    prefixed again (no 'tile A: tile A: ...')."""
    parts = []
    for k, v in sorted(status.items()):
        v = str(v)
        parts.append(v if v.lower().startswith(f"{k.lower()}:") else f"{k}: {v}")
    return "; ".join(parts)


def _default_hover():
    from .. import hover
    return hover


def _default_fit_engine():
    from .. import pose_suggest
    return pose_suggest


def _default_load_samples(session_dir: str):
    from ..session import load_samples
    return load_samples(session_dir)


def _default_load_touches(session_dir: str):
    from ..session import load_touches
    return load_touches(session_dir)


def _read_jsonl(path: str) -> List[dict]:
    out: List[dict] = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        continue
    except OSError:
        pass
    return out


class HoverManager:
    """State and actions of the approach assist (single-arm modes only).

    mode, cfgs     recorder mode; jog.arm_configs(mode)
    jog            JogController-like: status(), joint_window / joint_snapshot, plan_move, execute_move, clear_plan,
                   descend(page_id, arm, T_base_target, epoch=), tcp_pose_base(arm, require_standstill) -> (T|None, why),
                   tcp_fk(arm, q) -> (T|None, why), link6_to_tcp(arm) -> (T|None, why)
    session_dir_fn () -> session directory or None
    moveit         MoveItBridge-like: ready(), ik(cfg, T_base_link6, seed_q), valid(cfg, q), publish_goal(names, positions)
    hover          module-like: detect_targets, to_base, hover_tcp_poses, compare, project_points (default: ..hover)
    fit_engine     module-like with fit_arm (default: pose_suggest)
    image_fn       () -> (bgr, seq, rx) or None (RosBridge.latest_frame)
    scene_fn       () -> object with table_spec, sheet_spec (None, one HoverSheetSpec or a list: the hover tiles), detector
                   (None / raises = no targets)
    correction_mm  {'left': [x, y, z], 'right': [...]} in mm, base frame of that arm (default zero): ADDED to every mark's
                   predicted position, at the one place the predictions are produced (``_detect``); validated here
                   (ValueError: non-finite or longer than CORRECTION_MAX_MM).  Opt-in stopgap, logged with every record.
    """

    def __init__(self, mode: str, cfgs: Dict[str, Any], jog, session_dir_fn: Callable[[], Optional[str]], moveit=None,
                 hover=None, fit_engine=None, load_samples: Optional[Callable] = None,
                 load_touches: Optional[Callable] = None, image_fn: Optional[Callable] = None,
                 scene_fn: Optional[Callable] = None, clock: Callable[[], float] = time.monotonic,
                 wall_clock: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep,
                 mono: Callable[[], float] = time.monotonic,
                 n_frames: int = N_FRAMES, frame_poll_s: float = FRAME_POLL_S, frame_wait_s: float = FRAME_WAIT_S,
                 pred_max_age_s: float = PRED_MAX_AGE_S, correction_mm: Optional[Dict[str, Sequence[float]]] = None):
        self.mode, self.cfgs, self.jog = mode, cfgs, jog
        self.correction_mm: Dict[str, Tuple[float, float, float]] = {
            arm: validate_correction_mm((correction_mm or {}).get(arm), f"hover correction ({arm})") for arm in CORRECTION_ARMS}
        self.session_dir_fn, self.moveit = session_dir_fn, moveit
        self._hover, self._fit_engine = hover, fit_engine
        self._load_samples = load_samples or _default_load_samples
        self._load_touches = load_touches or _default_load_touches
        self.image_fn, self.scene_fn = image_fn, scene_fn
        self.clock, self.wall_clock, self.sleep, self.mono = clock, wall_clock, sleep, mono
        self.n_frames, self.frame_poll_s, self.frame_wait_s = int(n_frames), float(frame_poll_s), float(frame_wait_s)
        self.pred_max_age_s = float(pred_max_age_s)
        self.log = logging.getLogger("calibration_ui.hover")
        self.lock = threading.Lock()
        self.counter = SampleCounter()
        self._state, self._message = "idle", ""
        self._arm: Optional[str] = None
        self._preds: Dict[int, dict] = {}          # id -> {id, name, source, detected, quality, pred_base_mm, uv, reason, _p, _p_raw, _n, _x}; _p/pred_base_mm are CORRECTED
        self._catalog: List[Tuple[int, str, str]] = []
        self._status: Dict[str, str] = {}
        self._computed_at: Optional[float] = None
        self._based_on: Optional[int] = None
        self._calib_samples: Optional[int] = None
        self._selected: Optional[int] = None
        self._prepared: Optional[dict] = None
        self._plan: Optional[dict] = None           # {plan_id, point_id, arm} of the plan stored in the jog
        self._records_for: Optional[str] = None     # session dir the in-memory records belong to
        self._touch_records: List[dict] = []
        self._test_records: List[dict] = []
        self._gen = 0
        self._fit = None                            # the fit of the last "Detect points" (X, K, dist for fresh observations)
        self._targets: Dict[int, Any] = {}          # its TargetPoints (optical frame): the stored board / marker poses
        self._pending_move: Optional[dict] = None   # the hover move that was started: {id, plan_id, point_id, T, epoch}
        self._descended: Optional[dict] = None      # the descend that was started: {id, point_id, height_mm}
        self._computed_wall = 0.0                   # wall-clock time of the last successful detection (rows show newer results only)

    # ---- helpers ---------------------------------------------------------------------------------
    @property
    def supported(self) -> bool:
        return self.mode in ("single_left", "single_right")

    def _the_arm(self, arm: Optional[str] = None) -> str:
        if not self.supported:
            raise HoverRefused("approach assist: single-arm modes only", 400)
        only = next(iter(self.cfgs))
        if arm not in (None, only):
            raise HoverRefused(f"arm must be '{only}' in mode {self.mode}", 400)
        return only

    def _correction(self, arm: str) -> Tuple[float, float, float]:
        return self.correction_mm.get(arm, ZERO_CORRECTION_MM)

    def _engine(self):
        if self._hover is None:
            self._hover = _default_hover()
        return self._hover

    def _fits(self):
        if self._fit_engine is None:
            self._fit_engine = _default_fit_engine()
        return self._fit_engine

    def _jog_blocked(self) -> str:
        try:
            st = self.jog.status()
        except Exception as e:  # noqa: BLE001
            return f"jog state unknown ({e})"
        if st.get("available") is False:
            return "jog unavailable: " + str(st.get("error", ""))
        if st.get("busy") or st.get("stopping") or st.get("fault"):
            return "the arm is busy (a jog step / move is running, stopping or in fault)"
        return ""

    def _current_q(self, arm: str) -> Optional[Dict[str, float]]:
        cfg = self.cfgs[arm]
        jw = getattr(self.jog, "joint_window", None)
        if jw is not None:
            r = jw([arm], 0.5)
            if r is not None and all(n in r[0] for n in cfg.joint_names):
                return {n: float(r[0][n]) for n in cfg.joint_names}
            return None
        snap = getattr(self.jog, "joint_snapshot", None) or getattr(getattr(self.jog, "core", None), "joint_snapshot", None)
        if snap is None:
            return None
        v = snap(arm)
        return None if v is None else {n: float(x) for n, x in zip(cfg.joint_names, v)}

    def _clear_jog_plan(self) -> None:
        self._plan = None
        fn = getattr(self.jog, "clear_plan", None)
        if fn is not None:
            try:
                fn()
            except Exception:  # noqa: BLE001
                self.log.exception("jog.clear_plan failed")

    def _stop_epoch(self) -> Optional[int]:
        core = getattr(self.jog, "core", None)
        return getattr(core, "stop_epoch", None)

    def _voided(self, epoch) -> bool:
        """True if STOP was pressed after the request carrying ``epoch`` was ingressed (where the jog exposes it)."""
        now = self._stop_epoch()
        return epoch is not None and now is not None and now != epoch

    def _scene(self):
        if self.scene_fn is None:
            raise HoverRefused("the target definitions are not available in this UI", 503)
        try:
            scene = self.scene_fn()
        except Exception as e:  # noqa: BLE001
            raise HoverRefused(f"cannot load the targets file: {e}", 503)
        if scene is None:
            raise HoverRefused("no table board and no hover_tiles / hover_sheet in the targets file", 503)
        return scene

    @staticmethod
    def _catalog_of(scene) -> List[Tuple[int, str, str]]:
        """[(id, name, source)]: the table crosshairs ("table"), then every hover target - the legacy sheet ("sheet") or
        each tile ("tile A", ...); ``scene.sheet_spec`` may be None, one spec, a HoverTilesSpec or a list of specs."""
        out: List[Tuple[int, str, str]] = []
        table = getattr(scene, "table_spec", None)
        if table is not None:
            out += [(int(i), f"board {int(i)}", "table") for i in sorted(getattr(table, "touch_points", {}) or {})]
        for sheet in sheet_list(getattr(scene, "sheet_spec", None)):
            names = getattr(sheet, "names", {}) or {}
            src = str(getattr(sheet, "source", "sheet"))
            out += [(int(i), str(names.get(i, f"P{i}")), src) for i in sorted(getattr(sheet, "points", {}) or {})]
        return out

    def catalog_points(self) -> List[Tuple[int, str, str]]:
        """[(id, name, source)] of every mark the targets file defines (table crosshairs, hover tile / sheet points); loads the
        scene lazily; [] without one."""
        with self.lock:
            cat = list(self._catalog)
        if cat or self.scene_fn is None or not self.supported:
            return cat
        try:
            scene = self.scene_fn()
            return [] if scene is None else self._catalog_of(scene)
        except Exception:  # noqa: BLE001
            return []

    # ---- records (hover_log.jsonl / hover_test.jsonl) -------------------------------------------------
    def _ensure_records(self, session_dir: Optional[str]) -> None:
        with self.lock:
            if session_dir == self._records_for:
                return
            self._records_for = session_dir
            self._touch_records = _read_jsonl(os.path.join(session_dir, LOG_FILE))[-200:] if session_dir else []
            self._test_records = _read_jsonl(os.path.join(session_dir, TEST_FILE))[-200:] if session_dir else []

    @staticmethod
    def _append(session_dir: str, name: str, rec: dict) -> None:
        os.makedirs(session_dir, exist_ok=True)
        with open(os.path.join(session_dir, name), "a") as f:
            f.write(json.dumps(rec, allow_nan=False) + "\n")

    def _record(self, kind: str, arm: str, pid: int, touched_p: np.ndarray, pred: dict) -> dict:
        off = self._engine().compare({"p": pred["_p"], "n": pred["_n"], "x": pred["_x"], "name": pred["name"],
                                      "source": pred["source"]}, np.asarray(touched_p, dtype=float))
        off = {k: float(off[k]) for k in ("dx_mm", "dy_mm", "dn_mm", "lateral_mm", "dist_mm")}
        now = self.wall_clock()
        rec = clean({
            "time": now, "time_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)), "kind": kind,
            "arm": arm, "point_id": int(pid), "name": pred["name"], "source": pred["source"],
            "predicted_p_base_m": pred["_p"], "predicted_raw_p_base_m": pred["_p_raw"],
            "correction_mm": pred["_correction_mm"], "touched_p_base_m": np.asarray(touched_p, dtype=float),
            "offset_mm": off,
            "calibration": {"n_samples": self._calib_samples, "arm": arm},
            "quality": pred.get("quality") or {},
            "prediction_age_s": None if self._computed_at is None else max(0.0, self.clock() - self._computed_at)})
        return rec

    # ---- compute ---------------------------------------------------------------------------------
    def _collect_frames(self, fit) -> List[tuple]:
        if self.image_fn is None:
            raise HoverRefused("no camera image source in this UI", 503)
        K, dist = np.asarray(fit.K, dtype=float), np.asarray(fit.dist, dtype=float)
        want_wh = tuple(int(v) for v in fit.image_size) if getattr(fit, "image_size", None) is not None else None
        frames: List[tuple] = []
        seen = set()
        polls = max(1, int(math.ceil(self.frame_wait_s / max(self.frame_poll_s, 1e-3))))
        for i in range(polls):
            item = self.image_fn()
            if item is not None:
                bgr, seq, rx = item
                if rx is not None and self.mono() - rx > FRESH_FRAME_MAX_AGE_S:
                    seq = None                       # a stale frame (camera stalled?) never counts
                if seq is not None and seq not in seen:
                    seen.add(seq)
                    h, w = bgr.shape[:2]
                    if want_wh is not None and (w, h) != want_wh:
                        raise HoverRefused(f"the camera image is {w}x{h} but the calibration samples are "
                                           f"{want_wh[0]}x{want_wh[1]}: cannot predict points", 409)
                    frames.append((bgr, K, dist))
            if len(frames) >= self.n_frames:
                break
            if i + 1 < polls:
                self.sleep(self.frame_poll_s)
        if len(frames) < MIN_FRAMES:
            raise HoverRefused(f"only {len(frames)} fresh camera frame(s) arrived in {self.frame_wait_s:g} s "
                               f"(need {MIN_FRAMES}): is the camera running?", 503)
        return frames

    def compute(self, arm: Optional[str] = None) -> str:
        """Detect the marks in the newest frames, fit the arm's session samples, predict every detected mark in the arm
        base frame.  Blocking (about a second of frame collection plus the fit): call from a worker pool.  A refusal
        puts the panel in state "error" (so the text stays visible) unless it is a plain precondition."""
        arm = self._the_arm(arm)
        with self.lock:
            if self._state == "computing":
                raise HoverRefused("already detecting points", 409)
        scene = self._scene()
        with self.lock:
            self._catalog = self._catalog_of(scene)          # the rows exist even if this detection fails
        session_dir = self.session_dir_fn()
        if not session_dir:
            raise HoverRefused("session directory unknown (recorder status not available yet)", 503)
        try:
            samples = [s for s in self._load_samples(session_dir) if getattr(s, "arm", None) == arm]
        except Exception as e:  # noqa: BLE001
            raise HoverRefused(f"cannot read the session samples: {e}", 500)
        try:
            fit = self._fits().fit_arm(samples, arm)
        except ValueError as e:
            with self.lock:
                self._set_error(str(e))
            raise HoverRefused(str(e), 400)
        with self.lock:
            if self._state == "computing":
                raise HoverRefused("already detecting points", 409)
            self._gen += 1
            gen = self._gen
            self._state, self._message, self._arm = "computing", "detecting points ...", arm
        try:
            targets, status, preds = self._detect(scene, fit)
        except HoverRefused as e:
            with self.lock:
                if gen == self._gen:
                    self._set_error(e.message)
            raise
        except Exception as e:  # noqa: BLE001
            self.log.exception("hover detection failed")
            with self.lock:
                if gen == self._gen:
                    self._set_error(f"point detection failed: {type(e).__name__}: {e}")
            raise HoverRefused(f"point detection failed: {type(e).__name__}: {e}", 500)
        catalog = self._catalog_of(scene)
        with self.lock:
            if gen != self._gen:
                return "superseded"
            self._catalog, self._status = catalog, status
            self._preds = preds
            self._fit, self._pending_move, self._descended = fit, None, None     # a new detection voids the hover interlock
            self._targets = dict(targets) if hasattr(targets, "items") else {}
            self._computed_wall = self.wall_clock()
            self._computed_at, self._arm = self.clock(), arm
            self._based_on = self._calib_samples = len(samples)
            self._clear_jog_plan()
            self._prepared = None
            if preds:
                self._state = "ready"
                names = ", ".join(f"{p['name']}" for _, p in sorted(preds.items()))
                missing = [c for c in catalog if c[0] not in preds]
                self._message = f"{len(preds)} point(s) detected ({names})" + (
                    f"; not detected: {', '.join(str(c[0]) for c in missing)}" if missing else "")
                if status:
                    self._message += " [" + status_text(status) + "]"
                return self._message
            self._set_error("no point detected: " + (status_text(status)
                                                     or "neither the table board nor a hover tile is visible"), keep_status=True)
            msg = self._message
        raise HoverRefused(msg, 409)

    def _detect(self, scene, fit) -> Tuple[Any, Dict[str, str], Dict[int, dict]]:
        """One detection on fresh frames: (targets, status texts, {id: prediction})."""
        frames = self._collect_frames(fit)
        hv = self._engine()
        targets, status = hv.detect_targets(frames, scene.table_spec, scene.sheet_spec, scene.detector)
        based = hv.to_base(targets, fit.X)
        try:
            uv = hv.project_points(targets, fit.K, fit.dist)
        except Exception:  # noqa: BLE001 - only the overlay is lost
            self.log.exception("project_points failed")
            uv = {}
        status = {str(k): str(v) for k, v in dict(status or {}).items()}
        preds: Dict[int, dict] = {}
        t_now = self.clock()
        corr_mm = self._correction(self._the_arm())
        corr_m = np.asarray(corr_mm, dtype=float) * 1e-3 if any(corr_mm) else None
        for pid, d in dict(based).items():
            pid = int(pid)
            try:
                p, n, x = (np.asarray(d[k], dtype=float).reshape(3) for k in ("p", "n", "x"))
                if not (np.all(np.isfinite(p)) and np.all(np.isfinite(n)) and np.all(np.isfinite(x))):
                    raise ValueError("non-finite prediction")
                n = _unit(n)
            except (KeyError, ValueError, TypeError) as e:
                self.log.warning("dropping a malformed prediction for point %s: %s", pid, e)
                continue
            p_raw = p
            if corr_m is not None:       # THE place the opt-in base-frame correction is applied (stored and fresh alike)
                p = p_raw + corr_m
            tp = targets.get(pid) if hasattr(targets, "get") else None
            uvp = uv.get(pid) if hasattr(uv, "get") else None
            preds[pid] = {"id": pid, "name": str(d.get("name", f"P{pid}")), "source": str(d.get("source", "")),
                          "detected": True, "reason": "",
                          "quality": clean(getattr(tp, "quality", None) or {}),
                          "pred_base_mm": [float(v) * 1e3 for v in p],
                          "pred_raw_base_mm": [float(v) * 1e3 for v in p_raw],
                          "uv": None if uvp is None else [float(uvp[0]), float(uvp[1])],
                          "_p": p, "_p_raw": p_raw, "_correction_mm": list(corr_mm), "_n": n, "_x": x, "_t": t_now}
        return targets, status, preds

    def _observe(self, point_id: int, old: dict) -> dict:
        """Fresh observation right before a motion request: detect again on fresh frames; the point must be seen and agree
        with the stored prediction within OBSERVE_MAX_SHIFT_M / OBSERVE_MAX_TILT_RAD.  Returns the FRESH prediction (it
        replaces the stored one and is what the motion is aimed at)."""
        scene = self._scene()
        with self.lock:
            fit = self._fit
        if fit is None:
            raise HoverRefused("no detection to refine: press 'Detect points'", 409)
        try:
            _targets, status, preds = self._detect(scene, fit)
        except HoverRefused:
            raise
        except Exception as e:  # noqa: BLE001
            self.log.exception("fresh hover detection failed")
            raise HoverRefused(f"fresh point detection failed: {type(e).__name__}: {e}", 500)
        new = preds.get(point_id)
        if new is None:
            why = status_text(status)
            raise NotObserved(f"point {point_id} is not detected in the fresh camera frames"
                              + (f" ({why})" if why else "") + f": {MOVED_MSG}", 409)
        shift = float(np.linalg.norm(new["_p"] - old["_p"]))
        tilt = rotation_angle(np.eye(3), _rot_between(old["_n"], new["_n"]))
        if not (shift <= OBSERVE_MAX_SHIFT_M and tilt <= OBSERVE_MAX_TILT_RAD):
            raise HoverRefused(f"{MOVED_MSG} (point {point_id} moved {shift * 1e3:.1f} mm, normal {math.degrees(tilt):.1f} deg; "
                               f"limit {OBSERVE_MAX_SHIFT_M * 1e3:.1f} mm / {math.degrees(OBSERVE_MAX_TILT_RAD):.0f} deg)", 409)
        with self.lock:
            if self._state == "ready":
                self._preds[point_id] = new
        return new

    def _confirm_unmoved(self, point_id: int, pred: dict) -> dict:
        """The "paper has not moved" proof that works with the arm in the way (partial occlusion): the VISIBLE corners of fresh
        frames are projected with the STORED board / marker pose (hover.unmoved).  Passes -> the STORED prediction is returned
        and is the target; fails -> the board moved / is hidden by the arm (refused with the reason)."""
        scene = self._scene()
        with self.lock:
            fit, targets = self._fit, dict(self._targets)
        if fit is None or not targets:
            raise HoverRefused("no detection to compare with: press 'Detect points'", 409)
        stored = {k: v for k, v in targets.items() if getattr(v, "source", None) == pred["source"]}
        try:
            frames = self._collect_frames(fit)
            ok, why, stats = self._engine().unmoved(frames, stored, scene.table_spec, scene.sheet_spec, scene.detector,
                                                    sources=[pred["source"]])
        except HoverRefused:
            raise
        except Exception as e:  # noqa: BLE001
            self.log.exception("unmoved check failed")
            raise HoverRefused(f"cannot confirm the board has not moved: {type(e).__name__}: {e}", 500)
        if not ok:
            raise HoverRefused(why or MOVED_MSG, 409)
        self.log.info("hover: point %s confirmed unmoved (%s)", point_id, clean(stats))
        return pred

    def _set_error(self, message: str, keep_status: bool = False) -> None:
        """A failed detection: every row's reason becomes the failure text (the per-source status texts of the 'no point
        detected' case excepted), old predictions, the prepared pose and the hover interlock are gone."""
        self._gen += 1
        self._state, self._message = "error", message
        self._preds, self._prepared, self._pending_move, self._descended = {}, None, None, None
        self._targets = {}
        if not keep_status:
            self._status = {}
        self._computed_at = self._based_on = self._calib_samples = None
        self._clear_jog_plan()

    def _pred_for(self, point_id, check_age: bool = True) -> dict:
        """The stored prediction of a point.  ``check_age``: refuse one older than PRED_MAX_AGE_S (every MOTION; the pure
        comparisons - Record offset - need no age limit: they move nothing)."""
        if isinstance(point_id, bool) or not isinstance(point_id, int):
            raise HoverRefused("point_id must be an integer", 400)
        with self.lock:
            st, p = self._state, self._preds.get(point_id)
            status = dict(self._status)
            known = {c[0]: c for c in self._catalog}
        if st != "ready":
            raise HoverRefused("no detected points yet: press 'Detect points'", 409)
        if p is None:
            src = known[point_id][2] if point_id in known else None
            why = status.get(src, "") if src else ""
            raise HoverRefused(f"point {point_id} was not detected" + (f" ({why})" if why else "")
                               + ": press 'Detect points' again with the mark in view", 409)
        age = self.clock() - p.get("_t", float("-inf")) if check_age else 0.0
        if not age <= self.pred_max_age_s:
            raise HoverRefused(f"the prediction is {age:.0f} s old (limit {self.pred_max_age_s:.0f} s): the paper may "
                               "have moved, press 'Detect points' again", 409)
        return p

    # ---- prepare: IK + validity + plan ---------------------------------------------------------------
    def prepare(self, page_id: str, point_id, stage: str = "prehover", epoch=None) -> dict:
        arm = self._the_arm()
        if stage not in STAGES:
            raise HoverRefused(f"stage must be one of {sorted(STAGES)}", 400)
        # no age limit here: _observe() below re-detects on FRESH frames and refuses unless the mark is within
        # OBSERVE_MAX_SHIFT_M of this prediction, which is the actual "the paper has not moved" proof (a 60 s move
        # to the pre-hover pose would otherwise expire every prediction before the descend)
        pred = self._pred_for(point_id, check_age=False)
        why = self._jog_blocked()
        if why:
            raise HoverRefused("cannot prepare now: " + why, 409)
        if self.moveit is None:
            raise HoverRefused("MoveIt services are not available in this UI (no ROS bridge)", 503)
        ok, msg = self.moveit.ready()
        if not ok:
            raise HoverRefused(msg, 503)
        try:                                         # fresh frames: the plan is aimed at what the camera sees NOW ...
            pred = self._observe(point_id, pred)
        except NotObserved:                          # ... or, if the arm hides the board now, at the stored pose that is
            pred = self._confirm_unmoved(point_id, pred)     # proven unmoved by the visible corners
        with self.lock:
            self._pending_move = self._descended = None      # a new plan voids the hover interlock
        cfg = self.cfgs[arm]
        q0 = self._current_q(arm)
        if q0 is None:
            raise HoverRefused("current joint states of the arm are not available", 503)
        T_l6_tcp, why = self.jog.link6_to_tcp(arm)
        if T_l6_tcp is None:
            raise HoverRefused("cannot convert the TCP pose to the flange: " + why, 503)
        T_tcp_l6 = np.linalg.inv(np.asarray(T_l6_tcp, dtype=float))
        height = STAGES[stage]
        cands = list(self._engine().hover_tcp_poses(pred["_p"], pred["_n"], pred["_x"], height_m=height,
                                                    yaw_step_deg=YAW_STEP_DEG))
        if not cands:
            raise HoverRefused("no candidate pose above this point", 409)
        found, counts = [], {"ik_failed": 0, "ik_mismatch": 0}
        for i, T in enumerate(cands):
            if self._voided(epoch):
                raise HoverRefused("STOP was pressed after this request was sent: request void", 409)
            T = np.asarray(T, dtype=float).reshape(4, 4)
            try:
                q = self.moveit.ik(cfg, T @ T_tcp_l6, q0)
            except Exception as e:  # noqa: BLE001
                raise HoverRefused(f"IK service call failed: {type(e).__name__}: {e}", 503)
            if not q or not all(n in q and math.isfinite(float(q[n])) for n in cfg.joint_names):
                counts["ik_failed"] += 1
                continue
            q = {n: float(q[n]) for n in cfg.joint_names}
            # the IK is asked for the flange: check with the model's own FK that the TCP really gets there
            Tfk, why = self.jog.tcp_fk(arm, q)
            if Tfk is None:
                raise HoverRefused("cannot verify the IK solution: " + why, 503)
            if (np.linalg.norm(Tfk[:3, 3] - T[:3, 3]) > IK_VERIFY_POS_M
                    or rotation_angle(Tfk[:3, :3], T[:3, :3]) > IK_VERIFY_ROT_RAD):
                counts["ik_mismatch"] += 1
                continue
            change = max(abs(q[n] - q0[n]) for n in cfg.joint_names)
            found.append((change, i, q, T))
        if not found:
            raise HoverRefused(f"no reachable pre-hover pose above point {point_id} ({len(cands)} orientations tried: "
                               f"IK failed {counts['ik_failed']}, wrong TCP pose {counts['ik_mismatch']})", 409)
        found.sort(key=lambda t: (t[0], t[1]))
        best_deg = math.degrees(found[0][0])
        if best_deg > MAX_IK_CHANGE_DEG:
            raise HoverRefused(f"the nearest reachable pre-hover pose needs {best_deg:.0f} deg of joint motion (limit "
                               f"{MAX_IK_CHANGE_DEG:.0f}): jog the arm closer to the mark first", 409)
        chosen, invalid = None, 0
        for change, i, q, T in found:
            if math.degrees(change) > MAX_IK_CHANGE_DEG:
                break
            if self._voided(epoch):
                raise HoverRefused("STOP was pressed after this request was sent: request void", 409)
            ok, why = self.moveit.valid(cfg, q)
            if ok:
                chosen = (change, i, q, T)
                break
            invalid += 1
        if chosen is None:
            raise HoverRefused(f"all {invalid} reachable pre-hover poses above point {point_id} are in collision or "
                               "invalid", 409)
        change, i, q, T = chosen
        prepared = {"point_id": int(point_id), "stage": stage, "arm": arm, "q": q, "candidate": i,
                    "n_candidates": len(cands), "max_joint_change_deg": math.degrees(change),
                    "height_mm": height * 1e3, "T_base_tcp": T.tolist()}
        with self.lock:
            self._clear_jog_plan()
            self._prepared, self._selected = prepared, int(point_id)
        head = (f"IK pose {i + 1} of {len(cands)} above {pred['name']} ({point_id}), {height * 1e3:.0f} mm: max joint "
                f"change {math.degrees(change):.1f} deg")
        fn = getattr(self.jog, "plan_move", None)
        if fn is None:
            return {"ok": False, "message": head + "; the jog module has no plan_move", "prepared": True, "plan_id": None}
        res = fn(page_id, arm, dict(q), **({} if epoch is None else {"epoch": epoch}))
        res = dict(res) if isinstance(res, dict) else {"ok": False, "message": "bad answer from the jog module"}
        with self.lock:
            if res.get("ok") and res.get("plan_id") is not None:
                self._plan = {"plan_id": res["plan_id"], "point_id": int(point_id), "arm": arm}
            else:
                self._plan = None
        res["prepared"] = True
        res["point_id"] = int(point_id)
        if res.get("ok"):
            res["message"] = head + "; " + str(res.get("message") or "planned")
        else:
            real_off = res.get("message") == REAL_MOVE_DISABLED_MSG
            res["ok"] = real_off          # the goal itself is fine; only the preview / Move is switched off
            res["message"] = head + ("; " if real_off else "; planning failed: ") + str(res.get("message") or "")
            if real_off:
                res["message"] += " - the goal is ready for 'Send to RViz'"
        return res

    def send_rviz(self, point_id) -> str:
        """Publish the prepared joint goal as a RobotState on RViz's goal topic.  Nothing else is published."""
        self._the_arm()
        with self.lock:
            prep = None if self._prepared is None else dict(self._prepared)
        if prep is None or prep["point_id"] != point_id:
            raise HoverRefused("no prepared pose for this point: press 'Plan hover' first", 409)
        if self.moveit is None:
            raise HoverRefused("no ROS bridge: cannot publish to RViz", 503)
        cfg = self.cfgs[prep["arm"]]
        names = list(cfg.joint_names)
        self.moveit.publish_goal(names, [prep["q"][n] for n in names])
        with self.lock:
            self._selected = int(point_id)
        return RVIZ_HINT

    def move(self, page_id: str, plan_id, epoch=None) -> dict:
        """Execute exactly the plan prepared here (jog.execute_move: allow_real_move + enable + confirm rules apply)."""
        self._the_arm()
        with self.lock:
            mine = None if self._plan is None else dict(self._plan)
            prep = None if self._prepared is None else dict(self._prepared)
        if mine is None or mine["plan_id"] != plan_id:
            raise HoverRefused("no stored plan with this id: press 'Plan hover' first", 409)
        if prep is None or prep["point_id"] != mine["point_id"]:
            raise HoverRefused("the stored plan is for another point: press 'Plan hover' again", 409)
        fn = getattr(self.jog, "execute_move", None)
        if fn is None:
            raise HoverRefused("jog module has no execute_move", 503)
        res = fn(page_id, plan_id, **({} if epoch is None else {"epoch": epoch}))
        res = dict(res) if isinstance(res, dict) else {"ok": False, "message": "bad answer from the jog module"}
        if res.get("ok"):
            with self.lock:
                self._plan = None                 # the jog consumes the plan
                # the descend interlock: this very move (its jog id), its target TCP pose and the STOP epoch it started under
                self._pending_move = {"id": res.get("id"), "plan_id": plan_id, "point_id": mine["point_id"],
                                      "T": np.asarray(prep["T_base_tcp"], dtype=float).reshape(4, 4),
                                      "epoch": self._stop_epoch()}
        return res

    def _interlock(self, point_id=None, clear: bool = True) -> str:
        """"" if the LAST motion of the arm was the successfully finished hover move to the pre-hover pose of ``point_id``
        (None = any point); else why not.  Anything else invalidates it (and clears it when ``clear``): another motion
        (the jog's last id differs), a stopped / failed / refused / faulted move, a STOP since the move started, a new
        detection or plan.  (The TCP-within-2-mm/2-deg test is done by the caller with a measured pose.)"""
        with self.lock:
            pm = None if self._pending_move is None else dict(self._pending_move)
        if pm is None:
            return GO_PREHOVER_FIRST
        why = ""
        try:
            st = self.jog.status()
        except Exception as e:  # noqa: BLE001
            return f"jog state unknown ({e})"
        last = st.get("last") if isinstance(st.get("last"), dict) else {}
        now_epoch = self._stop_epoch()
        if point_id is not None and pm["point_id"] != point_id:
            return f"the arm was moved above point {pm['point_id']}, not {point_id}: {GO_PREHOVER_FIRST}"
        if pm["id"] is None or last.get("id") != pm["id"]:
            why = "another motion happened since the hover move"
        elif st.get("fault"):
            why = "the arm is in fault"
        elif pm["epoch"] is not None and now_epoch is not None and now_epoch != pm["epoch"]:
            why = "STOP was pressed since the hover move"
        elif last.get("phase") in ("planning", "planned", "checking", "executing", "stopping"):
            return "the hover move is still running: wait until it is done"
        elif last.get("phase") != "done" or last.get("ok") is False:
            why = f"the hover move ended as '{last.get('phase')}'"
        if why:
            if clear:
                with self.lock:
                    if self._pending_move is not None and self._pending_move["id"] == pm["id"]:
                        self._pending_move = None
            return f"{why}: {GO_PREHOVER_FIRST}"
        return ""

    def _real_move_blocked(self, arm: str) -> bool:
        """REAL hardware without allow_real_move (the jog enforces it too; this refuses before any camera / TF work)."""
        try:
            st = self.jog.status()
        except Exception:  # noqa: BLE001
            return False
        hw = (st.get("hardware") or {}).get(arm) or {}
        return hw.get("real") is True and st.get("allow_real_move") is not True

    # ---- descend ---------------------------------------------------------------------------------
    def descend(self, page_id: str, point_id, height_mm=DEFAULT_DESCEND_MM, epoch=None) -> dict:
        """Straight line along -n from the current TCP to ``height_mm`` (10 / 20 / 30) above the predicted point."""
        arm = self._the_arm()
        if isinstance(height_mm, bool) or not isinstance(height_mm, (int, float)) or float(height_mm) not in DESCEND_HEIGHTS_MM:
            raise HoverRefused(f"height_mm must be one of {list(DESCEND_HEIGHTS_MM)}", 400)
        height_m = float(height_mm) * 1e-3
        if height_m < MIN_CLEARANCE_M - 1e-12:
            raise HoverRefused(f"the final hover height must be at least {MIN_CLEARANCE_M * 1e3:g} mm", 400)
        pred = self._pred_for(point_id, check_age=False)     # freshness is proven by _observe() below (see prepare)
        why = self._jog_blocked()
        if why:
            raise HoverRefused("cannot descend now: " + why, 409)
        if self._real_move_blocked(arm):
            raise HoverRefused(REAL_MOVE_DISABLED_MSG, 409)
        fn = getattr(self.jog, "descend", None)
        if fn is None:
            raise HoverRefused("jog module has no descend", 503)
        why = self._interlock(point_id)
        if why:
            raise HoverRefused("cannot descend: " + why, 409)
        # The arm is at the pre-hover pose and hides part of the board (or all of the sheet): no full re-detection is demanded.
        # The visible corners must still sit where the STORED pose puts them; the target is the STORED prediction.
        pred = self._confirm_unmoved(point_id, pred)
        if self._voided(epoch):
            raise HoverRefused("STOP was pressed after this request was sent: request void", 409)
        T_now, why = self.jog.tcp_pose_base(arm, require_standstill=True)
        if T_now is None:
            raise HoverRefused("cannot descend: " + why, 409)
        T_now = np.asarray(T_now, dtype=float)
        with self.lock:
            T_pre = None if self._pending_move is None else self._pending_move["T"].copy()
        if T_pre is None:
            raise HoverRefused("cannot descend: " + GO_PREHOVER_FIRST, 409)
        dpos = float(np.linalg.norm(T_now[:3, 3] - T_pre[:3, 3]))
        drot = rotation_angle(T_now[:3, :3], T_pre[:3, :3])
        if not (dpos <= INTERLOCK_POS_M and drot <= INTERLOCK_ROT_RAD):
            raise HoverRefused(f"the TCP is {dpos * 1e3:.1f} mm / {math.degrees(drot):.1f} deg from the pre-hover pose that "
                               f"was moved to (limit {INTERLOCK_POS_M * 1e3:g} mm / {math.degrees(INTERLOCK_ROT_RAD):g} deg): "
                               f"{GO_PREHOVER_FIRST}", 409)
        c, zc = T_now[:3, 3], T_now[:3, 2]
        p, n = pred["_p"], pred["_n"]
        rel = c - p
        h = float(rel @ n)                                   # height above the mark along the normal
        lateral = float(np.linalg.norm(rel - h * n))
        tilt = math.degrees(math.acos(min(1.0, max(-1.0, float(zc @ (-n))))))
        remaining = h - height_m
        name = f"{pred['name']} ({point_id})"
        if tilt > DESCEND_MAX_TILT_DEG:
            raise HoverRefused(f"the tool axis is {tilt:.1f} deg off the paper normal at {name} (limit "
                               f"{DESCEND_MAX_TILT_DEG:g}): {GO_PREHOVER_FIRST}", 409)
        if lateral > DESCEND_MAX_LATERAL_M:
            raise HoverRefused(f"the TCP is {lateral * 1e3:.1f} mm off the line through {name} along the paper normal "
                               f"(limit {DESCEND_MAX_LATERAL_M * 1e3:g} mm): {GO_PREHOVER_FIRST}", 409)
        if remaining < DESCEND_MIN_REMAINING_M:
            raise HoverRefused(f"the TCP is only {h * 1e3:.1f} mm above {name}: nothing to descend to "
                               f"{height_m * 1e3:g} mm (remaining {remaining * 1e3:.1f} mm, minimum "
                               f"{DESCEND_MIN_REMAINING_M * 1e3:g} mm)", 409)
        if remaining > DESCEND_MAX_REMAINING_M:
            raise HoverRefused(f"the TCP is {h * 1e3:.1f} mm above {name}: more than {DESCEND_MAX_REMAINING_M * 1e3:g} mm "
                               f"to descend: {GO_PREHOVER_FIRST}", 409)
        T_target = T_now.copy()
        T_target[:3, 3] = c - remaining * n                  # straight along -n, orientation unchanged
        clearance = float((T_target[:3, 3] - p) @ n)         # the target's height above the predicted surface
        if not clearance >= MIN_CLEARANCE_M - 1e-9:
            raise HoverRefused(f"descend target would be only {clearance * 1e3:.1f} mm above the predicted surface "
                               f"(minimum {MIN_CLEARANCE_M * 1e3:g} mm)", 409)
        with self.lock:
            self._selected = int(point_id)
        res = fn(page_id, arm, T_target, **({} if epoch is None else {"epoch": epoch}))
        res = dict(res) if isinstance(res, dict) else {"ok": False, "message": "bad answer from the jog module"}
        if res.get("ok"):
            res["message"] = (f"descending {remaining * 1e3:.1f} mm along the paper normal to {height_m * 1e3:g} mm "
                              f"above {name}")
            with self.lock:       # the prepared pose is history now; the interlock goes with the next jog id (any later motion)
                self._prepared, self._plan = None, None
                self._descended = {"id": res.get("id"), "point_id": int(point_id), "height_mm": height_m * 1e3}
        res["point_id"] = int(point_id)
        return res

    # ---- offsets ---------------------------------------------------------------------------------
    def touch_count(self) -> int:
        """Number of touches in the session now (call before the capture_touch trigger, pass to after_touch)."""
        sd = self.session_dir_fn()
        if not sd:
            return 0
        try:
            return len(self._load_touches(sd))
        except Exception:  # noqa: BLE001
            return 0

    def after_touch(self, arm: str, point_id: int, n_before: int) -> Optional[dict]:
        """After a successful capture_touch: compare the recorded TCP with the prediction, log it, return the offset
        ({dx_mm, dy_mm, dn_mm, lateral_mm, dist_mm, text}), or None (no prediction for the point / no touch found)."""
        if not self.supported or arm not in self.cfgs:
            return None
        try:
            sd = self.session_dir_fn()
            with self.lock:
                pred = self._preds.get(point_id) if self._state == "ready" else None
            if not sd or pred is None:
                return None
            touches = list(self._load_touches(sd))[max(0, int(n_before)):]
            mine = [t for t in touches if getattr(t, "arm", None) == arm and int(getattr(t, "point_id", -1)) == int(point_id)]
            if not mine:
                return None
            T = np.asarray(mine[-1].T_base_tcp, dtype=float).reshape(4, 4)
            rec = self._record("touch", arm, point_id, T[:3, 3], pred)
            self._ensure_records(sd)
            self._append(sd, LOG_FILE, rec)
            with self.lock:
                self._touch_records.append(rec)
            return dict(rec["offset_mm"], text=offset_text(rec["offset_mm"]))
        except Exception:  # noqa: BLE001 - the touch itself is recorded; the comparison is a bonus
            self.log.exception("hover comparison after the touch failed")
            return None

    def record_offset(self, point_id) -> dict:
        """Hover test: the operator jogged the tip onto the mark; compare the CURRENT TCP with the prediction."""
        arm = self._the_arm()
        pred = self._pred_for(point_id, check_age=False)     # a comparison: moves nothing, no age limit
        why = self._jog_blocked()
        if why:
            raise HoverRefused("cannot record now: " + why, 409)
        T, why = self.jog.tcp_pose_base(arm, require_standstill=True)
        if T is None:
            raise HoverRefused("cannot record: " + why, 409)
        sd = self.session_dir_fn()
        if not sd:
            raise HoverRefused("session directory unknown (recorder status not available yet)", 503)
        rec = self._record("hover_test", arm, point_id, np.asarray(T, dtype=float)[:3, 3], pred)
        self._ensure_records(sd)
        try:
            self._append(sd, TEST_FILE, rec)
        except OSError as e:
            raise HoverRefused(f"cannot write {TEST_FILE}: {e}", 500)
        with self.lock:
            self._test_records.append(rec)
            self._selected = int(point_id)
        off = dict(rec["offset_mm"], text=offset_text(rec["offset_mm"]))
        return {"ok": True, "point_id": int(point_id), "offset": off,
                "message": f"{pred['name']} ({point_id}): " + off["text"] + f" - logged in {TEST_FILE}"}

    # ---- views -----------------------------------------------------------------------------------
    def _last_offsets(self) -> Dict[int, dict]:
        out: Dict[int, dict] = {}
        for rec in sorted(self._touch_records + self._test_records, key=lambda r: r.get("time") or 0):
            if (rec.get("time") or 0) < self._computed_wall:        # only results of the current detection (older: in the table)
                continue
            try:
                out[int(rec["point_id"])] = dict(rec["offset_mm"], kind=rec.get("kind"), time=rec.get("time"),
                                                 text=offset_text(rec["offset_mm"]))
            except (KeyError, TypeError, ValueError):
                continue
        return out

    def _test_summary(self) -> dict:
        per: Dict[int, dict] = {}
        for rec in self._test_records:
            try:
                pid, off = int(rec["point_id"]), rec["offset_mm"]
                lat = float(off["lateral_mm"])
            except (KeyError, TypeError, ValueError):
                continue
            d = per.setdefault(pid, {"point_id": pid, "name": rec.get("name"), "n": 0, "lats": [], "last": None})
            d["n"] += 1
            d["lats"].append(lat)
            d["last"] = dict(off, time=rec.get("time"))
        rows = []
        for pid in sorted(per):
            d = per[pid]
            rows.append({"point_id": pid, "name": d["name"], "n": d["n"], "last": d["last"],
                         "mean_lateral_mm": float(np.mean(d["lats"])), "max_lateral_mm": float(np.max(d["lats"]))})
        lats = [x for d in per.values() for x in d["lats"]]
        return clean({"n": len(lats), "points": rows,
                      "mean_lateral_mm": float(np.mean(lats)) if lats else None,
                      "max_lateral_mm": float(np.max(lats)) if lats else None})

    def state(self) -> dict:
        if not self.supported:
            return {"supported": False, "mode": self.mode, "state": "idle", "message": "single-arm modes only",
                    "arm": None, "calib_samples": None, "points": [], "test_summary": self._empty_summary(),
                    "correction_mm": {a: list(self._correction(a)) for a in CORRECTION_ARMS}, "correction_arm": None,
                    "prepared": None, "plan": None, "selected": None}
        sd = None
        try:
            sd = self.session_dir_fn()
        except Exception:  # noqa: BLE001
            pass
        self._ensure_records(sd)
        with self.lock:
            st, msg, arm = self._state, self._message, self._arm
            based, at, plan = self._based_on, self._computed_at, (dict(self._plan) if self._plan else None)
            prep = None if self._prepared is None else {k: v for k, v in self._prepared.items() if k not in ("q", "T_base_tcp")}
            sel, status, catalog = self._selected, dict(self._status), list(self._catalog)
            preds = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in self._preds.items()}
            last = self._last_offsets()
            summary = self._test_summary()
            calib = self._calib_samples
        count_arm = arm or next(iter(self.cfgs), None)
        try:
            now_n = self.counter.count(sd, count_arm) if count_arm else None
        except Exception:  # noqa: BLE001
            now_n = None
        points = []
        ids = [c[0] for c in catalog] + [k for k in sorted(preds) if k not in {c[0] for c in catalog}]
        names = {c[0]: c for c in catalog}
        for pid in ids:
            if pid in preds:
                e = dict(preds[pid])
            else:
                c = names.get(pid)
                src = c[2] if c else ""
                e = {"id": pid, "name": c[1] if c else f"P{pid}", "source": src, "detected": False, "quality": {},
                     "pred_base_mm": None, "uv": None,
                     "reason": self._reason(st, msg, status.get(src, ""))}
            e["last_offset"] = last.get(pid)
            points.append(clean(e))
        why_il = self._interlock(None, clear=False)
        with self.lock:
            reached = self._pending_move["point_id"] if (self._pending_move is not None and not why_il) else None
            dsc = None if self._descended is None else dict(self._descended)
        if dsc is not None:
            try:
                last = (self.jog.status().get("last") or {})
            except Exception:  # noqa: BLE001
                last = {}
            if dsc["id"] is not None and last.get("id") == dsc["id"]:
                dsc["phase"] = last.get("phase")
            else:
                dsc = None                      # another motion happened since: not "at hover height" any more
        return {"descended": clean(dsc), "catalog_ids": [c[0] for c in catalog], "supported": True, "mode": self.mode, "state": st, "message": msg, "arm": arm, "calib_samples": calib,
                "reached_point": reached, "interlock": why_il,
                "correction_mm": {a: list(self._correction(a)) for a in CORRECTION_ARMS}, "correction_arm": next(iter(self.cfgs)),
                "based_on_samples": based, "session_samples": now_n,
                "stale": bool(st == "ready" and based is not None and now_n is not None and now_n != based),
                "age_s": None if at is None else max(0.0, self.clock() - at),
                "points": points, "selected": sel, "prepared": clean(prep), "plan": plan,
                "test_summary": summary, "status": clean(status),
                "heights_mm": {"prehover": PREHOVER_HEIGHT_M * 1e3, "descend": list(DESCEND_HEIGHTS_MM),
                               "default_descend": DEFAULT_DESCEND_MM}}

    @staticmethod
    def _reason(state: str, message: str, source_status: str) -> str:
        """Why a row is not usable: after a FAILED detection every row says why (the camera text), never an old summary."""
        if state == "error":
            return source_status or message or "no detection"
        if state == "ready":
            return source_status or "not detected"
        return message or "press Detect points"

    @staticmethod
    def _empty_summary() -> dict:
        return {"n": 0, "points": [], "mean_lateral_mm": None, "max_lateral_mm": None}

    def overlay(self, selected: Optional[int] = None) -> Optional[dict]:
        """{points: [{id, name, uv, selected}], label} of the detected marks for the live image, or None."""
        with self.lock:
            if self._state != "ready" or not self._preds:
                return None
            sel = self._selected if selected is None else selected
            pts = [{"id": p["id"], "name": p["name"], "uv": p["uv"], "selected": p["id"] == sel}
                   for p in self._preds.values() if p.get("uv") is not None]
        return {"points": pts} if pts else None
