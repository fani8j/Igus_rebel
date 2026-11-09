"""Pure (ROS-free) logic of the passive side-cut probe, driven by `probe_node`.

Everything here takes plain numpy / dict inputs and returns plain outputs, so the node-level behaviour
(live gating, pair selection, TF staleness, LOOK sessions, record saving, overlay + JSON outputs) is
unit-tested with fakes. The node only moves messages in and publishes the returned outputs.

Times are seconds on the host clock (header stamps and `now` share it). Nothing here sleeps.
"""
from __future__ import annotations

import json
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional, Tuple

import numpy as np

from .calib import HandEye
from .camera import CameraError, CameraModel
from .detect import DetectParams
from .geom import make_T
from .measure import Measurement, agree, measure
from .model import ArmModel, Snapshot, SnapshotError, build_arm_model, load_snapshot
from .overlay import draw_overlay, encode_jpeg
from .record import SUBDIR, look_dir_name, save_record
from .toolmask import T_LINK6_TOOL0, tool_mask_from_geometry
from .trigger import LookTrigger, StillState, StillTracker, select_record_dir

ACTIVE_COORDINATOR_STATES = (3, 4)         # dual_cut_msgs/CoordinatorStatus WAITING_BARRIER, EXECUTING
MAX_PAIR_DT_S = 0.040                      # colour/depth stamp agreement
MAX_TF_AGE_S = 0.050                       # a joint-state sample must lie within this of the image stamp


# ----------------------------------------------------------------------------------------------- gating
class LiveGate:
    """Live processing runs while the coordinator is WAITING_BARRIER/EXECUTING (plus a short hold so the
    last moments of a clip are covered) or always when always_live."""

    def __init__(self, always_live: bool = False, hold_s: float = 3.0):
        self.always = always_live
        self.hold_s = hold_s
        self._last_active: Optional[float] = None
        self.state: Optional[int] = None

    def on_status(self, state: int, now: float) -> None:
        was_active = self.state in ACTIVE_COORDINATOR_STATES
        self.state = int(state)
        if self.state in ACTIVE_COORDINATOR_STATES or was_active:
            self._last_active = now          # the hold runs from the moment we learn the run ended

    def active(self, now: float) -> bool:
        if self.always:
            return True
        if self.state in ACTIVE_COORDINATOR_STATES:
            return True
        return self._last_active is not None and now - self._last_active <= self.hold_s


# ----------------------------------------------------------------------------------------------- frames
@dataclass
class Stamped:
    stamp: float
    data: np.ndarray
    frame_id: str = ""


class PairBuffer:
    """The newest few colour and depth frames of one arm; picks the best colour/depth pair."""

    def __init__(self, maxlen: int = 4):
        self.color: Deque[Stamped] = deque(maxlen=maxlen)
        self.depth: Deque[Stamped] = deque(maxlen=maxlen)

    def add_color(self, f: Stamped) -> None:
        self.color.append(f)

    def add_depth(self, f: Stamped) -> None:
        self.depth.append(f)

    def newest_pair(self, after_stamp: float = -1.0, max_dt: float = MAX_PAIR_DT_S
                    ) -> Tuple[Optional[Stamped], Optional[Stamped], List[str]]:
        """Newest colour newer than after_stamp with a depth within max_dt. Returns (color, depth, reasons);
        reasons is ['no_color'], ['no_depth'] or ['stale_pair'] when nothing usable."""
        colors = [c for c in self.color if c.stamp > after_stamp]
        if not self.color:
            return None, None, ["no_color"]
        if not colors:
            return None, None, []                       # nothing new yet: not an error
        if not self.depth:
            return None, None, ["no_depth"]
        best = None
        for c in reversed(colors):                      # newest first
            d = min(self.depth, key=lambda x: abs(x.stamp - c.stamp))
            if abs(d.stamp - c.stamp) <= max_dt:
                best = (c, d)
                break
        if best is None:
            return colors[-1], min(self.depth, key=lambda x: abs(x.stamp - colors[-1].stamp)), ["stale_pair"]
        return best[0], best[1], []


class JointHistory:
    """Stamps (and latest values) of /joint_states, to check the TF/joint data around an image stamp."""

    def __init__(self, maxlen: int = 400):
        self.stamps: Deque[float] = deque(maxlen=maxlen)
        self.names: List[str] = []
        self.position: List[float] = []
        self.velocity: List[float] = []

    def add(self, stamp: float, names, position, velocity=None) -> None:
        self.stamps.append(stamp)
        self.names, self.position = list(names), list(position)
        self.velocity = list(velocity) if velocity else []

    def nearest_dt(self, stamp: float) -> Optional[float]:
        if not self.stamps:
            return None
        return min(abs(s - stamp) for s in self.stamps)

    def snapshot(self, joint_names) -> dict:
        mp = dict(zip(self.names, self.position))
        return {"name": list(joint_names), "position": [mp.get(n) for n in joint_names],
                "stamps_newest": list(self.stamps)[-1:]}


# ----------------------------------------------------------------------------------------------- snapshot
class SnapshotCache:
    """Reload the snapshot file when its mtime/size changes; keep the last good one if a reload fails."""

    def __init__(self, path: str):
        self.path = path
        self._key: Optional[Tuple[float, int]] = None
        self.snapshot: Optional[Snapshot] = None
        self.error: Optional[str] = None
        self._models: Dict[str, ArmModel] = {}
        self.reloads = 0

    def poll(self) -> bool:
        """Returns True if the snapshot was (re)loaded in this call."""
        try:
            st = os.stat(self.path)
            key = (st.st_mtime, st.st_size)
        except OSError as exc:
            self.error = f"snapshot file not readable: {exc.strerror}"
            return False
        if key == self._key:
            return False
        self._key = key
        try:
            snap = load_snapshot(self.path)
        except SnapshotError as exc:
            self.error = str(exc)
            return False
        self.snapshot, self.error = snap, None
        self._models.clear()
        self.reloads += 1
        return True

    def model(self, arm: str) -> Optional[ArmModel]:
        if self.snapshot is None:
            return None
        m = self._models.get(arm)
        if m is None:
            try:
                m = self._models[arm] = build_arm_model(self.snapshot, arm)
            except SnapshotError as exc:
                self.error = str(exc)
                return None
        return m


# ----------------------------------------------------------------------------------------------- environment
class Env:
    """What the node provides: TF lookups and the clock. Tests supply fakes."""

    def lookup(self, target: str, source: str, stamp_s: Optional[float]) -> Optional[np.ndarray]:
        raise NotImplementedError

    def lookup_latest(self, target: str, source: str) -> Optional[Tuple[np.ndarray, float]]:
        """(transform, its stamp) of the newest TF sample, or None."""
        raise NotImplementedError

    def now_s(self) -> float:
        return time.time()


@dataclass
class ArmConfig:
    arm: str
    prefix: str
    world_frame: str = "world"
    joint_names: List[str] = field(default_factory=list)
    calib_status: str = ""

    @property
    def link6(self) -> str:
        return f"{self.prefix}link6"

    @property
    def tool0(self) -> str:
        return f"{self.prefix}igus_tool0"

    @property
    def tool_tip(self) -> str:
        return f"{self.prefix}xeg32_tool_tip"

    @staticmethod
    def default(arm: str, world_frame: str = "world") -> "ArmConfig":
        prefix = f"arm_{arm}_"
        return ArmConfig(arm, prefix, world_frame, [f"{prefix}joint{i}" for i in range(1, 7)])


@dataclass
class Outputs:
    arm: str
    overlay_jpeg: Optional[bytes] = None
    overlay_stamp: float = 0.0
    measurement_json: Optional[str] = None
    log: List[str] = field(default_factory=list)


@dataclass
class _LookFrame:
    meas: Measurement
    stamp: float
    color: np.ndarray
    depth: np.ndarray
    T_wo: np.ndarray
    T_wt: np.ndarray
    joints: dict
    t_proc: float


@dataclass
class _LookSession:
    start_t: float
    still_since: float
    frames: List[_LookFrame] = field(default_factory=list)


class ArmProbe:
    """One arm's passive probe: live measurement + once-per-still-period LOOK record."""

    def __init__(self, cfg: ArmConfig, hand_eye: HandEye, snapshots: SnapshotCache, out_dir: str,
                 snapshot_path: str, params: Optional[DetectParams] = None, live_rate_hz: float = 3.0,
                 look_min_still_s: float = 0.3, look_frames: int = 3, look_period_s: float = 0.12,
                 look_timeout_s: float = 2.0, gate: Optional[LiveGate] = None,
                 record_dir_fn: Callable[[str, float], str] = select_record_dir):
        self.cfg = cfg
        self.he = hand_eye
        self.snapshots = snapshots
        self.out_dir = out_dir
        self.snapshot_path = snapshot_path
        self.params = params or DetectParams()
        self.live_period = 1.0 / max(live_rate_hz, 0.1)
        self.look_frames = look_frames
        self.look_period = look_period_s
        self.look_timeout = look_timeout_s
        self.gate = gate or LiveGate()
        self.record_dir_fn = record_dir_fn
        self.pairs = PairBuffer()
        self.joints = JointHistory()
        self.tracker = StillTracker(cfg.joint_names)
        self.still = StillState(False, 0.0, float("nan"), False)
        self.trigger: Optional[LookTrigger] = None
        self._trigger_model_id: Optional[str] = None
        self.cam: Optional[CameraModel] = None
        self.look: Optional[_LookSession] = None
        self.last_stamp = -1.0
        self.last_proc_t = -1e9
        self._stale_stamp = None
        self.tool_mask: Optional[np.ndarray] = None
        self.tool_mask_kind = "static_polygon"
        self._T_l6_t0: Optional[np.ndarray] = None
        self._T_l6_t0_kind = "urdf_const"
        self.counters = {"frames": 0, "skipped_tf": 0, "skipped_pair": 0, "looks_saved": 0,
                         "looks_discarded": 0, "errors": 0}
        self.last_record: Optional[str] = None
        self.last_error: Optional[str] = None
        self._rec_dir_cache: Tuple[float, str] = (-1e9, "")

    # ------------------------------------------------------------------------- inputs
    def on_joint_state(self, stamp: float, names, position, velocity=None) -> None:
        self.joints.add(stamp, names, position, velocity)
        self.still = self.tracker.update(stamp, names, position)

    def set_camera(self, cam: CameraModel) -> None:
        if self.cam is None or (cam.K != self.cam.K).any() or (cam.D.shape != self.cam.D.shape) \
                or (cam.D != self.cam.D).any() or cam.size != self.cam.size:
            self.cam = cam
            self._build_tool_mask()

    def _build_tool_mask(self) -> None:
        self.tool_mask = None
        self.tool_mask_kind = "static_polygon"
        if self.cam is None:
            return
        try:
            self.tool_mask = tool_mask_from_geometry(self.cam, self.he.T_link6_optical, self._T_l6_t0)
            self.tool_mask_kind = f"hull_{self._T_l6_t0_kind}+near_field"
        except Exception as exc:                      # noqa: BLE001 - never fatal: static fallback
            self.last_error = f"tool mask: {exc}"

    def _ensure_tool0(self, env: Env) -> None:
        if self._T_l6_t0_kind == "tf":
            return
        T = env.lookup(self.cfg.link6, self.cfg.tool0, None)
        if T is not None and np.all(np.isfinite(T)) and np.abs(T[:3, 3] - T_LINK6_TOOL0[:3, 3]).max() < 0.02:
            self._T_l6_t0, self._T_l6_t0_kind = T, "tf"
            self._build_tool_mask()
        elif self._T_l6_t0 is None:
            self._T_l6_t0 = T_LINK6_TOOL0.copy()
            self._build_tool_mask()

    # ------------------------------------------------------------------------- main step
    def step(self, env: Env) -> List[Outputs]:
        now = env.now_s()
        out: List[Outputs] = []
        try:
            self.snapshots.poll()
            model = self.snapshots.model(self.cfg.arm)
            active = self.gate.active(now)
            self._ensure_tool0(env)
            if model is not None and (self.trigger is None or self._trigger_model_id != model.snapshot_id):
                self.trigger = LookTrigger(model, 0.3, 0.040)
                self._trigger_model_id = model.snapshot_id
            # LOOK trigger (needs the live gate: only during coordinator runs / always_live)
            if active and model is not None and self.trigger is not None and self.look is None:
                tip = env.lookup(self.cfg.world_frame, self.cfg.tool_tip, None)
                if tip is not None and self.trigger.update(self.still, tip[:3, 3]):
                    since = now - self.still.still_for_s
                    self.look = _LookSession(now, since)
                    out.append(Outputs(self.cfg.arm, log=[f"LOOK trigger {self.cfg.arm}: still "
                                                          f"{self.still.still_for_s:.2f}s at the cut-side edge"]))
            elif not active and self.look is None and self.trigger is not None:
                self.trigger.update(StillState(False, 0.0, float("nan"), False), None)   # re-arm
            if self.look is not None:
                out.extend(self._maybe_finish_look(env, now, model))
            if (not active and self.look is None) or model is None or self.cam is None:
                return out
            period = self.look_period if self.look is not None else self.live_period
            if now - self.last_proc_t < period:
                return out
            c, d, reasons = self.pairs.newest_pair(self.last_stamp)
            if c is None:
                return out
            if reasons:
                # the colour frame often arrives a few ms before its depth frame: retry on the next poll (the
                # period timer is NOT advanced); count a colour stamp as skipped only once
                if "stale_pair" in reasons and c is not None and c.stamp != self._stale_stamp:
                    self._stale_stamp = c.stamp
                    self.counters["skipped_pair"] += 1
                return out
            self.last_stamp = c.stamp
            self.last_proc_t = now
            res = self._process(env, now, c, d, model)
            if res is not None:
                out.append(res)
            if self.look is not None:
                out.extend(self._maybe_finish_look(env, now, model))
        except Exception as exc:                      # noqa: BLE001 - the probe must never die
            self.counters["errors"] += 1
            self.last_error = f"{type(exc).__name__}: {exc}"
            out.append(Outputs(self.cfg.arm, log=[f"ERROR {self.cfg.arm}: {self.last_error}"]))
        return out

    # ------------------------------------------------------------------------- one frame
    def _process(self, env: Env, now: float, c: Stamped, d: Stamped, model: ArmModel) -> Optional[Outputs]:
        cfg = self.cfg
        dt_j = self.joints.nearest_dt(c.stamp)
        T_l6 = env.lookup(cfg.world_frame, cfg.link6, c.stamp)
        T_wt = env.lookup(cfg.world_frame, cfg.tool_tip, c.stamp)
        tf_mode = "at_stamp"
        if T_l6 is None or T_wt is None:
            # robot_state_publisher only publishes TF when joints move (and at <= 20 Hz), so a TF at the exact
            # image stamp is missing when the arm is at rest or the image is newer than the last sample.
            # Accept the newest sample if it is within 60 ms of the image, or the arm has been still (joint-state
            # velocity test) long enough that the last published pose IS the current pose.
            l6 = env.lookup_latest(cfg.world_frame, cfg.link6)
            tt = env.lookup_latest(cfg.world_frame, cfg.tool_tip)
            if l6 is not None and tt is not None:
                near = abs(l6[1] - c.stamp) <= MAX_TF_AGE_S + 0.01 and abs(tt[1] - c.stamp) <= MAX_TF_AGE_S + 0.01
                at_rest = self.still.still and self.still.still_for_s >= 0.2 and c.stamp >= now - 2.0 \
                    and l6[1] <= c.stamp and tt[1] <= c.stamp
                if near or at_rest:
                    T_l6, T_wt, tf_mode = l6[0], tt[0], "latest_near" if near else "latest_at_rest"
        if (T_l6 is None or T_wt is None or dt_j is None or dt_j > MAX_TF_AGE_S
                or not np.all(np.isfinite(T_l6)) or not np.all(np.isfinite(T_wt))):
            self.counters["skipped_tf"] += 1
            return None
        T_wo = T_l6 @ self.he.T_link6_optical
        meta = {"calib_status": self.he.calib_status, "calib_sha256": self.he.sha256,
                "snapshot_id": model.snapshot_id, "tool_mask": self.tool_mask_kind,
                "image_stamp": c.stamp, "tf_mode": tf_mode}
        meas = measure(c.data, d.data, self.cam, T_wo, model, self.params, tip_world=T_wt[:3, 3],
                       color_stamp_s=c.stamp, depth_stamp_s=d.stamp, meta=meta, tool_mask=self.tool_mask,
                       tool_mask_kind=self.tool_mask_kind)
        self.counters["frames"] += 1
        state = "LOOK" if self.look is not None else "LIVE"
        ov = draw_overlay(c.data, self.cam, T_wo, model, meas, T_wt[:3, 3], state=state, ros_time_s=c.stamp,
                          calib_status=self.he.calib_status, snapshot_id=model.snapshot_id)
        jpeg = encode_jpeg(ov, 80)
        md = meas.to_dict()
        md["state"] = state
        md["stamp"] = c.stamp
        o = Outputs(cfg.arm, jpeg, c.stamp, json.dumps(md, allow_nan=False))
        self._append_jsonl(now, md)
        if self.look is not None and c.stamp >= self.look.still_since and self.still.still:
            self.look.frames.append(_LookFrame(meas, c.stamp, c.data, d.data, T_wo, T_wt,
                                               self.joints.snapshot(cfg.joint_names), now))
        return o

    # ------------------------------------------------------------------------- LOOK
    def _maybe_finish_look(self, env: Env, now: float, model: Optional[ArmModel]) -> List[Outputs]:
        s = self.look
        if s is None:
            return []
        moved = not self.still.still
        timeout = now - s.start_t > self.look_timeout
        if len(s.frames) < self.look_frames and not moved and not timeout:
            return []
        self.look = None
        out = []
        if len(s.frames) >= 2 and model is not None:
            try:
                path = self._save_look(env, now, s, model)
                self.counters["looks_saved"] += 1
                self.last_record = path
                out.append(Outputs(self.cfg.arm, log=[f"LOOK record saved: {path}"]))
            except Exception as exc:                  # noqa: BLE001
                self.counters["errors"] += 1
                self.last_error = f"save look: {type(exc).__name__}: {exc}"
                out.append(Outputs(self.cfg.arm, log=[f"ERROR {self.cfg.arm}: {self.last_error}"]))
        else:
            self.counters["looks_discarded"] += 1
            why = "arm moved" if moved else "timeout"
            out.append(Outputs(self.cfg.arm, log=[f"LOOK discarded ({why}): {len(s.frames)} frame(s)"]))
        return out

    def _record_dir(self, now: float) -> str:
        t, d = self._rec_dir_cache
        if now - t > 1.0:
            d = self.record_dir_fn(self.out_dir, now)
            self._rec_dir_cache = (now, d)
        return d

    def _save_look(self, env: Env, now: float, s: _LookSession, model: ArmModel) -> str:
        frames = s.frames
        last = frames[-1]
        pair_ok = all(agree(frames[i].meas, frames[i + 1].meas, 1.0) for i in range(len(frames) - 1))
        side = os.path.join(self._record_dir(now), SUBDIR)
        os.makedirs(side, exist_ok=True)
        path = os.path.join(side, look_dir_name(side, self.cfg.arm, now))
        c = last.meas
        T_wo, T_wt = last.T_wo, last.T_wt
        # the look carries its own copy of the facts needed to reproduce it offline
        c.meta.update({
            "calib_status": self.he.calib_status, "calib_sha256": self.he.sha256,
            "calib_path": self.he.path, "snapshot_id": model.snapshot_id,
            "snapshot_path": self.snapshot_path, "tool_mask": self.tool_mask_kind,
            "world_frame": self.cfg.world_frame, "image_stamp": last.stamp,
        })
        d = c.to_dict()
        ds = [[f.meas.delta_mm[i] for i in range(3)] + [f.meas.delta_yaw_deg] for f in frames]
        d["look"] = {"frames": len(frames), "per_frame_delta_mm_yaw": ds,
                     "unstable": not pair_ok, "still_since": s.still_since,
                     "image_stamps": [f.stamp for f in frames]}
        if not pair_ok and "unstable" not in c.reasons:
            c.reasons.append("unstable")
            d["reasons"] = list(c.reasons)
        ov = draw_overlay(last.color, self.cam, T_wo, model, c, T_wt[:3, 3], state="LOOK saved",
                          ros_time_s=last.stamp, calib_status=self.he.calib_status,
                          snapshot_id=model.snapshot_id,
                          extra_banners=["UNSTABLE"] if not pair_ok else None)
        save_record(path, self.cfg.arm, last.color, last.depth, self.cam, T_wo, T_wt, last.joints,
                    self.snapshot_path, self.he.path, c, ov)
        # measurement.json written by save_record has no look block: rewrite with it
        with open(os.path.join(path, "measurement.json"), "w") as f:
            json.dump(d, f, indent=2, allow_nan=False)
        return path

    # ------------------------------------------------------------------------- live log
    def _append_jsonl(self, now: float, md: dict) -> None:
        try:
            side = os.path.join(self._record_dir(now), SUBDIR)
            os.makedirs(side, exist_ok=True)
            line = {"t": round(now, 3), "stamp": md.get("stamp"), "arm": self.cfg.arm, "state": md.get("state"),
                    "ok": md["ok"], "seam_ok": md["seam_ok"], "reasons": md["reasons"],
                    "delta_mm": md["delta_mm"], "delta_yaw_deg": md["delta_yaw_deg"],
                    "tip_vs_seam": md["tip_vs_seam"], "across_convention": "outward",
                    "calib_status": self.he.calib_status, "snapshot_id": md["meta"].get("snapshot_id")}
            with open(os.path.join(side, f"{self.cfg.arm}_live.jsonl"), "a") as f:
                f.write(json.dumps(line, allow_nan=False) + "\n")
        except (OSError, ValueError) as exc:
            self.last_error = f"jsonl: {exc}"


def matrix_from_tf(translation, quat_xyzw) -> np.ndarray:
    from .geom import quat_xyzw_to_matrix
    return make_T(quat_xyzw_to_matrix(quat_xyzw), translation)
