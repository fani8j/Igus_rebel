"""Calibration session directory I/O (no ROS imports).

Layout::

    <session>/session.json      targets config, frames, creation time
    <session>/samples.jsonl     one JSON object per (capture, arm) sample
    <session>/table_obs.jsonl   table-board corners of images in which the fixed table board was detected
    <session>/touches.jsonl     TCP-on-board-point records (arm stationary)
    <session>/robot_description.urdf   the URDF used for forward kinematics (saved once; a TCP change after this is
                                       NOT picked up: touches carry T_link6_tcp, see ``solve --tcp-file``)
    <session>/images/cap_0001.png
    <session>/intrinsics.jsonl  detections of the intrinsics board (capture_intrinsics), images in intrinsics/
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .kinematics import ArmKinematics, KinematicsError, Urdf
from .solver import Sample, TableObservation, Touch
from .transforms import rotation_angle_deg


def samples_path(session_dir: str) -> str:
    return os.path.join(session_dir, "samples.jsonl")


def table_path(session_dir: str) -> str:
    return os.path.join(session_dir, "table_obs.jsonl")


def touches_path(session_dir: str) -> str:
    return os.path.join(session_dir, "touches.jsonl")


def intrinsics_path(session_dir: str) -> str:
    return os.path.join(session_dir, "intrinsics.jsonl")


def append_intrinsics_view(session_dir: str, view: dict) -> None:
    _append_jsonl(intrinsics_path(session_dir), view)


def load_intrinsics_views(session_dir: str) -> List[dict]:
    return _read_jsonl(intrinsics_path(session_dir))


def update_session_info(session_dir: str, info: dict) -> None:
    """Store the session metadata (ambient temperature / humidity, warm-up time, operator note) in session.json."""
    mp = os.path.join(session_dir, "session.json")
    meta = {}
    if os.path.exists(mp):
        with open(mp) as f:
            meta = json.load(f)
    if meta.get("session_info") == info:
        return
    meta["session_info"] = info
    tmp = mp + ".tmp"
    with open(tmp, "w") as f:
        json.dump(meta, f, indent=2)
    os.replace(tmp, mp)


def urdf_path(session_dir: str) -> str:
    return os.path.join(session_dir, "robot_description.urdf")


def init_session(session_dir: str, meta: dict) -> int:
    """Create the directory (and session.json if new).  Returns the number of existing samples."""
    os.makedirs(os.path.join(session_dir, "images"), exist_ok=True)
    meta_path = os.path.join(session_dir, "session.json")
    if not os.path.exists(meta_path):
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
    return len(load_samples(session_dir)) if os.path.exists(samples_path(session_dir)) else 0


def append_sample(session_dir: str, sample: Sample, extra: dict) -> None:
    d = sample.to_json_dict()
    d.update(extra)
    with open(samples_path(session_dir), "a") as f:
        f.write(json.dumps(d) + "\n")


def _read_jsonl(path: str) -> List[dict]:
    out = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
    return out


def _append_jsonl(path: str, d: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(d) + "\n")


def load_samples(session_dir: str) -> List[Sample]:
    return [Sample.from_json_dict(d) for d in _read_jsonl(samples_path(session_dir))]


def append_table_obs(session_dir: str, obs: TableObservation, extra: Optional[dict] = None) -> None:
    d = obs.to_json_dict()
    d.update(extra or {})
    _append_jsonl(table_path(session_dir), d)


def load_table_obs(session_dir: str) -> List[TableObservation]:
    return [TableObservation.from_json_dict(d) for d in _read_jsonl(table_path(session_dir))]


def append_touch(session_dir: str, touch: Touch, extra: Optional[dict] = None) -> None:
    d = touch.to_json_dict()
    d.update(extra or {})
    _append_jsonl(touches_path(session_dir), d)


def load_touches(session_dir: str) -> List[Touch]:
    return [Touch.from_json_dict(d) for d in _read_jsonl(touches_path(session_dir))]


def save_urdf_once(session_dir: str, text: str) -> bool:
    """Write robot_description.urdf unless it exists.  Returns True if written."""
    p = urdf_path(session_dir)
    if os.path.exists(p):
        return False
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, p)
    return True


def load_session(session_dir: str) -> Tuple[List[Sample], dict]:
    meta = {}
    mp = os.path.join(session_dir, "session.json")
    if os.path.exists(mp):
        with open(mp) as f:
            meta = json.load(f)
    return load_samples(session_dir), meta


class SessionMismatchError(ValueError):
    """Sessions that must share one camera disagree (intrinsics, image size, camera_link->optical)."""


# A merge is only meaningful if the camera and its calibration did not change between sessions.
TOL_FOCAL_CENTER_PX = 0.5      # fx, fy, cx, cy
TOL_DIST = 1e-3                # distortion coefficients
TOL_T_CAMLINK_M = 1e-3         # camera_link -> optical translation
TOL_T_CAMLINK_DEG = 0.1        # camera_link -> optical rotation


def check_session_consistency(per_session: Sequence[Tuple[str, Sequence[Sample]]]) -> None:
    """Raise SessionMismatchError unless all samples share image size, intrinsics and camera_link->optical.

    ``per_session`` is a list of (label, samples).  The first sample of the first non-empty session is
    the reference; the first disagreement per session is reported with the session label and image.
    """
    ref: Optional[Sample] = None
    ref_label = ""
    for label, samples in per_session:
        for s in samples:
            if ref is None:
                ref, ref_label = s, label
                continue
            problems = []
            if tuple(s.image_size) != tuple(ref.image_size):
                problems.append(f"image size {tuple(s.image_size)} vs {tuple(ref.image_size)}")
            dK = max(abs(s.K[i, j] - ref.K[i, j]) for i, j in ((0, 0), (1, 1), (0, 2), (1, 2)))
            if dK > TOL_FOCAL_CENTER_PX:
                problems.append(f"camera matrix differs by {dK:.2f} px in fx/fy/cx/cy (tolerance {TOL_FOCAL_CENTER_PX} px)")
            n = max(len(s.dist), len(ref.dist))
            d1 = np.zeros(n)
            d2 = np.zeros(n)
            d1[:len(s.dist)], d2[:len(ref.dist)] = s.dist, ref.dist
            dd = float(np.max(np.abs(d1 - d2))) if n else 0.0
            if dd > TOL_DIST:
                problems.append(f"distortion coefficients differ by {dd:.4f} (tolerance {TOL_DIST})")
            dt = float(np.linalg.norm(s.T_camlink_optical[:3, 3] - ref.T_camlink_optical[:3, 3]))
            dr = rotation_angle_deg(s.T_camlink_optical[:3, :3], ref.T_camlink_optical[:3, :3])
            if dt > TOL_T_CAMLINK_M or dr > TOL_T_CAMLINK_DEG:
                problems.append(f"camera_link->optical TF differs by {dt * 1e3:.2f} mm / {dr:.3f} deg "
                                f"(tolerance {TOL_T_CAMLINK_M * 1e3:.1f} mm / {TOL_T_CAMLINK_DEG} deg)")
            if problems:
                raise SessionMismatchError(
                    f"session '{label}' (sample {s.image or '?'}, arm {s.arm}) does not match the camera of "
                    f"session '{ref_label}': " + "; ".join(problems) +
                    ". Sessions can only be merged when they were recorded with the same camera, driver "
                    "calibration and mounting; record both arms again in one session or fix the camera setup.")


def unique_labels(session_dirs: Sequence[str]) -> List[str]:
    """Short unique labels (directory names, disambiguated with an index) for merged sessions."""
    names = [os.path.basename(os.path.normpath(d)) or d for d in session_dirs]
    out = []
    for i, n in enumerate(names):
        out.append(n if names.count(n) == 1 else f"{n}#{i + 1}")
    return out


@dataclass
class SessionData:
    samples: List[Sample]
    tables: List[TableObservation]
    touches: List[Touch]
    metas: List[dict]
    per_session: List[Tuple[str, List[Sample]]]
    kin: Dict[str, ArmKinematics] = field(default_factory=dict)
    kin_notes: List[str] = field(default_factory=list)


def _session_kinematics(d: str, meta: dict, notes: List[str]) -> Dict[str, ArmKinematics]:
    p = urdf_path(d)
    if not os.path.exists(p):
        return {}
    par = meta.get("parameters", {})
    out = {}
    try:
        urdf = Urdf.load(p)
    except (KinematicsError, OSError) as e:
        notes.append(f"{d}: cannot use robot_description.urdf ({e})")
        return {}
    for arm in par.get("active_arms", ["left", "right"]):
        base, tip = par.get(f"{arm}_base_frame"), par.get(f"{arm}_tip_frame")
        if not base or not tip:
            continue
        try:
            out[arm] = ArmKinematics.from_urdf(urdf, base, tip, par.get(f"{arm}_tcp_frame", ""))
        except KinematicsError as e:
            notes.append(f"{d}: no kinematics for arm {arm} ({e})")
    return out


def load_sessions_full(session_dirs: Sequence[str]) -> SessionData:
    """Load and merge several sessions: samples, table-board observations, touches and arm kinematics."""
    real = [os.path.realpath(d) for d in session_dirs]
    if len(set(real)) != len(real):
        raise SessionMismatchError("the same session directory was given more than once")
    labels = unique_labels(session_dirs)
    per_session, metas, tables, touches = [], [], [], []
    kin: Dict[str, ArmKinematics] = {}
    notes: List[str] = []
    multi = len(session_dirs) > 1
    for d, label in zip(session_dirs, labels):
        if not any(os.path.exists(f(d)) for f in (samples_path, table_path, touches_path)):
            raise SessionMismatchError(f"'{d}' has no samples.jsonl (not a recorder session directory)")
        samples, meta = load_session(d)
        tbl, tch = load_table_obs(d), load_touches(d)
        sk = _session_kinematics(d, meta, notes)
        for arm, k in sk.items():
            if arm in kin and not kin[arm].same_geometry(k, ignore_tcp_offset=True):
                raise SessionMismatchError(
                    f"session '{label}': the URDF kinematics of arm '{arm}' differ from the earlier session; "
                    "sessions can only be merged for the same robot")
            kin.setdefault(arm, k)
        for s in samples + tch:
            if s.arm in sk and s.joints:
                try:
                    s.q = sk[s.arm].q_from_joints(s.joints)
                except KinematicsError:
                    pass
        for t in tch:
            t.session = label
            # the link6 -> TCP transform the touch was recorded with: stored by the recorder, else the one of the
            # session URDF (legacy touches).  A TCP change between sessions is handled by solve --tcp-file.
            if t.T_link6_tcp is None and t.arm in sk and sk[t.arm].T_link6_tcp() is not None:
                t.T_link6_tcp, t.tcp_source = sk[t.arm].T_link6_tcp(), "session_urdf"
            elif t.T_link6_tcp is not None:
                t.tcp_source = "recorded"
        if multi:
            for s in samples + tbl:
                s.image = f"{label}/{s.image}"
        per_session.append((label, samples, tbl))
        metas.append(meta)
        tables += tbl
        touches += tch
    check_session_consistency([(label, list(s) + list(t)) for label, s, t in per_session])
    merged = [s for _, samples, _ in per_session for s in samples]
    return SessionData(merged, tables, touches, metas, [(l, s) for l, s, _ in per_session], kin, notes)


def load_sessions(session_dirs: Sequence[str]) -> Tuple[List[Sample], List[dict], List[Tuple[str, List[Sample]]]]:
    """Load and merge several sessions (samples only).

    Returns (merged samples, metas, [(label, samples)] per session).  Duplicate directories are rejected.
    Raises SessionMismatchError if the sessions were recorded with a different camera setup.  With more
    than one session the sample image names are prefixed with the session label so that they stay unique.
    """
    data = load_sessions_full(session_dirs)
    return data.samples, data.metas, data.per_session
