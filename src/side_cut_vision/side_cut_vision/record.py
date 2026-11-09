"""LOOK record folder: write and read (the layout the replay CLI consumes).

<record_dir>/side_cut_vision/<arm>_<NN>_<HHMMSS>/
    color.png  depth.png (16-bit)  camera_info.json  T_world_optical.json  T_world_tip.json
    joint_state.json  snapshot.json  calib.yaml  measurement.json  overlay.png
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import re
import shutil
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .camera import CameraModel
from .measure import Measurement
from .model import Snapshot, SnapshotError, load_snapshot
from .overlay import to_bgr

SUBDIR = "side_cut_vision"


def look_dir_name(side_dir: str, arm: str, when: Optional[float] = None) -> str:
    """Folder name <arm>_<NN>_<HHMMSS>; NN counts the arm's existing looks in side_dir."""
    n = 1
    if os.path.isdir(side_dir):
        pat = re.compile(r"^%s_\d+_" % re.escape(arm))
        n = 1 + sum(1 for e in os.listdir(side_dir) if pat.match(e) and os.path.isdir(os.path.join(side_dir, e)))
    t = _dt.datetime.fromtimestamp(when if when is not None else _dt.datetime.now().timestamp())
    return f"{arm}_{n:02d}_{t.strftime('%H%M%S')}"


def _write_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def save_record(path: str, arm: str, color_rgb: np.ndarray, depth: np.ndarray, cam: CameraModel,
                T_wo, T_w_tip, joint_state: dict, snapshot_path: str, calib_path: Optional[str],
                meas: Measurement, overlay_rgb: np.ndarray) -> str:
    os.makedirs(path, exist_ok=True)
    cv2.imwrite(os.path.join(path, "color.png"), to_bgr(color_rgb))
    cv2.imwrite(os.path.join(path, "depth.png"), depth.astype(np.uint16))
    _write_json(os.path.join(path, "camera_info.json"), cam.to_dict())
    _write_json(os.path.join(path, "T_world_optical.json"),
                {"frame": "world", "child": cam.frame_id, "T": np.asarray(T_wo).tolist()})
    _write_json(os.path.join(path, "T_world_tip.json"),
                {"frame": "world", "child": "tool_tip", "T": np.asarray(T_w_tip).tolist()})
    _write_json(os.path.join(path, "joint_state.json"), joint_state)
    if snapshot_path and os.path.isfile(snapshot_path):
        shutil.copyfile(snapshot_path, os.path.join(path, "snapshot.json"))
    if calib_path and os.path.isfile(calib_path):
        shutil.copyfile(calib_path, os.path.join(path, "calib.yaml"))
    d = meas.to_dict()
    d["arm"] = arm
    _write_json(os.path.join(path, "measurement.json"), d)
    cv2.imwrite(os.path.join(path, "overlay.png"), to_bgr(overlay_rgb))
    return path


@dataclass
class Record:
    path: str
    arm: str
    color_rgb: np.ndarray
    depth: np.ndarray
    cam: CameraModel
    T_wo: np.ndarray
    tip_world: Optional[np.ndarray]
    snapshot: Optional[Snapshot]
    measurement: Optional[dict]


class RecordError(ValueError):
    pass


def load_record(path: str, arm: Optional[str] = None, snapshot_path: Optional[str] = None) -> Record:
    def p(n):
        return os.path.join(path, n)
    try:
        bgr = cv2.imread(p("color.png"), cv2.IMREAD_COLOR)
        depth = cv2.imread(p("depth.png"), cv2.IMREAD_UNCHANGED)
        if bgr is None or depth is None:
            raise RecordError(f"{path}: color.png/depth.png unreadable")
        if depth.dtype != np.uint16 or depth.ndim != 2:
            raise RecordError(f"{path}: depth.png must be a 16-bit single channel image")
        cam = CameraModel.from_json(p("camera_info.json"))
        with open(p("T_world_optical.json")) as f:
            T_wo = np.array(json.load(f)["T"], dtype=float)
        tip = None
        if os.path.isfile(p("T_world_tip.json")):
            with open(p("T_world_tip.json")) as f:
                tip = np.array(json.load(f)["T"], dtype=float)[:3, 3]
        meas = None
        if os.path.isfile(p("measurement.json")):
            with open(p("measurement.json")) as f:
                meas = json.load(f)
        snap_file = snapshot_path or (p("snapshot.json") if os.path.isfile(p("snapshot.json")) else None)
        snap = load_snapshot(snap_file) if snap_file else None
    except (OSError, KeyError, json.JSONDecodeError, SnapshotError, ValueError) as exc:
        raise RecordError(f"{path}: {exc}") from exc
    a = arm or (meas or {}).get("arm") or os.path.basename(os.path.normpath(path)).split("_")[0]
    return Record(path, a, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), depth, cam, T_wo, tip, snap, meas)
