"""Hand-eye calibration (link6 -> colour optical frame) and the static tool-occlusion polygon."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional

import numpy as np
import yaml

from .geom import make_T, quat_xyzw_to_matrix

# The knife body enters from the right of the 1280x800 colour image and occludes about 25 % of
# it (S0 frames of 2026-10-20, left and right 305 alike: the holder is the same). Pixel polygon
# of the blade/handpiece and the housing behind it. A small margin is included on purpose.
DEFAULT_TOOL_POLYGON_PX = [
    (905, 305), (1135, 285), (1135, 150), (1280, 110),
    (1280, 730), (1135, 730), (1135, 530), (905, 490),
]

CALIB_OK = "left_handeye_20261013"
CALIB_SEED = "seed_from_left_UNCALIBRATED"


class CalibError(ValueError):
    pass


@dataclass
class HandEye:
    T_link6_optical: np.ndarray      # 4x4, parent link6 -> child colour optical frame
    parent: str
    child: str
    sha256: str
    path: str
    calib_status: str
    tool_polygon_px: list

    def to_meta(self) -> dict:
        return {"path": self.path, "sha256": self.sha256, "status": self.calib_status,
                "parent": self.parent, "child": self.child}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_hand_eye(path: str, seed_from_left: bool = False) -> HandEye:
    """Load a wrist_handeye/v1 result.yaml. seed_from_left flags the right arm's borrowed copy."""
    try:
        with open(path) as f:
            d = yaml.safe_load(f)
    except OSError as exc:
        raise CalibError(f"cannot read hand-eye file {path}: {exc}") from exc
    if not isinstance(d, dict) or d.get("schema") != "wrist_handeye/v1":
        raise CalibError(f"{path} is not a wrist_handeye/v1 result")
    try:
        xyz = np.asarray(d["xyz"], dtype=float).reshape(3)
        quat = np.asarray(d["quat_xyzw"], dtype=float).reshape(4)
        R = quat_xyzw_to_matrix(quat)
    except (KeyError, ValueError, TypeError) as exc:
        raise CalibError(f"{path}: bad xyz/quat_xyzw: {exc}") from exc
    if not np.all(np.isfinite(xyz)) or not np.all(np.isfinite(quat)):
        raise CalibError(f"{path}: non-finite hand-eye values")
    poly = d.get("tool_polygon_px") or DEFAULT_TOOL_POLYGON_PX
    return HandEye(make_T(R, xyz), str(d.get("parent", "link6")),
                   str(d.get("child", d.get("camera_frame_id", ""))), sha256_file(path), path,
                   CALIB_SEED if seed_from_left else CALIB_OK, [tuple(p) for p in poly])
