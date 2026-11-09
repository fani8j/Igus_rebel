"""Snapshot parsing and the per-arm carton model: top plane, cut-side edge, crossing, cut basis.

Everything is in the snapshot frame, which is the dual-arm `world` (= the left arm's base_link):
the left arm cuts the carton's left face (edge back_left - front_left, crossing seam_start) and the
right arm the right face (edge back_right - front_right, crossing seam_end), using the physical
labels (no rotate_180 handling).

Cut basis at the crossing S, as in the executor's cutDebugBasis():
  along  = edge direction back -> front, projected into the top plane,
  normal = snapshot surface_normal (up),
  across = normal x along, then sign-fixed so that +across points OUT of the carton.
(The executor's raw across = normal x along points INTO the carton for the left face, where
 along = -x and normal = +z give across = -y; the probe flips it so that +across is outward for
 both arms. Delta values reported here use the outward convention.)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .geom import unit

ARM_FACE = {
    "left": ("back_left", "front_left", "seam_start"),
    "right": ("back_right", "front_right", "seam_end"),
}
CORNERS = ("back_left", "back_right", "front_right", "front_left")


class SnapshotError(ValueError):
    pass


@dataclass
class Snapshot:
    snapshot_id: str
    frame_id: str
    stamp_ns: Optional[int]
    points: Dict[str, np.ndarray]
    normal: np.ndarray

    @property
    def age_s_hint(self) -> Optional[float]:
        return None if self.stamp_ns is None else self.stamp_ns * 1e-9


def _vec(d, key) -> np.ndarray:
    v = d.get(key)
    if not isinstance(v, dict):
        raise SnapshotError(f"snapshot is missing '{key}'")
    try:
        a = np.array([float(v["x"]), float(v["y"]), float(v["z"])])
    except (KeyError, TypeError, ValueError) as exc:
        raise SnapshotError(f"snapshot '{key}' is not an x/y/z point") from exc
    if not np.all(np.isfinite(a)):
        raise SnapshotError(f"snapshot '{key}' is not finite")
    return a


def parse_snapshot(d: dict) -> Snapshot:
    if not isinstance(d, dict):
        raise SnapshotError("snapshot is not a JSON object")
    if d.get("schema") != "box_h_cut_snapshot/v1":
        raise SnapshotError(f"unexpected snapshot schema {d.get('schema')!r}")
    pts = {}
    for k in set(CORNERS) | {"seam_start", "seam_end"}:
        pts[k] = _vec(d, k)
    n = _vec(d, "surface_normal")
    if np.linalg.norm(n) < 1e-9:
        raise SnapshotError("snapshot surface_normal has zero length")
    stamp = d.get("acquisition_stamp_ns")
    return Snapshot(str(d.get("snapshot_id", "")), str(d.get("frame_id", "")),
                    int(stamp) if isinstance(stamp, (int, float)) else None, pts, unit(n))


def load_snapshot(path: str) -> Snapshot:
    try:
        with open(path) as f:
            return parse_snapshot(json.load(f))
    except (OSError, json.JSONDecodeError) as exc:
        raise SnapshotError(f"cannot read snapshot {path}: {exc}") from exc


@dataclass
class ArmModel:
    arm: str
    snapshot_id: str
    S: np.ndarray               # model crossing: seam point projected onto the model edge line
    along: np.ndarray           # unit, back -> front, in the top plane
    across: np.ndarray          # unit, in the top plane, +across = out of the carton
    normal: np.ndarray          # unit, up
    corners: np.ndarray         # (4,3) top-face corners (back_left, back_right, front_right, front_left)
    edge_back: np.ndarray
    edge_front: np.ndarray
    centre: np.ndarray
    crossing_to_edge_mm: float  # snapshot crossing -> edge line distance (diagnostic)
    across_flipped: bool        # True when the executor's raw n x along points INTO the carton (flipped here)

    @property
    def basis(self) -> np.ndarray:
        """Rows (along, across, normal)."""
        return np.vstack([self.along, self.across, self.normal])

    def to_local(self, P: np.ndarray) -> np.ndarray:
        """World points (...,3) -> (a, c, h) relative to S."""
        return (np.asarray(P) - self.S) @ self.basis.T

    def from_local(self, L: np.ndarray) -> np.ndarray:
        return self.S + np.asarray(L) @ self.basis

    def polygon_local(self) -> np.ndarray:
        """(4,2) top-face corners in (a, c)."""
        return self.to_local(self.corners)[:, :2]

    def edge_extent_a(self) -> Tuple[float, float]:
        a = self.to_local(np.vstack([self.edge_back, self.edge_front]))[:, 0]
        return float(a.min()), float(a.max())

    def to_dict(self) -> dict:
        return {"arm": self.arm, "snapshot_id": self.snapshot_id, "S": self.S.tolist(),
                "along": self.along.tolist(), "across": self.across.tolist(),
                "normal": self.normal.tolist(), "corners": self.corners.tolist(),
                "crossing_to_edge_mm": round(self.crossing_to_edge_mm, 3),
                "across_flipped_vs_executor_raw": self.across_flipped}


def build_arm_model(snap: Snapshot, arm: str) -> ArmModel:
    if arm not in ARM_FACE:
        raise SnapshotError(f"unknown arm {arm!r} (left/right)")
    back_k, front_k, cross_k = ARM_FACE[arm]
    back, front, S0 = snap.points[back_k], snap.points[front_k], snap.points[cross_k]
    n = snap.normal
    axis = front - back
    axis = axis - (axis @ n) * n
    if np.linalg.norm(axis) < 1e-3:
        raise SnapshotError(f"degenerate {arm} edge")
    along = unit(axis)
    raw = np.cross(n, along)
    corners = np.vstack([snap.points[k] for k in CORNERS])
    centre = corners.mean(axis=0)
    # edge line through back/front; project the snapshot crossing onto it
    e = unit(front - back)
    S = back + ((S0 - back) @ e) * e
    outward_sign = -1.0 if (centre - S) @ raw > 0 else 1.0
    across = unit(outward_sign * raw)
    return ArmModel(arm, snap.snapshot_id, S, along, across, n, corners, back, front, centre,
                    float(np.linalg.norm(S0 - S) * 1000.0), outward_sign < 0)
