"""Combine the detector steps into one Measurement (delta vs the snapshot model, quality, reasons).

Delta = measured - model crossing, in the model cut basis [along, across (+ out of the carton),
normal] in mm, plus delta_yaw_deg = angle of the measured edge direction in the top plane relative
to the model along axis (positive = turning toward +across, i.e. outward, as it advances).

The measured crossing M is the point of the measured edge at the measured seam position (so a
yawed carton is compared at the seam, as the cut sees it). When the seam is not found, M is the
measured edge at the model's along position and delta_along is None.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

from .camera import CameraError, CameraModel
from .detect import (DetectParams, EdgeResult, Frame, PlaneResult, SeamResult, SidePlaneResult,
                     check_view, detect_edge, detect_seam, edge_frame, fit_side_plane, fit_top_plane,
                     lift_to_plane)
from .geom import is_valid_T, transform_points
from .model import ArmModel

MAX_PAIR_DT_S = 0.040


@dataclass
class TipVsSeam:
    along_mm: Optional[float]
    inward_from_edge_mm: float        # + = inside the carton, - = outside the edge
    below_top_mm: float               # + = below the measured top plane

    def to_dict(self) -> dict:
        return {"along_mm": _r(self.along_mm), "inward_from_edge_mm": _r(self.inward_from_edge_mm),
                "below_top_mm": _r(self.below_top_mm)}


@dataclass
class Measurement:
    arm: str
    ok: bool = False                  # top plane AND edge ok (delta across/normal/yaw valid)
    seam_ok: bool = False
    reasons: List[str] = field(default_factory=list)
    delta_mm: List[Optional[float]] = field(default_factory=lambda: [None, None, None])
    delta_yaw_deg: Optional[float] = None
    tip_vs_seam: Optional[TipVsSeam] = None
    plane: Optional[PlaneResult] = None
    edge: Optional[EdgeResult] = None
    seam: Optional[SeamResult] = None
    side: Optional[SidePlaneResult] = None
    view: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    # geometry for overlays (world frame)
    measured_crossing: Optional[np.ndarray] = None
    edge_O: Optional[np.ndarray] = None
    edge_d: Optional[np.ndarray] = None
    edge_u_out: Optional[np.ndarray] = None

    def to_dict(self) -> dict:
        return {
            "arm": self.arm, "across_convention": "outward", "ok": self.ok, "seam_ok": self.seam_ok,
            "reasons": list(self.reasons),
            "delta_mm": {"along": _r(self.delta_mm[0]), "across": _r(self.delta_mm[1]),
                         "normal": _r(self.delta_mm[2])},
            "delta_yaw_deg": _r(self.delta_yaw_deg, 3),
            "tip_vs_seam": None if self.tip_vs_seam is None else self.tip_vs_seam.to_dict(),
            "quality": {"plane": None if self.plane is None else self.plane.to_dict(),
                        "edge": None if self.edge is None else self.edge.to_dict(),
                        "seam": None if self.seam is None else self.seam.to_dict(),
                        "side_plane": None if self.side is None else self.side.to_dict()},
            "view": self.view, "meta": self.meta,
        }

    @property
    def flags(self) -> List[str]:
        return list(self.reasons)


def _r(x, nd=2):
    if x is None:
        return None
    x = float(x)
    return round(x, nd) if np.isfinite(x) else None


def check_pair(color_stamp_s: Optional[float], depth_stamp_s: Optional[float],
               max_dt_s: float = MAX_PAIR_DT_S) -> List[str]:
    """Reasons ([] = fine) for an unusable colour/depth pair by stamps."""
    if color_stamp_s is None or depth_stamp_s is None:
        return []
    if not (np.isfinite(color_stamp_s) and np.isfinite(depth_stamp_s)):
        return ["stale_pair"]
    return [] if abs(color_stamp_s - depth_stamp_s) <= max_dt_s else ["stale_pair"]


def _fail(m: Measurement, *reasons: str) -> Measurement:
    for r in reasons:
        if r not in m.reasons:
            m.reasons.append(r)
    m.ok = False
    return m


def measure(color_rgb: np.ndarray, depth: np.ndarray, cam: CameraModel, T_wo, arm: ArmModel,
            params: Optional[DetectParams] = None, tip_world=None,
            color_stamp_s: Optional[float] = None, depth_stamp_s: Optional[float] = None,
            tool_polygon_px=None, meta: Optional[dict] = None, with_side_plane: bool = True,
            tool_mask: Optional[np.ndarray] = None, tool_mask_kind: str = "static_polygon") -> Measurement:
    """Run the whole chain. Never raises on bad data; returns ok=False with reasons instead."""
    params = params or DetectParams()
    m = Measurement(arm=arm.arm, meta=dict(meta or {}))
    m.meta.setdefault("across_convention", "outward")
    m.meta["tool_mask"] = tool_mask_kind if tool_mask is not None else "static_polygon"
    try:
        return _measure(m, color_rgb, depth, cam, T_wo, arm, params, tip_world, color_stamp_s,
                        depth_stamp_s, tool_polygon_px, with_side_plane, tool_mask)
    except (CameraError, ValueError, FloatingPointError, np.linalg.LinAlgError, cv2.error) as exc:
        return _fail(m, "internal_error:%s:%s" % (type(exc).__name__, str(exc)[:120]))


def _measure(m, color_rgb, depth, cam, T_wo, arm, params, tip_world, color_stamp_s, depth_stamp_s,
             tool_polygon_px, with_side_plane, tool_mask=None):
    reasons = check_pair(color_stamp_s, depth_stamp_s)
    if reasons:
        return _fail(m, *reasons)
    if not is_valid_T(T_wo):
        return _fail(m, "non_finite_transform")
    T_wo = np.asarray(T_wo, dtype=float)
    if (not isinstance(color_rgb, np.ndarray) or not isinstance(depth, np.ndarray)
            or color_rgb.ndim != 3 or color_rgb.shape[2] != 3
            or color_rgb.shape[:2] != depth.shape or depth.shape != (cam.height, cam.width)):
        return _fail(m, "size_mismatch")
    if color_rgb.dtype != np.uint8:
        return _fail(m, "size_mismatch")
    if not np.any(depth > 0):
        return _fail(m, "no_depth")
    view_ok, view_reasons, view = check_view(cam, T_wo, arm, params)
    m.view = view
    if not view_ok:
        return _fail(m, *view_reasons)
    frame = Frame(color_rgb, depth, cam, T_wo, params, tool_polygon_px, tool_mask)
    plane = fit_top_plane(frame, arm, params)
    m.plane = plane
    if not plane.ok:
        return _fail(m, *plane.reasons)
    edge = detect_edge(frame, arm, plane, params)
    m.edge = edge
    seam = detect_seam(frame, arm, plane, edge, params)
    m.seam = seam
    if with_side_plane:
        m.side = fit_side_plane(frame, arm, plane, edge, params)
    if not edge.ok:
        return _fail(m, *edge.reasons)
    O, d, u_out, _ = edge_frame(arm, plane, edge)
    m.edge_O, m.edge_d, m.edge_u_out = O, d, u_out
    if seam.ok:
        M = O + seam.a_m * d
        m.seam_ok = True
    else:
        M = O
        m.reasons.extend(r for r in seam.reasons if r not in m.reasons)
    m.measured_crossing = M
    loc = arm.to_local(M)
    # the lifted crossing sits on the measured plane, so its h is the plane height change at M
    m.delta_mm = [(loc[0] * 1000.0) if m.seam_ok else None, loc[1] * 1000.0, loc[2] * 1000.0]
    m.delta_yaw_deg = float(np.degrees(np.arctan2(d @ arm.across, d @ arm.along)))
    if tip_world is not None:
        tip = np.asarray(tip_world, dtype=float).reshape(3)
        if np.all(np.isfinite(tip)):
            m.tip_vs_seam = tip_vs_crossing(tip, M, d, u_out, plane.n, m.seam_ok)
        else:
            m.reasons.append("non_finite_tip")
    m.ok = True
    return m


def tip_vs_crossing(tip: np.ndarray, M: np.ndarray, d: np.ndarray, u_out: np.ndarray,
                    n_plane: np.ndarray, seam_ok: bool) -> TipVsSeam:
    """Knife tip relative to the measured crossing, in the measured cut basis."""
    v = tip - M
    return TipVsSeam(float(v @ d) * 1000.0 if seam_ok else None,
                     -float(v @ u_out) * 1000.0, -float(v @ n_plane) * 1000.0)


def agree(m1: Measurement, m2: Measurement, tol_mm: float = 1.0) -> bool:
    """Two-frame agreement of the across/normal deltas (and along when both have it)."""
    if not (m1.ok and m2.ok):
        return False
    for a, b in zip(m1.delta_mm, m2.delta_mm):
        if a is None or b is None:
            continue
        if abs(a - b) > tol_mm:
            return False
    return True
