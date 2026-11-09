"""Overlay drawing on the colour image (downscaled to 960x600).

red    = snapshot model (cut-side edge, crossing, top-face outline)
green  = measured (edge, crossing, plane inlier hull)
yellow = knife tip
All arrays are RGB uint8; use `to_bgr` before cv2.imwrite.
"""
from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

from .camera import CameraModel
from .geom import inv_T, transform_points
from .measure import Measurement
from .model import ArmModel

OUT_W, OUT_H = 960, 600
RED = (230, 40, 40)
GREEN = (40, 220, 60)
YELLOW = (250, 220, 20)
WHITE = (245, 245, 245)
ORANGE = (255, 150, 20)


def to_bgr(img_rgb: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(img_rgb[..., ::-1])


def encode_jpeg(img_rgb: np.ndarray, quality: int = 80) -> bytes:
    ok, buf = cv2.imencode(".jpg", to_bgr(img_rgb), [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise ValueError("jpeg encode failed")
    return buf.tobytes()


def _project(cam: CameraModel, T_ow: np.ndarray, P_w: np.ndarray, scale: float) -> np.ndarray:
    return cam.project(transform_points(T_ow, np.asarray(P_w, dtype=float).reshape(-1, 3))) * scale


def _polyline(img, uv, color, thick=1):
    """Draw segments between consecutive finite points that lie within a sane pixel range."""
    for p, q in zip(uv[:-1], uv[1:]):
        if np.all(np.isfinite(p)) and np.all(np.isfinite(q)) and np.abs(p).max() < 1e5 and np.abs(q).max() < 1e5:
            cv2.line(img, tuple(np.rint(p).astype(int)), tuple(np.rint(q).astype(int)), color, thick,
                     cv2.LINE_AA)


def _densify(P: np.ndarray, per_seg: int = 24, closed: bool = False) -> np.ndarray:
    pts = []
    n = len(P)
    for i in range(n if closed else n - 1):
        p, q = P[i], P[(i + 1) % n]
        for t in np.linspace(0, 1, per_seg, endpoint=False):
            pts.append(p + t * (q - p))
    pts.append(P[0] if closed else P[-1])
    return np.array(pts)


def _marker(img, p, color, kind="circle", r=7):
    if p is None or not np.all(np.isfinite(p)) or np.abs(p).max() > 1e5:
        return
    c = tuple(np.rint(p).astype(int))
    if kind == "circle":
        cv2.circle(img, c, r, color, 2, cv2.LINE_AA)
    else:
        cv2.line(img, (c[0] - r, c[1]), (c[0] + r, c[1]), color, 2, cv2.LINE_AA)
        cv2.line(img, (c[0], c[1] - r), (c[0], c[1] + r), color, 2, cv2.LINE_AA)


def _text_block(img, lines: List[str], x=8, y=8, size=0.46):
    h = 17
    w = max(cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, size, 1)[0][0] for s in lines) + 10
    sub = img[y:y + h * len(lines) + 6, x:x + w]
    sub[:] = (sub * 0.35).astype(np.uint8)
    for i, s in enumerate(lines):
        cv2.putText(img, s, (x + 5, y + 14 + i * h), cv2.FONT_HERSHEY_SIMPLEX, size, WHITE, 1, cv2.LINE_AA)


def _banner(img, text, row=0):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)
    x = (img.shape[1] - tw) // 2
    y = img.shape[0] // 2 + 10 + row * 44
    cv2.rectangle(img, (x - 10, y - th - 8), (x + tw + 10, y + 10), (20, 20, 20), -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.9, ORANGE, 2, cv2.LINE_AA)


def _f(x, nd=1, sign=True):
    if x is None:
        return "  n/a"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def draw_overlay(color_rgb: np.ndarray, cam: CameraModel, T_wo, arm: ArmModel,
                 meas: Optional[Measurement] = None, tip_world=None, state: str = "LIVE",
                 ros_time_s: Optional[float] = None, calib_status: str = "",
                 snapshot_id: str = "", extra_banners: Optional[List[str]] = None) -> np.ndarray:
    """Return the 960x600 RGB overlay. Works with meas=None (model only) and on any failure."""
    img = cv2.resize(color_rgb, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA).copy()
    scale = OUT_W / float(cam.width)
    banners = list(extra_banners or [])
    try:
        T_ow = inv_T(np.asarray(T_wo, dtype=float))
        # --- model (red)
        a_lo, a_hi = arm.edge_extent_a()
        a = np.linspace(a_lo, a_hi, 60)
        edge = arm.from_local(np.stack([a, 0 * a, 0 * a], axis=1))
        _polyline(img, _project(cam, T_ow, edge, scale), RED, 2)
        outline = _densify(arm.corners, closed=True)
        _polyline(img, _project(cam, T_ow, outline, scale), RED, 1)
        cross = arm.from_local(np.array([[0, -0.04, 0], [0, 0.0, 0]]))
        _polyline(img, _project(cam, T_ow, _densify(cross), scale), RED, 1)
        _marker(img, _project(cam, T_ow, arm.S, scale)[0], RED, "circle")
        # --- measured (green)
        if meas is not None and meas.plane is not None and meas.plane.inlier_uv is not None \
                and len(meas.plane.inlier_uv) >= 3:
            hull = cv2.convexHull((meas.plane.inlier_uv * scale).astype(np.float32))
            cv2.polylines(img, [np.rint(hull).astype(np.int32)], True, GREEN, 1, cv2.LINE_AA)
        if meas is not None and meas.edge_O is not None and meas.edge is not None and meas.edge.ok:
            a2 = np.linspace(a_lo, a_hi, 60)
            Pm = meas.edge_O[None, :] + a2[:, None] * meas.edge_d[None, :]
            _polyline(img, _project(cam, T_ow, Pm, scale), GREEN, 2)
            if meas.measured_crossing is not None:
                M = meas.measured_crossing
                seg = np.array([M - 0.04 * meas.edge_u_out, M + 0.0 * meas.edge_u_out])
                _polyline(img, _project(cam, T_ow, _densify(seg), scale), GREEN, 1)
                _marker(img, _project(cam, T_ow, M, scale)[0], GREEN, "circle" if meas.seam_ok else "cross")
        # --- tip (yellow)
        if tip_world is not None and np.all(np.isfinite(tip_world)):
            _marker(img, _project(cam, T_ow, np.asarray(tip_world), scale)[0], YELLOW, "cross", 9)
    except (ValueError, np.linalg.LinAlgError, cv2.error):
        banners.append("OVERLAY ERROR")

    lines = [f"{arm.arm.upper()}  {state}   snap {snapshot_id[-26:]}"]
    if meas is not None and meas.ok:
        da, dc, dn = meas.delta_mm
        lines.append(f"delta along {_f(da)} across {_f(dc)} normal {_f(dn)} mm  yaw {_f(meas.delta_yaw_deg, 2)} deg")
        if meas.tip_vs_seam is not None:
            t = meas.tip_vs_seam
            lines.append(f"tip vs seam: along {_f(t.along_mm)} inward {_f(t.inward_from_edge_mm)} "
                         f"below top {_f(t.below_top_mm)} mm")
        q = []
        if meas.plane:
            q.append(f"plane {meas.plane.inliers}/{meas.plane.candidates} rms {meas.plane.rms_m * 1000:.2f}mm")
        if meas.edge:
            q.append(f"edge {meas.edge.n_inliers}pts rms {meas.edge.rms_m * 1000:.2f}mm")
        if meas.seam:
            q.append(f"seam conf {meas.seam.confidence:.1f}" + ("" if meas.seam.ok else " (not found)"))
        lines.append(" | ".join(q))
    elif meas is not None:
        lines.append("no measurement: " + ", ".join(meas.reasons[:3]))
    if meas is not None and meas.reasons and meas.ok:
        lines.append("flags: " + ", ".join(meas.reasons))
    if calib_status:
        lines.append(f"calib: {calib_status}")
    if ros_time_s is not None:
        lines.append(f"ros t={ros_time_s:.3f}")
    _text_block(img, lines)

    if meas is not None:
        rs = set(meas.reasons)
        if rs & {"edge_not_in_view", "carton_too_close", "carton_too_far"}:
            banners.append("NO CARTON IN VIEW")
        if "no_depth" in rs:
            banners.append("NO DEPTH")
        if rs & {"stale_pair", "non_finite_transform", "size_mismatch"}:
            banners.append("BAD INPUT: " + ",".join(sorted(rs & {"stale_pair", "non_finite_transform", "size_mismatch"})))
    if "SEED" in calib_status.upper():
        banners.append(f"CALIB SEED ({arm.arm})")
    for i, b in enumerate(banners[:3]):
        _banner(img, b, i)
    return img
