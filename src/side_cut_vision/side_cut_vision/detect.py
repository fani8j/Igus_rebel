"""Model-based detectors: top plane, cut-side edge, seam crossing, optional side-face plane.

Every step returns a dataclass with `ok` and `reasons` and never raises on bad image data. All
lengths are metres. Coordinates ("local") are (a, c, h) relative to the model crossing S in the cut
basis of `ArmModel`: a = along, c = across (+ out of the carton), h = normal (up).

Design (see STAGE_A_SPEC.md): the left 305 looks down at the top face near the cut-side edge, so
the *top* plane (height), the cut-side *edge* (depth and yaw) and the *seam* (flap gap: a dark line
perpendicular to the edge, under glossy clear tape) are measured. The edge is the last pixel still
on the fitted top plane before the depth falls to the floor, refined with the colour gradient.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

from .calib import DEFAULT_TOOL_POLYGON_PX
from .camera import CameraModel
from .geom import (angle_deg, inv_T, ray_plane, ransac_plane, robust_line_2d, transform_points,
                   unit)
from .model import ArmModel


@dataclass
class DetectParams:
    fit_stride: int = 2
    seed: int = 12345
    # view check
    min_range_m: float = 0.07
    max_range_m: float = 0.60
    min_edge_visible_m: float = 0.06
    # top plane
    top_inward_min_m: float = 0.008
    top_inward_max_m: float = 0.120
    top_h_window_m: float = 0.040
    tool_closer_m: float = 0.040
    plane_ransac_thr_m: float = 0.002
    plane_ransac_iters: int = 200
    plane_min_inliers: int = 1500
    plane_min_inlier_frac: float = 0.60
    plane_max_rms_m: float = 0.002
    plane_max_normal_dev_deg: float = 10.0
    # edge
    edge_line_step_m: float = 0.005
    edge_half_window_m: float = 0.040
    edge_sample_step_m: float = 0.0005
    edge_on_plane_m: float = 0.004
    edge_leave_m: float = 0.015
    edge_max_gap_m: float = 0.012
    edge_pick_m: float = 0.002
    edge_min_on_plane_m: float = 0.004
    edge_grad_refine_px: int = 3
    edge_grad_min: float = 6.0
    edge_min_points: int = 15
    edge_min_span_m: float = 0.060
    edge_min_inlier_frac: float = 0.60
    edge_max_rms_m: float = 0.0015
    edge_max_angle_deg: float = 10.0
    edge_line_thr_m: float = 0.0015
    # seam
    seam_half_along_m: float = 0.060
    seam_inward_min_m: float = 0.008
    seam_inward_max_m: float = 0.080
    seam_mm_per_px: float = 0.5
    seam_blackhat_px: int = 15
    seam_conf_min: float = 1.5
    seam_min_peak: float = 10.0
    seam_min_row_cover: float = 0.4
    # side plane (log only)
    side_band_m: float = 0.060
    side_min_points: int = 1500
    side_inlier_thr_m: float = 0.002


# ------------------------------------------------------------------------------ results
@dataclass
class PlaneResult:
    ok: bool
    reasons: List[str] = field(default_factory=list)
    n: Optional[np.ndarray] = None
    d: float = float("nan")
    inliers: int = 0
    candidates: int = 0
    inlier_frac: float = 0.0
    rms_m: float = float("nan")
    normal_dev_deg: float = float("nan")
    inlier_uv: Optional[np.ndarray] = None    # (<=2000, 2) pixels of inliers (full-res coords)

    def to_dict(self) -> dict:
        return {"ok": self.ok, "reasons": list(self.reasons), "inliers": self.inliers,
                "candidates": self.candidates, "inlier_frac": _r(self.inlier_frac, 3),
                "rms_mm": _r(self.rms_m * 1000, 3), "normal_dev_deg": _r(self.normal_dev_deg, 3)}


@dataclass
class EdgeResult:
    ok: bool
    reasons: List[str] = field(default_factory=list)
    c0: float = float("nan")            # measured edge c at a = 0 (m)
    slope: float = float("nan")         # dc/da
    n_lines: int = 0
    n_points: int = 0
    n_inliers: int = 0
    span_m: float = 0.0
    rms_m: float = float("nan")
    angle_deg: float = float("nan")
    pts_local: Optional[np.ndarray] = None   # (n,2) (a, c)
    inlier_mask: Optional[np.ndarray] = None

    @property
    def yaw_deg(self) -> float:
        return float(np.degrees(np.arctan(self.slope)))

    def to_dict(self) -> dict:
        return {"ok": self.ok, "reasons": list(self.reasons), "lines": self.n_lines,
                "points": self.n_points, "inliers": self.n_inliers,
                "span_mm": _r(self.span_m * 1000, 1), "rms_mm": _r(self.rms_m * 1000, 3),
                "angle_deg": _r(self.angle_deg, 3), "c0_mm": _r(self.c0 * 1000, 3),
                "yaw_deg": _r(self.yaw_deg, 3) if np.isfinite(self.slope) else None}


@dataclass
class SeamResult:
    ok: bool
    reasons: List[str] = field(default_factory=list)
    a_m: float = float("nan")           # seam position along the edge frame (m, 0 = model crossing)
    confidence: float = 0.0
    peak: float = 0.0
    frame: str = "measured"             # edge frame used for the ortho patch: measured | model
    profile: Optional[np.ndarray] = None

    def to_dict(self) -> dict:
        return {"ok": self.ok, "reasons": list(self.reasons), "a_mm": _r(self.a_m * 1000, 3),
                "confidence": _r(self.confidence, 3), "peak": _r(self.peak, 3), "frame": self.frame}


@dataclass
class SidePlaneResult:
    ok: bool
    reasons: List[str] = field(default_factory=list)
    n: Optional[np.ndarray] = None
    d: float = float("nan")
    inliers: int = 0
    rms_m: float = float("nan")

    def to_dict(self) -> dict:
        return {"ok": self.ok, "reasons": list(self.reasons), "inliers": self.inliers,
                "rms_mm": _r(self.rms_m * 1000, 3),
                "normal": None if self.n is None else [round(float(x), 5) for x in self.n]}


def _r(x, nd):
    x = float(x)
    return round(x, nd) if np.isfinite(x) else None


# ------------------------------------------------------------------------------ frame
def tool_mask_full(width: int, height: int, polygon_px=None) -> np.ndarray:
    """Static tool region (pixel polygon given for 1280x800) rasterised at the image size."""
    poly = np.array(polygon_px if polygon_px is not None else DEFAULT_TOOL_POLYGON_PX, dtype=np.float64)
    poly = poly * np.array([width / 1280.0, height / 800.0])
    m = np.zeros((height, width), np.uint8)
    cv2.fillPoly(m, [np.rint(poly).astype(np.int32)], 1)
    return m.astype(bool)


class Frame:
    """One colour+depth pair prepared for the detectors (world-frame points on a stride grid)."""

    def __init__(self, color_rgb: np.ndarray, depth: np.ndarray, cam: CameraModel, T_wo: np.ndarray,
                 params: DetectParams, tool_polygon_px=None, tool_mask: Optional[np.ndarray] = None):
        self.cam = cam
        self.params = params
        self.T_wo = np.asarray(T_wo, dtype=float)
        self.T_ow = inv_T(self.T_wo)
        self.origin = self.T_wo[:3, 3].copy()
        self.R_wo = self.T_wo[:3, :3]
        self.depth = depth
        self.color = color_rgb
        self.gray = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2GRAY)
        self.stride = int(params.fit_stride)
        xyz, valid = cam.deproject(depth, self.stride, min_z=params.min_range_m * 0.5, max_z=3.0)
        self.valid = valid
        pts = xyz.reshape(-1, 3).astype(np.float64)
        pw = np.full_like(pts, np.nan)
        v = valid.reshape(-1)
        pw[v] = transform_points(self.T_wo, pts[v])
        self.P_w = pw.reshape(xyz.shape)
        if tool_mask is not None and tool_mask.shape == (cam.height, cam.width):
            self.tool_full = tool_mask.astype(bool)
        else:
            self.tool_full = tool_mask_full(cam.width, cam.height, tool_polygon_px)
        self.tool = self.tool_full[::self.stride, ::self.stride]

    def pixel_dirs_world(self, uv: np.ndarray) -> np.ndarray:
        xy = self.cam.pixel_rays(uv)
        d = np.concatenate([xy, np.ones((len(xy), 1))], axis=1)
        return d @ self.R_wo.T

    def tool_at(self, uv: np.ndarray) -> np.ndarray:
        ui = np.clip(np.rint(uv[:, 0]), 0, self.cam.width - 1).astype(int)
        vi = np.clip(np.rint(uv[:, 1]), 0, self.cam.height - 1).astype(int)
        return self.tool_full[vi, ui]


def _convex_inside(poly: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Points (N,2) inside a convex polygon (M,2) of either orientation."""
    area = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        area += x1 * y2 - x2 * y1
    sgn = 1.0 if area > 0 else -1.0
    ok = np.ones(len(pts), bool)
    for i in range(len(poly)):
        p, q = poly[i], poly[(i + 1) % len(poly)]
        cr = (q[0] - p[0]) * (pts[:, 1] - p[1]) - (q[1] - p[1]) * (pts[:, 0] - p[0])
        ok &= sgn * cr >= 0
    return ok


# ------------------------------------------------------------------------------ view check
def check_view(cam: CameraModel, T_wo: np.ndarray, arm: ArmModel, params: DetectParams):
    """Is the model cut-side edge in the image and 70..600 mm away? -> (ok, reasons, info)."""
    a_lo, a_hi = arm.edge_extent_a()
    a = np.arange(a_lo, a_hi, 0.005)
    P = arm.from_local(np.stack([a, np.zeros_like(a), np.zeros_like(a)], axis=1))
    Pc = transform_points(inv_T(T_wo), P)
    uv = cam.project(Pc)
    z = Pc[:, 2]
    vis = cam.in_image(uv, margin=2) & (z >= params.min_range_m) & (z <= params.max_range_m)
    infront = z > 0.01
    info = {"edge_samples": int(len(a)), "visible_samples": int(vis.sum()),
            "edge_range_m": [float(np.nanmin(z)), float(np.nanmax(z))] if len(z) else None}
    visible_len = float(vis.sum()) * 0.005
    reasons = []
    if not infront.any() or not cam.in_image(uv, margin=2).any():
        reasons.append("edge_not_in_view")
    elif visible_len < params.min_edge_visible_m:
        inimg = cam.in_image(uv, margin=2) & infront
        if inimg.any() and np.all(z[inimg] < params.min_range_m):
            reasons.append("carton_too_close")
        elif inimg.any() and np.all(z[inimg] > params.max_range_m):
            reasons.append("carton_too_far")
        else:
            reasons.append("edge_not_in_view")
    info["visible_edge_mm"] = round(visible_len * 1000, 1)
    return (not reasons), reasons, info


# ------------------------------------------------------------------------------ top plane
def fit_top_plane(frame: Frame, arm: ArmModel, params: DetectParams) -> PlaneResult:
    res = PlaneResult(ok=False)
    L = arm.to_local(frame.P_w)                      # (H',W',3), NaN where invalid
    a, c, h = L[..., 0], L[..., 1], L[..., 2]
    with np.errstate(invalid="ignore"):
        sel = (frame.valid & ~frame.tool & (h <= params.tool_closer_m)
               & (np.abs(h) <= params.top_h_window_m)
               & (c <= -params.top_inward_min_m) & (c >= -params.top_inward_max_m))
    idx = np.nonzero(sel)
    if len(idx[0]):
        pts_local = np.stack([a[idx], c[idx]], axis=1)
        inside = _convex_inside(arm.polygon_local(), pts_local)
        idx = (idx[0][inside], idx[1][inside])
    n_cand = len(idx[0])
    res.candidates = n_cand
    if n_cand < 3:
        res.reasons.append("top_plane_insufficient_points")
        return res
    P = frame.P_w[idx]
    rng = np.random.default_rng(params.seed)
    out = ransac_plane(P, arm.normal, params.plane_ransac_thr_m, rng, params.plane_ransac_iters)
    if out is None:
        res.reasons.append("top_plane_insufficient_points")
        return res
    n, d, mask = out
    res.n, res.d = n, d
    res.inliers = int(mask.sum())
    res.inlier_frac = res.inliers / n_cand
    res.rms_m = float(np.sqrt(np.mean((P[mask] @ n - d) ** 2)))
    res.normal_dev_deg = angle_deg(n, arm.normal)
    sub = np.nonzero(mask)[0]
    if len(sub) > 2000:
        sub = np.random.default_rng(1).choice(sub, 2000, replace=False)
    res.inlier_uv = np.stack([idx[1][sub] * frame.stride, idx[0][sub] * frame.stride], axis=1).astype(float)
    if res.inliers < params.plane_min_inliers:
        res.reasons.append("top_plane_insufficient_points")
    if res.inlier_frac < params.plane_min_inlier_frac:
        res.reasons.append("top_plane_low_inlier_fraction")
    if not (res.rms_m <= params.plane_max_rms_m):
        res.reasons.append("top_plane_rms_high")
    if not (res.normal_dev_deg <= params.plane_max_normal_dev_deg):
        res.reasons.append("top_plane_normal_mismatch")
    res.ok = not res.reasons
    return res


def lift_to_plane(arm: ArmModel, plane: PlaneResult, a, c) -> np.ndarray:
    """World points on the measured plane with model-local (a, c) (lifted along the model normal)."""
    a = np.asarray(a, dtype=float)
    c = np.asarray(c, dtype=float)
    Q = arm.S + a[..., None] * arm.along + c[..., None] * arm.across
    k = (plane.d - Q @ plane.n) / (plane.n @ arm.normal)
    return Q + k[..., None] * arm.normal


# ------------------------------------------------------------------------------ edge
def detect_edge(frame: Frame, arm: ArmModel, plane: PlaneResult, params: DetectParams) -> EdgeResult:
    res = EdgeResult(ok=False)
    if plane.n is None:
        res.reasons.append("edge_needs_top_plane")
        return res
    cam = frame.cam
    a_lo, a_hi = arm.edge_extent_a()
    a_vals = np.arange(a_lo + 0.010, a_hi - 0.010, params.edge_line_step_m)
    c_vals = np.arange(-params.edge_half_window_m, params.edge_half_window_m + 1e-9,
                       params.edge_sample_step_m)
    K, M = len(a_vals), len(c_vals)
    A = np.repeat(a_vals[:, None], M, axis=1)
    C = np.repeat(c_vals[None, :], K, axis=0)
    Q = lift_to_plane(arm, plane, A.reshape(-1), C.reshape(-1))
    Qc = transform_points(frame.T_ow, Q)
    uv = cam.project(Qc)
    inimg = cam.in_image(uv, margin=2) & (Qc[:, 2] >= params.min_range_m * 0.5)
    dist_all = np.full(K * M, np.nan)                 # signed distance to the measured plane (+ = above)
    ui = np.where(inimg)[0]
    if len(ui):
        uvi = uv[ui]
        z = cam.sample_depth(frame.depth, uvi)
        zok = np.isfinite(z) & ~frame.tool_at(uvi)
        sel = ui[zok]
        if len(sel):
            pc = cam.points_from_pixels(uv[sel], z[zok])
            dist_all[sel] = transform_points(frame.T_wo, pc) @ plane.n - plane.d
    dist = dist_all.reshape(K, M)
    uv3 = uv.reshape(K, M, 2)
    gap_n = int(round(params.edge_max_gap_m / params.edge_sample_step_m))
    on_need = int(round(params.edge_min_on_plane_m / params.edge_sample_step_m))
    look_back = 4 * on_need
    cand = []                                          # (line index, sample index)
    with np.errstate(invalid="ignore"):
        far_all = dist < -params.edge_leave_m
        on_all = (np.abs(dist) <= params.edge_on_plane_m)
        # last-sample criterion: still within `edge_pick_m` of the plane. A vertical side face seen
        # obliquely lies within edge_on_plane_m of the plane for a few mm; picking by the tighter
        # edge_pick_m keeps that bias below ~1 mm.
        pick_all = (dist >= -params.edge_pick_m) & (dist <= params.edge_on_plane_m)
    for k in range(K):
        far = np.nonzero(far_all[k])[0]
        if not len(far):
            continue
        j = int(far[0])
        lo = max(0, j - look_back)
        on = np.nonzero(on_all[k, lo:j])[0]
        pk = np.nonzero(pick_all[k, lo:j])[0]
        if len(on) < on_need or not len(pk):
            continue
        i_e = lo + int(pk[-1])
        if j - i_e <= gap_n:
            cand.append((k, _ramp_crossing(dist[k], c_vals, i_e, j, gap_n, params)))
    res.n_lines = K
    pts = []
    g = params.edge_grad_refine_px
    if cand:
        ks = np.array([c[0] for c in cand])
        pos = np.array([c[1] for c in cand], dtype=float)
        ies = np.clip(np.floor(pos).astype(int), 0, M - 2)
        fr_ = (pos - ies)[:, None]
        uv_e = uv3[ks, ies] * (1 - fr_) + uv3[ks, ies + 1] * fr_
        ia = np.clip(ies - 2, 0, M - 1)
        ib = np.clip(ies + 3, 0, M - 1)
        tdir = uv3[ks, ib] - uv3[ks, ia]
        tn = np.linalg.norm(tdir, axis=1)
        good = np.isfinite(tn) & (tn > 1e-6) & np.isfinite(uv_e).all(axis=1)
        ks, ies, uv_e, tdir, tn = ks[good], ies[good], uv_e[good], tdir[good], tn[good]
        tdir = tdir / tn[:, None]
        offs = np.arange(-g, g + 1, dtype=np.float32)
        mx = (uv_e[:, 0:1] + tdir[:, 0:1] * offs[None, :]).astype(np.float32)
        my = (uv_e[:, 1:2] + tdir[:, 1:2] * offs[None, :]).astype(np.float32)
        prof = cv2.remap(cv2.GaussianBlur(frame.gray, (0, 0), 0.8), mx, my, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE).astype(np.float64)
        grad = np.abs(prof[:, 2:] - prof[:, :-2])      # positions offs[1:-1]
        best = np.argmax(grad, axis=1)
        gmax = grad[np.arange(len(best)), best]
        shift = np.where(gmax >= params.edge_grad_min, offs[1:-1][best], 0.0)
        uv_ref = uv_e + tdir * shift[:, None]
        dirs = frame.pixel_dirs_world(uv_ref)
        Pw = ray_plane(frame.origin, dirs, plane.n, plane.d)
        ok = np.isfinite(Pw).all(axis=1)
        loc = arm.to_local(Pw[ok])
        pts = loc[:, :2]
    pts = np.asarray(pts).reshape(-1, 2)
    res.n_points = len(pts)
    if len(pts) < params.edge_min_points:
        res.reasons.append("edge_insufficient_points")
        return res
    rng = np.random.default_rng(params.seed + 1)
    fit = robust_line_2d(pts[:, 0], pts[:, 1], params.edge_line_thr_m, rng)
    if fit is None:
        res.reasons.append("edge_insufficient_points")
        return res
    c0, m, mask, rms = fit
    res.c0, res.slope, res.rms_m = c0, m, rms
    res.pts_local, res.inlier_mask = pts, mask
    res.n_inliers = int(mask.sum())
    res.span_m = float(np.ptp(pts[mask, 0])) if res.n_inliers else 0.0
    res.angle_deg = float(np.degrees(np.abs(np.arctan(m))))
    if res.span_m < params.edge_min_span_m:
        res.reasons.append("edge_span_short")
    if res.n_inliers / len(pts) < params.edge_min_inlier_frac:
        res.reasons.append("edge_low_inlier_fraction")
    if not (rms <= params.edge_max_rms_m):
        res.reasons.append("edge_rms_high")
    if not (res.angle_deg <= params.edge_max_angle_deg):
        res.reasons.append("edge_angle_mismatch")
    res.ok = not res.reasons
    return res


def _ramp_crossing(dist_line: np.ndarray, c_vals: np.ndarray, i_e: int, j: int, gap_n: int,
                   params: DetectParams) -> float:
    """Fractional sample index of the edge on one search line.

    Seen obliquely, the side face just below the edge is nearly vertical but its pixels lie only a
    few mm below the plane for a few mm of plane-projected travel (dist falls linearly with c after
    the edge). Extrapolating that ramp back to dist = 0 removes the bias a fixed threshold has
    (about threshold * D/H, up to ~2 mm at 35 deg tilt). Falls back to the last on-plane sample.
    """
    M = len(dist_line)
    hi = min(M, j + gap_n)
    idx = np.arange(i_e, hi)
    d = dist_line[idx]
    with np.errstate(invalid="ignore"):
        sel = np.isfinite(d) & (d <= -params.edge_pick_m * 1.25) & (d >= -0.012)
    if sel.sum() < 4:
        return float(i_e)
    x = c_vals[idx[sel]]
    y = d[sel]
    xm, ym = x.mean(), y.mean()
    sxx = ((x - xm) ** 2).sum()
    if sxx < 1e-12:
        return float(i_e)
    b = ((x - xm) * (y - ym)).sum() / sxx
    if b >= 0:
        return float(i_e)
    c_e = xm - ym / b                      # where the fitted ramp reaches dist = 0
    pos = (c_e - c_vals[0]) / (c_vals[1] - c_vals[0])
    if not (i_e - 6 <= pos <= i_e + 24):
        return float(i_e)
    return float(pos)


def edge_frame(arm: ArmModel, plane: PlaneResult, edge: Optional[EdgeResult]):
    """(O, d, u_out) in world: a point of the (measured or, failing that, model) edge at model a=0
    lifted onto the measured plane, the edge direction (back->front sense), the in-plane outward
    unit vector. Returns also the label of the frame used."""
    if edge is not None and edge.ok:
        c0, m = edge.c0, edge.slope
        label = "measured"
    else:
        c0, m = 0.0, 0.0
        label = "model"
    O = lift_to_plane(arm, plane, 0.0, c0)
    P1 = lift_to_plane(arm, plane, 1.0, c0 + m)
    d = unit(P1 - O)
    u = np.cross(plane.n, d)
    if u @ arm.across < 0:
        u = -u
    return O, d, unit(u), label


# ------------------------------------------------------------------------------ seam
def detect_seam(frame: Frame, arm: ArmModel, plane: PlaneResult, edge: Optional[EdgeResult],
                params: DetectParams) -> SeamResult:
    res = SeamResult(ok=False)
    if plane.n is None:
        res.reasons.append("seam_needs_top_plane")
        return res
    O, d, u_out, label = edge_frame(arm, plane, edge)
    res.frame = label
    step = params.seam_mm_per_px * 1e-3
    a_vals = np.arange(-params.seam_half_along_m, params.seam_half_along_m + 1e-9, step)
    r_vals = np.arange(params.seam_inward_min_m, params.seam_inward_max_m + 1e-9, step)
    P = (O[None, None, :] + a_vals[None, :, None] * d[None, None, :]
         - r_vals[:, None, None] * u_out[None, None, :])
    shape = P.shape[:2]
    uv = frame.cam.project(transform_points(frame.T_ow, P.reshape(-1, 3)))
    finite = np.isfinite(uv).all(axis=1)
    uv_safe = np.where(finite[:, None], uv, -10.0)
    mx = uv_safe[:, 0].reshape(shape).astype(np.float32)
    my = uv_safe[:, 1].reshape(shape).astype(np.float32)
    patch = cv2.remap(frame.gray, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid = (finite & frame.cam.in_image(uv_safe, margin=2) & ~frame.tool_at(uv_safe)).reshape(shape)
    cover = valid.mean(axis=0)
    if valid.mean() < 0.3:
        res.reasons.append("seam_not_found")
        res.reasons.append("seam_patch_not_visible")
        return res
    med = float(np.median(patch[valid]))
    filled = np.where(valid, patch, int(med)).astype(np.uint8)
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (params.seam_blackhat_px, 1))
    bh = cv2.morphologyEx(cv2.GaussianBlur(filled, (0, 0), 0.7), cv2.MORPH_BLACKHAT, kern,
                                borderType=cv2.BORDER_REFLECT_101).astype(np.float64)
    w = valid.astype(np.float64)
    prof = (bh * w).sum(axis=0) / np.maximum(w.sum(axis=0), 1.0)
    prof[cover < params.seam_min_row_cover] = np.nan
    fill = np.nanmedian(prof) if np.isfinite(prof).any() else 0.0
    prof = np.where(np.isfinite(prof), prof, fill)
    ker = np.exp(-0.5 * (np.arange(-6, 7) / 1.5) ** 2)
    ker /= ker.sum()
    sm = np.convolve(np.pad(prof, 6, mode="edge"), ker, mode="valid")
    res.profile = sm
    # a seam within one kernel width of the window border cannot be told from the border response
    margin = params.seam_blackhat_px
    inner = np.zeros(len(sm), bool)
    inner[margin:len(sm) - margin] = True
    pk = int(np.argmax(np.where(inner, sm, -np.inf)))
    peak = float(sm[pk])
    excl = np.abs(np.arange(len(sm)) - pk) > 15
    base = sm[excl] if excl.any() else sm
    med_b = float(np.median(base))
    mad = float(np.median(np.abs(base - med_b))) * 1.4826
    background = med_b + mad + 1.0
    res.peak = peak
    res.confidence = peak / background
    if 0 < pk < len(sm) - 1:
        y0, y1, y2 = sm[pk - 1], sm[pk], sm[pk + 1]
        den = y0 - 2 * y1 + y2
        off = 0.5 * (y0 - y2) / den if abs(den) > 1e-9 else 0.0
        off = float(np.clip(off, -1, 1))
    else:
        off = 0.0
    res.a_m = float(a_vals[0] + (pk + off) * step)
    if res.confidence < params.seam_conf_min or peak < params.seam_min_peak:
        res.reasons.append("seam_not_found")
        return res
    res.ok = True
    return res


# ------------------------------------------------------------------------------ side-face plane (log only)
def fit_side_plane(frame: Frame, arm: ArmModel, plane: PlaneResult, edge: Optional[EdgeResult],
                   params: DetectParams) -> SidePlaneResult:
    res = SidePlaneResult(ok=False)
    if plane.n is None:
        res.reasons.append("side_plane_needs_top_plane")
        return res
    L = arm.to_local(frame.P_w)
    a, c, h = L[..., 0], L[..., 1], L[..., 2]
    c0 = edge.c0 if (edge is not None and edge.ok) else 0.0
    m = edge.slope if (edge is not None and edge.ok) else 0.0
    c_edge = c0 + m * a
    hh = (frame.P_w @ plane.n - plane.d)             # signed distance below/above the measured top
    with np.errstate(invalid="ignore"):
        sel = (frame.valid & ~frame.tool & (hh <= -0.004) & (hh >= -params.side_band_m)
               & (np.abs(c - c_edge) <= 0.015))
    idx = np.nonzero(sel)
    n = len(idx[0])
    if n < params.side_min_points:
        res.reasons.append("side_plane_insufficient_points")
        return res
    P = frame.P_w[idx]
    # prior: horizontal normal pointing out of the carton
    prior = arm.across
    out = ransac_plane(P, prior, params.side_inlier_thr_m, np.random.default_rng(params.seed + 2),
                       max_angle_deg=40.0)
    if out is None:
        res.reasons.append("side_plane_insufficient_points")
        return res
    nn, dd, mask = out
    res.n, res.d, res.inliers = nn, dd, int(mask.sum())
    res.rms_m = float(np.sqrt(np.mean((P[mask] @ nn - dd) ** 2)))
    if res.inliers < params.side_min_points:
        res.reasons.append("side_plane_insufficient_points")
    if not (res.rms_m <= 2 * params.plane_max_rms_m):
        res.reasons.append("side_plane_rms_high")
    res.ok = not res.reasons
    return res
