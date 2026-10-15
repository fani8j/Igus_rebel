"""Approach assist ("hover"): where are the printed touch crosshairs, and where must the TCP be to hover over them.

Pure geometry, no ROS.  Two sources of crosshairs are located in the camera (optical) frame from the latest few
camera frames:

* ``table``  the fixed ChArUco table board: PnP (IPPE + LM) of ALL its detected corners; the touch points are board
             coordinates ``(x, y, 0)`` of ``TargetSpec.touch_points`` (the very same board-frame coordinates the
             recorder stores in ``Touch.p_board`` and the solver maps with T_world_board).  Board frame: origin =
             board centre, x right, y down as read, z INTO the board, so the paper normal that points out of the
             paper (towards the camera) is ``-z``.
* ``sheet``  the operator's hover sheet (``targets.HoverSheetSpec``) or each of the hover TILES (``hover_tiles``: one
             ``HoverSheetSpec`` per tile, sources "tile A", "tile B", ...; the legacy single sheet is "sheet"): ONE ArUco
             marker per target, pose by SOLVEPNP_IPPE_SQUARE
             with the gates of ``compal_box_perception/aruco_verification.py`` (reprojection <= 1.0 px, alternate /
             best reprojection ratio >= 1.5, shortest side >= 30 px, one detection, convex, camera-facing normal).
             Marker frame (same as that tool): origin = marker centre, x right, y up (towards "TOP"), z = x cross y =
             OUT of the paper towards the viewer; corners TL, TR, BR, BL = (-h,+h), (+h,+h), (+h,-h), (-h,-h).

Conventions
-----------
* ``n_optical`` is the unit paper normal pointing OUT of the paper (towards the camera side) for BOTH sources.
* ``x_optical`` is the in-plane +x axis of the board / marker (to the right as read).  The in-plane "up as read"
  axis is ``n cross x`` (= board -y = marker +y) for both sources; ``compare`` reports ``dy_mm`` along it.
* A hover TCP has +Z pointing INTO the paper (``-n``) and sits ``height`` above the crosshair along ``+n``.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .solver import TableObservation, pnp_candidates
from .targets import (CHESS_ID_BASE, HoverSheetSpec, TargetDetector, TargetSpec, get_dictionary, make_detector_params,
                      refine_marker_corners)
from .transforms import inv_T, make_T

# Gates of compal_box_perception/aruco_verification.py (hover tool defaults)
SHEET_MAX_REPROJ_PX = 1.0
SHEET_MIN_IPPE_RATIO = 1.5
SHEET_MIN_SIDE_PX = 30.0
# Table board
TABLE_MIN_CORNERS = 12
TABLE_MAX_REPROJ_PX = 2.0          # same as the recorder's max_pnp_rms_px (real table images at 1.5 m measure 0.8 px)
MARKER_SIGMA_RATIO = 2.0       # marker corners vs chessboard saddles (same value as solver.SolverOptions)
# Multi-frame gate
MIN_FRAMES = 3
MAX_SPREAD_MM = 1.0


@dataclass
class TargetPoint:
    point_id: int
    name: str                 # "board 1", "TOP", "P2", ...
    source: str               # "table" | "sheet"
    p_optical: np.ndarray     # (3,) crosshair position in the optical frame [m]
    n_optical: np.ndarray     # (3,) unit paper normal pointing OUT of the paper (towards the camera side)
    x_optical: np.ndarray     # (3,) unit in-plane axis (board/marker +x)
    quality: Dict[str, float] = field(default_factory=dict)   # reproj_px, n_corners, distance_m, view_angle_deg, ...


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------
def _gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(3)
    return v / np.linalg.norm(v)


def _table_frame_pose(bgr, K, dist, spec: TargetSpec, det_table: TargetDetector):
    """(T_optical_board, rms_px, n_corners) or a reason string."""
    dets = det_table.detect(bgr)
    d = dets.get("table")
    if d is None:
        return "table board not found"
    ids, obj, pix = d.arrays()
    if len(ids) < TABLE_MIN_CORNERS:
        return f"only {len(ids)} corners (< {TABLE_MIN_CORNERS})"
    h, w = _gray(bgr).shape[:2]
    obs = TableObservation(ids=ids, obj=obj, pix=pix, K=K, dist=dist, image_size=(w, h))
    cands = [c for c in pnp_candidates(obs) if _faces_camera(c[0][:3, :3], c[0][:3, 3], flip=True)]
    if not cands:
        return "PnP failed (no camera-facing pose)"
    T, _ = cands[0]
    T = _refine_weighted(T, ids, obj, pix, K, dist)
    rms = _reproj_rms(T, obj, pix, K, dist)
    if rms > TABLE_MAX_REPROJ_PX:
        return f"reprojection {rms:.2f} px > {TABLE_MAX_REPROJ_PX:.2f} px"
    return T, rms, len(ids)


def _project(T: np.ndarray, obj: np.ndarray, K, dist) -> np.ndarray:
    rv = cv2.Rodrigues(np.ascontiguousarray(T[:3, :3]))[0]
    uv, _ = cv2.projectPoints(np.asarray(obj, dtype=np.float64).reshape(-1, 1, 3), rv, T[:3, 3].reshape(3, 1),
                              np.asarray(K, dtype=np.float64), np.asarray(dist, dtype=np.float64).reshape(-1))
    return uv.reshape(-1, 2)


def _reproj_rms(T, obj, pix, K, dist) -> float:
    return float(np.sqrt(np.mean(np.sum((_project(T, obj, K, dist) - pix) ** 2, axis=1))))


def _refine_weighted(T: np.ndarray, ids, obj, pix, K, dist) -> np.ndarray:
    """Weighted LM refinement of a board pose.  Marker corners are ~2-3x noisier than the chessboard saddles
    (``SolverOptions.marker_sigma_ratio`` = 2 in the solver), so they get half the weight; this removes most of the
    ~1 mm depth bias an unweighted PnP inherits from the marker corners."""
    from scipy.optimize import least_squares
    w = np.where(np.asarray(ids) >= CHESS_ID_BASE, 1.0, 1.0 / MARKER_SIGMA_RATIO)[:, None]

    def res(x):
        Tx = make_T(cv2.Rodrigues(x[:3].reshape(3, 1))[0], x[3:])
        return ((_project(Tx, obj, K, dist) - pix) * w).ravel()

    x0 = np.concatenate([cv2.Rodrigues(np.ascontiguousarray(T[:3, :3]))[0].reshape(3), T[:3, 3]])
    try:
        sol = least_squares(res, x0, method="lm")
    except Exception:  # noqa: BLE001 - keep the unweighted pose
        return T
    if not sol.success or not np.all(np.isfinite(sol.x)):
        return T
    return make_T(cv2.Rodrigues(sol.x[:3].reshape(3, 1))[0], sol.x[3:])


def _faces_camera(R: np.ndarray, t: np.ndarray, flip: bool) -> bool:
    """The out-of-paper normal (+z of the marker frame; -z of the board frame) must point towards the camera."""
    n = -R[:, 2] if flip else R[:, 2]
    return float(n @ t) < -1e-9


def _frame_gray(bgr, cache: Optional[dict]) -> np.ndarray:
    if cache is None:
        return _gray(bgr)
    if "gray" not in cache:
        cache["gray"] = _gray(bgr)
    return cache["gray"]


def as_sheets(sheet_spec) -> List[HoverSheetSpec]:
    """The hover targets as a list of ``HoverSheetSpec`` (each with its own ``source`` name): None -> [], one spec -> [spec],
    a ``HoverTilesSpec`` (has ``.tiles``) or any sequence of specs -> its specs.  Sources must be unique."""
    if sheet_spec is None:
        return []
    if hasattr(sheet_spec, "tiles"):
        sheet_spec = sheet_spec.tiles
    specs = list(sheet_spec) if isinstance(sheet_spec, (list, tuple)) else [sheet_spec]
    names = [getattr(s, "source", "sheet") for s in specs]
    if len(set(names)) != len(names):
        raise ValueError(f"hover targets with the same name: {sorted(n for n in set(names) if names.count(n) > 1)}")
    return specs


def _source_of(spec) -> str:
    return getattr(spec, "source", "sheet")


def _find_markers(gray: np.ndarray, dictionary: str, params, new_api: bool, cache: Optional[dict] = None):
    """(corners, ids) of every marker of ``dictionary`` in ``gray``; ``cache`` (one dict per frame) shares the detection
    between the tiles of one frame."""
    if cache is not None:
        key = ("markers", dictionary)
        if key not in cache:
            cache[key] = _find_markers(gray, dictionary, params, new_api)
        return cache[key]
    d = get_dictionary(dictionary)
    if new_api:
        corners, ids, _ = cv2.aruco.ArucoDetector(d, params).detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, d, parameters=params)
    if ids is None or len(ids) == 0:
        return [], []
    return list(corners), [int(i) for i in ids.reshape(-1)]


def _refine_quad_edges(gray: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Sub-pixel corners of a dark-bordered marker quad by FITTING THE FOUR BORDER EDGES.

    ``cornerSubPix`` on a blurred, perspective-skewed marker corner is biased by 0.2-0.3 px (the neighbouring bit
    cells pull it), which at ~90 px marker side and 0.9 m is a 2-4 mm DEPTH error of the IPPE pose.  A straight edge
    has no such bias: for each side the 50 % crossing of the dark (inside the border) -> light (quiet zone) transition
    is located along ~20 profiles perpendicular to the side, a line is fitted (one outlier-trimming pass) and the
    adjacent lines are intersected.  Falls back to ``refine_marker_corners`` if anything looks wrong (profile not
    dark -> light, corners moved by more than 15 % of the side, near the image border)."""
    c0 = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    fallback = refine_marker_corners(gray, c0)
    side = float(np.mean([np.linalg.norm(c0[i] - c0[(i + 1) % 4]) for i in range(4)]))
    r = max(3.0, 0.07 * side)                                  # half length of a profile (px)
    h, w = gray.shape[:2]
    if np.any(c0 < r + 3) or np.any(c0[:, 0] > w - r - 4) or np.any(c0[:, 1] > h - r - 4):
        return fallback
    g32 = np.ascontiguousarray(gray, dtype=np.float32)
    centre = c0.mean(axis=0)
    offs = np.arange(-r, r + 1e-9, 0.25)
    lines = []
    for i in range(4):
        a, b = c0[i], c0[(i + 1) % 4]
        d = (b - a) / np.linalg.norm(b - a)
        nrm = np.array([d[1], -d[0]])
        if nrm @ (0.5 * (a + b) - centre) < 0:
            nrm = -nrm                                          # outward
        ss = np.linspace(0.18, 0.82, 21) * np.linalg.norm(b - a)
        base = a[None, :] + ss[:, None] * d[None, :]
        pts = base[:, None, :] + offs[None, :, None] * nrm[None, None, :]
        prof = cv2.remap(g32, pts[..., 0].astype(np.float32), pts[..., 1].astype(np.float32), cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)
        edge_pts = []
        for k in range(len(ss)):
            pr = prof[k]
            n_end = max(2, int(1.2 / 0.25))
            lo, hi = float(pr[:n_end].mean()), float(pr[-n_end:].mean())    # inside (dark) ... outside (light)
            if hi - lo < 25.0:
                continue
            mid = 0.5 * (lo + hi)
            idx = np.flatnonzero((pr[:-1] < mid) & (pr[1:] >= mid))
            if len(idx) == 0:
                continue
            j = int(idx[np.argmin(np.abs(offs[idx]))])           # crossing closest to the detected side
            frac = (mid - pr[j]) / max(pr[j + 1] - pr[j], 1e-6)
            edge_pts.append(base[k] + (offs[j] + frac * 0.25) * nrm)
        if len(edge_pts) < 10:
            return fallback
        E = np.array(edge_pts)
        for _ in range(2):                                      # fit, drop > 0.4 px outliers, refit
            m = E.mean(axis=0)
            _, _, vt = np.linalg.svd(E - m)
            dirv = vt[0]
            res = (E - m) @ np.array([-dirv[1], dirv[0]])
            keep = np.abs(res) <= max(0.4, 2.5 * float(np.median(np.abs(res))))
            if keep.sum() < 8:
                return fallback
            E = E[keep]
        m = E.mean(axis=0)
        _, _, vt = np.linalg.svd(E - m)
        lines.append((m, vt[0]))
    out = np.zeros((4, 2))
    for i in range(4):                                          # corner i = intersection of side i-1 and side i
        (m1, d1), (m2, d2) = lines[(i - 1) % 4], lines[i]
        A = np.array([d1, -d2]).T
        if abs(np.linalg.det(A)) < 1e-3:
            return fallback
        t = np.linalg.solve(A, m2 - m1)
        out[i] = m1 + t[0] * d1
    if np.linalg.norm(out - c0, axis=1).max() > max(1.5, 0.15 * side) or not np.all(np.isfinite(out)):
        return fallback
    return out


def _sheet_frame_pose(bgr, K, dist, spec: HoverSheetSpec, params, new_api: bool, cache: Optional[dict] = None):
    """(T_optical_marker, reproj_px, ippe_ratio, min_side_px) or a reason string (gates of aruco_verification.py)."""
    gray = _frame_gray(bgr, cache)
    corners, ids = _find_markers(gray, spec.dictionary, params, new_api, cache)
    hits = [c for c, i in zip(corners, ids) if i == spec.marker_id]
    if not hits:
        return f"marker {spec.marker_id} not found"
    if len(hits) > 1:
        return f"marker {spec.marker_id} detected {len(hits)} times"
    img_pts = _refine_quad_edges(gray, hits[0]).reshape(4, 2)
    if not np.all(np.isfinite(img_pts)):
        return "marker corners invalid"
    edges = np.linalg.norm(np.roll(img_pts, -1, axis=0) - img_pts, axis=1)
    side = float(edges.min())
    if side < SHEET_MIN_SIDE_PX:
        return f"marker side {side:.1f} px < {SHEET_MIN_SIDE_PX:.0f} px"
    if not cv2.isContourConvex(img_pts.astype(np.float32)):
        return "marker quadrilateral not convex"
    obj = spec.marker_corners_obj()
    K = np.asarray(K, dtype=np.float64)
    dist = np.asarray(dist, dtype=np.float64).reshape(-1)
    flag = cv2.SOLVEPNP_IPPE if spec.anisotropic else cv2.SOLVEPNP_IPPE_SQUARE   # IPPE_SQUARE needs a square
    try:
        res = cv2.solvePnPGeneric(obj, img_pts, K, dist, flags=flag)
    except cv2.error as exc:
        return f"IPPE failed: {exc}"
    ok, rvecs, tvecs = res[:3]
    if not ok or len(rvecs) == 0:
        return "IPPE produced no pose"
    cands = []
    for rv, tv in zip(rvecs, tvecs):
        rv = np.asarray(rv, dtype=np.float64).reshape(3, 1)
        tv = np.asarray(tv, dtype=np.float64).reshape(3, 1)
        proj, _ = cv2.projectPoints(obj, rv, tv, K, dist)
        r = proj.reshape(4, 2) - img_pts
        cands.append((float(np.sqrt(np.mean(np.sum(r * r, axis=1)))), rv, tv))
    cands.sort(key=lambda c: c[0])
    err, rv, tv = cands[0]
    if tv[2, 0] <= 0.0:
        return f"IPPE best pose has non-positive depth {tv[2, 0]:.3f} m"
    if err > SHEET_MAX_REPROJ_PX:
        return f"marker reprojection {err:.2f} px > {SHEET_MAX_REPROJ_PX:.2f} px"
    R = cv2.Rodrigues(rv)[0]
    t = tv.reshape(3)
    if not _faces_camera(R, t, flip=False):
        return "marker +z normal is not camera-facing"
    ratio = float("inf")
    if len(cands) > 1:
        ratio = cands[1][0] / max(err, 1e-9)
        if ratio < SHEET_MIN_IPPE_RATIO:
            return f"ambiguous IPPE pose (alternate/best ratio {ratio:.2f} < {SHEET_MIN_IPPE_RATIO:.2f}; tilt the sheet)"
    return make_T(R, t), err, ratio, side


def _frame_points_table(spec: TargetSpec, T: np.ndarray) -> Dict[int, tuple]:
    R, t = T[:3, :3], T[:3, 3]
    n, x = -R[:, 2], R[:, 0]
    return {pid: (R @ np.array([xy[0], xy[1], 0.0]) + t, n, x, f"board {pid}") for pid, xy in spec.touch_points.items()}


def _frame_points_sheet(spec: HoverSheetSpec, T: np.ndarray) -> Dict[int, tuple]:
    R, t = T[:3, :3], T[:3, 3]
    n, x = R[:, 2], R[:, 0]
    return {pid: (R @ np.array([xy[0], xy[1], 0.0]) + t, n, x, spec.names.get(pid, f"P{pid}"))
            for pid, xy in spec.points.items()}


def _aggregate(source: str, per_frame: List[Optional[Dict[int, tuple]]], quals: List[Optional[dict]],
               min_frames: int, max_spread_mm: float) -> Tuple[Dict[int, TargetPoint], Dict[int, float]]:
    """Mean over the frames in which the source was found; a point needs >= min_frames and spread <= max_spread_mm.
    Returns (accepted points, {pid: spread_mm} of every candidate point)."""
    good = [(pf, q) for pf, q in zip(per_frame, quals) if pf is not None]
    out: Dict[int, TargetPoint] = {}
    spreads: Dict[int, float] = {}
    if not good:
        return out, spreads
    for pid in good[0][0]:
        P = np.array([pf[pid][0] for pf, _ in good])
        mean = P.mean(axis=0)
        spread_mm = float(np.linalg.norm(P - mean, axis=1).max() * 1e3)
        spreads[pid] = spread_mm
        if len(good) < min_frames or spread_mm > max_spread_mm:
            continue
        n = _unit(np.mean([pf[pid][1] for pf, _ in good], axis=0))
        x = np.mean([pf[pid][2] for pf, _ in good], axis=0)
        x = _unit(x - (x @ n) * n)
        q = {k: float(np.mean([qq[k] for _, qq in good])) for k in good[0][1]}
        q.update(frames=float(len(good)), spread_mm=spread_mm, distance_m=float(np.linalg.norm(mean)),
                 view_angle_deg=float(np.degrees(np.arccos(np.clip(-(n @ mean) / np.linalg.norm(mean), -1, 1)))))
        out[pid] = TargetPoint(pid, good[0][0][pid][3], source, mean, n, x, q)
    return out, spreads


def detect_targets(frames: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]],
                   table_spec: Optional[TargetSpec], sheet_spec,
                   detector: Optional[TargetDetector], *, min_frames: int = MIN_FRAMES,
                   max_spread_mm: float = MAX_SPREAD_MM) -> Tuple[Dict[int, TargetPoint], Dict[str, str]]:
    """Locate the touch crosshairs of the table board and of the hover sheet / hover tiles in the optical frame.

    ``sheet_spec``: None, one ``HoverSheetSpec`` (source "sheet"), a ``HoverTilesSpec`` or a list of specs (one source per
    tile: "tile A", "tile B", ...; each with its own IPPE pose, the same gates and its own status).

    ``frames``: (bgr, K, dist) of the latest N frames.  A target point is returned only if its source was found
    (all gates passed) in >= ``min_frames`` frames and the point positions agree to within ``max_spread_mm``
    (maximum distance of a frame's position from the mean; reported as ``quality["spread_mm"]``).
    ``detector``: a ``TargetDetector``; it is used for the table board when it knows a "table" spec, otherwise (or
    when None) a table-only detector is built.  Returns ``({touch id: TargetPoint}, {source: status})`` with the sources
    "table" and "sheet" / "tile A" ... ("sheet": "no hover sheet configured" when there is none).
    """
    points: Dict[int, TargetPoint] = {}
    status: Dict[str, str] = {}
    n = len(frames)

    # ---------------- table board ----------------
    if table_spec is None or not table_spec.touch_points:
        status["table"] = "table: no table board with touch points configured"
    elif n == 0:
        status["table"] = "table: no camera frames"
    else:
        det = detector if (detector is not None and "table" in detector.specs) else TargetDetector({"table": table_spec})
        per, quals, reasons = [], [], []
        corner_counts, rms_list = [], []
        for bgr, K, dist in frames:
            r = _table_frame_pose(bgr, K, dist, table_spec, det)
            if isinstance(r, str):
                per.append(None)
                quals.append(None)
                reasons.append(r)
                continue
            T, rms, nc = r
            per.append(_frame_points_table(table_spec, T))
            quals.append(dict(reproj_px=rms, n_corners=float(nc)))
            corner_counts.append(nc)
            rms_list.append(rms)
        pts, spreads = _aggregate("table", per, quals, min_frames, max_spread_mm)
        points.update(pts)
        status["table"] = _status("table", n, len(corner_counts), reasons, spreads, pts, min_frames, max_spread_mm,
                                  f"{int(round(np.mean(corner_counts)))} corners, {np.mean(rms_list):.2f} px"
                                  if corner_counts else "", {pid: f"board {pid}" for pid in table_spec.touch_points})

    # ---------------- hover sheet / hover tiles ----------------
    sheets = as_sheets(sheet_spec)
    if not sheets:
        status["sheet"] = "sheet: no hover sheet configured"
    elif n == 0:
        for sp in sheets:
            status[_source_of(sp)] = f"{_source_of(sp)}: no camera frames"
    else:
        params = getattr(detector, "_params", None) if detector is not None else None
        if params is None:
            params = make_detector_params(0.02)
        new_api = hasattr(cv2.aruco, "ArucoDetector")
        acc = {id(sp): ([], [], [], []) for sp in sheets}          # per tile: per-frame points, qualities, reasons, rms
        for bgr, K, dist in frames:
            cache: dict = {}                                       # one marker detection per frame and dictionary
            for sp in sheets:
                per, quals, reasons, rms_list = acc[id(sp)]
                r = _sheet_frame_pose(bgr, K, dist, sp, params, new_api, cache)
                if isinstance(r, str):
                    per.append(None)
                    quals.append(None)
                    reasons.append(r)
                    continue
                T, rms, ratio, side = r
                per.append(_frame_points_sheet(sp, T))
                quals.append(dict(reproj_px=rms, n_corners=4.0, ippe_ratio=min(ratio, 1e3), min_side_px=side))
                rms_list.append(rms)
        for sp in sheets:
            source = _source_of(sp)
            per, quals, reasons, rms_list = acc[id(sp)]
            pts, spreads = _aggregate(source, per, quals, min_frames, max_spread_mm)
            clash = set(pts) & set(points)
            if clash:
                raise ValueError(f"hover point ids {sorted(clash)} of {source} collide with another source's")
            points.update(pts)
            status[source] = _status(source, n, len(rms_list), reasons, spreads, pts, min_frames, max_spread_mm,
                                     f"marker {sp.marker_id}, {np.mean(rms_list):.2f} px" if rms_list else "",
                                     {pid: sp.names.get(pid, f"P{pid}") for pid in sp.points})

    return points, status


def _status(source: str, n: int, found: int, reasons: List[str], spreads: Dict[int, float],
            pts: Dict[int, TargetPoint], min_frames: int, max_spread_mm: float, what: str,
            names: Dict[int, str]) -> str:
    if found == 0:
        top = Counter(reasons).most_common(1)[0][0] if reasons else "not found"
        return f"{source}: {top} (0 of {n} frames)"
    if found < min_frames:
        why = f"; other frames: {Counter(reasons).most_common(1)[0][0]}" if reasons else ""
        return f"{source}: {what}, found in only {found} of {n} frames (< {min_frames} needed){why}"
    bad = sorted(pid for pid, s in spreads.items() if s > max_spread_mm)
    if bad:
        worst = max(spreads[pid] for pid in bad)
        return (f"{source}: {what}, {found} frames, position spread {worst:.2f} mm > {max_spread_mm:.2f} mm for "
                + ", ".join(names.get(pid, str(pid)) for pid in bad) + " (not used; hold still / improve the view)")
    worst = max(spreads.values()) if spreads else 0.0
    return f"{source}: {what}, {found} of {n} frames, spread {worst:.2f} mm"


# --------------------------------------------------------------------------
# "Has the paper moved?" under partial occlusion (descend gate)
# --------------------------------------------------------------------------
# At the pre-hover pose the arm / tool hides part of the table board (and often all of the hover sheet), so a full
# re-detection cannot be demanded.  Instead the board / marker pose from the compute is kept and the VISIBLE corners of
# fresh frames are projected with that STORED pose: if the paper has not moved they land on the detected corners.
#
# What this does and does not prove: one pixel at the board is distance / fx, e.g. ~0.9 m / 1400 px = 0.65 mm,
# ~1.3 m / 1300 px = 1.0 mm.  A median residual <= 1.0 px therefore means "no in-plane motion beyond about 1 mm"; a
# tilt of the paper shows up as a growing residual towards the board edge.  Motion ALONG the optical axis is only seen
# through perspective (2 mm of depth at a corner 0.3 m off the axis at 1.3 m is ~0.5 px): it is NOT caught by this test.
# The table board lies flat on the table, which makes a pure depth shift very unlikely; the hover sheet's 4 corners at
# 1.5 px are in the same range.
UNMOVED_MIN_CORNERS = 8            # table: at least this many visible corners ...
UNMOVED_MIN_LINES = 2              # ... spread over at least this many distinct board rows AND columns
UNMOVED_TABLE_MEDIAN_PX = 1.0
UNMOVED_TABLE_P90_PX = 2.0
UNMOVED_SHEET_MEDIAN_PX = 1.5
UNMOVED_MIN_FRAMES = 3
MOVED_REASON = "the board moved or the detection changed: Detect points again"
HIDDEN_REASON = ("the board is hidden by the arm: cannot confirm it has not moved - use teleop from here, "
                 "or Detect points with the arm away")


def stored_source_poses(stored: Dict[int, TargetPoint], table_spec: Optional[TargetSpec] = None,
                        sheet_spec=None) -> Dict[str, np.ndarray]:
    """T_optical_board / T_optical_marker rebuilt from the stored TargetPoints of ``detect_targets`` (they are the mean
    pose's touch points: position, out-of-paper normal and in-plane x).  Board frame: z INTO the board (= -n); marker
    frame: z OUT of the paper (= n); y = z cross x for both.  One entry per source: "table", "sheet" or each "tile X"
    (``sheet_spec`` as for ``detect_targets``)."""
    out: Dict[str, np.ndarray] = {}
    sources = [("table", None if table_spec is None else table_spec.touch_points, -1.0)]
    sources += [(_source_of(sp), sp.points, 1.0) for sp in as_sheets(sheet_spec)]
    for source, spec_pts, sign in sources:
        if not spec_pts:
            continue
        ts, R = [], None
        for pid, tp in stored.items():
            if tp.source != source or pid not in spec_pts:
                continue
            n = _unit(tp.n_optical)
            x = _unit(np.asarray(tp.x_optical, dtype=float) - (np.asarray(tp.x_optical) @ n) * n)
            z = sign * n
            R = np.column_stack([x, np.cross(z, x), z])
            xy = spec_pts[pid]
            ts.append(np.asarray(tp.p_optical, dtype=float) - R @ np.array([xy[0], xy[1], 0.0]))
        if ts and R is not None:
            out[source] = make_T(R, np.mean(ts, axis=0))
    return out


def _unmoved_table_frame(bgr, K, dist, T, det) -> Tuple[str, dict]:
    """("ok" | "moved" | "hidden", stats) of one frame against the stored board pose T."""
    d = det.detect(bgr).get("table")
    if d is None:
        return "hidden", {"visible": 0, "why": "table board not found"}
    ids, obj, pix = d.arrays()
    nvis = len(ids)
    if nvis < UNMOVED_MIN_CORNERS:
        return "hidden", {"visible": nvis, "why": f"only {nvis} corners visible (< {UNMOVED_MIN_CORNERS})"}
    cols = len(set(np.round(obj[:, 0], 4)))
    rows = len(set(np.round(obj[:, 1], 4)))
    if cols < UNMOVED_MIN_LINES or rows < UNMOVED_MIN_LINES:
        return "hidden", {"visible": nvis, "why": f"the {nvis} visible corners lie on {rows} row(s) / {cols} column(s)"}
    res = np.linalg.norm(_project(T, obj, K, dist) - pix, axis=1)
    med, p90 = float(np.median(res)), float(np.percentile(res, 90))
    stats = {"visible": nvis, "median_px": med, "p90_px": p90}
    if med <= UNMOVED_TABLE_MEDIAN_PX and p90 <= UNMOVED_TABLE_P90_PX:
        return "ok", stats
    stats["why"] = (f"reprojection of {nvis} visible corners: median {med:.2f} px (limit {UNMOVED_TABLE_MEDIAN_PX:g}), "
                    f"90th percentile {p90:.2f} px (limit {UNMOVED_TABLE_P90_PX:g})")
    return "moved", stats


def _unmoved_sheet_frame(bgr, K, dist, T, spec: HoverSheetSpec, params, new_api: bool,
                         cache: Optional[dict] = None) -> Tuple[str, dict]:
    gray = _frame_gray(bgr, cache)
    corners, ids = _find_markers(gray, spec.dictionary, params, new_api, cache)
    hits = [c for c, i in zip(corners, ids) if i == spec.marker_id]
    if not hits:
        return "hidden", {"visible": 0, "why": f"marker {spec.marker_id} not found"}
    if len(hits) > 1:
        return "moved", {"visible": 4, "why": f"marker {spec.marker_id} detected {len(hits)} times"}
    img_pts = _refine_quad_edges(gray, hits[0]).reshape(4, 2)
    if not np.all(np.isfinite(img_pts)):
        return "hidden", {"visible": 0, "why": "marker corners invalid"}
    res = np.linalg.norm(_project(T, spec.marker_corners_obj(), K, dist) - img_pts, axis=1)
    med = float(np.median(res))
    stats = {"visible": 4, "median_px": med, "p90_px": float(np.percentile(res, 90))}
    if med <= UNMOVED_SHEET_MEDIAN_PX:
        return "ok", stats
    stats["why"] = f"marker corners: median reprojection {med:.2f} px (limit {UNMOVED_SHEET_MEDIAN_PX:g})"
    return "moved", stats


def unmoved(frames: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]], stored: Dict[int, TargetPoint],
            table_spec: Optional[TargetSpec], sheet_spec,
            detector: Optional[TargetDetector], sources: Optional[Sequence[str]] = None
            ) -> Tuple[bool, str, Dict[str, dict]]:
    """Has the paper NOT moved since ``stored`` (the TargetPoints of the compute) was measured?  Works with the arm in the
    way (partial occlusion) - see the comment above for what it proves.

    ``frames``: (bgr, K, dist) of >= UNMOVED_MIN_FRAMES fresh frames, ALL of which must pass.  ``sources``: which of
    "table" / "sheet" / "tile X" to check (default: every source present in ``stored``); ``sheet_spec`` as for
    ``detect_targets`` (each tile is checked against its own stored marker pose).
    table: >= 8 visible corners over >= 2 rows and >= 2 columns, projected with the stored board pose: median residual
           <= 1.0 px and 90th percentile <= 2.0 px per frame.
    sheet / tile: the marker found, its 4 corners projected with the stored marker pose: median of the 4 <= 1.5 px.
    Returns (ok, reason, stats): reason is "" or MOVED_REASON / HIDDEN_REASON (+ the numbers in parentheses); stats
    {source: {"frames": n, "verdicts": [...], "visible": [...], "median_px": [...], "p90_px": [...], "why": [...]}}.
    A frame that is both hidden and (another frame) moved reports "moved"; too few frames is "hidden"."""
    sheets = as_sheets(sheet_spec)
    poses = stored_source_poses(stored, table_spec, sheets)
    want = [s for s in (sources if sources is not None else sorted({tp.source for tp in stored.values()}))]
    stats: Dict[str, dict] = {}
    if not want:
        return False, "no stored point to compare against: Detect points again", stats
    verdicts_all: List[str] = []
    whys: List[str] = []
    for source in want:
        if source not in poses:
            return False, f"no stored {source} pose: Detect points again", stats
        if len(frames) < UNMOVED_MIN_FRAMES:
            stats[source] = {"frames": len(frames), "verdicts": []}
            return False, f"{HIDDEN_REASON} (only {len(frames)} fresh frames, need {UNMOVED_MIN_FRAMES})", stats
        per, st = [], {"frames": len(frames), "verdicts": [], "visible": [], "median_px": [], "p90_px": [], "why": []}
        if source == "table":
            det = detector if (detector is not None and "table" in detector.specs) else TargetDetector({"table": table_spec})
            for bgr, K, dist in frames:
                per.append(_unmoved_table_frame(bgr, K, dist, poses["table"], det))
        else:
            sp = next((x for x in sheets if _source_of(x) == source), None)
            if sp is None:
                return False, f"no stored {source} pose: Detect points again", stats
            params = getattr(detector, "_params", None) if detector is not None else None
            if params is None:
                params = make_detector_params(0.02)
            new_api = hasattr(cv2.aruco, "ArucoDetector")
            for bgr, K, dist in frames:
                per.append(_unmoved_sheet_frame(bgr, K, dist, poses[source], sp, params, new_api))
        for v, d in per:
            st["verdicts"].append(v)
            st["visible"].append(d.get("visible"))
            if "median_px" in d:
                st["median_px"].append(d["median_px"])
                st["p90_px"].append(d["p90_px"])
            if d.get("why"):
                st["why"].append(d["why"])
        stats[source] = st
        verdicts_all += st["verdicts"]
        whys += [f"{source}: {w}" for w in st["why"][:2]]
    detail = f" ({'; '.join(whys)})" if whys else ""
    if "moved" in verdicts_all:
        return False, MOVED_REASON + detail, stats
    if "hidden" in verdicts_all:
        return False, HIDDEN_REASON + detail, stats
    return True, "", stats


# --------------------------------------------------------------------------
# Frame conversion and TCP poses
# --------------------------------------------------------------------------
def to_base(points: Dict[int, TargetPoint], X: np.ndarray) -> Dict[int, dict]:
    """Express the target points in an arm base frame.  ``X`` = T_optical_base (maps base points into the optical
    frame, as ``pose_suggest.ArmFit.X``); the points are mapped with ``inv(X)``.
    Returns ``{id: {"p": (3,), "n": (3,), "x": (3,), "name": str, "source": str}}``."""
    Tb = inv_T(np.asarray(X, dtype=float).reshape(4, 4))
    R, t = Tb[:3, :3], Tb[:3, 3]
    return {pid: {"p": R @ tp.p_optical + t, "n": R @ tp.n_optical, "x": R @ tp.x_optical,
                  "name": tp.name, "source": tp.source} for pid, tp in points.items()}


def _in_plane_x(n: np.ndarray, x: np.ndarray) -> np.ndarray:
    x = x - (x @ n) * n
    if np.linalg.norm(x) < 1e-9:                      # x parallel to n: any perpendicular
        x = np.cross(n, [1.0, 0.0, 0.0] if abs(n[0]) < 0.9 else [0.0, 1.0, 0.0])
    return _unit(x)


def hover_tcp_poses(p_base, n_base, x_base, height_m: float = 0.01, yaw_step_deg: float = 30.0) -> List[np.ndarray]:
    """Candidate 4x4 ``T_base_tcp`` hovering over a crosshair: origin at ``p + height * n``, TCP +Z = ``-n`` (into the
    paper), yaw about the normal every ``yaw_step_deg`` (360 / step poses, starting with TCP +X = the board/marker +x
    axis projected into the plane, then rotating about ``+n`` counter-clockwise seen from above the paper).  The TCP
    frame is right handed: X x Y = Z."""
    if yaw_step_deg <= 0:
        raise ValueError("yaw_step_deg must be > 0")
    n = _unit(n_base)
    p = np.asarray(p_base, dtype=float).reshape(3)
    x0 = _in_plane_x(n, np.asarray(x_base, dtype=float).reshape(3))
    z = -n
    poses = []
    for ang in np.arange(0.0, 360.0 - 1e-9, yaw_step_deg):
        a = np.radians(ang)
        x = np.cos(a) * x0 + np.sin(a) * np.cross(n, x0)       # rotate x0 about +n by a
        y = np.cross(z, x)
        poses.append(make_T(np.column_stack([x, y, z]), p + height_m * n))
    return poses


def compare(pred: dict, touched_p_base: np.ndarray) -> dict:
    """Offset of the touched TCP from the predicted crosshair (``pred`` = one ``to_base`` entry).

    ``dx_mm`` along the in-plane x axis (board/marker +x, right as read), ``dy_mm`` along ``n cross x`` (up as read),
    ``dn_mm`` along the paper normal (+ = above the paper), ``lateral_mm = hypot(dx, dy)``, ``dist_mm`` = 3-D norm."""
    n = _unit(pred["n"])
    x = _in_plane_x(n, np.asarray(pred["x"], dtype=float))
    y = np.cross(n, x)
    d = np.asarray(touched_p_base, dtype=float).reshape(3) - np.asarray(pred["p"], dtype=float).reshape(3)
    dx, dy, dn = float(d @ x) * 1e3, float(d @ y) * 1e3, float(d @ n) * 1e3
    return dict(dx_mm=dx, dy_mm=dy, dn_mm=dn, lateral_mm=float(np.hypot(dx, dy)),
                dist_mm=float(np.linalg.norm(d)) * 1e3)


def project_points(points: Dict[int, TargetPoint], K, dist) -> Dict[int, Tuple[float, float]]:
    """Distorted image pixels (u, v) of the crosshairs for the overlay; points at/behind the camera are omitted."""
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    dist = np.asarray(dist, dtype=np.float64).reshape(-1)
    out: Dict[int, Tuple[float, float]] = {}
    for pid, tp in points.items():
        p = np.asarray(tp.p_optical, dtype=np.float64).reshape(1, 1, 3)
        if p[0, 0, 2] <= 1e-6:
            continue
        uv, _ = cv2.projectPoints(p, np.zeros(3), np.zeros(3), K, dist)
        out[pid] = (float(uv[0, 0, 0]), float(uv[0, 0, 1]))
    return out
