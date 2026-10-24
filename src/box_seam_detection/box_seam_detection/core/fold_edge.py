# Provenance: moved into box_seam_detection (2026-10-12) from rgbd_captures/scripts/autolabel_v2/fold_edge.py
# at commit 6df59943f23e0d551cdc448b47ffea9e95ece20a (working tree identical to HEAD, unmodified).
# Only imports/sys.path handling changed; logic unchanged.
"""Move top-face edges from the strongest-gradient feature to the top/side fold.

Problem (blind labels, reviewed/blind_*labels_v5.json): the subpixel refiner
(precision_box_detector.subpixel_edge.refine_quad_edges) puts each edge on
the strongest |gradient| along the edge normal. Blind human labels mark the
FOLD, the inner boundary of the top face. The two differ in two ways:

* narrow visible side face (predicted projected width ~3..30 px): the
  strongest step can be the side face's OUTER silhouette against the table,
  several px outside the fold (worst blind case 043607 right edge, ~+9.5 px);
* every edge: the fold transition is a ramp (camera blur, a rounded
  cardboard fold, often a bright 1-3 px rim/tape highlight); the argmax sits
  in the middle/outer part of that ramp, the human marks its inner shoulder
  (~1-1.5 px inside).

Side-face model (per station along each current edge): back-project the
station to the top plane (n.x + d = 0, camera intrinsics K), move the 3-D
point down by the box height along -n (big 134.25 mm / small 86.3 mm), and
project both. The signed image offset along the edge's outward normal is the
predicted side-face width ``w`` (> 0 visible, <= 0 hidden behind the top).
The side walls are not exactly perpendicular to the fitted top plane
(cardboard bulge, flap tilt, plane-fit error), so ``w_lo``/``w_hi`` come
from the same projection with the down vector tilted +-``tilt_tol_deg``
about the 3-D edge direction.

Per-edge method (recorded in ``per_edge[edge]["method"]``):

``hidden`` (w_hi <= vis_min_px, side face cannot be visible) and
``grazing`` (w_hi <= snap_px + min_side_px: at most a few px of side face,
    fold and silhouette not separable)
    The current edge IS (or is within a few px of) the occluding contour.
    Default ``hidden_mode`` "keep" leaves it (reason ``side_hidden`` /
    ``side_grazing``); "shoulder" moves it to the inner shoulder of its own
    transition like the visible branch.
``wide`` (w_lo > reach: silhouette far outside the refiner's reach)
    The current edge must be the fold; snap to the fold ramp's inner
    shoulder near the current edge (``fold_shoulder``).
``narrow`` (otherwise)
    Per station: outer silhouette S = outermost strong step (|g| >=
    ``strong_rel`` x window max) in [-(w_hi+3), w_hi+3]. Structural test on
    the edge median S_med: S_med > snap_px -> a silhouette exists outside
    the current edge, which is therefore the fold (role
    ``current_is_fold``; shoulder snap around it, ``fold_shoulder``).
    |S_med| <= snap_px -> the current edge is the outermost step, i.e. the
    silhouette (role ``current_is_silhouette``): predicted fold
    F = median(S - w), tolerance tol = (w_hi-w_lo)/2 + 2; per station the
    RGB transition nearest F within F +- tol (and >= ``min_side_px``
    inside S), their median is the new fold centre, then the shoulder snap
    (``silhouette_minus_width``). S_med < -snap_px or no silhouette -> keep
    (``silhouette_inside_current`` / ``no_silhouette``). The width model
    decides WHERE to look; the structural test decides WHETHER to move,
    because predicted widths of near-grazing walls are off by up to ~10 px
    on real boxes (plane tilt, bulging cardboard).

Inner shoulder of a transition: the main |gradient| peak inside a +-2 px
snap window; if an opposite-sign peak of >= ``rim_rel`` x its strength lies
<= ``rim_max_px`` inside it (bright rim / dark groove band), that inner peak
starts the fold complex. Shoulder = where |g| first drops below
``shoulder_frac`` x peak going inward (linear interpolation). Per-station
points -> TLS line with MAD rejection; kept (with reason) on weak evidence
(too few stations, high RMS, implausible shift/rotation). Corners = adjacent
line intersections; the whole quad reverts when it becomes invalid.

No parameter here was fit to the blind labels. Thresholds are a-priori
choices (min_gradient 4 = refiner default; shoulder_frac 0.5 = half max;
tilt 3 deg ~ plane/cardboard tolerance). The DEFINITION "fold = inner
shoulder" was chosen after inspecting blind-label profiles (diagnosis),
which is stated in the evaluation report.

Corner order: [back_left, back_right, front_right, front_left];
edge order: [back, right, front, left].
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from . import box_profiles as _profiles

__all__ = [
    "EDGE_NAMES",
    "BOX_HEIGHT_M",
    "DEFAULTS",
    "box_class_from_xyz",
    "corners_xyz_on_plane",
    "relocate_edges",
    "side_face_widths",
]

#: Edge names in order.
EDGE_NAMES: tuple[str, str, str, str] = ("back", "right", "front", "left")

#: (start, end) corner indices per edge, matching EDGE_NAMES order.
_EDGE_PAIRS: tuple[tuple[int, int], ...] = ((0, 1), (1, 2), (2, 3), (3, 0))

#: Corner keys for dict-style corners_px.
_CORNER_KEYS: tuple[str, str, str, str] = (
    "back_left", "back_right", "front_right", "front_left",
)

#: Box heights (m) by footprint class (user-measured). Long top side
#: > BIG_LONG_SIDE_M means big.
#: Derived from box_profiles (single source of the box-class constants).
BOX_HEIGHT_M: dict[str, float] = _profiles.heights_m()
BIG_LONG_SIDE_M = _profiles.LEGACY_BIG_LONG_SIDE_M

#: A-priori thresholds (px values are for 1280 px wide images and scale by
#: image_width / 1280). None of these were fit to blind labels.
DEFAULTS: dict = {
    "hidden_mode": "keep",        # "keep" | "shoulder" for hidden side faces
    "station_spacing_px": 4.0,
    "end_margin_px": 10.0,        # skip stations this close to a corner
    "end_margin_frac": 0.08,
    "step_px": 0.5,               # profile sampling step
    "min_gradient": 4.0,          # gray levels / px (refiner default)
    "vis_min_px": 1.0,            # w_hi <= this -> side face hidden
    "tilt_tol_deg": 3.0,          # side-wall tilt tolerance for w_lo/w_hi
    "reach_px": 30.0,             # w_lo above this -> "wide" branch
    "strong_rel": 0.35,           # silhouette: |g| >= this x window max
    "min_side_px": 2.0,           # fold at least this far inside S
    "snap_px": 2.0,               # shoulder snap window half-width
    "rim_max_px": 4.0,            # rim band max width
    "rim_rel": 0.3,               # rim peak strength relative to main peak
    "shoulder_frac": 0.5,         # half max
    "max_walk_px": 3.0,           # shoulder walk limit from its peak
    "min_stations": 6,
    "min_station_frac": 0.4,
    "max_rms_px": 2.0,
    "max_shoulder_shift_px": 4.0,  # |shift| cap for shoulder moves (= rim_max_px)
    "max_rotation_deg": 3.0,      # new vs current edge direction
}


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _as_corners(corners_px) -> np.ndarray:
    if isinstance(corners_px, dict):
        pts = np.array([corners_px[k] for k in _CORNER_KEYS], dtype=np.float64)
    else:
        pts = np.asarray(corners_px, dtype=np.float64)
    if pts.shape != (4, 2):
        raise ValueError(f"corners_px must be (4,2) or corner dict, got {pts.shape}")
    if not np.all(np.isfinite(pts)):
        raise ValueError("corners_px must be finite")
    return pts


def _plane_nd(plane) -> tuple[np.ndarray, float] | tuple[None, None]:
    """(unit normal, d) with n.x + d == 0 and d > 0 (camera on the + side),
    or (None, None) if unusable."""
    try:
        if plane is None:
            return None, None
        if hasattr(plane, "normal"):
            n = np.asarray(plane.normal, dtype=np.float64).ravel()
            d = float(plane.d)
        elif isinstance(plane, dict):
            n = np.asarray(plane["normal"], dtype=np.float64).ravel()
            d = float(plane.get("offset_m", plane.get("d")))
        else:
            n = np.asarray(plane[0], dtype=np.float64).ravel()
            d = float(plane[1])
        if n.shape != (3,) or not np.all(np.isfinite(n)) or not math.isfinite(d):
            return None, None
        norm = float(np.linalg.norm(n))
        if norm < 1e-9 or abs(d) / norm < 1e-6:
            return None, None
        n, d = n / norm, d / norm
        if d < 0:
            n, d = -n, -d
        return n, d
    except (KeyError, TypeError, ValueError, IndexError):
        return None, None


class Camera:
    """Pinhole K + plumb_bob distortion (k1, k2, p1, p2, k3); dist None or
    all-zero = rectified image."""

    def __init__(self, k, dist=None):
        self.k = np.asarray(k, dtype=np.float64).reshape(3, 3)
        dd = (np.zeros(0) if dist is None
              else np.asarray(dist, dtype=np.float64).ravel())
        self.dist = dd if dd.size and np.any(dd != 0) else None


def _as_camera(cam) -> "Camera":
    return cam if isinstance(cam, Camera) else Camera(cam)


def _intrinsics(manifest) -> Camera | None:
    """Camera from a capture manifest (camera_info.k / .d), a dict with 'k'
    (and optional 'd'), or a 3x3 array. None when unusable."""
    try:
        if manifest is None:
            return None
        if isinstance(manifest, Camera):
            cam = manifest
        elif isinstance(manifest, dict):
            info = manifest.get("camera_info") or manifest
            k = info.get("k")
            if k is None:
                return None
            model = str(info.get("distortion_model", "plumb_bob"))
            dist = info.get("d") if model in ("plumb_bob",
                                             "rational_polynomial") else None
            if dist is not None and len(dist) > 8:
                dist = list(dist)[:8]
            cam = Camera(k, dist)
        else:
            cam = Camera(manifest)
    except (TypeError, ValueError):
        return None
    km = cam.k
    if not np.all(np.isfinite(km)) or km[0, 0] <= 0 or km[1, 1] <= 0:
        return None
    if cam.dist is not None and not np.all(np.isfinite(cam.dist)):
        return None
    return cam


def _backproject(uv: np.ndarray, cam, n: np.ndarray,
                 d: float) -> np.ndarray:
    """Rays through pixels (N,2) (undistorted when the camera has
    distortion) intersected with the plane -> (N,3); NaN rows where the ray
    misses (parallel / behind)."""
    cam = _as_camera(cam)
    km = cam.k
    uv = np.atleast_2d(np.asarray(uv, dtype=np.float64))
    if cam.dist is not None:
        xy = cv2.undistortPointsIter(
            uv.reshape(-1, 1, 2), km, cam.dist, None, None,
            (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50,
             1e-12)).reshape(-1, 2)
    else:
        xy = np.stack([(uv[:, 0] - km[0, 2]) / km[0, 0],
                       (uv[:, 1] - km[1, 2]) / km[1, 1]], axis=1)
    rays = np.concatenate([xy, np.ones((len(uv), 1))], axis=1)
    den = rays @ n
    with np.errstate(divide="ignore", invalid="ignore"):
        t = -d / den
    bad = ~np.isfinite(t) | (t <= 0) | (np.abs(den) < 1e-9)
    out = rays * t[:, None]
    out[bad] = np.nan
    return out


def _project(xyz: np.ndarray, cam) -> np.ndarray:
    """Camera-frame points -> pixels (distortion applied when present)."""
    cam = _as_camera(cam)
    km = cam.k
    xyz = np.atleast_2d(np.asarray(xyz, dtype=np.float64))
    with np.errstate(divide="ignore", invalid="ignore"):
        x = xyz[:, 0] / xyz[:, 2]
        y = xyz[:, 1] / xyz[:, 2]
    if cam.dist is not None:
        dd = np.zeros(5)
        dd[:min(5, cam.dist.size)] = cam.dist[:5]
        k1, k2, p1, p2, k3 = dd
        r2 = x * x + y * y
        rad = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 ** 3
        x, y = (x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x),
                y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y)
    return np.stack([km[0, 0] * x + km[0, 2], km[1, 1] * y + km[1, 2]],
                    axis=1)


def _rotate(v: np.ndarray, axis: np.ndarray, ang: float) -> np.ndarray:
    """Rodrigues rotation of v about unit axis by ang (rad)."""
    c, s = math.cos(ang), math.sin(ang)
    return v * c + np.cross(axis, v) * s + axis * float(axis @ v) * (1 - c)


def corners_xyz_on_plane(corners_px, manifest, plane) -> np.ndarray | None:
    """(4,3) camera-frame corners = (undistorted) pixel rays intersected
    with the plane, like the guided-plane corners, or None."""
    km = _intrinsics(manifest)
    n, d = _plane_nd(plane)
    if km is None or n is None:
        return None
    xyz = _backproject(_as_corners(corners_px), km, n, d)
    return xyz if np.all(np.isfinite(xyz)) else None


def box_class_from_xyz(corners_xyz) -> str | None:
    """'big' when the longer top-face side exceeds 150 mm, else 'small'."""
    try:
        x = np.asarray(corners_xyz, dtype=np.float64).reshape(4, 3)
    except (TypeError, ValueError):
        return None
    if not np.all(np.isfinite(x)):
        return None
    sides = [float(np.linalg.norm(x[j] - x[i])) for i, j in _EDGE_PAIRS]
    long_side = max(0.5 * (sides[0] + sides[2]), 0.5 * (sides[1] + sides[3]))
    return "big" if long_side > BIG_LONG_SIDE_M else "small"


def _box_height(box_class) -> float | None:
    if isinstance(box_class, bool):
        return None
    if isinstance(box_class, (int, float)):
        h = float(box_class)
        return h if math.isfinite(h) and h > 0 else None
    if isinstance(box_class, str) and box_class in BOX_HEIGHT_M:
        return BOX_HEIGHT_M[box_class]
    return None


def side_face_widths(stations: np.ndarray, x_edge: np.ndarray,
                     out_n: np.ndarray, km: np.ndarray, n: np.ndarray,
                     d: float, height: float, tilt_deg: float):
    """Predicted projected side-face width per station (px along out_n).

    Returns (w, w_lo, w_hi) arrays; NaN where the geometry fails.
    ``x_edge`` = (2,3) 3-D edge end points (for the tilt axis).
    """
    xs = _backproject(stations, km, n, d)
    axis = x_edge[1] - x_edge[0]
    na = float(np.linalg.norm(axis))
    down = -n
    downs = [down]
    if na > 1e-12 and tilt_deg > 0:
        axis = axis / na
        for sgn in (-1.0, 1.0):
            downs.append(_rotate(down, axis, sgn * math.radians(tilt_deg)))
    top = _project(xs, km)
    ws = []
    for dv in downs:
        bot = _project(xs + height * dv[None, :], km)
        ws.append((bot - top) @ out_n)
    ws = np.array(ws)
    return ws[0], np.min(ws, axis=0), np.max(ws, axis=0)


# ---------------------------------------------------------------------------
# Profile helpers
# ---------------------------------------------------------------------------

def _to_gray(bgr: np.ndarray) -> np.ndarray:
    arr = np.asarray(bgr)
    if arr.ndim == 2:
        return arr.astype(np.float32)
    if arr.ndim == 3 and arr.shape[2] in (3, 4):
        ch = np.ascontiguousarray(arr[..., :3])
        if ch.dtype != np.uint8:
            ch = np.clip(ch, 0, 255).astype(np.uint8)
        return cv2.cvtColor(ch, cv2.COLOR_BGR2GRAY).astype(np.float32)
    raise ValueError(f"bgr must be HxW or HxWx3/4, got {arr.shape}")


def _sample(gray: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Bilinear samples, integer pixel centres (same convention as the
    refiner). xs/ys (M,N) -> (M,N)."""
    return cv2.remap(gray, xs.astype(np.float32), ys.astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _peaks(absg: np.ndarray, thr: float) -> np.ndarray:
    """Indices of local maxima of |g| >= thr (plateaus: first index)."""
    if absg.size < 3:
        return np.zeros(0, dtype=np.int64)
    a = absg
    m = (a[1:-1] >= a[:-2]) & (a[1:-1] > a[2:]) & (a[1:-1] >= thr)
    return np.nonzero(m)[0] + 1


def _parabola(absg: np.ndarray, i: int) -> float:
    if i <= 0 or i >= len(absg) - 1:
        return 0.0
    den = float(absg[i - 1] - 2.0 * absg[i] + absg[i + 1])
    if den >= -1e-12:
        return 0.0
    return float(np.clip(0.5 * float(absg[i - 1] - absg[i + 1]) / den,
                         -0.5, 0.5))


def _inner_shoulder(g: np.ndarray, us: np.ndarray, i: int, frac: float,
                    max_walk: float) -> float | None:
    """Going inward (decreasing u) from peak i, first u where sign-aligned
    g drops below frac * g[i]; linear interpolation. None when the walk
    exceeds max_walk."""
    sgn = 1.0 if g[i] >= 0 else -1.0
    level = frac * sgn * float(g[i])
    j = i
    while j > 0:
        if sgn * float(g[j - 1]) < level:
            a, b = sgn * float(g[j - 1]), sgn * float(g[j])
            t = (b - level) / (b - a) if b != a else 0.0
            u = float(us[j] - t * (us[j] - us[j - 1]))
            return u if us[i] - u <= max_walk + 1e-9 else None
        j -= 1
        if us[i] - us[j] > max_walk:
            return None
    return None


def _fold_shoulder(g: np.ndarray, us: np.ndarray, center: float,
                   c: dict, scale: float, thr: float) -> float | None:
    """Inner shoulder of the fold complex whose main peak lies within
    +-snap_px of ``center``. None when no transition qualifies."""
    absg = np.abs(g)
    snap = float(c["snap_px"]) * scale
    idx = _peaks(absg, thr)
    if idx.size == 0:
        return None
    near = idx[np.abs(us[idx] - center) <= snap]
    if near.size == 0:
        return None
    main = int(near[np.argmax(absg[near])])
    start = main
    # rim band / groove: opposite-sign peak just inside the main peak
    rim_max = float(c["rim_max_px"]) * scale
    inner = idx[(us[idx] < us[main]) & (us[main] - us[idx] <= rim_max)
                & (np.sign(g[idx]) == -np.sign(g[main]))
                & (absg[idx] >= float(c["rim_rel"]) * absg[main])]
    if inner.size:
        start = int(inner[np.argmax(us[inner])])  # the one nearest main
    return _inner_shoulder(g, us, start, float(c["shoulder_frac"]),
                           float(c["max_walk_px"]) * scale)


def _tls_fit(points: np.ndarray, min_samples: int):
    """TLS line + MAD rejection. Returns (abc, n_inliers, rms, keep)."""
    n_total = int(points.shape[0])
    if n_total < min_samples:
        return None
    keep = np.ones(n_total, dtype=bool)
    for _ in range(3):
        block = points[keep]
        mean = block.mean(axis=0)
        _, _, vt = np.linalg.svd(block - mean, full_matrices=False)
        normal = vt[-1]
        dist = np.abs((points - mean) @ normal)
        med = float(np.median(dist[keep]))
        mad = float(np.median(np.abs(dist[keep] - med)))
        thr = max(med + 3.0 * 1.4826 * mad, 0.75)
        cand = dist <= thr
        if int(cand.sum()) < min_samples or np.array_equal(cand, keep):
            break
        keep = cand
    if int(keep.sum()) < min_samples:
        return None
    block = points[keep]
    mean = block.mean(axis=0)
    _, _, vt = np.linalg.svd(block - mean, full_matrices=False)
    a, b = float(vt[-1, 0]), float(vt[-1, 1])
    cc = float(-(a * mean[0] + b * mean[1]))
    rms = float(np.sqrt(np.mean(((block - mean) @ np.array([a, b])) ** 2)))
    return np.array([a, b, cc]), int(keep.sum()), rms, keep


def _line_through(p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    a = p1 - p0
    nv = np.array([-a[1], a[0]])
    nv = nv / float(np.linalg.norm(nv))
    return np.array([nv[0], nv[1], float(-(nv @ p0))])


def _intersect(l1: np.ndarray, l2: np.ndarray) -> np.ndarray | None:
    a1, b1, c1 = (float(v) for v in l1)
    a2, b2, c2 = (float(v) for v in l2)
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-12:
        return None
    return np.array([(b1 * c2 - b2 * c1) / det, (c1 * a2 - c2 * a1) / det])


def _is_convex(quad: np.ndarray) -> bool:
    signs = []
    for k in range(4):
        e1 = quad[(k + 1) % 4] - quad[k]
        e2 = quad[(k + 2) % 4] - quad[(k + 1) % 4]
        signs.append(float(e1[0] * e2[1] - e1[1] * e2[0]))
    return all(s > 1e-9 for s in signs) or all(s < -1e-9 for s in signs)


def _r(v, nd=2):
    return None if v is None or not math.isfinite(float(v)) else round(float(v), nd)


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def _keep_all(corners, reason):
    return {"corners_px": corners.copy(),
            "per_edge": {e: {"method": "keep", "reason": reason}
                         for e in EDGE_NAMES},
            "changed": False}


def _process_edge(gray, corners, ei, km, n, d, height, c, scale, W, H):
    i0, i1 = _EDGE_PAIRS[ei]
    p0, p1 = corners[i0], corners[i1]
    vec = p1 - p0
    length = float(np.linalg.norm(vec))
    rec: dict = {"method": "keep"}
    if length < 1e-6:
        rec["reason"] = "degenerate_edge"
        return rec, None
    tdir = vec / length
    out_n = np.array([-tdir[1], tdir[0]])
    if float(out_n @ ((p0 + p1) / 2.0 - corners.mean(axis=0))) < 0:
        out_n = -out_n
    margin = max(float(c["end_margin_px"]) * scale,
                 float(c["end_margin_frac"]) * length)
    ts = np.arange(margin, length - margin + 1e-9,
                   float(c["station_spacing_px"]) * scale)
    if ts.size < int(c["min_stations"]):
        rec["reason"] = "edge_too_short"
        return rec, None
    stations = p0[None, :] + tdir[None, :] * ts[:, None]

    x_edge = _backproject(np.stack([p0, p1]), km, n, d)
    if not np.all(np.isfinite(x_edge)):
        rec["reason"] = "backprojection_failed"
        return rec, None
    w, w_lo, w_hi = side_face_widths(stations, x_edge, out_n, km, n, d,
                                     height, float(c["tilt_tol_deg"]))
    if not np.all(np.isfinite(w)):
        rec["reason"] = "side_width_nan"
        return rec, None
    w_med, lo_med, hi_med = (float(np.median(w)), float(np.median(w_lo)),
                             float(np.median(w_hi)))
    rec.update({"w_pred_px": _r(w_med, 1), "w_lo_px": _r(lo_med, 1),
                "w_hi_px": _r(hi_med, 1), "n_stations": int(ts.size)})
    vis_min = float(c["vis_min_px"]) * scale
    reach = float(c["reach_px"]) * scale
    grazing_max = (float(c["snap_px"]) + float(c["min_side_px"])) * scale
    if hi_med <= vis_min:
        side = "hidden"
    elif hi_med <= grazing_max:
        # at most a few px of side face: fold and silhouette are not
        # separable by the silhouette test -> treated like hidden
        side = "grazing"
    elif lo_med > reach:
        side = "wide"
    else:
        side = "narrow"
    rec["side_face"] = side
    if (side in ("hidden", "grazing")
            and c.get("hidden_mode", "keep") != "shoulder"):
        rec["reason"] = f"side_{side}"
        return rec, None

    step = float(c["step_px"])
    thr = float(c["min_gradient"])
    if side == "narrow":
        ext = max(hi_med, 0.0) + 4.0 * scale
        u_in, u_out = ext + 4.0 * scale, ext
    else:
        u_in, u_out = 8.0 * scale, 8.0 * scale
    us = np.arange(-u_in, u_out + 1e-9, step)
    xs = stations[:, 0:1] + out_n[0] * us[None, :]
    ys = stations[:, 1:2] + out_n[1] * us[None, :]
    inside = ((xs.min(axis=1) >= 0) & (xs.max(axis=1) <= W - 1)
              & (ys.min(axis=1) >= 0) & (ys.max(axis=1) <= H - 1))
    prof = _sample(gray, xs, ys).astype(np.float64)
    k = np.array([0.25, 0.5, 0.25])
    sm = np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), 1, prof)
    g = np.zeros_like(sm)
    g[:, 1:-1] = 0.5 * (sm[:, 2:] - sm[:, :-2]) / step
    g[:, :2] = 0.0
    g[:, -2:] = 0.0

    fold_u = np.full(len(ts), np.nan)
    centers = np.zeros(len(ts))
    relocating = False
    if side == "narrow":
        # 1) outer silhouette per station: outermost strong step
        sil_u = np.full(len(ts), np.nan)
        for si in range(len(ts)):
            if not inside[si]:
                continue
            absg = np.abs(g[si])
            win = (us >= -(w_hi[si] + 3.0 * scale)) & (
                us <= w_hi[si] + 3.0 * scale)
            amax = float(absg[win].max()) if np.any(win) else 0.0
            strong = _peaks(absg, max(thr, float(c["strong_rel"]) * amax))
            strong = strong[win[strong]]
            if strong.size:
                s_i = int(strong[np.argmax(us[strong])])
                sil_u[si] = float(us[s_i] + _parabola(absg, s_i) * step)
        oks = np.isfinite(sil_u)
        rec["n_silhouette"] = int(oks.sum())
        if oks.sum() < max(int(c["min_stations"]), int(math.ceil(
                float(c["min_station_frac"]) * len(ts)))):
            rec["reason"] = "no_silhouette"
            return rec, None
        # 2) structural test: is the current edge the outermost strong
        #    step (= silhouette) or is there a silhouette outside it?
        s_med = float(np.median(sil_u[oks]))
        f_edge = float(np.median(sil_u[oks] - w[oks]))
        tol = 0.5 * float(np.median(w_hi - w_lo)) + 2.0 * scale
        rec.update({"silhouette_u_med_px": _r(s_med),
                    "fold_pred_u_px": _r(f_edge), "fold_pred_tol_px": _r(tol)})
        snap = float(c["snap_px"]) * scale
        if s_med > snap:
            rec["role"] = "current_is_fold"   # silhouette found outside
        elif s_med < -snap:
            rec["reason"] = "silhouette_inside_current"
            return rec, None
        else:
            rec["role"] = "current_is_silhouette"
            # current edge sits outside the fold (on the side face or its
            # silhouette): nearest transition to the prediction per
            # station, then one consensus centre for the shoulder snap
            picks = []
            for si in np.nonzero(oks)[0]:
                absg = np.abs(g[si])
                cand = _peaks(absg, thr)
                lo = f_edge - tol
                hi = min(f_edge + tol,
                         sil_u[si] - float(c["min_side_px"]) * scale)
                cand = cand[(us[cand] >= lo) & (us[cand] <= hi)]
                if cand.size:
                    picks.append(float(
                        us[cand[np.argmin(np.abs(us[cand] - f_edge))]]))
            if len(picks) < int(c["min_stations"]):
                rec["reason"] = "weak_fold_evidence"
                return rec, None
            centers[:] = float(np.median(picks))
            rec["fold_center_u_px"] = _r(centers[0])
            relocating = True
    for si in range(len(ts)):
        if not inside[si]:
            continue
        u_f = _fold_shoulder(g[si], us, float(centers[si]), c, scale, thr)
        if u_f is not None:
            fold_u[si] = u_f

    ok = np.isfinite(fold_u)
    n_ok = int(ok.sum())
    rec["n_used"] = n_ok
    if n_ok < max(int(c["min_stations"]),
                  int(math.ceil(float(c["min_station_frac"]) * len(ts)))):
        rec["reason"] = "weak_evidence"
        return rec, None
    pts = stations[ok] + out_n[None, :] * fold_u[ok][:, None]
    fit = _tls_fit(pts, int(c["min_stations"]))
    if fit is None:
        rec["reason"] = "tls_failed"
        return rec, None
    line, n_in, rms, _keep = fit
    # orient line normal like out_n; signed shift of the new line at the
    # station positions (outward +)
    if float(line[:2] @ out_n) < 0:
        line = -line
    mid = (p0 + p1) / 2.0
    shift_mid = -float(line[:2] @ mid + line[2])
    shift_med = float(np.median(fold_u[ok]))
    rot = math.degrees(math.acos(min(1.0, abs(float(line[:2] @ out_n)))))
    rec.update({"n_inliers": n_in, "rms_px": _r(rms),
                "shift_med_px": _r(shift_med), "shift_mid_px": _r(shift_mid),
                "rotation_deg": _r(rot)})
    if rms > float(c["max_rms_px"]) * scale:
        rec["reason"] = "high_rms"
        return rec, None
    if rot > float(c["max_rotation_deg"]):
        rec["reason"] = "rotation_sanity"
        return rec, None
    if relocating:
        cap = max(hi_med, 0.0) + 2.0 * scale
        if -shift_mid > cap:
            rec["reason"] = "shift_sanity"
            return rec, None
        rec["method"] = "silhouette_minus_width"
    else:
        if abs(shift_mid) > float(c["max_shoulder_shift_px"]) * scale:
            rec["reason"] = "shift_sanity"
            return rec, None
        rec["method"] = "fold_shoulder"
    rec.pop("reason", None)
    return rec, line


def relocate_edges(bgr, corners_px, manifest, plane, box_class, cfg=None,
                   outward_limit_px=None):
    """Relocate the four top-face edges onto the top/side fold.

    Args:
        bgr: HxW gray or HxWx3/4 BGR image.
        corners_px: (4,2) [BL, BR, FR, FL] current (refined) quad, or dict.
        manifest: capture manifest (camera_info.k), {"k": [...]}, or 3x3 K.
        plane: top-face plane: dict {normal, offset_m}, Plane-like object
            (.normal, .d) or (normal, d); n.x + d == 0, camera frame, m.
        box_class: "big" | "small" or a box height in metres.
        cfg: optional overrides of DEFAULTS.
        outward_limit_px: None (default) = unchanged behaviour. A number
            forbids an edge with a visible side face (predicted width > 3 px)
            to move OUTWARD (toward / onto the side wall) by more than this
            many px (at 1280 px width; scaled): that edge is kept instead
            (reason ``outward_blocked_side_wall``). Used for yawed boxes whose
            side-wall boundary can be mistaken for the fold.

    Returns:
        {"corners_px": (4,2) array, "per_edge": {edge: evidence dict},
         "changed": bool}. Never raises on data; unchanged corners with a
        reason when geometry or evidence is insufficient.
    """
    c = dict(DEFAULTS)
    if cfg:
        c.update({k: v for k, v in cfg.items() if k in DEFAULTS})
    try:
        corners = _as_corners(corners_px)
    except (ValueError, KeyError, TypeError) as exc:
        return {"corners_px": None,
                "per_edge": {e: {"method": "keep",
                                 "reason": f"bad_corners({exc})"}
                             for e in EDGE_NAMES},
                "changed": False}
    try:
        gray = _to_gray(bgr)
    except (ValueError, TypeError, cv2.error) as exc:
        return _keep_all(corners, f"bad_image({exc})")
    H, W = gray.shape
    n, d = _plane_nd(plane)
    if n is None:
        return _keep_all(corners, "bad_plane")
    height = _box_height(box_class)
    if height is None:
        return _keep_all(corners, "bad_box_class")
    km = _intrinsics(manifest)
    if km is None:
        return _keep_all(corners, "no_intrinsics")
    scale = W / 1280.0

    per_edge: dict = {}
    lines: list = [None] * 4
    for ei, name in enumerate(EDGE_NAMES):
        rec, line = _process_edge(gray, corners, ei, km, n, d, height, c,
                                  scale, W, H)
        per_edge[name] = rec
        lines[ei] = line
    if outward_limit_px is not None:
        cen = corners.mean(axis=0)
        for ei, (i0, i1) in enumerate(_EDGE_PAIRS):
            line = lines[ei]
            name = EDGE_NAMES[ei]
            if line is None or float(per_edge[name].get("w_pred_px") or 0.0) \
                    <= 3.0 * scale:
                continue
            p0, p1 = corners[i0], corners[i1]
            tv = p1 - p0
            tn = float(np.linalg.norm(tv))
            if tn < 1e-9:
                continue
            on = np.array([-tv[1], tv[0]]) / tn
            mid = 0.5 * (p0 + p1)
            if float(on @ (mid - cen)) < 0:
                on = -on
            den = float(line[0] * on[0] + line[1] * on[1])
            if abs(den) < 1e-9:
                continue
            u = -(float(line[0] * mid[0] + line[1] * mid[1]) + float(line[2])) / den
            if u > float(outward_limit_px) * scale:
                per_edge[name] = {**per_edge[name], "method": "keep",
                                  "reason": "outward_blocked_side_wall",
                                  "blocked_outward_px": _r(u, 1)}
                lines[ei] = None
    if all(line is None for line in lines):
        return {"corners_px": corners.copy(), "per_edge": per_edge,
                "changed": False}
    base = [lines[ei] if lines[ei] is not None
            else _line_through(corners[i0], corners[i1])
            for ei, (i0, i1) in enumerate(_EDGE_PAIRS)]
    back, right, front, left = base
    out = []
    for la, lb in ((left, back), (back, right), (right, front), (front, left)):
        hit = _intersect(la, lb)
        out.append(hit if hit is not None else np.full(2, np.nan))
    quad = np.array(out)
    ok = (np.all(np.isfinite(quad)) and _is_convex(quad)
          and np.all(quad[:, 0] >= 0) and np.all(quad[:, 0] <= W - 1)
          and np.all(quad[:, 1] >= 0) and np.all(quad[:, 1] <= H - 1))
    if ok:
        # corner displacement sanity: a corner may move at most the larger
        # of its two edges' allowed shifts (+ slack for the new angle)
        lim = []
        for ci in range(4):
            eds = [EDGE_NAMES[ci], EDGE_NAMES[(ci - 1) % 4]]
            m = 0.0
            for en in eds:
                r = per_edge[en]
                if r.get("method") == "silhouette_minus_width":
                    m = max(m, max(r.get("w_hi_px") or 0.0, 0.0) + 2.0 * scale)
                else:
                    m = max(m, float(c["max_shoulder_shift_px"]) * scale)
            lim.append(2.0 * m + 2.0 * scale)
        disp = np.linalg.norm(quad - corners, axis=1)
        ok = bool(np.all(disp <= np.array(lim)))
    if not ok:
        for en, line in zip(EDGE_NAMES, lines):
            if line is not None:
                per_edge[en] = {**per_edge[en], "method": "keep",
                                "reason": "quad_invalid_reverted"}
        return {"corners_px": corners.copy(), "per_edge": per_edge,
                "changed": False}
    return {"corners_px": quad, "per_edge": per_edge, "changed": True}
