"""Two-flap truncation recovery: geometry and gates (2026-10-22).

SAM3 sometimes outlines ONE flap of a taped carton. When the other flap is not
coplanar with it (a flap lifted a few mm), the coplanar growth of
:mod:`top_diag` stops at the seam and every recovery candidate fails. The H-cut
planner gets ONE top plane per snapshot and cuts 3 mm below it, so a second
flap may only be accepted when its surface is close to the PUBLISHED top plane
(``record.top_face.plane``, the plane the corners' 3-D points lie on):

1. pre-filter on the two flap planes: angle between normals <= ``MAX_ANGLE_DEG``,
   step (plane-to-plane distance at the seam midpoint) <= ``MAX_STEP_MM``;
2. cut-depth gate: for EACH flap polygon (clipped to the final rectangle, split
   by the seam) the largest |height of the flap's own plane - height of the
   published plane| over the polygon's vertices <= ``MAX_DEPTH_DEV_MM``. Both are
   planes seen along pixel rays, so the deviation is a ratio of linear functions
   of the pixel: monotone along any line, hence the maximum over a convex polygon
   sits at a vertex (this bounds the cut-depth error at every point of the top).

Everything here is pure numpy / cv2 with no pipeline state; the orchestration
(candidate generation, ``_try``) is ``pipeline._truncation_recovery``. All
distances in mm unless a name says ``_m``. Planes are ``(n, d)`` with a unit
normal and ``n.x + d == 0`` in the camera frame, ``d > 0`` (camera on the +
side), like :func:`fold_edge._plane_nd`.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

#: operator decision "small step only"; the config may only tighten them
MAX_STEP_MM = 5.0
MAX_ANGLE_DEG = 3.0
MAX_DEPTH_DEV_MM = 3.0

#: strip beyond the truncated edge in which the second flap's plane is fitted
STRIP_NEAR_MM = 10.0        # starts this far past the edge (the fold radius / flap gap)
STRIP_FAR_FRAC = 0.90       # ... and ends at this share of the quad's perpendicular extent
STRIP_MARGIN = 0.15         # along the edge: the middle 70 % of the span
STRIP_MAX_ABOVE_MM = 40.0   # points further than this from plane A are never flap points
MIN_POINTS = 1500           # fewer points in the strip / flap -> no plane
MIN_INLIER_FRAC = 0.60      # ... and fewer inliers than this share -> not a plane
MIN_SIDE_AREA_FRAC = 0.15   # each flap polygon covers at least this share of the final quad
FLAP_ERODE_MM = 10.0        # final flap region is eroded by this before its plane is refitted
RANSAC_THRESH_MM = 2.0      # floor of the plane inlier band
RANSAC_ITERS = 300
RANSAC_SAMPLE = 6000


def plane_nd(n, d) -> tuple[np.ndarray, float]:
    """Unit normal, ``d`` > 0 (camera on the + side)."""
    n = np.asarray(n, dtype=np.float64).reshape(3)
    ln = float(np.linalg.norm(n))
    n, d = n / ln, float(d) / ln
    if d < 0:
        n, d = -n, -d
    return n, d


def fit_plane(points: np.ndarray, thresh_m: float, iters: int = RANSAC_ITERS,
              seed: int = 0, max_sample: int = RANSAC_SAMPLE) -> dict | None:
    """Seeded RANSAC (on a subsample) + least-squares refit of the inliers.

    The least-squares step uses the eigenvectors of the 3 x 3 covariance (never
    an SVD of the N x 3 array). Returns ``{n, d, n_points, n_inliers,
    inlier_frac, rms_mm}`` or None when there are fewer than 3 finite points or
    no usable hypothesis."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] < 3:
        return None
    rng = np.random.default_rng(seed)
    sub = pts if pts.shape[0] <= max_sample else pts[rng.choice(pts.shape[0], max_sample,
                                                                 replace=False)]
    tri = rng.integers(0, sub.shape[0], size=(int(iters), 3))
    a, b, c = sub[tri[:, 0]], sub[tri[:, 1]], sub[tri[:, 2]]
    nrm = np.cross(b - a, c - a)
    ln = np.linalg.norm(nrm, axis=1)
    ok = ln > 1e-12
    if not np.any(ok):
        return None
    nrm = nrm[ok] / ln[ok, None]
    dd = -np.einsum("ij,ij->i", nrm, a[ok])
    cnt = (np.abs(sub @ nrm.T + dd[None, :]) <= thresh_m).sum(axis=0)
    best = int(np.argmax(cnt))
    n, d = nrm[best], float(dd[best])
    for _ in range(3):
        inl = np.abs(pts @ n + d) <= thresh_m
        if int(inl.sum()) < 3:
            return None
        q = pts[inl]
        c0 = q.mean(axis=0)
        cov = (q - c0).T @ (q - c0) / q.shape[0]
        _w, v = np.linalg.eigh(cov)
        n = v[:, 0]
        d = float(-n @ c0)
    n, d = plane_nd(n, d)
    res = pts @ n + d
    inl = np.abs(res) <= thresh_m
    n_in = int(inl.sum())
    if n_in < 3:
        return None
    return {"n": n, "d": d, "n_points": int(pts.shape[0]), "n_inliers": n_in,
            "inlier_frac": n_in / float(pts.shape[0]),
            "rms_mm": float(np.sqrt(np.mean(res[inl] ** 2)) * 1000.0)}


def _ray(cam: dict, uv) -> np.ndarray:
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    return np.stack([(uv[:, 0] - float(cam["cx"])) / float(cam["fx"]),
                     (uv[:, 1] - float(cam["cy"])) / float(cam["fy"]),
                     np.ones(uv.shape[0])], axis=1)


def _project(P: np.ndarray, cam: dict) -> np.ndarray:
    P = np.asarray(P, dtype=np.float64).reshape(-1, 3)
    return np.stack([float(cam["fx"]) * P[:, 0] / P[:, 2] + float(cam["cx"]),
                     float(cam["fy"]) * P[:, 1] / P[:, 2] + float(cam["cy"])], axis=1)


def ray_plane_points(cam: dict, uv, plane: tuple) -> np.ndarray:
    """Camera-frame points where the pixel rays of ``uv`` meet ``plane``;
    rows are NaN where the ray is parallel to the plane or hits it behind the camera."""
    n, d = plane
    r = _ray(cam, uv)
    den = r @ np.asarray(n, dtype=np.float64)
    out = np.full(r.shape, np.nan)
    good = np.abs(den) > 1e-6
    t = np.full(r.shape[0], np.nan)
    t[good] = -float(d) / den[good]
    good &= t > 0
    out[good] = r[good] * t[good, None]
    return out


def plane_pair_metrics(plane_a: tuple, plane_b: tuple, cam: dict, seam_mid_px) -> dict:
    """Angle between the normals (deg) and the step (mm) at the seam midpoint:
    the distance of the point where the midpoint's pixel ray meets plane A to
    plane B. ``{"angle_deg", "step_mm"}`` (inf step when the ray misses)."""
    na, da = plane_a
    nb, db = plane_b
    angle = math.degrees(math.acos(float(np.clip(np.dot(na, nb), -1.0, 1.0))))
    pa = ray_plane_points(cam, np.asarray(seam_mid_px, dtype=np.float64).reshape(1, 2),
                          (na, da))[0]
    step = float("inf") if not np.all(np.isfinite(pa)) else abs(float(nb @ pa + db)) * 1000.0
    return {"angle_deg": angle, "step_mm": step}


def gate_pair(step_mm: float, angle_deg: float, max_step_mm: float = MAX_STEP_MM,
              max_angle_deg: float = MAX_ANGLE_DEG) -> list[str]:
    """Reasons (empty = pass): ``flap_step_too_large(11.2mm)``,
    ``flap_angle_too_large(3.7deg)`` (step first)."""
    why = []
    if not np.isfinite(step_mm) or step_mm > max_step_mm:
        why.append(f"flap_step_too_large({step_mm:.1f}mm)")
    if not np.isfinite(angle_deg) or angle_deg > max_angle_deg:
        why.append(f"flap_angle_too_large({angle_deg:.1f}deg)")
    return why


def lift_deviation_mm(poly_px, own: tuple, published: tuple, cam: dict) -> tuple[float, list]:
    """Per vertex of a pixel polygon: |height of the flap's own plane - height of the
    published plane| (mm) where the vertex's pixel ray meets the own plane (height =
    distance of that point from the published plane). Returns ``(max, per_vertex)``;
    ``inf`` when a ray misses the own plane."""
    P = ray_plane_points(cam, poly_px, own)
    n, d = published
    if P.size == 0 or not np.all(np.isfinite(P)):
        return float("inf"), []
    dev = np.abs(P @ np.asarray(n, dtype=np.float64) + float(d)) * 1000.0
    return float(dev.max()), [round(float(v), 2) for v in dev]


def _clip_halfplane(poly: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman against ``f >= 0`` (``f`` = signed value per vertex, linear)."""
    out = []
    k = poly.shape[0]
    for i in range(k):
        a, b = poly[i], poly[(i + 1) % k]
        fa, fb = float(f[i]), float(f[(i + 1) % k])
        if fa >= 0:
            out.append(a)
        if (fa >= 0) != (fb >= 0):
            t = fa / (fa - fb)
            out.append(a + t * (b - a))
    return np.asarray(out, dtype=np.float64).reshape(-1, 2)


def split_by_line(poly_px, p0, p1, ref_px) -> tuple[np.ndarray, np.ndarray]:
    """Split a convex pixel polygon by the infinite line through ``p0``, ``p1``:
    ``(side of ref_px, other side)``; both contain the two seam intersection points."""
    poly = np.asarray(poly_px, dtype=np.float64).reshape(-1, 2)
    p0 = np.asarray(p0, dtype=np.float64).reshape(2)
    p1 = np.asarray(p1, dtype=np.float64).reshape(2)
    ref = np.asarray(ref_px, dtype=np.float64).reshape(2)
    dv = p1 - p0

    def s(x):
        x = np.asarray(x, dtype=np.float64).reshape(-1, 2) - p0
        return dv[0] * x[:, 1] - dv[1] * x[:, 0]

    sign = 1.0 if float(s(ref)[0]) >= 0 else -1.0
    f = sign * s(poly)
    return _clip_halfplane(poly, f), _clip_halfplane(poly, -f)


def polygon_area(poly) -> float:
    p = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    if p.shape[0] < 3:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def points_in_polygon(poly_px, xyz: np.ndarray, proj: tuple, image_wh: tuple,
                      erode_px: int = 0) -> np.ndarray:
    """Cloud points whose pixel falls in the polygon (eroded by ``erode_px``)."""
    idx, uv = proj
    w, h = image_wh
    poly = np.asarray(poly_px, dtype=np.float64).reshape(-1, 2)
    if idx is None or len(idx) == 0 or poly.shape[0] < 3:
        return np.zeros((0, 3))
    x0 = int(max(0, np.floor(poly[:, 0].min()) - 1))
    y0 = int(max(0, np.floor(poly[:, 1].min()) - 1))
    x1 = int(min(w, np.ceil(poly[:, 0].max()) + 2))
    y1 = int(min(h, np.ceil(poly[:, 1].max()) + 2))
    if x1 <= x0 or y1 <= y0:
        return np.zeros((0, 3))
    mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(mask, [np.rint(poly - [x0, y0]).astype(np.int32)], 1)
    if erode_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(erode_px) + 1,) * 2)
        mask = cv2.erode(mask, k)
    pu = np.rint(uv[:, 0]).astype(np.int64)
    pv = np.rint(uv[:, 1]).astype(np.int64)
    sel = (pu >= x0) & (pu < x1) & (pv >= y0) & (pv < y1)
    sel[sel] = mask[pv[sel] - y0, pu[sel] - x0] > 0
    return np.asarray(xyz, dtype=np.float64)[np.asarray(idx)[sel]]


def mm_per_px(poly_px, plane: tuple, cam: dict) -> float | None:
    """Mean ground size (mm) of a pixel of the quad on ``plane`` (sqrt of area ratio)."""
    P = ray_plane_points(cam, poly_px, plane)
    if not np.all(np.isfinite(P)):
        return None
    a_mm2 = polygon_area(P @ _basis(plane[0]).T) * 1e6
    a_px = polygon_area(poly_px)
    return math.sqrt(a_mm2 / a_px) if a_px > 0 and a_mm2 > 0 else None


def _basis(n) -> np.ndarray:
    n = np.asarray(n, dtype=np.float64)
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = a - (a @ n) * n
    e1 /= np.linalg.norm(e1)
    return np.stack([e1, np.cross(n, e1)])


def strip_beyond_edge(quad_px, edge: int, plane_a: tuple, cam: dict) -> dict | None:
    """Pixel polygon of the strip beyond ring edge ``edge`` of the quad (corners
    ``edge`` and ``edge + 1``) in plane A: from ``STRIP_NEAR_MM`` past the edge out to
    ``STRIP_FAR_FRAC`` of the quad's perpendicular extent, over the middle
    ``1 - 2 * STRIP_MARGIN`` of the edge. None when the geometry is unusable."""
    quad = np.asarray(quad_px, dtype=np.float64).reshape(4, 2)
    P = ray_plane_points(cam, quad, plane_a)
    if not np.all(np.isfinite(P)):
        return None
    i, j = int(edge) % 4, (int(edge) + 2) % 4
    e0, e1 = P[i], P[(i + 1) % 4]
    mid = 0.5 * (e0 + e1)
    opp = 0.5 * (P[j] + P[(j + 1) % 4])
    u = e1 - e0
    ln_u = float(np.linalg.norm(u))
    if ln_u < 1e-6:
        return None
    u /= ln_u
    out = mid - opp
    out = out - (out @ u) * u
    extent = float(np.linalg.norm(out))
    if extent < 0.03:
        return None
    w = out / extent
    near, far = STRIP_NEAR_MM / 1000.0, STRIP_FAR_FRAC * extent
    if far - near < 0.03:
        return None
    s0, s1 = STRIP_MARGIN, 1.0 - STRIP_MARGIN
    C = np.array([e0 + s0 * (e1 - e0) + near * w, e0 + s1 * (e1 - e0) + near * w,
                  e0 + s1 * (e1 - e0) + far * w, e0 + s0 * (e1 - e0) + far * w])
    return {"poly_px": _project(C, cam), "extent_mm": extent * 1000.0,
            "edge_mid_px": _project(mid.reshape(1, 3), cam)[0]}


def fit_flap_plane(points: np.ndarray, rms_floor_mm: float = 0.0,
                   thresh_mm: float = RANSAC_THRESH_MM) -> tuple[dict | None, str | None]:
    """:func:`fit_plane` + the support gates. ``(fit | None, reason | None)``."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] < MIN_POINTS:
        return None, f"too_few_points({pts.shape[0]})"
    fit = fit_plane(pts, max(thresh_mm, 2.0 * rms_floor_mm) / 1000.0)
    if fit is None:
        return None, "no_plane"
    if fit["inlier_frac"] < MIN_INLIER_FRAC:
        return fit, f"low_inlier_frac({fit['inlier_frac']:.2f})"
    return fit, None


def _r(v, nd: int = 2):
    """Rounded float for JSON; None for non-finite (strict JSON has no Infinity)."""
    return round(float(v), nd) if np.isfinite(v) else None


def plane_json(fit: dict) -> dict:
    return {"normal": [round(float(v), 5) for v in fit["n"]],
            "offset_m": round(float(fit["d"]), 5),
            "rms_mm": round(float(fit["rms_mm"]), 2),
            "inliers": int(fit["n_inliers"]), "points": int(fit["n_points"])}


def evaluate_final(final_quad_px, seam_px, orig_quad_px, published: tuple, cam: dict,
                   xyz: np.ndarray, proj: tuple, image_wh: tuple, rms_floor_mm: float = 0.0,
                   limits: tuple = (MAX_STEP_MM, MAX_ANGLE_DEG, MAX_DEPTH_DEV_MM)) -> dict:
    """The decisive gates on the FINAL detection.

    ``final_quad_px`` = ring-ordered published quad, ``seam_px`` = ``(p0, p1)`` the
    observed seam, ``orig_quad_px`` = the outlined flap's quad (flap A = the seam side that
    holds its centre), ``published`` = ``(n, d)`` of ``record.top_face.plane``. Both flap planes
    are refitted from the cloud inside their own polygon (eroded by ``FLAP_ERODE_MM``).
    Returns a JSON-able dict: ``ok``, ``reasons`` (empty when ok), ``step_mm``, ``angle_deg``,
    ``dev_mm`` {A, B}, the planes, the polygons."""
    max_step, max_angle, max_dev = limits
    out: dict = {"ok": False, "reasons": []}
    quad = np.asarray(final_quad_px, dtype=np.float64).reshape(4, 2)
    p0, p1 = (np.asarray(p, dtype=np.float64).reshape(2) for p in seam_px)
    ref = np.asarray(orig_quad_px, dtype=np.float64).reshape(4, 2).mean(axis=0)
    poly_a, poly_b = split_by_line(quad, p0, p1, ref)
    qa = polygon_area(quad)
    fa, fb = (polygon_area(poly_a) / qa if qa > 0 else 0.0,
              polygon_area(poly_b) / qa if qa > 0 else 0.0)
    out["area_frac"] = {"A": round(fa, 3), "B": round(fb, 3)}
    if min(fa, fb) < MIN_SIDE_AREA_FRAC:
        out["reasons"].append(f"seam_not_between_flaps(A={fa:.2f},B={fb:.2f})")
        return out
    mpp = mm_per_px(quad, published, cam)
    if mpp is None:
        out["reasons"].append("flap_scale_unknown")
        return out
    erode = max(2, int(round(FLAP_ERODE_MM / mpp)))
    fits = {}
    for name, poly in (("A", poly_a), ("B", poly_b)):
        fit, why = fit_flap_plane(points_in_polygon(poly, xyz, proj, image_wh, erode),
                                  rms_floor_mm)
        if why is not None or fit is None:
            out["reasons"].append(f"flap_{name}_plane({why})")
            return out
        fits[name] = fit
        out[f"plane_{name}"] = plane_json(fit)
    pa, pb = (fits["A"]["n"], fits["A"]["d"]), (fits["B"]["n"], fits["B"]["d"])
    met = plane_pair_metrics(pa, pb, cam, 0.5 * (p0 + p1))
    out["step_mm"] = _r(met["step_mm"])
    out["angle_deg"] = _r(met["angle_deg"])
    out["reasons"] += gate_pair(met["step_mm"], met["angle_deg"], max_step, max_angle)
    dev_a, per_a = lift_deviation_mm(poly_a, pa, published, cam)
    dev_b, per_b = lift_deviation_mm(poly_b, pb, published, cam)
    out["dev_mm"] = {"A": _r(dev_a), "B": _r(dev_b)}
    out["dev_vertices_mm"] = {"A": per_a, "B": per_b}
    out["polygons_px"] = {"A": np.round(poly_a, 1).tolist(), "B": np.round(poly_b, 1).tolist()}
    if not (dev_a <= max_dev and dev_b <= max_dev):
        out["reasons"].append(f"flap_depth_dev_too_large(A={dev_a:.1f},B={dev_b:.1f}mm)")
    out["ok"] = not out["reasons"]
    return out


def hint_for(step_mm: float | None, angle_deg: float | None, dev_mm: dict | None = None,
             max_step: float = MAX_STEP_MM, max_angle: float = MAX_ANGLE_DEG,
             max_dev: float = MAX_DEPTH_DEV_MM) -> str | None:
    """Operator hint for a rejected capture whose two flaps differ in height:
    ``raised_flap(step=11.2mm): press the flap flat`` (angle / dev only when they failed)."""
    parts = []
    if step_mm is not None and np.isfinite(step_mm) and step_mm > max_step:
        parts.append(f"step={step_mm:.1f}mm")
    if angle_deg is not None and np.isfinite(angle_deg) and angle_deg > max_angle:
        parts.append(f"angle={angle_deg:.1f}deg")
    devs = [v for v in (dev_mm or {}).values() if v is not None]
    if devs and max(devs) > max_dev:
        parts.append(f"dev={max(devs):.1f}mm")
    return f"raised_flap({','.join(parts)}): press the flap flat" if parts else None
