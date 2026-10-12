"""K1: gap-anchored known-size top face (2026-10-12), profiles with ``top_from_gap``.

For the large carton (outer 390 x 320 mm, flap gap centred) the RGB quad is
precise but biased: its edge that faces the camera lies on the rounded lip
(~8.5 mm too far forward), the far edge is ~2.5 mm inside. What is known:

* the observed flap-gap line (seam stage, ``seam_publish = "observed"``),
* the box is a 390 x 320 mm rectangle whose gap is (more or less) centred.

So the two edges PARALLEL to the seam are placed at gap +- 160 mm (half the
known across-seam dimension) in the top plane; the two edges the seam joins
keep their RGB position and direction, unless their measured distance along
the seam deviates from the known 390 mm by more than ``LEN_TOL_MM`` (then they
are re-centred symmetrically to 390). The top plane is re-fitted on the
central 60 % of the quad first (the plane fitted on the whole eroded quad is
tilted by the lid bulge / lip: 0.1-0.7 deg here).

K1 never invents a box: it declines (``applied`` False, reason recorded) unless
the RGB quad already agrees with the known size (across-seam distance within
``MAX_ACROSS_DEV_MM``, length within ``MAX_LEN_DEV_MM``, a centred gap, a
near-rectangular RGB quad, plausible per-edge corrections). Truncated /
two-box quads therefore keep their RGB quad (and their other gates).

All geometry is in the re-fitted plane; positions are reported with the
OUTWARD displacement of each new edge relative to the RGB edge (mm, + = the
new edge lies further out). Pure numpy/OpenCV; never raises on data.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

CENTRAL_FRAC = 0.60          # linear shrink of the quad used for the plane refit
REFIT_MIN_POINTS = 3000      # inliers needed in the central region
REFIT_BANDS_MM = (6.0, 6.0, 3.5, 3.5, 3.5)   # trimmed least squares passes
LEN_TOL_MM = 5.0             # re-centre the seam-end edges when |measured - known| exceeds this
MAX_ACROSS_DEV_MM = 12.0     # |RGB across-seam distance - known| above: decline
MAX_LEN_DEV_MM = 30.0        # |RGB along-seam distance - known| above: decline
MAX_GAP_OFF_CENTRE_MM = 15.0 # gap further than this from the RGB quad's middle: decline
MAX_RGB_RECT_DIFF_MM = 25.0  # RGB quad opposite sides differ more than this: decline
MAX_SIDE_DELTA_MM = 20.0     # |correction| of a seam-parallel edge above: decline
MAX_END_DELTA_MM = 15.0      # |correction| of a seam-end edge above: decline
MAX_TILT_DEG = 3.0           # refit plane vs the pipeline plane


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def refit_plane(xyz: np.ndarray, proj: tuple, quad: np.ndarray, n0: np.ndarray,
                d0: float, frac: float = CENTRAL_FRAC) -> dict | None:
    """Trimmed least-squares plane through the cloud points that project into
    the quad shrunk to ``frac`` about its centroid. Starts from ``(n0, d0)``.
    Returns ``{"n", "d", "n_points", "n_inliers", "rms_mm", "tilt_deg"}`` or None."""
    idx, uv = proj
    quad = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    c = quad.mean(axis=0)
    q = c + (quad - c) * float(frac)
    x0, y0 = np.floor(q.min(axis=0)).astype(int)
    x1, y1 = np.ceil(q.max(axis=0)).astype(int) + 1
    if x1 <= x0 or y1 <= y0:
        return None
    mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(mask, [np.rint(q - [x0, y0]).astype(np.int32)], 1)
    pu = np.rint(uv[:, 0]).astype(np.int64)
    pv = np.rint(uv[:, 1]).astype(np.int64)
    box = (pu >= x0) & (pu < x1) & (pv >= y0) & (pv < y1)
    rows = np.nonzero(box)[0]
    inside = mask[pv[rows] - y0, pu[rows] - x0] > 0
    pts = np.asarray(xyz, dtype=np.float64)[idx[rows[inside]]]
    if len(pts) < REFIT_MIN_POINTS:
        return None
    n, d = np.asarray(n0, dtype=np.float64).copy(), float(d0)
    keep = None
    for band in REFIT_BANDS_MM:
        h = (pts @ n + d) * 1000.0
        keep = np.abs(h) <= band
        if int(keep.sum()) < REFIT_MIN_POINTS:
            return None
        sub = pts[keep]
        cen = sub.mean(axis=0)
        _u, _s, vt = np.linalg.svd(sub - cen, full_matrices=False)
        nn = vt[2]
        if float(nn @ n0) < 0:
            nn = -nn
        n = nn / np.linalg.norm(nn)
        d = -float(n @ cen)
        if d < 0:
            n, d = -n, -d
    res = (pts[keep] @ n + d) * 1000.0
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(n @ n0)))))
    return {"n": n, "d": float(d), "n_points": int(len(pts)),
            "n_inliers": int(keep.sum()),
            "rms_mm": float(np.sqrt(np.mean(res * res))), "tilt_deg": tilt}


def _line_x(p0, p1, axis_val, axis):
    """Value of the OTHER coordinate where the line through 2-D points p0, p1
    has coordinate ``axis`` (0 = u, 1 = v) equal to ``axis_val``. Also returns
    the slope d(other)/d(axis). None when the line is parallel."""
    da = p1[axis] - p0[axis]
    if abs(da) < 1e-9:
        return None, None
    other = 1 - axis
    s = (p1[other] - p0[other]) / da
    return float(p0[other] + s * (axis_val - p0[axis])), float(s)


def _intersect(l1, l2):
    """Intersection of two 2-D lines given as (point, direction)."""
    (p, r), (q, s) = l1, l2
    den = r[0] * s[1] - r[1] * s[0]
    if abs(den) < 1e-12:
        return None
    t = ((q[0] - p[0]) * s[1] - (q[1] - p[1]) * s[0]) / den
    return np.array([p[0] + t * r[0], p[1] + t * r[1]])


def compute(quad_px: np.ndarray, seam_endpoints: dict, axis: str, xyz: np.ndarray,
            proj: tuple, plane_nd: tuple, backproject_factory, project,
            along_mm: float, across_mm: float, image_wh: tuple) -> dict:
    """Gap-anchored quad for a ring-ordered ``quad_px`` (4, 2).

    ``plane_nd``: the pipeline's (n, d) (camera on the + side).
    ``backproject_factory(n, d)`` -> ``f(uv (N,2)) -> (N,3)`` plane points (m);
    ``project(xyz (N,3)) -> (N,2)`` pixels. ``seam_endpoints``:
    ``{"seam_left": [u,v], "seam_right": [u,v]}``. Returns a dict with
    ``applied`` (bool) and ``reason``; when applied also ``quad_px``,
    ``corners_xyz``, ``plane`` (n, d, diag), ``seam_endpoints`` (None when the
    seam-end edges did not move), ``deltas_mm`` and the measurements.
    """
    out: dict = {"applied": False, "reason": None}

    def fail(reason, **kw):
        out["reason"] = reason
        out.update(kw)
        return out

    try:
        quad = np.asarray(quad_px, dtype=np.float64).reshape(4, 2)
        n0, d0 = plane_nd
        if axis not in ("lr", "bf"):
            return fail("axis_unknown")
        fit = refit_plane(xyz, proj, quad, n0, d0)
        if fit is None:
            return fail("refit_failed")
        n, d = fit["n"], fit["d"]
        out["plane"] = fit
        if fit["tilt_deg"] > MAX_TILT_DEG:
            return fail("refit_tilt_too_large")
        bp = backproject_factory(n, d)
        corners3 = bp(quad)
        seam_px = np.array([seam_endpoints["seam_left"], seam_endpoints["seam_right"]],
                           dtype=np.float64)
        s3 = bp(seam_px)
        if not (np.all(np.isfinite(corners3)) and np.all(np.isfinite(s3))):
            return fail("backproject_failed")
        # in-plane frame: u along the seam, v across it
        u = s3[1] - s3[0]
        u = u - (u @ n) * n
        u = _unit(u)
        v = _unit(np.cross(n, u))
        origin = 0.5 * (s3[0] + s3[1])

        def to2(p):
            q = np.atleast_2d(p) - origin
            return np.stack([q @ u, q @ v], axis=1) * 1000.0          # mm

        def to3(c):
            c = np.atleast_2d(c) / 1000.0
            return origin + c[:, 0:1] * u + c[:, 1:2] * v

        c2 = to2(corners3)                                              # (4,2) mm
        ring_edges = [(i, (i + 1) % 4) for i in range(4)]
        side_idx = [0, 2] if axis == "lr" else [1, 3]
        end_idx = [1, 3] if axis == "lr" else [0, 2]
        # the seam endpoints must sit on the seam-end edges
        for sp in seam_px:
            dists = []
            for (a, b) in ring_edges:
                pa, pb = quad[a], quad[b]
                t = float(np.clip(((sp - pa) @ (pb - pa)) / max(float((pb - pa) @ (pb - pa)), 1e-9), 0, 1))
                dists.append(float(np.linalg.norm(sp - (pa + t * (pb - pa)))))
            near = sorted(range(4), key=lambda k: dists[k])[0]
            if near not in end_idx or dists[near] > 6.0:
                return fail("seam_not_on_end_edges")
        # side (seam-parallel) edges: A below, B above the gap
        sides = sorted(side_idx, key=lambda k: float(c2[list(ring_edges[k])][:, 1].mean()))
        eA, eB = sides
        pos = {}
        for name, k in (("A", eA), ("B", eB)):
            a, b = ring_edges[k]
            val, slope = _line_x(c2[a], c2[b], 0.0, 0)
            if val is None:
                return fail("degenerate_side_edge")
            pos[name] = val
        across_rgb = pos["B"] - pos["A"]
        gap_off = 0.5 * (pos["A"] + pos["B"])
        out["across_rgb_mm"] = round(across_rgb, 2)
        out["gap_from_centre_mm"] = round(gap_off, 2)
        if abs(across_rgb - across_mm) > MAX_ACROSS_DEV_MM:
            return fail("across_mismatch")
        if abs(gap_off) > MAX_GAP_OFF_CENTRE_MM:
            return fail("gap_not_central")
        # seam-end edges
        ends = sorted(end_idx, key=lambda k: float(c2[list(ring_edges[k])][:, 0].mean()))
        eL, eR = ends
        bs, ss = {}, {}
        for name, k in (("L", eL), ("R", eR)):
            a, b = ring_edges[k]
            val, slope = _line_x(c2[a], c2[b], 0.0, 1)       # u where v = 0
            if val is None:
                return fail("degenerate_end_edge")
            bs[name], ss[name] = val, slope
        length_rgb = bs["R"] - bs["L"]
        out["along_rgb_mm"] = round(length_rgb, 2)
        if abs(length_rgb - along_mm) > MAX_LEN_DEV_MM:
            return fail("length_mismatch")
        # the RGB quad itself must be near-rectangular (3-D opposite sides)
        sides3 = [float(np.linalg.norm(corners3[b] - corners3[a])) * 1000.0
                  for a, b in ring_edges]
        rect = (abs(sides3[0] - sides3[2]), abs(sides3[1] - sides3[3]))
        out["rgb_rect_diffs_mm"] = [round(rect[0], 2), round(rect[1], 2)]
        if max(rect) > MAX_RGB_RECT_DIFF_MM:
            return fail("rgb_quad_not_rectangular")
        half = across_mm / 2.0
        recentred = abs(length_rgb - along_mm) > LEN_TOL_MM
        new_b = dict(bs)
        if recentred:
            cen = 0.5 * (bs["L"] + bs["R"])
            new_b = {"L": cen - along_mm / 2.0, "R": cen + along_mm / 2.0}
        deltas = {
            "side_A": half + pos["A"],             # A is at -v: outward = -v
            "side_B": half - pos["B"],
            "end_L": bs["L"] - new_b["L"],         # L is at -u
            "end_R": new_b["R"] - bs["R"],
        }
        if (max(abs(deltas["side_A"]), abs(deltas["side_B"])) > MAX_SIDE_DELTA_MM
                or max(abs(deltas["end_L"]), abs(deltas["end_R"])) > MAX_END_DELTA_MM):
            return fail("delta_implausible", deltas_mm={k: round(x, 2) for k, x in deltas.items()})
        lines = {
            eA: (np.array([0.0, -half]), np.array([1.0, 0.0])),
            eB: (np.array([0.0, half]), np.array([1.0, 0.0])),
            eL: (np.array([new_b["L"], 0.0]), np.array([ss["L"], 1.0])),
            eR: (np.array([new_b["R"], 0.0]), np.array([ss["R"], 1.0])),
        }
        new2 = np.zeros((4, 2))
        for i in range(4):
            e_prev, e_next = (i - 1) % 4, i
            p = _intersect(lines[e_prev], lines[e_next])
            if p is None:
                return fail("corner_intersection_failed")
            new2[i] = p
        new3 = to3(new2)
        new_px = np.asarray(project(new3), dtype=np.float64).reshape(4, 2)
        w, h = image_wh
        if not (np.all(np.isfinite(new_px)) and np.all(new_px[:, 0] >= 0)
                and np.all(new_px[:, 0] < w) and np.all(new_px[:, 1] >= 0)
                and np.all(new_px[:, 1] < h)):
            return fail("corners_outside_image")
        seam_new = None
        if recentred:
            e2 = np.array([[new_b["L"], 0.0], [new_b["R"], 0.0]])
            sp = np.asarray(project(to3(e2)), dtype=np.float64).reshape(2, 2)
            seam_new = {"seam_left": sp[0].tolist(), "seam_right": sp[1].tolist()}
            # keep the left/right (image-x sorted) labels of the input
            if sp[0][0] > sp[1][0]:
                seam_new = {"seam_left": sp[1].tolist(), "seam_right": sp[0].tolist()}
        by_edge = {0: None, 1: None, 2: None, 3: None}
        by_edge[eA], by_edge[eB] = deltas["side_A"], deltas["side_B"]
        by_edge[eL], by_edge[eR] = deltas["end_L"], deltas["end_R"]
        names = ("back", "right", "front", "left")             # ring edge order
        out.update({
            "applied": True, "reason": "ok",
            "quad_px": new_px,
            "corners_xyz": new3,
            "seam_endpoints": seam_new,
            "recentred_ends": bool(recentred),
            "end_edge_angle_deg": {"L": round(math.degrees(math.atan(ss["L"])), 2),
                                   "R": round(math.degrees(math.atan(ss["R"])), 2)},
            "deltas_mm": {names[k]: round(float(x), 2) for k, x in by_edge.items()},
            "sides_mm": [round(float(np.linalg.norm(new3[b] - new3[a])) * 1000.0, 2)
                         for a, b in ring_edges],
            "known_mm": {"along_seam": along_mm, "across_seam": across_mm},
        })
        return out
    except Exception as exc:  # noqa: BLE001 — K1 never breaks the pipeline
        return fail(f"error({type(exc).__name__}: {exc})")
