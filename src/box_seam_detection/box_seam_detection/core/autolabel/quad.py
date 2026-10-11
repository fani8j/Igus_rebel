# Provenance: moved into box_seam_detection (2026-10-12) from rgbd_captures/scripts/autolabel/quad.py
# at commit 6df59943f23e0d551cdc448b47ffea9e95ece20a (working tree identical to HEAD, unmodified).
# Only imports/sys.path handling changed; logic unchanged.
"""Line-driven carton top-face quadrilateral fitting (pure OpenCV/numpy).

Strategy: the brown-cardboard mask gives ROI candidates, but mask contours
merge the top face with side faces, so the quad itself is fit from LSD line
segments inside the ROI:

3. Within each orientation, lateral (rho) clustering -> supported boundary
   lines. Every strong pair forms a lattice of quad hypotheses (the top
   face is often an INTERIOR pair when the mask merges top+side faces).
4. 2x2 line intersections -> quads, then strict validation (convex, area,
   angles, edge length, mask fill, per-edge LSD support).
"""

from __future__ import annotations

import cv2
import numpy as np

__all__ = [
    "order_corners",
    "quad_area",
    "interior_angles",
    "fit_quad_from_segments",
    "quad_edge_support",
    "mask_fill",
    "mask_edge_support",
    "find_crease",
    "crease_side",
    "validate_quad",
    "fit_quad_candidates",
]


def _line_from_rho_theta(rho: float, theta: float):
    """Normal-form line: n=(cos t, sin t), n.p = rho. Return (point, dir)."""
    n = np.array([np.cos(theta), np.sin(theta)])
    return rho * n, np.array([-n[1], n[0]])


def intersect_lines(p1, d1, p2, d2):
    A = np.stack([d1, -d2], axis=1)
    if abs(np.linalg.det(A)) < 1e-9:
        return None
    t, _ = np.linalg.solve(A, p2 - p1)
    return p1 + t * d1


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 distinct points as [TL, TR, BR, BL] (centroid-angle sort).

    The points are sorted by polar angle around their centroid (clockwise
    in image coordinates, where y grows downwards), then rotated so the
    legacy top-left corner — smallest x+y, tie-break smallest x — comes
    first. Unlike the old sums/differences heuristic, which assigned all
    four roles by independent extrema and could map two output corners to
    the same input point on strongly oblique quads, the circular sort can
    never duplicate a corner; the legacy start corner only preserves the
    existing rectification correspondence on inputs the old code ordered
    without duplication. Output is permutation-invariant: any ordering of
    the same 4 points yields the same result.

    Raises ValueError on anything that cannot form a quad: wrong point
    count, non-finite coordinates, or duplicate/coincident points.
    """
    try:
        arr = np.asarray(pts, dtype=np.float64).reshape(4, 2)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"order_corners: need exactly 4 (x, y) points, got {np.shape(pts)}"
        ) from exc
    if not bool(np.all(np.isfinite(arr))):
        raise ValueError("order_corners: corners must be finite")
    dists = np.linalg.norm(arr[:, None, :] - arr[None, :, :], axis=-1)
    rows, cols = np.triu_indices(4, 1)
    if bool((dists[rows, cols] < 1e-6).any()):
        raise ValueError("order_corners: duplicate corners cannot form a quad")
    centroid = arr.mean(axis=0)
    angles = np.arctan2(arr[:, 1] - centroid[1], arr[:, 0] - centroid[0])
    # Ascending polar angle is clockwise in image (y-down) coordinates for
    # any quad whose centroid lies strictly inside it (all convex quads).
    circular = arr[np.argsort(angles, kind="stable")]
    # Canonical start: legacy top-left (min x+y, then min x). Deterministic
    # for every input; matches the old correspondence wherever the old
    # heuristic produced four distinct corners.
    sums = circular[:, 0] + circular[:, 1]
    first = int(np.lexsort((circular[:, 0], sums))[0])
    circular = np.roll(circular, -first, axis=0)
    # Keep [TL, TR, BR, BL] clockwise (positive shoelace area in x-right /
    # y-down): the angle sort already yields this for convex quads; the
    # flip only normalizes concave inputs, which validate_quad rejects.
    signed = float(np.sum(circular[:, 0] * np.roll(circular[:, 1], -1)
                          - np.roll(circular[:, 0], -1) * circular[:, 1]))
    if signed < 0:
        circular = np.stack([circular[0], circular[3], circular[2], circular[1]])
    return circular


def quad_area(q: np.ndarray) -> float:
    q = np.asarray(q, dtype=np.float64).reshape(4, 2)
    return 0.5 * abs(float(np.sum(q[:, 0] * np.roll(q[:, 1], -1)
                                 - np.roll(q[:, 0], -1) * q[:, 1])))


def interior_angles(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4, 2)
    angs = []
    for i in range(4):
        a = q[i - 1] - q[i]
        b = q[(i + 1) % 4] - q[i]
        c = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        angs.append(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))
    return np.array(angs)


def _ang_diff(a: float, b: float) -> float:
    """Circular distance on [0, 180)."""
    d = abs(float(a) - float(b)) % 180.0
    return min(d, 180.0 - d)


def detect_segments(gray: np.ndarray):
    lsd = cv2.createLineSegmentDetector(0)
    lines, *_ = lsd.detect(gray)
    if lines is None:
        return []
    return [tuple(map(float, L[0])) for L in lines]  # x1,y1,x2,y2


def _seg_len(s) -> float:
    return float(np.hypot(s[2] - s[0], s[3] - s[1]))


def _seg_angle(s) -> float:
    return float(np.degrees(np.arctan2(s[3] - s[1], s[2] - s[0]))) % 180.0


def fit_quad_from_segments(segments: list, diag: float, qcfg: dict):
    """Fit candidate quads from segments. Returns (list_of_quad4x2 or None, debug dict)."""
    dbg: dict = {"n_segments": len(segments)}
    min_len = qcfg["min_seg_frac"] * diag
    segs = [s for s in segments if _seg_len(s) >= min_len]
    dbg["n_kept"] = len(segs)
    if len(segs) < 4:
        dbg["fail"] = f"too_few_segments({len(segs)})"
        return None, dbg
    segs.sort(key=_seg_len, reverse=True)
    angs = [_seg_angle(s) for s in segs]
    # Seed two orientations from the longest pair with enough separation.
    seed_a, seed_b = angs[0], None
    for a in angs[1:]:
        if _ang_diff(a, seed_a) >= qcfg["seed_angle_sep_deg"]:
            seed_b = a
            break
    if seed_b is None:
        dbg["fail"] = "single_orientation"
        return None, dbg
    groups: dict[str, list] = {"a": [], "b": []}
    for s, a in zip(segs, angs):
        groups["a" if _ang_diff(a, seed_a) < _ang_diff(a, seed_b) else "b"].append(s)
    if len(groups["a"]) < 2 or len(groups["b"]) < 2:
        dbg["fail"] = "orientation_group_too_small"
        return None, dbg
    # Boundary lines per orientation: rho clustering. Every supported cluster
    # is a candidate boundary (the top face is often an INTERIOR pair when
    # the mask merges top+side faces), so all strong pairs form a lattice.
    lines = {}
    for key in ("a", "b"):
        g = groups[key]
        # Circular mean on the doubled angle: orientations live on [0, 180),
        # so a naive mean of e.g. 5deg and 175deg wrongly gives 90deg.
        dbl = np.array([_seg_angle(s) for s in g], dtype=np.float64)
        mean_ang = float((np.degrees(np.arctan2(
            np.sin(np.radians(2.0 * dbl)).mean(),
            np.cos(np.radians(2.0 * dbl)).mean())) / 2.0) % 180.0)
        theta = np.radians(mean_ang) + np.pi / 2  # lateral normal
        n = np.array([np.cos(theta), np.sin(theta)])
        rhos = []
        for s in g:
            mid = np.array([(s[0] + s[2]) / 2.0, (s[1] + s[3]) / 2.0])
            rhos.append((float(mid @ n), _seg_len(s), s))
        tol = qcfg["rho_tol_frac"] * diag
        rhos.sort()
        clusters: list[list] = []
        for rho, ln, s in rhos:
            if clusters and rho - clusters[-1][-1][0] <= tol:
                clusters[-1].append((rho, ln, s))
            else:
                clusters.append([(rho, ln, s)])
        # Weighted mean rho per cluster; keep clusters with real support.
        total = sum(ln for _, ln, _ in rhos)
        kept = []
        for cl in clusters:
            w = sum(ln for _, ln, _ in cl)
            if w >= qcfg.get("min_cluster_support", 0.07) * total:
                kept.append((sum(r * ln for r, ln, _ in cl) / w, w))
        kept.sort(key=lambda t: -t[1])
        kept = kept[: qcfg.get("max_clusters_per_axis", 3)]
        kept.sort()
        if len(kept) < 2:
            dbg["fail"] = f"rho_clusters_too_few({key})"
            return None, dbg
        lines[key] = ([k[0] for k in kept], theta)
        dbg[f"rho_{key}"] = [round(k[0], 1) for k in kept]
    (rhos_a, th_a), (rhos_b, th_b) = lines["a"], lines["b"]
    dbg["n_lattice"] = len(rhos_a) * (len(rhos_a) - 1) // 2 * len(rhos_b) * (len(rhos_b) - 1) // 2
    quads = []
    for i in range(len(rhos_a)):
        for j in range(i + 1, len(rhos_a)):
            for k in range(len(rhos_b)):
                for m in range(k + 1, len(rhos_b)):
                    pa0, da = _line_from_rho_theta(rhos_a[i], th_a)
                    pa1, _ = _line_from_rho_theta(rhos_a[j], th_a)
                    pb0, db = _line_from_rho_theta(rhos_b[k], th_b)
                    pb1, _ = _line_from_rho_theta(rhos_b[m], th_b)
                    corners = [intersect_lines(pa0, da, pb0, db), intersect_lines(pa0, da, pb1, db),
                               intersect_lines(pa1, da, pb0, db), intersect_lines(pa1, da, pb1, db)]
                    if any(c is None for c in corners):
                        continue
                    try:
                        quads.append(order_corners(np.stack(corners)))
                    except ValueError:
                        # Coincident intersections cannot form a quad; skip
                        # this lattice cell, never emit duplicated corners.
                        continue
    if not quads:
        dbg["fail"] = "parallel_boundaries"
        return None, dbg
    dbg["lattice"] = [[round(float(x), 1), round(float(y), 1)] for q in quads for x, y in q]
    return quads, dbg


def quad_edge_support(quad: np.ndarray, segments: list, diag: float, qcfg: dict) -> list[float]:
    """Fraction of each edge length covered by collinear segments."""
    quad = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    ang_tol = qcfg["angle_tol_deg"]
    dist_tol = qcfg["dist_tol_frac"] * diag
    support = []
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        elen = float(np.linalg.norm(b - a)) + 1e-9
        eang = float(np.degrees(np.arctan2(b[1] - a[1], b[0] - a[0]))) % 180.0
        n = np.array([-(b[1] - a[1]), b[0] - a[0]]) / elen
        covered = 0.0
        for s in segments:
            if _ang_diff(_seg_angle(s), eang) > ang_tol:
                continue
            p = np.array([s[0], s[1]])
            q2 = np.array([s[2], s[3]])
            if abs((p - a) @ n) > dist_tol or abs((q2 - a) @ n) > dist_tol:
                continue
            # project onto edge, clip to [0, elen]
            e = (b - a) / elen
            t0 = np.clip((p - a) @ e, 0.0, elen)
            t1 = np.clip((q2 - a) @ e, 0.0, elen)
            covered += abs(t1 - t0)
        support.append(min(1.0, covered / elen))
    return support


def mask_fill(quad: np.ndarray, mask: np.ndarray) -> float:
    """Mean mask value inside the quad (mask need not match quad coords if a
    same-size full-image mask is passed with image-space quad)."""
    m = np.zeros_like(mask)
    cv2.fillConvexPoly(m, np.asarray(quad, dtype=np.int32).reshape(-1, 1, 2), 255)
    inside = m > 0
    if inside.sum() == 0:
        return 0.0
    return float(mask[inside].mean() / 255.0)

def mask_edge_support(quad: np.ndarray, mask_dist: np.ndarray, diag: float,
                      qcfg: dict) -> list[float]:
    """Fraction of each edge running near the cardboard-mask boundary.

    mask_dist is a distance transform of the mask boundary (same size as the
    mask, image coordinates). Complements LSD evidence where a silhouette
    edge is fragmented or partly occluded.
    """
    quad = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    tol = qcfg["dist_tol_frac"] * diag
    H, W = mask_dist.shape[:2]
    out = []
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        elen = float(np.linalg.norm(b - a))
        n = max(8, int(elen / 4))
        ts = np.linspace(0.0, 1.0, n)
        xs = np.clip((a[0] + (b[0] - a[0]) * ts).round().astype(int), 0, W - 1)
        ys = np.clip((a[1] + (b[1] - a[1]) * ts).round().astype(int), 0, H - 1)
        out.append(float((mask_dist[ys, xs] <= tol).mean()))
    return out

def validate_quad(quad, segments, mask, img_wh: tuple, diag: float, qcfg: dict,
                  mask_dist=None):
    """Return (ok, reasons list, metrics dict).

    Per-edge support is the stronger of LSD collinear coverage and
    mask-boundary proximity (complementary cues: a silhouette edge may be
    fragmented for LSD yet crisp in the color mask, and vice versa).
    """
    quad = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    reasons: list[str] = []
    W, H = img_wh
    area = quad_area(quad)
    area_frac = area / (W * H)
    angs = interior_angles(quad)
    edges = [float(np.linalg.norm(quad[(i + 1) % 4] - quad[i])) for i in range(4)]
    lsd = quad_edge_support(quad, segments, diag, qcfg)
    msk = mask_edge_support(quad, mask_dist, diag, qcfg) if mask_dist is not None else [0.0] * 4
    support = [max(a, b) for a, b in zip(lsd, msk)]
    fill = mask_fill(quad, mask)
    metrics = {
        "area_frac": round(area_frac, 4),
        "angles_deg": [round(float(a), 1) for a in angs],
        "edge_len": [round(e, 1) for e in edges],
        "edge_support": [round(float(s), 3) for s in support],
        "edge_support_lsd": [round(float(s), 3) for s in lsd],
        "edge_support_mask": [round(float(s), 3) for s in msk],
        "mask_fill": round(fill, 3),
    }
    if not bool(cv2.isContourConvex(quad.astype(np.float32))):
        reasons.append("non_convex")
    if not (qcfg["min_quad_frac"] <= area_frac <= qcfg["max_quad_frac"]):
        reasons.append(f"area_frac_out_of_range({area_frac:.3f})")
    if bool(((angs < qcfg["min_angle_deg"]) | (angs > qcfg["max_angle_deg"])).any()):
        reasons.append("skewed_angles")
    if min(edges) < qcfg["min_edge_frac"] * diag:
        reasons.append("edge_too_short")
    if min(support) < qcfg["min_edge_support"]:
        reasons.append("weak_edge_support")
    if fill < qcfg["min_mask_fill"]:
        reasons.append("low_mask_fill")
    xs, ys = quad[:, 0], quad[:, 1]
    if xs.min() < 0 or ys.min() < 0 or xs.max() >= W or ys.max() >= H:
        reasons.append("out_of_bounds")
    return (len(reasons) == 0, reasons, metrics)


def find_crease(merged_quad, segments, diag: float, qcfg: dict):
    """Top/side crease inside a merged two-face quad. Returns a segment or None.

    Fragments are merged collinearly first (a crease shadow often breaks
    into short LSD pieces). A merged line qualifies when it spans most of
    the quad along its own direction, lies strictly inside, and divides the
    quad into two substantial parts. No orientation is assumed.
    """
    mq = np.asarray(merged_quad, dtype=np.float64).reshape(4, 2)
    qf = mq.astype(np.float32)
    max_ext = max(float(np.linalg.norm(mq[(i + 1) % 4] - mq[i])) for i in range(4))
    # interior sample grid (candidate-independent): coarse but decisive
    bx0, by0 = float(mq[:, 0].min()), float(mq[:, 1].min())
    bx1, by1 = float(mq[:, 0].max()), float(mq[:, 1].max())
    step = max(4.0, (bx1 - bx0) / 30.0, (by1 - by0) / 30.0)
    inside_pts = [(float(gx), float(gy))
                  for gy in np.arange(by0, by1 + 1e-6, step)
                  for gx in np.arange(bx0, bx1 + 1e-6, step)
                  if cv2.pointPolygonTest(qf, (float(gx), float(gy)), False) > 0]
    if len(inside_pts) < 20:
        return None
    grid = np.array(inside_pts)
    # merge collinear fragments sharing (theta, rho)
    groups: list[dict] = []
    for s in segments:
        x1, y1, x2, y2 = s
        if float(np.hypot(x2 - x1, y2 - y1)) < 0.05 * diag:
            continue
        ang = _seg_angle(s)
        th = np.radians(ang) + np.pi / 2
        n = np.array([np.cos(th), np.sin(th)])
        rho = float(np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0]) @ n)
        placed = False
        for g in groups:
            if (_ang_diff(ang, g["ang"]) <= 6.0
                    and abs(rho - g["rho"]) <= 0.03 * diag):
                g["segs"].append(s)
                placed = True
                break
        if not placed:
            groups.append({"ang": ang, "rho": rho, "segs": [s]})
    tol_px = float(qcfg.get("crease_on_tol_px", 6.0))
    best = None
    for g in groups:
        d = np.array([np.cos(np.radians(g["ang"])), np.sin(np.radians(g["ang"]))])
        t0 = np.array([g["segs"][0][0], g["segs"][0][1]])
        lo = hi = 0.0
        for s in g["segs"]:
            for px, py in ((s[0], s[1]), (s[2], s[3])):
                t = float((np.array([px, py]) - t0) @ d)
                lo, hi = min(lo, t), max(hi, t)
        proj_q = [float((v - t0) @ d) for v in mq]
        qlo, qhi = min(proj_q), max(proj_q)
        overlap = min(hi, qhi) - max(lo, qlo)
        if overlap / max(qhi - qlo, 1e-9) < float(qcfg.get("crease_min_span", 0.6)):
            continue
        a, b = t0 + d * lo, t0 + d * hi
        mid = (a + b) / 2.0
        if cv2.pointPolygonTest(qf, (float(mid[0]), float(mid[1])), False) <= 2.0:
            continue
        if (cv2.pointPolygonTest(qf, (float(a[0]), float(a[1])), True) < -tol_px
                or cv2.pointPolygonTest(qf, (float(b[0]), float(b[1])), True) < -tol_px):
            continue
        n = np.array([-d[1], d[0]])
        side = (grid - a) @ n
        frac = min(float((side > 0).mean()), float((side <= 0).mean()))
        if frac < float(qcfg.get("crease_min_split", 0.25)):
            continue
        if best is None or (hi - lo) > best[0]:
            best = (hi - lo, [float(a[0]), float(a[1]), float(b[0]), float(b[1])])
    return None if best is None else best[1]


def crease_side(crease, pt: tuple | list) -> float:
    """Signed distance of pt from the crease line (image coords)."""
    p = np.array([crease[0], crease[1]], dtype=float)
    q = np.array([crease[2], crease[3]], dtype=float)
    d = q - p
    n = np.array([-d[1], d[0]]) / (float(np.linalg.norm(d)) + 1e-12)
    return float((np.asarray(pt, dtype=float) - p) @ n)


def fit_quad_candidates(gray_full: np.ndarray, mask: np.ndarray, cfg: dict, seg_fn=None):
    """Yield (quad, roi segments in image coords, metrics, dbg) per lattice quad.

    dbg carries roi/component info plus per-quad "valid" and "reasons".
    A (None, segs, {}, dbg) yield reports an ROI where no quad fit at all.
    seg_fn(roi_gray) overrides the built-in OpenCV LSD detector (DeepLSD).
    """
    qcfg = cfg["quad"]
    scfg = cfg["segmentation"]
    H, W = gray_full.shape[:2]
    diag = float(np.hypot(W, H))
    ncomp, lab, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    bin8 = ((mask > 0).astype(np.uint8)) * 255
    medge = cv2.Canny(bin8, 50, 150)
    mask_dist = cv2.distanceTransform(255 - medge, cv2.DIST_L2, 3)
    comps = []
    for i in range(1, ncomp):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if area / (W * H) >= scfg["min_component_frac"]:
            comps.append((area, i))
    comps.sort(reverse=True)
    comps = comps[: scfg["max_candidates"]]


    for _, ci in comps:
        ys, xs = np.nonzero(lab == ci)
        # Truncation gate: a component bleeding off the frame edge is a cut
        # face — its missing corners cannot be honestly completed, so the
        # whole ROI is rejected (never fabricate off-frame points).
        c_side = int(scfg.get("truncation_contact_side_px", 30))
        c_bot = int(scfg.get("truncation_contact_bottom_px", 150))
        truncated = (int((xs == 0).sum()) >= c_side
                     or int((xs == W - 1).sum()) >= c_side
                     or int((ys == 0).sum()) >= c_side
                     or int((ys == H - 1).sum()) >= c_bot)
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        pad = qcfg["roi_pad"]
        x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
        x1, y1 = min(W - 1, x1 + pad), min(H - 1, y1 + pad)
        roi = gray_full[y0: y1 + 1, x0: x1 + 1]
        segs_roi = (seg_fn or detect_segments)(roi)
        segs = [(x1_ + x0, y1_ + y0, x2_ + x0, y2_ + y0) for (x1_, y1_, x2_, y2_) in segs_roi]
        found, dbg = fit_quad_from_segments(segs, float(np.hypot(x1 - x0, y1 - y0)), qcfg)
        dbg["component_frac"] = round(float((lab == ci).sum()) / (W * H), 4)
        dbg["roi"] = [x0, y0, x1, y1]
        dbg["truncated"] = bool(truncated)
        if found is None:
            yield None, segs, {}, dbg
            continue
        # Two-face context: the largest lattice quad approximates the whole
        # visible carton; its interior crease tells top face from side face.
        merged = max(found, key=quad_area)
        crease = find_crease(merged, segs, diag, qcfg)
        dbg["crease"] = crease
        dbg["merged_top"] = merged[np.argmin(merged[:, 1])].round(1).tolist()
        scored = []
        for quad in found:
            ok, reasons, metrics = validate_quad(quad, segs, mask, (W, H), diag, qcfg, mask_dist)
            scored.append((quad_area(quad), quad, ok, reasons, metrics))
        scored.sort(key=lambda t: -t[0])
        for _, quad, ok, reasons, metrics in scored:
            d = dict(dbg)
            d["valid"] = ok
            d["reasons"] = reasons
            yield quad, segs, metrics, d
