"""Top-face depth diagnostics of one detection (2026-10-12): D1 per-edge depth
support (edge_support), F4 multiple / touching boxes evidence, and the shadow
known-size fit + height-band components (never used for geometry).

One height grid of the neighbourhood of the quad (height above the top plane,
:func:`autolabel.depth_top.height_grid`) feeds all four. Everything is recorded
under ``record.top_face.depth_diag``; the only consumers are cut rules / reject
reasons that are enabled per profile in the detector. ``run`` never raises.
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from . import edge_support as _es
from . import known_size_fit as _ksf
from .autolabel import depth_top as _depth

#: a component is "box-sized" when its area is at least this share of the quad's
BOX_SIZED_FRAC = 0.40
#: depth step inside the top region: off-plane blob of at least this share of the quad
STEP_AREA_FRAC = 0.03
STEP_MIN_MM = 12.0
STEP_NOISE_K = 3.5
CLOSE_MM = 8.0           # closing size for the components: bridges the flap gap and dropouts
BAND_MIN_MM = 8.0
BAND_NOISE_K = 2.5


def _px_area_mm2(cam: dict, n: np.ndarray, d: float, uv: np.ndarray) -> float | None:
    fx, fy = float(cam["fx"]), float(cam["fy"])
    r = np.array([(uv[0] - float(cam["cx"])) / fx, (uv[1] - float(cam["cy"])) / fy, 1.0])
    den = float(n @ r)
    if abs(den) < 1e-6:
        return None
    z = -float(d) / den
    if not np.isfinite(z) or z <= 0:
        return None
    return z * z / (fx * fy * abs(den)) * 1e6


def _quad_area_mm2(corners3: np.ndarray, n: np.ndarray) -> float:
    c = corners3 - corners3.mean(axis=0)
    a = np.zeros(3)
    for i in range(4):
        a += np.cross(c[i], c[(i + 1) % 4])
    return abs(float(0.5 * (a @ n))) * 1e6


def _plane_basis(n: np.ndarray):
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = a - (a @ n) * n
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(n, e1)


def _quad_roi(quad: np.ndarray, image_wh: tuple) -> list:
    """Image window around a quad (same margin as :func:`run`)."""
    w, h = image_wh
    size = float(max(np.ptp(quad[:, 0]), np.ptp(quad[:, 1])))
    margin = int(max(100, 0.6 * size))
    return [int(max(0, np.floor(quad[:, 0].min() - margin))),
            int(max(0, np.floor(quad[:, 1].min() - margin))),
            int(min(w, np.ceil(quad[:, 0].max() + margin))),
            int(min(h, np.ceil(quad[:, 1].max() + margin)))]


def edge_check(quad_px: np.ndarray, plane_nd: tuple, rms_mm: float | None,
               xyz: np.ndarray, proj: tuple, backproject, image_wh: tuple) -> dict:
    """Only the D1 per-edge depth support of a ring-ordered quad (what
    :func:`run` records under ``edge_support``, with the same window and
    thresholds) plus its ``unsupported`` verdict (truncated edges). Used by
    the truncation recovery before the seam stage. Never raises: ``{"error":
    ...}`` (no ``unsupported`` key) on failure."""
    try:
        quad = np.asarray(quad_px, dtype=np.float64).reshape(4, 2)
        n, d = plane_nd
        rms_mm = (float(rms_mm) if rms_mm is not None and np.isfinite(rms_mm) else 0.0)
        roi = _quad_roi(quad, image_wh)
        grid = _depth.height_grid(xyz, proj, n, d, roi)
        cam_foot = -float(d) * np.asarray(n)
        out = _es.compute(quad, (n, d), grid, roi, backproject, rms_mm, cam_foot)
        out["unsupported"] = _es.decide(out)
        return out
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


#: coplanar-region growth (truncation recovery)
GROW_OPEN_PX = 9          # opening that removes thin fringes bridging to neighbours
GROW_MIN_SEED_FRAC = 0.30 # the chosen component must cover this share of the seed quad


def grow_coplanar_region(quad_px: np.ndarray, plane_nd: tuple, rms_mm: float | None,
                         xyz: np.ndarray, proj: tuple, cam: dict, image_wh: tuple,
                         band_mm: float | None = None) -> tuple:
    """:func:`grow_coplanar_region_ex` without the region mask: ``(hull_mask, info)``."""
    hull, _region, info = grow_coplanar_region_ex(quad_px, plane_nd, rms_mm, xyz, proj,
                                                  cam, image_wh, band_mm=band_mm)
    return hull, info


def grow_coplanar_region_ex(quad_px: np.ndarray, plane_nd: tuple, rms_mm: float | None,
                            xyz: np.ndarray, proj: tuple, cam: dict, image_wh: tuple,
                            band_mm: float | None = None) -> tuple:
    """The connected coplanar region of the top plane that the quad (a seed)
    lies in: pixels within ``band`` of the plane (``band`` = max(``BAND_MIN_MM``,
    ``BAND_NOISE_K`` rms, ``band_mm``) the depth cross-check's own band), closed
    over the flap gap / dropouts (``CLOSE_MM``), 8-connected, the component under the
    seed, cleaned by an opening and holes filled.

    Returns ``(hull_mask, region_mask, info)``: ``hull_mask`` = full-image uint8
    convex hull of the cleaned region (None when there is none), ``region_mask`` =
    the cleaned region itself (full-image uint8, not convexified; None with the
    hull), ``info`` = JSON-able details (``reason`` when None). Never raises."""
    info: dict = {}
    try:
        quad = np.asarray(quad_px, dtype=np.float64).reshape(4, 2)
        n, d = plane_nd
        w, h = image_wh
        rms = (float(rms_mm) if rms_mm is not None and np.isfinite(rms_mm) else 0.0)
        band = max(BAND_MIN_MM, BAND_NOISE_K * rms,
                   float(band_mm) if band_mm is not None else 0.0)
        info["band_mm"] = round(band, 1)
        roi = _quad_roi(quad, image_wh)
        grid = _depth.height_grid(xyz, proj, n, d, roi)
        quad_area = _quad_area_mm2(np.asarray(backproject_quad(quad, n, d, cam)), n)
        pxa0 = _px_area_mm2(cam, n, d, quad.mean(axis=0)) or 1.0
        close_px = int(max(3, round(CLOSE_MM / np.sqrt(pxa0)))) | 1
        comps = _depth.height_band_components(grid, roi, cam, n, d, band,
                                              min_area_mm2=0.05 * quad_area,
                                              close_px=close_px)
        labels = comps["labels"]
        seed = np.zeros(labels.shape, np.uint8)
        cv2.fillPoly(seed, [np.rint(quad - [roi[0], roi[1]]).astype(np.int32)], 1)
        sel = labels[seed > 0]
        sel = sel[sel > 0]
        if sel.size == 0:
            info["reason"] = "no_component_under_seed"
            return None, None, info
        lab = int(np.bincount(sel).argmax())
        seed_frac = float(sel[sel == lab].size) / float(max(1, int(seed.sum())))
        info["seed_coverage"] = round(seed_frac, 3)
        if seed_frac < GROW_MIN_SEED_FRAC:
            info["reason"] = f"seed_coverage_low({seed_frac:.2f})"
            return None, None, info
        comp = (labels == lab).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(comp)
        cv2.drawContours(filled, cnts, -1, 1, thickness=cv2.FILLED)
        filled = cv2.morphologyEx(
            filled, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (GROW_OPEN_PX, GROW_OPEN_PX)))
        n_cc, lab2, st, _c = cv2.connectedComponentsWithStats(filled, 8)
        if n_cc < 2:
            info["reason"] = "no_component_after_open"
            return None, None, info
        # the component with the most seed overlap (not simply the largest)
        ov = [int(((lab2 == k) & (seed > 0)).sum()) for k in range(1, n_cc)]
        best = 1 + int(np.argmax(ov))
        cnts, _ = cv2.findContours((lab2 == best).astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            info["reason"] = "no_contour"
            return None, None, info
        hull = cv2.convexHull(max(cnts, key=cv2.contourArea)) + np.array([[roi[0], roi[1]]])
        mask = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(mask, hull.reshape(-1, 2).astype(np.int32), 1)
        info["hull_area_px"] = int(mask.sum())
        region = np.zeros((h, w), np.uint8)
        region[roi[1]:roi[3], roi[0]:roi[2]] = (lab2 == best).astype(np.uint8)
        return mask, region, info
    except Exception as exc:  # noqa: BLE001
        info["reason"] = f"error({type(exc).__name__}: {exc})"
        return None, None, info


#: anchored rectangle (truncation recovery)
ANCHOR_STRIP_MARGIN = 0.15      # the extent of a truncated edge is read from the middle
                                # 70 % of the span (the corners of the grown region are chopped)
ANCHOR_EXTENT_PCT = 99.9        # outward extent percentile of the region points in that strip
                                # (the in-band region ends ~5-10 mm short of the lid edge at the
                                # 99.5th, at the lid edge within a few mm at the 99.9th)
ANCHOR_MIN_EXTENSION_MM = 15.0  # a truncated edge must move out by at least this much
ANCHOR_MAX_TILT_DEG = 8.0       # supported edges must be this close to parallel / perpendicular
ANCHOR_MIN_COVERAGE = 0.60      # the grown region must fill this share of the fitted rectangle
ANCHOR_MIN_INSIDE = 0.90        # ... and this share of the region must lie inside it


def anchored_rect_quad(quad_px: np.ndarray, truncated: list, plane_nd: tuple,
                       region_mask: np.ndarray, cam: dict, image_wh: tuple) -> tuple:
    """Rectangle in the top plane for a truncated top: keep what the quad measured
    reliably, move only the truncated edge(s) out to where the grown coplanar
    region ends.

    ``quad_px`` = the ring-ordered truncated quad, ``truncated`` = its truncated
    ring edges (edge ``i`` joins ring corners ``i`` and ``i + 1``, as
    :func:`edge_check`), ``region_mask`` = the cleaned grown region
    (:func:`grow_coplanar_region_ex`). The plane axes come from the LONGEST
    supported (not truncated) edge, the anchor: the other supported edges are
    snapped square to it. A supported edge sits at the mean of its two corners
    (a half-top's short side edges are only ~140 mm long: their own direction is
    off by several degrees, the position of their middle is not), a truncated
    edge at the ``ANCHOR_EXTENT_PCT`` outward extent of the region in the middle
    of the span. Unconstrained fits to the region's hull skew when a corner of
    the region is chopped (a lifted flap corner leaves the band); this one
    cannot.

    Fail-closed: ``(None, info)`` with ``info["reason"]`` unless the quad is one
    (or two adjacent) truncated edge(s), the supported edges are square to the
    anchor within ``ANCHOR_MAX_TILT_DEG``, every truncated edge moves out by at
    least ``ANCHOR_MIN_EXTENSION_MM`` and the region fills the rectangle
    (``ANCHOR_MIN_COVERAGE``) without spilling out of it (``ANCHOR_MIN_INSIDE``).
    Returns ``(quad (4,2) px in the input ring order | None, info)``; never raises.
    The quad is only a coarse candidate: edge refine and every normal gate still
    run on it."""
    info: dict = {}
    try:
        quad = np.asarray(quad_px, dtype=np.float64).reshape(4, 2)
        n = np.asarray(plane_nd[0], dtype=np.float64)
        d = float(plane_nd[1])
        w, h = image_wh
        tr = sorted({int(i) % 4 for i in truncated})
        info["truncated_edges"] = tr
        if not tr or len(tr) > 2 or (len(tr) == 2 and (tr[1] - tr[0]) % 2 == 0):
            info["reason"] = "truncation_pattern_unsupported"
            return None, info
        if region_mask is None or int(np.count_nonzero(region_mask)) < 1000:
            info["reason"] = "no_region"
            return None, info
        P = np.asarray(backproject_quad(quad, n, d, cam), dtype=np.float64)
        if not np.all(np.isfinite(P)):
            info["reason"] = "corners_not_on_plane"
            return None, info
        length = [float(np.linalg.norm(P[(i + 1) % 4] - P[i])) for i in range(4)]
        anchor = max((i for i in range(4) if i not in tr), key=lambda i: length[i])
        a = P[(anchor + 1) % 4] - P[anchor]
        a = a - (a @ n) * n
        a = a / np.linalg.norm(a)
        b = np.cross(n, a)
        o = P[anchor]

        def uv(Q):
            R = (np.asarray(Q, dtype=np.float64) - o) * 1000.0
            return np.stack([R @ a, R @ b], axis=1)

        c4 = uv(P)                                   # corner (u, v) in mm
        # edges parallel to the anchor: anchor, anchor + 2 (position = v);
        # perpendicular: anchor + 1, anchor + 3 (position = u)
        pos: dict = {}
        tilt_deg: dict = {}
        for i in range(4):
            e0, e1 = c4[i], c4[(i + 1) % 4]
            par = (i - anchor) % 2 == 0
            if i in tr:
                continue
            along = abs(e1[0] - e0[0]) if par else abs(e1[1] - e0[1])
            across = abs(e1[1] - e0[1]) if par else abs(e1[0] - e0[0])
            tilt_deg[i] = float(np.degrees(np.arctan2(across, max(along, 1e-6))))
            pos[i] = float(0.5 * (e0[1] + e1[1])) if par else float(0.5 * (e0[0] + e1[0]))
        info["anchor_edge"] = anchor
        info["supported_tilt_deg"] = {str(k): round(v, 2) for k, v in tilt_deg.items()}
        if max(tilt_deg.values()) > ANCHOR_MAX_TILT_DEG:
            info["reason"] = f"supported_edge_not_square({max(tilt_deg.values()):.1f}deg)"
            return None, info
        # region pixels -> (u, v) mm on the plane
        ys, xs = np.nonzero(region_mask)
        r = np.stack([(xs - float(cam["cx"])) / float(cam["fx"]),
                      (ys - float(cam["cy"])) / float(cam["fy"]),
                      np.ones(xs.size)], axis=1)
        den = r @ n
        okp = np.abs(den) > 1e-6
        t = np.zeros(xs.size)
        t[okp] = -d / den[okp]
        okp &= t > 0
        S = uv(r[okp] * t[okp, None])
        if S.shape[0] < 1000:
            info["reason"] = "no_region"
            return None, info
        # side / end positions of the rectangle: bounds[axis] = [lo, hi]
        # (axis 0 = u: the perpendicular edges, axis 1 = v: the parallel edges)
        bound = {0: [None, None], 1: [None, None]}
        cu = c4.mean(axis=0)
        side_of: dict = {}
        for i in range(4):
            par = (i - anchor) % 2 == 0
            ax = 1 if par else 0
            hi = (0.5 * (c4[i][ax] + c4[(i + 1) % 4][ax])) > cu[ax]
            side_of[i] = (ax, 1 if hi else 0)
            if i in pos:
                bound[ax][1 if hi else 0] = pos[i]
        for i in tr:                                 # provisional: the quad's own edge
            ax, sd = side_of[i]
            bound[ax][sd] = float(0.5 * (c4[i][ax] + c4[(i + 1) % 4][ax]))
        ext: dict = {}
        orig_bound = {k: list(v) for k, v in bound.items()}
        for i in tr:
            ax, sd = side_of[i]
            other = 1 - ax
            lo_o, hi_o = bound[other]
            wspan = hi_o - lo_o
            m = ANCHOR_STRIP_MARGIN * wspan
            strip = S[(S[:, other] > lo_o + m) & (S[:, other] < hi_o - m)][:, ax]
            if strip.size < 500:
                info["reason"] = "region_strip_too_small"
                return None, info
            e = float(np.percentile(strip, ANCHOR_EXTENT_PCT if sd == 1
                                    else 100.0 - ANCHOR_EXTENT_PCT))
            moved = (e - orig_bound[ax][sd]) if sd == 1 else (orig_bound[ax][sd] - e)
            ext[i] = round(float(moved), 1)
            if moved < ANCHOR_MIN_EXTENSION_MM:
                info["reason"] = f"region_does_not_extend_{i}({moved:.1f}mm)"
                info["extension_mm"] = {str(k): v for k, v in ext.items()}
                return None, info
            bound[ax][sd] = e
        info["extension_mm"] = {str(k): v for k, v in ext.items()}
        (u_lo, u_hi), (v_lo, v_hi) = bound[0], bound[1]
        if not (u_hi > u_lo and v_hi > v_lo):
            info["reason"] = "degenerate_rectangle"
            return None, info
        inside = ((S[:, 0] >= u_lo) & (S[:, 0] <= u_hi) & (S[:, 1] >= v_lo) & (S[:, 1] <= v_hi))
        inside_frac = float(inside.mean())
        # coverage = region area inside the rectangle / rectangle area (pixel footprints on
        # the plane are near-uniform over the top: count the 5 mm cells hit instead)
        cell = 5.0
        cells = np.unique(np.floor((S[inside] - [u_lo, v_lo]) / cell).astype(np.int64), axis=0)
        cover = float(cells.shape[0]) * cell * cell / ((u_hi - u_lo) * (v_hi - v_lo))
        info["rect_mm"] = [round(u_hi - u_lo, 1), round(v_hi - v_lo, 1)]
        info["region_inside_frac"] = round(inside_frac, 3)
        info["region_coverage"] = round(min(cover, 1.0), 3)
        if inside_frac < ANCHOR_MIN_INSIDE:
            info["reason"] = f"region_spills_out_of_rectangle({inside_frac:.2f})"
            return None, info
        if cover < ANCHOR_MIN_COVERAGE:
            info["reason"] = f"region_does_not_fill_rectangle({cover:.2f})"
            return None, info
        # corners in the input ring order: each takes its (u, v) side from the original corner
        out = np.zeros((4, 2))
        for k in range(4):
            uk = u_hi if c4[k][0] > cu[0] else u_lo
            vk = v_hi if c4[k][1] > cu[1] else v_lo
            Pk = o + (uk / 1000.0) * a + (vk / 1000.0) * b
            if Pk[2] <= 0:
                info["reason"] = "corner_behind_camera"
                return None, info
            out[k] = [float(cam["fx"]) * Pk[0] / Pk[2] + float(cam["cx"]),
                      float(cam["fy"]) * Pk[1] / Pk[2] + float(cam["cy"])]
        if (not np.all(np.isfinite(out)) or out[:, 0].min() < 0 or out[:, 1].min() < 0
                or out[:, 0].max() > w - 1 or out[:, 1].max() > h - 1):
            info["reason"] = "rectangle_outside_image"
            return None, info
        return out, info
    except Exception as exc:  # noqa: BLE001
        info["reason"] = f"error({type(exc).__name__}: {exc})"
        return None, info


def backproject_quad(quad: np.ndarray, n: np.ndarray, d: float, cam: dict) -> np.ndarray:
    """Camera-frame points where the pixel rays of ``quad`` meet the plane."""
    fx, fy = float(cam["fx"]), float(cam["fy"])
    out = []
    for u, v in np.asarray(quad, dtype=np.float64).reshape(-1, 2):
        r = np.array([(u - float(cam["cx"])) / fx, (v - float(cam["cy"])) / fy, 1.0])
        out.append(-float(d) / float(np.asarray(n) @ r) * r)
    return np.asarray(out)


def run(quad_px: np.ndarray, corners_xyz: np.ndarray | None, plane_nd: tuple,
        rms_mm: float | None, xyz: np.ndarray, proj: tuple, cam: dict,
        backproject, image_wh: tuple, known_dims_mm: tuple | None,
        profile_name: str | None = None) -> dict:
    """Diagnostics for a ring-ordered quad on the plane ``plane_nd``.

    ``known_dims_mm`` = (long, short) of the box for the shadow fit (None =
    skipped). Returns a JSON-able dict (see keys below) or ``{"error": ...}``.
    """
    t0 = time.perf_counter()
    try:
        quad = np.asarray(quad_px, dtype=np.float64).reshape(4, 2)
        n, d = plane_nd
        rms_mm = (float(rms_mm) if rms_mm is not None and np.isfinite(rms_mm) else 0.0)
        w, h = image_wh
        size = float(max(np.ptp(quad[:, 0]), np.ptp(quad[:, 1])))
        margin = int(max(100, 0.6 * size))
        roi = [int(max(0, np.floor(quad[:, 0].min() - margin))),
               int(max(0, np.floor(quad[:, 1].min() - margin))),
               int(min(w, np.ceil(quad[:, 0].max() + margin))),
               int(min(h, np.ceil(quad[:, 1].max() + margin)))]
        grid, rows = _depth.height_grid(xyz, proj, n, d, roi, return_index=True)
        cam_foot = -float(d) * np.asarray(n)
        out: dict = {"profile": profile_name, "roi": roi}
        out["edge_support"] = _es.compute(quad, (n, d), grid, roi, backproject,
                                          rms_mm, cam_foot)
        out["edge_support"]["unsupported"] = _es.decide(out["edge_support"])

        # ---- F4 multiple boxes / depth step + height-band components --------
        rms = float(rms_mm or 0.0)
        band = max(BAND_MIN_MM, BAND_NOISE_K * rms)
        c3 = (np.asarray(corners_xyz, dtype=np.float64)
              if corners_xyz is not None and np.all(np.isfinite(corners_xyz))
              else backproject(quad))
        quad_area = _quad_area_mm2(c3, n)
        pxa0 = _px_area_mm2(cam, n, d, quad.mean(axis=0)) or 1.0
        close_px = int(max(3, round(CLOSE_MM / np.sqrt(pxa0)))) | 1
        comps = _depth.height_band_components(grid, roi, cam, n, d, band,
                                              min_area_mm2=0.05 * quad_area,
                                              close_px=close_px)
        labels = comps["labels"]
        cen = quad.mean(axis=0)
        # own component = the one under the middle of the quad (majority in the central 50 %)
        q50 = cen + (quad - cen) * 0.5
        m50 = np.zeros(labels.shape, np.uint8)
        cv2.fillPoly(m50, [np.rint(q50 - [roi[0], roi[1]]).astype(np.int32)], 1)
        sel = labels[m50 > 0]
        sel = sel[sel > 0]
        own_label = int(np.bincount(sel).argmax()) if sel.size else 0
        comp_list = []
        own = None
        for c in comps["components"]:
            ratio = c["area_mm2"] / quad_area if quad_area > 0 else None
            rec = {"area_ratio": None if ratio is None else round(float(ratio), 3),
                   "bbox": c["bbox"], "own": c["label"] == own_label}
            if c["label"] == own_label:
                own = c
            comp_list.append(rec)
        others = [r for r in comp_list if not r["own"]
                  and r["area_ratio"] is not None and r["area_ratio"] >= BOX_SIZED_FRAC]
        # depth step inside the top region (the flap gap is a thin line: removed by the opening)
        step_mm = max(STEP_MIN_MM, STEP_NOISE_K * rms)
        q85 = cen + (quad - cen) * 0.85
        m85 = np.zeros(labels.shape, np.uint8)
        cv2.fillPoly(m85, [np.rint(q85 - [roi[0], roi[1]]).astype(np.int32)], 1)
        off = (np.isfinite(grid) & (np.abs(grid) > step_mm) & (m85 > 0)).astype(np.uint8)
        off = cv2.morphologyEx(off, cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        npx, lab2, st2, _c2 = cv2.connectedComponentsWithStats(off, connectivity=8)
        pxa = _px_area_mm2(cam, n, d, cen) or 0.0
        step_blobs = []
        for lab in range(1, npx):
            a_mm2 = float(st2[lab, cv2.CC_STAT_AREA]) * pxa
            if quad_area > 0 and a_mm2 >= STEP_AREA_FRAC * quad_area:
                above = float(np.nanmedian(grid[lab2 == lab])) > 0
                step_blobs.append({"area_ratio": round(a_mm2 / quad_area, 3),
                                   "above_plane": above})
        out["multi_box"] = {
            "band_mm": round(band, 1), "quad_area_cm2": round(quad_area / 100.0, 1),
            "n_components": len(comp_list), "components": comp_list[:6],
            "other_box_sized": len(others),
            "own_area_ratio": (round(own["area_mm2"] / quad_area, 3)
                               if own is not None and quad_area > 0 else None),
            "step_mm": round(step_mm, 1), "step_blobs": step_blobs,
            "suspect": bool(others or step_blobs),
            "evidence": [*(["other_box_sized_coplanar_component"] if others else []),
                         *(["depth_step_inside_top"] if step_blobs else [])],
        }

        # ---- shadow: known-size rectangle fit on the own component ---------
        ksf = None
        if known_dims_mm is not None and own is not None:
            e1, e2 = _plane_basis(np.asarray(n))
            r = rows[labels == own["label"]]
            r = r[r >= 0]
            if r.size:
                p3 = np.asarray(xyz, dtype=np.float64)[r]
                hh = (p3 @ n + d)
                p3 = p3 - hh[:, None] * np.asarray(n)[None, :]
                pts2 = np.stack([p3 @ e1, p3 @ e2], axis=1) * 1000.0
                long_mm, short_mm = sorted((float(known_dims_mm[0]),
                                            float(known_dims_mm[1])), reverse=True)
                f = _ksf.fit(pts2, long_mm, short_mm, 4.0)
                if f is not None:
                    q3 = c3 - ((c3 @ n + d)[:, None]) * np.asarray(n)[None, :]
                    q2 = np.stack([q3 @ e1, q3 @ e2], axis=1) * 1000.0
                    ksf = {"score": round(f["score"], 3),
                           "inside_ratio": round(f["inside_ratio"], 3),
                           "coverage": round(f["coverage"], 3),
                           "outside_mean_mm": round(f["outside_mean_mm"], 2),
                           "yaw_deg": round(f["yaw_deg"], 2),
                           "known_mm": [long_mm, short_mm], "tol_mm": 4.0,
                           "corner_disagreement_mm":
                               _ksf.corner_disagreement_mm(f["corners"], q2),
                           "n_points": int(r.size)}
        out["known_size_fit"] = ksf
        out["height_band_components"] = {
            "band_mm": round(band, 1),
            "n": len(comp_list),
            "largest_area_cm2": (round(comps["components"][0]["area_mm2"] / 100.0, 1)
                                 if comps["components"] else None),
        }
        out["time_s"] = round(time.perf_counter() - t0, 3)
        return out
    except Exception as exc:  # noqa: BLE001 — diagnostics never break the pipeline
        return {"error": f"{type(exc).__name__}: {exc}",
                "time_s": round(time.perf_counter() - t0, 3)}
