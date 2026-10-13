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
    """The connected coplanar region of the top plane that the quad (a seed)
    lies in: pixels within ``band`` of the plane (``band`` = max(``BAND_MIN_MM``,
    ``BAND_NOISE_K`` rms, ``band_mm``) the depth cross-check's own band), closed
    over the flap gap / dropouts (``CLOSE_MM``), 8-connected, the component under the
    seed, cleaned by an opening and holes filled.

    Returns ``(hull_mask, info)``: ``hull_mask`` = full-image uint8 convex hull of
    the cleaned region (None when there is none), ``info`` = JSON-able details
    (``reason`` when None). Never raises."""
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
            return None, info
        lab = int(np.bincount(sel).argmax())
        seed_frac = float(sel[sel == lab].size) / float(max(1, int(seed.sum())))
        info["seed_coverage"] = round(seed_frac, 3)
        if seed_frac < GROW_MIN_SEED_FRAC:
            info["reason"] = f"seed_coverage_low({seed_frac:.2f})"
            return None, info
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
            return None, info
        # the component with the most seed overlap (not simply the largest)
        ov = [int(((lab2 == k) & (seed > 0)).sum()) for k in range(1, n_cc)]
        best = 1 + int(np.argmax(ov))
        cnts, _ = cv2.findContours((lab2 == best).astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            info["reason"] = "no_contour"
            return None, info
        hull = cv2.convexHull(max(cnts, key=cv2.contourArea)) + np.array([[roi[0], roi[1]]])
        mask = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(mask, hull.reshape(-1, 2).astype(np.int32), 1)
        info["hull_area_px"] = int(mask.sum())
        return mask, info
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
