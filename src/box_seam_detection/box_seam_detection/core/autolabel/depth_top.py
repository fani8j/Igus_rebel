# Provenance: moved into box_seam_detection (2026-10-12) from rgbd_captures/scripts/autolabel/depth_top.py
# at commit 6df59943f23e0d551cdc448b47ffea9e95ece20a (working tree identical to HEAD, unmodified).
# Only imports/sys.path handling changed; logic unchanged.
"""Depth-first carton top-face geometry from a registered (unorganized) cloud.

The capture PCDs are treated as UNORGANIZED: point order is never assumed to
map to image pixels. Each point is pushed through an explicit pinhole model
built from the capture manifest's raw ``camera_info`` (``P`` preferred, ``K``
fallback; distortion ``D`` is NEVER applied — the registered cloud is already
expressed in the rectified camera frame).

Pipeline (numpy/OpenCV only, no Open3D):

1. workspace box crop (camera frame, config ``depth_top.ws_*``),
2. sequential SEEDED RANSAC plane fits at 5mm (``ransac_thresh``) — each fit
   strips its inliers and the next fit uses ``ransac_seed + plane_index``,
   so results are exactly reproducible,
3. table = the dominant plane; carton top = the single plane parallel to
   the table (``parallel_tol_deg``) with an offset in
   ``[offset_min, offset_max]`` and spatial support inside the table
   footprint; front/side faces = planes whose normal is near-perpendicular
   to the top (``front_min/max_angle_deg``) — top versus front/side is
   decided ONLY by normal angle, never by RGB,
4. top support = projected occupancy of the top inliers through the
   existing :mod:`autolabel.pcd` API (``project_points`` + ``z_buffer``);
   ONLY occupancy is consumed (never depth values), so projection rounding
   can shift the hull by at most +-1px and can never change plane geometry.
   The compact support grid additionally vetoes seams running off the
   measured top face (the convex hull alone can bridge occlusions),
5. largest occupancy component -> convex hull -> approxPolyDP quad
   (best IoU over ``quad_approx_eps``), ordered with
   :func:`autolabel.quad.order_corners` (no duplicate geometry here).

Depth NEVER detects the flap seam itself: the caller rectifies ONLY the
depth quad and reuses :func:`autolabel.seam.find_seam`, then gates the seam
with :func:`seam_inside_hull` plus :func:`seam_support_check`. Ambiguity
(two parallel top candidates) and off-plane evidence (seam outside the
hull or off the measured support) are explicit rejections — points are
never fabricated.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .pcd import project_points, z_buffer
from .quad import order_corners

__all__ = [
    "camera_from_capture_info",
    "load_capture_camera",
    "workspace_mask",
    "ransac_plane",
    "sequential_planes",
    "plane_angle_offset",
    "identify_table_top",
    "estimate_top",
    "seam_inside_hull",
    "seam_support_check",
]


# ---------------------------------------------------------------- calibration


def _req_float(value, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"capture camera_info: {what} must be a number, got {value!r}")
    if not math.isfinite(float(value)):
        raise ValueError(f"capture camera_info: {what} must be finite, got {value!r}")
    return float(value)


def camera_from_capture_info(info: dict) -> tuple[dict, str]:
    """Build a pinhole dict from a capture manifest's ``camera_info``.

    Uses the raw rectified projection matrix ``P`` (``fx=P[0], fy=P[5],
    cx=P[2], cy=P[6]``), falling back to ``K`` when ``P`` is absent.
    Distortion ``D`` is deliberately ignored. Returns ``(cam, source)``
    where ``cam`` has ``fx, fy, cx, cy, width, height`` and NO distortion
    keys, so :func:`autolabel.pcd.project_points` applies no distortion.
    """
    if not isinstance(info, dict):
        raise ValueError(
            "capture manifest has no camera_info mapping "
            f"(got {type(info).__name__}); depth-first needs capture calibration"
        )
    for key in ("width", "height"):
        val = info.get(key)
        if isinstance(val, bool) or not isinstance(val, (int, float)) \
                or not float(val).is_integer() or int(val) <= 0:
            raise ValueError(
                f"capture camera_info: {key} must be a positive integer, got {val!r}"
            )
    width, height = int(info["width"]), int(info["height"])
    matrix, source = None, ""
    raw_p = info.get("p")
    if raw_p is not None:
        try:
            flat = [float(v) for v in raw_p]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"capture camera_info.p must be 12 numbers: {exc}") from exc
        if len(flat) != 12:
            raise ValueError(
                f"capture camera_info.p must hold 12 numbers (3x4), got {len(flat)}"
            )
        matrix, source = (flat[0], flat[5], flat[2], flat[6]), "P"
    elif info.get("k") is not None:
        try:
            flat = [float(v) for v in info["k"]]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"capture camera_info.k must be 9 numbers: {exc}") from exc
        if len(flat) != 9:
            raise ValueError(
                f"capture camera_info.k must hold 9 numbers (3x3), got {len(flat)}"
            )
        matrix, source = (flat[0], flat[4], flat[2], flat[5]), "K"
    if matrix is None:
        raise ValueError(
            "capture camera_info has neither 'p' nor 'k'; "
            "depth-first needs raw CameraInfo P/K (no distortion is applied)"
        )
    fx, fy, cx, cy = (_req_float(v, k) for v, k in zip(matrix, ("fx", "fy", "cx", "cy")))
    if not (fx > 0 and fy > 0):
        raise ValueError(
            f"capture camera_info: fx/fy must be positive (got {fx!r}, {fy!r})"
        )
    return {"fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "width": width, "height": height}, source


def load_capture_camera(path: str) -> tuple[dict, str]:
    """Read a capture manifest JSON and return ``(cam, source)``."""
    import json

    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict) or "camera_info" not in data:
        raise ValueError(f"{path}: expected a capture manifest with 'camera_info'")
    return camera_from_capture_info(data["camera_info"])


# ---------------------------------------------------------------- RANSAC


_WS_CHUNK = 1 << 16


def workspace_mask(xyz: np.ndarray, dcfg: dict) -> np.ndarray:
    """Camera-frame box crop; finite points only.

    Evaluated in row chunks (pure per-row comparisons, so the mask is
    identical to a single full-array pass, with cache-resident temporaries).
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    bounds = (float(dcfg["ws_x_min"]), float(dcfg["ws_x_max"]),
              float(dcfg["ws_y_min"]), float(dcfg["ws_y_max"]),
              float(dcfg["ws_z_min"]), float(dcfg["ws_z_max"]))
    if xyz.ndim != 2 or xyz.shape[0] <= _WS_CHUNK:
        return _workspace_mask_block(xyz, bounds)
    keep = np.empty(xyz.shape[0], dtype=bool)
    for s in range(0, xyz.shape[0], _WS_CHUNK):
        keep[s:s + _WS_CHUNK] = _workspace_mask_block(xyz[s:s + _WS_CHUNK], bounds)
    return keep


def _workspace_mask_block(xyz: np.ndarray, bounds: tuple) -> np.ndarray:
    x0, x1, y0, y1, z0, z1 = bounds
    finite = np.isfinite(xyz).all(axis=1)
    keep = (finite
            & (xyz[:, 0] >= x0) & (xyz[:, 0] <= x1)
            & (xyz[:, 1] >= y0) & (xyz[:, 1] <= y1)
            & (xyz[:, 2] >= z0) & (xyz[:, 2] <= z1))
    return keep


_REFIT_MAX = 20000


def _fit_plane_svd(pts: np.ndarray) -> tuple[np.ndarray, float]:
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape[0] > _REFIT_MAX:
        # Deterministic stride subsample: SVD cost stays bounded while the
        # final inliers/RMS are still evaluated at full resolution.
        step = int(-(-pts.shape[0] // _REFIT_MAX))
        pts = pts[::step]
    centroid = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - centroid, full_matrices=False)
    normal = vt[-1] / np.linalg.norm(vt[-1])
    return normal, float(-(normal @ centroid))

# Vectorized RANSAC: all hypotheses are drawn in one seeded draw and their
# support counted _COUNT_BLOCK columns at a time, so one
# (downsample x _COUNT_BLOCK) float32 scratch matrix bounds memory
# (~51MB at defaults: 50000 x 256 float32) regardless of the iteration budget.
_COUNT_BLOCK = 256


# Voxel edge for guided triplet sampling: points sharing a small cell are
# usually coplanar, so triplets drawn inside one cell hit pure faces far
# more often than uniform triplets (PROSAC-style spatial coherence).
# Support is still measured fairly over the whole work set afterwards.
_GUIDE_VOXEL_M = 0.05


def _unique_rows_inverse_counts(cell: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``np.unique(cell, axis=0, return_inverse=True, return_counts=True)[1:]``.

    Fast path for non-negative int64 (N,3) cells: rows are packed into one
    int64 key whose integer order equals the lexicographic row order that
    ``np.unique(axis=0)`` sorts by, so inverse/counts are identical.
    Falls back to ``np.unique(axis=0)`` whenever packing is not provably safe.
    """
    c = np.asarray(cell)
    if (_NUMPY_LT_2 and c.ndim == 2 and c.shape[1] == 3 and c.shape[0] > 0
            and c.dtype == np.int64 and int(c.min()) >= 0):
        m1 = int(c[:, 1].max()) + 1
        m2 = int(c[:, 2].max()) + 1
        m0 = int(c[:, 0].max()) + 1
        if m0 * m1 * m2 < (1 << 62):
            key = (c[:, 0] * m1 + c[:, 1]) * m2 + c[:, 2]
            _, inv, counts = np.unique(key, return_inverse=True,
                                       return_counts=True)
            return inv, counts
    _, inv, counts = np.unique(c, axis=0, return_inverse=True,
                               return_counts=True)
    return inv, counts


# np.unique(axis=0) returns a 1-D inverse on numpy 1.x (the fast path's
# shape); numpy 2 changed that shape, so the fast path is 1.x only.
_NUMPY_LT_2 = np.lib.NumpyVersion(np.__version__) < "2.0.0"


def _guided_triplets(work: np.ndarray, rng, n_trials: int) -> np.ndarray:
    """Seeded (n_trials, 3) index array; triplets share a 5cm voxel cell.

    Cells are drawn proportionally to their point counts (dense faces get
    more hypotheses); cells with < 3 points and degenerate draws fall back
    to uniform triplets. Fully deterministic for a fixed (work, seed).
    """
    n = work.shape[0]
    idx = rng.integers(0, n, size=(n_trials, 3))  # uniform fallback base
    if n < 4:
        return idx
    lo = work.min(axis=0)
    cell = np.floor((work - lo) / _GUIDE_VOXEL_M).astype(np.int64)
    inv, counts = _unique_rows_inverse_counts(cell)
    order = np.argsort(inv, kind="stable")
    starts = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(counts, out=starts[1:])
    big = np.flatnonzero(counts >= 3)
    if big.size == 0:
        return idx
    probs = counts[big].astype(np.float64)
    probs /= probs.sum()
    draws = rng.choice(big, size=n_trials, p=probs)
    for t in range(n_trials):
        s = starts[draws[t]]
        members = order[s:s + counts[draws[t]]]
        if members.size >= 3:
            idx[t] = members[rng.choice(members.size, 3, replace=False)]
    return idx


def ransac_plane(points: np.ndarray, *, thresh: float, iters: int,
                 seed: int, downsample: int) -> tuple[np.ndarray, float, np.ndarray, float]:
    """Fit one plane with seeded RANSAC. Deterministic for fixed inputs.
    Triplets are drawn voxel-guided (same 5cm cell, dense cells preferred)
    so pure faces are hypothesized within a few hundred trials; support is
    then counted vectorized over the ``downsample`` subset in column blocks
    and the winner refit (SVD) on FULL-resolution inliers. Returns
    ``(normal, d, inliers, rms)`` with ``n.p + d = 0`` residuals and
    ``inliers`` masking ``points``.
    """
    pts = np.asarray(points, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    n_pts = pts.shape[0]
    if n_pts > downsample:
        work = pts[rng.choice(n_pts, int(downsample), replace=False)]
    else:
        work = pts
    best_n, best_d, best_count = None, 0.0, 0
    trials = int(iters)
    # One seeded draw for all hypotheses; support counted in column blocks
    # so a single (downsample x _COUNT_BLOCK) float32 scratch matrix bounds
    # memory (~20MB at defaults) regardless of the iteration budget.
    # float32 is ample for 5mm inlier tests; the final refit stays float64.
    idx = _guided_triplets(work, rng, trials)
    W32 = np.ascontiguousarray(work, dtype=np.float32)
    t0, t1, t2 = W32[idx[:, 0]], W32[idx[:, 1]], W32[idx[:, 2]]
    normals = np.cross(t1 - t0, t2 - t0)
    norms = np.linalg.norm(normals, axis=1)
    valid = (norms > 1e-6) & np.isfinite(norms)
    normals[valid] /= norms[valid, None]
    offs = -np.einsum("ij,ij->i", t0, normals).astype(np.float32)
    for s in range(0, trials, _COUNT_BLOCK):
        e = min(s + _COUNT_BLOCK, trials)
        block_ok = valid[s:e]
        if not block_ok.any():
            continue
        dist = np.abs(W32 @ normals[s:e].T + offs[s:e])
        counts = (dist < thresh).sum(axis=0)
        counts[~block_ok] = 0
        j = int(np.argmax(counts))
        if int(counts[j]) > best_count:
            best_count = int(counts[j])
            best_n = normals[s + j].astype(np.float64)
            best_d = float(offs[s + j])
    if best_n is None or best_count < 3:
        raise ValueError(
            f"RANSAC found no plane (best support {best_count}); "
            "workspace may be empty or downsample too small"
        )
    normal, dist = best_n, best_d
    inliers = np.abs(pts @ normal + dist) < thresh
    # Least-squares refit on full-resolution inliers, orientation pinned.
    for _ in range(2):
        if int(inliers.sum()) < 3:
            break
        refit_n, refit_d = _fit_plane_svd(pts[inliers])
        if float(refit_n @ normal) < 0:
            refit_n, refit_d = -refit_n, -refit_d
        normal, dist = refit_n, refit_d
        inliers = np.abs(pts @ normal + dist) < thresh
    rms = float(np.sqrt(np.mean((pts[inliers] @ normal + dist) ** 2)))
    return normal, dist, inliers, rms


def sequential_planes(points: np.ndarray, dcfg: dict) -> list[dict]:
    """Strip planes one at a time (seed + plane index). Bounded runtime.

    Each entry: ``n, d, n_inliers, rms, centroid`` plus ``mask`` over the
    input (disjoint across planes). Stops at ``max_planes`` or when the
    remainder cannot qualify as a top face.
    """
    pts = np.asarray(points, dtype=np.float64)
    thresh = float(dcfg["ransac_thresh"])
    planes: list[dict] = []
    remaining = np.ones(pts.shape[0], dtype=bool)
    for k in range(int(dcfg["max_planes"])):
        pool = pts[remaining]
        if pool.shape[0] < int(dcfg["top_min_inliers"]):
            break
        try:
            normal, dist, inl, rms = ransac_plane(
                pool, thresh=thresh, iters=int(dcfg["ransac_iters"]),
                seed=int(dcfg["ransac_seed"]) + k,
                downsample=int(dcfg["ransac_downsample"]))
        except ValueError:
            break
        if int(inl.sum()) < int(dcfg["top_min_inliers"]):
            break
        idx = np.flatnonzero(remaining)[inl]
        mask = np.zeros(pts.shape[0], dtype=bool)
        mask[idx] = True
        planes.append({
            "n": normal, "d": float(dist), "n_inliers": int(inl.sum()),
            "rms": float(rms), "centroid": pts[mask].mean(axis=0), "mask": mask,
        })
        remaining[idx] = False
    return planes


def plane_angle_offset(n1: np.ndarray, d1: float,
                       n2: np.ndarray, d2: float) -> tuple[float, float]:
    """Unsigned normal angle (deg, 0..90) and sign-aligned plane offset (m)."""
    cos_a = float(np.clip(abs(float(n1 @ n2)), 0.0, 1.0))
    angle = float(np.degrees(np.arccos(cos_a)))
    sign = 1.0 if float(n1 @ n2) >= 0 else -1.0
    return angle, float(abs(d1 - sign * d2))


def _signed_height_above(table: dict, point) -> float:
    """Height (m) of ``point`` above the ``table`` plane, + = camera side.

    The camera is at the origin; the plane normal is oriented so that the
    origin has positive signed distance."""
    n = np.asarray(table["n"], dtype=np.float64)
    d = float(table["d"])
    if d < 0:
        n, d = -n, -d
    return float(n @ np.asarray(point, dtype=np.float64) + d)


# ---------------------------------------------------------------- identify


def _plane_diag(p: dict) -> dict:
    return {
        "n_inliers": p["n_inliers"],
        "rms_mm": round(p["rms"] * 1000.0, 2),
        "normal": [round(float(v), 4) for v in p["n"]],
        "centroid": [round(float(v), 4) for v in p["centroid"]],
    }



def _polish_plane_top(cropped: np.ndarray, top: dict, thresh: float) -> dict:
    """LO-RANSAC-style core polish for the top candidate (deterministic).

    Sequential RANSAC can lock onto a crease-bridge plane (tilted through
    the top into a neighboring face): high support but higher RMS. Refitting
    on the tight core (60% band) drops the wedges and converges toward the
    dominant face; the mask/RMS/count are then re-expanded at full thresh.
    A pure plane is unaffected (sub-mm motion). Falls back to the input
    when the core is too small; records both RMS values.
    """
    core_band = 0.6 * float(thresh)
    pts = cropped[top["mask"]]
    core = pts[np.abs(pts @ top["n"] + top["d"]) < core_band]
    polished = dict(top)
    polished["polish_rms_mm"] = round(top["rms"] * 1000.0, 2)
    polished["polished"] = False
    if core.shape[0] < 100:
        return polished
    for _ in range(3):
        n_new, d_new = _fit_plane_svd(core)
        if float(n_new @ top["n"]) < 0:
            n_new, d_new = -n_new, -d_new
        moved = float(abs((core @ n_new + d_new).mean() - (core @ top["n"] + top["d"]).mean()))
        top = {"n": n_new, "d": float(d_new), **{k: v for k, v in top.items()
                                                if k not in ("n", "d")}}
        core = pts[np.abs(pts @ n_new + d_new) < core_band]
        if core.shape[0] < 100 or moved < 1e-6:
            break
    mask = np.abs(cropped @ top["n"] + top["d"]) < thresh
    if int(mask.sum()) < 3:
        return polished
    rms = float(np.sqrt(np.mean((cropped[mask] @ top["n"] + top["d"]) ** 2)))
    polished.update({"n": top["n"], "d": float(top["d"]), "mask": mask,
                     "n_inliers": int(mask.sum()), "rms": rms,
                     "centroid": cropped[mask].mean(axis=0),
                     "polish_rms_mm": round(rms * 1000.0, 2), "polished": True})
    return polished


def _merge_top_candidates(cropped: np.ndarray, cands: list,
                          dcfg: dict) -> tuple[list, dict | None]:
    """Merge same-height adjacent top patches (two flaps, one face).

    Candidates that are pairwise coplanar (offset <= top_merge_max_offset)
    with touching/overlapping 3D footprints (bbox gap <= top_merge_max_gap)
    merge into one SVD-refit top. Distinct boxes (height step or gap)
    stay separate. Returns (groups, merged): ``groups`` lists rank lists;
    ``merged`` is the refit plane dict when one group covers every
    candidate, else None. Deterministic.
    """
    items = [(int(i), p) for (i, p, _, _) in cands]
    bboxes: dict[int, tuple] = {}
    for i, p in items:
        pts = cropped[p["mask"]]
        if pts.shape[0] == 0:
            return [[i for (i, _) in items]], None
        bboxes[i] = (pts.min(axis=0), pts.max(axis=0))
    parent = {i: i for (i, _) in items}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def gap(ba, bb) -> float:
        sep = 0.0
        for ax in range(3):
            sep = max(sep, float(ba[0][ax] - bb[1][ax]),
                      float(bb[0][ax] - ba[1][ax]))
        return max(0.0, sep)

    by_rank = {i: p for (i, p) in items}
    for a in range(len(items)):
        for b in range(a + 1, len(items)):
            ia, ib = items[a][0], items[b][0]
            _, off = plane_angle_offset(by_rank[ia]["n"], by_rank[ia]["d"],
                                        by_rank[ib]["n"], by_rank[ib]["d"])
            if off > float(dcfg["top_merge_max_offset"]):
                continue
            if gap(bboxes[ia], bboxes[ib]) <= float(dcfg["top_merge_max_gap"]):
                parent[find(ia)] = find(ib)
    groups: dict[int, list] = {}
    for i, _ in items:
        groups.setdefault(find(i), []).append(i)
    grouped = [sorted(g) for g in groups.values()]
    if len(grouped) != 1:
        return grouped, None
    members = [by_rank[i] for i in grouped[0]]
    union = np.zeros(cropped.shape[0], dtype=bool)
    for i in grouped[0]:
        union |= by_rank[i]["mask"]
    pts = cropped[union]
    seed_plane = max(members, key=lambda p: p["n_inliers"])
    n_new, d_new = _fit_plane_svd(pts)
    if float(n_new @ seed_plane["n"]) < 0:
        n_new, d_new = -n_new, -d_new
    thresh = float(dcfg["ransac_thresh"])
    mask = np.abs(cropped @ n_new + d_new) < thresh
    rms = float(np.sqrt(np.mean((cropped[mask] @ n_new + d_new) ** 2)))
    return grouped, {"n": n_new, "d": float(d_new), "mask": mask,
                     "n_inliers": int(mask.sum()), "rms": rms,
                     "centroid": cropped[mask].mean(axis=0),
                     "merged_from": sorted(grouped[0])}


def identify_table_top(planes: list[dict], cropped: np.ndarray,
                       dcfg: dict) -> tuple[dict | None, dict | None, dict, dict, str | None]:
    """Pick table + carton top from sequential planes.

    ``cropped`` is the workspace array the plane ``mask``s index (masks are
    re-derived from the SELECTED table/top, never assumed from rank, so the
    table support is invariant to discovery order). Returns ``(table, top,
    front, diagnostics, reason)``; ``reason`` is None on success. ``front``
    is diagnostics-only (never gating).
    """
    cropped = np.asarray(cropped, dtype=np.float64)
    diag: dict = {"planes": [_plane_diag(p) for p in planes]}
    front: dict = {"status": "depth_no_front", "detail": "no_planes",
                   "inliers": None, "rms_mm": None, "angle_deg": None}
    if not planes:
        return None, None, front, diag, "depth_no_table"
    order = sorted(range(len(planes)), key=lambda i: -planes[i]["n_inliers"])
    table = planes[order[0]]
    if table["n_inliers"] < int(dcfg["table_min_inliers"]):
        diag["table_reject"] = "below_table_min_inliers"
        return None, None, front, diag, "depth_no_table"
    diag["table_rank"] = int(order[0])
    table_xyz = cropped[table["mask"]]
    tmin = table_xyz.min(axis=0) if table_xyz.shape[0] else np.zeros(3)
    tmax = table_xyz.max(axis=0) if table_xyz.shape[0] else np.zeros(3)
    expand = float(dcfg["support_expand_m"])
    cands, parallel_notes = [], []
    for i in order[1:]:
        p = planes[i]
        angle, offset = plane_angle_offset(table["n"], table["d"], p["n"], p["d"])
        note: dict = {"rank": int(i), "n_inliers": p["n_inliers"],
                      "rms_mm": round(p["rms"] * 1000.0, 2),
                      "angle_deg": round(angle, 2), "offset_mm": round(offset * 1000.0, 1)}
        if angle > float(dcfg["parallel_tol_deg"]):
            note["excluded"] = "not_parallel"
            parallel_notes.append(note)
            continue
        if not (float(dcfg["offset_min"]) <= offset <= float(dcfg["offset_max"])):
            note["excluded"] = "offset_out_of_range"
            parallel_notes.append(note)
            continue
        if offset > float(dcfg.get("offset_above_check_from", 0.30)):
            # Beyond the legacy 0.30 m limit (tall boxes): the plane must lie
            # on the CAMERA side of the table, else it is a floor / the real
            # table below a mis-picked "table" (the box top itself).
            if _signed_height_above(table, p["centroid"]) <= 0.0:
                note["excluded"] = "not_above_table"
                parallel_notes.append(note)
                continue
        if p["n_inliers"] < int(dcfg["top_min_inliers"]):
            note["excluded"] = "too_few_inliers"
            parallel_notes.append(note)
            continue
        c = p["centroid"]
        if not (tmin[0] - expand <= c[0] <= tmax[0] + expand
                and tmin[1] - expand <= c[1] <= tmax[1] + expand):
            note["excluded"] = "no_spatial_support"
            parallel_notes.append(note)
            continue
        cands.append((i, p, angle, offset))
        parallel_notes.append(note)
    diag["top_candidates"] = parallel_notes
    if not cands:
        return table, None, front, diag, "depth_no_top"
    if len(cands) > 1:
        # Same-height adjacent patches (two flaps, one face) merge into a
        # single refit top; distinct boxes stay ambiguous. Never fabricated:
        # the merged plane is refit from measured points.
        groups, merged = _merge_top_candidates(cropped, cands, dcfg)
        diag["top_merge_groups"] = [[int(r) for r in g] for g in groups]
        if merged is None:
            diag["ambiguity"] = sorted(
                ({"rank": int(i), "offset_mm": round(o * 1000.0, 1),
                  "n_inliers": c["n_inliers"]} for i, c, _, o in cands),
                key=lambda r: -r["n_inliers"])
            return table, None, front, diag, "depth_ambiguous_top"
        top = merged
        top_ranks = set(int(r) for r in merged["merged_from"])
        diag["top_rank"] = max(top_ranks)
        diag["top_merged_from"] = sorted(top_ranks)
    else:
        top = cands[0][1]
        top_ranks = {int(cands[0][0])}
        diag["top_rank"] = int(cands[0][0])
    # Gate 1: top plane quality (RMS) on the RAW sequential fit. A loose
    # fit is not the carton face; this runs BEFORE polish because core
    # refits can overfit a noisy slab (truncation bias).
    if top["rms"] > float(dcfg["top_rms_max"]):
        return table, None, front, diag, \
            f"depth_top_rms_high({top['rms'] * 1000.0:.2f}mm)"
    # Core polish: tight-core refit, re-expanded. Pure planes move sub-mm.
    top = _polish_plane_top(cropped, top, float(dcfg["ransac_thresh"]))
    diag["top_polished"] = bool(top.get("polished"))
    top_angle, top_offset = plane_angle_offset(table["n"], table["d"],
                                               top["n"], top["d"])
    diag["top_angle_deg"] = round(float(top_angle), 2)
    diag["top_offset_mm"] = round(float(top_offset) * 1000.0, 1)
    # Polish must not diverge: the polished fit keeps the same ceiling.
    if top["rms"] > float(dcfg["top_rms_max"]):
        return table, None, front, diag, \
            f"depth_top_rms_high({top['rms'] * 1000.0:.2f}mm)"
    # Gate 2: top share of the OBJECT faces (top + perpendicular family).
    # Only planes ADJACENT to the top count (same front_adjacency_max_m
    # condition as the front diagnostics), so a large distant wall in the
    # workspace can never dilute the ratio of a valid carton.
    perp_inliers = 0
    for i in order[1:]:
        if int(i) in top_ranks:
            continue
        p = planes[i]
        ang, _ = plane_angle_offset(top["n"], top["d"], p["n"], p["d"])
        adjacent = bool(float(np.linalg.norm(p["centroid"] - top["centroid"]))
                        <= float(dcfg["front_adjacency_max_m"]))
        if (float(dcfg["front_min_angle_deg"]) <= ang
                <= float(dcfg["front_max_angle_deg"])
                and p["n_inliers"] >= int(dcfg["front_min_inliers"])
                and adjacent):
            perp_inliers += p["n_inliers"]
    object_ratio = (float(top["n_inliers"])
                    / max(1, top["n_inliers"] + perp_inliers))
    diag["top_object_ratio"] = round(object_ratio, 3)
    diag["top_object_perp_inliers"] = int(perp_inliers)
    if object_ratio < float(dcfg["top_min_object_ratio"]):
        return table, None, front, diag, \
            f"depth_top_weak_object_support({object_ratio:.2f})"
    # Front/side faces: near-perpendicular to the TOP (diagnostics only).
    fnotes = []
    passing_ranks: list[int] = []
    for i in order[1:]:
        if int(i) in top_ranks:
            continue
        p = planes[i]
        angle, _ = plane_angle_offset(top["n"], top["d"], p["n"], p["d"])
        adjacent = bool(float(np.linalg.norm(p["centroid"] - top["centroid"]))
                        <= float(dcfg["front_adjacency_max_m"]))
        ok = (float(dcfg["front_min_angle_deg"]) <= angle <= float(dcfg["front_max_angle_deg"])
              and p["n_inliers"] >= int(dcfg["front_min_inliers"]) and adjacent)
        fnotes.append({"rank": int(i), "n_inliers": p["n_inliers"],
                       "rms_mm": round(p["rms"] * 1000.0, 2),
                       "angle_deg": round(float(angle), 2), "adjacent": adjacent,
                       "selected": False})
        if ok:
            passing_ranks.append(int(i))
    # Largest support wins; runners-up stay in diagnostics (never gating).
    if passing_ranks:
        by_support = sorted(passing_ranks,
                            key=lambda i: -planes[i]["n_inliers"])
        winner = by_support[0]
        wnote = next(n for n in fnotes if n["rank"] == winner)
        wnote["selected"] = True
        front = {"status": "ok", "detail": "",
                 "inliers": wnote["n_inliers"], "rms_mm": wnote["rms_mm"],
                 "angle_deg": wnote["angle_deg"], "rank": winner}
        if len(by_support) > 1:
            front["runner_up_inliers"] = planes[by_support[1]]["n_inliers"]
    else:
        front["detail"] = "no_perpendicular_adjacent_plane"
    diag["front_candidates"] = fnotes
    diag["front"] = {k: v for k, v in front.items() if k != "status"}
    diag["front_status"] = front["status"]
    return table, top, front, diag, None


# ---------------------------------------------------------------- support


def _quad_iou(quad: np.ndarray, hull_mask: np.ndarray) -> float:
    quad_mask = np.zeros_like(hull_mask)
    cv2.fillConvexPoly(quad_mask, np.asarray(quad, dtype=np.float64).astype(np.int32), 255)
    inter = float(((quad_mask > 0) & (hull_mask > 0)).sum())
    union = float(((quad_mask > 0) | (hull_mask > 0)).sum())
    return float(inter / union) if union > 0 else 0.0


def _support_cells(xs: np.ndarray, ys: np.ndarray, W: int, H: int,
                   cell: int) -> dict:
    """Compact top-support grid: set of occupied ``cell``-px cell keys.

    Key ``cy * stride + cx`` with ``stride = W // cell + 1``. In-memory
    only (a few thousand ints); never serialized — diagnostics carry
    counts, never the set.
    """
    cell = max(1, int(cell))
    stride = W // cell + 1
    keys = (np.asarray(ys, dtype=np.int64) // cell) * stride + \
        (np.asarray(xs, dtype=np.int64) // cell)
    uniq = np.unique(keys)
    return {"cell": cell, "stride": int(stride), "width": int(W),
            "height": int(H), "cells": frozenset(int(k) for k in uniq),
            "n_cells": int(uniq.size)}


def _cell_has_support(cells: frozenset, stride: int, cell: int,
                      x: float, y: float, radius: float) -> bool:
    """Any occupied cell with center within ``radius`` px of (x, y)."""
    r = float(radius)
    ix0 = int((x - r) // cell) - 1
    ix1 = int((x + r) // cell) + 1
    iy0 = int((y - r) // cell) - 1
    iy1 = int((y + r) // cell) + 1
    for iy in range(iy0, iy1 + 1):
        for ix in range(ix0, ix1 + 1):
            if iy * stride + ix in cells:
                cx, cy = (ix + 0.5) * cell, (iy + 0.5) * cell
                if (cx - x) ** 2 + (cy - y) ** 2 <= r * r:
                    return True
    return False


def seam_support_check(seam, support: dict | None, *, radius_px: float,
                       min_coverage: float, max_gap_frac: float,
                       endpoint_radius_px: float) -> tuple[bool, dict, str | None]:
    """Veto seams that lack nearby projected top-plane support.

    The convex hull can bridge occlusion or front-face pixels, so hull
    containment alone is not an off-plane veto. This samples the seam
    (endpoints plus regular line samples) against the compact top-support
    grid: each sample needs an occupied cell within ``radius_px``. Small
    holes (tape gaps) are tolerated via ``min_coverage``/``max_gap_frac``,
    but sustained absence means the line runs off the measured top face.
    Depth relief along the seam is never compared (no exact-relief test).
    Returns ``(ok, info, reason)``.
    """
    info: dict = {"n_samples": 0, "coverage": 0.0, "max_gap_frac": 0.0,
                  "endpoint_support": [False, False]}
    if not support or not support.get("cells"):
        return False, info, "seam_off_top_support(no_support_grid)"
    (x1, y1), (x2, y2) = ((float(seam[0][0]), float(seam[0][1])),
                          (float(seam[1][0]), float(seam[1][1])))
    length = float(np.hypot(x2 - x1, y2 - y1))
    if not np.isfinite(length) or length <= 0:
        return False, info, "seam_off_top_support(degenerate_seam)"
    cells, stride, cell = support["cells"], support["stride"], support["cell"]
    step = max(1.0, cell / 2.0)
    n = min(400, max(2, int(length / step) + 1))
    ts = np.linspace(0.0, 1.0, n)
    supported = np.array([
        _cell_has_support(cells, stride, cell,
                          x1 + t * (x2 - x1), y1 + t * (y2 - y1), radius_px)
        for t in ts], dtype=bool)
    ends = [_cell_has_support(cells, stride, cell, x1, y1, endpoint_radius_px),
            _cell_has_support(cells, stride, cell, x2, y2, endpoint_radius_px)]
    coverage = float(supported.mean())
    # Longest unsupported run as a fraction of the line.
    gap, run = 0, 0
    for flag in list(supported) + [True]:
        if not flag:
            run += 1
        else:
            gap = max(gap, run)
            run = 0
    gap_frac = float(gap / max(1, n))
    info.update({"n_samples": int(n), "coverage": round(coverage, 3),
                 "max_gap_frac": round(gap_frac, 3),
                 "endpoint_support": [bool(e) for e in ends]})
    if not all(ends):
        return False, info, \
            f"seam_off_top_support(unanchored_endpoint:{ends[0]!s},{ends[1]!s})"
    if coverage < float(min_coverage):
        return False, info, f"seam_off_top_support(coverage={coverage:.2f})"
    if gap_frac > float(max_gap_frac):
        return False, info, f"seam_off_top_support(gap={gap_frac:.2f})"
    return True, info, None


def support_hull_and_quad(top_xyz: np.ndarray, cam: dict, image_hw: tuple[int, int],
                          dcfg: dict
                          ) -> tuple[np.ndarray | None, np.ndarray | None, dict,
                                     dict | None, str | None]:
    """Project top inliers (existing pcd API, occupancy only) -> hull + quad.

    Returns ``(hull, quad, diag, support, reason)``. ``support`` is the
    compact in-memory top-support grid (``support_cell_px`` cells holding
    projected top points) for the caller's seam veto; it is NEVER part of
    the serialized ``diag`` (cell counts only) and must not be serialized.
    """
    H, W = int(image_hw[0]), int(image_hw[1])
    diag: dict = {}
    uv, depth, stats = project_points(np.asarray(top_xyz, dtype=np.float64), cam)
    diag["n_projected"] = int(stats["n_input"])
    diag["n_behind_or_too_close"] = int(stats["n_behind_or_too_close"])
    depth_map, _, n_conflicts = z_buffer(uv, depth, W, H)
    diag["z_conflicts"] = int(n_conflicts)
    occ = np.isfinite(depth_map).astype(np.uint8)
    diag["n_occupied_px"] = int(occ.sum())
    if not diag["n_occupied_px"]:
        return None, None, diag, None, "depth_small_support"
    n_labels, labels, comp_stats, _ = cv2.connectedComponentsWithStats(occ, 8)
    diag["n_components"] = int(n_labels - 1)
    if diag["n_components"] <= 0:
        return None, None, diag, None, "depth_small_support"
    areas = comp_stats[1:, cv2.CC_STAT_AREA].astype(np.int64)
    best = int(np.argmax(areas))
    largest = int(areas[best])
    diag["largest_component_px"] = largest
    diag["support_dominance"] = round(float(largest / max(1, diag["n_occupied_px"])), 3)
    if float(largest / (W * H)) < float(dcfg["min_hull_area_frac"]):
        return None, None, diag, None, "depth_small_support"
    if diag["support_dominance"] < float(dcfg["min_support_dominance"]):
        return None, None, diag, None, "depth_fragmented_support"
    ys, xs = np.nonzero(labels == (best + 1))
    if xs.size < 4:
        return None, None, diag, None, "depth_degenerate_hull"
    support = _support_cells(xs, ys, W, H, int(dcfg["support_cell_px"]))
    diag["support_cells"] = support["n_cells"]
    hull = cv2.convexHull(np.stack([xs, ys], axis=1).astype(np.int32))
    hull_area = float(cv2.contourArea(hull))
    diag["hull_area_frac"] = round(hull_area / (W * H), 4)
    if hull_area / (W * H) < float(dcfg["min_hull_area_frac"]):
        return None, None, diag, None, "depth_small_support"
    if hull.shape[0] < 3:
        return None, None, diag, None, "depth_degenerate_hull"
    hull_mask = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(hull_mask, hull, 255)
    perimeter = float(cv2.arcLength(hull, True))
    eps_list = dcfg["quad_approx_eps"]
    if isinstance(eps_list, (int, float)):
        eps_list = [eps_list]
    best_quad, best_iou = None, -1.0
    for eps in eps_list:
        approx = cv2.approxPolyDP(hull, float(eps) * perimeter, True)
        if len(approx) != 4 or not bool(cv2.isContourConvex(approx)):
            continue
        quad = approx.reshape(4, 2).astype(np.float64)
        # Near-180-degree hull vertices quantise to coincident approx
        # points (a fabricated corner); require distinct corners.
        edge_len = np.linalg.norm(quad - np.roll(quad, -1, axis=0), axis=1)
        if bool((edge_len < 2.0).any()):
            continue
        iou = _quad_iou(quad, hull_mask)
        if iou > best_iou:
            best_quad, best_iou = quad, iou
    if best_quad is None:
        diag["quad_iou"] = 0.0
        return hull, None, diag, None, "depth_quad_unsupported"
    diag["quad_iou"] = round(float(best_iou), 3)
    if best_iou < float(dcfg["min_quad_iou"]):
        return hull, None, diag, None, "depth_quad_unsupported"
    try:
        quad = order_corners(np.asarray(best_quad, dtype=np.float64).reshape(4, 2))
    except ValueError:
        # Coincident approx points after rounding (a fabricated corner):
        # explicit rejection, never duplicated geometry. Unreachable while
        # the >=2px edge filter above holds; kept so estimate_top never
        # raises on data.
        diag["quad_iou"] = 0.0
        return hull, None, diag, None, "depth_quad_unsupported"
    return hull, quad, diag, support, None


# ---------------------------------------------------------------- top-level


def estimate_top(xyz: np.ndarray, cam: dict, image_hw: tuple[int, int],
                 dcfg: dict, *, first_plane_out: dict | None = None) -> dict:
    """Full depth-first top estimate. Never raises on data; reports reasons.

    Returns ``{"ok", "reason", "quad", "hull", "depth", "support"}``;
    ``quad`` is (4,2) float64 image pixels and ``hull`` is OpenCV (M,1,2)
    int32 when ok. ``support`` is the compact in-memory top-support grid
    for the caller's seam veto (None unless the support stage ran); it is
    NEVER serialized — ``depth`` carries counts and the confidence
    breakdown only.

    ``first_plane_out`` (optional dict, filled in place; never affects the
    result): ``n``/``d``/``n_inliers``/``n_pool`` of the FIRST sequential
    plane, so a caller can reuse it as the table plane (same cropped cloud,
    same seed) instead of refitting.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    depth: dict = {"assumed_registered": True,
                   "calibration_distortion": "ignored (registered cloud)"}
    keep = workspace_mask(xyz, dcfg)
    depth["n_cloud"] = int(xyz.shape[0])
    depth["n_workspace"] = int(keep.sum())
    if depth["n_workspace"] < int(dcfg["top_min_inliers"]):
        depth["status"] = "depth_empty_workspace"
        return {"ok": False, "reason": "depth_empty_workspace",
                "quad": None, "hull": None, "depth": depth, "support": None}
    cropped = xyz[keep]
    planes = sequential_planes(cropped, dcfg)
    if first_plane_out is not None and planes:
        first_plane_out.update(
            n=np.array(planes[0]["n"], dtype=np.float64),
            d=float(planes[0]["d"]), n_inliers=int(planes[0]["n_inliers"]),
            rms=float(planes[0]["rms"]),
            n_pool=int(cropped.shape[0]))
    table, top, front, ident, reason = identify_table_top(planes, cropped, dcfg)
    depth.update(ident)
    if table is not None:
        depth["table_inliers"] = table["n_inliers"]
        depth["table_rms_mm"] = round(table["rms"] * 1000.0, 2)
    if reason is not None:
        depth["status"] = reason
        return {"ok": False, "reason": reason,
                "quad": None, "hull": None, "depth": depth, "support": None}
    assert top is not None
    depth["top_inliers"] = top["n_inliers"]
    depth["top_rms_mm"] = round(top["rms"] * 1000.0, 2)
    hull, quad, sup_diag, support, reason = support_hull_and_quad(
        cropped[top["mask"]], cam, image_hw, dcfg)
    depth.update(sup_diag)
    if reason is not None:
        depth["status"] = reason
        return {"ok": False, "reason": reason,
                "quad": None, "hull": hull, "depth": depth, "support": support}
    assert quad is not None and hull is not None
    depth["quad_area_frac"] = round(float(
        0.5 * abs(float(np.sum(quad[:, 0] * np.roll(quad[:, 1], -1)
                               - np.roll(quad[:, 0], -1) * quad[:, 1]))) / (image_hw[1] * image_hw[0])), 4)
    top_rms_max = float(dcfg["top_rms_max"])
    depth["confidence"] = {
        "plane_quality": round(max(0.0, 1.0 - top["rms"] / top_rms_max), 3),
        "top_rms_mm": round(top["rms"] * 1000.0, 2),
        "top_inliers": int(top["n_inliers"]),
        "top_object_ratio": depth.get("top_object_ratio"),
        "quad_iou": depth.get("quad_iou"),
        "support_dominance": depth.get("support_dominance"),
    }
    depth["status"] = "ok"
    return {"ok": True, "reason": None, "quad": quad, "hull": hull,
            "depth": depth, "support": support}


def seam_inside_hull(seam, hull: np.ndarray, tol_px: float) -> tuple[bool, float]:
    """Both seam endpoints (and midpoint) must lie in/on the depth hull.

    Returns ``(inside, min_signed_distance_px)``; negative distances are
    outside. ``tol_px`` absorbs residual +-1px projection quantization.
    """
    hull = np.asarray(hull).reshape(-1, 2).astype(np.float64)
    (x1, y1), (x2, y2) = ((float(seam[0][0]), float(seam[0][1])),
                          (float(seam[1][0]), float(seam[1][1])))
    pts = [(x1, y1), (x2, y2), ((x1 + x2) / 2.0, (y1 + y2) / 2.0)]
    dists = [float(cv2.pointPolygonTest(hull.astype(np.float32), p, True)) for p in pts]
    return (min(dists) >= -float(tol_px)), round(min(dists), 2)
