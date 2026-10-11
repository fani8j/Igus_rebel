# Provenance: moved into box_seam_detection (2026-10-12) from rgbd_captures/precision_box_detector/precision_box_detector/plane_fit.py
# at commit 6df59943f23e0d551cdc448b47ffea9e95ece20a (working tree identical to HEAD, unmodified).
# Only imports/sys.path handling changed; logic unchanged.
"""Robust plane estimation for the precision box detector (Milestones 1-4).

Pipeline: deterministic seeded RANSAC hypothesis search, then Huber-weighted
IRLS refinement of a total-least-squares (SVD) plane fit.

Conventions (project-wide, see contract ``local://precision_detector_contract.md``):

- meters internally; residual statistics are reported in millimeters
  (explicit ``*_mm`` fields).
- plane equation ``normal . x + d = 0`` with a unit normal, ``d`` in meters.
- every estimate carries ``frame_id``; frames are never transformed or
  assumed here -- the caller supplies points already expressed in one frame
  and the estimate is reported in that same frame.
- normal orientation is deterministic and explicit: by default the normal is
  flipped to point from the plane toward ``orient_toward`` (default: the
  frame origin, i.e. the sensor when points are in a camera frame). Pass an
  explicit ``orient_toward`` point, or ``align_with`` a reference normal
  (e.g. the table normal for a top-face fit), to control this.
- determinism: RANSAC draws come from ``numpy.random.default_rng(seed)``;
  IRLS/SVD are deterministic. Identical inputs + seed give bit-identical
  outputs (up to BLAS threading nondeterminism in SVD).

No Open3D, no ROS, no GUI: numpy only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

try:  # Owned by the CameraFramesSync agent; duck-typed fallback keeps
    # landing order irrelevant (import stays lightweight either way).
    from .camera_model import Plane as _SiblingPlane
except Exception:  # pragma: no cover - sibling may land later
    _SiblingPlane = None  # type: ignore[assignment]

__all__ = [
    "PlaneEstimate",
    "fit_plane_ransac_irls",
    "fit_table_plane",
    "fit_top_plane",
]


# ---------------------------------------------------------------------------
# Result container


@dataclass
class PlaneEstimate:
    """Robust plane fit result (contract type).

    ``normal`` is a unit (3,) vector, ``d`` in meters with
    ``normal . x + d == 0``. Residual statistics are over inlier absolute
    residuals, in millimeters. ``inlier_mask`` indexes the finite input
    points presented to the fitter (see each function's docstring for the
    exact indexing domain). ``inlier_ratio`` is ``n_inliers / n_points`` and
    ``confidence`` in [0, 1] folds inlier support and residual spread.
    """

    frame_id: str
    normal: np.ndarray
    d: float
    inlier_mask: np.ndarray
    rms_mm: float
    median_mm: float
    p90_mm: float
    p95_mm: float
    inlier_ratio: float
    confidence: float

    @property
    def n_inliers(self) -> int:
        return int(np.count_nonzero(self.inlier_mask))

    def to_plane(self) -> Any:
        """Return the sibling ``camera_model.Plane`` for this estimate."""
        if _SiblingPlane is None:
            raise ImportError(
                "camera_model.Plane is unavailable (sibling module not yet "
                "landed); use .normal/.d/.frame_id directly."
            )
        return _SiblingPlane(
            normal=np.asarray(self.normal, dtype=np.float64).reshape(3),
            d=float(self.d),
            frame_id=str(self.frame_id),
        )


# ---------------------------------------------------------------------------
# Low-level helpers


def _require_frame(frame_id: str) -> str:
    if not isinstance(frame_id, str) or not frame_id:
        raise ValueError("frame_id must be a non-empty string")
    return frame_id


def _as_xyz(points_xyz: np.ndarray, what: str = "points_xyz") -> np.ndarray:
    pts = np.asarray(points_xyz, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"{what} must have shape (N, 3), got {pts.shape}")
    if pts.shape[0] == 0:
        raise ValueError(f"{what} is empty")
    return pts


def _fit_plane_svd(
    pts: np.ndarray, weights: Optional[np.ndarray] = None
) -> tuple[np.ndarray, float]:
    """Total-least-squares plane fit (optionally weighted)."""
    pts = np.asarray(pts, dtype=np.float64)
    if weights is None:
        centroid = pts.mean(axis=0)
        centered = pts - centroid
    else:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        w_sum = float(w.sum())
        if not np.isfinite(w_sum) or w_sum <= 0.0:
            raise ValueError("SVD refit received non-positive weights")
        centroid = (pts * w[:, None]).sum(axis=0) / w_sum
        centered = (pts - centroid) * np.sqrt(w)[:, None]
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    normal = np.asarray(vt[-1], dtype=np.float64)
    norm = float(np.linalg.norm(normal))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("SVD plane fit is degenerate")
    normal = normal / norm
    return normal, float(-(normal @ centroid))


def _orient_toward(
    normal: np.ndarray,
    d: float,
    ref_point: np.ndarray,
    centroid: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Flip (normal, d) so the normal points toward ``ref_point``.

    Deterministic: ties (exactly perpendicular, measure zero) keep the
    incoming sign.
    """
    if float(normal @ (ref_point - centroid)) < 0.0:
        return -normal, -d
    return normal, d


def _residual_stats_mm(abs_resid_m: np.ndarray) -> tuple[float, float, float, float]:
    r = np.asarray(abs_resid_m, dtype=np.float64).reshape(-1)
    if r.size == 0:
        raise ValueError("no inliers to summarize")
    rms = float(np.sqrt(np.mean(r**2)) * 1000.0)
    median = float(np.median(r) * 1000.0)
    p90 = float(np.percentile(r, 90.0) * 1000.0)
    p95 = float(np.percentile(r, 95.0) * 1000.0)
    return rms, median, p90, p95


def _huber_weights(abs_resid: np.ndarray, delta: float) -> np.ndarray:
    delta = float(delta)
    if not np.isfinite(delta) or delta <= 0.0:
        raise ValueError("huber_delta_m must be positive finite")
    w = np.ones_like(abs_resid, dtype=np.float64)
    big = abs_resid > delta
    w[big] = delta / abs_resid[big]
    return w


def _irls_refine(
    pts: np.ndarray,
    normal: np.ndarray,
    d: float,
    *,
    huber_delta_m: float,
    irls_iters: int,
) -> tuple[np.ndarray, float]:
    """Huber-IRLS polish of an SVD plane fit over ``pts`` (deterministic)."""
    normal = np.asarray(normal, dtype=np.float64).reshape(3)
    d = float(d)
    for _ in range(int(irls_iters)):
        resid = pts @ normal + d
        w = _huber_weights(np.abs(resid), huber_delta_m)
        new_n, new_d = _fit_plane_svd(pts, w)
        if float(new_n @ normal) < 0.0:  # keep orientation fixed while iterating
            new_n, new_d = -new_n, -new_d
        if float(np.linalg.norm(new_n - normal)) < 1e-9 and abs(new_d - d) < 1e-12:
            normal, d = new_n, new_d
            break
        normal, d = new_n, new_d
    return normal, d


def _finish_estimate(
    pts: np.ndarray,
    normal: np.ndarray,
    d: float,
    frame_id: str,
    thresh_m: float,
    orient_toward: np.ndarray,
) -> PlaneEstimate:
    """Orient, gate inliers at ``thresh_m``, and summarize residual stats."""
    centroid = pts.mean(axis=0)
    normal, d = _orient_toward(normal, d, orient_toward, centroid)
    resid = pts @ normal + d
    inlier_mask = np.abs(resid) < float(thresh_m)
    abs_in = np.abs(resid[inlier_mask])
    rms_mm, median_mm, p90_mm, p95_mm = _residual_stats_mm(abs_in)
    inlier_ratio = float(np.count_nonzero(inlier_mask) / pts.shape[0])
    # Confidence folds support and tightness: full marks for a fully
    # supported fit with rms << thresh, decaying exponentially past that.
    confidence = float(
        np.clip(inlier_ratio * np.exp(-(rms_mm / 1000.0) / float(thresh_m)), 0.0, 1.0)
    )
    return PlaneEstimate(
        frame_id=frame_id,
        normal=normal,
        d=float(d),
        inlier_mask=inlier_mask,
        rms_mm=rms_mm,
        median_mm=median_mm,
        p90_mm=p90_mm,
        p95_mm=p95_mm,
        inlier_ratio=inlier_ratio,
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# Core fitter


def fit_plane_ransac_irls(
    points_xyz: np.ndarray,
    frame_id: str,
    *,
    thresh_m: float = 0.005,
    iters: int = 1000,
    seed: int = 0,
    huber_delta_m: Optional[float] = None,
    irls_iters: int = 10,
    min_inliers: int = 3,
    orient_toward: Optional[np.ndarray] = None,
    align_with: Optional[np.ndarray] = None,
    max_align_angle_deg: Optional[float] = None,
) -> PlaneEstimate:
    """Fit a plane with seeded RANSAC + Huber-IRLS SVD refinement.

    - RANSAC proposes planes from random point triplets drawn with
      ``numpy.random.default_rng(seed)`` and keeps the hypothesis with the
      most inliers inside ``thresh_m`` (first-wins ties: deterministic).
    - With ``align_with`` + ``max_align_angle_deg`` set, each triplet
      hypothesis normal is gated by the hemisphere-insensitive angle
      ``arccos(abs(n . ref_unit))`` before support ranking, so a large
      off-axis face can never outvote a smaller on-axis one. The gate
      uses the raw (unsigned) hypothesis normal and is therefore
      independent of orientation conventions.
    - The winner is refit with Huber-weighted IRLS over the winning
      RANSAC inlier set only, so a second parallel surface cannot pull
      the estimate between surfaces.
    - The final inlier mask and residual statistics are then computed
      against the full finite input population via ``_finish_estimate``.
    - Non-finite input rows are ignored (the returned ``inlier_mask``
      indexes the finite-row subset, in order).
    - With ``align_with`` set (alone or with a tolerance), the final
      normal is flipped into the reference hemisphere; with neither set
      the legacy ``orient_toward`` rule (default: frame origin) applies
      unchanged.

    Raises:
        ValueError: on empty/all-non-finite input, fewer than 3 finite
            points, no plane reaching ``min_inliers`` support, a
            non-finite/zero ``align_with`` vector, a non-finite or
            out-of-range ``max_align_angle_deg`` (must satisfy
            ``0 < tol <= 90``), or a tolerance without a reference.
    """
    frame_id = _require_frame(frame_id)
    thresh_m = float(thresh_m)
    if not np.isfinite(thresh_m) or thresh_m <= 0.0:
        raise ValueError("thresh_m must be positive finite")
    iters = int(iters)
    if iters <= 0:
        raise ValueError("iters must be positive")
    min_inliers = int(min_inliers)
    if min_inliers < 3:
        raise ValueError("min_inliers must be >= 3")
    align_unit: Optional[np.ndarray] = None
    cos_tol: Optional[float] = None
    if max_align_angle_deg is not None:
        max_align_angle_deg = float(max_align_angle_deg)
        if (
            not np.isfinite(max_align_angle_deg)
            or max_align_angle_deg <= 0.0
            or max_align_angle_deg > 90.0
        ):
            raise ValueError("max_align_angle_deg must be finite with 0 < tol <= 90")
        if align_with is None:
            raise ValueError("max_align_angle_deg requires align_with")
    if align_with is not None:
        align_unit = np.asarray(align_with, dtype=np.float64).reshape(3)
        if (
            align_unit.shape != (3,)
            or not np.all(np.isfinite(align_unit))
            or float(np.linalg.norm(align_unit)) <= 0.0
        ):
            raise ValueError("align_with must be a finite nonzero vector")
        align_unit = align_unit / float(np.linalg.norm(align_unit))
        if cos_tol is None and max_align_angle_deg is not None:
            cos_tol = float(np.cos(np.radians(float(max_align_angle_deg))))

    pts_all = _as_xyz(points_xyz)
    finite = np.isfinite(pts_all).all(axis=1)
    pts = np.ascontiguousarray(pts_all[finite])
    n = pts.shape[0]
    if n < 3:
        raise ValueError(f"need >= 3 finite points, got {n}")

    if orient_toward is None:
        orient_ref = np.zeros(3, dtype=np.float64)  # sensor origin by default
    else:
        orient_ref = np.asarray(orient_toward, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(orient_ref)):
            raise ValueError("orient_toward must be finite")

    rng = np.random.default_rng(int(seed))
    best_n: Optional[np.ndarray] = None
    best_d = 0.0
    best_count = 0
    best_mask: Optional[np.ndarray] = None
    for _ in range(iters):
        i0, i1, i2 = rng.choice(n, 3, replace=False)
        p0, p1, p2 = pts[i0], pts[i1], pts[i2]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = float(np.linalg.norm(normal))
        if not np.isfinite(norm) or norm < 1e-12:
            continue  # degenerate (collinear/coincident) triplet
        normal = normal / norm
        if cos_tol is not None and align_unit is not None:
            if abs(float(normal @ align_unit)) < cos_tol:
                continue  # off-axis face: never ranked, however large
        d = float(-(normal @ p0))
        mask = np.abs(pts @ normal + d) < thresh_m
        count = int(np.count_nonzero(mask))
        if count > best_count:
            best_count = count
            best_n = normal
            best_d = d
            best_mask = mask
            if count == n:
                break  # cannot be beaten; keeps full determinism

    if best_n is None or best_mask is None or best_count < min_inliers:
        raise ValueError(
            f"RANSAC found no plane with >= {min_inliers} inliers "
            f"(best support {best_count} of {n} points at {thresh_m * 1000:.1f} mm)"
        )

    delta = float(huber_delta_m) if huber_delta_m is not None else thresh_m
    pts_in = np.ascontiguousarray(pts[best_mask])
    ref_n, ref_d = _irls_refine(
        pts_in, best_n, best_d, huber_delta_m=delta, irls_iters=irls_iters
    )
    if cos_tol is not None and align_unit is not None:
        # Unconstrained IRLS can tilt the winner outside the RANSAC gate.
        # Measure the hemisphere-insensitive angle and, on drift, keep the
        # winning constrained normal and refit only the offset robustly.
        if abs(float(ref_n @ align_unit)) < float(cos_tol):
            ref_n = np.asarray(best_n, dtype=np.float64).reshape(3)
            ref_n = ref_n / float(np.linalg.norm(ref_n))
            ref_d = float(-np.median(pts_in @ ref_n))
    if align_unit is not None:
        if float(ref_n @ align_unit) < 0.0:
            ref_n, ref_d = -ref_n, -ref_d
        orient_ref = ref_n * 1e9  # keeps _finish_estimate from flipping back
    return _finish_estimate(pts, ref_n, ref_d, frame_id, thresh_m, orient_ref)


def fit_table_plane(
    points_xyz: np.ndarray,
    frame_id: str,
    *,
    thresh_m: float = 0.005,
    iters: int = 2000,
    seed: int = 0,
    huber_delta_m: Optional[float] = None,
    irls_iters: int = 10,
    min_inliers: int = 100,
    min_inlier_ratio: float = 0.15,
    orient_toward: Optional[np.ndarray] = None,
) -> PlaneEstimate:
    """Fit the dominant table plane (domain wrapper over RANSAC+IRLS).

    Same algorithm as :func:`fit_plane_ransac_irls` with table-scale
    defaults, plus a dominance gate: the fit must explain at least
    ``min_inlier_ratio`` of the finite points (the table is the dominant
    surface by definition).

    Raises:
        ValueError: on bad input, no plane found, or a non-dominant fit
            (tunnel-vision on a small patch rather than the table).
    """
    est = fit_plane_ransac_irls(
        points_xyz,
        frame_id,
        thresh_m=thresh_m,
        iters=iters,
        seed=seed,
        huber_delta_m=huber_delta_m,
        irls_iters=irls_iters,
        min_inliers=min_inliers,
        orient_toward=orient_toward,
    )
    min_inlier_ratio = float(min_inlier_ratio)
    if not 0.0 < min_inlier_ratio <= 1.0:
        raise ValueError("min_inlier_ratio must be in (0, 1]")
    if est.inlier_ratio < min_inlier_ratio:
        raise ValueError(
            f"table fit is not dominant: inlier_ratio {est.inlier_ratio:.3f} "
            f"< {min_inlier_ratio:.3f} (n_inliers={est.n_inliers}); "
            "workspace may miss the table or the threshold may be too tight"
        )
    return est


# ---------------------------------------------------------------------------
# Top-face fit with interior/edge/discontinuity/density rejection


def _grid_neighbor_counts(pts: np.ndarray, radius_m: float) -> np.ndarray:
    """Number of neighbors within ``radius_m`` per point (pure numpy).

    Grid-hash + exact-distance check; deterministic. O(N * local density).
    """
    radius_m = float(radius_m)
    cell = np.floor(pts / radius_m).astype(np.int64)
    buckets: dict[tuple[int, int, int], list[int]] = {}
    for i, key in enumerate(map(tuple, cell.tolist())):
        buckets.setdefault(key, []).append(i)
    counts = np.zeros(pts.shape[0], dtype=np.int64)
    r2 = radius_m * radius_m
    for i in range(pts.shape[0]):
        cx, cy, cz = cell[i]
        total = 0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    members = buckets.get((cx + dx, cy + dy, cz + dz))
                    if not members:
                        continue
                    block = pts[np.asarray(members)]
                    diff = block - pts[i]
                    total += int(np.count_nonzero(np.einsum("ij,ij->i", diff, diff) <= r2))
        counts[i] = total - 1  # exclude self
    return counts


def fit_top_plane(
    object_points: np.ndarray,
    frame_id: str,
    *,
    thresh_m: float = 0.003,
    iters: int = 2000,
    seed: int = 0,
    huber_delta_m: Optional[float] = None,
    irls_iters: int = 10,
    min_inliers: int = 50,
    edge_mask: Optional[np.ndarray] = None,
    discontinuity_m: Optional[float] = None,
    density_radius_m: Optional[float] = None,
    density_min_neighbors: int = 5,
    interior_margin_m: float = 0.015,
    min_interior_fraction: float = 0.3,
    orient_toward: Optional[np.ndarray] = None,
    align_with: Optional[np.ndarray] = None,
    max_align_angle_deg: Optional[float] = None,
    ) -> PlaneEstimate:
    """Fit the box top face from segmented object points.

    Robustness layers (all deterministic):

    1. **Edge mask**: entries with ``edge_mask == True`` (depth edges,
       known occlusion rims, image-boundary pixels mapped back to the cloud
       where organization exists) are excluded from every fit, but still
       scored in the returned ``inlier_mask``.
    2. **Local-density mask**: with ``density_radius_m`` set, points with
       fewer than ``density_min_neighbors`` neighbors inside the radius are
       treated as sparse outliers (flap curl, mixed pixels) and excluded
       from fitting.
    3. **Discontinuity rejection**: after a pilot fit, points farther than
       ``discontinuity_m`` (default ``3 * thresh_m``) off the pilot plane
       (side faces, tape ridges) are excluded from refinement.
    4. **Interior refit**: the final IRLS fit uses only points at least
       ``interior_margin_m`` inside the pilot inlier bounding box measured
       in plane coordinates, so rounded edges / chamfers never tilt the
       plane. If erosion leaves too few points, a percentile fallback
       (5-95% per plane axis) is used; if that is still too small, the full
       pilot core is used.

    Boundary depth is never used for sizing: this function estimates plane
    geometry only and returns no extent/size quantities.

    ``align_with`` (e.g. the table normal) pins the output hemisphere
    after the ``orient_toward`` rule is applied. With
    ``max_align_angle_deg`` also set, the pilot RANSAC additionally gates
    each triplet hypothesis by the hemisphere-insensitive angle
    ``arccos(abs(n . ref_unit))`` before support ranking, so a larger
    off-axis (e.g. vertical side) face can never outvote the top face;
    only the winning on-axis inlier set is IRLS-refined. ``None``
    (default) preserves the legacy unconstrained search.

    Raises:
        ValueError: on empty input, an ``edge_mask`` of the wrong shape,
            no usable points after masking, no plane reaching
            ``min_inliers``, a non-finite/zero ``align_with`` vector, a
            non-finite or out-of-range ``max_align_angle_deg`` (must
            satisfy ``0 < tol <= 90``), or a tolerance without a
            reference.
    """
    frame_id = _require_frame(frame_id)
    pts_all = _as_xyz(object_points, "object_points")
    finite = np.isfinite(pts_all).all(axis=1)
    if int(np.count_nonzero(finite)) < 3:
        raise ValueError("object_points has fewer than 3 finite points")

    if edge_mask is not None:
        edge = np.asarray(edge_mask, dtype=bool).reshape(-1)
        if edge.shape[0] != pts_all.shape[0]:
            raise ValueError(
                f"edge_mask length {edge.shape[0]} != {pts_all.shape[0]} points"
            )
    else:
        edge = np.zeros(pts_all.shape[0], dtype=bool)

    usable = finite & ~edge
    if int(np.count_nonzero(usable)) < 3:
        raise ValueError("no usable points: everything is non-finite or edge-masked")

    if density_radius_m is not None:
        density_radius_m = float(density_radius_m)
        if not np.isfinite(density_radius_m) or density_radius_m <= 0.0:
            raise ValueError("density_radius_m must be positive finite")
        cand = pts_all[usable]
        counts = _grid_neighbor_counts(cand, density_radius_m)
        keep = counts >= int(density_min_neighbors)
        tmp = np.zeros(pts_all.shape[0], dtype=bool)
        tmp[np.flatnonzero(usable)[keep]] = True
        usable = tmp
        if int(np.count_nonzero(usable)) < 3:
            raise ValueError("local-density mask removed all usable points")

    # Pilot fit on the masked core (orientation pinned toward the sensor so
    # the interior erosion below is deterministic).
    pilot = fit_plane_ransac_irls(
        pts_all[usable],
        frame_id,
        thresh_m=thresh_m,
        iters=iters,
        seed=seed,
        huber_delta_m=huber_delta_m,
        irls_iters=max(1, int(irls_iters) // 2),
        min_inliers=min(3, int(np.count_nonzero(usable))),
        orient_toward=orient_toward,
        align_with=align_with,
        max_align_angle_deg=max_align_angle_deg,
    )

    disc = float(discontinuity_m) if discontinuity_m is not None else 3.0 * float(thresh_m)
    resid_core = pts_all[usable] @ pilot.normal + pilot.d
    core_idx = np.flatnonzero(usable)[np.abs(resid_core) <= disc]
    if core_idx.size < 3:
        raise ValueError("discontinuity rejection removed all usable points")

    # Interior erosion in pilot-plane coordinates.
    core = pts_all[core_idx]
    centroid = core.mean(axis=0)
    # Orthonormal in-plane basis (deterministic: SVD Vt order is fixed).
    _, _, vt = np.linalg.svd(core - centroid, full_matrices=False)
    axis_u, axis_v = vt[0], vt[1]
    uv = (core - centroid) @ np.stack([axis_u, axis_v], axis=1)
    margin = float(interior_margin_m)
    interior = None
    if margin > 0.0:
        lo, hi = uv.min(axis=0), uv.max(axis=0)
        interior = (
            (uv[:, 0] >= lo[0] + margin)
            & (uv[:, 0] <= hi[0] - margin)
            & (uv[:, 1] >= lo[1] + margin)
            & (uv[:, 1] <= hi[1] - margin)
        )
        if int(np.count_nonzero(interior)) < max(
            3, int(float(min_interior_fraction) * core.shape[0])
        ):
            interior = None  # fall through to percentile fallback
    if interior is None and margin > 0.0:
        q_lo, q_hi = np.percentile(uv, [5.0, 95.0], axis=0)
        interior = (
            (uv[:, 0] >= q_lo[0])
            & (uv[:, 0] <= q_hi[0])
            & (uv[:, 1] >= q_lo[1])
            & (uv[:, 1] <= q_hi[1])
        )
        if int(np.count_nonzero(interior)) < 3:
            interior = np.ones(core.shape[0], dtype=bool)
    elif interior is None:
        interior = np.ones(core.shape[0], dtype=bool)

    refine_pts = core[interior]
    delta = float(huber_delta_m) if huber_delta_m is not None else float(thresh_m)
    ref_n, ref_d = _irls_refine(
        refine_pts, pilot.normal, pilot.d, huber_delta_m=delta, irls_iters=irls_iters
    )
    if max_align_angle_deg is not None and align_with is not None:
        # Same drift guard as the RANSAC path: interior IRLS is
        # unconstrained and must not tilt the pilot outside its gate.
        ref_vec_n = np.asarray(align_with, dtype=np.float64).reshape(3)
        ref_vec_n = ref_vec_n / float(np.linalg.norm(ref_vec_n))
        cos_tol_top = float(np.cos(np.radians(float(max_align_angle_deg))))
        if abs(float(ref_n @ ref_vec_n)) < cos_tol_top:
            ref_n = np.asarray(pilot.normal, dtype=np.float64).reshape(3)
            ref_n = ref_n / float(np.linalg.norm(ref_n))
            ref_d = float(-np.median(refine_pts @ ref_n))

    # Final scoring is over ALL finite points (edges included): the caller
    # sees which masked points still agree with the interior fit.
    # Orientation: align_with (when given) wins explicitly; otherwise the
    # orient_toward rule (default: frame origin) applies deterministically.
    if align_with is not None:
        ref_vec = np.asarray(align_with, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(ref_vec)) or float(np.linalg.norm(ref_vec)) <= 0.0:
            raise ValueError("align_with must be a finite nonzero vector")
        if float(ref_n @ ref_vec) < 0.0:
            ref_n, ref_d = -ref_n, -ref_d
        orient_ref = ref_n * 1e9  # keeps _finish_estimate from flipping back
    elif orient_toward is None:
        orient_ref = np.zeros(3, dtype=np.float64)
    else:
        orient_ref = np.asarray(orient_toward, dtype=np.float64).reshape(3)
    finite_pts = np.ascontiguousarray(pts_all[finite])
    est_finite = _finish_estimate(
        finite_pts, ref_n, ref_d, frame_id, float(thresh_m), orient_ref
    )
    # Expand the mask to the full input domain (non-finite rows → False).
    full_mask = np.zeros(pts_all.shape[0], dtype=bool)
    full_mask[np.flatnonzero(finite)[est_finite.inlier_mask]] = True
    est_finite.inlier_mask = full_mask

    if est_finite.n_inliers < int(min_inliers):
        raise ValueError(
            f"top plane has only {est_finite.n_inliers} inliers "
            f"(< {int(min_inliers)}); object may be missing or mis-segmented"
        )
    return est_finite
