# Provenance: moved into box_seam_detection (2026-10-12) from rgbd_captures/precision_box_detector/precision_box_detector/subpixel_edge.py
# at commit 6df59943f23e0d551cdc448b47ffea9e95ece20a (working tree identical to HEAD, unmodified).
# Only imports/sys.path handling changed; logic unchanged.
"""Reference-guided subpixel edge refinement of box-top quads.

Uses an accepted hybrid/PCD detector output only as a coarse reference
region; the four physical box top edges are re-estimated from the original
RGB frame. For each coarse edge, 1-D intensity profiles are sampled along
the edge normal (vectorized bilinear interpolation of grayscale), the edge
position per profile comes from a central-difference gradient peak with
quadratic subpixel interpolation, and each edge is fit with a robust
total-least-squares line (deterministic iterative MAD rejection). Final
corners are adjacent line intersections, never independent keypoints.

Corner order throughout is [back_left, back_right, front_right, front_left]
and edge order is [back, right, front, left].
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

__all__ = [
    "EDGE_NAMES",
    "EdgeSample",
    "RefinedEdge",
    "RefinedQuad",
    "refine_quad_edges",
]

#: Edge names in RefinedQuad.edges order.
EDGE_NAMES: tuple[str, str, str, str] = ("back", "right", "front", "left")

#: (start, end) coarse-corner indices per edge, matching EDGE_NAMES order.
_EDGE_PAIRS: tuple[tuple[int, int], ...] = ((0, 1), (1, 2), (2, 3), (3, 0))

_OK = "ok"


@dataclass
class EdgeSample:
    """One normal-profile measurement along a coarse edge."""

    coarse_px: np.ndarray  # profile center on the coarse edge, (2,)
    refined_px: np.ndarray  # subpixel edge point, (2,)
    offset_px: float  # signed position of the edge along the profile normal
    gradient: float  # interpolated gradient magnitude at the peak
    valid: bool
    reason: str  # "ok" or the rejection cause


@dataclass
class RefinedEdge:
    """Robust line fit for one named box edge."""

    name: str
    line_abc: np.ndarray  # normalized (a, b, c), a*x+b*y+c=0, hypot(a,b)==1
    samples: list[EdgeSample]
    residual_rms_px: float  # RMS perpendicular residual of inliers (NaN if failed)
    support_score: float  # inliers / total profiles sampled


@dataclass
class RefinedQuad:
    """Refined box-top quad with acceptance flag and diagnostics."""

    corners_px: np.ndarray  # (4,2) [BL, BR, FR, FL]; best effort when rejected
    edges: list[RefinedEdge]  # [back, right, front, left]
    accepted: bool
    reasons: list[str]  # empty when accepted
    coarse_corners_px: np.ndarray  # (4,2) input reference
    metrics: dict


def _as_grayscale(rgb: np.ndarray) -> tuple[np.ndarray, int, int]:
    arr = np.asarray(rgb)
    if arr.ndim == 2:
        gray = arr.astype(np.float64)
    elif arr.ndim == 3 and arr.shape[2] in (3, 4):
        ch = arr[..., :3]
        if ch.dtype == np.uint8:
            gray = cv2.cvtColor(ch, cv2.COLOR_RGB2GRAY).astype(np.float64)
        else:
            weights = np.array([0.299, 0.587, 0.114], dtype=np.float64)
            gray = np.asarray(ch, dtype=np.float64) @ weights
    else:
        raise ValueError(
            f"rgb must be HxW or HxWx3/Wx4, got shape {arr.shape}"
        )
    h, w = gray.shape
    return np.ascontiguousarray(gray), int(w), int(h)


def _as_corners(coarse: np.ndarray) -> np.ndarray:
    pts = np.asarray(coarse, dtype=np.float64)
    if pts.shape != (4, 2):
        raise ValueError(f"coarse_corners_px must have shape (4, 2), got {pts.shape}")
    if not np.all(np.isfinite(pts)):
        raise ValueError("coarse_corners_px must be finite")
    for k in range(4):
        if float(np.linalg.norm(pts[(k + 1) % 4] - pts[k])) < 1e-9:
            raise ValueError(f"degenerate coarse edge at corner index {k}")
    return pts


def _bilinear_values(gray: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Vectorized bilinear interpolation; callers guarantee in-bounds coords."""
    h, w = gray.shape
    x0 = np.floor(xs).astype(np.int64)
    y0 = np.floor(ys).astype(np.int64)
    dx = xs - x0
    dy = ys - y0
    x0c = np.clip(x0, 0, w - 1)
    y0c = np.clip(y0, 0, h - 1)
    x1c = np.clip(x0 + 1, 0, w - 1)
    y1c = np.clip(y0 + 1, 0, h - 1)
    ia = gray[y0c, x0c]
    ib = gray[y0c, x1c]
    ic = gray[y1c, x0c]
    id_ = gray[y1c, x1c]
    return (1.0 - dx) * (1.0 - dy) * ia + dx * (1.0 - dy) * ib + (1.0 - dx) * dy * ic + dx * dy * id_


def _sample_edge(
    gray: np.ndarray,
    w: int,
    h: int,
    p0: np.ndarray,
    p1: np.ndarray,
    half: int,
    spacing: float,
    min_gradient: float,
    max_offset: float,
) -> list[EdgeSample]:
    """Sample normal profiles along one coarse edge segment."""
    delta = p1 - p0
    length = float(np.linalg.norm(delta))
    direction = delta / length
    normal = np.array([-direction[1], direction[0]], dtype=np.float64)
    # Stay clear of the vertices where the adjacent edge corrupts the profile.
    margin = max(float(half), 0.05 * length)
    stations = np.arange(margin, length - margin + 1e-9, spacing, dtype=np.float64)
    across = np.arange(-half, half + 1, dtype=np.float64)

    samples: list[EdgeSample] = []
    for t in stations:
        center = p0 + direction * float(t)
        xs = center[0] + normal[0] * across
        ys = center[1] + normal[1] * across
        if (
            np.any(xs < 0.0)
            or np.any(xs > w - 1)
            or np.any(ys < 0.0)
            or np.any(ys > h - 1)
        ):
            samples.append(
                EdgeSample(
                    coarse_px=center.copy(),
                    refined_px=center.copy(),
                    offset_px=0.0,
                    gradient=0.0,
                    valid=False,
                    reason="out_of_bounds",
                )
            )
            continue
        values = _bilinear_values(gray, xs, ys)
        # Central-difference gradient along the profile normal.
        grad = np.zeros_like(values)
        grad[1:-1] = 0.5 * (values[2:] - values[:-2])
        mag = np.abs(grad)
        # Peak search stays strictly interior so the quadratic fit has neighbors.
        peak_idx = int(np.argmax(mag[1:-1])) + 1
        peak = float(mag[peak_idx])
        if peak < min_gradient:
            samples.append(
                EdgeSample(
                    coarse_px=center.copy(),
                    refined_px=center.copy(),
                    offset_px=0.0,
                    gradient=peak,
                    valid=False,
                    reason="weak_gradient",
                )
            )
            continue
        denom = float(mag[peak_idx - 1] - 2.0 * mag[peak_idx] + mag[peak_idx + 1])
        if denom < -1e-12:
            shift = 0.5 * float(mag[peak_idx - 1] - mag[peak_idx + 1]) / denom
            shift = float(np.clip(shift, -1.0, 1.0))
        else:
            shift = 0.0
        offset = float(across[peak_idx] + shift)
        if abs(offset) > max_offset:
            samples.append(
                EdgeSample(
                    coarse_px=center.copy(),
                    refined_px=center.copy(),
                    offset_px=offset,
                    gradient=peak,
                    valid=False,
                    reason="offset_too_large",
                )
            )
            continue
        peak_value = peak - 0.25 * float(mag[peak_idx - 1] - mag[peak_idx + 1]) * shift
        refined = center + normal * offset
        samples.append(
            EdgeSample(
                coarse_px=center.copy(),
                refined_px=refined,
                offset_px=offset,
                gradient=float(peak_value),
                valid=True,
                reason=_OK,
            )
        )
    return samples


def _robust_tls_fit(
    points: np.ndarray, min_samples: int
) -> tuple[np.ndarray, int, float] | None:
    """Total-least-squares line with deterministic iterative MAD rejection.

    Returns (line_abc, n_inliers, residual_rms_px) or None when too few
    points survive.
    """
    n_total = int(points.shape[0])
    if n_total < min_samples:
        return None
    keep = np.ones(n_total, dtype=bool)
    for _ in range(10):
        block = points[keep]
        mean = block.mean(axis=0)
        _, _, vt = np.linalg.svd(block - mean, full_matrices=False)
        normal = vt[-1]
        dist_all = np.abs((points - mean) @ normal)
        dist_in = dist_all[keep]
        median = float(np.median(dist_in))
        mad = float(np.median(np.abs(dist_in - median)))
        sigma = 1.4826 * mad
        if not np.isfinite(sigma) or sigma < 1e-9:
            break
        threshold = max(median + 2.5 * sigma, 1.0)
        candidate = dist_all <= threshold
        if int(candidate.sum()) < min_samples:
            break  # rejecting further would starve the fit; keep current set
        if bool(np.array_equal(candidate, keep)):
            keep = candidate
            break
        keep = candidate
    if int(keep.sum()) < min_samples:
        return None
    block = points[keep]
    mean = block.mean(axis=0)
    _, _, vt = np.linalg.svd(block - mean, full_matrices=False)
    a, b = float(vt[-1, 0]), float(vt[-1, 1])
    c = float(-(a * mean[0] + b * mean[1]))
    if a < 0.0 or (a == 0.0 and b < 0.0):
        a, b, c = -a, -b, -c
    rms = float(np.sqrt(np.mean(((block - mean) @ np.array([a, b])) ** 2)))
    return np.array([a, b, c], dtype=np.float64), int(keep.sum()), rms


def _intersect(l1: np.ndarray, l2: np.ndarray) -> np.ndarray | None:
    a1, b1, c1 = (float(v) for v in l1)
    a2, b2, c2 = (float(v) for v in l2)
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-12:
        return None
    x = (b1 * c2 - b2 * c1) / det
    y = (c1 * a2 - c2 * a1) / det
    return np.array([x, y], dtype=np.float64)


def _is_convex(quad: np.ndarray) -> bool:
    signs = []
    for k in range(4):
        e1 = quad[(k + 1) % 4] - quad[k]
        e2 = quad[(k + 2) % 4] - quad[(k + 1) % 4]
        signs.append(float(e1[0] * e2[1] - e1[1] * e2[0]))
    return all(s > 1e-9 for s in signs) or all(s < -1e-9 for s in signs)


def refine_quad_edges(
    rgb: np.ndarray,
    coarse_corners_px: np.ndarray,
    profile_half_width_px: int = 12,
    sample_spacing_px: float = 6,
    min_gradient: float = 4,
    max_offset_px: float = 15,
    min_samples: int = 8,
) -> RefinedQuad:
    """Refine four box-top edges from the original RGB frame.

    Args:
        rgb: HxW grayscale or HxWx3/4 image.
        coarse_corners_px: (4, 2) reference corners [BL, BR, FR, FL].
        profile_half_width_px: half-length of each normal profile in pixels.
        sample_spacing_px: station spacing along each coarse edge.
        min_gradient: minimum gradient magnitude to accept a profile peak.
        max_offset_px: maximum accepted subpixel offset from the coarse edge.
        min_samples: minimum TLS inliers required per edge.

    Returns:
        RefinedQuad with corners from adjacent line intersections. When the
        data cannot support a refinement, ``accepted`` is False with explicit
        ``reasons``; the coarse input is never silently returned as success.
    """
    half = int(round(profile_half_width_px))
    spacing = float(sample_spacing_px)
    min_gradient = float(min_gradient)
    max_offset = float(max_offset_px)
    min_samples = int(min_samples)
    if half < 2:
        raise ValueError(f"profile_half_width_px must be >= 2, got {profile_half_width_px}")
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError(f"sample_spacing_px must be positive, got {sample_spacing_px}")
    if not np.isfinite(max_offset) or max_offset <= 0.0:
        raise ValueError(f"max_offset_px must be positive, got {max_offset_px}")
    if min_samples < 2:
        raise ValueError(f"min_samples must be >= 2, got {min_samples}")

    gray, w, h = _as_grayscale(rgb)
    coarse = _as_corners(coarse_corners_px)

    edges: list[RefinedEdge] = []
    lines: list[np.ndarray | None] = []
    reasons: list[str] = []
    residual_map: dict[str, float] = {}
    support_map: dict[str, float] = {}
    valid_map: dict[str, int] = {}
    total_map: dict[str, int] = {}

    for name, (i0, i1) in zip(EDGE_NAMES, _EDGE_PAIRS):
        samples = _sample_edge(
            gray, w, h, coarse[i0], coarse[i1],
            half, spacing, min_gradient, max_offset,
        )
        n_total = len(samples)
        valid_pts = np.array(
            [s.refined_px for s in samples if s.valid], dtype=np.float64
        )
        n_valid = int(valid_pts.shape[0])
        fit = _robust_tls_fit(valid_pts, min_samples) if n_valid else None
        if fit is None:
            lines.append(None)
            residual_map[name] = float("nan")
            support_map[name] = 0.0
            edges.append(
                RefinedEdge(
                    name=name,
                    line_abc=np.full(3, np.nan, dtype=np.float64),
                    samples=samples,
                    residual_rms_px=float("nan"),
                    support_score=0.0,
                )
            )
            reasons.append(
                f"{name}: only {n_valid}/{n_total} valid profiles "
                f"(< {min_samples} inliers required)"
            )
        else:
            line_abc, n_inliers, rms = fit
            lines.append(line_abc)
            support = float(n_inliers) / float(n_total) if n_total else 0.0
            residual_map[name] = rms
            support_map[name] = support
            edges.append(
                RefinedEdge(
                    name=name,
                    line_abc=line_abc,
                    samples=samples,
                    residual_rms_px=rms,
                    support_score=support,
                )
            )
        valid_map[name] = n_valid
        total_map[name] = n_total

    if any(line is None for line in lines):
        return RefinedQuad(
            corners_px=coarse.copy(),
            edges=edges,
            accepted=False,
            reasons=reasons,
            coarse_corners_px=coarse.copy(),
            metrics={
                "residual_rms_px": residual_map,
                "support_score": support_map,
                "n_valid": valid_map,
                "n_total": total_map,
                "corner_shift_px": [0.0, 0.0, 0.0, 0.0],
                "mean_corner_shift_px": 0.0,
            },
        )

    assert all(line is not None for line in lines)
    back, right, front, left = (np.asarray(line, dtype=np.float64) for line in lines)
    pairs = ((left, back), (back, right), (right, front), (front, left))
    pair_names = (
        "left+back",
        "back+right",
        "right+front",
        "front+left",
    )
    corners: list[np.ndarray] = []
    for (line_a, line_b), label in zip(pairs, pair_names):
        hit = _intersect(line_a, line_b)
        if hit is None:
            reasons.append(f"parallel_edges: {label} are parallel")
            corners.append(np.full(2, np.nan, dtype=np.float64))
        else:
            corners.append(hit)
    refined = np.array(corners, dtype=np.float64)

    if not np.all(np.isfinite(refined)):
        reasons.append("non_finite_refinement: an edge intersection is not finite")
    elif not _is_convex(refined):
        reasons.append("non_convex_refinement: intersections do not form a convex quad")
    elif np.any(refined[:, 0] < 0.0) or np.any(refined[:, 0] > w - 1) or np.any(
        refined[:, 1] < 0.0
    ) or np.any(refined[:, 1] > h - 1):
        reasons.append("refined_corner_out_of_bounds: intersections leave the image")

    accepted = not reasons
    shifts = (
        np.linalg.norm(refined - coarse, axis=1)
        if np.all(np.isfinite(refined))
        else np.full(4, np.nan, dtype=np.float64)
    )
    return RefinedQuad(
        corners_px=refined if accepted else refined.copy(),
        edges=edges,
        accepted=accepted,
        reasons=reasons,
        coarse_corners_px=coarse.copy(),
        metrics={
            "residual_rms_px": residual_map,
            "support_score": support_map,
            "n_valid": valid_map,
            "n_total": total_map,
            "corner_shift_px": [float(v) for v in shifts],
            "mean_corner_shift_px": float(np.nanmean(shifts)),
        },
    )
