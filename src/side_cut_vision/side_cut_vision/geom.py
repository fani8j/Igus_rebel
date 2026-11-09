"""Small rigid-transform and plane helpers (pure numpy). All lengths in metres unless a name ends in _mm."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def quat_xyzw_to_matrix(q) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(4)
    n = np.linalg.norm(q)
    if not np.isfinite(n) or n < 1e-9:
        raise ValueError("zero or non-finite quaternion")
    x, y, z, w = q / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=float)
    tr = np.trace(R)
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def make_T(R: np.ndarray, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, dtype=float)
    T[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return T


def inv_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def is_valid_T(T) -> bool:
    """True for a finite 4x4 rigid transform (orthonormal rotation, bottom row 0 0 0 1)."""
    try:
        T = np.asarray(T, dtype=float)
    except (TypeError, ValueError):
        return False
    if T.shape != (4, 4) or not np.all(np.isfinite(T)):
        return False
    R = T[:3, :3]
    if np.abs(R @ R.T - np.eye(3)).max() > 1e-3 or abs(np.linalg.det(R) - 1.0) > 1e-3:
        return False
    return bool(np.allclose(T[3], [0, 0, 0, 1], atol=1e-9))


def transform_points(T: np.ndarray, P: np.ndarray) -> np.ndarray:
    P = np.asarray(P, dtype=float)
    return P @ T[:3, :3].T + T[:3, 3]


def rot_about_axis(axis, angle_rad: float) -> np.ndarray:
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle_rad) * K + (1 - np.cos(angle_rad)) * (K @ K)


def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if not np.isfinite(n) or n < 1e-12:
        raise ValueError("zero-length vector")
    return v / n


def angle_deg(a, b) -> float:
    a = unit(a)
    b = unit(b)
    return float(np.degrees(np.arccos(np.clip(a @ b, -1.0, 1.0))))


# ---------------------------------------------------------------- planes: n . p = d, |n| = 1
def fit_plane_lsq(P: np.ndarray, weights: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
    """Least-squares plane through points; returns (n, d) with n . p = d. Normal sign is arbitrary."""
    if weights is None:
        c = P.mean(axis=0)
        Q = P - c
    else:
        w = weights / weights.sum()
        c = (P * w[:, None]).sum(axis=0)
        Q = (P - c) * np.sqrt(w)[:, None]
    # smallest right singular vector of the centred points
    _, _, vt = np.linalg.svd(Q, full_matrices=False)
    n = vt[-1]
    return n, float(n @ c)


def ransac_plane(P: np.ndarray, prior_normal: np.ndarray, thr: float, rng: np.random.Generator,
                 iters: int = 200, max_angle_deg: float = 30.0, max_eval: int = 20000):
    """RANSAC plane with a normal prior, followed by two least-squares refits.

    Returns (n, d, inlier_mask) with n oriented along prior_normal, or None when no hypothesis fits.
    The hypotheses are scored on at most max_eval random points; the final inliers use all of P.
    """
    N = len(P)
    if N < 3:
        return None
    centre = P.mean(axis=0)
    Q = P - centre
    sub = Q if N <= max_eval else Q[rng.choice(N, max_eval, replace=False)]
    idx = rng.integers(0, N if N <= max_eval else len(sub), size=(iters, 3))
    src = Q if N <= max_eval else sub
    a, b, c = src[idx[:, 0]], src[idx[:, 1]], src[idx[:, 2]]
    nrm = np.cross(b - a, c - a)
    ln = np.linalg.norm(nrm, axis=1)
    good = ln > 1e-9
    if not good.any():
        return None
    nrm = nrm[good] / ln[good, None]
    a = a[good]
    nrm = nrm * np.where(nrm @ prior_normal < 0, -1.0, 1.0)[:, None]
    cosang = nrm @ prior_normal
    keep = cosang >= np.cos(np.radians(max_angle_deg))
    if not keep.any():
        return None
    nrm = nrm[keep]
    d = np.einsum("ij,ij->i", nrm, a[keep])
    dist = np.abs(sub @ nrm.T - d[None, :])
    score = (dist <= thr).sum(axis=0)
    best = int(np.argmax(score))
    n, dd = nrm[best], float(d[best])
    mask = np.abs(Q @ n - dd) <= thr
    for _ in range(2):
        if mask.sum() < 3:
            return None
        n, dd = fit_plane_lsq(Q[mask])
        if n @ prior_normal < 0:
            n, dd = -n, -dd
        mask = np.abs(Q @ n - dd) <= thr
    if mask.sum() < 3:
        return None
    return n, float(dd + n @ centre), mask


def plane_rms(P: np.ndarray, n: np.ndarray, d: float) -> float:
    return float(np.sqrt(np.mean((P @ n - d) ** 2))) if len(P) else float("nan")


def ray_plane(origin: np.ndarray, dirs: np.ndarray, n: np.ndarray, d: float) -> np.ndarray:
    """Intersect rays origin + t*dirs with plane n . p = d. Returns (N,3) with NaN where parallel/behind."""
    dirs = np.atleast_2d(dirs)
    denom = dirs @ n
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (d - origin @ n) / denom
    t = np.where((np.abs(denom) > 1e-9) & (t > 0), t, np.nan)
    return origin[None, :] + t[:, None] * dirs


def lsq_line_2d(x: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """Weighted least squares y = c0 + m x (x is centred internally); returns (c0 at x=0, slope m)."""
    if w is None:
        w = np.ones_like(x)
    sw = w.sum()
    xm = (w * x).sum() / sw
    ym = (w * y).sum() / sw
    sxx = (w * (x - xm) ** 2).sum()
    if sxx < 1e-18:
        return float(ym), 0.0
    m = (w * (x - xm) * (y - ym)).sum() / sxx
    return float(ym - m * xm), float(m)


def robust_line_2d(x: np.ndarray, y: np.ndarray, thr: float, rng: np.random.Generator,
                   iters: int = 200):
    """RANSAC (2 points) then IRLS (Tukey bisquare) line fit y = c0 + m x.

    Returns (c0, m, inlier_mask, rms) or None. thr is in the unit of y.
    """
    n = len(x)
    if n < 3:
        return None
    i = rng.integers(0, n, size=(iters, 2))
    dx = x[i[:, 1]] - x[i[:, 0]]
    ok = np.abs(dx) > 1e-9
    if not ok.any():
        return None
    i = i[ok]
    dx = dx[ok]
    m = (y[i[:, 1]] - y[i[:, 0]]) / dx
    c0 = y[i[:, 0]] - m * x[i[:, 0]]
    res = np.abs(y[:, None] - (c0[None, :] + m[None, :] * x[:, None]))
    score = (res <= thr).sum(axis=0)
    b = int(np.argmax(score))
    c, s = float(c0[b]), float(m[b])
    mask = np.abs(y - (c + s * x)) <= thr
    for _ in range(10):
        r = y - (c + s * x)
        scale = max(1.4826 * np.median(np.abs(r[mask])) if mask.any() else thr, 0.2 * thr)
        u = r / (4.685 * scale)
        w = np.where(np.abs(u) < 1, (1 - u ** 2) ** 2, 0.0)
        if (w > 0).sum() < 3:
            break
        c, s = lsq_line_2d(x, y, w)
    r = y - (c + s * x)
    mask = np.abs(r) <= thr
    if mask.sum() < 3:
        return None
    c, s = lsq_line_2d(x[mask], y[mask])
    r = y - (c + s * x)
    mask = np.abs(r) <= thr
    rms = float(np.sqrt(np.mean(r[mask] ** 2))) if mask.any() else float("nan")
    return c, s, mask, rms
