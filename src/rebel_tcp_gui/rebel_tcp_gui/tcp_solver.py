"""Pivot-calibration solver for the knife TCP position.

Solves ``p_i + R_i r = c`` for the TCP translation ``r`` in ``igus_tool0``
and the stationary pivot ``c`` in ``base_link``, where ``(p_i, R_i)`` are
the live ``base_link -> igus_tool0`` translation (metres) and rotation.

Mirrors the math proven in ``tcp_calibration_tutorial.html``: 6x6 normal
equations solved with partial-pivot Gaussian elimination.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

_ORTHONORMAL_TOL = 1e-3
_PIVOT_TOL = 1e-11
_MIN_POSES = 5


def _as_pose(pose, index: int) -> Tuple[np.ndarray, np.ndarray]:
    """Validate one pose entry and return ``(p, R)`` as float arrays."""
    try:
        p_in, R_in = pose
    except (TypeError, ValueError):
        raise ValueError(
            f"Pose {index}: expected a (p, R) pair with a length-3 translation "
            "and a 3x3 rotation matrix."
        ) from None
    try:
        p = np.asarray(p_in, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(
            f"Pose {index}: translation is not numeric; expected 3 values in metres."
        ) from None
    try:
        R = np.asarray(R_in, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(
            f"Pose {index}: rotation is not numeric; expected a 3x3 matrix."
        ) from None
    if p.shape != (3,):
        raise ValueError(
            f"Pose {index}: translation must have exactly 3 values in metres, "
            f"got shape {p.shape}."
        )
    if R.shape != (3, 3):
        raise ValueError(
            f"Pose {index}: rotation must be a 3x3 matrix, got shape {R.shape}."
        )
    if not (np.all(np.isfinite(p)) and np.all(np.isfinite(R))):
        raise ValueError(
            f"Pose {index}: pose contains NaN or infinite values; "
            "re-record that sample."
        )
    if not np.allclose(R @ R.T, np.eye(3), atol=_ORTHONORMAL_TOL):
        raise ValueError(
            f"Pose {index}: rotation is not orthonormal (R @ R.T != I, "
            "tol 1e-3); check the TF quaternion source."
        )
    det = float(np.linalg.det(R))
    if abs(det - 1.0) > _ORTHONORMAL_TOL:
        raise ValueError(
            f"Pose {index}: rotation determinant is {det:.6f}, expected +1 "
            "(tol 1e-3); a reflection (det = -1) is not a valid rotation."
        )
    return p, R


def _solve_6x6(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve ``A x = b`` with partial-pivot Gaussian elimination.

    Raises:
        ValueError: if a pivot is ~zero, i.e. the samples do not constrain
            all three TCP coordinates (insufficient orientation spread).
    """
    A = A.astype(float, copy=True)
    b = b.astype(float, copy=True)
    n = b.shape[0]
    for k in range(n):
        pivot = int(np.argmax(np.abs(A[k:, k]))) + k
        if abs(A[pivot, k]) < _PIVOT_TOL:
            raise ValueError(
                "Samples do not constrain all three TCP coordinates "
                "(singular normal equations). Capture more poses with widely "
                "separated orientations while keeping the tip on the pivot."
            )
        if pivot != k:
            A[[k, pivot]] = A[[pivot, k]]
            b[[k, pivot]] = b[[pivot, k]]
        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            A[i, k:] -= factor * A[k, k:]
            b[i] -= factor * b[k]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - A[i, i + 1:] @ x[i + 1:]) / A[i, i]
    return x


def solve_tcp(poses: Sequence) -> dict:
    """Solve the pivot calibration from sampled ``(p, R)`` poses.

    Args:
        poses: list of ``(p, R)`` where ``p`` is a length-3 array-like
            ``base_link -> igus_tool0`` translation in metres and ``R`` a
            3x3 array-like rotation matrix.

    Returns:
        Dict with ``r_m`` (3-tuple, TCP in ``igus_tool0``, metres),
        ``pivot_m`` (3-tuple, fixed point in ``base_link``, metres),
        ``rms_mm`` (float), ``max_mm`` (float) and ``residuals_mm``
        (list of float, one per pose, same order).

    Raises:
        ValueError: on fewer than 5 poses, non-finite values,
            non-rotation matrices, or under-constrained (singular) systems.
    """
    try:
        n = len(poses)  # type: ignore[arg-type]
    except TypeError:
        raise ValueError(
            "poses must be a sequence of (p, R) pairs with at least 5 entries."
        ) from None
    if n < _MIN_POSES:
        raise ValueError(
            f"Capture at least 5 measured poses; got {n}. "
            "8-12 poses with widely separated orientations are preferred."
        )
    validated = [_as_pose(pose, i) for i, pose in enumerate(poses)]

    ata = np.zeros((6, 6))
    atb = np.zeros(6)
    for p, R in validated:
        for k in range(3):
            a = np.zeros(6)
            a[:3] = R[k]
            a[3 + k] = -1.0
            b = -p[k]
            ata += np.outer(a, a)
            atb += a * b
    x = _solve_6x6(ata, atb)
    r = x[:3]
    c = x[3:]

    residuals_m = [
        float(np.linalg.norm(p + R @ r - c)) for p, R in validated
    ]
    residuals_mm = [e * 1000.0 for e in residuals_m]
    rms_mm = float(np.sqrt(sum(e * e for e in residuals_m) / n) * 1000.0)
    max_mm = float(max(residuals_m) * 1000.0) if residuals_m else 0.0
    return {
        "r_m": (float(r[0]), float(r[1]), float(r[2])),
        "pivot_m": (float(c[0]), float(c[1]), float(c[2])),
        "rms_mm": rms_mm,
        "max_mm": max_mm,
        "residuals_mm": residuals_mm,
    }
