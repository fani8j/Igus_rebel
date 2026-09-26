"""Shared rotation and vector helpers (ROS xyzw quaternion convention)."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def normalize(vector: Sequence[float]) -> np.ndarray:
    """Return ``vector`` scaled to unit length; reject zero or non-finite input."""
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("Vector must have a finite non-zero length")
    return value / norm


def normalize_quaternion(quaternion: Sequence[float]) -> np.ndarray:
    """Return a unit xyzw quaternion in the canonical ``w >= 0`` hemisphere."""
    value = np.asarray(quaternion, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("Quaternion must have a finite non-zero length")
    value = value / norm
    return -value if value[3] < 0.0 else value


def quaternion_angle_deg(first: Sequence[float], second: Sequence[float]) -> float:
    """Return the rotation angle between two orientations in degrees."""
    dot = abs(float(np.dot(normalize_quaternion(first), normalize_quaternion(second))))
    return math.degrees(2.0 * math.acos(min(1.0, dot)))


def quaternion_to_matrix(quaternion: Sequence[float]) -> np.ndarray:
    """Convert an xyzw quaternion (normalized first) to a 3x3 rotation matrix."""
    x, y, z, w = normalize_quaternion(quaternion)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def quaternion_from_matrix(rotation_matrix: Sequence[Sequence[float]]) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a canonical unit xyzw quaternion."""
    matrix = np.asarray(rotation_matrix, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (matrix[2, 1] - matrix[1, 2]) / scale
        qy = (matrix[0, 2] - matrix[2, 0]) / scale
        qz = (matrix[1, 0] - matrix[0, 1]) / scale
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        qw = (matrix[2, 1] - matrix[1, 2]) / scale
        qx = 0.25 * scale
        qy = (matrix[0, 1] + matrix[1, 0]) / scale
        qz = (matrix[0, 2] + matrix[2, 0]) / scale
    elif matrix[1, 1] > matrix[2, 2]:
        scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        qw = (matrix[0, 2] - matrix[2, 0]) / scale
        qx = (matrix[0, 1] + matrix[1, 0]) / scale
        qy = 0.25 * scale
        qz = (matrix[1, 2] + matrix[2, 1]) / scale
    else:
        scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        qw = (matrix[1, 0] - matrix[0, 1]) / scale
        qx = (matrix[0, 2] + matrix[2, 0]) / scale
        qy = (matrix[1, 2] + matrix[2, 1]) / scale
        qz = 0.25 * scale
    return normalize_quaternion((qx, qy, qz, qw))


def quaternion_multiply(first: Sequence[float], second: Sequence[float]) -> np.ndarray:
    """Return the canonical unit Hamilton product ``first * second`` (xyzw)."""
    x1, y1, z1, w1 = normalize_quaternion(first)
    x2, y2, z2, w2 = normalize_quaternion(second)
    return normalize_quaternion(
        (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        )
    )


def average_quaternions(quaternions: Sequence[Sequence[float]]) -> np.ndarray:
    """Return the sign-invariant (eigenvector) mean of unit quaternions."""
    if len(quaternions) == 0:
        raise ValueError("Cannot average an empty quaternion sequence")
    reference = normalize_quaternion(quaternions[0])
    accumulator = np.zeros((4, 4), dtype=np.float64)
    for quaternion in quaternions:
        aligned = normalize_quaternion(quaternion)
        if np.dot(aligned, reference) < 0.0:
            aligned = -aligned
        accumulator += np.outer(aligned, aligned)
    _, vectors = np.linalg.eigh(accumulator)
    return normalize_quaternion(vectors[:, -1])
