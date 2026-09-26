"""Plain pytest checks for rebel_tcp_gui.tcp_solver (no ROS imports)."""

import math

import numpy as np
import pytest

from rebel_tcp_gui.tcp_solver import solve_tcp

R_TRUE = np.array([0.05, 0.02, 0.10])
C_TRUE = np.array([0.4, 0.1, 0.5])


def _rot(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    t = 1.0 - c
    return np.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ]
    )


def _diverse_rotations():
    rng = np.random.default_rng(0)
    axes = rng.standard_normal((8, 3))
    angles = np.linspace(0.3, 2.6, 8)
    out = []
    for axis, angle in zip(axes, angles):
        out.append(_rot(axis, angle))
    # Add two fixed large-tilt rotations for guaranteed spread.
    out[0] = _rot([1, 0, 0], 0.9)
    out[1] = _rot([0, 1, 0], -1.2)
    return out


def _synthetic_poses(n=8):
    return [(C_TRUE - R @ R_TRUE, R) for R in _diverse_rotations()[:n]]


def test_recovery_of_known_tcp_and_pivot():
    poses = _synthetic_poses(8)
    assert len(poses) >= 6
    result = solve_tcp(poses)
    assert set(result) == {"r_m", "pivot_m", "rms_mm", "max_mm", "residuals_mm"}
    np.testing.assert_allclose(result["r_m"], R_TRUE, atol=1e-9)
    np.testing.assert_allclose(result["pivot_m"], C_TRUE, atol=1e-9)
    assert len(result["residuals_mm"]) == len(poses)
    assert result["rms_mm"] == pytest.approx(0.0, abs=1e-6)
    assert result["max_mm"] == pytest.approx(0.0, abs=1e-6)
    for residual in result["residuals_mm"]:
        assert residual == pytest.approx(0.0, abs=1e-6)


def test_fewer_than_five_poses_rejected():
    with pytest.raises(ValueError):
        solve_tcp(_synthetic_poses(8)[:4])


def test_translation_only_identical_orientations_rejected():
    eye = np.eye(3)
    poses = [
        (C_TRUE - eye @ R_TRUE + np.array([0.01 * i, 0.0, 0.0]), eye.copy())
        for i in range(6)
    ]
    with pytest.raises(ValueError):
        solve_tcp(poses)


def test_nan_input_rejected():
    poses = _synthetic_poses(6)
    bad_p = np.array([float("nan"), 0.0, 0.0])
    poses = [(bad_p, poses[0][1])] + list(poses[1:])
    with pytest.raises(ValueError):
        solve_tcp(poses)


def test_reflection_matrix_rejected():
    reflection = np.diag([1.0, 1.0, -1.0])  # det = -1
    poses = [(C_TRUE - reflection @ R_TRUE, reflection.copy()) for _ in range(6)]
    with pytest.raises(ValueError, match="(?i)reflect|determinant"):
        solve_tcp(poses)
