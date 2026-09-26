import math

import numpy as np
import pytest

from compal_box_perception.geometry import (
    average_quaternions,
    normalize,
    normalize_quaternion,
    quaternion_angle_deg,
    quaternion_from_matrix,
    quaternion_multiply,
    quaternion_to_matrix,
)


def _axis_angle_quaternion(axis, degrees):
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    half = math.radians(degrees) / 2.0
    return np.array([*(axis * math.sin(half)), math.cos(half)])


def test_normalize_rejects_zero_and_non_finite_vectors():
    np.testing.assert_allclose(normalize([0.0, 3.0, 4.0]), [0.0, 0.6, 0.8])
    with pytest.raises(ValueError):
        normalize([0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        normalize([np.nan, 0.0, 1.0])


def test_normalize_quaternion_uses_canonical_positive_w():
    np.testing.assert_allclose(
        normalize_quaternion([0.0, 0.0, 0.0, -2.0]), [0.0, 0.0, 0.0, 1.0]
    )
    with pytest.raises(ValueError):
        normalize_quaternion([0.0, 0.0, 0.0, 0.0])


@pytest.mark.parametrize(
    "axis,degrees",
    [((1, 0, 0), 30.0), ((0, 1, 0), 179.0), ((0, 0, 1), -90.0), ((1, 2, 3), 123.0)],
)
def test_matrix_round_trip_and_known_rotation(axis, degrees):
    quaternion = normalize_quaternion(_axis_angle_quaternion(axis, degrees))
    matrix = quaternion_to_matrix(quaternion)
    np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(matrix) == pytest.approx(1.0)
    np.testing.assert_allclose(quaternion_from_matrix(matrix), quaternion, atol=1e-12)


def test_quaternion_to_matrix_rotates_x_into_y_for_positive_yaw():
    matrix = quaternion_to_matrix(_axis_angle_quaternion((0, 0, 1), 90.0))
    np.testing.assert_allclose(matrix @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], atol=1e-12)


def test_multiply_composes_rotations_and_angle_is_sign_invariant():
    first = _axis_angle_quaternion((0, 0, 1), 20.0)
    second = _axis_angle_quaternion((0, 0, 1), 30.0)
    product = quaternion_multiply(first, second)
    np.testing.assert_allclose(
        quaternion_to_matrix(product),
        quaternion_to_matrix(first) @ quaternion_to_matrix(second),
        atol=1e-12,
    )
    assert quaternion_angle_deg(product, first) == pytest.approx(30.0)
    assert quaternion_angle_deg(first, -first) == pytest.approx(0.0, abs=1e-6)


def test_average_quaternions_ignores_sign_flips():
    a = _axis_angle_quaternion((0, 0, 1), 0.0)
    b = _axis_angle_quaternion((0, 0, 1), 4.0)
    mean = average_quaternions([a, -b])
    assert quaternion_angle_deg(mean, _axis_angle_quaternion((0, 0, 1), 2.0)) < 1e-6
    with pytest.raises(ValueError):
        average_quaternions([])
