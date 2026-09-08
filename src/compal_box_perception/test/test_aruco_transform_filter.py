import math

import numpy as np

from compal_box_perception.aruco_transform_filter import TransformMedianFilter


def _yaw_quaternion(degrees):
    angle = math.radians(degrees) / 2.0
    return (0.0, 0.0, math.sin(angle), math.cos(angle))


def test_filter_returns_translation_median_and_quaternion_mean():
    filter_ = TransformMedianFilter(3, 0.010, 5.0)
    assert filter_.update((0.000, 0.000, 0.200), _yaw_quaternion(0.0)) is not None
    assert filter_.update((0.002, -0.002, 0.202), _yaw_quaternion(2.0)) is not None
    translation, quaternion = filter_.update(
        (0.001, -0.001, 0.201), _yaw_quaternion(1.0)
    )
    np.testing.assert_allclose(translation, (0.001, -0.001, 0.201))
    assert abs(quaternion[2] - math.sin(math.radians(1.0) / 2.0)) < 1e-6
    assert abs(quaternion[3] - math.cos(math.radians(1.0) / 2.0)) < 1e-6


def test_filter_rejects_translation_and_rotation_outliers():
    filter_ = TransformMedianFilter(5, 0.010, 5.0)
    assert filter_.update((0.0, 0.0, 0.2), _yaw_quaternion(0.0)) is not None
    assert filter_.update((0.020, 0.0, 0.2), _yaw_quaternion(0.0)) is None
    assert filter_.update((0.0, 0.0, 0.2), _yaw_quaternion(10.0)) is None


def test_filter_treats_opposite_quaternion_signs_as_same_rotation():
    filter_ = TransformMedianFilter(3, 0.010, 5.0)
    quaternion = np.asarray(_yaw_quaternion(3.0))
    assert filter_.update((0.0, 0.0, 0.2), quaternion) is not None
    _, result = filter_.update((0.0, 0.0, 0.2), -quaternion)
    np.testing.assert_allclose(result, quaternion)
