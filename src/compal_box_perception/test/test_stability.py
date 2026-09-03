import numpy as np

from compal_box_perception.stability import (
    TemporalArrayMedian,
    TemporalScalarMedian,
    TemporalSeamFilter,
    TemporalUnitVectorMedian,
)


def test_filter_requires_consensus_and_returns_coordinate_medians():
    filter_ = TemporalSeamFilter(3, 0.02, 2)
    assert filter_.update(np.array([0.0, 0.0, 0.2]), np.array([0.1, 0.0, 0.2])) is None
    assert filter_.update(np.array([0.002, 0.0, 0.2]), np.array([0.102, 0.0, 0.2])) is None
    result = filter_.update(
        np.array([0.001, 0.0, 0.2]), np.array([0.101, 0.0, 0.2])
    )
    np.testing.assert_allclose(result[0], (0.001, 0.0, 0.2))
    np.testing.assert_allclose(result[1], (0.101, 0.0, 0.2))


def test_large_jump_restarts_confirmation_window():
    filter_ = TemporalSeamFilter(2, 0.02, 2)
    filter_.update(np.array([0.0, 0.0, 0.2]), np.array([0.1, 0.0, 0.2]))
    assert filter_.update(np.array([0.2, 0.0, 0.2]), np.array([0.3, 0.0, 0.2])) is None
    assert filter_.confirmation_count == 1


def test_repeated_invalid_frames_clear_tracking():
    filter_ = TemporalSeamFilter(2, 0.02, 2)
    filter_.update(np.array([0.0, 0.0, 0.2]), np.array([0.1, 0.0, 0.2]))
    assert filter_.mark_invalid() is False
    assert filter_.mark_invalid() is True
    assert filter_.confirmation_count == 0


def test_scalar_median_rejects_large_single_frame_jump():
    filter_ = TemporalScalarMedian(5, 0.04)
    assert filter_.update(0.36) == 0.36
    assert filter_.update(0.37) == 0.365
    assert filter_.update(0.44) == 0.365
    assert filter_.update(0.35) == 0.36


def test_scalar_median_adopts_consistent_persistent_shift():
    filter_ = TemporalScalarMedian(5, 0.04, jump_confirmation=3)
    assert filter_.update(0.36) == 0.36
    assert filter_.update(0.42) == 0.36
    assert filter_.update(0.425) == 0.36
    assert filter_.update(0.43) == 0.425


def test_geometry_median_rejects_large_corner_jump():
    filter_ = TemporalArrayMedian(3, 5.0)
    first = np.array([[10.0, 10.0], [20.0, 10.0]])
    second = np.array([[12.0, 10.0], [22.0, 10.0]])
    jumped = np.array([[40.0, 40.0], [50.0, 40.0]])
    np.testing.assert_allclose(filter_.update(first), first)
    np.testing.assert_allclose(
        filter_.update(second), [[11.0, 10.0], [21.0, 10.0]]
    )
    np.testing.assert_allclose(
        filter_.update(jumped), [[11.0, 10.0], [21.0, 10.0]]
    )


def test_unit_vector_median_aligns_sign_and_rejects_angle_jump():
    filter_ = TemporalUnitVectorMedian(3, 10.0)
    np.testing.assert_allclose(filter_.update([0.0, 0.0, 1.0]), [0.0, 0.0, 1.0])
    np.testing.assert_allclose(filter_.update([0.0, 0.0, -1.0]), [0.0, 0.0, 1.0])
    np.testing.assert_allclose(filter_.update([1.0, 0.0, 0.0]), [0.0, 0.0, 1.0])
