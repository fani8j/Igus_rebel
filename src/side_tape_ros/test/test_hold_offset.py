import pytest

from side_tape_ros import hold_offset as h


def test_median_spread():
    med, sp = h.median_and_spread([1.0, 1.1, 1.2, 1.0, 1.1])
    assert med == pytest.approx(1.1)
    assert sp == pytest.approx(h.percentile([1.0, 1.0, 1.1, 1.1, 1.2], 90) - h.percentile([1.0, 1.0, 1.1, 1.1, 1.2], 10))
    assert h.percentile([3.0], 50) == 3.0
    assert h.percentile([0.0, 10.0], 50) == pytest.approx(5.0)


def test_plan_trim_adds_median_to_current():
    new, med, sp = h.plan_hold_trim([0.5, 0.52, 0.48, 0.5, 0.51], 0.25, 5)
    assert med == pytest.approx(0.5)
    assert new == pytest.approx(0.75)
    assert sp < 0.3


def test_plan_trim_refuses_wide_spread_and_few_frames():
    with pytest.raises(ValueError, match="spread"):
        h.plan_hold_trim([0.0, 0.2, 0.4, 0.6, 0.8], 0.0, 5)
    with pytest.raises(ValueError, match="too few"):
        h.plan_hold_trim([0.5, 0.5], 0.0, 5)
    with pytest.raises(ValueError):
        h.plan_hold_trim([], 0.0, 0)


def test_arg_rules():
    assert h.check_hold_args(False, "air_align") is None
    assert h.check_hold_args(True, "in_material") is None
    assert "air_align" in h.check_hold_args(True, "air_align")
    assert h.check_hold_args(True, "in_material", align_only=True) is not None
    assert h.check_hold_args(True, "in_material", measure_only=True) is not None
