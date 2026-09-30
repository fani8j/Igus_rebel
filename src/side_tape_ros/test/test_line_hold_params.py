"""cut_mode / line_hold ROS-parameter helpers (pure: no rclpy) and the sequencer's --line-hold args."""

from __future__ import annotations

import pytest

from side_tape_ros import control_math as cm
from precision_box_detector.side_tape.line_hold import LineHoldCfg
from precision_box_detector.side_tape.supervisor import Supervisor
from precision_box_detector.side_tape.config import load_config
from side_tape_ros import core_path


def test_defaults_are_vision_mode():
    c = LineHoldCfg()
    assert c.mode == "vision" and c.vision_gain == 0.0
    assert set(cm.LINE_HOLD_PARAMS) == {"cut_mode", "line_vision_gain", "line_min_conf", "line_max_nudge_mm",
                                         "line_check_tol_mm", "line_check_frames"}
    sup = Supervisor(load_config(core_path.DEFAULT_CONFIG))
    assert sup.line_cfg.mode == "vision"


def test_params_apply_and_do_not_mutate_current():
    cur = LineHoldCfg()
    new = cm.line_hold_cfg_from_params(cur, {"cut_mode": "line_hold", "line_vision_gain": 0.2,
                                             "line_check_frames": 7, "line_max_nudge_mm": 1.0})
    assert (new.mode, new.vision_gain, new.check_frames, new.max_nudge_mm) == ("line_hold", 0.2, 7, 1.0)
    assert cur == LineHoldCfg()
    assert cm.line_hold_cfg_from_params(new, {"line_vision_gain": 0}).vision_gain == 0.0  # int -> float


@pytest.mark.parametrize("bad", [
    {"cut_mode": "auto"}, {"cut_mode": 1}, {"line_vision_gain": 1.5}, {"line_vision_gain": -0.1},
    {"line_vision_gain": "0.2"}, {"line_vision_gain": True}, {"line_vision_gain": float("nan")},
    {"line_min_conf": 1.2}, {"line_max_nudge_mm": 10.0}, {"line_max_nudge_mm": -1.0},
    {"line_check_tol_mm": 0.0}, {"line_check_tol_mm": 50.0}, {"line_check_frames": 0}, {"line_check_frames": 1.5},
    {"line_check_frames": True}, {"nonsense": 1},
])
def test_bad_params_are_rejected(bad):
    with pytest.raises(ValueError):
        cm.line_hold_cfg_from_params(LineHoldCfg(), bad)


def test_sequencer_enable_and_restore_params():
    assert cm.line_hold_enable_params(0.0) == {"cut_mode": "line_hold", "line_vision_gain": 0.0}
    assert cm.line_hold_enable_params(0.2) == {"cut_mode": "line_hold", "line_vision_gain": 0.2}
    assert cm.LINE_HOLD_RESTORE_PARAMS == {"cut_mode": "vision", "line_vision_gain": 0.0}
    with pytest.raises(ValueError):
        cm.line_hold_enable_params(2.0)


def test_sequencer_arg_checks():
    assert cm.check_line_hold_args(False, None, False, False) is None
    assert cm.check_line_hold_args(True, None, False, False) is None
    assert cm.check_line_hold_args(True, 0.2, False, False) is None
    assert "needs --line-hold" in cm.check_line_hold_args(False, 0.2, False, False)
    assert "measure-only" in cm.check_line_hold_args(True, None, True, False)
    assert "align-only" in cm.check_line_hold_args(True, None, False, True)
    assert "line-vision-gain" in cm.check_line_hold_args(True, 1.5, False, False)


# -- review findings 4, 5, 6 (pure part) ------------------------------------------------------------

def test_line_max_nudge_ceiling_is_1p5_through_the_ros_params():
    assert cm.line_hold_cfg_from_params(LineHoldCfg(), {"line_max_nudge_mm": 1.5}).max_nudge_mm == 1.5
    for bad in (1.6, 2.0, 3.0):
        with pytest.raises(ValueError):
            cm.line_hold_cfg_from_params(LineHoldCfg(), {"line_max_nudge_mm": bad})


def test_explicit_vision_mode_params_are_valid_and_minimal():
    assert cm.CUT_MODE_VISION_PARAMS == {"cut_mode": "vision"}
    assert cm.line_hold_cfg_from_params(LineHoldCfg(mode="line_hold"), cm.CUT_MODE_VISION_PARAMS).mode == "vision"


def test_params_mismatch():
    want = {"cut_mode": "line_hold", "line_vision_gain": 0.2}
    assert cm.params_mismatch(want, {"cut_mode": "line_hold", "line_vision_gain": 0.2}) == []
    assert cm.params_mismatch(want, {"cut_mode": "line_hold", "line_vision_gain": 0.2 + 1e-12}) == []
    assert len(cm.params_mismatch(want, {"cut_mode": "vision", "line_vision_gain": 0.2})) == 1
    assert len(cm.params_mismatch(want, {"cut_mode": "line_hold", "line_vision_gain": 0.3})) == 1
    assert len(cm.params_mismatch(want, {"cut_mode": "line_hold"})) == 1  # missing = unreadable
    assert len(cm.params_mismatch(want, {"cut_mode": None, "line_vision_gain": None})) == 2
    assert len(cm.params_mismatch(want, {"cut_mode": 1, "line_vision_gain": "0.2"})) == 2  # wrong types
    assert len(cm.params_mismatch({"line_check_frames": 5}, {"line_check_frames": 5.0})) == 0
    assert len(cm.params_mismatch({"x": True}, {"x": 1})) == 1


def test_nan_and_inf_are_rejected_everywhere_in_the_numeric_validation():
    nan, inf = float("nan"), float("inf")
    for name in ("line_vision_gain", "line_min_conf", "line_max_nudge_mm", "line_check_tol_mm"):
        for bad in (nan, inf, -inf):
            with pytest.raises(ValueError, match="finite"):
                cm.line_hold_cfg_from_params(LineHoldCfg(), {name: bad})
    assert "line-vision-gain" in cm.check_line_hold_args(True, nan, False, False)
    # a NaN read back can never "match" (abs(nan - w) > tol is False: it used to pass)
    assert len(cm.params_mismatch({"line_vision_gain": 0.2}, {"line_vision_gain": nan})) == 1
    assert len(cm.params_mismatch({"line_vision_gain": nan}, {"line_vision_gain": nan})) == 1
    assert len(cm.params_mismatch({"line_vision_gain": 0.2}, {"line_vision_gain": inf})) == 1


def test_hold_offset_refuses_nan_samples():
    from side_tape_ros import hold_offset
    for bad in ([0.1] * 14 + [float("nan")], [float("nan")] * 15, [0.1] * 14 + [float("inf")]):
        with pytest.raises(ValueError):
            hold_offset.plan_hold_trim(bad, 0.0, 15)
    assert hold_offset.plan_hold_trim([0.1] * 15, 0.0, 15)[1] == pytest.approx(0.1)
