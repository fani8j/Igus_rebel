"""Pure tests for side_tape_ros.handoff (uses a tmp copy of the example record)."""

from __future__ import annotations

import json
import os
import shutil
from types import SimpleNamespace

import pytest

from side_tape_ros.handoff import check_handoff, load_handoff

EXAMPLE = os.path.expanduser("~/.ros/side_tape_handoff.json")
CFG = SimpleNamespace(cut_axis_tool=[0.0, -1.0, 0.0], lateral_axis_tool=[1.0, 0.0, 0.0],
                      depth_axis_tool=[0.0, 0.0, 1.0])


@pytest.fixture
def rec_path(tmp_path):
    if not os.path.exists(EXAMPLE):
        pytest.skip("example hand-off record not present")
    dst = tmp_path / "h.json"
    shutil.copy(EXAMPLE, dst)
    return dst


def _edit(path, fn):
    d = json.loads(path.read_text())
    fn(d)
    path.write_text(json.dumps(d))


def _check(path, pos="rec", quat="rec", **kw):
    rec = load_handoff(str(path))
    if pos == "rec":
        pos = rec.position_m
    if quat == "rec":
        quat = rec.orientation_xyzw
    return check_handoff(rec, CFG, pos, quat, **kw)


def test_good_record(rec_path):
    assert _check(rec_path) == []


def test_no_current_pose_skips_pose_checks(rec_path):
    assert _check(rec_path, pos=None, quat=None) == []


def test_wrong_status(rec_path):
    _edit(rec_path, lambda d: d.update(status="consumed"))
    assert any("status" in s for s in _check(rec_path))


def test_bad_variant(rec_path):
    _edit(rec_path, lambda d: d.update(variant="nope"))
    assert any("variant" in s for s in _check(rec_path))


def test_old_axis_convention_fails(rec_path):
    def f(d):
        c = d["side_tape_convention"]
        c["lateral_axis_tool"] = {"x": 0, "y": 0, "z": 1}
        c["depth_axis_tool"] = {"x": -1, "y": 0, "z": 0}
    _edit(rec_path, f)
    probs = _check(rec_path)
    assert any("lateral_axis_tool" in s for s in probs)
    assert any("depth_axis_tool" in s for s in probs)


def test_large_search_angle(rec_path):
    _edit(rec_path, lambda d: d["left_parallel_orientation"].update(max_abs_angle_deg=12.0))
    assert any("max_abs_angle_deg" in s for s in _check(rec_path))
    _edit(rec_path, lambda d: d["left_parallel_orientation"].update(
        max_abs_angle_deg=5.0, rotation_from_f0_deg=11.0))
    assert any("rotation_from_f0_deg" in s for s in _check(rec_path))


def test_missing_orientation_block(rec_path):
    _edit(rec_path, lambda d: d.pop("left_parallel_orientation"))
    assert any("left_parallel_orientation" in s for s in _check(rec_path))


def test_pose_too_far(rec_path):
    rec = load_handoff(str(rec_path))
    far = (rec.position_m[0] + 0.05, rec.position_m[1], rec.position_m[2])
    assert any("mm from" in s for s in _check(rec_path, pos=far))
    # 30 deg about z-ish: rotate quaternion by composing with a big rotation
    import math
    h = math.radians(40) / 2
    qz = (0.0, 0.0, math.sin(h), math.cos(h))
    x1, y1, z1, w1 = qz
    x2, y2, z2, w2 = rec.orientation_xyzw
    q = (w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
         w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2)
    assert any("orientation differs" in s for s in _check(rec_path, quat=q))


def test_nonpositive_cut_length(rec_path):
    _edit(rec_path, lambda d: d.update(cut_length_mm=0.0))
    assert any("cut_length_mm" in s for s in _check(rec_path))


def test_wrong_frames(rec_path):
    _edit(rec_path, lambda d: d.update(planning_frame="world", tool_link="other"))
    probs = _check(rec_path)
    assert any("planning_frame" in s for s in probs) and any("tool_link" in s for s in probs)


def test_missing_keys_raise(rec_path):
    _edit(rec_path, lambda d: d.pop("start_pose"))
    with pytest.raises(ValueError, match="start_pose"):
        load_handoff(str(rec_path))
    _edit(rec_path, lambda d: d.update(schema="other/v9"))
    with pytest.raises(ValueError, match="schema"):
        load_handoff(str(rec_path))
