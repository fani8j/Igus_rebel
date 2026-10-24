# Provenance: ported from rgbd_captures/tests/test_tape_strip.py (HEAD 6df5994); imports made package-relative.
"""Synthetic tests for the tape-strip detector (scripts/autolabel_v2/tape_strip.py)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest


from box_seam_detection.core.tape_strip import (best_candidate, candidate_lines_image,  # noqa: E402
                                     detect_tape_strips,
                                     mm_per_px_from_xyz)
from synth.face_gen import make_face  # noqa: E402

MMPP = 0.8
MATERIALS = ["clear", "teflon", "brown", "white", "printed"]
BASES = ["brown", "white"]
W0, H0 = 300, 240


def _strip(axis="h", material="teflon", center=None, width=60.0, yaw=0.0,
           **kw):
    n = H0 if axis == "h" else W0
    d = dict(axis=axis, material=material,
             center_px=n / 2.0 if center is None else center,
             width_px=width, yaw_deg=yaw)
    d.update(kw)
    return d


def _detect(strips, base="brown", seed=0, **kw):
    f, truth = make_face(seed, W0, H0, base=base, strips=strips, **kw)
    return detect_tape_strips(f, MMPP), truth


def _axis_name(axis):
    return "lr" if axis == "h" else "bf"


@pytest.mark.parametrize("base", BASES)
@pytest.mark.parametrize("material", MATERIALS)
def test_single_strip_each_material(base, material):
    for k, (axis, yaw, wpx, off) in enumerate(
            [("h", 1.3, 60.0, 0.0), ("v", -2.2, 30.0, 17.0)]):
        n = H0 if axis == "h" else W0
        c = n / 2.0 + off
        res, _ = _detect([_strip(axis, material, c, wpx, yaw)], base, seed=k)
        assert res["ok"], res["reasons"]
        assert res["best_axis"] == _axis_name(axis)
        b = best_candidate(res)
        assert abs(b["center_px"] - c) <= 1.0
        assert abs(b["width_px"] - wpx) / wpx <= 0.10
        assert abs(b["width_mm"] - wpx * MMPP) <= 0.10 * wpx * MMPP
        assert abs(b["yaw_deg"] - yaw) <= 0.6
        assert b["n_channels_agree"] >= 2
        assert res["axis_margin"] >= 0.25


@pytest.mark.parametrize("base", BASES)
def test_blank_face_no_detection(base):
    for seed in range(3):
        res, _ = _detect([], base, seed=seed)
        assert not res["ok"]
        assert res["best_axis"] is None
        assert res["n_strips_per_axis"] == {"lr": 0, "bf": 0}
        assert all(not c["qualified"] for c in res["candidates"])


def test_blank_face_noisy_no_detection():
    res, _ = _detect([], "brown", seed=5, noise=2.0)
    assert not res["ok"]


def test_two_parallel_strips_both_found():
    cs = (80.0, 170.0)
    res, _ = _detect([_strip("h", "teflon", cs[0], 24.0, 0.5),
                      _strip("h", "clear", cs[1], 36.0, 0.5)], "brown")
    assert res["n_strips_per_axis"]["lr"] == 2
    found = sorted(c["center_px"] for c in res["candidates"]
                   if c["qualified"] and c["axis"] == "lr")
    assert len(found) == 2
    for f, t in zip(found, cs):
        assert abs(f - t) <= 1.0
    assert "multiple_strips_same_axis" in res["reasons"]


def test_two_perpendicular_strips_flagged_ambiguous():
    res, _ = _detect([_strip("h", "teflon", 120.0, 40.0),
                      _strip("v", "teflon", 150.0, 40.0)], "brown")
    assert res["n_strips_per_axis"] == {"lr": 1, "bf": 1}
    assert not res["ok"]
    assert "ambiguous_axis" in res["reasons"]
    assert "strips_on_both_axes" in res["reasons"]


@pytest.mark.parametrize("material", ["clear", "teflon"])
@pytest.mark.parametrize("base", BASES)
@pytest.mark.parametrize("streak", [
    dict(rel=0.0, width_px=5, length_frac=0.35, start_frac=0.3),
    dict(rel=0.5, width_px=4, length_frac=0.3, start_frac=0.4),
    dict(rel=1.0, width_px=5, length_frac=0.3, start_frac=0.4),   # across edge
    dict(rel=-1.0, width_px=5, length_frac=0.3, start_frac=0.4),
    dict(rel=0.3, width_px=4, length_frac=1.0, start_frac=0.0),   # full length
])
def test_specular_streak_does_not_move_centreline(material, base, streak):
    c = 135.0
    plain, _ = _detect([_strip("h", material, c, 48.0, 1.0)], base, seed=3)
    glossy, _ = _detect([_strip("h", material, c, 48.0, 1.0, streak=streak)],
                        base, seed=3)
    bp, bg = best_candidate(plain), best_candidate(glossy)
    assert bp is not None and bg is not None
    assert abs(bg["center_px"] - c) <= 1.0
    assert abs(bg["center_px"] - bp["center_px"]) <= 1.0
    assert abs(bg["width_px"] - 48.0) / 48.0 <= 0.10
    assert glossy["best_axis"] == "lr"


@pytest.mark.parametrize("off", [-70.0, -35.0, 40.0, 75.0])
@pytest.mark.parametrize("axis", ["h", "v"])
def test_off_centre_strip_found_at_true_offset(axis, off):
    n = H0 if axis == "h" else W0
    c = n / 2.0 + off
    res, _ = _detect([_strip(axis, "teflon", c, 30.0, -1.5)], "brown")
    b = best_candidate(res)
    assert res["ok"] and b["axis"] == _axis_name(axis)
    assert abs(b["center_px"] - c) <= 1.0


def test_gap_valley_reported_separately():
    gap = dict(offset_px=4.0, width_px=2.0, depth=0.5)
    res, truth = _detect([_strip("h", "clear", 120.0, 48.0, 0.0, gap=gap)],
                         "brown")
    b = best_candidate(res)
    assert b["gap_line"] is not None
    assert abs(b["gap_line"]["center_px"] - 124.0) <= 1.5
    assert abs(b["gap_line"]["offset_px"] - 4.0) <= 1.5
    assert b["gap_line"]["inside_band"]
    assert abs(b["center_px"] - 120.0) <= 1.0     # centre is from the edges


def test_gap_does_not_split_wide_strip():
    gap = dict(offset_px=0.0, width_px=3.0, depth=0.5)
    res, _ = _detect([_strip("h", "teflon", 120.0, 90.0, 1.0, gap=gap)],
                     "white")
    b = best_candidate(res)
    assert abs(b["center_px"] - 120.0) <= 1.0
    assert abs(b["width_px"] - 90.0) / 90.0 <= 0.10


def test_label_occluder_partial_coverage_still_detected():
    occ = dict(x0=60, y0=100, x1=140, y1=190, color=(240, 240, 240))
    res, _ = _detect([_strip("h", "teflon", 120.0, 40.0)], "brown",
                     occluder=occ)
    b = best_candidate(res)
    assert b is not None and b["axis"] == "lr"
    assert abs(b["center_px"] - 120.0) <= 1.5


def test_unknown_scale_uses_face_fraction():
    f, _ = make_face(0, W0, H0, base="brown",
                     strips=[_strip("h", "teflon", 110.0, 40.0)])
    res = detect_tape_strips(f, None)
    b = best_candidate(res)
    assert res["ok"] and abs(b["center_px"] - 110.0) <= 1.0
    assert b["width_mm"] is None


def test_width_range_is_configurable():
    f, _ = make_face(0, W0, H0, base="brown",
                     strips=[_strip("h", "teflon", 120.0, 8.0)])
    default = detect_tape_strips(f, MMPP)          # 6.4 mm < 10 mm minimum
    legacy = detect_tape_strips(f, MMPP, {"min_width_mm": 4.0,
                                          "max_width_mm": 15.0})
    assert not default["ok"]
    assert legacy["ok"]
    assert abs(best_candidate(legacy)["center_px"] - 120.0) <= 1.0


def test_result_is_json_serialisable_and_lines_map_to_image():
    res, _ = _detect([_strip("h", "teflon", 120.0, 40.0, 1.0,
                             gap=dict(offset_px=0, width_px=2))], "brown")
    json.dumps(res)
    Mi = np.eye(3)
    lines = candidate_lines_image(res, Mi)
    assert len(lines) == len(res["candidates"])
    b = best_candidate(res)
    i = res["candidates"].index(b)
    p0, p1 = lines[i]["centre"]
    assert abs(p0[0] - 0.0) < 1.0 and abs(p1[0] - W0) < 1.0
    assert abs((p0[1] + p1[1]) / 2 - 120.0) <= 1.5
    assert lines[i]["gap"] is not None


def test_mm_per_px_from_xyz():
    xyz = {"back_left": [0.1, 0, 0.5], "back_right": [0.1, 0.3, 0.5],
           "front_right": [-0.1, 0.3, 0.5], "front_left": [-0.1, 0, 0.5]}
    mx, my = mm_per_px_from_xyz(xyz, 300, 200)
    assert mx == pytest.approx(300.0 / 300)     # 0.3 m over 300 px
    assert my == pytest.approx(200.0 / 200)
    assert mm_per_px_from_xyz(None, 300, 200) is None
