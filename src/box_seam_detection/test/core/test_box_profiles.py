# Provenance: ported from rgbd_captures/tests/test_box_profiles.py (HEAD 6df5994); imports made package-relative.
"""Box profiles (scripts/autolabel_v2/box_profiles.py) and their routing into
fold_edge / config / seam_estimator / pipeline (multi-box plan M1)."""
import sys
from pathlib import Path

import numpy as np
import pytest


from box_seam_detection.core import box_profiles as bp  # noqa: E402
from box_seam_detection.core import fold_edge as fe  # noqa: E402
from box_seam_detection.core import pipeline as P  # noqa: E402
from box_seam_detection.core import seam_estimator as se  # noqa: E402
from box_seam_detection.core.config import default_config  # noqa: E402


def _xyz(long_mm, short_mm, z=0.6):
    """Rectangle in the camera frame: BL-BR = long, BL-FL = short."""
    a, b = long_mm / 1000.0, short_mm / 1000.0
    BL = np.array([0.0, 0.0, z])
    return np.array([BL, BL + [a, 0, 0], BL + [a, b, 0], BL + [0, b, 0]])


# ---- constants are today's, exactly ---------------------------------------

def test_profile_tables_equal_legacy_literals():
    assert bp.heights_m() == {"big": 0.13425, "small": 0.0863}
    assert fe.BOX_HEIGHT_M == {"big": 0.13425, "small": 0.0863}
    assert fe.BIG_LONG_SIDE_M == 0.150 == se.BIG_LONG_SIDE_M
    assert se.VIEW_SHIFT_MM == {"big": 0.5, "small": 3.2}
    assert bp.known_dims_mm_table() == {"small": [110.0, 98.0],
                                        "big": [193.0, 105.0]}
    cfg = default_config()
    assert cfg["reliability"]["r3_known_dims_mm"] == {
        "small": [110.0, 98.0], "big": [193.0, 105.0]}
    assert cfg["corner_fold"]["box_height_m"] == {"big": 0.13425,
                                                  "small": 0.0863}
    vs = cfg["seam_estimator"]["view_shift"]
    assert vs["shift_mm"] == {"big": 0.5, "small": 3.2}
    assert vs["big_long_side_m"] == 0.150
    ga = cfg["seam_estimator"]["geometry_axis"]
    assert ga["small_long_mm"] == [104.0, 116.0]
    assert ga["small_short_mm"] == [90.0, 104.0]
    assert tuple(se.GEOM_AXIS_DEFAULTS["small_long_mm"]) == (104.0, 116.0)


def test_classify_legacy_is_the_150mm_rule():
    assert bp.classify_legacy(0.150).name == "small"
    assert bp.classify_legacy(0.1501).name == "big"
    assert bp.classify_legacy(0.05).name == "small"
    assert bp.classify_legacy(0.39).name == "big"      # the legacy mis-class


# ---- classify / resolve ------------------------------------------------------

def test_classify_known_unordered_and_generic():
    assert bp.classify(110, 98)[0].name == "small"
    assert bp.classify(98, 110)[0].name == "small"
    assert bp.classify(193, 105, 134.25)[0].name == "big"
    p, info = bp.classify(390, 320, 480)
    assert p.is_generic and info["reason"] == "no_profile_within_tolerance"
    # right top, wrong height -> not that profile
    assert bp.classify(110, 98, 150)[0].is_generic


def test_resolve_known_boxes_keep_legacy_class():
    for lo, sh, h, name in ((0.1117, 0.0992, 86.2, "small"),
                            (0.1927, 0.1048, 136.4, "big"),
                            (0.1130, 0.0992, None, "small")):
        prof, info = bp.resolve(lo, sh, h)
        assert prof.name == name and not info["profile_mismatch"], info


def test_resolve_near_miss_keeps_legacy_with_flag():
    # badly measured small box: top 114 x 79 (a corner error), still small
    prof, info = bp.resolve(0.1141, 0.0792, 87.3)
    assert prof.name == "small" and info["profile_mismatch"]
    assert info["reason"] == "near_miss_keep_legacy"
    # height 12 mm off the small profile: flagged, still small
    prof, info = bp.resolve(0.111, 0.098, 98.0)
    assert prof.name == "small" and info["profile_mismatch"]


def test_resolve_known_height_with_badly_wrong_top_keeps_legacy():
    # real capture 20261002_074327: accepted top 101 x 33.8 mm, height 85.9
    prof, info = bp.resolve(0.101, 0.0338, 85.9)
    assert prof.name == "small" and info["profile_mismatch"]
    assert info["reason"] == "height_matches_dims_off_keep_legacy"
    # same dims without a height are not rescued
    assert bp.resolve(0.101, 0.0338, None)[0].is_generic


def test_resolve_truly_generic():
    prof, info = bp.resolve(0.39, 0.32, 480.0)
    assert prof.is_generic and info["legacy_class"] == "big"
    assert not prof.geometry_axis_validated and not prof.depth_edge_validated
    # same top as the small box but 300 mm tall is not the small box
    prof, _ = bp.resolve(0.110, 0.098, 300.0)
    assert prof.is_generic
    assert bp.is_generic_height(480.0) and not bp.is_generic_height(86.5)
    assert not bp.is_generic_height(None) and not bp.is_generic_height(136.4)


def test_resolve_box_profile_record_and_flags():
    h = {"measured_height_mm": 98.0, "height_unavailable": None}
    prof, rec = P.resolve_box_profile(_xyz(111.0, 98.0), h)
    assert prof.name == "small" and "height_mismatch" in rec["flags"]
    assert rec["depth_edge_validated"] and rec["geometry_axis_validated"]
    prof, rec = P.resolve_box_profile(_xyz(390, 320), {"measured_height_mm": 480.0})
    assert prof.is_generic and rec["name"] == "generic"
    assert rec["legacy_class"] == "big" and not rec["depth_edge_validated"]
    assert P.resolve_box_profile(None, None) == (None, None)


# ---- seam estimator: R10 and view shift per profile -------------------------

def test_r10_only_for_validated_profile():
    xyz = _xyz(94.6, 108.7)
    axis_legacy, _ = se.geometry_class_axis(xyz)
    assert axis_legacy is not None
    assert se.geometry_class_axis(xyz, None, "small")[0] == axis_legacy
    assert se.geometry_class_axis(xyz, None, None)[0] == axis_legacy
    for name in ("generic", "big"):
        ax, info = se.geometry_class_axis(xyz, None, name)
        assert ax is None and info["reason"] == "profile_not_validated"
        assert info["vetoed_axis"] == axis_legacy


def test_generic_view_shift_zero_known_unchanged():
    xyz = np.array([[-0.05, -0.05, 0.5], [0.05, -0.05, 0.5],
                    [0.05, 0.05, 0.5], [-0.05, 0.05, 0.5]])
    est = default_config()["seam_estimator"]
    base = se.expected_view_shift(xyz, "lr", est)
    assert base["expected_shift_mm"] == 3.2        # small class
    assert se.expected_view_shift(xyz, "lr", est, "small") == base
    g = se.expected_view_shift(xyz, "lr", est, "generic")
    assert g["expected_shift_mm"] == 0.0 and g["reason"] == "generic_profile"


# ---- corner fold per profile -------------------------------------------------

def test_fold_generic_without_height_is_skipped():
    q = np.array([[100, 100], [300, 100], [300, 250], [100, 250]], float)
    bgr = np.zeros((400, 400, 3), np.uint8)
    gen = bp.PROFILES["generic"]
    out, xyz, info = P.apply_corner_fold(
        bgr, q, {}, None, _xyz(390, 320), default_config()["corner_fold"],
        profile=gen, measured_height_m=None)
    assert info["reason"] == "no_height_generic" and info["box_class"] == "generic"
    assert np.array_equal(out, q)


def test_fold_known_profile_path_unchanged():
    q = np.array([[100, 100], [300, 100], [300, 250], [100, 250]], float)
    bgr = np.zeros((400, 400, 3), np.uint8)
    _, _, a = P.apply_corner_fold(bgr, q, {}, None, _xyz(111, 98),
                                  default_config()["corner_fold"])
    _, _, b = P.apply_corner_fold(bgr, q, {}, None, _xyz(111, 98),
                                  default_config()["corner_fold"],
                                  profile=bp.PROFILES["small"])
    assert a == b and a["box_class"] == "small" and a["height_m"] == 0.0863
