# Provenance: ported from rgbd_captures/tests/test_r10_r12.py (HEAD 6df5994); imports made package-relative.
"""R10 (seam axis from small-box geometry), R11 (r3 dims tolerance; percentile extents reverted: no benefit on replay),
R12 (R5 not circular for fitted rectangles). Synthetic data only."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest


from box_seam_detection.core import pipeline as _p
from box_seam_detection.core import seam_estimator as _se
from box_seam_detection.core.config import default_config

import test_r2_r3_r5 as _t  # noqa: E402


# ---- R10 ------------------------------------------------------------------

def _xyz(bl_fl_mm, bl_br_mm, skew_mm=0.0):
    """Rectangle in the camera frame (metres). BL-FL side = bl_fl_mm,
    BL-BR side = bl_br_mm."""
    a, b = bl_br_mm / 1000.0, bl_fl_mm / 1000.0
    z = 0.6
    BL = np.array([0.0, 0.0, z])
    BR = BL + np.array([a, 0.0, 0.0])
    FL = BL + np.array([0.0, b, 0.0])
    FR = BR + np.array([skew_mm / 1000.0, b, 0.0])
    return np.array([BL, BR, FR, FL])


def test_geometry_axis_small_box_long_sides():
    # BL-FL / BR-FR are the long sides -> seam joins them -> "lr"
    ax, info = _se.geometry_class_axis(_xyz(108.7, 94.6))
    assert ax == "lr", info
    ax, info = _se.geometry_class_axis(_xyz(94.6, 108.7))
    assert ax == "bf", info
    for lf, bb in ((108.6, 99.7), (109.0, 100.1), (108.8, 99.5)):
        assert _se.geometry_class_axis(_xyz(lf, bb))[0] == "lr"


@pytest.mark.parametrize("xyz,reason", [
    (_xyz(108.0, 108.0), "near_square"),
    (_xyz(108.0, 104.0), "not_small_class"),     # short 104 ok, diff<5
    (_xyz(130.0, 100.0), "not_small_class"),     # big box
    (_xyz(108.0, 98.0, skew_mm=5.0), "not_rectangular"),
    (None, "no_corners_xyz"),
    (np.full((4, 3), np.nan), "no_corners_xyz"),
])
def test_geometry_axis_rejections(xyz, reason):
    ax, info = _se.geometry_class_axis(xyz)
    assert ax is None
    assert info["reason"] in (reason, "near_square", "not_small_class")


def test_geometry_axis_ratio_and_diff_gates():
    # long 108, short 103.5: ratio 1.043 < 1.05 and diff 4.5 < 5
    assert _se.geometry_class_axis(_xyz(108.0, 103.5))[0] is None
    # disabled by config
    assert _se.geometry_class_axis(
        _xyz(108.7, 94.6), {"geometry_axis": {"enabled": False}})[0] is None


def _flat_quad_image():
    # uniform face: no tape/gap evidence -> axis ambiguous
    img = np.full((400, 600, 3), (60, 110, 170), np.uint8)
    quad = np.array([[150, 100], [450, 100], [450, 300], [150, 300]], float)
    return img, quad


def test_estimate_seam_geometry_axis_published_as_estimated():
    img, quad = _flat_quad_image()
    base = _se.estimate_seam(img, quad)
    assert base["status"] == "ambiguous" and base["axis"] is None
    xyz = _xyz(108.7, 94.6)  # BL-FL long -> lr
    res = _se.estimate_seam(img, quad, corners_xyz=xyz)
    assert res["status"] == "estimated" and res["axis"] == "lr"
    assert res["axis_source"] == "geometry_class"
    assert res["seam_axis_from_geometry"] is True
    assert res["source"] == "prior"
    assert res["prior_endpoints_px"]["seam_left"] == [150.0, 200.0]
    assert res["prior_endpoints_px"]["seam_right"] == [450.0, 200.0]
    res2 = _se.estimate_seam(img, quad, corners_xyz=_xyz(94.6, 108.7))
    assert res2["axis"] == "bf" and res2["status"] == "estimated"
    # near-square / big box / no xyz keep the old ambiguous behaviour
    for x in (_xyz(108.0, 108.0), _xyz(150.0, 120.0), None):
        r = _se.estimate_seam(img, quad, corners_xyz=x)
        assert r["status"] == "ambiguous" and r["axis"] is None
        assert r.get("axis_source") is None


def test_geometry_axis_never_ok(monkeypatch):
    img, quad = _flat_quad_image()
    real = _se.refine_on_gap

    def fake_gap(*a, **k):
        # a perfect, fully covered gap on the prior
        o, s, i = real(*a, **k)
        return o, s, i
    # force the core to report ok by wrapping its result
    core = _se._estimate_seam_core

    def ok_core(*a, **k):
        r = core(*a, **k)
        r.update({"status": "ok", "source": "prior_confirmed_by_gap",
                  "refined_endpoints_px": r["prior_endpoints_px"]})
        # refine/darkline modes publish the observed line on ok
        r["estimate"]["estimate_endpoints_px"] = {"seam_left": [1.0, 2.0],
                                                  "seam_right": [3.0, 4.0]}
        return r
    monkeypatch.setattr(_se, "_estimate_seam_core", ok_core)
    res = _se.estimate_seam(img, quad, corners_xyz=_xyz(108.7, 94.6))
    assert res["status"] == "estimated"
    assert res["refined_endpoints_px"] is None
    assert res["estimate"]["estimate_endpoints_px"] == res["prior_endpoints_px"]


def test_evidence_axis_source_normal_path():
    # Clear axis from real tape evidence keeps axis_source == "evidence".
    img = np.full((400, 600, 3), (60, 110, 170), np.uint8)
    img[195:205, 150:450] = (230, 230, 230)  # horizontal tape on the seam
    quad = np.array([[150, 100], [450, 100], [450, 300], [150, 300]], float)
    res = _se.estimate_seam(img, quad, corners_xyz=_xyz(108.7, 94.6))
    if res["axis"] is not None and res.get("axis_source") != "geometry_class":
        assert res["axis_source"] == "evidence"


def test_compute_seam_stage_flags_geometry_axis():
    img, quad = _flat_quad_image()
    cfg = default_config()
    sd, st = _p.compute_seam_stage(img, quad, cfg,
                                   corners_xyz=_xyz(108.7, 94.6))
    assert sd["status"] == "estimated" and sd["axis"] == "lr"
    assert "seam_axis_from_geometry" in st["reasons"]
    sd2, st2 = _p.compute_seam_stage(img, quad, cfg, corners_xyz=None)
    assert sd2["status"] == "ambiguous"
    assert "seam_axis_from_geometry" not in st2["reasons"]


# ---- R11 (only the informational r3 dims tolerance remains) -------------

def test_r3_dims_tol_is_8mm():
    assert default_config()["reliability"]["r3_dims_tol_mm"] == 8.0
    assert default_config()["reliability"]["r_keep_original_rect_mm"] == 10.0


# ---- R12 ------------------------------------------------------------------

def test_r5_skipped_when_r3_rectangle_fitted(monkeypatch, tmp_path):
    s = _t.make_scene(15.0)
    # control: a plain (non-rewritten) quad with no depth IoU is cleared
    rec0, _ = _t._no_check_run(monkeypatch, tmp_path, s["top_px"], s)
    assert rec0["checks"]["needs_review"] is False
    assert "r5" not in rec0["provenance"]
    # k4 quad -> R3 rectangle wins -> the hull IoU would be circular
    _t._patch(monkeypatch, s, cross_iou=None)
    rec, _ = _t._run(tmp_path, s["cand"], k4=True)
    assert "r3_rect" in rec["provenance"]["sam3"]
    chk = rec["checks"]
    assert chk["needs_review"] is True
    assert "no_independent_depth_check" in chk["reasons"]
    itc = chk["independent_top_check"]
    assert itc["skipped_rectangle_fitted"] is True
    assert itc["rectangle_source"] == "r3_rect"
    assert not itc.get("cleared_needs_review")
    assert rec["provenance"]["r5"] == "skipped_rectangle_fitted"


def test_r5_skipped_when_r2_trim_used(monkeypatch, tmp_path):
    s = _t.make_scene(30.0)
    _t._patch(monkeypatch, s, cross_iou=None)
    rec, _ = _t._run(tmp_path, s["cand"])
    if "r2_trim" not in rec["provenance"]["sam3"]:
        pytest.skip("R2 trim not selected in this synthetic scene")
    assert rec["checks"]["needs_review"] is True
    assert rec["checks"]["independent_top_check"]["rectangle_source"] \
        == "r2_trim"
    assert rec["provenance"]["r5"] == "skipped_rectangle_fitted"
