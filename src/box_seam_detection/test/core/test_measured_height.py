# Provenance: ported from rgbd_captures/tests/test_measured_height.py (HEAD 6df5994); imports made package-relative.
"""Measured box height (multi-box plan M2): top-face centroid to table plane,
on analytic table + box scenes (tests/synth/cloud_gen.py)."""
import functools
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest


from synth import cloud_gen as cg  # noqa: E402
from box_seam_detection.core.autolabel import depth_top as _depth  # noqa: E402
from box_seam_detection.core import pipeline as P  # noqa: E402
from box_seam_detection.core.config import default_config  # noqa: E402


@functools.lru_cache(maxsize=None)
def _scene(h, width, length, cam_dist, cy):
    sc = cg.Scene(h=h, width=width, length=length, yaw_deg=8.0,
                  cam_dist=cam_dist, center_px=(960.0, cy))
    g = sc.cast(step=1, noise_sigma_m=0.0004, seed=1)
    return sc, g["xyz"][g["valid"]].astype(np.float64), g


def _ctx(xyz, cam):
    return {"xyz": xyz, "cam": cam, "dcfg": default_config()["depth_top"]}


CASES = [(0.0863, 0.11, 0.098, 0.454, 720.0),
         (0.2, 0.2, 0.15, 0.8, 650.0),
         (0.48, 0.39, 0.32, 1.2, 600.0)]


@pytest.mark.parametrize("case", CASES)
def test_measured_height_matches_truth(case):
    sc, xyz, _ = _scene(*case)
    tr = sc.truth()
    ctx = _ctx(xyz, sc.cam)
    res = P.measure_box_height(ctx, NS(normal=tr["top_n"], d=tr["top_d"]),
                               tr["top_corners_xyz"], 0.5, 9000)
    assert res["height_unavailable"] is None
    assert abs(res["measured_height_mm"] - 1000.0 * case[0]) < 1.5, res
    assert res["table_inliers"] > 20000 and res["table_rms_mm"] > 0
    assert 0.0 < res["height_uncertainty_mm"] < 2.0
    assert res["top_inliers"] == 9000


def test_early_height_from_depth_estimate_tall_box():
    sc, xyz, _ = _scene(*CASES[2])
    est = _depth.estimate_top(xyz, sc.cam, (1080, 1920),
                              default_config()["depth_top"])
    assert est["ok"], est["reason"]
    assert abs(P.early_height_mm(est) - 480.0) < 3.0
    assert P.early_height_mm({"ok": False}) is None
    assert P.early_height_mm(None) is None


def test_height_unavailable_reasons():
    sc, xyz, g = _scene(*CASES[0])
    tr = sc.truth()
    plane = NS(normal=tr["top_n"], d=tr["top_d"])
    corners = tr["top_corners_xyz"]
    assert P.measure_box_height(None, plane, corners)["height_unavailable"] == "no_depth"
    # too few points for a table plane
    ctx = _ctx(xyz[:500], sc.cam)
    assert P.measure_box_height(ctx, plane, corners)["height_unavailable"] == "no_table_plane"
    ctx = _ctx(xyz, sc.cam)
    assert P.measure_box_height(ctx, None, corners)["height_unavailable"] == "no_top_plane"
    assert P.measure_box_height(ctx, plane, None)["height_unavailable"] == "no_top_plane"
    # a tilted "top" plane is not table-parallel
    tilt = np.array([0.0, 0.5, -0.87])
    r = P.measure_box_height(ctx, NS(normal=tilt / np.linalg.norm(tilt), d=0.37),
                             corners)
    assert r["height_unavailable"].startswith("top_not_table_parallel"), r
    assert r["measured_height_mm"] is None


def test_table_is_the_box_top_gives_unavailable_not_a_guess():
    """Box filling the view: the dominant plane IS the top -> height ~ 0."""
    sc, xyz, g = _scene(*CASES[0])
    top_only = g["xyz"][g["label"] == cg.LABEL_TOP].astype(np.float64)
    ctx = _ctx(top_only, sc.cam)
    ctx["dcfg"] = dict(ctx["dcfg"], table_min_inliers=1000)
    tr = sc.truth()
    r = P.measure_box_height(ctx, NS(normal=tr["top_n"], d=tr["top_d"]),
                             tr["top_corners_xyz"])
    assert r["measured_height_mm"] is None
    assert r["height_unavailable"].startswith("height_out_of_range"), r


def test_known_profile_height_flags():
    corners = np.array([[0, 0, .6], [.111, 0, .6], [.111, .098, .6], [0, .098, .6]])
    _, rec = P.resolve_box_profile(corners, {"measured_height_mm": 86.5})
    assert rec["flags"] == [] and rec["name"] == "small"
    _, rec = P.resolve_box_profile(corners, {"measured_height_mm": 93.0})
    assert rec["name"] == "small" and "height_mismatch" in rec["flags"]
    # unavailable height: no flag, profile height still the legacy one
    _, rec = P.resolve_box_profile(corners, {"measured_height_mm": None,
                                             "height_unavailable": "no_table_plane"})
    assert rec["name"] == "small" and rec["flags"] == []
    assert rec["height_unavailable"] == "no_table_plane"
