# Provenance: ported from rgbd_captures/tests/test_size_limits.py (HEAD 6df5994); imports made package-relative.
"""Size-agnostic limits (multi-box plan M3): depth_top offset gate for tall
boxes, table-below guard, plausible-height range, R2 height invariance,
generic-profile behaviour."""
import copy
import functools
import sys
from pathlib import Path

import numpy as np
import pytest


from synth import cloud_gen as cg  # noqa: E402
from box_seam_detection.core.autolabel import depth_top as D  # noqa: E402
from box_seam_detection.core import pipeline as P  # noqa: E402
from box_seam_detection.core.config import default_config  # noqa: E402


@functools.lru_cache(maxsize=None)
def _cloud(h, width, length, cam_dist, cy):
    sc = cg.Scene(h=h, width=width, length=length, yaw_deg=8.0,
                  cam_dist=cam_dist, center_px=(960.0, cy))
    g = sc.cast(step=1, noise_sigma_m=0.0004, seed=1)
    return sc, g["xyz"][g["valid"]].astype(np.float64)


def _dcfg(offset_max=None, legacy=False):
    c = copy.deepcopy(default_config()["depth_top"])
    if offset_max is not None:
        c["offset_max"] = offset_max
    if legacy:
        c.pop("offset_above_check_from", None)
    return c


def test_config_offset_max_covers_tall_boxes():
    assert default_config()["depth_top"]["offset_max"] >= 0.6
    assert default_config()["depth_top"]["offset_above_check_from"] == 0.30


@pytest.mark.parametrize("h,cam_dist", [(0.0863, 0.454), (0.3, 0.9), (0.48, 1.2)])
def test_estimate_top_ok_for_box_heights(h, cam_dist):
    sc, xyz = _cloud(h, 0.39 if h > 0.2 else 0.11, 0.32 if h > 0.2 else 0.098,
                     cam_dist, 600.0 if h > 0.2 else 720.0)
    r = D.estimate_top(xyz, sc.cam, (1080, 1920), default_config()["depth_top"])
    assert r["ok"], r["reason"]
    assert abs(r["depth"]["top_offset_mm"] - 1000 * h) < 4.0


def test_old_limit_rejected_tall_box_new_accepts():
    sc, xyz = _cloud(0.48, 0.39, 0.32, 1.2, 600.0)
    old = D.estimate_top(xyz, sc.cam, (1080, 1920), _dcfg(0.30, legacy=True))
    assert not old["ok"] and old["reason"] == "depth_no_top"
    assert any(n.get("excluded") == "offset_out_of_range"
               for n in old["depth"]["top_candidates"])
    new = D.estimate_top(xyz, sc.cam, (1080, 1920), _dcfg())
    assert new["ok"]


def test_small_box_identical_with_old_and_new_limit():
    sc, xyz = _cloud(0.0863, 0.11, 0.098, 0.454, 720.0)
    old = D.estimate_top(xyz, sc.cam, (1080, 1920), _dcfg(0.30, legacy=True))
    new = D.estimate_top(xyz, sc.cam, (1080, 1920), _dcfg())
    assert old["ok"] and new["ok"]
    assert np.array_equal(old["quad"], new["quad"])
    assert old["depth"]["top_offset_mm"] == new["depth"]["top_offset_mm"]


def _two_plane_cloud(z_big, z_small, rng):
    """Dominant plane at z_big (camera frame, facing the camera) plus a
    smaller parallel plane at z_small inside its footprint."""
    n_big, n_small = 60000, 12000
    big = np.stack([rng.uniform(-0.5, 0.5, n_big), rng.uniform(-0.4, 0.4, n_big),
                    np.full(n_big, z_big)], 1)
    small = np.stack([rng.uniform(-0.15, 0.15, n_small),
                      rng.uniform(-0.15, 0.15, n_small),
                      np.full(n_small, z_small)], 1)
    return np.vstack([big, small]) + rng.normal(0, 0.0005, (n_big + n_small, 3))


def _top_notes(xyz):
    dcfg = _dcfg()
    dcfg.update(max_planes=2, top_min_inliers=5000, table_min_inliers=20000)
    cropped = xyz[D.workspace_mask(xyz, dcfg)]
    planes = D.sequential_planes(cropped, dcfg)
    _, top, _, diag, reason = D.identify_table_top(planes, cropped, dcfg)
    return diag.get("top_candidates") or [], reason


def test_plane_below_table_is_not_a_top():
    """A parallel plane 0.45 m BEHIND the dominant plane (a floor) is never a
    top, however plausible its offset; one 0.45 m in FRONT is a candidate."""
    rng = np.random.default_rng(0)
    notes, reason = _top_notes(_two_plane_cloud(1.0, 1.45, rng))
    assert reason == "depth_no_top"
    assert [n.get("excluded") for n in notes] == ["not_above_table"]
    notes, reason = _top_notes(_two_plane_cloud(1.0, 0.55, rng))
    assert [n.get("excluded") for n in notes] == [None], (notes, reason)


def test_legacy_range_unchanged_below_old_limit():
    """Planes within the old 0.30 m limit never see the new guard (identical
    behaviour), including a plane behind the table."""
    rng = np.random.default_rng(1)
    notes, reason = _top_notes(_two_plane_cloud(1.0, 1.2, rng))
    assert [n.get("excluded") for n in notes] == [None]


def test_plausible_height_range_covers_tall_box():
    assert P.BOX_H_MAX_MM >= 600.0
    assert P._choice_ok(0.5, 480.0, P.BOX_H_MAX_MM)
    assert not P._choice_ok(0.5, 800.0, P.BOX_H_MAX_MM)
    assert not P._choice_ok(0.5, 10.0, P.BOX_H_MAX_MM)
    # known boxes keep the 400 mm selection limit (hand / arm protection)
    assert not P._choice_ok(0.5, 480.0) and P._choice_ok(0.5, 130.0)


def test_r2_geometry_is_independent_of_reference_height():
    """box_view_geometry only scales with ref height, so generic boxes may use
    the measured height with no change for known ones."""
    sc, xyz = _cloud(0.48, 0.39, 0.32, 1.2, 600.0)
    ctx = {"xyz": xyz, "cam": sc.cam, "dcfg": default_config()["depth_top"]}
    manifest = {"k": [sc.cam["fx"], 0, sc.cam["cx"], 0, sc.cam["fy"],
                      sc.cam["cy"], 0, 0, 1.0]}
    q = sc.truth()["top_corners_px"]
    g1 = P.box_view_geometry(q, manifest, ctx, 0.10)
    g2 = P.box_view_geometry(q, manifest, ctx, 0.48)
    assert g1 is not None and g2 is not None
    assert abs(g1["yaw45"] - g2["yaw45"]) < 0.05
    for k in g1["wall_cos"]:
        assert abs(g1["wall_cos"][k] - g2["wall_cos"][k]) < 0.01


def test_generic_tall_box_profile_end_to_end_components():
    """measured height -> generic -> no R10 / no depth-edge / fold height."""
    from types import SimpleNamespace as NS
    sc, xyz = _cloud(0.48, 0.39, 0.32, 1.2, 600.0)
    tr = sc.truth()
    ctx = {"xyz": xyz, "cam": sc.cam, "dcfg": default_config()["depth_top"]}
    h = P.measure_box_height(ctx, NS(normal=tr["top_n"], d=tr["top_d"]),
                             tr["top_corners_xyz"], 0.4, 20000)
    prof, rec = P.resolve_box_profile(tr["top_corners_xyz"], h)
    assert prof.is_generic and rec["name"] == "generic"
    assert rec["depth_edge_validated"] is False
    assert rec["geometry_axis_validated"] is False
    assert abs(rec["measured_height_mm"] - 480.0) < 1.5
    assert rec["top_dims_mm"][0] == pytest.approx(390.0, abs=1.0)


@pytest.mark.parametrize("case,h", [((0.0863, 0.11, 0.098, 0.454, 720.0), 86.3),
                                    ((0.48, 0.39, 0.32, 1.2, 600.0), 480.0)])
def test_quad_height_from_central_cloud_points(case, h):
    """Per-candidate box height (used to decide generic for the pre-refine
    rewrite) is the robust (p90) height of the quad's central points, also when the
    outline spills onto the side walls."""
    sc, xyz = _cloud(*case)
    ctx = {"xyz": xyz, "cam": sc.cam, "dcfg": default_config()["depth_top"]}
    q = sc.truth()["top_corners_px"].copy()
    assert abs(P.quad_height_mm(q, ctx, 1920, 1080) - h) < 3.0
    spill = q.copy()
    spill[np.argsort(q[:, 1])[2:], 1] += 25.0     # near corners pulled down
    assert abs(P.quad_height_mm(spill, ctx, 1920, 1080) - h) < 8.0
    assert P.quad_height_mm(q, None, 1920, 1080) is None
