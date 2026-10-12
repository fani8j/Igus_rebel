"""Scene configs (table_cell_v1 / dual_arm_v1), hidden_stand top identification,
noise-adaptive guided-plane gate, large_392 profile and its cut-permission rules."""
import copy
from pathlib import Path

import numpy as np
import pytest

from synth import cloud_gen as cg  # noqa: E402
from box_seam_detection.core import box_profiles as bp  # noqa: E402
from box_seam_detection.core import pipeline as P  # noqa: E402
from box_seam_detection.core import scene as sc_mod  # noqa: E402
from box_seam_detection.core.autolabel import depth_top as D  # noqa: E402
from box_seam_detection.core.config import V2_DEFAULTS, default_config  # noqa: E402

SCENES = sc_mod.scenes_dir()  # source tree or the installed ament share


# ---- loader -----------------------------------------------------------------

def test_default_scene_is_todays_config_exactly():
    base = default_config()
    cfg = sc_mod.builtin_default().apply(default_config())
    cfg.pop("scene")
    assert cfg == base                                     # no value changes


def test_table_cell_yaml_mirrors_builtin():
    import yaml
    data = yaml.safe_load((SCENES / "table_cell_v1.yaml").read_text())
    assert data == sc_mod.TABLE_CELL_V1
    scene = sc_mod.load_scene(SCENES / "table_cell_v1.yaml")
    assert scene.sha256 and scene.apply(default_config())["depth_top"] == V2_DEFAULTS["depth_top"]


def test_dual_arm_scene_values():
    s = sc_mod.load_scene("dual_arm_v1")
    cfg = s.apply(default_config())
    d = cfg["depth_top"]
    assert d["support"] == "hidden_stand" and d["ws_z_max"] >= 4.5
    assert (d["offset_min"], d["offset_max"]) == (0.30, 1.50)
    assert cfg["v2"]["noise_adaptive"]["enabled"] is True
    assert s.seam_prior_deg is None
    assert s.driver_block("reference") is None
    assert s.driver_block("fallback") is not None and s.driver_verified is False
    assert s.calib_file.endswith("dualcal_20261009_left_single.calib")
    assert cfg["scene"]["name"] == "dual_arm_v1"


def test_unknown_keys_and_bad_files_fail_closed(tmp_path):
    bad = tmp_path / "s.yaml"
    bad.write_text("scene: x\ndepth_top:\n  offest_min: 0.1\n")
    with pytest.raises(sc_mod.SceneError):
        sc_mod.load_scene(bad)
    bad.write_text("scene: x\nlive:\n  nope: 1\n")
    with pytest.raises(sc_mod.SceneError):
        sc_mod.load_scene(bad)
    with pytest.raises(sc_mod.SceneError):
        sc_mod.load_scene("does_not_exist_scene")


def test_seam_prior_and_driver_reference_are_scene_config():
    from box_seam_detection.live import base_link as bl
    ref, fb = bl.scene_driver_blocks(None)
    assert ref is bl.RECORDED_DRIVER_EXTRINSIC and fb is bl.RECORDED_DRIVER_EXTRINSIC
    dual = sc_mod.load_scene("dual_arm_v1")
    block = {**bl.RECORDED_DRIVER_EXTRINSIC, "translation": (0.5, 0.0, 0.0), "source": "t"}
    # no reference: a differing manifest block is NOT refused (warning only)
    info = bl.driver_extrinsic_check(block, "refuse", reference=bl.scene_driver_blocks(dual)[0])
    assert info["reference"] is None and info["used"] == "manifest"
    # table cell: still refused
    with pytest.raises(ValueError):
        bl.driver_extrinsic_check(block, "refuse")


# ---- hidden_stand top identification -------------------------------------------

def _plane_pts(n, d, u, v, step):
    e1 = np.cross(n, [1.0, 0.0, 0.0]); e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    uu, vv = np.meshgrid(np.arange(u[0], u[1], step), np.arange(v[0], v[1], step))
    p0 = np.array([0.0, 0.0, -d / n[2]])        # the plane point on the optical axis
    return p0 + uu.reshape(-1, 1) * e1 + vv.reshape(-1, 1) * e2


def _hidden_cloud(desk=False, seed=0):
    rng = np.random.default_rng(seed)
    n = np.array([0.0, -0.7, -0.7]); n /= np.linalg.norm(n)
    floor = _plane_pts(n, 1.7, (-2.0, 2.0), (-1.0, 1.5), 0.006)
    top = _plane_pts(n, 1.1, (-0.2, 0.2), (-0.16, 0.16), 0.002)
    # floor below the box is hidden by the box + stand
    e1 = np.cross(n, [1.0, 0.0, 0.0]); e1 /= np.linalg.norm(e1); e2 = np.cross(n, e1)
    p0 = np.array([0.0, 0.0, 1.7 / 0.7 * 0 - 1.7 / n[2]])
    fu, fv = (floor - p0) @ e1, (floor - p0) @ e2
    floor = floor[~((np.abs(fu) < 0.25) & (np.abs(fv) < 0.2))]
    parts = [floor, top]
    if desk:
        parts.append(_plane_pts(n, 0.85, (0.45, 0.85), (-0.2, 0.2), 0.002))
    xyz = np.concatenate(parts)
    return n, xyz + rng.normal(0, 0.002, xyz.shape)


def _dual_cfg():
    return sc_mod.load_scene("dual_arm_v1").apply(default_config())


def test_hidden_stand_finds_top_and_never_reports_floor_as_height():
    cam = cg.load_camera()
    n, xyz = _hidden_cloud()
    # visible_table (today's config): the floor is outside the crop -> no top
    old = D.estimate_top(xyz, cam, (1080, 1920), default_config()["depth_top"])
    assert not old["ok"]
    cfg = _dual_cfg()
    est = D.estimate_top(xyz, cam, (1080, 1920), cfg["depth_top"])
    assert est["ok"], est["reason"]
    assert est["depth"]["support_hidden"] is True
    assert abs(est["depth"]["top_offset_mm"] - 600.0) < 8.0       # floor distance, stand + box
    assert est["depth"]["noise_mm"] is not None
    ctx = {"xyz": xyz, "cam": cam, "dcfg": cfg["depth_top"]}
    h = P.measure_box_height(ctx, type("Pl", (), {"normal": n, "d": 1.1})(),
                             np.array([[0, 0, 1.0]] * 4), 1.0, 5000)
    assert h["measured_height_mm"] is None and h["height_unavailable"] == "support_hidden"
    prof, rec = P.resolve_box_profile(
        np.array([[-0.195, -0.16, 1.0], [0.195, -0.16, 1.0], [0.195, 0.16, 1.0], [-0.195, 0.16, 1.0]]), h)
    assert prof.name == "large_392" and rec["height_unavailable"] == "support_hidden"
    assert rec["profile_height_mm"] == 480.0 and "height_mismatch" not in rec["flags"]


def test_hidden_stand_requires_top_above_floor_and_in_range():
    cam = cg.load_camera()
    n, xyz = _hidden_cloud()
    cfg = _dual_cfg()
    cfg["depth_top"]["offset_max"] = 0.5                        # top is 0.6 m above the floor
    assert not D.estimate_top(xyz, cam, (1080, 1920), cfg["depth_top"])["ok"]


def test_hidden_stand_desk_is_ambiguous_and_roi_hint_resolves():
    cam = cg.load_camera()
    n, xyz = _hidden_cloud(desk=True)
    cfg = _dual_cfg()
    est = D.estimate_top(xyz, cam, (1080, 1920), cfg["depth_top"])
    assert not est["ok"] and est["reason"] == "depth_ambiguous_top"
    # ROI around the box only (no floor in it): the full-cloud floor is the support hint
    sel = np.abs(xyz @ n + 1.1) < 0.03          # the candidate's own ROI: just the box top
    roi = xyz[sel]
    hint = {"n": n, "d": 1.7, "n_inliers": 100000}
    no_hint = D.estimate_top(roi, cam, (1080, 1920), cfg["depth_top"])
    assert not no_hint["ok"]
    ok = D.estimate_top(roi, cam, (1080, 1920), cfg["depth_top"], support_hint=hint)
    assert ok["ok"] and abs(ok["depth"]["top_offset_mm"] - 600.0) < 8.0


# ---- noise-adaptive guided-plane gate -------------------------------------------

def _est_with_noise(noise):
    return {"ok": False, "depth": {"noise_mm": noise}}


def test_noise_gate_fixed_when_disabled_or_quiet():
    base = default_config()
    g = P.noise_adaptive_gate(base, _est_with_noise(5.0))
    assert g["enabled"] is False and g["band_mm"] == 3.0 and g["p95_limit_mm"] == 8.0
    dual = _dual_cfg()
    for noise in (None, 0.7, 1.07, 1.64, 1.8):                    # table-cell replay range
        g = P.noise_adaptive_gate(dual, _est_with_noise(noise))
        assert g["scaled"] is False and g["band_mm"] == 3.0 and g["p95_limit_mm"] == 8.0


def test_noise_gate_scales_with_scene_noise_and_is_bounded():
    dual = _dual_cfg()
    g = P.noise_adaptive_gate(dual, _est_with_noise(2.81))        # the new capture
    assert g["scaled"] and 6.5 < g["band_mm"] < 7.5 and 9.4 < g["p95_limit_mm"] < 10.5
    big = P.noise_adaptive_gate(dual, _est_with_noise(9.0), 5.0)
    assert big["band_mm"] == 12.0                                 # max_band_mm cap
    assert g["band_mm"] > 3.0 and g["p95_limit_mm"] > 8.0
    # bulge: only for large tops, bounded
    c = np.array([[0, 0, 1], [0.39, 0, 1], [0.39, 0.32, 1], [0, 0.32, 1.0]])
    assert 1.9 <= P._bulge_allowance_mm(dual, c) <= 2.0
    small = np.array([[0, 0, 1], [0.11, 0, 1], [0.11, 0.1, 1], [0, 0.1, 1.0]])
    assert P._bulge_allowance_mm(dual, small) == 0.0


# ---- large_392 profile ---------------------------------------------------------------

def test_large_profile_classification_and_constants():
    p = bp.PROFILES["large_392"]
    assert p.height_mm == 480.0 and p.outer_dims_mm == {"along_seam": 390.0, "across_seam": 320.0}
    assert p.seam_joins is None and not p.geometry_axis_validated and not p.depth_edge_validated
    assert p.view_shift_mm == 0.0 and p.dims_tol_mm == 15.0 and p.confirm_max_offset_mm == 4.0
    assert p.own_constants and not p.is_legacy and not p.is_generic
    prof, info = bp.resolve(0.3945, 0.3286, None)                 # no height (support hidden)
    assert prof.name == "large_392" and info["reason"] == "matched"
    assert bp.resolve(0.390, 0.320, 480.0)[0].name == "large_392"
    assert bp.resolve(0.390, 0.320, 300.0)[0].is_generic          # wrong measured height
    assert bp.resolve(0.1117, 0.0992, 86.2)[0].name == "small"    # old boxes unchanged
    assert bp.resolve(0.1927, 0.1048, 136.4)[0].name == "big"
    assert bp.is_generic_height(480.0)


def test_overlap_chooser_picks_the_candidate_under_the_sam3_quad():
    """Box top and a desk above the same floor: the SAM3 quad decides (no hint needed)."""
    import cv2
    cam = cg.load_camera()
    n, xyz = _hidden_cloud(desk=True)
    uv, _z, _ = D.project_points(xyz[np.abs(xyz @ n + 1.1) < 0.01], cam)
    quad = cv2.boxPoints(cv2.minAreaRect(np.asarray(uv, np.float32))).astype(float)
    cfg = _dual_cfg()
    assert D.estimate_top(xyz, cam, (1080, 1920), cfg["depth_top"])["reason"] == "depth_ambiguous_top"
    est = D.estimate_top(xyz, cam, (1080, 1920), cfg["depth_top"], prefer_quad=quad)
    assert est["ok"] and abs(est["depth"]["top_offset_mm"] - 600.0) < 8.0
    assert est["depth"]["top_chosen_by_overlap"] is not None
    # a quad over nothing clear (the floor) must not pick anything
    far = np.array([[100, 100], [160, 100], [160, 160], [100, 160]], float)
    assert not D.estimate_top(xyz, cam, (1080, 1920), cfg["depth_top"], prefer_quad=far)["ok"]


def test_range_model_gate_scales_with_box_range_only_when_noisy():
    dual = _dual_cfg()
    near = P.noise_adaptive_gate(dual, _est_with_noise(2.8), 1.17)
    far = P.noise_adaptive_gate(dual, _est_with_noise(2.8), 1.92)
    assert near["model"] == "range" and far["band_mm"] > near["band_mm"]
    assert 6.0 < near["band_mm"] < 7.5 and 9.5 < far["band_mm"] <= 12.0
    assert far["p95_limit_mm"] > 13.0 > near["p95_limit_mm"] > 8.0
    quiet = P.noise_adaptive_gate(dual, _est_with_noise(1.5), 1.92)
    assert quiet["scaled"] is False and quiet["band_mm"] == 3.0


# ---- the two real dual-arm captures (skipped without the data root) -----------------

_ROOT = Path("/media/farhan/My Passport/Igus_rebel/outputs/rgbd_captures")
_CAPS = {  # id: (SAM3 quad as refined: BL, BR, FR, FL pixels, range m)
    "frame_000001_20261011_225221_220788": ([[822.7, 698.0], [1143.0, 709.3], [1144.5, 976.9], [768.3, 954.6]], 1.17),
    "frame_000001_20261012_001211_445510": ([[777.3, 507.2], [853.3, 375.0], [1010.0, 408.9], [945.0, 548.4]], 1.92),
}


@pytest.mark.skipif(not _ROOT.is_dir(), reason="capture data root absent")
@pytest.mark.parametrize("cid", sorted(_CAPS))
def test_real_dual_arm_captures_pass_the_depth_gates(cid):
    import json
    from box_seam_detection.core.autolabel.pcd import read_pcd_xyz
    quad, z = _CAPS[cid]
    pcd, man = _ROOT / "pointcloud" / f"{cid}.pcd", _ROOT / "captures" / f"{cid}.json"
    if not (pcd.is_file() and man.is_file()):
        pytest.skip("capture files absent")
    manifest = json.loads(man.read_text())
    xyz = np.asarray(read_pcd_xyz(str(pcd))[0], dtype=np.float64)
    cam, _ = D.camera_from_capture_info(manifest["camera_info"])
    cfg = _dual_cfg()
    est = D.estimate_top(xyz, cam, (1080, 1920), cfg["depth_top"])
    ctx = {"xyz": xyz, "cam": cam, "dcfg": cfg["depth_top"]}
    q = np.array(quad, float)
    cross, _g = P.check_depth_cross(q, manifest, est, None, 1920, 1080, cfg, depth_ctx=ctx)
    assert not cross["wrong_face"], cross["reasons"]
    assert cross["iou"] is not None and cross["iou"] > 0.6
    assert abs(cross["plane_gate"]["z_top_m"] - z) < 0.05
    c = cross["corners_xyz"]
    sides = sorted(float(np.linalg.norm(c[(i + 1) % 4] - c[i])) * 1000 for i in range(4))
    assert 300 < sides[0] and sides[-1] < 410          # a 390 x 320 mm top (depth-plane tilt inflates it)
    assert P.measure_box_height(ctx, cross["plane"], c)["height_unavailable"] == "support_hidden"
