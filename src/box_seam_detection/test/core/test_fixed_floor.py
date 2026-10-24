"""Fixed floor (scene option ``live.fixed_floor``, core/fixed_floor.py): validation, camera-frame plane,
cloud check (fail closed), use by the hidden-stand top check and the pipeline's support plane, unchanged when
absent. The detector wiring (transform fails closed before SAM3) is in test/live/test_fixed_floor_gate.py."""
import copy

import numpy as np
import pytest

from synth import cloud_gen as cg  # noqa: E402
from box_seam_detection.core import fixed_floor as ff
from box_seam_detection.core import pipeline as P
from box_seam_detection.core import scene as sc
from box_seam_detection.core.autolabel import depth_top as D
from box_seam_detection.core.config import default_config

# the synthetic hidden-stand scene of test_scene.py: floor n.p + 1.7 = 0, box top 0.6 m above it
N_CAM = np.array([0.0, -0.7, -0.7]) / np.linalg.norm([0.0, -0.7, -0.7])
FLOOR_D, TOP_D = 1.7, 1.1


def _c2b():
    """base_link <- camera such that base +z (up) is N_CAM in the camera frame, camera 1.7 m above z=0."""
    r3 = N_CAM
    r1 = np.cross([0.0, 1.0, 0.0], r3)
    r1 /= np.linalg.norm(r1)
    r2 = np.cross(r3, r1)
    m = np.eye(4)
    m[:3, :3] = np.stack([r1, r2, r3])          # rows; R^T e_z = N_CAM
    m[:3, 3] = [0.3, -0.2, FLOOR_D]
    return m


GOOD = {"frame": "base_link", "normal": [0.0, 0.0, 1.0], "offset_m": 0.0, "tolerance_m": 0.02,
        "search_m": 0.07, "min_inliers": 5000, "min_inlier_frac": 0.5}


def _plane_pts(d, u, v, step):
    e1 = np.cross(N_CAM, [1.0, 0.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(N_CAM, e1)
    uu, vv = np.meshgrid(np.arange(u[0], u[1], step), np.arange(v[0], v[1], step))
    p0 = np.array([0.0, 0.0, -d / N_CAM[2]])
    return p0 + uu.reshape(-1, 1) * e1 + vv.reshape(-1, 1) * e2


def _cloud(seed=0, floor_noise=0.007, floor_shift=0.0, with_floor=True):
    """Thick noisy floor (7 mm) with a hole under the box, plus the box top (2 mm noise)."""
    rng = np.random.default_rng(seed)
    floor = _plane_pts(FLOOR_D + floor_shift, (-2.0, 2.0), (-1.0, 1.5), 0.006)
    e1 = np.cross(N_CAM, [1.0, 0.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(N_CAM, e1)
    p0 = np.array([0.0, 0.0, -FLOOR_D / N_CAM[2]])
    fu, fv = (floor - p0) @ e1, (floor - p0) @ e2
    floor = floor[~((np.abs(fu) < 0.25) & (np.abs(fv) < 0.2))]
    floor = floor + rng.normal(0, floor_noise, floor.shape)
    top = _plane_pts(TOP_D, (-0.2, 0.2), (-0.16, 0.16), 0.002)
    top = top + rng.normal(0, 0.002, top.shape)
    return np.concatenate([floor, top] if with_floor else [top])


def _scene_cfg(floor=True):
    s = sc.load_scene("dual_arm_v1")
    cfg = s.apply(default_config())
    if floor:
        cfg["depth_top"]["fixed_floor"] = ff.to_camera(ff.validate_fixed_floor(GOOD), _c2b())
    else:
        cfg["depth_top"].pop("fixed_floor", None)
    return cfg


# ---- validation ------------------------------------------------------------------------------

def test_valid_mapping_is_normalised_with_defaults():
    out = ff.validate_fixed_floor({"frame": "base_link", "normal": [0, 0, 2], "offset_m": -0.83})
    assert out["normal"] == [0.0, 0.0, 1.0] and out["offset_m"] == -0.83
    assert out["tolerance_m"] == 0.02 and out["search_m"] == 0.07
    assert out["min_inliers"] == 20000 and out["min_inlier_frac"] == 0.5
    assert out["max_median_offset_m"] == pytest.approx(0.01)
    assert ff.validate_fixed_floor(None) is None


@pytest.mark.parametrize("over,match", [
    ({"frame": "camera"}, "frame"),
    ({"frame": None}, "frame"),
    ({"normal": [0, 0]}, "normal"),
    ({"normal": [0, 0, float("nan")]}, "normal"),
    ({"normal": [0, 0, 0]}, "zero vector"),
    ({"normal": [0, 0, -1]}, "within"),                  # points down
    ({"normal": [0.5, 0, 0.5]}, "within"),                # 45 deg tilt
    ({"normal": "z"}, "normal"),
    ({"offset_m": float("inf")}, "offset_m"),
    ({"offset_m": True}, "offset_m"),
    ({"offset_m": None}, "offset_m"),
    ({"tolerance_m": 0.0}, "tolerance_m"),
    ({"tolerance_m": 0.5}, "tolerance_m"),
    ({"search_m": 0.02}, "search_m"),                     # must exceed tolerance_m
    ({"search_m": 0.5}, "search_m"),
    ({"min_inliers": 10}, "min_inliers"),
    ({"min_inliers": 5000.5}, "min_inliers"),
    ({"min_inlier_frac": 0.0}, "min_inlier_frac"),
    ({"min_inlier_frac": 1.5}, "min_inlier_frac"),
    ({"max_median_offset_m": 0.05}, "max_median_offset_m"),
    ({"nope": 1}, "unknown keys"),
])
def test_invalid_mappings_fail_closed(over, match):
    with pytest.raises(ff.FixedFloorError, match=match):
        ff.validate_fixed_floor({**copy.deepcopy(GOOD), **over})


def test_non_mapping_is_rejected():
    with pytest.raises(ff.FixedFloorError):
        ff.validate_fixed_floor([1, 2, 3])


def test_scene_loading(tmp_path):
    shipped = sc.load_scene("dual_arm_v1")
    assert shipped.fixed_floor is not None and shipped.fixed_floor["frame"] == "base_link"
    assert sc.load_scene("table_cell_v1").fixed_floor is None          # unchanged scenes stay off
    assert sc.builtin_default().fixed_floor is None
    f = tmp_path / "s.yaml"
    f.write_text("scene: x\nlive:\n  fixed_floor:\n    frame: base_link\n    normal: [0, 0, 1]\n"
                 "    offset_m: -0.8\n")
    with pytest.raises(sc.SceneError, match="hidden_stand"):            # visible_table scenes cannot use it
        sc.load_scene(f)
    f.write_text("scene: x\ndepth_top:\n  support: hidden_stand\nlive:\n  fixed_floor:\n    frame: base_link\n"
                 "    normal: [0, 0, 1]\n    offset_m: -0.8\n")
    assert sc.load_scene(f).fixed_floor["offset_m"] == -0.8
    f.write_text("scene: x\ndepth_top:\n  support: hidden_stand\nlive:\n  fixed_floor:\n    frame: map\n"
                 "    normal: [0, 0, 1]\n    offset_m: -0.8\n")
    with pytest.raises(sc.SceneError, match="frame"):
        sc.load_scene(f)


# ---- camera-frame plane -----------------------------------------------------------------------

def test_to_camera_matches_the_construction():
    cam = ff.to_camera(ff.validate_fixed_floor(GOOD), _c2b())
    assert np.allclose(cam["n"], N_CAM, atol=1e-9) and cam["d"] == pytest.approx(FLOOR_D)
    assert cam["tolerance_m"] == 0.02 and cam["camera_height_m"] == pytest.approx(FLOOR_D, abs=1e-4)
    # a floor point of the cloud lies on the plane
    p = _plane_pts(FLOOR_D, (0.5, 0.6), (0.1, 0.2), 0.05)[0]
    assert abs(p @ np.asarray(cam["n"]) + cam["d"]) < 1e-9


def test_to_camera_refuses_a_camera_below_the_floor_and_bad_transforms():
    f = ff.validate_fixed_floor({**GOOD, "offset_m": 5.0})              # floor above the camera
    with pytest.raises(ff.FixedFloorError, match="not above"):
        ff.to_camera(f, _c2b())
    with pytest.raises(ff.FixedFloorError):
        ff.to_camera(ff.validate_fixed_floor(GOOD), np.eye(3))
    bad = _c2b()
    bad[0, 0] = float("nan")
    with pytest.raises(ff.FixedFloorError):
        ff.to_camera(ff.validate_fixed_floor(GOOD), bad)


# ---- cloud check (fail closed) -----------------------------------------------------------------

def test_evaluate_accepts_a_thick_noisy_floor():
    cam = ff.to_camera(ff.validate_fixed_floor(GOOD), _c2b())
    table, diag = ff.evaluate(_cloud(), cam)
    assert table is not None and table["fixed"] is True and "mismatch" not in diag
    assert table["n_inliers"] == diag["n_inliers"] > 50000 and diag["inlier_frac"] > 0.5
    assert abs(diag["median_offset_mm"]) < 2.0
    assert table["mask"].dtype == bool and table["mask"].sum() == table["n_inliers"]


@pytest.mark.parametrize("shift", [0.05, -0.05, 0.12])
def test_evaluate_rejects_a_moved_floor(shift):                       # camera moved / another calibration
    cam = ff.to_camera(ff.validate_fixed_floor(GOOD), _c2b())
    table, diag = ff.evaluate(_cloud(floor_shift=shift), cam)
    assert table is None and diag["mismatch"]


def test_evaluate_rejects_missing_or_sparse_floor():
    cam = ff.to_camera(ff.validate_fixed_floor(GOOD), _c2b())
    table, diag = ff.evaluate(_cloud(with_floor=False), cam)
    assert table is None and "inliers" in diag["mismatch"]
    table, diag = ff.evaluate(np.zeros((0, 3)), cam)
    assert table is None and diag["n_band"] == 0 and diag["median_offset_mm"] is None
    strict = ff.to_camera(ff.validate_fixed_floor({**GOOD, "min_inliers": 10_000_000}), _c2b())
    table, diag = ff.evaluate(_cloud(), strict)
    assert table is None and "inliers" in diag["mismatch"]


def test_evaluate_rejects_a_floor_that_is_mostly_clutter():
    cam = ff.to_camera(ff.validate_fixed_floor(GOOD), _c2b())
    rng = np.random.default_rng(1)
    pts = _cloud()
    clutter = _plane_pts(FLOOR_D, (-2.0, 2.0), (-1.0, 1.5), 0.003) + 0.03 * N_CAM      # 30 mm above: in the band
    clutter = clutter + rng.normal(0, 0.001, clutter.shape)
    table, diag = ff.evaluate(np.concatenate([pts, clutter, clutter]), cam)
    assert table is None


# ---- measurement helper -------------------------------------------------------------------------

def test_fit_floor_recovers_a_tilted_plane():
    rng = np.random.default_rng(0)
    x, y = rng.uniform(1.0, 3.0, 40000), rng.uniform(-1.0, 1.0, 40000)
    z = -0.83 - 0.001 * x - 0.006 * y + rng.normal(0, 0.007, x.shape)
    junk = np.column_stack([rng.uniform(0, 1, 2000), rng.uniform(-1, 1, 2000), rng.uniform(-0.9, -0.5, 2000)])
    fit = ff.fit_floor(np.vstack([np.column_stack([x, y, z]), junk]), z_hint=-0.83)
    v = ff.validate_fixed_floor({"frame": "base_link", "normal": fit["normal"], "offset_m": fit["offset_m"]})
    assert v["normal"][2] > 0.999 and abs(fit["tilt_deg"] - np.degrees(np.arctan(np.hypot(0.001, 0.006)))) < 0.1
    assert abs(fit["offset_m"] - (-0.83 * v["normal"][2])) < 0.002
    with pytest.raises(ff.FixedFloorError):
        ff.fit_floor(np.zeros((10, 3)), z_hint=-0.83)


# ---- used by the hidden-stand top check ----------------------------------------------------------

def test_estimate_top_uses_the_fixed_floor_and_is_stable():
    cam = cg.load_camera()
    cfg = _scene_cfg()
    offs = []
    for seed in range(3):
        est = D.estimate_top(_cloud(seed=seed), cam, (1080, 1920), cfg["depth_top"])
        assert est["ok"], est["reason"]
        d = est["depth"]
        assert d["support_source"] == "fixed_floor" and d["fixed_floor"]["n_inliers"] > 50000
        assert d["support_hidden"] is True and d["floor_points"] == d["fixed_floor"]["n_inliers"]
        assert "table_rank" not in d and "floor_fragments" not in d     # no per-frame RANSAC facet as the floor
        offs.append(d["top_offset_mm"])
    assert max(offs) - min(offs) < 3.0 and abs(offs[0] - 600.0) < 5.0


def test_estimate_top_fails_closed_when_the_cloud_contradicts_the_floor():
    cam = cg.load_camera()
    cfg = _scene_cfg()
    est = D.estimate_top(_cloud(floor_shift=0.08), cam, (1080, 1920), cfg["depth_top"])
    assert not est["ok"] and est["reason"].startswith("depth_fixed_floor_mismatch(")
    assert est["depth"]["status"] == est["reason"] and est["depth"]["fixed_floor"]["mismatch"]
    assert est["quad"] is None


def test_hidden_stand_range_still_applies_with_the_fixed_floor():
    cam = cg.load_camera()
    cfg = _scene_cfg()
    cfg["depth_top"]["offset_max"] = 0.5                        # the top is 0.6 m above the floor
    est = D.estimate_top(_cloud(), cam, (1080, 1920), cfg["depth_top"])
    assert not est["ok"] and est["reason"] == "depth_no_top"


def test_roi_hint_carries_the_fixed_flag():
    cam = cg.load_camera()
    cfg = _scene_cfg()
    roi = _plane_pts(TOP_D, (-0.2, 0.2), (-0.16, 0.16), 0.002)
    ctx = {"xyz": _cloud(), "cam": cam, "dcfg": cfg["depth_top"]}
    hint = P.estimate_table_plane(ctx)
    assert hint["fixed"] is True
    est = D.estimate_top(roi, cam, (1080, 1920), cfg["depth_top"], support_hint=hint)
    assert est["ok"] and abs(est["depth"]["top_offset_mm"] - 600.0) < 5.0


# ---- used by the pipeline's support plane and the recorded clearance -------------------------------

def test_support_plane_and_clearance_are_the_fixed_plane():
    cam = cg.load_camera()
    cfg = _scene_cfg()
    corners = _plane_pts(TOP_D, (-0.1, 0.1), (-0.1, 0.1), 0.1)[:4]
    clearances = []
    for seed in range(3):
        ctx = {"xyz": _cloud(seed=seed), "cam": cam, "dcfg": cfg["depth_top"]}
        tbl = P.estimate_table_plane(ctx)
        assert tbl["fixed"] is True and np.allclose(tbl["n"], N_CAM, atol=1e-9) and tbl["d"] == pytest.approx(FLOOR_D)
        h = P.measure_box_height(ctx, None, corners, 1.0, 5000)
        assert h["height_unavailable"] == "support_hidden" and h["support_source"] == "fixed_floor"
        clearances.append(h["support_clearance_mm"])
    assert max(clearances) - min(clearances) < 1e-6          # the plane does not depend on the cloud noise
    assert abs(clearances[0] - 600.0) < 1.0


def test_support_plane_fails_closed_on_mismatch():
    cam = cg.load_camera()
    cfg = _scene_cfg()
    corners = np.array([[0.0, 0.0, 1.0]] * 4)
    ctx = {"xyz": _cloud(floor_shift=0.08), "cam": cam, "dcfg": cfg["depth_top"]}
    assert P.estimate_table_plane(ctx) is None
    h = P.measure_box_height(ctx, None, corners, 1.0, 5000)
    assert "support_clearance_mm" not in h                        # nothing for the height waiver to trust
    assert h["support_unavailable"] == "fixed_floor_mismatch" and h["fixed_floor"]["mismatch"]


# ---- unchanged when absent -----------------------------------------------------------------------

def test_absent_option_keeps_the_ransac_floor_path():
    cam = cg.load_camera()
    cfg = _scene_cfg(floor=False)
    assert "fixed_floor" not in cfg["depth_top"]
    est = D.estimate_top(_cloud(), cam, (1080, 1920), cfg["depth_top"])
    d = est["depth"]
    assert "fixed_floor" not in d and "support_source" not in d and "floor_points" not in d
    assert "table_rank" in d or not est["ok"]
    ctx = {"xyz": _cloud(), "cam": cam, "dcfg": cfg["depth_top"]}
    tbl = P.estimate_table_plane(ctx)
    assert tbl is not None and "fixed" not in tbl
    h = P.measure_box_height(ctx, None, np.array([[0.0, 0.0, 1.0]] * 4), 1.0, 5000)
    assert "support_source" not in h and "support_unavailable" not in h


def test_fixed_floor_is_ignored_outside_hidden_stand():
    cam = cg.load_camera()
    cfg = default_config()                                    # visible_table
    cfg["depth_top"]["fixed_floor"] = ff.to_camera(ff.validate_fixed_floor(GOOD), _c2b())
    ctx = {"xyz": _cloud(), "cam": cam, "dcfg": cfg["depth_top"]}
    tbl = P.estimate_table_plane(ctx)
    assert tbl is None or "fixed" not in tbl


# ---- start-independent top plane (depth_top.top_refit_iters) -----------------------------------------

def _tilted(n, deg, axis):
    a = np.radians(deg)
    k = np.asarray(axis, dtype=float)
    k = k / np.linalg.norm(k)
    return n * np.cos(a) + np.cross(k, n) * np.sin(a) + k * (k @ n) * (1 - np.cos(a))


def test_converge_plane_top_is_start_independent():
    rng = np.random.default_rng(0)
    top = _plane_pts(TOP_D, (-0.2, 0.2), (-0.16, 0.16), 0.003)
    top = top + rng.normal(0, 0.003, top.shape)
    thresh = 0.007
    results = []
    for deg in (0.0, 0.8, -1.5):
        n0 = _tilted(N_CAM, deg, [1.0, 0.0, 0.0])
        d0 = -float(n0 @ top.mean(axis=0))
        start = {"n": n0, "d": d0, "mask": np.abs(top @ n0 + d0) < thresh}
        out = D._converge_plane_top(top, start, thresh, 12, 0.01)
        results.append(out)
        assert out["refit_steps"] >= 1 and out["n_inliers"] > 0.95 * top.shape[0]
        assert abs(float(out["n"] @ N_CAM)) > np.cos(np.radians(0.25))
    ns = np.array([r["n"] * np.sign(r["n"] @ N_CAM) for r in results])
    assert max(np.degrees(np.arccos(np.clip(ns @ ns[0], -1, 1)))) < 0.15       # same plane from every start


def test_converge_plane_top_keeps_the_input_when_the_band_is_empty():
    pts = _plane_pts(TOP_D, (-0.05, 0.05), (-0.05, 0.05), 0.05)
    start = {"n": N_CAM, "d": 50.0, "mask": np.zeros(pts.shape[0], bool)}      # plane nowhere near the points
    assert D._converge_plane_top(pts, start, 0.007, 12, 0.01) is start


def test_refit_is_off_by_default_and_on_in_the_dual_arm_scene():
    assert default_config()["depth_top"]["top_refit_iters"] == 0
    cam = cg.load_camera()
    cfg = _scene_cfg()
    assert cfg["depth_top"]["top_refit_iters"] > 0
    est = D.estimate_top(_cloud(), cam, (1080, 1920), cfg["depth_top"])
    assert est["ok"] and est["depth"]["top_refit_steps"] >= 1
    off = dict(cfg["depth_top"], top_refit_iters=0)
    assert "top_refit_steps" not in D.estimate_top(_cloud(), cam, (1080, 1920), off)["depth"]
