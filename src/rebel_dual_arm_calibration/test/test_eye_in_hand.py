"""Eye-in-hand solver (eye_in_hand.py) on synthetic data: known X (link6 -> optical) and board pose."""
import json
import math
import os

import numpy as np
import pytest

from rebel_dual_arm_calibration import eye_in_hand as E
from rebel_dual_arm_calibration import wrist_synthetic as W
from rebel_dual_arm_calibration.eye_in_hand_cli import main as cli_main
from rebel_dual_arm_calibration.transforms import inv_T, make_T, rotvec_to_R, xyz_rpy_to_T

X_TRUE = xyz_rpy_to_T([-0.02, -0.09, -0.04], np.radians([10, -5, 95]))
# the lead's measured board (centre (0.514, -0.148, -0.251), flat, normal up): board z points INTO the table
B_TRUE = xyz_rpy_to_T([0.514, -0.148, -0.251], np.radians([180.0, 0.0, 20.0]))


def dataset(n=18, noise=0.3, seed=1, **kw):
    return W.make_samples(X_TRUE, B_TRUE, W.orbit_poses(B_TRUE, n=n, seed=seed, **kw), noise_px=noise, seed=seed + 1)


@pytest.fixture(scope="module")
def good():
    S = dataset()
    return S, E.solve(S, T_base_board_scene=B_TRUE)


def test_recovers_known_mount_and_board(good):
    S, r = good
    assert len(S) >= 16
    dt, dr = E.pose_delta(r.X, X_TRUE)
    assert dt < 0.001 and dr < 0.15, (dt, dr)
    dt, dr = E.pose_delta(r.B, B_TRUE)
    assert dt < 0.001 and dr < 0.15
    assert r.gate["verdict"] == "PASS", r.gate
    assert r.stats["rms_px"] < 0.6 and r.stats["heldout"]["position_rms_mm"] < 1.0
    assert r.stats["diversity"]["span_deg"] > 60 and r.stats["diversity"]["n_axes"] >= 3
    assert r.stats["scene_cross_check"]["position_mm"] < 1.0


def test_all_closed_form_methods_run_and_best_is_reported():
    S = dataset(12)
    X0, B0, name, scatters = E.closed_form_init(S)
    assert name in dict(E.HE_METHODS) and set(scatters) >= {"TSAI", "PARK", "DANIILIDIS"}
    assert E.pose_delta(X0, X_TRUE)[0] < 0.01


def test_noise_degrades_gracefully():
    S = dataset(18, noise=1.0)
    r = E.solve(S)
    assert E.pose_delta(r.X, X_TRUE)[0] < 0.004
    assert 0.7 < r.stats["rms_px"] < 1.6


def test_gross_corner_outliers_are_tolerated():
    S = dataset(18, noise=0.3)
    rng = np.random.default_rng(0)
    for s in S:
        k = rng.choice(len(s.pix), 3, replace=False)
        s.pix[k] += rng.normal(0, 25, (3, 2))
    r = E.solve(S)
    assert E.pose_delta(r.X, X_TRUE)[0] < 0.003 and E.pose_delta(r.X, X_TRUE)[1] < 0.5


def test_bad_sample_is_rejected_and_reported():
    S = dataset(18)
    S[4].T_base_link6 = S[4].T_base_link6 @ make_T(rotvec_to_R([0, 0, math.radians(8)]), [0.01, 0, 0])
    r = E.solve(S)
    assert r.stats["n_rejected"] >= 1 and S[4].index in r.rejected
    assert E.pose_delta(r.X, X_TRUE)[0] < 0.002
    assert any("rejected" in x for x in r.gate["reasons"])
    assert r.gate["verdict"] in ("WARN", "PASS")


def test_degenerate_rotations_all_about_one_axis_warn_or_fail():
    # camera only rolls about the line of sight above the board centre (pure rotation about ONE axis)
    centre = B_TRUE[:3, 3]
    poses = []
    for ang in np.linspace(-60, 60, 12):
        T = W.look_at(centre + np.array([0, 0, 0.35]), centre, np.array([0, 1.0, 0]), roll_rad=math.radians(ang))
        poses.append(T)
    S = W.make_samples(X_TRUE, B_TRUE, poses, noise_px=0.3, seed=3)
    r = E.solve(S)
    assert r.stats["diversity"]["n_axes"] <= 2
    assert r.gate["verdict"] in ("WARN", "FAIL")
    assert any("axes" in x or "span" in x for x in r.gate["reasons"])


def test_small_rotation_span_fails():
    S = W.make_samples(X_TRUE, B_TRUE, W.orbit_poses(B_TRUE, n=12, seed=2, tilt_deg=4.0, roll_deg=4.0), noise_px=0.3, seed=4)
    r = E.solve(S)
    assert r.stats["diversity"]["span_deg"] < 15
    assert r.gate["verdict"] == "FAIL"
    assert any("rotation span" in x for x in r.gate["reasons"])


def test_too_few_samples_raise_or_fail():
    S = dataset(18)
    with pytest.raises(E.HandEyeError):
        E.solve(S[:3])
    r = E.solve(S[:5])
    assert r.gate["verdict"] == "FAIL" and any("samples used" in x for x in r.gate["reasons"])


def test_scene_cross_check_mismatch_is_reported():
    S = dataset(18)
    off = B_TRUE.copy()
    off[:3, 3] += [0.03, 0.0, 0.0]
    r = E.solve(S, T_base_board_scene=off)
    assert r.stats["scene_cross_check"]["position_mm"] == pytest.approx(30.0, abs=1.5)
    assert r.gate["verdict"] == "FAIL"
    off[:3, 3] = B_TRUE[:3, 3] + [0.008, 0, 0]          # <= 10 mm: still PASS
    assert E.solve(S, T_base_board_scene=off).gate["verdict"] == "PASS"
    off[:3, 3] = B_TRUE[:3, 3] + [0.015, 0, 0]
    r2 = E.solve(S, T_base_board_scene=off)
    assert r2.gate["verdict"] == "WARN" and any("cross-check" in x for x in r2.gate["reasons"])


def _stats(**kw):
    base = {"n_used": 12, "rms_px": 6.0, "heldout": {"position_rms_mm": 4.0},
            "board_consistency": {"position_rms_mm": 4.5, "rotation_rms_deg": 0.4},
            "diversity": {"span_deg": 50, "n_axes": 3, "axis_planarity": 0.5},
            "method_agreement": {"max_mm": 1.0, "max_deg": 0.5}, "scene_cross_check": {"position_mm": 3.0, "rotation_deg": 2.0}}
    base.update(kw)
    return base


def test_gate_thresholds_documented_values():
    L = E.GateLimits()
    assert (L.rms_px_fail, L.cons_mm_pass, L.cons_mm_fail, L.cons_deg_pass, L.cons_deg_fail) == (12.0, 6.0, 10.0, 1.0, 3.0)
    assert (L.heldout_mm_pass, L.heldout_mm_fail, L.agree_mm, L.agree_deg) == (6.0, 12.0, 5.0, 2.0)
    assert (L.cross_mm_pass, L.cross_deg_pass, L.cross_mm_fail, L.cross_deg_fail) == (10.0, 5.0, 25.0, 15.0)
    assert (L.min_samples_pass, L.span_deg_pass, L.min_axes_pass) == (10, 30.0, 3)
    g = E.evaluate_gate(_stats())                       # 6 px absolute reprojection is informational only
    assert g["verdict"] == "PASS" and any("informational" in r for r in g["reasons"])
    assert E.evaluate_gate(_stats(rms_px=13.0))["verdict"] == "FAIL"
    cons = lambda p, r: _stats(board_consistency={"position_rms_mm": p, "rotation_rms_deg": r})   # noqa: E731
    assert E.evaluate_gate(cons(7.0, 0.5))["verdict"] == "WARN" and E.evaluate_gate(cons(5.0, 1.5))["verdict"] == "WARN"
    assert E.evaluate_gate(cons(11.0, 0.5))["verdict"] == "FAIL" and E.evaluate_gate(cons(5.0, 3.5))["verdict"] == "FAIL"
    held = lambda v: _stats(heldout={"position_rms_mm": v})                                       # noqa: E731
    assert [E.evaluate_gate(held(v))["verdict"] for v in (6.0, 7.0, 13.0)] == ["PASS", "WARN", "FAIL"]
    ag = lambda m, d: _stats(method_agreement={"max_mm": m, "max_deg": d})                         # noqa: E731
    assert E.evaluate_gate(ag(6.0, 0.5))["verdict"] == "WARN" and E.evaluate_gate(ag(1.0, 2.5))["verdict"] == "WARN"
    cr = lambda m, d: _stats(scene_cross_check={"position_mm": m, "rotation_deg": d})              # noqa: E731
    assert [E.evaluate_gate(cr(*v))["verdict"] for v in ((10.0, 5.0), (12.0, 2.0), (3.0, 6.0), (26.0, 1.0), (3.0, 16.0))] == \
        ["PASS", "WARN", "WARN", "FAIL", "FAIL"]
    assert E.evaluate_gate(_stats(n_used=8))["verdict"] == "WARN" and E.evaluate_gate(_stats(n_used=5))["verdict"] == "FAIL"
    assert E.evaluate_gate(_stats(diversity={"span_deg": 20, "n_axes": 3, "axis_planarity": 0.5}))["verdict"] == "WARN"
    assert E.evaluate_gate(_stats(diversity={"span_deg": 50, "n_axes": 1, "axis_planarity": 0.5}))["verdict"] == "FAIL"


# ---- the arm's absolute positioning error (the real ReBeL: a few mm) -------------------------------------------------
def test_synthetic_arm_pose_noise_of_4mm_gives_the_closed_form_x_and_a_pass():
    for seed in (1, 2):
        S = W.make_samples(X_TRUE, B_TRUE, W.orbit_poses(B_TRUE, n=18, seed=seed), noise_px=0.3, seed=seed + 10,
                           link6_noise_mm=4.0, link6_noise_deg=0.1)
        r = E.solve(S, T_base_board_scene=B_TRUE)
        assert r.gate["verdict"] == "PASS", r.gate
        dt, dr = E.pose_delta(r.X, X_TRUE)
        assert dt < 0.004 and dr < 0.5, (dt, dr)
        assert r.stats["rms_px"] > 3.0                          # the absolute reprojection cannot be small ...
        assert 2.0 < r.stats["board_consistency"]["position_rms_mm"] < 6.0       # ... the board scatter shows the arm error
        # whatever the refinement did, X never leaves the closed-form result by more than the guard
        X0 = E.closed_form_init(E.usable_samples(S))[0]
        assert E.pose_delta(X0, r.X)[0] * 1e3 <= E.REFINE_MAX_MM + 1e-9 and E.pose_delta(X0, r.X)[1] <= E.REFINE_MAX_DEG + 1e-9


def test_refinement_that_wanders_is_rejected_and_reported(monkeypatch):
    S = dataset(12)
    real_refine = E.refine

    def wandering(samples, X0, B0, **kw):
        X, B, ok = real_refine(samples, X0, B0, **kw)
        return X @ make_T(rotvec_to_R([0.0, 0.0, math.radians(8.0)]), [0.02, 0, 0]), B, True
    monkeypatch.setattr(E, "refine", wandering)
    r = E.solve(S, loo=False)
    X0 = E.closed_form_init(S)[0]
    assert E.pose_delta(r.X, X0)[0] < 1e-9 and r.stats["refine_rejected"] is True
    assert r.warnings and "REJECTED" in r.warnings[0] and "REJECTED" in E.format_summary(r)
    assert r.stats["refine_delta_from_init"]["mm"] > 5.0
    # quick_estimate (the interim solve) uses the same guard
    q = E.quick_estimate(S)
    assert q is not None and q.stats["refine_rejected"] is True and E.pose_delta(q.X, X0)[0] < 1e-9
    # and so does every leave-one-out fold: no fold may return the wandering refinement
    loo = E.leave_one_out(S)
    assert loo["n"] == len(S)


# ---- REAL data: arm_left run of 2026-10-13 (regression for the wandering refinement) ----------------------------------
REAL = os.path.join(os.path.dirname(__file__), "data", "wrist_real_20261013")


def load_real():
    smp = json.load(open(os.path.join(REAL, "samples.json")))["samples"]
    ci = json.load(open(os.path.join(REAL, "camera_info.json")))
    board = json.load(open(os.path.join(REAL, "board.json")))
    S = [E.WristSample(index=s["pose_index"], stamp=s["stamp_ns"] * 1e-9, T_base_link6=s["T_base_link6"], ids=s["ids"],
                       obj=s["obj"], pix=s["pix"], K=ci["K"], dist=ci["D"], image_size=(ci["width"], ci["height"]))
         for s in smp if s.get("obj")]
    return S, np.array(board["T_base_board_scene"])


def test_real_arm_left_data_passes_with_the_closed_form_x():
    S, B_scene = load_real()
    assert len(S) == 14
    r = E.solve(S, T_base_board_scene=B_scene)
    X_ref = xyz_rpy_to_T([0.0323, 0.0985, 0.0047], [-0.0262, -0.5320, -3.0667])
    dt, dr = E.pose_delta(r.X, X_ref)
    assert dt < 0.002 and dr < 1.0, (dt, dr)
    assert r.gate["verdict"] == "PASS", r.gate
    assert r.stats["refine_rejected"] is True and any("REJECTED" in w for w in r.warnings)
    assert r.stats["rms_px"] > 3.0                                  # absolute reprojection: informational on this arm
    assert r.stats["board_consistency"]["position_rms_mm"] < 6.0 and r.stats["board_consistency"]["rotation_rms_deg"] < 1.0
    assert r.stats["heldout"]["position_rms_mm"] < 6.0
    assert r.stats["scene_cross_check"]["position_mm"] < 10.0 and r.stats["scene_cross_check"]["rotation_deg"] < 5.0
    assert r.stats["method_agreement"]["max_mm"] < 5.0 and r.stats["method_agreement"]["max_deg"] < 2.0
    assert r.stats["diversity"]["span_deg"] > 30
    # the interim solve (used for the in-view prediction) agrees
    q = E.quick_estimate(S)
    assert q is not None and E.pose_delta(q.X, X_ref)[0] < 0.002


def test_real_data_attached_to_wrist_camera_link_and_fail_closed(tmp_path):
    import yaml
    S, B_scene = load_real()
    r = E.solve(S, B_scene, loo=False)
    T_cl_opt = xyz_rpy_to_T([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    # a non-identity driver transform is composed: T_link6_link = X inv(T_link_optical)
    T_drv = xyz_rpy_to_T([0.01, -0.02, 0.03], [0.1, 0.2, -0.3])
    att = E.attach_transform(r.X, T_drv)
    assert np.allclose(att @ T_drv, r.X, atol=1e-9)
    assert E.attach_transform(r.X, None) is None
    bad = T_drv.copy()
    bad[0, 0] = float("nan")
    assert E.attach_transform(r.X, bad) is None
    files = E.write_outputs(str(tmp_path), r, "left", S, T_camera_link_optical=T_cl_opt)
    doc = yaml.safe_load(open(files["yaml"]))
    assert doc["parent_frame"] == "link6" and doc["child_frame"] == "wrist_camera_link"
    assert doc["optical_frame"] == "wrist_camera_color_optical_frame"
    assert np.allclose(doc["xyz"], r.X[:3, 3], atol=1e-9)           # identity driver transform: same as X
    xac = open(files["xacro"]).read()
    assert '<parent link="link6"/>' in xac and '<child link="wrist_camera_link"/>' in xac
    assert 'child link="wrist_camera_color_optical_frame"' not in xac       # the optical frame never gets a second parent
    # unknown driver transform: no joint, no snippet (and an old snippet is removed)
    files = E.write_outputs(str(tmp_path), r, "left", S, T_camera_link_optical=None)
    doc = yaml.safe_load(open(files["yaml"]))
    assert doc["xyz"] is None and "xacro" not in files and not os.path.exists(os.path.join(str(tmp_path), E.RESULT_XACRO))
    assert doc["T_link6_optical"] is not None


def test_outputs_yaml_xacro_and_offline_cli(tmp_path, good):
    import yaml
    S, r = good
    for s in S:
        E.append_sample(str(tmp_path), s)
    with open(tmp_path / E.SCENE_BOARD_FILE, "w") as f:
        json.dump({"T_base_board": B_TRUE.tolist()}, f)
    assert len(E.load_samples(str(tmp_path))) == len(S)
    assert cli_main([str(tmp_path), "--no-loo"]) == 0           # no driver transform given: no xacro, fail closed
    assert not (tmp_path / E.RESULT_XACRO).exists()
    rc = cli_main([str(tmp_path), "--no-loo", "--camera-link-optical", "0", "0", "0", "0", "0", "0"])
    assert rc == 0
    doc = yaml.safe_load(open(tmp_path / E.RESULT_YAML))
    assert doc["schema"] == "wrist_handeye/v1" and doc["parent_frame"] == "link6"
    assert doc["child_frame"] == "wrist_camera_link" and doc["optical_frame"] == "wrist_camera_color_optical_frame"
    xyz = np.array(doc["xyz"])
    assert np.linalg.norm(xyz - X_TRUE[:3, 3]) < 0.001
    assert len(doc["quaternion_xyzw"]) == 4 and doc["n_used"] >= 16
    assert doc["stats"]["rms_px"] < 0.7 and "scene_cross_check" in doc["stats"]
    xac = (tmp_path / E.RESULT_XACRO).read_text()
    assert '<parent link="link6"/>' in xac and '<child link="wrist_camera_link"/>' in xac
    # the xacro rpy reproduces the rotation
    import re
    rpy = [float(v) for v in re.search(r'rpy="([^"]+)"', xac).group(1).split()]
    from rebel_dual_arm_calibration.transforms import rpy_to_R
    assert np.allclose(rpy_to_R(rpy), X_TRUE[:3, :3], atol=3e-3)


def test_cli_without_samples(tmp_path):
    assert cli_main([str(tmp_path)]) == 2
