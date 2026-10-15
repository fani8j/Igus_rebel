"""Table-board loop closure, joint zero offsets, held-out validation and diagnostics on synthetic data.

Scene: the real ReBeL URDF (both arms, with xeg32_tool_tip), joint zero offsets of 0.4-0.56 deg per joint
injected into the PHYSICAL arms (the recorded TF / joints are the encoder values), a fixed table board seen by
the camera and touched at 4 points by both arms.  Truth is compared in the gauge the data can identify
(``synthetic.effective_truth``: joint-1 offsets are not separable from the base yaw).
"""
import json
import os

import numpy as np
import pytest
import yaml

from rebel_dual_arm_calibration import session, solve_cli, solver, synthetic as S, validation
from rebel_dual_arm_calibration.targets import default_targets_path, load_targets_all, specs_to_config, table_to_config
from rebel_dual_arm_calibration.transforms import rotation_angle_deg, xyz_rpy_to_T

pytestmark = pytest.mark.skipif(S.rebel_urdf_path() is None, reason="igus_rebel2.urdf not found")
PRE = ("arm_left_", "arm_right_")


def _err(T, Tt):
    return float(np.linalg.norm(T[:3, 3] - Tt[:3, 3]) * 1e3), rotation_angle_deg(T[:3, :3], Tt[:3, :3])


class World:
    def __init__(self, offset_deg, seed=3):
        self.arms, self.table = load_targets_all(default_targets_path())
        self.kin = S.default_kinematics(PRE)
        self.scene = S.make_kin_scene(0.9, seed=seed, offset_deg=offset_deg)
        self.samples = S.simulate_kin_samples(self.scene, self.arms, self.kin, n_per_arm=30, seed=seed + 10)
        self.tables = S.simulate_table_obs(self.scene, self.table, 10, seed=seed + 20)
        self.touches = S.simulate_touches(self.scene, self.table, self.kin, seed=seed + 30)
        self.eff = S.effective_truth(self.scene)

    def solve(self, **kw):
        opts = kw.pop("options", None) or solver.SolverOptions(**kw.pop("opt", {}))
        return solver.solve(self.samples, opts, tables=kw.pop("tables", self.tables),
                            touches=kw.pop("touches", self.touches), kin=self.kin)

    def right_err(self, res):
        return _err(res.T_world_arm_right_base, self.eff["T_W_R"])

    def cam_err(self, res):
        return _err(res.T_world_optical, self.eff["T_W_O"])


@pytest.fixture(scope="module")
def w():
    return World(0.4)


@pytest.fixture(scope="module")
def res_nooff(w):
    return w.solve()


@pytest.fixture(scope="module")
def res_off(w):
    return w.solve(opt=dict(joint_offsets=True))


# ---- (i) joint zero offsets -------------------------------------------------------------------------
def test_joint_offsets_recover_base_and_camera_better_than_without(w, res_nooff, res_off):
    r_no, r_off = w.right_err(res_nooff), w.right_err(res_off)
    c_no, c_off = w.cam_err(res_nooff), w.cam_err(res_off)
    print(f"right base error: without offsets {r_no[0]:.2f} mm {r_no[1]:.3f} deg, with {r_off[0]:.2f} mm "
          f"{r_off[1]:.3f} deg; camera {c_no[0]:.2f} -> {c_off[0]:.2f} mm")
    assert r_no[0] > 1.5 and c_no[0] > 1.5                 # unmodelled offsets grow the error
    assert r_off[0] < 0.6 and r_off[1] < 0.05              # modelled: back at the detection-noise level
    assert r_off[0] < r_no[0] / 3 and c_off[0] < c_no[0] / 3
    assert res_off.report["joint"]["rms_px"]["all"] < 0.6 < res_nooff.report["joint"]["rms_px"]["all"]


def test_estimated_offsets_reported_with_std(w, res_off):
    o = res_off.report["joint"]["joint_offsets"]
    assert o["enabled"] and o["prior_sigma_deg"] == 0.3
    for a in ("left", "right"):
        est = np.array(o["estimated_deg"][a])
        true = np.degrees(w.scene.offsets[a])
        std = o["std_deg"][a]
        assert std[0] is None and est[0] == 0.0           # joint 1: base-yaw gauge, fixed
        assert all(v is not None and 0 < v < 0.35 for v in std[1:])   # a priori sigma 0.3 deg bounds the std
        assert np.max(np.abs(est[1:5] - true[1:5])) < 0.15, (est, true)   # joints 2..5 recovered
    assert "Joint zero offsets" in solver.format_report(res_off.report)


def test_offsets_require_urdf_kinematics(w):
    with pytest.raises(solver.SolverError, match="URDF"):
        solver.solve(w.samples, solver.SolverOptions(joint_offsets=True), tables=w.tables, touches=w.touches)


def test_heldout_error_improves_with_offsets_and_is_recommended(w, res_nooff):
    opts = solver.SolverOptions()
    v = validation.cross_validate(res_nooff, opts, w.touches, w.kin, k=5)
    a, b = v["without_offsets"], v["with_offsets"]
    print(validation.format_validation(v))
    assert b["heldout_rms_px"] < 0.5 * a["heldout_rms_px"] and b["heldout_rms_mm"] < 0.5 * a["heldout_rms_mm"]
    assert b["heldout_touch_rms_mm"] < a["heldout_touch_rms_mm"] and a["heldout_touch_rms_mm"] > 3.0
    assert v["keep_joint_offsets"] is True and v["recommendation"].startswith("KEEP")
    assert len(a["folds"]) == 5 and all("rms_px" in f for f in a["folds"])
    # deterministic folds
    f1 = validation.fold_assignment(res_nooff.fit_samples, 5)
    assert np.array_equal(f1, validation.fold_assignment(res_nooff.fit_samples, 5)) and set(f1) == set(range(5))
    text = solver.format_report({**res_nooff.report, "validation": v})
    assert "Held-out validation" in text and "KEEP joint offsets" in text


def test_holdout_split_and_offsets_not_recommended_without_real_offsets():
    w0 = World(0.0, seed=4)
    res = w0.solve()
    v = validation.cross_validate(res, solver.SolverOptions(), w0.touches, w0.kin, holdout=0.25)
    assert v["method"].startswith("single hold-out") and len(v["without_offsets"]["folds"]) == 1
    assert v["keep_joint_offsets"] is False and "do NOT keep" in v["recommendation"]
    assert abs(v["with_offsets"]["heldout_rms_px"] - v["without_offsets"]["heldout_rms_px"]) < 0.05
    # no URDF -> only the no-offset variant, with the reason
    v2 = validation.cross_validate(res, solver.SolverOptions(), w0.touches, None, k=3)
    assert v2["with_offsets"] is None and v2["keep_joint_offsets"] is None and "URDF" in v2["recommendation"]


# ---- (ii) loop closure ------------------------------------------------------------------------------
def test_touch_loop_closure_recovers_right_base_and_reports_mismatch(w, res_off):
    lc = res_off.report["loop_closure"]
    assert lc["n_touches"] == {"left": 4, "right": 4} and lc["points"]["left"] == [1, 2, 3, 4]
    assert w.right_err(res_off)[0] < 0.6
    # the offset-corrected numbers (diagnostic only; the robot never applies the offsets) are at the noise level ...
    oc = lc["offset_corrected"]
    assert oc["final"]["rms_mm"]["all"] < 1.2 and oc["camera_only"]["rms_mm"]["all"] < 1.8
    assert oc["out_of_fit_source"] == "camera_only" and oc["out_of_fit_rms_mm"] < 1.8
    assert set(oc["cross_arm"]) == {"left", "right"} and all(v["rms_mm"][a] < 2.0 for a, v in oc["cross_arm"].items())
    # ... while the NOMINAL-FK numbers (what the deployed robot sees, and what the trust gate uses) show the
    # unmodelled offsets
    assert lc["final"]["rms_mm"]["all"] > 2.0 and lc["out_of_fit_rms_mm"] > 2.0 and lc["out_of_fit_source"] == "camera_only"
    assert lc["fit_residual_fre_rms_mm"] == lc["final"]["rms_mm"]["all"]
    assert "nominal FK" in lc["model"]
    assert set(lc["cross_arm"]) == {"left", "right"}
    assert len(lc["final"]["per_touch"]) == 8 and {t["arm"] for t in lc["final"]["per_touch"]} == {"left", "right"}
    assert lc["right_base_with_vs_without_touches"]["position_mm"] < 0.5
    assert [x for x in res_off.report["warnings"] if "touch loop closure" in x and "nominal FK" in x]
    text = solver.format_report(res_off.report)
    assert "HONEST" not in text and "fit residual (FRE, in-sample)" in text and "out-of-fit touch mismatch" in text
    assert "per touch (mm)" in text and "R3:" in text
    assert any("touch residual" in s for s in res_off.report["accuracy_statement"])


def test_loop_closure_detects_wrong_right_base(w):
    """The right arm touches the board with its base 5.8 mm away from where the camera data put it."""
    wrong = xyz_rpy_to_T([0.004, 0.003, -0.003], [0, 0, 0])
    touches = S.simulate_touches(w.scene, w.table, w.kin, seed=33, right_base_error=wrong)
    res = w.solve(opt=dict(joint_offsets=True), touches=touches)
    lc = res.report["loop_closure"]["offset_corrected"]     # offsets applied: isolates the displaced base
    assert lc["camera_only"]["rms_mm"]["right"] > 4.0 and lc["camera_only"]["rms_mm"]["left"] < 1.5
    assert lc["cross_arm"]["right"]["rms_mm"]["right"] > 4.0
    assert any("touch loop closure" in x for x in res.report["warnings"])
    assert w.right_err(res)[0] < 1.0       # the camera data (not the inconsistent touches) still decide the base


def test_loop_closure_detects_wrong_table_board_scale(w):
    """The printed board is 1.5 % larger than the configured 40 mm squares."""
    tables = S.simulate_table_obs(w.scene, w.table, 10, seed=23, board_scale=1.015)
    touches = S.simulate_touches(w.scene, w.table, w.kin, seed=33, board_scale=1.015)
    res = w.solve(opt=dict(joint_offsets=True), tables=tables, touches=touches)
    lc = res.report["loop_closure"]
    assert lc["camera_only"]["rms_mm"]["all"] > 5.0 and lc["final"]["rms_mm"]["all"] > 2.5
    assert any("touch loop closure" in x for x in res.report["warnings"])


def test_touches_without_table_images_use_cross_arm_check(w):
    res = w.solve(opt=dict(joint_offsets=True), tables=[])
    lc = res.report["loop_closure"]
    assert lc["camera_only"] is None and lc["out_of_fit_source"] == "cross_arm"
    assert lc["offset_corrected"]["out_of_fit_source"] == "cross_arm" and lc["offset_corrected"]["out_of_fit_rms_mm"] < 2.0
    assert w.right_err(res)[0] < 0.8


def test_touch_points_need_three_distinct_points(w):
    few = [t for t in w.touches if t.point_id in (1, 2)]
    with pytest.raises(solver.SolverError, match="distinct"):
        w.solve(opt=dict(joint_offsets=True), tables=[], touches=few)


# ---- (iii) diagnostics ------------------------------------------------------------------------------
def test_correlation_warning_on_degenerate_dataset():
    from rebel_dual_arm_calibration.synthetic import make_scene, simulate_samples
    from test_solver import charuco_specs
    scene = make_scene(distance=0.9, seed=3)
    specs = charuco_specs(4, 0.025, 0.018)
    bad = solver.solve(simulate_samples(scene, specs, n_per_arm=15, noise_px=0.3, seed=9, rotation_diversity=False))
    good = solver.solve(simulate_samples(scene, specs, n_per_arm=15, noise_px=0.3, seed=9, rotation_diversity=True))
    txt = " ".join(bad.report["warnings"])
    assert "|correlation| > 0.8" in txt
    pairs = [p for p in bad.report["correlation"]["high_pairs"] if not p["structural"]]
    assert len(pairs) > len(good.report["correlation"]["high_pairs"])
    assert any(p["a"].split(".")[0] != p["b"].split(".")[0] for p in pairs)     # coupling BETWEEN blocks
    m = np.array(bad.report["correlation"]["matrix"])
    assert m.shape == (24, 24) and np.allclose(m, m.T, atol=1e-3) and np.allclose(np.diag(m), 1.0)
    assert "Parameter correlation" in solver.format_report(bad.report)


def test_default_thresholds_and_statements_without_touches_or_offsets():
    from rebel_dual_arm_calibration.synthetic import make_scene, simulate_samples
    from test_solver import charuco_specs
    o = solver.SolverOptions()
    assert o.rot_diversity_warn_deg == 30.0 and o.min_samples_warn == 20 and o.recommended_samples == (30, 40)
    scene = make_scene(distance=0.9, seed=2)
    ss = simulate_samples(scene, charuco_specs(4, 0.025, 0.018), n_per_arm=24, noise_px=0.3, seed=5)
    res = solver.solve(ss)
    rep = res.report
    assert any("detection noise only" in s for s in rep["accuracy_statement"])
    assert any("no touch samples" in n for n in rep["notes"]) and any("recommended" in n for n in rep["notes"])
    assert rep["loop_closure"] is None and rep["joint"]["joint_offsets"]["enabled"] is False
    assert rep["joint"]["std_basis"] == "detection noise"
    text = solver.format_report(rep)
    assert "STD DEVS" in text and "detection noise only" in text
    # an arm with 14 poses warns (threshold 20)
    few = solver.solve(simulate_samples(scene, charuco_specs(4, 0.025, 0.018), n_per_arm=14, noise_px=0.3, seed=5))
    assert any("only 14 distinct poses" in x and ">= 20" in x for x in few.report["warnings"])


def test_intrinsics_pull_always_reported_and_refinement_optional(w, res_off):
    ir = res_off.report["joint"]["intrinsics"]
    assert ir["refined"] is False and set(ir["delta_px"]) == {"fx", "fy", "cx", "cy"} and "right_base_shift" in ir
    assert max(abs(v) for v in ir["delta_px"].values()) < 4.0      # correct intrinsics: no big pull
    assert ir["source"] == "camera_info" and ir["pull_f_frac"] < 0.003 and ir["pull_c_px"] < 3.0       # what the gate reads
    assert ir["pull_f_frac"] == pytest.approx(max(abs(ir["delta_px"]["fx"]) / ir["nominal_px"]["fx"],
                                                  abs(ir["delta_px"]["fy"]) / ir["nominal_px"]["fy"]))
    assert "NOT refined" in solver.format_report(res_off.report)
    # wrong camera_info (fx, fy 0.5 % too large): the data pull them back, with and without refinement
    ss = []
    for s in w.samples:
        t = solver.Sample.from_json_dict(s.to_json_dict())
        t.q, t.K = s.q, s.K.copy()
        t.K[0, 0] *= 1.005
        t.K[1, 1] *= 1.005
        ss.append(t)
    tb = []
    for t0 in w.tables:
        t = solver.TableObservation.from_json_dict(t0.to_json_dict())
        t.K = t.K.copy()
        t.K[0, 0] *= 1.005
        t.K[1, 1] *= 1.005
        tb.append(t)
    kw = dict(tables=tb, touches=w.touches, kin=w.kin)
    fixed = solver.solve(ss, solver.SolverOptions(joint_offsets=True, loop_closure_variants=False), **kw)
    pull = fixed.report["joint"]["intrinsics"]["delta_px"]
    assert pull["fx"] < -1.0 and pull["fy"] < -1.0
    assert fixed.report["joint"]["intrinsics"]["pull_f_frac"] > 0.003            # a 0.5 % wrong f is above the gate's 0.3 %
    refined = solver.solve(ss, solver.SolverOptions(joint_offsets=True, refine_intrinsics=True,
                                                    loop_closure_variants=False), **kw)
    ir2 = refined.report["joint"]["intrinsics"]
    assert ir2["refined"] is True and ir2["delta_px"]["fx"] < -1.0
    assert refined.report["joint"]["rms_px"]["all"] <= fixed.report["joint"]["rms_px"]["all"] + 1e-6


# ---- (iv) sessions: merge with touches, CLI ----------------------------------------------------------------
def _meta(w, arms):
    par = {"active_arms": arms}
    for a, p in zip(("left", "right"), PRE):
        par.update({f"{a}_base_frame": f"{p}base_link", f"{a}_tip_frame": f"{p}link6",
                    f"{a}_tcp_frame": f"{p}xeg32_tool_tip"})
    return {"note": "synthetic", "targets": {**specs_to_config(w.arms), "table_board": table_to_config(w.table)},
            "parameters": par}


def _write_session(path, w, arm, tables, urdf_text=None):
    arms = ["left", "right"] if arm is None else [arm]
    d = str(path)
    session.init_session(d, _meta(w, arms))
    for s in w.samples:
        if s.arm in arms:
            session.append_sample(d, s, {})
    for t in w.touches:
        if t.arm in arms:
            session.append_touch(d, t, {})
    for t in tables:
        session.append_table_obs(d, t, {})
    session.save_urdf_once(d, urdf_text or S.dual_urdf_text(PRE))
    return d


def test_multi_session_merge_with_touches_matches_single_session(tmp_path, w):
    both = _write_session(tmp_path / "both", w, None, w.tables)
    left = _write_session(tmp_path / "left", w, "left", w.tables[:5])
    right = _write_session(tmp_path / "right", w, "right", w.tables[5:])
    data = session.load_sessions_full([left, right])
    assert len(data.tables) == 10 and len(data.touches) == 8 and set(data.kin) == {"left", "right"}
    assert all(s.q is not None for s in data.samples) and all(t.q is not None for t in data.touches)
    assert {t.session for t in data.touches} == {"left", "right"}
    one = session.load_sessions_full([both])
    opts = dict(joint_offsets=True, loop_closure_variants=False, intrinsics_report=False)
    ra = solver.solve(one.samples, solver.SolverOptions(**opts), tables=one.tables, touches=one.touches, kin=one.kin)
    rb = solver.solve(data.samples, solver.SolverOptions(**opts), tables=data.tables, touches=data.touches,
                      kin=data.kin)
    assert _err(ra.T_world_arm_right_base, rb.T_world_arm_right_base)[0] < 2e-2     # mm
    assert _err(ra.T_world_camera_link, rb.T_world_camera_link)[0] < 2e-2
    assert _err(rb.T_world_arm_right_base, w.eff["T_W_R"])[0] < 0.8
    assert rb.report["loop_closure"]["offset_corrected"]["final"]["rms_mm"]["all"] < 1.2
    # the CLI merges the same way and tells what it found
    out = str(tmp_path / "b.yaml")
    assert solve_cli.main([left, right, "--out", out, "--joint-offsets", "--no-validate"]) == 0
    d = yaml.safe_load(open(out))["dual_arm_calibration"]
    T_R = xyz_rpy_to_T(d["arm_right_base"]["xyz"], d["arm_right_base"]["rpy"])
    # --joint-offsets without a held-out validation: no KEEP decision, so the plain NOMINAL fit is written
    # (before: the offset fit, error 0.5 mm in the identifiable gauge, always)
    assert _err(T_R, rb.T_world_arm_right_base_nominal_refit)[0] < 0.05 and _err(rb.T_world_arm_right_base, w.eff["T_W_R"])[0] < 0.8
    assert "not run" in d["notes"] and "nominal fit" in d["notes"]
    assert "touch mismatch" in d["notes"] or "fit residual (FRE" in d["notes"]
    assert "joint offsets" in d["notes"]
    rep = json.load(open(str(tmp_path / "b_report.json")))
    assert rep["loop_closure"]["n_touches"] == {"left": 4, "right": 4}
    # ignoring touches / table images removes the loop closure section
    c = str(tmp_path / "c.yaml")
    assert solve_cli.main([both, "--out", c, "--no-touches", "--no-validate"]) == 0
    assert json.load(open(str(tmp_path / "c_report.json")))["loop_closure"] is None


def test_cli_full_report_with_validation_and_kinematics_mismatch(tmp_path, w, capsys):
    both = _write_session(tmp_path / "both", w, None, w.tables)
    out = str(tmp_path / "o.yaml")
    assert solve_cli.main([both, "--out", out, "--joint-offsets", "--kfold", "3"]) == 0     # auto-validation (URDF)
    text = capsys.readouterr().out
    for needle in ("Loop closure through the table board", "out-of-fit touch mismatch", "Joint zero offsets", "Held-out validation",
                   "Intrinsics NOT refined", "STD DEVS", "table_board"):
        assert needle in text, needle
    rep = json.load(open(str(tmp_path / "o_report.json")))
    assert rep["validation"]["method"] == "deterministic 3-fold" and "keep_joint_offsets" in rep["validation"]
    assert rep["correlation"]["threshold"] == 0.8
    # the same arm recorded with a different URDF geometry in another session is refused
    other = _write_session(tmp_path / "r", w, "right", w.tables[5:],
                           urdf_text=S.dual_urdf_text(PRE).replace('xyz="0 0.049 0"', 'xyz="0 0.050 0"'))
    assert solve_cli.main([both, other, "--out", str(tmp_path / "x.yaml")]) == 2
    assert "URDF kinematics" in capsys.readouterr().err
    # --joint-offsets without a URDF is refused with a clear message
    nourdf = str(tmp_path / "nourdf")
    session.init_session(nourdf, _meta(w, ["left", "right"]))
    for s in w.samples:
        session.append_sample(nourdf, s, {})
    assert solve_cli.main([nourdf, "--joint-offsets", "--no-validate", "--out", str(tmp_path / "y.yaml")]) == 2
    assert "URDF" in capsys.readouterr().err
