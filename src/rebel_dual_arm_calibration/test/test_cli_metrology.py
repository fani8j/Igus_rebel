"""solve / check_points command lines with the metrology options, the uncertainty Monte Carlo, written-base choice."""
import json

import numpy as np
import pytest
import yaml

from rebel_dual_arm_calibration import check_points_cli, metrology, solve_cli, solver, synthetic as S
from rebel_dual_arm_calibration.transforms import xyz_rpy_to_T
from test_loop_closure import World, _err, _write_session

needs_urdf = pytest.mark.skipif(S.rebel_urdf_path() is None, reason="igus_rebel2.urdf not found")


def _gate(rep):
    return {c["name"]: c for c in rep["trust_gate"]["checks"]}


@needs_urdf
def test_solve_cli_metrology_report_notes_mc_compare_and_check_points_cli(tmp_path, good_world, capsys):
    w, _ = good_world
    d = _write_session(tmp_path / "s", w, None, w.tables)
    a = str(tmp_path / "a.yaml")
    assert solve_cli.main([d, "--out", a, "--kfold", "3", "--uncertainty-mc", "3", "--mc-seed", "5"]) == 0
    text = capsys.readouterr().out
    rep = json.load(open(str(tmp_path / "a_report.json")))
    y = yaml.safe_load(open(a))["dual_arm_calibration"]
    assert rep["trust_gate"]["passed"], "\n".join(rep["trust_gate"]["text"])
    assert y["source"] == "calibrated"
    for needle in ("Check points (check-point error (TRE, held-out)", "recommended_padding_mm = max(20, 2 x max check-point error + 10)",
                   "Uncertainty budget (Monte Carlo", "accuracy demonstrated at the check points only", "[PASS] intrinsics validated",
                   "predicted 1-sigma of the touch-only base", "fit residual (FRE, in-sample)", "fit stickers L"):
        assert needle in text, needle
    assert "HONEST" not in text
    assert "recommended_padding_mm" in y["notes"] and "check-point error (TRE, held-out" in y["notes"]
    assert "accuracy demonstrated at the check points only" in y["notes"] and "MC 95 %" in y["notes"]
    assert rep["written_base"]["fit"] == "nominal_fit" and "no joint zero offsets" in rep["written_base"]["reason"]
    assert rep["check_points"]["n_points"] == 2 and rep["n_sticker_touches"]["left"] == {"fit": 12, "check": 6}
    mc = rep["uncertainty_mc"]
    assert mc["n_used"] == 3 and set(mc["arm_right_base"]) == {"x_mm", "y_mm", "z_mm", "roll_deg", "pitch_deg", "yaw_deg"}
    assert set(mc["tcp_disagreement"]) == {"carton_centre", "left_reach", "right_reach"}
    assert mc["terms"]["tcp_sigma_mm_per_axis"] == 0.5 and mc["terms"]["joint_offset_sigma_deg"] == 0.3
    assert all(v["half_width95"] > 0 and v["lo95"] <= v["hi95"] for v in mc["arm_right_base"].values())
    assert all(v["norm_p95_mm"] > 0 for v in mc["tcp_disagreement"].values())
    # session info and the intrinsics source are in the report
    assert rep["options"]["intrinsics_source"] == "camera_info" and rep["session_info"]
    # --compare: the same data again -> no change; a calibration 3 mm / 0.3 deg away -> exceeds the guidance
    b = str(tmp_path / "b.yaml")
    assert solve_cli.main([d, "--out", b, "--no-validate", "--compare", a]) == 0
    cmp_ = json.load(open(str(tmp_path / "b_report.json")))["compare"]
    assert cmp_["arm_right_base"]["position_mm"] < 0.05 and cmp_["within_guidance"]
    assert "Reproducibility: vs" in capsys.readouterr().out
    moved = str(tmp_path / "moved.yaml")
    T = xyz_rpy_to_T(y["arm_right_base"]["xyz"], y["arm_right_base"]["rpy"]) @ xyz_rpy_to_T([0.003, 0, 0], [0, 0, np.radians(0.3)])
    yy = {"dual_arm_calibration": dict(y, arm_right_base={"xyz": [float(v) for v in T[:3, 3]],
                                                          "rpy": [float(v) for v in solver.T_to_xyz_rpy(T)[1]]})}
    yaml.safe_dump(yy, open(moved, "w"))
    assert solve_cli.main([d, "--out", str(tmp_path / "c.yaml"), "--no-validate", "--compare", moved]) == 0
    c2 = json.load(open(str(tmp_path / "c_report.json")))["compare"]
    assert c2["arm_right_base"]["position_mm"] > 2.5 and not c2["within_guidance"] and "EXCEEDS" in c2["text"]
    assert solve_cli.main([d, "--out", str(tmp_path / "e.yaml"), "--no-validate", "--compare", str(tmp_path / "nope.yaml")]) == 2

    # ---- check_points command: only the check-point error of the touches against a given calibration, no fitting ----
    assert check_points_cli.main([d, "--calibration", a]) == 0
    out = capsys.readouterr().out
    assert "RESULT: PASS" in out and "point 111" in out and "point 112" in out and "recommended_padding_mm" in out
    # a power-cycled / displaced right arm: the deployed calibration is now 5 mm off at the check points
    off = str(tmp_path / "off.yaml")
    T5 = xyz_rpy_to_T(y["arm_right_base"]["xyz"], y["arm_right_base"]["rpy"]) @ xyz_rpy_to_T([0.0, 0.005, 0], [0, 0, 0])
    yy["dual_arm_calibration"]["arm_right_base"] = {"xyz": [float(v) for v in T5[:3, 3]],
                                                    "rpy": [float(v) for v in solver.T_to_xyz_rpy(T5)[1]]}
    yaml.safe_dump(yy, open(off, "w"))
    assert check_points_cli.main([d, "--calibration", off, "--json", str(tmp_path / "cp.json")]) == 1
    assert "RESULT: FAIL" in capsys.readouterr().out
    assert json.load(open(str(tmp_path / "cp.json")))["max_mm"] > 4.0
    # explicit ids override the recorded kind (here: the fit stickers as 'check' points)
    assert check_points_cli.main([d, "--calibration", a, "--check-point-ids", "101,102,103,104"]) == 0
    assert "point 101" in capsys.readouterr().out
    assert check_points_cli.main([d, "--calibration", str(tmp_path / "nope.yaml")]) == 2


@needs_urdf
def test_displaced_check_sticker_and_missing_evidence_fail_the_gate_from_the_command_line(tmp_path, good_world, capsys):
    w, _ = good_world
    # (i) the right arm touched the check sticker 111 five millimetres from where the left arm touched it
    disp = S.simulate_sticker_touches(w.scene, w.kin, {"check": {111: w.sticker_pos["check"][111]}}, repeats=3, seed=71,
                                      displace={111: np.array([0.005, 0.0, 0.0])}, displace_arm="right")
    base = [t for t in w.touches if not (t.point_id == 111 and t.arm == "right")]
    saved = w.touches
    try:
        w.touches = base + [t for t in disp if t.arm == "right"]
        d = _write_session(tmp_path / "bad", w, None, w.tables)
    finally:
        w.touches = saved
    out = str(tmp_path / "bad.yaml")
    assert solve_cli.main([d, "--out", out, "--kfold", "3"]) == 0
    rep = json.load(open(str(tmp_path / "bad_report.json")))
    g = _gate(rep)
    assert not rep["trust_gate"]["passed"] and not g["check_point_error"]["ok"] and g["check_point_error"]["value"] > 4.0
    assert yaml.safe_load(open(out))["dual_arm_calibration"]["source"] == "calibrated_unverified"
    assert rep["trust_gate"]["recommended_padding_mm"] >= 20.0        # 2 x ~5 + 10 = ~20: the minimum
    # (ii) no check stickers at all: nothing held out
    w.touches = [t for t in saved if t.kind != "check"]
    try:
        d2 = _write_session(tmp_path / "nocheck", w, None, w.tables)
    finally:
        w.touches = saved
    assert solve_cli.main([d2, "--out", str(tmp_path / "nc.yaml"), "--no-validate"]) == 0
    g2 = _gate(json.load(open(str(tmp_path / "nc_report.json"))))
    assert not g2["check_point_coverage"]["ok"] and not g2["check_point_error"]["ok"] and not g2["heldout_validation"]["ok"]
    # (iii) only the right arm touched anything and there are no table images: only in-fit numbers exist
    w.touches = [t for t in saved if t.arm == "right" and t.kind == "board"]
    try:
        d3 = _write_session(tmp_path / "oneside", w, None, [])
    finally:
        w.touches = saved
    assert solve_cli.main([d3, "--out", str(tmp_path / "os.yaml"), "--no-validate"]) == 0
    rep3 = json.load(open(str(tmp_path / "os_report.json")))
    assert rep3["loop_closure"]["out_of_fit_rms_mm"] is None and "in-fit" in rep3["loop_closure"]["out_of_fit_source"]
    assert not rep3["trust_gate"]["passed"] and not _gate(rep3)["touch_mismatch_rms"]["ok"]
    # (iv) held-out reprojection forced bad by the threshold
    d4 = _write_session(tmp_path / "ok", w, None, w.tables)
    assert solve_cli.main([d4, "--out", str(tmp_path / "ok.yaml"), "--kfold", "3", "--gate-heldout-px", "0.05"]) == 0
    rep4 = json.load(open(str(tmp_path / "ok_report.json")))
    assert not _gate(rep4)["heldout_validation"]["ok"] and "0.05" in _gate(rep4)["heldout_validation"]["text"]
    assert all(c["ok"] for n, c in _gate(rep4).items() if n != "heldout_validation")


@needs_urdf
def test_written_base_is_the_offset_fit_only_when_validation_keeps_the_offsets():
    w = World(0.4, seed=3)
    res = w.solve(opt=dict(joint_offsets=True))
    keep = {"keep_joint_offsets": True, "keep_decision_weak": False}
    no_keep = {"keep_joint_offsets": False, "keep_decision_weak": False}
    assert solver.choose_written_fit(res, keep)[0] == "offset_fit"
    assert "KEEP" in solver.choose_written_fit(res, keep)[1] and "never applied" in solver.choose_written_fit(res, keep)[1]
    which, why = solver.choose_written_fit(res, no_keep)
    assert which == "nominal_fit" and "does not recommend KEEP" in why
    assert solver.choose_written_fit(res, None)[0] == "nominal_fit" and "not run" in solver.choose_written_fit(res, None)[1]
    assert solver.choose_written_fit(res, {"keep_joint_offsets": None})[0] == "nominal_fit"
    T_off, C_off = res.T_world_arm_right_base, res.T_world_camera_link
    T_nom = res.T_world_arm_right_base_nominal_refit
    assert _err(T_off, T_nom)[0] > 1.0                       # the two candidates really differ with 0.4 deg offsets
    try:
        solver.set_written_base(res, "nominal_fit", why)
        assert np.allclose(solver.deployed_poses(res)[0], T_nom) and res.report["written_base"]["fit"] == "nominal_fit"
        assert np.allclose(solver.deployed_poses(res)[1], res.T_world_camera_link_nominal_refit)
        solver.set_written_base(res, "offset_fit", "x")
        assert np.allclose(solver.deployed_poses(res)[0], T_off) and np.allclose(solver.deployed_poses(res)[1], C_off)
        assert res.report["written_base"]["fit"] == "offset_fit"
    finally:
        solver.set_written_base(res, "offset_fit", "restored")
    # without --joint-offsets there is only the nominal fit
    r0 = World(0.0, seed=4).solve()
    assert solver.choose_written_fit(r0, keep)[0] == "nominal_fit"


@needs_urdf
def test_uncertainty_mc_is_deterministic_and_reflects_the_type_b_terms(good_world):
    from rebel_dual_arm_calibration import uncertainty as U
    _, res = good_world
    a = U.run_mc(res, 3, seed=11)
    b = U.run_mc(res, 3, seed=11)
    assert a["arm_right_base"] == b["arm_right_base"] and a["tcp_disagreement"] == b["tcp_disagreement"]
    assert a["n_used"] == 3 and a["written_fit"] == "nominal_fit"
    # with joint offsets not fitted, the joint-offset term is part of the budget; with an intrinsics file the
    # intrinsics terms shrink (0.1 % / 1 px)
    assert a["terms"]["joint_offset_sigma_deg"] == 0.3 and a["terms"]["focal_sigma_frac"] == 0.005
    f = U.run_mc(res, 2, seed=11, intrinsics_file=True)
    assert f["terms"]["focal_sigma_frac"] == 0.001 and f["terms"]["principal_point_sigma_px"] == 1.0
    # the TCP / print scale terms dominate the detection noise: the right base moves by more than its fit std
    std = res.report["joint"]["arm_right_base"]["std_xyz_mm"]
    assert max(v["std"] for k, v in a["arm_right_base"].items() if k.endswith("_mm")) > 0.5 * max(std)
    text = "\n".join(U.format_mc(a))
    assert "predicted arm-to-arm TCP disagreement" in text and "coarse 95 % interval" in text
    pts = a["tcp_disagreement"]
    assert pts["carton_centre"]["point_world_m"][0] > 0.2           # in front of the bases, towards the board
    with pytest.raises(ValueError):
        U.run_mc(res, 0)
