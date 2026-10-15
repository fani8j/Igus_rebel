"""Touch-only right base (Kabsch), TCP consistency, trust gate, YAML source, keep rule, validation leakage and
observability (synthetic; the scene tests need the ReBeL URDF)."""
import dataclasses
import json

import numpy as np
import pytest
import yaml

from rebel_dual_arm_calibration import base_check, solve_cli, solver, synthetic as S, validation
from rebel_dual_arm_calibration.base_check import GateOptions, evaluate_gate, tcp_consistency, touch_only_base
from rebel_dual_arm_calibration.solver import Touch
from rebel_dual_arm_calibration.transforms import inv_T, rotation_angle_deg, rpy_to_R, xyz_rpy_to_T
from test_loop_closure import World, _err, _write_session

needs_urdf = pytest.mark.skipif(S.rebel_urdf_path() is None, reason="igus_rebel2.urdf not found")


def _rot(rpy):
    return rpy_to_R(rpy)


def _synthetic_touches(T_W_R, n_points=4, noise=2e-4, seed=0, repeats=1, spread=0.12):
    """Both arms touch the same board points; positions are TCP positions in the arm base frames."""
    rng = np.random.default_rng(seed)
    pts = np.array([[-spread, -0.8 * spread, 0.0], [spread, -0.8 * spread, 0.0], [spread, 0.8 * spread, 0.0],
                    [-spread, 0.8 * spread, 0.0], [0.0, 0.0, 0.0], [0.0, spread, 0.0]])[:n_points]
    pts_w = pts + np.array([0.22, 0.30, 0.0])
    out = []
    for k, p in enumerate(pts_w):
        for _ in range(repeats):
            for arm in ("left", "right"):
                p_base = p if arm == "left" else inv_T(T_W_R)[:3, :3] @ p + inv_T(T_W_R)[:3, 3]
                T = np.eye(4)
                T[:3, :3] = _rot(rng.uniform(-1, 1, 3))
                T[:3, 3] = p_base + rng.normal(0, noise, 3)
                out.append(Touch(arm=arm, point_id=k + 1, p_board=pts[k], T_base_tcp=T))
    return out


# ---- touch-only base ---------------------------------------------------------------------------------
def test_kabsch_recovers_known_right_base_with_noise():
    T_R = xyz_rpy_to_T([0.4504, 0.012, -0.004], [0.004, -0.003, np.radians(2.0)])
    errs = []
    for seed in range(6):
        tb = touch_only_base(_synthetic_touches(T_R, noise=2e-4, seed=seed, repeats=2, spread=0.15), T_R)
        assert tb["available"] and tb["n_common"] == 4
        T = np.array(tb["T"])
        errs.append((np.linalg.norm(T[:3, 3] - T_R[:3, 3]) * 1e3, rotation_angle_deg(T[:3, :3], T_R[:3, :3])))
        # comparison with the true base used as the "joint" estimate matches the error
        assert abs(tb["comparison"]["origin_mm"] - errs[-1][0]) < 1e-6
    assert np.mean([e[0] for e in errs]) < 1.0 and max(e[1] for e in errs) < 0.3, errs
    # noise-free: exact
    tb = touch_only_base(_synthetic_touches(T_R, noise=0.0), T_R)
    assert tb["residual_rms_mm"] < 1e-6 and tb["comparison"]["origin_mm"] < 1e-6 and tb["comparison"]["rotation_deg"] < 1e-6
    assert tb["direct_mismatch_with_joint_base"]["rms_mm"] < 1e-6


def test_touch_only_base_requirements_and_spread_warning():
    T_R = xyz_rpy_to_T([0.45, 0, 0], [0, 0, 0])
    assert not touch_only_base(_synthetic_touches(T_R, n_points=2))["available"]
    # right arm touched only points 1-3, left arm 3-4: only one common point
    ts = [t for t in _synthetic_touches(T_R) if (t.arm == "left" and t.point_id >= 3) or
          (t.arm == "right" and t.point_id <= 3)]
    r = touch_only_base(ts)
    assert not r["available"] and r["n_common"] == 1
    # collinear common points are refused
    col = []
    for k in range(4):
        for arm in ("left", "right"):
            T = np.eye(4)
            T[:3, 3] = [0.1 * k, 0.3, 0.0] if arm == "left" else inv_T(T_R)[:3, :3] @ [0.1 * k, 0.3, 0.0] + inv_T(T_R)[:3, 3]
            col.append(Touch(arm=arm, point_id=k + 1, p_board=np.zeros(3), T_base_tcp=T))
    r = touch_only_base(col)
    assert not r["available"] and "collinear" in r["reason"]
    # small spread (points 20 mm apart): a warning about the yaw sensitivity
    small = touch_only_base(_synthetic_touches(T_R, spread=0.02), T_R)
    assert small["available"] and any("spread" in w for w in small["warnings"])
    wide = touch_only_base(_synthetic_touches(T_R, spread=0.25), T_R)
    assert not wide["warnings"] and wide["geometry"]["reach_amplification"] < small["geometry"]["reach_amplification"]


# ---- TCP consistency ----------------------------------------------------------------------------------
def test_tcp_consistency_measures_tcp_offset_error():
    T_R = xyz_rpy_to_T([0.45, 0, 0], [0, 0, 0])
    # 5 touches per point with random orientations: tilt diversity (sigma_min/sqrt(N) >= 0.2) holds at every point
    good = _synthetic_touches(T_R, noise=1e-4, repeats=5, seed=2)
    c = tcp_consistency(good)
    for arm in ("left", "right"):
        assert c["arms"][arm]["n_points"] == 4 and c["arms"][arm]["spread_rms_mm"] < 0.5
    # a 6 mm error of the TCP along the tool axis shows as a spread over different wrist orientations
    bad = []
    for t in good:
        T = t.T_base_tcp.copy()
        if t.arm == "right":
            T[:3, 3] += T[:3, :3] @ np.array([0.0, 0.0, 0.006])
        bad.append(Touch(arm=t.arm, point_id=t.point_id, p_board=t.p_board, T_base_tcp=T))
    cb = tcp_consistency(bad)
    assert cb["arms"]["right"]["spread_rms_mm"] > 3.0 and cb["arms"]["left"]["spread_rms_mm"] < 0.5
    assert any("right: TCP consistency" in w for w in cb["warnings"])
    # touches at (almost) the same orientation do not test the TCP
    same = [Touch(arm="left", point_id=1, p_board=np.zeros(3), T_base_tcp=np.eye(4)) for _ in range(2)]
    cs = tcp_consistency(same)
    assert cs["arms"]["left"]["n_points"] == 0 and cs["arms"]["left"]["skipped_similar_orientation"]


# ---- gate on synthetic scenes -------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def good(good_world):
    """See conftest.good_world: the consistent scene with fit and check stickers, solved and validated."""
    return good_world


@needs_urdf
def test_gate_passes_on_consistent_data_and_estimates_agree(good):
    w, res = good
    tb = res.report["touch_base"]
    assert tb["available"] and tb["n_common"] == 8             # 4 board crosshairs + 4 fit stickers (check points excluded)
    assert tb["point_kinds"] == {1: "board", 2: "board", 3: "board", 4: "board", 101: "fit", 102: "fit", 103: "fit", 104: "fit"}
    T_t = res.T_world_arm_right_base_touch
    assert _err(T_t, w.eff["T_W_R"])[0] < 1.0 and _err(T_t, w.eff["T_W_R"])[1] < 0.15       # was 2.5 mm / 0.4 deg (board only)
    assert tb["comparison"]["origin_mm"] < 1.0 and tb["comparison"]["reach_mm"] < 1.0 and tb["comparison"]["rotation_deg"] < 0.15
    # stickers never enter the joint solve, the loop closure or the validation folds
    assert res.report["n_touches"] == {"left": 32, "right": 32} and res.report["loop_closure"]["n_touches"] == {"left": 32, "right": 32}
    assert res.report["n_sticker_touches"]["right"] == {"fit": 12, "check": 6}
    assert res.report["validation"]["without_offsets"]["n_heldout_touches"] == 64
    assert all(t.kind == "board" for t in res.touches_fit) and len(res.touches_all) == 64 + 36
    g = evaluate_gate(res.report, GateOptions())
    print("\n".join(g["text"]))
    assert g["passed"] and g["source"] == "calibrated" and all(c["ok"] for c in g["checks"])
    assert g["text"][0].startswith("TRUST GATE") and "PASS" in g["text"][0]
    # the synthetic IK touches differ only by a roll about the tool axis: the TCP component along it is NOT observable,
    # so no point qualifies for the TCP consistency number (before: the total rotation angle alone qualified them)
    tcp = res.report["touch_base"]["tcp_consistency"]["arms"]["right"]
    assert tcp["n_points"] == 0 and len(tcp["insufficient_tilt"]) >= 4
    assert np.allclose(np.abs(tcp["insufficient_tilt"][0]["weakest_direction_tool"]), [0, 0, 1], atol=1e-3)
    assert "Touch-only right base" in solver.format_report({**res.report, "trust_gate": g})


@needs_urdf
def test_gate_fails_on_displaced_base_few_points_missing_validation_and_large_mismatch(good):
    w, res = good
    rep = res.report
    # (1) the right arm touched with its base 6 mm away from where the camera data put it
    wrong = xyz_rpy_to_T([0.004, 0.003, -0.003], [0, 0, 0])
    touches = (S.simulate_touches(w.scene, w.table, w.kin, repeats=3, seed=37, right_base_error=wrong)
               + [t for t in w.touches if t.kind != "board"])
    r1 = w.solve(touches=touches)
    r1.report["validation"] = rep["validation"]
    g1 = evaluate_gate(r1.report, GateOptions())
    assert not g1["passed"] and g1["source"] == "calibrated_unverified"
    failed = {c["name"] for c in g1["checks"] if not c["ok"]}
    assert "joint_vs_touch" in failed or "touch_mismatch_rms" in failed
    assert r1.report["touch_base"]["comparison"]["origin_mm"] > 3.0 or \
        r1.report["touch_base"]["direct_mismatch_with_joint_base"]["rms_mm"] > 1.5
    # (2) fewer than 4 common points
    t3 = [t for t in w.touches if t.point_id in (1, 2, 3)]            # 3 board points, no stickers
    r3 = w.solve(touches=t3)
    r3.report["validation"] = rep["validation"]
    g3 = evaluate_gate(r3.report, GateOptions())
    assert r3.report["touch_base"]["n_common"] == 3 and not g3["passed"]
    assert not next(c for c in g3["checks"] if c["name"] == "common_touch_points")["ok"]
    # (3) held-out validation not done
    rep2 = {k: v for k, v in rep.items() if k != "validation"}
    g2 = evaluate_gate(rep2, GateOptions())
    assert not g2["passed"] and not next(c for c in g2["checks"] if c["name"] == "heldout_validation")["ok"]
    # (4) a tight touch threshold fails otherwise good data; a loose one passes
    assert not evaluate_gate(rep, GateOptions(max_touch_rms_mm=1e-3))["passed"]
    assert evaluate_gate(rep, GateOptions(max_touch_rms_mm=50.0, max_joint_vs_touch_mm=50.0))["passed"]
    # (5) no touches at all
    r0 = w.solve(touches=[])
    r0.report["validation"] = rep["validation"]
    assert not evaluate_gate(r0.report, GateOptions())["passed"]


@needs_urdf
def test_cli_sets_source_by_gate_and_base_from(tmp_path, good, capsys):
    w, res = good
    d = _write_session(tmp_path / "s", w, None, w.tables)
    out = str(tmp_path / "ok.yaml")
    assert solve_cli.main([d, "--out", out, "--kfold", "3"]) == 0          # exit code 0 in every case
    y = yaml.safe_load(open(out))["dual_arm_calibration"]
    rep = json.load(open(str(tmp_path / "ok_report.json")))
    assert rep["trust_gate"]["passed"] and y["source"] == "calibrated" and "TRUST GATE PASS" in y["notes"]
    text = capsys.readouterr().out
    assert "TRUST GATE" in text and "[PASS]" in text and "Touch-only right base" in text
    T_y = xyz_rpy_to_T(y["arm_right_base"]["xyz"], y["arm_right_base"]["rpy"])
    assert np.allclose(T_y, res.T_world_arm_right_base, atol=2e-3)
    # --base-from touches writes the touch-only base
    out2 = str(tmp_path / "t.yaml")
    assert solve_cli.main([d, "--out", out2, "--base-from", "touches", "--kfold", "3"]) == 0
    y2 = yaml.safe_load(open(out2))["dual_arm_calibration"]
    rep2 = json.load(open(str(tmp_path / "t_report.json")))
    T_t = xyz_rpy_to_T(y2["arm_right_base"]["xyz"], y2["arm_right_base"]["rpy"])
    assert np.allclose(T_t, np.array(rep2["touch_base"]["T"]), atol=1e-9) and "touch-only" in y2["notes"]
    assert np.allclose(y2["camera_link"]["xyz"], y["camera_link"]["xyz"], atol=2e-3)
    # a failing gate: tight touch threshold -> calibrated_unverified, still exit code 0, gate in report and notes
    out3 = str(tmp_path / "f.yaml")
    assert solve_cli.main([d, "--out", out3, "--kfold", "3", "--gate-touch-rms-mm", "0.0001"]) == 0
    y3 = yaml.safe_load(open(out3))["dual_arm_calibration"]
    assert y3["source"] == "calibrated_unverified" and "TRUST GATE FAIL" in y3["notes"]
    assert json.load(open(str(tmp_path / "f_report.json")))["trust_gate"]["passed"] is False
    # --base-from touches without a usable estimate is refused
    nt = str(tmp_path / "notouch.yaml")
    assert solve_cli.main([d, "--out", nt, "--no-touches", "--base-from", "touches", "--no-validate"]) == 2
    assert "touch-only base" in capsys.readouterr().err


# ---- nominal FK is what matters; fitted offsets are nuisance ------------------------------------------------
@pytest.fixture(scope="module")
def w4():
    return World(0.4, seed=3)


@needs_urdf
def test_nominal_prediction_error_motivates_the_written_base(w4):
    """The right base written to the YAML is the offset-fit base (joint-1 offset absorbed in the yaw): it predicts the
    nominal-FK tip positions about as well as the physical base, better than a plain refit with the offsets forced
    to 0 (camera and base then absorb the offsets' effect)."""
    res = w4.solve(opt=dict(joint_offsets=True))
    pred = {n: S.nominal_prediction_error(w4.scene, w4.kin, T)["rms_mm"] for n, T in
            (("truth", w4.eff["T_W_R"]), ("offset_fit", res.T_world_arm_right_base),
             ("nominal_refit", res.T_world_arm_right_base_nominal_refit))}
    print(pred)
    assert pred["truth"] > 1.5                      # even the physical base leaves mm of nominal-model error
    assert pred["offset_fit"] < 1.2 * pred["truth"] and pred["offset_fit"] < pred["nominal_refit"]
    nr = res.report["nominal_refit"]
    assert nr["same_as_joint"] is False and nr["difference_vs_joint"]["arm_right_base"]["position_mm"] > 1.0
    assert res.report["loop_closure"]["offset_corrected"]["out_of_fit_rms_mm"] < res.report["loop_closure"]["out_of_fit_rms_mm"]
    assert any("NEVER" in s for s in res.report["accuracy_statement"])
    assert "NEVER apply" in res.report["joint"]["joint_offsets"]["note"]


# ---- keep rule ------------------------------------------------------------------------------------------------------
@needs_urdf
def test_keep_rule_needs_touch_improvement_and_is_weak_without_touches(w4):
    res = w4.solve()
    opts = solver.SolverOptions()
    v = validation.cross_validate(res, opts, w4.touches, w4.kin, k=4)
    a, b = v["without_offsets"], v["with_offsets"]
    assert v["keep_joint_offsets"] is True and v["keep_decision_weak"] is False
    gain = 1.0 - b["heldout_touch_rms_mm"] / a["heldout_touch_rms_mm"]
    assert gain > 0.1 and "NEVER be applied" in v["recommendation"]
    # require a larger touch improvement than there is: reprojection improves a lot, touch does not by enough
    strict = dataclasses.replace(opts, keep_touch_margin=min(0.99, gain + 0.05))
    v2 = validation.cross_validate(res, strict, w4.touches, w4.kin, k=4)
    assert v2["keep_joint_offsets"] is False and "do NOT keep" in v2["recommendation"]
    assert v2["with_offsets"]["heldout_rms_px"] < 0.5 * v2["without_offsets"]["heldout_rms_px"]
    # no touches: reprojection alone decides, and the decision is marked weak
    r0 = w4.solve(touches=[])
    v3 = validation.cross_validate(r0, opts, [], w4.kin, k=4)
    assert v3["keep_joint_offsets"] is True and v3["keep_decision_weak"] is True
    assert "weak" in v3["recommendation"] and "NEVER be applied" in v3["recommendation"]


# ---- held-out leakage ---------------------------------------------------------------------------------------------
@needs_urdf
def test_validation_folds_never_see_heldout_samples_in_init_or_filtering(w4, monkeypatch):
    samples = [S.Sample.from_json_dict(s.to_json_dict()) for s in w4.samples]
    for s, o in zip(samples, w4.samples):
        s.q = o.q
    # one gross outlier: the full-data fit rejects it, so the fold training sets must still contain it
    # (filtering is redone per fold) unless it is held out
    bad = samples[7]
    bad.pix = bad.pix + np.random.default_rng(1).normal(0, 60.0, bad.pix.shape)
    res = solver.solve(samples, solver.SolverOptions(), tables=w4.tables, touches=w4.touches, kin=w4.kin)
    assert not any(s is bad for s in res.fit_samples)                        # rejected by the full-data fit
    k = 4
    folds = validation.fold_assignment(res.input_samples, k)
    held = {f: {id(s) for s, ff in zip(res.input_samples, folds) if ff == f} for f in range(k)}
    calls = []
    real = solver._run_fit

    def spy(samples_in, options, tables, touches, kin):
        calls.append((options.joint_offsets, {id(s) for s in samples_in}))
        return real(samples_in, options, tables, touches, kin)

    monkeypatch.setattr(solver, "_run_fit", spy)
    validation.cross_validate(res, solver.SolverOptions(), w4.touches, w4.kin, k=k)
    assert len(calls) == 2 * k                                              # one pipeline run per fold and variant
    all_ids = {id(s) for s in res.input_samples}
    seen_bad = 0
    for i, (offs, ids) in enumerate(calls):
        f = (i % k) if not offs else (i - k) % k                               # variants run in order, folds 0..k-1
        assert not (ids & held[f]), "held-out samples reached the fold pipeline"
        assert ids | held[f] == all_ids                                       # raw samples: nothing pre-filtered
        seen_bad += id(bad) in ids
    assert seen_bad >= 2 * (k - 1)                                            # the outlier is in every other fold's raw input


# ---- observability (priors excluded) -----------------------------------------------------------------------------------
@needs_urdf
def test_observability_uses_data_only_jacobian_and_flags_prior_held_directions(w4):
    res0 = w4.solve(opt=dict(joint_offsets=True), tables=[], touches=[])      # joint-6 offset vs target mount: no touches
    ob = res0.report["observability"]
    assert ob["n_prior_rows_dropped"] == 10 and ob["n_parameters"] == len(ob["normalised_singular_values"])
    assert ob["smallest"] < 1e-6 < ob["threshold"]
    weak = [x for x in ob["weak_directions"] if x["below_threshold"]]
    assert weak and any(p["name"].endswith("joint6") for x in weak for p in x["top_parameters"])
    assert any(x["structural"] for x in weak)
    assert any("not determined by the data alone" in x for x in res0.report["warnings"])
    # the prior makes the covariance finite although the data do not determine the direction
    assert np.isfinite(res0.report["joint"]["joint_offsets"]["std_deg"]["left"][5])
    # a well-posed dataset without offsets: no observability warning
    res = w4.solve()
    assert res.report["observability"]["smallest"] > res.report["observability"]["threshold"]
    assert not any("not determined by the data alone" in x for x in res.report["warnings"])
    assert "Observability" in solver.format_report(res0.report)
    # a degenerate dataset (translation only) has exactly-zero singular values
    from rebel_dual_arm_calibration.synthetic import make_scene, simulate_samples
    from test_solver import charuco_specs
    scene = make_scene(distance=0.9, seed=3)
    deg = solver.solve(simulate_samples(scene, charuco_specs(4, 0.025, 0.018), n_per_arm=15, noise_px=0.3, seed=9,
                                        rotation_diversity=False))
    assert deg.report["observability"]["smallest"] < 1e-6
