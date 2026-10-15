"""Metrology: held-out check points (TRE), noise-aware agreement, gate VALUES (no 'it ran' passes), repeatability,
comparison of calibrations.  Pure numpy tests on synthetic touches (no solver, fast)."""
import copy
import math

import numpy as np
import pytest

from rebel_dual_arm_calibration import metrology as M
from rebel_dual_arm_calibration.base_check import GateOptions, evaluate_gate, geometry_adequate, touch_only_base
from rebel_dual_arm_calibration.solver import Touch
from rebel_dual_arm_calibration.transforms import inv_T, rotation_angle_deg, rpy_to_R, xyz_rpy_to_T

T_R = xyz_rpy_to_T([0.4504, 0.012, -0.004], [0.004, -0.003, np.radians(2.0)])
CENTRE = np.array([0.225, 0.30, 0.0])
BOARD = {1: (-0.06, -0.04), 2: (0.06, -0.04), 3: (0.06, 0.04), 4: (-0.06, 0.04)}      # board frame == world offsets
FIT = {101: (-0.14, -0.14, 0.0), 102: (0.14, -0.14, 0.0), 103: (0.14, 0.14, 0.0), 104: (-0.14, 0.14, 0.0)}
CHECK = {111: (0.10, -0.03, 0.040), 112: (-0.19, 0.04, 0.0)}


def _rot(rng):
    return rpy_to_R(rng.uniform(-0.6, 0.6, 3))


def make_touches(ids_xyz, kind, repeats=3, noise=2e-4, seed=0, right_shift=None, T_base=T_R, same_orientation=False):
    """Both arms touch the world points; TCP positions in each arm's own base frame."""
    rng = np.random.default_rng(seed)
    out = []
    R0 = _rot(rng)
    for pid, xyz in ids_xyz.items():
        p_w = CENTRE + np.array(xyz, float)
        for _ in range(repeats):
            for arm in ("left", "right"):
                p = p_w.copy()
                if arm == "right" and right_shift is not None and pid in right_shift:
                    p = p + np.asarray(right_shift[pid], float)
                p_b = p if arm == "left" else inv_T(T_base)[:3, :3] @ p + inv_T(T_base)[:3, 3]
                T = np.eye(4)
                T[:3, :3] = R0 if same_orientation else _rot(rng)
                T[:3, 3] = p_b + rng.normal(0, noise, 3)
                out.append(Touch(arm=arm, point_id=pid, p_board=np.zeros(3), T_base_tcp=T, kind=kind))
    return out


def board_touches(**kw):
    return make_touches({k: (x, y, 0.0) for k, (x, y) in BOARD.items()}, "board", **kw)


# ---- check points ------------------------------------------------------------------------------------------------
def test_check_point_error_is_arm_to_arm_disagreement_and_independent_of_fits():
    ts = board_touches() + make_touches(FIT, "fit", seed=1) + make_touches(CHECK, "check", seed=2,
                                                                          right_shift={111: [0.004, 0.0, 0.0]})
    cp = M.check_point_report(ts, T_R)
    assert cp["available"] and cp["n_points"] == 2
    by = {p["point"]: p for p in cp["points"]}
    assert 3.6 < by[111]["error_mm"] < 4.4 and by[112]["error_mm"] < 0.8          # the displaced point is seen
    assert cp["max_mm"] == pytest.approx(by[111]["error_mm"]) and cp["rms_mm"] < cp["max_mm"]
    # geometry measured from the touched positions: 40 mm block, 190 mm out
    assert 35 < by[111]["height_above_plane_mm"] < 45 and by[112]["dist_from_centre_mm"] > 180
    assert cp["recommended_padding_mm"] == pytest.approx(max(20.0, 2 * cp["max_mm"] + 10.0))
    assert "TRE" in cp["label"] and "check points only" in cp["scope"]
    # the written base matters: a wrong base shows up at EVERY point
    bad = M.check_point_report(ts, T_R @ xyz_rpy_to_T([0.002, 0, 0], [0, 0, 0]))
    assert all(p["error_mm"] > 1.0 for p in bad["points"])
    # a point touched by one arm only is not evaluated
    cp1 = M.check_point_report([t for t in ts if not (t.point_id == 112 and t.arm == "right")], T_R)
    assert cp1["n_points"] == 1 and cp1["incomplete"][0]["point"] == 112
    # no check data: only the 20 mm minimum is known
    none = M.check_point_report(board_touches(), T_R)
    assert not none["available"] and none["recommended_padding_mm"] == 20.0
    assert M.recommended_padding_mm(0.0) == 20.0 and M.recommended_padding_mm(5.0) == 20.0
    assert M.recommended_padding_mm(8.0) == 26.0


def test_check_points_are_never_used_by_the_touch_only_base():
    """A wildly wrong check sticker must not change the touch-only base: check points are excluded from the Kabsch."""
    ts = board_touches() + make_touches(FIT, "fit", seed=1)
    base = touch_only_base(ts, T_R)
    chk = make_touches(CHECK, "check", seed=2, right_shift={111: [0.05, 0.05, 0.0], 112: [-0.05, 0.0, 0.0]})
    with_check = touch_only_base(ts + chk, T_R)
    assert np.allclose(base["T"], with_check["T"], atol=1e-12) and with_check["n_common"] == base["n_common"] == 8
    assert 111 not in with_check["points"] and 112 not in with_check["points"]
    # the same ids declared as fit stickers DO change it
    as_fit = [copy.copy(t) for t in chk]
    for t in as_fit:
        t.kind = "fit"
    assert not np.allclose(touch_only_base(ts + as_fit, T_R)["T"], base["T"], atol=1e-6)


# ---- noise prediction ---------------------------------------------------------------------------------------------
def test_predicted_base_noise_matches_the_empirical_spread_and_shrinks_with_spread():
    rng = np.random.default_rng(3)

    def empirical(pts, sig_mm, n_rep, trials=300):
        P = np.array([inv_T(T_R)[:3, :3] @ (CENTRE + p) + inv_T(T_R)[:3, 3] for p in pts])
        Q = np.array([CENTRE + p for p in pts])
        sig = sig_mm / 1e3 / math.sqrt(n_rep)
        from rebel_dual_arm_calibration.transforms import kabsch
        yaws = []
        for _ in range(trials):
            T = kabsch(P + rng.normal(0, sig, P.shape), Q + rng.normal(0, sig, Q.shape))
            yaws.append(np.degrees(np.arctan2(T[1, 0], T[0, 0])))
        pred = M.predict_base_noise(P, Q, [n_rep] * len(P), [n_rep] * len(P), sig_mm, sig_mm,
                                    np.array([-0.2, 0.3, 0.0]), n_mc=600)
        return float(np.std(yaws)), pred["yaw_sigma_deg"], pred

    small = [np.array([x, y, 0.0]) for x, y in BOARD.values()]
    wide = small + [np.array(v) for v in FIT.values()]
    e_s, p_s, pred_s = empirical(small, 0.4, 3)
    e_w, p_w, pred_w = empirical(wide, 0.4, 3)
    assert abs(p_s - e_s) < 0.25 * e_s and abs(p_w - e_w) < 0.25 * e_w
    assert p_w < 0.55 * p_s                                      # wider spread = better geometry
    assert pred_w["reach_sigma_mm"] < pred_s["reach_sigma_mm"] and pred_w["n_mc"] == 600
    # deterministic
    again = M.predict_base_noise(np.zeros((4, 3)) + np.eye(4, 3), np.zeros((4, 3)) + np.eye(4, 3), [1] * 4, [1] * 4, 0.3, 0.3,
                                 np.array([0.4, 0, 0]))
    assert again == M.predict_base_noise(np.zeros((4, 3)) + np.eye(4, 3), np.zeros((4, 3)) + np.eye(4, 3), [1] * 4, [1] * 4,
                                         0.3, 0.3, np.array([0.4, 0, 0]))


def test_repeat_scatter_is_measured_per_arm_and_assumed_without_repeats():
    ts = make_touches({k: (x, y, 0.0) for k, (x, y) in BOARD.items()}, "board", repeats=5, noise=3e-4, seed=4)
    sc = M.repeat_scatter(ts)
    for arm in ("left", "right"):
        assert not sc[arm]["assumed"] and 0.2 < sc[arm]["sigma_axis_mm"] < 0.45      # 0.3 mm per axis injected
    one = M.repeat_scatter(make_touches({1: (0, 0, 0)}, "board", repeats=1))
    assert one["left"]["assumed"] and one["left"]["sigma_axis_mm"] == 0.5


def test_default_geometry_amplification_warns_and_the_wide_fit_geometry_is_adequate():
    """Dead branch fixed: the 4 board crosshairs alone always amplified noise > 2x (so 'prefer the touch-only base' was
    unreachable); with fit stickers the geometry is adequate and the recommendation is reachable."""
    small = touch_only_base(board_touches(repeats=3, noise=5e-4), T_R)
    assert any("spread only" in w for w in small["warnings"]) and small["geometry"]["reach_amplification"] > 2.0
    wide = touch_only_base(board_touches(repeats=3, noise=5e-4) + make_touches(FIT, "fit", repeats=3, noise=5e-4, seed=1), T_R)
    assert not wide["warnings"] and wide["geometry"]["reach_amplification"] < 1.5 and wide["geometry"]["n_fit_points"] == 8
    ok, txt = geometry_adequate(wide, GateOptions())
    assert ok, txt
    ok_s, txt_s = geometry_adequate(small, GateOptions())
    assert not ok_s and "touch spread insufficient" in txt_s and "add fit stickers" in txt_s


# ---- gate: values, never 'it ran' ---------------------------------------------------------------------------------
def good_report(shift=None, repeats=4, fit=True):
    ts = board_touches(repeats=repeats) + (make_touches(FIT, "fit", repeats=repeats, seed=1) if fit else []) \
        + make_touches(CHECK, "check", repeats=repeats, seed=2, right_shift=shift)
    tb = touch_only_base(ts, T_R)
    from rebel_dual_arm_calibration.base_check import tcp_consistency
    tb["tcp_consistency"] = tcp_consistency(ts)
    return {"touch_base": tb, "check_points": M.check_point_report(ts, T_R),
            "loop_closure": {"out_of_fit_rms_mm": 0.5, "out_of_fit_source": "camera_only", "final": {"rms_mm": {"all": 0.4}}},
            "validation": {"without_offsets": {"heldout_rms_px": 0.45, "heldout_touch_rms_nominal_mm": 0.5,
                                               "heldout_touch_rms_mm": 0.5}, "with_offsets": None},
            "joint": {"arm_right_base": {"std_xyz_mm": [0.1, 0.1, 0.1], "std_rpy_deg": [0.01, 0.01, 0.01]},
                      "intrinsics": {"pull_f_frac": 0.001, "pull_c_px": 1.0}},
            "written_base": {"fit": "nominal_fit"}, "options": {"intrinsics_source": "camera_info"}}


def names_failed(g):
    return {c["name"] for c in g["checks"] if not c["ok"]}


def test_gate_passes_on_good_data_and_every_check_is_a_value():
    g = evaluate_gate(good_report(), GateOptions())
    assert g["passed"], "\n".join(g["text"])
    assert {c["name"] for c in g["checks"]} == {"common_touch_points", "touch_mismatch_rms", "check_point_coverage",
                                                "check_point_error", "touch_geometry", "joint_vs_touch",
                                                "heldout_validation", "intrinsics_validated"}
    assert "accuracy demonstrated at the check points only" in "\n".join(g["text"])
    assert "recommended_padding_mm" in "\n".join(g["text"]) and g["recommended_padding_mm"] >= 20.0
    assert "HONEST" not in "\n".join(g["text"])


def test_gate_fails_for_displaced_check_point_and_scales_padding():
    rep = good_report(shift={111: [0.008, 0.0, 0.0]})
    g = evaluate_gate(rep, GateOptions())
    assert not g["passed"] and "check_point_error" in names_failed(g)
    assert rep["check_points"]["max_mm"] > 7.5 and g["recommended_padding_mm"] == pytest.approx(2 * rep["check_points"]["max_mm"] + 10)
    # a 5 mm displacement already fails the 3 mm limit, but the padding stays at the 20 mm minimum (2 x 5 + 10 = 20)
    g5 = evaluate_gate(good_report(shift={111: [0.005, 0.0, 0.0]}), GateOptions())
    assert "check_point_error" in names_failed(g5) and g5["recommended_padding_mm"] == pytest.approx(20.0, abs=1.0)
    assert g["source"] == "calibrated_unverified"


def test_gate_fails_without_check_points_or_with_only_one_or_wrong_geometry():
    rep = good_report()
    rep["check_points"] = M.check_point_report(board_touches(), T_R)           # nothing held out
    g = evaluate_gate(rep, GateOptions())
    assert {"check_point_coverage", "check_point_error"} <= names_failed(g)
    # two check points but both flat and near the centre: height and distance requirements fail separately
    ts = board_touches(repeats=3) + make_touches({111: (0.03, 0.0, 0.0), 112: (-0.04, 0.02, 0.0)}, "check", repeats=3, seed=2)
    rep["check_points"] = M.check_point_report(ts, T_R)
    g = evaluate_gate(rep, GateOptions())
    cov = next(c for c in g["checks"] if c["name"] == "check_point_coverage")
    assert not cov["ok"] and cov["value"]["max_height_mm"] < 30 and cov["value"]["max_dist_mm"] < 150
    # only the block (height ok, distance not) still fails; adding the far one passes
    ts2 = board_touches(repeats=3) + make_touches({111: (0.03, 0.0, 0.04), 112: (-0.04, 0.02, 0.0)}, "check", repeats=3, seed=2)
    rep["check_points"] = M.check_point_report(ts2, T_R)
    assert not next(c for c in evaluate_gate(rep, GateOptions())["checks"] if c["name"] == "check_point_coverage")["ok"]
    rep["check_points"] = good_report()["check_points"]
    assert next(c for c in evaluate_gate(rep, GateOptions())["checks"] if c["name"] == "check_point_coverage")["ok"]


def test_gate_fails_when_touch_spread_is_insufficient_and_says_how_to_fix_it():
    rep = good_report(fit=False)                   # only the 4 board crosshairs (+- 60 x 40 mm), 0.3 mm scatter: adequate
    assert "touch_geometry" not in names_failed(evaluate_gate(rep, GateOptions()))
    # noisier touches on the small board: the geometry check fails with the instruction
    ts = board_touches(repeats=4, noise=8e-4) + make_touches(CHECK, "check", repeats=4, seed=2)
    rep["touch_base"] = touch_only_base(ts, T_R)
    g = evaluate_gate(rep, GateOptions())
    assert "touch_geometry" in names_failed(g)
    assert "touch spread insufficient" in next(c for c in g["checks"] if c["name"] == "touch_geometry")["text"]


def test_in_fit_only_touch_numbers_can_never_satisfy_the_gate():
    rep = good_report()
    rep["loop_closure"] = {"out_of_fit_rms_mm": None, "out_of_fit_source": "none (only in-fit data)",
                           "final": {"rms_mm": {"all": 0.01}}}
    g = evaluate_gate(rep, GateOptions())
    c = next(c for c in g["checks"] if c["name"] == "touch_mismatch_rms")
    assert not c["ok"] and "in-fit" in c["text"] and not g["passed"]
    # the same without any touch data at all
    rep["loop_closure"] = None
    assert not evaluate_gate(rep, GateOptions())["passed"]


def test_validation_that_merely_ran_does_not_pass_values_do():
    rep = good_report()
    rep["validation"]["without_offsets"]["heldout_rms_px"] = 1.6                      # ran, but the held-out error is bad
    g = evaluate_gate(rep, GateOptions())
    assert "heldout_validation" in names_failed(g) and "1.600 px" in next(c for c in g["checks"] if c["name"] == "heldout_validation")["text"]
    rep = good_report()
    rep["validation"]["without_offsets"]["heldout_touch_rms_nominal_mm"] = 4.0        # held-out touches bad
    assert "heldout_validation" in names_failed(evaluate_gate(rep, GateOptions()))
    # the variant that is written decides
    rep = good_report()
    rep["validation"]["with_offsets"] = {"heldout_rms_px": 2.0, "heldout_touch_rms_nominal_mm": 0.5}
    assert evaluate_gate(rep, GateOptions())["passed"]
    rep["written_base"] = {"fit": "offset_fit"}
    assert "heldout_validation" in names_failed(evaluate_gate(rep, GateOptions()))
    # a threshold is configurable
    assert evaluate_gate(good_report(), GateOptions(max_heldout_px=0.2))["passed"] is False
    rep = good_report()
    rep.pop("validation")
    assert "heldout_validation" in names_failed(evaluate_gate(rep, GateOptions()))


def test_agreement_is_noise_aware_floor_and_three_sigma():
    rep = good_report()
    c = next(c for c in evaluate_gate(rep, GateOptions())["checks"] if c["name"] == "joint_vs_touch")
    assert c["ok"] and c["value"]["limit_origin_mm"] >= 1.0 and all(l >= 0.1 for l in c["value"]["limit_rotation_deg"])
    # a written base that is 2 mm / 0.5 deg away from the touch-only one fails
    tb = rep["touch_base"]
    T_off = np.array(tb["T"]) @ xyz_rpy_to_T([0.002, 0, 0], [0, 0, np.radians(0.5)])
    from rebel_dual_arm_calibration.base_check import _diff
    tb["comparison"] = _diff(T_off, np.array(tb["T"]), np.array(tb["reach_point_in_right_base"]))
    assert "joint_vs_touch" in names_failed(evaluate_gate(rep, GateOptions()))
    # the limit grows with the predicted noise (3 sigma) but never below the floor
    noisy = good_report()
    noisy["touch_base"]["noise"]["origin_sigma_mm"] = 1.2
    noisy["touch_base"]["noise"]["rotation_sigma_deg"] = [0.2, 0.2, 0.2]
    c2 = next(c for c in evaluate_gate(noisy, GateOptions())["checks"] if c["name"] == "joint_vs_touch")
    assert c2["value"]["limit_origin_mm"] >= 3.0 * 1.2 and c2["value"]["limit_rotation_deg"][2] >= 0.6


def test_intrinsics_check_file_or_small_pull_else_fail_with_instruction():
    rep = good_report()
    rep["joint"]["intrinsics"] = {"pull_f_frac": 0.006, "pull_c_px": 1.0}
    g = evaluate_gate(rep, GateOptions())
    c = next(c for c in g["checks"] if c["name"] == "intrinsics_validated")
    assert not c["ok"] and "calibrate the camera intrinsics" in c["text"]
    rep["joint"]["intrinsics"] = {"pull_f_frac": 0.001, "pull_c_px": 4.5}
    assert "intrinsics_validated" in names_failed(evaluate_gate(rep, GateOptions()))
    rep["joint"]["intrinsics"] = {"pull_f_frac": 0.01, "pull_c_px": 9.0}
    rep["options"] = {"intrinsics_source": "file"}                         # supplied file: validated
    assert evaluate_gate(rep, GateOptions())["passed"]
    rep["options"] = {"intrinsics_source": "camera_info"}
    rep["joint"]["intrinsics"] = {"error": "x"}                            # pull not computed: cannot pass
    assert "intrinsics_validated" in names_failed(evaluate_gate(rep, GateOptions()))


def test_touch_only_preferred_branch_is_reachable_with_adequate_geometry():
    rep = good_report()
    tb = rep["touch_base"]
    T_off = np.array(tb["T"]) @ xyz_rpy_to_T([0.02, 0, 0], [0, 0, 0])             # camera chain 20 mm away
    from rebel_dual_arm_calibration.base_check import _diff
    tb["comparison"] = _diff(T_off, np.array(tb["T"]), np.array(tb["reach_point_in_right_base"]))
    g = evaluate_gate(rep, GateOptions())
    txt = "\n".join(g["text"])
    assert "joint_vs_touch" in names_failed(g) and "the camera chain is the suspect" in txt and "--base-from touches" in txt


# ---- repeatability and comparison ----------------------------------------------------------------------------------
def test_repeatability_iso9283_inspired():
    ts = make_touches({1: (0, 0, 0)}, "board", repeats=8, noise=3e-4, seed=5, same_orientation=True)
    rp = M.repeatability(ts)
    assert len(rp["clusters"]) == 2 and {c["arm"] for c in rp["clusters"]} == {"left", "right"}
    for c in rp["clusters"]:
        P = np.array([t.T_base_tcp[:3, 3] for t in ts if t.arm == c["arm"]])
        l = np.linalg.norm(P - P.mean(axis=0), axis=1)
        assert c["n"] == 8 and c["rp_mm"] == pytest.approx((l.mean() + 3 * l.std(ddof=1)) * 1e3, rel=1e-9)
        assert 0.3 < c["rp_mm"] < 2.5
    # different wrist orientations (> 5 deg) are not repeats of each other; fewer than 5 repeats are skipped
    varied = make_touches({1: (0, 0, 0)}, "board", repeats=8, noise=3e-4, seed=5, same_orientation=False)
    assert not M.repeatability(varied)["clusters"]
    few = M.repeatability(make_touches({1: (0, 0, 0)}, "board", repeats=4, seed=5, same_orientation=True))
    assert not few["clusters"] and few["skipped"]
    assert "ISO 9283-inspired" in rp["note"]


def test_compare_calibrations_and_yaml_loader(tmp_path):
    from rebel_dual_arm_calibration import solver

    class R:
        T_world_arm_right_base = T_R
        T_world_camera_link = xyz_rpy_to_T([0.2, 0.9, 0.8], [0, 0.8, -1.5])
        report = {"date": "2026-10-01"}
    p = tmp_path / "prev.yaml"
    solver.write_calibration_yaml(str(p), R(), "n")
    prev = M.load_calibration_yaml(str(p))
    same = M.compare_calibrations(R.T_world_arm_right_base, R.T_world_camera_link, prev)
    assert same["arm_right_base"]["position_mm"] < 1e-9 and same["within_guidance"]
    moved = M.compare_calibrations(T_R @ xyz_rpy_to_T([0.003, 0, 0], [0, 0, np.radians(0.3)]), R.T_world_camera_link, prev)
    assert moved["arm_right_base"]["position_mm"] > 2.9 and moved["arm_right_base"]["rotation_deg"] > 0.29
    assert not moved["within_guidance"] and "EXCEEDS" in moved["text"] and "2.0 mm / 0.2 deg" in moved["text"]


def test_touch_kind_round_trip_defaults_to_board_and_validates():
    t = Touch(arm="left", point_id=111, p_board=np.zeros(3), T_base_tcp=np.eye(4), kind="check")
    d = t.to_json_dict()
    assert d["kind"] == "check" and Touch.from_json_dict(d).kind == "check"
    old = dict(d)
    old.pop("kind")                                   # a session recorded before the stickers existed
    assert Touch.from_json_dict(old).kind == "board"
    with pytest.raises(ValueError, match="board\\|fit\\|check"):
        Touch(arm="left", point_id=1, p_board=np.zeros(3), T_base_tcp=np.eye(4), kind="nonsense")


def test_sticker_config_validation():
    from rebel_dual_arm_calibration import targets as T
    arms, table = T.load_targets_all(T.default_targets_path())[0], T.load_targets_all(T.default_targets_path())[1]
    cfg = {"touch_stickers": {"fit_points": [101, 102], "check_points": [111]}}
    st = T.stickers_from_config(cfg, table)
    assert st.kind(101) == "fit" and st.kind(111) == "check" and st.kind(5) == "" and st.all_ids() == [101, 102, 111]
    for bad, msg in (({"fit_points": [101], "check_points": [101]}, "unique and disjoint"),
                     ({"fit_points": [2]}, "collide"), ({"fit_points": [0]}, "positive")):
        with pytest.raises(ValueError, match=msg):
            T.stickers_from_config({"touch_stickers": bad}, table)
    assert T.stickers_from_config({}, table).all_ids() == []
