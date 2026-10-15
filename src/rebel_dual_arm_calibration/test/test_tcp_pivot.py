"""Pivot-calibration solver: synthetic (seeded) tests plus a read-only look at real data."""
import csv
import json
import math
import os

import numpy as np
import pytest

from rebel_dual_arm_calibration import tcp_pivot as tp

REAL_XACRO = "/home/farhan/igus_rebel_ros2/src/igus_rebel_description/urdf/igus_rebel_xeg32_wrapper.urdf.xacro"
REAL_CSV = "/home/farhan/igus_rebel_ros2/scan_results/tcp_calibration_20260922/tcp_new_smc_v2.csv"

R_TRUE = np.array([0.0062, 0.0925, 0.1318])
C_TRUE = np.array([0.42, -0.10, 0.05])


def _R_down(r):
    """Rotation whose tool axis r/|r| points along base -z (knife vertical)."""
    a = r / np.linalg.norm(r)
    down = np.array([0.0, 0.0, -1.0])
    v = np.cross(a, down)
    ang = math.atan2(np.linalg.norm(v), a @ down)
    return tp._axis_angle(v, ang)


def _sample(R, r=R_TRUE, c=C_TRUE, noise=0.0, rng=None, role="fit", arm="arm_left_"):
    p = c - R @ r
    if noise and rng is not None:
        p = p + rng.normal(0.0, noise, 3)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return tp.PivotSample(arm=arm, T_base_tool0=T, role=role)


def _plan_rotations(r=R_TRUE):
    """Body-frame rotations for every plan item, applied to the vertical orientation."""
    R0 = _R_down(r)
    a = r / np.linalg.norm(r)
    out = {}
    for item in tp.POSE_PLAN:
        out[item["id"]] = R0 @ tp.plan_rotation(item, a)
    return out


def _plan_samples(noise=0.0, seed=0, r=R_TRUE, c=C_TRUE, roles=True):
    rng = np.random.default_rng(seed)
    out = []
    for item in tp.POSE_PLAN:
        R = _plan_rotations(r)[item["id"]]
        out.append(_sample(R, r, c, noise, rng, role=item["role"] if roles else "fit"))
    return out


# ------------------------------------------------------------------ recovery

def test_exact_recovery_and_good():
    res = tp.solve_pivot(_plan_samples(), current_r=tp.CURRENT_URDF_R_M)
    assert res.verdict == "GOOD" and res.ok, res.reasons
    assert np.allclose(res.r_tool0, R_TRUE, atol=1e-9)
    assert np.allclose(res.c_base, C_TRUE, atol=1e-9)
    assert res.rms_mm < 1e-6 and res.max_mm < 1e-6
    assert res.rank == 3 and res.obs_min >= 0.2
    assert res.n_fit == 11 and res.n_validate == 4
    assert res.outliers == [] and len(res.heldout_mm) == 4 and res.heldout_rms_mm < 1e-6
    assert np.allclose(res.delta_to_current_mm, (R_TRUE - tp.CURRENT_URDF_R_M) * 1000, atol=1e-6)
    assert len(res.tilt_deg) == 11 and max(res.tilt_deg) > 15 and len(res.roll_deg) == 11
    assert np.isclose(np.linalg.norm(res.weak_axis_tool0), 1.0)


def test_validate_samples_never_enter_fit():
    s = _plan_samples()
    # corrupt only validate samples: fit must be unchanged
    bad = [tp.PivotSample(arm=x.arm, T_base_tool0=x.T_base_tool0.copy(), role=x.role) for x in s]
    for x in bad:
        if x.role == "validate":
            x.T_base_tool0[:3, 3] += np.array([0.01, 0.0, 0.0])
    a, b = tp.solve_pivot(s), tp.solve_pivot(bad)
    assert np.allclose(a.r_tool0, b.r_tool0) and np.allclose(a.c_base, b.c_base)
    assert b.heldout_rms_mm > 5.0 and b.verdict == "FAIL"
    assert any("held-out RMS" in r for r in b.reasons)


def test_sigma_estimates_consistent_with_monte_carlo():
    sigma_n = 0.0008
    errs, preds = [], []
    base = tp.solve_pivot(_plan_samples(noise=sigma_n, seed=1))
    for k in range(300):
        res = tp.solve_pivot(_plan_samples(noise=sigma_n, seed=100 + k))
        errs.append(res.r_tool0 - R_TRUE)
        preds.append(res.sigma_axes_mm)
    errs = np.array(errs) * 1000.0
    # principal directions of the (fixed-geometry) covariance
    w, V = np.linalg.eigh(base.cov_r)
    V = V[:, np.argsort(w)[::-1]]
    emp = np.std(errs @ V, axis=0)
    pred = np.mean(preds, axis=0)
    assert np.all(emp / pred > 0.75) and np.all(emp / pred < 1.3), (emp, pred)
    assert np.all(np.diff(pred) <= 1e-9)           # descending
    # LS with 11 poses: noise estimate should be near the truth (3 mm-scale total)
    assert 0.5 * sigma_n * 1000 * math.sqrt(3) < base.rms_mm < 2.0 * sigma_n * 1000 * math.sqrt(3)


# ------------------------------------------------------------------ geometry refusal

def test_single_axis_rotation_refused_with_weak_axis():
    R0 = _R_down(R_TRUE)
    axis = np.array([0.3, 0.5, 0.81])
    axis /= np.linalg.norm(axis)
    samples = [_sample(R0 @ tp._axis_angle(axis, math.radians(d)))
               for d in (-40, -30, -20, -10, 0, 10, 20, 30, 40)]
    res = tp.solve_pivot(samples)
    assert res.verdict == "INSUFFICIENT" and not res.ok
    assert res.rank < 3
    assert abs(abs(res.weak_axis_tool0 @ axis) - 1.0) < 1e-6
    assert any("single axis" in r and "tool" in r for r in res.reasons)
    assert np.all(np.isnan(res.r_tool0))


def test_small_tilts_fail_observability():
    R0 = _R_down(R_TRUE)
    rng = np.random.default_rng(3)
    samples = []
    for _ in range(10):
        rv = np.degrees(rng.normal(0, 0.03, 3))          # ~1.7 deg tilts
        samples.append(_sample(R0 @ tp._rotvec_to_R(rv), noise=0.0002, rng=rng))
    for _ in range(3):
        samples.append(_sample(R0 @ tp._rotvec_to_R(np.degrees(rng.normal(0, 0.03, 3))), role="validate"))
    res = tp.solve_pivot(samples)
    assert res.verdict == "INSUFFICIENT" and res.obs_min < 0.2
    assert any("orientation spread" in r for r in res.reasons)


def test_too_few_fit_poses():
    s = [x for x in _plan_samples() if x.role == "fit"][:5]
    res = tp.solve_pivot(s)
    assert res.verdict == "INSUFFICIENT"
    assert any("at least 8" in r for r in res.reasons)
    assert tp.solve_pivot([]).verdict == "INSUFFICIENT"


def test_mixed_arms_and_bad_rotation_rejected():
    s = _plan_samples()
    s[0].arm = "arm_right_"
    with pytest.raises(ValueError):
        tp.solve_pivot(s)
    t = _plan_samples()
    t[2].T_base_tool0[:3, :3] *= 1.1
    with pytest.raises(ValueError):
        tp.solve_pivot(t)


# ------------------------------------------------------------------ duplicates / outliers

def test_duplicate_detection():
    R0 = _R_down(R_TRUE)
    T = [np.eye(4) for _ in range(3)]
    T[0][:3, :3] = R0
    T[1][:3, :3] = R0 @ tp._rotvec_to_R([30, 0, 0])
    T[2][:3, :3] = R0 @ tp._rotvec_to_R([0, 30, 0])
    new = np.eye(4)
    new[:3, :3] = R0 @ tp._rotvec_to_R([31, 3, 0])             # ~3 deg from T[1]
    assert tp.duplicate_of(new, T, 15.0) == 1
    far = np.eye(4)
    far[:3, :3] = R0 @ tp._rotvec_to_R([0, -30, 0])
    assert tp.duplicate_of(far, T, 15.0) is None
    assert tp.duplicate_of(new, [], 15.0) is None
    assert tp.rotation_distance_deg(T[0], T[1]) == pytest.approx(30.0, abs=1e-6)
    assert tp.rotation_distance_deg(T[0], T[0]) == pytest.approx(0.0, abs=1e-9)


def test_duplicate_pose_warns_and_blocks_good():
    s = _plan_samples()
    dup = tp.PivotSample(arm=s[1].arm, T_base_tool0=s[1].T_base_tool0.copy(), role="fit")
    res = tp.solve_pivot(s + [dup])
    assert any("near-duplicate" in r for r in res.reasons)
    assert res.verdict == "PASS"                      # exact data, but duplicates cap it below GOOD


def test_loo_flags_slipped_pose():
    rng = np.random.default_rng(7)
    s = _plan_samples(noise=0.0003, seed=7)
    k = 4
    s[k].T_base_tool0[:3, 3] += np.array([0.0, 0.0, 0.004])       # 4 mm slip
    res = tp.solve_pivot(s)
    assert k in res.outliers, (res.loo_mm, res.outliers)
    assert res.loo_mm[k] == max(res.loo_mm)
    assert any(f"pose #{k + 1} (fit)" in r and "outlier" in r for r in res.reasons)
    assert len(res.loo_mm) == res.n_fit
    # flagged, never dropped
    assert res.n_fit == 11 and len(res.residuals_mm) == 11
    _ = rng
    assert not any("index" in r or "fit pose" in r for r in res.reasons)        # one numbering: "#N"


def test_reasons_number_poses_by_position_in_the_sample_list():
    s = _plan_samples(noise=0.0003, seed=7)
    # put two validate poses first: "#N" must still be the 1-based position in the list passed in
    vals = [x for x in s if x.role == "validate"][:2]
    fits = [x for x in s if x.role == "fit"]
    rest = [x for x in s if x.role == "validate"][2:]
    mixed = vals + fits + rest
    k_fit = 4
    fits[k_fit].T_base_tool0[:3, 3] += np.array([0.0, 0.0, 0.004])
    res = tp.solve_pivot(mixed)
    pos = next(i for i, x in enumerate(mixed) if x is fits[k_fit]) + 1
    assert pos == k_fit + 3
    assert any(f"pose #{pos} (fit) is an outlier" in r for r in res.reasons), res.reasons
    assert res.outliers == [k_fit]                                              # (still an index into the fit list)


def test_heldout_uses_fixed_r_c():
    s = _plan_samples(noise=0.0003, seed=11)
    res = tp.solve_pivot(s)
    vals = [x for x in s if x.role == "validate"]
    manual = [np.linalg.norm(v.T_base_tool0[:3, 3] + v.T_base_tool0[:3, :3] @ res.r_tool0 - res.c_base) * 1000
              for v in vals]
    assert np.allclose(res.heldout_mm, manual)
    assert res.heldout_rms_mm == pytest.approx(math.sqrt(np.mean(np.square(manual))))


# ------------------------------------------------------------------ verdict tiers

def test_verdict_tiers():
    # GOOD: tiny noise
    assert tp.solve_pivot(_plan_samples(noise=0.0002, seed=5)).verdict == "GOOD"
    # PASS: RMS between 1 and 2 mm (isotropic noise 0.9 mm/axis -> rms ~ 1.2-1.6 mm with 11 poses)
    res = tp.solve_pivot(_plan_samples(noise=0.0009, seed=5))
    assert res.verdict in ("PASS", "GOOD")
    mid = tp.solve_pivot(_plan_samples(noise=0.0012, seed=6))
    assert mid.verdict == "PASS", (mid.rms_mm, mid.max_mm, mid.reasons)
    assert mid.ok and any("not GOOD" in r for r in mid.reasons)
    # FAIL: large noise
    bad = tp.solve_pivot(_plan_samples(noise=0.004, seed=5))
    assert bad.verdict == "FAIL" and not bad.ok
    assert any("RMS" in r and "re-seat" in r for r in bad.reasons)
    # INSUFFICIENT: no held-out poses but otherwise fine
    nov = tp.solve_pivot([x for x in _plan_samples() if x.role == "fit"])
    assert nov.verdict == "INSUFFICIENT" and not nov.ok
    assert any("held-out validation missing" in r for r in nov.reasons)
    assert nov.heldout_rms_mm is None
    # FAIL beats missing validation
    nov_bad = tp.solve_pivot([x for x in _plan_samples(noise=0.004, seed=5) if x.role == "fit"])
    assert nov_bad.verdict == "FAIL"
    assert any("held-out validation missing" in r for r in nov_bad.reasons)


def test_custom_options_change_thresholds():
    s = _plan_samples(noise=0.0012, seed=6)
    strict = tp.solve_pivot(s, tp.PivotOptions(rms_pass_mm=0.5))
    assert strict.verdict == "FAIL"
    fewer = tp.solve_pivot([x for x in s if x.role == "fit"], tp.PivotOptions(min_validate=0))
    assert fewer.verdict in ("PASS", "GOOD")


# ------------------------------------------------------------------ coverage / plan

def test_plan_has_no_internal_duplicates_between_validate_and_fit():
    rots = _plan_rotations()
    fit_ids = [i["id"] for i in tp.POSE_PLAN if i["role"] == "fit"]
    val_ids = [i["id"] for i in tp.POSE_PLAN if i["role"] == "validate"]
    for v in val_ids:
        for f in fit_ids:
            assert tp.rotation_distance_deg(rots[v], rots[f]) > 20.0, (v, f)
    for i, f in enumerate(fit_ids):
        for g in fit_ids[i + 1:]:
            assert tp.rotation_distance_deg(rots[f], rots[g]) > 20.0, (f, g)


def test_pose_plan_structure_and_coverage_complete():
    fit = [i for i in tp.POSE_PLAN if i["role"] == "fit"]
    val = [i for i in tp.POSE_PLAN if i["role"] == "validate"]
    assert len(fit) == 11 and len(val) == 4
    assert all(i["text"] and i["id"] for i in tp.POSE_PLAN)
    cov = tp.coverage(_plan_samples())
    assert cov["complete"] and cov["missing"] == [] and cov["next"] is None
    assert cov["reference_index"] == 0
    assert "complete" in cov["suggestion"]
    assert cov["max_tilt_deg"] > 25


def test_coverage_suggests_missing_pose():
    s = _plan_samples()
    keep = [x for i, x in enumerate(s) if tp.POSE_PLAN[i]["id"] != "y-30"]
    cov = tp.coverage(keep)
    assert cov["missing"] == ["y-30"] and cov["next"] == "y-30"
    assert "-30 deg about its y axis" in cov["suggestion"]
    empty = tp.coverage([])
    assert empty["next"] == "vertical" and not empty["complete"]
    no_vertical = tp.coverage(s[1:])
    assert no_vertical["reference_index"] is None and "vertical" in no_vertical["suggestion"]


# ------------------------------------------------------------------ xacro patch / mechanical

def test_xacro_patch_changes_exactly_one_line():
    with open(REAL_XACRO, encoding="utf-8") as fh:
        text = fh.read()
    from rebel_dual_arm_calibration.ui.tcp_ui import parse_xacro_r
    r_cur = parse_xacro_r(text)          # the live value (it changes with every applied TCP calibration)
    r_new = np.array([0.006412, 0.093877, 0.131204])
    diff = tp.xacro_patch(text, r_new)
    lines = diff.splitlines()
    minus = [l for l in lines if l.startswith("-") and not l.startswith("---")]
    plus = [l for l in lines if l.startswith("+") and not l.startswith("+++")]
    assert len(minus) == 1 and len(plus) == 1
    assert 'xyz="%s"' % " ".join(f"{v:.6f}" for v in r_cur) in minus[0]
    assert 'xyz="0.006412 0.093877 0.131204"' in plus[0]
    assert 'rpy="0 0 0"' in plus[0] and 'rpy="0 0 0"' in minus[0]
    new = tp.xacro_apply(text, r_new)
    assert len(new.splitlines()) == len(text.splitlines())
    assert sum(1 for a, b in zip(text.splitlines(), new.splitlines()) if a != b) == 1
    # only the TCP position joint changed (link6_to_igus_tool0 now lives in knife_c_tool.macro.xacro,
    # which the patcher never touches; every other wrapper line is unchanged, checked above)
    changed = [b for a, b in zip(text.splitlines(), new.splitlines()) if a != b]
    assert 'xyz="0.006412 0.093877 0.131204"' in changed[0]
    assert text.count('name="igus_tool0_to_xeg32_tool_tip"') == 1
    assert tp.xacro_patch(text, r_cur) == ""


def test_xacro_patch_errors():
    j = '<joint name="igus_tool0_to_xeg32_tool_tip" type="fixed"><origin xyz="1 2 3" rpy="0 0 0"/></joint>'
    with pytest.raises(ValueError):
        tp.xacro_patch("<robot></robot>", [0.1, 0.1, 0.1])
    with pytest.raises(ValueError):
        tp.xacro_patch("<robot>" + j + j + "</robot>", [0.1, 0.1, 0.1])
    with pytest.raises(ValueError):
        tp.xacro_patch("<robot>" + j + "</robot>", [np.nan, 0.1, 0.1])
    # commented-out copy is ignored, real one patched
    ok = tp.xacro_patch("<robot><!-- " + j + " -->\n" + j + "\n</robot>", [0.1, 0.2, 0.3])
    assert 'xyz="0.100000 0.200000 0.300000"' in ok
    with pytest.raises(ValueError):
        tp.xacro_patch('<robot><joint name="igus_tool0_to_xeg32_tool_tip"><origin rpy="0 0 0"/></joint></robot>',
                       [0.1, 0.2, 0.3])


def test_mechanical_check_numbers():
    cur = tp.mechanical_check(tp.CURRENT_URDF_R_M, 183.4)
    assert cur["predicted_current_mm"] == pytest.approx(183.399)
    assert cur["predicted_new_mm"] == pytest.approx(183.399)
    assert abs(cur["delta_new_mm"]) < 0.01 and cur["consistent_new"]
    new = tp.mechanical_check(np.array([0.006, 0.094, 0.1347]), 187.9)
    assert new["predicted_new_mm"] == pytest.approx(187.4)
    assert new["delta_new_mm"] == pytest.approx(0.5)
    assert new["delta_current_mm"] == pytest.approx(187.9 - 183.399)
    assert new["consistent_new"] and not new["consistent_current"]
    off = tp.mechanical_check(np.array([0.006, 0.094, 0.1347]), 183.4)
    assert not off["consistent_new"] and off["consistent_current"]
    assert "CURRENT" in off["text"]
    far = tp.mechanical_check(np.array([0.006, 0.094, 0.1347]), 200.0)
    assert not far["consistent_new"] and not far["consistent_current"] and "re-measure" in far["text"]


# ------------------------------------------------------------------ persistence

def test_json_round_trip():
    s = _plan_samples()[0]
    s.joints = {"joint1": 0.1, "joint2": -0.2}
    s.T_base_link6 = np.eye(4)
    s.stamp, s.note = 12.5, "hello"
    d = json.loads(json.dumps(s.to_json_dict()))
    s2 = tp.PivotSample.from_json_dict(d)
    assert s2.arm == s.arm and s2.role == s.role and s2.note == "hello" and s2.stamp == 12.5
    assert np.allclose(s2.T_base_tool0, s.T_base_tool0) and np.allclose(s2.T_base_link6, np.eye(4))
    assert s2.joints == s.joints
    with pytest.raises(ValueError):
        tp.PivotSample.from_json_dict({**d, "role": "bogus"})
    res = tp.solve_pivot(_plan_samples(noise=0.0003, seed=2), current_r=tp.CURRENT_URDF_R_M)
    out = json.loads(json.dumps(res.to_json_dict()))
    assert out["verdict"] == res.verdict and out["n_fit"] == 11
    assert np.allclose(out["r_tool0"], res.r_tool0)
    bad = tp.solve_pivot([])
    json.dumps(bad.to_json_dict())                      # nan/inf -> null, still valid JSON


def test_write_result_yaml(tmp_path):
    import yaml
    res = tp.solve_pivot(_plan_samples(noise=0.0003, seed=2), current_r=tp.CURRENT_URDF_R_M)
    path = tmp_path / "sub" / "pivot.yaml"
    tp.write_result_yaml(res, str(path), "arm_left_", "abc123", "/x/y.xacro", tp.CURRENT_URDF_R_M)
    doc = yaml.safe_load(path.read_text())
    assert doc["kind"] == tp.RESULT_YAML_KIND and doc["version"] == 1
    assert doc["arm"] == "arm_left_" and doc["urdf_hash"] == "abc123" and doc["xacro_path"] == "/x/y.xacro"
    assert np.allclose(doc["r_tool0_m"], res.r_tool0)
    assert np.allclose(doc["r_tool0_mm"], np.asarray(res.r_tool0) * 1000)
    for key in ("date", "n_fit", "rms_mm", "max_mm", "loo_rms_mm", "heldout_rms_mm", "sigma_axes_mm",
                "obs_min", "verdict", "reasons", "c_base_m", "delta_to_current_mm"):
        assert key in doc
    assert doc["verdict"] == res.verdict and doc["usable"] == res.ok
    assert not list(tmp_path.glob("sub/*.tmp*"))


# ------------------------------------------------------------------ real data (read-only)

def _quat_to_R(qx, qy, qz, qw):
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    x, y, z, w = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


@pytest.mark.skipif(not os.path.exists(REAL_CSV), reason="real CSV not available")
def test_real_csv_fails_with_rms_and_duplicates():
    samples = []
    with open(REAL_CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            T = np.eye(4)
            T[:3, :3] = _quat_to_R(*(float(row[k]) for k in ("qx", "qy", "qz", "qw")))
            T[:3, 3] = [float(row[k]) for k in ("tx", "ty", "tz")]
            samples.append(tp.PivotSample(arm="", T_base_tool0=T))
    assert len(samples) == 14
    res = tp.solve_pivot(samples, current_r=tp.CURRENT_URDF_R_M)
    print("REAL CSV VERDICT:", res.verdict, f"rms={res.rms_mm:.2f}", f"max={res.max_mm:.2f}")
    for r in res.reasons:
        print("  -", r)
    assert res.verdict == "FAIL" and not res.ok
    assert 4.0 < res.rms_mm < 6.5
    assert any("RMS" in r and "mm" in r for r in res.reasons)
    assert any("near-duplicate" in r for r in res.reasons)
    assert any("held-out validation missing" in r for r in res.reasons)
