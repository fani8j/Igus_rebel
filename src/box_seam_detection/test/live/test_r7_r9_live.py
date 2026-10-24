"""R7 (corner-angle gate), R8 (depth-edge hardening) and R9 (joint rectangle
repair, last resort) of live.detector on the synthetic scene of
test_autolabel_v2_depth_edge.py (tilted 110 x 97 mm top, seam axis bf)."""

import importlib.util
import math
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

TESTS = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("r79_r1_helpers", str(TESTS / "test_r1_depth_edge.py"))
r1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(r1)
base = r1.base
RING = base.RING
det = base.live.det  # fixture re-export


def _shift_corners(rec, shifts_a_mm):
    """Move rule corners along E_A (in the top plane) by mm; pixels follow."""
    for name, mm in shifts_a_mm.items():
        point = np.asarray(rec["top_face"]["corners_xyz_camera_m"][name]) + (mm / 1000.0) * base.E_A
        rec["top_face"]["corners_xyz_camera_m"][name] = [round(float(v), 5) for v in point]
        rec["top_face"]["corners_px"][name] = [round(v, 2) for v in base._px(point)]
    return rec


def _manifest_patch(det, monkeypatch):
    monkeypatch.setattr(det, "camera_from_manifest", lambda manifest: base._camera())
    return {}


def _sheared(front_mm):
    return _shift_corners(base._record((0.0, 0.0)),
                          {"front_left": front_mm, "front_right": front_mm})


# ------------------------------------------------------------------ R7

def test_r7_parallelogram_rejected_though_side_lengths_match(det):
    rec = _sheared(10.0)                              # atan(10/97) = 5.9 deg
    xyz = {k: np.asarray(v) for k, v in rec["top_face"]["corners_xyz_camera_m"].items()}
    assert max(det._rect_diffs_mm(xyz).values()) < 0.05      # length check alone passes
    with pytest.raises(det.Reject) as err:
        det.rectangle_check(rec, 3.0)
    assert err.value.stage == "rectangle" and err.value.retryable is True
    assert err.value.reasons[0].startswith("rectangle_skewed(max_dev=5.")
    assert err.value.details["max_corner_dev_deg"] == pytest.approx(5.9, abs=0.1)


def test_r7_near_rectangle_4p4_deg_is_accepted_and_recorded(det):
    shear = math.tan(math.radians(4.4)) * base.SHORT_MM
    result = det.rectangle_check(_sheared(shear), 3.0)
    assert result["pass"] is True
    assert result["max_corner_dev_deg"] == pytest.approx(4.4, abs=0.05)
    assert set(result["corner_angles_deg"]) == set(RING)


def test_r7_limit_is_configurable_and_lengths_reason_wins(det):
    rec = _sheared(10.0)
    assert det.rectangle_check(rec, 3.0, 6.5)["pass"] is True
    bad = _shift_corners(base._record((0.0, 0.0)), {"front_left": 6.0})
    with pytest.raises(det.Reject) as err:
        det.rectangle_check(bad, 3.0)
    assert err.value.reasons[0].startswith("rectangle_inconsistent")


def test_r7_cli_flag_parsed_and_validated(det):
    args = det._parse_args(["--image", "i", "--pcd", "p", "--capture-manifest", "m",
                            "--out-seam-json", "o", "--work-dir", "w",
                            "--rect-max-corner-dev-deg", "4.5"])
    assert args.rect_max_corner_dev_deg == 4.5
    assert det._parse_args(["--image", "i", "--pcd", "p", "--capture-manifest", "m",
                            "--out-seam-json", "o", "--work-dir", "w"]
                           ).rect_max_corner_dev_deg == det.RECT_MAX_CORNER_DEV_DEG == 5.0


def test_r7_retry_policy_treats_skewed_as_generic_retryable():
    import importlib
    # The signature helper only describes rectangle_inconsistent: a skewed
    # reject yields None (no same-failure shortcut) but "retryable": true is
    # what the retry loop reads, so it is still retried.
    mod = importlib.import_module("box_seam_detection.live.orchestrator")
    reject = {"reasons": ["rectangle_skewed(max_dev=5.90deg)"], "retryable": True,
              "rectangle": {"diffs_mm": {"A": 0.0, "B": 0.0}, "tol_mm": 3.0}}
    assert mod.rectangle_failure_signature(reject) is None


# ------------------------------------------------------------------ R9

def test_r9_joint_fit_recovers_quad_with_one_pushed_corner(det, monkeypatch):
    _manifest_patch(det, monkeypatch)
    rec = _shift_corners(base._record((0.0, 0.0)), {"front_left": 3.5})
    with pytest.raises(det.Reject):
        det.rectangle_check(rec, 3.0)
    seam = det.published_seam_line(rec)
    out, seam_out, rect, info = det.rectangle_gate_with_repair(rec, seam, {}, 3.0)
    assert info["applied"] is True and info["flag"] == "rectangle_repaired_joint_fit"
    assert rect["pass"] is True and rect["repaired"] is True
    assert max(info["corner_moves_mm"].values()) <= det.JOINT_FIT_MAX_MOVE_MM
    assert info["max_residual_deg"] <= det.JOINT_FIT_MAX_RESIDUAL_DEG
    det.rectangle_check(out, 3.0)                     # passes gate + R7
    assert out["top_face"]["corners_px"] != rec["top_face"]["corners_px"]
    assert info["seam"]["seam_line_recomputed"] is True
    assert seam_out["seam_left"] != seam["seam_left"] or seam_out["seam_right"] != seam["seam_right"]
    # Recomputed ends lie on the repaired back/front edges.
    for key, dist in info["seam"]["seam_edge_distance_px"].items():
        assert dist < 0.5
    # Inputs untouched.
    assert rec["top_face"]["corners_px"] == _shift_corners(
        base._record((0.0, 0.0)), {"front_left": 3.5})["top_face"]["corners_px"]


def test_r9_never_runs_for_a_passing_quad(det, monkeypatch):
    _manifest_patch(det, monkeypatch)

    def boom(*a, **k):
        raise AssertionError("repair must not run")

    monkeypatch.setattr(det, "joint_rectangle_repair", boom)
    rec = base._record((0.0, 0.0))
    seam = det.published_seam_line(rec)
    out, seam_out, rect, info = det.rectangle_gate_with_repair(rec, seam, {}, 3.0)
    assert out is rec and seam_out is seam and info is None and rect["pass"] is True


def test_r9_refused_when_residual_exceeds_3_deg(det, monkeypatch):
    _manifest_patch(det, monkeypatch)
    rec = _sheared(14.0)                              # theta residual ~4.1 deg per edge
    seam = det.published_seam_line(rec)
    with pytest.raises(det.Reject) as err:
        det.rectangle_gate_with_repair(rec, seam, {}, 3.0)
    assert err.value.reasons[0].startswith("rectangle_skewed")      # exactly as before
    assert err.value.retryable is True
    assert err.value.repair["applied"] is False
    assert err.value.repair["reason"].startswith("joint_fit_residual")


def test_r9_refused_when_a_corner_would_move_too_far(det, monkeypatch):
    _manifest_patch(det, monkeypatch)
    monkeypatch.setattr(det, "JOINT_FIT_MAX_MOVE_MM", 1.0)
    rec = _shift_corners(base._record((0.0, 0.0)), {"front_left": 3.5})
    seam = det.published_seam_line(rec)
    with pytest.raises(det.Reject) as err:
        det.rectangle_gate_with_repair(rec, seam, {}, 3.0)
    assert err.value.reasons[0].startswith("rectangle_inconsistent")
    assert err.value.repair["reason"].startswith("joint_fit_corner_move")


def test_r9_repairs_a_skewed_parallelogram_within_limits(det, monkeypatch):
    _manifest_patch(det, monkeypatch)
    rec = _sheared(10.0)                              # R7 reject, residual ~2.95 deg
    seam = det.published_seam_line(rec)
    out, _, rect, info = det.rectangle_gate_with_repair(rec, seam, {}, 3.0)
    assert rect["max_corner_dev_deg"] < 0.1 and info["applied"] is True


# ------------------------------------------------------------------ R8

def _ctx(rec, det, u_ref=None):
    n, d = det._plane_nd(rec["top_face"]["plane"])
    cloud = base._cloud()
    cloud = cloud[np.abs(cloud @ n + d) < det.DEPTH_EDGE_PLANE_TOL_M]
    ctx = {"n": n, "d": d, "camera": base._camera(), "cloud_plane": cloud,
           "min_points": 2000, "min_move_mm": 0.0, "sam_box": None}
    if u_ref is not None:
        ctx["u_ref"] = u_ref
    return ctx, n


def _corner_dicts(rec):
    xyz = {k: np.asarray(rec["top_face"]["corners_xyz_camera_m"][k], dtype=float) for k in RING}
    px = {k: np.asarray(rec["top_face"]["corners_px"][k], dtype=float) for k in RING}
    return xyz, px


def test_r8a_second_pair_is_perpendicular_to_the_first_axis(det, monkeypatch):
    monkeypatch.setenv("AUTOLABEL_V2_R8_OFF", "")          # R8(a) is off by default
    rec = _shift_corners(base._record((0.0, 0.0)), {"back_right": -1.0, "front_right": -1.0})
    ctx, n = _ctx(rec, det)
    xyz, px = _corner_dicts(rec)
    # First anchor axis: E_A rotated by 2 deg in the plane.
    c, s_ = math.cos(math.radians(2.0)), math.sin(math.radians(2.0))
    u_ref = c * base.E_A + s_ * np.cross(n, base.E_A)
    u_ref /= np.linalg.norm(u_ref)
    res = det._place_edge(_ctx(rec, det, u_ref)[0], "lr", "left", xyz, px, True)
    moved = res["moved"]
    assert set(moved) == {"back_right", "front_right"}, (moved.keys(), res["info"])
    edge = moved["front_right"][0] - moved["back_right"][0]
    edge /= np.linalg.norm(edge)
    assert abs(float(edge @ u_ref)) < 1e-6            # exactly perpendicular to the first axis
    # Control: without a reference the moved edge follows the (unrotated) anchor edge.
    res0 = det._place_edge(ctx, "lr", "left", xyz, px, True)
    e0 = res0["moved"]["front_right"][0] - res0["moved"]["back_right"][0]
    assert abs(float((e0 / np.linalg.norm(e0)) @ u_ref)) > 0.02


def test_r8b_is_withdrawn_low_support_seam_mode_fix_still_applies(det):
    # Replay evidence: supports ~0.5 were correct fixes. Seam mode has no minimum support.
    cloud = base._cloud()
    q = cloud - base.CENTRE
    a, b = q @ base.E_A * 1000.0, q @ base.E_B * 1000.0
    cloud = cloud[~((b > base.SHORT_MM / 2 - 7.0) & (a > 0) & (b < base.SHORT_MM))]
    try:
        _, _, info = r1._fix(det, base._record((2.0, 2.0)), cloud=cloud)
    except det._DepthEdgeSkip as skip:
        assert not str(skip).startswith("depth_edge_unsupported")
    else:
        assert info["applied"] is True and info["support_frac"] < 0.6
    assert not hasattr(det, "DEPTH_EDGE_SEAM_MIN_SUPPORT")


def _line_cloud(slope_deg, mid_offset_mm):
    """Dense on-plane (a,b) points above a boundary line b = mid + a*tan(slope)."""
    a = np.arange(-0.055, 0.055, 0.0005)
    t = math.tan(math.radians(slope_deg))
    rows = []
    for av in a:
        edge = mid_offset_mm / 1000.0 + av * t
        for off in np.arange(0.0, 0.020, 0.001):
            rows.append((av, edge + off))
    arr = np.asarray(rows)
    return arr[:, 0], arr[:, 1]


def test_r8c_ransac_line_accepts_support_when_fraction_gate_fails(det, monkeypatch):
    monkeypatch.setenv("AUTOLABEL_V2_R8_OFF", "")          # R8(c) is off by default
    pa, pb = _line_cloud(1.9, 1.4)           # b_target = 0: line 1.4 mm off at the middle
    sup = det._edge_support_ex(pa, pb, -0.055, 0.055, 0.0, 0.0)
    assert sup["frac"] < det.DEPTH_EDGE_SUPPORT_MIN_FRAC
    assert sup["ransac_ok"] and sup["criterion"] == "ransac"
    assert sup["effective"] >= det.DEPTH_EDGE_SUPPORT_MIN_FRAC
    assert sup["ransac_inlier_frac"] >= 0.6 and abs(sup["ransac_angle_deg"]) <= 2.0


def test_r8c_ransac_rejects_wrong_direction_and_reports_fraction_criterion(det):
    pa, pb = _line_cloud(3.0, 1.4)           # 3 deg off the anchor edge
    sup = det._edge_support_ex(pa, pb, -0.055, 0.055, 0.0, 0.0)
    assert sup["ransac_ok"] is False and sup["criterion"] == "none"
    pa, pb = _line_cloud(0.0, 0.0)           # straight along the placed line
    sup = det._edge_support_ex(pa, pb, -0.055, 0.055, 0.0, 0.0)
    assert sup["criterion"] == "fraction" and sup["frac"] >= 0.8


def test_r8c_criterion_is_recorded_in_the_fix_info(det):
    _, _, info = r1._fix(det, base._record((3.0, 3.0)))
    assert info["support_criterion"] in ("fraction", "ransac")
    assert "inlier_frac" in info["support_ransac"]


def test_r8d_outward_move_beyond_5mm_is_refused(det):
    rec = r1._with_box(base._record((6.0, 6.0)))      # RGB front 6 mm inside: fix moves it OUT
    with pytest.raises(det._DepthEdgeSkip) as skip:
        r1._fix(det, rec)
    assert str(skip.value).startswith("depth_edge_moved_outward(")
    out, _, info = r1._fix(det, r1._with_box(base._record((4.5, 4.5))))   # 4.5 mm: allowed
    assert info["applied"] is True and info["outward_mm"] == pytest.approx(4.5, abs=0.3)


def test_r8d_inward_move_is_not_limited_by_the_outward_guard(det):
    out, _, info = r1._fix(det, r1._with_box(base._record((-6.0, -6.0))))
    assert info["applied"] is True and info["outward_mm"] < 0


# ------------------------------------------------------------------ CLI wiring

def test_cli_skewed_quad_is_repaired_and_flagged_when_depth_edge_is_off(det, tmp_path, monkeypatch):
    import json
    cap = base._cli_capture(tmp_path)
    base.live._install_fake_autolabel(monkeypatch, det, base._record((4.5, 0.0)))
    out = base.live._stale(tmp_path)
    assert det.main(base.live._argv(cap, tmp_path, "--no-depth-edge")) == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert "rectangle_repaired_joint_fit" in doc["autolabel_v2"]["soft_flags"]
    repair = doc["autolabel_v2"]["rect_repair"]
    assert repair["applied"] is True and len(repair["corner_moves_mm"]) == 4
    assert doc["autolabel_v2"]["rect_check"]["repaired"] is True
    assert doc["autolabel_v2"]["rect_check"]["max_corner_dev_deg"] < 0.1


def test_cli_unrepairable_skew_rejects_as_before_with_repair_diagnostic(det, tmp_path, monkeypatch):
    import json
    cap = base._cli_capture(tmp_path)
    base.live._install_fake_autolabel(monkeypatch, det, base._record((12.0, 0.0)))
    assert det.main(base.live._argv(cap, tmp_path, "--no-depth-edge")) == 1
    reject = json.loads((tmp_path / "ov" / "cap_1_autolabel_v2_reject.json")
                        .read_text(encoding="utf-8"))
    assert reject["stage"] == "rectangle" and reject["retryable"] is True
    assert reject["reasons"][0].startswith("rectangle_inconsistent")
    assert reject["rect_repair"]["applied"] is False


# ------------------------------------------------------------------ env switch

def test_env_switch_default_a_c_off_and_recorded(det, monkeypatch):
    monkeypatch.delenv("AUTOLABEL_V2_R8_OFF", raising=False)
    assert det.rules_off() == {"a", "c"}
    assert det.reliability_rules()["on"] == ["d", "r7", "r9", "r10"]
    monkeypatch.setenv("AUTOLABEL_V2_R8_OFF", "")
    assert det.rules_off() == set()
    monkeypatch.setenv("AUTOLABEL_V2_R8_OFF", "d, R9")
    assert det.rules_off() == {"d", "r9"}
    assert det.reliability_rules()["off"] == ["d", "r9"]


def test_env_switch_disables_each_subrule(det, monkeypatch):
    monkeypatch.setenv("AUTOLABEL_V2_R8_OFF", "d")
    _, _, info = r1._fix(det, r1._with_box(base._record((6.0, 6.0))))
    assert info["applied"] is True                         # outward guard off
    monkeypatch.setenv("AUTOLABEL_V2_R8_OFF", "c")
    pa, pb = _line_cloud(1.9, 1.4)
    assert det._edge_support_ex(pa, pb, -0.055, 0.055, 0.0, 0.0)["criterion"] == "none"
    monkeypatch.setenv("AUTOLABEL_V2_R8_OFF", "r9")
    monkeypatch.setattr(det, "camera_from_manifest", lambda m: base._camera())
    rec = _shift_corners(base._record((0.0, 0.0)), {"front_left": 3.5})
    with pytest.raises(det.Reject) as err:
        det.rectangle_gate_with_repair(rec, det.published_seam_line(rec), {}, 3.0)
    assert not hasattr(err.value, "repair")
    monkeypatch.setenv("AUTOLABEL_V2_R8_OFF", "a")
    rec = _shift_corners(base._record((0.0, 0.0)), {"back_right": -1.0, "front_right": -1.0})
    ctx, n = _ctx(rec, det)
    xyz, px = _corner_dicts(rec)
    u_ref = np.cross(n, base.E_B)
    u_ref = math.cos(0.05) * u_ref + math.sin(0.05) * base.E_B
    ctx["u_ref"] = u_ref
    res = det._place_edge(ctx, "lr", "left", xyz, px, True)
    e = res["moved"]["front_right"][0] - res["moved"]["back_right"][0]
    assert abs(float((e / np.linalg.norm(e)) @ u_ref)) > 0.02    # u_ref ignored


def test_env_switch_r7_off_accepts_parallelogram_via_cli(det, tmp_path, monkeypatch):
    import json
    monkeypatch.setenv("AUTOLABEL_V2_R8_OFF", "r7")
    cap = base._cli_capture(tmp_path)
    base.live._install_fake_autolabel(monkeypatch, det, _sheared(10.0))
    out = base.live._stale(tmp_path)
    assert det.main(base.live._argv(cap, tmp_path, "--no-depth-edge")) == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["autolabel_v2"]["reliability_rules"]["off"] == ["r7"]


# ------------------------------------------------------------------ R10 live path

def _ambiguous(rec):
    rec["seam"] = {"status": "ambiguous", "axis": None}
    return rec


def _seam_dir_at(det, monkeypatch, offset):
    real = det.compute_seam_direction

    def fake(record, seam, manifest, *a, **k):
        out = real(record, seam, manifest, *a, **k)
        out.update({"available": True, "base_link_angle_deg": 83.0 + offset,
                    "offset_from_prior_deg": offset,
                    "seam_direction_outside_soft_range": abs(offset) > 45.0})
        return out
    monkeypatch.setattr(det, "compute_seam_direction", fake)


def test_r10_ambiguous_seam_gets_axis_from_final_quad_geometry(det, tmp_path, monkeypatch):
    import json
    monkeypatch.delenv("AUTOLABEL_V2_R8_OFF", raising=False)
    _seam_dir_at(det, monkeypatch, 10.0)
    cap = base._cli_capture(tmp_path)
    base.live._install_fake_autolabel(monkeypatch, det, _ambiguous(base._record((4.5, 0.0))))
    out = base.live._stale(tmp_path)
    assert det.main(base.live._argv(cap, tmp_path)) == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    a = doc["autolabel_v2"]
    assert a["depth_edge"]["mode"] == "quad_geometry" and a["depth_edge"]["applied"] is True
    assert "seam_axis_from_geometry" in a["soft_flags"]
    assert a["seam_axis"] == "bf" == a["seam_geometric_axis_rule"]     # joins the long sides
    assert doc["seam_status"] == "estimated" and doc["seam_observed"] is False
    assert a["seam_geometry"]["axis_source"] == "geometry_class"
    assert a["seam_direction"] is not None
    # Seam ends lie on the contract edges (the usual contract renaming applied).
    for key in ("left_mid", "right_mid"):
        assert len(doc[key]) == 2
    assert max(a["edge_distance_px"].values()) <= 2.0


def test_r10_midline_is_the_projective_midline_of_the_final_pixel_quad(det):
    from box_seam_detection.core import seam_estimator as est
    rec = _ambiguous(base._record((0.0, 0.0)))
    try:
        det.published_seam_line(rec)
    except det.Reject as reject:
        out, seam, info = det.geometry_seam(rec, reject)
    quad = np.array([rec["top_face"]["corners_px"][k] for k in RING], dtype=float)
    eps = est.projective_midlines(quad)["bf"]["endpoints"]
    assert seam["axis"] == "bf" and seam["status"] == "estimated" and seam["observed"] is False
    assert seam["seam_left"] == [round(float(v), 2) for v in eps[0]]
    assert out["seam"]["status"] == "estimated" and out["seam"]["estimate"]["axis_source"] == "geometry_class"
    assert rec["seam"]["status"] == "ambiguous"                   # input untouched
    assert det.published_seam_line(out)["axis"] == "bf"           # republishable


def test_r10_no_geometry_axis_rejects_ambiguous_with_geometry_reason(det):
    rec = _ambiguous(base._record((0.0, 0.0)))
    for k, ab in {"back_left": (-50, -50), "back_right": (50, -50),
                  "front_right": (50, 50), "front_left": (-50, 50)}.items():
        rec["top_face"]["corners_xyz_camera_m"][k] = [round(float(v), 5) for v in base._xyz(*ab)]
    try:
        det.published_seam_line(rec)
    except det.Reject as reject:
        with pytest.raises(det.Reject) as err:
            det.geometry_seam(rec, reject)
    assert err.value.stage == "seam"
    assert err.value.reasons[0].startswith("seam_status_ambiguous")
    assert "not_small_class" in err.value.reasons[1]    # 100x100: outside the small ranges


def test_r10_off_keeps_the_ambiguous_rejection(det, tmp_path, monkeypatch):
    import json
    monkeypatch.setenv("AUTOLABEL_V2_R8_OFF", "r10")
    cap = base._cli_capture(tmp_path)
    base.live._install_fake_autolabel(monkeypatch, det, _ambiguous(base._record((4.5, 0.0))))
    assert det.main(base.live._argv(cap, tmp_path)) == 1
    reject = json.loads((tmp_path / "ov" / "cap_1_autolabel_v2_reject.json")
                        .read_text(encoding="utf-8"))
    assert reject["stage"] == "seam" and reject["reasons"][0].startswith("seam_status_ambiguous")
    assert reject["reliability_rules"]["off"] == ["r10"]


def test_r10_geometry_axis_outside_soft_range_needs_review(det, tmp_path, monkeypatch):
    import json
    monkeypatch.delenv("AUTOLABEL_V2_R8_OFF", raising=False)
    _seam_dir_at(det, monkeypatch, 58.0)
    cap = base._cli_capture(tmp_path)
    base.live._install_fake_autolabel(monkeypatch, det, _ambiguous(base._record((4.5, 0.0))))
    assert det.main(base.live._argv(cap, tmp_path)) == 1
    reject = json.loads((tmp_path / "ov" / "cap_1_autolabel_v2_reject.json")
                        .read_text(encoding="utf-8"))
    assert reject["stage"] == "seam" and reject["retryable"] is False
    assert reject["reasons"][0].startswith(
        "needs_review(seam_axis_from_geometry_outside_soft_range")


def test_r10_geometry_axis_outside_soft_range_accepted_with_allow_needs_review(det, tmp_path,
                                                                             monkeypatch):
    import json
    monkeypatch.delenv("AUTOLABEL_V2_R8_OFF", raising=False)
    _seam_dir_at(det, monkeypatch, 58.0)
    cap = base._cli_capture(tmp_path)
    base.live._install_fake_autolabel(monkeypatch, det, _ambiguous(base._record((4.5, 0.0))))
    out = base.live._stale(tmp_path)
    # M4: the review override alone no longer cuts it: cut_permission is blocked
    # (separate switch --allow-cut-blocked).
    assert det.main(base.live._argv(cap, tmp_path, "--allow-needs-review")) == 1
    reject = json.loads((tmp_path / "ov" / "cap_1_autolabel_v2_reject.json")
                        .read_text(encoding="utf-8"))
    assert reject["stage"] == "cut_permission"
    assert reject["cut_permission"]["reasons"][0]["code"] == (
        "seam_axis_from_geometry_outside_soft_range")
    assert det.main(base.live._argv(cap, tmp_path, "--allow-needs-review",
                                    "--allow-cut-blocked")) == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    a = doc["autolabel_v2"]
    assert a["needs_review"] is True
    assert a["seam_geometry"]["needs_review_reason"] == "seam_axis_from_geometry_outside_soft_range"
    assert doc["cut_permission"]["status"] == "blocked"


def _pipeline_geometry_seam(rec):
    # The pipeline's seam stage already chose the axis from geometry (R10 in
    # seam_estimator): status estimated, axis_source geometry_class.
    rec["seam"].setdefault("estimate", {})
    rec["seam"]["status"] = "estimated"
    rec["seam"]["estimate"]["axis_source"] = "geometry_class"
    return rec


def test_pipeline_geometry_seam_outside_soft_range_needs_review(det, tmp_path, monkeypatch):
    import json
    monkeypatch.delenv("AUTOLABEL_V2_R8_OFF", raising=False)
    _seam_dir_at(det, monkeypatch, 58.0)
    cap = base._cli_capture(tmp_path)
    rec = _pipeline_geometry_seam(base._record((0.0, 0.0)))
    base.live._install_fake_autolabel(monkeypatch, det, rec)
    assert det.main(base.live._argv(cap, tmp_path)) == 1
    reject = json.loads((tmp_path / "ov" / "cap_1_autolabel_v2_reject.json")
                        .read_text(encoding="utf-8"))
    assert reject["reasons"][0].startswith(
        "needs_review(seam_axis_from_geometry_outside_soft_range")


def test_pipeline_geometry_seam_in_range_is_flagged(det, tmp_path, monkeypatch):
    import json
    monkeypatch.delenv("AUTOLABEL_V2_R8_OFF", raising=False)
    _seam_dir_at(det, monkeypatch, 10.0)
    cap = base._cli_capture(tmp_path)
    rec = _pipeline_geometry_seam(base._record((0.0, 0.0)))
    base.live._install_fake_autolabel(monkeypatch, det, rec)
    out = base.live._stale(tmp_path)
    assert det.main(base.live._argv(cap, tmp_path)) == 0
    a = json.loads(out.read_text(encoding="utf-8"))["autolabel_v2"]
    assert "seam_axis_from_geometry" in a["soft_flags"]
    assert a["seam_geometry"]["path"] == "pipeline_seam_stage"
    assert a["needs_review"] is False
