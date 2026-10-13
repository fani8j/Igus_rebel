"""M4 live part: cut_permission {status, reasons[{code, retryable, detail}]} in the
canonical seam JSON and the reject JSON; blocked -> reject (stage cut_permission)
unless --allow-cut-blocked (CLI, server request, orchestrator)."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("cutperm_live_helpers",
                                               str(TESTS / "test_autolabel_v2_live.py"))
live = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(live)
W, H = live.W, live.H


@pytest.fixture()
def det():
    return live._load(live.DETECT, "autolabel_v2_detect_cutperm")


def _perm(det, rec=None, seam=None, direction=None, geometry=None, profile=None, size=(W, H)):
    rec = rec or live._record()
    seam = seam if seam is not None else det.published_seam_line(rec)
    return det.evaluate_cut_permission(rec, seam, direction, geometry, profile, size)


def _codes(perm):
    return [r["code"] for r in perm["reasons"]]


def _generic(det):
    return det.box_profiles.PROFILES["generic"]


def _small(det):
    return det.box_profiles.PROFILES["small"]


def _with_height(rec=None, mm=480.0):
    rec = rec or live._record()
    rec["box_profile"] = {"name": "generic", "measured_height_mm": mm, "flags": []}
    return rec


def test_plain_detection_is_allowed(det):
    perm = _perm(det)
    assert perm == {"status": "allowed", "reasons": []}


def test_reason_shape(det):
    rec = live._record()
    rec["top_face"]["corners_px"]["back_left"] = [3.0, 100.0]
    perm = _perm(det, rec)
    assert perm["status"] == "blocked"
    assert set(perm["reasons"][0]) == {"code", "retryable", "detail"}


def test_image_border_blocks_and_is_retryable(det):
    rec = live._record()
    rec["top_face"]["corners_px"]["front_right"] = [W - 1.0 - 19.0, 250.0]    # 19 px
    perm = _perm(det, rec)
    assert _codes(perm) == ["top_face_touches_image_border"]
    assert perm["reasons"][0]["retryable"] is True


def test_image_border_margin_just_inside_is_allowed(det):
    rec = live._record()
    rec["top_face"]["corners_px"]["front_right"] = [W - 1.0 - 21.0, 250.0]
    assert _perm(det, rec)["status"] == "allowed"


def test_geometry_axis_outside_soft_range_blocks_not_retryable(det):
    rec = live._record()
    geometry = {"axis": "lr", "axis_source": "geometry_class"}
    direction = {"seam_direction_outside_soft_range": True, "offset_from_prior_deg": 60.0,
                 "base_link_angle_deg": 143.0}
    perm = _perm(det, rec, direction=direction, geometry=geometry, profile=_small(det))
    assert _codes(perm) == ["seam_axis_from_geometry_outside_soft_range"]
    assert perm["reasons"][0]["retryable"] is False


def test_geometry_axis_inside_soft_range_is_allowed_for_small(det):
    direction = {"seam_direction_outside_soft_range": False, "base_link_angle_deg": 80.0}
    perm = _perm(det, direction=direction, geometry={"axis": "lr"}, profile=_small(det))
    assert perm["status"] == "allowed"


def test_geometry_axis_direction_unknown_blocks(det):
    perm = _perm(det, direction={"base_link_angle_deg": None}, geometry={"axis": "lr"},
                 profile=_small(det))
    assert _codes(perm) == ["seam_axis_from_geometry_direction_unknown"]


def test_geometry_axis_on_generic_profile_always_blocked(det):
    direction = {"seam_direction_outside_soft_range": False, "base_link_angle_deg": 83.0}
    rec = _with_height()
    perm = _perm(det, rec, direction=direction, geometry={"axis": "lr"}, profile=_generic(det))
    assert _codes(perm) == ["geometry_axis_generic_profile"]
    assert perm["reasons"][0]["retryable"] is False


def test_pipeline_geometry_axis_source_counts_as_geometry_seam(det):
    rec = live._record()
    rec["seam"]["estimate"]["axis_source"] = "geometry_class"
    direction = {"seam_direction_outside_soft_range": False, "base_link_angle_deg": 83.0}
    rec["box_profile"] = {"name": "generic", "measured_height_mm": 480.0, "flags": []}
    assert _codes(_perm(det, rec, direction=direction, profile=_generic(det))) == [
        "geometry_axis_generic_profile"]


@pytest.mark.parametrize("status", ["ambiguous", "invisible"])
def test_seam_status_ambiguous_or_invisible_blocks(det, status):
    perm = _perm(det, seam={"status": status})
    assert _codes(perm) == [f"seam_status_{status}"]


def test_low_gap_coverage_blocks_only_below_threshold(det):
    rec = live._record()
    rec["seam"]["estimate"]["coverage"] = 0.01
    assert _codes(_perm(det, rec)) == ["seam_coverage_low"]
    rec["seam"]["estimate"]["coverage"] = 0.041       # lowest seen on the 410 accepted replays
    assert _perm(det, rec)["status"] == "allowed"
    rec["seam"]["estimate"].pop("coverage")
    assert _perm(det, rec)["status"] == "allowed"


def test_height_mismatch_in_box_profile_flags_blocks(det):
    rec = live._record()
    rec["box_profile"] = {"name": "small", "flags": ["height_mismatch"],
                          "measured_height_mm": 93.0}
    assert _codes(_perm(det, rec)) == ["height_mismatch"]
    rec["box_profile"]["flags"] = ["some_other_flag"]       # other flags do not block
    assert _perm(det, rec)["status"] == "allowed"


def _set_top_mm(rec, long_mm, short_mm, z=0.5):
    a, b = long_mm / 2000.0, short_mm / 2000.0
    rec["top_face"]["corners_xyz_camera_m"] = {
        "back_left": [-a, -b, z], "back_right": [a, -b, z],
        "front_right": [a, b, z], "front_left": [-a, b, z]}
    return rec


def test_final_top_face_far_from_profile_blocks_retryable(det):
    # known (small) box but the final quad is a 101 x 33 mm tape band
    rec = _set_top_mm(live._record(), 101.0, 33.0)
    rec["box_profile"] = {"name": "small", "flags": ["profile_mismatch"],
                          "measured_height_mm": 86.4}
    perm = _perm(det, rec, profile=_small(det))
    assert _codes(perm) == ["top_face_does_not_match_profile"]
    assert perm["reasons"][0]["retryable"] is True
    # truncated top (100 x 77 mm, ~21 mm off) -> blocked; close to the profile -> not
    rec = _set_top_mm(live._record(), 100.0, 77.0)
    rec["box_profile"] = {"name": "small", "flags": [], "measured_height_mm": 86.4}
    assert "top_face_does_not_match_profile" in _codes(_perm(det, rec, profile=_small(det)))
    rec = _set_top_mm(live._record(), 106.0, 92.0)
    rec["box_profile"] = {"name": "small", "flags": ["profile_mismatch"],
                          "measured_height_mm": 86.4}
    assert "top_face_does_not_match_profile" not in _codes(_perm(det, rec, profile=_small(det)))
    # without the pipeline's box_profile (corner fallback) the rule does not apply
    rec = _set_top_mm(live._record(), 101.0, 33.0)
    assert "top_face_does_not_match_profile" not in _codes(_perm(det, rec, profile=_small(det)))
    # generic profile: rule does not apply (no profile dims)
    rec = _set_top_mm(_with_height(), 101.0, 33.0)
    assert "top_face_does_not_match_profile" not in _codes(
        _perm(det, rec, seam={"status": "ok"}, profile=_generic(det)))

def test_height_mismatch_old_guessed_places_are_not_read(det):
    rec = live._record()
    rec["checks"]["reasons"] = ["height_mismatch(+9.1mm)"]
    rec["height_mismatch"] = True
    assert _perm(det, rec)["status"] == "allowed"


def test_generic_with_unconfirmed_seam_is_blocked(det):
    rec = _with_height()
    perm = _perm(det, rec, seam={"status": "estimated"}, profile=_generic(det))
    assert _codes(perm) == ["generic_seam_unconfirmed"]
    assert perm["reasons"][0]["retryable"] is False
    assert _perm(det, rec, seam={"status": "ok"}, profile=_generic(det))["status"] == "allowed"
    # known profiles keep accepting estimated seams
    assert _perm(det, live._record(), seam={"status": "estimated"},
                 profile=_small(det))["status"] == "allowed"


def test_generic_without_measured_height_is_blocked(det):
    rec = live._record()
    rec["box_profile"] = {"name": "generic", "measured_height_mm": None,
                          "height_unavailable": "no_table_plane", "flags": []}
    perm = _perm(det, rec, profile=_generic(det))
    assert _codes(perm) == ["height_unavailable"]
    assert perm["reasons"][0]["retryable"] is False
    _set_top_mm(rec, 110.0, 98.0)                         # a real small-box top
    # any profile: no measured height -> blocked (a taller box could share the footprint)
    assert _codes(_perm(det, rec, profile=_small(det))) == ["height_unavailable"]
    rec["box_profile"] = {"name": "small", "measured_height_mm": 86.4, "flags": []}
    assert _perm(det, rec, profile=_small(det))["status"] == "allowed"
    # old-format record without any box_profile: nothing to demand
    assert _perm(det, _set_top_mm(live._record(), 110.0, 98.0),
                 profile=_small(det))["status"] == "allowed"


def test_tape_strip_shadow_is_never_read(det):
    rec = _with_height()
    rec["seam"]["tape_strip_shadow"] = {"candidates": [{"axis": "bf"}, {"axis": "lr"}, {}],
                                        "ok": False}
    assert _perm(det, rec, profile=_generic(det))["status"] == "allowed"


def test_tape_evidence_counts_only_qualified_candidates_with_different_axes(det):
    rec = live._record()
    cand = lambda axis, ok=True: {"axis": axis, "ok": ok}  # noqa: E731
    rec["seam"]["tape_evidence"] = {"candidates": [cand("bf"), cand("lr")]}
    perm = _perm(det, rec)
    assert _codes(perm) == ["multiple_tape_candidates"]
    assert perm["reasons"][0]["retryable"] is False
    rec["seam"]["tape_evidence"] = {"candidates": [{"axis": "bf", "qualified": True},
                                                   {"axis": "lr", "qualified": True}]}
    assert _codes(_perm(det, rec)) == ["multiple_tape_candidates"]
    # unqualified extra candidate (noise) does not block
    rec["seam"]["tape_evidence"] = {"candidates": [cand("bf"), cand("lr", ok=False)]}
    assert _perm(det, rec)["status"] == "allowed"
    # two qualified strips on the SAME axis do not conflict on axis
    rec["seam"]["tape_evidence"] = {"candidates": [cand("bf"), cand("bf")]}
    assert _perm(det, rec)["status"] == "allowed"
    rec["seam"]["tape_evidence"] = {"candidates": [cand("bf")]}
    assert _perm(det, rec)["status"] == "allowed"
    rec["seam"].pop("tape_evidence")
    assert _perm(det, rec)["status"] == "allowed"


def test_several_reasons_are_all_listed(det):
    rec = live._record()
    rec["top_face"]["corners_px"]["back_left"] = [2.0, 100.0]
    rec["box_profile"] = {"name": "small", "flags": ["height_mismatch"]}
    assert sorted(_codes(_perm(det, rec))) == ["height_mismatch", "top_face_touches_image_border"]


# ------------------------------------------------------------ CLI / live behaviour

def _near_border_record():
    corners = dict(live.RULE)
    corners["back_left"] = [4.0, 100.0]
    return live._record(corners=corners, axis="bf")


def test_blocked_capture_is_rejected_at_stage_cut_permission(det, tmp_path, monkeypatch):
    cap = live._capture(tmp_path)
    live._install_fake_autolabel(monkeypatch, det, _near_border_record())
    out = live._stale(tmp_path)
    assert det.main(live._argv(cap, tmp_path, "--no-depth-edge")) == 1
    reject = json.loads((tmp_path / "ov" / "cap_1_autolabel_v2_reject.json")
                        .read_text(encoding="utf-8"))
    assert reject["stage"] == "cut_permission" and reject["retryable"] is True
    assert reject["cut_permission"]["status"] == "blocked"
    assert reject["cut_permission"]["reasons"][0]["code"] == "top_face_touches_image_border"
    assert not out.exists()


def test_allow_cut_blocked_accepts_and_records_blocked(det, tmp_path, monkeypatch):
    cap = live._capture(tmp_path)
    live._install_fake_autolabel(monkeypatch, det, _near_border_record())
    out = live._stale(tmp_path)
    assert det.main(live._argv(cap, tmp_path, "--no-depth-edge", "--allow-cut-blocked")) == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["cut_permission"]["status"] == "blocked"
    assert doc["cut_permission"]["override"] == "allow_cut_blocked"
    assert doc["autolabel_v2"]["cut_permission"] == doc["cut_permission"]


def test_allowed_capture_records_allowed_and_profile(det, tmp_path, monkeypatch):
    cap = live._capture(tmp_path)
    rec = live._record(axis="bf")
    rec["box_profile"] = {"name": "small"}
    rec["measured_height_mm"] = 86.4
    _set_top_mm(rec, 110.0, 98.0)          # top face matching the small profile
    live._install_fake_autolabel(monkeypatch, det, rec)
    out = live._stale(tmp_path)
    assert det.main(live._argv(cap, tmp_path, "--no-depth-edge")) == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["cut_permission"] == {"status": "allowed", "reasons": []}
    assert doc["autolabel_v2"]["box_profile"]["name"] == "small"
    assert doc["autolabel_v2"]["box_profile"]["source"] == "record"
    assert doc["autolabel_v2"]["measured_height_mm"] == 86.4


def test_other_rejects_carry_a_blocked_cut_permission(det, tmp_path, monkeypatch):
    cap = live._capture(tmp_path)
    live._install_fake_autolabel(monkeypatch, det, live._record(wrong_face=True))
    assert det.main(live._argv(cap, tmp_path)) == 1
    reject = json.loads((tmp_path / "ov" / "cap_1_autolabel_v2_reject.json")
                        .read_text(encoding="utf-8"))
    assert reject["cut_permission"]["status"] == "blocked"
    assert reject["cut_permission"]["reasons"][0]["code"] == "rejected_record"


def test_cli_and_server_request_switch(det):
    base = ["--image", "i", "--pcd", "p", "--capture-manifest", "m",
            "--out-seam-json", "o", "--work-dir", "w"]
    assert det._parse_args(base).allow_cut_blocked is False
    assert det._parse_args([*base, "--allow-cut-blocked"]).allow_cut_blocked is True
    assert det.request_argv({"id": 1, "allow_cut_blocked": True}) == ["--allow-cut-blocked"]
    assert det.request_argv({"id": 1, "allow_cut_blocked": False}) == []


def test_orchestrator_passes_allow_cut_blocked_to_the_detector(det, tmp_path, monkeypatch):
    cap = live._capture(tmp_path)
    mod = live._load(live.ORCHESTRATOR, "capture_and_query_depth_cutperm")
    script = tmp_path / "autolabel_v2_detect.py"
    script.write_text("# fake\n", encoding="utf-8")

    class _Probe:
        returncode = 0
        stdout = ""

    monkeypatch.setattr(mod.subprocess, "run", lambda cmd, *a, **k: _Probe())
    seen = []

    def fake_run_step(title, cmd, capture=False, check=True):
        seen.append(cmd)
        live._halt(mod, title)

    monkeypatch.setattr(mod, "run_step", fake_run_step)
    code = mod.main([
        "--downstream-inprocess", "off", "--autolabel-v2",
        "--autolabel-v2-script", str(script),
        "--autolabel-v2-sam3-python", str(cap["sam3_python"]), "--allow-cut-blocked",
        "--from-image", str(cap["image"]), "--pcd", str(cap["pcd"]),
        "--capture-manifest", str(cap["manifest"]),
        "--calib-file", str(live._calib(tmp_path)),
        "--output-dir", str(tmp_path / "root"),
    ])
    assert code == live.STOP_CODE
    assert "--allow-cut-blocked" in seen[0] and "--allow-needs-review" not in seen[0]


def test_cut_block_retryable_only_when_every_reason_is_retryable(det, tmp_path, monkeypatch):
    cap = live._capture(tmp_path)
    rec = _near_border_record()
    rec["checks"]["needs_review"] = False
    rec["box_profile"] = {"name": "small", "flags": ["height_mismatch"], "measured_height_mm": 93.0}
    live._install_fake_autolabel(monkeypatch, det, rec)
    live._stale(tmp_path)
    assert det.main(live._argv(cap, tmp_path, "--no-depth-edge")) == 1
    reject = json.loads((tmp_path / "ov" / "cap_1_autolabel_v2_reject.json")
                        .read_text(encoding="utf-8"))
    codes = [r["code"] for r in reject["cut_permission"]["reasons"]]
    assert "top_face_touches_image_border" in codes and "height_mismatch" in codes
    assert reject["retryable"] is False          # one non-retryable reason stops the retry


def test_orchestrator_cut_permission_signature_for_smart_retry():
    orch = live._load(live.ORCHESTRATOR, "capture_and_query_depth_cutsig")
    rej = lambda *codes, stage="cut_permission": {  # noqa: E731
        "stage": stage, "cut_permission": {"reasons": [{"code": c} for c in codes]}}
    assert orch.cut_permission_signature(rej("b", "a")) == ["a", "b"]
    assert orch.cut_permission_signature(rej("a", "b")) == orch.cut_permission_signature(
        rej("b", "a", "a"))
    assert orch.cut_permission_signature(rej("a", stage="rectangle")) is None
    assert orch.cut_permission_signature(rej()) is None
    assert orch.cut_permission_signature(None) is None


# ---- support_hidden (dual-arm scene): known profile keeps its height --------------

def _hidden_rec(name):
    rec = live._record()
    rec["box_profile"] = {"name": name, "measured_height_mm": None,
                          "height_unavailable": "support_hidden", "flags": []}
    return rec


def test_support_hidden_does_not_block_a_known_profile(det):
    prof = det.box_profiles.PROFILES["large_392"]
    rec = _hidden_rec("large_392")
    rec["seam"]["status"] = "ok"
    perm = _perm(det, rec, profile=prof)
    assert "height_unavailable" not in _codes(perm)


def test_support_hidden_still_blocks_generic_and_plain_missing_height(det):
    rec = _hidden_rec("generic")
    assert "height_unavailable" in _codes(_perm(det, rec, profile=_generic(det)))
    rec = _hidden_rec("large_392")
    rec["box_profile"]["height_unavailable"] = "no_table_plane"      # a real failure
    assert "height_unavailable" in _codes(_perm(det, rec, profile=det.box_profiles.PROFILES["large_392"]))


def test_large_profile_needs_a_confirmed_seam(det):
    prof = det.box_profiles.PROFILES["large_392"]
    rec = _hidden_rec("large_392")
    perm = _perm(det, rec, seam={"status": "estimated"}, profile=prof)
    assert "generic_seam_unconfirmed" in _codes(perm)


def test_seam_too_curved_blocks_only_profiles_with_a_limit(det):
    large = det.box_profiles.PROFILES["large_392"]
    rec = _with_height()
    rec["box_profile"]["name"] = "large_392"
    rec["seam"] = {"estimate": {"seam_curvature_mm": 6.5}}
    perm = _perm(det, rec, seam={"status": "ok"}, profile=large)
    assert "seam_too_curved" in _codes(perm)
    assert next(r for r in perm["reasons"] if r["code"] == "seam_too_curved")["retryable"] is False
    rec["seam"]["estimate"]["seam_curvature_mm"] = 3.0
    assert "seam_too_curved" not in _codes(_perm(det, rec, seam={"status": "ok"}, profile=large))
    # no limit on legacy / generic profiles
    rec["seam"]["estimate"]["seam_curvature_mm"] = 50.0
    assert "seam_too_curved" not in _codes(_perm(det, rec, seam={"status": "ok"}, profile=_small(det)))
    assert "seam_too_curved" not in _codes(_perm(det, rec, seam={"status": "ok"}, profile=_generic(det)))


# ---- generic_cut_without_height (scene option, dual-arm) -----------------------

def _generic_hidden_rec(confirmed=True, unsupported=()):
    rec = _hidden_rec("generic")
    rec["seam"]["status"] = "ok"
    rec.setdefault("checks", {})["independent_top_check"] = {"confirmed": confirmed}
    edges = {e: {"status": "ok", "valid_frac": 1.0, "extent_mm": 2.0, "continue_frac": 0.0}
             for e in ("back", "right", "front", "left")}
    rec["top_face"]["depth_diag"] = {"edge_support": {"unsupported": list(unsupported), "edges": edges}}
    rec["box_profile"]["support_clearance_mm"] = 970.0
    return rec


def _perm_g(det, rec, flag=True, profile=None):
    return det.evaluate_cut_permission(
        rec, det.published_seam_line(rec), None, None, profile or _generic(det), (W, H),
        generic_cut_without_height=flag)


def test_generic_cut_without_height_allowed_when_confirmed_and_supported(det):
    perm = _perm_g(det, _generic_hidden_rec())
    assert "height_unavailable" not in _codes(perm)
    assert perm["flags"] == ["height_waived(support_hidden)"]


def test_generic_cut_without_height_still_blocked_otherwise(det):
    assert "height_unavailable" in _codes(_perm_g(det, _generic_hidden_rec(confirmed=False)))
    assert "height_unavailable" in _codes(_perm_g(det, _generic_hidden_rec(confirmed=None)))
    assert "height_unavailable" in _codes(_perm_g(det, _generic_hidden_rec(unsupported=["front"])))
    rec = _generic_hidden_rec()
    rec["top_face"].pop("depth_diag")                                    # no edge-support evidence
    assert "height_unavailable" in _codes(_perm_g(det, rec))
    rec = _generic_hidden_rec()                                          # all edges insufficient_depth
    for e in rec["top_face"]["depth_diag"]["edge_support"]["edges"].values():
        e.clear(); e["status"] = "insufficient_depth"
    assert "height_unavailable" in _codes(_perm_g(det, rec))
    rec = _generic_hidden_rec()                                          # one edge poorly sampled
    rec["top_face"]["depth_diag"]["edge_support"]["edges"]["left"]["valid_frac"] = 0.5
    assert "height_unavailable" in _codes(_perm_g(det, rec))
    rec = _generic_hidden_rec()                                          # top continues 15 mm past the
    for side in ("left", "right"):                                       # seam ends (real lid larger than
        rec["top_face"]["depth_diag"]["edge_support"]["edges"][side].update(  # the quad), not "unsupported"
            extent_mm=15.0, continue_frac=1.0)
    assert "height_unavailable" in _codes(_perm_g(det, rec))
    rec = _generic_hidden_rec()                                          # extent missing
    rec["top_face"]["depth_diag"]["edge_support"]["edges"]["back"].pop("extent_mm")
    assert "height_unavailable" in _codes(_perm_g(det, rec))
    rec = _generic_hidden_rec()                                          # edge_support without edges
    rec["top_face"]["depth_diag"]["edge_support"]["edges"] = {}
    assert "height_unavailable" in _codes(_perm_g(det, rec))
    rec = _generic_hidden_rec()                                          # circular R5 skip
    rec["checks"]["independent_top_check"]["skipped_rectangle_fitted"] = True
    assert "height_unavailable" in _codes(_perm_g(det, rec))
    quad = [[700.0, 600.0], [1200.0, 600.0], [1200.0, 900.0], [700.0, 900.0]]
    rec["top_face"]["corners_px"] = dict(zip(("back_left", "back_right", "front_right", "front_left"), quad))
    rec["provenance"] = {"attempts": [{"status": "accepted", "selected": True,       # ... unless the
                                       "original_coarse": [[p[0] + 5, p[1] + 4] for p in quad]}]}  # SAM3 quad agrees
    assert "height_unavailable" not in _codes(_perm_g(det, rec))
    rec["provenance"]["attempts"][0]["original_coarse"] = [[p[0] + 20, p[1]] for p in quad]  # 20 px off
    assert "height_unavailable" in _codes(_perm_g(det, rec))
    rec["provenance"]["attempts"][0].pop("selected")                                       # no selected attempt
    rec["provenance"]["attempts"][0]["original_coarse"] = quad
    assert "height_unavailable" in _codes(_perm_g(det, rec))
    rec = _generic_hidden_rec()                                          # support not measured / unchecked
    rec["box_profile"].pop("support_clearance_mm")
    assert "height_unavailable" in _codes(_perm_g(det, rec))
    rec = _generic_hidden_rec()
    rec["box_profile"]["support_gravity_unchecked"] = True
    assert "height_unavailable" in _codes(_perm_g(det, rec))
    perm = _perm_g(det, _generic_hidden_rec(), flag=False)               # scene option off
    assert "height_unavailable" in _codes(perm) and "flags" not in perm
    rec = _generic_hidden_rec()
    rec["box_profile"]["height_unavailable"] = "no_table_plane"          # a real failure
    assert "height_unavailable" in _codes(_perm_g(det, rec))


def test_generic_cut_without_height_scene_flag():
    from box_seam_detection.core import scene as sc
    assert sc.load_scene("dual_arm_v1").generic_cut_without_height is True
    assert sc.load_scene("table_cell_v1").generic_cut_without_height is False
    assert sc.builtin_default().generic_cut_without_height is False
