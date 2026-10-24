"""Detector rules fed by the top-face depth records: top_edge_unsupported (cut rule),
multiple_boxes_suspect (reject), the K1 gap-anchored quad's gates (rectangle gate and
profile size check look at the measured RGB quad)."""
import importlib

import pytest

from box_seam_detection.core import corner_names

DETECT = "box_seam_detection.live.detector"
RING = ("back_left", "back_right", "front_right", "front_left")


@pytest.fixture()
def det():
    return importlib.import_module(DETECT)


def _live():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location(
        "live_helpers_tdr", str(Path(__file__).resolve().parent / "test_autolabel_v2_live.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


live = _live()


def _diag(unsupported=(), suspect=False, evidence=()):
    return {"edge_support": {"unsupported": list(unsupported),
                             "edges": {e: {"extent_mm": 56.0, "continue_frac": 1.0}
                                       for e in unsupported}},
            "multi_box": {"suspect": suspect, "evidence": list(evidence), "other_box_sized": int(suspect),
                          "step_blobs": [], "own_area_ratio": 0.9, "components": []}}


def _perm(det, rec, profile):
    seam = det.published_seam_line(rec)
    return det.evaluate_cut_permission(rec, seam, None, None, profile, (live.W, live.H))


def test_unsupported_edge_blocks_only_where_the_rule_is_enabled(det):
    rec = live._record()
    rec["top_face"]["depth_diag"] = _diag(["front"])
    large = det.box_profiles.PROFILES["large_392"]
    small = det.box_profiles.PROFILES["small"]
    perm = _perm(det, rec, large)
    codes = [r["code"] for r in perm["reasons"]]
    assert "top_edge_unsupported(front)" in codes
    assert perm["status"] == "blocked"
    assert [r for r in perm["reasons"] if r["code"].startswith("top_edge")][0]["retryable"] is True
    # shadow-only for the other profiles and for records without diagnostics
    assert not any(r["code"].startswith("top_edge") for r in _perm(det, rec, small)["reasons"])
    rec2 = live._record()
    assert not any(r["code"].startswith("top_edge") for r in _perm(det, rec2, large)["reasons"])


def test_multi_box_suspect_needs_the_profile_flag(det):
    rec = live._record()
    rec["top_face"]["depth_diag"] = _diag(suspect=True, evidence=["other_box_sized_coplanar_component"])
    large = det.box_profiles.PROFILES["large_392"]
    small = det.box_profiles.PROFILES["small"]
    assert det.multi_box_suspect(rec, large)["evidence"] == ["other_box_sized_coplanar_component"]
    assert det.multi_box_suspect(rec, small) is None
    assert det.multi_box_suspect(live._record(), large) is None
    assert "leave only the box" in det.MULTI_BOX_HINT


def _k1_record(rgb_scale=1.0):
    rec = live._record()
    xyz = rec["top_face"]["corners_xyz_camera_m"]
    rgb = {k: list(v) for k, v in xyz.items()}
    # make the RGB quad's front edge 30 mm longer than its back edge
    rgb["front_right"][0] += 0.015
    rgb["front_left"][0] -= 0.015
    rec["top_face"]["top_from_gap"] = {"applied": True, "rgb_quad": {"corners_xyz_camera_m": rgb}}
    return rec


def test_rectangle_gate_also_checks_the_measured_rgb_quad_of_a_k1_record(det):
    rec = _k1_record()
    with pytest.raises(det.Reject) as err:
        det.rectangle_check(rec, 20.0)
    assert "rectangle_inconsistent" in err.value.reasons[0]
    # without K1 the published (perfect) rectangle passes
    rec["top_face"]["top_from_gap"]["applied"] = False
    out = det.rectangle_check(rec, 20.0)
    assert out["pass"] and "rgb_quad_diffs_mm" not in out


def test_profile_size_check_uses_the_rgb_quad_of_a_k1_record(det):
    large = det.box_profiles.PROFILES["large_392"]
    rec = live._record()
    # published quad is exactly the profile size, the measured one is 40 mm too short
    pl, ps = large.top_dims_mm
    pub = {"back_left": [-pl / 2000, -ps / 2000, 1.2], "back_right": [pl / 2000, -ps / 2000, 1.2],
           "front_right": [pl / 2000, ps / 2000, 1.2], "front_left": [-pl / 2000, ps / 2000, 1.2]}
    rec["top_face"]["corners_xyz_camera_m"] = pub
    assert det._final_top_dims_dev_mm(rec, large) == pytest.approx(0.0, abs=1e-6)
    short = {k: [v[0] * (pl - 40) / pl, v[1], v[2]] for k, v in pub.items()}
    rec["top_face"]["top_from_gap"] = {"applied": True, "rgb_quad": {"corners_xyz_camera_m": short}}
    assert det._final_top_dims_dev_mm(rec, large) == pytest.approx(40.0, abs=0.5)


def test_rename_record_permutes_depth_records(monkeypatch):
    rec = {"top_face": {
        "corners_px": {"back_left": [0, 0], "back_right": [10, 0], "front_right": [10, 5],
                       "front_left": [0, 5]},
        "plane": None,
        "top_from_gap": {"rgb_quad": {"corners_px": {"back_left": [1, 1], "back_right": [2, 2],
                                                     "front_right": [3, 3], "front_left": [4, 4]}},
                         "deltas_mm": {"back": 1.0, "right": 2.0, "front": 3.0, "left": 4.0},
                         "sides_mm": [10.0, 20.0, 30.0, 40.0]},
        "depth_diag": {"edge_support": {"edges": {"back": {"x": 1}, "right": {"x": 2},
                                                  "front": {"x": 3}, "left": {"x": 4}},
                                        "unsupported": ["front", "left"]},
                       "known_size_fit": {"corner_disagreement_mm": [1.0, 2.0, 3.0, 4.0]}}},
        "checks": {"reasons": [], "corner_flags": {"corners": {}}}, "seam": {}, "provenance": {}}
    new2old = {"back_left": "front_right", "back_right": "front_left",
               "front_right": "back_left", "front_left": "back_right"}   # 180 deg turn

    def fake(named, plane, manifest, **kw):
        return named, {"from_names": new2old, "renamed": True, "flags": [], "ambiguous": False,
                       "method": "x", "orientation": "cw", "long_side_mm": 1, "short_side_mm": 1,
                       "back_mid_dist_m": 1, "front_mid_dist_m": 1}

    monkeypatch.setattr(corner_names, "canonical_names_for_manifest", fake)
    out, info = corner_names.rename_record(rec, None)
    tf = out["top_face"]
    assert tf["top_from_gap"]["rgb_quad"]["corners_px"]["back_left"] == [3, 3]
    # 180 deg turn: new back = old front, new right = old left
    assert tf["top_from_gap"]["deltas_mm"] == {"back": 3.0, "right": 4.0, "front": 1.0, "left": 2.0}
    assert tf["top_from_gap"]["sides_mm"] == [30.0, 40.0, 10.0, 20.0]
    es = tf["depth_diag"]["edge_support"]
    assert es["edges"]["back"] == {"x": 3} and es["unsupported"] == ["back", "right"]
    assert tf["depth_diag"]["known_size_fit"]["corner_disagreement_mm"] == [3.0, 4.0, 1.0, 2.0]


def test_border_rule_uses_measured_rgb_quad_when_top_from_gap(det_mod=None):
    """The image-border visibility rule must judge the MEASURED (RGB) outline, not the
    model-placed top_from_gap corners."""
    import importlib
    det = importlib.import_module("box_seam_detection.live.detector")
    inside = {"back_left": [400.0, 300.0], "back_right": [900.0, 300.0],
              "front_right": [900.0, 700.0], "front_left": [400.0, 700.0]}
    near_edge = dict(inside, front_left=[400.0, 1072.0], front_right=[900.0, 1072.0])
    rec = {"top_face": {"corners_px": inside,
                        "top_from_gap": {"rgb_quad": {"corners_px": near_edge}}}}
    perm = det.evaluate_cut_permission(rec, {"status": "ok"}, image_size=(1920, 1080))
    assert "top_face_touches_image_border" in [r["code"] for r in perm["reasons"]]
    rec["top_face"]["top_from_gap"]["rgb_quad"]["corners_px"] = inside
    perm = det.evaluate_cut_permission(rec, {"status": "ok"}, image_size=(1920, 1080))
    assert "top_face_touches_image_border" not in [r["code"] for r in perm["reasons"]]
