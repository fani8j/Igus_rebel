"""Hard top-size gate (scene option ``live.top_size_range_mm``): any profile, never bypassed by
--allow-needs-review / --allow-cut-blocked; scene validation; smart-retry signature."""
import importlib.util
import json
import math
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent


def _helpers(name, fname):
    spec = importlib.util.spec_from_file_location(name, str(TESTS / fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


live = _helpers("topsize_live_helpers", "test_autolabel_v2_live.py")
r4 = _helpers("topsize_r4_helpers", "test_r4_retry_policy.py")

RANGE = {"long": [330.0, 450.0], "short": [230.0, 380.0]}


@pytest.fixture()
def det():
    return live._load(live.DETECT, "autolabel_v2_detect_topsize")


def _xyz(long_mm, short_mm, deg=0.0, z=1.2, centre=(0.1, -0.05)):
    """Rule-named 3-D rectangle (camera frame, m) rotated by ``deg`` about the optical axis."""
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    a, b = long_mm / 2000.0, short_mm / 2000.0
    out = {}
    for name, (x, y) in {"back_left": (-a, -b), "back_right": (a, -b),
                         "front_right": (a, b), "front_left": (-a, b)}.items():
        out[name] = [centre[0] + c * x - s * y, centre[1] + s * x + c * y, z]
    return out


def _rec(long_mm, short_mm, deg=0.0):
    rec = live._record()
    rec["top_face"]["corners_xyz_camera_m"] = _xyz(long_mm, short_mm, deg)
    return rec


# ---- the pure check ----------------------------------------------------------

@pytest.mark.parametrize("long_mm,short_mm", [(394.5, 328.6), (390.0, 280.0), (331.0, 231.0),
                                              (449.0, 379.0)])
def test_in_range_passes_and_records_dims(det, long_mm, short_mm):
    out = det.top_size_check(_rec(long_mm, short_mm), RANGE)
    assert out["pass"] is True and out["enabled"] is True and out["measured"] is True
    assert out["long_mm"] == pytest.approx(long_mm, abs=0.01)
    assert out["short_mm"] == pytest.approx(short_mm, abs=0.01)
    assert out["range_mm"] == RANGE


@pytest.mark.parametrize("long_mm,short_mm", [
    (320.0, 300.0),     # long too small
    (460.0, 330.0),     # long too big
    (372.0, 144.0),     # half flap (real capture): short too small
    (391.0, 117.0),     # half flap (real capture)
    (400.0, 390.0),     # short too big
])
def test_out_of_range_rejects_retryable(det, long_mm, short_mm):
    with pytest.raises(det.Reject) as info:
        det.top_size_check(_rec(long_mm, short_mm), RANGE)
    err = info.value
    assert err.stage == "top_size" and err.retryable is True
    assert err.reasons[0].startswith(
        f"top_size_out_of_range(long={long_mm:.1f}mm,short={short_mm:.1f}mm,range=")
    assert err.details["top_size_check"]["pass"] is False


@pytest.mark.parametrize("deg", [0.0, 17.0, 45.0, 90.0, 133.0])
def test_rotation_invariant(det, deg):
    assert det.top_size_check(_rec(394.5, 328.6, deg), RANGE)["pass"] is True
    with pytest.raises(det.Reject):
        det.top_size_check(_rec(372.0, 144.0, deg), RANGE)


def test_swapped_pair_lengths_give_the_same_result(det):
    # the "long" pair may be either RING pair
    rec = _rec(328.6, 394.5)
    out = det.top_size_check(rec, RANGE)
    assert out["long_mm"] == pytest.approx(394.5, abs=0.01)


@pytest.mark.parametrize("corners", [None, {"back_left": [0, 0, 1]}, "junk"])
def test_unmeasurable_quad_rejects_fail_closed(det, corners):
    rec = live._record()
    rec["top_face"]["corners_xyz_camera_m"] = corners
    with pytest.raises(det.Reject) as info:
        det.top_size_check(rec, RANGE)
    assert info.value.stage == "top_size"
    assert info.value.reasons[0].startswith("top_size_unmeasurable(")
    assert not info.value.reasons[0].startswith("top_size_out_of_range")


def test_nonfinite_corner_rejects(det):
    rec = _rec(394.5, 328.6)
    rec["top_face"]["corners_xyz_camera_m"]["front_left"][2] = float("nan")
    with pytest.raises(det.Reject) as info:
        det.top_size_check(rec, RANGE)
    assert info.value.reasons[0].startswith("top_size_unmeasurable(")


def test_disabled_without_option(det):
    assert det.top_size_check(_rec(372.0, 144.0), None) == {"enabled": False}
    assert det.top_size_check(live._record(), None) == {"enabled": False}      # even unmeasurable


def test_measures_the_rgb_quad_of_a_gap_anchored_record(det):
    rec = _rec(394.5, 328.6)                                  # published (model) quad looks fine
    rec["top_face"]["top_from_gap"] = {"applied": True, "rgb_quad": {
        "corners_xyz_camera_m": _xyz(372.0, 144.0)}}          # but the measured one is a half flap
    with pytest.raises(det.Reject):
        det.top_size_check(rec, RANGE)


# ---- through main(): flags do not bypass, scene option wiring ------------------------

def _scene_file(tmp_path, body):
    path = tmp_path / "scene.yaml"
    path.write_text(f"scene: topsize_test\nlive:\n{body}", encoding="utf-8")
    return path


SCENE_ON = "  top_size_range_mm: {long: [330, 450], short: [230, 380]}\n"


def _run(det, tmp_path, monkeypatch, rec, *extra, scene_body=SCENE_ON):
    cap = live._capture(tmp_path)
    live._install_fake_autolabel(monkeypatch, det, rec)
    real_import = det.import_autolabel

    def patched(repo=None):
        pipeline, _cfg, worker, overlay = real_import(repo)
        return pipeline, (lambda scene=None: {}), worker, overlay

    monkeypatch.setattr(det, "import_autolabel", patched)
    scene = ["--scene-config", str(_scene_file(tmp_path, scene_body))] if scene_body is not None else []
    code = det.main(live._argv(cap, tmp_path, "--no-depth-edge", *scene, *extra))
    return code


def _reject(tmp_path):
    return json.loads((tmp_path / "ov" / "cap_1_autolabel_v2_reject.json").read_text(encoding="utf-8"))


FLAGS = ("--allow-needs-review", "--allow-cut-blocked")


def test_half_flap_rejected_even_with_both_override_flags(det, tmp_path, monkeypatch):
    code = _run(det, tmp_path, monkeypatch, _rec(372.0, 144.0), *FLAGS)
    assert code == 1
    reject = _reject(tmp_path)
    assert reject["stage"] == "top_size" and reject["retryable"] is True
    assert reject["reasons"][0].startswith("top_size_out_of_range(long=372.0mm,short=144.0mm,")
    assert reject["details"]["top_size_check"]["long_mm"] == pytest.approx(372.0, abs=0.01)
    assert not (tmp_path / "out" / "seam.json").exists()


def test_half_flap_rejected_without_flags(det, tmp_path, monkeypatch):
    assert _run(det, tmp_path, monkeypatch, _rec(372.0, 144.0)) == 1
    assert _reject(tmp_path)["stage"] == "top_size"


def test_needs_review_record_still_rejected_by_size_with_flags(det, tmp_path, monkeypatch):
    rec = _rec(391.0, 117.0)
    rec["checks"]["needs_review"] = True
    assert _run(det, tmp_path, monkeypatch, rec, *FLAGS) == 1
    assert _reject(tmp_path)["stage"] == "top_size"


def test_in_range_accepted_and_recorded_in_the_seam_json(det, tmp_path, monkeypatch):
    out = tmp_path / "out" / "seam.json"
    assert _run(det, tmp_path, monkeypatch, _rec(394.5, 328.6, 30.0), *FLAGS) == 0
    check = json.loads(out.read_text(encoding="utf-8"))["autolabel_v2"]["top_size_check"]
    assert check["pass"] is True and check["range_mm"] == RANGE
    assert check["long_mm"] == pytest.approx(394.5, abs=0.01)
    assert check["short_mm"] == pytest.approx(328.6, abs=0.01)


def test_unmeasurable_rejects_through_main(det, tmp_path, monkeypatch):
    rec = live._record()
    rec["top_face"]["corners_xyz_camera_m"] = None
    assert _run(det, tmp_path, monkeypatch, rec, *FLAGS, "--no-rect-check") == 1
    reject = _reject(tmp_path)
    assert reject["stage"] == "top_size" and reject["reasons"][0].startswith("top_size_unmeasurable(")
    assert reject["retryable"] is False


def test_gate_off_without_the_scene_option(det, tmp_path, monkeypatch):
    out = tmp_path / "out" / "seam.json"
    # no --scene-config at all, and a scene without the option: same old behaviour
    assert _run(det, tmp_path, monkeypatch, _rec(372.0, 144.0), *FLAGS, scene_body=None) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["autolabel_v2"]["top_size_check"] == {"enabled": False}
    out.unlink()
    assert _run(det, tmp_path / "b", monkeypatch, _rec(372.0, 144.0), *FLAGS,
                scene_body="  rect_tol_mm: null\n") == 0


# ---- scene loading ----------------------------------------------------------

def test_shipped_scenes():
    from box_seam_detection.core import scene as sc
    assert sc.load_scene("dual_arm_v1").top_size_range_mm == RANGE
    assert sc.load_scene("table_cell_v1").top_size_range_mm is None
    assert sc.builtin_default().top_size_range_mm is None


@pytest.mark.parametrize("value", [
    "[1, 2]", "{long: [330, 450]}", "{long: [330, 450], short: [230, 380], extra: 1}",
    "{long: [450, 330], short: [230, 380]}", "{long: [330, 330], short: [230, 380]}",
    "{long: [0, 450], short: [230, 380]}", "{long: [-5, 450], short: [230, 380]}",
    "{long: [330], short: [230, 380]}", "{long: [330, 450, 500], short: [230, 380]}",
    "{long: ['a', 450], short: [230, 380]}", "{long: [true, 450], short: [230, 380]}",
    "{long: [330, .inf], short: [230, 380]}", "{long: 330, short: [230, 380]}",
])
def test_invalid_scene_option_fails_clearly(tmp_path, value):
    from box_seam_detection.core import scene as sc
    with pytest.raises(sc.SceneError, match="top_size_range_mm"):
        sc.load_scene(_scene_file(tmp_path, f"  top_size_range_mm: {value}\n"))


def test_valid_scene_option_normalised_to_floats(tmp_path):
    from box_seam_detection.core import scene as sc
    scene = sc.load_scene(_scene_file(tmp_path, "  top_size_range_mm: {long: [330, 450], short: [230, 380]}\n"))
    assert scene.top_size_range_mm == RANGE
    assert all(isinstance(x, float) for k in RANGE for x in scene.top_size_range_mm[k])


# ---- smart retry ------------------------------------------------------------

def _size_reject(long_mm, short_mm):
    return {"status": "rejected", "stage": "top_size", "retryable": True,
            "reasons": [f"top_size_out_of_range(long={long_mm:.1f}mm,short={short_mm:.1f}mm,range=x)"],
            "details": {"top_size_check": {"long_mm": long_mm, "short_mm": short_mm, "pass": False}}}


def test_identical_top_size_rejects_stop_early(tmp_path, monkeypatch, capsys):
    outcomes = [{"reject": _size_reject(372.0, 144.0)}, {"reject": _size_reject(374.0, 146.0)}, "ok", "ok"]
    code, state = r4._orchestrate(live._load(live.DETECT, "autolabel_v2_detect_topsize_r"),
                                  tmp_path, monkeypatch, outcomes)
    assert code == 1 and state["detector"] == 2
    err = capsys.readouterr().err
    assert "[stop_early_same_failure]" in err and "outside the expected box size" in err


def test_different_top_size_rejects_keep_retrying(tmp_path, monkeypatch):
    outcomes = [{"reject": _size_reject(372.0, 144.0)}, {"reject": _size_reject(391.0, 117.0)}, "ok"]
    code, state = r4._orchestrate(live._load(live.DETECT, "autolabel_v2_detect_topsize_r2"),
                                  tmp_path, monkeypatch, outcomes)
    assert code == 0 and state["detector"] == 3


def test_top_size_signature_helpers():
    orch = live._load(live.ORCHESTRATOR, "capture_and_query_depth_topsize")
    sig = orch.top_size_signature(_size_reject(372.0, 144.0))
    assert sig == {"long_mm": 372.0, "short_mm": 144.0}
    assert orch.top_size_signature({"stage": "top_size", "reasons": ["top_size_unmeasurable(x)"]}) is None
    assert orch.top_size_signature({"stage": "rectangle", "reasons": []}) is None
    assert orch.same_top_size_failure(sig, {"long_mm": 379.0, "short_mm": 151.0}) is True
    assert orch.same_top_size_failure(sig, {"long_mm": 391.0, "short_mm": 117.0}) is False
    assert orch.same_top_size_failure(None, sig) is False
