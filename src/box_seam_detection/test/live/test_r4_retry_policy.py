"""R4 smart retry policy of ``capture_and_query_depth.py --autolabel-v2``.

A rectangle_inconsistent reject repeated the same way (same failing pair, diff
within 3 mm, corners within 5 px) stops early; flaky failures keep retrying;
policy "all" keeps the old behaviour. Subprocesses and the camera are faked.
"""

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("r4_retry_helpers",
                                               str(TESTS / "test_autolabel_v2_retry.py"))
retry = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(retry)
live = retry.live
det = live.det  # fixture re-export
W, H = live.W, live.H


@pytest.fixture()
def orch():
    return live._load(live.ORCHESTRATOR, "capture_and_query_depth")


RECT = {"back_left": [-0.10, -0.075, 0.5], "back_right": [0.10, -0.075, 0.5],
        "front_right": [0.10, 0.075, 0.5], "front_left": [-0.10, 0.075, 0.5]}
PX = {"back_left": [200.0, 150.0], "back_right": [400.0, 150.0],
      "front_right": [400.0, 300.0], "front_left": [200.0, 300.0]}


def _rotated_xyz(deg, long_m=0.2, short_m=0.15, z=0.5):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    out = {}
    for name, (x, y) in {"back_left": (-long_m / 2, -short_m / 2), "back_right": (long_m / 2, -short_m / 2),
                         "front_right": (long_m / 2, short_m / 2),
                         "front_left": (-long_m / 2, short_m / 2)}.items():
        out[name] = [c * x - s * y, s * x + c * y, z]
    return out


def _reject(a=6.0, b=0.1, front=194.0, tol=3.0, hint=None):
    rect = {"enabled": True, "tol_mm": tol,
            "lengths_mm": {"back": 200.0, "front": front, "left": 150.0, "right": 150.0},
            "diffs_mm": {"A": a, "B": b}, "pairs": {"A": ["back", "front"], "B": ["left", "right"]},
            "pass": False}
    diag = {"status": "rejected", "stage": "rectangle", "retryable": True,
            "reasons": [f"rectangle_inconsistent(A={a:.2f}mm,B={b:.2f}mm)"], "rectangle": rect}
    if hint:
        diag["operator_hint"] = {"suggestion": hint}
    return diag


def _top(px=None, xyz=None):
    return {"corners_px": px or PX, "corners_xyz_camera_m": xyz or RECT, "plane": None}


def _shift(px, dx):
    return {k: [v[0] + dx, v[1]] for k, v in px.items()}


def _orchestrate(det, tmp_path, monkeypatch, outcomes, *extra_argv):
    """``outcomes``: per detector call "ok", or {"reject": json, "top": top_face}."""
    mod = live._load(live.ORCHESTRATOR, "capture_and_query_depth")
    root = tmp_path / "root"
    caps = [retry._make_capture(root, f"cap_{i}") for i in range(1, len(outcomes) + 3)]
    script = tmp_path / "autolabel_v2_detect.py"
    script.write_text("# fake\n", encoding="utf-8")
    sam3 = tmp_path / "fake_python"
    sam3.write_text("", encoding="utf-8")

    class _Probe:
        returncode = 0
        stdout = ""

    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **k: _Probe())
    state = {"recorder": 0, "detector": 0, "stamp": None}
    queue = list(outcomes)

    def fake_run_step(title, cmd, capture=False, check=True):
        if "rgbd_recorder" in title:
            cap = caps[state["recorder"]]
            state["recorder"] += 1
            return retry._recorder_stdout(cap)
        if "autolabel_v2" in title:
            state["detector"] += 1
            flag = lambda name: cmd[cmd.index(name) + 1]  # noqa: E731
            cid = json.loads(Path(flag("--capture-manifest")).read_text())["capture_id"]
            outcome = queue.pop(0)
            if outcome == "ok":
                Path(flag("--out-seam-json")).write_text(
                    json.dumps(retry._good_doc(det, flag("--image"), cid)), encoding="utf-8")
                return 0
            ov = Path(flag("--overlay-dir"))
            ov.mkdir(parents=True, exist_ok=True)
            (ov / f"{cid}_autolabel_v2_reject.json").write_text(
                json.dumps({"capture_id": cid, **outcome["reject"]}))
            if outcome.get("top") is not None:
                (Path(flag("--work-dir")) / f"{cid}_autolabel_v2_record.json").write_text(
                    json.dumps({"top_face": outcome["top"]}))
            return 1
        if "query_depth" in title:
            Path(cmd[cmd.index("--output-json") + 1]).write_text("{}", encoding="utf-8")
            return None
        if "base_link" in title:
            Path(cmd[cmd.index("--planner-output-json") + 1]).write_text("{}", encoding="utf-8")
            return None
        raise AssertionError(title)

    monkeypatch.setattr(mod, "run_step", fake_run_step)
    monkeypatch.setattr(mod, "add_side_cutting_points", lambda path: None)
    monkeypatch.setattr(mod, "validate_planner_snapshot", lambda *a: None)
    monkeypatch.setattr(mod, "stamp_detector_method",
                        lambda path, method, extra=None: state.update(stamp=extra))
    code = mod.main([
        "--downstream-inprocess", "off", "--autolabel-v2", "--autolabel-v2-script", str(script),
        "--autolabel-v2-sam3-python", str(sam3), "--calib-file", str(live._calib(tmp_path)),
        "--output-dir", str(root), "--autolabel-v2-average", "1", "--no-side-seed", *extra_argv])
    return code, state


def _fail(a=6.0, top=None, **kw):
    return {"reject": _reject(a=a, **kw), "top": top or _top()}


# ------------------------------------------------------------ orchestrator

def test_identical_repeated_failure_stops_after_2(det, tmp_path, monkeypatch, capsys):
    code, state = _orchestrate(det, tmp_path, monkeypatch,
                               [_fail(6.0), _fail(7.0, top=_top(_shift(PX, 3))), "ok", "ok"])
    assert code == 1 and state["detector"] == 2 and state["recorder"] == 2
    err = capsys.readouterr().err
    assert "2/4" in err and "相同方式" in err and "operator hint" in err
    assert "damaged/irregular" in err and "[stop_early_same_failure]" in err


def test_flaky_different_failure_uses_all_attempts(det, tmp_path, monkeypatch, capsys):
    outcomes = [_fail(6.0), _fail(a=0.1, b=7.0), _fail(6.0), _fail(10.0)]
    code, state = _orchestrate(det, tmp_path, monkeypatch, outcomes)
    assert code == 1 and state["detector"] == 4
    assert "4/4 次嘗試全部被拒絕" in capsys.readouterr().err


def test_flaky_much_smaller_diff_or_moved_corners_keep_retrying(det, tmp_path, monkeypatch):
    outcomes = [_fail(9.0), _fail(4.0), _fail(4.0, top=_top(_shift(PX, 12))), "ok"]
    code, state = _orchestrate(det, tmp_path, monkeypatch, outcomes)
    assert code == 0 and state["detector"] == 4
    summary = state["stamp"]["autolabel_v2_attempts"]
    assert summary["retry_policy"] == "smart" and summary["used"] == 4
    decisions = [e.get("decision") for e in summary["attempts"]]
    assert decisions == ["retry", "retry", "retry", None]


def test_success_on_attempt_2_unchanged(det, tmp_path, monkeypatch):
    code, state = _orchestrate(det, tmp_path, monkeypatch, [_fail(6.0), "ok"])
    assert code == 0 and state["detector"] == 2
    first, second = state["stamp"]["autolabel_v2_attempts"]["attempts"]
    assert first["decision"] == "retry" and first["failure_signature"]["pairs"] == ["A"]
    assert second["outcome"] == "accepted"


def test_policy_all_is_old_behaviour(det, tmp_path, monkeypatch, capsys):
    code, state = _orchestrate(det, tmp_path, monkeypatch, [_fail(6.0)] * 4,
                               "--autolabel-v2-retry-policy", "all")
    assert code == 1 and state["detector"] == 4
    assert "4/4 次嘗試全部被拒絕" in capsys.readouterr().err


def test_non_retryable_reject_unchanged(det, tmp_path, monkeypatch, capsys):
    bad = {"reject": {"status": "rejected", "stage": "record", "reasons": ["wrong_face_suspect"],
                      "retryable": False}}
    code, state = _orchestrate(det, tmp_path, monkeypatch, [bad, "ok"])
    assert code == 1 and state["detector"] == 1 and state["recorder"] == 1
    assert "拒絕此 frame" in capsys.readouterr().err


# ------------------------------------------------------------ helpers / hints

@pytest.mark.parametrize("deg", [0.0, 10.0, 30.0, 40.0])
def test_yaw_estimate_3d(orch, deg):
    assert orch.estimate_top_face_yaw_deg(_top(xyz=_rotated_xyz(deg))) == pytest.approx(deg, abs=0.1)


def test_yaw_folds_to_0_45(orch):
    assert orch.estimate_top_face_yaw_deg(_top(xyz=_rotated_xyz(60.0))) == pytest.approx(30.0, abs=0.1)


def test_yaw_falls_back_to_pixels(orch):
    top = {"corners_px": PX}
    assert orch.estimate_top_face_yaw_deg(top) == pytest.approx(0.0, abs=0.1)
    assert orch.estimate_top_face_yaw_deg({}) is None


def test_hint_high_yaw(orch):
    top = _top(xyz=_rotated_xyz(28.0))
    hints = orch.build_operator_hint(_reject(), top, True, (W, H))
    assert any("box rotated ~28" in h and "closer to square" in h for h in hints)
    assert not any("damaged" in h for h in hints)


def test_hint_image_edge(orch):
    top = _top(px=_shift(PX, -190))  # back_left x = 10
    hints = orch.build_operator_hint(_reject(), top, False, (W, H))
    assert any("near the image/table edge" in h for h in hints)


def test_hint_damaged_top_and_detector_hint(orch):
    hints = orch.build_operator_hint(_reject(a=8.0, hint="re-seat the box"), _top(), True, (W, H))
    assert any("damaged/irregular top (opposite sides differ 8 mm consistently)" in h for h in hints)
    assert any("re-seat the box" in h for h in hints)


def test_no_damaged_hint_when_not_consistent(orch):
    assert orch.build_operator_hint(_reject(), _top(), False, (W, H)) == []


def test_signature_same_rules(orch):
    sig = lambda *a, **k: orch.rectangle_failure_signature(_reject(*a, **k), _top())  # noqa: E731
    assert orch.same_rectangle_failure(sig(6.0), sig(8.9))
    assert not orch.same_rectangle_failure(sig(6.0), sig(9.5))
    assert orch.rectangle_failure_signature({"reasons": ["x"], "retryable": True}) is None


def test_signature_skewed(orch):
    def rej(dev):
        return {"reasons": [f"rectangle_skewed(max_dev={dev:.2f}deg)"], "retryable": True,
                "rectangle": {"tol_mm": 3.0, "diffs_mm": {"A": 0.5, "B": 0.4},
                              "max_corner_dev_deg": dev}}
    s1 = orch.rectangle_failure_signature(rej(6.0), _top())
    assert s1["pairs"] == ["skew"]
    assert orch.same_rectangle_failure(s1, orch.rectangle_failure_signature(rej(6.8), _top()))
    assert not orch.same_rectangle_failure(s1, orch.rectangle_failure_signature(rej(7.5), _top()))
    incons = orch.rectangle_failure_signature(_reject(6.0), _top())
    assert not orch.same_rectangle_failure(s1, incons)


def test_hint_damaged_top_skew_in_degrees(orch):
    rej = {"reasons": ["rectangle_skewed(max_dev=6.20deg)"], "retryable": True,
           "rectangle": {"tol_mm": 3.0, "diffs_mm": {"A": 0.5, "B": 0.4},
                         "max_corner_dev_deg": 6.2}}
    hints = orch.build_operator_hint(rej, _top(), True, (W, H))
    assert any("corners 6 deg off square consistently" in h for h in hints)
    assert not any(" mm consistently" in h for h in hints)
