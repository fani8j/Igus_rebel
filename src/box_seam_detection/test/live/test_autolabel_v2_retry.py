"""Rectangle-consistency gate (live.detector) and the live automatic
recapture loop of live.orchestrator (``run_detection`` / ``main``).

Detector: the 3-D top-face side lengths (rule names) must form a rectangle
(pair A back/front, pair B left/right, |diff| <= --rect-tol-mm) else exit 1
with a retryable reject JSON. Orchestrator: in LIVE mode a retryable
rejection re-acquires a NEW capture and re-runs the detector (up to
--autolabel-v2-attempts); any other outcome stops as before; offline never
retries. Subprocesses, SAM3 and the camera are faked.
"""

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("autolabel_v2_live_helpers",
                                               str(TESTS / "test_autolabel_v2_live.py"))
live = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(live)

det = live.det  # fixture re-export
RING = live.RING
W, H = live.W, live.H
FRAME_ID = live.FRAME_ID


def _skewed_xyz(front_short_mm=6.0):
    """Front edge shorter than the back edge by ``front_short_mm`` (a skewed quad)."""
    xyz = {k: list(v) for k, v in live.RECT_XYZ.items()}
    xyz["front_right"][0] -= front_short_mm / 1000.0
    return xyz


def _rec(xyz=None, **kwargs):
    rec = live._record(axis="bf", **kwargs)
    if xyz is not None:
        rec["top_face"]["corners_xyz_camera_m"] = xyz
    return rec


# ------------------------------------------------------------ detector gate

def test_rectangle_check_consistent_quad(det):
    result = det.rectangle_check(_rec(), 3.0)
    assert result["pass"] is True and result["enabled"] is True
    assert result["lengths_mm"] == pytest.approx(
        {"back": 200.0, "front": 200.0, "left": 150.0, "right": 150.0})
    assert result["diffs_mm"] == pytest.approx({"A": 0.0, "B": 0.0})


def test_rectangle_check_skewed_quad_raises_retryable(det):
    with pytest.raises(det.Reject) as info:
        det.rectangle_check(_rec(_skewed_xyz(6.0)), 3.0)
    error = info.value
    assert error.stage == "rectangle" and error.retryable is True
    assert error.reasons[0].startswith("rectangle_inconsistent(A=6.00mm,B=")
    assert error.details["lengths_mm"]["front"] == pytest.approx(194.0, abs=1e-3)
    # B: right edge is slightly longer (front_right moved in x only).
    right = 1000.0 * math.dist(_skewed_xyz(6.0)["back_right"], _skewed_xyz(6.0)["front_right"])
    assert error.details["diffs_mm"]["B"] == pytest.approx(right - 150.0, abs=1e-3)


def test_rectangle_check_tolerance_boundary(det):
    # 2.5 mm difference: passes at 3.0, fails at 2.0.
    assert det.rectangle_check(_rec(_skewed_xyz(2.5)), 3.0)["pass"] is True
    with pytest.raises(det.Reject):
        det.rectangle_check(_rec(_skewed_xyz(2.5)), 2.0)


def test_detector_accepts_consistent_quad_and_records_lengths(det, tmp_path, monkeypatch):
    cap = live._capture(tmp_path)
    live._install_fake_autolabel(monkeypatch, det, _rec())
    out = live._stale(tmp_path)
    assert det.main(live._argv(cap, tmp_path)) == 0
    rect = json.loads(out.read_text(encoding="utf-8"))["autolabel_v2"]["rect_check"]
    assert rect["enabled"] is True and rect["pass"] is True and rect["tol_mm"] == 3.0
    assert set(rect["lengths_mm"]) == {"back", "front", "left", "right"}
    assert set(rect["diffs_mm"]) == {"A", "B"}


def test_detector_skewed_quad_exit_1_retryable_reject_json(det, tmp_path, monkeypatch):
    cap = live._capture(tmp_path)
    live._install_fake_autolabel(monkeypatch, det, _rec(_skewed_xyz(6.0)))
    out = live._stale(tmp_path)
    assert det.main(live._argv(cap, tmp_path)) == 1
    assert not out.exists()
    diag = json.loads((tmp_path / "ov" / "cap_1_autolabel_v2_reject.json").read_text())
    assert diag["retryable"] is True and diag["stage"] == "rectangle"
    assert diag["reasons"][0].startswith("rectangle_inconsistent(A=6.00mm,B=")
    assert diag["rectangle"]["lengths_mm"]["back"] == pytest.approx(200.0)
    assert diag["rectangle"]["lengths_mm"]["front"] == pytest.approx(194.0)
    assert set(diag["rectangle"]["lengths_mm"]) == {"back", "front", "left", "right"}


def test_detector_custom_tolerance_accepts_skew(det, tmp_path, monkeypatch):
    cap = live._capture(tmp_path)
    live._install_fake_autolabel(monkeypatch, det, _rec(_skewed_xyz(6.0)))
    out = live._stale(tmp_path)
    assert det.main(live._argv(cap, tmp_path, "--rect-tol-mm", "8")) == 0
    rect = json.loads(out.read_text(encoding="utf-8"))["autolabel_v2"]["rect_check"]
    assert rect["tol_mm"] == 8.0 and rect["diffs_mm"]["A"] == pytest.approx(6.0)


@pytest.mark.parametrize("tol", ["0", "-1", "20.01", "nan", "inf"])
def test_detector_invalid_rect_tol_exit_2(det, tmp_path, monkeypatch, tol):
    cap = live._capture(tmp_path)
    calls = live._install_fake_autolabel(monkeypatch, det, _rec())
    out = live._stale(tmp_path)
    assert det.main(live._argv(cap, tmp_path, "--rect-tol-mm", tol)) == 2
    assert not out.exists() and "started" not in calls


def test_detector_rect_tol_upper_bound_accepted(det, tmp_path, monkeypatch):
    cap = live._capture(tmp_path)
    live._install_fake_autolabel(monkeypatch, det, _rec())
    assert det.main(live._argv(cap, tmp_path, "--rect-tol-mm", "20")) == 0


def test_detector_no_rect_check_accepts_skew(det, tmp_path, monkeypatch):
    cap = live._capture(tmp_path)
    live._install_fake_autolabel(monkeypatch, det, _rec(_skewed_xyz(9.0)))
    out = live._stale(tmp_path)
    assert det.main(live._argv(cap, tmp_path, "--no-rect-check")) == 0
    assert json.loads(out.read_text())["autolabel_v2"]["rect_check"] == {"enabled": False}


@pytest.mark.parametrize("xyz", [
    None,
    {k: [0.0, 0.0, 0.5] for k in RING},                      # degenerate
    {**live.RECT_XYZ, "front_left": [float("nan"), 0.0, 0.5]},
    {k: v for k, v in live.RECT_XYZ.items() if k != "back_right"},
])
def test_detector_missing_xyz_unverifiable_not_retryable(det, tmp_path, monkeypatch, xyz):
    cap = live._capture(tmp_path)
    rec = _rec()
    rec["top_face"]["corners_xyz_camera_m"] = xyz
    live._install_fake_autolabel(monkeypatch, det, rec)
    out = live._stale(tmp_path)
    assert det.main(live._argv(cap, tmp_path)) == 1
    assert not out.exists()
    diag = json.loads((tmp_path / "ov" / "cap_1_autolabel_v2_reject.json").read_text())
    assert diag["retryable"] is False
    assert diag["reasons"][0].startswith("rectangle_unverifiable")


def test_other_rejections_are_not_retryable(det, tmp_path, monkeypatch):
    cap = live._capture(tmp_path)
    live._install_fake_autolabel(monkeypatch, det, _rec(wrong_face=True))
    assert det.main(live._argv(cap, tmp_path)) == 1
    diag = json.loads((tmp_path / "ov" / "cap_1_autolabel_v2_reject.json").read_text())
    assert diag["retryable"] is False


# ------------------------------------------------------------ orchestrator

def _make_capture(root, cid):
    """A fake rgbd_recorder capture <root>/{rgb,pointcloud,captures}/<cid>.*"""
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    image = root / "rgb" / f"{cid}.png"
    pcd = root / "pointcloud" / f"{cid}.pcd"
    manifest_path = root / "captures" / f"{cid}.json"
    for path in (image, pcd, manifest_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(image), np.zeros((H, W, 3), dtype=np.uint8))
    pcd.write_bytes(b"fake-pcd")
    manifest = {
        "schema": "rgbd_capture/v1", "capture_id": cid,
        "image": {"path": str(image), "width": W, "height": H,
                  "header": {"frame_id": FRAME_ID, "stamp_ns": 1000}},
        "pointcloud": {"path": str(pcd), "header": {"frame_id": FRAME_ID, "stamp_ns": 1001}},
        "camera_info": {"header": {"frame_id": FRAME_ID}, "width": W, "height": H},
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return {"image": image, "pcd": pcd, "manifest": manifest_path, "id": cid}


def _recorder_stdout(capture):
    return (f"[INFO] 已儲存影像: {capture['image']}\n"
            f"[INFO] 已儲存點雲: {capture['pcd']}\n"
            f"[INFO] 已儲存 capture manifest: {capture['manifest']}\n")


def _good_doc(det, image, cid):
    rec = _rec()
    seam = det.published_seam_line(rec)
    rel = det.contract_relabel(rec["top_face"]["corners_px"], seam["seam_left"],
                               seam["seam_right"])
    return det.build_canonical_seam_doc(rec, str(image), W, H, cid, seam, rel,
                                        "rec.json", {}, False,
                                        cut_permission={"status": "allowed", "reasons": []})


def _orchestrate(det, tmp_path, monkeypatch, outcomes, *extra_argv, offline=False,
                 captures=None, average="1"):
    """Drive main() with fake subprocesses.

    ``outcomes``: per detector call, one of "ok", "retry" (exit 1 retryable
    rectangle_inconsistent), "reject" (exit 1 non-retryable), 2 (exit 2).
    Returns (exit_code, state) — state records recorder calls, detector
    commands and the query_depth command (the run stops at step 3).
    """
    mod = live._load(live.ORCHESTRATOR, "capture_and_query_depth")
    root = tmp_path / "root"
    n = len(outcomes) + 2
    captures = captures or [_make_capture(root, f"cap_{i}") for i in range(1, n + 1)]
    script = tmp_path / "autolabel_v2_detect.py"
    script.write_text("# fake\n", encoding="utf-8")
    sam3 = tmp_path / "fake_python"
    sam3.write_text("", encoding="utf-8")

    class _Probe:
        returncode = 0
        stdout = ""

    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **k: _Probe())
    state = {"recorder": [], "detector": [], "titles": [], "query": None}
    queue = list(outcomes)

    def fake_run_step(title, cmd, capture=False, check=True):
        state["titles"].append(title)
        if "rgbd_recorder" in title:
            assert capture is True and "--once" in cmd
            cap = captures[len(state["recorder"])]
            state["recorder"].append(cap["id"])
            return _recorder_stdout(cap)
        if "autolabel_v2" in title:
            assert check is False
            state["detector"].append(cmd)
            flag = lambda name: cmd[cmd.index(name) + 1]  # noqa: E731
            manifest = json.loads(Path(flag("--capture-manifest")).read_text())
            cid = manifest["capture_id"]
            assert Path(flag("--image")) == Path(manifest["image"]["path"])
            assert Path(flag("--pcd")) == Path(manifest["pointcloud"]["path"])
            out = Path(flag("--out-seam-json"))
            ov = Path(flag("--overlay-dir"))
            outcome = queue.pop(0)
            if outcome == "ok":
                out.write_text(json.dumps(_good_doc(det, flag("--image"), cid)), encoding="utf-8")
                return 0
            if outcome == 2:
                return 2
            ov.mkdir(parents=True, exist_ok=True)
            retry = outcome == "retry"
            diag = {"status": "rejected", "capture_id": cid, "stage":
                    "rectangle" if retry else "record",
                    "reasons": ["rectangle_inconsistent(A=6.00mm,B=0.10mm)"] if retry
                    else ["wrong_face_suspect"], "retryable": retry}
            (ov / f"{cid}_autolabel_v2_reject.json").write_text(json.dumps(diag))
            # A stale-looking seam JSON left by a rejected attempt must never
            # be picked up downstream.
            out.write_text(json.dumps({"stale": cid}), encoding="utf-8")
            return 1
        if "query_depth" in title:
            state["query"] = cmd
            live._halt(mod, title)
        raise AssertionError(f"unexpected step {title}")

    monkeypatch.setattr(mod, "run_step", fake_run_step)
    argv = ["--downstream-inprocess", "off", "--autolabel-v2",
            "--autolabel-v2-script", str(script),
            "--autolabel-v2-sam3-python", str(sam3),
            "--calib-file", str(live._calib(tmp_path)),
            "--output-dir", str(root), *extra_argv]
    if average is not None:
        # These tests cover single-capture (--once) retry mechanics; burst
        # averaging (the live default) is covered in test_autolabel_v2_average.py.
        argv += ["--autolabel-v2-average", str(average)]
    if offline:
        c = captures[0]
        argv += ["--from-image", str(c["image"]), "--pcd", str(c["pcd"]),
                 "--capture-manifest", str(c["manifest"])]
    code = mod.main(argv)
    if code == live.STOP_CODE:
        code = "stopped_at_query"
    return code, state, captures


def test_live_retry_succeeds_on_attempt_2_only_new_capture_flows(det, tmp_path, monkeypatch):
    code, state, caps = _orchestrate(det, tmp_path, monkeypatch, ["retry", "ok"])
    assert code == "stopped_at_query"
    assert state["recorder"] == ["cap_1", "cap_2"]
    assert len(state["detector"]) == 2
    first, second = state["detector"]
    flag = lambda cmd, name: cmd[cmd.index(name) + 1]  # noqa: E731
    # Per-attempt outputs never collide: each capture has its own pipeline dir.
    assert flag(first, "--work-dir") != flag(second, "--work-dir")
    assert "/cap_1/" in flag(first, "--work-dir") and "/cap_2/" in flag(second, "--work-dir")
    # The rejected attempt's files are kept for debugging.
    first_ov = Path(flag(first, "--overlay-dir"))
    assert (first_ov / "cap_1_autolabel_v2_reject.json").is_file()
    # Only the successful capture flows downstream.
    q = state["query"]
    assert flag(q, "--seam-json") == flag(second, "--out-seam-json")
    assert flag(q, "--pcd") == str(caps[1]["pcd"])
    assert flag(q, "--capture-manifest") == str(caps[1]["manifest"])
    assert "cap_1" not in " ".join(q)
    # Titles carry the attempt count.
    assert any("第 2/4 次" in t for t in state["titles"] if "autolabel_v2" in t)


def test_live_stops_after_n_attempts_exit_1(det, tmp_path, monkeypatch, capsys):
    code, state, _ = _orchestrate(det, tmp_path, monkeypatch, ["retry", "retry"],
                                  "--autolabel-v2-attempts", "2")
    assert code == 1
    assert state["recorder"] == ["cap_1", "cap_2"] and len(state["detector"]) == 2
    assert state["query"] is None
    err = capsys.readouterr().err
    assert "2/2 次嘗試全部被拒絕" in err
    assert "#1 cap_1: rectangle_inconsistent" in err and "#2 cap_2: rectangle_inconsistent" in err


def test_live_default_four_attempts(det, tmp_path, monkeypatch):
    # default = 1 capture + 3 retries
    code, state, _ = _orchestrate(det, tmp_path, monkeypatch, ["retry"] * 4)
    assert code == 1 and state["recorder"] == ["cap_1", "cap_2", "cap_3", "cap_4"]


def test_live_no_retry_on_non_retryable_rejection(det, tmp_path, monkeypatch):
    code, state, _ = _orchestrate(det, tmp_path, monkeypatch, ["reject"])
    assert code == 1
    assert state["recorder"] == ["cap_1"] and len(state["detector"]) == 1
    assert state["query"] is None


def test_live_no_retry_after_retry_then_non_retryable(det, tmp_path, monkeypatch):
    code, state, _ = _orchestrate(det, tmp_path, monkeypatch, ["retry", "reject"])
    assert code == 1 and state["recorder"] == ["cap_1", "cap_2"]


def test_live_no_retry_on_exit_2(det, tmp_path, monkeypatch):
    code, state, _ = _orchestrate(det, tmp_path, monkeypatch, [2])
    assert code == 2
    assert state["recorder"] == ["cap_1"] and len(state["detector"]) == 1


def test_live_attempts_1_disables_retry(det, tmp_path, monkeypatch):
    code, state, _ = _orchestrate(det, tmp_path, monkeypatch, ["retry"],
                                  "--autolabel-v2-attempts", "1")
    assert code == 1 and state["recorder"] == ["cap_1"]


def test_live_same_capture_twice_is_refused(det, tmp_path, monkeypatch, capsys):
    cap = _make_capture(tmp_path / "root", "cap_1")
    code, state, _ = _orchestrate(det, tmp_path, monkeypatch, ["retry", "ok"],
                                  captures=[cap, cap, cap])
    assert code == 1 and len(state["detector"]) == 1
    assert "沒有產生新的 capture" in capsys.readouterr().err


def test_offline_never_retries(det, tmp_path, monkeypatch):
    code, state, _ = _orchestrate(det, tmp_path, monkeypatch, ["retry", "ok"],
                                  "--autolabel-v2-attempts", "5", offline=True)
    assert code == 1
    assert state["recorder"] == [] and len(state["detector"]) == 1


def test_rect_options_forwarded_to_detector(det, tmp_path, monkeypatch):
    code, state, _ = _orchestrate(det, tmp_path, monkeypatch, ["ok"], "--rect-tol-mm", "4.5")
    cmd = state["detector"][0]
    assert cmd[cmd.index("--rect-tol-mm") + 1] == "4.5" and "--no-rect-check" not in cmd
    code, state, _ = _orchestrate(det, tmp_path / "b", monkeypatch, ["ok"], "--no-rect-check")
    cmd = state["detector"][0]
    assert "--no-rect-check" in cmd and "--rect-tol-mm" not in cmd
    assert "--no-depth-edge" not in cmd


def test_no_depth_edge_forwarded_to_detector(det, tmp_path, monkeypatch):
    code, state, _ = _orchestrate(det, tmp_path, monkeypatch, ["ok"], "--no-depth-edge")
    assert "--no-depth-edge" in state["detector"][0]


@pytest.mark.parametrize("extra, message", [
    (["--autolabel-v2", "--autolabel-v2-attempts", "0"], "--autolabel-v2-attempts"),
    (["--autolabel-v2", "--autolabel-v2-attempts", "11"], "--autolabel-v2-attempts"),
    (["--autolabel-v2", "--rect-tol-mm", "0"], "--rect-tol-mm"),
    (["--autolabel-v2", "--rect-tol-mm", "25"], "--rect-tol-mm"),
    (["--autolabel-v2", "--rect-tol-mm", "3", "--no-rect-check"], "不能一起用"),
])
def test_orchestrator_option_validation_exit_2(extra, message):
    proc = live._run_orchestrator(*extra)
    assert proc.returncode == 2, proc.stdout
    assert message in proc.stdout


def test_attempts_summary_stamped_into_planner_snapshot(tmp_path):
    mod = live._load(live.ORCHESTRATOR, "capture_and_query_depth")
    snap = tmp_path / "planner.json"
    snap.write_text(json.dumps({"schema": "box_h_cut_snapshot/v1", "keep": 1}))
    summary = {"max_attempts": 3, "used": 2, "attempts": [{"attempt": 1}, {"attempt": 2}]}
    out = mod.stamp_detector_method(snap, "autolabel_v2_sam3_fold_midline",
                                    {"autolabel_v2_attempts": summary})
    assert out["autolabel_v2_attempts"] == summary and out["keep"] == 1
    assert out["detector_method"] == "autolabel_v2_sam3_fold_midline"
    with pytest.raises(ValueError):
        mod.stamp_detector_method(snap, "x", {"keep": 2})
