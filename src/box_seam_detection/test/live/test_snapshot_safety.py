"""Fail-closed snapshot rules of live.orchestrator.run_detection (DESIGN.md):

1. a non-dry run removes the existing snapshot (and the side-seed quarantine file) first;
2. the snapshot is published only for an accepted detection whose cut permission is allowed
   (or overridden by allow_cut_blocked) -- and it carries the ``cut_permission`` block;
3. acquisition_stamp_ns is the colour image stamp, a seed failure quarantines;
4. exceptions of next_capture propagate (snapshot already gone); dry_run never touches the path.
Subprocess steps, SAM3 and the camera are faked (fake run_step, like test_side_face_seed)."""

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))
import test_autolabel_v2_live as live  # noqa: E402
import test_side_face_seed as sf  # noqa: E402

orch = importlib.import_module(live.ORCHESTRATOR)
det = importlib.import_module(live.DETECT)

ALLOWED = {"status": "allowed", "reasons": []}
BLOCKED = {"status": "blocked", "reasons": [{"code": "height_mismatch", "retryable": False, "detail": "x"}]}


class _Probe:
    returncode = 0
    stdout = ""


def _setup(tmp_path, monkeypatch, cut_permission=ALLOWED, detector_exit=0, seed_exit="real"):
    cap = live._capture(tmp_path)
    script = tmp_path / "autolabel_v2_detect.py"
    script.write_text("# fake\n", encoding="utf-8")
    calib = tmp_path / "test.calib"
    calib.write_text(sf.CALIB_TEXT, encoding="utf-8")
    real_run = subprocess.run

    def fake_subprocess_run(cmd, *a, **k):
        if len(cmd) > 2 and str(cmd[2]).endswith("live.side_seed"):
            if seed_exit == "real":
                return real_run(cmd, *a, **k)

            class _R:
                returncode = seed_exit
            return _R()
        return _Probe()

    monkeypatch.setattr(orch.subprocess, "run", fake_subprocess_run)

    def fake_run_step(title, cmd, capture=False, check=True):
        if "autolabel_v2" in title:
            out = Path(cmd[cmd.index("--out-seam-json") + 1])
            if detector_exit != 0:
                ov = Path(cmd[cmd.index("--overlay-dir") + 1])
                ov.mkdir(parents=True, exist_ok=True)
                (ov / "cap_1_autolabel_v2_reject.json").write_text(json.dumps({
                    "capture_id": "cap_1", "status": "rejected", "stage": "record",
                    "reasons": ["wrong_face_suspect"], "retryable": False}))
                return detector_exit
            rec = live._record(axis="bf")
            seam = det.published_seam_line(rec)
            rel = det.contract_relabel(rec["top_face"]["corners_px"], seam["seam_left"], seam["seam_right"])
            doc = det.build_canonical_seam_doc(rec, str(cap["image"]), live.W, live.H, "cap_1", seam, rel,
                                               "rec.json", {}, False, cut_permission=cut_permission)
            out.write_text(json.dumps(doc), encoding="utf-8")
        elif "query_depth" in title:
            Path(cmd[cmd.index("--output-json") + 1]).write_text("{}", encoding="utf-8")
        elif "base_link" in title:
            planner = Path(cmd[cmd.index("--planner-output-json") + 1])
            snap, _ = sf._box(rot=sf._rot(yaw_deg=-62.0, roll_deg=1.4))
            planner.write_text(json.dumps(snap), encoding="utf-8")
        else:
            raise AssertionError(title)
        return None

    monkeypatch.setattr(orch, "run_step", fake_run_step)
    files = orch.CaptureFiles(capture_id="cap_1", image_path=cap["image"], pcd_path=cap["pcd"],
                              manifest_path=cap["manifest"], stamp_ns=1000)
    snapshot = tmp_path / "exec" / "seam_3d_base_link.json"
    snapshot.parent.mkdir()
    cfg = orch.DetectConfig(
        max_attempts=1, live=False, autolabel_v2_server="off", sam3_python=str(cap["sam3_python"]),
        calib_file=str(calib), output_root=tmp_path / "root", snapshot_path=snapshot,
        downstream_inprocess=False, driver_extrinsic="recorded", detector_script=str(script))
    return cfg, files, snapshot


def _stale(snapshot):
    snapshot.write_text('{"stale": true}', encoding="utf-8")
    orch.side_seed_rejected_path(snapshot).write_text('{"stale": true}', encoding="utf-8")


def test_accepted_publishes_snapshot_with_cut_permission(tmp_path, monkeypatch):
    cfg, files, snapshot = _setup(tmp_path, monkeypatch)
    _stale(snapshot)
    result = orch.run_detection(lambda attempt: files, cfg)
    assert result.success and result.status == "accepted" and result.exit_code == 0
    assert result.snapshot_path == str(snapshot) and result.capture_id == "cap_1"
    published = json.loads(snapshot.read_text(encoding="utf-8"))
    assert published == result.snapshot
    assert published["cut_permission"] == ALLOWED
    assert published["acquisition_stamp_ns"] == 1000 == published["side_tape_seed"]["source"]["acquisition_stamp_ns"]
    assert published["side_tape_seed"]["source"]["snapshot_id"] == published["snapshot_id"]
    assert not orch.side_seed_rejected_path(snapshot).exists()          # stale quarantine file removed
    assert result.overlay_path is None or result.overlay_path.endswith("cap_1_autolabel_v2.png")
    assert [a["outcome"] for a in result.attempts] == ["accepted"]
    assert not list(snapshot.parent.glob(".*.tmp"))                      # atomic write left no temp file


def test_rejection_removes_the_old_snapshot_and_writes_nothing(tmp_path, monkeypatch):
    cfg, files, snapshot = _setup(tmp_path, monkeypatch, detector_exit=1)
    _stale(snapshot)
    result = orch.run_detection(lambda attempt: files, cfg)
    assert not result.success and result.status == "rejected" and result.exit_code == 1
    assert "wrong_face_suspect" in result.reasons
    assert not snapshot.exists() and not orch.side_seed_rejected_path(snapshot).exists()


def test_detector_prerequisite_failure_is_an_error_result(tmp_path, monkeypatch):
    cfg, files, snapshot = _setup(tmp_path, monkeypatch, detector_exit=2)
    _stale(snapshot)
    result = orch.run_detection(lambda attempt: files, cfg)
    assert not result.success and result.status == "error" and result.exit_code == 2
    assert not snapshot.exists()


def test_blocked_cut_is_rejected_without_the_override(tmp_path, monkeypatch):
    cfg, files, snapshot = _setup(tmp_path, monkeypatch, cut_permission=BLOCKED)
    _stale(snapshot)
    result = orch.run_detection(lambda attempt: files, cfg)
    assert not result.success and result.status == "blocked" and result.exit_code == 1
    assert result.cut_permission == BLOCKED and "cut_blocked(height_mismatch: x)" in result.reasons
    assert not snapshot.exists()


def test_missing_cut_permission_is_blocked(tmp_path, monkeypatch):
    cfg, files, snapshot = _setup(tmp_path, monkeypatch, cut_permission=None)
    result = orch.run_detection(lambda attempt: files, cfg)
    assert result.status == "blocked" and not snapshot.exists()


def test_override_publishes_with_the_override_recorded(tmp_path, monkeypatch):
    cfg, files, snapshot = _setup(tmp_path, monkeypatch, cut_permission={**BLOCKED, "override": "allow_cut_blocked"})
    cfg.allow_cut_blocked = True
    result = orch.run_detection(lambda attempt: files, cfg)
    assert result.success
    published = json.loads(snapshot.read_text(encoding="utf-8"))
    assert published["cut_permission"]["status"] == "blocked"
    assert published["cut_permission"]["override"] == "allow_cut_blocked"


@pytest.mark.parametrize("seed_exit", [1, 2])
def test_seed_failure_quarantines_and_leaves_no_snapshot(tmp_path, monkeypatch, seed_exit):
    cfg, files, snapshot = _setup(tmp_path, monkeypatch, seed_exit=seed_exit)
    _stale(snapshot)
    result = orch.run_detection(lambda attempt: files, cfg)
    assert not result.success and result.exit_code == seed_exit and result.status == "error"
    assert not snapshot.exists()
    quarantined = json.loads(orch.side_seed_rejected_path(snapshot).read_text(encoding="utf-8"))
    assert quarantined["schema"] == "box_h_cut_snapshot/v1" and "side_tape_seed" not in quarantined
    assert quarantined["cut_permission"] == ALLOWED


def test_next_capture_exception_propagates_and_snapshot_is_gone(tmp_path, monkeypatch):
    cfg, files, snapshot = _setup(tmp_path, monkeypatch)
    _stale(snapshot)

    class CaptureTimeout(Exception):
        pass

    def boom(attempt):
        raise CaptureTimeout("no frames")

    with pytest.raises(CaptureTimeout):
        orch.run_detection(boom, cfg)
    assert not snapshot.exists() and not orch.side_seed_rejected_path(snapshot).exists()


def test_dry_run_never_touches_the_snapshot_path(tmp_path, monkeypatch):
    cfg, files, snapshot = _setup(tmp_path, monkeypatch)
    cfg.dry_run = True
    snapshot.write_text('{"keep": 1}', encoding="utf-8")
    result = orch.run_detection(lambda attempt: files, cfg)
    assert result.success and result.snapshot["cut_permission"] == ALLOWED
    assert json.loads(snapshot.read_text(encoding="utf-8")) == {"keep": 1}
    assert result.snapshot_path is None                        # nothing was written to snapshot_path
    assert Path(result.staged_snapshot_path).is_file()


def test_stamp_mismatch_between_capture_files_and_manifest_is_refused(tmp_path, monkeypatch):
    cfg, files, snapshot = _setup(tmp_path, monkeypatch)
    files.stamp_ns = 999
    result = orch.run_detection(lambda attempt: files, cfg)
    assert not result.success and "stamp" in result.reason and not snapshot.exists()


def test_snapshot_removal_failure_is_an_error(tmp_path, monkeypatch):
    cfg, files, snapshot = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(orch, "remove_snapshot", lambda path: PermissionError("read-only"))
    result = orch.run_detection(lambda attempt: files, cfg)
    assert not result.success and result.status == "error" and "stale snapshot" in result.reason


def test_cli_without_planner_output_keeps_the_snapshot_in_the_pipeline_dir(tmp_path, monkeypatch):
    cfg, files, _ = _setup(tmp_path, monkeypatch)
    cfg.snapshot_path = None
    result = orch.run_detection(lambda attempt: files, cfg)
    assert result.success and result.snapshot_path is None
    assert Path(result.staged_snapshot_path).parent == Path(result.pipeline_dir)
    assert Path(result.staged_snapshot_path).name == "planner_snapshot.json"


def test_recorder_wrapper_parses_the_recorder_output(tmp_path):
    rec = importlib.import_module("box_seam_detection.live.recorder")
    cap = live._capture(tmp_path)
    out = (f"[INFO] 已儲存影像: {cap['image']}\n[INFO] 已儲存點雲: {cap['pcd']}\n"
           f"[INFO] 已儲存 capture manifest: {cap['manifest']}\n")
    assert rec.parse_recorder_output(out) == (cap["image"].resolve(), cap["pcd"].resolve(),
                                              cap["manifest"].resolve())
    assert rec.parse_recorder_output("nothing") is None
    cmd = rec.recorder_command(tmp_path, 12.5)
    assert cmd[1:3] == ["-m", "box_seam_detection.live.recorder"] and "--once" in cmd
    assert cmd[cmd.index("--timeout-sec") + 1] == "12.5"


def test_capture_once_returns_capture_files(tmp_path, monkeypatch):
    rec = importlib.import_module("box_seam_detection.live.recorder")
    cap = live._capture(tmp_path)

    class _Done:
        returncode = 0
        stdout = (f"已儲存影像: {cap['image']}\n已儲存點雲: {cap['pcd']}\n"
                  f"已儲存 capture manifest: {cap['manifest']}\n")

    monkeypatch.setattr(rec.subprocess, "run", lambda cmd, **k: _Done())
    files = rec.capture_once(tmp_path, 1, 5.0, echo=False)
    assert (files.capture_id, files.stamp_ns) == ("cap_1", 1000)
    assert files.image_path == cap["image"].resolve()

    class _Timeout(_Done):
        returncode = rec.EXIT_TIMEOUT

    monkeypatch.setattr(rec.subprocess, "run", lambda cmd, **k: _Timeout())
    with pytest.raises(rec.RecorderTimeout):
        rec.capture_once(tmp_path, 1, 5.0, echo=False)


# ----------------------------------------------------------------- review fixes

def test_abort_check_before_each_attempt_and_before_publish(tmp_path, monkeypatch):
    cfg, files, snapshot = _setup(tmp_path, monkeypatch)
    _stale(snapshot)
    # (a) cancelled before the first attempt: nothing captured, the stale snapshot stays removed
    called = []
    result = orch.run_detection(lambda attempt: called.append(attempt) or files, cfg,
                                abort_check=lambda: True)
    assert not result.success and result.status == "cancelled" and called == []
    assert not snapshot.exists() and not orch.side_seed_rejected_path(snapshot).exists()
    # (b) cancelled immediately before publishing: the finished staged snapshot exists, nothing published
    polls = []

    def check():
        polls.append(1)
        return len(polls) >= 2           # 1st poll (before attempt 1) False, 2nd (before publish) True

    result = orch.run_detection(lambda attempt: files, cfg, abort_check=check)
    assert result.status == "cancelled" and not result.success
    assert result.reason == "cancelled before publish" and len(polls) == 2
    assert not snapshot.exists()
    # (c) no hook: unchanged behaviour; a raising hook fails closed
    assert orch.run_detection(lambda attempt: files, cfg).success
    snapshot.unlink()

    def broken():
        raise RuntimeError("x")

    assert orch.run_detection(lambda attempt: files, cfg, abort_check=broken).status == "cancelled"
    assert not snapshot.exists()


def test_dry_run_seed_failure_quarantines_only_in_the_pipeline_dir(tmp_path, monkeypatch):
    cfg, files, snapshot = _setup(tmp_path, monkeypatch, seed_exit=1)
    cfg.dry_run = True
    production_quarantine = orch.side_seed_rejected_path(snapshot)
    production_quarantine.write_text('{"production": true}', encoding="utf-8")
    snapshot.write_text('{"keep": 1}', encoding="utf-8")
    result = orch.run_detection(lambda attempt: files, cfg)
    assert not result.success and result.snapshot_path is None
    assert json.loads(production_quarantine.read_text(encoding="utf-8")) == {"production": True}
    assert json.loads(snapshot.read_text(encoding="utf-8")) == {"keep": 1}
    inside = list(Path(result.pipeline_dir).glob("*.side_seed_rejected"))
    assert len(inside) == 1


def test_override_is_explicit_in_the_result(tmp_path, monkeypatch):
    cfg, files, snapshot = _setup(tmp_path, monkeypatch, cut_permission=BLOCKED)
    cfg.allow_cut_blocked = True
    result = orch.run_detection(lambda attempt: files, cfg)
    assert result.success and result.cut_summary == "overridden"
    assert result.cut_permission["status"] == "blocked"
    assert result.cut_permission["override"] == "allow_cut_blocked"
    cfg2, files2, snapshot2 = _setup(tmp_path / "b", monkeypatch)
    assert orch.run_detection(lambda attempt: files2, cfg2).cut_summary == "allowed"
    cfg3, files3, _ = _setup(tmp_path / "c", monkeypatch, cut_permission=BLOCKED)
    assert orch.run_detection(lambda attempt: files3, cfg3).cut_summary == "blocked"


@pytest.mark.parametrize("empty", ["", Path(""), "  "])
def test_empty_snapshot_path_means_no_publish(tmp_path, monkeypatch, empty):
    cfg, files, snapshot = _setup(tmp_path, monkeypatch)
    cfg.snapshot_path = empty
    cfg.__post_init__()
    assert cfg.snapshot_path is None
    assert orch.DetectConfig(snapshot_path="").snapshot_path is None
    result = orch.run_detection(lambda attempt: files, cfg)
    assert result.success and result.snapshot_path is None and Path(result.staged_snapshot_path).is_file()
    assert not Path(".").joinpath("seam_3d_base_link.json").exists()


def _manifest_with_extrinsic(files, **overrides):
    manifest = json.loads(Path(files.manifest_path).read_text(encoding="utf-8"))
    base = importlib.import_module("box_seam_detection.live.base_link")
    rec = base.RECORDED_DRIVER_EXTRINSIC
    block = {"parent_frame": "camera_link", "child_frame": "camera_color_optical_frame",
             "translation": list(rec["translation"]),
             "quaternion_xyzw": [-v for v in rec["quaternion_xyzw"]],     # opposite sign: same rotation
             "source": "tf_live", "stamp_ns": 1000, **overrides}
    manifest["driver_extrinsic"] = block
    Path(files.manifest_path).write_text(json.dumps(manifest), encoding="utf-8")


def test_driver_extrinsic_matching_the_constant_is_accepted(tmp_path, monkeypatch):
    cfg, files, snapshot = _setup(tmp_path, monkeypatch)
    _manifest_with_extrinsic(files)
    assert orch.run_detection(lambda attempt: files, cfg).success


@pytest.mark.parametrize("policy, ok", [("refuse", False), ("warn", True), ("use_manifest", True)])
def test_driver_extrinsic_mismatch_policy(tmp_path, monkeypatch, capsys, policy, ok):
    cfg, files, snapshot = _setup(tmp_path, monkeypatch)
    base = importlib.import_module("box_seam_detection.live.base_link")
    t = list(base.RECORDED_DRIVER_EXTRINSIC["translation"])
    t[0] += 0.002                                                          # 2 mm off
    _manifest_with_extrinsic(files, translation=t)
    cfg.driver_extrinsic_policy = policy
    _stale(snapshot)
    result = orch.run_detection(lambda attempt: files, cfg)
    assert result.success is ok
    if not ok:
        assert result.status == "error" and "driver_extrinsic_mismatch(" in result.reason
        assert not snapshot.exists()
    else:
        assert snapshot.exists()
        assert ("driver_extrinsic_mismatch" in capsys.readouterr().err) == (policy == "warn")


def test_driver_extrinsic_rotation_mismatch_is_refused(tmp_path, monkeypatch):
    cfg, files, snapshot = _setup(tmp_path, monkeypatch)
    base = importlib.import_module("box_seam_detection.live.base_link")
    import math
    rec = base.RECORDED_DRIVER_EXTRINSIC["quaternion_xyzw"]
    half = math.radians(0.3) / 2                                           # 0.3 deg about z on top
    q = base.rotation_matrix_to_quaternion(
        base.quaternion_to_rotation_matrix(0, 0, math.sin(half), math.cos(half))
        @ base.quaternion_to_rotation_matrix(*rec))
    _manifest_with_extrinsic(files, quaternion_xyzw=list(q))
    result = orch.run_detection(lambda attempt: files, cfg)
    assert not result.success and "driver_extrinsic_mismatch(" in result.reason


def test_base_link_records_which_extrinsic_was_used_and_the_difference(tmp_path):
    import math
    import numpy as np
    base = importlib.import_module("box_seam_detection.live.base_link")
    s = math.sqrt(0.5)
    calib = tmp_path / "t.calib"
    calib.write_text(
        "parameters:\n  name: t\n  calibration_type: eye_on_base\n  robot_base_frame: base_link\n"
        "  tracking_base_frame: camera_link\ntransform:\n  translation: {x: 0.5, y: 0.0, z: 0.2}\n"
        f"  rotation: {{x: 0.0, y: 0.0, z: {s!r}, w: {s!r}}}\n", encoding="utf-8")
    rec = base.RECORDED_DRIVER_EXTRINSIC
    block = {"parent_frame": "camera_link", "child_frame": "camera_color_optical_frame",
             "translation": [rec["translation"][0] + 0.0005, *rec["translation"][1:]],
             "quaternion_xyzw": list(rec["quaternion_xyzw"]), "source": "tf_live"}
    _, out = base.resolve_camera_to_base(calib, "recorded", "base_link", "camera_color_optical_frame", 1,
                                         driver_block=block)
    cmp = out["source"]["driver_extrinsic_comparison"]
    assert cmp["used"] == "manifest" and cmp["policy"] == "refuse" and cmp["exceeds_tolerance"] is False
    assert cmp["translation_diff_m"] == pytest.approx(0.0005, abs=1e-9) and cmp["rotation_diff_deg"] < 1e-6
    block["translation"][0] += 0.002
    with pytest.raises(ValueError, match="driver_extrinsic_mismatch"):
        base.resolve_camera_to_base(calib, "recorded", "base_link", "camera_color_optical_frame", 1,
                                    driver_block=block)
    _, out = base.resolve_camera_to_base(calib, "recorded", "base_link", "camera_color_optical_frame", 1,
                                         driver_block=block, driver_policy="use_manifest")
    assert out["source"]["driver_extrinsic_comparison"]["exceeds_tolerance"] is True
    # no block -> no comparison key, constant used (today's captures)
    _, out = base.resolve_camera_to_base(calib, "recorded", "base_link", "camera_color_optical_frame", 1)
    assert "driver_extrinsic_comparison" not in out["source"]


def test_detector_gets_the_configured_calibration_file(tmp_path, monkeypatch):
    cfg, files, snapshot = _setup(tmp_path, monkeypatch)
    seen = []
    inner = orch.run_step

    def spy(title, cmd, capture=False, check=True):
        if "autolabel_v2" in title:
            seen.append(cmd)
        return inner(title, cmd, capture, check)

    monkeypatch.setattr(orch, "run_step", spy)
    assert orch.run_detection(lambda attempt: files, cfg).success
    cmd = seen[0]
    assert cmd[cmd.index("--calib-file") + 1] == str(Path(cfg.calib_file).expanduser())
