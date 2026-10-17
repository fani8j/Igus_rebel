"""RAM-first capture staging (live.capture_staging): success archive, failed_recent ring, fallbacks, pipeline
record rewrite, crash-leftover sweep, multi-attempt runs, CLI integration. Pure Python, tmp dirs only (no /dev/shm,
no ROS, no real archive)."""
import json
import os
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from box_seam_detection.live import capture_staging as cs
from box_seam_detection.live import orchestrator as orch


def _quiet(*_args, **_kwargs):
    pass


def write_capture(session_root, capture_id, extra=None, stamp_ns=1_000_000_000):
    """A schema-valid rgbd_capture/v1 capture written into <session_root>/{rgb,pointcloud,captures}."""
    session_root = Path(session_root)
    for sub in cs.CAPTURE_SUBDIRS:
        (session_root / sub).mkdir(parents=True, exist_ok=True)
    image = session_root / "rgb" / f"{capture_id}.png"
    pcd = session_root / "pointcloud" / f"{capture_id}.pcd"
    manifest_path = session_root / "captures" / f"{capture_id}.json"
    image.write_bytes(b"png-" + capture_id.encode())
    pcd.write_bytes(b"pcd-" + capture_id.encode() * 100)
    manifest = {
        "schema": "rgbd_capture/v1", "capture_id": capture_id,
        "image": {"path": str(image), "width": 4, "height": 3, "header": {"frame_id": "optical", "stamp_ns": stamp_ns}},
        "pointcloud": {"path": str(pcd), "header": {"frame_id": "optical", "stamp_ns": stamp_ns}},
        "camera_info": {"width": 4, "height": 3, "header": {"frame_id": "optical", "stamp_ns": stamp_ns}},
    }
    manifest.update(extra or {})
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return image, pcd, manifest_path


def open_session(tmp_path, **kwargs):
    archive = tmp_path / "archive"
    session = cs.open_staging(tmp_path / "stage", archive, log=_quiet, warn=_quiet, **kwargs)
    assert session is not None
    return session, archive


def result_for(winner=None, attempt_ids=(), status=None, pipeline_dir=None, averaging=None):
    attempts = [{"attempt": i + 1, "capture_id": cid, "pipeline_dir": None} for i, cid in enumerate(attempt_ids)]
    if averaging is not None:
        attempts[-1]["averaging"] = averaging
    success = winner is not None
    return SimpleNamespace(success=success, status=status or ("accepted" if success else "rejected"),
                           capture_id=winner or (attempt_ids[-1] if attempt_ids else None), attempts=attempts,
                           pipeline_dir=pipeline_dir, snapshot_path=None, staged_snapshot_path=None)


def ring_ids(archive):
    ring = archive / cs.FAILED_RING_DIR
    return sorted(p.stem for p in (ring / "captures").glob("*.json")) if (ring / "captures").is_dir() else []


def staged_files(session):
    return [p for sub in cs.CAPTURE_SUBDIRS for p in (session.root / sub).iterdir()]


# ----------------------------------------------------------------------------- success archive
def test_success_capture_goes_to_normal_layout_with_rewritten_manifest(tmp_path):
    session, archive = open_session(tmp_path)
    image, pcd, manifest = write_capture(session.root, "cap_ok")
    report = session.archive_run(result_for("cap_ok", ["cap_ok"]))
    assert report.success_ids == ["cap_ok"] and not report.failed_ids and not report.errors
    a_image, a_pcd = archive / "rgb" / "cap_ok.png", archive / "pointcloud" / "cap_ok.pcd"
    a_manifest = archive / "captures" / "cap_ok.json"
    assert a_image.read_bytes() == b"png-cap_ok" and a_pcd.read_bytes().startswith(b"pcd-cap_ok")
    data = json.loads(a_manifest.read_text())
    assert Path(data["image"]["path"]) == a_image.resolve() and Path(data["pointcloud"]["path"]) == a_pcd.resolve()
    # the archived capture is a normal capture: the manifest validates from its new location
    orch.load_capture_manifest(a_manifest, str(a_image), str(a_pcd))
    assert orch.find_matching_pcd(str(a_image)) == str(a_pcd.resolve())
    assert oct(a_manifest.stat().st_mode & 0o777) == "0o644"
    # staged files are gone from RAM; nothing went to the ring
    assert not image.exists() and not pcd.exists() and not manifest.exists()
    assert not (archive / cs.FAILED_RING_DIR).exists()
    assert "成功" in report.summary() and "cap_ok" in report.summary()


def test_existing_archive_files_are_never_overwritten(tmp_path):
    session, archive = open_session(tmp_path)
    (archive / "rgb").mkdir(parents=True)
    (archive / "rgb" / "cap_ok.png").write_bytes(b"precious")
    write_capture(session.root, "cap_ok")
    report = session.archive_run(result_for("cap_ok", ["cap_ok"]))
    assert report.errors and not report.success_ids
    assert (archive / "rgb" / "cap_ok.png").read_bytes() == b"precious"
    assert not (archive / "pointcloud" / "cap_ok.pcd").exists()
    assert (session.root / "captures" / "cap_ok.json").exists()      # kept in RAM, not lost


# ----------------------------------------------------------------------------- failed ring
def test_failed_ring_keeps_exactly_three_and_only_deletes_its_own_files(tmp_path):
    session, archive = open_session(tmp_path)
    ring = archive / cs.FAILED_RING_DIR
    for sub in cs.CAPTURE_SUBDIRS:
        (ring / sub).mkdir(parents=True)
        (ring / sub / f"foreign{cs._SUFFIX[sub]}").write_text("not ours")
    normal = archive / "rgb" / "old_success.png"
    normal.parent.mkdir(parents=True)
    normal.write_text("normal archive")
    for i in range(5):
        cid = f"fail_{i}"
        write_capture(session.root, cid)
        report = session.archive_run(result_for(None, [cid], status="rejected"), ids=[cid])
        assert report.failed_ids == [cid] and not report.errors
        time.sleep(0.002)
    assert ring_ids(archive) == ["fail_2", "fail_3", "fail_4", "foreign"]   # last 3 + the foreign file
    for sub in cs.CAPTURE_SUBDIRS:
        assert (ring / sub / f"foreign{cs._SUFFIX[sub]}").read_text() == "not ours"
        assert not (ring / sub / f"fail_0{cs._SUFFIX[sub]}").exists()
        assert not (ring / sub / f"fail_1{cs._SUFFIX[sub]}").exists()
    assert normal.read_text() == "normal archive"
    assert staged_files(session) == []
    # ring manifests validate from the ring location
    for cid in ("fail_2", "fail_3", "fail_4"):
        orch.load_capture_manifest(ring / "captures" / f"{cid}.json", str(ring / "rgb" / f"{cid}.png"),
                                   str(ring / "pointcloud" / f"{cid}.pcd"))


def test_ring_batch_of_failures_in_one_archive_call(tmp_path):
    session, archive = open_session(tmp_path)
    for i in range(5):
        write_capture(session.root, f"b_{i}")
        time.sleep(0.002)
    report = session.archive_run(result_for(None, [f"b_{i}" for i in range(5)]))
    assert len(report.failed_ids) == 5
    assert ring_ids(archive) == ["b_2", "b_3", "b_4"]


def test_run_that_raised_archives_everything_as_failed(tmp_path):
    session, archive = open_session(tmp_path)
    write_capture(session.root, "boom_1")
    report = session.archive_run(None)
    assert report.failed_ids == ["boom_1"] and ring_ids(archive) == ["boom_1"]
    assert not (archive / "rgb").exists()


@pytest.mark.parametrize("status", ["rejected", "blocked", "error", "cancelled"])
def test_non_accepted_statuses_are_failures(tmp_path, status):
    session, archive = open_session(tmp_path)
    write_capture(session.root, "c1")
    res = result_for(None, ["c1"], status=status)
    assert cs.classify_run(res)[0] is None
    report = session.archive_run(res)
    assert report.failed_ids == ["c1"] and not (archive / "captures").exists()


# ----------------------------------------------------------------------------- multi attempt
def test_multi_attempt_run_archives_winner_and_rings_the_rest(tmp_path):
    session, archive = open_session(tmp_path)
    for i, cid in enumerate(("a1", "a2", "a3")):
        write_capture(session.root, cid, stamp_ns=1_000 + i)
        time.sleep(0.002)
    res = result_for("a3", ["a1", "a2", "a3"])
    assert cs.classify_run(res) == ("a3", set())
    report = session.archive_run(res)
    assert report.success_ids == ["a3"] and sorted(report.failed_ids) == ["a1", "a2"]
    assert (archive / "captures" / "a3.json").is_file() and not (archive / "captures" / "a1.json").exists()
    assert ring_ids(archive) == ["a1", "a2"]
    assert staged_files(session) == []


def test_averaged_burst_sources_are_discarded_not_archived(tmp_path):
    session, archive = open_session(tmp_path)
    for cid in ("f1", "f2", "f3"):
        write_capture(session.root, cid)
    write_capture(session.root, "f2_avg3", extra={"averaging": {"source_capture_ids": ["f1", "f2", "f3"]}})
    res = result_for("f2_avg3", ["f2_avg3"], averaging={"source_capture_ids": ["f1", "f2", "f3"]})
    assert cs.classify_run(res) == ("f2_avg3", {"f1", "f2", "f3"})
    report = session.archive_run(res)
    assert report.success_ids == ["f2_avg3"] and sorted(report.discarded_ids) == ["f1", "f2", "f3"]
    assert not (archive / cs.FAILED_RING_DIR).exists() and staged_files(session) == []


# ----------------------------------------------------------------------------- pipeline records
def test_pipeline_json_paths_rewritten_but_never_the_planner_snapshot(tmp_path):
    session, archive = open_session(tmp_path)
    image, pcd, manifest = write_capture(session.root, "cap_p")
    run_dir = archive / "pipeline" / "cap_p" / "20261016T000000_000000Z_1"
    (run_dir / "autolabel_v2_overlays").mkdir(parents=True)
    record = run_dir / "cap_p_autolabel_v2_record.json"
    record.write_text(json.dumps({"source_image": str(image), "pcd": str(pcd), "n": 1}))
    nested = run_dir / "autolabel_v2_overlays" / "reject.json"
    nested.write_text(json.dumps({"manifest": str(manifest)}))
    unrelated = run_dir / "other.json"
    unrelated.write_text('{"x": 1}')
    staged_snapshot = run_dir / "planner_snapshot.json"
    staged_snapshot.write_text(json.dumps({"hint": str(image)}))
    published = tmp_path / "seam_3d_base_link.json"
    published.write_text(json.dumps({"hint": str(image)}))
    quarantine = tmp_path / "seam_3d_base_link.json.side_seed_rejected"
    quarantine.write_text(json.dumps({"hint": str(image)}))
    before = {p: (p.read_text(), p.stat().st_mtime_ns) for p in (staged_snapshot, published, quarantine, unrelated)}
    cfg = SimpleNamespace(snapshot_path=published, pipeline_root=None, camera_geometry_path=None,
                          base_points_path=None)
    res = result_for("cap_p", ["cap_p"], pipeline_dir=str(run_dir))
    report = session.archive_run(res, **cs.run_archive_kwargs(cfg, res))
    assert report.success_ids == ["cap_p"]
    a_image, a_pcd = (archive / "rgb" / "cap_p.png").resolve(), (archive / "pointcloud" / "cap_p.pcd").resolve()
    assert json.loads(record.read_text()) == {"source_image": str(a_image), "pcd": str(a_pcd), "n": 1}
    assert json.loads(nested.read_text()) == {"manifest": str((archive / "captures" / "cap_p.json").resolve())}
    for path, (text, mtime) in before.items():           # snapshot files + unrelated JSON: byte- and mtime-identical
        assert (path.read_text(), path.stat().st_mtime_ns) == (text, mtime), path
    assert not list(run_dir.glob(".*rewrite*"))


def test_pipeline_rewrite_covers_explicit_camera_geometry_override(tmp_path):
    session, archive = open_session(tmp_path)
    image, pcd, manifest = write_capture(session.root, "cap_g")
    geometry = tmp_path / "elsewhere" / "camera_geometry.json"
    geometry.parent.mkdir()
    geometry.write_text(json.dumps({"capture": {"manifest_path": str(manifest)}}))
    cfg = SimpleNamespace(snapshot_path=None, pipeline_root=None, camera_geometry_path=geometry,
                          base_points_path=None)
    res = result_for("cap_g", ["cap_g"])
    session.archive_run(res, **cs.run_archive_kwargs(cfg, res))
    assert json.loads(geometry.read_text())["capture"]["manifest_path"] == str(
        (archive / "captures" / "cap_g.json").resolve())


# ----------------------------------------------------------------------------- settings / fallbacks
def test_off_means_direct_to_archive(tmp_path):
    assert cs.resolve_staging_setting("off", environ={}) is None
    assert cs.resolve_staging_setting("OFF", environ={}) is None
    assert cs.open_staging("off", tmp_path / "archive", log=_quiet, warn=_quiet) is None
    assert not (tmp_path / "archive").exists()


def test_setting_precedence_cli_then_env_then_default(tmp_path):
    assert cs.resolve_staging_setting(None, environ={}) == Path(cs.DEFAULT_STAGING_ROOT)
    assert cs.resolve_staging_setting("", environ={}) == Path(cs.DEFAULT_STAGING_ROOT)
    env = {cs.STAGING_ENV: str(tmp_path / "envstage")}
    assert cs.resolve_staging_setting(None, environ=env) == tmp_path / "envstage"
    # a parameter left at the built-in default does not mask the environment fallback
    assert cs.resolve_staging_setting(cs.DEFAULT_STAGING_ROOT, environ=env) == tmp_path / "envstage"
    assert cs.resolve_staging_setting(str(tmp_path / "cli"), environ=env) == tmp_path / "cli"
    assert cs.resolve_staging_setting(None, environ={cs.STAGING_ENV: "off"}) is None
    assert cs.resolve_staging_setting("off", environ=env) is None


def test_fallback_when_root_cannot_be_created(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir")
    warnings = []
    assert cs.open_staging(blocker / "stage", tmp_path / "archive", log=_quiet, warn=warnings.append) is None
    assert warnings and "archive" in warnings[0]


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_fallback_when_root_is_not_writable(tmp_path):
    root = tmp_path / "ro"
    root.mkdir()
    root.chmod(0o500)
    try:
        warnings = []
        assert cs.open_staging(root, tmp_path / "archive", log=_quiet, warn=warnings.append) is None
        assert warnings
    finally:
        root.chmod(0o700)


def test_fallback_when_less_than_512mb_free(tmp_path, monkeypatch):
    real = shutil.disk_usage(tmp_path)
    monkeypatch.setattr(shutil, "disk_usage", lambda _p: real._replace(free=cs.MIN_FREE_BYTES - 1))
    warnings = []
    assert cs.open_staging(tmp_path / "stage", tmp_path / "archive", log=_quiet, warn=warnings.append) is None
    assert warnings and "512" in warnings[0]
    assert not list((tmp_path / "stage").glob("s*"))          # no half-created session left behind


# ----------------------------------------------------------------------------- crash leftovers
def test_sweep_archives_dead_sessions_into_failed_recent_and_skips_live_ones(tmp_path):
    stage, archive = tmp_path / "stage", tmp_path / "archive"
    live = cs.open_staging(stage, archive, log=_quiet, warn=_quiet)
    write_capture(live.root, "alive_cap")
    dead = stage / "s424242_deadbeef"                     # a crashed process: files, an unlocked .lock
    write_capture(dead, "dead_1")
    write_capture(dead, "dead_2")
    (dead / ".lock").write_text("")
    (dead / "rgb" / ".half.tmp-1").write_text("junk")
    reports = cs.sweep_stale(stage, archive, exclude=live.root, log=_quiet, warn=_quiet)
    assert len(reports) == 1 and sorted(reports[0].failed_ids) == ["dead_1", "dead_2"]
    assert ring_ids(archive) == ["dead_1", "dead_2"]
    assert not dead.exists()
    assert (live.root / "captures" / "alive_cap.json").is_file()        # a live owner's files are never touched
    orch.load_capture_manifest(archive / cs.FAILED_RING_DIR / "captures" / "dead_1.json",
                               str(archive / cs.FAILED_RING_DIR / "rgb" / "dead_1.png"),
                               str(archive / cs.FAILED_RING_DIR / "pointcloud" / "dead_1.pcd"))


def test_open_staging_sweeps_leftovers_of_earlier_processes(tmp_path):
    stage, archive = tmp_path / "stage", tmp_path / "archive"
    dead = stage / "s1_00000000"
    write_capture(dead, "left_over")
    (dead / ".lock").write_text("")
    session = cs.open_staging(stage, archive, log=_quiet, warn=_quiet)
    assert session is not None
    assert ring_ids(archive) == ["left_over"] and not dead.exists()
    assert session.root.is_dir()


def test_sweep_ignores_foreign_dirs_and_fresh_lockless_dirs(tmp_path):
    stage, archive = tmp_path / "stage", tmp_path / "archive"
    foreign = stage / "my_notes"
    foreign.mkdir(parents=True)
    (foreign / "keep.txt").write_text("x")
    fresh = stage / "s7_abcdef01"          # being created right now: no .lock yet, brand new
    write_capture(fresh, "mid_create")
    assert cs.sweep_stale(stage, archive, log=_quiet, warn=_quiet) == []
    assert (foreign / "keep.txt").exists() and (fresh / "captures" / "mid_create.json").exists()


# ----------------------------------------------------------------------------- background archiver
def test_background_archiver_serializes_and_close_waits(tmp_path):
    session, archive = open_session(tmp_path)
    logged = []
    session._log = logged.append
    for i in range(4):
        write_capture(session.root, f"bg_{i}")
        time.sleep(0.002)
    for i in range(4):
        cid = f"bg_{i}"
        session.submit_archive(result_for(None, [cid]), [cid])
    assert session.wait_idle(10.0)
    assert ring_ids(archive) == ["bg_1", "bg_2", "bg_3"]
    write_capture(session.root, "bg_last")
    session.submit_archive(result_for("bg_last", ["bg_last"]), ["bg_last"])
    root = session.root
    assert session.close(10.0)                 # waits for the pending job, then drops the empty session dir
    assert (archive / "captures" / "bg_last.json").is_file() and not root.exists()
    assert any("歸檔" in line for line in logged)


def test_close_keeps_a_session_with_unarchived_files(tmp_path):
    session, archive = open_session(tmp_path)
    write_capture(session.root, "stuck")
    warnings = []
    session._warn = warnings.append
    assert session.close(1.0)
    assert session.root.exists() and warnings
    # a later start sweeps it into failed_recent
    cs.sweep_stale(session.root.parent, archive, log=_quiet, warn=_quiet)
    assert ring_ids(archive) == ["stuck"]


# ----------------------------------------------------------------------------- orchestrator integration
def test_pipeline_dir_stays_in_archive_when_the_capture_is_staged(tmp_path):
    session, archive = open_session(tmp_path)
    _image, _pcd, manifest_path = write_capture(session.root, "cap_pp")
    manifest = json.loads(manifest_path.read_text())
    cfg = orch.DetectConfig(output_root=archive, capture_staging_dir=session.root)
    pipeline_dir, _geo, _base, _snap = orch.pipeline_output_paths(cfg, manifest_path.resolve(), manifest,
                                                                  archive.resolve())
    assert pipeline_dir.parent.parent == archive.resolve() / "pipeline"
    # unstaged capture in the archive layout keeps today's rule (next to its captures/ dir)
    other = tmp_path / "other_root"
    _i, _p, other_manifest = write_capture(other, "cap_pp2")
    cfg2 = orch.DetectConfig(output_root=archive)
    pipeline_dir2, *_ = orch.pipeline_output_paths(cfg2, other_manifest.resolve(),
                                                   json.loads(other_manifest.read_text()), archive.resolve())
    assert pipeline_dir2.parent.parent == other.resolve() / "pipeline"


def test_live_capture_source_records_into_the_staging_dir(tmp_path, monkeypatch):
    session, archive = open_session(tmp_path)
    seen = []

    def fake_acquire(cfg, output_dir, title=""):
        seen.append(Path(output_dir))
        image, pcd, manifest_path = write_capture(output_dir, "rec_1")
        return str(image), str(pcd), manifest_path, json.loads(manifest_path.read_text())

    monkeypatch.setattr(orch, "acquire_live_capture", fake_acquire)
    cfg = orch.DetectConfig(output_root=archive, capture_staging_dir=session.root)
    orch._live_capture_source(cfg)(1)
    assert seen == [session.root]
    cfg_off = orch.DetectConfig(output_root=archive)
    orch._live_capture_source(cfg_off)(1)
    assert seen[-1] == archive


def _main_with_fake_run(tmp_path, monkeypatch, behaviour, *extra):
    archive = tmp_path / "archive"
    stage = tmp_path / "stage"
    order = []

    def fake_source(cfg):
        def next_capture(attempt):
            raise AssertionError("fake run_detection never calls it")
        return next_capture

    def fake_run(next_capture, cfg, *a, **k):
        order.append(("run", cfg.capture_staging_dir))
        return behaviour(cfg, order)

    monkeypatch.setattr(orch, "_live_capture_source", fake_source)
    monkeypatch.setattr(orch, "run_detection", fake_run)
    code = orch.main(["--output-dir", str(archive), "--capture-staging-dir", str(stage),
                      "--planner-output-json", str(tmp_path / "snap.json"), *extra])
    return code, archive, stage, order


def test_cli_archives_after_the_snapshot_is_published(tmp_path, monkeypatch, capsys):
    snapshot = tmp_path / "snap.json"

    def behaviour(cfg, order):
        write_capture(cfg.capture_staging_dir, "rej_1")
        write_capture(cfg.capture_staging_dir, "rej_2")
        write_capture(cfg.capture_staging_dir, "win_1")
        snapshot.write_text('{"schema": "box_h_cut_snapshot/v1"}')
        order.append(("snapshot_published", snapshot.exists()))
        return orch.DetectResult(success=True, status="accepted", capture_id="win_1",
                                 attempts=[{"attempt": 1, "capture_id": "rej_1", "pipeline_dir": None},
                                           {"attempt": 2, "capture_id": "rej_2", "pipeline_dir": None},
                                           {"attempt": 3, "capture_id": "win_1", "pipeline_dir": None}])

    code, archive, stage, order = _main_with_fake_run(tmp_path, monkeypatch, behaviour)
    assert code == 0
    assert (archive / "captures" / "win_1.json").is_file()
    assert ring_ids(archive) == ["rej_1", "rej_2"]
    assert order[0][0] == "run" and order[1] == ("snapshot_published", True)
    assert snapshot.read_text() == '{"schema": "box_h_cut_snapshot/v1"}'
    assert not list(stage.glob("s*"))                                  # session dir removed from RAM
    out = capsys.readouterr().out
    assert out.count("capture 歸檔") == 1 and "用時" in out


def test_cli_archives_everything_as_failed_when_the_run_raises(tmp_path, monkeypatch):
    def behaviour(cfg, order):
        write_capture(cfg.capture_staging_dir, "half_1")
        raise RuntimeError("recorder died")

    code, archive, stage, _order = _main_with_fake_run(tmp_path, monkeypatch, behaviour)
    assert code == 2
    assert ring_ids(archive) == ["half_1"] and not (archive / "captures").exists()
    assert not list(stage.glob("s*"))


def test_cli_rejected_run_leaves_only_the_ring(tmp_path, monkeypatch):
    def behaviour(cfg, order):
        write_capture(cfg.capture_staging_dir, "r1")
        return orch.DetectResult(success=False, status="rejected", reason="x", capture_id="r1",
                                 attempts=[{"attempt": 1, "capture_id": "r1", "pipeline_dir": None}], exit_code=1)

    code, archive, _stage, _order = _main_with_fake_run(tmp_path, monkeypatch, behaviour)
    assert code == 1 and ring_ids(archive) == ["r1"] and not (archive / "rgb").exists()


def test_cli_staging_off_runs_without_a_staging_dir(tmp_path, monkeypatch):
    archive = tmp_path / "archive"
    seen = {}
    monkeypatch.setattr(orch, "_live_capture_source", lambda cfg: (lambda attempt: None))
    monkeypatch.setattr(orch, "run_detection", lambda nc, cfg, *a, **k: (
        seen.setdefault("dir", cfg.capture_staging_dir), orch.DetectResult(success=True, status="accepted"))[1])
    assert orch.main(["--output-dir", str(archive), "--capture-staging-dir", "off"]) == 0
    assert seen["dir"] is None


def test_offline_from_image_never_stages_or_moves_anything(tmp_path, monkeypatch):
    archive = tmp_path / "archive"
    image, pcd, manifest = write_capture(archive, "old_cap")
    before = sorted(str(p) for p in archive.rglob("*"))
    monkeypatch.setattr(orch, "open_staging", lambda *a, **k: pytest.fail("offline must not stage"))
    captured = {}

    def fake_run(next_capture, cfg, *a, **k):
        captured["cfg"] = cfg
        captured["files"] = next_capture(1)
        return orch.DetectResult(success=True, status="accepted", capture_id="old_cap")

    monkeypatch.setattr(orch, "run_detection", fake_run)
    code = orch.main(["--output-dir", str(archive), "--from-image", str(image),
                      "--capture-staging-dir", str(tmp_path / "stage")])
    assert code == 0 and captured["cfg"].capture_staging_dir is None
    assert Path(captured["files"].image_path) == image.resolve()
    assert sorted(str(p) for p in archive.rglob("*")) == before and not (tmp_path / "stage").exists()


# ----------------------------------------------------------------------------- shared helpers (burst averaging)
def test_burst_average_still_links_pcd_and_writes_a_validating_manifest(tmp_path):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    records = []
    root = tmp_path / "burst"
    for i, cid in enumerate(("b0", "b1", "b2")):
        image, pcd, manifest_path = write_capture(root, cid, stamp_ns=10 + i)
        cv2.imwrite(str(image), np.full((3, 4, 3), 100 + i, dtype=np.uint8))
        records.append(orch._load_pcd_sequence_record(manifest_path))
    result = orch.build_averaged_capture(records, 50.0)
    assert result["status"] == "ok"
    assert oct(Path(result["manifest_path"]).stat().st_mode & 0o777) == "0o644"
    assert Path(result["pcd_path"]).read_bytes() == (root / "pointcloud" / "b1.pcd").read_bytes()
