"""Multi-frame RGB averaging for the live ``live.orchestrator`` CLI.

Each live attempt records a burst of N synced captures (rgbd_recorder.py
max_frames:=N), checks the burst is static (per-frame mean |frame - median
frame| <= --average-max-diff), and writes ONE new rgbd_capture/v1 capture
``<middle id>_avg<N>``: RGB = per-pixel mean, point cloud = middle frame's
PCD, manifest = middle frame's + an additive "averaging" block. A non-static
burst is a retryable rejection ``burst_not_static``. N=1 is the old single
``--once`` capture; offline ``--from-image`` never averages. The camera,
subprocesses and SAM3 are faked.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

TESTS = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("autolabel_v2_retry_helpers",
                                               str(TESTS / "test_autolabel_v2_retry.py"))
retry = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(retry)
live = retry.live

det = live.det  # fixture re-export
W, H = live.W, live.H
FRAME_ID = live.FRAME_ID
QUERY_DEPTH = "box_seam_detection.live.query_depth"


@pytest.fixture()
def orch():
    return live._load(live.ORCHESTRATOR, "capture_and_query_depth")


def _frame(value, noise_seed=None, amplitude=0):
    frame = np.full((H, W, 3), value, dtype=np.int16)
    if noise_seed is not None:
        rng = np.random.default_rng(noise_seed)
        frame += rng.integers(-amplitude, amplitude + 1, size=frame.shape, dtype=np.int16)
    return np.clip(frame, 0, 255).astype(np.uint8)


def _make_capture(root, cid, frame, stamp, camera_info=None):
    image = root / "rgb" / f"{cid}.png"
    pcd = root / "pointcloud" / f"{cid}.pcd"
    manifest_path = root / "captures" / f"{cid}.json"
    for path in (image, pcd, manifest_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(image), frame)
    pcd.write_bytes(f"fake-pcd-{cid}".encode())
    manifest = {
        "schema": "rgbd_capture/v1", "capture_id": cid, "created_at_ns": stamp,
        "image": {"path": str(image), "width": W, "height": H, "encoding": "bgr8",
                  "header": {"frame_id": FRAME_ID, "stamp_ns": stamp}},
        "pointcloud": {"path": str(pcd), "width": W * H, "height": 1,
                       "header": {"frame_id": FRAME_ID, "stamp_ns": stamp + 1}},
        "camera_info": camera_info or {
            "header": {"frame_id": FRAME_ID, "stamp_ns": stamp}, "width": W, "height": H,
            "distortion_model": "plumb_bob", "d": [0.0] * 5,
            "k": [500.0, 0.0, W / 2, 0.0, 500.0, H / 2, 0.0, 0.0, 1.0],
            "r": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "p": [500.0, 0.0, W / 2, 0.0, 0.0, 500.0, H / 2, 0.0, 0.0, 0.0, 1.0, 0.0]},
        "sync_delta_ns": 1,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return {"image": image, "pcd": pcd, "manifest": manifest_path, "id": cid,
            "frame": frame}


def _burst(root, prefix, frames, stamp0=1000):
    return [_make_capture(root, f"{prefix}_{i + 1}", frame, stamp0 + 100 * i)
            for i, frame in enumerate(frames)]


def _records(orch, captures):
    return [orch._load_pcd_sequence_record(c["manifest"]) for c in captures]


def _static_frames(n=5, base=100, seed=0):
    return [_frame(base, noise_seed=seed + i, amplitude=3) for i in range(n)]


# ------------------------------------------------------------ pure averaging

def test_average_rgb_frames_is_rounded_float_mean(orch):
    frames = [_frame(10), _frame(11), _frame(13), _frame(200, noise_seed=1, amplitude=20)]
    mean, diffs = orch.average_rgb_frames(frames)
    expected = np.rint(np.mean(np.stack(frames).astype(np.float64), axis=0)).astype(np.uint8)
    assert mean.dtype == np.uint8 and np.array_equal(mean, expected)
    assert len(diffs) == 4 and diffs[-1] > 50 > diffs[0]


def test_average_rgb_frames_rejects_mismatched_shapes(orch):
    with pytest.raises(ValueError):
        orch.average_rgb_frames([_frame(1), np.zeros((H, W + 1, 3), np.uint8)])


# ------------------------------------------------------------ averaged capture

def test_build_averaged_capture_png_manifest_and_middle_pcd(orch, tmp_path):
    frames = _static_frames(5)
    caps = _burst(tmp_path / "root", "cap", frames)
    result = orch.build_averaged_capture(_records(orch, caps), 6.0)
    assert result["status"] == "ok"
    middle = caps[2]
    cid = "cap_3_avg5"
    root = (tmp_path / "root").resolve()
    assert Path(result["image_path"]) == root / "rgb" / f"{cid}.png"
    assert Path(result["pcd_path"]) == root / "pointcloud" / f"{cid}.pcd"
    assert Path(result["manifest_path"]) == root / "captures" / f"{cid}.json"
    # RGB = rounded per-pixel mean of the N frames.
    expected = np.rint(np.mean(np.stack(frames).astype(np.float64), axis=0)).astype(np.uint8)
    assert np.array_equal(cv2.imread(result["image_path"], cv2.IMREAD_COLOR), expected)
    # PCD = the middle frame's bytes.
    assert Path(result["pcd_path"]).read_bytes() == middle["pcd"].read_bytes()
    manifest = json.loads(Path(result["manifest_path"]).read_text())
    source = json.loads(middle["manifest"].read_text())
    assert manifest["schema"] == "rgbd_capture/v1" and manifest["capture_id"] == cid
    assert manifest["camera_info"] == source["camera_info"]
    assert manifest["image"]["header"] == source["image"]["header"]
    assert manifest["pointcloud"]["header"] == source["pointcloud"]["header"]
    assert {k: v for k, v in manifest["pointcloud"].items() if k != "path"} == {
        k: v for k, v in source["pointcloud"].items() if k != "path"}
    assert manifest["image"]["path"] == result["image_path"]
    averaging = manifest["averaging"]
    assert averaging["n_frames"] == 5 and averaging["method"] == "rgb_mean_pcd_middle"
    assert averaging["source_capture_ids"] == [f"cap_{i}" for i in range(1, 6)]
    assert averaging["middle_capture_id"] == "cap_3"
    assert 0.0 < averaging["max_frame_diff"] <= 6.0
    assert len(averaging["frame_diffs"]) == 5
    assert averaging["source_manifests"] == [str(c["manifest"].resolve()) for c in caps]
    assert result["averaging"]["n_frames"] == 5
    # Source captures are untouched.
    assert all(json.loads(c["manifest"].read_text())["capture_id"] == c["id"] for c in caps)


def test_averaged_capture_passes_all_binding_checks(orch, det, tmp_path, monkeypatch):
    caps = _burst(tmp_path / "root", "cap", _static_frames(3))
    result = orch.build_averaged_capture(_records(orch, caps), 6.0)
    image, pcd, manifest_path = result["image_path"], result["pcd_path"], result["manifest_path"]
    # Orchestrator binding (also run inside build_averaged_capture).
    orch.load_capture_manifest(manifest_path, image, pcd)
    assert orch.find_capture_manifest(image) == Path(manifest_path)
    assert orch.find_matching_pcd(image) == pcd
    # query_depth_at_pixel.py binding.
    qd = live._load(QUERY_DEPTH)
    assert qd.load_capture_manifest(str(manifest_path), pcd)["capture_id"] == "cap_2_avg3"
    # autolabel_v2_detect.py binding (manifest <-> image <-> pcd, real image size).
    cap = {"image": Path(image), "pcd": Path(pcd), "manifest": Path(manifest_path),
           "sam3_python": tmp_path / "fake_python"}
    cap["sam3_python"].write_text("", encoding="utf-8")
    calls = live._install_fake_autolabel(monkeypatch, det, retry._rec())
    assert det.main(live._argv(cap, tmp_path)) == 0
    assert calls["paths"]["image"] == image and calls["paths"]["pcd"] == pcd
    doc = json.loads((tmp_path / "out" / "seam.json").read_text())
    assert doc["capture_id"] == "cap_2_avg3"


def test_non_static_burst_writes_nothing(orch, tmp_path):
    frames = _static_frames(5)
    frames[4] = _frame(180)  # something moved / lighting jumped
    caps = _burst(tmp_path / "root", "cap", frames)
    result = orch.build_averaged_capture(_records(orch, caps), 6.0)
    assert result["status"] == "not_static"
    assert result["averaging"]["max_frame_diff"] > 6.0
    assert not list((tmp_path / "root").rglob("*_avg5*"))


def test_inconsistent_camera_info_raises(orch, tmp_path):
    caps = _burst(tmp_path / "root", "cap", _static_frames(3))
    manifest = json.loads(caps[1]["manifest"].read_text())
    manifest["camera_info"]["k"][0] = 600.0
    caps[1]["manifest"].write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="CameraInfo"):
        orch.build_averaged_capture(_records(orch, caps), 6.0)


# ------------------------------------------------------------ orchestrator

def _orchestrate(det, tmp_path, monkeypatch, bursts, outcomes, *extra_argv, offline=False,
                 run_to_end=False):
    """Drive main() with fake subprocesses.

    ``bursts``: per recorder call, a list of captures that call "saves".
    ``outcomes``: per detector call ("ok" | "retry" | "reject").
    With ``run_to_end`` steps 3-4 are faked too and the stamped
    ``autolabel_v2_attempts`` summary is captured in state["stamp"].
    """
    mod = live._load(live.ORCHESTRATOR, "capture_and_query_depth")
    root = tmp_path / "root"
    script = tmp_path / "autolabel_v2_detect.py"
    script.write_text("# fake\n", encoding="utf-8")
    sam3 = tmp_path / "fake_python"
    sam3.write_text("", encoding="utf-8")

    class _Probe:
        returncode = 0
        stdout = ""

    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **k: _Probe())
    state = {"recorder": [], "recorder_cmds": [], "detector": [], "titles": [],
             "query": None, "stamp": None}
    queue = list(outcomes)

    def fake_run_step(title, cmd, capture=False, check=True):
        state["titles"].append(title)
        if "rgbd_recorder" in title:
            assert capture is True
            burst = bursts[len(state["recorder"])]
            state["recorder"].append([c["id"] for c in burst])
            state["recorder_cmds"].append(cmd)
            return "".join(retry._recorder_stdout(c) for c in burst)
        if "autolabel_v2" in title:
            state["detector"].append(cmd)
            flag = lambda name: cmd[cmd.index(name) + 1]  # noqa: E731
            manifest = json.loads(Path(flag("--capture-manifest")).read_text())
            cid = manifest["capture_id"]
            assert Path(flag("--image")) == Path(manifest["image"]["path"])
            assert Path(flag("--pcd")) == Path(manifest["pointcloud"]["path"])
            out = Path(flag("--out-seam-json"))
            outcome = queue.pop(0)
            if outcome == "ok":
                out.write_text(json.dumps(retry._good_doc(det, flag("--image"), cid)))
                return 0
            ov = Path(flag("--overlay-dir"))
            ov.mkdir(parents=True, exist_ok=True)
            is_retry = outcome == "retry"
            (ov / f"{cid}_autolabel_v2_reject.json").write_text(json.dumps(
                {"status": "rejected", "capture_id": cid,
                 "reasons": ["rectangle_inconsistent(A=6.00mm,B=0.10mm)"] if is_retry
                 else ["wrong_face_suspect"], "retryable": is_retry}))
            return 1
        if "query_depth" in title:
            state["query"] = cmd
            if not run_to_end:
                live._halt(mod, title)
            out = Path(cmd[cmd.index("--output-json") + 1])
            out.write_text("{}", encoding="utf-8")
            return None
        if "base_link" in title and run_to_end:
            out = Path(cmd[cmd.index("--planner-output-json") + 1])
            out.write_text("{}", encoding="utf-8")
            return None
        raise AssertionError(f"unexpected step {title}")

    monkeypatch.setattr(mod, "run_step", fake_run_step)
    if run_to_end:
        monkeypatch.setattr(mod, "add_side_cutting_points", lambda path: None)
        monkeypatch.setattr(mod, "validate_planner_snapshot", lambda *a: None)

        def fake_stamp(path, method, extra=None):
            state["stamp"] = extra

        monkeypatch.setattr(mod, "stamp_detector_method", fake_stamp)
    argv = ["--downstream-inprocess", "off", "--autolabel-v2",
            "--autolabel-v2-script", str(script),
            "--autolabel-v2-sam3-python", str(sam3),
            "--calib-file", str(live._calib(tmp_path)),
            "--output-dir", str(root), *extra_argv]
    # live default is 1 frame (averaging showed no live gain); these tests exercise bursts
    if not offline and "--autolabel-v2-average" not in extra_argv:
        argv += ["--autolabel-v2-average", "5"]
    if run_to_end:
        argv.append("--no-side-seed")
    if offline:
        c = bursts[0][0]
        argv += ["--from-image", str(c["image"]), "--pcd", str(c["pcd"]),
                 "--capture-manifest", str(c["manifest"])]
    code = mod.main(argv)
    if code == live.STOP_CODE:
        code = "stopped_at_query"
    return code, state


def _flag(cmd, name):
    return cmd[cmd.index(name) + 1]


def test_live_default_burst_of_5_averaged_capture_flows_downstream(det, tmp_path, monkeypatch):
    root = tmp_path / "root"
    burst = _burst(root, "cap", _static_frames(5))
    code, state = _orchestrate(det, tmp_path, monkeypatch, [burst], ["ok"])
    assert code == "stopped_at_query"
    cmd = state["recorder_cmds"][0]
    assert "--once" not in cmd and "max_frames:=5" in cmd and "save_rate_hz:=5.0" in cmd
    assert len(state["detector"]) == 1
    det_cmd = state["detector"][0]
    avg_manifest = (root / "captures" / "cap_3_avg5.json").resolve()
    assert _flag(det_cmd, "--capture-manifest") == str(avg_manifest)
    assert _flag(det_cmd, "--image") == str((root / "rgb" / "cap_3_avg5.png").resolve())
    assert "/cap_3_avg5/" in _flag(det_cmd, "--work-dir")
    q = state["query"]
    assert _flag(q, "--pcd") == str((root / "pointcloud" / "cap_3_avg5.pcd").resolve())
    assert _flag(q, "--capture-manifest") == str(avg_manifest)
    assert _flag(q, "--seam-json") == _flag(det_cmd, "--out-seam-json")


def test_live_custom_average_and_rate(det, tmp_path, monkeypatch):
    burst = _burst(tmp_path / "root", "cap", _static_frames(3))
    code, state = _orchestrate(det, tmp_path, monkeypatch, [burst], ["ok"],
                               "--autolabel-v2-average", "3", "--average-rate-hz", "10")
    assert code == "stopped_at_query"
    cmd = state["recorder_cmds"][0]
    assert "max_frames:=3" in cmd and "save_rate_hz:=10.0" in cmd
    assert "cap_2_avg3" in _flag(state["detector"][0], "--capture-manifest")


def test_live_non_static_burst_is_retried_with_new_burst(det, tmp_path, monkeypatch, capsys):
    root = tmp_path / "root"
    moving = _static_frames(5)
    moving[0] = _frame(30)
    first = _burst(root, "a", moving)
    second = _burst(root, "b", _static_frames(5, seed=10), stamp0=5000)
    code, state = _orchestrate(det, tmp_path, monkeypatch, [first, second], ["ok"],
                               run_to_end=True)
    assert code == 0
    assert len(state["recorder"]) == 2 and len(state["detector"]) == 1
    assert "b_3_avg5" in _flag(state["detector"][0], "--capture-manifest")
    assert not list(root.rglob("a_3_avg5*"))
    err = capsys.readouterr().err
    assert "burst_not_static" in err
    summary = state["stamp"]["autolabel_v2_attempts"]
    assert summary["used"] == 2 and summary["average_frames"] == 5
    first_entry, second_entry = summary["attempts"]
    assert first_entry["outcome"] == "rejected" and first_entry["retryable"] is True
    assert first_entry["reasons"][0].startswith("burst_not_static(max_diff=")
    assert first_entry["exit_code"] is None and first_entry["pipeline_dir"] is None
    assert first_entry["averaging"]["source_capture_ids"] == [f"a_{i}" for i in range(1, 6)]
    assert first_entry["averaging"]["max_frame_diff"] > 6.0
    assert second_entry["outcome"] == "accepted" and second_entry["capture_id"] == "b_3_avg5"
    avg = second_entry["averaging"]
    assert avg["n_frames"] == 5 and avg["method"] == "rgb_mean_pcd_middle"
    assert avg["source_capture_ids"] == [f"b_{i}" for i in range(1, 6)]
    assert 0.0 <= avg["max_frame_diff"] <= 6.0


def test_live_rectangle_retry_takes_a_new_burst(det, tmp_path, monkeypatch):
    root = tmp_path / "root"
    first = _burst(root, "a", _static_frames(5))
    second = _burst(root, "b", _static_frames(5, seed=10), stamp0=5000)
    code, state = _orchestrate(det, tmp_path, monkeypatch, [first, second], ["retry", "ok"])
    assert code == "stopped_at_query"
    assert [Path(_flag(c, "--capture-manifest")).stem for c in state["detector"]] == [
        "a_3_avg5", "b_3_avg5"]
    assert "b_3_avg5" in _flag(state["query"], "--pcd")
    assert any("第 2/4 次" in t for t in state["titles"] if "rgbd_recorder" in t)


def test_live_all_bursts_non_static_exit_1(det, tmp_path, monkeypatch, capsys):
    root = tmp_path / "root"
    bursts = []
    for k in range(2):
        frames = _static_frames(5, seed=10 * k)
        frames[2] = _frame(250)
        bursts.append(_burst(root, f"m{k}", frames, stamp0=1000 * (k + 1)))
    code, state = _orchestrate(det, tmp_path, monkeypatch, bursts, [],
                               "--autolabel-v2-attempts", "2")
    assert code == 1 and state["detector"] == [] and state["query"] is None
    err = capsys.readouterr().err
    assert "2/2 次嘗試全部被拒絕" in err and "burst_not_static" in err


def test_live_reused_burst_frames_are_refused(det, tmp_path, monkeypatch, capsys):
    burst = _burst(tmp_path / "root", "cap", _static_frames(5))
    code, state = _orchestrate(det, tmp_path, monkeypatch, [burst, burst], ["retry", "ok"])
    assert code == 1 and len(state["detector"]) == 1
    assert "沒有產生新的 capture" in capsys.readouterr().err


def test_live_incomplete_burst_exit_1(det, tmp_path, monkeypatch, capsys):
    burst = _burst(tmp_path / "root", "cap", _static_frames(4))
    code, state = _orchestrate(det, tmp_path, monkeypatch, [burst], ["ok"])
    assert code == 1 and state["detector"] == []
    assert "4 組 capture manifest" in capsys.readouterr().err


def test_live_average_1_is_single_once_capture(det, tmp_path, monkeypatch):
    cap = _make_capture(tmp_path / "root", "cap_1", _frame(50), 1000)
    code, state = _orchestrate(det, tmp_path, monkeypatch, [[cap]], ["ok"],
                               "--autolabel-v2-average", "1", run_to_end=True)
    assert code == 0
    assert "--once" in state["recorder_cmds"][0]
    assert _flag(state["detector"][0], "--capture-manifest") == str(cap["manifest"].resolve())
    assert not list((tmp_path / "root").rglob("*_avg*"))
    summary = state["stamp"]["autolabel_v2_attempts"]
    assert summary["average_frames"] == 1
    assert "averaging" not in summary["attempts"][0]


def test_offline_ignores_average_with_note(det, tmp_path, monkeypatch, capsys):
    cap = _make_capture(tmp_path / "root", "cap_1", _frame(50), 1000)
    code, state = _orchestrate(det, tmp_path, monkeypatch, [[cap]], ["ok"],
                               "--autolabel-v2-average", "7", offline=True)
    assert code == "stopped_at_query"
    assert state["recorder"] == []
    assert _flag(state["detector"][0], "--image") == str(cap["image"].resolve())
    assert "--autolabel-v2-average 不適用" in capsys.readouterr().out
    assert not list((tmp_path / "root").rglob("*_avg*"))


def test_offline_default_is_single_image_without_note(det, tmp_path, monkeypatch, capsys):
    cap = _make_capture(tmp_path / "root", "cap_1", _frame(50), 1000)
    code, state = _orchestrate(det, tmp_path, monkeypatch, [[cap]], ["ok"], offline=True)
    assert code == "stopped_at_query" and state["recorder"] == []
    assert "--autolabel-v2-average 不適用" not in capsys.readouterr().out


@pytest.mark.parametrize("extra, message", [
    (["--autolabel-v2", "--autolabel-v2-average", "0"], "--autolabel-v2-average"),
    (["--autolabel-v2", "--autolabel-v2-average", "16"], "--autolabel-v2-average"),
    (["--autolabel-v2", "--average-max-diff", "0"], "--average-max-diff"),
    (["--autolabel-v2", "--average-max-diff", "nan"], "--average-max-diff"),
    (["--autolabel-v2", "--average-rate-hz", "0"], "--average-rate-hz"),
    (["--autolabel-v2", "--average-rate-hz", "31"], "--average-rate-hz"),
])
def test_average_option_validation_exit_2(extra, message):
    proc = live._run_orchestrator(*extra)
    assert proc.returncode == 2, proc.stdout
    assert message in proc.stdout
