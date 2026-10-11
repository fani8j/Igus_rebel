"""Regression: the orchestrator must initialise output_dir before both capture branches.

Covers the user-reproduced live bug (``NameError: name 'output_dir' is not defined`` when building
the recorder command; the offline ``pipeline_root`` fallback had the same dangling reference).
All tests run without a camera: the recorder / subprocess / detection steps are stubbed or invoked
as failing preflight. (Ported from the old capture_and_query_depth.py tests; the learned-depth-first
variants became the --autolabel-v2 flow.)
"""

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))
import test_autolabel_v2_live as live  # noqa: E402

ORCHESTRATOR = live.ORCHESTRATOR


class _Probe:
    returncode = 0
    stdout = ""


def _load_orchestrator():
    return importlib.import_module(ORCHESTRATOR)


def test_live_branch_passes_output_dir_to_recorder(tmp_path, monkeypatch):
    """Live flow wires --output-dir into the recorder command (no camera)."""
    mod = _load_orchestrator()
    out_root = tmp_path / "rgbd_captures"
    script = tmp_path / "autolabel_v2_detect.py"
    script.write_text("# fake\n", encoding="utf-8")
    sam3 = tmp_path / "fake_python"
    sam3.write_text("", encoding="utf-8")
    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **k: _Probe())
    seen = {}

    def fake_run_step(title, cmd, capture=False, check=True):
        seen["title"] = title
        seen["cmd"] = cmd
        live._halt(mod, "halted at recorder step")

    monkeypatch.setattr(mod, "run_step", fake_run_step)
    code = mod.main([
        "--downstream-inprocess", "off", "--output-dir", str(out_root),
        "--autolabel-v2-script", str(script), "--autolabel-v2-sam3-python", str(sam3),
        "--autolabel-v2-server", "off", "--calib-file", str(live._calib(tmp_path)),
    ])
    assert code == live.STOP_CODE       # pre-fix: NameError: name 'output_dir' is not defined
    assert "拍照" in seen["title"]
    assert seen["cmd"][1:3] == ["-m", "box_seam_detection.live.recorder"]
    assert seen["cmd"][3] == str(out_root.resolve())
    assert out_root.is_dir()


def test_offline_pipeline_root_falls_back_to_output_dir(tmp_path, monkeypatch):
    """Offline flow uses --output-dir when the manifest is not under captures/."""
    mod = _load_orchestrator()
    out_root = tmp_path / "rgbd_captures"
    image = tmp_path / "frame.png"
    image.write_bytes(b"fake-image")
    cloud = tmp_path / "frame.pcd"
    cloud.write_bytes(b"fake-pcd")
    manifest = tmp_path / "other" / "manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("{}", encoding="utf-8")
    pipe_dir = tmp_path / "pipe"
    pipe_dir.mkdir()
    script = tmp_path / "autolabel_v2_detect.py"
    script.write_text("# fake\n", encoding="utf-8")
    sam3 = tmp_path / "fake_python"
    sam3.write_text("", encoding="utf-8")
    seen = {}
    monkeypatch.setattr(mod, "find_capture_manifest", lambda *a, **k: manifest)
    monkeypatch.setattr(mod, "load_capture_manifest", lambda *a, **k: {
        "capture_id": "cap_001", "image": {"header": {"stamp_ns": 5}, "width": 1, "height": 1}})
    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **k: _Probe())

    def fake_output_dir(root, capture_id):
        seen["root"] = Path(root)
        return pipe_dir

    monkeypatch.setattr(mod, "detection_output_directory", fake_output_dir)

    def fake_run_step(title, cmd, capture=False, check=True):
        live._halt(mod, "halted at the detector step")

    monkeypatch.setattr(mod, "run_step", fake_run_step)
    code = mod.main([
        "--downstream-inprocess", "off", "--output-dir", str(out_root),
        "--from-image", str(image), "--pcd", str(cloud),
        "--autolabel-v2-script", str(script), "--autolabel-v2-sam3-python", str(sam3),
        "--calib-file", str(live._calib(tmp_path)),
    ])
    assert code == live.STOP_CODE       # pre-fix: NameError at the pipeline_root fallback
    assert seen["root"] == out_root.resolve()


def test_predict_out_dir_redirects_the_pipeline_dirs(tmp_path, monkeypatch):
    mod = _load_orchestrator()
    image = tmp_path / "rgb" / "frame.png"
    image.parent.mkdir()
    image.write_bytes(b"x")
    cloud = tmp_path / "frame.pcd"
    cloud.write_bytes(b"x")
    manifest = tmp_path / "captures" / "frame.json"
    manifest.parent.mkdir()
    manifest.write_text("{}", encoding="utf-8")
    script = tmp_path / "d.py"
    script.write_text("#", encoding="utf-8")
    sam3 = tmp_path / "fake_python"
    sam3.write_text("", encoding="utf-8")
    seen = {}
    monkeypatch.setattr(mod, "find_capture_manifest", lambda *a, **k: manifest)
    monkeypatch.setattr(mod, "load_capture_manifest", lambda *a, **k: {
        "capture_id": "cap_001", "image": {"header": {"stamp_ns": 5}, "width": 1, "height": 1}})
    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **k: _Probe())
    pipe = tmp_path / "pipe"
    pipe.mkdir()

    def fake_output_dir(root, capture_id):
        seen["root"] = Path(root)
        return pipe

    monkeypatch.setattr(mod, "detection_output_directory", fake_output_dir)
    monkeypatch.setattr(mod, "run_step", lambda *a, **k: live._halt(mod, "stop"))
    code = mod.main(["--downstream-inprocess", "off", "--output-dir", str(tmp_path / "o"),
                     "--predict-out-dir", str(tmp_path / "redirected"),
                     "--from-image", str(image), "--pcd", str(cloud),
                     "--autolabel-v2-script", str(script), "--autolabel-v2-sam3-python", str(sam3),
                     "--calib-file", str(live._calib(tmp_path))])
    assert code == live.STOP_CODE
    assert seen["root"] == (tmp_path / "redirected").resolve()     # not next to captures/


def test_preflight_failure_creates_no_output_dir(tmp_path):
    """An unusable SAM3 interpreter is rejected before any output-dir side effect (exit 2)."""
    out_root = tmp_path / "rgbd_captures"
    proc = live._run_orchestrator("--output-dir", str(out_root),
                                  "--autolabel-v2-sam3-python", "/tmp/absent_python")
    assert proc.returncode == 2
    assert "SAM3" in proc.stdout
    assert not out_root.exists()
