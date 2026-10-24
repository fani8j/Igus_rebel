"""Fixed floor through the detector CLI (scene option ``live.fixed_floor``): the base_link plane is handed to the
pipeline in the camera frame, an unusable transform fails closed BEFORE SAM3 is touched, feature off when absent.
The plane maths and the cloud check are in test/core/test_fixed_floor.py."""
import importlib.util
import json
import types
from pathlib import Path

import numpy as np
import pytest

TESTS = Path(__file__).resolve().parent


def _helpers(name, fname):
    spec = importlib.util.spec_from_file_location(name, str(TESTS / fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


live = _helpers("fixedfloor_live_helpers", "test_autolabel_v2_live.py")

SCENE = ("scene: ff_gate_test\ndepth_top:\n  support: hidden_stand\nlive:\n  fixed_floor:\n"
         "    frame: base_link\n    normal: [0, 0, 1]\n    offset_m: -0.5\n    tolerance_m: 0.02\n"
         "    min_inliers: 4000\n")
SCENE_OFF = "scene: ff_gate_test\ndepth_top:\n  support: hidden_stand\n"


def _c2b(tx=0.4, ty=-0.4, tz=0.0):
    m = np.eye(4)
    m[:3, 3] = [tx, ty, tz]
    return m


@pytest.fixture()
def det():
    return live._load(live.DETECT, "autolabel_v2_detect_fixedfloor")


def _install(monkeypatch, det, rec, c2b):
    calls = live._install_fake_autolabel(monkeypatch, det, rec)
    real_import = det.import_autolabel

    def patched(repo=None):
        pipeline, _cfg, worker, overlay = real_import(repo)

        def process_capture(capture_id, paths, cfg, source, session):
            calls["cfg"] = cfg
            return pipeline.process_capture(capture_id, paths, cfg, source, session)

        return types.SimpleNamespace(process_capture=process_capture), (lambda scene=None: {}), worker, overlay

    monkeypatch.setattr(det, "import_autolabel", patched)
    monkeypatch.setattr(det, "_camera_to_base_matrix",
                        lambda calib_file=None, manifest=None, scene=None: c2b)
    return calls


def _run(det, tmp_path, monkeypatch, scene_text=SCENE, c2b=None):
    cap = live._capture(tmp_path)
    calls = _install(monkeypatch, det, live._record(), _c2b() if c2b is None else c2b)
    scene = tmp_path / "scene.yaml"
    scene.write_text(scene_text, encoding="utf-8")
    code = det.main(live._argv(cap, tmp_path, "--no-depth-edge", "--scene-config", str(scene),
                               "--allow-needs-review", "--allow-cut-blocked"))
    return code, calls


def _reject(tmp_path):
    return json.loads((tmp_path / "ov" / "cap_1_autolabel_v2_reject.json").read_text(encoding="utf-8"))


def test_plane_reaches_the_pipeline_in_the_camera_frame(det, tmp_path, monkeypatch):
    code, calls = _run(det, tmp_path, monkeypatch)
    assert code == 0
    ff = calls["cfg"]["depth_top"]["fixed_floor"]
    # base z = camera z + 0 (translation only): floor z = -0.5 -> camera-frame plane n = +z... camera 0.5 m above it
    assert ff["n"] == pytest.approx([0.0, 0.0, 1.0]) and ff["d"] == pytest.approx(0.5)
    assert ff["tolerance_m"] == 0.02 and ff["min_inliers"] == 4000 and ff["search_m"] == 0.07


def test_rotated_camera_transform_is_applied(det, tmp_path, monkeypatch):
    c2b = np.eye(4)
    c2b[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]          # camera y -> base z
    c2b[:3, 3] = [0.0, 0.0, 1.0]
    code, calls = _run(det, tmp_path, monkeypatch, c2b=c2b)
    assert code == 0
    ff = calls["cfg"]["depth_top"]["fixed_floor"]
    assert ff["n"] == pytest.approx([0.0, 1.0, 0.0]) and ff["d"] == pytest.approx(1.5)


def test_missing_transform_fails_closed_before_sam3(det, tmp_path, monkeypatch):
    cap = live._capture(tmp_path)
    calls = _install(monkeypatch, det, live._record(), None)
    (tmp_path / "scene.yaml").write_text(SCENE, encoding="utf-8")
    code = det.main(live._argv(cap, tmp_path, "--no-depth-edge", "--scene-config",
                               str(tmp_path / "scene.yaml"), "--allow-needs-review", "--allow-cut-blocked"))
    assert code == 1
    reject = _reject(tmp_path)
    assert reject["stage"] == "fixed_floor" and reject["retryable"] is False
    assert reject["reasons"][0].startswith("fixed_floor_transform_unavailable(")
    assert "started" not in calls and "query_args" not in calls         # SAM3 never touched
    assert not (tmp_path / "out" / "seam.json").exists()


def test_camera_below_the_floor_fails_closed(det, tmp_path, monkeypatch):
    code, calls = _run(det, tmp_path, monkeypatch, c2b=_c2b(tz=-3.0))   # camera 3 m below base z=0 > floor -0.5
    assert code == 1
    reject = _reject(tmp_path)
    assert reject["stage"] == "fixed_floor" and reject["reasons"][0].startswith("fixed_floor_invalid(")
    assert "started" not in calls and "query_args" not in calls


def test_feature_off_when_absent(det, tmp_path, monkeypatch):
    # the pipeline config carries no fixed floor
    code, calls = _run(det, tmp_path, monkeypatch, scene_text=SCENE_OFF)
    assert code == 0
    assert "fixed_floor" not in (calls["cfg"].get("depth_top") or {})
