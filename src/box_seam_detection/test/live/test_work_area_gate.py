"""Robot work area through the detector CLI (scene option ``live.work_area``): cloud crop before the
pipeline, hard reject of a box outside the area (never bypassed by --allow-needs-review / --allow-cut-blocked),
fail closed without a base_link transform, feature off when absent, record in ``autolabel_v2.work_area_check``."""
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


live = _helpers("workarea_live_helpers", "test_autolabel_v2_live.py")

# 0.8 m square in base_link; the fake record's top is a 200 x 150 mm rectangle at camera z = 0.5
SCENE_BODY = ("  work_area:\n    frame: base_link\n"
              "    polygon_xy_m: [[0.0, -0.8], [0.8, -0.8], [0.8, 0.0], [0.0, 0.0]]\n"
              "    top_z_range_m: [0.4, 0.6]\n    outside_keep_every: 4\n")
FLAGS = ("--allow-needs-review", "--allow-cut-blocked")


def wa_hash(area):
    from box_seam_detection.core import work_area as wa
    return wa.polygon_hash(area)


def _c2b(tx=0.4, ty=-0.4, tz=0.0):
    m = np.eye(4)
    m[:3, 3] = [tx, ty, tz]
    return m


@pytest.fixture()
def det():
    return live._load(live.DETECT, "autolabel_v2_detect_workarea")


def _write_cloud(cap, n_in=300, n_out=200):
    """Real binary PCD in base_link == camera coordinates (identity transform + c2b translation is
    applied by the test transform): inside points around the area, outside points far away."""
    from box_seam_detection.core import work_area as wa
    rng = np.random.default_rng(3)
    inside = np.column_stack([rng.uniform(0.05, 0.75, n_in), rng.uniform(-0.75, -0.05, n_in),
                              rng.uniform(0.0, 1.0, n_in)])
    outside = np.column_stack([rng.uniform(2.0, 3.0, n_out), rng.uniform(1.0, 2.0, n_out),
                               rng.uniform(0.0, 1.0, n_out)])
    cloud = np.vstack([inside, outside]).astype(np.float32) - np.array([0.4, -0.4, 0.0], dtype=np.float32)
    wa.write_pcd_xyz(cap["pcd"], cloud)
    return n_in, n_out


def _install(monkeypatch, det, rec, c2b="default"):
    calls = live._install_fake_autolabel(monkeypatch, det, rec)
    real_import = det.import_autolabel

    def patched(repo=None):
        pipeline, _cfg, worker, overlay = real_import(repo)

        def process_capture(capture_id, paths, cfg, source, session):
            from box_seam_detection.core.autolabel.pcd import read_pcd_xyz
            calls["pipeline_pcd"] = paths["pcd"]
            calls["pipeline_manifest"] = paths["manifest"]
            try:
                calls["pipeline_points"] = read_pcd_xyz(paths["pcd"])[0].shape[0]
            except Exception:  # noqa: BLE001 - feature-off tests use the fake cloud bytes
                calls["pipeline_points"] = None
            calls["pipeline_manifest_doc"] = json.loads(Path(paths["manifest"]).read_text())
            calls["cfg"] = cfg
            return pipeline.process_capture(capture_id, paths, cfg, source, session)

        return types.SimpleNamespace(process_capture=process_capture), (lambda scene=None: {}), worker, overlay

    monkeypatch.setattr(det, "import_autolabel", patched)
    monkeypatch.setattr(det, "_camera_to_base_matrix",
                        lambda calib_file=None, manifest=None, scene=None: _c2b() if isinstance(c2b, str) else c2b)
    return calls


def _scene_file(tmp_path, body=SCENE_BODY):
    path = tmp_path / "scene.yaml"
    path.write_text(f"scene: wa_gate_test\nlive:\n{body}", encoding="utf-8")
    return path


def _run(det, tmp_path, monkeypatch, rec, *extra, scene_body=SCENE_BODY, c2b="default", cloud=True):
    cap = live._capture(tmp_path)
    counts = _write_cloud(cap) if cloud else None
    calls = _install(monkeypatch, det, rec, c2b)
    scene = ["--scene-config", str(_scene_file(tmp_path, scene_body))] if scene_body is not None else []
    code = det.main(live._argv(cap, tmp_path, "--no-depth-edge", *scene, *extra))
    return code, calls, cap, counts


def _reject(tmp_path):
    return json.loads((tmp_path / "ov" / "cap_1_autolabel_v2_reject.json").read_text(encoding="utf-8"))


def _doc(tmp_path):
    return json.loads((tmp_path / "out" / "seam.json").read_text(encoding="utf-8"))


# ---- accepted + crop ---------------------------------------------------------------------

def test_inside_is_accepted_cloud_is_cropped_and_recorded(det, tmp_path, monkeypatch):
    # camera-frame corners (+-0.1, +-0.075, z 0.5) -> base (0.3..0.5, -0.475..-0.325, 0.5): inside, z in range
    code, calls, cap, (n_in, n_out) = _run(det, tmp_path, monkeypatch, live._record(), *FLAGS)
    assert code == 0
    # the pipeline ran on the cropped cloud: all inside points + every 4th outside point
    assert calls["pipeline_points"] == n_in + n_out // 4
    assert calls["pipeline_pcd"] != str(cap["pcd"]) and calls["pipeline_pcd"].startswith(str(tmp_path / "work"))
    pc = calls["pipeline_manifest_doc"]["pointcloud"]
    assert pc["path"] == calls["pipeline_pcd"] and pc["point_count"] == n_in + n_out // 4
    assert calls["pipeline_manifest_doc"]["capture_id"] == "cap_1" and pc["height"] == 1
    rt = calls["cfg"]["work_area_runtime"]
    assert rt["polygon_hash"] and rt["min_inside_frac"] == 0.8 and np.allclose(rt["camera_to_base"], _c2b())
    # the originals are untouched and the temporary cropped files are removed
    assert cap["pcd"].exists() and Path(cap["manifest"]).exists()
    assert not list((tmp_path / "work").glob("*_work_area*"))
    check = _doc(tmp_path)["autolabel_v2"]["work_area_check"]
    assert check["enabled"] is True and check["pass"] is True and check["measured"] is True
    assert check["crop"]["points_total"] == n_in + n_out and check["crop"]["points_inside"] == n_in
    assert check["crop"]["points_kept"] == n_in + n_out // 4 and check["crop"]["outside_keep_every"] == 4
    from box_seam_detection.core import scene as sc
    assert check["crop"]["time_s"] >= 0
    assert check["polygon_hash"] == wa_hash(sc.load_scene(tmp_path / "scene.yaml").work_area)
    assert check["corners_base_link"]["back_left"] == pytest.approx([0.3, -0.475, 0.5], abs=1e-3)
    assert check["xy_outside"] == [] and check["z_outside"] == [] and check["candidates_dropped"] == []
    assert _doc(tmp_path)["autolabel_v2"]["timing_s"]["work_area_crop"] >= 0


def test_debug_keeps_the_cropped_files(det, tmp_path, monkeypatch):
    code, calls, _cap, _n = _run(det, tmp_path, monkeypatch, live._record(), "--debug")
    assert code == 0
    assert Path(calls["pipeline_pcd"]).is_file() and Path(calls["pipeline_manifest"]).is_file()


def test_dropped_sam3_candidates_are_listed(det, tmp_path, monkeypatch):
    rec = live._record(attempts=[
        {"source": "sam3_live", "status": "no_coarse",
         "reason": "sam3_candidate_outside_work_area(inside=0.12<0.80)"},
        {"source": "sam3_live", "alternate_rank": 1, "status": "accepted", "selected": True}])
    code, *_ = _run(det, tmp_path, monkeypatch, rec)
    assert code == 0
    dropped = _doc(tmp_path)["autolabel_v2"]["work_area_check"]["candidates_dropped"]
    assert dropped == [{"source": "sam3_live", "alternate_rank": None,
                        "reason": "sam3_candidate_outside_work_area(inside=0.12<0.80)"}]


# ---- hard reject ----------------------------------------------------------------------------

@pytest.mark.parametrize("c2b,where", [(_c2b(tx=0.9), "xy"),       # x 0.8..1.0: outside the polygon
                                       (_c2b(ty=0.2), "xy"),       # y > 0
                                       (_c2b(tz=0.2), "z"),        # z 0.7 > 0.6
                                       (_c2b(tz=-0.2), "z")])      # z 0.3 < 0.4
@pytest.mark.parametrize("flags", [(), FLAGS])
def test_box_outside_is_rejected_even_with_override_flags(det, tmp_path, monkeypatch, c2b, where, flags):
    code, _calls, _cap, _n = _run(det, tmp_path, monkeypatch, live._record(), *flags, c2b=c2b)
    assert code == 1
    reject = _reject(tmp_path)
    assert reject["stage"] == "work_area" and reject["retryable"] is False
    assert reject["reasons"][0].startswith("box_outside_work_area(")
    assert (f"{where}: " in reject["reasons"][0])
    check = reject["details"]["work_area_check"]
    assert check["pass"] is False and (check["xy_outside"] if where == "xy" else check["z_outside"])
    assert check["crop"]["points_kept"] > 0 and reject["work_area_check"] == check
    assert not (tmp_path / "out" / "seam.json").exists()


def test_needs_review_record_outside_is_still_rejected_with_flags(det, tmp_path, monkeypatch):
    rec = live._record(needs_review=True, reasons=["depth_top_ambiguous_unresolved"])
    code, *_ = _run(det, tmp_path, monkeypatch, rec, *FLAGS, c2b=_c2b(tx=0.9))
    assert code == 1 and _reject(tmp_path)["stage"] == "work_area"


def test_one_corner_outside_is_enough(det, tmp_path, monkeypatch):
    rec = live._record()
    rec["top_face"]["corners_xyz_camera_m"]["front_right"] = [0.45, 0.075, 0.5]     # base x 0.85
    code, *_ = _run(det, tmp_path, monkeypatch, rec, *FLAGS, "--no-rect-check")
    assert code == 1
    reject = _reject(tmp_path)
    assert reject["details"]["work_area_check"]["xy_outside"] == ["front_right"]
    assert "front_right=(0.850,-0.325)" in reject["reasons"][0]


def test_gap_anchored_rgb_quad_is_checked_too(det, tmp_path, monkeypatch):
    rec = live._record()
    rec["top_face"]["top_from_gap"] = {"applied": True, "rgb_quad": {"corners_xyz_camera_m": {
        **{k: list(v) for k, v in live.RECT_XYZ.items()}, "back_right": [0.5, -0.075, 0.5]}}}
    code, *_ = _run(det, tmp_path, monkeypatch, rec, "--no-rect-check")
    assert code == 1
    assert _reject(tmp_path)["details"]["work_area_check"]["xy_outside"] == ["rgb_back_right"]


# ---- fail closed ---------------------------------------------------------------------------------

def test_missing_transform_fails_closed_before_sam3(det, tmp_path, monkeypatch):
    code, calls, _cap, _n = _run(det, tmp_path, monkeypatch, live._record(), *FLAGS, c2b=None)
    assert code == 1
    reject = _reject(tmp_path)
    assert reject["stage"] == "work_area" and reject["retryable"] is False
    assert reject["reasons"][0].startswith("work_area_transform_unavailable(")
    assert not reject["reasons"][0].startswith("box_outside_work_area")
    assert reject["details"]["work_area_check"]["measured"] is False
    assert "started" not in calls and "query_args" not in calls         # SAM3 never touched
    assert not (tmp_path / "out" / "seam.json").exists()


def test_unmeasurable_corners_fail_closed(det, tmp_path, monkeypatch):
    rec = live._record()
    rec["top_face"]["corners_xyz_camera_m"] = None
    code, *_ = _run(det, tmp_path, monkeypatch, rec, *FLAGS, "--no-rect-check")
    assert code == 1
    reject = _reject(tmp_path)
    assert reject["stage"] == "work_area" and reject["reasons"][0].startswith("work_area_unmeasurable(")
    assert reject["retryable"] is False


def test_unreadable_cloud_is_a_prerequisite_failure(det, tmp_path, monkeypatch, capsys):
    code, *_ = _run(det, tmp_path, monkeypatch, live._record(), cloud=False)      # the "fake-pcd" bytes
    assert code == 2
    assert "work-area crop failed" in capsys.readouterr().err
    assert not (tmp_path / "out" / "seam.json").exists()


# ---- feature off -----------------------------------------------------------------------------------

@pytest.mark.parametrize("scene_body", [None, "  rect_tol_mm: null\n", "  work_area: null\n"])
def test_feature_off_when_absent(det, tmp_path, monkeypatch, scene_body):
    # the originals are read as is: the transform is not even consulted, the box may be anywhere
    code, calls, cap, _n = _run(det, tmp_path, monkeypatch, live._record(), *FLAGS, scene_body=scene_body,
                                c2b=_c2b(tx=50.0), cloud=False)
    assert code == 0
    assert calls["paths"]["pcd"] == str(cap["pcd"]) and calls["paths"]["manifest"] == str(cap["manifest"])
    assert "work_area_runtime" not in calls["cfg"]
    assert _doc(tmp_path)["autolabel_v2"]["work_area_check"] == {"enabled": False}
    assert "work_area_crop" not in _doc(tmp_path)["autolabel_v2"]["timing_s"]
