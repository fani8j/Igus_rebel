"""Live-pipeline integration of the autolabel_v2 detector.

Covers box_seam_detection.live.detector (contract corner relabel, published
seam line, reject/prerequisite exit codes, no stale output) and
box_seam_detection.live.orchestrator (prerequisite exit 2, exact subprocess
command). SAM3 and the autolabel_v2 pipeline are faked: no GPU, camera or TF
needed. (Ported from rgbd_captures/tests; the module-under-test names below are
the package modules.)
"""

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

DETECT = "box_seam_detection.live.detector"
ORCHESTRATOR = "box_seam_detection.live.orchestrator"
RING = ("back_left", "back_right", "front_right", "front_left")
W, H = 640, 480
FRAME_ID = "camera_color_optical_frame"


def _load(target, name=None):
    """The package module ``target`` (dotted name); ``name`` is kept for the old call sites."""
    return importlib.import_module(target)


@pytest.fixture()
def det():
    return importlib.import_module(DETECT)


def _mid(a, b):
    return [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0]


def _signed_area(points):
    area = 0.0
    for index in range(4):
        x0, y0 = points[index]
        x1, y1 = points[(index + 1) % 4]
        area += x0 * y1 - x1 * y0
    return area / 2.0


# Rule-named quad (image coords, v down): back edge on top.
RULE = {"back_left": [100.0, 100.0], "back_right": [300.0, 110.0],
        "front_right": [310.0, 250.0], "front_left": [90.0, 240.0]}


# 3-D top face (rule names): a 200 x 150 mm rectangle at z = 0.5 m.
RECT_XYZ = {"back_left": [-0.10, -0.075, 0.5], "back_right": [0.10, -0.075, 0.5],
            "front_right": [0.10, 0.075, 0.5], "front_left": [-0.10, 0.075, 0.5]}


def _record(corners=None, status="ok", axis="lr", endpoints=None,
            needs_review=False, wrong_face=False, mask_source="sam3_live",
            attempts=None, reasons=None):
    corners = corners or RULE
    if endpoints is None:
        if axis == "lr":
            endpoints = (_mid(corners["back_left"], corners["front_left"]),
                         _mid(corners["back_right"], corners["front_right"]))
        else:
            endpoints = (_mid(corners["back_left"], corners["back_right"]),
                         _mid(corners["front_left"], corners["front_right"]))
    left, right = sorted(endpoints, key=lambda p: (p[0], p[1]))
    eps = {"seam_left": left, "seam_right": right}
    seam = {"status": status, "axis": axis if status in ("ok", "estimated") else None,
            "endpoints_px": eps if status == "ok" else None,
            "score": 0.7 if status == "ok" else 0.0,
            "source": "prior_confirmed_by_gap" if status == "ok" else "prior",
            "estimate": {"prior_endpoints_px": eps, "axis": axis,
                         "estimate": {"estimate_endpoints_px": eps}}}
    return {
        "schema": "box_autolabel/v5",
        "provenance": {"mask_source": mask_source, "attempts": attempts or [],
                       "sam3": {"prompt": "cardboard box top face", "concept_score": 0.9,
                                "model_revision": "rev"},
                       "corner_names": {"rule": "long_side_back_v1", "ambiguous": False,
                                        "flags": []}},
        "top_face": {"corners_px": dict(corners),
                     "corners_xyz_camera_m": {k: list(v) for k, v in RECT_XYZ.items()},
                     "plane": None},
        "seam": seam,
        "checks": {"wrong_face_suspect": wrong_face, "needs_review": needs_review,
                   "corner_flags": {"corners": {}, "any_flag": False},
                   "reasons": list(reasons or [])},
        "confidence": 0.85, "confidence_quad": 0.85, "confidence_seam": 0.7,
    }


# ------------------------------------------------------------ relabel

def test_relabel_lr_keeps_rule_names(det):
    rec = _record(axis="lr")
    seam = det.published_seam_line(rec)
    rel = det.contract_relabel(rec["top_face"]["corners_px"], seam["seam_left"],
                               seam["seam_right"])
    assert rel["shift"] == 0
    assert rel["contract_from_rule"] == {k: k for k in RING}
    assert rel["corners"] == RULE
    assert rel["left_mid"] == _mid(RULE["back_left"], RULE["front_left"])
    assert rel["right_mid"] == _mid(RULE["back_right"], RULE["front_right"])
    assert rel["geometric_axis"] == "lr"
    assert max(rel["edge_distance_px"].values()) < 1e-9


def test_relabel_lr_keeps_rule_names_even_when_rule_left_is_image_right(det):
    # Box seen "upside down": rule left edge is on the image right.
    flipped = {"back_left": [300.0, 250.0], "back_right": [100.0, 240.0],
               "front_right": [100.0, 100.0], "front_left": [300.0, 110.0]}
    rec = _record(corners=flipped, axis="lr")
    seam = det.published_seam_line(rec)
    rel = det.contract_relabel(flipped, seam["seam_left"], seam["seam_right"])
    assert rel["shift"] == 0
    assert rel["left_mid"] == _mid(flipped["back_left"], flipped["front_left"])
    assert rel["left_mid"][0] > rel["right_mid"][0]


@pytest.mark.parametrize("corners", [
    RULE,
    # Same box, rule names rotated the other way round the ring.
    {"back_left": [310.0, 250.0], "back_right": [90.0, 240.0],
     "front_right": [100.0, 100.0], "front_left": [300.0, 110.0]},
])
def test_relabel_bf_rotates_to_contract_left_right(det, corners):
    rec = _record(corners=corners, axis="bf")
    seam = det.published_seam_line(rec)
    rel = det.contract_relabel(corners, seam["seam_left"], seam["seam_right"])
    assert rel["shift"] in (1, 3)
    assert rel["geometric_axis"] == "bf"
    c = rel["corners"]
    # Cyclic relabel of the same 4 points with the same ring orientation.
    assert sorted(map(tuple, c.values())) == sorted(map(tuple, corners.values()))
    assert (_signed_area([c[k] for k in RING]) > 0) == (
        _signed_area([corners[k] for k in RING]) > 0)
    for key, rule in rel["contract_from_rule"].items():
        assert c[key] == corners[rule]
    # left_mid on contract left edge, right_mid on contract right edge.
    assert rel["left_mid"] == pytest.approx(_mid(c["back_left"], c["front_left"]))
    assert rel["right_mid"] == pytest.approx(_mid(c["back_right"], c["front_right"]))
    # Of the two valid rotations, left_mid is the image-left endpoint
    # (smaller u, ties broken by smaller v).
    assert tuple(rel["left_mid"]) < tuple(rel["right_mid"])


def test_relabel_rejects_endpoint_off_edge(det):
    good = _mid(RULE["back_left"], RULE["front_left"])
    inside = [good[0] + 5.0, good[1]]
    right = _mid(RULE["back_right"], RULE["front_right"])
    with pytest.raises(det.Reject) as info:
        det.contract_relabel(RULE, inside, right, tol_px=2.0)
    assert info.value.stage == "contract_edges"
    # Within tolerance is accepted.
    near = [good[0] + 1.5, good[1]]
    rel = det.contract_relabel(RULE, near, right, tol_px=2.0)
    assert rel["shift"] == 0


def test_published_seam_line_sources(det):
    ok = det.published_seam_line(_record(status="ok"))
    assert ok["observed"] is True and ok["source"] == "seam.endpoints_px"
    est_rec = _record(status="estimated")
    est_rec["seam"]["estimate"]["estimate"]["estimate_endpoints_px"] = {
        "seam_left": [95.0, 170.0], "seam_right": [305.0, 180.0]}
    est = det.published_seam_line(est_rec)
    assert est["observed"] is False
    assert est["seam_left"] == [95.0, 170.0]
    assert est["source"] == "seam.estimate.estimate.estimate_endpoints_px"
    for status in ("ambiguous", "invisible"):
        with pytest.raises(det.Reject):
            det.published_seam_line(_record(status=status))


# ------------------------------------------------------------ CLI main

def _capture(tmp_path):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    image = tmp_path / "rgb" / "cap_1.png"
    pcd = tmp_path / "pointcloud" / "cap_1.pcd"
    manifest_path = tmp_path / "captures" / "cap_1.json"
    for path in (image, pcd, manifest_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(image), np.zeros((H, W, 3), dtype=np.uint8))
    pcd.write_bytes(b"fake-pcd")
    manifest = {
        "schema": "rgbd_capture/v1", "capture_id": "cap_1",
        "image": {"path": str(image), "width": W, "height": H,
                  "header": {"frame_id": FRAME_ID, "stamp_ns": 1000}},
        "pointcloud": {"path": str(pcd), "header": {"frame_id": FRAME_ID, "stamp_ns": 1001}},
        "camera_info": {"header": {"frame_id": FRAME_ID}, "width": W, "height": H},
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    sam3_python = tmp_path / "fake_python"
    sam3_python.write_text("", encoding="utf-8")
    return {"image": image, "pcd": pcd, "manifest": manifest_path,
            "manifest_dict": manifest, "sam3_python": sam3_python}


def _install_fake_autolabel(monkeypatch, det, record, worker_fails=False):
    calls = {}

    class FakeWorker:
        def __init__(self, python_bin, stderr_log=None, **_):
            calls["python_bin"] = python_bin

        def start(self):
            if worker_fails:
                raise RuntimeError("sam3 worker unavailable: weights absent")
            calls["started"] = True
            return self

        def query(self, *a, **k):
            calls["query_args"] = a
            calls["query_kwargs"] = k
            return {"ok": True}

        def close(self):
            calls["closed"] = True

    class FakePipeline:
        @staticmethod
        def process_capture(capture_id, paths, cfg, source, session):
            calls["source"] = source
            calls["paths"] = paths
            session.query(paths["image"])
            return json.loads(json.dumps(record)), {}

    def fake_write_overlay(path, bgr, rec, debug):
        calls["record_overlay"] = str(path)

    monkeypatch.setattr(det, "import_autolabel",
                        lambda repo=None: (FakePipeline, lambda: {}, FakeWorker,
                                      fake_write_overlay))
    return calls


def _argv(cap, tmp_path, *extra):
    return ["--image", str(cap["image"]), "--pcd", str(cap["pcd"]),
            "--capture-manifest", str(cap["manifest"]),
            "--out-seam-json", str(tmp_path / "out" / "seam.json"),
            "--work-dir", str(tmp_path / "work"),
            "--overlay-dir", str(tmp_path / "ov"),
            "--sam3-python", str(cap["sam3_python"]), *extra]


def _stale(tmp_path):
    out = tmp_path / "out" / "seam.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('{"stale": true}', encoding="utf-8")
    return out


def test_accept_writes_canonical_doc_that_orchestrator_accepts(det, tmp_path, monkeypatch):
    cap = _capture(tmp_path)
    calls = _install_fake_autolabel(monkeypatch, det, _record(axis="bf", status="estimated"))
    out = _stale(tmp_path)
    assert det.main(_argv(cap, tmp_path)) == 0
    assert calls["source"] == "sam3_live" and calls["closed"]
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["method"] == "autolabel_v2_sam3_fold_midline"
    assert doc["lr_ambiguous"] is False and doc["incomplete_groups"] == []
    assert doc["seam_status"] == "estimated" and doc["seam_observed"] is False
    a2 = doc["autolabel_v2"]
    assert a2["rule_corner_names"] == RULE
    assert a2["contract_shift"] in (1, 3)
    for key, rule in a2["contract_from_rule"].items():
        assert doc[key] == RULE[rule]
    assert Path(a2["record_path"]).is_file()
    assert set(a2["timing_s"]) >= {"sam3_load", "sam3_inference", "pipeline_total"}
    assert (tmp_path / "ov" / "cap_1_autolabel_v2.png").is_file()
    assert calls["record_overlay"].endswith("cap_1_autolabel_v2_record.png")
    orch = _load(ORCHESTRATOR, "capture_and_query_depth")
    orch.validate_detection(out, str(cap["image"]), cap["manifest_dict"])


@pytest.mark.parametrize("record, reason", [
    (lambda: {**_record(), "top_face": {"corners_px": None, "corners_xyz_camera_m": None,
                                        "plane": None}}, "no_top_face"),
    (lambda: _record(wrong_face=True), "wrong_face_suspect"),
    (lambda: _record(needs_review=True, reasons=["no_independent_depth_check"]),
     "needs_review"),
    (lambda: _record(status="ambiguous"), "seam_status_ambiguous"),
    (lambda: _record(mask_source="depth_only"), "mask_source_depth_only"),
])
def test_scene_rejections_exit_1_without_stale_output(det, tmp_path, monkeypatch,
                                                      record, reason):
    cap = _capture(tmp_path)
    _install_fake_autolabel(monkeypatch, det, record())
    out = _stale(tmp_path)
    assert det.main(_argv(cap, tmp_path)) == 1
    assert not out.exists()
    diag = json.loads((tmp_path / "ov" / "cap_1_autolabel_v2_reject.json").read_text())
    assert diag["status"] == "rejected" and diag["capture_id"] == "cap_1"
    assert any(str(r).startswith(reason) for r in diag["reasons"]), diag


def test_needs_review_allowed_with_flag(det, tmp_path, monkeypatch):
    cap = _capture(tmp_path)
    _install_fake_autolabel(monkeypatch, det, _record(needs_review=True))
    out = _stale(tmp_path)
    assert det.main(_argv(cap, tmp_path, "--allow-needs-review")) == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["autolabel_v2"]["needs_review"] is True
    assert doc["autolabel_v2"]["allow_needs_review"] is True


def test_sam3_worker_unavailable_exit_2(det, tmp_path, monkeypatch):
    cap = _capture(tmp_path)
    _install_fake_autolabel(monkeypatch, det, _record(), worker_fails=True)
    out = _stale(tmp_path)
    assert det.main(_argv(cap, tmp_path)) == 2
    assert not out.exists()


def test_sam3_live_transport_failure_in_record_exit_2(det, tmp_path, monkeypatch):
    cap = _capture(tmp_path)
    rec = {**_record(), "top_face": {"corners_px": None, "corners_xyz_camera_m": None,
                                     "plane": None}}
    rec["provenance"]["attempts"] = [{"source": "sam3_live", "status": "no_coarse",
                                      "reason": "sam3_live_unavailable(read timed out)"}]
    _install_fake_autolabel(monkeypatch, det, rec)
    out = _stale(tmp_path)
    assert det.main(_argv(cap, tmp_path)) == 2
    assert not out.exists()


def test_missing_sam3_interpreter_and_inputs_exit_2(det, tmp_path, monkeypatch):
    cap = _capture(tmp_path)
    _install_fake_autolabel(monkeypatch, det, _record())
    out = _stale(tmp_path)
    argv = _argv(cap, tmp_path)
    argv[argv.index("--sam3-python") + 1] = str(tmp_path / "absent_python")
    assert det.main(argv) == 2
    assert not out.exists()
    _stale(tmp_path)
    argv = _argv(cap, tmp_path)
    argv[argv.index("--pcd") + 1] = str(tmp_path / "absent.pcd")
    assert det.main(argv) == 2
    assert not out.exists()


def test_manifest_image_binding_mismatch_exit_2(det, tmp_path, monkeypatch):
    cap = _capture(tmp_path)
    _install_fake_autolabel(monkeypatch, det, _record())
    other = tmp_path / "other.png"
    other.write_bytes(cap["image"].read_bytes())
    argv = _argv(cap, tmp_path)
    argv[argv.index("--image") + 1] = str(other)
    assert det.main(argv) == 2


def test_sam3_timeout_is_forwarded_to_session_query(det, tmp_path, monkeypatch):
    cap = _capture(tmp_path)
    calls = _install_fake_autolabel(monkeypatch, det, _record(axis="bf"))
    assert det.main(_argv(cap, tmp_path, "--sam3-timeout", "42")) == 0
    assert calls["query_kwargs"] == {"timeout_s": 42}
    assert det.main(_argv(cap, tmp_path)) == 0
    assert calls["query_kwargs"] == {"timeout_s": 300}


@pytest.mark.parametrize("tol", ["0", "-1", "5.01", "nan", "inf"])
def test_invalid_edge_tol_exit_2_without_stale_output(det, tmp_path, monkeypatch, tol):
    cap = _capture(tmp_path)
    calls = _install_fake_autolabel(monkeypatch, det, _record(axis="bf"))
    out = _stale(tmp_path)
    assert det.main(_argv(cap, tmp_path, "--edge-tol-px", tol)) == 2
    assert not out.exists() and "started" not in calls


def test_edge_tol_upper_bound_accepted(det, tmp_path, monkeypatch):
    cap = _capture(tmp_path)
    _install_fake_autolabel(monkeypatch, det, _record(axis="bf"))
    assert det.main(_argv(cap, tmp_path, "--edge-tol-px", "5")) == 0


@pytest.mark.parametrize("where", ["pipeline", "doc"])
def test_unexpected_exception_exit_2_not_rejection(det, tmp_path, monkeypatch, capsys, where):
    cap = _capture(tmp_path)
    calls = _install_fake_autolabel(monkeypatch, det, _record(axis="bf"))
    if where == "doc":
        def boom(*_a, **_k):
            raise KeyError("corners_px")
        monkeypatch.setattr(det, "build_canonical_seam_doc", boom)
    else:
        pipeline, cfg, worker, overlay = det.import_autolabel(None)

        class BrokenPipeline:
            @staticmethod
            def process_capture(*_a, **_k):
                raise ZeroDivisionError("float division by zero")
        monkeypatch.setattr(det, "import_autolabel",
                            lambda repo=None: (BrokenPipeline, cfg, worker, overlay))
    out = _stale(tmp_path)
    assert det.main(_argv(cap, tmp_path)) == 2
    err = capsys.readouterr().err
    assert "internal error" in err and "not a scene rejection" in err
    assert not out.exists()
    assert not list(tmp_path.rglob("*_reject.json"))      # not reported as a rejection
    if where == "pipeline":
        assert calls.get("closed") is True               # worker still shut down


def test_atomic_writer_removes_temp_on_serialisation_error(det, tmp_path):
    target = tmp_path / "x.json"
    with pytest.raises(TypeError):
        det.write_json_atomically(target, {"bad": object()})
    assert list(tmp_path.iterdir()) == []


def test_detect_cli_help_runs_without_heavy_imports():
    proc = subprocess.run([sys.executable, "-m", DETECT, "--help"],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, timeout=60, env=_pkg_env())
    assert proc.returncode == 0
    assert "--allow-needs-review" in proc.stdout


# ------------------------------------------------------------ orchestrator

def _pkg_env():
    """Environment of a `python -m box_seam_detection...` subprocess (package root on PYTHONPATH)."""
    import os
    root = str(Path(__file__).resolve().parents[2])
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([root, *[p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]])
    return env


def _run_orchestrator(*argv):
    return subprocess.run([sys.executable, "-m", ORCHESTRATOR, *argv],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, timeout=120, env=_pkg_env())


# (The legacy-branch mutual-exclusion / "option requires --autolabel-v2" tests of the old orchestrator are not
# ported: those branches and flags no longer exist; --autolabel-v2 is an accepted no-op.)

def test_orchestrator_prerequisite_missing_script_exit_2(tmp_path):
    out_root = tmp_path / "rgbd_captures"
    proc = _run_orchestrator("--autolabel-v2", "--output-dir", str(out_root),
                             "--autolabel-v2-script", str(tmp_path / "absent.py"))
    assert proc.returncode == 2
    assert "autolabel_v2_detect.py" in proc.stdout or "autolabel_v2_detect.py" in proc.stderr
    assert not out_root.exists()


STOP_CODE = 97       # exit code of a fake step that halts the orchestrator (replaces the old _Stop exception)


class _Stop(Exception):
    pass


def _halt(mod, title):
    """Halt run_detection from inside a faked step (the run turns it into a DetectResult / exit code)."""
    raise mod.DetectionAbort(STOP_CODE, title)


def _calib(tmp_path):
    """Minimal valid eye_on_base calib so the orchestrator's calib preflight passes
    without depending on the production file."""
    path = tmp_path / "test.calib"
    path.write_text(
        "parameters:\n  name: t\n  calibration_type: eye_on_base\n"
        "  robot_base_frame: base_link\n  tracking_base_frame: camera_link\n"
        "transform:\n  translation: {x: 0.0, y: 0.0, z: 0.5}\n"
        "  rotation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}\n", encoding="utf-8")
    return path


def test_orchestrator_branch_builds_detector_command(det, tmp_path, monkeypatch):
    cap = _capture(tmp_path)
    mod = _load(ORCHESTRATOR, "capture_and_query_depth")
    script = tmp_path / "autolabel_v2_detect.py"
    script.write_text("# fake (subprocess is stubbed)\n", encoding="utf-8")
    probes = []

    class _Probe:
        returncode = 0
        stdout = ""

    def fake_subprocess_run(cmd, *a, **k):
        probes.append(cmd)
        return _Probe()

    monkeypatch.setattr(mod.subprocess, "run", fake_subprocess_run)
    seen = []

    def fake_run_step(title, cmd, capture=False, check=True):
        seen.append((title, cmd))
        if "autolabel_v2" in title:
            out = Path(cmd[cmd.index("--out-seam-json") + 1])
            rec = _record(axis="bf")
            seam = det.published_seam_line(rec)
            rel = det.contract_relabel(rec["top_face"]["corners_px"],
                                       seam["seam_left"], seam["seam_right"])
            doc = det.build_canonical_seam_doc(
                rec, str(cap["image"]), W, H, "cap_1", seam, rel, "rec.json", {}, False,
                cut_permission={"status": "allowed", "reasons": []})
            out.write_text(json.dumps(doc), encoding="utf-8")
            return None
        _halt(mod, title)

    monkeypatch.setattr(mod, "run_step", fake_run_step)
    code = mod.main([
        "--downstream-inprocess", "off", "--autolabel-v2",
        "--autolabel-v2-script", str(script),
        "--autolabel-v2-sam3-python", str(cap["sam3_python"]),
        "--allow-needs-review", "--debug",
        "--from-image", str(cap["image"]), "--pcd", str(cap["pcd"]),
        "--capture-manifest", str(cap["manifest"]),
        "--calib-file", str(_calib(tmp_path)),
        "--output-dir", str(tmp_path / "root"),
    ])
    assert code == STOP_CODE
    # Prerequisite probe: the SAM3 interpreter (the detector environment is imported in-process now).
    assert probes[0][0] == str(cap["sam3_python"])
    title, cmd = seen[0]
    assert cmd[:2] == [sys.executable, str(script.resolve())]
    flag = lambda name: cmd[cmd.index(name) + 1]  # noqa: E731
    assert flag("--image") == str(cap["image"])
    assert flag("--pcd") == str(cap["pcd"])
    assert flag("--capture-manifest") == str(cap["manifest"])
    assert flag("--sam3-python") == str(cap["sam3_python"])
    out = Path(flag("--out-seam-json"))
    assert out.name == "cap_1_autolabel_v2_seam.json"
    assert Path(flag("--work-dir")) == out.parent
    assert Path(flag("--overlay-dir")) == out.parent / "autolabel_v2_overlays"
    assert "--allow-needs-review" in cmd and "--debug" in cmd
    # Continues exactly like the guided branch: query depth with this seam JSON.
    q_title, q_cmd = seen[1]
    assert q_cmd[1:3] == ["-m", "box_seam_detection.live.query_depth"]
    assert q_cmd[q_cmd.index("--seam-json") + 1] == str(out)


def test_orchestrator_branch_rejection_exit_1_no_depth(det, tmp_path, monkeypatch):
    cap = _capture(tmp_path)
    mod = _load(ORCHESTRATOR, "capture_and_query_depth")
    script = tmp_path / "autolabel_v2_detect.py"
    script.write_text("# fake\n", encoding="utf-8")

    class _Probe:
        returncode = 0
        stdout = ""

    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **k: _Probe())
    seen = []

    def fake_run_step(title, cmd, capture=False, check=True):
        seen.append(title)
        if "autolabel_v2" in title:
            return None  # detector "rejected": no output file
        raise AssertionError(f"must not reach {title}")

    monkeypatch.setattr(mod, "run_step", fake_run_step)
    code = mod.main([
        "--downstream-inprocess", "off", "--autolabel-v2",
        "--autolabel-v2-script", str(script),
        "--autolabel-v2-sam3-python", str(cap["sam3_python"]),
        "--from-image", str(cap["image"]), "--pcd", str(cap["pcd"]),
        "--capture-manifest", str(cap["manifest"]),
        "--calib-file", str(_calib(tmp_path)),
        "--output-dir", str(tmp_path / "root"),
    ])
    assert code == 1
    assert len(seen) == 1
