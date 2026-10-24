"""Live-pipeline tests: lens distortion in live.query_depth and the
calibration-file camera->base_link transform in live.base_link (incl. the capture-manifest
``driver_extrinsic`` block written by the ROS capture node). No camera / ROS needed.
"""

import hashlib
import importlib
import os
import json
import math
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

QUERY_SCRIPT = "box_seam_detection.live.query_depth"
BASE_SCRIPT = "box_seam_detection.live.base_link"
ENV = {**os.environ, "PYTHONPATH": os.pathsep.join(
    [str(Path(__file__).resolve().parents[2]), *[p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]])}
REAL_CALIB = Path("/home/farhan/.ros2/easy_handeye2/calibrations/rebel_handeye_h_v2.calib")

NAMES = ("back_left", "back_right", "front_left", "front_right", "left_mid", "right_mid")
D_ORBBEC = [0.007051904685795307, -0.049205388873815536, 0.00026131237973459065,
            9.177037281915545e-05, 0.03320758789777756, 0.0, 0.0, 0.0]
K = [691.8780517578125, 0.0, 651.336669921875, 0.0, 691.772705078125, 365.9878845214844, 0.0, 0.0, 1.0]


def _load(target, name=None):
    return importlib.import_module(target)


@pytest.fixture(scope="module")
def query():
    return _load(QUERY_SCRIPT, "query_depth_at_pixel_under_test")


@pytest.fixture(scope="module")
def base():
    return _load(BASE_SCRIPT, "camera_point_to_base_link_under_test")


def _camera_info(d):
    fx, _, cx, _, fy, cy = K[:6]
    return {
        "width": 1280, "height": 720, "distortion_model": "plumb_bob", "d": list(d), "k": list(K),
        "p": [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0],
        "header": {"frame_id": "camera_color_optical_frame", "stamp_ns": 1},
    }


# Tilted carton top plane in the optical frame: n . X + offset = 0.
PLANE_NORMAL = np.array([0.02, -0.70, -0.71])
PLANE_NORMAL = PLANE_NORMAL / np.linalg.norm(PLANE_NORMAL)
PLANE_POINT = np.array([-0.10, 0.02, 0.46])
PLANE_OFFSET = -float(PLANE_NORMAL @ PLANE_POINT)


def _plane_point(x, y):
    """Point on the plane with given camera x, y (solve for z)."""
    z = -(PLANE_OFFSET + PLANE_NORMAL[0] * x + PLANE_NORMAL[1] * y) / PLANE_NORMAL[2]
    return np.array([x, y, z])


def _synthetic_scene():
    """Dense plane cloud + six known 3-D plane points (off-axis, where distortion matters)."""
    truth = {
        "back_left": _plane_point(-0.33, -0.12), "back_right": _plane_point(0.05, -0.12),
        "front_left": _plane_point(-0.33, 0.10), "front_right": _plane_point(0.05, 0.10),
    }
    truth["left_mid"] = _plane_point(-0.31, -0.01)
    truth["right_mid"] = _plane_point(0.03, -0.01)
    xs, ys = np.meshgrid(np.linspace(-0.36, 0.08, 150), np.linspace(-0.15, 0.13, 100))
    pts = np.array([_plane_point(x, y) for x, y in zip(xs.ravel(), ys.ravel())])
    cloud = np.zeros(pts.shape[0], dtype=[("x", np.float32), ("y", np.float32), ("z", np.float32)])
    cloud["x"], cloud["y"], cloud["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    return truth, cloud


def _project_cv2(points, d):
    import cv2
    pixels, _ = cv2.projectPoints(np.asarray(points, dtype=np.float64).reshape(-1, 1, 3),
                                  np.zeros(3), np.zeros(3), np.asarray(K).reshape(3, 3),
                                  np.asarray(d, dtype=np.float64))
    return pixels.reshape(-1, 2)


# ---------------------------------------------------------------------------
# A. lens distortion


def test_forward_and_inverse_match_opencv_and_camera_model(query):
    import cv2
    from box_seam_detection.core.pbd import camera_model

    _, d8 = query._camera_distortion(_camera_info(D_ORBBEC))
    pts = np.array([[x, y, 0.5] for x in np.linspace(-0.4, 0.4, 9) for y in np.linspace(-0.25, 0.25, 7)])
    xd, yd = query._distort_normalized(pts[:, 0] / pts[:, 2], pts[:, 1] / pts[:, 2], d8)
    ours = np.column_stack((K[0] * xd + K[2], K[4] * yd + K[5]))
    assert np.max(np.abs(ours - _project_cv2(pts, D_ORBBEC))) < 1e-6

    u, v = ours[:, 0], ours[:, 1]
    xn, yn = query._undistort_normalized((u - K[2]) / K[0], (v - K[5]) / K[4], d8)
    assert np.max(np.abs(xn - pts[:, 0] / pts[:, 2])) < 1e-10
    assert np.max(np.abs(yn - pts[:, 1] / pts[:, 2])) < 1e-10
    cv_und = cv2.undistortPoints(ours.reshape(-1, 1, 2), np.asarray(K).reshape(3, 3),
                                 np.asarray(D_ORBBEC)).reshape(-1, 2)
    assert np.max(np.abs(cv_und[:, 0] - xn)) < 1e-6
    model = camera_model.CameraModel("camera_color_optical_frame", 1280, 720, K, d=D_ORBBEC[:5])
    rays = camera_model.pixel_to_ray(u, v, model).xyz
    ours_rays = np.column_stack((xn, yn, np.ones_like(xn)))
    ours_rays /= np.linalg.norm(ours_rays, axis=1, keepdims=True)
    assert np.max(np.abs(rays - ours_rays)) < 1e-12


def test_undistort_round_trip_recovers_plane_points(query):
    truth, cloud = _synthetic_scene()
    pixels = _project_cv2([truth[n] for n in NAMES], D_ORBBEC)
    seam = {n: [float(p[0]), float(p[1])] for n, p in zip(NAMES, pixels)}
    results, meta = query.recover_surface_points(cloud, seam, _camera_info(D_ORBBEC))
    assert meta["distortion"]["applied"] is True
    assert meta["distortion"]["model"] == "plumb_bob"
    assert meta["distortion"]["d"] == D_ORBBEC
    for name in NAMES:
        got = np.array([results[name][c] for c in "xyz"])
        # float32 cloud -> plane fit noise ~1e-8 m; demand 0.05 mm
        assert np.linalg.norm(got - truth[name]) < 5e-5, name
        assert results[name]["local_surface_support"] >= 6
    # Pinhole rays on distorted pixels would be off by far more than that at the corners.
    fx, fy, cx, cy = K[0], K[4], K[2], K[5]
    u, v = seam["back_left"]
    ray = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
    pinhole = ray * (-meta["plane_offset_m"] / float(np.dot(meta["plane_normal"], ray)))
    assert np.linalg.norm(pinhole - truth["back_left"]) > 5e-4


def test_zero_distortion_equals_pinhole_behaviour(query):
    truth, cloud = _synthetic_scene()
    pixels = _project_cv2([truth[n] for n in NAMES], [0.0] * 5)
    seam = {n: [float(p[0]), float(p[1])] for n, p in zip(NAMES, pixels)}
    zero, zero_meta = query.recover_surface_points(cloud, seam, _camera_info([0.0] * 8))
    info_empty = _camera_info([])
    empty, empty_meta = query.recover_surface_points(cloud, seam, info_empty)
    assert zero_meta["distortion"]["applied"] is False
    assert empty_meta["distortion"]["applied"] is False
    fx, fy, cx, cy = K[0], K[4], K[2], K[5]
    normal = np.asarray(zero_meta["plane_normal"])
    for name in NAMES:
        assert zero[name] == empty[name]
        u, v = seam[name]
        ray = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])  # the pre-change formula
        expected = ray * (-zero_meta["plane_offset_m"] / float(normal @ ray))
        assert np.allclose([zero[name][c] for c in "xyz"], expected, atol=1e-15, rtol=0)
    # Cloud projection without distortion is exactly the old pinhole projection.
    xyz, u, v = query._project_valid_cloud(cloud, fx, fy, cx, cy, 1280, 720)
    assert np.array_equal(u, fx * xyz[:, 0] / xyz[:, 2] + cx)
    assert np.array_equal(v, fy * xyz[:, 1] / xyz[:, 2] + cy)


def test_unsupported_distortion_fails_closed(query):
    info = _camera_info(D_ORBBEC)
    info["distortion_model"] = "equidistant"
    with pytest.raises(ValueError, match="distortion_model"):
        query._camera_distortion(info)


# ---------------------------------------------------------------------------
# B. calibration-file camera -> base_link


def _write_calib(path, translation, quaternion, calibration_type="eye_on_base", tracking="camera_link"):
    path.write_text(textwrap.dedent(f"""\
        parameters:
          name: test_handeye
          calibration_type: {calibration_type}
          robot_base_frame: base_link
          robot_effector_frame: tool
          tracking_base_frame: {tracking}
          tracking_marker_frame: marker
        transform:
          translation:
            x: {translation[0]!r}
            y: {translation[1]!r}
            z: {translation[2]!r}
          rotation:
            x: {quaternion[0]!r}
            y: {quaternion[1]!r}
            z: {quaternion[2]!r}
            w: {quaternion[3]!r}
        """), encoding="utf-8")
    return path


def _rot_z(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1.0]])


def test_calib_parsing_and_composition_match_hand_computation(base, tmp_path):
    # base <- camera_link: rotate +90 deg about z, then translate (1, 2, 3).
    s = math.sqrt(0.5)
    calib = _write_calib(tmp_path / "t.calib", (1.0, 2.0, 3.0), (0.0, 0.0, s, s))
    parsed = base.load_handeye_calibration(calib)
    assert parsed["sha256"] == hashlib.sha256(calib.read_bytes()).hexdigest()
    assert parsed["name"] == "test_handeye"
    assert parsed["tracking_base_frame"] == "camera_link"
    hand_calib = np.eye(4)
    hand_calib[:3, :3] = _rot_z(90)
    hand_calib[:3, 3] = (1.0, 2.0, 3.0)
    assert np.allclose(parsed["matrix"], hand_calib, atol=1e-12)

    matrix, block = base.resolve_camera_to_base(
        calib, "recorded", "base_link", "camera_color_optical_frame", 1)
    rec = base.RECORDED_DRIVER_EXTRINSIC
    hand_driver = np.eye(4)
    hand_driver[:3, :3] = base.quaternion_to_rotation_matrix(*rec["quaternion_xyzw"])
    hand_driver[:3, 3] = rec["translation"]
    assert np.allclose(matrix, hand_calib @ hand_driver, atol=1e-12)
    # Optical z (forward) = camera_link x rotated by +90 deg about z -> base +y.
    assert np.allclose(matrix[:3, 2], [0.0, 1.0, 0.0], atol=5e-3)
    assert block["source"]["driver_extrinsic_source"] == "recorded"

    for bad in (dict(calibration_type="eye_in_hand"), dict(tracking="camera_depth_frame")):
        bad_path = _write_calib(tmp_path / "bad.calib", (0, 0, 0), (0, 0, 0, 1.0), **bad)
        with pytest.raises(ValueError):
            base.resolve_camera_to_base(bad_path, "recorded", "base_link", "camera_color_optical_frame", 1)


@pytest.mark.skipif(not REAL_CALIB.is_file(), reason="production calib not on this machine")
def test_real_calib_composition_reproduces_recorded_driver_values(base):
    parsed = base.load_handeye_calibration(REAL_CALIB)
    assert parsed["robot_base_frame"] == "base_link"
    assert np.allclose(parsed["translation"], (-0.052363, -0.306337, 0.452080), atol=1e-6)
    matrix, _ = base.resolve_camera_to_base(
        REAL_CALIB, "recorded", "base_link", "camera_color_optical_frame", 1)
    driver = np.linalg.inv(parsed["matrix"]) @ matrix
    assert np.allclose(driver[:3, 3] * 1000.0, (1.634, -13.983, -0.043), atol=1e-3)


class _FakeTf:
    def __init__(self, table):
        self.table = table
        self.calls = []

    def lookup(self, target, source, stamp_ns, timeout_sec):
        self.calls.append((target, source))
        if (target, source) not in self.table:
            raise RuntimeError(f"no TF {source} -> {target}")
        return self.table[(target, source)]


def test_live_tf_mismatch_warns_and_never_uses_live(base, tmp_path):
    calib = _write_calib(tmp_path / "t.calib", (0.1, -0.3, 0.45), (0.0, 0.0, 0.0, 1.0))
    rec = base.RECORDED_DRIVER_EXTRINSIC
    driver = base.transform_matrix(rec["translation"], rec["quaternion_xyzw"])
    composed_expected = base.load_handeye_calibration(calib)["matrix"] @ driver
    frames = ("base_link", "camera_color_optical_frame")

    # Agreeing live TF: no warning, recorded as compared but not used.
    logs = []
    fake = _FakeTf({("camera_link", frames[1]): driver, frames: composed_expected.copy()})
    matrix, block = base.resolve_camera_to_base(calib, "tf", *frames, 5, tf_provider=fake, log=logs.append)
    assert logs == []
    assert block["source"]["driver_extrinsic_source"] == "tf_live"
    assert block["source"]["live_full_tf_comparison"]["available"] is True
    assert block["source"]["live_full_tf_comparison"]["exceeds_tolerance"] is False

    # Live full TF off by 2 mm: warning, composed transform still used.
    live = composed_expected.copy()
    live[0, 3] += 0.002
    logs = []
    fake = _FakeTf({("camera_link", frames[1]): driver, frames: live})
    matrix, block = base.resolve_camera_to_base(calib, "tf", *frames, 5, tf_provider=fake, log=logs.append)
    assert np.allclose(matrix, composed_expected, atol=1e-15)
    assert any("WARNING" in line and "2.000 mm" in line for line in logs)
    comparison = block["source"]["live_full_tf_comparison"]
    assert comparison["exceeds_tolerance"] is True and comparison["used"] is False

    # Rotation-only mismatch of 0.2 deg also warns.
    live = composed_expected.copy()
    live[:3, :3] = live[:3, :3] @ _rot_z(0.2)
    logs = []
    fake = _FakeTf({("camera_link", frames[1]): driver, frames: live})
    base.resolve_camera_to_base(calib, "tf", *frames, 5, tf_provider=fake, log=logs.append)
    assert any("WARNING" in line for line in logs)

    # No live full TF: fine, marked unavailable. Missing driver TF: fails closed.
    fake = _FakeTf({("camera_link", frames[1]): driver})
    _, block = base.resolve_camera_to_base(calib, "tf", *frames, 5, tf_provider=fake, log=logs.append)
    assert block["source"]["live_full_tf_comparison"]["available"] is False
    with pytest.raises(RuntimeError):
        base.resolve_camera_to_base(calib, "tf", *frames, 5, tf_provider=_FakeTf({}), log=logs.append)


def _camera_geometry(path):
    rng = np.random.default_rng(1)
    results = {}
    for name in NAMES:
        x, y, z = rng.uniform(-0.2, 0.2), rng.uniform(-0.1, 0.1), rng.uniform(0.4, 0.5)
        results[name] = {"x": x, "y": y, "z": z, "valid": True,
                         "depth_source": "measured_carton_plane_ray_intersection"}
    data = {
        "schema": "seam_camera_geometry/v1",
        "capture": {"capture_id": "frame_test", "image_stamp_ns": 1_000_000_000,
                    "pointcloud_stamp_ns": 1_000_000_001, "source_frame": "camera_color_optical_frame"},
        "surface": {"plane_normal": [0.0, -0.7, -0.71], "plane_rms_m": 0.001},
        "results": results,
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    return data


_BLOCK_ROS = (
    "import sys, runpy\n"
    "class _Block:\n"
    "    def find_spec(self, name, path=None, target=None):\n"
    "        if name.split('.')[0] in ('rclpy', 'tf2_ros'):\n"
    "            raise ImportError('blocked for test: ' + name)\n"
    "sys.meta_path.insert(0, _Block())\n"
    "sys.argv = sys.argv[1:]\n"
    "runpy.run_module(sys.argv[0], run_name='__main__', alter_sys=True)\n"
)


def test_offline_mode_without_rclpy_writes_transform_schema(tmp_path):
    camera_path = tmp_path / "camera_geometry.json"
    camera = _camera_geometry(camera_path)
    s = math.sqrt(0.5)
    calib = _write_calib(tmp_path / "t.calib", (0.5, 0.0, 0.2), (0.0, 0.0, s, s))
    out = tmp_path / "base_geometry.json"
    planner = tmp_path / "planner.json"
    proc = subprocess.run(
        [sys.executable, "-c", _BLOCK_ROS, str(BASE_SCRIPT), "--json", str(camera_path),
         "--calib-file", str(calib), "--driver-extrinsic", "recorded",
         "--output-json", str(out), "--planner-output-json", str(planner)],
        capture_output=True, text=True, timeout=60, env=ENV)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(out.read_text())
    assert data["schema"] == "seam_base_geometry/v1"
    assert set(data) == {"schema", "capture", "surface", "results", "source_file", "transform"}
    transform = data["transform"]
    matrix = np.asarray(transform["matrix"])
    assert matrix.shape == (4, 4) and np.allclose(matrix[3], [0, 0, 0, 1])
    source = transform["source"]
    assert source["calib_sha256"] == hashlib.sha256(calib.read_bytes()).hexdigest()
    assert source["calib_file"] == str(calib.resolve())
    assert source["calibration_name"] == "test_handeye"
    assert source["driver_extrinsic_source"] == "recorded"
    assert transform["frames"] == {"parent_frame": "base_link",
                                   "child_frame": "camera_color_optical_frame",
                                   "calibration_tracking_frame": "camera_link"}
    for name in NAMES:
        entry = data["results"][name]
        assert set(entry) == {"valid", "camera_frame", "camera_xyz", "target_frame", "target_xyz"}
        cam = np.array([camera["results"][name][c] for c in "xyz"])
        assert np.allclose(entry["target_xyz"], matrix[:3, :3] @ cam + matrix[:3, 3], atol=1e-12)
    snapshot = json.loads(planner.read_text())
    assert snapshot["schema"] == "box_h_cut_snapshot/v1"
    assert set(snapshot) == {
        "schema", "snapshot_id", "capture_id", "acquisition_stamp_ns", "frame_id", "source_frame",
        "source_file", "seam_start", "seam_end", "seam_center", "back_left", "back_right",
        "front_left", "front_right", "surface_normal", "plane_rms_m"}
    assert np.allclose([snapshot["seam_start"][c] for c in "xyz"], data["results"]["left_mid"]["target_xyz"])


def test_tf_mode_without_rclpy_fails_closed(tmp_path):
    camera_path = tmp_path / "camera_geometry.json"
    _camera_geometry(camera_path)
    calib = _write_calib(tmp_path / "t.calib", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    out = tmp_path / "base_geometry.json"
    proc = subprocess.run(
        [sys.executable, "-c", _BLOCK_ROS, str(BASE_SCRIPT), "--json", str(camera_path),
         "--calib-file", str(calib), "--output-json", str(out)],
        capture_output=True, text=True, timeout=60, env=ENV)
    assert proc.returncode == 1
    assert "base-frame conversion failed" in proc.stderr
    assert not out.exists()


# ---------------------------------------------------------------------------
# C. orchestrator calib preflight (live.orchestrator.preflight_calibration)

ORCHESTRATOR = "box_seam_detection.live.orchestrator"


@pytest.fixture()
def orch():
    return _load(ORCHESTRATOR)


def _preflight_args(calib=None, target="base_link", driver=None):
    import argparse
    return argparse.Namespace(calib_file=None if calib is None else str(calib),
                              target_frame=target, driver_extrinsic=driver)


def test_preflight_accepts_valid_calib(orch, tmp_path):
    calib = _write_calib(tmp_path / "ok.calib", (0.1, -0.3, 0.45), (0.0, 0.0, 0.0, 1.0))
    for driver in (None, "tf", "recorded"):
        parsed = orch.preflight_calibration(_preflight_args(calib, driver=driver))
        assert parsed["robot_base_frame"] == "base_link"
        assert parsed["tracking_base_frame"] == "camera_link"


@pytest.mark.parametrize("make, driver, message", [
    (lambda d: d / "absent.calib", None, "找不到手眼標定檔"),
    (lambda d: _write_calib(d / "h.calib", (0, 0, 0), (0, 0, 0, 1.0),
                            calibration_type="eye_in_hand"), None, "eye_on_base"),
    (lambda d: _write_calib(d / "t.calib", (0, 0, 0), (0, 0, 0, 1.0),
                            tracking="camera_depth_frame"), "recorded", "tracking_base_frame"),
    (lambda d: _write_calib(d / "q.calib", (0, 0, 0), (0, 0, 0.5, 0.5)), None, "quaternion"),
    (lambda d: (d / "junk.calib").write_text("{not: [yaml", encoding="utf-8")
     and d / "junk.calib", None, "cannot parse"),
])
def test_preflight_rejects_bad_calib(orch, tmp_path, make, driver, message):
    calib = make(tmp_path)
    with pytest.raises(ValueError, match=message):
        orch.preflight_calibration(_preflight_args(calib, driver=driver))


def test_preflight_rejects_wrong_target_frame(orch, tmp_path):
    calib = _write_calib(tmp_path / "ok.calib", (0, 0, 0), (0, 0, 0, 1.0))
    with pytest.raises(ValueError, match="robot_base_frame"):
        orch.preflight_calibration(_preflight_args(calib, target="world"))


def test_preflight_default_calib_path_is_the_orchestrator_default(orch, monkeypatch, tmp_path):
    monkeypatch.setattr(orch, "DEFAULT_CALIB_FILE", str(tmp_path / "missing_default.calib"))
    with pytest.raises(ValueError, match="missing_default.calib"):
        orch.preflight_calibration(_preflight_args(None))


def test_preflight_requires_pyyaml(orch, tmp_path, monkeypatch):
    calib = _write_calib(tmp_path / "ok.calib", (0, 0, 0), (0, 0, 0, 1.0))
    monkeypatch.setitem(sys.modules, "yaml", None)     # import yaml -> ImportError
    with pytest.raises(ValueError, match="PyYAML"):
        orch.preflight_calibration(_preflight_args(calib))


def test_orchestrator_bad_calib_exit_2_before_acquisition(orch, tmp_path, monkeypatch, capsys):
    out_root = tmp_path / "rgbd_out"
    sam3 = tmp_path / "fake_python"
    sam3.write_text("", encoding="utf-8")

    class _Probe:
        returncode = 0
        stdout = ""

    monkeypatch.setattr(orch.subprocess, "run", lambda *a, **k: _Probe())
    monkeypatch.setattr(orch, "acquire_live_capture",
                        lambda *a, **k: pytest.fail("must not acquire"))
    code = orch.main(["--calib-file", str(tmp_path / "absent.calib"), "--output-dir", str(out_root),
                      "--autolabel-v2-server", "off", "--autolabel-v2-sam3-python", str(sam3)])
    assert code == 2
    assert "找不到手眼標定檔" in capsys.readouterr().err
    assert not list(out_root.rglob("*.png"))                   # nothing captured


def test_manifest_driver_extrinsic_block_is_used_and_quaternion_sign_is_irrelevant(base, tmp_path):
    """The ROS capture node records the driver extrinsic (same keys as the constant + source/stamp_ns)
    in the manifest: with --driver-extrinsic recorded it replaces the constant; q and -q are the same
    rotation. A block with other frames or a non-unit quaternion fails closed."""
    s = math.sqrt(0.5)
    calib = _write_calib(tmp_path / "t.calib", (0.5, 0.0, 0.2), (0.0, 0.0, s, s))
    rec = base.RECORDED_DRIVER_EXTRINSIC
    block = {"parent_frame": "camera_link", "child_frame": "camera_color_optical_frame",
             "translation": [0.002, -0.0139, 0.0001],
             "quaternion_xyzw": [-v for v in rec["quaternion_xyzw"]],     # opposite sign, same rotation
             "source": "tf_live", "stamp_ns": 123}
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"driver_extrinsic": block}), encoding="utf-8")
    got = base.driver_extrinsic_from_manifest(manifest, "camera_link", "camera_color_optical_frame")
    assert got["source"] == "tf_live"
    matrix, out = base.resolve_camera_to_base(
        calib, "recorded", "base_link", "camera_color_optical_frame", 1, driver_block=got)
    constant = base.transform_matrix(rec["translation"], rec["quaternion_xyzw"])
    expected = base.load_handeye_calibration(calib)["matrix"] @ base.transform_matrix(
        block["translation"], block["quaternion_xyzw"])
    assert np.allclose(matrix, expected, atol=1e-12)
    assert np.allclose(base.transform_matrix([0, 0, 0], block["quaternion_xyzw"])[:3, :3],
                       constant[:3, :3], atol=1e-12)                      # same rotation as the constant
    assert out["source"]["driver_extrinsic_origin"] == "capture_manifest"
    # no block (today's captures) -> None -> the constant
    manifest.write_text(json.dumps({"schema": "rgbd_capture/v1"}), encoding="utf-8")
    assert base.driver_extrinsic_from_manifest(manifest, "camera_link", "camera_color_optical_frame") is None
    matrix_c, out_c = base.resolve_camera_to_base(calib, "recorded", "base_link",
                                                  "camera_color_optical_frame", 1)
    assert "driver_extrinsic_origin" not in out_c["source"]
    # fail closed
    for bad in ({**block, "parent_frame": "other"}, {**block, "quaternion_xyzw": [0, 0, 0, 2.0]},
                {**block, "translation": [float("nan"), 0, 0]}):
        manifest.write_text(json.dumps({"driver_extrinsic": bad}), encoding="utf-8")
        with pytest.raises(ValueError):
            base.driver_extrinsic_from_manifest(manifest, "camera_link", "camera_color_optical_frame")
