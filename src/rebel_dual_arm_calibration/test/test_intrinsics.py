"""Intrinsics: file format, override of camera_info, calibration tool on synthetic (rendered-geometry) views."""
import json
import os

import numpy as np
import pytest
import yaml

from rebel_dual_arm_calibration import intrinsics as I, session, solve_cli
from rebel_dual_arm_calibration.solver import project_points
from rebel_dual_arm_calibration.targets import default_targets_path, load_targets_all
from rebel_dual_arm_calibration.transforms import rotvec_to_R

SIZE = (1920, 1080)
K_TRUE = np.array([[1003.0, 0, 951.5], [0, 1000.0, 548.0], [0, 0, 1.0]])
D_TRUE = np.array([-0.052, 0.021, 0.0006, -0.0004, -0.004])


def make_views(n=30, noise=0.15, seed=0, K=K_TRUE, D=D_TRUE, K_info=None):
    """Views of the table board (the same points the recorder detects) at varied poses, projected through the true model."""
    _, table = load_targets_all(default_targets_path())
    skip = table.touch_corner_ids()
    pts = {k: v for k, v in table.all_points_obj().items() if k not in skip}
    ids = np.array(list(pts))
    obj = np.array(list(pts.values()))
    rng = np.random.default_rng(seed)
    views, tries = [], 0
    K_info = K if K_info is None else K_info
    while len(views) < n and tries < 20000:
        tries += 1
        z = rng.uniform(0.45, 1.3)
        c = np.array([rng.uniform(-0.55, 0.55) * z, rng.uniform(-0.3, 0.3) * z, z])
        tilt = rng.normal(0, 0.45, 3)
        R = rotvec_to_R(np.array([np.pi, 0, 0]) * 0 + tilt) @ rotvec_to_R([0, 0, rng.uniform(0, np.pi)])
        P = obj @ R.T + c
        if np.any(P[:, 2] < 0.2):
            continue
        uv = project_points(K, D, P)
        if uv[:, 0].min() < 5 or uv[:, 1].min() < 5 or uv[:, 0].max() > SIZE[0] - 5 or uv[:, 1].max() > SIZE[1] - 5:
            continue
        views.append({"image": f"intrinsics/intr_{len(views) + 1:04d}.png", "ids": ids.tolist(), "obj": obj.tolist(),
                      "pix": (uv + rng.normal(0, noise, uv.shape)).tolist(), "K": K_info.tolist(), "dist": [0.0] * 5,
                      "image_size": list(SIZE), "board_distance_m": float(np.linalg.norm(c))})
    assert len(views) == n
    return views


def test_recovers_known_intrinsics_from_views():
    r = I.calibrate_views(make_views(30, 0.15), drop_outlier_views=False)
    K = np.array(r["K"])
    assert r["ok"] and r["rms_px"] < 0.3 and r["n_views"] == 30
    assert abs(K[0, 0] / K_TRUE[0, 0] - 1) < 0.003 and abs(K[1, 1] / K_TRUE[1, 1] - 1) < 0.003
    # the principal point of a small planar board is the weakest parameter (std 1.5-2 px with 30 views): 3 sigma
    assert abs(K[0, 2] - K_TRUE[0, 2]) < 3 * r["std_px"]["cx"] and abs(K[1, 2] - K_TRUE[1, 2]) < 3 * r["std_px"]["cy"]
    assert r["std_px"]["cx"] < 3.0
    assert abs(r["dist"][0] - D_TRUE[0]) < 0.03       # k1 (k1, k2, k3 and the principal point are correlated)
    assert r["coverage"]["n_corners"] > 1000 and r["board_distance_m"]["max"] > 1.0 > r["board_distance_m"]["min"] - 0.7


def test_reports_difference_to_camera_info():
    K_info = K_TRUE.copy()
    K_info[0, 0] *= 1.012
    K_info[1, 1] *= 1.012
    K_info[0, 2] += 6.0
    r = I.calibrate_views(make_views(30, 0.15, K_info=K_info), drop_outlier_views=False)
    d = r["difference_to_camera_info"]
    assert d["df_x_pct"] == pytest.approx(-1.2, abs=0.35) and d["dc_px"][0] == pytest.approx(-6.0, abs=3 * r["std_px"]["cx"])
    text = "\n".join(I.format_calibration(r))
    assert "difference to camera_info" in text and "coverage" in text and "RMS" in text


def test_requirements_views_and_rms():
    few = I.calibrate_views(make_views(18, 0.15, seed=2), drop_outlier_views=False)
    assert not few["ok"] and any("views" in p for p in few["problems"])
    noisy = I.calibrate_views(make_views(30, 1.2, seed=3), drop_outlier_views=False)
    assert noisy["rms_px"] > 0.5 and not noisy["ok"] and any("RMS" in p for p in noisy["problems"])
    # a badly-covered set (only the centre of the image) is called out
    views = [v for v in make_views(60, 0.15, seed=4) if np.all(np.abs(np.array(v["pix"]) - [960, 540]) < [500, 300])][:26]
    if len(views) >= 6:
        r = I.calibrate_views(views, min_views=len(views), drop_outlier_views=False)
        assert r["coverage"]["warnings"]


def test_cli_writes_a_file_readable_by_solve_and_refuses_bad_data(tmp_path, capsys):
    d = str(tmp_path / "sess")
    session.init_session(d, {})
    for v in make_views(28, 0.15, seed=5):
        session.append_intrinsics_view(d, v)
    out = str(tmp_path / "intr.yaml")
    assert I.main([d, "--out", out]) == 0
    K, dist, info = I.load_intrinsics_file(out)
    assert K.shape == (3, 3) and abs(K[0, 0] / K_TRUE[0, 0] - 1) < 0.003 and len(dist) == 5
    assert info["image_size"] == SIZE and info["rms_px"] < 0.5 and info["requirements_met"]
    assert os.path.exists(str(tmp_path / "intr_report.json"))
    assert "difference to camera_info" in capsys.readouterr().out
    # too few views: not written (exit 2), --force writes it anyway
    d2 = str(tmp_path / "few")
    session.init_session(d2, {})
    for v in make_views(10, 0.15, seed=6):
        session.append_intrinsics_view(d2, v)
    out2 = str(tmp_path / "few.yaml")
    assert I.main([d2, "--out", out2]) == 2 and not os.path.exists(out2)
    assert I.main([d2, "--out", out2, "--force", "--min-views", "5"]) == 0 and os.path.exists(out2)
    assert I.main([str(tmp_path / "nothing"), "--out", out2]) == 2


def test_file_formats_and_override(tmp_path):
    p = str(tmp_path / "cam.yaml")
    I.write_intrinsics_file(p, K_TRUE, D_TRUE, SIZE, {"rms_px": 0.2})
    K, D, info = I.load_intrinsics_file(p)
    assert np.allclose(K, K_TRUE) and np.allclose(D, D_TRUE) and info["image_size"] == SIZE
    # sensor_msgs/CameraInfo-like YAML
    q = str(tmp_path / "info.yaml")
    yaml.safe_dump({"width": 1920, "height": 1080, "k": K_TRUE.reshape(-1).tolist(), "d": D_TRUE.tolist()}, open(q, "w"))
    K2, D2, info2 = I.load_intrinsics_file(q)
    assert np.allclose(K2, K_TRUE) and np.allclose(D2, D_TRUE) and info2["image_size"] == SIZE
    with pytest.raises(ValueError):
        yaml.safe_dump({"foo": 1}, open(str(tmp_path / "bad.yaml"), "w"))
        I.load_intrinsics_file(str(tmp_path / "bad.yaml"))

    class Item:
        K = np.eye(3)
        dist = np.zeros(5)
        image_size = SIZE
    items = [Item(), Item()]
    assert I.apply_intrinsics(items, K, D, SIZE) == 2 and np.allclose(items[0].K, K_TRUE) and np.allclose(items[1].dist, D_TRUE)
    with pytest.raises(ValueError, match="image size"):
        I.apply_intrinsics(items, K, D, (1280, 720))


def test_solve_cli_with_intrinsics_file_overrides_camera_info_and_passes_the_gate_check(tmp_path, capsys):
    from rebel_dual_arm_calibration.synthetic import make_scene, simulate_samples
    from rebel_dual_arm_calibration.targets import TargetSpec
    specs = {"left": TargetSpec("left", "charuco", "DICT_4X4_50", 0.018, list(range(8)), 0.025, 4, 4),
             "right": TargetSpec("right", "charuco", "DICT_4X4_50", 0.018, list(range(20, 28)), 0.025, 4, 4)}
    scene = make_scene(distance=0.9, seed=1)
    samples = simulate_samples(scene, specs, n_per_arm=14, noise_px=0.3, seed=2)
    sdir = str(tmp_path / "s")
    session.init_session(sdir, {})
    wrong = np.array(scene.K).copy()
    for s in samples:                           # camera_info 2 % off in f: recorded as is
        s.K = wrong * np.array([[1.02, 1, 1], [1, 1.02, 1], [1, 1, 1]])
        session.append_sample(sdir, s, {})
    good = str(tmp_path / "good.yaml")
    I.write_intrinsics_file(good, scene.K, scene.dist, scene.size)
    out_a, out_b = str(tmp_path / "a.yaml"), str(tmp_path / "b.yaml")
    assert solve_cli.main([sdir, "--out", out_a, "--no-validate"]) == 0
    a = json.load(open(str(tmp_path / "a_report.json")))
    assert a["options"]["intrinsics_source"] == "camera_info" and a["joint"]["intrinsics"]["pull_f_frac"] > 0.003
    ga = {c["name"]: c for c in a["trust_gate"]["checks"]}["intrinsics_validated"]
    assert not ga["ok"] and "calibrate the camera intrinsics" in ga["text"]
    assert solve_cli.main([sdir, "--out", out_b, "--no-validate", "--intrinsics-file", good]) == 0
    b = json.load(open(str(tmp_path / "b_report.json")))
    assert b["options"]["intrinsics_source"] == "file" and b["joint"]["intrinsics"]["source"] == "file"
    gb = {c["name"]: c for c in b["trust_gate"]["checks"]}["intrinsics_validated"]
    assert gb["ok"] and "--intrinsics-file" in gb["text"]
    assert b["joint"]["rms_px"]["all"] < a["joint"]["rms_px"]["all"]            # the right intrinsics fit better
    assert "override camera_info" in capsys.readouterr().out
    # image size mismatch is refused
    bad = str(tmp_path / "bad.yaml")
    I.write_intrinsics_file(bad, scene.K, scene.dist, (1280, 720))
    assert solve_cli.main([sdir, "--out", out_b, "--no-validate", "--intrinsics-file", bad]) == 2


def test_dropped_views_are_recalibrated_and_never_paired_with_wrong_poses():
    """Regression: after dropping outlier views the camera is recalibrated on the kept views only, and every
    per-view entry belongs to a kept view (per-view poses were once paired with a stale, larger view set)."""
    views = make_views(32, 0.15, seed=5)
    rng = np.random.default_rng(9)
    for i in (3, 17):  # two grossly corrupted views
        pix = np.array(views[i]["pix"])
        views[i]["pix"] = (pix + rng.normal(0, 6.0, pix.shape)).tolist()
    r = I.calibrate_views(views, drop_outlier_views=True)
    assert r["n_views_dropped"] >= 2
    dropped = set(r["dropped_views"])
    assert {views[3]["image"], views[17]["image"]} <= dropped
    names = [v["image"] for v in r["per_view"]]
    assert len(names) == r["n_views"] == len(views) - r["n_views_dropped"]
    assert not dropped & set(names)
    assert max(v["rms_px"] for v in r["per_view"]) < 1.0
    assert abs(r["K"][0][0] - K_TRUE[0, 0]) / K_TRUE[0, 0] < 0.003
