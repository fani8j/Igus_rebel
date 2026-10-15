"""Session round trip and the `solve` command on a synthetic session."""
import json

import numpy as np
import yaml

from rebel_dual_arm_calibration import session, solve_cli
from rebel_dual_arm_calibration.synthetic import make_scene, simulate_samples
from rebel_dual_arm_calibration.targets import TargetSpec
from rebel_dual_arm_calibration.transforms import rotation_angle_deg, xyz_rpy_to_T


def test_solve_cli_end_to_end(tmp_path):
    specs = {"left": TargetSpec("left", "charuco", "DICT_4X4_50", 0.018, list(range(8)), 0.025, 4, 4),
             "right": TargetSpec("right", "charuco", "DICT_4X4_50", 0.018, list(range(20, 28)), 0.025, 4, 4)}
    scene = make_scene(distance=0.9, seed=1)
    samples = simulate_samples(scene, specs, n_per_arm=14, noise_px=0.3, seed=2, n_corner_outliers=1)
    sdir = str(tmp_path / "sess")
    session.init_session(sdir, {"note": "synthetic"})
    for s in samples:
        session.append_sample(sdir, s, {"capture": 1})
    loaded, meta = session.load_session(sdir)
    assert meta["note"] == "synthetic" and len(loaded) == len(samples)
    assert np.allclose(loaded[0].pix, samples[0].pix) and loaded[0].arm == samples[0].arm

    out = str(tmp_path / "cal.yaml")
    calib_dir = str(tmp_path / "calibs")
    rc = solve_cli.main([sdir, "--out", out, "--easy-handeye-dir", calib_dir])
    assert rc == 0
    d = yaml.safe_load(open(out))["dual_arm_calibration"]
    # no touches / no common points: the trust gate cannot pass, so the base is flagged as unverified
    assert d["source"] == "calibrated_unverified" and "TRUST GATE FAIL" in d["notes"] and "samples used" in d["notes"] and sdir in d["notes"]
    T_R = xyz_rpy_to_T(d["arm_right_base"]["xyz"], d["arm_right_base"]["rpy"])
    assert np.linalg.norm(T_R[:3, 3] - scene.T_W_R[:3, 3]) < 1e-3
    assert rotation_angle_deg(T_R[:3, :3], scene.T_W_R[:3, :3]) < 0.15
    T_C = xyz_rpy_to_T(d["camera_link"]["xyz"], d["camera_link"]["rpy"])
    assert np.linalg.norm(T_C[:3, 3] - scene.T_W_cam[:3, 3]) < 1.5e-3
    rep = json.load(open(str(tmp_path / "cal_report.json")))
    assert {"joint", "separate", "warnings", "samples"} <= set(rep)
    assert (tmp_path / "calibs" / "dual_arm_left.calib").exists()


def test_solve_cli_reports_missing_arm(tmp_path, capsys):
    specs = {"left": TargetSpec("left", "charuco", "DICT_4X4_50", 0.018, list(range(8)), 0.025, 4, 4),
             "right": TargetSpec("right", "charuco", "DICT_4X4_50", 0.018, list(range(20, 28)), 0.025, 4, 4)}
    scene = make_scene(distance=0.9, seed=1)
    samples = [s for s in simulate_samples(scene, specs, n_per_arm=6, seed=2) if s.arm == "left"]
    sdir = str(tmp_path / "s")
    session.init_session(sdir, {})
    for s in samples:
        session.append_sample(sdir, s, {})
    assert solve_cli.main([sdir]) == 2
    assert "right" in capsys.readouterr().err
