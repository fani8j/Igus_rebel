"""solve over several session directories: merge, camera-consistency refusal, label handling."""
import os

import numpy as np
import pytest
import yaml

from rebel_dual_arm_calibration import session, solve_cli
from rebel_dual_arm_calibration.session import SessionMismatchError, check_session_consistency, load_sessions
from rebel_dual_arm_calibration.solver import Sample
from rebel_dual_arm_calibration.synthetic import make_scene, simulate_samples
from rebel_dual_arm_calibration.targets import TargetSpec
from rebel_dual_arm_calibration.transforms import rotation_angle_deg, xyz_rpy_to_T

SPECS = {"left": TargetSpec("left", "charuco", "DICT_4X4_50", 0.018, list(range(8)), 0.025, 4, 4),
         "right": TargetSpec("right", "charuco", "DICT_4X4_50", 0.018, list(range(20, 28)), 0.025, 4, 4)}


def _copy(s: Sample) -> Sample:
    return Sample.from_json_dict(s.to_json_dict())


def _write(path, samples, meta=None):
    session.init_session(str(path), meta or {"note": "synthetic"})
    for s in samples:
        session.append_sample(str(path), s, {})
    return str(path)


@pytest.fixture(scope="module")
def scene_samples():
    scene = make_scene(distance=0.9, seed=1)
    samples = simulate_samples(scene, SPECS, n_per_arm=14, noise_px=0.3, seed=2, n_corner_outliers=1)
    return scene, samples


def test_merge_of_per_arm_sessions_recovers_ground_truth(tmp_path, scene_samples, capsys):
    scene, samples = scene_samples
    left = _write(tmp_path / "rec_left", [s for s in samples if s.arm == "left"])
    right = _write(tmp_path / "rec_right", [s for s in samples if s.arm == "right"])
    out = str(tmp_path / "cal.yaml")
    assert solve_cli.main([left, right, "--out", out]) == 0
    printed = capsys.readouterr().out
    assert "Session " + left in printed and "Session " + right in printed and "Loaded 28 samples in total" in printed
    d = yaml.safe_load(open(out))["dual_arm_calibration"]
    assert left in d["notes"] and right in d["notes"]
    T_R = xyz_rpy_to_T(d["arm_right_base"]["xyz"], d["arm_right_base"]["rpy"])
    assert np.linalg.norm(T_R[:3, 3] - scene.T_W_R[:3, 3]) < 1e-3
    assert rotation_angle_deg(T_R[:3, :3], scene.T_W_R[:3, :3]) < 0.15
    T_C = xyz_rpy_to_T(d["camera_link"]["xyz"], d["camera_link"]["rpy"])
    assert np.linalg.norm(T_C[:3, 3] - scene.T_W_cam[:3, 3]) < 1.5e-3


def test_merge_matches_single_session_result(tmp_path, scene_samples):
    scene, samples = scene_samples
    both = _write(tmp_path / "both", samples)
    left = _write(tmp_path / "l", [s for s in samples if s.arm == "left"])
    right = _write(tmp_path / "r", [s for s in samples if s.arm == "right"])
    a, b = str(tmp_path / "a.yaml"), str(tmp_path / "b.yaml")
    assert solve_cli.main([both, "--out", a]) == 0
    assert solve_cli.main([left, right, "--out", b]) == 0
    da, db = (yaml.safe_load(open(p))["dual_arm_calibration"] for p in (a, b))
    for key in ("arm_right_base", "camera_link"):
        assert np.allclose(da[key]["xyz"], db[key]["xyz"], atol=1e-6)
        assert np.allclose(da[key]["rpy"], db[key]["rpy"], atol=1e-6)


def test_same_arm_in_two_sessions_is_merged(tmp_path, scene_samples):
    _, samples = scene_samples
    ls = [s for s in samples if s.arm == "left"]
    rs = [s for s in samples if s.arm == "right"]
    d1 = _write(tmp_path / "s1", ls[:7] + rs[:7])
    d2 = _write(tmp_path / "s2", ls[7:] + rs[7:])
    merged, metas, per = load_sessions([d1, d2])
    assert len(merged) == 28 and len(metas) == 2 and [l for l, _ in per] == ["s1", "s2"]
    assert merged[0].image.startswith("s1/") and merged[-1].image.startswith("s2/")


def _mismatch_cases(base: Sample):
    def with_(**kw):
        s = _copy(base)
        for k, v in kw.items():
            setattr(s, k, v)
        return s
    K = base.K.copy()
    K[0, 0] += 5.0
    K2 = base.K.copy()
    K2[0, 2] += 3.0
    T = base.T_camlink_optical.copy()
    T[0, 3] += 0.005
    R = xyz_rpy_to_T([0, 0, 0], [0, 0, np.radians(1.0)])
    T_rot = base.T_camlink_optical @ R
    return {
        "fx": (with_(K=K), "camera matrix"),
        "cx": (with_(K=K2), "camera matrix"),
        "dist": (with_(dist=base.dist + 0.05), "distortion"),
        "size": (with_(image_size=(1280, 720)), "image size"),
        "T_camlink_translation": (with_(T_camlink_optical=T), "camera_link->optical"),
        "T_camlink_rotation": (with_(T_camlink_optical=T_rot), "camera_link->optical"),
    }


@pytest.mark.parametrize("case", ["fx", "cx", "dist", "size", "T_camlink_translation", "T_camlink_rotation"])
def test_mismatching_camera_is_refused(tmp_path, scene_samples, capsys, case):
    _, samples = scene_samples
    ls = [s for s in samples if s.arm == "left"]
    rs = [s for s in samples if s.arm == "right"]
    bad, phrase = _mismatch_cases(rs[0])[case]
    right = [bad] + [_copy(s) for s in rs[1:]]
    left_dir = _write(tmp_path / "left_session", ls)
    right_dir = _write(tmp_path / "right_session", right)
    out = str(tmp_path / "cal.yaml")
    assert solve_cli.main([left_dir, right_dir, "--out", out]) == 2
    err = capsys.readouterr().err
    assert "right_session" in err and "left_session" in err and phrase in err and "does not match" in err
    assert not os.path.exists(out)


def test_negligible_difference_is_accepted(tmp_path, scene_samples):
    _, samples = scene_samples
    ls = [s for s in samples if s.arm == "left"]
    rs = [_copy(s) for s in samples if s.arm == "right"]
    for s in rs:
        s.K = s.K + 0.1 * np.array([[1, 0, 1], [0, 1, 1], [0, 0, 0]])
        s.T_camlink_optical = s.T_camlink_optical.copy()
        s.T_camlink_optical[2, 3] += 2e-4
    check_session_consistency([("a", ls), ("b", rs)])


def test_single_dir_and_cli_edge_cases(tmp_path, scene_samples, capsys):
    _, samples = scene_samples
    ls = [s for s in samples if s.arm == "left"]
    rs = [s for s in samples if s.arm == "right"]
    left = _write(tmp_path / "l", ls)
    right = _write(tmp_path / "r", rs)
    # several sessions need --out
    assert solve_cli.main([left, right]) == 2
    assert "--out is required" in capsys.readouterr().err
    # the same directory twice
    assert solve_cli.main([left, left, "--out", str(tmp_path / "x.yaml")]) == 2
    assert "more than once" in capsys.readouterr().err
    # not a session directory
    assert solve_cli.main([left, str(tmp_path / "nothing"), "--out", str(tmp_path / "x.yaml")]) == 2
    assert "no samples.jsonl" in capsys.readouterr().err
    # merged data still lacking an arm is reported by the solver
    left2 = _write(tmp_path / "l2", ls[:6])
    assert solve_cli.main([left, left2, "--out", str(tmp_path / "x.yaml")]) == 2
    assert "right" in capsys.readouterr().err


def test_equal_basenames_get_unique_labels(tmp_path, scene_samples):
    _, samples = scene_samples
    a = tmp_path / "a" / "session"
    b = tmp_path / "b" / "session"
    os.makedirs(tmp_path / "a")
    os.makedirs(tmp_path / "b")
    _write(a, [s for s in samples if s.arm == "left"])
    _write(b, [s for s in samples if s.arm == "right"])
    _, _, per = load_sessions([str(a), str(b)])
    assert [l for l, _ in per] == ["session#1", "session#2"]
