"""YAML writer and rpy conversions."""
import numpy as np
import yaml

from rebel_dual_arm_calibration import solver
from rebel_dual_arm_calibration.solve_cli import write_easy_handeye_calibs
from rebel_dual_arm_calibration.transforms import (R_to_rpy, inv_T, rpy_to_R, xyz_rpy_to_T, T_to_xyz_rpy,
                                                   rotation_angle_deg)


class FakeResult:
    def __init__(self, T_right, T_cam):
        self.T_world_arm_right_base = T_right
        self.T_world_camera_link = T_cam
        self.report = {"date": "2026-10-06"}


def test_rpy_round_trip_and_urdf_convention():
    rng = np.random.default_rng(0)
    for _ in range(200):
        rpy = rng.uniform([-np.pi, -1.5, -np.pi], [np.pi, 1.5, np.pi])
        R = rpy_to_R(rpy)
        assert np.allclose(R_to_rpy(R), rpy, atol=1e-9)
    # URDF: R = Rz(yaw) Ry(pitch) Rx(roll)
    r, p, y = 0.3, -0.4, 1.1
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    assert np.allclose(rpy_to_R([r, p, y]), Rz @ Ry @ Rx)


def test_yaml_structure_and_round_trip(tmp_path):
    T_right = xyz_rpy_to_T([0.4504, 0.0021, -0.0013], [0.0012, -0.0007, np.radians(0.5)])
    T_cam = xyz_rpy_to_T([0.2425, 0.9286, 0.8473], [0.01, 0.8639, -1.5675])
    path = tmp_path / "cal.yaml"
    solver.write_calibration_yaml(str(path), FakeResult(T_right, T_cam),
                                  'notes with "quotes": 38 samples, rms 0.4 px, dataset /tmp/x')
    data = yaml.safe_load(path.read_text())
    assert list(data) == ["dual_arm_calibration"]
    c = data["dual_arm_calibration"]
    assert set(c) == {"source", "date", "world_frame", "arm_right_base", "camera_link", "notes"}
    assert c["source"] == "calibrated" and c["world_frame"] == "world"
    assert c["date"] == "2026-10-06" and isinstance(c["date"], str)
    assert 'quotes' in c["notes"] and "dataset /tmp/x" in c["notes"]
    for key, T in (("arm_right_base", T_right), ("camera_link", T_cam)):
        assert set(c[key]) == {"xyz", "rpy"} and len(c[key]["xyz"]) == 3 and len(c[key]["rpy"]) == 3
        T2 = xyz_rpy_to_T(c[key]["xyz"], c[key]["rpy"])
        assert np.allclose(T2, T, atol=1e-12)


def test_yaml_dict_matches_file(tmp_path):
    T = xyz_rpy_to_T([1, 2, 3], [0.1, 0.2, 0.3])
    res = FakeResult(T, T)
    path = tmp_path / "c.yaml"
    solver.write_calibration_yaml(str(path), res, "n")
    assert yaml.safe_load(path.read_text()) == solver.calibration_yaml_dict(res, "n")


def test_easy_handeye_calibs(tmp_path):
    T_right = xyz_rpy_to_T([0.45, 0.0, 0.0], [0, 0, 0.01])
    T_cam = xyz_rpy_to_T([0.2, 0.9, 0.8], [0.0, 0.9, -1.5])
    paths = write_easy_handeye_calibs(str(tmp_path), FakeResult(T_right, T_cam), "x")
    left = yaml.safe_load(open([p for p in paths if "left" in p][0]))
    right = yaml.safe_load(open([p for p in paths if "right" in p][0]))
    assert left["parameters"]["calibration_type"] == "eye_on_base"
    assert left["transform"]["translation"]["y"] == 0.9
    assert abs(right["transform"]["translation"]["x"] - (0.2 - 0.45)) < 0.01
