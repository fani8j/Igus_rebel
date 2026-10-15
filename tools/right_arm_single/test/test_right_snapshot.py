import copy
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import right_snapshot as rs  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
CAPTURES = [ROOT / "scan_results/left_from_corner_20261014/snapshot" / n
            for n in ("capture_1720.json", "capture_1549.json")]
CALIB = rs.DEFAULT_CALIBRATION


def vec(node):
    return np.array([node["x"], node["y"], node["z"]], dtype=float)


def load(path):
    return json.loads(Path(path).read_text())


@pytest.fixture
def calibration():
    return rs.load_calibration(CALIB)


def reference_pose(xyz, rpy):
    """Independent reference: scipy extrinsic xyz rotation == URDF fixed-axis rpy."""
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    pose[:3, 3] = xyz
    return pose


def fresh_copy(tmp_path, capture, name="src.json", age_s=1.0):
    dest = tmp_path / name
    doc = json.loads(Path(capture).read_text())
    doc["acquisition_stamp_ns"] = int((time.time() - age_s) * 1e9)  # the executor checks the stamp too
    dest.write_text(json.dumps(doc))
    t = time.time() - age_s
    os.utime(dest, (t, t))
    return dest


def test_pose_matches_scipy(calibration):
    for key in ("arm_right_base", "camera_link"):
        c = calibration[key]
        assert np.allclose(rs.pose_from_xyz_rpy(c["xyz"], c["rpy"]), reference_pose(c["xyz"], c["rpy"]), atol=1e-14)


@pytest.mark.parametrize("capture", CAPTURES)
def test_round_trip_against_numpy_reference(tmp_path, calibration, capture):
    src = fresh_copy(tmp_path, capture)
    out = tmp_path / "out.json"
    rs.convert_file(src, out, CALIB)
    before, after = load(src), load(out)
    base = calibration["arm_right_base"]
    b = reference_pose(base["xyz"], base["rpy"])
    b_inv = np.linalg.inv(b)
    mapping = {"back_left": "front_right", "front_right": "back_left", "back_right": "front_left",
               "front_left": "back_right", "seam_start": "seam_end", "seam_end": "seam_start",
               "back_middle_left": "front_middle_right", "front_middle_right": "back_middle_left",
               "front_middle_left": "back_middle_right", "back_middle_right": "front_middle_left",
               "seam_center": "seam_center"}
    for name, source in mapping.items():
        expected = (b_inv @ np.append(vec(before[source]), 1.0))[:3]
        assert np.allclose(vec(after[name]), expected, atol=1e-12), name
    expected_n = b_inv[:3, :3] @ vec(before["surface_normal"])
    assert np.allclose(vec(after["surface_normal"]), expected_n, atol=1e-12)
    assert abs(np.linalg.norm(vec(after["surface_normal"])) - np.linalg.norm(vec(before["surface_normal"]))) < 1e-12
    # back to the world frame: B * p must reproduce the original labelled points
    for name, source in mapping.items():
        world = (b @ np.append(vec(after[name]), 1.0))[:3]
        assert np.allclose(world, vec(before[source]), atol=1e-12)


@pytest.mark.parametrize("capture", CAPTURES)
def test_relabel_is_180_deg_rotation_about_normal_and_keeps_midpoints(tmp_path, capture):
    """New labels == old corners rotated by 180 deg about the carton centre/normal, and the
    canonical midpoint formulas still hold (header comment of snapshot_frame.hpp)."""
    src = fresh_copy(tmp_path, capture)
    out = tmp_path / "out.json"
    cal = rs.load_calibration(CALIB)
    rs.convert_file(src, out, CALIB)
    after = load(out)
    p = {n: vec(after[n]) for n in rs.POINT_NAMES}
    # corner order after relabel: back_left, back_right, front_left, front_right still a rectangle
    assert np.allclose(p["back_left"] + p["front_right"], p["back_right"] + p["front_left"], atol=2e-2)

    def resid(value, a, b):
        return np.abs(value - 0.5 * (a + b)).max()

    assert resid(p["back_middle_left"], p["back_left"], p["seam_start"]) < 1e-9
    assert resid(p["front_middle_left"], p["front_left"], p["seam_start"]) < 1e-9
    assert resid(p["back_middle_right"], p["back_right"], p["seam_end"]) < 1e-9
    assert resid(p["front_middle_right"], p["front_right"], p["seam_end"]) < 1e-9
    # geometric rotate-180: swapped corner pairs are point-reflected about the seam centre line
    before = load(src)
    b_inv = np.linalg.inv(reference_pose(cal["arm_right_base"]["xyz"], cal["arm_right_base"]["rpy"]))
    old = {n: (b_inv @ np.append(vec(before[n]), 1.0))[:3] for n in rs.POINT_NAMES}
    n = b_inv[:3, :3] @ vec(before["surface_normal"])
    n = n / np.linalg.norm(n)
    centre = 0.25 * (old["back_left"] + old["back_right"] + old["front_left"] + old["front_right"])
    rot = Rotation.from_rotvec(math.pi * n).as_matrix()
    for name, src_name in (("back_left", "front_right"), ("back_right", "front_left"),
                           ("front_left", "back_right"), ("front_right", "back_left")):
        # old[name] rotated by pi about the normal through the centre lands at p[src_name's old position]
        assert np.allclose(rot @ (old[name] - centre) + centre, old[src_name], atol=2e-2), name


def test_relabel_matches_header_mapping_on_labels():
    pts = {n: np.array([i, 10 * i, 100 * i], dtype=float) for i, n in enumerate(rs.POINT_NAMES)}
    pts[rs.NORMAL_NAME] = np.array([0.0, 0.0, 1.0])
    out = rs.relabel_rotate_180(pts)
    # from snapshot_frame.hpp relabelRotate180, copied by hand
    assert np.array_equal(out["back_left"], pts["front_right"])
    assert np.array_equal(out["front_right"], pts["back_left"])
    assert np.array_equal(out["back_right"], pts["front_left"])
    assert np.array_equal(out["front_left"], pts["back_right"])
    assert np.array_equal(out["seam_start"], pts["seam_end"])
    assert np.array_equal(out["seam_end"], pts["seam_start"])
    assert np.array_equal(out["back_middle_left"], pts["front_middle_right"])
    assert np.array_equal(out["front_middle_right"], pts["back_middle_left"])
    assert np.array_equal(out["front_middle_left"], pts["back_middle_right"])
    assert np.array_equal(out["back_middle_right"], pts["front_middle_left"])
    assert np.array_equal(out["seam_center"], pts["seam_center"])
    assert np.array_equal(out["surface_normal"], pts["surface_normal"])


def mirror(points):
    """snapshot_frame.hpp relabelMirrorLeftRight (invalid relabel)."""
    out = dict(points)
    out["back_left"], out["back_right"] = points["back_right"], points["back_left"]
    out["front_left"], out["front_right"] = points["front_right"], points["front_left"]
    return out


def test_handedness_sign_and_mirror_detected():
    doc = load(CAPTURES[0])
    pts = {n: vec(doc[n]) for n in rs.POINT_NAMES}
    pts[rs.NORMAL_NAME] = vec(doc["surface_normal"])
    assert rs.label_handedness(pts) != 0
    assert rs.same_handedness(pts, rs.relabel_rotate_180(pts))
    assert not rs.same_handedness(pts, mirror(pts))
    flat = dict(pts)
    for n in ("back_left", "back_right", "front_left", "front_right"):
        flat[n] = np.zeros(3)
    assert rs.label_handedness(flat) == 0
    assert not rs.same_handedness(flat, flat)


def test_degenerate_corners_refused(tmp_path):
    doc = load(CAPTURES[0])
    for n in ("back_left", "back_right", "front_left", "front_right"):
        doc[n] = {"x": 0.1, "y": 0.1, "z": 0.1}
    src = tmp_path / "src.json"
    src.write_text(json.dumps(doc))
    with pytest.raises(rs.ConversionError, match="handedness"):
        rs.convert_file(src, tmp_path / "out.json", CALIB, allow_stale=True)
    assert not (tmp_path / "out.json").exists()


def test_double_conversion_refused(tmp_path):
    src = fresh_copy(tmp_path, CAPTURES[0])
    out = tmp_path / "out.json"
    rs.convert_file(src, out, CALIB)
    with pytest.raises(rs.ConversionError, match="double conversion"):
        rs.convert_file(out, tmp_path / "out2.json", CALIB, allow_stale=True)


def test_wrong_frame_and_schema_refused(tmp_path):
    for key, value, pattern in (("frame_id", "arm_right_base_link", "frame_id"),
                                ("schema", "other/v1", "schema")):
        doc = load(CAPTURES[0])
        doc[key] = value
        src = tmp_path / "bad.json"
        src.write_text(json.dumps(doc))
        with pytest.raises(rs.ConversionError, match=pattern):
            rs.convert_file(src, tmp_path / "o.json", CALIB, allow_stale=True)


def test_missing_point_refused(tmp_path):
    doc = load(CAPTURES[0])
    del doc["front_left"]
    src = tmp_path / "bad.json"
    src.write_text(json.dumps(doc))
    with pytest.raises(rs.ConversionError, match="front_left"):
        rs.convert_file(src, tmp_path / "o.json", CALIB, allow_stale=True)


def test_mtime_preserved_and_atomic(tmp_path):
    src = fresh_copy(tmp_path, CAPTURES[0], age_s=100.0)
    out = tmp_path / "out.json"
    out.write_text("old content")
    rs.convert_file(src, out, CALIB)
    assert out.stat().st_mtime_ns == src.stat().st_mtime_ns
    assert json.loads(out.read_text())["schema"] == "box_h_cut_snapshot/v1"
    assert not [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]


def test_stale_refused_and_allow_stale_marked(tmp_path):
    src = fresh_copy(tmp_path, CAPTURES[0], age_s=300.0)
    out = tmp_path / "out.json"
    with pytest.raises(rs.ConversionError, match="stale"):
        rs.convert_file(src, out, CALIB)
    assert not out.exists()
    rs.convert_file(src, out, CALIB, allow_stale=True)
    prov = json.loads(out.read_text())[rs.PROVENANCE_KEY]
    assert prov["allow_stale"] is True
    assert out.stat().st_mtime_ns == src.stat().st_mtime_ns  # still as old as the capture


def test_old_stamp_with_fresh_mtime_refused(tmp_path):
    src = fresh_copy(tmp_path, CAPTURES[0], age_s=1.0)
    doc = json.loads(src.read_text())
    doc["acquisition_stamp_ns"] = int((time.time() - 500) * 1e9)
    src.write_text(json.dumps(doc))  # mtime is now fresh again
    with pytest.raises(rs.ConversionError, match="acquisition_stamp_ns"):
        rs.convert_file(src, tmp_path / "o.json", CALIB)


def test_fresh_not_marked_stale(tmp_path):
    src = fresh_copy(tmp_path, CAPTURES[0], age_s=10.0)
    out = tmp_path / "out.json"
    rs.convert_file(src, out, CALIB)
    assert json.loads(out.read_text())[rs.PROVENANCE_KEY]["allow_stale"] is False


def test_provenance_and_other_fields_kept(tmp_path, calibration):
    src = fresh_copy(tmp_path, CAPTURES[1])
    out = tmp_path / "out.json"
    rs.convert_file(src, out, CALIB)
    before, after = load(src), load(out)
    prov = after[rs.PROVENANCE_KEY]
    assert prov["source_sha256"] == hashlib.sha256(src.read_bytes()).hexdigest()
    assert prov["calibration_sha256"] == hashlib.sha256(CALIB.read_bytes()).hexdigest()
    assert prov["relabel"] == "rotate_180"
    assert prov["arm_right_base"]["xyz"] == calibration["arm_right_base"]["xyz"]
    assert prov["source_mtime"] == src.stat().st_mtime
    assert after["frame_id"] == "base_link"
    skipped = set(rs.POINT_NAMES) | {rs.NORMAL_NAME, rs.PROVENANCE_KEY}
    for key in before:
        if key not in skipped:
            assert after[key] == before[key], key
    assert after["acquisition_stamp_ns"] == before["acquisition_stamp_ns"]


def test_same_source_and_output_refused(tmp_path):
    src = fresh_copy(tmp_path, CAPTURES[0])
    with pytest.raises(rs.ConversionError, match="different"):
        rs.convert_file(src, src, CALIB)


def test_camera_in_right_base_matches_reference(calibration):
    xyz, q = rs.camera_in_right_base(calibration)
    ref = np.linalg.inv(reference_pose(calibration["arm_right_base"]["xyz"], calibration["arm_right_base"]["rpy"])) @ \
        reference_pose(calibration["camera_link"]["xyz"], calibration["camera_link"]["rpy"])
    assert np.allclose(xyz, ref[:3, 3], atol=1e-12)
    assert np.allclose(Rotation.from_quat(q).as_matrix(), ref[:3, :3], atol=1e-12)
    # a few arbitrary rotations through the quaternion helper
    for rot in Rotation.random(20, random_state=1):
        m = rot.as_matrix()
        qq = rs.quaternion_xyzw(m)
        assert np.allclose(Rotation.from_quat(qq).as_matrix(), m, atol=1e-12)


def test_cli_ok_and_error(tmp_path, capsys):
    src = fresh_copy(tmp_path, CAPTURES[0])
    assert rs.main(["convert", str(src), str(tmp_path / "o.json")]) == 0
    assert rs.main(["convert", str(tmp_path / "o.json"), str(tmp_path / "p.json")]) == 2
    assert "double conversion" in capsys.readouterr().err
