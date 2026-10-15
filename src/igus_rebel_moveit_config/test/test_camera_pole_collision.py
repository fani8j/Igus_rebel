"""Tests for camera_pole_collision.pole_objects (pure geometry; needs a sourced ROS environment).

Run: python3 -m pytest -p no:anyio src/igus_rebel_moveit_config/test/test_camera_pole_collision.py
"""
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "camera_pole_collision.py"
spec = importlib.util.spec_from_file_location("camera_pole_collision", SCRIPT)
cpc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cpc)


def quat_from_rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = (math.cos(r / 2), math.sin(r / 2), math.cos(p / 2), math.sin(p / 2),
                              math.cos(y / 2), math.sin(y / 2))
    return (sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy)


def rot(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def tf(pos, q):
    return SimpleNamespace(transform=SimpleNamespace(
        translation=SimpleNamespace(x=pos[0], y=pos[1], z=pos[2]),
        rotation=SimpleNamespace(x=q[0], y=q[1], z=q[2], w=q[3])))


OPTICAL_IN_CAMERA = quat_from_rpy(-math.pi / 2, 0.0, -math.pi / 2)


def make(pos, yaw, pitch_down):
    """camera_link at pos, yawed and pitched down (REP-103: +pitch about y tilts x downward)."""
    q = quat_from_rpy(0.0, pitch_down, yaw)
    r_opt = rot(q) @ rot(OPTICAL_IN_CAMERA)
    # rotation matrix -> quaternion (xyzw), via scipy-free helper
    optical_q = mat_to_quat(r_opt)
    return tf(pos, q), tf(pos, optical_q), q


def mat_to_quat(m):
    t = np.trace(m)
    if t > 0:
        s = math.sqrt(t + 1) * 2
        return ((m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, s / 4)
    i = int(np.argmax(np.diag(m)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(1 + m[i, i] - m[j, j] - m[k, k]) * 2
    q = [0.0] * 4
    q[i] = s / 4
    q[j] = (m[j, i] + m[i, j]) / s
    q[k] = (m[k, i] + m[i, k]) / s
    q[3] = (m[k, j] - m[j, k]) / s
    return tuple(q)


def old_pole(camera, optical):
    """The pre-change formula (HEAD_SIZE 0.10 cube): pole pose x, y, yaw quaternion z, w."""
    q = optical.transform.rotation
    norm = math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
    x, y, z, w = q.x / norm, q.y / norm, q.z / norm, q.w / norm
    fx, fy = 2 * (x * z + w * y), 2 * (y * z - w * x)
    h = math.hypot(fx, fy)
    fx, fy = fx / h, fy / h
    p = camera.transform.translation
    yaw = math.atan2(fy, fx)
    back = 0.10 / 2.0 + 0.03 / 2.0
    return (p.x - back * fx, p.y - back * fy, 0.5, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


CASES = [
    ("level", (0.1, -0.2, 0.48), 0.0, 0.0),
    ("tilt30", (-0.05, -0.21, 0.87), 0.4, math.radians(30)),
    ("tilt45", (0.3, 0.1, 0.5), -1.2, math.radians(45)),
    ("yawed_level", (0.0, 0.0, 0.5), 2.5, 0.0),
]


@pytest.mark.parametrize("name,pos,yaw,pitch", CASES)
def test_pole_bit_identical_to_old_formula(name, pos, yaw, pitch):
    camera, optical, _ = make(pos, yaw, pitch)
    pole, _ = cpc.pole_objects(camera, optical, "base_link")
    pp = pole.primitive_poses[0]
    got = (pp.position.x, pp.position.y, pp.position.z, pp.orientation.z, pp.orientation.w)
    assert got == old_pole(camera, optical)
    assert list(pole.primitives[0].dimensions) == [0.03, 0.03, 1.0]
    assert pole.id == "camera_pole" and pole.header.frame_id == "base_link"


@pytest.mark.parametrize("name,pos,yaw,pitch", CASES)
def test_head_dims_and_pose_follow_full_orientation(name, pos, yaw, pitch):
    camera, optical, q = make(pos, yaw, pitch)
    _, head = cpc.pole_objects(camera, optical, "base_link")
    dims = np.array(head.primitives[0].dimensions)
    # Gemini 336 body 30.5 x 89.6 x 25.2 mm (rounded outward) + 2 x 10 mm margin
    np.testing.assert_allclose(dims, [0.0501, 0.1096, 0.0452], atol=1e-9)
    hp = head.primitive_poses[0]
    got_q = np.array([hp.orientation.x, hp.orientation.y, hp.orientation.z, hp.orientation.w])
    np.testing.assert_allclose(got_q, q, atol=1e-12)
    # centre = camera_link + R * (body centre in camera_link)
    centre_local = np.array([(-0.0258 + 0.0043) / 2, (-0.0698 + 0.0198) / 2, 0.0])
    expect = np.array(pos) + rot(q) @ centre_local
    np.testing.assert_allclose([hp.position.x, hp.position.y, hp.position.z], expect, atol=1e-12)
    # every body-mesh corner (+margin) is inside the box in the world frame
    R = rot(got_q)
    c = np.array([hp.position.x, hp.position.y, hp.position.z])
    for cx in (-0.0258, 0.0043):
        for cy in (-0.0698, 0.0198):
            for cz in (-0.0126, 0.0126):
                local = R.T @ (np.array(pos) + rot(q) @ np.array([cx, cy, cz]) - c)
                assert np.all(np.abs(local) <= dims / 2 - 0.0099)


def test_level_camera_head_is_thin_in_z_and_tilted_camera_is_not_axis_aligned():
    camera, optical, _ = make((0, 0, 0.5), 0.0, 0.0)
    _, head = cpc.pole_objects(camera, optical, "base_link")
    assert head.primitive_poses[0].orientation.w == pytest.approx(1.0)
    camera, optical, _ = make((0, 0, 0.5), 0.0, math.radians(30))
    _, head = cpc.pole_objects(camera, optical, "base_link")
    assert abs(head.primitive_poses[0].orientation.y) > 0.2


def test_fail_closed():
    camera, optical, _ = make((0, 0, 0.5), 0.3, 0.2)
    with pytest.raises(ValueError):
        cpc.pole_objects(tf((0, 0, 0.5), (0, 0, 0, 0)), optical, "f")
    with pytest.raises(ValueError):
        cpc.pole_objects(camera, tf((0, 0, 0.5), (0, 0, 0, 0)), "f")
    with pytest.raises(ValueError):
        cpc.pole_objects(camera, tf((0, 0, 0.5), (float("nan"),) * 4), "f")
    with pytest.raises(ValueError):
        cpc.pole_objects(tf((0, float("nan"), 0.5), (0, 0, 0, 1)), optical, "f")
    with pytest.raises(ValueError):
        cpc.pole_objects(tf((0, 0, float("inf")), (0, 0, 0, 1)), optical, "f")
