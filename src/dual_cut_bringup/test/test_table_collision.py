"""table_collision.py: the dual cell's table in each per-arm planning frame (pose math vs the dual URDF)."""
import importlib.util
import math
import os
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import yaml
from ament_index_python.packages import get_package_share_directory
from scipy.spatial.transform import Rotation

_HERE = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("table_collision", os.path.join(_HERE, "..", "scripts", "table_collision.py"))
table = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(table)

DESCRIPTION = get_package_share_directory("rebel_dual_table_description")
XACRO = os.path.join(DESCRIPTION, "urdf", "rebel_dual_table.urdf.xacro")
CALIBRATION = os.path.join(DESCRIPTION, "config", "dual_arm_calibration.yaml")

# Large, unequal angles so a wrong rotation order or a missing inverse cannot hide.
SYNTHETIC = """dual_arm_calibration:
  source: test
  arm_right_base:
    xyz: [0.031, -0.4521, 0.0153]
    rpy: [0.12, -0.2, 0.35]
"""


def transform(xyz, rpy):
    t = np.eye(4)
    t[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()      # URDF rpy = fixed-axis x, y, z
    t[:3, 3] = xyz
    return t


def urdf(calibration, *args):
    out = subprocess.run(["xacro", XACRO, "hardware_protocol:=mock_hardware", f"calibration_file:={calibration}",
                          *args], check=True, capture_output=True, text=True).stdout
    return ET.fromstring(out)


def joint_origin(root, name):
    joint = [j for j in root.findall("joint") if j.get("name") == name][0]
    origin = joint.find("origin")
    return [float(v) for v in origin.get("xyz").split()], [float(v) for v in origin.get("rpy").split()]


def table_box(root):
    link = [x for x in root.findall("link") if x.get("name") == "table"][0]
    return [float(v) for v in link.find("collision/geometry/box").get("size").split()]


def object_transform(collision):
    pose = collision.primitive_poses[0]
    t = np.eye(4)
    t[:3, :3] = Rotation.from_quat([pose.orientation.x, pose.orientation.y, pose.orientation.z,
                                    pose.orientation.w]).as_matrix()
    t[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
    return t


@pytest.fixture
def synthetic(tmp_path):
    path = tmp_path / "calibration.yaml"
    path.write_text(SYNTHETIC)
    return str(path)


def collision_for(arm, calibration, **kwargs):
    xyz, rpy = table.load_right_base(calibration)
    return table.table_collision_object(arm, xyz, rpy, **kwargs)


@pytest.mark.parametrize("which", ["installed", "synthetic"])
@pytest.mark.parametrize("arm", ["left", "right"])
def test_table_in_arm_frame_matches_the_dual_urdf(arm, which, synthetic):
    calibration = CALIBRATION if which == "installed" else synthetic
    root = urdf(calibration)
    world_table = transform(*joint_origin(root, "world_to_table"))
    world_arm = np.eye(4) if arm == "left" else transform(*joint_origin(root, "arm_right_base_joint"))
    expected = np.linalg.inv(world_arm) @ world_table
    collision = collision_for(arm, calibration)
    assert collision.id == "table" and collision.header.frame_id == "base_link"
    np.testing.assert_allclose(object_transform(collision), expected, atol=1e-9)
    # the box is the URDF collision box, padding 0 by default
    np.testing.assert_allclose(collision.primitives[0].dimensions, table_box(root), atol=1e-12)


def test_table_top_is_at_world_z_minus_one_mm(synthetic):
    # top face centre, mapped back to world through the arm frame
    for arm, calibration in (("left", CALIBRATION), ("right", CALIBRATION), ("right", synthetic)):
        xyz, rpy = table.load_right_base(calibration)
        collision = table.table_collision_object(arm, xyz, rpy)
        arm_world = np.eye(4) if arm == "left" else transform(xyz, rpy)
        top = np.append(object_transform(collision)[:3, :3] @ [0, 0, collision.primitives[0].dimensions[2] / 2]
                        + object_transform(collision)[:3, 3], 1.0)
        assert (arm_world @ top)[2] == pytest.approx(-0.001, abs=1e-9)


def test_non_default_table_arguments_follow_the_xacro(synthetic):
    root = urdf(synthetic, "table_size:=0.8", "table_thickness:=0.05", "table_base_setback:=0.2")
    world_table = transform(*joint_origin(root, "world_to_table"))
    world_arm = transform(*joint_origin(root, "arm_right_base_joint"))
    collision = collision_for("right", synthetic, size=0.8, thickness=0.05, setback=0.2)
    np.testing.assert_allclose(object_transform(collision), np.linalg.inv(world_arm) @ world_table, atol=1e-9)
    np.testing.assert_allclose(collision.primitives[0].dimensions, table_box(root), atol=1e-12)


def test_defaults_equal_the_xacro_defaults():
    root = ET.fromstring(subprocess.run(["xacro", XACRO, "hardware_protocol:=mock_hardware"], check=True,
                                        capture_output=True, text=True).stdout)
    assert table_box(root) == [table.DEFAULT_TABLE_SIZE, table.DEFAULT_TABLE_SIZE, table.DEFAULT_TABLE_THICKNESS]
    xyz, _ = joint_origin(root, "world_to_table")
    right_xyz, right_rpy = table.load_right_base(CALIBRATION)
    assert xyz == pytest.approx(table.table_world(right_xyz))
    assert xyz[0] == pytest.approx(table.DEFAULT_TABLE_BASE_SETBACK - table.DEFAULT_TABLE_SIZE / 2)
    # the launch defaults are the same numbers
    spec = importlib.util.spec_from_file_location("par", os.path.join(_HERE, "..", "launch", "per_arm_planning.launch.py"))
    planning = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(planning)
    args = {e.name: e.default_value[0].text for e in planning.generate_launch_description().entities
            if hasattr(e, "default_value") and e.default_value}
    assert float(args["table_size"]) == table.DEFAULT_TABLE_SIZE
    assert float(args["table_thickness"]) == table.DEFAULT_TABLE_THICKNESS
    assert float(args["table_base_setback"]) == table.DEFAULT_TABLE_BASE_SETBACK
    assert float(args["table_padding_m"]) == 0.010


def test_padding_grows_every_face_and_keeps_the_centre(synthetic):
    plain = collision_for("right", synthetic)
    padded = collision_for("right", synthetic, padding=0.003)
    np.testing.assert_allclose(object_transform(padded), object_transform(plain), atol=1e-12)
    np.testing.assert_allclose(np.array(padded.primitives[0].dimensions) - plain.primitives[0].dimensions, 0.006)
    with pytest.raises(ValueError):
        collision_for("right", synthetic, padding=-0.001)


def test_quaternion_is_unit_and_left_is_identity(synthetic):
    xyz, rpy = table.load_right_base(synthetic)
    position, q = table.table_pose_in_arm_frame("right", xyz, rpy)
    assert math.sqrt(sum(v * v for v in q)) == pytest.approx(1.0)
    left_position, left_q = table.table_pose_in_arm_frame("left", xyz, rpy)
    assert left_q == (0.0, 0.0, 0.0, 1.0)
    assert left_position == pytest.approx(table.table_world(xyz))
    with pytest.raises(ValueError):
        table.table_pose_in_arm_frame("middle", xyz, rpy)


def test_bad_calibration_is_refused(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({"dual_arm_calibration": {"arm_right_base": {"xyz": [0, 0], "rpy": [0, 0, 0]}}}))
    with pytest.raises(ValueError):
        table.load_right_base(str(bad))


# ---- ACM exemption of the base link ----

def _acm(names, allowed_pairs=(), defaults=None):
    from moveit_msgs.msg import AllowedCollisionEntry, AllowedCollisionMatrix
    matrix = [[False] * len(names) for _ in names]
    for a, b in allowed_pairs:
        matrix[names.index(a)][names.index(b)] = matrix[names.index(b)][names.index(a)] = True
    acm = AllowedCollisionMatrix()
    acm.entry_names = list(names)
    for row in matrix:
        entry = AllowedCollisionEntry()
        entry.enabled = row
        acm.entry_values.append(entry)
    acm.default_entry_names = list(defaults or [])
    acm.default_entry_values = [True] * len(acm.default_entry_names)
    return acm


def _allowed(acm, a, b):
    i, j = acm.entry_names.index(a), acm.entry_names.index(b)
    return bool(acm.entry_values[i].enabled[j]) and bool(acm.entry_values[j].enabled[i])


def test_acm_exemption_is_added_for_the_base_link_only_and_keeps_the_rest():
    acm = _acm(["base_link", "link1", "link2"], [("base_link", "link1"), ("link1", "link2")], defaults=["link2"])
    new, changed = table.acm_with_exemptions(acm, "table", ["base_link"])
    assert changed
    assert new.entry_names == ["base_link", "link1", "link2", "table"]
    assert _allowed(new, "table", "base_link")
    for link in ("link1", "link2"):
        assert not _allowed(new, "table", link)
    # old entries untouched, table row/column added symmetric, defaults kept
    assert _allowed(new, "base_link", "link1") and _allowed(new, "link1", "link2")
    assert not _allowed(new, "base_link", "link2")
    assert list(new.default_entry_names) == ["link2"] and list(new.default_entry_values) == [True]
    assert all(len(e.enabled) == 4 for e in new.entry_values)
    assert new.entry_values[3].enabled[3] is False or not new.entry_values[3].enabled[3]


def test_acm_exemption_is_idempotent_and_handles_an_existing_table_entry():
    acm = _acm(["base_link", "link1"])
    once, changed = table.acm_with_exemptions(acm, "table", ["base_link"])
    twice, changed_again = table.acm_with_exemptions(once, "table", ["base_link"])
    assert changed and not changed_again
    assert [list(e.enabled) for e in twice.entry_values] == [list(e.enabled) for e in once.entry_values]
    # table already listed but not allowed with the base: only the pair is switched
    existing = _acm(["base_link", "table"])
    new, changed = table.acm_with_exemptions(existing, "table", ["base_link"])
    assert changed and _allowed(new, "table", "base_link") and new.entry_names == ["base_link", "table"]
    # no links to exempt: the matrix is only extended by the table name, nothing allowed
    none, _ = table.acm_with_exemptions(_acm(["base_link"]), "table", [])
    assert not _allowed(none, "table", "base_link")


def test_acm_exemption_refuses_an_empty_or_ragged_matrix():
    from moveit_msgs.msg import AllowedCollisionEntry, AllowedCollisionMatrix
    with pytest.raises(ValueError):
        table.acm_with_exemptions(AllowedCollisionMatrix(), "table", ["base_link"])   # would replace the scene's ACM
    ragged = _acm(["a", "b"])
    ragged.entry_values[0] = AllowedCollisionEntry(enabled=[True])
    with pytest.raises(ValueError):
        table.acm_with_exemptions(ragged, "table", ["a"])
