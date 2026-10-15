"""AABB of all collision geometry of the single-arm robot model at joint zero ("home").

Used to build the static `left_arm_parked` collision object for the right-arm-alone planning
scene: the left ReBeL is physically in the cell (parked at the SRDF "home" pose, all joints 0) but
is not part of the single-arm right model, so its volume is added to the planning scene as a box.

The AABB is axis-aligned in the LEFT base frame (collision geometry at joint zero, expressed in the
robot's base_link), padded on every side, and then placed at the left base pose as seen from the
right base (B_right^-1), i.e. an oriented box in the right arm's base_link frame.
"""

import math
import re
import struct
import xml.etree.ElementTree as ET

import numpy as np

import right_snapshot as rs


def _floats(text, default):
    if text is None:
        return list(default)
    return [float(v) for v in text.split()]


def _origin_pose(element):
    if element is None:
        return np.eye(4)
    return rs.pose_from_xyz_rpy(_floats(element.get("xyz"), (0, 0, 0)),
                                _floats(element.get("rpy"), (0, 0, 0)))


def read_stl_vertices(path):
    """All vertices (N x 3) of a binary or ASCII STL."""
    with open(path, "rb") as handle:
        data = handle.read()
    if len(data) >= 84:
        count = struct.unpack("<I", data[80:84])[0]
        if 84 + 50 * count == len(data):
            dtype = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
            tri = np.frombuffer(data, dtype=dtype, count=count, offset=84)
            return tri["v"].reshape(-1, 3).astype(float)
    text = data.decode("ascii", errors="ignore")
    verts = re.findall(r"vertex\s+(\S+)\s+(\S+)\s+(\S+)", text)
    if not verts:
        raise ValueError(f"cannot parse STL {path}")
    return np.array(verts, dtype=float)


def link_poses_at_zero(root):
    """{link: 4x4 pose in the URDF root link}. Every joint (revolute included) at 0."""
    children = {}
    child_links = set()
    for joint in root.findall("joint"):
        parent, child = joint.find("parent").get("link"), joint.find("child").get("link")
        children.setdefault(parent, []).append((child, _origin_pose(joint.find("origin"))))
        child_links.add(child)
    roots = [link.get("name") for link in root.findall("link") if link.get("name") not in child_links]
    if len(roots) != 1:
        raise ValueError(f"URDF must have exactly one root link, found {roots}")
    poses = {roots[0]: np.eye(4)}
    stack = [roots[0]]
    while stack:
        parent = stack.pop()
        for child, origin in children.get(parent, []):
            poses[child] = poses[parent] @ origin
            stack.append(child)
    return poses


def _geometry_extent(geometry, pose, resolve):
    """(min, max) world-axis-aligned extent of one collision geometry placed at `pose`."""
    rot, pos = pose[:3, :3], pose[:3, 3]
    box = geometry.find("box")
    cyl = geometry.find("cylinder")
    sph = geometry.find("sphere")
    mesh = geometry.find("mesh")
    if box is not None:
        half = np.array(_floats(box.get("size"), (0, 0, 0))) / 2.0
        ext = np.abs(rot) @ half
        return pos - ext, pos + ext
    if cyl is not None:
        radius, length = float(cyl.get("radius")), float(cyl.get("length"))
        axis = rot[:, 2]
        ext = np.array([0.5 * length * abs(axis[i]) + radius * math.sqrt(max(0.0, 1.0 - axis[i] ** 2))
                        for i in range(3)])
        return pos - ext, pos + ext
    if sph is not None:
        radius = float(sph.get("radius"))
        return pos - radius, pos + radius
    if mesh is not None:
        scale = np.array(_floats(mesh.get("scale"), (1, 1, 1)))
        verts = read_stl_vertices(resolve(mesh.get("filename"))) * scale
        world = verts @ rot.T + pos
        return world.min(axis=0), world.max(axis=0)
    raise ValueError("unsupported collision geometry")


def collision_aabb(urdf_xml, resolve):
    """(min, max, n_geometries) of all collision geometry at joint zero in the URDF root frame.
    `resolve(filename)` maps package:// (or file) URIs to local paths."""
    root = ET.fromstring(urdf_xml)
    poses = link_poses_at_zero(root)
    lows, highs, count = [], [], 0
    for link in root.findall("link"):
        for collision in link.findall("collision"):
            pose = poses[link.get("name")] @ _origin_pose(collision.find("origin"))
            low, high = _geometry_extent(collision.find("geometry"), pose, resolve)
            lows.append(low)
            highs.append(high)
            count += 1
    if not count:
        raise ValueError("URDF has no collision geometry")
    return np.min(lows, axis=0), np.max(highs, axis=0), count


def parked_box(low, high, padding, calibration):
    """Box for the left arm in the right base frame.
    Returns (dims[3], position[3], quaternion xyzw): the padded AABB (left base axes) placed at
    the left base pose seen from the right base, B_right^-1."""
    low = np.asarray(low) - padding
    high = np.asarray(high) + padding
    dims = high - low
    centre_left = 0.5 * (low + high)
    left_in_right = rs.invert_pose(rs.right_base_pose(calibration))
    position = left_in_right[:3, :3] @ centre_left + left_in_right[:3, 3]
    return dims, position, rs.quaternion_xyzw(left_in_right[:3, :3])
