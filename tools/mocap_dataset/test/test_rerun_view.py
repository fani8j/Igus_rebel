"""Tests for tools/mocap_dataset/rerun_view.py.  Run: python3 -m pytest -p no:anyio tools/mocap_dataset/test"""
import os
import re
import subprocess
import xml.etree.ElementTree as ET
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import rerun_view as rv  # noqa: E402

DATASET_ROOT = Path(os.environ.get(
    "MOCAP_DATASET_ROOT",
    "~/Downloads/MOCAP_DATASET_FLAP_TOP_OPEN-20261027T064403Z-1-001/MOCAP_DATASET_FLAP_TOP_OPEN")).expanduser()
EPISODE = DATASET_ROOT / "mocap_left_new_10"

MINI_URDF = """<robot name="mini">
  <link name="world"/><link name="a"/><link name="b"/><link name="tip"/>
  <joint name="j0" type="fixed"><parent link="world"/><child link="a"/><origin xyz="0 0 0.1" rpy="0 0 0.3"/></joint>
  <joint name="arm_left_joint1" type="revolute"><parent link="a"/><child link="b"/><origin xyz="0.1 0 0.2" rpy="0.2 0 0"/><axis xyz="0 0 1"/></joint>
  <joint name="tool" type="fixed"><parent link="b"/><child link="tip"/><origin xyz="0 0 0.3"/></joint>
</robot>"""


def quat_to_R(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def test_joint_quaternions_match_axis_rotation():
    kin = rv._import_kinematics()
    axis = np.array([0.0, -1.0, 0.0])
    ang = np.array([0.0, 0.7, -2.9, np.pi])
    qs = rv.joint_quaternions(ang, axis)
    assert qs.shape == (4, 4) and np.allclose(np.linalg.norm(qs, axis=1), 1.0)
    assert np.allclose(qs[0], [0, 0, 0, 1])
    Rs = kin._axis_rotations(axis, ang)[:, :3, :3]
    for q, R in zip(qs, Rs):
        assert np.allclose(quat_to_R(q), R, atol=1e-12)


def test_entity_paths_follow_joint_tree():
    kin = rv._import_kinematics()
    u = kin.Urdf.parse(MINI_URDF)
    p = rv.entity_paths(u, "mini", prefix="cell")
    assert p["world"] == "/cell/mini/world"
    assert p["b"] == "/cell/mini/world/j0/a/arm_left_joint1/b"
    assert p["tip"] == p["b"] + "/tool/tip"


DAE_TETRA = """<?xml version="1.0" encoding="utf-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
<asset><unit name="meter" meter="1"/><up_axis>Z_UP</up_axis></asset>
<library_geometries><geometry id="g" name="g"><mesh>
<source id="pos"><float_array id="pos-a" count="12">0 0 0  0.1 0 0  0 0.2 0  0 0 0.3</float_array>
<technique_common><accessor source="#pos-a" count="4" stride="3"><param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/></accessor></technique_common></source>
<vertices id="v"><input semantic="POSITION" source="#pos"/></vertices>
<triangles count="4"><input semantic="VERTEX" source="#v" offset="0"/><p>0 2 1  0 1 3  0 3 2  1 2 3</p></triangles>
</mesh></geometry></library_geometries>
<library_visual_scenes><visual_scene id="s"><node id="n"><instance_geometry url="#g"/></node></visual_scene></library_visual_scenes>
<scene><instance_visual_scene url="#s"/></scene></COLLADA>
"""


def test_dae_to_glb_rewrite(tmp_path):
    pytest.importorskip("trimesh")
    ament = tmp_path / "install" / "fakepkg"
    mesh_dir = ament / "share" / "fakepkg" / "meshes"
    mesh_dir.mkdir(parents=True)
    (mesh_dir / "cube.dae").write_text(DAE_TETRA)
    (mesh_dir / "hull.stl").write_text("solid x\nendsolid x\n")
    urdf = ('<robot><link name="a"><visual><geometry><mesh filename="package://fakepkg/meshes/cube.dae"/></geometry></visual>'
            '<collision><geometry><mesh filename="package://fakepkg/meshes/hull.stl"/></geometry></collision></link></robot>')
    cache = tmp_path / "cache"
    out = rv.rewrite_dae_meshes(urdf, str(ament), cache)
    assert ".dae" not in out and "package://fakepkg/meshes/hull.stl" in out       # only .dae rewritten
    glb = Path(re.search(r'filename="([^"]+\.glb)"', out).group(1))
    assert glb.is_absolute() and glb.parent == cache and glb.read_bytes()[:4] == b"glTF"
    import trimesh
    b = trimesh.load(str(glb), force="scene").bounds
    assert np.allclose(b, [[0, 0, 0], [0.1, 0.2, 0.3]], atol=1e-6)                # metres, Z-up values unchanged
    # cached by path + mtime: same file reused, new mtime -> new file
    mt = glb.stat().st_mtime_ns
    assert rv.rewrite_dae_meshes(urdf, str(ament), cache) == out and glb.stat().st_mtime_ns == mt
    dae = mesh_dir / "cube.dae"
    os.utime(dae, ns=(dae.stat().st_atime_ns, dae.stat().st_mtime_ns + 10**9))
    assert rv.rewrite_dae_meshes(urdf, str(ament), cache) != out
    with pytest.raises(FileNotFoundError):
        rv.rewrite_dae_meshes(urdf.replace("fakepkg", "nopkg"), str(ament), cache)


def test_real_arm_glb_matches_collision_hulls(cell, tmp_path):
    """Converted ReBeL visuals (same link frame) must agree with the collision hull bounding boxes."""
    import trimesh
    kin, urdf, text, ament = cell
    root = ET.fromstring(text)
    checked = 0
    for L in root.findall("link"):
        n = L.get("name")
        if not n.startswith("arm_left_") or L.find("visual/geometry/mesh") is None:
            continue
        vf = L.find("visual/geometry/mesh").get("filename")
        if not vf.endswith(".dae"):
            continue
        glb = rv.dae_to_glb(rv._resolve_package_file(*re.match(r"package://([^/]+)/(.*)", vf).groups(), ament), tmp_path)
        vb = trimesh.load(str(glb), force="scene").bounds
        hb = []
        for c in L.findall("collision/geometry/mesh"):
            pkg, rel = re.match(r"package://([^/]+)/(.*)", c.get("filename")).groups()
            hb.append(trimesh.load(str(rv._resolve_package_file(pkg, rel, ament))).bounds)
        hb = np.array(hb)
        hull = np.array([hb[:, 0].min(0), hb[:, 1].max(0)])
        assert np.abs(vb - hull).max() < 0.005, (n, vb, hull)       # < 5 mm (measured worst 1.5 mm)
        checked += 1
    assert checked == 12


def _compose_entity_tree(kin, urdf, paths, tip, q, joints):
    """What rerun does: product over the entity chain of the static joint origin and the animated link rotation."""
    chain = urdf.chain(rv.ROOT_LINK, tip)
    angle = {name: q[k] for k, (name, _, _) in enumerate(joints)}
    axis = {name: ax for name, _, ax in joints}
    T = np.eye(4)
    for j in chain.joints:
        T = T @ kin._origin_T(j.xyz, j.rpy)                      # static Transform3D on the joint entity
        if j.name in angle:                                       # animated Transform3D on the child link entity
            Rt = np.eye(4)
            Rt[:3, :3] = quat_to_R(rv.joint_quaternions(np.array([angle[j.name]]), axis[j.name])[0])
            T = T @ Rt
    return T


@pytest.fixture(scope="module")
def cell():
    try:
        text, ament = rv.expand_cell_urdf()
    except SystemExit as e:
        pytest.skip(f"xacro / workspace not available: {e}")
    kin = rv._import_kinematics()
    return kin, kin.Urdf.parse(text), text, ament


def test_real_urdf_entity_composition_equals_fk(cell):
    kin, urdf, text, _ = cell
    paths = rv.entity_paths(urdf, "rebel_dual_table")
    joints = rv.joint_entities(urdf, paths)
    assert [n for n, _, _ in joints] == [f"arm_left_joint{i}" for i in range(1, 7)]
    for name, ent, _ in joints:                   # the animated entity is the child link, below its joint entity
        assert f"/{name}/arm_left_link" in ent
    chain = urdf.chain(rv.ROOT_LINK, rv.TCP_LINK)
    assert chain.joint_names == [n for n, _, _ in joints]
    rng = np.random.default_rng(1)
    for q in rng.uniform(-2.5, 2.5, size=(5, 6)):
        assert np.allclose(_compose_entity_tree(kin, urdf, paths, rv.TCP_LINK, q, joints), chain.fk(q), atol=1e-9)
    # the right arm is untouched and the collision entities are listed for hiding
    cols = rv.collision_entities(text, paths)
    assert cols and all("/collision_" in c for c in cols)


@pytest.mark.skipif(not EPISODE.is_dir(), reason=f"dataset {EPISODE} not present")
def test_end_to_end_save(tmp_path):
    import rerun.dataframe as rd
    out = tmp_path / "ep.rrd"
    r = subprocess.run([sys.executable, str(HERE.parent / "rerun_view.py"), str(EPISODE), "--save", str(out)],
                       cwd=tmp_path, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "skipped all-zero channels" in r.stdout and "TCP (" in r.stdout
    rec = rd.load_recording(str(out))
    temporal = {}
    for c in rec.schema().component_columns():
        if not c.is_static:
            temporal.setdefault(c.entity_path, []).append(c.component)
    for v in ("head", "wrist_left", "knife"):
        assert f"/video/{v}" in temporal and f"/video_assets/{v}" not in temporal
    for j in range(1, 7):
        assert f"/plots/arm_left_joint{j}/state" in temporal and f"/plots/arm_left_joint{j}/action" in temporal
    assert not any("arm_right" in e and e.startswith("/plots") for e in temporal)
    assert "/tcp/tip" in temporal
    assert sum("Transform3D:quaternion" in v for v in temporal.values()) == 6
    t = rec.view(index="time", contents="/plots/arm_left_joint1/**").select().read_all()["time"].to_pylist()
    ep = rv.load_episode(EPISODE)
    assert len(t) == len(ep.timestamp) and abs(t[-1].total_seconds() - ep.timestamp[-1]) < 1e-3
    fr = rec.view(index="frame", contents="/tcp/tip").select().read_all()["frame"].to_pylist()
    assert fr[0] == 0 and fr[-1] == len(ep.timestamp) - 1


def test_parent_dir_requires_episode_dir():
    if not DATASET_ROOT.is_dir():
        pytest.skip("dataset not present")
    with pytest.raises(SystemExit):
        rv.find_episode_root(DATASET_ROOT, None)
    assert rv.find_episode_root(DATASET_ROOT, "mocap_left_13").name == "mocap_left_13"
