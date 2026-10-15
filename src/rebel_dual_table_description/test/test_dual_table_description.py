"""Static checks of rebel_dual_table_description (no ROS graph needed; xacro + installed share dirs)."""
import os
import subprocess
import xml.etree.ElementTree as ET

import yaml
from ament_index_python.packages import get_package_share_directory

SHARE = get_package_share_directory("rebel_dual_table_description")
SINGLE_CFG = os.path.join(get_package_share_directory("igus_rebel"), "config")


def _xacro(*args):
    return subprocess.run(["xacro", os.path.join(SHARE, "urdf", "rebel_dual_table.urdf.xacro"), *args],
                          check=True, capture_output=True, text=True).stdout


def test_mock_xacro_has_two_generic_systems_and_no_scene_camera():
    root = ET.fromstring(_xacro("hardware_protocol:=mock_hardware"))
    blocks = {r.get("name"): r for r in root.findall("ros2_control")}
    assert set(blocks) == {"arm_left_hw", "arm_right_hw"}
    for b in blocks.values():
        assert b.find("hardware/plugin").text == "mock_components/GenericSystem"
        assert len(b.findall("joint")) == 6
    # the SCENE camera is not a URDF link (static TF from the calibration YAML); the only camera links are
    # the two wrist cameras that are part of the variant-C tool
    cams = sorted(ln.get("name") for ln in root.findall("link") if "camera" in ln.get("name"))
    assert cams == ["arm_left_knife_c_wrist_camera", "arm_right_knife_c_wrist_camera"]


def test_rebel_xacro_distinct_ips_nodes_and_fault_group():
    root = ET.fromstring(_xacro("hardware_protocol:=rebel"))
    params = {}
    for r in root.findall("ros2_control"):
        params[r.get("name")] = {p.get("name"): p.text for p in r.findall("hardware/param")}
        assert r.find("hardware/plugin").text == "Igus/Rebel"
        assert all(j.find("command_interface").get("name") == "velocity" for j in r.findall("joint"))
    left, right = params["arm_left_hw"], params["arm_right_hw"]
    assert left["robot_ip"] != right["robot_ip"]
    assert left["node_name"] != right["node_name"]
    assert left["fault_group"] == right["fault_group"] == "dual_arm_table"
    assert left["robot_port"] == right["robot_port"] == "3920"


def test_calibration_drives_arm_right_pose():
    path = os.path.join(SHARE, "config", "dual_arm_calibration.yaml")
    d = yaml.safe_load(open(path))["dual_arm_calibration"]
    assert d["world_frame"] == "world" and set(d) >= {"source", "date", "arm_right_base", "camera_link", "notes"}
    root = ET.fromstring(_xacro("hardware_protocol:=mock_hardware"))
    j = [x for x in root.findall("joint") if x.get("name") == "arm_right_base_joint"][0]
    xyz = [float(v) for v in j.find("origin").get("xyz").split()]
    assert xyz == d["arm_right_base"]["xyz"]


def _controller_params(path, prefix):
    d = yaml.safe_load(open(path))
    return d[prefix + "trajectory_controller"]["ros__parameters"]


def _strip(p, prefix):
    s = yaml.dump(p)
    return s.replace(prefix, "")


def test_controller_yaml_matches_single_arm():
    for dual, single in (("controllers.yaml", "controller.yaml"), ("controllers_mock.yaml", "controller_mock.yaml")):
        s = yaml.safe_load(open(os.path.join(SINGLE_CFG, single)))["rebel_arm_trajectory_controller"]["ros__parameters"]
        for prefix in ("arm_left_", "arm_right_"):
            p = _controller_params(os.path.join(SHARE, "config", dual), prefix)
            assert yaml.safe_load(_strip(p, prefix)) == s, (dual, prefix)
        cm = yaml.safe_load(open(os.path.join(SHARE, "config", dual)))["controller_manager"]["ros__parameters"]
        assert cm["update_rate"] == 100
        assert {"joint_state_broadcaster", "arm_left_trajectory_controller", "arm_right_trajectory_controller"} <= set(cm)


def test_fk_equivalence():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "check_fk", os.path.join(os.path.dirname(__file__), "..", "scripts", "check_fk_equivalence.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ok, worst, nlinks = mod.run(n=10, tol=1e-9, verbose=False)
    assert ok, worst
    assert nlinks >= 18      # arm (13 incl. motors) + igus_tool0 + tip + tcp point + holder/handpiece/camera


def _links_and_joints(args=("hardware_protocol:=mock_hardware",)):
    root = ET.fromstring(_xacro(*args))
    return root, {ln.get("name") for ln in root.findall("link")}, {j.get("name"): j for j in root.findall("joint")}


def test_both_arms_carry_the_shared_knife_c_tool_and_no_old_gripper():
    _, links, joints = _links_and_joints()
    for arm in ("arm_left_", "arm_right_"):
        for ln in ("igus_tool0", "knife_c_tcp_point", "xeg32_tool_tip", "knife_c_holder", "knife_c_handpiece",
                   "knife_c_wrist_camera"):
            assert arm + ln in links, arm + ln
        assert joints[arm + "igus_tool0_to_xeg32_tool_tip"].find("parent").get("link") == arm + "igus_tool0"
        assert joints[arm + "knife_c_tcp_point_to_xeg32_tool_tip"].find("child").get("link") == arm + "xeg32_tool_tip"
    assert not [ln for ln in links if "xeg32_knife" in ln or "xeg32_base" in ln or "carriage" in ln]


def test_tool_tip_has_no_collision_and_blade_has_none():
    root, _, _ = _links_and_joints()
    for ln in root.findall("link"):
        if ln.get("name").endswith("xeg32_tool_tip") or ln.get("name").endswith("knife_c_tcp_point"):
            assert not ln.findall("collision"), ln.get("name")


def test_link5_collision_is_the_single_arm_hull():
    root, _, _ = _links_and_joints()
    for arm in ("arm_left_", "arm_right_"):
        link5 = [x for x in root.findall("link") if x.get("name") == arm + "link5"][0]
        mesh = link5.find("collision/geometry/mesh")
        assert mesh is not None and mesh.get("filename").endswith("collision/link5_hull.stl")
        assert link5.find("collision/geometry/box") is None


def test_dual_tcp_default_equals_single_arm_wrapper_joint():
    """The per-arm TCP position defaults to the single-arm value (the wrapper keeps the literal joint)."""
    wrapper = os.path.join(get_package_share_directory("igus_rebel_description"), "urdf",
                           "igus_rebel_xeg32_wrapper.urdf.xacro")
    text = open(wrapper).read()
    import re
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    m = re.findall(r'<joint name="igus_tool0_to_xeg32_tool_tip".*?<origin xyz="([^"]+)" rpy="0 0 0"/>', text, flags=re.S)
    assert len(m) == 1          # the literal knife_c joint (the xeg32_knife_v1 joint is named through a property)
    single = m[0]
    _, _, joints = _links_and_joints()
    for arm in ("arm_left_", "arm_right_"):
        got = joints[arm + "igus_tool0_to_xeg32_tool_tip"].find("origin").get("xyz")
        assert [float(v) for v in got.split()] == [float(v) for v in single.split()]


def test_arm_macro_is_up_to_date_with_single_arm_urdf():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "make_arm_macro", os.path.join(os.path.dirname(__file__), "..", "scripts", "make_arm_macro.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    single = os.path.join(get_package_share_directory("igus_rebel_description"), "urdf", "igus_rebel2.urdf")
    assert mod.generate(single) == open(os.path.join(SHARE, "urdf", "rebel_arm.macro.xacro")).read()


def test_camera_tf_for_pole_exists_in_launch():
    text = open(os.path.join(SHARE, "launch", "dual_table.launch.py")).read()
    assert "camera_color_optical_frame" in text and "publish_optical_tf" in text


# ---- dual-only collision hulls (meshes/collision, make_arm_macro.DUAL_COLLISION_OVERRIDES) ----
# Hard-coded on purpose (not read from the table under test): the links whose single-arm primitives do not
# contain the real mesh. link5 is NOT here: the single-arm URDF already uses its hull.
EXPECTED_OVERRIDDEN = {"base_link", "motor1", "link1", "motor2", "link2", "motor3", "link3", "motor4", "link4",
                       "motor5", "link6"}
HULL_TOL = 1e-4          # containment tolerance 0.1 mm


def _load_script(name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(os.path.dirname(__file__), "..", "scripts", name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _mesh_path(uri):
    prefix = "package://rebel_dual_table_description/"
    assert uri.startswith(prefix), uri
    return os.path.join(SHARE, uri[len(prefix):])


def test_override_table_is_the_expected_link_set_and_files_exist():
    table = _load_script("make_arm_macro").DUAL_COLLISION_OVERRIDES
    assert set(table) == EXPECTED_OVERRIDDEN
    for link, meshes in table.items():
        assert 1 <= len(meshes) <= 4, link
        for m in meshes:
            assert os.path.isfile(_mesh_path(m)), m


def test_dual_collision_equals_single_arm_except_overridden_links():
    """Both arms: every collision/visual entry equals the single arm's, except the collision of the overridden
    links, which are exactly the override meshes (and nothing else)."""
    fk = _load_script("check_fk_equivalence")
    single_root = fk.xacro_to_urdf(
        get_package_share_directory("igus_rebel_description") + "/urdf/igus_rebel_xeg32_wrapper.urdf.xacro",
        ["hardware_protocol:=mock_hardware"])
    dual_root = fk.xacro_to_urdf(os.path.join(SHARE, "urdf", "rebel_dual_table.urdf.xacro"),
                                 ["hardware_protocol:=mock_hardware"])
    single = fk.geometry_signature(single_root, "")
    single.pop("world")
    table = _load_script("make_arm_macro").DUAL_COLLISION_OVERRIDES
    for arm in ("arm_left_", "arm_right_"):
        dual = fk.geometry_signature(dual_root, arm)
        assert "knife_c_blade" in dual                      # the one dual-only link (tested separately below)
        dual.pop("knife_c_blade")
        assert set(dual) == set(single)
        for link in single:
            if link in EXPECTED_OVERRIDDEN:
                cols = [e for e in dual[link] if e[0] == "collision"]
                assert [dict(e[4])["filename"] for e in cols] == table[link], (arm, link)
                assert all(e[1] == (0.0, 0.0, 0.0) and e[2] == (0.0, 0.0, 0.0) and e[3] == "mesh"
                           and len(e[4]) == 1 for e in cols), (arm, link)
                assert [e for e in dual[link] if e[0] == "visual"] == [e for e in single[link] if e[0] == "visual"]
            else:
                assert dual[link] == single[link], (arm, link)
    # the expectation builder used by check_fk_equivalence really distinguishes a primitive from the override
    exp = fk.expected_dual_signature(single, table)
    for link in EXPECTED_OVERRIDDEN:
        assert exp[link] != single[link]
    assert all(exp[k] == single[k] for k in single if k not in EXPECTED_OVERRIDDEN)


def test_override_hulls_contain_the_real_link_meshes():
    """Independent containment proof against the installed real meshes: every vertex and 100k surface samples of
    each real link (visual origin applied) lie inside at least one of its hull pieces; each piece is a watertight
    convex STL with a low face count."""
    import numpy as np
    import trimesh
    from scipy.spatial import ConvexHull
    table = _load_script("make_arm_macro").DUAL_COLLISION_OVERRIDES
    igus = get_package_share_directory("igus_rebel_description")
    urdf = ET.parse(os.path.join(igus, "urdf", "igus_rebel2.urdf")).getroot()
    for link, meshes in table.items():
        el = [x for x in urdf.findall("link") if x.get("name") == link][0]
        o = el.find("visual/origin")
        off = np.array([float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()])
        assert (o is None) or [float(v) for v in o.get("rpy", "0 0 0").split()] == [0, 0, 0]
        real = trimesh.load(os.path.join(igus, "meshes", "rebel_d00617809", link + ".dae"), force="mesh")
        pts = np.vstack([real.vertices + off,
                         trimesh.sample.sample_surface(real, 100000, seed=3)[0] + off])
        best = np.full(len(pts), np.inf)
        total_faces = 0
        for m in meshes:
            h = trimesh.load(_mesh_path(m), force="mesh")
            total_faces += len(h.faces)
            # facet planes from qhull on the STL vertices (the STL triangles near merged corners are slivers)
            ch = ConvexHull(h.vertices)
            assert h.is_watertight and abs(h.volume - ch.volume) <= 1e-4 * ch.volume, m   # closed surface, same volume as the hull of its vertices
            v = (pts @ ch.equations[:, :3].T + ch.equations[:, 3]).max(axis=1)
            best = np.minimum(best, v)
        assert best.max() <= HULL_TOL, (link, best.max())
        assert total_faces <= 600, (link, total_faces)


# ---- dual-only blade collision box (rebel_tool.macro.xacro) ----
def _single_xacro(*args):
    return subprocess.run(
        ["xacro", os.path.join(get_package_share_directory("igus_rebel_description"), "urdf",
                               "igus_rebel_xeg32_wrapper.urdf.xacro"), "hardware_protocol:=mock_hardware", *args],
        capture_output=True, text=True)


def test_single_arm_has_no_blade_collision_by_default():
    """blade_collision is OPT-IN in the shared macro / single-arm wrapper: the default model has no blade link."""
    for args in ((), ("blade_collision:=false",), ("tool_variant:=xeg32_knife_v1",)):
        single = ET.fromstring(_single_xacro(*args).stdout)
        assert not [ln for ln in single.findall("link") if "blade" in ln.get("name")], args
        assert not [m for m in single.findall("material") if "blade" in m.get("name")], args


def test_single_arm_blade_collision_is_the_dual_blade_box():
    """ONE definition: blade_collision:=true on the single-arm wrapper emits exactly the dual model's blade link /
    joint / materials (same box, frame, margins), and nothing else changes."""
    single = ET.fromstring(_single_xacro("blade_collision:=true").stdout)
    base = ET.fromstring(_single_xacro().stdout)
    dual = ET.fromstring(_xacro("hardware_protocol:=mock_hardware"))

    def norm(element, prefix):
        text = ET.tostring(element, encoding="unicode").replace(prefix, "")
        return "".join(text.split())

    for tag, name in (("link", "knife_c_blade"), ("joint", "igus_tool0_to_knife_c_blade"),
                      ("material", "knife_c_blade_material"), ("material", "knife_c_blade_box_material")):
        a = [e for e in single.findall(tag) if e.get("name") == name]
        b = [e for e in dual.findall(tag) if e.get("name") == "arm_left_" + name]
        assert len(a) == 1 and len(b) == 1, (tag, name)
        assert norm(a[0], "") == norm(b[0], "arm_left_"), (tag, name)
    added = [e for e in single if "blade" in (e.get("name") or "")]
    assert len(added) == 4
    for element in added:
        single.remove(element)
    assert norm(single, "") == norm(base, "")


def test_single_arm_blade_collision_rejects_bad_values():
    assert _single_xacro("blade_collision:=maybe").returncode != 0
    assert _single_xacro("blade_collision:=true", "tool_variant:=xeg32_knife_v1").returncode != 0


def test_wrapper_blade_tcp_equals_tcp_joint():
    """The wrapper passes the TCP xyz twice (the macro cannot read the joint block): the blade box would silently
    stop containing the TCP after a pivot patch (xacro_apply edits only the joint) if the copies drift."""
    import re
    wrapper = os.path.join(get_package_share_directory("igus_rebel_description"), "urdf",
                           "igus_rebel_xeg32_wrapper.urdf.xacro")
    text = re.sub(r"<!--.*?-->", "", open(wrapper).read(), flags=re.S)
    joint = re.findall(r'<joint name="igus_tool0_to_xeg32_tool_tip".*?<origin xyz="([^"]+)" rpy="0 0 0"/>', text,
                       flags=re.S)
    blade = re.findall(r'blade_tcp_xyz="([^"]+)"', text)
    assert len(joint) == 1 and len(blade) == 1
    assert [float(v) for v in joint[0].split()] == [float(v) for v in blade[0].split()]


def test_blade_link_box_contains_blade_and_tcp_segment_for_both_arms():
    import numpy as np
    fk = _load_script("check_fk_equivalence")
    root = ET.fromstring(_xacro("hardware_protocol:=mock_hardware"))
    model = fk.Model(root)
    s0, cad = fk.blade_cad_points()
    for arm in ("arm_left_", "arm_right_"):
        bxyz, bsize = fk.blade_box(root, arm)
        link = [x for x in root.findall("link") if x.get("name") == arm + "knife_c_blade"][0]
        assert len(link.findall("visual")) == 2 and link.find("visual/material").get("name").endswith("blade_material")
        T = model.fk(arm + "igus_tool0", arm + "knife_c_blade", {})
        tcp = model.fk(arm + "igus_tool0", arm + "xeg32_tool_tip", {})[:3, 3]
        assert np.abs(T[:3, 3] - s0).max() < 1e-6                     # frame = seat face centre
        pts = cad + [s0, tcp] + [s0 + t * (tcp - s0) for t in np.linspace(0, 1, 21)]
        margin = fk.blade_margin_ok(bxyz, bsize, T, pts)
        assert margin >= 0.001 - 1e-9, margin
        assert bsize.max() < 0.03
        # the blade point of the CAD (185.2 mm) is inside, the TCP is 4.1 mm from it
        tip = s0 + 0.0144 * fk.BLADE_AXIS
        assert np.linalg.norm(tip - tcp) < 0.0045
        # negative controls: a box that is 2 mm too short, or shifted off the blade, fails the margin
        short = bsize - np.array([0, 0, 0.004])
        assert fk.blade_margin_ok(bxyz, short, T, pts) < 0.001
        assert fk.blade_margin_ok(bxyz + np.array([0.003, 0, 0]), bsize, T, pts) < 0.001
        # the joint is fixed to igus_tool0 (rigid with handpiece and holder)
        j = [x for x in root.findall("joint") if x.get("name") == arm + "igus_tool0_to_knife_c_blade"][0]
        assert j.get("type") == "fixed" and j.find("parent").get("link") == arm + "igus_tool0"


def test_blade_handpiece_overlap_is_rigid_and_holder_is_clear():
    """Rigid proof for the SRDF: blade, handpiece and holder are all fixed to igus_tool0, so blade-handpiece contact is
    configuration independent. The box starts 1 mm inside the seat (a point 0.5 mm behind the seat face on the axis lies
    in the seat hull AND in the blade box -> contact at every configuration); the holder hulls never touch the box."""
    import numpy as np
    import trimesh
    from scipy.spatial import ConvexHull
    fk = _load_script("check_fk_equivalence")
    root = ET.fromstring(_xacro("hardware_protocol:=mock_hardware"))
    model = fk.Model(root)
    coll = os.path.join(get_package_share_directory("igus_rebel_description"), "meshes", "knife_c", "collision")
    arm = "arm_left_"
    T = model.fk(arm + "igus_tool0", arm + "knife_c_blade", {})
    bxyz, bsize = fk.blade_box(root, arm)
    p_k = np.array([0.0, 0.0, -0.0005])                    # 0.5 mm behind the seat front face, on the axis
    assert (np.abs(p_k - bxyz) <= bsize / 2).all()          # inside the blade box
    p_t = (T @ np.append(p_k, 1.0))[:3]
    seat = ConvexHull(trimesh.load(os.path.join(coll, "handpiece_seat.stl"), force="mesh").vertices)
    assert ((seat.equations[:, :3] @ p_t + seat.equations[:, 3]) < 0).all(), "not inside the seat hull"
    # the whole box against every holder piece: no point inside (all sampled points violate at least one plane)
    g = np.linspace(-0.5, 0.5, 9)
    pts_k = np.array([bxyz + bsize * np.array([a, b, c]) for a in g for b in g for c in g])
    pts_t = (T[:3, :3] @ pts_k.T).T + T[:3, 3]
    import glob
    pieces = sorted(glob.glob(os.path.join(coll, "holder_*.stl")))
    assert len(pieces) == 12 or len(pieces) >= 12, pieces
    for f in pieces:
        h = ConvexHull(trimesh.load(f, force="mesh").vertices)
        inside = ((pts_t @ h.equations[:, :3].T + h.equations[:, 3]).max(axis=1) < 0).any()
        assert not inside, f"blade box overlaps {os.path.basename(f)}"
