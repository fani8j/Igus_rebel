"""Checks of config/rebel_dual_table.srdf against the dual URDF (xacro) and the single-arm SRDF."""
import importlib.util
import os
import subprocess
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory

SHARE = get_package_share_directory("rebel_dual_table_moveit_config")
DESC = get_package_share_directory("rebel_dual_table_description")
SRDF = os.path.join(SHARE, "config", "rebel_dual_table.srdf")
ARMS = ("arm_left_", "arm_right_")


def _urdf_links():
    out = subprocess.run(["xacro", os.path.join(DESC, "urdf", "rebel_dual_table.urdf.xacro"),
                          "hardware_protocol:=mock_hardware"], check=True, capture_output=True, text=True).stdout
    return {ln.get("name") for ln in ET.fromstring(out).findall("link")}


def _srdf():
    return ET.parse(SRDF).getroot()


def _make_srdf():
    spec = importlib.util.spec_from_file_location(
        "make_srdf", os.path.join(os.path.dirname(__file__), "..", "scripts", "make_srdf.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_srdf_is_up_to_date_with_generator():
    mod = _make_srdf()
    text, _, _ = mod.generate(
        os.path.join(get_package_share_directory("igus_rebel_moveit_config"), "config", "igus_rebel2.srdf"),
        os.path.join(get_package_share_directory("igus_rebel_description"), "urdf",
                     "igus_rebel_xeg32_wrapper.urdf.xacro"))
    assert text == open(SRDF).read()


def test_all_srdf_links_exist_in_urdf():
    links = _urdf_links()
    root = _srdf()
    for d in root.findall("disable_collisions"):
        assert d.get("link1") in links and d.get("link2") in links, (d.get("link1"), d.get("link2"))
    for g in root.findall("group"):
        for ln in g.findall("link"):
            assert ln.get("name") in links
        for c in g.findall("chain"):
            assert c.get("tip_link") in links


def test_no_cross_arm_pair_is_disabled_and_every_intra_arm_pair_is_symmetric():
    root = _srdf()
    pairs = [(d.get("link1"), d.get("link2")) for d in root.findall("disable_collisions")]
    for a, b in pairs:
        arm_a = [p for p in ARMS if a.startswith(p)]
        arm_b = [p for p in ARMS if b.startswith(p)]
        assert not (arm_a and arm_b and arm_a != arm_b), (a, b)
    left = {(a[len(ARMS[0]):], b[len(ARMS[0]):]) for a, b in pairs if a.startswith(ARMS[0]) and b.startswith(ARMS[0])}
    right = {(a[len(ARMS[1]):], b[len(ARMS[1]):]) for a, b in pairs if a.startswith(ARMS[1]) and b.startswith(ARMS[1])}
    assert left == right and len(left) > 60


def test_groups_and_end_effectors_use_the_tcp():
    root = _srdf()
    groups = {g.get("name"): g for g in root.findall("group")}
    assert {"arm_left", "arm_right", "both_arms", "knife_c_tool_left", "knife_c_tool_right"} <= set(groups)
    assert not [g for g in groups if g.startswith("xeg32")]
    for arm in ARMS:
        chain = groups[arm.rstrip("_")].find("chain")
        assert chain.get("base_link") == "world" and chain.get("tip_link") == arm + "xeg32_tool_tip"
        tool = {ln.get("name") for ln in groups["knife_c_tool_" + arm.split("_")[1]].findall("link")}
        assert tool == {arm + "knife_c_holder", arm + "knife_c_handpiece", arm + "knife_c_wrist_camera",
                        arm + "knife_c_blade"}      # blade = dual-only collision box, part of the tool
    assert [x.get("name") for x in groups["both_arms"].findall("group")] == ["arm_left", "arm_right"]
    for e in root.findall("end_effector"):
        assert e.get("parent_link").endswith("xeg32_tool_tip")


def test_knife_c_pairs_are_present_for_both_arms():
    root = _srdf()
    pairs = {(d.get("link1"), d.get("link2")) for d in root.findall("disable_collisions")}
    for arm in ARMS:
        assert (arm + "link6", arm + "knife_c_holder") in pairs
        assert (arm + "knife_c_handpiece", arm + "knife_c_holder") in pairs
        # kept checked in the single-arm SRDF, so they must stay checked here
        assert (arm + "link5", arm + "knife_c_wrist_camera") not in pairs
        assert (arm + "knife_c_wrist_camera", arm + "link5") not in pairs


# ---- pairs that the dual-only hull collision geometry forces back to "checked" ----
# Hard-coded on purpose (not read from the generator's table).
EXPECTED_REENABLED = {frozenset(p) for p in (
    ("knife_c_handpiece", "link2"), ("knife_c_holder", "link2"), ("knife_c_holder", "link4"),
    ("knife_c_wrist_camera", "link2"), ("link1", "motor4"), ("link3", "motor1"))}


def _intra_pairs(root, arm):
    return {(d.get("link1")[len(arm):], d.get("link2")[len(arm):], d.get("reason")) for d in root.findall("disable_collisions")
            if d.get("link1").startswith(arm) and d.get("link2").startswith(arm)}


def test_reenabled_pairs_are_checked_and_nothing_else_differs_from_the_single_arm_srdf():
    import re
    mod = _make_srdf()
    assert {frozenset((a, b)) for a, b, _ in mod.DUAL_REENABLED_PAIRS} == EXPECTED_REENABLED
    single_text = re.sub(r"<!--.*?-->", "", open(os.path.join(
        get_package_share_directory("igus_rebel_moveit_config"), "config", "igus_rebel2.srdf")).read(), flags=re.S)
    single = {(d.get("link1"), d.get("link2"), d.get("reason")) for d in ET.fromstring(single_text).findall("disable_collisions")}
    links = _urdf_links()
    root = _srdf()
    for arm in ARMS:
        dual = _intra_pairs(root, arm)
        dual_keys = {frozenset((a, b)) for a, b, _ in dual}
        assert not (dual_keys & EXPECTED_REENABLED), "a re-enabled pair is disabled again"
        expected = {(a, b, r) for a, b, r in single
                    if arm + a in links and arm + b in links and frozenset((a, b)) not in EXPECTED_REENABLED}
        expected |= {("knife_c_blade", "knife_c_handpiece", "Adjacent")}    # the only dual-only disabled pair
        assert dual == expected, sorted(dual ^ expected)     # no pair is disabled that the single-arm SRDF keeps checked
    for key in EXPECTED_REENABLED:
        assert any(frozenset((a, b)) == key and r == "Never" for a, b, r in single), key   # only Never pairs are re-enabled


def test_blade_is_disabled_only_against_its_own_rigid_handpiece():
    """knife_c_blade (dual-only box) overlaps the handpiece seat rigidly -> that one pair is disabled. The holder (no
    contact at any configuration), every other own-arm link, the table and everything of the other arm stay checked."""
    root = _srdf()
    pairs = [(d.get("link1"), d.get("link2"), d.get("reason")) for d in root.findall("disable_collisions")]
    blade_pairs = [(a, b, r) for a, b, r in pairs if "knife_c_blade" in a or "knife_c_blade" in b]
    assert sorted(blade_pairs) == sorted(
        (arm + "knife_c_blade", arm + "knife_c_handpiece", "Adjacent") for arm in ARMS), blade_pairs
    mod = _make_srdf()
    assert mod.DUAL_ONLY_DISABLED_PAIRS == [("knife_c_blade", "knife_c_handpiece", "Adjacent")]
    assert "knife_c_blade" in mod.DUAL_ONLY_TOOL_LINKS
    assert "knife_c_blade" in _urdf_links() or any(arm + "knife_c_blade" in _urdf_links() for arm in ARMS)
