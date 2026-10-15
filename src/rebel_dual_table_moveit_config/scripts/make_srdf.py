#!/usr/bin/env python3
"""Generate config/rebel_dual_table.srdf from the single-arm SRDF.

    python3 scripts/make_srdf.py [--single <igus_rebel2.srdf>] [--urdf <single-arm urdf>] [--out config/rebel_dual_table.srdf]
    python3 scripts/make_srdf.py --check        # exit 1 if the checked-in SRDF is stale

Every intra-arm disable_collisions pair of the single-arm SRDF (igus_rebel_moveit_config/config/
igus_rebel2.srdf: arm links, variant-C knife_c_* tool links, ...) is copied per arm with the arm prefix
(arm_left_ / arm_right_). There are no hand-written link lists: pairs that name a link which does not
exist in the single-arm tool model the dual model shares (the old xeg32_* gripper / knife v1 links
kept in the single SRDF for the xeg32_knife_v1 variant) are dropped, and the dropped pairs are
printed. NO pair between the two arms is ever disabled (asserted), so arm-arm and arm-other-tool
collisions stay checked. The only extra pairs are base_link-table adjacency.

DUAL_REENABLED_PAIRS (below): intra-arm pairs that the single-arm SRDF disables as "Never" (computed with the
single-arm box/cylinder primitives) but that collide in the dual model, whose link collisions are convex hulls of the
real meshes (rebel_dual_table_description/meshes/collision). They are NOT copied (= they stay checked). Evidence:
scan_results/dual_arm_stack_20261014/p0b_hulls/ (empty-ACM sampling of 1.8 M random states).

Groups: arm_left / arm_right (chain world -> <prefix>xeg32_tool_tip, the TCP), both_arms, and per arm
a tool link group (the knife_c_* links of that arm) used by the end effector of the arm. The single-arm
tool group is read from the single SRDF's xeg32_gripper group, so a new tool link added there shows
up here too.
"""
import argparse
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", ".."))
DEFAULT_SINGLE = os.path.join(SRC, "igus_rebel_moveit_config", "config", "igus_rebel2.srdf")
DEFAULT_WRAPPER = os.path.join(SRC, "igus_rebel_description", "urdf", "igus_rebel_xeg32_wrapper.urdf.xacro")
DEFAULT_OUT = os.path.normpath(os.path.join(HERE, "..", "config", "rebel_dual_table.srdf"))
ARMS = ("arm_left_", "arm_right_")
# Pairs (unprefixed link names) of the single-arm SRDF that are NOT disabled in the dual SRDF, with the evidence.
# All six were "Never" in the single-arm SRDF; with the hull collision of link1-4 / motor1 they collide in
# 1.8 M random states of the full joint ranges (hits over 2 arms x 1.8 M arm samples, joint3 near its +139 deg limit,
# i.e. folded elbow; see p0b_hulls/logs/pair_report_empty_acm.txt). This list may only GROW the set of checked pairs.
DUAL_REENABLED_PAIRS = [
    ("knife_c_handpiece", "link2", "Never in single arm; hull hits: 502"),
    ("knife_c_holder", "link2", "Never in single arm; hull hits: 894"),
    ("knife_c_holder", "link4", "Never in single arm; hull hits: 393"),
    ("knife_c_wrist_camera", "link2", "Never in single arm; hull hits: 149"),
    ("link1", "motor4", "Never in single arm; hull hits: 11"),
    ("link3", "motor1", "Never in single arm; hull hits: 490"),
]
# Dual-only tool link (rebel_dual_table_description/urdf/rebel_tool.macro.xacro): the blade collision box. It is
# not part of the single-arm model, so it is added to the tool groups explicitly.
DUAL_ONLY_TOOL_LINKS = ["knife_c_blade"]
# Dual-only disabled pairs (unprefixed, same arm only). The blade box is rigid with igus_tool0, like the handpiece and
# holder, so a pair is either in contact at EVERY configuration or at none. Only the pair that is rigid contact is
# disabled: the box starts 1 mm inside the blade seat of the handpiece (proof: p0c_blade sampling, 200000 of 200000
# arm samples; test_blade_handpiece_overlap_is_rigid). blade-holder has 0 hits (the holder ends 62 mm behind the seat
# front), so it stays checked, as does the blade against every other link, the table and the other arm.
DUAL_ONLY_DISABLED_PAIRS = [("knife_c_blade", "knife_c_handpiece", "Adjacent")]
JOINTS = [f"joint{i}" for i in range(1, 7)]


def single_arm_links(urdf_or_xacro):
    """Link names of the single-arm model with the default (variant C) tool."""
    if urdf_or_xacro.endswith(".xacro"):
        text = subprocess.run(["xacro", urdf_or_xacro, "hardware_protocol:=mock_hardware"],
                              check=True, capture_output=True, text=True).stdout
    else:
        text = open(urdf_or_xacro).read()
    return {ln.get("name") for ln in ET.fromstring(text).findall("link")}


def parse_single(path):
    text = re.sub(r"<!--.*?-->", "", open(path).read(), flags=re.S)  # ignore commented-out pairs
    root = ET.fromstring(text)
    pairs = [(d.get("link1"), d.get("link2"), d.get("reason")) for d in root.findall("disable_collisions")]
    tool_links = [ln.get("name") for g in root.findall("group") if g.get("name") == "xeg32_gripper"
                  for ln in g.findall("link")]
    return pairs, tool_links


def generate(single, urdf):
    links = single_arm_links(urdf)
    pairs, tool_links_all = parse_single(single)
    tool_links = [t for t in tool_links_all if t in links] + DUAL_ONLY_TOOL_LINKS
    kept, dropped = [], []
    reenabled = {frozenset((a, b)) for a, b, _ in DUAL_REENABLED_PAIRS}
    single_pairs = {frozenset((l1, l2)): reason for l1, l2, reason in pairs}
    for key in reenabled:
        # a stale table entry must fail loudly: the pair has to be disabled in the single-arm SRDF
        assert key in single_pairs, f"DUAL_REENABLED_PAIRS names {sorted(key)}, which the single-arm SRDF does not disable"
        assert single_pairs[key] == "Never", f"{sorted(key)} is {single_pairs[key]}, only Never pairs may be re-enabled"
    for l1, l2, reason in pairs:
        if frozenset((l1, l2)) in reenabled:
            continue
        (kept if l1 in links and l2 in links else dropped).append((l1, l2, reason))
    L = []
    L.append('<?xml version="1.0" encoding="utf-8"?>')
    L.append('<!-- GENERATED by scripts/make_srdf.py from igus_rebel_moveit_config/config/igus_rebel2.srdf; '
             'do not hand-edit. -->')
    L.append('<robot name="rebel_dual_table">')
    L.append('    <!-- The URDF root is the fixed "world" link, so no virtual joint is declared. The tool')
    L.append('         joints are fixed, so no passive joints either. The arm groups end at the kinematic TCP')
    L.append('         (<prefix>xeg32_tool_tip), the pose reference for goals and IK. -->')
    for arm in ARMS:
        L.append(f'    <group name="{arm.rstrip("_")}">')
        L.append(f'        <chain base_link="world" tip_link="{arm}xeg32_tool_tip" />')
        L.append('    </group>')
    L.append('    <group name="both_arms">')
    for arm in ARMS:
        L.append(f'        <group name="{arm.rstrip("_")}" />')
    L.append('    </group>')
    for arm in ARMS:
        side = arm.split("_")[1]
        L.append(f'    <group name="knife_c_tool_{side}">')
        for ln in tool_links:
            L.append(f'        <link name="{arm}{ln}" />')
        L.append('    </group>')
    for g, joints in (("arm_left", [ARMS[0] + j for j in JOINTS]), ("arm_right", [ARMS[1] + j for j in JOINTS]),
                      ("both_arms", [x + j for x in ARMS for j in JOINTS])):
        L.append(f'    <group_state name="home" group="{g}">')
        for j in joints:
            L.append(f'        <joint name="{j}" value="0" />')
        L.append('    </group_state>')
    for arm in ARMS:
        side = arm.split("_")[1]
        L.append(f'    <end_effector name="knife_c_{side}" parent_link="{arm}xeg32_tool_tip" '
                 f'group="knife_c_tool_{side}" parent_group="{arm.rstrip("_")}" />')
    L.append('    <!-- Self-collision pairs per arm: every pair of the single-arm SRDF whose links exist in the')
    L.append('         variant-C model (see its comments for how the matrix was generated), EXCEPT the pairs of')
    L.append('         make_srdf.DUAL_REENABLED_PAIRS, which stay checked because the dual-arm hull collision geometry')
    L.append('         hits at some joint angles where the single-arm primitives never did:')
    for a, b, why in DUAL_REENABLED_PAIRS:
        L.append(f'         {a} - {b} ({why})')
    L.append('         Cross-arm pairs are deliberately NOT disabled. -->')
    for arm in ARMS:
        for l1, l2, reason in kept:
            L.append(f'    <disable_collisions link1="{arm}{l1}" link2="{arm}{l2}" reason="{reason}" />')
    L.append('    <!-- Dual-only blade box: disabled only against the handpiece it overlaps rigidly (see make_srdf.py);')
    L.append('         every other blade pair, in particular with the holder and with the other arm, stays checked. -->')
    for arm in ARMS:
        for l1, l2, reason in DUAL_ONLY_DISABLED_PAIRS:
            L.append(f'    <disable_collisions link1="{arm}{l1}" link2="{arm}{l2}" reason="{reason}" />')
    L.append('    <!-- Base / table adjacency: the base flange sits on the table (top surface 1 mm below z = 0). -->')
    for arm in ARMS:
        L.append(f'    <disable_collisions link1="{arm}base_link" link2="table" reason="Adjacent" />')
    L.append('</robot>')
    text = "\n".join(L) + "\n"
    # never disable a pair between the two arms
    for m in re.finditer(r'<disable_collisions link1="([^"]+)" link2="([^"]+)"', text):
        a, b = m.group(1), m.group(2)
        assert not ((a.startswith(ARMS[0]) and b.startswith(ARMS[1])) or
                    (a.startswith(ARMS[1]) and b.startswith(ARMS[0]))), f"cross-arm pair {a} {b}"
    return text, kept, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", default=DEFAULT_SINGLE)
    ap.add_argument("--urdf", default=DEFAULT_WRAPPER,
                    help="single-arm URDF or xacro whose links define which SRDF pairs are valid")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    text, kept, dropped = generate(a.single, a.urdf)
    if a.check:
        old = open(a.out).read() if os.path.exists(a.out) else ""
        if old != text:
            print(f"{a.out} is STALE: re-run scripts/make_srdf.py", file=sys.stderr)
            return 1
        print("rebel_dual_table.srdf is up to date")
        return 0
    with open(a.out, "w") as f:
        f.write(text)
    print(f"re-enabled (kept checked) pairs: {[(x, y) for x, y, _ in DUAL_REENABLED_PAIRS]}")
    print(f"wrote {a.out}: {len(kept)} pairs x {len(ARMS)} arms + {len(ARMS)} table pairs; "
          f"dropped {len(dropped)} single-arm pairs naming links absent from the variant-C model:")
    for l1, l2, r in dropped:
        print(f"  dropped {l1} {l2} ({r})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
