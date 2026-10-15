#!/usr/bin/env python3
"""Generate config/rebel_medicine_grasp.srdf (MoveIt - medicine grasp) from the expanded rebel_medicine_grasp URDF.

  source install/setup.bash
  python3 scripts/make_srdf.py --out config/rebel_medicine_grasp.srdf

ADAPTED from rebel_dual_arm_stand_moveit_config/scripts/make_srdf.py (same conservative policy, group structure and `home` state; derived from compal_robot, MIT, see ../NOTICE).
Disabled collision pairs are CONSERVATIVE (no sampling):
  * Adjacent : every pair of links joined by a joint (parent/child), as the MoveIt Setup Assistant does (this covers link6 <-> suction_tool, link6 <-> gripper_base,
               gripper_base <-> finger_left, gripper_base <-> finger_right);
  * Adjacent : finger_left <-> finger_right: the two jaw solids touch face to face and their convex collision hulls overlap in the closed/half-closed range
               (sim_assets/medical_box_tools README: "hulls overlap in x below q = 18.4 mm"); the mechanism is built that way, so the pair is not checkable;
  * Adjacent : all collision-bearing tool links of one arm among each other (rigid or mechanism-coupled on link6);
  * Never/Adjacent: the arm-internal pairs of OUR single-arm SRDF (igus_rebel_moveit_config/config/igus_rebel2.srdf, a Setup-Assistant sampling result for the same
               arm links), prefixed per arm; tool and rig links are NOT covered by it;
  * Adjacent : the head-camera parts among themselves (all fixed on `stand`, overlapping placeholder boxes).
  * Adjacent : d200_base_link (the AMR collision box) <-> stand and <-> arm_left_base_link / arm_right_base_link (stand mounted on the AMR; arm bases ride on the sliders far above it).
Deliberately ENABLED (as in the stand config): stand <-> slider_*, slider_left <-> slider_right, arm <-> arm, tool <-> other arm, tool <-> link5/link4 etc.,
and the fingers <-> link6 (the jaw plane is 150 mm in front of the flange; there is no overlap and no reason to disable). The tool frames (TCPs, pad frames, camera
frames) have no collision geometry. The suction pad buffers are fixed in this URDF: there is one rigid `suction_tool` link.
"""
import argparse
import itertools
import os
import subprocess
import xml.etree.ElementTree as ET

HOME_LIFT = -0.2
SLIDER_IN = 0.005973        # (0.459946 - 0.448) / 2 : arm spacing s = 0.448 m
GRIPPER_OPEN_Q = 0.024      # pad gap 8 + 2 q = 56 mm
GRIPPER_CLOSED_Q = 0.0      # pad gap 8 mm
ARM_LINKS = ['base_link', 'motor1', 'link1', 'motor2', 'link2', 'motor3', 'link3', 'motor4', 'link4', 'motor5', 'link5', 'link6']
SINGLE_ARM_SRDF = '/home/farhan/igus_rebel_ros2/src/igus_rebel_moveit_config/config/igus_rebel2.srdf'
LEFT_TCP, RIGHT_TCP = 'arm_left_suction_tcp', 'arm_right_gripper_tcp'
FINGER = 'arm_right_finger_left_joint'     # the actuated finger; arm_right_finger_right_joint is its URDF mimic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--single-arm-srdf', default=SINGLE_ARM_SRDF)
    a = ap.parse_args()
    from ament_index_python.packages import get_package_share_directory
    xacro_file = os.path.join(get_package_share_directory('rebel_medicine_grasp_description'), 'urdf', 'rebel_medicine_grasp.urdf.xacro')
    urdf = ET.fromstring(subprocess.run(['xacro', xacro_file, 'hardware_protocol:=mock_hardware'], check=True, capture_output=True, text=True).stdout)
    links = {l.get('name'): l for l in urdf.findall('link')}
    joints = urdf.findall('joint')
    has_col = {n for n, l in links.items() if l.findall('collision')}

    disabled = {}

    def dis(x, y, why):
        if x != y and x in links and y in links:
            disabled.setdefault(frozenset((x, y)), why)

    for j in joints:
        dis(j.find('parent').get('link'), j.find('child').get('link'), 'Adjacent')
    stand_parts = ['stand', 'camera_riser', 'camera_boom', 'gemini336L_camera_link']
    for x, y in itertools.combinations(stand_parts, 2):
        dis(x, y, 'Adjacent')
    # D200 AMR (one collision box on d200_base_link): the stand is bolted on it (parent/child joint above = Adjacent), and the two arm base links are the closest arm parts
    # (their own links are rigidly on the stand/sliders); everything else (lift, sliders, arm links, tools, cameras) stays ENABLED against the AMR.
    for arm in ('arm_left_', 'arm_right_'):
        dis('d200_base_link', arm + 'base_link', 'Adjacent')
    dis('d200_base_link', 'stand', 'Adjacent')
    tool_names = {}
    sroot = ET.parse(a.single_arm_srdf).getroot()
    for arm in ('arm_left_', 'arm_right_'):
        tool = sorted(n for n in has_col if n.startswith(arm) and n.split(arm, 1)[1] not in ARM_LINKS)
        tool_names[arm] = tool
        for x, y in itertools.combinations(tool, 2):
            dis(x, y, 'Adjacent')
        dis(arm + 'link6', arm + 'suction_tool', 'Adjacent')
        dis(arm + 'link6', arm + 'gripper_base', 'Adjacent')
        for dc in sroot.findall('disable_collisions'):
            x, y = dc.get('link1'), dc.get('link2')
            if x in ARM_LINKS and y in ARM_LINKS:
                dis(arm + x, arm + y, dc.get('reason'))

    srdf = ET.Element('robot', name='rebel_medicine_grasp')
    srdf.append(ET.Comment(
        ' MoveIt - medicine grasp. Generated by scripts/make_srdf.py (group structure and the rig Adjacent pairs follow rebel_dual_arm_stand_moveit_config, i.e. '
        'compal_robot dual_arm_rig.srdf @ bb76327, MIT, see NOTICE). Collision matrix is conservative: Adjacent/rigid pairs and our single-arm SRDF pairs only, '
        'no sampled "Never" pairs. '))
    g = ET.SubElement(srdf, 'group', name='gantry')
    for jn in ('lift_joint', 'slider_left_joint', 'slider_right_joint'):
        ET.SubElement(g, 'joint', name=jn)
    g = ET.SubElement(srdf, 'group', name='arm_left')
    ET.SubElement(g, 'chain', base_link='arm_left_base_link', tip_link=LEFT_TCP)
    g = ET.SubElement(srdf, 'group', name='arm_right')
    ET.SubElement(g, 'chain', base_link='arm_right_base_link', tip_link=RIGHT_TCP)
    g = ET.SubElement(srdf, 'group', name='dual_arm')
    for sub in ('gantry', 'arm_left', 'arm_right'):
        ET.SubElement(g, 'group', name=sub)
    g = ET.SubElement(srdf, 'group', name='right_gripper')
    ET.SubElement(g, 'joint', name=FINGER)
    g = ET.SubElement(srdf, 'group', name='left_suction_tool')
    ET.SubElement(g, 'link', name='arm_left_suction_tool')
    ET.SubElement(srdf, 'end_effector', name='left_suction', parent_link='arm_left_link6', group='left_suction_tool', parent_group='arm_left')
    ET.SubElement(srdf, 'end_effector', name='right_gripper_ee', parent_link='arm_right_link6', group='right_gripper', parent_group='arm_right')

    def state(name, group, vals):
        gs = ET.SubElement(srdf, 'group_state', name=name, group=group)
        for k, v in vals:
            ET.SubElement(gs, 'joint', name=k, value=repr(v) if isinstance(v, float) else str(v))
    gantry = [('lift_joint', HOME_LIFT), ('slider_left_joint', -SLIDER_IN), ('slider_right_joint', SLIDER_IN)]
    left = [('arm_left_joint%d' % i, 0.0) for i in range(1, 7)]
    right = [('arm_right_joint%d' % i, 0.0) for i in range(1, 7)]
    state('home', 'gantry', gantry)
    state('home', 'arm_left', left)
    state('home', 'arm_right', right)
    state('home', 'dual_arm', gantry + left + right)
    state('gripper_open', 'right_gripper', [(FINGER, GRIPPER_OPEN_Q)])        # pad gap 56 mm
    state('gripper_closed', 'right_gripper', [(FINGER, GRIPPER_CLOSED_Q)])    # pad gap 8 mm

    for key in sorted(disabled, key=lambda k: tuple(sorted(k))):
        x, y = sorted(key)
        ET.SubElement(srdf, 'disable_collisions', link1=x, link2=y, reason=disabled[key])
    ET.indent(srdf, space='  ')
    ET.ElementTree(srdf).write(a.out, xml_declaration=True, encoding='utf-8')
    print('wrote', a.out, 'disabled pairs:', len(disabled), 'collision links:', len(has_col))
    for arm, t in tool_names.items():
        print(arm, 'collision-bearing tool links:', t)


if __name__ == '__main__':
    main()
