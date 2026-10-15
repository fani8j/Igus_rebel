#!/usr/bin/env python3
"""config/medicine_shelf_pose.json must equal what add_medicine_shelf.py derives from config/medicine_shelf_scene.yaml (the JSON is the hand-over to the Isaac scene),
the shelf / plinth / cartons keep >= 50 mm from the AMR box, the cartons are packed with zero gap and the SRDF has the AMR + gripper entries.   pytest -p no:anyio test/check_scene_pose.py"""
import json
import os
import sys
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PKG, 'scripts'))
import add_medicine_shelf as s  # noqa: E402

YAML = os.path.join(PKG, 'config', 'medicine_shelf_scene.yaml')


def test_pose_json_matches_yaml():
    assert json.load(open(os.path.join(PKG, 'config', 'medicine_shelf_pose.json'))) == json.loads(json.dumps(s.pose_json(YAML)))


def test_amr_clearance():
    for pull in (0.0, 0.1):
        for k, v in s.amr_clearance(YAML, pull).items():
            assert v >= 0.05, (pull, k, v)


def test_carton_packing():
    import yaml
    L = s.carton_layout(yaml.safe_load(open(YAML)))
    for a, b in zip(L, L[1:]):
        assert abs((b[2] - b[3] / 2) - (a[2] + a[3] / 2)) < 1e-9


def test_srdf():
    r = ET.parse(os.path.join(PKG, 'config', 'rebel_medicine_grasp.srdf')).getroot()
    groups = {g.get('name') for g in r.findall('group')}
    assert {'gantry', 'arm_left', 'arm_right', 'dual_arm', 'right_gripper'} <= groups
    states = {(g.get('group'), g.get('name')) for g in r.findall('group_state')}
    assert ('right_gripper', 'gripper_open') in states and ('right_gripper', 'gripper_closed') in states and ('dual_arm', 'home') in states
    pairs = {frozenset((d.get('link1'), d.get('link2'))) for d in r.findall('disable_collisions')}
    assert frozenset(('d200_base_link', 'stand')) in pairs and frozenset(('d200_base_link', 'arm_left_base_link')) in pairs
    assert not any('d200_base_link' in p and ('arm_left_link6' in p or 'arm_left_suction_tool' in p) for p in pairs)
    ee = {e.get('name') for e in r.findall('end_effector')}
    assert ee == {'left_suction', 'right_gripper_ee'}
