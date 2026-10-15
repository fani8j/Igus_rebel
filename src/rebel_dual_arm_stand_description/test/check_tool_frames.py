#!/usr/bin/env python3
"""The tool frames in the expanded dual-arm URDF must reproduce the design numbers: ${arm}knife_tip == T_link6_tcp, the camera optical frames ==
T_link6_camera / T_link6_camB, for both arms (tolerance 1e-5 m / 1e-5 rad: the design matrices are rounded to 6 decimals).  Also checks that the
TCP and the blade have no collision geometry.  Needs the package to be installed / sourced (xacro $(find ...)).  Run: python3 test/check_tool_frames.py"""
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from check_arm_fk import Urdf, T_origin, DUAL_XACRO, PKG_DIR  # noqa: E402

DESIGN = os.path.join(PKG_DIR, '..', '..', 'tools', 'cell_planner_check', 'designs', 'v3_r_1269_A0_B0.json')


def pose_in_link6(urdf, link6, frame):
    T = np.eye(4)
    name = frame
    chain = []
    while name != link6:
        j = [j for j in urdf.joints.values() if j.find('child').get('link') == name][0]
        chain.append(j)
        name = j.find('parent').get('link')
    for j in reversed(chain):
        T = T @ T_origin(j)
    return T


def rot_err(A, B):
    R = A[:3, :3].T @ B[:3, :3]
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2.0
    return float(np.arctan2(np.linalg.norm(v), (np.trace(R) - 1) / 2))


def run():
    out = subprocess.run(['xacro', DUAL_XACRO, 'hardware_protocol:=mock_hardware'], check=True, capture_output=True, text=True).stdout
    u = Urdf(ET.fromstring(out))
    cfg = yaml.safe_load(open(os.path.join(PKG_DIR, 'config', 'cell_tool_v3_r_1269.yaml')))
    exp = {'knife_tip': np.array(cfg['T_link6_tcp']),
           'wrist_camera_optical_frame': np.array(cfg['cameras']['wrist_camera']['T_link6_optical']),
           'knife_camera_optical_frame': np.array(cfg['cameras']['knife_camera']['T_link6_optical'])}
    if os.path.exists(DESIGN):
        d = json.load(open(DESIGN))
        assert np.abs(np.array(d['T_link6_tcp']) - exp['knife_tip']).max() == 0
        assert np.abs(np.array(d['camera']['T_link6_camera']) - exp['wrist_camera_optical_frame']).max() == 0
        assert np.abs(np.array(d['camera_B']['T_link6_camera']) - exp['knife_camera_optical_frame']).max() == 0
    res = {}
    for arm in ('arm_left_', 'arm_right_'):
        for fr, T in exp.items():
            got = pose_in_link6(u, arm + 'link6', arm + fr)
            res[arm + fr] = dict(pos_err_m=float(np.linalg.norm(got[:3, 3] - T[:3, 3])), rot_err_rad=rot_err(got, T))
        for nocol in ('knife_tip', 'knife_blade'):
            assert not u.links[arm + nocol].findall('collision'), arm + nocol + ' must have no collision geometry'
    return res


def test_tool_frames():
    res = run()
    for k, v in res.items():
        assert v['pos_err_m'] < 1e-5 and v['rot_err_rad'] < 1e-5, (k, v)


if __name__ == '__main__':
    r = run()
    print(json.dumps(r, indent=1))
    sys.exit(0 if all(v['pos_err_m'] < 1e-5 and v['rot_err_rad'] < 1e-5 for v in r.values()) else 1)
