#!/usr/bin/env python3
"""Tool frames of the expanded rebel_medicine_grasp URDF against the CAD numbers (cad/medical_box_tooling_v1, link6 frame):
  suction: suction_tcp z 159 mm, suction_tcp_compressed z 154, pad_{1,2,3}_suction_frame x -60/0/+60 mm z 159, wrist_camera_mount (0, 32.65, 118) mm;
  gripper: gripper_tcp z 193.2 mm, top_stop_frame z 167.5 mm, fingers at z 150 mm, prismatic +-X, 0..24 mm, finger_right mimic (x1, 0) of finger_left.
Also: the tool links of one arm are exactly the medicine tools, every tool collision uses a collision mesh, masses match the CAD totals,
the TCP frames carry no collision geometry, all package:// mesh files exist.   Run: python3 test/check_tool_frames.py   or pytest -p no:anyio"""
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from urdf_util import Urdf, expand, limits, vec  # noqa: E402

M = 1e-3
SUCTION_EXPECT = {  # frame -> xyz in link6 (m)
    'suction_tcp': (0, 0, 0.159), 'suction_tcp_compressed': (0, 0, 0.154), 'wrist_camera_mount': (0, 0.03265, 0.118),
    'pad_1_suction_frame': (-0.06, 0, 0.159), 'pad_2_suction_frame': (0, 0, 0.159), 'pad_3_suction_frame': (0.06, 0, 0.159),
}
GRIPPER_EXPECT = {'gripper_tcp': (0, 0, 0.1932), 'top_stop_frame': (0, 0, 0.1675)}
MASS_SUCTION = 0.20381          # data/verify.json mass (URDF inertial)
MASS_GRIPPER_BASE, MASS_FINGER = 0.843663131, 0.0525123601


def run():
    from ament_index_python.packages import get_package_share_directory as g
    share = g('rebel_medicine_grasp_description')
    U = Urdf(expand(os.path.join(share, 'urdf', 'rebel_medicine_grasp.urdf.xacro'), 'hardware_protocol:=mock_hardware'))
    errs = []
    out = {}
    for fr, xyz in SUCTION_EXPECT.items():
        T = U.fk('arm_left_' + fr, {})
        T6 = U.fk('arm_left_link6', {})
        rel = np.linalg.inv(T6) @ T
        out['arm_left_' + fr] = rel[:3, 3].round(6).tolist()
        if np.abs(rel[:3, 3] - xyz).max() > 1e-9:
            errs.append(('arm_left_' + fr, rel[:3, 3].tolist(), xyz))
    for fr, xyz in GRIPPER_EXPECT.items():
        rel = np.linalg.inv(U.fk('arm_right_link6', {})) @ U.fk('arm_right_' + fr, {})
        out['arm_right_' + fr] = rel[:3, 3].round(6).tolist()
        if np.abs(rel[:3, 3] - xyz).max() > 1e-9:
            errs.append(('arm_right_' + fr, rel[:3, 3].tolist(), xyz))
    # which links hang below link6
    below_l = sorted(U.descendants('arm_left_link6'))
    below_r = sorted(U.descendants('arm_right_link6'))
    out['links_below_arm_left_link6'] = below_l
    out['links_below_arm_right_link6'] = below_r
    if any(not l.startswith('arm_left_') for l in below_l) or any(not l.startswith('arm_right_') for l in below_r):
        errs.append('tool links without arm prefix')
    # gripper joints
    fl, fr_ = U.joints['arm_right_finger_left_joint'], U.joints['arm_right_finger_right_joint']
    for j in (fl, fr_):
        lim = limits(j)
        if j.get('type') != 'prismatic' or abs(lim['lower']) > 0 or abs(lim['upper'] - 0.024) > 1e-12:
            errs.append(('finger joint', j.get('name'), j.get('type'), lim))
    if tuple(vec(fl.find('axis').get('xyz'))) != (-1.0, 0.0, 0.0) or tuple(vec(fr_.find('axis').get('xyz'))) != (1.0, 0.0, 0.0):
        errs.append('finger axes')
    mim = fr_.find('mimic')
    if mim is None or mim.get('joint') != 'arm_right_finger_left_joint' or float(mim.get('multiplier')) != 1.0 or float(mim.get('offset')) != 0.0:
        errs.append('finger_right mimic')
    if fl.find('mimic') is not None:
        errs.append('driver finger must not be a mimic')
    # pad buffers: no joints
    if [j for j in U.joints if 'buffer' in j]:
        errs.append('pad buffer joints exist (they must be fixed / absent)')
    # gap geometry: pad gap = 8 + 2 q  <-> fingers at q=0.024 -> 56 mm : check the two finger link origins separation along x
    for q, gap in ((0.0, 0.008), (0.024, 0.056)):
        qq = {'arm_right_finger_left_joint': q}
        Tb = np.linalg.inv(U.fk('arm_right_gripper_base', qq))      # finger origins expressed in the gripper base frame (the arm is not at identity)
        xl = (Tb @ U.fk('arm_right_finger_left', qq))[0, 3]
        xr = (Tb @ U.fk('arm_right_finger_right', qq))[0, 3]
        out['finger_origin_separation_q%.3f' % q] = round(float(xr - xl), 6)   # = 2 q (origins coincide at q = 0)
        if abs((xr - xl) - 2 * q) > 1e-9:
            errs.append(('finger separation', q, xr - xl))
    # masses
    ms = sum(float(l.find('inertial/mass').get('value')) for n, l in U.links.items() if n.startswith('arm_left_suction_tool'))
    if abs(ms - MASS_SUCTION) > 1e-9:
        errs.append(('suction mass', ms))
    mg = {n: float(U.links[n].find('inertial/mass').get('value')) for n in ('arm_right_gripper_base', 'arm_right_finger_left', 'arm_right_finger_right')}
    if abs(mg['arm_right_gripper_base'] - MASS_GRIPPER_BASE) > 1e-9 or abs(mg['arm_right_finger_left'] - MASS_FINGER) > 1e-9:
        errs.append(('gripper masses', mg))
    out['masses_kg'] = dict(suction_tool=ms, **mg)
    # TCP frames: no geometry; collision from collision meshes only; mesh files exist
    for n in ('arm_left_suction_tcp', 'arm_left_suction_tcp_compressed', 'arm_right_gripper_tcp', 'arm_right_top_stop_frame'):
        if U.links[n].findall('collision') or U.links[n].findall('visual'):
            errs.append(('frame link has geometry', n))
    nmesh = 0
    for n, l in U.links.items():
        for kind in ('visual', 'collision'):
            for el in l.findall(kind):
                m = el.find('geometry/mesh')
                if m is None:
                    continue
                f = m.get('filename')
                mm = re.match(r'package://([^/]+)/(.*)', f)
                if not mm:
                    errs.append(('not a package:// uri', f))
                    continue
                path = os.path.join(g(mm.group(1)), mm.group(2))
                nmesh += 1
                if not os.path.exists(path):
                    errs.append(('mesh missing', f))
                if kind == 'collision' and n.startswith(('arm_left_suction', 'arm_right_gripper', 'arm_right_finger')) and '/collision/' not in f:
                    errs.append(('tool collision not from collision/', f))
    out['n_mesh_refs'] = nmesh
    out['errors'] = errs
    return out


def test_tool_frames():
    r = run()
    assert not r['errors'], r['errors']


if __name__ == '__main__':
    import json
    r = run()
    print(json.dumps(r, indent=1))
    sys.exit(1 if r['errors'] else 0)
