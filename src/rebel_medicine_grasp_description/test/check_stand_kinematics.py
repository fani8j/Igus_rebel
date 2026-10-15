#!/usr/bin/env python3
"""The stand / lift / sliders / head camera / arm chains of rebel_medicine_grasp.urdf.xacro (mounted on the D200 AMR) must be IDENTICAL to rebel_dual_arm_stand_description's
rebel_dual_arm.urdf.xacro (the cutting rig): same links and joints (names, type, parent/child, origin, axis, limits, mimic), same inertials and the
same visual/collision geometry for every link that is not part of the tooling (= not below arm_*_link6), and the same forward kinematics
(pose of every such link RELATIVE TO THE `stand` LINK; the new URDF has the D200 chain above `stand`, the original has stand = world) for random values of the 15 stand/arm joints, the 2 limit corners, the mid-point and zero.

  * the tool links of the ORIGINAL (knife, XEG-32 gripper, cameras ...) must be absent from the new URDF;
  * the NEW URDF below arm_*_link6 contains exactly the medicine tools: tool frames are listed in check_tool_frames.py.

Needs both packages installed/sourced.   Run:  python3 test/check_stand_kinematics.py [--n 300]    or    python3 -m pytest -p no:anyio test/check_stand_kinematics.py"""
import argparse
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from urdf_util import Urdf, expand, limits, rot_err, vec  # noqa: E402


def find_xacros():
    from ament_index_python.packages import get_package_share_directory as g
    orig = os.path.join(g('rebel_dual_arm_stand_description'), 'urdf', 'rebel_dual_arm.urdf.xacro')
    new = os.path.join(g('rebel_medicine_grasp_description'), 'urdf', 'rebel_medicine_grasp.urdf.xacro')
    return orig, new


def canon(el):
    """canonical string of an XML subtree (sorted attributes, numbers re-formatted) for geometry/inertial comparison"""
    if el is None:
        return None
    def num(s):
        try:
            return ' '.join('%.9g' % float(v) for v in s.split())
        except ValueError:
            return s
    return (el.tag, tuple(sorted((k, num(v)) for k, v in el.attrib.items())), tuple(canon(c) for c in el))


def link_sig(l):
    return dict(inertial=canon(l.find('inertial')), visual=[canon(v) for v in l.findall('visual')], collision=[canon(c) for c in l.findall('collision')])


def run(n=300, seed=1):
    orig_x, new_x = find_xacros()
    O = Urdf(expand(orig_x, 'hardware_protocol:=mock_hardware'))
    N = Urdf(expand(new_x, 'hardware_protocol:=mock_hardware'))
    res = dict(errors=[])
    err = res['errors']

    tool_o = set()
    for arm in ('arm_left_', 'arm_right_'):
        tool_o |= O.descendants(arm + 'link6')
    tool_n = set()
    for arm in ('arm_left_', 'arm_right_'):
        tool_n |= N.descendants(arm + 'link6')
    # everything on the stand (the stand link and all its descendants) except the tooling; `world` / the D200 links are not part of the comparison
    core_o = ({'stand'} | O.descendants('stand')) - tool_o
    core_n = ({'stand'} | N.descendants('stand')) - tool_n
    if core_o != core_n:
        err.append('core link sets differ: only original %s, only new %s' % (sorted(core_o - core_n), sorted(core_n - core_o)))
    # the cutting tooling must be gone
    bad = [l for l in N.links if any(s in l for s in ('knife', 'xeg', 'cell_tool', 'tool_flange', 'wrist_camera_link', 'knife_camera'))]
    if bad:
        err.append('cutting-tool link(s) present in the new URDF: %s' % bad)
    only_tool_o = sorted(l for l in tool_o if l in N.links)
    if only_tool_o:
        err.append('original tool link names also present in the new URDF: %s' % only_tool_o)
    res['n_core_links'] = len(core_o & core_n)

    # joints between core links
    core_j_o = {k: j for k, j in O.joints.items() if j.find('child').get('link') in core_o and k != 'root_joint'}
    core_j_n = {k: j for k, j in N.joints.items() if j.find('child').get('link') in core_n and k != 'root_joint'}
    if set(core_j_o) != set(core_j_n):
        err.append('core joint sets differ: %s' % sorted(set(core_j_o) ^ set(core_j_n)))
    for k in sorted(set(core_j_o) & set(core_j_n)):
        a, b = core_j_o[k], core_j_n[k]
        for what, fa, fb in (('type', a.get('type'), b.get('type')),
                             ('parent', a.find('parent').get('link'), b.find('parent').get('link')),
                             ('child', a.find('child').get('link'), b.find('child').get('link')),
                             ('origin', canon(a.find('origin')), canon(b.find('origin'))),
                             ('axis', canon(a.find('axis')), canon(b.find('axis'))),
                             ('limit', limits(a), limits(b)), ('mimic', canon(a.find('mimic')), canon(b.find('mimic'))),
                             ('dynamics', canon(a.find('dynamics')), canon(b.find('dynamics')))):
            if fa != fb:
                err.append('joint %s %s differs: %s vs %s' % (k, what, fa, fb))
    res['n_core_joints'] = len(core_j_o)

    for k in sorted(core_o & core_n):
        if link_sig(O.links[k]) != link_sig(N.links[k]):
            err.append('link %s inertial/visual/collision differs' % k)

    # FK
    act = [k for k, j in core_j_o.items() if j.get('type') in ('revolute', 'prismatic', 'continuous')]
    res['n_actuated_core_joints'] = len(act)
    lo = np.array([limits(core_j_o[k])['lower'] for k in act])
    hi = np.array([limits(core_j_o[k])['upper'] for k in act])
    rng = np.random.default_rng(seed)
    qs = [rng.uniform(lo, hi) for _ in range(n)] + [lo, hi, (lo + hi) / 2, np.zeros(len(act))]
    maxp = maxr = 0.0
    TinvN = np.linalg.inv(N.fk('stand', {}))
    for qv in qs:
        q = dict(zip(act, qv))
        for l in sorted(core_o & core_n):
            A, B = O.fk(l, q), TinvN @ N.fk(l, q)     # both relative to the `stand` link
            maxp = max(maxp, float(np.linalg.norm(A[:3, 3] - B[:3, 3])))
            maxr = max(maxr, rot_err(A, B))
    res['fk_configs'] = len(qs)
    res['fk_max_pos_err_m'] = maxp
    res['fk_max_rot_err_rad'] = maxr
    if maxp > 1e-9 or maxr > 1e-9:
        err.append('FK differs: %g m / %g rad' % (maxp, maxr))

    # the D200 mount: stand frame in d200_base_footprint = Rz(+90 deg) * translation (0.1668696, 0, 0.2645)  (USD upper_fixed_joint of d200_with_dual_arm.usd)
    Ts = N.fk('stand', {})
    exp = np.array([[0, -1, 0, 0.1668696], [1, 0, 0, 0], [0, 0, 1, 0.2645], [0, 0, 0, 1.0]])
    # N.fk is relative to the URDF root `world`; world -> d200_base_footprint is identity
    res['stand_in_footprint_err'] = float(np.abs(Ts - exp).max())
    if res['stand_in_footprint_err'] > 1e-9:
        err.append('stand mount differs from the USD: %s' % Ts.round(6).tolist())
    if N.joints['root_joint'].find('parent').get('link') != 'd200_base_link' or N.joints['d200_base_footprint_joint'].find('parent').get('link') != 'world':
        err.append('AMR joint chain')
    amr_col = N.links['d200_base_link'].findall('collision')
    if len(amr_col) != 1 or amr_col[0].find('geometry/box') is None:
        err.append('AMR collision must be one box (not the STL)')

    # mimic of the new gripper
    mj = [k for k, j in N.joints.items() if j.find('mimic') is not None]
    res['mimic_joints'] = mj
    if mj != ['arm_right_finger_right_joint']:
        err.append('unexpected mimic joints %s' % mj)
    return res


def test_stand_kinematics():
    r = run(100)
    assert not r['errors'], r['errors']


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=300)
    a = ap.parse_args()
    r = run(a.n)
    import json
    print(json.dumps(r, indent=1))
    sys.exit(1 if r['errors'] else 0)
