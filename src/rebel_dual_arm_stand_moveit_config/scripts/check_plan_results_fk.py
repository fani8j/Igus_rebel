#!/usr/bin/env python3
"""Independent check of plan_results.json (verify_plan_only.py): for every successful pose-goal plan, compute the FINAL knife_tip pose from the
planned joint values with plain-numpy FK on the expanded URDF and compare it with the requested target (position error, tool-z direction).
Written new for this package.   python3 check_plan_results_fk.py plan_results.json [out.json]"""
import json
import math
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
from ament_index_python.packages import get_package_share_directory


def vec(s):
    return np.array([float(v) for v in s.split()])


def rpy_R(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr], [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], [-sp, cp * sr, cp * cr]])


def main():
    res = json.load(open(sys.argv[1]))
    xf = get_package_share_directory('rebel_dual_arm_stand_description') + '/urdf/rebel_dual_arm.urdf.xacro'
    root = ET.fromstring(subprocess.run(['xacro', xf, 'hardware_protocol:=mock_hardware'], check=True, capture_output=True, text=True).stdout)
    by_child = {j.find('child').get('link'): j for j in root.findall('joint')}

    def tip_pose(link, q):
        chain = []
        while link in by_child:
            chain.append(by_child[link])
            link = by_child[link].find('parent').get('link')
        T = np.eye(4)
        for j in reversed(chain):
            o = j.find('origin')
            To = np.eye(4)
            if o is not None:
                To[:3, :3] = rpy_R(*vec(o.get('rpy', '0 0 0')))
                To[:3, 3] = vec(o.get('xyz', '0 0 0'))
            T = T @ To
            if j.get('type') in ('revolute', 'prismatic'):
                v = q.get(j.get('name'), 0.0)
                ax = vec(j.find('axis').get('xyz'))
                Tj = np.eye(4)
                if j.get('type') == 'prismatic':
                    Tj[:3, 3] = ax * v
                else:
                    a = ax / np.linalg.norm(ax)
                    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
                    Tj[:3, :3] = np.eye(3) + math.sin(v) * K + (1 - math.cos(v)) * (K @ K)
                T = T @ Tj
        return T

    home = res['home_joint_state']
    out = []
    for arm, r in res['arms'].items():
        for e in r['pose_goals']:
            for key in ('plan_pose_goal', 'plan_to_ik_solution'):
                p = e.get(key)
                if not p or p.get('error_code') != 1:
                    continue
                q = dict(home)
                q.update(p['final'])
                T = tip_pose(arm + '_knife_tip', q)
                tgt = np.array(e['target_xyz'])
                z = T[:3, 2]
                out.append(dict(arm=arm, roll_deg=e['roll_deg'], via=key, pos_err_m=float(np.linalg.norm(T[:3, 3] - tgt)),
                                tool_z_world=[round(float(v), 4) for v in z], tool_z_angle_to_down_deg=round(math.degrees(math.acos(max(-1, min(1, -z[2])))), 3)))
    for o in out:
        print('%-9s roll %5.0f %-18s pos_err %.2e m  tool_z %s  (%.3f deg from straight down)' % (o['arm'], o['roll_deg'], o['via'], o['pos_err_m'], o['tool_z_world'], o['tool_z_angle_to_down_deg']))
    if len(sys.argv) > 2:
        json.dump(out, open(sys.argv[2], 'w'), indent=1)


if __name__ == '__main__':
    main()
