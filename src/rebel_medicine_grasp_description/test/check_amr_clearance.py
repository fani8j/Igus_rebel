#!/usr/bin/env python3
"""How close can the robot get to the D200 AMR?  Random joint states with the lift at its lower limit (-0.6235 m), left and right arm bent down/forward (plain-numpy FK on
the expanded URDF): the lowest TCP height / the lowest origin of any non-AMR link is compared with the AMR top (0.2515 m).  Result: the arms cannot reach the AMR deck, so the
only parts that can touch the AMR box are the stand-mounted ones (excluded by the SRDF as Adjacent) - the check is a regression test for a changed mount height / tool length.
Run: python3 test/check_amr_clearance.py [--n 4000]    or    pytest -p no:anyio"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from urdf_util import Urdf, expand  # noqa: E402

AMR_TOP = 0.2515


def run(n=4000, seed=1):
    from ament_index_python.packages import get_package_share_directory as g
    U = Urdf(expand(os.path.join(g('rebel_medicine_grasp_description'), 'urdf', 'rebel_medicine_grasp.urdf.xacro')))
    rng = np.random.default_rng(seed)
    out = {}
    for arm, tcp in (('arm_left_', 'arm_left_suction_tcp'), ('arm_right_', 'arm_right_gripper_tcp')):
        low_tcp = low_link = 9.0
        for _ in range(n):
            q = {'lift_joint': -0.6235, arm + 'joint1': rng.uniform(-1.2, 1.2), arm + 'joint2': rng.uniform(0.3, 1.4), arm + 'joint3': rng.uniform(0.0, 2.4),
                 arm + 'joint4': 0.0, arm + 'joint5': rng.uniform(-1.4, 1.4), arm + 'joint6': 0.0}
            low_tcp = min(low_tcp, U.fk(tcp, q)[2, 3])
            low_link = min(low_link, min(U.fk(arm + l, q)[2, 3] for l in ('link2', 'link3', 'link5', 'link6')))
        out[arm] = dict(lowest_tcp_z=round(low_tcp, 3), lowest_link_origin_z=round(low_link, 3))
    out['amr_top_z'] = AMR_TOP
    out['n'] = n
    return out


def test_amr_clearance():
    r = run(300)
    for arm in ('arm_left_', 'arm_right_'):
        assert r[arm]['lowest_tcp_z'] > AMR_TOP + 0.05, r


if __name__ == '__main__':
    n = int(sys.argv[sys.argv.index('--n') + 1]) if '--n' in sys.argv else 4000
    print(json.dumps(run(n), indent=1))
