#!/usr/bin/env python3
"""v5 adapter: design JSON (schema cell_planner_check/design/v1 + 'v5_cad') from a cell_v5_20261007 recommended.json entry and the CAD v3 collision-mesh directory.

  python3 make_design_v5.py <recommended.json> <rank> <collision_dir> -> designs/v5_<label>.json

The tooling collision geometry is NOT built from primitives here: gen_cell_v5.py replaces the gripper / fingers / clamp / handpiece / camera links by the exact CAD v3 collision meshes
(`<collision_dir>/manifest.json`, metres, link6 frame).  The rest of the entry (TCP, camera A pose for the harness' camera link, park, fixtures) is copied.  New file; the harness is unchanged."""
import json, os, sys
sys.dont_write_bytecode = True
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def design_from_entry(e, collision_dir):
    tpl = json.load(open(os.path.join(HERE, 'designs', 'v2_E5.json')))
    T = np.array(e['holder']['T_link6_tcp'])
    M = np.array(e['holder']['clamp_point_M_m']); a = np.array(e['holder']['knife_axis_link6'])
    cA = e['camera_A']
    des = dict(schema='cell_planner_check/design/v1', name='v5_' + e['label'], source='cell_v5_20261007/recommended.json rank %s' % e.get('rank'), params=e['holder']['params'], T_link6_tcp=T.tolist(),
               layout=e.get('layout'), gripper=tpl['gripper'], fingers=tpl['fingers'])
    des['knife'] = dict(clamp=dict(centre=M.tolist(), radius=0.02), post=None, handpiece=dict(frm=M.tolist(), to=(M + 0.05 * a).tolist(), radius=0.01),
                        blade=dict(frm=T[:3, 3].tolist(), to=T[:3, 3].tolist(), exempt_last_m=0.02, collision=False))
    des['camera'] = dict(T_link6_camera=cA['T_link6_cam'], body_size=cA['body_size_m'], pad=0.0, bracket=dict(frm=cA['bracket']['from_m'], to=cA['bracket']['to_m'], radius=0.001))
    des['v5_cad'] = dict(collision_dir=os.path.abspath(collision_dir), manifest=os.path.join(os.path.abspath(collision_dir), 'manifest.json'))
    return des


def main():
    rec = json.load(open(sys.argv[1])); rank = int(sys.argv[2]); cdir = sys.argv[3]
    os.makedirs(os.path.join(HERE, 'designs'), exist_ok=True)
    for c in rec['combos']:
        if c['rank'] == rank:
            d = design_from_entry(c, cdir)
            json.dump(d, open(os.path.join(HERE, 'designs', d['name'] + '.json'), 'w'), indent=1)
            print('wrote designs/%s.json' % d['name'])


if __name__ == '__main__':
    main()
