#!/usr/bin/env python3
"""v3 adapter: design JSON (schema cell_planner_check/design/v1 + 'camera_B') from a cell_v3_20261004 recommended.json entry.

  python3 make_design_v3.py /home/farhan/igus_rebel_ros2/scan_results/cell_v3_20261004/recommended.json [rank ...]   -> designs/v3_<label>.json

The entry carries holder params (dalib.make_design parametrisation), camera A (Gemini 305 on the gripper / flange adapter, T_link6_camA + bracket) and camera B (small USB camera on
the holder clamp / post, T_link6_camB + bracket), all in link6.  Fingers = the real finger boxes of the earlier studies (copied from designs/v2_E5.json).  New file; the existing
harness is not modified.
"""
import json, os, sys
sys.dont_write_bytecode = True
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
FLANGE_Z = 0.0527


def design_from_entry(e, template=None):
    tpl = json.load(open(os.path.join(HERE, 'designs', 'v2_E5.json'))) if template is None else template
    p = e['holder']['params']
    T = np.array(e['holder']['T_link6_tcp'])
    M = np.array(e['holder']['clamp_point_M_m']); a = np.array(e['holder']['knife_axis_link6'])
    L = p['L']
    end = M + max(L - 0.04, 0.0) * a
    des = dict(schema='cell_planner_check/design/v1', name='v3_' + e['label'], source='cell_v3_20261004/recommended.json rank %s' % e.get('rank'), params=p, T_link6_tcp=T.tolist(),
               layout=e.get('layout'), gripper=tpl['gripper'], fingers=tpl['fingers'])
    post = None
    if M[2] > FLANGE_Z + 0.01:
        post = dict(frm=[float(M[0]), float(M[1]), FLANGE_Z], to=M.tolist(), radius=0.012)
    des['knife'] = dict(clamp=dict(centre=M.tolist(), radius=0.025), post=post, handpiece=dict(frm=M.tolist(), to=end.tolist(), radius=0.02),
                        blade=dict(frm=end.tolist(), to=T[:3, 3].tolist(), exempt_last_m=0.02, collision=False))
    cA, cB = e['camera_A'], e['camera_B']
    des['camera'] = dict(T_link6_camera=cA['T_link6_camA'], body_size=cA['body_size_m'], pad=0.003,
                         bracket=dict(frm=cA['bracket']['from_m'], to=cA['bracket']['to_m'], radius=0.008))
    des['camera_B'] = dict(T_link6_camera=cB['T_link6_camB'], body_size=cB['body_size_m'], pad=0.003,
                           bracket=dict(frm=cB['bracket']['from_m'], to=cB['bracket']['to_m'], radius=0.008), lens=cB.get('lens'), f0_m=cB.get('focus_distance_m'))
    return des


def main():
    rec = json.load(open(sys.argv[1]))
    ranks = [int(x) for x in sys.argv[2:]] or [c['rank'] for c in rec['combos']]
    os.makedirs(os.path.join(HERE, 'designs'), exist_ok=True)
    for c in rec['combos']:
        if c['rank'] in ranks:
            d = design_from_entry(c)
            json.dump(d, open(os.path.join(HERE, 'designs', d['name'] + '.json'), 'w'), indent=1)
            print('wrote designs/%s.json' % d['name'])


if __name__ == '__main__':
    main()
