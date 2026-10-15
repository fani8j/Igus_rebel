#!/usr/bin/env python3
"""Write design JSONs (tool geometry in the link6 frame, metres) from the offline study outputs.

  designs/<label>.json   for every combo in holder_camera_fingers_20261002/recommended.json (h000n.0 first)
  designs/baseline.json  old single knife (current TCP, no fingers, no camera), proxies exactly as dalib.baseline_design()

Schema cell_planner_check/design/v1.  Read-only use of the study directories.
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = '/home/farhan/igus_rebel_ros2'
FING = REPO + '/scan_results/holder_camera_fingers_20261002'
SMC_STL = '/home/farhan/Downloads/SMC_JP_LEHF20K24800 (2)/LEHF20K2-48(0_0).stl'
FINGER_STL_CANDIDATES = [
    HERE + '/assets/finger_full.stl',
    '/tmp/claude-1000/-home-farhan-igus-rebel-ros2/35870b10-b923-4400-954f-bbd47a4db61d/scratchpad/finger/finger_full.stl']
GRIP_BOX = [[-0.094, 0.064], [-0.0415, 0.0415], [0.0527, 0.0877]]


def proxy_block(des_study, cam=None):
    """sphere / box proxy geometry of the offline study (dalib.all_spheres), expressed in rigid link frames so gen_cell.py --geometry proxy can rebuild the
    study's collision model inside MoveIt:  upper-arm spheres in the link2 frame, forearm spheres in link3, wrist/flange/gripper/tool/finger/bracket
    spheres in link6.  (study: UPPER_R 60 mm, FORE_R 55 mm, wrist chain, 18-sphere gripper cover, 120-sphere finger cover, capsule chains.)"""
    sys.path.insert(0, FING)
    import dalib as D
    import fingerlib as FL
    import reach_lib as R
    # frames of link2 / link3 at q = 0 (all joint rotations vanish; only the fixed offsets and the J_motor1 yaw remain)
    T = np.eye(4)
    frames = {}
    for j in R.CHAIN:
        M = np.eye(4)
        M[:3, :3] = R.Rz(j['rpy'][2]) if abs(j['rpy'][2]) > 0 else np.eye(3)
        M[:3, 3] = j['xyz']
        T = T @ M
        if j['type'] == 'revolute':
            frames[j['name']] = T.copy()
    T2, T3 = frames['joint2'], frames['joint3']
    j2 = np.array([0, 0, D.P_J2_Z])
    j3 = j2 + np.array([0, 0, D.L2])
    T6 = R.fk_flange(np.zeros((1, 6)))[0]
    P5 = T6[:3, 3] - D.L56 * T6[:3, 2]
    inv = lambda Tm, p: (np.linalg.inv(Tm) @ np.append(p, 1.0))[:3]
    upper = [inv(T2, j2 + (j3 - j2) * f).tolist() for f in (0.0, 0.35, 0.7, 1.0)]
    fore = [inv(T3, j3 + (P5 - j3) * f).tolist() for f in (0.25, 0.5, 0.75)]
    sp = des_study.tool_spheres(0.0)
    out = dict(upper=dict(centres=upper, r=D.UPPER_R), fore=dict(centres=fore, r=D.FORE_R),
               wrist=dict(centres=[[0, 0, z] for z, r in D.WRIST_Z + D.FLANGE_SPH], radii=[r for z, r in D.WRIST_Z + D.FLANGE_SPH]),
               gripper=dict(centres=D.GRIP_SPH.tolist(), r=D.GRIP_SPH_R),
               hp=dict(centres=np.asarray(sp['hp']).tolist(), r=D.HP_R), clamp=dict(centres=np.asarray(sp['clamp']).tolist(), r=D.CLAMP_R),
               post=dict(centres=np.asarray(sp['post']).tolist(), r=D.POST_R),
               camera_pad=0.003, camera_half_extra=0.003)
    fg = getattr(des_study, 'fing', None)
    if fg is not None:
        out['fingers'] = dict(centres=np.asarray(fg[0]).tolist(), radii=np.asarray(fg[1]).tolist())
    cam = getattr(des_study, 'cam', None)
    if cam is not None:
        out['bracket'] = dict(centres=np.asarray(cam.bracket).tolist(), r=0.008)
    return out


def combo_to_design(c, fingers):
    h = c['holder']
    T = h['T_link6_tcp']
    des = dict(
        schema='cell_planner_check/design/v1',
        name=c['label'],
        source='holder_camera_fingers_20261002/recommended.json',
        params=h['params'],
        T_link6_tcp=T,
        gripper=dict(stl=SMC_STL, box=GRIP_BOX),
        fingers=dict(mount=fingers['mount'], boxes_grasp=fingers['boxes_grasp'], boxes_open=fingers['boxes_open'],
                     stl=next((p for p in FINGER_STL_CANDIDATES if os.path.exists(p)), None)),
        knife=dict(
            clamp=dict(centre=c['clamp_block']['centre_m'], radius=c['clamp_block']['radius_m']),
            post=dict(frm=c['post']['from_xyz_m'], to=c['post']['to_xyz_m'], radius=c['post']['radius_m']),
            handpiece=dict(frm=c['handpiece']['from_xyz_m'], to=c['handpiece']['to_xyz_m'], radius=c['handpiece']['radius_m']),
            blade=dict(frm=c['handpiece']['to_xyz_m'], to=c['tcp_xyz_m'], exempt_last_m=0.02, collision=False)),
        camera=dict(T_link6_camera=c['T_link6_camera'], body_size=c['camera_body_size_m'], body_centre=c['camera_body_centre_m'],
                    pad=c['camera_pad_m'],
                    bracket=dict(frm=c['bracket']['segment_from_camera_back_m'], to=c['bracket']['segment_to_mount_m'], radius=c['bracket']['radius_m'])),
        study_scores=c.get('scores'))
    sys.path.insert(0, FING)
    try:
        from common import des_from_params
        import camlib
        ds = des_from_params(h['params'], fingers=True)
        combos = json.load(open(FING + '/combos_top.json'))
        it = next((x for x in combos if x['label'] == c['label']), None)
        if it is not None:
            ds.cam = camlib.Cam(np.asarray(it['o']), np.asarray(it['Rc']), np.asarray(it['brp']))
        sys.path.insert(0, FING)
        des['proxy'] = proxy_block(ds)
    except Exception as e:                       # proxy block is optional
        print('proxy block skipped for', c['label'], repr(e))
    return des


def v2_designs(rec_path=None):
    """designs from the v2 study (cell_v2_20261003/recommended.json): same schema; fingers block = the v1 recommended fingers (same finger model);
    the study sphere-proxy block is rebuilt with the v1 modules (identical dalib semantics) from the v2 finals.json camera bracket points."""
    V2 = REPO + '/scan_results/cell_v2_20261003'
    rec_path = rec_path or V2 + '/recommended.json'
    R = json.load(open(rec_path))
    fingers = json.load(open(FING + '/recommended.json'))['fingers']
    finals = {x['label']: x for x in json.load(open(V2 + '/finals.json'))} if os.path.exists(V2 + '/finals.json') else {}
    out = []
    for c in R['combos']:
        h = c['holder']
        cam = c['camera']
        des = dict(
            schema='cell_planner_check/design/v1', name='v2_' + c['label'], source='cell_v2_20261003/recommended.json (rank %s, layout %s)' % (c.get('rank'), c.get('layout')),
            params=h['params'], T_link6_tcp=h['T_link6_tcp'], layout=c.get('layout'),
            gripper=dict(stl=SMC_STL, box=GRIP_BOX),
            fingers=dict(mount=fingers['mount'], boxes_grasp=fingers['boxes_grasp'], boxes_open=fingers['boxes_open'],
                         stl=next((p for p in FINGER_STL_CANDIDATES if os.path.exists(p)), None)),
            knife=dict(clamp=dict(centre=h['clamp_block']['centre_m'], radius=h['clamp_block']['radius_m']),
                       post=dict(frm=h['post']['from_xyz_m'], to=h['post']['to_xyz_m'], radius=h['post']['radius_m']),
                       handpiece=dict(frm=h['handpiece']['from_xyz_m'], to=h['handpiece']['to_xyz_m'], radius=h['handpiece']['radius_m']),
                       blade=dict(frm=h['handpiece']['to_xyz_m'], to=h['tcp_xyz_m'], exempt_last_m=0.02, collision=False)),
            camera=dict(T_link6_camera=cam['T_link6_camera'], body_size=cam['camera_body_size_m'], pad=0.003,
                        bracket=dict(frm=cam['bracket']['from_attach_on_camera_back_m'], to=cam['bracket']['to_mount_point_m'], radius=cam['bracket']['strut_radius_m'])),
            study_results=c.get('results'))
        try:
            sys.path.insert(0, FING)
            from common import des_from_params
            import camlib
            ds = des_from_params(h['params'], fingers=True)
            f = finals.get(c['label'])
            if f is not None:
                ds.cam = camlib.Cam(np.asarray(f['cam']['o']), np.asarray(f['cam']['Rc']), np.asarray(f['cam']['brp']))
            des['proxy'] = proxy_block(ds)
        except Exception as e:
            print('proxy block skipped for v2', c['label'], repr(e))
        out.append(des)
    return out


def baseline_design():
    sys.path.insert(0, FING)
    from common import baseline_design as bd      # study dalib.baseline_design (read-only import)
    from dalib import FLANGE_Z, HP_R, CLAMP_R
    d = bd()
    M = d.M.tolist()
    a = d.a
    end = (np.asarray(d.M) + max(d.L - 0.04, 0.0) * a).tolist()
    return dict(
        schema='cell_planner_check/design/v1', name='baseline', source='dalib.baseline_design() proxies (current knife, no fingers, no camera)',
        params=d.params, T_link6_tcp=d.T.tolist(),
        gripper=dict(stl=SMC_STL, box=GRIP_BOX), fingers=None, camera=None,
        knife=dict(clamp=dict(centre=M, radius=CLAMP_R), post=None,
                   handpiece=dict(frm=M, to=end, radius=HP_R),
                   blade=dict(frm=end, to=d.t.tolist(), exempt_last_m=0.02, collision=False)),
        proxy=proxy_block(d))


def main():
    os.makedirs(HERE + '/designs', exist_ok=True)
    if '--v2' in sys.argv:
        for d in v2_designs():
            json.dump(d, open('%s/designs/%s.json' % (HERE, d['name']), 'w'), indent=1)
            print('wrote designs/%s.json' % d['name'])
        return
    R = json.load(open(FING + '/recommended.json'))
    for c in R['combos']:
        des = combo_to_design(c, R['fingers'])
        json.dump(des, open('%s/designs/%s.json' % (HERE, c['label']), 'w'), indent=1)
        print('wrote designs/%s.json' % c['label'])
    json.dump(baseline_design(), open(HERE + '/designs/baseline.json', 'w'), indent=1)
    print('wrote designs/baseline.json')


if __name__ == '__main__':
    main()
