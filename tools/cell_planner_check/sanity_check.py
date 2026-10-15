#!/usr/bin/env python3
"""Sanity checks of the MoveIt model against the offline study (needs the stack up on domain 187).

  export ROS_DOMAIN_ID=187 ROS_LOCALHOST_ONLY=1
  python3 sanity_check.py --build build/h000n.0_s0355 [--cases cases.json] --out /path/sanity.json

Checks
  1. FK: study joint solutions (random in-limit states + optional --cases from the job builder) -> /compute_fk of armN_knife_tip vs the study
     pose  T_world = translate(0, y_arm, 0) * fk_flange(q) * T_link6_tcp   (tolerance 1 mm, 0.1 deg).
  2. Frame: arm base / link6 at q=0 in CAD coordinates vs layout.json (the study's Y-up CAD frame).
  3. Structure: link origins of the structure boxes vs the CAD AABBs of layout.json (converted CAD->W, per-arm slide shift for spacing s).
  4. Scene: carton box / table placed by the validator vs the study's dalib.box_sdf convention (corner positions).
  5. No ros2_control node, services present.
"""
import argparse, json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, '/home/farhan/igus_rebel_ros2/scan_results/holder_camera_fingers_20261002')
import cellgeom as G
import moveit_client as MC
import rclpy


def study_tcp_world(q, arm_idx, s, T_tcp):
    import reach_lib as R
    T6 = R.fk_flange(np.asarray(q, float)[None])[0]
    T = T6 @ T_tcp
    T[1, 3] += G.arm_y(arm_idx, s)
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--build', required=True)
    ap.add_argument('--cases', default=None)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n_random', type=int, default=40)
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.build if os.path.isabs(a.build) else os.path.join(HERE, a.build), 'cell_meta.json')))
    s = meta['s']
    T_tcp = np.array(meta['T_link6_tcp'])
    MC.init()
    cl = MC.MoveItClient(meta, 'sanity_check')
    out = dict(design=meta['design'], s=s, domain=os.environ.get('ROS_DOMAIN_ID'), localhost_only=os.environ.get('ROS_LOCALHOST_ONLY'))
    rng = np.random.default_rng(7)
    lo = np.array([-178, -79, -79, -178, -94, -178.0]) * math.pi / 180
    hi = -lo
    hi[1] = hi[2] = 139 * math.pi / 180
    lo = np.array([-178, -79, -79, -178, -94, -178.0]) * math.pi / 180
    # ---- 1. FK
    rows = []
    cases = []
    for i in range(a.n_random):
        qs = [list(rng.uniform(lo, hi)), list(rng.uniform(lo, hi))]
        cases.append(dict(label='random%d' % i, q={'arm1': qs[0], 'arm2': qs[1]}))
    if a.cases:
        for c in json.load(open(a.cases)):
            qq = {'arm1': [0.0] * 6, 'arm2': [0.0] * 6}
            qq[c['arm']] = c['q']
            cases.append(dict(label=c['label'], q=qq, arm=c['arm'], expect=c.get('expect_tcp_world')))
    for c in cases:
        for ai, arm in enumerate(G.ARMS):
            if 'arm' in c and c['arm'] != arm:
                continue
            Tm = cl.fk(c['q'], meta['arms'][arm]['tip'])
            Ts = study_tcp_world(c['q'][arm], ai, s, T_tcp)
            dp = float(np.linalg.norm(Tm[:3, 3] - Ts[:3, 3]) * 1000)
            dr = G.rot_angle_deg(Tm[:3, :3].T @ Ts[:3, :3])
            row = dict(label=c['label'], arm=arm, pos_err_mm=dp, rot_err_deg=dr)
            if c.get('expect'):
                Te = G.pose_to_T(c['expect'])
                row['vs_target_pos_mm'] = float(np.linalg.norm(Tm[:3, 3] - Te[:3, 3]) * 1000)
                row['vs_target_rot_deg'] = G.rot_angle_deg(Tm[:3, :3].T @ Te[:3, :3])
            rows.append(row)
    out['fk_rows'] = rows
    out['fk_max_pos_mm'] = max(r['pos_err_mm'] for r in rows)
    out['fk_max_rot_deg'] = max(r['rot_err_deg'] for r in rows)
    tgt = [r for r in rows if 'vs_target_pos_mm' in r]
    if tgt:
        out['target_max_pos_mm'] = max(r['vs_target_pos_mm'] for r in tgt)
        out['target_max_rot_deg'] = max(r['vs_target_rot_deg'] for r in tgt)
    out['fk_pass'] = bool(out['fk_max_pos_mm'] < 1.0 and out['fk_max_rot_deg'] < 0.1)
    # ---- 2. frame vs layout.json (CAD)
    L = json.load(open(G.REPO + '/scan_results/dual_arm_cell_20261002/layout.json'))
    q0 = {'arm1': [0.0] * 6, 'arm2': [0.0] * 6}
    fr = {}
    for ai, arm in enumerate(G.ARMS):
        lay = L['arms'][arm]['link_frame_origins_world_mm_at_cad_pose']
        for link in ('base_link', 'link3', 'link6'):
            Tm = cl.fk(q0, '%s_%s' % (arm, link))
            cad = G.world_to_cad(Tm[:3, 3]) * 1000
            ref = np.array(lay[link])
            # the study layout is for spacing 0.459944: shift the reference X by the commanded spacing change
            ref = ref.copy()
            ref[0] += (G.arm_y(ai, s) - (G.CAD_SPACING / 2 if ai == 0 else -G.CAD_SPACING / 2)) * 1000
            fr['%s_%s' % (arm, link)] = dict(moveit_cad_mm=cad.round(2).tolist(), layout_cad_mm=ref.tolist(), err_mm=float(np.linalg.norm(cad - ref)))
    out['frame_vs_layout'] = fr
    out['frame_max_err_mm'] = max(v['err_mm'] for v in fr.values())
    # ---- 3. structure links
    st = {}
    if meta['structure'] == 'cad':
        sys.path.insert(0, HERE)
        import gen_cell as GC
        exp = {n: (c, sz) for n, c, sz, _ in GC.structure_boxes('cad', s)}
        for nm in meta['structure_links']:
            Tm = cl.fk(q0, nm)
            c, sz = exp[nm]
            st[nm] = dict(err_mm=float(np.linalg.norm(Tm[:3, 3] - c) * 1000), centre_w=Tm[:3, 3].round(4).tolist(), size=np.round(sz, 4).tolist())
        out['structure_max_err_mm'] = max(v['err_mm'] for v in st.values())
        out['structure'] = st
    # ---- 4. carton placement vs study box_sdf convention
    import dalib as D
    sc = []
    chk = []
    for box, yaw in (('BIG370', 0), ('BIG370', 30), ('BIG370', 135), ('SMALL', 60)):
        cg = G.carton_geometry(box, 0.0, 0.40, yaw)
        # corners via the study convention: local x along a=(-sin psi, cos psi) half A/2, local y along b=(cos psi, sin psi) half B/2
        A, B, H = D.BOXES[box]
        a_, b_ = D.carton_axes(yaw)
        Rm = G.quat_to_R(cg['quat'])
        err = 0.0
        cs = [np.array([0.40, 0, 0]) + sx * a_ * A / 2 + sy * b_ * B / 2 for sx in (-1, 1) for sy in (-1, 1)]
        cm = [np.array(cg['center']) + Rm @ np.array([sx * A / 2, sy * B / 2, 0.0]) for sx in (-1, 1) for sy in (-1, 1)]
        for ps in cs:     # corner sets must coincide (corner-by-corner order is irrelevant)
            err = max(err, min(float(np.linalg.norm(ps[:2] - pm[:2]) * 1000) for pm in cm))
        top_err = abs((cg['center'][2] + H / 2) - 0.0) * 1000
        chk.append(dict(box=box, yaw=yaw, max_corner_err_mm=err, top_z_err_mm=top_err))
    out['carton_convention'] = chk
    # ---- 5. environment
    import subprocess
    nl = subprocess.run(['ros2', 'node', 'list'], capture_output=True, text=True, env=dict(os.environ)).stdout.split()
    out['nodes'] = nl
    out['ros2_control_node_present'] = any(n.rsplit('/', 1)[-1] in ('ros2_control_node', 'controller_manager') for n in nl)
    out['n_service_calls'] = cl.ncalls
    json.dump(out, open(a.out, 'w'), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ('fk_rows', 'structure', 'frame_vs_layout')}, indent=1))
    print('frame_vs_layout', json.dumps({k: v['err_mm'] for k, v in fr.items()}))
    cl.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
