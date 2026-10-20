"""Headless Isaac Sim 5.1: the tools mounted on a robot link6 (single ReBeL variants + the dual-arm stand rig).
For each assembly: loads, holds its pose under gravity for 3 s, joint names vs the URDF, the Surface Gripper closes on a kinematic block held at the pad lips,
the finger drives close / open.   Run:  nice -n 19 /home/farhan/isaac-sim/python.sh tests/test_assemblies.py [name ...]
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tools'))
from isaac_common import *            # noqa: E402,F401
import numpy as np                    # noqa: E402
import tool_api                       # noqa: E402

DT = 1 / 240.0
CHAIN = 'base_link/motor1/link1/motor2/link2/motor3/link3/motor4/link4/motor5/link5/link6'
URDF_RIG_JOINTS = ['lift_joint', 'slider_left_joint', 'slider_right_joint'] + [f'arm_{s}_joint{k}' for s in ('left', 'right') for k in range(1, 7)]
ASSEMBLIES = {
    'rebel_left_suction': dict(file='rebel_left_suction.usda', art='/World/rebel', expect=[f'joint{k}' for k in range(1, 7)],
                               suction='/World/left_suction_tool', gripper=None),
    'rebel_right_gripper': dict(file='rebel_right_gripper.usda', art='/World/rebel', expect=[f'joint{k}' for k in range(1, 7)],
                                suction=None, gripper='/World/right_gripper_tool'),
    'dual_arm_rig_medical': dict(file='dual_arm_rig_medical.usda', art='/World/dual_arm_rig/stand', expect=URDF_RIG_JOINTS,
                                 suction=f'/World/dual_arm_rig/arm_left/{CHAIN}/left_suction_tool',
                                 gripper=f'/World/dual_arm_rig/arm_right/{CHAIN}/right_gripper_tool'),
}


def run_one(app, name, cfg, chk, res):
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics
    from isaacsim.core.prims import Articulation, SingleRigidPrim
    stage, sim = open_scene(app, cfg['file'], DT)
    r = res[name] = {}
    sim.initialize_physics()
    sim.play()
    art = Articulation(cfg['art'])
    art.initialize()
    names = list(art.dof_names)
    r['dof_names'] = names
    missing = [j for j in cfg['expect'] if j not in names]
    r['missing_vs_urdf'] = missing
    r['extra_dofs'] = [n for n in names if n not in cfg['expect']]
    chk.check(f'[{name}] articulation found, {len(names)} DOFs', len(names) >= len(cfg['expect']), len(names))
    chk.check(f'[{name}] no joint of the URDF is missing ({len(cfg["expect"])} expected)', not missing, missing)
    q0 = art.get_joint_positions()[0].copy()
    tool_prims = [p for p in (cfg['suction'], cfg['gripper']) if p]
    base_paths = [t + '/base' for t in tool_prims]
    bodies = [SingleRigidPrim(b) for b in base_paths]
    p0 = [b.get_world_pose()[0].copy() for b in bodies]
    for _ in range(int(2.0 / DT)):
        sim.step(render=False)
    q2 = art.get_joint_positions()[0].copy()
    p2 = [b.get_world_pose()[0].copy() for b in bodies]
    for _ in range(int(1.0 / DT)):
        sim.step(render=False)
    q1 = art.get_joint_positions()[0]
    p1 = [b.get_world_pose()[0] for b in bodies]
    dq = np.abs(q1 - q0)
    r['settled_dq_max'] = float(np.abs(q1 - q2).max())
    r['settled_tool_move_mm'] = [float(np.linalg.norm(a - b) * 1000) for a, b in zip(p2, p1)]
    ang = [i for i, n in enumerate(names) if 'joint' in n and n.startswith(('arm_', 'joint'))]
    lin = [i for i, n in enumerate(names) if n in ('lift_joint', 'slider_left_joint', 'slider_right_joint')]
    r['max_dq_angular_deg'] = float(np.degrees(dq[ang]).max()) if ang else 0.0
    r['max_dq_linear_mm'] = float(dq[lin].max() * 1000) if lin else 0.0
    r['tool_drift_mm'] = [float(np.linalg.norm(a - b) * 1000) for a, b in zip(p0, p1)]
    chk.check(f'[{name}] finite joint state after 3 s', bool(np.all(np.isfinite(q1))))
    chk.check(f'[{name}] no collapse: arm joints sag < 2 deg from the start state (3 s)', r['max_dq_angular_deg'] < 2.0, round(r['max_dq_angular_deg'], 4))
    chk.check(f'[{name}] settled: joints move < 0.05 deg / 0.05 mm during the last second', r['settled_dq_max'] < 1e-3, r['settled_dq_max'])
    if lin:
        r['q_start_linear'] = {names[i]: float(q0[i]) for i in lin}
        r['settled_linear_mm'] = float(np.abs(q1 - q2)[lin].max() * 1000)
        chk.check(f'[{name}] lift/sliders at rest (< 0.05 mm in the last second); start-to-3 s change {r["max_dq_linear_mm"]:.1f} mm (settling from the authored start state)',
                  r['settled_linear_mm'] < 0.05, round(r['settled_linear_mm'], 4))
        want = {'lift_joint': -0.2, 'slider_left_joint': -0.005973, 'slider_right_joint': 0.005973}
        got = {n: float(q1[names.index(n)]) for n in want}
        r['home_state'] = got
        chk.check(f'[{name}] README home state: lift -0.2, sliders -+0.005973 (+-0.5 mm)', all(abs(got[n] - want[n]) < 0.0005 for n in want), {k: round(v, 5) for k, v in got.items()})
    chk.check(f'[{name}] tool is at rest (< 0.5 mm in the last second); moved {[round(d, 1) for d in r["tool_drift_mm"]]} mm from the start state',
              all(d < 40.0 for d in r['tool_drift_mm']) and all(d < 0.5 for d in r['settled_tool_move_mm']), ([round(d, 2) for d in r['tool_drift_mm']], [round(d, 3) for d in r['settled_tool_move_mm']]))
    r['q_home_deg_or_m'] = [float(x) for x in q1]
    # ---- tool-specific
    if cfg['suction']:
        tp = cfg['suction']
        pad2 = SingleRigidPrim(tp + '/pad_2')
        t, qn = pad2.get_world_pose()
        from scipy.spatial.transform import Rotation as Rot
        Rm = Rot.from_quat([qn[1], qn[2], qn[3], qn[0]]).as_matrix()
        qn_block = np.array(qn)
        lip = t + Rm[:, 2] * 0.159 + Rm[:, 0] * 0.0          # pad_2 body frame = link6 frame; lip centre at tool z 0.159
        # phase 2: rebuild with a kinematic block touching the three lips (tool x: pad pitch 60 mm), sim restarted from the authored state
        sim.stop()
        blk = UsdGeom.Cube.Define(stage, '/World/test_block')
        blk.CreateSizeAttr(1.0)
        Tm = np.eye(4)
        Tm[:3, :3] = Rm
        Tm[:3, 3] = lip + Rm[:, 2] * (0.0151 + 0.010)           # starts 10 mm clear of the lips (the arm settles from its start state first)
        xf = UsdGeom.Xformable(blk)
        xf.ClearXformOpOrder()
        m = Gf.Matrix4d(1.0)
        m.SetRotateOnly(Gf.Matrix3d(*[float(x) for x in Rm.T.reshape(9)]))
        m.SetTranslateOnly(Gf.Vec3d(*[float(x) for x in Tm[:3, 3]]))
        xf.AddTransformOp().Set(m)
        xf.AddScaleOp().Set(Gf.Vec3f(0.20, 0.04, 0.03))        # 200 mm along tool x (covers the pads at -60/0/+60), 30 mm along z
        UsdPhysics.CollisionAPI.Apply(blk.GetPrim())
        UsdPhysics.RigidBodyAPI.Apply(blk.GetPrim()).CreateKinematicEnabledAttr(True)
        sim.initialize_physics()
        sim.play()
        for _ in range(int(3.0 / DT)):
            sim.step(render=False)
        art = Articulation(cfg['art'])
        art.initialize()
        blk_body = SingleRigidPrim('/World/test_block')
        for k in range(1, 21):                                  # slide the kinematic block down onto the lips (10 mm in 20 steps)
            blk_body.set_world_pose(position=Tm[:3, 3] - Rm[:, 2] * 0.010 * k / 20, orientation=qn_block)
            sim.step(render=False)
        for _ in range(int(0.5 / DT)):
            sim.step(render=False)
        iface = tool_api.suction_interface()
        sg = tp + '/surface_gripper'
        tool_api.suction_close(iface, sg)
        for _ in range(int(1.0 / DT)):
            sim.step(render=False)
        st, objs = tool_api.suction_status(iface, sg), list(iface.get_gripped_objects(sg))
        r['suction_close'] = dict(status=st, gripped=objs)
        chk.check(f'[{name}] mounted Surface Gripper closes on the block at the pad lips', st == 'Closed' and objs == ['/World/test_block'], (st, objs))
        qn = art.get_joint_positions()[0]
        comp = [-float(qn[art.dof_names.index(f'pad_{i}_buffer_joint')]) * 1000 for i in (1, 2, 3)]
        r['pad_compression_mm'] = comp
        chk.check(f'[{name}] pad buffers respond (0 .. 10 mm compression, finite)', all(-0.5 <= c <= 10.5 for c in comp), [round(c, 2) for c in comp])
        tool_api.suction_open(iface, sg)
        for _ in range(int(0.3 / DT)):
            sim.step(render=False)
        chk.check(f'[{name}] mounted Surface Gripper opens', tool_api.suction_status(iface, sg) == 'Open' and list(iface.get_gripped_objects(sg)) == [])
    if cfg['gripper']:
        gp = cfg['gripper']
        il, ir = art.dof_names.index('finger_left_joint'), art.dof_names.index('finger_right_joint')

        def drive(q_from, q_to, push):
            for q in tool_api.finger_target_profile(q_from, q_to, DT, push=push):
                art.set_joint_position_targets(np.array([[q, q]]), joint_indices=np.array([il, ir]))
                sim.step(render=False)
            for _ in range(int(0.5 / DT)):
                sim.step(render=False)
            qq = art.get_joint_positions()[0]
            return 0.008 + qq[il] + qq[ir], qq[il] - qq[ir]
        gap_o, _ = drive(0.0, 0.024, False) if False else (None, None)
        qq = art.get_joint_positions()[0]
        r['fingers_start_gap_mm'] = (0.008 + qq[il] + qq[ir]) * 1000
        chk.check(f'[{name}] fingers start open (56 mm)', abs(r['fingers_start_gap_mm'] - 56.0) < 0.6, round(r['fingers_start_gap_mm'], 2))
        gap_c, asym = drive(0.024, tool_api.GRIP_CLOSE_Q, True)
        r['finger_close_gap_mm'], r['finger_close_asym_mm'] = gap_c * 1000, asym * 1000
        chk.check(f'[{name}] finger drives close to the 8 mm stop, symmetric', abs(gap_c * 1000 - 8.0) < 0.6 and abs(asym) < 0.0005, (round(gap_c * 1000, 2), round(asym * 1000, 3)))
        gap_o, asym = drive(tool_api.GRIP_CLOSE_Q, 0.024, False)
        r['finger_open_gap_mm'] = gap_o * 1000
        chk.check(f'[{name}] finger drives reopen to 56 mm', abs(gap_o * 1000 - 56.0) < 0.6, round(gap_o * 1000, 2))
    if name.startswith('rebel'):
        # joint 6 motion carries the tool: rotate joint6 by +0.5 rad about its axis, tool z axis must stay, tool x rotates
        j6 = art.dof_names.index('joint6')
        body = SingleRigidPrim((cfg['suction'] or cfg['gripper']) + '/base')
        qb = body.get_world_pose()[1].copy()
        tgt = art.get_joint_positions()[0].copy()
        tgt[j6] += 0.5
        art.set_joint_position_targets(tgt.reshape(1, -1))
        for _ in range(int(2.0 / DT)):
            sim.step(render=False)
        qa = body.get_world_pose()[1]
        from scipy.spatial.transform import Rotation as Rot
        rel = Rot.from_quat([qa[1], qa[2], qa[3], qa[0]]) * Rot.from_quat([qb[1], qb[2], qb[3], qb[0]]).inv()
        ang_deg = np.degrees(rel.magnitude())
        r['joint6_plus_0p5rad_tool_rotation_deg'] = float(ang_deg)
        chk.check(f'[{name}] tool follows joint6 (+0.5 rad = 28.6 deg)', abs(ang_deg - 28.65) < 1.5, round(float(ang_deg), 2))
    sim.stop()


def main():
    app = start_app()
    print('### STAGE_OPEN')
    chk = Checks('assemblies')
    res = {}
    todo = sys.argv[1:] or list(ASSEMBLIES)
    for n in todo:
        print('### ASSEMBLY', n)
        run_one(app, n, ASSEMBLIES[n], chk, res)
    ok = chk.summary()
    json.dump(dict(res, checks=chk.rows), open(RESULTS / ('assemblies.json' if not sys.argv[1:] else 'assemblies_partial.json'), 'w'), indent=1, default=float)
    sys.stdout.flush()
    app.close()
    os._exit(0 if ok else 1)


if __name__ == '__main__':
    main()
