"""Headless Isaac Sim 5.1: the MAIN rig (dual-arm stand on the D200 AMR, static placeholder, + medical tools) in d200_rig_test_scene.usda.
loads, joint list, stands still 5 s at home, tools respond. (Tipping margin, shelf placement and AMR driving are deferred.)
Run: nice -n 19 /home/farhan/isaac-sim/python.sh tests/test_d200.py   -> tests/results/d200.json
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
ROOT = '/World/D200'
U = ROOT + '/UpperAssembly/dual_arm_rig'
ARM_J = [f'arm_{s}_joint{k}' for s in ('left', 'right') for k in range(1, 7)]
STAND_J = ['lift_joint', 'slider_left_joint', 'slider_right_joint'] + ARM_J
AMR_J = ['left_wheel_joint', 'right_wheel_joint'] + [f'{p}_{s}_swivel_joint' for p in ('front', 'rear') for s in ('left', 'right')] + \
        [f'{p}_{s}_wheel_{k}_joint' for p in ('front', 'rear') for s in ('left', 'right') for k in (1, 2)]
TOOL_J = [f'arm_left_pad_{i}_buffer_joint' for i in (1, 2, 3)] + ['arm_right_finger_left_joint', 'arm_right_finger_right_joint']        # names as in the URDF (D200 tool variants)
URDF_DIR = Path('/home/farhan/igus_rebel_ros2/src/rebel_medicine_grasp_description/urdf')
WHEEL_R, TRACK = 0.0762, 0.586
WHEEL_R_ = WHEEL_R


def hull_margin(pts, c):
    """min distance from point c to the edges of the convex hull of pts (m, negative if outside)."""
    from scipy.spatial import ConvexHull
    h = ConvexHull(pts)
    d = []
    for eq in h.equations:                      # n . x + b <= 0 inside
        d.append(-(eq[:2] @ c + eq[2]))
    return float(min(d)), h


def main():
    app = start_app()
    print('### STAGE_OPEN')
    apply_d200_physics_settings()
    from pxr import UsdPhysics, UsdGeom, Gf
    from isaacsim.core.prims import Articulation, SingleRigidPrim
    from scipy.spatial.transform import Rotation as Rot
    stage, sim = open_scene(app, 'd200_rig_test_scene.usda', DT)
    chk = Checks('d200 rig')
    res = {}
    blk = UsdGeom.Cube.Define(stage, '/World/test_block')          # kinematic block, parked far away until the Surface Gripper test
    blk.CreateSizeAttr(1.0)
    bx = UsdGeom.Xformable(blk)
    m0 = Gf.Matrix4d(1.0)
    m0.SetTranslateOnly(Gf.Vec3d(0, 0, 8.0))
    bx.AddTransformOp().Set(m0)
    bx.AddScaleOp().Set(Gf.Vec3f(0.20, 0.04, 0.03))
    UsdPhysics.CollisionAPI.Apply(blk.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(blk.GetPrim()).CreateKinematicEnabledAttr(True)
    sim.initialize_physics()
    sim.play()
    art = Articulation(ROOT + '/base_link')
    art.initialize()
    names = list(art.dof_names)
    res['n_dof'] = len(names)
    res['dof_names'] = names
    exp = STAND_J + AMR_J + TOOL_J
    missing = [j for j in exp if j not in names]
    extra = [j for j in names if j not in exp]
    res['missing'], res['extra'] = missing, extra
    chk.check(f'articulation root = AMR base_link, {len(names)} DOFs (15 stand/arm + 14 AMR + 5 tool = {len(exp)})', len(names) == len(exp), len(names))
    chk.check('no joint missing (lift, sliders, arm_*_joint1..6, wheels, swivels, caster wheels, pads, fingers)', not missing, missing)
    chk.check('no unexpected joint (the XEG-32 joints are gone)', not extra, extra)
    if URDF_DIR.exists():
        txt = (URDF_DIR / 'rebel_medicine_grasp.ros2_control.xacro').read_text() + (URDF_DIR / 'rebel_medicine_grasp.urdf.xacro').read_text()
        urdf_ok = all(n in txt for n in ('arm_right_finger_left_joint', 'arm_right_finger_right_joint', 'lift_joint', 'slider_left_joint', 'slider_right_joint'))
        chk.check('joint names agree with the MoveIt URDF / ros2_control (lift, sliders, arm_right_finger_left (driver) / finger_right (mimic), arm_*_joint1..6 by prefix)', urdf_ok)
    # bodies: links of the articulation (tensor view); body names are unique per link
    class Link:
        def __init__(self, name):
            self.name = name

        def world_pose(self):
            i = LP.index(self.name)
            t = np.array(art._physics_view.get_link_transforms()[0][i])
            return t[:3].copy(), np.array([t[6], t[3], t[4], t[5]])             # (pos, quat wxyz)

        def get_world_pose(self):
            return self.world_pose()
    LP = [str(x) for x in art._physics_view.link_paths[0]]
    body_paths = LP
    bodies = {n: Link(n) for n in body_paths}
    base = bodies[ROOT + '/base_link']
    pose = lambda b: b.world_pose()

    def yaw_tilt(q):
        R = Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        return float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))), float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))
    # ---- stand 5 s, wheels 0
    first = []
    for k in range(int(3.0 / DT)):                                   # start transient: authored start state -> equilibrium
        sim.step(render=False)
        if k % 240 == 239:
            first.append(base.get_world_pose()[0].copy())
    res['start_transient_xy_mm'] = float(np.linalg.norm(first[-1][:2]) * 1000)
    p0, q0 = pose(base)
    l6 = bodies[f'{U}/arm_left/link6']
    p60 = l6.get_world_pose()[0].copy()
    samples = []
    for k in range(int(5.0 / DT)):
        sim.step(render=False)
        if k % 24 == 0:
            samples.append((base.get_world_pose()[0].copy(), base.get_world_pose()[1].copy()))
    p1, q1 = samples[-1]
    drift = np.linalg.norm(p1[:2] - p0[:2]) * 1000
    dz = (p1[2] - p0[2]) * 1000
    yaw0, tilt0 = yaw_tilt(q0)
    yaw1, tilt1 = yaw_tilt(q1)
    maxtilt = max(yaw_tilt(q)[1] for _, q in samples)
    exc = float(max(np.linalg.norm(sp[0][:2] - np.mean([x[0][:2] for x in samples], axis=0)) for sp in samples) * 1000)
    res['stand'] = dict(excursion_mm=exc, base_z_m=float(p1[2]), drift_xy_mm=float(drift), dz_mm=float(dz), yaw_drift_deg=yaw1 - yaw0, max_tilt_deg=maxtilt, tilt_end_deg=tilt1,
                        link6_left_world=[float(x) for x in l6.get_world_pose()[0]], link6_motion_mm=float(np.linalg.norm(l6.get_world_pose()[0] - p60) * 1000))
    chk.check('stands on the ground for 5 s (wheels 0, arms home): base_link z = 0.0792 +-5 mm (axle height + 3 mm)', abs(p1[2] - 0.0792) < 0.005, round(float(p1[2]), 4))
    chk.check(f'no drift (static AMR; after a 3 s start transient of {res["start_transient_xy_mm"]:.0f} mm): net xy < 10 mm, wander (casters) < 15 mm, yaw < 0.2 deg over 5 s', drift < 10.0 and exc < 15.0 and abs(yaw1 - yaw0) < 0.2,
              (round(float(drift), 3), round(exc, 3), round(yaw1 - yaw0, 4)))
    chk.check('no tipping / sinking: max tilt < 0.1 deg, dz < 2 mm', maxtilt < 0.1 and abs(dz) < 2.0, (round(maxtilt, 4), round(float(dz), 3)))
    q = art.get_joint_positions()[0]
    home = {'lift_joint': -0.2, 'slider_left_joint': -0.005973, 'slider_right_joint': 0.005973}
    got = {n: float(q[names.index(n)]) for n in home}
    res['home_state'] = got
    arm_dq = max(abs(float(q[names.index(n)])) for n in ARM_J)
    res['arm_joint_max_abs_rad'] = arm_dq
    chk.check('home state: lift -0.2, sliders -+0.005973 (+-1 mm), arms upright (|q| < 1.5 deg)', all(abs(got[n] - home[n]) < 0.001 for n in home) and arm_dq < np.radians(1.5),
              ({k: round(v, 5) for k, v in got.items()}, round(float(np.degrees(arm_dq)), 3)))
    # ---- tools respond (block at the left pad lips; fingers)
    iface = tool_api.suction_interface()
    tp = f'{U}/arm_left/left_suction_tool'
    gp = f'{U}/arm_right/right_gripper_tool'
    # fingers
    il, ir = names.index('arm_right_finger_left_joint'), names.index('arm_right_finger_right_joint')

    def fingers(q_from, q_to, push):
        for q in tool_api.finger_target_profile(q_from, q_to, DT, push=push):
            art.set_joint_position_targets(np.array([[q, q]]), joint_indices=np.array([il, ir]))
            sim.step(render=False)
        for _ in range(int(0.5 / DT)):
            sim.step(render=False)
        qq = art.get_joint_positions()[0]
        return 0.008 + qq[il] + qq[ir], qq[il] - qq[ir]
    g0 = 0.008 + art.get_joint_positions()[0][il] + art.get_joint_positions()[0][ir]
    gc, asym = fingers(0.024, tool_api.GRIP_CLOSE_Q, True)
    go, _ = fingers(tool_api.GRIP_CLOSE_Q, 0.024, False)
    res['fingers'] = dict(start_gap_mm=float(g0 * 1000), closed_gap_mm=float(gc * 1000), open_gap_mm=float(go * 1000), asym_mm=float(asym * 1000))
    chk.check('fingers: open 56 mm, close to the 8 mm stop (symmetric), reopen 56 mm', abs(g0 * 1000 - 56) < 0.6 and abs(gc * 1000 - 8) < 0.6 and abs(go * 1000 - 56) < 0.6 and abs(asym) < 0.0005,
              (round(g0 * 1000, 2), round(gc * 1000, 2), round(go * 1000, 2)))
    # surface gripper: kinematic block just above the left pad lips (tool +Z = up at home), created before a restart so that the physics parses it
    for _ in range(int(3.0 / DT)):
        sim.step(render=False)
    pad2 = Link(tp + '/pad_2')
    t2, qn2 = pad2.get_world_pose()
    Rm = Rot.from_quat([qn2[1], qn2[2], qn2[3], qn2[0]]).as_matrix()
    lip = t2 + Rm[:, 2] * 0.159
    blk_body = SingleRigidPrim('/World/test_block')
    start = lip + Rm[:, 2] * (0.0151 + 0.010)
    blk_body.set_world_pose(position=start, orientation=qn2)
    for _ in range(int(0.3 / DT)):
        sim.step(render=False)
    for k in range(1, 21):
        blk_body.set_world_pose(position=start - Rm[:, 2] * 0.010 * k / 20, orientation=qn2)
        sim.step(render=False)
    for _ in range(int(0.5 / DT)):
        sim.step(render=False)
    sg = tp + '/surface_gripper'
    tool_api.suction_close(iface, sg)
    for _ in range(int(1.0 / DT)):
        sim.step(render=False)
    st, objs = tool_api.suction_status(iface, sg), list(iface.get_gripped_objects(sg))
    res['suction'] = dict(status=st, gripped=objs)
    chk.check('Surface Gripper on arm_left closes on the block at the pad lips (Closed, gripped = block)', st == 'Closed' and objs == ['/World/test_block'], (st, objs))
    tool_api.suction_open(iface, sg)
    for _ in range(int(0.3 / DT)):
        sim.step(render=False)
    chk.check('Surface Gripper opens', tool_api.suction_status(iface, sg) == 'Open')
    pads = [-float(art.get_joint_positions()[0][art.dof_names.index(f'arm_left_pad_{i}_buffer_joint')]) * 1000 for i in (1, 2, 3)]
    chk.check('pad buffers respond (finite, 0..10 mm)', all(-0.5 <= x <= 10.5 for x in pads), [round(x, 2) for x in pads])
    ok = chk.summary()
    json.dump(dict(res, checks=chk.rows), open(RESULTS / 'd200.json', 'w'), indent=1, default=float)
    sys.stdout.flush()
    os._exit(0 if ok else 1)


if __name__ == '__main__':
    main()
