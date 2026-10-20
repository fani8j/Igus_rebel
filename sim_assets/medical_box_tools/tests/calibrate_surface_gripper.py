"""Headless Isaac Sim 5.1: calibrate the Surface Gripper detach force against the configured limits with a CLEAN pull test.
One free carton (no shelf, no neighbours, no floor, gravity 0, 82 g), the tool is held still by its slide drive, the pads touch the spine, the gripper is closed, then a
force ramp is applied to the carton along the pad axis (coaxial, away from the pads) or in the lip plane (shear along tool x = pad row, or tool y), at the lip centroid of
the attached pads, until the gripper leaves 'Closed'.  Quasi-static (rate independent, checked).
Sections:  A baseline (uncalibrated: authored limits = physical)  B independence of the two limits  C one pad  S coaxial-factor scan  D the authored defaults (shear factor 2.2) + SuctionLimitMonitor (physical pull-off)
Run: nice -n 19 /home/farhan/isaac-sim/python.sh tests/calibrate_surface_gripper.py   -> tests/results/surface_gripper_calibration.json (exit code 1 if section D fails)
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
TOOL = '/World/Tools/left_suction_tool'
CARR = TOOL + '_carrier'
SG = TOOL + '/surface_gripper'
CARTON = '/World/Cartons/carton_2_accuforce'
PAD_X = {1: -0.06, 2: 0.0, 3: 0.06}
TOL = 0.15


def main():
    app = start_app()
    print('### STAGE_OPEN')
    from pxr import Sdf
    from isaacsim.core.prims import RigidPrim, SingleRigidPrim, Articulation
    from scipy.spatial.transform import Rotation as Rot
    stage, sim = open_scene(app, 'pick_test_scene.usda', DT)
    for p in ('/World/Shelf', '/World/Ground', '/World/Tools/right_gripper_tool', '/World/Tools/right_gripper_tool_carrier',
              '/World/Cartons/carton_0_agent_monorail', '/World/Cartons/carton_1_angiolite', '/World/Cartons/carton_3_conqueror_nc_pro', '/World/Cartons/carton_4_agent_monorail'):
        pr = stage.GetPrimAtPath(p)
        if pr.IsValid():
            pr.SetActive(False)
    stage.GetPrimAtPath('/World/PhysicsScene').GetAttribute('physics:gravityMagnitude').Set(0.0)
    iface = tool_api.suction_interface()
    sgp = stage.GetPrimAtPath(SG)
    area, mu = sgp.GetCustomDataByKey('pad_area_m2'), sgp.GetCustomDataByKey('shear_mu')

    def run(kpa, direction, rate, pads=(1, 2, 3), authored=None, monitor=False):
        """authored = (coaxial, shear) absolute authored limits per point; None = the USD defaults at this vacuum (calibration factors applied)."""
        sim.stop()
        tool_api.set_vacuum_level(stage, SG, kpa)                      # calibrated defaults
        co_phys, sh_phys = tool_api.physical_limits(stage, SG)
        if authored is not None:
            sgp.GetAttribute('isaac:coaxialForceLimit').Set(float(authored[0]))
            sgp.GetAttribute('isaac:shearForceLimit').Set(float(authored[1]))
        sgp.GetRelationship('isaac:attachmentPoints').SetTargets([Sdf.Path(f'{TOOL}/surface_gripper_joints/pad_{i}_attachment') for i in pads])
        sim.initialize_physics()
        sim.play()
        carton = RigidPrim(CARTON)
        base = SingleRigidPrim(TOOL + '/base')
        art = Articulation(CARR)
        art.initialize()
        si = art.dof_names.index('world_slide')
        for _ in range(240):
            sim.step(render=False)
        s = -0.030
        while s < -0.0047:                                              # lips 0.3 mm from the spine, buffers not compressed
            s = min(s + 0.005 * DT, -0.0047)
            art.set_joint_position_targets(np.array([[s]]), joint_indices=np.array([si]))
            sim.step(render=False)
        for _ in range(120):
            sim.step(render=False)
        tool_api.suction_close(iface, SG)
        for _ in range(120):
            sim.step(render=False)
        if tool_api.suction_status(iface, SG) != 'Closed':
            return dict(error='not closed')
        mon = tool_api.SuctionLimitMonitor(art, iface, SG, co_phys).attach() if monitor else None
        qb = base.get_world_pose()[1]
        Rm = Rot.from_quat([qb[1], qb[2], qb[3], qb[0]]).as_matrix()
        d = {'coaxial': Rm[:, 2], 'push': -Rm[:, 2], 'shear_x': Rm[:, 0], 'shear_y': Rm[:, 1]}[direction]      # tool +Z points from the pads INTO the carton: pull-off = +Z, compression = -Z
        pos = base.get_world_pose()[0] + Rm @ np.array([np.mean([PAD_X[i] for i in pads]), 0.0, 0.159])   # force line through the centroid of the attached lips
        t = 0.0
        F = 0.0
        while t < 200.0:
            F = rate * t
            if os.environ.get('SG_LOG') and int(t / DT) % 480 == 0:
                e = art.get_measured_joint_efforts()[0]
                print('LOG', direction, round(F, 2), 'pad efforts', np.round([e[art.dof_names.index(f'pad_{i}_buffer_joint')] for i in (1, 2, 3)], 3), flush=True)
            carton.apply_forces_and_torques_at_pos(forces=np.array([d * F]), positions=np.array([pos]), is_global=True)
            sim.step(render=False)
            t += DT
            if tool_api.suction_status(iface, SG) != 'Closed' or not list(iface.get_gripped_objects(SG)):
                break
        if mon:
            mon.detach()
        n = len(pads)
        phys_total = n * (co_phys if direction in ('coaxial', 'push') else sh_phys)
        r = dict(kpa=kpa, direction=direction, pads=list(pads), rate=rate, detach_N=float(F), physical_total_N=float(phys_total),
                 ratio_to_physical=float(F / phys_total), authored_coaxial_N=float(sgp.GetAttribute('isaac:coaxialForceLimit').Get()),
                 authored_shear_N=float(sgp.GetAttribute('isaac:shearForceLimit').Get()), monitor=bool(monitor),
                 monitor_tripped=(list(mon.tripped_at) if mon and mon.tripped_at else None))
        return r

    def show(tag, r):
        print(tag, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)

    only = os.environ.get('SG_ONLY', 'ABCSD')
    out = {'A_baseline': [], 'B_independence': [], 'C_single_pad_2': [], 'D_calibrated': []}
    ph = lambda k: (k * 1e3 * area, mu * k * 1e3 * area)
    for kpa in ((25, 50, 75) if 'A' in only else ()):
        for dr, rate in (('coaxial', 4.0), ('shear_x', 0.5), ('shear_y', 0.5)):
            r = run(kpa, dr, rate, authored=ph(kpa))
            out['A_baseline'].append(r)
            show('A', r)
    for c, sh in (((58.0, 2.9), (5.8, 29.0), (0.05, 2.9)) if 'B' in only else ()):
        for dr, rate in (('coaxial', 8.0), ('shear_x', 2.0)):
            r = run(50, dr, rate, authored=(c, sh))
            r.update(coax_limit=c, shear_limit=sh)
            out['B_independence'].append(r)
            show('B', r)
    for dr, rate in ((('coaxial', 8.0), ('shear_x', 0.5), ('shear_y', 0.5)) if 'C' in only else ()):
        r = run(50, dr, rate, pads=(2,), authored=ph(50))
        out['C_single_pad_2'].append(r)
        show('C', r)
    out['S_scan'] = []
    for f in ((0.30, 0.35, 0.40) if 'S' in only else ()):   # scan of the coaxial factor with the calibrated shear limit (shows that a factor alone cannot reach the physical value)
        for kpa in (25, 50, 75):
            r = run(kpa, 'coaxial', 2.0, authored=(f * ph(kpa)[0], 2.2 * ph(kpa)[1]))
            r['factor'] = f
            out['S_scan'].append(r)
            show('S', r)
    ok = True
    for kpa in ((25, 50, 75) if 'D' in only else ()):
        for dr, rate in (('coaxial', 2.0), ('shear_x', 0.5), ('shear_y', 0.5)):
            r = run(kpa, dr, rate, monitor=True)
            r['within_15pct'] = abs(r['ratio_to_physical'] - 1.0) <= TOL
            ok &= r['within_15pct']
            out['D_calibrated'].append(r)
            show('D', r)
    out['all_D_within_15pct'] = ok
    json.dump(out, open(RESULTS / 'surface_gripper_calibration.json', 'w'), indent=1)
    print('CALIBRATION', 'PASS' if ok else 'FAIL', flush=True)
    sys.stdout.flush()
    os._exit(0 if ok else 1)


if __name__ == '__main__':
    main()
