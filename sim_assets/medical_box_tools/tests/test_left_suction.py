"""Headless Isaac Sim 5.1 test of the left suction tool (v5): buffers, Surface Gripper attach, pull 100 mm, release, detach threshold.
Run:  nice -n 19 /home/farhan/isaac-sim/python.sh tests/test_left_suction.py   (writes tests/results/left_suction.{json,log})
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tools'))
from isaac_common import *            # noqa: E402,F401  (sets ROS_DOMAIN_ID=170, ROS_LOCALHOST_ONLY=1)
import numpy as np                    # noqa: E402
import tool_api                       # noqa: E402

DT = 1 / 240.0
TOOL = '/World/Tools/left_suction_tool'
CARR = TOOL + '_carrier'
SG = TOOL + '/surface_gripper'
CARTON = '/World/Cartons/carton_2_accuforce'
VACUUM_DELAY_S = 0.15          # real suction needs 0.05..0.2 s to build up; the Surface Gripper engages instantly -> we wait this long after contact
APPROACH_SPEED, PULL_SPEED = 0.020, 0.050
S_START, S_PRESS = -0.030, 0.0  # slide s: 0 = nominal handoff pose, pads compressed 5 mm


class Run:
    def __init__(self, app, stage, sim, iface):
        self.app, self.stage, self.sim, self.iface = app, stage, sim, iface

    def reset(self, mass=None, vacuum_kpa=None, n_attach=3):
        from isaacsim.core.prims import SingleRigidPrim, Articulation
        self.sim.stop()
        st = self.stage
        cp = st.GetPrimAtPath(CARTON)
        m0 = 0.107 if mass is None else mass
        inertia_scale = m0 / cp.GetAttribute('physics:mass').Get()
        d0 = cp.GetAttribute('physics:diagonalInertia').Get()
        self._orig = getattr(self, '_orig', (cp.GetAttribute('physics:mass').Get(), d0))
        cp.GetAttribute('physics:mass').Set(float(self._orig[0] if mass is None else mass))
        base_I = self._orig[1]
        k = (1.0 if mass is None else mass / self._orig[0])
        cp.GetAttribute('physics:diagonalInertia').Set(type(base_I)(*[float(x * k) for x in base_I]))
        if vacuum_kpa is not None:
            tool_api.set_vacuum_level(st, SG, vacuum_kpa)
        rel = st.GetPrimAtPath(SG).GetRelationship('isaac:attachmentPoints')
        rel.SetTargets([Sdf.Path(f'{TOOL}/surface_gripper_joints/pad_{i}_attachment') for i in range(1, n_attach + 1)])
        self.sim.initialize_physics()
        self.sim.play()
        self.carton = SingleRigidPrim(CARTON)
        self.base = SingleRigidPrim(TOOL + '/base')
        self.pads = [SingleRigidPrim(f'{TOOL}/pad_{i}') for i in (1, 2, 3)]
        self.art = Articulation(CARR)
        self.art.initialize()
        self.si = self.art.dof_names.index('world_slide')
        self.pi = [self.art.dof_names.index(f'pad_{i}_buffer_joint') for i in (1, 2, 3)]
        self.s_target = S_START
        if getattr(self, 'mon', None):
            self.mon.detach()
        co_phys, _ = tool_api.physical_limits(st, SG)
        self.phys_total = 3 * co_phys if n_attach == 3 else n_attach * co_phys
        self.mon = tool_api.SuctionLimitMonitor(self.art, self.iface, SG, co_phys).attach()    # physical pull-off limit (the 5.1 extension does not enforce it)
        self.step(240)
        self.log = []

    def set_s(self, s):
        self.s_target = s
        self.art.set_joint_position_targets(np.array([[s]]), joint_indices=np.array([self.si]))

    def step(self, n=1):
        for _ in range(n):
            self.sim.step(render=False)

    def q(self):
        return self.art.get_joint_positions()[0]

    def slide_force(self):
        return float(self.art.get_measured_joint_efforts()[0][self.si])

    def carton_y(self):
        return float(self.carton.get_world_pose()[0][1])

    def base_y(self):
        return float(self.base.get_world_pose()[0][1])

    def ramp(self, s_to, speed, rec=None):
        s = self.s_target
        n = int(abs(s_to - s) / speed / DT)
        for k in range(n):
            self.set_s(s + (s_to - s) * (k + 1) / n)
            self.step(1)
            if rec is not None:
                rec(self)
        self.set_s(s_to)

    def gripped(self):
        return list(self.iface.get_gripped_objects(SG))

    def status(self):
        return tool_api.suction_status(self.iface, SG)


def pick(run, chk, mass=None, tag='nominal', full=True, n_attach=3):
    run.reset(mass=mass, n_attach=n_attach)
    y0 = run.carton_y()
    out = dict(tag=tag, mass_kg=run.stage.GetPrimAtPath(CARTON).GetAttribute('physics:mass').Get())
    run.ramp(S_PRESS, APPROACH_SPEED)
    run.step(int(0.3 / DT))
    q = run.q()
    comp = [-q[i] * 1000 for i in run.pi]
    kpad = 200.0
    spring_force = [kpad * (0.005 - q[i]) for i in run.pi]
    eff = [float(run.art.get_measured_joint_efforts()[0][i]) for i in run.pi]
    out.update(compression_mm=comp, spring_force_N=spring_force, measured_pad_effort_N=eff,
               carton_pushed_mm=(run.carton_y() - y0) * 1000)
    if full:
        chk.check('buffers compressed ~5 mm on all 3 pads', all(4.0 <= c <= 6.0 for c in comp), [round(c, 2) for c in comp])
        chk.check('spring force 2.0 N +-0.3 at 5 mm (200 N/m, preload 1 N)', all(abs(f - 2.0) < 0.3 for f in spring_force), [round(f, 2) for f in spring_force])
        chk.check('carton not pushed by the pressing (< 1 mm)', abs(out['carton_pushed_mm']) < 1.0, round(out['carton_pushed_mm'], 3))
    # pump on -> vacuum build-up delay -> Surface Gripper close (it engages instantly)
    run.step(int(VACUUM_DELAY_S / DT))
    tool_api.suction_close(run.iface, SG)
    run.step(int(1.0 / DT))
    st = run.status()
    objs = run.gripped()
    b1 = [str(run.stage.GetPrimAtPath(f'{TOOL}/surface_gripper_joints/pad_{i}_attachment').GetRelationship('physics:body1').GetTargets()[0]) for i in (1, 2, 3)]
    out.update(status_after_close=st, gripped=objs, attachment_body1=b1)
    if full:
        chk.check('surface gripper status Closed', st == 'Closed', st)
        chk.check('gripped object = target carton', objs == [CARTON], objs)
        chk.check('3 attachment points registered in the IsaacSurfaceGripper', len(run.stage.GetPrimAtPath(SG).GetRelationship('isaac:attachmentPoints').GetTargets()) == 3)
    # pull back 100 mm of carton travel (tool retracts 100 mm + the 5 mm buffer extension)
    y_start, by0 = run.carton_y(), run.base_y()
    forces = []
    errs = []

    events = []

    def rec(r):
        forces.append(r.slide_force())
        if r.status() != 'Closed' and not events:
            events.append(forces[-1])
        lip = r.base_y() + 0.159 + float(r.q()[r.pi[1]])
        errs.append((r.carton_y() - 0.125) - (lip - 0.0))       # carton spine (y) minus pad-2 lip y (spine = carton_y - 0.125)
    run.ramp(S_PRESS - 0.105, PULL_SPEED, rec)
    run.step(int(0.5 / DT))
    disp = (y_start - run.carton_y()) * 1000
    tool_disp = (by0 - run.base_y()) * 1000
    st2, objs2 = run.status(), run.gripped()
    out.update(first_open_event_force_N=(abs(events[0]) if events else None), carton_disp_mm=disp, tool_disp_mm=tool_disp, status_after_pull=st2, peak_slide_force_N=float(np.max(np.abs(forces))),
               max_spine_to_lip_mm=float(np.max(np.abs(errs))) * 1000)
    if full:
        chk.check('carton followed the pull: displacement 100 mm +-3', abs(disp - 100.0) <= 3.0, round(disp, 2))
        chk.check('still Closed + gripped after the pull', st2 == 'Closed' and objs2 == [CARTON], (st2, objs2))
        chk.check('spine stays on the pad lip during the pull (< 3 mm)', out['max_spine_to_lip_mm'] < 3.0, round(out['max_spine_to_lip_mm'], 3))
        out['neighbours_y_mm'] = []
        # release
    if full:
        y_rel = run.carton_y()
        tool_api.suction_open(run.iface, SG)
        run.step(int(1.0 / DT))
        run.ramp(run.s_target - 0.030, PULL_SPEED)
        run.step(int(1.0 / DT))
        mv = abs(run.carton_y() - y_rel) * 1000
        out.update(status_after_release=run.status(), carton_moved_after_release_mm=mv, gripped_after_release=run.gripped())
        chk.check('release: status Open, nothing gripped', run.status() == 'Open' and run.gripped() == [], (run.status(), run.gripped()))
        chk.check('release: carton stays (moves < 1 mm over 2 s incl. tool retract)', mv < 1.0, round(mv, 3))
        z = float(run.carton.get_world_pose()[0][2])
        chk.check('carton still resting on the shelf floor (z = h/2 +-1 mm)', abs(z - 0.110) < 0.001, round(z, 4))
    return out


def main():
    app = start_app()
    global Sdf
    from pxr import Sdf
    print('### STAGE_OPEN')
    stage, sim = open_scene(app, 'pick_test_scene.usda', DT)
    iface = tool_api.suction_interface()
    chk = Checks('left suction tool')
    run = Run(app, stage, sim, iface)
    res = {'nominal': pick(run, chk, tag='nominal')}
    def sweep(kpa, masses):
        tool_api.set_vacuum_level(stage, SG, kpa)
        co, sh = (stage.GetPrimAtPath(SG).GetAttribute(f'isaac:{n}ForceLimit').Get() for n in ('coaxial', 'shear'))
        phys_total = 3 * tool_api.physical_limits(stage, SG)[0]
        rows = []
        for m in masses:
            r = pick(run, chk, mass=m, tag=f'{kpa} kPa, {m} kg', full=False)
            r['detached'] = r['first_open_event_force_N'] is not None or r['carton_disp_mm'] < 50.0
            r['vacuum_kpa'] = kpa
            print('SWEEP', kpa, 'kPa', json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items() if k in ('mass_kg', 'status_after_pull', 'carton_disp_mm', 'peak_slide_force_N', 'first_open_event_force_N', 'detached')}))
            rows.append(r)
        held = [r for r in rows if not r['detached']]
        lost = [r for r in rows if r['detached']]
        first = min(lost, key=lambda r: r['mass_kg']) if lost else None
        return dict(vacuum_kpa=kpa, physical_total_N=phys_total, authored_coaxial_N=co, authored_shear_N=sh, rows=rows,
                    max_held_mass_kg=max([r['mass_kg'] for r in held] or [0.0]), min_detached_mass_kg=(first['mass_kg'] if first else float('nan')),
                    detach_force_N=(first['first_open_event_force_N'] if first and first['first_open_event_force_N'] is not None else float('nan')))
    sw50 = sweep(50.0, (2.0, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 8.0))
    sw25 = sweep(25.0, (1.0, 1.5, 2.0, 2.5, 3.0))
    res['sweep_50kPa'], res['sweep_25kPa'] = sw50, sw25
    r1 = pick(run, chk, mass=4.0, tag='1 pad only, 4 kg', full=False, n_attach=1)
    res['one_pad_4kg'] = r1
    for sw in (sw50, sw25):
        tag = f"{sw['vacuum_kpa']:.0f} kPa"
        e = sw['detach_force_N']
        chk.check(f'{tag}: friction pull detaches at the PHYSICAL pull-off limit 3 x coaxial = {sw["physical_total_N"]:.1f} N +-15 % (first detaching carton {sw["min_detached_mass_kg"]} kg, pull {e:.1f} N)',
                  abs(e / sw['physical_total_N'] - 1.0) <= 0.15, round(e / sw['physical_total_N'], 3))
        chk.check(f'{tag}: lighter cartons held (largest held {sw["max_held_mass_kg"]} kg), all heavier ones detach',
                  sw['max_held_mass_kg'] < sw['min_detached_mass_kg'] and all(r['detached'] for r in sw['rows'] if r['mass_kg'] >= sw['min_detached_mass_kg']))
    chk.check('lower vacuum (-25 kPa) detaches at a lower carton mass than -50 kPa', sw25['min_detached_mass_kg'] < sw50['min_detached_mass_kg'], (sw25['min_detached_mass_kg'], sw50['min_detached_mass_kg']))
    chk.check('a single attachment point (lost pad) cannot hold 4 kg (3 pads hold it below the limit)', r1['first_open_event_force_N'] is not None or r1['carton_disp_mm'] < 50.0,
              round(r1['carton_disp_mm'], 1))
    ok = chk.summary()
    json.dump(dict(res, checks=chk.rows), open(RESULTS / 'left_suction.json', 'w'), indent=1, default=float)
    sys.stdout.flush()
    import os
    app.close()
    os._exit(0 if ok else 1)


if __name__ == '__main__':
    main()
