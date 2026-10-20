"""Headless Isaac Sim 5.1 test of the right gripper (SMC LEHF20K2-48): open, close on a 16 mm and a 12 mm carton, lift 50 mm, release.
Run:  nice -n 19 /home/farhan/isaac-sim/python.sh tests/test_right_gripper.py   (writes tests/results/right_gripper.json)
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
TOOL = '/World/Tools/right_gripper_tool'
CARR = TOOL + '_carrier'
PULL_OUT = 0.100
CARTONS = {'16 mm (box 3, idx 2)': ('/World/Cartons/carton_2_accuforce', 0.016, 0.220),
           '12 mm (box 5, idx 3)': ('/World/Cartons/carton_3_conqueror_nc_pro', 0.012, 0.230)}
MAX_F = 24.3


def shift_xform(stage, path, d):
    """Translate a prim authored with a single xformOp:transform by d (world, metres)."""
    from pxr import Gf, UsdGeom
    x = UsdGeom.Xformable(stage.GetPrimAtPath(path))
    op = x.GetOrderedXformOps()[0]
    m = op.Get()
    m.SetTranslateOnly(m.ExtractTranslation() + Gf.Vec3d(*d))
    op.Set(m)


class G:
    def __init__(self, stage, sim):
        self.stage, self.sim = stage, sim
        self.orig = {}

    def reset(self, carton_path, h, dx):
        from pxr import Gf
        from isaacsim.core.prims import SingleRigidPrim, Articulation
        self.sim.stop()
        st = self.stage
        for p in ('/World/Tools/left_suction_tool', '/World/Tools/left_suction_tool_carrier'):    # the left tool stands in the pulled-out carton's way
            st.GetPrimAtPath(p).SetActive(False)
        for path, d in self.orig.items():                  # undo previous edits
            pass
        # pull the carton 100 mm out of the shelf (it keeps resting on the floor); fresh each reset from the saved original
        if not self.orig:
            from pxr import UsdGeom
            self.orig['carton'] = {p: UsdGeom.Xformable(st.GetPrimAtPath(p)).GetOrderedXformOps()[0].Get() for p, _, _ in [(v[0], 0, 0) for v in CARTONS.values()]}
            self.orig['grip'] = {p: (UsdGeom.Xformable(st.GetPrimAtPath(p)).GetOrderedXformOps()[0].Get() if 'carrier' in p or p == TOOL else None) for p in (TOOL, CARR)}
            self.orig['rj'] = st.GetPrimAtPath(CARR + '/root_joint').GetAttribute('physics:localPos0').Get()
        from pxr import UsdGeom
        for p, m in self.orig['carton'].items():
            UsdGeom.Xformable(st.GetPrimAtPath(p)).GetOrderedXformOps()[0].Set(m)
        for p, m in self.orig['grip'].items():
            UsdGeom.Xformable(st.GetPrimAtPath(p)).GetOrderedXformOps()[0].Set(m)
        st.GetPrimAtPath(CARR + '/root_joint').GetAttribute('physics:localPos0').Set(self.orig['rj'])
        shift_xform(st, carton_path, (0, -PULL_OUT, 0))
        dz = h - 0.220                                        # gripper stop plane follows the carton top edge
        for p in (TOOL, CARR):
            shift_xform(st, p, (dx, 0, dz))
        st.GetPrimAtPath(CARR + '/root_joint').GetAttribute('physics:localPos0').Set(self.orig['rj'] + Gf.Vec3f(dx, 0, dz))
        self.sim.initialize_physics()
        self.sim.play()
        self.carton = SingleRigidPrim(carton_path)
        self.base = SingleRigidPrim(TOOL + '/base')
        self.art = Articulation(CARR)
        self.art.initialize()
        n = self.art.dof_names
        self.si, self.il, self.ir = n.index('world_slide'), n.index('finger_left_joint'), n.index('finger_right_joint')
        self.s = -0.120
        self.trace = []
        self.step(120)

    def step(self, n=1):
        for _ in range(n):
            self.sim.step(render=False)
            q = self.art.get_joint_positions()[0]
            self.trace.append((q[self.il], q[self.ir]))

    def set_s(self, s):
        self.s = s
        self.art.set_joint_position_targets(np.array([[s]]), joint_indices=np.array([self.si]))

    def fingers(self, q):
        self.art.set_joint_position_targets(np.array([[q, q]]), joint_indices=np.array([self.il, self.ir]))

    def q(self):
        return self.art.get_joint_positions()[0]

    def eff(self):
        e = self.art.get_measured_joint_efforts()[0]
        return float(e[self.il]), float(e[self.ir])

    def gap(self):
        q = self.q()
        return 0.008 + q[self.il] + q[self.ir]

    def ramp(self, s_to, speed):
        n = max(1, int(abs(s_to - self.s) / speed / DT))
        s0 = self.s
        for k in range(n):
            self.set_s(s0 + (s_to - s0) * (k + 1) / n)
            self.step(1)
        self.set_s(s_to)


def run_one(g, chk, label, path, t, h, dx):
    g.reset(path, h, dx)
    out = dict(label=label, thickness_mm=t * 1000)
    tag = label
    # 1. open
    g.fingers(tool_api.GRIP_OPEN_Q)
    g.step(int(0.5 / DT))
    out['mimic'] = bool(g.stage.GetPrimAtPath(TOOL + '/joints/finger_right_joint').HasAPI(__import__('pxr').PhysxSchema.PhysxMimicJointAPI, 'rotX'))
    q = g.q()
    out['open_q_mm'] = [q[g.il] * 1000, q[g.ir] * 1000]
    out['open_gap_mm'] = g.gap() * 1000
    chk.check(f'[{tag}] open: pad gap 56 mm +-0.5', abs(g.gap() * 1000 - 56.0) <= 0.5, round(g.gap() * 1000, 3))
    # 2. descend (tool z down) to the handoff pose
    z0 = float(g.carton.get_world_pose()[0][2])
    y0 = float(g.carton.get_world_pose()[0][1])
    g.ramp(0.0, 0.040)
    g.step(int(0.3 / DT))
    chk.check(f'[{tag}] descent did not disturb the carton (< 1 mm)', abs(float(g.carton.get_world_pose()[0][1]) - y0) < 0.001 and abs(float(g.carton.get_world_pose()[0][2]) - z0) < 0.001)
    # 3. close: fast to a 32 mm gap, then push at the catalog pushing speed to a target past the stop -> the drive force saturates
    i0 = len(g.trace)
    for q in tool_api.finger_target_profile(tool_api.GRIP_OPEN_Q, tool_api.GRIP_CLOSE_Q, DT, push=True):
        g.fingers(q)
        g.step(1)
    g.step(int(0.25 / DT))
    effs = []
    for _ in range(int(0.25 / DT)):          # sample the finger joint efforts over the hold
        g.step(1)
        effs.append(g.eff())
    effs = np.abs(np.array(effs))
    tr = np.array(g.trace[i0:])
    gap_tr = 0.008 + tr[:, 0] + tr[:, 1]
    final_gap = g.gap()
    out['final_gap_mm'] = final_gap * 1000
    out['min_gap_mm_during_close'] = float(gap_tr.min() * 1000)
    out['gap_jitter_last_0.5s_um'] = float(np.ptp(gap_tr[-120:]) * 1e6)
    vel = np.abs(np.diff(tr, axis=0)) / DT
    out['max_finger_speed_mm_s'] = float(vel.max() * 1000)
    chk.check(f'[{tag}] final pad gap = carton thickness {t * 1000:.0f} mm +-0.5', abs(final_gap - t) <= 0.0005, round(final_gap * 1000, 3))
    chk.check(f'[{tag}] carton centred between the fingers (|q_left - q_right| < 0.3 mm)', abs(g.q()[g.il] - g.q()[g.ir]) < 0.0003, round((g.q()[g.il] - g.q()[g.ir]) * 1000, 3))
    chk.check(f'[{tag}] no overshoot below the carton (min gap within 0.5 mm of final)', out['min_gap_mm_during_close'] >= final_gap * 1000 - 0.5, round(out['min_gap_mm_during_close'], 3))
    chk.check(f'[{tag}] no jitter (gap peak-to-peak over the last 0.5 s < 50 um)', out['gap_jitter_last_0.5s_um'] < 50.0, round(out['gap_jitter_last_0.5s_um'], 2))
    el, er = g.eff()
    out['grip_force_measured_N'] = [el, er]
    out['grip_force_hold_mean_N'] = effs.mean(0).tolist()
    out['grip_force_hold_max_N'] = effs.max(0).tolist()
    # the drive limit is 24.3 N per finger: the mean over the hold must equal it (+-2 %), no sample may exceed it by more than 2 % (solver noise); the SUM is limited by the master drive (48.6 N)
    chk.check(f'[{tag}] grip force per finger: hold mean {effs.mean(0).round(2).tolist()} N = 24.3 N +-2 %, max sample {effs.max(0).round(2).tolist()} N <= 24.3 x 1.04 (per-finger solver noise, the sum is the hard limit below), and >= 11 N',
              all(abs(m - MAX_F) <= 0.02 * MAX_F for m in effs.mean(0)) and all(x <= MAX_F * 1.04 for x in effs.max(0)) and all(m >= 11.0 for m in effs.mean(0)))
    chk.check(f'[{tag}] never above the limit: sum of both fingers <= 2 x 24.3 N x 1.02', float(effs.sum(1).max()) <= 2 * MAX_F * 1.02, round(float(effs.sum(1).max()), 2))
    out['max_finger_speed_ok'] = out['max_finger_speed_mm_s'] <= 45.0
    chk.check(f'[{tag}] finger speed <= 45 mm/s (max {out["max_finger_speed_mm_s"]:.1f})', out['max_finger_speed_mm_s'] <= 45.0)
    # 4. lift 50 mm
    cz0 = float(g.carton.get_world_pose()[0][2])
    bz0 = float(g.base.get_world_pose()[0][2])
    g.ramp(-0.050, 0.025)
    g.step(int(0.7 / DT))
    cz1 = float(g.carton.get_world_pose()[0][2])
    bz1 = float(g.base.get_world_pose()[0][2])
    lift_c, lift_g = (cz1 - cz0) * 1000, (bz1 - bz0) * 1000
    out.update(carton_lift_mm=lift_c, gripper_lift_mm=lift_g, slip_mm=lift_g - lift_c)
    chk.check(f'[{tag}] lifted 50 mm: slip <= 2 mm', abs(lift_g - lift_c) <= 2.0 and lift_g > 49.0, dict(carton=round(lift_c, 2), gripper=round(lift_g, 2), slip=round(lift_g - lift_c, 3)))
    # 5. set down, open, retract
    g.ramp(0.0, 0.025)
    g.step(int(0.3 / DT))
    for q in tool_api.finger_target_profile(tool_api.GRIP_CLOSE_Q, tool_api.GRIP_OPEN_Q, DT):
        g.fingers(q)
        g.step(1)
    g.step(int(0.3 / DT))
    chk.check(f'[{tag}] open again: pad gap 56 mm +-0.5', abs(g.gap() * 1000 - 56.0) <= 0.5, round(g.gap() * 1000, 3))
    cz2 = float(g.carton.get_world_pose()[0][2])
    g.ramp(-0.060, 0.040)
    g.step(int(0.5 / DT))
    cz3 = float(g.carton.get_world_pose()[0][2])
    out['carton_z_after_release_mm'] = cz3 * 1000
    chk.check(f'[{tag}] released: carton stays on the shelf floor while the gripper rises (z = h/2 +-1 mm)', abs(cz3 - h / 2) < 0.001 and abs(cz3 - cz2) < 0.001, round(cz3, 4))
    return out


def main():
    app = start_app()
    print('### STAGE_OPEN')
    stage, sim = open_scene(app, 'pick_test_scene.usda', DT)
    chk = Checks('right gripper')
    g = G(stage, sim)
    res = {}
    cx = {'16 mm (box 3, idx 2)': 0.0, '12 mm (box 5, idx 3)': None}
    xs = {l: float(stage.GetPrimAtPath(v[0]).GetAttribute('xformOp:transform').Get().ExtractTranslation()[0]) for l, v in CARTONS.items()}
    for label, (path, t, h) in CARTONS.items():
        x = xs[label]
        res[label] = run_one(g, chk, label, path, t, h, x)
    ok = chk.summary()
    json.dump(dict(res, checks=chk.rows), open(RESULTS / 'right_gripper.json', 'w'), indent=1, default=float)
    sys.stdout.flush()
    app.close()
    os._exit(0 if ok else 1)


if __name__ == '__main__':
    main()
