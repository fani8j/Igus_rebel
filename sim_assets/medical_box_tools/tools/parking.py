"""Pick sequences (front / top-down) of both arms as Cartesian waypoint lists, with IK, branch-flip and collision evaluation, for a given parking pose.
Uses tools/robot_kin.py (same URDF as MoveIt, world = d200_base_footprint).  See tests/results/amr_parking.md for the study built on it."""
import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

import robot_kin as rk

CLR_ARM, CLR_TOOL_CAB, CLR_TOOL_NB, CLR_TOOL_TOOL = 0.015, 0.015, 0.003, 0.010     # m (CAD rule for tool vs the neighbouring cartons is 3 mm; 15 mm is impossible there)
STEP = 0.005
SPEEDS = dict(approach=0.020, press=0.020, pull=0.050, retract=0.050, descend=0.020, extract=0.050, lift=0.025, high=0.050)
R_SUCTION = np.column_stack([[0, 0, 1], [1, 0, 0], [0, 1, 0]]).astype(float)           # cabinet frame: tool x = +Z, y = +X, z = +Y (into the shelf)
R_GRIP_TD = np.column_stack([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).astype(float)          # tool z down, y = -Y
R_GRIP_FR = np.column_stack([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).astype(float)           # tool z = +Y, y = -Z (jaws across, pad width vertical)
SEEDS = [np.zeros(6)] + [np.array([0.0, a, b, 0.0, c, 0.0]) for a in (0.4, 0.9, 1.4) for b in (0.0, 0.6, 1.2, -0.6) for c in (0.0, 0.5, -0.5)]


def finger_q(gap):
    return (gap - 0.008) / 2.0


@dataclass
class Seg:
    name: str
    kind: str                      # 'cart' | 'joint'
    T0: np.ndarray = None
    T1: np.ndarray = None
    speed: float = 0.05
    fingers: float = None
    carton: tuple = (0.0, 0.0)     # carton pull offset (m, along -Y_c) at start/end of the segment
    pull_from: float = 0.0         # tool travel before the carton starts to move (press compression)
    q_end: np.ndarray = None


def build_sequences(world, mode, pull=None, suction_R=None, stage_y=-0.20, stage_up=0.12, left_home=False):
    R_S = R_SUCTION if suction_R is None else suction_R
    tg = world.target
    h, l, t = world.h, world.l, world.t_carton
    cx, cy, cz = tg['c']
    spine = cy - l / 2
    pull = rk.PULL_OUT if pull is None else pull
    sp_pulled = spine - pull
    TW = world.T_world
    Ps = np.array([cx, spine, cz])
    zt, yt = np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])
    pre = TW(R_S, Ps - 0.05 * zt)
    press = TW(R_S, Ps)
    pulled = TW(R_S, Ps - (pull + 0.005) * zt)
    stage_y = min(stage_y, spine - 0.05)
    stage = TW(R_S, np.array([cx, stage_y, cz]))              # TCP in free air in front of the cabinet face (tool axis along the approach line)
    L = [Seg('L0 home->stage', 'joint', T1=stage),
         Seg('L0b stage->pre', 'cart', stage, pre, SPEEDS['high']),
         Seg('L1 approach', 'cart', pre, press, SPEEDS['approach']),
         Seg('L2 press', 'cart', press, press, 0.0),
         Seg('L3 pull', 'cart', press, pulled, SPEEDS['pull'], carton=(0.0, pull), pull_from=0.005)]
    side = -1.0 if cx > 0 else 1.0
    if mode == 'front' or left_home:
        ret = TW(R_S, Ps - (pull + 0.005 + 0.100) * zt)
        L += [Seg('L4 retract 100 mm', 'cart', pulled, ret, SPEEDS['retract'], carton=(pull, pull)),
              Seg('L4b retract to stage', 'cart', ret, stage, SPEEDS['high'], carton=(pull, pull)),
              Seg('L5 ->home', 'joint', q_end=np.zeros(6), carton=(pull, pull))]
        left_final = 'home'
    else:
        r1 = TW(R_S, Ps - (pull + 0.005 + 0.030) * zt)
        aside = TW(R_S, Ps - (pull + 0.005 + 0.030) * zt + side * 0.12 * yt)
        L += [Seg('L4 retract 30 mm', 'cart', pulled, r1, SPEEDS['retract'], carton=(pull, pull)),
              Seg('L5 aside 120 mm', 'cart', r1, aside, SPEEDS['retract'], carton=(pull, pull))]
        left_final = 'aside'
    t_pre, t_cl = finger_q(t + 0.004), finger_q(t)
    y_rear = cy + l / 2 - pull
    D = y_rear + 0.02
    if mode == 'front':
        fin = np.array([cx, sp_pulled + (rk.GRIP_TCP - rk.GRIP_STOP - rk.G_NOM), cz])
        zt_g = np.array([0.0, 1.0, 0.0])
        Rg = R_GRIP_FR
        pre_g = fin - 0.100 * zt_g
        high = np.array([fin[0], min(stage_y, pre_g[1]), fin[2]])             # gripper in free air in front of the cabinet face at pad-row height
        ext = fin + np.array([0, -D, 0])
        R = [Seg('R0 home->stage', 'joint', T1=TW(Rg, high), fingers=t_pre, carton=(pull, pull)),
             Seg('R1 stage->pre', 'cart', TW(Rg, high), TW(Rg, pre_g), SPEEDS['high'], fingers=t_pre, carton=(pull, pull)),
             Seg('R2 approach 100 mm', 'cart', TW(Rg, pre_g), TW(Rg, fin), SPEEDS['approach'], fingers=t_pre, carton=(pull, pull)),
             Seg('R3 clamp', 'cart', TW(Rg, fin), TW(Rg, fin), 0.0, fingers=t_cl, carton=(pull, pull)),
             Seg('R4 extract', 'cart', TW(Rg, fin), TW(Rg, ext), SPEEDS['extract'], fingers=t_cl, carton=(pull, pull + D)),
             Seg('R5 lift 50 mm', 'cart', TW(Rg, ext), TW(Rg, ext + np.array([0, 0, 0.05])), SPEEDS['lift'], fingers=t_cl, carton=(pull + D, pull + D))]
    else:
        zb = cz - h / 2
        fin = np.array([cx, sp_pulled + 0.0425, zb + h + rk.G_NOM - (rk.GRIP_TCP - rk.GRIP_STOP)])
        Rg = R_GRIP_TD
        pre_g = fin + np.array([0, 0, 0.070])
        ext = fin + np.array([0, -D, 0])
        up = pre_g + np.array([0, 0, stage_up])
        R = [Seg('R0 home->stage', 'joint', T1=TW(Rg, up), fingers=t_pre, carton=(pull, pull)),
             Seg('R0b stage->pre', 'cart', TW(Rg, up), TW(Rg, pre_g), SPEEDS['high'], fingers=t_pre, carton=(pull, pull)),
             Seg('R1 descend 70 mm', 'cart', TW(Rg, pre_g), TW(Rg, fin), SPEEDS['descend'], fingers=t_pre, carton=(pull, pull)),
             Seg('R3 clamp', 'cart', TW(Rg, fin), TW(Rg, fin), 0.0, fingers=t_cl, carton=(pull, pull)),
             Seg('R4 extract', 'cart', TW(Rg, fin), TW(Rg, ext), SPEEDS['extract'], fingers=t_cl, carton=(pull, pull + D)),
             Seg('R5 lift 50 mm', 'cart', TW(Rg, ext), TW(Rg, ext + np.array([0, 0, 0.05])), SPEEDS['lift'], fingers=t_cl, carton=(pull + D, pull + D))]
    return dict(left=L, right=R, left_final=left_final, info=dict(pull=pull, D=D, spine=spine, mode=mode))


class Evaluator:
    def __init__(self, robot, world, lift, sl, sr, right_ev=None):
        self.right_ev = right_ev                 # optional second evaluator at another lift height for the right-arm phase (lift moved between the phases)
        self.R, self.W = robot, world
        self.lift, self.sl, self.sr = lift, sl, sr
        self.Tb = {'left': robot.arm_base('left', lift, sl), 'right': robot.arm_base('right', lift, sr)}
        self.Tb_inv = {a: np.linalg.inv(T) for a, T in self.Tb.items()}
        self.static = robot.static_points(lift, sl, sr)
        self.amr_tree = cKDTree(self.static['amr'])
        self.stand_tree = cKDTree(self.static['stand'])

    def ik_pose(self, arm, Tw, seeds):
        return self.R.ik_multi(arm, self.Tb_inv[arm] @ Tw, seeds)

    def ik_path(self, arm, T0, T1, q_start, step=STEP):
        n = max(1, int(math.ceil(np.linalg.norm(T1[:3, 3] - T0[:3, 3]) / step)))
        qs, q, mj = [], q_start, 0.0
        for i in range(1, n + 1):
            T = T0.copy()
            T[:3, 3] = T0[:3, 3] + (T1[:3, 3] - T0[:3, 3]) * i / n
            Tb = self.Tb_inv[arm] @ T
            qn = self.R.ik(arm, Tb, q)
            if qn is None:
                qn = self.R.ik_multi(arm, Tb, [q] + SEEDS[:4])
                if qn is None:
                    return qs, False, f'no IK at {i}/{n}', mj
            j = float(np.max(np.abs(qn - q)))
            mj = max(mj, j)
            if j > 0.12:
                return qs, False, f'branch flip at {i}/{n} (jump {j:.2f} rad)', mj
            qs.append(qn)
            q = qn
        return qs, True, '', mj


    TRANSIT_ORDERS = ([0, 1, 2, 3, 4, 5], [1, 2, 4, 0, 3, 5], [0], [1, 2], [2, 1, 4], [1, 2, 4], [4, 3, 5], [3, 5])

    def _leg_ok(self, arm, qa, qb, fingers, other_q, other_fingers, carton, n=10):
        for i in range(0, n + 1):
            qq = qa + (qb - qa) * (i / n)
            c = self.clearances(arm, qq, fingers, other_q, other_fingers, carton)
            if self.check(c):
                return False
        return True

    def transit(self, arm, q0, q1, fingers, other_q, other_fingers, carton, collisions=True, n=24):
        """free-space joint-space transit q0 -> q1.  Stand-in for the MoveIt free-space planner: direct joint interpolation, then joint-grouped orderings, then a few random via points
        (all collision checked); returns the sample list or None"""
        direct = [q0 + (q1 - q0) * (i / n) for i in range(1, n + 1)]
        if not collisions:
            return direct
        if self._leg_ok(arm, q0, q1, fingers, other_q, other_fingers, carton, 12):
            return direct
        cands = []
        for order in self.TRANSIT_ORDERS[2:]:
            qm = q0.copy()
            qm[order] = q1[order]
            cands.append([qm])
        rng = np.random.default_rng(7)
        for _ in range(12):
            qm = q0 + (q1 - q0) * rng.uniform(0.2, 0.8) + rng.normal(0, 0.5, 6) * np.array([0.3, 1.0, 1.0, 0.5, 0.8, 0.3])
            qm = np.clip(qm, self.R.qmin[arm] + 0.02, self.R.qmax[arm] - 0.02)
            cands.append([qm])
        for via in cands:
            pts = [q0] + via + [q1]
            if all(self._leg_ok(arm, pts[i], pts[i + 1], fingers, other_q, other_fingers, carton, 10) for i in range(len(pts) - 1)):
                out = []
                per = max(2, n // (len(pts) - 1))
                for i in range(len(pts) - 1):
                    out += [pts[i] + (pts[i + 1] - pts[i]) * (k / per) for k in range(1, per + 1)]
                return out
        return None

    def clearances(self, arm, q, fingers, other_q, other_fingers, carton_pull):
        oth = 'right' if arm == 'left' else 'left'
        A, Tt = self.R.arm_points(arm, self.Tb[arm], q, fingers if arm == 'right' else None)
        Ao, To = self.R.arm_points(oth, self.Tb[oth], other_q, other_fingers if oth == 'right' else None)
        A, Tt = A[::2], Tt[::2]
        out = {}
        d = self.W.static_dist(A)
        out['arm_cabinet'] = min(v for k, v in d.items() if k.startswith('cab:'))
        out['arm_blocks'] = min([v for k, v in d.items() if k.startswith('carton:')] or [9.9])
        out['arm_carton'] = self.W.carton_dist(A, carton_pull)
        d = self.W.static_dist(Tt)
        out['tool_cabinet'] = min(v for k, v in d.items() if k.startswith('cab:'))
        out['tool_neighbours'] = min([v for k, v in d.items() if k.startswith('carton:')] or [9.9])
        P = np.vstack([A[len(A) // 4:], Tt])
        out['robot_amr'] = float(self.amr_tree.query(P[::2])[0].min())
        out['robot_stand'] = float(self.stand_tree.query(P[::2])[0].min())
        o_all = cKDTree(np.vstack([Ao, To]))
        out['arm_arm'] = float(o_all.query(A[::2])[0].min())
        out['tool_tool'] = float(cKDTree(To).query(Tt[::2])[0].min())
        return out

    def check(self, c):
        lim = dict(arm_cabinet=CLR_ARM, arm_blocks=CLR_ARM, arm_carton=CLR_ARM, tool_cabinet=CLR_TOOL_CAB, tool_neighbours=CLR_TOOL_NB, robot_amr=CLR_ARM,
                   robot_stand=CLR_ARM, arm_arm=CLR_ARM, tool_tool=CLR_TOOL_TOOL)
        return [k for k, v in lim.items() if c[k] < v]

    def ik_all(self, arm, Tw, seeds, tol=0.25, max_n=8):
        Tb = self.Tb_inv[arm] @ Tw
        sols = []
        for sd in seeds:
            q = self.R.ik(arm, Tb, sd)
            if q is not None and all(np.max(np.abs(q - o)) > tol for o in sols):
                sols.append(q)
        sols.sort(key=lambda q: float(np.linalg.norm(q)))
        return sols[:max_n]

    def run_arm(self, arm, segs, other_q, other_fingers, record=False, collisions=True):
        """tries the IK branches of the first joint-space target (stage pose) in turn; returns the first branch whose whole sequence passes"""
        first = segs[0]
        if first.kind == 'joint' and first.q_end is None and first.T1 is not None:
            sols = self.ik_all(arm, first.T1, SEEDS)
            if not sols:
                return dict(ok=False, reason=f'{first.name}: no IK', clr={}, jmargin=9.9)
            best = None
            for q1 in sols:
                import copy
                seg2 = copy.copy(first)
                seg2.q_end = q1
                r = self._run_arm([seg2] + list(segs[1:]), arm, other_q, other_fingers, record, collisions)
                if r['ok']:
                    r['branch'] = [float(x) for x in q1]
                    return r
                if best is None:
                    best = r
            best['reason'] += f' (all {len(sols)} IK branches of the stage pose tried)'
            return best
        return self._run_arm(segs, arm, other_q, other_fingers, record, collisions)

    def _run_arm(self, segs, arm, other_q, other_fingers, record=False, collisions=True):
        q = np.zeros(6)
        traj = []
        minc, jmargin, maxjump = {}, 9.9, 0.0
        fin_prev = 0.024
        key_q = {}
        for sg in segs:
            if sg.kind == 'joint':
                if sg.q_end is not None:
                    q1 = sg.q_end
                else:
                    q1 = self.ik_pose(arm, sg.T1, [q] + SEEDS)
                    if q1 is None:
                        return dict(ok=False, reason=f'{sg.name}: no IK', clr=minc, jmargin=jmargin)
                qs = self.transit(arm, q, q1, f_tr := (sg.fingers if sg.fingers is not None else fin_prev), other_q, other_fingers, sg.carton[0], collisions)
                if qs is None:
                    return dict(ok=False, reason=f'{sg.name}: collision in every tried joint-space transit (direct, joint-ordered, random via points)', clr=minc, jmargin=jmargin)
                n = len(qs)
                carton = [sg.carton[0]] * n
            else:
                if sg.speed > 0:
                    qs, ok, why, mj = self.ik_path(arm, sg.T0, sg.T1, q)
                    maxjump = max(maxjump, mj)
                    if not ok:
                        return dict(ok=False, reason=f'{sg.name}: {why}', clr=minc, jmargin=jmargin)
                else:
                    qs = [q.copy(), q.copy()]
                n = len(qs)
                tot = np.linalg.norm(sg.T1[:3, 3] - sg.T0[:3, 3])
                carton = []
                for i in range(n):
                    s = (i + 1) / n
                    if sg.name.startswith('L3'):
                        trav = tot * s
                        carton.append(max(0.0, trav - sg.pull_from))
                    else:
                        carton.append(sg.carton[0] + (sg.carton[1] - sg.carton[0]) * s)
            f = sg.fingers if sg.fingers is not None else fin_prev
            fin_prev = f
            key_q[sg.name] = qs[-1]
            if collisions:
                for i in list(range(0, len(qs), 3)) + [len(qs) - 1]:
                    qq = qs[i]
                    jmargin = min(jmargin, float(np.min(np.minimum(qq - self.R.qmin[arm], self.R.qmax[arm] - qq))))
                    c = self.clearances(arm, qq, f, other_q, other_fingers, carton[i])
                    for k, v in c.items():
                        minc[k] = min(minc.get(k, 9.9), v)
                    bad = self.check(c)
                    if bad:
                        return dict(ok=False, reason=f'{sg.name}: collision {bad} {[round(c[b], 3) for b in bad]}', clr=minc, jmargin=jmargin)
            if record:
                traj.append(dict(name=sg.name, kind=sg.kind, q=np.array(qs), fingers=f, carton=np.array(carton), speed=sg.speed))
            q = qs[-1]
        res = dict(ok=True, reason='', clr=minc, jmargin=jmargin, maxjump=maxjump, q_final=q, key_q=key_q)
        if record:
            res['traj'] = traj
        return res

    def evaluate(self, seqs, record=False, collisions=True):
        """seqs: one build_sequences() dict or a list of them (stage-pose variants, tried in order per arm; the first variant whose whole sequence passes is used)"""
        variants = [seqs] if isinstance(seqs, dict) else list(seqs)
        home = np.zeros(6)
        rl, vl = None, None
        for k, sq in enumerate(variants):
            r = self.run_arm('left', sq['left'], home, 0.024, record, collisions)
            if rl is None:
                rl = r
            if r['ok']:
                rl, vl = r, k
                break
        if vl is None:
            return dict(ok=False, reason='left: ' + rl['reason'], left=rl)
        qL = home if variants[vl]['left_final'] == 'home' else rl['q_final']
        rr, vr = None, None
        for k, sq in enumerate(variants):
            r = (self.right_ev or self).run_arm('right', sq['right'], qL, 0.024, record, collisions)
            if rr is None:
                rr = r
            if r['ok']:
                rr, vr = r, k
                break
        if vr is None:
            return dict(ok=False, reason='right: ' + rr['reason'], left=rl, right=rr)
        return dict(ok=True, reason='', left=rl, right=rr, stage_variant=(vl, vr), left_seq=variants[vl]['left'], right_seq=variants[vr]['right'])

    def screen_arm(self, arm, segs):
        """cheap reach screen: IK of the key poses in sequence (no path, no collision)"""
        prev = None
        for s in segs:
            T = s.T1 if s.kind == 'joint' else s.T1
            if T is None:
                continue
            q = self.ik_pose(arm, T, SEEDS[:8] if prev is None else [prev] + SEEDS[:6])
            if q is None:
                return False
            prev = q
        return True


STAGE_VARIANTS = ((-0.20, 0.12), (-0.10, 0.08), (0.0, 0.04), (9.0, 0.0))        # (suction / gripper-front stage y in the cabinet frame, gripper top-down stage height); 9.0 = stage at the pre pose


def build_all(world, mode, pull=None, suction_R=None, left_home=False):
    return [build_sequences(world, mode, pull, suction_R, stage_y=sy, stage_up=su, left_home=left_home) for sy, su in STAGE_VARIANTS]
