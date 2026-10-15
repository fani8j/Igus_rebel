"""Wrapper around the offline study code (read-only import of holder_camera_fingers_20261002) that returns, per carton placement,
the study's verdict, allocation / pairing and the best joint paths + TCP poses of every task.  Pure numpy, no ROS.

  from study_eval import StudyEval
  se = StudyEval('h000n.0')                       # design json label under designs/ (camera + fingers) or 'baseline'
  data = se.evaluate(layout=(0.0, 0.40, 0.355), scenarios=[('BIG370', 0, 0.0, 0.0), ...])
"""
import os, sys, json, math, time, pickle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = '/home/farhan/igus_rebel_ros2/scan_results/holder_camera_fingers_20261002'
sys.path.insert(0, HERE)
sys.path.insert(0, STUDY)
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import cellgeom as G
from common import *                      # study: dalib (*), des_from_params, OUTDIR ...
from fulleval import eval_scenarios, build_cands
import camlib
import fingerlib as FL
import dalib as D
from dalib import stroke_paths, ALLOC_INFO, ALLOCS, TASK_I, NPTS, Scen, StrokeSet, task_targets, ORI15
from reach_lib import fk_flange, ik_flange, LIM_LO, LIM_HI

ARM_NAME = ['arm1', 'arm2']
NOMINAL_ORI_IDX = [i for i, (a, b) in enumerate(ORI15) if abs(a) < 1e-12 and abs(b) < 1e-12][0]


def pose_list(T):
    return [G.T_to_pose(t) for t in T]


def pair_first(PA, rA, mA, PB, rB, mB, clr=None, chunk=24, CA=None, CB=None):
    """copy of dalib.pair_any that returns the first collision-free combination (index into A's and B's candidate arrays) or None.
    Combinations are tried best-margin first exactly like the study does."""
    from dalib import PAIR_OFFS, CLR_ARM
    clr = CLR_ARM if clr is None else clr
    KA, KB = PA.shape[0], PB.shape[0]
    oa = np.argsort(-mA); ob = np.argsort(-mB)
    ra = np.empty(KA, int); ra[oa] = np.arange(KA); rb = np.empty(KB, int); rb[ob] = np.arange(KB)
    IA, IB = np.meshgrid(np.arange(KA), np.arange(KB), indexing='ij')
    IA = IA.ravel(); IB = IB.ravel()
    o = np.argsort(ra[IA] + rb[IB], kind='stable'); IA = IA[o]; IB = IB[o]
    thr2 = (rA[:, None] + rB[None, :] + clr) ** 2
    if CA is not None:
        import camlib as CL
    for c0 in range(0, len(IA), chunk):
        ia = IA[c0:c0 + chunk]; ib = IB[c0:c0 + chunk]
        ok = np.ones(len(ia), bool)
        for (i, j) in PAIR_OFFS:
            a = PA[ia, i]; b = PB[ib, j]
            d2 = ((a[:, :, None, :] - b[:, None, :, :]) ** 2).sum(-1)
            ok &= np.all(d2 >= thr2[None], axis=(1, 2))
            if CA is not None and ok.any():
                n = len(ia)
                hb = np.broadcast_to(CL.CAM_HALF, (n, 3))
                sd = CL.sd_sphere_obb(b, CA[0][ia, i][:, None, :], CA[1][ia, i][:, None, :, :], hb[:, None, :])
                ok &= np.all(sd >= rB[None, :] + clr, axis=1)
                sd = CL.sd_sphere_obb(a, CB[0][ib, j][:, None, :], CB[1][ib, j][:, None, :, :], hb[:, None, :])
                ok &= np.all(sd >= rA[None, :] + clr, axis=1)
                ok &= ~CL.obb_overlap(CA[0][ia, i], CA[1][ia, i], hb, CB[0][ib, j], CB[1][ib, j], hb + clr)
            if not ok.any():
                break
        if ok.any():
            k = int(np.argmax(ok))
            return int(ia[k]), int(ib[k])
    return None


class StudyEval:
    def __init__(self, label):
        self.label = label
        des_json = json.load(open(os.path.join(HERE, 'designs', label + '.json')))
        self.design = des_json
        if label == 'baseline':
            self.des = D.baseline_design()
            self.cam = None
        else:
            self.des = des_from_params(des_json['params'], fingers=True)
            combos = json.load(open(STUDY + '/combos_top.json'))
            item = next(c for c in combos if c['label'] == label)
            self.cam = camlib.Cam(np.asarray(item['o']), np.asarray(item['Rc']), np.asarray(item['brp']))
            self.des.cam = self.cam
        self.T_tcp = np.asarray(self.des.T, float)
        self.T_tcp_inv = np.linalg.inv(self.T_tcp)

    # ------------------------------------------------------------------ helpers
    def ik_branches(self, Tw, arm_i, s):
        """closed-form IK of the TCP pose Tw (world, 4x4) for arm index arm_i -> list of (q (6,), margin_deg), valid ones with margin >= 0, best first"""
        T6 = Tw @ self.T_tcp_inv
        T6 = T6.copy()
        T6[1, 3] -= G.arm_y(arm_i, s)
        q, valid = ik_flange(T6[None])
        out = []
        for b in range(q.shape[1]):
            if not valid[0, b]:
                continue
            qd = q[0, b] / D.D2R
            m = float(np.min(np.minimum(qd - LIM_LO, LIM_HI - qd)))
            if m >= 0:
                out.append((q[0, b], m))
        out.sort(key=lambda x: -x[1])
        return out

    def tcp_world_from_q(self, q5, arm_i, s):
        T6 = fk_flange(np.asarray(q5, float))
        T = T6 @ self.T_tcp
        T[:, 1, 3] += G.arm_y(arm_i, s)
        return T

    def best_candidate(self, SS, si, arm, task, mcol, rear, thr):
        """replicates fulleval.build_cands but also returns the stroke index of the best path: (q (5,6), margin, idx) or None"""
        base = np.ravel_multi_index((si, arm, TASK_I[task], 0, 0), SS.shape)
        nstroke = SS.shape[3] * SS.shape[4]
        idx = base + np.arange(nstroke)
        m = mcol[idx]
        sel = idx[m >= thr]
        if len(sel) == 0:
            return None
        bc, bj, qp = stroke_paths(SS, self.des, sel, rear)
        good = np.where(bc >= thr)[0]
        if len(good) == 0:
            return None
        k = good[int(np.argmax(bc[good]))]
        return qp[k], float(bc[k]), int(sel[k])

    # ------------------------------------------------------------------ main
    def evaluate(self, layout, scenarios, rear=0.0, log=print):
        h, d, s = layout
        sc = [Scen(h, d, s, b, y, dx, dy) for (b, y, dx, dy) in scenarios]
        t0 = time.time()
        res = eval_scenarios(sc, self.des, self.des, rear=rear, do_pairs=True, detail=True, thr_list=(5.0,))
        log('study eval_scenarios: %d scenarios in %.0f s' % (len(sc), time.time() - t0))
        SS = res['SS']
        mcol = res['_mcol_flat']
        Mc = res['Mc']
        out = []
        for si, s_ in enumerate(sc):
            rec = dict(box=s_.box, yaw=s_.yaw, dx=s_.dx, dy=s_.dy, valid_layout=bool(SS.valid[si]), feasible=bool(res['feas'][5.0][si]),
                       alloc_str=res['alloc'][5.0][si], allocs_ok=list(res['allocs_ok'][5.0][si]),
                       task_margin={ARM_NAME[a]: {t: (None if Mc[si, a, TASK_I[t]] < -1e8 else round(float(Mc[si, a, TASK_I[t]]), 2)) for t in D.TASKS} for a in (0, 1)})
            if rec['feasible']:
                nm, pi = rec['alloc_str'].split('/')
                pi = int(pi)
                rec['alloc'] = nm
                rec['pairing'] = pi
            else:
                # best-effort allocation = fewest individually failing tasks (same rule as peryaw.per_yaw_table)
                best = None
                for nm in ALLOCS:
                    a0, a1, _ = ALLOC_INFO[nm]
                    fails = [(0, t) for t in a0 if Mc[si, 0, TASK_I[t]] < 5.0] + [(1, t) for t in a1 if Mc[si, 1, TASK_I[t]] < 5.0]
                    if best is None or len(fails) < len(best[1]):
                        best = (nm, fails)
                rec['alloc'] = best[0]
                rec['pairing'] = 0
                rec['best_effort'] = True
                rec['failing_tasks'] = ['%s:%s' % (ARM_NAME[a], t) for (a, t) in best[1]]
            a0, a1, pairings = ALLOC_INFO[rec['alloc']]
            # study pair rule: the chosen allocation is feasible because SOME combination of candidate paths of each concurrent pair is clear at the paired samples
            # (k,k), (k,k+-1).  Use that combination (not simply the best-margin path of each task) so the job reproduces the study's own pair verdict.
            chosen_idx = {}
            if rec['feasible']:
                from fulleval import build_cands as _bc
                cand_objs = {}
                for ai_, tl_ in ((0, a0), (1, a1)):
                    for t_ in tl_:
                        cand_objs[(ai_, t_)] = _bc(SS, self.des, si, ai_, t_, mcol, rear, 5.0)
                for (x, y) in pairings[rec['pairing']]:
                    cx, cy = cand_objs.get(x), cand_objs.get(y)
                    if cx is None or cy is None:
                        continue
                    fx, fy = chosen_idx.get(x), chosen_idx.get(y)
                    ax_ = (cx.m >= 5.0) if fx is None else (np.arange(len(cx.m)) == fx)
                    ay_ = (cy.m >= 5.0) if fy is None else (np.arange(len(cy.m)) == fy)
                    ix = np.where(ax_)[0]; iy = np.where(ay_)[0]
                    if len(ix) == 0 or len(iy) == 0:
                        continue
                    r_ = pair_first(cx.P[ix], cx.r, cx.m[ix], cy.P[iy], cy.r, cy.m[iy],
                                    CA=None if cx.CC is None else (cx.CC[ix], cx.CR[ix]), CB=None if cy.CC is None else (cy.CC[iy], cy.CR[iy]))
                    if r_ is not None:
                        chosen_idx[x] = int(ix[r_[0]]); chosen_idx[y] = int(iy[r_[1]])
                rec['pair_choice'] = {'%s:%s' % (ARM_NAME[k[0]], k[1]): v for k, v in chosen_idx.items()}
            tasks = {}
            for ai, tl in ((0, a0), (1, a1)):
                for t in tl:
                    tid = '%s:%s' % (ARM_NAME[ai], t)
                    cand = None
                    src = None
                    if (ai, t) in chosen_idx:
                        co = cand_objs[(ai, t)]
                        kk = chosen_idx[(ai, t)]
                        cand = (co.q[kk], float(co.m[kk]), -1)
                        src = 'study_path_5deg_pair_consistent'
                    for thr, nme in ((5.0, 'study_path_5deg'), (0.0, 'study_path_0deg')):
                        if cand is not None:
                            break
                        cand = self.best_candidate(SS, si, ai, t, mcol, rear, thr) if Mc[si, ai, TASK_I[t]] >= thr else None
                        if cand is not None:
                            src = nme
                            break
                    ti = TASK_I[t]
                    if cand is not None:
                        q5, marg, sidx = cand
                        T5 = self.tcp_world_from_q(q5, ai, s)
                        info = dict(source=src, margin=marg, q=q5.tolist())
                    else:
                        # nominal stroke (direction 0, tilt 0): poses from the StrokeSet, no joint path
                        sidx = np.ravel_multi_index((si, ai, ti, 0, NOMINAL_ORI_IDX), SS.shape)
                        P = SS.Pf[sidx].copy()
                        P[:, 1] += G.arm_y(ai, s)
                        Rn = SS.Rf[sidx]
                        T5 = np.tile(np.eye(4), (NPTS, 1, 1))
                        T5[:, :3, :3] = Rn
                        T5[:, :3, 3] = P
                        info = dict(source='nominal_no_study_path', margin=None, q=None)
                    info['poses5'] = pose_list(T5)
                    info['stroke_idx'] = int(sidx)
                    # alternative IK branches (closed form, joint-limit margin >= 0) for the start pose
                    info['branches_start'] = [(q.tolist(), m) for q, m in self.ik_branches(T5[0], ai, s)[:6]]
                    info['branches_end'] = [(q.tolist(), m) for q, m in self.ik_branches(T5[-1], ai, s)[:6]]
                    # outward surface normal at the stroke (tool Z = push direction into the material)
                    info['push'] = [float(v) for v in T5[0][:3, 2]]
                    tasks[tid] = info
            rec['tasks'] = tasks
            # phases from the pairing (pairs first in listed order, leftover tasks alone afterwards, in each arm's own order)
            pr = pairings[rec['pairing']]
            phases, used = [], set()
            for (x, y) in pr:
                tx, ty = '%s:%s' % (ARM_NAME[x[0]], x[1]), '%s:%s' % (ARM_NAME[y[0]], y[1])
                phases.append([tx, ty])
                used |= {tx, ty}
            for tl_arm, tl in ((0, a0), (1, a1)):
                for t in tl:
                    tid = '%s:%s' % (ARM_NAME[tl_arm], t)
                    if tid not in used:
                        phases.append([tid])
                        used.add(tid)
            rec['phases'] = phases
            out.append(rec)
        return out, res


def cached_evaluate(label, layout, scenarios, cache_dir, rear=0.0, tag='', log=print):
    os.makedirs(cache_dir, exist_ok=True)
    key = '%s_%s_%s%s' % (label, '_'.join('%g' % v for v in layout), 'r%g' % rear, tag)
    f = os.path.join(cache_dir, 'study_%s.json' % key)
    want = [list(x) for x in scenarios]
    if os.path.exists(f):
        c = json.load(open(f))
        if c['scenarios'] == want:
            log('study cache hit', f)
            return c['data']
    se = StudyEval(label)
    data, _ = se.evaluate(layout, scenarios, rear=rear, log=log)
    json.dump(dict(label=label, layout=list(layout), scenarios=want, data=data), open(f, 'w'))
    return data
