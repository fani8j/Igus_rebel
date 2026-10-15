#!/usr/bin/env python3
"""Planner-level validation of the dual-arm cut cell in the real MoveIt 2 (Humble) planner (planning only, nothing is executed).

  export ROS_DOMAIN_ID=187 ROS_LOCALHOST_ONLY=1
  python3 cellcheck.py --job jobs/<job>.json --build build/<cell> --out <result_dir> [--placements id1,id2] [--ik-only]

The job JSON (schema cell_planner_check/job/v1, written by build_job.py) lists per carton placement, per arm, ordered tasks made of steps:
  {"type":"transit",  "name":.., "target":{"candidates":[{"pose":{xyz,quat},"seeds":[[6 joints],..],"name":..}, ...]}, "optional":bool}
  {"type":"cartesian","name":.., "waypoints":[pose,..], "contact":bool}
plus the concurrency phases (each phase = tasks run at the same time, one per arm).  For every placement the harness
  1. sets the planning scene (carton box, table) with /apply_planning_scene,
  2. per arm and task, in order: /compute_ik (avoid_collisions, seeded from the study joints), /plan_kinematic_path (joint-space goal = IK solution,
     OMPL RRTConnect) for transits, /compute_cartesian_path (avoid_collisions, fraction must be 1.0, tool/carton ACM allowance scoped to the
     contact strokes via /apply_planning_scene) for enter / stroke / retreat,
  3. for each concurrent pair, time-aligns both trajectories by their planned durations and runs /check_state_validity of the combined dual-arm
     state along the time axis; a colliding pair is serialised,
  4. records success / reason / durations per step (two scaling sets, default transit 0.20/0.10 and 1.0/0.1, contact 0.1376/0.0954) and the cycle time
     per carton = snapshot + sum over phases of max(arm durations) (or the sum when the pair must be serialised).
"""
import argparse, json, math, os, sys, time, traceback
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cellgeom as G


def require_isolated_domain():
    if os.environ.get('ROS_DOMAIN_ID') != '187' or os.environ.get('ROS_LOCALHOST_ONLY') != '1':
        sys.exit('refusing to run: export ROS_DOMAIN_ID=187 ROS_LOCALHOST_ONLY=1 in this shell (domain 11 is the real robot)')


require_isolated_domain()
import rclpy
import moveit_client as MC

CARTON = 'detected_carton'
TABLE = 'table'
DEFAULT_SCALING = dict(names=['exec', 'fast'], transit=[[0.20, 0.10], [1.0, 0.1]], contact=[0.1376, 0.0954])
D2R = math.pi / 180.0


def deg(x):
    return float(x) / D2R


class Checker:
    def __init__(self, meta, job, opts):
        self.meta = meta
        self.job = job
        self.opts = opts
        MC.init()
        self.cl = MC.MoveItClient(meta, 'cell_check')
        self.arms = self.cl.arms
        sc = dict(DEFAULT_SCALING)
        sc.update(job.get('scaling') or {})
        self.sc = sc
        self.nsets = len(sc['names'])
        self.tip = {a: meta['arms'][a]['tip'] for a in self.arms}
        self.neutral = {a: [0.0] * 6 for a in self.arms}

    # ------------------------------------------------------------------ scene
    def set_scene(self, pl):
        cl = self.cl
        cg = pl['carton']
        boxes = [dict(name=CARTON, size=cg['size'], centre=cg['center'], quat=cg['quat'])]
        tb = self.job.get('table', {})
        x0, x1, hy, th = tb.get('x0', 0.12), tb.get('x1', 1.2), tb.get('half_y', 1.0), tb.get('thickness', 0.05)
        ztop = cg['center'][2] - cg['size'][2] / 2           # table top = carton bottom
        boxes.append(dict(name=TABLE, size=[x1 - x0, 2 * hy, th], centre=[(x0 + x1) / 2, 0.0, ztop - th / 2], quat=[0, 0, 0, 1]))
        ok = cl.set_scene(boxes, remove=[CARTON, TABLE])
        # tool/carton allowance is applied only during cut strokes (see run_cartesian); make sure it is off now
        pairs = [(self.tip[a], CARTON) for a in self.arms]
        cl.set_acm_allow(pairs, False)
        return ok

    def allow_contact(self, on):
        pairs = [(self.tip[a], CARTON) for a in self.arms]
        return self.cl.set_acm_allow(pairs, on)

    # ------------------------------------------------------------------ IK
    def solve_ik(self, arm, pose, seeds, q_cur, q_other):
        """returns dict(ok, q, reason, seed_index, contacts)"""
        cl = self.cl
        T = G.pose_to_T(pose)
        other = [a for a in self.arms if a != arm][0]
        tries = [(f'seed{i}', s) for i, s in enumerate(seeds or [])] + [('current', q_cur), ('neutral', [0.0] * 6)]
        ik_to = self.opts.get('ik_timeout', 0.5)
        errs = []
        for name, seed in tries:
            q, e = cl.ik(arm, T, {arm: list(seed), other: q_other}, avoid_collisions=True, timeout_s=ik_to)
            if q is not None:
                Tf = cl.fk({arm: q, other: q_other}, self.tip[arm])
                perr = float(np.linalg.norm(Tf[:3, 3] - T[:3, 3]) * 1000)
                rerr = G.rot_angle_deg(Tf[:3, :3].T @ T[:3, :3])
                return dict(ok=True, q=q, seed=name, pos_err_mm=perr, rot_err_deg=rerr, reason='ok')
            errs.append((name, e))
        # classify: is there any IK at all (ignoring collisions)?
        for name, seed in tries[:2] + tries[-2:]:
            q, e = cl.ik(arm, T, {arm: list(seed), other: q_other}, avoid_collisions=False, timeout_s=ik_to)
            if q is not None:
                v, contacts = cl.validity({arm: q, other: q_other}, 'both')
                return dict(ok=False, reason='ik_solution_in_collision' if not v else 'ik_collision_checker_rejects', q_noncollision=q, contacts=[list(c) for c in contacts][:8],
                            seed=name)
        return dict(ok=False, reason='no_ik_solution (KDL, %d seeds: %s)' % (len(tries), errs[0][1]))

    # ------------------------------------------------------------------ steps
    def run_transit(self, arm, step, q_state, q_study_end=None):
        """q_state: dict of both arms.  returns dict(ok, q_end, per-set results...)"""
        cl = self.cl
        other = [a for a in self.arms if a != arm][0]
        cands = step['target']['candidates']
        res = dict(name=step['name'], type='transit', ok=False, tried=[])
        for ci, cand in enumerate(cands):
            if cand.get('q_goal') is not None:
                ik = dict(ok=True, q=list(cand['q_goal']), seed='explicit_joint_goal', pos_err_mm=0.0, rot_err_deg=0.0, reason='ok')
            else:
                ik = self.solve_ik(arm, cand['pose'], cand.get('seeds'), q_state[arm], q_state[other])
            rec = dict(candidate=cand.get('name', ci), ik_ok=ik['ok'], ik_reason=ik['reason'])
            if cand.get('study_closed_form_margin_deg') is not None or 'study_closed_form_margin_deg' in cand:
                rec['study_closed_form_margin_deg'] = cand.get('study_closed_form_margin_deg')
            if not ik['ok']:
                if 'contacts' in ik:
                    rec['contacts'] = ik['contacts']
                res['tried'].append(rec)
                continue
            rec.update(ik_seed=ik['seed'], ik_pos_err_mm=ik['pos_err_mm'], ik_rot_err_deg=ik['rot_err_deg'])
            if cand.get('seeds'):
                rec['ik_dq_vs_seed0_deg'] = float(max(abs(deg(a - b)) for a, b in zip(ik['q'], cand['seeds'][0])))
            q_goal = ik['q']
            # skip the plan when start == goal
            dq = max(abs(a - b) for a, b in zip(q_state[arm], q_goal))
            plans = []
            ok_all = True
            for si in range(self.nsets):
                v, a = self.sc['transit'][si]
                if dq < 1e-4:
                    plans.append(dict(ok=True, q=[list(q_goal)], t=[0.0], duration=0.0, npoints=1, max_joint_step=0.0, planning_time=0.0, error='trivial'))
                    continue
                p = None
                errs_p = []
                for pname in self.opts.get('transit_planners', ['pilz:PTP', 'ompl:RRTConnectkConfigDefault']):
                    pipe, pid = pname.split(':')
                    pipe = {'pilz': 'pilz_industrial_motion_planner'}.get(pipe, pipe)
                    n_try = self.opts.get('ompl_attempts', 3) if pipe == 'ompl' else 1       # OMPL is randomised; Pilz PTP is deterministic
                    for k_try in range(n_try):
                        p = cl.plan(arm, q_state, q_goal, v, a, planner=pid, pipeline=pipe, planning_time=self.opts.get('planning_time', 5.0))
                        p['planner'] = pname
                        p['attempt'] = k_try + 1
                        if p['ok']:
                            break
                        errs_p.append('%s#%d:%s' % (pname, k_try + 1, p['error']))
                    if p['ok']:
                        break
                p['planner_errors'] = errs_p
                plans.append(p)
                if not p['ok']:
                    ok_all = False
                    rec['plan_error'] = '%s[%s]: %s' % (self.sc['names'][si], cand.get('name', ci), ' / '.join(errs_p))
                    break
            rec['plan_ok'] = ok_all
            res['tried'].append(rec)
            if ok_all:
                res.update(ok=True, candidate=cand.get('name', ci), q_end=q_goal, plans=plans, ik=ik,
                           duration=[p['duration'] for p in plans], planning_time=[p['planning_time'] for p in plans], planner=[p.get('planner') for p in plans],
                           ik_dq_vs_seed0_deg=rec.get('ik_dq_vs_seed0_deg'))
                break
        if not res['ok']:
            last = res['tried'][-1] if res['tried'] else {}
            res['reason'] = 'transit failed: ' + ('; '.join('%s: %s' % (t['candidate'], t.get('plan_error') or t['ik_reason']) for t in res['tried'][:4]))
        return res

    def run_cartesian(self, arm, step, q_state):
        cl = self.cl
        res = dict(name=step['name'], type='cartesian', ok=False)
        contact = bool(step.get('contact'))
        variants = [step['waypoints']] + list(step.get('alternatives') or [])
        step0 = self.opts.get('cart_step', 0.005)
        steps_try = [step0] + ([step0 / 5.0] if self.opts.get('flip_retry', True) else [])
        max_travel = self.opts.get('max_joint_travel_rad', 2.0)
        for vi, wp_list in enumerate(variants):
            wps = [G.pose_to_T(p) for p in wp_list]
            if contact:
                self.allow_contact(True)
            try:
                for step_m in steps_try:
                    outs = []
                    for si in range(self.nsets):
                        v, a = self.sc['contact'] if len(self.sc['contact']) == 2 and not isinstance(self.sc['contact'][0], list) else self.sc['contact'][si]
                        if not contact and step.get('free_space'):
                            v, a = self.sc['transit'][si]
                        o = cl.cartesian(arm, q_state, wps, v, a, max_step=step_m, avoid_collisions=True, jump_rev=self.opts.get('jump_rev', 0.25))
                        outs.append(o)
                        if si == 0 and not o['ok']:
                            break
                    o = outs[0]
                    if o['ok'] and o.get('q'):
                        Qa = np.array(o['q'])
                        trav = float(np.abs(np.diff(Qa, axis=0)).sum(0).max()) if len(Qa) > 1 else 0.0
                        o['joint_travel_rad'] = trav
                        if trav > max_travel:                       # KDL followed the line on another wrist branch (compute_cartesian_path's jump filter did not catch it)
                            o['ok'] = False
                            o['error'] = 'JOINT_BRANCH_FLIP(joint travel %.1f rad)' % trav
                    if o['ok']:
                        res['step_used_m'] = step_m
                        break
            finally:
                if contact:
                    self.allow_contact(False)
            if vi == 0:
                first_o = o
                wps0 = wps
            if o['ok']:
                if vi > 0:
                    res['used_alternative'] = vi
                break
            res.setdefault('failed_variants', []).append(dict(variant=vi, fraction=o['fraction'], error=o['error']))
        res['fraction'] = o['fraction']
        res['error'] = o['error']
        res['max_joint_step_deg'] = deg(o.get('max_joint_step', 0.0))
        if not o['ok']:
            o = first_o
            res['fraction'] = o['fraction']
            res['reason'] = 'cartesian fraction %.3f (%s)' % (o['fraction'], o['error'])
            res['diagnosis'] = self.diagnose_cartesian(arm, q_state, wps0, o, contact)
            res['reason'] += ' -> ' + res['diagnosis']['cause']
            return res
        res.update(ok=True, q_end=o['q'][-1], plans=outs, duration=[x['duration'] for x in outs], npoints=o['npoints'])
        return res

    def diagnose_cartesian(self, arm, q_state, wps, o_fail, contact):
        """why a Cartesian step did not reach fraction 1.0: rerun without collision checking / jump filter and locate the first colliding state"""
        cl = self.cl
        other = [x for x in self.arms if x != arm][0]
        d = dict(fraction=o_fail['fraction'])
        if str(o_fail.get('error', '')).startswith('JOINT_BRANCH_FLIP'):
            d['cause'] = 'wrist/joint branch flip along the stroke (%s): the straight-line IK followed another branch than the study' % o_fail['error']
            return d
        if contact:
            self.allow_contact(True)
        try:
            o2 = cl.cartesian(arm, q_state, wps, 0.1, 0.1, max_step=self.opts.get('cart_step', 0.005), avoid_collisions=False, jump_rev=0.0)
        finally:
            if contact:
                self.allow_contact(False)
        d['fraction_no_collision_check_no_jump_filter'] = o2['fraction']
        if o2['fraction'] < 1.0 - 1e-9:
            d['cause'] = 'IK fails along the line (reach / joint limits / KDL) at %.0f%% of the path' % (100 * o2['fraction'])
            return d
        # path exists without collision checks: find the first invalid state and the jump
        Q = np.array(o2['q'])
        steps = np.abs(np.diff(Q, axis=0)).max(1) if len(Q) > 1 else np.zeros(0)
        d['max_joint_step_deg_no_filter'] = deg(steps.max()) if len(steps) else 0.0
        first = None
        for k, qq in enumerate(Q):
            v, c = cl.validity({arm: qq.tolist(), other: q_state[other]}, 'both')
            if not v:
                first = (k, c)
                break
        if first is not None:
            d['cause'] = 'collision at %.0f%% of the path: %s' % (100.0 * first[0] / max(len(Q) - 1, 1), ', '.join('%s~%s' % c for c in first[1][:4]))
            d['first_collision_index'] = first[0]
            d['contacts'] = [list(c) for c in first[1][:8]]
        elif len(steps) and deg(steps.max()) > deg(self.opts.get('jump_rev', 0.25)):
            d['cause'] = 'joint jump %.1f deg between 5 mm steps (branch flip / singularity)' % deg(steps.max())
        else:
            d['cause'] = 'unknown (path exists and is collision-free without the filters)'
        return d

    # ------------------------------------------------------------------ one task
    def run_task(self, task, q_state, study_task=None, partner_by_step=None):
        """steps of one task for one arm.  q_state: dict both arms (the other arm held)."""
        arm = task['arm']
        other = [a for a in self.arms if a != arm][0]
        out = dict(id=task['id'], arm=arm, ok=True, steps=[], duration=[0.0] * self.nsets)
        q = {k: list(v) for k, v in q_state.items()}
        segs = [[] for _ in range(self.nsets)]       # per set: list of (t_local, q) concatenated trajectory
        bounds = [[] for _ in range(self.nsets)]     # per set: (step name, t0, t1)
        tcur = [0.0] * self.nsets
        step_q_end = []
        for sidx, step in enumerate(task['steps']):
            if partner_by_step:                                   # partner held at ITS state after the same step index (barrier-synchronised phase)
                pq = partner_by_step[min(sidx, len(partner_by_step) - 1)]
                if pq is not None:
                    q[other] = list(pq)
            if step['type'] == 'transit':
                r = self.run_transit(arm, step, q)
            else:
                r = self.run_cartesian(arm, step, q)
            if not r['ok']:
                if step.get('optional'):
                    r['skipped_optional'] = True
                    out['steps'].append({k: v for k, v in r.items() if k not in ('plans', 'q_end')})
                    continue
                step_q_end.append(None)
                out['ok'] = False
                out['failed_step'] = step['name']
                out['reason'] = '%s: %s' % (step['name'], r.get('reason', 'failed'))
                out['steps'].append({k: v for k, v in r.items() if k not in ('plans', 'q_end')})
                break
            for si in range(self.nsets):
                p = r['plans'][si]
                for tt, qq in zip(p['t'], p['q']):
                    segs[si].append((tcur[si] + tt, qq))
                bounds[si].append((step['name'], tcur[si], tcur[si] + p['duration']))
                tcur[si] += p['duration']
            if step.get('dwell_s'):                      # hold still at the end of this step (e.g. 1 s look dwell of the v2 study)
                for si in range(self.nsets):
                    segs[si].append((tcur[si] + step['dwell_s'], list(r['q_end'])))
                    bounds[si].append(('dwell', tcur[si], tcur[si] + step['dwell_s']))
                    tcur[si] += step['dwell_s']
            q[arm] = r['q_end']
            step_q_end.append(list(r['q_end']))
            slim = {k: v for k, v in r.items() if k not in ('plans', 'q_end', 'ik')}
            slim['q_end_deg'] = [round(deg(v), 2) for v in r['q_end']]
            if r.get('ik'):
                slim['ik_pos_err_mm'] = r['ik']['pos_err_mm']
                slim['ik_rot_err_deg'] = r['ik']['rot_err_deg']
            slim['max_joint_step_deg'] = max(deg(p.get('max_joint_step', 0.0)) for p in r['plans'])
            out['steps'].append(slim)
        out['duration'] = tcur
        out['segments'] = segs                    # kept for the pair check; stripped before JSON dump
        out['step_bounds'] = bounds
        out['q_start'] = q_state[arm]
        out['q_end'] = q[arm]
        out['step_q_end'] = step_q_end
        # study comparison on the stroke end points
        if study_task and out['ok']:
            out['study_margin_deg'] = study_task.get('margin')
        return out

    # ------------------------------------------------------------------ pair check
    @staticmethod
    def interp(seg, q_start, t):
        """seg: list of (t, q) ascending; piecewise-linear; before first -> q_start, after last -> last"""
        if not seg:
            return list(q_start)
        ts = np.array([s[0] for s in seg])
        Q = np.array([s[1] for s in seg])
        if t >= ts[-1]:
            return Q[-1].tolist()
        if t <= ts[0]:
            return Q[0].tolist() if ts[0] <= 0 else list(q_start)
        return [float(np.interp(t, ts, Q[:, j])) for j in range(Q.shape[1])]

    def pair_check(self, tA, tB, si, qA_start, qB_start):
        """time-aligned combined validity.  tA/tB task results (with 'segments').  returns dict"""
        cl = self.cl
        A, B = tA['arm'], tB['arm']
        sa, sb = tA['segments'][si], tB['segments'][si]
        TA, TB = tA['duration'][si], tB['duration'][si]
        T = max(TA, TB)
        # time step: bounded joint motion per sample (0.03 rad) and at most N samples
        vmax = 0.0
        for seg in (sa, sb):
            for (t0, q0), (t1, q1) in zip(seg[:-1], seg[1:]):
                if t1 > t0:
                    vmax = max(vmax, max(abs(x - y) for x, y in zip(q0, q1)) / (t1 - t0))
        dt = max(self.opts.get('pair_dt_min', 0.02), 0.03 / vmax) if vmax > 0 else 0.5
        dt = min(dt, 0.5)
        n = int(math.ceil(T / dt)) + 1
        if n > self.opts.get('pair_max_samples', 2500):
            n = self.opts.get('pair_max_samples', 2500)
        ts = np.linspace(0.0, T, max(n, 2))
        first_bad = None
        nbad = 0
        pairs = {}
        for t in ts:
            q = {A: self.interp(sa, qA_start, t), B: self.interp(sb, qB_start, t)}
            v, contacts = cl.validity(q, 'both')
            if not v:
                nbad += 1
                if first_bad is None:
                    first_bad = float(t)
                for c in contacts:
                    pairs[c] = pairs.get(c, 0) + 1
        def step_at(task, t):
            for nm, t0, t1 in task['step_bounds'][si]:
                if t0 <= t <= t1 + 1e-9:
                    return nm
            return 'idle_after_end'
        res = dict(ok=nbad == 0, n_samples=int(len(ts)), dt=float(dt), n_bad=nbad, first_bad_t=first_bad, T=float(T),
                   contacts=[dict(a=a_, b=b_, n=c) for (a_, b_), c in sorted(pairs.items(), key=lambda kv: -kv[1])[:8]])
        if first_bad is not None:
            res['steps_at_first_collision'] = {tA['id']: step_at(tA, first_bad), tB['id']: step_at(tB, first_bad)}
        return res

    def serial_check(self, first, second, si, q_first_start, q_second_start):
        """serial execution first -> second: (i) first's trajectory with `second` waiting at its phase-start state, (ii) second's trajectory with
        `first` parked at its end state.  Both were planned with the partner at neutral, so both are re-validated here."""
        cl = self.cl
        for who, other_q, tag in ((first, q_second_start, 'first_vs_second_waiting'), (second, first['q_end'], 'second_vs_first_parked')):
            seg = who['segments'][si]
            T = who['duration'][si]
            q_start = q_first_start if who is first else q_second_start
            ts = np.linspace(0.0, T, max(2, int(math.ceil(T / 0.25)) + 1))
            other_arm = second['arm'] if who is first else first['arm']
            for t in ts:
                q = {who['arm']: self.interp(seg, q_start, t), other_arm: other_q}
                v, c = cl.validity(q, 'both')
                if not v:
                    return dict(ok=False, where=tag, t=float(t), contacts=[list(x) for x in c][:6])
        return dict(ok=True)

    def serialise(self, taskA, taskB, tA, tB, phase_start):
        """the concurrent pair collides: run the two tasks one after the other.  Tier 1: RE-PLAN the first task with the waiting arm held at its
        phase-start state and then the second with the first parked at its end state (both orders tried, shorter total wins).  Tier 2: the waiting arm first
        parks at neutral (park transit, then the task planned with the partner neutral, then the waiting arm's task replanned from neutral) -- extra time
        counted.  Returns dict(ok, tier, order, time per scaling set, ...)."""
        out = dict(ok=False, attempts=[])
        best = None
        for (T1, r1_orig, T2, r2_orig) in ((taskA, tA, taskB, tB), (taskB, tB, taskA, tA)):
            a1, a2 = T1['arm'], T2['arm']
            order = '%s then %s' % (T1['id'], T2['id'])
            q1, q2 = phase_start[a1], phase_start[a2]
            att = dict(order=order, tier=1)
            try:
                r1 = self.run_task(T1, {a1: list(q1), a2: list(q2)})
                if r1['ok']:
                    r2 = self.run_task(T2, {a2: list(q2), a1: list(r1['q_end'])})
                    if r2['ok']:
                        att.update(ok=True, time=[r1['duration'][si] + r2['duration'][si] for si in range(self.nsets)])
                    else:
                        att.update(ok=False, why='second task replanned: %s' % r2.get('reason'))
                else:
                    att.update(ok=False, why='first task replanned with the partner waiting at its phase start: %s' % r1.get('reason'))
            except Exception as e:
                att.update(ok=False, why='exception %r' % e)
            out['attempts'].append(att)
            if att.get('ok') and (best is None or att['time'][0] < best['time'][0]):
                best = att
        if best is None:
            for (T1, r1_orig, T2, r2_orig) in ((taskA, tA, taskB, tB), (taskB, tB, taskA, tA)):
                a1, a2 = T1['arm'], T2['arm']
                order = '%s then %s (waiting arm parks first)' % (T1['id'], T2['id'])
                att = dict(order=order, tier=2)
                try:
                    park = dict(id='%s:park0' % a2, arm=a2, kind='park', steps=[dict(type='transit', name='park', target=dict(candidates=[dict(name='neutral', q_goal=[0.0] * 6)]))])
                    rp = self.run_task(park, {a2: list(phase_start[a2]), a1: list(phase_start[a1])})
                    if rp['ok']:
                        r2 = self.run_task(T2, {a2: list(self.neutral[a2]), a1: list(r1_orig['q_end'])})
                        if r2['ok']:
                            att.update(ok=True, time=[rp['duration'][si] + r1_orig['duration'][si] + r2['duration'][si] for si in range(self.nsets)])
                        else:
                            att.update(ok=False, why='second task from neutral: %s' % r2.get('reason'))
                    else:
                        att.update(ok=False, why='park: %s' % rp.get('reason'))
                except Exception as e:
                    att.update(ok=False, why='exception %r' % e)
                out['attempts'].append(att)
                if att.get('ok') and (best is None or att['time'][0] < best['time'][0]):
                    best = att
        if best is not None:
            out.update(ok=True, tier=best['tier'], order=best['order'], time=best['time'])
        return out

    # ------------------------------------------------------------------ placement
    def run_placement(self, pl):
        t0 = time.time()
        cl = self.cl
        out = dict(id=pl['id'], box=pl.get('box'), yaw_deg=pl.get('yaw_deg'), study=pl.get('study'), tasks={}, phases=[])
        if not pl.get('tasks'):
            out.update(feasible=False, skipped='no tasks in the job (the study found no plan for this placement)', cycle_s={n: None for n in self.sc['names']})
            return out
        if not self.set_scene(pl):
            out['error'] = 'apply_planning_scene failed'
            return out
        state = {a: list(self.neutral[a]) for a in self.arms}
        v, c = cl.validity(state, 'both')
        out['neutral_valid'] = bool(v)
        if not v:
            out['neutral_contacts'] = [list(x) for x in c][:8]
        tasks = dict(pl['tasks'])
        study_tasks = (pl.get('study') or {}).get('tasks', {})
        # implicit park: an arm that has finished its last task returns to the neutral (upright) pose concurrently with the partner's next phase,
        # so a finished arm never sits above the carton while the partner cuts (the study assumes the idle arm is parked clear)
        phases_run = [list(ph) for ph in pl['phases']]
        if self.opts.get('park_idle', True):
            last = {}
            for pi, ph in enumerate(phases_run):
                for tid in ph:
                    last[tasks[tid]['arm']] = pi
            for arm_, pi_last in last.items():
                if pi_last + 1 < len(phases_run):
                    tid = '%s:park' % arm_
                    tasks[tid] = dict(id=tid, arm=arm_, kind='park', steps=[dict(type='transit', name='park', target=dict(candidates=[dict(name='neutral', q_goal=[0.0] * 6)]))])
                    phases_run[pi_last + 1].append(tid)
        pl_phases = phases_run
        phase_times = [[] for _ in range(self.nsets)]
        all_ok = True
        for pi, phase in enumerate(pl_phases):
            phase_start = {a: list(state[a]) for a in self.arms}
            res_tasks = {}
            for tid in phase:
                task = tasks[tid]
                arm = task['arm']
                other = [a for a in self.arms if a != arm][0]
                # solo phase: the idle arm stays where it is.  Concurrent phase: the partner is modelled as parked (neutral, upright) while this arm's task
                # is planned (the study assumes the same), the true interaction is judged afterwards by the time-aligned pair check below
                if len(phase) == 2 and self.opts.get('concurrent_other', 'neutral') == 'neutral':
                    q_in = {arm: list(state[arm]), other: list(self.neutral[other])}
                else:
                    q_in = {arm: list(state[arm]), other: list(phase_start[other])}
                try:
                    r = self.run_task(task, q_in, study_tasks.get(tid))
                except Exception as e:
                    traceback.print_exc()
                    r = dict(id=tid, arm=arm, ok=False, reason='exception: %r' % e, steps=[], duration=[0.0] * self.nsets, segments=[[] for _ in range(self.nsets)],
                             q_start=state[arm], q_end=state[arm])
                res_tasks[tid] = r
                out['tasks'][tid] = {k: v for k, v in r.items() if k not in ('segments',)}
            # pass 2 (concurrent phases): a task that failed with the partner parked at neutral is replanned with the partner held at ITS state after the same
            # step index (step-synchronous: 'the other arm at its concurrent state'), using the partner's pass-1 step end states
            if len(phase) == 2 and self.opts.get('partner_pass2', True) and not all(res_tasks[t]['ok'] for t in phase):
                for tid in phase:
                    if res_tasks[tid]['ok']:
                        continue
                    ptid = [x for x in phase if x != tid][0]
                    pres = res_tasks[ptid]
                    task = tasks[tid]
                    arm = task['arm']
                    other = [x for x in self.arms if x != arm][0]
                    pbs = [(q if q is not None else list(self.neutral[other])) for q in pres.get('step_q_end', [])]
                    if not pbs or all(not any(abs(v) > 1e-9 for v in q) for q in pbs):
                        continue
                    r2 = self.run_task(task, {arm: list(phase_start[arm]), other: list(self.neutral[other])}, study_tasks.get(tid), partner_by_step=pbs)
                    r2['pass'] = 2
                    r2['pass1_reason'] = res_tasks[tid].get('reason')
                    if r2['ok']:
                        res_tasks[tid] = r2
                        out['tasks'][tid] = {k: v for k, v in r2.items() if k not in ('segments',)}
            ph = dict(tasks=list(phase), ok=all(res_tasks[t]['ok'] for t in phase))
            # advance the arm states (failed tasks leave the arm where it was)
            for tid in phase:
                r = res_tasks[tid]
                if r['ok']:
                    state[r['arm']] = list(r['q_end'])
            if not ph['ok']:
                all_ok = False
            # concurrent pair check
            if len(phase) == 2 and all(res_tasks[t]['ok'] for t in phase):
                tA, tB = res_tasks[phase[0]], res_tasks[phase[1]]
                ph['pair'] = []
                serial = [False] * self.nsets
                ser_res = None
                for si in range(self.nsets):
                    pc = self.pair_check(tA, tB, si, phase_start[tA['arm']], phase_start[tB['arm']])
                    pc['set'] = self.sc['names'][si]
                    if not pc['ok']:
                        serial[si] = True
                        if ser_res is None:
                            ser_res = self.serialise(tasks[phase[0]], tasks[phase[1]], tA, tB, phase_start)
                        pc['serial_ok'] = ser_res['ok']
                        pc['serial'] = {k: v for k, v in ser_res.items() if k != 'time'}
                        pc['serial_time'] = ser_res.get('time', [None] * self.nsets)[si]
                    ph['pair'].append(pc)
                ph['serialised'] = serial
                for si in range(self.nsets):
                    da, db = tA['duration'][si], tB['duration'][si]
                    if serial[si] and ph['pair'][si].get('serial_ok'):
                        ph.setdefault('time', []).append(ph['pair'][si]['serial_time'])
                    else:
                        ph.setdefault('time', []).append(da + db if serial[si] else max(da, db))
                    if serial[si] and not ph['pair'][si].get('serial_ok'):
                        ph['ok'] = False
                        all_ok = False
            elif ph['ok']:
                ph['serialised'] = [False] * self.nsets
                ph['time'] = [max(res_tasks[t]['duration'][si] for t in phase) for si in range(self.nsets)]
            out['phases'].append(ph)
            if ph.get('time'):
                for si in range(self.nsets):
                    phase_times[si].append(ph['time'][si])
        snap = self.job.get('snapshot_s', 2.0)
        out['feasible'] = bool(all_ok)
        out['cycle_s'] = {self.sc['names'][si]: (snap + sum(phase_times[si]) if all_ok else None) for si in range(self.nsets)}
        out['elapsed_s'] = time.time() - t0
        return out


def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if k not in ('segments',)}
    if isinstance(o, list):
        return [strip(v) for v in o]
    return o


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--job', required=True)
    ap.add_argument('--build', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--placements', default=None, help='comma separated placement ids')
    ap.add_argument('--planning-time', type=float, default=5.0)
    ap.add_argument('--ik-timeout', type=float, default=0.5)
    ap.add_argument('--transit-planners', default='pilz:PTP,ompl:RRTConnectkConfigDefault',
                    help='ordered list tried for every transit (default: Pilz PTP joint interpolation first, like the executor, then OMPL RRTConnect)')
    ap.add_argument('--no-park', action='store_true', help='do not park a finished arm during the partner\'s next phase')
    ap.add_argument('--no-partner-pass2', action='store_true', help='disable the step-synchronous partner replanning of failed concurrent tasks')
    ap.add_argument('--concurrent-other', default='neutral', choices=['neutral', 'phase_start'])
    a = ap.parse_args()
    b = a.build if os.path.isabs(a.build) else os.path.join(HERE, a.build)
    meta = json.load(open(os.path.join(b, 'cell_meta.json')))
    job = json.load(open(a.job))
    if abs(job['layout']['s'] - meta['s']) > 1e-6:
        sys.exit('job layout s=%s does not match the running cell build s=%s' % (job['layout']['s'], meta['s']))
    if job.get('design') and os.path.basename(job['design']).replace('.json', '') != meta['design']:
        print('WARNING: job design %s != build design %s' % (job['design'], meta['design']))
    os.makedirs(a.out, exist_ok=True)
    chk = Checker(meta, job, dict(planning_time=a.planning_time, ik_timeout=a.ik_timeout, transit_planners=a.transit_planners.split(','), concurrent_other=a.concurrent_other, park_idle=not a.no_park, partner_pass2=not a.no_partner_pass2))
    sel = set(a.placements.split(',')) if a.placements else None
    res = dict(schema='cell_planner_check/result/v1', job=job.get('name'), design=meta['design'], build=b, layout=job['layout'], scaling=chk.sc,
               domain=os.environ.get('ROS_DOMAIN_ID'), placements=[])
    outf = os.path.join(a.out, 'result_%s.json' % job.get('name', 'job'))
    for pl in job['placements']:
        if sel and pl['id'] not in sel:
            continue
        t0 = time.time()
        r = chk.run_placement(pl)
        res['placements'].append(strip(r))
        st = (pl.get('study') or {}).get('feasible')
        print('%-18s study=%-5s moveit=%-5s cycle=%s  %.0fs' % (pl['id'], st, r.get('feasible'), r.get('cycle_s'), time.time() - t0), flush=True)
        json.dump(res, open(outf, 'w'), indent=1)
    res['service_calls'] = chk.cl.ncalls
    res['service_time_s'] = chk.cl.tcalls
    json.dump(res, open(outf, 'w'), indent=1)
    print('wrote', outf)
    chk.cl.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
