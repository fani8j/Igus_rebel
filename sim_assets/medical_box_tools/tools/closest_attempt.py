"""Best-effort ("closest attempt") poses for targets that are NOT reachable (T1, T2): closest the suction tool / gripper can get to the press / grasp pose with the lift at its lowest limit,
sliders at home, d = 410 mm, under joint limits and clearance >= 2 mm (no penetration) against cabinet, stock, stand, AMR and the other arm.
  python3 tools/closest_attempt.py [--targets T1,T2]   -> tests/results/unreachable_closest.json, data/plans/closest_<T>.json (time-parametrised approach paths)
Also: --whatif  : how much lower the lift limit would have to be (50 mm steps) for the full front sequence of T1 / T2 to become feasible (parking_study style full evaluation)."""
import argparse, json, sys
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial import cKDTree
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import parking as pk, parking_study as ps, robot_kin as rk
import export_plan as ep

D, LAT, LIFT_LOW = 0.41, 0.005, -0.6235
SL, SR = -0.005973, 0.005973
MARGIN = 0.002


def make(R, cab, tname, lift=LIFT_LOW, d=D, lat=LAT):
    shelf, box, which = ps.TARGETS[tname]
    tg = cab.pick(shelf, box, which)
    pull = rk.PULL_OUT
    W = rk.World(cab, shelf, tg, d, lat)
    seqs = pk.build_all(W, 'front', pull)
    ev = pk.Evaluator(R, W, lift, SL, SR)
    return W, seqs, ev, tg, pull


def clr_min(ev, arm, q, fingers, other_q, pull):
    c = ev.clearances(arm, q, fingers, other_q, 0.024, 0.0)
    # tool vs the (target) carton is intended contact; tool/neighbour uses the same 2 mm margin here
    c = {k: v for k, v in c.items()}
    return c


def link_report(R, ev, W, arm, q, fingers, thr=0.02):
    """per link: (object, distance) for everything closer than thr"""
    qd = dict(zip(R.arm_joints[arm], q))
    if arm == 'right':
        qd['arm_right_finger_left_joint'] = fingers
        qd['arm_right_finger_right_joint'] = fingers
    T = R.subtree(f'arm_{arm}_base_link', ev.Tb[arm], qd)
    al, tl = R.groups(arm)
    out = []
    for l in al + tl:
        P = R.col_points(l)
        if not len(P):
            continue
        Pw = P @ T[l][:3, :3].T + T[l][:3, 3]
        for k, v in W.static_dist(Pw[::2]).items():
            if v < thr:
                out.append((l, k, round(v * 1000, 1)))
        for nm, tree in (('AMR', ev.amr_tree), ('stand/lift/sliders/camera', ev.stand_tree)):
            v = float(tree.query(Pw[::2])[0].min())
            if v < thr and not (l.endswith('base_link') or l.endswith('motor1')):
                out.append((l, nm, round(v * 1000, 1)))
    return out


def best_effort(R, ev, W, arm, Tt, fingers, other_q, pull, n_seeds=28, rng=None):
    rng = rng or np.random.default_rng(3)
    qmin, qmax = R.qmin[arm], R.qmax[arm]

    def res(q, with_coll=True):
        T = ev.Tb[arm] @ R.fk_tcp(arm, q)
        r = [(T[:3, 3] - Tt[:3, 3]) * 1000.0]
        rv = Rot.from_matrix(Tt[:3, :3].T @ T[:3, :3]).as_rotvec()
        r.append(np.degrees(rv) * 2.0)                 # 1 deg ~ 2 mm
        if with_coll:
            c = ev.clearances(arm, q, fingers, other_q, 0.024, 0.0)
            # target carton itself is excluded for the tool (intended contact): clearances() has no tool-vs-target term; arm_carton is the arm only
            viol = [max(0.0, MARGIN - v) * 1000.0 * 30 for k, v in c.items() if k in ('arm_cabinet', 'arm_blocks', 'arm_carton', 'tool_cabinet', 'tool_neighbours', 'robot_amr', 'robot_stand', 'arm_arm', 'tool_tool')]
            r.append(np.array(viol))
        return np.concatenate(r)
    seeds = [s for s in pk.SEEDS[:14]] + [rng.uniform(qmin + 0.1, qmax - 0.1) for _ in range(n_seeds - 14)]
    best = None
    for s in seeds:
        s = np.clip(s, qmin + 1e-3, qmax - 1e-3)
        sol = least_squares(res, s, bounds=(qmin, qmax), method='trf', diff_step=1e-4, max_nfev=60)
        q = sol.x
        r = res(q)
        nv = len(r) - 6
        feasible = np.all(r[6:] < 1e-6) if nv else True
        cost = float(np.linalg.norm(r[:6]))
        key = (0 if feasible else 1, cost + (0 if feasible else 1e3 * float(r[6:].max())))
        if best is None or key < best[0]:
            best = (key, q, feasible)
    return best[1], best[2]


def errors(R, ev, W, arm, q, Tt):
    T = ev.Tb[arm] @ R.fk_tcp(arm, q)
    dp_w = T[:3, 3] - Tt[:3, 3]
    dp_c = W.R.T @ dp_w                       # cabinet frame: x lateral, y depth (into the cabinet), z up
    rot = float(np.degrees(np.linalg.norm(Rot.from_matrix(Tt[:3, :3].T @ T[:3, :3]).as_rotvec())))
    return dict(total_mm=float(np.linalg.norm(dp_w) * 1000), vertical_mm=float(dp_c[2] * 1000), depth_mm=float(dp_c[1] * 1000), lateral_mm=float(dp_c[0] * 1000), orientation_deg=rot,
                tcp_cabinet=(W.R.T @ (T[:3, 3] - W.t)).tolist(), target_cabinet=(W.R.T @ (Tt[:3, 3] - W.t)).tolist()), T


def analyse(R, cab, tname):
    W, seqs, ev, tg, pull = make(R, cab, tname)
    sq = seqs[0]
    Tl = next(s for s in sq['left'] if s.name.startswith('L2')).T1
    Tr = next(s for s in sq['right'] if s.name.startswith('R3')).T1
    out = dict(target=tname, parking=dict(d_m=D, lateral_m=LAT, lift_m=LIFT_LOW, slider_left=SL, slider_right=SR), arms={})
    plans = {}
    for arm, Tt, other, fingers in (('left', Tl, np.zeros(6), 0.024), ('right', Tr, np.zeros(6), 0.024)):
        q, feas = best_effort(R, ev, W, arm, Tt, fingers, other, pull)
        err, T = errors(R, ev, W, arm, q, Tt)
        at_limit = [f'joint{i + 1} ({np.degrees(q[i]):.1f} deg, limit {np.degrees(R.qmin[arm][i] if abs(q[i] - R.qmin[arm][i]) < abs(q[i] - R.qmax[arm][i]) else R.qmax[arm][i]):.0f} deg)'
                    for i in range(6) if min(q[i] - R.qmin[arm][i], R.qmax[arm][i] - q[i]) < np.radians(1.0)]
        c = ev.clearances(arm, q, fingers, other, 0.024, 0.0)
        active = {k: round(v * 1000, 1) for k, v in c.items() if v < 0.008}
        lr = link_report(R, ev, W, arm, q, fingers, 0.008)
        if at_limit and not lr:
            block = 'joint limit: ' + ', '.join(at_limit)
        elif lr and not at_limit:
            block = 'collision: ' + '; '.join(f'{l} vs {o} ({d} mm)' for l, o, d in lr[:6])
        elif lr and at_limit:
            block = 'joint limit ' + ', '.join(at_limit) + ' and collision ' + '; '.join(f'{l} vs {o} ({d} mm)' for l, o, d in lr[:6])
        else:
            block = 'pure reach (no joint at a limit, no clearance active)'
        # approach path home -> best pose
        full_check = ev.check
        ev.check = lambda c: [k for k, v in c.items() if k in ('arm_cabinet', 'arm_blocks', 'arm_carton', 'tool_cabinet', 'tool_neighbours', 'robot_amr', 'robot_stand', 'arm_arm', 'tool_tool') and v < 0.0015]
        qs = None
        Tb_best = ev.Tb[arm] @ R.fk_tcp(arm, q)
        for back in (0.25, 0.18, 0.12, 0.06):
            Ts = Tb_best.copy()
            Ts[:3, 3] = Tb_best[:3, 3] - back * Tb_best[:3, 2]                      # retracted along the tool axis (staging pose in free air)
            qs0 = R.ik_multi(arm, ev.Tb_inv[arm] @ Ts, [q] + pk.SEEDS[:8])
            if qs0 is None:
                continue
            seg, ok, why, _ = ev.ik_path(arm, Ts, Tb_best, qs0)
            if not ok or any(ev.check(ev.clearances(arm, qq, fingers, other, 0.024, 0.0)) for qq in seg[::2]):
                continue
            tr = ev.transit(arm, np.zeros(6), qs0, fingers, other, 0.0, True)       # relaxed: >= 1.5 mm along the whole path (the pose itself is at the 2 mm margin)
            if tr is None:
                continue
            qs = list(tr) + list(seg) + [q]
            stage_back = back
            break
        if qs is None:                                                                  # fallback: joint-space staging pose near the best pose (random search, relaxed check)
            rng = np.random.default_rng(11)
            relaxed = ev.check
            for trial in range(60):
                sig = 0.15 + 0.01 * trial
                qs0 = np.clip(q + rng.normal(0, sig, 6) * np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.3]), R.qmin[arm] + 0.01, R.qmax[arm] - 0.01)
                if not ev._leg_ok(arm, qs0, q, fingers, other, 0.024 * 0, 14):
                    continue
                tr = ev.transit(arm, np.zeros(6), qs0, fingers, other, 0.0, True)
                if tr is None:
                    continue
                leg = [qs0 + (q - qs0) * (i / 14) for i in range(1, 15)]
                qs = list(tr) + leg
                stage_back = f'joint-space staging pose (random search, trial {trial})'
                break
        ev.check = full_check
        out['arms'][arm] = dict(error=err, collision_free_at_pose=bool(feas), joint_deg=[round(float(np.degrees(x)), 2) for x in q], joints_at_limit=at_limit, active_clearances_mm=active, blocked_by=block,
                                approach_path_collision_free=qs is not None, approach_staging_back_m=(stage_back if qs is not None else None), approach_path_min_clearance_rule='>= 1.5 mm to cabinet/stock/stand/AMR/other arm along the whole path', min_clearance_at_pose_mm=round(min(c.values()) * 1000, 1))
        # footer caption (one line)
        short = ('joint limit: ' + ', '.join(x.split(' ')[0] for x in at_limit)) if (at_limit and not lr) else \
                ('collision: ' + lr[0][0].replace('arm_' + arm + '_', '') + ' vs ' + lr[0][1].replace('cab:', '').replace('carton:', 'stock ')) if lr else 'pure reach'
        tool = 'suction pad' if arm == 'left' else 'gripper'
        pt = 'press point' if arm == 'left' else 'grasp point'
        vv = err['vertical_mm']
        out['arms'][arm]['caption'] = f"{tool} stops {err['total_mm']:.0f} mm from the {pt} ({abs(vv):.0f} mm {'above' if vv > 0 else 'below'}), {short}"
        out['arms'][arm]['short_block'] = short
        if err['total_mm'] < 2.0:
            out['arms'][arm]['caption'] = 'gripper reaches its handoff grasp pose (gap 0 mm): not the limiting arm for this target'
        if qs is not None:
            qs = np.asarray(qs)
            t = ep.time_param(np.zeros(6), qs, None, 0.0, ep.VEL_SCALE_JOINT, ep.ACC_SCALE_JOINT)
            P = np.vstack([np.zeros(6), qs])[::-1]
            tb = ep.time_param(P[0], P[1:], None, 0.0, ep.VEL_SCALE_JOINT, ep.ACC_SCALE_JOINT)
            plans[arm] = dict(q0=[0.0] * 6, q=qs.tolist(), t=t.tolist(), q_best=q.tolist(), q_back=P[1:].tolist(), t_back=tb.tolist(), tcp_cabinet=err['tcp_cabinet'], target_cabinet=err['target_cabinet'])
    out['cabinet_to_world'] = dict(R=W.R.tolist(), t=W.t.tolist())
    # T2-style transit problem: stage pose of the suction arm and the direct joint interpolation home -> stage
    stage_T = sq['left'][0].T1
    sols = ev.ik_all('left', stage_T, pk.SEEDS)
    diag = dict(stage_pose_ik_solutions=len(sols))
    if sols:
        q1 = sols[0]
        c = ev.clearances('left', q1, 0.024, np.zeros(6), 0.024, 0.0)
        diag['clearance_at_stage_mm'] = {k: round(v * 1000, 1) for k, v in c.items() if v < 0.02}
        diag['links_at_stage'] = link_report(R, ev, W, 'left', q1, 0.024, 0.015)
        first = None
        for i in range(0, 13):
            qq = q1 * i / 12
            lr = link_report(R, ev, W, 'left', qq, 0.024, 0.0)
            cc = ev.clearances('left', qq, 0.024, np.zeros(6), 0.024, 0.0)
            if ev.check(cc) and first is None:
                first = dict(step=f'{i}/12 of the direct joint interpolation', violated=ev.check(cc), links=link_report(R, ev, W, 'left', qq, 0.024, 0.015))
        diag['first_violation_on_direct_transit'] = first
        r = ev.run_arm('left', sq['left'], np.zeros(6), 0.024)
        diag['full_left_sequence_result'] = r['reason'] or 'ok'
    else:
        diag['note'] = 'no IK solution for the stage pose'
    out['stage_transit_diagnosis'] = diag
    return out, plans, W


def whatif(R, cab):
    """lift range extended below the real limit in 50 mm steps: smallest extension for which the full front sequence (both arms) is feasible, per target, same parking d / lateral"""
    res = {}
    for tn in ('T1', 'T2'):
        row = {}
        shelf, box, which = ps.TARGETS[tn]
        tg = cab.pick(shelf, box, which)
        for ext in np.arange(0.0, 0.801, 0.05):
            lift = LIFT_LOW - float(ext)
            ok_d = []
            for d in (0.36, 0.41, 0.46):
                for lat in (0.005, 0.055):
                    ev, seqs = ps.make_eval(R, cab, tg, shelf, 'front', rk.PULL_OUT, d, lat, lift, SL, SR)
                    r = ev.evaluate(seqs)
                    if r['ok']:
                        ok_d.append((d, lat))
            row[f'{ext:.2f}'] = ok_d
            print(tn, f'lift {lift:.4f}', ok_d, flush=True)
            if ok_d:
                break
        res[tn] = row
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--targets', default='T1,T2')
    ap.add_argument('--whatif', action='store_true')
    a = ap.parse_args()
    R, cab = rk.Robot(), rk.Cabinet()
    tag = ('whatif' if a.whatif else a.targets.replace(',', '_'))
    fn = HERE.parent / 'tests' / 'results' / f'unreachable_closest_{tag}.json'
    allr = {}
    if a.whatif:
        allr['what_if_lower_lift'] = whatif(R, cab)
    else:
        for tn in a.targets.split(','):
            out, plans, W = analyse(R, cab, tn)
            allr[tn] = out
            (HERE.parent / 'data' / 'plans' / f'closest_{tn}.json').write_text(json.dumps(dict(target=tn, cabinet_to_world=out['cabinet_to_world'], parking=out['parking'], arms=plans)))
            print(tn, json.dumps({arm: {k: v for k, v in o.items() if k in ('error', 'blocked_by', 'approach_path_collision_free')} for arm, o in out['arms'].items()}, default=float), flush=True)
            print(tn, 'T2-style diag', json.dumps(out['stage_transit_diagnosis'], default=float), flush=True)
    fn.write_text(json.dumps(allr, indent=1, default=float))


if __name__ == "__main__":
    main()
