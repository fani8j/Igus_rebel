#!/usr/bin/env python3
"""Job-JSON builder: converts offline-study results into jobs for cellcheck.py.

  python3 build_job.py v1 --design h000n.0 --layout 0.0,0.40,0.355 --boxes BIG370,SMALL --yaws 0:180:15 --out jobs/h000n.0_v1.json
  python3 build_job.py v1 --design baseline --layout 0.0,0.35,0.46 --boxes BIG370,SMALL --yaws 0:90:30 --out jobs/baseline_v1.json
  python3 build_job.py v2 --recommended /home/farhan/igus_rebel_ros2/scan_results/cell_v2_20261003/recommended.json --out jobs/v2.json   (adapter, see v2_adapter)

v1 source = holder_camera_fingers_20261002 (read-only import via study_eval.py): the cut strokes, allocations (H / F / G) and concurrency pairings are
computed with the study's own code (closed-form IK, 8 branches + DP, collision proxies, camera + fingers), then each task is converted into poses:
  look      transit to a camera pose ~25 cm in front of the surface the arm cuts first (optical axis <= 20 deg off the surface normal), closed-form IK
            candidates ranked by joint margin and study collision proxies; none -> recorded in placement['notes'] and skipped
  approach  transit to the stroke start pose backed off `approach` m along the tool -Z (outward normal)
  enter     Cartesian approach -> start;  stroke  Cartesian start -> end (fixed tool orientation incl. tilt);  retreat  Cartesian end -> end backed off
Poses are armN_knife_tip poses in the planning frame "world" (cellgeom.py), seeds are the study joint solutions (+ other closed-form IK branches).
The job also lists, per placement, the study verdict (feasible / alloc / margins) so cellcheck can compare.
"""
import argparse, json, math, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cellgeom as G

DEPTH = 0.010


def parse_yaws(s):
    a, b, c = [int(v) for v in s.split(':')]
    return list(range(a, b, c))


def translate(T, v):
    T2 = np.array(T, float).copy()
    T2[:3, 3] += np.asarray(v, float)
    return T2


# ------------------------------------------------------------------------------------------------ look poses
def look_candidates(se, design, arm_i, s, target, n_out, scen, K=6, dists=(0.25, 0.22, 0.28, 0.20, 0.30), max_tilt=20.0, log=None):
    """camera poses facing the surface point `target` from the outside (n_out = outward surface normal) at ~25 cm, axis within max_tilt deg of -n_out.
    Returns (candidates list of {pose, seeds, name, margin, study_collision_free, dist, tilt, roll}, note)."""
    import camlib as CL
    import dalib as D
    from dalib import Scen, StrokeSet, ORI5
    cam = design.get('camera')
    if not cam:
        return [], 'design has no camera'
    T_l6_cam = np.array(cam['T_link6_camera'])
    T_cam_l6 = np.linalg.inv(T_l6_cam)
    n_out = np.asarray(n_out, float) / np.linalg.norm(n_out)
    zc0 = -n_out
    ref = np.array([1.0, 0, 0]) if abs(zc0[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(zc0, ref)
    u /= np.linalg.norm(u)
    v = np.cross(zc0, u)
    SS = StrokeSet([scen], ORI5)
    row = lambda a: np.ravel_multi_index((0, a, 0, 0, 0), SS.shape)
    cands = []
    for dist in dists:
        for tilt in (0.0, 10.0, 20.0):
            if tilt > max_tilt + 1e-9:
                continue
            for az in ([0.0] if tilt == 0 else range(0, 360, 45)):
                za = math.radians(tilt)
                aza = math.radians(az)
                zc = math.cos(za) * zc0 + math.sin(za) * (math.cos(aza) * u + math.sin(aza) * v)
                p_cam = np.asarray(target, float) - dist * zc
                for roll in range(0, 360, 15):
                    Rc = CL.cam_frames(zc[None], np.array([float(roll)]))[0]
                    T_wc = G.T_from(Rc, p_cam)
                    T_w_l6 = T_wc @ T_cam_l6
                    Tw = T_w_l6 @ se.T_tcp
                    br = se.ik_branches(Tw, arm_i, s)
                    if not br:
                        continue
                    qs = np.array([b[0] for b in br])
                    ms = np.array([b[1] for b in br])
                    ok = np.zeros(len(qs), bool)
                    if True:
                        ok = D.collision_free(qs, SS.par, np.full(len(qs), row(arm_i)), se.des)
                    for q, m, c in zip(qs, ms, ok):
                        cands.append(dict(q=q, margin=float(m), cf=bool(c), dist=dist, tilt=tilt, az=az, roll=roll, Tw=Tw))
    if not cands:
        return [], 'no closed-form IK solution within the joint limits for any look pose (dist %s, tilt <= %g deg)' % (list(dists), max_tilt)
    # rank: study-collision-free first, margin >= 5 deg next, then closeness to 25 cm / small tilt, then margin
    def key(c):
        return (not c['cf'], c['margin'] < 5.0, abs(c['dist'] - 0.25) + 0.002 * c['tilt'], -min(c['margin'], 25.0))
    cands.sort(key=key)
    out, seen = [], []
    for c in cands:
        sig = (round(c['dist'], 3), c['tilt'], c['az'], c['roll'] // 45)
        if sig in seen:
            continue
        seen.append(sig)
        out.append(dict(name='look_d%03d_t%d_a%d_r%d' % (c['dist'] * 1000, c['tilt'], c['az'], c['roll']), pose=G.T_to_pose(c['Tw']), seeds=[c['q'].tolist()],
                        margin_deg=c['margin'], study_collision_free=c['cf'], dist=c['dist'], tilt_deg=c['tilt'], roll_deg=c['roll']))
        if len(out) >= K:
            break
    note = '%d candidates (%d study-collision-free of %d IK solutions)' % (len(out), sum(1 for c in cands if c['cf']), len(cands))
    return out, note


# ------------------------------------------------------------------------------------------------ v1
def cut_task(tid, arm, info, approach_m, retreat_m, branches_fn=None, standoffs=None, look_step=None):
    """approach / enter / stroke / retreat steps for one study stroke.  The approach transit lists several stand-off candidates (approach_m, then smaller
    ones): the first one that has an IK solution and a collision-free transit is used."""
    P = [G.pose_to_T(p) for p in info['poses5']]
    Ts, Te = P[0], P[-1]
    push = Ts[:3, 2]                                    # tool Z = into the material
    arm_i = G.ARM_INDEX[arm]
    q0 = np.array(info['q'][0]) if info.get('q') else None
    cands = []
    for so in (standoffs or [approach_m]):
        Ta = translate(Ts, -push * so)
        seeds = []
        if q0 is not None:
            seeds.append(q0.tolist())
        best_margin = None
        if branches_fn is not None:
            br = branches_fn(Ta, arm_i)
            if br:
                best_margin = max(m for _, m in br)
            if q0 is not None:
                br.sort(key=lambda x: float(np.max(np.abs(x[0] - q0))))      # closest to the study start solution first
            for q, m in br[:4]:
                if not any(np.allclose(q, s_, atol=1e-6) for s_ in seeds):
                    seeds.append(q.tolist())
        cands.append(dict(name='standoff_%dmm' % round(so * 1000), pose=G.T_to_pose(Ta), seeds=seeds, standoff_m=so, study_closed_form_margin_deg=best_margin))
    retreats = [G.T_to_pose(translate(Te, -push * r)) for r in ([retreat_m] + [x for x in (0.025, 0.012) if x < retreat_m])]
    steps = ([look_step] if look_step else []) + [
        dict(type='transit', name='approach', target=dict(candidates=cands)),
        dict(type='cartesian', name='enter', contact=True, waypoints=[G.T_to_pose(Ts)]),
        dict(type='cartesian', name='stroke', contact=True, waypoints=[G.T_to_pose(Te)]),
        dict(type='cartesian', name='retreat', contact=True, waypoints=[retreats[0]], alternatives=[[r] for r in retreats[1:]]),
    ]
    return dict(id=tid, arm=arm, kind='cut', steps=steps, study_source=info['source'], study_margin_deg=info.get('margin'),
                start=info['poses5'][0], end=info['poses5'][-1])


def build_v1(a):
    import study_eval as SE
    from dalib import Scen
    h, d, s = [float(v) for v in a.layout.split(',')]
    design = json.load(open(os.path.join(HERE, 'designs', a.design + '.json')))
    se = SE.StudyEval(a.design)
    boxes = a.boxes.split(',')
    yaws = parse_yaws(a.yaws)
    offs = [(0.0, 0.0)] if not a.offsets else [tuple(float(x) / 1000 for x in o.split(':')) for o in a.offsets.split(',')]
    scen = [(b, y, dx, dy) for b in boxes for y in yaws for (dx, dy) in offs]
    cache = a.cache or os.path.join(G.REPO, 'scan_results/planner_check_20261003/study_cache')
    data = SE.cached_evaluate(a.design, (h, d, s), scen, cache, rear=0.0, tag=a.tag)
    job = dict(schema='cell_planner_check/job/v1', name=a.name or ('%s_v1' % a.design), design='designs/%s.json' % a.design, source='v1 holder_camera_fingers_20261002',
               layout=dict(h=h, d=d, s=s), snapshot_s=2.0,
               scaling=dict(names=['exec', 'fast'], transit=[[0.20, 0.10], [1.0, 0.1]], contact=[0.1376, 0.0954]),
               table=dict(x0=0.12, x1=1.2, half_y=1.0, thickness=0.05), placements=[])
    for rec in data:
        pid = '%s_y%03d' % (rec['box'], rec['yaw']) + ('' if (rec['dx'] == 0 and rec['dy'] == 0) else '_dx%+d_dy%+d' % (round(rec['dx'] * 1000), round(rec['dy'] * 1000)))
        scn = Scen(h, d, s, rec['box'], rec['yaw'], rec['dx'], rec['dy'])
        pl = dict(id=pid, box=rec['box'], yaw_deg=rec['yaw'], dx=rec['dx'], dy=rec['dy'],
                  carton=G.carton_geometry(rec['box'], h, d, rec['yaw'], rec['dx'], rec['dy']),
                  study=dict(feasible=rec['feasible'], alloc=rec['alloc'], pairing=rec['pairing'], allocs_ok=rec['allocs_ok'], best_effort=rec.get('best_effort', False),
                             failing_tasks=rec.get('failing_tasks', []), valid_layout=rec['valid_layout'], task_margin=rec['task_margin'], tasks={}),
                  tasks={}, notes=[])
        phases = [list(p) for p in rec['phases']]
        first_task = {}
        for ph in phases:
            for tid in ph:
                arm = tid.split(':')[0]
                first_task.setdefault(arm, tid)
        for tid, info in rec['tasks'].items():
            arm = tid.split(':')[0]
            pl['tasks'][tid] = cut_task(tid, arm, info, a.approach, a.retreat, lambda T, i: se.ik_branches(T, i, s), [a.approach, 0.02, 0.01])
            pl['study']['tasks'][tid] = dict(margin=info.get('margin'), source=info['source'], q=info.get('q'))
        # look poses (one per arm, facing the surface of the arm's first cut)
        look_phase = []
        for arm in G.ARMS:
            if arm not in first_task:
                continue
            ft = rec['tasks'][first_task[arm]]
            Ts, Te = G.pose_to_T(ft['poses5'][0]), G.pose_to_T(ft['poses5'][-1])
            n_out = -Ts[:3, 2]
            target = 0.5 * (Ts[:3, 3] + Te[:3, 3]) + n_out * DEPTH       # tape on the surface above the stroke centre
            cands, note = look_candidates(se, design, G.ARM_INDEX[arm], s, target, n_out, scn, K=a.look_k)
            tid = '%s:look' % arm
            if cands:
                pl['tasks'][tid] = dict(id=tid, arm=arm, kind='look', steps=[dict(type='transit', name='look', target=dict(candidates=cands))],
                                        look_target=[float(v) for v in target], look_normal=[float(v) for v in n_out], look_for=first_task[arm])
                look_phase.append(tid)
            pl['notes'].append('%s look for %s: %s' % (arm, first_task[arm], note if cands else ('NONE: ' + note + ' (skipped)')))
        pl['phases'] = ([look_phase] if look_phase else []) + phases
        job['placements'].append(pl)
    # expected study poses for the FK sanity check (study joint solutions -> TCP pose)
    cases = []
    for pl in job['placements']:
        for tid, st in pl['study']['tasks'].items():
            if st.get('q'):
                arm = tid.split(':')[0]
                t = pl['tasks'][tid]
                cases.append(dict(label='%s/%s/start' % (pl['id'], tid), arm=arm, q=st['q'][0], expect_tcp_world=t['start']))
                cases.append(dict(label='%s/%s/end' % (pl['id'], tid), arm=arm, q=st['q'][-1], expect_tcp_world=t['end']))
    return job, cases


# ------------------------------------------------------------------------------------------------ v2 adapter
V2_DIR = os.path.join(G.REPO, 'scan_results/cell_v2_20261003')


def v2_single_candidates(C2, des, scens, band):
    """per placement and arm: best v2 candidate path (approach + 5 stroke points + retreat, gamma <= band) with a look pose for each of the tasks sP, sM, tF, using
    cell2's own building blocks (StrokeSet, eval_strokes, build_cands2, look_poses, patch_for) exactly as cell2.evaluate() does.  A path is only kept when a look
    pose exists for its patch.  order: tF first (towards the end it finishes at), then the side face at that end, then the other side face."""
    from dalib import NEG
    gam = tuple(g for g in C2.GAMMAS if g <= band)
    SSg = {g: C2.StrokeSet(scens, C2.ORI15, C2.CUT_NEW, g * C2.D2R, 0.0) for g in gam}
    Mg = {}
    for g in gam:
        SS = SSg[g]
        idx = np.arange(SS.S) if g == gam[0] else np.where(np.isin(SS.task_of, (0, 1)))[0]
        idx = idx[SS.valid[SS.sc_of[idx]]]
        m = np.full(SS.S, NEG)
        if len(idx):
            m[idx] = C2.eval_strokes(SS, des, collide=True, idx=idx)['m_col']
        Mg[g] = m
    out = []
    for si, sc in enumerate(scens):
        rec = {}
        for arm in (0, 1):
            tasks = {}
            for t in ('sP', 'sM', 'tF'):
                parts = []
                for g in (gam if C2.is_side(t) else [gam[0]]):
                    cd = C2.build_cands2(SSg[g], des, si, arm, t, Mg[g], C2.THR, ext=True)
                    if cd is not None:
                        parts.append(cd)
                best = None
                if parts:
                    Cc = {k: np.concatenate([p[k] for p in parts]) for k in ('q', 'm', 'dr', 'g')}
                    order = np.argsort(-Cc['m'])
                    looks = {}
                    for k in order[:40]:
                        dr = 0 if C2.is_side(t) else int(Cc['dr'][k])
                        if dr not in looks:
                            c, u, v, nin = C2.patch_for(sc, t, dr, C2.CUT_NEW)
                            looks[dr] = C2.look_poses(SSg[gam[0]], si, arm, des, c, u, v, nin, C2.THR, C2.MAX_LOOKS, Cc['q'][k, 0])
                        L = looks[dr]
                        if L is not None and len(L['m']):
                            j = int(np.argmin(np.abs(L['q'] - Cc['q'][k, 0][None]).max(-1)))
                            best = dict(q7=Cc['q'][k].tolist(), q_look=L['q'][j].tolist(), margin=float(Cc['m'][k]), gamma=float(Cc['g'][k]), dr=dr)
                            break
                    if best is None:                      # no look pose for any candidate: keep the best-margin path without a look (reach-only view)
                        k = int(order[0])
                        best = dict(q7=Cc['q'][k].tolist(), q_look=None, margin=float(Cc['m'][k]), gamma=float(Cc['g'][k]),
                                    dr=(0 if C2.is_side(t) else int(Cc['dr'][k])), look_missing=True)
                if best is None:
                    # the study has no collision-free path: try the best joint-limit-feasible path IGNORING collisions (so MoveIt's own collision checker decides)
                    from dalib import ik_flange, margins, dp_paths, NPTS
                    for g in sorted(gam, reverse=True):
                        SSx = SSg[g]
                        base = np.ravel_multi_index((si, arm, C2.TASK_I[t], 0, 0), SSx.shape)
                        idx = base + np.arange(SSx.shape[3] * SSx.shape[4])
                        if not SSx.valid[si]:
                            continue
                        ml = C2.eval_strokes(SSx, des, collide=False, idx=idx)['m_lim']
                        sel = idx[ml >= C2.THR]
                        if len(sel) == 0:
                            sel = idx[ml >= 0.0]
                        if len(sel) == 0:
                            continue
                        n = len(sel)
                        Rf = SSx.Rf[sel] @ des.R.T
                        off = Rf @ des.t
                        pf = SSx.Pf[sel] - off[:, None, :]
                        Tf = np.zeros((n, NPTS, 4, 4)); Tf[..., 3, 3] = 1
                        Tf[..., :3, :3] = Rf[:, None]; Tf[..., :3, 3] = pf
                        q8, valid8 = ik_flange(Tf.reshape(n * NPTS, 4, 4))
                        val, jid = margins(q8, valid8)
                        q8 = q8.reshape(n, NPTS, 8, 6); val = val.reshape(n, NPTS, 8); jid = jid.reshape(n, NPTS, 8)
                        bc, bj, br = dp_paths(val, jid, q8, want_path=True)
                        k = int(np.argmax(bc))
                        if bc[k] < 0:
                            continue
                        q5 = q8[k][np.arange(NPTS), br[k]]
                        dr = int(((sel[k] - base) // SSx.shape[4]))
                        best = dict(q7=None, q5=q5.tolist(), q_look=None, margin=float(bc[k]), gamma=float(g), dr=dr, limits_only=True, look_missing=True)
                        break
                if best is None:
                    # not even a joint-limit-feasible path: nominal stroke (gamma 0, tilt 0, direction 0) so the planner can still be asked
                    SS0 = SSg[gam[0]]
                    sidx = int(np.ravel_multi_index((si, arm, C2.TASK_I[t], 0, [i for i, (u1, u2) in enumerate(C2.ORI15) if abs(u1) < 1e-12 and abs(u2) < 1e-12][0]), SS0.shape))
                    P = SS0.Pf[sidx].copy()
                    P[:, 1] += (sc.s / 2 if arm == 0 else -sc.s / 2)
                    Rn = SS0.Rf[sidx]
                    best = dict(q7=None, q_look=None, margin=None, gamma=0.0, dr=0, nominal=True, look_missing=True,
                                poses5=[dict(xyz=P[i].tolist(), quat=[float(v) for v in __import__('cellgeom').R_to_quat(Rn)]) for i in range(P.shape[0])])
                tasks[t] = best
            reach_only = all(not tasks[t].get('nominal') and not tasks[t].get('limits_only') for t in tasks)
            reach = reach_only and all(not tasks[t].get('look_missing') for t in tasks)
            end_plus = (tasks['tF']['dr'] == 0)          # task_targets: tF dr 0 runs from the -a end to the +a end
            order_ = ['tF', 'sP', 'sM'] if end_plus else ['tF', 'sM', 'sP']
            rec[str(arm)] = dict(reachable=reach, reach_only=reach_only, tasks=tasks, order=order_)
        out.append(rec)
    return out


def build_v2(a):
    """Adapter for the 2026-10-03 cell_v2 study (recommended.json + finals.json, imported READ-ONLY from scan_results/cell_v2_20261003).  Run it as its own
    process (the v2 python modules share their names with the v1 ones).  Per placement the v2 `cell2.evaluate()` (gamma <= band, 30 mm approach / retreat,
    look pose before every cut, >= 30 mm arm clearance, concurrency phases) is re-run for the recommended design; its chosen per-task joint paths
    (approach, 5 stroke points, retreat) and look poses become poses of armN_knife_tip:  look (1 s dwell) -> approach -> enter -> stroke -> retreat.
    Study verdict = fully simultaneous success (`simul`), `serial_ok` kept separately.  Placements the study cannot do are kept with no tasks (skipped)."""
    import ast
    rec = a.recommended or os.path.join(V2_DIR, 'recommended.json')
    if not os.path.exists(rec):
        sys.exit('v2 adapter: %s does not exist yet' % rec)
    R = json.load(open(rec))
    combo = next((c for c in R['combos'] if c['label'] == a.design), R['combos'][0])
    lay = combo['layout']
    if abs(lay.get('toe_deg', 0.0)) > 1e-9:
        sys.exit('v2 adapter: toe-in layouts (toe_deg=%s) need a rotated arm base in the URDF; not supported yet' % lay['toe_deg'])
    h, d, s = lay['h'], lay['d'], lay['s']
    sys.path.insert(0, V2_DIR)
    from common import des_from_params
    import camlib as V2cam
    import cell2 as C2
    from dalib import Scen, fk_flange, ik_flange, LIM_LO, LIM_HI, D2R
    finals = {x['label']: x for x in json.load(open(os.path.join(V2_DIR, 'finals.json')))}
    f = finals[combo['label']]
    des = des_from_params(combo['holder']['params'], fingers=True)
    des.cam = V2cam.Cam(np.asarray(f['cam']['o']), np.asarray(f['cam']['Rc']), np.asarray(f['cam']['brp']))
    T_tcp = np.asarray(des.T, float)
    T_inv = np.linalg.inv(T_tcp)
    design_name = 'v2_' + combo['label']
    boxes = a.boxes.split(',')
    yaws = parse_yaws(a.yaws)
    offs = [(0.0, 0.0)] if not a.offsets else [tuple(float(x) / 1000 for x in o.split(':')) for o in a.offsets.split(',')]
    scen = [(b, y, dx, dy) for b in boxes for y in yaws for (dx, dy) in offs]
    band = a.band
    cache = os.path.join(a.cache or os.path.join(G.REPO, 'scan_results/planner_check_20261003/study_cache'), ('study_v2single_%s_%g_%g_%g_b%d.json' if a.mode == 'single' else 'study_v2_%s_%g_%g_%g_b%d.json') % (combo['label'], h, d, s, band))
    res = None
    if os.path.exists(cache):
        c = json.load(open(cache))
        if c['scenarios'] == [list(x) for x in scen]:
            res = c['res']
            print('v2 study cache hit', cache)
    if res is None and a.mode == 'single':
        scens = [Scen(h, d, s, b, y, dx, dy) for (b, y, dx, dy) in scen]
        t0 = time.time()
        res = v2_single_candidates(C2, des, scens, band)
        print('v2 single-arm candidates: %d placements in %.0f s' % (len(scens), time.time() - t0))
        json.dump(dict(scenarios=[list(x) for x in scen], res=res), open(cache, 'w'))
    if res is None:
        scens = [Scen(h, d, s, b, y, dx, dy) for (b, y, dx, dy) in scen]
        gam = tuple(g for g in C2.GAMMAS if g <= band)
        t0 = time.time()
        res = C2.evaluate(scens, des, gammas=gam, bands=(band,), look=True, pairs=True, toe=0.0, mode='sync', verbose=True)
        print('v2 cell2.evaluate: %d placements in %.0f s' % (len(scens), time.time() - t0))
        def jsonable(o):
            if isinstance(o, dict):
                return {(k if isinstance(k, (str, int, float, bool)) or k is None else str(k)): jsonable(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [jsonable(v) for v in o]
            if hasattr(o, 'tolist'):
                return o.tolist()
            return o
        res = jsonable(res)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        json.dump(dict(scenarios=[list(x) for x in scen], res=res), open(cache, 'w'))

    def tcp_world(q, arm_i):
        T6 = fk_flange(np.atleast_2d(np.asarray(q, float)))
        T = T6 @ T_tcp
        T[:, 1, 3] += G.arm_y(arm_i, s)
        return T

    def branches(T, arm_i):
        T6 = T @ T_inv
        T6 = T6.copy()
        T6[1, 3] -= G.arm_y(arm_i, s)
        q, valid = ik_flange(T6[None])
        out = []
        for b in range(q.shape[1]):
            if valid[0, b]:
                qd = q[0, b] / D2R
                m = float(np.min(np.minimum(qd - LIM_LO, LIM_HI - qd)))
                if m >= 0:
                    out.append((q[0, b], m))
        out.sort(key=lambda x: -x[1])
        return out

    job = dict(schema='cell_planner_check/job/v1', name=a.name or ('%s_v2' % design_name), design='designs/%s.json' % design_name, source='v2 cell_v2_20261003 (band gamma<=%d)' % band,
               layout=dict(h=h, d=d, s=s), snapshot_s=2.0,
               scaling=dict(names=['exec', 'fast'], transit=[[0.20, 0.10], [1.0, 0.1]], contact=[0.1376, 0.0954]),
               table=dict(x0=0.12, x1=1.2, half_y=1.0, thickness=0.05), placements=[])
    cases = []
    if a.mode == 'single':
        for (box, yaw, dx, dy), Rr in zip(scen, res):
            for arm_i in (0, 1):
                arm = G.ARMS[arm_i]
                R1 = Rr[str(arm_i)]
                pid = '%s_y%03d_%sonly' % (box, yaw, arm) + ('' if (dx == 0 and dy == 0) else '_dx%+d_dy%+d' % (round(dx * 1000), round(dy * 1000)))
                pl = dict(id=pid, box=box, yaw_deg=yaw, dx=dx, dy=dy, carton=G.carton_geometry(box, h, d, yaw, dx, dy),
                          study=dict(feasible=bool(R1['reachable']), reach_only_feasible=bool(R1['reach_only']), one_arm=arm,
                                     per_task={t: (None if v is None else dict(margin=v['margin'], gamma=v['gamma'], has_look=v['q_look'] is not None)) for t, v in R1['tasks'].items()},
                                     tasks={}), tasks={}, notes=[], phases=[])
                nomi = [t for t, v in R1['tasks'].items() if v.get('nominal') or v.get('limits_only')]
                if nomi:
                    pl['notes'].append('the study has no collision-free path for %s (arm %s): the best joint-limit-feasible path ignoring collisions (or a nominal stroke) is given to the planner instead' % (nomi, arm))
                if not R1['reachable'] and not nomi:
                    pl['notes'].append('study verdict = infeasible because no look pose exists for %s; the cut paths themselves are tested here without a look step' % [t for t, v in R1['tasks'].items() if v.get('look_missing')])
                order = R1['order']
                for t in order:
                    v = R1['tasks'][t]
                    if v.get('nominal'):
                        poses5 = v['poses5']
                        info = dict(poses5=poses5, q=None, margin=None, source='nominal_no_study_path', branches_start=[])
                    elif v.get('limits_only'):
                        q5 = np.array(v['q5'])
                        poses5 = [G.T_to_pose(T) for T in tcp_world(q5, arm_i)]
                        info = dict(poses5=poses5, q=q5.tolist(), margin=v['margin'], source='v2_limits_only_path_ignoring_collisions(gamma=%g)' % v['gamma'], branches_start=[])
                    else:
                        q7 = np.array(v['q7'])
                        T7 = tcp_world(q7, arm_i)
                        poses5 = [G.T_to_pose(T) for T in T7[1:6]]
                        info = dict(poses5=poses5, q=q7[1:6].tolist(), margin=v['margin'], source='v2_single_arm_path(gamma=%g)' % v['gamma'], branches_start=[])
                    look_step = None
                    if v['q_look'] is not None:
                        Tl = tcp_world(v['q_look'], arm_i)[0]
                        look_step = dict(type='transit', name='look', dwell_s=1.0, target=dict(candidates=[dict(name='v2_look', pose=G.T_to_pose(Tl), seeds=[v['q_look']])]))
                    tid = '%s:%s' % (arm, t)
                    task = cut_task(tid, arm, info, 0.03, 0.03, lambda T, i: branches(T, i), [0.03, 0.02, 0.01], look_step)
                    task['gamma_deg'] = v['gamma']
                    pl['tasks'][tid] = task
                    pl['study']['tasks'][tid] = dict(margin=v['margin'], source=info['source'], q=info['q'])
                    if info['q']:
                        cases.append(dict(label='%s/%s/start' % (pid, tid), arm=arm, q=info['q'][0], expect_tcp_world=poses5[0]))
                        cases.append(dict(label='%s/%s/end' % (pid, tid), arm=arm, q=info['q'][-1], expect_tcp_world=poses5[-1]))
                pl['phases'] = [['%s:%s' % (arm, t)] for t in order]
                pl['notes'].append('one arm does all three cuts serially (order %s); the other arm stays parked' % order)
                job['placements'].append(pl)
        return job, cases
    for (box, yaw, dx, dy), Rr in zip(scen, res):
        b = Rr['bands'][str(band)] if str(band) in Rr['bands'] else Rr['bands'][band]
        pid = '%s_y%03d' % (box, yaw) + ('' if (dx == 0 and dy == 0) else '_dx%+d_dy%+d' % (round(dx * 1000), round(dy * 1000)))
        pl = dict(id=pid, box=box, yaw_deg=yaw, dx=dx, dy=dy, carton=G.carton_geometry(box, h, d, yaw, dx, dy),
                  study=dict(feasible=bool(b.get('simul')), serial_ok=bool(b.get('serial')), individually_reachable=bool(b.get('ind')), alloc=b.get('alloc'),
                             cycle_model_s=b.get('cyc'), tasks={}), tasks={}, notes=[], phases=[])
        if not b.get('path_q'):
            pl['notes'].append('v2 study found no complete plan (ind=%s, simul=%s, serial=%s): no tasks generated, placement skipped by the harness' % (b.get('ind'), b.get('simul'), b.get('serial')))
            job['placements'].append(pl)
            continue
        pq = {ast.literal_eval(k): v for k, v in b['path_q'].items()}
        lq = {ast.literal_eval(k): v for k, v in (b.get('look_q') or {}).items()}
        alloc = b['alloc']
        seq = {int(k): v for k, v in b['op_seq'].items()}
        opt = next((op for op in C2.options(alloc) if {int(k): v for k, v in op['seq'].items()} == seq), None)
        if opt is not None and opt['phases'] is not None:
            phases = [['%s:%s' % (G.ARMS[x[0]], x[1]) for x in ph] for ph in opt['phases']]
        else:                                       # G allocation (independent arms): full seam by one arm overlaps the other arm's two cuts
            w = [k for k, v in seq.items() if len(v) == 1][0]
            o = 1 - w
            phases = [['%s:tF' % G.ARMS[w], '%s:%s' % (G.ARMS[o], seq[o][0])], ['%s:%s' % (G.ARMS[o], seq[o][1])]]
            pl['notes'].append('G allocation: phases approximated as (tF || first side cut), second side cut alone')
        for (arm_i, t), P in pq.items():
            tid = '%s:%s' % (G.ARMS[arm_i], t)
            q7 = np.array(P['q'])
            T7 = tcp_world(q7, arm_i)
            poses5 = [G.T_to_pose(T) for T in T7[1:6]]
            info = dict(poses5=poses5, q=q7[1:6].tolist(), margin=None, source='v2_study_path(gamma=%g)' % P.get('gamma', 0.0), branches_start=[])
            look_step = None
            if (arm_i, t) in lq:
                Tl = tcp_world(lq[(arm_i, t)]['q'], arm_i)[0]
                look_step = dict(type='transit', name='look', dwell_s=1.0, target=dict(candidates=[dict(name='v2_look', pose=G.T_to_pose(Tl), seeds=[lq[(arm_i, t)]['q']])]))
            task = cut_task(tid, G.ARMS[arm_i], info, 0.03, 0.03, lambda T, i: branches(T, i), [0.03, 0.02, 0.01], look_step)
            task['gamma_deg'] = P.get('gamma')
            pl['tasks'][tid] = task
            pl['study']['tasks'][tid] = dict(margin=None, source=info['source'], q=info['q'])
            cases.append(dict(label='%s/%s/start' % (pid, tid), arm=G.ARMS[arm_i], q=info['q'][0], expect_tcp_world=poses5[0]))
            cases.append(dict(label='%s/%s/end' % (pid, tid), arm=G.ARMS[arm_i], q=info['q'][-1], expect_tcp_world=poses5[-1]))
        pl['phases'] = phases
        job['placements'].append(pl)
    return job, cases


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='src', required=True)
    p = sub.add_parser('v1')
    p.add_argument('--design', required=True)
    p.add_argument('--layout', default='0.0,0.40,0.355')
    p.add_argument('--boxes', default='BIG370,SMALL')
    p.add_argument('--yaws', default='0:180:15', help='start:stop:step in degrees (stop exclusive)')
    p.add_argument('--offsets', default=None, help='dx:dy in mm, comma separated, e.g. 0:0,30:0')
    p.add_argument('--approach', type=float, default=0.04)
    p.add_argument('--retreat', type=float, default=0.05)
    p.add_argument('--look-k', type=int, default=12)
    p.add_argument('--name', default=None)
    p.add_argument('--tag', default='')
    p.add_argument('--cache', default=None)
    p.add_argument('--out', required=True)
    q = sub.add_parser('v2')
    q.add_argument('--recommended', default=None)
    q.add_argument('--design', default='D00', help='combo label in the v2 recommended.json')
    q.add_argument('--boxes', default='BIG370')
    q.add_argument('--yaws', default='0:180:15')
    q.add_argument('--offsets', default=None)
    q.add_argument('--band', type=int, default=45, help='gamma band (deg)')
    q.add_argument('--mode', default='full', choices=['full', 'single'], help='full = the study allocation (two arms, concurrent phases); single = ONE arm does all three cuts serially (the other stays parked)')
    q.add_argument('--name', default=None)
    q.add_argument('--cache', default=None)
    q.add_argument('--out', required=True)
    a = ap.parse_args()
    if a.src == 'v1':
        job, cases = build_v1(a)
    else:
        job, cases = build_v2(a)
    out = a.out if os.path.isabs(a.out) else os.path.join(HERE, a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(job, open(out, 'w'), indent=1)
    json.dump(cases, open(out.replace('.json', '_fkcases.json'), 'w'), indent=1)
    nf = sum(1 for p in job['placements'] if p['study']['feasible'])
    print('wrote %s: %d placements (%d study-feasible), %d fk cases' % (out, len(job['placements']), nf, len(cases)))


if __name__ == '__main__':
    main()
