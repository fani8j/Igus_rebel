"""AMR parking study: for every target (T1..T8, E1, E2) and both modes find the standoff d (AMR front bumper to cabinet front face), lateral offset (target carton y in the D200 frame),
lift_joint and slider positions at the CENTRE of the feasible window, and check the +-20 mm / +-1.5 deg parking error box.
  step 1  reach maps (IK interior depth per tool orientation, tools/reach_maps.py) -> score of every (d, lat, lift, sL, sR) = min over all key poses of the interior depth
  step 2  top candidates by score -> full evaluation: continuous Cartesian paths with IK continuation (no branch flips), joint limits, collisions (>= 15 mm; CAD 3 mm tool-neighbour rule; 10 mm tool-tool)
  step 3  best passing candidate -> 8 corner poses of the parking error box (full evaluation each) -> d window scan at 10 mm steps (full evaluation)
Run: nice -n 19 python3 tools/parking_study.py [--modes front,topdown] [--targets T1,T2] [--workers 4]   -> tests/results/amr_parking.json / .md"""
import argparse
import itertools
import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import parking as pk  # noqa: E402
import robot_kin as rk  # noqa: E402

OUT = HERE.parent
TARGETS = {'T1': (1, 'ultimaster_nagomi', 'mid'), 'T2': (2, 'accuforce', 'mid'), 'T3': (3, 'agent_monorail', 'mid'), 'T4': (3, 'conqueror_nc_pro', 'mid'),
           'T5': (4, 'angiolite', 'mid'), 'T6': (5, 'agent_monorail', 'mid'), 'T7': (6, 'accuforce', 'mid'), 'T8': (6, 'conqueror_nc_pro', 'mid'),
           'E1': (3, 'agent_monorail', 'last'), 'E2': (6, 'angiolite', 'first')}
PULL_BY_MODE_SHELF = {('topdown', 6): 0.150}
D_MIN = 0.02
D_GRID = np.arange(0.02, 0.701, 0.02)
LAT_GRID = np.arange(-0.20, 0.2001, 0.02)
LIFT_GRID = np.arange(-0.62, 0.6201, 0.02)
S_GRID = np.array([0.0, 0.03, 0.06, 0.10, 0.15, 0.20])       # inward slider travel (m): slider_left = -s, slider_right = +s (home: 0.005973)
CORNERS = [(dx, dy, dyaw) for dx in (-0.02, 0.02) for dy in (-0.02, 0.02) for dyaw in (-1.5, 1.5)]
MAPS = None


def maps():
    global MAPS
    if MAPS is None:
        MAPS = dict(np.load(OUT / 'data' / 'reach_maps.npz'))
        for f in (OUT / 'data').glob('reach_maps_*.npz'):
            MAPS.update({k: v for k, v in np.load(f).items() if k not in ('origin', 'step')})
    return MAPS


def lookup(depth, org, step, P):
    idx = np.rint((P - org) / step).astype(int)
    ok = (idx[..., 0] >= 0) & (idx[..., 0] < depth.shape[0]) & (idx[..., 1] >= 0) & (idx[..., 1] < depth.shape[1]) & (idx[..., 2] >= 0) & (idx[..., 2] < depth.shape[2])
    out = np.zeros(P.shape[:-1])
    ii = idx[ok]
    out[ok] = depth[ii[:, 0], ii[:, 1], ii[:, 2]]
    return out


def key_points_world0(R, cab, tg, shelf, mode, pull, suction_map):
    W = rk.World(cab, shelf, tg, 0.0, 0.0)
    seqs = pk.build_sequences(W, mode, pull)
    pts = {}
    for arm in ('left', 'right'):
        P = []
        for s in seqs[arm]:
            for T in (s.T1,):
                if T is not None:
                    P.append(T[:3, 3])
        pts[arm] = np.array(P)
    return pts


def arm_base_positions(R):
    lift_pos = {}
    out = {}
    for arm in ('left', 'right'):
        sgn = -1.0 if arm == 'left' else 1.0
        B = np.zeros((len(LIFT_GRID), len(S_GRID), 3))
        for i, l in enumerate(LIFT_GRID):
            for j, s in enumerate(S_GRID):
                B[i, j] = R.arm_base(arm, l, sgn * s if s > 0 else sgn * 0.005973)[:3, 3]
        out[arm] = B
    return out


def score_grid(R, cab, tg, shelf, mode, pull, suction_map):
    m = maps()
    org, step = m['origin'], float(m['step'])
    kp = key_points_world0(R, cab, tg, shelf, mode, pull, suction_map)
    B = arm_base_positions(R)
    shift = np.zeros((len(D_GRID), len(LAT_GRID), 3))
    shift[..., 0] = D_GRID[:, None]
    shift[..., 1] = LAT_GRID[None, :]
    S = {}
    for arm, mapname in (('left', suction_map), ('right', 'right_front' if mode == 'front' else 'right_topdown')):
        depth = m[mapname + '_depth']
        sc = None
        for p in kp[arm]:
            P = (p[None, None, None, None, :] + shift[:, :, None, None, :]) - B[arm][None, None, :, :, :]
            v = lookup(depth, org, step, P)                  # (nd, nl, nlift, ns)
            sc = v if sc is None else np.minimum(sc, v)
        S[arm] = sc
    return S


def pick_candidates(S, n=150, spacing=1, split=False):
    """ranked candidates.  single lift: triples (d, lat, lift) ranked by min(best left slider score, best right slider score) -> (idx, sL, sR, score) with lift index;
    split: pairs (d, lat) ranked by min(best over lift and slider of each arm) -> (idx, sL, sR, score) with lift index = (liftL index, liftR index)"""
    if not split:
        bl, br = S['left'].argmax(axis=3), S['right'].argmax(axis=3)
        comb = np.minimum(S['left'].max(axis=3), S['right'].max(axis=3))
        nd = 3
    else:
        nl, ns = S['left'].shape[2], S['left'].shape[3]
        fl = S['left'].reshape(S['left'].shape[:2] + (-1,))
        fr = S['right'].reshape(S['right'].shape[:2] + (-1,))
        al, ar = fl.argmax(axis=2), fr.argmax(axis=2)
        comb = np.minimum(fl.max(axis=2), fr.max(axis=2))
        nd = 2
    flat = np.argsort(comb, axis=None)[::-1]
    chosen = []
    for f in flat:
        v = comb.flat[f]
        if v <= 0:
            break
        idx = np.unravel_index(f, comb.shape)
        if any(all(abs(idx[a] - c[0][a]) <= spacing for a in range(nd)) for c in chosen):
            continue
        if not split:
            chosen.append((idx, int(bl[idx]), int(br[idx]), float(v)))
        else:
            zl, sl_ = divmod(int(al[idx]), ns)
            zr, sr_ = divmod(int(ar[idx]), ns)
            chosen.append(((idx[0], idx[1], (zl, zr)), sl_, sr_, float(v)))
        if len(chosen) >= n:
            break
    return chosen, comb


def make_eval(R, cab, tg, shelf, mode, pull, d, lat, lift, sl, sr, dx=0.0, dy=0.0, dyaw=0.0, yaw180=False):
    """lift: float (one lift height for the whole pick) or (lift_for_suction_phase, lift_for_gripper_phase) = lift moved between the phases (suction arm back home first)"""
    W = rk.World(cab, shelf, tg, d, lat, dx, dy, dyaw)
    split = isinstance(lift, (tuple, list))
    suction_R = pk.R_SUCTION @ np.diag([-1.0, -1.0, 1.0]) if yaw180 else None
    seqs = pk.build_all(W, mode, pull, suction_R=suction_R, left_home=split)
    if split:
        return pk.Evaluator(R, W, lift[0], sl, sr, right_ev=pk.Evaluator(R, W, lift[1], sl, sr)), seqs
    return pk.Evaluator(R, W, lift, sl, sr), seqs


def slider_values(si, ri):
    return -(0.005973 if S_GRID[si] == 0 else float(S_GRID[si])), (0.005973 if S_GRID[ri] == 0 else float(S_GRID[ri]))


def feasible(R, cab, tg, shelf, mode, pull, d, lat, lift, sl, sr, yaw180, dx=0.0, dy=0.0, dyaw=0.0):
    ev, seqs = make_eval(R, cab, tg, shelf, mode, pull, d, lat, lift, sl, sr, dx, dy, dyaw, yaw180)
    r = ev.evaluate(seqs)
    return r


def interval(fn, centre, step, lo_lim, hi_lim, max_steps=60):
    """contiguous feasible interval of a 1-D parameter around `centre` (fn(x) -> bool); returns (lo, hi)"""
    lo = hi = centre
    for _ in range(max_steps):
        x = round(lo - step, 4)
        if x < lo_lim - 1e-9 or not fn(x):
            break
        lo = x
    for _ in range(max_steps):
        x = round(hi + step, 4)
        if x > hi_lim + 1e-9 or not fn(x):
            break
        hi = x
    return lo, hi


def corners_pass(R, cab, tg, shelf, mode, pull, d, lat, lift, sl, sr, yaw180):
    bad = []
    for dx, dy, dyaw in CORNERS:
        r = feasible(R, cab, tg, shelf, mode, pull, d, lat, lift, sl, sr, yaw180, dx, dy, dyaw)
        if not r['ok']:
            bad.append(((dx, dy, dyaw), r['reason'][:80]))
    return not bad, bad


def arm_ok(R, cab, tg, shelf, mode, pull, arm, d, lat, lift, s, yaw180, left_home=True):
    """full evaluation (continuous paths, IK branches, collisions) of ONE arm phase; the other arm sits at home"""
    W = rk.World(cab, shelf, tg, d, lat)
    seqs = pk.build_all(W, mode, pull, suction_R=(pk.R_SUCTION @ np.diag([-1.0, -1.0, 1.0]) if yaw180 else None), left_home=left_home)
    ev = pk.Evaluator(R, W, lift, -s if arm == 'left' else 0.005973, 0.005973 if arm == 'left' else s)
    if arm == 'left':
        ev = pk.Evaluator(R, W, lift, -s, 0.005973)
    else:
        ev = pk.Evaluator(R, W, lift, -0.005973, s)
    for sq in seqs:
        if ev.run_arm(arm, sq[arm], np.zeros(6), 0.024)['ok']:
            return True
    return False


def split_start(R, cab, tg, shelf, mode, pull, yaw180, S):
    """lift-split start candidates: scan each arm alone on a coarse (d, lift) grid at lat 0 / home sliders, take the d range where both have a feasible lift, centre the lifts"""
    ds = np.round(np.arange(0.16, 0.60, 0.04), 2)
    lifts = np.round(np.arange(-0.62, 0.63, 0.04), 2)
    FL = np.zeros((len(ds), len(lifts)), bool)
    FR = np.zeros_like(FL)
    for arm, F, key in (('left', FL, 'left'), ('right', FR, 'right')):
        m = S[key].max(axis=3)                         # (nd, nl, nlift) map score
        for i, d in enumerate(ds):
            di = int(np.argmin(np.abs(D_GRID - d)))
            for j, z in enumerate(lifts):
                zi = int(np.argmin(np.abs(LIFT_GRID - z)))
                if m[di, :, zi].max() <= 0:
                    continue                           # outside the reach map for every lateral offset
                F[i, j] = arm_ok(R, cab, tg, shelf, mode, pull, arm, float(d), 0.0, float(z), 0.0 if True else 0.0, yaw180)
    out = []
    for i, d in enumerate(ds):
        if FL[i].any() and FR[i].any():
            def centre(row):
                js = np.where(row)[0]
                runs, start = [], js[0]
                for a, b in zip(js[:-1], js[1:]):
                    if b != a + 1:
                        runs.append((start, a))
                        start = b
                runs.append((start, js[-1]))
                a, b = max(runs, key=lambda r: r[1] - r[0])
                return float(lifts[(a + b) // 2]), float(lifts[a]), float(lifts[b])
            out.append((float(d), centre(FL[i]), centre(FR[i])))
    return out, (ds, lifts, FL, FR)


def lift_of(zi):
    return (float(LIFT_GRID[zi[0]]), float(LIFT_GRID[zi[1]])) if isinstance(zi, tuple) else float(LIFT_GRID[zi])


def set_lift(lift, which, x):
    if isinstance(lift, tuple):
        return (x, lift[1]) if which == 0 else (lift[0], x)
    return x


def study_target(args):
    tname, mode, yaw180 = args
    t0 = time.time()
    R, cab = rk.Robot(), rk.Cabinet()
    shelf, box, which = TARGETS[tname]
    tg = cab.pick(shelf, box, which)
    pull = PULL_BY_MODE_SHELF.get((mode, shelf), rk.PULL_OUT)
    suction_map = 'left_suction_180' if yaw180 else 'left_suction'
    res = dict(target=tname, mode=mode, shelf=shelf, box=box, which=which, pull_m=pull, yaw180=yaw180, carton=tg['name'])
    try:
        S = score_grid(R, cab, tg, shelf, mode, pull, suction_map)
    except KeyError as e:
        res.update(feasible=False, reason=f'missing reach map {e}')
        return res
    out = None
    for split in ((False, True) if mode == 'topdown' else (False,)):
        out = study_one(R, cab, tg, tname, mode, shelf, pull, yaw180, S, split, dict(res), t0)
        if out['feasible']:
            break
    return out


def study_one(R, cab, tg, tname, mode, shelf, pull, yaw180, S, split, res, t0):
    cands, comb = pick_candidates(S, 150, split=split)
    res['lift_split_between_phases'] = split
    seeds_split = []
    if split:
        starts, scan = split_start(R, cab, tg, shelf, mode, pull, yaw180, S)
        res['split_scan_d_with_both_arms_feasible'] = [x[0] for x in starts]
        for d, (zl, _, _), (zr, _, _) in starts:
            seeds_split.append((d, zl, zr))
    res['candidates_inside_both_reach_maps'] = int((comb > 0).sum())
    if not cands:
        res.update(feasible=False, reason='no (d, lat, lift, slider) puts all key poses of both arms inside the IK reach map' + (' (lift may differ between the phases)' if split else ' (same lift for both arms)'), elapsed=time.time() - t0)
        return res
    fails, tried = {}, 0
    passing = []
    sd = [(d, 0.0, (zl, zr)) for d, zl, zr in seeds_split]
    mid = len(sd) // 2
    for d, lat, lift in ([sd[mid]] + sd[:mid] + sd[mid + 1:] if sd else []):
        r = feasible(R, cab, tg, shelf, mode, pull, d, lat, lift, -0.005973, 0.005973, yaw180)
        tried += 1
        if r['ok']:
            passing.append(dict(d=d, lat=lat, lift=lift, sl=-0.005973, sr=0.005973, score=0.0))
            break
        key = 'scan-seed ' + r['reason'][:70]
        fails[key] = fails.get(key, 0) + 1
    for idx, si, ri, sc in ([] if passing else cands):
        d, lat, lift = float(D_GRID[idx[0]]), float(LAT_GRID[idx[1]]), (lift_of(idx[2]) if split else float(LIFT_GRID[idx[2]]))
        sl, sr = slider_values(si, ri)
        r = feasible(R, cab, tg, shelf, mode, pull, d, lat, lift, sl, sr, yaw180)
        tried += 1
        if r['ok']:
            passing.append(dict(d=d, lat=lat, lift=lift, sl=sl, sr=sr, score=sc))
            if len(passing) >= 6:
                break
        else:
            key = r['reason'].split(':')[0] + ':' + (r['reason'].split(':')[1] if ':' in r['reason'] else '')[:40]
            fails[key] = fails.get(key, 0) + 1
    res['candidates_fully_evaluated'] = tried
    res['full_eval_failures'] = fails
    if not passing:
        res.update(feasible=False, reason='every map-feasible candidate fails the full path / collision evaluation (see full_eval_failures)', elapsed=time.time() - t0)
        return res
    LL, LH = -0.6235, 0.6265
    best = None
    for cand in passing[:4]:
        d, lat, lift, sl, sr = cand['d'], cand['lat'], cand['lift'], cand['sl'], cand['sr']
        for _ in range(2):
            for w in ((0, 1) if split else (0,)):
                cur = lift[w] if split else lift
                lo, hi = interval(lambda x: feasible(R, cab, tg, shelf, mode, pull, d, lat, set_lift(lift, w, x), sl, sr, yaw180)['ok'], cur, 0.02, LL, LH)
                lift = set_lift(lift, w, round(max(LL, min(LH, (lo + hi) / 2)), 3))
            lo, hi = interval(lambda x: feasible(R, cab, tg, shelf, mode, pull, d, x, lift, sl, sr, yaw180)['ok'], lat, 0.01, -0.25, 0.25)
            lat = round((lo + hi) / 2, 3)
            lo, hi = interval(lambda x: feasible(R, cab, tg, shelf, mode, pull, x, lat, lift, sl, sr, yaw180)['ok'], d, 0.01, 0.0, 0.9)
            d = round((lo + hi) / 2, 3)
        nominal = feasible(R, cab, tg, shelf, mode, pull, d, lat, lift, sl, sr, yaw180)
        if not nominal['ok']:
            continue
        ok_c, bad = corners_pass(R, cab, tg, shelf, mode, pull, d, lat, lift, sl, sr, yaw180)
        entry = dict(d=d, lat=lat, lift=lift, sl=sl, sr=sr, corners_ok=ok_c, corner_failures=bad)
        if best is None or (ok_c and not best['corners_ok']):
            best = entry
        if ok_c:
            break
    if best is None:
        res.update(feasible=False, reason='candidates passed the first evaluation but not after centring', elapsed=time.time() - t0)
        return res
    d, lat, lift, sl, sr = best['d'], best['lat'], best['lift'], best['sl'], best['sr']
    dlo, dhi = interval(lambda x: feasible(R, cab, tg, shelf, mode, pull, x, lat, lift, sl, sr, yaw180)['ok'], d, 0.01, 0.0, 0.9)
    alo, ahi = interval(lambda x: feasible(R, cab, tg, shelf, mode, pull, d, x, lift, sl, sr, yaw180)['ok'], lat, 0.01, -0.25, 0.25)
    lw = []
    for w in ((0, 1) if split else (0,)):
        cur = lift[w] if split else lift
        lw.append(list(interval(lambda x: feasible(R, cab, tg, shelf, mode, pull, d, lat, set_lift(lift, w, x), sl, sr, yaw180)['ok'], cur, 0.02, LL, LH)))
    ev, seqs = make_eval(R, cab, tg, shelf, mode, pull, d, lat, lift, sl, sr, yaw180=yaw180)
    rr = ev.evaluate(seqs)
    clr = {}
    for part in ('left', 'right'):
        for k, v in rr[part]['clr'].items():
            clr[f'{part}.{k}'] = round(v, 4)
    manip = {}
    for part in ('left', 'right'):
        qs = [q for nm, q in rr[part]['key_q'].items() if np.linalg.norm(q) > 1e-6]
        manip[part] = round(min(R.manipulability(part, q)[0] for q in qs), 4)
    res.update(feasible=True, best_d=d, lateral_offset_m=lat, lift=lift, slider_left=sl, slider_right=sr, feasible_d_window_m=[dlo, dhi], feasible_lift_window_m=lw if split else lw[0],
               feasible_lateral_window_m=[alo, ahi], corners_box_ok=best['corners_ok'], corner_failures=best['corner_failures'], min_clearance_m=clr,
               joint_margin_rad=round(min(rr['left']['jmargin'], rr['right']['jmargin']), 4), min_manipulability=manip, stage_variant=rr['stage_variant'],
               max_ik_jump_rad=round(max(rr['left']['maxjump'], rr['right']['maxjump']), 4), elapsed=time.time() - t0)
    return res


def _init():
    os.nice(19)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--modes', default='front,topdown')
    ap.add_argument('--targets', default='all')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--yaw180', action='store_true', help='E2 extra variant: suction tool turned 180 deg about the pad axis')
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    names = list(TARGETS) if a.targets == 'all' else a.targets.split(',')
    jobs = [(t, m, a.yaw180) for m in a.modes.split(',') for t in names]
    with mp.Pool(min(4, a.workers), initializer=_init) as pool:
        out = []
        for r in pool.imap(study_target, jobs):
            out.append(r)
            print(r['target'], r['mode'], 'feasible' if r.get('feasible') else 'NO', r.get('best_d'), r.get('lateral_offset_m'), r.get('lift'), r.get('feasible_d_window_m'), r.get('corners_box_ok'), r.get('reason', ''), r.get('full_eval_failures', ''), flush=True)
    fn = OUT / 'tests' / 'results' / (a.out or ('amr_parking_yaw180.json' if a.yaw180 else 'amr_parking_raw.json'))
    fn.write_text(json.dumps(out, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)))


if __name__ == '__main__':
    main()
