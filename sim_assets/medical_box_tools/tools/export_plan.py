"""Export a time-parametrised joint-space plan (both arms) for one parking result, for tests/test_cabinet_grasp_robot.py.
  python3 tools/export_plan.py --parking tests/results/amr_parking_front.json --mode front [--targets T3,T5] [--yaw180]
-> data/plans/<mode>_<target>[_yaw180].json
Time parametrisation (stand-in for the MoveIt time parametrisation, same limits as rebel_medicine_grasp_moveit_config/config/joint_limits.yaml): joint velocity <= 0.785 rad/s,
acceleration <= 5.24 rad/s^2, scaled by VEL_SCALE_CART / ACC_SCALE_CART on straight-line (contact / approach) segments and by VEL_SCALE_JOINT / ACC_SCALE_JOINT on free-space joint
transits; the tool speed of a Cartesian segment is also capped (seg.speed); every segment starts and ends at rest."""
import argparse, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import parking as pk, parking_study as ps, robot_kin as rk

VMAX, AMAX = 0.785, 5.24
VEL_SCALE_CART, ACC_SCALE_CART = 1.0, 0.5
VEL_SCALE_JOINT, ACC_SCALE_JOINT = 0.5, 0.5


def time_param(q0, qs, ds_tool, tool_speed, vscale, ascale):
    """q0: start (6,), qs: (n,6) samples after it, ds_tool: tool path length per sample (n,) or None -> sample times (n,) from 0 (rest to rest)"""
    pts = np.vstack([q0, qs])
    du = np.max(np.abs(np.diff(pts, axis=0)), axis=1)              # L-inf joint step per sample
    n = len(du)
    vlim = np.full(n, VMAX * vscale)
    if ds_tool is not None and tool_speed > 0:
        with np.errstate(divide='ignore', invalid='ignore'):
            vt = np.where(ds_tool > 1e-9, tool_speed * du / ds_tool, np.inf)
        vlim = np.minimum(vlim, vt)
    a = AMAX * ascale
    v = np.zeros(n + 1)                                            # speed at the sample boundaries (rest at both ends)
    for i in range(n):
        v[i + 1] = min(np.sqrt(v[i] ** 2 + 2 * a * du[i]), vlim[i])
    v[n] = 0.0
    for i in range(n - 1, -1, -1):
        v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2 * a * du[i]))
    dt = np.zeros(n)
    for i in range(n):
        vm = max(0.5 * (v[i] + v[i + 1]), 1e-4)
        dt[i] = du[i] / vm if du[i] > 1e-12 else 0.0
    return np.cumsum(dt)


def export(R, cab, tname, mode, par, yaw180):
    shelf, box, which = ps.TARGETS[tname]
    tg = cab.pick(shelf, box, which)
    pull = ps.PULL_BY_MODE_SHELF.get((mode, shelf), rk.PULL_OUT)
    lift = par['lift']
    ev, seqs = ps.make_eval(R, cab, tg, shelf, mode, pull, par['best_d'], par['lateral_offset_m'], tuple(lift) if isinstance(lift, list) else lift,
                            par['slider_left'], par['slider_right'], yaw180=yaw180)
    r = ev.evaluate(seqs, record=True)
    assert r['ok'], r['reason']
    W = rk.World(cab, shelf, tg, par['best_d'], par['lateral_offset_m'])
    plan = dict(target=tname, mode=mode, shelf=shelf, box=box, which=which, carton=tg['name'], pull_m=pull, yaw180=yaw180, d=par['best_d'], lat=par['lateral_offset_m'],
                lift=lift, slider_left=par['slider_left'], slider_right=par['slider_right'],
                cabinet_to_world=dict(R=W.R.tolist(), t=W.t.tolist()), stage_variant=r['stage_variant'], arms={}, scales=dict(cart=[VEL_SCALE_CART, ACC_SCALE_CART], joint=[VEL_SCALE_JOINT, ACC_SCALE_JOINT]))
    for arm, seq in (('left', r['left_seq']), ('right', r['right_seq'])):
        segs = {s.name: s for s in seq}
        q = np.zeros(6)
        out = []
        for tr in r[arm]['traj']:
            sg = segs[tr['name']]
            qs = np.asarray(tr['q'])
            if sg.kind == 'cart' and sg.speed > 0:
                tot = float(np.linalg.norm(sg.T1[:3, 3] - sg.T0[:3, 3]))
                t = time_param(q, qs, np.full(len(qs), tot / len(qs)), sg.speed, VEL_SCALE_CART, ACC_SCALE_CART)
            elif sg.kind == 'cart':
                t = np.zeros(1); qs = qs[-1:]
            else:
                t = time_param(q, qs, None, 0.0, VEL_SCALE_JOINT, ACC_SCALE_JOINT)
            out.append(dict(name=tr['name'], kind=sg.kind, speed=sg.speed, fingers=tr['fingers'], q0=q.tolist(), q=qs.tolist(), t=t.tolist(), duration=float(t[-1]) if len(t) else 0.0))
            q = qs[-1]
        plan['arms'][arm] = out
    plan['clearance_planned_m'] = {f'{p}.{k}': round(v, 4) for p in ('left', 'right') for k, v in r[p]['clr'].items()}
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parking', required=True)
    ap.add_argument('--mode', required=True)
    ap.add_argument('--targets', default='all')
    ap.add_argument('--yaw180', action='store_true')
    a = ap.parse_args()
    R, cab = rk.Robot(), rk.Cabinet()
    pr = {r['target']: r for r in json.load(open(a.parking)) if r['mode'] == a.mode and r.get('feasible') and r.get('yaw180', False) == a.yaw180}
    names = list(pr) if a.targets == 'all' else a.targets.split(',')
    outdir = HERE.parent / 'data' / 'plans'
    outdir.mkdir(exist_ok=True, parents=True)
    for tn in names:
        if tn not in pr:
            print(tn, 'no feasible parking result'); continue
        plan = export(R, cab, tn, a.mode, pr[tn], a.yaw180)
        fn = outdir / f"{a.mode}_{tn}{'_yaw180' if a.yaw180 else ''}.json"
        fn.write_text(json.dumps(plan))
        print(tn, fn.name, {k: [round(s['duration'], 1) for s in v] for k, v in plan['arms'].items()})


if __name__ == '__main__':
    main()
