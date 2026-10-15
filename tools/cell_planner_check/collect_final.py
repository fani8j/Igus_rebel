#!/usr/bin/env python3
"""Collect the headline numbers of result JSONs into one JSON (and print a compact text table).   python3 collect_final.py out.json result1.json [job1.json] result2.json ..."""
import json, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from summarize import q_close, first_failure, reason_class
from analyze import planner_stats, study_speed_cycle


def one(rf, jf=None):
    R = json.load(open(rf))
    P = R['placements']
    J = {pl['id']: pl for pl in json.load(open(jf))['placements']} if jf else {}
    n = len(P)
    sf = [p['id'] for p in P if p['study']['feasible']]
    mf = [p['id'] for p in P if p.get('feasible')]
    so = [p['id'] for p in P if p['study']['feasible'] and not p.get('feasible')]
    mo = [p['id'] for p in P if not p['study']['feasible'] and p.get('feasible')]
    names = R['scaling']['names']
    cyc = {nm: [p['cycle_s'][nm] for p in P if p.get('cycle_s') and p['cycle_s'].get(nm) is not None] for nm in names}
    si = names.index('fast') if 'fast' in names else len(names) - 1
    ss = [study_speed_cycle(p, J.get(p['id']), si) for p in P] if J else []
    ss = [x for x in ss if x is not None]
    both = [p for p in P if p['study']['feasible'] and p.get('feasible')]
    dq = []
    for p in both:
        for x in q_close(p):
            dq.append((p['id'], x['task'], x['dq_start'], x['dq_end']))
    st, pt, errs = planner_stats(P)
    fails = {}
    for p in P:
        t, w = first_failure(p)
        if w:
            fails[reason_class(w)] = fails.get(reason_class(w), 0) + 1
    ser = sum(1 for p in P if p.get('feasible') and any(any(ph.get('serialised', [])) for ph in p.get('phases', [])))
    simul = [p['id'] for p in P if p.get('feasible') and not any(any(ph.get('serialised', [])) for ph in p.get('phases', []))]
    out = dict(file=rf, job=R.get('job'), design=R.get('design'), layout=R['layout'], n=n, study_feasible=len(sf), moveit_feasible=len(mf),
               agree=int(sum(1 for p in P if bool(p['study']['feasible']) == bool(p.get('feasible')))), study_only=so, moveit_only=mo,
               moveit_simultaneous=len(simul), moveit_feasible_with_serialised_phase=ser,
               study_simul_ids=[p['id'] for p in P if p['study']['feasible']], study_serial_ok=[p['id'] for p in P if p['study'].get('serial_ok')],
               cycle={nm: (None if not v else dict(n=len(v), mean=float(np.mean(v)), median=float(np.median(v)), min=float(min(v)), max=float(max(v)))) for nm, v in cyc.items()},
               cycle_study_speeds=(None if not ss else dict(n=len(ss), mean=float(np.mean(ss)), median=float(np.median(ss)), min=float(min(ss)), max=float(max(ss)))),
               study_cycle_b_v10=[(J[p['id']]['study'].get('cycle_model_s') or {}).get('b_v10', {}).get('cycle') for p in both if J.get(p['id'])] if J else None,
               dq_vs_study_gt5deg=[x for x in dq if x[2] > 5 or x[3] > 5], dq_n=len(dq), failure_classes=fails,
               planners=dict(st), planning_time=pt, errors=dict(errs))
    return out


def main():
    out = sys.argv[1]
    args = sys.argv[2:]
    res = []
    i = 0
    while i < len(args):
        rf = args[i]
        jf = None
        if i + 1 < len(args) and 'jobs' in args[i + 1] and 'result' not in os.path.basename(args[i + 1]):
            jf = args[i + 1]
            i += 1
        res.append(one(rf, jf))
        i += 1
    json.dump(res, open(out, 'w'), indent=1)
    for r in res:
        print('%-28s n=%2d study=%2d moveit=%2d agree=%2d study-only=%s moveit-only=%s simul=%d serial-phase=%d' % (
            r['job'], r['n'], r['study_feasible'], r['moveit_feasible'], r['agree'], r['study_only'], r['moveit_only'], r['moveit_simultaneous'], r['moveit_feasible_with_serialised_phase']))
        print('   cycle', {k: (None if v is None else (round(v['median'], 1), round(v['min'], 1), round(v['max'], 1))) for k, v in r['cycle'].items()}, 'study-speeds', None if not r['cycle_study_speeds'] else round(r['cycle_study_speeds']['median'], 1),
              'dq>5deg', len(r['dq_vs_study_gt5deg']), '/', r['dq_n'])


if __name__ == '__main__':
    main()
