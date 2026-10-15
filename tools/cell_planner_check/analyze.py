#!/usr/bin/env python3
"""Cross-experiment analysis: study vs MoveIt (real geometry) vs MoveIt with the study's sphere proxies, planner statistics, failure causes.

  python3 analyze.py --real <result.json> [--proxy <result.json>] --out analysis.md [--title ..]
"""
import argparse, json, os, sys
from collections import Counter, defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from summarize import first_failure, reason_class


def planner_stats(P):
    st = Counter()
    ptime = defaultdict(list)
    errs = Counter()
    for p in P:
        for tid, t in p.get('tasks', {}).items():
            for s in t.get('steps', []):
                if s.get('type') == 'transit' and s.get('ok'):
                    for k, pl in enumerate(s.get('planner') or []):
                        st[pl] += 1
                        if s.get('planning_time'):
                            ptime[pl].append(s['planning_time'][k])
                for tr in s.get('tried', []) or []:
                    if tr.get('plan_error'):
                        for part in tr['plan_error'].split(' / '):
                            for e in part.replace('#', ' ').split():
                                if ':' in e and e.split(':')[-1].isupper():
                                    errs[e.split(':')[-1]] += 1
                    if tr.get('ik_reason', '').startswith('no_ik'):
                        errs['KDL_NO_IK(candidate)'] += 1
                        if tr.get('study_closed_form_margin_deg') is not None:
                            errs['KDL_NO_IK_but_closed_form_solution_exists'] += 1
                        elif 'study_closed_form_margin_deg' in tr:
                            errs['KDL_NO_IK_and_closed_form_agrees_none'] += 1
                    if tr.get('ik_reason') == 'ik_solution_in_collision':
                        errs['IK_IN_COLLISION(candidate)'] += 1
    return st, {k: (float(np.mean(v)), float(np.max(v)), len(v)) for k, v in ptime.items()}, errs


def study_speed_cycle(p, jp, set_idx, v_cut=0.010, v_app=0.030, dwell=1.0, snapshot=2.0):
    """cycle time of a feasible MoveIt placement with the Cartesian steps re-timed at the v2 study's speeds (cut v_cut, 30 mm approach/retreat at v_app) and the
    study's 1 s dwell per look; transits keep their MoveIt (Pilz/OMPL + TOTG) durations of scaling set `set_idx`."""
    if not p.get('feasible') or jp is None:
        return None
    import cellgeom as G
    def seg_len(a, b):
        return float(np.linalg.norm(np.array(a['xyz']) - np.array(b['xyz'])))
    tt = {}
    for tid, t in p['tasks'].items():
        jt = jp['tasks'].get(tid)
        tot = 0.0
        for st in t['steps']:
            d = st['duration'][set_idx] if isinstance(st.get('duration'), list) else 0.0
            if jt is not None and st['name'] in ('enter', 'stroke', 'retreat') and jt.get('start') and jt.get('end'):
                if st['name'] == 'stroke':
                    d = seg_len(jt['start'], jt['end']) / v_cut
                else:
                    wp = [x for x in jt['steps'] if x['name'] == st['name']][0]['waypoints'][0]
                    other = jt['start'] if st['name'] == 'enter' else jt['end']
                    d = seg_len(wp, other) / v_app
            if st['name'] == 'look':
                d += dwell
            tot += d
        tt[tid] = tot
    total = snapshot
    for ph in p['phases']:
        ds = [tt[x] for x in ph['tasks'] if x in tt]
        if not ds:
            continue
        ser = ph.get('serialised', [False])[set_idx] if ph.get('serialised') else False
        total += sum(ds) if ser else max(ds)
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--real', required=True)
    ap.add_argument('--proxy', default=None)
    ap.add_argument('--job', default=None, help='job JSON of the real run: adds the cycle time re-timed at the study speeds')
    ap.add_argument('--out', required=True)
    ap.add_argument('--title', default='Study vs MoveIt analysis')
    a = ap.parse_args()
    R = json.load(open(a.real))
    X = json.load(open(a.proxy)) if a.proxy else None
    pr = {p['id']: p for p in X['placements']} if X else {}
    L = ['# %s' % a.title, '', 'real geometry: `%s`' % a.real] + (['proxy geometry: `%s`' % a.proxy] if a.proxy else []) + ['']
    # per placement three-way table
    hdr = '| placement | study | MoveIt real | ' + ('MoveIt proxy | ' if X else '') + 'cycle exec / fast (real) | real: first failure | ' + ('proxy: first failure |' if X else '')
    L += [hdr, '|' + '--|' * (hdr.count('|') - 1)]
    cnt = Counter()
    for p in R['placements']:
        q = pr.get(p['id'])
        tid, why = first_failure(p)
        row = [p['id'], 'yes' if p['study']['feasible'] else 'no', 'yes' if p.get('feasible') else 'no']
        if X:
            row.append('yes' if q and q.get('feasible') else 'no')
        cy = p.get('cycle_s') or {}
        ks = list(cy.keys())
        row.append(' / '.join('%.1f' % cy[k] if cy.get(k) is not None else '-' for k in ks) if ks else '-')
        row.append(('%s: %s' % (tid, reason_class(why))) if why else '')
        if X:
            t2, w2 = first_failure(q) if q else (None, None)
            row.append(('%s: %s' % (t2, reason_class(w2))) if w2 else '')
        cnt[(bool(p['study']['feasible']), bool(p.get('feasible')), bool(q and q.get('feasible')) if X else None)] += 1
        L.append('| ' + ' | '.join(row) + ' |')
    L += ['']
    L.append('**Verdict counts (study, real, proxy)**: ' + '; '.join('%s -> %d' % (k, v) for k, v in sorted(cnt.items(), key=lambda kv: -kv[1])))
    L.append('')
    for name, res in (('real', R), ('proxy', X)):
        if res is None:
            continue
        st, pt, errs = planner_stats(res['placements'])
        L.append('**Planner statistics (%s geometry)**: transit plans by planner %s; planning time mean/max/n %s; recorded planner/IK errors %s' % (name, dict(st), pt, dict(errs)))
        L.append('')
    if a.job:
        J = {pl['id']: pl for pl in json.load(open(a.job))['placements']}
        names = R['scaling']['names']
        si = names.index('fast') if 'fast' in names else len(names) - 1
        rows = []
        for p in R['placements']:
            c = study_speed_cycle(p, J.get(p['id']), si)
            if c is not None:
                cm = ((J[p['id']]['study'].get('cycle_model_s') or {}).get('b_v10') or {}).get('cycle')
                rows.append((p['id'], c, cm))
        if rows:
            L += ['**Cycle time with the Cartesian steps re-timed at the study speeds (10 mm/s cut, 30 mm/s approach/retreat, 1 s look dwell, MoveIt transit durations at %s scaling) vs the study model b_v10**' % names[si], '',
                  '| placement | MoveIt, study speeds (s) | study b_v10 (s) |', '|--|--|--|']
            for r in rows:
                L.append('| %s | %.1f | %s |' % (r[0], r[1], ('%.1f' % r[2]) if r[2] else '-'))
            L += ['', '- median MoveIt (study speeds) %.1f s over %d placements; study b_v10 median %.1f s' % (np.median([r[1] for r in rows]), len(rows), np.median([r[2] for r in rows if r[2]]) if any(r[2] for r in rows) else float('nan')), '']
    open(a.out, 'w').write('\n'.join(L) + '\n')
    print('wrote', a.out)


if __name__ == '__main__':
    main()
