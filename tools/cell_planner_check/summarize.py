#!/usr/bin/env python3
"""Markdown summary + agreement statistics of one or more cellcheck result JSONs against the offline study.

  python3 summarize.py result_a.json [result_b.json ...] --out summary.md [--title "..."]
"""
import argparse, json, os, sys
from collections import Counter
import numpy as np


def first_failure(p):
    for tid, t in p.get('tasks', {}).items():
        if not t.get('ok', True):
            return tid, t.get('reason', 'failed')
    for ph in p.get('phases', []):
        for pc in ph.get('pair', []):
            if not pc.get('ok') and not pc.get('serial_ok'):
                return '/'.join(ph['tasks']), 'pair collision, no serial order valid (%s)' % pc.get('contacts', [{}])[0:1]
    return None, None


def q_close(p, job_pl=None):
    """max |dq| (deg) between the MoveIt joint solutions at the stroke start (end of 'enter') / end and the study joint paths"""
    out = []
    st = (p.get('study') or {}).get('tasks', {})
    for tid, t in p.get('tasks', {}).items():
        sq = (st.get(tid) or {}).get('q')
        if not sq or not t.get('ok'):
            continue
        steps = {s['name']: s for s in t['steps']}
        if 'enter' in steps and 'stroke' in steps:
            a = np.array(steps['enter']['q_end_deg'])
            b = np.array(steps['stroke']['q_end_deg'])
            sa = np.degrees(np.array(sq[0]))
            sb = np.degrees(np.array(sq[-1]))
            out.append(dict(task=tid, dq_start=float(np.abs(a - sa).max()), dq_end=float(np.abs(b - sb).max())))
    return out


def reason_class(r):
    if r is None:
        return None
    r = r.lower()
    for key, cls in (('no_ik_solution', 'KDL: no IK solution'), ('ik_solution_in_collision', 'IK exists but in collision'),
                     ('ik fails along the line', 'Cartesian: IK fails along the stroke'), ('collision at', 'Cartesian: collision along the stroke'),
                     ('joint jump', 'Cartesian: joint jump'), ('branch flip', 'Cartesian: wrist/joint branch flip'), ('pair collision', 'arm-arm collision (no serial order)'),
                     ('plan', 'transit planning failed'), ('exception', 'exception')):
        if key in r:
            return cls
    return r[:60]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('results', nargs='+')
    ap.add_argument('--out', required=True)
    ap.add_argument('--title', default='Planner-level validation summary')
    a = ap.parse_args()
    L = ['# %s' % a.title, '']
    for rf in a.results:
        R = json.load(open(rf))
        P = R['placements']
        L += ['## %s  (design `%s`, layout h=%.2f d=%.2f s=%.3f)' % (R.get('job'), R.get('design'), R['layout']['h'], R['layout']['d'], R['layout']['s']), '',
              'build `%s`; domain %s; scaling sets %s' % (R.get('build'), R.get('domain'), json.dumps(R.get('scaling'))), '']
        names = R['scaling']['names']
        # agreement
        tab = Counter()
        for p in P:
            tab[(bool(p['study']['feasible']), bool(p.get('feasible')))] += 1
        n = len(P)
        L += ['**Agreement (placement feasible: study / MoveIt)**: both feasible %d, both infeasible %d, study-only %d, MoveIt-only %d, of %d (agree %.0f%%)' %
              (tab[(True, True)], tab[(False, False)], tab[(True, False)], tab[(False, True)], n, 100.0 * (tab[(True, True)] + tab[(False, False)]) / max(n, 1)), '']
        has_serial = any('serial_ok' in p['study'] for p in P)
        if has_serial:
            def msim(p):
                return bool(p.get('feasible')) and not any(any(ph.get('serialised', [])) for ph in p.get('phases', []))
            t2 = Counter()
            for p in P:
                t2[(bool(p['study']['feasible']), msim(p))] += 1
            t3 = Counter()
            for p in P:
                t3[(bool(p['study'].get('serial_ok')), bool(p.get('feasible')))] += 1
            L += ['**v2 view, simultaneous (study `simul` / MoveIt feasible with no serialised phase)**: both %d, neither %d, study-only %d, MoveIt-only %d' %
                  (t2[(True, True)], t2[(False, False)], t2[(True, False)], t2[(False, True)]),
                  '**v2 view, serial allowed (study `serial_ok` / MoveIt feasible, serialisation allowed)**: both %d, neither %d, study-only %d, MoveIt-only %d' %
                  (t3[(True, True)], t3[(False, False)], t3[(True, False)], t3[(False, True)]), '']
        for box in sorted({p['box'] for p in P}):
            pb = [p for p in P if p['box'] == box]
            ns = sum(1 for p in pb if p['study']['feasible'])
            nm = sum(1 for p in pb if p.get('feasible'))
            L.append('- %s: study feasible %d / %d (%.1f%%), MoveIt feasible %d / %d (%.1f%%)' % (box, ns, len(pb), 100.0 * ns / len(pb), nm, len(pb), 100.0 * nm / len(pb)))
        L.append('')
        hdr = '| placement | study | alloc (study) | MoveIt | first failure (MoveIt) | pairs serialised | ' + ' | '.join('cycle %s (s)' % nm for nm in names) + ' | max dq vs study (deg) start / end |'
        L += [hdr, '|' + '--|' * (hdr.count('|') - 1)]
        fails = Counter()
        cyc = {nm: [] for nm in names}
        for p in P:
            tid, why = first_failure(p)
            if why:
                fails[reason_class(why)] += 1
            ser = []
            for ph in p.get('phases', []):
                if any(ph.get('serialised', [])):
                    ser.append('/'.join(x.split(':')[-1] + '@' + x.split(':')[0][-1] for x in ph['tasks']))
            qc = q_close(p)
            dq = '%.1f / %.1f' % (max(x['dq_start'] for x in qc), max(x['dq_end'] for x in qc)) if qc else '-'
            cy = p.get('cycle_s') or {}
            for nm in names:
                if cy.get(nm) is not None:
                    cyc[nm].append(cy[nm])
            L.append('| %s | %s | %s | %s | %s | %s | %s | %s |' % (
                p['id'], 'yes' if p['study']['feasible'] else 'no', str(p['study'].get('alloc') or '-') + ('*' if p['study'].get('best_effort') else ''),
                'yes' if p.get('feasible') else 'no', ('%s: %s' % (tid, why))[:150] if why else '', ', '.join(ser) if ser else '',
                ' | '.join(('%.1f' % cy[nm]) if cy.get(nm) is not None else '-' for nm in names), dq))
        L += ['', '(* = study-infeasible placement evaluated with its best-effort allocation and nominal strokes)', '']
        if fails:
            L += ['**MoveIt failure classes**: ' + '; '.join('%s x%d' % kv for kv in fails.most_common()), '']
        for nm in names:
            if cyc[nm]:
                L.append('- cycle time %s over %d feasible placements: mean %.1f s, min %.1f, max %.1f (includes snapshot %.1f s)' %
                         (nm, len(cyc[nm]), np.mean(cyc[nm]), min(cyc[nm]), max(cyc[nm]), 2.0))
        L.append('')
    open(a.out, 'w').write('\n'.join(L) + '\n')
    print('wrote', a.out)


if __name__ == '__main__':
    main()
