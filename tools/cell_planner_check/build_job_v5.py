#!/usr/bin/env python3
"""v5 job builder: build_job_v3.build (unchanged) + ALTERNATIVE PARK POSES.  The `<arm>:park` transit tasks get the study's alternative park poses (plans file `park_alt`: {'0': [q..], '1': [q..]}, best first,
all collision-free and outside the working zone in the study) as extra explicit joint-goal candidates after the primary park pose; cellcheck_v5.py tries them in order with more planning attempts / time.

  python3 build_job_v5.py --plans <plans_<label>.json> --design v5_<label> --kind BIG|SMALL --out jobs/<name>.json
"""
import argparse, json, os, sys
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_job_v3 as BJ3
import cellgeom as G


def build(plans, design_name, kind, max_n=None):
    job = BJ3.build(plans, design_name, kind, max_n)
    alt = plans.get('park_alt') or {}
    n = 0
    for pl in job['placements']:
        for tid, task in pl.get('tasks', {}).items():
            if task.get('kind') != 'park': continue
            arm = task['arm']
            ai = [k for k in (0, 1) if G.ARMS[k] == arm][0]
            for st in task['steps']:
                if st['name'] == 'park':
                    for k, q in enumerate(alt.get(str(ai), [])):
                        st['target']['candidates'].append(dict(name='park_alt%d' % (k + 1), q_goal=list(q)))
                        n += 1
    job['park_alternatives_added'] = n
    job['source'] = job.get('source', '') + ' + alternative park poses'
    return job


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plans', required=True); ap.add_argument('--design', required=True); ap.add_argument('--kind', default='BIG'); ap.add_argument('--out', required=True); ap.add_argument('--max', type=int, default=None)
    a = ap.parse_args()
    job = build(json.load(open(a.plans)), a.design, a.kind, a.max)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(job, open(a.out, 'w'), indent=1)
    print('wrote %s: %d placements, %d alternative park candidates' % (a.out, len(job['placements']), job['park_alternatives_added']))


if __name__ == '__main__':
    main()
