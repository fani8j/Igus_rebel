#!/usr/bin/env python3
"""Direct audit of the study's own joint paths in MoveIt's collision checker (no planning involved).

For every study-feasible placement of a job: the stroke samples (5 joint vectors per task) and a dense joint-space interpolation between them are checked with
/check_state_validity (the other arm parked at neutral, as the study assumes), and for each concurrent pair the study's paired samples (k,k) and (k,k+-1) of both
arms are checked together.  Reports which study-feasible paths MoveIt's geometry rejects and by which link pairs.  Needs the stack up on domain 187.

  python3 path_audit.py --job jobs/h000n.0_v1.json --build build/h000n.0_s0355 --out audit.json [--dense_deg 1.0]
"""
import argparse, json, math, os, sys
from collections import Counter
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cellgeom as G
if os.environ.get('ROS_DOMAIN_ID') != '187' or os.environ.get('ROS_LOCALHOST_ONLY') != '1':
    sys.exit('export ROS_DOMAIN_ID=187 ROS_LOCALHOST_ONLY=1 first')
import moveit_client as MC
import rclpy

CARTON, TABLE = 'detected_carton', 'table'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--job', required=True)
    ap.add_argument('--build', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--dense_deg', type=float, default=1.0)
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.build if os.path.isabs(a.build) else os.path.join(HERE, a.build), 'cell_meta.json')))
    job = json.load(open(a.job))
    MC.init()
    cl = MC.MoveItClient(meta, 'path_audit')
    z = [0.0] * 6
    rows = []
    cont = Counter()
    pair_cont = Counter()
    for pl in job['placements']:
        if not pl['study']['feasible'] or not pl.get('tasks'):
            continue
        cg = pl['carton']
        tb = job.get('table', {})
        x0, x1, hy, th = tb.get('x0', 0.12), tb.get('x1', 1.2), tb.get('half_y', 1.0), tb.get('thickness', 0.05)
        ztop = cg['center'][2] - cg['size'][2] / 2
        cl.set_scene([dict(name=CARTON, size=cg['size'], centre=cg['center'], quat=cg['quat']),
                      dict(name=TABLE, size=[x1 - x0, 2 * hy, th], centre=[(x0 + x1) / 2, 0.0, ztop - th / 2], quat=[0, 0, 0, 1])], remove=[CARTON, TABLE])
        q5 = {}
        for tid, st in pl['study']['tasks'].items():
            if st.get('q'):
                q5[tid] = np.array(st['q'])
        # single arm strokes: samples + dense interpolation, other arm neutral
        for tid, Q in q5.items():
            arm = tid.split(':')[0]
            other = [x for x in G.ARMS if x != arm][0]
            bad_s, bad_d = [], []
            c_here = Counter()
            for k in range(len(Q)):
                v, c = cl.validity({arm: Q[k].tolist(), other: z}, 'both')
                if not v:
                    bad_s.append(k)
                    for x in c:
                        c_here[x] += 1
            n_dense = 0
            for k in range(len(Q) - 1):
                n = max(1, int(math.ceil(np.degrees(np.abs(Q[k + 1] - Q[k]).max()) / a.dense_deg)))
                for j in range(1, n):
                    qq = Q[k] + (Q[k + 1] - Q[k]) * (j / n)
                    n_dense += 1
                    v, c = cl.validity({arm: qq.tolist(), other: z}, 'both')
                    if not v:
                        bad_d.append((k, j))
                        for x in c:
                            c_here[x] += 1
            for x, n in c_here.items():
                cont[x] += 1
            rows.append(dict(placement=pl['id'], task=tid, samples_bad=bad_s, dense_checked=n_dense, dense_bad=len(bad_d), contacts=[list(x) for x in c_here][:8]))
        # concurrent pairs: the study's paired samples (k,k), (k,k+1), (k+1,k) of both arms
        for ph in pl['phases']:
            if len(ph) != 2 or not all(t in q5 for t in ph):
                continue
            A, B = ph
            aA, aB = A.split(':')[0], B.split(':')[0]
            n = len(q5[A])
            offs = [(k, k) for k in range(n)] + [(k, k + 1) for k in range(n - 1)] + [(k + 1, k) for k in range(n - 1)]
            bad = []
            cc = Counter()
            for (i, j) in offs:
                v, c = cl.validity({aA: q5[A][i].tolist(), aB: q5[B][j].tolist()}, 'both')
                if not v:
                    bad.append((i, j))
                    for x in c:
                        cc[x] += 1
            for x in cc:
                pair_cont[x] += 1
            rows.append(dict(placement=pl['id'], pair=[A, B], paired_samples=len(offs), paired_bad=len(bad), contacts=[list(x) for x in cc][:8]))
    ntask = [r for r in rows if 'task' in r]
    npair = [r for r in rows if 'pair' in r]
    summ = dict(job=a.job, build=a.build, geometry=meta.get('geometry'), tasks_checked=len(ntask),
                tasks_with_bad_sample=sum(1 for r in ntask if r['samples_bad']), tasks_with_bad_dense_only=sum(1 for r in ntask if not r['samples_bad'] and r['dense_bad']),
                tasks_clean=sum(1 for r in ntask if not r['samples_bad'] and not r['dense_bad']),
                pairs_checked=len(npair), pairs_with_bad=sum(1 for r in npair if r['paired_bad']),
                top_task_contacts=[('~'.join(k), v) for k, v in cont.most_common(10)], top_pair_contacts=[('~'.join(k), v) for k, v in pair_cont.most_common(10)])
    json.dump(dict(summary=summ, rows=rows), open(a.out, 'w'), indent=1)
    print(json.dumps(summ, indent=1))
    cl.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
