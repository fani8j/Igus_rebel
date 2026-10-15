#!/usr/bin/env python3
"""KDL IK audit against the study's closed-form IK.  For every stroke start/end pose of a job that the study solved (joint path known) ask /compute_ik
(collision checking OFF, so only the solver matters) from (a) the study seed, (b) the neutral (all-zero) seed, (c) N random in-limit seeds, and compare
with the study joints (same branch if max |dq| < 5 deg).  Needs the stack up on domain 187.

  python3 ik_audit.py --job jobs/h000n.0_v1.json --build build/h000n.0_s0355 --out result.json [--n_random 5]
"""
import argparse, json, math, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cellgeom as G
if os.environ.get('ROS_DOMAIN_ID') != '187' or os.environ.get('ROS_LOCALHOST_ONLY') != '1':
    sys.exit('export ROS_DOMAIN_ID=187 ROS_LOCALHOST_ONLY=1 first')
import moveit_client as MC
import rclpy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--job', required=True)
    ap.add_argument('--build', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n_random', type=int, default=5)
    ap.add_argument('--timeout', type=float, default=0.25)
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.build if os.path.isabs(a.build) else os.path.join(HERE, a.build), 'cell_meta.json')))
    job = json.load(open(a.job))
    MC.init()
    cl = MC.MoveItClient(meta, 'ik_audit')
    rng = np.random.default_rng(3)
    lo = np.array([-178, -79, -79, -178, -94, -178.0]) * math.pi / 180
    hi = np.array([178, 139, 139, 178, 94, 178.0]) * math.pi / 180
    rows = []
    z = [0.0] * 6
    for pl in job['placements']:
        for tid, task in pl['tasks'].items():
            st = pl['study']['tasks'].get(tid)
            if not st or not st.get('q'):
                continue
            arm = task['arm']
            other = [x for x in G.ARMS if x != arm][0]
            for end, pose, qs in (('start', task['start'], st['q'][0]), ('end', task['end'], st['q'][-1])):
                T = G.pose_to_T(pose)
                rec = dict(placement=pl['id'], task=tid, end=end)
                for name, seed in [('study_seed', qs), ('neutral_seed', z)] + [('rand%d' % i, list(rng.uniform(lo, hi))) for i in range(a.n_random)]:
                    q, e = cl.ik(arm, T, {arm: list(seed), other: z}, avoid_collisions=False, timeout_s=a.timeout)
                    if q is None:
                        rec[name] = dict(ok=False, err=e)
                    else:
                        dq = float(np.degrees(np.abs(np.array(q) - np.array(qs)).max()))
                        rec[name] = dict(ok=True, dq_vs_study_deg=dq, same_branch=bool(dq < 5.0))
                rows.append(rec)
    def rate(key, pred=lambda r: True):
        v = [r[key] for r in rows if key in r]
        return dict(n=len(v), ok=sum(1 for x in v if x['ok']), same_branch=sum(1 for x in v if x['ok'] and x['same_branch']))
    summ = dict(study_seed=rate('study_seed'), neutral_seed=rate('neutral_seed'))
    rr = [r['rand%d' % i] for r in rows for i in range(a.n_random)]
    summ['random_seeds'] = dict(n=len(rr), ok=sum(1 for x in rr if x['ok']), same_branch=sum(1 for x in rr if x['ok'] and x['same_branch']))
    json.dump(dict(job=a.job, kdl_timeout_s=a.timeout, summary=summ, rows=rows), open(a.out, 'w'), indent=1)
    print(json.dumps(summ, indent=1))
    cl.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
