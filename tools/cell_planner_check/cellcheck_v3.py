#!/usr/bin/env python3
"""v3 extension of cellcheck.py (existing harness untouched): the idle arm waits at a PARK POSE taken from the job (`park_q`: {arm: [6 joints]}) instead of the neutral upright
pose, and every task result carries the minimum joint-limit margin of its planned trajectory.  Everything else (IK, Pilz PTP / OMPL transits, Cartesian strokes, ACM allowance,
branch-flip check, cycle) is the unchanged Checker.

  export ROS_DOMAIN_ID=187 ROS_LOCALHOST_ONLY=1
  python3 cellcheck_v3.py --job jobs/<job>.json --build build/<cell> --out <result_dir> [--placements ...]
Serial jobs: one task per phase, explicit '<arm>:park' tasks between arm switches (build_job_v3.py writes them); run with the implicit park of cellcheck disabled (this script does).
"""
import sys as _s; _s.dont_write_bytecode = True
import argparse, json, math, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cellcheck as CC
import numpy as np

LIM = {'joint1': (-178, 178), 'joint2': (-79, 139), 'joint3': (-79, 139), 'joint4': (-178, 178), 'joint5': (-94, 94), 'joint6': (-178, 178)}
LO = np.array([LIM['joint%d' % i][0] for i in range(1, 7)]); HI = np.array([LIM['joint%d' % i][1] for i in range(1, 7)])


def min_margin_deg(segs):
    if not segs:
        return None
    Q = np.degrees(np.array([q for _, q in segs]))
    return float(np.minimum(Q - LO, HI - Q).min())


class Checker3(CC.Checker):
    def __init__(self, meta, job, opts):
        super().__init__(meta, job, opts)
        pq = job.get('park_q')
        if pq:
            self.neutral = {a: list(pq[a]) for a in self.arms}

    def run_task(self, task, q_state, study_task=None, partner_by_step=None):
        out = super().run_task(task, q_state, study_task, partner_by_step)
        try:
            out['min_margin_deg'] = [min_margin_deg(out['segments'][si]) for si in range(self.nsets)]
        except Exception:
            out['min_margin_deg'] = None
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--job', required=True)
    ap.add_argument('--build', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--placements', default=None)
    ap.add_argument('--planning-time', type=float, default=5.0)
    ap.add_argument('--ik-timeout', type=float, default=0.5)
    ap.add_argument('--transit-planners', default='pilz:PTP,ompl:RRTConnectkConfigDefault')
    a = ap.parse_args()
    b = a.build if os.path.isabs(a.build) else os.path.join(HERE, a.build)
    meta = json.load(open(os.path.join(b, 'cell_meta.json')))
    job = json.load(open(a.job))
    if abs(job['layout']['s'] - meta['s']) > 1e-6:
        sys.exit('job layout s=%s does not match the running cell build s=%s' % (job['layout']['s'], meta['s']))
    os.makedirs(a.out, exist_ok=True)
    chk = Checker3(meta, job, dict(planning_time=a.planning_time, ik_timeout=a.ik_timeout, transit_planners=a.transit_planners.split(','), concurrent_other='neutral',
                                   park_idle=False, partner_pass2=False))
    sel = set(a.placements.split(',')) if a.placements else None
    res = dict(schema='cell_planner_check/result/v1', job=job.get('name'), design=meta['design'], build=b, layout=job['layout'], scaling=chk.sc,
               domain=os.environ.get('ROS_DOMAIN_ID'), park_q=job.get('park_q'), placements=[])
    outf = os.path.join(a.out, 'result_%s.json' % job.get('name', 'job'))
    for pl in job['placements']:
        if sel and pl['id'] not in sel:
            continue
        t0 = time.time()
        r = chk.run_placement(pl)
        res['placements'].append(CC.strip(r))
        st = (pl.get('study') or {}).get('feasible')
        print('%-26s study=%-5s moveit=%-5s cycle=%s  %.0fs' % (pl['id'], st, r.get('feasible'), r.get('cycle_s'), time.time() - t0), flush=True)
        json.dump(res, open(outf, 'w'), indent=1)
    res['service_calls'] = chk.cl.ncalls
    res['service_time_s'] = chk.cl.tcalls
    json.dump(res, open(outf, 'w'), indent=1)
    print('wrote', outf)
    chk.cl.destroy_node()
    CC.rclpy.shutdown()


if __name__ == '__main__':
    main()
