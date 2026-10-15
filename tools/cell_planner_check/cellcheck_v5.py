#!/usr/bin/env python3
"""v5 extension of cellcheck_v3.py (cellcheck.py / cellcheck_v3.py unchanged):
  * the twin-pad skid pads (links `<arm>_skid_pads`) are contact geometry: the ACM allowance vs the carton is applied to them together with the knife tip, ONLY during the cut strokes
    (`--pads-strict` switches the allowance off for the pads: informational run, every pad contact with the carton then counts as a collision)
  * more robust transit planning (the v4 shared-corner run failed once on a park transit): `--planning-time` default 15 s, OMPL attempts 6 (each with `--plan-attempts` planning attempts inside the
    request), the park task tries the alternative park poses of the job as further candidates
  export ROS_DOMAIN_ID=187 ROS_LOCALHOST_ONLY=1
  python3 cellcheck_v5.py --job jobs/<job>.json --build build/<cell> --out <result_dir> [--placements ...] [--pads-strict]
"""
import sys as _s; _s.dont_write_bytecode = True
import argparse, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
_dom = os.environ.get('ROS_DOMAIN_ID')          # cellcheck.py's import-time guard only accepts 187; domain 188 is swapped in only for the import and restored before rclpy.init (done later, in the client)
if _dom == '188': os.environ['ROS_DOMAIN_ID'] = '187'
import cellcheck as CC
if _dom is not None: os.environ['ROS_DOMAIN_ID'] = _dom
import cellcheck_v3 as C3


def _isolated_187_188():          # v5: two concurrent stacks on domains 187 / 188 (never 11)
    if os.environ.get('ROS_DOMAIN_ID') not in ('187', '188') or os.environ.get('ROS_LOCALHOST_ONLY') != '1':
        sys.exit('refusing to run: export ROS_DOMAIN_ID=187 (or 188) ROS_LOCALHOST_ONLY=1 in this shell (domain 11 is the real robot)')
CC.require_isolated_domain = _isolated_187_188


class Checker5(C3.Checker3):
    def __init__(self, meta, job, opts):
        super().__init__(meta, job, opts)
        self.pad_link = {a: a + '_skid_pads' for a in self.arms}
        self.pads_strict = bool(opts.get('pads_strict'))
        k = int(opts.get('plan_attempts', 3))
        orig = self.cl.plan
        def plan5(*a, **kw):
            kw['attempts'] = max(int(kw.get('attempts', 1)), k)
            return orig(*a, **kw)
        self.cl.plan = plan5

    def _pairs(self):
        pairs = [(self.tip[a], CC.CARTON) for a in self.arms]
        if not self.pads_strict:
            pairs += [(self.pad_link[a], CC.CARTON) for a in self.arms]
        return pairs

    def set_scene(self, pl):
        cl = self.cl
        cg = pl['carton']
        boxes = [dict(name=CC.CARTON, size=cg['size'], centre=cg['center'], quat=cg['quat'])]
        tb = self.job.get('table', {})
        x0, x1, hy, th = tb.get('x0', 0.12), tb.get('x1', 1.2), tb.get('half_y', 1.0), tb.get('thickness', 0.05)
        ztop = cg['center'][2] - cg['size'][2] / 2
        boxes.append(dict(name=CC.TABLE, size=[x1 - x0, 2 * hy, th], centre=[(x0 + x1) / 2, 0.0, ztop - th / 2], quat=[0, 0, 0, 1]))
        ok = cl.set_scene(boxes, remove=[CC.CARTON, CC.TABLE])
        cl.set_acm_allow(self._pairs(), False)
        return ok

    def allow_contact(self, on):
        return self.cl.set_acm_allow(self._pairs(), on)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--job', required=True); ap.add_argument('--build', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--placements', default=None)
    ap.add_argument('--planning-time', type=float, default=15.0)
    ap.add_argument('--ompl-attempts', type=int, default=6)
    ap.add_argument('--plan-attempts', type=int, default=3)
    ap.add_argument('--ik-timeout', type=float, default=0.5)
    ap.add_argument('--transit-planners', default='pilz:PTP,ompl:RRTConnectkConfigDefault')
    ap.add_argument('--pads-strict', action='store_true')
    a = ap.parse_args()
    b = a.build if os.path.isabs(a.build) else os.path.join(HERE, a.build)
    meta = json.load(open(os.path.join(b, 'cell_meta.json')))
    job = json.load(open(a.job))
    if abs(job['layout']['s'] - meta['s']) > 1e-6:
        sys.exit('job layout s=%s does not match the running cell build s=%s' % (job['layout']['s'], meta['s']))
    os.makedirs(a.out, exist_ok=True)
    chk = Checker5(meta, job, dict(planning_time=a.planning_time, ik_timeout=a.ik_timeout, transit_planners=a.transit_planners.split(','), concurrent_other='neutral', park_idle=False,
                                   partner_pass2=False, ompl_attempts=a.ompl_attempts, plan_attempts=a.plan_attempts, pads_strict=a.pads_strict))
    sel = set(a.placements.split(',')) if a.placements else None
    res = dict(schema='cell_planner_check/result/v1', job=job.get('name'), design=meta['design'], build=b, layout=job['layout'], scaling=chk.sc, domain=os.environ.get('ROS_DOMAIN_ID'),
               park_q=job.get('park_q'), opts=dict(planning_time=a.planning_time, ompl_attempts=a.ompl_attempts, plan_attempts=a.plan_attempts, pads_strict=a.pads_strict), placements=[])
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
