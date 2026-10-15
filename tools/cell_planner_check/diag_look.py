#!/usr/bin/env python3
"""Diagnose a failing transit of a job task (default: E5 BIG370_y150 arm2:tP look): goal validity, straight-line joint path validity, planners, partner states.
  python3 diag_look.py --job jobs/v2_E5_nominal.json --build build/v2_E5_s046_proxy --placement BIG370_y150 --task arm2:tP --step 0 --out diag.json"""
import argparse, json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import cellgeom as G
if os.environ.get('ROS_DOMAIN_ID') != '187' or os.environ.get('ROS_LOCALHOST_ONLY') != '1': sys.exit('export ROS_DOMAIN_ID=187 ROS_LOCALHOST_ONLY=1')
import moveit_client as MC, rclpy
ap = argparse.ArgumentParser()
for k in ('job', 'build', 'placement', 'task', 'out'): ap.add_argument('--' + k, required=True)
ap.add_argument('--step', type=int, default=0)
a = ap.parse_args()
meta = json.load(open(os.path.join(a.build, 'cell_meta.json'))); job = json.load(open(a.job))
pl = next(p for p in job['placements'] if p['id'] == a.placement); task = pl['tasks'][a.task]; arm = task['arm']; other = [x for x in G.ARMS if x != arm][0]
MC.init(); cl = MC.MoveItClient(meta, 'diag_look'); z = [0.0] * 6
cg = pl['carton']; tb = job['table']; ztop = cg['center'][2] - cg['size'][2] / 2
cl.set_scene([dict(name='detected_carton', size=cg['size'], centre=cg['center'], quat=cg['quat']), dict(name='table', size=[tb['x1'] - tb['x0'], 2 * tb['half_y'], tb['thickness']], centre=[(tb['x0'] + tb['x1']) / 2, 0, ztop - tb['thickness'] / 2], quat=[0, 0, 0, 1])], remove=['detected_carton', 'table'])
cand = task['steps'][a.step]['target']['candidates'][0]
out = dict(placement=a.placement, task=a.task, candidate=cand.get('name'))
# partner states to try: neutral, partner's look/first-step pose of the other task of the same phase
partners = {'neutral': z}
ph = next(p for p in pl['phases'] if a.task in p)
if len(ph) == 2:
    ot = [t for t in ph if t != a.task][0]
    oc = pl['tasks'][ot]['steps'][0]['target']['candidates'][0]
    qo, e = cl.ik(other, G.pose_to_T(oc['pose']), {other: oc['seeds'][0] if oc.get('seeds') else z, arm: z}, avoid_collisions=True)
    out['partner_first_step_ik'] = e
    if qo: partners['partner_first_step_pose'] = qo
for pn, qp in partners.items():
    res = {}
    q, e = cl.ik(arm, G.pose_to_T(cand['pose']), {arm: cand['seeds'][0], other: qp}, avoid_collisions=True)
    res['ik_with_collisions'] = e
    q2, e2 = cl.ik(arm, G.pose_to_T(cand['pose']), {arm: cand['seeds'][0], other: qp}, avoid_collisions=False)
    res['ik_no_collisions'] = e2
    qg = q or q2
    if qg is not None:
        v, c = cl.validity({arm: qg, other: qp}, 'both'); res['goal_valid'] = bool(v); res['goal_contacts'] = [list(x) for x in c][:6]
        # straight-line joint path from neutral
        bad = None; n = 200
        for k in range(n + 1):
            qq = [z[j] + (qg[j] - z[j]) * k / n for j in range(6)]
            v, c = cl.validity({arm: qq, other: qp}, 'both')
            if not v: bad = (k / n, [list(x) for x in c][:4]); break
        res['straight_line_first_collision'] = bad
        res['goal_deg'] = [round(math.degrees(v), 1) for v in qg]
        for pipe, pid, tl in (('pilz_industrial_motion_planner', 'PTP', 5), ('ompl', 'RRTConnectkConfigDefault', 20), ('ompl', 'BiTRRTkConfigDefault', 20), ('ompl', 'RRTkConfigDefault', 20)):
            p = cl.plan(arm, {arm: z, other: qp}, qg, 0.2, 0.1, planner=pid, pipeline=pipe, planning_time=tl)
            res['plan_%s' % pid] = (p['error'], round(p['planning_time'], 2), round(p.get('duration', 0), 1))
    out['partner_' + pn] = res
json.dump(out, open(a.out, 'w'), indent=1); print(json.dumps(out, indent=1)); cl.destroy_node(); rclpy.shutdown()
