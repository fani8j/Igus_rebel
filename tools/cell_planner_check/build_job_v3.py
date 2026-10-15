#!/usr/bin/env python3
"""v3 job builder: offline plans (scan_results/cell_v3_20261004, serial dual-arm / single-arm, camera-B look poses, park pose) -> job JSON (schema cell_planner_check/job/v1) for cellcheck_v3.py.

  python3 build_job_v3.py --plans <plans_<label>.json> --design v3_<label> --kind BIG|SMALL --out jobs/<name>.json [--max N]

plans file: {label, layout{h,s}, T_link6_tcp, park_q{'0':[6],'1':[6]}, placements:[{id, box, h, cx, cy, yaw, study{feasible, margin, used}, plan{'arm:task': {q7 (7 x 6), look_q|None, gamma, ...}}}]}
Per task: [look transit + 1 s dwell (only if the offline task needs a look pose)] -> approach (30 mm along the knife axis; 20 / 10 mm fallbacks) -> enter -> stroke -> retreat, with the study joint
paths as IK seeds.  Phases are SERIAL (one task each); a '<arm>:park' transit (explicit joint goal = park pose) is inserted whenever the working arm changes; both arms start parked.
"""
import argparse, json, math, os, sys
sys.dont_write_bytecode = True
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cellgeom as G
import build_job as BJ
V3 = '/home/farhan/igus_rebel_ros2/scan_results/cell_v3_20261004'
sys.dont_write_bytecode = True
sys.path.insert(0, V3)
from reach_lib import fk_flange, ik_flange, LIM_LO, LIM_HI, D2R

ORDER = ['tF', 'tP', 'tM', 'sP', 'sM']


def build(plans, design_name, kind, max_n=None):
    h, s = plans['layout']['h'], plans['layout']['s']
    T_tcp = np.array(plans['T_link6_tcp'])
    T_inv = np.linalg.inv(T_tcp)

    def tcp_world(q, arm_i):
        T = fk_flange(np.atleast_2d(np.asarray(q, float))) @ T_tcp
        T[:, 1, 3] += G.arm_y(arm_i, s)
        return T

    def branches(T, arm_i):
        T6 = (T @ T_inv).copy()
        T6[1, 3] -= G.arm_y(arm_i, s)
        q, valid = ik_flange(T6[None])
        out = []
        for b in range(q.shape[1]):
            if valid[0, b]:
                qd = q[0, b] / D2R
                m = float(np.min(np.minimum(qd - LIM_LO, LIM_HI - qd)))
                if m >= 0:
                    out.append((q[0, b], m))
        out.sort(key=lambda x: -x[1])
        return out

    park = {G.ARMS[int(k)]: v for k, v in plans['park_q'].items()}
    job = dict(schema='cell_planner_check/job/v1', name='%s_%s' % (design_name, kind), design='designs/%s.json' % design_name, source='cell_v3_20261004 (%s, serial, park pose, camera-B look poses)' % kind,
               layout=dict(h=h, d=0.0, s=s), snapshot_s=2.0, park_q=park,
               scaling=dict(names=['exec', 'fast'], transit=[[0.20, 0.10], [1.0, 0.1]], contact=[0.1376, 0.0954]),
               table=dict(x0=0.12, x1=1.2, half_y=1.0, thickness=0.05), placements=[])
    for rec in plans['placements']:
        if max_n and len(job['placements']) >= max_n:
            break
        pl = dict(id=rec['id'], box=rec['box'], yaw_deg=rec['yaw'], dx=0.0, dy=0.0, carton=G.carton_geometry(rec['box'], rec['h'], rec['cx'], rec['yaw'], 0.0, rec['cy']),
                  study=dict(feasible=bool(rec['study']['feasible']), margin=rec['study'].get('margin'), used=rec['study'].get('used'), tasks={}), tasks={}, notes=[], phases=[])
        if not rec.get('plan'):
            pl['notes'].append('offline verdict: infeasible, no plan to replay (not run)')
            job['placements'].append(pl)
            continue
        seq = {}
        for k, v in rec['plan'].items():
            arm_i, t = k.split(':')
            seq.setdefault(int(arm_i), []).append((t, v))
        arms = sorted(seq, key=lambda a: (-len(seq[a]), a))
        phases = []
        for ai, arm_i in enumerate(arms):
            arm = G.ARMS[arm_i]
            for t, v in sorted(seq[arm_i], key=lambda tv: ORDER.index(tv[0])):
                q7 = np.array(v['q7'])
                T7 = tcp_world(q7, arm_i)
                poses5 = [G.T_to_pose(T) for T in T7[1:6]]
                info = dict(poses5=poses5, q=q7[1:6].tolist(), margin=v.get('margin'), source='v3_offline_path(gamma=%g)' % v.get('gamma', 0.0), branches_start=[])
                look_step = None
                if v.get('look_q') is not None:
                    Tl = tcp_world(v['look_q'], arm_i)[0]
                    look_step = dict(type='transit', name='look', dwell_s=1.0, target=dict(candidates=[dict(name='v3_look_q', q_goal=v['look_q']), dict(name='v3_look', pose=G.T_to_pose(Tl), seeds=[v['look_q']])]))
                tid = '%s:%s' % (arm, t)
                task = BJ.cut_task(tid, arm, info, 0.03, 0.03, lambda T, i: branches(T, i), [0.03, 0.02, 0.01], look_step)
                # the offline approach configuration is tried first as an explicit joint goal (avoids KDL landing on another wrist branch); the IK candidates remain as fallbacks
                for st in task['steps']:
                    if st['name'] == 'approach':
                        c0 = st['target']['candidates'][0]
                        st['target']['candidates'].insert(0, dict(name='v3_approach_q', q_goal=q7[0].tolist(), standoff_m=0.03))
                # Cartesian steps: alternative waypoint splits (the Humble KDL path sometimes lands on the other wrist branch for ONE long step; the split variants are tried in order)
                Ts_, Te_ = T7[1], T7[5]
                push_ = Ts_[:3, 2]
                seg = dict(enter=(Ts_[:3, 3] - 0.03 * push_, Ts_[:3, 3]), stroke=(Ts_[:3, 3], Te_[:3, 3]), retreat=(Te_[:3, 3], Te_[:3, 3] - 0.03 * push_))
                for st in task['steps']:
                    if st['type'] == 'cartesian' and st['name'] in seg:
                        a_, b_ = seg[st['name']]
                        R_ = G.pose_to_T(st['waypoints'][-1])[:3, :3]
                        def wp(pt):
                            T_ = np.eye(4); T_[:3, :3] = R_; T_[:3, 3] = pt
                            return G.T_to_pose(T_)
                        end_wp = st['waypoints'][-1]
                        alts = [[wp(a_ + (b_ - a_) * 0.5), end_wp], [wp(a_ + (b_ - a_) * f) for f in (0.25, 0.5, 0.75)] + [end_wp]]
                        st['alternatives'] = (st.get('alternatives') or []) + alts
                task['gamma_deg'] = v.get('gamma')
                pl['tasks'][tid] = task
                pl['study']['tasks'][tid] = dict(margin=v.get('margin'), source=info['source'], q=info['q'])
                phases.append([tid])
            if ai + 1 < len(arms):          # park the finished arm before the partner starts
                pid = '%s:park' % arm
                pl['tasks'][pid] = dict(id=pid, arm=arm, kind='park', steps=[dict(type='transit', name='park', target=dict(candidates=[dict(name='park', q_goal=park[arm])]))])
                phases.append([pid])
        pl['phases'] = phases
        job['placements'].append(pl)
    return job


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plans', required=True)
    ap.add_argument('--design', required=True)
    ap.add_argument('--kind', default='BIG')
    ap.add_argument('--out', required=True)
    ap.add_argument('--max', type=int, default=None)
    a = ap.parse_args()
    plans = json.load(open(a.plans))
    job = build(plans, a.design, a.kind, a.max)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(job, open(a.out, 'w'), indent=1)
    print('wrote %s: %d placements (%d with plans)' % (a.out, len(job['placements']), sum(1 for p in job['placements'] if p['tasks'])))


if __name__ == '__main__':
    main()
