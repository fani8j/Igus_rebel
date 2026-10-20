"""Merge the parking study files into tests/results/amr_parking.json / .md and check one COMMON standoff for the front mode.
inputs (tests/results): amr_parking_front.json, amr_parking_topdown_diag.json, amr_parking_yaw180.json (optional), moveit_confirm_front.json (optional)"""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import parking_study as ps, robot_kin as rk
RES = HERE.parent / 'tests' / 'results'


def load(n):
    f = RES / n
    return json.load(open(f)) if f.exists() else None


def main():
    front = {r['target']: r for r in load('amr_parking_front.json')}
    diag = load('amr_parking_topdown_diag.json') or {}
    yaw = {r['target']: r for r in (load('amr_parking_yaw180.json') or [])}
    mv = load('moveit_confirm_front.json') or {}
    R, cab = rk.Robot(), rk.Cabinet()
    feas = [t for t, r in front.items() if r.get('feasible')]
    # common standoff: the d (10 mm steps) at which the most targets pass nominal + all 8 corners (each target at its own lateral offset / lift / sliders)
    ds = np.round(np.arange(0.30, 0.56, 0.01), 3)
    table = {}
    for d in ds:
        ok = []
        for t in feas:
            r = front[t]
            shelf, box, which = ps.TARGETS[t]
            tg = cab.pick(shelf, box, which)
            pull = ps.PULL_BY_MODE_SHELF.get(('front', shelf), rk.PULL_OUT)
            lift = r['lift']
            if not ps.feasible(R, cab, tg, shelf, 'front', pull, float(d), r['lateral_offset_m'], lift, r['slider_left'], r['slider_right'], False)['ok']:
                continue
            ok.append(t)
        table[float(d)] = ok
        print(d, ok, flush=True)
    # corner check at the best common d candidates (centre of the longest run where all feasible targets pass)
    full = [d for d, v in table.items() if len(v) == len(feas)]
    common = None
    if full:
        common = float(full[len(full) // 2])
        corners = {}
        for t in feas:
            r = front[t]
            shelf, box, which = ps.TARGETS[t]
            tg = cab.pick(shelf, box, which)
            pull = ps.PULL_BY_MODE_SHELF.get(('front', shelf), rk.PULL_OUT)
            okc, bad = ps.corners_pass(R, cab, tg, shelf, 'front', pull, common, r['lateral_offset_m'], r['lift'], r['slider_left'], r['slider_right'], False)
            corners[t] = okc
        print('common', common, corners)
    out = dict(front=front, topdown_pose_level_diagnostic=diag, yaw180_extra=yaw, moveit_confirmation=mv, common_standoff_scan=table,
               common_standoff_m=common, common_standoff_corner_box_ok=corners if full else None)
    (RES / 'amr_parking.json').write_text(json.dumps(out, indent=1, default=float))
    L = ['# AMR parking study (static AMR placeholder; only a recommendation where to park)', '',
         'Frame: d = AMR front bumper to cabinet front face (m); lateral = AMR centre line relative to the target carton centre (+ = robot y); lift / sliders = joint values (fixed during one pick).',
         'Window = contiguous range of d (10 mm steps) with the full sequence feasible at the best lateral/lift/sliders; best d = centre of that window; `box` = the 8 corners of +-20 mm x/y and +-1.5 deg yaw all pass.', '',
         '## Front mode (recommended default)', '',
         '| target | shelf / carton | feasible | best d [mm] | d window [mm] | lateral [mm] (window) | lift [m] (window) | sliders L/R [m] | joint margin [deg] | min sigma_min L/R | min clearance [mm] | +-20 mm / +-1.5 deg box | MoveIt mock plan |', '|' + '---|' * 13]
    for t in ps.TARGETS:
        r = front[t]
        if r.get('feasible'):
            mc = r['min_clearance_m']
            L.append(f"| {t} | {r['shelf']} / {r['box']} ({r['which']}) | yes | {r['best_d'] * 1000:.0f} | {r['feasible_d_window_m'][0] * 1000:.0f}..{r['feasible_d_window_m'][1] * 1000:.0f} | {r['lateral_offset_m'] * 1000:.0f} ({r['feasible_lateral_window_m'][0] * 1000:.0f}..{r['feasible_lateral_window_m'][1] * 1000:.0f}) | "
                     f"{r['lift']:+.3f} ({r['feasible_lift_window_m'][0]:+.2f}..{r['feasible_lift_window_m'][1]:+.2f}) | {r['slider_left']:+.4f} / {r['slider_right']:+.4f} | {np.degrees(r['joint_margin_rad']):.1f} | {r['min_manipulability']['left']:.3f} / {r['min_manipulability']['right']:.3f} | "
                     f"{min(mc.values()) * 1000:.0f} | {'pass' if r['corners_box_ok'] else 'FAIL'} | {('all OK' if mv[t]['all_ok'] else 'FAILED') if t in mv else 'not run'} |")
        else:
            L.append(f"| {t} | {r['shelf']} / {r['box']} ({r['which']}) | NO | - | - | - | - | - | - | - | - | - | - |")
    L += ['', 'Reasons for NO: ' + '; '.join(f"{t}: {front[t]['reason']} {front[t].get('full_eval_failures', '')}" for t in ps.TARGETS if not front[t].get('feasible')), '']
    if common:
        L += [f'**Common front standoff: d = {common * 1000:.0f} mm** passes for {len(feas)} of {len(feas)} reachable targets (each at its own lateral/lift); corner box: {corners}. Targets needing something else: ' +
              ', '.join(t for t in ps.TARGETS if t not in feas) + ' (not reachable, see reasons).', '']
    L += ['## Top-down mode', '', 'No target has a parking pose where the right-arm top-down grasp is collision free (>= 15 mm) with the real arm; pose-level diagnostic (necessary condition, independent of the transit model), 1500 sampled IK-reachable (d, lateral, lift, slider) per target:', '',
          '| target | IK-reachable map cells | collision-free pre/grasp poses | clearance violations (count of samples) |', '|---|---|---|---|']
    for t in ps.TARGETS:
        d_ = diag.get(t, {})
        L.append(f"| {t} | {d_.get('map_cells_with_ik')} | {d_.get('pre_pose_clear')} | {d_.get('reasons')} |")
    if yaw:
        L += ['', '## E2 with the suction tool turned 180 deg about the pad axis (extra variant)', '', json.dumps({t: {k: r.get(k) for k in ('feasible', 'best_d', 'lateral_offset_m', 'lift', 'feasible_d_window_m', 'corners_box_ok', 'reason')} for t, r in yaw.items()}, indent=1, default=float)]
    (RES / 'amr_parking.md').write_text('\n'.join(L) + '\n')
    print('written')

main()
