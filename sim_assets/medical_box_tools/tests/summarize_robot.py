#!/usr/bin/env python3
"""Markdown tables from results/cabinet_grasp_robot_<mode>.json -> results/cabinet_grasp_robot_summary.md"""
import json
from pathlib import Path
R = Path(__file__).resolve().parent / 'results'
out = ['# Cabinet grasp with the tools on the D200 dual-arm robot (arm joint drives, static AMR)', '']
f = lambda v, s='%.1f': '-' if v is None else s % v
for mode in ('front', 'topdown'):
    fn = R / f'cabinet_grasp_robot_{mode}.json'
    if not fn.exists():
        continue
    d = json.load(open(fn))
    out += [f'## {mode}', '', '| target | result | d mm | lat mm | lift m | press push-back mm | pull net mm | pad tension max N | nb disp mm | nb tilt deg | clamp shift mm | extract slip mm | lift rise mm | carton rot deg | arm joint err max mrad (pull / extract / all) | joint effort max / limit | min clr SX-nb / GR-nb / ARM-any mm | robot-cabinet contacts | failed |', '|' + '---|' * 19]
    for r in d['runs']:
        if 'error' in r or 'skipped' in r:
            out.append(f"| {r['target']} | {'ERROR ' + r['error'] if 'error' in r else 'SKIPPED: ' + r['skipped']} |" + ' |' * 17)
            continue
        S, c = r['steps'], r['min_clearance_mm']
        fails = [k for k, v in S.items() if not v.get('ok', True)] + (['contacts ' + str(r['unexpected_contacts'])] if r['unexpected_contacts'] else [])
        arm = [v for k, v in c.items() if k.startswith('ARM.') and v is not None]
        p = r['parking']
        out.append(f"| {r['target']}{'/' + r['variant'] if r['variant'] != 'base' else ''} | {'OK' if r['success'] else 'FAIL'} | {p['d_mm']:.0f} | {p['lateral_mm']:.0f} | {p['lift_m']} | {f(S['1_press']['carton_pushed_back_mm'])} | {f(S['3_pull']['carton_net_disp_from_rest_mm'])} | "
                   f"{f(S['3_pull'].get('monitor_max_pad_tension_N'), '%.2f')} | {f(r['neighbour_max_disp_mm'], '%.2f')} | {f(r['neighbour_max_tilt_deg'], '%.2f')} | {f(S['4b_clamp']['carton_shift_during_close_mm'], '%.2f')} | {f(S['6_extract']['slip_mm'], '%.2f')} | "
                   f"{f(S['7_lift50']['carton_rise_mm'])} | {f(r['carton_max_rotation_deg'], '%.2f')} | {S['3_pull']['arm_joint_err_max_rad_during_pull'] * 1000:.1f} / {S['6_extract']['arm_joint_err_max_rad'] * 1000:.1f} / {r['arm_joint_tracking_error_max_rad'] * 1000:.1f} | "
                   f"{r['arm_effort_ratio_max'] * 100:.0f} % of maxForce | {f(c.get('SX.neighbours'))} / {f(c.get('GR.neighbours'))} / {f(min(arm) if arm else None)} | {r['robot_cabinet_contacts'] or 'none'} | {', '.join(fails) or '-'} |")
    out.append('')
(R / 'cabinet_grasp_robot_summary.md').write_text('\n'.join(out) + '\n')
print('\n'.join(out))
