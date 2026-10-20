#!/usr/bin/env python3
"""Markdown tables from results/cabinet_grasp_{topdown,front}.json -> results/cabinet_grasp_summary.md (stdout too)."""
import json
from pathlib import Path

R = Path(__file__).resolve().parent / 'results'
data = {m: json.load(open(R / f'cabinet_grasp_{m}.json')) for m in ('topdown', 'front') if (R / f'cabinet_grasp_{m}.json').exists()}
out = []


def mm(v, f='%.1f'):
    return '-' if v is None else f % v


def row(m, r):
    S = r.get('steps', {})
    if 'error' in r:
        return None
    g = lambda k, f, d=None: S.get(k, {}).get(f, d)
    clr = r.get('min_clearance_mm', {})
    fails = [k for k, v in S.items() if not v.get('ok', True)] + (['contacts ' + str(r['unexpected_contacts'])] if r.get('unexpected_contacts') else [])
    return [r['target'] + ('' if r['variant'] == 'base' else '/' + r['variant']), f"S{r['shelf']}", r['box'].split('_')[0], f"{r['thickness_mm']:.0f}", f"{r['clear_height_mm']:.0f}",
            'OK' if r['success'] else 'FAIL', mm(g('1_press', 'carton_pushed_back_mm')), mm(g('3_pull', 'follow_error_mm'), '%.2f'), mm(g('3_pull', 'carton_net_disp_from_rest_mm')),
            mm(r.get('neighbour_max_disp_mm'), '%.2f'), mm(r.get('neighbour_max_tilt_deg'), '%.2f'), mm(g('4b_clamp', 'carton_shift_during_close_mm'), '%.2f'),
            mm(g('6_extract', 'slip_mm'), '%.2f'), mm(g('6_extract', 'carton_lift_max_mm'), '%.2f'), mm(g('7_lift50', 'carton_rise_mm')), mm(r.get('carton_max_rotation_deg'), '%.2f'),
            mm(clr.get('SX.shelf_above')), mm(clr.get('GR.shelf_above')), mm(clr.get('GR.shelf_below')), mm(min([v for k, v in clr.items() if k.endswith('neighbours') and v is not None] or [None])),
            ', '.join(fails) or '-']


hdr = ['target', 'lvl', 'box', 't mm', 'clear mm', 'result', 'press push-back mm', 'pull follow err mm', 'pull net mm', 'nb disp mm', 'nb tilt deg', 'shift on close mm', 'extract slip mm',
       'lift on floor mm', 'lift rise mm', 'carton rot deg', 'SX shelf-above mm', 'GR shelf-above mm', 'GR shelf-below mm', 'min nb clr mm', 'failed steps']
for m, d in data.items():
    out.append(f'### {m}\n')
    out.append('| ' + ' | '.join(hdr) + ' |')
    out.append('|' + '---|' * len(hdr))
    for r in d['runs']:
        x = row(m, r)
        out.append('| ' + ' | '.join(x) + ' |' if x else f"| {r.get('target')} | ERROR {r.get('error')} |")
    out.append('')
if len(data) == 2:
    out.append('### top-down vs front (base runs)\n')
    out.append('| target | lvl | top-down | front | front: carton shift on approach mm | front: shift on close mm | top-down: shift on close mm | front GR to neighbours mm (approach) | front GR shelf above / below mm (approach) |')
    out.append('|---|---|---|---|---|---|---|---|---|')
    f = {r['target']: r for r in data['front']['runs'] if r.get('variant') == 'base'}
    for r in data['topdown']['runs']:
        if r.get('variant') != 'base':
            continue
        q = f.get(r['target'])
        if not q:
            continue
        ap = q.get('min_clearance_by_phase_mm', {}).get('4_gripper_approach_clamp', {})
        out.append(f"| {r['target']} | S{r['shelf']} | {'OK' if r['success'] else 'FAIL'} | {'OK' if q['success'] else 'FAIL'} | {mm(q['steps'].get('4a_approach', {}).get('carton_shift_mm'), '%.3f')} | "
                   f"{mm(q['steps'].get('4b_clamp', {}).get('carton_shift_during_close_mm'), '%.3f')} | {mm(r['steps'].get('4b_clamp', {}).get('carton_shift_during_close_mm'), '%.3f')} | "
                   f"{mm(ap.get('GR.neighbours'))} | {mm(ap.get('GR.shelf_above'))} / {mm(ap.get('GR.shelf_below'))} |")
txt = '\n'.join(out)
(R / 'cabinet_grasp_summary.md').write_text(txt)
print(txt)
