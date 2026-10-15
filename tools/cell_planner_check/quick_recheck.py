#!/usr/bin/env python3
"""v5 QUICK RE-CHECK of the current recommended design after a small change (see README_quick.md).  Subcommands (driven by quick_recheck.sh):
  prep CHANGE.json TAG      work dir scan_results/planner_check_v5_20261007/quick/<TAG>/ : patched copy of cad/dual_arm_tooling_v3 (code + data), env.sh, design json
  cad TAG                   parametric CAD rebuild (build_all, verify, verify_skid, stiffness_mass) in the work dir; writes cad_result.json
  offline TAG NW            27-point primary region of BIG and SMALL, table + stand, HARDEST POINTS FIRST, stop at the first failure (QR_ALL=1: evaluate everything) -> offline_result.json
  moveitprep TAG            cell builds (table, stand) from the work-dir collision meshes + baseline jobs (same stroke / transit targets)
  report TAG                pass / fail table from cad_result.json, offline_result.json and the MoveIt runs
Supported change keys: blade_exposed_mm, skid_exposure_mm, pad_r_mm, pad_off_mm, camA_shift_mm [x,y,z], camB_shift_mm [x,y,z] (link6 mm), mass_fixed_kg {gripper, fingers, handpiece, camA, camB},
fixture {BIG|SMALL: {dx_mm, dy_mm, dyaw_deg}} (offline only; MoveIt then reuses the old stroke targets and is skipped)."""
import os, sys, json, glob, shutil, pickle, time, copy, subprocess
sys.dont_write_bytecode = True
REPO = '/home/farhan/igus_rebel_ros2'
S = REPO + '/scan_results/cell_v5_20261007'
J = REPO + '/scan_results/planner_check_v5_20261007'
CAD = REPO + '/cad/dual_arm_tooling_v3'
HERE = os.path.dirname(os.path.abspath(__file__))
PY_CAD = os.environ.get('CAD_PY', '/media/farhan/My Passport/igus_nvidia/venvs/cad/bin/python')
THR = dict(clear_mm=5.0, margin_deg=10.0, pen_mm=3.5, defl_mm=0.30, pad_blade_mm=2.0)


def W(tag): return os.path.join(J, 'quick', tag)


def rank1():
    return json.load(open(S + '/recommended.json'))['combos'][0]


def load_change(tag): return json.load(open(os.path.join(W(tag), 'change.json')))


def envs(ch):
    e = {}
    if 'blade_exposed_mm' in ch: e['V5_BLADE_EXP'] = str(ch['blade_exposed_mm'])
    if 'skid_exposure_mm' in ch: e['V5_EXPOSURE'] = str(ch['skid_exposure_mm'])
    if 'pad_r_mm' in ch: e['V5_PAD_R'] = str(ch['pad_r_mm'])
    if 'pad_off_mm' in ch: e['V5_PAD_OFF'] = str(ch['pad_off_mm'])
    m = ch.get('mass_fixed_kg', {})
    if 'gripper' in m: e['V5_MASS_GRIPPER'] = str(m['gripper'])
    if 'fingers' in m: e['V5_MASS_FINGERS'] = str(m['fingers'])
    return e


def prep(chfile, tag):
    w = W(tag); os.makedirs(w + '/cad/data', exist_ok=True)
    ch = json.load(open(chfile)); json.dump(ch, open(w + '/change.json', 'w'), indent=1)
    for f in glob.glob(CAD + '/*.py') + [CAD + '/run_all.sh']:
        shutil.copy(f, w + '/cad/')
    for f in ('design.json', 'cad_overrides.json', 'plans_table_BIG.json', 'plans_table_SMALL.json', 'plans_stand_BIG.json', 'plans_stand_SMALL.json'):
        if os.path.exists(CAD + '/data/' + f): shutil.copy(CAD + '/data/' + f, w + '/cad/data/' + f)
    d = json.load(open(w + '/cad/data/design.json')); sk = d['skid']
    ex = float(ch.get('skid_exposure_mm', sk['exposure_mm'])); r = float(ch.get('pad_r_mm', sk['pad_r_mm'])); off = float(ch.get('pad_off_mm', sk['pad_off_mm']))
    sk.update(exposure_mm=ex, pad_r_mm=r, pad_off_mm=off, pad_centres_K_mm=[[-off, sk['pad_y_mm'], -(ex + r)], [off, sk['pad_y_mm'], -(ex + r)]])
    for key, nm in (('camA_shift_mm', 'camA'), ('camB_shift_mm', 'camB')):
        if key in ch and d.get(nm):
            v = [float(x) for x in ch[key]]
            for k in range(3): d[nm]['T_mm'][k][3] += v[k]
            for k in ('bracket_from_mm', 'bracket_to_mm'): d[nm][k] = [a + b for a, b in zip(d[nm][k], v)]
    json.dump(d, open(w + '/cad/data/design.json', 'w'), indent=1)
    ov = json.load(open(w + '/cad/data/cad_overrides.json')) if os.path.exists(w + '/cad/data/cad_overrides.json') else {}
    for k in ('blade_exposed_mm', 'mass_fixed_kg'):
        if k in ch: ov[k] = ch[k]
    json.dump(ov, open(w + '/cad/data/cad_overrides.json', 'w'), indent=1)
    c = rank1(); name = c['label'] + '_q' + tag
    open(w + '/label_rank1.txt', 'w').write(name)
    for sub in ('build', 'jobs', 'runs'): os.makedirs(w + '/' + sub, exist_ok=True)
    print('prepared', w, 'label', name)


def cad(tag):
    w = W(tag) + '/cad'; env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    t0 = time.time(); log = open(W(tag) + '/cad.log', 'w'); out = {}
    for sc in ('build_all.py', 'check_exports.py', 'verify.py', 'verify_skid.py', 'stiffness_mass.py'):
        if sc == 'verify_skid.py' and os.environ.get('QR_FAST'): continue
        rc = subprocess.call(['nice', '-n', '19', PY_CAD, sc], cwd=w, env=env, stdout=log, stderr=subprocess.STDOUT)
        out[sc] = rc
        if rc != 0 and sc in ('build_all.py',): break
    dd = w + '/data/'
    ld = lambda f: json.load(open(dd + f)) if os.path.exists(dd + f) else {}
    v, k, s = ld('verification_part1.json'), ld('verification_skid.json'), ld('stiffness_mass.json')
    defl = s.get('tip_deflection_10N_PA12-CF', {})
    res = dict(rc=out, seconds=round(time.time() - t0), interference_pairs=v.get('interference_pairs_gt_0p5mm3'), keepout_ok=v.get('keepout_ok'), finger_clearance_min_mm=v.get('finger_clearance_min_mm'),
               pad_to_blade_min_mm=k.get('pad_to_blade_min_mm'), pad_to_blade_bent_mm=k.get('pad_to_blade_min_bent_mm'), axial_stop_present=k.get('axial_stop_present'),
               flap_gap_ok=k.get('flap_gap_ok_up_to_6mm_tilt_25deg'), pen_min_mm=(k.get('poses') or {}).get('pen_min_mm'), pad_first=((k.get('poses') or {}).get('pad_first_count'), (k.get('poses') or {}).get('n')),
               camB_tip_clear=k.get('camB_tip_clear'), camB_tape_ahead_clear=k.get('camB_tape_ahead_clear'),
               defl_max_mm=max([x['total_mm'] for x in defl.values()], default=None), mass_kg=(s.get('mass_kg') or {}).get('total_PA12CF_factor055'),
               moment_wrist_bottle_Nm=((s.get('moment') or {}).get('with_bottle_0p5kg') or {}).get('wrist_centre_Nm'))
    json.dump(res, open(W(tag) + '/cad_result.json', 'w'), indent=1); print(json.dumps(res))


def offline(tag, nw):
    ch = load_change(tag); os.environ.update(envs(ch))
    sys.path.insert(0, S); os.chdir(S)
    import numpy as np, multiprocessing as mp
    import v5lib, final3 as F3, deep4 as D4, cell3 as C3
    O = pickle.load(open(S + '/deep_out.pkl', 'rb')); Dd = pickle.load(open(S + '/hdesigns.pkl', 'rb')); sl = O['sel'][0]
    spec = copy.deepcopy(Dd[sl['name']]); import numpy as np
    for key, nm in (('camA_shift_mm', 'camA'), ('camB_shift_mm', 'camB')):
        if key in ch and spec.get(nm):
            v = np.asarray(ch[key], float) * 1e-3; spec[nm]['o'] = np.asarray(spec[nm]['o']) + v; spec[nm]['brp'] = np.asarray(spec[nm]['brp']) + v
    Dd = {sl['name']: spec}
    base = {}
    for r in pickle.load(open(S + '/override_results.pkl', 'rb')):
        base[(r['tag'])] = {tuple(round(float(x), 4) for x in p['off']): p['margin'] for p in r['per'] if p['ok']}
    jobs = []
    for nm, f in (('big', sl['big']), ('small', sl['small'])):
        fx = ch.get('fixture', {}).get(nm.upper(), {})
        for mode, suf in (('table', '_primary'), ('stand', '_stand')):
            hard = base.get(nm + suf, {})
            for (a, b, c) in C3.region_offsets(D4.TOL1, 3):
                j = dict(spec=spec, box=f['box'], h=f['h'], s=sl['s'], cx=f['cx'] + a + fx.get('dx_mm', 0) * 1e-3, cy=f['cy'] + b + fx.get('dy_mm', 0) * 1e-3, yaw=f['yaw'] + c + fx.get('dyaw_deg', 0.0),
                         tol=(0.0, 0.0, 0.0), n=1, band=45, keep_plan=False, park={int(k): v for k, v in sl['park'].items()}, tag='%s%s|%.4f|%.4f|%.2f' % (nm, suf, a, b, c), mode=mode, scene_cam=True, design=sl['name'])
                j['_hard'] = hard.get(tuple(round(float(x), 4) for x in (a, b, c)), 0.0)
                jobs.append(j)
    jobs.sort(key=lambda j: j['_hard'])                  # hardest (smallest baseline margin) first
    t0 = time.time(); res = []; fails = []
    allm = os.environ.get('QR_ALL') == '1'
    with mp.Pool(nw) as pool:
        for r in pool.imap(F3.region_job, jobs, chunksize=1):
            res.append(r)
            if r['frac'] < 1.0:
                fails.append(r['tag'])
                if not allm: pool.terminate(); break
    summ = {}
    for r in res:
        k = r['tag'].split('|')[0]
        d = summ.setdefault(k, dict(n=0, ok=0, min_margin=1e9, min_clear=1e9))
        d['n'] += 1; d['ok'] += int(r['frac'] == 1.0)
        if r['frac'] == 1.0: d['min_margin'] = min(d['min_margin'], r['min_margin']); d['min_clear'] = min(d['min_clear'], r['min_clear'])
    out = dict(evaluated=len(res), of=len(jobs), stopped_at_first_failure=bool(fails) and not allm, failures=fails[:5], summary={k: dict(v, min_clear_mm=v['min_clear'] * 1000) for k, v in summ.items()}, seconds=round(time.time() - t0))
    json.dump(out, open(W(tag) + '/offline_result.json', 'w'), indent=1); print(json.dumps(out))


def moveitprep(tag):
    ch = load_change(tag); sys.path.insert(0, HERE)
    import numpy as np, make_design_v5 as MD
    w = W(tag); c = rank1(); name = open(w + '/label_rank1.txt').read().strip()
    des = MD.design_from_entry(c, w + '/cad/collision'); des['name'] = 'v5_' + name
    json.dump(des, open(w + '/design.json', 'w'), indent=1)
    for st in ('table', 'stand'):
        subprocess.check_call(['python3', 'gen_cell_v5.py', '--design', w + '/design.json', '--s', str(c['layout']['s']), '--out', '%s/build/%s_%s' % (w, name, st), '--structure', st, '--scene-camera'], cwd=HERE)
    for f in glob.glob(J + '/jobs/%s_*.json' % c['label']):
        b = os.path.basename(f)[len(c['label']) + 1:]
        if b.startswith('sh_') or b.startswith('stand_sh_') or 'pk1' in b: continue
        shutil.copy(f, '%s/jobs/%s_%s' % (w, name, b))
    if ch.get('fixture'): print('NOTE: fixture changed: MoveIt reuses the old stroke targets (not valid) -> skip the MoveIt step')


def report(tag):
    w = W(tag); ch = load_change(tag)
    cr = json.load(open(w + '/cad_result.json')) if os.path.exists(w + '/cad_result.json') else {}
    orr = json.load(open(w + '/offline_result.json')) if os.path.exists(w + '/offline_result.json') else {}
    rows = []
    def add(n, v, ok, unit=''): rows.append((n, v, 'PASS' if ok else ('FAIL' if ok is False else '-'), unit))
    add('offline 27 pts x BIG/SMALL x table/stand', '%s/%s evaluated%s' % (orr.get('evaluated'), orr.get('of'), ', STOPPED at first failure' if orr.get('stopped_at_first_failure') else ''), (not orr.get('failures')) if orr else None)
    for k, v in sorted(orr.get('summary', {}).items()):
        add('  %s ok / min margin / min clearance beyond padding' % k, '%d/%d  %.1f deg  %.1f mm' % (v['ok'], v['n'], v['min_margin'], v['min_clear_mm']), v['ok'] == v['n'] and v['min_clear_mm'] >= THR['clear_mm'] and v['min_margin'] >= THR['margin_deg'])
    if cr:
        add('CAD interference (pairs > 0.5 mm3)', len(cr.get('interference_pairs') or []), not cr.get('interference_pairs'))
        add('keep-out / finger clearance mm', '%s / %s' % (cr.get('keepout_ok'), cr.get('finger_clearance_min_mm') and round(cr['finger_clearance_min_mm'], 1)), bool(cr.get('keepout_ok')) and (cr.get('finger_clearance_min_mm') or 0) >= 5)
        add('pad -> blade clearance (rest / bent) mm', '%s / %s' % (cr.get('pad_to_blade_min_mm'), cr.get('pad_to_blade_bent_mm')), (cr.get('pad_to_blade_bent_mm') or 0) >= THR['pad_blade_mm'])
        add('penetration before skid contact, min (study strokes) mm', cr.get('pen_min_mm') and round(cr['pen_min_mm'], 2), (cr.get('pen_min_mm') or 0) >= THR['pen_mm'])
        add('pad first contact in all strokes', cr.get('pad_first'), bool(cr.get('pad_first')) and cr['pad_first'][0] == cr['pad_first'][1])
        add('axial stop / flap-gap 0-6 mm support', '%s / %s' % (cr.get('axial_stop_present'), cr.get('flap_gap_ok')), bool(cr.get('axial_stop_present')) and bool(cr.get('flap_gap_ok')))
        add('camera B sees tip / tape ahead', '%s / %s' % (cr.get('camB_tip_clear'), cr.get('camB_tape_ahead_clear')), None)
        add('tip deflection 10 N PA12-CF, max mm', cr.get('defl_max_mm') and round(cr['defl_max_mm'], 3), (cr.get('defl_max_mm') or 9) <= THR['defl_mm'])
        add('mass kg / wrist moment with 0.5 kg bottle Nm', '%s / %s' % (cr.get('mass_kg') and round(cr['mass_kg'], 3), cr.get('moment_wrist_bottle_Nm') and round(cr['moment_wrist_bottle_Nm'], 2)), None)
    mv = []
    for f in sorted(glob.glob(w + '/runs/*/result_*.json')):
        d = json.load(open(f)); P = d['placements']; ok = sum(1 for p in P if p.get('feasible'))
        mm = [m for p in P for t in p['tasks'].values() for m in (t.get('min_margin_deg') or []) if m is not None]
        add('MoveIt ' + os.path.basename(os.path.dirname(f)).split('_q')[-1], '%d/%d  min planned margin %s' % (ok, len(P), mm and round(min(mm), 1)), ok == len(P))
    print('\nQUICK RE-CHECK %s   change: %s' % (tag, json.dumps(ch)))
    for n, v, st, u in rows: print('%-62s %-44s %s' % (n, v, st))
    allp = [r[2] for r in rows if r[2] in ('PASS', 'FAIL')]
    print('RESULT:', 'ALL PASS' if allp and all(x == 'PASS' for x in allp) else 'FAIL / incomplete')
    json.dump(dict(rows=rows), open(w + '/report.json', 'w'), indent=1)


if __name__ == '__main__':
    c = sys.argv[1]
    {'prep': lambda: prep(sys.argv[2], sys.argv[3]), 'cad': lambda: cad(sys.argv[2]), 'offline': lambda: offline(sys.argv[2], int(sys.argv[3])), 'moveitprep': lambda: moveitprep(sys.argv[2]), 'report': lambda: report(sys.argv[2])}[c]()
