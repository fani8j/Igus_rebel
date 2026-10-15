#!/usr/bin/env python3
"""v6 CAD verification of the NEW study strokes with the UNCHANGED CAD v3 geometry (work copy in scan_results/planner_check_v6_20261007/cad; cad/dual_arm_tooling_v3 is not modified).
  python3 cad_verify_v6.py prep     copy CAD v3 code + data to the work dir, install the v6 plans as data/plans_{table,stand}_{BIG,SMALL}.json, patch verify_skid.py with the v6 checks
  python3 cad_verify_v6.py run      run verify_skid.py (nice 19, 1-2 threads) -> work/cad/data/verification_skid.json
  python3 cad_verify_v6.py report   v6 pass / fail numbers
v6 checks added to verify_skid: second-pad lift (|pen_pad0 - pen_pad1|) <= 1.0 mm at every stroke sample, penetration >= 3.5 mm, pad first contact in all samples, flap-gap support 0-6 mm at the study tilt."""
import os, sys, json, glob, shutil, subprocess
REPO = '/home/farhan/igus_rebel_ros2'
CAD = REPO + '/cad/dual_arm_tooling_v3'
S = REPO + '/scan_results/cell_v6_20261007'
W = REPO + '/scan_results/planner_check_v6_20261007/cad'
PY = os.environ.get('CAD_PY', '/media/farhan/My Passport/igus_nvidia/venvs/cad/bin/python')


def prep():
    os.makedirs(W + '/data', exist_ok=True)
    for f in glob.glob(CAD + '/*.py'): shutil.copy(f, W)
    for f in glob.glob(CAD + '/data/*.json'):
        if os.path.basename(f).startswith('plans_'): continue
        shutil.copy(f, W + '/data/')
    label = json.load(open(S + '/recommended.json'))['combos'][0]['label']
    for st, pre in (('table', ''), ('stand', 'stand_')):
        for kind in ('BIG', 'SMALL'):
            shutil.copy('%s/plans_%s_%s%s.json' % (S, label, pre, kind), '%s/data/plans_%s_%s.json' % (W, st, kind))
    s = open(W + '/verify_skid.py').read()
    s = s.replace("sys.path.insert(0, P.REPO + '/scan_results/cell_v5_20261007')", "sys.path.insert(0, P.REPO + '/scan_results/cell_v6_20261007')")
    anchor = "        tmax = OUT['poses']['effective_pitch_ux_max_deg']"
    assert anchor in s
    add = """        lift = np.array([abs(r['pen_pad0'] - r['pen_pad1']) for r in rec])
        OUT['v6_second_pad_lift_mm'] = dict(max=float(lift.max()), mean=float(lift.mean()), n_over_1p0=int((lift > 1.0 + 1e-6).sum()), ok=bool(lift.max() <= 1.0 + 1e-6), n=len(rec))
        OUT['v6_by_file'] = {f: dict(n=int(sum(1 for r in rec if r['file'] == f)), pen_min=float(min(r['pen_pad'] for r in rec if r['file'] == f)), lift_max=float(max(abs(r['pen_pad0'] - r['pen_pad1']) for r in rec if r['file'] == f)),
                                     tilt_max=float(max(r['tilt_deg'] for r in rec if r['file'] == f))) for f in sorted(set(r['file'] for r in rec))}
"""
    s = s.replace(anchor, add + anchor)
    open(W + '/verify_skid.py', 'w').write(s)
    print('prepared', W)


def run():
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    log = open(W + '/verify_skid.log', 'w')
    rc = subprocess.call(['nice', '-n', '19', PY, 'verify_skid.py'], cwd=W, env=env, stdout=log, stderr=subprocess.STDOUT)
    print('verify_skid rc', rc)
    sys.exit(rc)


def report():
    k = json.load(open(W + '/data/verification_skid.json')); p = k.get('poses', {})
    lift = k.get('v6_second_pad_lift_mm', {})
    fg = k.get('flap_gap_margin_at_study_tilt_mm', {})
    rows = [('penetration min mm (>= 3.5)', round(p.get('pen_min_mm', -1), 3), p.get('pen_min_mm', 0) >= 3.5),
            ('pad first contact', '%s/%s' % (p.get('pad_first_count'), p.get('n')), p.get('pad_first_count') == p.get('n')),
            ('second-pad lift max mm (<= 1.0)', round(lift.get('max', 9), 3), bool(lift.get('ok'))),
            ('second-pad lift mean mm', round(lift.get('mean', 9), 3), None),
            ('flap-gap support at study tilt, gap 0..6 mm (margin >= 1 mm at 6)', fg, (fg.get('gap_6_mm', -9) >= 1.0)),
            ('flap-gap static (tilt 25 deg) ok', k.get('flap_gap_ok_up_to_6mm_tilt_25deg'), bool(k.get('flap_gap_ok_up_to_6mm_tilt_25deg'))),
            ('pad->blade clearance rest / bent mm', '%s / %s' % (k.get('pad_to_blade_min_mm'), k.get('pad_to_blade_min_bent_mm')), (k.get('pad_to_blade_min_bent_mm') or 0) >= 2.0),
            ('axial stop present', k.get('axial_stop_present'), bool(k.get('axial_stop_present'))),
            ('study tilt max (effective pitch ux) deg', p.get('effective_pitch_ux_max_deg'), None),
            ('margin to next part mm (other_min - pad)', p.get('other_min_minus_pad_min_mm'), (p.get('other_min_minus_pad_min_mm') or 0) > 0)]
    for n, v, ok in rows: print('%-70s %-40s %s' % (n, v, '-' if ok is None else ('PASS' if ok else 'FAIL')))
    print('by file', json.dumps(k.get('v6_by_file')))
    print('by task', json.dumps(p.get('by_task')))


if __name__ == '__main__':
    {'prep': prep, 'run': run, 'report': report}[sys.argv[1]]()
