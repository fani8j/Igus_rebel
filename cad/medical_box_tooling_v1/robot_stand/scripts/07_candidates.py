"""Rank supported free-box positions per mounting family, add metrics, and re-verify finalists against the actual mesh
surface (point-to-box distance over the full sweep, independent of the voxel grid).
TARGET=A|B selects the arm that gets the suction tool ("left"); default A (= left if the robot faces +Z with +Y up).
Writes data/candidates_target{T}.json."""
import sys, os, json, itertools, importlib, numpy as np
sys.path.insert(0, __file__.rsplit('/',1)[0])
from common import *
fs = importlib.import_module('06_freespace'); vox = importlib.import_module('05_voxelize')
T = os.environ.get('TARGET', 'A'); O = 'B' if T == 'A' else 'A'
P = 5.0; ORG = np.array([-1700., -1550., 100.]); STROKE = 625.; XSLIDE = 260.
BASE = {'A': np.array([-424., -242., 855.]), 'B': np.array([-884., -242., 855.])}     # arm base-plate centres (CAD pose)
XR = {'A': (-XSLIDE, 0.), 'B': (0., XSLIDE)}            # x-slide range of each arm carriage on the 660 mm rail
ARM_LEN = 960.; SLACK = 150.; REACH = 700.
ENV_X = (-984., -324.); ENV_Z = (278., 1000.)           # stand envelope in the XZ plane (Y is up)
occ = fs.occ
GR = {'F': ['NAUO12','NAUO13','NAUO14','NAUO1_body'], 'Y': ['NAUO1_carriage','NAUO2','NAUO5','NAUO15','NAUO16'],
      'A': ['NAUO3','NAUO4','NAUO8','NAUO10'], 'B': ['NAUO6','NAUO7','NAUO9','NAUO11']}
NAMES = {'NAUO12':'gusset/column','NAUO13':'side plate','NAUO14':'side plate','NAUO1_body':'ZLW body','NAUO1_carriage':'ZLW carriage','NAUO2':'rail bracket',
 'NAUO5':'X rail','NAUO15':'carrier plate A','NAUO16':'carrier plate B','NAUO3':'carriage A','NAUO4':'clamp A','NAUO8':'upper plate A','NAUO10':'arm A',
 'NAUO6':'carriage B','NAUO7':'clamp B','NAUO9':'upper plate B','NAUO11':'arm B'}
def hemi_conflict(ctr, dims, base, xr, yr):
    """box (centre, dims) intersects the arm reach hemisphere (R=700, y >= y_base-20, i.e. above the base plane) for any base position in rect xr x yr"""
    g = np.stack(np.meshgrid(np.linspace(-.5,.5,5), np.linspace(-.5,.5,4), np.linspace(-.5,.5,4), indexing='ij'), -1).reshape(-1,3)
    pts = ctr[:,None,:] + g[None]*dims[:,None,:]
    xb = np.clip(pts[...,0], xr[0], xr[1]); yb = np.clip(pts[...,1], yr[0], yr[1])
    valid = yb <= pts[...,1] + 20
    d = np.sqrt((pts[...,0]-xb)**2 + (pts[...,1]-yb)**2 + (pts[...,2]-base[2])**2)
    return ((d <= REACH) & valid).any(1)
def surf_points(keys, sp=4.0):
    L = load_leaves(); out = []
    for l in L:
        k = l['top'] if l['top'] != 'NAUO1' else ('NAUO1_carriage' if l['group']=='zlw_carriage' else 'NAUO1_body')
        if k in keys:
            v, f = load_mesh(l['idx'])
            if len(f): out.append(vox.raster_leaf(v, f, sp))
    return np.vstack(out)
def min_clear(lo, hi, pts, shifts):
    best = 1e9
    for s in shifts:
        q = pts + s; d = np.maximum(np.maximum(lo - q, q - hi), 0); best = min(best, np.sqrt((d**2).sum(1)).min())
    return best
ys = np.arange(-STROKE, STROKE+1, 12.5)
def Z(x=0., y=0.): return np.array([x, y, 0.])
def shifts_for(fam):
    xa = np.arange(XR['A'][0], XR['A'][1]+1, 20.); xb = np.arange(XR['B'][0], XR['B'][1]+1, 20.)
    if fam in ('FIXED', 'DECK'):
        return {'F':[Z()], 'Y':[Z(0,y) for y in ys], 'A':[Z(x,y) for x in xa for y in ys], 'B':[Z(x,y) for x in xb for y in ys]}
    if fam == 'CARRIAGE':
        return {'F':[Z(0,-y) for y in ys], 'Y':[Z()], 'A':[Z(x) for x in xa], 'B':[Z(x) for x in xb]}
    # arm frame: the target arm's carriage is the frame; everything else moves relative to it
    t = fam[-1]; o = 'B' if t == 'A' else 'A'
    xt = np.arange(XR[t][0], XR[t][1]+1, 20.); xo = np.arange(XR[o][0], XR[o][1]+1, 20.)
    # relative x of other arm: (x_o - x_t) with the no-crossing limit (carriages 200 wide, centres >= 200 apart)
    rel_o = [Z(xx - xt_) for xt_ in xt for xx in xo if abs((BASE[o][0]+xx) - (BASE[t][0]+xt_)) >= 200]
    return {'F':[Z(-x,-y) for x in xt for y in ys], 'Y':[Z(-x) for x in xt], o:rel_o, t:[Z()]}
def main():
    pts = {}
    for g in 'FYAB':
        p = surf_points(GR[g])
        if len(p) > 80000: p = p[np.random.default_rng(0).choice(len(p), 80000, replace=False)]
        pts[g] = p
    res = {}
    for fam in ('FIXED', 'DECK', 'CARRIAGE', f'ARM_{T}'):
        d = np.load(f"{W}/freespace_{fam}.npz"); ctr = []; dims = []
        for k in d.files:
            c = d[k]; ctr.append(c); dims += [np.array([float(t) for t in k.split('x')])]*len(c)
        ctr = np.vstack(ctr).astype(float); dims = np.array(dims)
        # base position ranges of target (T) and other (O) arm in the family frame
        yfull = (BASE['A'][1]-STROKE, BASE['A'][1]+STROKE); yfix = (BASE['A'][1], BASE['A'][1])
        if fam in ('FIXED','DECK'): rT = (tuple(BASE[T][0]+np.array(XR[T])), yfull); rO = (tuple(BASE[O][0]+np.array(XR[O])), yfull)
        elif fam == 'CARRIAGE': rT = (tuple(BASE[T][0]+np.array(XR[T])), yfix); rO = (tuple(BASE[O][0]+np.array(XR[O])), yfix)
        else: rT = ((BASE[T][0],)*2, yfix); rO = (tuple(BASE[O][0]+np.array(XR[O])), yfix)
        def l1(r):
            worst = np.zeros(len(ctr))
            for bx in r[0]:
                for by in r[1]: worst = np.maximum(worst, np.abs(ctr - np.array([bx, by, BASE['A'][2]])).sum(1))
            return worst
        legT = l1(rT); legO = l1(rO)
        totT = legT + ARM_LEN + SLACK; totO = legO + ARM_LEN + SLACK
        nomT = np.abs(ctr - BASE[T]).sum(1) + ARM_LEN + SLACK
        cT = np.zeros(len(ctr), bool); cO = np.zeros(len(ctr), bool)
        for i in range(0, len(ctr), 50000):
            s = slice(i, i+50000)
            cT[s] = hemi_conflict(ctr[s], dims[s], BASE[T], rT[0], rT[1]); cO[s] = hemi_conflict(ctr[s], dims[s], BASE[O], rO[0], rO[1])
        lo = ctr - dims/2; hi = ctr + dims/2
        prot = np.maximum(np.maximum(np.maximum(ENV_X[0]-lo[:,0], hi[:,0]-ENV_X[1]), np.maximum(ENV_Z[0]-lo[:,2], hi[:,2]-ENV_Z[1])), 0)
        conflict = cT | cO
        def pick(mask, key, k):
            ch = []
            for i in [i for i in np.argsort(key) if mask[i]]:
                if all(np.linalg.norm(ctr[i]-ctr[j]) > 150 for j in ch): ch.append(i)
                if len(ch) >= k: break
            return ch
        tag = {}
        for i in pick(~conflict & (prot == 0), totT, 10): tag[i] = 'strict'
        for i in pick(np.ones(len(ctr), bool), totT, 10): tag.setdefault(i, 'relaxed')
        sh_ = shifts_for(fam); recs = []
        for i in tag:
            cl = {g: float(min_clear(lo[i], hi[i], pts[g], sh_[g])) for g in 'FYAB'}
            ix = np.floor((np.concatenate([lo[i]-40, hi[i]+40]) - np.tile(ORG,2))/P).astype(int)
            near = [NAMES[k] for k in NAMES if occ[k][max(ix[0],0):ix[3]+1, max(ix[1],0):ix[4]+1, max(ix[2],0):ix[5]+1].any()]
            recs.append(dict(family=fam, list=tag[i], centre=[round(float(v),1) for v in ctr[i]], dims_xyz=[float(v) for v in dims[i]],
                lo=[round(float(v),1) for v in lo[i]], hi=[round(float(v),1) for v in hi[i]], height_above_deck_mm=[round(float(lo[i][1]+1351)), round(float(hi[i][1]+1351))],
                leg_to_target_base_mm=round(float(legT[i])), tube_total_worst_mm=round(float(totT[i])), tube_total_nominal_mm=round(float(nomT[i])),
                tube_total_to_other_arm_mm=round(float(totO[i])), in_reach_target=bool(cT[i]), in_reach_other=bool(cO[i]), outside_stand_envelope_mm=round(float(prot[i])),
                clearance_exact_mm={k: round(v,1) for k, v in cl.items()}, min_clearance_exact_mm=round(min(cl.values()),1), near_solids=near))
            r = recs[-1]; print(fam, r['list'], r['centre'], r['dims_xyz'], 'leg', r['leg_to_target_base_mm'], 'tube', r['tube_total_worst_mm'], 'reach', cT[i], cO[i], 'out', r['outside_stand_envelope_mm'], 'clr', r['min_clearance_exact_mm'], near[:4], flush=True)
        res[fam] = recs
    json.dump(dict(target=T, results=res), open(f'data/candidates_target{T}.json','w'), indent=1)
main()
