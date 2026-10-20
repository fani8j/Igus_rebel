"""Region-level numbers: (1) bracket pocket on the carriage (extent + max uniform clearance), (2) free deck area next to the column,
(3) free volume beside the column (heights). Writes data/regions.json."""
import sys, json, itertools, numpy as np
sys.path.insert(0, __file__.rsplit('/',1)[0])
from common import *
import importlib; fs = importlib.import_module('06_freespace')
P = fs.P; ORG = fs.ORG; occ = fs.occ; U = fs.U; sweep = fs.sweep
Z0 = fs.Z0
F = U('NAUO12','NAUO13','NAUO14','NAUO1_body'); Ym = U('NAUO1_carriage','NAUO2','NAUO5','NAUO15','NAUO16')
Araw = U('NAUO3','NAUO4','NAUO8','NAUO10'); Braw = U('NAUO6','NAUO7','NAUO9','NAUO11')
MOV = Ym | fs.sweep(Araw,0,-260,0) | fs.sweep(Braw,0,0,260)
forb_c = sweep(F,1,-625,625) | MOV
sl = (slice(40,400), slice(10,470), slice(10,230)); OFF = np.array([s.start for s in sl])
def sat(a):
    s = np.zeros(tuple(n+1 for n in a.shape), np.int32); s[1:,1:,1:] = a.astype(np.int32).cumsum(0).cumsum(1).cumsum(2); return s
def boxsum(S, d):
    nx, ny, nz = d; X, Y, Z = S.shape
    return (S[nx:,ny:,nz:] - S[:X-nx,ny:,nz:] - S[nx:,:Y-ny,nz:] - S[nx:,ny:,:Z-nz] + S[:X-nx,:Y-ny,nz:] + S[:X-nx,ny:,:Z-nz] + S[nx:,:Y-ny,:Z-nz] - S[:X-nx,:Y-ny,:Z-nz])
def free_centres(forb_sat, perm, clr):
    d = [int(round((b+2*clr)/P)) for b in perm]
    idx = np.argwhere(boxsum(forb_sat, d) == 0)
    return (idx + OFF + np.array(d)/2.0)*P + ORG
out = {}
perms = sorted(set(itertools.permutations(fs.BOX)))
# (1) pocket between the bracket gussets: inner walls x -749..-555, y -631..-291 (below the shelf), in front of floor plate z>=747
Sc = sat(forb_c[sl]); pocket = {}
for clr in (10, 15, 20, 25, 30, 40):
    rows = []
    for perm in perms:
        c = free_centres(Sc, perm, clr); h = np.array(perm)/2
        m = (c[:,0]-h[0] >= -749) & (c[:,0]+h[0] <= -555) & (c[:,1]-h[1] >= -631) & (c[:,1]+h[1] <= -291) & (c[:,2]-h[2] >= 747)
        if m.any(): rows.append(dict(perm=[float(v) for v in perm], n=int(m.sum()), cx=[float(c[m,0].min()), float(c[m,0].max())], cy=[float(c[m,1].min()), float(c[m,1].max())], cz=[float(c[m,2].min()), float(c[m,2].max())]))
    pocket[str(clr)] = rows
    print('pocket clr', clr, [(r['perm'], r['n']) for r in rows])
out['pocket_on_carriage'] = pocket
# (2) deck next to the column: boxes whose bottom is 10..20 mm above the deck plane y=-1351 (virtual), carriage group swept, F static
deck = Z0.copy(); deck[:, :int(round((-1351-ORG[1])/P)), :] = True
Sd = sat((F | sweep(MOV,1,-625,625) | deck)[sl]); res = []
foot = np.array([[-702,278],[-602,675]])   # column foot in x,z
for perm in perms:
    c = free_centres(Sd, perm, 10); h = np.array(perm)/2
    m = (c[:,1]-h[1] <= -1351+15)
    c = c[m]
    if not len(c): continue
    dx = np.maximum(np.maximum(foot[0,0]-(c[:,0]+h[0]), (c[:,0]-h[0])-foot[1,0]), 0); dz = np.maximum(np.maximum(foot[0,1]-(c[:,2]+h[2]), (c[:,2]-h[2])-foot[1,1]), 0)
    gap = np.hypot(dx, dz); i = np.argmin(gap)
    # free deck region extent within the search domain, per quadrant relative to the column
    res.append(dict(perm=[float(v) for v in perm], n=int(len(c)), nearest_to_column_foot=dict(centre=[float(v) for v in c[i]], gap_xz_mm=float(gap[i])),
                    x_range=[float(c[:,0].min()), float(c[:,0].max())], z_range=[float(c[:,2].min()), float(c[:,2].max())]))
    print('deck', perm, len(c), res[-1]['nearest_to_column_foot'])
out['deck'] = res
json.dump(out, open('data/regions.json','w'), indent=1)
