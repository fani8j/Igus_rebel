"""Free-box search for the pump box (200x110x100 mm, 6 axis-aligned orientations, >=10 mm clearance, 5 mm voxels).

Mounting families (relative-motion frames). DECK = box on a virtual AMR deck plane at y=-1351 (the gusset's lowest face; the CAD contains no AMR):
  FIXED     box rigid with gusset/side plates/ZLW body; must clear everything the moving groups sweep
            (ZLW carriage y +-625 mm, arms+carriages sliding on the 660 mm X rail).
  CARRIAGE  box rigid with the ZLW carriage group (moves in y only); must clear the fixed parts over the whole y stroke
            and the arms/carriages sliding in x on the rail.
  ARM_B     box rigid with arm B's X-carriage (moves in x AND y with the arm); must clear everything else in B's frame.
Writes the raw per-position results (centres) to the external drive: freespace_<family>.npz."""
import sys, itertools, numpy as np
sys.path.insert(0, __file__.rsplit('/',1)[0])
from common import *
P = 5.0; ORG = np.array([-1700., -1550., 100.])
BOX = (200., 110., 100.); CLR = 10.; GAP = 40.   # GAP: max box-face distance to a mount solid to count as "supported" (via bracket/standoff)
STROKE = 625.; XSLIDE = 260.
occ = dict(np.load(f"{W}/occ5_comp.npz"))
Z0 = np.zeros_like(occ['NAUO12'])
def U(*keys):
    a = Z0.copy()
    for k in keys: a |= occ[k]
    return a
def sweep(a, axis, lo_mm, hi_mm):
    """union of `a` translated by every offset in [lo_mm, hi_mm] along `axis` (exact, cumulative sums)"""
    lo = int(round(lo_mm/P)); hi = int(round(hi_mm/P)); n = a.shape[axis]
    cp = np.concatenate([np.zeros_like(np.take(a, [0], axis=axis), dtype=np.int32), np.cumsum(a.astype(np.int32), axis=axis)], axis=axis)
    idx = np.arange(n); j0 = np.clip(idx-hi, 0, n); j1 = np.clip(idx-lo+1, 0, n)
    return (np.take(cp, j1, axis=axis) - np.take(cp, j0, axis=axis)) > 0
if __name__ == '__main__':
    F = U('NAUO12','NAUO13','NAUO14','NAUO1_body')
    Ym = U('NAUO1_carriage','NAUO2','NAUO5','NAUO15','NAUO16')
    Araw = U('NAUO3','NAUO4','NAUO8','NAUO10'); Braw = U('NAUO6','NAUO7','NAUO9','NAUO11')
    Ax = sweep(Araw, 0, -XSLIDE, 0); Bx = sweep(Braw, 0, 0, XSLIDE)
    MOV_static_x = Ym | Ax | Bx
    DECKSOLID = Z0.copy(); DECKSOLID[:, :int(round((-1351-ORG[1])/P)), :] = True      # virtual AMR deck: everything below y=-1351 (assumed; the CAD has no AMR)
    fam = {
      'FIXED':    (F | sweep(MOV_static_x, 1, -STROKE, STROKE), U('NAUO12','NAUO13','NAUO14')),
      'CARRIAGE': (sweep(F, 1, -STROKE, STROKE) | MOV_static_x, U('NAUO2','NAUO1_carriage')),
      # B frame: relative to B everything else moves by (-b in x, -yo in y); A relative to B: x in [-XSLIDE,0] (they cannot cross)
      'ARM_A':    (sweep(sweep(F, 1, -STROKE, STROKE), 0, 0, XSLIDE) | sweep(Ym | Braw, 0, 0, XSLIDE) | Araw, U('NAUO3','NAUO4','NAUO8')),
      'DECK':     (F | sweep(MOV_static_x, 1, -STROKE, STROKE) | DECKSOLID, DECKSOLID),
      'ARM_B':    (sweep(sweep(F, 1, -STROKE, STROKE), 0, -XSLIDE, 0) | sweep(Ym | Araw, 0, -XSLIDE, 0) | Braw, U('NAUO6','NAUO7','NAUO9')),
    }
    sl = (slice(40,400), slice(10,470), slice(10,230)); OFF = np.array([s.start for s in sl])   # x -1500..300, y -1500..800, z 150..1250
    def sat(a):
        s = np.zeros(tuple(n+1 for n in a.shape), np.int32); s[1:,1:,1:] = a.astype(np.int32).cumsum(0).cumsum(1).cumsum(2); return s
    def boxsum(S, d):
        nx, ny, nz = d; X, Y, Z = S.shape
        return (S[nx:,ny:,nz:] - S[:X-nx,ny:,nz:] - S[nx:,:Y-ny,nz:] - S[nx:,ny:,:Z-nz] + S[:X-nx,:Y-ny,nz:] + S[:X-nx,ny:,:Z-nz] + S[nx:,:Y-ny,:Z-nz] - S[:X-nx,:Y-ny,:Z-nz])
    for name, (forb, mount) in fam.items():
        Sf = sat(forb[sl]); Sm = sat(mount[sl]); out = {}
        for perm in sorted(set(itertools.permutations(BOX))):
            d_in = [int(round((b+2*CLR)/P)) for b in perm]; d_gap = [int(round((b+2*GAP)/P)) for b in perm]
            free = boxsum(Sf, d_in) == 0
            marg = [(g-i)//2 for g, i in zip(d_gap, d_in)]
            Smm = boxsum(Sm, d_gap); X, Y, Z = free.shape; gx, gy, gz = Smm.shape; xa, ya, za = marg
            tmp = np.zeros_like(free); xe, ye, ze = min(xa+gx, X), min(ya+gy, Y), min(za+gz, Z)
            tmp[xa:xe, ya:ye, za:ze] = Smm[:xe-xa, :ye-ya, :ze-za] > 0      # tmp[i] = Smm[i - marg]
            ok = free & tmp
            idx = np.argwhere(ok); ctr = (idx + OFF + np.array(d_in)/2.0)*P + ORG   # true-box centre in mm
            out[f"{perm[0]:.0f}x{perm[1]:.0f}x{perm[2]:.0f}"] = ctr.astype(np.float32)
            print(name, perm, "collision-free", int(free.sum()), "supported", len(idx), flush=True)
        np.savez_compressed(f"{W}/freespace_{name}.npz", **out)
