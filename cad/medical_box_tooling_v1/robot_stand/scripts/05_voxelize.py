"""Voxelise every leaf (conservative surface raster + cavity fill) onto a common 5 mm grid; save per-group occupancy."""
import sys, numpy as np, time
sys.path.insert(0, __file__.rsplit('/',1)[0])
from common import *
from scipy import ndimage as ndi
PITCH = 5.0
ORG = np.array([-1700., -1550., 100.])
SHAPE = (440, 640, 240)      # x -1700..500, y -1550..1650, z 100..1300
def raster_leaf(v, f, sp=PITCH/3.0):
    tri = v[f]; a = tri[:,0]; b = tri[:,1]; c = tri[:,2]
    e = np.maximum(np.linalg.norm(b-a,axis=1), np.maximum(np.linalg.norm(c-b,axis=1), np.linalg.norm(a-c,axis=1)))
    n = np.maximum(1, np.ceil(e/sp).astype(int)); pts=[]
    for k in np.unique(n):
        idx = np.where(n==k)[0]
        i, j = np.meshgrid(np.arange(k+1), np.arange(k+1), indexing='ij'); m = (i+j)<=k
        u = (i[m]/k)[None,:,None]; w = (j[m]/k)[None,:,None]
        P = a[idx][:,None,:]*(1-u-w) + b[idx][:,None,:]*u + c[idx][:,None,:]*w
        pts.append(P.reshape(-1,3))
    return np.vstack(pts)
if __name__ == '__main__':
    L = load_leaves(); occ = {}; t0=time.time()
    for l in L:
        v, f = load_mesh(l['idx'])
        if len(f)==0: continue
        P = raster_leaf(v, f)
        ij = np.floor((P-ORG)/PITCH).astype(int)
        lo = ij.min(0); hi = ij.max(0)+1
        sub = np.zeros(hi-lo+2, bool); q = ij-lo+1; sub[q[:,0],q[:,1],q[:,2]] = True
        # cavity fill only if closed; solid leaves are closed
        filled = ndi.binary_fill_holes(sub)
        key = l['top'] if l['top']!='NAUO1' else ('NAUO1_carriage' if l['group']=='zlw_carriage' else 'NAUO1_body')
        g = occ.setdefault(key, np.zeros(SHAPE, bool))
        sl = tuple(slice(lo[k]-1, lo[k]-1+filled.shape[k]) for k in range(3))
        g[sl] |= filled
    print("done", time.time()-t0)
    np.savez_compressed(f"{W}/occ5_comp.npz", **occ)
    for k, g in occ.items(): print(k, int(g.sum()), "voxels ->", round(g.sum()*PITCH**3/1e6,2), "L")
