import sys, numpy as np
sys.path.insert(0, __file__.rsplit('/',1)[0])
from common import *; from vtkrender import *
L = load_leaves()
import collections
G = collections.defaultdict(lambda: ([], [], 0))
V = collections.defaultdict(list); F = collections.defaultdict(list); off = collections.defaultdict(int)
for l in L:
    v, f = load_mesh(l['idx'])
    if len(f)==0: continue
    V[l['group']].append(v); F[l['group']].append(f+off[l['group']]); off[l['group']] += len(v)
items = [(np.vstack(V[g]), np.vstack(F[g]), COLORS[g], 1.0) for g in V]
leg = [(g, COLORS[g]) for g in V]
render(items, (1,1.5,-0.9), (0,0,1), sys.argv[1], 'overview', legend=leg)
