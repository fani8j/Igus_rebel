import numpy as np, sys
sys.path.insert(0,__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from e2_study import *
tp0=tool_points()
def variant(name):
    tp={k:v.copy() for k,v in tp0.items()}
    if 'mirror' in name:      # collector bay + collector to +y side, 4 mm lower
        b=tp['tool_body']; sel=(b[:,1]<-20)&(b[:,2]<100)
        b2=b[sel].copy(); b2[:,1]*=-1; b2[:,2]-=4
        tp['tool_body']=np.vstack([b[~sel],b2])
        c=tp['collector']; c=c.copy(); c[:,1]*=-1; c[:,2]-=4; tp['collector']=c
    if 'shave' in name:       # wall side (-y) limited to -8 mm for z>=84 (stem, bar, washers, nuts, bosses)
        for k in ('tool_body',):
            v=tp[k]; m=(v[:,2]>=84)&(v[:,1]<-8); v[m,1]=-8
        for k in [k for k in tp if k.startswith(('washer','nut'))]:
            v=tp[k]; m=v[:,1]<-8; v[m,1]=-8
    if 'noelbow' in name:
        for k in [k for k in tp if k.startswith('elbow')]: tp.pop(k)
    if 'padsonly' in name:
        tp={k:v for k,v in tp.items() if k.startswith('pad')}
    return tp
def check(tp,yaw,off,margin=0.002):
    bad=set()
    for k,v in tp.items():
        Pc=place(v,yaw,off)
        for o,d in dist(Pc).items():
            if (d<margin).any(): bad.add(f'{k}:{o}')
    return bad
for name in ('v5','v5_mirror','v5_mirror_shave','v5_mirror_shave_noelbow','padsonly'):
    tp=variant(name)
    for off in (0,2):
        res=None
        for yaw in range(0,41):
            b=check(tp,yaw,off)
            if not b: res=yaw; break
        # list remaining at yaw = 15 for info
        print(name,'offset',off,'mm: min yaw with clearance >= 2 mm =',res, '| at yaw 15:',sorted(check(tp,15,off))[:5])
