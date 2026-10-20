import numpy as np, sys
sys.path.insert(0,__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import e2_study as S
tp=S.tool_points()
res={}
for gap in np.arange(60,-0.1,-1.0):       # mm before the pressed pose (compressed TCP at the spine = gap 0)
    S.P_save=S.P.copy()
    S.P=np.array([S.P_save[0],S.P_save[1]-gap/1000,S.P_save[2]])
    for k,v in tp.items():
        Pc=S.place(v,0,0)
        for o,d in S.dist(Pc).items():
            if (d<0.0005).any() and (k,o) not in res: res[(k,o)]=gap
    S.P=S.P_save
for (k,o),g in sorted(res.items(),key=lambda kv:-kv[1]): print(f'{k:14s} hits {o:15s} first at {g:4.0f} mm before the pressed pose')
