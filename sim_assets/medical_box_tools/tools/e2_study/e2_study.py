import sys, glob, os, json
import numpy as np, trimesh
sys.path.insert(0,'/home/farhan/orca/workspaces/igus_rebel_ros2/box-3d/sim_assets/medical_box_tools/tools')
import robot_kin as rk
V5='/home/farhan/orca/workspaces/igus_rebel_ros2/box-3d/cad/medical_box_tooling_v1/suction_tool_v5/meshes/collision/'
cab=rk.Cabinet(); tg=cab.pick(6,'angiolite','first'); c=np.array(tg['c'])
spine=c[1]-0.125; P=np.array([c[0],spine,c[2]])
def tool_points(shift_back=0.0, drop=()):
    pts={}
    for f in sorted(glob.glob(V5+'*.stl')):
        n=os.path.basename(f).rsplit('_',1)[0]
        if n in drop: continue
        m=trimesh.load(f)
        p=np.vstack([m.vertices, m.sample(1500)])*1000.0
        pts.setdefault(n,[]).append(p)
    return {k:np.vstack(v) for k,v in pts.items()}
obst={k:v for k,v in cab.boxes.items() if k in ('Body_Left','FrontTrim_Left','Shelf_06','Shelf_07','TopClosedPanel','Body_Back')}
st=cab.static_obbs(6,tg)
for k,(cc,h) in st.items():
    if k.startswith('carton:6'): obst[k]=(cc-h,cc+h)
def dist(Pc):
    out={}
    for k,(lo,hi) in obst.items():
        d=np.maximum(np.maximum(lo-Pc,Pc-hi),0); out[k]=np.linalg.norm(d,axis=1)
    return out
def place(pts_mm, yaw_deg, off_mm=0.0, tcp=154.0):
    th=np.radians(yaw_deg)
    z=np.array([-np.sin(th),np.cos(th),0]); x=np.array([0,0,1.0]); y=np.cross(z,x)
    R=np.column_stack([x,y,z])
    Q=(pts_mm-np.array([0,0,tcp]))/1000.0
    return P+np.array([off_mm/1000,0,0])+Q@R.T
if __name__=='__main__':
    tp=tool_points()
    # profile table: wall-side and stock-side half width vs z (5 mm bins), per whole tool
    allp=np.vstack(list(tp.values()))
    print('z range',allp[:,2].min(),allp[:,2].max())
    print('z   ymin(-y side)  ymax(+y side)  limiting part(-y)')
    for z0 in range(50,160,6):
        sel={k:v[(v[:,2]>=z0)&(v[:,2]<z0+6)] for k,v in tp.items()}
        sel={k:v for k,v in sel.items() if len(v)}
        if not sel: continue
        mn=min(sel.items(),key=lambda kv:kv[1][:,1].min()); mx=max(sel.items(),key=lambda kv:kv[1][:,1].max())
        print(f'{z0:4d}-{z0+6:<4d} {mn[1][:,1].min():7.1f} ({mn[0]:<13s}) {mx[1][:,1].max():7.1f} ({mx[0]})   depth inside front plane y_cab(mm) {(spine*1000-(154-z0)):6.1f}..{(spine*1000-(154-z0-6)):6.1f}')
    print('straight approach, offset sweep: parts closer than 0.5 mm to cabinet/stock (name:obstacle:count)')
    for off in (-3,0,3):
        Pc={k:place(v,0,off) for k,v in tp.items()}
        hits={}
        for k,pp in Pc.items():
            for o,d in dist(pp).items():
                n=int((d<0.0005).sum())
                if n: hits[f'{k}:{o}']=n
        print(' offset',off,'mm',hits)
    print('yaw sweep (offset 0): colliding part groups')
    for yaw in (0,5,8,10,12,15,18,20,25,30):
        Pc={k:place(v,yaw,0) for k,v in tp.items()}
        hits=set(); mind=9
        for k,pp in Pc.items():
            for o,d in dist(pp).items():
                if (d<0.0005).any(): hits.add(f'{k}:{o}')
                mind=min(mind,d.min())
        print(' yaw',yaw,sorted(hits)[:8], len(hits))
