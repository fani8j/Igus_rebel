"""OpenUSD validation for use INSIDE Isaac Sim; invoked by run_amr.py.
These checks have not been executed in the delivery environment (pxr/Isaac Sim unavailable there).
"""
from __future__ import annotations
from pxr import Usd,UsdGeom,UsdPhysics,Gf

def validate_stage(stage) -> dict:
    errors=[]
    def need(ok,msg):
        if not ok:errors.append(msg)
    if stage is None:return {'status':'FAIL','errors':['No stage'],'physical_motion_tested':False}
    need(UsdGeom.GetStageUpAxis(stage)=='Z','Expected Z-up')
    need(abs(UsdGeom.GetStageMetersPerUnit(stage)-1.)<1e-9,'Expected metres')
    all_prims=list(stage.Traverse())
    bodies=[p for p in all_prims if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    roots=[p for p in all_prims if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    joints=[UsdPhysics.RevoluteJoint(p) for p in all_prims if p.IsA(UsdPhysics.RevoluteJoint)]
    driven=[j for j in joints if j.GetPrim().HasAPI(UsdPhysics.DriveAPI,'angular')]
    need(len(roots)==1,'Expected one articulation root')
    need(len(driven)==2,'Expected two driven joints')
    need(len(joints)==len(bodies)-1,'Joint tree edge count mismatch')
    cache=UsdGeom.XformCache();max_anchor=0.
    for j in joints:
        r0=j.GetBody0Rel().GetTargets();r1=j.GetBody1Rel().GetTargets()
        need(len(r0)==1 and len(r1)==1,'Invalid body relationship '+str(j.GetPath()))
        if len(r0)!=1 or len(r1)!=1:continue
        p0=stage.GetPrimAtPath(r0[0]);p1=stage.GetPrimAtPath(r1[0])
        need(bool(p0) and bool(p1),'Unresolved joint link')
        if not p0 or not p1:continue
        w0=cache.GetLocalToWorldTransform(p0).Transform(Gf.Vec3d(j.GetLocalPos0Attr().Get()))
        w1=cache.GetLocalToWorldTransform(p1).Transform(Gf.Vec3d(j.GetLocalPos1Attr().Get()))
        err=(w0-w1).GetLength();max_anchor=max(max_anchor,err)
        need(err<1e-5,'Joint anchor mismatch: '+str(j.GetPath()))
    total_mass=0.
    for b in bodies:
        api=UsdPhysics.MassAPI(b);m=api.GetMassAttr().Get()
        need(m is not None and m>0,'Missing mass '+str(b.GetPath()))
        if m is not None:total_mass+=m
    for p in all_prims:
        for rel in p.GetRelationships():
            for t in rel.GetTargets():need(bool(stage.GetPrimAtPath(t.GetPrimPath())),'Missing relationship target '+str(t))
        if p.IsA(UsdGeom.Mesh):
            m=UsdGeom.Mesh(p)
            result=UsdGeom.Mesh.ValidateTopology(m.GetFaceVertexIndicesAttr().Get(),m.GetFaceVertexCountsAttr().Get(),len(m.GetPointsAttr().Get()))
            valid,reason=result if isinstance(result,tuple) else (bool(result),'Invalid mesh topology')
            need(valid,str(m.GetPath())+': '+str(reason))
    return {'status':'PASS_OPENUSD_CHECK' if not errors else 'FAIL','errors':errors,
        'rigid_bodies':len(bodies),'joints':len(joints),'drives':len(driven),'mass_kg':total_mass,
        'max_joint_anchor_error_m':max_anchor,'physical_motion_tested':False}
