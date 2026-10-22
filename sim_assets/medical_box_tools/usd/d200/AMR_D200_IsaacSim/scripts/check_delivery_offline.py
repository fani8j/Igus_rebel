#!/usr/bin/env python3
"""Independent read-back of this package's authored USDA subset and geometry.
This is NOT Pixar's USD parser, usdchecker, or an Isaac Sim runtime test.
Requires numpy, scipy, trimesh, lark. Run drive_math.py separately for pure-Python drive-math tests.
"""
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from collections import Counter
import json, math
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from lark import Lark, Transformer

GRAMMAR=r'''
start: HEADER meta prim+
meta: "(" assignment* ")"
assignment: LISTOP? NAME "=" value
prim: "def" NAME ESCAPED_STRING meta? "{" (prim | prop)* "}"
prop: QUALIFIER* typename NAME ("=" value)? meta?
typename: NAME ARRAYTYPE?
?value: SIGNED_NUMBER -> num
      | ESCAPED_STRING -> string
      | "true" -> true
      | "false" -> false
      | PATH -> path
      | array
      | tuple
      | dictionary
array: "[" [value ("," value)* [","]] "]"
tuple: "(" [value ("," value)* [","]] ")"
dictionary: "{" dictitem* "}"
dictitem: NAME NAME "=" value
HEADER: /#usda[^\n]*/
LISTOP.3: /(?:prepend|append|delete)\b/
QUALIFIER.3: /(?:uniform|custom|varying)\b/
ARRAYTYPE: "[]"
NAME: /[A-Za-z_][A-Za-z0-9_:.-]*/
PATH: /<[^>]+>/
%import common.ESCAPED_STRING
%import common.SIGNED_NUMBER
%import common.WS
%ignore WS
'''

class PathValue(str):pass
@dataclass
class Property:
    type: str
    name: str
    value: object
    meta: dict
@dataclass
class Node:
    type: str
    name: str
    meta: dict
    props: dict
    children: list

class Reader(Transformer):
    def num(self,x):return float(x[0])
    def string(self,x):return json.loads(x[0])
    def true(self,x):return True
    def false(self,x):return False
    def path(self,x):return PathValue(str(x[0])[1:-1])
    def array(self,x):return list(x)
    def tuple(self,x):return tuple(x)
    def assignment(self,x):return str(x[-2]),x[-1]
    def meta(self,x):return dict(x)
    def dictionary(self,x):return dict(x)
    def dictitem(self,x):return str(x[1]),x[2]
    def typename(self,x):return ''.join(map(str,x))
    def prop(self,x):
        while x and getattr(x[0],'type','')=='QUALIFIER':x.pop(0)
        typ,name=x[:2];v=x[2] if len(x)>2 and not isinstance(x[2],dict) else None
        meta=x[-1] if x and isinstance(x[-1],dict) else {}
        return Property(str(typ),str(name),v,meta)
    def prim(self,x):
        typ,name=x[:2];rest=x[2:];meta=rest.pop(0) if rest and isinstance(rest[0],dict) else {}
        props=[p for p in rest if isinstance(p,Property)]
        if len(set(p.name for p in props))!=len(props):raise ValueError('Duplicate property')
        return Node(str(typ),json.loads(name),meta,{p.name:p for p in props},[p for p in rest if isinstance(p,Node)])
    def start(self,x):return {'meta':x[1],'roots':x[2:]}

def value(n,k,default=None):return n.props[k].value if k in n.props else default

def check(path):
    parser=Lark(GRAMMAR,parser='lalr',transformer=Reader(),maybe_placeholders=False)
    doc=parser.parse(path.read_text(encoding='utf-8'))
    nodes={};world={};errors=[]
    def need(ok,msg):
        if not bool(ok):errors.append(msg)
    def walk(n,parent='',parent_mat=None):
        p=parent+'/'+n.name;need(p not in nodes,'Duplicate prim: '+p);nodes[p]=n
        t=np.eye(4)
        for op in value(n,'xformOpOrder',[]):
            a=value(n,op);m=np.eye(4)
            if op=='xformOp:translate':m[:3,3]=a
            elif op=='xformOp:scale':m[:3,:3]=np.diag(a)
            elif op=='xformOp:orient':m[:3,:3]=Rotation.from_quat([*a[1:],a[0]]).as_matrix()
            else:raise ValueError(op)
            t=t@m
        world[p]=(np.eye(4) if parent_mat is None else parent_mat)@t
        for c in n.children:walk(c,p,world[p])
    for n in doc['roots']:walk(n)
    need(doc['meta']['metersPerUnit']==1,'Unit scale')
    need(doc['meta']['upAxis']=='Z','Up axis')
    need('/'+doc['meta']['defaultPrim'] in nodes,'defaultPrim missing')
    for p,n in nodes.items():
        for a in n.props.values():
            vs=a.value if isinstance(a.value,list) else [a.value]
            for v in vs:
                if isinstance(v,PathValue):
                    target,sep,prop=v.partition('.')
                    need(target in nodes,'Unresolved prim path: '+str(v))
                    if sep and target in nodes:need(prop in nodes[target].props,'Missing connection target: '+str(v))
    bodies={p:n for p,n in nodes.items() if 'PhysicsRigidBodyAPI' in n.meta.get('apiSchemas',[])}
    roots=[p for p,n in nodes.items() if 'PhysicsArticulationRootAPI' in n.meta.get('apiSchemas',[])]
    joints={p:n for p,n in nodes.items() if n.type=='PhysicsRevoluteJoint'}
    drives=[p for p,n in joints.items() if 'PhysicsDriveAPI:angular' in n.meta.get('apiSchemas',[])]
    need(len(roots)==1,'Exactly one articulation root')
    need(len(drives)==2,'Exactly two driven joints')
    need(len(joints)==len(bodies)-1,'Articulation tree edge count')
    parents={};anchor_errors=[]
    for p,n in joints.items():
        b0=value(n,'physics:body0');b1=value(n,'physics:body1')
        need(b0 in bodies and b1 in bodies,'Joint body missing')
        need(b1 not in parents,'Multiple parent joints: '+str(b1));parents[b1]=b0
        if b0 in bodies and b1 in bodies:
            p0=world[b0]@np.r_[value(n,'physics:localPos0'),1]
            p1=world[b1]@np.r_[value(n,'physics:localPos1'),1]
            err=np.linalg.norm(p0[:3]-p1[:3]);anchor_errors.append(float(err));need(err<1e-8,'Joint anchor mismatch: '+p)
        for k in ['physics:localRot0','physics:localRot1']:
            need(abs(np.linalg.norm(value(n,k))-1)<1e-6,'Nonunit joint frame')
    for b,n in bodies.items():
        need(value(n,'physics:mass',0)>0,'Invalid mass: '+b)
        ii=np.array(value(n,'physics:diagonalInertia'));need((ii>0).all(),'Invalid inertia')
        need((2*ii<=ii.sum()+1e-8).all(),'Unphysical diagonal inertia triangle inequality')
        need(value(n,'physics:kinematicEnabled') is False,'Kinematic body present')
        visited=set();p=b
        while p in parents:
            need(p not in visited,'Cycle in joint tree')
            if p in visited:break
            visited.add(p);p=parents[p]
        need(p==roots[0],'Link not connected to root: '+b)
    colliders={p:n for p,n in nodes.items() if 'PhysicsCollisionAPI' in n.meta.get('apiSchemas',[])}
    meshes={p:n for p,n in nodes.items() if n.type=='Mesh'}
    for p,n in meshes.items():
        v=np.array(value(n,'points'));indices=np.array(value(n,'faceVertexIndices'),int)
        counts=np.array(value(n,'faceVertexCounts'),int)
        need(np.isfinite(v).all(),'NaN vertex');need((counts==3).all(),'Nontriangular face in generated subset')
        need(indices.min()>=0 and indices.max()<len(v),'Mesh index out of bounds')
        need(counts.sum()==len(indices),'Mesh face count mismatch')
        mesh=trimesh.Trimesh(v,indices.reshape(-1,3),process=False)
        need(mesh.is_watertight,'Mesh not watertight: '+p)
        need(mesh.is_winding_consistent,'Mesh winding inconsistent: '+p)
        need(mesh.volume>0,'Mesh orientation negative: '+p)
        if p in colliders:need(len(v)<=128,'Collider vertices exceed authored cooking limit')
    visuals={p:n for p,n in nodes.items() if 'source:componentIndex' in n.props}
    need(len(visuals)==59,'Source component count changed')
    need(set(int(value(n,'source:componentIndex')) for n in visuals.values())==set(range(59)),'Components missing/duplicated')
    visual_types=Counter(n.type for n in visuals.values())
    need(visual_types=={'Cylinder':44,'Cube':10,'Mesh':5},'Unexpected primitive reconstruction count')
    source_mesh=trimesh.load_mesh(path.parent/'source/amr_d200.stl').split(only_watertight=False)
    R=np.array([[0,1,0],[-1,0,0],[0,0,1]])
    # The scene has a +3 mm spawn clearance, excluded from source-pose comparison.
    root=[p for p,n in nodes.items() if value(n,'amr:assetId')][0]
    root_inv=np.linalg.inv(world[root]);pose_error=[];mesh_vertex_error=[];bottoms=[]
    for p,n in visuals.items():
        i=int(value(n,'source:componentIndex'));m=source_mesh[i]
        if n.type in ['Cylinder','Cube']:
            c=(root_inv@world[p])[:3,3];source_c=(m.bounds.mean(0)@R.T)
            pose_error.append(float(np.linalg.norm(c-source_c)))
            need(np.linalg.norm(c-source_c)<1e-6,'Changed source shape centre: '+p)
        else:
            vv=np.array(value(n,'points'));T=root_inv@world[p]
            w=vv@T[:3,:3].T+T[:3,3]
            mesh_vertex_error.append(float(np.abs(w-m.vertices@R.T).max()))
            need(mesh_vertex_error[-1]<1e-8,'Source mesh moved: '+p)
    for p,n in colliders.items():
        if n.type=='Cylinder':
            axis=value(n,'axis');need(axis=='Y','Unexpected wheel axis')
            bottom=world[p][2,3]-value(n,'radius');bottoms.append(float(bottom))
    mode=value(nodes[root],'amr:casterMode')
    need(len(bodies)==(15 if mode=='articulated' else 3),'Wrong body count')
    params=json.loads((path.parent/'amr_parameters.json').read_text())
    expected_mass=params['base_mass_kg']+2*params['drive_wheel_mass_kg']+4*params['caster_fork_mass_kg']+8*params['caster_wheel_mass_kg']
    need(abs(sum(value(n,'physics:mass') for n in bodies.values())-expected_mass)<1e-5,'Total mass does not match parameters')
    r=value(nodes[root],'amr:wheelRadius');b=value(nodes[root],'amr:wheelTrack')
    left=value(nodes[root+'/joints/left_wheel_joint'],'physics:body1');right=value(nodes[root+'/joints/right_wheel_joint'],'physics:body1')
    need(world[left][1,3]>world[right][1,3],'Left/right ordering')
    need(abs(world[left][1,3]-world[right][1,3]-b)<1e-8,'Wheel track mismatch')
    for p in drives:need(value(nodes[p],'physics:axis')=='Y','Drive axis must be +Y')
    report={'file':path.name,'status':'PASS_OFFLINE_SUBSET_CHECK' if not errors else 'FAIL','errors':errors,
            'usd_prims':len(nodes),'rigid_bodies':len(bodies),'joints':len(joints),'drive_joints':len(drives),
            'colliders':len(colliders),'source_visual_types':dict(visual_types),'total_mass_kg':sum(value(n,'physics:mass') for n in bodies.values()),
            'max_joint_anchor_error_m':max(anchor_errors),'max_primitive_center_error_m':max(pose_error),
            'max_preserved_visual_mesh_coordinate_error_m':max(mesh_vertex_error),
            'wheel_collision_bottom_world_z_range_m':[min(bottoms),max(bottoms)],
            'self_contained':True,'external_asset_references':0,
            'validation_scope':'Read back with package-specific Lark USDA grammar and independent numeric/topology checks; NOT OpenUSD or Isaac Sim validation.',
            'openusd_readback_tested':False,'isaac_sim_runtime_tested':False}
    return report,doc,nodes,world

if __name__=='__main__':
    package=Path(__file__).resolve().parent.parent
    reports=[]
    for p in sorted(package.glob('*.usd')):
        report,*_=check(p);reports.append(report);print(p.name,report['status'],report['errors'])
    out={'results':reports,'all_offline_checks_pass':all(r['status']=='PASS_OFFLINE_SUBSET_CHECK' for r in reports),
         'isaac_sim_runtime_tested':False,'openusd_readback_tested':False}
    (package/'reports/offline_validation.json').write_text(json.dumps(out,indent=2))
    if not out['all_offline_checks_pass']:raise SystemExit(1)
