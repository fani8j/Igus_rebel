#!/usr/bin/env python3
"""Rebuild portable ASCII USD from the supplied STL (Python + numpy/trimesh/scipy/shapely).
No Omniverse or pxr installation is needed for authoring. Runtime tests require Isaac Sim.
Source geometry has no units; metres are inferred by matching the 0.95 x 0.65 body to the supplied image.
"""
from __future__ import annotations
import json, math, hashlib
from pathlib import Path
from dataclasses import dataclass, field
import numpy as np
import trimesh
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from shapely.geometry import Polygon

PACKAGE = Path(__file__).resolve().parent.parent
SOURCE = PACKAGE / 'source' / 'amr_d200.stl'
PARAMS = PACKAGE / 'amr_parameters.json'
R = np.array([[0.,1.,0.],[-1.,0.,0.],[0.,0.,1.]])


def number(x):
    v=float(x)
    if not math.isfinite(v):
        raise ValueError('Non-finite value in asset')
    return '0' if abs(v)<1e-13 else format(v,'.12g')
def vec(a): return '('+', '.join(number(x) for x in a)+')'
def text(s): return json.dumps(s,ensure_ascii=False)
def arr_str(a): return '['+', '.join(text(str(x)) for x in a)+']'
def arr_vec(a): return '['+', '.join(vec(v) for v in a)+']'
def arr_num(a): return '['+', '.join(str(int(x)) for x in a)+']'

@dataclass
class Prim:
    type: str
    name: str
    meta: list[str] = field(default_factory=list)
    props: list[str] = field(default_factory=list)
    children: list['Prim'] = field(default_factory=list)
    def add(self,p): self.children.append(p); return p
    def render(self,indent=0):
        s='    '*indent
        lines=[s+f'def {self.type} {text(self.name)}']
        if self.meta:
            lines+=[s+'(']+[s+'    '+m for m in self.meta]+[s+')']
        lines += [s+'{']
        lines += [s+'    '+p for p in self.props]
        for p in self.children: lines.extend(p.render(indent+1))
        lines += [s+'}']
        return lines

def schemas(*names): return 'prepend apiSchemas = '+arr_str(names)
def xform(p,trans=(0,0,0),scale=None,quat=None):
    ops=[]
    p.props.append('double3 xformOp:translate = '+vec(trans));ops.append('xformOp:translate')
    if quat is not None: p.props.append('quatf xformOp:orient = '+vec(quat));ops.append('xformOp:orient')
    if scale is not None: p.props.append('double3 xformOp:scale = '+vec(scale));ops.append('xformOp:scale')
    p.props.append('uniform token[] xformOpOrder = '+arr_str(ops))

def material(name,color,rough=.65,metal=0.):
    p=Prim('Material',name)
    p.props.append(f'token outputs:surface.connect = <__LOOKS__/{name}/Shader.outputs:surface>')
    sh=p.add(Prim('Shader','Shader'))
    sh.props += ['uniform token info:id = "UsdPreviewSurface"',f'color3f inputs:diffuseColor = {vec(color)}',
                 f'float inputs:roughness = {number(rough)}',f'float inputs:metallic = {number(metal)}','token outputs:surface']
    return p

def physics_material(name,mu_static,mu_dynamic):
    p=Prim('Material',name,[schemas('PhysicsMaterialAPI','PhysxMaterialAPI')])
    p.props += [f'float physics:staticFriction = {number(mu_static)}',f'float physics:dynamicFriction = {number(mu_dynamic)}',
                'float physics:restitution = 0','token physxMaterial:frictionCombineMode = "min"','token physxMaterial:restitutionCombineMode = "min"']
    return p

COLORS={'Body':(.78,.80,.82),'Dark':(.024,.031,.041),'Rubber':(.018,.021,.027),
        'Metal':(.42,.46,.50),'Red':(.62,.035,.026),'LED':(.035,.50,.40),'Floor':(.22,.25,.29)}

def visual_binding(p,mat):
    p.props += [f'rel material:binding = <__LOOKS__/{mat}>',f'color3f[] primvars:displayColor = [{vec(COLORS[mat])}]']

def collider_props(p,mat='TirePhysics',mesh=False,rest=0.):
    names=['PhysicsCollisionAPI','PhysxCollisionAPI','MaterialBindingAPI']
    if mesh: names += ['PhysicsMeshCollisionAPI','PhysxConvexHullCollisionAPI']
    p.meta.append(schemas(*names))
    p.props += ['bool physics:collisionEnabled = true','float physxCollision:contactOffset = 0.001',
                f'float physxCollision:restOffset = {number(rest)}',f'rel material:binding:physics = <__LOOKS__/{mat}>']
    if mesh:
        p.props += ['uniform token physics:approximation = "convexHull"','int physxConvexHullCollision:hullVertexLimit = 128']

def add_extent(p,extent): p.props.append('float3[] extent = '+arr_vec(extent))

def cylinder(name,center,r,height,axis,mat='Rubber',collision=False,hidden=False,rest=0.):
    p=Prim('Cylinder',name)
    p.props += [f'double radius = {number(r)}',f'double height = {number(height)}',f'uniform token axis = "{axis}"']
    ext=np.full(3,r);ext['XYZ'.index(axis)]=height/2
    add_extent(p,[-ext,ext]); xform(p,center)
    if collision: collider_props(p,rest=rest)
    else: p.meta.append(schemas('MaterialBindingAPI'))
    if hidden: p.props += ['token visibility = "invisible"']
    else: visual_binding(p,mat)
    return p

def cube(name,center,size,mat='Metal',collision=False,hidden=False):
    p=Prim('Cube',name)
    p.props += ['double size = 1']; add_extent(p,[[-.5]*3,[.5]*3]); xform(p,center,size)
    if collision: collider_props(p,mat='BodyPhysics')
    else: p.meta.append(schemas('MaterialBindingAPI'))
    if hidden: p.props.append('token visibility = "invisible"')
    else: visual_binding(p,mat)
    return p

def mesh(name,vertices,faces,mat='Body',collision=False,hidden=False):
    p=Prim('Mesh',name)
    p.props += ['uniform token subdivisionScheme = "none"','uniform token orientation = "rightHanded"',
                'bool doubleSided = false','point3f[] points = '+arr_vec(vertices),
                'int[] faceVertexCounts = '+arr_num(np.full(len(faces),3)),
                'int[] faceVertexIndices = '+arr_num(np.asarray(faces).ravel())]
    add_extent(p,[np.min(vertices,0),np.max(vertices,0)])
    if collision: collider_props(p,mat='BodyPhysics',mesh=True)
    else: p.meta.append(schemas('MaterialBindingAPI'))
    if hidden: p.props.append('token visibility = "invisible"')
    else:
        normals=trimesh.Trimesh(vertices=vertices,faces=faces,process=False).face_normals
        p.props.append('normal3f[] normals = '+arr_vec(normals)+' (interpolation = "uniform")')
        visual_binding(p,mat)
    return p

def cylinder_fit(m):
    """Recognize only constant-radius two-ring closed cylinders; never fit arbitrary curves."""
    if len(m.faces)!=124:return None
    for axis in range(3):
        if len(np.unique(np.round(m.vertices[:,axis],6)))!=2:continue
        inds=[j for j in range(3) if j!=axis];v=m.vertices[:,inds]
        # Circle algebraic least squares, with coordinates centred for conditioning.
        q=v-v.mean(0);A=np.c_[2*q,np.ones(len(q))];b=(q*q).sum(1)
        sol=np.linalg.lstsq(A,b,rcond=None)[0];center2=sol[:2]+v.mean(0)
        radii=np.linalg.norm(v-center2,axis=1);radius=float(radii.mean());err=float(abs(radii-radius).max())
        if err>2e-7:continue
        center=m.bounds.mean(0);center[inds]=center2
        return {'axis':'YXZ'[axis],'center':(R@center).tolist(),'radius':radius,'height':float(m.extents[axis]),
                'max_vertex_radial_residual_m':err,
                'source_chord_sagitta_m':radius*(1-math.cos(math.pi/32))}
    return None

def collider_outline(m,origin,tol):
    """Simplify only convex chassis collider outline, not displayed STL geometry."""
    w=m.vertices@R.T
    xy=w[:,:2]
    hull=ConvexHull(xy)
    poly=Polygon(xy[hull.vertices])
    simplified=poly.simplify(tol,preserve_topology=True)
    pts=np.asarray(simplified.exterior.coords[:-1])
    # At most 128 total vertices; original outline curvature already densely sampled.
    if len(pts)>64:raise ValueError('Collider vertex limit exceeded; change tolerance deliberately.')
    lo,hi=w[:,2].min(),w[:,2].max()
    v=np.vstack([np.c_[pts,np.full(len(pts),lo)],np.c_[pts,np.full(len(pts),hi)]])
    out=trimesh.convex.convex_hull(v)
    return out.vertices-origin,out.faces,{
        'vertices':len(out.vertices),'triangles':len(out.faces),
        'outline_hausdorff_m':float(poly.boundary.hausdorff_distance(simplified.boundary)),
        'visual_geometry_changed':False}

def inertia_box(mass,size):
    x,y,z=np.asarray(size);return mass/12*np.array([y*y+z*z,x*x+z*z,x*x+y*y])
def inertia_cylinder(mass,r,h):
    i=mass*(3*r*r+h*h)/12
    return [i,.5*mass*r*r,i]

def rigid(name,pos,mass,inertia,com=(0,0,0),root=False):
    apis=['PhysicsRigidBodyAPI','PhysicsMassAPI','PhysxRigidBodyAPI']
    if root: apis += ['PhysicsArticulationRootAPI','PhysxArticulationAPI']
    p=Prim('Xform',name,[schemas(*apis)])
    xform(p,pos)
    p.props += ['bool physics:rigidBodyEnabled = true','bool physics:kinematicEnabled = false',
                'bool physxRigidBody:disableGravity = false',f'float physics:mass = {number(mass)}',
                'point3f physics:centerOfMass = '+vec(com),'float3 physics:diagonalInertia = '+vec(inertia),
                'quatf physics:principalAxes = (1, 0, 0, 0)',
                'float physxRigidBody:linearDamping = 0.02','float physxRigidBody:angularDamping = 0.02',
                'float physxRigidBody:maxDepenetrationVelocity = 1']
    if root:
        p.props += ['bool physxArticulation:articulationEnabled = true','bool physxArticulation:enabledSelfCollisions = false',
                    'int physxArticulation:solverPositionIterationCount = 32','int physxArticulation:solverVelocityIterationCount = 8']
    return p

def joint(name,b0,b1,p0,axis,drive=None):
    apis=['PhysxJointAPI']
    if drive is not None:apis.append('PhysicsDriveAPI:angular')
    p=Prim('PhysicsRevoluteJoint',name,[schemas(*apis)])
    p.props += [f'rel physics:body0 = <__ROOT__/{b0}>',f'rel physics:body1 = <__ROOT__/{b1}>',
                'point3f physics:localPos0 = '+vec(p0),'point3f physics:localPos1 = (0, 0, 0)',
                'quatf physics:localRot0 = (1, 0, 0, 0)','quatf physics:localRot1 = (1, 0, 0, 0)',
                f'uniform token physics:axis = "{axis}"','bool physics:jointEnabled = true','bool physics:collisionEnabled = false']
    if drive is not None:
        p.props += ['uniform token drive:angular:physics:type = "force"','float drive:angular:physics:stiffness = 0',
                    f'float drive:angular:physics:damping = {number(drive["damping"])}',
                    f'float drive:angular:physics:maxForce = {number(drive["max_torque_nm"])}',
                    f'float drive:angular:physics:targetVelocity = {number(drive.get("target_deg_s",0))}']
    else:p.props += ['float physxJoint:jointFriction = 0.001']
    return p


def build_all():
    params=json.loads(PARAMS.read_text())
    full=trimesh.load_mesh(SOURCE,process=True)
    parts=list(full.split(only_watertight=False))
    if len(parts)!=59:raise ValueError('This builder is for the supplied D200 STL, expected 59 components.')
    fits={i:cylinder_fit(m) for i,m in enumerate(parts)}
    if sum(f is not None for f in fits.values())!=44:raise ValueError('Cylinder recognizer did not match source.')
    base_pos=np.array([0.,0.,fits[34]['center'][2]])
    radius=fits[34]['radius'];track=fits[34]['center'][1]-fits[37]['center'][1]
    source_report={'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'source_faces':len(full.faces),'source_vertices_after_welding':len(full.vertices),'connected_components':len(parts),
        'watertight_components':sum(m.is_watertight for m in parts),'winding_consistent_components':sum(m.is_winding_consistent for m in parts),
        'input_bounds':full.bounds.tolist(),'output_axis_convention':'+X forward (source +Y); +Y left (source -X); +Z up',
        'stl_units':'STL is unitless. Meters inferred from nominal body dimensions.',
        'drive_radius_m':radius,'drive_track_m':track,'caster_radius_m':fits[42]['radius'],
        'screenshot_drive_diameter_m':.2032,'stl_drive_diameter_m':2*radius,
        'screenshot_vs_stl_discrepancy':'8-inch / 101.6-mm clearance in image versus 6-inch / 55-mm lower-body bottom in STL. STL dimensions retained.',
        'native_cylinder_visuals':44,'native_cube_visuals':10,'preserved_mesh_visuals':5,
        'cylinder_max_vertex_residual_m':max(f['max_vertex_radial_residual_m'] for f in fits.values() if f),
        'cylinder_max_source_chord_sagitta_m':max(f['source_chord_sagitta_m'] for f in fits.values() if f),
        'isaac_sim_runtime_tested':False,'openusd_runtime_parse_tested':False,
        'warnings':['Mass, inertia, friction and motor torque are simulation assumptions, not measured hardware values.',
                    'LiDAR geometry and attachment frame only; no working LiDAR sensor, ROS 2 bridge, SLAM or Nav2 configured.',
                    'Source geometry contains overlaps. Articulation self-collisions disabled.',
                    'Cylinder visuals reconstructed from 32-sided rings; true curves differ from source chords.']}
    (PACKAGE/'reports/source_inspection.json').write_text(json.dumps(source_report,indent=2,ensure_ascii=False))

    component_rows=[];collider_rows=[];asset_reports={}
    casters=[('front_left',40,41,42,43),('front_right',44,45,46,47),
             ('rear_left',48,49,50,51),('rear_right',52,53,54,55)]

    def make_robot(mode='articulated',target_v=0.):
        robot=Prim('Xform','AMR_D200',['kind = "component"'])
        xform(robot)
        robot.props += ['custom string amr:assetId = "AMR_D200_STL_6IN_V1"',
                        f'custom string amr:casterMode = "{mode}"',
                        f'custom double amr:wheelRadius = {number(radius)}',f'custom double amr:wheelTrack = {number(track)}',
                        'custom string[] amr:driveJointNames = ["left_wheel_joint", "right_wheel_joint"]',
                        'custom string amr:units = "m, kg, s; USD angular drive target: deg/s"',
                        'custom string amr:dimensionSource = "amr_d200.stl; six-inch tires, NOT the screenshot eight-inch design"',
                        'custom bool amr:runtimeValidated = false']
        looks=robot.add(Prim('Scope','Looks'))
        for name,col in COLORS.items():looks.add(material(name,col,metal=.65 if name=='Metal' else .0))
        looks.add(physics_material('TirePhysics',params['tire_static_friction'],params['tire_dynamic_friction']))
        looks.add(physics_material('BodyPhysics',.5,.4))
        looks.add(physics_material('GroundPhysics',1.,1.))
        looks.add(physics_material('CasterProxyPhysics',0.,0.))
        base_mass=params['base_mass_kg']
        if mode=='navigation':base_mass+=4*params['caster_fork_mass_kg']+8*params['caster_wheel_mass_kg']
        base=robot.add(rigid('base_link',base_pos,base_mass,inertia_box(base_mass,[.95,.65,.195]),
                            [0,0,params['base_com_height_m']-base_pos[2]],True))
        visuals=base.add(Prim('Scope','Visuals'));collisions=base.add(Prim('Scope','Collisions'))
        owners={i:(visuals,base_pos,'base_link') for i in range(59)}
        body_info={'base_link':{'position':base_pos.tolist(),'mass':base_mass}}
        joints=robot.add(Prim('Scope','joints')); joint_info=[]
        drive={**params['motor'],'target_deg_s':math.degrees(target_v/radius)}
        for side,ids in [('left',[34,35,36]),('right',[37,38,39])]:
            c=np.array(fits[ids[0]]['center']);bname=side+'_wheel';mass=params['drive_wheel_mass_kg']
            link=robot.add(rigid(bname,c,mass,inertia_cylinder(mass,radius,fits[ids[0]]['height'])))
            vis=link.add(Prim('Scope','Visuals'))
            for i in ids:owners[i]=(vis,c,bname)
            collision=link.add(Prim('Scope','Collisions'))
            collision.add(cylinder('Tire',(0,0,0),radius,fits[ids[0]]['height'],'Y',collision=True,hidden=True))
            joints.add(joint(side+'_wheel_joint','base_link',bname,c-base_pos,'Y',drive))
            joint_info.append({'name':side+'_wheel_joint','body0':'base_link','body1':bname,'axis':'Y','local_pos0':(c-base_pos).tolist(),'actuated':True})
            body_info[bname]={'position':c.tolist(),'mass':mass}
        for name,mount,bracket,*wheel_ids in casters:
            pivot=np.array(fits[mount]['center'])
            if mode=='articulated':
                linkname=name+'_fork';mass=params['caster_fork_mass_kg']
                link=robot.add(rigid(linkname,pivot,mass,inertia_box(mass,[.056,.056,.046]),[0,0,-.012]))
                vis=link.add(Prim('Scope','Visuals'))
                for i in (mount,bracket):owners[i]=(vis,pivot,linkname)
                # Conservative bracket-only collision; mount disk is inside the chassis.
                box_bounds=(parts[bracket].vertices@R.T)
                link.add(cube('ForkCollision',box_bounds.mean(0)-pivot,box_bounds.max(0)-box_bounds.min(0),collision=True,hidden=True))
                joints.add(joint(name+'_swivel_joint','base_link',linkname,pivot-base_pos,'Z'))
                joint_info.append({'name':name+'_swivel_joint','body0':'base_link','body1':linkname,'axis':'Z','local_pos0':(pivot-base_pos).tolist(),'actuated':False})
                body_info[linkname]={'position':pivot.tolist(),'mass':mass}
                for k,i in enumerate(wheel_ids):
                    c=np.array(fits[i]['center']);wheelname=name+'_wheel_'+str(k+1)
                    mass=params['caster_wheel_mass_kg'];r=fits[i]['radius'];h=fits[i]['height']
                    wl=robot.add(rigid(wheelname,c,mass,inertia_cylinder(mass,r,h)))
                    wv=wl.add(Prim('Scope','Visuals'));owners[i]=(wv,c,wheelname)
                    wl.add(cylinder('TireCollision',(0,0,0),r,h,'Y',collision=True,hidden=True,rest=-.0002))
                    jname=wheelname+'_joint';joints.add(joint(jname,linkname,wheelname,c-pivot,'Y'))
                    joint_info.append({'name':jname,'body0':linkname,'body1':wheelname,'axis':'Y','local_pos0':(c-pivot).tolist(),'actuated':False})
                    body_info[wheelname]={'position':c.tolist(),'mass':mass}
            else:
                # Deliberate navigation proxy: supports are frictionless spheres fixed to base.
                center=(np.array(fits[wheel_ids[0]]['center'])+np.array(fits[wheel_ids[1]]['center']))/2
                s=Prim('Sphere',name+'_SupportProxy')
                s.props += [f'double radius = {number(fits[wheel_ids[0]]["radius"])}','token visibility = "invisible"']
                rr=fits[wheel_ids[0]]['radius'];add_extent(s,[[-rr]*3,[rr]*3]);xform(s,center-base_pos)
                collider_props(s,mat='CasterProxyPhysics',rest=-.0002);collisions.add(s)
        dark={0,7,8,17,18,19,20,21,56,57,58}
        rubber={34,37,42,43,46,47,50,51,54,55}
        body={5,6,40,41,44,45,48,49,52,53}
        for i,m in enumerate(parts):
            parent,origin,owner=owners[i]
            mat='Rubber' if i in rubber else ('Dark' if i in dark else ('Body' if i in body else ('Red' if i in [10,12] else ('LED' if i in [13,14,15,16] else 'Metal'))))
            fit=fits[i]
            if fit:
                geom=cylinder(f'P{i:02d}',np.array(fit['center'])-origin,fit['radius'],fit['height'],fit['axis'],mat)
                kind='Cylinder'
            elif len(m.faces)==12:
                w=m.vertices@R.T
                geom=cube(f'P{i:02d}',w.mean(0)-origin,w.max(0)-w.min(0),mat);kind='Cube'
            else:geom=mesh(f'P{i:02d}',m.vertices@R.T-origin,m.faces,mat);kind='Mesh'
            geom.props += [f'custom int source:componentIndex = {i}',f'custom int source:triangles = {len(m.faces)}']
            if fit:geom.props.append(f'custom double source:circleVertexResidual = {number(fit["max_vertex_radial_residual_m"])}')
            parent.add(geom)
            if mode=='articulated' and target_v==0:
                component_rows.append({'component':i,'usd_type':kind,'body':owner,'source_faces':len(m.faces),**(fit or {})})
        for i in [0,5,6,7,8]:
            vs,fs,info=collider_outline(parts[i],base_pos,params['chassis_collider_outline_tolerance_m'])
            collisions.add(mesh(f'Chassis_{i:02d}',vs,fs,collision=True,hidden=True))
            if mode=='articulated' and target_v==0:collider_rows.append({'component':i,**info})
        sensors=base.add(Prim('Scope','SensorMounts'))
        lidar=sensors.add(Prim('Xform','lidar_mount'));xform(lidar,[.38,0,.224-base_pos[2]])
        lidar.props += ['custom string amr:status = "attachment frame only, no sensor configured"']
        payload=base.add(Prim('Xform','payload_mount'));xform(payload,[0,0,.2515-base_pos[2]])
        payload.props += ['custom string amr:status = "attachment frame only; payload mass not included"']
        asset_reports[mode]={'bodies':body_info,'joints':joint_info,'rigid_body_count':len(body_info),
            'joint_count':len(joint_info),'actuated_joint_count':2,'total_mass_kg':sum(v['mass'] for v in body_info.values())}
        return robot

    def header(default,scene=False):
        s=['#usda 1.0','(','    defaultPrim = '+text(default),'    metersPerUnit = 1',
           '    kilogramsPerUnit = 1','    upAxis = "Z"','    timeCodesPerSecond = 60','    framesPerSecond = 60',
           '    startTimeCode = 0','    endTimeCode = 216000']
        if scene:
            s+=['    customLayerData = {','        dictionary cameraSettings = {',
                '            dictionary Perspective = {','                double3 position = (1.6, -1.7, 1.1)',
                '                double3 target = (0, 0, 0.13)','            }',
                '            string boundCamera = "/World/OverviewCamera"','        }','    }']
        s+=[')'];return s
    def output(robot,filename,scene=False):
        if scene:
            world=Prim('Xform','World');world.add(robot)
            # Robot origin remains nominal floor frame; this is a spawn clearance, not a wheel radius change.
            robot.props[0]='double3 xformOp:translate = (0, 0, 0.003)'
            ph=world.add(Prim('PhysicsScene','PhysicsScene',[schemas('PhysxSceneAPI')]))
            ph.props += ['vector3f physics:gravityDirection = (0, 0, -1)','float physics:gravityMagnitude = 9.81',
                         'uint physxScene:timeStepsPerSecond = 120','bool physxScene:enableGPUDynamics = false',
                         'bool physxScene:enableSceneQuerySupport = true','token physxScene:solverType = "TGS"',
                         'token physxScene:broadphaseType = "MBP"']
            g=world.add(cube('Ground',(0,0,-.05),(40,40,.1),'Floor',True))
            # Ground has no rigid-body API: static collider, with high friction controlled by tire material.
            g.props=[v.replace('<__LOOKS__/BodyPhysics>','<__LOOKS__/GroundPhysics>') for v in g.props]
            light=world.add(Prim('DomeLight','EnvironmentLight'));light.props += ['float inputs:intensity = 700','color3f inputs:color = (1, 1, 1)']
            sun=world.add(Prim('DistantLight','KeyLight'));sun.props += ['float inputs:intensity = 1800','float inputs:angle = 1']
            xform(sun,quat=[.880476,.279848,.364705,-.115917])
            cam=world.add(Prim('Camera','OverviewCamera'))
            c=np.array([1.6,-1.7,1.1]);t=np.array([0,0,.13]);back=c-t;back/=np.linalg.norm(back)
            right=np.cross([0,0,1],back);right/=np.linalg.norm(right);up=np.cross(back,right)
            q=Rotation.from_matrix(np.column_stack([right,up,back])).as_quat()
            xform(cam,c,quat=[q[3],*q[:3]])
            cam.props += ['token projection = "perspective"','float focalLength = 28','float2 clippingRange = (0.01, 1000)']
            root='/World/AMR_D200';lines=header('World',True)+world.render()
        else:root='/AMR_D200';lines=header('AMR_D200')+robot.render()
        s='\n'.join(lines)+'\n';s=s.replace('__LOOKS__',root+'/Looks').replace('__ROOT__',root)
        (PACKAGE/filename).write_text(s,encoding='utf-8')
    output(make_robot(),'AMR_D200.usd')
    output(make_robot(target_v=.2),'AMR_D200_drive_demo.usd',True)
    output(make_robot(mode='navigation'),'AMR_D200_navigation.usd')
    output(make_robot(mode='navigation',target_v=.2),'AMR_D200_navigation_demo.usd',True)
    (PACKAGE/'reports/components.json').write_text(json.dumps(component_rows,indent=2,ensure_ascii=False))
    (PACKAGE/'reports/collider_approximation.json').write_text(json.dumps(collider_rows,indent=2))
    (PACKAGE/'reports/articulation_structure.json').write_text(json.dumps(asset_reports,indent=2))
    print(json.dumps({'cylinders':44,'bodies':{k:v['rigid_body_count'] for k,v in asset_reports.items()},
        'max_collider_outline_error_m':max(r['outline_hausdorff_m'] for r in collider_rows)},indent=2))

if __name__=='__main__':build_all()
