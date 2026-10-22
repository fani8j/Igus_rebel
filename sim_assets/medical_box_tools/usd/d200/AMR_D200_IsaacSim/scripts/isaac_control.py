"""Native USD wheel-drive control for an already running Isaac Sim application.
No root transform animation, teleportation, or base velocity forcing is used.
USD angular drive targetVelocity is deg/s, not the rad/s used by ArticulationAction.
"""
from __future__ import annotations
import math
from pathlib import Path
import json
from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf
import carb.settings
from drive_math import twist_to_wheels,slew

def configure_physics() -> None:
    settings=carb.settings.get_settings()
    # Old and new Kit physics names. Unused settings in another Kit version are harmless.
    settings.set_bool('/physics/collisionCylinderCustomGeometry',True)
    settings.set_bool('/physics/collisionApproximateCylinders',False)
    settings.set_bool('/physics/updateToUsd',True)
    # Leave renderer settings unchanged. This is a small CPU physics test asset.

def find_robot_root(stage, requested: str | None=None) -> str:
    if requested:
        prim=stage.GetPrimAtPath(requested)
        if not prim or not prim.GetAttribute('amr:wheelRadius'):
            raise ValueError('Not an AMR D200 root: '+requested)
        return str(prim.GetPath())
    found=[str(p.GetPath()) for p in stage.Traverse() if p.GetAttribute('amr:assetId').Get()=='AMR_D200_STL_6IN_V1']
    if len(found)!=1:raise RuntimeError('Expected one D200 asset. Specify robot_root explicitly. Found: '+str(found))
    return found[0]

class D200Drive:
    def __init__(self,stage,robot_root: str | None=None):
        self.stage=stage;self.root=find_robot_root(stage,robot_root)
        root=stage.GetPrimAtPath(self.root)
        self.radius=float(root.GetAttribute('amr:wheelRadius').Get())
        self.track=float(root.GetAttribute('amr:wheelTrack').Get())
        self.targets=[]
        for name in ('left_wheel_joint','right_wheel_joint'):
            prim=stage.GetPrimAtPath(self.root+'/joints/'+name)
            drive=UsdPhysics.DriveAPI(prim,'angular')
            if not drive:raise RuntimeError('Missing angular drive: '+str(prim.GetPath()))
            self.targets.append(drive.GetTargetVelocityAttr())
        path=Path(__file__).resolve().parent.parent/'amr_parameters.json'
        self.limits=json.loads(path.read_text())['command_limits']
        self.v=0.;self.w=0.;self.last_wheel_rad_s=(0.,0.)
        self.stop()
    def _write(self,left: float,right: float) -> None:
        # Session-only overrides: never save a moving state into the supplied asset file.
        with Usd.EditContext(self.stage,self.stage.GetSessionLayer()):
            with Sdf.ChangeBlock():
                self.targets[0].Set(float(math.degrees(left)))
                self.targets[1].Set(float(math.degrees(right)))
        self.last_wheel_rad_s=(left,right)
    def set_twist(self,v: float,yaw_rate: float,dt: float=1/60,immediate: bool=False) -> None:
        if not self.stage.GetPrimAtPath(self.root):raise RuntimeError('AMR was removed from the stage')
        if not all(math.isfinite(float(x)) for x in (v,yaw_rate,dt)):raise ValueError('Nonfinite command')
        vmax=self.limits['linear_m_s'];wmax=self.limits['angular_rad_s']
        v=max(-vmax,min(vmax,v));yaw_rate=max(-wmax,min(wmax,yaw_rate))
        dt=max(0.,min(.1,dt))
        if immediate:self.v,self.w=v,yaw_rate
        else:
            self.v=slew(self.v,v,self.limits['linear_accel_m_s2'],dt)
            self.w=slew(self.w,yaw_rate,self.limits['angular_accel_rad_s2'],dt)
        self._write(*twist_to_wheels(self.v,self.w,self.radius,self.track))
    def stop(self) -> None:
        self.v=self.w=0.
        self._write(0.,0.)
    def pose(self) -> dict:
        prim=self.stage.GetPrimAtPath(self.root+'/base_link')
        m=UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim)
        p=m.ExtractTranslation()
        forward=m.TransformDir(Gf.Vec3d(1.,0.,0.));up=m.TransformDir(Gf.Vec3d(0.,0.,1.))
        return {'xyz_m':[float(x) for x in p],
                'yaw_rad':math.atan2(float(forward[1]),float(forward[0])),
                'up_z':float(up[2])}

class KeyboardCommand:
    """Poll the actual key state, rather than accumulating sticky key-down events."""
    def __init__(self):
        import carb.input
        import omni.appwindow
        self.keys=carb.input.KeyboardInput
        self.input=carb.input.acquire_input_interface()
        window=omni.appwindow.get_default_app_window()
        if window is None:raise RuntimeError('Keyboard control requires an Isaac Sim GUI window')
        self.keyboard=window.get_keyboard()
    def pressed(self,key) -> bool:
        return self.input.get_keyboard_value(self.keyboard,key)>0.5
    def read(self) -> tuple[float,float,bool,bool]:
        k=self.keys
        v=.30*(int(self.pressed(k.W) or self.pressed(k.UP))-int(self.pressed(k.S) or self.pressed(k.DOWN)))
        w=.70*(int(self.pressed(k.A) or self.pressed(k.LEFT))-int(self.pressed(k.D) or self.pressed(k.RIGHT)))
        return v,w,self.pressed(k.SPACE),self.pressed(k.ESCAPE)
