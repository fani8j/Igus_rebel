"""Run this file inside the Isaac Sim Script Editor AFTER opening a D200 demo USD.
Paste/execute: exec(open('/absolute/path/AMR_D200_IsaacSim/scripts/teleop_in_isaac.py').read(), {'__file__': '/absolute/path/AMR_D200_IsaacSim/scripts/teleop_in_isaac.py'})
Keyboard: W/S forward/reverse; A/D turn; release to decelerate; SPACE zero command; ESC disable.
This script does not replace your scene or auto-start the timeline.
"""
from pathlib import Path
import sys
import builtins
import omni.kit.app
import omni.timeline
import omni.usd

if '__file__' not in globals():
    raise RuntimeError('Use Script Editor Open File/Run, or the exec(..., {"__file__": path}) command in README.')
sys.path.insert(0,str(Path(__file__).resolve().parent))
from isaac_control import D200Drive,KeyboardCommand,configure_physics

ROBOT_ROOT=None  # Set e.g. '/World/AMR_D200' when more than one D200 exists.
previous=getattr(builtins,'AMR_D200_TELEOP',None)
if previous is not None:previous.close()

class D200Teleop:
    def __init__(self):
        configure_physics()
        self.stage=omni.usd.get_context().get_stage()
        if self.stage is None:raise RuntimeError('Open AMR_D200_drive_demo.usd first')
        self.drive=D200Drive(self.stage,ROBOT_ROOT);self.keyboard=KeyboardCommand()
        self.timeline=omni.timeline.get_timeline_interface();self.closed=False
        self.subscription=omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(self.update,name='AMR D200 wheel teleop')
        try:
            from omni.kit.viewport.utility import get_active_viewport
            vp=get_active_viewport()
            if vp and self.stage.GetPrimAtPath('/World/OverviewCamera'):vp.set_active_camera('/World/OverviewCamera')
        except (ImportError,AttributeError):pass
        print('[D200] Ready. Click the viewport and press Play. W/S A/D; SPACE = stop; ESC = disable.')
    def update(self,event):
        if self.closed:return
        try:
            if omni.usd.get_context().get_stage()!=self.stage:
                self.close();return
            if not self.timeline.is_playing():
                self.drive.stop();return
            v,w,stop,exit_=self.keyboard.read()
            if exit_:self.close();return
            if stop:self.drive.stop();return
            dt=float(event.payload.get('dt',1/60))
            self.drive.set_twist(v,w,dt)
        except Exception as exc:
            print('[D200] Teleop stopped:',exc);self.close()
    def close(self):
        self.closed=True
        try:self.drive.stop()
        except Exception:pass
        self.subscription=None
        print('[D200] Teleop disabled; wheel targets set to zero.')

builtins.AMR_D200_TELEOP=D200Teleop()
