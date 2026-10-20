"""Headless viewport captures (RTX, 1280x720) of the pick tests and the rig: tests/results/*.png.   Run: nice -n 19 /home/farhan/isaac-sim/python.sh tests/capture_scenes.py"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tools'))
from isaac_common import *            # noqa: E402,F401
import numpy as np                    # noqa: E402


def shot(app, sim, name, eye, target):
    from isaacsim.core.utils.viewports import set_camera_view
    from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
    set_camera_view(eye=np.array(eye), target=np.array(target), camera_prim_path='/OmniverseKit_Persp')
    sim.pause()
    for _ in range(40):
        app.update()
    vp = get_active_viewport()
    capture_viewport_to_file(vp, str(RESULTS / name))
    for _ in range(40):
        app.update()
    print('captured', name, flush=True)
    sim.play()


def main():
    app = start_app(extra_exts=('isaacsim.robot.surface_gripper', 'omni.kit.viewport.utility'))
    import test_left_suction as TL
    import test_right_gripper as TR
    from pxr import UsdLux, Sdf
    TL.Sdf = Sdf
    stage, sim = open_scene(app, 'pick_test_scene.usda', 1 / 240.0)
    if not any(p.IsA(UsdLux.DomeLight) for p in stage.Traverse()):
        UsdLux.DomeLight.Define(stage, '/World/Dome').CreateIntensityAttr(1500.0)
    iface = TL.tool_api.suction_interface()
    # --- left tool: approach, press, grip, pull 100 mm
    run = TL.Run(app, stage, sim, iface)
    run.reset()
    shot(app, sim, 'left_suction_start.png', (0.45, -0.55, 0.40), (0.0, 0.0, 0.12))
    run.ramp(TL.S_PRESS, TL.APPROACH_SPEED)
    run.step(100)
    TL.tool_api.suction_close(iface, TL.SG)
    run.step(200)
    shot(app, sim, 'left_suction_pressed_gripped.png', (0.35, -0.45, 0.30), (0.0, 0.0, 0.12))
    run.ramp(TL.S_PRESS - 0.105, TL.PULL_SPEED)
    run.step(100)
    shot(app, sim, 'left_suction_pulled_100mm.png', (0.55, -0.40, 0.35), (0.0, -0.05, 0.12))
    # --- right gripper: grip and lift 50 mm
    for p in ('/World/Tools/left_suction_tool', '/World/Tools/left_suction_tool_carrier'):
        stage.GetPrimAtPath(p).SetActive(False)
    g = TR.G(stage, sim)
    path, t, h = TR.CARTONS['16 mm (box 3, idx 2)']
    g.reset(path, h, 0.0)
    g.ramp(0.0, 0.04)
    g.step(60)
    for q in TR.tool_api.finger_target_profile(0.024, TR.tool_api.GRIP_CLOSE_Q, 1 / 240.0, push=True):
        g.fingers(q)
        g.step(1)
    g.step(60)
    g.ramp(-0.05, 0.025)
    g.step(60)
    shot(app, sim, 'right_gripper_lift_50mm.png', (0.30, -0.40, 0.33), (0.0, -0.06, 0.18))
    sys.stdout.flush()
    # --- rig
    sim.stop()
    stage, sim = open_scene(app, 'dual_arm_rig_medical.usda', 1 / 240.0)
    from pxr import UsdLux as L2
    if not any(p.IsA(L2.DomeLight) for p in stage.Traverse()):
        L2.DomeLight.Define(stage, '/World/Dome').CreateIntensityAttr(1500.0)
    sim.initialize_physics()
    sim.play()
    for _ in range(int(3.0 / (1 / 240.0))):
        sim.step(render=False)
    shot(app, sim, 'dual_arm_rig_medical.png', (2.6, -2.2, 2.3), (0.1, -0.35, 1.25))
    os._exit(0)


if __name__ == '__main__':
    main()
