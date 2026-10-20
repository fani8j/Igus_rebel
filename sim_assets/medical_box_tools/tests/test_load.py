"""Headless Isaac Sim 5.1: open every USD of this folder (and optionally others) with the Surface Gripper extension enabled, step physics for 1 s, close.
Prints '### LOAD <file>' / '### END <file>' markers; tests/run_all.sh greps the Kit log lines between them for schema / physics warnings and errors.
Run:  nice -n 19 /home/farhan/isaac-sim/python.sh tests/test_load.py [file.usd ...]   (names relative to usd/, or absolute paths)
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from isaac_common import *            # noqa: E402,F401

FILES = ['left_suction_tool.usda', 'right_gripper_tool.usda', 'right_gripper_tool_independent.usda', 'shelf_handoff_scene.usda',
         'pick_test_scene.usda', 'shelf_cartons.usda', 'rebel_left_suction.usda', 'rebel_right_gripper.usda', 'dual_arm_rig_medical.usda',
         'd200_dual_arm_medical.usda', 'd200_rig_test_scene.usda', 'cabinet_grasp_test_scene.usda']


def main():
    app = start_app()
    apply_d200_physics_settings()
    import omni.usd
    from pxr import UsdPhysics
    from isaacsim.core.api import SimulationContext
    files = sys.argv[1:] or FILES
    for f in files:
        print(f'### LOAD {f}')
        ctx = omni.usd.get_context()
        ctx.open_stage(f if os.path.isabs(f) else str(USD / f))
        for _ in range(10):
            app.update()
        stage = ctx.get_stage()
        if not any(p.IsA(UsdPhysics.Scene) for p in stage.Traverse()):          # tools alone: add a scene and weld the base to the world (stand-alone use)
            UsdPhysics.Scene.Define(stage, '/PhysicsScene')
            for a in stage.Traverse():
                if a.GetName() == 'world_anchor':
                    a.GetAttribute('physics:jointEnabled').Set(True)
        sim = SimulationContext(physics_dt=1 / 240.0, rendering_dt=1 / 60.0, stage_units_in_meters=1.0)
        sim.initialize_physics()
        sim.play()
        for _ in range(240):
            sim.step(render=False)
        sim.stop()
        n_prims = sum(1 for _ in stage.Traverse())
        print(f'### END {f}  prims={n_prims}')
    sys.stdout.flush()
    app.close()
    os._exit(0)


if __name__ == '__main__':
    main()
