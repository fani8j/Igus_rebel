"""Shared helpers for the headless Isaac Sim 5.1 tests (run with /home/farhan/isaac-sim/python.sh).

Process rules: ROS_DOMAIN_ID=170, ROS_LOCALHOST_ONLY=1 are forced here; no ROS 2 bridge extension is enabled.
CPU physics is used (the Surface Gripper extension needs it).
"""
import functools
import os
import sys
import time
from pathlib import Path

os.environ['ROS_DOMAIN_ID'] = '170'
os.environ['ROS_LOCALHOST_ONLY'] = '1'
print = functools.partial(print, flush=True)       # Kit exits without flushing stdout

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent
USD = TOOLS / 'usd'
RESULTS = HERE / 'results'
RESULTS.mkdir(exist_ok=True)


def start_app(headless=True, extra_exts=('isaacsim.robot.surface_gripper',)):
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': headless, 'renderer': 'RayTracedLighting'})
    import omni.kit.app
    em = omni.kit.app.get_app().get_extension_manager()
    for e in extra_exts:
        em.set_extension_enabled_immediate(e, True)
    return app


class Checks:
    """Collects PASS/FAIL lines with numbers; exit code 1 if any FAIL."""

    def __init__(self, name):
        self.name = name
        self.rows = []

    def check(self, label, ok, value=''):
        self.rows.append((label, bool(ok), str(value)))
        print(f"[{'PASS' if ok else 'FAIL'}] {label}  {value}")
        return ok

    def summary(self):
        n = sum(1 for r in self.rows if r[1])
        print(f'=== {self.name}: {n}/{len(self.rows)} checks passed ===')
        return n == len(self.rows)


# ------------------------------------------------------------------ stage / sim helpers (call after start_app)
def open_scene(app, usd_name, dt=1 / 240.0):
    """Open a USD from usd/, create the SimulationContext (physics only, no render), return (stage, sim)."""
    import omni.usd
    ctx = omni.usd.get_context()
    ctx.open_stage(str(USD / usd_name))
    for _ in range(10):
        app.update()
    stage = ctx.get_stage()
    from isaacsim.core.api import SimulationContext
    sim = SimulationContext(physics_dt=dt, rendering_dt=dt, stage_units_in_meters=1.0)
    return stage, sim


def pose_of(stage, path):
    """World translation (np.array) and 3x3 rotation of a prim from the USD (updated by PhysX during the simulation)."""
    import numpy as np
    import omni.usd
    m = omni.usd.get_world_transform_matrix(stage.GetPrimAtPath(path))
    t = m.ExtractTranslation()
    R = np.array(m.ExtractRotationMatrix()).T          # Gf is row-vector convention
    return np.array([t[0], t[1], t[2]]), R


def apply_d200_physics_settings():
    """carb settings the D200 AMR needs for its cylinder tyres (AMR README); call before the stage is simulated."""
    import carb.settings
    st = carb.settings.get_settings()
    st.set_bool('/physics/collisionCylinderCustomGeometry', True)
    st.set_bool('/physics/collisionApproximateCylinders', False)
    st.set_bool('/physics/updateToUsd', True)
