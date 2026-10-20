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


class GuiViewer:
    """--gui: shows a test live in the Isaac Sim window.  Physics is still advanced only by sim.step(render=False); tick() just redraws with
    sim.render() (no physics step) every `every` steps and sleeps so playback runs at `speed` x real time (it cannot run faster than the physics)."""

    def __init__(self, app, sim, speed=1.0, every=8, dt=1 / 240.0):
        self.app, self.sim, self.speed, self.every, self.dt = app, sim, max(speed, 1e-3), every, dt
        self.reset()

    def reset(self):
        self.t0, self.n0 = None, 0

    def tick(self, step_i):
        if step_i % self.every:
            return
        if not self.app.is_running():                      # window closed mid-run: stop here (results of the unfinished target are not written)
            print('[gui] window closed, exiting', flush=True)
            os._exit(0)
        if self.t0 is None:
            self.t0, self.n0 = time.monotonic(), step_i
        ahead = (step_i - self.n0) * self.dt / self.speed - (time.monotonic() - self.t0)
        if ahead > 0:
            time.sleep(ahead)
        self.sim.render()

    def caption(self, txt):
        print('[gui]', txt)

    def look(self, eye, target):
        from isaacsim.core.utils.viewports import set_camera_view
        set_camera_view(eye=[float(v) for v in eye], target=[float(v) for v in target])
        self.sim.render()

    def hold(self, seconds=None):
        """keep drawing the current pose (no physics) for `seconds`, or until the window is closed when None"""
        t_end = None if seconds is None else time.monotonic() + seconds
        while self.app.is_running() and (t_end is None or time.monotonic() < t_end):
            self.sim.render()
            time.sleep(1 / 60)


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
