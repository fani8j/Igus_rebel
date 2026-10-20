"""Read-only bridge to v3 (../suction_tool) and ../common.  Nothing in those folders is modified or copied; v3 helpers are imported by path.
Module names bound here (v3 'params', 'lib', 'parts', 'massprops', 'meshlib', 'render') are NOT used as file names in this folder."""
import os, sys, importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
V3 = os.path.join(ROOT, 'suction_tool')
COMMON = os.path.join(ROOT, 'common')
SCRATCH = '/tmp/claude-1000/-home-farhan-orca-workspaces-igus-rebel-ros2-box-3d/8cf63b70-021b-42b6-8d55-cd06a7482230/scratchpad'
os.makedirs(SCRATCH, exist_ok=True)

if V3 not in sys.path:
    sys.path.insert(0, V3)
import params as P3          # v3 tool params (also loads common/params.py as P3.C)
import lib as L              # v3 CadQuery helpers
import parts as PA3          # v3 parts (pad_body, nut, fillet_path, gemini_hull, ...)
import massprops as MP3      # v3 mass / CoG helper
import meshlib as ML         # cq -> trimesh
C = P3.C


def load_v3_render():
    spec = importlib.util.spec_from_file_location('v3_render', os.path.join(V3, 'render.py'))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def load_handoff():
    """common/handoff.py does `from params import BOXES, PULL_OUT` (common/params.py); v3 already owns the name 'params', so swap it temporarily"""
    saved = sys.modules.pop('params', None)
    sys.path.insert(0, COMMON)
    try:
        spec = importlib.util.spec_from_file_location('common_handoff', os.path.join(COMMON, 'handoff.py'))
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    finally:
        sys.path.remove(COMMON)
        sys.modules.pop('params', None)
        if saved is not None:
            sys.modules['params'] = saved
    return m
