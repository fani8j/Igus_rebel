"""IK reach maps of the two arms for the three fixed tool orientations of the pick sequences (left suction, right gripper front, right gripper top-down), in the arm base frame.
Grid: x 0.10..0.90, y -0.60..0.60, z -0.90..0.60 m at 20 mm; a cell is True when KDL IK (37 seeds, joint limits) finds a solution for the TCP at that position.
Saved to data/reach_maps.npz with the interior depth (m) = distance to the nearest infeasible cell.  Run: nice -n 19 python3 tools/reach_maps.py  (4 worker processes)"""
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import parking as pk  # noqa: E402
import robot_kin as rk  # noqa: E402

STEP = 0.02
X0, Y0, Z0 = 0.10, -0.60, -0.90
NX, NY, NZ = 41, 61, 76
ORI = {'left_suction': ('left', pk.R_SUCTION), 'right_front': ('right', pk.R_GRIP_FR), 'right_topdown': ('right', pk.R_GRIP_TD),
       'left_suction_180': ('left', pk.R_SUCTION @ np.diag([-1.0, -1.0, 1.0]))}      # suction tool turned 180 deg about the pad axis (E2 extra variant)
_R = None


def _init():
    global _R
    os.nice(19)
    _R = rk.Robot()


def _work(args):
    name, cells = args
    arm, Rc = ORI[name]
    Rw = rk.Rz(-math.pi / 2) @ Rc
    out = []
    for (i, j, k) in cells:
        p = np.array([X0 + i * STEP, Y0 + j * STEP, Z0 + k * STEP])
        T = np.eye(4)
        T[:3, :3] = Rw
        T[:3, 3] = p
        out.append(_R.ik_multi(arm, T, pk.SEEDS) is not None)
    return out


def main():
    t0 = time.time()
    res = {}
    only = sys.argv[1:]
    names = only or [n for n in ORI if n != 'left_suction_180']
    sh = np.array([0.0, 0.0, 0.25])             # shoulder (joint 2) above the arm base
    cells = [(i, j, k) for i in range(NX) for j in range(NY) for k in range(NZ)
             if np.linalg.norm(np.array([X0 + i * STEP, Y0 + j * STEP, Z0 + k * STEP]) - sh) <= 0.85]
    print('cells in the reach sphere:', len(cells), flush=True)
    chunks = [cells[i:i + 400] for i in range(0, len(cells), 400)]
    with mp.Pool(2 if only else 4, initializer=_init) as pool:
        for name in names:
            m = np.zeros((NX, NY, NZ), bool)
            outs = pool.map(_work, [(name, c) for c in chunks])
            for c, o in zip(chunks, outs):
                for (i, j, k), v in zip(c, o):
                    m[i, j, k] = v
            res[name] = m
            print(name, 'feasible cells', int(m.sum()), 'of', len(cells), round(time.time() - t0), 's', flush=True)
    np.savez_compressed(HERE.parent / 'data' / ('reach_maps_' + '_'.join(only) + '.npz' if only else 'reach_maps.npz'), origin=np.array([X0, Y0, Z0]), step=STEP,
                        **{k: v for k, v in res.items()}, **{k + '_depth': distance_transform_edt(np.pad(v, 1), sampling=STEP)[1:-1, 1:-1, 1:-1] for k, v in res.items()})
    print('saved', round(time.time() - t0), 's')


if __name__ == '__main__':
    main()
