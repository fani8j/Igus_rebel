"""Top-down feasibility diagnostic: is the RIGHT-arm top-down grasp (pre pose 70 mm above, final pose) reachable AND collision free (>= 15 mm, arm links and tool vs cabinet, stocked
neighbours / upper shelves) anywhere in the d / lat / lift / slider grid?  Pose-level only (no transit), so it is a necessary condition independent of the transit model."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import parking_study as ps, parking as pk, robot_kin as rk

def main():
    R, cab = rk.Robot(), rk.Cabinet()
    out = {}
    for tn in (sys.argv[1].split(',') if len(sys.argv) > 1 else list(ps.TARGETS)):
        shelf, box, which = ps.TARGETS[tn]
        tg = cab.pick(shelf, box, which)
        pull = ps.PULL_BY_MODE_SHELF.get(('topdown', shelf), rk.PULL_OUT)
        S = ps.score_grid(R, cab, tg, shelf, 'topdown', pull, 'left_suction')
        sr_map = S['right']
        idxs = np.argwhere(sr_map > 0)
        n_ik = len(idxs)
        okc, why = 0, {}
        best = None
        rng = np.random.default_rng(1)
        if n_ik > 1500:
            idxs = idxs[rng.choice(n_ik, 1500, replace=False)]
        for di, li, zi, si in idxs:
            d, lat, lift = float(ps.D_GRID[di]), float(ps.LAT_GRID[li]), float(ps.LIFT_GRID[zi])
            s = 0.005973 if ps.S_GRID[si] == 0 else float(ps.S_GRID[si])
            W = rk.World(cab, shelf, tg, d, lat)
            sq = pk.build_sequences(W, 'topdown', pull, left_home=True, stage_y=9.0, stage_up=0.0)
            ev = pk.Evaluator(R, W, lift, -0.005973, s)
            bad = set()
            for seg in sq['right'][:3]:      # stage(=pre), stage->pre, descend end
                T = seg.T1
                if T is None:
                    continue
                q = ev.ik_pose('right', T, pk.SEEDS)
                if q is None:
                    bad.add('no IK'); break
                c = ev.clearances('right', q, 0.024, np.zeros(6), 0.024, pull)
                bad.update(ev.check(c))
            if not bad:
                okc += 1
                best = (d, lat, lift, s)
            else:
                for b in bad:
                    why[b] = why.get(b, 0) + 1
        out[tn] = dict(map_cells_with_ik=int(n_ik), tested=int(len(idxs)), pre_pose_clear=okc, reasons=why, example=best)
        print(tn, out[tn], flush=True)
    json.dump(out, open(Path(__file__).resolve().parent.parent / 'tests' / 'results' / 'amr_parking_topdown_diag.json', 'w'), indent=1)
main()
