#!/usr/bin/env python3
"""Collision-pair sampler (own implementation of the Setup-Assistant 'default collision matrix' idea) using the running planning-only stack.

Samples N uniform random dual-arm states inside the real joint limits, asks /check_state_validity for ALL contacts, and counts per link pair how
often it collides.  Pairs that never collide (0 of N) are written as 'Never', pairs that collide in >= 95 % of states as 'Default' by
gen_cell.py --matrix.  Run it against a stack that was generated WITHOUT --matrix (so no pair is hidden), then regenerate with --matrix.

  export ROS_DOMAIN_ID=187 ROS_LOCALHOST_ONLY=1
  python3 sample_collisions.py --build build/<cell> --n 30000 --out build/<cell>/collision_matrix.json [--extra_states states.json]
extra_states.json: list of {"arm1":[6],"arm2":[6]} (e.g. the study joint paths) sampled in addition to the uniform ones.
"""
import argparse, json, math, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import moveit_client as MC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--build', required=True)
    ap.add_argument('--n', type=int, default=30000)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=12345)
    ap.add_argument('--extra_states', default=None)
    ap.add_argument('--save_valid', action='store_true', help='store the per-state validity bit string (for matrix-equivalence checks)')
    a = ap.parse_args()
    b = a.build if os.path.isabs(a.build) else os.path.join(HERE, a.build)
    meta = json.load(open(os.path.join(b, 'cell_meta.json')))
    MC.init()
    cl = MC.MoveItClient(meta, 'sample_collisions')
    rng = np.random.default_rng(a.seed)
    lo = np.array([-178, -79, -79, -178, -94, -178.0]) * math.pi / 180
    hi = np.array([178, 139, 139, 178, 94, 178.0]) * math.pi / 180
    counts, n, ninv = {}, 0, 0
    bits = []
    t0 = time.time()
    states = [{'arm1': list(rng.uniform(lo, hi)), 'arm2': list(rng.uniform(lo, hi))} for _ in range(a.n)]
    nextra = 0
    if a.extra_states:
        ex = json.load(open(a.extra_states))
        states += ex
        nextra = len(ex)
    for q in states:
        v, c = cl.validity(q)
        n += 1
        ninv += (not v)
        bits.append('1' if v else '0')
        for p in c:
            k = '|'.join(sorted(p))
            counts[k] = counts.get(k, 0) + 1
        if n % 5000 == 0:
            print('%d samples, %d distinct pairs, %.0f s' % (n, len(counts), time.time() - t0), flush=True)
    json.dump(dict(n_samples=n, n_uniform=a.n, n_extra=nextra, n_in_collision=ninv, seed=a.seed, counts=counts, seconds=time.time() - t0, valid_bits=''.join(bits) if a.save_valid else None), open(a.out, 'w'), indent=1)
    print('done: %d samples, %d in collision (%.1f%%), %d distinct colliding pairs' % (n, ninv, 100.0 * ninv / n, len(counts)))


if __name__ == '__main__':
    main()
