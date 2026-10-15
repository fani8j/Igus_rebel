#!/usr/bin/env python3
"""Move the arm from a start pose that sits inside the camera-head/pole SAFETY PADDING (e.g. the zero/home pose,
straight up next to the scene camera) to the nearest reachable first calibration view, so that wrist_handeye.py can
start from a clear pose.

- The goal is chosen with the FULL safety scene (slab, board stand, pole/head pads +50 mm): IK with collision checks.
- Only for planning/executing this one move are whe_pole_pad / whe_head_pad lifted (the start state is inside them);
  the real camera_pole / camera_head stay in the scene, the table slab and board stand stay applied.
- Same plan checks as wrist_handeye.py (first point, goal, detour, joint6, waypoint collision checks).
- Plan-only by default; --execute moves (real hardware: --real + typed MOVE + Enter). Ctrl-C / q cancels at once.
- All whe_* objects are removed at exit.

    python3 leave_home.py                    # plan only
    python3 leave_home.py --execute --real   # move (real)
"""
import argparse
import os
import signal
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import handeye_math as hm  # noqa: E402
import wrist_handeye as wh  # noqa: E402

PADS = ("whe_pole_pad", "whe_head_pad")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--scale", type=float, default=0.10)
    ap.add_argument("--group", default="igus_rebel_arm")
    ap.add_argument("--targets", default=wh.DEFAULT_TARGETS)
    ap.add_argument("--board-pose", type=float, nargs=7, help="mock only")
    ap.add_argument("--slab-thickness", type=float, default=0.30)
    ap.add_argument("--stand-extra", type=float, default=0.03)
    args = ap.parse_args(argv)
    if not 0.01 <= args.scale <= 0.25:
        raise SystemExit("--scale must be within 0.01..0.25")
    wh.require_domain()
    import ros_io
    from ros_io import UserStop
    from rebel_dual_arm_calibration import targets

    spec = targets.load_table_board(args.targets)
    det = targets.TargetDetector({"table": spec})
    bw, bh = spec.board_size()
    io = ros_io.IO(args.group)
    signal.signal(signal.SIGINT, lambda s, f: io.request_stop("SIGINT"))
    signal.signal(signal.SIGTERM, lambda s, f: io.request_stop("SIGTERM"))
    rc = 1
    try:
        comps = io.hardware_components()
        if not comps:
            raise SystemExit("no hardware component reported - refusing")
        is_mock = all(c == "mock_components/GenericSystem" for _, c, _ in comps)
        wh.info("hardware: %s -> %s" % (comps, "MOCK" if is_mock else "REAL"))
        if args.real and is_mock:
            raise SystemExit("REFUSED: --real on mock hardware")
        if args.execute and not is_mock and not args.real:
            raise SystemExit("REFUSED: real hardware needs --real")
        if args.board_pose and not is_mock:
            raise SystemExit("REFUSED: --board-pose is mock only")
        if args.board_pose:
            T_bb = hm.T_from_xyz_quat(*args.board_pose)
        else:
            T_bb, spread, nb = wh.locate_board_scene(io, spec, det)
            wh.info("board from the scene camera: %d frames, spread %.2f mm" % (nb, spread["pos_rms_mm"]))
        corners = wh.board_report(T_bb, bw, bh)
        io.wait_standstill(1.0, 0.002, timeout=5.0)
        q0 = io.wait_joint_states()
        wh.info("start joints (deg): %s" % np.round(np.degrees(q0), 1).tolist())
        ok0, c0 = io.state_validity(q0)
        if not ok0:
            raise RuntimeError("start state collides even WITHOUT the safety pads: %s - refusing" % c0)
        wh.apply_safety_scene(io, corners, args)
        ok1, c1 = io.state_validity(q0)
        if ok1:
            wh.info("start pose is already clear of the safety scene: nothing to do, run wrist_handeye.py")
            rc = 0
            return rc
        bad = [p for p in c1 if not any(x in PADS for x in p)]
        if bad:
            raise RuntimeError("start state collides with more than the pole/head pads: %s - refusing" % bad)
        wh.info("start pose is inside the pole/head padding only: %s" % c1)

        # goal: nearest reachable view, IK against the FULL safety scene, seeded from a forward-leaning posture
        X = hm.nominal_T_link6_optical()
        seeds = [np.radians([0, 45, 60, 0, 60, 0]), np.radians([-30, 45, 60, 0, 60, 0])]
        best = None
        found = 0
        cands = hm.orbit_candidates(T_bb, tilts=(15, 25, 35), rolls=(0,), dists=(0.36, 0.42))
        wh.info("searching the first view: %d candidates x %d seeds (stops after 6 solutions; up to ~1 min) ..."
                % (len(cands), len(seeds)))
        for i, c in enumerate(cands):
            io.check_stop()
            if found >= 6:
                break
            T6 = c.T_base_cam @ hm.inv_T(X)
            for s in seeds:
                q = io.compute_ik(T6, s, avoid_collisions=True)
                if q is None or abs(q[5] - q0[5]) > 3.14 or np.max(np.abs(q - q0)) > np.radians(120):
                    continue
                found += 1
                d = float(np.max(np.abs(q - q0)))
                if best is None or d < best[0]:
                    best = (d, q, c)
            if (i + 1) % 8 == 0:
                wh.info("  %d/%d candidates checked, %d solutions" % (i + 1, len(cands), found))
        if best is None:
            raise RuntimeError("no reachable first view found from this start pose - jog the arm forward manually")
        _, qg, cg = best
        wh.info("goal: view tilt %.0f az %.0f dist %.2f ; joints (deg) %s" % (cg.tilt, cg.azimuth, cg.dist, np.round(np.degrees(qg), 1).tolist()))
        okg, cgc = io.state_validity(qg)
        if not okg:
            raise RuntimeError("goal state invalid in the full safety scene: %s" % cgc)

        # lift ONLY the pole/head pads for this move (real pole/head stay)
        io.remove_objects(list(PADS))
        times, pos, jt, _ = io.plan(q0, qg, args.scale)
        cf, _ = io.path_collision_free(pos)
        prob = wh.check_plan(io.wait_joint_states(), qg, pos, q0, cf)
        if prob:
            raise RuntimeError("plan rejected: %s" % "; ".join(prob))
        wh.info("plan: %.1f s, max joint change %.1f deg, %d points, collision-free (real pole/head, slab, stand)"
                % (times[-1], np.degrees(np.max(np.abs(qg - q0))), len(pos)))
        if not args.execute:
            wh.info("PLAN ONLY - nothing moved. Re-run with --execute%s to move." % ("" if is_mock else " --real"))
            rc = 0
            return rc
        if not is_mock:
            wh.info("*** REAL HARDWARE: the arm WILL MOVE from home to the first calibration view. ***")
            if io.prompt("Type MOVE to continue: ").strip() != "MOVE":
                raise SystemExit("not confirmed")
        io.prompt("Press Enter to execute (q + Enter cancels): ")
        q_now = io.wait_joint_states()
        if np.max(np.abs(q_now - q0)) > 0.01:
            raise RuntimeError("the arm moved since planning - refusing")
        wh.info("executing ... (q or Ctrl-C cancels immediately)")
        io.execute(jt)
        io.wait_standstill(1.0, 0.002)
        q1 = io.wait_joint_states()
        wh.info("reached; joints (deg) %s ; error %.4f rad" % (np.round(np.degrees(q1), 1).tolist(), np.max(np.abs(q1 - qg))))
        wh.info("now run: python3 wrist_handeye.py --execute --real --auto")
        rc = 0
    except UserStop as e:
        wh.info("STOPPED: %s - active goal cancelled, the arm holds where it is" % e)
    except SystemExit as e:
        wh.info(str(e))
    except Exception as e:  # noqa: BLE001
        wh.info("ERROR: %s" % e)
    finally:
        io.cleaning = True
        try:
            left = io.whe_objects_present()
            if left:
                io.remove_objects(left)
            wh.info("safety scene cleanup: %s" % ("all whe_* objects removed" if not io.whe_objects_present() else "FAILED - remove whe_* manually"))
        except Exception as e:  # noqa: BLE001
            wh.info("cleanup error: %s" % e)
        io.shutdown()
    return rc


if __name__ == "__main__":
    sys.exit(main())
