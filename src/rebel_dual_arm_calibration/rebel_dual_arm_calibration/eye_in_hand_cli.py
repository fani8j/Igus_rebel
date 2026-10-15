"""Re-solve a recorded wrist (eye-in-hand) session offline.

    python3 -m rebel_dual_arm_calibration.eye_in_hand_cli <session_dir> [--out DIR] [--no-loo] [--arm left]

Reads ``<session_dir>/wrist_samples.jsonl`` (and ``scene_board.json`` for the cross-check if present), prints the summary
and writes ``wrist_handeye_result.yaml`` / ``wrist_handeye.xacro.txt`` / ``wrist_handeye_report.json`` to ``--out``
(default: the session directory).  No ROS needed; nothing is sent to a robot.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from .transforms import xyz_rpy_to_T

from . import eye_in_hand as eih


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session_dir")
    ap.add_argument("--out", default="", help="output directory (default: the session directory)")
    ap.add_argument("--arm", default="left")
    ap.add_argument("--no-loo", action="store_true", help="skip the leave-one-out held-out evaluation")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--camera-link-optical", type=float, nargs=6, metavar=("X", "Y", "Z", "R", "P", "YAW"), default=None,
                    help="transform wrist_camera_link -> optical frame of the camera driver (m, rad; identity on the real "
                         "camera). Without it the link6 -> wrist_camera_link joint and the xacro snippet are NOT written.")
    args = ap.parse_args(argv)
    samples = eih.load_samples(args.session_dir)
    if not samples:
        print(f"no samples in {args.session_dir}/{eih.SAMPLES_FILE}", file=sys.stderr)
        return 2
    scene = eih.load_scene_board(args.session_dir)
    T_scene = np.array(scene["T_base_board"], dtype=float) if scene and scene.get("T_base_board") else None
    try:
        res = eih.solve(samples, T_scene, loo=not args.no_loo)
    except eih.HandEyeError as e:
        print(f"solve failed: {e}", file=sys.stderr)
        return 3
    print(eih.format_summary(res))
    if not args.no_write:
        cam = {"frame_id": samples[0].camera_frame_id, "image_size": list(samples[0].image_size),
               "K": samples[0].K.tolist(), "dist": samples[0].dist.tolist()}
        T_cl = None if args.camera_link_optical is None else xyz_rpy_to_T(args.camera_link_optical[:3],
                                                                         args.camera_link_optical[3:])
        paths = eih.write_outputs(args.out or args.session_dir, res, args.arm, samples, camera=cam,
                                  T_base_board_scene=T_scene, T_camera_link_optical=T_cl)
        if T_cl is None:
            print("note: no --camera-link-optical given: the link6 -> wrist_camera_link joint / xacro were not written")
        for k, v in paths.items():
            print(f"{k}: {v}")
    return 0 if res.gate["verdict"] != "FAIL" else 1


if __name__ == "__main__":
    sys.exit(main())
