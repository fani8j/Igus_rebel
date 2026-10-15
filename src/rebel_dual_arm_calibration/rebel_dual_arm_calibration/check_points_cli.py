"""``ros2 run rebel_dual_arm_calibration check_points <session_dir> [...] --calibration <yaml>``

Computes ONLY the check-point error (TRE, held out) of the touches in the given sessions against a given calibration:
``||T_world_left p_left - T_world_right(base) p_right||`` per check point with the base of the YAML and NOMINAL forward
kinematics.  No fitting.  Use it for the re-reference check after a power cycle / re-homing of the arms (re-touch the
same check stickers and compare with the calibration that is deployed) and as an independent verification of a
calibration with check stickers that were touched after the calibration.

Exit code 0 = PASS (>= min check points, max error <= limit), 1 = FAIL / not evaluable, 2 = bad input.
Accuracy is demonstrated at the check points only."""
from __future__ import annotations

import argparse
import json
import sys
from typing import Optional, Sequence

from .base_check import GateOptions
from .metrology import check_point_report, load_calibration_yaml, recommended_padding_mm
from .session import SessionMismatchError, load_sessions_full


def evaluate(touches, T_R, opts: GateOptions) -> dict:
    cp = check_point_report(touches, T_R)
    pts = cp["points"]
    problems = []
    if len(pts) < opts.min_check_points:
        problems.append(f"{len(pts)} check point(s) touched by both arms (< {opts.min_check_points})")
    if pts and cp["max_mm"] > opts.max_check_error_mm:
        problems.append(f"max check-point error {cp['max_mm']:.2f} mm > {opts.max_check_error_mm:.1f} mm")
    cp["limit_mm"] = opts.max_check_error_mm
    cp["problems"] = problems
    cp["passed"] = not problems
    cp["recommended_padding_mm"] = recommended_padding_mm(cp.get("max_mm"))
    return cp


def format_result(cp: dict, calib_path: str) -> list:
    L = [f"Check points against {calib_path}: " + cp["label"]]
    for p in cp["points"]:
        geo = ("" if "height_above_plane_mm" not in p else
               f", {p['height_above_plane_mm']:.0f} mm off the plane, {p['dist_from_centre_mm']:.0f} mm from the centre")
        L.append(f"  point {p['point']}: {p['error_mm']:.2f} mm (L x{p['n_left']}, R x{p['n_right']}{geo})")
    for i in cp["incomplete"]:
        L.append(f"  point {i['point']}: touched by the {i['touched_by']} arm only, not evaluated")
    if cp["points"]:
        L.append(f"  RMS {cp['rms_mm']:.2f} mm, max {cp['max_mm']:.2f} mm over {cp['n_points']} point(s) (limit "
                 f"{cp['limit_mm']:.1f} mm)")
    L.append(f"  recommended_padding_mm = max(20, 2 x max check-point error + 10) = {cp['recommended_padding_mm']:.0f}")
    L.append("  " + cp["scope"])
    L.append("  RESULT: " + ("PASS" if cp["passed"] else "FAIL: " + "; ".join(cp["problems"])))
    return L


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] + " (no fitting)")
    ap.add_argument("session_dirs", nargs="+", metavar="session_dir")
    ap.add_argument("--calibration", required=True, metavar="YAML", help="dual_arm_calibration.yaml to check")
    ap.add_argument("--check-point-ids", default="", help="comma-separated touch ids to treat as check points "
                                                           "(overrides the recorded kind)")
    ap.add_argument("--min-check-points", type=int, default=2)
    ap.add_argument("--max-error-mm", type=float, default=3.0)
    ap.add_argument("--json", default=None, help="also write the result as JSON")
    args = ap.parse_args(argv)
    try:
        cal = load_calibration_yaml(args.calibration)
        data = load_sessions_full(args.session_dirs)
    except (OSError, KeyError, ValueError, SessionMismatchError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    touches = data.touches
    try:
        ids = {int(v) for v in args.check_point_ids.split(",") if v.strip()}
    except ValueError:
        print("ERROR: --check-point-ids needs comma-separated integers", file=sys.stderr)
        return 2
    for t in touches:
        if t.point_id in ids:
            t.kind = "check"
    opts = GateOptions(min_check_points=args.min_check_points, max_check_error_mm=args.max_error_mm)
    cp = evaluate(touches, cal["T_R"], opts)
    print("\n".join(format_result(cp, args.calibration)))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(cp, f, indent=2)
    return 0 if cp["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
