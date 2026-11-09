#!/usr/bin/env python3
"""Read-only, information only: where is the carton in the snapshot, and which placement problems seen so far does it have?

The logic is shared with the operator UI (src/dual_cut_ui/dual_cut_ui/placement.py), so the page and this script agree.
Four checks, the first from the 2026-11-07 wrist model, the others calibrated on the 42 detections of 2026-10-22 planned by the dual stack in mock
(scan_results/today_plans_20261022/, scan_results/left_face_fail_20261022/):

  two-arm reach wrist model (any box size): can the left wrist get within 0.52 m of its base at back_left AND the right wrist at
                least 0.185 m from its base at front_right, within 0.515 m at back_right / 0.35 m at front_right (plus tip guards and
                'carton at least 115 mm in front of the arms')?  Failing -> advice: the cheapest move / turn / both
                (model estimate; the planner decides).  Replaces the single-arm left reach, which is off by default now.
  region        the box (seam centre x/y, seam yaw) in which today's detections planned: x 0.380 .. 0.435, y -0.245 .. -0.171,
                yaw -92.5 .. -77.7 deg.  Outside it a plan is untested; inside it a plan can still be refused for other reasons.
  left reach    (OFF by default, enable with {"left_reach_enabled": true}) estimate that the left arm can reach the tape band on the left face ("left varying orientation search failed
                closed"): wrist centre of the nominal knife pose within 541 mm (+11 mm for tilting the blade) of the left
                shoulder.  Flags t090925 (centre x 0.456) and t093229 (yaw -97.0), none of the other 40.
  right holder  SOFT: close cartons (x <= 0.4217) turned to yaw <= -89.5 were refused because the right holder touched the
                table edge; may be fixed once the table is in the planning scene.

Replaces the 2026-10-18 window (centre y -0.136 +/- 20 mm of x 0.430, yaw -94.5 .. -89.5), which no longer describes what plans
(mock study dual_window_15mm_20261018 / standoff_sweep_20261018, an older right holder); it is kept for reference: --legacy.

Usage: check_placement.py [SNAPSHOT] [--legacy] [--limits JSON | --limits-file FILE]
  --limits '{"right_table_x_max_m": 0.41, "left_reach_enabled": false}'   override any threshold of PlacementLimits
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "dual_cut_ui"
if (SRC / "dual_cut_ui" / "placement.py").is_file():          # the workspace source first: an installed copy may be older
    sys.path.insert(0, str(SRC))
from dual_cut_ui import placement as P   # noqa: E402

DEFAULT_SNAPSHOT = "/home/farhan/Igus_rebel/outputs/seam_3d_base_link.json"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("snapshot", nargs="?", default=DEFAULT_SNAPSHOT)
    ap.add_argument("--legacy", action="store_true", help="also print the outdated 2026-10-18 window check")
    ap.add_argument("--limits", default="", help="JSON object overriding PlacementLimits fields")
    ap.add_argument("--limits-file", default="", help="file with that JSON object")
    args = ap.parse_args(argv)
    try:
        text = Path(args.limits_file).read_text() if args.limits_file else args.limits
        lim = P.PlacementLimits.from_json(text)
    except (OSError, ValueError) as e:
        print(f"bad limits: {e}", file=sys.stderr)
        return 2
    try:
        doc = json.load(open(args.snapshot))
    except (OSError, ValueError) as e:
        print(f"cannot read the snapshot {args.snapshot}: {e}", file=sys.stderr)
        return 2
    a = P.assess_placement(doc, lim)
    if a is None:
        print("snapshot has no usable seam_start / seam_end / seam_center", file=sys.stderr)
        return 2
    age = time.time() - os.path.getmtime(args.snapshot)
    print(f"snapshot  {args.snapshot}  age {age:.0f} s (limit 240)  cut_permission {doc.get('cut_permission')}")
    reg = a["region"]
    by_axis = {d["axis"]: d for d in reg["deviations"]}
    ok = lambda axis: "OK" if axis not in by_axis else "OUTSIDE"          # noqa: E731
    print(f"seam yaw  {a['yaw_deg']:.1f} deg   (region {lim.region_yaw_deg[0]:.1f} .. {lim.region_yaw_deg[1]:.1f})  {ok('yaw')}")
    print(f"centre    x {a['centre_x_m']:.3f} {ok('x')} (region {lim.region_x_m[0]:.3f} .. {lim.region_x_m[1]:.3f}),"
          f"  y {a['centre_y_m']:.3f} {ok('y')} (region {lim.region_y_m[0]:.3f} .. {lim.region_y_m[1]:.3f}) m")
    tr = a.get("two_arm_reach")
    if tr is None:
        print("two-arm reach  not estimated (snapshot lacks the four top corners or the check is off)")
    else:
        m = tr["margins_mm"]
        print(f"two-arm reach  margins mm: left wrist {m['left_wrist']:.0f}, right wrist {m['right_wrist']:.0f}, left tip {m['left_tip']:.0f}, "
              f"right tip {m['right_tip']:.0f}, right far end {m['right_far']:.0f}, right start {m['right_start']:.0f}, too close {m['close']:.0f} "
              f"-> {tr['status'].upper()} (binding: {tr['binding']}; marginal below {tr['marginal_mm']:.0f} mm; turn {tr['turn_from_ref_deg']:+.0f} deg from the tested pose)")
    lr = a.get("left_reach")
    if lr is None:
        print("left reach  not estimated (off by default, or the snapshot lacks back_middle_left / front_middle_left)")
    else:
        print(f"left reach  wrist centre {lr['wrist_radius_mm']:.0f} mm from the left shoulder at the back end of the tape band "
              f"(flag > {lr['flag_radius_mm']:.0f}, arm limit {lr['envelope_mm']:.0f})  "
              f"{'PROBABLY OUT OF REACH' if lr['flag'] else 'OK'}")
    rt = a["right_table"]
    print(f"right holder vs table  {'MAY BE REFUSED' if rt['flag'] else 'no known problem'} "
          f"(refused so far for x <= {rt['x_max_m']:.3f} and yaw <= {rt['yaw_max_deg']:.1f}; soft)")
    for w in a["warnings"]:
        print(f"-> [{w['level']}] {w['title']}: {w['hint']}")
    if not a["warnings"]:
        print("-> no placement warning (information only)")
    if args.legacy:
        old = P.legacy_check(doc.get("seam_start"), doc.get("seam_end"), doc.get("seam_center"))
        print(f"legacy 2026-10-18 window (outdated): yaw {old['yaw_deg']:.1f} {'OK' if old['yaw_ok'] else 'OUTSIDE'} "
              f"(-94.5 .. -89.5), {old['offset_mm']:.0f} mm from (0.430, -0.136) {'OK' if old['offset_ok'] else 'UNTESTED'} (<= 20 mm)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
