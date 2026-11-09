#!/usr/bin/env python3
"""Markdown table: placement pre-check prediction vs what the dual stack did, for the 42 detections of 2026-10-22.
Usage: placement_table.py [FIXTURE_JSON]   (default: src/dual_cut_ui/test/data/placement_20261022.json)"""
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, os.path.join(ROOT, "src", "dual_cut_ui"))
from dual_cut_ui import placement as P   # noqa: E402

fix = json.load(open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "src", "dual_cut_ui", "test", "data", "placement_20261022.json")))["detections"]
SHORT = {"OK": "plans", "LEFT_FACE": "LEFT FACE reach", "RIGHT_HOLDER_TABLE": "R holder/table", "LEFT_TOP_WRIST": "L top wrist", "TOPFACE": "R top-face check"}
print("| id | yaw | x / y (m) | permission | wrist r mm | left reach | actual left face | right table note | actual R holder/table | region | a / b / c outcome |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
tp = fp = fn = tn = 0
for name, d in fix.items():
    a = P.assess_placement(d["snapshot"])
    lr, rt = a["left_reach"], a["right_table"]
    plans = d["plans"]
    left_fail = any(plans[c]["cause"] == "LEFT_FACE" for c in plans)
    holder = any(plans[c]["cause"] == "RIGHT_HOLDER_TABLE" for c in plans)
    tp += lr["flag"] and left_fail
    fp += lr["flag"] and not left_fail
    fn += (not lr["flag"]) and left_fail
    tn += (not lr["flag"]) and not left_fail
    cp = d["snapshot"]["cut_permission"]["status"]
    outcome = " / ".join(("OK" if plans[c]["ok"] else SHORT.get(plans[c]["cause"], plans[c]["cause"])) for c in "abc")
    print(f"| {name} | {a['yaw_deg']:.1f} | {a['centre_x_m']:.3f} / {a['centre_y_m']:.3f} | {cp} | {lr['wrist_radius_mm']:.1f} | "
          f"{'**FLAG**' if lr['flag'] else '-'} | {'**FAILED**' if left_fail else '-'} | {'note' if rt['flag'] else '-'} | "
          f"{'refused' if holder else '-'} | {'inside' if a['region']['inside'] else 'outside (' + ','.join(x['axis'] for x in a['region']['deviations']) + ')'} | {outcome} |")
print(f"\nleft reach: {tp} true positives, {fp} false positives, {fn} false negatives, {tn} true negatives (of {len(fix)})")
