#!/usr/bin/env python3
"""Rebuild src/dual_cut_ui/test/data/placement_20261022.json from the 2026-10-22 mock study (read-only on the study).

The fixture holds, for each of the 42 detections of that day, the snapshot fields the placement pre-check reads plus what
the dual stack did with it (cells a/b/c of scan_results/today_plans_20261022/results.csv: ok + cause group), so the UI tests
need nothing outside the package.  Usage: make_placement_fixture.py [STUDY_DIR] [OUT_JSON]
"""
import csv
import glob
import json
import os
import sys

STUDY = sys.argv[1] if len(sys.argv) > 1 else "/home/farhan/igus_rebel_ros2/scan_results/today_plans_20261022"
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src", "dual_cut_ui", "test", "data", "placement_20261022.json")
KEYS = ("seam_start", "seam_end", "seam_center", "back_middle_left", "front_middle_left", "back_left", "front_left",
        "back_right", "front_right", "cut_permission")

cells = {}
for r in csv.DictReader(open(os.path.join(STUDY, "results.csv"))):
    cells.setdefault(r["snapshot"], {})[r["cell"]] = {"ok": r["ok"] == "True", "cause": r["cause_group"],
                                                      "back_trim_mm": float(r["left_face_back_trim_mm"]) if r["left_face_back_trim_mm"] else None}
out = {}
for f in sorted(glob.glob(os.path.join(STUDY, "snapshots", "t*.json"))):
    name = os.path.basename(f)[:-5]
    doc = json.load(open(f))
    cp = doc.get("cut_permission") or {}
    out[name] = {"snapshot": {k: doc[k] for k in KEYS if k in doc},
                 "plans": {c: cells[name][c] for c in "abc" if c in cells.get(name, {})}}
    out[name]["snapshot"]["cut_permission"] = {"status": cp.get("status"), "reasons": [{"code": r.get("code")} for r in cp.get("reasons") or []]}
json.dump({"source": "scan_results/today_plans_20261022 (snapshots/ + results.csv); cells: a sequence, b sequence+detour, c phases top_first",
           "detections": out}, open(OUT, "w"), indent=0, sort_keys=True)
print(f"{len(out)} detections -> {OUT}")
