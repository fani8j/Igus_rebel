#!/usr/bin/env python3
"""MOCK ONLY - stands in for capture_and_query_depth.py when testing the dual-cut UI without a camera.  NOT FOR THE REAL CELL.

Takes the same options as the real detector (so the UI's ``detect_command`` can simply point at this script), copies a
recorded snapshot to --planner-output-json with ``acquisition_stamp_ns`` = now (and a fresh file mtime), and creates the
run directory the UI looks for:

    <output-dir>/pipeline/<capture_id>/<run>/planner_snapshot.json                       (byte-identical to the copy)
    <output-dir>/pipeline/<capture_id>/<run>/autolabel_v2_overlays/<capture_id>_autolabel_v2.png

    mock_detect.py --source-snapshot recorded.json [--overlay-png some.png] --output-dir /tmp/mock_det \
        --planner-output-json /tmp/mock_det/seam_3d_base_link.json [--sleep 3] [--exit-code 0]

It refuses to write the real snapshot file (/home/farhan/Igus_rebel/outputs/seam_3d_base_link.json) or anything in the
real capture directory.  Exit code: 0 ok; --exit-code N simulates a rejection (1) or a prerequisite failure (2).
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

REAL_SNAPSHOT = "/home/farhan/Igus_rebel/outputs/seam_3d_base_link.json"
REAL_CAPTURE_DIR = "/home/farhan/Igus_rebel/outputs"


def refuse_reason(planner_json: str, output_dir: str):
    """Why this invocation must not run (None if fine).  Compares real paths, so symlinks do not get around it."""
    real = os.path.realpath
    if real(planner_json) == real(REAL_SNAPSHOT):
        return f"refusing to write the REAL snapshot {REAL_SNAPSHOT}: this is the MOCK detector"
    out = real(output_dir)
    if out == real(REAL_CAPTURE_DIR) or out.startswith(real(REAL_CAPTURE_DIR) + os.sep):
        return f"refusing to write below the real capture directory {REAL_CAPTURE_DIR}: this is the MOCK detector"
    return None


def synth_overlay(doc: dict, path: str) -> None:
    import cv2
    import numpy as np
    img = np.full((540, 720, 3), (30, 24, 16), np.uint8)
    keys = ("front_left", "front_right", "back_right", "back_left")
    pts = [doc.get(k) for k in keys] + [doc.get("seam_start"), doc.get("seam_end")]
    if all(isinstance(p, dict) for p in pts):
        xs, ys = [p["x"] for p in pts], [p["y"] for p in pts]
        k = 0.8 * min(720 / max(max(ys) - min(ys), 1e-3), 540 / max(max(xs) - min(xs), 1e-3))
        cx, cy = (max(ys) + min(ys)) / 2, (max(xs) + min(xs)) / 2

        def px(p):
            return int(360 + (cx - p["y"]) * k), int(270 + (cy - p["x"]) * k)
        cv2.polylines(img, [np.array([px(doc[q]) for q in keys], np.int32)], True, (180, 160, 120), 2)
        cv2.line(img, px(doc["seam_start"]), px(doc["seam_end"]), (60, 220, 255), 3)
        for q in keys:
            cv2.circle(img, px(doc[q]), 6, (80, 200, 80), -1)
    cv2.putText(img, "MOCK DETECTION OVERLAY", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(img, str(doc.get("capture_id")), (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.imwrite(path, img)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-snapshot", required=True, help="recorded box_h_cut_snapshot/v1 JSON to replay")
    ap.add_argument("--overlay-png", default="", help="PNG to use as the overlay (default: synthesized)")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--planner-output-json", required=True)
    ap.add_argument("--target-frame", default="base_link")
    ap.add_argument("--sleep", type=float, default=3.0)
    ap.add_argument("--exit-code", type=int, default=0, help="exit with this code instead of succeeding (simulate a failure)")
    for flag in ("--autolabel-v2", "--allow-needs-review", "--allow-cut-blocked"):     # accepted and ignored
        ap.add_argument(flag, action="store_true")
    a = ap.parse_args(argv)
    why = refuse_reason(a.planner_output_json, a.output_dir)
    if why:
        print(why, file=sys.stderr, flush=True)
        return 3
    print("[mock-detect] MOCK detector (no camera is used)", flush=True)
    with open(a.source_snapshot) as f:
        doc = json.load(f)
    print("[mock-detect] capturing ...", flush=True)
    time.sleep(a.sleep / 3)
    print("[mock-detect] detecting seam ...", flush=True)
    time.sleep(a.sleep / 3)
    if a.exit_code:
        print(f"[mock-detect] simulated failure, exit {a.exit_code}", flush=True)
        return a.exit_code
    print("[mock-detect] measuring carton top ...", flush=True)
    time.sleep(a.sleep / 3)
    doc["acquisition_stamp_ns"] = time.time_ns()
    doc["mock_restamped"] = True
    raw = (json.dumps(doc, indent=2) + "\n").encode()
    capture_id = str(doc.get("capture_id") or "mock_capture")
    run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_mock"
    run_dir = os.path.join(a.output_dir, "pipeline", capture_id, run)
    os.makedirs(os.path.join(run_dir, "autolabel_v2_overlays"), exist_ok=True)
    with open(os.path.join(run_dir, "planner_snapshot.json"), "wb") as f:
        f.write(raw)
    overlay = os.path.join(run_dir, "autolabel_v2_overlays", f"{capture_id}_autolabel_v2.png")
    if a.overlay_png:
        with open(a.overlay_png, "rb") as src, open(overlay, "wb") as dst:
            dst.write(src.read())
    else:
        synth_overlay(doc, overlay)
    os.makedirs(os.path.dirname(os.path.abspath(a.planner_output_json)), exist_ok=True)
    tmp = a.planner_output_json + ".tmp"
    with open(tmp, "wb") as f:
        f.write(raw)
    os.replace(tmp, a.planner_output_json)           # fresh mtime, atomic for readers
    print(f"[mock-detect] wrote {a.planner_output_json} and {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
