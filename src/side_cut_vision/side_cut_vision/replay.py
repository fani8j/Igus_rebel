"""Offline replay CLI (console script side_cut_vision_replay).

  side_cut_vision_replay record <look_dir>... [--arm left|right] [--snapshot FILE] [--out-dir DIR]
      Re-run the detector on saved LOOK records (color.png, depth.png, camera_info.json,
      T_world_optical.json, snapshot.json) and re-draw the overlay + measurement JSON.

  side_cut_vision_replay color-only <image.jpg|video.mp4>... [--at-s T ...] [--out-dir DIR]
      Colour-only qualitative mode for frames of the recorded clips (no depth): the strongest
      straight edge and dark-line seam candidates in pixels. No metric output.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

import cv2
import numpy as np

from .coloronly import detect_color_only, draw_color_only
from .detect import DetectParams
from .measure import measure
from .model import build_arm_model
from .overlay import draw_overlay, to_bgr
from .record import RecordError, load_record


def replay_record(path: str, arm: Optional[str] = None, snapshot: Optional[str] = None,
                  out_dir: Optional[str] = None, calib_status: str = "") -> dict:
    rec = load_record(path, arm=arm, snapshot_path=snapshot)
    if rec.snapshot is None:
        raise RecordError(f"{path}: no snapshot.json (pass --snapshot)")
    model = build_arm_model(rec.snapshot, rec.arm)
    m = measure(rec.color_rgb, rec.depth, rec.cam, rec.T_wo, model, DetectParams(),
                tip_world=rec.tip_world, meta={"replay_of": os.path.abspath(path)})
    status = calib_status or ((rec.measurement or {}).get("meta") or {}).get("calib_status", "")
    ov = draw_overlay(rec.color_rgb, rec.cam, rec.T_wo, model, m, rec.tip_world, state="REPLAY",
                      calib_status=status, snapshot_id=rec.snapshot.snapshot_id)
    out = out_dir or path
    os.makedirs(out, exist_ok=True)
    cv2.imwrite(os.path.join(out, "replay_overlay.png"), to_bgr(ov))
    d = m.to_dict()
    with open(os.path.join(out, "replay_measurement.json"), "w") as f:
        json.dump(d, f, indent=2)
    return d


def _read_frames(path: str, at_s: List[float]):
    """Yield (label, rgb) for an image file or for the requested times of a video."""
    ext = os.path.splitext(path)[1].lower()
    base = os.path.splitext(os.path.basename(path))[0]
    if ext in (".mp4", ".avi", ".mkv", ".mov"):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RecordError(f"cannot open video {path}")
        for t in at_s or [0.0]:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, bgr = cap.read()
            if ok:
                yield f"{base}_t{int(round(t)):03d}", cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        cap.release()
    else:
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RecordError(f"cannot read image {path}")
        yield base, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def replay_color_only(path: str, out_dir: str, at_s: Optional[List[float]] = None,
                      orientation: str = "any") -> List[dict]:
    os.makedirs(out_dir, exist_ok=True)
    out = []
    for label, rgb in _read_frames(path, at_s or []):
        res = detect_color_only(rgb, orientation)
        ov = draw_color_only(rgb, res, label)
        fn = os.path.join(out_dir, f"{label}_coloronly.png")
        cv2.imwrite(fn, to_bgr(ov))
        d = res.to_dict()
        d["frame"] = label
        d["overlay"] = fn
        out.append(d)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="side_cut_vision_replay", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record", help="re-run the detector on saved LOOK record folders")
    r.add_argument("dirs", nargs="+")
    r.add_argument("--arm", choices=["left", "right"])
    r.add_argument("--snapshot")
    r.add_argument("--out-dir", help="write replay outputs here instead of into each record")
    c = sub.add_parser("color-only", help="colour-only qualitative detection on images / video frames")
    c.add_argument("inputs", nargs="+")
    c.add_argument("--at-s", type=float, nargs="*", default=[], help="video times (s)")
    c.add_argument("--out-dir", required=True)
    c.add_argument("--orientation", choices=["any", "horizontal", "vertical"], default="any")
    args = ap.parse_args(argv)
    rc = 0
    if args.cmd == "record":
        for p in args.dirs:
            try:
                d = replay_record(p, args.arm, args.snapshot,
                                  os.path.join(args.out_dir, os.path.basename(os.path.normpath(p)))
                                  if args.out_dir else None)
            except RecordError as exc:
                print(f"{p}: ERROR {exc}", file=sys.stderr)
                rc = 1
                continue
            dm = d["delta_mm"]
            print(f"{p}: ok={d['ok']} seam_ok={d['seam_ok']} delta(a,c,n)=({dm['along']}, {dm['across']}, "
                  f"{dm['normal']}) mm yaw={d['delta_yaw_deg']} reasons={d['reasons']}")
    else:
        for p in args.inputs:
            try:
                results = replay_color_only(p, args.out_dir, args.at_s, args.orientation)
            except RecordError as exc:
                print(f"{p}: ERROR {exc}", file=sys.stderr)
                rc = 1
                continue
            for d in results:
                seams = ", ".join(f"{s['confidence']}" for s in d["seams"]) or "-"
                print(f"{d['frame']}: ok={d['ok']} edge_support={d['edge_support_px']}px "
                      f"side_contrast={d['side_contrast']} seam_conf=[{seams}] {d['reasons']} -> {d['overlay']}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
