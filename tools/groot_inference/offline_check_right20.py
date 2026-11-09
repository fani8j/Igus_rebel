#!/usr/bin/env python3
"""Offline check of the GR00T server running the RIGHT-arm checkpoint rebel_open_front_20ep_360p (no robot, no ROS).

The checkpoint was trained on the 20-episode LeRobot v2.1 dataset `real_right` (29 Oct, right arm, "open the front flap of the
cardboard box"; its statistics match the checkpoint's exactly), every frame (30 Hz, action delta_indices 0..15), video keys
head + wrist_right, images "360p".  For each episode and every --stride-th frame t this sends what the live client sends:
head + wrist_right frame t (decoded from the 1152x648 h264 videos, then PIL BILINEAR to 640x360), state[t] and the instruction,
and compares the returned 16x6 chunk with the recorded action[t .. t+15].

Reports per episode and overall: MAE pred vs demo, MAE of "hold still" (pred = state), first step |pred[0] - state| (the
client's first_step gate), max row-to-row step (step_jump gate) and how many samples would pass a few gate settings.
Episode 19 is corrupt (state jumps to zeros) and skipped by default.

    nice -n 19 ~/groot/Isaac-GR00T/.venv/bin/python -I tools/groot_inference/offline_check_right20.py \\
        --dataset "/media/farhan/My Passport/igus_nvidia/datasets/real_right_20261029_raw/real_right" --json-out /tmp/oc_right20.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from offline_check import Client, read_frames, STATE_KEY, LANGUAGE_KEY  # noqa: E402

SRC_W, SRC_H = 1152, 648
VIDEO_SIZE = (640, 360)


def to_model(rgb):
    return np.asarray(Image.fromarray(rgb).resize(VIDEO_SIZE, Image.BILINEAR))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--episodes", type=int, nargs="*", default=list(range(19)))
    ap.add_argument("--stride", type=int, default=60)
    ap.add_argument("--instruction", default="open the front flap of the cardboard box")
    ap.add_argument("--wrist-black", action="store_true", help="ablation: send a black wrist_right image")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()
    root = Path(a.dataset)
    cli = Client(a.host, a.port, timeout_ms=120000)
    print("modality:", json.dumps(cli.call("get_modality_config"), default=str)[:400])
    rows = []
    for ep in a.episodes:
        d = pd.read_parquet(root / f"data/chunk-000/episode_{ep:06d}.parquet")
        S = np.stack(d["observation.state"].values).astype(np.float64)
        A = np.stack(d["action"].values).astype(np.float64)
        ts = list(range(0, len(S) - 16, a.stride))
        vid = {k: read_frames(root / f"videos/chunk-000/observation.images.{k}/episode_{ep:06d}.mp4", ts, h=SRC_H, w=SRC_W)
               for k in ("head", "wrist_right")}
        for t in ts:
            if t not in vid["head"] or t not in vid["wrist_right"]:
                continue
            wrist = np.zeros((VIDEO_SIZE[1], VIDEO_SIZE[0], 3), np.uint8) if a.wrist_black else to_model(vid["wrist_right"][t])
            obs = {"video": {"head": to_model(vid["head"][t])[None, None], "wrist_right": wrist[None, None]},
                   "state": {STATE_KEY: S[t].astype(np.float32)[None, None]},
                   "language": {LANGUAGE_KEY: [[a.instruction]]}}
            t0 = time.time()
            act = cli.get_action(obs)
            lat = (time.time() - t0) * 1000
            p = np.asarray(act[STATE_KEY], np.float64).reshape(-1, 6)[:16]
            demo = A[t:t + 16]
            r = {"ep": ep, "t": t, "mae": float(np.abs(p - demo).mean()), "mae_hold": float(np.abs(S[t] - demo).mean()),
                 "first": float(np.abs(p[0] - S[t]).max()), "jump": float(np.abs(np.diff(p, axis=0)).max()),
                 "end_err": float(np.abs(p[-1] - demo[-1]).max()), "lat_ms": lat}
            dm, pm = demo[-1] - S[t], p[-1] - S[t]
            if np.linalg.norm(dm) > 0.02:
                r["cos"] = float(dm @ pm / (np.linalg.norm(dm) * np.linalg.norm(pm) + 1e-9))
            rows.append(r)
        e = [r for r in rows if r["ep"] == ep]
        if e:
            print(f"ep{ep:02d} n={len(e):3d} mae {np.mean([r['mae'] for r in e]):.4f} hold {np.mean([r['mae_hold'] for r in e]):.4f} "
                  f"first med {np.median([r['first'] for r in e]):.3f} max {max(r['first'] for r in e):.3f} "
                  f"jump max {max(r['jump'] for r in e):.3f} cos med {np.median([r['cos'] for r in e if 'cos' in r] or [np.nan]):.2f}", flush=True)
    m = lambda k: float(np.mean([r[k] for r in rows]))
    first = np.array([r["first"] for r in rows])
    jump = np.array([r["jump"] for r in rows])
    cos = np.array([r["cos"] for r in rows if "cos" in r])
    summ = {"n": len(rows), "mae": m("mae"), "mae_hold": m("mae_hold"), "ratio_vs_hold": m("mae") / m("mae_hold"),
            "first_median": float(np.median(first)), "first_p95": float(np.percentile(first, 95)),
            "jump_p95": float(np.percentile(jump, 95)), "cos_median": float(np.median(cos)) if len(cos) else None,
            "cos_neg_frac": float((cos < 0).mean()) if len(cos) else None,
            "pass_first_0.05_jump_0.1": float(((first <= 0.05) & (jump <= 0.1)).mean()),
            "pass_first_0.10_jump_0.1": float(((first <= 0.10) & (jump <= 0.1)).mean()),
            "latency_ms_median": float(np.median([r["lat_ms"] for r in rows]))}
    print("SUMMARY", json.dumps(summ, indent=1))
    if a.json_out:
        Path(a.json_out).write_text(json.dumps({"summary": summ, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
