#!/usr/bin/env python3
"""Offline check of the GR00T server running the RIGHT-arm checkpoint real_right_3flaps_35ep_15fps (no robot, no ROS).

The checkpoint was trained on 35 episodes at 15 fps (every 2nd frame of the 30 fps recordings; the checkpoint's state and
relative-action statistics match exactly that): the 20 `real_right` episodes of 29 Oct (front flap, task "open the front flap
of the cardboard box") + the 15 `real_right_3nov_raw` robot-playback episodes of 3 Nov (5 box poses x front / back / right flap,
task "open the <front|back|right> flap of the cardboard box").  Video keys head + wrist_right, source videos 1152x648.

For each episode and every --stride-th 30 fps frame t this sends what the live client sends: head + wrist_right frame t
(PIL BILINEAR to --video-size), state[t] and the episode's task text (or --instruction), and compares the returned 16x6 chunk
with the recorded action[t + 2k], k = 0..15 (15 fps model steps).  Reports MAE vs "hold still", first step, step jump, gates.

    nice -n 19 ~/groot/Isaac-GR00T/.venv/bin/python -I tools/groot_inference/offline_check_3flaps.py --json-out /tmp/oc.json
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

DS = "/media/farhan/My Passport/igus_nvidia/datasets/"
ROOTS = {"oct29": DS + "real_right_20261029_raw/real_right", "nov3": DS + "real_right_3nov_raw/real_right"}
SRC_W, SRC_H = 1152, 648
STEP = 2        # 30 fps frames per 15 fps model step
FRONT = "open the front flap of the cardboard box"


def tasks_of(root):
    p = Path(root) / "meta/episodes.jsonl"
    return {json.loads(l)["episode_index"]: json.loads(l) for l in p.read_text().splitlines() if l.strip()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="*", default=["oct29", "nov3"], choices=list(ROOTS))
    ap.add_argument("--oct29-episodes", type=int, nargs="*", default=list(range(19)), help="ep 19 is corrupt")
    ap.add_argument("--nov3-episodes", type=int, nargs="*", default=list(range(15)))
    ap.add_argument("--stride", type=int, default=90, help="30 fps frames between samples")
    ap.add_argument("--video-size", type=int, nargs=2, default=[640, 360], metavar=("W", "H"))
    ap.add_argument("--instruction", default=None, help="override every episode's task text")
    ap.add_argument("--wrist-black", action="store_true")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()
    size = tuple(a.video_size)
    cli = Client(a.host, a.port, timeout_ms=120000)
    print("modality:", json.dumps(cli.call("get_modality_config"), default=str)[:300])
    rows = []
    for name in a.sets:
        root = Path(ROOTS[name])
        meta = tasks_of(root)
        for ep in (a.oct29_episodes if name == "oct29" else a.nov3_episodes):
            task = a.instruction or (meta[ep]["tasks"][0] if ep in meta else FRONT)
            d = pd.read_parquet(root / f"data/chunk-000/episode_{ep:06d}.parquet")
            S = np.stack(d["observation.state"].values).astype(np.float64)
            A = np.stack(d["action"].values).astype(np.float64)
            ts = list(range(0, len(S) - 15 * STEP - 1, a.stride))
            vid = {k: read_frames(root / f"videos/chunk-000/observation.images.{k}/episode_{ep:06d}.mp4", ts, h=SRC_H, w=SRC_W)
                   for k in ("head", "wrist_right")}
            for t in ts:
                if t not in vid["head"] or t not in vid["wrist_right"]:
                    continue
                im = {k: np.asarray(Image.fromarray(vid[k][t]).resize(size, Image.BILINEAR)) for k in vid}
                if a.wrist_black:
                    im["wrist_right"] = np.zeros_like(im["wrist_right"])
                obs = {"video": {k: v[None, None] for k, v in im.items()},
                       "state": {STATE_KEY: S[t].astype(np.float32)[None, None]},
                       "language": {LANGUAGE_KEY: [[task]]}}
                t0 = time.time()
                act = cli.get_action(obs)
                lat = (time.time() - t0) * 1000
                p = np.asarray(act[STATE_KEY], np.float64).reshape(-1, 6)[:16]
                demo = A[t:t + 16 * STEP:STEP]
                r = {"set": name, "ep": ep, "t": t, "task": task, "mae": float(np.abs(p - demo).mean()),
                     "mae_hold": float(np.abs(S[t] - demo).mean()), "first": float(np.abs(p[0] - S[t]).max()),
                     "jump": float(np.abs(np.diff(p, axis=0)).max()), "lat_ms": lat}
                dm, pm = demo[-1] - S[t], p[-1] - S[t]
                if np.linalg.norm(dm) > 0.02:
                    r["cos"] = float(dm @ pm / (np.linalg.norm(dm) * np.linalg.norm(pm) + 1e-9))
                rows.append(r)
            e = [r for r in rows if r["set"] == name and r["ep"] == ep]
            if e:
                print(f"{name} ep{ep:02d} {task[9:14]:5s} n={len(e):3d} mae {np.mean([r['mae'] for r in e]):.4f} "
                      f"hold {np.mean([r['mae_hold'] for r in e]):.4f} first max {max(r['first'] for r in e):.3f} "
                      f"jump max {max(r['jump'] for r in e):.3f} "
                      f"cos med {np.median([r['cos'] for r in e if 'cos' in r] or [np.nan]):.2f}", flush=True)

    def summ(rs):
        first = np.array([r["first"] for r in rs]); jump = np.array([r["jump"] for r in rs])
        cos = np.array([r["cos"] for r in rs if "cos" in r])
        mae, hold = np.mean([r["mae"] for r in rs]), np.mean([r["mae_hold"] for r in rs])
        return {"n": len(rs), "mae": float(mae), "mae_hold": float(hold), "ratio_vs_hold": float(mae / hold),
                "first_median": float(np.median(first)), "first_max": float(first.max()),
                "jump_p95": float(np.percentile(jump, 95)), "cos_median": float(np.median(cos)) if len(cos) else None,
                "pass_first_0.05_jump_0.1": float(((first <= 0.05) & (jump <= 0.1)).mean()),
                "latency_ms_median": float(np.median([r["lat_ms"] for r in rs]))}
    out = {"video_size": size, "all": summ(rows)}
    for name in a.sets:
        rs = [r for r in rows if r["set"] == name]
        if rs:
            out[name] = summ(rs)
    for flap in ("front", "back", "right"):
        rs = [r for r in rows if r["set"] == "nov3" and f" {flap} " in r["task"]]
        if rs:
            out["nov3_" + flap] = summ(rs)
    print("SUMMARY", json.dumps(out, indent=1))
    if a.json_out:
        Path(a.json_out).write_text(json.dumps({"summary": out, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
