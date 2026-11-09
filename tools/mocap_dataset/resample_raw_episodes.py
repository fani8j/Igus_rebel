#!/usr/bin/env python3
"""Resample raw lid_open recorder episodes (real_collect.py output: meta.json, steps.jsonl, images/<cam>/NNNNNN.jpg)
to a higher fixed rate, so that `to_lerobot_gr00t.py` writes a dataset at the rate TurboVLA expects (30 fps).

    python3 tools/mocap_dataset/resample_raw_episodes.py SRC_DIR --out OUT_DIR [--fps 30] [--fill-stale 3]

The source is only read. For every episode folder in SRC_DIR (those with meta.json):
* joints: repeats of <= --fill-stale frames (a /joint_states value seen again, the recorder samples faster than the topic
  updates) are first linearly interpolated between the neighbouring updated frames (same rule as the converter's
  --fill-stale), then ALL joints are linearly interpolated by the step time `t` on the grid t = k / fps, k = 0..floor(t_end * fps);
* images: zero-order hold, the image recorded at the latest step with t_i <= t_k (causal, never a picture from the
  future); it is a symlink to the source jpg, so the output is small and the converter encodes the video from it;
* img_age / welded / phase: taken from the same previous step; meta.json gets fps, num_steps and a `resampled` record.
The 29 Oct real_right set was recorded at 30 Hz directly (real_collect.py `record.hz: 30`), the 3 Nov set at 15 Hz.
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np


def fill_stale(state, t, max_run):
    """Same rule as to_lerobot_gr00t.fill_stale: runs of <= max_run frames equal to the previous one are interpolated
    (by t) between the previous and the next frame that changed. Returns (new state, frames filled)."""
    state = state.copy()
    changed = np.r_[True, np.any(np.diff(state, axis=0) != 0, axis=1)]
    idx = np.flatnonzero(changed)
    filled = 0
    for a, b in zip(idx[:-1], idx[1:]):
        run = b - a - 1
        if 0 < run <= max_run:
            w = ((t[a + 1:b] - t[a]) / (t[b] - t[a]))[:, None]
            state[a + 1:b] = (1 - w) * state[a] + w * state[b]
            filled += run
    return state, filled


def resample(t, q, fps, max_stale_run=3):
    """t (N,) strictly increasing, q (N, J) -> (grid t (M,), q on the grid (M, J), source step index per grid point (M,), filled)."""
    t = np.asarray(t, float)
    if len(t) < 2 or np.any(np.diff(t) <= 0):
        raise ValueError("step times must be strictly increasing")
    q = np.asarray(q, float)
    filled = 0
    if max_stale_run > 0:
        q, filled = fill_stale(q, t, max_stale_run)
    m = int(np.floor((t[-1] - t[0]) * fps + 1e-9)) + 1
    tk = t[0] + np.arange(m) / fps
    qk = np.stack([np.interp(tk, t, q[:, j]) for j in range(q.shape[1])], axis=1)
    src = np.searchsorted(t, tk + 1e-9, side="right") - 1
    return tk, qk, np.clip(src, 0, len(t) - 1), filled


def convert_episode(src, out, fps, max_stale_run):
    meta = json.loads((src / "meta.json").read_text())
    steps = [json.loads(line) for line in (src / "steps.jsonl").read_text().splitlines() if line.strip()]
    cams = sorted(p.name for p in (src / "images").iterdir() if p.is_dir())
    for c in cams:
        n_img = len(list((src / "images" / c).glob("*.jpg")))
        if n_img != len(steps):
            raise SystemExit(f"{src}: {len(steps)} steps but {n_img} images in {c}")
    names = list(steps[0]["joints"])
    t = np.array([s["t"] for s in steps])
    q = np.array([[s["joints"][j] for j in names] for s in steps])
    tk, qk, idx, filled = resample(t, q, fps, max_stale_run)
    out.mkdir(parents=True)
    with open(out / "steps.jsonl", "w") as f:
        for k in range(len(tk)):
            s = steps[idx[k]]
            row = {"t": round(float(tk[k] - t[0]), 6), "joints": {j: float(qk[k, i]) for i, j in enumerate(names)},
                   "welded": s.get("welded"), "phase": s.get("phase"), "img_age": s.get("img_age"), "src_step": int(idx[k])}
            f.write(json.dumps(row) + "\n")
    for c in cams:
        d = out / "images" / c
        d.mkdir(parents=True)
        for k in range(len(tk)):
            os.symlink(src / "images" / c / f"{idx[k]:06d}.jpg", d / f"{k:06d}.jpg")
    meta2 = dict(meta, fps=float(fps), num_steps=len(tk),
                 resampled={"from_fps": meta.get("fps"), "from_steps": len(steps), "to_fps": fps, "joints": "linear in t",
                            "images": "zero-order hold (previous step)", "fill_stale_max_run": max_stale_run,
                            "stale_frames_filled": int(filled), "source": str(src)})
    (out / "meta.json").write_text(json.dumps(meta2, indent=2))
    return len(steps), len(tk), filled


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--fill-stale", type=int, default=3)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    src, out = Path(a.src).resolve(), Path(a.out)
    if out.exists():
        if not a.overwrite:
            sys.exit(f"{out} exists (--overwrite)")
        shutil.rmtree(out)
    eps = sorted(p for p in src.iterdir() if (p / "meta.json").exists())
    if not eps:
        sys.exit(f"no episode folders in {src}")
    out.mkdir(parents=True)
    for p in eps:
        n0, n1, filled = convert_episode(p, out / p.name, a.fps, a.fill_stale)
        print(f"{p.name}: {n0} steps -> {n1} frames at {a.fps:g} fps, {filled} stale frames filled")


if __name__ == "__main__":
    main()
