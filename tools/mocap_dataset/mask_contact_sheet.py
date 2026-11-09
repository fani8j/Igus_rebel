#!/usr/bin/env python3
"""QA contact sheet: mask overlays (robot red, box green) on frames across several episodes for one camera.

    python3 mask_contact_sheet.py SRC_ROOT MASK_ROOT CAMERA OUT.png [--n 12] [--episodes 0,5,11]
Runs in either python (needs numpy + cv2; reads parquet only through segment_masks.load_episodes)."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import segment_masks as sm


def overlay(img, lab):
    o = img.astype(np.float32).copy()
    for c, col in ((1, (255, 40, 40)), (2, (40, 255, 40))):
        m = lab == c
        o[m] = 0.45 * o[m] + 0.55 * np.array(col, np.float32)
    edge = cv2.morphologyEx((lab > 0).astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    o[edge] = (255, 255, 0)
    return o.astype(np.uint8)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("masks"); ap.add_argument("camera"); ap.add_argument("out")
    ap.add_argument("--n", type=int, default=12); ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--episodes", default="")
    a = ap.parse_args(argv)
    eps = sm.load_episodes(Path(a.src))
    have = [e for e in eps if (Path(a.masks) / e["name"] / a.camera / "labels.npz").exists()]
    if a.episodes:
        want = {int(x) for x in a.episodes.split(",")}
        have = [e for e in have if e["episode_index"] in want]
    rng = np.random.default_rng(0)
    cells = []
    per = int(np.ceil(a.n / max(len(have), 1)))
    for e in have:
        lab = np.load(Path(a.masks) / e["name"] / a.camera / "labels.npz")["labels"]
        n = len(lab)
        idx = np.sort(rng.choice(n, size=min(per, n), replace=False))
        frames = sm.read_episode_frames(e["cams"][a.camera]["path"], e["cams"][a.camera]["from_ts"], e["length"])
        for i in idx:
            c = overlay(frames[i], lab[i])
            cv2.putText(c, f"{e['name'][-12:]} f{i}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cells.append(cv2.resize(c, (480, 360)))
    cells = cells[:a.n]
    while len(cells) % a.cols:
        cells.append(np.zeros_like(cells[0]))
    rows = [np.concatenate(cells[i:i + a.cols], 1) for i in range(0, len(cells), a.cols)]
    cv2.imwrite(a.out, np.concatenate(rows, 0)[:, :, ::-1])
    print(a.out)

if __name__ == "__main__":
    main()
