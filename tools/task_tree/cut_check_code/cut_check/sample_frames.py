"""Sample frames from the box-cutting videos into an image-segmentation dataset.

For every time segment in frame_sampling.json the video is split into windows
of `every_s` seconds; the sharpest frame of each window is kept, unless it is
almost identical to the previous kept frame. Frames are saved at full
resolution, grouped by cut phase, with a frames.csv describing each image.

Usage:
    python sample_frames.py                 # uses frame_sampling.json
    python sample_frames.py --overwrite     # regenerate an existing dataset
"""
import argparse
import csv
import json
import os
import shutil

import cv2
import numpy as np


def sharpness(gray):
    """Variance of the Laplacian: higher means less motion blur."""
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def sample_segment(cap, fps, start_s, end_s, every_s, min_diff):
    """Yield (frame_idx, frame, sharpness) for the frames kept in one segment."""
    first, last = int(start_s * fps), int(end_s * fps)
    window = max(1, round(every_s * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    prev_small, best = None, None
    for idx in range(first, last + 1):
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        score = sharpness(gray)
        if best is None or score > best[2]:
            best = (idx, frame, score, gray)
        if (idx - first + 1) % window == 0 or idx == last:
            small = cv2.resize(best[3], (64, 40)).astype(np.float32)
            if prev_small is None or np.abs(small - prev_small).mean() >= min_diff:
                yield best[:3]
                prev_small = small
            best = None


def contact_sheet(paths, out_path, cols=8, size=(240, 150)):
    thumbs = [cv2.resize(cv2.imread(p), size) for p in paths[:cols * 6]]
    if not thumbs:
        return
    while len(thumbs) % cols:
        thumbs.append(np.zeros_like(thumbs[0]))
    rows = [np.hstack(thumbs[i:i + cols]) for i in range(0, len(thumbs), cols)]
    cv2.imwrite(out_path, np.vstack(rows))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", help="default: frame_sampling.json in the project folder")
    ap.add_argument("--overwrite", action="store_true",
                    help="delete previously sampled images before sampling")
    args = ap.parse_args()
    project = os.path.dirname(os.path.abspath(__file__))
    args.config = os.path.abspath(args.config) if args.config else os.path.join(project, "frame_sampling.json")
    os.chdir(project)  # paths below are relative to the project

    with open(args.config) as f:
        cfg = json.load(f)
    out_dir = cfg["output_dir"]
    img_dir = os.path.join(out_dir, "images")
    if os.path.isdir(img_dir) and os.listdir(img_dir):
        if not args.overwrite:
            raise SystemExit(f"{img_dir} already has images; rerun with --overwrite to replace them")
        shutil.rmtree(img_dir)

    depth_text = {phase: "-".join(str(d) for d in spec["depth_mm"])
                  for phase, spec in cfg["cut_spec"].items()}
    rows, by_phase = [], {}
    for video in cfg["videos"]:
        cap = cv2.VideoCapture(video["file"])
        if not cap.isOpened():
            raise SystemExit(f"cannot open {video['file']}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        stem = os.path.splitext(os.path.basename(video["file"]))[0]
        every_s = video.get("every_s", cfg["every_s"])
        for seg in video["segments"]:
            phase = seg["phase"]
            os.makedirs(os.path.join(img_dir, phase), exist_ok=True)
            kept = 0
            for idx, frame, score in sample_segment(cap, fps, seg["start"], seg["end"],
                                                    every_s, cfg["min_diff"]):
                rel = os.path.join("images", phase, f"{stem}_{idx:05d}.jpg")
                path = os.path.join(out_dir, rel)
                cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                rows.append({"image": rel.replace(os.sep, "/"), "video": stem, "phase": phase,
                             "frame": idx, "time_s": round(idx / fps, 2),
                             "sharpness": round(score, 1),
                             "target_depth_mm": depth_text.get(phase, "")})
                by_phase.setdefault(phase, []).append(path)
                kept += 1
            print(f"{stem:12s} {phase:9s} {seg['start']:>3}-{seg['end']:<3}s -> {kept} frames")
        cap.release()

    with open(os.path.join(out_dir, "frames.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(os.path.join(out_dir, "classes.txt"), "w") as f:
        f.write("\n".join(cfg["classes"]) + "\n")
    for phase, paths in by_phase.items():
        contact_sheet(paths, os.path.join(out_dir, f"preview_{phase}.jpg"))
    print(f"total {len(rows)} frames -> {out_dir}/")


if __name__ == "__main__":
    main()
