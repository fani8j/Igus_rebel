"""Build a YOLOv8 segmentation dataset: generated boxes + robot wrist-camera videos.

1. Generated boxes (synth_box.py): a box is drawn, its seam is cut by a random
   plan with sections left uncut or cut badly, so every label is exact.
2. The real carton from the overhead scene camera: its top seam and the two
   box edges next to it were annotated by hand, and cuts are drawn along them
   by random plans, so labels are exact too.
3. Real robot frames: labels are made automatically by the classical detector
   (cut_detector.py). In every frame of a top-seam cutting pass, the seam
   behind the blade is labelled along its length.

Classes:

    0 cut_ok        strong open slit in the tape      (cutting properly)
    1 partial_cut   weak slit                         (not cutting accurately)
    2 not_cut       no slit, film still closed        (missed)
    3 knife         the blade holder (fixed in the wrist camera)

Frames where the knife is not on the top seam (moving, edges, floor) are added
with only the knife labelled, so the model learns not to draw a cut line there.

For a fair test, the middle 20 % of every real cutting pass (a continuous
block, not scattered frames) and a separate set of generated boxes are kept
out of training and used for validation.

Usage:
    python make_yolo_dataset.py            # -> dataset_yolo/
"""
import os
import shutil
import time

import cv2
import numpy as np

import analyze_video as av
import cut_detector as cd
import synth_box as sb

SOURCES = [("videos/wrist.mp4", "wrist"), ("videos/right_wrist.mp4", "right_wrist")]
CLASSES = ["cut_ok", "partial_cut", "not_cut", "knife"]
STATE_TO_CLASS = {cd.CUT: 0, cd.UNCERTAIN: 1, cd.NOT_CUT: 2}
OUT = "dataset_yolo"
POS_STEP = 1      # every frame of a cutting pass (only two passes are recorded, so use them all)
NEG_STEP = 20     # every 20th frame outside the top-seam passes
VAL_BLOCK = (0.4, 0.6)  # middle part of each pass goes to validation
LINE_BAND = 0.035  # labelled cut-line width, fraction of image height
SYNTH = {"train": 400, "val": 60}  # generated boxes per split (different seeds, never shared)
REAL_BOX = {"train": 500, "val": 60}  # real scene-camera box cut by plan; val uses held-out frames


def label_mask(frame, cam):
    """Class-id mask (255 = background) for one frame of a top-seam pass."""
    h, w = frame.shape[:2]
    _, _, report, _ = cd.detect(frame, [cam["path"]], "top", cam["knife"], align=cam["align"])
    pts, labels = report["samples"][0]
    mask = np.full((h, w), 255, np.uint8)
    band = max(6, int(LINE_BAND * min(h, w)))
    for i in range(len(pts) - 1):
        if labels[i] in STATE_TO_CLASS:
            a = (int(pts[i][0] * w), int(pts[i][1] * h))
            b = (int(pts[i + 1][0] * w), int(pts[i + 1][1] * h))
            cv2.line(mask, a, b, STATE_TO_CLASS[labels[i]], band)
    return mask


def knife_polygon(frame, cam):
    h, w = frame.shape[:2]
    return cd.to_px(cam["knife"], w, h).astype(np.int32)


def add_synthetic(split, count, seed0):
    """Generated boxes cut by random plans, at random cutting progress."""
    rng = np.random.default_rng(seed0)
    for k in range(count):
        scene = sb.Scene(seed=seed0 + k)
        plan = sb.random_plan(rng)
        progress = 100.0 if rng.random() < 0.5 else float(rng.uniform(20, 100))
        knife = progress if (progress < 100 or rng.random() < 0.5) else None
        img, mask = scene.render(plan, knife_at=knife, progress=progress)
        h, w = mask.shape
        name = f"synth_{seed0 + k:05d}"
        cv2.imwrite(os.path.join(OUT, "images", split, name + ".jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        with open(os.path.join(OUT, "labels", split, name + ".txt"), "w") as f:
            f.write("\n".join(yolo_lines(mask, None, w, h)) + "\n")


CUT_ORDER = ["top", "left", "right"]  # the knife cuts the top seam first, then the two edges


def add_real_box(split, count, frames, seed0):
    """The real carton from the scene camera, its top seam and both edges cut by random plans.

    Each sample stops at a random point of the cutting sequence: lines before
    it are finished, the current line is partly done (with the knife on it),
    lines after it are not cut yet.
    """
    rng = np.random.default_rng(seed0)
    for k in range(count):
        scene = sb.RealBoxScene(int(rng.choice(list(frames))), seed=seed0 + k)
        plans = {line: sb.random_plan(rng) for line in CUT_ORDER}
        stage = int(rng.integers(0, len(CUT_ORDER) + 1))  # len(CUT_ORDER) = everything finished
        progress = {line: 100.0 if i < stage else 0.0 for i, line in enumerate(CUT_ORDER)}
        knife = None
        if stage < len(CUT_ORDER):
            line = CUT_ORDER[stage]
            progress[line] = float(rng.uniform(5, 100))
            knife = (line, progress[line])
        elif rng.random() < 0.3:
            knife = (CUT_ORDER[-1], 100.0)
        img, mask = scene.render_lines(plans, progress, knife)
        h, w = mask.shape
        name = f"scene_{seed0 + k:05d}"
        cv2.imwrite(os.path.join(OUT, "images", split, name + ".jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        with open(os.path.join(OUT, "labels", split, name + ".txt"), "w") as f:
            f.write("\n".join(yolo_lines(mask, None, w, h)) + "\n")


def yolo_lines(mask, knife_poly, w, h):
    """YOLO-seg label lines: class x1 y1 x2 y2 ... (normalised).

    The knife comes from knife_poly (real videos) or from class 3 in the mask
    (generated boxes).
    """
    lines = []
    for cls in range(3 if knife_poly is not None else 4):
        contours, _ = cv2.findContours((mask == cls).astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) < 40:
                continue
            c = cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2)
            if len(c) >= 3:
                lines.append(f"{cls} " + " ".join(f"{x / w:.5f} {y / h:.5f}" for x, y in c))
    if knife_poly is not None:
        knife = knife_poly.copy()
        knife[:, 0] = knife[:, 0].clip(0, w - 1)
        knife[:, 1] = knife[:, 1].clip(0, h - 1)
        lines.append("3 " + " ".join(f"{x / w:.5f} {y / h:.5f}" for x, y in knife))
    return lines


def top_passes(video, cam):
    cap = cv2.VideoCapture(video)
    flags = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        flags.append(cd.on_seam(frame))
    passes = av.find_passes(flags)
    kinds = [av.pass_kind(cap, cam, a, b) for a, b in passes]
    cap.release()
    return [p for p, k in zip(passes, kinds) if k == "top"], len(flags)


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    for attempt in range(10):  # OneDrive may hold a file open for a moment while syncing it
        try:
            if os.path.isdir(OUT):
                shutil.rmtree(OUT)
            break
        except PermissionError:
            time.sleep(2)
    for split in ("train", "val"):
        os.makedirs(os.path.join(OUT, "images", split))
        os.makedirs(os.path.join(OUT, "labels", split))
    counts = {"train": [0, 0], "val": [0, 0]}  # [cutting frames, other frames]
    class_px = np.zeros(3)
    for video, camera in SOURCES:
        cam = cd.CAMERAS[camera]
        passes, n = top_passes(video, cam)
        plan = {}
        for a, b in passes:
            for i in range(a, b + 1, POS_STEP):
                frac = (i - a) / max(1, b - a)
                plan[i] = ("val" if VAL_BLOCK[0] <= frac < VAL_BLOCK[1] else "train", True)
        in_pass = np.zeros(n, bool)
        for a, b in passes:
            in_pass[max(0, a - 15):b + 16] = True  # keep a margin around passes out of the negatives
        for k, i in enumerate(range(0, n, NEG_STEP)):
            if not in_pass[i]:
                plan[i] = ("val" if k % 5 == 0 else "train", False)
        cap = cv2.VideoCapture(video)
        stem = os.path.splitext(os.path.basename(video))[0]
        for i in sorted(plan):
            split, positive = plan[i]
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, frame = cap.read()
            if not ok:
                continue
            h, w = frame.shape[:2]
            mask = label_mask(frame, cam) if positive else np.full((h, w), 255, np.uint8)
            kpoly = knife_polygon(frame, cam)
            if positive:
                class_px += [(mask == c).sum() for c in range(3)]
            name = f"{stem}_{i:05d}"
            cv2.imwrite(os.path.join(OUT, "images", split, name + ".jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            with open(os.path.join(OUT, "labels", split, name + ".txt"), "w") as f:
                f.write("\n".join(yolo_lines(mask, kpoly, w, h)) + "\n")
            counts[split][0 if positive else 1] += 1
        cap.release()
        print(f"{stem}: top-seam passes {[(round(a / 15, 1), round(b / 15, 1)) for a, b in passes]}")
    add_synthetic("train", SYNTH["train"], seed0=10000)
    add_synthetic("val", SYNTH["val"], seed0=50000)
    print(f"generated boxes: {SYNTH['train']} train, {SYNTH['val']} val")
    add_real_box("train", REAL_BOX["train"], sb.SCENE_TRAIN_FRAMES, seed0=20000)
    add_real_box("val", REAL_BOX["val"], sb.SCENE_TEST_FRAMES, seed0=60000)
    print(f"real box (scene camera) cut by plan: {REAL_BOX['train']} train, {REAL_BOX['val']} val")
    with open(os.path.join(OUT, "data.yaml"), "w") as f:
        f.write(f"path: {os.path.abspath(OUT)}\ntrain: images/train\nval: images/val\nnames:\n")
        for i, c in enumerate(CLASSES):
            f.write(f"  {i}: {c}\n")
    share = 100 * class_px / max(1, class_px.sum())
    print(f"train: {counts['train'][0]} cutting + {counts['train'][1]} other frames | "
          f"val: {counts['val'][0]} cutting + {counts['val'][1]} other frames")
    print("cut-line pixels: " + ", ".join(f"{CLASSES[c]} {share[c]:.0f}%" for c in range(3)))
    print(f"wrote {OUT}/data.yaml")


if __name__ == "__main__":
    main()
