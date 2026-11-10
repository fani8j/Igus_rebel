"""Run the trained YOLOv8 cut-segmentation model and draw its result.

Classes: 0 cut_ok, 1 partial_cut, 2 not_cut, 3 knife.
Shown in two colours: green = cut properly, red = skipped or not cut properly
(partial_cut and not_cut both mean "re-cut"); the knife is purple.
"""
import os

import cv2
import numpy as np

CUT_OK, PARTIAL, NOT_CUT, KNIFE, BACKGROUND = 0, 1, 2, 3, 255
COLORS = {CUT_OK: (40, 190, 40), PARTIAL: (40, 40, 230), NOT_CUT: (40, 40, 230), KNIFE: (200, 60, 160)}
NAMES = {CUT_OK: "cut properly", PARTIAL: "NOT cut properly", NOT_CUT: "NOT cut"}
DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "cut_seg_yolov8n.pt")


def load_model(path=DEFAULT_MODEL):
    from ultralytics import YOLO  # imported here so the rest of the project runs without it
    if not os.path.isfile(path):
        raise SystemExit(f"model not found: {path} - run train_yolo.py first")
    return YOLO(path)


def segment(model, frame, imgsz=512, conf=0.25, merge_partial=True):
    """Class mask (BACKGROUND where nothing was found) for one BGR frame.

    Where detections overlap, the worse state wins (not cut over not accurate
    over cut OK), and within one class the most confident one: a missed part
    left on the box stops it from opening, so a confident "not cut" detection
    must never be hidden under a neighbouring "cut" detection. With merge_partial
    (the default) "not cut accurately" is reported as "not cut": in the
    overhead camera a faint bad cut and an untouched seam look almost the same,
    so the model separates them unreliably, while cut vs not cut is reliable.
    """
    h, w = frame.shape[:2]
    mask = np.full((h, w), BACKGROUND, np.uint8)
    result = model.predict(frame, imgsz=imgsz, conf=conf, device="cpu", verbose=False)[0]
    if result.masks is None:
        return mask
    classes = result.boxes.cls.cpu().numpy().astype(int)
    scores = result.boxes.conf.cpu().numpy()
    # drawn in order, later on top: cut OK, then not accurate, then not cut, then the knife;
    # within a class the least confident first
    rank = {CUT_OK: 0, PARTIAL: 1, NOT_CUT: 2, KNIFE: 3}
    order = sorted(range(len(classes)), key=lambda i: (rank[int(classes[i])], scores[i]))
    for i in order:
        poly = result.masks.xy[i]
        if len(poly) >= 3:
            cls = NOT_CUT if merge_partial and classes[i] == PARTIAL else int(classes[i])
            cv2.fillPoly(mask, [poly.astype(np.int32)], cls)
    return mask


def draw(frame, mask):
    out = frame.copy()
    for cls, col in COLORS.items():
        sel = mask == cls
        alpha = 0.35 if cls == KNIFE else 0.8
        out[sel] = (out[sel] * (1 - alpha) + np.array(col) * alpha).astype(np.uint8)
    return out


def line_share(mask):
    """% of the visible cut line that is cut OK / not accurate / not cut (None if no line)."""
    counts = np.array([(mask == c).sum() for c in (CUT_OK, PARTIAL, NOT_CUT)], np.float64)
    if counts.sum() == 0:
        return None
    return 100 * counts / counts.sum()


def seam_profile(mask, start_xy, end_xy, bins=100):
    """Class along the seam from start_xy to end_xy: one label per % (BACKGROUND if none).

    Every cut-line pixel is projected onto the start-end axis; each 1 % bin
    takes the most common class of its pixels.
    """
    ys, xs = np.nonzero(mask <= NOT_CUT)
    labels = np.full(bins, BACKGROUND, np.uint8)
    if len(xs) == 0:
        return labels
    a, b = np.float32(start_xy), np.float32(end_xy)
    axis = b - a
    t = ((np.stack([xs, ys], 1) - a) @ axis) / float(axis @ axis)
    idx = np.clip((t * bins).astype(int), 0, bins - 1)
    inside = (t >= 0) & (t <= 1)
    cls = mask[ys, xs]
    for k in range(bins):
        sel = inside & (idx == k)
        if sel.any():
            labels[k] = np.bincount(cls[sel], minlength=3)[:3].argmax()
    return labels


def _assign_to_lines(mask, lines):
    """For every cut-line pixel: its nearest line, distance to it, and position along every line."""
    ys_, xs = np.nonzero(mask <= NOT_CUT)
    pts = np.stack([xs, ys_], 1).astype(np.float32)
    dists, ts = [], []
    for name in lines:
        a, b = (np.float32(p) for p in lines[name])
        axis = b - a
        t = ((pts - a) @ axis) / float(axis @ axis)
        closest = a + np.clip(t, 0, 1)[:, None] * axis
        dists.append(np.linalg.norm(pts - closest, axis=1))
        ts.append(t)
    dists = np.stack(dists) if lines else np.zeros((0, len(xs)))
    nearest = dists.argmin(0) if len(xs) else np.zeros(0, int)
    return ys_, xs, nearest, dists, ts


def hide_unreached(mask, lines, progress, max_dist=30, lag=3.0):
    """Blank the result on line parts the knife has not reached yet.

    progress: {name: % of the line already cut}. A line at 0 % shows nothing;
    the line being cut shows only the part behind the knife, minus the last
    `lag` % right at the blade, where the fresh cut is still under the knife.
    """
    out = mask.copy()
    ys_, xs, nearest, dists, ts = _assign_to_lines(mask, lines)
    if len(xs) == 0:
        return out
    hide = np.zeros(len(xs), bool)
    for k, name in enumerate(lines):
        mine = (nearest == k) & (dists[k] <= max_dist)
        p = progress.get(name, 100.0)
        if p <= 0:
            hide |= mine  # knife has not started this line
        elif p < 100:
            hide |= mine & (ts[k] * 100 > p - lag)
    hide |= dists.min(0) > max_dist  # not on any line
    out[ys_[hide], xs[hide]] = BACKGROUND
    return out


def line_profiles(mask, lines, bins=100, max_dist=30):
    """seam_profile for several lines at once: {name: labels per %}.

    lines: {name: (start_xy, end_xy)}. Each cut-line pixel counts only for
    the nearest line (and only within max_dist px of it), so where the top
    seam meets an edge the two do not leak into each other.
    """
    out = {name: np.full(bins, BACKGROUND, np.uint8) for name in lines}
    ys_, xs, nearest, dists, ts = _assign_to_lines(mask, lines)
    if len(xs) == 0:
        return out
    names = list(lines)
    cls = mask[ys_, xs]
    for k, name in enumerate(names):
        t = ts[k]
        sel = (nearest == k) & (dists[k] <= max_dist) & (t >= 0) & (t <= 1)
        idx = np.clip((t[sel] * bins).astype(int), 0, bins - 1)
        c = cls[sel]
        for b in range(bins):
            hit = c[idx == b]
            if len(hit):
                out[name][b] = np.bincount(hit, minlength=3)[:3].argmax()
    return out


def fill_gaps(labels):
    """Bins with no detected line take the label of the nearest bin that has one."""
    known = np.nonzero(labels != BACKGROUND)[0]
    if len(known) == 0:
        return labels
    out = labels.copy()
    for k in np.nonzero(labels == BACKGROUND)[0]:
        out[k] = labels[known[np.argmin(np.abs(known - k))]]
    return out


def merge_short(labels, min_len=3):
    """Merge runs shorter than min_len % into their longer neighbour.

    Where the top seam meets an edge, a few % of one line can pick up the
    other line's colour; such slivers are not real sections.
    """
    out = labels.copy()
    while True:
        runs = sections(out)
        short = [r for r in runs if r[1] - r[0] < min_len]
        if not short or len(runs) == 1:
            return out
        k = runs.index(short[0])
        nb = [runs[j] for j in (k - 1, k + 1) if 0 <= j < len(runs)]
        a, b, _ = short[0]
        out[a:b] = max(nb, key=lambda r: r[1] - r[0])[2]


def sections(labels):
    """[(from %, to %, class)] runs of equal labels."""
    out, start = [], 0
    for k in range(1, len(labels) + 1):
        if k == len(labels) or labels[k] != labels[start]:
            out.append((start, k, int(labels[start])))
            start = k
    return out


def put(img, text, org, scale=0.6, color=(30, 30, 30), thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_bar(img, labels, x0, x1, y, height=40, title=""):
    """A seam bar: one coloured block per % along the seam."""
    n = len(labels)
    for k, lab in enumerate(labels):
        a = int(x0 + (x1 - x0) * k / n)
        b = int(x0 + (x1 - x0) * (k + 1) / n)
        cv2.rectangle(img, (a, y), (b, y + height), COLORS.get(int(lab), (200, 200, 200)), -1)
    cv2.rectangle(img, (x0, y), (x1, y + height), (30, 30, 30), 1)
    for p in (0, 25, 50, 75, 100):
        x = int(x0 + (x1 - x0) * p / 100)
        cv2.line(img, (x, y + height), (x, y + height + 6), (30, 30, 30), 1)
        put(img, f"{p}%", (x - 14, y + height + 22), 0.45, (110, 110, 110))
    if title:
        put(img, title, (x0, y - 10), 0.6, (30, 30, 30), 2)
