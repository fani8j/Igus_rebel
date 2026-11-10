"""Automatically segment which parts of the planned cut path are cut / not cut.

A finished cut opens a thin dark slit in the tape / stretch film; where the
film is not cut it still bridges the seam and no continuous slit is visible.
For each point along the planned cut path (which the robot already knows) the
detector averages the brightness profile across the path over a short stretch
along it, and measures how much darker a thin line is than its two sides
("slit strength", normalised by the image contrast). Averaging along the path
keeps a continuous slit and cancels random film wrinkles and glare.

Each part of the path is then labelled:
    cut properly      slit strength >= hi threshold   (green)
    not cut           slit strength <  lo threshold   (red)
    not cut properly  in between: a weak slit         (red, re-cut)
Only two colours are shown: green = cut properly, red = skipped or not cut
properly. The weak-slit state (UNCERTAIN) is kept internally, e.g. as the
"partial cut" class when building the YOLO training data.

Only the planned path and the knife region are given to the detector. The
cut / not-cut labels in examples/annotations.json are used *only* to score it.

To analyse a robot video use analyze_video.py, which builds on this module.

Usage (checks the detector against the hand-labelled example frames):
    python cut_detector.py
"""
import argparse
import json
import os
import warnings

import cv2
import numpy as np

# Slit-strength thresholds per cut type. Edge cuts sit on a box corner where a
# cut does not open a dark gap, so the signal there is weak and the uncertain
# band is wide.
PARAMS = {
    "top": {"lo": 0.15, "hi": 0.18, "confidence": "normal"},
    "edge": {"lo": 0.07, "hi": 0.11, "confidence": "LOW"},
}
CUT, NOT_CUT, UNCERTAIN, KNIFE, BACKGROUND = 1, 2, 3, 0, 255
COLORS = {KNIFE: (200, 60, 160), CUT: (0, 210, 0), NOT_CUT: (0, 0, 235), UNCERTAIN: (0, 0, 235)}
NAMES = {CUT: "cut", NOT_CUT: "NOT CUT", UNCERTAIN: "NOT cut properly"}
# Robot wrist cameras: the knife is fixed in the image, and during the top cut
# the seam runs under the blade. The knife moves up the image, so the seam
# below the blade is the part it has just cut; only that part is judged (the
# part ahead of the blade is not cut yet by definition). The path is snapped
# sideways onto the seam in every frame (`align`).
CAMERAS = {
    "wrist": {
        "knife": [[0.73, 0.43], [0.80, 0.40], [0.88, 0.36], [0.90, 0.20], [1.0, 0.15], [1.0, 0.82],
                  [0.92, 0.76], [0.80, 0.62], [0.73, 0.58]],
        "path": [[0.725, 0.58], [0.74, 1.0]],
        "behind_y": (0.60, 0.80),
        "align": 0.03,
        "kind": "top",
    },
    "right_wrist": {
        "knife": [[0.72, 0.38], [0.80, 0.35], [0.88, 0.33], [0.88, 0.15], [1.0, 0.12], [1.0, 0.85],
                  [0.90, 0.80], [0.80, 0.58], [0.72, 0.56]],
        "path": [[0.725, 0.56], [0.735, 1.0]],
        "behind_y": (0.58, 0.80),
        "align": 0.03,
        "kind": "top",
    },
}
# The knife is on the box seam when the region behind the blade shows a strong
# vertical line structure (the seam / box edge) and is warmer in colour than the
# rest of the frame (cardboard, not floor or ceiling). Measured on the recorded
# wrist videos; see seam_under_knife().
ON_SEAM = {"roi": (0.62, 0.86, 0.58, 1.0), "vert_min": 1.4, "warm_min": 5.0}


def seam_under_knife(frame):
    """Return (vertical-structure ratio, relative warmth) of the region behind the blade."""
    h, w = frame.shape[:2]
    x0, x1, y0, y1 = ON_SEAM["roi"]
    roi = frame[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gx = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0)).mean()
    gy = np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1)).mean()
    warm_roi = float(np.mean(roi[..., 2].astype(np.float32) - roi[..., 0]))
    warm_all = float(np.mean(frame[..., 2].astype(np.float32) - frame[..., 0]))
    return gx / (gy + 1e-3), warm_roi - warm_all


def on_seam(frame):
    vert, warm = seam_under_knife(frame)
    return vert > ON_SEAM["vert_min"] and warm > ON_SEAM["warm_min"]


# A top seam has the same cardboard / tape on both sides of the cut line; a box
# edge (corner) has the box face on one side and the side face on the other, so
# the brightness steps across the line. Measured on the recorded wrist videos:
# top passes 0.10-0.12, edge passes 0.28-0.33 (median per pass).
EDGE_STEP_MIN = 0.2


# The knife / camera mounting can change (it did between the 20 and 22 Oct
# recordings: the seam sat ~55 px further right). So for every cutting pass the
# seam position is measured first: the median, over several frames of the pass,
# of the strongest line within +-SEAM_SEARCH of the planned path. The knife rides
# in the seam, so the seam is the line that stays put while the box moves under it.
SEAM_SEARCH = 0.09


def knife_mask(cam, h, w):
    ignore = np.zeros((h, w), np.uint8)
    cv2.fillPoly(ignore, [to_px(cam["knife"], w, h).astype(np.int32)], 1)
    return ignore


def seam_offset(frames, cam):
    """Where the seam sits in these frames of one pass: sideways shift of the planned
    path, as a fraction of the image size (0 = as calibrated)."""
    shifts = []
    for frame in frames:
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        s, val = best_shift(gray, to_px(cam["path"], w, h), knife_mask(cam, h, w), SEAM_SEARCH, step=0.006)
        if np.isfinite(val):
            shifts.append(s)
    return float(np.median(shifts)) if shifts else 0.0


def side_step(frame, cam, shift=0.0):
    """Brightness difference between the two sides of the (aligned) cut line."""
    h, w = frame.shape[:2]
    m = min(h, w)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ignore = knife_mask(cam, h, w)
    line = align_path(gray, shift_line(to_px(cam["path"], w, h), shift, h, w), ignore, cam["align"])
    pts, dirs = sample_path(line, step=6)
    normals = np.stack([-dirs[:, 1], dirs[:, 0]], 1)
    off, band = int(0.05 * m), int(0.03 * m)
    g = gray.astype(np.float32)

    def side(sign):
        vals = []
        for o in range(off - band // 2, off + band // 2, 3):
            q = pts + sign * normals * o
            vals.append(g[q[:, 1].astype(int).clip(0, h - 1), q[:, 0].astype(int).clip(0, w - 1)])
        return np.mean(vals, 0)

    contrast = max(1.0, float(np.percentile(g, 95) - np.percentile(g, 5)))
    return float(np.median(np.abs(side(1) - side(-1))) / contrast)


def to_px(points, w, h):
    return np.array([[x * w, y * h] for x, y in points], np.float32)


def sample_path(line_px, step=3):
    """Points along a polyline every `step` px, with the unit direction at each."""
    pts, dirs = [], []
    for a, b in zip(line_px[:-1], line_px[1:]):
        length = float(np.linalg.norm(b - a))
        if length < 1:
            continue
        d = (b - a) / length
        for t in np.arange(0, length, step):
            pts.append(a + d * t)
            dirs.append(d)
    return np.array(pts, np.float32), np.array(dirs, np.float32)


def prepare(gray):
    """Per-frame work shared by every slit_strength call: blurred image and contrast."""
    # 5th / 95th percentile from the histogram: same values as np.percentile, much faster
    cdf = np.cumsum(cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel())

    def pct(q):
        rank = q * (cdf[-1] - 1)
        k = int(rank)
        a, b = np.searchsorted(cdf, k + 1), np.searchsorted(cdf, k + 2)
        return a + (b - a) * (rank - k)
    contrast = max(1.0, float(pct(0.95) - pct(0.05)))
    return cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32), contrast


def slit_strength(gray, pts, dirs, ignore, prep=None):
    """Oriented thin-dark-line strength at each path point (NaN where ignored)."""
    h, w = gray.shape
    m = min(h, w)
    band, gap, half = int(0.015 * m), max(3, int(0.009 * m)), int(0.033 * m)
    g, contrast = prep or prepare(gray)
    normals = np.stack([-dirs[:, 1], dirs[:, 0]], 1)
    offs = np.arange(-band - gap, band + gap + 1, dtype=np.float32)
    along = np.arange(-half, half + 1, 2, dtype=np.float32)
    xy = (pts[:, None, None, :] + normals[:, None, None, :] * offs[None, :, None, None]
          + dirs[:, None, None, :] * along[None, None, :, None])
    x, y = np.rint(xy[..., 0]).astype(int), np.rint(xy[..., 1]).astype(int)
    inside = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    vals = np.where(inside, g[y.clip(0, h - 1), x.clip(0, w - 1)], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        profile = np.nanmean(vals, axis=2)  # brightness across the path
        ridge = (profile[:, :-2 * gap] + profile[:, 2 * gap:]) / 2 - profile[:, gap:-gap]
        score = np.nanmax(ridge, axis=1)
    px, py = pts[:, 0].astype(int), pts[:, 1].astype(int)
    on_image = (px >= 0) & (px < w) & (py >= 0) & (py < h)
    score[~on_image] = np.nan
    score[on_image] = np.where(ignore[py[on_image], px[on_image]] == 1, np.nan, score[on_image])
    return score / contrast


def smooth(values, window=15):
    """Running median over neighbouring valid samples."""
    out = values.copy()
    valid = np.where(np.isfinite(values))[0]
    v, k = values[valid], window // 2
    if len(v) >= window:  # full windows at once; the shorter windows at both ends one by one
        out[valid[k:len(v) - k]] = np.median(np.lib.stride_tricks.sliding_window_view(v, window), axis=1)
        ends = list(range(k)) + list(range(len(v) - k, len(v)))
    else:
        ends = range(len(v))
    for i in ends:
        out[valid[i]] = np.median(v[max(0, i - k):min(len(v), i + k + 1)])
    return out


def label(scores, prm):
    lab = np.full(len(scores), BACKGROUND, np.uint8)
    ok = np.isfinite(scores)
    lab[ok & (scores >= prm["hi"])] = CUT
    lab[ok & (scores < prm["lo"])] = NOT_CUT
    lab[ok & (scores >= prm["lo"]) & (scores < prm["hi"])] = UNCERTAIN
    return lab


def runs(labels):
    """Collapse labels into (start_frac, end_frac, name) runs along the path."""
    out, n = [], len(labels)
    for i, lab in enumerate(labels):
        if lab == BACKGROUND:
            continue
        if out and out[-1][2] == lab and out[-1][1] == i - 1:
            out[-1][1] = i
        else:
            out.append([i, i, lab])
    return [(round(a / n, 2), round((b + 1) / n, 2), NAMES[lab]) for a, b, lab in out]


def shift_line(line_px, shift, h, w):
    """The polyline moved sideways by `shift` (fraction of the image size)."""
    d = line_px[-1] - line_px[0]
    normal = np.array([-d[1], d[0]], np.float32) / max(1e-6, float(np.linalg.norm(d)))
    return line_px + normal * shift * min(h, w)


def best_shift(gray, line_px, ignore, max_shift, step=0.004, prep=None):
    """Sideways shift (fraction of the image size, within +-max_shift) that puts the
    path on the strongest line, and that line's mean strength.

    Same measure as slit_strength on each shifted path, but the brightness profile
    across the path is sampled once, wide enough for every shift, and each shift
    takes its slice of it (whole-pixel shifts): many times faster.
    """
    h, w = gray.shape
    m = min(h, w)
    g, contrast = prep or prepare(gray)
    pts, dirs = sample_path(line_px)
    if len(pts) == 0:
        return 0.0, -np.inf
    band, gap, half = int(0.015 * m), max(3, int(0.009 * m)), int(0.033 * m)
    shifts = np.unique(np.rint(np.arange(-max_shift, max_shift + 1e-9, step) * m).astype(int))
    reach = int(np.abs(shifts).max()) + band + gap
    normals = np.stack([-dirs[:, 1], dirs[:, 0]], 1)
    offs = np.arange(-reach, reach + 1, dtype=np.float32)
    along = np.arange(-half, half + 1, 2, dtype=np.float32)
    xy = (pts[:, None, None, :] + normals[:, None, None, :] * offs[None, :, None, None]
          + dirs[:, None, None, :] * along[None, None, :, None])
    x, y = np.rint(xy[..., 0]).astype(int), np.rint(xy[..., 1]).astype(int)
    inside = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    vals = np.where(inside, g[y.clip(0, h - 1), x.clip(0, w - 1)], np.nan)
    best, best_s = -np.inf, 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        profile = np.nanmean(vals, axis=2)
        ridge = (profile[:, :-2 * gap] + profile[:, 2 * gap:]) / 2 - profile[:, gap:-gap]
        ridge_offs = offs[gap:-gap]
        for s in shifts:
            score = np.nanmax(ridge[:, np.abs(ridge_offs - s) <= band], axis=1)
            q = pts + normals * s
            px, py = q[:, 0].astype(int), q[:, 1].astype(int)
            ok = (px >= 0) & (px < w) & (py >= 0) & (py < h)
            ok[ok] = ignore[py[ok], px[ok]] == 0
            val = np.nanmean(np.where(ok, score, np.nan)) / contrast
            if np.isfinite(val) and val > best:
                best, best_s = val, float(s) / m
    return best_s, best


def align_path(gray, line_px, ignore, max_shift, step=0.004, prep=None):
    """Shift the path sideways (up to max_shift of the image size) onto the seam.

    The seam wanders slightly relative to the planned path, so the shift with
    the strongest line response along the path is used.
    """
    s, _ = best_shift(gray, line_px, ignore, max_shift, step, prep)
    return shift_line(line_px, s, *gray.shape)


def detect(img, paths, kind, knife=None, align=0.0, shift=0.0):
    """paths: list of normalised polylines. Returns (overlay, label_mask, report, per_path_labels).

    shift: move each path sideways first by this fraction of the image size
    (where the seam sits for this camera / knife mounting, see seam_offset).
    align: if > 0, each path may then be shifted further by up to this
    fraction to sit on the seam in this frame.
    """
    h, w = img.shape[:2]
    prm = PARAMS[kind]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    prep = prepare(gray)
    ignore = np.zeros((h, w), np.uint8)
    if knife:
        cv2.fillPoly(ignore, [to_px(knife, w, h).astype(np.int32)], 1)
    band = max(6, int(0.012 * min(h, w)))
    mask = np.full((h, w), BACKGROUND, np.uint8)
    report = {"kind": kind, "confidence": prm["confidence"], "paths": []}
    all_labels = []
    for line in paths:
        line_px = shift_line(to_px(line, w, h), shift, h, w)
        if align > 0:
            line_px = align_path(gray, line_px, ignore, align, prep=prep)
        pts, dirs = sample_path(line_px)
        labels = label(smooth(slit_strength(gray, pts, dirs, ignore, prep)), prm)
        for i in range(len(pts) - 1):
            if labels[i] != BACKGROUND:
                a, b = tuple(pts[i].astype(int)), tuple(pts[i + 1].astype(int))
                cv2.line(mask, a, b, int(labels[i]), band)
        report["paths"].append(runs(labels))
        report.setdefault("samples", []).append((pts / [w, h], labels))
        all_labels.append(labels)
    if knife:
        mask[ignore == 1] = KNIFE
    decided = np.concatenate(all_labels) if all_labels else np.array([], np.uint8)
    decided = decided[decided != BACKGROUND]
    for lab, key in ((CUT, "cut_percent"), (NOT_CUT, "not_cut_percent"), (UNCERTAIN, "uncertain_percent")):
        report[key] = round(100.0 * np.mean(decided == lab), 1) if len(decided) else None

    # colour only where something was drawn (the knife and the cut lines), not the whole frame
    out = img.copy()
    drawn = [np.concatenate([p for p, _ in report.get("samples", [])]) * [w, h]] if all_labels else []
    if knife:
        drawn.append(to_px(knife, w, h))
    if drawn:
        x0, y0, bw, bh = cv2.boundingRect(np.concatenate(drawn).astype(np.int32))
        x0, y0 = max(0, x0 - band), max(0, y0 - band)
        x1, y1 = min(w, x0 + bw + 2 * band), min(h, y0 + bh + 2 * band)
        roi, mroi = out[y0:y1, x0:x1], mask[y0:y1, x0:x1]
        for lab, col in COLORS.items():
            sel = mroi == lab
            if sel.any():
                alpha = 0.45 if lab == KNIFE else 0.85
                tint = cv2.add(cv2.convertScaleAbs(roi, alpha=1 - alpha), tuple(c * alpha for c in col) + (0,))
                np.copyto(roi, tint, where=sel[..., None])
    return out, mask, report, all_labels


def add_header(img, lines, longest=900):
    s = longest / max(img.shape[:2])
    img = cv2.resize(img, None, fx=s, fy=s)
    bar = np.full((14 + 30 * len(lines), img.shape[1], 3), 255, np.uint8)
    for k, (text, col) in enumerate(lines):
        cv2.putText(bar, text, (8, 32 + 30 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.62, col, 2)
    return np.vstack([bar, img])


def status_lines(name, report):
    red = round((report["not_cut_percent"] or 0) + (report["uncertain_percent"] or 0), 1)
    full = red == 0
    return [(name, (0, 0, 0)),
            (f"{report['kind']}: cut properly {report['cut_percent']}% | NOT cut properly {red}%", (0, 0, 0)),
            ("FULLY CUT" if full else "NOT FULLY CUT - re-cut red parts", (0, 150, 0) if full else (0, 0, 220)),
            (f"confidence: {report['confidence']}", (0, 0, 0))]


def read_frame(video, time_s):
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(time_s * cap.get(cv2.CAP_PROP_FPS)))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"cannot read {video} at {time_s}s")
    return frame


def legend(width):
    bar = np.full((50, width, 3), 255, np.uint8)
    x = 10
    for lab, name in ((CUT, "cut properly"), (NOT_CUT, "skipped / not cut properly"), (KNIFE, "knife (ignored)")):
        cv2.rectangle(bar, (x, 15), (x + 26, 37), COLORS[lab], -1)
        cv2.putText(bar, name, (x + 32, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        x += 70 + 13 * len(name)
    return bar


def run_frames(out_dir):
    with open("examples/annotations.json") as f:
        frames = json.load(f)["frames"]
    os.makedirs(out_dir, exist_ok=True)
    panels, reports = [], {}
    for ann in frames:
        kind = "top" if "top_cut" in ann else "edge"
        cut_lines, not_lines = ann[f"{kind}_cut"], ann[f"{kind}_not_cut"]
        img = read_frame(ann["video"], ann["time_s"])
        # the detector only sees the planned path (both labelled parts, labels dropped)
        overlay, mask, report, labels = detect(img, cut_lines + not_lines, kind, ann.get("knife"))
        truth = [CUT] * len(cut_lines) + [NOT_CUT] * len(not_lines)
        report["agreement_with_hand_labels"] = agreement(labels, truth)
        reports[ann["name"]] = report
        panel = add_header(overlay, status_lines(ann["name"], report))
        cv2.imwrite(os.path.join(out_dir, ann["name"] + "_detected.jpg"), panel)
        cv2.imwrite(os.path.join(out_dir, ann["name"] + "_mask.png"), mask)
        panels.append(panel)
        red = report["not_cut_percent"] + report["uncertain_percent"]
        print(f"{ann['name']:22s} cut properly {report['cut_percent']:5.1f}%  not cut properly {red:5.1f}%  "
              f"| agreement {report['agreement_with_hand_labels']}")
    for r in reports.values():
        r.pop("samples", None)
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(reports, f, indent=2)
    height = 780
    row = [cv2.resize(p, (int(p.shape[1] * height / p.shape[0]), height)) for p in panels]
    gap = np.full((height, 10, 3), 255, np.uint8)
    sheet = row[0]
    for p in row[1:]:
        sheet = np.hstack([sheet, gap, p])
    cv2.imwrite(os.path.join(out_dir, "summary_detected.jpg"), np.vstack([sheet, legend(sheet.shape[1])]))


def agreement(labels, truth):
    """Of the samples the detector decided (cut / not cut), % matching the hand label."""
    hits = decided = unsure = 0
    for lab, t in zip(labels, truth):
        decided += int(np.sum((lab == CUT) | (lab == NOT_CUT)))
        hits += int(np.sum(lab == t))
        unsure += int(np.sum(lab == UNCERTAIN))
    pct = round(100.0 * hits / decided, 1) if decided else None
    return {"agree_percent_of_decided": pct, "uncertain_samples": unsure}


def behind_knife(report, y_range):
    """Majority label of the path just behind the knife, or None if nothing decided there."""
    pts, labels = report["samples"][0]
    sel = (pts[:, 1] >= y_range[0]) & (pts[:, 1] <= y_range[1]) & (labels != BACKGROUND)
    if not sel.any():
        return None
    values, counts = np.unique(labels[sel], return_counts=True)
    return int(values[np.argmax(counts)])


def missed_ranges(history, start, fps, min_frames=3):
    """Time ranges (s) where the seam behind the knife stayed not cut for >= min_frames."""
    out, run = [], None
    for i, lab in enumerate(history + [None]):
        if lab == NOT_CUT:
            run = i if run is None else run
        elif run is not None:
            if i - run >= min_frames:
                out.append((round(start + run / fps, 1), round(start + i / fps, 1)))
            run = None
    return out


def frame_lines(title, behind, pct):
    verdict = {CUT: ("seam behind knife: CUT", (0, 150, 0)),
               NOT_CUT: ("seam behind knife: NOT CUT", (0, 0, 220)),
               UNCERTAIN: ("seam behind knife: NOT CUT PROPERLY", (0, 0, 220))}.get(
                   behind, ("seam behind knife: not visible", (90, 90, 90)))
    return [(title, (0, 0, 0)), verdict, (f"this pass so far: {pct}% cut", (0, 0, 0))]


def main():
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # paths below are relative to the project
    run_frames(os.path.join("results", "auto"))


if __name__ == "__main__":
    main()
