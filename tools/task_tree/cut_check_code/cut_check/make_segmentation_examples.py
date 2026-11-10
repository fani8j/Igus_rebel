"""Build example segmentation masks for cut / not-cut detection from the videos.

Each frame listed in examples/annotations.json is read straight from its video.
The rough box polygon is refined to the real box outline with GrabCut; the knife
is a polygon and the cut path is a polyline drawn as a thin band, labelled
cut or not cut. For every frame the script writes:

    <name>_image.jpg    the original frame
    <name>_mask.png     label map (pixel value = class id, 255 = background)
    <name>_overlay.jpg  colour mask over the frame, with the cut coverage
    <name>.txt          YOLO-seg labels (class x1 y1 x2 y2 ... normalised)

plus summary.jpg with all overlays side by side.

Usage:
    python make_segmentation_examples.py
"""
import json
import os

import cv2
import numpy as np

CLASSES = ["box", "knife", "top_cut", "top_not_cut", "edge_cut", "edge_not_cut"]
COLORS = {  # BGR
    "box": (180, 130, 70),
    "knife": (200, 60, 160),
    "top_cut": (0, 200, 0),
    "top_not_cut": (0, 0, 230),
    "edge_cut": (170, 230, 0),
    "edge_not_cut": (0, 140, 255),
}
BACKGROUND = 255
CUT_PAIRS = [("top_cut", "top_not_cut"), ("edge_cut", "edge_not_cut")]


def read_frame(video, time_s):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(time_s * cap.get(cv2.CAP_PROP_FPS)))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"cannot read {video} at {time_s}s")
    return frame


def to_px(points, w, h):
    return np.array([[x * w, y * h] for x, y in points], np.int32)


def refine_box(img, polygon):
    """GrabCut the box out of the image, seeded by a rough polygon."""
    h, w = img.shape[:2]
    scale = 640 / max(h, w)
    small = cv2.resize(img, None, fx=scale, fy=scale)
    sh, sw = small.shape[:2]
    rough = np.zeros((sh, sw), np.uint8)
    cv2.fillPoly(rough, [to_px(polygon, sw, sh)], 1)
    k = max(3, sw // 40)
    sure_fg = cv2.erode(rough, np.ones((k, k), np.uint8))
    maybe = cv2.dilate(rough, np.ones((k, k), np.uint8))
    gc = np.full((sh, sw), cv2.GC_BGD, np.uint8)
    gc[maybe == 1] = cv2.GC_PR_BGD
    gc[rough == 1] = cv2.GC_PR_FGD
    gc[sure_fg == 1] = cv2.GC_FGD
    cv2.grabCut(small, gc, None, np.zeros((1, 65)), np.zeros((1, 65)), 5, cv2.GC_INIT_WITH_MASK)
    box = np.isin(gc, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
    box = cv2.morphologyEx(box, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return cv2.resize(box, (w, h), interpolation=cv2.INTER_NEAREST)


def build_mask(img, ann):
    h, w = img.shape[:2]
    mask = np.full((h, w), BACKGROUND, np.uint8)
    if "box" in ann:
        mask[refine_box(img, ann["box"]) == 1] = CLASSES.index("box")
    band = max(8, int(0.012 * min(h, w)))
    lengths = {}
    for name in ("top_cut", "top_not_cut", "edge_cut", "edge_not_cut"):
        for line in ann.get(name, []):
            pts = to_px(line, w, h)
            cv2.polylines(mask, [pts], False, CLASSES.index(name), band)
            lengths[name] = lengths.get(name, 0) + cv2.arcLength(pts.astype(np.float32), False)
    if "knife" in ann:  # the knife occludes everything behind it
        cv2.fillPoly(mask, [to_px(ann["knife"], w, h)], CLASSES.index("knife"))
    return mask, lengths


def coverage_text(lengths):
    parts = []
    for cut, not_cut in CUT_PAIRS:
        total = lengths.get(cut, 0) + lengths.get(not_cut, 0)
        if total:
            label = cut.split("_")[0]
            parts.append(f"{label} cut: {100 * lengths.get(cut, 0) / total:.0f}% of path")
    return ", ".join(parts)


def overlay(img, mask, title_lines):
    color = np.zeros_like(img)
    for i, name in enumerate(CLASSES):
        color[mask == i] = COLORS[name]
    out = img.copy()
    labelled = mask != BACKGROUND
    out[labelled] = cv2.addWeighted(img, 0.45, color, 0.55, 0)[labelled]
    lines = (mask >= CLASSES.index("top_cut")) & labelled
    out[lines] = color[lines]  # cut bands drawn solid so they stay visible
    s = 900 / max(out.shape[:2])
    out = cv2.resize(out, None, fx=s, fy=s)
    bar = np.full((14 + 30 * len(title_lines), out.shape[1], 3), 255, np.uint8)
    for k, text in enumerate(title_lines):
        cv2.putText(bar, text, (8, 32 + 30 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
    return np.vstack([bar, out])


def yolo_lines(mask):
    h, w = mask.shape
    lines = []
    for i in range(len(CLASSES)):
        contours, _ = cv2.findContours((mask == i).astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) < 50:
                continue
            c = cv2.approxPolyDP(c, 1.5, True).reshape(-1, 2)
            if len(c) < 3:
                continue
            coords = " ".join(f"{x / w:.5f} {y / h:.5f}" for x, y in c)
            lines.append(f"{i} {coords}")
    return lines


def legend(width):
    bar = np.full((50, width, 3), 255, np.uint8)
    x = 10
    for name in CLASSES:
        cv2.rectangle(bar, (x, 15), (x + 26, 37), COLORS[name], -1)
        cv2.putText(bar, name, (x + 32, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        x += 60 + 13 * len(name)
    return bar


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # paths below are relative to the project
    with open("examples/annotations.json") as f:
        frames = json.load(f)["frames"]
    out_dir = os.path.join("examples", "output")
    os.makedirs(out_dir, exist_ok=True)
    panels = []
    for ann in frames:
        img = read_frame(ann["video"], ann["time_s"])
        mask, lengths = build_mask(img, ann)
        base = os.path.join(out_dir, ann["name"])
        cv2.imwrite(base + "_image.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(base + "_mask.png", mask)
        cov = coverage_text(lengths)
        title = [ann["name"], f"{os.path.basename(ann['video'])[:24]} @{ann['time_s']}s", cov]
        panel = overlay(img, mask, title)
        cv2.imwrite(base + "_overlay.jpg", panel)
        with open(base + ".txt", "w") as f:
            f.write("\n".join(yolo_lines(mask)) + "\n")
        panels.append(panel)
        print(f"{ann['name']:22s} {cov}")

    height = 700
    row = [cv2.resize(p, (int(p.shape[1] * height / p.shape[0]), height)) for p in panels]
    gap = np.full((height, 10, 3), 255, np.uint8)
    sheet = row[0]
    for p in row[1:]:
        sheet = np.hstack([sheet, gap, p])
    cv2.imwrite(os.path.join(out_dir, "summary.jpg"), np.vstack([sheet, legend(sheet.shape[1])]))
    print(f"wrote {len(frames)} examples to {out_dir}/")


if __name__ == "__main__":
    main()
