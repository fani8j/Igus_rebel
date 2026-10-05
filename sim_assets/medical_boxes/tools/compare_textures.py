#!/usr/bin/env python3
"""Compare two texture sets (e.g. before/after cleanup): a side-by-side sheet per carton
plus colour metrics taken on the reference patches in extract_textures.COLOR_REFS.

Metrics, per carton (CIELAB, D65):
  white cast   chroma C* of the white label stock on each face (0 = neutral grey/white)
  white L*     lightness of that stock (how evenly exposed the faces are)
  brand dE     largest CIEDE76 difference of the brand colour between the faces of one carton
               (how consistently front, back and spine render the same print)

    python3 tools/compare_textures.py --before <dir>/textures --after textures \
        --out usd/previews/texture_before_after.jpg
"""
import argparse
import itertools
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from extract_textures import COLOR_REFS  # noqa: E402  (reference patches, output-pixel rects)

ROOT = os.path.dirname(HERE)
BOXES = ["agent_monorail", "angiolite", "accuforce", "ultimaster_nagomi", "conqueror_nc_pro"]
FACES = ["front", "back", "top"]


def lab_of(img_bgr, rect, mode):
    x0, y0, x1, y1 = rect
    patch = img_bgr[y0:y1, x0:x1].reshape(-1, 3).astype(np.float32)
    if mode == "white":  # label stock: the brighter half, so print on it does not count
        lum = patch.sum(1)
        patch = patch[lum >= np.median(lum)]
    bgr = np.median(patch, axis=0).reshape(1, 1, 3) / 255.0
    return cv2.cvtColor(bgr.astype(np.float32), cv2.COLOR_BGR2Lab).reshape(3)


def metrics(tex_dir, box):
    whites, brands = {}, {}
    for face in FACES:
        refs = COLOR_REFS.get((box, face), {})
        img = cv2.imread(os.path.join(tex_dir, box, f"{face}.png"), cv2.IMREAD_COLOR)
        for name, (rect, mode) in refs.items():
            lab = lab_of(img, rect, mode)
            (whites if name == "white" else brands.setdefault(name, {}))[face] = lab
    cast = {f: float(np.hypot(v[1], v[2])) for f, v in whites.items()}
    light = {f: float(v[0]) for f, v in whites.items()}
    de = None
    for faces in brands.values():
        for a, b in itertools.combinations(faces.values(), 2):
            d = float(np.linalg.norm(a - b))
            de = d if de is None else max(de, d)
    return cast, light, de


def label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (len(text) * 11 + 12, 26), (255, 255, 255), -1)
    cv2.putText(out, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)
    return out


def sheet(before, after, box, height=420):
    tiles = []
    for face in ("front", "back"):
        for tag, d in (("before", before), ("after", after)):
            im = cv2.imread(os.path.join(d, box, f"{face}.png"), cv2.IMREAD_COLOR)
            s = height / im.shape[0]
            tiles.append(label(cv2.resize(im, (round(im.shape[1] * s), height), interpolation=cv2.INTER_AREA),
                               f"{face} {tag}"))
    row = np.hstack([np.pad(t, ((0, 0), (0, 8), (0, 0)), constant_values=255) for t in tiles])
    spines = []
    for tag, d in (("spine before", before), ("spine after", after)):
        im = cv2.imread(os.path.join(d, box, "top.png"), cv2.IMREAD_COLOR)
        s = (row.shape[1] / 2 - 8) / im.shape[1]
        spines.append(label(cv2.resize(im, (round(im.shape[1] * s), round(im.shape[0] * s)),
                                       interpolation=cv2.INTER_AREA), tag))
    h = max(sp.shape[0] for sp in spines)
    spines = [np.pad(sp, ((0, h - sp.shape[0]), (0, 8), (0, 0)), constant_values=255) for sp in spines]
    srow = np.hstack(spines)
    srow = np.pad(srow, ((0, 0), (0, max(0, row.shape[1] - srow.shape[1])), (0, 0)), constant_values=255)
    title = np.full((34, row.shape[1], 3), 255, np.uint8)
    cv2.putText(title, box, (6, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2, cv2.LINE_AA)
    return np.vstack([title, row, np.full((8, row.shape[1], 3), 255, np.uint8), srow[:, :row.shape[1]],
                      np.full((18, row.shape[1], 3), 255, np.uint8)])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", default=os.path.join(ROOT, "textures"))
    ap.add_argument("--out", default=os.path.join(ROOT, "usd", "previews", "texture_before_after.jpg"))
    a = ap.parse_args()

    print(f"{'carton':18s} {'white cast C* (front/back/spine)':>36s}   {'white L* spread':>16s}   "
          f"{'brand dE max':>13s}")
    for box in BOXES:
        row = []
        for d in (a.before, a.after):
            cast, light, de = metrics(d, box)
            row.append((cast, light, de))
        (cb, lb, db), (ca, la, da) = row

        def fmt(c):
            return "/".join(f"{c[f]:.1f}" if f in c else "-" for f in FACES)

        spread = lambda l: f"{max(l.values()) - min(l.values()):.1f}" if len(l) > 1 else "-"  # noqa: E731
        print(f"{box:18s} {fmt(cb):>16s} -> {fmt(ca):<16s}   {spread(lb):>5s} -> {spread(la):<6s}   "
              f"{'-' if db is None else f'{db:.1f}':>5s} -> {'-' if da is None else f'{da:.1f}'}")

    rows = [sheet(a.before, a.after, b) for b in BOXES]
    w = max(r.shape[1] for r in rows)
    out = np.vstack([np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0)), constant_values=255) for r in rows])
    s = min(1.0, 1800 / out.shape[1])
    out = cv2.resize(out, (round(out.shape[1] * s), round(out.shape[0] * s)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(a.out, out, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"wrote {a.out} ({out.shape[1]}x{out.shape[0]}, {os.path.getsize(a.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
