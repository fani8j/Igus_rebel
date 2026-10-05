#!/usr/bin/env python3
"""Regenerate the face textures of the three medical cartons (deterministic).

Run from anywhere:  python3 sim_assets/medical_boxes/tools/extract_textures.py
Writes textures/<box>/{front,back,top,bottom,left,right}.png.  Orientation and
image sizes follow boxes.yaml:
  front/back  W=height H=length      top/bottom  W=height H=thickness
  left/right  W=length H=thickness
Spine ("top") and front/back images are 4 px/mm; plain-colour faces are 1 px/mm.

SPINE CORNERS (hand-picked on the 1200x1600 phone photos, zoomed 5x crops, +-2 px).
Photo coordinates (x, y), in the order top-left, top-right, bottom-right,
bottom-left *of the photo*, i.e. the printed text runs bottom -> top in all three
photos, so the photo is rotated 90 deg clockwise: photo-left edge becomes the
output top, photo-bottom becomes the output left.  Corners are the outermost
extents of the printed spine panel (end band to end band), excluding the
background, the hand and the adjacent faces; OUTPUT_INSET_PX trims a further sliver.

PLAIN COLOURS (sRGB, median over a clean patch of the real photos; re-sampled
and printed on each run, the SAMPLE_* boxes below are (x0, y0, x1, y1)):
  agent_monorail  front/back: 'AGENT' box blue in photos/2.jpeg (front blue strip, x 200-380 y 296-316)
                  edges/bottom: the blue band of the spine in photos/1.jpeg
  angiolite       magenta band of the spine in photos/2.jpeg (front face in
                  photos/1.jpeg, bottom right, is the same pink)
  accuforce       navy from the background of internet/accuforce_back_face.jpg
"""
import os
import sys

import cv2
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "source")
OUT = os.path.join(ROOT, "textures")

PX_MM = 4                 # spine / front / back resolution
FLAT_PX_MM = 1            # plain-colour faces
OUTPUT_INSET_PX = 2       # trim at the warped panel edge (output px)

# (TL, TR, BR, BL) of the spine quad in the photo, photo orientation (text runs upward).
SPINE_CORNERS = {
    "agent_monorail": ((477, 331), (564, 331), (567, 1541), (482, 1541)),
    "angiolite":      ((420, 159), (524, 159), (547, 1438), (462, 1438)),
    "accuforce":      ((593, 284), (669, 284), (645, 1522), (557, 1522)),
}
PHOTO = {"agent_monorail": "1.jpeg", "angiolite": "2.jpeg", "accuforce": "3.jpeg"}

# Colour sample patches (photo file, x0, y0, x1, y1)
SAMPLES = {
    "agent_monorail": {
        "face": ("photos/2.jpeg", 200, 296, 380, 316),      # AGENT box blue (background)
        "edge": ("photos/1.jpeg", 500, 1385, 560, 1440),    # spine blue end band
    },
    "angiolite": {
        "face": ("photos/2.jpeg", 440, 1290, 520, 1380),    # spine magenta end band
        "edge": ("photos/2.jpeg", 440, 1290, 520, 1380),
    },
    "accuforce": {
        "face": ("internet/accuforce_back_face.jpg", 20, 120, 840, 300),
        "edge": ("internet/accuforce_back_face.jpg", 20, 120, 840, 300),
    },
}
# Accuforce front/back crop (non-box border), (top, bottom, left, right) px
CROP_BACK = (6, 8, 4, 4)
CROP_FRONT = (3, 5, 0, 0)
FRONT_X_STRETCH = 1.03    # max horizontal stretch of the front photo (3 %)
# Agent front: reseller photo (medicalmaterials, faint "MMI" watermarks) of the
# US carton (catalog 39608-3027) on a flat blue backdrop.  Box face found at
# rows 28-931, cols 222-1057 (836x904 px, aspect 0.925 vs measured 230/250 =
# 0.920); crop (y0, y1, x0, x1) with a 2 px inset.
AGENT_FRONT_CROP = (30, 930, 224, 1056)


def load(rel):
    im = cv2.imread(os.path.join(SRC, rel), cv2.IMREAD_COLOR)
    if im is None:
        sys.exit("cannot read " + rel)
    return im


def median_rgb(rel, x0, y0, x1, y1):
    patch = load(rel)[y0:y1, x0:x1].reshape(-1, 3)
    bgr = np.median(patch, axis=0)
    return tuple(int(round(v)) for v in bgr[::-1])


def save(path, img_bgr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 9])


def rectify_spine(box, dims):
    img = load("photos/" + PHOTO[box])
    w, h = int(dims["height"] * PX_MM), int(dims["thickness"] * PX_MM)
    m = OUTPUT_INSET_PX
    tl, tr, br, bl = SPINE_CORNERS[box]
    # photo rotated 90 deg CW: output TL <- photo BL, TR <- photo TL,
    # BR <- photo TR, BL <- photo BR.  The quad is mapped onto a canvas of
    # (w+2m, h+2m) whose outer ring of m px is then cut off (the inset) and the
    # rest resized to exactly w x h (forces the measured mm aspect).
    src = np.float32([bl, tl, tr, br])
    cw, ch = w + 2 * m, h + 2 * m
    dst = np.float32([[0, 0], [cw, 0], [cw, ch], [0, ch]])
    big = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (cw, ch),
                              flags=cv2.INTER_CUBIC)
    return cv2.resize(big[m:ch - m, m:cw - m], (w, h), interpolation=cv2.INTER_CUBIC)


def flat(rgb, w_mm, h_mm):
    img = np.empty((int(h_mm * FLAT_PX_MM), int(w_mm * FLAT_PX_MM), 3), np.uint8)
    img[:] = rgb[::-1]
    return img


def accuforce_back(dims):
    t, b, l, r = CROP_BACK
    im = load("internet/accuforce_back_face.jpg")
    im = im[t:im.shape[0] - b, l:im.shape[1] - r]
    return cv2.resize(im, (int(dims["height"] * PX_MM), int(dims["length"] * PX_MM)),
                      interpolation=cv2.INTER_CUBIC)


def accuforce_front(dims):
    """Source is cropped on the right (the word 'Accuforce' is cut off), so it is
    too narrow.  Stretch x by <= 3 %, then extend the right edge with a mirrored,
    strongly blurred copy of the last columns (dark-blue swirl background),
    feathered into the photo.  The mirrored strip is blurred so no mirrored
    letters are visible."""
    W, H = int(dims["height"] * PX_MM), int(dims["length"] * PX_MM)
    t, b, l, r = CROP_FRONT
    im = load("internet/accuforce_front_face.jpg")
    im = im[t:im.shape[0] - b, l:im.shape[1] - r]
    sw = int(round(im.shape[1] * (H / im.shape[0]) * FRONT_X_STRETCH))
    im = cv2.resize(im, (sw, H), interpolation=cv2.INTER_CUBIC)
    gap = W - sw
    assert 0 < gap < sw // 4, gap
    out = np.zeros((H, W, 3), np.float32)
    out[:, :sw] = im
    # mirrored copy of the last `gap` columns, then a heavy Gaussian blur
    out[:, sw:] = im[:, sw - gap:sw][:, ::-1]
    blurred = cv2.GaussianBlur(out, (0, 0), sigmaX=9, sigmaY=9)
    # use the blurred version for the strip and feather it over the last 24 photo px
    f = 24
    wgt = np.zeros(W, np.float32)
    wgt[sw - f:sw] = np.linspace(0, 1, f)
    wgt[sw:] = 1.0
    out = out * (1 - wgt[None, :, None]) + blurred * wgt[None, :, None]
    return np.clip(out, 0, 255).astype(np.uint8)


def agent_front(dims):
    y0, y1, x0, x1 = AGENT_FRONT_CROP
    im = load("internet/agent_front_face.png")[y0:y1, x0:x1]
    return cv2.resize(im, (int(dims["height"] * PX_MM), int(dims["length"] * PX_MM)),
                      interpolation=cv2.INTER_AREA)


def main():
    with open(os.path.join(ROOT, "boxes.yaml")) as f:
        boxes = yaml.safe_load(f)["boxes"]
    for name, spec in boxes.items():
        d = spec["dims_mm"]
        hgt, ln, th = d["height"], d["length"], d["thickness"]
        sm = SAMPLES[name]
        face = median_rgb(*sm["face"])
        edge = median_rgb(*sm["edge"])
        print("%-15s face sRGB %s  edge/bottom sRGB %s" % (name, face, edge))
        o = os.path.join(OUT, name)
        save(os.path.join(o, "top.png"), rectify_spine(name, d))
        save(os.path.join(o, "bottom.png"), flat(edge, hgt, th))
        save(os.path.join(o, "left.png"), flat(edge, ln, th))
        save(os.path.join(o, "right.png"), flat(edge, ln, th))
        if name == "accuforce":
            save(os.path.join(o, "front.png"), accuforce_front(d))
            save(os.path.join(o, "back.png"), accuforce_back(d))
        elif name == "agent_monorail":
            save(os.path.join(o, "front.png"), agent_front(d))
            save(os.path.join(o, "back.png"), flat(face, hgt, ln))
        else:
            save(os.path.join(o, "front.png"), flat(face, hgt, ln))
            save(os.path.join(o, "back.png"), flat(face, hgt, ln))


if __name__ == "__main__":
    main()
