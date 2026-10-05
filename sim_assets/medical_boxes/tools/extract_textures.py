#!/usr/bin/env python3
"""Regenerate the face textures of the five medical cartons (deterministic).

Run from anywhere:  python3 sim_assets/medical_boxes/tools/extract_textures.py
Writes textures/<box>/{front,back,top,bottom,left,right}.png.  Orientation and
image sizes follow boxes.yaml:
  front/back  W=height H=length      top/bottom  W=height H=thickness
  left/right  W=length H=thickness
Spine ("top") and front/back images are 4 px/mm; plain-colour faces are 1 px/mm.

SPINE CORNERS (hand-picked on the 1200x1600 phone photos, zoomed 5x crops, +-2 px).
Photo coordinates (x, y), in the order top-left, top-right, bottom-right,
bottom-left *of the photo*, i.e. the printed text runs bottom -> top in all three
photos (SPINE_ROTATION overrides this per box: 4.jpeg reads top -> bottom and is
rotated 90 deg CCW), so the photo is rotated 90 deg clockwise: photo-left edge becomes the
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
  ultimaster_nagomi  front/back: white label of photos/4.jpeg; edges/bottom: green top band
  conqueror_nc_pro   light-aqua top band of photos/5.jpeg
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
    # 4.jpeg: green end band top edge y~160-164, TERUMO band bottom edge y~1553-1556;
    # the spine leans slightly (left edge 579 -> 567, right edge 659 -> 654).
    "ultimaster_nagomi": ((579, 160), (659, 164), (654, 1553), (567, 1556)),
    # 5.jpeg: panel edges (excluding the dark grazing side face and the fingers
    # behind it): left 555 -> 497, right 629 -> 578; ends y 111 / 1518.
    "conqueror_nc_pro": ((555, 111), (629, 111), (578, 1518), (497, 1518)),
}
PHOTO = {"agent_monorail": "1.jpeg", "angiolite": "2.jpeg", "accuforce": "3.jpeg",
         "ultimaster_nagomi": "4.jpeg", "conqueror_nc_pro": "5.jpeg"}
# Photo rotation that makes the printed text read upright left -> right.
#   "cw"  text runs bottom -> top in the photo (default; boxes 1-3, 5)
#   "ccw" text runs top -> bottom in the photo (box 4)
SPINE_ROTATION = {"ultimaster_nagomi": "ccw"}

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
    "ultimaster_nagomi": {
        "face": ("photos/4.jpeg", 605, 815, 655, 838),      # white label area, clear of text
        "edge": ("photos/4.jpeg", 588, 230, 606, 330),      # green top band, left of the text
    },
    "conqueror_nc_pro": {
        "face": ("photos/5.jpeg", 565, 125, 615, 290),      # light-aqua top band
        "edge": ("photos/5.jpeg", 565, 125, 615, 290),
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
    if SPINE_ROTATION.get(box, "cw") == "cw":
        src = np.float32([bl, tl, tr, br])
    else:   # photo rotated 90 deg CCW: output TL <- photo TR, TR <- BR, BR <- BL, BL <- TL
        src = np.float32([tr, br, bl, tl])
    cw, ch = w + 2 * m, h + 2 * m
    dst = np.float32([[0, 0], [cw, 0], [cw, ch], [0, ch]])
    big = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (cw, ch),
                              flags=cv2.INTER_CUBIC)
    return cv2.resize(big[m:ch - m, m:cw - m], (w, h), interpolation=cv2.INTER_CUBIC)


# ---------------------------------------------------------------------------
# FACE PHOTOS (source/face_photos/<box>_{front,back}.jpeg): the big faces held in a hand,
# mild perspective.  Corners are the outer edge of the printed face (TL, TR, BR, BL of
# the *upright photo*), hand-picked on zoomed 5x crops with a 10 px grid, +-3 px.  Side
# walls (the thin lit rim on the tab edge) and loose tabs are excluded.  The Nagomi
# photos are cropped by the photo border on the left (and the back also on the right):
# those corners are EXTRAPOLATED outside the image (negative x / x > 1200) along the
# visible edge, so a wedge of the face is missing there.  The warp replicates the border
# pixels into that wedge and FACE_VALID_MASK tells match_net.py where data is real.
# In the photos the vertical edges are the box HEIGHT edges and the horizontal edges the
# LENGTH edges (measured aspects: Agent 0.89-0.91, Angiolite 0.94, Accuforce 0.80-0.86,
# Nagomi 0.89-0.90, Conqueror 0.93 vs height/length 0.92 / 0.96 / 0.88 / 0.85 / 0.92);
# only the nearly square Agent is also tried the other way round (ASPECT_CANDIDATES).
FACE_CORNERS = {
    "agent_monorail": {
        "front": ((10, 520), (795, 479), (826, 1198), (12, 1242)),
        "back":  ((16, 520), (837, 510), (884, 1254), (29, 1317)),
    },
    "angiolite": {
        "front": ((9, 330), (1168, 280), (1157, 1338), (94, 1374)),
        "back":  ((115, 474), (1081, 476), (1071, 1341), (140, 1384)),
    },
    "accuforce": {
        "front": ((26, 525), (1193, 500), (1185, 1426), (119, 1489)),
        "back":  ((40, 464), (1171, 401), (1170, 1298), (152, 1393)),
    },
    "ultimaster_nagomi": {
        "front": ((-48, 321), (1196, 288), (1148, 1367), (55, 1344)),
        "back":  ((-58, 201), (1245, 179), (1181, 1265), (55, 1294)),
    },
    "conqueror_nc_pro": {
        "front": ((19, 371), (1165, 332), (1160, 1383), (76, 1376)),
        "back":  ((38, 373), (1153, 371), (1160, 1411), (38, 1408)),
    },
}
ASPECT_CANDIDATES = {"agent_monorail": ("L", "H")}   # default ("L",): photo horizontal = length


def photo_face(box, side, aspect="L", dims=None):
    """Rectify one face photo.  Returns (bgr, valid) in the PHOTO frame (upright photo
    orientation): aspect "L" -> W = length, H = height; "H" -> W = height, H = length;
    4 px/mm, OUTPUT_INSET_PX trimmed.  valid is a bool mask (False where the quad lies
    outside the photo; those pixels hold replicated border colour)."""
    img = load("face_photos/%s_%s.jpeg" % (box, side))
    if dims is None:
        with open(os.path.join(ROOT, "boxes.yaml")) as f:
            dims = yaml.safe_load(f)["boxes"][box]["dims_mm"]
    a, b = ((dims["length"], dims["height"]) if aspect == "L"
            else (dims["height"], dims["length"]))
    w, h = int(a * PX_MM), int(b * PX_MM)
    m = OUTPUT_INSET_PX
    cw, ch = w + 2 * m, h + 2 * m
    src = np.float32(FACE_CORNERS[box][side])
    dst = np.float32([[0, 0], [cw, 0], [cw, ch], [0, ch]])
    M = cv2.getPerspectiveTransform(src, dst)
    big = cv2.warpPerspective(img, M, (cw, ch), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    ones = np.full(img.shape[:2], 255, np.uint8)
    vm = cv2.warpPerspective(ones, M, (cw, ch), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    out = cv2.resize(big[m:ch - m, m:cw - m], (w, h), interpolation=cv2.INTER_CUBIC)
    vm = cv2.resize(vm[m:ch - m, m:cw - m], (w, h), interpolation=cv2.INTER_LINEAR)
    return out, vm > 250


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


# Old front/back providers (internet photos / plain colour).  UNUSED now that the face
# photos below are used; kept so they can be switched back by adding them to these dicts.
FRONT = {}
BACK = {}

# Orientation of the face photos, from tools/match_net.py (spine-fold artwork continuity
# + tuck-tab constraint).  (aspect, k): photo rectified with aspect "L" (photo horizontal =
# length) then rotated k quarter turns CCW (np.rot90).  k=1: the photo's RIGHT edge becomes
# the top (spine) edge; k=3: the photo's LEFT edge does.  Confidence per box:
#   agent_monorail     clear      (dE 8.9 vs next 12.9; green MR corner wrap; tab opposite)
#   accuforce          weak       (spine as-is beats rot180 by 2 dE, corr 0.76 vs 0.56)
#   angiolite          undecided  (as-is vs rot180 differ by 0.35 dE; faces are firm: 20 vs 25+)
#   ultimaster_nagomi  undecided  (front/back k=3,3 chosen by tab edges; photos are cut by the
#                                  photo border so the fold strips are only ~60 % valid)
#   conqueror_nc_pro   undecided  (as-is vs rot180 differ by 0.6 dE; faces firm: 14.8 vs 32)
FACE_ORIENT = {
    "agent_monorail":    {"front": ("L", 1), "back": ("L", 3)},
    "angiolite":         {"front": ("L", 1), "back": ("L", 3)},
    "accuforce":         {"front": ("L", 1), "back": ("L", 3)},
    "ultimaster_nagomi": {"front": ("L", 3), "back": ("L", 3)},
    "conqueror_nc_pro":  {"front": ("L", 1), "back": ("L", 3)},
}
# True -> top.png is written rotated 180 deg (text upside down, ends swapped).  Only set
# where match_net.py gives a non-"undecided" verdict for it; none of the five qualifies:
# agent (clear) and accuforce (weak) both prefer the spine exactly as photographed.
SPINE_FLIP = {name: False for name in FACE_ORIENT}


def face_texture(box, side):
    asp, k = FACE_ORIENT[box][side]
    img, _ = photo_face(box, side, asp)
    return np.ascontiguousarray(np.rot90(img, k))


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
        spine = rectify_spine(name, d)
        if SPINE_FLIP[name]:
            spine = np.ascontiguousarray(np.rot90(spine, 2))
        save(os.path.join(o, "top.png"), spine)
        save(os.path.join(o, "bottom.png"), flat(edge, hgt, th))
        save(os.path.join(o, "left.png"), flat(edge, ln, th))
        save(os.path.join(o, "right.png"), flat(edge, ln, th))
        save(os.path.join(o, "front.png"), face_texture(name, "front"))
        save(os.path.join(o, "back.png"), face_texture(name, "back"))


if __name__ == "__main__":
    main()
