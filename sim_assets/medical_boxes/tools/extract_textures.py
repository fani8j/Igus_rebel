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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import texture_clean as tc  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "source")
OUT = os.environ.get("TEXTURE_OUT_DIR") or os.path.join(ROOT, "textures")   # override for dry runs

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
    """Atomic write: encode, write a temp file next to the target, fsync, os.replace.
    Skipped when the encoded bytes equal the existing file (mtime stays)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ok, buf = cv2.imencode(".png", img_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    if not ok:
        sys.exit("cannot encode " + path)
    data = buf.tobytes()
    if os.path.isfile(path):
        with open(path, "rb") as f:
            if f.read() == data:
                return
    tmp = "%s.tmp-%d" % (path, os.getpid())
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


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
    """Raw rectified face (rotated to the texture orientation) and its validity mask
    (False where the warp had to extrapolate outside the photo)."""
    asp, k = FACE_ORIENT[box][side]
    img, valid = photo_face(box, side, asp)
    return (np.ascontiguousarray(np.rot90(img, k)),
            np.ascontiguousarray(np.rot90(valid, k)))


# ---------------------------------------------------------------------------
# TEXTURE CLEAN-UP.  Everything below only changes pixel VALUES of the textures built
# above (same size, orientation and artwork placement).  All coordinates are in
# OUTPUT-texture pixels (x right, y down) of the named face image.
#
# Pipeline per face:  1 border slivers -> 2 occlusion / glare / damage fills ->
#                     3 illumination flattening -> 4 per-face colour matching.
# ---------------------------------------------------------------------------

# 1. Border slivers (top, bottom, left, right) px: warp edge pixels that show the
# background, a darker box-edge falloff or a skin/shadow fringe.  Replaced by the
# mirrored interior (the artwork there is flat or a smooth gradient).
BANDS = {
    ("agent_monorail", "front"): (0, 0, 0, 10),       # right: box-edge shading falloff 77 -> 136
    ("agent_monorail", "back"): (12, 0, 0, 2),         # top: dark background sliver
    ("angiolite", "front"): (6, 0, 0, 0),             # top: darker warp-edge rows
    ("accuforce", "front"): (12, 4, 13, 13),          # edge falloff on three sides + dark bottom row
    ("accuforce", "back"): (7, 8, 0, 5),              # top/bottom/right: white background sliver
    ("conqueror_nc_pro", "back"): (0, 0, 0, 5),       # right: bright warp-edge columns
}

# 2. Fills.  (what it covers, method, polygon(s), params)
#   inpaint    : cv2.inpaint (Navier-Stokes) on a 1/4-scale copy, upsampled, feathered
#   smear_down : rows below the polygon's top edge become the (blurred) rows just above
#                it (box edge bleeds off a solid print / the silver band continues)
#   stock      : flat white label stock, dark neutral ink kept; params = (x0, y0, x1, y1)
#                clean stock patch the colour is taken from
#   rows       : per-row median colour of the clean columns (xa, xb); for vertical
#                gradients that are constant across the width
def ellipse(cx, cy, rx, ry, n=48):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return [(int(round(cx + rx * np.cos(a))), int(round(cy + ry * np.sin(a)))) for a in t]


FILLS = {
    ("agent_monorail", "front"): [
        # operator's hand under the box bottom edge (the face ends at the silver band;
        # the band and the blue tab bleed off the edge, so they are continued downwards)
        ("hand below the bottom edge, left of the silver band's shadow (sharp columns)",
         "smear_down", [(0, 976), (300, 973), (400, 969), (548, 966), (548, 1000), (0, 1000)],
         (6, 2.0)),
        ("hand below the bottom edge, right part (columns heavily blurred in x)",
         "smear_down", [(548, 966), (680, 960), (760, 955), (920, 955), (920, 1000), (548, 1000)],
         (8, 25.0, 26)),
    ],
    ("agent_monorail", "back"): [
        # torn, delaminated label stock (white fibres over the blue box + printed label);
        # physical damage, not artwork: its barcode / STERILE / symbol boxes are shredded,
        # so everything is flat white stock (colour interpolated between the clean label
        # strip above and the big label below) except the still intact CE mark
        ("torn label fibres", "stock",
         [(568, 0), (770, 0), (770, 366), (568, 366)],
         dict(refs=[(30, (480, 8, 562, 55)), (380, (630, 372, 760, 392))],
              keep=[(718, 78, 768, 116)])),
    ],
    ("angiolite", "back"): [
        ("fingertip over the green header strip", "inpaint",
         [(584, 0), (664, 0), (664, 15), (584, 15)], None),
        # foil-like glare wash over the pink/white gradient, right and lower right (no
        # text there): per-row print gradient from the clean left margin (x 30-130), scaled
        # to the clean right margin above the glare (x 830-950, y 300-440)
        ("specular glare, right", "rows",
         [(808, 455), (960, 455), (960, 800), (808, 802)],
         dict(xa=30, xb=130, smooth=4.0, feather=14.0, match=(830, 300, 950, 440),
              taper_y=840)),
        ("specular glare, lower right (stops short of the iVascular logo, x < 640)", "rows",
         [(809, 790), (960, 790), (960, 986), (670, 986), (670, 872), (640, 850),
          (625, 826), (700, 812), (760, 806)],
         dict(xa=30, xb=130, smooth=4.0, feather=10.0)),
    ],
    ("ultimaster_nagomi", "front"): [
        ("grey photo background in the top-left corner", "inpaint",
         [(0, 0), (14, 0), (14, 40), (0, 40)], None),
    ],
    ("conqueror_nc_pro", "front"): [
        ("grey shadow of the box side wall along the left edge, replaced by the mirrored "
         "interior columns", "mirror_left", [(0, 38), (700, 30), (740, 22), (999, 22)], None),
        ("green dirt specks on the aqua", "inpaint",
         [(835, 72), (897, 72), (897, 118), (835, 118)], None),
        ("specular hot spot on the aqua, top", "inpaint", ellipse(460, 28, 85, 55), None),
        ("white glare fleck on the blue wave, bottom (cloned from 30 px to the left)",
         "clone", [(485, 951), (505, 951), (505, 971), (485, 971)], (-30, 0)),
        ("white glare flecks on the blue wave, bottom (cloned from 50 px to the right, "
         "after the fleck above is gone)",
         "clone", [(404, 938), (448, 938), (448, 980), (404, 980)], (50, 0)),
        ("two remaining glare dots", "inpaint",
         [[(395, 945), (405, 945), (405, 956), (395, 956)],
          [(439, 943), (449, 943), (449, 954), (439, 954)]],
         dict(scale=1, radius=3, method=cv2.INPAINT_TELEA, grow=1)),
    ],
}

# 3. Illumination flattening: rects (x0, y0, x1, y1) of large uniform print (label
# stock / solid brand colour), polynomial degree, clamp of the gain.
FLATTEN = {
    ("agent_monorail", "front"): dict(rects=[(150, 110, 420, 230), (660, 500, 900, 780),
                                              (10, 640, 400, 800)]),
    ("agent_monorail", "back"): dict(rects=[(200, 110, 500, 330), (780, 330, 900, 700)]),
    ("angiolite", "front"): dict(rects=[(10, 20, 140, 200), (830, 20, 950, 200)], deg="x"),
    ("angiolite", "back"): dict(rects=[(10, 200, 130, 420), (830, 200, 950, 420)], deg="x"),
    ("accuforce", "back"): dict(rects=[(50, 700, 350, 900), (500, 700, 850, 950),
                                       (40, 520, 170, 900)]),
    ("conqueror_nc_pro", "front"): dict(rects=[(300, 720, 560, 790), (60, 340, 400, 440)]),
    ("conqueror_nc_pro", "back"): dict(rects=[(20, 100, 150, 480), (780, 100, 900, 500)]),
}

# 4. Colour references: name -> (rect, mode).  'white' = the white label stock
# (brighter half of the patch, so ink does not matter); the other names are brand
# colours that occur on several faces of the same box.  Spine = face name "top".
COLOR_REFS = {
    ("agent_monorail", "front"): {"white": ((320, 30, 470, 90), "white"),
                                  "blue": ((150, 110, 420, 230), "median")},
    ("agent_monorail", "back"): {"white": ((200, 6, 430, 55), "white"),
                                 "blue": ((230, 110, 500, 330), "median")},
    ("agent_monorail", "top"): {"white": ((160, 4, 420, 30), "white"),
                                "blue": ((120, 4, 142, 56), "median")},
    ("angiolite", "front"): {"white": ((730, 20, 805, 170), "white"),
                             "magenta": ((10, 60, 130, 180), "median")},
    ("angiolite", "back"): {"white": ((170, 100, 800, 330), "white"),
                            "magenta": ((10, 200, 130, 420), "median")},
    ("angiolite", "top"): {"magenta": ((112, 8, 138, 56), "median")},
    # accuforce front: no colour reference. Its only white is a small glossy import sticker
    # that is not neutral itself; anchoring on it lifted the whole navy face ~10 % and tinted
    # it lilac, and the swirl artwork has no flat area to anchor on instead.
    ("accuforce", "back"): {"white": ((190, 300, 670, 560), "white"),
                            "navy": ((50, 700, 350, 900), "median")},
    ("accuforce", "top"): {"white": ((190, 4, 430, 60), "white"),
                           "navy": ((3, 3, 175, 10), "median")},
    ("ultimaster_nagomi", "front"): {"white": ((545, 880, 960, 1090), "white")},
    ("ultimaster_nagomi", "back"): {"white": ((240, 60, 420, 170), "white")},
    ("ultimaster_nagomi", "top"): {"white": ((230, 4, 450, 60), "white")},
    ("conqueror_nc_pro", "front"): {"white": ((300, 720, 560, 790), "white")},
    ("conqueror_nc_pro", "back"): {"white": ((20, 100, 150, 480), "white")},
    ("conqueror_nc_pro", "top"): {"white": ((440, 4, 640, 44), "white")},
}
WHITE_TARGET = 240        # sRGB of the white label stock after correction (neutral)

# Plain edge colour (bottom/left/right) re-sampled from the corrected textures:
# (face, rect, mode).
EDGE_SRC = {
    "agent_monorail": ("top", (120, 4, 142, 56), "median"),      # Boston Sci. blue band
    "angiolite": ("top", (112, 8, 138, 56), "median"),           # magenta end band
    "accuforce": ("back", (50, 700, 350, 900), "median"),        # navy field
    "ultimaster_nagomi": ("top", (4, 4, 200, 18), "median"),     # green band
    "conqueror_nc_pro": ("top", (5, 4, 150, 16), "median"),      # aqua band
}


def clean_face(box, side, img, valid, report):
    key = (box, side)
    h, w = img.shape[:2]
    if (~valid).any():                         # extrapolated wedge of the Nagomi photos
        inv = np.ascontiguousarray((~valid).astype(np.uint8) * 255)
        inv = cv2.dilate(inv, np.ones((9, 9), np.uint8))
        img = tc.inpaint(img, inv, scale=4, radius=5, grow=0)
    if key in BANDS:
        img = tc.mirror_bands(img, BANDS[key])
    excl = np.zeros((h, w), np.uint8)
    for what, method, polys, par in FILLS.get(key, []):
        if method == "mirror_left":
            img = tc.mirror_left(img, polys)
            continue
        if isinstance(polys[0][0], (int, float)):
            polys = [polys]
        m = tc.poly_mask(img.shape, polys)
        excl = cv2.bitwise_or(excl, m)
        if method == "inpaint":
            img = tc.inpaint(img, m, **(par or {}))
        elif method == "stock":
            img = tc.stock_fill(img, m, **par)
        elif method == "rows":
            img = tc.row_fill(img, m, **par)
        elif method == "clone":
            img = tc.clone(img, m, *par)
        elif method == "smear_down":
            img = tc.smear_down(img, polys[0], *par)
        else:
            raise ValueError(method)
    if key in FLATTEN:
        fl = FLATTEN[key]
        img, info = tc.flatten(img, fl["rects"], fl.get("deg", 1),
                               fl.get("lo", 0.85), fl.get("hi", 1.2), exclude=excl)
        report.append("  flatten %s/%s: ref L %.0f, gain %.2f..%.2f, fit resid %.1f"
                      % (box, side, info["ref"], info["gmin"], info["gmax"], info["resid"]))
    return img


def to_rgb(bgr):
    return tuple(int(round(v)) for v in np.asarray(bgr)[::-1])


def main():
    with open(os.path.join(ROOT, "boxes.yaml")) as f:
        boxes = yaml.safe_load(f)["boxes"]
    for name, spec in boxes.items():
        d = spec["dims_mm"]
        hgt, ln, th = d["height"], d["length"], d["thickness"]
        sm = SAMPLES[name]
        print("%-15s photo face sRGB %s  edge/bottom sRGB %s"
              % (name, median_rgb(*sm["face"]), median_rgb(*sm["edge"])))
        report = []
        faces = {}
        spine = rectify_spine(name, d)
        if SPINE_FLIP[name]:
            spine = np.ascontiguousarray(np.rot90(spine, 2))
        faces["top"] = clean_face(name, "top", spine, np.ones(spine.shape[:2], bool), report)
        for side in ("front", "back"):
            raw, valid = face_texture(name, side)
            faces[side] = clean_face(name, side, raw, valid, report)

        # colour matching: common target per reference colour
        refs = {f: COLOR_REFS.get((name, f), {}) for f in faces}
        meas = {f: {n: tc.sample(faces[f], r, mode) for n, (r, mode) in refs[f].items()}
                for f in faces}
        target = {"white": np.full(3, float(WHITE_TARGET))}
        for n in {n for f in meas for n in meas[f] if n != "white"}:
            target[n] = np.mean([meas[f][n] for f in meas if n in meas[f]], axis=0)
        for f in faces:
            if not meas[f]:
                continue
            tgt = {n: target[n] for n in meas[f]}
            M = tc.solve_matrix(meas[f], tgt)
            faces[f] = tc.apply_matrix(faces[f], M)
            after = {n: tc.sample(faces[f], r, mode) for n, (r, mode) in refs[f].items()}
            for n in meas[f]:
                report.append("  %-6s %-8s sRGB %-15s -> %-15s (target %s)"
                              % (f, n, to_rgb(meas[f][n]), to_rgb(after[n]), to_rgb(target[n])))
            report.append("  %-6s RGB matrix %s" % (f, np.round(M[::-1, ::-1], 2).tolist()))
        edge = to_rgb(tc.sample(faces[EDGE_SRC[name][0]], EDGE_SRC[name][1], EDGE_SRC[name][2])) \
            if name in EDGE_SRC else median_rgb(*sm["edge"])
        report.append("  edge/bottom sRGB %s" % (edge,))
        print("\n".join(report))

        o = os.path.join(OUT, name)
        save(os.path.join(o, "top.png"), faces["top"])
        save(os.path.join(o, "bottom.png"), flat(edge, hgt, th))
        save(os.path.join(o, "left.png"), flat(edge, ln, th))
        save(os.path.join(o, "right.png"), flat(edge, ln, th))
        save(os.path.join(o, "front.png"), faces["front"])
        save(os.path.join(o, "back.png"), faces["back"])


if __name__ == "__main__":
    main()
