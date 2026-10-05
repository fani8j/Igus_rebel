#!/usr/bin/env python3
"""Front / back / bottom / left / right face textures for every catalog family (deterministic).

Run:  nice -n 19 python3 sim_assets/medical_boxes/tools/catalog_faces.py [--only fam,fam] [--sheets DIR] [--no-write]

Needs tools/catalog_spines.py to have run first (catalog/spines.yaml, textures/<fam>/top.png and, for
the large-face families, textures/<fam>/front_photo_crop.png).

Inputs : catalog/dims.yaml (FINAL dimensions), catalog/spines.yaml, catalog/research/*.json (image urls),
         research images on the external drive (RESEARCH_DIR), the table FRONTS / BASE below.
Output : catalog/textures/<fam>/{front,back,bottom,left,right}.png   (top.png belongs to catalog_spines.py)
         catalog/faces.yaml  (per family and face: source = photo_spine | redrawn | photo_face |
                              research_image(url) | plain(rgb), plus notes)

Conventions (box axes X = height, Y = length, Z = thickness):
  front (+Z) / back (-Z): W x H = height x length; front right = +X, back right = -X, up = +Y
  top (+Y, the labelled spine): height x thickness;  bottom (-Y): height x thickness
  left / right (-X / +X): length x thickness
Photo-derived images: 4 px/mm (capped so that no side exceeds 2048 px); plain colours: 1 px/mm.
Orientation rule for cartons: the spine edge (+Y, image top) is the carton edge opposite the "Open here"
tuck tab when it is visible ("tab_opposite": the image is rotated 180 deg if the tab is at its top),
otherwise the artwork is kept upright with the spine at the image top ("upright").
Fit rule: the rectified carton is stretched by at most +-8 % in aspect; the rest is padded at the bottom
(or the sides) with a smoothed copy of the carton's edge colour.
"""
import argparse
import json
import math
import os
import sys

import cv2
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import catalog_spines as cs                                  # noqa: E402

ROOT = cs.ROOT
CAT = cs.CAT
RESEARCH_DIR = "/media/farhan/My Passport/igus_nvidia/medical_boxes_research"
PX_MM = 4
MAX_PX = 2048
MAX_STRETCH = 0.08

# ----------------------------------------------------------------------------------------
# dominant base colour of each carton (bottom / left / right, and front / back where no photo exists).
# End-band colour for coloured cartons, white / grey stock for white ones; picked by eye from
# spines.yaml band_colors (and the front image where one exists).  value: (hex, reason)
# ----------------------------------------------------------------------------------------
BASE = {
    "abbott_armada_35": ("auto", "light blue carton (front photo border), blue spine end bands"),
    "abbott_xience_skypoint": ("auto", "light/mid blue carton (front photo), blue spine end bands"),
    "apt_medical_tresfort_35": ("#E6E6E2", "white carton, orange label only"),
    "avantigard_d_logo": ("#1E2326", "near-black spine end bands / carton"),
    "bard_conquest_pta": ("#1F63A8", "blue large face (photo)"),
    "biosensors_chroma": ("#BCD0E2", "pale blue-grey left end band"),
    "biotronik_orsiro_mission": ("#E8E9E9", "white carton (magenta label panel only)"),
    "biotronik_passeo_18": ("#E9EAE8", "white carton (dark-red label panel only)"),
    "biotronik_passeo_35_xeo": ("#EEEEEC", "white carton (plum label panel only)"),
    "biotronik_pk_papyrus": ("#E6E6E6", "white carton"),
    "biotronik_red_blue_unknown": ("#9D2F34", "red left end band (blue right end band is the smaller)"),
    "boston_scientific_mustang": ("#AE1A22", "red lower field of the front / red end bands"),
    "boston_scientific_synergy_xd": ("auto", "orange field of the front, orange end band"),
    "boston_scientific_v18_control_wire": ("#0E1116", "black-navy front and end band"),
    "bsci_mach1_guiding_catheter": ("#E4E6E7", "white-grey stock of the large face"),
    "bsci_nc_emerge_monorail": ("#2B2D30", "dark charcoal front and spine end band"),
    "bsci_permanent_sled_bag_carton": ("#1E3470", "navy end bands"),
    "bsci_wolverine_cutting_balloon": ("auto", "green front / end bands"),
    "elixir_desyne_x2": ("#143268", "navy end band"),
    "ivascular_essential_pro_balloon": ("#D81B73", "magenta end bands (lavender is the label only)"),
    "ivascular_oceanus_14_pro": ("#1C2865", "navy end bands"),
    "ivascular_oceanus_18": ("#1F1F5C", "dark navy end bands"),
    "ivascular_oceanus_35": ("#33348A", "indigo-navy end bands"),
    "medalliance_selution_slr_dcb": ("#E4E4E2", "white carton"),
    "medtronic_amphirion_deep": ("#E4E6E7", "white carton (dark slate block at one end only)"),
    "medtronic_export_advance_edge": ("#1F7050", "deep Medtronic green end band"),
    "medtronic_nc_euphora_edge": ("auto", "white / pale grey front (front photo)"),
    "medtronic_pacific_plus": ("auto", "silvery white front (front photo)"),
    "medtronic_prevail_dcb": ("#E6E6E6", "white carton (thin black ends)"),
    "medtronic_resolute_onyx": ("#EBD400", "yellow border of the front / yellow spine"),
    "meril_nexgen_balloon": ("#9B1B30", "dark red carton board under the white label"),
    "nse_alpha_scoring_balloon": ("#7A3A52", "plum-maroon end band"),
    "orbusneich_combo_plus": ("#1B3F9A", "navy end bands"),
    "orbusneich_sapphire_3": ("#E4E7E8", "pale grey-white carton"),
    "orbusneich_scoreflex": ("#BDB966", "khaki-olive end band"),
    "teleflex_guideliner": ("auto", "pale white-grey carton (front photo)"),
    "terumo_heartrail_ii_guiding_catheter": ("#E6E8EA", "white stock of the large face"),
    "terumo_navicross": ("auto", "very pale blue-white front (front photo)"),
    "terumo_ryurei_balloon": ("auto", "pale green-white carton (back photo)"),
    "trireme_tri_wedge": ("#2A1A2C", "dark purple-black carton, white label only"),
    "unknown_blue_des": ("#2A5A88", "blue end band"),
    "unknown_teal_boston_style": ("#3CC8A6", "teal cap (grey front is the artwork only)"),
}

# ----------------------------------------------------------------------------------------
# research images used for a front (or back) face.  quad: carton corners TL,TR,BR,BL in image px
# (picked on gridded views, then edge-snapped unless snap[i] is False; sides are top,right,bottom,left).
# ----------------------------------------------------------------------------------------
FRONTS = {
    "abbott_armada_35": dict(
        file="abbott_armada_35/front_4.0x20x80_medmat.png", quad=[(239, 100), (1038, 100), (1038, 859), (239, 859)],
        watermark=True, rule="upright", check="Armada 35 front: light-blue carton, Abbott logo, lilac 'Armada 35' label; matches the spine (blue ends, 'Armada 35')"),
    "abbott_xience_skypoint": dict(
        file="abbott_xience_skypoint/front_2.5x33_medmat.png", quad=[(204, 81), (1076, 81), (1076, 906), (204, 906)],
        watermark=True, rule="upright", check="XIENCE Skypoint front: light-blue carton, Abbott logo, purple/green label corner; matches the spine"),
    "boston_scientific_mustang": dict(
        file="boston_scientific_mustang/front_3.0x40x135.jpg", quad=[(137, 91), (858, 90), (859, 914), (138, 915)],
        watermark=False, rule="upright", check="Mustang over-the-wire front: navy header, red field, white label; matches the spine (red ends, MUSTANG panel)"),
    "boston_scientific_synergy_xd": dict(
        file="boston_scientific_synergy_xd/front_synergy_xd_3.0x20_mmi_watermark.jpg", quad=[(239, 60), (1040, 60), (1040, 899), (239, 899)],
        watermark=True, rule="upright", check="SYNERGY XD front: orange field, stent image; matches the orange XD spine"),
    "boston_scientific_v18_control_wire": dict(
        file="boston_scientific_v18_control_wire/front_V-18_300cm_long_taper.jpg", quad=[(11, 138), (932, 138), (932, 1112), (11, 1112)],
        watermark=False, rule="upright", check="V-18 Control Wire front: navy-black carton, yellow '018' corner; matches the black spine with yellow end"),
    "bsci_nc_emerge_monorail": dict(
        file="bsci_nc_emerge_monorail/front_4.5x15_mmi_watermark.jpg", quad=[(279, 82), (996, 88), (996, 880), (273, 876)],
        watermark=True, rule="upright", check="NC Emerge Monorail front: dark carton with Boston Scientific logo, white label, green MR corner; matches the spine (MR block)"),
    "bsci_wolverine_cutting_balloon": dict(
        file="bsci_wolverine_cutting_balloon/front.png", quad=[(17, 78), (868, 78), (868, 1005), (17, 1005)],
        snap=[True, True, False, True], watermark=False, rule="upright",
        check="Wolverine cutting balloon front: green carton; matches the green spine with MR block. Bottom ~8 % cropped (a seller business card stands in front of it)"),
    "medtronic_nc_euphora_edge": dict(
        file="medtronic_nc_euphora_edge/front_3.5x20_medmat.png", quad=[(149, 122), (1130, 122), (1130, 837), (149, 837)],
        watermark=True, rule="upright", check="NC Euphora front: white carton, teal band; matches the NC Euphora spine"),
    "medtronic_pacific_plus": dict(
        file="medtronic_pacific_plus/front_4.0x80x130_medmat.png", quad=[(282, 88), (1048, 88), (1048, 861), (282, 861)],
        watermark=True, rule="upright", check="Pacific Plus front: silvery white carton, green band; matches the Pacific Plus spine"),
    "medtronic_resolute_onyx": dict(
        file="medtronic_resolute_onyx/front_3.0x15_OTW_medmat.png", quad=[(152, 79), (1127, 79), (1127, 880), (152, 880)],
        watermark=True, rule="upright", check="Resolute Onyx front: yellow border, white panel; matches the yellow spine"),
    "teleflex_guideliner": dict(
        file="teleflex_guideliner/front_5.5F_V3.png", quad=[(40, 36), (1092, 36), (1092, 958), (40, 958)],
        snap=[True, True, False, True], watermark=False, rule="upright",
        check="GuideLiner V3 (5.5F) front: plain pale grey carton with a transparent-looking white label; same family as the white 6F spine (other French size). Bottom ~12 % cropped (a seller business card stands in front of it)"),
    "terumo_navicross": dict(
        file="terumo_navicross/front_NC35900_0.035_90cm.png", quad=[(228, 97), (1050, 97), (1050, 863), (228, 863)],
        watermark=True, rule="upright", check="NaviCross front: very pale carton with the blue 'NaviCross' wordmark and TERUMO logo; matches the spine"),
    "unknown_teal_boston_style": dict(
        file="unknown_teal_boston_style/front_synergy_megatron_3.5x8_mmi_watermark.png", quad=[(245, 67), (1034, 67), (1034, 893), (245, 893)],
        watermark=True, rule="upright",
        check="SYNERGY MEGATRON front: grey field, white label, green MR corner. Family identification is by elimination (research confidence low-medium); the grey/teal leaning carton on the hospital shelf photo matches this front"),
}
BACKS = {
    "terumo_ryurei_balloon": dict(
        file="terumo_ryurei_balloon/back_label_1.25x10.jpg", quad=[(102, 44), (865, 44), (865, 924), (102, 924)],
        watermark=False, rule="tab_opposite", rot180=True,
        check="Ryurei back label (white label with Ryurei 1.25 x 10 spec table), pale green-white carton; matches the Ryurei spine (Terumo green right end). 'Open here' tab is at the photo top, so the image is rotated 180 deg to put the spine edge at the top"),
}
REJECTED = {
    "biotronik_orsiro_mission": [("medicalecart_orsiro_404676_3.5x40.jpg", "plain Orsiro (REF 404676) carton, not the Orsiro Mission product of this family")],
    "biotronik_passeo_18": [("pouch_label_not_carton_4x120_130.jpg", "sterile pouch label, not the carton")],
    "biotronik_red_blue_unknown": [("medicalecart_pantera_leo_367039.jpg", "Pantera LEO NC carton (orange header) - research says it is NOT the red/blue box")],
    "boston_scientific_mustang": [("front_8.0x40x75.png", "same artwork, a seller card stands in front; the cleaner 3.0x40x135 image was used"),
                                  ("front_spine_angled_4.0x40x75.jpg", "angled view, not a straight face")],
    "boston_scientific_synergy_xd": [("front_original_synergy_2.5x16_older_gen.jpg", "older non-XD SYNERGY artwork (black/grey), different generation")],
    "boston_scientific_v18_control_wire": [("label_older_meditech_46-860_300cm.jpg", "older MEDI-TECH generation, white box")],
    "bsci_mach1_guiding_catheter": [("front_7F_JR4_full_length.png", "device carton photo is only 1.2 px/mm and the shelf photo already gives the label face at 1.95 px/mm; shelf photo used"),
                                    ("alt_view_7F_JR4_device_visible_through_window.png", "second view of the same carton with the catheter loop visible")],
    "bsci_wolverine_cutting_balloon": [("front_mmi_watermark.png", "blank-label duplicate with watermark; the clean front.png was used")],
    "bard_conquest_pta": [("front_6x20_75cm_MMI_watermark.png", "watermarked product shot with the device in front of the carton; the shelf photo of the real carton face (no watermark) is used")],
    "ivascular_essential_pro_balloon": [("front_render_marketing.png", "marketing render with the carton in a banner, not a carton photo")],
    "medtronic_export_advance_edge": [("front_6F_medicalecart.jpg", "partial, strongly keystoned rotated label face that fills the frame (no carton edges); kept plain")],
    "teleflex_guideliner": [("label_5.5F_V2_older_gen.jpg", "older V2 Vascular Solutions label crop, not a full face")],
}

# ----------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------
hexrgb = cs.hexrgb


def bgr(hexs):
    r, g, b = hexrgb(hexs)
    return np.array([b, g, r], np.uint8)


def plain(w, h, hexs):
    img = np.empty((h, w, 3), np.uint8)
    img[:] = bgr(hexs)
    return img


def snap_quad(img, quad, flags, half=12):
    """Snap each side of the quad (TL,TR,BR,BL; sides top,right,bottom,left) to the strongest straight
    edge within +-half px, then intersect adjacent sides.  Returns (quad, confidences)."""
    blur = cv2.GaussianBlur(img, (0, 0), 1.5)
    q = [np.array(p, float) for p in quad]
    lines, confs = [], []
    for i in range(4):
        p0, p1 = q[i], q[(i + 1) % 4]
        conf = None
        if flags[i]:
            a, b, conf, n = cs.snap_line(blur, p0, p1, half, t_lo=0.1, t_hi=0.9)
            if conf >= 1.2:
                p0, p1 = p0 + a * n, p1 + b * n
            else:
                conf = -conf                                  # negative: weak, kept the picked line
        lines.append((p0, p1))
        confs.append(None if conf is None else round(float(conf), 2))
    out = []
    for i in range(4):
        (a0, a1), (b0, b1) = lines[(i - 1) % 4], lines[i]
        M = np.array([a1 - a0, -(b1 - b0)]).T
        t = np.linalg.solve(M, b0 - a0)
        out.append(a0 + t[0] * (a1 - a0))
    return np.array(out), confs


def smooth_edge_strip(strip, axis, k=61):
    """strip: edge rows (axis 0) or columns (axis 1) of the image; returns one smoothed colour line."""
    med = np.median(strip, axis=axis).astype(np.float32)          # (len, 3)
    kk = k | 1
    med = cv2.GaussianBlur(med.reshape(-1, 1, 3), (1, kk), 0).reshape(-1, 3)
    return med


def fit_to(img, W, H):
    """Fit img into W x H: aspect stretch <= MAX_STRETCH, remaining space padded at the bottom (too wide)
    or equally left/right (too narrow) with a smoothed edge colour.  Returns (img, info)."""
    h0, w0 = img.shape[:2]
    ar, t = w0 / h0, W / H
    r = t / ar
    info = dict(source_px=[w0, h0], source_aspect=round(ar, 3), target_aspect=round(t, 3))
    if abs(r - 1) <= MAX_STRETCH:
        out = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA if w0 > W else cv2.INTER_LANCZOS4)
        info.update(aspect_change_pct=round(100 * (r - 1), 1), pad_pct=0.0, pad_side="none")
        return out, info
    f = min(max(r, 1 / (1 + MAX_STRETCH)), 1 + MAX_STRETCH)        # stretch toward the target as far as allowed
    eff = ar * f
    if eff > t:                                                    # too wide: fit width, pad bottom
        nw, nh = W, int(round(W / eff))
        body = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if w0 > nw else cv2.INTER_LANCZOS4)
        line = smooth_edge_strip(body[-6:], 0)
        pad = np.repeat(np.clip(line[None], 0, 255).astype(np.uint8), H - nh, axis=0)
        out = np.vstack([body, pad])
        info.update(aspect_change_pct=round(100 * (f - 1), 1), pad_pct=round(100 * (H - nh) / H, 1), pad_side="bottom")
    else:                                                          # too narrow: fit height, pad both sides
        nh, nw = H, int(round(H * eff))
        body = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if h0 > nh else cv2.INTER_LANCZOS4)
        padw = W - nw
        l, rr = padw // 2, padw - padw // 2
        ll = smooth_edge_strip(body[:, :6].transpose(1, 0, 2), 0)
        rl = smooth_edge_strip(body[:, -6:].transpose(1, 0, 2), 0)
        left = np.repeat(np.clip(ll[:, None], 0, 255).astype(np.uint8), l, axis=1)
        right = np.repeat(np.clip(rl[:, None], 0, 255).astype(np.uint8), rr, axis=1)
        out = np.hstack([left, body, right])
        info.update(aspect_change_pct=round(100 * (f - 1), 1), pad_pct=round(100 * padw / W, 1), pad_side="left+right")
    return out, info


def make_photo_face(spec, W, H):
    path = os.path.join(RESEARCH_DIR, spec["file"])
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("cannot read " + path)
    flags = spec.get("snap", [True, True, True, True])
    quad, confs = snap_quad(img, spec["quad"], flags)
    w0 = int(round((np.linalg.norm(quad[1] - quad[0]) + np.linalg.norm(quad[2] - quad[3])) / 2))
    h0 = int(round((np.linalg.norm(quad[3] - quad[0]) + np.linalg.norm(quad[2] - quad[1])) / 2))
    nat = cs.rectify(img, quad, w0, h0)
    if spec.get("rot180"):
        nat = cv2.rotate(nat, cv2.ROTATE_180)
    nat, wb = cs.white_balance(nat)
    out, info = fit_to(nat, W, H)
    out = cs.unsharp(out, sigma=1.0, amount=0.35, thresh=3)
    info.update(crop_quad_px=cs.rnd_pts(quad), picked_quad_px=cs.rnd_pts(spec["quad"]), edge_conf=confs, wb=wb)
    return out, info


def edge_hex(img):
    h, w = img.shape[:2]
    k = max(int(w * 0.03), 2)
    strip = np.concatenate([img[int(h * .2):int(h * .8), :k].reshape(-1, 3), img[int(h * .2):int(h * .8), w - k:].reshape(-1, 3)])
    b, g, r = [int(v) for v in np.median(strip, 0)]
    return "#%02X%02X%02X" % (r, g, b)


def face_px(mm_w, mm_h):
    """4 px/mm, capped so that no side exceeds MAX_PX."""
    ppm = min(float(PX_MM), MAX_PX / max(mm_w, mm_h))
    return int(round(mm_w * ppm)), int(round(mm_h * ppm)), ppm


def compose_large_front(fid, sp, dims, base_hex):
    """Large-face families: place the rectified label-face crop at true scale from the +X end (image right)."""
    fc = sp["front_face_crop"]
    crop = cv2.imread(os.path.join(CAT, fc["file"]), cv2.IMREAD_COLOR)
    H_mm, L_mm = dims["height"], dims["length"]
    ppm = min(float(PX_MM), MAX_PX / H_mm)
    W, H = int(round(H_mm * ppm)), int(round(L_mm * ppm))
    assert crop.shape[0] == H, (fid, crop.shape, H)
    cw = crop.shape[1]
    out = np.empty((H, W, 3), np.uint8)
    out[:] = bgr("#%02X%02X%02X" % tuple(fc["base_rgb"]))
    if fc.get("remainder_fill") == "last_rows" and cw < W:
        edge = crop[:, :24].astype(np.float32).mean(1)               # image-left columns = photo bottom end
        edge = cv2.GaussianBlur(edge.reshape(-1, 1, 3), (1, 9), 0).reshape(-1, 3)
        out[:, : W - cw] = np.clip(edge[:, None, :], 0, 255).astype(np.uint8)
    out[:, W - cw:] = crop[:, : cw]
    return out, dict(size_px=[W, H], px_per_mm=round(ppm, 3), crop_px=[cw, H], placed="right edge (+X, photo top end)",
                     covers_mm=fc["covers_height_mm"], remainder_fill=fc.get("remainder_fill", "base"))


def write_png(path, img, no_write):
    if not no_write:
        cs.save_png(path, img)


def load_urls():
    urls = {}
    import glob
    for f in sorted(glob.glob(os.path.join(CAT, "research", "*.json"))):
        for fid, v in json.load(open(f))["families"].items():
            for im in v.get("images") or []:
                urls[os.path.basename(im.get("file", ""))] = im.get("url")
    return urls


def rgb_str(hexs):
    r, g, b = hexrgb(hexs)
    return "plain(rgb %d,%d,%d)" % (r, g, b)


# ----------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--sheets", default="", help="directory for the contact sheets")
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args()
    dims_all = yaml.safe_load(open(os.path.join(CAT, "dims.yaml")))["families"]
    spines = yaml.safe_load(open(os.path.join(CAT, "spines.yaml")))["families"]
    urls = load_urls()
    only = [x for x in a.only.split(",") if x]
    result = {}
    sheet_rows = []
    for fid in sorted(dims_all):
        if only and fid not in only:
            continue
        d = dims_all[fid]["dims_mm"]
        height, length, thick = float(d["height"]), float(d["length"]), float(d["thickness"])
        sp = spines[fid]
        base_hex, base_why = BASE[fid]
        photo_imgs = {}
        tdir = os.path.join(CAT, "textures", fid)
        faces = {}
        notes_f = []
        # ---------------- top (owned by catalog_spines.py) ----------------
        src = sp.get("source")
        top_img = cv2.imread(os.path.join(tdir, "top.png"), cv2.IMREAD_COLOR)
        if src == "photo":
            faces["top"] = dict(source="photo_spine", file=sp["texture"]["file"], grade=sp.get("grade"),
                                notes=sp.get("occlusion_notes"))
        elif src == "redrawn":
            faces["top"] = dict(source="redrawn", file=sp["texture"]["file"], photo_reference=sp["photo_reference"]["file"],
                                notes="clean procedural redraw; warped photo kept as top_photo.png")
        else:
            faces["top"] = dict(source=rgb_str(sp["plain_hex"]), file=sp["texture"]["file"],
                                notes="plain base colour; the photo quad of this family is the large front face")
        fw, fh, fppm = face_px(height, length)
        if base_hex == "auto":                       # median body colour of the carton image (side strips)
            spec_ = FRONTS.get(fid) or BACKS.get(fid)
            im_, _ = make_photo_face(spec_, fw, fh)
            base_hex = edge_hex(im_)
            base_why += " [sampled %s from the carton image side strips]" % base_hex
        # ---------------- front ----------------
        fw, fh, fppm = face_px(height, length)
        front_img = None
        if "front_face_crop" in sp:
            front_img, info = compose_large_front(fid, sp, d, base_hex)
            faces["front"] = dict(source="photo_face", file="textures/%s/front.png" % fid,
                                  photo=sp["photo"], px_per_mm=info["px_per_mm"], size_px=info["size_px"],
                                  placement=info["placed"], covers_mm_of_height=info["covers_mm"],
                                  stretch_along_height=sp["front_face_crop"]["stretch_along_height"],
                                  clipped_photo_bottom_mm=sp["front_face_crop"]["clipped_photo_bottom_mm"],
                                  remainder_fill=info["remainder_fill"],
                                  notes="rectified label-face crop of the shelf photo at true scale from the +X (photo top) end; "
                                        "image right = +X = photo top, image up = +Y = photo left; remainder = %s" % info["remainder_fill"])
            if sp["front_face_crop"]["stretch_along_height"] != 1.0:
                faces["front"]["notes"] += ("; compressed by %.0f %% along the height because the photo shows a longer/narrower "
                                            "face than dims.yaml; photo bottom end (%.0f mm, partly hidden by torn cardboard) dropped" % (
                                                100 * (1 - sp["front_face_crop"]["stretch_along_height"]),
                                                sp["front_face_crop"]["clipped_photo_bottom_mm"]))
        elif fid in FRONTS:
            spec = FRONTS[fid]
            front_img, info = make_photo_face(spec, fw, fh)
            u = urls.get(os.path.basename(spec["file"]))
            faces["front"] = dict(source="research_image(%s)" % u, url=u, file="textures/%s/front.png" % fid,
                                  research_file=spec["file"], px_per_mm=round(fppm, 3), size_px=[fw, fh],
                                  orientation_rule=spec["rule"], watermark=spec["watermark"], family_check=spec["check"], fit=info)
            if spec["watermark"]:
                faces["front"]["notes"] = "seller watermark (translucent MMI logo/text) kept: no cheap safe removal; flagged"
        else:
            front_img = plain(int(round(height)), int(round(length)), base_hex)
            faces["front"] = dict(source=rgb_str(base_hex), file="textures/%s/front.png" % fid, px_per_mm=1,
                                  size_px=[int(round(height)), int(round(length))], notes="no usable carton front image; base colour: " + base_why)
        # ---------------- back ----------------
        if fid in BACKS:
            spec = BACKS[fid]
            back_img, info = make_photo_face(spec, fw, fh)
            u = urls.get(os.path.basename(spec["file"]))
            faces["back"] = dict(source="research_image(%s)" % u, url=u, file="textures/%s/back.png" % fid,
                                 research_file=spec["file"], px_per_mm=round(fppm, 3), size_px=[fw, fh],
                                 orientation_rule=spec["rule"], watermark=spec["watermark"], family_check=spec["check"], fit=info)
        else:
            back_img = plain(int(round(height)), int(round(length)), base_hex)
            faces["back"] = dict(source=rgb_str(base_hex), file="textures/%s/back.png" % fid, px_per_mm=1,
                                 size_px=[int(round(height)), int(round(length))], notes="no real back photo; base colour: " + base_why)
        # ---------------- bottom / left / right ----------------
        plains = dict(bottom=(int(round(height)), int(round(thick))), left=(int(round(length)), int(round(thick))),
                      right=(int(round(length)), int(round(thick))))
        for nm, (w, h) in plains.items():
            faces[nm] = dict(source=rgb_str(base_hex), file="textures/%s/%s.png" % (fid, nm), px_per_mm=1, size_px=[w, h],
                             notes="base colour: " + base_why)
        # ---------------- write ----------------
        write_png(os.path.join(tdir, "front.png"), front_img, a.no_write)
        write_png(os.path.join(tdir, "back.png"), back_img, a.no_write)
        for nm, (w, h) in plains.items():
            write_png(os.path.join(tdir, nm + ".png"), plain(w, h, base_hex), a.no_write)
        result[fid] = dict(dims_mm=dict(height=height, length=length, thickness=thick),
                           base_color=dict(hex=base_hex, reason=base_why), faces=faces)
        if fid in REJECTED:
            result[fid]["rejected_research_images"] = [dict(file=f, reason=r) for f, r in REJECTED[fid]]
        sheet_rows.append((fid, front_img, top_img, back_img, height, thick))
    if not a.no_write and not only:
        doc = dict(generator="tools/catalog_faces.py",
                   conventions="front/back W x H = height x length (front right=+X, back right=-X, up=+Y); top = spine; "
                               "photo-derived 4 px/mm capped at 2048 px, plain colours 1 px/mm",
                   orientation_rule="spine edge (+Y, image top) is opposite the 'Open here' tab when visible (tab_opposite), else artwork upright (upright)",
                   families=result)
        cs.save_text(os.path.join(CAT, "faces.yaml"),
                     yaml.safe_dump(cs.to_py(doc), sort_keys=True, default_flow_style=None, width=140))
    if a.sheets:
        os.makedirs(a.sheets, exist_ok=True)
        make_sheets(a.sheets, sheet_rows)
    n_photo = sum(1 for r in result.values() if r["faces"]["front"]["source"].startswith(("research", "photo")))
    print("families %d, with photo front %d" % (len(result), n_photo))


def make_sheets(outdir, rows):
    normal = [r for r in rows if r[4] < 500]
    large = [r for r in rows if r[4] >= 500]

    def cell(r, scale):
        fid, front, top, back, height, thick = r
        def rs(im, w_mm, h_mm):
            w, h = max(int(round(w_mm * scale)), 2), max(int(round(h_mm * scale)), 2)
            return cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
        fh = front.shape[0] / front.shape[1] * height
        f = rs(front, height, fh)
        b = rs(back, height, fh)
        t = rs(top, height, thick)
        gap = np.full((f.shape[0], 6, 3), 90, np.uint8)
        grid = np.hstack([f, gap, b])
        tpad = np.full((t.shape[0] + 4, grid.shape[1], 3), 90, np.uint8)
        tpad[2:2 + t.shape[0], : t.shape[1]] = t
        lab = np.full((16, grid.shape[1], 3), 40, np.uint8)
        cv2.putText(lab, fid + "   front | back; spine below", (3, 12), 0, 0.42, (255, 255, 255), 1)
        return np.vstack([lab, grid, tpad])

    def sheet(rs_, scale, cols, per, name):
        cells = [cell(r, scale) for r in rs_]
        for k in range(0, len(cells), per):
            grp = cells[k:k + per]
            rows_ = []
            for i in range(0, len(grp), cols):
                g = grp[i:i + cols]
                h = max(c.shape[0] for c in g)
                g = [np.pad(c, ((0, h - c.shape[0]), (0, 8), (0, 0))) for c in g]
                rows_.append(np.hstack(g))
            w = max(r.shape[1] for r in rows_)
            img = np.vstack([np.pad(r, ((0, 8), (0, w - r.shape[1]), (0, 0))) for r in rows_])
            cv2.imwrite(os.path.join(outdir, "%s_%d.png" % (name, k // per)), img)

    sheet(normal, 1.6, 3, 9, "faces")
    sheet(large, 0.5, 1, 3, "faces_large")


if __name__ == "__main__":
    main()
