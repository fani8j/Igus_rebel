#!/usr/bin/env python3
"""Add the stocked-cabinet slide and one detail slide per carton to a copy of the
MoveIt reasoning deck, before its References slide, in the deck's own CONTENT layout,
colours, footer, sources line and slide numbers. The source deck is not modified.

Masses, shelf orientation (up_end) and textures are read from boxes.yaml,
shelf_layout.yaml and textures/. The 3D renders come from tools/render_slide_images.sh.

    tools/render_slide_images.sh                      # -> ~/.cache/medical_boxes_slides
    python3 tools/make_slides.py --src ~/Downloads/MoveIt_Reasoning.pptx \
        --out ~/Downloads/MoveIt_Reasoning_with_cartons.pptx
"""
import argparse
import copy
import os

import yaml
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, PP_PLACEHOLDER
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.dirname(HERE)
A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
P = "{http://schemas.openxmlformats.org/presentationml/2006/main}"

NAVY, TEAL, CARD = "0B1F3A", "0E7C86", "EEF3F8"
TEXT, MUTED, ORANGE, GREEN = "1A2433", "5B6B82", "E89B2C", "2E9E6B"

SHELVES = [  # top to bottom, as in the picture
    ("Shelf 06", "280", "19 Angiolite · 18 Accuforce · 24 Conqueror"),
    ("Shelf 05", "290", "61 Agent Monorail"),
    ("Shelf 04", "310", "57 Angiolite"),
    ("Shelf 03", "260", "30 Agent Monorail · 37 Conqueror"),
    ("Shelf 02", "230", "57 Accuforce (only carton short enough)"),
    ("Shelf 01", "300", "57 Ultimaster Nagomi"),
]
BOXES = [
    dict(key="agent_monorail", title="Carton 1: Boston Scientific Agent Monorail",
         device="Paclitaxel-coated PTCA balloon, 2.75 × 30 mm",
         ref="REF H74939222273010 · GTIN 08714729831587", size="230 × 250 × 15 mm",
         shelf="Shelf 03 (30) and Shelf 05 (61) · 91 total", orient="MR end up · front faces left",
         link=("Confirmed", GREEN, "green MR band continues onto both faces")),
    dict(key="angiolite", title="Carton 2: iVascular Angiolite",
         device="Sirolimus-eluting coronary stent, 4.0 × 19 mm",
         ref="REF SCCDSR14150400019 · GTIN 08435387305004", size="240 × 250 × 16 mm",
         shelf="Shelf 04 (57) and Shelf 06 (19) · 76 total", orient="iVascular end up · front faces left",
         link=("Convention", ORANGE, "photo match too close to call")),
    dict(key="accuforce", title="Carton 3: Terumo Accuforce",
         device="Non-compliant PTCA balloon, 2.5 × 12 mm",
         ref="REF DC-RM2512HHW", size="220 × 250 × 16 mm",
         shelf="Shelf 02 (57) and Shelf 06 (18) · 75 total", orient="Accuforce end up · front faces right",
         link=("Convention", ORANGE, "weak photo evidence, same direction")),
    dict(key="ultimaster_nagomi", title="Carton 4: Terumo Ultimaster Nagomi",
         device="Sirolimus-eluting coronary stent, 3.0 × 24 mm",
         ref="REF DE-RS3024ASM", size="250 × 295 × 16 mm",
         shelf="Shelf 01 (57) · 57 total", orient="Ultimaster Nagomi end up · front faces right",
         link=("Convention", ORANGE, "photo match too close to call")),
    dict(key="conqueror_nc_pro", title="Carton 5: APT Medical Conqueror NC Pro",
         device="Non-compliant PTCA balloon, 4.00 × 8 mm",
         ref="REF 134400080", size="230 × 250 × 12 mm",
         shelf="Shelf 03 (37) and Shelf 06 (24) · 61 total", orient="Lollipop end up · front faces left",
         link=("Convention", ORANGE, "photo match too close to call")),
]


def rgb(h):
    return RGBColor.from_string(h)


class Deck:
    def __init__(self, src, img_dir, tmp_dir):
        self.prs = Presentation(src)
        self.img, self.tmp = img_dir, tmp_dir
        self.layout = next(l for l in self.prs.slide_layouts if l.name == "CONTENT")
        self.template = self.prs.slides[7]  # "Step 4": copy its sources line and slide number
        self.src_line = next(s for s in self.template.shapes
                             if s.has_text_frame and s.text_frame.text.startswith("Sources"))
        self.sld_num = next(s for s in self.template.shapes if s.name.startswith("Slide Number"))

    def jpg(self, src, name, max_side, rotate=0):
        im = Image.open(src).convert("RGB")
        if rotate:
            im = im.rotate(rotate, expand=True)  # degrees counter-clockwise
        im.thumbnail((max_side, max_side))
        p = os.path.join(self.tmp, name)
        im.save(p, quality=88)
        return p, im.width / im.height

    @staticmethod
    def title_of(slide):
        return next((ph for ph in slide.placeholders if ph.placeholder_format.type == PP_PLACEHOLDER.TITLE), None)

    def new_slide(self, title, sources, notes):
        s = self.prs.slides.add_slide(self.layout)
        self.title_of(s).text = title
        for el in (self.src_line, self.sld_num):
            s.shapes._spTree.append(copy.deepcopy(el._element))
        src = [sh for sh in s.shapes if sh.has_text_frame and sh.text_frame.text.startswith("Sources")][-1]
        src.text_frame.paragraphs[0].runs[0].text = sources
        ns = s.notes_slide
        if ns.notes_text_frame is None:  # this deck's notes master has no body placeholder
            body = next(sh for sh in self.template.notes_slide.placeholders
                        if sh.placeholder_format.type == PP_PLACEHOLDER.BODY)
            ns.shapes._spTree.append(copy.deepcopy(body._element))
        tf = ns.notes_text_frame
        for extra in tf.paragraphs[1:]:
            extra._p.getparent().remove(extra._p)
        para = tf.paragraphs[0]
        for r in para.runs[1:]:
            r._r.getparent().remove(r._r)
        (para.runs[0] if para.runs else para.add_run()).text = notes
        return s


def text(slide, x, y, w, h, runs, size=14, color=TEXT, bold=False, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP):
    """runs: a string, or a list of paragraphs, each a list of (text, {size,bold,color})."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    paras = [[(runs, {})]] if isinstance(runs, str) else runs
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        for t, o in para:
            r = p.add_run()
            r.text = t
            r.font.size = Pt(o.get("size", size))
            r.font.bold = o.get("bold", bold)
            r.font.color.rgb = rgb(o.get("color", color))
    return tb


def shape(slide, kind, x, y, w, h, fill, adj, name):
    s = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    s.adjustments[0] = adj
    s.fill.solid()
    s.fill.fore_color.rgb = rgb(fill)
    s.line.color.rgb = rgb(fill)
    s.shadow.inherit = False
    s.name = name
    return s


def picture(slide, path, aspect, x, y, w, h, name):
    """Fit inside (w, h), centred."""
    pw, ph = (h * aspect, h) if w / h > aspect else (w, w / aspect)
    pic = slide.shapes.add_picture(path, Inches(x + (w - pw) / 2), Inches(y + (h - ph) / 2),
                                   Inches(pw), Inches(ph))
    pic.name = name
    return pic


def cabinet_slide(d):
    s = d.new_slide("Isaac Sim Scene: Stocked Medicine Cabinet", "Sources: [15] [16]",
                    "The operator's MedicineCabinet.usd stocked with the five carton assets. Cartons "
                    "stand upright like books, spine out, length into the shelf, grouped by product, "
                    "matching the hospital shelf photos; which spine end is on top follows the "
                    "operator's own shelf. The cabinet is a static collider; every carton is a dynamic "
                    "rigid body (instanced reference). Shelf heights are read from the cabinet; Shelf 02 "
                    "has 230 mm clear, so only Accuforce (220 mm) fits there, with 9 mm headroom.")
    p, a = d.jpg(os.path.join(d.img, "cabinet_crop.jpg"), "s_cabinet.jpg", 1600)
    picture(s, p, a, 0.6, 1.3, 3.0, 5.5, "cabinet render")
    p, a = d.jpg(os.path.join(ASSETS, "usd/previews/cabinet_front.png"), "s_elevation.jpg", 1600)
    picture(s, p, a, 3.75, 1.3, 2.75, 5.5, "front elevation")
    text(s, 0.6, 6.82, 3.0, 0.2, "3D view", size=10, color=MUTED, align=PP_ALIGN.CENTER)
    text(s, 3.75, 6.82, 2.75, 0.2, "Front view: spines as stocked", size=10, color=MUTED,
         align=PP_ALIGN.CENTER)
    for i, (big, small) in enumerate([("360", "cartons"), ("5", "carton types"),
                                      (d.checks, "checks passed")]):
        x = 6.9 + i * 1.98
        shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, x, 1.3, 1.83, 1.25, CARD, 0.04, f"stat {i}")
        text(s, x, 1.38, 1.83, 0.7, big, size=32, color=TEAL if i < 2 else GREEN, bold=True,
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        text(s, x, 2.05, 1.83, 0.35, small, size=12, color=MUTED, align=PP_ALIGN.CENTER)
    text(s, 6.9, 2.8, 5.83, 0.35, "What is on each shelf", size=14, color=NAVY, bold=True)
    for i, (sh, clear, what) in enumerate(SHELVES):
        y = 3.2 + i * 0.47
        shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, 6.9, y, 5.83, 0.41, CARD, 0.04, f"shelf row {i}")
        text(s, 7.05, y, 0.95, 0.41, sh, size=12, color=NAVY, bold=True, anchor=MSO_ANCHOR.MIDDLE)
        text(s, 8.0, y, 0.95, 0.41, f"{clear} mm", size=12, color=MUTED, anchor=MSO_ANCHOR.MIDDLE)
        text(s, 8.95, y, 3.7, 0.41, what, size=12, color=TEXT, anchor=MSO_ANCHOR.MIDDLE)
    text(s, 6.9, 6.15, 5.83, 0.6,
         [[("Open in Isaac Sim: ", {"bold": True, "color": NAVY}),
           ("sim_assets/medical_boxes/usd/medicine_cabinet_filled.usda", {"color": TEXT})],
          [("Static cabinet collider · dynamic cartons", {"color": MUTED})]], size=12)


def carton_slide(d, b, spec, up_end):
    k = b["key"]
    me = spec["mass_estimate_g"]
    mass = f"{spec['mass_kg'] * 1000:.0f} g estimate · range {me['low']}–{me['high']} g"
    split = f"carton {me['carton']} g + contents {me['contents']} g"
    s = d.new_slide(b["title"], "Sources: [15] [16]; dimensions by caliper, mass estimated",
                    f"{b['device']}. Carton {b['size']} (height × length × thickness, caliper "
                    f"measured); no manufacturer publishes carton dimensions or weight, so the mass is "
                    f"estimated as carton board plus contents (catheter, hoop, pouches, leaflet): "
                    f"{mass} ({split}). Textures: spine, front and back rectified and cleaned from the "
                    f"operator's photos. On the shelf: {b['shelf']}, {b['orient']}. Spine-to-face link: "
                    f"{b['link'][0].lower()} ({b['link'][2]}).")
    p, a = d.jpg(os.path.join(d.img, f"{k}_crop.jpg"), f"s_{k}.jpg", 1400)
    shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, 0.6, 1.3, 4.3, 5.5, "FFFFFF", 0.04, "render frame")
    picture(s, p, a, 0.75, 1.4, 4.0, 5.3, "carton render")

    rows = [("Device", b["device"]), ("REF / GTIN", b["ref"]), ("Carton (h × l × t)", b["size"]),
            ("Mass", mass), ("In the cabinet", b["shelf"]), ("Standing", b["orient"]),
            ("USD asset", f"sim_assets/medical_boxes/usd/{k}.usda")]
    shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, 5.2, 1.3, 7.53, 0.25 + 0.42 * len(rows), CARD, 0.04, "spec card")
    for i, (lab, val) in enumerate(rows):
        y = 1.42 + i * 0.42
        text(s, 5.45, y, 1.9, 0.38, lab, size=13, color=MUTED, anchor=MSO_ANCHOR.MIDDLE)
        text(s, 7.35, y, 5.25, 0.38, val, size=13, color=TEXT, bold=(lab == "Carton (h × l × t)"),
             anchor=MSO_ANCHOR.MIDDLE)

    word, col, why = b["link"]
    shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, 5.2, 4.67, 1.45, 0.36, col, 0.5, "link pill")
    text(s, 5.2, 4.67, 1.45, 0.36, word.upper(), size=11, color="FFFFFF", bold=True,
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    text(s, 6.8, 4.67, 5.9, 0.36, [[("Spine-to-face link: ", {"bold": True, "color": NAVY}),
                                     (why, {"color": TEXT})]], size=13, anchor=MSO_ANCHOR.MIDDLE)

    text(s, 5.2, 5.2, 7.5, 0.3, "Textures from the operator's photos", size=12, color=NAVY, bold=True)
    tex = os.path.join(ASSETS, "textures", k)
    # each face as it looks with the carton standing on the shelf (world Z up): with the "-X"
    # spine end up the front texture turns 90 deg clockwise, with "+X" counter-clockwise, and
    # the back the other way round
    rot = -90 if up_end == "-X" else 90
    pf, af = d.jpg(os.path.join(tex, "front.png"), f"t_{k}_front.jpg", 500, rotate=rot)
    pb, ab = d.jpg(os.path.join(tex, "back.png"), f"t_{k}_back.jpg", 500, rotate=-rot)
    sp = os.path.join(tex, "top.png")
    with Image.open(sp) as im:
        asp = im.width / im.height
    picture(s, pf, af, 5.2, 5.55, 1.6, 1.25, "front texture")
    picture(s, pb, ab, 6.9, 5.55, 1.6, 1.25, "back texture")
    picture(s, sp, asp, 8.75, 5.75, 3.98, 0.45, "spine texture")
    text(s, 5.2, 6.82, 3.3, 0.2, "front · back, as standing on the shelf", size=10, color=MUTED,
         align=PP_ALIGN.CENTER)
    text(s, 8.75, 6.3, 3.98, 0.2, "spine (labelled edge, faces out of the cabinet)", size=10, color=MUTED)


def finish(d):
    prs = d.prs
    ids = prs.slides._sldIdLst
    items = list(ids)
    refs, new = items[10], items[11:]
    for el in new:
        ids.remove(el)
    for el in new:
        refs.addprevious(el)

    ref_slide = next(sl for sl in prs.slides if d.title_of(sl) is not None and d.title_of(sl).text == "References")
    body = max((sh for sh in ref_slide.shapes if sh.has_text_frame and not sh.is_placeholder),
               key=lambda sh: len(sh.text_frame.paragraphs))
    last = body.text_frame.paragraphs[-1]._p
    for t in ["Carton caliper measurements and photos (spine, front, back, own shelf), operator, 2026-10-05",
              "Medical carton USD assets and stocked cabinet — sim_assets/medical_boxes "
              "(igus_rebel_ros2, box-3d branch)"]:
        p = copy.deepcopy(last)
        for r in p.findall(f"{A}r")[1:]:
            p.remove(r)
        p.find(f"{A}r").find(f"{A}t").text = t
        last.addnext(p)
        last = p

    # unique shape ids on every new slide (deep-copied shapes keep their source ids)
    for sl in prs.slides:
        t = d.title_of(sl)
        if t is not None and (t.text.startswith("Carton ") or t.text.startswith("Isaac Sim Scene")):
            for n, el in enumerate(sl.shapes._spTree.iter(f"{P}cNvPr"), start=2):
                el.set("id", str(n))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=os.path.expanduser("~/Downloads/MoveIt_Reasoning.pptx"))
    ap.add_argument("--out", default=os.path.expanduser("~/Downloads/MoveIt_Reasoning_with_cartons.pptx"))
    ap.add_argument("--images", default=os.path.expanduser("~/.cache/medical_boxes_slides"))
    ap.add_argument("--checks", default="", help="total checks passed, shown on the cabinet slide")
    a = ap.parse_args()
    if os.path.abspath(a.src) == os.path.abspath(a.out):
        raise SystemExit("--out must differ from --src; the source deck is never modified")
    spec = yaml.safe_load(open(os.path.join(ASSETS, "boxes.yaml")))["boxes"]
    up = yaml.safe_load(open(os.path.join(ASSETS, "shelf_layout.yaml")))["up_end"]
    tmp = os.path.join(a.images, "slide_tmp")
    os.makedirs(tmp, exist_ok=True)
    d = Deck(a.src, a.images, tmp)
    d.checks = a.checks or "all"
    cabinet_slide(d)
    for b in BOXES:
        carton_slide(d, b, spec[b["key"]], up[b["key"]])
    finish(d)
    d.prs.save(a.out)
    print("wrote", a.out, len(d.prs.slides), "slides")


if __name__ == "__main__":
    main()
