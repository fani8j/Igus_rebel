#!/usr/bin/env python3
"""Print sheet (A4, 600 dpi, exact scale) with the touch crosshair stickers of config/targets.yaml `touch_stickers`.

Fit stickers (default 101-104) widen the touch-only base; check stickers (default 111, 112) are never used in any fit
and give the held-out accuracy (one goes on a ~40 mm block, one >= 150 mm from the table-board centre).
Positions are not measured: stick them anywhere both arms can reach, flat, and do not move them until both arms are done.

Usage: make_sticker_sheet.py [--targets <targets.yaml>] [--out-dir ~/dual_arm_targets]
"""
import argparse
import os

import yaml
from PIL import Image, ImageDraw, ImageFont

DPI = 600
PX_PER_MM = DPI / 25.4
A4_MM = (210.0, 297.0)


def mm(v):
    return int(round(v * PX_PER_MM))


def font(size_px):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf"):
        if os.path.exists(path):
            return ImageFont.truetype(path, size_px)
    return ImageFont.load_default()


def draw_sticker(d, cx, cy, sid, kind):
    """60 x 60 mm sticker: cut frame, thin crosshair with a gap at the centre, 1 mm centre ring, label."""
    half = mm(30)
    d.rectangle([cx - half, cy - half, cx + half, cy + half], outline=150, width=2)
    arm, gap, w = mm(22), mm(1.5), max(2, mm(0.25))
    for sx, sy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        d.line([cx + sx * gap, cy + sy * gap, cx + sx * arm, cy + sy * arm], fill=0, width=w)
    r = mm(0.5)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=0, width=max(1, mm(0.12)))
    for t in (-20, -10, 10, 20):  # 10 mm ticks to judge the centre by eye
        d.line([cx + mm(t), cy - mm(1.2), cx + mm(t), cy + mm(1.2)], fill=0, width=w)
        d.line([cx - mm(1.2), cy + mm(t), cx + mm(1.2), cy + mm(t)], fill=0, width=w)
    label = f"{sid}  {'FIT' if kind == 'fit' else 'CHECK (held-out)'}"
    d.text((cx - half + mm(2), cy - half + mm(2)), label, fill=0, font=font(mm(4)))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default=os.path.join(here, "..", "config", "targets.yaml"))
    ap.add_argument("--out-dir", default="~/dual_arm_targets")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.targets)) or {}
    st = cfg.get("touch_stickers") or {}
    items = [(int(i), "fit") for i in st.get("fit_points", [])] + [(int(i), "check") for i in st.get("check_points", [])]
    if not items:
        raise SystemExit("no touch_stickers in " + a.targets)
    if len(items) > 6:
        raise SystemExit("this sheet holds 6 stickers")
    page = Image.new("L", (mm(A4_MM[0]), mm(A4_MM[1])), 255)
    d = ImageDraw.Draw(page)
    d.text((mm(15), mm(10)), "TOUCH STICKERS - knife tip on the crosshair centre (ring 1 mm)", fill=0, font=font(mm(5)))
    for k, (sid, kind) in enumerate(items):
        col, row = k % 2, k // 2
        draw_sticker(d, mm(55 + col * 100), mm(55 + row * 75), sid, kind)
    notes = [
        "Print at 100 %. Cut along the grey frames. Stick FLAT; do not move until both arms are recorded.",
        "FIT stickers: spread about +-150 mm around the table board (wider = better rotation estimate).",
        "CHECK stickers: never used in the fit. Put one on a ~40 mm block (off the board plane) and one",
        ">= 150 mm from the table-board centre. Both arms touch every sticker, same ids.",
    ]
    for i, t in enumerate(notes):
        d.text((mm(15), mm(265) + i * mm(6)), t, fill=0, font=font(mm(3.6)))
    out = os.path.expanduser(a.out_dir)
    os.makedirs(out, exist_ok=True)
    pdf = os.path.join(out, "touch_stickers.pdf")
    page.save(pdf, "PDF", resolution=DPI)
    page.save(os.path.join(out, "touch_stickers.png"), dpi=(DPI, DPI))
    print(f"wrote {pdf} ({len(items)} stickers: {', '.join(str(i) for i, _ in items)})")


if __name__ == "__main__":
    main()
