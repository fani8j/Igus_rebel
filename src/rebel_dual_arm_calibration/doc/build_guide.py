#!/usr/bin/env python3
"""Build calibration_guide.html from the template, embedding the board previews and an example detection.

Usage: python3 build_guide.py [--targets-dir ~/dual_arm_targets] [--detection <jpg>]
"""
import argparse
import base64
import datetime
import io
import os

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))


def data_uri(path, width, crop_top_frac=None):
    img = Image.open(os.path.expanduser(path)).convert("RGB")
    if crop_top_frac:
        img = img.crop((0, 0, img.width, int(img.height * crop_top_frac)))
    img.thumbnail((width, 10 * width))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets-dir", default="~/dual_arm_targets")
    ap.add_argument("--detection", default=os.path.join(
        REPO, "scan_results/dual_arm_calibration_e2e_20261007/images/left_cap_0003_detections.jpg"))
    ap.add_argument("--out", default=os.path.join(HERE, "calibration_guide.html"))
    a = ap.parse_args()
    html = open(os.path.join(HERE, "calibration_guide.template.html"), encoding="utf-8").read()
    html = html.replace("{{DATE}}", datetime.date.today().isoformat())
    html = html.replace("{{IMG_BOARD_LEFT}}", data_uri(os.path.join(a.targets_dir, "target_left.png"), 520, 0.56))
    html = html.replace("{{IMG_BOARD_TABLE}}", data_uri(os.path.join(a.targets_dir, "target_table.png"), 620, 0.53))
    html = html.replace("{{IMG_DETECTION}}", data_uri(a.detection, 900))
    html = html.replace("{{IMG_STICKERS}}", data_uri(os.path.join(a.targets_dir, "touch_stickers.png"), 520, 0.85))
    assert "{{" not in html, "unfilled placeholder"
    open(a.out, "w", encoding="utf-8").write(html)
    print(f"wrote {a.out} ({os.path.getsize(a.out) / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
