#!/usr/bin/env bash
# Render the slide images (stocked cabinet + one view per carton, standing as on the shelf)
# from viewer.html in Orca's built-in browser, then trim the white margins.
# Usage: tools/render_slide_images.sh [out_dir]   (default ~/.cache/medical_boxes_slides)
# Needs an Orca terminal (the `orca` CLI) and a current viewer.html.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$HERE")"
OUT="${1:-$HOME/.cache/medical_boxes_slides}"
mkdir -p "$OUT"

orca tab create --url "file://$ROOT/viewer.html" --json >/dev/null
orca wait --text "Stocked medicine cabinet" --json >/dev/null

render() {  # name width height
  local js
  js="$(sed "s/__W__/$2/;s/__H__/$3/;s/__WHICH__/$1/" "$HERE/slide_render.js")"
  orca eval --expression "$js" --json > "$OUT/$1.json"
  python3 - "$OUT" "$1" <<'EOF'
import base64, json, sys
out, name = sys.argv[1], sys.argv[2]
d = json.load(open(f"{out}/{name}.json"))
if not d.get("ok"):
    sys.exit(f"render {name} failed: {d}")
open(f"{out}/{name}.png", "wb").write(base64.b64decode(d["result"]["result"].split(",", 1)[1]))
EOF
  rm -f "$OUT/$1.json"
}

render cabinet 1600 1800
for b in agent_monorail angiolite accuforce ultimaster_nagomi conqueror_nc_pro; do
  render "$b" 1400 1400
done
orca tab close --index 0 --json >/dev/null || true

python3 - "$OUT" <<'EOF'
import sys
from PIL import Image, ImageChops
out = sys.argv[1]
for n in ["cabinet", "agent_monorail", "angiolite", "accuforce", "ultimaster_nagomi", "conqueror_nc_pro"]:
    im = Image.open(f"{out}/{n}.png").convert("RGB")
    bb = ImageChops.difference(im, Image.new("RGB", im.size, (255, 255, 255))).point(
        lambda v: 255 if v > 6 else 0).getbbox()
    pad = 30
    bb = (max(bb[0] - pad, 0), max(bb[1] - pad, 0), min(bb[2] + pad, im.width), min(bb[3] + pad, im.height))
    im.crop(bb).save(f"{out}/{n}_crop.jpg", quality=90)
print(f"slide images in {out}")
EOF
