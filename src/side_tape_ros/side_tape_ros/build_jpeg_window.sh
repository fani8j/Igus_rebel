#!/usr/bin/env bash
# Optional prebuild of the windowed-JPEG helper next to the package (needs gcc + libjpeg headers).
# Not required: jpeg_window.py compiles the same source on first use into ~/.cache/side_tape_ros
# and perception_node falls back to cv2.imdecode if neither works.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
${CC:-cc} -O2 -Wall -shared -fPIC -o "$here/_jpeg_window.so" "$here/jpeg_window.c" -ljpeg
echo "built $here/_jpeg_window.so"
