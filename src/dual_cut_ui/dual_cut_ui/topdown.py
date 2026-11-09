"""Top-down SVG of the carton: corners and seam from the snapshot, the seam interval each arm cuts on top.

View: seen from above, standing behind the two robot bases and looking forward (+x of base_link is UP on the page,
+y of base_link is to the LEFT).  The left arm (+y side) therefore appears on the left, the right arm on the right.
Seam positions ``s`` of the coverage are millimetres along seam_start -> seam_end.  No ROS; numbers only, so the string
is safe to put into the page with innerHTML.
"""
import html
from typing import Optional

from .snapshot import CORNER_KEYS

LEFT_COLOUR = "#4aa3cf"
RIGHT_COLOUR = "#f0a040"
# Canvas layout: side margins hold the arm labels, the bottom band holds the legend (up to three lines) and the caption, so nothing overlaps
W, H = 640, 470
MARGIN_X, MARGIN_TOP, BAND_BOTTOM = 112, 34, 96
CUT_WIDTH = 8
OVERLAP_DASH = "4 4"


def _f(v: float) -> str:
    return f"{v:.1f}"


def render_topdown_svg(points: dict, coverage: Optional[dict]) -> Optional[str]:
    """``points``: name -> [x, y, z] (snapshot summary ``points``); ``coverage``: parsed top_coverage or None."""
    corners = [points.get(k) for k in CORNER_KEYS]
    s0, s1 = points.get("seam_start"), points.get("seam_end")
    if any(c is None for c in corners) or s0 is None or s1 is None:
        return None
    pts = corners + [s0, s1]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    x_min, x_max, y_min, y_max = min(xs), max(xs), min(ys), max(ys)
    span_x, span_y = max(x_max - x_min, 1e-3), max(y_max - y_min, 1e-3)
    avail_w, avail_h = W - 2 * MARGIN_X, H - MARGIN_TOP - BAND_BOTTOM
    k = min(avail_w / span_y, avail_h / span_x)
    ox = MARGIN_X + (avail_w - span_y * k) / 2.0
    oy = MARGIN_TOP + (avail_h - span_x * k) / 2.0

    def proj(p):
        return ox + (y_max - p[1]) * k, oy + (x_max - p[0]) * k

    poly = " ".join(f"{_f(u)},{_f(v)}" for u, v in (proj(c) for c in corners))
    (u0, v0), (u1, v1) = proj(s0), proj(s1)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img" '
           f'aria-label="Top-down view of the carton and the seam intervals each arm cuts">',
           f'<rect width="{W}" height="{H}" fill="#0b1320" rx="8"/>',
           f'<polygon points="{poly}" fill="#223650" stroke="#8ea6bf" stroke-width="2"/>',
           f'<line x1="{_f(u0)}" y1="{_f(v0)}" x2="{_f(u1)}" y2="{_f(v1)}" stroke="#c6d3e2" stroke-width="2" '
           f'stroke-dasharray="6 5"/>']

    def seg(a, b, colour, extra=""):            # a piece of the seam itself (fractions a..b): both arms cut ON the seam
        ax, ay = u0 + (u1 - u0) * a, v0 + (v1 - v0) * a
        bx, by = u0 + (u1 - u0) * b, v0 + (v1 - v0) * b
        return (f'<line x1="{_f(ax)}" y1="{_f(ay)}" x2="{_f(bx)}" y2="{_f(by)}" stroke="{colour}" '
                f'stroke-width="{CUT_WIDTH}"{extra}/>')
    legend, spans = [], {}
    if coverage and coverage.get("seam_length_mm"):
        length = coverage["seam_length_mm"]
        for side, key, colour in (("Left arm", "left_mm", LEFT_COLOUR), ("Right arm", "right_mm", RIGHT_COLOUR)):
            iv = coverage.get(key)
            if not iv:
                continue
            a, b = max(0.0, iv[0] / length), min(1.0, iv[1] / length)
            if b <= a:
                continue
            spans[key] = (a, b)
            out.append(seg(a, b, colour, ' stroke-linecap="butt"'))
            legend.append((colour, None, f"{side} top cut {round(iv[0])} to {round(iv[1])} mm"))   # round() -> int: no "-0"
        if len(spans) == 2:
            a, b = max(spans["left_mm"][0], spans["right_mm"][0]), min(spans["left_mm"][1], spans["right_mm"][1])
            if b > a:                               # both arms cut here: blue/orange stripes, so it reads as one joined cut
                out.append(seg(a, b, LEFT_COLOUR, f' stroke-dasharray="{OVERLAP_DASH}"'))
                legend.append((LEFT_COLOUR, RIGHT_COLOUR, f"Both arms (overlap) {round(a * length)} to {round(b * length)} mm"))
    out += [f'<circle cx="{_f(u0)}" cy="{_f(v0)}" r="5" fill="#eef3fa"/>',
            f'<text x="{_f(u0 + 8)}" y="{_f(v0 - 12)}" fill="#eef3fa" font-size="14">seam start</text>',
            f'<circle cx="{_f(u1)}" cy="{_f(v1)}" r="5" fill="#eef3fa"/>',
            f'<text x="{_f(u1 - 8)}" y="{_f(v1 - 12)}" fill="#eef3fa" font-size="14" text-anchor="end">seam end</text>']
    for i, (colour, colour2, text) in enumerate(legend):
        y = H - 76 + 19 * i
        out.append(f'<rect x="16" y="{y - 12}" width="14" height="14" fill="{colour}"/>')
        if colour2:                                 # split swatch for the overlap
            out.append(f'<rect x="23" y="{y - 12}" width="7" height="14" fill="{colour2}"/>')
        out.append(f'<text x="38" y="{y}" fill="#eef3fa" font-size="15">{html.escape(text)}</text>')
    mid = _f(MARGIN_TOP + (H - MARGIN_TOP - BAND_BOTTOM) / 2.0)
    out += [f'<text x="10" y="{mid}" fill="{LEFT_COLOUR}" font-size="16" font-weight="700">LEFT arm</text>',
            f'<text x="{W - 10}" y="{mid}" fill="{RIGHT_COLOUR}" font-size="16" font-weight="700" text-anchor="end">RIGHT arm</text>',
            f'<text x="{W / 2:.0f}" y="{H - 10}" fill="#9db0c6" font-size="14" text-anchor="middle">robot bases (seen from above, +x up, +y left)</text>',
            f'<text x="{W / 2:.0f}" y="20" fill="#9db0c6" font-size="14" text-anchor="middle">away from the robots (+x)</text>',
            '</svg>']
    return "".join(out)
