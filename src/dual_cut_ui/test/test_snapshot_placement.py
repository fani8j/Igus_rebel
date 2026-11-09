"""Snapshot summary / freshness, overlay lookup, top-down SVG.  (The placement pre-check itself: test_placement_precheck.py.)"""
import json
import math
import os
import re
import time

import pytest

from dual_cut_ui.placement import compute_placement
from dual_cut_ui.snapshot import SnapshotError, SnapshotWatcher, find_overlay, load_snapshot, summarize
from dual_cut_ui.topdown import render_topdown_svg
from helpers import read_data, write_snapshot


def test_placement_accepts_lists_and_rejects_garbage():
    assert compute_placement([0, 0, 0], [0, -1, 0], [0, -0.5, 0])["yaw_deg"] == pytest.approx(-90.0)
    assert compute_placement(None, [0, -1, 0], [0, 0, 0]) is None
    assert compute_placement({"x": "a", "y": 0, "z": 0}, [0, -1, 0], [0, 0, 0]) is None
    assert compute_placement([0, 0, 0], [0, 0, 0], [float("nan"), 0, 0]) is None


def test_summary_of_the_recorded_snapshot():
    doc = json.loads(read_data("snapshot_MOCK_ONLY.json"))
    now = doc["acquisition_stamp_ns"] * 1e-9 + 30
    s = summarize(doc, now - 100.0, now, 240.0)
    assert s["age_s"] == pytest.approx(100.0) and s["file_age_s"] == pytest.approx(100.0)   # the OLDER of the two
    assert s["acquisition_age_s"] == pytest.approx(30.0)
    assert s["fresh"] is True and s["remaining_s"] == pytest.approx(140.0)
    assert s["cut_permission"]["status"] == "allowed" and s["cut_permission"]["flags"] == ["height_waived(support_hidden)"]
    assert s["points"]["seam_start"][0] == pytest.approx(0.4587, abs=1e-3)
    assert s["placement"]["yaw_deg"] == pytest.approx(-91.6, abs=0.2)
    stale = summarize(doc, now - 100.0, now + 300, 240.0)
    assert stale["fresh"] is False
    doc2 = dict(doc, acquisition_stamp_ns=0)
    assert summarize(doc2, now, now, 240.0)["fresh"] is False
    doc3 = dict(doc, acquisition_stamp_ns=int((now + 50) * 1e9))
    s3 = summarize(doc3, now, now, 240.0)
    assert s3["fresh"] is False and "future" in s3["problems"][0]


def test_cut_permission_with_reasons():
    doc = json.loads(read_data("snapshot_MOCK_ONLY.json"))
    doc["cut_permission"] = {"status": "blocked", "reasons": [{"code": "height_unavailable", "detail": "x"}, "odd"], "override": "allow_cut_blocked"}
    s = summarize(doc, time.time(), time.time(), 240.0)
    assert s["cut_permission"]["reasons"][0]["code"] == "height_unavailable" and s["cut_permission"]["reasons"][1]["code"] == "odd"
    assert s["cut_permission"]["override"] == "allow_cut_blocked"


def test_watcher_rereads_only_on_change_and_reports_errors(tmp_path):
    p = tmp_path / "s.json"
    w = SnapshotWatcher(str(p), time.time)
    assert w.summary(240)["available"] is False and "cannot read" in w.summary(240)["error"]
    write_snapshot(str(p), age_s=5)
    s = w.summary(240)
    assert s["available"] and s["fresh"] and 4 < s["age_s"] < 8
    p.write_text("{not json")
    s = w.summary(240)
    assert s["available"] is False and "not valid JSON" in s["error"]
    p.write_text("[1]")
    assert "not a JSON object" in w.summary(240)["error"]
    with pytest.raises(SnapshotError):
        load_snapshot(str(tmp_path / "nope.json"))


def make_run(out, capture_id, run, snapshot_bytes, overlays=("x_autolabel_v2.png",), mtime=None):
    d = out / "pipeline" / capture_id / run
    (d / "autolabel_v2_overlays").mkdir(parents=True)
    (d / "planner_snapshot.json").write_bytes(snapshot_bytes)
    for o in overlays:
        (d / "autolabel_v2_overlays" / o).write_bytes(b"png")
    if mtime:
        os.utime(d / "planner_snapshot.json", (mtime, mtime))
    return d


def test_overlay_lookup_picks_the_byte_identical_run(tmp_path):
    snap = tmp_path / "seam.json"
    write_snapshot(str(snap))
    raw = snap.read_bytes()
    cap = json.loads(raw)["capture_id"]
    out = tmp_path / "out"
    make_run(out, cap, "run1_other", raw + b" ", overlays=("other_autolabel_v2.png",))           # not identical
    good = make_run(out, cap, "run2_good", raw, overlays=("a_autolabel_v2_record.png", "a_autolabel_v2.png"))
    make_run(out, "other_capture", "run3", raw)                                                    # wrong capture dir
    found = find_overlay(str(out), str(snap))
    assert found["run_dir"] == str(good) and found["overlay"].endswith("a_autolabel_v2.png")
    assert not found["overlay"].endswith("_record.png")


def test_overlay_lookup_newest_identical_wins_and_missing_cases(tmp_path):
    snap = tmp_path / "seam.json"
    write_snapshot(str(snap))
    raw = snap.read_bytes()
    cap = json.loads(raw)["capture_id"]
    out = tmp_path / "out"
    now = time.time()
    make_run(out, cap, "old", raw, mtime=now - 100)
    new = make_run(out, cap, "new", raw, mtime=now - 5)
    assert find_overlay(str(out), str(snap))["run_dir"] == str(new)
    # a newer identical run without an overlay falls back to an older run that has one
    make_run(out, cap, "newest_no_overlay", raw, overlays=("a_autolabel_v2_record.png",), mtime=now)
    assert find_overlay(str(out), str(snap))["run_dir"] == str(new)
    assert find_overlay(str(tmp_path / "empty"), str(snap)) is None
    assert find_overlay(str(out), str(tmp_path / "missing.json")) is None


@pytest.mark.parametrize("cap", ["../x", "a/b", ".hidden", "", None, 5])
def test_overlay_lookup_ignores_hostile_capture_ids(tmp_path, cap):
    snap = tmp_path / "seam.json"
    write_snapshot(str(snap), mutate=lambda d: d.__setitem__("capture_id", cap))
    assert find_overlay(str(tmp_path), str(snap)) is None


def test_topdown_svg_shape_and_content():
    s = summarize(json.loads(read_data("snapshot_MOCK_ONLY.json")), time.time(), time.time(), 240.0)
    cov = {"seam_length_mm": 393.5, "left_mm": [0.0, 334.5], "right_mm": [322.5, 393.5]}
    svg = render_topdown_svg(s["points"], cov)
    assert svg.startswith("<svg") and svg.endswith("</svg>") and "<polygon" in svg
    assert "LEFT arm" in svg and "RIGHT arm" in svg and "Left arm top cut 0 to 334 mm" in svg and "Right arm top cut 322 to 394 mm" in svg
    assert "nan" not in svg.lower()
    import xml.dom.minidom
    xml.dom.minidom.parseString(svg)
    assert "stroke-width=\"8\"" not in render_topdown_svg(s["points"], None)
    pts = dict(s["points"], front_left=None)
    assert render_topdown_svg(pts, cov) is None


def test_topdown_legend_never_shows_negative_zero():
    s = summarize(json.loads(read_data("snapshot_MOCK_ONLY.json")), time.time(), time.time(), 240.0)
    cov = {"seam_length_mm": 395.1, "left_mm": [-4.7e-16, 395.1], "right_mm": [355.6, 395.1]}     # as the coordinator reports it
    svg = render_topdown_svg(s["points"], cov)
    assert "Left arm top cut 0 to 395 mm" in svg and "-0" not in svg.split("Left arm top cut")[1][:20]


def test_topdown_left_arm_is_drawn_on_the_left():
    """seam_start is at +y (the left arm's side): its screen x must be LEFT of seam_end's."""
    s = summarize(json.loads(read_data("snapshot_MOCK_ONLY.json")), time.time(), time.time(), 240.0)
    svg = render_topdown_svg(s["points"], {"seam_length_mm": 100.0, "left_mm": [0, 50], "right_mm": [50, 100]})
    m = re.findall(r'<line x1="([\d.]+)" y1="[\d.]+" x2="([\d.]+)" y2="[\d.]+" stroke="(#4aa3cf|#f0a040)"', svg)
    left = next(x for x in m if x[2] == "#4aa3cf")
    right = next(x for x in m if x[2] == "#f0a040")
    assert float(left[0]) < float(right[1]) and float(left[0]) < float(left[1]) and float(right[0]) < float(right[1])


def test_topdown_both_cuts_lie_on_the_seam_and_the_overlap_is_striped():
    """Operator 10-22: an offset right-arm line read as 'the cut is not connected'.  Both arms are drawn ON the seam."""
    s = summarize(json.loads(read_data("snapshot_MOCK_ONLY.json")), time.time(), time.time(), 240.0)
    svg = render_topdown_svg(s["points"], {"seam_length_mm": 393.0, "left_mm": [0.0, 373.0], "right_mm": [354.0, 393.0]})
    lines = re.findall(r'<line x1="([\d.]+)" y1="([\d.]+)" x2="([\d.]+)" y2="([\d.]+)" stroke="([^"]+)"([^/]*)/>', svg)
    seam = next(l for l in lines if l[4] == "#c6d3e2")
    (x0, y0, x1, y1) = (float(v) for v in seam[:4])
    def off(px, py):                                   # distance of a point from the seam line, in px
        return abs((x1 - x0) * (y0 - py) - (x0 - px) * (y1 - y0)) / ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
    cuts = [l for l in lines if l[4] in ("#4aa3cf", "#f0a040")]
    assert len(cuts) == 3                              # left, right, striped overlap
    for l in cuts:
        assert off(float(l[0]), float(l[1])) < 0.2 and off(float(l[2]), float(l[3])) < 0.2
    right = next(l for l in cuts if l[4] == "#f0a040")
    left = next(l for l in cuts if l[4] == "#4aa3cf")
    assert float(right[0]) < float(left[2])            # right starts before left ends: drawn as one joined cut
    stripe = [l for l in cuts if "stroke-dasharray" in l[5]]
    assert len(stripe) == 1 and stripe[0][4] == "#4aa3cf"
    assert "Both arms (overlap) 354 to 373 mm" in svg
    no_overlap = render_topdown_svg(s["points"], {"seam_length_mm": 100.0, "left_mm": [0, 40], "right_mm": [60, 100]})
    assert "overlap" not in no_overlap and "stroke-dasharray=\"4 4\"" not in no_overlap


def test_topdown_has_margins_and_nothing_overlaps():
    """Round 3: arm labels sit in side margins (not clipped / not on the carton), the legend does not touch the caption."""
    import xml.dom.minidom
    s = summarize(json.loads(read_data("snapshot_MOCK_ONLY.json")), time.time(), time.time(), 240.0)
    svg = render_topdown_svg(s["points"], {"seam_length_mm": 393.5, "left_mm": [0.0, 334.5], "right_mm": [322.5, 393.5]})
    doc = xml.dom.minidom.parseString(svg).documentElement
    W, H = [float(v) for v in doc.getAttribute("viewBox").split()[2:]]
    texts = {}
    for t in doc.getElementsByTagName("text"):
        texts[t.firstChild.data] = (float(t.getAttribute("x")), float(t.getAttribute("y")), float(t.getAttribute("font-size")), t.getAttribute("text-anchor"))
    width = lambda txt, fs: 0.62 * fs * len(txt)          # generous estimate of the rendered width      # noqa: E731
    poly = [tuple(float(c) for c in p.split(",")) for p in doc.getElementsByTagName("polygon")[0].getAttribute("points").split()]
    left_x, _, fs, _ = texts["LEFT arm"]
    assert left_x >= 0 and left_x + width("LEFT arm", fs) < min(p[0] for p in poly)            # clear of the carton, inside the canvas
    rx, _, fs, anchor = texts["RIGHT arm"]
    assert anchor == "end" and rx <= W and rx - width("RIGHT arm", fs) > max(p[0] for p in poly)
    legend_ys = [y for k, (x, y, f, a) in texts.items() if "arm top cut" in k]
    caption_y = texts["robot bases (seen from above, +x up, +y left)"][1]
    assert len(legend_ys) == 2 and caption_y - max(legend_ys) >= 14   # a full text line apart
    assert caption_y <= H and max(p[1] for p in poly) < min(legend_ys) - 12                    # carton ends above the legend
    for k, (x, y, f, a) in texts.items():
        w = width(k, f)
        x0 = x - w / 2 if a == "middle" else (x - w if a == "end" else x)
        assert x0 >= 0 and x0 + w <= W, k
