"""Placement pre-check (information only): left face reach estimate, region, right-holder/table note, thresholds, the shared
logic with tools/dual_run/check_placement.py, and that none of it can ever block anything.

Fixture test/data/placement_20261022.json = the 42 detections of 2026-10-22 and what the dual stack did with them in mock
(scan_results/today_plans_20261022/, rebuilt by tools/dual_run/make_placement_fixture.py)."""
import copy
import json
import math
import os
import subprocess
import sys
import time

import pytest

from dual_cut_ui import placement as P
from dual_cut_ui.snapshot import summarize
from helpers import Env, old_placement_limits, read_data, write_snapshot

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = json.load(open(os.path.join(HERE, "data", "placement_20261022.json")))["detections"]
SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))), "tools", "dual_run", "check_placement.py")

# ---- what the dual stack did (cell a = sequence, the live default; b = + unwind detour; c = phases top_first)
LEFT_FACE_FAILURES = {"t090925", "t093229"}                     # "left varying orientation search failed closed"
HOLDER_TABLE_REFUSED = {n for n, d in FIX.items() if any(c["cause"] == "RIGHT_HOLDER_TABLE" for c in d["plans"].values())}
PLANNED_A = {n for n, d in FIX.items() if d["plans"]["a"]["ok"]}


OLD_JSON = '{"left_reach_enabled": true, "two_arm_reach_enabled": false}'
OLD = old_placement_limits()      # the checks of 10-22 only; the two-arm reach check has its own file (test_placement_advisor.py)


def assess(name, limits=None):
    return P.assess_placement(FIX[name]["snapshot"], limits or OLD)


def moved(doc, dx=0.0, rot_deg=0.0):
    """The snapshot with the carton moved by dx (m, along x) and turned anticlockwise by rot_deg about the seam centre."""
    d = copy.deepcopy(doc)
    c = d["seam_center"]
    a = math.radians(rot_deg)
    for k, v in d.items():
        if isinstance(v, dict) and set("xyz") <= set(v):
            rx, ry = v["x"] - c["x"], v["y"] - c["y"]
            v["x"], v["y"] = c["x"] + math.cos(a) * rx - math.sin(a) * ry + dx, c["y"] + math.sin(a) * rx + math.cos(a) * ry
    return d


# ================================================================================================ left face reach (1a)
def test_fixture_is_the_42_detections():
    assert len(FIX) == 42 and LEFT_FACE_FAILURES <= set(FIX) and len(PLANNED_A) == 25
    assert {n for n, d in FIX.items() if d["plans"]["a"]["cause"] == "LEFT_FACE"} == LEFT_FACE_FAILURES


def test_left_reach_flags_exactly_the_two_left_face_failures():
    flagged = {n for n in FIX if assess(n)["left_reach"]["flag"]}
    assert flagged == LEFT_FACE_FAILURES
    for n in LEFT_FACE_FAILURES:
        w = [w for w in assess(n)["warnings"] if w["id"] == "left_reach"]
        assert len(w) == 1 and w[0]["level"] == "warn" and "toward -x" in w[0]["hint"]


def test_left_reach_separates_with_a_wide_margin():
    """The flag radius sits inside a gap of >= 15 mm between the closest plan and the closest failure."""
    r = {n: assess(n)["left_reach"]["wrist_radius_mm"] for n in FIX}
    worst_plan = max(v for n, v in r.items() if n not in LEFT_FACE_FAILURES)
    best_fail = min(r[n] for n in LEFT_FACE_FAILURES)
    flag = P.DEFAULT_LIMITS.wrist_flag_m * 1000
    assert worst_plan < flag < best_fail
    assert best_fail - worst_plan > 15 and flag - worst_plan > 6 and best_fail - flag > 6
    assert worst_plan == pytest.approx(543.5, abs=0.3) and best_fail == pytest.approx(560.4, abs=0.3)      # t074923 / t090925


@pytest.mark.parametrize("name", sorted(LEFT_FACE_FAILURES))
def test_left_reach_hint_numbers_really_fix_it(name):
    a = assess(name)
    hint = a["left_reach"]["hint"]
    assert 15 <= hint["move_x_mm"] <= 30                                           # move ~20-25 mm toward -x
    after = P.assess_placement(moved(FIX[name]["snapshot"], dx=-hint["move_x_mm"] / 1000.0), OLD)
    assert not after["left_reach"]["flag"] and after["left_reach"]["margin_mm"] > 8 - 1e-6   # hint margin: 8 mm below the flag
    assert "turn_ccw_deg" in hint and hint["turn_to_yaw_deg"] <= P.DEFAULT_LIMITS.region_yaw_deg[1]
    turned = P.assess_placement(moved(FIX[name]["snapshot"], rot_deg=hint["turn_ccw_deg"]), OLD)
    assert not turned["left_reach"]["flag"]


def test_the_rotation_hint_matches_the_mock_that_planned_at_yaw_minus_92():
    """left_face_fail_20261022 run C: t093229 turned to -92 deg (+5.05 deg about the seam centre) planned; the estimate puts it under the flag."""
    turned = P.assess_placement(moved(FIX["t093229"]["snapshot"], rot_deg=5.045), OLD)
    assert turned["yaw_deg"] == pytest.approx(-91.96, abs=0.05)
    assert not turned["left_reach"]["flag"]


def test_left_reach_without_the_middle_points_uses_the_corner_midpoints():
    doc = copy.deepcopy(FIX["t090925"]["snapshot"])
    ref = P.left_face_reach(doc)["wrist_radius_mm"]
    for k in ("back_middle_left", "front_middle_left"):
        doc.pop(k)
    assert P.left_face_reach(doc)["wrist_radius_mm"] == pytest.approx(ref, abs=3.0)
    doc.pop("back_left")
    assert P.left_face_reach(doc) is None and P.assess_placement(doc, OLD)["left_reach"] is None     # not estimable: no warning, no error


def test_left_reach_thresholds_are_configurable():
    lim = P.PlacementLimits.from_dict({"wrist_tilt_allowance_m": 0.0, "left_reach_enabled": True, "two_arm_reach_enabled": False})                 # flag at the bare 541 mm envelope
    flagged = {n for n in FIX if assess(n, lim)["left_reach"]["flag"]}
    assert LEFT_FACE_FAILURES < flagged and "t074923" in flagged
    off = P.PlacementLimits.from_dict({"left_reach_enabled": False})
    assert assess("t090925", off)["left_reach"] is None
    assert not [w for w in assess("t090925", off)["warnings"] if w["id"] == "left_reach"]


# ================================================================================================ region (1c)
def test_region_is_the_bounding_box_of_the_plans_of_cell_a():
    xs = [FIX[n]["snapshot"]["seam_center"]["x"] for n in PLANNED_A]
    ys = [FIX[n]["snapshot"]["seam_center"]["y"] for n in PLANNED_A]
    yaws = [assess(n)["yaw_deg"] for n in PLANNED_A]
    lim = P.DEFAULT_LIMITS
    assert (min(xs), max(xs)) == pytest.approx((0.380, 0.4346), abs=1e-3) and (min(ys), max(ys)) == pytest.approx((-0.2449, -0.1706), abs=1e-3)
    assert (min(yaws), max(yaws)) == pytest.approx((-92.46, -77.74), abs=0.02)
    assert lim.region_x_m[0] <= min(xs) and max(xs) <= lim.region_x_m[1] and lim.region_y_m[0] <= min(ys) and max(ys) <= lim.region_y_m[1]
    assert lim.region_yaw_deg[0] <= min(yaws) and max(yaws) <= lim.region_yaw_deg[1]
    # rounded to the numbers the study quotes
    assert (round(lim.region_x_m[0], 2), round(lim.region_x_m[1], 3)) == (0.38, 0.435) and lim.region_yaw_deg == (-92.5, -77.7)


def test_every_plan_of_cell_a_is_inside_and_most_refusals_are_outside():
    inside = {n for n in FIX if assess(n)["region"]["inside"]}
    assert PLANNED_A <= inside
    refused_inside = inside - PLANNED_A
    assert refused_inside <= {"t054203", "t054601", "t055736", "t060947", "t083258"}       # refusals inside: top-face check x4, holder/table x1
    outside_refused = (set(FIX) - PLANNED_A) - inside
    assert len(outside_refused) >= 10                                              # the box does exclude most of what was refused


def test_region_deviation_hints():
    a = assess("t090925")["region"]
    assert not a["inside"] and [d["axis"] for d in a["deviations"]] == ["x"]
    assert a["deviations"][0]["excess_mm"] == pytest.approx(21.4, abs=0.5) and "25 mm toward -x" in a["deviations"][0]["hint"]
    ang = math.radians(-60.0)
    s = [0.30, -0.50, 0.03]
    e = [s[0] + 0.4 * math.cos(ang), s[1] + 0.4 * math.sin(ang), 0.03]
    far = P.compute_placement(s, e, s)["region"]
    axes = {d["axis"]: d["hint"] for d in far["deviations"]}
    assert set(axes) == {"x", "y", "yaw"}
    assert "toward +x (away from the arms)" in axes["x"] and "toward +y (to the left)" in axes["y"] and axes["yaw"].startswith("turn the carton clockwise")
    near = P.compute_placement([0.5, 0.0, 0.03], [0.5, -0.4, 0.03], [0.5, 0.0, 0.03])["region"]
    hints = {d["axis"]: d["hint"] for d in near["deviations"]}
    assert "toward -x" in hints["x"] and "toward -y (to the right)" in hints["y"] and "yaw" not in hints


def test_region_yaw_hints_have_the_right_rotation_sense():
    def yaw_case(yaw_deg):
        a = math.radians(yaw_deg)
        c = [0.41, -0.20, 0.03]
        s = [c[0] - 0.2 * math.cos(a), c[1] - 0.2 * math.sin(a), 0.03]
        e = [c[0] + 0.2 * math.cos(a), c[1] + 0.2 * math.sin(a), 0.03]
        return P.compute_placement(s, e, c)["region"]
    assert yaw_case(-97.0)["deviations"][0]["hint"].startswith("turn the carton anticlockwise") and "by about 5 deg" in yaw_case(-97.0)["deviations"][0]["hint"]
    assert yaw_case(-70.0)["deviations"][0]["hint"].startswith("turn the carton clockwise")
    assert yaw_case(-85.0)["inside"] is True


# ================================================================================================ right holder (1b)
def test_right_table_note_catches_every_holder_table_refusal():
    flagged = {n for n in FIX if assess(n)["right_table"]["flag"]}
    assert HOLDER_TABLE_REFUSED <= flagged and len(HOLDER_TABLE_REFUSED) == 8
    # the soft note: what it also flags are known: the top-face refusal t054203 (holder/table in the diagnostic cell d) and t090346 (plans in b/c)
    assert flagged - HOLDER_TABLE_REFUSED == {"t054203", "t090346"}
    for n in flagged:
        w = [w for w in assess(n)["warnings"] if w["id"] == "right_holder_table"][0]
        assert w["level"] == "note" and "may" in w["title"].lower() and "table" in w["text"]
    assert not {n for n in PLANNED_A} & flagged


def test_right_table_thresholds_are_configurable():
    strict = P.PlacementLimits.from_dict({"right_table_x_max_m": 0.40})
    assert {n for n in FIX if assess(n, strict)["right_table"]["flag"]} < {n for n in FIX if assess(n)["right_table"]["flag"]}
    assert not assess("t075330", P.PlacementLimits.from_dict({"right_table_enabled": False}))["right_table"]["flag"]
    assert not assess("t075330", P.PlacementLimits.from_dict({"right_table_yaw_max_deg": -95.0}))["right_table"]["flag"]
    assert assess("t075330", P.PlacementLimits.from_dict({"right_table_yaw_max_deg": -85.0}))["right_table"]["flag"]


def test_right_table_edge_cases():
    lim = P.DEFAULT_LIMITS
    assert P.right_table_risk(lim.right_table_x_max_m, lim.right_table_yaw_max_deg)["flag"] is True       # both inclusive
    assert P.right_table_risk(lim.right_table_x_max_m + 1e-4, -95.0)["flag"] is False
    assert P.right_table_risk(0.38, lim.right_table_yaw_max_deg + 0.1)["flag"] is False


def test_limits_json_and_validation():
    assert P.PlacementLimits.from_json("") == P.PlacementLimits()
    lim = P.PlacementLimits.from_json('{"right_table_x_max_m": 0.41, "region_x_m": [0.3, 0.5], "left_reach_enabled": false}')
    assert lim.right_table_x_max_m == 0.41 and lim.region_x_m == (0.3, 0.5) and lim.left_reach_enabled is False
    for bad in ('{"nope": 1}', '[1]', '{"right_table_x_max_m": "a"}', '{"region_x_m": [1]}', '{"left_reach_enabled": 1}', "{", '{"wrist_envelope_m": NaN}'):
        with pytest.raises(ValueError):
            P.PlacementLimits.from_json(bad)
    assert P.PlacementLimits.from_dict(lim.to_dict()) == lim                                         # round trip


# ================================================================================================ legacy window (1c)
def test_the_old_window_is_kept_and_still_computes_what_the_old_script_did():
    old = P.legacy_check({"x": 0.43, "y": -0.136 + 0.2, "z": 0}, {"x": 0.43, "y": -0.136 - 0.2, "z": 0}, {"x": 0.43, "y": -0.136, "z": 0})
    assert old["yaw_deg"] == pytest.approx(-90.0) and old["yaw_ok"] and old["offset_mm"] == pytest.approx(0, abs=1e-6) and old["offset_ok"]
    assert old["yaw_window_deg"] == [-94.5, -89.5] and old["planned_centre_m"] == [0.43, -0.136] and old["offset_max_mm"] == 20.0
    a = math.radians(-97.3)
    s = [0.43, -0.136, 0.0]
    e = [0.43 + 0.4 * math.cos(a), -0.136 + 0.4 * math.sin(a), 0.0]
    o = P.legacy_check(s, e, s)
    assert o["rotation_hint"]["direction"] == "ANTI-CLOCKWISE" and o["rotation_hint"]["deg"] == pytest.approx(5.3, abs=0.05) and not o["yaw_ok"]
    assert P.LEGACY_WINDOW_20261018["centre_xy_m"] == (0.430, -0.136)
    # today's placements lie almost entirely outside it (that is why it was replaced)
    inside_old = [n for n in FIX if P.legacy_check(*(FIX[n]["snapshot"][k] for k in ("seam_start", "seam_end", "seam_center")))["offset_ok"]]
    assert inside_old == []


# ================================================================================================ info only, robustness
def test_garbage_never_raises():
    assert P.assess_placement({}) is None
    assert P.assess_placement({"seam_start": [0, 0, 0], "seam_end": [float("nan"), 1, 0], "seam_center": [0, 0, 0]}) is None
    doc = {"seam_start": [0.4, 0, 0], "seam_end": [0.4, -0.4, 0], "seam_center": [0.4, -0.2, 0], "back_middle_left": "x", "front_middle_left": [1, 2]}
    a = P.assess_placement(doc)
    assert a is not None and a["left_reach"] is None and a["info_only"] is True


def test_summary_survives_a_failing_placement_check(monkeypatch):
    from dual_cut_ui import snapshot as S
    monkeypatch.setattr(S, "assess_placement", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    doc = json.loads(read_data("snapshot_MOCK_ONLY.json"))
    s = summarize(doc, time.time(), time.time(), 240.0)
    assert s["placement"] is None and s["points"]["seam_start"] is not None and s["cut_permission"]["status"] == "allowed"


def test_summary_carries_the_warnings_and_the_limits():
    doc = FIX["t090925"]["snapshot"]
    s = summarize(dict(doc, acquisition_stamp_ns=int(time.time() * 1e9)), time.time(), time.time(), 240.0, OLD)
    ids = [w["id"] for w in s["placement"]["warnings"]]
    assert ids == ["left_reach", "outside_region"] and s["placement"]["info_only"] is True
    assert s["placement"]["limits"]["right_table_x_max_m"] == P.DEFAULT_LIMITS.right_table_x_max_m
    s2 = summarize(dict(doc, acquisition_stamp_ns=int(time.time() * 1e9)), time.time(), time.time(), 240.0,
                   P.PlacementLimits.from_dict({"left_reach_enabled": False, "two_arm_reach_enabled": False}))
    assert [w["id"] for w in s2["placement"]["warnings"]] == ["outside_region"]


def test_placement_warnings_never_block_detect_plan_or_cut(tmp_path):
    """Operator decision: placement is information only.  The recorded mock snapshot (centre x 0.453) triggers the left reach
    warning, and detect, plan and the Cut pre-check still go through."""
    env = Env(tmp_path, placement_limits=OLD)
    try:
        st = env.core.state("op")
        assert [w["id"] for w in st["snapshot"]["placement"]["warnings"]][:1] == ["left_reach"]
        assert st["cut"]["can_cut"] is True and st["cut"]["blockers"] == []
        assert env.act("detect_and_plan")[0] == 200
        job = env.wait_job()
        assert job.state == "done" and all(s["state"] == "ok" for s in job.steps)
        assert "PLACEMENT WARNING" in job.steps[0]["detail"] and len(job.steps[0]["detail"]) < 80
        env.bridge.set_ready()
        status, reply = env.act("execute", {"plan_seq": 3})
        assert status == 200 and reply["ok"] is True
    finally:
        env.close()


def test_config_limits_reach_the_page_state(tmp_path):
    env = Env(tmp_path, placement_limits=P.PlacementLimits.from_dict({"left_reach_enabled": False, "two_arm_reach_enabled": False, "region_x_m": [0.30, 0.60]}))
    try:
        pl = env.core.state("op")["snapshot"]["placement"]
        assert pl["warnings"] == [] and pl["region"]["inside"] is True and pl["limits"]["left_reach_enabled"] is False
    finally:
        env.close()


def test_a_snapshot_with_every_warning_still_gives_a_cut_ready_state(tmp_path):
    env = Env(tmp_path, placement_limits=OLD)
    try:
        write_snapshot(env.snap, mutate=lambda d: d.update({k: FIX["t075330"]["snapshot"][k] for k in ("seam_start", "seam_end", "seam_center")}))
        st = env.core.state("op")
        assert "right_holder_table" in [w["id"] for w in st["snapshot"]["placement"]["warnings"]]
        assert st["cut"]["can_cut"] is True
    finally:
        env.close()


# ================================================================================================ the shared script
def run_script(*args):
    r = subprocess.run([sys.executable, "-I", SCRIPT, *args], capture_output=True, text=True, timeout=30)
    return r.returncode, r.stdout, r.stderr


def test_check_placement_script_shares_the_logic(tmp_path):
    for name in ("t090925", "t075330", "t134030"):
        p = tmp_path / f"{name}.json"
        p.write_text(json.dumps(FIX[name]["snapshot"]))
        rc, out, err = run_script(str(p), "--limits", OLD_JSON)
        a = assess(name)
        assert rc == 0, err
        assert f"seam yaw  {a['yaw_deg']:.1f} deg" in out
        for w in a["warnings"]:
            assert f"[{w['level']}] {w['title']}: {w['hint']}" in out
        if not a["warnings"]:
            assert "no placement warning" in out
    rc, out, _ = run_script(str(tmp_path / "t090925.json"), "--legacy", "--limits", OLD_JSON)
    assert "legacy 2026-10-18 window (outdated)" in out and "PROBABLY OUT OF REACH" in out


def test_check_placement_script_limits_and_errors(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps(FIX["t075330"]["snapshot"]))
    rc, out, _ = run_script(str(p), "--limits", '{"right_table_enabled": false}')
    assert rc == 0 and "no known problem" in out and "MAY BE REFUSED" not in out
    rc, out, err = run_script(str(p), "--limits", '{"bogus": 1}')
    assert rc == 2 and "unknown placement limit" in err
    assert run_script(str(tmp_path / "missing.json"))[0] == 2
    (tmp_path / "bad.json").write_text("{}")
    assert run_script(str(tmp_path / "bad.json"))[0] == 2
