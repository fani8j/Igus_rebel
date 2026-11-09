"""Two-arm reach model + placement advisor (information only).  Ground truth cells = the mock coordinator study of 2026-11-07
(scan_results/negative_yaw_20261107, placement_workspace_20261107) on the template snapshot test/data/template_20261022_082449.json,
moved like harness/make_placement.py: yaw anticlockwise about the top-face centre, then (dx, dy) mm."""
import copy
import json
import math
import os
import time

import pytest

from dual_cut_ui import placement as P
from dual_cut_ui import placement_advisor as A

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = json.load(open(os.path.join(HERE, "data", "template_20261022_082449.json")))
CORNERS = ("back_left", "back_right", "front_left", "front_right")


def place(doc, dx_mm, dy_mm, yaw_deg):
    """make_placement.place for the points the model uses (corners, seam points, normal)."""
    d = copy.deepcopy(doc)
    cx = sum(d[k]["x"] for k in CORNERS) / 4
    cy = sum(d[k]["y"] for k in CORNERS) / 4
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    for k, v in list(d.items()):
        if isinstance(v, dict) and set("xyz") <= set(v):
            if k == "surface_normal":
                v["x"], v["y"] = c * v["x"] - s * v["y"], s * v["x"] + c * v["y"]
            else:
                x, y = v["x"] - cx, v["y"] - cy
                v["x"], v["y"] = cx + c * x - s * y + dx_mm / 1000, cy + s * x + c * y + dy_mm / 1000
    return d


def at(cell):
    yaw, dx, dy = cell
    return place(TEMPLATE, dx, dy, yaw)


# (yaw, dx, dy) -> coordinator class: F = both faces >= 90 %, o = planned < 90 %, x = refused
GOOD = [(0, -80, 0), (0, -120, 40), (-10, -120, -80)]                  # F and the model agrees with >= 28 mm to spare
BAD = [(0, 0, 0), (-10, 0, 0), (-20, 0, 0), (-30, -120, -80)]          # o / x / x / o, the model fails by >= 40 mm
NEAR_EDGE_F = [(0, -40, 0), (-10, -80, -40)]                           # F in the study; the model says fail by 1..5 mm (fitted threshold)


def run(cell, **lim):
    return P.assess_placement(at(cell), P.PlacementLimits.from_dict(lim) if lim else None)


@pytest.mark.parametrize("cell", GOOD)
def test_good_cells_pass(cell):
    a = run(cell)
    t = a["two_arm_reach"]
    assert t["status"] == "ok" and t["ok"] and t["margin_mm"] > 20
    assert "two_arm_reach" not in [w["id"] for w in a["warnings"]] and t["advice"] is None


@pytest.mark.parametrize("cell", BAD)
def test_bad_cells_fail_with_a_warning_first(cell):
    a = run(cell)
    t = a["two_arm_reach"]
    assert t["status"] == "fail" and not t["ok"] and t["margin_mm"] < -30
    w = a["warnings"][0]
    assert w["id"] == "two_arm_reach" and w["level"] == "warn" and w["text"] and w["hint"].startswith("To fix:")
    assert a["info_only"] is True


def test_the_binding_constraint_is_named_per_arm():
    assert run((0, 0, 0))["two_arm_reach"]["binding"] == "left_wrist"              # left start corner too far from the left base
    assert "left arm wrist" in run((0, 0, 0))["warnings"][0]["text"]
    t = run((-20, -120, -120))["two_arm_reach"]                                    # clockwise + close to the right base
    assert t["margins_mm"]["right_tip"] < t["margins_mm"]["left_tip"] and t["margins_mm"]["right_wrist"] < 20


def test_marginal_is_a_soft_note():
    a = run((0, -40, 0), left_wrist_max_m=0.5215)                                  # passes by ~ 0.5 mm
    t = a["two_arm_reach"]
    assert t["status"] == "marginal" and 0 <= t["margin_mm"] < 15
    assert a["warnings"][0]["id"] == "two_arm_reach" and a["warnings"][0]["level"] == "note"


@pytest.mark.parametrize("cell", BAD + NEAR_EDGE_F)
def test_every_advised_option_passes_the_model_when_applied(cell):
    """The hint, applied to the snapshot (independent implementation of the move), satisfies the model with the advice margin."""
    t = run(cell)["two_arm_reach"]
    assert t["advice"]["options"], "an option exists for these cells"
    for o in t["advice"]["options"][:2]:
        moved = place(TEMPLATE, cell[1] + o["dx_mm"], cell[2] + o["dy_mm"], cell[0] + o["turn_deg"])    # same pivot: the centre
        after = P.assess_placement(moved)["two_arm_reach"]
        assert after["margin_mm"] >= 20 - 0.5 and after["status"] == "ok"
        assert abs(o["dx_mm"]) <= 250 and abs(o["dy_mm"]) <= 250 and abs(o["turn_deg"]) <= 30
        assert o["margin_mm"] == pytest.approx(after["margin_mm"], abs=1.0)


def test_hint_wording():
    h = run((0, 0, 0))["warnings"][0]["hint"]
    # (lead 2026-11-07) the template pose no longer gets a turn past +10 deg (unvalidated range), so the cheapest fix is a move
    assert "toward the arms (-x)" in h
    from dual_cut_ui import placement_advisor as pa
    assert "anticlockwise (seen from above" in pa._turn_words(8.0) and "clockwise (seen" in pa._turn_words(-8.0)
    h2 = run((-20, 0, 0))["warnings"][0]["hint"]
    assert "toward the arms (-x)" in h2
    assert A.option_text({"dx_mm": 30.0, "dy_mm": -60.0, "turn_deg": 0.0}) == "move it 30 mm away from the arms (+x) and 60 mm toward the right arm (-y)"
    assert A.option_text({"dx_mm": 0.0, "dy_mm": 40.0, "turn_deg": -7.0}).startswith("turn it 7 deg clockwise")
    assert "toward the left arm (+y)" in A.option_text({"dx_mm": 0.0, "dy_mm": 40.0, "turn_deg": 0.0})


def test_nothing_found_says_so_plainly():
    a = run((0, 0, 0), advice_max_shift_mm=20, advice_max_turn_deg=2)
    assert a["warnings"][0]["hint"].startswith("No placement within ±20 mm / ±2 deg found")
    assert "too large or too far" in a["warnings"][0]["hint"]
    assert a["two_arm_reach"]["advice"]["options"] == []


def test_a_combined_option_when_neither_alone_works():
    """Shrink both searches so that only turn + move works; the result must apply cleanly."""
    cell = (0, 40, 0)
    t = run(cell, advice_max_shift_mm=60, advice_max_turn_deg=8)["two_arm_reach"]
    assert t["status"] == "fail" and t["advice"]["options"]
    o = t["advice"]["options"][0]
    assert o["kind"] == "turn_move" and o["turn_deg"] != 0 and (o["dx_mm"] or o["dy_mm"])
    after = P.assess_placement(place(TEMPLATE, cell[1] + o["dx_mm"], cell[2] + o["dy_mm"], cell[0] + o["turn_deg"]))["two_arm_reach"]
    assert after["ok"] and after["margin_mm"] >= 20 - 0.5


def test_a_larger_box_does_not_crash_and_is_answered_sanely():
    """Synthetic 330 x 570 mm top face (along x 330, across y 570), flat, left edge at y ~ 0 like the template."""
    for x0 in (0.20, 0.30, 0.45):
        c = {"back_left": (x0 + 0.33, 0.0), "front_left": (x0, 0.0), "front_right": (x0, -0.57), "back_right": (x0 + 0.33, -0.57)}
        doc = {k: {"x": v[0], "y": v[1], "z": 0.03} for k, v in c.items()}
        a = P.assess_placement(dict(doc, seam_start={"x": x0 + 0.2, "y": 0.0, "z": 0.03}, seam_end={"x": x0 + 0.2, "y": -0.57, "z": 0.03},
                                    seam_center={"x": x0 + 0.2, "y": -0.285, "z": 0.03}))
        t = a["two_arm_reach"]
        assert t["status"] in ("ok", "marginal", "fail") and math.isfinite(t["margin_mm"])
        if t["status"] != "ok":
            assert a["warnings"][0]["id"] == "two_arm_reach"
            for o in t["advice"]["options"]:           # sane: inside the search range and verified by the model
                assert abs(o["dx_mm"]) <= 250 and abs(o["dy_mm"]) <= 250 and abs(o["turn_deg"]) <= 30
            if not t["advice"]["options"]:
                assert "No placement" in a["warnings"][0]["hint"]
    # the along-x length changes where the back_left start corner is: a longer box needs to sit closer to the arms
    short = {k: {"x": TEMPLATE[k]["x"], "y": TEMPLATE[k]["y"], "z": TEMPLATE[k]["z"]} for k in CORNERS}
    far = copy.deepcopy(short)
    for k in ("back_left", "back_right"):
        far[k]["x"] += 0.1
    ms = lambda d: A.evaluate(A.pose_from_doc(dict(d, surface_normal={"x": 0, "y": 0, "z": 1})), P.DEFAULT_LIMITS)["margins_mm"]["left_wrist"]  # noqa: E731
    assert ms(far) < ms(short) - 50


def test_missing_normal_is_derived_from_the_corners_and_garbage_is_ignored():
    d = {k: TEMPLATE[k] for k in CORNERS}
    a = A.two_arm_reach(d, P.DEFAULT_LIMITS)
    assert a is not None and a["status"] in ("ok", "marginal", "fail")
    assert A.two_arm_reach({"back_left": [0, 0, 0]}, P.DEFAULT_LIMITS) is None
    bad = dict(TEMPLATE, back_left={"x": float("nan"), "y": 0, "z": 0})
    assert A.two_arm_reach(bad, P.DEFAULT_LIMITS) is None
    p = P.assess_placement(bad)
    assert p is not None and p["two_arm_reach"] is None                          # the rest of the assessment is unaffected


def test_flipped_normal_and_tiny_box_do_not_crash():
    d = dict(place(TEMPLATE, 0, 0, 0), surface_normal={"x": 0, "y": 0, "z": -1})
    assert A.two_arm_reach(d, P.DEFAULT_LIMITS)["status"] in ("ok", "marginal", "fail")
    tiny = {k: {"x": 0.4 + 0.001 * i, "y": 0.001 * j, "z": 0.0} for k, (i, j) in
            zip(CORNERS, ((1, 0), (1, 1), (0, 0), (0, 1)))}
    assert A.two_arm_reach(tiny, P.DEFAULT_LIMITS) is not None


def test_limits_override_and_disable():
    base = run((0, 0, 0))["two_arm_reach"]
    loose = run((0, 0, 0), left_wrist_max_m=0.60)["two_arm_reach"]
    assert base["status"] == "fail" and loose["margin_mm"] == pytest.approx(base["margin_mm"] + 80, abs=0.5)
    assert run((0, 0, 0), two_arm_reach_enabled=False)["two_arm_reach"] is None
    assert [w["id"] for w in run((0, 0, 0), two_arm_reach_enabled=False)["warnings"]] == []
    no_advice = run((0, 0, 0), advice_enabled=False)
    assert no_advice["warnings"][0]["id"] == "two_arm_reach" and no_advice["two_arm_reach"]["advice"] is None
    # the right base position is a limit too: moving it 100 mm away from the carton frees the right tip guard
    r = run((-20, -120, -120), right_base_m=[0.0, -0.8, 0.0])["two_arm_reach"]["margins_mm"]["right_tip"]
    assert r > run((-20, -120, -120))["two_arm_reach"]["margins_mm"]["right_tip"] + 100
    with pytest.raises(ValueError):
        P.PlacementLimits.from_dict({"left_wrist_max": 0.5})
    with pytest.raises(ValueError):
        P.PlacementLimits.from_dict({"right_base_m": [0.0, 1.0]})
    assert P.PlacementLimits.from_json('{"advice_margin_mm": 30, "tool_offset_mm": [-42.2, 89.6, -157.2]}').advice_margin_mm == 30.0
    assert P.PlacementLimits.from_dict(P.DEFAULT_LIMITS.to_dict()) == P.DEFAULT_LIMITS


def test_old_left_reach_is_off_by_default_but_still_available():
    assert P.DEFAULT_LIMITS.left_reach_enabled is False
    assert run((0, 0, 0))["left_reach"] is None
    assert run((0, 0, 0), left_reach_enabled=True)["left_reach"] is not None


def test_the_other_checks_are_unchanged_and_come_after():
    a = P.assess_placement(at((0, 0, 0)))
    ids = [w["id"] for w in a["warnings"]]
    assert ids[0] == "two_arm_reach" and set(ids) <= {"two_arm_reach", "right_holder_table", "outside_region"}


def test_runtime_of_a_full_advice_search():
    P.assess_placement(at((0, 0, 0)))                    # warm up
    worst = 0.0
    for cell in ((0, 0, 0), (-20, 0, 0), (-30, 0, 0)):
        d = at(cell)
        t = time.perf_counter()
        P.assess_placement(d)
        worst = max(worst, time.perf_counter() - t)
    assert worst < 1.0


# ---- model validity window (lead, 2026-11-07): beyond +10 deg anticlockwise the model agreed with the planner on only 39 %
def test_turn_from_ref_is_zero_for_template():
    from dual_cut_ui import placement_advisor as pa
    from dual_cut_ui.placement import PlacementLimits
    import json, pathlib
    doc = json.loads((pathlib.Path(__file__).parent / "data" / "template_20261022_082449.json").read_text())
    pose = pa.pose_from_doc(doc)
    assert abs(pa.turn_from_ref_deg(pose, PlacementLimits())) < 0.5
    assert abs(pa.turn_from_ref_deg(pa.turned(pose, 25.0), PlacementLimits()) - 25.0) < 0.5


def test_large_ccw_turn_is_unvalidated_not_ok():
    """+40 deg is beyond the validated range (-30..+30 since 2026-11-08; the model's passes at +35..+50 were all wrong)."""
    from dual_cut_ui import placement_advisor as pa
    from dual_cut_ui.placement import PlacementLimits
    import json, pathlib
    lim = PlacementLimits()
    doc = json.loads((pathlib.Path(__file__).parent / "data" / "template_20261022_082449.json").read_text())
    pose = pa.turned(pa.pose_from_doc(doc), 40.0)       # (2026-11-08: the validated range now reaches +30)
    d = dict(doc)
    for k, v in pose["c"].items():
        d[k] = {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}
    d.pop("surface_normal", None)
    out = pa.two_arm_reach(d, lim)
    assert out["status"] in ("unvalidated", "fail")
    assert not out["validated"]


def test_advice_never_turns_past_validated_range():
    from dual_cut_ui import placement_advisor as pa
    from dual_cut_ui.placement import PlacementLimits
    import json, pathlib
    lim = PlacementLimits()
    doc = json.loads((pathlib.Path(__file__).parent / "data" / "template_20261022_082449.json").read_text())
    base = pa.pose_from_doc(doc)
    for start in (-20.0, -10.0, 0.0, 5.0):
        pose = pa.turned(base, start)
        for o in pa.advise(pose, lim)["options"]:
            assert start + o["turn_deg"] <= lim.model_valid_turn_deg[1] + 1e-6


# ---- three more carton sizes (mock study box_sizes_20261107, 1208 cells): fixtures = snapshots/box{1,2,3}_base.json, moved like make_placement.py
# box 1 330 x 570 mm, box 2 300 x 450 mm, box 3 280 x 400 mm.  Cell = (box, yaw, dx, dy); class from results.csv (F = both faces >= 90 %, x = refused).
BOXES = {b: json.load(open(os.path.join(HERE, "data", f"box{b}_base.json"))) for b in (1, 2, 3)}
BOX_F = [(1, 0, -120, -40), (1, 0, -80, -160), (2, 10, -80, 0), (2, 0, -120, -80), (2, 20, -40, 40), (3, 10, -80, 80), (3, 20, -40, 0), (3, 10, -40, -80)]
# refused cells the 2026-11-07 four-constraint model passed (cell -> constraint that now catches it)
BOX_X_NEW = {(1, -10, -160, -80): "close", (3, 10, -240, -160): "close", (1, 0, -160, -320): "right_start",
             (1, 10, 0, -240): "right_far", (1, 20, 0, -160): "right_far"}
BOX_X_OLD = [(1, -20, -40, 0), (2, -10, 40, 0), (3, -20, 0, -160)]            # refused and already caught by the left start corner


def box_at(cell):
    b, yaw, dx, dy = cell
    return place(BOXES[b], dx, dy, yaw)


def box_run(cell, **lim):
    return P.assess_placement(box_at(cell), P.PlacementLimits.from_dict(lim) if lim else None)["two_arm_reach"]


@pytest.mark.parametrize("cell", BOX_F)
def test_box_sizes_known_F_cells_pass_and_are_validated(cell):
    t = box_run(cell)
    assert t["status"] in ("ok", "marginal") and t["ok"] and t["validated"], t["margins_mm"]
    assert set(t["margins_mm"]) == {"left_wrist", "right_wrist", "left_tip", "right_tip", "right_far", "right_start", "close"}


@pytest.mark.parametrize("cell,binding", list(BOX_X_NEW.items()))
def test_box_sizes_new_constraints_catch_refused_cells(cell, binding):
    t = box_run(cell)
    assert t["status"] == "fail" and t["binding"] == binding and t["margin_mm"] < -30
    old_like = box_run(cell, right_wrist_min_m=0.19, right_far_max_m=9.0, right_start_max_m=9.0, carton_min_x_m=-9.0)
    assert old_like["ok"], "the four old constraints alone pass this refused cell (that is why the new ones exist)"
    assert P.assess_placement(box_at(cell))["warnings"][0]["id"] == "two_arm_reach"


@pytest.mark.parametrize("cell", BOX_X_OLD)
def test_box_sizes_refused_cells_still_caught_by_the_left_start_corner(cell):
    t = box_run(cell)
    assert t["status"] == "fail" and t["binding"] == "left_wrist"


def test_new_constraint_wording_names_the_arm_and_the_limit():
    txt = box_run((1, 10, 0, -240))["text"]
    assert "right arm wrist at the back-right far end" in txt and "515 mm" in txt
    assert "too close to the arms" in box_run((1, -10, -160, -80))["text"]
    assert "front-right start corner" in box_run((1, 0, -160, -320))["text"]


@pytest.mark.parametrize("cell", [(1, 10, 0, -240), (1, -10, -160, -80), (1, 0, -160, -320), (3, 10, -240, -160)])
def test_box_sizes_advice_applied_passes_the_model(cell):
    t = box_run(cell)
    assert t["advice"]["options"], "an option exists"
    b, yaw, dx, dy = cell
    for o in t["advice"]["options"][:2]:
        after = P.assess_placement(place(BOXES[b], dx + o["dx_mm"], dy + o["dy_mm"], yaw + o["turn_deg"]))["two_arm_reach"]
        assert after["status"] == "ok" and after["margin_mm"] >= 20 - 0.5
        assert yaw + o["turn_deg"] <= 30.5 + 1e-6


def test_the_validity_window_reaches_plus_30_but_not_beyond():
    assert P.DEFAULT_LIMITS.model_valid_turn_deg[1] >= 30.0
    t = box_run((2, 20, -40, 40))
    assert t["validated"] and t["status"] != "unvalidated"
    t30 = box_run((3, 30, -120, 80), carton_min_x_m=-9.0)               # +30 grid yaw is +29.4..+30.02 deg measured: still inside
    assert t30["validated"]
    t40 = P.assess_placement(place(TEMPLATE, -80, 0, 40))["two_arm_reach"]
    assert not t40["validated"] and t40["status"] in ("unvalidated", "fail")


def test_new_limits_are_overridable_and_have_documented_defaults():
    lim = P.DEFAULT_LIMITS
    assert (lim.right_far_max_m, lim.right_start_max_m, lim.carton_min_x_m) == (0.515, 0.35, 0.115)
    assert (lim.left_wrist_max_m, lim.right_wrist_min_m, lim.left_tip_min_m, lim.right_tip_min_m) == (0.52, 0.185, 0.09, 0.105)
    loose = box_run((1, 10, 0, -240), right_far_max_m=9.0)
    assert loose["margins_mm"]["right_far"] > 1000
    tight = box_run((1, 0, -120, -40), carton_min_x_m=0.30)
    assert tight["binding"] == "close" and tight["status"] == "fail"
