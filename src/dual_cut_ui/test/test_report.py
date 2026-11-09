"""plan_report / exec_report parsing against REAL coordinator output and defensive shapes."""
import json
import math

import pytest

from dual_cut_ui.report import clean, loads_lenient, parse_exec_report, parse_plan_report
from helpers import read_data


def test_sequence_mode_report():
    r = parse_plan_report(read_data("sequence_plan_report.json"))
    assert r["parsed"] and r["ok"] is True and r["schedule_mode"] == "sequence"
    assert r["makespan_s"] == pytest.approx(73.0978257) and r["makespan_kind"] == "sequence"
    assert r["min_cross_m"] == pytest.approx(0.0500002032)
    assert r["sequence"]["delayed_arm"] == "left" and r["sequence"]["delay_s"] == 9
    assert r["sequence"]["duration_left_s"] == pytest.approx(64.0978257)
    assert r["phases"] == []
    assert r["top_split_mode"] == "left_max" and r["top_split"]["left_reach_fraction"] == 0.85
    cov = r["top_coverage"]
    assert cov["seam_length_mm"] == pytest.approx(393.50477) and cov["right_mm"][0] == pytest.approx(322.479055)
    assert cov["left_mm"][0] == pytest.approx(0.0, abs=1e-6) and cov["right_fraction"][0] == pytest.approx(0.819504817)
    assert cov["overlap_mm"] == 12 and r["top_coverage_ok"] is True
    assert r["plans"]["left"]["job1"]["pattern"] == "left_only" and r["plans"]["left"]["job2"]["n_segments"] == 8
    assert r["snapshot"]["capture"].startswith("frame_000008") and r["snapshot"]["age_at_plan_end_s"] == pytest.approx(12.18, abs=0.01)
    assert "SEQUENCE schedule" in r["message"]


def test_phases_mode_side_first_report():
    r = parse_plan_report(read_data("side_first_report.json"))
    assert r["schedule_mode"] == "phases" and r["phase_order"] == "side_first" and r["ok"] is True
    assert [p["name"] for p in r["phases"]] == ["A", "B"]
    a, b = r["phases"]
    assert a["mode"] == "parallel" and a["duration_left_s"] == pytest.approx(41.1097719) and a["duration_right_s"] == pytest.approx(36.8323803)
    assert b["mode"] == "serial" and b["serial_first"] == "left" and b["min_cross_effective_m"] == pytest.approx(0.0502529562)
    # A is parallel -> max(41.1, 36.8); B is serial -> 23.0 + 27.3 (the old "max" rule gave 68.4 s)
    assert a["duration_s"] == pytest.approx(41.1097719) and b["duration_s"] == pytest.approx(22.9880539 + 27.2869577)
    assert r["makespan_s"] == pytest.approx(41.1097719 + 22.9880539 + 27.2869577) and r["makespan_kind"] == "est."
    assert r["min_cross_m"] == pytest.approx(0.0500000656)
    assert r["sequence"] is None and r["executor_wall_s"] == pytest.approx(5.53143659)


def test_phases_mode_top_first_report_without_phase_order_key():
    r = parse_plan_report(read_data("top_first_report.json"))
    assert r["schedule_mode"] == "phases" and r["phase_order"] is None and r["ok"] is True
    assert len(r["phases"]) == 2 and r["makespan_kind"] == "est." and r["makespan_s"] > 0
    assert "serial" in r["phases"][0]["message"] or r["phases"][0]["mode"] == "serial"


def test_exec_report_real():
    r = parse_exec_report(read_data("sequence_exec_report.json"))
    assert r["completed"] is True and r["schedule"] == "sequence" and r["mode"] == "sequence"
    assert r["wall_s"] == pytest.approx(74.092) and r["min_cross_distance_during_execution_m"] == pytest.approx(0.157181)
    assert r["max_tracking_error_rad"] == pytest.approx(0.0131035) and r["stop_latched"] is False and r["stop_reason"] == ""


def test_exec_report_stopped():
    r = parse_exec_report(json.dumps({"phase": "A", "completed": False, "mode": "parallel", "wall_s": 3.1, "stop_latched": True,
                                      "stop_reason": "software stop", "max_tracking_error_rad": None}))
    assert r["completed"] is False and r["stop_latched"] is True and r["stop_reason"] == "software stop"
    assert r["max_tracking_error_rad"] is None and r["planned_duration_left_s"] is None


def test_refusal_report_with_missing_keys():
    r = parse_plan_report('{"executor_calls":{"wall_s":0.5},"ok":false,"message":"top split source snapshot: file not found"}')
    assert r["parsed"] and r["ok"] is False and r["message"].startswith("top split source")
    assert r["makespan_s"] is None and r["min_cross_m"] is None and r["top_coverage"] is None and r["phases"] == []
    assert r["snapshot"] is None and r["top_split"] is None and r["schedule_mode"] is None


def test_sequence_fallback_to_phases_uses_the_phase_estimate():
    doc = json.loads(read_data("side_first_report.json"))
    doc["sequence"] = {"fallback_to_phases": True, "chosen": {"makespan_s": 1.0}}
    doc["schedule_mode"] = "sequence"
    r = parse_plan_report(json.dumps(doc))
    assert r["makespan_kind"] == "est." and r["makespan_s"] == pytest.approx(41.1097719 + 22.9880539 + 27.2869577)
    assert r["sequence"]["fallback_to_phases"] is True


@pytest.mark.parametrize("text", ["", "not json", "[1,2]", "null", "{"])
def test_garbage_is_not_parsed(text):
    r = parse_plan_report(text)
    assert r["parsed"] is False and r["ok"] is None
    assert parse_exec_report(text)["parsed"] is False


def test_weird_types_do_not_crash():
    r = parse_plan_report(json.dumps({"ok": "yes", "phases": {"A": "x", "B": {"duration_left_s": "5", "duration_right_s": True}},
                                      "top_coverage": {"left_mm": [1], "right_mm": "x", "seam_length_mm": None},
                                      "top_split": "left_max", "snapshot": [], "plans": {"left": {"A": 3}}, "sequence": {"chosen": 5}}))
    assert r["parsed"] and r["ok"] is None and r["makespan_s"] is None
    assert r["top_coverage"]["left_mm"] is None and r["top_coverage"]["right_mm"] is None


def test_nan_and_inf_from_ostream_are_survived():
    text = '{"ok":true,"message":"m","phases":{"A":{"ok":true,"duration_left_s":nan,"duration_right_s":12.5,"min_cross_effective_m":-nan}},"x":inf}'
    r = parse_plan_report(text)
    assert r["parsed"] and r["phases"][0]["duration_left_s"] is None and r["makespan_s"] == 12.5
    assert loads_lenient("{\"a\": NaN}") == {"a": float("nan")} or math.isnan(loads_lenient("{\"a\": NaN}")["a"])


def test_clean_replaces_non_finite_floats():
    out = clean({"a": float("nan"), "b": [1.0, float("inf"), {"c": -float("inf")}], "d": (float("nan"),), "e": "nan"})
    assert out == {"a": None, "b": [1.0, None, {"c": None}], "d": [None], "e": "nan"}
    json.dumps(out, allow_nan=False)


def test_top_first_real_report_phase_durations_follow_the_mode():
    r = parse_plan_report(read_data("top_first_report.json"))
    doc = json.loads(read_data("top_first_report.json"))
    expected = 0.0
    for name, p in sorted(doc["phases"].items()):
        pair = (p["duration_left_s"], p["duration_right_s"])
        expected += sum(pair) if p["mode"] == "serial" else max(pair)
    assert r["makespan_s"] == pytest.approx(expected)
    assert {p["mode"] for p in r["phases"]} == {"serial", "parallel"}


@pytest.mark.parametrize("mode,first,left,right,expect,how", [
    ("parallel", "", 41.1, 37.1, 41.1, "parallel"),
    ("serial", "left", 23.0, 27.2, 50.2, "serial"),
    ("serial: left arm first", "left", 23.0, 27.2, 50.2, "serial"),
    ("Serial", None, 10.0, 5.0, 15.0, "serial"),
    ("mystery", "", 10.0, 5.0, 15.0, "assumed serial (unknown mode)"),
    (None, "", 10.0, 5.0, 15.0, "assumed serial (unknown mode)"),
    ("", "right", 10.0, 5.0, 15.0, "serial"),
    ("parallel", "", 10.0, None, 10.0, "parallel"),
    ("serial", "left", None, 7.0, 7.0, "serial"),
])
def test_phase_duration_rule(mode, first, left, right, expect, how):
    from dual_cut_ui.report import phase_duration
    d, h = phase_duration(mode, first, left, right)
    assert d == pytest.approx(expect) and h == how


def test_phase_without_durations_is_unknown():
    from dual_cut_ui.report import phase_duration
    assert phase_duration("parallel", "", None, None) == (None, "unknown")


def test_lead_case_91_seconds():
    """A parallel 41.1/37.1 + B serial 23.0/27.2 = 91.3 s (the UI showed 68.3 s)."""
    doc = {"ok": True, "message": "m", "phases": {
        "A": {"ok": True, "mode": "parallel", "duration_left_s": 41.1, "duration_right_s": 37.1},
        "B": {"ok": True, "mode": "serial", "serial_first": "left", "duration_left_s": 23.0, "duration_right_s": 27.2}}}
    r = parse_plan_report(json.dumps(doc))
    assert r["makespan_s"] == pytest.approx(91.3) and r["makespan_kind"] == "est."
