"""Pure tests for side_tape_ros.entry (arg rules, offset trim, record) and the sequencer's --entry parsing."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

from side_tape_ros import entry
from side_tape_ros.handoff import load_handoff

REC = {
    "schema": "box_h_cut_side_tape_handoff/v1", "status": "active", "face": "left", "variant": "standoff",
    "cut_length_mm": 54.9, "penetration_mm": 3.7, "standoff_mm": 9.0, "planning_frame": "base_link",
    "tool_link": "xeg32_tool_tip",
    "start_pose": {"position_m": {"x": 0.27, "y": -0.32, "z": 0.08},
                   "orientation_xyzw": {"x": 0.58, "y": 0.46, "z": -0.48, "w": 0.47}},
    "side_tape_convention": {"cut_axis_tool": {"x": 0, "y": -1, "z": 0}, "lateral_axis_tool": {"x": 1, "y": 0, "z": 0},
                             "depth_axis_tool": {"x": 0, "y": 0, "z": 1}},
    "left_parallel_orientation": {"max_abs_angle_deg": 5.0, "rotation_from_f0_deg": 7.0},
    "snapshot_id": "snap_1", "stamp_ns": 1790669152388051072,
}


def _rec(tmp_path, **over):
    d = json.loads(json.dumps(REC))
    d.update(over)
    p = tmp_path / "h.json"
    p.write_text(json.dumps(d))
    return p, load_handoff(str(p))


def _args(rec=None, **kw):
    base = dict(entry=True, handoff="h.json", handoff_rec=rec, measure_only=False, align_only=False,
                line_hold=False, hold_start_offset=False, ignore_handoff_pose=False, line_vision_gain=None,
                seed_json=None, standoff_mm=None, cut_length_mm=None, mode=None,
                penetration_mm=None, entry_tol_mm=None, entry_offset_mm=None, insert_speed_mm_s=None,
                max_retries=None, entry_record_path=None, pre_contact_mm=None, contact_margin_mm=None,
                backoff_mm=None, calibrate_rim_contact=False)
    base.update(kw)
    return SimpleNamespace(**base)


# -- trim ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("entry_off, cfg_off, trim", [(0.0, 0.0, 0.0), (0.0, 1.5, -1.5), (0.5, 1.5, -1.0),
                                                       (-1.0, -3.0, 2.0), (2.0, 0.0, 2.0)])
def test_trim_makes_the_target_the_rim_plus_entry_offset(entry_off, cfg_off, trim):
    t = entry.trim_for_entry(entry_off, cfg_off)
    assert t == pytest.approx(trim)
    # perception: effective = offset + parallax * standoff + trim; at standoff 0 it is the entry offset,
    # and at the approach standoff only the parallax term is added on top of it
    assert entry.effective_offset_mm(cfg_off, 0.4, 0.0, t) == pytest.approx(entry_off)
    assert entry.effective_offset_mm(cfg_off, 0.4, 9.0, t) == pytest.approx(entry_off + 0.4 * 9.0)


# -- arg rules ------------------------------------------------------------------------------------

def test_entry_args_valid_and_defaults(tmp_path):
    _, rec = _rec(tmp_path)
    a = _args(rec)
    assert entry.check_entry_args(a) is None
    entry.apply_entry_defaults(a)
    assert (a.penetration_mm, a.entry_tol_mm, a.entry_offset_mm, a.insert_speed_mm_s, a.max_retries) == (
        4.8, 0.5, 0.0, 2.0, 2)  # 4.8 mm = the operator-confirmed cut depth past contact
    assert (a.pre_contact_mm, a.contact_margin_mm, a.backoff_mm) == (2.0, 0.0, 1.0)
    a2 = _args(rec, penetration_mm=4.0, max_retries=0)
    entry.apply_entry_defaults(a2)
    assert a2.penetration_mm == 4.0 and a2.max_retries == 0 and a2.entry_tol_mm == 0.5


def test_entry_flags_need_entry():
    for name, val in (("penetration_mm", 3.0), ("entry_tol_mm", 0.5), ("entry_offset_mm", 0.0),
                      ("insert_speed_mm_s", 2.0), ("max_retries", 2), ("entry_record_path", "x.json"),
                      ("pre_contact_mm", 2.0), ("contact_margin_mm", 0.0), ("backoff_mm", 1.0)):
        err = entry.check_entry_args(_args(entry=False, **{name: val}))
        assert err and "needs --entry" in err, name
    assert entry.check_entry_args(_args(entry=False)) is None


def test_entry_needs_handoff_standoff_variant_and_no_conflicts(tmp_path):
    _, rec = _rec(tmp_path)
    assert "needs --handoff" in entry.check_entry_args(_args(None, handoff=None))
    _, im = _rec(tmp_path, variant="in_material", standoff_mm=0.0)
    assert "variant 'standoff'" in entry.check_entry_args(_args(im))
    for kw, frag in ((dict(measure_only=True), "--measure-only"), (dict(align_only=True), "--align-only"),
                     (dict(line_hold=True), "--line-hold"), (dict(hold_start_offset=True), "--hold-start-offset"),
                     (dict(ignore_handoff_pose=True), "--ignore-handoff-pose"),
                     (dict(line_vision_gain=0.2), "--line-vision-gain"), (dict(seed_json="s.json"), "--seed-json"),
                     (dict(standoff_mm=5.0), "--standoff-mm"), (dict(cut_length_mm=40.0), "--cut-length-mm"),
                     (dict(mode="in_material"), "--mode"), (dict(calibrate_rim_contact=True), "--calibrate-rim-contact")):
        err = entry.check_entry_args(_args(rec, **kw))
        assert err and frag in err, kw
    assert entry.check_entry_args(_args(rec, mode="air_align")) is None


@pytest.mark.parametrize("kw", [
    dict(penetration_mm=0.0), dict(penetration_mm=-1.0), dict(penetration_mm=10.5), dict(penetration_mm=float("nan")),
    dict(entry_tol_mm=0.0), dict(entry_tol_mm=2.5), dict(entry_offset_mm=6.0), dict(entry_offset_mm=-5.1),
    dict(entry_offset_mm=float("inf")), dict(insert_speed_mm_s=0.0), dict(insert_speed_mm_s=5.1),
    dict(insert_speed_mm_s=-2.0), dict(max_retries=-1), dict(max_retries=6), dict(max_retries=1.5),
    dict(max_retries=True), dict(pre_contact_mm=0.4), dict(pre_contact_mm=10.5), dict(contact_margin_mm=3.1),
    dict(contact_margin_mm=-2.1), dict(backoff_mm=-0.1), dict(backoff_mm=3.1), dict(pre_contact_mm=float("nan"))])
def test_entry_numeric_ranges_are_enforced(tmp_path, kw):
    _, rec = _rec(tmp_path)
    assert entry.check_entry_args(_args(rec, **kw)) is not None


def test_entry_numeric_limits_are_inclusive_where_documented(tmp_path):
    _, rec = _rec(tmp_path)
    for kw in (dict(penetration_mm=10.0), dict(entry_tol_mm=2.0), dict(entry_offset_mm=-5.0),
               dict(entry_offset_mm=5.0), dict(insert_speed_mm_s=5.0), dict(max_retries=5), dict(max_retries=0),
               dict(pre_contact_mm=0.5), dict(pre_contact_mm=10.0), dict(contact_margin_mm=-2.0),
               dict(contact_margin_mm=3.0), dict(backoff_mm=0.0), dict(backoff_mm=3.0)):
        assert entry.check_entry_args(_args(rec, **kw)) is None, kw


# -- record ---------------------------------------------------------------------------------------

def test_record_content_and_schema(tmp_path):
    _, rec = _rec(tmp_path)
    r = entry.build_record(status="ok", tool_pose=((0.1, 0.2, 0.3), (0.0, 0.0, 0.0, 1.0)), final_e_lat_mm=0.12,
                           entry_offset_mm=0.0, penetration_mm=3.0, standoff_mm=9.0, tol_mm=0.5,
                           insert_speed_mm_s=2.0, iterations=2, max_retries=2, e_lat_history_mm=[0.9, 0.12],
                           spread_mm=0.05, handoff_rec=rec, time_iso="2026-09-30T12:00:00.000Z", penetrated=True,
                           pre_contact_mm=2.0, contact_margin_mm=0.0, backoff_mm=1.0)
    assert r["schema"] == "side_tape_entry/v1" == entry.SCHEMA
    assert r["status"] == "ok" and r["final_e_lat_mm"] == 0.12 and r["iterations"] == 2 and r["retries_used"] == 1
    assert r["handoff_stamp_ns"] == 1790669152388051072 and r["handoff_snapshot_id"] == "snap_1"
    assert r["tool_pose_base"] == {"frame": "base_link", "position_m": {"x": 0.1, "y": 0.2, "z": 0.3},
                                   "orientation_xyzw": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}
    assert r["entry_offset_mm"] == 0.0 and r["penetration_mm"] == 3.0 and r["time_iso"].endswith("Z")
    assert r["e_lat_history_mm"] == [0.9, 0.12]
    assert r["measured_at"] == "contact" and r["penetrated"] is True
    assert (r["pre_contact_mm"], r["contact_margin_mm"], r["backoff_mm"]) == (2.0, 0.0, 1.0)
    assert any("PENETRATED" in ln for ln in entry.summary_lines(r))
    out_of_tol = entry.build_record(status="out_of_tolerance", entry_offset_mm=0.0, penetration_mm=4.8)
    assert out_of_tol["penetrated"] is False
    assert any("NOT penetrated" in ln for ln in entry.summary_lines(out_of_tol))
    with pytest.raises(ValueError):
        entry.build_record(status="done", entry_offset_mm=0, penetration_mm=3)


def test_record_failed_without_pose_and_iso_time():
    r = entry.build_record(status="failed", reason="handoff_check_failed", entry_offset_mm=0.0, penetration_mm=3.0)
    assert r["status"] == "failed" and r["reason"] == "handoff_check_failed" and r["tool_pose_base"] is None
    assert r["final_e_lat_mm"] is None and r["handoff_stamp_ns"] is None
    from datetime import datetime
    datetime.fromisoformat(r["time_iso"].replace("Z", "+00:00"))
    assert any("failed" in ln for ln in entry.summary_lines(r))


def test_write_record_atomic_to_tmp_path(tmp_path):
    path = tmp_path / "sub" / "side_tape_entry.json"
    first = entry.build_record(status="failed", reason="x", entry_offset_mm=0.0, penetration_mm=3.0)
    assert entry.write_record_atomic(str(path), first) == str(path)
    assert json.loads(path.read_text())["status"] == "failed"
    second = entry.build_record(status="ok", entry_offset_mm=0.0, penetration_mm=3.0, final_e_lat_mm=0.1)
    entry.write_record_atomic(str(path), second)
    assert json.loads(path.read_text())["status"] == "ok"
    assert sorted(os.listdir(path.parent)) == ["side_tape_entry.json"]  # no temp files left behind


def test_write_record_atomic_failure_keeps_the_old_file_and_leaves_no_temp(tmp_path):
    path = tmp_path / "e.json"
    entry.write_record_atomic(str(path), entry.build_record(status="ok", entry_offset_mm=0.0, penetration_mm=3.0))
    bad = entry.build_record(status="ok", entry_offset_mm=0.0, penetration_mm=3.0)
    bad["final_e_lat_mm"] = float("nan")  # allow_nan=False -> serialisation fails mid-write
    with pytest.raises(ValueError):
        entry.write_record_atomic(str(path), bad)
    assert json.loads(path.read_text())["status"] == "ok"
    assert sorted(os.listdir(tmp_path)) == ["e.json"]


def test_write_record_expands_user(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    p = entry.write_record_atomic("~/.ros/side_tape_entry.json",
                                  entry.build_record(status="ok", entry_offset_mm=0.0, penetration_mm=3.0))
    assert p == str(tmp_path / ".ros" / "side_tape_entry.json") and os.path.exists(p)


# -- sequencer argument handling (needs a sourced ROS environment; skipped otherwise) ----------------

def _sequencer(monkeypatch):
    pytest.importorskip("rclpy")
    pytest.importorskip("side_tape_interfaces")
    from side_tape_ros import sequencer

    class Stop(Exception):
        pass

    def _init(*a, **k):
        raise Stop()

    monkeypatch.setattr(sequencer.rclpy, "init", _init)
    return sequencer, Stop


@pytest.mark.parametrize("argv_extra, msg", [
    (["--entry"], "--handoff"),
    (["--entry", "--handoff", "H", "--measure-only"], "--measure-only"),
    (["--entry", "--handoff", "H", "--cut-length-mm", "30"], "--cut-length-mm"),
    (["--entry", "--handoff", "H", "--penetration-mm", "0"], "--penetration-mm"),
    (["--entry", "--handoff", "H", "--pre-contact-mm", "20"], "--pre-contact-mm"),
    (["--entry", "--handoff", "H", "--pre-contact-mm", "8.5"], "standoff_mm"),
    (["--entry", "--handoff", "H", "--calibrate-rim-contact"], "--calibrate-rim-contact"),
    (["--calibrate-rim-contact", "--handoff", "H"], "--handoff"),
    (["--calibrate-rim-contact", "--cut-length-mm", "30"], "--cut-length-mm"),
    (["--calibrate-rim-contact", "--measure-only"], "--measure-only"),
    (["--calibrate-rim-contact", "--penetration-mm", "3"], "needs --entry"),
    (["--entry", "--handoff", "H", "--max-retries", "9"], "--max-retries"),
    (["--entry", "--handoff", "H", "--ignore-handoff-pose"], "--ignore-handoff-pose"),
    (["--penetration-mm", "3", "--cut-length-mm", "30"], "needs --entry"),
    (["--backoff-mm", "1"], "needs --entry"),
])
def test_sequencer_rejects_bad_entry_invocations(monkeypatch, tmp_path, capsys, argv_extra, msg):
    seq, Stop = _sequencer(monkeypatch)
    p, _ = _rec(tmp_path)
    with pytest.raises(SystemExit) as ei:
        seq.main([str(p) if a == "H" else a for a in argv_extra])
    assert ei.value.code == 2
    assert msg in capsys.readouterr().err


def test_sequencer_rejects_in_material_record_and_bad_standoff(monkeypatch, tmp_path, capsys):
    seq, Stop = _sequencer(monkeypatch)
    p, _ = _rec(tmp_path, variant="in_material", standoff_mm=0.0)
    with pytest.raises(SystemExit):
        seq.main(["--entry", "--handoff", str(p)])
    assert "variant 'standoff'" in capsys.readouterr().err
    p, _ = _rec(tmp_path, standoff_mm=1.0)  # too close for a back-out
    with pytest.raises(SystemExit):
        seq.main(["--entry", "--handoff", str(p)])
    assert "standoff_mm" in capsys.readouterr().err


def test_sequencer_entry_parse_takes_standoff_from_the_record(monkeypatch, tmp_path, capsys):
    seq, Stop = _sequencer(monkeypatch)
    p, _ = _rec(tmp_path)
    with pytest.raises(Stop):
        seq.main(["--entry", "--handoff", str(p), "--penetration-mm", "4.0", "--entry-record-path",
                  str(tmp_path / "e.json")])
    out = capsys.readouterr().out
    assert "ENTRY mode: standoff_mm=9.000" in out and "--penetration-mm 4.00" in out
    assert "contact is NOT sensed" in out


def test_sequencer_entry_default_penetration_is_4p8(monkeypatch, tmp_path, capsys):
    seq, Stop = _sequencer(monkeypatch)
    p, _ = _rec(tmp_path)
    with pytest.raises(Stop):
        seq.main(["--entry", "--handoff", str(p)])
    assert "--penetration-mm 4.80" in capsys.readouterr().out


def test_sequencer_calibrate_parse_needs_no_handoff_or_cut_length(monkeypatch, capsys):
    seq, Stop = _sequencer(monkeypatch)
    with pytest.raises(Stop):
        seq.main(["--calibrate-rim-contact", "--auto"])
    assert "--calibrate-rim-contact: no service calls, no motion" in capsys.readouterr().out


# -- --calibrate-rim-contact (pure part) ----------------------------------------------------------------------

def test_rim_contact_offset_matches_the_robot_measurement():
    """Operator datum 2026-09-30: tip on the rim at contact, offset -0.71, camera e_lat -0.18 -> offset -0.89."""
    new, med, spread = entry.rim_contact_offset(-0.71, [-0.18] * 15, 15)
    assert new == pytest.approx(-0.89) and med == pytest.approx(-0.18) and spread == pytest.approx(0.0)


def test_rim_contact_offset_makes_e_lat_zero_sign_proof():
    """e_lat = tool - target and target = rim + offset: a new offset = old + e_lat gives e_lat' = e_lat - (new - old) = 0."""
    old, e = 1.25, 0.37
    new, med, _ = entry.rim_contact_offset(old, [e + d for d in (-0.02, 0.0, 0.02, 0.01, -0.01)], 5)
    assert new == pytest.approx(old + e)
    assert (e - (new - old)) == pytest.approx(0.0, abs=1e-9)


def test_rim_contact_offset_uses_the_median_and_refuses_bad_data():
    new, med, spread = entry.rim_contact_offset(0.0, [0.10, 0.11, 0.09, 0.10, 0.90, 0.10, 0.11, 0.09, 0.10, 0.10,
                                                     0.10, 0.09, 0.11, 0.10, 0.10], 15)
    assert med == pytest.approx(0.10) and new == pytest.approx(0.10)  # one outlier does not move the median
    with pytest.raises(ValueError, match="too few"):
        entry.rim_contact_offset(0.0, [0.1] * 5, 15)
    with pytest.raises(ValueError, match="spread"):
        entry.rim_contact_offset(0.0, [0.0, 1.0] * 8, 15)  # unstable measurement (p90-p10 = 1 > 0.3)
    with pytest.raises(ValueError):
        entry.rim_contact_offset(0.0, [], 1)


def test_calibrate_args_rules():
    ok = SimpleNamespace(calibrate_rim_contact=True, entry=False, measure_only=False, align_only=False,
                         line_hold=False, hold_start_offset=False, handoff=None, seed_json=None,
                         cut_length_mm=None, standoff_mm=None, line_vision_gain=None, mode=None)
    assert entry.check_calibrate_args(ok) is None
    assert entry.check_calibrate_args(SimpleNamespace(calibrate_rim_contact=False)) is None
    for kw in (dict(entry=True), dict(measure_only=True), dict(align_only=True), dict(line_hold=True),
               dict(hold_start_offset=True), dict(handoff="h.json"), dict(seed_json="s.json"),
               dict(cut_length_mm=30.0), dict(standoff_mm=9.0), dict(line_vision_gain=0.1), dict(mode="air_align")):
        bad = SimpleNamespace(**{**vars(ok), **kw})
        assert entry.check_calibrate_args(bad), kw
