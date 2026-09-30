"""sequencer --entry safety (advisor defects 3, 4, 6 and the lesser items). Needs a sourced ROS environment
(skipped otherwise); nothing is spun, no node is created: the node, the clock and time are fakes."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest

from side_tape_ros import entry

from test_entry import _rec  # noqa: E402  (same test directory)


@pytest.fixture
def seq(monkeypatch):
    pytest.importorskip("rclpy")
    pytest.importorskip("side_tape_interfaces")
    from side_tape_ros import sequencer
    return sequencer


class FakeNode:
    """Virtual clock: every rclpy.spin_once advances 50 ms and runs ``hook(node)``."""

    def __init__(self, hook=None):
        self.t = 1000.0
        self.hook = hook
        self.latest_obs = None
        self.latest_status = None
        self.cli_stop = "stop-client"
        self.stop_calls = 0
        self.stop_reply = (True, "phase=ABORT")
        self.on_stop = None

    def get_clock(self):
        return SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=int(self.t * 1e9)))

    def call_trigger(self, client, name, timeout_sec=5.0, aborted=None):
        self.stop_calls += 1
        if self.on_stop:
            self.on_stop(self)
        return self.stop_reply

    def step(self):
        self.t += 0.05
        if self.hook:
            self.hook(self)


def _install(monkeypatch, seq, node):
    monkeypatch.setattr(seq.rclpy, "ok", lambda: True)
    monkeypatch.setattr(seq.rclpy, "spin_once", lambda n, timeout_sec=0.0: n.step())
    monkeypatch.setattr(seq, "time", SimpleNamespace(monotonic=lambda: node.t, sleep=lambda s: None))


def _stamp(t):
    return SimpleNamespace(sec=int(t), nanosec=int((t - int(t)) * 1e9))


def _obs(node, seq_no, age=0.05, state="LOCKED", e_lat=0.1, conf=0.8, yaw=0.5, has_e_lat=True, has_yaw=True):
    return SimpleNamespace(frame_seq=seq_no, header=SimpleNamespace(stamp=_stamp(node.t - age)), state=state,
                           has_e_lat=has_e_lat, e_lat_mm=e_lat, conf=conf, has_e_yaw=has_yaw, e_yaw_deg=yaw)


def _status(node, phase, move_id=0, v=(0.0, 0.0, 0.0), age=0.0, reasons=()):
    return SimpleNamespace(stamp=_stamp(node.t - age), phase=phase, move_id=move_id, v_cut_mps=v[0], v_lat_mps=v[1],
                           v_depth_mps=v[2], reasons=list(reasons), depth_dev_mm=0.0, lat_dev_mm=0.0, along_mm=0.0)


CFG = None


def _cfg():
    global CFG
    if CFG is None:
        from precision_box_detector.side_tape.config import load_config
        from side_tape_ros import core_path
        CFG = load_config(core_path.DEFAULT_CONFIG)
    return CFG


# -- defect 4: every sample of the contact decision is gated -----------------------------------------------------

GOOD = dict(stamp_s=10.0, now_s=10.05, state="LOCKED", has_e_lat=True, e_lat=0.1, conf=0.8, has_e_yaw=True,
            e_yaw=1.0, stale_s=0.15, min_conf=0.15, max_yaw_deg=12.0)


def test_defect4_sample_ok_gate_table():
    assert entry.sample_ok(**GOOD) is None
    bad = {
        "not_locked": dict(state="COASTING"), "no_e_lat": dict(has_e_lat=False),
        "no_e_lat ": dict(e_lat=math.nan), "stale": dict(now_s=10.5), "future_stamp": dict(stamp_s=10.5),
        "low_conf": dict(conf=0.0), "low_conf ": dict(conf=math.nan), "bad_yaw": dict(e_yaw=90.0),
        "bad_yaw ": dict(e_yaw=-12.5), "bad_yaw  ": dict(has_e_yaw=False), "bad_yaw   ": dict(e_yaw=math.nan),
        "bad_stamp": dict(stamp_s=math.nan),
    }
    for want, kw in bad.items():
        assert entry.sample_ok(**{**GOOD, **kw}) == want.strip(), kw
    assert entry.sample_ok(**{**GOOD, "stamp_s": 10.08}) is None  # small clock skew (< 0.05 s) tolerated
    assert entry.sample_ok(**{**GOOD, "e_yaw": 12.0}) is None and entry.sample_ok(**{**GOOD, "conf": 0.15}) is None


def _feed(kind, seq_start=1):
    """Hook publishing one new observation per spin of the given (bad) kind."""
    counter = {"n": seq_start}

    def hook(node):
        counter["n"] += 1
        node.latest_obs = kind(node, counter["n"])
    return hook


BAD_KINDS = {
    "zero_conf": lambda n, i: _obs(n, i, conf=0.0),
    "old": lambda n, i: _obs(n, i, age=1.0),
    "yaw90": lambda n, i: _obs(n, i, yaw=90.0),
    "future": lambda n, i: _obs(n, i, age=-1.0),
    "coasting": lambda n, i: _obs(n, i, state="COASTING"),
    "nan": lambda n, i: _obs(n, i, e_lat=math.nan),
}


@pytest.mark.parametrize("name", list(BAD_KINDS))
def test_defect4_old_collector_accepted_bad_frames_new_gate_rejects_them(monkeypatch, seq, name):
    kind = BAD_KINDS[name]
    node = FakeNode(_feed(kind))
    _install(monkeypatch, seq, node)
    aborted = {"flag": False}
    if name in ("zero_conf", "old", "yaw90", "future"):  # the reproduction: the old collector took 15 of them
        assert len(seq._collect_e_lats(node, 15, 5.0, aborted)) == 15
    node = FakeNode(_feed(kind))
    _install(monkeypatch, seq, node)
    good, rejected = seq._collect_gated_e_lats(node, _cfg(), 15, 3.0, aborted)
    assert good == [] and sum(rejected.values()) > 20


def test_defect4_entry_measure_fails_when_there_are_not_enough_good_samples(monkeypatch, seq):
    node = FakeNode(_feed(BAD_KINDS["yaw90"]))
    node.pub_standoff = SimpleNamespace(publish=lambda m: None)
    _install(monkeypatch, seq, node)
    args = SimpleNamespace(cfg=_cfg(), measure_frames=15, measure_timeout=2.0)
    med, spread, n, msg = seq._entry_measure(node, args, {"flag": False})
    assert med is None and "too few" in msg and "bad_yaw" in msg


def test_defect4_mixed_stream_uses_only_the_good_samples(monkeypatch, seq):
    def kind(n, i):
        return _obs(n, i, e_lat=0.25) if i % 3 == 0 else _obs(n, i, e_lat=9.0, conf=0.0)  # bad ones would skew

    node = FakeNode(_feed(kind))
    _install(monkeypatch, seq, node)
    good, rejected = seq._collect_gated_e_lats(node, _cfg(), 15, 60.0, {"flag": False})
    assert len(good) == 15 and all(g == 0.25 for g in good) and rejected["low_conf"] > 15


def test_defect4_calibrate_uses_the_gate_too(monkeypatch, seq, capsys):
    node = FakeNode(_feed(BAD_KINDS["zero_conf"]))
    _install(monkeypatch, seq, node)
    args = SimpleNamespace(cfg=_cfg(), measure_frames=15, measure_timeout=2.0, auto=True)
    assert seq._run_calibrate_rim_contact(node, args, {"flag": False}) == 1
    assert "rejected by the gate" in capsys.readouterr().out


# -- defect 3: standoff bound, tighter hand-off tolerance ----------------------------------------------------------

def test_defect3_max_standoff_option(monkeypatch, tmp_path, capsys, seq):
    class Stop(Exception):
        pass

    def _init(*a, **k):
        raise Stop()

    monkeypatch.setattr(seq.rclpy, "init", _init)
    p16, _ = _rec(tmp_path, standoff_mm=16.0)
    with pytest.raises(SystemExit):
        seq.main(["--entry", "--handoff", str(p16)])
    assert "exceeds --max-standoff-mm 15" in capsys.readouterr().err
    with pytest.raises(Stop):  # raised deliberately: accepted
        seq.main(["--entry", "--handoff", str(p16), "--max-standoff-mm", "20"])
    with pytest.raises(SystemExit):
        seq.main(["--entry", "--handoff", str(p16), "--max-standoff-mm", "30"])
    p9, _ = _rec(tmp_path, standoff_mm=9.0)
    with pytest.raises(Stop):
        seq.main(["--entry", "--handoff", str(p9)])


def test_defect3_entry_handoff_pose_tolerance_is_5mm_5deg(monkeypatch, tmp_path, seq):
    _, rec = _rec(tmp_path)
    args = SimpleNamespace(entry=True, ignore_handoff_pose=False, handoff_rec=rec, cfg=_cfg())
    x, y, z = rec.position_m

    def at(dx_mm):
        monkeypatch.setattr(seq, "_lookup_tool_pose", lambda node, cfg, timeout_sec=3.0:
                            ((x + dx_mm / 1000.0, y, z), rec.orientation_xyzw))
    at(7.0)  # 7 mm: fine for the cut flows (15 mm), too far for the entry
    assert seq._handoff_check(None, args) is False
    args.entry = False
    assert seq._handoff_check(None, args) is True
    args.entry = True
    at(4.0)
    assert seq._handoff_check(None, args) is True
    assert (entry.ENTRY_POS_TOL_MM, entry.ENTRY_ORI_TOL_DEG) == (5.0, 5.0)


def test_defect3_check_entry_args_standoff_rules(tmp_path):
    from test_entry import _args
    _, rec = _rec(tmp_path, standoff_mm=16.0)
    assert "max-standoff-mm" in entry.check_entry_args(_args(rec))
    assert entry.check_entry_args(_args(rec, max_standoff_mm=16.0)) is None
    _, rec9 = _rec(tmp_path, standoff_mm=9.0)
    for bad in (0.0, -1.0, 26.0, math.nan):
        assert entry.check_entry_args(_args(rec9, max_standoff_mm=bad)) is not None
    assert entry.check_entry_args(_args(None, entry=False, max_standoff_mm=15.0)) == "--max-standoff-mm needs --entry"


# -- defect 6: stop confirmed, move id correlation, failures stop the arm -------------------------------------------

def test_defect6_wait_move_done_ignores_statuses_of_other_moves(monkeypatch, seq):
    node = FakeNode()
    _install(monkeypatch, seq, node)
    states = iter([("INSERTED", 1)] * 6 + [("INSERT", 2)] * 3 + [("INSERTED", 2)] * 3)  # stale INSERTED of move 1

    def hook(n):
        ph, mid = next(states, ("INSERTED", 2))
        n.latest_status = _status(n, ph, mid)
    node.hook = hook
    ok, msg, st = seq._wait_move_done(node, {"flag": False}, "approach", 30.0, move_id=2)
    assert ok and st.move_id == 2
    assert node.t - 1000.0 > 0.45  # >= 10 spins: not fooled by the first six stale samples


def test_defect6_wait_move_done_failure_and_timeout(monkeypatch, seq):
    node = FakeNode(lambda n: setattr(n, "latest_status", _status(n, "ABORT", 3, reasons=["stall"])))
    _install(monkeypatch, seq, node)
    ok, msg, st = seq._wait_move_done(node, {"flag": False}, "x", 30.0, move_id=3)
    assert not ok and "ABORT" in msg
    node = FakeNode(lambda n: setattr(n, "latest_status", _status(n, "INSERT", 3, v=(0, 0, 0.002))))
    _install(monkeypatch, seq, node)
    ok, msg, st = seq._wait_move_done(node, {"flag": False}, "x", 2.0, move_id=3)
    assert not ok and "did not finish" in msg  # the caller must now stop the arm
    node = FakeNode(lambda n: setattr(n, "latest_status", _status(n, "INSERTED", 1)))  # only other moves' statuses
    _install(monkeypatch, seq, node)
    assert not seq._wait_move_done(node, {"flag": False}, "x", 2.0, move_id=2)[0]


def test_defect6_stop_and_confirm_waits_for_a_post_stop_zero_twist_status(monkeypatch, seq):
    node = FakeNode()
    state = {"stopped_at": None}

    def on_stop(n):
        state["stopped_at"] = n.t

    def hook(n):  # the supervisor keeps driving for 0.4 s after the request, and status samples lag
        if state["stopped_at"] is None or n.t < state["stopped_at"] + 0.4:
            n.latest_status = _status(n, "INSERT", 4, v=(0, 0, 0.002))
        else:
            n.latest_status = _status(n, "ABORT", 4, reasons=["operator_stop", "operator_required"])
    node.hook, node.on_stop = hook, on_stop
    node.latest_status = _status(node, "INSERT", 4, v=(0, 0, 0.002))
    _install(monkeypatch, seq, node)
    ok, msg = seq._stop_and_confirm(node)
    assert ok and node.stop_calls == 1 and "ABORT" in msg
    assert node.t >= state["stopped_at"] + 0.4 + 0.3  # not before the twist was zero for 0.3 s


def test_defect6_stop_and_confirm_never_assumes(monkeypatch, seq):
    node = FakeNode(lambda n: setattr(n, "latest_status", _status(n, "INSERT", 4, v=(0, 0, 0.002))))
    node.stop_reply = (False, "stop_no_response")
    _install(monkeypatch, seq, node)
    ok, msg = seq._stop_and_confirm(node, timeout_s=2.0)
    assert not ok and node.stop_calls == 3 and "NOT confirmed" in msg  # retried the lost reply, still no proof
    node = FakeNode(lambda n: setattr(n, "latest_status", _status(n, "ABORT", 4, age=5.0)))  # stale statuses only
    _install(monkeypatch, seq, node)
    assert not seq._stop_and_confirm(node, timeout_s=2.0)[0]


def test_is_stopped_predicate():
    assert entry.is_stopped("ABORT", 0.0, 0.0, 0.0) and entry.is_stopped("INSERTED", 0, 0, 0) and entry.is_stopped("IDLE", 0, 0, 0)
    assert not entry.is_stopped("INSERT", 0, 0, 0) and not entry.is_stopped("ABORT", 0, 0, -0.003)
    assert not entry.is_stopped("HOLD", 0, 0, math.nan)


# -- _run_entry with the whole world faked ---------------------------------------------------------------------------

def _entry_args(tmp_path, rec, **kw):
    a = dict(handoff_rec=rec, cfg=_cfg(), standoff_mm=9.0, penetration_mm=4.8, entry_tol_mm=0.5, max_retries=1,
             pre_contact_mm=2.0, contact_margin_mm=0.0, backoff_mm=1.0, max_standoff_mm=15.0, insert_speed_mm_s=2.0,
             entry_offset_mm=0.0, entry_record_path=str(tmp_path / "entry.json"), measure_frames=15,
             measure_timeout=1.0)
    a.update(kw)
    return SimpleNamespace(**a)


def _patch_entry(monkeypatch, seq, node, *, move_reply=(True, "insert_started", 1), wait_reply=(False, "timeout", None),
                 stop_reply=(True, "stopped"), measure=None):
    calls = {"stop": 0, "moves": 0}
    _install(monkeypatch, seq, node)
    node.latest_status = _status(node, "INSERTED", 1)  # resting between moves
    monkeypatch.setattr(seq, "_tf_quiet", lambda n, cfg, **k: True)
    monkeypatch.setattr(seq, "_lookup_pose_or_none", lambda n, cfg: ((0.1, 0.2, 0.3), (0, 0, 0, 1)))
    monkeypatch.setattr(seq, "StartInsert", SimpleNamespace(Request=lambda: SimpleNamespace()))
    monkeypatch.setattr(seq, "StartLateral", SimpleNamespace(Request=lambda: SimpleNamespace()))
    node.cli_insert = node.cli_lateral = object()

    def call_move(n, client, req, name, aborted, timeout_sec=5.0):
        calls["moves"] += 1
        return move_reply

    def stop(n, timeout_s=8.0, quiet_s=0.3):
        calls["stop"] += 1
        return stop_reply

    monkeypatch.setattr(seq, "_call_move_service", call_move)
    monkeypatch.setattr(seq, "_wait_move_done", lambda *a, **k: wait_reply)
    monkeypatch.setattr(seq, "_stop_and_confirm", stop)
    if measure is not None:
        monkeypatch.setattr(seq, "_entry_measure", measure)
    return calls


def test_defect6_timeout_of_a_move_stops_the_arm_and_records_failed(monkeypatch, seq, tmp_path):
    """Reproduced by the advisor: a failed result while the supervisor stayed in INSERT commanding +depth."""
    _, rec = _rec(tmp_path)
    node = FakeNode()
    calls = _patch_entry(monkeypatch, seq, node)
    rc = seq._run_entry(node, _entry_args(tmp_path, rec), {"flag": False})
    assert rc == 1 and calls["stop"] == 1
    r = json.loads((tmp_path / "entry.json").read_text())
    assert r["status"] == "failed" and "approach_failed" in r["reason"] and "MOTION_NOT" not in r["reason"]


def test_defect6_lost_reply_and_refusal_and_exception_stop_the_arm(monkeypatch, seq, tmp_path):
    _, rec = _rec(tmp_path)
    for reply in ((False, "start_insert_no_response", 0), (False, "start_insert_refused:busy", 0)):
        node = FakeNode()
        calls = _patch_entry(monkeypatch, seq, node, move_reply=reply)
        assert seq._run_entry(node, _entry_args(tmp_path, rec), {"flag": False}) == 1 and calls["stop"] == 1
    node = FakeNode()
    calls = _patch_entry(monkeypatch, seq, node)

    def boom(*a, **k):
        raise RuntimeError("kaboom")
    monkeypatch.setattr(seq, "_call_move_service", boom)
    assert seq._run_entry(node, _entry_args(tmp_path, rec), {"flag": False}) == 1 and calls["stop"] == 1
    assert "exception:RuntimeError:kaboom" in json.loads((tmp_path / "entry.json").read_text())["reason"]


def test_defect6_measurement_failure_and_ctrl_c_stop_the_arm(monkeypatch, seq, tmp_path):
    _, rec = _rec(tmp_path)
    node = FakeNode()
    calls = _patch_entry(monkeypatch, seq, node, move_reply=(True, "ok", 1), wait_reply=(True, "ok", None),
                         measure=lambda n, a, ab: (None, None, 0, "too few"))
    assert seq._run_entry(node, _entry_args(tmp_path, rec), {"flag": False}) == 1 and calls["stop"] == 1
    node = FakeNode()
    calls = _patch_entry(monkeypatch, seq, node, move_reply=(True, "ok", 1), wait_reply=(False, "aborted", None))
    aborted = {"flag": True}  # Ctrl-C: the entry path must STOP (motionless), not call /side_tape/abort (retract)
    assert seq._run_entry(node, _entry_args(tmp_path, rec), aborted) == 1 and calls["stop"] == 1
    assert json.loads((tmp_path / "entry.json").read_text())["reason"] == "aborted"


def test_defect6_unconfirmed_stop_is_loud_in_the_record_and_exit_code(monkeypatch, seq, tmp_path, capsys):
    _, rec = _rec(tmp_path)
    node = FakeNode()
    _patch_entry(monkeypatch, seq, node, stop_reply=(False, "NOT confirmed stopped"))
    assert seq._run_entry(node, _entry_args(tmp_path, rec), {"flag": False}) == 1
    assert "MOTION_NOT_CONFIRMED_STOPPED" in json.loads((tmp_path / "entry.json").read_text())["reason"]
    assert "COULD NOT CONFIRM" in capsys.readouterr().out


def test_out_of_tolerance_needs_no_stop_it_is_resting(monkeypatch, seq, tmp_path):
    _, rec = _rec(tmp_path)
    node = FakeNode()
    calls = _patch_entry(monkeypatch, seq, node, move_reply=(True, "ok", 1), wait_reply=(True, "ok", None),
                         measure=lambda n, a, ab: (1.9, 0.01, 15, "ok"))
    rc = seq._run_entry(node, _entry_args(tmp_path, rec, max_retries=0), {"flag": False})
    assert rc == 2 and calls["stop"] == 0
    assert json.loads((tmp_path / "entry.json").read_text())["penetrated"] is False


# -- lesser: a failed record write is a non-zero exit --------------------------------------------------------------------

def test_failed_record_write_makes_the_exit_code_nonzero_even_for_ok(monkeypatch, seq, tmp_path, capsys):
    _, rec = _rec(tmp_path)
    node = FakeNode()
    _patch_entry(monkeypatch, seq, node, move_reply=(True, "ok", 1), wait_reply=(True, "ok", None),
                 measure=lambda n, a, ab: (0.1, 0.01, 15, "ok"))
    good_args = _entry_args(tmp_path, rec)
    assert seq._run_entry(node, good_args, {"flag": False}) == 0  # control: all fine, record written
    blocker = tmp_path / "file_not_dir"
    blocker.write_text("x")
    bad_args = _entry_args(tmp_path, rec, entry_record_path=str(blocker / "sub" / "entry.json"))
    assert seq._run_entry(node, bad_args, {"flag": False}) == 1  # would have been 0
    assert "could not write the entry record" in capsys.readouterr().out
    oot = _entry_args(tmp_path, rec, entry_record_path=str(blocker / "sub" / "entry.json"), max_retries=0)
    monkeypatch.setattr(seq, "_entry_measure", lambda n, a, ab: (1.9, 0.01, 15, "ok"))
    assert seq._run_entry(node, oot, {"flag": False}) == 1  # would have been 2
