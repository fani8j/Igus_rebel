"""sequencer: controller parameters are set AND read back; the sequencer refuses to start on a mismatch,
and states cut_mode=vision explicitly when --line-hold is not given (review findings 4 and 5).
Needs a sourced ROS environment (skipped otherwise); no node is created and nothing is spun."""

from __future__ import annotations

import pytest


@pytest.fixture
def seq(monkeypatch):
    pytest.importorskip("rclpy")
    pytest.importorskip("side_tape_interfaces")
    from side_tape_ros import sequencer
    return sequencer


class _Pub:
    def publish(self, msg):
        pass


class _FakeNode:
    def __init__(self):
        self.pub_standoff = _Pub()
        self.pub_trim = _Pub()
        self.latest_obs = None
        self.latest_status = None

    def destroy_node(self):
        pass


def _patch_main(monkeypatch, seq, set_result, readback):
    """main() with rclpy / the node faked; records the order of controller calls."""
    calls = []
    monkeypatch.setattr(seq.rclpy, "init", lambda *a, **k: None)
    monkeypatch.setattr(seq.rclpy, "shutdown", lambda *a, **k: None)
    monkeypatch.setattr(seq.rclpy, "spin_once", lambda *a, **k: None)
    monkeypatch.setattr(seq, "_Sequencer", _FakeNode)
    monkeypatch.setattr(seq.signal, "signal", lambda *a, **k: None)
    monkeypatch.setattr(seq.time, "sleep", lambda *a, **k: None)

    def fake_set(node, params, timeout_sec=3.0):
        calls.append(("set", dict(params)))
        return set_result(params)

    def fake_get(node, names, timeout_sec=3.0):
        calls.append(("get", list(names)))
        return readback(list(names))

    monkeypatch.setattr(seq, "_set_controller_params", fake_set)
    monkeypatch.setattr(seq, "_get_controller_params", fake_get)
    monkeypatch.setattr(seq, "_run_in_material", lambda *a, **k: calls.append(("run", None)) or 0)
    monkeypatch.setattr(seq, "_run_air_align", lambda *a, **k: calls.append(("run", None)) or 0)
    monkeypatch.setattr(seq, "_publish_trim", lambda *a, **k: None)
    return calls


def test_verify_helper_cases(monkeypatch, seq):
    node = object()
    monkeypatch.setattr(seq, "_set_controller_params", lambda n, p, timeout_sec=3.0: (True, "ok"))
    monkeypatch.setattr(seq, "_get_controller_params", lambda n, names, timeout_sec=3.0: ({"cut_mode": "vision"}, "ok"))
    assert seq._set_and_verify_controller_params(node, {"cut_mode": "vision"})[0] is True
    monkeypatch.setattr(seq, "_get_controller_params",
                        lambda n, names, timeout_sec=3.0: ({"cut_mode": "line_hold"}, "ok"))
    ok, msg = seq._set_and_verify_controller_params(node, {"cut_mode": "vision"})
    assert not ok and "read-back mismatch" in msg
    monkeypatch.setattr(seq, "_get_controller_params", lambda n, names, timeout_sec=3.0: ({}, "get_parameters_timeout"))
    ok, msg = seq._set_and_verify_controller_params(node, {"cut_mode": "vision"})
    assert not ok and "read-back failed" in msg
    monkeypatch.setattr(seq, "_set_controller_params", lambda n, p, timeout_sec=3.0: (False, "set_parameters_timeout"))
    assert seq._set_and_verify_controller_params(node, {"cut_mode": "vision"}) == (False, "set_parameters_timeout")


def test_without_line_hold_vision_is_set_verified_before_the_run(monkeypatch, seq, capsys):
    calls = _patch_main(monkeypatch, seq, lambda p: (True, "ok"), lambda names: ({"cut_mode": "vision"}, "ok"))
    with pytest.raises(SystemExit) as ei:
        seq.main(["--cut-length-mm", "30", "--mode", "in_material", "--standoff-mm", "0"])
    assert ei.value.code == 0
    assert calls == [("set", {"cut_mode": "vision"}), ("get", ["cut_mode"]), ("run", None)]
    assert "cut_mode=vision set and verified" in capsys.readouterr().out


def test_stale_line_hold_left_on_the_controller_is_refused_not_ignored(monkeypatch, seq, capsys):
    """The controller reports success but reads back line_hold (an earlier run's leftover): no start."""
    calls = _patch_main(monkeypatch, seq, lambda p: (True, "ok"), lambda names: ({"cut_mode": "line_hold"}, "ok"))
    with pytest.raises(SystemExit) as ei:
        seq.main(["--cut-length-mm", "30", "--mode", "in_material", "--standoff-mm", "0"])
    assert ei.value.code == 1
    assert ("run", None) not in calls
    assert "nothing was commanded" in capsys.readouterr().out


def test_set_failure_and_unreadable_controller_refuse_to_start(monkeypatch, seq):
    for set_res, rb in (((False, "set_parameters_unavailable"), ({"cut_mode": "vision"}, "ok")),
                        ((True, "ok"), ({}, "get_parameters_unavailable"))):
        calls = _patch_main(monkeypatch, seq, lambda p, r=set_res: r, lambda names, r=rb: r)
        with pytest.raises(SystemExit) as ei:
            seq.main(["--cut-length-mm", "30", "--mode", "in_material", "--standoff-mm", "0"])
        assert ei.value.code == 1 and ("run", None) not in calls


def test_line_hold_readback_mismatch_refuses_and_restores(monkeypatch, seq):
    calls = _patch_main(monkeypatch, seq, lambda p: (True, "ok"), lambda names: ({"cut_mode": "vision",
                                                                                  "line_vision_gain": 0.0}, "ok"))
    with pytest.raises(SystemExit) as ei:
        seq.main(["--cut-length-mm", "30", "--mode", "in_material", "--standoff-mm", "0", "--line-hold"])
    assert ei.value.code == 1
    assert ("run", None) not in calls
    assert calls[0] == ("set", {"cut_mode": "line_hold", "line_vision_gain": 0.0})
    assert ("set", {"cut_mode": "vision", "line_vision_gain": 0.0}) in calls  # restored right away


def test_line_hold_ok_runs_then_restores_verified(monkeypatch, seq):
    state = {"mode": "line_hold"}

    def set_(p):
        state["mode"] = p["cut_mode"]
        return True, "ok"

    calls = _patch_main(monkeypatch, seq, set_, lambda names: ({"cut_mode": state["mode"], "line_vision_gain": 0.0}, "ok"))
    with pytest.raises(SystemExit) as ei:
        seq.main(["--cut-length-mm", "30", "--mode", "in_material", "--standoff-mm", "0", "--line-hold"])
    assert ei.value.code == 0
    kinds = [c[0] for c in calls]
    assert kinds == ["set", "get", "run", "set", "get"]
    assert calls[3] == ("set", {"cut_mode": "vision", "line_vision_gain": 0.0})


def test_measure_only_touches_no_controller_parameters(monkeypatch, seq):
    calls = _patch_main(monkeypatch, seq, lambda p: (True, "ok"), lambda names: ({}, "ok"))
    monkeypatch.setattr(seq, "_run_measure_only", lambda *a, **k: calls.append(("measure", None)) or 0)
    with pytest.raises(SystemExit) as ei:
        seq.main(["--measure-only"])
    assert ei.value.code == 0 and calls == [("measure", None)]
