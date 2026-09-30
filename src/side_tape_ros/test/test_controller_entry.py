"""controller_node ENTRY services, in-process (no spinning, isolated domain): advisor defect 1 (overlapping
start requests), the entry-wide frozen reference, /side_tape/stop, ACM restore on an ownership-loss terminal
edge. Needs a sourced ROS environment (skipped otherwise). Never touches domain 11: the domain is forced."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture(scope="module")
def rcl():
    rclpy = pytest.importorskip("rclpy")
    pytest.importorskip("side_tape_interfaces")
    import os
    os.environ["ROS_DOMAIN_ID"] = "94"
    os.environ["ROS_LOCALHOST_ONLY"] = "1"
    rclpy.init(args=None)
    yield rclpy
    rclpy.shutdown()


class _Lock:
    """A lock that never blocks: 'no start serialization', to show the phase re-check alone protects the frame."""
    def acquire(self, blocking=True):
        return True

    def release(self):
        pass


@pytest.fixture
def node(rcl):
    from side_tape_ros.controller_node import SideTapeController
    n = SideTapeController()  # dry_run default True: no Servo calls, no twist to Servo
    pos = {"p": np.array([0.3, -0.3, 0.1])}

    def fake_tf():
        stamp = n.get_clock().now().to_msg()
        return SimpleNamespace(
            header=SimpleNamespace(stamp=stamp),
            transform=SimpleNamespace(
                translation=SimpleNamespace(x=float(pos["p"][0]), y=float(pos["p"][1]), z=float(pos["p"][2])),
                rotation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)))

    n._lookup_tf_now = fake_tf
    n.pos = pos
    # the supervisor's config may be uncalibrated in tests; the entry start does not need vision
    yield n
    n.destroy_node()


def _req_insert(depth=7.0, speed=2.0, limit=0.0):
    from side_tape_interfaces.srv import StartInsert
    r = StartInsert.Request()
    r.depth_mm, r.speed_mm_s, r.depth_limit_mm = depth, speed, limit
    return r


def _resp_insert():
    from side_tape_interfaces.srv import StartInsert
    return StartInsert.Response()


def _slow_prepare(n, seconds=0.4):
    def prep():
        time.sleep(seconds)
        return True, ""
    n._prepare_live_servo = prep


def _two_starts(n, delay=0.1):
    out = {}

    def call(tag, wait):
        time.sleep(wait)
        out[tag] = n._on_start_insert(_req_insert(), _resp_insert())

    a = threading.Thread(target=call, args=("a", 0.0))
    b = threading.Thread(target=call, args=("b", delay))
    a.start(); b.start(); a.join(); b.join()
    return out["a"], out["b"]


def test_defect1_second_start_during_servo_preparation_is_refused_busy(node):
    from precision_box_detector.side_tape.types import Phase
    _slow_prepare(node)
    a, b = _two_starts(node)
    assert a.accepted and a.message == "insert_started" and a.move_id == 1
    assert not b.accepted and "busy" in b.message
    assert node.supervisor.phase == Phase.INSERT and node.supervisor.move_id == 1


def test_defect1_frozen_reference_is_never_replaced_while_a_move_is_active(node):
    """Without the transaction lock the second request used to overwrite _vecs after the first had started
    (advisor: 9 mm physical travel vs 7 mm accounted). The phase re-check under the lock alone must hold."""
    from precision_box_detector.side_tape.types import Phase
    node._start_txn = _Lock()  # disable serialization on purpose
    _slow_prepare(node)
    vecs_after_first = {}

    orig = node.supervisor.start_insert

    def spy(*args, **kw):
        orig(*args, **kw)
        if node.supervisor.phase == Phase.INSERT and "v" not in vecs_after_first:
            vecs_after_first["v"] = node._vecs  # the frame the accepted move is measured in

    node.supervisor.start_insert = spy
    a, b = _two_starts(node, delay=0.1)
    assert a.accepted and not b.accepted and "phase_changed" in b.message
    assert node._vecs is vecs_after_first["v"]  # unchanged
    # a later request while the move is active: wrong phase, reference untouched
    node.pos["p"] = node.pos["p"] + np.array([0.001, 0.0, 0.0])
    c = node._on_start_insert(_req_insert(depth=2.0), _resp_insert())
    assert not c.accepted and "wrong_phase" in c.message
    assert node._vecs is vecs_after_first["v"]


def test_entry_reference_is_frozen_once_not_between_moves(node):
    from precision_box_detector.side_tape.types import Phase
    a = node._on_start_insert(_req_insert(depth=3.0), _resp_insert())
    assert a.accepted
    v0 = node._vecs
    node.supervisor.phase = Phase.INSERTED  # as after a finished move (resting)
    node.pos["p"] = node.pos["p"] + np.array([0.0005, 0.0, 0.0])  # it drifted while resting
    b = node._on_start_insert(_req_insert(depth=2.0), _resp_insert())
    assert b.accepted and b.move_id == 2
    assert node._vecs is v0  # ONE entry-wide reference: depth_dev is the net depth from the entry start


def test_concurrent_other_start_services_are_serialized_too(node):
    from side_tape_interfaces.srv import StartCut
    node._start_txn.acquire()  # a start transaction is in flight
    try:
        r = node._on_start_cut(StartCut.Request(), StartCut.Response())
        assert not r.accepted and "busy" in r.message
        from std_srvs.srv import Trigger
        t = node._on_start_align(Trigger.Request(), Trigger.Response())
        assert not t.success and "busy" in t.message
    finally:
        node._start_txn.release()


def test_stop_service_is_motionless_and_reports_the_latch(node):
    from std_srvs.srv import Trigger
    from precision_box_detector.side_tape.types import Phase
    assert node._on_start_insert(_req_insert(), _resp_insert()).accepted
    r = node._on_stop(Trigger.Request(), Trigger.Response())
    assert r.success and node.supervisor.phase == Phase.ABORT and node.supervisor._abort_stop
    assert "phase=ABORT" in r.message


def test_ownership_loss_terminal_edge_restores_the_acm(node):
    from precision_box_detector.side_tape.types import ControlOutput, Phase
    calls = []
    node._restore_carton_acm_async = lambda why: calls.append(why)
    node._prev_terminal = False  # was INSERTED (not terminal)
    node._publish_twists(0.0, np.zeros(3), ControlOutput.zero(Phase.IDLE, "handed_back"), arm_owned=False)
    assert calls and node._prev_terminal is True
    node._publish_twists(0.0, np.zeros(3), ControlOutput.zero(Phase.IDLE, "handed_back"), arm_owned=False)
    assert len(calls) == 1  # once, on the edge


def test_status_carries_the_move_id(node):
    assert node._on_start_insert(_req_insert(), _resp_insert()).accepted
    node._control_step(node._now_s())
    assert node._last_move_id == 1
    node._on_status_timer()  # builds the message with move_id without raising
