"""Auto capture state machine with a fake clock, joints and capture function (no ROS, no web server)."""
import math

import pytest

from rebel_dual_arm_calibration.ui import autocapture as A

PAGE, OTHER = "page-a", "page-b"
OK_MSG = "cap 3: left: 20 corners (4 markers, 4 chess), PnP 0.30 px, sample #3"


class World:
    """Fake clock + joints + capture; ``tick(dt)`` advances the clock in 0.5 s ticks like the periodic callback."""

    def __init__(self, **kw):
        self.t = 100.0
        self.q = {f"j{i}": 0.0 for i in range(1, 7)}
        self.spread = 0.0
        self.covered = True
        self.fresh = True
        self.busy = False
        self.recorder = True
        self.replies = []                       # scripted capture replies; the last one repeats
        self.calls = []
        self.auto = A.AutoCapture(lambda: self.t, self._joints, self._capture, lambda: self.busy,
                                  lambda: self.recorder, **kw)

    def _joints(self, window_s):
        assert window_s == 2.0
        return A.JointReading(dict(self.q), self.spread, self.covered) if self.fresh else None

    def _capture(self):
        self.calls.append(self.t)
        r = self.replies[min(len(self.calls) - 1, len(self.replies) - 1)] if self.replies else (True, OK_MSG)
        return r

    def run(self, seconds, step=0.5, hb=True):
        for _ in range(int(round(seconds / step))):
            self.t += step
            if hb:
                self.auto.heartbeat(PAGE)
            self.auto.tick()

    def move_to(self, deg, joint="j1"):
        self.q[joint] = math.radians(deg)

    def snap(self):
        return self.auto.snapshot(PAGE)


def started(**kw):
    w = World(**kw)
    ok, msg = w.auto.start(PAGE, 2.0)
    assert ok, msg
    return w


def test_new_pose_is_captured_once_and_same_pose_is_not_captured_again():
    w = started()
    w.run(1.0)
    assert len(w.calls) == 1
    s = w.snap()
    assert s["on"] and s["owner_is_you"] and s["state"] == "captured" and s["message"] == "captured (1 this run)"
    assert s["captured"] == 1 and s["attempts"] == 1
    w.run(5.0)                                            # still standing there: no second capture
    assert len(w.calls) == 1 and w.snap()["message"] == "captured (1 this run)"
    # move away (moving), come back to the first pose: handled already
    w.spread = 0.05
    w.run(1.0)
    assert w.snap()["state"] == "moving"
    w.spread = 0.0
    w.run(1.0)
    assert len(w.calls) == 1 and w.snap()["state"] == "waiting" and w.snap()["message"] == "waiting for a new pose"
    # a genuinely new pose
    w.move_to(10.0)
    w.run(1.0)
    assert len(w.calls) == 2 and w.snap()["message"] == "captured (2 this run)"
    ev = w.snap()["events"]
    assert [e["kind"] for e in ev] == ["captured", "captured"] and ev[0]["level"] == "ok" and ev[0]["message"] == OK_MSG


def test_min_change_is_respected_against_every_handled_pose():
    w = started()
    w.run(1.0)
    assert len(w.calls) == 1
    w.move_to(1.5)                                        # < 2 deg from the first pose
    w.run(2.0)
    assert len(w.calls) == 1
    w.move_to(2.5)                                        # >= 2 deg
    w.run(1.0)
    assert len(w.calls) == 2
    w.move_to(1.0)                                        # within 2 deg of the first pose (and 1.5 deg from the second)
    w.run(2.0)
    assert len(w.calls) == 2
    w.move_to(5.0)                                        # 2.5 deg from the second pose, 5 from the first: new
    w.run(1.0)
    assert len(w.calls) == 3
    w.move_to(5.0, "j4")                                  # a different joint counts as well
    w.run(1.0)
    assert len(w.calls) == 4


def test_min_change_setting_1_and_5_degrees():
    w = World()
    assert w.auto.start(PAGE, 1.0)[0]
    w.run(1.0)
    w.move_to(1.2)
    w.run(1.0)
    assert len(w.calls) == 2                              # 1.2 deg >= 1 deg
    w.auto.stop(PAGE)
    assert w.auto.start(PAGE, 5.0)[0]
    w.move_to(30.0)
    w.run(1.0)
    n = len(w.calls)
    w.move_to(33.0)                                       # only 3 deg from the previous
    w.run(2.0)
    assert len(w.calls) == n
    w.move_to(36.0)
    w.run(1.0)
    assert len(w.calls) == n + 1
    assert not w.auto.start(PAGE, 3.0)[0] and "one of" in w.auto.start(PAGE, 3.0)[1]


def test_same_pose_reply_marks_the_pose_handled():
    w = started()
    w.replies = [(False, "nothing recorded: left: same pose as an existing sample, skipped (20 corners)")]
    w.run(5.0)
    s = w.snap()
    assert len(w.calls) == 1 and s["state"] == "skipped" and s["message"] == "same pose, skipped" and s["skipped"] == 1
    assert s["events"][-1]["level"] == "warn" and "same pose" in s["events"][-1]["message"]


def test_board_still_moving_is_retried_every_half_second_until_it_works():
    w = started()
    still = (False, "left: board still moving (common 0.31 px > 0.20 px, single max 0.62 px over 20 corners), rejected")
    w.replies = [still, still, still, (True, OK_MSG)]
    w.run(0.5)
    assert len(w.calls) == 1 and w.snap()["state"] == "retrying" and w.snap()["message"].startswith("retrying (1): ")
    w.run(2.0)
    assert len(w.calls) == 4
    assert [round(b - a, 1) for a, b in zip(w.calls, w.calls[1:])] == [0.5, 0.5, 0.5]
    s = w.snap()
    assert s["state"] == "captured" and s["captured"] == 1 and s["attempts"] == 4
    assert [e["kind"] for e in s["events"]] == ["captured"]            # retries are not logged individually


@pytest.mark.parametrize("reply", [
    (False, "arms not stationary yet (still for 1.20 s, need 2.00 s)"),
    (False, "no earlier image >= 0.4 s older than the newest one after the stationary start"),
    (False, "newest image predates the stationary window; retry in a moment"),
    (False, "newest image is 1.20 s old (max 1.0 s)"),
    (False, "nothing recorded: left: cannot verify stillness: not detected in the earlier frame, rejected"),
    (True, "NO ARM SAMPLE RECORDED (table-board image only) - cap 4: left: board still moving (common 0.5 px > 0.2 px)"),
])
def test_timing_refusals_are_retried(reply):
    assert A.classify_reply(*reply)[0] == "timing"


def test_gives_up_after_15_s_and_does_not_retry_until_a_new_pose():
    w = started()
    w.replies = [(False, "left: board still moving (common 0.31 px > 0.20 px, single max 0.62 px over 20 corners), rejected")]
    w.run(14.0)
    assert w.snap()["state"] == "retrying"
    w.run(2.0)
    s = w.snap()
    assert s["state"] == "gave_up" and s["message"].startswith("gave up: ") and "board still moving" in s["message"]
    assert s["gave_up"] == 1 and s["events"][-1]["level"] == "warn"
    n = len(w.calls)
    assert w.calls[-1] - w.calls[0] >= 15.0
    w.run(10.0)
    assert len(w.calls) == n                                       # no more tries at this pose
    w.move_to(8.0)
    w.run(1.0)
    assert len(w.calls) > n                                        # a new pose is tried again


def test_pose_problem_is_retried_once_then_reported():
    w = started()
    w.replies = [(False, "nothing recorded: left: not detected")]
    w.run(0.5)
    assert len(w.calls) == 1 and w.snap()["state"] == "retrying"
    w.run(0.5)
    assert len(w.calls) == 2
    s = w.snap()
    assert s["state"] == "unusable" and s["message"].startswith("board not usable here: ") and s["gave_up"] == 1
    w.run(10.0)
    assert len(w.calls) == 2
    # the second try may succeed
    w2 = started()
    w2.replies = [(True, "NO ARM SAMPLE RECORDED (table-board image only) - cap 1: left: only 5 corners (< 8)"),
                  (True, OK_MSG)]
    w2.run(1.5)
    assert w2.snap()["state"] == "captured" and len(w2.calls) == 2
    for msg in ("left: PnP reprojection 1.90 px > 0.7, rejected", "left: only 5 corners (< 8)",
                "NO ARM SAMPLE RECORDED (table-board image only; ...)"):
        assert A.classify_reply(False if not msg.startswith("NO") else True, msg)[0] == "pose"


def test_other_failures_retry_every_2_s_and_never_mark_the_pose_handled():
    w = started()
    w.replies = [(False, "recorder service not available")] * 4 + [(True, OK_MSG)]
    w.run(1.0)
    assert len(w.calls) == 1 and w.snap()["state"] == "retrying"
    w.run(20.0)
    assert [round(b - a, 1) for a, b in zip(w.calls, w.calls[1:3])] == [2.0, 2.0]
    s = w.snap()
    assert s["state"] == "captured" and len(w.calls) == 5            # no give-up, however long it takes
    assert [e["kind"] for e in s["events"]] == ["other", "captured"]  # the persistent failure is logged once


def test_manual_capture_in_progress_postpones_without_counting():
    w = started()
    w.replies = [None, None, (True, OK_MSG)]
    w.run(1.5)
    s = w.snap()
    assert len(w.calls) == 3 and s["attempts"] == 1 and s["captured"] == 1


def test_pauses_while_a_jog_step_runs_and_restarts_the_stationary_window():
    w = started()
    w.run(1.0)
    assert len(w.calls) == 1
    w.busy = True
    w.move_to(20.0)
    w.run(3.0)
    assert w.snap()["state"] == "paused" and len(w.calls) == 1
    w.busy = False
    w.run(0.5)
    assert w.snap()["state"] == "settling"                         # joints look still, but the window restarts
    w.run(1.0)
    assert len(w.calls) == 1
    w.run(1.5)
    assert len(w.calls) == 2 and w.snap()["state"] == "captured"
    # a fault / stopping jog counts as busy as well (the predicate is injected), and so does missing joint states
    w.fresh = False
    w.run(1.0)
    assert w.snap()["state"] == "no_joints"


def test_not_stationary_by_the_encoders_means_no_attempt():
    w = started()
    w.spread = 0.003
    w.run(3.0)
    assert not w.calls and w.snap()["state"] == "moving"
    w.spread = 0.0019
    w.run(1.0)
    assert len(w.calls) == 1
    w2 = started()
    w2.covered = False                                              # not enough history yet
    w2.run(2.0)
    assert not w2.calls and w2.snap()["state"] == "settling"


def test_owner_heartbeat_loss_stops_the_run():
    w = started()
    w.run(4.0)
    assert w.snap()["on"]
    w.run(6.0, hb=False)
    s = w.snap()
    assert not s["on"] and not s["owner_is_you"] and s["state"] == "off" and "stopped polling" in s["message"]
    n = len(w.calls)
    w.move_to(20.0)
    w.run(2.0)
    assert len(w.calls) == n                                        # stopped: nothing more is captured
    assert w.auto.start(OTHER, 2.0)[0]                              # and the next page can start its own run


def test_second_page_is_refused_and_cannot_stop_it():
    w = started()
    ok, msg = w.auto.start(OTHER, 2.0)
    assert not ok and msg == "auto capture is running on another page"
    ok, msg = w.auto.stop(OTHER)
    assert not ok and w.snap()["on"]
    assert not w.auto.snapshot(OTHER)["owner_is_you"] and w.auto.snapshot(PAGE)["owner_is_you"]
    w.auto.heartbeat(OTHER)                                         # only the owner's polling keeps it alive
    w.run(7.0, hb=False)
    assert not w.snap()["on"]
    ok, msg = w.auto.stop(OTHER)
    assert ok and "not running" in msg


def test_owner_stop_and_restart_resets_the_handled_poses():
    w = started()
    w.run(1.0)
    assert w.auto.stop(PAGE)[0]
    assert not w.snap()["on"]
    w.run(2.0)
    assert len(w.calls) == 1
    assert w.auto.start(PAGE, 2.0)[0]
    w.run(1.0)
    assert len(w.calls) == 2                                        # a new run handles the same pose again (recorder skips it)
    assert w.auto.start(PAGE, 5.0) == (True, "auto capture is already running") and w.snap()["min_change_deg"] == 5.0


def test_recorder_disappearing_stops_the_run_and_start_needs_the_recorder():
    w = started()
    w.run(1.0)
    w.recorder = False
    w.run(0.5)
    s = w.snap()
    assert not s["on"] and "recorder not available" in s["message"]
    assert w.auto.start(PAGE, 2.0) == (False, "recorder not available")


def test_start_refused_without_a_joint_source():
    w = World(unavailable="joint states are not available (jog module not loaded)")
    w.auto._joints = None
    assert w.auto.start(PAGE, 2.0) == (False, "joint states are not available (jog module not loaded)")


def test_stop_during_a_capture_does_not_mark_anything():
    w = started()

    def capture():
        w.auto.stop(PAGE)                                           # the operator stops while the call is in flight
        return True, OK_MSG
    w.auto._capture = capture
    w.run(1.0)
    s = w.snap()
    assert not s["on"] and s["captured"] == 0 and s["state"] == "off"
    assert [e["kind"] for e in s["events"]] == ["captured"]         # but the stored sample is still logged
