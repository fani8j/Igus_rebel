import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from rebel_gripper_io.jxc_handshake import JxcConfig, JxcHandshake, SampleTracker  # noqa: E402
from sim_jxc import SimJxc  # noqa: E402

CFG = JxcConfig()


def make(**kw):
    sim = SimJxc(**kw)
    return sim, JxcHandshake(sim, sim.cfg)


def ready(sim, hs):
    assert hs.servo_on().success
    assert hs.home().success


def final_off(sim):
    sim.sleep(0.2)  # let pending output changes land
    assert CFG.out_drive not in sim.dout
    assert CFG.out_setup not in sim.dout


def test_servo_on_home_and_state():
    sim, hs = make()
    r = hs.servo_on()
    assert r.success, r.message
    assert sim.svre and CFG.out_svon in sim.dout
    r = hs.home()
    assert r.success, r.message
    assert sim.seton
    final_off(sim)
    assert hs.last_error == ""


def test_move_step_sequence_and_step_bits_before_drive():
    for step in range(4):
        sim, hs = make()
        ready(sim, hs)
        sim.log.clear()
        r = hs.move_step(step)
        assert r.success, r.message
        assert sim._steps_at_drive == [step]  # IN0/IN1 were applied when DRIVE rose
        # the request order: step bits first, then DRIVE on, DRIVE off last
        flat = [p for _, pairs in sim.log for p in pairs]
        idx_drive_on = flat.index((CFG.out_drive, True))
        assert flat.index((CFG.out_in0, bool(step & 1))) < idx_drive_on
        assert flat.index((CFG.out_in1, bool(step & 2))) < idx_drive_on
        t_bits = sim.log[0][0]
        t_drive = [t for t, pairs in sim.log if (CFG.out_drive, True) in pairs][0]
        assert t_drive - t_bits >= CFG.step_setup_delay_s
        final_off(sim)
        assert sim.drive_trace[-1][1] is False


def test_refuses_step_out_of_range_and_without_home():
    sim, hs = make()
    assert not hs.move_step(4).success
    assert not hs.move_step(0).success            # no servo
    hs.servo_on()
    r = hs.move_step(0)
    assert not r.success and "SETON" in r.message
    assert CFG.out_drive not in sim.dout


def test_home_requires_svre():
    sim, hs = make()
    r = hs.home()
    assert not r.success and "SVRE" in r.message
    assert CFG.out_setup not in [p for _, pairs in sim.log for p, s in pairs if s]


def test_servo_on_timeout_turns_svon_off():
    sim, hs = make(servo_delay=100.0)
    hs.cfg.servo_on_timeout_s = 0.3
    r = hs.servo_on()
    assert not r.success and "timeout" in r.message
    sim.sleep(0.1)
    assert CFG.out_svon not in sim.dout
    assert hs.last_error == r.message


def test_home_timeout_turns_setup_off():
    sim, hs = make(home_time=100.0)
    hs.cfg.home_timeout_s = 0.4
    assert hs.servo_on().success
    r = hs.home()
    assert not r.success and "timeout" in r.message
    final_off(sim)


def test_move_timeout_means_no_part_and_drive_off():
    sim, hs = make()
    ready(sim, hs)
    sim.never_reaches = True
    hs.cfg.move_timeout_s = 0.4
    r = hs.move_step(2)
    assert not r.success and "gripped" in r.message
    final_off(sim)
    assert sim.drive_trace[-1][1] is False


def test_inp_never_clears_is_not_an_error_without_busy():
    # Operator decision 2026-11-13: INP staying ON (already in position) is accepted; the result says
    # the start was not seen, and DRIVE still ends OFF.
    sim, hs = make()
    ready(sim, hs)
    sim.never_clears_inp = True
    r = hs.move_step(1)
    assert r.success and "start of the move not seen" in r.message
    final_off(sim)


def test_inp_never_clears_then_never_on_still_fails():
    sim, hs = make()
    ready(sim, hs)
    sim.never_clears_inp = True
    sim.never_reaches = True
    # INP drops late (near the end of the start window) and never comes back -> still a failure
    sim.events.append((sim.t + 0.6, lambda s: setattr(s, "inp", False)))
    r = hs.move_step(1)
    assert not r.success and "INP not ON" in r.message
    final_off(sim)


def test_alarm_during_move_aborts_and_drive_off():
    sim, hs = make(move_time=2.0)
    ready(sim, hs)
    sim.events.append((sim.t + 0.2, lambda s: setattr(s, "alarm_ok", False)))
    t0 = sim.t
    r = hs.move_step(1)
    assert not r.success and "ALARM" in r.message
    assert sim.t - t0 < 1.0   # aborted early, not after the 2 s move
    final_off(sim)


def test_alarm_before_command_refused():
    sim, hs = make()
    sim.alarm_ok = False
    sim.sleep(0.05)
    r = hs.servo_on()
    assert not r.success and "ALARM" in r.message
    assert CFG.out_svon not in {p for _, pairs in sim.log for p, s in pairs if s}


def test_stale_io_refused_and_aborts():
    sim, hs = make()
    sim.io_valid = False
    r = hs.servo_on()
    assert not r.success and "fresh" in r.message
    sim.io_valid = True
    sim.sleep(0.05)
    ready(sim, hs)
    sim.events.append((sim.t + 0.2, lambda s: setattr(s, "io_valid", False)))
    r = hs.move_step(0)
    assert not r.success and "stale" in r.message
    # DRIVE-off was requested even though io is stale
    assert sim.log[-1][1][0] == (CFG.out_drive, False)
    sim.io_valid = True
    assert hs.outputs_uncertain          # cleanup could not be confirmed on stale io
    assert hs.release_all().success      # fresh io again: confirmed all-off clears the latch
    assert not hs.outputs_uncertain
    final_off(sim)


def test_output_service_failure_latches():
    sim, hs = make()
    ready(sim, hs)
    sim.reject_outputs = True
    r = hs.move_step(0)
    assert not r.success and "NOT CONFIRMED" in r.message
    assert hs.outputs_uncertain


def test_busy_rejects_concurrent():
    sim, hs = make()
    ready(sim, hs)
    seen = {}

    original = sim.sleep

    def sleeping(seconds):
        if "r" not in seen:
            seen["busy"] = hs.busy
            seen["r"] = hs.release_all()   # re-entrant call while running
        original(seconds)

    sim.sleep = sleeping
    r = hs.move_step(1)
    assert r.success
    assert seen["busy"] is True
    assert not seen["r"].success and "busy" in seen["r"].message
    assert not hs.busy


def test_drive_already_on_refused_and_release_all_recovers():
    sim, hs = make()
    ready(sim, hs)
    sim.dout.add(CFG.out_drive)
    sim.sleep(0.05)
    sim.log.clear()
    r = hs.move_step(0)
    assert not r.success and "DRIVE" in r.message
    assert sim.log == []     # a refusal touches no output
    assert hs.release_all().success
    sim.sleep(0.1)
    assert not (sim.dout & set(CFG.jxc_outputs))


def test_servo_off():
    sim, hs = make()
    ready(sim, hs)
    r = hs.servo_off()
    assert r.success, r.message
    assert not sim.svre


# ---------------------------------------------------------------- review additions
def sent(sim, pin, state):
    return any((pin, state) in pairs for _, pairs in sim.log)


def test_repeated_home_with_seton_does_not_pulse_setup():
    sim, hs = make()
    ready(sim, hs)
    sim.sleep(0.05)
    sim.log.clear()
    r = hs.home()
    assert r.success and "already homed" in r.message
    assert not sent(sim, CFG.out_setup, True)


def test_home_times_out_with_setup_confirmed_and_turns_it_off():
    sim, hs = make(home_time=100.0)
    hs.cfg.home_timeout_s = 0.3
    hs.servo_on()
    sim.seton = False
    r = hs.home()
    assert not r.success and "timeout" in r.message
    assert sent(sim, CFG.out_setup, True)
    final_off(sim)


def test_sample_tracker_stale_zero_and_regressing_stamps():
    tr = SampleTracker()
    assert not tr.view(10.0).valid and tr.view(10.0).stamp == 0.0
    assert not tr.accept(True, 0.0, [1], [])          # zero stamp invalid
    assert tr.accept(True, 5.0, [1], [2])
    v = tr.view(5.1)
    assert v.valid and v.age_s == pytest.approx(0.1) and v.din_on == {1} and v.dout_on == {2}
    assert not tr.accept(True, 4.0, [9], [9])         # regressing: ignored
    assert tr.view(5.2).din_on == {1}
    assert tr.accept(True, 5.0, [1], [2])             # heartbeat republish of the same stamp
    assert tr.view(6.0).age_s == pytest.approx(1.0)   # ... still ages
    assert not tr.accept(True, 0.0, [1], [2])         # zero stamp: invalid from now on
    assert not tr.view(6.0).valid


def test_heartbeat_old_stamp_is_not_a_confirmation():
    sim, hs = make()
    ready(sim, hs)
    sim.sleep(0.1)
    # old snapshot (everything off) frozen for 0.25 s (< stale limit): the confirmation of the
    # release_all must wait for a NEWER sample, not accept the old one that happens to show OFF.
    sim.dout.clear()
    sim.sleep(0.03)
    sim.freeze_until = sim.t + 0.25
    t0 = sim.t
    r = hs.release_all()
    assert r.success, r.message
    assert sim.t - t0 >= 0.24


def test_frozen_stamp_goes_stale_and_refuses():
    sim, hs = make()
    sim.freeze_until = sim.t + 5.0
    sim.sleep(0.5)                # old stamp ages beyond 0.3 s although messages "keep coming"
    r = hs.servo_on()
    assert not r.success and "fresh" in r.message


def test_drive_off_ignored_by_sim_latches_then_release_clears():
    sim, hs = make()
    ready(sim, hs)
    sim.ignore_drive_off = True
    r = hs.move_step(1)
    assert not r.success and "step 1 reached" in r.message   # the move finished ...
    assert "CLEANUP NOT CONFIRMED" in r.message              # ... but DRIVE off never confirmed
    assert hs.outputs_uncertain
    r = hs.move_step(1)
    assert not r.success and "outputs_uncertain" in r.message
    assert not hs.home().success and not hs.servo_on().success
    assert "outputs_uncertain" in hs.last_error
    assert not hs.release_all().success  # still not confirmed -> latch stays
    assert hs.outputs_uncertain
    sim.ignore_drive_off = False
    assert hs.release_all().success
    assert not hs.outputs_uncertain
    assert hs.servo_on().success and hs.home().success   # release_all dropped SVON
    assert hs.move_step(1).success


def test_set_outputs_timeout_latches():
    sim, hs = make()
    ready(sim, hs)
    sim.set_fail, sim.set_fail_applies = "set_digital_outputs timed out", True
    r = hs.move_step(0)
    assert not r.success and hs.outputs_uncertain
    sim.set_fail = None
    assert hs.release_all().success and not hs.outputs_uncertain


def test_cleanup_exception_path_still_retries_and_latches():
    sim, hs = make()
    ready(sim, hs)
    sim.set_exception = True
    n = len(sim.log)
    r = hs.move_step(0)
    assert not r.success and hs.outputs_uncertain
    assert len(sim.log) - n >= 3     # the failed set + two cleanup attempts, none propagated
    sim.set_exception = False
    assert hs.release_all().success


def test_svre_drop_mid_move_aborts():
    sim, hs = make(move_time=3.0)
    ready(sim, hs)
    sim.events.append((sim.t + 0.3, lambda s: (setattr(s, "svre", False), setattr(s, "_svon_since", None))))
    t0 = sim.t
    r = hs.move_step(1)
    assert not r.success and "SVRE" in r.message
    assert sim.t - t0 < 1.5     # returned early
    final_off(sim)


def test_seton_drop_mid_move_aborts():
    sim, hs = make(move_time=3.0)
    ready(sim, hs)
    sim.events.append((sim.t + 0.3, lambda s: setattr(s, "seton", False)))
    r = hs.move_step(1)
    assert not r.success and "SETON" in r.message
    final_off(sim)


def test_drive_dropped_by_robot_mid_move_aborts():
    sim, hs = make(move_time=3.0)
    ready(sim, hs)
    sim.events.append((sim.t + 0.3, lambda s: s.dout.discard(CFG.out_drive)))
    t0 = sim.t
    r = hs.move_step(1)
    assert not r.success and "DRIVE output dropped" in r.message
    assert sim.t - t0 < 1.5
    final_off(sim)


def test_servo_on_aborts_if_svon_disappears():
    sim, hs = make(servo_delay=2.0)
    sim.events.append((sim.t + 0.2, lambda s: s.dout.discard(CFG.out_svon)))
    r = hs.servo_on()
    assert not r.success and "SVON" in r.message


def test_inp_already_low_before_drive_does_not_complete_early():
    sim, hs = make(move_time=0.4)
    ready(sim, hs)
    sim.inp = False
    sim.sleep(0.05)
    t0 = sim.t
    r = hs.move_step(1)
    assert r.success, r.message
    drive_on = [t for t, st in sim.drive_trace if st][0]
    assert sim.t - drive_on >= 0.4      # only after the move really ended (INP back ON)


def test_inp_pre_drive_sample_does_not_confirm_drive():
    # io latency 0.2 s: the DRIVE bit appears late; INP-off/on logic must not start before it
    sim, hs = make(io_latency=0.2, move_time=0.2)
    hs.cfg.dout_confirm_timeout_s = 1.0
    ready(sim, hs)
    r = hs.move_step(2)
    assert r.success, r.message


def test_busy_path():
    cfg = JxcConfig(in_busy=25)
    sim = SimJxc(cfg=cfg, move_time=0.3)
    hs = JxcHandshake(sim, cfg)
    ready(sim, hs)
    r = hs.move_step(1)
    assert r.success, r.message
    assert sim.busy_out is False
    # BUSY never comes on (e.g. drive ignored): fails clearly
    sim2 = SimJxc(cfg=JxcConfig(in_busy=25))
    hs2 = JxcHandshake(sim2, sim2.cfg)
    ready(sim2, hs2)
    sim2.never_clears_inp = True
    sim2._on_drive_rising = lambda: None
    r = hs2.move_step(1)
    assert not r.success and "BUSY never came ON" in r.message
    final_off(sim2)


def test_busy_wired_home_waits_for_busy_off():
    cfg = JxcConfig(in_busy=25)
    sim = SimJxc(cfg=cfg, home_time=0.2)
    hs = JxcHandshake(sim, cfg)
    assert hs.servo_on().success
    sim.busy_out = True
    sim.events.append((sim.t + 0.6, lambda s: setattr(s, "busy_out", False)))
    t0 = sim.t
    assert hs.home().success
    assert sim.t - t0 >= 0.59


def test_shutdown_outputs_off_confirms_and_aborts_running():
    sim, hs = make()
    ready(sim, hs)
    sim.dout.update({CFG.out_drive})
    sim.sleep(0.05)
    ok, why = hs.shutdown_outputs_off()
    assert ok, why
    sim.sleep(0.05)
    assert CFG.out_drive not in sim.dout
    assert not hs.move_step(0).success       # aborted by the shutdown flag


# ---------------------------------------------------------------- RESET / reset_alarm
def test_default_pins_operator_confirmed():
    assert (CFG.out_in0, CFG.out_in1, CFG.out_setup, CFG.out_drive, CFG.out_reset,
            CFG.out_svon) == (21, 22, 23, 24, 25, 26)
    assert CFG.out_reset in CFG.jxc_outputs


def test_reset_alarm_happy_path():
    sim, hs = make()
    sim.dout.update({CFG.out_drive, CFG.out_setup})
    sim.alarm_ok = False
    sim.sleep(0.1)
    r = hs.reset_alarm()
    assert r.success, r.message
    sim.sleep(0.1)
    assert sim.alarm_ok
    assert not ({CFG.out_reset, CFG.out_drive, CFG.out_setup} & sim.dout)
    assert not hs.outputs_uncertain
    # order: DRIVE/SETUP off before RESET on, RESET off last
    seq = [(p, st) for _, pairs in sim.log for p, st in pairs]
    assert seq.index((CFG.out_drive, False)) < seq.index((CFG.out_reset, True))
    assert seq.index((CFG.out_setup, False)) < seq.index((CFG.out_reset, True))
    assert seq.index((CFG.out_reset, True)) < seq.index((CFG.out_reset, False))


def test_reset_alarm_refused_without_alarm():
    sim, hs = make()
    sim.sleep(0.1)
    r = hs.reset_alarm()
    assert not r.success and "no alarm" in r.message
    assert not sim.log          # nothing sent


def test_reset_alarm_refused_when_stale():
    sim, hs = make()
    sim.alarm_ok = False
    sim.sleep(0.05)
    sim.freeze_until = sim.t + 5.0
    sim.sleep(0.5)
    r = hs.reset_alarm()
    assert not r.success and "not valid/fresh" in r.message
    assert not sim.log


def test_reset_alarm_non_resettable_times_out_and_resets_off():
    sim, hs = make()
    sim.alarm_ok = False
    sim.alarm_resettable = False
    sim.sleep(0.1)
    r = hs.reset_alarm()
    assert not r.success and "alarm did not clear" in r.message and "power cycle" in r.message
    sim.sleep(0.1)
    assert CFG.out_reset not in sim.dout
    assert sent(sim, CFG.out_reset, True) and sent(sim, CFG.out_reset, False)
    assert not hs.outputs_uncertain


def test_reset_alarm_reset_off_unconfirmed_latches():
    sim, hs = make()
    sim.alarm_ok = False
    sim.sleep(0.1)
    orig = sim._step

    def stuck_step():
        orig()
        # RESET off is "sent" but the output stays on in digital_io
        if CFG.out_reset not in sim.dout and any(
                p == CFG.out_reset and not st for _, pairs in sim.log for p, st in pairs):
            sim.dout.add(CFG.out_reset)
    sim._step = stuck_step
    r = hs.reset_alarm()
    assert not r.success and hs.outputs_uncertain
    sim._step = orig
    assert not hs.servo_on().success      # latched
    sim.dout.discard(CFG.out_reset)
    assert hs.release_all().success and not hs.outputs_uncertain


def test_release_all_includes_reset():
    sim, hs = make()
    sim.dout.add(CFG.out_reset)
    sim.sleep(0.1)
    assert hs.release_all().success
    assert (CFG.out_reset, False) in [pr for _, pairs in sim.log for pr in pairs]
    sim.sleep(0.1)
    assert CFG.out_reset not in sim.dout


def test_reset_alarm_refused_while_other_command_runs():
    sim, hs = make()
    sim.alarm_ok = False
    sim.sleep(0.1)
    seen = {}
    orig = sim.sleep

    def sleeper(sec):
        if "r" not in seen:
            seen["r"] = hs.reset_alarm()
        orig(sec)
    sim.sleep = sleeper
    hs.release_all()
    assert not seen["r"].success and "busy" in seen["r"].message


# ---------------------------------------------------------------- automatic alarm reset
def auto_make(**cfg_kw):
    sim = SimJxc(cfg=JxcConfig(**cfg_kw), move_time=0.2)
    sim.push_alarm_delay_s = 0.2
    hs = JxcHandshake(sim, sim.cfg)
    hs.logs = []
    hs.log = lambda level, message: hs.logs.append((level, message))
    return sim, hs


def monitor(sim, hs, seconds, period=0.05):
    end = sim.t + seconds
    results = []
    while sim.t < end:
        r = hs.auto_reset_poll()
        if r is not None:
            results.append(r)
        sim.sleep(period)
    return results


def test_auto_reset_after_grip():
    sim, hs = auto_make()
    ready(sim, hs)
    r = hs.move_step(2)
    assert r.success and "reached" in r.message      # alarm comes only after success
    sim.log.clear()
    results = monitor(sim, hs, 1.5)
    assert len(results) == 1 and results[0].success
    assert sim.alarm_ok and sim.svre and sim.seton and sim.inp
    assert CFG.out_reset not in sim.dout and CFG.out_svon in sim.dout
    flat = [p for _, pairs in sim.log for p in pairs]
    assert (CFG.out_reset, True) in flat and (CFG.out_reset, False) in flat
    assert (CFG.out_drive, True) not in flat and (CFG.out_svon, True) not in flat  # no move/servo
    assert (CFG.out_setup, True) not in flat                                   # no re-home
    assert hs.auto_reset_state["count"] == 1 and hs.auto_reset_state["enabled"]
    assert any("auto reset #1: alarm cleared, SVRE back" in m for _, m in hs.logs)


def test_auto_reset_svre_not_returning_is_warned():
    sim, hs = auto_make(servo_on_timeout_s=0.3)
    sim.svre_return_delay = 5.0
    ready(sim, hs)
    hs.move_step(2)
    monitor(sim, hs, 1.5)
    assert sim.alarm_ok and hs.auto_reset_state["count"] == 1
    assert any("SVRE did not return" in m for _, m in hs.logs)


def test_auto_reset_waits_for_running_command():
    sim, hs = auto_make()
    sim.push_alarm_delay_s = None
    ready(sim, hs)
    sim.move_time = 1.0
    sim.events.append((sim.t + 0.3, lambda s: setattr(s, "alarm_ok", False)))
    seen = []
    orig = sim.sleep

    def sleeper(sec):
        seen.append(hs.auto_reset_poll())     # a monitor that polls while the command runs
        orig(sec)
    sim.sleep = sleeper
    sim.log.clear()
    r = hs.move_step(1)
    sim.sleep = orig
    assert not r.success and "ALARM" in r.message
    assert all(x is None for x in seen)
    assert (CFG.out_reset, True) not in [p for _, pairs in sim.log for p in pairs]
    results = monitor(sim, hs, 1.0)           # command finished and cleaned up: now it resets
    assert len(results) == 1 and results[0].success and sim.alarm_ok


def test_auto_reset_not_when_uncertain_or_stale_or_disabled_param():
    sim, hs = auto_make()
    sim.alarm_ok = False
    sim.sleep(0.1)
    hs._latch("test")
    assert monitor(sim, hs, 0.3) == [] and not sim.log
    hs._uncertain = False
    sim.freeze_until = sim.t + 5.0
    sim.sleep(0.5)
    assert monitor(sim, hs, 0.3) == [] and not sim.log
    sim.freeze_until = 0.0
    sim.sleep(0.1)
    hs.cfg.auto_reset_alarm = False
    assert monitor(sim, hs, 0.3) == [] and not sim.log and not hs.auto_reset_state["enabled"]
    assert "false" in hs.auto_reset_state["reason"]


def test_auto_reset_rate_limit_and_manual_reset_reenables():
    sim, hs = auto_make(auto_reset_max=2, auto_reset_window_s=60.0)
    ready(sim, hs)
    for _ in range(2):
        assert hs.move_step(2).success
        assert len(monitor(sim, hs, 1.0)) == 1 and sim.alarm_ok
    assert hs.move_step(2).success
    results = monitor(sim, hs, 1.0)           # 3rd within the window: refused, disabled
    assert results == [] and not sim.alarm_ok
    st = hs.auto_reset_state
    assert not st["enabled"] and "repeated JXC alarms" in st["reason"] and st["count"] == 2
    assert any(level == "error" and "repeated JXC alarms" in m for level, m in hs.logs)
    assert monitor(sim, hs, 0.5) == []        # stays disabled
    assert hs.reset_alarm().success           # manual reset re-enables
    assert hs.auto_reset_state["enabled"]
    sim.sleep(0.4)
    assert hs.move_step(2).success
    assert len(monitor(sim, hs, 1.0)) == 1


def test_auto_reset_window_slides():
    sim, hs = auto_make(auto_reset_max=1, auto_reset_window_s=5.0)
    ready(sim, hs)
    assert hs.move_step(2).success
    assert len(monitor(sim, hs, 1.0)) == 1
    sim.sleep(6.0)                            # first reset leaves the window
    assert hs.move_step(2).success
    assert len(monitor(sim, hs, 1.0)) == 1 and hs.auto_reset_state["enabled"]


def test_auto_reset_failure_disables_without_retry():
    sim, hs = auto_make()
    ready(sim, hs)
    sim.alarm_resettable = False
    assert hs.move_step(2).success
    results = monitor(sim, hs, 3.0)
    assert len(results) == 1 and not results[0].success
    assert not hs.auto_reset_state["enabled"]
    assert any(level == "error" and "failed" in m for level, m in hs.logs)
    n = len([1 for _, pairs in sim.log if (CFG.out_reset, True) in pairs])
    assert n == 1                              # no retry loop
    assert monitor(sim, hs, 1.0) == []


def test_service_refused_during_auto_reset():
    sim, hs = auto_make()
    ready(sim, hs)
    assert hs.move_step(2).success
    seen = {}
    orig = sim.sleep

    def sleeper(sec):
        if hs.auto_reset_state["running"] and "move" not in seen:
            seen["move"] = hs.move_step(1)
            seen["servo"] = hs.servo_on()
            seen["reset"] = hs.reset_alarm()
        orig(sec)
    sim.sleep = sleeper
    monitor(sim, hs, 1.5)
    sim.sleep = orig
    for k in ("move", "servo", "reset"):
        assert not seen[k].success and "busy: auto alarm reset running" in seen[k].message
    assert sim.alarm_ok and hs.auto_reset_state["count"] == 1
    assert not hs.auto_reset_state["running"]
