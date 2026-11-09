import os
import json
import time

import numpy as np
import pytest

import synth
from side_cut_vision.trigger import (LookTrigger, StillTracker, default_out_dir, open_clip_dir,
                                     select_record_dir)

JOINTS = [f"arm_left_joint{i}" for i in range(1, 7)]


def feed(tr, t0, dur, q, rate=100.0, noise=0.0, drift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    n = int(dur * rate)
    for i in range(n + 1):
        t = t0 + i / rate
        qq = np.array(q) + noise * rng.normal(size=6) + drift * (t - t0)
        out.append((t, tr.update(t, JOINTS, qq.tolist())))
    return out


def test_still_after_motion_needs_window_and_duration():
    tr = StillTracker(JOINTS)
    q = np.zeros(6)
    res = feed(tr, 0.0, 1.0, q, drift=0.5)             # moving at 0.5 rad/s
    assert not any(s.still for _, s in res)
    res = feed(tr, 1.01, 1.0, q + 0.5)                 # stopped
    first_still = next(t for t, s in res if s.still)
    assert first_still - 1.01 < 0.2                    # speed window 0.1 s
    still_for = [s.still_for_s for _, s in res if s.still]
    assert max(still_for) >= 0.8
    reached = next(t for t, s in res if s.still_for_s >= 0.3)
    assert 1.29 <= reached < 1.7                       # the arm stopped at t=1.0


def test_encoder_quantisation_noise_is_not_motion():
    tr = StillTracker(JOINTS)
    res = feed(tr, 0.0, 1.0, np.zeros(6), noise=1e-5)  # 1e-5 rad jitter at 100 Hz = 0.1 rad/s per sample
    assert all(s.still for t, s in res if t > 0.15)


def test_slow_creep_above_threshold_is_motion():
    tr = StillTracker(JOINTS)
    res = feed(tr, 0.0, 1.0, np.zeros(6), drift=0.02)
    assert not any(s.still for _, s in res)
    tr = StillTracker(JOINTS)
    res = feed(tr, 0.0, 1.0, np.zeros(6), drift=0.005)
    assert any(s.still for _, s in res)


def test_unknown_inputs_are_not_still_and_reset():
    tr = StillTracker(JOINTS)
    feed(tr, 0.0, 0.5, np.zeros(6))
    s = tr.update(0.6, JOINTS[:5], [0] * 5)           # a joint missing
    assert not s.still and not s.known
    s = tr.update(0.61, JOINTS, [0, 0, 0, 0, 0, float("nan")])
    assert not s.still
    s = tr.update(0.62, JOINTS, [0] * 6)              # history was reset: not yet known
    assert not s.still
    # a time gap resets the timer
    feed(tr, 1.0, 0.5, np.zeros(6))
    s = tr.update(3.0, JOINTS, [0] * 6)
    assert not s.still
    # non-monotonic time
    s = tr.update(2.0, JOINTS, [0] * 6)
    assert not s.still


def test_look_trigger_fires_once_per_still_period_at_the_edge():
    arm = synth.load_arm("left")
    lt = LookTrigger(arm)
    tip_ok = arm.S + 0.0165 * arm.across - 0.003 * arm.normal + 0.03 * arm.along
    tr = StillTracker(JOINTS)
    fires = []
    t = 0.0
    for t_, s in feed(tr, t, 1.0, np.zeros(6), drift=0.5):
        fires.append(lt.update(s, tip_ok))
    t = 1.01
    for t_, s in feed(tr, t, 1.5, np.ones(6)):
        fires.append(lt.update(s, tip_ok))
    assert sum(fires) == 1
    # moves again, stops again -> a second look
    for t_, s in feed(tr, 2.6, 0.5, np.ones(6), drift=0.5):
        fires.append(lt.update(s, tip_ok))
    for t_, s in feed(tr, 3.2, 1.0, np.ones(6) * 2):
        fires.append(lt.update(s, tip_ok))
    assert sum(fires) == 2


def test_look_trigger_gates():
    arm = synth.load_arm("left")
    still = type("S", (), {"still": True, "still_for_s": 1.0})()
    short = type("S", (), {"still": True, "still_for_s": 0.2})()
    far = arm.S + 0.08 * arm.across                    # 80 mm outside: too far
    inside = arm.S - 0.02 * arm.across                 # inside the carton
    low = arm.S + 0.02 * arm.across - 0.05 * arm.normal
    near = arm.S + 0.0165 * arm.across
    off_end = arm.S + 0.0165 * arm.across + 0.5 * arm.along   # far beyond the edge extent
    for tip in (far, inside, low, off_end, [np.nan, 0, 0]):
        assert not LookTrigger(arm).update(still, tip)
    assert not LookTrigger(arm).update(short, near)
    assert LookTrigger(arm).update(still, near)
    lt = LookTrigger(arm)
    assert lt.update(still, near) and not lt.update(still, near)     # only once


# ----------------------------------------------------------------- record dir selection
def _clip(parent, name, recording, age_s=0.0, now=None):
    d = parent / name
    d.mkdir()
    cj = d / "clip.json"
    cj.write_text(json.dumps({"recording": recording}))
    t = (now or time.time()) - age_s
    os.utime(cj, (t, t))
    return str(d)


def test_record_dir_uses_open_clip_else_dated_folder(tmp_path):
    now = time.time()
    _clip(tmp_path, "20261020_100000", False, 5000, now)
    old_stale = _clip(tmp_path, "20261019_100000", True, 86400, now)        # crashed recorder, stale
    assert open_clip_dir(str(tmp_path), now) is None
    day = time.strftime("%Y%m%d", time.localtime(now))
    assert select_record_dir(str(tmp_path), now) == str(tmp_path / f"side_cut_vision_{day}")
    newest = _clip(tmp_path, "20261021_120000", True, 3, now)
    _clip(tmp_path, "20261021_110000", True, 600, now)                       # older open clip
    assert open_clip_dir(str(tmp_path), now) == newest
    assert select_record_dir(str(tmp_path), now) == newest
    assert old_stale != newest


def test_record_dir_robust_to_bad_files(tmp_path):
    now = time.time()
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "clip.json").write_text("{not json")
    (tmp_path / "y").mkdir()                                                 # no clip.json
    (tmp_path / "z.txt").write_text("hi")
    assert open_clip_dir(str(tmp_path), now) is None
    assert open_clip_dir(str(tmp_path / "missing"), now) is None
    day = time.strftime("%Y%m%d", time.localtime(now))
    assert select_record_dir(str(tmp_path / "missing"), now).endswith(f"side_cut_vision_{day}")


def test_default_out_dir_falls_back_without_the_drive():
    assert default_out_dir(lambda p: True, lambda p, m: True).endswith("igus_nvidia/cut_videos")
    assert default_out_dir(lambda p: False, lambda p, m: True) == "/home/farhan/Igus_rebel/outputs/cut_videos"
    assert default_out_dir(lambda p: True, lambda p, m: False) == "/home/farhan/Igus_rebel/outputs/cut_videos"


def test_look_trigger_ignores_an_arm_that_was_already_at_rest():
    arm = synth.load_arm("left")
    near = arm.S + 0.0165 * arm.across
    rest = type("S", (), {"still": True, "still_for_s": 287.0})()
    lt = LookTrigger(arm)
    assert not lt.update(rest, near)
    # even if the stillness is reported again with a plausible duration, the trigger stays disarmed until motion
    dwell = type("S", (), {"still": True, "still_for_s": 0.4})()
    assert not lt.update(dwell, near)
    moving = type("S", (), {"still": False, "still_for_s": 0.0})()
    lt.update(moving, near)
    assert lt.update(dwell, near)
