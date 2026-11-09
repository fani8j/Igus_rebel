"""Perturbation / relabel math of augment_state_noise.py on synthetic data (no dataset, no lerobot needed)."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import augment_state_noise as A  # noqa: E402

FPS = 30.0
LIM = np.radians(np.array([[-178, 178], [-79, 139], [-79, 139], [-178, 178], [-94, 94], [-178, 178]], dtype=float))


def _traj(n=300, amp=0.3):
    t = np.arange(n) / FPS
    return np.stack([amp * np.sin(0.5 * t + j) for j in range(6)], axis=1)


def test_decay_profile_step_and_decay():
    p = A.decay_profile(100, 20, 0.3, FPS)
    assert (p[:20] == 0).all() and p[20] == 1.0
    assert p[21] == pytest.approx(math.exp(-1 / (0.3 * FPS)))
    assert p[50] == pytest.approx(math.exp(-30 / (0.3 * FPS)))
    assert (np.diff(p[20:]) < 0).all()


def test_next_state_action_consistency():
    s = _traj()
    a = A.next_state_action(s)
    assert np.array_equal(a[:-1], s[1:]) and np.array_equal(a[-1], s[-1])


def test_action_offset_excludes_unstarted_event():
    n, t0 = 100, 40
    eps = np.radians(np.array([3.0, -2, 1, 0.5, -3, 2]))
    ao = A.action_offset(n, [(t0, eps)], 0.3, FPS)
    assert (ao[:t0] == 0).all()                       # action[t0-1] knows nothing about the event
    assert ao[t0] == pytest.approx(eps * math.exp(-1 / (0.3 * FPS)))
    assert ao[50] == pytest.approx(eps * math.exp(-11 / (0.3 * FPS)))
    assert ao[-1] == pytest.approx(eps * math.exp(-(n - 1 - t0) / (0.3 * FPS)))     # last frame repeats s_obs[-1]


def test_event_sampling_consistent_and_decaying():
    s = _traj()
    n = len(s)
    cand = np.zeros(n, bool); cand[20:250] = True
    near = np.zeros(n, bool); near[100:150] = True
    far, nr = np.radians(np.full(6, 3.0)), np.radians(np.full(6, 1.5))
    off, ev, rej, unpl = A.sample_offsets(np.random.default_rng(1), s, fps=FPS, cand=cand, near=near, eps_far_rad=far,
                                          eps_near_rad=nr, tau_s=0.3, n_events=3, min_spacing_frames=30, limits=LIM,
                                          margin_rad=0.0, vmax=A.VMAX)
    assert len(ev) == 3 and unpl == 0
    exp = sum(A.decay_profile(n, e["t0"], 0.3, FPS)[:, None] * np.radians(e["eps0_deg"])[None] for e in ev)
    assert np.allclose(off, exp, atol=1e-9)
    for e in ev:
        lim = 1.5 if e["near_contact"] else 3.0
        assert np.abs(e["eps0_deg"]).max() <= lim + 1e-3
        assert 20 <= e["t0"] < 250
    assert (off[:min(e["t0"] for e in ev)] == 0).all()
    s_obs = s + off
    evs = [(e["t0"], np.radians(e["eps0_deg"])) for e in ev]
    act = A.next_state_action(s) + A.action_offset(n, evs, 0.3, FPS)
    # action[t] == s_obs[t+1] everywhere except t0-1, where it is s_obs[t0] - eps0 (events are > 1 frame apart)
    t0s = sorted(e["t0"] for e in ev)
    mask = np.ones(n - 1, bool); mask[[t - 1 for t in t0s]] = False
    assert np.allclose(act[:-1][mask], s_obs[1:][mask], atol=1e-12)
    for e in ev:
        t0 = e["t0"]
        older = sum(A.decay_profile(n, o["t0"], 0.3, FPS)[t0] * np.radians(o["eps0_deg"]) for o in ev if o["t0"] < t0)
        assert np.allclose(act[t0 - 1], s[t0] + older, atol=1e-12)           # no trace of the starting event
        assert np.allclose(act[t0 - 1], s_obs[t0] - np.radians(e["eps0_deg"]), atol=1e-12)
    assert all(b - a > 30 for a, b in zip(t0s, t0s[1:]))
    assert A.check_perturbed(s, off, LIM, 0.0, A.VMAX, FPS, t0s, act_offset=act - A.next_state_action(s)) is None
    # commanded displacement is small even though the state step is 3 deg
    assert np.abs(act - s_obs).max() * FPS < A.VMAX


def test_step_is_exempt_from_speed_only_at_step_frame():
    s = np.zeros((100, 6))
    off = A.decay_profile(100, 30, 0.3, FPS)[:, None] * np.radians(np.full(6, 3.0))[None]
    assert A.check_perturbed(s, off, LIM, 0.0, A.VMAX, FPS, [30]) is None          # 3 deg step = 1.57 rad/s, exempt
    assert "speed" in A.check_perturbed(s, off, LIM, 0.0, A.VMAX, FPS, [])          # without the exemption it fails
    # a second, unexempted jump is caught
    off2 = off.copy(); off2[60:] += np.radians(3.0)
    assert "speed" in A.check_perturbed(s, off2, LIM, 0.0, A.VMAX, FPS, [30])


def test_action_has_no_exemption():
    s = np.zeros((100, 6))
    eps = np.radians(np.full(6, 3.0))
    off = A.decay_profile(100, 30, 0.3, FPS)[:, None] * eps[None]
    ok = A.action_offset(100, [(30, eps)], 0.3, FPS)
    assert A.check_perturbed(s, off, LIM, 0.0, A.VMAX, FPS, [30], act_offset=ok) is None
    # the old (wrong) label action[t] = s_obs[t+1] carries the step at t0-1: commands 1.6 rad/s, rejected
    wrong = A.next_state_action(off)
    assert "commanded" in A.check_perturbed(s, off, LIM, 0.0, A.VMAX, FPS, [30], act_offset=wrong)


def test_recovery_speed_violation_rejected():
    s = np.zeros((100, 6))
    # very short tau and big amplitude: the recovery itself is faster than vmax
    off = A.decay_profile(100, 30, 0.02, FPS)[:, None] * np.radians(np.full(6, 3.0))[None]
    assert "speed" in A.check_perturbed(s, off, LIM, 0.0, A.VMAX, FPS, [30])


def test_limit_violation_rejected_and_resampled():
    n = 200
    s = np.zeros((n, 6)); s[:, 4] = LIM[4, 1] - np.radians(0.5)           # j5 0.5 deg from its upper limit
    off = np.zeros((n, 6)); off[50:, 4] = np.radians(2.0)
    assert "joint5" in A.check_perturbed(s, off, LIM, 0.0, A.VMAX, FPS, [50])
    cand = np.zeros(n, bool); cand[20:150] = True
    far = np.radians(np.full(6, 3.0))
    o, ev, rej, unpl = A.sample_offsets(np.random.default_rng(0), s, fps=FPS, cand=cand, near=np.zeros(n, bool), eps_far_rad=far,
                                        eps_near_rad=far, tau_s=0.3, n_events=1, min_spacing_frames=30, limits=LIM,
                                        margin_rad=0.0, vmax=A.VMAX, max_tries=500)
    assert rej > 0 and len(ev) == 1                                          # positive j5 draws were rejected
    assert (s + o)[:, 4].max() <= LIM[4, 1]
    assert ev[0]["eps0_deg"][4] < 0.5


def test_source_at_limit_does_not_block_unperturbed_frames():
    s = np.zeros((50, 6)); s[:, 0] = LIM[0, 1] + 0.01                        # source already outside
    assert A.check_perturbed(s, np.zeros((50, 6)), LIM, 0.0, A.VMAX, FPS, []) is None


def test_candidate_mask_excludes_idle_and_pauses():
    m = A.candidate_mask(300, 40, 260, [(100, 160)], 9)
    assert not m[:41].any() and not m[251:].any() and not m[100:160].any()
    assert m[41] and m[99] and m[160] and m[250]


def test_deterministic_for_same_seed():
    s = _traj(); n = len(s)
    cand = np.zeros(n, bool); cand[20:250] = True
    kw = dict(fps=FPS, cand=cand, near=np.zeros(n, bool), eps_far_rad=np.radians(np.full(6, 3.0)),
              eps_near_rad=np.radians(np.full(6, 1.5)), tau_s=0.3, n_events=2, min_spacing_frames=30, limits=LIM,
              margin_rad=0.0, vmax=A.VMAX)
    a = A.sample_offsets(np.random.default_rng([7, 1, 2]), s, **kw)[0]
    b = A.sample_offsets(np.random.default_rng([7, 1, 2]), s, **kw)[0]
    c = A.sample_offsets(np.random.default_rng([7, 1, 3]), s, **kw)[0]
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_near_contact_mask_lowest_fraction():
    z = np.linspace(0, 1, 101)
    m = A.near_contact_mask(z, 0, 100, 0.3)
    assert m.sum() == pytest.approx(30, abs=2) and m[0] and not m[-1]


def test_prepare_output_refuses_existing(tmp_path):
    with pytest.raises(A.AugmentError):
        A.prepare_output(tmp_path, False)
    with pytest.raises(A.AugmentError):
        A.prepare_output(tmp_path, True)                                      # exists, no marker -> never removed
    (tmp_path / A.MARKER).write_text('{"tool": "augment_state_noise.py"}')
    assert A.prepare_output(tmp_path, True).parent == tmp_path.parent


def test_urdf_limits_right_arm():
    lim = A.urdf_limits()
    assert lim.shape == (6, 2) and np.degrees(lim[1]).round().tolist() == [-79, 139]
