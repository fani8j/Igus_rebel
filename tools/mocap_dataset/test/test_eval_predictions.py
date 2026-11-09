"""Tests for eval_predictions.py (ACT ensemble, gate shares, groups).

    cd tools/mocap_dataset && python3 -m pytest -p no:anyio test/test_eval_predictions.py
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import eval_predictions as ep  # noqa: E402


def reference_ensemble(chunks, decay):
    """Straight copy of the client's ACTTemporalEnsembler.step (deque history, oldest first)."""
    import collections
    horizon = chunks.shape[1]
    history, out = collections.deque(maxlen=horizon), []
    for step, chunk in enumerate(chunks):
        history.append((step, chunk))
        cand = np.stack([pred[step - q] for q, pred in history])
        w = np.exp(-decay * np.arange(len(cand)))
        out.append(np.sum(cand * (w / w.sum())[:, None], axis=0))
    return np.array(out)


def test_act_ensemble_matches_the_clients_ensembler():
    rng = np.random.default_rng(1)
    chunks = rng.normal(size=(70, 30, 14))
    assert np.allclose(ep.act_ensemble(chunks, 0.01), reference_ensemble(chunks, 0.01), atol=1e-12)
    assert np.allclose(ep.act_ensemble(chunks, 0.0), reference_ensemble(chunks, 0.0), atol=1e-12)


def test_gate_shares_counts_frames_per_episode_and_ignores_other_channels():
    # two episodes of 40 frames, right arm in channels 7..12; chunk = state + offset
    T, H = 80, 30
    state = np.zeros((T, 14))
    pred = np.zeros((T, H, 14))
    pred[:, :, 7:13] = 0.05                                   # everything within both gates
    pred[40:60, :, 7] = 0.2                                   # episode 1, frames 0..19: first step 0.2 > 0.15, target 0.2 > 0.10
    pred[:, :, 0] = 5.0                                       # left arm channel must not matter
    episodes = np.repeat([0, 1], 40)
    g = ep.gate_shares(pred, state, episodes, 7)
    assert g["first_step_pass"] == pytest.approx(60 / 80)
    assert g["both_pass"] <= g["first_step_pass"] and g["both_pass"] <= g["target_step_pass"]
    assert g["target_step_pass"] < 1.0
    ok = ep.gate_shares(np.full((T, H, 14), 0.01), state, episodes, 0)
    assert ok["first_step_pass"] == ok["target_step_pass"] == ok["both_pass"] == 1.0


def test_analyse_groups_and_parse_groups(tmp_path):
    T, H = 60, 30
    rng = np.random.default_rng(0)
    state = rng.normal(0, .1, (T, 14))
    target = state[:, None, :] + rng.normal(0, .05, (T, H, 14))
    np.savez(tmp_path / "p.npz", prediction_rad=target + 0.01, target_rad=target, state_rad=state,
             valid=np.ones((T, H), dtype=bool), episode_index=np.repeat([0, 1, 2], 20))
    r = ep.analyse(tmp_path / "p.npz", "right", groups=ep.parse_groups(["a=0,2", "b=1"]))
    assert r["groups"]["a"]["frames"] == 40 and r["groups"]["b"]["frames"] == 20 and r["all"]["frames"] == 60
    assert r["groups"]["a"]["episodes"] == [0, 2] and set(r["per_episode"]) == {0, 1, 2}
    assert "gates" in r["all"] and "gates" in r["per_episode"][1]
    with pytest.raises(ValueError):
        ep.parse_groups(["nogroup"])
