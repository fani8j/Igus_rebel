"""Tests for episode_score.py (per-episode quality score) on synthetic episodes; no dataset needed.

    cd tools/mocap_dataset && python3 -m pytest -p no:anyio test/test_episode_score.py
"""
import sys
from pathlib import Path

import av
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import episode_score as es  # noqa: E402

FPS = 30.0


def motion(n=900, branch=-1, seed=0, amp=0.6):
    """(n, 6) smooth task-like motion; j4/j6 counter-rotate to +-2.8 rad on ``branch`` (-1: j4-/j6+, +1 opposite, 0 none)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FPS
    ph = np.sin(np.pi * t / t[-1]) ** 2                       # 0 -> 1 -> 0: returns to the start pose
    q = np.column_stack([amp * ph * (1 + 0.1 * j) for j in range(6)]) + 0.02 * rng.standard_normal((1, 6))
    q[:, 3] = branch * 2.8 * ph
    q[:, 5] = -branch * 2.8 * ph
    return q


def episode(name, q, prefiltered=True, ts=None, group="g"):
    n = len(q)
    return {"group": group, "prefiltered": prefiltered, "name": name, "label": name, "episode_index": 0, "root": "r",
            "arm": "arm_right", "fps": FPS, "ts": np.arange(n) / FPS if ts is None else ts, "q_raw": q, "version": "v3.0",
            "videos": {}, "meta_row": {"length": n}}


def score(eps):
    mets = [es.episode_metrics(e) for e in eps]
    return es.score_group(eps, mets, None)


def dataset(extra=None, n_ok=8):
    eps = [episode(f"ok{i}", motion(900 + 6 * i, seed=i)) for i in range(n_ok)]
    return eps + (extra or [])


def by_name(res):
    return {r["label"]: r for r in res}


def test_weights_and_ramp():
    assert sum(es.WEIGHTS.values()) == 100
    assert es.ramp(1, 1, 4) == 1 and es.ramp(4, 1, 4) == 0 and es.ramp(2.5, 1, 4) == pytest.approx(0.5)
    assert es.ramp(0.6, 0.6, 0.25) == 1 and es.ramp(0.25, 0.6, 0.25) == 0     # inverted pair
    assert es.ramp(float("nan"), 1, 4) == 0


def test_clean_dataset_is_kept():
    res = score(dataset())
    assert all(r["recommendation"] == "keep" for r in res), [(r["label"], r["total"], r["reason"]) for r in res]
    assert all(r["total"] > 85 for r in res)
    assert all(r["sub_scores"]["camera"] is None for r in res)               # no video given: weight redistributed


def test_opposite_and_mixed_wrist_branch_flagged():
    opp = episode("opp", motion(branch=+1, seed=11))
    both = episode("both", np.vstack([motion(450, branch=-1), motion(450, branch=+1)]))
    r = by_name(score(dataset([opp, both])))
    assert r["opp"]["raw"]["branch_kind"] == "other" and r["opp"]["recommendation"] == "review"
    assert any("differs from the dataset majority" in f for f in r["opp"]["flags"])
    assert r["both"]["raw"]["branch_kind"] == "both" and r["both"]["recommendation"] != "keep"


def test_start_pose_flagged():
    q = motion(seed=12)
    q[:, 0] += np.radians(80)                                                # whole episode offset: start and end far away
    r = by_name(score(dataset([episode("shift", q)])))
    assert r["shift"]["sub_scores"]["start_end"] < 0.1 and r["shift"]["recommendation"] == "review"


def test_aborted_episode_is_dropped():
    r = by_name(score(dataset([episode("short", motion(150, seed=13))])))
    assert r["short"]["recommendation"] == "drop"
    assert any(f.startswith("ABORTED") for f in r["short"]["hard_flags"])


def test_corrupt_jump_and_timestamps_are_dropped():
    q = motion(seed=14)
    q[400:, 1] += 0.5                                                        # 28 deg step in one frame
    ts = np.arange(len(q)) / FPS
    ts2 = ts.copy()
    ts2[300:] -= 0.5                                                         # time goes back
    r = by_name(score(dataset([episode("jump", q), episode("time", motion(seed=15), ts=ts2)])))
    assert r["jump"]["recommendation"] == "drop" and any("impossible joint step" in f for f in r["jump"]["hard_flags"])
    assert r["time"]["recommendation"] == "drop" and any("non-increasing" in f for f in r["time"]["hard_flags"])
    assert r["jump"]["sub_scores"]["integrity"] == 0


def test_pauses_and_nan():
    q = motion(seed=16)
    q[300:700] = q[300]                                                      # 13 s standing still
    r = by_name(score(dataset([episode("still", q)])))
    assert r["still"]["sub_scores"]["pauses"] < 0.2 and any("standing still" in f for f in r["still"]["flags"])
    qn = motion(seed=17)
    qn[100, 2] = np.nan
    r = by_name(score(dataset([episode("nan", qn)])))
    assert r["nan"]["recommendation"] == "drop" and any("NaN" in f for f in r["nan"]["hard_flags"])


def test_raw_source_is_filtered_and_noise_reduces_smoothness_gap():
    q = motion(seed=18)
    noisy = q + 0.004 * np.sin(2 * np.pi * 6.3 * np.arange(len(q))[:, None] / FPS)
    m_raw = es.episode_metrics(episode("raw", noisy, prefiltered=False))
    m_pre = es.episode_metrics(episode("pre", noisy, prefiltered=True))
    assert m_raw["hf_rms_deg"] < 0.1 * m_pre["hf_rms_deg"]                   # filter applied in memory for raw only
    assert m_raw["filter_dev_max_deg"] < 1.0 and "stale_share_moving" in m_raw


def _write_mp4(path, frames):
    with av.open(str(path), "w") as c:
        s = c.add_stream("libx264", rate=30)
        s.width, s.height, s.pix_fmt = frames[0].shape[1], frames[0].shape[0], "yuv420p"
        s.options = {"crf": "10", "g": "5"}
        for f in frames:
            for p in s.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
                c.mux(p)
        for p in s.encode():
            c.mux(p)


def test_video_scan_detects_black_and_frozen(tmp_path):
    rng = np.random.default_rng(0)
    base = rng.integers(60, 200, (64, 96, 3), dtype=np.uint8)
    good = [np.clip(base.astype(int) + rng.integers(-20, 20, base.shape), 0, 255).astype(np.uint8) for _ in range(90)]
    _write_mp4(tmp_path / "good.mp4", good)
    frozen = good[:20] + [good[19]] * 50 + good[20:40]
    _write_mp4(tmp_path / "frozen.mp4", frozen)
    _write_mp4(tmp_path / "black.mp4", [np.zeros((64, 96, 3), np.uint8)] * 90)
    out = {}
    for n, rows in (("good", 90), ("frozen", 90), ("black", 90)):
        r = es.scan_video_file({"path": str(tmp_path / f"{n}.mp4"), "fps": FPS,
                                "segments": [{"key": "k", "from_ts": 0.0, "rows": rows, "cam": "c", "moving": np.ones(rows, bool)}]})
        out[n] = r["segments"]["k"]
    assert out["good"]["frozen_share"] == 0 and out["good"]["black_share"] == 0 and out["good"]["brightness_mean"] > 60
    assert out["frozen"]["frozen_share"] > 0.4
    assert out["black"]["black_share"] == 1.0
    assert out["good"]["blur_median"] > 10 * (out["black"]["blur_median"] + 1e-6)
