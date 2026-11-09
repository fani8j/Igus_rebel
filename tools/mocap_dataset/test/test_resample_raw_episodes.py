import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import resample_raw_episodes as rr  # noqa: E402


def test_resample_linear_and_previous_image():
    t = np.arange(0, 11) / 15.0
    q = np.stack([t * 2.0, np.sin(t)], axis=1)
    tk, qk, idx, filled = rr.resample(t, q, 30, 0)
    assert len(tk) == 21 and filled == 0
    assert np.allclose(qk[:, 0], tk * 2.0)
    assert list(idx[:6]) == [0, 0, 1, 1, 2, 2] and idx[-1] == 10      # previous image, never a future one
    assert np.all(t[idx] <= tk + 1e-9)


def test_fill_stale_short_runs_only():
    t = np.arange(8) / 15.0
    q = np.array([[0], [1], [1], [3], [3], [3], [3], [4]], float)
    out, n = rr.fill_stale(q, t, 2)
    assert n == 1 and np.allclose(out[:4, 0], [0, 1, 2, 3]) and np.allclose(out[4:7, 0], 3)


def test_convert_episode_end_to_end(tmp_path):
    src = tmp_path / "src" / "real_0001"
    n = 6
    (src / "images" / "head").mkdir(parents=True)
    for i in range(n):
        (src / "images" / "head" / f"{i:06d}.jpg").write_bytes(b"x%d" % i)
    (src / "meta.json").write_text(json.dumps({"fps": 15.0, "num_steps": n}))
    with open(src / "steps.jsonl", "w") as f:
        for i in range(n):
            f.write(json.dumps({"t": i / 15, "joints": {"j": float(i)}, "welded": False, "phase": "play", "img_age": {"head": 0.1}}) + "\n")
    out = tmp_path / "out"
    rr.main([str(tmp_path / "src"), "--out", str(out)])
    steps = [json.loads(line) for line in (out / "real_0001" / "steps.jsonl").read_text().splitlines()]
    assert len(steps) == 11 and steps[1]["joints"]["j"] == 0.5 and steps[1]["src_step"] == 0
    assert json.loads((out / "real_0001" / "meta.json").read_text())["fps"] == 30.0
    assert (out / "real_0001" / "images" / "head" / "000003.jpg").read_bytes() == b"x1"
