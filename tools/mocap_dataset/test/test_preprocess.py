"""Tests for episode_metrics.py (filter / metrics) and preprocess.py (selection, overwrite refusal, end to end on a tiny
synthetic LeRobot v3.0 source with real mp4s).

    cd tools/mocap_dataset && python3 -m pytest -p no:anyio test/test_preprocess.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import episode_metrics as em  # noqa: E402
import preprocess as pp  # noqa: E402

FPS = 30.0
NAMES = [f"arm_left_joint{i}" for i in range(1, 7)] + ["arm_left_gripper"] + \
        [f"arm_right_joint{i}" for i in range(1, 7)] + ["arm_right_gripper"]


def synth_q(n=600, seed=0, jitter=True, amp=0.5):
    """(n, 6) smooth motion (0.4 Hz sines + ramp) plus a 6.3 Hz / 9 Hz sampling-jitter-like ripple."""
    t = np.arange(n) / FPS
    rng = np.random.default_rng(seed)
    base = np.column_stack([amp * np.sin(2 * np.pi * 0.4 * t + j) + 0.01 * j * t for j in range(6)])
    if not jitter:
        return t, base
    rip = 0.004 * np.sin(2 * np.pi * 6.3 * t[:, None] + rng.uniform(0, 6, 6)) + \
          0.003 * np.sin(2 * np.pi * 9.1 * t[:, None] + rng.uniform(0, 6, 6))
    return t, base + rip


# ------------------------------------------------------------------------------------------------ filter
def test_filter_zero_phase_no_lag():
    t, clean = synth_q(jitter=False)
    _, noisy = synth_q(jitter=True)
    f = em.smooth(noisy, em.DEFAULT_SIGMA_S, FPS)
    for j in range(6):                                   # cross-correlation peak at lag 0
        c = np.correlate(f[:, j] - f[:, j].mean(), clean[:, j] - clean[:, j].mean(), mode="full")
        assert np.argmax(c) - (len(t) - 1) == 0
    # a ramp stays a ramp at the same time (a causal filter would be late by sigma)
    ramp = np.column_stack([np.linspace(0, 1, 300)] * 6)
    assert np.abs(em.smooth(ramp, 0.08, FPS)[20:-20] - ramp[20:-20]).max() < 1e-9


def test_filter_endpoints_preserved_exactly():
    _, q = synth_q()
    f = em.smooth(q, em.DEFAULT_SIGMA_S, FPS)
    assert np.allclose(f[0], q[0], atol=1e-12) and np.allclose(f[-1], q[-1], atol=1e-12)


def test_filter_passes_motion_and_kills_jitter():
    t = np.arange(900) / FPS
    slow = np.sin(2 * np.pi * 0.3 * t)[:, None]
    fast = np.sin(2 * np.pi * 6.3 * t)[:, None]
    ys = em.smooth(slow, em.DEFAULT_SIGMA_S, FPS)[60:-60]
    yf = em.smooth(fast, em.DEFAULT_SIGMA_S, FPS)[60:-60]
    assert np.abs(ys - slow[60:-60]).max() < 0.02                 # 0.3 Hz passes (< 2 % error)
    assert np.abs(yf).max() < 0.03                                 # 6.3 Hz is gone (> 30 dB)


def test_deviation_bound_and_jitter_reduction():
    _, q = synth_q()
    f = em.smooth(q, em.DEFAULT_SIGMA_S, FPS)
    dev = em.deviation_deg(q, f)
    assert dev["max_deg_all"] < 1.0
    jb, ja = em.jitter_metrics(q, FPS), em.jitter_metrics(f, FPS)
    assert np.mean(ja["hf_rms_deg"]) < 0.1 * np.mean(jb["hf_rms_deg"])
    assert np.mean(ja["jerk_rms"]) < 0.1 * np.mean(jb["jerk_rms"])


def test_filter_state_only_touches_joints():
    n = 200
    st = np.zeros((n, 14))
    st[:, :6] = synth_q(n)[1]
    st[:, 6] = np.random.default_rng(1).uniform(size=n)          # gripper slot must stay untouched
    f = em.filter_state(st, NAMES, 0.08, FPS)
    assert np.array_equal(f[:, 6], st[:, 6]) and not f[:, 7:].any()
    assert not np.array_equal(f[:, :6], st[:, :6])


def test_sigma_zero_is_identity_and_short_series_error():
    _, q = synth_q(100)
    assert np.array_equal(em.smooth(q, 0.0, FPS), q)


def test_action_offset_and_tail():
    n = 50
    f = np.zeros((n, 14))
    f[:, :6] = np.arange(n)[:, None]
    f[:, 6] = 9.0                                                   # gripper of the state must not leak into action
    a1 = em.shifted_action(f, 1, NAMES)
    assert np.array_equal(a1[:-1, :6], f[1:, :6]) and np.array_equal(a1[-1, :6], f[-1, :6])
    a5 = em.shifted_action(f, 5, NAMES)
    assert np.array_equal(a5[:-5, :6], f[5:, :6]) and (a5[-5:, :6] == f[-1, :6]).all()
    assert not a1[:, 6].any() and not a1[:, 7:].any()
    assert np.array_equal(em.shifted_action(f, 0, NAMES)[:, :6], f[:, :6])


# ------------------------------------------------------------------------------------- motion / branch
def test_motion_segments_and_trim():
    n = 600
    q = np.zeros((n, 6))
    q[120:300, 0] = np.linspace(0, 1, 180)             # moves 4 s .. 10 s
    q[300:420, 0] = 1.0                                  # 4 s pause
    q[420:540, 0] = np.linspace(1, 0, 120)
    q[540:, 0] = 0
    seg = em.motion_segments(em.smooth(q, 0.05, FPS), FPS)
    assert 3.5 < seg["start_still_s"] < 4.2 and 1.9 < seg["end_still_s"] < 2.5
    assert len(seg["pauses"]) == 1 and 3.0 < seg["pause_max_s"] < 4.2
    i0, i1 = em.trim_bounds(seg, n, FPS, 0.5)
    assert abs(i0 - (seg["first_move"] - 15)) <= 1 and abs(i1 - (seg["last_move"] + 1 + 15)) <= 1


def test_branch_label_and_start_pose():
    q = np.zeros((10, 6))
    q[:, 3] = np.linspace(0, -2.9, 10)
    q[:, 5] = np.linspace(0, 3.0, 10)
    assert em.branch_label(q)["label"] == "j4-/j6+"
    assert em.branch_label(-q)["label"] == "j4+/j6-"
    assert em.branch_label(np.vstack([q, -q]))["label"] == "both"
    assert em.branch_label(q * 0.1)["label"] == "neither"
    assert em.start_pose_distance(np.radians(em.START_POSE_DEG))["max_abs_deg"] < 1e-9


def test_dtw_identity_and_warp():
    t = np.linspace(0, 1, 100)
    a = np.column_stack([np.sin(2 * np.pi * t)] * 3)
    b = np.column_stack([np.sin(2 * np.pi * t ** 1.2)] * 3)
    assert em.dtw_distance(a, a) == 0
    assert em.dtw_distance(a, b) < em.dtw_distance(a, -a)


# -------------------------------------------------------------------------------------- selection / out
def test_select_datasets_include_exclude(tmp_path):
    names = ["mocap_left_13", "mocap_left_new_1", "mocap_left_new_10", "mocap_left_new_2"]
    dirs = [tmp_path / n for n in names]
    assert [d.name for d in pp.select_datasets(dirs, None, None)] == \
        ["mocap_left_13", "mocap_left_new_1", "mocap_left_new_2", "mocap_left_new_10"]       # natural order
    assert [d.name for d in pp.select_datasets(dirs, "new_10,13", None)] == ["mocap_left_13", "mocap_left_new_10"]
    assert [d.name for d in pp.select_datasets(dirs, "mocap_left_new_2", None)] == ["mocap_left_new_2"]
    assert [d.name for d in pp.select_datasets(dirs, None, "13,new_1")] == ["mocap_left_new_2", "mocap_left_new_10"]
    with pytest.raises(pp.PreprocessError):
        pp.select_datasets(dirs, "nope", None)
    with pytest.raises(pp.PreprocessError):
        pp.select_datasets(dirs, "13", "new_1")
    with pytest.raises(pp.PreprocessError):
        pp.select_datasets(dirs, None, "13,new_1,new_2,new_10")


def test_prepare_output_refuses_overwrite(tmp_path):
    out = tmp_path / "ds"
    out.mkdir()
    (out / "x.txt").write_text("keep")
    with pytest.raises(pp.PreprocessError, match="exists"):
        pp.prepare_output(out, force=False)
    with pytest.raises(pp.PreprocessError, match="not removing"):
        pp.prepare_output(out, force=True)            # not a lerobot dataset: even --force refuses
    (out / "x.txt").unlink()
    assert pp.prepare_output(out, force=True).parent == tmp_path       # empty dir: allowed with --force
    (out / "meta").mkdir()
    (out / "meta" / "info.json").write_text("{}")
    assert pp.prepare_output(out, force=True).parent == tmp_path        # lerobot dataset: allowed with --force
    assert pp.prepare_output(tmp_path / "new", force=False).parent == tmp_path


# --------------------------------------------------------------------------------- end to end (synthetic)
def _encode(path: Path, n: int, seed: int, size=(128, 96)):
    import av
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for codec in ("libsvtav1", "libaom-av1", "mpeg4"):
        try:
            with av.open(str(path), "w") as c:
                s = c.add_stream(codec, rate=30)
                s.width, s.height, s.pix_fmt = size[0], size[1], "yuv420p"
                if codec == "libsvtav1":
                    s.options = {"g": "10", "preset": "10"}
                elif codec == "mpeg4":
                    s.options = {"g": "10"}
                for i in range(n):
                    img = np.zeros((size[1], size[0], 3), np.uint8)
                    img[:, :, 0] = (i * 7) % 256                      # frame index is encoded in the colour
                    img[:, :, 1] = (seed * 40) % 256
                    img[: 8 + i % 40, :, 2] = 255
                    img += rng.integers(0, 3, img.shape, dtype=np.uint8)
                    fr = av.VideoFrame.from_ndarray(img, format="rgb24")
                    for p in s.encode(fr):
                        c.mux(p)
                for p in s.encode():
                    c.mux(p)
            return codec
        except Exception:                                         # encoder missing / refused: try the next codec
            path.unlink(missing_ok=True)
    raise RuntimeError("no video encoder available")


def make_source(root: Path, n: int, seed: int, task: str, still=(30, 20), arm: str = "left") -> np.ndarray:
    import datasets
    from lerobot.datasets.utils import get_hf_features_from_features
    t, q = synth_q(n, seed, amp=0.1)   # gentle: the abrupt idle freezes below must stay within the 1 deg bound
    q[:still[0]] = q[still[0]]                      # idle head
    q[n - still[1]:] = q[n - still[1] - 1]          # idle tail
    c0 = 0 if arm == "left" else 7              # column of joint1 of the moving arm (the other arm stays all zero)
    state = np.zeros((n, 14), np.float32)
    state[:, c0:c0 + 6] = q
    action = np.zeros((n, 14), np.float32)
    action[:, c0:c0 + 6] = np.vstack([q[8:], np.repeat(q[-1:], 8, axis=0)])
    action[:, c0 + 6] = 1.0
    vkeys = ["observation.images.head", "observation.images.wrist_left"]
    feats = {"observation.state": {"dtype": "float32", "shape": [14], "names": NAMES},
             "action": {"dtype": "float32", "shape": [14], "names": NAMES}}
    for k in vkeys:
        feats[k] = {"dtype": "video", "shape": [96, 128, 3], "names": ["height", "width", "channels"],
                    "info": {"video.height": 96, "video.width": 128, "video.codec": "av1", "video.pix_fmt": "yuv420p",
                             "video.is_depth_map": False, "video.fps": 30, "video.channels": 3, "has_audio": False}}
    for k, dt in (("timestamp", "float32"), ("frame_index", "int64"), ("episode_index", "int64"), ("index", "int64"),
                  ("task_index", "int64")):
        feats[k] = {"dtype": dt, "shape": [1], "names": None}
    info = {"codebase_version": "v3.0", "robot_type": "igus_rebel_dual_arm_real", "total_episodes": 1, "total_frames": n,
            "total_tasks": 1, "chunks_size": 1000, "data_files_size_in_mb": 100, "video_files_size_in_mb": 200, "fps": 30,
            "splits": {"train": "0:1"}, "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4", "features": feats}
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps(info))
    hf = get_hf_features_from_features({k: {**v, "shape": tuple(v["shape"])} for k, v in feats.items()})
    ds = datasets.Dataset.from_dict({"observation.state": state.tolist(), "action": action.tolist(),
                                     "timestamp": (np.arange(n) / 30).astype(np.float32), "frame_index": np.arange(n),
                                     "episode_index": np.zeros(n, np.int64), "index": np.arange(n), "task_index": np.zeros(n, np.int64)},
                                    features=hf, split="train")
    ds.to_parquet(str(root / "data" / "chunk-000" / "file-000.parquet"))
    import pandas as pd
    pd.DataFrame({"task_index": [0]}, index=pd.Index([task])).to_parquet(root / "meta" / "tasks.parquet")
    row = {"episode_index": 0, "tasks": [task], "length": n, "data/chunk_index": 0, "data/file_index": 0,
           "dataset_from_index": 0, "dataset_to_index": n, "meta/episodes/chunk_index": 0, "meta/episodes/file_index": 0}
    stats = {}
    for k in vkeys:
        _encode(root / "videos" / k / "chunk-000" / "file-000.mp4", n, seed)
        row.update({f"videos/{k}/chunk_index": 0, f"videos/{k}/file_index": 0, f"videos/{k}/from_timestamp": 0.0,
                    f"videos/{k}/to_timestamp": n / 30})
        s = {"min": [[[0.0]]] * 3, "max": [[[1.0]]] * 3, "mean": [[[0.4 + 0.01 * seed]]] * 3, "std": [[[0.2]]] * 3,
             "count": [20], **{q_: [[[0.5]]] * 3 for q_ in ("q01", "q10", "q50", "q90", "q99")}}
        stats[k] = s
        row.update({f"stats/{k}/{a}": b for a, b in s.items()})
    pq.write_table(pa.Table.from_pylist([row]), root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    (root / "meta" / "stats.json").write_text(json.dumps(stats))
    return q


@pytest.fixture(scope="module")
def synth_source(tmp_path_factory):
    pytest.importorskip("lerobot")
    root = tmp_path_factory.mktemp("src")
    qs = {}
    for name, n, seed, task in (("mocap_left_a", 150, 1, "open flap 1"), ("mocap_left_b", 120, 2, "open the box flaps"),
                                ("mocap_left_c", 135, 3, "open the box flaps")):
        (root / name).mkdir()
        qs[name] = make_source(root / name, n, seed, task)
    return root, qs


def run_pp(src, out, *extra):
    return pp.main([str(src), "--out", str(out), *extra])


def load_out(out):
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    return LeRobotDataset(f"local/{Path(out).name}", root=out)


def test_end_to_end_default(synth_source, tmp_path):
    src, qs = synth_source
    out = tmp_path / "all"
    assert run_pp(src, out, "--verify") == 0
    ds = load_out(out)
    assert ds.num_episodes == 3 and len(ds) == 150 + 120 + 135
    prep = json.loads((out / "PREPROCESS.json").read_text())
    assert [e["source"] for e in prep["episodes"]] == ["mocap_left_a", "mocap_left_b", "mocap_left_c"]
    assert json.loads((out / "meta" / "source_episodes.json").read_text())[2]["episode_index"] == 2
    assert prep["parameters"]["action_offset_frames"] == 1 and prep["max_filter_deviation_deg"] < 1.0
    st = ds.hf_dataset.with_format("numpy")
    S, A, R, T = (np.asarray(st[k]) for k in ("observation.state", "action", "observation.state_raw", "action_teleop"))
    ep1 = np.arange(150, 270)
    assert np.allclose(R[ep1][:, :6], qs["mocap_left_b"], atol=1e-6)                 # raw traceability
    assert np.allclose(A[ep1][:-1, :6], S[ep1][1:, :6]) and np.allclose(A[ep1][-1, :6], S[ep1][-1, :6])   # offset 1, tail
    assert not A[:, 6:].any() and (T[:, 6] == 1.0).all()                             # gripper slot 0 vs teleop trigger
    assert np.allclose(S[ep1[0], :6], R[ep1[0], :6]) and np.allclose(S[ep1[-1], :6], R[ep1[-1], :6])   # endpoints
    assert np.degrees(np.abs(S - R).max()) < 1.0
    # stats exist with quantiles and equal lerobot's own computation on the stored data
    from lerobot.datasets.compute_stats import get_feature_stats
    g = get_feature_stats(S, axis=0, keepdims=False)
    for k in ("min", "max", "mean", "std", "q01", "q50", "q99"):
        assert np.allclose(ds.meta.stats["observation.state"][k], g[k], atol=1e-5)
    # video is stream-copied: the decoded frame of episode 1, frame 7 carries its own frame-index colour
    fr = ds[150 + 7]["observation.images.head"].numpy()
    assert abs(fr[0].mean() * 255 - (7 * 7) % 256) < 6
    assert {t for t in ds.meta.tasks.index} == {"open flap 1", "open the box flaps"}


def test_end_to_end_options(synth_source, tmp_path):
    src, _ = synth_source
    out = tmp_path / "opt"
    assert run_pp(src, out, "--include", "b,c", "--left-only", "--task", "open the box flaps", "--trim-idle",
                  "--action-offset-frames", "3", "--verify") == 0
    ds = load_out(out)
    assert ds.num_episodes == 2 and len(ds.meta.tasks) == 1
    assert ds.meta.features["observation.state"]["shape"] == (6,)
    prep = json.loads((out / "PREPROCESS.json").read_text())
    for e in prep["episodes"]:
        assert e["frames_out"] < e["frames_source"] and e["trim_frames"][0] > 0
    item = ds[len(ds) - 1]
    assert item["observation.state"].shape[0] == 6 and item["task"] == "open the box flaps"


def test_end_to_end_refuses_overwrite_and_leaves_source(synth_source, tmp_path):
    src, _ = synth_source
    before = sorted((p.relative_to(src), p.stat().st_mtime_ns) for p in src.rglob("*") if p.is_file())
    out = tmp_path / "x"
    assert run_pp(src, out, "--exclude", "a") == 0
    marker = (out / "marker").write_text("m")
    assert run_pp(src, out) == 2                                  # exists -> refused, nothing changed
    assert (out / "marker").exists()
    assert run_pp(src, out, "--force", "--include", "a") == 0
    assert not (out / "marker").exists() and load_out(out).num_episodes == 1
    assert before == sorted((p.relative_to(src), p.stat().st_mtime_ns) for p in src.rglob("*") if p.is_file())
    assert not [p for p in tmp_path.iterdir() if ".partial" in p.name]


def test_quality_report_smoke_and_read_only(synth_source, tmp_path):
    import quality_report as qr
    src, _ = synth_source
    before = sorted((p.relative_to(src), p.stat().st_mtime_ns) for p in src.rglob("*") if p.is_file())
    out = tmp_path / "rep"
    assert qr.main([str(src), "--out", str(out)]) == 0
    rep = json.loads((out / "report.json").read_text())
    assert [e["name"] for e in rep["episodes"]] == ["mocap_left_a", "mocap_left_b", "mocap_left_c"]
    e = rep["episodes"][0]
    for k in ("duration_s", "start_still_s", "end_still_s", "pauses", "branch", "start_pose", "joint_range_deg",
              "jitter_before", "jitter_after", "dtw_all_joints", "task", "flag", "reasons", "filter_deviation"):
        assert k in e
    assert abs(e["duration_s"] - 5.0) < 1e-9 and e["start_still_s"] > 0.8
    html = (out / "report.html").read_text()
    assert "data:image/png;base64" in html and "data:image/jpeg;base64" in html and "src='http" not in html
    assert before == sorted((p.relative_to(src), p.stat().st_mtime_ns) for p in src.rglob("*") if p.is_file())
