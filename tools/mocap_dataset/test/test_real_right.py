"""Tests for the v2.1 path (lerobot_episode), the generalised arm layout, pause trimming (episode_metrics.plan_pause_trim,
preprocess.py --trim-pauses with video re-encoding) and the quality report on a many-episode v2.1 dataset.

    cd tools/mocap_dataset && python3 -m pytest -p no:anyio test/test_real_right.py

The synthetic v2.1 source mimics ``real_right``: ONE dataset, 6-D ``arm_right_joint1..6`` state, ``action[t] == state[t+1]``,
one H.264 mp4 per episode and camera whose colour encodes the frame index, ``meta/episodes.jsonl`` with a ``box_pose``.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import episode_metrics as em  # noqa: E402
import lerobot_episode as le  # noqa: E402
import preprocess as pp  # noqa: E402
from test_preprocess import FPS, NAMES, make_source, load_out  # noqa: E402

ARM = [f"arm_right_joint{i}" for i in range(1, 7)]
COLOR_STEP = 11                      # blue/red level step per frame in the synthetic videos


def pulse_motion(n: int, amp: float = 0.25, a1=(20, 80), a2=None, seed: int = 0) -> np.ndarray:
    """(n, 6): two smooth sin^2 velocity pulses with an exactly still stretch in between (a mid-episode pause)."""
    a2 = a2 or (n - 80, n - 20)
    v = np.zeros(n)
    for a, b in (a1, a2):
        t = np.arange(b - a)
        v[a:b] = np.sin(np.pi * t / (b - a)) ** 2
    scale = amp * 2 / ((a1[1] - a1[0]) / FPS)           # displacement amp per pulse
    q1 = np.cumsum(v) / FPS * scale
    gains = np.array([1.0, 0.8, -0.6, 0.5, -0.4, 0.7]) * (1 + 0.05 * seed)
    return q1[:, None] * gains[None, :] + np.array([0.1, 0.3, 0.8, 0.0, -0.4, 0.0])


def _encode_idx_video(path: Path, n: int, seed: int, size=(128, 96)) -> None:
    import av
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    with av.open(str(path), "w") as c:
        s = c.add_stream("libx264", rate=30)
        s.width, s.height, s.pix_fmt = size[0], size[1], "yuv420p"
        s.options = {"g": "10", "crf": "12"}
        for i in range(n):
            img = np.zeros((size[1], size[0], 3), np.uint8)
            img[:, :, 0] = (i * COLOR_STEP) % 256                 # the frame index is encoded in the colour
            img[:, :, 2] = (i * COLOR_STEP * 3) % 256
            img[:, :, 1] = (seed * 40) % 256
            img += rng.integers(0, 2, img.shape, dtype=np.uint8)
            for p in s.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                c.mux(p)
        for p in s.encode():
            c.mux(p)


def make_v21(root: Path, lengths, box_poses, task="open the front flap of the cardboard box"):
    """Synthetic LeRobot v2.1 dataset (GR00T layout) with one 6-D arm; returns the list of (n, 6) float32 states."""
    pytest.importorskip("av")
    vkeys = ["observation.images.head", "observation.images.wrist_right"]
    feats = {"observation.state": {"dtype": "float32", "shape": [6], "names": ARM},
             "action": {"dtype": "float32", "shape": [6], "names": ARM}}
    for k, dt in (("timestamp", "float32"), ("frame_index", "int64"), ("episode_index", "int64"), ("index", "int64"),
                  ("task_index", "int64"), ("annotation.human.task_description", "int64"), ("next.reward", "float32"),
                  ("next.done", "bool")):
        feats[k] = {"dtype": dt, "shape": [1], "names": None}
    for k in vkeys:
        feats[k] = {"dtype": "video", "shape": [96, 128, 3], "names": ["height", "width", "channel"],
                    "info": {"video.fps": 30.0, "video.height": 96, "video.width": 128, "video.channels": 3,
                             "video.codec": "h264", "video.pix_fmt": "yuv420p", "video.is_depth_map": False, "has_audio": False}}
    info = {"codebase_version": "v2.1", "robot_type": "igus_rebel_dual_arm_rig_arm_right", "total_episodes": len(lengths),
            "total_frames": int(sum(lengths)), "total_tasks": 1, "total_videos": 2 * len(lengths), "total_chunks": 1,
            "chunks_size": 1000, "fps": 30, "splits": {"train": f"0:{len(lengths)}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4", "features": feats}
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps(info))
    (root / "meta" / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": task}) + "\n")
    rows, states = [], []
    for i, (n, bp) in enumerate(zip(lengths, box_poses)):
        q = pulse_motion(n, seed=i).astype(np.float32)
        states.append(q)
        act = np.vstack([q[1:], q[-1:]])                      # action[t] == state[t+1]
        done = np.zeros(n, bool)
        done[-1] = True
        t = pa.table({"observation.state": pa.array(q.tolist(), pa.list_(pa.float32())),
                      "action": pa.array(act.tolist(), pa.list_(pa.float32())),
                      "timestamp": pa.array((np.arange(n) / 30).astype(np.float32)), "frame_index": np.arange(n),
                      "episode_index": np.full(n, i), "index": np.arange(n) + sum(lengths[:i]), "task_index": np.zeros(n, np.int64),
                      "annotation.human.task_description": np.zeros(n, np.int64),
                      "next.reward": pa.array(done.astype(np.float32)), "next.done": pa.array(done)})
        (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
        pq.write_table(t, root / "data" / "chunk-000" / f"episode_{i:06d}.parquet")
        for k in vkeys:
            _encode_idx_video(root / "videos" / "chunk-000" / k / f"episode_{i:06d}.mp4", n, seed=i)
        rows.append({"episode_index": i, "tasks": [task], "length": n, "flap": "LidFront", "source": f"real_{i:04d}",
                     "box_pose": list(bp), "kind": "real", "rig": "real_v3"})
    (root / "meta" / "episodes.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return states


@pytest.fixture(scope="module")
def v21(tmp_path_factory):
    pytest.importorskip("lerobot")
    root = tmp_path_factory.mktemp("v21") / "real_right"
    lengths = [300, 330, 360, 300, 340, 320]
    box = [(0.0, -0.59, 85.0), (0.05, -0.59, 80.0), (0.1, -0.60, 88.0), (0.1, -0.60, 88.0), (0.02, -0.58, 95.0), (0.06, -0.60, 90.0)]
    states = make_v21(root, lengths, box)
    return root, states, lengths, box


# ------------------------------------------------------------------------------------------ v2.1 loader
def test_v21_loader_matches_files(v21):
    root, states, lengths, box = v21
    refs = le.list_episodes(root)
    assert [r.name for r in refs] == [f"real_{i:04d}" for i in range(6)]
    assert refs[3].aliases == ("3", "episode_000003", "ep3", "ep003") and all(r.n_in_dataset == 6 for r in refs)
    ep = le.load_episode(root, 2)
    assert ep.version == "v2.1" and ep.name == "real_0002" and ep.names == ARM and ep.arms == ["arm_right"]
    assert ep.active_arm() == "arm_right"
    assert np.array_equal(ep.state, states[2].astype(np.float64)) and np.allclose(ep.action[:-1], ep.state[1:])
    assert ep.meta["box_pose"] == [0.1, -0.6, 88.0] and ep.meta["source"] == "real_0002"
    assert set(ep.extras) >= {"next.reward", "next.done", "annotation.human.task_description"} and ep.extras["next.done"][-1]
    path, from_ts = ep.videos["observation.images.head"]
    assert path.name == "episode_000002.mp4" and from_ts == 0.0 and path.is_file()
    assert np.allclose(ep.timestamp, np.arange(lengths[2]) / 30, atol=1e-6)
    with pytest.raises(ValueError, match="no episode 9"):
        le.load_episode(root, 9)


def test_v21_check_loader_and_videos(v21):
    from verify_output import check_loader
    assert check_loader(v21[0])


def test_arm_detection_14d_right_arm(tmp_path):
    """The generalised layout: a 14-D dataset whose RIGHT arm moves is detected as arm_right (left stays zero)."""
    pytest.importorskip("lerobot")
    (tmp_path / "mocap_left_r").mkdir()
    make_source(tmp_path / "mocap_left_r", 150, 1, "t", arm="right")
    ep = le.load_episode(tmp_path / "mocap_left_r")
    assert ep.arms == ["arm_left", "arm_right"] and ep.active_arm() == "arm_right" and ep.name == "mocap_left_r"
    p = pp.process_episode(ep, sigma_s=0.08, action_offset=1, trim_idle=False, trim_margin_s=0.5, arm="arm_right", arm_only=True)
    assert p["state"].shape == (150, 6) and np.allclose(p["state_raw"], ep.arm("arm_right"), atol=1e-6)
    assert np.allclose(p["action"][:-1], p["state"][1:])
    out = tmp_path / "out"
    assert pp.main([str(tmp_path), "--out", str(out), "--arm", "arm_right", "--arm-only", "--verify"]) == 0
    ds = load_out(out)
    assert ds.meta.features["observation.state"]["names"] == ARM
    prep = json.loads((out / "PREPROCESS.json").read_text())
    assert prep["parameters"]["arm"] == "arm_right" and prep["parameters"]["video_mode"] == "copy"


# ------------------------------------------------------------------------------------------ pause trimming
def test_plan_pause_trim_cuts_only_settled_frames():
    n = 300
    q = pulse_motion(n)
    qf = em.smooth(q, 0.08, FPS)
    plan = em.plan_pause_trim(qf, FPS)
    keep = plan["keep"]
    assert len(plan["cuts"]) == 1 and not plan["skipped"] and np.all(np.diff(keep) > 0)
    c = plan["cuts"][0]
    assert 78 <= c["removed_from"] <= 100 and 200 <= c["removed_to"] <= 220 and c["removed_frames"] == plan["removed"] > 90
    assert plan["head_frames"] + plan["tail_frames"] == 12                       # 0.4 s kept of the settled stretch
    sp = em.arm_speed(qf, FPS)
    assert (sp[np.setdiff1d(np.arange(n), keep)] <= em.PAUSE_CORE_SPEED).all()    # nothing that moves is removed
    assert c["join_step_deg"] <= em.PAUSE_MAX_JOIN_DEG
    assert np.degrees(np.abs(np.diff(qf[keep], axis=0)).max()) <= np.degrees(np.abs(np.diff(qf, axis=0)).max()) + 1e-9
    assert em.kept_ranges(keep) == [[0, c["removed_from"]], [c["removed_to"], n]]


def test_plan_pause_trim_respects_thresholds():
    n = 300
    qf = em.smooth(pulse_motion(n), 0.08, FPS)
    assert em.plan_pause_trim(qf, FPS, min_pause_s=6.0)["removed"] == 0                  # pause (4.7 s) shorter than min
    assert em.plan_pause_trim(qf, FPS, keep_s=10.0, min_pause_s=11.0)["removed"] == 0
    slow = qf.copy()
    slow[100:220] += np.linspace(0, 0.0035, 120)[:, None]                                 # 0.2 deg creep: still cut
    assert em.plan_pause_trim(slow, FPS)["removed"] > 90
    creep = qf.copy()
    creep[100:220] += np.linspace(0, 0.01, 120)[:, None]                                  # 0.57 deg across the pause: not cut
    plan = em.plan_pause_trim(creep, FPS)
    assert plan["removed"] == 0 and plan["skipped"] and "drifts" in plan["skipped"][0]["why"]
    assert em.plan_pause_trim(creep, FPS, max_join_deg=1.0)["removed"] > 90
    # an episode without any pause is untouched
    moving = np.column_stack([np.sin(np.arange(n) / 30.0)] * 6)
    assert em.plan_pause_trim(em.smooth(moving, 0.08, FPS), FPS)["removed"] == 0


def test_helpers_corruption_limits_clusters():
    q = np.zeros((10, 6))
    q[5:, 2] = np.radians(10)                                  # a 10 deg step in one frame
    ev = em.speed_outliers(q, 30.0, 5.7)
    assert ev["n_events"] == 1 and ev["events"][0]["joint"] == 3 and abs(ev["max_rad_s"] - np.radians(10) * 30) < 1e-6
    assert em.zero_pose_frames(q) == 5 and em.zero_pose_frames(q + 0.1) == 0
    lim = np.array([[-1.0, 1.0]] * 6)
    m = em.joint_limit_margin_deg(np.full((3, 6), 0.9), lim)
    assert abs(m["min_deg"] - np.degrees(0.1)) < 1e-9 and m["side"] == "upper"
    assert em.cluster_poses(np.array([[0] * 6, [1] * 6, [30] * 6, [31] * 6, [0.5] * 6]), 3.0) == [0, 0, 1, 1, 0]
    assert em.start_pose_distance(np.radians(em.START_POSE_DEG))["max_abs_deg"] < 1e-9
    assert abs(em.start_pose_distance(np.zeros(6), np.array([3.0, 0, 0, 0, 0, 0]))["max_abs_deg"] - 3.0) < 1e-9
    assert em.arm_columns(NAMES, "arm_right") == list(range(7, 13))


def test_video_state_consistency_detects_lag():
    rng = np.random.default_rng(0)
    sp = np.convolve(np.abs(rng.normal(size=600)), np.ones(15) / 15, "same")
    vid = np.roll(sp, 6) + 0.01 * rng.normal(size=600)                   # the video follows the state by 6 frames
    c = em.video_state_consistency(vid, sp, 30.0)
    assert abs(c["best_lag_frames"] - 6) <= 1 and c["corr_best"] > 0.95
    assert em.video_state_consistency(rng.normal(size=600), sp, 30.0)["corr_best"] < 0.4


def _frame_colour(ds, row: int, key: str) -> float:
    return float(ds[row][key].numpy()[0].mean() * 255)          # channel 0 carries (source_frame * 11) % 256


def test_v21_end_to_end_trim_pauses(v21, tmp_path):
    root, states, lengths, box = v21
    out = tmp_path / "clean"
    assert pp.main([str(root), "--out", str(out), "--exclude", "5", "--trim-pauses", "--workers", "2", "--verify"]) == 0
    prep = json.loads((out / "PREPROCESS.json").read_text())
    ds = load_out(out)
    assert ds.num_episodes == 5 and prep["parameters"]["video_mode"] == "reencode" and prep["parameters"]["source_version"] == "v2.1"
    assert prep["dropped_source_columns"] == ["annotation.human.task_description"]
    assert [e["source"] for e in prep["episodes"]] == [f"real_{i:04d}" for i in range(5)]
    info = json.loads((out / "info.json").read_text()) if (out / "info.json").exists() else json.loads((out / "meta" / "info.json").read_text())
    assert info["codebase_version"] == "v3.0" and info["features"]["observation.state"]["shape"] == [6]
    assert "annotation.human.task_description" not in info["features"] and "source_frame_index" in info["features"]
    st = ds.hf_dataset.with_format("numpy")
    S, A = np.asarray(st["observation.state"]), np.asarray(st["action"])
    R, T = np.asarray(st["observation.state_raw"]), np.asarray(st["action_teleop"])
    SI, TS = np.asarray(st["source_frame_index"]), np.asarray(st["timestamp"])
    removed_total = 0
    for e in prep["episodes"]:
        i = e["episode_index"]
        a, n = int(ds.meta.episodes[i]["dataset_from_index"]), e["frames_out"]
        sl = slice(a, a + n)
        keep = SI[sl]
        assert n == lengths[i] - e["pause_trim"]["removed_frames"] and e["pause_trim"]["removed_frames"] > 90
        removed_total += e["pause_trim"]["removed_frames"]
        assert np.array_equal(R[sl], states[i][keep]) and np.array_equal(T[sl], np.vstack([states[i][1:], states[i][-1:]])[keep])
        assert np.allclose(TS[sl], np.arange(n) / 30, atol=1e-6)                                  # contiguous at 30 fps
        assert np.array_equal(np.asarray(st["frame_index"])[sl], np.arange(n))
        assert np.allclose(A[sl][:-1], S[sl][1:]) and np.array_equal(A[sl][-1], S[sl][-1])         # action = next KEPT state
        gap = int(np.flatnonzero(np.diff(keep) != 1)[0])                                           # the cut
        assert keep[gap + 1] - keep[gap] > 90 and np.degrees(np.abs(S[sl][gap + 1] - S[sl][gap]).max()) <= em.PAUSE_MAX_JOIN_DEG
        assert np.array_equal(np.asarray(st["next.done"])[sl].nonzero()[0], [n - 1])                # done stays on the last frame
        assert np.allclose(ds.meta.episodes[i]["box_pose"], box[i])
        for k in (0, gap, gap + 1, n - 1):                                                         # decoded frame == source frame
            for key in ("observation.images.head", "observation.images.wrist_right"):
                got = _frame_colour(ds, a + k, key)
                want = (int(keep[k]) * COLOR_STEP) % 256
                assert min(abs(got - want), 256 - abs(got - want)) < 5, (i, k, key, got, want)
    assert removed_total > 0
    side = json.loads((out / "meta" / "source_episodes.json").read_text())
    assert side[4]["source_meta"]["box_pose"] == list(box[4]) and side[4]["keep_ranges"] == prep["episodes"][4]["keep_ranges"]
    # stats exist for the re-encoded images (sampled from the source frames) and the new columns
    s = ds.meta.stats
    assert np.asarray(s["observation.images.head"]["mean"]).shape == (3, 1, 1) and "source_frame_index" in s
    # the source is untouched and a rerun without --force is refused
    assert pp.main([str(root), "--out", str(out)]) == 2


def test_v21_without_trim_is_contiguous_and_reencoded(v21, tmp_path):
    root, states, lengths, box = v21
    out = tmp_path / "plain"
    assert pp.main([str(root), "--out", str(out), "--include", "0,3", "--sigma-s", "0", "--workers", "2", "--verify"]) == 0
    ds = load_out(out)
    assert ds.num_episodes == 2 and len(ds) == lengths[0] + lengths[3]
    st = ds.hf_dataset.with_format("numpy")
    S, R = np.asarray(st["observation.state"]), np.asarray(st["observation.state_raw"])
    assert np.array_equal(S, R)                                                  # sigma 0 = measured state
    assert np.array_equal(np.asarray(st["source_frame_index"])[:lengths[0]], np.arange(lengths[0]))
    prep = json.loads((out / "PREPROCESS.json").read_text())
    assert prep["parameters"]["video_mode"] == "reencode" and prep["video_encode"]["codec"] == "libx264"
    assert abs(_frame_colour(ds, 7, "observation.images.head") - (7 * COLOR_STEP) % 256) < 5


def test_select_numeric_alias_only_for_multi_episode(v21, tmp_path):
    refs = le.list_episodes(v21[0])
    assert [r.name for r in pp.select_datasets(refs, None, "5,episode_000001")] == ["real_0000", "real_0002", "real_0003", "real_0004"]
    assert [r.name for r in pp.select_datasets(refs, "real_0004,2", None)] == ["real_0002", "real_0004"]
    with pytest.raises(pp.PreprocessError):
        pp.select_datasets(refs, "9", None)


# ------------------------------------------------------------------------------------------ report / viewer
def test_quality_report_v21_box_pose_and_holdout(v21, tmp_path):
    import quality_report as qr
    root = v21[0]
    before = sorted((p.relative_to(root), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file())
    out = tmp_path / "rep"
    assert qr.main([str(root), "--out", str(out), "--arm", "arm_right", "--rule", "end_pose_review_deg=50"]) == 0
    rep = json.loads((out / "report.json").read_text())
    assert rep["arm"] == "arm_right" and [e["name"] for e in rep["episodes"]] == [f"real_{i:04d}" for i in range(6)]
    e = rep["episodes"][3]
    assert e["box_pose"] == [0.1, -0.6, 88.0] and e["box_duplicates"] == ["real_0002"]
    assert e["pause_trim_plan"]["removed_frames"] > 90 and e["pauses"] and e["pause_share"] > 0.3
    assert {"start_variant", "end_variant", "limit_margin", "corruption", "stale", "still_frames_raw_share"} <= set(e)
    assert rep["flag_rules"]["end_pose_review_deg"] == 50
    hold = rep["holdout"]
    assert len(hold["chosen"]) == 2 and not ({"real_0002", "real_0003"} & set(hold["chosen"]))        # duplicate-pose pair never held out
    assert (out / "box_pose.png").is_file() and "Suggested held-out pair" in (out / "report.html").read_text()
    assert before == sorted((p.relative_to(root), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file())


def test_quality_report_flags_corrupt_episode(tmp_path):
    """A state that starts at all-zeros and jumps 10 deg in one frame is flagged CORRUPT (impossible step, zero frames)."""
    pytest.importorskip("lerobot")
    import quality_report as qr
    root = tmp_path / "ds"
    make_v21(root, [300, 300, 300, 300, 300], [(0.0, -0.6, 85.0 + i) for i in range(5)])
    f = root / "data" / "chunk-000" / "episode_000004.parquet"
    t = pq.read_table(f).to_pandas()
    q = np.stack(t["observation.state"].to_numpy()).copy()
    q[:10] = 0.0
    q[150:] += np.radians(10.0)
    t["observation.state"] = [list(r) for r in q.astype(np.float32)]
    pq.write_table(pa.Table.from_pandas(t, preserve_index=False), f)
    out = tmp_path / "rep"
    assert qr.main([str(root), "--out", str(out), "--arm", "arm_right", "--no-thumbs"]) == 0
    rep = json.loads((out / "report.json").read_text())
    bad = rep["episodes"][4]
    assert bad["flag"] == "review" and bad["corruption"]["n_events"] >= 1 and bad["corruption"]["zero_pose_frames"] == 10
    assert any("CORRUPT" in w for w in bad["reasons"])
    assert not any("CORRUPT" in w for e in rep["episodes"][:4] for w in e["reasons"])


def test_rerun_view_right_arm_save(v21, tmp_path):
    import rerun_view as rv
    out = tmp_path / "x.rrd"
    info = rv.view_episode(v21[0], 1, "both", str(out), no_video=True, no_robot=True)
    assert out.is_file() and out.stat().st_size > 1000 and "arm_right_joint1" in info["channels"]
    assert rv.VIDEO_ORDER[:3] == ["head", "wrist_left", "wrist_right"]
