"""Tests for tools/mocap_dataset/lerobot_to_plan.py (synthetic LeRobot v3.0 datasets, no real data needed).

    cd tools/mocap_dataset && python3 -m pytest -p no:anyio test/test_lerobot_to_plan.py
"""
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import lerobot_to_plan as L  # noqa: E402

PUBLISH_PLAN = HERE.parents[1] / "lid_open_friend" / "scripts" / "lid_open" / "publish_plan.py"
NAMES = [f"arm_{s}_{j}" for s in ("left", "right") for j in [f"joint{i}" for i in range(1, 7)] + ["gripper"]]
FPS = 30.0


def make_dataset(root: Path, left: np.ndarray, action: np.ndarray | None = None, right=None) -> Path:
    """Write a minimal one-episode LeRobot v3.0 dataset (state/action 14 floats, no videos)."""
    n = len(left)
    root.mkdir(parents=True, exist_ok=True)
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    info = {"codebase_version": "v3.0", "robot_type": "igus_rebel_dual_arm_real", "fps": FPS,
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "features": {"observation.state": {"dtype": "float32", "shape": [14], "names": NAMES},
                         "action": {"dtype": "float32", "shape": [14], "names": NAMES}}}
    (root / "meta" / "info.json").write_text(json.dumps(info))

    def full(arm_left):
        a = np.zeros((n, 14))
        a[:, :6] = arm_left
        if right is not None:
            a[:, 7:13] = right
        return a

    st, ac = full(left), full(left if action is None else action)
    pq.write_table(pa.table({"episode_index": [0], "data/chunk_index": [0], "data/file_index": [0],
                             "tasks": [["open flap"]]}), root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    pq.write_table(pa.table({
        "episode_index": np.zeros(n, dtype=np.int64), "frame_index": np.arange(n, dtype=np.int64),
        "timestamp": (np.arange(n) / FPS).astype(np.float32),
        "observation.state": [list(map(float, r)) for r in st], "action": [list(map(float, r)) for r in ac]}),
        root / "data" / "chunk-000" / "file-000.parquet")
    return root


def sweep(seconds: float, amp=(0.4, 0.3, 0.5, 0.6, 0.3, 0.8), base=(0.0, 0.2, 1.0, 0.0, 0.5, 0.0)) -> np.ndarray:
    """Smooth, slow multi-joint motion (peak ~0.1 rad/s) well inside the limits."""
    t = np.arange(int(seconds * FPS) + 1) / FPS
    return np.column_stack([b + a * np.sin(2 * math.pi * t / 40.0 + i) for i, (a, b) in enumerate(zip(amp, base))])


def run(root, *extra, out=None):
    argv = [str(root), *extra]
    if out is not None:
        argv += ["--out", str(out)]
    return L.main(argv)


@pytest.fixture(scope="module")
def player():
    spec = importlib.util.spec_from_file_location("publish_plan_vendored", PUBLISH_PLAN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- conversion
def test_convert_plan_fields_and_player_accepts_it(tmp_path, player, capsys):
    q = sweep(25.0)
    root = make_dataset(tmp_path / "ds", q)
    out = tmp_path / "plan.json"
    assert run(root, "--segment-s", "10", out=out) == 0
    doc = json.loads(out.read_text())
    task = doc["tasks"][0]
    # fields that publish_plan.main / build_targets read
    assert set(doc["box_pose"]) >= {"x", "y", "yaw_deg"} and "created" in doc
    for k in ("idx", "arm", "flap", "demo", "grab", "shift_mm", "replay", "joint_names", "start_q", "park_q", "steps"):
        assert k in task
    assert task["arm"] == "arm_left" and task["joint_names"] == [f"arm_left_joint{i}" for i in range(1, 7)]
    assert all(s["kind"] == "arc" and {"n", "label", "qs"} <= set(s) for s in task["steps"])
    assert [s["n"] for s in task["steps"]] == list(range(1, len(task["steps"]) + 1))
    # start_q is the first sample
    assert task["start_q"] == pytest.approx([round(float(v), 6) for v in q[0]])
    # the player's own checks pass (limits, per-step jump) and it makes: approach, one group per arc, park
    targets = player.build_targets(doc)
    start = dict(player.SAMPLE_POSE)
    errs, _ = player.check_targets(targets, 1.0, start, 60.0)
    assert errs == []
    groups, skipped = player.traj_groups(targets)
    assert skipped == 0 and len(groups) == 1 + len(task["steps"]) + 1
    assert len(groups[0][1]) == 1                                   # approach is its own single-point group
    assert [len(g[1]) for g in groups[1:-1]] == [len(s["qs"]) for s in task["steps"]]


def test_last_arc_ends_on_last_sample_and_park_is_noop(tmp_path, player):
    q = sweep(21.0)
    root = make_dataset(tmp_path / "ds", q)
    out = tmp_path / "plan.json"
    assert run(root, out=out) == 0
    doc = json.loads(out.read_text())
    task = doc["tasks"][0]
    assert task["steps"][-1]["qs"][-1] == pytest.approx([round(float(v), 6) for v in q[-1]])
    assert task["park_q"] == task["steps"][-1]["qs"][-1]
    # in the player the park group is a no-op move from the arrival point
    targets = player.build_targets(doc)
    assert targets[-1][2] == "approach"
    assert max(abs(a - b) for a, b in zip(targets[-1][1].values(), task["steps"][-1]["qs"][-1])) == 0.0
    # --no-park of the player drops it completely
    assert len(player.build_targets(doc, park=False)) == len(targets) - 1


def test_park_start_option(tmp_path):
    root = make_dataset(tmp_path / "ds", sweep(12.0))
    out = tmp_path / "plan.json"
    assert run(root, "--park", "start", out=out) == 0
    task = json.loads(out.read_text())["tasks"][0]
    assert task["park_q"] == task["start_q"] and task["park_q"] != task["steps"][-1]["qs"][-1]


def test_no_out_writes_nothing(tmp_path, capsys):
    root = make_dataset(tmp_path / "ds", sweep(8.0))
    assert run(root) == 0
    assert "nothing written" in capsys.readouterr().out
    assert not list(tmp_path.glob("*.json"))


def test_slice_and_source_action(tmp_path):
    q = sweep(30.0)
    act = q + 0.05                                                   # action ahead of state by a constant
    root = make_dataset(tmp_path / "ds", q, action=act)
    out = tmp_path / "plan.json"
    assert run(root, "--start-s", "5", "--end-s", "15", "--source", "action", out=out) == 0
    task = json.loads(out.read_text())["tasks"][0]
    i0, i1 = int(5 * FPS), int(15 * FPS)
    assert task["start_q"] == pytest.approx([round(float(v), 6) for v in act[i0]])
    assert task["steps"][-1]["qs"][-1] == pytest.approx([round(float(v), 6) for v in act[i1]])
    assert "[action] 5.0-15.0s" in task["demo"]


# ---------------------------------------------------------------- segmenting / downsampling
def test_split_segments_by_data_time():
    t = np.arange(0, 25.01, 0.2)
    segs = L.split_segments(t, 10.0)
    assert len(segs) == 3 and segs[0][0] == 0 and segs[-1][1] == len(t)
    assert all(a[1] == b[0] for a, b in zip(segs, segs[1:]))        # contiguous
    # a short tail (< 25 % of the segment) is merged into the previous chunk
    t2 = np.arange(0, 22.01, 0.2)
    assert len(L.split_segments(t2, 10.0)) == 2
    assert L.split_segments(t, 0.0) == [(0, len(t))]


def test_segments_in_plan(tmp_path):
    root = make_dataset(tmp_path / "ds", sweep(25.0))
    out = tmp_path / "plan.json"
    assert run(root, "--segment-s", "10", out=out) == 0
    steps = json.loads(out.read_text())["tasks"][0]["steps"]
    assert len(steps) == 3
    pts = [p for s in steps for p in s["qs"]]
    assert len(pts) == len({tuple(p) for p in pts})                  # no duplicated boundary point (it would add a 1 s hold)
    assert "seg 1/3" in steps[0]["label"] and "seg 3/3" in steps[2]["label"]
    run_one = tmp_path / "one.json"
    assert run(root, "--segment-s", "0", out=run_one) == 0
    assert len(json.loads(run_one.read_text())["tasks"][0]["steps"]) == 1


def test_downsample_drops_dwell_keeps_ends():
    t = np.arange(0, 20, 1 / FPS)
    q = np.zeros((len(t), 6))
    q[:, 0] = 0.5 + 0.0 * t
    q[t > 10, 0] = 0.5 + 0.3 * (t[t > 10] - 10) / 10                # still for 10 s, then a ramp
    keep = L.downsample(t, q, 5.0, math.radians(1.0))
    assert keep[0] == 0 and keep[-1] == len(t) - 1
    dwell = [i for i in keep if t[i] < 10]
    assert len(dwell) <= 2                                           # the 10 s dwell is not 50 points
    assert L.shape_error(t, q, keep)[0] < math.radians(1.0)
    # every kept step is big enough for the player to move at max_speed, except possibly the last one
    steps = np.abs(np.diff(q[keep], axis=0)).max(1)
    assert (steps[:-1] >= math.radians(1.0) - 1e-9).all()


def test_shape_error_small_on_smooth_path():
    q = sweep(40.0)
    t = np.arange(len(q)) / FPS
    keep = L.downsample(t, q, 5.0, math.radians(1.0))
    assert L.shape_error(t, q, keep)[0] < math.radians(1.5)


def test_player_times_matches_vendored_player(player):
    rng = np.random.default_rng(0)
    prev = rng.uniform(-1, 1, 6)
    qs = [list(prev + rng.uniform(-0.2, 0.2, 6) * k) for k in range(1, 12)]
    qs += [qs[-1]] * 3                                               # dwell points hit the 0.2 s floor
    assert L.player_times(prev, qs, 0.15) == pytest.approx(player.traj_times(prev, qs, 0.15))


# ---------------------------------------------------------------- fail closed
def refused(tmp_path, left, *extra, **ds):
    root = make_dataset(tmp_path / "ds", left, **ds)
    out = tmp_path / "plan.json"
    rc = run(root, *extra, out=out)
    assert not out.exists(), "a refused conversion must not write a plan"
    return rc


def test_refuses_all_zero_arm(tmp_path, capsys):
    root = make_dataset(tmp_path / "ds", sweep(8.0))
    out = tmp_path / "plan.json"
    assert run(root, "--arm", "arm_right", out=out) == 2
    assert "all zeros" in capsys.readouterr().err and not out.exists()


def test_right_arm_ok_when_recorded(tmp_path):
    q = sweep(8.0)
    root = make_dataset(tmp_path / "ds", q, right=q)
    out = tmp_path / "plan.json"
    assert run(root, "--arm", "arm_right", out=out) == 0
    assert json.loads(out.read_text())["tasks"][0]["arm"] == "arm_right"


@pytest.mark.parametrize("joint,value", [(0, 3.10), (1, -1.37), (2, 2.42), (3, -3.10), (4, 1.63), (5, 3.1)])
def test_refuses_near_limit_values(tmp_path, capsys, joint, value):
    q = sweep(8.0)
    q[100:110, joint] = value                                        # inside the URDF limit but within the 2 deg margin
    assert refused(tmp_path, q) == 2
    assert f"joint{joint + 1}" in capsys.readouterr().err


def test_limit_margin_option(tmp_path):
    q = sweep(8.0, amp=(0.4, 0.3, 0.5, 0.6, 0.0, 0.8), base=(0.0, 0.2, 1.0, 0.0, 1.63, 0.0))   # j5 constant, limit 1.6406
    assert refused(tmp_path, q) == 2
    out = tmp_path / "ok.json"
    assert run(tmp_path / "ds", "--limit-margin-deg", "0.1", out=out) == 0


def test_refuses_jump_and_wrap(tmp_path, capsys):
    q = sweep(8.0)
    q[120:, 3] += 2 * math.pi                                        # a +-pi wrap in the recording (also beyond the limit)
    assert refused(tmp_path, q) == 2
    q2 = sweep(8.0)
    q2[120:, 0] += 0.3                                               # a 17 deg glitch inside the limits
    assert refused(tmp_path / "b", q2) == 2
    assert "jump" in capsys.readouterr().err


def test_refuses_non_finite(tmp_path, capsys):
    q = sweep(8.0)
    q[50, 2] = float("nan")
    assert refused(tmp_path, q) == 2
    assert "non-finite" in capsys.readouterr().err


def test_check_series_direct():
    lim = L.urdf_limits()
    assert lim.shape == (6, 2) and lim[0, 1] == pytest.approx(3.106686) and lim[4, 1] == pytest.approx(1.640609)
    assert lim[1, 0] == pytest.approx(-1.378810) and lim[2, 1] == pytest.approx(2.426008)
    t = np.arange(10) / FPS
    q = np.tile(np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]), (10, 1))
    L.check_series(t, q, lim, 0.03, 0.1)                              # ok
    with pytest.raises(L.PlanError, match="at least 2"):
        L.check_series(t[:1], q[:1], lim, 0.03, 0.1)
    bad_t = t.copy()
    bad_t[5] = bad_t[4]
    with pytest.raises(L.PlanError, match="increasing"):
        L.check_series(bad_t, q, lim, 0.03, 0.1)


def test_fk_z_summary_and_unavailable_urdf(tmp_path, capsys):
    root = make_dataset(tmp_path / "ds", sweep(8.0))
    assert run(root) == 0
    assert "link6 z in base_link: min +" in capsys.readouterr().out
    assert run(root, "--urdf", str(tmp_path / "missing.urdf")) == 0     # FK is informational, never blocks the conversion
    assert "unavailable" in capsys.readouterr().out
