"""Tests for the pure helpers of to_turbovla.py (held-out choice, drop window shrinking).

    cd tools/mocap_dataset && python3 -m pytest -p no:anyio test/test_to_turbovla.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import to_turbovla as tt  # noqa: E402


def write_report(tmp_path, distances, medoid="mocap_left_new_10"):
    rows = [{"name": n, "dtw_all_joints": {"to_medoid_deg": d, "medoid": medoid}} for n, d in distances.items()]
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"episodes": rows}))
    return path


def test_heldout_is_the_middle_pair_without_the_medoid(tmp_path):
    d = {"mocap_left_new_10": 0.0, "mocap_left_new_3": 10.5, "mocap_left_new_8": 10.9, "mocap_left_new_5": 11.0,
         "mocap_left_new_4": 11.3, "mocap_left_new_2": 11.6, "mocap_left_new_6": 14.6, "mocap_left_new_7": 16.1,
         "mocap_left_new_1": 18.6, "mocap_left_13": 27.8}
    keep = [n for n in d if n != "mocap_left_13"]
    result = tt.choose_heldout(write_report(tmp_path, d), keep)
    assert result["validation"] == ["mocap_left_new_2", "mocap_left_new_4"]
    assert "mocap_left_new_10" in result["train"] and len(result["train"]) == 7


def test_heldout_requires_the_medoid_to_be_kept(tmp_path):
    d = {"mocap_left_new_10": 0.0, "mocap_left_new_1": 5.0, "mocap_left_new_2": 6.0}
    with pytest.raises(ValueError):
        tt.choose_heldout(write_report(tmp_path, d), ["mocap_left_new_1", "mocap_left_new_2"])


def test_canonical_and_natural_order():
    assert tt.canonical("new_3") == "mocap_left_new_3" == tt.canonical("mocap_left_new_3")
    assert sorted(["new_10", "new_2", "new_1"], key=tt.natural_key) == ["new_1", "new_2", "new_10"]


def test_shrink_drop_window_avoids_filtered_action_transitions():
    action = np.zeros((100, 14))
    action[:, 0] = 0.0
    action[70:, 0] = np.linspace(0, 1, 30)          # ramp starts at frame 70 (first transition 70->71)
    pause = {"drop_start_frame": 15, "drop_end_frame_exclusive": 75, "removed_frames": 60}
    out = tt.shrink_to_filtered_flat(pause, action)
    assert out["drop_start_frame"] == 15 and out["drop_end_frame_exclusive"] == 70
    assert out["drop_window_before_filter_flat_check"] == [15, 75]
    # no transition touches a dropped frame
    changed = np.abs(np.diff(action, axis=0)).max(axis=1) > tt.ACTION_EPS
    for i in range(out["drop_start_frame"], out["drop_end_frame_exclusive"]):
        assert not (changed[i] if i < len(changed) else False) and not (i > 0 and changed[i - 1])


def test_shrink_returns_none_when_every_frame_is_touched():
    action = np.cumsum(np.full((60, 14), 0.01), axis=0)
    assert tt.shrink_to_filtered_flat({"drop_start_frame": 15, "drop_end_frame_exclusive": 45}, action) is None


# ------------------------------------------------------------------------------------------ right arm
import shutil  # noqa: E402
import subprocess  # noqa: E402

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

needs_video = pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="ffmpeg missing")


def test_canonical_right_accepts_numbers_and_names():
    assert tt.canonical("5", "right") == "real_0005" == tt.canonical("real_0005", "right") == tt.canonical("0005", "right")
    assert tt.canonical("new_3") == "mocap_left_new_3"            # left unchanged
    # the 14-D right-arm mocap sets keep their own names (no real_ prefix in front)
    assert tt.canonical("mocap_right_2_ep008", "right") == "mocap_right_2_ep008"
    assert tt.canonical("mocap_right_farhan_ep004", "right") == "mocap_right_farhan_ep004"
    with pytest.raises(ValueError):
        tt.canonical("5", "middle")


def test_camera_map_defaults_and_validation():
    assert tt.parse_camera_map(None, "left") == {"head": "head", "knife": "knife", "wrist_left": "wrist_left"}
    assert tt.parse_camera_map(None, "right") == {"head": "head", "knife": "black", "wrist_left": "wrist_right"}
    assert tt.parse_camera_map("head=wrist_right, knife=black,wrist_left=head", "right")["head"] == "wrist_right"
    for bad in ("head=head,knife=black", "head=head,head=head,knife=black,wrist_left=x", "top=head,knife=a,wrist_left=b",
                "head=,knife=black,wrist_left=x"):
        with pytest.raises(ValueError):
            tt.parse_camera_map(bad, "right")


def test_expand_channels_places_the_arm_and_zeroes_the_rest():
    q = np.arange(1, 13, dtype=float).reshape(2, 6)
    right, left = tt.expand_channels(q, "right"), tt.expand_channels(q, "left")
    assert right.shape == (2, 14) and np.array_equal(right[:, 7:13], q)
    assert not right[:, :7].any() and not right[:, 13:].any()
    assert np.array_equal(left[:, 0:6], q) and not left[:, 6:].any()
    assert tt.CHANNELS[7] == "arm_right_joint1" and tt.CHANNELS[12] == "arm_right_joint6" and len(tt.CHANNELS) == 14
    with pytest.raises(ValueError):
        tt.expand_channels(np.zeros((3, 14)), "right")


def encode_episode(path, frames, color, size="64x48"):
    """h264 clip with a distinct moving pattern, GOP 2 like preprocess.py's output."""
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=s={size}:r=30:d={frames / 30}",
                    "-vf", f"hue=h={color}", "-frames:v", str(frames), "-c:v", "libx264", "-g", "2", "-bf", "0",
                    "-pix_fmt", "yuv420p", str(path)], check=True)


def pack(paths, out, tmp_path):
    listing = tmp_path / "list.txt"
    listing.write_text("".join(f"file '{p}'\n" for p in paths))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy",
                    str(out)], check=True)


@needs_video
def test_extract_segment_is_exact_and_black_video_has_the_right_length(tmp_path):
    a, b = tmp_path / "a.mp4", tmp_path / "b.mp4"
    encode_episode(a, 41, 0)                      # odd length: the next episode starts on an odd frame
    encode_episode(b, 30, 90)
    packed = tmp_path / "packed.mp4"
    pack([a, b], packed, tmp_path)
    cut = tt.extract_segment(packed, tmp_path / "out/b.mp4", 41, 30)
    assert tt.packet_count(cut) == 30
    black = tt.black_video(tmp_path / "out/black.mp4", 17, 64, 48)
    assert tt.packet_count(black) == 17
    frame = tt.decode_frames(black, [0])[0]
    assert frame.shape == (48, 64, 3) and frame.max() <= 20
    with pytest.raises(ValueError):                # a span that runs past the file is not accepted silently
        tt.extract_segment(packed, tmp_path / "out/bad.mp4", 41, 40)


def make_right_dataset(root, tmp_path, lengths=(40, 32)):
    """Tiny merged v3.0 right-arm dataset (6-D, three cameras packed into one file each)."""
    cams = ("head", "wrist_left", "wrist_right")
    names = [f"arm_right_joint{i}" for i in range(1, 7)]
    feature = lambda n: {"dtype": "float32", "shape": [n], "names": names if n == 6 else None, "fps": 30}  # noqa: E731
    video = {"dtype": "video", "shape": [48, 64, 3], "names": ["height", "width", "channel"], "info": {}}
    info = {"codebase_version": "v3.0", "fps": 30, "total_episodes": len(lengths), "features": {
        "observation.state": feature(6), "action": feature(6), "observation.state_raw": feature(6),
        "action_teleop": feature(6), "timestamp": feature(1), "frame_index": feature(1), "episode_index": feature(1),
        "index": feature(1), "task_index": feature(1), **{f"observation.images.{c}": dict(video) for c in cams}}}
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    (root / "meta/info.json").write_text(json.dumps(info))
    rng = np.random.default_rng(0)
    rows, sources, columns, start = [], [], {k: [] for k in ("state", "action", "raw", "teleop")}, 0
    for i, n in enumerate(lengths):
        q = np.cumsum(rng.normal(0, .01, (n, 6)), axis=0)
        for k, v in (("state", q), ("action", q + .01), ("raw", q), ("teleop", q)):
            columns[k].append(v)
        row = {"episode_index": i, "length": n, "dataset_from_index": start, "dataset_to_index": start + n}
        sources.append({"episode_index": i, "source": f"real_{i:04d}", "source_path": "/nonexistent"})
        rows.append(row)
        start += n
    for c, color in zip(cams, (0, 60, 120)):
        clips = []
        for i, n in enumerate(lengths):
            clips.append(tmp_path / f"{c}_{i}.mp4")
            encode_episode(clips[-1], n, color + 30 * i)
        pack(clips, root / f"videos/observation.images.{c}/chunk-000/file-000.mp4", tmp_path) if (
            (root / f"videos/observation.images.{c}/chunk-000").mkdir(parents=True) or True) else None
        t = 0
        for i, n in enumerate(lengths):
            rows[i].update({f"videos/observation.images.{c}/chunk_index": 0, f"videos/observation.images.{c}/file_index": 0,
                            f"videos/observation.images.{c}/from_timestamp": t / 30,
                            f"videos/observation.images.{c}/to_timestamp": (t + n) / 30})
            t += n
    total = sum(lengths)
    vec = lambda arr: pa.array(np.concatenate(arr).astype(np.float32).tolist(), type=pa.list_(pa.float32(), 6))  # noqa: E731
    pq.write_table(pa.table({
        "observation.state": vec(columns["state"]), "action": vec(columns["action"]),
        "observation.state_raw": vec(columns["raw"]), "action_teleop": vec(columns["teleop"]),
        "timestamp": pa.array(np.zeros(total), pa.float32()), "frame_index": pa.array(np.zeros(total), pa.int64()),
        "episode_index": pa.array(np.zeros(total), pa.int64()), "index": pa.array(np.zeros(total), pa.int64()),
        "task_index": pa.array(np.zeros(total), pa.int64())}), root / "data/chunk-000/file-000.parquet")
    pq.write_table(pa.Table.from_pylist(rows), root / "meta/episodes/chunk-000/file-000.parquet")
    (root / "meta/source_episodes.json").write_text(json.dumps(sources))
    return np.concatenate(columns["state"])


@needs_video
def test_stage_recordings_right_arm_14d_contract_and_camera_slots(tmp_path):
    src = tmp_path / "src"
    state = make_right_dataset(src, tmp_path)
    stage = tmp_path / "stage"
    rec = tt.stage_recordings(src, ["real_0000", "real_0001"], stage, "right", tt.parse_camera_map(None, "right"), "packed")
    assert rec["real_0001"]["frames"] == 32
    s1 = rec["real_0001"]["state"]
    assert s1.shape == (32, 14) and np.allclose(s1[:, 7:13], state[40:72], atol=1e-6)
    assert not s1[:, :7].any() and not s1[:, 13:].any()
    assert np.allclose(rec["real_0001"]["action"][:, 7:13] - s1[:, 7:13], .01, atol=1e-6)
    assert rec["real_0001"]["state_raw"].shape == (32, 14)
    ep = stage / "real_0001"
    info = json.loads((ep / "meta/info.json").read_text())
    assert info["features"]["action"]["names"] == tt.CHANNELS and info["features"]["observation.state"]["shape"] == [14]
    assert sorted(k for k in info["features"] if "images" in k) == [f"observation.images.{c}" for c in sorted(tt.CAMERAS)]
    for cam in tt.CAMERAS:
        video = ep / f"videos/observation.images.{cam}/chunk-000/file-000.mp4"
        assert tt.packet_count(video) == 32
    table = pq.read_table(ep / "data/chunk-000/file-000.parquet")
    assert table.schema.field("observation.state").type == pa.list_(pa.float32(), 14)
    # slot wrist_left carries the source wrist_right clip, the knife slot is black, head is head
    wl = tt.decode_frames(ep / "videos/observation.images.wrist_left/chunk-000/file-000.mp4", [0])[0]
    src_wr = tt.decode_frames(src / "videos/observation.images.wrist_right/chunk-000/file-000.mp4", [40])[40]
    assert np.array_equal(wl, src_wr)
    assert tt.decode_frames(ep / "videos/observation.images.knife/chunk-000/file-000.mp4", [0])[0].max() <= 20


def test_build_plan_reports_unapplied_pauses_when_reduction_is_off():
    class FakeTrim:
        @staticmethod
        def pause_candidates(state, action, fps):
            return [{"stationary_from_frame": 10, "stationary_to_frame_exclusive": 70, "drop_start_frame": 25,
                     "drop_end_frame_exclusive": 55, "removed_frames": 30}]

        @staticmethod
        def check_pause_images(video, pause, fps):
            return {"image_stability_passed": True}

    flat = np.zeros((100, 14))
    rec = {"real_0000": {"frames": 100, "dir": Path("/x"), "state_raw": flat, "action_teleop": flat, "action": flat,
                         "state": flat}}
    off = tt.build_plan(FakeTrim, rec, {"real_0000": "train"}, False, "task", report_pauses=True)["episodes"][0]
    assert off["pause_removals"] == [] and len(off["unapplied_pause_candidates"]) == 1
    nothing = tt.build_plan(FakeTrim, rec, {"real_0000": "train"}, False, "task")
    assert nothing["episodes"][0]["unapplied_pause_candidates"] == [] and nothing["canonical_instruction"] == "task"
    on = tt.build_plan(FakeTrim, rec, {"real_0000": "train"}, True, "task")["episodes"][0]
    assert len(on["pause_removals"]) == 1 and on["unapplied_pause_candidates"] == []


@needs_video
def test_stage_many_combines_sources_and_rejects_unknown_or_duplicate_names(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    (tmp_path / "a_tmp").mkdir(); (tmp_path / "b_tmp").mkdir()
    sa = make_right_dataset(a, tmp_path / "a_tmp")
    sb = make_right_dataset(b, tmp_path / "b_tmp", lengths=(36, 28))
    (b / "meta/source_episodes.json").write_text(json.dumps([
        {"episode_index": i, "source": f"mocap_right_9_ep{i:03d}", "source_path": "/nonexistent"} for i in range(2)]))
    names = ["mocap_right_9_ep001", "real_0000", "mocap_right_9_ep000"]
    rec, owner = tt.stage_many([a, b], names, tmp_path / "stage", "right", tt.parse_camera_map(None, "right"), "packed")
    assert list(rec) == names                                   # order of `names`, not of the sources
    assert owner == {"mocap_right_9_ep001": str(b), "real_0000": str(a), "mocap_right_9_ep000": str(b)}
    assert rec["real_0000"]["frames"] == 40 and rec["mocap_right_9_ep000"]["frames"] == 36 and rec["mocap_right_9_ep001"]["frames"] == 28
    assert np.allclose(rec["mocap_right_9_ep001"]["state"][:, 7:13], sb[36:64], atol=1e-6)
    assert np.allclose(rec["real_0000"]["state"][:, 7:13], sa[:40], atol=1e-6)
    with pytest.raises(ValueError, match="Not in any --src"):
        tt.stage_many([a, b], ["real_0007"], tmp_path / "stage2", "right", tt.parse_camera_map(None, "right"), "packed")
    (b / "meta/source_episodes.json").write_text(json.dumps([
        {"episode_index": i, "source": f"real_{i:04d}", "source_path": "/nonexistent"} for i in range(2)]))
    with pytest.raises(ValueError, match="is in both"):
        tt.stage_many([a, b], ["real_0000"], tmp_path / "stage3", "right", tt.parse_camera_map(None, "right"), "packed")


def test_episode_name_and_parse_src():
    assert tt.episode_name({"source": "mocap_right_2_ep001"}) == "mocap_right_2_ep001"                       # plain set
    assert tt.episode_name({"source": "mocap_right_2_ep001", "copy_index": 0}) == "mocap_right_2_ep001"      # original
    assert tt.episode_name({"source": "mocap_right_2_ep001", "copy_index": 3}) == "mocap_right_2_ep001__c3"
    assert tt.episode_name({"source": "x_ep001", "copy_index": 2}, "bg") == "x_ep001__bg2"
    assert tt.episode_name({"source": "x_ep001", "copy_index": 0}, "bg") == "x_ep001__bg0"
    assert tt.parse_src("sn=/media/farhan/My Passport/d") == ("sn", Path("/media/farhan/My Passport/d"))
    assert tt.parse_src("/a/b=c/d") == (None, Path("/a/b=c/d"))          # '=' inside a path is not a label
    assert tt.parse_src("real_right_clean_v1") == (None, Path("real_right_clean_v1"))


def test_build_plan_keeps_a_pause_whose_filtered_state_is_not_flat():
    class FakeTrim:
        @staticmethod
        def pause_candidates(state, action, fps):
            return [{"stationary_from_frame": 10, "stationary_to_frame_exclusive": 70, "drop_start_frame": 25,
                     "drop_end_frame_exclusive": 55, "removed_frames": 30}]

        @staticmethod
        def check_pause_images(video, pause, fps):
            return {"image_stability_passed": True}

    flat = np.zeros((100, 14))
    decaying = flat.copy()
    decaying[20:40, 7] = np.linspace(0.05, 0.0, 20)            # a state-noise offset decaying inside the stationary span
    ok = {"real_0000": {"frames": 100, "dir": Path("/x"), "state_raw": flat, "action_teleop": flat, "action": flat, "state": flat}}
    bad = {"real_0000": {**ok["real_0000"], "state": decaying}}
    assert len(tt.build_plan(FakeTrim, ok, {"real_0000": "train"}, True, "t")["episodes"][0]["pause_removals"]) == 1
    entry = tt.build_plan(FakeTrim, bad, {"real_0000": "train"}, True, "t")["episodes"][0]
    assert entry["pause_removals"] == [] and "not flat" in entry["rejected_pause_candidates"][0]["reason"]
    assert tt.filtered_span_ok({"stationary_from_frame": 10, "stationary_to_frame_exclusive": 70}, flat, flat)
    assert not tt.filtered_span_ok({"stationary_from_frame": 10, "stationary_to_frame_exclusive": 70}, decaying, flat)


@needs_video
def test_stage_many_with_labelled_augmented_sources(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    (tmp_path / "a_tmp").mkdir(); (tmp_path / "b_tmp").mkdir()
    make_right_dataset(a, tmp_path / "a_tmp", lengths=(40, 32))
    sb = make_right_dataset(b, tmp_path / "b_tmp", lengths=(40, 32, 40, 32))
    (a / "meta/source_episodes.json").write_text(json.dumps([
        {"episode_index": i, "source": f"mocap_right_9_ep{i:03d}", "source_path": "/x", "copy_index": 0} for i in range(2)]))
    (b / "meta/source_episodes.json").write_text(json.dumps([
        {"episode_index": i, "source": f"mocap_right_9_ep{i % 2:03d}", "source_path": "/x", "copy_index": i // 2}
        for i in range(4)]))
    names = ["mocap_right_9_ep001", "mocap_right_9_ep000__bg1", "mocap_right_9_ep001__bg1"]
    rec, owner = tt.stage_many([a, b], names, tmp_path / "stage", "right", tt.parse_camera_map(None, "right"), "packed",
                               {b: "bg"})
    assert owner == {"mocap_right_9_ep001": str(a), "mocap_right_9_ep000__bg1": str(b), "mocap_right_9_ep001__bg1": str(b)}
    assert rec["mocap_right_9_ep001"]["frames"] == 32 and rec["mocap_right_9_ep000__bg1"]["frames"] == 40
    assert np.allclose(rec["mocap_right_9_ep001__bg1"]["state"][:, 7:13], sb[112:144], atol=1e-6)
    assert (tmp_path / "stage/mocap_right_9_ep001__bg1/data/chunk-000/file-000.parquet").exists()
    with pytest.raises(ValueError, match="Not in any --src"):                  # a labelled set has no plain names
        tt.stage_many([a, b], ["mocap_right_9_ep000__c1"], tmp_path / "stage2", "right",
                      tt.parse_camera_map(None, "right"), "packed", {b: "bg"})
