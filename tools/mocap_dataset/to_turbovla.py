#!/usr/bin/env python3
"""Adapter: merged LeRobot v3.0 dataset (preprocess.py output) -> Compal TurboVLA pause-compressed bundle.

The Compal/TurboVLA handoff builds its training data in three stages (all code is theirs, none is
rewritten here):

  1. scripts/compal/trim_mocap_dataset.py   build()     : single-episode v3.0 recordings + pause plan
                                                           -> "pause-compressed bundle" (manifest.json,
                                                           normalization.json, per-episode parquet + h264 mp4)
  2. scripts/compal/mocap.py prepare  (experiments/compal/mocap_dataset.py import_bundle)
                                                         : bundle -> LeRobot v2.1 train/validation datasets
  3. scripts/compal/mocap.py images   (experiments/compal/mocap_images.py)
                                                         : MP4-backed datasets -> inline 224x224 PNG datasets

This script only does the glue in front of stage 1:

  * `heldout`: reproducible choice of the validation episodes from quality_report.py's report.json
    (the episodes whose DTW distance to the medoid is the median of the candidates, medoid excluded).
  * `bundle`: stage our merged dataset as single-episode v3.0 recordings (filtered observation.state and
    action = next filtered state; the 3 original AV1 videos are the untouched per-episode source files that
    preprocess.py stream-copied), find stagnant spans with THEIR pause_candidates()/check_pause_images()
    (same rule as the handoff: teleop command flat >= 1.5 s, raw state settled, 0.5 s guards, head-camera ROI
    stable), then call THEIR build().  Normalization is theirs too (action min/max over ALL kept episodes,
    state mean/std over the TRAIN episodes after pause removal).

Stages 2 and 3 are run afterwards with the handoff's mocap.py (see the commands printed at the end), with a
source overlay whose dataset ids / episode counts / task string are set by the env vars COMPAL_MOCAP_DATASET_STEM,
COMPAL_MOCAP_EPISODES and COMPAL_MOCAP_TASK (defaults: the clean9 values).  Never run this with a dataset directory as
the working directory.

Left arm (default, unchanged): the source is a 14-D dataset, the three original per-episode videos are linked as they
are.  Right arm (`--arm right`, e.g. real_right_clean_v1): the source has 6-D state/action; they are placed in the
`arm_right_joint1..6` channels 7..12 of the 14-D contract, all other channels (left arm, both grippers) are 0.  The model's
three camera slots (head, knife, wrist_left) are filled from `--camera-map` (default for the right arm:
head=head, knife=black, wrist_left=wrist_right, i.e. the right gripper camera feeds the `wrist_left` slot and the
knife slot is a black video).  The right-arm source packs several episodes per mp4 (`--videos packed`): each
episode's frames are cut out by stream copy (no re-encode) and verified.  The mapping is written into the bundle
manifest (`camera_mapping`, `channel_placement`) and carried by the importer into import-manifest.json / split.json.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

FPS = 30
INSTRUCTION = "Open the box flaps."
STATE_KEY, ACTION_KEY = "observation.state", "action"
CAMERAS = ("head", "knife", "wrist_left")          # the model's camera slots (mocap_dataset.CAMERAS)
CHANNELS = [f"arm_{side}_{joint}" for side in ("left", "right")
            for joint in [*(f"joint{i}" for i in range(1, 7)), "gripper"]]
ARM_FIRST = {"left": 0, "right": 7}                 # first of the six joint channels of each arm
NAME_PREFIX = {"left": "mocap_left_", "right": "real_"}
RIGHT_PREFIXES = ("real_", "mocap_right_")
BLACK = "black"                                     # camera-map source meaning "generated black video"
DEFAULT_CAMERA_MAP = {
    "left": {"head": "head", "knife": "knife", "wrist_left": "wrist_left"},
    "right": {"head": "head", "knife": BLACK, "wrist_left": "wrist_right"}}
TABLE_COLUMNS = (STATE_KEY, ACTION_KEY, "timestamp", "frame_index", "episode_index", "index", "task_index")
STATE_SPAN = 0.003          # trim_mocap_dataset.pause_candidates default (state_span)
ACTION_EPS = 1e-6           # same, action_epsilon
MAX_ACTION_SPAN = STATE_SPAN                         # pause_policy.maximum_action_span checked by trim_mocap_dataset.build()


def load_trim(handoff_source: Path):
    path = handoff_source / "scripts/compal/trim_mocap_dataset.py"
    spec = importlib.util.spec_from_file_location("compal_trim_mocap_dataset", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canonical(name: str, arm: str = "left") -> str:
    """Source episode name.  left: mocap_left_<x>.  right: real_NNNN (a bare number 5 -> real_0005); names that already
    start with real_ or mocap_right_ (the 14-D right-arm mocap sets, e.g. mocap_right_2_ep008) are kept as they are."""
    if arm == "left":
        return name if name.startswith("mocap_left_") else f"mocap_left_{name}"
    if arm != "right":
        raise ValueError(f"Unknown arm {arm!r}")
    if name.isdigit():
        return f"real_{int(name):04d}"
    return name if name.startswith(RIGHT_PREFIXES) else f"real_{name}"


def parse_camera_map(text: str | None, arm: str) -> dict:
    """'slot=source,...' -> {slot: source}; every model slot exactly once, source 'black' = generated black video."""
    if not text:
        return dict(DEFAULT_CAMERA_MAP[arm])
    result = {}
    for item in text.split(","):
        slot, _, source = item.partition("=")
        slot, source = slot.strip(), source.strip()
        if slot not in CAMERAS or not source or slot in result:
            raise ValueError(f"Bad camera map entry {item!r}; slots are {CAMERAS}, each once")
        result[slot] = source
    if set(result) != set(CAMERAS):
        raise ValueError(f"Camera map must give all of {CAMERAS}, got {sorted(result)}")
    return result


def expand_channels(values: np.ndarray, arm: str) -> np.ndarray:
    """(N, 6) joints of one arm -> (N, 14) in the contract's channel order, every other channel 0."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError(f"Expected (N, 6) joint values, got {values.shape}")
    out = np.zeros((len(values), len(CHANNELS)))
    out[:, ARM_FIRST[arm]:ARM_FIRST[arm] + 6] = values
    return out


def episode_name(entry: dict, label: str | None = None) -> str:
    """Unique name of one episode of a merged dataset (its meta/source_episodes.json entry).

    Plain preprocess.py output and the originals of augmented sets (`copy_index` 0 or absent): the source name, as always.
    Without a label, copy k > 0 of an augmented set is `<source>__c<k>`.  With a label (`--src LABEL=PATH`) EVERY episode of
    that source is `<source>__<label><k>`, so the copies of two augmented sets (e.g. state noise `sn`, background swap `bg`)
    and the plain originals can be requested side by side.  Held-out logic must still be done by `entry["source"]`."""
    copy = int(entry.get("copy_index") or 0)
    if label:
        return f"{entry['source']}__{label}{copy}"
    return entry["source"] if copy == 0 else f"{entry['source']}__c{copy}"


def parse_src(text: str):
    """`PATH` or `LABEL=PATH` (LABEL = letters/digits, e.g. sn=/data/aug_statenoise) -> (label | None, Path)."""
    label, sep, rest = text.partition("=")
    if sep and re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", label):
        return label, Path(rest)
    return None, Path(text)


def natural_key(name: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


# --------------------------------------------------------------------------------------------- heldout
def choose_heldout(report_path: Path, keep: list[str], count: int = 2, key: str = "dtw_all_joints"):
    """Median-DTW rule.  Candidates = kept episodes except the medoid; take the `count` episodes in the
    middle of the sorted to_medoid_deg distances (for 8 candidates: ranks 4 and 5)."""
    report = json.loads(report_path.read_text())
    rows = {e["name"]: e[key] for e in report["episodes"]}
    medoids = {r["medoid"] for r in rows.values()}
    if len(medoids) != 1:
        raise ValueError(f"Expected one medoid, got {medoids}")
    medoid = medoids.pop()
    if medoid not in keep:
        raise ValueError(f"Medoid {medoid} is not among the kept episodes")
    candidates = sorted(((rows[n]["to_medoid_deg"], n) for n in keep if n != medoid))
    start = (len(candidates) - count) // 2
    chosen = [n for _, n in candidates[start:start + count]]
    return {"metric": key, "medoid": medoid, "candidates_sorted": [(n, d) for d, n in candidates],
            "validation": sorted(chosen, key=natural_key),
            "train": sorted((n for n in keep if n not in chosen), key=natural_key)}


# ---------------------------------------------------------------------------------------------- bundle
def read_json(path):
    return json.loads(Path(path).read_text())


def run_ffmpeg(args, timeout=600):
    result = subprocess.run(["ffmpeg", "-v", "error", "-y", "-threads", "2", *args], capture_output=True, text=True,
                            timeout=timeout)
    if result.returncode:
        raise ValueError(f"ffmpeg failed: {result.stderr[:500]}")


def packet_count(path) -> int:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                          "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True, timeout=120).stdout
    return int(out.strip().rstrip(","))


def extract_segment(src, dst, first: int, frames: int, fps: int = FPS):
    """Frames [first, first+frames) of a packed mp4 as its own mp4 by stream copy (no re-encode).

    The packed files (preprocess.py, GOP 2, no B-frames) start every episode on a keyframe, so a copy cut is exact.
    Verified: packet count, first packet is a keyframe, and the first, middle and last frame decode identically to the
    packed source."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Seek a quarter frame past the first frame: copy mode starts at the keyframe at or before the seek time, and the
    # packed files' timestamps can be a few ticks off k/fps (then an exact k/fps would start one frame early).
    run_ffmpeg(["-ss", f"{(first + 0.25) / fps:.6f}", "-i", str(src), "-frames:v", str(frames), "-c", "copy",
                "-avoid_negative_ts", "make_zero", "-an", str(dst)])
    count = packet_count(dst)
    flags = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-read_intervals", "%+#1",
                            "-show_entries", "packet=flags", "-of", "csv=p=0", str(dst)],
                           capture_output=True, text=True, check=True, timeout=120).stdout
    if count != frames or "K" not in flags:
        raise ValueError(f"Segment copy of {src} [{first}, +{frames}) is not frame exact: {count} packets, flags {flags!r}")
    ends = [(first, 0), (first + frames // 2, frames // 2), (first + frames - 1, frames - 1)]
    want = decode_frames(src, [i for i, _ in ends], fps)
    got = decode_frames(dst, [j for _, j in ends], fps)
    if any(not np.array_equal(want[i], got[j]) for i, j in ends):
        raise ValueError(f"Segment copy of {src} [{first}, +{frames}) decodes differently from the source")
    return dst


def decode_frames(path, indices, fps: int = FPS):
    """RGB frames at the given frame numbers of an mp4, single-threaded PyAV (the multi-threaded reader in
    video_frames.py intermittently deadlocked inside pytest; one thread is plenty for a handful of frames)."""
    import av
    out = {}
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "NONE"
        stream.codec_context.thread_count = 1
        tb = float(stream.time_base)
        for index in indices:
            t = index / fps
            container.seek(int(max(t - 1.0 / fps, 0.0) / tb), stream=stream, backward=True, any_frame=False)
            for frame in container.decode(stream):
                if float(frame.pts) * tb >= t - 0.5 / fps:
                    out[index] = frame.to_ndarray(format="rgb24")
                    break
            else:
                raise ValueError(f"{path}: no frame {index}")
    return out


def black_video(dst, frames: int, width: int, height: int, fps: int = FPS):
    """All-black h264 yuv420p video with exactly `frames` frames (fills a camera slot that has no source)."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(["-f", "lavfi", "-i", f"color=c=black:s={width}x{height}:r={fps}", "-frames:v", str(frames),
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", str(dst)])
    if packet_count(dst) != frames:
        raise ValueError(f"Black video has the wrong frame count: {dst}")
    return dst


def stage_recordings(src: Path, names: list[str], stage: Path, arm: str = "left", camera_map: dict | None = None,
                     videos: str = "raw", label: str | None = None):
    """One single-episode v3.0 recording per episode: filtered table (14-D) + one video per model camera slot.

    videos="raw": link the original per-episode files of the source (left-arm recordings; source_path/videos/...).
    videos="packed": cut the episode out of the source dataset's own packed mp4s (needed when frames were removed
    from the middle of an episode, e.g. preprocess.py --trim-pauses).
    Episode names are `episode_name(entry, label)`: the source name for plain sets and copy 0, copies of augmented sets
    `<source>__c<k>` (or `<source>__<label><k>` with a label)."""
    camera_map = camera_map or DEFAULT_CAMERA_MAP[arm]
    info = read_json(src / "meta/info.json")
    if info.get("codebase_version") != "v3.0" or info.get("fps") != FPS:
        raise ValueError("Expected a 30 fps LeRobot v3.0 dataset from preprocess.py")
    sources = {episode_name(e, label): e for e in read_json(src / "meta/source_episodes.json")}
    episodes = pq.read_table(src / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
    data_files = sorted((src / "data").glob("chunk-*/*.parquet"))
    if len(data_files) != 1:
        raise ValueError("Expected the single data file written by preprocess.py")
    table = pq.read_table(data_files[0])
    source_dim = info["features"][STATE_KEY]["shape"][0]
    if source_dim not in (6, 14):
        raise ValueError(f"Unsupported state dimension {source_dim}")
    for cam in set(camera_map.values()) - {BLACK}:
        if f"observation.images.{cam}" not in info["features"]:
            raise ValueError(f"Camera map source {cam!r} is not a video of {src}")

    def widen(values):                         # (N, source_dim) -> (N, 14) contract order
        return expand_channels(values, arm) if source_dim == 6 else np.asarray(values, dtype=np.float64)

    raw = {c: np.asarray(table[c].to_pylist(), dtype=np.float64) for c in ("observation.state_raw", "action_teleop")}
    raw = {c: widen(v) for c, v in raw.items()}
    list_type = pa.list_(pa.float32(), len(CHANNELS))
    result = {}
    for name in names:
        if name not in sources:
            raise ValueError(f"{name} is not in {src}/meta/source_episodes.json")
        index = sources[name]["episode_index"]
        row = episodes[index]
        lo, hi = row["dataset_from_index"], row["dataset_to_index"]
        n = hi - lo
        part = table.slice(lo, n)
        schema = table.schema
        if source_dim == 14:
            state_col, action_col = part[STATE_KEY], part[ACTION_KEY]
        else:
            state_col = pa.array(widen(np.asarray(part[STATE_KEY].to_pylist())).tolist(), type=list_type)
            action_col = pa.array(widen(np.asarray(part[ACTION_KEY].to_pylist())).tolist(), type=list_type)
        columns = {
            STATE_KEY: state_col, ACTION_KEY: action_col,
            "timestamp": pa.array(np.arange(n) / FPS, type=schema.field("timestamp").type),
            "frame_index": pa.array(np.arange(n), type=schema.field("frame_index").type),
            "episode_index": pa.array(np.zeros(n), type=schema.field("episode_index").type),
            "index": pa.array(np.arange(n), type=schema.field("index").type),
            "task_index": pa.array(np.zeros(n), type=schema.field("task_index").type)}
        staged = pa.table({k: columns[k] for k in TABLE_COLUMNS})
        target = stage / name
        (target / "data/chunk-000").mkdir(parents=True)
        (target / "meta").mkdir(parents=True)
        pq.write_table(staged, target / "data/chunk-000/file-000.parquet")
        ep_info = json.loads(json.dumps(info))
        for extra in ("observation.state_raw", "action_teleop"):
            ep_info["features"].pop(extra, None)
        ep_info.update(total_episodes=1, total_frames=n, total_tasks=1, splits={"train": "0:1"})
        if source_dim == 6:
            for key in (STATE_KEY, ACTION_KEY):
                ep_info["features"][key].update(shape=[len(CHANNELS)], names=list(CHANNELS))
        # Video features follow the model's camera slots; a slot's feature describes its source camera.
        for key in [k for k in ep_info["features"] if k.startswith("observation.images.")]:
            del ep_info["features"][key]
        shapes = {}
        for slot in CAMERAS:
            source_cam = camera_map[slot]
            feature = info["features"].get(f"observation.images.{source_cam}") or info["features"][
                next(k for k in info["features"] if k.startswith("observation.images."))]
            ep_info["features"][f"observation.images.{slot}"] = json.loads(json.dumps(feature))
            shapes[slot] = feature["shape"]
        (target / "meta/info.json").write_text(json.dumps(ep_info, indent=2))
        for slot in CAMERAS:
            source_cam = camera_map[slot]
            link = target / "videos" / f"observation.images.{slot}" / "chunk-000/file-000.mp4"
            if source_cam == BLACK:
                height, width = shapes[slot][0], shapes[slot][1]
                black_video(link, n, width, height)
            elif videos == "raw":
                original = (Path(sources[name]["source_path"]) / "videos" / f"observation.images.{source_cam}"
                            / "chunk-000/file-000.mp4").resolve(strict=True)
                link.parent.mkdir(parents=True)
                link.symlink_to(original)
            elif videos == "packed":
                key = f"observation.images.{source_cam}"
                chunk, file_ = row[f"videos/{key}/chunk_index"], row[f"videos/{key}/file_index"]
                first = round(row[f"videos/{key}/from_timestamp"] * FPS)
                if abs(row[f"videos/{key}/from_timestamp"] * FPS - first) > 1e-3 or \
                        abs((row[f"videos/{key}/to_timestamp"] - row[f"videos/{key}/from_timestamp"]) * FPS - n) > 1e-3:
                    raise ValueError(f"{name}/{key}: video span does not match {n} frames at {FPS} fps")
                extract_segment(src / "videos" / key / f"chunk-{chunk:03d}/file-{file_:03d}.mp4", link, first, n)
            else:
                raise ValueError(f"Unknown videos mode {videos!r}")
        result[name] = {
            "frames": n, "dir": target,
            "state": np.asarray(staged[STATE_KEY].to_pylist(), dtype=np.float64),
            "action": np.asarray(staged[ACTION_KEY].to_pylist(), dtype=np.float64),
            "state_raw": raw["observation.state_raw"][lo:hi], "action_teleop": raw["action_teleop"][lo:hi]}
    return result


def stage_many(srcs: list, names: list[str], stage: Path, arm: str = "left", camera_map: dict | None = None,
               videos: str = "raw", labels: dict | None = None) -> tuple[dict, dict]:
    """stage_recordings over one or several merged datasets (preprocess.py outputs or augmented sets).  Every name must be
    an episode of exactly one of them (their meta/source_episodes.json, names as in `episode_name`).  `labels`: {src: label}
    for sources given as `--src LABEL=PATH`.  Returns (recordings in `names` order, {name: src})."""
    labels = {Path(k): v for k, v in (labels or {}).items()}
    owner = {}
    for src in srcs:
        for entry in read_json(Path(src) / "meta/source_episodes.json"):
            name = episode_name(entry, labels.get(Path(src)))
            if name in names:
                if name in owner:
                    raise ValueError(f"{name} is in both {owner[name]} and {src}")
                owner[name] = Path(src)
    missing = [n for n in names if n not in owner]
    if missing:
        raise ValueError(f"Not in any --src: {missing}")
    recordings = {}
    for src in dict.fromkeys(owner.values()):
        recordings.update(stage_recordings(src, [n for n in names if owner[n] == src], stage, arm, camera_map, videos,
                                           labels.get(Path(src))))
    return {n: recordings[n] for n in names}, {n: str(owner[n]) for n in names}


def shrink_to_filtered_flat(pause, action):
    """The importer rejects any dropped frame next to a changing *filtered* action.  Our action is the
    Gaussian-filtered next state, which bleeds a few frames into a teleop-flat pause; shrink the drop
    window to the longest run of frames that no filtered-action transition touches."""
    changed = np.abs(np.diff(action, axis=0)).max(axis=1) > ACTION_EPS
    touched = np.zeros(len(action), dtype=bool)      # transition k joins frames k and k+1
    touched[:-1] |= changed
    touched[1:] |= changed
    ds, de = pause["drop_start_frame"], pause["drop_end_frame_exclusive"]
    best, start = (0, ds, ds), None
    for i in range(ds, de + 1):
        if i < de and not touched[i]:
            start = i if start is None else start
        else:
            if start is not None and i - start > best[0]:
                best = (i - start, start, i)
            start = None
    length, first, last = best
    out = dict(pause, drop_start_frame=int(first), drop_end_frame_exclusive=int(last), removed_frames=int(length),
               drop_window_before_filter_flat_check=[int(ds), int(de)])
    return out if length > 0 else None


def filtered_span_ok(pause, state, action) -> bool:
    """The check of trim_mocap_dataset.build() on the staged (filtered) arrays over the whole stationary span."""
    b, z = pause["stationary_from_frame"], pause["stationary_to_frame_exclusive"]
    return bool(np.ptp(action[b:z], axis=0).max() <= MAX_ACTION_SPAN and np.ptp(state[b:z], axis=0).max() <= STATE_SPAN)


def build_plan(trim, recordings, split_of, pause_reduction, instruction=INSTRUCTION, report_pauses=False):
    """pause_reduction: remove the stagnant spans.  report_pauses: also detect them when nothing is removed and list
    them (with the checked/shrunk drop windows) under `unapplied_pause_candidates` / `rejected_pause_candidates`."""
    entries = []
    for name, rec in recordings.items():
        removals, rejected, unapplied = [], [], []
        if pause_reduction or report_pauses:
            # Detection on the ORIGINAL signals, exactly as in the handoff: raw state + teleop command.
            candidates = trim.pause_candidates(rec["state_raw"], rec["action_teleop"], float(FPS))
            for pause in candidates:
                checked = {**pause, **trim.check_pause_images(rec["dir"] / "videos/observation.images.head/chunk-000/file-000.mp4",
                                                              pause, FPS)}
                if not checked["image_stability_passed"]:
                    rejected.append(checked)
                    continue
                shrunk = shrink_to_filtered_flat(checked, rec["action"])
                if shrunk and not filtered_span_ok(checked, rec["state"], rec["action"]):
                    # build() refuses a pause whose FILTERED state/action move by more than STATE_SPAN anywhere in the whole
                    # stationary span.  Never true for the recorded originals (the detection is on the same signals); it can
                    # be for state-perturbed copies whose offset decays inside the span: keep those frames.
                    rejected.append({**checked, "reason": "filtered state/action not flat over the stationary span"})
                    continue
                if shrunk and not pause_reduction:
                    unapplied.append(shrunk)
                else:
                    (removals if shrunk else rejected).append(shrunk or {**checked, "reason": "no filtered-flat window"})
        entries.append({"id": name, "split": split_of[name], "source_frames": rec["frames"], "fps": FPS,
                        "start_frame": 0, "pause_removals": removals, "rejected_pause_candidates": rejected,
                        "unapplied_pause_candidates": unapplied})
    return {"schema": "turbovla.compal.mocap.pause-plan.v1", "retain_all_channels": True, "retain_full_task": True,
            "action_state_time_shift_frames": 0, "canonical_instruction": instruction,
            # build() checks these on the staged (filtered) arrays over each stationary span.
            "pause_policy": {"maximum_action_span": MAX_ACTION_SPAN, "maximum_state_span_rad": STATE_SPAN,
                             "minimum_seconds": 1.5, "guard_seconds": 0.5,
                             "detection": "raw observation.state + teleop action (handoff rule), "
                                          "drop window shrunk to the filtered-action-flat run"},
            "episodes": entries}


def cmd_heldout(args):
    keep = sorted((canonical(n, args.arm) for n in args.keep.split(",")), key=natural_key)
    print(json.dumps(choose_heldout(args.report, keep), indent=2))


def cmd_bundle(args):
    handoff_source = args.handoff_source.resolve()
    trim = load_trim(handoff_source)
    train = sorted((canonical(n, args.arm) for n in args.train.split(",")), key=natural_key)
    val = sorted((canonical(n, args.arm) for n in args.validation.split(",")), key=natural_key)
    if set(train) & set(val):
        raise ValueError("Train and validation overlap")
    camera_map = parse_camera_map(args.camera_map, args.arm)
    videos = args.videos or ("packed" if args.arm == "right" else "raw")
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"{out} exists; choose a new directory")
    stage = out / "recordings"
    names = train + val
    parsed = [parse_src(t) for t in args.src]
    srcs = [p.resolve() for _, p in parsed]
    labels = {p.resolve(): label for label, p in parsed if label}
    recordings, owner = stage_many(srcs, names, stage, args.arm, camera_map, videos, labels)
    split_of = {**{n: "train" for n in train}, **{n: "validation" for n in val}}
    plan = build_plan(trim, recordings, split_of, not args.no_pause_reduction, args.task, args.report_pauses)
    (out / "pause-plan.json").write_text(json.dumps(plan, indent=2))
    for e in plan["episodes"]:
        print(f"PAUSES {e['id']:<18} {e['source_frames']:>5} frames: remove "
              f"{[(p['drop_start_frame'], p['removed_frames']) for p in e['pause_removals']]}, rejected "
              f"{[(p['stationary_from_frame'], p.get('reason') or 'image') for p in e['rejected_pause_candidates']]}, "
              f"unapplied {[(p['drop_start_frame'], p['removed_frames']) for p in e['unapplied_pause_candidates']]}", flush=True)
    trim.build(stage, out / "pause-plan.json", out / "bundle")
    # Provenance for inference: carried by the importer into import-manifest.json and split.json.
    manifest_path = out / "bundle/manifest.json"
    manifest = read_json(manifest_path)
    dims = {str(s): read_json(s / "meta/info.json")["features"][STATE_KEY]["shape"][0] for s in srcs}
    manifest["camera_mapping"] = {"model_slot_from_source": camera_map, "black": "all-zero RGB frame",
                                  "source_dataset": str(srcs[0]) if len(srcs) == 1 else [str(s) for s in srcs],
                                  "video_mode": videos}
    if len(srcs) > 1:
        manifest["camera_mapping"]["source_dataset_of_episode"] = owner
    if labels:
        manifest["camera_mapping"]["source_labels"] = {str(k): v for k, v in labels.items()}
    manifest["channel_placement"] = {"arm": args.arm,
                                     "source_channels": next(iter(dims.values())) if len(dims) == 1 else dims,
                                     "first_channel": ARM_FIRST[args.arm], "channels": CHANNELS,
                                     "other_channels": "constant 0 (6-D sources) / as in the 14-D source"}
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"\nbundle: {out/'bundle'}\nNext (handoff mocap.py with the overlay source, see README): prepare -> images -> check")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    h = sub.add_parser("heldout", help="print the reproducible validation / train split")
    h.add_argument("--report", type=Path, required=True, help="quality_report.py report.json")
    h.add_argument("--keep", required=True, help="comma-separated kept episodes (with or without the mocap_left_ / real_ prefix)")
    h.add_argument("--arm", choices=sorted(ARM_FIRST), default="left")
    b = sub.add_parser("bundle", help="build the pause-compressed bundle with the handoff's trim_mocap_dataset.build()")
    b.add_argument("--src", required=True, action="append", metavar="[LABEL=]PATH",
                   help="merged dataset written by preprocess.py or an augmented set; repeat it to combine several (an episode "
                        "name must be in exactly one of them; the sources may differ in 6-D/14-D and video size). Episode "
                        "names: <source> (plain / copy 0), <source>__c<k> for copy k of an augmented set, and with LABEL=PATH "
                        "every episode of that set is <source>__<LABEL><copy>")
    b.add_argument("--handoff-source", type=Path, required=True, help="<bundle>/source of the handoff")
    b.add_argument("--train", required=True)
    b.add_argument("--validation", required=True)
    b.add_argument("--arm", choices=sorted(ARM_FIRST), default="left",
                   help="arm of the recording; right: a 6-D source is placed in channels 7..12 (a 14-D source is used as it is; "
                   "default left: 14-D source)")
    b.add_argument("--camera-map", help="slot=source,... for head,knife,wrist_left; source 'black' = black video "
                   "(default left: identity; right: head=head,knife=black,wrist_left=wrist_right)")
    b.add_argument("--videos", choices=("raw", "packed"),
                   help="raw: link the original per-episode videos (default left); packed: cut from the dataset's own "
                        "mp4s (default right)")
    b.add_argument("--task", default=INSTRUCTION, help="canonical instruction stored in the bundle plan")
    b.add_argument("--report-pauses", action="store_true",
                   help="with --no-pause-reduction: still detect the handoff's stagnant spans and list them as unapplied")
    b.add_argument("--out", type=Path, required=True, help="new directory (recordings/, pause-plan.json, bundle/)")
    b.add_argument("--no-pause-reduction", action="store_true", help="keep every frame (deviates from the handoff recipe)")
    args = p.parse_args()
    {"heldout": cmd_heldout, "bundle": cmd_bundle}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
