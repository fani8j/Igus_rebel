#!/usr/bin/env python3
"""Read one episode of a LeRobot dataset with pyarrow only, no lerobot install needed.

Two on-disk versions are understood:

* **v3.0** (the robot_remote recorder format, robot_type igus_rebel_dual_arm_real, and the output of preprocess.py):
  episodes share data/video files, ``meta/episodes/*.parquet`` has the per-episode rows;
* **v2.1** (GR00T layout, e.g. ``real_right``): one parquet and one mp4 per episode and camera, ``meta/episodes.jsonl``
  has the per-episode rows (``length``, ``tasks`` and any extra keys such as ``box_pose``, kept in ``Episode.meta``).
  v2.1 is read in place: no copy and no conversion.  (lerobot 0.4.4's own v2.1 -> v3.0 converter needs
  ``meta/episodes_stats.jsonl`` and works in place on a local copy; see README "Right-arm real_right dataset".)

Shared by rerun_view.py (visualisation), lerobot_to_plan.py (real-arm replay), quality_report.py and preprocess.py.
The dataset is untrusted input: only JSON and parquet are read, nothing is executed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


@dataclass
class Episode:
    root: Path
    episode_index: int
    fps: float
    task: str
    names: list[str]                      # feature names, e.g. arm_left_joint1..6, arm_left_gripper, arm_right_... (14) or 6
    timestamp: np.ndarray                 # (N,) seconds from episode start
    state: np.ndarray                     # (N, D) observation.state
    action: np.ndarray                    # (N, D) action
    # video key -> (mp4 path, from_timestamp of this episode inside that file)
    videos: dict[str, tuple[Path, float]] = field(default_factory=dict)
    name: str = ""                        # label used in reports / selection (see episode_name)
    version: str = "v3.0"                 # codebase_version of the dataset on disk
    meta: dict = field(default_factory=dict)      # v2.1: the episodes.jsonl row (box_pose, source, rig, ...)
    extras: dict = field(default_factory=dict)    # other per-frame columns, e.g. next.reward / next.done: (N,) arrays
    robot_type: str = ""

    def arm(self, arm: str, source: str = "state") -> np.ndarray:
        """(N, 6) joint positions [rad] of 'arm_left' / 'arm_right', from 'state' or 'action'."""
        data = self.state if source == "state" else self.action
        idx = [self.names.index(f"{arm}_joint{i}") for i in range(1, 7)]
        return data[:, idx]

    @property
    def arms(self) -> list[str]:
        """Arm prefixes that have joint1..6 in ``names`` (order of appearance)."""
        return detect_arms(self.names)

    def active_arm(self) -> str:
        """The arm with the largest joint travel (an all-zero arm, as the right arm of the left-arm recordings,
        never wins).  Raises if the dataset has no ``<arm>_joint1..6`` columns."""
        arms = self.arms
        if not arms:
            raise ValueError(f"{self.root}: no '<arm>_joint1..6' columns in {self.names}")
        travel = [float(np.abs(np.diff(self.arm(a), axis=0)).sum()) if len(self.state) > 1 else 0.0 for a in arms]
        return arms[int(np.argmax(travel))]


def detect_arms(names: list[str]) -> list[str]:
    """['arm_left', 'arm_right'] for names containing ``<arm>_joint1..6`` (only complete 6-joint arms)."""
    out = []
    for n in names:
        if n.endswith("_joint1"):
            arm = n[: -len("_joint1")]
            if all(f"{arm}_joint{i}" in names for i in range(1, 7)) and arm not in out:
                out.append(arm)
    return out


# ------------------------------------------------------------------------------------------- discovery
def is_dataset(path: Path) -> bool:
    return (Path(path) / "meta" / "info.json").is_file()


def find_datasets(root: str | Path) -> list[Path]:
    """Dataset roots under ``root``: ``root`` itself if it has meta/info.json, else its subfolders that have one
    (sorted by name).  Only the directory layout is read."""
    root = Path(root).expanduser().resolve()
    if is_dataset(root):
        return [root]
    subs = sorted(p for p in root.iterdir() if is_dataset(p))
    if not subs:
        raise FileNotFoundError(f"{root}: no LeRobot dataset (meta/info.json) here or in its subfolders")
    return subs


def _info(root: Path) -> dict:
    return json.loads((root / "meta" / "info.json").read_text())


def _is_v2(info: dict) -> bool:
    return str(info.get("codebase_version", "")).startswith("v2")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _v3_episode_rows(root: Path) -> list[dict]:
    files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    if not files:
        raise ValueError(f"{root}: no meta/episodes/*.parquet")
    return [r for f in files for r in pq.read_table(f).to_pylist()]


def episode_rows(root: str | Path) -> list[dict]:
    """The per-episode metadata rows of a dataset (v3.0: meta/episodes parquet; v2.1: meta/episodes.jsonl), sorted by
    episode_index."""
    root = Path(root).expanduser().resolve()
    info = _info(root)
    rows = _read_jsonl(root / "meta" / "episodes.jsonl") if _is_v2(info) else _v3_episode_rows(root)
    return sorted(rows, key=lambda r: r["episode_index"])


def episode_name(root: Path, episode_index: int, n_episodes: int, row: dict | None = None) -> str:
    """Label of an episode: the folder name for a one-episode dataset (the left-arm recordings), else the ``source``
    key of its metadata row (v2.1 ``real_0003``) or ``<folder>_ep003``."""
    if n_episodes == 1:
        return root.name
    if row and isinstance(row.get("source"), str) and row["source"]:
        return row["source"]
    return f"{root.name}_ep{episode_index:03d}"


@dataclass(frozen=True)
class EpisodeRef:
    """One selectable episode of a source: (dataset root, episode_index) plus its label and extra selector aliases."""
    name: str
    root: Path
    episode_index: int
    n_in_dataset: int = 1
    aliases: tuple = ()


def list_episodes(source_root: str | Path) -> list[EpisodeRef]:
    """Every episode of every dataset under ``source_root`` (a dataset, or a folder of datasets), in dataset-name then
    episode_index order.  Numeric aliases ("19", "episode_000019") exist only for multi-episode datasets."""
    refs = []
    for d in find_datasets(source_root):
        rows = episode_rows(d)
        multi = len(rows) > 1
        for r in rows:
            i = int(r["episode_index"])
            aliases = (str(i), f"episode_{i:06d}", f"ep{i}", f"ep{i:03d}") if multi else ()
            refs.append(EpisodeRef(episode_name(d, i, len(rows), r), d, i, len(rows), aliases))
    return refs


# ------------------------------------------------------------------------------------------------ loading
def _stack(col) -> np.ndarray:
    return np.stack(col.to_numpy(zero_copy_only=False)).astype(np.float64)


def _extras(t, skip: set) -> dict:
    out = {}
    for n in t.column_names:
        if n in skip:
            continue
        col = t.column(n)
        if str(col.type).startswith("list"):
            continue
        out[n] = np.asarray(col.to_numpy())
    return out


def load_episode(root: str | Path, episode_index: int = 0) -> Episode:
    root = Path(root).expanduser().resolve()
    info = _info(root)
    version = str(info.get("codebase_version", ""))
    if _is_v2(info):
        return _load_v2(root, info, episode_index)
    if not version.startswith("v3"):
        raise ValueError(f"{root}: expected LeRobot v2.x or v3.x, got {info.get('codebase_version')!r}")
    return _load_v3(root, info, episode_index)


_CORE = {"observation.state", "action", "timestamp", "frame_index", "episode_index", "index", "task_index"}


def _load_v3(root: Path, info: dict, episode_index: int) -> Episode:
    names = list(info["features"]["observation.state"]["names"])
    eps = _v3_episode_rows(root)
    rows = [e for e in eps if e["episode_index"] == episode_index]
    if not rows:
        raise ValueError(f"{root}: no episode {episode_index} (have {[e['episode_index'] for e in eps]})")
    ep = rows[0]

    data_rel = info["data_path"].format(chunk_index=ep["data/chunk_index"], file_index=ep["data/file_index"])
    t = pq.read_table(root / data_rel)
    sel = np.asarray(t.column("episode_index").to_numpy()) == episode_index
    order = np.argsort(np.asarray(t.column("frame_index").to_numpy())[sel])
    timestamp = np.asarray(t.column("timestamp").to_numpy(), dtype=np.float64)[sel][order]
    state = _stack(t.column("observation.state"))[sel][order]
    action = _stack(t.column("action"))[sel][order]
    extras = {k: v[sel][order] for k, v in _extras(t, _CORE).items()}

    videos = {}
    for key, feat in info["features"].items():
        if feat.get("dtype") != "video":
            continue
        rel = info["video_path"].format(video_key=key, chunk_index=ep[f"videos/{key}/chunk_index"],
                                        file_index=ep[f"videos/{key}/file_index"])
        videos[key] = (root / rel, float(ep[f"videos/{key}/from_timestamp"]))

    tasks = ep.get("tasks") or [""]
    meta = {k: v for k, v in ep.items() if not (k.startswith(("stats/", "videos/", "data/", "meta/")) or k in (
        "episode_index", "tasks", "length", "dataset_from_index", "dataset_to_index"))}
    return Episode(root, episode_index, float(info["fps"]), str(tasks[0]), names,
                   timestamp, state, action, videos, name=episode_name(root, episode_index, len(eps), ep),
                   version=str(info["codebase_version"]), meta=meta, extras=extras,
                   robot_type=str(info.get("robot_type", "")))


def _load_v2(root: Path, info: dict, episode_index: int) -> Episode:
    names = list(info["features"]["observation.state"]["names"])
    rows = _read_jsonl(root / "meta" / "episodes.jsonl")
    sel = [r for r in rows if r["episode_index"] == episode_index]
    if not sel:
        raise ValueError(f"{root}: no episode {episode_index} (have {[r['episode_index'] for r in rows]})")
    row = sel[0]
    chunk = episode_index // int(info.get("chunks_size", 1000))
    t = pq.read_table(root / info["data_path"].format(episode_chunk=chunk, episode_index=episode_index))
    order = np.argsort(np.asarray(t.column("frame_index").to_numpy()))
    n = len(order)
    if "length" in row and int(row["length"]) != n:
        raise ValueError(f"{root}: episode {episode_index} has {n} parquet rows but episodes.jsonl says {row['length']}")
    timestamp = np.asarray(t.column("timestamp").to_numpy(), dtype=np.float64)[order]
    state = _stack(t.column("observation.state"))[order]
    action = _stack(t.column("action"))[order]
    extras = {k: v[order] for k, v in _extras(t, _CORE).items()}
    videos = {}
    for key, feat in info["features"].items():
        if feat.get("dtype") != "video":
            continue
        rel = info["video_path"].format(episode_chunk=chunk, video_key=key, episode_index=episode_index)
        videos[key] = (root / rel, 0.0)          # one mp4 per episode and camera, starts at 0
    tasks = row.get("tasks") or [""]
    meta = {k: v for k, v in row.items() if k not in ("episode_index", "tasks", "length")}
    return Episode(root, episode_index, float(info["fps"]), str(tasks[0]), names, timestamp, state, action, videos,
                   name=episode_name(root, episode_index, len(rows), row), version=str(info["codebase_version"]),
                   meta=meta, extras=extras, robot_type=str(info.get("robot_type", "")))


if __name__ == "__main__":
    import sys
    e = load_episode(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0)
    print(f"{e.name} ({e.version}) ep {e.episode_index}: '{e.task}' {len(e.timestamp)} frames @ {e.fps:g} fps, "
          f"{e.timestamp[-1]:.1f} s, videos {sorted(e.videos)}, meta {e.meta}")
