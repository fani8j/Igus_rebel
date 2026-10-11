# Provenance: moved into box_seam_detection (2026-10-12) from seam_handoff_20260908/scripts/capture_and_query_depth.py
# (snapshot post-processing helpers) at commit 08d1041 (working tree identical to HEAD, unmodified).
"""Planner-snapshot (box_h_cut_snapshot/v1) helpers: side cutting midpoints, validation, provenance
stamps, atomic same-directory writes and the fail-closed removal / side-seed quarantine."""

import json
import math
import os
import sys
import tempfile
from pathlib import Path

from .side_points import compute_midpoints

SIDE_CUTTING_POINT_NAMES = (
    "back_middle_left", "front_middle_left",
    "back_middle_right", "front_middle_right",
)


def _snapshot_xyz(snapshot, name):
    point = snapshot.get(name)
    if not isinstance(point, dict):
        raise ValueError(f"planner snapshot 缺少 {name}")
    values = [point.get(axis) for axis in ("x", "y", "z")]
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) for value in values):
        raise ValueError(f"planner snapshot 的 {name} 不是有限 XYZ 點")
    return values


def add_side_cutting_points(snapshot_path):
    """Atomically add side-cutting midpoints derived from canonical base-frame points."""
    snapshot_path = Path(snapshot_path)
    with open(snapshot_path, encoding="utf-8") as input_file:
        snapshot = json.load(input_file)
    if snapshot.get("schema") != "box_h_cut_snapshot/v1":
        raise ValueError("side cutting points require box_h_cut_snapshot/v1")

    computed = compute_midpoints(
        _snapshot_xyz(snapshot, "back_left"),
        _snapshot_xyz(snapshot, "front_left"),
        _snapshot_xyz(snapshot, "seam_start"),
        _snapshot_xyz(snapshot, "seam_end"),
        _snapshot_xyz(snapshot, "back_right"),
        _snapshot_xyz(snapshot, "front_right"),
    )
    for name, point in zip(SIDE_CUTTING_POINT_NAMES, computed):
        values = [float(value) for value in point]
        if len(values) != 3 or not all(math.isfinite(value) for value in values):
            raise ValueError(f"side_cutting_point.py produced invalid {name}")
        snapshot[name] = dict(zip(("x", "y", "z"), values))
    snapshot["side_cutting_points"] = {
        "schema": "side_cutting_midpoints/v1",
        "source_mapping": {
            "back_middle_left": ["back_left", "seam_start"],
            "front_middle_left": ["front_left", "seam_start"],
            "back_middle_right": ["back_right", "seam_end"],
            "front_middle_right": ["front_right", "seam_end"],
        },
    }

    write_json_atomically(snapshot_path, snapshot)
    return snapshot


def stamp_detector_method(snapshot_path, detector_method, extra=None):
    """Atomically add the additive provenance key ``detector_method`` (which
    detector branch produced the six points) to the planner snapshot written by
    camera_point_to_base_link.py (which this orchestrator does not modify);
    every existing key is kept unchanged. ``extra`` (optional dict) adds
    further provenance keys (e.g. ``autolabel_v2_attempts``); it may not
    replace a key the snapshot already has."""
    snapshot_path = Path(snapshot_path)
    with open(snapshot_path, encoding="utf-8") as input_file:
        snapshot = json.load(input_file)
    # Called right after validate_planner_snapshot (which enforces the schema).
    if not isinstance(snapshot, dict):
        raise ValueError("planner snapshot is not a JSON object")
    snapshot["detector_method"] = str(detector_method)
    for key, value in (extra or {}).items():
        if key in snapshot:
            raise ValueError(f"planner snapshot 已有 {key}，拒絕覆寫")
        snapshot[key] = value
    write_json_atomically(snapshot_path, snapshot)
    return snapshot


def write_json_atomically(path, value):
    """Write JSON via a same-directory temp file + os.replace; the temp file is
    removed on any failure (serialisation, fsync or rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent,
                prefix=f".{path.name}.", suffix=".tmp", delete=False) as output_file:
            temporary_path = Path(output_file.name)
            json.dump(value, output_file, ensure_ascii=False, indent=2)
            output_file.write("\n")
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def validate_planner_snapshot(snapshot_path, manifest, target_frame):
    try:
        with open(snapshot_path, encoding="utf-8") as input_file:
            snapshot = json.load(input_file)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"無法讀取 canonical planner snapshot {snapshot_path}: {error}") from error
    if snapshot.get("schema") != "box_h_cut_snapshot/v1":
        raise ValueError(f"不支援的 planner snapshot schema: {snapshot.get('schema')!r}")
    if snapshot.get("capture_id") != manifest["capture_id"]:
        raise ValueError("planner snapshot capture_id 與 capture manifest 不一致")
    if snapshot.get("acquisition_stamp_ns") != manifest["image"]["header"]["stamp_ns"]:
        raise ValueError("planner snapshot acquisition timestamp 與 capture manifest 不一致")
    if snapshot.get("frame_id") != target_frame:
        raise ValueError("planner snapshot frame_id 與要求 target frame 不一致")
    side_metadata = snapshot.get("side_cutting_points")
    if not isinstance(side_metadata, dict) or (
            side_metadata.get("schema") != "side_cutting_midpoints/v1"):
        raise ValueError("planner snapshot 缺少 side_cutting_midpoints/v1 metadata")
    for name in SIDE_CUTTING_POINT_NAMES:
        _snapshot_xyz(snapshot, name)
    expected_mapping = {
        "back_middle_left": ["back_left", "seam_start"],
        "front_middle_left": ["front_left", "seam_start"],
        "back_middle_right": ["back_right", "seam_end"],
        "front_middle_right": ["front_right", "seam_end"],
    }
    if side_metadata.get("source_mapping") != expected_mapping:
        raise ValueError("planner snapshot 的 side cutting source mapping 不正確")
    expected_points = compute_midpoints(
        _snapshot_xyz(snapshot, "back_left"),
        _snapshot_xyz(snapshot, "front_left"),
        _snapshot_xyz(snapshot, "seam_start"),
        _snapshot_xyz(snapshot, "seam_end"),
        _snapshot_xyz(snapshot, "back_right"),
        _snapshot_xyz(snapshot, "front_right"),
    )
    for name, expected in zip(SIDE_CUTTING_POINT_NAMES, expected_points):
        actual = _snapshot_xyz(snapshot, name)
        if any(not math.isclose(value, float(reference), rel_tol=0.0, abs_tol=1e-12)
               for value, reference in zip(actual, expected)):
            raise ValueError(f"planner snapshot 的 {name} 不是來源點的中點")
    return snapshot


SIDE_SEED_REJECTED_SUFFIX = ".side_seed_rejected"


def side_seed_rejected_path(snapshot_path):
    snapshot_path = Path(snapshot_path)
    return snapshot_path.with_name(snapshot_path.name + SIDE_SEED_REJECTED_SUFFIX)


def _quarantine_planner_snapshot(snapshot_path, rejected=None):
    """Fail closed: move the step-5 snapshot aside to <snapshot>.side_seed_rejected.

    Atomic rename (replaces any older rejected file) so no valid-looking
    box_h_cut_snapshot/v1 remains at the planner path. If the rename fails the
    snapshot is deleted instead. ``rejected`` overrides the quarantine path (staged snapshot ->
    <final snapshot>.side_seed_rejected). Returns the quarantine path or None.
    """
    snapshot_path = Path(snapshot_path)
    rejected = Path(rejected) if rejected is not None else side_seed_rejected_path(snapshot_path)
    try:
        os.replace(snapshot_path, rejected)
        return rejected
    except FileNotFoundError:
        return None
    except OSError as error:
        print(f"無法把 planner snapshot 移到 {rejected}: {error}；改為刪除", file=sys.stderr)
        try:
            snapshot_path.unlink(missing_ok=True)
        except OSError as unlink_error:
            print(f"planner snapshot 刪除也失敗: {unlink_error}（{snapshot_path} 不可使用）",
                  file=sys.stderr)
        return None


_write_json_atomically = write_json_atomically   # historical name


def remove_snapshot(path):
    """Fail-closed removal of a stale planner snapshot and its side-seed quarantine file.
    Returns None on success (also when nothing existed) or the OSError that prevented it."""
    path = Path(path)
    for target in (path, side_seed_rejected_path(path)):
        try:
            target.unlink(missing_ok=True)
        except OSError as error:
            return error
    return None

