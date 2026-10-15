#!/usr/bin/env python3
"""Convert a perception snapshot (left base frame) into the RIGHT arm's base frame.

Perception writes box_h_cut_snapshot/v1 in the LEFT arm base frame (frame_id "base_link" = world of
the dual cell). The normal single-arm box_h_cut_executor (cut_pattern left_only / middle_only) is
started for the right arm alone, so it needs a snapshot that is

  1. expressed in the right arm's base frame        p = B^-1 p_world   (normal: rotation only)
  2. relabelled by a 180 degree rotation about the carton normal (the carton's RIGHT face becomes
     the executor's "left" face, the top seam runs right end -> left end).

This is a line-by-line port of src/igus_rebel_moveit_config/include/igus_rebel_moveit_config/
snapshot_frame.hpp (poseFromXyzRpy, toPlanningFrame, relabelRotate180, labelHandedness,
sameHandedness). B = arm_right_base of the dual-arm calibration file (URDF fixed-axis rpy,
R = Rz(yaw) Ry(pitch) Rx(roll)).

The executor's staleness check uses the snapshot FILE MODIFICATION TIME, so the output file gets
the mtime of the SOURCE file (it is not made to look fresher than the capture).

CLI:  right_snapshot.py convert SRC OUT [--calib FILE] [--allow-stale]
"""

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml

DEFAULT_CALIBRATION = (
    Path(__file__).resolve().parents[2]
    / "src" / "rebel_dual_table_description" / "config" / "dual_arm_calibration.yaml"
)
MAX_AGE_S = 240.0  # box_h_cut_executor.yaml external_snapshot_max_age_s
PROVENANCE_KEY = "right_arm_single"
EXPECTED_SCHEMA = "box_h_cut_snapshot/v1"
EXPECTED_FRAME = "base_link"

POINT_NAMES = (
    "seam_start", "seam_end", "seam_center",
    "back_left", "back_right", "front_left", "front_right",
    "back_middle_left", "front_middle_left", "back_middle_right", "front_middle_right",
)
NORMAL_NAME = "surface_normal"

# relabelRotate180 (snapshot_frame.hpp): out.<key> = in.<value>
ROTATE_180_MAP = {
    "back_left": "front_right",
    "front_right": "back_left",
    "back_right": "front_left",
    "front_left": "back_right",
    "seam_start": "seam_end",
    "seam_end": "seam_start",
    "back_middle_left": "front_middle_right",
    "front_middle_right": "back_middle_left",
    "front_middle_left": "back_middle_right",
    "back_middle_right": "front_middle_left",
}


class ConversionError(Exception):
    """The snapshot cannot be converted safely."""


# ----------------------------------------------------------------------------- geometry
def pose_from_xyz_rpy(xyz, rpy):
    """4x4 pose, URDF fixed-axis rpy: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    roll, pitch, yaw = (float(v) for v in rpy)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    pose = np.eye(4)
    pose[:3, :3] = rz @ ry @ rx
    pose[:3, 3] = [float(v) for v in xyz]
    return pose


def invert_pose(pose):
    inv = np.eye(4)
    inv[:3, :3] = pose[:3, :3].T
    inv[:3, 3] = -pose[:3, :3].T @ pose[:3, 3]
    return inv


def quaternion_xyzw(rotation):
    """Unit quaternion (x, y, z, w) of a rotation matrix (w >= 0)."""
    m = np.asarray(rotation, dtype=float)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        q = [0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        q = [(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        q = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s]
    q = np.array(q) / np.linalg.norm(q)
    if q[3] < 0.0:
        q = -q
    return tuple(float(v) for v in q)


def to_planning_frame(points, base_in_snapshot):
    """snapshot_frame::toPlanningFrame: p = B^-1 p (normal rotated only). `points` is a dict
    name -> 3-vector including NORMAL_NAME; returns a new dict."""
    inv = invert_pose(base_in_snapshot)
    out = {}
    for name in POINT_NAMES:
        out[name] = inv[:3, :3] @ np.asarray(points[name], dtype=float) + inv[:3, 3]
    out[NORMAL_NAME] = inv[:3, :3] @ np.asarray(points[NORMAL_NAME], dtype=float)
    return out


def relabel_rotate_180(points):
    """snapshot_frame::relabelRotate180 (seam_center and the normal are kept)."""
    out = dict(points)
    for name, source in ROTATE_180_MAP.items():
        out[name] = points[source]
    return out


def label_handedness(points):
    """snapshot_frame::labelHandedness: sign of ((br - bl) x (fl - bl)) . n, 0 if degenerate."""
    across = np.asarray(points["back_right"]) - np.asarray(points["back_left"])
    along = np.asarray(points["front_left"]) - np.asarray(points["back_left"])
    value = float(np.dot(np.cross(across, along), np.asarray(points[NORMAL_NAME])))
    if not math.isfinite(value) or abs(value) < 1e-12:
        return 0
    return 1 if value > 0.0 else -1


def same_handedness(before, after):
    a, b = label_handedness(before), label_handedness(after)
    return a != 0 and a == b


# ------------------------------------------------------------------------- calibration
def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_calibration(path):
    """Returns dict with arm_right_base / camera_link entries {'xyz': [3], 'rpy': [3]}."""
    with open(path, "r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    try:
        cal = document["dual_arm_calibration"]
        result = {}
        for key in ("arm_right_base", "camera_link"):
            xyz = [float(v) for v in cal[key]["xyz"]]
            rpy = [float(v) for v in cal[key]["rpy"]]
            if len(xyz) != 3 or len(rpy) != 3 or not all(map(math.isfinite, xyz + rpy)):
                raise ValueError(f"{key} must have 3 finite xyz and 3 finite rpy values")
            result[key] = {"xyz": xyz, "rpy": rpy}
        result["source"] = str(cal.get("source", ""))
        return result
    except (KeyError, TypeError, ValueError) as exc:
        raise ConversionError(f"calibration file {path} is unusable: {exc}") from exc


def right_base_pose(calibration):
    base = calibration["arm_right_base"]
    return pose_from_xyz_rpy(base["xyz"], base["rpy"])


def camera_in_right_base(calibration):
    """(xyz, quaternion xyzw) of camera_link in the right arm's base_link = B^-1 * T_world_camera."""
    camera = pose_from_xyz_rpy(calibration["camera_link"]["xyz"], calibration["camera_link"]["rpy"])
    in_right = invert_pose(right_base_pose(calibration)) @ camera
    return tuple(float(v) for v in in_right[:3, 3]), quaternion_xyzw(in_right[:3, :3])


# --------------------------------------------------------------------------- conversion
def _read_points(document):
    points = {}
    for name in POINT_NAMES:
        node = document.get(name)
        try:
            vec = [float(node["x"]), float(node["y"]), float(node["z"])]
        except (KeyError, TypeError, ValueError) as exc:
            raise ConversionError(f"snapshot point '{name}' missing or invalid") from exc
        if not all(map(math.isfinite, vec)):
            raise ConversionError(f"snapshot point '{name}' is not finite")
        points[name] = np.array(vec)
    node = document.get(NORMAL_NAME)
    try:
        normal = [float(node["x"]), float(node["y"]), float(node["z"])]
    except (KeyError, TypeError, ValueError) as exc:
        raise ConversionError("snapshot surface_normal missing or invalid") from exc
    if not all(map(math.isfinite, normal)):
        raise ConversionError("snapshot surface_normal is not finite")
    points[NORMAL_NAME] = np.array(normal)
    return points


def convert_document(document, calibration):
    """Pure conversion of a parsed snapshot. Returns a new document (without provenance)."""
    if document.get("schema") != EXPECTED_SCHEMA:
        raise ConversionError(f"snapshot schema must be {EXPECTED_SCHEMA}")
    if PROVENANCE_KEY in document:
        raise ConversionError(
            f"snapshot already carries a '{PROVENANCE_KEY}' block: refusing a double conversion")
    if document.get("frame_id") != EXPECTED_FRAME:
        raise ConversionError(
            f"snapshot frame_id is '{document.get('frame_id')}', expected '{EXPECTED_FRAME}' "
            "(the left arm base frame)")
    points = _read_points(document)
    in_right = to_planning_frame(points, right_base_pose(calibration))
    relabelled = relabel_rotate_180(in_right)
    if not same_handedness(in_right, relabelled):
        raise ConversionError(
            "rotate_180 would change the handedness of the carton labels (mirror or degenerate "
            "corners); refusing")
    output = json.loads(json.dumps(document))  # deep copy, keeps key order
    for name in POINT_NAMES:
        output[name] = {k: float(v) for k, v in zip("xyz", relabelled[name])}
    output[NORMAL_NAME] = {k: float(v) for k, v in zip("xyz", relabelled[NORMAL_NAME])}
    return output


def convert_file(source, output, calibration_file=DEFAULT_CALIBRATION, allow_stale=False,
                 max_age_s=MAX_AGE_S, now=None):
    """Convert `source` -> `output` atomically; the output mtime is set to the source mtime."""
    source, output = Path(source), Path(output)
    calibration_file = Path(calibration_file)
    if source.resolve() == output.resolve():
        raise ConversionError("source and output must be different files")
    if not source.is_file():
        raise ConversionError(f"source snapshot not found: {source}")
    source_stat = source.stat()
    now_s = time.time() if now is None else now
    age_s = now_s - source_stat.st_mtime
    stale = age_s > max_age_s
    stale_reason = f"file mtime is {age_s:.1f} s old"
    source_bytes = source.read_bytes()
    try:
        document = json.loads(source_bytes)
    except json.JSONDecodeError as exc:
        raise ConversionError(f"source snapshot is not valid JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise ConversionError("source snapshot is not a JSON object")
    # The executor checks BOTH the file mtime (readSnapshotDocument) and acquisition_stamp_ns
    # (parseSnapshotFromDocument) against external_snapshot_max_age_s.
    stamp_ns = document.get("acquisition_stamp_ns")
    stamp_age_s = None
    if isinstance(stamp_ns, int) and stamp_ns > 0:
        stamp_age_s = now_s - stamp_ns / 1e9
        if stamp_age_s > max_age_s:
            stale = True
            stale_reason = (stale_reason + "; " if age_s > max_age_s else "") + \
                f"acquisition_stamp_ns is {stamp_age_s:.1f} s old"
    if stale and not allow_stale:
        raise ConversionError(
            f"source snapshot is stale ({stale_reason}; maximum {max_age_s:.0f} s): {source}; "
            "capture a new one (--allow-stale is for mock only)")
    calibration = load_calibration(calibration_file)
    converted = convert_document(document, calibration)
    base = calibration["arm_right_base"]
    converted[PROVENANCE_KEY] = {
        "source_file": str(source.resolve()),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "source_mtime": source_stat.st_mtime,
        "source_age_s_at_conversion": round(age_s, 3),
        "stamp_age_s_at_conversion": None if stamp_age_s is None else round(stamp_age_s, 3),
        "allow_stale": bool(stale and allow_stale),
        "calibration_file": str(calibration_file.resolve()),
        "calibration_sha256": sha256_file(calibration_file),
        "calibration_source": calibration["source"],
        "arm_right_base": {"xyz": base["xyz"], "rpy": base["rpy"]},
        "relabel": "rotate_180",
        "frame": "right arm base_link (frame_id kept as base_link)",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(prefix=output.name + ".", suffix=".tmp", dir=output.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(converted, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # mtime = the capture's age, set BEFORE the rename so the file never appears fresher.
        os.utime(temp_name, ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns))
        os.replace(temp_name, output)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise
    return converted


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    convert = sub.add_parser("convert", help="convert SRC (left base frame) to OUT (right base frame)")
    convert.add_argument("source")
    convert.add_argument("output")
    convert.add_argument("--calib", default=str(DEFAULT_CALIBRATION),
                         help="dual_arm_calibration.yaml (default: %(default)s)")
    convert.add_argument("--allow-stale", action="store_true",
                         help="MOCK ONLY: accept a source older than 240 s (marked in the output; "
                              "the executor will still refuse the file unless it is touched)")
    args = parser.parse_args(argv)
    try:
        convert_file(args.source, args.output, args.calib, args.allow_stale)
    except ConversionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"converted {args.source} -> {args.output} (right arm base frame, rotate_180)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
