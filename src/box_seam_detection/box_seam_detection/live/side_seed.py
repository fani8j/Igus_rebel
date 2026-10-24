#!/usr/bin/env python3
# Provenance: moved into box_seam_detection (2026-10-12) from seam_handoff_20260908/scripts/side_face_seed.py
# at commit 08d1041 (working tree identical to HEAD, unmodified).
# Changes: box_profiles is a package import (was file-path loading).
"""Side-face seed for the knife-camera side-tape cut (schema side_tape_seed/v1.1).

Reads a canonical planner snapshot (box_h_cut_snapshot/v1, frame_id base_link)
and derives, from the static camera's top-face box model plus the known box
height, the four side-face planes of the box in base_link, with the two
seam-end faces ("left" = face under the contract LEFT edge back_left-front_left,
"right" = face under the contract RIGHT edge back_right-front_right) carrying
the expected tape centreline running DOWN the face from the seam endpoint.

The block is added IN PLACE to the snapshot (atomic rewrite, every existing key
kept) and optionally written to a standalone JSON. Validation failures exit 1
without touching any file; argparse usage errors exit 2.

Conventions (see rgbd_captures/docs/side_tape_seed_contract.md):
  * all points/vectors in base_link, metres; points are {"x","y","z"} dicts
  * top_normal: unit normal of the top plane, pointing up (+z of base_link side)
  * face normal: unit, perpendicular to the face's top edge and to top_normal
    (i.e. lies in the top plane), pointing OUTWARD away from the box centre
  * face plane: n . (p - point) = 0; signed distance > 0 is outside the box
  * down = -top_normal
  * v1.1: faces.<k>.plane is the OUTER WALL plane: the top-edge plane shifted
    outward along its normal by wall_offset_m = (outer_dim - measured_top_dim)/2,
    from the user-measured outside-wall box dimensions per class (OUTER_DIMS_M).
    The old top-edge plane is kept as faces.<k>.top_edge_plane.
"""

import argparse
import importlib.util
import json
import math
import os
import sys
import tempfile
from pathlib import Path

SCHEMA = "side_tape_seed/v1.1"


from ..core import box_profiles  # noqa: E402

PROFILE_NAMES = tuple(box_profiles.PROFILES)      # small, big, generic
# Outside-wall to outside-wall box dimensions per class (metres), user-measured.
#   along_seam  = distance between the two seam-end faces (contract left/right)
#   across_seam = distance between the other two faces (contract back/front),
#                 i.e. along the edges the seam crosses.
# None = unknown -> no wall offset, wall_source "outer_wall_unknown".
OUTER_DIMS_M = {
    name: {key: (None if value is None else value / 1000.0)
           for key, value in profile.outer_dims_mm.items()}
    for name, profile in box_profiles.PROFILES.items()}
WALL_OFFSET_RANGE_M = (0.0, 0.008)   # valid outer-wall offset per face
WALL_OFFSET_EPS_M = 1e-9             # float slack at the range ends
WALL_OFFSET_NEG_TOL_M = 0.001        # accept down to -1 mm (lid flush / corner noise), clamped to 0
# face -> which outer dimension spans it and its opposite face
FACE_DIM = {"left": "along_seam", "right": "along_seam",
            "back": "across_seam", "front": "across_seam"}
SNAPSHOT_SCHEMA = "box_h_cut_snapshot/v1"
BOX_HEIGHTS_M = {name: profile.height_m for name, profile in box_profiles.PROFILES.items()
                 if profile.height_m is not None}   # generic: measured height only
BIG_LONG_SIDE_M = box_profiles.LEGACY_BIG_LONG_SIDE_M   # top face long side > 150 mm -> big box
LEFT_RIGHT_DOT_MAX = -0.95       # left/right outward normals roughly opposite
FRONT_BACK_DOT_MAX = -0.95       # front/back outward normals roughly opposite
SEAM_EDGE_TOL_M = 0.002          # seam endpoint to its top edge (3-D distance)
SEAM_EDGE_PARAM_MARGIN_M = 0.002  # seam endpoint along its edge: -2 mm .. length + 2 mm
CORNER_ANGLE_RANGE_DEG = (75.0, 105.0)  # top-quad interior angles (rectangular box)
TOP_NORMAL_MAX_TILT_DEG = 60.0   # top normal vs base_link +z (box upright)
INNER_FOLD_OFFSET_M = (0.001, 0.0015)
CORNERS = ("back_left", "back_right", "front_left", "front_right")
RING = ("back_left", "back_right", "front_right", "front_left")   # top-quad cycle
# face -> (top edge start, top edge end); left/right match the seam contract.
FACE_EDGES = {
    "left": ("back_left", "front_left"),
    "right": ("back_right", "front_right"),
    "back": ("back_left", "back_right"),
    "front": ("front_left", "front_right"),
}
SEAM_POINT_FOR_FACE = {"left": "seam_start", "right": "seam_end"}
SIDE_TAPE_PLANE_TOL_M = 0.015    # measured side-tape point to its face's outer-wall plane
HEIGHT_MIN_M = 0.005             # measured height sanity range (generic profile)
HEIGHT_MAX_M = 1.5


class SeedError(ValueError):
    """Input or geometry validation failure (exit 1)."""


# ------------------------------------------------------------ vector helpers

def _add(a, b):
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]


def _sub(a, b):
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def _scale(a, s):
    return [a[0] * s, a[1] * s, a[2] * s]


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return [a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]]


def _norm(a):
    return math.sqrt(_dot(a, a))


def _unit(a, what):
    length = _norm(a)
    if not math.isfinite(length) or length < 1e-9:
        raise SeedError(f"degenerate vector for {what}")
    return _scale(a, 1.0 / length)


def _angle_deg(a, b):
    c = max(-1.0, min(1.0, _dot(_unit(a, "angle"), _unit(b, "angle"))))
    return math.degrees(math.acos(c))


def _xyz(p):
    return {"x": float(p[0]), "y": float(p[1]), "z": float(p[2])}


def _read_point(snapshot, name):
    point = snapshot.get(name)
    if not isinstance(point, dict):
        raise SeedError(f"planner snapshot is missing {name}")
    values = [point.get(axis) for axis in ("x", "y", "z")]
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           for v in values):
        raise SeedError(f"planner snapshot {name} is not a finite XYZ point")
    return [float(v) for v in values]


def _closest_on_segment(p, a, b):
    ab = _sub(b, a)
    t = _dot(_sub(p, a), ab) / _dot(ab, ab)
    t_clamped = max(0.0, min(1.0, t))
    return _add(a, _scale(ab, t_clamped)), t


# ------------------------------------------------------------ core

def classify_box(long_side_m, box_class="auto"):
    """Profile name: ``auto`` = today's rule (classify_legacy); else the named profile
    (small / big / generic)."""
    if box_class == "auto":
        return box_profiles.classify_legacy(long_side_m).name
    if box_class not in PROFILE_NAMES:
        raise SeedError(f"unknown box class {box_class!r}")
    return box_class


def _outer_dims(cls, outer_along_seam_mm=None, outer_across_seam_mm=None):
    """Outer dims (m or None) per dimension and their source."""
    dims, sources = {}, {}
    for key, override in (("along_seam", outer_along_seam_mm),
                          ("across_seam", outer_across_seam_mm)):
        if override is not None:
            if not (isinstance(override, (int, float)) and not isinstance(override, bool)
                    and math.isfinite(override) and override > 0):
                raise SeedError(f"--outer-{key.replace('_', '-')}-mm must be positive, "
                                f"got {override!r}")
            dims[key] = float(override) / 1000.0
            sources[key] = "override"
        else:
            dims[key] = OUTER_DIMS_M[cls][key]
            sources[key] = f"class_default:{cls}" if dims[key] is not None else "unknown"
    return dims, sources


def _profile_name_from_snapshot(snapshot):
    value = snapshot.get("box_profile")
    name = value.get("name") if isinstance(value, dict) else value
    return name if isinstance(name, str) and name in PROFILE_NAMES else None


def _read_measured_tape(side_tape, snapshot):
    """Validated {face: {"top": [xyz], "bottom": [xyz], "source": str}} from the
    optional measured side-tape centrelines (left / right faces only)."""
    raw = side_tape if side_tape is not None else snapshot.get("side_tape_measured")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise SeedError("side_tape centrelines must be an object {left/right: {top, bottom}}")
    out = {}
    for face, entry in raw.items():
        if face not in SEAM_POINT_FOR_FACE:
            raise SeedError(f"side_tape: unknown face {face!r} (only left/right carry tape)")
        if entry is None:
            continue
        if not isinstance(entry, dict):
            raise SeedError(f"side_tape.{face} must be an object with top and bottom points")
        points = {}
        for key in ("top", "bottom"):
            value = entry.get(key)
            if isinstance(value, dict):
                value = [value.get("x"), value.get("y"), value.get("z")]
            if (not isinstance(value, (list, tuple)) or len(value) != 3
                    or any(isinstance(v, bool) or not isinstance(v, (int, float))
                           or not math.isfinite(v) for v in value)):
                raise SeedError(f"side_tape.{face}.{key} is not a finite XYZ point")
            points[key] = [float(v) for v in value]
        out[face] = {**points, "source": str(entry.get("source") or "measured")}
    return out


SIDE_TAPE_MAX_TILT_DEG = 15.0       # measured line vs the box's vertical
SIDE_TAPE_HEIGHT_SLACK_M = 0.010    # extent <= box height + 10 mm
SIDE_TAPE_SEAM_ALONG_TOL_M = 0.020  # measured line vs the extrapolated seam point, along the face


def _measured_tape_problem(m_top, m_bottom, wall_mid, outward, up, down, height, edge_unit,
                           expected_top):
    """Reason string when a measured side-tape centreline is implausible, else None."""
    for label, point in (("top", m_top), ("bottom", m_bottom)):
        off = abs(_dot(_sub(point, wall_mid), outward))
        if off > SIDE_TAPE_PLANE_TOL_M:
            return f"{label} {off * 1000:.1f} mm from the face wall plane"
    vector = _sub(m_bottom, m_top)
    length = _norm(vector)
    if length < 0.005:
        return f"degenerate ({length * 1000:.2f} mm long)"
    if _dot(vector, up) >= 0.0:
        return "top point not above the bottom point"
    if length > height + SIDE_TAPE_HEIGHT_SLACK_M:
        return f"extent {length * 1000:.0f} mm > box height {height * 1000:.0f} mm + 10 mm"
    tilt = _angle_deg(vector, down)
    if tilt > SIDE_TAPE_MAX_TILT_DEG:
        return f"line {tilt:.1f} deg from vertical"
    for label, point in (("top", m_top), ("bottom", m_bottom)):
        along = abs(_dot(_sub(point, expected_top), edge_unit))
        if along > SIDE_TAPE_SEAM_ALONG_TOL_M:
            return f"{label} {along * 1000:.0f} mm from the seam point along the face"
    return None


def compute_seed(snapshot, box_height_mm=None, box_class="auto",
                 outer_along_seam_mm=None, outer_across_seam_mm=None,
                 measured_height_mm=None, side_tape_expected=False, side_tape=None,
                 box_class_source=None):
    """Return the side_tape_seed/v1.1 block for ``snapshot``; raise SeedError.

    ``box_class``: auto (snapshot ``box_profile`` when present, else today's
    150 mm rule) | small | big | generic. ``generic`` has no construction facts:
    planes at the top edges (outer_wall_unknown) and the MEASURED height
    (``measured_height_mm`` / snapshot ``measured_height_mm``; none -> SeedError
    height_unavailable). For known profiles the measured height is recorded only.
    ``side_tape``: optional measured side-tape centrelines per face, base_link:
    {"left": {"top": xyz, "bottom": xyz}, "right": {...}}; absent faces keep the
    extrapolated centreline. ``side_tape_expected`` (or snapshot
    ``side_tape_expected``) flags ``side_tape_expected_not_found`` for a left/right
    face without a measured centreline."""
    if snapshot.get("schema") != SNAPSHOT_SCHEMA:
        raise SeedError(f"expected {SNAPSHOT_SCHEMA}, got {snapshot.get('schema')!r}")
    if snapshot.get("frame_id") != "base_link":
        raise SeedError(f"expected frame_id base_link, got {snapshot.get('frame_id')!r}")
    corners = {name: _read_point(snapshot, name) for name in CORNERS}
    seam = {name: _read_point(snapshot, name) for name in ("seam_start", "seam_end")}
    raw_normal = _read_point(snapshot, "surface_normal")

    n = _unit(raw_normal, "surface_normal")
    normal_flipped = n[2] < 0.0
    if normal_flipped:
        n = _scale(n, -1.0)
    tilt = _angle_deg(n, [0.0, 0.0, 1.0])
    if tilt > TOP_NORMAL_MAX_TILT_DEG:
        raise SeedError(f"top normal is {tilt:.1f} deg from base_link +z "
                        f"(> {TOP_NORMAL_MAX_TILT_DEG}); box not upright?")
    down = _scale(n, -1.0)

    edge_len = {face: _norm(_sub(corners[b], corners[a]))
                for face, (a, b) in FACE_EDGES.items()}
    for face, length in edge_len.items():
        if not math.isfinite(length) or length < 0.01:
            raise SeedError(f"{face} top edge is degenerate ({length * 1000:.2f} mm)")
    side_lr = 0.5 * (edge_len["left"] + edge_len["right"])     # contract left/right edges
    side_fb = 0.5 * (edge_len["back"] + edge_len["front"])
    long_side = max(side_lr, side_fb)
    class_source = box_class_source
    if box_class == "auto" and _profile_name_from_snapshot(snapshot) is not None:
        box_class, class_source = _profile_name_from_snapshot(snapshot), "snapshot"
    cls = classify_box(long_side, box_class)
    profile = box_profiles.PROFILES[cls]
    if class_source is None:
        class_source = "auto" if box_class == "auto" else "override"
    if measured_height_mm is None:
        measured_height_mm = snapshot.get("measured_height_mm")
    if measured_height_mm is not None and not (
            isinstance(measured_height_mm, (int, float)) and not isinstance(measured_height_mm, bool)
            and math.isfinite(measured_height_mm) and measured_height_mm > 0):
        raise SeedError(f"measured height must be positive, got {measured_height_mm!r}")
    if box_height_mm is not None:
        if not (isinstance(box_height_mm, (int, float)) and math.isfinite(box_height_mm)
                and box_height_mm > 0):
            raise SeedError(f"--box-height-mm must be positive, got {box_height_mm!r}")
        height = float(box_height_mm) / 1000.0
        height_source = "override"
    elif profile.height_m is None:      # generic: measured only, never a guess
        if measured_height_mm is None:
            raise SeedError(f"height_unavailable: profile {cls} has no default height and no "
                            f"measured height was given (--measured-height-mm / --box-height-mm)")
        height = float(measured_height_mm) / 1000.0
        if not HEIGHT_MIN_M <= height <= HEIGHT_MAX_M:
            raise SeedError(f"measured height {height * 1000:.1f} mm outside "
                            f"[{HEIGHT_MIN_M * 1000:.0f}, {HEIGHT_MAX_M * 1000:.0f}] mm")
        height_source = "measured"
    else:
        height = BOX_HEIGHTS_M[cls]
        height_source = f"class_default:{cls}"
    measured_tape = _read_measured_tape(side_tape, snapshot)
    invalid_tape = []
    expected_tape = bool(side_tape_expected or snapshot.get("side_tape_expected")
                         or getattr(profile, "side_tape_expected", False))

    centre = _scale(_add(_add(corners["back_left"], corners["back_right"]),
                         _add(corners["front_left"], corners["front_right"])), 0.25)
    table_point = _add(centre, _scale(down, height))

    # Interior angles of the top quad, measured in the top plane.
    corner_angles = {}
    for index, name in enumerate(RING):
        prev_vec = _sub(corners[RING[index - 1]], corners[name])
        next_vec = _sub(corners[RING[(index + 1) % 4]], corners[name])
        prev_vec = _sub(prev_vec, _scale(n, _dot(prev_vec, n)))
        next_vec = _sub(next_vec, _scale(n, _dot(next_vec, n)))
        corner_angles[name] = _angle_deg(prev_vec, next_vec)
    low, high = CORNER_ANGLE_RANGE_DEG
    bad_corners = {name: round(angle, 2) for name, angle in corner_angles.items()
                   if not low <= angle <= high}
    if bad_corners:
        raise SeedError(f"top quad is not rectangular: corner angles {bad_corners} deg "
                        f"outside [{low:.0f}, {high:.0f}]")

    # Measured top-face size in the top plane: distance between the two
    # opposite top edges = mean of the two in-plane lengths of the edges that
    # join them (left/right faces: back & front edges; back/front: left & right).
    def _in_plane_len(face):
        a, b = corners[FACE_EDGES[face][0]], corners[FACE_EDGES[face][1]]
        d = _sub(b, a)
        return _norm(_sub(d, _scale(n, _dot(d, n))))
    measured_top = {"along_seam": 0.5 * (_in_plane_len("back") + _in_plane_len("front")),
                    "across_seam": 0.5 * (_in_plane_len("left") + _in_plane_len("right"))}
    outer_dims, outer_sources = _outer_dims(cls, outer_along_seam_mm, outer_across_seam_mm)
    wall = {}
    flags = []
    wall_errors = []   # raised after the geometry checks (a bad quad is the root cause)
    lo_off, hi_off = WALL_OFFSET_RANGE_M
    neg_tol = float(getattr(profile, "wall_neg_tol_mm", 1.0)) / 1000.0
    for key in ("along_seam", "across_seam"):
        outer = outer_dims[key]
        if outer is None:
            wall[key] = {"offset": 0.0, "source": "outer_wall_unknown"}
            if "outer_wall_unknown" not in flags:
                flags.append("outer_wall_unknown")
            continue
        offset = 0.5 * (outer - measured_top[key])
        if not (lo_off - neg_tol - WALL_OFFSET_EPS_M <= offset <= hi_off + WALL_OFFSET_EPS_M):
            hint = (" (negative: detected top face is BIGGER than the outer box -> wrong "
                    "box, wrong class or bad corners)" if offset < 0 else
                    " (detected top face far smaller than the box -> wrong class or bad "
                    "corners)")
            wall_errors.append(
                f"outer-wall offset {key} = ({outer * 1000:.2f} - "
                f"{measured_top[key] * 1000:.2f})/2 = {offset * 1000:.2f} mm outside "
                f"[{lo_off * 1000:.0f}, {hi_off * 1000:.0f}] mm for {cls} box{hint}")
            offset = 0.0
        if offset < lo_off:   # within the -1 mm tolerance: clamp to the top edge and say so
            if "wall_offset_clamped_negative" not in flags:
                flags.append("wall_offset_clamped_negative")
        wall[key] = {"offset": max(0.0, min(hi_off, offset)), "source": "user_outer_dims"}

    faces = {}
    # face_normal_vs_top_deg is 90 by construction (the face normal is built as
    # top_normal x edge); it is reported as a numerical sanity diagnostic only.
    diagnostics = {"edge_tilt_from_top_plane_deg": {}, "face_normal_vs_top_deg": {},
                   "corner_angles_deg": corner_angles}
    for face, (a_name, b_name) in FACE_EDGES.items():
        a, b = corners[a_name], corners[b_name]
        edge_dir = _sub(b, a)
        edge_in_plane = _sub(edge_dir, _scale(n, _dot(edge_dir, n)))
        outward = _unit(_cross(n, edge_in_plane), f"{face} face normal")
        mid = _scale(_add(a, b), 0.5)
        if _dot(_sub(mid, centre), outward) < 0.0:
            outward = _scale(outward, -1.0)
        diagnostics["face_normal_vs_top_deg"][face] = _angle_deg(outward, n)
        diagnostics["edge_tilt_from_top_plane_deg"][face] = abs(
            90.0 - _angle_deg(edge_dir, n))
        dim_key = FACE_DIM[face]
        offset = wall[dim_key]["offset"]
        shift = _scale(outward, offset)
        wall_a, wall_b = _add(a, shift), _add(b, shift)
        entry = {
            "top_edge_corners": [a_name, b_name],
            # plane = OUTER WALL (v1.1); top_edge_plane = the v1 plane.
            "plane": {"point": _xyz(_add(mid, shift)), "normal": _xyz(outward)},
            "top_edge_plane": {"point": _xyz(mid), "normal": _xyz(outward)},
            "wall_offset_m": offset,
            "outer_dim_m": outer_dims[dim_key],
            "outer_dim_axis": dim_key,
            "measured_top_dim_m": measured_top[dim_key],
            "wall_source": wall[dim_key]["source"],
            "top_edge": {"start": _xyz(a), "end": _xyz(b)},
            "wall_top_edge": {"start": _xyz(wall_a), "end": _xyz(wall_b)},
            "bottom_edge": {"start": _xyz(_add(wall_a, _scale(down, height))),
                            "end": _xyz(_add(wall_b, _scale(down, height)))},
            "top_edge_bottom_edge": {"start": _xyz(_add(a, _scale(down, height))),
                                     "end": _xyz(_add(b, _scale(down, height)))},
            "top_edge_length_m": edge_len[face],
        }
        if face in SEAM_POINT_FOR_FACE:
            seam_name = SEAM_POINT_FOR_FACE[face]
            seam_raw = seam[seam_name]
            on_edge, t = _closest_on_segment(seam_raw, a, b)   # t is UNclamped
            along_m = t * edge_len[face]
            if not (-SEAM_EDGE_PARAM_MARGIN_M <= along_m
                    <= edge_len[face] + SEAM_EDGE_PARAM_MARGIN_M):
                raise SeedError(
                    f"{seam_name} projects outside the {face} top edge {a_name}-{b_name} "
                    f"(edge parameter t={t:.4f}, {along_m * 1000:.2f} mm along a "
                    f"{edge_len[face] * 1000:.2f} mm edge; margin "
                    f"{SEAM_EDGE_PARAM_MARGIN_M * 1000:.1f} mm)")
            distance = _norm(_sub(seam_raw, on_edge))
            if distance > SEAM_EDGE_TOL_M:
                raise SeedError(f"{seam_name} is {distance * 1000:.2f} mm from the {face} "
                                f"top edge {a_name}-{b_name} (> {SEAM_EDGE_TOL_M * 1000:.1f} mm)")
            on_wall = _add(on_edge, shift)
            entry["tape_centerline"] = {
                "top": _xyz(on_wall),
                "bottom": _xyz(_add(on_wall, _scale(down, height))),
                "top_edge_point": _xyz(on_edge),
                "top_edge_point_bottom": _xyz(_add(on_edge, _scale(down, height))),
                "wall_offset_m": offset,
                "direction": _xyz(down),
                "length_m": height,
                "seam_point": seam_name,
                "seam_point_raw": _xyz(seam_raw),
                "seam_point_to_edge_m": distance,
                "edge_param_t": t,
                "source": "extrapolated_from_seam",
            }
            if face in measured_tape:
                tape = entry["tape_centerline"]
                m_top, m_bottom = measured_tape[face]["top"], measured_tape[face]["bottom"]
                edge_unit = _unit(_sub(b, a), f"{face} top edge")
                invalid = _measured_tape_problem(
                    m_top, m_bottom, wall_mid=_add(mid, shift), outward=outward, up=n,
                    down=down, height=height, edge_unit=edge_unit, expected_top=on_wall)
                if invalid is not None:
                    invalid_tape.append(f"side_tape_measured_invalid({face}: {invalid})")
                    del measured_tape[face]
                else:
                    vector = _sub(m_bottom, m_top)
                    moved = ("top", "bottom", "direction", "length_m", "top_edge_point",
                             "top_edge_point_bottom")
                    tape["extrapolated"] = {key: tape.pop(key) for key in moved}
                    tape.update({"top": _xyz(m_top), "bottom": _xyz(m_bottom),
                                 "direction": _xyz(_unit(vector, "measured side tape")),
                                 "length_m": _norm(vector),
                                 "source": measured_tape[face]["source"]})
        faces[face] = entry

    lr_dot = _dot(_read_normal(faces["left"]), _read_normal(faces["right"]))
    fb_dot = _dot(_read_normal(faces["front"]), _read_normal(faces["back"]))
    diagnostics["left_right_normal_dot"] = lr_dot
    diagnostics["front_back_normal_dot"] = fb_dot
    if lr_dot > LEFT_RIGHT_DOT_MAX:
        raise SeedError(f"left/right face normals not opposite (dot {lr_dot:.4f} > "
                        f"{LEFT_RIGHT_DOT_MAX})")
    if fb_dot > FRONT_BACK_DOT_MAX:
        raise SeedError(f"front/back face normals not opposite (dot {fb_dot:.4f} > "
                        f"{FRONT_BACK_DOT_MAX})")
    if wall_errors:
        raise SeedError("; ".join(wall_errors))
    flags.extend(invalid_tape)
    tape_sources = {face: faces[face]["tape_centerline"]["source"] for face in SEAM_POINT_FOR_FACE}
    if expected_tape and any(face not in measured_tape for face in SEAM_POINT_FOR_FACE):
        flags.append("side_tape_expected_not_found")

    return {
        "schema": SCHEMA,
        "frame_id": "base_link",
        "units": "m",
        "source": {
            "snapshot_schema": snapshot.get("schema"),
            "snapshot_id": snapshot.get("snapshot_id"),
            "capture_id": snapshot.get("capture_id"),
            "acquisition_stamp_ns": snapshot.get("acquisition_stamp_ns"),
            "generator": "side_face_seed.py",
        },
        "box_class": cls,
        "box_class_source": class_source,
        "box_height_m": height,
        "box_height_source": height_source,
        "measured_height_mm": (None if measured_height_mm is None
                               else float(measured_height_mm)),
        "side_tape": {"expected": expected_tape,
                      "measured_faces": sorted(measured_tape),
                      "centerline_source": tape_sources},
        "top_face_size_m": {"left_right_edges": side_lr, "front_back_edges": side_fb,
                            "long_side": long_side},
        "measured_top_dims_m": dict(measured_top),
        "outer_dims_m": dict(outer_dims),
        "outer_dims_source": outer_sources,
        "wall_offsets_m": {key: wall[key]["offset"] for key in wall},
        "wall_sources": {key: wall[key]["source"] for key in wall},
        "flags": flags,
        "top_normal": _xyz(n),
        "top_normal_flipped_to_up": normal_flipped,
        "top_centre": _xyz(centre),
        "plane_rms_m": snapshot.get("plane_rms_m"),
        "table_z_estimate": table_point[2],
        "table_point_estimate": _xyz(table_point),
        "faces": faces,
        "diagnostics": diagnostics,
        "tolerances": {"left_right_dot_max": LEFT_RIGHT_DOT_MAX,
                       "front_back_dot_max": FRONT_BACK_DOT_MAX,
                       "seam_edge_m": SEAM_EDGE_TOL_M,
                       "seam_edge_param_margin_m": SEAM_EDGE_PARAM_MARGIN_M,
                       "corner_angle_range_deg": list(CORNER_ANGLE_RANGE_DEG),
                       "wall_offset_range_m": list(WALL_OFFSET_RANGE_M)},
        "notes": [
            "Tape width is NOT measured here: tape_centerline is the expected centre "
            "line of the tape strip; the knife-camera tracker finds the actual tape edges.",
            "Top-face corners (and therefore every top_edge_plane) are the INNER FOLD "
            "corners of the flat top seen from above. faces.<k>.plane is the OUTER WALL: "
            "top_edge_plane shifted outward by wall_offset_m = (outer_dim - "
            "measured_top_dim)/2 from the user-measured outside-wall box size "
            "(small: 106.25 mm along the seam, 112.0 mm across; seam-end walls sit "
            "~4.8 mm outside their top edges, the other two ~1 mm).",
            "wall_source outer_wall_unknown (e.g. big box, outer size not measured yet): "
            "no offset applied, plane == top_edge_plane; the physical wall may sit "
            "several mm further out (small box: ~4.8 mm on the seam-end faces) - "
            "add it to the penetration budget.",
            "bottom_edge / tape_centerline.bottom / table_*_estimate assume a rigid box of "
            "box_height_m standing on a plane parallel to the top face (informational).",
        ],
        "inner_fold_outer_offset_m": list(INNER_FOLD_OFFSET_M),
    }


def _read_normal(face_entry):
    normal = face_entry["plane"]["normal"]
    return [normal["x"], normal["y"], normal["z"]]


def atomic_write_json(path, data):
    """Same-directory temp file + os.replace; the temp file is removed on any
    failure (serialisation, fsync or rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent,
                prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def add_seed_to_snapshot(snapshot_path, box_height_mm=None, box_class="auto",
                         out_json=None, outer_along_seam_mm=None,
                         outer_across_seam_mm=None, measured_height_mm=None,
                         side_tape_expected=False, side_tape=None, box_class_source=None):
    """Compute the seed and write it in place (and to ``out_json``). Returns the block.

    Nothing is written unless validation passes.
    """
    snapshot_path = Path(snapshot_path)
    try:
        with open(snapshot_path, encoding="utf-8") as handle:
            snapshot = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise SeedError(f"cannot read planner snapshot {snapshot_path}: {error}") from error
    if not isinstance(snapshot, dict):
        raise SeedError("planner snapshot is not a JSON object")
    seed = compute_seed(snapshot, box_height_mm=box_height_mm, box_class=box_class,
                        outer_along_seam_mm=outer_along_seam_mm,
                        outer_across_seam_mm=outer_across_seam_mm,
                        measured_height_mm=measured_height_mm,
                        side_tape_expected=side_tape_expected, side_tape=side_tape,
                        box_class_source=box_class_source)
    snapshot["side_tape_seed"] = seed
    if out_json is not None:
        atomic_write_json(out_json, seed)
    atomic_write_json(snapshot_path, snapshot)
    return seed


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="side_face_seed.py",
        description="Side-face seed for the knife-camera side-tape cut (schema "
                    "side_tape_seed/v1.1): adds side-face planes and tape centrelines "
                    "to a box_h_cut_snapshot/v1 planner snapshot in place.")
    parser.add_argument("--planner-json", required=True,
                        help="box_h_cut_snapshot/v1 planner snapshot (updated in place)")
    parser.add_argument("--out-json", default=None,
                        help="optional standalone copy of the side_tape_seed block")
    parser.add_argument("--box-height-mm", type=float, default=None,
                        help="override the box height (default: by class, small 86.3 / big 134.25)")
    parser.add_argument("--box-class", choices=("auto", *PROFILE_NAMES), default="auto",
                        help="auto = the snapshot's box_profile, else top-face long side "
                             "> 150 mm is big")
    parser.add_argument("--box-profile", choices=PROFILE_NAMES, default=None,
                        help="box profile recorded by the detector (wins over --box-class auto)")
    parser.add_argument("--measured-height-mm", type=float, default=None,
                        help="measured box height (record.measured_height_mm); the height of "
                             "the generic profile, recorded only for known profiles")
    parser.add_argument("--side-tape-expected", action="store_true",
                        help="the box has side tape: flag side_tape_expected_not_found when a "
                             "face has no measured centreline")
    parser.add_argument("--side-tape-json", default=None,
                        help="measured side-tape centrelines per face (base_link): "
                             '{"left": {"top": {x,y,z}, "bottom": {x,y,z}}, "right": {...}}')
    parser.add_argument("--outer-along-seam-mm", type=float, default=None,
                        help="outside-wall distance between the two seam-end faces "
                             "(default by class: small 106.25, big unknown)")
    parser.add_argument("--outer-across-seam-mm", type=float, default=None,
                        help="outside-wall distance between the two non-seam-end faces, "
                             "i.e. along the edges the seam crosses (default by class: "
                             "small 112.0, big unknown)")
    args = parser.parse_args(argv)
    try:
        side_tape = None
        if args.side_tape_json is not None:
            try:
                with open(args.side_tape_json, encoding="utf-8") as handle:
                    side_tape = json.load(handle)
            except (OSError, json.JSONDecodeError) as error:
                raise SeedError(f"cannot read --side-tape-json: {error}") from error
        box_class, class_source = args.box_class, None
        if args.box_profile is not None and args.box_class == "auto":
            box_class, class_source = args.box_profile, "record_profile"
        seed = add_seed_to_snapshot(args.planner_json, args.box_height_mm,
                                    box_class, args.out_json,
                                    args.outer_along_seam_mm, args.outer_across_seam_mm,
                                    measured_height_mm=args.measured_height_mm,
                                    side_tape_expected=args.side_tape_expected,
                                    side_tape=side_tape, box_class_source=class_source)
    except (SeedError, OSError) as error:
        print(f"side_face_seed: FAILED, planner snapshot left unchanged: {error}",
              file=sys.stderr)
        return 1
    left = seed["faces"]["left"]["tape_centerline"]
    right = seed["faces"]["right"]["tape_centerline"]
    print(f"side_face_seed: {seed['box_class']} box, height {seed['box_height_m'] * 1000:.2f} mm, "
          f"seam-to-edge {left['seam_point_to_edge_m'] * 1000:.2f}/"
          f"{right['seam_point_to_edge_m'] * 1000:.2f} mm, left.right dot "
          f"{seed['diagnostics']['left_right_normal_dot']:.4f}, corner angles "
          + "/".join(f"{seed['diagnostics']['corner_angles_deg'][name]:.1f}" for name in RING)
          + f" deg, wall offset along/across seam "
          f"{seed['wall_offsets_m']['along_seam'] * 1000:.2f}/"
          f"{seed['wall_offsets_m']['across_seam'] * 1000:.2f} mm"
          + (f" [{','.join(seed['flags'])}]" if seed["flags"] else "")
          + f" -> {args.planner_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
