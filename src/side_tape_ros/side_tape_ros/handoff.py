"""Hand-off record from box_h_cut_executor (``box_h_cut_side_tape_handoff/v1``).

Pure Python (no rclpy) so it is unit-testable. ``load_handoff`` parses and
validates the structure; ``check_handoff`` returns a list of human-readable
problems (empty = OK) and never moves anything.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

SCHEMA = "box_h_cut_side_tape_handoff/v1"
_VARIANTS = ("in_material", "standoff")

__all__ = ["SCHEMA", "HandoffRecord", "load_handoff", "check_handoff", "mode_for_variant"]


@dataclass
class HandoffRecord:
    status: str
    face: str
    variant: str
    cut_length_mm: float
    penetration_mm: float
    standoff_mm: float
    planning_frame: str
    tool_link: str
    position_m: tuple
    orientation_xyzw: tuple
    cut_axis_tool: tuple
    lateral_axis_tool: tuple
    depth_axis_tool: tuple
    left_parallel: dict | None = field(default=None)
    raw: dict = field(default_factory=dict)


def mode_for_variant(variant: str) -> str:
    return "in_material" if variant == "in_material" else "air_align"


def _xyz(d, name):
    try:
        return (float(d["x"]), float(d["y"]), float(d["z"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"handoff: bad or missing vector {name!r}: {exc!r}") from exc


def load_handoff(path) -> HandoffRecord:
    try:
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
    except OSError as exc:
        raise ValueError(f"handoff: cannot read {path!r}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"handoff: {path!r} is not valid JSON: {exc}") from exc
    if not isinstance(d, dict):
        raise ValueError("handoff: top-level JSON is not an object")
    if d.get("schema") != SCHEMA:
        raise ValueError(f"handoff: wrong schema {d.get('schema')!r}, expected {SCHEMA!r}")
    try:
        sp = d["start_pose"]
        conv = d["side_tape_convention"]
        pos = _xyz(sp["position_m"], "start_pose.position_m")
        q = sp["orientation_xyzw"]
        quat = (float(q["x"]), float(q["y"]), float(q["z"]), float(q["w"]))
        lp = d.get("left_parallel_orientation")
        return HandoffRecord(
            status=str(d["status"]), face=str(d["face"]), variant=str(d["variant"]),
            cut_length_mm=float(d["cut_length_mm"]), penetration_mm=float(d["penetration_mm"]),
            standoff_mm=float(d["standoff_mm"]), planning_frame=str(d["planning_frame"]),
            tool_link=str(d["tool_link"]), position_m=pos, orientation_xyzw=quat,
            cut_axis_tool=_xyz(conv["cut_axis_tool"], "cut_axis_tool"),
            lateral_axis_tool=_xyz(conv["lateral_axis_tool"], "lateral_axis_tool"),
            depth_axis_tool=_xyz(conv["depth_axis_tool"], "depth_axis_tool"),
            left_parallel=lp if isinstance(lp, dict) else None, raw=d)
    except KeyError as exc:
        raise ValueError(f"handoff: missing key {exc}") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError(f"handoff: malformed record: {exc}") from exc


def _angle_deg(a, b) -> float:
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na < 1e-9 or nb < 1e-9:
        return 180.0
    c = sum(x * y for x, y in zip(a, b)) / (na * nb)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def _quat_angle_deg(a, b) -> float:
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na < 1e-9 or nb < 1e-9:
        return 180.0
    d = abs(sum(x * y for x, y in zip(a, b))) / (na * nb)
    return math.degrees(2.0 * math.acos(max(-1.0, min(1.0, d))))


def check_handoff(rec: HandoffRecord, cfg_robot, current_pos_m, current_quat_xyzw, *,
                  max_search_deg: float = 10.0, pos_tol_mm: float = 15.0,
                  ori_tol_deg: float = 10.0) -> list:
    """Return the list of problems (empty = OK). Pose checks skipped if current pose is None."""
    p: list = []
    if rec.status != "active":
        p.append(f"status is {rec.status!r}, expected 'active'")
    if rec.variant not in _VARIANTS:
        p.append(f"variant {rec.variant!r} is not one of {_VARIANTS}")
    for name, rv, cv in (("cut_axis_tool", rec.cut_axis_tool, cfg_robot.cut_axis_tool),
                         ("lateral_axis_tool", rec.lateral_axis_tool, cfg_robot.lateral_axis_tool),
                         ("depth_axis_tool", rec.depth_axis_tool, cfg_robot.depth_axis_tool)):
        ang = _angle_deg(rv, [float(x) for x in cv])
        if ang > 5.0:
            p.append(f"{name} in record {tuple(round(x, 3) for x in rv)} differs from config "
                     f"{tuple(round(float(x), 3) for x in cv)} by {ang:.1f} deg (> 5)")
    if rec.planning_frame != "base_link":
        p.append(f"planning_frame is {rec.planning_frame!r}, expected 'base_link'")
    if rec.tool_link != "xeg32_tool_tip":
        p.append(f"tool_link is {rec.tool_link!r}, expected 'xeg32_tool_tip'")
    lp = rec.left_parallel
    if not lp:
        p.append("left_parallel_orientation block is missing")
    else:
        for key in ("max_abs_angle_deg", "rotation_from_f0_deg"):
            try:
                v = float(lp[key])
            except (KeyError, TypeError, ValueError):
                p.append(f"left_parallel_orientation.{key} missing or invalid")
                continue
            if not v <= max_search_deg:
                p.append(f"left_parallel_orientation.{key}={v:.2f} deg exceeds {max_search_deg:.1f}")
    if not rec.cut_length_mm > 0:
        p.append(f"cut_length_mm={rec.cut_length_mm} must be > 0")
    if current_pos_m is not None and current_quat_xyzw is not None:
        dist_mm = 1000.0 * math.dist(current_pos_m, rec.position_m)
        if dist_mm > pos_tol_mm:
            p.append(f"tool is {dist_mm:.1f} mm from the hand-off start pose (> {pos_tol_mm:.1f})")
        ori = _quat_angle_deg(current_quat_xyzw, rec.orientation_xyzw)
        if ori > ori_tol_deg:
            p.append(f"tool orientation differs from the hand-off start pose by {ori:.1f} deg "
                     f"(> {ori_tol_deg:.1f})")
    return p
