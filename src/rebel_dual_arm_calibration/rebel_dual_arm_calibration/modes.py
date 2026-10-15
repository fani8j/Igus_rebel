"""Recorder modes: which arm(s) are recorded and which frames / joint prefixes they use (no ROS imports).

dual         both arms from the dual-arm stack: arm_left_base_link/arm_left_link6 and
             arm_right_base_link/arm_right_link6, joints prefixed arm_left_ / arm_right_.
single_left  today's single-arm stack drives the LEFT arm: base_link/link6, joints joint1..6 (no prefix).
single_right same, but the arm in front of the camera is the RIGHT arm (recorded as arm "right").

In a single_* mode only that arm is recorded; the other arm's TF and joints do not exist and are
never looked up.  The solver needs data of both arms, so record the two arms in two sessions
(possibly with two different bring-ups) and merge them with ``solve <dir1> <dir2> --out ...``.
"""
from __future__ import annotations

from typing import Dict, List

MODES = ("dual", "single_left", "single_right")


def mode_parameters(mode: str) -> Dict[str, object]:
    """Recorder node parameters (active_arms, frames, joint prefixes) for a mode."""
    if mode == "dual":
        return {"active_arms": ["left", "right"],
                "left_joint_prefix": "arm_left_", "right_joint_prefix": "arm_right_",
                "left_base_frame": "arm_left_base_link", "left_tip_frame": "arm_left_link6",
                "right_base_frame": "arm_right_base_link", "right_tip_frame": "arm_right_link6"}
    if mode in ("single_left", "single_right"):
        arm = mode.split("_")[1]
        return {"active_arms": [arm],
                f"{arm}_joint_prefix": "",
                f"{arm}_base_frame": "base_link", f"{arm}_tip_frame": "link6"}
    raise ValueError(f"unknown mode '{mode}', expected one of {', '.join(MODES)}")


def active_arms_of(value) -> List[str]:
    arms = [str(a) for a in value]
    bad = [a for a in arms if a not in ("left", "right")]
    if bad or not arms:
        raise ValueError(f"active_arms must be a non-empty subset of ['left', 'right'], got {list(value)}")
    return arms
