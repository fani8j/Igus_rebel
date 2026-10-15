"""Capture-quality helpers (numpy only, no ROS).

Encoder stillness does not prove that the printed structure is still: a target on a flexible bracket can keep
vibrating after the joints stopped.  The recorder therefore also compares the detected corners of two images that
are >= ``min_gap_s`` apart (both after the stationary start).

Corner detection noise (about 0.14 px per frame at 1920x1080) makes the MAX displacement of any single corner of a
perfectly still board about 0.5 px over 20 corners, so a per-corner limit of 0.2 px rejects everything.  Real sway
moves all corners coherently, noise does not.  The check therefore fits a least-squares 2D similarity transform
(rotation, uniform scale, translation; closed-form Umeyama) from the earlier to the newer corners and judges the
largest displacement the fitted transform produces at the corners ("common motion", limit ``max_corner_motion_px``).
A gross single-corner displacement (shaking, a mis-detected corner) is limited separately
(``max_corner_single_px``).  Either one rejects the capture ("board still moving ...")."""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np


def _common_corners(points_a: Sequence, points_b: Sequence):
    a = {int(p[0]): np.asarray(p[2], float) for p in points_a}
    b = {int(p[0]): np.asarray(p[2], float) for p in points_b}
    common = sorted(set(a) & set(b))
    return np.array([a[k] for k in common]).reshape(-1, 2), np.array([b[k] for k in common]).reshape(-1, 2)


def corner_motion_px(points_a: Sequence, points_b: Sequence) -> Tuple[Optional[float], int]:
    """(max displacement of any single common corner in px, number of common corners).

    ``points_*`` are the ``Detection.points`` lists ``(id, obj_xyz, (u, v))``; None if there is no common corner."""
    pa, pb = _common_corners(points_a, points_b)
    if len(pa) == 0:
        return None, 0
    return float(np.linalg.norm(pa - pb, axis=1).max()), len(pa)


def corner_motion_stats(points_prev: Sequence, points_new: Sequence) -> Tuple[Optional[float], Optional[float], int]:
    """(coherent motion px, single-corner max px, number of common corners).

    Coherent motion = max over the common corners of ``|T(p_prev) - p_prev|`` where T is the least-squares 2D
    similarity transform (rotation, uniform scale, translation) from the earlier to the newer corners.  Written with
    complex numbers: ``z_new = c (z_prev - mean_prev) + mean_new``, ``c = sum(conj(a) b) / sum(|a|^2)`` over the
    centred corners (rotation and scale in one complex factor; no reflection).  (None, None, 0) without a common
    corner; with one corner (or all at one point) the fit is a pure translation."""
    pa, pb = _common_corners(points_prev, points_new)
    n = len(pa)
    if n == 0:
        return None, None, 0
    single = float(np.linalg.norm(pb - pa, axis=1).max())
    ma, mb = pa.mean(axis=0), pb.mean(axis=0)
    a = (pa - ma) @ np.array([1.0, 1j])
    b = (pb - mb) @ np.array([1.0, 1j])
    denom = float(np.sum(np.abs(a) ** 2))
    c = np.sum(np.conj(a) * b) / denom if denom > 1e-12 else 1.0 + 0j
    shift = complex(*(mb - ma))
    disp = (c - 1.0) * a + shift                       # T(p_prev) - p_prev at every common corner
    return float(np.abs(disp).max()), single, n


def pick_previous_frame(history: Sequence[Tuple[float, object]], stamp: float, since: float,
                        min_gap_s: float) -> Optional[Tuple[float, object]]:
    """The newest history entry (stamp, message) that is >= ``min_gap_s`` older than ``stamp`` and not older than
    ``since`` (the start of the stationary window), or None."""
    best = None
    for t, m in history:
        if t >= since and t <= stamp - min_gap_s and (best is None or t > best[0]):
            best = (t, m)
    return best


def judge_motion(coherent: Optional[float], single: Optional[float], n_common: int, limit_px: float,
                 single_limit_px: float, min_common: int) -> Tuple[bool, str]:
    """(ok, text) for one target: ok = enough common corners, the coherent motion within ``limit_px`` and no single
    corner farther than ``single_limit_px``."""
    if coherent is None or single is None or n_common < min_common:
        return False, f"cannot verify stillness: only {n_common} common corner(s) in the earlier frame (< {min_common})"
    if coherent > limit_px:
        return False, (f"board still moving (common {coherent:.2f} px > {limit_px:.2f} px, single max {single:.2f} px "
                       f"over {n_common} corners)")
    if single > single_limit_px:
        return False, (f"board still moving (single corner {single:.2f} px > {single_limit_px:.2f} px, common "
                       f"{coherent:.2f} px over {n_common} corners)")
    return True, f"still (common {coherent:.2f} px, single max {single:.2f} px over {n_common} corners)"
