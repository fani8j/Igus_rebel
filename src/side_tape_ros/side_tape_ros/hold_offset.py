"""Pure helpers for the sequencer's --hold-start-offset mode (no ROS imports)."""

from __future__ import annotations

import math
import statistics

MAX_SPREAD_MM = 0.3      # p90 - p10 of the measured e_lat
CONVERGED_MM = 0.2       # |e_lat| after applying the trim


def percentile(values, q: float) -> float:
    """Linear-interpolated percentile, q in [0, 100]."""
    v = sorted(float(x) for x in values)
    if not v:
        raise ValueError("no values")
    if len(v) == 1:
        return v[0]
    pos = (len(v) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def median_and_spread(e_lats):
    """(median, p90 - p10) of e_lat samples."""
    return statistics.median(e_lats), percentile(e_lats, 90) - percentile(e_lats, 10)


def plan_hold_trim(e_lats, current_trim_mm: float, min_frames: int, max_spread_mm: float = MAX_SPREAD_MM):
    """Return (new_trim_mm, median, spread). Raises ValueError to refuse.

    Sign: e_lat' = e_lat - delta_trim, so new trim = current trim + median e_lat.
    """
    if len(e_lats) < max(1, int(min_frames)):
        raise ValueError(f"too few fresh LOCKED frames ({len(e_lats)} < {min_frames})")
    if any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in e_lats):
        raise ValueError("non-finite e_lat sample (NaN/inf) in the measurement")
    med, spread = median_and_spread(e_lats)
    if not spread <= max_spread_mm:  # written so that a NaN spread refuses too
        raise ValueError(f"e_lat spread p90-p10 = {spread:.3f} mm > {max_spread_mm:.2f} mm (unstable measurement)")
    return float(current_trim_mm) + med, med, spread


def check_hold_args(hold_start_offset: bool, mode: str, measure_only: bool = False, align_only: bool = False):
    """Return an error string, or None if the flag combination is allowed."""
    if not hold_start_offset:
        return None
    if mode != "in_material" or align_only:
        return "--hold-start-offset only works with --mode in_material (not air_align/--align-only)"
    if measure_only:
        return "--hold-start-offset cannot be combined with --measure-only"
    return None
