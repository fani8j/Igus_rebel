"""Pure helpers for ``sequencer --entry`` (no rclpy): arg rules, offset trim, the result record.

ENTRY workflow (revised 2026-09-30 after the first real-robot test; motion and plan in
precision_box_detector.side_tape.insert_move): the MoveIt executor has put the tip ``standoff_mm`` in
front of the box side face (hand-off variant ``standoff``). Vision is not used that far away.

    approach (TF only)  standoff - pre_contact  ->  contact step  pre_contact + contact_margin
    measure e_lat AT CONTACT (published standoff 0)  ->  |e_lat| > tol: back off, lateral fix, re-contact
    within tol: penetrate ``penetration_mm`` along +depth (lateral held on TF), no vision afterwards

Contact is NOT sensed: it is the hand-off record's standoff_mm (measured by the static camera) plus
``--contact-margin-mm``. The contact accuracy is therefore the static camera's standoff accuracy, and a
depth error also shows up in e_lat at contact (~0.8 mm per mm of depth behind/before the face plane).
It never cuts and never retracts. The outcome goes to ``~/.ros/side_tape_entry.json``.

Offset trim. Perception's target = rim + effective_offset, effective_offset = config offset_mm +
parallax_mm_per_mm * standoff + trim (tracker.py). To make the target the rim + ``entry_offset``:

    trim = entry_offset - config offset_mm           (published explicitly on /side_tape/offset_trim_mm)
    => effective_offset = entry_offset + parallax * standoff

so at standoff 0 (published for the contact measurement) the effective offset is exactly ``entry_offset``.

``--calibrate-rim-contact`` (:func:`rim_contact_offset`): with the tip at contact and visually exactly on
the rim, e_lat (trim 0, standoff 0) says how far the configured offset is from that datum; the
``target.offset_mm`` that makes e_lat 0 is ``config offset + median e_lat`` (printed, never written).
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from datetime import datetime, timezone

SCHEMA = "side_tape_entry/v1"
DEFAULT_RECORD_PATH = "~/.ros/side_tape_entry.json"
STATUSES = ("ok", "out_of_tolerance", "failed")

DEFAULTS = {
    "penetration_mm": 4.8,  # operator-confirmed cut depth past contact
    "entry_tol_mm": 0.5,
    "entry_offset_mm": 0.0,
    "insert_speed_mm_s": 2.0,
    "max_retries": 2,
    "pre_contact_mm": 2.0,
    "contact_margin_mm": 0.0,
    "backoff_mm": 1.0,
    "max_standoff_mm": 15.0,  # contact is not sensed: refuse a standoff above this (a plunge before any measurement)
}
ENTRY_FLAG_NAMES = ("penetration_mm", "entry_tol_mm", "entry_offset_mm", "insert_speed_mm_s", "max_retries",
                    "pre_contact_mm", "contact_margin_mm", "backoff_mm", "max_standoff_mm", "entry_record_path")

# Hand-off pose proximity for --entry (the executor's approach pose is where the TF-only approach starts):
# tighter than the cut flows' 15 mm / 10 deg.
ENTRY_POS_TOL_MM = 5.0
ENTRY_ORI_TOL_DEG = 5.0

MAX_INSERT_SPEED_MM_S = 5.0  # == InsertCfg.max_speed_mm_s (the supervisor enforces it again)


def trim_for_entry(entry_offset_mm: float, cfg_offset_mm: float) -> float:
    """Trim to publish so that the target is the rim + ``entry_offset_mm`` (see module docstring)."""
    return float(entry_offset_mm) - float(cfg_offset_mm)


def effective_offset_mm(cfg_offset_mm: float, parallax_mm_per_mm: float, standoff_mm: float, trim_mm: float) -> float:
    """What perception applies (tracker.py): offset + parallax * standoff + trim."""
    return float(cfg_offset_mm) + float(parallax_mm_per_mm) * float(standoff_mm) + float(trim_mm)


def _num(v, lo, hi, lo_incl=False):
    return (isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
            and (v >= lo if lo_incl else v > lo) and v <= hi)


def apply_entry_defaults(args) -> None:
    for name, default in DEFAULTS.items():
        if getattr(args, name, None) is None:
            setattr(args, name, default)


def check_entry_args(args) -> str | None:
    """Error message for an invalid ``--entry`` combination, else None. ``args`` is the argparse
    namespace BEFORE defaults are applied (entry-only flags are None when not given)."""
    given = [n for n in ENTRY_FLAG_NAMES if getattr(args, n, None) is not None]
    if not getattr(args, "entry", False):
        if given:
            return "--" + given[0].replace("_", "-") + " needs --entry"
        return None
    if getattr(args, "handoff", None) is None:
        return "--entry needs --handoff JSON (the executor's hand-off record, variant 'standoff')"
    for flag, name in (("measure_only", "--measure-only"), ("align_only", "--align-only"),
                       ("line_hold", "--line-hold"), ("hold_start_offset", "--hold-start-offset"),
                       ("ignore_handoff_pose", "--ignore-handoff-pose")):
        if getattr(args, flag, False):
            return f"--entry cannot be combined with {name}"
    if getattr(args, "calibrate_rim_contact", False):
        return "--entry cannot be combined with --calibrate-rim-contact"
    if getattr(args, "line_vision_gain", None) is not None:
        return "--entry cannot be combined with --line-vision-gain"
    if getattr(args, "seed_json", None) is not None:
        return "--entry cannot be combined with --seed-json"
    if getattr(args, "standoff_mm", None) is not None:
        return "--entry takes the standoff from the hand-off record; do not pass --standoff-mm"
    if getattr(args, "cut_length_mm", None) is not None:
        return "--entry does not cut; do not pass --cut-length-mm"
    if getattr(args, "mode", None) not in (None, "air_align"):
        return "--entry needs --mode air_align (or no --mode)"
    rec = getattr(args, "handoff_rec", None)
    if rec is not None and rec.variant != "standoff":
        return f"--entry needs hand-off variant 'standoff', got {rec.variant!r}"
    a = {n: (getattr(args, n) if getattr(args, n, None) is not None else DEFAULTS[n]) for n in DEFAULTS}
    if not _num(a["penetration_mm"], 0.0, 10.0):
        return f"--penetration-mm must be in (0, 10], got {a['penetration_mm']!r}"
    if not _num(a["entry_tol_mm"], 0.0, 2.0):
        return f"--entry-tol-mm must be in (0, 2], got {a['entry_tol_mm']!r}"
    if not _num(a["entry_offset_mm"], -5.0, 5.0, lo_incl=True):
        return f"--entry-offset-mm must be in [-5, 5], got {a['entry_offset_mm']!r}"
    if not _num(a["insert_speed_mm_s"], 0.0, MAX_INSERT_SPEED_MM_S):
        return f"--insert-speed-mm-s must be in (0, {MAX_INSERT_SPEED_MM_S}], got {a['insert_speed_mm_s']!r}"
    r = a["max_retries"]
    if isinstance(r, bool) or not isinstance(r, int) or not 0 <= r <= 5:
        return f"--max-retries must be an integer in [0, 5], got {r!r}"
    if not _num(a["pre_contact_mm"], 0.5, 10.0, lo_incl=True):
        return f"--pre-contact-mm must be in [0.5, 10], got {a['pre_contact_mm']!r}"
    if not _num(a["contact_margin_mm"], -2.0, 3.0, lo_incl=True):
        return f"--contact-margin-mm must be in [-2, 3], got {a['contact_margin_mm']!r}"
    if not _num(a["backoff_mm"], 0.0, 3.0, lo_incl=True):
        return f"--backoff-mm must be in [0, 3], got {a['backoff_mm']!r}"
    if not _num(a["max_standoff_mm"], 0.0, 25.0):
        return f"--max-standoff-mm must be in (0, 25], got {a['max_standoff_mm']!r}"
    rec = getattr(args, "handoff_rec", None)
    if rec is not None and not (rec.standoff_mm <= a["max_standoff_mm"]):
        return (f"hand-off standoff_mm={rec.standoff_mm} exceeds --max-standoff-mm {a['max_standoff_mm']} (contact "
                f"is not sensed: a wrong standoff plunges the blade before any measurement); raise the limit "
                f"deliberately or fix the hand-off")
    return None


def check_calibrate_args(args) -> str | None:
    """Error for an invalid ``--calibrate-rim-contact`` combination (no motion, no service calls)."""
    if not getattr(args, "calibrate_rim_contact", False):
        return None
    for flag, name in (("entry", "--entry"), ("measure_only", "--measure-only"), ("align_only", "--align-only"),
                       ("line_hold", "--line-hold"), ("hold_start_offset", "--hold-start-offset")):
        if getattr(args, flag, False):
            return f"--calibrate-rim-contact cannot be combined with {name}"
    for flag, name in (("handoff", "--handoff"), ("seed_json", "--seed-json"), ("cut_length_mm", "--cut-length-mm"),
                       ("standoff_mm", "--standoff-mm"), ("line_vision_gain", "--line-vision-gain"),
                       ("mode", "--mode")):
        if getattr(args, flag, None) is not None:
            return f"--calibrate-rim-contact cannot be combined with {name}"
    return None


def rim_contact_offset(cfg_offset_mm: float, e_lats, min_frames: int, max_spread_mm: float | None = None):
    """(new ``target.offset_mm``, median e_lat, spread) from e_lat samples taken AT CONTACT with the tip
    on the rim, trim 0 and standoff 0: e_lat' = e_lat - delta_offset, so new offset = config offset + median
    e_lat. Raises ValueError (too few frames / unstable) like the hold-start-offset helper."""
    from . import hold_offset

    kw = {} if max_spread_mm is None else {"max_spread_mm": max_spread_mm}
    trim, med, spread = hold_offset.plan_hold_trim(e_lats, 0.0, min_frames, **kw)
    return float(cfg_offset_mm) + trim, med, spread


def sample_ok(*, stamp_s, now_s, state, has_e_lat, e_lat, conf, has_e_yaw, e_yaw, stale_s, min_conf,
              max_yaw_deg, future_eps_s: float = 0.05):
    """None if one observation may be used for the contact decision / the calibration datum, else the reason.
    Every criterion must hold: LOCKED, finite e_lat, age in [-future_eps, stale_s] (not old, not from the
    future), conf >= min_conf, a finite |e_yaw| <= max_yaw_deg (edge roughly parallel to the cut axis)."""
    def fin(x):
        return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
    if state != "LOCKED":
        return "not_locked"
    if not has_e_lat or not fin(e_lat):
        return "no_e_lat"
    if not fin(stamp_s) or not fin(now_s):
        return "bad_stamp"
    age = now_s - stamp_s
    if age > stale_s:
        return "stale"
    if age < -future_eps_s:
        return "future_stamp"
    if not fin(conf) or conf < min_conf:
        return "low_conf"
    if not has_e_yaw or not fin(e_yaw) or abs(e_yaw) > max_yaw_deg:
        return "bad_yaw"
    return None


def is_stopped(phase: str, v_cut: float, v_lat: float, v_depth: float) -> bool:
    """A status sample that confirms the arm is not being commanded: not in a driving phase, zero twist."""
    if phase in ("ALIGN", "CUT", "INSERT", "SHIFT"):
        return False
    return all(isinstance(v, (int, float)) and math.isfinite(v) and abs(v) <= 1e-9 for v in (v_cut, v_lat, v_depth))


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def build_record(*, status: str, reason: str = "", tool_pose=None, final_e_lat_mm=None, entry_offset_mm: float,
                 penetration_mm: float, standoff_mm=None, tol_mm=None, insert_speed_mm_s=None, iterations: int = 0,
                 max_retries: int = 0, e_lat_history_mm=(), spread_mm=None, handoff_rec=None,
                 base_frame: str = "base_link", time_iso: str | None = None, penetrated: bool = False,
                 pre_contact_mm=None, contact_margin_mm=None, backoff_mm=None) -> dict:
    """The ``side_tape_entry/v1`` record. ``tool_pose`` = ((x, y, z) m, (qx, qy, qz, qw)) in base_frame or None.
    ``final_e_lat_mm`` is the last measurement AT CONTACT (there is no vision after the penetration);
    ``penetrated`` says whether the tip was pushed in (status ok) or left at contact (anything else)."""
    if status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}, got {status!r}")
    pose = None
    if tool_pose is not None:
        (x, y, z), (qx, qy, qz, qw) = tool_pose
        pose = {"frame": base_frame, "position_m": {"x": x, "y": y, "z": z},
                "orientation_xyzw": {"x": qx, "y": qy, "z": qz, "w": qw}}
    raw = getattr(handoff_rec, "raw", None) or {}
    return {
        "schema": SCHEMA,
        "status": status,
        "reason": reason,
        "time_iso": time_iso or iso_now(),
        "handoff_stamp_ns": raw.get("stamp_ns"),
        "handoff_snapshot_id": raw.get("snapshot_id"),
        "tool_pose_base": pose,
        "final_e_lat_mm": None if final_e_lat_mm is None else float(final_e_lat_mm),
        "entry_offset_mm": float(entry_offset_mm),
        "penetration_mm": float(penetration_mm),
        "standoff_mm": None if standoff_mm is None else float(standoff_mm),
        "entry_tol_mm": None if tol_mm is None else float(tol_mm),
        "insert_speed_mm_s": None if insert_speed_mm_s is None else float(insert_speed_mm_s),
        "measured_at": "contact",
        "penetrated": bool(penetrated),
        "pre_contact_mm": None if pre_contact_mm is None else float(pre_contact_mm),
        "contact_margin_mm": None if contact_margin_mm is None else float(contact_margin_mm),
        "backoff_mm": None if backoff_mm is None else float(backoff_mm),
        "iterations": int(iterations),
        "retries_used": max(int(iterations) - 1, 0),
        "max_retries": int(max_retries),
        "e_lat_history_mm": [float(e) for e in e_lat_history_mm],
        "measure_spread_mm": None if spread_mm is None else float(spread_mm),
    }


def write_record_atomic(path: str, record: dict) -> str:
    """Write ``record`` as JSON to ``path`` atomically (temp file in the same directory, fsync, rename):
    a reader never sees a partial file. Returns the expanded path."""
    path = os.path.expanduser(path)
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".side_tape_entry.", suffix=".tmp", dir=d)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=2, sort_keys=False, allow_nan=False)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    try:
        dfd = os.open(d, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except OSError:
        pass
    return path


def summary_lines(record: dict) -> list:
    pose = record.get("tool_pose_base")
    p = pose["position_m"] if pose else None
    return [
        f"status={record['status']}" + (f" reason={record['reason']}" if record.get("reason") else ""),
        "final e_lat AT CONTACT = " + ("n/a" if record.get("final_e_lat_mm") is None
                                       else f"{record['final_e_lat_mm']:+.3f} mm")
        + f" (tol {record.get('entry_tol_mm')} mm), e_lat history {record.get('e_lat_history_mm')}",
        f"entry offset {record['entry_offset_mm']:+.3f} mm, penetration {record['penetration_mm']:.2f} mm "
        f"({'PENETRATED' if record.get('penetrated') else 'NOT penetrated: tip left at/near contact'}), "
        f"contact measurements {record['iterations']} (retries {record['retries_used']}/{record['max_retries']})",
        "tool pose in " + (f"{pose['frame']}: x={p['x']:.4f} y={p['y']:.4f} z={p['z']:.4f} m" if pose else "n/a"),
        "this tool never retracts: the blade stays wherever it is (see the tool pose)",
    ]
