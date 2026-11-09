"""Parsing of the coordinator's ``plan_report`` / ``exec_report`` JSON strings and cleaning of values for the page.

Defensive on purpose: every key may be missing (a refusal report is short), values may be NaN; the page shows "-" for
anything that is None here.  No ROS.
"""
import json
import math
import re
from typing import Any, List, Optional

_NONFINITE = re.compile(r"(?<![\w\"])-?(?:nan|inf)(?![\w\"])", re.IGNORECASE)


def clean(obj: Any) -> Any:
    """Recursively replace non-finite floats by None (so ``json.dumps(allow_nan=False)`` works)."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    return obj


def loads_lenient(text: str) -> Optional[dict]:
    """The coordinator prints doubles with ostream, so a NaN can appear as nan / -nan / inf: map those to null."""
    for candidate in (text, _NONFINITE.sub("null", text) if isinstance(text, str) else text):
        try:
            doc = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        return doc if isinstance(doc, dict) else None
    return None


def _num(v) -> Optional[float]:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None


def _pair(v) -> Optional[List[float]]:
    if isinstance(v, (list, tuple)) and len(v) >= 2 and _num(v[0]) is not None and _num(v[1]) is not None:
        return [float(v[0]), float(v[1])]
    return None


def _get(d, *keys):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def phase_duration(mode, serial_first, left, right):
    """(duration_s, how) of one phase.  parallel: both arms at once -> max.  serial ("serial", the first arm moves while
    the other is held, then the second): left + right.  Anything else is unknown -> the conservative sum."""
    known = [d for d in (left, right) if d is not None]
    if not known:
        return None, "unknown"
    m = mode.strip().lower() if isinstance(mode, str) else ""
    if m.startswith("parallel"):
        return max(known), "parallel"
    if m.startswith("serial") or serial_first:
        return sum(known), "serial"
    return sum(known), "assumed serial (unknown mode)"


def _phase_summary(name: str, p: Any) -> dict:
    p = p if isinstance(p, dict) else {}
    left, right = _num(p.get("duration_left_s")), _num(p.get("duration_right_s"))
    dur, how = phase_duration(p.get("mode"), p.get("serial_first") or None, left, right)
    return {"name": name, "ok": p.get("ok") if isinstance(p.get("ok"), bool) else None, "mode": p.get("mode"),
            "serial_first": p.get("serial_first") or None, "message": p.get("message"),
            "duration_left_s": left, "duration_right_s": right, "duration_s": dur, "duration_how": how,
            "min_cross_effective_m": _num(p.get("min_cross_effective_m"))}


def parse_plan_report(text: str) -> dict:
    """Summary of one ~/plan_report message.  ``parsed`` False if the text is not a JSON object."""
    doc = loads_lenient(text)
    if doc is None:
        return {"parsed": False, "ok": None, "message": "plan report is not valid JSON"}
    phases = []
    raw_phases = doc.get("phases") if isinstance(doc.get("phases"), dict) else {}
    for name in sorted(raw_phases):
        phases.append(_phase_summary(name, raw_phases[name]))
    mode = doc.get("schedule_mode") if doc.get("schedule_mode") in ("phases", "sequence") else ("phases" if phases else None)
    seq = doc.get("sequence") if isinstance(doc.get("sequence"), dict) else {}
    chosen = seq.get("chosen") if isinstance(seq.get("chosen"), dict) else {}
    fallback = seq.get("fallback_to_phases") is True
    if mode == "sequence" and not fallback and _num(chosen.get("makespan_s")) is not None:
        makespan, makespan_kind = _num(chosen.get("makespan_s")), "sequence"
    else:
        durs = [p["duration_s"] for p in phases if p["duration_s"] is not None]
        makespan, makespan_kind = (sum(durs), "est.") if durs else (None, None)
    crosses = [p["min_cross_effective_m"] for p in phases if p["min_cross_effective_m"] is not None]
    min_cross = _num(chosen.get("min_cross_window_m")) if mode == "sequence" and not fallback else None
    if min_cross is None and crosses:
        min_cross = min(crosses)
    cov = doc.get("top_coverage") if isinstance(doc.get("top_coverage"), dict) else None
    coverage = None
    if cov is not None:
        coverage = {"seam_length_mm": _num(cov.get("seam_length_mm")), "left_mm": _pair(cov.get("left_mm")),
                    "right_mm": _pair(cov.get("right_mm")), "left_fraction": _pair(cov.get("left_fraction")),
                    "right_fraction": _pair(cov.get("right_fraction")), "overlap_mm": _num(cov.get("overlap_mm"))}
    ts = doc.get("top_split") if isinstance(doc.get("top_split"), dict) else {}
    snap = doc.get("snapshot") if isinstance(doc.get("snapshot"), dict) else {}
    plans = {}
    for arm in ("left", "right"):
        per = _get(doc, "plans", arm)
        if isinstance(per, dict):
            plans[arm] = {k: {"pattern": v.get("pattern"), "duration_s": _num(v.get("duration_s")),
                              "n_segments": len(v["segments"]) if isinstance(v.get("segments"), list) else None}
                          for k, v in per.items() if isinstance(v, dict)}
    sequence = None
    if seq:
        sequence = {"fallback_to_phases": fallback, "delayed_arm": chosen.get("delayed_arm") or None,
                    "delay_s": _num(chosen.get("delay_s")), "makespan_s": _num(chosen.get("makespan_s")),
                    "min_cross_window_m": _num(chosen.get("min_cross_window_m")),
                    "duration_left_s": _num(seq.get("duration_left_s")), "duration_right_s": _num(seq.get("duration_right_s")),
                    "phases_estimate_lower_bound_s": _num(seq.get("phases_estimate_lower_bound_s"))}
    return {
        "parsed": True, "ok": doc.get("ok") if isinstance(doc.get("ok"), bool) else None, "message": doc.get("message"),
        "schedule_mode": mode, "phase_order": doc.get("phase_order"), "top_split_mode": ts.get("mode"),
        "top_split": {"mode": ts.get("mode"), "seam_length_m": _num(ts.get("seam_length_m")),
                      "left_reach_fraction": _num(ts.get("left_reach_fraction")),
                      "start_fraction": _num(ts.get("start_fraction"))} if ts else None,
        "top_coverage": coverage, "top_coverage_ok": doc.get("top_coverage_ok") if isinstance(doc.get("top_coverage_ok"), bool) else None,
        "phases": phases, "sequence": sequence, "plans": plans,
        "makespan_s": makespan, "makespan_kind": makespan_kind, "min_cross_m": min_cross,
        "snapshot": {"id": snap.get("id"), "capture": snap.get("capture"),
                     "acquisition_stamp_ns": snap.get("acquisition_stamp_ns") if isinstance(snap.get("acquisition_stamp_ns"), int) else None,
                     "age_at_plan_end_s": _num(snap.get("age_at_plan_end_s"))} if snap else None,
        "executor_wall_s": _num(_get(doc, "executor_calls", "wall_s")),
        "detection_blocked": doc.get("detection_blocked") if isinstance(doc.get("detection_blocked"), bool) else None,
    }


def parse_exec_report(text: str) -> dict:
    doc = loads_lenient(text)
    if doc is None:
        return {"parsed": False}
    return {
        "parsed": True, "phase": doc.get("phase"), "schedule": doc.get("schedule"),
        "completed": doc.get("completed") if isinstance(doc.get("completed"), bool) else None, "mode": doc.get("mode"),
        "planned_duration_left_s": _num(doc.get("planned_duration_left_s")),
        "planned_duration_right_s": _num(doc.get("planned_duration_right_s")),
        "wall_s": _num(doc.get("wall_s")),
        "min_cross_distance_during_execution_m": _num(doc.get("min_cross_distance_during_execution_m")),
        "max_tracking_error_rad": _num(doc.get("max_tracking_error_rad")),
        "stop_latched": doc.get("stop_latched") if isinstance(doc.get("stop_latched"), bool) else None,
        "stop_reason": doc.get("stop_reason") or "",
    }
