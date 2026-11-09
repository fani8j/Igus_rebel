#!/usr/bin/env python3
"""Analyse rosbag2 recordings of tools/a2_ab_test.py (rosbag2_py, no extra dependencies).

    python3 tools/a2_ab_analyze.py BAG_DIR [BAG_DIR ...]

Per run and per joint, from /<driver>/cri_diagnostics: peak and RMS |current_raw|, peak
|jog_percent_sent|, jog sign changes per second and the fraction of frames with 0 < |jog| < 1 %,
separately for joints that hold still (start == goal) and joints that move. Also reports any
overcurrent / hardware error seen in /rosout or in hardware_state, and the A2 (joint2) current in
the last second before the first trip.

The motion window comes from <BAG_DIR>.summary.json (written by a2_ab_test.py) when present;
--all uses the whole bag. current_raw is the raw CURRENTJOINTS value of the CRI STATUS.
"""

import argparse
import json
import math
import os
import re
import sys

N_JOINTS = 6
HOLD_TOL_DEG = 0.01
SMALL_JOG_PERCENT = 1.0
TRIP_WINDOW_S = 1.0
ROSOUT_ERROR = 40  # rcl_interfaces/Log ERROR
TRIP_PATTERN = re.compile(
    r"over.?current|latched|hardware error|robot controller reports a stop|emergency|e-?stop|"
    r"motor not enabled|kinematics error|fault", re.IGNORECASE)


# ----------------------------------------------------------------------------------------------
# Pure functions (unit-tested in tools/test_a2_ab.py)
# ----------------------------------------------------------------------------------------------
def sign_changes(values):
    """Number of sign flips between consecutive NON-ZERO values (zeros are skipped)."""
    flips = 0
    previous = 0
    for value in values:
        if value == 0 or not math.isfinite(value):
            continue
        sign = 1 if value > 0 else -1
        if previous and sign != previous:
            flips += 1
        previous = sign
    return flips


def small_jog_fraction(jogs, limit=SMALL_JOG_PERCENT):
    """Fraction of values with 0 < |jog| < limit; 0.0 for an empty list."""
    if not jogs:
        return 0.0
    return sum(1 for v in jogs if 0.0 < abs(v) < limit) / len(jogs)


def rms(values):
    return math.sqrt(sum(v * v for v in values) / len(values)) if values else 0.0


def joint_stats(times, currents, jogs):
    """Statistics of one joint over a window. times in seconds (same length as the others)."""
    n = len(times)
    duration = (times[-1] - times[0]) if n > 1 else 0.0
    return {
        "frames": n,
        "duration_s": duration,
        "peak_current": max((abs(c) for c in currents), default=0),
        "rms_current": rms(currents),
        "peak_jog": max((abs(j) for j in jogs), default=0.0),
        "sign_changes": sign_changes(jogs),
        "sign_changes_per_s": (sign_changes(jogs) / duration) if duration > 0 else 0.0,
        "small_jog_fraction": small_jog_fraction(jogs),
    }


def classify_joints(start_deg, goal_deg, tolerance=HOLD_TOL_DEG):
    """'holding' where start == goal (within tolerance), else 'moving', per joint."""
    return ["holding" if abs(b - a) <= tolerance else "moving"
            for a, b in zip(start_deg, goal_deg)]


def in_window(times, t_start, t_end):
    """Indices of times inside [t_start, t_end] (None = open end)."""
    return [i for i, t in enumerate(times)
            if (t_start is None or t >= t_start) and (t_end is None or t <= t_end)]


def group_stats(per_joint, groups, wanted):
    """Aggregate of the joints in group `wanted`: max peaks, pooled RMS / fraction, mean flips/s."""
    selected = [i for i, g in enumerate(groups) if g == wanted]
    if not selected:
        return None
    frames = sum(per_joint[i]["frames"] for i in selected)
    pooled_rms = (math.sqrt(sum(per_joint[i]["rms_current"] ** 2 * per_joint[i]["frames"]
                                for i in selected) / frames) if frames else 0.0)
    small = (sum(per_joint[i]["small_jog_fraction"] * per_joint[i]["frames"] for i in selected)
             / frames) if frames else 0.0
    return {
        "joints": [i + 1 for i in selected],
        "peak_current": max(per_joint[i]["peak_current"] for i in selected),
        "rms_current": pooled_rms,
        "peak_jog": max(per_joint[i]["peak_jog"] for i in selected),
        "sign_changes_per_s": sum(per_joint[i]["sign_changes_per_s"] for i in selected)
        / len(selected),
        "small_jog_fraction": small,
    }


def find_trip_events(rosout, hardware_states):
    """Overcurrent / hardware error evidence, sorted by time.

    rosout: list of (t_s, level, name, text); hardware_states: list of
    (t_s, latched, latch_reason, drives_disabled_mask). A rosout line counts if it is ERROR level
    or fatal, or matches the trip pattern; a hardware_state counts while latched or while a drive
    is reported disabled.
    """
    events = []
    for t, level, name, text in rosout:
        if level >= ROSOUT_ERROR or TRIP_PATTERN.search(text or ""):
            events.append({"t": t, "source": "rosout", "level": level, "detail": f"{name}: {text}"})
    seen_latch = False
    seen_mask = 0
    for t, latched, reason, mask in hardware_states:
        if latched and not seen_latch:
            events.append({"t": t, "source": "hardware_state", "level": None,
                           "detail": f"latched: {reason}"})
        seen_latch = seen_latch or latched
        if mask and mask != seen_mask:
            events.append({"t": t, "source": "hardware_state", "level": None,
                           "detail": f"drives_disabled_mask=0x{mask:02x}"})
        seen_mask = mask
    events.sort(key=lambda e: e["t"])
    return events


def first_trip_time(events):
    return events[0]["t"] if events else None


def window_before(times, values, t_trip, window_s=TRIP_WINDOW_S):
    """Values with t_trip - window_s <= t <= t_trip."""
    return [v for t, v in zip(times, values) if t_trip - window_s <= t <= t_trip]


def trip_window_summary(times, currents, jogs, t_trip, window_s=TRIP_WINDOW_S):
    """A2 current / jog in the last window_s before a trip (None if no trip)."""
    if t_trip is None:
        return None
    cur = window_before(times, currents, t_trip, window_s)
    jog = window_before(times, jogs, t_trip, window_s)
    return {
        "frames": len(cur),
        "peak_current": max((abs(c) for c in cur), default=0),
        "rms_current": rms(cur),
        "mean_current": (sum(cur) / len(cur)) if cur else 0.0,
        "peak_jog": max((abs(j) for j in jog), default=0.0),
    }


def extract_diagnostics(messages):
    """messages: objects shaped like igus_rebel_msgs/CriDiagnostics -> dict of lists."""
    out = {"t": [], "current": [[] for _ in range(N_JOINTS)], "jog": [[] for _ in range(N_JOINTS)],
           "position_deg": [[] for _ in range(N_JOINTS)], "source": []}
    for msg in messages:
        out["t"].append(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
        for i in range(N_JOINTS):
            out["current"][i].append(int(msg.current_raw[i]))
            out["jog"][i].append(float(msg.jog_percent_sent[i]))
            out["position_deg"][i].append(float(msg.position_deg[i]))
        out["source"].append(msg.jog_source)
    return out


def analyse_run(diag, groups, t_start=None, t_end=None, trip_events=()):
    """All statistics of one run. groups: 'holding'/'moving' per joint."""
    idx = in_window(diag["t"], t_start, t_end)
    times = [diag["t"][i] for i in idx]
    per_joint = []
    for j in range(N_JOINTS):
        per_joint.append(joint_stats(
            times, [diag["current"][j][i] for i in idx], [diag["jog"][j][i] for i in idx]))
    trip_t = first_trip_time(trip_events)
    # The trip window is looked at over the whole recording, not just the motion window.
    trip_a2 = trip_window_summary(diag["t"], diag["current"][1], diag["jog"][1], trip_t)
    return {
        "per_joint": per_joint,
        "groups": groups,
        "holding": group_stats(per_joint, groups, "holding"),
        "moving": group_stats(per_joint, groups, "moving"),
        "trip_time": trip_t,
        "trip_events": list(trip_events),
        "trip_a2": trip_a2,
        "frames": len(idx),
        "sources": sorted({diag["source"][i] for i in idx}),
    }


def format_report(name, summary, analysis):
    lines = [f"=== {name}"]
    if summary:
        result = summary.get("result", {})
        lines.append(
            f"label={summary.get('label')} mode={summary.get('mode')} "
            f"deadband={summary.get('deadband_percent')} success={result.get('success')} "
            f"error={result.get('error_string') or result.get('message') or ''!r}")
    lines.append(f"diagnostic frames in window: {analysis['frames']} "
                 f"(jog sources: {', '.join(analysis['sources']) or '-'})")
    lines.append(f"{'joint':>5} {'group':>8} {'peak|I|':>9} {'rms|I|':>9} {'peak|jog|%':>11} "
                 f"{'flips/s':>8} {'0<|jog|<1%':>11}")
    for j, stats in enumerate(analysis["per_joint"]):
        lines.append(
            f"{j + 1:>5} {analysis['groups'][j]:>8} {stats['peak_current']:>9} "
            f"{stats['rms_current']:>9.1f} {stats['peak_jog']:>11.3f} "
            f"{stats['sign_changes_per_s']:>8.2f} {stats['small_jog_fraction']:>11.3f}")
    for key in ("holding", "moving"):
        g = analysis[key]
        if g:
            lines.append(
                f"  {key:>8} joints {g['joints']}: peak|I|={g['peak_current']} "
                f"rms|I|={g['rms_current']:.1f} peak|jog|={g['peak_jog']:.3f}% "
                f"flips/s(mean)={g['sign_changes_per_s']:.2f} small-jog frac={g['small_jog_fraction']:.3f}")
    if analysis["trip_events"]:
        lines.append(f"TRIP / ERROR evidence ({len(analysis['trip_events'])} events), first at "
                     f"t={analysis['trip_time']:.3f}:")
        for event in analysis["trip_events"][:8]:
            lines.append(f"  t={event['t']:.3f} [{event['source']}] {event['detail'][:160]}")
        a2 = analysis["trip_a2"]
        if a2:
            lines.append(
                f"  A2 (joint2) in the last {TRIP_WINDOW_S:.0f} s before the first event: "
                f"{a2['frames']} frames, peak|I|={a2['peak_current']} rms|I|={a2['rms_current']:.1f} "
                f"mean I={a2['mean_current']:.1f} peak|jog|={a2['peak_jog']:.3f}%")
    else:
        lines.append("no overcurrent / hardware error in /rosout or hardware_state")
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------
# Bag reading (needs rosbag2_py)
# ----------------------------------------------------------------------------------------------
def read_bag(bag_dir):
    """{topic: [(bag_t_ns, message), ...]} of every topic in the bag."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    storage_id = "sqlite3"
    metadata_path = os.path.join(bag_dir, "metadata.yaml")
    if os.path.exists(metadata_path):
        with open(metadata_path) as handle:
            for line in handle:
                if "storage_identifier" in line:
                    storage_id = line.split(":", 1)[1].strip()
                    break
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag_dir, storage_id=storage_id),
                rosbag2_py.ConverterOptions("", ""))
    types = {t.name: get_message(t.type) for t in reader.get_all_topics_and_types()}
    data = {name: [] for name in types}
    while reader.has_next():
        topic, raw, t_ns = reader.read_next()
        data[topic].append((t_ns, deserialize_message(raw, types[topic])))
    return data


def find_topic(data, suffix):
    for name in data:
        if name.endswith(suffix):
            return name
    return None


def load_summary(bag_dir):
    base = bag_dir.rstrip("/")
    for path in (base + ".summary.json", os.path.join(base, "summary.json")):
        if os.path.exists(path):
            with open(path) as handle:
                return json.load(handle)
    return None


def analyse_bag(bag_dir, use_window=True):
    data = read_bag(bag_dir)
    diag_topic = find_topic(data, "/cri_diagnostics")
    if diag_topic is None or not data[diag_topic]:
        raise RuntimeError(f"{bag_dir}: no /cri_diagnostics messages (driver without diagnostics?)")
    diag = extract_diagnostics([m for _, m in data[diag_topic]])
    summary = load_summary(bag_dir)
    t_start = t_end = None
    if summary and use_window and "motion_start_ns" in summary:
        t_start = summary["motion_start_ns"] * 1e-9
        t_end = summary.get("motion_end_ns", summary["motion_start_ns"]) * 1e-9
    if summary and "start_deg" in summary and "goal_deg" in summary:
        groups = classify_joints(summary["start_deg"], summary["goal_deg"])
    else:
        # No summary: a joint that moved by more than 0.5 deg in the window counts as moving.
        idx = in_window(diag["t"], t_start, t_end) or [0]
        groups = ["moving" if abs(diag["position_deg"][j][idx[-1]] - diag["position_deg"][j][idx[0]]) > 0.5
                  else "holding" for j in range(N_JOINTS)]
    rosout = []
    topic = find_topic(data, "rosout")
    for _, msg in data.get(topic, []) if topic else []:
        rosout.append((msg.stamp.sec + msg.stamp.nanosec * 1e-9, int(msg.level), msg.name, msg.msg))
    hardware = []
    topic = find_topic(data, "/hardware_state")
    for _, msg in data.get(topic, []) if topic else []:
        hardware.append((msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9, bool(msg.latched),
                         msg.latch_reason, int(msg.drives_disabled_mask)))
    events = find_trip_events(rosout, hardware)
    return summary, analyse_run(diag, groups, t_start, t_end, events)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("bags", nargs="+", metavar="BAG_DIR")
    parser.add_argument("--all", action="store_true", help="whole bag, not only the motion window")
    args = parser.parse_args(argv)
    code = 0
    for bag in args.bags:
        try:
            summary, analysis = analyse_bag(bag, use_window=not args.all)
        except Exception as exc:  # noqa: BLE001 - report and continue with the next bag
            print(f"=== {bag}\nERROR: {exc}")
            code = 1
            continue
        print(format_report(os.path.basename(bag.rstrip("/")), summary, analysis))
    return code


if __name__ == "__main__":
    sys.exit(main())
