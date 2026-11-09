"""The perception snapshot (box_h_cut_snapshot/v1 JSON): summary, freshness, and the detection overlay lookup.

No ROS.  Freshness follows the coordinator: the snapshot is as old as the OLDER of its file mtime and its
``acquisition_stamp_ns`` (the larger age), and it must be younger than ``external_snapshot_max_age_s``.
"""
import glob
import json
import logging
import math
import os
import threading
from typing import Callable, List, Optional, Tuple

from .placement import PlacementLimits, assess_placement, xyz

LOG = logging.getLogger("dual_cut_ui")

CORNER_KEYS = ("front_left", "front_right", "back_right", "back_left")     # polygon order
POINT_KEYS = ("seam_start", "seam_end", "seam_center") + CORNER_KEYS


class SnapshotError(Exception):
    pass


def load_snapshot(path: str) -> Tuple[dict, bytes, float]:
    """(document, raw bytes, mtime) of the snapshot file; SnapshotError if missing / unreadable / not an object."""
    try:
        st = os.stat(path)
        with open(path, "rb") as f:
            raw = f.read()
    except OSError as e:
        raise SnapshotError(f"cannot read snapshot {path}: {e.strerror or e}") from e
    try:
        doc = json.loads(raw)
    except ValueError as e:
        raise SnapshotError(f"snapshot {path} is not valid JSON: {e}") from e
    if not isinstance(doc, dict):
        raise SnapshotError(f"snapshot {path} is not a JSON object")
    return doc, raw, st.st_mtime


def _finite(v) -> Optional[float]:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None


def summarize(doc: dict, mtime: float, now_wall: float, max_age_s: float, limits: Optional[PlacementLimits] = None) -> dict:
    """Plain-JSON summary of a loaded snapshot (ages at ``now_wall``, freshness against ``max_age_s``)."""
    file_age = max(0.0, now_wall - mtime)
    stamp = doc.get("acquisition_stamp_ns")
    acq_age = None
    if isinstance(stamp, int) and not isinstance(stamp, bool) and stamp > 0:
        acq_age = now_wall - stamp * 1e-9
    problems: List[str] = []
    if acq_age is None:
        problems.append("no valid acquisition_stamp_ns")
        age = None
    elif acq_age < -1.0:
        problems.append("acquisition stamp is in the future")
        age = None
    else:
        age = max(file_age, max(0.0, acq_age))
    fresh = age is not None and age <= max_age_s
    points = {k: xyz(doc.get(k)) for k in POINT_KEYS}
    try:
        placement = assess_placement(doc, limits)      # info only: a failure here must never cost the summary (or block a step)
    except Exception:  # noqa: BLE001
        LOG.exception("placement check failed (information only, ignored)")
        placement = None
    cp = doc.get("cut_permission")
    cut_permission = None
    if isinstance(cp, dict):
        reasons = []
        for r in cp.get("reasons") or []:
            reasons.append({"code": str(r.get("code")), "detail": r.get("detail")} if isinstance(r, dict) else {"code": str(r)})
        cut_permission = {"status": cp.get("status"), "reasons": reasons, "override": cp.get("override"),
                          "flags": [str(f) for f in cp.get("flags") or []]}
    return {
        "snapshot_id": doc.get("snapshot_id"), "capture_id": doc.get("capture_id"), "frame_id": doc.get("frame_id"),
        "detector_method": doc.get("detector_method"), "plane_rms_m": _finite(doc.get("plane_rms_m")),
        "age_s": age, "file_age_s": file_age, "acquisition_age_s": acq_age,
        "max_age_s": max_age_s, "remaining_s": None if age is None else max_age_s - age, "fresh": fresh,
        "problems": problems, "points": {k: (list(v) if v else None) for k, v in points.items()},
        "cut_permission": cut_permission, "placement": placement,
    }


class SnapshotWatcher:
    """Reads the snapshot file only when its (mtime, size) changed; the summary is recomputed for the current time."""

    def __init__(self, path: str, wall: Callable[[], float], limits: Optional[PlacementLimits] = None):
        self.path = path
        self._wall = wall
        self._limits = limits
        self._lock = threading.Lock()
        self._sig = None
        self._doc = None
        self._mtime = 0.0
        self._error = None

    def _refresh(self) -> None:
        try:
            st = os.stat(self.path)
            sig = (st.st_mtime_ns, st.st_size)
        except OSError as e:
            self._sig, self._doc, self._error = None, None, f"cannot read snapshot {self.path}: {e.strerror or e}"
            return
        if sig == self._sig and self._doc is not None:
            return
        try:
            doc, _raw, mtime = load_snapshot(self.path)
        except SnapshotError as e:
            self._sig, self._doc, self._error = sig, None, str(e)
            return
        self._sig, self._doc, self._mtime, self._error = sig, doc, mtime, None

    def summary(self, max_age_s: float) -> dict:
        """Summary dict; ``available`` False (and ``error`` set) when the file is missing or unreadable."""
        with self._lock:
            self._refresh()
            if self._doc is None:
                return {"available": False, "error": self._error, "path": self.path, "fresh": False, "age_s": None,
                        "max_age_s": max_age_s, "remaining_s": None, "points": {}, "placement": None,
                        "cut_permission": None, "problems": []}
            out = summarize(self._doc, self._mtime, self._wall(), max_age_s, self._limits)
            out.update({"available": True, "error": None, "path": self.path, "signature": list(self._sig)})
            return out


def find_overlay(output_dir: str, snapshot_path: str) -> Optional[dict]:
    """The detection overlay that belongs to the snapshot file.

    The detector writes ``<output_dir>/pipeline/<capture_id>/<run>/planner_snapshot.json`` (byte-identical to the file
    given as --planner-output-json) and the overlay ``<run>/autolabel_v2_overlays/*_autolabel_v2.png`` (the
    ``*_record.png`` next to it is not wanted).  Among several matching runs the newest wins.
    Returns {"overlay": path, "run_dir": path, "mtime": float} or None."""
    try:
        doc, raw, _ = load_snapshot(snapshot_path)
    except SnapshotError:
        return None
    capture_id = doc.get("capture_id")
    if not isinstance(capture_id, str) or not capture_id or "/" in capture_id or capture_id.startswith("."):
        return None
    candidates = []
    for snap in glob.glob(os.path.join(glob.escape(output_dir), "pipeline", glob.escape(capture_id), "*", "planner_snapshot.json")):
        try:
            with open(snap, "rb") as f:
                same = f.read() == raw
            if same:
                candidates.append((os.path.getmtime(snap), os.path.dirname(snap)))
        except OSError:
            continue
    for _mt, run_dir in sorted(candidates, reverse=True):
        overlays = [p for p in glob.glob(os.path.join(glob.escape(run_dir), "autolabel_v2_overlays", "*_autolabel_v2.png"))
                    if not p.endswith("_record.png")]
        if overlays:
            best = max(overlays, key=os.path.getmtime)
            return {"overlay": best, "run_dir": run_dir, "mtime": os.path.getmtime(best)}
    return None
