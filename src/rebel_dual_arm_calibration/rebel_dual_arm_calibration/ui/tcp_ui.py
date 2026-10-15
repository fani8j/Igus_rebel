"""TCP pivot calibration tab ("0. TCP") of the calibration UI: ROS-free state machine around ``tcp_pivot.py``.

    TcpManager   checklist -> capture "fit" / "validate" poses -> solve -> mechanical check -> write the result file
                 and show the xacro patch as a text diff.

Safety: this module never moves the robot and never publishes anything.  The operator positions the arm with the MoveIt
Servo teleop keyboard (``rebel_servo_teleop_keyboard``) and lifts / re-seats the knife tip in a fixed shallow countersink
between poses; the tab only READS (joint history through the jog controller, TF through an injected function).  The xacro
file is only READ: the patch is shown as a diff for the operator to review, commit and rebuild by hand.  A capture is
refused unless the whole checklist is ticked, the arm is measured stationary (2.0 s, < 2 mrad), the TF is fresh and the
pose differs from every stored one by at least 15 deg.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import logging
import math
import os
import re
import shutil
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .suggest import SuggestRefused, clean

DEFAULT_XACRO_PATH = "/home/farhan/igus_rebel_ros2/src/igus_rebel_description/urdf/igus_rebel_xeg32_wrapper.urdf.xacro"
TCP_JOINT = "igus_tool0_to_xeg32_tool_tip"
SAMPLES_FILE = "tcp_pivot.jsonl"
RESULT_FILE = "tcp_result_{arm}.yaml"

STANDSTILL_S = 2.0                 # the arm must have been stationary for this long ...
STANDSTILL_MAX_RAD = 0.002         # ... no joint moved more than 2 mrad (max-min) in that time
TF_MAX_AGE_S = 0.5                 # a TF reading older than this is not used
MIN_ROT_DEG = 15.0                 # a new pose must differ from every stored one by at least this rotation
MECH_LENGTH_RANGE_MM = (50.0, 400.0)
GOOD_VERDICTS = ("GOOD", "PASS")
MAX_NOTE_LEN = 200
CHECKLIST_TTL_S = 30 * 60          # a ticked checklist expires after this long without a tick or a capture
MAX_PAGES = 16                     # checklists of at most this many pages are kept (oldest dropped)
ROLES = ("fit", "validate")

CHECKLIST = (
    ("ultrasonic_off", "Ultrasonic power is OFF (generator off, not just idle)."),
    ("fixture", "Fixture clamped at working height in the touch region: hard, shallow (~2 mm deep) 90 deg countersink."),
    ("single_arm", "Only this arm is connected; the other arm is parked."),
    ("estop", "My hand is on the hardware e-stop."),
)

REBASE_WARNING = ("After a TCP change, touches recorded with the old TCP must be re-based (solve --tcp-file ...) "
                  "or re-taken.")
APPLY_STEPS = (
    "Review the diff below; this tab never edits the xacro.",
    "Apply it by hand to igus_rebel_xeg32_wrapper.urdf.xacro (joint igus_tool0_to_xeg32_tool_tip) and commit it.",
    "Rebuild: colcon build --symlink-install --packages-select igus_rebel_description igus_rebel_moveit_config",
    "Restart the robot stack (rebel.launch.py, move_group.launch.py) and this UI so the new TCP is loaded.",
    "Re-base (solve --tcp-file ...) or re-take the touches recorded with the old TCP.",
)
CURRENT_URDF_FLANGE_TO_TIP_MM = 183.4     # what the current URDF implies (0.0527 + 0.130699 m), for the page note
OLD_PIVOT_FLANGE_TO_TIP_MM = 168.0        # an old pivot data set suggested about this: the 15 mm z question is open


class TcpRefused(SuggestRefused):
    """A request that is refused with a clear message (HTTP ``status``)."""


def _default_pivot():
    from .. import tcp_pivot
    return tcp_pivot


def _vec_list(v) -> Optional[List[float]]:
    try:
        a = np.asarray(v, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    return a.tolist() if a.size and np.all(np.isfinite(a)) else None


def _mm(v) -> Optional[List[float]]:
    a = _vec_list(v)
    return None if a is None else [x * 1e3 for x in a]


def _iso(stamp) -> str:
    try:
        return datetime.datetime.fromtimestamp(float(stamp)).isoformat(timespec="seconds")
    except (TypeError, ValueError, OverflowError, OSError):
        return ""


def parse_xacro_r(text: str, joint_name: str = TCP_JOINT) -> Optional[np.ndarray]:
    """The ``origin xyz`` (m) of the TCP joint in the xacro source text; None if it cannot be read as three numbers."""
    m = re.search(r'<joint\b[^>]*\bname="' + re.escape(joint_name) + r'"[^>]*>(.*?)</joint>', text, re.S)
    if not m:
        return None
    o = re.search(r'<origin\b[^>]*\bxyz="([^"]*)"', m.group(1))
    if not o:
        return None
    try:
        v = [float(x) for x in o.group(1).split()]
    except ValueError:
        return None
    return np.array(v) if len(v) == 3 and all(math.isfinite(x) for x in v) else None


DEFAULT_KNIFE_AXIS = np.array([0.005937, 0.094121, 0.130699])    # the URDF r at the time of writing (direction only)


def _axis_angle(axis, angle_rad: float) -> np.ndarray:
    a = np.asarray(axis, dtype=float)
    n = float(np.linalg.norm(a))
    if n < 1e-15 or abs(angle_rad) < 1e-15:
        return np.eye(3)
    a = a / n
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(angle_rad) * K + (1.0 - math.cos(angle_rad)) * (K @ K)


def swing_twist_deg(R_rel: np.ndarray, a: np.ndarray) -> Tuple[float, float]:
    """(tilt, roll) in deg of the rotation ``R_rel`` about the unit axis ``a`` (tool frame): tilt = angle between ``a`` and
    ``R_rel a`` (the lean of the knife axis), roll = the remaining rotation about ``a``.  Guidance numbers for the
    operator, same convention as ``tcp_pivot.coverage``."""
    b = R_rel @ a
    cr = np.cross(a, b)
    tilt = math.atan2(float(np.linalg.norm(cr)), float(a @ b))
    tw = _axis_angle(cr, tilt).T @ R_rel
    v = 0.5 * np.array([tw[2, 1] - tw[1, 2], tw[0, 2] - tw[2, 0], tw[1, 0] - tw[0, 1]])
    roll = math.atan2(float(a @ v), 0.5 * (float(np.trace(tw)) - 1.0))
    return math.degrees(tilt), math.degrees(roll)


def tilt_from_vertical_deg(R: np.ndarray, a: np.ndarray) -> float:
    """Angle between the knife axis ``R a`` (base frame) and straight down (base -z)."""
    d = np.asarray(R, dtype=float) @ a
    return math.degrees(math.atan2(float(np.linalg.norm(np.cross(d, [0.0, 0.0, -1.0]))), float(-d[2])))


def make_tf_source(node):
    """tf_fn for the manager from a rclpy node: (parent, child) -> (T 4x4 | None, age_s | None, reason).  Its own tf2
    listener on the node (the node is spun by the backend's executor); read-only."""
    from rclpy.time import Time
    from tf2_ros import Buffer, TransformListener

    from .jog import quat_to_R
    buf = Buffer()
    TransformListener(buf, node, spin_thread=False)

    def tf(parent: str, child: str):
        try:
            t = buf.lookup_transform(parent, child, Time())
        except Exception as e:  # noqa: BLE001 - reported as a refusal text
            return None, None, f"TF {parent} -> {child} not available: {e}"
        tr, q = t.transform.translation, t.transform.rotation
        T = np.eye(4)
        T[:3, :3] = quat_to_R(q.x, q.y, q.z, q.w)
        T[:3, 3] = [tr.x, tr.y, tr.z]
        age = None
        try:
            stamp = Time.from_msg(t.header.stamp)
            age = (node.get_clock().now() - stamp).nanoseconds * 1e-9
        except Exception:  # noqa: BLE001
            pass
        return T, age, ""
    return tf


class TcpManager:
    """State and actions of the TCP pivot tab (single-arm modes only).

    mode, cfgs     recorder mode; jog.arm_configs(mode)
    jog            JogController-like: status(), joint_window(arms, window_s) -> (q dict, spread rad, covered) | None
    session_dir_fn () -> session directory or None
    tf_fn          (parent, child) -> (T 4x4 | None, age_s | None, reason); None = TF not available (captures refuse)
    pivot          module-like: PivotSample, solve_pivot, duplicate_of, rotation_distance_deg, coverage, POSE_PLAN,
                   write_result_yaml, xacro_patch, mechanical_check (default: ..tcp_pivot)
    xacro_path     the xacro SOURCE file the patch is made against (read only)
    """

    def __init__(self, mode: str, cfgs: Dict[str, Any], jog, session_dir_fn: Callable[[], Optional[str]],
                 tf_fn: Optional[Callable] = None, pivot=None, xacro_path: str = DEFAULT_XACRO_PATH,
                 wall_clock: Callable[[], float] = time.time, urdf_hash_fn: Optional[Callable[[str], str]] = None,
                 standstill_s: float = STANDSTILL_S, standstill_max_rad: float = STANDSTILL_MAX_RAD,
                 tf_max_age_s: float = TF_MAX_AGE_S, min_rot_deg: float = MIN_ROT_DEG,
                 clock: Callable[[], float] = time.monotonic, checklist_ttl_s: float = CHECKLIST_TTL_S):
        self.mode, self.cfgs, self.jog = mode, cfgs, jog
        self.session_dir_fn, self.tf_fn, self._pivot = session_dir_fn, tf_fn, pivot
        self.xacro_path = os.path.expanduser(str(xacro_path or DEFAULT_XACRO_PATH))
        self.wall_clock = wall_clock
        self.urdf_hash_fn = urdf_hash_fn or self._default_hash
        self.standstill_s, self.standstill_max_rad = float(standstill_s), float(standstill_max_rad)
        self.tf_max_age_s, self.min_rot_deg = float(tf_max_age_s), float(min_rot_deg)
        self.log = logging.getLogger("calibration_ui.tcp")
        self.lock = threading.RLock()
        self.clock, self.checklist_ttl_s = clock, float(checklist_ttl_s)
        # the safety checklist is per PAGE (X-Page-Id): a reload (new page id) or a UI restart starts with nothing ticked,
        # and a page's ticks expire after `checklist_ttl_s` without a tick or a capture
        self._pages: Dict[str, Dict[str, Any]] = {}
        self._entries: List[Dict[str, Any]] = []        # {"raw": str, "sample": PivotSample | None}, file order
        self._loaded_path: Optional[str] = None                 # what _entries mirror: the file ...
        self._loaded: Optional[Tuple[str, int, int]] = None     # ... and its (path, mtime_ns, size)
        self._bad_lines = 0
        self._result: Optional[Dict[str, Any]] = None
        self._mech_len: Optional[float] = None
        self._mech: Optional[Dict[str, Any]] = None
        self._written: Optional[Dict[str, Any]] = None
        self._cov_cache: Optional[Tuple[tuple, Dict[str, Any]]] = None
        self._rows_cache: Optional[Tuple[tuple, List[Dict[str, Any]]]] = None

    # ---- helpers ---------------------------------------------------------------------------------
    @property
    def supported(self) -> bool:
        return self.mode in ("single_left", "single_right") and bool(self.cfgs)

    @property
    def arm(self) -> Optional[str]:
        return next(iter(self.cfgs)) if self.supported else None

    def _tp(self):
        if self._pivot is None:
            self._pivot = _default_pivot()
        return self._pivot

    def _need_supported(self) -> str:
        if not self.supported:
            raise TcpRefused("TCP calibration: single-arm modes only", 400)
        return self.arm

    @staticmethod
    def _default_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def _frames(self, arm: str) -> Dict[str, str]:
        p = self.cfgs[arm].joint_prefix
        return {"base": f"{p}base_link", "tool0": f"{p}igus_tool0", "link6": f"{p}link6", "tip": f"{p}xeg32_tool_tip"}

    def _session_dir(self) -> Optional[str]:
        try:
            d = self.session_dir_fn()
        except Exception:  # noqa: BLE001
            d = None
        return os.path.expanduser(d) if d else None

    def _need_session_dir(self) -> str:
        d = self._session_dir()
        if not d:
            raise TcpRefused("session directory unknown (recorder status not available and no session_dir "
                             "parameter): cannot store TCP poses", 503)
        return d

    def _tf(self, parent: str, child: str):
        if self.tf_fn is None:
            return None, None, "TF is not available (the UI has no TF listener)"
        try:
            T, age, why = self.tf_fn(parent, child)
        except Exception as e:  # noqa: BLE001
            return None, None, f"TF {parent} -> {child} failed: {e}"
        if T is None:
            return None, None, why or f"TF {parent} -> {child} not available"
        T = np.asarray(T, dtype=float)
        if T.shape != (4, 4) or not np.all(np.isfinite(T)):
            return None, None, f"TF {parent} -> {child} is not a finite 4x4 transform"
        return T, age, ""

    def _fresh_tf(self, parent: str, child: str) -> np.ndarray:
        T, age, why = self._tf(parent, child)
        if T is None:
            raise TcpRefused(why, 409)
        if age is None or not math.isfinite(age):
            raise TcpRefused(f"TF {parent} -> {child}: its age is unknown, refusing", 409)
        if age > self.tf_max_age_s:
            raise TcpRefused(f"TF {parent} -> {child} is stale ({age:.2f} s old, limit {self.tf_max_age_s:.1f} s): "
                             "is the robot state publisher running?", 409)
        return T

    def _jog_blocked(self) -> str:
        try:
            st = self.jog.status()
        except Exception as e:  # noqa: BLE001
            return f"jog state unknown ({e})"
        if st.get("available") is False:
            return "jog unavailable: " + str(st.get("error", ""))
        if st.get("busy") or st.get("stopping") or st.get("fault"):
            return "the arm is busy (a jog step / move is running, stopping or in fault)"
        return ""

    def _motion(self, arm: str) -> Tuple[Optional[float], str, bool]:
        """(spread rad over the window | None, problem text, stationary)."""
        jw = getattr(self.jog, "joint_window", None)
        if jw is None:
            return None, "joint states are not available (jog module not loaded)", False
        try:
            r = jw([arm], self.standstill_s)
        except Exception as e:  # noqa: BLE001
            return None, f"joint states failed: {e}", False
        if r is None:
            return None, f"joint states of arm '{arm}' are missing, stale or not finite", False
        _, spread, covered = r
        if not covered:
            return float(spread), (f"less than {self.standstill_s:.1f} s of joint history yet: wait until the arm has "
                                   "been still that long"), False
        if not (spread < self.standstill_max_rad):
            return float(spread), (f"the arm is not stationary: a joint moved {spread * 1e3:.1f} mrad in the last "
                                   f"{self.standstill_s:.1f} s (limit {self.standstill_max_rad * 1e3:.0f} mrad); release "
                                   "the teleop keys and wait"), False
        return float(spread), "", True

    # ---- samples file ------------------------------------------------------------------------------
    def _path(self, session_dir: str) -> str:
        return os.path.join(session_dir, SAMPLES_FILE)

    def _file_key(self, path: str) -> Optional[Tuple[str, int, int]]:
        try:
            st = os.stat(path)
        except OSError:
            return None
        return (path, st.st_mtime_ns, st.st_size)

    def _sync(self, session_dir: Optional[str]) -> None:
        """Mirror the samples file into ``_entries`` (reload only if the file or the session directory changed)."""
        with self.lock:
            path = self._path(session_dir) if session_dir else None
            key = self._file_key(path) if path else None
            if (path, key) == (self._loaded_path, self._loaded):
                return
            entries, bad = [], 0
            if key is not None:
                try:
                    with open(path, encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                s = self._tp().PivotSample.from_json_dict(json.loads(line))
                            except Exception:  # noqa: BLE001 - kept in the file, shown as a warning
                                s = None
                                bad += 1
                            entries.append({"raw": line, "sample": s})
                except OSError as e:
                    self.log.error("cannot read %s: %s", path, e)
            self._entries, self._bad_lines = entries, bad
            self._loaded_path, self._loaded = path, key
            self._result_changed_samples()

    def _result_changed_samples(self) -> None:
        self._cov_cache = self._rows_cache = None

    def _arm_entries(self, arm: str) -> List[Dict[str, Any]]:
        return [e for e in self._entries if e["sample"] is not None and getattr(e["sample"], "arm", arm) == arm]

    def _arm_samples(self, arm: str) -> List[Any]:
        return [e["sample"] for e in self._arm_entries(arm)]

    @staticmethod
    def _signature(samples: Sequence[Any]) -> tuple:
        return tuple((round(float(getattr(s, "stamp", 0.0)), 6), str(getattr(s, "role", ""))) for s in samples)

    def _append(self, path: str, line: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())

    def _rewrite(self, path: str, lines: List[str]) -> None:
        """Replace the file by ``lines`` (atomic); the previous content is kept as ``<file>.bak``."""
        if os.path.exists(path):
            shutil.copyfile(path, path + ".bak")
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write("".join(l + "\n" for l in lines))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    # ---- current URDF r --------------------------------------------------------------------------
    def _read_xacro(self) -> Tuple[Optional[str], str]:
        try:
            with open(self.xacro_path, encoding="utf-8") as f:
                return f.read(), ""
        except OSError as e:
            return None, f"cannot read the xacro {self.xacro_path}: {e}"

    def current_r(self, arm: str) -> Tuple[Optional[np.ndarray], str, Optional[float]]:
        """(r in igus_tool0 in m | None, source text, disagreement TF vs source xacro in mm | None).  TF
        igus_tool0 -> xeg32_tool_tip is the live robot_description; the xacro source is the fallback."""
        fr = self._frames(arm)
        T, _age, _why = self._tf(fr["tool0"], fr["tip"])
        text, _err = self._read_xacro()
        rx = parse_xacro_r(text) if text else None
        if T is not None:
            r = T[:3, 3].copy()
            diff = None if rx is None else float(np.linalg.norm(r - rx)) * 1e3
            return r, f"TF {fr['tool0']} -> {fr['tip']} (live robot_description)", diff
        if rx is not None:
            return rx, "xacro source file (TF not available)", None
        return None, "not available (no TF and the xacro could not be read)", None

    # ---- checklist (per page) ----------------------------------------------------------------------
    @staticmethod
    def _need_page(page_id) -> str:
        if not isinstance(page_id, str) or not page_id:
            raise TcpRefused("X-Page-Id header missing or malformed", 400)
        return page_id

    def _page(self, page_id: Optional[str], create: bool = False) -> Optional[Dict[str, Any]]:
        """The page's checklist record {"checked": {key: bool}, "t": last tick / capture time, "expired": bool}; its ticks
        are cleared (and ``expired`` set) once it is older than the time limit."""
        with self.lock:
            now = self.clock()
            rec = self._pages.get(page_id) if page_id else None
            if rec is not None and any(rec["checked"].values()) and now - rec["t"] > self.checklist_ttl_s:
                rec["checked"] = {k: False for k, _ in CHECKLIST}
                rec["expired"] = True
            if rec is None and create and page_id:
                if len(self._pages) >= MAX_PAGES:
                    oldest = min(self._pages, key=lambda k: self._pages[k]["t"])
                    del self._pages[oldest]
                rec = self._pages[page_id] = {"checked": {k: False for k, _ in CHECKLIST}, "t": now, "expired": False}
            return rec

    def set_checklist(self, items: Dict[str, bool], page_id=None) -> Dict[str, Any]:
        self._need_supported()
        self._need_page(page_id)
        keys = {k for k, _ in CHECKLIST}
        bad = [k for k in items if k not in keys]
        if bad:
            raise TcpRefused(f"unknown checklist item(s): {', '.join(sorted(map(str, bad)))}", 400)
        with self.lock:
            rec = self._page(page_id, create=True)
            for k, v in items.items():
                rec["checked"][k] = bool(v)
            rec["t"], rec["expired"] = self.clock(), False
            return self._checklist_view(page_id)

    def _checklist_view(self, page_id=None) -> Dict[str, Any]:
        with self.lock:
            rec = self._page(page_id)
            checked = rec["checked"] if rec else {}
            return {"items": [{"key": k, "text": t, "checked": bool(checked.get(k))} for k, t in CHECKLIST],
                    "all_checked": all(checked.get(k) for k, _ in CHECKLIST),
                    "expired": bool(rec and rec["expired"] and not any(checked.values())),
                    "ttl_min": self.checklist_ttl_s / 60.0}

    # ---- capture / delete ------------------------------------------------------------------------
    def capture(self, role: str, note: str = "", page_id=None) -> Dict[str, Any]:
        arm = self._need_supported()
        self._need_page(page_id)
        if role not in ROLES:
            raise TcpRefused(f"role must be one of {list(ROLES)}", 400)
        if not isinstance(note, str) or len(note) > MAX_NOTE_LEN:
            raise TcpRefused(f"note must be a string of at most {MAX_NOTE_LEN} characters", 400)
        with self.lock:
            rec = self._page(page_id)
            missing = [t for k, t in CHECKLIST if not (rec and rec["checked"].get(k))]
            if missing:
                lead = (f"the safety checklist expired ({self.checklist_ttl_s / 60:.0f} min without a capture): tick it again. "
                        if rec and rec["expired"] else "tick the whole safety checklist first. ")
                raise TcpRefused(lead + f"Not ticked ({len(missing)}): " + " | ".join(missing), 409)
            session_dir = self._need_session_dir()
            why = self._jog_blocked()
            if why:
                raise TcpRefused(why, 409)
            self._sync(session_dir)
            _, problem, ok = self._motion(arm)
            if not ok:
                raise TcpRefused(problem, 409)
            fr = self._frames(arm)
            T_tool0 = self._fresh_tf(fr["base"], fr["tool0"])
            T_link6 = self._fresh_tf(fr["base"], fr["link6"])
            q = self.jog.joint_window([arm], self.standstill_s)
            if q is None:                           # (cannot happen right after _motion, but never store without joints)
                raise TcpRefused(f"joint states of arm '{arm}' are missing, stale or not finite", 409)
            names = list(self.cfgs[arm].joint_names)
            joints = {n: float(q[0][n]) for n in names if n in q[0]}
            existing = self._arm_entries(arm)
            tp = self._tp()
            dup = tp.duplicate_of(T_tool0, [e["sample"].T_base_tool0 for e in existing], self.min_rot_deg)
            if dup is not None:
                other = existing[int(dup)]["sample"]
                raise TcpRefused(
                    f"too similar to pose #{int(dup) + 1} ({getattr(other, 'role', '?')}): less than "
                    f"{self.min_rot_deg:.0f} deg of tool rotation apart. Lift the tip, rotate the tool to a clearly "
                    "different direction (see the pose plan) and re-seat it in the countersink", 409)
            sample = tp.PivotSample(arm=arm, T_base_tool0=T_tool0, T_base_link6=T_link6, joints=joints,
                                    stamp=float(self.wall_clock()), role=role, note=note)
            line = json.dumps(clean(sample.to_json_dict()), allow_nan=False, sort_keys=True)
            path = self._path(session_dir)
            self._append(path, line)
            rec["t"] = self.clock()                 # a capture keeps this page's ticks alive for another time limit
            self._entries.append({"raw": line, "sample": sample})
            self._loaded = self._file_key(path)
            self._result_changed_samples()
            n_role = sum(1 for e in self._arm_entries(arm) if getattr(e["sample"], "role", "") == role)
            n = len(self._arm_entries(arm))
            return {"index": n - 1,
                    "message": f"captured {role} pose #{n} ({n_role} {role}); stored in {path}"}

    def delete(self, index) -> Dict[str, Any]:
        arm = self._need_supported()
        if isinstance(index, bool) or not isinstance(index, int):
            raise TcpRefused("index must be an integer", 400)
        with self.lock:
            session_dir = self._need_session_dir()
            self._sync(session_dir)
            arm_entries = self._arm_entries(arm)
            if not 0 <= index < len(arm_entries):
                raise TcpRefused(f"no pose #{index + 1} (there are {len(arm_entries)})", 400)
            gone = arm_entries[index]
            path = self._path(session_dir)
            self._rewrite(path, [e["raw"] for e in self._entries if e is not gone])
            self._entries = [e for e in self._entries if e is not gone]
            self._loaded = self._file_key(path)
            self._result_changed_samples()
            role = getattr(gone["sample"], "role", "?")
            return {"message": f"deleted pose #{index + 1} ({role}); the old file is kept as {SAMPLES_FILE}.bak. "
                               "Re-capture a pose in that direction, then solve again"}

    # ---- solve ---------------------------------------------------------------------------------------
    def solve(self) -> Dict[str, Any]:
        arm = self._need_supported()
        with self.lock:
            session_dir = self._session_dir()
            self._sync(session_dir)
            samples = list(self._arm_samples(arm))
        n_fit = sum(1 for s in samples if getattr(s, "role", "") == "fit")
        if n_fit == 0:
            raise TcpRefused("no fit poses captured yet", 409)
        cur, cur_src, _ = self.current_r(arm)
        try:
            res = self._tp().solve_pivot(samples, current_r=cur)   # (outside the lock: the polls stay responsive)
        except ValueError as e:                                     # malformed transform / mixed arms: not retryable as is
            raise TcpRefused(f"the solver rejected the poses: {e}", 409) from e
        view = self._result_view(res, samples, cur)
        diff, diff_err = self._make_diff(res)
        with self.lock:
            self._result = {"res": res, "arm": arm, "sig": self._signature(samples), "samples": samples,
                            "current_r": cur, "current_src": cur_src, "view": view, "diff": diff,
                            "diff_error": diff_err, "time": self.wall_clock()}
            self._mech_compute()
        verdict = view.get("verdict")
        r_txt = ("r = [" + ", ".join(f"{x:.2f}" for x in view["r_mm"]) + "] mm") if view.get("r_mm") else "no r"
        return {"message": f"solved: {verdict}, {r_txt}", "verdict": verdict}

    def _result_view(self, res, samples: Sequence[Any], cur) -> Dict[str, Any]:
        g = lambda name, default=None: getattr(res, name, default)   # noqa: E731
        fit_idx = [i for i, s in enumerate(samples) if getattr(s, "role", "") == "fit"]
        val_idx = [i for i, s in enumerate(samples) if getattr(s, "role", "") == "validate"]
        resid, loo, held = g("residuals_mm"), g("loo_mm"), g("heldout_mm")
        out_set = set(int(i) for i in (g("outliers") or []))
        rows = []

        def seq(v):
            try:
                return [None if not math.isfinite(float(x)) else float(x) for x in np.asarray(v, dtype=float).reshape(-1)]
            except (TypeError, ValueError):
                return []
        r_list, l_list, h_list = seq(resid), seq(loo), seq(held)
        for k, i in enumerate(fit_idx):
            rows.append({"index": i, "role": "fit", "residual_mm": r_list[k] if k < len(r_list) else None,
                         "loo_mm": l_list[k] if k < len(l_list) else None, "heldout_mm": None,
                         "outlier": k in out_set})
        for k, i in enumerate(val_idx):
            rows.append({"index": i, "role": "validate", "residual_mm": None, "loo_mm": None,
                         "heldout_mm": h_list[k] if k < len(h_list) else None, "outlier": False})
        r = g("r_tool0")
        view = {
            "ok": bool(g("ok", False)), "verdict": str(g("verdict", "")),
            "reasons": [str(x) for x in (g("reasons") or [])],
            "r_mm": _mm(r), "c_base_mm": _mm(g("c_base")),
            "current_r_mm": _mm(cur) if cur is not None else None,
            "delta_to_current_mm": clean(g("delta_to_current_mm")),
            "n_fit": g("n_fit"), "n_validate": g("n_validate"),
            "rms_mm": g("rms_mm"), "max_mm": g("max_mm"), "loo_rms_mm": g("loo_rms_mm"),
            "heldout_rms_mm": g("heldout_rms_mm"), "sigma_axes_mm": _vec_list(g("sigma_axes_mm")),
            "sigma_xyz_mm": self._sigma_xyz(g("cov_r")), "weak_axis_tool0": clean(g("weak_axis_tool0")),
            "max_tilt_deg": self._nanmax(g("tilt_deg")), "max_abs_roll_deg": self._nanmax(g("roll_deg"), absolute=True),
            "obs_min": g("obs_min"), "cond_B": g("cond_B"), "rank": g("rank"),
            "rows": rows, "outliers": sorted(fit_idx[k] for k in out_set if 0 <= k < len(fit_idx)),
        }
        return clean(view)

    @staticmethod
    def _sigma_xyz(cov) -> Optional[List[float]]:
        """1-sigma of r along tool0 x, y, z in mm from the covariance (m^2)."""
        try:
            c = np.asarray(cov, dtype=float).reshape(3, 3)
            d = np.sqrt(np.diag(c)) * 1e3
        except (TypeError, ValueError):
            return None
        return d.tolist() if np.all(np.isfinite(d)) else None

    @staticmethod
    def _nanmax(v, absolute: bool = False) -> Optional[float]:
        try:
            a = np.asarray(v, dtype=float).reshape(-1)
            a = np.abs(a) if absolute else a
            a = a[np.isfinite(a)]
        except (TypeError, ValueError):
            return None
        return float(a.max()) if a.size else None

    def _patch_path(self) -> str:
        """The path shown in the diff headers: relative to the workspace root (``patch -p1`` from there)."""
        p = self.xacro_path
        i = p.rfind("/src/")
        return p[i + 1:] if i >= 0 else os.path.basename(p)

    def _make_diff(self, res) -> Tuple[Optional[str], str]:
        r = getattr(res, "r_tool0", None)
        if r is None or _vec_list(r) is None:
            return None, "no r to patch"
        if str(getattr(res, "verdict", "")) not in GOOD_VERDICTS or not getattr(res, "ok", False):
            return None, (f"no patch: the verdict is {getattr(res, 'verdict', '?')} (only GOOD / PASS results get a "
                          "patch; fix the reasons above and solve again)")
        text, err = self._read_xacro()
        if text is None:
            return None, err
        try:
            try:
                diff = self._tp().xacro_patch(text, np.asarray(r, dtype=float), path=self._patch_path())
            except TypeError:                       # (a tcp_pivot without the path keyword)
                diff = self._tp().xacro_patch(text, np.asarray(r, dtype=float))
            return str(diff), ""
        except Exception as e:  # noqa: BLE001
            return None, f"the xacro patch could not be made: {e}"

    # ---- mechanical sanity check ----------------------------------------------------------------
    def mechanical(self, length_mm) -> Dict[str, Any]:
        self._need_supported()
        if isinstance(length_mm, bool) or not isinstance(length_mm, (int, float)) or not math.isfinite(float(length_mm)):
            raise TcpRefused("length_mm must be a finite number", 400)
        lo, hi = MECH_LENGTH_RANGE_MM
        if not lo <= float(length_mm) <= hi:
            raise TcpRefused(f"length_mm must be between {lo:.0f} and {hi:.0f} mm (flange face to blade tip, calipers)", 400)
        arm = self.arm
        cur, _, _ = self.current_r(arm)
        with self.lock:
            have_solved = self._result is not None and _vec_list(getattr(self._result["res"], "r_tool0", None)) is not None
            if cur is None and not have_solved:
                raise TcpRefused("the current URDF r is not available (no TF, xacro unreadable) and there is no solve", 503)
            self._mech_len = float(length_mm)
            self._mech_compute(cur=cur)
            return {"message": "mechanical check done", "mechanical": self._mech}

    def _mech_compute(self, cur=None) -> None:
        """Compare the caliper length with the SOLVED r when there is one, else with the current URDF r."""
        if self._mech_len is None:
            self._mech = None
            return
        if cur is None:
            cur, _, _ = self.current_r(self.arm)
        out: Dict[str, Any] = {"length_mm": self._mech_len, "r_source": "", "check": None, "error": ""}
        try:
            r_new = getattr(self._result["res"], "r_tool0", None) if self._result else None
            if r_new is not None and _vec_list(r_new) is not None:
                out["r_source"], r = "the solved r", r_new
            else:
                out["r_source"], r = "the current URDF r (no valid solve yet)", cur
            if r is not None:
                out["check"] = clean(self._tp().mechanical_check(np.asarray(r, dtype=float), self._mech_len))
        except Exception as e:  # noqa: BLE001
            out["error"] = f"mechanical check failed: {e}"
        self._mech = out

    # ---- write ------------------------------------------------------------------------------------
    def write(self) -> Dict[str, Any]:
        arm = self._need_supported()
        with self.lock:
            session_dir = self._need_session_dir()
            self._sync(session_dir)
            cur_res = self._result
            if cur_res is None:
                raise TcpRefused("solve first", 409)
            if cur_res["sig"] != self._signature(self._arm_samples(arm)):
                raise TcpRefused("the poses changed since the last solve: solve again before writing", 409)
            res = cur_res["res"]
            verdict = str(getattr(res, "verdict", ""))
            if verdict not in GOOD_VERDICTS or not getattr(res, "ok", False):
                raise TcpRefused(f"the verdict is {verdict or '?'}: only GOOD / PASS results are written. Fix the "
                                 "reasons shown (delete and re-capture outliers, add poses) and solve again", 409)
            text, err = self._read_xacro()
            if text is None:
                raise TcpRefused(err, 503)
            path = os.path.join(session_dir, RESULT_FILE.format(arm=arm))
            if os.path.exists(path):
                shutil.copyfile(path, path + ".bak")
            self._tp().write_result_yaml(res, path, arm, self.urdf_hash_fn(text), self.xacro_path, cur_res["current_r"])
            diff, diff_err = self._make_diff(res)
            cur_res["diff"], cur_res["diff_error"] = diff, diff_err
            self._written = {"path": path, "time": self.wall_clock(), "verdict": verdict}
            return {"message": f"result written to {path}. The xacro was NOT changed: apply the diff by hand "
                               "(steps below)", "path": path}

    # ---- state --------------------------------------------------------------------------------------
    def _knife_axis(self, cur) -> np.ndarray:
        v = np.asarray(cur if cur is not None else DEFAULT_KNIFE_AXIS, dtype=float).reshape(3)
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-9 else DEFAULT_KNIFE_AXIS / np.linalg.norm(DEFAULT_KNIFE_AXIS)

    def _coverage(self, samples: Sequence[Any], cur) -> Dict[str, Any]:
        key = (self._signature(samples), None if cur is None else tuple(round(float(x), 6) for x in cur))
        if self._cov_cache is not None and self._cov_cache[0] == key:
            return self._cov_cache[1]
        tp = self._tp()
        try:
            cov = tp.coverage(list(samples), r_tool0=cur) if cur is not None else tp.coverage(list(samples))
        except Exception as e:  # noqa: BLE001 - guidance only
            cov = {"suggestion": f"coverage failed: {e}"}
        covered = cov.get("covered") if isinstance(cov, dict) else None
        plan = []
        for item in list(getattr(tp, "POSE_PLAN", [])):
            if not isinstance(item, dict):
                item = {"id": str(item), "text": ""}
            iid = item.get("id", item.get("name", ""))
            plan.append({"id": iid, "role": item.get("role", ""), "label": item.get("label", item.get("name", iid)),
                         "text": item.get("text", ""),
                         "covered": None if covered is None else (iid in covered)})
        out = {"suggestion": str(cov.get("suggestion", "")), "next": cov.get("next"), "complete": cov.get("complete"),
               "reference_index": cov.get("reference_index"), "max_tilt_deg": cov.get("max_tilt_deg"),
               "tilt_from_vertical_deg": clean(cov.get("tilt_from_vertical_deg") or []), "plan": plan}
        out = clean(out)
        self._cov_cache = (key, out)
        return out

    def _live(self, arm: str, samples: Sequence[Any], cov: Dict[str, Any], axis: np.ndarray) -> Dict[str, Any]:
        fr = self._frames(arm)
        spread, problem, stationary = self._motion(arm)
        T, age, why = self._tf(fr["base"], fr["tool0"])
        live: Dict[str, Any] = {"tf_ok": T is not None and age is not None and age <= self.tf_max_age_s,
                                "tf_age_s": age, "tf_problem": why, "stationary": stationary,
                                "motion_problem": problem, "spread_mrad": None if spread is None else spread * 1e3,
                                "tilt_vertical_deg": None, "tilt_deg": None, "roll_deg": None, "nearest_deg": None,
                                "nearest_index": None, "duplicate_of": None}
        if T is not None and age is not None and age > self.tf_max_age_s:
            live["tf_problem"] = f"TF {fr['base']} -> {fr['tool0']} is stale ({age:.2f} s old)"
        blockers: List[str] = []                    # why a capture would be refused right now (the checklist is per page)
        if not self._session_dir():
            blockers.append("session directory unknown: poses cannot be stored")
        busy = self._jog_blocked()
        if busy:
            blockers.append(busy)
        if not stationary and problem:
            blockers.append(problem)
        if not live["tf_ok"]:
            blockers.append(live["tf_problem"] or "TF is not fresh")
        live["blockers"] = blockers
        if T is None:
            return clean(live)
        try:
            live["tilt_vertical_deg"] = tilt_from_vertical_deg(T[:3, :3], axis)
            ref = cov.get("reference_index")
            if ref is not None and 0 <= ref < len(samples):
                live["tilt_deg"], live["roll_deg"] = swing_twist_deg(
                    np.asarray(samples[ref].T_base_tool0)[:3, :3].T @ T[:3, :3], axis)
            if samples:
                tp = self._tp()
                dists = [float(tp.rotation_distance_deg(T, s.T_base_tool0)) for s in samples]
                k = int(np.argmin(dists))
                live["nearest_deg"], live["nearest_index"] = dists[k], k
                d = tp.duplicate_of(T, [s.T_base_tool0 for s in samples], self.min_rot_deg)
                live["duplicate_of"] = None if d is None else int(d)
                if d is not None:
                    blockers.append(f"too similar to pose #{int(d) + 1}: rotate the tool at least {self.min_rot_deg:.0f} deg "
                                    "away from it, then re-seat the tip")
        except Exception as e:  # noqa: BLE001 - guidance only
            live["guidance_error"] = str(e)
        return clean(live)

    def state(self, page_id=None) -> Dict[str, Any]:
        base = {"supported": self.supported, "mode": self.mode, "arm": self.arm,
                "checklist": self._checklist_view(page_id),
                "thresholds": {"standstill_s": self.standstill_s, "standstill_mrad": self.standstill_max_rad * 1e3,
                               "tf_max_age_s": self.tf_max_age_s, "min_rotation_deg": self.min_rot_deg},
                "rebase_warning": REBASE_WARNING, "apply_steps": list(APPLY_STEPS),
                "mechanical_note": {"current_urdf_mm": CURRENT_URDF_FLANGE_TO_TIP_MM,
                                    "old_pivot_mm": OLD_PIVOT_FLANGE_TO_TIP_MM},
                "xacro_path": self.xacro_path}
        if not self.supported:
            base.update(message="TCP calibration: single-arm modes only", samples=[], n_fit=0, n_validate=0,
                        live=None, coverage=None, result=None, mechanical=None, written=None, diff=None,
                        session_dir=None, file=None, warnings=[])
            return base
        arm = self.arm
        session_dir = self._session_dir()
        warnings: List[str] = []
        with self.lock:
            self._sync(session_dir)
            samples = list(self._arm_samples(arm))
            bad = self._bad_lines
            result = self._result
            mech = self._mech
            written = self._written
        if bad:
            warnings.append(f"{bad} line(s) of {SAMPLES_FILE} could not be read and are ignored (kept in the file)")
        if not session_dir:
            warnings.append("session directory unknown: poses cannot be stored yet")
        cur, cur_src, mism = self.current_r(arm)
        if mism is not None and mism > 0.1:
            warnings.append(f"the live robot_description r differs from the source xacro by {mism:.2f} mm: the stack "
                            "was not rebuilt/restarted after the last xacro change (or the xacro is not the one in use)")
        axis = self._knife_axis(cur)
        cov = self._coverage(samples, cur)
        res_view = None
        stale = False
        if result is not None:
            stale = result["sig"] != self._signature(samples)
            if stale:       # per-pose numbers, outliers and reasons refer to the OLD poses: show the stale line only
                res_view = {"stale": True, "verdict": result["view"].get("verdict"), "rows": [], "outliers": [],
                            "reasons": []}
            else:
                res_view = dict(result["view"])
                res_view["stale"] = False
            res_view["time"] = _iso(result["time"])
        if stale and mech:
            mech = dict(mech, r_source=str(mech.get("r_source", "")) + " (STALE: solve again)")
        base.update(
            message="", session_dir=session_dir, file=self._path(session_dir) if session_dir else None,
            samples=self._sample_rows(samples, cov, axis), n_fit=sum(1 for s in samples if getattr(s, "role", "") == "fit"),
            n_validate=sum(1 for s in samples if getattr(s, "role", "") == "validate"),
            live=self._live(arm, samples, cov, axis), coverage=cov,
            current_r_mm=_mm(cur), current_r_source=cur_src, result=res_view, mechanical=clean(mech),
            written=clean(dict(written, time=_iso(written["time"]))) if written else None,
            diff=None if (stale or not result) else result["diff"],
            diff_error=("the poses changed since the last solve: solve again to get a patch" if stale
                        else (result["diff_error"] if result else "")),
            warnings=warnings)
        return base

    def _sample_rows(self, samples: Sequence[Any], cov: Dict[str, Any], axis: np.ndarray) -> List[Dict[str, Any]]:
        key = (self._signature(samples), tuple(round(float(x), 6) for x in axis))
        if self._rows_cache is not None and self._rows_cache[0] == key:
            return self._rows_cache[1]
        tp = self._tp()
        ref = cov.get("reference_index")
        tv = cov.get("tilt_from_vertical_deg") or []
        rows = []
        for i, s in enumerate(samples):
            row = {"index": i, "role": getattr(s, "role", ""), "stamp": getattr(s, "stamp", None),
                   "time": _iso(getattr(s, "stamp", None)), "note": getattr(s, "note", ""),
                   "tilt_vertical_deg": tv[i] if i < len(tv) else None, "tilt_deg": None, "roll_deg": None,
                   "nearest_deg": None}
            try:
                if ref is not None and 0 <= ref < len(samples):
                    row["tilt_deg"], row["roll_deg"] = swing_twist_deg(
                        np.asarray(samples[ref].T_base_tool0)[:3, :3].T @ np.asarray(s.T_base_tool0)[:3, :3], axis)
                others = [float(tp.rotation_distance_deg(s.T_base_tool0, o.T_base_tool0))
                          for j, o in enumerate(samples) if j != i]
                row["nearest_deg"] = min(others) if others else None
            except Exception:  # noqa: BLE001 - guidance only
                pass
            rows.append(clean(row))
        self._rows_cache = (key, rows)
        return rows
