"""Carton placement checks.  INFORMATIONAL ONLY: nothing here may ever block detection, planning or execution (operator
decision); the page and ``tools/dual_run/check_placement.py`` show the result and the operator decides.  No ROS.

Four independent checks on a detected snapshot (box_h_cut_snapshot/v1, base_link = left base frame, +x away from the
arms, +y towards the left arm side), all calibrated on the 42 detections of 2026-10-22 planned by the dual stack in mock
(``scan_results/today_plans_20261022/``, ``scan_results/left_face_fail_20261022/``):

* ``region``      the centre / seam-yaw box in which today's detections planned (soft note when outside).
* ``two_arm_reach`` (new 2026-11-07, ``placement_advisor.py``, listed first) wrist-model check that BOTH arms can start their
                  cuts (left wrist <= 0.52 m from its base at back_left, right wrist >= 0.19 m from its base at front_right, tip
                  guards), with advice: the cheapest move / turn / both that satisfies it with a margin.  Uses only the
                  snapshot's corners and normal, so it works for any box size.  Warn when failing, note when marginal.
* ``left_reach``  OFF BY DEFAULT (``left_reach_enabled``), superseded by ``two_arm_reach``: it places the wrist at a fixed
                  world offset from the tape band, ignoring the along-travel wrist offset (the wrist trails ~90 mm behind the
                  tip) and the orientation search, and says nothing about the right arm.  Kept for reference; geometric estimate of "left face cut fails closed (left varying orientation search)": the tape band
                  (+-25 mm around the seam crossing on the left face cut line) must be reachable by the left arm's wrist
                  centre (<= 541 mm from the shoulder, plus what tilting the blade within the 25 deg cap buys).  Flags
                  t090925 and t093229 and none of the other 40 (see the README of ``placement_precheck_20261022``).
* ``right_table`` the right holder touched the table edge for close cartons (centre x <= 0.4217 and yaw <= -89.5).
                  SOFT note ("may be refused"): another change adds the table to each arm's planning scene.

Every threshold lives in ``PlacementLimits`` (override with ``PlacementLimits.from_dict`` / the ``placement_limits``
node parameter, a JSON object).  The old 2026-10-18 window is kept as ``LEGACY_WINDOW_20261018`` for reference and for
``check_placement.py --legacy``.
"""
import dataclasses
import json
import math
from typing import Dict, List, Optional, Sequence, Tuple

from . import placement_advisor

CORNER_KEYS = ("front_left", "front_right", "back_right", "back_left")

# The 2026-10-18 window (mock study dual_window_15mm_20261018 / standoff_sweep_20261018, right holder of that day): the
# two-arm cut planned only for seam centres near (0.430, -0.136) m at seam yaw -94.5 .. -89.5 deg.  Outdated since the
# 10-22 detections (centre y -0.17 .. -0.245) planned outside it; kept for reference only.
LEGACY_WINDOW_20261018 = {"yaw_deg": (-94.5, -89.5), "yaw_aim_deg": -92.0, "centre_xy_m": (0.430, -0.136), "offset_max_mm": 20.0}


def _num(v) -> Optional[float]:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None


def xyz(v) -> Optional[Tuple[float, float, float]]:
    """{"x":..,"y":..,"z":..} or [x, y, z] -> (x, y, z), else None."""
    if isinstance(v, dict):
        vals = [_num(v.get(k)) for k in ("x", "y", "z")]
    elif isinstance(v, (list, tuple)) and len(v) >= 3:
        vals = [_num(c) for c in v[:3]]
    else:
        return None
    return None if any(c is None for c in vals) else tuple(vals)


@dataclasses.dataclass(frozen=True)
class PlacementLimits:
    # ---- region where today's detections planned (cell "a": sequence, the live default): 25 of 42 plans, bounding box of
    # their seam centres / seam yaws (x 0.3800..0.4346, y -0.2449..-0.1706, yaw -92.46..-77.74), 0.5 mm outward rounding.
    region_x_m: Tuple[float, float] = (0.3795, 0.4350)
    region_y_m: Tuple[float, float] = (-0.2455, -0.1705)
    region_yaw_deg: Tuple[float, float] = (-92.5, -77.7)
    # ---- left face reach estimate
    shoulder_m: Tuple[float, float, float] = (0.0, 0.0, 0.252)    # left base + (kD1Z + kO2Z) along z (rebel_analytic_ik.cpp)
    wrist_envelope_m: float = 0.541                               # wrist centre <= this from the shoulder (L1 + L2)
    wrist_tilt_allowance_m: float = 0.011                         # what tilting the blade (25 deg cap) buys; calibrated, see below
    wrist_lateral_m: float = 0.24                                 # wrist centre = tip + 0.24 m outward (+y) + 0.09 m up, measured
    wrist_up_m: float = 0.09                                      #   on the planned left face strokes (nominal knife pose)
    band_half_m: float = 0.025                                    # tape band = seam crossing +- 25 mm along the left face cut line
    hint_margin_m: float = 0.008                                  # the move / turn hint aims this far below the flag radius
    # Calibration (nominal wrist-centre radius at the back end of the band, mm): failures t090925 560.4, t093229 561.9;
    # the nearest plans t074923 543.5, t083008 539.7, then <= 531.2.  The flag radius 541 + 11 = 552 mm sits in the gap.
    # Superseded by two_arm_reach (it ignores the along-travel wrist offset and the right arm): off by default, code kept.
    left_reach_enabled: bool = False
    # ---- right holder vs table (soft)
    right_table_enabled: bool = True
    right_table_x_max_m: float = 0.4217       # refused: centre x 0.3728 .. 0.4215 (planned from x 0.4221 on)
    right_table_yaw_max_deg: float = -89.5    # refused: yaw -90.0 .. -96.1 (planned cartons at yaw >= -88.9 for x < 0.422)

    # ---- two-arm reach model + placement advisor (placement_advisor.py; thresholds refitted 2026-11-08 on the template sweeps + 3 carton sizes, 1804 cells)
    two_arm_reach_enabled: bool = True
    left_base_m: Tuple[float, float, float] = (0.0, 0.0, 0.0)                       # world frame = left arm base_link
    right_base_m: Tuple[float, float, float] = (-8.357992070818208e-05, -0.4397165297206802, 0.0046762267300018028)
    tool_offset_mm: Tuple[float, float, float] = (-42.2, 89.6, -157.2)             # wrist minus tip in the TOOL frame
    left_wrist_max_m: float = 0.52          # left: min over orientations of the horizontal wrist distance at back_left <= this
    right_wrist_min_m: float = 0.185        # right: max over orientations of that distance at front_right >= this
    left_tip_min_m: float = 0.09            # front_left tip from the left base
    right_tip_min_m: float = 0.105          # front_right tip from the right base
    right_far_max_m: float = 0.515          # right: min over orientations of the wrist distance at back_right (far end of its travel) <= this
    right_start_max_m: float = 0.35         # right: the same at front_right (start corner) <= this (empirical; the planner's right arm
                                            #   runs out of joint margin when the start corner is far out)
    carton_min_x_m: float = 0.115           # smallest x of the four top corners (world = left base frame; arms at x ~ 0)
    orient_tilt_max_deg: float = 25.0       # orientation search of the executor: tilt about 'along'
    orient_pitch_max_deg: float = 25.0      #   pitch about 'across'
    orient_yaw_max_deg: float = 30.0        #   yaw about the face normal
    orient_step_deg: float = 5.0
    blade_cap_deg: float = 25.0             # blade angle theta_p cap
    penetration_min_dot: float = 0.7        # penetration gate
    reach_marginal_mm: float = 15.0         # passes with less than this to spare -> soft note
    advice_enabled: bool = True
    advice_margin_mm: float = 20.0          # the advised placement satisfies the model with this much to spare
    advice_max_shift_mm: float = 250.0
    advice_shift_step_mm: float = 10.0
    advice_max_turn_deg: float = 30.0
    advice_turn_step_deg: float = 1.0
    advice_cw_cost_factor: float = 1.1      # clockwise turns count this much more (anticlockwise is the good direction)
    # Where the model was validated against the planner: carton turns relative to the 2026-10-22 template (left face travel
    # back_left -> front_left at -178.4 deg).  With the 2026-11-08 constraints the false-OK rate of the cells the model passes is
    # <= 12 % per box from -30 to +30 deg (it was 39 % agreement beyond +10 before); beyond +30 only the template box was swept (+35..+50:
    # every passed cell failed), so there a pass is reported as unvalidated and advice never turns past the upper bound.
    model_face_yaw_ref_deg: float = -178.4
    model_valid_turn_deg: Tuple[float, float] = (-30.0, 30.5)   # +30 grid yaw = +30.02 measured

    @property
    def wrist_flag_m(self) -> float:
        return self.wrist_envelope_m + self.wrist_tilt_allowance_m

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "PlacementLimits":
        """Defaults overridden by the keys of ``d``; unknown keys are an error (a typo must not pass silently)."""
        d = dict(d or {})
        fields = {f.name: f for f in dataclasses.fields(cls)}
        bad = sorted(set(d) - set(fields))
        if bad:
            raise ValueError("unknown placement limit(s): " + ", ".join(bad))
        out = {}
        for k, v in d.items():
            if isinstance(fields[k].default, tuple):
                if not isinstance(v, (list, tuple)) or len(v) != len(fields[k].default):
                    raise ValueError(f"placement limit {k} must be a list of {len(fields[k].default)} numbers")
                v = tuple(float(c) for c in v)
            elif isinstance(fields[k].default, bool):
                if not isinstance(v, bool):
                    raise ValueError(f"placement limit {k} must be true or false")
            else:
                if _num(v) is None:
                    raise ValueError(f"placement limit {k} must be a finite number")
                v = float(v)
            out[k] = v
        return cls(**out)

    @classmethod
    def from_json(cls, text: str) -> "PlacementLimits":
        text = (text or "").strip()
        if not text:
            return cls()
        try:
            d = json.loads(text)
        except ValueError as e:
            raise ValueError(f"placement_limits is not valid JSON: {e}") from e
        if not isinstance(d, dict):
            raise ValueError("placement_limits must be a JSON object")
        return cls.from_dict(d)

    def to_dict(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in dataclasses.asdict(self).items()}


DEFAULT_LIMITS = PlacementLimits()


def _yaw_deg(s, e) -> float:
    return math.degrees(math.atan2(e[1] - s[1], e[0] - s[0]))


def _round_up(mm: float, step: float = 5.0) -> int:
    return int(math.ceil(mm / step - 1e-9) * step)


# ------------------------------------------------------------------------------------------------------------ region
def compute_placement(seam_start, seam_end, seam_center, limits: Optional[PlacementLimits] = None) -> Optional[dict]:
    """Seam yaw, centre and the check against the region where today's detections planned; None if a point is missing."""
    lim = limits or DEFAULT_LIMITS
    s, e, c = xyz(seam_start), xyz(seam_end), xyz(seam_center)
    if s is None or e is None or c is None:
        return None
    yaw = _yaw_deg(s, e)
    devs = []
    (x0, x1), (y0, y1), (w0, w1) = lim.region_x_m, lim.region_y_m, lim.region_yaw_deg
    if c[0] > x1:
        devs.append({"axis": "x", "value": c[0], "limit": x1, "excess_mm": 1000 * (c[0] - x1),
                     "hint": f"move the carton about {_round_up(1000 * (c[0] - x1))} mm toward -x (toward the arms)"})
    elif c[0] < x0:
        devs.append({"axis": "x", "value": c[0], "limit": x0, "excess_mm": 1000 * (x0 - c[0]),
                     "hint": f"move the carton about {_round_up(1000 * (x0 - c[0]))} mm toward +x (away from the arms)"})
    if c[1] > y1:
        devs.append({"axis": "y", "value": c[1], "limit": y1, "excess_mm": 1000 * (c[1] - y1),
                     "hint": f"move the carton about {_round_up(1000 * (c[1] - y1))} mm toward -y (to the right)"})
    elif c[1] < y0:
        devs.append({"axis": "y", "value": c[1], "limit": y0, "excess_mm": 1000 * (y0 - c[1]),
                     "hint": f"move the carton about {_round_up(1000 * (y0 - c[1]))} mm toward +y (to the left)"})
    if yaw > w1:
        devs.append({"axis": "yaw", "value": yaw, "limit": w1, "excess_deg": yaw - w1,
                     "hint": f"turn the carton clockwise (seen from above) by about {math.ceil(yaw - w1)} deg"})
    elif yaw < w0:
        devs.append({"axis": "yaw", "value": yaw, "limit": w0, "excess_deg": w0 - yaw,
                     "hint": f"turn the carton anticlockwise (seen from above) by about {math.ceil(w0 - yaw)} deg"})
    return {
        "yaw_deg": yaw, "centre_x_m": c[0], "centre_y_m": c[1],
        "region": {"x_m": list(lim.region_x_m), "y_m": list(lim.region_y_m), "yaw_deg": list(lim.region_yaw_deg),
                   "inside": not devs, "deviations": devs,
                   "basis": "where the 25 plannable of today's 42 detections (2026-10-22) lie; refusals can still happen inside"},
    }


def legacy_check(seam_start, seam_end, seam_center) -> Optional[dict]:
    """The outdated 2026-10-18 window (yaw -94.5..-89.5, centre within 20 mm of (0.430, -0.136)); reference only."""
    s, e, c = xyz(seam_start), xyz(seam_end), xyz(seam_center)
    if s is None or e is None or c is None:
        return None
    w = LEGACY_WINDOW_20261018
    yaw = _yaw_deg(s, e)
    off_mm = 1000.0 * math.hypot(c[0] - w["centre_xy_m"][0], c[1] - w["centre_xy_m"][1])
    hint = None
    if yaw > w["yaw_deg"][1]:
        hint = {"direction": "CLOCKWISE", "deg": yaw - w["yaw_aim_deg"]}
    elif yaw < w["yaw_deg"][0]:
        hint = {"direction": "ANTI-CLOCKWISE", "deg": w["yaw_aim_deg"] - yaw}
    return {"yaw_deg": yaw, "yaw_window_deg": list(w["yaw_deg"]), "yaw_aim_deg": w["yaw_aim_deg"],
            "yaw_ok": w["yaw_deg"][0] <= yaw <= w["yaw_deg"][1], "centre_x_m": c[0], "centre_y_m": c[1],
            "planned_centre_m": list(w["centre_xy_m"]), "offset_mm": off_mm, "offset_max_mm": w["offset_max_mm"],
            "offset_ok": off_mm <= w["offset_max_mm"], "rotation_hint": hint}


# ------------------------------------------------------------------------------------------------------ left face reach
def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _norm(a) -> float:
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def _band_geometry(doc: dict) -> Optional[dict]:
    """Back_middle_left, front_middle_left (the left face cut line) and the seam crossing on it; None if not derivable."""
    pt = lambda k: xyz(doc.get(k))   # noqa: E731
    ss = pt("seam_start")
    bml, fml = pt("back_middle_left"), pt("front_middle_left")
    mid = lambda a, b: tuple(0.5 * (p + q) for p, q in zip(a, b))   # noqa: E731
    if bml is None and pt("back_left") and ss:
        bml = mid(pt("back_left"), ss)            # side_cutting_midpoints/v1: back_middle_left = mean(back_left, seam_start)
    if fml is None and pt("front_left") and ss:
        fml = mid(pt("front_left"), ss)
    if ss is None or bml is None or fml is None:
        return None
    d = _sub(fml, bml)
    n = _norm(d)
    if n < 0.05:
        return None
    u = tuple(c / n for c in d)                   # along the left face cut line, back -> front
    s_cross = sum((ss[i] - bml[i]) * u[i] for i in range(3))
    return {"bml": bml, "u": u, "s_cross": s_cross}


def _wrist_radius(geom: dict, lim: PlacementLimits, dxyz=(0.0, 0.0, 0.0), rot_deg: float = 0.0,
                  pivot=(0.0, 0.0, 0.0)) -> Tuple[float, tuple]:
    """(largest wrist-centre distance from the shoulder over the two ends of the tape band, tip point of that end).  The
    carton may be moved by ``dxyz`` and turned anticlockwise by ``rot_deg`` about the vertical axis through ``pivot``."""
    a = math.radians(rot_deg)
    ca, sa = math.cos(a), math.sin(a)
    wrist_off = (0.0, lim.wrist_lateral_m, lim.wrist_up_m)       # world frame: the planned strokes keep it ~constant
    best, best_pt = -1.0, None
    for sign in (-1.0, 1.0):
        s = geom["s_cross"] + sign * lim.band_half_m
        p = tuple(geom["bml"][i] + geom["u"][i] * s for i in range(3))
        rx, ry = p[0] - pivot[0], p[1] - pivot[1]
        p = (pivot[0] + ca * rx - sa * ry + dxyz[0], pivot[1] + sa * rx + ca * ry + dxyz[1], p[2] + dxyz[2])
        w = tuple(p[i] + wrist_off[i] for i in range(3))
        r = _norm(_sub(w, lim.shoulder_m))
        if r > best:
            best, best_pt = r, p
    return best, best_pt


def left_face_reach(doc: dict, limits: Optional[PlacementLimits] = None) -> Optional[dict]:
    """Estimate whether the left arm can reach the tape band on the left face.  None if the snapshot lacks the points."""
    lim = limits or DEFAULT_LIMITS
    geom = _band_geometry(doc)
    if geom is None:
        return None
    r, tip = _wrist_radius(geom, lim)
    flag = r > lim.wrist_flag_m
    out = {"wrist_radius_mm": 1000 * r, "flag_radius_mm": 1000 * lim.wrist_flag_m, "envelope_mm": 1000 * lim.wrist_envelope_m,
           "margin_mm": 1000 * (lim.wrist_flag_m - r), "flag": flag, "band_back_tip_m": list(tip),
           "enabled": lim.left_reach_enabled, "hint": None}
    if not flag:
        return out
    target = lim.wrist_flag_m - lim.hint_margin_m
    hint = {}
    for k in range(0, 301):                       # move toward -x in 1 mm steps
        if _wrist_radius(geom, lim, dxyz=(-0.001 * k, 0.0, 0.0))[0] <= target:
            hint["move_x_mm"] = _round_up(k)
            break
    ss, se, sc = xyz(doc.get("seam_start")), xyz(doc.get("seam_end")), xyz(doc.get("seam_center"))
    if ss and se and sc:
        yaw = _yaw_deg(ss, se)
        for k in range(1, 31):                    # turn anticlockwise about the seam centre in 0.5 deg steps
            deg = 0.5 * k
            if yaw + deg > lim.region_yaw_deg[1]:
                break
            if _wrist_radius(geom, lim, rot_deg=deg, pivot=sc)[0] <= target:
                hint["turn_ccw_deg"] = math.ceil(deg)
                hint["turn_to_yaw_deg"] = yaw + math.ceil(deg)
                break
    out["hint"] = hint or None
    return out


def _left_reach_text(lr: dict) -> Tuple[str, str]:
    text = (f"The left arm probably cannot reach the tape on the left face ('left varying orientation search failed closed'): "
            f"wrist centre {lr['wrist_radius_mm']:.0f} mm from the shoulder at the back end of the tape band, "
            f"flag above {lr['flag_radius_mm']:.0f} mm (arm limit {lr['envelope_mm']:.0f} mm).")
    h = lr.get("hint") or {}
    parts = []
    if "move_x_mm" in h:
        parts.append(f"move the carton about {h['move_x_mm']} mm toward -x (toward the arms)")
    if "turn_ccw_deg" in h:
        parts.append(f"turn it anticlockwise (seen from above) by about {h['turn_ccw_deg']} deg, to seam yaw {h['turn_to_yaw_deg']:.0f}")
    hint = ("To fix: " + ", or ".join(parts) + ".") if parts else "Move the carton closer to the arms (toward -x)."
    return text, hint


# ---------------------------------------------------------------------------------------------------- right holder / table
def right_table_risk(centre_x: float, yaw_deg: float, limits: Optional[PlacementLimits] = None) -> dict:
    lim = limits or DEFAULT_LIMITS
    flag = lim.right_table_enabled and centre_x <= lim.right_table_x_max_m and yaw_deg <= lim.right_table_yaw_max_deg
    out = {"flag": bool(flag), "enabled": lim.right_table_enabled, "x_max_m": lim.right_table_x_max_m,
           "yaw_max_deg": lim.right_table_yaw_max_deg, "hint": None}
    if flag:
        out["hint"] = (f"move the carton about {_round_up(1000 * (lim.right_table_x_max_m - centre_x) + 1)} mm toward +x "
                       f"(away from the arms), or turn it anticlockwise (seen from above) by about "
                       f"{math.ceil(lim.right_table_yaw_max_deg - yaw_deg + 0.5)} deg")
    return out


# --------------------------------------------------------------------------------------------------------------- all
def assess_placement(doc: dict, limits: Optional[PlacementLimits] = None) -> Optional[dict]:
    """Everything the page and check_placement.py show for one snapshot document.  None if the seam points are missing.
    ``warnings`` is the list to display, most important first; ``info_only`` is always true."""
    lim = limits or DEFAULT_LIMITS
    base = compute_placement(doc.get("seam_start"), doc.get("seam_end"), doc.get("seam_center"), lim)
    if base is None:
        return None
    warnings: List[dict] = []
    tar = None
    if lim.two_arm_reach_enabled:
        try:
            tar = placement_advisor.two_arm_reach(doc, lim)
        except Exception:      # information only: a model failure must never cost the rest of the assessment
            tar = None
        if tar and tar["status"] != "ok":
            warnings.append({"id": "two_arm_reach", "level": "warn" if tar["status"] == "fail" else "note",
                             "title": ("Both arms probably cannot cut this carton" if tar["status"] == "fail"
                                       else "Two-arm reach not validated for this turn" if tar["status"] == "unvalidated"
                                       else "Two-arm reach is marginal"),
                             "text": tar["text"], "hint": tar["hint"]})
    lr = left_face_reach(doc, lim) if lim.left_reach_enabled else None
    if lr and lr["flag"]:
        text, hint = _left_reach_text(lr)
        warnings.append({"id": "left_reach", "level": "warn", "title": "Left face probably out of reach", "text": text, "hint": hint})
    rt = right_table_risk(base["centre_x_m"], base["yaw_deg"], lim)
    if rt["flag"]:
        warnings.append({"id": "right_holder_table", "level": "note", "title": "Right holder may be refused (table)",
                         "text": (f"Close cartons (centre x <= {lim.right_table_x_max_m:.3f} m) turned to yaw <= {lim.right_table_yaw_max_deg:.1f} deg "
                                  f"were refused today because the right holder touched the table edge (this carton: x {base['centre_x_m']:.3f}, "
                                  f"yaw {base['yaw_deg']:.1f}). May already be fixed if the table is in the planning scene."),
                         "hint": "To avoid it: " + rt["hint"] + "."})
    reg = base["region"]
    if not reg["inside"]:
        what = "; ".join(f"{d['axis']} {d['value']:.3f}" + (" deg" if d["axis"] == "yaw" else " m") for d in reg["deviations"])
        warnings.append({"id": "outside_region", "level": "note", "title": "Outside the region that planned today",
                         "text": (f"Outside the box where today's detections planned (centre x {lim.region_x_m[0]:.3f}..{lim.region_x_m[1]:.3f}, "
                                  f"y {lim.region_y_m[0]:.3f}..{lim.region_y_m[1]:.3f} m, seam yaw {lim.region_yaw_deg[0]:.1f}..{lim.region_yaw_deg[1]:.1f}): {what}. "
                                  "Untested placement: the plan may fail."),
                         "hint": "To get inside: " + "; ".join(d["hint"] for d in reg["deviations"]) + "."})
    base.update({"two_arm_reach": tar, "left_reach": lr, "right_table": rt, "warnings": warnings, "info_only": True,
                 "limits": lim.to_dict()})
    return base
