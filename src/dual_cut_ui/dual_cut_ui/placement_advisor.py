"""Two-arm reach model and placement advisor.  INFORMATIONAL ONLY: nothing here may block detection, planning or execution
(operator decision); the planner stays the authority.  No ROS, numpy only.

Port of the closed-form wrist model fitted in ``scan_results/negative_yaw_20261107`` (harness/wrist_model.py, model_map.py).
The wrist (link6 origin) sits at ``tip + R * P_TOOL`` (P_TOOL = tool offset, ~90 mm behind the tip along travel and ~157 mm out
from the face), where R runs over the orientations the executor's varying-orientation search tries (tilt about "along", pitch
about "across", yaw about the face normal, on top of the reference frame X = -normal, Y = -travel, Z = X x Y).  The binding
constraints are the START corners of each arm's travel (left arm: back_left -> front_left; right arm: front_right ->
back_right):

* left:  min over the allowed orientations of the horizontal wrist distance from the left base at back_left  <= left_wrist_max_m
* right: max over the allowed orientations of the horizontal wrist distance from the right base at front_right >= right_wrist_min_m
* tip guards: front_left tip >= left_tip_min_m from the left base, front_right tip >= right_tip_min_m from the right base.
* (2026-11-08, fitted on three more carton sizes) the right arm also has to REACH its travel: the wrist at the far end of the right
  face (back_right) must be able to come within right_far_max_m of the right base (min over the orientations, the mirror of the left
  start corner), and at the start corner (front_right) within right_start_max_m;
* and the carton must not be too close to the arms: the smallest x of its four top corners (world frame = left base, the arms sit at
  x ~ 0) >= carton_min_x_m.

Only the snapshot's own corners and surface normal are used, so any box size works.  Thresholds are in ``PlacementLimits`` (fitted on
the 2026-11-07 template sweeps and the 2026-11-07 box-size sweep, see scan_results/placement_advisor_20261107/README.md).

Advice: if the model fails (or passes with less than ``reach_marginal_mm``), search for the cheapest corrective action that
satisfies all four constraints with ``advice_margin_mm`` to spare: (a) a translation in the table plane, (b) a turn about the
top-face centre, (c) both.  Cost = how far the most-moved corner travels (mm); clockwise turns cost ``advice_cw_cost_factor``
times more (the study shows anticlockwise is the good direction).  ``lim`` is any object with the PlacementLimits fields."""
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

CORNER_KEYS = ("front_left", "front_right", "back_right", "back_left")
_CHUNK = 4_000_000      # max elements of one (shifts x orientations) distance block


def _pt(v) -> Optional[np.ndarray]:
    if isinstance(v, dict):
        vals = [v.get(k) for k in "xyz"]
    elif isinstance(v, (list, tuple)) and len(v) >= 3:
        vals = list(v[:3])
    else:
        return None
    try:
        a = np.array([float(c) for c in vals])
    except (TypeError, ValueError):
        return None
    return a if np.all(np.isfinite(a)) else None


def pose_from_doc(doc: dict) -> Optional[dict]:
    """{"c": {name: xyz}, "n": unit normal pointing up} from a snapshot document; None if a corner is missing."""
    c = {k: _pt(doc.get(k)) for k in CORNER_KEYS}
    if any(v is None for v in c.values()):
        return None
    n = _pt(doc.get("surface_normal"))
    if n is None or np.linalg.norm(n) < 1e-6:
        n = np.cross(c["back_right"] - c["back_left"], c["front_left"] - c["back_left"])
    if np.linalg.norm(n) < 1e-9:
        return None
    n = n / np.linalg.norm(n)
    if n[2] < 0:
        n = -n
    return {"c": c, "n": n}


def _rot(axis: np.ndarray, deg) -> np.ndarray:
    """Rodrigues rotation matrices about ``axis`` for an array of angles (deg) -> (A, 3, 3)."""
    a = axis / np.linalg.norm(axis)
    t = np.radians(np.atleast_1d(np.asarray(deg, float)))
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3)[None] + np.sin(t)[:, None, None] * K[None] + (1 - np.cos(t))[:, None, None] * (K @ K)[None]


def _angles(max_deg: float, step: float) -> np.ndarray:
    k = int(max_deg // step + 1e-9)
    return step * np.arange(-k, k + 1)


def wrist_offsets(u: np.ndarray, n: np.ndarray, lim) -> np.ndarray:
    """(K, 3) wrist minus tip, metres, over the allowed orientations for a travel direction ``u`` on a face with normal ``n``."""
    u = u - n * (u @ n)
    nu = np.linalg.norm(u)
    if nu < 1e-9:
        return np.zeros((0, 3))
    u = u / nu
    across = np.cross(u, n)
    ref = np.stack([-n, -u, np.cross(-n, -u)], 1)
    Ry = _rot(n, _angles(lim.orient_yaw_max_deg, lim.orient_step_deg))
    Rp = _rot(across, _angles(lim.orient_pitch_max_deg, lim.orient_step_deg))
    Rt = _rot(u, _angles(lim.orient_tilt_max_deg, lim.orient_step_deg))
    R = (Ry[None, None, :] @ Rp[None, :, None] @ Rt[:, None, None] @ ref).reshape(-1, 3, 3)      # [tilt, pitch, yaw]
    thp = np.degrees(np.arccos(np.minimum(1.0, np.abs(R[:, :, 0] @ n))))
    dotx = R[:, :, 0] @ (-n)
    dotz = R[:, :, 2] @ (-n)
    ok = (thp <= lim.blade_cap_deg + 1e-9) & ((dotx >= lim.penetration_min_dot) | (dotz >= lim.penetration_min_dot))
    p_tool = np.asarray(lim.tool_offset_mm, float) / 1000.0
    return R[ok] @ p_tool


class _Geometry:
    """Everything the constraints need for one carton orientation; translations are then cheap (orientation offsets
    do not depend on where the carton is)."""

    def __init__(self, pose: dict, lim):
        c, n = pose["c"], pose["n"]
        self.lim = lim
        self.BL = np.asarray(lim.left_base_m, float)[:2]
        self.BR = np.asarray(lim.right_base_m, float)[:2]
        self.qL = c["back_left"][:2]
        self.qR = c["front_right"][:2]
        self.tipL = c["front_left"][:2]
        self.tipR = c["front_right"][:2]
        self.qRfar = c["back_right"][:2]
        self.xmin = float(min(v[0] for v in c.values()))
        WL = wrist_offsets(c["front_left"] - c["back_left"], n, lim)[:, :2]
        WR = wrist_offsets(c["back_right"] - c["front_right"], n, lim)[:, :2]
        self.WL = np.unique(np.round(WL, 5), axis=0) if len(WL) else WL
        self.WR = np.unique(np.round(WR, 5), axis=0) if len(WR) else WR

    def margins(self, shifts: np.ndarray) -> np.ndarray:
        """(G, 7) margins in metres, columns as NAMES, for carton shifts (G, 2) in metres.  Positive = constraint met."""
        lim = self.lim
        S = np.atleast_2d(shifts).astype(float)
        G = len(S)
        out = np.empty((G, len(NAMES)))
        if len(self.WL):
            cen = self.BL - self.qL - S                                  # wrist offsets must come within TL of this point
            step = max(1, _CHUNK // len(self.WL))
            for i in range(0, G, step):
                d = cen[i:i + step, None, :] - self.WL[None, :, :]
                out[i:i + step, 0] = lim.left_wrist_max_m - np.sqrt(np.einsum("gkc,gkc->gk", d, d).min(1))
        else:
            out[:, 0] = -np.inf
        if len(self.WR):
            cen = self.BR - self.qR - S
            d = cen[:, None, :] - self.WR[None, :, :]
            out[:, 1] = np.sqrt(np.einsum("gkc,gkc->gk", d, d).max(1)) - lim.right_wrist_min_m
            out[:, 5] = lim.right_start_max_m - np.sqrt(np.einsum("gkc,gkc->gk", d, d).min(1))
            cen = self.BR - self.qRfar - S
            d = cen[:, None, :] - self.WR[None, :, :]
            out[:, 4] = lim.right_far_max_m - np.sqrt(np.einsum("gkc,gkc->gk", d, d).min(1))
        else:
            out[:, 1] = out[:, 4] = out[:, 5] = -np.inf
        out[:, 2] = np.hypot(*(self.tipL + S - self.BL).T) - lim.left_tip_min_m
        out[:, 3] = np.hypot(*(self.tipR + S - self.BR).T) - lim.right_tip_min_m
        out[:, 6] = self.xmin + S[:, 0] - lim.carton_min_x_m
        return out


NAMES = ("left_wrist", "right_wrist", "left_tip", "right_tip", "right_far", "right_start", "close")


def evaluate(pose: dict, lim) -> dict:
    """The model verdict for the carton as it lies: margins in mm per constraint, the smallest one, and ok = all met."""
    m = _Geometry(pose, lim).margins(np.zeros((1, 2)))[0] * 1000.0
    k = int(np.argmin(m))
    return {"margins_mm": {n: float(v) for n, v in zip(NAMES, m)}, "margin_mm": float(m[k]), "binding": NAMES[k], "ok": bool(m[k] >= 0)}


# ------------------------------------------------------------------------------------------------------------- moving
def centre_of(pose: dict) -> np.ndarray:
    return np.mean([pose["c"][k] for k in CORNER_KEYS], axis=0)


def turned(pose: dict, deg: float) -> dict:
    """The carton turned anticlockwise (seen from above) by ``deg`` about the vertical axis through its top-face centre."""
    a = math.radians(deg)
    R = np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])
    ctr = centre_of(pose)
    return {"c": {k: ctr + R @ (v - ctr) for k, v in pose["c"].items()}, "n": R @ pose["n"]}


def _corner_travel_mm(pose: dict, deg: float, shift_mm=(0.0, 0.0)) -> float:
    """How far the most-moved top corner goes for a turn of ``deg`` about the centre followed by the shift (mm)."""
    ctr = centre_of(pose)
    a = math.radians(deg)
    best = 0.0
    for v in pose["c"].values():
        r = (v - ctr)[:2]
        r2 = np.array([math.cos(a) * r[0] - math.sin(a) * r[1], math.sin(a) * r[0] + math.cos(a) * r[1]]) + np.asarray(shift_mm) / 1000.0
        best = max(best, float(np.linalg.norm(r2 - r)))
    return 1000.0 * best


def _grid(max_mm: float, step_mm: float) -> np.ndarray:
    k = int(max_mm // step_mm + 1e-9)
    ax = step_mm * np.arange(-k, k + 1)
    gx, gy = np.meshgrid(ax, ax, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel()], 1)          # mm, (G, 2)


def _best_shift(geo: _Geometry, grid_mm: np.ndarray, need_m: float, rot_cost_mm: float, pose, deg, cw_factor=1.0):
    """Cheapest feasible shift of ``grid_mm`` for one orientation: (cost, shift_mm, margin_mm) or None."""
    m = geo.margins(grid_mm / 1000.0).min(1)
    ok = m >= need_m
    if not ok.any():
        return None
    idx = np.flatnonzero(ok)
    # cost = |shift| (mm) + the corner travel of the turn; the most-moved corner of turn+shift is evaluated exactly for the winner only
    cost = np.hypot(*grid_mm[idx].T) + rot_cost_mm
    j = int(np.argmin(cost))
    sh = grid_mm[idx[j]]
    return float(cost[j]), sh, float(m[idx[j]] * 1000.0)


# ------------------------------------------------------------------------------------------------------------- wording
def _shift_words(dx_mm: float, dy_mm: float) -> str:
    parts = []
    if abs(dx_mm) >= 0.5:
        parts.append(f"{abs(dx_mm):.0f} mm " + ("toward the arms (-x)" if dx_mm < 0 else "away from the arms (+x)"))
    if abs(dy_mm) >= 0.5:
        parts.append(f"{abs(dy_mm):.0f} mm " + ("toward the right arm (-y)" if dy_mm < 0 else "toward the left arm (+y)"))
    return " and ".join(parts)


def _turn_words(deg: float) -> str:
    return f"{abs(deg):.0f} deg " + ("anticlockwise" if deg > 0 else "clockwise") + " (seen from above, about the carton centre)"


def option_text(opt: dict) -> str:
    mv = _shift_words(opt["dx_mm"], opt["dy_mm"])
    tn = _turn_words(opt["turn_deg"]) if abs(opt["turn_deg"]) > 0 else ""
    if tn and mv:
        return f"turn it {tn}, then move it {mv}"
    return f"turn it {tn}" if tn else f"move it {mv}"


def _option(kind: str, dx: float, dy: float, turn: float, margin_mm: float, cost: float) -> dict:
    o = {"kind": kind, "dx_mm": float(dx), "dy_mm": float(dy), "turn_deg": float(turn), "margin_mm": float(margin_mm), "cost_mm": float(cost)}
    o["text"] = option_text(o)
    return o


def apply_option(pose: dict, opt: dict) -> dict:
    """The carton after the option (turn about the centre, then move); for checking an option against the model."""
    p = turned(pose, opt["turn_deg"]) if opt["turn_deg"] else pose
    d = np.array([opt["dx_mm"] / 1000.0, opt["dy_mm"] / 1000.0, 0.0])
    return {"c": {k: v + d for k, v in p["c"].items()}, "n": p["n"]}


def turn_from_ref_deg(pose: dict, lim) -> float:
    """Anticlockwise turn of the carton relative to the validated template pose (left face travel direction), [-180, 180)."""
    d = pose["c"]["front_left"] - pose["c"]["back_left"]
    a = math.degrees(math.atan2(d[1], d[0])) - lim.model_face_yaw_ref_deg
    return (a + 180.0) % 360.0 - 180.0


# --------------------------------------------------------------------------------------------------------------- search
def advise(pose: dict, lim) -> dict:
    """Options that satisfy the model with ``advice_margin_mm`` to spare; ``options`` is cheapest first (empty = none found)."""
    need = lim.advice_margin_mm / 1000.0
    grid = _grid(lim.advice_max_shift_mm, lim.advice_shift_step_mm)
    cur = turn_from_ref_deg(pose, lim)
    turns = [t for t in _angles(lim.advice_max_turn_deg, lim.advice_turn_step_deg)
             if t != 0 and cur + t <= lim.model_valid_turn_deg[1] + 1e-9]       # never advise into the unvalidated range
    cw = lim.advice_cw_cost_factor
    options: List[dict] = []
    # (a) translation only
    geo0 = _Geometry(pose, lim)
    r = _best_shift(geo0, grid, need, 0.0, pose, 0.0)
    if r:
        options.append(_option("move", r[1][0], r[1][1], 0.0, r[2], r[0]))
    # (b) rotation only (about the centre), both directions
    zero = np.zeros((1, 2))
    best_turn = None
    geos: Dict[float, _Geometry] = {}
    for t in turns:
        g = _Geometry(turned(pose, t), lim)
        geos[t] = g
        m = float(g.margins(zero).min())
        if m >= need:
            cost = _corner_travel_mm(pose, t) * (cw if t < 0 else 1.0)
            if best_turn is None or cost < best_turn[0]:
                best_turn = (cost, t, m * 1000.0)
    if best_turn:
        options.append(_option("turn", 0.0, 0.0, best_turn[1], best_turn[2], best_turn[0]))
    # (c) both, only if neither alone works
    if not options:
        coarse_g = _grid(lim.advice_max_shift_mm, 2 * lim.advice_shift_step_mm)
        cands = []
        for t in turns[::2] if len(turns) > 8 else turns:
            g = geos[t]
            rc = _corner_travel_mm(pose, t) * (cw if t < 0 else 1.0)
            r = _best_shift(g, coarse_g, need, rc, pose, t)
            if r:
                cands.append((r[0], t, r[1]))
        cands.sort(key=lambda x: x[0])
        best = None
        fine_shift = np.array([[a, b] for a in (-10, 0, 10) for b in (-10, 0, 10)], float)
        for _, t0, sh0 in cands[:3]:
            for t in sorted({t0 + d for d in (-1, 0, 1)} & set(geos)):
                g = geos[t]
                rc = _corner_travel_mm(pose, t) * (cw if t < 0 else 1.0)
                r = _best_shift(g, sh0 + fine_shift, need, rc, pose, t)
                if r and (best is None or r[0] < best[0]):
                    best = (r[0], t, r[1], r[2])
        if best:
            options.append(_option("turn_move", best[2][0], best[2][1], best[1], best[3], best[0]))
    options.sort(key=lambda o: o["cost_mm"])
    return {"options": options, "searched": {"shift_mm": lim.advice_max_shift_mm, "turn_deg": lim.advice_max_turn_deg,
                                             "margin_mm": lim.advice_margin_mm}}


# ------------------------------------------------------------------------------------------------------------ top level
_LABEL = {"left_wrist": "left arm wrist at the back-left start corner (must stay within {lim:.0f} mm of its base)",
          "right_wrist": "right arm wrist at the front-right start corner (must stay at least {lim:.0f} mm from its base)",
          "left_tip": "left blade tip at the front-left corner (at least {lim:.0f} mm from the left base)",
          "right_tip": "right blade tip at the front-right corner (at least {lim:.0f} mm from the right base)",
          "right_far": "right arm wrist at the back-right far end of its travel (must be able to come within {lim:.0f} mm of its base)",
          "right_start": "right arm wrist at the front-right start corner (must be able to come within {lim:.0f} mm of its base)",
          "close": "carton too close to the arms (its nearest corner must be at least {lim:.0f} mm in front of the arm bases, x)"}


def two_arm_reach(doc: dict, lim) -> Optional[dict]:
    """The model verdict plus advice for one snapshot document; None if the corners are missing.  Never raises on odd data."""
    pose = pose_from_doc(doc)
    if pose is None:
        return None
    ev = evaluate(pose, lim)
    status = "ok" if ev["margin_mm"] >= lim.reach_marginal_mm else ("marginal" if ev["ok"] else "fail")
    turn = turn_from_ref_deg(pose, lim)
    validated = lim.model_valid_turn_deg[0] <= turn <= lim.model_valid_turn_deg[1]
    out = dict(ev, status=status, enabled=lim.two_arm_reach_enabled, marginal_mm=lim.reach_marginal_mm, advice=None, text="", hint="",
               turn_from_ref_deg=turn, validated=validated)
    if status != "fail" and not validated:
        out["status"] = "unvalidated"
        out["text"] = (f"The carton is turned {turn:+.0f} deg from the tested pose, outside the range where the reach model was checked "
                       f"against the planner ({lim.model_valid_turn_deg[0]:+.0f}..{lim.model_valid_turn_deg[1]:+.0f} deg). The model "
                       "says both arms reach, but the planner refused many such placements in mock.")
        out["hint"] = (f"Turning it about {turn - lim.model_valid_turn_deg[1]:.0f} deg clockwise brings it back into the tested range."
                       if turn > lim.model_valid_turn_deg[1] else "")
        return out
    if status == "ok":
        return out
    lims_mm = {"left_wrist": lim.left_wrist_max_m, "right_wrist": lim.right_wrist_min_m, "left_tip": lim.left_tip_min_m, "right_tip": lim.right_tip_min_m,
              "right_far": lim.right_far_max_m, "right_start": lim.right_start_max_m, "close": lim.carton_min_x_m}
    bad = [k for k, v in ev["margins_mm"].items() if v < lim.reach_marginal_mm]
    if status == "fail":
        what = "; ".join(f"{_LABEL[k].format(lim=1000 * lims_mm[k])}: short by {-ev['margins_mm'][k]:.0f} mm" for k in bad if ev["margins_mm"][k] < 0)
        out["text"] = ("The two arms probably cannot both cut this carton (model estimate, the planner decides): " + what + ".")
    else:
        what = "; ".join(f"{k.replace('_', ' ')} {ev['margins_mm'][k]:.0f} mm" for k in bad)
        out["text"] = f"Both arms probably reach, but with little to spare (margin {ev['margin_mm']:.0f} mm, wanted {lim.reach_marginal_mm:.0f}): {what}."
    adv = advise(pose, lim) if lim.advice_enabled else None
    out["advice"] = adv
    if adv is None:
        out["hint"] = ""
    elif adv["options"]:
        opts = adv["options"][:2]
        out["hint"] = "To fix: " + " ; or ".join(f"{o['text']} (model margin {o['margin_mm']:.0f} mm)" for o in opts) + "."
    else:
        word = "placement" if status == "fail" else "placement with a clearly better margin"
        out["hint"] = (f"No {word} within ±{lim.advice_max_shift_mm:.0f} mm / ±{lim.advice_max_turn_deg:.0f} deg found"
                       + ("; the carton is probably too large or too far for this cell." if status == "fail" else "."))
    return out
