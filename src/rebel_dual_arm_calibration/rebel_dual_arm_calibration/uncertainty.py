"""Monte Carlo uncertainty budget of the written calibration (``solve --uncertainty-mc N``; numpy/scipy only).

The detection-noise standard deviations of the fit do not contain the systematic (GUM "Type B") terms.  This module
re-solves the problem N times from the written solution, each time with the data perturbed by draws of:

  * print scale along x and y of EACH board (left target, right target, table board) ~ N(1, 0.15 %) independent;
    the table-board scale also scales the board touch coordinates;
  * focal length (fx, fy together) ~ N(1, 0.5 %) and principal point (cx, cy independent) ~ N(0, 3 px)
    (0.1 % / 1 px if an intrinsics file is used);
  * TCP offset ~ N(0, 0.5 mm) per axis per arm (systematic: the same draw for all touches of an arm, in the tool frame);
  * touch noise from the MEASURED repeat scatter of the touches (per arm, per axis);
  * joint zero offsets ~ N(0, 0.3 deg) per joint and arm (systematic) - only if the written fit does not estimate them.

The numbers are 1-sigma values of Gaussian draws.  Every draw is a warm-started re-fit of the final stage (fixed inlier
sets and corner masks, no validation, no loop-closure variants, no covariance) so that it costs about one fit
iteration set (about 1-3 s with 70 samples), i.e. N draws take about N x 2 s.  The 95 % intervals are the 2.5 / 97.5
percentiles of the draws (use N >= 200 for a stable interval; N = 20 only shows the order of magnitude).
"""
from __future__ import annotations

import dataclasses
import math
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from .transforms import T_to_xyz_rpy, inv_T

ARMS = ("left", "right")


def _scale_obj(obj: np.ndarray, sx: float, sy: float) -> np.ndarray:
    o = np.array(obj, float)
    o[:, 0] *= sx
    o[:, 1] *= sy
    return o


def workspace_points(T_R: np.ndarray, ref_world: Optional[np.ndarray], reach_m: float = 0.4) -> Dict[str, np.ndarray]:
    """Three workspace points (world frame): the carton centre ~0.35 m in front of the bases, midway between them
    (``in front`` = towards the touched board), and the reach point (``reach_m``) of each arm towards the board."""
    t_r = T_R[:3, 3]
    mid = t_r / 2.0
    base = np.array([t_r[0], t_r[1], 0.0])
    f = np.array([-base[1], base[0], 0.0])
    f = f / np.linalg.norm(f) if np.linalg.norm(f) > 1e-9 else np.array([1.0, 0.0, 0.0])
    toward = (ref_world - mid) if ref_world is not None else np.array([1.0, 0.0, 0.0])
    if float(np.dot(f, toward)) < 0:
        f = -f
    carton = mid + 0.35 * f
    carton[2] = float(ref_world[2]) if ref_world is not None else 0.0

    def reach(origin):
        d = (ref_world - origin) if ref_world is not None else f
        d = np.array([d[0], d[1], 0.0])
        d = d / np.linalg.norm(d) if np.linalg.norm(d) > 1e-9 else f
        p = origin + reach_m * d
        p[2] = float(ref_world[2]) if ref_world is not None else 0.0
        return p
    return {"carton_centre": carton, "left_reach": reach(np.zeros(3)), "right_reach": reach(np.array([t_r[0], t_r[1], 0.0]))}


def run_mc(res, n: int, intrinsics_file: bool = False, seed: int = 20261008, sigma_scale: float = 0.0015,
           sigma_f: Optional[float] = None, sigma_c_px: Optional[float] = None, sigma_tcp_mm: float = 0.5,
           sigma_joint_deg: float = 0.3, progress=None) -> dict:
    """Monte Carlo of the written solution ``res`` (a ``CalibrationResult`` after ``set_written_base``)."""
    from . import solver as S
    if n <= 0:
        raise ValueError("n must be > 0")
    sigma_f = (0.001 if intrinsics_file else 0.005) if sigma_f is None else sigma_f
    sigma_c_px = (1.0 if intrinsics_file else 3.0) if sigma_c_px is None else sigma_c_px
    fit = res.fit if getattr(res, "written_fit", "offset_fit") == "offset_fit" else res.fit_nominal
    model, flags, options, kin = fit.model, fit.model.flags, res.options, res.kin
    T_cl_o = res.T_camlink_optical
    touches = list(res.touches_fit) if flags.touches else []
    rng_master = np.random.default_rng(seed)
    scatter = ((res.report.get("touch_base") or {}).get("noise") or {}).get("scatter") or {}
    sig_touch = {a: (scatter.get(a) or {}).get("sigma_axis_mm", 0.0) * 1e-3 for a in ARMS}
    touch_noise_assumed = any((scatter.get(a) or {}).get("assumed", True) for a in ARMS)
    perturb_joints = (not flags.offsets) and kin is not None
    notes: List[str] = []
    if not flags.offsets and kin is None:
        notes.append("no URDF kinematics in the sessions: the joint zero offset term (+-0.3 deg) was NOT included")
    # fixed joint-offset bookkeeping: A = T_recorded FK(q)^-1 per item (identity for an exact TF = FK)
    A_s, A_t = {}, {}
    if perturb_joints:
        for k, sm in enumerate(model.samples):
            if sm.q is not None and sm.arm in kin:
                A_s[k] = sm.T_base_link6 @ np.linalg.inv(kin[sm.arm].chain6.fk(sm.q))
        for k, t in enumerate(touches):
            if t.q is not None and t.arm in kin and kin[t.arm].chain_tcp is not None:
                A_t[k] = t.T_base_tcp @ np.linalg.inv(kin[t.arm].chain_tcp.fk(t.q))
        if len(A_s) < len(model.samples):
            notes.append("some samples have no joint values matching the URDF: joint-offset term applied to the "
                         f"{len(A_s)} of {len(model.samples)} that have them")
    base_ref = fit.params.T_W_R
    ref_world = None
    cp_ref = (res.report.get("check_points") or {}).get("reference")
    if cp_ref:
        ref_world = np.array(cp_ref["centre"])
    elif fit.params.T_W_B is not None:
        ref_world = fit.params.T_W_B[:3, 3]
    pts = workspace_points(base_ref, ref_world)
    pts_r = {k: inv_T(base_ref)[:3, :3] @ p + inv_T(base_ref)[:3, 3] for k, p in pts.items()}
    rows_R, rows_C, rows_D = [], [], {k: [] for k in pts}
    n_fail, t0 = 0, time.time()
    for it in range(n):
        rng = np.random.default_rng(rng_master.integers(1 << 62))
        board_scale = {b: (1.0 + rng.normal(0, sigma_scale), 1.0 + rng.normal(0, sigma_scale))
                       for b in ("left", "right", "table")}
        df = 1.0 + rng.normal(0, sigma_f)
        dc = rng.normal(0, sigma_c_px, 2)
        tcp = {a: rng.normal(0, sigma_tcp_mm * 1e-3, 3) for a in ARMS}
        dq = {a: np.radians(rng.normal(0, sigma_joint_deg, 6)) for a in ARMS}
        dK = lambda K: _perturb_K(K, df, dc)  # noqa: E731
        new_s, new_t, new_touch, masks = [], [], [], {}
        for k, sm in enumerate(model.samples):
            sx, sy = board_scale[sm.arm]
            T6 = sm.T_base_link6
            if k in A_s:
                T6 = A_s[k] @ kin[sm.arm].chain6.fk(sm.q + dq[sm.arm])
            c = dataclasses.replace(sm, obj=_scale_obj(sm.obj, sx, sy), K=dK(sm.K), T_base_link6=T6)
            new_s.append(c)
            masks[id(c)] = fit.masks[k]
        for k, tb in enumerate(model.tables):
            sx, sy = board_scale["table"]
            c = dataclasses.replace(tb, obj=_scale_obj(tb.obj, sx, sy), K=dK(tb.K))
            new_t.append(c)
            masks[id(c)] = fit.masks[model.n_s + k]
        sx, sy = board_scale["table"]
        for k, t in enumerate(touches):
            T = t.T_base_tcp.copy()
            if k in A_t:
                T = A_t[k] @ kin[t.arm].chain_tcp.fk(t.q + dq[t.arm])
            T[:3, 3] += T[:3, :3] @ tcp[t.arm] + rng.normal(0, sig_touch[t.arm], 3)
            p = np.array(t.p_board, float)
            p[0] *= sx
            p[1] *= sy
            new_touch.append(dataclasses.replace(t, T_base_tcp=T, p_board=p))
        try:
            f = S._joint_refine(new_s, new_t, new_touch, fit.params, flags, options, kin, masks)
        except S.SolverError:
            n_fail += 1
            continue
        T_R = f.params.T_W_R
        T_C = f.params.T_W_O @ inv_T(T_cl_o)
        a, b = T_to_xyz_rpy(T_R), T_to_xyz_rpy(T_C)
        rows_R.append(np.concatenate([a[0] * 1e3, np.degrees(a[1])]))
        rows_C.append(np.concatenate([b[0] * 1e3, np.degrees(b[1])]))
        for kname, p_r in pts_r.items():
            rows_D[kname].append((T_R[:3, :3] @ p_r + T_R[:3, 3] - pts[kname]) * 1e3)
        if progress:
            progress(it + 1, n, time.time() - t0)
    if not rows_R:
        raise S.SolverError("uncertainty Monte Carlo: every draw failed")
    R, C = np.array(rows_R), np.array(rows_C)
    labels = ["x_mm", "y_mm", "z_mm", "roll_deg", "pitch_deg", "yaw_deg"]
    ref_R = np.concatenate([T_to_xyz_rpy(base_ref)[0] * 1e3, np.degrees(T_to_xyz_rpy(base_ref)[1])])
    T_cam_ref = fit.params.T_W_O @ inv_T(T_cl_o)
    ref_C = np.concatenate([T_to_xyz_rpy(T_cam_ref)[0] * 1e3, np.degrees(T_to_xyz_rpy(T_cam_ref)[1])])

    def interval(M, ref):
        lo, hi = np.percentile(M, 2.5, axis=0), np.percentile(M, 97.5, axis=0)
        return {l: {"written": float(r), "median": float(m), "lo95": float(a), "hi95": float(b),
                    "half_width95": float((b - a) / 2.0), "std": float(s)}
                for l, r, m, a, b, s in zip(labels, ref, np.median(M, axis=0), lo, hi, M.std(axis=0))}

    disagreement = {}
    for k, rows in rows_D.items():
        D = np.array(rows)
        nrm = np.linalg.norm(D, axis=1)
        disagreement[k] = {"point_world_m": [float(v) for v in pts[k]],
                           "norm_mean_mm": float(nrm.mean()), "norm_p95_mm": float(np.percentile(nrm, 95)),
                           "norm_max_mm": float(nrm.max()),
                           "axis_lo95_mm": [float(v) for v in np.percentile(D, 2.5, axis=0)],
                           "axis_hi95_mm": [float(v) for v in np.percentile(D, 97.5, axis=0)]}
    return {"n_requested": n, "n_used": len(R), "n_failed": n_fail, "seed": seed,
            "runtime_s": time.time() - t0, "written_fit": getattr(res, "written_fit", "offset_fit"),
            "terms": {"print_scale_sigma_frac": sigma_scale, "focal_sigma_frac": sigma_f,
                      "principal_point_sigma_px": sigma_c_px, "intrinsics_file": bool(intrinsics_file),
                      "tcp_sigma_mm_per_axis": sigma_tcp_mm,
                      "joint_offset_sigma_deg": sigma_joint_deg if perturb_joints else None,
                      "touch_noise_sigma_axis_mm": {a: v * 1e3 for a, v in sig_touch.items()},
                      "touch_noise_assumed": touch_noise_assumed, "boards": ["left", "right", "table"]},
            "arm_right_base": interval(R, ref_R), "camera_link": interval(C, ref_C),
            "tcp_disagreement": disagreement, "notes": notes,
            "statement": "1-sigma Gaussian Type-B terms, 95 % interval = 2.5/97.5 percentile of the re-solves; "
                         "covers print scale, intrinsics, TCP offset, touch noise and (if not fitted) joint zero "
                         "offsets; not link-length / axis errors, flatness or camera-mount flexure"}


def _perturb_K(K: np.ndarray, df: float, dc: np.ndarray) -> np.ndarray:
    K = np.array(K, float).copy()
    K[0, 0] *= df
    K[1, 1] *= df
    K[0, 2] += dc[0]
    K[1, 2] += dc[1]
    return K


def format_mc(mc: dict) -> List[str]:
    t = mc["terms"]
    L = [f"Uncertainty budget (Monte Carlo, {mc['n_used']} of {mc['n_requested']} re-solves, {mc['runtime_s']:.0f} s, "
         f"written fit: {mc['written_fit']}):",
         f"  terms (1 sigma): print scale {100 * t['print_scale_sigma_frac']:.2f} % x and y per board, f "
         f"{100 * t['focal_sigma_frac']:.2f} %, c {t['principal_point_sigma_px']:.1f} px"
         + (" (intrinsics file)" if t["intrinsics_file"] else "") + f", TCP {t['tcp_sigma_mm_per_axis']:.1f} mm/axis, "
         f"touch noise L/R {t['touch_noise_sigma_axis_mm']['left']:.2f}/{t['touch_noise_sigma_axis_mm']['right']:.2f} "
         f"mm/axis{'*' if t['touch_noise_assumed'] else ''}, joint offsets "
         + ("-" if t["joint_offset_sigma_deg"] is None else f"{t['joint_offset_sigma_deg']:.1f} deg")]
    for name in ("arm_right_base", "camera_link"):
        L.append(f"  {name} 95 % intervals (written value, +- half width):")
        for k, v in mc[name].items():
            L.append(f"    {k:<10} {v['written']:+10.3f}  +-{v['half_width95']:.3f}  [{v['lo95']:+.3f} .. {v['hi95']:+.3f}]"
                     f"  median offset {v['median'] - v['written']:+.3f}")
    L.append("  predicted arm-to-arm TCP disagreement from the base uncertainty (a point both arms touch):")
    for k, v in mc["tcp_disagreement"].items():
        L.append(f"    {k:<14} at world {_f3(v['point_world_m'])} m: mean {v['norm_mean_mm']:.2f} mm, 95th percentile "
                 f"{v['norm_p95_mm']:.2f} mm, max {v['norm_max_mm']:.2f} mm")
    L += [f"  note: {n}" for n in mc["notes"]]
    if mc["n_used"] < 100:
        L.append(f"  note: {mc['n_used']} draws give a coarse 95 % interval; use --uncertainty-mc 200 or more for a "
                 "report value")
    L.append("  " + mc["statement"])
    return L


def _f3(v) -> str:
    return "[" + ", ".join(f"{x:.3f}" for x in v) + "]"
