"""Held-out evaluation of the calibration (deterministic k-fold or a single hold-out split).

Every fold is solved from scratch (PnP, closed-form initialisation, per-arm refinement, outlier rejection, joint
fit) on its training samples only - nothing is reused from the full-data fit - and then scored on the samples that
were left out: reprojection error in px, the same error converted to a lateral error in mm at
the target depth, and (if recorded) the touch mismatch in mm.  The evaluation is done once WITHOUT and once WITH
joint zero offsets; the offsets are only recommended if they improve the held-out error.

No ROS imports.
"""
from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import solver as _solver
from .solver import (ARMS, Flags, SolverError, SolverOptions, Sample, Touch, _FullModel, _TouchBlock, _attach_q,
                     _rms)


def fold_assignment(items: Sequence, k: int) -> np.ndarray:
    """Fold index of every item: the n-th item of an arm goes to fold n % k (deterministic, arm-balanced)."""
    counts = {a: 0 for a in ARMS}
    out = np.zeros(len(items), int)
    for i, it in enumerate(items):
        out[i] = counts[it.arm] % k
        counts[it.arm] += 1
    return out


def _heldout_errors(held: Sequence[Sample], fit, offsets: bool, options: SolverOptions, kin) -> Dict[str, object]:
    """Per-point px and mm errors of held-out samples under the parameters of a fold fit.

    Outlier handling of the HELD-OUT samples uses only the fold's own training fit: the thresholds are those of
    ``solver._flag_from_norms`` computed from the training residuals (a corner above max(corner_outlier_px,
    5 sigma_train) is masked, a sample whose median residual is above max(sample_outlier_px, factor x typical
    training sample) is dropped and counted).  Nothing from the full-data fit is used."""
    empty = {"px": np.zeros(0), "mm": np.zeros(0), "n_rejected": 0}
    if not held:
        return empty
    tr = np.concatenate([n[m] for n, m in zip(fit.norms, fit.masks)])
    tr_all = np.concatenate(fit.norms)
    sigma = max(float(np.median(tr_all)) / 1.1774, 0.05)
    med_score = float(np.median([np.median(n) for n in fit.norms[:fit.model.n_s]])) if fit.model.n_s else 0.0
    s_thr = max(options.sample_outlier_px, options.sample_outlier_factor * med_score)
    c_thr = max(options.corner_outlier_px, options.corner_outlier_sigma * sigma)
    model = _FullModel(held, [], [], fit.params, Flags(board=False, offsets=offsets), options, kin)
    zero = np.zeros(model.n_params)
    norms = model.pixel_norm_list(zero)
    T = model.item_poses_p(fit.params)[model.pidx]
    depth = np.einsum("mj,mj->m", T[:, 2, :3], model.obj) + T[:, 2, 3]
    f = np.sqrt(model.K0[model.pidx, 0] * model.K0[model.pidx, 1])
    px, mm, rejected = [], [], 0
    for i, s in enumerate(model.items):
        n = norms[i]
        m = n <= c_thr
        if np.median(n) > s_thr or m.sum() < 4:
            rejected += 1
            continue
        sl = slice(model.starts[i], model.starts[i + 1])
        px.append(n[m])
        mm.append((n * depth[sl] / f[sl] * 1e3)[m])
    return {"px": np.concatenate(px) if px else np.zeros(0), "mm": np.concatenate(mm) if mm else np.zeros(0),
            "n_rejected": rejected, "train_rms_px": _rms(tr)}


def _variant(raw_samples, tables, touches, offsets: bool, options: SolverOptions, kin,
             folds_s: np.ndarray, folds_t: np.ndarray, run_folds: Sequence[int]) -> dict:
    """Fold fits from RAW items: every fold re-initialises (PnP, closed form, per-arm fits) and outlier-filters
    from its own training items only, so the held-out samples influence neither the start values nor the
    inlier selection."""
    opts_v = dataclasses.replace(options, joint_offsets=offsets)
    per_fold, px_all, mm_all, touch_all, touch_nom_all = [], [], [], [], []
    for f in run_folds:
        tr_s = [s for s, ff in zip(raw_samples, folds_s) if ff != f]
        ho_s = [s for s, ff in zip(raw_samples, folds_s) if ff == f and len(s.ids) >= options.min_points]
        tr_t = [t for t, ff in zip(touches, folds_t) if ff != f]
        ho_t = [t for t, ff in zip(touches, folds_t) if ff == f]
        try:
            _, fit = _solver._run_fit(tr_s, opts_v, tables, tr_t, kin)
            err = _heldout_errors(ho_s, fit, offsets, opts_v, kin)
            touch_mm, touch_nom = np.zeros(0), np.zeros(0)
            if ho_t and fit.params.T_W_B is not None:
                mism = _TouchBlock(ho_t, kin, offsets).mismatch(fit.params, offsets)
                touch_mm = np.linalg.norm(mism, axis=1) * 1e3
                # what the deployed robot sees: nominal FK (the fitted offsets are never applied)
                nom = _TouchBlock(ho_t, kin, False).mismatch(fit.params, False)
                touch_nom = np.linalg.norm(nom, axis=1) * 1e3
        except SolverError as e:
            if offsets and any(w in str(e) for w in ("URDF", "joint values", "TCP frame")):
                raise                       # offsets not evaluable at all: reported by cross_validate
            per_fold.append({"fold": int(f), "error": str(e)})
            continue
        per_fold.append({"fold": int(f), "n_heldout_samples": len(ho_s), "n_heldout_rejected": err["n_rejected"],
                         "n_heldout_touches": len(ho_t), "n_train_samples": len(fit.model.samples),
                         "rms_px": _rms(err["px"]), "rms_mm": _rms(err["mm"]), "touch_rms_mm": _rms(touch_mm),
                         "touch_rms_nominal_mm": _rms(touch_nom)})
        px_all.append(err["px"])
        mm_all.append(err["mm"])
        touch_all.append(touch_mm)
        touch_nom_all.append(touch_nom)
    px = np.concatenate(px_all) if px_all else np.zeros(0)
    mm = np.concatenate(mm_all) if mm_all else np.zeros(0)
    tc = np.concatenate(touch_all) if touch_all else np.zeros(0)
    tn = np.concatenate(touch_nom_all) if touch_nom_all else np.zeros(0)
    return {"offsets": offsets, "heldout_rms_px": _rms(px), "heldout_rms_mm": _rms(mm),
            "heldout_touch_rms_mm": _rms(tc), "heldout_touch_rms_nominal_mm": _rms(tn),
            "n_points": int(px.size), "n_heldout_touches": int(tc.size), "folds": per_fold}


def cross_validate(result, options: SolverOptions, touches: Optional[Sequence[Touch]] = None, kin=None,
                   k: int = 5, holdout: Optional[float] = None) -> dict:
    """Held-out evaluation around a ``solver.solve`` result.

    Each fold is solved from the RAW input samples/table images (``result.input_samples`` / ``input_tables``)
    without the fold's held-out samples; initialisation and outlier filtering never see them."""
    raw = list(result.input_samples if result.input_samples is not None else result.fit_samples)
    tables = list(result.input_tables if result.input_tables is not None else result.fit_tables)
    # only the printed board crosshairs take part (free fit / check stickers are never part of the joint solve)
    touches = [t for t in (touches or []) if getattr(t, "kind", "board") == "board"]
    if holdout is not None:
        if not 0.0 < holdout < 0.5:
            raise ValueError("holdout must be a fraction in (0, 0.5)")
        k = max(2, int(round(1.0 / holdout)))
        run_folds, method = [0], f"single hold-out split, {100.0 / k:.0f} % of the samples (every {k}th)"
    else:
        if k < 2:
            raise ValueError("k-fold needs k >= 2")
        run_folds, method = list(range(k)), f"deterministic {k}-fold"
    folds_s = fold_assignment(raw, k)
    folds_t = fold_assignment(touches, k)
    out: dict = {"method": method, "k": k,
                 "leakage_note": "every fold re-initialises and outlier-filters from its own training items only"}
    out["without_offsets"] = _variant(raw, tables, touches, False, options, kin, folds_s, folds_t, run_folds)
    out["with_offsets"] = None
    out["with_offsets_unavailable"] = None
    try:
        # offsets need q for every sample/touch and the URDF chains
        _attach_q(raw + touches, kin)
        out["with_offsets"] = _variant(raw, tables, touches, True, options, kin, folds_s, folds_t, run_folds)
    except SolverError as e:
        out["with_offsets_unavailable"] = str(e)
    a, b = out["without_offsets"], out["with_offsets"]
    nuisance = ("the fitted offsets are nuisance parameters for the solve only and must NEVER be applied to the "
                "robot/URDF (re-referencing invalidates them; they absorb backlash, compliance and link errors)")
    out["offsets_never_applied"] = nuisance
    if b is None:
        out["keep_joint_offsets"] = None
        out["keep_decision_weak"] = None
        out["recommendation"] = ("joint offsets could not be evaluated (" + str(out["with_offsets_unavailable"]) +
                                 "); keep them off")
    else:
        have_touch = a["heldout_touch_rms_mm"] is not None and b["heldout_touch_rms_mm"] is not None
        px_ok = b["heldout_rms_px"] is not None and a["heldout_rms_px"] is not None \
            and b["heldout_rms_px"] < (1.0 - options.keep_px_margin) * a["heldout_rms_px"]
        touch_ok = have_touch and b["heldout_touch_rms_mm"] < (1.0 - options.keep_touch_margin) * a["heldout_touch_rms_mm"]
        # with touches: reprojection AND touch mismatch must both improve; without: reprojection alone (weak)
        keep = bool(px_ok and (touch_ok if have_touch else True))
        weak = not have_touch
        out["keep_joint_offsets"] = keep
        out["keep_decision_weak"] = weak
        txt = (f"held-out reprojection {a['heldout_rms_px']:.3f} px -> {b['heldout_rms_px']:.3f} px, "
               f"{a['heldout_rms_mm']:.2f} mm -> {b['heldout_rms_mm']:.2f} mm")
        if have_touch:
            txt += (f", held-out touch {a['heldout_touch_rms_mm']:.2f} mm -> {b['heldout_touch_rms_mm']:.2f} mm "
                    f"(needs >= {100 * options.keep_touch_margin:.0f} % better)")
        else:
            txt += "; no held-out touches, so the decision is weak (reprojection only)"
        out["recommendation"] = (("KEEP joint offsets in the solve: " if keep else
                                  "do NOT keep joint offsets (no clear held-out improvement in both reprojection "
                                  "and touch mismatch): ") + txt + "; " + nuisance)
    return out


def format_validation(v: dict) -> List[str]:
    L = [f"Held-out validation ({v['method']}):"]
    for key, label in (("without_offsets", "without joint offsets"), ("with_offsets", "with joint offsets   ")):
        r = v.get(key)
        if r is None:
            L.append(f"  {label}: not evaluated ({v.get('with_offsets_unavailable')})")
            continue
        s = (f"  {label}: held-out reprojection {r['heldout_rms_px']:.3f} px = {r['heldout_rms_mm']:.2f} mm "
             f"lateral at the target ({r['n_points']} points)")
        if r["heldout_touch_rms_mm"] is not None:
            s += f", held-out touch mismatch {r['heldout_touch_rms_mm']:.2f} mm ({r['n_heldout_touches']} touches)"
            if r.get("heldout_touch_rms_nominal_mm") is not None and key == "with_offsets":
                s += f" [nominal FK, as deployed: {r['heldout_touch_rms_nominal_mm']:.2f} mm]"
        L.append(s)
    L.append("  -> " + v["recommendation"])
    return L
