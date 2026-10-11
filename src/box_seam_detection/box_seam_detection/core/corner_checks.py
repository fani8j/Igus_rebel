# Provenance: moved into box_seam_detection (2026-10-12) from rgbd_captures/scripts/autolabel_v2/corner_checks.py
# at commit 6df59943f23e0d551cdc448b47ffea9e95ece20a (working tree identical to HEAD, unmodified).
# Only imports/sys.path handling changed; logic unchanged.
"""Corner quality gates: flag (never reject) suspect refined corners.

A refined corner is suspect when its edge snapped onto a parallel foreign
edge (white shelf / dark background boundary, sleeve, bed frame, torn
flap), pushing the corner outward — worst at acute vertices. Four
per-corner gates, persisted in ``checks.corner_flags``; any flag sets
``checks.needs_review`` (never rejects the quad):

1. ``shift``: coarse→refined displacement > 10 px (scaled by W/1280).
2. ``border``: refined corner within 8 px (scaled) of the image border.
3. ``ambiguous_edge``: per-edge ambiguous_frac > 0.3 — fraction of
   profile stations (refine config: half-width, spacing, min_gradient)
   whose runner-up gradient peak is ≥ 60% of the chosen peak and ≥ 4 px
   away. Profiles are re-sampled here by an independent re-implementation
   of the normal-profile peak search (``precision_box_detector`` is
   wrapped, never edited); station geometry (margin, spacing) mirrors
   ``refine_quad_edges``.
4. ``depth``: fraction of valid z-buffer-winning PCD points projecting
   within ±3 px (scaled) of the corner that lie < 8 mm from the top
   plane is < 0.5, or no valid points project there. Projection uses the
   capture camera (``precision_box_detector.camera_model``, wrapped);
   nearest-depth-wins resolves pixel collisions.

``check_corners`` is the pipeline entry point (needs gray, coarse,
refined, image size, PCD xyz, camera mapping, plane). ``recheck_corners``
recomputes the same gates from stored record fields (for ``--recheck``).
"""

from __future__ import annotations

import numpy as np

from .config import CORNER_ORDER

SHIFT_PX = 10.0
BORDER_PX = 8.0
AMBIG_FRAC = 0.3
AMBIG_RUNNER_RATIO = 0.6
AMBIG_RUNNER_MIN_SEP_PX = 4
DEPTH_RADIUS_PX = 3.0
DEPTH_INLIER_MM = 8.0
DEPTH_MIN_FRAC = 0.5


def _scale(w: int) -> float:
    try:
        return float(w) / 1280.0
    except (TypeError, ValueError):
        return 1.0


def _bilinear(gray: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    x0 = np.floor(xs).astype(np.int64)
    y0 = np.floor(ys).astype(np.int64)
    dx = xs - x0
    dy = ys - y0
    x0c = np.clip(x0, 0, w - 1)
    y0c = np.clip(y0, 0, h - 1)
    x1c = np.clip(x0 + 1, 0, w - 1)
    y1c = np.clip(y0 + 1, 0, h - 1)
    g = np.asarray(gray, dtype=np.float64)
    return ((1.0 - dx) * (1.0 - dy) * g[y0c, x0c]
            + dx * (1.0 - dy) * g[y0c, x1c]
            + (1.0 - dx) * dy * g[y1c, x0c]
            + dx * dy * g[y1c, x1c])


def edge_ambiguous_frac(gray: np.ndarray, p0: np.ndarray, p1: np.ndarray,
                        half_width_px: int = 12,
                        spacing_px: float = 6.0,
                        min_gradient: float = 4.0,
                        runner_ratio: float = AMBIG_RUNNER_RATIO,
                        runner_sep_px: int = AMBIG_RUNNER_MIN_SEP_PX
                        ) -> tuple[float | None, int]:
    """Ambiguous-profile fraction for one edge segment.

    Mirrors the ``refine_quad_edges`` station geometry (margin =
    max(half, 5% of length), profile across ±half, interior argmax peak
    with ``min_gradient`` floor). A station is ambiguous when a runner-up
    gradient peak outside ±``runner_sep_px`` of the chosen peak reaches
    ``runner_ratio`` of the chosen peak. Returns (frac|None, n_stations).
    """
    try:
        p0 = np.asarray(p0, dtype=np.float64).reshape(2)
        p1 = np.asarray(p1, dtype=np.float64).reshape(2)
        gray = np.asarray(gray, dtype=np.float64)
        if gray.ndim != 2:
            return None, 0
    except (TypeError, ValueError):
        return None, 0
    half = int(half_width_px)
    d = p1 - p0
    length = float(np.linalg.norm(d))
    if not np.isfinite(length) or length < 1e-9 or half < 2:
        return None, 0
    u = d / length
    n = np.array([-u[1], u[0]])
    margin = max(float(half), 0.05 * length)
    try:
        spacing = float(spacing_px)
        if not np.isfinite(spacing) or spacing <= 0:
            return None, 0
    except (TypeError, ValueError):
        return None, 0
    ts = np.arange(margin, length - margin + 1e-9, spacing)
    across = np.arange(-half, half + 1, dtype=np.float64)
    return _ambiguous_stations(gray, p0, u, n, ts, across, min_gradient,
                               runner_ratio, runner_sep_px)


def _ambiguous_stations(gray, p0, u, n, ts, across, min_gradient,
                        runner_ratio, runner_sep_px):
    """All stations at once; per-station arithmetic identical to
    :func:`_ambiguous_stations_loop` (elementwise ops broadcast over the
    station axis, per-row argmax/max), so the result is identical."""
    h, w = gray.shape
    ts = np.asarray(ts, dtype=np.float64).reshape(-1)
    L = across.shape[0]
    if ts.size == 0 or L < 2 * runner_sep_px + 1:
        return None, 0
    cx = p0[0] + u[0] * ts
    cy = p0[1] + u[1] * ts
    XS = cx[:, None] + (n[0] * across)[None, :]
    YS = cy[:, None] + (n[1] * across)[None, :]
    inside = ~(np.any(XS < 0, axis=1) | np.any(XS > w - 1, axis=1)
               | np.any(YS < 0, axis=1) | np.any(YS > h - 1, axis=1))
    if not inside.any():
        return None, 0
    V = _bilinear(gray, XS[inside], YS[inside])
    G = np.zeros_like(V)
    G[:, 1:-1] = 0.5 * (V[:, 2:] - V[:, :-2])
    mag = np.abs(G)
    rows = np.arange(mag.shape[0])
    peak_idx = np.argmax(mag[:, 1:-1], axis=1) + 1
    peak = mag[rows, peak_idx]
    ok = np.isfinite(peak) & ~(peak < min_gradient)
    tot = int(np.count_nonzero(ok))
    if tot == 0:
        return None, 0
    cols = np.arange(L)[None, :]
    lo = np.maximum(0, peak_idx - runner_sep_px)[:, None]
    hi = np.minimum(L, peak_idx + runner_sep_px + 1)[:, None]
    keep = ~((cols >= lo) & (cols < hi))
    has = keep.any(axis=1)
    runner = np.where(keep, mag, -np.inf).max(axis=1)
    runner = np.where(has, runner, 0.0)
    amb = int(np.count_nonzero(ok & (runner >= runner_ratio * peak)))
    return float(amb) / float(tot), int(tot)


def _ambiguous_stations_loop(gray, p0, u, n, ts, across, min_gradient,
                             runner_ratio, runner_sep_px):
    """Reference per-station loop (kept for equivalence tests)."""
    h, w = gray.shape
    amb = tot = 0
    for t in ts:
        c = p0 + u * float(t)
        xs = c[0] + n[0] * across
        ys = c[1] + n[1] * across
        if (np.any(xs < 0) or np.any(xs > w - 1)
                or np.any(ys < 0) or np.any(ys > h - 1)):
            continue
        v = _bilinear(gray, xs, ys)
        g = np.zeros_like(v)
        g[1:-1] = 0.5 * (v[2:] - v[:-2])
        mag = np.abs(g)
        if len(mag) < 2 * runner_sep_px + 1:
            continue
        peak_idx = int(np.argmax(mag[1:-1])) + 1
        peak = float(mag[peak_idx])
        if not np.isfinite(peak) or peak < min_gradient:
            continue
        mask = np.ones_like(mag, dtype=bool)
        lo = max(0, peak_idx - runner_sep_px)
        hi = min(len(mag), peak_idx + runner_sep_px + 1)
        mask[lo:hi] = False
        runner = float(mag[mask].max()) if mask.any() else 0.0
        tot += 1
        if runner >= runner_ratio * peak:
            amb += 1
    if tot == 0:
        return None, 0
    return float(amb) / float(tot), int(tot)


def corner_depth_frac(uv_win: np.ndarray, res_win: np.ndarray,
                      corner_px: np.ndarray,
                      radius_px: float,
                      inlier_mm: float = DEPTH_INLIER_MM
                      ) -> tuple[float | None, int]:
    """Depth membership of one corner pixel.

    ``uv_win``/``res_win`` are z-buffer-winning pixel coords (Nx2 int) and
    plane residuals in mm (N,). Returns (frac within ±``inlier_mm``|None
    when no winners project within ``radius_px``, n_winners).
    """
    try:
        c = np.asarray(corner_px, dtype=np.float64).reshape(2)
        if not np.all(np.isfinite(c)):
            return None, 0
        uv = np.asarray(uv_win, dtype=np.float64).reshape(-1, 2)
        rs = np.asarray(res_win, dtype=np.float64).reshape(-1)
        if len(uv) != len(rs) or len(uv) == 0:
            return None, 0
    except (TypeError, ValueError):
        return None, 0
    return _depth_frac_prepared(uv, rs, c, radius_px, inlier_mm)


def _depth_frac_prepared(uv: np.ndarray, rs: np.ndarray, c: np.ndarray,
                         radius_px: float, inlier_mm: float
                         ) -> tuple[float | None, int]:
    """Core of :func:`corner_depth_frac` on validated float64 inputs
    (``uv`` (N,2), ``rs`` (N,), same length > 0, ``c`` finite (2,))."""
    # Speed: only rows inside a (radius + 1 px) box around the corner can
    # satisfy dd <= radius, so the exact distance test below runs on that
    # subset (same rows, same order, same arithmetic -> identical result).
    r_box = float(radius_px) + 1.0
    if np.isfinite(r_box):
        box = ((uv[:, 0] >= c[0] - r_box) & (uv[:, 0] <= c[0] + r_box)
               & (uv[:, 1] >= c[1] - r_box) & (uv[:, 1] <= c[1] + r_box))
        uv, rs = uv[box], rs[box]
    dd = np.sqrt(((uv - c) ** 2).sum(axis=1))
    sel = dd <= float(radius_px)
    n = int(sel.sum())
    if n == 0:
        return None, 0
    return float((np.abs(rs[sel]) < float(inlier_mm)).mean()), n


def depth_query_region(refined, image_size, corner_cfg: dict | None = None,
                       margin_px: float = 2.0) -> tuple[int, int, int, int] | None:
    """Inclusive pixel box (u0, v0, u1, v1) holding every z-buffer winner
    that :func:`check_corners` can use for ``refined`` (all pixels within
    the depth radius of a corner, plus ``margin_px``). ``None`` when the
    corners/image size are unusable (callers then use all winners)."""
    try:
        W = int(image_size[0])
        rq = np.asarray(refined, dtype=np.float64).reshape(4, 2)
        r = float(dict(corner_cfg or {}).get("depth_radius_px",
                                              DEPTH_RADIUS_PX)) * _scale(W)
    except (TypeError, ValueError, IndexError):
        return None
    if not (np.all(np.isfinite(rq)) and np.isfinite(r)):
        return None
    pad = abs(r) + float(margin_px)
    lo = np.floor(rq.min(axis=0) - pad)
    hi = np.ceil(rq.max(axis=0) + pad)
    return int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1])


def check_corners(gray: np.ndarray, coarse: np.ndarray,
                  refined: np.ndarray, image_size: tuple[int, int],
                  uv_win=None, res_win=None,
                  refine_cfg: dict | None = None,
                  corner_cfg: dict | None = None) -> dict:
    """Run the four corner gates; return per-corner flags + values.

    ``gray``: HxW float image. ``coarse``/``refined``: (4,2) [BL,BR,FR,FL].
    ``uv_win``/``res_win``: z-buffer winners (or None → depth gate
    reports ``no_depth`` without flagging). Returns per-corner flags,
    shift/border/ambiguous/depth values, and ``any_flag``.
    """
    ccfg = dict(corner_cfg or {})
    try:
        W, H = int(image_size[0]), int(image_size[1])
    except (TypeError, ValueError, IndexError):
        W, H = 1280, 720
    sc = _scale(W)
    shift_thr = float(ccfg.get("shift_px", SHIFT_PX)) * sc
    border_thr = float(ccfg.get("border_px", BORDER_PX)) * sc
    amb_thr = float(ccfg.get("ambiguous_frac", AMBIG_FRAC))
    depth_r = float(ccfg.get("depth_radius_px", DEPTH_RADIUS_PX)) * sc
    depth_mm = float(ccfg.get("depth_inlier_mm", DEPTH_INLIER_MM))
    depth_min = float(ccfg.get("depth_min_frac", DEPTH_MIN_FRAC))
    rcfg = dict(refine_cfg or {})
    half = int(rcfg.get("profile_half_width_px", 12))
    spacing = float(rcfg.get("sample_spacing_px", 6.0))
    mingrad = float(rcfg.get("min_gradient", 4.0))
    out: dict = {"corners": {}, "any_flag": False}
    try:
        cq = np.asarray(coarse, dtype=np.float64).reshape(4, 2)
        rq = np.asarray(refined, dtype=np.float64).reshape(4, 2)
    except (TypeError, ValueError):
        return out
    edge_af: list[float | None] = []
    for i in range(4):
        try:
            f, _ = edge_ambiguous_frac(gray, rq[i], rq[(i + 1) % 4],
                                       half, spacing, mingrad)
        except Exception:  # noqa: BLE001 — one bad edge never kills gates
            f = None
        edge_af.append(f)
    # Winners converted/validated once for all four corners (speed only;
    # corner_depth_frac does the identical conversion per call).
    depth_prepared = None
    if uv_win is not None and res_win is not None:
        try:
            _uv = np.asarray(uv_win, dtype=np.float64).reshape(-1, 2)
            _rs = np.asarray(res_win, dtype=np.float64).reshape(-1)
            if len(_uv) == len(_rs) and len(_uv) > 0:
                depth_prepared = (_uv, _rs)
        except (TypeError, ValueError):
            depth_prepared = None
    for k, name in enumerate(CORNER_ORDER):
        flags: list[str] = []
        vals: dict = {}
        try:
            shift = float(np.linalg.norm(rq[k] - cq[k]))
        except (TypeError, ValueError):
            shift = float("nan")
        vals["shift_px"] = round(shift, 2) if np.isfinite(shift) else None
        if np.isfinite(shift) and shift > shift_thr:
            flags.append("large_shift")
        try:
            b = min(float(rq[k][0]), float(rq[k][1]),
                    float(W - 1 - rq[k][0]), float(H - 1 - rq[k][1]))
        except (TypeError, ValueError, IndexError):
            b = float("nan")
        vals["border_px"] = round(float(b), 2) if np.isfinite(b) else None
        if np.isfinite(b) and b < border_thr:
            flags.append("near_border")
        af = max([v for v in (edge_af[(k - 1) % 4], edge_af[k])
                  if v is not None], default=None)
        vals["ambiguous_frac"] = round(float(af), 3) if af is not None else None
        # Ambiguous-edge as specified fires on ~25% of good corners
        # (tape stripes parallel to edges), so it only flags when
        # corner_cfg flag_ambiguous is true; values always persist.
        flag_ambiguous = bool(ccfg.get("flag_ambiguous", False))
        if af is not None and af > amb_thr and flag_ambiguous:
            flags.append("ambiguous_edge")
        if uv_win is None or res_win is None:
            vals["depth_frac"] = None
            vals["depth_n"] = 0
            vals["depth_note"] = "no_depth"
        else:
            c_k = np.asarray(rq[k], dtype=np.float64).reshape(2)
            if depth_prepared is not None and np.all(np.isfinite(c_k)):
                f, n = _depth_frac_prepared(depth_prepared[0],
                                            depth_prepared[1], c_k,
                                            depth_r, depth_mm)
            else:
                f, n = corner_depth_frac(uv_win, res_win, rq[k], depth_r,
                                         depth_mm)
            vals["depth_frac"] = round(float(f), 3) if f is not None else None
            vals["depth_n"] = int(n)
            # Depth membership as specified flags most good corners
            # (z-buffer winners near a box tip are usually background),
            # so it only flags when corner_cfg flag_depth is true;
            # values are always persisted for audit.
            flag_depth = bool(ccfg.get("flag_depth", False))
            if f is None:
                vals["depth_note"] = "no_points"
                if flag_depth:
                    flags.append("depth_off_plane")
            elif f < depth_min:
                vals["depth_note"] = "off_plane"
                if flag_depth:
                    flags.append("depth_off_plane")
            else:
                vals["depth_note"] = "on_plane"
        out["corners"][name] = {"flags": flags, **vals}
        if flags:
            out["any_flag"] = True
    return out
