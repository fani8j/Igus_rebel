"""D1: per-edge DEPTH support of the top-face quad (2026-10-12).

The RGB quad is precise but nothing independent tells whether each of its four
edges really is a top-face boundary (a quad that ends on the flap gap, a tape
band, a truncated top all end on crisp image lines). The depth cloud can: walk
OUTWARD from every edge, station by station, and look at the signed height
above the fitted top plane (``h``; the support / background is negative):

* ``cliff`` = the outward distance (mm, + away from the quad) where the
  height first drops below ``-thr`` (persistently). A real top edge has its
  cliff at ~0: the lid ends, then the side wall / background falls away.
  A quad that stops short of the real edge has its cliff far outside
  (``continues``): the coplanar top continues beyond the edge ->
  ``truncated`` evidence (``extent_mm`` = how far, capped at ``probe_mm``).
* For an edge FACING the camera (front) the cliff offset is the
  RGB-edge-to-depth-edge bias (measured -3.5 mm, range -6.5..-1.5, on the 17
  good dual-arm captures: the rounded lip falls away before the RGB edge).
* For an OCCLUDING edge (back/left/right: the side wall is hidden) the depth
  jump to the background must be there right behind the edge:
  ``outside_h_mm`` (median height 4-12 px behind the cliff) is strongly
  negative for a real silhouette and ~0 when the top continues.

Everything is recorded per edge; ``unsupported`` is a verdict on the
``continues`` evidence only (a conservative rule, see ``decide``). Pure
numpy/OpenCV; ``compute`` never raises (``{"error": ...}`` instead).
"""
from __future__ import annotations

import math
import warnings

import numpy as np

#: ring edge i joins ring corners i and (i + 1) % 4; names in ring order
EDGE_NAMES = ("back", "right", "front", "left")

DEFAULTS: dict = {
    "n_stations": 40,         # stations along each edge (10 %..90 % of its length)
    "span": (0.10, 0.90),
    "inner_px": 10,           # profile starts this far INSIDE the RGB edge
    "probe_mm": 80.0,         # profile ends this far outside
    "max_probe_px": 130,
    "thr_min_mm": 10.0,       # cliff = height below -thr (advisor: -10 mm)
    "thr_noise_k": 3.0,       # ... and below -k * plane rms
    "min_valid_frac": 0.30,   # a station needs this share of valid samples
    "continue_mm": 12.0,      # a station "continues" when its cliff is farther than this
    "outside_px": (4, 12),    # outside_h window behind the cliff
}


def _cfg(over: dict | None) -> dict:
    out = dict(DEFAULTS)
    if over:
        out.update(over)
    return out


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _mm_per_px(a_px: np.ndarray, b_px: np.ndarray, backproject) -> float | None:
    """In-plane distance (mm) between the plane points under two pixels."""
    try:
        p = backproject(np.array([a_px, b_px], dtype=np.float64))
    except Exception:  # noqa: BLE001
        return None
    if p is None or not np.all(np.isfinite(p)):
        return None
    dist = float(np.linalg.norm(p[1] - p[0])) * 1000.0
    d_px = float(np.linalg.norm(np.asarray(b_px) - np.asarray(a_px)))
    return dist / d_px if d_px > 0 and dist > 0 else None


def _station_cliff(h: np.ndarray, ts: np.ndarray, thr: float, inner_px: int,
                   min_valid_frac: float):
    """Cliff of one station profile.

    ``h``: heights (mm, NaN = no depth) at outward offsets ``ts`` (px,
    ascending, starting ``-inner_px``). Returns ``(status, t_px)`` with status
    ``ok`` (t_px = interpolated crossing of -thr), ``continues`` (no drop
    before the end of the probe), ``early`` (the first valid sample is
    already below -thr: t_px = its offset, an upper bound), ``no_data``.
    """
    valid = np.isfinite(h)
    if valid.sum() < max(4, int(min_valid_frac * len(h))):
        return "no_data", None
    # the inside must look like top: median of the inner samples (t < -2)
    inner = h[(ts <= -2) & valid]
    if inner.size >= 2 and abs(float(np.median(inner))) > 2.0 * thr:
        return "no_data", None
    below = valid & (h <= -thr)
    n = len(h)
    prev_valid = None
    for k in range(n):
        if not valid[k]:
            continue
        if below[k]:
            # persistence: of this and the next two VALID samples at least two are below
            nxt = [j for j in range(k + 1, min(n, k + 6)) if valid[j]][:2]
            votes = 1 + sum(1 for j in nxt if below[j])
            if len(nxt) == 0 or votes >= 2:
                if prev_valid is None:
                    return "early", float(ts[k])
                j = prev_valid
                f = (h[j] + thr) / (h[j] - h[k]) if h[j] != h[k] else 0.0
                return "ok", float(ts[j] + f * (ts[k] - ts[j]))
        prev_valid = k
    return "continues", None


def compute(quad_px: np.ndarray, plane_nd: tuple, grid: np.ndarray, roi: list,
            backproject, rms_mm: float | None, cam_foot: np.ndarray | None = None,
            cfg: dict | None = None) -> dict:
    """Per-edge depth support of a ring-ordered quad.

    ``quad_px`` (4, 2); ``plane_nd`` = (n, d) unit normal, camera on the +
    side; ``grid`` = :func:`depth_top.height_grid` over ``roi``;
    ``backproject(uv) -> (N, 3)`` plane points (m); ``cam_foot``: the plane
    point nearest the camera (``-d n``), used to tell camera-facing edges.
    """
    c = _cfg(cfg)
    quad = np.asarray(quad_px, dtype=np.float64).reshape(4, 2)
    n, d = plane_nd
    thr = max(float(c["thr_min_mm"]),
              float(c["thr_noise_k"]) * float(rms_mm or 0.0))
    out: dict = {"threshold_mm": round(thr, 2), "edges": {}}
    centre = quad.mean(axis=0)
    x0, y0 = int(roi[0]), int(roi[1])
    gh, gw = grid.shape
    corners3 = None
    try:
        corners3 = backproject(quad)
    except Exception:  # noqa: BLE001
        corners3 = None
    for i, name in enumerate(EDGE_NAMES):
        a, b = quad[i], quad[(i + 1) % 4]
        along = b - a
        length_px = float(np.linalg.norm(along))
        e: dict = {"length_px": round(length_px, 1)}
        out["edges"][name] = e
        if length_px < 5:
            e["status"] = "degenerate"
            continue
        tdir = along / length_px
        o = np.array([tdir[1], -tdir[0]])
        mid = 0.5 * (a + b)
        if float(o @ (mid - centre)) < 0:
            o = -o
        mm_px = _mm_per_px(mid, mid + o, backproject)
        if mm_px is None:
            e["status"] = "no_scale"
            continue
        e["mm_per_px"] = round(mm_px, 4)
        # facing the camera: the outward in-plane direction points to the camera foot
        facing = None
        parallax = None
        if corners3 is not None and cam_foot is not None and np.all(np.isfinite(corners3)):
            m3 = 0.5 * (corners3[i] + corners3[(i + 1) % 4])
            e3 = corners3[(i + 1) % 4] - corners3[i]
            out3 = np.cross(n, e3)
            out3 = _unit(out3)
            cen3 = corners3.mean(axis=0)
            if float(out3 @ (m3 - cen3)) < 0:
                out3 = -out3
            facing = bool(float(out3 @ (np.asarray(cam_foot) - m3)) > 0)
            # a point `thr` mm BELOW the plane that is seen at pixel offset t sits
            # (thr/d) * (m3 . out3) mm further out in the image than its orthogonal
            # projection on the plane (parallax of the side wall): in-plane coordinate
            parallax = float(thr) * float(m3 @ out3) / float(d)
        e["facing_camera"] = facing
        tmax = int(min(int(c["max_probe_px"]), math.ceil(float(c["probe_mm"]) / mm_px)))
        ts = np.arange(-int(c["inner_px"]), tmax + 1, 1.0)
        fr = np.linspace(c["span"][0], c["span"][1], int(c["n_stations"]))
        stat_cliff_mm, stat_out_h = [], []
        n_ok = n_cont = n_early = n_nodata = 0
        for f in fr:
            p0 = a + f * along
            hs = np.full((3, len(ts)), np.nan)
            for li, lat in enumerate((-1.0, 0.0, 1.0)):
                pts = p0[None, :] + lat * tdir[None, :] + ts[:, None] * o[None, :]
                pu = np.rint(pts[:, 0]).astype(int) - x0
                pv = np.rint(pts[:, 1]).astype(int) - y0
                ok = (pu >= 0) & (pu < gw) & (pv >= 0) & (pv < gh)
                hs[li, ok] = grid[pv[ok], pu[ok]]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                h = np.nanmedian(hs, axis=0)
            status, t_cl = _station_cliff(h, ts, thr, int(c["inner_px"]),
                                          float(c["min_valid_frac"]))
            if status == "no_data":
                n_nodata += 1
            elif status == "continues":
                n_cont += 1
                stat_cliff_mm.append(float(c["probe_mm"]))
            else:
                if status == "early":
                    n_early += 1
                else:
                    n_ok += 1
                stat_cliff_mm.append(t_cl * mm_px)
                lo, hi = c["outside_px"]
                sel = (ts >= t_cl + lo) & (ts <= t_cl + hi) & np.isfinite(h)
                if sel.sum() >= 3:
                    stat_out_h.append(float(np.median(h[sel])))
        n_valid = n_ok + n_cont + n_early
        e["n_stations"] = int(len(fr))
        e["n_valid"] = int(n_valid)
        e["n_continues"] = int(n_cont)
        e["n_early"] = int(n_early)
        if n_valid < 8:
            e["status"] = "insufficient_depth"
            continue
        cl = np.asarray(stat_cliff_mm)
        cont_mm = float(c["continue_mm"])
        e["status"] = "ok"
        e["cliff_mm"] = round(float(np.median(cl)), 2)
        e["cliff_p25_mm"] = round(float(np.percentile(cl, 25)), 2)
        e["cliff_p75_mm"] = round(float(np.percentile(cl, 75)), 2)
        if facing and parallax is not None:
            e["cliff_plane_mm"] = round(float(np.median(cl)) + parallax, 2)
        e["extent_mm"] = round(float(max(0.0, np.median(cl))), 2)
        e["continue_frac"] = round(float(np.mean(cl >= cont_mm)), 3)
        e["valid_frac"] = round(n_valid / len(fr), 3)
        e["outside_h_mm"] = (round(float(np.median(stat_out_h)), 1)
                             if stat_out_h else None)
    return out


#: decision thresholds of the (cut-blocking) rule; see `decide`
UNSUPPORTED_CONTINUE_FRAC = 0.5
UNSUPPORTED_EXTENT_MM = 20.0


def decide(result: dict, continue_frac: float = UNSUPPORTED_CONTINUE_FRAC,
           extent_mm: float = UNSUPPORTED_EXTENT_MM) -> list:
    """Edges with truncation evidence: the coplanar top continues beyond the
    edge for at least ``continue_frac`` of its stations AND the median extent
    reaches ``extent_mm``. Edges without enough depth are NOT flagged (no
    evidence is not evidence of a problem; the other gates cover them)."""
    bad = []
    for name in EDGE_NAMES:
        e = (result.get("edges") or {}).get(name) or {}
        if e.get("status") != "ok":
            continue
        if (e.get("continue_frac", 0.0) >= continue_frac
                and e.get("extent_mm", 0.0) >= extent_mm):
            bad.append(name)
    return bad
