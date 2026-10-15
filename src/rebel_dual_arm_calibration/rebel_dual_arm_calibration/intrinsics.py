"""Camera intrinsics: file format, override of camera_info, and the ChArUco calibration tool (no ROS).

The factory ``camera_info`` is only trusted if the data do not pull it (see the ``intrinsics validated`` check of the
trust gate).  ``calibrate_intrinsics <session_dir> --out intrinsics.yaml`` calibrates the camera from the views saved
by the recorder service ``capture_intrinsics`` (a handheld or rigidly mounted board of the table-board spec, detected
with the SAME custom detector as everything else); the resulting file is read by ``solve --intrinsics-file`` (and the
recorder parameter ``intrinsics_file``) and replaces camera_info everywhere.

File format (YAML)::

    intrinsics:
      image_width: 1920
      image_height: 1080
      camera_matrix: {fx: ..., fy: ..., cx: ..., cy: ...}
      distortion_model: plumb_bob
      distortion: [k1, k2, p1, p2, k3]
      rms_px: 0.21
      ...

A sensor_msgs/CameraInfo YAML dump (``k`` / ``d`` or ``camera_matrix: {data: [...]}``) is also accepted.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

from .session import load_intrinsics_views

MIN_VIEWS = 25
RMS_LIMIT_PX = 0.5


# --------------------------------------------------------------------------
# file I/O and override
# --------------------------------------------------------------------------
def load_intrinsics_file(path: str) -> Tuple[np.ndarray, np.ndarray, dict]:
    """(K 3x3, dist 1-D, full dict) from an intrinsics YAML (see module docstring)."""
    with open(os.path.expanduser(path)) as f:
        d = yaml.safe_load(f)
    c = d.get("intrinsics", d)
    size = None
    if "camera_matrix" in c:
        cm = c["camera_matrix"]
        if isinstance(cm, dict) and "fx" in cm:
            K = np.array([[cm["fx"], 0, cm["cx"]], [0, cm["fy"], cm["cy"]], [0, 0, 1.0]], float)
        else:
            data = cm["data"] if isinstance(cm, dict) else cm
            K = np.array(data, float).reshape(3, 3)
        dist = c.get("distortion", None)
        if dist is None:
            dc = c.get("distortion_coefficients")
            dist = dc["data"] if isinstance(dc, dict) else dc
    elif "k" in c:
        K = np.array(c["k"], float).reshape(3, 3)
        dist = c.get("d", [])
    else:
        raise ValueError(f"{path}: no camera_matrix / k in the intrinsics file")
    if "image_width" in c and "image_height" in c:
        size = (int(c["image_width"]), int(c["image_height"]))
    elif "width" in c and "height" in c:
        size = (int(c["width"]), int(c["height"]))
    out = dict(c)
    out["image_size"] = size
    if not np.all(np.isfinite(K)) or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError(f"{path}: invalid camera matrix")
    return K, np.array(dist if dist is not None else [], float).reshape(-1), out


def write_intrinsics_file(path: str, K: np.ndarray, dist: np.ndarray, image_size: Tuple[int, int],
                          extra: Optional[dict] = None) -> None:
    d = {"intrinsics": {
        "image_width": int(image_size[0]), "image_height": int(image_size[1]),
        "camera_matrix": {"fx": float(K[0, 0]), "fy": float(K[1, 1]), "cx": float(K[0, 2]), "cy": float(K[1, 2])},
        "distortion_model": "plumb_bob", "distortion": [float(v) for v in np.asarray(dist).reshape(-1)],
        **(extra or {})}}
    with open(path, "w") as f:
        yaml.safe_dump(d, f, sort_keys=False)


def apply_intrinsics(items: Sequence, K: np.ndarray, dist: np.ndarray, image_size: Optional[Tuple[int, int]],
                     label: str = "intrinsics file") -> int:
    """Replace K and distortion of samples / table observations in place (camera_info is overridden everywhere).
    Raises ValueError if the image size of the file differs from an item's."""
    n = 0
    for it in items:
        if image_size is not None and tuple(it.image_size) != tuple(image_size):
            raise ValueError(f"{label}: image size {tuple(image_size)} differs from the recorded images "
                             f"{tuple(it.image_size)}")
        it.K = np.array(K, float).copy()
        it.dist = np.array(dist, float).copy()
        n += 1
    return n


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------
def _coverage(views: Sequence[dict], size: Tuple[int, int], grid: int = 4, band: float = 0.15) -> dict:
    W, H = size
    counts = np.zeros((grid, grid), int)
    quad = np.zeros(4, int)
    border = {"left": 0, "right": 0, "top": 0, "bottom": 0}
    n = 0
    for v in views:
        for u, w in v["pix"]:
            gx, gy = min(int(u / W * grid), grid - 1), min(int(w / H * grid), grid - 1)
            counts[max(gy, 0), max(gx, 0)] += 1
            quad[(0 if w < H / 2 else 2) + (0 if u < W / 2 else 1)] += 1
            border["left"] += u < band * W
            border["right"] += u > (1 - band) * W
            border["top"] += w < band * H
            border["bottom"] += w > (1 - band) * H
            n += 1
    warn = []
    names = ["top-left", "top-right", "bottom-left", "bottom-right"]
    for q, c in zip(names, quad):
        if c < 0.10 * max(n, 1):
            warn.append(f"quadrant {q} has only {100 * c / max(n, 1):.0f} % of the corners (< 10 %)")
    for b, c in border.items():
        if c < 0.03 * max(n, 1):
            warn.append(f"the {b} {int(band * 100)} % band of the image has only {c} corners: hold the board at the "
                        f"{b} edge too (distortion is largest there)")
    return {"grid": grid, "corners_per_cell": counts.tolist(), "quadrants": dict(zip(names, map(int, quad))),
            "border_bands": {k: int(v) for k, v in border.items()}, "n_corners": n, "warnings": warn}


def calibrate_views(views: Sequence[dict], min_views: int = MIN_VIEWS, rms_limit: float = RMS_LIMIT_PX,
                    fix_k3: bool = False, chess_only: bool = False, drop_outlier_views: bool = True) -> dict:
    """OpenCV plumb_bob calibration from recorder views (ids / obj / pix / K / image_size per view).

    Returns a report dict with ``K``, ``dist``, ``rms_px``, per-view errors, standard deviations, coverage, depth range,
    the difference to camera_info and ``ok`` / ``problems``."""
    views = [v for v in views if len(v["ids"]) >= 6]
    if not views:
        raise ValueError("no usable intrinsics views")
    size = tuple(int(v) for v in views[0]["image_size"])
    if any(tuple(v["image_size"]) != size for v in views):
        raise ValueError("views have different image sizes")
    K_info = np.mean([np.array(v["K"], float).reshape(3, 3) for v in views], axis=0)
    D_info = np.array(views[0]["dist"], float).reshape(-1)
    from .targets import CHESS_ID_BASE

    def points(v):
        ids, obj, pix = np.array(v["ids"]), np.array(v["obj"], float), np.array(v["pix"], float)
        if chess_only:
            m = ids >= CHESS_ID_BASE
            ids, obj, pix = ids[m], obj[m], pix[m]
        return obj.astype(np.float32).reshape(-1, 1, 3), pix.astype(np.float32).reshape(-1, 1, 2)

    flags = cv2.CALIB_USE_INTRINSIC_GUESS | (cv2.CALIB_FIX_K3 if fix_k3 else 0)
    keep = list(range(len(views)))
    dropped: List[int] = []
    def calibrate(keep_idx):
        pts = [points(views[i]) for i in keep_idx]
        out = cv2.calibrateCameraExtended(
            [p[0] for p in pts], [p[1] for p in pts], size, K_info.copy(), np.zeros(5), flags=flags)
        return out[0], out[1], out[2], out[3], out[4], out[5], np.asarray(out[7]).reshape(-1)

    rms, K, D, rv, tv, std_i, per = calibrate(keep)
    for _ in range(2):
        if not drop_outlier_views or len(keep) <= min_views:
            break
        # per-view RMS from the reprojection (the 'per' output is the RMS of each view)
        bad = [k for k, e in zip(keep, per) if e > max(1.0, 3.0 * float(np.median(per)))]
        if not bad:
            break
        dropped += bad
        keep = [k for k in keep if k not in bad]
        # Recalibrate on the reduced set so K, D and the per-view poses always belong to `keep`.
        rms, K, D, rv, tv, std_i, per = calibrate(keep)
    P = [points(views[i]) for i in keep]
    per_view = []
    for i, (o, p), r, t in zip(keep, P, rv, tv):
        pr, _ = cv2.projectPoints(o, r, t, K, D)
        e = np.linalg.norm(pr.reshape(-1, 2) - p.reshape(-1, 2), axis=1)
        per_view.append({"image": views[i].get("image", ""), "rms_px": float(np.sqrt(np.mean(e ** 2))),
                         "n_points": int(len(e)), "board_distance_m": float(np.linalg.norm(t))})
    z = np.array([float(t.reshape(3)[2]) for t in tv])
    std = np.asarray(std_i).reshape(-1)
    names = ("fx", "fy", "cx", "cy")
    dfx, dfy = (K[0, 0] - K_info[0, 0]) / K_info[0, 0], (K[1, 1] - K_info[1, 1]) / K_info[1, 1]
    dcx, dcy = K[0, 2] - K_info[0, 2], K[1, 2] - K_info[1, 2]
    cov = _coverage([views[i] for i in keep], size)
    problems, warnings = [], list(cov["warnings"])
    if len(keep) < min_views:
        problems.append(f"only {len(keep)} usable views (< {min_views})")
    if rms >= rms_limit:
        problems.append(f"calibration RMS {rms:.3f} px >= {rms_limit:.2f} px: blurred/loose board, wrong measured "
                        "square size, or too few well-spread views")
    if z.size and (z.max() - z.min()) < 0.25 * float(np.median(z)):
        warnings.append(f"board depth range only {z.min():.2f}-{z.max():.2f} m; vary the distance (focal length and "
                        "distortion are only separated by depth variation and tilt)")
    if np.any(~np.isfinite(std)) or (std[:4] > 5.0).any():
        warnings.append("large parameter standard deviation: more varied views needed")
    return {"ok": not problems, "problems": problems, "warnings": warnings, "rms_px": float(rms), "n_views": len(keep),
            "n_views_dropped": len(dropped), "dropped_views": [views[i].get("image", str(i)) for i in dropped],
            "image_size": list(size), "K": K.tolist(), "dist": [float(v) for v in D.reshape(-1)],
            "std_px": dict(zip(names, (float(v) for v in std[:4]))), "per_view": per_view,
            "board_distance_m": {"min": float(z.min()), "max": float(z.max()), "median": float(np.median(z))},
            "coverage": cov, "fix_k3": fix_k3, "chess_only": chess_only,
            "camera_info": {"fx": float(K_info[0, 0]), "fy": float(K_info[1, 1]), "cx": float(K_info[0, 2]),
                            "cy": float(K_info[1, 2]), "dist": [float(v) for v in D_info]},
            "difference_to_camera_info": {"df_x_pct": float(100 * dfx), "df_y_pct": float(100 * dfy),
                                          "dc_px": [float(dcx), float(dcy)]}}


def format_calibration(r: dict) -> List[str]:
    K = np.array(r["K"])
    d = r["difference_to_camera_info"]
    L = [f"Intrinsics calibration: {r['n_views']} views ({r['n_views_dropped']} dropped), RMS {r['rms_px']:.3f} px "
         f"(must be < {RMS_LIMIT_PX:.1f}), image {r['image_size'][0]}x{r['image_size'][1]}",
         f"  fx {K[0, 0]:.2f} +-{r['std_px']['fx']:.2f}  fy {K[1, 1]:.2f} +-{r['std_px']['fy']:.2f}  "
         f"cx {K[0, 2]:.2f} +-{r['std_px']['cx']:.2f}  cy {K[1, 2]:.2f} +-{r['std_px']['cy']:.2f}",
         "  distortion (k1 k2 p1 p2 k3): " + " ".join(f"{v:+.5f}" for v in r["dist"]),
         f"  difference to camera_info: f {d['df_x_pct']:+.2f} % / {d['df_y_pct']:+.2f} %, c "
         f"{d['dc_px'][0]:+.2f} / {d['dc_px'][1]:+.2f} px",
         f"  board depth {r['board_distance_m']['min']:.2f}-{r['board_distance_m']['max']:.2f} m (median "
         f"{r['board_distance_m']['median']:.2f})",
         "  coverage (corners per cell, rows top->bottom): "
         + " | ".join(" ".join(f"{c:4d}" for c in row) for row in r["coverage"]["corners_per_cell"]),
         "  quadrants: " + ", ".join(f"{k} {v}" for k, v in r["coverage"]["quadrants"].items())
         + "; border bands: " + ", ".join(f"{k} {v}" for k, v in r["coverage"]["border_bands"].items())]
    L += [f"  WARNING: {w}" for w in r["warnings"]]
    L += [f"  PROBLEM: {p}" for p in r["problems"]]
    return L


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Calibrate the camera intrinsics (plumb_bob) from the views saved by the recorder service "
                    "~/capture_intrinsics; writes a file for 'solve --intrinsics-file'.")
    ap.add_argument("session_dirs", nargs="+", metavar="dir", help="recorder session directory(ies) with intrinsics.jsonl")
    ap.add_argument("--out", required=True, help="intrinsics YAML to write")
    ap.add_argument("--report", default=None, help="full report JSON (default next to --out)")
    ap.add_argument("--min-views", type=int, default=MIN_VIEWS)
    ap.add_argument("--rms-limit-px", type=float, default=RMS_LIMIT_PX)
    ap.add_argument("--fix-k3", action="store_true", help="keep the 6th-order radial term at 0")
    ap.add_argument("--chess-only", action="store_true", help="use only the chessboard saddles, not the marker corners")
    ap.add_argument("--force", action="store_true", help="write the file although the requirements are not met")
    args = ap.parse_args(argv)
    views = []
    for d in args.session_dirs:
        v = load_intrinsics_views(d)
        for x in v:
            x["image"] = os.path.join(os.path.basename(os.path.normpath(d)), x.get("image", ""))
        views += v
    if not views:
        print("ERROR: no intrinsics.jsonl views in " + ", ".join(args.session_dirs), file=sys.stderr)
        return 2
    try:
        r = calibrate_views(views, args.min_views, args.rms_limit_px, args.fix_k3, args.chess_only)
    except (ValueError, cv2.error) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    print("\n".join(format_calibration(r)))
    rep = args.report or os.path.splitext(args.out)[0] + "_report.json"
    with open(rep, "w") as f:
        json.dump(r, f, indent=2)
    if not r["ok"] and not args.force:
        print(f"NOT written (requirements not met; --force overrides). Report: {rep}", file=sys.stderr)
        return 2
    write_intrinsics_file(args.out, np.array(r["K"]), np.array(r["dist"]), tuple(r["image_size"]), {
        "rms_px": r["rms_px"], "n_views": r["n_views"], "std_px": r["std_px"],
        "difference_to_camera_info": r["difference_to_camera_info"], "date": datetime.date.today().isoformat(),
        "source": ", ".join(os.path.abspath(d) for d in args.session_dirs),
        "requirements_met": bool(r["ok"])})
    print(f"Wrote {args.out}\nWrote {rep}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
