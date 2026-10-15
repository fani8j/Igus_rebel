"""Accuracy / detectability study for the calibration target size (synthetic).

    OMP_NUM_THREADS=1 nice -n 19 python3 -m rebel_dual_arm_calibration.study --out <dir> [--trials 6] [--workers 3]

Part 1 simulates the full solver on synthetic scenes for several targets and camera distances.
Part 2 renders each target at a range of marker sizes (with blur and sensor noise) and records
whether the ArUco detector + corner refinement still finds all markers / corners.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time

import cv2
import numpy as np

from . import solver
from .synthetic import make_scene, marker_side_px, simulate_samples
from .targets import CHESS_ID_BASE, TargetDetector, TargetSpec, render_board
from .transforms import rotation_angle_deg

CONFIGS = {
    "a_3x3_35mm": ("charuco", 3, 0.035, 0.026),
    "b_4x4_25mm": ("charuco", 4, 0.025, 0.018),
    "c_3x3_30mm": ("charuco", 3, 0.030, 0.022),
    "d_aruco_59mm": ("aruco", 0, 0, 0.059),
    "e_aruco_80mm": ("aruco", 0, 0, 0.080),
}
DISTANCES = (0.8, 1.0, 1.2)
MIN_MARKER_PX = 18.0   # shortest projected marker side treated as detectable (see detection study)
VARIANTS = ((55.0, 1.0), (40.0, 1.0), (55.0, 2.0))   # (max view angle off the camera axis, noise multiplier)


def make_specs(name):
    typ, n, sq, ml = CONFIGS[name]
    if typ == "aruco":
        return {"left": TargetSpec("left", "aruco", "DICT_4X4_50", ml, [13]),
                "right": TargetSpec("right", "aruco", "DICT_4X4_50", ml, [14])}
    cnt = (n * n) // 2
    return {"left": TargetSpec("left", "charuco", "DICT_4X4_50", ml, list(range(cnt)), sq, n, n),
            "right": TargetSpec("right", "charuco", "DICT_4X4_50", ml, list(range(20, 20 + cnt)), sq, n, n)}


def _trial(args):
    name, dist, seed, noise_scale, n_per_arm, view_deg = args
    specs = make_specs(name)
    scene = make_scene(distance=dist, seed=seed)
    stats = {}
    try:
        samples = simulate_samples(scene, specs, n_per_arm=n_per_arm, noise_px=0.3 * noise_scale, seed=1000 + seed,
                                   n_sample_outliers=0, n_corner_outliers=0, marker_noise_factor=5.0 / 3.0,
                                   max_view_deg=view_deg, min_marker_px=MIN_MARKER_PX, stats=stats)
    except RuntimeError:
        return dict(name=name, dist=dist, seed=seed, view_deg=view_deg, noise_scale=noise_scale,
                    failed="placement", usable_fraction=stats["usable"] / max(stats["valid"], 1))
    side = marker_side_px(specs, samples)
    usable = stats["usable"] / max(stats["valid"], 1)
    try:
        res = solver.solve(samples)
    except solver.SolverError as e:
        return dict(name=name, dist=dist, seed=seed, failed=str(e), min_side=side)
    return dict(
        name=name, dist=dist, seed=seed, noise_scale=noise_scale, view_deg=view_deg, min_side=side,
        usable_fraction=usable,
        right_mm=float(np.linalg.norm(res.T_world_arm_right_base[:3, 3] - scene.T_W_R[:3, 3]) * 1e3),
        right_deg=rotation_angle_deg(res.T_world_arm_right_base[:3, :3], scene.T_W_R[:3, :3]),
        cam_mm=float(np.linalg.norm(res.T_world_camera_link[:3, 3] - scene.T_W_cam[:3, 3]) * 1e3),
        cam_deg=rotation_angle_deg(res.T_world_camera_link[:3, :3], scene.T_W_cam[:3, :3]),
        std_right_mm=res.report["joint"]["calibration_position_std_mm"]["arm_right_base"],
        std_cam_mm=res.report["joint"]["calibration_position_std_mm"]["camera_link"],
        warnings=len(res.report["warnings"]))


def accuracy_study(trials, workers, n_per_arm=20):
    jobs = [(n, d, s, ns, n_per_arm, vd) for n in CONFIGS for d in DISTANCES for s in range(trials)
            for vd, ns in VARIANTS]
    with mp.Pool(workers) as pool:
        rows = pool.map(_trial, jobs, chunksize=1)
    return rows


def _tilted_scene(spec, side_px, tilt_deg, blur_sigma, noise_levels, seed):
    """Two copies of the board (one per arm) rendered so the marker side is side_px (fronto-parallel)."""
    rng = np.random.default_rng(seed)
    ppm = 4000.0
    hi, (ox, oy) = render_board(spec, ppm, margin=0.012)
    f = side_px / (spec.marker_length * ppm)
    img = cv2.resize(hi, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    a = np.radians(15.0)
    c = np.cos(np.radians(tilt_deg))
    # foreshorten along x about the image centre, then rotate
    cx, cy = img.shape[1] / 2, img.shape[0] / 2
    H = np.array([[c, 0, cx * (1 - c)], [0, 1, 0], [0, 0, 1.0]])
    R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])
    T = np.array([[1, 0, 700 - cx], [0, 1, 400 - cy], [0, 0, 1.0]])
    M = T @ R @ H
    scene = np.full((1080, 1920), 190, np.uint8)
    warped = cv2.warpPerspective(img, M, (1920, 1080), flags=cv2.INTER_LINEAR, borderValue=190)
    mask = cv2.warpPerspective(np.full_like(img, 255), M, (1920, 1080), flags=cv2.INTER_NEAREST)
    scene = np.where(mask > 0, warped, scene)
    if blur_sigma > 0:
        scene = cv2.GaussianBlur(scene, (0, 0), blur_sigma)
    scene = np.clip(scene + rng.normal(0, noise_levels, scene.shape), 0, 255).astype(np.uint8)
    return scene


def detection_study(sides=(8, 10, 12, 14, 16, 18, 20, 25, 30, 40), tilts=(0.0, 55.0), blurs=(0.0, 1.0), reps=3):
    out = []
    for name in CONFIGS:
        specs = make_specs(name)
        spec = specs["left"]
        det = TargetDetector({"left": spec})
        n_expected_markers = spec.n_markers()
        n_expected_chess = len(spec.chess_corners_obj())
        for blur in blurs:
            for tilt in tilts:
                for side in sides:
                    mk, ch = [], []
                    for r in range(reps):
                        img = _tilted_scene(spec, side, tilt, blur, 2.0, r)
                        d = det.detect(img).get("left")
                        mk.append(0 if d is None else d.n_markers)
                        ch.append(0 if d is None else d.n_chess)
                    out.append(dict(name=name, blur_px=blur, tilt_deg=tilt, marker_side_px=side,
                                    markers_found=float(np.mean(mk)), markers_expected=n_expected_markers,
                                    chess_found=float(np.mean(ch)), chess_expected=n_expected_chess))
    return out


def summarize(rows):
    table, failed = {}, {}
    for r in rows:
        key = (r["name"], r["dist"], r["view_deg"], r["noise_scale"])
        if "failed" in r:
            failed[key] = failed.get(key, 0) + 1
            continue
        table.setdefault(key, []).append(r)
    summary = []
    for key, n_fail in failed.items():
        if key not in table:
            summary.append(dict(name=key[0], dist=key[1], view_deg=key[2], noise_scale=key[3], trials=0,
                                failed=n_fail))
    for (name, dist, vd, ns), rs in sorted(table.items()):
        rms = lambda k: float(np.sqrt(np.mean([r[k] ** 2 for r in rs])))  # noqa: E731
        summary.append(dict(name=name, dist=dist, view_deg=vd, noise_scale=ns, trials=len(rs),
                            failed=failed.get((name, dist, vd, ns), 0),
                            usable_fraction=float(np.mean([r['usable_fraction'] for r in rs])),
                            right_mm=rms("right_mm"), right_deg=rms("right_deg"),
                            cam_mm=rms("cam_mm"), cam_deg=rms("cam_deg"),
                            right_mm_max=max(r["right_mm"] for r in rs), cam_mm_max=max(r["cam_mm"] for r in rs),
                            std_right_mm=float(np.mean([r["std_right_mm"] for r in rs])),
                            min_side_px=float(min(r["min_side"] for r in rs)),
                            median_min_side_px=float(np.median([r["min_side"] for r in rs]))))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--trials", type=int, default=6)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    rows = accuracy_study(args.trials, args.workers)
    summary = summarize(rows)
    det = detection_study()
    with open(os.path.join(args.out, "study_results.json"), "w") as f:
        json.dump({"cv2": cv2.__version__, "accuracy_trials": rows, "accuracy_summary": summary,
                   "detection": det, "seconds": time.time() - t0}, f, indent=1)
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
