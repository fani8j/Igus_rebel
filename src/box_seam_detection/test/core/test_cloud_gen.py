# Provenance: ported from rgbd_captures/tests/test_cloud_gen.py (HEAD 6df5994); imports made package-relative.
"""tests/synth/cloud_gen.py: analytic scenes give the expected plane fits."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest


from synth import cloud_gen as cg  # noqa: E402
from box_seam_detection.core.autolabel import pcd as _pcd  # noqa: E402
from box_seam_detection.core.autolabel import depth_top as _depth  # noqa: E402


def fit_plane(p):
    c = p.mean(axis=0)
    _, _, vt = np.linalg.svd(p - c, full_matrices=False)
    n = vt[2]
    if n[2] > 0:
        n = -n
    return n, -float(n @ c), float(np.sqrt(np.mean(((p - c) @ n) ** 2)))


def height_from_planes(g):
    xyz, lab = g["xyz"], g["label"]
    nt, dt, _ = fit_plane(xyz[lab == cg.LABEL_TABLE])
    ntop, dtop, rms = fit_plane(xyz[lab == cg.LABEL_TOP])
    return dt - dtop, nt, ntop, rms


def test_camera_is_the_real_manifest_camera():
    sc = cg.Scene()
    assert (sc.cam["width"], sc.cam["height"]) == (1920, 1080)
    assert sc.cam["fx"] == pytest.approx(1037.817, abs=1e-3)
    assert "dist" not in sc.cam and "d" not in sc.cam


@pytest.mark.parametrize("h,w,l,dist,cpx", [
    (0.0863, 0.19, 0.10, cg.REAL_TABLE_DIST, (960, 720)),
    (0.1343, 0.19, 0.13, cg.REAL_TABLE_DIST, (960, 720)),
    (0.30, 0.30, 0.20, 1.0, (960, 640)),
    (0.50, 0.45, 0.32, 1.6, (960, 620)),
])
@pytest.mark.parametrize("yaw", [0.0, 37.0])
def test_noise_free_plane_fits(h, w, l, dist, cpx, yaw):
    sc = cg.Scene(h=h, width=w, length=l, yaw_deg=yaw, cam_dist=dist, center_px=cpx)
    g = sc.cast(step=4)
    assert (g["label"] == cg.LABEL_TOP).sum() > 500
    hh, nt, ntop, rms = height_from_planes(g)
    assert hh == pytest.approx(h, abs=1e-3)       # spec: within 1 mm (really ~1e-6)
    assert rms < 1e-6
    assert abs(nt @ sc.n) > 1 - 1e-9 and abs(ntop @ sc.n) > 1 - 1e-9
    # all four top corners in view
    px = sc.truth()["top_corners_px"]
    assert (px[:, 0] > 0).all() and (px[:, 0] < 1920).all()
    assert (px[:, 1] > 0).all() and (px[:, 1] < 1080).all()


def test_walls_visible_and_geometry_consistent():
    sc = cg.Scene(h=0.2, width=0.2, length=0.2, yaw_deg=20, cam_dist=0.9, center_px=(960, 640))
    g = sc.cast(step=2)
    labs = set(np.unique(g["label"]).tolist())
    assert {cg.LABEL_TABLE, cg.LABEL_TOP} <= labs
    assert len(labs & set(cg.LABEL_WALLS)) >= 1
    # every wall point lies on a box-wall plane: local coordinate = +-half size
    t = sc.truth()
    xyz = g["xyz"]
    Pt = (xyz.reshape(-1, 3) - sc.O) @ sc.Rm                       # to table frame
    lab = g["label"].reshape(-1)
    loc = np.stack([(Pt[:, :2] - sc.c12) @ sc.a1, (Pt[:, :2] - sc.c12) @ sc.a2], 1)
    for l, (ax, s) in zip(cg.LABEL_WALLS, ((0, 1), (0, -1), (1, 1), (1, -1))):
        sel = lab == l
        if sel.any():
            assert np.allclose(loc[sel, ax], s * 0.1, atol=1e-6)
    # top corners from truth match the projected top-face pixel extent
    top_px = np.stack([g["u"].reshape(-1)[lab == cg.LABEL_TOP],
                       g["v"].reshape(-1)[lab == cg.LABEL_TOP]], 1)
    px = t["top_corners_px"]
    assert top_px[:, 0].min() >= px[:, 0].min() - 3 and top_px[:, 0].max() <= px[:, 0].max() + 3


def test_visibility_nearest_wins_and_unorganised_matches_organised():
    sc = cg.Scene(yaw_deg=15)
    g = sc.cast(step=4)
    org = sc.cloud(organised=True, step=4, max_range=0.8)
    flat = sc.cloud(organised=False, step=4, max_range=0.8)
    g = sc.cast(step=4, max_range=0.8)
    assert org.shape == g["xyz"].shape and np.isnan(org).any()
    assert flat.shape[0] == int(g["valid"].sum())
    # a ray pixel through the box is blocked by it: no table point inside the footprint
    t = sc.truth()
    assert np.isfinite(flat).all()
    z_top = g["xyz"][g["label"] == cg.LABEL_TOP][:, 2]
    assert z_top.max() < sc.cam_dist * 2


def test_noise_option():
    sc = cg.Scene()
    g0 = sc.cast(step=4)
    g1 = sc.cast(step=4, noise_sigma_m=0.002, seed=3)
    g2 = sc.cast(step=4, noise_sigma_m=0.002, seed=3)
    d = g1["xyz"][g0["valid"], 2] - g0["xyz"][g0["valid"], 2]
    assert np.std(d) == pytest.approx(0.002, rel=0.1)
    assert abs(d.mean()) < 3e-4
    assert np.array_equal(np.nan_to_num(g1["xyz"]), np.nan_to_num(g2["xyz"]))
    hh, _, _, rms = height_from_planes(g1)
    assert hh == pytest.approx(sc.h, abs=1e-3)
    assert 0.5 * 0.002 < rms <= 0.002   # noise is along the ray, not the normal


def test_write_scene_roundtrip_through_pipeline_readers(tmp_path):
    sc = cg.Scene(yaw_deg=12)
    out = cg.write_scene(tmp_path, "synth_000001", sc, step=2)
    xyz, info = _pcd.read_pcd_xyz(out["pcd"])
    assert xyz.shape[0] == out["n_points"] and info["data"] == "binary"
    cam, _src = _depth.load_capture_camera(str(out["manifest"]))
    assert cam["fx"] == pytest.approx(sc.cam["fx"])
    m = json.loads(Path(out["manifest"]).read_text())
    assert m["pointcloud"]["path"] == str(out["pcd"]) and m["image"]["path"] == str(out["image"])
    # plane fit from the file: top plane height above table to 1 mm
    nt, dt, _ = fit_plane(xyz[np.abs(xyz @ sc.n + sc.cam_dist) < 1e-4].astype(np.float64))
    assert dt == pytest.approx(sc.cam_dist, abs=1e-3)
    # organised + ascii variants are readable too
    out2 = cg.write_scene(tmp_path, "synth_000002", sc, step=8, organised=True)
    xyz2, info2 = _pcd.read_pcd_xyz(out2["pcd"])
    assert info2["organized"] and xyz2.shape[0] == out2["n_points"]
    cg.write_pcd(tmp_path / "a.pcd", xyz2[:50], binary=False)
    xa, ia = _pcd.read_pcd_xyz(tmp_path / "a.pcd")
    assert ia["data"] == "ascii" and np.allclose(xa, xyz2[:50], atol=1e-5)


def test_rgb_image_colours(tmp_path):
    import cv2
    sc = cg.Scene()
    out = cg.write_scene(tmp_path, "synth_000003", sc, step=8)
    im = cv2.imread(str(out["image"]), cv2.IMREAD_COLOR)[..., ::-1]
    assert im.shape == (1080, 1920, 3)
    g = sc.cast(step=1)
    top = im[g["label"] == cg.LABEL_TOP]
    tab = im[g["label"] == cg.LABEL_TABLE]
    assert (top == cg.RGB_TOP).all() and (tab == cg.RGB_TABLE).all()
    wall = g["label"] == cg.LABEL_WALLS[0]
    for l in cg.LABEL_WALLS:
        sel = g["label"] == l
        if sel.any():
            assert im[sel].sum(axis=1).mean() < top.sum(axis=1).mean()   # darker than top
