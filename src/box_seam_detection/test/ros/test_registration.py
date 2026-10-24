"""Pure tests of cloud.register_depth_to_color (node-side depth -> colour registration), real Gemini 336 calibration."""
import json
from pathlib import Path

import numpy as np
import pytest

from box_seam_detection.ros import cloud

# measured on the live Gemini 336 (2026-10-20)
K_C = [1037.8170166015625, 0.0, 977.0050048828125, 0.0, 1037.6590576171875, 548.9818115234375, 0.0, 0.0, 1.0]
D_C = [0.007051904685795307, -0.049205388873815536, 0.00026131237973459065, 9.177037281915545e-05,
       0.03320758789777756, 0.0, 0.0, 0.0]
K_D = [624.600830078125, 0.0, 633.5, 0.0, 624.600830078125, 399.0, 0.0, 0.0, 1.0]
D_D = [0.0] * 8
T_CD = cloud.transform_matrix([-0.013975946115669187, -9.226609903706818e-05, -0.0016874226577281194],
                              [-0.0007030647418722058, -0.0019073589236682786, 0.0016889231456703457,
                               0.9999965076041581])
DW, DH, CW, CH = 1280, 800, 1920, 1080

SAMPLE_DIR = Path("/media/farhan/My Passport/igus_nvidia/bsd_raw_depth_samples_20261020")


def register(depth, scale=1.0, **kw):
    args = dict(k_depth=K_D, d_depth=D_D, color_k=K_C, color_d=D_C, color_model="plumb_bob",
                t_color_depth=T_CD, out_w=CW, out_h=CH)
    args.update(kw)
    return cloud.register_depth_to_color(depth, scale, **args)


# ------------------------------------------------------------------------------------------ synthetic scene
class Scene:
    """Planes in the COLOUR frame; each is (n, d, region) with n.P = d and region(P_c) -> bool (None = everywhere)."""

    def __init__(self):
        self.surfaces = [
            (np.array([0.0, 0.0, 1.0]), 1.0, None),                                   # backdrop Z = 1.0 m
            (np.array([0.0, 0.0, 1.0]), 0.7,                                           # nearer box face Z = 0.7 m
             lambda P: (np.abs(P[0] - 0.05) < 0.12) & (np.abs(P[1]) < 0.08)),
        ]
        # tilt the backdrop slightly so the planar test is not trivially fronto-parallel
        n = np.array([0.15, -0.10, 1.0])
        self.surfaces[0] = (n / np.linalg.norm(n), 1.0 / np.linalg.norm(n), None)

    def hit(self, origin, dirs):
        """Nearest hit of rays origin + s*dirs (dirs (N,3) in the colour frame). Returns (z_c (N,), has_hit)."""
        best_s = np.full(dirs.shape[0], np.inf)
        for n, d, region in self.surfaces:
            denom = dirs @ n
            with np.errstate(divide="ignore", invalid="ignore"):
                s = (d - origin @ n) / denom
            P = origin + s[:, None] * dirs
            ok = (s > 0) & np.isfinite(s)
            if region is not None:
                ok &= region(P.T)
            best_s = np.where(ok & (s < best_s), s, best_s)
        P = origin + best_s[:, None] * dirs
        return P[:, 2], np.isfinite(best_s)

    def depth_image(self, scale=1.0, dtype=np.float32):
        """What the depth camera sees: z in the depth frame at each depth pixel."""
        us, vs = np.meshgrid(np.arange(DW), np.arange(DH))
        rays_d = np.stack([(us - K_D[2]) / K_D[0], (vs - K_D[5]) / K_D[4], np.ones_like(us, dtype=float)], -1)
        rays_d = rays_d.reshape(-1, 3)
        R, t = T_CD[:3, :3], T_CD[:3, 3]
        dirs_c = rays_d @ R.T
        # s scales the depth-frame ray (z_d = s because the ray has z = 1)
        best = np.full(dirs_c.shape[0], np.inf)
        for n, d, region in self.surfaces:
            denom = dirs_c @ n
            s = (d - t @ n) / denom
            P_c = t + s[:, None] * dirs_c
            ok = (s > 0) & np.isfinite(s)
            if region is not None:
                ok &= region(P_c.T)
            best = np.where(ok & (s < best), s, best)
        z = (best / scale).reshape(DH, DW)
        z[~np.isfinite(z)] = 0
        return z.astype(dtype)

    def expected_color_z(self, us, vs):
        """Analytic z in the colour frame at raw (distorted) colour pixels, by undistorting the pixel ray."""
        import cv2
        pts = np.stack([us, vs], -1).astype(np.float64).reshape(-1, 1, 2)
        und = cv2.undistortPoints(pts, np.array(K_C).reshape(3, 3), np.array(D_C[:5]), None, None).reshape(-1, 2)
        dirs = np.concatenate([und, np.ones((und.shape[0], 1))], 1)
        return self.hit(np.zeros(3), dirs)


@pytest.fixture(scope="module")
def scene_depth():
    sc = Scene()
    return sc, sc.depth_image()


def test_registered_z_matches_analytic_within_1mm(scene_depth):
    sc, depth = scene_depth
    out = register(depth)
    assert out.dtype == depth.dtype and out.shape == (CH, CW)
    rng = np.random.default_rng(3)
    us = rng.integers(60, CW - 60, 6000)
    vs = rng.integers(60, CH - 60, 6000)
    z_exp, has = sc.expected_color_z(us, vs)
    z_got = out[vs, us]
    # skip pixels near a surface edge in the colour view (a +-3 px neighbourhood must agree analytically)
    stable = has.copy()
    for du, dv in ((3, 0), (-3, 0), (0, 3), (0, -3)):
        z2, h2 = sc.expected_color_z(us + du, vs + dv)
        stable &= h2 & (np.abs(z2 - z_exp) < 0.01)
    # ... and not in the occlusion shadow behind the nearer box (no data there is correct, checked separately)
    measurable = stable & (z_got > 0)
    assert measurable.sum() > 3000
    err = np.abs(z_got[measurable] - z_exp[measurable])
    assert err.max() < 1e-3, f"max z error {err.max() * 1e3:.3f} mm"
    # essentially every stable pixel is covered: only a thin shadow band beside the box may be empty
    assert (stable & (z_got == 0)).sum() < 0.03 * stable.sum()


def test_uint16_millimetre_encoding_within_1_5mm(scene_depth):
    sc, depth = scene_depth
    d16 = np.rint(depth * 1000.0).astype(np.uint16)
    out = register(d16, 0.001)
    assert out.dtype == np.uint16
    rng = np.random.default_rng(4)
    us, vs = rng.integers(60, CW - 60, 4000), rng.integers(60, CH - 60, 4000)
    z_exp, has = sc.expected_color_z(us, vs)
    ok = has & (out[vs, us] > 0)
    for du, dv in ((3, 0), (-3, 0), (0, 3), (0, -3)):
        ok &= np.abs(sc.expected_color_z(us + du, vs + dv)[0] - z_exp) < 0.01
    assert ok.sum() > 2000
    assert np.abs(out[vs, us][ok] * 0.001 - z_exp[ok]).max() < 1.5e-3   # 0.5 mm in + 0.5 mm out quantisation


def test_near_object_wins_zbuffer(scene_depth):
    sc, depth = scene_depth
    out = register(depth)
    # centre of the nearer box (X=0.05, Y=0 at Z=0.7) in the colour image: only 0.7 m values, never the 1.0 m backdrop
    u = int(round(K_C[0] * 0.05 / 0.7 + K_C[2]))
    v = int(round(K_C[4] * 0.0 / 0.7 + K_C[5]))
    patch = out[v - 40:v + 41, u - 60:u + 61]
    assert (patch > 0).all()
    assert np.abs(patch - 0.7).max() < 1e-3
    # the near surface is also written over the backdrop wherever both project to the same colour pixel: the
    # histogram of z has no value of the far surface inside the box footprint (checked on a strip across its edge)
    row = out[v, :]
    inside = row[(row > 0) & (row < 0.85)]
    assert inside.size > 100 and np.abs(inside - 0.7).max() < 1e-3


@pytest.mark.parametrize("z_plane", [0.35, 1.0, 2.5])
def test_no_holes_inside_a_planar_region(z_plane):
    depth = np.full((DH, DW), z_plane, dtype=np.float32)
    out = register(depth)
    # interior of the colour image fully inside the depth field of view (depth FOV is wider): zero holes
    inner = out[40:CH - 40, 40:CW - 40]
    assert (inner > 0).all(), f"{int((inner == 0).sum())} holes at Z={z_plane}"
    # a tilted plane too
    us, vs = np.meshgrid(np.arange(DW), np.arange(DH))
    tilted = (z_plane * (1 + 2e-4 * (us - 640) + 1.5e-4 * (vs - 400))).astype(np.float32)
    out = register(tilted)
    inner = out[40:CH - 40, 40:CW - 40]
    assert (inner > 0).all(), f"{int((inner == 0).sum())} holes on the tilted plane at Z={z_plane}"
    # depth noise (0.2 % of the range, i.e. 0.7 mm at 0.35 m, 5 mm at 2.5 m) must not crack the surface either
    noisy = (depth * (1 + np.random.default_rng(5).normal(0, 0.002, depth.shape))).astype(np.float32)
    inner = register(noisy)[40:CH - 40, 40:CW - 40]
    assert (inner > 0).all(), f"{int((inner == 0).sum())} holes with depth noise at Z={z_plane}"


def test_pixel_positions_match_cv2_project_points():
    import cv2
    k = np.array(K_C).reshape(3, 3)
    sensitive = 0
    for (u, v) in [(640, 400), (80, 110), (1185, 110), (80, 690), (1185, 690), (200, 400), (1100, 650)]:
        z = 1.2
        depth = np.zeros((DH, DW), dtype=np.float32)
        depth[v, u] = z
        out = register(depth)
        vs, us = np.nonzero(out)
        assert vs.size >= 1, f"pixel ({u},{v}) vanished"
        p_d = np.array([(u - K_D[2]) / K_D[0] * z, (v - K_D[5]) / K_D[4] * z, z])
        p_c = T_CD[:3, :3] @ p_d + T_CD[:3, 3]
        proj, _ = cv2.projectPoints(p_c.reshape(1, 1, 3), np.zeros(3), np.zeros(3), k, np.array(D_C[:5]))
        pu, pv = proj.reshape(2)
        pinhole = k @ (p_c / p_c[2])
        if abs(pu - pinhole[0]) > 2 or abs(pv - pinhole[1]) > 2:
            sensitive += 1
        # the footprint covers the projected centre and is centred on it within half a colour pixel
        assert us.min() - 0.5 <= pu <= us.max() + 0.5 and vs.min() - 0.5 <= pv <= vs.max() + 0.5
        assert abs(us.mean() - pu) < 0.8 and abs(vs.mean() - pv) < 0.8
        assert np.allclose(out[vs, us], p_c[2], atol=1e-5)
        assert us.size <= 16
    assert sensitive >= 2   # the test would catch a missing distortion model (corner pixels move > 2 px)


def _reference_zbuffer(depth, scale, kd, kc, T, w, h, margin=cloud.FOOTPRINT_MARGIN_PX):
    """Slow per-pixel reference of the documented footprint rule (no distortion)."""
    R, t = T[:3, :3], T[:3, 3]
    z = np.full((h, w), np.inf)
    for v in range(depth.shape[0]):
        for u in range(depth.shape[1]):
            zd = depth[v, u] * scale
            if not zd > 0:
                continue
            xs, ys, zs = [], [], []
            for du, dv in ((-0.5, -0.5), (0.5, -0.5), (-0.5, 0.5), (0.5, 0.5)):
                p = R @ np.array([(u + du - kd[2]) / kd[0] * zd, (v + dv - kd[5]) / kd[4] * zd, zd]) + t
                xs.append(kc[0] * p[0] / p[2] + kc[2])
                ys.append(kc[4] * p[1] / p[2] + kc[5])
                zs.append(p[2])
            xl, xh = np.ceil(min(xs) - margin), np.ceil(max(xs) + margin) - 1
            yl, yh = np.ceil(min(ys) - margin), np.ceil(max(ys) + margin) - 1
            xh, yh = max(xh, xl), max(yh, yl)
            zc = float(np.mean(zs))
            for y in range(int(yl), int(yh) + 1):
                for x in range(int(xl), int(xh) + 1):
                    if 0 <= x < w and 0 <= y < h:
                        z[y, x] = min(z[y, x], zc)
    z[~np.isfinite(z)] = 0
    return z


def test_matches_bruteforce_zbuffer_with_many_collisions():
    rng = np.random.default_rng(11)
    kd = [60.0, 0, 39.5, 0, 60.0, 29.5, 0, 0, 1]
    kc = [95.0, 0, 63.5, 0, 95.0, 47.5, 0, 0, 1]
    T = cloud.transform_matrix([-0.06, 0.02, -0.01], [0.0, 0.0, 0.0, 1.0])
    depth = rng.uniform(0.3, 1.5, (60, 80)).astype(np.float32)
    depth[rng.random((60, 80)) < 0.1] = 0
    got = cloud.register_depth_to_color(depth, 1.0, kd, [0] * 5, kc, [0] * 5, "plumb_bob", T, 128, 96)
    ref = _reference_zbuffer(depth, 1.0, kd, kc, T, 128, 96)
    assert (ref > 0).sum() > 5000
    np.testing.assert_allclose(got, ref, atol=1e-6)


def test_equal_depth_ties_are_deterministic(scene_depth):
    _, depth = scene_depth
    a, b = register(depth), register(depth.copy())
    assert (a == b).all()


def test_numpy_fancy_assignment_keeps_last_for_repeated_index():
    # register_depth_to_color relies on this (documented as unspecified order by numpy, sequential in practice)
    idx = np.repeat(np.arange(1000), 7)
    vals = np.tile(np.arange(7, dtype=float), 1000)
    buf = np.zeros(1000)
    buf[idx] = vals
    assert (buf == 6).all()


def test_float_invalid_values_and_dtype_preserved():
    depth = np.full((DH, DW), 1.0, dtype=np.float32)
    depth[::7, ::5] = np.nan
    depth[3, :] = -1.0
    out = register(depth)
    assert out.dtype == np.float32 and np.isfinite(out).all() and (out >= 0).all()


@pytest.mark.parametrize("mut", [
    dict(d_depth=[0.0, 0.0, 0.001, 0.0, 0.0]),                                  # depth lens distortion
    dict(d_depth=[0.1] + [0.0] * 7),
    dict(color_model="equidistant"),
    dict(color_model="fisheye"),
    dict(color_d=[0.1, 0.0, 0.0]),                                              # wrong coefficient count
    dict(color_d=[np.nan, 0, 0, 0, 0]),
    dict(color_k=[0.0, 0, 1, 0, 0, 1, 0, 0, 1]),                                # fx = 0
    dict(color_k=[1.0, 2.0]),
    dict(k_depth=[600, 0, 0, 0, -600, 0, 0, 0, 1]),
    dict(t_color_depth=np.eye(3)),
    dict(t_color_depth=np.diag([2.0, 2.0, 2.0, 1.0])),                          # not a rotation
    dict(t_color_depth=np.array([[1, 0, 0, 5.0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])),   # 5 m baseline
    dict(t_color_depth=np.full((4, 4), np.nan)),
    dict(out_w=0),
    dict(out_h=-5),
])
def test_fail_closed(mut):
    depth = np.full((DH, DW), 1.0, dtype=np.float32)
    with pytest.raises(cloud.CaptureError):
        register(depth, **mut)


@pytest.mark.parametrize("bad", [np.zeros((3, DH, DW), dtype=np.float32), np.zeros((0, 5)), [[1, 2]],
                                 np.zeros((4, 4), dtype=bool)])
def test_fail_closed_bad_depth(bad):
    with pytest.raises(cloud.CaptureError):
        register(bad)


def test_fail_closed_bad_scale():
    for s in (0.0, -1.0, float("nan")):
        with pytest.raises(cloud.CaptureError):
            register(np.ones((4, 4), dtype=np.uint16), s)


def test_all_zero_distortion_variants_accepted():
    depth = np.full((DH, DW), 1.0, dtype=np.float32)
    assert (register(depth, d_depth=[]) > 0).any()
    assert (register(depth, d_depth=[0.0] * 5) > 0).any()
    assert (register(depth, color_d=[], color_model="") > 0).any()


def test_transform_matrix_validation():
    with pytest.raises(cloud.CaptureError):
        cloud.transform_matrix([0, 0], [0, 0, 0, 1])
    with pytest.raises(cloud.CaptureError):
        cloud.transform_matrix([0, 0, 0], [0, 0, 0, 3])
    m = cloud.transform_matrix([1, 2, 3], [0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)])   # 90 deg about z
    assert np.allclose(m[:3, :3], [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-12) and np.allclose(m[:3, 3], [1, 2, 3])


# ------------------------------------------------------------------------------------------ real data
def _sample():
    if not (SAMPLE_DIR / "sample.npz").exists():
        pytest.skip("real depth sample not available")
    return np.load(SAMPLE_DIR / "sample.npz"), json.loads((SAMPLE_DIR / "rec.json").read_text())


def _edge_map(depth_mm, rel=0.03, floor_mm=25):
    """Depth-discontinuity mask: neighbouring valid pixels whose depth differs by more than rel*z + floor_mm."""
    valid = depth_mm > 0
    d = depth_mm.astype(np.float32)
    e = np.zeros(d.shape, dtype=bool)
    lim = rel * d + floor_mm
    dx = np.abs(d[:, 1:] - d[:, :-1]) > np.maximum(lim[:, 1:], lim[:, :-1])
    dy = np.abs(d[1:, :] - d[:-1, :]) > np.maximum(lim[1:, :], lim[:-1, :])
    e[:, 1:] |= dx & valid[:, 1:] & valid[:, :-1]
    e[1:, :] |= dy & valid[1:, :] & valid[:-1, :]
    return e


NEAR_BOX_SEED = (820, 800)           # a pixel on the cardboard box in the foreground of the sample (colour grid)
NEAR_BOX_ROI = (700, 1080, 560, 1120)  # y0, y1, x0, x1 around it


def _box_outline_offset(depth_mm, gray, radius=20):
    """Offset (dx, dy) in colour pixels that best moves the depth outline of the foreground box onto the colour
    box outline: the shift that maximises the mean colour-gradient magnitude along the depth outline (thousands of
    contour pixels, so robust against single-edge stereo fattening). Also returns the contour length."""
    import cv2
    g = cv2.GaussianBlur(gray, (0, 0), 1.2).astype(np.float32)
    mag = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    zc = int(depth_mm[NEAR_BOX_SEED[1], NEAR_BOX_SEED[0]])
    assert zc > 0
    near = ((depth_mm > 0) & (np.abs(depth_mm.astype(np.int32) - zc) < 120)).astype(np.uint8)
    _, lab = cv2.connectedComponents(near)
    box = (lab == lab[NEAR_BOX_SEED[1], NEAR_BOX_SEED[0]]).astype(np.uint8)
    contour = (box - cv2.erode(box, np.ones((3, 3), np.uint8))) > 0
    y0, y1, x0, x1 = NEAR_BOX_ROI
    roi = np.zeros_like(contour)
    roi[y0:y1, x0:x1] = True
    vs, us = np.nonzero(contour & roi)
    best, best_s = (0, 0), -1.0
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            sc = float(mag[np.clip(vs + dy, 0, gray.shape[0] - 1), np.clip(us + dx, 0, gray.shape[1] - 1)].mean())
            if sc > best_s:
                best, best_s = (dx, dy), sc
    return best[0], best[1], int(vs.size), box.astype(bool)


def test_real_sample_registration_and_overlay():
    import cv2
    z, rec = _sample()
    key = next(k for k in z.files if k.startswith("unal_0_"))
    ckey = next(k for k in z.files if k.startswith("color_0_"))
    raw = z[key]
    info = rec["info"]
    t = rec["tf"]["camera_color_optical_frame<-camera_depth_optical_frame"]
    T = cloud.transform_matrix(t[:3], t[3:])
    stats = {}

    def reg(TT, st=None):
        return cloud.register_depth_to_color(raw, 0.001, info["depth_1280"]["K"], info["depth_1280"]["D"],
                                             info["color_1920"]["K"], info["color_1920"]["D"],
                                             info["color_1920"]["model"], TT, 1920, 1080, stats=st)
    out = reg(T, stats)
    cov = float((out > 0).mean())
    in_cov = float((raw > 0).mean())
    # the depth field of view is wider than the colour one: ~18 % of the valid depth falls outside the colour image
    assert stats["input_valid"] == int(((raw > 0) & (raw < 65535)).sum())   # 65535 = invalid marker, dropped
    assert stats["outside_image"] < 0.35 * stats["input_valid"] and stats["dropped_degenerate"] == 0
    assert cov > 0.5 * in_cov and cov > 0.4, f"coverage {cov:.3f} (depth valid {in_cov:.3f})"
    assert out.dtype == np.uint16 and int(out[out > 0].min()) >= 100 and int(out.max()) < 20000
    assert abs(float(np.median(out[out > 0])) - float(np.median(raw[raw > 0]))) < 100

    colour = z[ckey]
    gray = cv2.cvtColor(np.ascontiguousarray(colour[:, :, ::-1]), cv2.COLOR_BGR2GRAY)
    dx, dy, n_contour, box = _box_outline_offset(out, gray)
    # comparison: the same raw depth without the extrinsic, and a plain resize (what a missing registration looks like)
    idx, idy, _, _ = _box_outline_offset(reg(np.eye(4)), gray)
    naive = cv2.resize(raw, (1920, 1080), interpolation=cv2.INTER_NEAREST)
    ndx, ndy, _, _ = _box_outline_offset(naive, gray, radius=30)
    print(f"\nbox outline: colour edge is ({dx},{dy}) px from the registered depth outline ({n_contour} contour px); "
          f"identity extrinsic ({idx},{idy}); plain resize ({ndx},{ndy}); coverage {cov:.3f}; stats {stats}")
    assert max(abs(dx), abs(dy)) <= 3, f"registered depth outline is off by ({dx},{dy}) px from the colour box outline"
    assert abs(idx) > abs(dx) + 4        # the extrinsic matters at this range (14 mm baseline = ~15 px at 0.95 m)

    overlay = np.ascontiguousarray(colour[:, :, ::-1]).copy()     # BGR
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    overlay[cv2.dilate(_edge_map(out).astype(np.uint8), k3) > 0] = (0, 0, 255)                 # depth edges: red
    cont, _ = cv2.findContours(box.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(overlay, cont, -1, (0, 255, 0), 1)                                         # foreground box: green
    cv2.imwrite("/tmp/bsd_reg_overlay.png", overlay)
    zoom = overlay[640:1080, 520:1180]
    cv2.imwrite("/tmp/bsd_reg_overlay_box_zoom.png", cv2.resize(zoom, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
    naive_overlay = np.ascontiguousarray(colour[:, :, ::-1]).copy()
    naive_overlay[cv2.dilate(_edge_map(naive).astype(np.uint8), k3) > 0] = (0, 255, 255)
    cv2.imwrite("/tmp/bsd_reg_overlay_naive_resize.png", naive_overlay)
    assert Path("/tmp/bsd_reg_overlay.png").exists()


def test_real_sample_through_backprojection_is_a_clean_cloud():
    z, rec = _sample()
    key = next(k for k in z.files if k.startswith("unal_1_"))
    ckey = next(k for k in z.files if k.startswith("color_0_"))
    info = rec["info"]
    t = rec["tf"]["camera_color_optical_frame<-camera_depth_optical_frame"]
    out = cloud.register_depth_to_color(z[key], 0.001, info["depth_1280"]["K"], info["depth_1280"]["D"],
                                        info["color_1920"]["K"], info["color_1920"]["D"], "plumb_bob",
                                        cloud.transform_matrix(t[:3], t[3:]), 1920, 1080)
    pts = cloud.backproject_depth(out, z[ckey], info["color_1920"]["K"], 0.001)
    assert pts.shape[0] == int((out > 0).sum()) and np.isfinite(pts[:, :3]).all()
