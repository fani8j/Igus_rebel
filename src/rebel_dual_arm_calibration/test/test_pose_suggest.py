"""Next-calibration-pose suggestion engine: synthetic arm, seeded, no ROS and no recorded session data."""
import math
import time

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from rebel_dual_arm_calibration import pose_suggest as P
from rebel_dual_arm_calibration.solver import (Sample, SolverOptions, _ArmModel, _numeric_jacobian, project_points,
                                               rotation_diversity_deg)
from rebel_dual_arm_calibration.targets import CHESS_ID_BASE, TargetSpec
from rebel_dual_arm_calibration.transforms import inv_T, make_T, rotation_angle_deg, xyz_rpy_to_T

W, H = 1920, 1080
K = np.array([[1038.0, 0.0, 977.0], [0.0, 1037.7, 549.0], [0.0, 0.0, 1.0]])
DIST = np.array([0.007, -0.049, 0.0003, 0.0001, 0.033, 0.0, 0.0, 0.0])
# camera ~0.87 m above the base looking forward and down at ~45 deg (like the real cell)
X_TRUE = np.array([[-0.0628, -0.9973, -0.0382, 0.2421],
                   [-0.7106, 0.0715, -0.7000, 0.5651],
                   [0.7008, -0.0168, -0.7132, 0.6465],
                   [0.0, 0.0, 0.0, 1.0]])
X_TRUE[:3, :3] = Rotation.from_matrix(X_TRUE[:3, :3]).as_matrix()      # re-orthonormalise the rounded values
Z_TRUE = xyz_rpy_to_T([0.018, -0.103, 0.046], [-1.58, 0.0, 1.573])
SPEC = TargetSpec(arm="right", type="charuco", dictionary="DICT_4X4_50", marker_length=0.026, marker_ids=[20, 21, 22, 23],
                  square_length=0.035, squares_x=3, squares_y=3)
_PTS = SPEC.all_points_obj()
IDS = np.array(sorted(_PTS))
OBJ = np.array([_PTS[i] for i in IDS])
NOISE_PX = 0.15


def _ray(u, v):
    xy = cv2.undistortPoints(np.array([[[u, v]]], dtype=np.float64), K, DIST).reshape(2)
    r = np.array([xy[0], xy[1], 1.0])
    return r / np.linalg.norm(r)


def _board_pose(u, v, d, turn=0.0, tip=0.0, spin=0.0):
    """Optical-frame board pose: board centre on the ray through pixel (u,v) at distance d, printed face towards the
    camera, rotated by turn/tip/spin (deg) in the line-of-sight frame."""
    r = _ray(u, v)
    Rf = P._ray_frames(r[None])[0]
    M = Rotation.from_euler("YXZ", [turn, tip, spin], degrees=True).as_matrix()
    return make_T(Rf @ M, r * d)


def _flange(T_ob):
    return inv_T(X_TRUE) @ T_ob @ inv_T(Z_TRUE)


def _sample(T_ob, rng, joints=None, noise=NOISE_PX):
    A = _flange(T_ob)
    Pc = (X_TRUE @ A @ Z_TRUE)
    pts = OBJ @ Pc[:3, :3].T + Pc[:3, 3]
    pix = project_points(K, DIST, pts) + rng.normal(0.0, noise, (len(OBJ), 2))
    return Sample(arm="right", T_base_link6=A, ids=IDS.copy(), obj=OBJ.copy(), pix=pix, K=K.copy(), dist=DIST.copy(),
                  image_size=(W, H), joints=dict(joints or {}))


def make_samples(n=16, seed=3, cells=None):
    """Samples with limited rotation variety (+-20 deg turn, +-12 deg tip, +-12 deg spin) on a ring of image points
    (restricted to the given 3x3 cells if ``cells`` is given)."""
    rng = np.random.default_rng(seed)
    out, k = [], 0
    while len(out) < n:
        u, v = rng.uniform(300, W - 300), rng.uniform(200, H - 200)
        if cells is not None and int(v // (H / 3)) * 3 + int(u // (W / 3)) not in cells:
            continue
        d = rng.uniform(0.6, 0.9)
        T = _board_pose(u, v, d, rng.uniform(-20, 20), rng.uniform(-12, 12), rng.uniform(-12, 12))
        joints = {f"joint{i + 1}": float(x) for i, x in enumerate(rng.uniform(-1.0, 1.0, 6))}
        out.append(_sample(T, rng, joints))
        k += 1
    return out


@pytest.fixture(scope="module")
def world():
    ss = make_samples(16, seed=3)
    return ss, P.fit_arm(ss, "right")


@pytest.fixture(scope="module")
def cands(world):
    return P.generate_candidates(world[1])


def ik_map(T, seed_q=None, prefix=""):
    """Deterministic fake IK: joints are a smooth function of the flange pose (not real kinematics)."""
    rv = Rotation.from_matrix(T[:3, :3]).as_rotvec()
    v = np.concatenate([T[:3, 3] * 1.0, rv * 0.3])
    return {f"{prefix}joint{i + 1}": float(x) for i, x in enumerate(v)}


def _always_valid(q):
    return True, ""


# ---------------------------------------------------------------------------------------------------------------
def test_fit_recovers_calibration(world):
    ss, fit = world
    assert len(fit.samples) == len(ss)
    # the quantity that matters: board poses predicted for arbitrary flange poses
    rng = np.random.default_rng(11)
    for _ in range(5):
        T = _board_pose(rng.uniform(300, W - 300), rng.uniform(200, H - 200), rng.uniform(0.6, 0.9),
                        rng.uniform(-40, 40), rng.uniform(-35, 35), rng.uniform(-30, 30))
        A = _flange(T)
        Tp = fit.X @ A @ fit.Z
        assert np.linalg.norm(Tp[:3, 3] - T[:3, 3]) < 3e-3
        assert rotation_angle_deg(Tp[:3, :3], T[:3, :3]) < 0.5
    assert np.linalg.norm(fit.X[:3, 3] - X_TRUE[:3, 3]) < 5e-3
    assert rotation_angle_deg(fit.X[:3, :3], X_TRUE[:3, :3]) < 0.5
    assert np.linalg.norm(fit.Z[:3, 3] - Z_TRUE[:3, 3]) < 5e-3
    assert rotation_angle_deg(fit.Z[:3, :3], Z_TRUE[:3, :3]) < 0.5
    assert fit.image_size == (W, H) and fit.obj_full.shape == (len(OBJ), 3) and fit.info_old.shape == (12, 12)


def test_fit_estimates_board_outline_size(world):
    fit = world[1]
    o = P._outline_obj(fit)
    assert np.allclose(np.ptp(o[:, 0]), 0.105, atol=1e-6) and np.allclose(np.ptp(o[:, 1]), 0.105, atol=1e-6)
    # every detected corner of every sample lies inside the predicted outline polygon
    for s in world[0]:
        poly = np.array(P.predicted_outline(fit, s.T_base_link6), dtype=np.float32)
        for u, v in s.pix:
            assert cv2.pointPolygonTest(poly, (float(u), float(v)), False) >= 0
    fit2 = P.fit_arm(world[0], "right", target=SPEC)
    assert fit2.board_size == pytest.approx((0.105, 0.105))


def test_too_few_samples_raises():
    ss = make_samples(8, seed=4)
    with pytest.raises(ValueError, match="at least 6"):
        P.fit_arm(ss[:5], "right")
    with pytest.raises(ValueError):
        P.fit_arm(ss, "left")                   # no samples of that arm
    P.fit_arm(ss[:6] + ss[6:], "right")        # >= 6 is fine


def test_candidate_geometry(world, cands):
    ss, fit = world
    opt = P.SuggestOptions()
    assert len(cands) > 100
    assert [c.rank for c in cands] == list(range(1, len(cands) + 1))
    assert all(a.score >= b.score for a, b in zip(cands, cands[1:]))
    cam_pos_view = []
    for c in cands:
        assert c.q is None and c.arm == "right"
        assert len(c.outline_px) == 4
        for u, v in c.outline_px:
            assert opt.image_margin_px <= u <= W - opt.image_margin_px
            assert opt.image_margin_px <= v <= H - opt.image_margin_px
        # recompute independently
        A = c.T_base_link6
        assert (inv_T(X_TRUE) @ (X_TRUE @ A @ Z_TRUE))[2, 3] > 0.0
        T_ob = fit.X @ A @ fit.Z
        centre_base = (inv_T(fit.X) @ T_ob)[:3, 3]
        assert centre_base[2] >= opt.min_height_m - 1e-9
        assert np.linalg.norm(A[:3, 3]) <= opt.max_reach_m + 1e-9
        ray = T_ob[:3, 3] / np.linalg.norm(T_ob[:3, 3])
        view = math.degrees(math.acos(min(1.0, float(T_ob[:3, 2] @ ray))))        # board z points away from the camera
        assert view <= opt.max_view_angle_deg + 1e-6
        assert view == pytest.approx(c.gains["view_angle_deg"], abs=1e-3)
        assert c.gains["distance_m"] == pytest.approx(np.linalg.norm(T_ob[:3, 3]))
        assert np.allclose(c.outline_px, P.predicted_outline(fit, A), atol=1e-6)
        assert math.isfinite(c.score)
        for k in ("info_gain", "rot_spread2_before", "rot_spread2_after", "empty_cell", "view_angle_deg",
                  "distance_m", "max_joint_change_deg"):
            assert k in c.gains
        cam_pos_view.append(view)
    assert max(cam_pos_view) > 30.0            # tilted views are really proposed


def _plane_slopes(fit, c):
    """Depth slopes (z per unit x, z per unit y) of the board plane in the line-of-sight frame, from the 4 outer corners."""
    T_ob = fit.X @ c.T_base_link6 @ fit.Z
    pts = P._outline_obj(fit) @ T_ob[:3, :3].T + T_ob[:3, 3]
    ray = T_ob[:3, 3] / np.linalg.norm(T_ob[:3, 3])
    Rf = P._ray_frames(ray[None])[0]
    q = (pts - T_ob[:3, 3]) @ Rf              # coordinates in the line-of-sight frame (x right, y down, z forward)
    a, b, _ = np.linalg.lstsq(np.c_[q[:, 0], q[:, 1], np.ones(4)], q[:, 2], rcond=None)[0]
    return a, b


def test_description_words_match_geometry(world, cands):
    fit = world[1]
    import re
    pat = re.compile(r"^(\w+)-(\w+) of image, (left|right) edge closer (\d+) deg, top tipped (away|towards) (\d+) deg"
                     r"(?:, wrist roll [+-]\d+ deg vs usual)?, ([0-9.]+) m$")
    seen = set()
    for c in cands:
        d = c.description
        m = pat.match(d) or (re.match(r"^centre of image, (left|right) edge closer (\d+) deg, top tipped (away|towards)"
                                      r" (\d+) deg(?:, wrist roll [+-]\d+ deg vs usual)?, ([0-9.]+) m$", d))
        assert m, d
        g = m.groups()
        side, turn, ud, tip = (g[2], g[3], g[4], g[5]) if len(g) == 7 else (g[0], g[1], g[2], g[3])
        a, b = _plane_slopes(fit, c)
        # right edge closer <=> depth decreases towards image right; top towards <=> depth increases downwards
        if abs(a) > 1e-3:
            assert (side == "right") == (a < 0), d
        if abs(b) > 1e-3:
            assert (ud == "towards") == (b > 0), d
        assert int(turn) == pytest.approx(abs(math.degrees(math.atan(a))), abs=0.6), d
        n = np.array([-a, -b, 1.0]) / np.linalg.norm([a, b, 1.0])
        assert int(tip) == pytest.approx(abs(math.degrees(math.asin(-n[1]))), abs=0.6), d
        # position words from the projected centre
        u, v = c.centre_px
        row = "upper" if v < H / 3 else "lower" if v > 2 * H / 3 else "middle"
        col = "left" if u < W / 3 else "right" if u > 2 * W / 3 else "centre"
        assert d.startswith("centre of image" if (row, col) == ("middle", "centre") else f"{row}-{col} of image"), d
        assert f"{c.gains['distance_m']:.2f} m" in d
        seen.add((side, ud))
    assert len(seen) == 4                      # all four turn/tip sign combinations occur


def test_sign_conventions_numeric(world):
    """Hand-built poses: positive turn = image-right side nearer the camera, positive tip = top nearer."""
    fit = world[1]
    for turn, tip, exp in ((30.0, 0.0, "right edge closer 30 deg, top tipped away 0 deg"),
                           (-30.0, 0.0, "left edge closer 30 deg, top tipped away 0 deg"),
                           (0.0, 20.0, "top tipped towards 20 deg"),
                           (0.0, -20.0, "top tipped away 20 deg")):
        T_ob = _board_pose(W / 2, H / 2, 0.7, turn, tip)
        s = P.score_poses(fit, [_flange(T_ob)])[0]
        assert exp in s.description, s.description
        pts = P._outline_obj(fit) @ T_ob[:3, :3].T + T_ob[:3, 3]        # TL TR BR BL in the optical frame
        ray = T_ob[:3, 3] / np.linalg.norm(T_ob[:3, 3])
        Rf = P._ray_frames(ray[None])[0]
        q = (pts - T_ob[:3, 3]) @ Rf
        if turn > 0:
            assert q[1, 2] < q[0, 2] and q[2, 2] < q[3, 2]               # TR, BR nearer than TL, BL
        if turn < 0:
            assert q[1, 2] > q[0, 2]
        if tip > 0:
            assert q[0, 2] < q[3, 2] and q[1, 2] < q[2, 2]               # top corners nearer
        if tip < 0:
            assert q[0, 2] > q[3, 2]
    # the image position of the centre: the centre pixel of the middle cell
    s = P.score_poses(fit, [_flange(_board_pose(W / 2, H / 2, 0.7))])[0]
    assert s.description.startswith("centre of image")
    assert s.centre_px == pytest.approx((W / 2, H / 2), abs=0.5)


def test_info_gain_matches_reference_model(world, cands):
    ss, fit = world
    model0 = _ArmModel(fit.samples, fit.X, fit.Z)
    from rebel_dual_arm_calibration.solver import _flag_outliers
    _, cmask = _flag_outliers(model0, np.zeros(12), SolverOptions())
    J0 = _numeric_jacobian(lambda th: model0.residuals(th, cmask), np.zeros(12))
    I0 = J0.T @ J0
    assert np.allclose(I0, fit.info_old, rtol=1e-9)
    for c in (cands[0], cands[len(cands) // 2], cands[-1]):
        T_ob = fit.X @ c.T_base_link6 @ fit.Z
        pts = fit.obj_full @ T_ob[:3, :3].T + T_ob[:3, 3]
        syn = Sample(arm="right", T_base_link6=c.T_base_link6, ids=fit.ids_full, obj=fit.obj_full,
                     pix=project_points(fit.K, fit.dist, pts), K=fit.K, dist=fit.dist, image_size=fit.image_size)
        syn.sigma = np.where(syn.ids >= CHESS_ID_BASE, 1.0, 2.0)
        m = _ArmModel([syn], fit.X, fit.Z)
        Jc = _numeric_jacobian(lambda th: m.residuals(th), np.zeros(12))
        ref = np.linalg.slogdet(I0 + Jc.T @ Jc)[1] - np.linalg.slogdet(I0)[1]
        assert c.gains["info_gain"] == pytest.approx(ref, rel=1e-5, abs=1e-8)
        assert c.gains["info_gain"] > 0.0
        # rotation spreads equal the solver function
        Rs = [s.T_base_link6[:3, :3] for s in fit.samples]
        before = rotation_diversity_deg(Rs)
        after = rotation_diversity_deg(Rs + [c.T_base_link6[:3, :3]])
        assert c.gains["rot_spread2_before"] == pytest.approx(before[1], rel=1e-9)
        assert c.gains["rot_spread2_after"] == pytest.approx(after[1], rel=1e-6)
        assert c.gains["rot_spread3_after"] == pytest.approx(after[2], rel=1e-6)


def test_rotation_variety_scores_higher_than_near_duplicate(world):
    ss, fit = world
    s0 = ss[2]
    dup = s0.T_base_link6.copy()
    dup[:3, 3] += [4e-4, -3e-4, 2e-4]                       # near-duplicate of a recorded sample
    # same flange position, but rotated 45 deg about the flange z axis + 25 deg about its x axis
    div = s0.T_base_link6.copy()
    div[:3, :3] = div[:3, :3] @ Rotation.from_euler("zx", [45.0, 25.0], degrees=True).as_matrix()
    # keep the board centre where the duplicate has it, so both land in the same image cell
    res = P.score_poses(fit, [dup, div])
    d, v = res
    assert v.gains["rot_gain_deg"] > d.gains["rot_gain_deg"] + 0.5
    assert v.gains["rot_spread2_after"] > d.gains["rot_spread2_after"]
    assert v.gains["info_gain"] > d.gains["info_gain"]
    assert v.score > d.score
    assert d.gains["info_gain"] < 0.25 * max(c.gains["info_gain"] for c in P.generate_candidates(fit))


def test_empty_cell_bonus(world):
    ss = make_samples(14, seed=5, cells={3, 4, 5})          # samples only in the middle row of cells
    fit = P.fit_arm(ss, "right")
    cs = P.generate_candidates(fit)
    taken = set()
    for s in fit.samples:
        uv, _ = P._project_batch(fit, (fit.X @ s.T_base_link6 @ fit.Z)[None], np.zeros((1, 3)))
        taken.add(int(P._cells(uv[:, 0], (W, H), 3, 3)[0]))
    assert taken <= {3, 4, 5}
    flags = {}
    for c in cs:
        cell = int(c.gains["cell"])
        want = 0.0 if cell in taken else 1.0
        assert c.gains["empty_cell"] == want
        flags.setdefault(want, 0)
        flags[want] += 1
    assert set(flags) == {0.0, 1.0}                         # both kinds are proposed
    # with every other weight at zero the score is exactly the bonus
    only = P.SuggestOptions(w_info=0.0, w_rot=0.0, w_dist=0.0, w_cell=0.5)
    for c in P.generate_candidates(fit, only):
        assert c.score == pytest.approx(0.5 * c.gains["empty_cell"])
    # with the default weights an empty-cell candidate leads for this data
    assert cs[0].gains["empty_cell"] == 1.0


def test_distance_bonus_and_weights(world, cands):
    fit = world[1]
    d_ex = [np.linalg.norm((fit.X @ s.T_base_link6 @ fit.Z)[:3, 3]) for s in fit.samples]
    for c in cands[:50]:
        near = any(abs(d - c.gains["distance_m"]) < 0.075 for d in d_ex)
        assert c.gains["dist_bonus"] == (0.0 if near else 1.0)
    o = P.SuggestOptions(w_info=0.0, w_rot=0.0, w_cell=0.0, w_dist=0.0)
    assert all(c.score == 0.0 for c in P.generate_candidates(fit, o))


def test_runtime_budget(world):
    fit = world[1]
    t0 = time.perf_counter()
    cs = P.generate_candidates(fit)
    dt = time.perf_counter() - t0
    assert dt < 3.0, dt
    assert len(cs) > 100
    # also with the full-size default grid but cheap geometry off (all 2700 raw candidates survive the filters)
    wide = P.SuggestOptions(min_height_m=-5.0, max_reach_m=50.0, max_view_angle_deg=89.0, image_margin_px=-1e6)
    t0 = time.perf_counter()
    cs2 = P.generate_candidates(fit, wide)
    assert time.perf_counter() - t0 < 3.0
    assert len(cs2) > 900


# ---- suggest() ---------------------------------------------------------------------------------------------------
def test_suggest_accepts_best_first(world, cands):
    fit = world[1]
    calls = []

    def ik(T, seed_q=None):
        calls.append((T.copy(), seed_q))
        return ik_map(T, prefix="arm_right_")

    cur = {f"joint{i + 1}": 0.0 for i in range(6)}
    o = P.SuggestOptions(max_joint_change_deg=1e6)
    acc, st = P.suggest(fit, ik, _always_valid, cur, n=5, options=o)
    assert [s.rank for s in acc] == [1, 2, 3, 4, 5]
    assert st["accepted"] == 5 and st["ik_calls"] == len(calls)
    assert all(s.q is not None and set(s.q) == {f"arm_right_joint{i}" for i in range(1, 7)} for s in acc)
    assert all(c[1] is cur for c in calls)                     # seed_q is current_q
    assert acc[0].T_base_link6 == pytest.approx(cands[0].T_base_link6)
    assert all(a.score >= b.score for a, b in zip(acc, acc[1:]))
    for s in acc:
        qm = P._joint_map(s.q)
        want = max(abs(qm[i + 1] - 0.0) for i in range(6))
        assert s.gains["max_joint_change_deg"] == pytest.approx(math.degrees(want))
    # accepted suggestions are mutually distinct
    for i in range(len(acc)):
        for j in range(i):
            assert not P._too_close(fit, acc[i], acc[j], o)
    # candidates are not mutated by the walk
    assert all(c.q is None for c in P.generate_candidates(fit)[:5])


def test_suggest_ik_failures_and_budget(world):
    fit = world[1]
    acc, st = P.suggest(fit, lambda T, seed_q=None: None, _always_valid, {}, n=5)
    assert acc == [] and st["ik_calls"] == 150 and st["ik_failed"] == 150 and st["max_ik_calls_reached"] == 1
    o = P.SuggestOptions(max_ik_calls=7)
    acc, st = P.suggest(fit, lambda T, seed_q=None: None, _always_valid, {}, n=5, options=o)
    assert st["ik_calls"] == 7 and st["max_ik_calls_reached"] == 1
    # an IK that raises is counted, not propagated
    def boom(T, seed_q=None):
        raise RuntimeError("ik crashed")
    acc, st = P.suggest(fit, boom, _always_valid, {}, n=3, options=o)
    assert acc == [] and st["ik_error"] == 7
    # half of the calls fail: the others are accepted until n is reached, budget not exhausted
    cnt = {"n": 0}

    def flaky(T, seed_q=None):
        cnt["n"] += 1
        return None if cnt["n"] % 2 else ik_map(T)
    acc, st = P.suggest(fit, flaky, _always_valid, {}, n=3, options=P.SuggestOptions(max_joint_change_deg=1e6))
    assert len(acc) == 3 and st["ik_failed"] >= 3 and st["max_ik_calls_reached"] == 0
    assert [s.rank for s in acc] == [1, 2, 3]


def test_suggest_invalid_reasons(world):
    fit = world[1]
    o = P.SuggestOptions(max_joint_change_deg=1e6, max_ik_calls=20)
    seen = []

    def valid(q):
        seen.append(q)
        return (False, "collision with camera pole") if len(seen) % 2 else (True, "")

    acc, st = P.suggest(fit, lambda T, seed_q=None: ik_map(T), valid, {}, n=4, options=o)
    assert len(acc) == 4
    assert st["invalid"] == 4 and st["ik_calls"] == 8
    assert st["invalid: collision with camera pole"] == st["invalid"]
    acc, st = P.suggest(fit, lambda T, seed_q=None: ik_map(T), lambda q: (False, "joint limit"), {}, n=4, options=o)
    assert acc == [] and st["invalid"] == 20 and st["invalid: joint limit"] == 20 and st["max_ik_calls_reached"] == 1


def test_suggest_joint_change_limit(world):
    fit = world[1]
    cur = {f"joint{i + 1}": 0.0 for i in range(6)}
    n_calls = {"n": 0}

    def ik(T, seed_q=None):
        n_calls["n"] += 1
        q = ik_map(T, prefix="arm_right_")
        if n_calls["n"] <= 3:
            q["arm_right_joint4"] = math.radians(95.0)       # just above the 90 deg default
        elif n_calls["n"] == 4:
            q["arm_right_joint4"] = math.radians(85.0)       # just below
        else:
            q["arm_right_joint4"] = 0.0
        for k in q:
            if not k.endswith("joint4"):
                q[k] = 0.1 * (n_calls["n"] % 3)
        return q

    acc, st = P.suggest(fit, ik, _always_valid, cur, n=3)
    assert st["joint_change"] == 3 and len(acc) == 3
    assert acc[0].gains["max_joint_change_deg"] == pytest.approx(85.0)
    # a custom limit
    acc, st = P.suggest(fit, lambda T, seed_q=None: dict(cur, joint1=math.radians(30.0)), _always_valid, cur, n=2,
                        options=P.SuggestOptions(max_joint_change_deg=20.0, max_ik_calls=10))
    assert acc == [] and st["joint_change"] == 10


def test_suggest_rejects_repeats_of_recorded_samples(world, cands):
    ss, fit = world
    top = cands[0]
    q_top = ik_map(top.T_base_link6)
    # a recorded sample whose joints equal the top goal to within 1.5 deg (names without prefix)
    s = fit.samples[0]
    saved = dict(s.joints)
    try:
        s.joints = {k: v + math.radians(1.5) for k, v in q_top.items()}
        o = P.SuggestOptions(max_joint_change_deg=1e6)
        acc, st = P.suggest(fit, lambda T, seed_q=None: ik_map(T, prefix="arm_right_"), _always_valid, {}, n=3,
                            options=o)
        assert st["duplicate_sample"] >= 1
        assert not any(np.allclose(a.T_base_link6, top.T_base_link6) for a in acc)
        assert len(acc) == 3
        # 3 deg apart on one joint -> not a repeat
        s.joints = dict(q_top)
        s.joints["joint1"] += math.radians(3.0)
        acc, st = P.suggest(fit, lambda T, seed_q=None: ik_map(T), _always_valid, {}, n=1, options=o)
        assert st["duplicate_sample"] == 0 and np.allclose(acc[0].T_base_link6, top.T_base_link6)
    finally:
        s.joints = saved


def test_suggest_rejects_too_close_to_accepted(world, cands):
    fit = world[1]
    o = P.SuggestOptions(max_joint_change_deg=1e6)
    acc, st = P.suggest(fit, lambda T, seed_q=None: ik_map(T), _always_valid, {}, n=8, options=o)
    assert st["too_close"] > 0 and len(acc) == 8
    # without the separation rule more near-identical poses are accepted
    loose = P.SuggestOptions(max_joint_change_deg=1e6, min_separation_m=0.0, min_separation_px_frac=0.0)
    acc2, st2 = P.suggest(fit, lambda T, seed_q=None: ik_map(T), _always_valid, {}, n=8, options=loose)
    assert st2["too_close"] == 0
    assert len(acc2) == 8
