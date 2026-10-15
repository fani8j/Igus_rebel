import math
import os
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import handeye_math as hm  # noqa: E402

K = np.array([[626.0, 0, 640.0], [0, 626.0, 400.0], [0, 0, 1.0]])
D = np.array([0.02, -0.03, 0.0005, 0.0003, 0.0])
SIZE = (1280, 800)
W, H = 0.199, 0.160


def grid_obj():
    xs = np.linspace(-W / 2 + 0.02, W / 2 - 0.02, 7)
    ys = np.linspace(-H / 2 + 0.02, H / 2 - 0.02, 6)
    return np.array([[x, y, 0.0] for y in ys for x in xs])


def board_flat(x=0.514, y=-0.148, z=-0.251, flip=False):
    R = Rot.from_euler("x", math.pi).as_matrix() if flip else np.eye(3)
    return hm.make_T(R, [x, y, z])


X_TRUE = hm.T_from_xyz_rpy(0.021, -0.087, -0.040, math.radians(5), math.radians(-12), math.radians(95))


def synth_samples(B, X, noise=0.2, n=16, seed=0, outlier=True, rolls=(-40, -20, 0, 20, 40), tilts=(0, 15, 25, 35)):
    rng = np.random.default_rng(seed)
    cands = hm.orbit_candidates(B, tilts=tilts, rolls=rolls)
    rots = [c.T_base_cam[:3, :3] for c in cands]
    idx = hm.select_diverse(rots, n)
    obj = grid_obj()
    samples = []
    for k, i in enumerate(idx):
        Tc = cands[i].T_base_cam
        pix, _ = hm.project_board(Tc, B, obj, K, D)
        pix = pix + rng.normal(0, noise, pix.shape)
        if outlier and k == 3:
            pix[:8] += rng.normal(0, 15, (8, 2))
        samples.append(hm.Sample(Tc @ hm.inv_T(X), obj, pix, "s%d" % k))
    return samples


@pytest.mark.parametrize("flip", [False, True])
def test_recovers_known_X_with_noise_and_outlier(flip):
    B = board_flat(flip=flip)
    samples = synth_samples(B, X_TRUE)
    res = hm.solve_handeye(samples, K, D, T_base_board_scene=B)
    mm, dg = hm.pose_diff(res["X"], X_TRUE)
    assert mm < 1.0 and dg < 0.1, (mm, dg)
    mmb, dgb = hm.pose_diff(res["T_base_board"], B)
    assert mmb < 1.5 and dgb < 0.2, (mmb, dgb)
    assert res["rotation"]["span_deg"] > 30
    assert res["loo"] is not None
    # the outlier sample is visible in the statistics (max px) rather than silently hidden
    assert res["max_px"] > 5.0


def test_clean_synthetic_data_passes_gate():
    B = board_flat()
    res = hm.solve_handeye(synth_samples(B, X_TRUE, noise=0.2, outlier=False), K, D, T_base_board_scene=B)
    assert res["status"] == "PASS", (res["reasons"], res["rms_px"], res["loo"])
    assert res["rms_px"] < 0.5 and res["loo"]["pos_rms_mm"] < 3.0
    assert res["scene_cross_check"]["mm"] < 1.5


def test_degenerate_rotations_warn():
    """All link6 rotations about ONE axis: no observability of X, must WARN (or FAIL), never PASS."""
    B = board_flat()
    obj = grid_obj()
    samples = []
    Tc0 = hm.look_at_pose(B, B[:3, 3] + np.array([0, 0, 0.35]), B[:3, 3], 0.0)
    for a in np.linspace(-45, 45, 12):
        # rotate the camera about its own optical axis through the board centre: pure rotation about one axis
        Tc = hm.look_at_pose(B, B[:3, 3] + np.array([0, 0, 0.35]), B[:3, 3], a)
        pix, _ = hm.project_board(Tc, B, obj, K, D)
        samples.append(hm.Sample(Tc @ hm.inv_T(X_TRUE), obj, pix))
    div = hm.rotation_diversity([s.T_base_link6[:3, :3] for s in samples])
    assert div["axis_eig"][1] < 0.05
    r = hm.gate({"n_samples": 12, "rms_px": 0.2, "loo": {"pos_rms_mm": 0.5}, "rotation": div})
    assert r["status"] != "PASS"
    assert any("DEGENERATE" in x for x in r["reasons"])


def test_gate_pass_and_fail():
    ok = {"n_samples": 14, "rms_px": 0.4, "loo": {"pos_rms_mm": 1.0}, "rotation": {"span_deg": 60, "axis_eig": [0.5, 0.3, 0.2]}}
    assert hm.gate(dict(ok))["status"] == "PASS"
    assert hm.gate(dict(ok, n_samples=8))["status"] == "WARN"
    # absolute reprojection is dominated by the arm accuracy: informational unless huge
    assert hm.gate(dict(ok, rms_px=6.6))["status"] == "PASS"
    assert hm.gate(dict(ok, rms_px=13.0))["status"] == "FAIL"
    assert hm.gate(dict(ok, board_consistency={"pos_rms_mm": 4.5, "rot_rms_deg": 0.3}))["status"] == "PASS"
    assert hm.gate(dict(ok, board_consistency={"pos_rms_mm": 4.5, "rot_rms_deg": 11.5}))["status"] == "FAIL"
    assert hm.gate(dict(ok, scene_cross_check={"mm": 3.0, "deg": 3.0}))["status"] == "PASS"
    assert hm.gate(dict(ok, scene_cross_check={"mm": 19.9, "deg": 45.0}))["status"] == "FAIL"
    assert hm.gate(dict(ok, method_agreement={"max_mm": 8.0, "max_deg": 1.0}))["status"] == "WARN"
    assert hm.gate(dict(ok, rotation={"span_deg": 20, "axis_eig": [0.5, 0.3, 0.2]}))["status"] == "WARN"


@pytest.mark.parametrize("flip", [False, True])
def test_orbit_geometry(flip):
    B = board_flat(flip=flip)
    cands = hm.orbit_candidates(B)
    assert len(cands) == (1 + 3 * 8) * 5 * 3
    centre = B[:3, 3]
    for c in cands:
        # board centre sits on the optical axis at the requested distance
        pc = hm.inv_T(c.T_base_cam) @ np.append(centre, 1.0)
        assert abs(pc[0]) < 1e-9 and abs(pc[1]) < 1e-9
        assert pc[2] == pytest.approx(c.dist, abs=1e-9)
        assert c.dist >= 0.25
        # tilt from the board normal as requested, camera above the board
        u = hm.board_up_normal(B)
        v = c.T_base_cam[:3, 3] - centre
        assert math.degrees(math.acos(np.dot(v / np.linalg.norm(v), u))) == pytest.approx(c.tilt, abs=1e-6)
        assert np.dot(v, np.array([0, 0, 1.0])) > 0
        R = c.T_base_cam[:3, :3]
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-9) and np.linalg.det(R) == pytest.approx(1.0)
    assert {c.dist for c in cands} == {0.30, 0.36, 0.42}
    assert {c.roll for c in cands} == {-40, -20, 0, 20, 40}


def test_corners_in_view():
    B = board_flat()
    Tc = hm.look_at_pose(B, B[:3, 3] + np.array([0, 0, 0.35]), B[:3, 3])
    assert hm.corners_in_view(Tc, B, W, H, K, D, SIZE)
    far = hm.look_at_pose(B, B[:3, 3] + np.array([0.0, 0, 0.12]), B[:3, 3])
    assert not hm.corners_in_view(far, B, W, H, K, D, SIZE)
    behind = hm.look_at_pose(B, B[:3, 3] + np.array([0, 0, 0.35]), B[:3, 3] + np.array([0.0, 0, 1.0]))
    assert not hm.corners_in_view(behind, B, W, H, K, D, SIZE)


def corners_base(B):
    return hm.transform_points(B, hm.board_corners_obj(W, H))


def test_slab_horizontal_top_and_clip():
    B = board_flat()
    # a tilted board: highest corner defines the top
    Bt = hm.make_T(Rot.from_euler("x", math.radians(5)).as_matrix() @ B[:3, :3], B[:3, 3])
    for T in (B, Bt):
        c = corners_base(T)
        slab = hm.build_slab(c)
        assert slab["top"] == pytest.approx(c[:, 2].max() + 0.05)
        cx, cy, cz = slab["center"]
        sx, sy, sz = slab["size"]
        assert cz + sz / 2 == pytest.approx(slab["top"])
        x0, x1, y0, y1 = cx - sx / 2, cx + sx / 2, cy - sy / 2, cy + sy / 2
        assert hm.rect_axis_distance(x0, x1, y0, y1) >= 0.15 - 1e-9
        assert x0 <= c[:, 0].min() and x1 >= c[:, 0].max() and y0 <= c[:, 1].min() and y1 >= c[:, 1].max()
        assert sx <= 1.2 + 1e-9 and sy <= 1.2 + 1e-9
    # the measured cell: the 1.2 m square would contain the base axis, so it must be clipped
    assert hm.rect_axis_distance(0.514 - 0.6, 0.514 + 0.6, -0.148 - 0.6, -0.148 + 0.6) == 0.0
    s = hm.build_slab(corners_base(B))
    assert s["size"][0] * s["size"][1] < 1.2 * 1.2


def test_slab_unclipped_when_far():
    B = board_flat(x=1.2, y=0.9)
    s = hm.build_slab(corners_base(B))
    assert s["size"][:2] == pytest.approx([1.2, 1.2])


def test_slab_fails_when_board_too_near_base():
    B = board_flat(x=0.05, y=0.0)
    with pytest.raises(ValueError):
        hm.build_slab(corners_base(B))


def test_board_stand_and_pad_inflation():
    c = corners_base(board_flat())
    slab = hm.build_slab(c)
    stand = hm.build_board_stand(c, slab["top"])
    assert stand["size"][0] == pytest.approx(c[:, 0].ptp() + 0.1) and stand["size"][1] == pytest.approx(c[:, 1].ptp() + 0.1)
    assert stand["center"][2] - stand["size"][2] / 2 == pytest.approx(slab["top"])
    pad = hm.inflate_box([1, 2, 3], [0.03, 0.03, 1.0], 0.05)
    assert pad["size"] == pytest.approx([0.13, 0.13, 1.10]) and pad["center"] == [1.0, 2.0, 3.0]
    head = hm.inflate_box([0, 0, 0], [0.1, 0.1, 0.1], 0.05)
    assert head["size"] == pytest.approx([0.2, 0.2, 0.2])


def test_nominal_mount_is_proper_rotation_and_matches_mesh():
    X = hm.nominal_T_link6_optical()
    assert np.allclose(X[:3, :3].T @ X[:3, :3], np.eye(3), atol=1e-9) and np.linalg.det(X[:3, :3]) == pytest.approx(1.0)
    mesh = "/home/farhan/igus_rebel_ros2/src/igus_rebel_description/meshes/knife_c/visual/gemini_305_body.stl"
    if not os.path.exists(mesh):
        pytest.skip("mesh not available")
    pytest.importorskip("trimesh")
    c, n = hm.derive_nominal_from_mesh(mesh)
    assert np.allclose(c, hm.NOMINAL_OPTICAL_CENTRE_TOOL0, atol=1e-4)
    assert np.allclose(n, hm.NOMINAL_OPTICAL_Z_TOOL0, atol=2e-3)


def test_plan_sequence_with_fake_ik_enforces_limits():
    B = board_flat()
    cands = hm.orbit_candidates(B, tilts=(0, 15), n_az=4, rolls=(-20, 0, 20), dists=(0.3, 0.4))
    rng = np.random.default_rng(1)
    q0 = np.zeros(6)
    for c in cands:  # fake IK: joints are a smooth function of position, joint6 follows the roll
        c.q = np.array([c.T_base_cam[0, 3], c.T_base_cam[1, 3], c.dist, c.tilt / 90, 0.0, math.radians(c.roll)])

    def ik(T6, seed):
        Tcam = T6 @ hm.nominal_T_link6_optical()
        # recover roll from the image-x axis relative to the board x axis
        ang = math.degrees(math.atan2(np.dot(Tcam[:3, 0], B[:3, 1]), np.dot(Tcam[:3, 0], B[:3, 0])))
        return np.array([Tcam[0, 3], Tcam[1, 3], 0.3, 0.0, 0.0, math.radians(ang)])

    seq, reasons = hm.plan_sequence(cands, q0, ik, hm.nominal_T_link6_optical(), 8, max_step_deg=90.0)
    assert 1 <= len(seq) <= 8
    prev = q0
    for c in seq:
        assert np.max(np.abs(c.q - prev)) <= math.radians(90) + 1e-9
        assert abs(c.q[5] - q0[5]) <= 3.14
        prev = c.q
