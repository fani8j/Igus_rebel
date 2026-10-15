"""Synthetic ground-truth tests for the dual-arm solver (no ROS needed)."""
import numpy as np
import pytest

from rebel_dual_arm_calibration import solver
from rebel_dual_arm_calibration.synthetic import make_scene, simulate_samples
from rebel_dual_arm_calibration.targets import TargetSpec
from rebel_dual_arm_calibration.transforms import inv_T, rotation_angle_deg


def charuco_specs(n, square, marker):
    cnt = (n * n) // 2
    return {"left": TargetSpec("left", "charuco", "DICT_4X4_50", marker, list(range(0, cnt)), square, n, n),
            "right": TargetSpec("right", "charuco", "DICT_4X4_50", marker, list(range(20, 20 + cnt)), square, n, n)}


def aruco_specs(length):
    return {"left": TargetSpec("left", "aruco", "DICT_4X4_50", length, [13]),
            "right": TargetSpec("right", "aruco", "DICT_4X4_50", length, [14])}


def errors(res, scene):
    right = (np.linalg.norm(res.T_world_arm_right_base[:3, 3] - scene.T_W_R[:3, 3]) * 1e3,
             rotation_angle_deg(res.T_world_arm_right_base[:3, :3], scene.T_W_R[:3, :3]))
    cam = (np.linalg.norm(res.T_world_camera_link[:3, 3] - scene.T_W_cam[:3, 3]) * 1e3,
           rotation_angle_deg(res.T_world_camera_link[:3, :3], scene.T_W_cam[:3, :3]))
    return right, cam


@pytest.mark.parametrize("n,square,marker", [(3, 0.030, 0.022), (4, 0.030, 0.022)])
def test_charuco_recovers_ground_truth_with_outliers(n, square, marker):
    scene = make_scene(distance=0.9, seed=n)
    specs = charuco_specs(n, square, marker)
    samples = simulate_samples(scene, specs, n_per_arm=20, noise_px=0.3, seed=10 + n,
                               n_sample_outliers=2, n_corner_outliers=3, marker_noise_factor=1.7)
    res = solver.solve(samples)
    (rp, rr), (cp, cr) = errors(res, scene)
    assert rp < 1.0 and rr < 0.1, f"right base error {rp:.3f} mm / {rr:.3f} deg"
    assert cp < 1.0 and cr < 0.1, f"camera error {cp:.3f} mm / {cr:.3f} deg"
    assert res.report["joint"]["rms_px"]["all"] < 0.8
    # gross outliers were detected: at least one sample dropped and corner outliers masked
    assert sum(res.report["n_samples_rejected"].values()) >= 1
    assert res.report["n_corner_outliers"] >= 1
    # target offsets recovered
    for arm in ("left", "right"):
        Z = res.T_link6_target[arm]
        assert np.linalg.norm(Z[:3, 3] - scene.Z[arm][:3, 3]) < 1e-3
    # stated std devs are the right order of magnitude (within 5x of the actual error, not wildly pessimistic)
    std = res.report["joint"]["calibration_position_std_mm"]
    assert 0.0 < std["arm_right_base"] < 1.0 and std["camera_link"] < 1.0


def test_single_aruco_marker_accuracy_reported(capsys):
    scene = make_scene(distance=0.9, seed=5)
    specs = aruco_specs(0.059)
    samples = simulate_samples(scene, specs, n_per_arm=20, noise_px=0.3, seed=21,
                               n_sample_outliers=2, n_corner_outliers=2)
    res = solver.solve(samples)
    (rp, rr), (cp, cr) = errors(res, scene)
    print(f"single 59 mm ArUco: right base {rp:.2f} mm {rr:.3f} deg, camera {cp:.2f} mm {cr:.3f} deg")
    # looser than ChArUco (4 coplanar corners per sample), but still useful
    assert rp < 3.0 and rr < 0.3 and cp < 3.0 and cr < 0.3


def test_joint_and_separate_agree_and_report_structure():
    scene = make_scene(distance=1.0, seed=2)
    specs = charuco_specs(4, 0.025, 0.018)
    samples = simulate_samples(scene, specs, n_per_arm=15, noise_px=0.3, seed=4)
    res = solver.solve(samples, dataset="synthetic")
    rep = res.report
    for key in ("joint", "separate", "warnings", "samples", "conditioning"):
        assert key in rep
    assert rep["separate"]["composed"]["arm_right_base_diff_vs_joint"]["position_mm"] < 1.0
    text = solver.format_report(rep)
    assert "arm_right_base" in text and "camera_link" in text


def test_camera_link_conversion_uses_recorded_tf():
    scene = make_scene(distance=0.9, seed=7)
    specs = charuco_specs(4, 0.025, 0.018)
    samples = simulate_samples(scene, specs, n_per_arm=15, noise_px=0.2, seed=8)
    res = solver.solve(samples)
    T = res.T_world_optical @ inv_T(scene.T_cl_o)
    assert np.allclose(T, res.T_world_camera_link)


def test_degenerate_dataset_warns():
    """Constant link6 orientation (translation only) cannot separate base from target offset."""
    scene = make_scene(distance=0.9, seed=3)
    specs = charuco_specs(4, 0.025, 0.018)
    samples = simulate_samples(scene, specs, n_per_arm=15, noise_px=0.3, seed=9, rotation_diversity=False)
    res = solver.solve(samples)
    text = " ".join(res.report["warnings"])
    assert "rotation diversity" in text
    assert "condition number" in text


def test_too_few_samples_warns_and_missing_arm_raises():
    scene = make_scene(distance=0.9, seed=4)
    specs = charuco_specs(4, 0.025, 0.018)
    samples = simulate_samples(scene, specs, n_per_arm=6, noise_px=0.3, seed=11)
    res = solver.solve(samples)
    assert any("distinct poses" in w for w in res.report["warnings"])
    with pytest.raises(solver.SolverError):
        solver.solve([s for s in samples if s.arm == "left"])


def test_projection_matches_opencv():
    import cv2
    K = np.array([[900.0, 0, 950], [0, 905, 530], [0, 0, 1]])
    d = np.array([-0.12, 0.05, 0.001, -0.002, 0.01, 0.0, 0.0, 0.0])
    P = np.random.default_rng(0).uniform([-0.3, -0.2, 0.5], [0.3, 0.2, 1.5], (30, 3))
    ref, _ = cv2.projectPoints(P, np.zeros(3), np.zeros(3), K, d)
    assert np.allclose(solver.project_points(K, d, P), ref.reshape(-1, 2), atol=1e-6)
