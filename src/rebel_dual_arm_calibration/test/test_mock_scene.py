"""MOCK ONLY helpers: the rendered scene of mock_scene_camera.py must be detected by the real detector at the pose
the hidden truth gives (no ROS needed)."""
import cv2
import numpy as np

from rebel_dual_arm_calibration import mock_common as M
from rebel_dual_arm_calibration.targets import TargetDetector, default_targets_path, load_targets_all
from rebel_dual_arm_calibration.transforms import R_to_rotvec, inv_T


def test_rendered_boards_are_detected_at_the_true_pose():
    truth = M.load_truth()
    specs, table = load_targets_all(default_targets_path())
    det = TargetDetector({**specs, "table": table})
    # a left-arm board 0.6 m in front of the camera, facing it with a modest tilt, plus the table board
    T_o_t = np.eye(4)
    T_o_t[:3, :3] = cv2.Rodrigues(np.array([0.3, -0.4, 0.05]))[0]
    T_o_t[:3, 3] = [-0.05, 0.02, 0.60]
    T_w_t = truth.T_w_opt @ T_o_t
    assert M.arm_board_visibility(truth, specs["left"], T_w_t)[0]
    img = M.SceneRenderer(truth).render({"left": T_w_t, "table": truth.T_w_board})
    found = det.detect(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    assert found["left"].n_markers == 4 and "right" not in found and found["table"].n_markers == 10
    for name, T_truth_in_world in (("left", T_w_t), ("table", truth.T_w_board)):
        ids, obj, pix = found[name].arrays()
        T = inv_T(truth.T_w_opt) @ T_truth_in_world
        proj, _ = cv2.projectPoints(obj, R_to_rotvec(T[:3, :3]), T[:3, 3], truth.K, truth.D)
        err = np.linalg.norm(proj.reshape(-1, 2) - pix, axis=1)
        assert err.mean() < 0.8 and err.max() < 2.5, (name, err.mean(), err.max())


def test_truth_file_is_consistent_with_the_nominal_seed():
    truth = M.load_truth()
    # the right base differs from the nominal seed (y = -0.4474) by a few mm and a fraction of a degree
    assert 0.002 < np.linalg.norm(truth.T_w_rb[:3, 3] - np.array([0.0, -0.4474, 0.0])) < 0.01
    assert truth.size == (1920, 1080) and abs(truth.K[0, 2] - 957.3) < 1e-9
