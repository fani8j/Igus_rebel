import numpy as np
import pytest

import synth
from side_cut_vision.camera import CameraError, CameraModel


def test_real_coefficients_are_eight_and_used_as_rational():
    cam = synth.load_camera()
    assert len(cam.D) == 8
    assert cam.to_dict()["distortion_model"] == "rational_polynomial"
    assert (cam.width, cam.height) == (1280, 800)


def test_rational_roundtrip_is_exact_and_plumb_bob_is_not():
    cam = synth.load_camera()
    u, v = np.meshgrid(np.arange(0, 1280, 160), np.arange(0, 800, 100))
    uv = np.stack([u.ravel(), v.ravel()], 1).astype(float)
    xy = cam.pixel_rays(uv)
    P = np.concatenate([xy, np.ones((len(xy), 1))], 1) * 0.3
    back = cam.project(P)
    assert np.abs(back - uv).max() < 0.01          # undistort -> project round trip with all 8 coeffs
    pb = cam.plumb_bob_only()                       # what a plumb_bob reader (5 coefficients) would use
    wrong = pb.project(P)
    err = np.linalg.norm(wrong - uv, axis=1)
    assert err.max() > 10.0, "dropping k4..k6 must matter on the real calibration"


def test_deproject_project_consistency_and_depth_scale():
    cam = synth.load_camera()
    depth = np.full((800, 1280), 300, np.uint16)    # 300 mm everywhere
    xyz, valid = cam.deproject(depth, stride=4)
    assert valid.all() and xyz.shape == (200, 320, 3)
    assert np.allclose(xyz[..., 2], 0.3, atol=1e-6)
    uv = cam.project(xyz.reshape(-1, 3))
    g_u, g_v = np.meshgrid(np.arange(0, 1280, 4), np.arange(0, 800, 4))
    assert np.abs(uv[:, 0] - g_u.ravel()).max() < 0.05
    assert np.abs(uv[:, 1] - g_v.ravel()).max() < 0.05


def test_zero_depth_invalid_and_points_behind_camera_nan():
    cam = synth.load_camera()
    depth = np.zeros((800, 1280), np.uint16)
    _, valid = cam.deproject(depth, 2)
    assert not valid.any()
    uv = cam.project(np.array([[0.0, 0.0, -0.1], [0.0, 0.0, 0.2]]))
    assert np.isnan(uv[0]).all() and np.isfinite(uv[1]).all()


def test_bad_input_raises_cameraerror():
    cam = synth.load_camera()
    with pytest.raises(CameraError):
        cam.deproject(np.zeros((400, 640), np.uint16))
    with pytest.raises(CameraError):
        CameraModel(np.eye(3) * np.nan, [0] * 8, 10, 10)
    with pytest.raises(CameraError):
        CameraModel(np.eye(3), [0] * 7, 10, 10)
    with pytest.raises(CameraError):
        CameraModel.from_dict({"K": np.eye(3).tolist()})
