import cv2
import numpy as np
import pytest

from compal_box_perception.depth_utils import (
    build_xyzrgb_points,
    decode_compressed_depth,
    depth_scale_for_encoding,
    depth_from_plane_at_pixel,
    fit_top_plane_quad,
    get_median_depth,
    inset_segment,
    normalized_camera_rays,
    pixel_to_camera_xyz,
)


def test_depth_encoding_units_are_explicit():
    assert depth_scale_for_encoding("16UC1") == 0.001
    assert depth_scale_for_encoding("32FC1") == 1.0
    with pytest.raises(ValueError, match="Unsupported"):
        depth_scale_for_encoding("8UC1")

def test_compressed_depth_png_preserves_millimetres():
    depth = np.array([[0, 812], [817, 1000]], dtype=np.uint16)
    success, encoded = cv2.imencode(".png", depth)
    assert success
    decoded, encoding = decode_compressed_depth(
        b"\0" * 12 + encoded.tobytes(), "16UC1; compressedDepth png"
    )
    assert encoding == "16UC1"
    np.testing.assert_array_equal(decoded, depth)

def test_local_pointcloud_combines_registered_depth_and_color():
    depth = np.array([[1000, 0], [2000, 3000]], dtype=np.uint16)
    bgr = np.array(
        [[[3, 2, 1], [0, 0, 0]], [[30, 20, 10], [255, 128, 64]]],
        dtype=np.uint8,
    )
    rays = normalized_camera_rays(
        2, 2, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], (), 1
    )
    points = build_xyzrgb_points(depth, bgr, rays, scale=0.001, stride=1)
    assert points.shape == (2, 2)
    assert points["z"][0, 0] == pytest.approx(1.0)
    assert np.isnan(points["z"][0, 1])
    assert points["x"][1, 0] == pytest.approx(0.0)
    assert points["y"][1, 0] == pytest.approx(2.0)
    assert points["rgb"].view(np.uint32)[1, 0] == 0x0A141E


def test_depth_plane_recovers_top_face_without_background_plane():
    depth = np.full((100, 100), 2000, dtype=np.uint16)
    depth[25:66, 25:76] = 1000
    camera_matrix = [100.0, 0.0, 50.0, 0.0, 100.0, 50.0, 0.0, 0.0, 1.0]

    corners, normal, offset, inlier_ratio, rms = fit_top_plane_quad(
        depth,
        camera_matrix,
        (),
        scale=0.001,
        fit_roi=(20, 20, 80, 50),
        classify_roi=(20, 20, 80, 80),
        sample_stride=2,
        distance_threshold_m=0.005,
        iterations=50,
        min_inliers=100,
        min_inlier_ratio=0.30,
    )

    assert corners.shape == (4, 2)
    assert np.min(corners[:, 0]) == pytest.approx(26, abs=2)
    assert np.max(corners[:, 0]) == pytest.approx(74, abs=2)
    assert np.min(corners[:, 1]) == pytest.approx(26, abs=2)
    assert np.max(corners[:, 1]) == pytest.approx(64, abs=2)
    assert abs(normal[2]) == pytest.approx(1.0, abs=1e-3)
    assert abs(abs(offset) - 1.0) < 1e-3
    assert inlier_ratio > 0.4
    assert rms < 1e-6


def test_plane_fit_rejects_wrong_surface_orientation():
    depth = np.full((80, 80), 1000, dtype=np.uint16)
    camera_matrix = [100.0, 0.0, 40.0, 0.0, 100.0, 40.0, 0.0, 0.0, 1.0]
    with pytest.raises(ValueError, match="RANSAC"):
        fit_top_plane_quad(
            depth,
            camera_matrix,
            (),
            scale=0.001,
            fit_roi=(10, 10, 70, 50),
            classify_roi=(10, 10, 70, 70),
            sample_stride=2,
            min_inliers=100,
            expected_normal=(1.0, 0.0, 0.0),
            max_normal_deviation_deg=5.0,
        )


def test_plane_intersection_returns_boundary_depth_without_pixel_sampling():
    camera_matrix = [100.0, 0.0, 50.0, 0.0, 100.0, 50.0, 0.0, 0.0, 1.0]
    depth = depth_from_plane_at_pixel(
        80.0,
        60.0,
        camera_matrix,
        (),
        plane_normal=(0.0, 0.0, 1.0),
        plane_offset=-1.0,
    )
    assert depth == pytest.approx(1.0)


def test_median_depth_clips_at_image_boundary_without_wrapping():
    depth = np.array(
        [
            [1000, 1000, 0, 0],
            [1000, 2000, 0, 0],
            [0, 0, 9000, 9000],
            [0, 0, 9000, 9000],
        ],
        dtype=np.uint16,
    )
    result = get_median_depth(
        depth, 0, 0, radius=1, scale=0.001, min_valid_samples=3
    )
    assert result == pytest.approx(1.0)


def test_median_depth_rejects_too_few_valid_samples():
    depth = np.zeros((5, 5), dtype=np.float32)
    depth[2, 2] = 0.8
    assert (
        get_median_depth(depth, 2, 2, radius=1, min_valid_samples=2) is None
    )


def test_inset_segment_moves_both_ends_inward():
    start, end = inset_segment((0, 0), (100, 20), 0.1)
    np.testing.assert_allclose(start, (10, 2))
    np.testing.assert_allclose(end, (90, 18))


def test_pixel_to_camera_xyz_uses_camera_info_intrinsics():
    xyz = pixel_to_camera_xyz(
        651.0,
        366.0,
        0.8,
        [692.0, 0.0, 651.0, 0.0, 692.0, 366.0, 0.0, 0.0, 1.0],
    )
    np.testing.assert_allclose(xyz, (0.0, 0.0, 0.8))
