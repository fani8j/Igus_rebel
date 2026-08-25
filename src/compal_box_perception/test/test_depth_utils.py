import cv2
import numpy as np
import pytest

from compal_box_perception.depth_utils import (
    build_xyzrgb_points,
    decode_compressed_depth,
    depth_scale_for_encoding,
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
