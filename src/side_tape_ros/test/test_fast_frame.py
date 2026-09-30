"""Crop-and-rotate equivalence: the ROI canvas must equal derotate_image inside the ROI."""

import numpy as np
import pytest

from side_tape_ros import core_path  # noqa: F401
from side_tape_ros.fast_frame import RoiCanvas, raw_window
from precision_box_detector.side_tape.backends.classical import ClassicalBackend
from precision_box_detector.side_tape.config import DetectorCfg
from precision_box_detector.side_tape.geometry import derotate_image, roi_around

W, H = 1280, 720


def _frame(seed=0):
    return np.random.default_rng(seed).integers(0, 256, (H, W, 3), dtype=np.uint8)


@pytest.mark.parametrize("rot", [True, False])
@pytest.mark.parametrize("center", [(618.4, 125.0), (5.0, 3.0), (1275.0, 715.0), (640.0, 360.0), (100.5, 700.2)])
def test_canvas_matches_derotate_inside_roi(rot, center):
    raw = _frame()
    roi = roi_around(center, 150, 110, W, H)
    x0, y0, x1, y1 = roi
    full = derotate_image(raw, rot)
    canvas = RoiCanvas(W, H, rot).fill(raw, roi)
    assert canvas.shape == full.shape and canvas.dtype == full.dtype
    assert np.array_equal(canvas[y0:y1, x0:x1], full[y0:y1, x0:x1])


def test_canvas_reuse_moves_window():
    c = RoiCanvas(W, H, True)
    r1, r2 = _frame(1), _frame(2)
    roi1, roi2 = roi_around((300, 200), 100, 80, W, H), roi_around((900, 500), 100, 80, W, H)
    c.fill(r1, roi1)
    out = c.fill(r2, roi2)
    x0, y0, x1, y1 = roi2
    assert np.array_equal(out[y0:y1, x0:x1], derotate_image(r2, True)[y0:y1, x0:x1])


def test_raw_window_is_self_inverse_mapping():
    roi = (10, 20, 111, 91)
    rx0, ry0, rx1, ry1 = raw_window(roi, W, H, True)
    assert (rx1 - rx0, ry1 - ry0) == (101, 71)
    assert raw_window((rx0, ry0, rx1, ry1), W, H, True) == roi
    assert raw_window(roi, W, H, False) == roi


def test_shape_mismatch_rejected():
    with pytest.raises(ValueError):
        RoiCanvas(W, H, True).fill(np.zeros((H, W), np.uint8), (0, 0, 50, 50))


@pytest.mark.parametrize("rot", [True, False])
def test_backend_output_identical_full_vs_canvas(rot):
    # Synthetic bright/dark step edge, roughly along the cut axis, with noise.
    rng = np.random.default_rng(3)
    scene = np.full((H, W, 3), 60, np.uint8)
    scene[:, :] = 60
    yy = np.arange(H)[:, None]
    scene[(yy > 0.05 * np.arange(W)[None, :] + 140)] = 200
    scene = np.clip(scene + rng.normal(0, 3, scene.shape), 0, 255).astype(np.uint8)
    raw = derotate_image(scene, rot)  # so the de-rotated frame is `scene`
    J = np.array([[-8.0, 1.4], [0.24, -7.4]])
    roi = roi_around((618.0, 125.0), 150, 110, W, H)
    backend = ClassicalBackend(DetectorCfg(), None)
    a = backend.detect(derotate_image(raw, rot), roi, None, J)
    b = backend.detect(RoiCanvas(W, H, rot).fill(raw, roi), roi, None, J)
    assert len(a) == len(b) and len(a) > 0
    for ca, cb in zip(a, b):
        assert np.array_equal(ca.line, cb.line) and ca.polarity == cb.polarity
