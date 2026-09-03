import inspect

import cv2
import numpy as np

import detect_tape_seam


def test_estimate_auto_roi_uses_position_prior_api():
    parameters = inspect.signature(detect_tape_seam.estimate_auto_roi).parameters

    assert "search_roi" not in parameters
    assert "expected_center_ratio" in parameters
    assert "max_center_dist_ratio" in parameters


def test_position_prior_selects_the_expected_cardboard_candidate():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    cardboard = (100, 140, 180)
    cv2.rectangle(image, (310, 70), (420, 180), cardboard, -1)
    cv2.rectangle(image, (100, 260), (300, 450), cardboard, -1)

    top_roi, top_debug = detect_tape_seam.estimate_auto_roi(
        image, expected_center_ratio=(0.57, 0.26), max_center_dist_ratio=0.20
    )
    bottom_roi, bottom_debug = detect_tape_seam.estimate_auto_roi(
        image, expected_center_ratio=(0.31, 0.74), max_center_dist_ratio=0.20
    )

    assert top_roi[1] < 200
    assert bottom_roi[1] > 200
    assert top_debug["used_fallback"] is False
    assert bottom_debug["used_fallback"] is False
