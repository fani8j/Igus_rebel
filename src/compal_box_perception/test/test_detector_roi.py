import cv2
import numpy as np

import detect_tape_seam


def test_search_roi_prevents_larger_cardboard_distractor_selection():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    cardboard = (50, 100, 170)
    cv2.rectangle(image, (310, 70), (420, 180), cardboard, -1)
    cv2.rectangle(image, (100, 260), (300, 450), cardboard, -1)

    full_roi, _ = detect_tape_seam.estimate_auto_roi(image)
    constrained_roi, debug = detect_tape_seam.estimate_auto_roi(
        image, search_roi=(260, 20, 470, 230)
    )

    assert full_roi[1] > 200
    assert 260 <= constrained_roi[0] < constrained_roi[2] <= 470
    assert 20 <= constrained_roi[1] < constrained_roi[3] <= 230
    assert debug["search_roi"] == (260, 20, 470, 230)
    chosen_x, chosen_y, _, _ = debug["chosen"]["bbox"]
    assert chosen_x >= 300
    assert chosen_y < 200


def test_concave_brightness_notch_is_removed_before_quad_refinement():
    mask = np.zeros((160, 220), dtype=np.uint8)
    pentagon = np.array(
        [[20, 20], [200, 20], [190, 140], [110, 85], [25, 140]],
        dtype=np.int32,
    )
    cv2.fillPoly(mask, [pentagon], 255)

    corners, _ = detect_tape_seam.find_quad_corners(mask)

    assert corners.shape == (4, 2)
    assert cv2.isContourConvex(corners.reshape(-1, 1, 2).astype(np.int32))
    assert not np.any(np.all(np.isclose(corners, (110, 85)), axis=1))
