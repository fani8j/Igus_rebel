"""Headless calibration must retain the rest of the mount config and rasterize the mask."""

import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from side_tape_ros.calib_tip_mask import main  # noqa: E402


def test_headless_tip_mask_updates_only_mount_fields(tmp_path):
    config = tmp_path / "knife.yaml"
    original = {
        "camera": {"width": 80, "height": 60, "rotate_180": True,
                   "image_topic": "/test_camera"},
        "mount": {"mount_id": "test_mount", "calibrated": True,
                  "tip_uv": [5, 7], "J": [[-10, 0], [0, -10]]},
        "target": {"offset_mm": 3, "edge_polarity": -1},
    }
    config.write_text(yaml.safe_dump(original), encoding="utf-8")
    image = tmp_path / "scene.png"
    assert cv2.imwrite(str(image), np.zeros((60, 80, 3), np.uint8))

    main(["--config", str(config), "--image", str(image), "--no-gui",
          "--tip", "40", "25", "--mask-poly", "30,20;50,20;50,40;30,40"])

    saved = yaml.safe_load(config.read_text(encoding="utf-8"))
    assert saved["camera"] == original["camera"]
    assert saved["target"] == original["target"]
    assert saved["mount"] == {**original["mount"], "tip_uv": [40.0, 25.0],
                              "occluder_mask_path": "test_mount_occluder.png"}
    mask = cv2.imread(str(tmp_path / saved["mount"]["occluder_mask_path"]), cv2.IMREAD_GRAYSCALE)
    assert mask.shape == (60, 80)
    assert mask[30, 40] == 255
    assert mask[10, 10] == 0
