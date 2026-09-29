"""Offline checks for calibration input, save gates and config persistence."""

from datetime import datetime, timezone
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from side_tape_ros import core_path  # noqa: E402,F401 (register core import path)
from side_tape_ros.calib_jacobian import (  # noqa: E402
    _decode_compressed_gray, _jog_safety_reason, _save_rejection_reasons,
    _should_save, _update_mount_yaml,
)
from precision_box_detector.side_tape.config import load_config  # noqa: E402


def test_fit_gate_checks_both_columns_and_thresholds():
    good = np.array([[0.5, 0.0], [0.0, -0.5]])
    assert _save_rejection_reasons(good, 1.0, 1.0) == []
    assert _should_save(good, 1.0, 1.0, False)

    small = np.array([[0.49, 0.0], [0.0, -0.49]])
    reasons = _save_rejection_reasons(small, 1.01, 1.02)
    assert len(reasons) == 4
    assert any("cut RMS" in reason for reason in reasons)
    assert any("lateral RMS" in reason for reason in reasons)
    assert any("cut |J|" in reason for reason in reasons)
    assert any("lateral |J|" in reason for reason in reasons)
    assert not _should_save(small, 1.01, 1.02, False)
    assert _should_save(small, 1.01, 1.02, True)
    assert not _should_save(good, 1.01, 0.0, False)
    assert not _should_save(good, 0.0, 1.01, False)
    assert not _should_save(np.array([[np.nan, 0.0], [0.0, -1.0]]), 0.0, 0.0, True)
    assert not _should_save(good, float("inf"), 0.0, True)


def test_jog_safety_reason_on_ownership_and_status_age():
    assert "ownership is unknown" in _jog_safety_reason(None, 0, 0.0)
    assert "trajectory" in _jog_safety_reason([2], 0, 0.0)
    assert "status" in _jog_safety_reason([], 0, None)
    assert "status" in _jog_safety_reason([], 0, 1.001)
    assert _jog_safety_reason([], 0, 1.0) is None
    assert "HALT" in _jog_safety_reason([], 2, 0.1)


def _config_file(tmp_path):
    path = tmp_path / "knife.yaml"
    data = {
        "camera": {"image_topic": "/test_camera", "width": 1280, "height": 720,
                   "rotate_180": True},
        "mount": {"mount_id": "test_mount", "calibrated": False, "tip_uv": [600, 300],
                  "J": None},
        "target": {"offset_mm": 3.0, "edge_polarity": -1},
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path, data


def test_backup_created_with_utc_timestamp_before_rewrite(tmp_path, capsys):
    path, _ = _config_file(tmp_path)
    before = path.read_bytes()
    instant = datetime(2026, 9, 28, 12, 34, 56, 123456, tzinfo=timezone.utc)
    backup = _update_mount_yaml(str(path), np.diag([2.0, -3.0]), True, 0.25, instant)
    assert backup.name == "knife.yaml.bak-20260928T123456123456Z"
    assert backup.read_bytes() == before
    assert str(backup) in capsys.readouterr().out
    assert path.read_bytes() != before


def test_yaml_comment_round_trips_through_core_loader(tmp_path):
    path, original = _config_file(tmp_path)
    instant = datetime(2026, 9, 28, 12, 34, 56, tzinfo=timezone.utc)
    _update_mount_yaml(str(path), np.array([[2.0, -0.1], [0.2, -3.0]]), True, 0.347, instant)
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# J calibrated 2026-09-28T12:34:56+00:00 residual 0.347 px\n")
    saved = yaml.safe_load(text)
    assert list(saved) == list(original)
    assert saved["camera"] == original["camera"]
    assert saved["target"] == original["target"]
    assert saved["mount"] == {**original["mount"], "J": [[2.0, -0.1], [0.2, -3.0]],
                              "calibrated": True}
    loaded = load_config(path)
    assert loaded.mount.J == saved["mount"]["J"]
    assert loaded.mount.calibrated


def test_compressed_decode_and_derotate():
    image = np.zeros((18, 24, 3), dtype=np.uint8)
    image[1:5, 2:7] = (210, 20, 70)
    image[10:16, 16:22] = (30, 220, 10)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 100])
    assert ok
    gray = _decode_compressed_gray(encoded.tobytes(), False)
    assert gray.shape == image.shape[:2]
    np.testing.assert_array_equal(_decode_compressed_gray(encoded.tobytes(), True), gray[::-1, ::-1])
    with pytest.raises(ValueError, match="decode"):
        _decode_compressed_gray(b"not a jpeg", True)


def test_fit_jacobian_absorbs_cross_motion_and_bias():
    """Regression (real arm 2026-09-29): lateral jogs also moved ~0.5 mm along the cut and the
    reference pixel was off by a few px; the per-axis fit gave 2.5 px residuals."""
    import numpy as np
    from side_tape_ros.calib_jacobian import fit_jacobian
    J_true = np.array([[-6.1, -0.9], [1.1, -7.6]])
    bias = np.array([4.0, -1.5])
    moves = [(0, 0), (2.5, 0.1), (-3.0, -0.2), (0.4, 2.6), (-0.5, -2.7)]
    samples = [(c, l, *(J_true @ [c, l] + bias)) for c, l in moves]
    J, rms = fit_jacobian(samples)
    np.testing.assert_allclose(J, J_true, atol=1e-9)
    assert rms < 1e-9
