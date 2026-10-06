"""draw_overlay(scale=...): default unchanged; half-scale overlay lands on the same geometry."""

import numpy as np

from side_tape_ros import core_path  # noqa: F401
from precision_box_detector.side_tape.config import load_config
from precision_box_detector.side_tape.overlay import draw_overlay
from precision_box_detector.side_tape.types import Observation, TrackState


def _obs(line):
    return Observation(stamp_s=0.0, frame_seq=0, state=TrackState.LOCKED, line=np.asarray(line, float),
                       conf=0.9, e_lat_mm=0.1, e_yaw_deg=0.2, proc_ms=1.0)


def test_scale_default_is_unchanged_and_half_scale_matches_geometry():
    cfg = load_config(core_path.DEFAULT_CONFIG)
    full = np.zeros((cfg.camera.height, cfg.camera.width, 3), np.uint8)
    half = np.zeros((cfg.camera.height // 2, cfg.camera.width // 2, 3), np.uint8)
    # horizontal edge y = 400 in full-size pixels:  0*x + 1*y - 400 = 0
    obs = _obs((0.0, 1.0, -400.0))
    a = draw_overlay(full, obs, cfg)
    b = draw_overlay(full, obs, cfg, scale=1.0)
    assert np.array_equal(a, b)
    s = draw_overlay(half, obs, cfg, scale=0.5)
    assert s.shape == half.shape and s.dtype == np.uint8
    lit_rows = np.where((s[:, 1000 // 2 - 3:1000 // 2 + 3] != 0).any(axis=(1, 2)))[0]
    lit_rows = lit_rows[(lit_rows > 120) & (lit_rows < 300)]
    assert lit_rows.size and abs(lit_rows.mean() - 200.0) < 3.0  # y=400 full-size -> row 200 at half size
