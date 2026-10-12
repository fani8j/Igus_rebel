"""MotionGate: capture once the scene changed and then stayed still (no ROS)."""
import numpy as np

from box_seam_detection.live.auto_capture import MotionGate, small_gray


def _scene(box_x):
    img = np.full((1080, 1920, 3), 120, np.uint8)
    img[500:700, box_x:box_x + 300] = (60, 120, 180)     # the box
    return img


def test_first_capture_after_stillness_then_requires_change():
    g = MotionGate(still_s=1.5, min_interval_s=0.0)
    f = small_gray(_scene(800))
    t = 0.0
    fired = []
    for _ in range(10):                     # 2.5 s of a still scene at 4 Hz
        fired.append(g.update(f, t)); t += 0.25
    assert any(fired) and fired.index(True) >= 6      # not before 1.5 s of stillness
    g.captured(f, t)
    for _ in range(20):                     # same scene: never again
        assert not g.update(f, t); t += 0.25
    assert g.state == "waiting"


def test_moving_box_resets_then_new_position_is_captured():
    g = MotionGate(still_s=1.0, min_interval_s=0.0)
    t = 0.0
    f0 = small_gray(_scene(800))
    for _ in range(8):
        if g.update(f0, t):
            g.captured(f0, t)
        t += 0.25
    assert g.ref is not None
    for x in range(820, 1100, 40):          # box being moved: no capture while moving
        assert not g.update(small_gray(_scene(x)), t); t += 0.25
    f1 = small_gray(_scene(1100))
    got = False
    for _ in range(8):
        got = got or g.update(f1, t); t += 0.25
    assert got


def test_noise_does_not_count_as_motion():
    rng = np.random.default_rng(0)
    g = MotionGate(still_s=0.5, min_interval_s=0.0)
    base = _scene(800).astype(np.int16)
    t, fired = 0.0, False
    for _ in range(8):
        noisy = np.clip(base + rng.normal(0, 3, base.shape), 0, 255).astype(np.uint8)
        fired = fired or g.update(small_gray(noisy), t); t += 0.25
    assert fired
