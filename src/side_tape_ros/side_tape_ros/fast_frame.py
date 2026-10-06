"""Opt-in fast path: rotate only the ROI crop instead of the whole frame.

The classical backend reads pixels only inside the ROI (crop, gray, mask slice), so a
full-size canvas whose ROI window holds the de-rotated crop is indistinguishable to it
from ``derotate_image(raw)``. Pixels outside the ROI are stale and must never be used
(the debug overlay and anything else that needs the full frame must use the full path).
"""

from __future__ import annotations

import numpy as np


def raw_window(roi, width: int, height: int, rotate_180: bool):
    """Map a de-rotated ROI (x0, y0, x1, y1; x1/y1 exclusive) to the raw-frame window."""
    x0, y0, x1, y1 = (int(v) for v in roi)
    if rotate_180:
        return width - x1, height - y1, width - x0, height - y0
    return x0, y0, x1, y1


class RoiCanvas:
    """Reusable full-size frame that only ever holds the current ROI window."""

    def __init__(self, width: int, height: int, rotate_180: bool, channels: int = 3):
        self.width, self.height, self.rotate_180 = int(width), int(height), bool(rotate_180)
        shape = (self.height, self.width) if channels == 1 else (self.height, self.width, channels)
        self._canvas = np.zeros(shape, np.uint8)

    def fill(self, raw: np.ndarray, roi) -> np.ndarray:
        """Return the canvas with the de-rotated ``raw`` valid inside ``roi``."""
        if raw.shape[:2] != (self.height, self.width) or raw.ndim != self._canvas.ndim:
            raise ValueError("raw frame shape does not match the canvas")
        x0, y0, x1, y1 = (int(v) for v in roi)
        rx0, ry0, rx1, ry1 = raw_window(roi, self.width, self.height, self.rotate_180)
        window = raw[ry0:ry1, rx0:rx1]
        self._canvas[y0:y1, x0:x1] = window[::-1, ::-1] if self.rotate_180 else window
        return self._canvas

    def fill_window(self, window: np.ndarray, roi) -> np.ndarray:
        """Like :meth:`fill`, but ``window`` is only the raw-frame window ``raw_window(roi)`` (not a full frame)."""
        x0, y0, x1, y1 = (int(v) for v in roi)
        if window.shape != (y1 - y0, x1 - x0, self._canvas.shape[2] if self._canvas.ndim == 3 else 1):
            raise ValueError("raw window shape does not match the roi")
        self._canvas[y0:y1, x0:x1] = window[::-1, ::-1] if self.rotate_180 else window
        return self._canvas


def mcu_aligned_window(x0, y0, w, h, width, height, mcu_w=16, mcu_h=16, margin=16):
    """Python mirror of ``jpeg_window_plan`` in jpeg_window.c.

    The window [x0,x0+w) x [y0,y0+h) grown by ``margin`` px, snapped outwards to the MCU grid and clipped
    to the image: (ax0, ay0, ax1, ay1) with exclusive upper bounds. ``None`` if empty or outside the image.
    """
    if w <= 0 or h <= 0 or x0 < 0 or y0 < 0 or x0 + w > width or y0 + h > height or margin < 0:
        return None
    ax0 = max(0, x0 - margin) // mcu_w * mcu_w
    ay0 = max(0, y0 - margin) // mcu_h * mcu_h
    ax1 = min(width, -(-(x0 + w + margin) // mcu_w) * mcu_w)
    ay1 = min(height, -(-(y0 + h + margin) // mcu_h) * mcu_h)
    return ax0, ay0, ax1, ay1


class PartialRoiDecoder:
    """JPEG bytes -> RoiCanvas holding exactly the de-rotated ROI pixels, decoding only the raw window.

    ``decoder`` is a :class:`jpeg_window.WindowDecoder`. :meth:`decode_roi` returns ``None`` on ANY problem
    (unsupported JPEG, libjpeg warning/error, size mismatch); the caller must then use ``cv2.imdecode``.
    """

    def __init__(self, decoder, canvas: RoiCanvas):
        self.decoder, self.canvas = decoder, canvas
        self._buf = np.empty(canvas.width * canvas.height * 3, np.uint8)  # reused raw-window storage
        self.failures = 0
        self.last_status = 0

    def decode_roi(self, data: np.ndarray, roi):
        c = self.canvas
        rx0, ry0, rx1, ry1 = raw_window(roi, c.width, c.height, c.rotate_180)
        w, h = rx1 - rx0, ry1 - ry0
        if w <= 0 or h <= 0:
            return None
        window = self._buf[:w * h * 3].reshape(h, w, 3)
        status, iw, ih = self.decoder.decode(data, rx0, ry0, w, h, window)
        self.last_status = status
        if status != 0 or (iw, ih) != (c.width, c.height):
            self.failures += 1
            return None
        return c.fill_window(window, roi)
