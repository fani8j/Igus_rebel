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
