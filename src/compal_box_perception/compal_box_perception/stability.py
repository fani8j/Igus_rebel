"""Temporal consensus for rejecting one-frame seam jumps."""

from collections import deque
from typing import Optional, Tuple

import numpy as np

class TemporalArrayMedian:
    """Median-filter a fixed-shape geometry array and reject one-frame jumps."""

    def __init__(self, window_size: int, max_point_jump: float) -> None:
        if window_size < 1:
            raise ValueError("window_size must be positive")
        if max_point_jump <= 0.0:
            raise ValueError("max_point_jump must be positive")
        self._values = deque(maxlen=window_size)
        self.max_point_jump = max_point_jump

    def update(self, value: np.ndarray) -> np.ndarray:
        array = np.asarray(value, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != 2:
            raise ValueError("geometry must have shape (N, 2)")
        if not np.all(np.isfinite(array)):
            raise ValueError("geometry must be finite")
        if self._values:
            if array.shape != self._values[0].shape:
                raise ValueError("geometry shape changed")
            median = np.median(np.stack(self._values), axis=0)
            point_jumps = np.linalg.norm(array - median, axis=1)
            if float(np.max(point_jumps)) > self.max_point_jump:
                return median
        self._values.append(array.copy())
        return np.median(np.stack(self._values), axis=0)

    def reset(self) -> None:
        self._values.clear()


class TemporalUnitVectorMedian:
    """Median-filter a direction while preserving a consistent sign."""

    def __init__(self, window_size: int, max_angle_jump_deg: float) -> None:
        if window_size < 1:
            raise ValueError("window_size must be positive")
        if not 0.0 < max_angle_jump_deg <= 180.0:
            raise ValueError("max_angle_jump_deg must be in (0, 180]")
        self._values = deque(maxlen=window_size)
        self._minimum_alignment = np.cos(np.deg2rad(max_angle_jump_deg))

    def update(self, value: np.ndarray) -> np.ndarray:
        vector = np.asarray(value, dtype=np.float64)
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            raise ValueError("unit vector input must be a finite 3-vector")
        norm = float(np.linalg.norm(vector))
        if norm < 1e-9:
            raise ValueError("unit vector input must be non-zero")
        vector /= norm
        if self._values:
            median = np.median(np.stack(self._values), axis=0)
            median /= np.linalg.norm(median)
            alignment = float(np.dot(vector, median))
            if alignment < 0.0:
                vector = -vector
                alignment = -alignment
            if alignment < self._minimum_alignment:
                return median
        self._values.append(vector.copy())
        median = np.median(np.stack(self._values), axis=0)
        return median / np.linalg.norm(median)

    def reset(self) -> None:
        self._values.clear()


class TemporalScalarMedian:
    """Robustly smooth a physical scalar that should vary slowly."""

    def __init__(
        self, window_size: int, max_jump: float, jump_confirmation: int = 3
    ) -> None:
        if window_size < 1:
            raise ValueError("window_size must be positive")
        if max_jump <= 0.0:
            raise ValueError("max_jump must be positive")
        if jump_confirmation < 1:
            raise ValueError("jump_confirmation must be positive")
        self._values = deque(maxlen=window_size)
        self._pending = deque(maxlen=jump_confirmation)
        self.max_jump = max_jump

    def update(self, value: float) -> float:
        if not np.isfinite(value):
            raise ValueError("scalar value must be finite")
        value = float(value)
        if self._values:
            median = float(np.median(self._values))
            if abs(value - median) > self.max_jump:
                self._pending.append(value)
                if (
                    len(self._pending) == self._pending.maxlen
                    and np.ptp(self._pending) <= self.max_jump
                ):
                    self._values.clear()
                    self._values.extend(self._pending)
                    self._pending.clear()
                    return float(np.median(self._values))
                return median
        self._pending.clear()
        self._values.append(value)
        return float(np.median(self._values))

    def reset(self) -> None:
        self._values.clear()
        self._pending.clear()



class TemporalSeamFilter:
    def __init__(
        self,
        confirmation_frames: int,
        max_endpoint_jump_m: float,
        lost_after_invalid_frames: int,
    ) -> None:
        if confirmation_frames < 1:
            raise ValueError("confirmation_frames must be positive")
        if max_endpoint_jump_m <= 0.0:
            raise ValueError("max_endpoint_jump_m must be positive")
        if lost_after_invalid_frames < 1:
            raise ValueError("lost_after_invalid_frames must be positive")
        self.confirmation_frames = confirmation_frames
        self.max_endpoint_jump_m = max_endpoint_jump_m
        self.lost_after_invalid_frames = lost_after_invalid_frames
        self._samples = deque(maxlen=confirmation_frames)
        self._invalid_frames = 0

    @property
    def confirmation_count(self) -> int:
        return len(self._samples)

    def update(
        self, start_xyz: np.ndarray, end_xyz: np.ndarray
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        start = np.asarray(start_xyz, dtype=np.float64)
        end = np.asarray(end_xyz, dtype=np.float64)
        if start.shape != (3,) or end.shape != (3,):
            raise ValueError("seam endpoints must be three-dimensional")
        if not np.all(np.isfinite(start)) or not np.all(np.isfinite(end)):
            raise ValueError("seam endpoints must be finite")

        if self._samples:
            previous_starts = np.stack([sample[0] for sample in self._samples])
            previous_ends = np.stack([sample[1] for sample in self._samples])
            median_start = np.median(previous_starts, axis=0)
            median_end = np.median(previous_ends, axis=0)
            jump = max(
                float(np.linalg.norm(start - median_start)),
                float(np.linalg.norm(end - median_end)),
            )
            if jump > self.max_endpoint_jump_m:
                self._samples.clear()

        self._samples.append((start, end))
        self._invalid_frames = 0
        if len(self._samples) < self.confirmation_frames:
            return None

        starts = np.stack([sample[0] for sample in self._samples])
        ends = np.stack([sample[1] for sample in self._samples])
        return np.median(starts, axis=0), np.median(ends, axis=0)

    def mark_invalid(self) -> bool:
        """Return true when tracking must be cleared after repeated failures."""
        self._invalid_frames += 1
        if self._invalid_frames < self.lost_after_invalid_frames:
            return False
        self._samples.clear()
        self._invalid_frames = 0
        return True

    def reset(self) -> None:
        self._samples.clear()
        self._invalid_frames = 0
