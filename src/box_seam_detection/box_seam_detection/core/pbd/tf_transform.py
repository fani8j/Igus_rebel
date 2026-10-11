# Provenance: moved into box_seam_detection (2026-10-12) from rgbd_captures/precision_box_detector/precision_box_detector/tf_transform.py
# at commit 6df59943f23e0d551cdc448b47ffea9e95ece20a (working tree identical to HEAD, unmodified).
# Only imports/sys.path handling changed; logic unchanged.
"""Rigid-body frame transforms for framed geometric primitives.

Implements the plan section 3.1 convention::

    X_target = R @ X_source + t

for points, and for planes ``n_s . x_s + d_s == 0``::

    n_target = R @ n_source
    d_target = d_s - n_target . t

with the transformed plane renormalized to a unit normal (the
:class:`Plane` constructor normalizes ``(normal, d)`` jointly, so the
geometric plane is preserved).

Frames are explicit: every transform takes a target ``frame_id`` and
returns a new framed object. Nothing is transformed silently.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "as_homogeneous",
    "transform_point",
    "transform_plane",
]


def _require_frame_id(frame_id: Any) -> str:
    if not isinstance(frame_id, str) or not frame_id:
        raise ValueError(f"frame_id must be a non-empty string, got {frame_id!r}")
    return frame_id


def as_homogeneous(R: Any, t: Any) -> np.ndarray:
    """Build a 4x4 homogeneous matrix from a 3x3 ``R`` and length-3 ``t``."""
    Rm = np.asarray(R, dtype=np.float64)
    tv = np.asarray(t, dtype=np.float64).reshape(-1)
    if Rm.shape != (3, 3):
        raise ValueError(f"R must have shape (3, 3), got {Rm.shape}")
    if tv.shape != (3,):
        raise ValueError(f"t must have shape (3,), got {tv.shape}")
    if not np.all(np.isfinite(Rm)) or not np.all(np.isfinite(tv)):
        raise ValueError("R and t must be finite")
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rm
    T[:3, 3] = tv
    return T


def _split_homogeneous(T: Any) -> tuple[np.ndarray, np.ndarray]:
    mat = np.asarray(T, dtype=np.float64)
    if mat.shape != (4, 4):
        raise ValueError(f"T must have shape (4, 4), got {mat.shape}")
    if not np.all(np.isfinite(mat)):
        raise ValueError("T must be finite")
    if not np.allclose(mat[3, :], (0.0, 0.0, 0.0, 1.0), atol=1e-9):
        raise ValueError(
            f"T last row must be [0, 0, 0, 1], got {mat[3, :].tolist()}"
        )
    return mat[:3, :3].copy(), mat[:3, 3].copy()


def transform_point(point: Any, T: Any, target_frame: str) -> Any:
    """Transform point(s) into ``target_frame`` via 4x4 ``T``.

    ``point`` is a :class:`FramedPoint` or a raw ``(3,)``/``(N, 3)`` array
    already in the source frame. Returns a ``FramedPoint`` in
    ``target_frame``.
    """
    from .camera_model import FramedPoint

    target = _require_frame_id(target_frame)
    R, t = _split_homogeneous(T)
    if isinstance(point, FramedPoint):
        arr = point.xyz
    else:
        arr = np.asarray(point, dtype=np.float64)
    if arr.shape != (3,) and not (
        arr.ndim == 2 and arr.shape[0] > 0 and arr.shape[1] == 3
    ):
        raise ValueError(
            f"point xyz must have shape (3,) or (N, 3), got {arr.shape}"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("point xyz must be finite")
    single = arr.ndim == 1
    mat = arr.reshape(-1, 3)
    out = mat @ R.T + t.reshape(1, 3)
    return FramedPoint(out[0] if single else out, target)


def transform_plane(
    plane: Any,
    R_or_T: Any,
    t_or_target_frame: Any = None,
    target_frame: Any = None,
) -> Any:
    """Transform a plane into ``target_frame`` with the plan 3.1 formula.

    Two call shapes (same function, so both ``camera_model`` and
    ``tf_transform`` callers agree)::

        transform_plane(plane, T, target_frame)        # T is 4x4
        transform_plane(plane, R, t, target_frame)     # R 3x3, t length-3

    ``n_target = R @ n_source``,
    ``d_target = d_source - n_target . t``; the result is renormalized to
    a unit normal by the ``Plane`` constructor.
    """
    from .camera_model import Plane

    if target_frame is None:
        # 3-arg form: (plane, T, target_frame).
        if t_or_target_frame is None:
            raise ValueError(
                "transform_plane needs (plane, T, target_frame) "
                "or (plane, R, t, target_frame)"
            )
        target = _require_frame_id(t_or_target_frame)
        R, t = _split_homogeneous(R_or_T)
    else:
        target = _require_frame_id(target_frame)
        R = np.asarray(R_or_T, dtype=np.float64)
        t = np.asarray(t_or_target_frame, dtype=np.float64).reshape(-1)
        if R.shape != (3, 3):
            raise ValueError(f"R must have shape (3, 3), got {R.shape}")
        if t.shape != (3,):
            raise ValueError(f"t must have shape (3,), got {t.shape}")
        if not np.all(np.isfinite(R)) or not np.all(np.isfinite(t)):
            raise ValueError("R and t must be finite")

    if isinstance(plane, Plane):
        normal = plane.normal
        d = float(plane.d)
    else:
        normal = getattr(plane, "normal", None)
        d = getattr(plane, "d", None)
        if normal is None or d is None:
            raise TypeError(
                "plane must be a camera_model.Plane "
                f"(normal/d/frame_id), got {type(plane).__name__}"
            )
        normal = np.asarray(normal, dtype=np.float64).reshape(-1)
        d = float(d)
        if normal.shape != (3,):
            raise ValueError(
                f"plane normal must have shape (3,), got {normal.shape}"
            )
    n_target = R @ normal
    d_target = d - float(n_target @ t)
    return Plane(n_target, d_target, target)
