# Provenance: moved into box_seam_detection (2026-10-12) from rgbd_captures/precision_box_detector/precision_box_detector/camera_model.py
# at commit 6df59943f23e0d551cdc448b47ffea9e95ece20a (working tree identical to HEAD, unmodified).
# Only imports/sys.path handling changed; logic unchanged.
"""Frame-explicit pinhole camera geometry for the precision box detector.

Conventions (meters internally; every geometric object carries ``frame_id``):

- ``CameraModel`` is a pinhole model built from raw ``K`` (distorted image)
  or rectified ``P`` (already-undistorted image) calibration. It ingests
  plain dicts (:meth:`CameraModel.from_mapping`) and ROS ``CameraInfo``-like
  objects (:meth:`CameraModel.from_camera_info`, duck-typed, no ROS import).
- Distortion follows the ROS ``plumb_bob`` model
  (k1, k2, p1, p2 [, k3]). Only ``plumb_bob`` (and explicitly undistorted
  ``none``) models are supported; anything else raises.
- A rectified model NEVER undistorts: ``is_rectified=True`` requires empty
  (all-zero) distortion coefficients, otherwise construction raises instead
  of silently double-undistorting.
- Planes satisfy ``normal . x + d == 0`` with unit ``normal`` and ``d`` in
  meters. Non-unit normals are normalized together with ``d`` so the
  geometric plane described by the inputs is preserved.
- Rigid-body transforms live in :mod:`tf_transform` and are re-exported here
  so both ``camera_model.transform_*`` and ``tf_transform.transform_*``
  refer to the same functions.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
import math
import operator

import numpy as np

__all__ = [
    "CameraModel",
    "FramedPoint",
    "Plane",
    "pixel_to_ray",
    "project_xyz",
    "deproject_depth",
    "ray_plane_intersection",
    "transform_point",
    "transform_plane",
]


# ---------------------------------------------------------------------------
# Small validation helpers


def _require_frame_id(frame_id: Any) -> str:
    if not isinstance(frame_id, str) or not frame_id:
        raise ValueError(f"frame_id must be a non-empty string, got {frame_id!r}")
    return frame_id
def _as_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number, got {value!r}") from exc
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return out


def _flatten_nested(values: Any) -> list:
    """Flatten one-or-more levels of nested K/P/D sequences to a flat list."""
    if isinstance(values, np.ndarray):
        return list(values.reshape(-1).tolist())
    flat: list = []
    for v in values:
        if isinstance(v, (list, tuple, np.ndarray)):
            flat.extend(_flatten_nested(v))
        else:
            flat.append(v)
    return flat


def _as_float_list(values: Any, name: str, size: int) -> list:
    if isinstance(values, np.ndarray):
        seq = list(values.reshape(-1).tolist())
    elif isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple, np.ndarray)):
        try:
            seq = list(values)
        except TypeError as exc:
            raise ValueError(f"{name} must hold {size} numbers") from exc
    else:
        seq = list(values)
    seq = _flatten_nested(seq)
    if len(seq) != size:
        raise ValueError(f"{name} must hold {size} numbers, got {len(seq)}")
    return [_as_float(v, f"{name}[{i}]") for i, v in enumerate(seq)]


def _as_pos_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    try:
        out = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer, got {value!r}") from exc
    if out <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return out


# ---------------------------------------------------------------------------
# Framed geometry primitives


@dataclass(eq=False)
class FramedPoint:
    """Point(s) with an explicit coordinate frame.

    ``xyz`` is ``(3,)`` for a single point or ``(N, 3)`` for a batch, in
    meters. ``frame_id`` is REQUIRED and never defaulted.
    """

    xyz: Any
    frame_id: str

    def __post_init__(self) -> None:
        arr = np.asarray(self.xyz, dtype=np.float64)
        if arr.shape != (3,) and not (
            arr.ndim == 2 and arr.shape[0] > 0 and arr.shape[1] == 3
        ):
            raise ValueError(
                f"FramedPoint xyz must have shape (3,) or (N, 3), got {arr.shape}"
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError("FramedPoint xyz must be finite")
        self.xyz = arr
        self.frame_id = _require_frame_id(self.frame_id)


@dataclass(eq=False)
class Plane:
    """Plane ``normal . x + d == 0`` with an explicit coordinate frame.

    ``normal`` is normalized to unit length on construction and ``d`` is
    scaled by the same factor, so the geometric plane described by the
    inputs is preserved. ``d`` is in meters. ``frame_id`` is REQUIRED.
    """

    normal: Any
    d: Any
    frame_id: str

    def __post_init__(self) -> None:
        arr = np.asarray(self.normal, dtype=np.float64)
        if arr.shape != (3,):
            raise ValueError(f"Plane normal must have shape (3,), got {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError("Plane normal must be finite")
        norm = float(np.linalg.norm(arr))
        if norm < 1e-12:
            raise ValueError("Plane normal must be non-zero")
        self.normal = arr / norm
        self.d = _as_float(self.d, "Plane.d") / norm
        self.frame_id = _require_frame_id(self.frame_id)


# ---------------------------------------------------------------------------
# Camera model


_SUPPORTED_DISTORTION_MODELS = ("plumb_bob", "none", "")


class CameraModel:
    """Pinhole camera model with explicit frame and distortion semantics.

    ``k`` holds the 3x3 intrinsic matrix row-major
    (fx=k[0], fy=k[4], cx=k[2], cy=k[5]). ``d`` holds ``plumb_bob``
    coefficients (k1, k2, p1, p2 [, k3]) and MUST be empty when
    ``is_rectified`` is True (a rectified image is already undistorted).
    """

    def __init__(
        self,
        frame_id: str,
        width: int,
        height: int,
        k: Any,
        d: Any = (),
        distortion_model: Any = "plumb_bob",
        is_rectified: bool = False,
    ) -> None:
        self.frame_id = _require_frame_id(frame_id)
        self.width = _as_pos_int(width, "width")
        self.height = _as_pos_int(height, "height")

        flat_k = _as_float_list(k, "k", 9)
        mat = np.array(flat_k, dtype=np.float64).reshape(3, 3)
        if not (abs(mat[0, 1]) < 1e-9 and abs(mat[2, 0]) < 1e-9
                and abs(mat[2, 1]) < 1e-9 and mat[2, 2] == 1.0):
            raise ValueError(
                "k must be a pinhole matrix (zero skew, last row [0, 0, 1]), "
                f"got {flat_k!r}"
            )
        if not (mat[0, 0] > 0.0 and mat[1, 1] > 0.0):
            raise ValueError(
                f"k must have positive fx/fy, got fx={mat[0, 0]!r} fy={mat[1, 1]!r}"
            )
        self.k = mat

        model = "plumb_bob" if distortion_model is None else str(distortion_model)
        if model not in _SUPPORTED_DISTORTION_MODELS:
            raise ValueError(
                f"unsupported distortion_model {distortion_model!r}; "
                f"supported: {list(_SUPPORTED_DISTORTION_MODELS)}"
            )
        self.distortion_model = model
        coeffs = [] if d is None else _flatten_nested(list(d))
        self.d = tuple(_as_float(v, f"d[{i}]") for i, v in enumerate(coeffs))
        if model in ("none", ""):
            if len(self.d) > 0 and any(v != 0.0 for v in self.d):
                raise ValueError(
                    "distortion_model='none' is inconsistent with non-zero d"
                )
            self.d = ()
        elif len(self.d) > 0 and all(v == 0.0 for v in self.d):
            self.d = ()
        if self.distortion_model == "plumb_bob" and len(self.d) not in (0, 4, 5, 8, 12, 14):
            raise ValueError(
                f"plumb_bob d must hold 0, 4, 5, 8, 12 or 14 coefficients, got {len(self.d)}"
            )

        self.is_rectified = bool(is_rectified)
        if self.is_rectified and any(v != 0.0 for v in self.d):
            raise ValueError(
                "is_rectified=True is inconsistent with non-zero distortion "
                "coefficients; a rectified image must not be undistorted twice"
            )

    @property
    def fx(self) -> float:
        return float(self.k[0, 0])

    @property
    def fy(self) -> float:
        return float(self.k[1, 1])

    @property
    def cx(self) -> float:
        return float(self.k[0, 2])

    @property
    def cy(self) -> float:
        return float(self.k[1, 2])

    def has_distortion(self) -> bool:
        """True when projection must apply the forward distortion model."""
        return (
            not self.is_rectified
            and self.distortion_model == "plumb_bob"
            and any(v != 0.0 for v in self.d)
        )

    # -- constructors ----------------------------------------------------

    @classmethod
    def from_mapping(cls, info: Mapping[str, Any]) -> "CameraModel":
        """Build from a plain dict (capture manifest ``camera_info`` style).

        Prefers rectified ``p``/``P`` (12 numbers) when present and falls
        back to ``k``/``K`` (9 numbers). Rectification is inferred from the
        data: ``P`` that disagrees with ``K`` means the image is already
        rectified; an explicit ``is_rectified`` entry always wins. ``P``
        plus non-zero ``d``/``D`` raises as inconsistent.
        """
        if not isinstance(info, Mapping):
            raise ValueError(
                "camera info must be a mapping, "
                f"got {type(info).__name__}"
            )
        frame = None
        for key in ("frame_id", "frame", "frameId"):
            if info.get(key) is not None:
                frame = info.get(key)
                break
        if frame is None:
            raise ValueError("camera info has no frame_id")
        if "width" not in info or "height" not in info:
            raise ValueError("camera info needs 'width' and 'height'")

        model = info.get("distortion_model", info.get("distortion", "plumb_bob"))
        dist = info.get("d", info.get("D", ()))
        raw_p = info.get("p", info.get("P", None))
        raw_k = info.get("k", info.get("K", None))
        explicit = info.get("is_rectified", info.get("rectified", None))

        if raw_p is not None:
            flat_p = _as_float_list(raw_p, "p", 12)
            from_p = [
                flat_p[0], 0.0, flat_p[2],
                0.0, flat_p[5], flat_p[6],
                0.0, 0.0, 1.0,
            ]
            if explicit is not None:
                rectified = bool(explicit)
            elif raw_k is not None:
                flat_k = _as_float_list(raw_k, "k", 9)
                rectified = not np.allclose(flat_k, from_p, atol=1e-9)
            else:
                rectified = True
            return cls(
                frame, info["width"], info["height"], from_p,
                d=dist, distortion_model=model, is_rectified=rectified,
            )
        if raw_k is not None:
            flat_k = _as_float_list(raw_k, "k", 9)
            rectified = bool(explicit) if explicit is not None else False
            return cls(
                frame, info["width"], info["height"], flat_k,
                d=dist, distortion_model=model, is_rectified=rectified,
            )
        raise ValueError("camera info has neither 'p'/'P' nor 'k'/'K'")

    @classmethod
    def from_camera_info(cls, msg: Any) -> "CameraModel":
        """Build from a ROS ``CameraInfo``-like object (duck-typed).

        Accepts a mapping (delegates to :meth:`from_mapping`) or an object
        with ``width``/``height``/``k``/``d``/``distortion_model``/``p``
        attributes and either ``header.frame_id`` or ``frame_id``. No ROS
        import is required.
        """
        if isinstance(msg, Mapping):
            return cls.from_mapping(msg)

        def _attr(*names: str, default: Any = None) -> Any:
            for name in names:
                if hasattr(msg, name):
                    return getattr(msg, name)
            return default

        width = _attr("width")
        height = _attr("height")
        if width is None or height is None:
            raise ValueError("CameraInfo-like object needs width/height")
        header = _attr("header")
        frame = None
        if header is not None:
            if isinstance(header, Mapping):
                frame = header.get("frame_id", header.get("frameId"))
            else:
                frame = getattr(header, "frame_id", None)
        if frame is None:
            frame = _attr("frame_id", "frame", "frameId")
        return cls.from_mapping(
            {
                "frame_id": frame,
                "width": width,
                "height": height,
                "k": _attr("k", "K"),
                "d": _attr("d", "D", default=()),
                "distortion_model": _attr(
                    "distortion_model", "distortionModel", default="plumb_bob"
                ),
                "p": _attr("p", "P", default=None),
            }
        )

    def __repr__(self) -> str:
        return (
            f"CameraModel(frame_id={self.frame_id!r}, {self.width}x{self.height}, "
            f"fx={self.fx:.3f} fy={self.fy:.3f} cx={self.cx:.3f} cy={self.cy:.3f}, "
            f"model={self.distortion_model!r}, rectified={self.is_rectified})"
        )


# ---------------------------------------------------------------------------
# Distortion helpers (plumb_bob, ROS/CameraInfo convention)


def _distort_normalized(
    xn: np.ndarray, yn: np.ndarray, d: tuple
) -> tuple[np.ndarray, np.ndarray]:
    k1, k2, p1, p2, k3 = (list(d) + [0.0] * 5)[:5]
    r2 = xn * xn + yn * yn
    radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    xd = xn * radial + 2.0 * p1 * xn * yn + p2 * (r2 + 2.0 * xn * xn)
    yd = yn * radial + p1 * (r2 + 2.0 * yn * yn) + 2.0 * p2 * xn * yn
    return xd, yd


def _undistort_normalized(
    xd: np.ndarray,
    yd: np.ndarray,
    d: tuple,
    iterations: int = 100,
    tol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Iteratively invert the forward plumb_bob model (fixed point)."""
    k1, k2, p1, p2, k3 = (list(d) + [0.0] * 5)[:5]
    xn = np.array(xd, dtype=np.float64, copy=True)
    yn = np.array(yd, dtype=np.float64, copy=True)
    for _ in range(iterations):
        r2 = xn * xn + yn * yn
        radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        xn_new = (xd - 2.0 * p1 * xn * yn - p2 * (r2 + 2.0 * xn * xn)) / radial
        yn_new = (yd - p1 * (r2 + 2.0 * yn * yn) - 2.0 * p2 * xn * yn) / radial
        step = max(
            float(np.max(np.abs(xn_new - xn), initial=0.0)),
            float(np.max(np.abs(yn_new - yn), initial=0.0)),
        )
        xn, yn = xn_new, yn_new
        if step < tol:
            break
    return xn, yn


def _coerce_point_array(xyz: Any, camera: CameraModel, name: str) -> np.ndarray:
    if isinstance(xyz, FramedPoint):
        if xyz.frame_id != camera.frame_id:
            raise ValueError(
                f"frame mismatch: {name} is in {xyz.frame_id!r} but the camera "
                f"is in {camera.frame_id!r}; transform it first"
            )
        arr = xyz.xyz
    else:
        arr = np.asarray(xyz, dtype=np.float64)
    if arr.shape != (3,) and not (
        arr.ndim == 2 and arr.shape[0] > 0 and arr.shape[1] == 3
    ):
        raise ValueError(f"{name} must have shape (3,) or (N, 3), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


# ---------------------------------------------------------------------------
# Projection / deprojection


def pixel_to_ray(u: Any, v: Any, camera: CameraModel) -> FramedPoint:
    """Unit ray(s) from the camera center through pixel(s), in frame.

    Distorted pixels are undistorted exactly once via the model; rectified
    models use the pinhole directly. Scalar input returns shape ``(3,)``,
    otherwise ``(N, 3)`` flattened in C order.
    """
    scalar = np.ndim(u) == 0 and np.ndim(v) == 0
    uu, vv = np.broadcast_arrays(
        np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64)
    )
    if not np.all(np.isfinite(uu)) or not np.all(np.isfinite(vv)):
        raise ValueError("pixel coordinates must be finite")
    xd = (uu - camera.cx) / camera.fx
    yd = (vv - camera.cy) / camera.fy
    if camera.has_distortion():
        xn, yn = _undistort_normalized(xd, yd, camera.d)
    else:
        xn, yn = xd, yd
    dirs = np.stack([xn.reshape(-1), yn.reshape(-1),
                     np.ones(xn.size)], axis=1)
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    return FramedPoint(dirs[0] if scalar else dirs, camera.frame_id)


def project_xyz(xyz: Any, camera: CameraModel) -> np.ndarray:
    """Project camera-frame point(s) to pixels (applies distortion once).

    Accepts a :class:`FramedPoint` (its frame MUST match the camera) or a
    raw ``(3,)``/``(N, 3)`` array assumed to already be in the camera frame.
    Points at/behind the camera (z <= 0) raise instead of projecting.
    """
    arr = _coerce_point_array(xyz, camera, "xyz")
    single = arr.ndim == 1
    pts = arr.reshape(-1, 3)
    z = pts[:, 2]
    if np.any(z <= 0.0):
        bad = int(np.sum(z <= 0.0))
        raise ValueError(f"cannot project {bad} point(s) with z <= 0")
    xn = pts[:, 0] / z
    yn = pts[:, 1] / z
    if camera.has_distortion():
        xd, yd = _distort_normalized(xn, yn, camera.d)
    else:
        xd, yd = xn, yn
    uv = np.stack([camera.fx * xd + camera.cx,
                   camera.fy * yd + camera.cy], axis=1)
    return uv[0] if single else uv


def deproject_depth(u: Any, v: Any, depth: Any, camera: CameraModel) -> FramedPoint:
    """Lift pixel(s) with depth (meters) to 3D points in the camera frame.

    Distorted pixels are undistorted exactly once; rectified models use the
    pinhole directly. Non-positive depths raise. Returns a
    :class:`FramedPoint` in ``camera.frame_id``.
    """
    uu, vv, zz = np.broadcast_arrays(
        np.asarray(u, dtype=np.float64),
        np.asarray(v, dtype=np.float64),
        np.asarray(depth, dtype=np.float64),
    )
    if not np.all(np.isfinite(uu)) or not np.all(np.isfinite(vv)):
        raise ValueError("pixel coordinates must be finite")
    if not np.all(np.isfinite(zz)) or np.any(zz <= 0.0):
        raise ValueError("depth must be finite and positive (meters)")
    xd = (uu - camera.cx) / camera.fx
    yd = (vv - camera.cy) / camera.fy
    if camera.has_distortion():
        xn, yn = _undistort_normalized(xd, yd, camera.d)
    else:
        xn, yn = xd, yd
    flat = lambda a: a.reshape(-1)
    pts = np.stack([flat(xn) * flat(zz), flat(yn) * flat(zz), flat(zz)], axis=1)
    scalar = np.ndim(u) == 0 and np.ndim(v) == 0 and np.ndim(depth) == 0
    return FramedPoint(pts[0] if scalar else pts, camera.frame_id)


def ray_plane_intersection(
    u: Any, v: Any, plane: Plane, camera: CameraModel
) -> FramedPoint:
    """Intersect camera ray(s) with a plane; frames MUST already match.

    Raises on frame mismatch, on (near-)parallel rays, and when the
    intersection lies behind the camera. Returns a :class:`FramedPoint`
    in ``camera.frame_id``.
    """
    if not isinstance(plane, Plane):
        raise TypeError(
            f"plane must be a camera_model.Plane, got {type(plane).__name__}"
        )
    if plane.frame_id != camera.frame_id:
        raise ValueError(
            f"frame mismatch: plane is in {plane.frame_id!r} but the camera "
            f"is in {camera.frame_id!r}; transform the plane first"
        )
    rays = pixel_to_ray(u, v, camera).xyz
    single = rays.ndim == 1
    mat = rays.reshape(-1, 3)
    denom = mat @ plane.normal
    if np.any(np.abs(denom) < 1e-12):
        raise ValueError("ray is parallel to the plane")
    s = -plane.d / denom
    if np.any(s <= 0.0):
        raise ValueError("ray-plane intersection is behind the camera")
    pts = mat * s[:, None]
    return FramedPoint(pts[0] if single else pts, camera.frame_id)


# Re-exported here so the contracted ``camera_model.transform_*`` names exist;
# the implementation lives in tf_transform (the TF boundary).
from .tf_transform import as_homogeneous, transform_plane, transform_point  # noqa: E402
