"""Scene configuration: the camera/cell-specific constants of the detector.

A *scene* bundles everything that depends on where the camera is and what the
box stands on, and nothing else: depth-top workspace limits and support mode,
the guided-plane (wrong-face) gate thresholds, reliability constants, the
default hand-eye calibration, the reference driver extrinsic, and the seam
direction soft prior. Two scenes ship in ``config/scenes/``:

- ``table_cell_v1``: the original single-arm table cell (today's values, the
  default). Built in below as :data:`TABLE_CELL_V1` and mirrored byte for byte
  by ``config/scenes/table_cell_v1.yaml`` (a unit test keeps them equal), so
  running without ``--scene-config`` never reads a file and is identical to the
  behaviour before scenes existed.
- ``dual_arm_v1``: the dual-arm cell (camera on its own mount, box on a small
  stand that hides the support surface, 1.0-1.3 m from the top).

A scene YAML overrides keys of :func:`config.default_config` (sections
``depth_top``, ``v2``, ``guided_plane``, ``reliability``, ``corner_checks``,
...; nested dicts merge) and carries a ``live`` block for the constants that
live outside the pipeline config. An override key that does not already exist
in the defaults is an error (a typo must never silently change nothing).

``driver_extrinsic`` (live block): the camera_link <- colour optical frame
transform is a property of the camera unit, so the constant the extrinsic check
compares a capture-manifest block against is scene config:

- ``reference``: ``recorded_constant`` (the Orbbec constant of
  ``live.base_link.RECORDED_DRIVER_EXTRINSIC``), an explicit block, or ``null``
  (no reference known: a manifest block is used as is, with a warning, never
  refused; the live-TF comparison of ``--driver-extrinsic tf`` is unaffected).
- ``fallback``: the transform composed when a ``recorded`` run has no manifest
  block (``recorded_constant`` or an explicit block). It is tagged
  ``recorded_constant_assumed`` in the snapshot when the scene says
  ``verified: false``.
"""
from __future__ import annotations

import copy
import hashlib
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_SCENE_NAME = "table_cell_v1"

#: top-level keys of a scene file (everything else is a config section override)
_META_KEYS = ("scene", "description", "calib_file", "live")

TABLE_CELL_CALIB = "~/.ros2/easy_handeye2/calibrations/rebel_handeye_h_v2.calib"

#: live block defaults = the behaviour before scenes existed
LIVE_DEFAULTS: dict = {
    # seam direction soft prior in base_link (deg, mod 180); null disables the prior
    "seam_direction_prior_deg": 83.0,
    "seam_direction_soft_range_deg": 45.0,
    # default max |opposite 3-D side difference| (mm) of the rectangle gate when
    # --rect-tol-mm is not given; null = the detector default (3.0)
    "rect_tol_mm": None,
    # generic (unknown-size) box on a hidden stand: the cut is not blocked for the
    # missing height when the independent depth check confirmed the top and every
    # top edge is depth-supported (detector.evaluate_cut_permission)
    "generic_cut_without_height": False,
    # hard top-size gate: {long: [min, max], short: [min, max]} in mm (long = mean of the longer
    # opposite-side pair of the FINAL 3-D top quad, short = the other pair); a top outside it is
    # rejected (top_size_out_of_range, retryable) regardless of --allow-* flags. null = gate off.
    "top_size_range_mm": None,
    # robot work area (core/work_area.py): {frame: base_link, polygon_xy_m, top_z_range_m, outside_keep_every,
    # margin_m?}. The capture cloud is cropped to it (speed), SAM3 candidates outside it are ignored, and a box
    # top with any corner outside it is a hard reject (work_area, not bypassable). null = feature off.
    "work_area": None,
    # fixed floor / support reference plane (core/fixed_floor.py): {frame: base_link, normal, offset_m,
    # tolerance_m, ...}. hidden_stand scenes only: the floor is this measured plane instead of a per-frame
    # RANSAC facet; a cloud that contradicts it fails closed (depth_fixed_floor_mismatch). null = off.
    "fixed_floor": None,
    # recorded_constant | null | explicit block (see module docstring)
    "driver_extrinsic": {
        "reference": "recorded_constant",
        "fallback": "recorded_constant",
        "verified": True,
    },
}

#: the built-in default scene (= today's values; the YAML mirrors it)
TABLE_CELL_V1: dict = {
    "scene": DEFAULT_SCENE_NAME,
    "description": "single-arm table cell: camera ~0.8 m, box on a visible table (today's values)",
    "calib_file": TABLE_CELL_CALIB,
    "depth_top": {
        "ws_x_min": -1.2, "ws_x_max": 1.2, "ws_y_min": -0.8, "ws_y_max": 0.8,
        "ws_z_min": 0.2, "ws_z_max": 2.0,
        "offset_min": 0.05, "offset_max": 0.60,
        "support": "visible_table",
        "support_expand_m": 0.10,
    },
    "guided_plane": {"ransac_threshold_m": 0.003},
    "v2": {
        "min_plane_inlier_ratio": 0.80,
        "max_plane_p95_mm": 8.0,
        "noise_adaptive": {"enabled": False},
    },
    "live": copy.deepcopy(LIVE_DEFAULTS),
}


class SceneError(ValueError):
    """Unreadable / invalid scene configuration (fail closed)."""


@dataclass(frozen=True)
class Scene:
    name: str
    path: str | None
    sha256: str | None
    description: str = ""
    calib_file: str = TABLE_CELL_CALIB
    overrides: dict = field(default_factory=dict)      # config section overrides
    live: dict = field(default_factory=lambda: copy.deepcopy(LIVE_DEFAULTS))

    # ---- config ---------------------------------------------------------
    def apply(self, cfg: dict) -> dict:
        """``cfg`` (a :func:`config.default_config` dict) with this scene's
        overrides merged in (in place, also returned)."""
        _merge(cfg, self.overrides, "")
        cfg["scene"] = self.provenance()
        return cfg

    def provenance(self) -> dict:
        return {"name": self.name, "path": self.path, "sha256": self.sha256}

    # ---- live constants ---------------------------------------------------
    @property
    def seam_prior_deg(self) -> float | None:
        v = self.live.get("seam_direction_prior_deg")
        return None if v is None else float(v)

    @property
    def rect_tol_mm(self) -> float | None:
        v = self.live.get("rect_tol_mm")
        return None if v is None else float(v)

    @property
    def generic_cut_without_height(self) -> bool:
        return self.live.get("generic_cut_without_height") is True

    @property
    def top_size_range_mm(self) -> dict | None:
        """``{"long": [min, max], "short": [min, max]}`` (floats, validated at load) or None."""
        v = self.live.get("top_size_range_mm")
        return None if v is None else {k: [float(x) for x in v[k]] for k in ("long", "short")}

    @property
    def work_area(self) -> dict | None:
        """Normalised ``live.work_area`` mapping (validated at load, see core/work_area.py) or None (off)."""
        return self.live.get("work_area")

    @property
    def fixed_floor(self) -> dict | None:
        """Normalised ``live.fixed_floor`` mapping (validated at load, see core/fixed_floor.py) or None (off)."""
        return self.live.get("fixed_floor")

    @property
    def seam_soft_range_deg(self) -> float:
        return float(self.live.get("seam_direction_soft_range_deg", 45.0))

    def driver_block(self, which: str):
        """The scene's driver extrinsic block (``which``: reference|fallback)
        or None (reference null)."""
        spec = (self.live.get("driver_extrinsic") or {}).get(which)
        if spec is None:
            return None
        if spec == "recorded_constant":
            from ..live import base_link  # noqa: PLC0415
            return base_link.RECORDED_DRIVER_EXTRINSIC
        if isinstance(spec, dict):
            return {**spec,
                    "translation": tuple(float(v) for v in spec["translation"]),
                    "quaternion_xyzw": tuple(float(v) for v in spec["quaternion_xyzw"])}
        raise SceneError(f"driver_extrinsic.{which}: {spec!r}")

    @property
    def driver_verified(self) -> bool:
        return bool((self.live.get("driver_extrinsic") or {}).get("verified", True))


# ---------------------------------------------------------------------------

def _merge(base: dict, over: dict, where: str) -> None:
    for key, value in over.items():
        here = f"{where}.{key}" if where else key
        if key not in base:
            raise SceneError(f"unknown config key {here!r} in scene overrides")
        if isinstance(base[key], dict) and isinstance(value, dict):
            _merge(base[key], value, here)
        elif isinstance(base[key], dict) != isinstance(value, dict) and base[key] is not None:
            raise SceneError(f"config key {here!r}: expected "
                             f"{'a mapping' if isinstance(base[key], dict) else 'a scalar/list'}")
        else:
            base[key] = copy.deepcopy(value)


def scenes_dir() -> Path | None:
    """Directory holding the shipped scene files (source tree first, then the
    ament share directory of an installed package)."""
    src = Path(__file__).resolve().parents[2] / "config" / "scenes"
    if src.is_dir():
        return src
    try:
        from ament_index_python.packages import get_package_share_directory  # noqa: PLC0415
        share = Path(get_package_share_directory("box_seam_detection")) / "config" / "scenes"
        return share if share.is_dir() else None
    except Exception:  # noqa: BLE001
        return None


def resolve_scene_path(spec: str) -> Path:
    """A file path, or a shipped scene name (``dual_arm_v1``)."""
    p = Path(os.path.expanduser(str(spec)))
    if p.is_file():
        return p
    d = scenes_dir()
    if d is not None and "/" not in str(spec):
        for cand in (d / str(spec), d / f"{spec}.yaml"):
            if cand.is_file():
                return cand
    raise SceneError(f"scene config not found: {spec!r}")


def _validate_top_size_range(value) -> dict | None:
    """Normalised ``{"long": [lo, hi], "short": [lo, hi]}`` (floats) or None (gate off)."""
    if value is None:
        return None
    what = "live.top_size_range_mm"
    if not isinstance(value, dict) or set(value) != {"long", "short"}:
        raise SceneError(f"{what} must be a mapping with exactly the keys long and short "
                         f"(each [min, max] in mm) or null, got {value!r}")
    out = {}
    for key in ("long", "short"):
        pair = value[key]
        if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                or any(isinstance(x, bool) or not isinstance(x, (int, float))
                       or not math.isfinite(float(x)) for x in pair)):
            raise SceneError(f"{what}.{key} must be [min, max] with two finite numbers, got {pair!r}")
        lo, hi = float(pair[0]), float(pair[1])
        if not (0.0 < lo < hi):
            raise SceneError(f"{what}.{key} must satisfy 0 < min < max, got [{lo:g}, {hi:g}]")
        out[key] = [lo, hi]
    return out


def _from_mapping(data: dict, path: str | None, sha: str | None) -> Scene:
    if not isinstance(data, dict):
        raise SceneError("scene config must be a mapping")
    unknown_live = sorted(set(data.get("live") or {}) - set(LIVE_DEFAULTS))
    if unknown_live:
        raise SceneError(f"unknown live key(s) {unknown_live}")
    live = copy.deepcopy(LIVE_DEFAULTS)
    for k, v in (data.get("live") or {}).items():
        if k == "driver_extrinsic":
            if not isinstance(v, dict) or set(v) - {"reference", "fallback", "verified"}:
                raise SceneError("live.driver_extrinsic: keys are reference, fallback, verified")
            live[k] = {**live[k], **v}
        else:
            live[k] = v
    for key in ("seam_direction_prior_deg", "seam_direction_soft_range_deg", "rect_tol_mm"):
        v = live.get(key)
        if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float))
                              or not math.isfinite(float(v))):
            raise SceneError(f"live.{key} must be a finite number or null, got {v!r}")
    live["top_size_range_mm"] = _validate_top_size_range(live.get("top_size_range_mm"))
    from .work_area import WorkAreaError, validate_work_area  # noqa: PLC0415
    try:
        live["work_area"] = validate_work_area(live.get("work_area"))
    except WorkAreaError as error:
        raise SceneError(str(error)) from error
    from .fixed_floor import FixedFloorError, validate_fixed_floor  # noqa: PLC0415
    try:
        live["fixed_floor"] = validate_fixed_floor(live.get("fixed_floor"))
    except FixedFloorError as error:
        raise SceneError(str(error)) from error
    if live["fixed_floor"] is not None and (
            (data.get("depth_top") or {}).get("support") != "hidden_stand"):
        raise SceneError("live.fixed_floor requires depth_top.support: hidden_stand")
    if not isinstance(live.get("generic_cut_without_height"), bool):
        raise SceneError("live.generic_cut_without_height must be true or false")
    if live["driver_extrinsic"].get("fallback") is None:
        raise SceneError("live.driver_extrinsic.fallback must not be null")
    overrides = {k: v for k, v in data.items() if k not in _META_KEYS}
    scene = Scene(name=str(data.get("scene") or Path(path or DEFAULT_SCENE_NAME).stem),
                  path=path, sha256=sha,
                  description=str(data.get("description") or ""),
                  calib_file=str(data.get("calib_file") or TABLE_CELL_CALIB),
                  overrides=copy.deepcopy(overrides), live=live)
    # validate the overrides now (unknown keys), not at first use
    from .config import V2_DEFAULTS  # noqa: PLC0415
    _merge(copy.deepcopy(V2_DEFAULTS), scene.overrides, "")
    for k in ("reference", "fallback"):
        scene.driver_block(k)
    return scene


def builtin_default() -> Scene:
    return _from_mapping(copy.deepcopy(TABLE_CELL_V1), None, None)


def load_scene(spec: str | os.PathLike | None = None) -> Scene:
    """The scene for ``spec`` (None / ``table_cell_v1`` without a file = the
    built-in default, no file is read)."""
    if spec is None or str(spec) == "":
        return builtin_default()
    if str(spec) == DEFAULT_SCENE_NAME and not Path(str(spec)).is_file():
        return builtin_default()
    path = resolve_scene_path(str(spec))
    raw = path.read_bytes()
    try:
        import yaml  # noqa: PLC0415
        data = yaml.safe_load(raw.decode("utf-8"))
    except Exception as error:  # noqa: BLE001
        raise SceneError(f"cannot parse scene config {path}: {error}") from error
    return _from_mapping(data, str(path.resolve()), hashlib.sha256(raw).hexdigest())


def scene_config(spec=None) -> tuple[Scene, dict]:
    """``(scene, cfg)``: the scene and the v2 config with its overrides applied."""
    from .config import default_config  # noqa: PLC0415
    scene = load_scene(spec)
    return scene, scene.apply(default_config())
