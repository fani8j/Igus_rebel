"""Calibration target definitions, detection and print-ready board generation.

The two arms carry boards with DISJOINT marker IDs, so one image can contain
both targets and the detector still knows which arm is which.

The ChArUco boards are implemented here (marker layout, rendering, and corner
refinement) instead of through ``cv2.aruco.CharucoBoard`` because the system
OpenCV on Ubuntu 22.04 / Humble (4.5.4) cannot set custom marker IDs on a
ChArUco board, while OpenCV >= 4.6 changed the legacy pattern.  Using only
``detectMarkers`` plus ``cornerSubPix`` works identically on 4.5.4 and 4.12.

Target frame: origin at the board centre, x to the right, y down (as seen when
reading the printed board), z into the board (right handed).  Chessboard
squares (i, j) with (i + j) even are black; markers sit in the white squares
((i + j) odd) in row-major order.  Units are metres.

Point IDs (unique per arm): marker k corner c -> 4*k + c (c: TL, TR, BR, BL),
inner chessboard corner n -> CHESS_ID_BASE + n.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

CHESS_ID_BASE = 1000
MM = 1e-3


# --------------------------------------------------------------------------
# cv2.aruco version shims (4.5.4 old API  <->  4.7+ new API)
# --------------------------------------------------------------------------
def get_dictionary(name: str):
    if not hasattr(cv2.aruco, name):
        raise ValueError(f"unknown ArUco dictionary '{name}'")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def dictionary_marker_size(name: str) -> int:
    """Number of bit cells per marker side (4 for DICT_4X4_*)."""
    return int(name.split("_")[1].split("X")[0]) if "X" in name else int(get_dictionary(name).markerSize)


def dictionary_capacity(name: str) -> int:
    return int(get_dictionary(name).bytesList.shape[0])


def marker_bits(name: str, marker_id: int) -> np.ndarray:
    """(n+2)x(n+2) boolean array, True = white, including the 1-cell black border."""
    n = dictionary_marker_size(name)
    cell = 10
    side = (n + 2) * cell
    d = get_dictionary(name)
    if hasattr(cv2.aruco, "generateImageMarker"):
        img = cv2.aruco.generateImageMarker(d, int(marker_id), side, borderBits=1)
    else:
        img = cv2.aruco.drawMarker(d, int(marker_id), side, borderBits=1)
    centres = np.arange(n + 2) * cell + cell // 2
    return img[np.ix_(centres, centres)] > 127


def make_detector_params(min_perimeter_rate: float = 0.02):
    if hasattr(cv2.aruco, "DetectorParameters_create"):
        p = cv2.aruco.DetectorParameters_create()
    else:
        p = cv2.aruco.DetectorParameters()
    # Marker corners are refined by refine_marker_corners() with a window scaled to the
    # marker; the built-in window of OpenCV 4.5 (fixed 5 px) is larger than a bit cell of
    # small markers and pulls corners onto the neighbouring bits.
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
    p.minMarkerPerimeterRate = float(min_perimeter_rate)
    return p


def refine_marker_corners(gray: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Sub-pixel refine the 4 corners of one marker with a window scaled to the marker size."""
    c = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    side = float(np.mean([np.linalg.norm(c[i] - c[(i + 1) % 4]) for i in range(4)]))
    win = int(max(1, min(5, round(0.09 * side))))
    h, w = gray.shape[:2]
    if np.any(c < win + 2) or np.any(c[:, 0] > w - win - 3) or np.any(c[:, 1] > h - win - 3):
        return c.astype(float)
    out = c.reshape(-1, 1, 2).copy()
    cv2.cornerSubPix(gray, out, (win, win), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3))
    out = out.reshape(4, 2)
    if np.linalg.norm(out - c, axis=1).max() > max(1.5, 0.15 * side):
        return c.astype(float)  # moved too far: keep the unrefined detection
    return out.astype(float)


# --------------------------------------------------------------------------
# Target specification
# --------------------------------------------------------------------------
@dataclass
class TargetSpec:
    arm: str
    type: str                       # "charuco" | "aruco"
    dictionary: str
    marker_length: float            # m
    marker_ids: List[int]           # charuco: one per marker square (row-major); aruco: [id]
    square_length: float = 0.0      # m (charuco)
    squares_x: int = 0
    squares_y: int = 0
    # touch points (table board only): id -> (x, y) in the target frame (m), z = 0
    touch_points: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    paper: str = ""                 # preferred print paper (table board: A4, landscape)
    # Printers scale x and y differently (feed vs. head direction): optional measured y values; default = x values.
    # ``square_length`` / ``marker_length`` are the sizes along the board x axis, ``*_y`` along the board y axis.
    square_length_y: Optional[float] = None
    marker_length_y: Optional[float] = None

    # ---- geometry ------------------------------------------------------
    @property
    def sq_y(self) -> float:
        return self.square_length if self.square_length_y is None else self.square_length_y

    @property
    def mk_y(self) -> float:
        return self.marker_length if self.marker_length_y is None else self.marker_length_y

    @property
    def anisotropic(self) -> bool:
        return abs(self.sq_y - self.square_length) > 1e-12 or abs(self.mk_y - self.marker_length) > 1e-12

    def marker_squares(self) -> List[Tuple[int, int]]:
        if self.type == "aruco":
            return []
        return [(i, j) for j in range(self.squares_y) for i in range(self.squares_x) if (i + j) % 2 == 1]

    def n_markers(self) -> int:
        return 1 if self.type == "aruco" else len(self.marker_squares())

    def board_size(self) -> Tuple[float, float]:
        if self.type == "aruco":
            return self.marker_length, self.mk_y
        return self.squares_x * self.square_length, self.squares_y * self.sq_y

    def marker_corners_obj(self, k: int) -> np.ndarray:
        """4x3 object points (TL, TR, BR, BL) of the k-th marker."""
        ml, mly = self.marker_length, self.mk_y
        if self.type == "aruco":
            x0, y0 = -ml / 2.0, -mly / 2.0
        else:
            i, j = self.marker_squares()[k]
            w, h = self.board_size()
            x0 = -w / 2.0 + i * self.square_length + (self.square_length - ml) / 2.0
            y0 = -h / 2.0 + j * self.sq_y + (self.sq_y - mly) / 2.0
        return np.array([[x0, y0, 0], [x0 + ml, y0, 0], [x0 + ml, y0 + mly, 0], [x0, y0 + mly, 0]], dtype=float)

    def chess_corners_obj(self) -> List[Tuple[int, np.ndarray]]:
        """Inner chessboard corners as (point_id, xyz)."""
        if self.type == "aruco":
            return []
        w, h = self.board_size()
        out = []
        n = 0
        for j in range(1, self.squares_y):
            for i in range(1, self.squares_x):
                out.append((CHESS_ID_BASE + n,
                            np.array([-w / 2.0 + i * self.square_length, -h / 2.0 + j * self.sq_y, 0.0])))
                n += 1
        return out

    def touch_corner_ids(self) -> set:
        """Chessboard point ids lying under a printed touch crosshair (not used as image observations:
        the crosshair is drawn over the saddle the corner refinement looks at)."""
        if self.type == "aruco" or not self.touch_points:
            return set()
        w, h = self.board_size()
        out = set()
        for pid, p in self.chess_corners_obj():
            if any(abs(p[0] - xy[0]) < 1e-9 and abs(p[1] - xy[1]) < 1e-9 for xy in self.touch_points.values()):
                out.add(pid)
        return out

    def all_points_obj(self) -> Dict[int, np.ndarray]:
        pts = {}
        for k in range(self.n_markers()):
            for c, p in enumerate(self.marker_corners_obj(k)):
                pts[4 * k + c] = p
        for pid, p in self.chess_corners_obj():
            pts[pid] = p
        return pts

    def describe(self) -> str:
        if self.type == "aruco":
            return (f"{self.arm}: single ArUco {self.dictionary} id {self.marker_ids[0]}, "
                    f"{self.marker_length * 1e3:.1f}" + (f" x {self.mk_y * 1e3:.1f}" if self.anisotropic else "")
                    + " mm")
        w, h = self.board_size()
        aniso = self.anisotropic
        return (f"{self.arm}: ChArUco {self.squares_x}x{self.squares_y} {self.dictionary}, "
                f"square {self.square_length * 1e3:.2f}" + (f" x {self.sq_y * 1e3:.2f}" if aniso else "")
                + f" mm, marker {self.marker_length * 1e3:.2f}" + (f" x {self.mk_y * 1e3:.2f}" if aniso else "")
                + f" mm, "
                f"board {w * 1e3:.0f}x{h * 1e3:.0f} mm, marker ids {self.marker_ids[0]}..{self.marker_ids[-1]}")


def _spec_from_dict(arm: str, d: dict) -> TargetSpec:
    typ = str(d.get("type", "charuco")).lower()
    if typ not in ("charuco", "aruco"):
        raise ValueError(f"{arm}: unknown target type '{typ}'")
    dictionary = str(d.get("dictionary", "DICT_4X4_50"))
    sq_y = float(d["square_length_y"]) if d.get("square_length_y") is not None else None
    mk_y = float(d["marker_length_y"]) if d.get("marker_length_y") is not None else None
    if typ == "aruco":
        mid = d.get("marker_id", d.get("ids", [d.get("first_marker_id", 0)]))
        ids = [int(mid[0] if isinstance(mid, (list, tuple)) else mid)]
        spec = TargetSpec(arm, typ, dictionary, float(d["marker_length"]), ids, marker_length_y=mk_y)
    else:
        spec = TargetSpec(arm, typ, dictionary, float(d["marker_length"]), [],
                          float(d["square_length"]), int(d["squares_x"]), int(d["squares_y"]),
                          square_length_y=sq_y, marker_length_y=mk_y)
        n = spec.n_markers()
        if "ids" in d:
            ids = [int(v) for v in d["ids"]]
        else:
            first = int(d.get("first_marker_id", 0))
            ids = list(range(first, first + n))
        if len(ids) != n:
            raise ValueError(f"{arm}: board needs {n} marker ids, got {len(ids)}")
        spec.marker_ids = ids
        if spec.marker_length >= spec.square_length or spec.mk_y >= spec.sq_y:
            raise ValueError(f"{arm}: marker_length must be smaller than square_length (x and y)")
    spec.paper = str(d.get("paper", ""))
    spec.touch_points = _parse_touch_points(spec, d.get("touch_points", []))
    return spec


def _parse_touch_points(spec: TargetSpec, entries) -> Dict[int, Tuple[float, float]]:
    """Touch points: ``{id: n, corner: [i, j]}`` (inner chessboard corner, 1-based) or ``{id: n, xy: [x, y]}`` (m)."""
    out: Dict[int, Tuple[float, float]] = {}
    for e in entries or []:
        pid = int(e["id"])
        if pid <= 0 or pid in out:
            raise ValueError(f"{spec.arm}: touch point id {pid} must be a unique positive integer")
        if "corner" in e:
            if spec.type != "charuco":
                raise ValueError(f"{spec.arm}: touch point 'corner' needs a ChArUco board")
            i, j = (int(v) for v in e["corner"])
            if not (1 <= i < spec.squares_x and 1 <= j < spec.squares_y):
                raise ValueError(f"{spec.arm}: touch point {pid} corner {[i, j]} is not an inner corner")
            w, h = spec.board_size()
            out[pid] = (-w / 2.0 + i * spec.square_length, -h / 2.0 + j * spec.sq_y)
        elif "xy" in e:
            out[pid] = (float(e["xy"][0]), float(e["xy"][1]))
        else:
            raise ValueError(f"{spec.arm}: touch point {pid} needs 'corner' or 'xy'")
    return out


def validate_specs(specs: Dict[str, TargetSpec]) -> None:
    """Raise ValueError unless all marker IDs are valid and disjoint between arms."""
    seen: Dict[Tuple[str, int], str] = {}
    for arm, s in specs.items():
        cap = dictionary_capacity(s.dictionary)
        if len(set(s.marker_ids)) != len(s.marker_ids):
            raise ValueError(f"{arm}: duplicate marker ids")
        for mid in s.marker_ids:
            if not 0 <= mid < cap:
                raise ValueError(f"{arm}: marker id {mid} outside dictionary {s.dictionary} (0..{cap - 1})")
            key = (s.dictionary, mid)
            if key in seen:
                raise ValueError(f"marker id {mid} ({s.dictionary}) used by both '{seen[key]}' and '{arm}': "
                                 "the two arms must have disjoint marker ids")
            seen[key] = arm


def specs_from_config(cfg: dict) -> Dict[str, TargetSpec]:
    arms = cfg.get("arms", cfg)
    specs = {str(a): _spec_from_dict(str(a), d) for a, d in arms.items()
             if isinstance(d, dict) and str(a) != "table_board"}
    if not specs:
        raise ValueError("no targets in config")
    validate_specs(specs)
    return specs


def load_targets(path: str) -> Dict[str, TargetSpec]:
    """The per-arm targets of a targets YAML (the optional ``table_board`` entry is NOT included)."""
    with open(path) as f:
        return specs_from_config(yaml.safe_load(f))


def table_from_config(cfg: dict) -> Optional[TargetSpec]:
    """The fixed table board (``table_board`` entry), or None.  Its marker ids must be disjoint from both arms."""
    d = cfg.get("table_board")
    if not isinstance(d, dict):
        return None
    spec = _spec_from_dict("table", d)
    if spec.type != "charuco":
        raise ValueError("table_board must be a ChArUco board")
    validate_specs({**specs_from_config(cfg), "table": spec})
    return spec


@dataclass
class TouchStickers:
    """Free crosshair stickers placed anywhere both arms reach; their positions are NOT known in any board frame.

    ``fit_points``   touched by both arms and used (together with the board crosshairs) in the touch-only Kabsch base
                     to widen its spread; never in the joint solve.
    ``check_points`` touched by both arms and NEVER used in any fit: the held-out check-point (TRE) metric
                     ||T_world_left p_left - T_world_right p_right|| of the gate."""
    fit_points: List[int] = field(default_factory=list)
    check_points: List[int] = field(default_factory=list)

    def kind(self, pid: int) -> str:
        if pid in self.check_points:
            return "check"
        if pid in self.fit_points:
            return "fit"
        return ""

    def all_ids(self) -> List[int]:
        return sorted(set(self.fit_points) | set(self.check_points))

    def to_config(self) -> dict:
        return {"fit_points": list(self.fit_points), "check_points": list(self.check_points)}


def stickers_from_config(cfg: dict, table: Optional[TargetSpec] = None) -> TouchStickers:
    """Parse the optional ``touch_stickers`` section; ids must be unique, positive and disjoint from the table
    board's printed touch points."""
    d = cfg.get("touch_stickers") or {}
    out = TouchStickers([int(v) for v in d.get("fit_points", [])], [int(v) for v in d.get("check_points", [])])
    allv = out.fit_points + out.check_points
    if len(set(allv)) != len(allv):
        raise ValueError("touch_stickers: fit_points and check_points must be unique and disjoint")
    if any(v <= 0 for v in allv):
        raise ValueError("touch_stickers: ids must be positive integers (use ids >= 100)")
    if table is not None and set(allv) & set(table.touch_points):
        raise ValueError("touch_stickers: ids collide with the table board's touch point ids "
                         f"{sorted(set(allv) & set(table.touch_points))}")
    return out


def load_stickers(path: str) -> TouchStickers:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return stickers_from_config(cfg, table_from_config(cfg))


# --------------------------------------------------------------------------
# Hover sheet (approach assist): one ArUco marker with printed touch crosshairs
# --------------------------------------------------------------------------
DEFAULT_DESIGN_MARKER_LENGTH = 0.060     # m: marker edge the sheet was DESIGNED with (aruco_tcp_hover_test_A4.pdf)


@dataclass
class HoverSheetSpec:
    """The operator's hover sheet: one ArUco marker (default DICT_4X4_50 id 13) and crosshairs printed at fixed
    positions around it.

    Marker frame (same as compal_box_perception/aruco_verification.py): origin = marker centre, x to the right,
    y towards "TOP" as printed (UP as read), z = x cross y = OUT of the paper towards the viewer.  The detector's
    corner order TL, TR, BR, BL maps to (-h, +hy), (h, +hy), (h, -hy), (-h, -hy).

    ``points`` are the crosshair positions in that frame in metres, ALREADY scaled to the measured print:
    ``xy_designed * (marker_length / design_marker_length, marker_length_y / design_marker_length)``.  The crosshairs
    are printed at absolute millimetres on the design sheet, so a print that is stretched (the operator measures the
    marker edge along x and y) moves them by the same factors as the marker."""
    marker_id: int
    dictionary: str
    marker_length: float                       # m, x edge (measured)
    marker_length_y: float                     # m, y edge (measured; default = marker_length)
    points: Dict[int, Tuple[float, float]] = field(default_factory=dict)   # touch id -> (x, y) m, marker frame
    names: Dict[int, str] = field(default_factory=dict)                    # touch id -> "TOP", "P2", ...
    design_marker_length: float = DEFAULT_DESIGN_MARKER_LENGTH
    # ---- tile fields (``hover_tiles``); the legacy single sheet leaves them at their defaults ----
    label: str = ""                                    # tile label ("A", "B", ...); "" = the legacy ``hover_sheet``
    tile_size: float = 0.0                             # m, DESIGNED tile edge (cut line); 0 = not a tile
    tile_center_mm: Tuple[float, float] = (0.0, 0.0)   # tile centre in the marker frame, DESIGN mm (y towards TOP)
    points_mm: Dict[int, Tuple[float, float]] = field(default_factory=dict)   # touch id -> DESIGN mm, marker frame

    @property
    def source(self) -> str:
        """Name of this target in ``hover.detect_targets`` statuses and ``TargetPoint.source``: "tile A" / "sheet"."""
        return f"tile {self.label}" if self.label else "sheet"

    @property
    def anisotropic(self) -> bool:
        return abs(self.marker_length_y - self.marker_length) > 1e-12

    def marker_corners_obj(self) -> np.ndarray:
        """4x3 object points in the order the detector returns the corners (TL, TR, BR, BL), y up."""
        h, hy = self.marker_length / 2.0, self.marker_length_y / 2.0
        return np.array([[-h, hy, 0.0], [h, hy, 0.0], [h, -hy, 0.0], [-h, -hy, 0.0]], dtype=float)

    def describe(self) -> str:
        size = f"{self.marker_length * 1e3:.2f}" + (f" x {self.marker_length_y * 1e3:.2f}" if self.anisotropic else "")
        what = f"hover {self.source}"
        return (f"{what}: ArUco {self.dictionary} id {self.marker_id}, {size} mm, "
                f"points {', '.join(f'{pid}={self.names.get(pid, pid)}' for pid in sorted(self.points))}")


@dataclass
class HoverTilesSpec:
    """The hover tiles of one print (``hover_tiles`` section): small ArUco tiles, each with its own marker and 3 touch
    crosshairs.  ``marker_length(_y)`` are the MEASURED values (one for the whole print); every tile in ``tiles`` is a
    ``HoverSheetSpec`` (label set) whose points are already scaled by them."""
    dictionary: str
    tile_size: float                      # m, designed tile edge (cut line)
    design_marker_length: float           # m
    marker_length: float                  # m, measured x
    marker_length_y: float                # m, measured y
    tiles: List[HoverSheetSpec] = field(default_factory=list)

    @property
    def anisotropic(self) -> bool:
        return abs(self.marker_length_y - self.marker_length) > 1e-12

    def describe(self) -> str:
        size = f"{self.marker_length * 1e3:.2f}" + (f" x {self.marker_length_y * 1e3:.2f}" if self.anisotropic else "")
        return (f"hover tiles ({self.dictionary}, tile {self.tile_size * 1e3:.0f} mm, marker {size} mm measured / "
                f"{self.design_marker_length * 1e3:.0f} mm designed): "
                + "; ".join(f"{t.source}: marker {t.marker_id}, points {', '.join(str(p) for p in sorted(t.points))}"
                            for t in self.tiles))


def _finite(v, what: str, section: str = "hover_sheet") -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"{section}: {what} must be a number (got {v!r})") from None
    if not np.isfinite(f):
        raise ValueError(f"{section}: {what} must be finite (got {v!r})")
    return f


def hover_sheet_from_config(cfg: dict) -> Optional[HoverSheetSpec]:
    """The optional ``hover_sheet`` section, or None.  Validates: the marker id is inside the dictionary and not used
    by an arm board or the table board; the point ids are positive, unique and disjoint from the table board's touch
    point ids (ids that are also ``touch_stickers`` are allowed: that is the intent); all numbers finite."""
    d = cfg.get("hover_sheet")
    if not isinstance(d, dict):
        return None
    dictionary = str(d.get("dictionary", "DICT_4X4_50"))
    cap = dictionary_capacity(dictionary)
    if "marker_id" not in d:
        raise ValueError("hover_sheet: marker_id is required")
    mid = int(d["marker_id"])
    if not 0 <= mid < cap:
        raise ValueError(f"hover_sheet: marker id {mid} outside dictionary {dictionary} (0..{cap - 1})")
    ml = _finite(d.get("marker_length"), "marker_length")
    mly = _finite(d.get("marker_length_y", ml), "marker_length_y")
    design = _finite(d.get("design_marker_length", DEFAULT_DESIGN_MARKER_LENGTH), "design_marker_length")
    if ml <= 0 or mly <= 0 or design <= 0:
        raise ValueError("hover_sheet: marker_length, marker_length_y and design_marker_length must be > 0")
    # marker id disjoint from every board in the same file (same dictionary only)
    table = table_from_config(cfg)
    boards = dict(specs_from_config(cfg))
    if table is not None:
        boards["table"] = table
    used: Dict[int, str] = {}
    for arm, s in boards.items():
        if s.dictionary == dictionary:
            for m in s.marker_ids:
                used[m] = arm
    if mid in used:
        raise ValueError(f"hover_sheet: marker id {mid} ({dictionary}) is already used by '{used[mid]}': the hover "
                         "sheet marker must be disjoint from the arm boards and the table board")
    pts: Dict[int, Tuple[float, float]] = {}
    pts_mm: Dict[int, Tuple[float, float]] = {}
    names: Dict[int, str] = {}
    entries = d.get("points") or {}
    if isinstance(entries, list):                      # also accept [{id: 101, name: TOP, xy_mm: [0, 60]}, ...]
        entries = {e["id"]: e for e in entries}
    for key, e in entries.items():
        pid = int(key)
        if pid <= 0 or pid in pts:
            raise ValueError(f"hover_sheet: point id {pid} must be a unique positive integer")
        if table is not None and pid in table.touch_points:
            raise ValueError(f"hover_sheet: point id {pid} collides with the table board's touch point ids "
                             f"{sorted(table.touch_points)}")
        if not isinstance(e, dict) or "xy_mm" not in e:
            raise ValueError(f"hover_sheet: point {pid} needs 'xy_mm: [x, y]' (mm from the marker centre, y towards TOP)")
        xy = e["xy_mm"]
        if not isinstance(xy, (list, tuple)) or len(xy) != 2:
            raise ValueError(f"hover_sheet: point {pid} xy_mm must be [x, y]")
        x, y = _finite(xy[0], f"point {pid} x"), _finite(xy[1], f"point {pid} y")
        pts[pid] = (x * MM * ml / design, y * MM * mly / design)
        pts_mm[pid] = (x, y)
        names[pid] = str(e.get("name", f"P{pid}"))
    if not pts:
        raise ValueError("hover_sheet: no points")
    return HoverSheetSpec(mid, dictionary, ml, mly, pts, names, design, points_mm=pts_mm)


def load_hover_sheet(path: str) -> Optional[HoverSheetSpec]:
    """The ``hover_sheet`` section of a targets YAML, or None if absent."""
    with open(path) as f:
        return hover_sheet_from_config(yaml.safe_load(f))


# --------------------------------------------------------------------------
# Hover tiles (approach assist): several small ArUco tiles, each with 3 touch crosshairs
# --------------------------------------------------------------------------
DEFAULT_TILE_SIZE = 0.100                  # m: printed tile edge (cut line)
DEFAULT_TILE_MARKER = 0.050                # m: designed marker edge
MIN_MARKER_EDGE_MM = 8.0                   # a crosshair is at least this far from the marker's outer edge ...
MIN_TILE_EDGE_MM = 6.0                     # ... and from the tile's cut edge
MIN_POINT_SEPARATION_MM = 15.0             # ... and from every other crosshair of the tile
MIN_MARKER_TILE_MARGIN_MM = 3.0            # the marker's outer edge is at least this far inside the tile


def _tile_geometry_errors(label: str, pts_mm: Dict[int, Tuple[float, float]], design_marker_mm: float,
                          tile_size_mm: float, centre_mm: Tuple[float, float]) -> List[str]:
    """Why a tile layout is not printable / usable (design millimetres, marker frame: origin = marker centre, y towards
    the tile's TOP); [] when it is fine."""
    errs: List[str] = []
    hm, ht = design_marker_mm / 2.0, tile_size_mm / 2.0
    cx, cy = centre_mm
    margin = min(ht - abs(cx) - hm, ht - abs(cy) - hm)          # marker edge to the nearest tile edge
    if margin < MIN_MARKER_TILE_MARGIN_MM - 1e-9:
        errs.append(f"tile {label}: the marker is only {margin:.1f} mm from the tile edge (need >= "
                    f"{MIN_MARKER_TILE_MARGIN_MM:g} mm; marker {design_marker_mm:g} mm, tile {tile_size_mm:g} mm, "
                    f"tile_center {[cx, cy]})")
    for pid, (x, y) in sorted(pts_mm.items()):
        edge = min(ht - abs(x - cx), ht - abs(y - cy))
        if edge < MIN_TILE_EDGE_MM - 1e-9:
            where = "is not on the tile" if edge < 0 else f"is only {edge:.1f} mm from the tile edge"
            errs.append(f"tile {label}: crosshair {pid} at ({x:g}, {y:g}) mm {where} (need >= {MIN_TILE_EDGE_MM:g} mm)")
        dmark = float(np.hypot(max(abs(x) - hm, 0.0), max(abs(y) - hm, 0.0)))
        if dmark < MIN_MARKER_EDGE_MM - 1e-9:
            errs.append(f"tile {label}: crosshair {pid} at ({x:g}, {y:g}) mm is only {dmark:.1f} mm from the marker edge "
                        f"(need >= {MIN_MARKER_EDGE_MM:g} mm)")
    ids = sorted(pts_mm)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            dd = float(np.hypot(pts_mm[a][0] - pts_mm[b][0], pts_mm[a][1] - pts_mm[b][1]))
            if dd < MIN_POINT_SEPARATION_MM - 1e-9:
                errs.append(f"tile {label}: crosshairs {a} and {b} are only {dd:.1f} mm apart (need >= "
                            f"{MIN_POINT_SEPARATION_MM:g} mm)")
    return errs


def _xy_pair(v, what: str, section: str = "hover_tiles") -> Tuple[float, float]:
    if not isinstance(v, (list, tuple)) or len(v) != 2:
        raise ValueError(f"{section}: {what} must be [x, y] in mm")
    return _finite(v[0], f"{what} x", section), _finite(v[1], f"{what} y", section)


def hover_tiles_from_config(cfg: dict) -> Optional[HoverTilesSpec]:
    """The optional ``hover_tiles`` section, or None.

    ``tile_size`` / ``design_marker_length`` are the DESIGN values (m); ``marker_length`` / ``marker_length_y`` the MEASURED
    marker edge of the print (default: the design length), one pair for all tiles of one print.  Per tile: ``marker_id``,
    ``label``, ``points: {touch id: [x_mm, y_mm]}`` in the marker frame (origin = marker centre, x right, y towards the
    tile's TOP as printed; design millimetres, scaled by ``marker_length(_y) / design_marker_length`` like the sheet) and
    the optional ``tile_center: [x_mm, y_mm]`` (the tile centre in the marker frame, default [0, 0] = marker centred).

    Validation: marker ids in the dictionary, unique, disjoint from every arm / table board marker (and from a legacy
    ``hover_sheet``); point ids positive, unique across the tiles, disjoint from the table touch ids (and the legacy sheet);
    the marker lies >= 3 mm inside the tile; every crosshair is on the tile >= 6 mm from its edge, >= 8 mm from the marker
    edge and >= 15 mm from the other crosshairs of its tile; all numbers finite."""
    d = cfg.get("hover_tiles")
    if not isinstance(d, dict):
        return None
    sec = "hover_tiles"
    dictionary = str(d.get("dictionary", "DICT_4X4_50"))
    cap = dictionary_capacity(dictionary)
    tile_size = _finite(d.get("tile_size", DEFAULT_TILE_SIZE), "tile_size", sec)
    design = _finite(d.get("design_marker_length", DEFAULT_TILE_MARKER), "design_marker_length", sec)
    ml = _finite(d.get("marker_length", design), "marker_length", sec)
    mly = _finite(d.get("marker_length_y", ml), "marker_length_y", sec)
    if min(tile_size, design, ml, mly) <= 0:
        raise ValueError(f"{sec}: tile_size, design_marker_length, marker_length and marker_length_y must be > 0")
    entries = d.get("tiles")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{sec}: 'tiles' must be a non-empty list")
    table = table_from_config(cfg)
    boards = dict(specs_from_config(cfg))
    if table is not None:
        boards["table"] = table
    used: Dict[int, str] = {}
    for arm, s_ in boards.items():
        if s_.dictionary == dictionary:
            for m in s_.marker_ids:
                used[m] = arm
    seen_marker: Dict[int, str] = {}
    seen_point: Dict[int, str] = {}
    labels: List[str] = []
    tiles: List[HoverSheetSpec] = []
    for k, e in enumerate(entries):
        if not isinstance(e, dict):
            raise ValueError(f"{sec}: tile {k + 1} must be a mapping with marker_id, label and points")
        label = e.get("label", chr(ord("A") + k) if k < 26 else str(k + 1))
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"{sec}: tile {k + 1} label must be text (quote it: 'Y' / 'N' are booleans in YAML), got {label!r}")
        label = label.strip()
        if label in labels:
            raise ValueError(f"{sec}: duplicate tile label '{label}'")
        labels.append(label)
        if "marker_id" not in e:
            raise ValueError(f"{sec}: tile {label}: marker_id is required")
        mid = int(e["marker_id"])
        if not 0 <= mid < cap:
            raise ValueError(f"{sec}: tile {label}: marker id {mid} outside dictionary {dictionary} (0..{cap - 1})")
        if mid in used:
            raise ValueError(f"{sec}: tile {label}: marker id {mid} ({dictionary}) is already used by '{used[mid]}': the tile "
                             "markers must be disjoint from the arm boards and the table board")
        if mid in seen_marker:
            raise ValueError(f"{sec}: marker id {mid} is used by tile {seen_marker[mid]} and tile {label}: marker ids must be unique")
        seen_marker[mid] = label
        centre = _xy_pair(e.get("tile_center", [0.0, 0.0]), f"tile {label} tile_center", sec)
        raw = e.get("points")
        if not isinstance(raw, dict) or not raw:
            raise ValueError(f"{sec}: tile {label}: no points (points: {{id: [x_mm, y_mm], ...}})")
        pts_mm: Dict[int, Tuple[float, float]] = {}
        for key, xy in raw.items():
            pid = int(key)
            if pid <= 0:
                raise ValueError(f"{sec}: tile {label}: point id {pid} must be a positive integer")
            if pid in seen_point:
                raise ValueError(f"{sec}: point id {pid} is used by tile {seen_point[pid]} and tile {label}: point ids must be unique")
            if table is not None and pid in table.touch_points:
                raise ValueError(f"{sec}: tile {label}: point id {pid} collides with the table board's touch point ids "
                                 f"{sorted(table.touch_points)}")
            seen_point[pid] = label
            pts_mm[pid] = _xy_pair(xy, f"tile {label} point {pid}", sec)
        errs = _tile_geometry_errors(label, pts_mm, design * 1e3, tile_size * 1e3, centre)
        if errs:
            raise ValueError(f"{sec}: " + "; ".join(errs))
        order = sorted(pts_mm)
        tiles.append(HoverSheetSpec(
            mid, dictionary, ml, mly,
            {pid: (x * MM * ml / design, y * MM * mly / design) for pid, (x, y) in pts_mm.items()},
            {pid: f"{label}{i + 1}" for i, pid in enumerate(order)}, design,
            label=label, tile_size=tile_size, tile_center_mm=centre, points_mm=dict(pts_mm)))
    return HoverTilesSpec(dictionary, tile_size, design, ml, mly, tiles)


def load_hover_tiles(path: str) -> Optional[HoverTilesSpec]:
    """The ``hover_tiles`` section of a targets YAML, or None if absent."""
    with open(path) as f:
        return hover_tiles_from_config(yaml.safe_load(f))


def hover_sheets_from_config(cfg: dict) -> List[HoverSheetSpec]:
    """Every hover target of a targets file as a list of ``HoverSheetSpec``: the tiles of ``hover_tiles`` (sources "tile A",
    ...) followed by the legacy ``hover_sheet`` (source "sheet") if both are given - their marker and point ids must not
    clash.  [] when neither section exists."""
    tiles = hover_tiles_from_config(cfg)
    sheet = hover_sheet_from_config(cfg)
    out: List[HoverSheetSpec] = list(tiles.tiles) if tiles is not None else []
    if sheet is not None:
        for t in out:
            if t.marker_id == sheet.marker_id and t.dictionary == sheet.dictionary:
                raise ValueError(f"hover_sheet marker id {sheet.marker_id} is also used by {t.source}")
            both = sorted(set(t.points) & set(sheet.points))
            if both:
                raise ValueError(f"hover_sheet point ids {both} are also used by {t.source}")
        out.append(sheet)
    return out


def load_hover_sheets(path: str) -> List[HoverSheetSpec]:
    """``hover_sheets_from_config`` of a targets YAML."""
    with open(path) as f:
        return hover_sheets_from_config(yaml.safe_load(f))


def load_targets_all(path: str) -> Tuple[Dict[str, TargetSpec], Optional[TargetSpec]]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return specs_from_config(cfg), table_from_config(cfg)


def load_table_board(path: str) -> Optional[TargetSpec]:
    return load_targets_all(path)[1]


def specs_to_config(specs: Dict[str, TargetSpec]) -> dict:
    out = {}
    for arm, s in specs.items():
        d = {"type": s.type, "dictionary": s.dictionary, "marker_length": s.marker_length}
        if s.marker_length_y is not None:
            d["marker_length_y"] = s.marker_length_y
        if s.type == "aruco":
            d["marker_id"] = s.marker_ids[0]
        else:
            d.update(squares_x=s.squares_x, squares_y=s.squares_y, square_length=s.square_length,
                     ids=list(s.marker_ids))
            if s.square_length_y is not None:
                d["square_length_y"] = s.square_length_y
        out[arm] = d
    return {"arms": out}


def table_to_config(spec: TargetSpec) -> dict:
    d = specs_to_config({"table": spec})["arms"]["table"]
    d["touch_points"] = [{"id": pid, "xy": [xy[0], xy[1]]} for pid, xy in sorted(spec.touch_points.items())]
    if spec.paper:
        d["paper"] = spec.paper
    return d


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------
@dataclass
class Detection:
    arm: str
    points: List[Tuple[int, Tuple[float, float, float], Tuple[float, float]]] = field(default_factory=list)
    n_markers: int = 0
    n_chess: int = 0

    def arrays(self):
        ids = np.array([p[0] for p in self.points], dtype=int)
        obj = np.array([p[1] for p in self.points], dtype=float).reshape(-1, 3)
        pix = np.array([p[2] for p in self.points], dtype=float).reshape(-1, 2)
        return ids, obj, pix


class TargetDetector:
    def __init__(self, specs: Dict[str, TargetSpec], min_marker_perimeter_rate: float = 0.02,
                 refine_chess: bool = True):
        validate_specs(specs)
        self.specs = specs
        self.refine_chess = refine_chess
        self._params = make_detector_params(min_marker_perimeter_rate)
        self._dicts = {s.dictionary: get_dictionary(s.dictionary) for s in specs.values()}
        self._new_api = hasattr(cv2.aruco, "ArucoDetector")
        self._detectors = {}
        if self._new_api:
            self._detectors = {n: cv2.aruco.ArucoDetector(d, self._params) for n, d in self._dicts.items()}
        self._id_to_k = {a: {mid: k for k, mid in enumerate(s.marker_ids)} for a, s in specs.items()}

    def _detect_markers(self, gray, dict_name):
        if self._new_api:
            corners, ids, _ = self._detectors[dict_name].detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self._dicts[dict_name], parameters=self._params)
        if ids is None or len(ids) == 0:
            return {}
        out = {}
        for c, i in zip(corners, ids.reshape(-1)):
            out.setdefault(int(i), refine_marker_corners(gray, c))
        return out

    def detect(self, image: np.ndarray) -> Dict[str, Detection]:
        """Return {arm: Detection} for every arm whose target is visible (>= 1 marker)."""
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        by_dict = {name: self._detect_markers(gray, name) for name in self._dicts}
        result: Dict[str, Detection] = {}
        for arm, spec in self.specs.items():
            found = by_dict[spec.dictionary]
            det = Detection(arm)
            marker_obj, marker_pix = [], []
            for mid, k in self._id_to_k[arm].items():
                if mid not in found:
                    continue
                det.n_markers += 1
                objc = spec.marker_corners_obj(k)
                for c in range(4):
                    pid = 4 * k + c
                    det.points.append((pid, tuple(objc[c]), tuple(found[mid][c])))
                marker_obj.append(objc[:, :2])
                marker_pix.append(found[mid])
            if det.n_markers == 0:
                continue
            if spec.type == "charuco" and self.refine_chess and det.n_markers >= 2:
                chess = self._refine_chess(gray, spec, np.vstack(marker_obj), np.vstack(marker_pix))
                skip = spec.touch_corner_ids()
                chess = [c for c in chess if c[0] not in skip]
                for pid, xyz, uv in chess:
                    det.points.append((pid, tuple(xyz), tuple(uv)))
                det.n_chess = len(chess)
            result[arm] = det
        return result

    @staticmethod
    def _refine_chess(gray, spec: TargetSpec, obj_xy: np.ndarray, pix: np.ndarray):
        if len(obj_xy) >= 12:
            H, _ = cv2.findHomography(obj_xy, pix, cv2.RANSAC, 3.0)
        else:
            H, _ = cv2.findHomography(obj_xy, pix, 0)
        if H is None:
            return []
        corners = spec.chess_corners_obj()
        if not corners:
            return []
        xy = np.array([[c[1][0], c[1][1]] for c in corners], dtype=float)
        proj = cv2.perspectiveTransform(xy.reshape(-1, 1, 2), H).reshape(-1, 2)
        p0 = cv2.perspectiveTransform(xy[:1].reshape(-1, 1, 2), H).reshape(2)
        steps = np.array([[spec.square_length, 0.0], [0.0, spec.sq_y]]) + xy[:1]
        p1 = cv2.perspectiveTransform(steps.reshape(-1, 1, 2), H).reshape(-1, 2)
        spacing_x, spacing_y = (float(np.linalg.norm(q - p0)) for q in p1)
        spacing = min(spacing_x, spacing_y)
        # The saddle's neighbourhood must stay inside the square's white/black fields;
        # the nearest marker border is (square - marker) / 2 away (x and y).
        half_gap = min(0.5 * (spec.square_length - spec.marker_length) / spec.square_length * spacing_x,
                       0.5 * (spec.sq_y - spec.mk_y) / spec.sq_y * spacing_y)
        win = int(max(2, min(0.35 * spacing, 0.8 * half_gap + 1.0, 8)))
        max_shift = max(2.5, 0.25 * spacing)
        h, w = gray.shape[:2]
        out = []
        valid = [(k, p) for k, p in enumerate(proj) if win + 2 < p[0] < w - win - 3 and win + 2 < p[1] < h - win - 3]
        if not valid:
            return []
        init = np.array([p for _, p in valid], dtype=np.float32).reshape(-1, 1, 2)
        refined = init.copy()
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3)
        cv2.cornerSubPix(gray, refined, (win, win), (-1, -1), crit)
        for (k, _), a, b in zip(valid, init.reshape(-1, 2), refined.reshape(-1, 2)):
            if np.linalg.norm(a - b) > max_shift:
                continue
            pid, xyz = corners[k]
            out.append((pid, xyz, (float(b[0]), float(b[1]))))
        return out


# --------------------------------------------------------------------------
# Rendering / print-ready output
# --------------------------------------------------------------------------
def render_board(spec: TargetSpec, px_per_m: float, margin: float = 0.0) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Render the target as a uint8 image (255 = white).

    Returns (image, origin_px) where origin_px is the pixel position (x, y) of the
    target-frame origin (board centre) so callers can map object points to pixels:
    u = ox + x*px_per_m, v = oy + y*px_per_m.  Cell edges are rounded individually
    so there are no gaps; the worst-case scale error is 0.5 px per edge.
    """
    w, h = spec.board_size()
    W = int(round((w + 2 * margin) * px_per_m))
    H = int(round((h + 2 * margin) * px_per_m))
    img = np.full((H, W), 255, np.uint8)
    ox, oy = (w / 2.0 + margin) * px_per_m, (h / 2.0 + margin) * px_per_m

    def fill(x0, y0, x1, y1, val):
        a, b = int(round(ox + x0 * px_per_m)), int(round(oy + y0 * px_per_m))
        c, d = int(round(ox + x1 * px_per_m)), int(round(oy + y1 * px_per_m))
        img[b:d, a:c] = val

    if spec.type == "charuco":
        sq, sqy = spec.square_length, spec.sq_y
        for j in range(spec.squares_y):
            for i in range(spec.squares_x):
                if (i + j) % 2 == 0:
                    x0, y0 = -w / 2 + i * sq, -h / 2 + j * sqy
                    fill(x0, y0, x0 + sq, y0 + sqy, 0)
    n = dictionary_marker_size(spec.dictionary)
    cell, celly = spec.marker_length / (n + 2), spec.mk_y / (n + 2)
    for k in range(spec.n_markers()):
        bits = marker_bits(spec.dictionary, spec.marker_ids[k])
        tl = spec.marker_corners_obj(k)[0]
        for r in range(n + 2):
            for c in range(n + 2):
                fill(tl[0] + c * cell, tl[1] + r * celly, tl[0] + (c + 1) * cell, tl[1] + (r + 1) * celly,
                     255 if bits[r, c] else 0)
    return img, (ox, oy)


_PAPER_MM = {"A3": (297.0, 420.0), "A4": (210.0, 297.0), "A5": (148.0, 210.0), "LETTER": (215.9, 279.4)}


def _font(px: int):
    from PIL import ImageFont
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/usr/share/fonts/dejavu/DejaVuSans.ttf"):
        if os.path.exists(path):
            return ImageFont.truetype(path, px)
    return ImageFont.load_default()


def _draw_touch_points(d, spec: TargetSpec, origin_px, px_per_mm: float, dpi: int) -> None:
    """Fine diagonal crosshairs (white halo + black core, visible on black and white squares) at the touch
    points of the table board, with the point number printed in the white gap of the neighbouring marker square."""
    font = _font(max(8, int(3.0 * px_per_mm)))
    ox, oy = origin_px
    arm = 3.5 * px_per_mm
    core = max(1, int(round(0.15 * px_per_mm)))
    halo = max(core + 2, int(round(0.55 * px_per_mm)))
    w, h = spec.board_size()
    for pid, (x, y) in sorted(spec.touch_points.items()):
        cx, cy = ox + x * 1e3 * px_per_mm, oy + y * 1e3 * px_per_mm
        for width, fill in ((halo, 255), (core, 0)):
            d.line([cx - arm, cy - arm, cx + arm, cy + arm], fill=fill, width=width)
            d.line([cx - arm, cy + arm, cx + arm, cy - arm], fill=fill, width=width)
        # label: centre of the 5 mm gap of an adjacent marker square (odd parity), else beside the point
        i = (x + w / 2.0) / spec.square_length
        j = (y + h / 2.0) / spec.sq_y
        dx, dy = 1.0, -1.0
        if abs(i - round(i)) < 1e-6 and abs(j - round(j)) < 1e-6:
            i, j = int(round(i)), int(round(j))
            for a in (i - 1, i):
                for b in (j - 1, j):
                    if (a + b) % 2 == 1:
                        dx = -1.0 if a == i - 1 else 1.0
                        dy = -1.0 if b == j - 1 else 1.0
        # inside the 5 mm gap strip along the square's edge, beyond the end of the crosshair arm (3.5 mm)
        tx, ty = cx + dx * 5.2 * px_per_mm, cy + dy * 1.6 * px_per_mm
        text = str(pid)
        bb = d.textbbox((0, 0), text, font=font)
        d.text((tx - (bb[0] + bb[2]) / 2.0, ty - (bb[1] + bb[3]) / 2.0), text, fill=0, font=font)


def render_print_page(spec: TargetSpec, dpi: int = 600, paper: str = "A4", margin_mm: float = 10.0,
                      scale_bar_mm: float = 50.0):
    """Render a print-ready page (PIL grayscale image) at exact physical scale.

    Returns (PIL.Image, info dict).  Portrait: the board is placed top-centre with the scale bar and a label
    block below it.  A board too wide for the portrait page (the 200 mm table board on A4) is put on the
    landscape page instead, top-left, with the scale bar and the (wrapped) labels in a column to its right.
    Pixel size is 25.4/dpi mm, so the page maps 1:1 to paper.
    """
    from PIL import Image, ImageDraw

    pw_mm, ph_mm = _PAPER_MM[paper.upper()]
    px_per_mm = dpi / 25.4
    board, board_origin = render_board(spec, px_per_mm * 1000.0, margin=margin_mm * MM)
    bw, bh = board.shape[1], board.shape[0]
    edge = int(10 * px_per_mm)                       # minimum page margin
    landscape = bw > int(round(pw_mm * px_per_mm)) - 2 * edge
    if landscape:
        pw_mm, ph_mm = ph_mm, pw_mm
        col_mm = pw_mm - 10.0 - bw / px_per_mm - 6.0 - 8.0       # right column: 6 mm gap, 8 mm page edge
        if col_mm < scale_bar_mm + 2.0 or bh > int(round(ph_mm * px_per_mm)) - 16 * px_per_mm - edge:
            raise ValueError(f"board {bw / px_per_mm:.0f} x {bh / px_per_mm:.0f} mm does not fit on {paper}")
    page = Image.new("L", (int(round(pw_mm * px_per_mm)), int(round(ph_mm * px_per_mm))), 255)
    if landscape:
        top, left = int(round(16 * px_per_mm)), edge
    else:
        top, left = int(round(20 * px_per_mm)), (page.size[0] - bw) // 2
    page.paste(Image.fromarray(board), (left, top))
    d = ImageDraw.Draw(page)
    # thin cut line at the outer edge of the white margin
    d.rectangle([left, top, left + bw - 1, top + bh - 1], outline=170, width=max(1, int(dpi / 300)))
    if spec.touch_points:
        _draw_touch_points(d, spec, (left + board_origin[0], top + board_origin[1]), px_per_mm, dpi)

    f_big, f_mid, f_small = _font(int(5.5 * px_per_mm)), _font(int(3.8 * px_per_mm)), _font(int(3.0 * px_per_mm))
    w_mm, h_mm = spec.board_size()
    d.text((int((10 if landscape else 15) * px_per_mm), int((5 if landscape else 6) * px_per_mm)),
           (f"TABLE BOARD (fixed on the work area)  -  ChArUco, {len(spec.touch_points)} touch points"
            if spec.arm == "table" else
            f"ARM {spec.arm.upper()}  -  {'ChArUco' if spec.type == 'charuco' else 'ArUco'} calibration target"),
           fill=0, font=f_big)
    if spec.type == "charuco":
        lines = [f"Dictionary {spec.dictionary}, {spec.squares_x} x {spec.squares_y} squares",
                 f"Nominal square {spec.square_length * 1e3:.2f}" + (f" x {spec.sq_y * 1e3:.2f}" if spec.anisotropic else "")
                 + f" mm, marker {spec.marker_length * 1e3:.2f}"
                 + (f" x {spec.mk_y * 1e3:.2f}" if spec.anisotropic else "") + " mm",
                 f"Board (squares only) {w_mm * 1e3:.1f} x {h_mm * 1e3:.1f} mm, white margin {margin_mm:.0f} mm",
                 f"Marker IDs {spec.marker_ids[0]}..{spec.marker_ids[-1]} ({len(spec.marker_ids)} markers)"]
    else:
        lines = [f"Dictionary {spec.dictionary}, marker ID {spec.marker_ids[0]}",
                 f"Nominal marker side {spec.marker_length * 1e3:.2f} mm (black border outer edge), "
                 f"white margin {margin_mm:.0f} mm"]
    if spec.touch_points:
        lines.append("Touch points (knife tip on the crosshair centre): " + "; ".join(
            f"{pid}: x {xy[0] * 1e3:+.1f} y {xy[1] * 1e3:+.1f} mm" for pid, xy in sorted(spec.touch_points.items())))
    lines += ["MEASURE the printed square/marker with calipers and enter the measured value in",
              "targets.yaml (square_length / marker_length, and square_length_y / marker_length_y: x and y can differ)",
              "before recording. Mount perfectly flat and rigid.",
              "Target frame: origin = board centre, x right, y down (as read), z into the board."]

    if landscape:
        x0 = left + bw + int(6 * px_per_mm)
        y = top + int(10 * px_per_mm)
        text_font, step = f_small, int(4.2 * px_per_mm)
        text_x, text_w = x0, page.size[0] - int(8 * px_per_mm) - x0
    else:
        x0 = left + int(margin_mm * px_per_mm)
        y = top + bh + int(12 * px_per_mm)
        text_font, step = f_mid, int(6 * px_per_mm)
        text_x, text_w = int(15 * px_per_mm), None
    # scale bar with ticks every 10 mm
    bar_h = int(2.5 * px_per_mm)
    d.rectangle([x0, y, x0 + int(round(scale_bar_mm * px_per_mm)), y + bar_h], fill=0)
    for k in range(0, int(scale_bar_mm) + 1, 10):
        xt = x0 + int(round(k * px_per_mm))
        d.line([xt, y - int(3 * px_per_mm), xt, y + bar_h], fill=0, width=max(2, int(dpi / 150)))
        d.text((xt - int(3 * px_per_mm), y - int(9 * px_per_mm)), f"{k}", fill=0, font=f_small)
    if landscape:
        y += bar_h + int(4 * px_per_mm)
        lines = [f"{scale_bar_mm:.0f} mm scale bar: print at 100 % (\"Actual size\", landscape), "
                 "verify with calipers", ""] + lines
    else:
        d.text((x0 + int(round(scale_bar_mm * px_per_mm)) + int(6 * px_per_mm), y - int(7 * px_per_mm)),
               f"{scale_bar_mm:.0f} mm scale bar", fill=0, font=f_small)
        d.text((x0 + int(round(scale_bar_mm * px_per_mm)) + int(6 * px_per_mm), y),
               "print at 100 %, verify with calipers", fill=0, font=f_small)
        y += int(14 * px_per_mm)
    if text_w is not None:                           # greedy word wrap to the column width
        wrapped = []
        for ln in lines:
            cur = ""
            for word in ln.split(" "):
                trial = word if not cur else cur + " " + word
                if cur and d.textlength(trial, font=text_font) > text_w:
                    wrapped.append(cur)
                    cur = word
                else:
                    cur = trial
            wrapped.append(cur)
        lines = wrapped
    for ln in lines:
        d.text((text_x, y), ln, fill=0, font=text_font)
        y += step
    if y > page.size[1]:
        raise ValueError(f"labels of the {spec.arm} board overflow the {paper} page")
    info = dict(dpi=dpi, px_per_mm=px_per_mm, board_px=(bw, bh), board_origin_px=(left, top),
                page_px=page.size, landscape=landscape)
    return page, info


def write_print_files(spec: TargetSpec, out_dir: str, dpi: int = 600, paper: str = "A4", margin_mm: float = 10.0):
    os.makedirs(out_dir, exist_ok=True)
    page, info = render_print_page(spec, dpi=dpi, paper=paper, margin_mm=margin_mm)
    png = os.path.join(out_dir, f"target_{spec.arm}.png")
    pdf = os.path.join(out_dir, f"target_{spec.arm}.pdf")
    page.save(png, dpi=(dpi, dpi))
    page.save(pdf, "PDF", resolution=float(dpi))
    return png, pdf, info


def board_cli(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Generate print-ready calibration target PNG+PDF at exact scale.")
    ap.add_argument("--targets", default=None, help="targets.yaml (default: packaged config/targets.yaml)")
    ap.add_argument("--out-dir", default="./dual_arm_targets")
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--paper", default=None, choices=sorted(_PAPER_MM),
                    help="paper size (default: the board's own 'paper' entry, else A4; boards too wide for portrait are put on the landscape page)")
    ap.add_argument("--margin-mm", type=float, default=10.0, help="white quiet margin around the board")
    args = ap.parse_args(argv)
    path = args.targets or default_targets_path()
    specs, table = load_targets_all(path)
    all_specs = list(specs.values()) + ([table] if table is not None else [])
    for spec in all_specs:
        paper = args.paper or spec.paper or "A4"
        png, pdf, info = write_print_files(spec, args.out_dir, args.dpi, paper, args.margin_mm)
        print(spec.describe())
        print(f"  -> {png}\n  -> {pdf}   ({info['dpi']} dpi, print at 100 %)")
    return 0


# --------------------------------------------------------------------------
# Hover tiles print: all tiles on ONE page at exact scale
# --------------------------------------------------------------------------
TILE_PAGE_EDGE_MM = 3.0            # smallest distance of a tile cut line from the paper edge (sides)
TILE_PAGE_TOP_MM = 5.0
TILE_GAP_MM = 3.0                  # gap between neighbouring tile cut lines
TILE_TEXT_BLOCK_MM = 76.0          # room reserved below the tiles for the scale bar and the instructions


def _draw_tile_crosshair(d, cx: float, cy: float, px_per_mm: float) -> None:
    """A fine '+' (0.15 mm line, 4.5 mm arms, gap at the centre) around a 1 mm radius ring (the knife tip goes there)."""
    arm, gap = 4.5 * px_per_mm, 1.0 * px_per_mm
    core = max(2, int(round(0.15 * px_per_mm)))
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        d.line([cx + dx * gap, cy + dy * gap, cx + dx * arm, cy + dy * arm], fill=0, width=core)
    r = 1.0 * px_per_mm
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=0, width=core)


def _rect_gap(a, b) -> float:
    """Distance between two axis-aligned rectangles (x0, y0, x1, y1); 0 when they overlap."""
    return float(np.hypot(max(a[0] - b[2], b[0] - a[2], 0.0), max(a[1] - b[3], b[1] - a[3], 0.0)))


def render_tiles_page(tiles: HoverTilesSpec, dpi: int = 600, paper: str = "A4", scale_bar_mm: float = 50.0):
    """Print-ready page (PIL grayscale) with ALL hover tiles at exact physical scale (pixel = 25.4/dpi mm).

    Portrait page, tiles in rows of two (cut outline = ``tile_size``), the instructions and a 50 mm scale bar below.  Each
    tile: thin cut outline, the ArUco marker at its DESIGN size and position (``tile_center``), the tile label and marker id,
    and every crosshair (fine cross, 1 mm centre ring) with its touch id next to it.  The page always shows the DESIGN
    geometry, never the measured one.  Returns (PIL.Image, info); ``info["tiles"][label]`` has the page pixel positions
    (``left``, ``top``, ``size``, ``marker_centre``, ``points``)."""
    from PIL import Image, ImageDraw

    pw_mm, ph_mm = _PAPER_MM[paper.upper()]
    ts = tiles.tile_size * 1e3
    px_per_mm = dpi / 25.4
    n = len(tiles.tiles)
    if n == 0:
        raise ValueError("no hover tiles to print")
    cols = 2 if ts * 2 + TILE_GAP_MM + 2 * TILE_PAGE_EDGE_MM <= pw_mm else 1
    cols = min(cols, n)
    rows = -(-n // cols)
    need_w = cols * ts + (cols - 1) * TILE_GAP_MM + 2 * TILE_PAGE_EDGE_MM
    need_h = TILE_PAGE_TOP_MM + rows * ts + (rows - 1) * TILE_GAP_MM + TILE_TEXT_BLOCK_MM
    if ts + 2 * TILE_PAGE_EDGE_MM > pw_mm or need_h > ph_mm:
        raise ValueError(f"{n} tiles of {ts:.0f} mm do not fit on one {paper} page "
                         f"(need {need_w:.0f} x {need_h:.0f} mm, page {pw_mm:.0f} x {ph_mm:.0f} mm)")
    page = Image.new("L", (int(round(pw_mm * px_per_mm)), int(round(ph_mm * px_per_mm))), 255)
    d = ImageDraw.Draw(page)
    f_big, f_mid = _font(int(7.0 * px_per_mm)), _font(int(3.8 * px_per_mm))
    f_small, f_id = _font(int(3.0 * px_per_mm)), _font(int(3.4 * px_per_mm))
    line_w = max(2, int(dpi / 300))
    info_tiles: Dict[str, dict] = {}

    def text_size(text, font):
        bb = d.textbbox((0, 0), text, font=font)
        return (bb[2] - bb[0]) / px_per_mm, (bb[3] - bb[1]) / px_per_mm

    for k, t in enumerate(tiles.tiles):
        r, c = divmod(k, cols)
        in_row = min(cols, n - r * cols)
        left_mm = (pw_mm - (in_row * ts + (in_row - 1) * TILE_GAP_MM)) / 2.0 + c * (ts + TILE_GAP_MM)
        top_mm = TILE_PAGE_TOP_MM + r * (ts + TILE_GAP_MM)
        left, top, size = int(round(left_mm * px_per_mm)), int(round(top_mm * px_per_mm)), int(round(ts * px_per_mm))
        cxm, cym = t.tile_center_mm
        # tile coordinates in mm from the tile's top-left corner: marker frame (x right, y up) -> (u, v)
        def to_mm(x, y):
            return ts / 2.0 - cxm + x, ts / 2.0 + cym - y
        mk = t.design_marker_length * 1e3
        mu, mv = to_mm(0.0, 0.0)
        d.rectangle([left, top, left + size, top + size], outline=90, width=line_w)      # thin cut line
        mspec = TargetSpec("tile", "aruco", t.dictionary, t.design_marker_length, [t.marker_id])
        img, (ox, oy) = render_board(mspec, px_per_mm * 1000.0)
        mx = left + int(round(mu * px_per_mm - ox))
        my = top + int(round(mv * px_per_mm - oy))
        page.paste(Image.fromarray(img), (mx, my))
        marker_centre = (left + mu * px_per_mm, top + mv * px_per_mm)
        occupied = [(mu - mk / 2 - 2.0, mv - mk / 2 - 2.0, mu + mk / 2 + 2.0, mv + mk / 2 + 2.0)]     # marker + 2 mm
        centres = {pid: to_mm(*xy) for pid, xy in t.points_mm.items()}
        pts_px = {}
        for pid, (u, v) in sorted(centres.items()):
            px, py = left + u * px_per_mm, top + v * px_per_mm
            _draw_tile_crosshair(d, px, py, px_per_mm)
            pts_px[pid] = (px, py)
            occupied.append((u - 5.0, v - 5.0, u + 5.0, v + 5.0))                                   # its own cross
        # touch id beside each crosshair: the side that is free (inside the tile, clear of the marker and the others)
        for pid, (u, v) in sorted(centres.items()):
            txt = str(pid)
            w0, h0 = text_size(txt, f_id)
            tw, th = w0, h0
            cands = [(u + 5.6, v - th / 2.0), (u - 5.6 - tw, v - th / 2.0), (u - tw / 2.0, v - 5.6 - th),
                     (u - tw / 2.0, v + 5.6)]
            best = None
            for (a, b) in cands:
                rect = (a, b, a + tw, b + th)
                if a < 1.5 or b < 1.5 or rect[2] > ts - 1.5 or rect[3] > ts - 1.5:
                    continue
                if any(_rect_gap(rect, o) <= 0.0 for o in occupied):
                    continue
                best = (a, b)
                break
            if best is None:
                best = cands[0]
            a, b = best
            bb = d.textbbox((0, 0), txt, font=f_id)
            d.text((left + a * px_per_mm - bb[0], top + b * px_per_mm - bb[1]), txt, fill=0, font=f_id)
            occupied.append((a - 1.0, b - 1.0, a + tw + 1.0, b + th + 1.0))
        # tile label + marker id: the free spot with the largest clearance from everything else
        lab, mid_txt = t.label or "?", f"marker {t.marker_id}"
        lw, lh = text_size(lab, f_big)
        iw, ih = text_size(mid_txt, f_mid)
        bw, bh = max(lw, iw), lh + 1.5 + ih
        best, best_clear = None, -1.0
        for a in np.arange(2.0, ts - bw - 1.9, 1.0):
            for b in np.arange(2.0, ts - bh - 1.9, 1.0):
                rect = (a, b, a + bw, b + bh)
                clear = min(_rect_gap(rect, o) for o in occupied)
                if clear > best_clear:
                    best, best_clear = (a, b), clear
        if best is None or best_clear < 1.0:       # no room on the tile: above its cut line
            a, b = 0.0, -bh - 1.0
            if top_mm + b < 0.5:
                a, b = 0.0, 1.0
        else:
            a, b = best
        bb = d.textbbox((0, 0), lab, font=f_big)
        d.text((left + a * px_per_mm - bb[0], top + b * px_per_mm - bb[1]), lab, fill=0, font=f_big)
        bb = d.textbbox((0, 0), mid_txt, font=f_mid)
        d.text((left + a * px_per_mm - bb[0], top + (b + lh + 1.5) * px_per_mm - bb[1]), mid_txt, fill=0, font=f_mid)
        info_tiles[t.label] = dict(left=left, top=top, size=size, marker_centre=marker_centre, points=pts_px,
                                   marker_id=t.marker_id, marker_px=(mx, my, img.shape[1], img.shape[0]))

    # ---- text block + scale bar below the tiles ----
    y = int(round((TILE_PAGE_TOP_MM + rows * ts + (rows - 1) * TILE_GAP_MM + 21.0) * px_per_mm))
    x0 = int(round(14.0 * px_per_mm))
    bar_y = y
    d.text((x0, y - int(15 * px_per_mm)), "HOVER TILES - " + ", ".join(t.label for t in tiles.tiles), fill=0, font=_font(int(5.0 * px_per_mm)))
    bar_h = int(2.5 * px_per_mm)
    bar_w = int(round(scale_bar_mm * px_per_mm))
    d.rectangle([x0, y, x0 + bar_w - 1, y + bar_h], fill=0)
    for kk in range(0, int(scale_bar_mm) + 1, 10):
        xt = x0 + int(round(kk * px_per_mm))
        d.line([xt, y - int(3 * px_per_mm), xt, y + bar_h], fill=0, width=max(2, int(dpi / 150)))
        d.text((xt - int(3 * px_per_mm), y - int(8 * px_per_mm)), f"{kk}", fill=0, font=f_small)
    d.text((x0 + bar_w + int(8 * px_per_mm), y - int(1 * px_per_mm)), f"{scale_bar_mm:.0f} mm scale bar", fill=0, font=f_mid)
    y += bar_h + int(6 * px_per_mm)
    mk_mm = tiles.design_marker_length * 1e3
    lines = [f"print at 100 % / Actual size; measure the {scale_bar_mm:.0f} mm bar in x and y; marker edge {mk_mm:g} mm",
             f"Dictionary {tiles.dictionary}; tile {ts:g} x {ts:g} mm (thin outline = cut line); y = towards the tile TOP as printed.",
             "Measure the printed marker edge in x and y with calipers and enter it in targets.yaml:",
             "hover_tiles: marker_length (x) / marker_length_y (y), one value for all tiles of this print.",
             "Cut along the outline, glue each tile flat on its own 100 x 100 mm plate (rigid, flat)."]
    lines += [f"tile {t.label}: marker id {t.marker_id}, touch ids " + ", ".join(str(p) for p in sorted(t.points))
              for t in tiles.tiles]
    step = int(5.2 * px_per_mm)
    for ln in lines:
        d.text((x0, y), ln, fill=0, font=f_mid if ln.startswith("print") else f_small)
        y += step
    if y > page.size[1]:
        raise ValueError(f"the instructions overflow the {paper} page")
    info = dict(dpi=dpi, px_per_mm=px_per_mm, page_px=page.size, tiles=info_tiles,
                scale_bar_px=(x0, bar_y, bar_w, bar_h))
    return page, info


def write_tiles_files(tiles: HoverTilesSpec, out_dir: str, dpi: int = 600, paper: str = "A4"):
    os.makedirs(out_dir, exist_ok=True)
    page, info = render_tiles_page(tiles, dpi=dpi, paper=paper)
    png = os.path.join(out_dir, "hover_tiles.png")
    pdf = os.path.join(out_dir, "hover_tiles.pdf")
    page.save(png, dpi=(dpi, dpi))
    page.save(pdf, "PDF", resolution=float(dpi))
    return png, pdf, info


def tiles_cli(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Generate the print-ready hover tiles page (all tiles of the targets file's "
                                             "'hover_tiles' section on ONE page, exact scale) as PNG+PDF.")
    ap.add_argument("--targets", default=None, help="targets.yaml (default: packaged config/targets.yaml)")
    ap.add_argument("--out-dir", default="~/dual_arm_targets")
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--paper", default="A4", choices=sorted(_PAPER_MM))
    args = ap.parse_args(argv)
    path = os.path.expanduser(args.targets) if args.targets else default_targets_path()
    tiles = load_hover_tiles(path)
    if tiles is None:
        print(f"{path}: no 'hover_tiles' section (a legacy 'hover_sheet' has no tile geometry to print)")
        return 2
    png, pdf, info = write_tiles_files(tiles, os.path.expanduser(args.out_dir), args.dpi, args.paper)
    print(tiles.describe())
    print(f"  -> {png}\n  -> {pdf}   ({info['dpi']} dpi, print at 100 %)")
    return 0


def default_targets_path() -> str:
    try:
        from ament_index_python.packages import get_package_share_directory
        p = os.path.join(get_package_share_directory("rebel_dual_arm_calibration"), "config", "targets.yaml")
        if os.path.exists(p):
            return p
    except Exception:  # noqa: BLE001 - ament not installed / package not built
        pass
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "targets.yaml")


if __name__ == "__main__":
    raise SystemExit(board_cli())
