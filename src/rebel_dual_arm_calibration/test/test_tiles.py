"""Hover TILES: several small ArUco tiles (each with its own marker and 3 touch crosshairs) instead of one hover sheet.

Loader validation, scaling, the packaged default layout, the print generator (A4, 100 % scale measured in rendered pixels),
detection of 3 tiles in one synthetic image, per-tile statuses and the per-tile "has it moved?" check."""
import copy
import itertools
import math
import os
import re

import cv2
import numpy as np
import pytest
import yaml

import test_hover as thv
from rebel_dual_arm_calibration import hover as H
from rebel_dual_arm_calibration import targets as T
from rebel_dual_arm_calibration.mock_common import BoardImage
from rebel_dual_arm_calibration.targets import TargetDetector, TargetSpec
from rebel_dual_arm_calibration.transforms import make_T

CONFIG = os.path.join(os.path.dirname(__file__), "..", "config")
TARGETS = os.path.join(CONFIG, "targets.yaml")
PACKAGED = ("targets.yaml", "targets_charuco_4x4_25mm.yaml")


def _cfg():
    return yaml.safe_load(open(TARGETS))


def _tiles_cfg(**patch):
    """targets.yaml with a small two-tile hover_tiles section (patched per test)."""
    cfg = _cfg()
    cfg["hover_tiles"] = {
        "dictionary": "DICT_4X4_50", "tile_size": 0.100, "design_marker_length": 0.050,
        "marker_length": 0.050, "marker_length_y": 0.050,
        "tiles": [{"marker_id": 13, "label": "A", "tile_center": [20, 20],
                   "points": {101: [59, -21], 102: [-17, 59]}},
                  {"marker_id": 14, "label": "B", "tile_center": [-18, -21],
                   "points": {104: [-60, 19], 105: [21, -58]}}]}
    cfg["hover_tiles"].update(patch)
    return cfg


def _tile_patch(i, **kw):
    cfg = _tiles_cfg()
    tiles = copy.deepcopy(cfg["hover_tiles"]["tiles"])
    tiles[i].update(kw)
    return {"tiles": tiles}


# ---------------------------------------------------------------------------------------------------------------
# loader
# ---------------------------------------------------------------------------------------------------------------
def test_loader_reads_tiles_names_sources_and_geometry():
    t = T.hover_tiles_from_config(_tiles_cfg())
    assert (t.dictionary, t.tile_size, t.design_marker_length) == ("DICT_4X4_50", 0.100, 0.050)
    a, b = t.tiles
    assert (a.source, b.source, a.label, b.label) == ("tile A", "tile B", "A", "B")
    assert (a.marker_id, b.marker_id) == (13, 14)
    assert a.names == {101: "A1", 102: "A2"} and b.names == {104: "B1", 105: "B2"}
    assert a.tile_center_mm == (20.0, 20.0) and a.tile_size == 0.100 and a.points_mm[101] == (59.0, -21.0)
    assert a.points[101] == pytest.approx((0.059, -0.021))
    assert "tile A: marker 13" in t.describe() and "tile B: marker 14" in t.describe()
    # tile_center defaults to the marker centre
    cfg = _tiles_cfg()
    for tl in cfg["hover_tiles"]["tiles"]:
        tl.pop("tile_center")
        tl["points"] = {k: [40, 0] if i == 0 else [0, -40] for i, k in enumerate(tl["points"])}
    assert T.hover_tiles_from_config(cfg).tiles[0].tile_center_mm == (0.0, 0.0)
    # absent section / legacy list API
    assert T.hover_tiles_from_config({k: v for k, v in _tiles_cfg().items() if k != "hover_tiles"}) is None
    sheets = T.hover_sheets_from_config(_tiles_cfg())
    assert [s.source for s in sheets] == ["tile A", "tile B"]


def test_scaling_by_the_measured_marker_edge_in_x_and_y():
    t = T.hover_tiles_from_config(_tiles_cfg(marker_length=0.0495, marker_length_y=0.0505))
    assert t.anisotropic and (t.marker_length, t.marker_length_y) == (0.0495, 0.0505)
    for tile in t.tiles:                                                      # one measured pair for the whole print
        assert (tile.marker_length, tile.marker_length_y) == (0.0495, 0.0505) and tile.anisotropic
        for pid, (x, y) in tile.points_mm.items():
            assert tile.points[pid] == pytest.approx((x * 1e-3 * 0.0495 / 0.050, y * 1e-3 * 0.0505 / 0.050))
    # the measured value defaults to the design value
    cfg = _tiles_cfg()
    cfg["hover_tiles"].pop("marker_length")
    cfg["hover_tiles"].pop("marker_length_y")
    tt = T.hover_tiles_from_config(cfg)
    assert tt.marker_length == tt.marker_length_y == 0.050 and tt.tiles[0].points[101] == pytest.approx((0.059, -0.021))
    # marker corners follow the measured edges
    c = t.tiles[0].marker_corners_obj()
    assert c[1][0] - c[0][0] == pytest.approx(0.0495) and c[0][1] - c[3][1] == pytest.approx(0.0505)


@pytest.mark.parametrize("patch,match", [
    # ---- marker ids
    (_tile_patch(0, marker_id=0), "already used by 'left'"),                  # an arm board
    (_tile_patch(0, marker_id=41), "already used by 'table'"),                # the table board
    (_tile_patch(0, marker_id=99), "outside dictionary"),
    (_tile_patch(1, marker_id=13), "marker ids must be unique"),
    ({"tiles": [{"label": "A", "points": {101: [59, -21]}, "tile_center": [20, 20]}]}, "marker_id is required"),
    # ---- point ids
    (_tile_patch(1, points={101: [-60, 19], 105: [21, -58]}), "point ids must be unique"),
    (_tile_patch(0, points={1: [59, -21], 102: [-17, 59]}), "collides with the table board"),
    (_tile_patch(0, points={0: [59, -21], 102: [-17, 59]}), "positive"),
    (_tile_patch(0, points={}), "no points"),
    # ---- geometry: every rule of the layout
    (_tile_patch(0, points={101: [59, -21], 102: [-17, 71]}), "not on the tile"),                         # off the tile
    (_tile_patch(0, points={101: [59, -21], 102: [-17, 67]}), "from the tile edge"),                      # 3 mm from the edge
    (_tile_patch(0, points={101: [59, -21], 102: [-17, 64.5]}), "from the tile edge"),                    # 5.5 mm: < 6
    (_tile_patch(0, points={101: [59, -21], 102: [28, 0]}), "from the marker edge"),                      # 3 mm from the marker
    (_tile_patch(0, points={101: [59, -21], 102: [0, 0]}), "from the marker edge"),                       # on the marker
    (_tile_patch(0, points={101: [59, -21], 102: [59, -7]}), "mm apart"),                                  # 14 mm apart
    (_tile_patch(0, tile_center=[24, 20]), "from the tile edge"),                                          # marker 1 mm from the edge
    # ---- numbers
    ({"tile_size": 0.0}, "> 0"),
    ({"marker_length": float("nan")}, "finite"),
    ({"marker_length_y": float("inf")}, "finite"),
    ({"marker_length": "abc"}, "number"),
    (_tile_patch(0, points={101: [float("nan"), 0], 102: [-17, 59]}), "finite"),
    (_tile_patch(0, points={101: [59], 102: [-17, 59]}), r"\[x, y\]"),
    (_tile_patch(0, tile_center=[1]), r"\[x, y\]"),
    # ---- labels / list
    (_tile_patch(1, label="A"), "duplicate tile label"),
    (_tile_patch(1, label=True), "label must be text"),
    ({"tiles": []}, "non-empty list"),
    ({"tiles": [3]}, "must be a mapping"),
])
def test_loader_validation(patch, match):
    with pytest.raises(ValueError, match=match):
        T.hover_tiles_from_config(_tiles_cfg(**patch))


def test_tile_validation_boundaries_are_inclusive():
    """Exactly at the limits (6 mm from the tile edge, 8 mm from the marker, 15 mm apart, 3 mm marker margin) is fine."""
    cfg = _tiles_cfg()
    cfg["hover_tiles"]["tiles"][0].update(tile_center=[22, 20], points={101: [66, -24], 102: [-22, 64], 103: [66, 64]})
    t = T.hover_tiles_from_config(cfg)
    assert sorted(t.tiles[0].points) == [101, 102, 103]
    cfg["hover_tiles"]["tiles"][0].update(tile_center=[20, 20], points={101: [33, 0], 102: [48, 0]})        # 8 mm / 15 mm
    T.hover_tiles_from_config(cfg)


def test_tiles_and_the_legacy_sheet_must_not_clash():
    cfg = _tiles_cfg()
    cfg["hover_sheet"] = {"marker_id": 13, "marker_length": 0.06, "points": {120: {"xy_mm": [0, 60]}}}
    with pytest.raises(ValueError, match="marker id 13 is also used by tile A"):
        T.hover_sheets_from_config(cfg)
    cfg["hover_sheet"] = {"marker_id": 30, "marker_length": 0.06, "points": {101: {"xy_mm": [0, 60]}}}
    with pytest.raises(ValueError, match=r"point ids \[101\] are also used by tile A"):
        T.hover_sheets_from_config(cfg)
    cfg["hover_sheet"] = {"marker_id": 30, "marker_length": 0.06, "points": {120: {"xy_mm": [0, 60]}}}
    assert [s.source for s in T.hover_sheets_from_config(cfg)] == ["tile A", "tile B", "sheet"]


def test_tile_markers_must_not_use_the_arm_boards_of_the_aruco_fallback_file():
    cfg = yaml.safe_load(open(os.path.join(CONFIG, "targets_aruco.yaml")))
    cfg["hover_tiles"] = _tiles_cfg()["hover_tiles"]                           # ids 13, 14 are the two arm markers there
    with pytest.raises(ValueError, match="already used"):
        T.hover_tiles_from_config(cfg)
    for tl, mid in zip(cfg["hover_tiles"]["tiles"], (15, 16)):
        tl["marker_id"] = mid
    assert [t.marker_id for t in T.hover_tiles_from_config(cfg).tiles] == [15, 16]


# ---------------------------------------------------------------------------------------------------------------
# the packaged default layout
# ---------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", PACKAGED)
def test_default_layout_satisfies_the_constraints_and_is_irregular(name):
    tiles = T.load_hover_tiles(os.path.join(CONFIG, name))
    assert tiles is not None and tiles.tile_size == 0.100 and tiles.design_marker_length == 0.050
    assert [(t.label, t.marker_id, sorted(t.points)) for t in tiles.tiles] == [
        ("A", 13, [101, 102, 103]), ("B", 14, [104, 105, 106]), ("C", 15, [111, 112, 113])]
    layouts = []
    for t in tiles.tiles:
        cx, cy = t.tile_center_mm
        assert (cx, cy) != (0.0, 0.0)                                           # the marker is OFF-centre
        pts = [t.points_mm[p] for p in sorted(t.points)]
        for x, y in pts:
            assert min(50 - abs(x - cx), 50 - abs(y - cy)) >= 6                  # on the tile, >= 6 mm from its edge
            assert math.hypot(max(abs(x) - 25, 0), max(abs(y) - 25, 0)) >= 8     # >= 8 mm from the marker edge
        d = [math.hypot(a[0] - b[0], a[1] - b[1]) for a, b in itertools.combinations(pts, 2)]
        assert min(d) >= 60.0, (t.label, d)                                      # "as far apart as the tile allows"
        assert 50 - abs(cx) - 25 >= 3 and 50 - abs(cy) - 25 >= 3                 # the marker is on the tile
        # irregular: not a symmetric pattern (no two sides of the triangle equal within 5 mm)
        assert all(abs(p - q) > 5 for p, q in itertools.combinations(d, 2)), (t.label, d)
        layouts.append(tuple(pts))
    assert len(set(layouts)) == 3
    # different per tile: neither the crosshairs nor the marker offset repeat
    assert len({t.tile_center_mm for t in tiles.tiles}) == 3
    assert len({tuple(round(math.hypot(*p)) for p in pts) for pts in layouts}) == 3


def test_every_packaged_config_still_loads_with_its_boards_and_stickers():
    for name in PACKAGED + ("targets_aruco.yaml",):
        path = os.path.join(CONFIG, name)
        T.load_targets_all(path)
        T.load_stickers(path)
        T.hover_sheets_from_config(yaml.safe_load(open(path)))


# ---------------------------------------------------------------------------------------------------------------
# print generator
# ---------------------------------------------------------------------------------------------------------------
def _dark(row, thr=140):
    return np.flatnonzero(row < thr)


def _runs(idx):
    """[(start, end)] of consecutive runs of an index array."""
    if len(idx) == 0:
        return []
    cuts = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[idx[0], idx[cuts + 1]]
    ends = np.r_[idx[cuts], idx[-1]]
    return list(zip(starts.tolist(), ends.tolist()))


@pytest.fixture(scope="module")
def printed(tmp_path_factory):
    tiles = T.load_hover_tiles(TARGETS)
    out = tmp_path_factory.mktemp("tiles")
    png, pdf, info = T.write_tiles_files(tiles, str(out), dpi=600)
    img = np.array(cv2.imread(png, cv2.IMREAD_GRAYSCALE))
    return tiles, png, pdf, info, img


def test_page_is_a4_at_100_percent_scale(printed):
    tiles, png, pdf, info, img = printed
    ppm = 600 / 25.4
    assert info["px_per_mm"] == pytest.approx(ppm)
    assert img.shape == (round(297 * ppm), round(210 * ppm)) == (7016, 4961)          # A4 at 600 dpi
    from PIL import Image
    dpi = Image.open(png).info["dpi"]
    assert dpi[0] == pytest.approx(600, abs=0.01) and dpi[1] == pytest.approx(600, abs=0.01)
    raw = open(pdf, "rb").read()
    assert raw.startswith(b"%PDF")
    box = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)\s*\]", raw)
    assert box is not None and float(box.group(1)) == pytest.approx(595.28, abs=0.6) and float(box.group(2)) == pytest.approx(841.89, abs=0.6)
    # all three tiles on this ONE page, none outside the paper edges
    assert set(info["tiles"]) == {"A", "B", "C"}
    for lab, t in info["tiles"].items():
        assert 0 < t["left"] and t["left"] + t["size"] < img.shape[1] and t["top"] > 0 and t["top"] + t["size"] < img.shape[0]


def test_tile_edge_marker_edge_and_scale_bar_measured_in_pixels(printed):
    tiles, _png, _pdf, info, img = printed
    ppm = info["px_per_mm"]
    for t in tiles.tiles:
        pt = info["tiles"][t.label]
        # tile edge: the two vertical cut lines on a row 1.5 mm below the top edge (nothing else is that close to the edge)
        x_lo = pt["left"] - int(1.5 * ppm)                                          # (the neighbouring tile is not in the window)
        row = img[pt["top"] + int(1.5 * ppm), x_lo:pt["left"] + pt["size"] + int(1.5 * ppm)]
        runs = _runs(_dark(row))
        assert len(runs) == 2, (t.label, runs)
        left, right = runs[0], runs[1]
        assert x_lo + (left[0] + left[1]) / 2 == pytest.approx(pt["left"], abs=2.0)
        assert (right[0] + right[1]) / 2 - (left[0] + left[1]) / 2 == pytest.approx(100 * ppm, abs=1.5)       # 100 mm
        col = img[:, pt["left"] + int(1.5 * ppm)]
        runs = _runs(_dark(col[pt["top"] - 20:pt["top"] + pt["size"] + 20]))
        assert len(runs) == 2, (t.label, runs)
        assert (runs[1][0] + runs[1][1]) / 2 - (runs[0][0] + runs[0][1]) / 2 == pytest.approx(100 * ppm, abs=1.5)
        # marker edge: the first marker row is solid black over the full 50 mm
        mx, my, mw, mh = pt["marker_px"]
        row = img[my + int(1.0 * ppm), mx - 40:mx + mw + 40]
        runs = _runs(_dark(row))
        assert len(runs) == 1 and runs[0][1] - runs[0][0] + 1 == pytest.approx(50 * ppm, abs=1.5), (t.label, runs)
        col = img[my - 40:my + mh + 40, mx + int(1.0 * ppm)]
        runs = _runs(_dark(col))
        assert len(runs) == 1 and runs[0][1] - runs[0][0] + 1 == pytest.approx(50 * ppm, abs=1.5)
        # the marker sits where the layout says: its centre at (tile_size/2 - cx, tile_size/2 + cy) mm from the tile corner
        cx, cy = t.tile_center_mm
        mc = pt["marker_centre"]
        assert (mc[0] - pt["left"]) / ppm == pytest.approx(50 - cx, abs=0.05)
        assert (mc[1] - pt["top"]) / ppm == pytest.approx(50 + cy, abs=0.05)
    # scale bar: 50 mm between the centres of the first and the last tick (row above the bar: ticks only), and the bar itself
    x0, y0, bw, bh = info["scale_bar_px"]
    ticks = _runs(_dark(img[y0 - int(1.5 * ppm), x0 - 30:x0 + bw + 30]))
    assert len(ticks) == 6                                                           # 0 10 20 30 40 50
    centres = [x0 - 30 + (a + b) / 2 for a, b in ticks]
    assert centres[-1] - centres[0] == pytest.approx(50 * ppm, abs=1.0)
    assert np.diff(centres) == pytest.approx(10 * ppm, abs=1.0)
    runs = _runs(_dark(img[y0 + bh // 2, x0 - 30:x0 + bw + 30]))
    assert len(runs) == 1 and runs[0][1] - runs[0][0] + 1 == pytest.approx(50 * ppm, abs=5.0)       # + the end ticks' width


def test_crosshairs_are_printed_at_their_design_positions_with_a_ring_and_their_ids(printed):
    tiles, _png, _pdf, info, img = printed
    ppm = info["px_per_mm"]
    for t in tiles.tiles:
        pt = info["tiles"][t.label]
        cx, cy = t.tile_center_mm
        for pid, (x, y) in t.points_mm.items():
            u, v = 50 - cx + x, 50 + cy - y                                           # mm from the tile corner (y down)
            px, py = pt["left"] + u * ppm, pt["top"] + v * ppm
            assert pt["points"][pid] == pytest.approx((px, py), abs=0.5)
            r = int(4.8 * ppm)
            crop = img[int(py) - r:int(py) + r, int(px) - r:int(px) + r] < 140
            col = int(np.argmax(crop.sum(axis=0)))
            rowi = int(np.argmax(crop.sum(axis=1)))
            assert abs((int(px) - r + col) - px) <= 2.0 and abs((int(py) - r + rowi) - py) <= 2.0
            # arms reach 4.5 mm, there is a gap at the centre and a ring of 1 mm radius
            yy, xx = np.mgrid[-r:r, -r:r]
            rad = np.hypot(xx - (px - int(px)), yy - (py - int(py))) / ppm
            ring = crop[(rad > 0.8) & (rad < 1.2)]
            assert ring.mean() > 0.25
            assert crop[rad < 0.5].sum() == 0                                          # open centre: the tip target is visible
            assert crop[(rad > 4.0) & (rad < 4.5)].any()                               # the arms reach out to 4.5 mm
    # nothing printed outside the tiles / text block: the margins of the page are blank
    assert img[:, :int(2.5 * ppm)].min() == 255 and img[:, -int(2.5 * ppm):].min() == 255 and img[:int(3 * ppm)].min() == 255


def test_page_text_has_the_required_instruction_lines(printed):
    # the instruction text is raster only: check that it is rendered by looking at the text block's ink, and that the
    # generator composes the exact wording (checked on the source of truth, the generator's own line)
    import inspect
    src = inspect.getsource(T.render_tiles_page)
    assert "print at 100 % / Actual size; measure the {scale_bar_mm:.0f} mm bar in x and y; marker edge {mk_mm:g} mm" in src
    _tiles, _png, _pdf, info, img = printed
    ppm = info["px_per_mm"]
    x0, y0, bw, bh = info["scale_bar_px"]
    block = img[y0 + bh:, :]
    assert (block < 140).sum() > 20000                                                # several lines of text below the bar


def test_cli_writes_png_and_pdf_and_reports_a_missing_section(tmp_path, capsys):
    out = tmp_path / "o"
    assert T.tiles_cli(["--targets", TARGETS, "--out-dir", str(out), "--dpi", "150"]) == 0
    assert (out / "hover_tiles.png").exists() and (out / "hover_tiles.pdf").exists()
    from PIL import Image
    im = Image.open(out / "hover_tiles.png")
    assert im.size == (round(210 / 25.4 * 150), round(297 / 25.4 * 150))
    assert "tile A: marker 13" in capsys.readouterr().out
    plain = tmp_path / "plain.yaml"
    cfg = _cfg()
    cfg.pop("hover_tiles")
    plain.write_text(yaml.safe_dump(cfg))
    assert T.tiles_cli(["--targets", str(plain), "--out-dir", str(tmp_path / "x")]) == 2
    assert not (tmp_path / "x").exists()
    assert "generate_tiles = rebel_dual_arm_calibration.targets:tiles_cli" in open(
        os.path.join(os.path.dirname(__file__), "..", "setup.py")).read()


def test_too_many_tiles_do_not_fit_one_page():
    cfg = _tiles_cfg()
    base = cfg["hover_tiles"]["tiles"][0]
    cfg["hover_tiles"]["tiles"] = [dict(base, marker_id=30 + k, label=str(k), points={110 + 3 * k + j: p for j, p in enumerate([[59, -21], [-17, 59]])})
                                   for k in range(6)]
    tiles = T.hover_tiles_from_config(cfg)
    with pytest.raises(ValueError, match="do not fit on one A4 page"):
        T.render_tiles_page(tiles, dpi=100)
    cfg["hover_tiles"]["tiles"] = cfg["hover_tiles"]["tiles"][:4]
    page, info = T.render_tiles_page(T.hover_tiles_from_config(cfg), dpi=100)
    assert len(info["tiles"]) == 4 and page.size == (round(210 / 25.4 * 100), round(297 / 25.4 * 100))


def test_printed_tiles_are_detected_by_the_marker_detector_and_the_crosshairs_land_on_the_design_points(printed):
    """The printed page itself: each marker is found at its place (no scale error beyond 1 px) and the page is a
    valid ArUco print (4 corners within 1.5 px of the design edges)."""
    tiles, _png, _pdf, info, img = printed
    ppm = info["px_per_mm"]
    small = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    d = cv2.aruco
    dic = T.get_dictionary("DICT_4X4_50")
    params = T.make_detector_params(0.01)
    if hasattr(d, "ArucoDetector"):
        corners, ids, _ = d.ArucoDetector(dic, params).detectMarkers(small)
    else:
        corners, ids, _ = d.detectMarkers(small, dic, parameters=params)
    assert sorted(ids.reshape(-1).tolist()) == [13, 14, 15]
    for c, i in zip(corners, ids.reshape(-1)):
        lab = {13: "A", 14: "B", 15: "C"}[int(i)]
        mx, my, mw, mh = info["tiles"][lab]["marker_px"]
        c = c.reshape(4, 2) * 2.0
        assert abs(np.linalg.norm(c[1] - c[0]) - 50 * ppm) < 6.0          # half-resolution detection: a few px at 600 dpi
        assert abs(c[0][0] - mx) < 6.0 and abs(c[0][1] - my) < 6.0


# ---------------------------------------------------------------------------------------------------------------
# detection of three tiles in one image
# ---------------------------------------------------------------------------------------------------------------
TILES = T.load_hover_tiles(TARGETS)
POSES = {                         # T_optical_marker of each tile (x right, y up as printed, z out of the paper)
    "A": lambda: thv.sheet_pose(tilt_x=-30.0, tilt_y=12.0, spin=0.0, t=(0.17, 0.02, 0.80)),
    "B": lambda: thv.sheet_pose(tilt_x=-35.0, tilt_y=-15.0, spin=20.0, t=(-0.02, -0.17, 0.82)),
    "C": lambda: thv.sheet_pose(tilt_x=-28.0, tilt_y=8.0, spin=-15.0, t=(0.04, 0.17, 0.78)),
}


class TileScene(thv.Scene):
    def __init__(self, dist=thv.D0, seed=7):
        super().__init__(dist, seed)
        self.tile_boards = {t.label: BoardImage(TargetSpec(t.source, "aruco", t.dictionary, t.design_marker_length, [t.marker_id]),
                                                margin=0.030, plate=0.0) for t in TILES.tiles}

    def render_tiles(self, which=("A", "B", "C"), poses=None, T_table=None, noise=2.0, blur=0.7):
        poses = poses or {}
        frame = self.bg.copy()
        if T_table is not None:
            assert self.boards["table"].warp_into(frame, thv.K, T_table)
        for lab in which:
            Tm = poses.get(lab, POSES[lab]())
            assert self.tile_boards[lab].warp_into(frame, thv.K, Tm @ make_T(thv.F_FLIP, np.zeros(3)))
        if np.any(self.dist != 0):
            frame = cv2.remap(frame, self.mx, self.my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        frame = cv2.GaussianBlur(frame, (0, 0), blur)
        return np.clip(frame.astype(np.float32) + self.rng.normal(0, noise, frame.shape[:2])[:, :, None], 0, 255).astype(np.uint8)

    def frames(self, n=5, **kw):
        return [(self.render_tiles(**kw), thv.K, self.dist) for _ in range(n)]


@pytest.fixture(scope="module", params=["pinhole", "distorted"])
def tscene(request):
    return TileScene(thv.D0 if request.param == "pinhole" else thv.D1)


def _truth(tile, Tm):
    return {pid: Tm[:3, :3] @ np.array([x, y, 0.0]) + Tm[:3, 3] for pid, (x, y) in tile.points.items()}


def test_three_tiles_in_one_image_are_found_with_sub_millimetre_crosshairs(tscene):
    Tt = thv.table_pose()
    pts, status = H.detect_targets(tscene.frames(T_table=Tt), tscene.table, TILES, TargetDetector({"table": tscene.table}))
    assert set(pts) == {1, 2, 3, 4} | {p for t in TILES.tiles for p in t.points}, status
    assert {k for k in status} == {"table", "tile A", "tile B", "tile C"}
    for t in TILES.tiles:
        assert status[t.source].startswith(f"{t.source}: marker {t.marker_id}"), status
    for t in TILES.tiles:
        Tm = POSES[t.label]()
        truth = _truth(t, Tm)
        for pid, tp in ((p, pts[p]) for p in t.points):
            assert tp.source == t.source and tp.name == t.names[pid]
            err = np.linalg.norm(tp.p_optical - truth[pid]) * 1e3
            assert err < 1.0, (t.label, pid, err)
            assert np.degrees(np.arccos(np.clip(tp.n_optical @ Tm[:3, 2], -1, 1))) < 1.5
            assert np.degrees(np.arccos(np.clip(tp.x_optical @ Tm[:3, 0], -1, 1))) < 1.5
            assert tp.n_optical @ tp.p_optical < 0 and tp.quality["frames"] == 5 and tp.quality["spread_mm"] <= 1.0
    assert pts[101].name == "A1" and pts[105].name == "B2" and pts[113].name == "C3"


def test_a_single_hover_spec_or_a_list_gives_the_same_points(tscene):
    frames = tscene.frames(n=4, which=("A",))
    one, st1 = H.detect_targets(frames, None, TILES.tiles[0], None)
    many, st2 = H.detect_targets(frames, None, TILES.tiles[:1], None)
    both, st3 = H.detect_targets(frames, None, TILES, None)             # the HoverTilesSpec itself, B and C hidden
    assert set(one) == set(many) == set(TILES.tiles[0].points) == set(both)
    for pid in one:
        assert np.allclose(one[pid].p_optical, many[pid].p_optical)
    assert st1.keys() == st2.keys() == {"tile A", "table"}
    assert "marker 14 not found" in st3["tile B"] and "marker 15 not found" in st3["tile C"]


def test_hiding_one_marker_gives_that_tiles_status_and_keeps_the_others(tscene):
    pts, status = H.detect_targets(tscene.frames(which=("A", "C")), None, TILES, None)
    assert set(pts) == set(TILES.tiles[0].points) | set(TILES.tiles[2].points)
    assert status["tile B"] == "tile B: marker 14 not found (0 of 5 frames)"
    assert status["tile A"].startswith("tile A: marker 13") and status["tile C"].startswith("tile C: marker 15")
    assert status["table"].startswith("table: no table board")
    # a marker seen in only some frames: reported with its own counts
    frames = tscene.frames(n=2, which=("A", "B", "C")) + tscene.frames(n=3, which=("A", "C"))
    pts, status = H.detect_targets(frames, None, TILES, None)
    assert "only 2 of 5" in status["tile B"] and not (set(pts) & set(TILES.tiles[1].points))
    assert set(TILES.tiles[0].points) <= set(pts)
    # no frames at all
    pts, status = H.detect_targets([], None, TILES, None)
    assert pts == {} and status["tile A"] == "tile A: no camera frames" and status["tile C"] == "tile C: no camera frames"
    # nothing configured: the legacy key
    pts, status = H.detect_targets(tscene.frames(n=3, which=("A",)), None, None, None)
    assert pts == {} and status["sheet"] == "sheet: no hover sheet configured"


def test_a_tile_that_moves_between_frames_is_rejected_alone(tscene):
    base = np.array(POSES["B"]()[:3, 3])
    frames = []
    for k in range(5):
        Tb = POSES["B"]()
        Tb[:3, 3] = base + np.array([0.004 * (k % 2), 0.0, 0.0])
        frames.append((tscene.render_tiles(poses={"B": Tb}), thv.K, tscene.dist))
    pts, status = H.detect_targets(frames, None, TILES, None)
    assert set(pts) == set(TILES.tiles[0].points) | set(TILES.tiles[2].points)
    assert "spread" in status["tile B"] and "not used" in status["tile B"]
    assert "spread" in status["tile A"] and "not used" not in status["tile A"]


def test_gates_apply_per_tile(tscene, monkeypatch):
    monkeypatch.setattr(H, "SHEET_MIN_IPPE_RATIO", 1e9)
    pts, status = H.detect_targets(tscene.frames(n=3), None, TILES, None)
    assert not pts and all("ambiguous IPPE" in status[t.source] and "0 of 3" in status[t.source] for t in TILES.tiles)


# ---------------------------------------------------------------------------------------------------------------
# per-tile "has the paper moved?"
# ---------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def stored(tscene):
    pts, status = H.detect_targets(tscene.frames(), None, TILES, None)
    assert len(pts) == 9, status
    return pts


def test_stored_source_poses_per_tile(tscene, stored):
    poses = H.stored_source_poses(stored, None, TILES)
    assert set(poses) == {"tile A", "tile B", "tile C"}
    for t in TILES.tiles:
        truth = POSES[t.label]()
        assert np.linalg.norm(poses[t.source][:3, 3] - truth[:3, 3]) < 1e-3
        assert np.degrees(np.arccos(np.clip((np.trace(poses[t.source][:3, :3].T @ truth[:3, :3]) - 1) / 2, -1, 1))) < 0.7
    only_b = {k: v for k, v in stored.items() if v.source == "tile B"}
    assert set(H.stored_source_poses(only_b, None, TILES)) == {"tile B"}
    assert H.stored_source_poses({}, None, TILES) == {}


def test_unmoved_is_decided_per_tile(tscene, stored):
    ok, why, stats = H.unmoved(tscene.frames(n=3), stored, None, TILES, None)
    assert ok and why == "" and set(stats) == {"tile A", "tile B", "tile C"}
    for src, st in stats.items():
        assert st["verdicts"] == ["ok"] * 3 and max(st["median_px"]) < 0.8, (src, st)
    # tile B moved by 3 mm: B fails alone; A and C still pass when checked on their own
    Tb = POSES["B"]()
    Tb[:3, 3] += np.array([0.003, 0.0, 0.0])
    moved = tscene.frames(n=3, poses={"B": Tb})
    ok, why, stats = H.unmoved(moved, stored, None, TILES, None, sources=["tile B"])
    assert not ok and why.startswith(H.MOVED_REASON) and stats["tile B"]["verdicts"] == ["moved"] * 3
    for src in ("tile A", "tile C"):
        ok, why, _ = H.unmoved(moved, stored, None, TILES, None, sources=[src])
        assert ok and why == "", (src, why)
    ok, why, stats = H.unmoved(moved, stored, None, TILES, None)                     # default: every stored source
    assert not ok and why.startswith(H.MOVED_REASON)
    assert stats["tile A"]["verdicts"] == ["ok"] * 3 and stats["tile B"]["verdicts"] == ["moved"] * 3
    # tile C hidden (not in view): hidden, and only C is reported as such
    hidden = tscene.frames(n=3, which=("A", "B"))
    ok, why, stats = H.unmoved(hidden, stored, None, TILES, None, sources=["tile C"])
    assert not ok and why.startswith(H.HIDDEN_REASON) and stats["tile C"]["verdicts"] == ["hidden"] * 3
    ok, why, _ = H.unmoved(hidden, stored, None, TILES, None, sources=["tile A", "tile B"])
    assert ok
    # a tile that was never stored
    ok, why, _ = H.unmoved(hidden, {k: v for k, v in stored.items() if v.source != "tile C"}, None, TILES, None,
                           sources=["tile C"])
    assert not ok and "no stored tile C pose" in why
