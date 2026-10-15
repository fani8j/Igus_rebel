"""Table board (loop-closure board): config, ID disjointness, detection with touch corners excluded, print page."""
import os

import numpy as np
import pytest
import yaml
from PIL import ImageChops

from rebel_dual_arm_calibration import targets as T
from rebel_dual_arm_calibration.targets import TargetDetector, load_targets, load_targets_all, render_print_page
from test_targets import CONFIG, _scene_with_boards


@pytest.mark.parametrize("cfg", ["targets.yaml", "targets_charuco_4x4_25mm.yaml", "targets_aruco.yaml"])
def test_table_board_config_loads_and_is_disjoint(cfg):
    arms, table = load_targets_all(os.path.join(CONFIG, cfg))
    assert set(arms) == {"left", "right"}                      # load_targets() is unchanged: arms only
    assert set(load_targets(os.path.join(CONFIG, cfg))) == {"left", "right"}
    assert table is not None and (table.squares_x, table.squares_y) == (5, 4)
    assert table.square_length == pytest.approx(0.040) and table.paper == "A4"
    used = set(table.marker_ids)
    assert len(used) == 10 and not used & set(arms["left"].marker_ids) and not used & set(arms["right"].marker_ids)
    assert sorted(table.touch_points) == [1, 2, 3, 4]
    # inner corners (1,1), (4,1), (4,3), (1,3) of a 200 x 160 mm board centred on the origin
    assert table.touch_points[1] == pytest.approx((-0.06, -0.04))
    assert table.touch_points[3] == pytest.approx((0.06, 0.04))
    assert len(table.touch_corner_ids()) == 4


def test_table_board_overlapping_ids_and_bad_touch_points_rejected(tmp_path):
    cfg = yaml.safe_load(open(os.path.join(CONFIG, "targets.yaml")))
    bad = dict(cfg, table_board=dict(cfg["table_board"], first_marker_id=2))   # overlaps the left board (0..3)
    with pytest.raises(ValueError, match="disjoint"):
        T.table_from_config(bad)
    bad = dict(cfg, table_board=dict(cfg["table_board"], touch_points=[{"id": 1, "corner": [0, 1]}]))
    with pytest.raises(ValueError, match="inner corner"):
        T.table_from_config(bad)
    bad = dict(cfg, table_board=dict(cfg["table_board"], touch_points=[{"id": 1, "corner": [1, 1]},
                                                                      {"id": 1, "corner": [2, 1]}]))
    with pytest.raises(ValueError, match="unique"):
        T.table_from_config(bad)
    assert T.table_from_config({"arms": cfg["arms"]}) is None


def test_table_board_detected_without_touch_corners():
    arms, table = load_targets_all(os.path.join(CONFIG, "targets.yaml"))
    specs = {"table": table, "left": arms["left"]}
    img, Hs = _scene_with_boards(specs, px_per_m=4000.0, scale=0.30, noise=2.0)
    det = TargetDetector({**arms, "table": table}).detect(img)
    assert set(det) == {"table", "left"}          # right board is outside this scene
    d = det["table"]
    assert d.n_markers == 10 and d.n_chess == 12 - 4      # the 4 crosshair corners are not observed
    ids, obj, pix = d.arrays()
    assert not set(ids) & table.touch_corner_ids()
    xy = np.hstack([obj[:, :2], np.ones((len(obj), 1))])
    proj = (Hs["table"] @ xy.T).T
    proj = proj[:, :2] / proj[:, 2:3]
    assert np.linalg.norm(proj - pix, axis=1).max() < 2.0


def test_print_page_has_touch_crosshairs_near_the_points():
    _, table = load_targets_all(os.path.join(CONFIG, "targets.yaml"))
    dpi = 150
    page, info = render_print_page(table, dpi=dpi, paper="A3")
    assert page.size[0] == round(297 / 25.4 * dpi)
    plain = T.TargetSpec(**{**table.__dict__, "touch_points": {}})
    page0, _ = render_print_page(plain, dpi=dpi, paper="A3")
    diff = ImageChops.difference(page, page0).convert("L")
    bx, by = info["board_origin_px"]
    px_mm = dpi / 25.4
    # board origin in page pixels = pasted board left/top + (margin + w/2, margin + h/2)
    ox, oy = bx + (10 + 100) * px_mm, by + (10 + 80) * px_mm
    arr = np.array(diff)
    bw, bh = info["board_px"]
    ys, xs = np.nonzero(arr[by:by + bh, bx:bx + bw] > 0)
    ys, xs = ys + by, xs + bx
    assert len(xs) > 20
    near = np.zeros(len(xs), bool)
    for x, y in table.touch_points.values():
        near |= (np.abs(xs - (ox + x * 1e3 * px_mm)) < 9 * px_mm) & (np.abs(ys - (oy + y * 1e3 * px_mm)) < 9 * px_mm)
    assert near.all()                              # everything that was added lies at a touch point


def test_board_cli_writes_the_table_board(tmp_path):
    assert T.board_cli(["--out-dir", str(tmp_path), "--dpi", "100"]) == 0
    for name in ("target_left", "target_right", "target_table"):
        assert (tmp_path / f"{name}.png").exists() and (tmp_path / f"{name}.pdf").exists()


def test_table_board_prints_on_a4_landscape_at_exact_scale():
    """A4-only printers: the 220 mm wide board (+ margins) goes on the landscape page at 1:1, labels fit."""
    _, table = load_targets_all(os.path.join(CONFIG, "targets.yaml"))
    dpi = 150
    px_mm = dpi / 25.4
    page, info = render_print_page(table, dpi=dpi, paper="A4")
    assert info["landscape"] and page.size == (round(297 * px_mm), round(210 * px_mm))
    bw, bh = info["board_px"]
    assert abs(bw - 220 * px_mm) <= 1 and abs(bh - 180 * px_mm) <= 1     # 200 x 160 mm + 2 x 10 mm margin
    bx, by = info["board_origin_px"]
    assert bx >= 9.5 * px_mm and by + bh <= page.size[1] - 9.5 * px_mm   # >= 10 mm from every page edge
    # the arm boards still use the portrait page
    arms, _ = load_targets_all(os.path.join(CONFIG, "targets.yaml"))
    assert not render_print_page(arms["left"], dpi=dpi, paper="A4")[1]["landscape"]
