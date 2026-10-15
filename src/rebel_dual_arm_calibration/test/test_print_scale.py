"""Print scale anisotropy: optional square_length_y / marker_length_y (printers scale x and y differently)."""
import os

import cv2
import numpy as np
import pytest
import yaml

from rebel_dual_arm_calibration import solver, synthetic as S, targets as T
from rebel_dual_arm_calibration.targets import TargetDetector, TargetSpec, load_targets_all, render_board
from test_targets import CONFIG, _scene_with_boards
from test_loop_closure import World


def aniso_spec(arm="left", first=0, sq=0.025, sqy=0.0262, mk=0.018, mky=0.0189, n=4):
    ids = list(range(first, first + (n * n) // 2))
    return TargetSpec(arm, "charuco", "DICT_4X4_50", mk, ids, sq, n, n, square_length_y=sqy, marker_length_y=mky)


def test_default_y_equals_x_and_geometry_uses_separate_scales():
    iso = TargetSpec("left", "charuco", "DICT_4X4_50", 0.018, list(range(8)), 0.025, 4, 4)
    assert iso.sq_y == 0.025 and iso.mk_y == 0.018 and not iso.anisotropic and iso.board_size() == (0.1, 0.1)
    a = aniso_spec()
    assert a.anisotropic and a.board_size() == pytest.approx((0.1, 4 * 0.0262))
    m = a.marker_corners_obj(0)
    assert np.linalg.norm(m[1] - m[0]) == pytest.approx(0.018) and np.linalg.norm(m[3] - m[0]) == pytest.approx(0.0189)
    c = {pid: p for pid, p in a.chess_corners_obj()}
    assert c[T.CHESS_ID_BASE + 1][0] - c[T.CHESS_ID_BASE][0] == pytest.approx(0.025)             # along x
    assert c[T.CHESS_ID_BASE + 3][1] - c[T.CHESS_ID_BASE][1] == pytest.approx(0.0262)            # next row, y
    assert "25.00 x 26.20" in a.describe() and "18.00 x 18.90" in a.describe()
    # touch-point corners use the y square size
    spec = T._spec_from_dict("table", {"type": "charuco", "squares_x": 5, "squares_y": 4, "square_length": 0.04,
                                       "square_length_y": 0.0404, "marker_length": 0.03, "marker_length_y": 0.0303,
                                       "first_marker_id": 40, "touch_points": [{"id": 1, "corner": [1, 1]}]})
    assert spec.touch_points[1] == pytest.approx((-0.06, -0.0404 * 4 / 2 + 0.0404))


def test_config_round_trip_and_validation():
    a = aniso_spec()
    cfg = T.specs_to_config({"left": a})
    assert cfg["arms"]["left"]["square_length_y"] == 0.0262 and cfg["arms"]["left"]["marker_length_y"] == 0.0189
    b = T._spec_from_dict("left", cfg["arms"]["left"])
    assert b.sq_y == a.sq_y and b.mk_y == a.mk_y and b.board_size() == a.board_size()
    iso = T.specs_to_config({"left": TargetSpec("left", "charuco", "DICT_4X4_50", 0.018, list(range(8)), 0.025, 4, 4)})
    assert "square_length_y" not in iso["arms"]["left"]
    with pytest.raises(ValueError, match="smaller"):
        T._spec_from_dict("left", dict(cfg["arms"]["left"], marker_length_y=0.03))
    # the default configs are unchanged (y defaults to x)
    arms, table = load_targets_all(os.path.join(CONFIG, "targets.yaml"))
    assert not arms["left"].anisotropic and not table.anisotropic


def test_rendered_anisotropic_board_is_detected_and_matches_the_geometry():
    specs = {"left": aniso_spec("left", 0), "right": aniso_spec("right", 20, sq=0.025, sqy=0.0238, mk=0.018, mky=0.0171)}
    img, Hs = _scene_with_boards(specs, px_per_m=4000.0, scale=0.30, noise=2.0)
    det = TargetDetector(specs).detect(img)
    assert set(det) == {"left", "right"}
    for arm in ("left", "right"):
        d = det[arm]
        assert d.n_markers == 8 and d.n_chess == 9
        ids, obj, pix = d.arrays()
        xy = np.hstack([obj[:, :2], np.ones((len(obj), 1))])
        proj = (Hs[arm] @ xy.T).T
        proj = proj[:, :2] / proj[:, 2:3]
        err = np.linalg.norm(proj - pix, axis=1)
        assert err.max() < 2.0 and err[ids >= T.CHESS_ID_BASE].max() < 0.6


def test_print_page_reports_both_scales(tmp_path):
    png, _, info = T.write_print_files(aniso_spec(), str(tmp_path), dpi=150)
    assert os.path.getsize(png) > 1000
    page, _ = T.render_print_page(aniso_spec(), dpi=150)
    assert page.size[0] > 100


@pytest.mark.skipif(S.rebel_urdf_path() is None, reason="igus_rebel2.urdf not found")
def test_solver_needs_the_measured_y_scale_of_the_table_board():
    """The printed table board is 3 % larger in y than configured.  Solving with the nominal geometry shows the
    inconsistency at the board touches (out-of-fit), entering the measured y scale removes it."""
    w = World(0.0, seed=4)
    sy = 1.03
    meas = T._spec_from_dict("table", dict(T.table_to_config(w.table), square_length_y=0.040 * sy, marker_length_y=0.030 * sy,
                                           touch_points=[{"id": k, "corner": c} for k, c in
                                                         zip((1, 2, 3, 4), ([1, 1], [4, 1], [4, 3], [1, 3]))]))
    assert meas.anisotropic and meas.touch_points[1][1] == pytest.approx(w.table.touch_points[1][1] * sy)
    # physical board = measured geometry; the solver is told the nominal one (wrong) or the measured one (right)
    def run(spec_for_solver, obs_scale):
        tables = S.simulate_table_obs(w.scene, spec_for_solver, 10, seed=24, board_scale=obs_scale)
        touches = S.simulate_touches(w.scene, spec_for_solver, w.kin, seed=34, board_scale=obs_scale)
        return w.solve(tables=tables, touches=touches).report["loop_closure"]
    wrong = run(w.table, (1.0, sy))          # nominal geometry in the solver, physical y scale 1.03
    right = run(meas, (1.0, 1.0))            # measured geometry in the solver (physical == measured)
    assert wrong["out_of_fit_rms_mm"] > 1.0 and right["out_of_fit_rms_mm"] < 0.8 < wrong["out_of_fit_rms_mm"], \
        (wrong["out_of_fit_rms_mm"], right["out_of_fit_rms_mm"])
