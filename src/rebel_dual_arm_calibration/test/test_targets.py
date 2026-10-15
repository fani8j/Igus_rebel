"""Target definitions, disjoint ids, board rendering and detection."""
import os

import cv2
import numpy as np
import pytest

from rebel_dual_arm_calibration import targets as T
from rebel_dual_arm_calibration.solver import Sample, pnp_candidates
from rebel_dual_arm_calibration.targets import TargetDetector, TargetSpec, load_targets, render_board

CONFIG = os.path.join(os.path.dirname(__file__), "..", "config")


def test_default_configs_load_and_are_disjoint():
    for name in ("targets.yaml", "targets_charuco_4x4_25mm.yaml", "targets_aruco.yaml"):
        specs = load_targets(os.path.join(CONFIG, name))
        assert set(specs) == {"left", "right"}
        assert not set(specs["left"].marker_ids) & set(specs["right"].marker_ids)


def test_overlapping_ids_rejected():
    a = TargetSpec("left", "charuco", "DICT_4X4_50", 0.018, list(range(8)), 0.025, 4, 4)
    b = TargetSpec("right", "charuco", "DICT_4X4_50", 0.018, list(range(7, 15)), 0.025, 4, 4)
    with pytest.raises(ValueError, match="disjoint"):
        T.validate_specs({"left": a, "right": b})
    c = TargetSpec("right", "charuco", "DICT_4X4_50", 0.018, list(range(45, 53)), 0.025, 4, 4)
    with pytest.raises(ValueError):
        T.validate_specs({"left": a, "right": c})  # ids beyond dictionary size


def test_board_geometry():
    s = load_targets(os.path.join(CONFIG, "targets_charuco_4x4_25mm.yaml"))["left"]
    assert s.n_markers() == 8 and len(s.chess_corners_obj()) == 9
    assert s.board_size() == pytest.approx((0.1, 0.1))
    m = s.marker_corners_obj(0)
    assert np.linalg.norm(m[1] - m[0]) == pytest.approx(s.marker_length)
    pts = s.all_points_obj()
    assert len(pts) == 8 * 4 + 9
    # board centred on the origin
    xs = np.array([p[0] for p in pts.values()])
    assert abs(xs.mean()) < 1e-9


def test_default_config_is_3x3_35mm():
    s = load_targets(os.path.join(CONFIG, "targets.yaml"))["right"]
    assert (s.squares_x, s.squares_y, s.square_length, s.marker_ids) == (3, 3, 0.035, [20, 21, 22, 23])
    assert s.board_size() == pytest.approx((0.105, 0.105))


def test_print_files_have_exact_scale(tmp_path):
    s = load_targets(os.path.join(CONFIG, "targets_charuco_4x4_25mm.yaml"))["left"]
    png, pdf, info = T.write_print_files(s, str(tmp_path), dpi=300)
    assert os.path.getsize(png) > 1000 and os.path.getsize(pdf) > 1000
    from PIL import Image
    im = Image.open(png)
    assert im.info["dpi"][0] == pytest.approx(300, abs=0.5)
    a = np.array(im)
    # 100 mm board = 1181 px at 300 dpi: measure the black extent of the board region
    bx, by = info["board_origin_px"]
    bw, bh = info["board_px"]
    region = a[by:by + bh, bx:bx + bw]
    cols = np.flatnonzero((region < 128).any(axis=0))
    mm = (cols.max() - cols.min() + 1) / (300 / 25.4)
    assert mm == pytest.approx(100.0, abs=0.2)
    # scale bar: 50 mm
    row = a[by + bh + int(14 * 300 / 25.4):, :]
    dark_rows = np.flatnonzero((a[by + bh:] < 128).sum(axis=1) > 400)
    assert len(dark_rows)
    bar = a[by + bh + dark_rows[0]] < 128
    assert (bar.sum() / (300 / 25.4)) == pytest.approx(50.0, abs=0.3)
    assert pdf.endswith(".pdf") and open(pdf, "rb").read(5) == b"%PDF-"


def _scene_with_boards(specs, px_per_m=2400.0, scale=0.8, angle_deg=0.0, noise=0.0, seed=0):
    """Render each board, warp with a homography into a 1920x1080 grey scene.
    Returns image and the true board->pixel homographies (object xy metres -> pixel)."""
    rng = np.random.default_rng(seed)
    scene = np.full((1080, 1920), 190, np.uint8)
    Hs = {}
    for k, (arm, spec) in enumerate(specs.items()):
        hi, (ox, oy) = render_board(spec, px_per_m, margin=0.012)
        # anti-aliased down-scale first (INTER_AREA), then a mild rotation + perspective warp.
        # Pixel-centre convention: u' = f * (ox + x * ppm) - 0.5 maps metres -> resized pixels.
        img = cv2.resize(hi, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        A = np.array([[scale * px_per_m, 0, scale * ox - 0.5], [0, scale * px_per_m, scale * oy - 0.5], [0, 0, 1.0]])
        a = np.radians(angle_deg + 25 * k)
        S = np.array([[np.cos(a), -np.sin(a), 500 + 700 * k],
                      [np.sin(a), np.cos(a), 250],
                      [2e-5 * (k + 1) / scale, -1e-5 / scale, 1.0]])
        Hs[arm] = S @ A
        warped = cv2.warpPerspective(img, S, (1920, 1080), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=190)
        mask = cv2.warpPerspective(np.full_like(img, 255), S, (1920, 1080), flags=cv2.INTER_NEAREST)
        scene = np.where(mask > 0, warped, scene)
    if noise:
        scene = np.clip(scene + rng.normal(0, noise, scene.shape), 0, 255).astype(np.uint8)
    return scene, Hs


@pytest.mark.parametrize("n,square,marker,first_right", [(4, 0.025, 0.018, 20), (3, 0.035, 0.026, 10)])
def test_rendered_boards_detected_with_correct_arm_ids(n, square, marker, first_right):
    cnt = (n * n) // 2
    specs = {"left": TargetSpec("left", "charuco", "DICT_4X4_50", marker, list(range(cnt)), square, n, n),
             "right": TargetSpec("right", "charuco", "DICT_4X4_50", marker, list(range(first_right, first_right + cnt)),
                                 square, n, n)}
    img, Hs = _scene_with_boards(specs, px_per_m=4000.0, scale=0.30, noise=2.0)
    det = TargetDetector(specs).detect(img)
    assert set(det) == {"left", "right"}
    for arm in ("left", "right"):
        d = det[arm]
        assert d.n_markers == cnt
        assert d.n_chess == (n - 1) ** 2
        ids, obj, pix = d.arrays()
        # every detected pixel agrees with the true projection of its object point
        xy = np.hstack([obj[:, :2], np.ones((len(obj), 1))])
        proj = (Hs[arm] @ xy.T).T
        proj = proj[:, :2] / proj[:, 2:3]
        err = np.linalg.norm(proj - pix, axis=1)
        chess = ids >= T.CHESS_ID_BASE
        assert np.median(err) < 0.6 and err.max() < 2.0, err.max()
        assert err[chess].max() < 0.5, err[chess].max()


def test_only_one_board_visible_is_assigned_to_correct_arm():
    specs = load_targets(os.path.join(CONFIG, "targets.yaml"))
    only_right = {"right": specs["right"]}
    img, _ = _scene_with_boards(only_right, px_per_m=4000.0, scale=0.3)
    det = TargetDetector(specs).detect(img)
    assert set(det) == {"right"}


def test_single_aruco_detected():
    specs = load_targets(os.path.join(CONFIG, "targets_aruco.yaml"))
    img, Hs = _scene_with_boards(specs, px_per_m=8000.0, scale=0.18, noise=1.0)
    det = TargetDetector(specs).detect(img)
    assert set(det) == {"left", "right"}
    for arm in specs:
        assert len(det[arm].points) == 4


def test_detected_board_gives_good_pnp():
    specs = load_targets(os.path.join(CONFIG, "targets.yaml"))
    img, _ = _scene_with_boards(specs, px_per_m=4000.0, scale=0.3, noise=1.0)
    det = TargetDetector(specs).detect(img)
    K = np.array([[960.0, 0, 960], [0, 960, 540], [0, 0, 1]])
    for arm, d in det.items():
        ids, obj, pix = d.arrays()
        s = Sample(arm, np.eye(4), ids, obj, pix, K, np.zeros(5), (1920, 1080))
        cands = pnp_candidates(s)
        assert cands and cands[0][1] < 0.8


@pytest.mark.parametrize("cfg", ["targets.yaml", "targets_charuco_4x4_25mm.yaml", "targets_aruco.yaml"])
def test_generated_print_page_is_detected_with_correct_ids(tmp_path, cfg):
    """The actual print-ready PNG (not just the internal renderer) detects as the right arm only."""
    specs = load_targets(os.path.join(CONFIG, cfg))
    detector = TargetDetector(specs)
    for arm, spec in specs.items():
        png, _, _ = T.write_print_files(spec, str(tmp_path), dpi=300)
        page = cv2.imread(png, cv2.IMREAD_GRAYSCALE)
        for marker_px in (25, 60):
            f = marker_px / (spec.marker_length * 1000 * 300 / 25.4)
            small = cv2.resize(page, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
            det = detector.detect(small)
            assert set(det) == {arm}, (arm, marker_px, set(det))
            assert det[arm].n_markers == spec.n_markers()
            if spec.type == "charuco":
                assert det[arm].n_chess == (spec.squares_x - 1) * (spec.squares_y - 1)
