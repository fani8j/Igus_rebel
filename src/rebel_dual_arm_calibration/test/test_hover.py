"""Approach assist geometry: crosshair detection (table board + hover sheet), frame conversion, hover TCP poses."""
import os

import cv2
import numpy as np
import pytest
import yaml

from rebel_dual_arm_calibration import hover as H
from rebel_dual_arm_calibration import targets as T
from rebel_dual_arm_calibration.mock_common import BoardImage
from rebel_dual_arm_calibration.solver import TableObservation, pnp_candidates
from rebel_dual_arm_calibration.targets import TargetDetector, TargetSpec, load_targets_all
from rebel_dual_arm_calibration.transforms import inv_T, make_T

CONFIG = os.path.join(os.path.dirname(__file__), "..", "config")
TARGETS = os.path.join(CONFIG, "targets.yaml")
W, HGT = 1920, 1080
K = np.array([[1400.0, 0, 960.0], [0, 1400.0, 540.0], [0, 0, 1.0]])
D0 = np.zeros(5)
D1 = np.array([-0.08, 0.05, 0.0005, -0.0004, 0.0])

SHEET_SPEC = T.HoverSheetSpec(
    13, "DICT_4X4_50", 0.060, 0.060,
    {101: (0.0, 0.060), 102: (0.060, 0.0), 103: (0.0, -0.060), 104: (-0.060, 0.0),
     111: (-0.070, 0.035), 112: (0.020, -0.085), 113: (0.045, 0.060)},
    {101: "TOP", 102: "RIGHT", 103: "BOTTOM", 104: "LEFT", 111: "P2", 112: "P3", 113: "P1"})

F_FLIP = np.diag([1.0, -1.0, -1.0])      # marker frame (x right, y up, z out)  ->  board renderer frame (y down, z in)


def rx(deg):
    a = np.radians(deg)
    return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])


def ry(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def rz(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def table_pose(tilt_x=-25.0, tilt_y=15.0, t=(-0.12, 0.02, 0.9)):
    """T_optical_board of the table board (x right, y down, z into the board), facing the camera when untilted."""
    return make_T(rx(tilt_x) @ ry(tilt_y), t)


def sheet_pose(tilt_x=-30.0, tilt_y=12.0, spin=0.0, t=(0.15, 0.03, 0.9)):
    """T_optical_marker of the sheet (x right, y up as printed, z out of the paper); upright when spin = 0."""
    return make_T(rx(tilt_x) @ ry(tilt_y) @ rz(spin) @ F_FLIP, t)


class Scene:
    def __init__(self, dist=D0, seed=3):
        arms, self.table = load_targets_all(TARGETS)
        self.sheet_board_spec = TargetSpec("sheet", "aruco", "DICT_4X4_50", 0.060, [13])
        self.boards = {"table": BoardImage(self.table, margin=0.010, plate=0.0, touch_crosshairs=True),
                       "sheet": BoardImage(self.sheet_board_spec, margin=0.030, plate=0.0)}
        self.dist = dist
        self.rng = np.random.default_rng(seed)
        rr = np.random.default_rng(1)
        yy, xx = np.mgrid[0:HGT, 0:W].astype(np.float32)
        self.bg = np.repeat((150 + 30 * (yy / HGT)).astype(np.uint8)[:, :, None], 3, axis=2)
        grid = np.stack(np.meshgrid(np.arange(W, dtype=np.float32), np.arange(HGT, dtype=np.float32)), -1).reshape(-1, 1, 2)
        und = cv2.undistortPoints(grid, K, dist, P=K).reshape(HGT, W, 2)
        self.mx, self.my = und[..., 0].astype(np.float32), und[..., 1].astype(np.float32)
        del rr

    def render(self, T_table=None, T_marker=None, noise=2.0, blur=0.7):
        frame = self.bg.copy()
        if T_table is not None:
            assert self.boards["table"].warp_into(frame, K, T_table)
        if T_marker is not None:
            assert self.boards["sheet"].warp_into(frame, K, T_marker @ make_T(F_FLIP, np.zeros(3)))
        if np.any(self.dist != 0):
            frame = cv2.remap(frame, self.mx, self.my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        frame = cv2.GaussianBlur(frame, (0, 0), blur)
        return np.clip(frame.astype(np.float32) + self.rng.normal(0, noise, frame.shape[:2])[:, :, None], 0, 255).astype(np.uint8)


@pytest.fixture(scope="module", params=["pinhole", "distorted"])
def scene(request):
    return Scene(D0 if request.param == "pinhole" else D1)


def truth_table(spec, Tb):
    return {pid: (Tb[:3, :3] @ np.array([x, y, 0.0]) + Tb[:3, 3]) for pid, (x, y) in spec.touch_points.items()}


def truth_sheet(spec, Tm):
    return {pid: (Tm[:3, :3] @ np.array([x, y, 0.0]) + Tm[:3, 3]) for pid, (x, y) in spec.points.items()}


def frames_of(scene, n=5, **kw):
    return [(scene.render(**kw), K, scene.dist) for _ in range(n)]


# ---------------------------------------------------------------------------------------------------------------
# detection
# ---------------------------------------------------------------------------------------------------------------
def test_both_sources_recovered_within_1mm_and_1deg(scene):
    Tt, Tm = table_pose(), sheet_pose()
    pts, status = H.detect_targets(frames_of(scene, T_table=Tt, T_marker=Tm), scene.table, SHEET_SPEC,
                                   TargetDetector({"table": scene.table}))
    assert set(pts) == {1, 2, 3, 4} | set(SHEET_SPEC.points), status
    assert status["table"].startswith("table: ") and "corners" in status["table"]
    assert status["sheet"].startswith("sheet: marker 13")
    tt, ts = truth_table(scene.table, Tt), truth_sheet(SHEET_SPEC, Tm)
    for pid, tp in pts.items():
        truth = tt[pid] if tp.source == "table" else ts[pid]
        n_true = -Tt[:3, 2] if tp.source == "table" else Tm[:3, 2]
        x_true = Tt[:3, 0] if tp.source == "table" else Tm[:3, 0]
        err_mm = np.linalg.norm(tp.p_optical - truth) * 1e3
        ang = np.degrees(np.arccos(np.clip(tp.n_optical @ n_true, -1, 1)))
        assert err_mm < 1.0, (pid, tp.name, err_mm)
        assert ang < 1.0, (pid, tp.name, ang)
        assert np.degrees(np.arccos(np.clip(tp.x_optical @ x_true, -1, 1))) < 1.0
        assert np.linalg.norm(tp.n_optical) == pytest.approx(1.0) and abs(tp.n_optical @ tp.x_optical) < 1e-9
        assert tp.n_optical @ tp.p_optical < 0                     # normal points towards the camera
        q = tp.quality
        assert q["frames"] == 5 and q["spread_mm"] <= 1.0 and 0.8 < q["distance_m"] < 1.0
        assert 0 <= q["view_angle_deg"] < 70 and q["reproj_px"] < 1.0 and q["n_corners"] >= 4
    assert pts[1].name == "board 1" and pts[1].source == "table"
    assert pts[101].name == "TOP" and pts[101].source == "sheet"


def test_sheet_orientation_top_is_up_and_right_is_right(scene):
    """Upright print (TOP towards the top of the image): TOP projects above the centre, RIGHT to the right of it."""
    Tm = sheet_pose(tilt_x=-35.0, tilt_y=0.0)
    pts, status = H.detect_targets(frames_of(scene, T_marker=Tm), None, SHEET_SPEC, None)
    assert set(pts) == set(SHEET_SPEC.points), status
    assert "table" in status
    uv = H.project_points(pts, K, scene.dist)
    c = np.array(cv2.projectPoints(Tm[:3, 3].reshape(1, 1, 3), np.zeros(3), np.zeros(3), K, scene.dist)[0].reshape(2))
    top, bottom, right, left = (np.array(uv[i]) - c for i in (101, 103, 102, 104))
    assert top[1] < -20 and abs(top[0]) < 15            # TOP: smaller v (up in the image), same column
    assert bottom[1] > 20
    assert right[0] > 20 and abs(right[1]) < 15
    assert left[0] < -20
    # P2 (-70, +35): left and up; P3 (+20, -85): below, slightly right; P1 (+45, +60): up and right
    p2, p3, p1 = (np.array(uv[i]) - c for i in (111, 112, 113))
    assert p2[0] < 0 and p2[1] < 0 and p3[1] > 0 and p3[0] > 0 and p1[0] > 0 and p1[1] < 0
    # axis vectors in the optical frame: marker +x is image right, "up as read" (n x x) is image up (optical -y)
    tp = pts[101]
    assert tp.x_optical[0] > 0.9
    up = np.cross(tp.n_optical, tp.x_optical)
    assert up[1] < -0.5
    # and the pixel positions match the pixels of the truth
    for pid, pxy in uv.items():
        truth_px = cv2.projectPoints((Tm[:3, :3] @ np.array([*SHEET_SPEC.points[pid], 0.0]) + Tm[:3, 3]).reshape(1, 1, 3),
                                     np.zeros(3), np.zeros(3), K, scene.dist)[0].reshape(2)
        assert np.linalg.norm(np.array(pxy) - truth_px) < 1.5


def test_table_crosshairs_land_on_the_inner_corners(scene):
    """Ids 1-4 are the inner corners (1,1) (4,1) (4,3) (1,3): same board coordinates the solver/recorder use."""
    spec = scene.table
    Tt = table_pose(-20.0, 10.0)
    frames = frames_of(scene, T_table=Tt)
    pts, status = H.detect_targets(frames, spec, None, TargetDetector({"table": spec}))
    assert set(pts) == {1, 2, 3, 4}, status
    assert status["sheet"].startswith("sheet: no hover sheet")
    # the solver's own conversion: PnP of the detected corners, then the touch point as the board point (x, y, 0)
    bgr, _, _ = frames[0]
    ids, obj, pix = TargetDetector({"table": spec}).detect(bgr)["table"].arrays()
    cand = pnp_candidates(TableObservation(ids=ids, obj=obj, pix=pix, K=K, dist=scene.dist, image_size=(W, HGT)))[0][0]
    for pid, (x, y) in spec.touch_points.items():
        solver_p = cand[:3, :3] @ np.array([x, y, 0.0]) + cand[:3, 3]
        assert np.linalg.norm(solver_p - pts[pid].p_optical) < 0.5e-3
    # geometry: inner corner (i, j) of the 5x4 board with 40 mm squares (spec says sq 0.0398 x 0.040? no: targets.yaml 0.040)
    assert pts[1].p_optical == pytest.approx(truth_table(spec, Tt)[1], abs=1e-3)
    corner = {1: (1, 1), 2: (4, 1), 3: (4, 3), 4: (1, 3)}
    chess = {tuple(np.round(p[:2], 9)): pid for pid, p in spec.chess_corners_obj()}
    for pid, _ in corner.items():
        assert tuple(np.round(spec.touch_points[pid], 9)) in chess
    # the projected crosshair sits on a chessboard saddle of the image (rendered without a crosshair)
    plain = Scene(scene.dist)
    plain.boards["table"] = BoardImage(spec, margin=0.010, plate=0.0, touch_crosshairs=False)
    gray = cv2.cvtColor(plain.render(T_table=Tt, noise=0.0), cv2.COLOR_BGR2GRAY)
    uv = H.project_points(pts, K, scene.dist)
    for pid, (u, v) in uv.items():
        c = np.array([[[u, v]]], np.float32)
        cv2.cornerSubPix(gray, c, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3))
        assert np.hypot(c[0, 0, 0] - u, c[0, 0, 1] - v) < 1.0, (pid, c, u, v)


def test_anisotropic_sheet_pose():
    sc = Scene(D0)
    sc.sheet_board_spec = TargetSpec("sheet", "aruco", "DICT_4X4_50", 0.0594, [13], marker_length_y=0.0588)
    # fine raster: the 0.6 mm x/y difference is below the 0.25 mm pixel pitch of the default 4000 px/m board image
    sc.boards["sheet"] = BoardImage(sc.sheet_board_spec, px_per_m=16000.0, margin=0.030, plate=0.0)
    aspec = T.HoverSheetSpec(13, "DICT_4X4_50", 0.0594, 0.0588, {101: (0.0, 0.0588), 102: (0.0594, 0.0)},
                             {101: "TOP", 102: "RIGHT"})
    Tm = sheet_pose()
    pts, status = H.detect_targets(frames_of(sc, T_marker=Tm), None, aspec, None)
    assert set(pts) == {101, 102}, status
    for pid, (x, y) in aspec.points.items():
        truth = Tm[:3, :3] @ np.array([x, y, 0.0]) + Tm[:3, 3]
        assert np.linalg.norm(pts[pid].p_optical - truth) < 1e-3


# ---------------------------------------------------------------------------------------------------------------
# gates and status
# ---------------------------------------------------------------------------------------------------------------
def test_multi_frame_spread_gate_and_min_frames():
    sc = Scene(D0)
    base = np.array([0.15, 0.03, 0.9])
    frames = []
    for k in range(5):
        Tm = sheet_pose(t=base + np.array([0.004 * (k % 2), 0.0, 0.0]))       # sheet moves 4 mm between frames
        frames.append((sc.render(T_table=table_pose(), T_marker=Tm), K, D0))
    pts, status = H.detect_targets(frames, sc.table, SHEET_SPEC, None)
    assert set(pts) == {1, 2, 3, 4}                                           # table is fine, sheet is rejected
    assert "spread" in status["sheet"] and "mm" in status["sheet"] and "not used" in status["sheet"]
    # the same frames are accepted with a loose gate (and report the spread)
    pts2, _ = H.detect_targets(frames, sc.table, SHEET_SPEC, None, max_spread_mm=10.0)
    assert set(SHEET_SPEC.points) <= set(pts2) and pts2[101].quality["spread_mm"] > 1.5
    # fewer than 3 frames
    two, status2 = H.detect_targets(frames_of(sc, n=2, T_table=table_pose(), T_marker=sheet_pose()),
                                    sc.table, SHEET_SPEC, None)
    assert not two and "2 of 2" in status2["table"] and "2 of 2" in status2["sheet"]


def test_missing_marker_and_missing_table_status():
    sc = Scene(D0)
    pts, status = H.detect_targets(frames_of(sc, n=3, T_table=table_pose()), sc.table, SHEET_SPEC, None)
    assert set(pts) == {1, 2, 3, 4}
    assert status["sheet"] == "sheet: marker 13 not found (0 of 3 frames)"
    pts, status = H.detect_targets(frames_of(sc, n=3, T_marker=sheet_pose()), sc.table, SHEET_SPEC, None)
    assert set(pts) == set(SHEET_SPEC.points)
    assert status["table"].startswith("table: table board not found")
    pts, status = H.detect_targets([], sc.table, SHEET_SPEC, None)
    assert pts == {} and "no camera frames" in status["table"]
    pts, status = H.detect_targets(frames_of(sc, n=3), None, None, None)
    assert pts == {} and "configured" in status["table"] and "configured" in status["sheet"]


def test_sheet_gates_reject_with_a_reason(monkeypatch):
    sc = Scene(D0)
    frames = frames_of(sc, n=3, T_marker=sheet_pose())
    monkeypatch.setattr(H, "SHEET_MIN_IPPE_RATIO", 1e9)                      # every pose "ambiguous"
    pts, status = H.detect_targets(frames, None, SHEET_SPEC, None)
    assert not pts and "ambiguous IPPE" in status["sheet"] and "0 of 3" in status["sheet"]
    monkeypatch.setattr(H, "SHEET_MIN_IPPE_RATIO", 1.5)
    monkeypatch.setattr(H, "SHEET_MIN_SIDE_PX", 500.0)
    pts, status = H.detect_targets(frames, None, SHEET_SPEC, None)
    assert not pts and "side" in status["sheet"]
    monkeypatch.setattr(H, "SHEET_MIN_SIDE_PX", 30.0)
    monkeypatch.setattr(H, "SHEET_MAX_REPROJ_PX", 1e-6)
    pts, status = H.detect_targets(frames, None, SHEET_SPEC, None)
    assert not pts and "reprojection" in status["sheet"]


def test_marker_found_in_too_few_frames_reports_the_reason():
    sc = Scene(D0)
    frames = frames_of(sc, n=2, T_marker=sheet_pose()) + frames_of(sc, n=2)
    pts, status = H.detect_targets(frames, None, SHEET_SPEC, None)
    assert not pts and "only 2 of 4" in status["sheet"] and "not found" in status["sheet"]


# ---------------------------------------------------------------------------------------------------------------
# frames, hover poses, compare
# ---------------------------------------------------------------------------------------------------------------
def _tp(pid=7, source="sheet"):
    n = np.array([0.1, -0.7, -0.7]); n /= np.linalg.norm(n)
    x = np.array([1.0, 0.1, 0.0]); x -= (x @ n) * n; x /= np.linalg.norm(x)
    return H.TargetPoint(pid, "TOP", source, np.array([0.1, 0.0, 0.9]), n, x, {})


def test_to_base_with_known_X():
    X = make_T(rz(40) @ rx(150), [0.3, -0.1, 0.8])          # T_optical_base
    p_base, n_base = np.array([0.25, 0.1, 0.02]), np.array([0.0, 0.0, 1.0])
    x_base = np.array([1.0, 0.0, 0.0])
    # build the optical-frame target from base-frame truth
    tp = H.TargetPoint(5, "TOP", "sheet", X[:3, :3] @ p_base + X[:3, 3], X[:3, :3] @ n_base, X[:3, :3] @ x_base, {})
    out = H.to_base({5: tp}, X)
    assert set(out) == {5} and out[5]["name"] == "TOP" and out[5]["source"] == "sheet"
    assert out[5]["p"] == pytest.approx(p_base, abs=1e-12)
    assert out[5]["n"] == pytest.approx(n_base, abs=1e-12) and out[5]["x"] == pytest.approx(x_base, abs=1e-12)
    # inv(X) applied: the optical origin is at inv(X)[:3, 3] in the base frame
    origin_tp = H.TargetPoint(1, "o", "table", np.zeros(3), np.array([0, 0, -1.0]), np.array([1.0, 0, 0]), {})
    assert H.to_base({1: origin_tp}, X)[1]["p"] == pytest.approx(inv_T(X)[:3, 3])


def test_hover_tcp_poses():
    n = np.array([0.0, 0.3, 0.9539392]); n /= np.linalg.norm(n)
    x = np.array([1.0, 0.2, -0.1])
    p = np.array([0.3, 0.1, 0.05])
    poses = H.hover_tcp_poses(p, n, x, height_m=0.012, yaw_step_deg=30.0)
    assert len(poses) == 12
    xs = []
    for T_ in poses:
        R = T_[:3, :3]
        assert R.T @ R == pytest.approx(np.eye(3), abs=1e-12) and np.linalg.det(R) == pytest.approx(1.0)
        assert R[:, 2] == pytest.approx(-n, abs=1e-12)                         # TCP +Z into the paper
        assert T_[:3, 3] == pytest.approx(p + 0.012 * n, abs=1e-12)              # hover height along +n
        assert (T_[:3, 3] - p) @ n == pytest.approx(0.012)
        xs.append(R[:, 0])
    # first pose: TCP x = board x projected into the plane
    xp = x - (x @ n) * n
    assert xs[0] == pytest.approx(xp / np.linalg.norm(xp), abs=1e-12)
    for a, b in zip(xs, xs[1:] + xs[:1]):                                      # yaw spacing 30 deg about the normal
        assert np.degrees(np.arccos(np.clip(a @ b, -1, 1))) == pytest.approx(30.0, abs=1e-9)
        assert np.cross(a, b) @ n > 0                                          # consistent turning direction
    assert len(H.hover_tcp_poses(p, n, x, yaw_step_deg=45.0)) == 8
    assert H.hover_tcp_poses(p, n, x)[0][:3, 3] == pytest.approx(p + 0.01 * n)  # default height 10 mm
    with pytest.raises(ValueError):
        H.hover_tcp_poses(p, n, x, yaw_step_deg=0.0)
    # degenerate x parallel to n still yields a valid frame
    assert np.linalg.det(H.hover_tcp_poses(p, n, n)[0][:3, :3]) == pytest.approx(1.0)


def test_compare_signs():
    n = np.array([0.0, 0.0, 1.0]); x = np.array([1.0, 0.0, 0.0])          # y_up = n x x = +y
    pred = {"p": np.array([0.3, 0.1, 0.04]), "n": n, "x": x, "name": "TOP", "source": "sheet"}
    r = H.compare(pred, pred["p"] + np.array([0.002, 0.0, 0.0]))
    assert r["dx_mm"] == pytest.approx(2.0) and r["dy_mm"] == pytest.approx(0.0, abs=1e-9)
    assert r["dn_mm"] == pytest.approx(0.0, abs=1e-9) and r["lateral_mm"] == pytest.approx(2.0) \
        and r["dist_mm"] == pytest.approx(2.0)
    r = H.compare(pred, pred["p"] + np.array([0.0, 0.003, 0.001]))
    assert r["dy_mm"] == pytest.approx(3.0) and r["dn_mm"] == pytest.approx(1.0)            # above the paper = +
    assert r["lateral_mm"] == pytest.approx(3.0) and r["dist_mm"] == pytest.approx(np.hypot(3.0, 1.0))
    r = H.compare(pred, pred["p"] - 0.0005 * n)
    assert r["dn_mm"] == pytest.approx(-0.5)
    # a tilted frame: along-x is measured in-plane (x slightly off the plane is projected)
    n2 = np.array([0.0, np.sin(0.4), np.cos(0.4)]); x2 = np.array([1.0, 0.0, 0.0]); y2 = np.cross(n2, x2)
    pred2 = dict(pred, n=n2, x=x2)
    r = H.compare(pred2, pred2["p"] + 0.002 * x2 - 0.001 * y2 + 0.004 * n2)
    assert (r["dx_mm"], r["dy_mm"], r["dn_mm"]) == pytest.approx((2.0, -1.0, 4.0))


def test_project_points_skips_points_behind_the_camera():
    pts = {1: H.TargetPoint(1, "a", "table", np.array([0.0, 0.0, 1.0]), np.array([0, 0, -1.0]), np.array([1.0, 0, 0]), {}),
           2: H.TargetPoint(2, "b", "table", np.array([0.0, 0.0, -1.0]), np.array([0, 0, -1.0]), np.array([1.0, 0, 0]), {})}
    uv = H.project_points(pts, K, D0)
    assert set(uv) == {1} and uv[1] == pytest.approx((960.0, 540.0))


# ---------------------------------------------------------------------------------------------------------------
# loader
# ---------------------------------------------------------------------------------------------------------------
def _cfg():
    cfg = yaml.safe_load(open(TARGETS))
    return cfg


def test_packaged_configs_have_hover_tiles_where_markers_13_to_15_are_free():
    for name in ("targets.yaml", "targets_charuco_4x4_25mm.yaml"):
        tiles = T.load_hover_tiles(os.path.join(CONFIG, name))
        assert tiles is not None and [t.marker_id for t in tiles.tiles] == [13, 14, 15]
        assert [sorted(t.points) for t in tiles.tiles] == [[101, 102, 103], [104, 105, 106], [111, 112, 113]]
        assert T.load_hover_sheet(os.path.join(CONFIG, name)) is None            # the legacy single sheet is gone from them
        # ids also in touch_stickers are allowed; table board and stickers still load
        T.load_stickers(os.path.join(CONFIG, name))
    assert T.load_hover_tiles(os.path.join(CONFIG, "targets_aruco.yaml")) is None   # arms use ids 13/14: section disabled
    assert T.load_hover_sheet(os.path.join(CONFIG, "targets_aruco.yaml")) is None


def test_legacy_hover_sheet_section_still_loads_as_one_sheet():
    cfg = _cfg()
    cfg.pop("hover_tiles")
    cfg["hover_sheet"] = {"dictionary": "DICT_4X4_50", "marker_id": 13, "design_marker_length": 0.060,
                          "marker_length": 0.059, "marker_length_y": 0.059,
                          "points": {101: {"name": "TOP", "xy_mm": [0, 60]}, 111: {"name": "P2", "xy_mm": [-70, 35]}}}
    sheets = T.hover_sheets_from_config(cfg)
    assert len(sheets) == 1 and sheets[0].source == "sheet" and sheets[0].names[101] == "TOP"
    assert sheets[0].points_mm[111] == (-70.0, 35.0)
    assert T.hover_sheets_from_config({k: v for k, v in cfg.items() if k != "hover_sheet"}) == []


def test_loader_scaling_and_defaults():
    cfg = _cfg()
    cfg["hover_sheet"] = {"dictionary": "DICT_4X4_50", "marker_id": 13, "marker_length": 0.059,
                          "points": {101: {"name": "TOP", "xy_mm": [0, 60]}, 111: {"name": "P2", "xy_mm": [-70, 35]}}}
    s = T.hover_sheet_from_config(cfg)
    assert (s.marker_id, s.dictionary, s.marker_length, s.marker_length_y) == (13, "DICT_4X4_50", 0.059, 0.059)
    assert s.points[101] == pytest.approx((0.0, 0.060 * 0.059 / 0.060))
    assert s.points[111] == pytest.approx((-0.070 * 0.059 / 0.060, 0.035 * 0.059 / 0.060))
    assert s.design_marker_length == 0.060 and not s.anisotropic
    # anisotropic print: x and y scaled separately
    cfg["hover_sheet"].update(marker_length=0.0594, marker_length_y=0.0588)
    s = T.hover_sheet_from_config(cfg)
    assert s.anisotropic
    assert s.points[111] == pytest.approx((-0.070 * 0.0594 / 0.060, 0.035 * 0.0588 / 0.060))
    assert s.points[101] == pytest.approx((0.0, 0.060 * 0.0588 / 0.060))
    # explicit design length
    cfg["hover_sheet"].update(marker_length=0.050, marker_length_y=0.050, design_marker_length=0.050)
    assert T.hover_sheet_from_config(cfg).points[101] == pytest.approx((0.0, 0.060))
    # absent section
    assert T.hover_sheet_from_config({k: v for k, v in _cfg().items() if k != "hover_sheet"}) is None


@pytest.mark.parametrize("patch,match", [
    ({"marker_id": 0}, "already used"),                      # left arm board
    ({"marker_id": 41}, "already used"),                     # table board
    ({"marker_id": 99}, "outside dictionary"),
    ({"marker_length": 0.0}, "> 0"),
    ({"marker_length": float("nan")}, "finite"),
    ({"marker_length_y": float("inf")}, "finite"),
    ({"marker_length": "abc"}, "number"),
    ({"points": {1: {"name": "x", "xy_mm": [0, 0]}}}, "collides"),            # table touch id
    ({"points": {101: {"name": "x", "xy_mm": [float("nan"), 0]}}}, "finite"),
    ({"points": {101: {"name": "x", "xy_mm": [0]}}}, "xy_mm"),
    ({"points": {101: {"name": "x"}}}, "xy_mm"),
    ({"points": {0: {"name": "x", "xy_mm": [0, 0]}}}, "positive"),
    ({"points": {}}, "no points"),
])
def test_loader_validation(patch, match):
    cfg = _cfg()
    hs = {"dictionary": "DICT_4X4_50", "marker_id": 13, "marker_length": 0.059,
          "points": {101: {"name": "TOP", "xy_mm": [0, 60]}}}
    hs.update(patch)
    cfg["hover_sheet"] = hs
    with pytest.raises(ValueError, match=match):
        T.hover_sheet_from_config(cfg)


def test_loader_duplicate_point_ids_in_list_form():
    cfg = _cfg()
    cfg["hover_sheet"] = {"marker_id": 13, "marker_length": 0.06,
                          "points": [{"id": 101, "name": "A", "xy_mm": [0, 1]}, {"id": 102, "name": "B", "xy_mm": [1, 0]}]}
    assert sorted(T.hover_sheet_from_config(cfg).points) == [101, 102]


# ---------------------------------------------------------------------------------------------------------------
# unmoved(): "has the paper moved?" under partial occlusion (the descend gate)
# ---------------------------------------------------------------------------------------------------------------
def _stored(scene, Tt, Tm):
    det = TargetDetector({"table": scene.table})
    pts, status = H.detect_targets(frames_of(scene, n=5, T_table=Tt, T_marker=Tm), scene.table, SHEET_SPEC, det)
    assert set(pts) == ({1, 2, 3, 4} if Tt is not None else set()) | (set(SHEET_SPEC.points) if Tm is not None else set()), status
    return pts, det


def _table_corner_pix(scene, frame):
    d = TargetDetector({"table": scene.table}).detect(frame)["table"]
    return d.arrays()[2]


def _blank(frame, x0, y0, x1, y1, grey=150):
    out = frame.copy()
    out[y0:y1, x0:x1] = grey
    return out


def _occluded_frames(scene, Tt, Tm, frac=0.5, n=3):
    """n frames in which the right `frac` of the table board is hidden by a blank rectangle (the arm)."""
    frames = []
    for _ in range(n):
        f = scene.render(T_table=Tt, T_marker=Tm)
        pix = _table_corner_pix(scene, f)
        lo, hi = pix[:, 0].min(), pix[:, 0].max()
        frames.append((_blank(f, int(lo + (1 - frac) * (hi - lo)), 0, W, HGT), K, scene.dist))
    return frames


def test_stored_source_poses_rebuild_the_detected_board_and_marker_poses(scene):
    Tt, Tm = table_pose(), sheet_pose()
    pts, _ = _stored(scene, Tt, Tm)
    poses = H.stored_source_poses(pts, scene.table, SHEET_SPEC)
    assert set(poses) == {"table", "sheet"}
    for key, truth in (("table", Tt), ("sheet", Tm)):
        assert np.linalg.norm(poses[key][:3, 3] - truth[:3, 3]) < 1e-3
        assert np.degrees(np.arccos(np.clip((np.trace(poses[key][:3, :3].T @ truth[:3, :3]) - 1) / 2, -1, 1))) < 0.5
    assert H.stored_source_poses({}, scene.table, SHEET_SPEC) == {}
    assert set(H.stored_source_poses({k: v for k, v in pts.items() if v.source == "sheet"}, scene.table, SHEET_SPEC)) == {"sheet"}


def test_unmoved_passes_with_the_board_fully_visible_and_with_half_of_it_hidden(scene):
    Tt, Tm = table_pose(), sheet_pose()
    pts, det = _stored(scene, Tt, Tm)
    ok, why, stats = H.unmoved(frames_of(scene, n=3, T_table=Tt, T_marker=Tm), pts, scene.table, SHEET_SPEC, det)
    assert ok and why == "", (why, stats)
    assert stats["table"]["verdicts"] == ["ok"] * 3 and max(stats["table"]["median_px"]) < 0.6
    full = min(stats["table"]["visible"])
    ok, why, stats = H.unmoved(_occluded_frames(scene, Tt, None, 0.5), {k: v for k, v in pts.items() if v.source == "table"},
                               scene.table, None, det, sources=["table"])
    vis = stats["table"]["visible"]
    assert ok, (why, stats)
    assert H.UNMOVED_MIN_CORNERS <= min(vis) <= 0.7 * full                    # about half of the corners were really hidden
    assert max(stats["table"]["median_px"]) <= 1.0 and max(stats["table"]["p90_px"]) <= 2.0


def test_unmoved_catches_a_2_mm_shift_with_the_board_half_hidden(scene):
    Tt = table_pose()
    pts, det = _stored(scene, Tt, sheet_pose())
    tab = {k: v for k, v in pts.items() if v.source == "table"}
    Tshift = Tt.copy()
    Tshift[:3, 3] += Tt[:3, :3] @ np.array([0.002, 0.0, 0.0])                  # 2 mm along the board's x (about 3 px)
    ok, why, stats = H.unmoved(frames_of(scene, n=3, T_table=Tshift), tab, scene.table, None, det, sources=["table"])
    assert not ok and why.startswith(H.MOVED_REASON) and "median" in why, why
    assert stats["table"]["verdicts"] == ["moved"] * 3
    ok, why, stats = H.unmoved(_occluded_frames(scene, Tshift, None, 0.5), tab, scene.table, None, det, sources=["table"])
    assert not ok and why.startswith(H.MOVED_REASON)
    # a 0.4 mm shift is below what one pixel resolves here and still passes
    Tsmall = Tt.copy()
    Tsmall[:3, 3] += Tt[:3, :3] @ np.array([0.0004, 0.0, 0.0])
    ok, why, _ = H.unmoved(frames_of(scene, n=3, T_table=Tsmall), tab, scene.table, None, det, sources=["table"])
    assert ok, why
    # a turn of the board by 3 deg in its own plane moves the corners by several mm in the image: caught
    # (a depth-only shift or a small out-of-plane tilt is NOT resolved by this test - see the comment in hover.py)
    Ttilt = Tt @ make_T(rz(3.0), np.zeros(3))
    ok, why, _ = H.unmoved(frames_of(scene, n=3, T_table=Ttilt), tab, scene.table, None, det, sources=["table"])
    assert not ok and why.startswith(H.MOVED_REASON)


def test_unmoved_one_bad_frame_fails_all_frames_must_pass(scene):
    Tt = table_pose()
    pts, det = _stored(scene, Tt, None)
    Tshift = Tt.copy()
    Tshift[:3, 3] += Tt[:3, :3] @ np.array([0.003, 0.0, 0.0])
    frames = frames_of(scene, n=2, T_table=Tt) + frames_of(scene, n=1, T_table=Tshift)
    ok, why, stats = H.unmoved(frames, pts, scene.table, None, det, sources=["table"])
    assert not ok and why.startswith(H.MOVED_REASON) and stats["table"]["verdicts"] == ["ok", "ok", "moved"]


def test_unmoved_hidden_board_is_refused_with_the_hidden_text(scene):
    Tt = table_pose()
    pts, det = _stored(scene, Tt, None)
    tab = {k: v for k, v in pts.items() if v.source == "table"}
    frames = []
    for _ in range(3):                                                         # only a thin strip of the board stays visible
        f = scene.render(T_table=Tt)
        pix = _table_corner_pix(scene, f)
        lo, hi = pix[:, 0].min(), pix[:, 0].max()
        frames.append((_blank(f, int(lo + 0.12 * (hi - lo)), 0, W, HGT), K, scene.dist))
    ok, why, stats = H.unmoved(frames, tab, scene.table, None, det, sources=["table"])
    assert not ok and why.startswith(H.HIDDEN_REASON) and "cannot confirm it has not moved" in why and "teleop" in why
    assert all(v == "hidden" for v in stats["table"]["verdicts"]) and max(stats["table"]["visible"]) < H.UNMOVED_MIN_CORNERS
    # nothing of the board at all
    ok, why, _ = H.unmoved(frames_of(scene, n=3), tab, scene.table, None, det, sources=["table"])
    assert not ok and why.startswith(H.HIDDEN_REASON) and "table board not found" in why
    # too few fresh frames
    ok, why, _ = H.unmoved(frames_of(scene, n=2, T_table=Tt), tab, scene.table, None, det, sources=["table"])
    assert not ok and why.startswith(H.HIDDEN_REASON) and "only 2 fresh frames" in why


def test_unmoved_corners_on_one_row_are_not_enough(scene, monkeypatch):
    Tt = table_pose()
    pts, det = _stored(scene, Tt, None)
    tab = {k: v for k, v in pts.items() if v.source == "table"}
    monkeypatch.setattr(H, "UNMOVED_MIN_LINES", 10 ** 6)                       # no board can spread over that many rows
    ok, why, _ = H.unmoved(frames_of(scene, n=3, T_table=Tt), tab, scene.table, None, det, sources=["table"])
    assert not ok and why.startswith(H.HIDDEN_REASON) and "row(s)" in why


def test_unmoved_sheet_all_four_corners_and_1_5_px(scene):
    Tm = sheet_pose()
    pts, det = _stored(scene, table_pose(), Tm)
    sheet = {k: v for k, v in pts.items() if v.source == "sheet"}
    ok, why, stats = H.unmoved(frames_of(scene, n=3, T_marker=Tm), sheet, None, SHEET_SPEC, det, sources=["sheet"])
    assert ok and stats["sheet"]["verdicts"] == ["ok"] * 3 and max(stats["sheet"]["median_px"]) < 0.8, (why, stats)
    Tshift = Tm.copy()
    Tshift[:3, 3] += Tm[:3, :3] @ np.array([0.002, 0.0, 0.0])                  # 2 mm = ~3 px
    ok, why, stats = H.unmoved(frames_of(scene, n=3, T_marker=Tshift), sheet, None, SHEET_SPEC, det, sources=["sheet"])
    assert not ok and why.startswith(H.MOVED_REASON) and "median reprojection" in why
    # hidden: no marker, or a corner of it covered (then the marker is not found at all)
    ok, why, _ = H.unmoved(frames_of(scene, n=3), sheet, None, SHEET_SPEC, det, sources=["sheet"])
    assert not ok and why.startswith(H.HIDDEN_REASON) and "marker 13 not found" in why
    covered = []
    for _ in range(3):
        f = scene.render(T_marker=Tm)
        c = det.detect(f)                                                      # (arm boards only: find the marker by projection)
        uv = H._project(Tm, SHEET_SPEC.marker_corners_obj(), K, scene.dist)
        covered.append((_blank(f, int(uv[:, 0].mean()), int(uv[:, 1].min()) - 5, int(uv[:, 0].max()) + 5, int(uv[:, 1].max()) + 5), K, scene.dist))
        assert c is not None
    ok, why, _ = H.unmoved(covered, sheet, None, SHEET_SPEC, det, sources=["sheet"])
    assert not ok and why.startswith(H.HIDDEN_REASON)


def test_unmoved_checks_only_the_requested_source(scene):
    Tt, Tm = table_pose(), sheet_pose()
    pts, det = _stored(scene, Tt, Tm)
    frames = frames_of(scene, n=3, T_table=Tt)                                 # the sheet is not in view
    ok, why, stats = H.unmoved(frames, pts, scene.table, SHEET_SPEC, det, sources=["table"])
    assert ok and set(stats) == {"table"}
    ok, why, stats = H.unmoved(frames, pts, scene.table, SHEET_SPEC, det)      # default: every stored source
    assert not ok and why.startswith(H.HIDDEN_REASON) and set(stats) == {"table", "sheet"}
    ok, why, _ = H.unmoved(frames, {}, scene.table, SHEET_SPEC, det)
    assert not ok and "Detect points again" in why
    ok, why, _ = H.unmoved(frames, {k: v for k, v in pts.items() if v.source == "table"}, scene.table, SHEET_SPEC, det, sources=["sheet"])
    assert not ok and "no stored sheet pose" in why
