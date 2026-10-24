"""Robot work area (scene option ``live.work_area``, core/work_area.py): scene validation, polygon tests, cloud
crop, SAM3 candidate filter (image prism) and the pure corner check. The detector wiring (hard reject,
fail closed, flags do not bypass) is in test/live/test_work_area_gate.py."""
import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest

from box_seam_detection.core import pipeline as P
from box_seam_detection.core import scene as sc
from box_seam_detection.core import work_area as wa
from box_seam_detection.core.autolabel.pcd import read_pcd_xyz

BRIEF_POLYGON = [[0.11, -0.77], [0.28, -0.72], [0.32, -0.67], [0.48, -0.6], [0.65, -0.41],
                 [0.73, -0.24], [0.64, -0.03], [0.49, 0.14], [0.19, 0.29], [0.11, 0.29]]

SQUARE = {"frame": "base_link", "polygon_xy_m": [[0.0, -0.8], [0.8, -0.8], [0.8, 0.0], [0.0, 0.0]],
          "top_z_range_m": [-0.15, 0.30], "outside_keep_every": 10}


def _valid(**over):
    return wa.validate_work_area({**copy.deepcopy(SQUARE), **over})


# ---- validation -----------------------------------------------------------------

def test_valid_mapping_is_normalised():
    out = wa.validate_work_area({"frame": "base_link", "polygon_xy_m": [[0, 0], [1, 0], [1, 1]],
                                 "top_z_range_m": [0, 1], "outside_keep_every": 0})
    assert out["polygon_xy_m"] == [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]
    assert out["top_z_range_m"] == [0.0, 1.0] and out["outside_keep_every"] == 0
    assert out["margin_m"] == wa.DEFAULT_MARGIN_M
    assert wa.validate_work_area(None) is None
    assert wa.validate_work_area({**SQUARE, "margin_m": 0})["margin_m"] == 0.0
    assert out["floor_keep_below_z_m"] is None
    assert wa.validate_work_area({**SQUARE, "floor_keep_below_z_m": -0.75})["floor_keep_below_z_m"] == -0.75


@pytest.mark.parametrize("over,match", [
    ({"polygon_xy_m": [[0, 0], [1, 0]]}, "at least 3"),
    ({"polygon_xy_m": "x"}, "at least 3"),
    ({"polygon_xy_m": [[0, 0], [1, 0], [1, float("nan")]]}, "finite"),
    ({"polygon_xy_m": [[0, 0], [1, 0], [float("inf"), 1]]}, "finite"),
    ({"polygon_xy_m": [[0, 0], [1, 0], [1, 1, 1]]}, r"\[x, y\]"),
    ({"polygon_xy_m": [[0, 0], [1, 0], [True, 1]]}, "finite"),
    ({"polygon_xy_m": [[0, 0], [1, 1], [1, 0], [0, 1]]}, "not a simple polygon"),      # bow tie
    ({"polygon_xy_m": [[0, 0], [1, 0], [1, 0], [1, 1]]}, "zero-length"),
    ({"polygon_xy_m": [[0, 0], [1, 0], [2, 0]]}, "fold back|zero area"),
    ({"polygon_xy_m": [[0, 0], [2, 0], [1, 0], [1, 1]]}, "fold back"),
    ({"top_z_range_m": [0.3, 0.3]}, "zmin < zmax"),
    ({"top_z_range_m": [0.3, -0.1]}, "zmin < zmax"),
    ({"top_z_range_m": [0.0, float("nan")]}, "zmin < zmax"),
    ({"top_z_range_m": [0.0]}, "zmin < zmax"),
    ({"outside_keep_every": -1}, "outside_keep_every"),
    ({"outside_keep_every": 2.5}, "outside_keep_every"),
    ({"outside_keep_every": True}, "outside_keep_every"),
    ({"outside_keep_every": "10"}, "outside_keep_every"),
    ({"margin_m": -0.1}, "margin_m"),
    ({"margin_m": 9.0}, "margin_m"),
    ({"floor_keep_below_z_m": float("nan")}, "floor_keep_below_z_m"),
    ({"floor_keep_below_z_m": "low"}, "floor_keep_below_z_m"),
    ({"floor_keep_below_z_m": True}, "floor_keep_below_z_m"),
    ({"frame": "world"}, "frame"),
    ({"surprise": 1}, "unknown key"),
])
def test_invalid_mappings_fail_closed(over, match):
    with pytest.raises(wa.WorkAreaError, match=match):
        _valid(**over)


def test_not_a_mapping():
    with pytest.raises(wa.WorkAreaError, match="mapping"):
        wa.validate_work_area([1, 2, 3])


def test_concave_polygon_is_accepted_and_hash_is_stable():
    ell = _valid(polygon_xy_m=[[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]])
    assert len(ell["polygon_xy_m"]) == 6
    h = wa.polygon_hash(ell)
    assert h == wa.polygon_hash(copy.deepcopy(ell)) and len(h) == 12
    moved = copy.deepcopy(ell)
    moved["polygon_xy_m"][0][0] = 0.01
    assert wa.polygon_hash(moved) != h
    other_z = copy.deepcopy(ell)
    other_z["top_z_range_m"] = [-0.2, 0.3]
    assert wa.polygon_hash(other_z) != h


# ---- scene wiring -------------------------------------------------------------------

def _scene_file(tmp_path, body):
    path = tmp_path / "scene.yaml"
    path.write_text(f"scene: wa_test\nlive:\n{body}", encoding="utf-8")
    return path


def test_shipped_scenes():
    dual = sc.load_scene("dual_arm_v1")
    assert dual.work_area["polygon_xy_m"] == BRIEF_POLYGON
    assert dual.work_area["top_z_range_m"] == [-0.20, 0.06]
    assert dual.work_area["outside_keep_every"] >= 1 and dual.work_area["frame"] == "base_link"
    assert dual.work_area["floor_keep_below_z_m"] == -0.75         # dense floor: see DESIGN.md "Work area"
    assert sc.load_scene("table_cell_v1").work_area is None
    assert sc.builtin_default().work_area is None


def test_feature_off_when_absent(tmp_path):
    assert sc.load_scene(_scene_file(tmp_path, "  rect_tol_mm: null\n")).work_area is None
    assert sc.load_scene(_scene_file(tmp_path, "  work_area: null\n")).work_area is None


def test_scene_loads_a_valid_work_area_and_rejects_bad_ones(tmp_path):
    ok = _scene_file(tmp_path, "  work_area:\n    frame: base_link\n"
                     "    polygon_xy_m: [[0, 0], [1, 0], [1, 1]]\n    top_z_range_m: [0, 1]\n")
    s = sc.load_scene(ok)
    assert s.work_area["outside_keep_every"] == 10 and s.work_area["polygon_xy_m"][2] == [1.0, 1.0]
    for body in ("  work_area: 3\n",
                 "  work_area: {frame: base_link, polygon_xy_m: [[0, 0], [1, 0]], top_z_range_m: [0, 1]}\n",
                 "  work_area: {frame: base_link, polygon_xy_m: [[0, 0], [1, 1], [1, 0], [0, 1]], "
                 "top_z_range_m: [0, 1]}\n",
                 "  work_area: {frame: base_link, polygon_xy_m: [[0, 0], [1, 0], [1, 1]], "
                 "top_z_range_m: [1, 0]}\n"):
        with pytest.raises(sc.SceneError, match="work_area"):
            sc.load_scene(_scene_file(tmp_path, body))


# ---- point in polygon -----------------------------------------------------------------

def test_points_in_area_square_and_margin():
    area = _valid()
    pts = np.array([[0.4, -0.4], [0.9, -0.4], [-0.05, -0.4], [0.4, 0.04], [0.4, 0.2], [0.82, 0.02]])
    assert wa.points_in_area(pts, area, 0.0).tolist() == [True, False, False, False, False, False]
    assert wa.points_in_area(pts, area, 0.1).tolist() == [True, True, True, True, False, True]


def test_points_in_area_concave_polygon():
    ell = _valid(polygon_xy_m=[[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]])
    pts = np.array([[0.5, 1.5], [1.5, 1.5], [1.5, 0.5], [0.5, 0.5]])
    assert wa.points_in_area(pts, ell, 0.0).tolist() == [True, False, True, True]
    # the notch is within 0.6 of the boundary, but not within 0.3
    assert wa.points_in_area(np.array([[1.5, 1.5]]), ell, 0.3).tolist() == [False]
    assert wa.points_in_area(np.array([[1.5, 1.5]]), ell, 0.6).tolist() == [True]


def test_point_in_area_boundary_counts_and_nan_does_not():
    area = _valid()
    assert wa.point_in_area(0.4, 0.0, area) and wa.point_in_area(0.0, -0.8, area)
    assert not wa.point_in_area(0.4, 0.001, area)
    assert not wa.point_in_area(float("nan"), 0.0, area)


def test_brief_polygon_contains_todays_box():
    area = wa.validate_work_area({"frame": "base_link", "polygon_xy_m": BRIEF_POLYGON,
                                  "top_z_range_m": [-0.15, 0.30]})
    for x, y in [(0.30, -0.40), (0.55, -0.40), (0.30, -0.05), (0.55, -0.05)]:
        assert wa.point_in_area(x, y, area), (x, y)
    assert not wa.point_in_area(0.05, -0.3, area)         # behind the robot table front
    assert not wa.point_in_area(0.9, -0.3, area)


# ---- crop -----------------------------------------------------------------------------

def _rotated_c2b():
    """camera -> base: x_cam -> +y_base, y_cam -> -x_base, z_cam -> +z_base, plus a translation."""
    m = np.eye(4)
    m[:3, :3] = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    m[:3, 3] = [0.5, -0.4, 0.2]
    return m


def _cloud_in_base(base_pts, c2b):
    """camera-frame float32 points whose base_link position is ``base_pts``."""
    m = np.asarray(c2b)
    return ((np.asarray(base_pts) - m[:3, 3]) @ m[:3, :3]).astype(np.float32)   # inverse rotation


def test_crop_keeps_inside_all_heights_and_thins_outside():
    area = _valid(outside_keep_every=4)
    c2b = _rotated_c2b()
    rng = np.random.default_rng(1)
    inside = np.column_stack([rng.uniform(0.05, 0.75, 500), rng.uniform(-0.75, -0.05, 500),
                              rng.uniform(-1.0, 1.0, 500)])               # any height, floor included
    near = np.array([[0.85, -0.4, 0.0], [-0.05, -0.4, -0.9]])             # inside the 0.10 margin
    outside = np.column_stack([rng.uniform(1.2, 2.0, 400), rng.uniform(-0.7, -0.1, 400),
                               rng.uniform(-1.0, 0.0, 400)])
    base = np.vstack([inside, near, outside])
    cloud = _cloud_in_base(base, c2b)
    keep, stats = wa.crop_mask(cloud, c2b, area)
    assert keep[:502].all()                                               # inside + margin: every point
    assert int(keep[502:].sum()) == 100 and keep[502::4].all()           # every 4th outside point
    assert stats == {"points_total": 902, "points_inside": 502, "points_outside": 400,
                     "points_outside_kept": 100, "points_kept": 602, "points_floor_kept": 0,
                     "kept_fraction": round(602 / 902, 4), "outside_keep_every": 4, "margin_m": 0.1,
                     "floor_keep_below_z_m": None}


def test_crop_keeps_the_floor_dense_and_thins_the_rest():
    area = _valid(outside_keep_every=10, floor_keep_below_z_m=-0.75)
    c2b = _rotated_c2b()
    rng = np.random.default_rng(2)
    floor = np.column_stack([rng.uniform(1.0, 2.0, 600), rng.uniform(-1.0, 1.0, 600), rng.normal(-0.83, 0.004, 600)])
    wall = np.column_stack([rng.uniform(1.0, 2.0, 500), rng.uniform(-1.0, 1.0, 500), rng.uniform(-0.6, 0.5, 500)])
    inside_floor = np.array([[0.4, -0.4, -0.83]])                         # inside the polygon: kept anyway
    cloud = _cloud_in_base(np.vstack([floor, wall, inside_floor]), c2b)
    keep, stats = wa.crop_mask(cloud, c2b, area)
    assert keep[:600].all() and keep[-1]                                  # floor: every point, any xy
    assert int(keep[600:1100].sum()) == 50                                # other outside points: every 10th
    assert stats["points_floor_kept"] == 600 and stats["points_inside"] == 1
    assert stats["points_kept"] == 651 and stats["floor_keep_below_z_m"] == -0.75
    # the 'every' count restarts on the points that are neither inside nor floor
    keep0, _ = wa.crop_mask(cloud, c2b, _valid(outside_keep_every=0, floor_keep_below_z_m=-0.75))
    assert int(keep0.sum()) == 601


def test_crop_keep_every_zero_drops_all_outside_and_one_keeps_all():
    c2b = np.eye(4)
    base = np.array([[0.4, -0.4, 0.0], [5.0, 5.0, 0.0], [6.0, 5.0, 0.0], [-3.0, 0.0, 0.0]])
    keep, stats = wa.crop_mask(base.astype(np.float32), c2b, _valid(outside_keep_every=0))
    assert keep.tolist() == [True, False, False, False] and stats["points_outside_kept"] == 0
    keep, stats = wa.crop_mask(base.astype(np.float32), c2b, _valid(outside_keep_every=1))
    assert keep.all() and stats["points_outside_kept"] == 3


def test_crop_capture_writes_unorganized_pcd_and_matching_manifest(tmp_path):
    c2b = np.eye(4)
    base = np.array([[0.4, -0.4, 0.1], [0.5, -0.2, -0.5], [9.0, 0.0, 0.0], [9.5, 0.0, 0.0]], dtype=np.float32)
    manifest = {"schema": "rgbd_capture/v1", "capture_id": "c1",
                "image": {"path": "/x.png", "width": 4, "height": 4},
                "camera_info": {"k": [1] * 9},
                "pointcloud": {"path": "/orig.pcd", "width": 4, "height": 1, "point_count": 4,
                               "point_step": 16, "row_step": 64, "fields": []}}
    pcd, man, stats = wa.crop_capture(base, c2b, _valid(outside_keep_every=2), manifest, tmp_path, "c1")
    xyz, info = read_pcd_xyz(str(pcd))
    assert info["organized"] is False and info["points_total"] == 3 and xyz.shape == (3, 3)
    assert np.array_equal(xyz, base[[0, 1, 2]])                           # inside two + every 2nd outside
    doc = json.loads(Path(man).read_text())
    pc = doc["pointcloud"]
    assert pc["path"] == str(pcd) and pc["point_count"] == pc["width"] == 3 and pc["height"] == 1
    assert pc["row_step"] == 36 and pc["work_area_source"]["path"] == "/orig.pcd"
    assert pc["work_area_crop"]["points_kept"] == 3 == stats["points_kept"]
    assert doc["camera_info"] == manifest["camera_info"] and doc["image"] == manifest["image"]
    assert manifest["pointcloud"]["path"] == "/orig.pcd"                  # the input manifest is untouched


# ---- candidate filter ---------------------------------------------------------------------

W_IMG, H_IMG = 640, 480
CAM = {"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0, "width": W_IMG, "height": H_IMG}


def _down_looking_c2b():
    """Camera 1.5 m above the area looking straight down (optical axis = -z_base)."""
    m = np.eye(4)
    m[:3, :3] = np.diag([1.0, -1.0, -1.0])
    m[:3, 3] = [0.4, -0.4, 1.5]
    return m


def _runtime(area=None, c2b=None):
    return wa.runtime_config(area or _valid(), _down_looking_c2b() if c2b is None else c2b)


def test_prism_mask_covers_the_area_projection():
    mask, err = wa.prism_image_mask(_runtime(), CAM, W_IMG, H_IMG)
    assert err is None and mask.shape == (H_IMG, W_IMG)
    assert mask[240, 320] == 1                                   # area centre
    # area half size 0.4 m: the nearer top (z=0.3, 1.2 m) spans +-167 px, the bottom (z=-0.15, 1.65 m) +-121 px
    assert mask[240, 320 + 160] == 1 and mask[240, 320 + 175] == 0
    assert mask[5, 5] == 0 and mask[470, 630] == 0
    assert 0.20 < mask.mean() < 0.40


def test_candidate_filter_keeps_inside_drops_outside_and_partial():
    filt = wa.CandidateFilter(_runtime(), CAM, W_IMG, H_IMG)
    inside = [[250, 170], [390, 170], [390, 310], [250, 310]]
    keep, info = filt(inside)
    assert keep and info["applied"] is True and info["inside_frac"] == 1.0
    assert info["min_inside_frac"] == 0.8 and info["polygon_hash"] == wa.polygon_hash(_valid())
    keep, info = filt([[10, 10], [90, 10], [90, 80], [10, 80]])
    assert not keep and info["inside_frac"] == 0.0
    # half of the quad inside the prism silhouette (right edge at u ~ 468): below 0.8 -> dropped
    keep, info = filt([[420, 200], [520, 200], [520, 280], [420, 280]])
    assert not keep and 0.3 < info["inside_frac"] < 0.7
    # 90 % inside -> kept
    keep, info = filt([[408, 200], [452, 200], [452, 280], [408, 280]])
    assert keep and info["inside_frac"] >= 0.8


def test_candidate_filter_fails_open_when_the_prism_cannot_be_projected():
    behind = np.eye(4)                      # camera at x_base = 1 looking along +x_base: the area (x < 0.8) is behind it
    behind[:3, :3] = np.array([[0, 0, 1.0], [-1.0, 0, 0], [0, -1.0, 0]])
    behind[:3, 3] = [1.0, -0.4, 0.0]
    filt = wa.CandidateFilter(wa.runtime_config(_valid(), behind), CAM, W_IMG, H_IMG)
    keep, info = filt([[10, 10], [90, 10], [90, 80], [10, 80]])
    assert keep and info["applied"] is False and info["reason"] == "prism_behind_camera"
    keep, info = wa.CandidateFilter(_runtime(), CAM, W_IMG, H_IMG)([[float("nan"), 1], [2, 2], [3, 3], [4, 4]])
    assert keep and info["applied"] is False


class _Session:
    checkpoint_sha256 = "sha"
    model_revision = "rev"

    def __init__(self, resp):
        self.resp = resp

    def query(self, *a, **k):
        return self.resp


def _resp(primary, alternates):
    return {"ok": True, "corners_px": primary, "concept_score": 0.9, "prompt": "box", "box_xyxy": [0, 0, 1, 1],
            "n_detections": 1 + len(alternates), "detections": [], "picked": 0, "mask_quad_iou": 0.9,
            "alternates": [{"corners_px": a, "concept_score": 0.5 - 0.1 * i, "box_xyxy": [0, 0, 1, 1],
                            "mask_quad_iou": 0.9, "det_index": i + 1} for i, a in enumerate(alternates)]}


def test_gather_drops_outside_sam3_candidates_before_ranking():
    cfg = {"sam3": {"checkpoint_sha256": None}}
    outside = [[10, 10], [90, 10], [90, 80], [10, 80]]
    inside = [[250, 170], [390, 170], [390, 310], [250, 310]]
    filt = wa.CandidateFilter(_runtime(), CAM, W_IMG, H_IMG)
    cands = P._gather_coarse_candidates("c", "sam3_live", "img.png", _Session(_resp(outside, [inside, outside])),
                                        None, None, None, cfg, cand_filter=filt)
    assert [c["coarse"] is None for c in cands] == [True, False, True]
    assert cands[0]["reason"].startswith(wa.CANDIDATE_DROP_REASON + "(inside=0.00<0.80)")
    assert cands[0]["prov"]["work_area"]["applied"] is True
    assert cands[1]["reason"] is None and cands[1]["alternate_rank"] == 1
    assert np.allclose(cands[1]["coarse"], np.asarray(inside, dtype=float))
    # without a filter: nothing changes
    plain = P._gather_coarse_candidates("c", "sam3_live", "img.png", _Session(_resp(outside, [inside])),
                                        None, None, None, cfg)
    assert all(c["coarse"] is not None and "work_area" not in c["prov"] for c in plain)


def test_depth_only_and_skipped_candidates_are_not_filtered():
    filt = wa.CandidateFilter(_runtime(), CAM, W_IMG, H_IMG)
    outside = np.array([[10, 10], [90, 10], [90, 80], [10, 80]], dtype=float)
    cands = [{"source": "depth_only", "coarse": outside, "prov": {}, "reason": None},
             {"source": "sam3_archived", "coarse": None, "prov": {}, "reason": "x", "skipped": True},
             {"source": "sam3_archived", "coarse": outside, "prov": {}, "reason": None}]
    P._drop_candidates_outside_work_area(cands, filt)
    assert cands[0]["coarse"] is not None and "work_area" not in cands[0]["prov"]
    assert cands[1]["reason"] == "x"
    assert cands[2]["coarse"] is None and cands[2]["reason"].startswith(wa.CANDIDATE_DROP_REASON)


# ---- corner check ----------------------------------------------------------------------------

def test_corners_check_pass_and_failures():
    area = _valid()
    good = {"a": [0.1, -0.1, 0.0], "b": [0.7, -0.1, 0.1], "c": [0.7, -0.7, 0.2], "d": [0.1, -0.7, 0.3]}
    out = wa.corners_check(good, area)
    assert out["pass"] is True and out["xy_outside"] == [] and out["z_outside"] == []
    assert out["polygon_hash"] == wa.polygon_hash(area) and out["corners_base_link"]["a"] == [0.1, -0.1, 0.0]
    out = wa.corners_check({**good, "b": [0.81, -0.1, 0.1]}, area)
    assert out["pass"] is False and out["xy_outside"] == ["b"] and out["z_outside"] == []
    out = wa.corners_check({**good, "c": [0.7, -0.7, 0.31]}, area)
    assert out["pass"] is False and out["z_outside"] == ["c"] and out["xy_outside"] == []
    out = wa.corners_check({**good, "d": [0.1, -0.7, -0.151]}, area)
    assert out["z_outside"] == ["d"]
    # boundary values count as inside
    assert wa.corners_check({**good, "a": [0.0, 0.0, 0.3]}, area)["pass"] is True
    assert wa.corners_check({**good, "a": [0.1, -0.1, float("nan")]}, area)["z_outside"] == ["a"]
