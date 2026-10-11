"""live.side_seed (side_tape_seed/v1.1) and its step in live.orchestrator.

Synthetic boxes with a known answer (axis-aligned, yawed, tilted), outward
normals, height by class, validation failures (nothing written), in-place
update keeping every key, and the orchestrator's --autolabel-v2 step 6
(subprocess mocked like tests/test_autolabel_v2_live.py).
"""

import importlib
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

SEED = "box_seam_detection.live.side_seed"
ORCHESTRATOR = "box_seam_detection.live.orchestrator"
LIVE_TESTS = Path(__file__).resolve().parent / "test_autolabel_v2_live.py"


def _load(target, name=None):
    """A package module (dotted name) or, for the sibling test helper, a file path."""
    if isinstance(target, Path):
        spec = importlib.util.spec_from_file_location(name, str(target))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    return importlib.import_module(target)


@pytest.fixture()
def seed():
    return _load(SEED, "side_face_seed_under_test")


# ------------------------------------------------------------ synthetic boxes

def _rot(yaw_deg=0.0, roll_deg=0.0, pitch_deg=0.0):
    """R = Rz(yaw) @ Ry(pitch) @ Rx(roll) as row lists."""
    cy, sy = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    cp, sp = math.cos(math.radians(pitch_deg)), math.sin(math.radians(pitch_deg))
    cr, sr = math.cos(math.radians(roll_deg)), math.sin(math.radians(roll_deg))
    rz = [[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]
    ry = [[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]]
    rx = [[1, 0, 0], [0, cr, -sr], [0, sr, cr]]

    def mul(a, b):
        return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    return mul(rz, mul(ry, rx))


def _apply(r, v):
    return [sum(r[i][k] * v[k] for k in range(3)) for i in range(3)]


def _p(v):
    return {"x": v[0], "y": v[1], "z": v[2]}


def _v(d):
    return [d["x"], d["y"], d["z"]]


def _box(lr_len=0.112, fb_len=0.100, rot=None, centre=(0.3, -0.2, 0.09),
         seam_t=0.5, flip_normal=False):
    """Top face in a local frame: x back(+)/front(-), y left(+)/right(-), z up.

    Left edge (back_left-front_left) has length lr_len; seam_start/seam_end are
    at fraction seam_t from back along the left/right edges.
    """
    r = rot or _rot()
    hx, hy = lr_len / 2.0, fb_len / 2.0
    local = {"back_left": [hx, hy, 0], "front_left": [-hx, hy, 0],
             "back_right": [hx, -hy, 0], "front_right": [-hx, -hy, 0]}
    world = {k: [c + d for c, d in zip(centre, _apply(r, v))] for k, v in local.items()}

    def along(a, b, t):
        return [pa + t * (pb - pa) for pa, pb in zip(world[a], world[b])]
    snap = {
        "schema": "box_h_cut_snapshot/v1", "snapshot_id": "snap_1", "capture_id": "cap_1",
        "acquisition_stamp_ns": 1000, "frame_id": "base_link",
        "source_frame": "camera_color_optical_frame",
        "seam_start": _p(along("back_left", "front_left", seam_t)),
        "seam_end": _p(along("back_right", "front_right", seam_t)),
        "plane_rms_m": 0.0008,
        "surface_normal": _p([-c if flip_normal else c for c in _apply(r, [0, 0, 1])]),
        "custom_key": {"keep": [1, 2, 3]},
    }
    snap["seam_center"] = _p([(a + b) / 2 for a, b in zip(_v(snap["seam_start"]),
                                                          _v(snap["seam_end"]))])
    for k, v in world.items():
        snap[k] = _p(v)
    axes = {"left": _apply(r, [0, 1, 0]), "right": _apply(r, [0, -1, 0]),
            "back": _apply(r, [1, 0, 0]), "front": _apply(r, [-1, 0, 0]),
            "up": _apply(r, [0, 0, 1])}
    return snap, axes


def _close(a, b, tol=1e-9):
    return all(abs(x - y) <= tol for x, y in zip(a, b))


# ------------------------------------------------------------ geometry

@pytest.mark.parametrize("rot", [
    _rot(),
    _rot(yaw_deg=37.0),
    _rot(yaw_deg=-118.0, roll_deg=8.0, pitch_deg=-5.0),
])
def test_known_answer_faces_and_centerlines(seed, rot):
    snap, axes = _box(rot=rot)
    block = seed.compute_seed(snap)
    h = 0.0863
    # default synthetic top 112 x 100 vs small outer 112 x 106.25:
    # back/front offset 0, left/right (seam-end) offset 3.125 mm.
    offsets = {"left": 0.003125, "right": 0.003125, "back": 0.0, "front": 0.0}
    assert block["schema"] == "side_tape_seed/v1.1" and block["frame_id"] == "base_link"
    assert block["box_class"] == "small" and block["box_height_m"] == pytest.approx(h)
    up = _v(block["top_normal"])
    assert _close(up, axes["up"])
    for face in ("left", "right", "back", "front"):
        entry = block["faces"][face]
        n = _v(entry["plane"]["normal"])
        assert _close(n, axes[face]), face          # outward, unit, known answer
        assert abs(sum(a * b for a, b in zip(n, up))) < 1e-9
        a, b = entry["top_edge_corners"]
        assert _v(entry["top_edge"]["start"]) == _v(snap[a])
        assert _v(entry["top_edge"]["end"]) == _v(snap[b])
        off = offsets[face]
        assert entry["wall_offset_m"] == pytest.approx(off, abs=1e-12)
        for key in ("start", "end"):
            top = _v(entry["top_edge"][key])
            wall_top = [p + off * m for p, m in zip(top, n)]
            assert _close(_v(entry["wall_top_edge"][key]), wall_top)
            expected = [p - h * u for p, u in zip(wall_top, up)]
            assert _close(_v(entry["bottom_edge"][key]), expected)
            assert _close(_v(entry["top_edge_bottom_edge"][key]),
                          [p - h * u for p, u in zip(top, up)])
        # plane point is on the plane of its own top edge, bottom edge too.
        pt = _v(entry["plane"]["point"])
        for key in ("start", "end"):
            d = [q - p for q, p in zip(_v(entry["bottom_edge"][key]), pt)]
            assert abs(sum(x * y for x, y in zip(d, n))) < 1e-9
    assert block["faces"]["left"]["top_edge_corners"] == ["back_left", "front_left"]
    assert block["faces"]["right"]["top_edge_corners"] == ["back_right", "front_right"]
    for face, seam_name in (("left", "seam_start"), ("right", "seam_end")):
        tape = block["faces"][face]["tape_centerline"]
        n = _v(block["faces"][face]["plane"]["normal"])
        assert tape["seam_point"] == seam_name
        assert _close(_v(tape["top_edge_point"]), _v(snap[seam_name]))
        on_wall = [p + 0.003125 * m for p, m in zip(_v(snap[seam_name]), n)]
        assert _close(_v(tape["top"]), on_wall)
        assert _close(_v(tape["bottom"]), [p - h * u for p, u in zip(on_wall, up)])
        assert _close(_v(tape["direction"]), [-u for u in up])
        assert tape["seam_point_to_edge_m"] < 1e-9
    assert "tape_centerline" not in block["faces"]["back"]
    centre = _v(block["top_centre"])
    assert block["table_z_estimate"] == pytest.approx(centre[2] - h * up[2])
    assert block["diagnostics"]["left_right_normal_dot"] == pytest.approx(-1.0)
    notes = " ".join(block["notes"])
    assert "not measured" in notes.lower() and "inner fold" in notes.lower()
    assert "outer wall" in notes.lower() and "4.8 mm" in notes


def test_normal_pointing_down_is_flipped_up(seed):
    snap, axes = _box(rot=_rot(yaw_deg=20.0), flip_normal=True)
    block = seed.compute_seed(snap)
    assert block["top_normal_flipped_to_up"] is True
    assert _close(_v(block["top_normal"]), axes["up"])
    assert _close(_v(block["faces"]["left"]["plane"]["normal"]), axes["left"])


def test_seam_off_centre_projects_onto_edge(seed):
    snap, _ = _box(seam_t=0.3)
    snap["seam_start"]["z"] += 0.001            # 1 mm above the edge: within 2 mm
    block = seed.compute_seed(snap)
    tape = block["faces"]["left"]["tape_centerline"]
    assert tape["edge_param_t"] == pytest.approx(0.3)
    assert tape["seam_point_to_edge_m"] == pytest.approx(0.001)
    assert tape["top"]["z"] == pytest.approx(snap["back_left"]["z"])
    assert tape["seam_point_raw"] == snap["seam_start"]


@pytest.mark.parametrize("lr, fb, cls, h, outer", [
    (0.112, 0.100, "small", 0.0863, {}),
    (0.193, 0.105, "big", 0.13425, {}),
    (0.105, 0.193, "big", 0.13425, {}),       # long side on the front/back edges
    # just under the 150 mm threshold (outer dims overridden so the wall offset is valid)
    (0.1499, 0.100, "small", 0.0863, {"outer_across_seam_mm": 152.0,
                                      "outer_along_seam_mm": 102.0}),
])
def test_height_by_class(seed, lr, fb, cls, h, outer):
    snap, _ = _box(lr_len=lr, fb_len=fb)
    block = seed.compute_seed(snap, **outer)
    assert block["box_class"] == cls and block["box_height_m"] == pytest.approx(h)
    assert block["box_class_source"] == "auto"
    assert block["box_height_source"] == f"class_default:{cls}"


def test_class_and_height_overrides(seed):
    snap, _ = _box()
    block = seed.compute_seed(snap, box_class="big")
    assert block["box_class"] == "big" and block["box_height_m"] == pytest.approx(0.13425)
    assert block["box_class_source"] == "override"
    block = seed.compute_seed(snap, box_height_mm=90.0)
    assert block["box_class"] == "small" and block["box_height_m"] == pytest.approx(0.090)
    assert block["box_height_source"] == "override"
    with pytest.raises(seed.SeedError):
        seed.compute_seed(snap, box_height_mm=-1.0)


# ------------------------------------------------------------ outer wall (v1.1)

def _dist_to_plane(point, plane):
    n, q = _v(plane["normal"]), _v(plane["point"])
    return sum(m * (a - b) for m, a, b in zip(n, point, q))


@pytest.mark.parametrize("rot", [_rot(), _rot(yaw_deg=-71.0, roll_deg=5.0, pitch_deg=3.0)])
def test_outer_wall_known_answer_small_box(seed, rot):
    # Camera flat top 110 (seam-crossed edges) x 96.7 (along the seam); outer
    # 112.0 x 106.25 -> back/front walls 1.0 mm out, seam-end walls 4.775 mm out.
    snap, axes = _box(lr_len=0.110, fb_len=0.0967, rot=rot)
    block = seed.compute_seed(snap)
    expected = {"left": 0.004775, "right": 0.004775, "back": 0.0010, "front": 0.0010}
    assert block["wall_offsets_m"]["along_seam"] == pytest.approx(0.004775, abs=1e-9)
    assert block["wall_offsets_m"]["across_seam"] == pytest.approx(0.0010, abs=1e-9)
    assert block["measured_top_dims_m"]["along_seam"] == pytest.approx(0.0967)
    assert block["measured_top_dims_m"]["across_seam"] == pytest.approx(0.110)
    assert block["outer_dims_m"] == {"along_seam": 0.10625, "across_seam": 0.112}
    assert block["flags"] == []
    for face, off in expected.items():
        entry = block["faces"][face]
        assert entry["wall_offset_m"] == pytest.approx(off, abs=1e-9)
        assert entry["wall_source"] == "user_outer_dims"
        assert entry["outer_dim_m"] == (0.10625 if face in ("left", "right") else 0.112)
        n = _v(entry["plane"]["normal"])
        assert _close(n, axes[face])
        assert entry["top_edge_plane"]["normal"] == entry["plane"]["normal"]
        shift = [a - b for a, b in zip(_v(entry["plane"]["point"]),
                                       _v(entry["top_edge_plane"]["point"]))]
        assert _close(shift, [off * m for m in n])        # outward along the normal
        # the top-edge corners are `off` inside the outer wall plane
        for name in entry["top_edge_corners"]:
            assert _dist_to_plane(_v(snap[name]), entry["plane"]) == pytest.approx(-off)
            assert _dist_to_plane(_v(snap[name]), entry["top_edge_plane"]) == \
                pytest.approx(0.0, abs=1e-12)
    # outer walls are 106.25 / 112.0 mm apart
    left, right = block["faces"]["left"]["plane"], block["faces"]["right"]["plane"]
    assert -_dist_to_plane(_v(right["point"]), left) == pytest.approx(0.10625)
    back, front = block["faces"]["back"]["plane"], block["faces"]["front"]["plane"]
    assert -_dist_to_plane(_v(front["point"]), back) == pytest.approx(0.112)
    for face in ("left", "right"):
        tape = block["faces"][face]["tape_centerline"]
        assert _dist_to_plane(_v(tape["top"]), block["faces"][face]["plane"]) == \
            pytest.approx(0.0, abs=1e-12)
        assert _dist_to_plane(_v(tape["bottom"]), block["faces"][face]["plane"]) == \
            pytest.approx(0.0, abs=1e-12)
        gap = math.dist(_v(tape["top"]), _v(tape["top_edge_point"]))
        assert gap == pytest.approx(0.004775)
        assert tape["wall_offset_m"] == pytest.approx(0.004775)


def test_outer_wall_small_negative_offset_clamped(seed):
    # Top 113 mm > 112 mm outer: -0.5 mm is within the -1 mm tolerance -> clamped to 0, flagged.
    snap, _ = _box(lr_len=0.113, fb_len=0.0967)
    out = seed.compute_seed(snap)
    assert out["wall_offsets_m"]["across_seam"] == 0.0
    assert "wall_offset_clamped_negative" in out["flags"]


def test_outer_wall_negative_offset_fails_and_writes_nothing(seed, tmp_path, capsys):
    # Top 115 mm along the seam-crossed edges > 112 mm outer (-1.5 mm): wrong box/class.
    snap, _ = _box(lr_len=0.115, fb_len=0.0967)
    with pytest.raises(seed.SeedError, match=r"across_seam.*-1\.50 mm.*BIGGER"):
        seed.compute_seed(snap)
    path = tmp_path / "planner.json"
    text = json.dumps(snap)
    path.write_text(text, encoding="utf-8")
    assert seed.main(["--planner-json", str(path), "--out-json",
                      str(tmp_path / "seed.json")]) == 1
    assert path.read_text(encoding="utf-8") == text
    assert sorted(p.name for p in tmp_path.iterdir()) == ["planner.json"]
    assert "outer-wall offset" in capsys.readouterr().err


def test_outer_wall_offset_above_8mm_fails(seed):
    snap, _ = _box(lr_len=0.110, fb_len=0.088)      # (106.25 - 88)/2 = 9.1 mm
    with pytest.raises(seed.SeedError, match=r"along_seam.*9\.1[23] mm"):
        seed.compute_seed(snap)
    snap, _ = _box(lr_len=0.112, fb_len=0.09025)    # exactly 8.0 mm is accepted
    assert seed.compute_seed(snap)["wall_offsets_m"]["along_seam"] == \
        pytest.approx(0.008, abs=1e-9)


def test_big_box_outer_wall_unknown_flag_no_shift(seed):
    snap, _ = _box(lr_len=0.193, fb_len=0.105, rot=_rot(yaw_deg=33.0))
    block = seed.compute_seed(snap)
    assert block["box_class"] == "big"
    assert block["flags"] == ["outer_wall_unknown"]
    assert block["outer_dims_m"] == {"along_seam": None, "across_seam": None}
    for face, entry in block["faces"].items():
        assert entry["wall_source"] == "outer_wall_unknown", face
        assert entry["wall_offset_m"] == 0.0 and entry["outer_dim_m"] is None
        assert entry["plane"] == entry["top_edge_plane"]
        assert entry["bottom_edge"] == entry["top_edge_bottom_edge"]
    for face in ("left", "right"):
        tape = block["faces"][face]["tape_centerline"]
        assert tape["top"] == tape["top_edge_point"]


def test_outer_dim_overrides(seed, tmp_path):
    snap, _ = _box(lr_len=0.110, fb_len=0.0967)
    block = seed.compute_seed(snap, outer_along_seam_mm=100.7, outer_across_seam_mm=114.0)
    assert block["wall_offsets_m"]["along_seam"] == pytest.approx(0.002, abs=1e-9)
    assert block["wall_offsets_m"]["across_seam"] == pytest.approx(0.002, abs=1e-9)
    assert block["outer_dims_source"] == {"along_seam": "override", "across_seam": "override"}
    # override one dimension only on a big box: that pair shifted, the other flagged
    big, _ = _box(lr_len=0.193, fb_len=0.105)
    block = seed.compute_seed(big, outer_along_seam_mm=109.0)
    assert block["faces"]["left"]["wall_offset_m"] == pytest.approx(0.002, abs=1e-9)
    assert block["faces"]["left"]["wall_source"] == "user_outer_dims"
    assert block["faces"]["back"]["wall_source"] == "outer_wall_unknown"
    assert block["flags"] == ["outer_wall_unknown"]
    with pytest.raises(seed.SeedError, match="must be positive"):
        seed.compute_seed(snap, outer_along_seam_mm=0.0)
    # CLI flags
    path = tmp_path / "planner.json"
    path.write_text(json.dumps(snap), encoding="utf-8")
    assert seed.main(["--planner-json", str(path), "--outer-along-seam-mm", "100.7",
                      "--outer-across-seam-mm", "114"]) == 0
    written = json.loads(path.read_text(encoding="utf-8"))["side_tape_seed"]
    assert written["wall_offsets_m"]["along_seam"] == pytest.approx(0.002, abs=1e-9)


def test_backward_compat_keys_present(seed):
    block = seed.compute_seed(_box(lr_len=0.110, fb_len=0.0967)[0])
    for face, entry in block["faces"].items():
        for key in ("top_edge_corners", "plane", "top_edge", "bottom_edge",
                    "top_edge_length_m", "top_edge_plane", "wall_top_edge",
                    "top_edge_bottom_edge", "wall_offset_m", "outer_dim_m",
                    "measured_top_dim_m", "wall_source"):
            assert key in entry, (face, key)
        assert set(entry["plane"]) == {"point", "normal"}
    for face in ("left", "right"):
        tape = block["faces"][face]["tape_centerline"]
        for key in ("top", "bottom", "direction", "length_m", "seam_point",
                    "seam_point_raw", "seam_point_to_edge_m", "edge_param_t",
                    "top_edge_point", "top_edge_point_bottom"):
            assert key in tape, (face, key)
    for key in ("box_class", "box_height_m", "top_face_size_m", "top_normal",
                "top_centre", "table_z_estimate", "diagnostics", "tolerances", "notes",
                "inner_fold_outer_offset_m"):
        assert key in block, key


def test_real_snapshot_rerun_is_idempotent(seed, tmp_path):
    real = Path("/tmp/claude-1000/live2/planner_3.json")
    if not real.is_file():
        pytest.skip("live2 planner snapshot not on this machine")
    path = tmp_path / "planner_3.json"
    path.write_bytes(real.read_bytes())       # may already hold a v1 side_tape_seed
    assert seed.main(["--planner-json", str(path)]) == 0
    first = path.read_text(encoding="utf-8")
    block = json.loads(first)["side_tape_seed"]
    assert block["schema"] == "side_tape_seed/v1.1" and block["box_class"] == "small"
    for key in ("along_seam", "across_seam"):
        assert 0.0 <= block["wall_offsets_m"][key] <= 0.008
    assert seed.main(["--planner-json", str(path)]) == 0
    assert path.read_text(encoding="utf-8") == first


# ------------------------------------------------------------ validation

def _mutate_seam_off_edge(snap):
    snap["seam_start"]["x"] += 0.01
    snap["seam_start"]["y"] += 0.01


def _reseam(snap):
    snap["seam_start"] = _p([(a + b) / 2 for a, b in zip(_v(snap["back_left"]),
                                                         _v(snap["front_left"]))])
    snap["seam_end"] = _p([(a + b) / 2 for a, b in zip(_v(snap["back_right"]),
                                                       _v(snap["front_right"]))])


def _mutate_right_edge_skew(snap):
    # Right edge rotated ~28 deg in the top plane: corners far from 90 deg.
    snap["front_right"]["y"] += 0.06
    _reseam(snap)


def _mutate_parallelogram_shear(snap):
    # Front corners sheared 20 deg: left||right and back||front (both dots -1)
    # but the corner angles are 70/110 deg -> only the rectangularity check fires.
    shift = 0.112 * math.tan(math.radians(20.0))
    snap["front_left"]["y"] += shift
    snap["front_right"]["y"] += shift
    _reseam(snap)


def _mutate_left_right_converge(snap):
    # Left and right edges each tilted 9.5 deg the other way (angles 80.5/99.5,
    # inside [75, 105]) -> left/right normals 19 deg from opposite (dot -0.9455).
    e = 0.056 * math.tan(math.radians(9.5))
    snap["back_left"]["y"] -= e
    snap["front_left"]["y"] += e
    snap["back_right"]["y"] += e
    snap["front_right"]["y"] -= e
    _reseam(snap)


def _mutate_front_back_converge(snap):
    # Back and front edges each tilted 9.5 deg the other way -> front/back
    # normals 19 deg from opposite; left/right stay exactly opposite.
    d = 0.100 * math.tan(math.radians(9.5))
    snap["back_right"]["x"] -= d
    snap["front_right"]["x"] += d
    _reseam(snap)


def _mutate_seam_beyond_edge_end(snap):
    # seam_start 3 mm past front_left along the left edge (t ~ 1.027).
    a, b = _v(snap["back_left"]), _v(snap["front_left"])
    t = 1.0 + 0.003 / 0.112
    snap["seam_start"] = _p([pa + t * (pb - pa) for pa, pb in zip(a, b)])


def _mutate_missing_corner(snap):
    del snap["front_left"]


def _mutate_frame(snap):
    snap["frame_id"] = "camera_color_optical_frame"


def _mutate_schema(snap):
    snap["schema"] = "other/v1"


def _mutate_nan(snap):
    snap["surface_normal"]["x"] = float("nan")


def _mutate_box_on_side(snap):
    snap["surface_normal"] = {"x": 1.0, "y": 0.0, "z": 0.0}


def _mutate_degenerate_edge(snap):
    snap["front_left"] = dict(snap["back_left"])


@pytest.mark.parametrize("mutate, message", [
    (_mutate_seam_off_edge, "seam_start is"),
    (_mutate_right_edge_skew, "not rectangular"),
    (_mutate_parallelogram_shear, "not rectangular"),
    (_mutate_left_right_converge, "left/right face normals not opposite"),
    (_mutate_front_back_converge, "front/back face normals not opposite"),
    (_mutate_seam_beyond_edge_end, "projects outside the left top edge"),
    (_mutate_missing_corner, "missing front_left"),
    (_mutate_frame, "base_link"),
    (_mutate_schema, "box_h_cut_snapshot/v1"),
    (_mutate_nan, "not a finite"),
    (_mutate_box_on_side, "not upright"),
    (_mutate_degenerate_edge, "degenerate"),
])
def test_validation_failures_exit_1_and_write_nothing(seed, tmp_path, mutate, message, capsys):
    snap, _ = _box()
    mutate(snap)
    path = tmp_path / "planner.json"
    text = json.dumps(snap, indent=2)
    path.write_text(text, encoding="utf-8")
    out = tmp_path / "seed.json"
    with pytest.raises(seed.SeedError, match=message):
        seed.compute_seed(json.loads(text))
    assert seed.main(["--planner-json", str(path), "--out-json", str(out)]) == 1
    assert path.read_text(encoding="utf-8") == text
    assert not out.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["planner.json"]  # no temp left
    assert "planner snapshot left unchanged" in capsys.readouterr().err


def test_face_normal_orthogonality_is_diagnostic_only(seed):
    # The face normal is built as top_normal x edge, so it is 90 deg to the top
    # normal by construction: reported, never used as a validation.
    block = seed.compute_seed(_box(rot=_rot(yaw_deg=25.0, roll_deg=4.0))[0])
    for face, angle in block["diagnostics"]["face_normal_vs_top_deg"].items():
        assert angle == pytest.approx(90.0), face
    assert "face_ortho_deg" not in block["tolerances"]
    assert not hasattr(seed, "FACE_ORTHO_TOL_DEG")


def test_meaningful_checks_reported(seed):
    block = seed.compute_seed(_box(rot=_rot(yaw_deg=-40.0))[0])
    diag, tol = block["diagnostics"], block["tolerances"]
    assert diag["front_back_normal_dot"] == pytest.approx(-1.0)
    assert set(diag["corner_angles_deg"]) == {"back_left", "back_right",
                                              "front_right", "front_left"}
    assert all(a == pytest.approx(90.0) for a in diag["corner_angles_deg"].values())
    assert tol["front_back_dot_max"] == -0.95
    assert tol["corner_angle_range_deg"] == [75.0, 105.0]
    assert tol["seam_edge_param_margin_m"] == 0.002


def test_corner_angles_inside_range_pass(seed):
    # Mild 10 deg shear (angles 80/100) is still a valid box model.
    snap, _ = _box()
    shift = 0.112 * math.tan(math.radians(10.0))
    snap["front_left"]["y"] += shift
    snap["front_right"]["y"] += shift
    _reseam(snap)
    # sheared left/right edges are 113.7 mm: widen the outer size to keep the offset valid
    angles = seed.compute_seed(snap, outer_across_seam_mm=116.0)["diagnostics"][
        "corner_angles_deg"]
    assert min(angles.values()) == pytest.approx(80.0, abs=1e-6)
    assert max(angles.values()) == pytest.approx(100.0, abs=1e-6)


def test_seam_edge_param_uses_unclamped_t_with_2mm_margin(seed):
    snap, _ = _box()
    a, b = _v(snap["back_left"]), _v(snap["front_left"])
    t = 1.0 + 0.0015 / 0.112                      # 1.5 mm past the corner: inside margin
    snap["seam_start"] = _p([pa + t * (pb - pa) for pa, pb in zip(a, b)])
    tape = seed.compute_seed(snap)["faces"]["left"]["tape_centerline"]
    assert tape["edge_param_t"] == pytest.approx(t)          # unclamped
    assert tape["seam_point_to_edge_m"] == pytest.approx(0.0015)
    t = -0.0025 / 0.112                           # 2.5 mm before back_left
    snap["seam_start"] = _p([pa + t * (pb - pa) for pa, pb in zip(a, b)])
    with pytest.raises(seed.SeedError, match=r"t=-0\.02"):
        seed.compute_seed(snap)


def test_atomic_write_removes_temp_on_serialisation_error(seed, tmp_path):
    target = tmp_path / "out.json"
    target.write_text("old", encoding="utf-8")
    with pytest.raises(TypeError):
        seed.atomic_write_json(target, {"bad": object()})
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out.json"]
    assert target.read_text(encoding="utf-8") == "old"


def test_real_review_snapshot_passes_new_checks(seed, tmp_path):
    real = Path("/tmp/claude-1000/review12/planner_copy.json")
    if not real.is_file():
        pytest.skip("review12 planner copy not on this machine")
    snap = json.loads(real.read_text(encoding="utf-8"))
    snap.pop("side_tape_seed", None)
    block = seed.compute_seed(snap)
    angles = block["diagnostics"]["corner_angles_deg"]
    assert all(85.0 < a < 95.0 for a in angles.values()), angles
    assert block["diagnostics"]["front_back_normal_dot"] < -0.99
    for face in ("left", "right"):
        assert 0.0 <= block["faces"][face]["tape_centerline"]["edge_param_t"] <= 1.0


def test_unreadable_snapshot_exit_1(seed, tmp_path):
    path = tmp_path / "planner.json"
    path.write_text("{not json", encoding="utf-8")
    assert seed.main(["--planner-json", str(path)]) == 1
    assert path.read_text(encoding="utf-8") == "{not json"
    assert seed.main(["--planner-json", str(tmp_path / "absent.json")]) == 1


# ------------------------------------------------------------ in-place update

def test_in_place_update_keeps_all_keys_and_standalone_copy(seed, tmp_path):
    snap, _ = _box(rot=_rot(yaw_deg=63.0, roll_deg=3.0))
    path = tmp_path / "planner.json"
    path.write_text(json.dumps(snap), encoding="utf-8")
    out = tmp_path / "sub" / "seed.json"
    assert seed.main(["--planner-json", str(path), "--out-json", str(out),
                      "--box-class", "big"]) == 0
    updated = json.loads(path.read_text(encoding="utf-8"))
    assert {k: v for k, v in updated.items() if k != "side_tape_seed"} == snap
    assert updated["side_tape_seed"] == json.loads(out.read_text(encoding="utf-8"))
    assert updated["side_tape_seed"]["box_class"] == "big"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["planner.json", "sub"]
    # Re-running replaces the block (idempotent), still keeps everything else.
    assert seed.main(["--planner-json", str(path)]) == 0
    again = json.loads(path.read_text(encoding="utf-8"))
    assert again["side_tape_seed"]["box_class"] == "small"
    assert {k: v for k, v in again.items() if k != "side_tape_seed"} == snap


def test_cli_subprocess_runs():
    import os
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(Path(__file__).resolve().parents[2]), *[p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]])
    proc = subprocess.run([sys.executable, "-m", SEED, "--help"], stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, timeout=60, env=env)
    assert proc.returncode == 0 and "--box-class" in proc.stdout


# ------------------------------------------------------------ orchestrator

class _Probe:
    returncode = 0
    stdout = ""


CALIB_TEXT = """parameters:
  name: test_handeye
  calibration_type: eye_on_base
  robot_base_frame: base_link
  robot_effector_frame: tool
  tracking_base_frame: camera_link
  tracking_marker_frame: marker
transform:
  translation: {x: 0.0, y: -0.3, z: 0.45}
  rotation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}
"""


def _orchestrate(tmp_path, monkeypatch, seed_behaviour, *extra_argv):
    """Drive capture_and_query_depth.main() through steps 2-6 with fakes.

    seed_behaviour: "real" runs side_face_seed.py for real, an int fakes that
    exit code. Returns (exit code or None, seed calls, planner path).
    """
    live = _load(LIVE_TESTS, "_autolabel_v2_live_helpers")
    det = _load("box_seam_detection.live.detector")
    cap = live._capture(tmp_path)
    mod = _load(ORCHESTRATOR, "capture_and_query_depth")
    script = tmp_path / "autolabel_v2_detect.py"
    script.write_text("# fake (subprocess is stubbed)\n", encoding="utf-8")
    real_run = subprocess.run
    seed_calls = []
    state = {}

    def fake_subprocess_run(cmd, *a, **k):
        if len(cmd) > 2 and str(cmd[2]).endswith("live.side_seed"):
            seed_calls.append(list(cmd))
            if seed_behaviour == "real":
                return real_run(cmd, *a, **k)
            if seed_behaviour == "oserror":
                raise OSError("exec format error")

            class _R:
                returncode = seed_behaviour
            return _R()
        return _Probe()

    monkeypatch.setattr(mod.subprocess, "run", fake_subprocess_run)

    def fake_run_step(title, cmd, capture=False, check=True):
        if "autolabel_v2" in title:
            out = Path(cmd[cmd.index("--out-seam-json") + 1])
            rec = live._record(axis="bf")
            seam = det.published_seam_line(rec)
            rel = det.contract_relabel(rec["top_face"]["corners_px"],
                                       seam["seam_left"], seam["seam_right"])
            doc = det.build_canonical_seam_doc(
                rec, str(cap["image"]), live.W, live.H, "cap_1", seam, rel, "rec.json", {},
                False, cut_permission={"status": "allowed", "reasons": []})
            out.write_text(json.dumps(doc), encoding="utf-8")
        elif "query_depth" in title:
            Path(cmd[cmd.index("--output-json") + 1]).write_text("{}", encoding="utf-8")
        elif "base_link" in title:
            planner = Path(cmd[cmd.index("--planner-output-json") + 1])
            snap, _ = _box(rot=_rot(yaw_deg=-62.0, roll_deg=1.4))
            planner.parent.mkdir(parents=True, exist_ok=True)
            planner.write_text(json.dumps(snap), encoding="utf-8")
            state["planner"] = planner
        else:
            raise AssertionError(f"unexpected step {title}")
        return None

    monkeypatch.setattr(mod, "run_step", fake_run_step)
    calib = tmp_path / "test.calib"
    calib.write_text(CALIB_TEXT, encoding="utf-8")
    code = mod.main([
        "--autolabel-v2",
        "--autolabel-v2-script", str(script),
        "--autolabel-v2-sam3-python", str(cap["sam3_python"]),
        "--from-image", str(cap["image"]), "--pcd", str(cap["pcd"]),
        "--capture-manifest", str(cap["manifest"]),
        "--calib-file", str(calib), "--downstream-inprocess", "off",  # these fakes mock subprocess
        "--output-dir", str(tmp_path / "root"), *extra_argv,
    ])
    return code, seed_calls, state.get("planner")


def _rejected(planner):
    return planner.with_name(planner.name + ".side_seed_rejected")


def test_orchestrator_autolabel_v2_runs_seed_after_step5(tmp_path, monkeypatch):
    code, calls, planner = _orchestrate(tmp_path, monkeypatch, "real")
    assert code == 0
    assert len(calls) == 1
    assert calls[0][:3] == [sys.executable, "-m", SEED]
    assert calls[0][calls[0].index("--planner-json") + 1] == str(planner)
    snap = json.loads(planner.read_text(encoding="utf-8"))
    assert snap["side_cutting_points"]["schema"] == "side_cutting_midpoints/v1"  # step 5 kept
    assert "back_middle_left" in snap and snap["custom_key"] == {"keep": [1, 2, 3]}
    assert snap["detector_method"] == "autolabel_v2_sam3_fold_midline"
    assert snap["side_tape_seed"]["schema"] == "side_tape_seed/v1.1"
    assert snap["side_tape_seed"]["box_class"] == "small"
    assert not _rejected(planner).exists()


def test_orchestrator_clears_stale_rejected_file(tmp_path, monkeypatch):
    planner_path = tmp_path / "fixed" / "planner.json"
    old = _rejected(planner_path)
    old.parent.mkdir(parents=True)
    old.write_text('{"stale": true}', encoding="utf-8")
    code, _, staged = _orchestrate(tmp_path, monkeypatch, "real",
                                   "--planner-output-json", str(planner_path))
    # the snapshot is built in the pipeline dir (staged) and published atomically to the explicit path
    assert code == 0 and staged != planner_path
    published = json.loads(planner_path.read_text(encoding="utf-8"))
    assert "side_tape_seed" in published and published == json.loads(staged.read_text(encoding="utf-8"))
    assert "cut_permission" in published and published["cut_permission"]["status"] == "allowed"
    assert not old.exists()


@pytest.mark.parametrize("behaviour, expected_code", [(1, 1), (2, 2), ("oserror", 2)])
def test_orchestrator_seed_failure_fails_closed(tmp_path, monkeypatch, capsys,
                                                behaviour, expected_code):
    code, calls, planner = _orchestrate(tmp_path, monkeypatch, behaviour)
    assert code == expected_code and len(calls) == 1
    assert not planner.exists()                         # nothing valid-looking left
    rejected = _rejected(planner)
    snap = json.loads(rejected.read_text(encoding="utf-8"))
    assert snap["schema"] == "box_h_cut_snapshot/v1"     # step-5 content, kept aside
    assert "side_tape_seed" not in snap
    assert snap["side_cutting_points"]["schema"] == "side_cutting_midpoints/v1"
    assert snap["detector_method"] == "autolabel_v2_sam3_fold_midline"
    err = capsys.readouterr().err
    assert "side_seed_rejected" in err and "不可用於切割" in err


def test_quarantine_replaces_older_rejected_file(tmp_path):
    mod = _load(ORCHESTRATOR, "capture_and_query_depth")
    planner = tmp_path / "planner.json"
    planner.write_text('{"schema": "box_h_cut_snapshot/v1", "new": 1}', encoding="utf-8")
    rejected = _rejected(planner)
    rejected.write_text('{"older": true}', encoding="utf-8")
    assert mod._quarantine_planner_snapshot(planner) == rejected
    assert not planner.exists()
    assert json.loads(rejected.read_text(encoding="utf-8"))["new"] == 1
    assert sorted(p.name for p in tmp_path.iterdir()) == [rejected.name]


def test_quarantine_falls_back_to_delete_when_rename_fails(tmp_path, monkeypatch):
    mod = _load(ORCHESTRATOR, "capture_and_query_depth")
    planner = tmp_path / "planner.json"
    planner.write_text("{}", encoding="utf-8")

    def broken_replace(*_a, **_k):
        raise PermissionError("read-only")

    monkeypatch.setattr(mod.os, "replace", broken_replace)
    assert mod._quarantine_planner_snapshot(planner) is None
    assert not planner.exists()


def test_detector_method_constant():
    mod = _load(ORCHESTRATOR)
    assert mod.DETECTOR_METHOD == "autolabel_v2_sam3_fold_midline"


def test_orchestrator_no_side_seed_opt_out(tmp_path, monkeypatch):
    code, calls, planner = _orchestrate(tmp_path, monkeypatch, 1, "--no-side-seed")
    assert code == 0 and calls == []
    assert "side_tape_seed" not in json.loads(planner.read_text(encoding="utf-8"))
