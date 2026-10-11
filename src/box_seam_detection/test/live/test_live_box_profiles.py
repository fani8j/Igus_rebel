"""M1 live side: autolabel_v2_detect.py and side_face_seed.py read the shared box
profiles (scripts/autolabel_v2/box_profiles.py). Today's two boxes keep their
constants bit-identically; ``generic`` skips the depth-edge fix and R10, and the
side seed uses the measured height with outer_wall_unknown."""

import importlib.util
import json
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent


def _helper(name):
    spec = importlib.util.spec_from_file_location(f"{name}_mbc", str(TESTS / f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


live = _helper("test_autolabel_v2_live")
sf = _helper("test_side_face_seed")


@pytest.fixture()
def det():
    return live._load(live.DETECT, "autolabel_v2_detect_profiles")


@pytest.fixture()
def seed():
    return sf._load(sf.SEED, "side_face_seed_profiles")


# ------------------------------------------------------------ constants identical

def test_detector_constants_equal_the_old_literals(det):
    assert det.DEPTH_EDGE_SMALL_MAX_LONG_M == 0.150
    assert det.DEPTH_EDGE_BOX_CLASSES == {"small": (110.0, 98.0), "big": (193.0, 105.0)}
    assert det.DEPTH_EDGE_CLASS_TOL_MM == 8.0


def test_seed_constants_equal_the_old_literals(seed):
    assert seed.BOX_HEIGHTS_M["small"] == 0.0863 and seed.BOX_HEIGHTS_M["big"] == 0.13425
    assert "generic" not in seed.BOX_HEIGHTS_M
    assert seed.BIG_LONG_SIDE_M == 0.150
    assert seed.OUTER_DIMS_M["small"] == {"along_seam": 0.10625, "across_seam": 0.1120}
    assert seed.OUTER_DIMS_M["big"] == {"along_seam": None, "across_seam": None}
    assert seed.OUTER_DIMS_M["generic"] == {"along_seam": None, "across_seam": None}


@pytest.mark.parametrize("long_side", [0.05, 0.11, 0.1499999, 0.150])
def test_classify_box_auto_small_up_to_150mm(seed, long_side):
    assert seed.classify_box(long_side) == "small"


@pytest.mark.parametrize("long_side", [0.1500001, 0.193, 0.39])
def test_classify_box_auto_big_above_150mm(seed, long_side):
    assert seed.classify_box(long_side) == "big"


def test_class_match_uses_the_profile_table(det):
    import numpy as np
    def quad(long_mm, short_mm):
        l, s = long_mm / 2000.0, short_mm / 2000.0
        return {"back_left": np.array([-l, -s, 0.5]), "back_right": np.array([l, -s, 0.5]),
                "front_right": np.array([l, s, 0.5]), "front_left": np.array([-l, s, 0.5])}
    assert det._class_match(quad(110.0, 98.0)) == "small"
    assert det._class_match(quad(117.9, 105.9)) == "small"
    assert det._class_match(quad(193.0, 105.0)) == "big"
    assert det._class_match(quad(120.0, 98.0)) is None
    assert det._class_match(quad(390.0, 320.0)) is None


# ------------------------------------------------------------ profile resolution

def test_record_box_profile_is_preferred(det):
    rec = live._record()
    rec["box_profile"] = {"name": "generic", "source": "classify"}
    profile, info = det.resolve_box_profile(rec)
    assert profile.name == "generic" and info["source"] == "record"
    rec["box_profile"] = "small"
    assert det.resolve_box_profile(rec)[0].name == "small"


def test_fallback_classifies_from_corners_with_the_legacy_rule(det):
    rec = live._record()                        # 200 x 150 mm top -> long side > 150 mm
    profile, info = det.resolve_box_profile(rec)
    assert profile.name == "big" and info["source"] == "corners_legacy"
    rec["top_face"]["corners_xyz_camera_m"] = {
        "back_left": [-0.055, -0.049, 0.5], "back_right": [0.055, -0.049, 0.5],
        "front_right": [0.055, 0.049, 0.5], "front_left": [-0.055, 0.049, 0.5]}
    assert det.resolve_box_profile(rec)[0].name == "small"


def test_unknown_record_profile_name_is_ignored_and_reported(det):
    rec = live._record()
    rec["box_profile"] = {"name": "huge"}
    profile, info = det.resolve_box_profile(rec)
    assert profile.name == "big" and info["ignored_record_profile"] == "huge"


def test_no_corners_and_no_profile_is_generic(det, monkeypatch):
    rec = live._record()
    rec["top_face"]["corners_xyz_camera_m"] = None
    profile, info = det.resolve_box_profile(rec)
    assert profile.is_generic and "no corners_xyz" in info["source"]


def test_r10_only_runs_for_a_validated_profile(det, tmp_path, monkeypatch):
    for name in ("big", "generic"):
        cap = live._capture(tmp_path / name)
        rec = live._record(status="ambiguous")
        rec["box_profile"] = {"name": name, "measured_height_mm": 100.0, "flags": []}
        live._install_fake_autolabel(monkeypatch, det, rec)

        def boom(*a, **k):
            raise AssertionError("R10 must not run")
        monkeypatch.setattr(det, "geometry_seam", boom)
        assert det.main(live._argv(cap, tmp_path / name, "--no-depth-edge")) == 1
        reject = json.loads((tmp_path / name / "ov" / "cap_1_autolabel_v2_reject.json")
                            .read_text(encoding="utf-8"))
        assert reject["stage"] == "seam"


def test_geometry_seam_refuses_an_unvalidated_profile(det):
    rec = live._record(status="ambiguous")
    err = det.Reject("seam", ["seam_status_ambiguous(x)"])
    for name in ("big", "generic"):
        with pytest.raises(det.Reject) as raised:
            det.geometry_seam(rec, err, None, det.box_profiles.PROFILES[name])
        assert raised.value is err


# ------------------------------------------------------------ generic: no depth-edge fix

def test_depth_edge_skipped_for_generic_profile(det, tmp_path):
    profile = det.box_profiles.PROFILES["generic"]
    rec = live._record()
    _, seam_out, info = det.apply_depth_edge(
        rec, det.published_seam_line(rec), {}, tmp_path / "missing.pcd", profile=profile)
    assert info["applied"] is False
    assert info["reason"] == "depth_edge_not_validated_for_profile(generic)"


def test_depth_edge_unchanged_without_profile(det, tmp_path):
    rec = live._record()
    _, _, info = det.apply_depth_edge(rec, det.published_seam_line(rec), {},
                                      tmp_path / "missing.pcd")
    assert info["reason"] == "depth_edge_no_plane"      # the ordinary skip, not the profile one


def test_generic_profile_never_uses_r10_geometry_seam(det, tmp_path, monkeypatch):
    cap = live._capture(tmp_path)
    rec = live._record(status="ambiguous")
    rec["box_profile"] = {"name": "generic"}
    live._install_fake_autolabel(monkeypatch, det, rec)

    def boom(*a, **k):
        raise AssertionError("R10 must not run for the generic profile")
    monkeypatch.setattr(det, "geometry_seam", boom)
    assert det.main(live._argv(cap, tmp_path, "--no-depth-edge")) == 1
    reject = json.loads((tmp_path / "ov" / "cap_1_autolabel_v2_reject.json")
                        .read_text(encoding="utf-8"))
    assert reject["stage"] == "seam"
    assert reject["reasons"][0].startswith("seam_status_ambiguous")
    assert reject["box_profile"]["name"] == "generic"
    assert reject["cut_permission"]["status"] == "blocked"


# ------------------------------------------------------------ side seed profiles

def test_seed_small_and_big_are_unchanged(seed):
    snap, _ = sf._box()
    block = seed.compute_seed(snap)
    assert block["box_class"] == "small" and block["box_height_m"] == 0.0863
    assert block["box_class_source"] == "auto"
    assert block["box_height_source"] == "class_default:small"
    assert block["wall_offsets_m"]["along_seam"] == pytest.approx(0.003125)
    big, _ = sf._box(lr_len=0.193, fb_len=0.105)
    block = seed.compute_seed(big)
    assert block["box_class"] == "big" and block["box_height_m"] == 0.13425
    assert "outer_wall_unknown" in block["flags"]


def test_seed_snapshot_box_profile_wins_over_auto(seed):
    snap, _ = sf._box()
    snap["box_profile"] = {"name": "big"}
    block = seed.compute_seed(snap)
    assert block["box_class"] == "big" and block["box_class_source"] == "snapshot"


def test_seed_generic_uses_measured_height_and_unknown_walls(seed):
    snap, _ = sf._box(lr_len=0.39, fb_len=0.32, centre=(0.4, 0.0, 0.48))
    block = seed.compute_seed(snap, box_class="generic", measured_height_mm=480.0)
    assert block["box_class"] == "generic"
    assert block["box_height_m"] == pytest.approx(0.48)
    assert block["box_height_source"] == "measured"
    assert block["measured_height_mm"] == 480.0
    assert "outer_wall_unknown" in block["flags"]
    assert block["wall_offsets_m"] == {"along_seam": 0.0, "across_seam": 0.0}
    for face in ("left", "right", "back", "front"):
        entry = block["faces"][face]
        assert entry["plane"] == entry["top_edge_plane"]         # planes at the top edges
        assert entry["wall_source"] == "outer_wall_unknown"
    assert block["faces"]["left"]["tape_centerline"]["length_m"] == pytest.approx(0.48)


def test_seed_generic_without_a_height_fails_closed(seed):
    snap, _ = sf._box(lr_len=0.39, fb_len=0.32)
    with pytest.raises(seed.SeedError, match="height_unavailable"):
        seed.compute_seed(snap, box_class="generic")


def test_seed_generic_measured_height_from_snapshot_and_range_check(seed):
    snap, _ = sf._box(lr_len=0.39, fb_len=0.32)
    snap["box_profile"] = "generic"
    snap["measured_height_mm"] = 300.0
    assert seed.compute_seed(snap)["box_height_m"] == pytest.approx(0.3)
    with pytest.raises(seed.SeedError, match="outside"):
        seed.compute_seed(snap, measured_height_mm=3000.0)
    with pytest.raises(seed.SeedError, match="positive"):
        seed.compute_seed(snap, measured_height_mm=-1.0)


def test_seed_known_profile_measured_height_is_recorded_only(seed):
    snap, _ = sf._box()
    block = seed.compute_seed(snap, measured_height_mm=86.6)
    assert block["box_height_m"] == 0.0863 and block["measured_height_mm"] == 86.6


def test_seed_cli_profile_flags(seed, tmp_path):
    snap, _ = sf._box(lr_len=0.39, fb_len=0.32, centre=(0.4, 0.0, 0.48))
    path = tmp_path / "snap.json"
    path.write_text(json.dumps(snap), encoding="utf-8")
    assert seed.main(["--planner-json", str(path), "--box-profile", "generic",
                      "--measured-height-mm", "480"]) == 0
    block = json.loads(path.read_text(encoding="utf-8"))["side_tape_seed"]
    assert block["box_class"] == "generic" and block["box_class_source"] == "record_profile"
    path.write_text(json.dumps(snap), encoding="utf-8")
    assert seed.main(["--planner-json", str(path), "--box-profile", "generic"]) == 1
    assert json.loads(path.read_text(encoding="utf-8")) == snap          # unchanged


def test_fallback_honours_the_profile_exclude_env(det, monkeypatch):
    """Leave-one-type-out: with the small profile excluded the corner fallback must
    not call a small box 'small' (uses box_profiles.resolve, owned by the pipeline)."""
    rec = live._record()
    rec["top_face"]["corners_xyz_camera_m"] = {
        "back_left": [-0.055, -0.049, 0.5], "back_right": [0.055, -0.049, 0.5],
        "front_right": [0.055, 0.049, 0.5], "front_left": [-0.055, 0.049, 0.5]}
    assert det.resolve_box_profile(rec)[0].name == "small"
    monkeypatch.setenv("AUTOLABEL_V2_PROFILES_EXCLUDE", "small")
    profile, info = det.resolve_box_profile(rec)
    assert profile.is_generic and info["source"] == "corners_legacy"
    assert info["excluded_by_env"] == "small"
    monkeypatch.setenv("AUTOLABEL_V2_PROFILES_EXCLUDE", "big")
    assert det.resolve_box_profile(rec)[0].name == "small"
    rec["box_profile"] = {"name": "small"}                 # the record still wins
    monkeypatch.setenv("AUTOLABEL_V2_PROFILES_EXCLUDE", "small")
    assert det.resolve_box_profile(rec)[0].name == "small"


def test_measured_height_read_from_box_profile_block(det):
    rec = live._record()
    rec["box_profile"] = {"name": "small", "measured_height_mm": 86.5, "flags": []}
    assert det._measured_height_mm(rec) == 86.5
    rec["box_profile"]["measured_height_mm"] = None
    assert det._measured_height_mm(rec) is None
