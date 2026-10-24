"""M5 interface: live.side_seed accepts optional MEASURED side-tape centrelines
per face (absent -> today's extrapolated centreline, unchanged), records which
was used, and flags side_tape_expected_not_found. No detection."""

import importlib.util
import json
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("sti_seed_helpers", str(TESTS / "test_side_face_seed.py"))
sf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sf)


@pytest.fixture()
def seed():
    return sf._load(sf.SEED, "side_face_seed_tape_iface")


def _measured(block, face, dx=0.001, dz=0.0, length=0.07):
    """A measured centreline on the face's wall plane, shifted along the top edge by dx."""
    centre = block["faces"][face]["tape_centerline"]
    top = centre["top"]
    edge = block["faces"][face]["top_edge"]
    direction = [edge["end"][a] - edge["start"][a] for a in "xyz"]
    norm = sum(c * c for c in direction) ** 0.5
    unit = [c / norm for c in direction]
    down = [centre["direction"][a] for a in "xyz"]
    t = [top[a] + dx * u for a, u in zip("xyz", unit)]
    b = [t[i] + length * down[i] for i in range(3)]
    return {"top": {"x": t[0], "y": t[1], "z": t[2]}, "bottom": {"x": b[0], "y": b[1], "z": b[2]}}


def test_absent_measurement_is_todays_centreline_with_source_recorded(seed):
    snap, _ = sf._box()
    block = seed.compute_seed(snap)
    for face in ("left", "right"):
        tape = block["faces"][face]["tape_centerline"]
        assert tape["source"] == "extrapolated_from_seam" and "extrapolated" not in tape
    assert block["side_tape"] == {"expected": False, "measured_faces": [],
                                  "centerline_source": {"left": "extrapolated_from_seam",
                                                        "right": "extrapolated_from_seam"}}
    assert "side_tape_expected_not_found" not in block["flags"]


def test_absent_measurement_leaves_all_other_values_identical(seed):
    snap, _ = sf._box(rot=sf._rot(yaw_deg=21.0))
    plain = seed.compute_seed(snap)
    measured = {"left": _measured(plain, "left")}
    with_tape = seed.compute_seed(snap, side_tape=measured)
    # the right face and every non-tape field are untouched
    assert with_tape["faces"]["right"] == plain["faces"]["right"]
    assert with_tape["faces"]["back"] == plain["faces"]["back"]
    assert with_tape["box_height_m"] == plain["box_height_m"]
    assert with_tape["faces"]["left"]["plane"] == plain["faces"]["left"]["plane"]


def test_measured_centreline_replaces_the_extrapolated_one_and_keeps_it(seed):
    snap, _ = sf._box()
    plain = seed.compute_seed(snap)
    measured = {"left": _measured(plain, "left", dx=0.002, length=0.05)}
    block = seed.compute_seed(snap, side_tape=measured)
    tape = block["faces"]["left"]["tape_centerline"]
    assert tape["source"] == "measured"
    assert tape["top"] == measured["left"]["top"] and tape["bottom"] == measured["left"]["bottom"]
    assert tape["length_m"] == pytest.approx(0.05)
    assert {k: tape["extrapolated"][k] for k in ("top", "bottom", "direction", "length_m")} == {
        k: plain["faces"]["left"]["tape_centerline"][k]
        for k in ("top", "bottom", "direction", "length_m")}
    assert block["faces"]["right"]["tape_centerline"]["source"] == "extrapolated_from_seam"
    assert block["side_tape"]["measured_faces"] == ["left"]


def test_expected_without_measurement_is_flagged(seed):
    snap, _ = sf._box()
    block = seed.compute_seed(snap, side_tape_expected=True)
    assert "side_tape_expected_not_found" in block["flags"]
    assert block["side_tape"]["expected"] is True


def test_expected_with_both_faces_measured_is_not_flagged(seed):
    snap, _ = sf._box()
    plain = seed.compute_seed(snap)
    measured = {f: _measured(plain, f) for f in ("left", "right")}
    block = seed.compute_seed(snap, side_tape_expected=True, side_tape=measured)
    assert "side_tape_expected_not_found" not in block["flags"]


def test_expected_with_one_face_missing_is_flagged(seed):
    snap, _ = sf._box()
    plain = seed.compute_seed(snap)
    block = seed.compute_seed(snap, side_tape_expected=True,
                              side_tape={"left": _measured(plain, "left")})
    assert "side_tape_expected_not_found" in block["flags"]


def test_expected_flag_from_snapshot(seed):
    snap, _ = sf._box()
    snap["side_tape_expected"] = True
    assert "side_tape_expected_not_found" in seed.compute_seed(snap)["flags"]


def test_profile_attribute_marks_expected(seed, monkeypatch):
    snap, _ = sf._box()
    profile = seed.box_profiles.PROFILES["small"]
    monkeypatch.setattr(type(profile), "side_tape_expected", True, raising=False)
    assert "side_tape_expected_not_found" in seed.compute_seed(snap)["flags"]


def test_measured_snapshot_key_is_accepted(seed):
    snap, _ = sf._box()
    plain = seed.compute_seed(snap)
    snap["side_tape_measured"] = {"right": _measured(plain, "right")}
    block = seed.compute_seed(snap)
    assert block["faces"]["right"]["tape_centerline"]["source"] == "measured"


@pytest.mark.parametrize("bad, message", [
    ({"back": {"top": [0, 0, 0], "bottom": [0, 0, 0.1]}}, "unknown face"),
    ({"left": {"top": [0, 0, 0]}}, "left.bottom"),
    ({"left": {"top": [0, 0, float("nan")], "bottom": [0, 0, 0.1]}}, "finite"),
    ({"left": "x"}, "must be an object"),
    ("x", "must be an object"),
])
def test_invalid_measurements_raise(seed, bad, message):
    snap, _ = sf._box()
    with pytest.raises(seed.SeedError, match=message):
        seed.compute_seed(snap, side_tape=bad)


def _flags(block):
    return [f for f in block["flags"] if f.startswith("side_tape_measured_invalid")]


def _invalid_case(seed, mutate, text):
    snap, _ = sf._box()
    plain = seed.compute_seed(snap)
    m = _measured(plain, "left")
    mutate(m, plain["faces"]["left"])
    block = seed.compute_seed(snap, side_tape_expected=True, side_tape={"left": m})
    tape = block["faces"]["left"]["tape_centerline"]
    assert tape["source"] == "extrapolated_from_seam"          # the measurement is ignored
    assert tape["top"] == plain["faces"]["left"]["tape_centerline"]["top"]
    flags = _flags(block)
    assert len(flags) == 1 and text in flags[0] and "left" in flags[0]
    assert "side_tape_expected_not_found" in block["flags"]
    assert block["side_tape"]["measured_faces"] == []


def test_measured_point_far_from_the_wall_is_ignored_and_flagged(seed):
    def mutate(m, face):
        n = face["plane"]["normal"]
        m["top"] = {a: m["top"][a] + 0.05 * n[a] for a in "xyz"}
    _invalid_case(seed, mutate, "face wall plane")


def test_degenerate_measured_line_is_ignored_and_flagged(seed):
    snap, _ = sf._box()
    plain = seed.compute_seed(snap)
    m = _measured(plain, "left", length=0.001)
    block = seed.compute_seed(snap, side_tape={"left": m})
    assert "degenerate" in _flags(block)[0]


def test_top_below_bottom_is_ignored(seed):
    def mutate(m, face):
        m["top"], m["bottom"] = m["bottom"], m["top"]
    _invalid_case(seed, mutate, "not above")


def test_too_long_line_is_ignored(seed):
    snap, _ = sf._box()
    plain = seed.compute_seed(snap)
    m = _measured(plain, "left", length=0.0863 + 0.02)
    assert "extent" in _flags(seed.compute_seed(snap, side_tape={"left": m}))[0]
    ok = _measured(plain, "left", length=0.0863 + 0.009)
    assert _flags(seed.compute_seed(snap, side_tape={"left": ok})) == []


def test_tilted_line_is_ignored(seed):
    def mutate(m, face):
        edge = face["top_edge"]
        d = [edge["end"][a] - edge["start"][a] for a in "xyz"]
        norm = sum(c * c for c in d) ** 0.5
        # tilt the bottom 25 mm along the top edge over a 70 mm line (~20 deg)
        m["bottom"] = {a: m["bottom"][a] + 0.025 * d[i] / norm for i, a in enumerate("xyz")}
    _invalid_case(seed, mutate, "from vertical")


def test_line_far_from_the_seam_point_along_the_face_is_ignored(seed):
    snap, _ = sf._box()
    plain = seed.compute_seed(snap)
    m = _measured(plain, "left", dx=0.03)
    assert "seam point along the face" in _flags(seed.compute_seed(snap, side_tape={"left": m}))[0]
    ok = _measured(plain, "left", dx=0.019)
    assert _flags(seed.compute_seed(snap, side_tape={"left": ok})) == []


def test_used_measurement_drops_inconsistent_extrapolated_fields(seed):
    snap, _ = sf._box()
    plain = seed.compute_seed(snap)
    block = seed.compute_seed(snap, side_tape={"left": _measured(plain, "left")})
    tape = block["faces"]["left"]["tape_centerline"]
    old = plain["faces"]["left"]["tape_centerline"]
    for key in ("top_edge_point", "top_edge_point_bottom"):
        assert key not in tape
        assert tape["extrapolated"][key] == old[key]
    assert tape["seam_point"] == old["seam_point"]               # seam-side facts stay


def test_cli_side_tape_json_and_expected_flag(seed, tmp_path):
    snap, _ = sf._box()
    path = tmp_path / "snap.json"
    path.write_text(json.dumps(snap), encoding="utf-8")
    plain = seed.compute_seed(snap)
    tape_path = tmp_path / "tape.json"
    tape_path.write_text(json.dumps({"left": _measured(plain, "left")}), encoding="utf-8")
    assert seed.main(["--planner-json", str(path), "--side-tape-json", str(tape_path),
                      "--side-tape-expected"]) == 0
    block = json.loads(path.read_text(encoding="utf-8"))["side_tape_seed"]
    assert block["faces"]["left"]["tape_centerline"]["source"] == "measured"
    assert "side_tape_expected_not_found" in block["flags"]       # right face has none


def test_cli_unreadable_side_tape_json_fails_closed(seed, tmp_path):
    snap, _ = sf._box()
    path = tmp_path / "snap.json"
    path.write_text(json.dumps(snap), encoding="utf-8")
    assert seed.main(["--planner-json", str(path),
                      "--side-tape-json", str(tmp_path / "absent.json")]) == 1
    assert json.loads(path.read_text(encoding="utf-8")) == snap


# ------------------------------------------------------------ orchestrator hand-off

def test_orchestrator_passes_detector_profile_and_height_to_the_seed(tmp_path):
    orch = sf._load(sf.ORCHESTRATOR, "capture_and_query_depth_sti")
    seam = tmp_path / "seam.json"
    seam.write_text(json.dumps({"autolabel_v2": {"box_profile": {"name": "generic", "source": "record"},
                                                 "measured_height_mm": 480.25}}),
                    encoding="utf-8")
    assert orch.side_seed_profile_args(seam) == [
        "--box-profile", "generic", "--measured-height-mm", "480.25"]
    seam.write_text(json.dumps({"autolabel_v2": {"box_profile": {"name": "small",
                                                                 "source": "corners_legacy"},
                                                 "measured_height_mm": None}}), encoding="utf-8")
    assert orch.side_seed_profile_args(seam) == []
    seam.write_text(json.dumps({"autolabel_v2": {"box_profile": {"name": "weird", "source": "record"},
                                                 "measured_height_mm": None}}),
                    encoding="utf-8")
    assert orch.side_seed_profile_args(seam) == []
    assert orch.side_seed_profile_args(tmp_path / "absent.json") == []
    assert orch.side_seed_profile_args(None) == []
