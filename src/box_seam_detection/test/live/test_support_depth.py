"""snapshot.support_depth_below_top: optional additive snapshot field for the executor's carton collision height."""
import math

from box_seam_detection.live.snapshot import support_depth_below_top


def _doc(record=None, measured=None):
    return {"autolabel_v2": {"box_profile": {"name": "generic", "source": "record", "record": record},
                             "measured_height_mm": measured}}


def test_hidden_stand_clearance():
    out = support_depth_below_top(_doc({"support_clearance_mm": 967.0}))
    assert math.isclose(out["support_depth_below_top_m"], 0.967)
    assert out["support_depth_below_top_source"] == "support_clearance_mm"


def test_gravity_unchecked_is_ignored():
    rec = {"support_clearance_mm": 967.0, "support_gravity_unchecked": True}
    assert support_depth_below_top(_doc(rec)) == {}
    out = support_depth_below_top(_doc(rec, measured=134.25))
    assert math.isclose(out["support_depth_below_top_m"], 0.13425)
    assert out["support_depth_below_top_source"] == "measured_height_mm"


def test_table_cell_measured_height():
    out = support_depth_below_top(_doc({"measured_height_mm": 86.3}, measured=86.3))
    assert math.isclose(out["support_depth_below_top_m"], 0.0863)


def test_absent_or_invalid_omits():
    assert support_depth_below_top(_doc({})) == {}
    assert support_depth_below_top({}) == {}
    assert support_depth_below_top(None) == {}
    for bad in (0.0, -5.0, float("nan"), float("inf"), True, "9", None):
        assert support_depth_below_top(_doc({"support_clearance_mm": bad})) == {}
        assert support_depth_below_top(_doc({}, measured=bad)) == {}
