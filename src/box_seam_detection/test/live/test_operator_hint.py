"""Operator hint for no_top_face rejects (live.detector)."""
import importlib
from pathlib import Path

import numpy as np
import pytest

def _mod():
    return importlib.import_module("box_seam_detection.live.detector")


WH = (640, 480)
BOX = (200, 150, 300, 250)


def test_thresholds():
    m = _mod()
    f = m.hint_suggestions
    assert "obliquely (70 deg)" in f(70.0, 0.0, BOX, WH)[0]
    assert "move it toward the table centre" in f(70.0, 0.0, BOX, WH)[0]
    assert "top face not found" in f(62.0, 0.15, BOX, WH)[0]   # strict >
    assert "obliquely" in f(62.1, 0.0, BOX, WH)[0]
    s = f(40.0, 0.151, BOX, WH)
    assert len(s) == 1 and "overexposed (15% saturated)" in s[0]
    assert "reduce light / lock camera exposure" in s[0]
    assert "partly outside" in f(40.0, 0.0, (3, 150, 300, 250), WH)[0]
    assert "top face not found" in f(40.0, 0.0, (6, 150, 300, 250), WH)[0]
    assert "partly outside" in f(None, None, (200, 150, 636, 250), WH)[0]
    assert f(None, None, None, None) == [
        "top face not found - re-run or check placement"]


def test_saturated_fraction_and_box_from_record():
    m = _mod()
    g = np.zeros((480, 640), np.uint8)
    g[100:300, 150:350] = 255
    # box 200..300 x 150..250 grown by 50 px each side = 150..350 x 100..300
    assert m.saturated_fraction(g, BOX) == pytest.approx(1.0)
    g[:] = 0
    g[150:250, 200:300] = 255          # only the box: 10^4 of 200^2
    assert m.saturated_fraction(g, BOX) == pytest.approx(0.25)
    rec = {"provenance": {"attempts": [
        {"source": "sam3_live", "plane_trimmed": True,
         "coarse": [[0, 0], [1, 0], [1, 1], [0, 1]]},
        {"source": "sam3_live", "coarse": [[200, 150], [300, 150],
                                           [300, 250], [200, 250]]}]}}
    assert m._sam3_box_px(rec) == (200.0, 150.0, 300.0, 250.0)
    assert m._sam3_box_px({}) is None
