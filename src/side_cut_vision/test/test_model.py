import json

import numpy as np
import pytest

import synth
from side_cut_vision.model import SnapshotError, build_arm_model, load_snapshot, parse_snapshot


def test_left_and_right_basis_conventions():
    snap = load_snapshot(synth.SNAPSHOT)
    L = build_arm_model(snap, "left")
    R = build_arm_model(snap, "right")
    for m in (L, R):
        B = m.basis
        assert np.allclose(B @ B.T, np.eye(3), atol=1e-9)
        assert np.linalg.norm(m.along @ m.normal) < 1e-9
        # +across points out of the carton: away from the carton centre
        assert (m.S - m.centre) @ m.across > 0.05
        assert m.crossing_to_edge_mm < 1.0
    assert L.across[1] > 0.99 and R.across[1] < -0.99     # left face outward = +y, right = -y (world)
    assert L.along[0] < -0.99 and R.along[0] < -0.99       # back -> front = -x
    assert L.across_flipped and not R.across_flipped       # executor raw n x along is inward for left
    assert np.allclose(L.S, [0.4224510, -0.0083212, 0.0228989], atol=1e-6)


def test_local_roundtrip():
    m = build_arm_model(load_snapshot(synth.SNAPSHOT), "left")
    P = np.random.default_rng(0).normal(size=(5, 3))
    assert np.allclose(m.from_local(m.to_local(P)), P)


def test_bad_snapshots_raise_snapshoterror():
    with open(synth.SNAPSHOT) as f:
        good = json.load(f)
    with pytest.raises(SnapshotError):
        parse_snapshot({})
    bad = dict(good)
    bad["schema"] = "other/v9"
    with pytest.raises(SnapshotError):
        parse_snapshot(bad)
    bad = dict(good)
    del bad["seam_start"]
    with pytest.raises(SnapshotError):
        parse_snapshot(bad)
    bad = dict(good)
    bad["surface_normal"] = {"x": 0, "y": 0, "z": float("nan")}
    with pytest.raises(SnapshotError):
        parse_snapshot(bad)
    with pytest.raises(SnapshotError):
        load_snapshot("/nonexistent/snapshot.json")
