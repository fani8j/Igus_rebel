# Provenance: ported from rgbd_captures/tests/test_s1_concurrency.py (HEAD 6df5994); imports made package-relative.
"""S1: SAM3-independent depth work runs concurrently with the SAM3 query;
outputs are identical to the sequential path (AUTOLABEL_V2_S1_OFF=1)."""
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest


import test_autolabel_v2_plane_trim as T  # noqa: E402
from box_seam_detection.core.autolabel import depth_top as _depth  # noqa: E402
from box_seam_detection.core.autolabel import pcd as _pcd  # noqa: E402
from box_seam_detection.core import pipeline as _p  # noqa: E402
from box_seam_detection.core.config import default_config  # noqa: E402


@pytest.fixture
def scene(monkeypatch, tmp_path):
    xyz = T._cloud()
    monkeypatch.setattr(_depth, "load_capture_camera",
                        lambda m: (T.CAM, "synthetic"))
    monkeypatch.setattr(_pcd, "read_pcd_xyz", lambda p: (xyz.astype(np.float32), None))
    return T._setup(tmp_path), xyz


def _run(paths, source, session, monkeypatch, off):
    if off:
        monkeypatch.setenv("AUTOLABEL_V2_S1_OFF", "1")
    else:
        monkeypatch.delenv("AUTOLABEL_V2_S1_OFF", raising=False)
    rec, _ = _p.process_capture("x", paths, default_config(), source=source,
                                session=session)
    return json.dumps(rec, sort_keys=True, default=str)


def test_depth_overlaps_sam3_query(scene, monkeypatch):
    paths, _ = scene
    real = _depth.estimate_top
    ev = {}

    def slow_top(*a, **k):
        ev["top_start"] = time.perf_counter()
        ev["top_thread"] = threading.current_thread().name
        time.sleep(0.4)
        out = real(*a, **k)
        ev["top_end"] = time.perf_counter()
        return out

    class Sess(T._Session):
        def query(self, image_path, timeout_s=300):
            ev["q_start"] = time.perf_counter()
            time.sleep(0.4)
            ev["q_end"] = time.perf_counter()
            return super().query(image_path, timeout_s)

    monkeypatch.setattr(_depth, "estimate_top", slow_top)
    _run(paths, "sam3_live", Sess(T.TRUE_TOP), monkeypatch, off=False)
    assert ev["top_thread"] == "s1-depth"
    assert ev["top_start"] < ev["q_end"] and ev["q_start"] < ev["top_end"]


def test_off_switch_is_sequential(scene, monkeypatch):
    paths, _ = scene
    real = _depth.estimate_top
    ev = {}

    def top(*a, **k):
        ev["thread"] = threading.current_thread().name
        ev["end"] = time.perf_counter()
        return real(*a, **k)

    class Sess(T._Session):
        def query(self, image_path, timeout_s=300):
            ev["q_start"] = time.perf_counter()
            return super().query(image_path, timeout_s)

    monkeypatch.setattr(_depth, "estimate_top", top)
    _run(paths, "sam3_live", Sess(T.TRUE_TOP), monkeypatch, off=True)
    assert ev["thread"] == threading.main_thread().name
    assert ev["end"] <= ev["q_start"]


@pytest.mark.parametrize("source", ["depth_only", "sam3_live", "auto"])
def test_records_identical_on_off(scene, monkeypatch, source):
    paths, _ = scene
    a = _run(paths, source, T._Session(T.TRUE_TOP), monkeypatch, off=True)
    b = _run(paths, source, T._Session(T.TRUE_TOP), monkeypatch, off=False)
    assert a == b


@pytest.mark.parametrize("what", ["estimate", "pcd", "calib"])
def test_depth_errors_surface_identically(scene, monkeypatch, what):
    paths, _ = scene
    if what == "estimate":
        def boom(*a, **k):
            raise RuntimeError("kaboom")
        monkeypatch.setattr(_depth, "estimate_top", boom)
    elif what == "pcd":
        def boom(p):
            raise OSError("no pcd")
        monkeypatch.setattr(_pcd, "read_pcd_xyz", boom)
    else:
        def boom(m):
            raise ValueError("bad cal")
        monkeypatch.setattr(_depth, "load_capture_camera", boom)
    outs = []
    for off in (True, False):
        outs.append(_run(paths, "depth_only", T._Session(T.TRUE_TOP),
                         monkeypatch, off=off))
    assert outs[0] == outs[1]
    assert {"estimate": "depth_estimate_failed(kaboom)",
            "pcd": "depth_pcd_unreadable(no pcd)",
            "calib": "depth_bad_calibration(bad cal)"}[what] in outs[0]


def test_table_plane_reuse_is_bit_identical(scene):
    paths, xyz = scene
    xyz = xyz.astype(np.float32)  # what read_pcd_xyz returns (old ctx xyz)
    cfg = default_config()
    res = _p._run_depth_work(_depth, _pcd.read_pcd_xyz, paths, T.CAM, T.H,
                             T.W, cfg, True, True)
    assert res["xyz"].dtype == np.float64
    table = res["extra"]["_table"]
    xyz = xyz.astype(np.float32)  # what read_pcd_xyz hands the pipeline
    lazy = _p.estimate_table_plane({"xyz": xyz, "cam": T.CAM,
                                    "dcfg": cfg["depth_top"]})
    assert np.array_equal(table["n"], lazy["n"]) and table["d"] == lazy["d"]
    ref_idx, ref_uv = _p._depth_projection({"xyz": xyz, "cam": T.CAM})
    idx, uv = res["extra"]["_proj"]
    assert np.array_equal(idx, ref_idx) and np.array_equal(uv, ref_uv)


def test_table_plane_not_reused_below_table_min(scene):
    paths, _ = scene
    cfg = default_config()
    cfg["depth_top"] = dict(cfg["depth_top"], table_min_inliers=10 ** 9)
    res = _p._run_depth_work(_depth, _pcd.read_pcd_xyz, paths, T.CAM, T.H,
                             T.W, cfg, True, False)
    assert "_table" not in res["extra"]
