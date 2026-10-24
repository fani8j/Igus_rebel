"""The truncation re-run (``process_capture(..., _forced=...)``) reuses the
first pass's SAM3-independent depth work and decoded image instead of
recomputing / re-decoding them, with an identical record."""
import copy
import json

import numpy as np
import pytest

import test_autolabel_v2_plane_trim as T  # noqa: E402
from box_seam_detection.core.autolabel import depth_top as _depth  # noqa: E402
from box_seam_detection.core.autolabel import pcd as _pcd  # noqa: E402
from box_seam_detection.core import pipeline as _p  # noqa: E402
from box_seam_detection.core.config import default_config  # noqa: E402


@pytest.fixture
def scene(monkeypatch, tmp_path):
    real_top = _depth.estimate_top
    T._patch(monkeypatch)             # camera, cloud, stubbed refine + depth cross-check
    monkeypatch.setattr(_depth, "estimate_top", real_top)   # keep the real depth estimate
    monkeypatch.setenv("AUTOLABEL_V2_TRUNC_RECOVERY", "on")
    return T._setup(tmp_path)


def _dump(rec):
    return json.dumps(rec, sort_keys=True, default=str)


def _run(paths, monkeypatch, reuse):
    with monkeypatch.context() as mp:      # undone after each variant (no stacked wrappers)
        return _run_in(paths, mp, reuse)


def _run_in(paths, monkeypatch, reuse):
    """First pass; the recovery hook re-runs the accepted winner through the
    real ``process_capture`` (``reuse`` False = the old forced dict without the
    depth / image hand-over). Returns (record of the re-run, counters)."""
    cnt = {"depth": 0, "imread": 0, "hook": 0, "est": 0}
    real_rdw, real_imread, real_est = _p._run_depth_work, _p.cv2.imread, _depth.estimate_top

    def rdw(*a, **k):
        cnt["depth"] += 1
        return real_rdw(*a, **k)

    def imread(*a, **k):
        cnt["imread"] += 1
        return real_imread(*a, **k)

    def est(*a, **k):
        cnt["est"] += 1
        return real_est(*a, **k)

    def hook(win0, cands, mode, cfg, depth_ctx, manifest, W, H, attempts, reasons, evaluate, rerun):
        cnt["hook"] += 1
        if reuse:
            return rerun(win0), {"mode": mode, "triggered": True, "recovered": True}
        rec = _p.process_capture("x", paths, cfg, "sam3_live", None, T._Session(T.TRUE_TOP),
                                 _forced={"winner": win0, "reasons": list(reasons),
                                          "attempts": list(attempts)})
        return rec, {"mode": mode, "triggered": True, "recovered": True}

    monkeypatch.setattr(_p, "_run_depth_work", rdw)
    monkeypatch.setattr(_p.cv2, "imread", imread)
    monkeypatch.setattr(_depth, "estimate_top", est)
    monkeypatch.setattr(_p, "_truncation_recovery", hook)
    rec, dbg = _p.process_capture("x", paths, default_config(), source="sam3_live",
                                  session=T._Session(T.TRUE_TOP))
    assert cnt["hook"] == 1
    return (rec, dbg), cnt


def test_rerun_reuses_depth_work_and_image(scene, monkeypatch):
    (r_new, d_new), c_new = _run(scene, monkeypatch, reuse=True)
    (r_old, d_old), c_old = _run(scene, monkeypatch, reuse=False)
    # first pass + re-run: the old path did everything twice, the new one once
    assert (c_old["depth"], c_old["est"], c_old["imread"]) == (2, 2, 2)
    assert (c_new["depth"], c_new["est"], c_new["imread"]) == (1, 1, 1)
    # the hook's return value is the re-run's (record, debug) pair
    assert _dump(r_new) == _dump(r_old)
    assert np.array_equal(d_new["bgr"], d_old["bgr"])
    assert json.dumps(d_new["depth_est"], default=str, sort_keys=True) == \
        json.dumps(d_old["depth_est"], default=str, sort_keys=True)


def test_rerun_without_depth_snapshot_recomputes(scene, monkeypatch):
    cnt = {"depth": 0}
    real = _p._run_depth_work

    def rdw(*a, **k):
        cnt["depth"] += 1
        return real(*a, **k)

    monkeypatch.setattr(_p, "_run_depth_work", rdw)
    rec, _ = _p.process_capture("x", scene, default_config(), source="sam3_live",
                                session=T._Session(T.TRUE_TOP))
    assert cnt["depth"] == 1          # no recovery hook ran: unchanged single pass


def test_forced_depth_and_image_are_used_verbatim(scene, monkeypatch):
    """A handed-over depth result / image replaces the file work."""
    seen = {}

    def hook(win0, cands, mode, cfg, depth_ctx, manifest, W, H, attempts, reasons, evaluate, rerun):
        seen["win"] = win0
        seen["attempts"], seen["reasons"] = list(attempts), list(reasons)
        seen["ctx_est"] = copy.deepcopy(depth_ctx.get("_table"))
        return None, None

    monkeypatch.setattr(_p, "_truncation_recovery", hook)
    cfg = default_config()
    session = T._Session(T.TRUE_TOP)
    ref_rec, ref_dbg = _p.process_capture("x", scene, cfg, source="sam3_live", session=session)
    # now a poisoned environment: any recompute / decode would raise
    snap = _p._clone_depth_work(_p._run_depth_work(
        _depth, _pcd.read_pcd_xyz, scene, ref_dbg["cam"], T.H, T.W, cfg, True, True))

    def boom(*a, **k):
        raise AssertionError("must not recompute / decode")

    monkeypatch.setattr(_p, "_run_depth_work", boom)
    monkeypatch.setattr(_p.cv2, "imread", boom)
    rec, dbg = _p.process_capture(
        "x", scene, cfg, source="sam3_live", session=session,
        _forced={"winner": seen["win"], "reasons": seen["reasons"],
                 "attempts": seen["attempts"], "depth": snap, "bgr": ref_dbg["bgr"].copy()})
    assert _dump(dbg["depth_est"]) == _dump(snap["est"])
    assert rec["top_face"]["corners_px"] == ref_rec["top_face"]["corners_px"]
    # the handed-over result is cloned per use: the re-run cannot alter the snapshot
    assert dbg["depth_est"] is not snap["est"]
