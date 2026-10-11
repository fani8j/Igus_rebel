# Provenance: ported from rgbd_captures/tests/test_integration_round.py (HEAD 6df5994); imports made package-relative.
"""Integration round: tape shadow, generic confirmation tolerance,
leave-one-type-out profile exclusion, per-request SAM3 gates, reseam profile."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest


from box_seam_detection.core import box_profiles as bp  # noqa: E402
from box_seam_detection.core import pipeline as P  # noqa: E402
from box_seam_detection.core import seam_estimator as se  # noqa: E402
from box_seam_detection.core.config import default_config  # noqa: E402
import test_seam_estimator as _ts  # noqa: E402
import test_box_profiles as _tb  # noqa: E402


# ---- 3. leave-one-type-out ---------------------------------------------------

def test_exclude_default_empty_is_todays_behaviour(monkeypatch):
    monkeypatch.delenv(bp.ENV_EXCLUDE, raising=False)
    assert bp.excluded_names() == ()
    assert bp.classify_legacy(0.11).name == "small"
    assert bp.resolve(0.1117, 0.0992, 86.2)[0].name == "small"
    assert set(bp.heights_m()) == {"big", "small"}


def test_exclude_small_everywhere(monkeypatch):
    monkeypatch.setenv(bp.ENV_EXCLUDE, "small, SMALL")
    assert bp.excluded_names() == ("small",)
    assert bp.classify(110, 98, 86.3)[0].is_generic
    assert bp.classify_legacy(0.11).is_generic
    assert bp.classify_legacy(0.19).name == "big"
    prof, info = bp.resolve(0.1117, 0.0992, 86.2)
    assert prof.is_generic and info["reason"] == "legacy_class_excluded"
    assert info["legacy_class"] == "small" and info["profiles_excluded"] == ["small"]
    # the height-matches-legacy rescue must not bring the class back
    assert bp.resolve(0.101, 0.0338, 85.9)[0].is_generic
    assert "small" not in bp.heights_m() and "small" not in bp.known_dims_mm_table()
    assert "small" not in bp.view_shift_mm_table()
    assert bp.is_generic_height(86.5) and not bp.is_generic_height(136.0)
    # big still resolves
    assert bp.resolve(0.1927, 0.1048, 136.4)[0].name == "big"


def test_exclude_all_known_makes_every_height_generic(monkeypatch):
    monkeypatch.setenv(bp.ENV_EXCLUDE, "small,big")
    assert bp.is_generic_height(86.0) and bp.is_generic_height(134.0)
    assert bp.resolve(0.1927, 0.1048, 136.4)[0].is_generic


def test_exclude_recorded_in_box_profile_and_generic_downstream(monkeypatch):
    monkeypatch.setenv(bp.ENV_EXCLUDE, "small")
    prof, rec = P.resolve_box_profile(_tb._xyz(111.0, 98.0),
                                      {"measured_height_mm": 86.4})
    assert prof.is_generic and rec["profiles_excluded"] == ["small"]
    assert rec["legacy_class"] == "small" and not rec["geometry_axis_validated"]
    # generic downstream: no R10, shift 0, fold uses the measured height
    assert se.geometry_class_axis(_tb._xyz(94.6, 108.7), None, prof.name)[0] is None
    est = default_config()["seam_estimator"]
    xyz = np.array([[-.05, -.05, .5], [.05, -.05, .5], [.05, .05, .5], [-.05, .05, .5]])
    assert se.expected_view_shift(xyz, "lr", est, prof.name)["expected_shift_mm"] == 0.0
    q = np.array([[100, 100], [300, 100], [300, 250], [100, 250]], float)
    _, _, info = P.apply_corner_fold(np.zeros((400, 400, 3), np.uint8), q, {}, None,
                                     _tb._xyz(111, 98), default_config()["corner_fold"],
                                     profile=prof, measured_height_m=None)
    assert info["reason"] == "no_height_generic"


# ---- 2. generic confirmation tolerance ----------------------------------------

def test_profile_confirm_tolerance_values():
    assert bp.PROFILES["small"].confirm_max_offset_mm == 1.5
    assert bp.PROFILES["big"].confirm_max_offset_mm == 1.5
    assert bp.PROFILES["generic"].confirm_max_offset_mm == 4.0
    assert default_config()["seam_estimator"]["confirm_max_offset_mm"] == 1.5


def test_generic_confirms_a_2mm_offset_known_profiles_do_not():
    xyz = _ts._small_box_xyz()
    img, corners, _ = _ts._synth_face(seam_axis="lr", gap_offset_px=6)
    cfg = _ts._shift_cfg(2.0, False)          # no view shift: raw ~2 mm offset
    res_small = se.estimate_seam(img, corners, est_cfg=cfg, corners_xyz=xyz,
                                 profile="small")
    res_none = se.estimate_seam(img, corners, est_cfg=cfg, corners_xyz=xyz)
    res_gen = se.estimate_seam(img, corners, est_cfg=cfg, corners_xyz=xyz,
                               profile="generic")
    assert res_small["status"] == "estimated" == res_none["status"]
    assert res_gen["status"] == "ok" and res_gen["axis"] == "lr"
    # the published line is still the prior
    for k in ("seam_left", "seam_right"):
        for a, b in zip(res_gen["refined_endpoints_px"][k],
                        res_gen["prior_endpoints_px"][k]):
            assert abs(a - b) < 0.05


# ---- 4. SAM3 gates -------------------------------------------------------------

class _FakeProc:
    def __init__(self):
        self.sent = []
        self.stdin = self
        self.stdout = self

    def write(self, s):
        self.sent.append(s)

    def flush(self):
        pass

    def poll(self):
        return None


def _worker(monkeypatch, mode="subprocess"):
    from box_seam_detection.core import sam3_session as S
    w = S.Sam3Worker.__new__(S.Sam3Worker)
    w.mode, w.proc, w._seq, w.threshold = mode, _FakeProc(), 0, 0.5
    import threading
    w._lock = threading.RLock()
    w.model_revision = None
    w.daemon_mode = "on"
    monkeypatch.setattr(w, "_readline", lambda timeout_s: json.dumps(
        {"id": w._seq, "ok": True}) + "\n", raising=False)
    return S, w


def test_query_gates_subprocess_path(monkeypatch):
    S, w = _worker(monkeypatch)
    w.query("a.png")
    w.query("a.png", gates={"max_area_frac": 0.5})
    plain = json.loads(w.proc.sent[0])
    gated = json.loads(w.proc.sent[1])
    assert "gates" not in plain          # default: request unchanged
    assert gated["gates"] == {"max_area_frac": 0.5}
    assert set(plain) == {"id", "image", "prompts", "threshold"}


def test_query_gates_daemon_path(monkeypatch):
    S, w = _worker(monkeypatch, mode="daemon")
    seen = []

    def fake(req, timeout_s):
        seen.append(dict(req))
        return {"ok": True}

    monkeypatch.setattr(w, "_query_daemon", fake)
    w.query("a.png")
    w.query("a.png", gates={"min_brown_frac": 0.0})
    assert "gates" not in seen[0] and set(seen[0]) == {"id", "image", "prompts", "threshold"}
    assert seen[1]["gates"] == {"min_brown_frac": 0.0}


# ---- 5. reseam passes the record's profile --------------------------------------

def test_reseam_record_passes_profile(monkeypatch, tmp_path):
    import cv2
    from box_seam_detection.core import run as R
    seen = {}

    def fake_stage(bgr, quad, cfg, corners_xyz=None, profile=None):
        seen["profile"] = profile
        return ({"status": "estimated", "endpoints_px": None, "axis": "lr",
                 "score": 0.0, "source": "prior", "estimate": {}},
                {"reasons": []})

    monkeypatch.setattr(R, "compute_seam_stage", fake_stage)
    img = np.full((60, 80, 3), 90, np.uint8)
    cv2.imwrite(str(tmp_path / "a.png"), img)
    names = ("back_left", "back_right", "front_right", "front_left")
    pts = [[10, 10], [60, 10], [60, 50], [10, 50]]
    rec = {"image": str(tmp_path / "a.png"),
           "top_face": {"corners_px": dict(zip(names, pts)),
                        "corners_xyz_camera_m": None},
           "provenance": {"corner_fold": {"x": 1}},
           "checks": {"reasons": []}, "box_profile": {"name": "generic"},
           "seam": {}}
    cfg = default_config()
    try:
        R.reseam_record(rec, {"image": str(tmp_path / "a.png")}, cfg, "src")
    except Exception:
        pass                      # only the stage call matters here
    assert seen["profile"] == "generic"
    rec2 = dict(rec)
    rec2.pop("box_profile")
    try:
        R.reseam_record(rec2, {"image": str(tmp_path / "a.png")}, cfg, "src")
    except Exception:
        pass
    assert seen["profile"] is None


# ---- 1. tape shadow ---------------------------------------------------------------

def test_tape_shadow_off_by_default(monkeypatch):
    monkeypatch.delenv("AUTOLABEL_V2_TAPE_SHADOW", raising=False)
    img, corners, _ = _ts._synth_face(seam_axis="lr")
    sd, _ = P.compute_seam_stage(img, corners, default_config(),
                                 corners_xyz=_ts._small_box_xyz())
    assert "tape_strip_shadow" not in sd


def test_tape_shadow_recorded_and_never_changes_the_seam(monkeypatch):
    img, corners, _ = _ts._synth_face(seam_axis="lr")
    xyz = _ts._small_box_xyz()
    cfg = default_config()
    monkeypatch.delenv("AUTOLABEL_V2_TAPE_SHADOW", raising=False)
    base, _ = P.compute_seam_stage(img, corners, cfg, corners_xyz=xyz)
    monkeypatch.setenv("AUTOLABEL_V2_TAPE_SHADOW", "1")
    sh, _ = P.compute_seam_stage(img, corners, cfg, corners_xyz=xyz)
    t = sh.pop("tape_strip_shadow")
    assert json.dumps(sh, sort_keys=True, default=str) == json.dumps(
        base, sort_keys=True, default=str)
    for k in ("ok", "best_axis", "axis_margin", "n_candidates", "n_qualified",
              "best", "gap_line_found", "agrees_with_published_axis", "time_s"):
        assert k in t, k
    assert "error" not in t, t
    assert set(t["n_candidates"]) == {"lr", "bf"}
    json.dumps(t)
    if t["best_axis"] is not None:
        assert t["agrees_with_published_axis"] == (t["best_axis"] == sh["axis"])


def test_tape_shadow_never_raises(monkeypatch):
    monkeypatch.setenv("AUTOLABEL_V2_TAPE_SHADOW", "1")
    t = P.tape_strip_shadow(np.zeros((10, 10, 3), np.uint8),
                            np.zeros((4, 2)), None, "lr")
    assert t["ok"] is False and "time_s" in t


def test_tape_shadow_axis_label_swapped_with_corner_names():
    from box_seam_detection.core import corner_names as cn
    seam = {"axis": "lr", "tape_strip_shadow": {
        "best_axis": "lr", "n_candidates": {"lr": 2, "bf": 0},
        "best": {"axis": "lr"}}}
    sw = cn.swap_axis_labels(seam)
    assert sw["axis"] == "bf" and sw["tape_strip_shadow"]["best_axis"] == "bf"
    assert sw["tape_strip_shadow"]["n_candidates"] == {"bf": 2, "lr": 0}
    assert "tape_evidence" not in sw
