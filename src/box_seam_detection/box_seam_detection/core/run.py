# Provenance: moved into box_seam_detection (2026-10-12) from rgbd_captures/scripts/autolabel_v2/run.py
# at commit 6df59943f23e0d551cdc448b47ffea9e95ece20a (working tree identical to HEAD, unmodified).
# Only imports/sys.path handling changed; logic unchanged.
"""CLI runner: per-capture v5 JSON + overlay PNG + summary.json.

Usage:
  python3 -m box_seam_detection.core.run --split-key pilot_25 \
      --source auto --run-name pilot25_v2
  python3 -m box_seam_detection.core.run --ids ID [ID ...] --source depth_only \
      --run-name debug1
  python3 -m box_seam_detection.core.run --rename-corners-from \
      <out_root>/all853_v11 --run-name all853_v12   # names only

Outputs under /home/farhan/Igus_rebel/outputs/rgbd_captures/autolabels_v5/:
  <run_name>/<capture_id>.json, overlays/<capture_id>.png, summary.json.
One capture never crashes the run: exceptions become failure records.

With --source auto (or sam3_live) one persistent SAM3 worker subprocess
serves the whole run; its checkpoint sha256 (of model.safetensors) is
computed once and cached in the summary.
"""

from __future__ import annotations

import os as _blas_os  # noqa: E402  (BLAS pin must precede the first numpy import)
if _blas_os.environ.get("AUTOLABEL_V2_BLAS_THREADS", "").strip().lower() != "default":
    if "OPENBLAS_NUM_THREADS" not in _blas_os.environ:   # RANSAC gemv: 7-8x faster
        _blas_os.environ["OPENBLAS_NUM_THREADS"] = "1"
        _blas_os.environ["AUTOLABEL_V2_BLAS_PINNED"] = "1"  # ours, not the user's

import argparse
import hashlib
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

# External review data (the split file stays in the rgbd_captures data repo).
SPLIT_PATH = Path(os.environ.get(
    "AUTOLABEL_V2_SPLIT_PATH",
    "/home/farhan/Downloads/rgbd_captures/reviewed/split_v5.json"))

DATA_ROOT = Path("/home/farhan/Igus_rebel/outputs/rgbd_captures")
OUT_ROOT = DATA_ROOT / "autolabels_v5"

from .config import default_config  # noqa: E402
from .overlay import write_overlay  # noqa: E402
from .pipeline import (  # noqa: E402
    _fail_record, compute_seam_stage, process_capture)


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description="v2 autolabel runner (box_autolabel/v5)")
    src = ap.add_mutually_exclusive_group(required=False)
    src.add_argument("--split-key", default=None)
    src.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--source", default="auto",
                    choices=["auto", "sam3_archived", "depth_only", "sam3_live"])
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--out-root", default=str(OUT_ROOT))
    ap.add_argument("--reseam-from", default=None,
                    help="existing run dir: recompute only the seam stage "
                    "+ overlay + summary from stored corners (no SAM3)")
    ap.add_argument("--recheck-corners", action="store_true",
                    help="with --reseam-from: also recompute the corner "
                    "quality gates from stored corners + image + PCD "
                    "(no SAM3) and persist them in checks.corner_flags, "
                    "setting needs_review on flag (never reject)")
    ap.add_argument("--corner-fold", action="store_true",
                    help="enable corner_fold (fold_edge side-face model). "
                    "With --reseam-from: relocate the stored corners to the "
                    "fold first (top_face corners/xyz + provenance."
                    "corner_fold), then reseam (+ recheck); other fields "
                    "are kept")
    ap.add_argument("--no-corner-fold", action="store_true",
                    help="disable corner_fold (on by default since "
                    "2026-09-27); with --reseam-from keeps stored corners "
                    "byte-identical")
    ap.add_argument("--corner-fold-hidden", default=None,
                    choices=["keep", "shoulder"],
                    help="override corner_fold.hidden_mode")
    ap.add_argument("--no-view-shift", action="store_true",
                    help="pre-v10 seam confirmation: disable "
                    "seam_estimator.view_shift (confirm the observed "
                    "gap/darkline against the plain prior) AND the darkline "
                    "per-shoulder gate (implies --no-shoulder-gate); "
                    "reproduces all853_v8 seam results")
    ap.add_argument("--rename-corners-from", default=None,
                    help="existing run dir: rewrite ONLY the corner names by "
                    "the long_side_back_v1 rule (corner_names.py) - no "
                    "geometry recomputed; permutes corners/xyz/flags/fold "
                    "per-edge data, swaps the seam axis label when needed, "
                    "stamps provenance.corner_names, redraws overlays")
    ap.add_argument("--no-overlay", action="store_true",
                    help="with --rename-corners-from: skip overlay redraw "
                    "(overlay field set to null)")
    ap.add_argument("--no-corner-names", action="store_true",
                    help="disable the corner naming rule (pre-v12 image "
                    "ordering) for fresh / reseam runs")
    ap.add_argument("--no-shoulder-gate", action="store_true",
                    help="set seam_estimator.darkline_min_shoulder = 0 "
                    "(darkline valley need not be darker than EACH "
                    "shoulder; pre-v10 behaviour)")
    return ap.parse_args(argv)

def _resolve_ids(args) -> list[str]:
    if args.ids:
        return list(args.ids)
    doc = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    node = doc.get(args.split_key)
    if node is None:
        raise SystemExit(f"unknown split key {args.split_key!r}")
    if isinstance(node, dict):
        ids: list[str] = []
        for burst in node.get("bursts", []):
            ids.extend(burst.get("captures", []))
        return ids
    return list(node)


def _paths_for(capture_id: str) -> dict:
    return {
        "image": str(DATA_ROOT / "rgb" / f"{capture_id}.png"),
        "pcd": str(DATA_ROOT / "pointcloud" / f"{capture_id}.pcd"),
        "manifest": str(DATA_ROOT / "captures" / f"{capture_id}.json"),
    }


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", suffix=".json",
                               dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        os.replace(tmp, str(path))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _checkpoint_sha256(session) -> str | None:
    """sha256 of the worker's model.safetensors, computed once per run."""
    rev = getattr(session, "model_revision", None)
    if session is not None and getattr(session, "checkpoint_sha256", None):
        return session.checkpoint_sha256
    if not rev:
        return None
    fp = Path(str(rev)) / "model.safetensors"
    if not fp.is_file():
        return None
    h = hashlib.sha256()
    with open(fp, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    digest = h.hexdigest()
    if session is not None:
        session.checkpoint_sha256 = digest
    return digest


def _null_record(cid: str, paths: dict, cfg: dict, why: str) -> dict:
    """Runner-level failure (missing input / exception) via the shared
    pipeline builder: null corners, confidence 0, needs_review true."""
    return _fail_record(cid, paths, cfg, "runner_failed", [why], "unknown",
                        image_size=[0, 0], attempts=[])


def _reseam_ids(src_dir: Path) -> list[str]:
    import re as _re

    pat = _re.compile(r"^frame_\d{6}_\d{8}_\d{6}_\d{6}$")
    return sorted(p.stem for p in src_dir.glob("*.json")
                  if p.is_file() and pat.match(p.stem))


def recheck_record(src_rec: dict, paths: dict, cfg: dict,
                   reseam_from: str) -> tuple[dict, dict]:
    """Reseam + corner recheck from stored corners + image + PCD (no SAM3).

    Runs ``reseam_record`` (confirm-mode seam from stored corners), then
    recomputes the corner quality gates from stored corners: coarse from
    the accepted attempt in ``provenance.attempts``, refined = stored
    ``top_face.corners_px``, gray from the image, depth winners from the
    PCD + stored plane via ``_corner_depth_winners``. Corner flags persist
    in ``checks.corner_flags``; any flag sets ``checks.needs_review``
    (never rejects). ``provenance.recheck`` stamps
    {from, code_git_rev, config}. Returns ``(record, debug)``.
    """
    import cv2 as _cv2
    import numpy as _np

    from . import corner_checks as _cc
    from .pipeline import _code_git_rev, _corner_depth_winners

    rec, debug = reseam_record(src_rec, paths, cfg, reseam_from)
    bgr = debug.get("bgr")
    if bgr is None:
        return rec, debug
    named = ((rec.get("top_face") or {}).get("corners_px"))
    if not named or not all(named.get(k) for k in
                            ("back_left", "back_right",
                             "front_right", "front_left")):
        return rec, debug
    quad = _np.array(
        [[named[k][0], named[k][1]] for k in
         ("back_left", "back_right", "front_right", "front_left")],
        dtype=_np.float64)
    H, W = bgr.shape[:2]
    coarse = None
    for entry in ((rec.get("provenance") or {}).get("attempts") or []):
        if entry.get("status") == "accepted" and entry.get("coarse"):
            try:
                coarse = _np.asarray(entry["coarse"],
                                     dtype=_np.float64).reshape(4, 2)
                break
            except (TypeError, ValueError):
                coarse = None
    report: dict = {"corners": {}, "any_flag": False}
    if coarse is not None:
        try:
            gray = _cv2.cvtColor(bgr, _cv2.COLOR_BGR2GRAY)
        except _cv2.error:
            gray = None
        if gray is not None:
            plane = None
            try:
                pl = ((rec.get("top_face") or {}).get("plane"))
                if pl and pl.get("normal") is not None \
                        and pl.get("offset_m") is not None:
                    from types import SimpleNamespace as _NS
                    plane = _NS(normal=_np.asarray(
                        pl["normal"], dtype=_np.float64),
                        d=float(pl["offset_m"]))
            except (TypeError, ValueError):
                plane = None
            manifest = None
            try:
                manifest = __import__("json").loads(
                    open(paths["manifest"], encoding="utf-8").read())
            except (OSError, ValueError):
                manifest = None
            uv_win, res_win = _corner_depth_winners(
                manifest, plane, W, H)
            try:
                report = _cc.check_corners(
                    _np.asarray(gray, dtype=_np.float64), coarse, quad,
                    [W, H], uv_win=uv_win, res_win=res_win,
                    refine_cfg=cfg.get("refine"),
                    corner_cfg=(cfg.get("corner_checks") or {}))
            except Exception:  # noqa: BLE001 — gates never break reseam
                report = {"corners": {}, "any_flag": False,
                          "error": "corner_check_failed"}
    else:
        report = {"corners": {}, "any_flag": False,
                  "error": "no_accepted_coarse"}
    rec["checks"]["corner_flags"] = report
    if report.get("any_flag"):
        rec["checks"]["needs_review"] = True
        reasons = list(rec["checks"].get("reasons") or [])
        flagged = sorted(n for n, c in (report.get("corners") or {}).items()
                         if (c.get("flags") if isinstance(c, dict) else []))
        reasons.append(f"corner_flags({','.join(flagged)})")
        for n in flagged:
            c = report["corners"][n]
            reasons.append(f"corner_{n}({','.join(c.get('flags', []))})")
        rec["checks"]["reasons"] = reasons
    rec.setdefault("provenance", {})["recheck"] = {
        "from": str(reseam_from), "code_git_rev": _code_git_rev(),
        "config": cfg.get("corner_checks")}
    return rec, debug


def fold_record(src_rec: dict, paths: dict, cfg: dict, bgr,
                reseam_from: str) -> dict:
    """Stored corners -> fold relocation (no SAM3, no refit).

    Returns a deep copy of ``src_rec`` whose ``top_face.corners_px`` /
    ``corners_xyz_camera_m`` are the relocated corners (xyz = pixel rays on
    the stored plane), with ``provenance.corner_fold`` = {from,
    code_git_rev, config, enabled, changed, box_class, height_m,
    corners_px_before, per_edge[, reason]} and, when anything moved, a
    ``corner_fold_moved(...)`` entry in ``checks.reasons``. Every other
    field is untouched. Unchanged corners when the stored plane/xyz/
    manifest are missing (reason recorded).
    """
    import numpy as _np

    from .pipeline import (_code_git_rev, apply_corner_fold,
                                       fold_reason)

    rec = json.loads(json.dumps(src_rec))
    tf = rec.get("top_face") or {}
    named = tf["corners_px"]
    keys = ("back_left", "back_right", "front_right", "front_left")
    quad = _np.array([[named[k][0], named[k][1]] for k in keys],
                     dtype=_np.float64)
    xyz_named = tf.get("corners_xyz_camera_m")
    xyz = (_np.array([xyz_named[k] for k in keys], dtype=_np.float64)
           if xyz_named and all(xyz_named.get(k) for k in keys) else None)
    plane = tf.get("plane")
    try:
        manifest = json.loads(Path(paths["manifest"]).read_text(
            encoding="utf-8"))
    except (OSError, ValueError, KeyError, TypeError):
        manifest = None
    if plane is None or manifest is None:
        info = {"enabled": True, "changed": False, "per_edge": None,
                "reason": "no_plane" if plane is None else "no_manifest"}
    else:
        new_quad, new_xyz, info = apply_corner_fold(
            bgr, quad, manifest, plane, xyz, cfg.get("corner_fold"))
        if info.get("changed"):
            tf["corners_px"] = {
                k: [round(float(new_quad[i][0]), 1),
                    round(float(new_quad[i][1]), 1)]
                for i, k in enumerate(keys)}
            tf["corners_xyz_camera_m"] = {
                k: [round(float(new_xyz[i][j]), 4) for j in range(3)]
                for i, k in enumerate(keys)}
            fr = fold_reason(info)
            if fr:
                rec.setdefault("checks", {}).setdefault(
                    "reasons", []).append(fr)
    info = {"from": str(reseam_from), "code_git_rev": _code_git_rev(),
            "config": cfg.get("corner_fold"), **info}
    rec.setdefault("provenance", {})["corner_fold"] = info
    return rec


def reseam_record(src_rec: dict, paths: dict, cfg: dict,
                  reseam_from: str) -> tuple[dict, dict]:
    """Recompute only the seam stage from stored corners (no SAM3/depth).

    Only seam fields, seam reasons, overlay and confidence_seam change.
    Everything else is preserved byte-identical: top_face, image_size,
    mask_source/provenance (original config and code_git_rev kept) with
    the reseam stamp added under ``provenance.reseam`` = {from,
    code_git_rev, config}. The estimator gets corners_xyz from the source
    record for the width-in-mm gate. Returns ``(record, debug)``; debug
    holds ``bgr`` for the overlay writer. Raises on missing image/corners.
    """
    import cv2 as _cv2
    import numpy as _np

    from .pipeline import _code_git_rev

    image_path = (src_rec.get("image") or paths.get("image"))
    bgr = _cv2.imread(str(image_path), _cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"reseam unreadable image {image_path}")
    named = ((src_rec.get("top_face") or {}).get("corners_px"))
    if not named or not all(named.get(k) for k in
                            ("back_left", "back_right",
                             "front_right", "front_left")):
        raise ValueError("reseam needs stored corners_px")
    already_folded = bool((src_rec.get("provenance") or {})
                          .get("corner_fold"))
    fold_status = None
    if (cfg.get("corner_fold") or {}).get("enabled"):
        if already_folded:
            # never fold twice when reseaming a run that was already folded
            fold_status = "skipped_already_folded"
            src_mode = (((src_rec["provenance"]["corner_fold"].get("config")
                          or {}).get("hidden_mode")))
            if src_mode != cfg["corner_fold"].get("hidden_mode"):
                fold_status += (f"(source hidden_mode={src_mode}, "
                                f"current={cfg['corner_fold'].get('hidden_mode')})")
        else:
            src_rec = fold_record(src_rec, paths, cfg, bgr, reseam_from)
            named = src_rec["top_face"]["corners_px"]
            fold_status = "folded"
    quad = _np.array(
        [[named[k][0], named[k][1]] for k in
         ("back_left", "back_right", "front_right", "front_left")],
        dtype=_np.float64)
    src_xyz = ((src_rec.get("top_face") or {})
               .get("corners_xyz_camera_m"))
    # the record's box profile (None for records written before profiles:
    # legacy behaviour)
    src_profile = ((src_rec.get("box_profile") or {}).get("name")
                   if isinstance(src_rec.get("box_profile"), dict) else None)
    seam_dict, stage = compute_seam_stage(
        bgr, quad, cfg, corners_xyz=src_xyz, profile=src_profile)
    rec = json.loads(json.dumps(src_rec))  # deep copy via JSON
    rec["seam"] = {"status": seam_dict["status"],
                   "endpoints_px": seam_dict["endpoints_px"],
                   "axis": seam_dict["axis"],
                   "score": seam_dict["score"],
                   "source": seam_dict["source"],
                   "estimate": seam_dict["estimate"]}
    if "tape_strip_shadow" in seam_dict:
        rec["seam"]["tape_strip_shadow"] = seam_dict["tape_strip_shadow"]
    from .pipeline import compute_confidence_seam

    rec["confidence_seam"] = round(float(compute_confidence_seam(
        seam_dict["status"], seam_dict["score"])), 3)
    reasons = list((rec.get("checks") or {}).get("reasons") or [])
    reasons.extend(stage["reasons"])
    rec["checks"]["reasons"] = reasons
    prov = rec.setdefault("provenance", {})
    prov["reseam"] = {"from": str(reseam_from),
                      "code_git_rev": _code_git_rev(),
                      "config": cfg.get("seam_estimator"),
                      "corner_fold": fold_status}
    rec["overlay"] = None
    debug = {"bgr": bgr, "estimate": seam_dict["estimate"],
             "seam": (seam_dict["endpoints_px"] and
                      [tuple(seam_dict["endpoints_px"]["seam_left"]),
                       tuple(seam_dict["endpoints_px"]["seam_right"])])}
    # final corner names by rule (names only; recheck then pairs the
    # renamed accepted coarse with the renamed corners)
    from .pipeline import apply_corner_names

    rec = apply_corner_names(rec, _read_manifest(paths), cfg, debug,
                             source=str(reseam_from))
    return rec, debug


def _read_manifest(paths: dict):
    try:
        return json.loads(Path(paths["manifest"]).read_text(
            encoding="utf-8"))
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _corner_multiset(rec: dict):
    tf = rec.get("top_face") or {}
    px = tf.get("corners_px")
    xyz = tf.get("corners_xyz_camera_m")
    return (sorted(tuple(v) for v in px.values()) if isinstance(px, dict)
            else None,
            sorted(tuple(v) for v in xyz.values()) if isinstance(xyz, dict)
            else None)


def _main_rename(args, out_dir: Path) -> int:
    """Rename-only mode: corner names by rule, geometry untouched."""
    import time as _time

    from .corner_names import rename_record
    from .pipeline import _code_git_rev

    t0 = _time.monotonic()
    src_dir = Path(args.rename_corners_from)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty {out_dir}")
    if not src_dir.is_dir():
        raise SystemExit(f"rename source dir missing: {src_dir}")
    if out_dir.resolve() == src_dir.resolve():
        raise SystemExit("rename output must differ from its source")
    cids = _reseam_ids(src_dir)
    if args.ids:
        want = set(args.ids)
        cids = [c for c in cids if c in want]
    rev = _code_git_rev()
    n_renamed = n_swapped = 0
    by_class: dict[str, dict[str, int]] = {}
    by_date: dict[str, dict[str, int]] = {}
    flags: dict[str, int] = {}
    methods: dict[str, int] = {}
    mismatches: list[dict] = []
    failures: list[dict] = []
    renamed_ids: list[str] = []
    for cid in cids:
        try:
            src_rec = json.loads((src_dir / f"{cid}.json").read_text(
                encoding="utf-8"))
        except (OSError, ValueError) as exc:
            failures.append({"capture_id": cid,
                             "reason": f"unreadable_source({exc})"})
            continue
        paths = _paths_for(cid)
        rec, info = rename_record(src_rec, _read_manifest(paths),
                                  source=str(src_dir), code_git_rev=rev)
        # geometry must be byte-identical up to names
        if (_corner_multiset(rec) != _corner_multiset(src_rec)
                or (rec.get("seam") or {}).get("endpoints_px")
                != (src_rec.get("seam") or {}).get("endpoints_px")):
            mismatches.append({"capture_id": cid})
        cls = _box_class_of(src_rec)
        date = cid.split("_")[2] if cid.count("_") >= 3 else "?"
        key = "renamed" if info.get("renamed") else (
            "kept" if info.get("reason") != "no_corners" else "no_corners")
        by_class.setdefault(cls, {}).setdefault(key, 0)
        by_class[cls][key] += 1
        by_date.setdefault(date, {}).setdefault(key, 0)
        by_date[date][key] += 1
        if info.get("renamed"):
            n_renamed += 1
            renamed_ids.append(cid)
        if info.get("axis_swapped"):
            n_swapped += 1
        for f in info.get("flags") or []:
            flags[f] = flags.get(f, 0) + 1
        if info.get("method"):
            methods[info["method"]] = methods.get(info["method"], 0) + 1
        rec["overlay"] = None
        if not args.no_overlay and (rec.get("top_face") or {}).get(
                "corners_px"):
            import cv2 as _cv2

            bgr = _cv2.imread(str(rec.get("image") or paths["image"]),
                              _cv2.IMREAD_COLOR)
            if bgr is not None:
                seam = rec.get("seam") or {}
                eps = seam.get("endpoints_px")
                debug = {"estimate": seam.get("estimate"),
                         "seam": (eps and [tuple(eps["seam_left"]),
                                           tuple(eps["seam_right"])])}
                rec["overlay"] = write_overlay(
                    out_dir / "overlays" / f"{cid}.png", bgr, rec, debug)
        _atomic_write_json(out_dir / f"{cid}.json", rec)
    summary = {"run_name": args.run_name, "source": "rename_corners",
               "rename_corners_from": str(src_dir),
               "rule": "long_side_back_v1", "code_git_rev": rev,
               "n_captures": len(cids), "n_renamed": n_renamed,
               "n_axis_swapped": n_swapped,
               "by_box_class": by_class, "by_date": by_date,
               "flags": flags, "methods": methods,
               "n_geometry_mismatch": len(mismatches),
               "geometry_mismatches": mismatches,
               "renamed_ids": renamed_ids,
               "elapsed_s": round(_time.monotonic() - t0, 1),
               "failures": failures}
    _atomic_write_json(out_dir / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items()
                      if k != "renamed_ids"}, indent=2))
    return 0 if not mismatches and not failures else 1



def main(argv=None) -> int:
    args = _parse_args(argv)
    cfg = default_config()
    if args.corner_fold and args.no_corner_fold:
        raise SystemExit("--corner-fold and --no-corner-fold conflict")
    if args.corner_fold:
        cfg["corner_fold"]["enabled"] = True
    if args.no_corner_fold:
        cfg["corner_fold"]["enabled"] = False
    if (cfg["corner_fold"]["enabled"] and args.reseam_from
            and not args.recheck_corners and not args.rename_corners_from):
        # corner flags must describe the relocated corners
        raise SystemExit("corner_fold is enabled (default): with "
                         "--reseam-from add --recheck-corners, or pass "
                         "--no-corner-fold to keep stored corners")
    if args.corner_fold_hidden:
        cfg["corner_fold"]["hidden_mode"] = args.corner_fold_hidden
    apply_seam_cli_overrides(cfg, args)
    if args.no_corner_names:
        cfg["corner_names"]["enabled"] = False
    out_dir = Path(args.out_root) / args.run_name
    if args.rename_corners_from:
        if args.reseam_from:
            raise SystemExit("--rename-corners-from and --reseam-from "
                             "conflict")
        return _main_rename(args, out_dir)
    if args.reseam_from:
        return _main_reseam(args, cfg, out_dir)
    ids = _resolve_ids(args)
    records: dict[str, dict] = {}
    failures: list[dict] = []
    session = None
    if args.source in ("auto", "sam3_live"):
        from .sam3_session import Sam3Worker

        # Checkpoint identity is known before the loop (NIT: no lazy
        # per-capture stamping): hash model.safetensors once from the
        # worker's revision, store on the session, and let every live
        # provenance entry read it.
        try:
            session = Sam3Worker(
                stderr_log=str(out_dir / "sam3_worker.stderr.log")).start()
            digest = _checkpoint_sha256(session)
            session.checkpoint_sha256 = digest
            cfg.setdefault("sam3", {})["checkpoint_sha256"] = digest
        except Exception as exc:  # noqa: BLE001 — live degrades to fail-clean
            print(f"sam3 worker unavailable, live attempts will fail clean: "
                  f"{exc}", file=sys.stderr)
            session = None
    try:
        for cid in ids:
            paths = _paths_for(cid)
            missing = [k for k, p in paths.items() if not Path(p).is_file()]
            if missing:
                rec = _null_record(
                    cid, paths, cfg, f"missing_input:{','.join(missing)}")
                records[cid] = rec
                _atomic_write_json(out_dir / f"{cid}.json", rec)
                failures.append({"capture_id": cid,
                                 "reason": f"missing_input:{','.join(missing)}"})
                continue
            try:
                rec, debug = process_capture(cid, paths, cfg,
                                             source=args.source,
                                             session=session)
                bgr = debug.get("bgr")
                if bgr is not None:
                    rec["overlay"] = write_overlay(
                        out_dir / "overlays" / f"{cid}.png", bgr, rec, debug)
                records[cid] = rec
                _atomic_write_json(out_dir / f"{cid}.json", rec)
                if rec["top_face"]["corners_px"] is None:
                    failures.append({"capture_id": cid,
                                     "reason": ";".join(
                                         rec["checks"]["reasons"])})
            except Exception as exc:  # noqa: BLE001 — per-capture isolation
                why = f"runner_exception({type(exc).__name__}:{exc})"
                rec = _null_record(cid, paths, cfg, why)
                rec["checks"]["reasons"].append(traceback.format_exc(limit=3))
                records[cid] = rec
                _atomic_write_json(out_dir / f"{cid}.json", rec)
                failures.append({"capture_id": cid, "reason": why})
    finally:
        if session is not None:
            session.close()
    by_source: dict[str, int] = {}
    by_seam: dict[str, int] = {}
    by_topo: dict[str, int] = {"quad_ok": 0, "quad_null": 0,
                               "wrong_face": 0}
    hist = {"0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0,
            "0.6-0.8": 0, "0.8-1.0": 0}
    for rec in records.values():
        src = (rec.get("provenance") or {}).get("mask_source", "?")
        by_source[src] = by_source.get(src, 0) + 1
        st = (rec.get("seam") or {}).get("status", "?")
        by_seam[st] = by_seam.get(st, 0) + 1
        if (rec.get("top_face") or {}).get("corners_px"):
            by_topo["quad_ok"] += 1
        else:
            by_topo["quad_null"] += 1
        if (rec.get("checks") or {}).get("wrong_face_suspect"):
            by_topo["wrong_face"] += 1
        c = float(rec.get("confidence", 0.0))
        if c < 0.2:
            hist["0.0-0.2"] += 1
        elif c < 0.4:
            hist["0.2-0.4"] += 1
        elif c < 0.6:
            hist["0.4-0.6"] += 1
        elif c < 0.8:
            hist["0.6-0.8"] += 1
        else:
            hist["0.8-1.0"] += 1
    by_seam_source: dict[str, int] = {}
    for rec in records.values():
        s = ((rec.get("seam") or {}).get("source")) or "null"
        by_seam_source[s] = by_seam_source.get(s, 0) + 1
    summary = {"run_name": args.run_name, "source": args.source,
               "n_captures": len(ids), "by_source": by_source,
               "by_seam_status": by_seam, "by_seam_source": by_seam_source,
               "topology": by_topo,
               "confidence_histogram": hist, "failures": failures,
               "sam3_checkpoint_sha256": cfg.get("sam3", {}).get(
                   "checkpoint_sha256") if isinstance(cfg.get("sam3"), dict)
               else None}
    _atomic_write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


def apply_seam_cli_overrides(cfg: dict, args) -> dict:
    """Apply the seam CLI switches to ``cfg`` in place (and return it).

    ``--no-view-shift`` restores the pre-v10 confirmation exactly: view
    shift off AND darkline per-shoulder gate off (it implies
    ``--no-shoulder-gate``); ``--no-shoulder-gate`` alone only zeroes
    ``darkline_min_shoulder``.
    """
    se = cfg["seam_estimator"]
    if getattr(args, "no_view_shift", False):
        se["view_shift"]["enabled"] = False
        se["darkline_min_shoulder"] = 0.0
    if getattr(args, "no_shoulder_gate", False):
        se["darkline_min_shoulder"] = 0.0
    return cfg


def _box_class_of(rec: dict) -> str:
    """'big' / 'small' from the stored top-face xyz (long side > 150 mm),
    'unknown' without usable corners (summary bookkeeping only)."""
    from .fold_edge import box_class_from_xyz
    xyz = ((rec.get("top_face") or {}).get("corners_xyz_camera_m"))
    if not isinstance(xyz, dict):
        return "unknown"
    try:
        cls = box_class_from_xyz(
            [xyz[k] for k in ("back_left", "back_right", "front_right",
                              "front_left")])
    except (KeyError, TypeError):
        cls = None
    return cls or "unknown"


def _main_reseam(args, cfg, out_dir: Path) -> int:
    """Reseam mode: recompute seams from stored corners, no SAM3/depth."""
    import time as _time

    t0 = _time.monotonic()
    src_dir = Path(args.reseam_from)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(
            f"refusing to overwrite non-empty {out_dir}")
    if not src_dir.is_dir():
        raise SystemExit(f"reseam source dir missing: {src_dir}")
    cids = _reseam_ids(src_dir)
    if args.ids:
        want = set(args.ids)
        cids = [c for c in cids if c in want]
    records: dict[str, dict] = {}
    failures: list[dict] = []
    n_null_corners = 0
    for cid in cids:
        try:
            src_rec = json.loads((src_dir / f"{cid}.json").read_text(
                encoding="utf-8"))
        except (OSError, ValueError) as exc:
            rec = _null_record(cid, _paths_for(cid), cfg,
                               f"reseam_unreadable_source({exc})")
            records[cid] = rec
            _atomic_write_json(out_dir / f"{cid}.json", rec)
            failures.append({"capture_id": cid,
                             "reason": f"reseam_unreadable_source({exc})"})
            continue
        if not ((src_rec.get("top_face") or {}).get("corners_px")):
            n_null_corners += 1
            # Preserve the source record: same metadata, seam stays as
            # stored, reseam stamp under provenance.reseam. With
            # --recheck-corners also stamp provenance.recheck with the
            # no-quad outcome (nothing to recompute); top_face unchanged.
            from .pipeline import _code_git_rev as _rev

            rec = json.loads(json.dumps(src_rec))
            rec.setdefault("provenance", {})["reseam"] = {
                "from": str(src_dir), "code_git_rev": _rev(),
                "config": cfg.get("seam_estimator")}
            if args.recheck_corners:
                rec["checks"]["corner_flags"] = {
                    "corners": {}, "any_flag": False, "note": "no_quad"}
                rec.setdefault("provenance", {})["recheck"] = {
                    "from": str(src_dir), "code_git_rev": _rev(),
                    "config": cfg.get("corner_checks")}
            rec["overlay"] = None
            records[cid] = rec
            _atomic_write_json(out_dir / f"{cid}.json", rec)
            failures.append({"capture_id": cid,
                             "reason": "reseam_null_corners"})
            continue
        try:
            if args.recheck_corners:
                rec, debug = recheck_record(src_rec, _paths_for(cid), cfg,
                                            str(src_dir))
            else:
                rec, debug = reseam_record(src_rec, _paths_for(cid), cfg,
                                           str(src_dir))
            bgr = debug.get("bgr")
            if bgr is not None:
                rec["overlay"] = write_overlay(
                    out_dir / "overlays" / f"{cid}.png", bgr, rec, debug)
            records[cid] = rec
            _atomic_write_json(out_dir / f"{cid}.json", rec)
            if rec["top_face"]["corners_px"] is None:
                failures.append({"capture_id": cid,
                                 "reason": ";".join(
                                     rec["checks"]["reasons"])})
        except Exception as exc:  # noqa: BLE001 — per-capture isolation
            # Keep the source record (top_face, image_size, provenance);
            # never _null_record: only stamp the error and leave the seam
            # status as stored, or "ambiguous" when the source had no
            # usable seam. Only seam reasons grow.
            why = f"reseam_exception({type(exc).__name__}:{exc})"
            from .pipeline import _code_git_rev as _rev2

            rec = json.loads(json.dumps(src_rec))
            rec.setdefault("provenance", {})["reseam"] = {
                "from": str(src_dir), "code_git_rev": _rev2(),
                "config": cfg.get("seam_estimator"), "error": why}
            seam = rec.get("seam")
            if not isinstance(seam, dict) or seam.get("status") not in (
                    "ok", "estimated", "ambiguous", "invisible"):
                rec["seam"] = {"status": "ambiguous", "endpoints_px": None,
                               "axis": None, "score": None, "source": None,
                               "estimate": None}
            reasons = list((rec.get("checks") or {}).get("reasons") or [])
            reasons.append(why)
            rec.setdefault("checks", {})["reasons"] = reasons
            records[cid] = rec
            _atomic_write_json(out_dir / f"{cid}.json", rec)
            failures.append({"capture_id": cid, "reason": why})
    by_seam: dict[str, int] = {}
    by_seam_source: dict[str, int] = {}
    by_class_seam: dict[str, dict[str, int]] = {}
    n_needs_review = 0
    n_corner_flagged = 0
    for rec in records.values():
        st = (rec.get("seam") or {}).get("status", "?")
        by_seam[st] = by_seam.get(st, 0) + 1
        cls = _box_class_of(rec)
        by_class_seam.setdefault(cls, {})
        by_class_seam[cls][st] = by_class_seam[cls].get(st, 0) + 1
        s = ((rec.get("seam") or {}).get("source")) or "null"
        by_seam_source[s] = by_seam_source.get(s, 0) + 1
        if (rec.get("checks") or {}).get("needs_review"):
            n_needs_review += 1
        cf = ((rec.get("checks") or {}).get("corner_flags") or {})
        if cf.get("any_flag"):
            n_corner_flagged += 1
    fold_counts: dict[str, dict[str, int]] = {}
    n_fold_changed = 0
    n_fold_skipped = 0
    for rec in records.values():
        if str(((rec.get("provenance") or {}).get("reseam") or {})
               .get("corner_fold") or "").startswith("skipped_already_folded"):
            n_fold_skipped += 1
        cf = (rec.get("provenance") or {}).get("corner_fold") or {}
        if cf.get("changed"):
            n_fold_changed += 1
        for e, r in (cf.get("per_edge") or {}).items():
            key = r.get("method", "?") + (
                f":{r['reason']}" if r.get("reason") else "")
            fold_counts.setdefault(e, {})
            fold_counts[e][key] = fold_counts[e].get(key, 0) + 1
    summary = {"run_name": args.run_name, "source": "reseam",
               "reseam_from": str(src_dir), "n_captures": len(cids),
               "recheck_corners": bool(args.recheck_corners),
               **({"corner_fold_config": cfg["corner_fold"],
                   "n_fold_changed": n_fold_changed,
                   "n_fold_skipped_already_folded": n_fold_skipped,
                   "fold_per_edge": fold_counts}
                  if cfg["corner_fold"].get("enabled") else {}),
               "by_seam_status": by_seam, "by_seam_source": by_seam_source,
               "by_box_class_seam_status": by_class_seam,
               "n_needs_review": n_needs_review,
               "n_corner_flagged": n_corner_flagged,
               "n_null_corners": n_null_corners,
               "elapsed_s": round(_time.monotonic() - t0, 1),
               "estimator_config": cfg.get("seam_estimator"),
               "corner_config": cfg.get("corner_checks"),
               "failures": failures}
    _atomic_write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
