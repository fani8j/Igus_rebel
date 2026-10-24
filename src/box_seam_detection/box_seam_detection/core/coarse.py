# Provenance: moved into box_seam_detection (2026-10-12) from rgbd_captures/scripts/autolabel_v2/coarse.py
# at commit 6df59943f23e0d551cdc448b47ffea9e95ece20a (working tree identical to HEAD, unmodified).
# Only imports/sys.path handling changed; logic unchanged.
"""Coarse mask/quad sources for the v2 pipeline.

Pluggable source returning (4, 2) [BL, BR, FR, FL] coarse corners plus
provenance fields:

- ``sam3_archived``: corners_px from the genuine SAM3 benchmark JSON
  (``comparisons/autoguide_bench/sam3/<id>.json``). The archived rows store
  ``corners_px`` in the same [BL, BR, FR, FL] convention that
  ``run_auto_guide_compare.build_auto_terminal_doc`` maps positionally to
  (back_left, back_right, front_right, front_left), so the mapping here is
  identical: row i -> CORNER_ORDER[i].
- ``depth_only``: quad from ``scripts/autolabel/depth_top.estimate_top``
  over the registered PCD. The depth quad comes back in
  ``order_corners`` [TL, TR, BR, BL] image convention, which is remapped
  to [BL, BR, FR, FL] via ``seam.assign_names`` (u-split then v-split per
  pair, matching the annotator convention).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import CORNER_ORDER

SAM3_ARCHIVE_DIR = Path(
    "/home/farhan/Igus_rebel/outputs/rgbd_captures/comparisons/autoguide_bench/sam3"
)


def load_sam3_archived(capture_id: str, archive_dir: str | Path | None = None):
    """Load archived SAM3 coarse corners for one capture.

    Returns ``(corners (4,2) float64 [BL,BR,FR,FL], provenance dict)`` or
    ``(None, {"reason": ...})`` when the archive row is missing/invalid.
    """
    base = Path(archive_dir) if archive_dir is not None else SAM3_ARCHIVE_DIR
    path = base / f"{capture_id}.json"
    if not path.is_file():
        return None, {"reason": f"no archived sam3 row for {capture_id}"}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return None, {"reason": f"archived sam3 row unreadable: {exc}"}
    if doc.get("model") != "sam3":
        return None, {"reason": f"archived row model is {doc.get('model')!r}, not sam3"}
    corners = doc.get("corners_px")
    try:
        arr = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    except (TypeError, ValueError):
        return None, {"reason": "archived corners_px malformed"}
    if not np.all(np.isfinite(arr)):
        return None, {"reason": "archived corners_px non-finite"}
    sel = doc.get("selected_mask") or doc.get("selected_detection") or {}
    score = sel.get("concept_score")
    try:
        score = float(score) if score is not None else None
    except (TypeError, ValueError):
        score = None
    if score is not None and not np.isfinite(score):
        score = None
    prompt = doc.get("prompt") or doc.get("prompt_primary")
    prov = {
        "mask_source": "sam3_archived",
        "sam3": {
            "checkpoint_sha256": doc.get("checkpoint_sha256"),
            "prompt": prompt if isinstance(prompt, str) else None,
            "concept_score": score,
        },
        "archive_path": str(path),
        "archive_status": doc.get("status"),
        "archive_quad_valid": doc.get("quad_valid"),
    }
    return np.ascontiguousarray(arr), prov


def depth_quad_to_named(quad_tltr: np.ndarray) -> np.ndarray:
    """Map an ``order_corners`` [TL, TR, BR, BL] quad to [BL, BR, FR, FL].

    Uses the annotator u-split convention from ``seam.assign_names``:
    left pair (small u) sorted by v -> back_left, front_left; right pair
    sorted by v -> back_right, front_right. Image back = smaller v (higher
    in the image); this matches assign_names' v-sort within each u-pair
    (top of pair = back).
    """
    q = np.asarray(quad_tltr, dtype=np.float64).reshape(4, 2)
    by_u = sorted(q.tolist(), key=lambda p: p[0])
    left = sorted(by_u[:2], key=lambda p: p[1])
    right = sorted(by_u[2:], key=lambda p: p[1])
    named = {
        "back_left": left[0],
        "front_left": left[1],
        "back_right": right[0],
        "front_right": right[1],
    }
    return np.array([named[n] for n in CORNER_ORDER], dtype=np.float64)


def archived_available(capture_id: str, archive_dir: str | Path | None = None) -> bool:
    """Whether an archived SAM3 row exists for this capture."""
    base = Path(archive_dir) if archive_dir is not None else SAM3_ARCHIVE_DIR
    return bool((base / f"{capture_id}.json").is_file())
