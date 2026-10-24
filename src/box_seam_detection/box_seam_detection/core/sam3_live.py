# Provenance: moved into box_seam_detection (2026-10-12) from rgbd_captures/scripts/autolabel_v2/sam3_live.py
# at commit 6df59943f23e0d551cdc448b47ffea9e95ece20a (working tree identical to HEAD, unmodified).
# Only imports/sys.path handling changed; logic unchanged.
"""Live SAM3 inference for the v2 pipeline (parent side).

Runs genuine SAM3 inference in the SAM3-capable interpreter
(``/home/farhan/miniconda3/envs/book_grasping/bin/python``, transformers
with ``Sam3Model``/``Sam3Processor``) with prompt
``"cardboard box top face"`` (fallback ``"cardboard box"`` only when the
primary yields zero detections), bf16 autocast on CUDA, threshold 0.5.
The model stays loaded across captures: one worker subprocess per run
(see :mod:`sam3_session` / :mod:`sam3_worker`, JSON-lines protocol), not
one per image.

Offline: the worker is spawned with ``HF_HUB_OFFLINE=1`` /
``TRANSFORMERS_OFFLINE=1`` and uses ``local_files_only=True``, so nothing
can download. When weights are absent the worker fails cleanly and the
pipeline maps that to a per-capture failure with reason, never a crash or
a fabricated quad.
"""

from __future__ import annotations

import json
import os
import sys

BOOK_GRASPING_PYTHON = os.environ.get(
    "BOX_SEAM_SAM3_PYTHON",
    "/home/farhan/miniconda3/envs/book_grasping/bin/python")
PRIMARY_PROMPT = "cardboard box top face"
FALLBACK_PROMPT = "cardboard box"
LIVE_THRESHOLD = 0.5


class Sam3LiveUnavailable(RuntimeError):
    """SAM3 live inference cannot run in this environment (weights absent)."""


def run_sam3_live(image_path: str, python_bin: str = BOOK_GRASPING_PYTHON,
                  timeout_s: int = 300, threshold: float = LIVE_THRESHOLD,
                  session=None) -> dict:
    """Query live SAM3 for one image; return the worker payload dict.

    With ``session`` (a :class:`sam3_session.Sam3Worker`) the persistent
    worker is reused. Without one, a throwaway worker is started for this
    single call (used by the CLI probe and unit tests). A per-capture
    ``ok: false`` payload raises :class:`Sam3LiveUnavailable` carrying the
    worker's reason, so single-source callers keep fail-clean semantics;
    the fallback chain uses ``session.query`` directly to inspect reasons.
    """
    if session is not None:
        resp = session.query(str(image_path), timeout_s=int(timeout_s))
        if not resp.get("ok", False):
            raise Sam3LiveUnavailable(
                f"sam3_live unavailable: {resp.get('reason', 'unknown')}")
        return resp
    from .sam3_session import Sam3Worker

    with Sam3Worker(python_bin=python_bin,
                    threshold=float(threshold)) as worker:
        resp = worker.query(str(image_path), timeout_s=int(timeout_s))
    if not resp.get("ok", False):
        raise Sam3LiveUnavailable(
            f"sam3_live unavailable: {resp.get('reason', 'unknown')}")
    return resp


def main(argv=None) -> int:  # CLI probe: python3 -m scripts.autolabel_v2.sam3_live --image IMG
    import argparse

    ap = argparse.ArgumentParser(description="Probe live SAM3 availability (no downloads)")
    ap.add_argument("--image", required=True)
    ap.add_argument("--python-bin", default=BOOK_GRASPING_PYTHON)
    args = ap.parse_args(argv)
    try:
        payload = run_sam3_live(args.image, python_bin=args.python_bin)
    except Sam3LiveUnavailable as exc:
        print(f"SAM3_LIVE_UNAVAILABLE: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
