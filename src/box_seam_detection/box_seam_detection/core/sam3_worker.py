# Provenance: moved into box_seam_detection (2026-10-12) from rgbd_captures/scripts/autolabel_v2/sam3_worker.py
# at commit 6df59943f23e0d551cdc448b47ffea9e95ece20a (working tree identical to HEAD, unmodified).
# Only imports/sys.path handling changed; logic unchanged.
"""Persistent SAM3 GPU worker (runs ONLY in the book_grasping python).

JSON-lines protocol over stdin/stdout: the parent sends one JSON object per
line ``{"id": ..., "image": path, "prompts": [...], "threshold": float}``
and receives one JSON object per line with either the accepted coarse quad
(``ok: true``) or a clean failure (``ok: false`` + reason). The model stays
loaded across captures: one subprocess per run, not one per image.

Inference contract (matches the archived benchmark):
- prompts tried in order: "cardboard box top face", then "cardboard box";
  the fallback prompt is used ONLY when the primary yields zero detections.
- bf16 autocast on CUDA (like the archived benchmark), threshold 0.5.
- Winner = argmax concept score among that prompt's detections (same rule
  the archived rows follow: selected_mask.box/score selects the mask).
- ``alternates`` (optional key): the other sanity-passing detections, in
  descending score order, each with its named quad; the parent tries them
  only after the winner fails downstream checks.
- Mask -> coarse quad: largest contour -> convex hull -> best-IoU
  approxPolyDP quad over eps [0.005..0.05] (convex, distinct corners),
  mapped to [BL, BR, FR, FL] via the annotator u-split convention.

Offline: HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE are forced on here as well,
so the worker can never download even if the parent forgets the env.
"""
from __future__ import annotations

import json
import math
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

PRIMARY_PROMPT = "cardboard box top face"
FALLBACK_PROMPT = "cardboard box"
APPROX_EPS = (0.005, 0.01, 0.015, 0.02, 0.03, 0.05)
MODEL_ID = "facebook/sam3"


def _fail(req_id, reason: str) -> dict:
    return {"ok": False, "id": req_id, "reason": str(reason)}


K4_MIN_IOU = 0.80
K4_MAX_EXT_DIAM = 0.5  # extension intersection must lie within this x hull diameter


def _hull_min_area_k4(hull_pts):
    """Reduce a convex polygon to 4 vertices by minimum-added-area removal.

    Repeatedly removes the edge (b, c) whose two neighbouring edges, when
    extended, meet at x (replacing b and c by x) adding the least area
    (triangle b-x-c). A cut/crushed corner thus returns to where the two
    adjacent edges meet. Invalid / parallel / far (> K4_MAX_EXT_DIAM x hull
    diameter) intersections are skipped. Returns a 4x2 float array or None.
    """
    import numpy as np

    P = [np.asarray(p, dtype=float) for p in hull_pts]
    if len(P) < 4:
        return None
    diam = max(float(np.linalg.norm(a - b)) for a in P for b in P)
    if not diam > 0:
        return None
    while len(P) > 4:
        n = len(P)
        best = None
        for i in range(n):
            a, b = P[i - 1], P[i]
            c, d = P[(i + 1) % n], P[(i + 2) % n]
            r, t = b - a, c - d
            den = r[0] * t[1] - r[1] * t[0]
            if abs(den) < 1e-9:
                continue
            w = d - a
            u = (w[0] * t[1] - w[1] * t[0]) / den
            v = (w[0] * r[1] - w[1] * r[0]) / den
            if u <= 1.0 or v <= 1.0:  # must meet beyond both b and c
                continue
            x = a + u * r
            if float(np.linalg.norm(x - b)) > K4_MAX_EXT_DIAM * diam:
                continue
            add = 0.5 * abs((x[0] - b[0]) * (c[1] - b[1])
                            - (x[1] - b[1]) * (c[0] - b[0]))
            if best is None or add < best[0]:
                best = (add, i, x)
        if best is None:
            return None
        _, i, x = best
        P = [x] + P[i + 2:] + P[:i] if i + 2 <= n else [x] + P[(i + 2) % n:i]
    return np.array(P, dtype=float)


def _mask_to_quad(mask, H: int, W: int):
    """Largest-contour hull -> best-IoU 4-gon. Returns (quad4x2|None, iou)."""
    q, iou, _m = _mask_to_quad_ex(mask, H, W)
    return q, iou


def _mask_to_quad_ex(mask, H: int, W: int):
    """As ``_mask_to_quad`` plus the method: (quad|None, iou, method|None).

    method is "approx_poly" (approxPolyDP found a convex 4-gon) or
    "min_area_k4" (fallback for a cut/crushed corner).
    """
    import cv2
    import numpy as np

    m = (np.asarray(mask) > 0).astype("uint8")
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, 0.0, None
    hull = cv2.convexHull(max(cnts, key=cv2.contourArea))
    peri = float(cv2.arcLength(hull, True))
    if not peri > 0:
        return None, 0.0, None
    best = None
    for eps in APPROX_EPS:
        ap = cv2.approxPolyDP(hull, float(eps) * peri, True)
        if len(ap) != 4 or not bool(cv2.isContourConvex(ap)):
            continue
        q = ap.reshape(4, 2).astype(float)
        if bool((np.linalg.norm(q - np.roll(q, -1, axis=0), axis=1) < 2.0).any()):
            continue
        qm = np.zeros((H, W), "uint8")
        cv2.fillConvexPoly(qm, ap.reshape(4, 2), 1)
        inter = float(((qm > 0) & (m > 0)).sum())
        union = float(((qm > 0) | (m > 0)).sum())
        iou = float(inter / union) if union > 0 else 0.0
        if best is None or iou > best[1]:
            best = (q, iou, float(eps))
    if best is None:
        q = _hull_min_area_k4(hull.reshape(-1, 2))
        if q is None or not bool(np.isfinite(q).all()):
            return None, 0.0, None
        qi = q.astype("float32").reshape(-1, 1, 2)
        if not bool(cv2.isContourConvex(qi)):
            return None, 0.0, None
        if bool((np.linalg.norm(q - np.roll(q, -1, axis=0), axis=1)
                 <= 2.0).any()):
            return None, 0.0, None
        qm = np.zeros((H, W), "uint8")
        cv2.fillConvexPoly(qm, np.round(q).astype("int32"), 1)
        inter = float(((qm > 0) & (m > 0)).sum())
        union = float(((qm > 0) | (m > 0)).sum())
        iou = float(inter / union) if union > 0 else 0.0
        if iou < K4_MIN_IOU:
            return None, 0.0, None
        return q, iou, "min_area_k4"
    return best[0], best[1], "approx_poly"


def _to_named(quad):
    """4x2 quad -> [BL, BR, FR, FL] via annotator u-split convention."""
    import numpy as np

    q = np.asarray(quad, dtype=float).reshape(4, 2)
    by_u = sorted(q.tolist(), key=lambda p: p[0])
    left = sorted(by_u[:2], key=lambda p: p[1])
    right = sorted(by_u[2:], key=lambda p: p[1])
    return [left[0], right[0], right[1], left[1]]


def _brown_frac(bgr, mask) -> float | None:
    """Cardboard-brown share inside a binary mask (0..1, None on error)."""
    try:
        import cv2 as _cv2
        import numpy as _np
        hsv = _cv2.cvtColor(bgr, _cv2.COLOR_BGR2HSV)
        lo = _np.array([8, 40, 70], dtype=_np.uint8)
        hi = _np.array([30, 220, 240], dtype=_np.uint8)
        brown = _cv2.inRange(hsv, lo, hi) > 0
        m = _np.asarray(mask) > 0
        if not m.any():
            return None
        return float((brown & m).sum() / m.sum())
    except Exception:  # noqa: BLE001 — diagnostic only
        return None


# Sanity ranges from the 121 archived rows' selected_mask distribution:
# area p5..p95 ~ 0.003..0.054 (max 0.06), brown p5..p95 ~ 0.07..0.88.
# The floor rejects specks; the ceiling rejects whole-table masks. Small
# but brown-consistent patches (the archived frame_000003_* background
# quads: area 0.003, brown ~0.88) PASS the gate — it only removes absurd
# sizes and non-cardboard colours, never the top-vs-patch decision (that
# stays argmax + the depth cross-check downstream).
MIN_MASK_AREA_FRAC = 0.002
MAX_MASK_AREA_FRAC = 0.25
MIN_MASK_BROWN_FRAC = 0.05


def _resolve_gates(req: dict):
    """Per-request sanity-gate overrides: ``req["gates"]`` with optional
    ``min_area_frac`` / ``max_area_frac`` / ``min_brown_frac``. Missing key
    -> module defaults (identical to the pre-override behaviour). Returns
    ``(min_area, max_area, min_brown, error_or_None)``."""
    lo, hi, br = MIN_MASK_AREA_FRAC, MAX_MASK_AREA_FRAC, MIN_MASK_BROWN_FRAC
    g = req.get("gates")
    if g is None:
        return lo, hi, br, None
    if not isinstance(g, dict):
        return lo, hi, br, "gates must be an object"
    vals = {"min_area_frac": lo, "max_area_frac": hi, "min_brown_frac": br}
    for k, v in g.items():
        if k not in vals:
            return lo, hi, br, f"unknown gate key: {k}"
        if v is None:
            continue
        if isinstance(v, bool):
            return lo, hi, br, f"gate {k} must be a number, not a boolean"
        try:
            f = float(v)
        except (TypeError, ValueError):
            return lo, hi, br, f"gate {k} must be a number"
        if not math.isfinite(f):
            return lo, hi, br, f"gate {k} must be finite"
        if f < 0.0 or f > 1.0:
            return lo, hi, br, (f"gate {k} must be a fraction in [0, 1] "
                                f"(got {v!r}; a percentage is an error)")
        vals[k] = f
    if vals["min_area_frac"] > vals["max_area_frac"]:
        return lo, hi, br, "min_area_frac must be <= max_area_frac"
    return vals["min_area_frac"], vals["max_area_frac"], \
        vals["min_brown_frac"], None


def _handle_one(model, processor, device: str, req: dict) -> dict:
    import numpy as np
    import torch
    from PIL import Image

    req_id = req.get("id")
    image_path = req.get("image")
    prompts = req.get("prompts") or [PRIMARY_PROMPT, FALLBACK_PROMPT]
    min_area, max_area, min_brown, gate_err = _resolve_gates(req)
    if gate_err:
        return _fail(req_id, gate_err)
    try:
        threshold = float(req.get("threshold", 0.5))
    except (TypeError, ValueError):
        return _fail(req_id, "threshold must be a number")
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as exc:  # noqa: BLE001 — reported per capture
        return _fail(req_id, f"cannot read image: {exc}")
    W, H = image.size
    try:
        import cv2 as _cv2
        bgr = _cv2.imread(str(image_path), _cv2.IMREAD_COLOR)
    except Exception:  # noqa: BLE001 — brown frac degrades to None
        bgr = None
    prompt_used = None
    try:
        for prompt in prompts:
            inputs = processor(images=image, text=str(prompt),
                               return_tensors="pt")
            inputs = {k: (v.to(device) if hasattr(v, "to") else v)
                      for k, v in inputs.items()}
            with torch.inference_mode(), torch.autocast("cuda",
                                                        torch.bfloat16):
                outputs = model(**inputs)
            res = processor.post_process_instance_segmentation(
                outputs, threshold=threshold,
                target_sizes=[image.size[::-1]])[0]
            n = len(res["scores"])
            if n == 0:
                continue  # fallback prompt only when primary yields nothing
            prompt_used = str(prompt)
            # Scores/boxes may be bf16 under autocast: never .numpy() them
            # (numpy has no bf16); convert elementwise via float().
            scores_list = [float(s) for s in res["scores"].detach().cpu()]
            boxes_list = [[float(x) for x in row]
                          for row in res["boxes"].detach().cpu()]
            masks_np = res["masks"].detach().cpu().numpy()
            scores_np = np.asarray(scores_list, dtype=np.float64)
            boxes_np = np.asarray(boxes_list, dtype=np.float64)
            dets = []
            for di in range(n):
                m = (masks_np[di] > 0).astype("uint8")
                area = float(m.sum()) / (H * W)
                dets.append({
                    "score": round(float(scores_np[di]), 4),
                    "box_xyxy": [round(float(x), 1) for x in boxes_np[di]],
                    "area_frac": round(area, 4),
                    "brown_frac": (round(bf, 3) if (bf := _brown_frac(
                        bgr, m)) is not None else None),
                })
            order = sorted(range(n),
                           key=lambda di: float(scores_np[di]), reverse=True)
            picked = None
            for di in order:
                dd = dets[di]
                if not (min_area <= dd["area_frac"] <= max_area):
                    dd["rejected"] = "area_out_of_range"
                    continue
                if dd["brown_frac"] is not None and dd["brown_frac"] < \
                        min_brown:
                    dd["rejected"] = "brown_too_low"
                    continue
                picked = di
                break
            if picked is None:
                out = _fail(req_id, "all_detections_failed_sanity_gate")
                out["detections"] = dets
                out["prompt"] = prompt_used
                return out
            bi = picked
            score = float(scores_list[bi])
            box = [float(x) for x in boxes_list[bi]]
            quad, quad_iou, quad_method = _mask_to_quad_ex(masks_np[bi], H, W)
            if quad is None:
                out = _fail(req_id, "mask_quad_unsupported")
                out["detections"] = dets
                out["prompt"] = prompt_used
                return out
            # Alternates: every OTHER detection passing the same sanity
            # gate, descending score. ``dets`` is not mutated (the winner
            # response fields stay identical to the pre-alternate protocol).
            alternates = []
            for di in order:
                if di == bi:
                    continue
                dd = dets[di]
                if not (min_area <= dd["area_frac"] <= max_area):
                    continue
                if dd["brown_frac"] is not None and dd["brown_frac"] < \
                        min_brown:
                    continue
                aq, aiou, amethod = _mask_to_quad_ex(masks_np[di], H, W)
                if aq is None:
                    continue
                alternates.append({
                    "corners_px": _to_named(np.asarray(aq, dtype=float)),
                    "concept_score": float(scores_list[di]),
                    "box_xyxy": [float(x) for x in boxes_list[di]],
                    "mask_quad_iou": round(float(aiou), 3),
                    "quad_method": amethod,
                    "det_index": int(di)})
            return {"ok": True, "id": req_id,
                    "corners_px": _to_named(np.asarray(quad, dtype=float)),
                    "concept_score": score, "prompt": prompt_used,
                    "box_xyxy": box, "n_detections": int(n),
                    "detections": dets, "picked": int(bi),
                    "mask_quad_iou": round(float(quad_iou), 3),
                    "quad_method": quad_method,
                    "alternates": alternates,
                    "image_size": [int(W), int(H)]}
        return _fail(req_id, "no_detections_above_threshold")
    except Exception as exc:  # noqa: BLE001 — per-capture isolation
        return _fail(req_id, f"inference_failed({type(exc).__name__}:{exc})")

def load_model(note=None):
    """Load processor + SAM3 model (offline). Returns (model, processor,
    device, revision); raises on failure. Shared by the per-run worker and
    the persistent daemon (``sam3_daemon.py``) so both run identical code."""
    import torch
    from transformers import Sam3Model, Sam3Processor

    def _n(msg: str) -> None:
        if note is not None:
            note(msg)

    model_id = os.environ.get("SAM3_MODEL_ID", MODEL_ID)
    _n(f"loading {model_id} (local_files_only)")
    try:
        import huggingface_hub as _hub

        revision = _hub.snapshot_download(
            repo_id=model_id, local_files_only=True)
    except Exception as exc:
        _n(f"revision lookup failed: {exc}")
        revision = None
    processor = Sam3Processor.from_pretrained(model_id,
                                              local_files_only=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    _n(f"device={device}")
    model = Sam3Model.from_pretrained(model_id, local_files_only=True,
                                      dtype=torch.bfloat16,
                                      device_map=device)
    model.eval()
    return model, processor, device, revision


def main() -> int:
    def _note(msg: str) -> None:
        # Init notices go to STDERR (stdout is the JSON-lines response
        # channel; the parent tees stderr to <run>/sam3_worker.stderr.log).
        print(f"sam3_worker: {msg}", file=sys.stderr, flush=True)

    try:
        model, processor, device, revision = load_model(_note)
    except Exception as exc:  # noqa: BLE001 — parent reports cleanly
        _note(f"init failed: {exc}")
        print(json.dumps({"ok": False, "id": None,
                          "reason": f"worker_init_failed: {exc}"}), flush=True)
        return 1
    _note(f"ready revision={revision}")
    # Handshake line on STDOUT, consumed by the parent before any query:
    # {"type": "handshake", "model_revision": ...}.
    print(json.dumps({"type": "handshake", "model_revision": revision}),
          flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except ValueError:
            print(json.dumps(_fail(None, "bad_request_json")), flush=True)
            continue
        if not isinstance(req, dict) or req.get("id") is None:
            print(json.dumps(_fail(None, "request needs an id")), flush=True)
            continue
        resp = _handle_one(model, processor, device, req)
        resp["model_revision"] = revision
        print(json.dumps(resp), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
