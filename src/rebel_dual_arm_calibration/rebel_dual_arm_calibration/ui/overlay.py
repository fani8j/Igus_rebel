"""Live image: throttled target detection plus the JPEG overlay (no ROS imports; the frame source is injected)."""
import logging
import os
import threading
import time
from typing import Callable, Dict, Optional, Tuple

import cv2
import numpy as np

GREEN, ORANGE, RED, WHITE = (0, 200, 0), (0, 140, 255), (0, 0, 255), (255, 255, 255)
MAGENTA = (255, 0, 255)           # suggested pose outline (distinct from the detection colours)
CYAN, YELLOW = (255, 220, 0), (0, 235, 255)      # approach assist: predicted touch crosshairs / the selected one
GRID = 4
DEFAULT_MIN = {"table": 16}


def default_min_corners(name: str, spec) -> int:
    if name in DEFAULT_MIN:
        return DEFAULT_MIN[name]
    return 4 if getattr(spec, "type", "") == "aruco" else 8


def coverage_counts(status: Optional[dict], size: Tuple[int, int], grid: int = GRID):
    """grid x grid counts of captured intrinsics views covering each image cell (bbox overlap; centre if no bbox)."""
    counts = np.zeros((grid, grid), dtype=int)
    if not status:
        return counts
    w, h = size
    boxes = status.get("intrinsics_bboxes") or []
    if not boxes:
        boxes = [[c[0], c[1], c[0], c[1]] for c in status.get("intrinsics_centres") or []]
    for x0, y0, x1, y1 in boxes:
        c0, c1 = int(np.clip(x0 / w * grid, 0, grid - 1e-6)), int(np.clip(x1 / w * grid, 0, grid - 1e-6))
        r0, r1 = int(np.clip(y0 / h * grid, 0, grid - 1e-6)), int(np.clip(y1 / h * grid, 0, grid - 1e-6))
        counts[r0:r1 + 1, c0:c1 + 1] += 1
    return counts


class ImagePipeline:
    """Worker thread: newest frame -> TargetDetector (<= ``max_hz``) -> cached (frame, detections).
    ``render_jpeg`` draws the overlay on that frame, so the picture and the corners always belong together."""

    def __init__(self, frame_source: Callable[[], Optional[tuple]], detector_factory: Callable[[], object],
                 min_corners_fn: Optional[Callable[[], Optional[dict]]] = None, max_hz: float = 5.0):
        self._source = frame_source          # () -> (bgr, seq, rx_monotonic) or None
        self._factory = detector_factory
        self._min_fn = min_corners_fn
        self._period = 1.0 / max(max_hz, 0.1)
        self._detector = None
        self._detector_error: Optional[str] = None
        self._lock = threading.Lock()
        self._frame = None                   # (bgr, seq, rx, dets {name: Detection}, summary {name: {n, ok}})
        self._last_seq = -1
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.log = logging.getLogger(__name__)

    # ---- worker ------------------------------------------------------------------------------
    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="detector", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        try:
            os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), 10)
        except (OSError, AttributeError):
            pass
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                self.process_once()
            except Exception:  # noqa: BLE001 - never let the worker die
                self.log.exception("detector worker")
            self._stop.wait(max(self._period - (time.monotonic() - t0), 0.02))

    def _min_corners(self, name: str) -> int:
        override = None
        try:
            override = (self._min_fn() or {}).get(name) if self._min_fn else None
        except Exception:  # noqa: BLE001
            pass
        if override is not None:
            return int(override)
        spec = getattr(self._detector, "specs", {}).get(name)
        return default_min_corners(name, spec)

    def process_once(self) -> bool:
        """Detect on the newest unseen frame.  Returns True if a frame was processed."""
        item = self._source()
        if item is None:
            return False
        bgr, seq, rx = item
        if seq == self._last_seq:
            return False
        self._last_seq = seq
        if self._detector is None and self._detector_error is None:
            try:
                self._detector = self._factory()
            except Exception as e:  # noqa: BLE001
                self._detector_error = str(e)
                self.log.error("target detector unavailable: %s", e)
        dets, summary = {}, {}
        if self._detector is not None:
            try:
                dets = self._detector.detect(bgr)
            except (cv2.error, ValueError) as e:
                self.log.warning("detect failed: %s", e)
            for name in self._detector.specs:
                n = len(dets[name].points) if name in dets else 0
                summary[name] = {"n": n, "ok": n >= self._min_corners(name)}
        with self._lock:
            self._frame = (bgr, seq, rx, dets, summary)
        return True

    # ---- readers -----------------------------------------------------------------------------
    def detections(self) -> Dict[str, dict]:
        with self._lock:
            return dict(self._frame[4]) if self._frame else {}

    def has_frame(self) -> bool:
        with self._lock:
            return self._frame is not None

    def render_jpeg(self, width: int = 960, overlay: str = "detect", status: Optional[dict] = None,
                    quality: int = 80, suggest: Optional[dict] = None, hover: Optional[dict] = None) -> Optional[bytes]:
        """``suggest``: {outline_px: [(u, v) x4 in full-resolution image pixels], label} drawn on top (any overlay mode).
        ``hover``: {points: [{id, name, uv (full-resolution pixels), selected}]} predicted touch crosshairs (approach assist)."""
        with self._lock:
            frame = self._frame
        if frame is None:
            return None
        bgr, _, _, dets, summary = frame
        h0, w0 = bgr.shape[:2]
        width = int(max(160, min(width, w0)))
        s = width / w0
        img = cv2.resize(bgr, (width, int(round(h0 * s))), interpolation=cv2.INTER_AREA) if width != w0 else bgr.copy()
        if overlay == "intrinsics":
            self._draw_grid(img, coverage_counts(status, (w0, h0)))
        if overlay in ("detect", "intrinsics"):
            self._draw_detections(img, s, dets, summary)
            if self._detector is None:
                cv2.putText(img, "detector unavailable: " + (self._detector_error or "starting"), (10, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2, cv2.LINE_AA)
        if suggest:
            self._draw_suggest(img, s, suggest.get("outline_px"), str(suggest.get("label") or "suggested"))
        if hover:
            self._draw_hover(img, s, hover.get("points"))
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        return buf.tobytes() if ok else None

    @staticmethod
    def _draw_detections(img, s, dets, summary):
        for name, det in dets.items():
            ok = summary.get(name, {}).get("ok", False)
            col = GREEN if ok else ORANGE
            uv = np.array([p[2] for p in det.points], dtype=float).reshape(-1, 2) * s
            for u, v in uv:
                cv2.circle(img, (int(round(u)), int(round(v))), 3, col, -1, cv2.LINE_AA)
            cu, cv_ = uv.mean(axis=0)
            label = f"{name} {len(det.points)}"
            cv2.putText(img, label, (int(cu) - 30, int(cv_)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, label, (int(cu) - 30, int(cv_)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)

    @staticmethod
    def _draw_suggest(img, s, outline, label):
        """Predicted board outline of a suggested pose (pixel coords of the full camera image, scaled by ``s``)."""
        try:
            pts = np.asarray(outline, dtype=float).reshape(-1, 2)
        except (TypeError, ValueError):
            return
        pts = pts[np.isfinite(pts).all(axis=1)] * s
        if len(pts) < 2:
            return
        poly = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [poly], True, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.polylines(img, [poly], True, MAGENTA, 2, cv2.LINE_AA)
        h, w = img.shape[:2]
        x = int(np.clip(pts[:, 0].min(), 4, max(w - 200, 4)))
        y = int(np.clip(pts[:, 1].min() - 8, 22, h - 6))
        cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, MAGENTA, 2, cv2.LINE_AA)

    @staticmethod
    def _draw_hover(img, s, points):
        """Predicted touch points: a crosshair with ring, the id and the name; the selected one bigger and yellow."""
        h, w = img.shape[:2]
        for pt in points or []:
            try:
                u, v = (float(c) * s for c in pt["uv"])
            except (KeyError, TypeError, ValueError):
                continue
            if not (np.isfinite(u) and np.isfinite(v)) or not (-50 <= u <= w + 50 and -50 <= v <= h + 50):
                continue
            sel = bool(pt.get("selected"))
            col, arm_len, thick = (YELLOW, 22, 2) if sel else (CYAN, 14, 1)
            c = (int(round(u)), int(round(v)))
            for width, colour in ((thick + 3, (0, 0, 0)), (thick, col)):
                cv2.line(img, (c[0] - arm_len, c[1]), (c[0] + arm_len, c[1]), colour, width, cv2.LINE_AA)
                cv2.line(img, (c[0], c[1] - arm_len), (c[0], c[1] + arm_len), colour, width, cv2.LINE_AA)
                cv2.circle(img, c, arm_len // 2, colour, width, cv2.LINE_AA)
            label = f"{pt.get('id')} {pt.get('name') or ''}".strip()
            x = int(np.clip(c[0] + arm_len + 4, 4, max(w - 120, 4)))
            y = int(np.clip(c[1] - arm_len, 18, h - 6))
            cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)

    @staticmethod
    def _draw_grid(img, counts):
        h, w = img.shape[:2]
        layer = img.copy()
        n = counts.shape[0]
        for r in range(n):
            for c in range(n):
                k = int(counts[r, c])
                col = RED if k == 0 else ORANGE if k == 1 else GREEN
                x0, x1 = int(c * w / n), int((c + 1) * w / n)
                y0, y1 = int(r * h / n), int((r + 1) * h / n)
                cv2.rectangle(layer, (x0, y0), (x1, y1), col, -1)
        cv2.addWeighted(layer, 0.25, img, 0.75, 0, img)
        for r in range(n):
            for c in range(n):
                x0, x1 = int(c * w / n), int((c + 1) * w / n)
                y0, y1 = int(r * h / n), int((r + 1) * h / n)
                cv2.rectangle(img, (x0, y0), (x1, y1), WHITE, 1)
                cv2.putText(img, str(int(counts[r, c])), (x0 + 8, y0 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2,
                            cv2.LINE_AA)
