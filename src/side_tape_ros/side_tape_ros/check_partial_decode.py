"""Offline proof that ``partial_decode`` is equivalent to the normal decode path.

    python3 -m side_tape_ros.check_partial_decode BAG [BAG ...] [--target seam|edge|config] [--max-frames N]

For every bag and tracker mode (align, cut) it runs the tracker twice on the same JPEG stream:
  normal   cv2.imdecode -> derotate_image -> tracker.step
  partial  libjpeg windowed decode of the raw ROI window into the RoiCanvas -> tracker.step
and compares the FULL Observation on every frame (state, line a/b/c, polarity, e_lat_mm, e_yaw_deg,
conf, n_inliers, rms_px, reasons; proc_ms is wall time and excluded), exactly (NaN == NaN).  It also
reports the max pixel difference inside the ROI window between the windowed decode and cv2.imdecode,
an extra sweep of random windows (incl. image borders, margin 16 and 0), and per-frame decode timings.
``--target edge`` runs a temp copy of the config with ``detector.target: edge``.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from . import core_path  # noqa: F401
from . import jpeg_window
from .bench_perception import TOPIC, tracker_roi
from .fast_frame import PartialRoiDecoder, RoiCanvas
from precision_box_detector.side_tape.backends.classical import ClassicalBackend
from precision_box_detector.side_tape.config import load_config
from precision_box_detector.side_tape.geometry import derotate_image
from precision_box_detector.side_tape.replay import load_mask
from precision_box_detector.side_tape.tracker import EdgeTracker

FIELDS = ("state", "line", "polarity", "e_lat_mm", "e_yaw_deg", "conf", "n_inliers", "rms_px", "reasons")


def _same(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return a is not None and b is not None and np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)
    if isinstance(a, float) and isinstance(b, float):
        return (math.isnan(a) and math.isnan(b)) or a == b
    return a == b


def _value(obs, name):
    v = getattr(obs, name)
    return v.value if name == "state" else v


def iter_bag(path, topic=TOPIC, skip=0, max_frames=None):
    """Stream (stamp_s, jpeg uint8 array) from a rosbag2 without holding the bag in memory."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    info = rosbag2_py.Info().read_metadata(str(path), "")
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(path), storage_id=info.storage_identifier),
                rosbag2_py.ConverterOptions("", ""))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))
    msg_type = get_message(types[topic])
    n = 0
    while reader.has_next():
        _, data, _ = reader.read_next()
        n += 1
        if n <= skip:
            continue
        if max_frames is not None and n - skip > max_frames:
            return
        msg = deserialize_message(data, msg_type)
        yield msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9, np.frombuffer(bytes(msg.data), np.uint8)


class Lane:
    """One tracker + decode path (normal or partial) for one mode; fed frame by frame."""

    def __init__(self, cfg, config_path, mode, partial, decoder):
        self.cfg, self.mode, self.partial = cfg, mode, partial
        self.tracker = EdgeTracker(cfg, ClassicalBackend(cfg.detector, load_mask(cfg, config_path)))
        self.tracker.set_mode(mode)
        self.rot = cfg.camera.rotate_180
        self.canvas = RoiCanvas(cfg.camera.width, cfg.camera.height, self.rot, 3)
        self.pdec = PartialRoiDecoder(decoder, self.canvas) if partial else None
        self.times, self.fallbacks, self.px_max, self.seq = [], 0, 0, 0

    def feed(self, stamp, buf, ref_full=None):
        """Returns the Observation. ``ref_full`` (imdecode+derotate) is only used for the pixel check."""
        roi = tracker_roi(self.cfg, self.mode)  # same ROI the tracker will use (mode-dependent)
        t0 = time.perf_counter()
        image = None
        if self.partial and roi is not None:
            image = self.pdec.decode_roi(buf, roi)
            if image is None:
                self.fallbacks += 1
        if image is None:
            image = derotate_image(cv2.imdecode(buf, cv2.IMREAD_COLOR), self.rot)
        self.times.append(time.perf_counter() - t0)
        if self.partial and roi is not None and image is self.canvas._canvas:
            x0, y0, x1, y1 = roi
            self.px_max = max(self.px_max, int(np.abs(image[y0:y1, x0:x1].astype(np.int16)
                                                      - ref_full[y0:y1, x0:x1].astype(np.int16)).max()))
        obs = self.tracker.step(image, stamp, self.seq)
        self.seq += 1
        if obs.state.value == "LOST":  # as perception_node does while the phase is IDLE
            self.tracker.reset()
        return obs


def window_sweep(decoder, bufs, cfg, n_windows=40, seed=0):
    """Max pixel diff of random raw windows (borders forced in) vs cv2.imdecode."""
    rng = np.random.default_rng(seed)
    W, H = cfg.camera.width, cfg.camera.height
    worst, count, bad_status, worst0 = 0, 0, 0, 0
    for buf in bufs:
        full = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        rects = [(0, 0, 1, 1), (W - 1, H - 1, 1, 1), (0, 0, W, H), (0, H - 240, 320, 240), (W - 320, 0, 320, 240)]
        for _ in range(n_windows):
            w, h = int(rng.integers(1, 400)), int(rng.integers(1, 300))
            rects.append((int(rng.integers(0, W - w + 1)), int(rng.integers(0, H - h + 1)), w, h))
        for x, y, w, h in rects:
            for margin in (16, 0):  # margin 0 is informational: shows what the safety margin protects
                o = np.empty((h, w, 3), np.uint8)
                st, _, _ = decoder.decode(buf, x, y, w, h, o, margin)
                if st:
                    bad_status += 1
                    continue
                d = int(np.abs(o.astype(np.int16) - full[y:y + h, x:x + w]).max())
                if margin:
                    worst, count = max(worst, d), count + 1
                else:
                    worst0 = max(worst0, d)
    return worst, count, bad_status, worst0


def _pct(v, q):
    return float(np.percentile(v, q)) * 1000.0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bags", nargs="+")
    ap.add_argument("--config", default=core_path.DEFAULT_CONFIG)
    ap.add_argument("--target", choices=("config", "seam", "edge"), default="config",
                    help="override detector.target in a temp copy of the config")
    ap.add_argument("--modes", default="align,cut")
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--skip", type=int, default=0, help="skip the first N frames of each bag")
    args = ap.parse_args(argv)
    decoder, why = jpeg_window.load_decoder_or_reason()
    if decoder is None:
        print(f"FAIL: windowed decoder unavailable: {why}")
        return 2
    config_path = args.config
    tmp = None
    if args.target != "config":
        import yaml
        tmp = tempfile.TemporaryDirectory()
        data = yaml.safe_load(Path(args.config).read_text())
        data.setdefault("detector", {})["target"] = args.target
        config_path = str(Path(tmp.name) / "config.yaml")
        Path(config_path).write_text(yaml.safe_dump(data))
    cfg = load_config(config_path)
    print(f"config {args.config} detector.target={cfg.detector.target} rotate_180={cfg.camera.rotate_180}")
    ok_all = True
    modes = args.modes.split(",")
    for bag in args.bags:
        print(f"\n===== {bag} =====")
        lanes = {(m, p): Lane(cfg, config_path, m, p, decoder) for m in modes for p in (False, True)}
        mism = {m: {f: 0 for f in FIELDS} for m in modes}
        first = {m: {} for m in modes}
        states = {m: {} for m in modes}
        samples, n = [], 0
        for stamp, buf in iter_bag(bag, skip=args.skip, max_frames=args.max_frames):
            ref = derotate_image(cv2.imdecode(buf, cv2.IMREAD_COLOR), cfg.camera.rotate_180)
            for m in modes:
                a = lanes[(m, False)].feed(stamp, buf)
                b = lanes[(m, True)].feed(stamp, buf, ref)
                states[m][a.state.value] = states[m].get(a.state.value, 0) + 1
                for f in FIELDS:
                    if not _same(_value(a, f), _value(b, f)):
                        mism[m][f] += 1
                        first[m].setdefault(f, n)
            if n % 400 == 0 and len(samples) < 15:
                samples.append(buf)
            n += 1
        wmax, wcount, wbad, w0 = window_sweep(decoder, samples, cfg)
        print(f"  {n} frames. random-window sweep: {wcount} windows at margin 16: max pixel diff {wmax}, "
              f"bad status {wbad} (informational, margin 0: max diff {w0})")
        ok_all &= n > 0 and wmax == 0 and wbad == 0
        for m in modes:
            ln, lp = lanes[(m, False)], lanes[(m, True)]
            ok = not any(mism[m].values()) and lp.px_max == 0 and lp.fallbacks == 0
            ok_all &= ok
            print(f"  mode={m}: {'PASS' if ok else 'FAIL'}  states={states[m]}")
            print("    mismatches per field: " + "  ".join(f"{f}={c}" for f, c in mism[m].items())
                  + (f"   first at {first[m]}" if first[m] else ""))
            print(f"    ROI pixel max diff (window decode vs imdecode+derotate): {lp.px_max}   "
                  f"fallbacks to imdecode: {lp.fallbacks}")
            t_n, t_p = ln.times, lp.times
            print(f"    decode ms/frame  normal(imdecode+derotate) mean {statistics.fmean(t_n)*1000:.2f} "
                  f"p50 {_pct(t_n, 50):.2f} p95 {_pct(t_n, 95):.2f} | partial mean {statistics.fmean(t_p)*1000:.2f} "
                  f"p50 {_pct(t_p, 50):.2f} p95 {_pct(t_p, 95):.2f}")
        t_r = []
        for buf in samples:
            for _ in range(5):
                t = time.perf_counter()
                cv2.imdecode(buf, cv2.IMREAD_REDUCED_COLOR_2)
                t_r.append(time.perf_counter() - t)
        if t_r:
            print(f"  IMREAD_REDUCED_COLOR_2 decode ms: mean {statistics.fmean(t_r)*1000:.2f} p50 {_pct(t_r, 50):.2f}")
    print("\nOVERALL:", "PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
