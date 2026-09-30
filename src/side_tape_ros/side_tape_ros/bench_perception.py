"""Offline per-stage profile and fast-vs-normal equivalence for the side_tape perception path.

Reads rosbag2 CompressedImage streams (no ROS graph, no node started):

    python3 -m side_tape_ros.bench_perception BAG [BAG ...] [--mode align|cut] [--max-frames N]

Per bag it runs the tracker twice, exactly as perception_node does (decode -> de-rotate ->
tracker.step): the normal path and the ``fast_decode`` path (ROI-only rotate), prints a per-stage
timing table, and checks that both give the same line (within --tol px) and state sequence.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections import defaultdict

import cv2
import numpy as np

from . import core_path  # noqa: F401
from .fast_frame import RoiCanvas
from precision_box_detector.side_tape import geometry as geo
from precision_box_detector.side_tape.backends import classical as classical_mod
from precision_box_detector.side_tape.backends.classical import ClassicalBackend
from precision_box_detector.side_tape.config import load_config
from precision_box_detector.side_tape.replay import load_mask
from precision_box_detector.side_tape.tracker import EdgeTracker

TOPIC = "/knife_camera/image_raw/compressed"


def read_bag(path, topic=TOPIC, max_frames=None):
    """Return [(stamp_s, jpeg_bytes_as_uint8_array)] for the topic."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    info = rosbag2_py.Info().read_metadata(str(path), "")
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(path), storage_id=info.storage_identifier),
                rosbag2_py.ConverterOptions("", ""))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if topic not in types:
        raise ValueError(f"{topic} not in {path}; has {sorted(types)}")
    reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))
    msg_type = get_message(types[topic])
    frames = []
    while reader.has_next() and (max_frames is None or len(frames) < max_frames):
        _, data, _ = reader.read_next()
        msg = deserialize_message(data, msg_type)
        frames.append((msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                       np.frombuffer(bytes(msg.data), np.uint8)))
    return frames


class Timers:
    """Accumulates per-stage seconds; wraps the backend's internals without editing it."""

    def __init__(self):
        self.samples = defaultdict(list)
        self._active = None  # per-frame accumulator while inside backend.detect

    def add(self, name, dt):
        self.samples[name].append(dt)

    def wrap(self, owner, attr, name):
        orig = getattr(owner, attr)

        def timed(*a, **k):
            t = time.perf_counter()
            try:
                return orig(*a, **k)
            finally:
                if self._active is not None:
                    self._active[name] += time.perf_counter() - t
        setattr(owner, attr, timed)
        return lambda: setattr(owner, attr, orig)


class _LsdProxy:
    def __init__(self, inner, timers):
        self._inner, self._t = inner, timers

    def detect(self, img):
        t = time.perf_counter()
        try:
            return self._inner.detect(img)
        finally:
            if self._t._active is not None:
                self._t._active["lsd"] += time.perf_counter() - t


def tracker_roi(cfg, mode):
    center = geo.lookahead_center(cfg.tip, cfg.J, cfg.detector.lookahead_mm) if mode == "cut" else cfg.tip
    try:
        return geo.roi_around(center, cfg.detector.roi_half_w, cfg.detector.roi_half_h,
                              cfg.camera.width, cfg.camera.height)
    except ValueError:
        return None


DETECT_STAGES = ("gray", "glare", "blur", "lsd", "canny", "hough", "refine")


def run_path(cfg, config_path, frames, fast, mode, profile=True):
    timers = Timers()
    backend = ClassicalBackend(cfg.detector, load_mask(cfg, config_path))
    restores = []
    if profile:
        backend._lsd = _LsdProxy(backend._lsd, timers)
        restores += [timers.wrap(classical_mod.cv2, "cvtColor", "gray"),
                     timers.wrap(classical_mod.cv2, "GaussianBlur", "blur"),
                     timers.wrap(classical_mod.cv2, "Canny", "canny"),
                     timers.wrap(classical_mod.cv2, "HoughLinesP", "hough"),
                     timers.wrap(classical_mod, "exclude_glare_fray", "glare"),
                     timers.wrap(backend, "_refine", "refine")]
        raw_detect = backend.detect

        def detect(*a, **k):
            timers._active = defaultdict(float)
            t = time.perf_counter()
            try:
                return raw_detect(*a, **k)
            finally:
                total = time.perf_counter() - t
                for stage in DETECT_STAGES:
                    timers.add(stage, timers._active.get(stage, 0.0))
                timers.add("detect_total", total)
                timers._active = None
        backend.detect = detect
    tracker = EdgeTracker(cfg, backend)
    tracker.set_mode(mode)
    canvas = None
    out = []
    try:
        for seq, (stamp, buf) in enumerate(frames):
            t0 = time.perf_counter()
            image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            t1 = time.perf_counter()
            roi = tracker_roi(cfg, mode) if fast else None
            if roi is None:
                image = geo.derotate_image(image, cfg.camera.rotate_180)
            else:
                if canvas is None:
                    canvas = RoiCanvas(cfg.camera.width, cfg.camera.height, cfg.camera.rotate_180, image.shape[2])
                image = canvas.fill(image, roi)
            t2 = time.perf_counter()
            obs = tracker.step(image, stamp, seq)
            t3 = time.perf_counter()
            timers.add("decode", t1 - t0)
            timers.add("derotate" if not fast else "roi_rotate", t2 - t1)
            timers.add("tracker_step", t3 - t2)
            timers.add("obs_proc_ms", obs.proc_ms / 1000.0)
            timers.add("total", t3 - t0)
            out.append((obs.state.value, None if obs.line is None else np.asarray(obs.line, float)))
    finally:
        for restore in restores:
            restore()
    return timers, out


def _pct(values, q):
    return float(np.percentile(values, q)) * 1000.0 if values else float("nan")


def print_table(title, timers, fast):
    rows = ["decode", "roi_rotate" if fast else "derotate", "tracker_step"]
    print(f"\n{title}  (ms per frame; stages with count 0 are frames that skipped them)")
    print(f"  {'stage':<22}{'mean':>8}{'p50':>8}{'p95':>8}{'max':>8}")
    def line(name, label=None):
        v = timers.samples.get(name, [])
        if not v:
            return
        print(f"  {label or name:<22}{statistics.fmean(v)*1000:8.2f}{_pct(v,50):8.2f}{_pct(v,95):8.2f}{max(v)*1000:8.2f}")
    for r in rows:
        line(r)
    if "detect_total" in timers.samples:
        line("detect_total", "  backend.detect")
        for s in DETECT_STAGES:
            line(s, f"    {s}" + (" (cvtColor)" if s == "gray" else ""))
        d = np.array(timers.samples["detect_total"])
        parts = sum(np.array(timers.samples[s]) for s in DETECT_STAGES)
        step = np.array(timers.samples["tracker_step"])
        print(f"  {'    other in detect':<22}{np.mean(d-parts)*1000:8.2f}")
        print(f"  {'  tracker (no detect)':<22}{np.mean(step-d)*1000:8.2f}")
    line("obs_proc_ms", "obs.proc_ms (logged)")
    line("total", "TOTAL decode+step")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bags", nargs="+")
    ap.add_argument("--config", default=core_path.DEFAULT_CONFIG)
    ap.add_argument("--mode", choices=("align", "cut"), default="align")
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--tol", type=float, default=0.01, help="max |delta(a,b,c)| for equivalence")
    ap.add_argument("--no-profile", action="store_true", help="skip per-stage wrappers (cleaner totals)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    ok_all = True
    for bag in args.bags:
        frames = read_bag(bag, max_frames=args.max_frames)
        print(f"\n===== {bag}  ({len(frames)} frames, mode={args.mode}) =====")
        run_path(cfg, args.config, frames[:20], False, args.mode, profile=False)  # warm-up
        normal, out_n = run_path(cfg, args.config, frames, False, args.mode, profile=not args.no_profile)
        fast, out_f = run_path(cfg, args.config, frames, True, args.mode, profile=not args.no_profile)
        print_table("NORMAL", normal, False)
        print_table("FAST (fast_decode)", fast, True)
        # Uninstrumented totals for honest before/after numbers.
        clean_n, _ = run_path(cfg, args.config, frames, False, args.mode, profile=False)
        clean_f, _ = run_path(cfg, args.config, frames, True, args.mode, profile=False)
        print("\n  uninstrumented (ms)       p50     p95    (obs.proc_ms | total decode+derotate+step)")
        for label, t in (("normal", clean_n), ("fast", clean_f)):
            print(f"  {label:<8} obs.proc_ms {_pct(t.samples['obs_proc_ms'],50):7.2f} {_pct(t.samples['obs_proc_ms'],95):7.2f}"
                  f"   total {_pct(t.samples['total'],50):7.2f} {_pct(t.samples['total'],95):7.2f}")
        bad_state = sum(1 for (sn, _), (sf, _) in zip(out_n, out_f) if sn != sf)
        max_d, bad_line = 0.0, 0
        for (_, ln), (_, lf) in zip(out_n, out_f):
            if (ln is None) != (lf is None):
                bad_line += 1
                continue
            if ln is not None:
                d = float(np.abs(ln - lf).max())
                max_d = max(max_d, d)
                bad_line += d > args.tol
        states = defaultdict(int)
        for s, _ in out_n:
            states[s] += 1
        ok = bad_state == 0 and bad_line == 0
        ok_all &= ok
        print(f"\n  equivalence: {'PASS' if ok else 'FAIL'}  state mismatches={bad_state}  "
              f"line mismatches(>{args.tol})={bad_line}  max|d(a,b,c)|={max_d:.3g}  states={dict(states)}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
