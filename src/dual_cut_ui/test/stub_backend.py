#!/usr/bin/env python3
"""MOCK ONLY - the real web UI (server.py, core.py, static/) on a stubbed backend, no ROS, no robot, no coordinator.

For looking at the page in a browser: it serves the real Tornado app on 127.0.0.1 with a ``FakeBridge`` that behaves like a
minimal coordinator (status, plan, execute, stop, gates, allow_blocked_detection) and a "detector" that writes the next
snapshot of a list on every Detect.  It never opens a ROS context, so the ROS domain does not matter; it still refuses to run
unless ROS_DOMAIN_ID is set explicitly (and is not 11, the real robot) and ROS_LOCALHOST_ONLY=1, like the other mock scripts.

    ROS_DOMAIN_ID=208 ROS_LOCALHOST_ONLY=1 python3 stub_backend.py --port 8296 \
        --snapshots t090925,t061508,t062904 --fixture data/placement_20261022.json

Each Detect writes the next snapshot of ``--snapshots`` (names from the fixture, cycling).  The stub coordinator emulates the part
of the real one that matters for the blocked-detection switch: its plan report carries ``detection_blocked`` and ``execute`` is
refused while the detection is blocked and ``allow_blocked_detection`` is false.  Execute never moves anything: the fake run lasts
``--run-s`` seconds, then the state returns to IDLE (the UI then closes its gates and the allow flag, as on the real cell).
"""
import argparse
import asyncio
import json
import os
import signal
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import tornado.ioloop   # noqa: E402

from dual_cut_ui import bridge as B   # noqa: E402
from dual_cut_ui.auth import Auth   # noqa: E402
from dual_cut_ui.camera import CameraHub, DEFAULT_CAMERAS   # noqa: E402
from dual_cut_ui.core import Config, UiCore   # noqa: E402
from dual_cut_ui.server import Context, make_app   # noqa: E402
from helpers import FakeBridge, status_fields   # noqa: E402
from test_camera import synthetic_jpeg   # noqa: E402


def check_environment() -> None:
    domain, localhost = os.environ.get("ROS_DOMAIN_ID"), os.environ.get("ROS_LOCALHOST_ONLY")
    if domain is None or not domain.isdigit() or int(domain) == 11 or localhost != "1":
        raise SystemExit("stub_backend.py (MOCK ONLY): set ROS_DOMAIN_ID to an explicit unused domain (not 11) and ROS_LOCALHOST_ONLY=1")


class StubBridge(FakeBridge):
    """FakeBridge with live timestamps, a tiny coordinator state machine and cameras that deliver synthetic frames."""

    def __init__(self, run_s: float, plan_report_text: str, snapshot_writer):
        super().__init__(time.monotonic)
        self.run_s = run_s
        self.plan_text = plan_report_text
        self.write_snapshot = snapshot_writer
        self.state = B.STATE_IDLE
        self.plan_seq = 0
        self.retained = False
        self.stopped = False
        self.gate_vals = {p: False for p in B.GATE_PARAMS}
        self.allow = False
        self.current_blocked = False
        self.detection_hook_set()
        self.detect_feedback = [("capture", 1, "grabbing frames"), ("detect", 1, "seam found")]
        self.log_lines = []
        self.results["plan"] = (True, "stub plan: sequence, makespan 78 s")
        self.cam_cb = {}
        threading.Thread(target=self._camera_loop, daemon=True, name="stub-cameras").start()

    def detection_hook_set(self):
        self.detect_hook = lambda: setattr(self, "current_blocked", self.write_snapshot())

    # ---- cached data, always "just received"
    def status(self):
        f = status_fields(state=self.state, plan_seq=self.plan_seq, plan_retained=self.retained, phase_validated=self.retained,
                          stop_latched=self.stopped, stop_reason="software stop" if self.stopped else "",
                          phase="A" if self.state == B.STATE_EXECUTING else "", min_cross_distance_m=0.062)
        return f, time.monotonic()

    def plan_report(self):
        if not self.retained:
            return None
        doc = json.loads(self.plan_text)
        doc["detection_blocked"] = bool(self.current_blocked)
        return json.dumps(doc), time.monotonic()

    def health(self):
        return {"controllers": {"left": True, "right": True}, "hardware": {"left": True, "right": True}}, time.monotonic()

    def gates(self):
        vals = dict(self.gate_vals)
        vals.update({"phase_order": "side_first", "schedule_mode": "sequence", "top_split_mode": "left_max", "external_snapshot_max_age_s": 240.0})
        return vals, time.monotonic()

    def allow_blocked(self):
        return self.allow, time.monotonic()

    # ---- calls
    def trigger(self, name, timeout_s):
        self.record("trigger", name, timeout_s)
        if name == "plan":
            time.sleep(1.0)
            self.plan_seq += 1
            self.retained, self.state, self.stopped = True, B.STATE_PLANNED, False
            return True, "stub plan: sequence, makespan 78 s"
        if name == "execute":
            if self.current_blocked and not self.allow:
                return False, "execute refused: the detection is blocked (cut_permission) and allow_blocked_detection is false"
            if not all(self.gate_vals[p] for p in B.ENABLE_PARAMS):
                return False, "execute refused: execution is not enabled"
            self.state = B.STATE_EXECUTING
            threading.Timer(self.run_s, self._run_done).start()
            return True, "dispatch accepted (stub: nothing moves)"
        if name == "stop":
            self.state, self.stopped = B.STATE_STOPPED, True
            self.retained = False
            return True, "stopped"
        if name == "rearm":
            self.state, self.stopped, self.retained = B.STATE_IDLE, False, False
            return True, "rearmed"
        return False, "unknown"

    def _run_done(self):
        if self.state == B.STATE_EXECUTING:
            self.state = B.STATE_IDLE
            self.retained = False

    def set_gates(self, enabled, timeout_s):
        self.record("set_gates", enabled, timeout_s)
        for p in B.gate_params_for(enabled):
            self.gate_vals[p] = enabled
        return True, f"execution gates set {enabled}"

    def set_allow_blocked(self, enabled, timeout_s):
        self.record("set_allow_blocked", enabled, timeout_s)
        self.allow = bool(enabled)
        print(f"[stub coordinator] allow_blocked_detection -> {self.allow}", flush=True)
        return True, f"allow_blocked_detection set {str(self.allow).lower()}"

    def set_schedule_mode(self, mode, timeout_s):
        return True, f"schedule_mode set {mode}"

    def cancel_all(self, arm, timeout_s):
        return True, "no active goal"

    # ---- cameras: one synthetic frame per second for every subscribed topic
    def subscribe_camera(self, name, topic, callback):
        self.cam_cb[name] = callback

    def unsubscribe_camera(self, name):
        self.cam_cb.pop(name, None)

    def _camera_loop(self):
        seed = 0
        while True:
            time.sleep(0.5)
            seed += 1
            for name, cb in list(self.cam_cb.items()):
                try:
                    cb(synthetic_jpeg(640, 360, seed=(seed + len(name)) % 9))
                except Exception:  # noqa: BLE001
                    pass


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", type=int, default=8296)
    ap.add_argument("--fixture", default=os.path.join(HERE, "data", "placement_20261022.json"))
    ap.add_argument("--snapshots", default="t090925", help="comma separated names from the fixture; each Detect writes the next one")
    ap.add_argument("--run-s", type=float, default=6.0)
    ap.add_argument("--workdir", default="/tmp/dual_cut_ui_stub")
    args = ap.parse_args(argv)
    check_environment()
    fixture = json.load(open(args.fixture))["detections"]
    names = [n for n in args.snapshots.split(",") if n]
    for n in names:
        if n not in fixture:
            raise SystemExit(f"unknown snapshot {n}")
    os.makedirs(args.workdir, exist_ok=True)
    snap_path = os.path.join(args.workdir, "seam_3d_base_link_STUB.json")
    cursor = {"i": 0, "name": names[0], "blocked": False}

    def write_snapshot(advance=True) -> bool:
        if advance:
            cursor["name"] = names[cursor["i"] % len(names)]
            cursor["i"] += 1
        doc = dict(fixture[cursor["name"]]["snapshot"])
        doc.update({"schema": "box_h_cut_snapshot/v1", "snapshot_id": "stub_" + cursor["name"], "capture_id": "stub_" + cursor["name"],
                    "frame_id": "base_link", "detector_method": "STUB (placement fixture)", "plane_rms_m": 0.0023,
                    "acquisition_stamp_ns": int(time.time() * 1e9)})
        tmp = snap_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(doc, f)
        os.replace(tmp, snap_path)
        cursor["blocked"] = (doc.get("cut_permission") or {}).get("status") == "blocked"
        return cursor["blocked"]

    write_snapshot()                                            # the first snapshot is there at start
    plan_text = open(os.path.join(HERE, "data", "sequence_plan_report.json")).read()
    bridge = StubBridge(args.run_s, plan_text, write_snapshot)
    bridge.current_blocked = cursor["blocked"]
    cfg = Config(snapshot_file=snap_path, detect_mode="action", detect_output_dir=os.path.join(args.workdir, "det"),
                 schedule_mode="sequence", plan_timeout_s=30.0)
    core = UiCore(bridge, cfg)
    auth = Auth(args.port)
    hub = CameraHub(bridge, DEFAULT_CAMERAS)
    ctx = Context(core, hub, auth)
    core.start()

    def restamp():                                              # keep the current snapshot fresh while the page is open
        while True:
            time.sleep(30)
            write_snapshot(advance=False)
    threading.Thread(target=restamp, daemon=True, name="stub-restamp").start()

    async def serve():
        app = make_app(ctx)
        server = app.listen(args.port, address="127.0.0.1")
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop.set)
        tornado.ioloop.PeriodicCallback(lambda: ctx.image_pool.submit(hub.sweep), 1000).start()
        print(f"STUB dual-cut UI (MOCK ONLY, no ROS): {auth.url()}", flush=True)
        await stop.wait()
        server.stop()
    try:
        asyncio.run(serve())
    finally:
        core.shutdown()
        hub.shutdown()
        ctx.shutdown()


if __name__ == "__main__":
    main()
