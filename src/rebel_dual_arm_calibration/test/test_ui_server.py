"""Calibration web UI backend: auth, API contract, jobs, STOP independence (fake ROS bridge and jog; no ROS graph)."""
import asyncio
import json
import os
import sys
import threading
import time

import cv2
import numpy as np
import pytest

pytest.importorskip("tornado")

from tornado.testing import AsyncHTTPTestCase, gen_test   # noqa: E402

from rebel_dual_arm_calibration.targets import TargetDetector, default_targets_path, load_targets_all   # noqa: E402
from rebel_dual_arm_calibration.ui import server as ui   # noqa: E402
from rebel_dual_arm_calibration.ui.auth import Auth   # noqa: E402
from rebel_dual_arm_calibration.ui.jobs import JobManager   # noqa: E402
from rebel_dual_arm_calibration.ui.overlay import ImagePipeline, coverage_counts   # noqa: E402

TOKEN = "test-token-123"
PAGE = "page-abc123"

STATUS = {
    "session_dir": "/tmp/uiback_session/s1", "active_arms": ["left", "right"], "n_intrinsics": 3,
    "touch_list": [{"id": 1, "kind": "board"}, {"id": 2, "kind": "board"}, {"id": 101, "kind": "fit"},
                   {"id": 111, "kind": "check"}],
    "touched_ids": {"left": [1, 101], "right": [1]}, "min_corners": {"left": 8, "right": 8, "table": 16},
    "intrinsics_bboxes": [[0, 0, 600, 400]], "intrinsics_centres": [[300, 200]],
}


class FakeBridge:
    def __init__(self):
        self.status = dict(STATUS)
        self.calls = []
        self.block = None                      # threading.Event: trigger() waits for it
        self.frame = None
        self.fail_params = False

    def health(self):
        return {"image": {"ok": self.frame is not None, "age_s": 0.1 if self.frame is not None else None,
                          "size": [640, 480] if self.frame is not None else None},
                "camera_info": True, "recorder": self.status is not None}

    def recorder_status(self):
        return self.status

    def latest_frame(self):
        return None if self.frame is None else (self.frame, 1, time.monotonic())

    def trigger(self, name, timeout_s=15.0):
        self.calls.append(("trigger", name))
        if self.block is not None:
            self.block.wait(10.0)
        return True, f"{name} done"

    def set_parameters(self, values, timeout_s=5.0):
        self.calls.append(("params", dict(values)))
        return (False, "rejected") if self.fail_params else (True, "parameters set")


class FakeJog:
    def __init__(self):
        self.calls = []
        self.busy = False
        self.q = {f"joint{i}": 0.0 for i in range(1, 7)}       # joint_window for the auto capture
        self.spread = 0.0

    def status(self):
        return {"available": True, "enabled": False, **({"busy": True} if self.busy else {})}

    def joint_window(self, arms, window_s):
        return dict(self.q), self.spread, True

    def enable_real(self, page_id):
        self.calls.append(("enable", page_id))
        return {"ok": True, "message": "enabled"}

    def disable(self, reason=""):
        self.calls.append(("disable", reason))

    def step(self, page_id, arm, mode, axis, sign, size, frame="world"):
        self.calls.append(("step", page_id, arm, mode, axis, sign, size, frame))
        return {"ok": True, "id": 7, "message": "queued"}

    def stop(self):
        self.calls.append(("stop",))
        return {"ok": True, "message": "stopped"}

    def clear_fault(self):
        self.calls.append(("clear_fault",))
        return {"ok": False, "message": "no fault to clear"}

    def plan_move(self, page_id, arm, q_goal):
        self.calls.append(("plan_move", page_id, arm, dict(q_goal)))
        return {"ok": True, "plan_id": "plan7", "message": "planned", "duration_s": 6.0,
                "max_joint_change_deg": 12.0, "n_points": 30}

    def execute_move(self, page_id, plan_id):
        self.calls.append(("execute_move", page_id, plan_id))
        return {"ok": True, "message": "moving"}

    def clear_plan(self):
        self.calls.append(("clear_plan",))

    def shutdown(self):
        pass


class Base(AsyncHTTPTestCase):
    mode = "dual"
    job_cmd = staticmethod(lambda session_dir, out: [sys.executable, "-c", "print('calibrating', 1)"])

    def get_app(self):
        self.auth = Auth(self.get_http_port(), token=TOKEN)
        self.bridge, self.jog = FakeBridge(), FakeJog()
        specs, table = load_targets_all(default_targets_path())
        self.pipeline = ImagePipeline(self.bridge.latest_frame, lambda: TargetDetector({**specs, "table": table}))
        self.ctx = ui.Context(self.bridge, self.jog, self.pipeline, self.auth, self.mode, JobManager(),
                              self.job_cmd)
        self.static = os.path.join(os.path.dirname(__file__), "_ui_static_tmp")
        os.makedirs(self.static, exist_ok=True)
        with open(os.path.join(self.static, "index.html"), "w") as f:
            f.write("<html>calibration</html>")
        return ui.make_app(self.ctx, self.static)

    def tearDown(self):
        self.ctx.jobs.shutdown()
        self.ctx.shutdown()
        try:
            os.remove(os.path.join(self.static, "index.html"))
            os.rmdir(self.static)
        except OSError:
            pass
        super().tearDown()

    def hdr(self, **extra):
        h = {"X-Token": TOKEN, "X-Page-Id": PAGE}
        h.update(extra)
        return h

    def get(self, path, **kw):
        return self.fetch(path, headers=kw.pop("headers", self.hdr()), **kw)

    def post(self, path, body=None, headers=None):
        return self.fetch(path, method="POST", body=json.dumps({} if body is None else body),
                          headers=self.hdr() if headers is None else headers)

    @staticmethod
    def j(resp):
        return json.loads(resp.body)


class TestAuth(Base):
    def test_token_required_everywhere_in_api(self):
        for path in ("/api/state", "/api/image.jpg"):
            assert self.fetch(path).code == 401
            assert self.fetch(path, headers={"X-Token": "wrong"}).code == 401
        for path in ("/api/capture", "/api/touch", "/api/session", "/api/intrinsics/capture",
                     "/api/intrinsics/calibrate", "/api/jog/enable", "/api/jog/disable", "/api/jog/step",
                     "/api/jog/stop", "/api/auto_capture/start", "/api/auto_capture/stop",
                     "/api/suggest/compute", "/api/suggest/select", "/api/suggest/send_rviz", "/api/suggest/plan",
                     "/api/suggest/move", "/api/hover/compute", "/api/hover/prepare", "/api/hover/send_rviz",
                     "/api/hover/move", "/api/hover/descend", "/api/hover/record_offset",
                     "/api/tcp/checklist", "/api/tcp/capture", "/api/tcp/delete", "/api/tcp/solve",
                     "/api/tcp/mechanical", "/api/tcp/write"):
            r = self.fetch(path, method="POST", body="{}")
            assert r.code == 401 and self.j(r)["ok"] is False
        assert self.jog.calls == [] and self.bridge.calls == []

    def test_index_is_public_and_api_with_token_works(self):
        r = self.fetch("/")
        assert r.code == 200 and b"calibration" in r.body
        assert self.get("/api/state").code == 200

    def test_origin_check_on_post(self):
        port = self.get_http_port()
        evil = self.post("/api/capture", headers=self.hdr(Origin="http://evil.example"))
        assert evil.code == 403 and self.bridge.calls == []
        for origin in (f"http://127.0.0.1:{port}", f"http://localhost:{port}"):
            assert self.post("/api/capture", headers=self.hdr(Origin=origin)).code == 200
        assert self.post("/api/capture").code == 200                       # no Origin header
        assert self.post("/api/capture", headers=self.hdr(Origin="http://127.0.0.1:1")).code == 403

    def test_foreign_host_rejected(self):
        assert self.get("/api/state", headers=self.hdr(Host="evil.example")).code == 403


class TestFrontendContract(Base):
    def test_static_and_extra_body_fields_are_tolerated(self):
        open(os.path.join(self.static, "app.js"), "w").write("//js")
        try:
            assert self.fetch("/static/app.js").code == 200
        finally:
            os.remove(os.path.join(self.static, "app.js"))
        assert self.post("/api/capture", {"arm": "left"}).code == 200
        assert self.post("/api/jog/enable", {"arm": "left", "junk": 1}).code == 200
        body = {"arm": "left", "mode": "joint", "axis": "j1", "sign": -1, "size": "fine", "frame": "world", "x": 1}
        assert self.post("/api/jog/step", body).code == 200


class TestJog(Base):
    def test_page_id_reaches_jog(self):
        assert self.post("/api/jog/enable", {"arm": "left"}).code == 200
        body = {"arm": "left", "mode": "joint", "axis": "j1", "sign": 1, "size": "fine"}
        r = self.post("/api/jog/step", body)
        assert r.code == 200 and self.j(r)["id"] == 7
        assert self.jog.calls[0] == ("enable", PAGE)
        assert self.jog.calls[1] == ("step", PAGE, "left", "joint", "j1", 1, "fine", "world")
        self.post("/api/jog/step", dict(body, frame="tool"))
        assert self.jog.calls[2][-1] == "tool"
        assert self.post("/api/jog/disable").code == 200 and self.jog.calls[-1] == ("disable", "operator")

    def test_missing_page_id_or_fields(self):
        h = {"X-Token": TOKEN}
        assert self.post("/api/jog/enable", headers=h).code == 400
        assert self.post("/api/jog/step", {"arm": "left"}).code == 400
        assert self.jog.calls == []

    def test_stop_needs_no_page_id(self):
        r = self.post("/api/jog/stop", headers={"X-Token": TOKEN})
        assert r.code == 200 and self.jog.calls == [("stop",)]

    def test_clear_fault_needs_no_page_id_and_passes_the_result(self):
        r = self.post("/api/jog/clear_fault", headers={"X-Token": TOKEN})
        assert r.code == 200 and self.j(r) == {"ok": False, "message": "no fault to clear"}
        assert self.jog.calls == [("clear_fault",)]

    def test_jog_exception_is_reported_not_swallowed(self):
        self.jog.stop = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        r = self.post("/api/jog/stop")
        assert r.code == 500 and "boom" in self.j(r)["message"]


class TestTouch(Base):
    def test_touch_sets_params_then_captures(self):
        r = self.post("/api/touch", {"arm": "right", "point_id": 2})
        assert r.code == 200 and self.j(r) == {"ok": True, "message": "capture_touch done"}
        assert self.bridge.calls == [("params", {"touch_arm": "right", "touch_point_id": 2}),
                                     ("trigger", "capture_touch")]

    def test_touch_validation(self):
        for body in ({"arm": "middle", "point_id": 1}, {"arm": "left", "point_id": 999}, {"arm": "left"},
                     {"arm": "left", "point_id": "1"}, {"arm": "left", "point_id": True},
                     {"arm": "left", "point_id": 1.5}):
            r = self.post("/api/touch", body)
            assert r.code == 400 and self.j(r)["ok"] is False, body
        assert self.bridge.calls == []
        for pid in (1, 101, 111):                  # board point, fit sticker, check sticker are all valid
            assert self.post("/api/touch", {"arm": "left", "point_id": pid}).code == 200

    def test_touch_without_recorder_status(self):
        self.bridge.status = None
        assert self.post("/api/touch", {"arm": "left", "point_id": 1}).code == 503

    def test_touch_param_failure_does_not_trigger(self):
        self.bridge.fail_params = True
        r = self.post("/api/touch", {"arm": "left", "point_id": 1})
        assert self.j(r)["ok"] is False and ("trigger", "capture_touch") not in self.bridge.calls


class TestSingleArm(Base):
    mode = "single_right"

    def test_other_arm_refused(self):
        assert self.post("/api/touch", {"arm": "left", "point_id": 1}).code == 400
        assert self.post("/api/touch", {"arm": "right", "point_id": 1}).code == 200
        assert self.j(self.get("/api/state"))["mode"] == "single_right"


class TestState(Base):
    def test_state_shape(self):
        s = self.j(self.get("/api/state"))
        assert s["ok"] and s["mode"] == "dual"
        assert set(s["health"]) == {"image", "camera_info", "recorder"}
        assert set(s["health"]["image"]) == {"ok", "age_s", "size"}
        assert s["recorder"]["session_dir"] == STATUS["session_dir"]
        assert s["jog"] == {"available": True, "enabled": False}
        assert s["jobs"] == {} and s["detections"] == {}
        tl = {t["id"]: t for t in s["touch_list"]}
        assert tl[1] == {"id": 1, "kind": "board", "done": {"left": True, "right": True}}
        assert tl[2]["done"] == {"left": False, "right": False}
        assert tl[101] == {"id": 101, "kind": "fit", "done": {"left": True, "right": False}}
        assert tl[111]["kind"] == "check"
        assert s["tcp"]["supported"] is False and "single-arm modes only" in s["tcp"]["message"]     # dual mode

    def test_state_without_recorder_and_broken_jog(self):
        self.bridge.status = None
        self.jog.status = lambda: 1 / 0
        s = self.j(self.get("/api/state"))
        assert s["ok"] and s["recorder"] is None and s["touch_list"] == [] and s["health"]["recorder"] is False
        assert s["jog"]["available"] is False


class TestImage(Base):
    def test_404_then_jpeg_with_detections(self):
        r = self.get("/api/image.jpg")
        assert r.code == 404 and self.j(r)["ok"] is False
        from test_targets import _scene_with_boards
        specs, table = load_targets_all(default_targets_path())
        gray, _ = _scene_with_boards({"left": specs["left"], "table": table})
        self.bridge.frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        assert self.pipeline.process_once() is True
        assert self.pipeline.process_once() is False                       # same frame: skipped
        for overlay in ("detect", "intrinsics", "none"):
            r = self.get(f"/api/image.jpg?w=480&overlay={overlay}")
            assert r.code == 200 and r.headers["Content-Type"] == "image/jpeg"
            assert "no-store" in r.headers["Cache-Control"]
            img = cv2.imdecode(np.frombuffer(r.body, np.uint8), cv2.IMREAD_COLOR)
            assert img.shape == (270, 480, 3)
        det = self.j(self.get("/api/state"))["detections"]
        assert det["left"]["n"] >= 8 and det["left"]["ok"] is True
        assert det["table"]["n"] > 0 and "right" in det and det["right"] == {"n": 0, "ok": False}
        assert self.get("/api/image.jpg?overlay=bogus").code == 400

    def test_coverage_counts(self):
        c = coverage_counts(STATUS, (1920, 1080))
        assert c.shape == (4, 4) and c[0, 0] == 1 and c[0, 1] == 1 and c[3, 3] == 0 and c[2, 0] == 0
        c2 = coverage_counts({"intrinsics_centres": [[1900, 1070], [1900, 1070]]}, (1920, 1080))
        assert c2[3, 3] == 2 and c2.sum() == 2
        assert coverage_counts(None, (10, 10)).sum() == 0


class TestSession(Base):
    def test_session_parameters(self):
        r = self.post("/api/session", {"ambient_temp_c": 21.5, "warmup_min": 30, "operator_note": "n"})
        assert r.code == 200 and self.j(r)["ok"] is True
        assert self.bridge.calls == [("params", {"ambient_temp_c": 21.5, "warmup_min": 30.0, "operator_note": "n"})]
        for bad in ({}, {"ambient_temp_c": "hot"}, {"ambient_rh_pct": float("inf")}, {"operator_note": 5}):
            body = json.dumps(bad).encode()
            r = self.fetch("/api/session", method="POST", body=body, headers=self.hdr())
            assert r.code == 400, bad


class TestJobs(Base):
    job_cmd = staticmethod(lambda session_dir, out: [
        sys.executable, "-c", "import sys,time; print('start', sys.argv[1:]); sys.stdout.flush(); time.sleep(1.5); print('end')",
        session_dir, out])

    def wait_job(self, job_id, state, timeout=10.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            job = self.j(self.get("/api/state"))["jobs"][job_id]
            if job["state"] == state:
                return job
            time.sleep(0.1)
        raise AssertionError(f"job did not reach {state}: {job}")

    def test_lifecycle_busy_and_stop(self):
        r = self.post("/api/intrinsics/calibrate")
        assert r.code == 200 and self.j(r)["ok"] and self.j(r)["job"] == "job1"
        job = self.j(self.get("/api/state"))["jobs"]["job1"]
        assert job["state"] == "running" and job["returncode"] is None and job["result_file"] is None
        # a second job is refused and STOP is answered while the job runs
        assert self.post("/api/intrinsics/calibrate").code == 409
        t0 = time.time()
        assert self.post("/api/jog/stop").code == 200 and time.time() - t0 < 1.0
        assert self.jog.calls == [("stop",)]
        job = self.wait_job("job1", "done")
        assert job["returncode"] == 0 and job["output_tail"][-1] == "end"
        assert job["output_tail"][0].startswith("start") and "/tmp/uiback_session/intrinsics.yaml" in job["output_tail"][0]
        assert job["result_file"] is None                                  # the fake command wrote no file

    def test_result_file_and_failure(self):
        out = "/tmp/uiback_session/intrinsics.yaml"
        os.makedirs("/tmp/uiback_session", exist_ok=True)
        open(out, "w").write("x")
        try:
            self.ctx.job_command = lambda sd, o: [sys.executable, "-c", "print('ok')"]
            self.post("/api/intrinsics/calibrate")
            assert self.wait_job("job1", "done")["result_file"] == out
        finally:
            os.remove(out)
        self.ctx.job_command = lambda sd, o: [sys.executable, "-c", "import sys; print('bad', file=sys.stderr); sys.exit(2)"]
        self.post("/api/intrinsics/calibrate")
        job = self.wait_job("job2", "failed")
        assert job["returncode"] == 2 and job["output_tail"] == ["bad"] and job["result_file"] is None
        self.ctx.job_command = lambda sd, o: ["/nonexistent/cmd"]
        self.post("/api/intrinsics/calibrate")
        assert self.wait_job("job3", "failed")["returncode"] == -1

    def test_needs_views_and_session(self):
        self.bridge.status = dict(STATUS, n_intrinsics=0)
        assert self.post("/api/intrinsics/calibrate").code == 400
        self.bridge.status = None
        assert self.post("/api/intrinsics/calibrate").code == 503

    def test_shutdown_kills_the_process_group(self):
        self.ctx.job_command = lambda sd, o: ["bash", "-c", "sleep 30 & sleep 30; wait"]
        self.post("/api/intrinsics/calibrate")
        proc = self.ctx.jobs._jobs["job1"].proc
        time.sleep(0.3)
        self.ctx.jobs.shutdown()
        assert proc.poll() is not None
        with pytest.raises(ProcessLookupError):
            os.killpg(proc.pid, 0)

    def test_default_command_matches_the_real_cli(self):
        cmd = ui.default_job_command("/s/s1", "/s/intrinsics.yaml")
        assert cmd[:3] == ["nice", "-n", "19"] and cmd[3:7] == ["ros2", "run", "rebel_dual_arm_calibration",
                                                                  "calibrate_intrinsics"]
        assert cmd[7:] == ["/s/s1", "--out", "/s/intrinsics.yaml"]


class TestStopIndependence(Base):
    @gen_test(timeout=20)
    async def test_stop_while_capture_is_blocked_and_action_busy(self):
        self.bridge.block = threading.Event()
        base = self.get_url("")
        cap = asyncio.ensure_future(self.http_client.fetch(base + "/api/capture", method="POST", body="{}",
                                                           headers=self.hdr()))
        await asyncio.sleep(0.3)
        # the recorder action is in flight: another one is refused immediately, STOP and state are answered
        r = await self.http_client.fetch(base + "/api/touch", method="POST", raise_error=False, headers=self.hdr(),
                                         body=json.dumps({"arm": "left", "point_id": 1}))
        assert r.code == 409
        t0 = time.time()
        r = await self.http_client.fetch(base + "/api/jog/stop", method="POST", body="{}", headers=self.hdr())
        assert r.code == 200 and time.time() - t0 < 1.0 and self.jog.calls == [("stop",)]
        assert (await self.http_client.fetch(base + "/api/state", headers=self.hdr())).code == 200
        self.bridge.block.set()
        assert json.loads((await cap).body)["ok"] is True
        r = await self.http_client.fetch(base + "/api/capture", method="POST", body="{}", headers=self.hdr())
        assert r.code == 200                                               # lock released after the action


OTHER_PAGE = "page-other9"


class TestAutoCapture(Base):
    def setUp(self):
        super().setUp()
        self.now = [1000.0]
        self.ctx.auto.clock = lambda: self.now[0]
        self.other = {"X-Token": TOKEN, "X-Page-Id": OTHER_PAGE}

    def ac(self, headers=None):
        return self.j(self.get("/api/state", headers=headers or self.hdr()))["auto_capture"]

    def start(self, deg=2.0, headers=None):
        return self.post("/api/auto_capture/start", {"min_change_deg": deg}, headers=headers)

    def tick(self, dt=0.5):
        self.now[0] += dt
        self.ctx.auto.tick()

    def test_state_has_the_auto_capture_field_and_defaults(self):
        a = self.ac()
        assert a["on"] is False and a["owner_is_you"] is False and a["state"] == "off"
        assert {"message", "attempts", "captured", "skipped", "gave_up", "min_change_deg"} <= set(a)
        assert a["min_change_deg"] == 2.0

    def test_auth_origin_and_page_id(self):
        port = self.get_http_port()
        assert self.post("/api/auto_capture/start", headers=self.hdr(Origin="http://evil.example")).code == 403
        assert self.post("/api/auto_capture/stop", headers=self.hdr(Origin="http://evil.example")).code == 403
        assert self.start(headers=self.hdr(Origin=f"http://127.0.0.1:{port}")).code == 200
        assert self.post("/api/auto_capture/stop", headers=self.hdr(Origin=f"http://127.0.0.1:{port}")).code == 200
        h = {"X-Token": TOKEN}
        assert self.start(headers=h).code == 400 and self.post("/api/auto_capture/stop", headers=h).code == 400
        assert self.bridge.calls == [] and not self.ac()["on"]

    def test_min_change_must_be_1_2_or_5(self):
        for bad in (0, 3, 2.5, "2", True, None):
            r = self.post("/api/auto_capture/start", {"min_change_deg": bad})
            assert r.code == 400, bad
        for ok in (1, 2, 5, 5.0):
            assert self.start(ok).code == 200
            assert self.ac()["min_change_deg"] == float(ok)
            self.post("/api/auto_capture/stop")
        assert self.post("/api/auto_capture/start", {}).code == 200            # default 2 deg
        assert self.ac()["min_change_deg"] == 2.0

    def test_start_stop_and_ownership(self):
        r = self.start(5)
        assert r.code == 200 and self.j(r)["ok"] and self.j(r)["auto_capture"]["on"]
        mine, theirs = self.ac(), self.ac(self.other)
        assert mine["on"] and mine["owner_is_you"] and theirs["on"] and not theirs["owner_is_you"]
        assert mine["min_change_deg"] == 5.0 and mine["state"] == "waiting"
        r = self.start(2, headers=self.other)
        assert r.code == 409 and "another page" in self.j(r)["message"]
        r = self.post("/api/auto_capture/stop", headers=self.other)
        assert r.code == 409 and self.ac()["on"]
        r = self.post("/api/auto_capture/stop")
        assert r.code == 200 and not self.ac()["on"]
        assert self.start(2, headers=self.other).code == 200                    # free again
        assert self.jog.calls == []                                             # it never touches the jog / robot

    def test_captures_a_new_pose_through_the_capture_trigger_and_skips_the_same_pose(self):
        self.start()
        self.tick()
        assert self.bridge.calls == [("trigger", "capture")]
        a = self.ac()
        assert a["state"] == "captured" and a["message"] == "captured (1 this run)" and a["captured"] == 1 and a["attempts"] == 1
        assert a["events"][-1]["message"] == "capture done" and a["events"][-1]["level"] == "ok"
        for _ in range(5):
            self.tick()
        assert self.bridge.calls == [("trigger", "capture")]                    # same pose: no second capture
        self.jog.q["joint1"] = 0.2                                              # 11.5 deg
        self.tick()
        assert len(self.bridge.calls) == 2 and self.ac()["message"] == "captured (2 this run)"
        assert not any(c[0] == "params" for c in self.bridge.calls)             # no parameter changes either

    def test_waits_while_the_arm_moves_or_a_jog_step_runs(self):
        self.start()
        self.jog.spread = 0.01
        self.tick()
        assert self.bridge.calls == [] and self.ac()["state"] == "moving"
        self.jog.spread = 0.0
        self.jog.busy = True
        self.tick()
        assert self.bridge.calls == [] and self.ac()["state"] == "paused"
        self.jog.busy = False
        self.tick()
        assert self.bridge.calls == [] and self.ac()["state"] == "settling"
        self.tick(2.0)
        assert self.bridge.calls == [("trigger", "capture")]

    def test_capture_calls_are_serialised_with_manual_captures(self):
        self.start()
        assert self.ctx.action_lock.acquire(blocking=False)                      # a manual action is running
        try:
            self.tick()
            assert self.bridge.calls == [] and self.ac()["attempts"] == 0
        finally:
            self.ctx.action_lock.release()
        # the other way round: a manual capture while the automatic one is in flight is refused with 409
        self.bridge.block = threading.Event()
        t = threading.Thread(target=self.tick, daemon=True)
        t.start()
        for _ in range(100):
            if self.bridge.calls:
                break
            time.sleep(0.02)
        assert self.bridge.calls == [("trigger", "capture")]
        assert self.post("/api/capture").code == 409
        assert self.ac()["state"] == "capturing"                                 # state stays readable meanwhile
        self.bridge.block.set()
        t.join(5)
        assert self.ac()["captured"] == 1 and self.post("/api/capture").code == 200

    def test_heartbeat_comes_from_the_owner_polling_state(self):
        self.start()
        for _ in range(4):
            self.now[0] += 2.0
            self.ac()                                                           # the owner polls: stays alive
            self.ctx.auto.tick()
        assert self.ac()["on"]
        self.ac(self.other)                                                     # another page polling does not count
        self.now[0] += 6.0
        self.ctx.auto.tick()
        a = self.ac()
        assert not a["on"] and "stopped polling" in a["message"]

    def test_recorder_disappearing_stops_it(self):
        self.start()
        self.bridge.status = None
        self.tick()
        a = self.ac()
        assert not a["on"] and "recorder not available" in a["message"]
        assert self.start().code == 409

    def test_refused_when_the_jog_module_cannot_report_joints(self):
        ctx = ui.Context(self.bridge, ui.UnavailableJog("no ros"), self.pipeline, self.auth, "dual", JobManager())
        try:
            ok, msg = ctx.auto.start(PAGE, 2.0)
            assert not ok and "not available" in msg
        finally:
            ctx.shutdown()

    def test_periodic_submit_runs_one_tick_at_a_time(self):
        self.start()
        self.ctx.submit_auto_tick()
        self.ctx._auto_tick_future.result(5)
        assert self.bridge.calls == [("trigger", "capture")]
        self.ctx.shutdown()
        self.ctx.submit_auto_tick()                                              # pool is closed: no exception


class TestSuggest(Base):
    """The "Suggested pose" endpoints with a SuggestManager built on fakes (no ROS, no engine)."""
    mode = "single_left"

    def setUp(self):
        super().setUp()
        from test_ui_suggest import FakeEngine, FakeMoveIt, samples
        from rebel_dual_arm_calibration.ui import suggest as sg
        self.moveit, self.engine = FakeMoveIt(), FakeEngine()
        self.ctx.suggest = sg.SuggestManager(
            self.mode, self.ctx.suggest.cfgs, self.jog, lambda: "/tmp/uiback_session/s1", moveit=self.moveit,
            engine=self.engine, load_samples=lambda d: samples(8))
        self.ctx.suggest.jog = self.jog

    def compute_ready(self):
        r = self.post("/api/suggest/compute", {})
        assert r.code == 200 and self.j(r)["ok"], r.body
        for _ in range(200):
            st = self.j(self.get("/api/state"))["suggest"]
            if st["state"] != "computing":
                return st
            time.sleep(0.02)
        raise AssertionError("still computing")

    def test_auth_origin_and_page_id(self):
        for a in ("compute", "select", "send_rviz", "plan", "move"):
            r = self.post(f"/api/suggest/{a}", headers=self.hdr(Origin="http://evil.example"))
            assert r.code == 403, a
        h = {"X-Token": TOKEN}
        assert self.post("/api/suggest/plan", {"index": 0}, headers=h).code == 400
        assert self.post("/api/suggest/move", {"plan_id": "x"}, headers=h).code == 400
        assert self.fetch("/api/suggest/compute", method="POST", body="[1]", headers=self.hdr()).code == 400
        assert self.jog.calls == [] and self.moveit.published == []

    def test_state_field_before_and_after(self):
        s = self.j(self.get("/api/state"))["suggest"]
        assert s["state"] == "idle" and s["items"] == [] and s["selected"] is None and s["supported"] is True
        assert {"state", "message", "based_on_samples", "items", "selected", "stats"} <= set(s)
        st = self.compute_ready()
        assert st["state"] == "ready" and len(st["items"]) == 5 and st["selected"] == 0 and st["based_on_samples"] == 8
        assert {"rank", "description", "gains", "centre_px", "outline_px", "q_deg"} == set(st["items"][0])

    def test_refusals(self):
        self.engine.n = 5
        r = self.post("/api/suggest/compute", {"arm": "middle"})
        assert r.code == 400
        self.ctx.suggest.session_dir_fn = lambda: "/x"
        self.ctx.suggest._load = lambda d: []
        r = self.post("/api/suggest/compute", {})
        assert r.code == 400 and "at least 6" in self.j(r)["message"]
        assert self.j(self.get("/api/state"))["suggest"]["state"] == "error"
        assert self.post("/api/suggest/select", {"index": 0}).code == 409          # nothing computed
        assert self.post("/api/suggest/send_rviz", {"index": 0}).code == 409
        self.ctx.suggest._load = lambda d: [__import__("types").SimpleNamespace(arm="left")] * 8
        self.jog.busy = True
        r = self.post("/api/suggest/compute", {})
        assert r.code == 409 and "busy" in self.j(r)["message"]
        self.jog.busy = False
        self.compute_ready()
        assert self.post("/api/suggest/select", {"index": 9}).code == 400
        assert self.post("/api/suggest/select", {"index": "1"}).code == 400
        assert self.post("/api/suggest/move", {"plan_id": "plan7"}).code == 409    # no stored plan yet
        assert not any(c[0] == "execute_move" for c in self.jog.calls)

    def test_select_and_send_rviz_publish_one_goal_and_never_move(self):
        self.compute_ready()
        r = self.post("/api/suggest/select", {"index": 2})
        assert r.code == 200 and self.j(r)["selected"] == 2
        assert self.j(self.get("/api/state"))["suggest"]["selected"] == 2
        r = self.post("/api/suggest/send_rviz", {"index": 1})
        assert r.code == 200 and "Allow External Comm." in self.j(r)["message"]
        assert len(self.moveit.published) == 1
        names, pos = self.moveit.published[0]
        assert names == [f"joint{i}" for i in range(1, 7)] and len(pos) == 6
        assert not any(c[0] in ("plan_move", "execute_move", "step") for c in self.jog.calls)

    def test_plan_then_move_with_the_page_id(self):
        self.compute_ready()
        r = self.post("/api/suggest/plan", {"index": 1})
        assert r.code == 200 and self.j(r)["plan_id"] == "plan7" and self.j(r)["index"] == 1
        plan_calls = [c for c in self.jog.calls if c[0] == "plan_move"]
        assert plan_calls[0][1:3] == (PAGE, "left") and len(plan_calls[0][3]) == 6
        assert self.j(self.get("/api/state"))["suggest"]["plan"]["plan_id"] == "plan7"
        r = self.post("/api/suggest/move", {"plan_id": "plan7"})
        assert r.code == 200 and self.j(r) == {"ok": True, "message": "moving"}
        assert self.jog.calls[-1] == ("execute_move", PAGE, "plan7")
        assert self.post("/api/suggest/move", {"plan_id": "plan7"}).code == 409    # the plan is consumed

    def test_stop_is_still_answered_while_a_search_runs(self):
        self.engine.gate = threading.Event()
        assert self.post("/api/suggest/compute", {}).code == 200
        assert self.j(self.get("/api/state"))["suggest"]["state"] == "computing"
        assert self.post("/api/suggest/compute", {}).code == 409
        t0 = time.time()
        assert self.post("/api/jog/stop").code == 200 and time.time() - t0 < 1.0
        self.engine.gate.set()

    def test_image_overlay_draws_the_selected_outline(self):
        from test_targets import _scene_with_boards
        specs, table = load_targets_all(default_targets_path())
        gray, _ = _scene_with_boards({"left": specs["left"], "table": table})
        self.bridge.frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        assert self.pipeline.process_once() is True
        h, w = self.bridge.frame.shape[:2]

        def magenta(url):
            r = self.get(url)
            assert r.code == 200
            img = cv2.imdecode(np.frombuffer(r.body, np.uint8), cv2.IMREAD_COLOR)
            return int(((img[:, :, 0] > 200) & (img[:, :, 1] < 80) & (img[:, :, 2] > 200)).sum()), img.shape
        assert magenta("/api/image.jpg?w=480&overlay=none&suggest=1")[0] == 0         # nothing selected yet
        self.compute_ready()
        n0, _ = magenta("/api/image.jpg?w=480&overlay=none")
        n1, shape = magenta("/api/image.jpg?w=480&overlay=none&suggest=1")
        assert n0 == 0 and n1 > 50 and shape[1] == 480
        assert magenta("/api/image.jpg?w=480&overlay=detect&suggest=0")[0] == 0

    def test_dual_mode_says_single_arm_only(self):
        self.ctx.suggest = ui.Context(self.bridge, self.jog, self.pipeline, self.auth, "dual").suggest
        s = self.j(self.get("/api/state"))["suggest"]
        assert s["supported"] is False and "single-arm" in s["message"]
        r = self.post("/api/suggest/compute", {})
        assert r.code == 400 and "single-arm" in self.j(r)["message"]


def test_overlay_draws_in_the_scaled_position():
    from rebel_dual_arm_calibration.ui.overlay import ImagePipeline
    frame = np.zeros((1080, 1920, 3), np.uint8)
    pipe = ImagePipeline(lambda: (frame, 1, time.monotonic()), lambda: TargetDetector({}))
    pipe.process_once()
    plain = cv2.imdecode(np.frombuffer(pipe.render_jpeg(960, "none"), np.uint8), cv2.IMREAD_COLOR)
    drawn = cv2.imdecode(np.frombuffer(pipe.render_jpeg(
        960, "none", suggest={"outline_px": [(400, 300), (1400, 300), (1400, 800), (400, 800)], "label": "suggested #2"}),
        np.uint8), cv2.IMREAD_COLOR)
    assert plain.max() == 0
    ys, xs = np.nonzero((drawn[:, :, 0] > 150) & (drawn[:, :, 2] > 150) & (drawn[:, :, 1] < 100))
    assert xs.min() == pytest.approx(200, abs=3) and xs.max() == pytest.approx(700, abs=3)      # 400/2 .. 1400/2
    assert ys.max() == pytest.approx(400, abs=3)                                                # 800/2
    assert pipe.render_jpeg(960, "none", suggest={"outline_px": "junk"}) is not None


def test_static_dir_found_from_source_or_share(tmp_path):
    assert os.path.isdir(os.path.dirname(ui.find_static_dir()))


# ---------------------------------------------------------------------------------------------------
# safety review: motion ingress (F3) and the plan binding (F7)

class GatedJog(FakeJog):
    """FakeJog with the ingress gate / STOP epoch of the real jog controller."""

    def __init__(self):
        super().__init__()
        self.epoch, self.tokens, self.queued, self.released = 0, 0, None, []
        self.busy_msg = None
        self.hold = None                       # threading.Event: step / plan_move wait for it

    def reserve_motion(self):
        self.calls.append(("reserve",))
        if self.busy_msg:
            return {"ok": False, "message": self.busy_msg}
        if self.queued is not None:
            return {"ok": False, "message": "busy: another motion request is already queued"}
        self.tokens += 1
        self.queued = self.tokens
        return {"ok": True, "message": "reserved", "epoch": self.epoch, "token": self.tokens}

    def release_motion(self, token):
        if self.queued == token:
            self.queued = None
        self.released.append(token)

    def stop(self):
        self.epoch += 1
        return super().stop()

    def _void(self, epoch):
        return epoch is not None and epoch != self.epoch

    def step(self, page_id, arm, mode, axis, sign, size, frame="world", epoch=None):
        if self.hold is not None:
            self.hold.wait(10)
        if self._void(epoch):
            return {"ok": False, "id": 9, "message": "STOP was pressed after this request was sent: request void"}
        return super().step(page_id, arm, mode, axis, sign, size, frame)

    def plan_move(self, page_id, arm, q_goal, epoch=None):
        if self.hold is not None:
            self.hold.wait(10)
        if self._void(epoch):
            return {"ok": False, "plan_id": None, "message": "STOP was pressed after this request was sent: request void"}
        return super().plan_move(page_id, arm, q_goal)

    def execute_move(self, page_id, plan_id, epoch=None):
        if self._void(epoch):
            return {"ok": False, "message": "STOP was pressed after this request was sent: request void"}
        return super().execute_move(page_id, plan_id)


STEP_BODY = {"arm": "left", "mode": "joint", "axis": 1, "sign": 1, "size": 1}


class TestMotionIngress(Base):
    mode = "single_left"

    def setUp(self):
        super().setUp()
        self.jog = self.ctx.jog = GatedJog()

    def test_f3_busy_jog_is_refused_at_ingress_before_anything_is_queued(self):
        self.jog.busy_msg = "busy: a step or move on arm 'left' is still running"
        r = self.post("/api/jog/step", STEP_BODY)
        assert r.code == 409 and "still running" in self.j(r)["message"] and self.j(r)["ok"] is False
        assert not any(c[0] == "step" for c in self.jog.calls)
        self.jog.busy_msg = "fault: the last stop was not confirmed"
        assert self.post("/api/jog/step", STEP_BODY).code == 409
        assert self.post("/api/suggest/plan", {"index": 0}).code in (409, 400, 503)       # (no suggestion: still no jog call)
        assert not any(c[0] in ("step", "plan_move", "execute_move") for c in self.jog.calls)

    def test_f3_epoch_and_release_reach_the_worker_and_the_slot_is_freed(self):
        r = self.post("/api/jog/step", STEP_BODY)
        assert r.code == 200 and self.j(r)["ok"] is True
        assert self.jog.released == [1] and self.jog.queued is None
        assert self.post("/api/jog/step", STEP_BODY).code == 200                           # the slot is reusable

    def test_f3_a_failing_request_still_frees_the_slot(self):
        self.jog.step = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
        assert self.post("/api/jog/step", STEP_BODY).code == 500
        assert self.jog.queued is None

    def test_f3_missing_fields_do_not_reserve(self):
        assert self.post("/api/jog/step", {"arm": "left"}).code == 400
        assert not any(c[0] == "reserve" for c in self.jog.calls)

    @gen_test(timeout=20)
    async def test_f3_second_request_while_one_is_queued_is_refused_and_stop_voids_the_queued_one(self):
        self.jog.hold = threading.Event()
        base = self.get_url("")
        first = asyncio.ensure_future(self.http_client.fetch(base + "/api/jog/step", method="POST",
                                                             body=json.dumps(STEP_BODY), headers=self.hdr()))
        await asyncio.sleep(0.3)                                 # it holds the single worker
        r = await self.http_client.fetch(base + "/api/jog/step", method="POST", body=json.dumps(STEP_BODY),
                                         headers=self.hdr(), raise_error=False)
        assert r.code == 409 and "already queued" in json.loads(r.body)["message"]
        r = await self.http_client.fetch(base + "/api/jog/stop", method="POST", body="{}", headers=self.hdr())
        assert r.code == 200                                     # STOP is answered at once, it never queues
        self.jog.hold.set()
        res = json.loads((await first).body)
        assert res["ok"] is False and "STOP was pressed after this request was sent" in res["message"]
        assert not any(c[0] == "step" for c in self.jog.calls)   # the queued step never reached the controller
        assert self.jog.queued is None
        r = await self.http_client.fetch(base + "/api/jog/step", method="POST", body=json.dumps(STEP_BODY),
                                         headers=self.hdr())
        assert json.loads(r.body)["ok"] is True                  # a request sent after the STOP is fine


class TestMotionIngressSuggest(TestSuggest):
    def setUp(self):
        super().setUp()
        self.jog = self.ctx.jog = self.ctx.suggest.jog = GatedJog()

    def test_f3_plan_and_move_are_gated_and_carry_the_epoch(self):
        self.compute_ready()
        self.jog.busy_msg = "busy: arm 'left' is stopping"
        r = self.post("/api/suggest/plan", {"index": 0})
        assert r.code == 409 and "stopping" in self.j(r)["message"]
        assert not any(c[0] == "plan_move" for c in self.jog.calls)
        self.jog.busy_msg = None
        r = self.post("/api/suggest/plan", {"index": 0})
        assert r.code == 200 and self.j(r)["plan_id"] == "plan7" and self.jog.queued is None

    def test_f3_a_stop_between_ingress_and_the_worker_voids_plan_and_move(self):
        self.compute_ready()
        orig = self.jog.reserve_motion

        def reserve_then_stop():
            res = orig()
            self.jog.epoch += 1                                  # STOP pressed right after the request was ingressed
            return res
        self.jog.reserve_motion = reserve_then_stop
        r = self.post("/api/suggest/plan", {"index": 0})
        assert r.code == 200 and self.j(r)["ok"] is False and "STOP was pressed" in self.j(r)["message"]
        assert not any(c[0] == "plan_move" for c in self.jog.calls)
        self.jog.reserve_motion = orig
        assert self.post("/api/suggest/plan", {"index": 0}).code == 200
        self.jog.reserve_motion = reserve_then_stop
        r = self.post("/api/suggest/move", {"plan_id": "plan7"})
        assert "STOP was pressed" in self.j(r).get("message", "")
        assert not any(c[0] == "execute_move" for c in self.jog.calls)

    def test_f7_move_with_another_plan_id_is_refused_and_nothing_moves(self):
        self.compute_ready()
        assert self.post("/api/suggest/plan", {"index": 0}).code == 200
        r = self.post("/api/suggest/move", {"plan_id": "plan-shown-earlier"})
        assert r.code == 409 and "no stored plan with this id" in self.j(r)["message"]
        assert not any(c[0] == "execute_move" for c in self.jog.calls)
        r = self.post("/api/suggest/move", {"plan_id": "plan7"})              # the stored id still works
        assert r.code == 200 and self.jog.calls[-1] == ("execute_move", PAGE, "plan7")


# ---------------------------------------------------------------------------------------------------
# approach assist (hover): the endpoints with a HoverManager built on fakes (test_ui_hover.py)

class TestHover(Base):
    mode = "single_left"

    def setUp(self):
        super().setUp()
        import pathlib
        import tempfile
        import test_ui_hover as th
        self.th = th
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.gated = self.jog = self.ctx.jog = GatedJog()
        self.mgr = th.make(self.tmp)
        self.fk = self.mgr.fakes
        gated = self.gated

        class EpochView:                                           # core.stop_epoch mirrors the gated jog's STOP epoch
            stop_epoch = property(lambda _s: gated.epoch)
        self.fk.jog.core = EpochView()
        self.ctx.hover = self.mgr

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)
        super().tearDown()

    def detect(self):
        r = self.post("/api/hover/compute", {})
        assert r.code == 200 and self.j(r)["ok"], r.body
        return self.j(self.get("/api/state"))["hover"]

    def reach(self, **tcp):
        """Plan hover + Move for point 101 through the endpoints (the descend interlock), TCP at `tcp_pose(**tcp)`."""
        assert self.post("/api/hover/prepare", {"point_id": 101}).code == 200
        assert self.post("/api/hover/move", {"plan_id": "plan-1"}).code == 200
        T = self.th.tcp_pose(**tcp)
        self.fk.jog.T_tcp = T
        self.mgr._pending_move["T"] = T.copy()

    def hover_calls(self, *names):
        return [c for c in self.fk.jog.calls if c[0] in names]

    def test_auth_origin_page_id_and_bodies(self):
        for a in ("compute", "prepare", "send_rviz", "move", "descend", "record_offset"):
            r = self.post(f"/api/hover/{a}", headers=self.hdr(Origin="http://evil.example"))
            assert r.code == 403, a
            assert self.fetch(f"/api/hover/{a}", method="POST", body="[1]", headers=self.hdr()).code == 400, a
        h = {"X-Token": TOKEN}
        for a, body in (("prepare", {"point_id": 101}), ("move", {"plan_id": "x"}), ("descend", {"point_id": 101})):
            assert self.post(f"/api/hover/{a}", body, headers=h).code == 400, a              # X-Page-Id is needed
        for a in ("prepare", "send_rviz", "descend", "record_offset"):
            for bad in ({}, {"point_id": "101"}, {"point_id": True}, {"point_id": 1.5}):
                assert self.post(f"/api/hover/{a}", bad).code == 400, (a, bad)
        assert self.post("/api/hover/prepare", {"point_id": 101, "stage": 5}).code == 400
        assert self.post("/api/hover/compute", {"arm": "middle"}).code == 400
        assert self.hover_calls("plan_move", "execute_move", "descend") == [] and self.fk.moveit.published == []
        assert not any(c[0] == "reserve" for c in self.gated.calls)                         # bad bodies never reserve

    def test_state_field_before_and_after_detection(self):
        h = self.j(self.get("/api/state"))["hover"]
        assert h["supported"] is True and h["state"] == "idle" and h["points"] == [] and h["arm"] is None
        assert {"state", "message", "arm", "calib_samples", "points", "test_summary"} <= set(h)
        h = self.detect()
        assert h["state"] == "ready" and h["arm"] == "left" and h["calib_samples"] == 8
        pts = {p["id"]: p for p in h["points"]}
        assert set(pts) == {1, 2, 101, 102, 113}
        assert {"id", "name", "source", "detected", "quality", "pred_base_mm", "uv", "last_offset"} <= set(pts[101])
        assert pts[101]["detected"] and not pts[113]["detected"] and pts[113]["pred_base_mm"] is None

    def test_refusals_are_json_with_the_http_status(self):
        r = self.post("/api/hover/prepare", {"point_id": 101})
        assert r.code == 409 and "Detect points" in self.j(r)["message"] and self.j(r)["ok"] is False
        assert self.post("/api/hover/send_rviz", {"point_id": 101}).code == 409
        assert self.post("/api/hover/move", {"plan_id": "plan-1"}).code == 409
        assert self.post("/api/hover/descend", {"point_id": 101}).code == 409
        assert self.post("/api/hover/record_offset", {"point_id": 101}).code == 409
        self.mgr.scene_fn = None
        r = self.post("/api/hover/compute", {})
        assert r.code == 503 and "target definitions" in self.j(r)["message"]
        assert self.hover_calls("plan_move", "execute_move", "descend") == []

    def test_prepare_send_rviz_move_flow(self):
        self.detect()
        r = self.post("/api/hover/prepare", {"point_id": 101})
        assert r.code == 200 and self.j(r)["ok"] and self.j(r)["plan_id"] == "plan-1" and "IK pose" in self.j(r)["message"]
        plan = self.hover_calls("plan_move")[0]
        assert plan[1:3] == (PAGE, "left") and plan[4] == {"epoch": 0}
        assert self.gated.queued is None                                    # the ingress slot is freed
        st = self.j(self.get("/api/state"))["hover"]
        assert st["prepared"]["point_id"] == 101 and st["plan"]["plan_id"] == "plan-1" and "q" not in st["prepared"]
        r = self.post("/api/hover/send_rviz", {"point_id": 101})
        assert r.code == 200 and "Allow External Comm." in self.j(r)["message"]
        assert len(self.fk.moveit.published) == 1 and self.fk.moveit.published[0][0] == [f"joint{i}" for i in range(1, 7)]
        assert self.post("/api/hover/move", {"plan_id": "plan-other"}).code == 409
        r = self.post("/api/hover/move", {"plan_id": "plan-1"})
        assert r.code == 200 and self.j(r) == {"ok": True, "id": 5, "message": "moving"}
        assert self.hover_calls("execute_move") == [("execute_move", PAGE, "plan-1", {"epoch": 0})]
        assert self.post("/api/hover/move", {"plan_id": "plan-1"}).code == 409        # consumed

    def test_descend_endpoint(self):
        self.detect()
        r = self.post("/api/hover/descend", {"point_id": 101})
        assert r.code == 409 and "go to the pre-hover pose first (Plan hover -> Move)" in self.j(r)["message"]   # no hover move yet
        self.reach(lateral_mm=1.0, height_mm=60.0)
        r = self.post("/api/hover/descend", {"point_id": 101})
        assert r.code == 200 and self.j(r)["ok"] and "descending 40.0 mm" in self.j(r)["message"]
        call = self.hover_calls("descend")[0]
        assert call[1:3] == (PAGE, "left") and call[4] == {"epoch": 0}
        r = self.post("/api/hover/descend", {"point_id": 101, "height_mm": 30})
        assert r.code == 200 and "descending 30.0 mm" in self.j(r)["message"] and "30 mm above" in self.j(r)["message"]
        assert self.post("/api/hover/descend", {"point_id": 101, "height_mm": 15}).code == 400
        assert self.post("/api/hover/descend", {"point_id": 101, "height_mm": "20"}).code == 400
        assert self.post("/api/hover/descend", {"point_id": 101, "height_mm": True}).code == 400
        self.fk.jog.T_tcp = self.th.tcp_pose(lateral_mm=9.0)
        r = self.post("/api/hover/descend", {"point_id": 101})
        assert r.code == 409 and "pre-hover pose first" in self.j(r)["message"]
        assert len(self.hover_calls("descend")) == 2
        assert self.gated.queued is None

    def test_motion_endpoints_are_gated_at_ingress_but_the_others_are_not(self):
        self.detect()
        self.fk.jog.T_tcp = self.th.tcp_pose()
        self.gated.busy_msg = "busy: a step or move on arm 'left' is still running"
        for a, body in (("prepare", {"point_id": 101}), ("move", {"plan_id": "plan-1"}), ("descend", {"point_id": 101})):
            r = self.post(f"/api/hover/{a}", body)
            assert r.code == 409 and "still running" in self.j(r)["message"], a
        assert self.hover_calls("plan_move", "execute_move", "descend") == [] and self.fk.moveit.ik_calls == []
        assert self.post("/api/hover/compute", {}).code == 200                               # no motion: not gated
        self.gated.busy_msg = None
        assert self.post("/api/hover/send_rviz", {"point_id": 101}).code == 409              # (nothing prepared) but not gated
        self.gated.busy_msg = "fault: the last stop was not confirmed"
        assert self.post("/api/hover/record_offset", {"point_id": 101}).code == 200          # no motion: not gated
        assert self.gated.queued is None

    def test_a_stop_between_ingress_and_the_worker_voids_the_request(self):
        self.detect()
        self.reach()
        self.fk.moveit.ik_calls.clear()
        n_plans = len(self.hover_calls("plan_move"))
        orig = self.gated.reserve_motion

        def reserve_then_stop():
            res = orig()
            self.gated.epoch += 1                                  # STOP pressed right after the request was ingressed
            return res
        self.gated.reserve_motion = reserve_then_stop
        r = self.post("/api/hover/prepare", {"point_id": 101})
        assert r.code == 409 and "STOP was pressed" in self.j(r)["message"]
        assert self.fk.moveit.ik_calls == [] and len(self.hover_calls("plan_move")) == n_plans
        self.gated.reserve_motion = orig
        assert self.post("/api/hover/prepare", {"point_id": 101}).code == 200                # a request after the STOP is fine
        assert self.gated.queued is None
        # a STOP after the hover move was started also voids the descend interlock (here: a new plan did as well)
        self.fk.jog.T_tcp = self.th.tcp_pose()
        r = self.post("/api/hover/descend", {"point_id": 101})
        assert r.code == 409 and "pre-hover pose first" in self.j(r)["message"]
        assert self.hover_calls("descend") == []

    def test_record_offset_endpoint_and_summary_in_the_state(self):
        self.detect()
        T = np.eye(4)
        T[:3, 3] = self.th.P_BASE + [0.011, 0.002, 0.0]
        self.fk.jog.T_tcp = T
        r = self.post("/api/hover/record_offset", {"point_id": 101})
        assert r.code == 200 and self.j(r)["ok"] and self.j(r)["offset"]["dx_mm"] == pytest.approx(1.0)
        h = self.j(self.get("/api/state"))["hover"]
        assert h["test_summary"]["n"] == 1 and h["test_summary"]["points"][0]["point_id"] == 101
        assert os.path.exists(os.path.join(str(self.tmp), "hover_test.jsonl"))

    def test_capture_touch_reply_carries_the_predicted_vs_touched_offset(self):
        self.detect()
        mgr = self.mgr

        def trigger(name, timeout_s=15.0):
            self.bridge.calls.append(("trigger", name))
            if name == "capture_touch":
                mgr.touches.append(self.th.touch("left", 101, p=np.array([0.0019, -0.0012, 0.0001])))
            return True, "touch recorded"
        self.bridge.trigger = trigger
        r = self.post("/api/touch", {"arm": "left", "point_id": 101})
        out = self.j(r)
        assert r.code == 200 and out["ok"] and out["hover_offset"]["dx_mm"] == pytest.approx(1.9, abs=1e-6)
        assert out["message"].startswith("touch recorded; predicted vs touched: 2.2 mm lateral (x +1.9, y -1.2, above paper +0.1")
        pts = {p["id"]: p for p in self.j(self.get("/api/state"))["hover"]["points"]}
        assert pts[101]["last_offset"]["dy_mm"] == pytest.approx(-1.2, abs=1e-6)
        assert os.path.exists(os.path.join(str(self.tmp), "hover_log.jsonl"))
        r = self.post("/api/touch", {"arm": "left", "point_id": 2})            # a point without a prediction: plain reply
        assert self.j(r) == {"ok": True, "message": "touch recorded"}

    def test_touch_list_shows_hover_sheet_points_the_recorder_does_not_list(self):
        tl = self.j(self.get("/api/state"))["touch_list"]
        ids = [t["id"] for t in tl]
        assert ids == [1, 2, 101, 111, 102, 113]                                      # recorder's own ids first, no duplicates
        extra = {t["id"]: t for t in tl if t["kind"] == "hover"}
        assert set(extra) == {102, 113} and extra[113]["capturable"] is False and extra[113]["name"] == "OPT"
        assert all("capturable" not in t for t in tl if t["kind"] != "hover")
        # but a point the recorder does not list can still not be captured
        assert self.post("/api/touch", {"arm": "left", "point_id": 113}).code == 400
        assert ("trigger", "capture_touch") not in self.bridge.calls
        # driven by the recorder's list: once it lists 113 (as a check sticker) it is the recorder's entry
        self.bridge.status = dict(self.bridge.status, touch_list=self.bridge.status["touch_list"] + [{"id": 113, "kind": "check"}])
        tl = self.j(self.get("/api/state"))["touch_list"]
        assert [t["id"] for t in tl].count(113) == 1 and {t["id"]: t["kind"] for t in tl}[113] == "check"
        assert self.post("/api/touch", {"arm": "left", "point_id": 113}).code == 200
        self.bridge.status = None                                                      # no recorder status: no list at all
        assert self.j(self.get("/api/state"))["touch_list"] == []

    def test_touch_list_labels_hover_tile_points_and_groups_the_extra_ones(self):
        self.mgr = self.ctx.hover = self.th.tiles_mgr(self.tmp)
        tl = self.j(self.get("/api/state"))["touch_list"]
        by = {t["id"]: t for t in tl}
        assert [t["id"] for t in tl] == [1, 2, 101, 111, 102, 103, 104, 105, 106]          # the recorder's own ids, then the tile extras
        assert by[101]["tile"] == "tile A" and by[101]["tile_name"] == "A1" and by[101]["kind"] != "hover"   # a sticker that is on a tile
        assert "tile" not in by[1] and "tile" not in by[2]                                    # table board points are not hover points
        extra = {i: t for i, t in by.items() if t["kind"] == "hover"}
        assert set(extra) == {102, 103, 104, 105, 106}
        assert {i: t["tile"] for i, t in extra.items()} == {102: "tile A", 103: "tile A", 104: "tile B", 105: "tile B", 106: "tile B"}
        assert extra[105]["tile_name"] == "B2" and all(t["capturable"] is False for t in extra.values())
        assert self.post("/api/touch", {"arm": "left", "point_id": 105}).code == 400

    def test_state_reports_one_status_per_tile(self):
        self.mgr = self.ctx.hover = self.th.tiles_mgr(self.tmp)
        self.fk = self.mgr.fakes
        h = self.detect()
        assert set(h["status"]) == {"table", "tile A", "tile B"}
        pts = {p["id"]: p for p in h["points"]}
        assert pts[101]["source"] == "tile A" and pts[104]["source"] == "tile B" and not pts[104]["detected"]
        assert pts[104]["reason"] == "tile B: marker 14 not found (0 of 5 frames)"
        assert "tile B: marker 14 not found" in h["message"]

    def test_make_hover_scene_loads_the_tiles_of_the_targets_file(self):
        import os
        cfgdir = os.path.join(os.path.dirname(__file__), "..", "config")
        scene = ui.make_hover_scene(os.path.join(cfgdir, "targets.yaml"))()
        assert [s.source for s in scene.sheet_spec] == ["tile A", "tile B", "tile C"] and scene.table_spec is not None
        cat = self.mgr._catalog_of(scene)
        assert [c for c in cat if c[2] == "tile C"] == [(111, "C1", "tile C"), (112, "C2", "tile C"), (113, "C3", "tile C")]
        # a file with a legacy hover_sheet only keeps working: one spec, source "sheet"
        import yaml
        cfg = yaml.safe_load(open(os.path.join(cfgdir, "targets.yaml")))
        cfg.pop("hover_tiles")
        cfg["hover_sheet"] = {"marker_id": 13, "marker_length": 0.06, "points": {101: {"name": "TOP", "xy_mm": [0, 60]}}}
        path = os.path.join(str(self.tmp), "legacy.yaml")
        yaml.safe_dump(cfg, open(path, "w"))
        legacy = ui.make_hover_scene(path)()
        assert [s.source for s in legacy.sheet_spec] == ["sheet"]
        assert self.mgr._catalog_of(legacy)[-1] == (101, "TOP", "sheet")
        cfg.pop("hover_sheet")
        yaml.safe_dump(cfg, open(path, "w"))
        assert ui.make_hover_scene(path)().sheet_spec is None

    def test_default_manager_without_collaborators_refuses(self):
        ctx = ui.Context(self.bridge, self.jog, self.pipeline, self.auth, "single_left")
        assert ctx.hover.state()["supported"] is True and ctx.hover.state()["points"] == []
        dual = ui.Context(self.bridge, self.jog, self.pipeline, self.auth, "dual")
        assert dual.hover.state()["supported"] is False
        self.ctx.hover = dual.hover
        r = self.post("/api/hover/compute", {})
        assert r.code == 400 and "single-arm" in self.j(r)["message"]
        assert self.j(self.get("/api/state"))["hover"]["supported"] is False

    def test_image_overlay_draws_the_predicted_crosshairs(self):
        self.bridge.frame = np.zeros((480, 640, 3), np.uint8)
        assert self.pipeline.process_once() is True

        def colour(url, kind):
            r = self.get(url)
            assert r.code == 200
            img = cv2.imdecode(np.frombuffer(r.body, np.uint8), cv2.IMREAD_COLOR)
            b, g, rr = (img[:, :, i].astype(int) for i in range(3))
            if kind == "cyan":
                return int(((b > 200) & (g > 150) & (rr < 80)).sum())
            return int(((b < 80) & (g > 180) & (rr > 200)).sum())
        assert colour("/api/image.jpg?w=640&overlay=none&hover=1", "cyan") == 0          # nothing detected yet
        self.detect()
        assert colour("/api/image.jpg?w=640&overlay=none", "cyan") == 0                  # not requested
        n_plain = colour("/api/image.jpg?w=640&overlay=none&hover=1", "cyan")
        assert n_plain > 100 and colour("/api/image.jpg?w=640&overlay=none&hover=1", "yellow") == 0
        assert colour("/api/image.jpg?w=640&overlay=none&hover=1&sel=101", "yellow") > 50      # the selected one
        assert colour("/api/image.jpg?w=640&overlay=none&hover=1&sel=abc", "cyan") > 100      # a bad sel is ignored


def test_overlay_hover_crosshairs_are_drawn_at_the_scaled_position():
    from rebel_dual_arm_calibration.ui.overlay import ImagePipeline
    frame = np.zeros((1080, 1920, 3), np.uint8)
    pipe = ImagePipeline(lambda: (frame, 1, time.monotonic()), lambda: TargetDetector({}))
    pipe.process_once()
    ov = {"points": [{"id": 101, "name": "TOP", "uv": (1000.0, 400.0), "selected": False},
                     {"id": 5, "name": "x", "uv": (float("nan"), 1.0), "selected": True},
                     {"id": 6, "name": "far", "uv": (9999.0, 9999.0), "selected": False}, {"id": 7}]}
    img = cv2.imdecode(np.frombuffer(pipe.render_jpeg(960, "none", hover=ov), np.uint8), cv2.IMREAD_COLOR)
    cyan = (img[:, :, 0] > 200) & (img[:, :, 1] > 150) & (img[:, :, 2] < 80)
    ys, xs = np.nonzero(cyan[184:217, 484:517])                                      # the crosshair (the label is to its right)
    assert len(xs) > 15 and (xs.mean() + 484) == pytest.approx(500, abs=1.5) and (ys.mean() + 184) == pytest.approx(200, abs=1.5)
    ys, xs = np.nonzero(cyan)
    assert xs.min() > 470 and xs.max() < 700 and ys.min() > 170 and ys.max() < 260    # nothing else: NaN / far / empty points skipped
    assert pipe.render_jpeg(960, "none", hover={"points": "junk"}) is not None


STATIC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rebel_dual_arm_calibration", "ui", "static")


def test_f7_confirm_dialog_text_and_plan_binding_in_the_page():
    html = open(os.path.join(STATIC, "index.html")).read()
    assert "The knife has no collision model: keep the blade covered and the path clear." in html
    js = open(os.path.join(STATIC, "app.js")).read()
    assert "The knife has no collision model: keep the blade covered and the path clear." in js     # confirm() fallback
    i = js.index("let moveShown = null;")
    block = js[i:js.index("// ----------------------------------------------------------------- touches")]
    ok_handler = block[block.index("$('move-ok').addEventListener"):]
    assert "moveShown" in block and "const shown = moveShown;" in ok_handler
    assert "sgDoMove(shown.plan_id)" in ok_handler
    assert "sgFreshPlan" not in ok_handler                      # the OK handler never re-reads the current plan
    open_handler = block[block.index("$('sg-move').addEventListener"):block.index("$('move-cancel')")]
    assert "moveShown = { plan_id: plan.plan_id" in open_handler       # captured when the dialog OPENS


def test_suggest_card_shows_an_error_text_only_once():
    js = open(os.path.join(STATIC, "app.js")).read()
    i = js.index("// a finished search is reported once in the result line")
    block = js[i:js.index("sgLast = g ? st : null;")]
    assert "st !== 'error'" in block                           # the status line already carries the error text
    assert "rl.textContent === status" in block                # and an identical result line is cleared


def test_hover_page_contract():
    html = open(os.path.join(STATIC, "index.html")).read()
    js = open(os.path.join(STATIC, "app.js")).read()
    assert 'data-tab="hovertest"' in html and 'id="page-hovertest"' in html and 'id="hover-card-touches"' in html
    assert 'id="jog-slot-hovertest"' in html and 'id="img-hovertest"' in html
    for a in ("compute", "prepare", "send_rviz", "move", "descend", "record_offset"):
        assert f"/api/hover/{a}" in js, a
    assert "The knife has no collision model: keep the blade covered and the path clear." in html
    # click-to-select: within 25 displayed px of a predicted crosshair, scaled to camera pixels
    assert "hv-result" not in js and js.count("[data-hvres]") >= 3      # setResult() replaces className: never select by class
    assert "const HV_PICK_PX = 25;" in js and "function hvImageClick" in js and "naturalWidth" in js
    # the selected point goes to the overlay
    assert "'&hover=1' + (hvSelId() != null ? '&sel=' + hvSelId() : '')" in js
    # selectable final hover height, default 20 mm, sent with every descend
    assert "let hvHeight = 20;" in js and '<option value="10">10 mm</option><option value="20">20 mm</option><option value="30">30 mm</option>' in js
    assert "height_mm: hvHeight" in js and "'Plan hover'" in js
    # the operator hint after the hover pose is reached
    assert "Fine-tune with teleop (or the jog panel), release the keys, then Capture touch / Record offset." in js
    # the REAL confirm dialog is bound to the plan id shown when it opened (like Move there)
    i = js.index("$('hover-ok').addEventListener")
    ok_handler = js[i:js.index("// --------------------------------------------------------------- jog panel")]
    assert "const shown = hvMoveShown;" in ok_handler and "shown.plan_id" in ok_handler and "hvFreshPlan" not in ok_handler
    j = js.index("hvMoveShown = { plan_id: plan.plan_id")
    assert "hwKind(selArm) !== 'real'" in js[j - 400:j]


def test_hover_page_wording_and_state_driven_buttons():
    js = open(os.path.join(STATIC, "app.js")).read()
    # one shared height setting: a change updates every selector at once
    assert "$$('.hv-height').forEach((el) => { el.value = String(hvHeight); });" in js
    # "Move done: done: ..." -> the phase prefix of the jog message is dropped
    assert "replace(new RegExp('^' + phase + ':" in js
    # Descend only after Plan hover -> Move for THIS point, and never on the REAL robot without allow_real_move
    assert "g.reached_point === p.id" in js and "at(S, 'jog', 'allow_real_move') !== true" in js
    assert "Descend is disabled on the REAL robot (start the UI with allow_real_move:=true); use teleop instead" in js
    assert "(canDescend ? '' : ' disabled')" in js
    # the strip follows the arm: at the pre-hover pose / descending / at N mm, not the stale "Prepared ... no stored plan" line
    assert "At the pre-hover pose (" in js and "Descending to " in js and "mm above ' + where + '.</span> '" in js
    # hover tile points the recorder does not list appear in the Touches grid (one group per tile) but cannot be captured
    assert "['board', 'fit', 'check', 'hover']" in js and "notCapturable" in js
    assert "hoverGroups(items)" in js and "tile + ': ' + its.map((it) => it.id).join(', ')" in js
    # the hover rows come with one header per tile / board: "tile A: 101, 102, 103" and that source's status text
    assert "srcIds[srcName].join(', ')" in js and "srcStatus[srcName]" in js
    assert "const pts = hvPoints();" in js            # the same rows in both tabs (not filtered by a hard-coded id list)


def test_tcp_tab_page_contract():
    html = open(os.path.join(STATIC, "index.html")).read()
    js = open(os.path.join(STATIC, "app.js")).read()
    # "0. TCP" is the FIRST tab; the other tabs keep their numbers
    tabs = [t for t in html.split('<button class="tab"')[1:]]
    assert 'data-tab="tcp"' in tabs[0] and ">0. TCP<" in tabs[0]
    assert ">1. Setup<" in tabs[1] and ">2. Intrinsics (optional)<" in tabs[2] and ">5. Hover test<" in tabs[5]
    assert 'id="page-tcp"' in html and 'id="tcp-checklist"' in html and 'id="tcp-diff"' in html
    for a in ("checklist", "capture", "delete", "solve", "mechanical", "write"):
        assert f"/api/tcp/{a}" in js, a
    assert "role: 'fit'" in js and "role: 'validate'" in js
    # the page says that it never moves the robot and never edits the xacro, and gives the by-hand steps
    assert "never move the robot" in html and "never edits the xacro" in html
    assert "Servo teleop keyboard" in html and "recommended" in html and "jog panel" in html and 'id="jog-slot-tcp"' in html
    assert 'id="tcp-cap-why"' in html and "Capture is off: " in js and "tick the whole safety checklist first" in js
    assert "'✓' : '○'" in js and "not covered yet" in js                  # the pose plan: covered / not covered, clearly
    assert "Press Solve again." in js and "name === 'tcp'" in js
    assert "single-arm modes only" in js or "tcp-unsupported" in html
    # no jog panel, no motion calls from this tab
    block = js[js.index("// --------------------------------------------------------------------- TCP"):js.index("// ------------------------------------------------------------------- tabs")]
    assert "/api/jog" not in block and "/api/suggest" not in block and "/api/hover" not in block
    assert "tcp-cap-fit" in block and "window.confirm('Delete pose" in block
    assert "if (tab === 'tcp') renderTcp();" in js


def test_tcp_launch_argument_and_node_parameter():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    launch = open(os.path.join(here, "launch", "calibration_ui.launch.py")).read()
    assert "tcp_xacro_path" in launch
    src = open(ui.__file__).read()
    assert '("tcp_xacro_path", DEFAULT_XACRO_PATH)' in src
    from rebel_dual_arm_calibration.ui.tcp_ui import DEFAULT_XACRO_PATH
    assert DEFAULT_XACRO_PATH.endswith("src/igus_rebel_description/urdf/igus_rebel_xeg32_wrapper.urdf.xacro")
