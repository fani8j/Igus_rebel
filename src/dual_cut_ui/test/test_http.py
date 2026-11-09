"""HTTP layer: token, Host/Origin, routes, whitelist over HTTP, camera and detection endpoints (fake bridge, no ROS)."""
import json
import os
import threading
import time

import pytest

pytest.importorskip("tornado")
from tornado.testing import AsyncHTTPTestCase   # noqa: E402

from dual_cut_ui.auth import Auth   # noqa: E402
from dual_cut_ui.camera import CameraHub, DEFAULT_CAMERAS   # noqa: E402
from dual_cut_ui.server import Context, find_static_dir, make_app   # noqa: E402
from helpers import Env   # noqa: E402
from test_camera import synthetic_jpeg   # noqa: E402

TOKEN = "test-token-123"
PAGE = "page-op-1"
OTHER = "page-ro-2"


class Base(AsyncHTTPTestCase):
    def get_app(self):
        import tempfile
        import pathlib
        self._tmp = tempfile.TemporaryDirectory()
        self.env = Env(pathlib.Path(self._tmp.name))
        self.env.act("release_control")                                 # nobody is the operator until a page polls
        self.auth = Auth(self.get_http_port(), token=TOKEN)
        self.hub = CameraHub(self.env.bridge, DEFAULT_CAMERAS, clock=self.env.clock)
        self.ctx = Context(self.env.core, self.hub, self.auth)
        self.static = os.path.join(self._tmp.name, "static")
        os.makedirs(self.static)
        with open(os.path.join(self.static, "index.html"), "w") as f:
            f.write("<html>dual cut</html>")
        return make_app(self.ctx, self.static)

    def tearDown(self):
        self.env.close()
        self.ctx.shutdown()
        self._tmp.cleanup()
        super().tearDown()

    def hdr(self, page=PAGE, **extra):
        h = {"X-Token": TOKEN, "X-Page-Id": page}
        h.update(extra)
        return h

    def get(self, path, page=PAGE, **kw):
        return self.fetch(path, headers=self.hdr(page, **kw.pop("headers", {})), **kw)

    def act(self, action, args=None, page=PAGE, **hdr):
        body = {"action": action}
        if args is not None:
            body["args"] = args
        return self.fetch("/api/action", method="POST", body=json.dumps(body), headers=self.hdr(page, **hdr))

    @staticmethod
    def j(resp):
        return json.loads(resp.body)


class TestAuth(Base):
    def test_token_required_on_every_api_route(self):
        for path in ("/api/state", "/api/camera/scene.jpg", "/api/detection.jpg"):
            assert self.fetch(path).code == 401
            assert self.fetch(path, headers={"X-Token": "wrong"}).code == 401
        r = self.fetch("/api/action", method="POST", body='{"action":"software_stop"}')
        assert r.code == 401 and self.env.bridge.calls == []             # even the stop needs the token
        assert self.fetch("/api/action", method="POST", body='{"action":"software_stop"}', headers={"X-Token": "x"}).code == 401

    def test_index_is_public_and_static_served(self):
        r = self.fetch("/")
        assert r.code == 200 and b"dual cut" in r.body and r.headers["Cache-Control"] == "no-store"
        open(os.path.join(self.static, "app.js"), "w").write("//js")
        assert self.fetch("/static/app.js").code == 200
        r = self.fetch("/static/app.js")
        assert r.headers.get("Cache-Control") == "no-cache"                       # revalidated: no stale JS after a rebuild
        assert self.fetch("/static/app.js", headers={"If-None-Match": r.headers["Etag"]}).code == 304

    def test_index_versions_the_script_and_css(self):
        with open(os.path.join(self.static, "index.html"), "w") as f:
            f.write('<link href="/static/style.css"><script src="/static/app.js"></script>')
        open(os.path.join(self.static, "app.js"), "w").write("//js")
        body = self.fetch("/").body.decode()
        v = str(os.stat(os.path.join(self.static, "app.js")).st_mtime_ns)
        assert f'src="/static/app.js?v={v}"' in body
        assert 'href="/static/style.css"' in body                                  # missing file: left alone
        assert self.fetch(f"/static/app.js?v={v}").body == b"//js"
        assert self.fetch("/api/nothing", headers=self.hdr()).code == 404

    def test_origin_check_on_post(self):
        port = self.get_http_port()
        self.get("/api/state")                                           # becomes the operator
        assert self.act("rearm", {}, Origin="http://evil.example").code == 403
        assert self.act("software_stop", Origin="http://evil.example").code == 403    # a foreign page cannot even stop
        assert self.env.bridge.calls == []
        for origin in (f"http://127.0.0.1:{port}", f"http://localhost:{port}"):
            assert self.act("rearm", {}, Origin=origin).code == 200
        assert self.act("rearm", {}).code == 200
        assert self.act("rearm", {}, Origin="http://127.0.0.1:1").code == 403

    def test_foreign_host_rejected(self):
        assert self.get("/api/state", headers={"Host": "evil.example"}).code == 403
        assert self.get("/api/camera/scene.jpg", headers={"Host": "evil.example"}).code == 403

    def test_the_server_binds_loopback_only(self):
        from dual_cut_ui import server
        import inspect
        src = inspect.getsource(server.main)
        assert 'address="127.0.0.1"' in src and "0.0.0.0" not in src


class TestActions(Base):
    def test_unknown_actions_are_400_over_http(self):
        for name in ("call_service", "set_parameter", "publish_topic", "trigger", "Software_Stop"):
            r = self.act(name, {"service": "/x"})
            assert r.code == 400 and self.j(r)["ok"] is False
        assert self.env.bridge.calls == []
        assert self.fetch("/api/action", method="POST", body="[1]", headers=self.hdr()).code == 400
        assert self.fetch("/api/action", method="POST", body="{not json", headers=self.hdr()).code == 400
        assert self.fetch("/api/action", method="POST", body='{"action": 5}', headers=self.hdr()).code == 400

    def test_get_on_action_is_not_allowed(self):
        assert self.get("/api/action").code == 405

    def test_state_claims_operator_then_read_only_pages_get_403(self):
        st = self.j(self.get("/api/state"))
        assert st["operator"]["is_operator"] is True and "cameras" in st and set(st["cameras"]) == set(DEFAULT_CAMERAS)
        assert st["cameras_default_off"] == []                            # no hint configured in this test hub
        assert self.j(self.get("/api/state", page=OTHER))["operator"]["is_operator"] is False
        r = self.act("rearm", {}, page=OTHER)
        assert r.code == 403 and "read-only" in self.j(r)["message"]
        assert self.act("rearm", {}, page=PAGE).code == 200

    def test_software_stop_from_read_only_page_over_http(self):
        self.get("/api/state")
        self.get("/api/state", page=OTHER)
        r = self.act("software_stop", page=OTHER)
        assert r.code == 200 and self.j(r)["ok"] is True
        assert self.env.bridge.count("cancel") >= 2

    def test_software_stop_not_blocked_by_slow_actions(self):
        gate = threading.Event()
        self.env.bridge.block["rearm"] = gate
        self.get("/api/state")
        # saturate the action pool with blocked requests
        for _ in range(4):
            self.ctx.action_pool.submit(gate.wait, 5)
        t0 = time.time()
        r = self.act("software_stop")
        gate.set()
        assert r.code == 200 and time.time() - t0 < 3.0

    def test_execute_over_http(self):
        self.get("/api/state")
        r = self.act("execute", {"plan_seq": 3})
        assert r.code == 200 and self.j(r)["message"].startswith("dispatch accepted")
        r = self.act("execute", {"plan_seq": 9})
        assert r.code == 409

    def test_state_is_strict_json_with_nan(self):
        r = self.get("/api/state")
        assert b"NaN" not in r.body
        st = self.j(r)
        assert st["coordinator"]["min_cross_distance_m"] is None
        assert st["coordinator"]["state_name"] == "PLANNED" and st["cut"]["can_cut"] is True
        assert st["snapshot"]["available"] and st["snapshot"]["placement"] is not None and "top cut" not in st["svg"]   # carton only, no intervals before a plan report
        assert st["gates"]["all_enabled"] is True and st["health"]["all_active"] is True

    def test_state_offline_when_status_is_old(self):
        self.env.clock.advance(5)
        st = self.j(self.get("/api/state"))
        assert st["coordinator"]["online"] is False and st["coordinator"]["state_name"] == "COORDINATOR OFFLINE"
        assert st["cut"]["can_cut"] is False

    def test_plan_report_appears_with_svg(self):
        from helpers import read_data
        self.env.bridge._plan_report = (read_data("sequence_plan_report.json"), self.env.clock())
        st = self.j(self.get("/api/state"))
        assert st["plan"]["schedule_mode"] == "sequence" and st["plan"]["makespan_s"] > 70
        assert st["svg"].startswith("<svg") and "Left arm top cut" in st["svg"]
        self.env.bridge._exec_reports = [(open(os.path.join(os.path.dirname(__file__), "data", "sequence_exec_report.json")).read(), self.env.clock())]
        assert self.j(self.get("/api/state"))["exec_reports"][0]["completed"] is True


class TestCameras(Base):
    def test_unknown_camera_404_and_bad_size_400(self):
        assert self.get("/api/camera/nope.jpg").code == 404
        assert self.get("/api/camera/scene.jpg?size=huge").code == 400
        assert self.get("/api/camera/../state.jpg").code == 404
        assert self.env.bridge.count("subscribe") == 0

    def test_no_signal_then_jpeg(self):
        r = self.get("/api/camera/scene.jpg?size=small")
        assert r.code == 503 and self.j(r)["message"] == "no signal"
        self.env.bridge.subscriptions["scene"](synthetic_jpeg())
        r = self.get("/api/camera/scene.jpg?size=small")
        assert r.code == 200 and r.headers["Content-Type"] == "image/jpeg" and r.headers["Cache-Control"] == "no-store"
        assert r.body[:2] == b"\xff\xd8"
        r = self.get("/api/camera/scene.jpg?size=large")
        assert r.code == 200

    def test_detection_image(self):
        assert self.get("/api/detection.jpg").code == 404
        self.env.act("take_control")
        assert self.env.act("detect")[0] == 200
        self.env.wait_job()
        r = self.get("/api/detection.jpg?size=small")
        assert r.code == 200 and r.headers["Content-Type"] == "image/jpeg"
        assert self.get("/api/detection.jpg?size=weird").code == 400
        st = self.j(self.get("/api/state"))
        assert st["detection"]["available"] is True


def test_find_static_dir_finds_the_package_page():
    d = find_static_dir()
    assert os.path.isfile(os.path.join(d, "index.html")) and os.path.isfile(os.path.join(d, "app.js"))


class TestStopLatency(Base):
    def test_state_builds_run_on_their_own_pool(self):
        assert self.ctx.state_pool is not self.ctx.stop_pool and self.ctx.state_pool is not self.ctx.action_pool
        assert self.ctx.state_pool is not self.ctx.image_pool

    def test_a_slow_state_build_does_not_keep_the_ioloop_from_accepting_a_stop(self):
        gate = threading.Event()
        core, original = self.env.core, self.env.core.state

        def slow_state(*a, **k):
            gate.wait(10)
            return original(*a, **k)
        core.state = slow_state
        started = []
        try:
            fut = self.http_client.fetch(self.get_url("/api/state"), headers=self.hdr(), raise_error=False)
            for _ in range(3):                                           # several polls pile up, as a real page's would
                started.append(self.http_client.fetch(self.get_url("/api/state"), headers=self.hdr(), raise_error=False))
            t0 = time.time()
            r = self.act("software_stop")
            assert r.code == 200 and time.time() - t0 < 3.0
            assert not fut.done()                                        # the state build is still blocked
            assert self.env.bridge.count("cancel") >= 2
        finally:
            gate.set()
        resp = self.io_loop.run_sync(lambda: fut)
        assert resp.code == 200
        for f in started:
            self.io_loop.run_sync(lambda f=f: f)

    def test_detection_lookup_is_off_the_loop_too(self):
        import inspect
        from dual_cut_ui import server
        assert "image_pool, ctx.core.detection_info" in inspect.getsource(server.DetectionHandler.get)
