"""rgbd_recorder.py speed-ups (perf/speed W2) keep the captured data identical.

Covers box_seam_detection/live/recorder.py without ROS: rclpy,
message_filters, cv_bridge, sensor_msgs and sensor_msgs_py are replaced by
small stubs (read_points mirrors the Humble implementation: a structured
numpy view of msg.data). The tagged baseline recorder
(``git show live-v1-2026-09-28:scripts/rgbd_recorder.py``) is loaded with the
same stubs, and both write the same synthetic messages:

- PCD files are byte-identical (so every PCD reader parses identical XYZ;
  also checked with core/autolabel/pcd.py read_pcd_xyz), PNG files are
  byte-identical and decode to the input pixels, manifests agree.
- --once (max_frames == 1) commits the first synced pair without waiting for
  the save timer; bursts (max_frames > 1) still save on the timer only.
- --timeout-sec: exit code 3 + a stderr message naming the silent topics.
"""

import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

TESTS = Path(__file__).resolve().parent
PACKAGE_DIR = TESTS.parents[1] / "box_seam_detection"
RECORDER = PACKAGE_DIR / "live" / "recorder.py"
# The frozen external seam_handoff repo only provides the tagged BASELINE recorder for the identity check.
SEAM = Path("/home/farhan/Downloads/seam_handoff_20260908")
BASELINE_TAG = "live-v1-2026-09-28"
FRAME_ID = "camera_color_optical_frame"
N_REAL = 1703745  # point count of a real capture (frame_000001_20260928_055948_824096)


# ------------------------------------------------------------------ ROS stubs

class _Header:
    def __init__(self, frame_id=FRAME_ID, sec=0, nanosec=0):
        self.frame_id = frame_id
        self.stamp = types.SimpleNamespace(sec=sec, nanosec=nanosec)


class Image:
    def __init__(self):
        self.header = _Header()
        self.height = self.width = self.step = 0
        self.encoding = ""
        self.is_bigendian = 0
        self.data = b""


class PointField:
    INT8, UINT8, INT16, UINT16, INT32, UINT32, FLOAT32, FLOAT64 = range(1, 9)

    def __init__(self, name="", offset=0, datatype=7, count=1):
        self.name, self.offset, self.datatype, self.count = name, offset, datatype, count


class PointCloud2:
    def __init__(self):
        self.header = _Header()
        self.height = self.width = self.point_step = self.row_step = 0
        self.fields = []
        self.is_bigendian = False
        self.is_dense = True
        self.data = b""


class CameraInfo:
    def __init__(self):
        self.header = _Header()
        self.width, self.height = 1920, 1080
        self.distortion_model = "plumb_bob"
        self.d = [0.007, -0.049, 0.0003, 0.0001, 0.033, 0.0, 0.0, 0.0]
        self.k = [1037.8, 0.0, 977.0, 0.0, 1037.7, 549.0, 0.0, 0.0, 1.0]
        self.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        self.p = [1037.8, 0.0, 977.0, 0.0, 0.0, 1037.7, 549.0, 0.0, 0.0, 0.0, 1.0, 0.0]


_NP_OF_DATATYPE = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8"}


def _read_points(cloud, field_names=None, skip_nans=False, uvs=None, reshape_organized_cloud=False):
    """sensor_msgs_py.point_cloud2.read_points (Humble) for the recorder's call."""
    dtype = np.dtype({"names": [f.name for f in cloud.fields],
                      "formats": ["<" + _NP_OF_DATATYPE[f.datatype] for f in cloud.fields],
                      "offsets": [f.offset for f in cloud.fields],
                      "itemsize": cloud.point_step})
    points = np.ndarray(shape=(cloud.width * cloud.height,), dtype=dtype, buffer=cloud.data)
    if field_names is not None:
        points = points[list(field_names)]
    assert not skip_nans and uvs is None and not reshape_organized_cloud
    return points


class _CvBridge:
    def imgmsg_to_cv2(self, msg, desired_encoding="passthrough"):
        assert msg.encoding == "rgb8" and desired_encoding == "bgr8"
        rgb = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


class _Logger:
    def __init__(self, sink):
        self.sink = sink

    def info(self, message):
        self.sink.append(("info", message))

    def warn(self, message):
        self.sink.append(("warn", message))

    def error(self, message):
        self.sink.append(("error", message))


class _Param:
    def __init__(self, value):
        self.value = value

    def get_parameter_value(self):
        v = self.value
        return types.SimpleNamespace(string_value=v, double_value=v, bool_value=v, integer_value=v)


class _Env:
    """Shared state of the fake ROS runtime (parameter overrides, clock, nodes)."""

    def __init__(self):
        self.overrides = {}
        self.clock = 1000.0
        self.nodes = []
        self.spins = 0


class _Node:
    env = None

    def __init__(self, name):
        self.name = name
        self.logs = []
        self._params = {}
        self.subscriptions = {}
        self.timers = []
        _Node.env.nodes.append(self)

    def declare_parameter(self, name, default):
        value = _Node.env.overrides.get(name, default)
        if isinstance(default, float):
            value = float(value)
        elif isinstance(default, bool):
            value = value in (True, "true", "True")
        elif isinstance(default, int):
            value = int(value)
        self._params[name] = value

    def get_parameter(self, name):
        return _Param(self._params[name])

    def create_subscription(self, msg_type, topic, callback, qos):
        self.subscriptions[topic] = callback
        return object()

    def create_timer(self, period, callback):
        self.timers.append((period, callback))
        return object()

    def get_logger(self):
        return _Logger(self.logs)

    def destroy_node(self):
        pass


class _Subscriber:
    def __init__(self, node, msg_type, topic, qos_profile=None):
        self.topic = topic
        self.callbacks = []
        node.subscriptions[topic] = self

    def registerCallback(self, callback):
        self.callbacks.append(callback)

    def deliver(self, msg):
        for callback in self.callbacks:
            callback(msg)


class _Sync:
    def __init__(self, subscribers, queue_size, slop):
        self.subscribers, self.slop = subscribers, slop
        self.callbacks = []

    def registerCallback(self, callback):
        self.callbacks.append(callback)

    def emit(self, rgb, pc):  # message_filters' own matching is not under test
        for callback in self.callbacks:
            callback(rgb, pc)


def _remove_ros_args(args):
    args = list(args)
    return args[:args.index("--ros-args")] if "--ros-args" in args else args


def _stub_modules(env):
    _Node.env = env

    def init(args=None):
        args = list(args or [])
        if "--ros-args" in args:
            rest = args[args.index("--ros-args") + 1:]
            for flag, value in zip(rest, rest[1:]):
                if flag == "-p":
                    key, val = value.split(":=", 1)
                    env.overrides[key] = val

    def spin_once(node, timeout_sec=None):
        env.spins += 1
        env.clock += timeout_sec

    rclpy = types.ModuleType("rclpy")
    rclpy.init, rclpy.ok, rclpy.spin_once, rclpy.shutdown = init, (lambda: True), spin_once, (lambda: None)
    node_mod = types.ModuleType("rclpy.node")
    node_mod.Node = _Node
    utilities = types.ModuleType("rclpy.utilities")
    utilities.remove_ros_args = _remove_ros_args
    qos = types.ModuleType("rclpy.qos")
    qos.qos_profile_sensor_data = object()
    mf = types.ModuleType("message_filters")
    mf.Subscriber, mf.ApproximateTimeSynchronizer = _Subscriber, _Sync
    sensor_msgs = types.ModuleType("sensor_msgs")
    msg = types.ModuleType("sensor_msgs.msg")
    msg.Image, msg.PointCloud2, msg.PointField, msg.CameraInfo = Image, PointCloud2, PointField, CameraInfo
    smp = types.ModuleType("sensor_msgs_py")
    pc2 = types.ModuleType("sensor_msgs_py.point_cloud2")
    pc2.read_points = _read_points
    smp.point_cloud2 = pc2
    bridge = types.ModuleType("cv_bridge")
    bridge.CvBridge = _CvBridge
    return {"rclpy": rclpy, "rclpy.node": node_mod, "rclpy.utilities": utilities, "rclpy.qos": qos,
            "message_filters": mf, "sensor_msgs": sensor_msgs, "sensor_msgs.msg": msg,
            "sensor_msgs_py": smp, "sensor_msgs_py.point_cloud2": pc2, "cv_bridge": bridge}


def _load(name, path, monkeypatch, env):
    for mod_name, module in _stub_modules(env).items():
        monkeypatch.setitem(sys.modules, mod_name, module)
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeTime:
    """Replaces the recorder module's ``time``: monotonic follows env.clock."""

    def __init__(self, env):
        import time as real_time
        self._env, self.time_ns = env, real_time.time_ns

    def monotonic(self):
        return self._env.clock


@pytest.fixture
def env():
    return _Env()


@pytest.fixture
def rec(monkeypatch, env):
    if not RECORDER.is_file():
        pytest.skip(f"missing {RECORDER}")
    module = _load("rgbd_recorder_new", RECORDER, monkeypatch, env)
    monkeypatch.setattr(module, "time", _FakeTime(env))
    return module


@pytest.fixture
def old_rec(monkeypatch, env, tmp_path):
    try:
        source = subprocess.run(["git", "-C", str(SEAM), "show", f"{BASELINE_TAG}:scripts/rgbd_recorder.py"],
                                check=True, capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        pytest.skip(f"baseline recorder unavailable: {error}")
    path = tmp_path / "rgbd_recorder_baseline.py"
    path.write_text(source, encoding="utf-8")
    return _load("rgbd_recorder_baseline", path, monkeypatch, env)


@pytest.fixture
def pcd_reader():
    path = PACKAGE_DIR / "core" / "autolabel" / "pcd.py"
    spec = importlib.util.spec_from_file_location("autolabel_pcd_for_recorder_test", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------ synthetic data

def make_image(seed=0, width=1920, height=1080, stamp=(100, 0)):
    rng = np.random.default_rng(seed)
    # Smooth gradient + noise: compressible like a camera frame, not trivially.
    yy, xx = np.mgrid[0:height, 0:width]
    base = np.stack([(xx * 255 // width), (yy * 255 // height), ((xx + yy) % 256)], axis=-1)
    rgb = np.clip(base + rng.integers(-12, 13, size=base.shape), 0, 255).astype(np.uint8)
    msg = Image()
    msg.height, msg.width, msg.encoding, msg.step = height, width, "rgb8", width * 3
    msg.data = rgb.tobytes()
    msg.header = _Header(sec=stamp[0], nanosec=stamp[1])
    return msg, rgb


def make_cloud(n=N_REAL, seed=1, width=None, height=1, color="rgb", color_datatype=7, stamp=(100, 800000)):
    """PointCloud2 laid out like the Orbbec depth_registered cloud: x,y,z @0/4/8, rgb @16, step 20."""
    rng = np.random.default_rng(seed)
    step = 20 if color else 16
    buf = np.zeros(n * step, np.uint8)
    rec_view = buf.view(np.float32).reshape(n, step // 4)
    xyz = rng.normal([0.0, 0.0, 0.9], [0.4, 0.3, 0.2], size=(n, 3)).astype(np.float32)
    xyz[rng.random(n) < 0.05] = np.nan  # invalid points keep their NaNs
    rec_view[:, 0:3] = xyz
    rec_view[:, 3] = np.float32(rng.normal(size=n))  # padding garbage, never written
    fields = [PointField("x", 0, 7, 1), PointField("y", 4, 7, 1), PointField("z", 8, 7, 1)]
    if color:
        packed = rng.integers(0, 2 ** 24, size=n, dtype=np.uint32)
        packed[: n // 3] |= np.uint32(0xFF000000)  # alpha 255 -> NaN bit patterns as float32
        buf.view(np.uint32).reshape(n, 5)[:, 4] = packed
        fields.append(PointField(color, 16, color_datatype, 1))
    msg = PointCloud2()
    msg.width = n if width is None else width
    msg.height = height
    msg.point_step, msg.row_step = step, step * msg.width
    msg.fields, msg.data = fields, buf.tobytes()
    msg.header = _Header(sec=stamp[0], nanosec=stamp[1])
    return msg


def make_node(module, env, out_dir, **overrides):
    env.overrides.update({k: v for k, v in overrides.items() if k not in ("once", "timeout")})
    return module.RgbdRecorder(cli_output_dir=str(out_dir),
                               cli_max_frames=1 if overrides.get("once") else None,
                               **({"cli_timeout_sec": overrides["timeout"]} if "timeout" in overrides else {}))


def saved_files(out_dir):
    return {sub: sorted((Path(out_dir) / sub).glob("*")) for sub in ("rgb", "pointcloud", "captures")}


# ------------------------------------------------------------- data identity

@pytest.mark.parametrize("case", [
    dict(),                                       # real layout, 1.7 M points, float32 rgb
    dict(n=240 * 320, width=320, height=240),     # organized cloud
    dict(n=50_000, color=None),                   # xyz only
    dict(n=50_000, color="rgba", color_datatype=6),  # uint32 packed color
    dict(n=0),                                    # empty cloud
])
def test_pcd_bytes_identical_to_baseline(rec, old_rec, env, tmp_path, pcd_reader, case):
    cloud = make_cloud(**case)
    old_node = old_rec.RgbdRecorder(cli_output_dir=str(tmp_path / "old"))
    new_node = rec.RgbdRecorder(cli_output_dir=str(tmp_path / "new"))
    old_path, old_n, old_h = old_node._save_pointcloud(cloud, "f")
    new_path, new_n, new_h = new_node._save_pointcloud(cloud, "f")
    assert (new_n, new_h) == (old_n, old_h)
    assert Path(new_path).read_bytes() == Path(old_path).read_bytes()
    if case.get("n", N_REAL):
        old_xyz, _ = pcd_reader.read_pcd_xyz(old_path)
        new_xyz, _ = pcd_reader.read_pcd_xyz(new_path)
        assert old_xyz.dtype == new_xyz.dtype and old_xyz.shape == new_xyz.shape
        assert np.array_equal(old_xyz, new_xyz, equal_nan=True)
        src = _read_points(cloud, ["x", "y", "z"])
        expected = np.column_stack([src["x"], src["y"], src["z"]]).astype(new_xyz.dtype)
        expected = expected[np.isfinite(expected).all(axis=1)]  # read_pcd_xyz drops NaN rows
        assert np.array_equal(new_xyz.reshape(-1, 3), expected)


def test_pcd_ascii_identical_to_baseline(rec, old_rec, env, tmp_path):
    env.overrides["pcd_binary"] = False
    cloud = make_cloud(n=2000)
    old_path, *_ = old_rec.RgbdRecorder(cli_output_dir=str(tmp_path / "old"))._save_pointcloud(cloud, "f")
    new_path, *_ = rec.RgbdRecorder(cli_output_dir=str(tmp_path / "new"))._save_pointcloud(cloud, "f")
    assert Path(new_path).read_bytes() == Path(old_path).read_bytes()


def test_png_identical_to_baseline(rec, old_rec, env, tmp_path):
    image, rgb = make_image()
    old_path = old_rec.RgbdRecorder(cli_output_dir=str(tmp_path / "old"))._save_rgb(image, "f")
    new_path = rec.RgbdRecorder(cli_output_dir=str(tmp_path / "new"))._save_rgb(image, "f")
    assert Path(new_path).read_bytes() == Path(old_path).read_bytes()
    decoded = cv2.cvtColor(cv2.imread(new_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
    assert np.array_equal(decoded, rgb)


def _strip_volatile(manifest):
    manifest = json.loads(json.dumps(manifest))
    manifest.pop("created_at_ns")
    manifest.pop("capture_id")
    manifest["image"].pop("path")
    manifest["pointcloud"].pop("path")
    return manifest


def test_full_capture_matches_baseline(rec, old_rec, env, tmp_path):
    """Timer-driven save of the same pair: identical PNG/PCD bytes and manifest content."""
    image, _ = make_image(seed=3)
    cloud = make_cloud(n=100_000, seed=4)
    info = CameraInfo()
    results = {}
    for tag, module in (("old", old_rec), ("new", rec)):
        node = module.RgbdRecorder(cli_output_dir=str(tmp_path / tag))
        node._camera_info_callback(info)
        node._sync_callback(image, cloud)
        node._save_timer_callback()
        files = saved_files(tmp_path / tag)
        assert all(len(v) == 1 for v in files.values()), files
        results[tag] = files
    for sub in ("rgb", "pointcloud"):
        assert results["new"][sub][0].read_bytes() == results["old"][sub][0].read_bytes()
    old_manifest = json.loads(results["old"]["captures"][0].read_text())
    new_manifest = json.loads(results["new"]["captures"][0].read_text())
    assert _strip_volatile(new_manifest) == _strip_volatile(old_manifest)
    # manifest point-cloud metadata is consistent with the file written
    pcd = results["new"]["pointcloud"][0].read_bytes()
    header = dict(line.split(" ", 1) for line in pcd[:400].decode("ascii", "replace").splitlines()[1:11])
    pc = new_manifest["pointcloud"]
    assert int(header["POINTS"]) == pc["point_count"] == cloud.width * cloud.height
    assert int(header["HEIGHT"]) == pc["stored_height"]
    assert pc["point_step"] == cloud.point_step and pc["row_step"] == cloud.row_step
    assert [f["name"] for f in pc["fields"]] == ["x", "y", "z", "rgb"]


# ------------------------------------------------------ --once immediate save

def test_once_saves_first_synced_pair_without_timer(rec, env, tmp_path):
    node = make_node(rec, env, tmp_path, once=True)
    assert node.save_immediately and node.max_frames == 1 and node.timeout_sec == 20.0
    image, _ = make_image(seed=5)
    cloud = make_cloud(n=10_000)
    node._camera_info_callback(CameraInfo())
    node._sync_callback(image, cloud)  # no timer tick
    files = saved_files(tmp_path)
    assert all(len(v) == 1 for v in files.values())
    assert node.done is True
    manifest = json.loads(files["captures"][0].read_text())
    assert manifest["image"]["header"]["stamp_ns"] == 100 * 10 ** 9
    assert manifest["pointcloud"]["header"]["stamp_ns"] == 100 * 10 ** 9 + 800000
    assert manifest["sync_delta_ns"] == 800000
    # later pairs / timer ticks never add a second capture
    node._sync_callback(*make_image(seed=6)[:1], make_cloud(n=10_000, seed=7))
    for _, callback in node.timers:
        callback()
    assert all(len(v) == 1 for v in saved_files(tmp_path).values())
    logged = "\n".join(m for _, m in node.logs)
    assert "已儲存影像:" in logged and "已儲存點雲:" in logged and "已儲存 capture manifest:" in logged


def test_once_pair_before_camera_info_waits_then_saves_that_pair(rec, env, tmp_path):
    node = make_node(rec, env, tmp_path, once=True)
    image, rgb = make_image(seed=8)
    node._sync_callback(image, make_cloud(n=5000))
    assert all(len(v) == 0 for v in saved_files(tmp_path).values())
    node._camera_info_callback(CameraInfo())
    files = saved_files(tmp_path)
    assert all(len(v) == 1 for v in files.values()) and node.done
    decoded = cv2.cvtColor(cv2.imread(str(files["rgb"][0])), cv2.COLOR_BGR2RGB)
    assert np.array_equal(decoded, rgb)


def test_once_failed_commit_falls_back_to_timer(rec, env, tmp_path):
    node = make_node(rec, env, tmp_path, once=True)
    info = CameraInfo()
    info.header.frame_id = "other_frame"  # manifest check rejects the capture
    node._camera_info_callback(info)
    node._sync_callback(make_image(seed=9)[0], make_cloud(n=1000))
    assert node.save_immediately is False and not node.done
    assert saved_files(tmp_path)["captures"] == []
    assert any(level == "error" for level, _ in node.logs)


# -------------------------------------------------------- burst unchanged

def test_burst_saves_on_timer_only(rec, env, tmp_path):
    node = make_node(rec, env, tmp_path, max_frames=3, save_rate_hz=5.0)
    assert not node.save_immediately and node.timeout_sec == 20.0
    assert node.timers[0][0] == pytest.approx(0.2)
    node._camera_info_callback(CameraInfo())
    timer = node.timers[0][1]
    for i in range(3):
        node._sync_callback(make_image(seed=10 + i, width=64, height=48)[0], make_cloud(n=500, seed=i))
        assert len(saved_files(tmp_path)["captures"]) == i  # no save before the tick
        timer()
        assert len(saved_files(tmp_path)["captures"]) == i + 1
    assert node.done
    timer()  # empty tick after done: nothing new
    assert len(saved_files(tmp_path)["captures"]) == 3


def test_continuous_mode_has_no_default_timeout(rec, env, tmp_path):
    node = make_node(rec, env, tmp_path)
    assert node.max_frames == 0 and node.timeout_sec == 0 and not node.save_immediately
    env.clock += 10 ** 6
    assert node.timed_out() is False


@pytest.mark.parametrize("max_frames,rate,expected", [
    (1, 0.05, 20.0),            # --once keeps the fixed 20 s, whatever the rate
    (3, 5.0, 20.0),             # fast burst: floor of 20 s
    (3, 0.2, 20.0),             # 2/0.2 + 5 = 15 -> still 20
    (3, 0.1, 25.0),             # 2/0.1 + 5 = 25
    (5, 0.02, 105.0),           # 2/0.02 + 5
    (0, 0.1, 0.0),              # continuous: no default timeout
])
def test_default_timeout_scales_with_burst_save_rate(rec, env, tmp_path, max_frames, rate,
                                                     expected):
    assert rec.default_timeout_sec(max_frames, rate) == pytest.approx(expected)
    node = make_node(rec, env, tmp_path, max_frames=max_frames, save_rate_hz=rate)
    assert node.timeout_sec == pytest.approx(expected)


def test_explicit_timeout_wins_over_burst_default(rec, env, tmp_path):
    node = make_node(rec, env, tmp_path, max_frames=3, save_rate_hz=0.05, timeout=4.0)
    assert node.timeout_sec == 4.0


def test_slow_burst_does_not_time_out_between_timer_saves(rec, env, tmp_path):
    # 0.1 Hz burst: the scaled default is 2/0.1 + 5 = 25 s; at 24 s the old
    # fixed 20 s budget would already have tripped.
    node = make_node(rec, env, tmp_path, max_frames=3, save_rate_hz=0.1)
    env.clock += 24.0
    assert not node.timed_out()
    env.clock += 1.5
    assert node.timed_out()


# ----------------------------------------------------------------- timeout

def _run_main(rec, tmp_path, *extra):
    return rec.main(["rgbd_recorder.py", str(tmp_path), *extra])


def test_timeout_exit_code_and_message(rec, env, tmp_path, capsys):
    code = _run_main(rec, tmp_path, "--once", "--timeout-sec", "2",
                     "--ros-args", "-p", "pointcloud_topic:=/camera/depth_registered/points")
    assert code == rec.EXIT_TIMEOUT == 3
    err = capsys.readouterr().err
    assert "timeout" in err and "2 s" in err
    assert "no messages on: /camera/color/image_raw (rgb), /camera/depth_registered/points (pointcloud), " \
           "/camera/color/camera_info (camera_info)" in err
    assert 4 <= env.spins <= 5  # 0.5 s spins until the 2 s deadline


def test_timeout_names_only_silent_topics(rec, env, tmp_path, capsys):
    node = make_node(rec, env, tmp_path, once=True, timeout=5.0)
    node.subscriptions["/camera/color/image_raw"].deliver(make_image(width=8, height=4)[0])
    node._camera_info_callback(CameraInfo())
    env.clock += 4.9
    assert not node.timed_out()
    env.clock += 0.2
    assert node.timed_out()
    message = node.timeout_message()
    assert "no messages on: /camera/depth/points (pointcloud)" in message
    assert "/camera/color/image_raw (rgb)" not in message.split("no messages on:")[1].splitlines()[0]
    assert "rgb=1, pointcloud=0, camera_info=1, synced_pairs=0" in message


def test_timeout_all_topics_but_no_sync(rec, env, tmp_path):
    node = make_node(rec, env, tmp_path, once=True)
    node.subscriptions["/camera/color/image_raw"].deliver(object())
    node.subscriptions["/camera/depth/points"].deliver(object())
    node._camera_info_callback(CameraInfo())
    assert "no RGB/point-cloud pair synced" in node.timeout_message()


def test_timeout_resets_after_each_saved_burst_frame(rec, env, tmp_path):
    node = make_node(rec, env, tmp_path, max_frames=2, timeout=3.0)
    node._camera_info_callback(CameraInfo())
    env.clock += 2.5
    node._sync_callback(make_image(width=8, height=4)[0], make_cloud(n=10))
    node.timers[0][1]()
    env.clock += 2.5
    assert not node.timed_out()  # 2.5 s since the last save
    env.clock += 1.0
    assert node.timed_out()
    assert "since the last saved capture (saved 1/2)" in node.timeout_message()


def test_timeout_zero_waits_forever_and_success_exits_zero(rec, env, tmp_path, monkeypatch):
    spins = {"n": 0}

    def spin_once(node, timeout_sec=None):
        spins["n"] += 1
        env.clock += 100.0  # far beyond any default timeout
        if spins["n"] == 50:
            node._camera_info_callback(CameraInfo())
            node._sync_callback(make_image(width=8, height=4)[0], make_cloud(n=10))

    monkeypatch.setattr(sys.modules["rclpy"], "spin_once", spin_once)
    assert _run_main(rec, tmp_path, "--once", "--timeout-sec", "0") == 0
    assert spins["n"] == 50 and len(saved_files(tmp_path)["captures"]) == 1


def test_negative_timeout_rejected(rec, tmp_path):
    with pytest.raises(SystemExit) as info:
        _run_main(rec, tmp_path, "--once", "--timeout-sec", "-1")
    assert info.value.code == 2
