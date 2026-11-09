"""Camera pipeline: lazy subscription, reduced decode + resize, per-seq cache, rate limit, no-signal."""
import cv2
import numpy as np
import pytest

from dual_cut_ui import camera as C
from dual_cut_ui.camera import CameraHub, DEFAULT_CAMERAS, jpeg_width, parse_camera_spec, reencode
from helpers import FakeBridge, FakeClock


def synthetic_jpeg(w=1280, h=720, seed=0, quality=90):
    rng = np.random.RandomState(seed)
    img = (rng.rand(h // 8, w // 8, 3) * 255).astype(np.uint8)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
    cv2.putText(img, str(seed), (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
    ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return enc.tobytes()


def decoded_width(jpeg):
    return cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR).shape[1]


@pytest.fixture
def hub():
    clock = FakeClock(100.0)
    bridge = FakeBridge(clock)
    return CameraHub(bridge, DEFAULT_CAMERAS, clock=clock), bridge, clock


def test_default_camera_list_is_the_approved_one():
    assert dict(DEFAULT_CAMERAS) == {
        "scene": "/camera/color/image_raw/compressed", "knife": "/knife_camera/image_raw/compressed",
        "left_wrist": "/arm_left/wrist_camera/color/image_raw/compressed",
        "right_wrist": "/arm_right/wrist_camera/color/image_raw/compressed",
        "left_gripper": "/arm_left/gripper_camera/image_raw/compressed",
        "right_gripper": "/arm_right/gripper_camera/image_raw/compressed"}


def test_parse_camera_spec():
    assert list(parse_camera_spec("")) == list(DEFAULT_CAMERAS)
    assert dict(parse_camera_spec("a=/x/y, b_2=/z")) == {"a": "/x/y", "b_2": "/z"}
    for bad in ("a", "a=", "=/x", "a=x", "a b=/x", ","):
        with pytest.raises(ValueError):
            parse_camera_spec(bad)


def test_jpeg_width_without_decoding():
    assert jpeg_width(synthetic_jpeg(1280, 720)) == 1280
    assert jpeg_width(synthetic_jpeg(333, 200)) == 333
    assert jpeg_width(b"") is None and jpeg_width(b"not a jpeg at all") is None
    png = cv2.imencode(".png", np.zeros((10, 10, 3), np.uint8))[1].tobytes()
    assert jpeg_width(png) is None


@pytest.mark.parametrize("src_w,target,factor", [(1920, 480, 4), (1920, 960, 2), (1280, 480, 2), (1280, 960, 1), (640, 480, 1), (3840, 480, 8)])
def test_reduction_factor_keeps_the_result_at_least_target_wide(monkeypatch, src_w, target, factor):
    flags = []
    real = cv2.imdecode
    monkeypatch.setattr(C.cv2, "imdecode", lambda buf, flag: flags.append(flag) or real(buf, flag))
    out = reencode(synthetic_jpeg(src_w, src_w * 9 // 16), target, passthrough=0)       # passthrough has its own test
    expected = {1: cv2.IMREAD_COLOR, 2: cv2.IMREAD_REDUCED_COLOR_2, 4: cv2.IMREAD_REDUCED_COLOR_4, 8: cv2.IMREAD_REDUCED_COLOR_8}[factor]
    assert flags == [expected]
    assert decoded_width(out) == min(target, src_w)


def test_never_enlarges_and_keeps_aspect():
    out = reencode(synthetic_jpeg(320, 240), 480, passthrough=0)
    img = cv2.imdecode(np.frombuffer(out, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[:2] == (240, 320)
    out = reencode(synthetic_jpeg(1280, 720), 480)
    assert cv2.imdecode(np.frombuffer(out, np.uint8), cv2.IMREAD_COLOR).shape[:2] == (270, 480)


def test_passthrough_returns_the_camera_jpeg_unchanged(monkeypatch):
    """A source at most 1.35 x the target wide is forwarded as is: no decode, no encode (this is what keeps the large tile cheap)."""
    src = synthetic_jpeg(1280, 720)
    monkeypatch.setattr(C.cv2, "imdecode", lambda *a: pytest.fail("decoded although a pass-through was possible"))
    monkeypatch.setattr(C.cv2, "imencode", lambda *a: pytest.fail("encoded although a pass-through was possible"))
    assert reencode(src, 960) == src                     # 1280 <= 960 * 1.35
    assert reencode(src, 1280) == src


def test_no_passthrough_when_the_source_is_much_wider():
    src = synthetic_jpeg(1920, 1080)
    out = reencode(src, 960)
    assert out != src and decoded_width(out) == 960


def test_undecodable_data_gives_none():
    assert reencode(b"garbage", 480) is None


def test_lazy_subscription_and_sizes(hub):
    h, bridge, clock = hub
    assert bridge.count("subscribe") == 0                              # nothing until somebody asks
    data, reason = h.request("scene", "small")
    assert data is None and reason == "no signal"
    assert [c[1:3] for c in bridge.names("subscribe")] == [("scene", "/camera/color/image_raw/compressed")]
    h.request("scene", "large")
    assert bridge.count("subscribe") == 1                              # one subscription per camera
    bridge.subscriptions["scene"](synthetic_jpeg(1920, 1080))
    small, r1 = h.request("scene", "small")
    large, r2 = h.request("scene", "large")
    assert r1 == r2 == "ok"
    assert decoded_width(small) == 320 and decoded_width(large) == 960
    assert h.status()["scene"] == {"topic": "/camera/color/image_raw/compressed", "subscribed": True, "signal": True}
    assert h.status()["left_gripper"]["subscribed"] is False


def test_unknown_camera_and_bad_size_do_not_subscribe(hub):
    h, bridge, _ = hub
    assert h.request("nope", "small") == (None, "unknown camera")
    assert h.request("scene", "huge") == (None, "bad size")
    assert bridge.count("subscribe") == 0


def test_cache_per_frame_seq_and_rate_cap(hub):
    h, bridge, clock = hub
    h.request("left_gripper", "small")
    cb = bridge.subscriptions["left_gripper"]
    cb(synthetic_jpeg(1280, 720, seed=1))
    first, _ = h.request("left_gripper", "small")
    again, _ = h.request("left_gripper", "small")
    assert again is first                                              # same frame: cached bytes, no re-encode
    cb(synthetic_jpeg(1280, 720, seed=2))
    clock.advance(0.3)                                                 # new frame but < 0.4 s since the last small encode
    capped, _ = h.request("left_gripper", "small")
    assert capped is first
    clock.advance(0.2)
    fresh, _ = h.request("left_gripper", "small")
    assert fresh is not first and decoded_width(fresh) == 320
    big, _ = h.request("left_gripper", "large")                               # separate cache per size (1280 px source: pass-through)
    assert decoded_width(big) == 1280


def test_large_tile_may_refresh_faster_than_small_tiles(hub):
    h, bridge, clock = hub
    h.request("left_gripper", "large")
    h.request("left_gripper", "small")
    cb = bridge.subscriptions["left_gripper"]
    cb(synthetic_jpeg(1920, 1080, seed=1))
    s1, _ = h.request("left_gripper", "small")
    l1, _ = h.request("left_gripper", "large")
    cb(synthetic_jpeg(1920, 1080, seed=2))
    clock.advance(0.15)
    assert h.request("left_gripper", "small")[0] is s1                         # small: 0.4 s cap
    assert h.request("left_gripper", "large")[0] is not l1                     # large: 0.125 s cap


def test_no_signal_after_3_seconds_without_frames(hub):
    h, bridge, clock = hub
    h.request("scene", "small")
    bridge.subscriptions["scene"](synthetic_jpeg())
    assert h.request("scene", "small")[1] == "ok"
    clock.advance(3.5)
    assert h.request("scene", "small") == (None, "no signal")
    assert h.status()["scene"]["signal"] is False
    bridge.subscriptions["scene"](synthetic_jpeg())
    assert h.request("scene", "small")[1] == "ok"


def test_idle_subscriptions_are_destroyed_after_10_seconds(hub):
    h, bridge, clock = hub
    for n in ("scene", "left_gripper"):
        h.request(n, "small")
    clock.advance(6)
    h.request("scene", "small")                                        # scene stays wanted, left_gripper does not
    clock.advance(5)
    h.sweep()
    assert [c[1] for c in bridge.names("unsubscribe")] == ["left_gripper"] and "scene" in bridge.subscriptions
    clock.advance(11)
    h.sweep()
    assert sorted(c[1] for c in bridge.names("unsubscribe")) == ["left_gripper", "scene"] and not bridge.subscriptions
    h.sweep()
    assert bridge.count("unsubscribe") == 2                            # not twice


def test_resubscribes_on_the_next_request_and_drops_old_frames(hub):
    h, bridge, clock = hub
    h.request("scene", "small")
    bridge.subscriptions["scene"](synthetic_jpeg())
    clock.advance(11)
    h.sweep()
    assert bridge.count("unsubscribe") == 1
    data, reason = h.request("scene", "small")
    assert data is None and reason == "no signal"                      # the old frame is gone, not served as live
    assert bridge.count("subscribe") == 2


def test_frames_arriving_after_unsubscribe_are_ignored(hub):
    h, bridge, clock = hub
    h.request("scene", "small")
    cb = bridge.subscriptions["scene"]
    clock.advance(11)
    h.sweep()
    cb(synthetic_jpeg())
    assert h._frame == {}


def test_callback_stores_bytes_without_decoding(hub, monkeypatch):
    h, bridge, clock = hub
    h.request("scene", "small")
    monkeypatch.setattr(C.cv2, "imdecode", lambda *a: pytest.fail("decoded in the ROS callback"))
    bridge.subscriptions["scene"](synthetic_jpeg())


def test_shutdown_unsubscribes_everything(hub):
    h, bridge, clock = hub
    for n in ("scene", "left_gripper"):
        h.request(n, "small")
    h.shutdown()
    assert not bridge.subscriptions


def test_overlay_jpeg_downscales(tmp_path):
    p = tmp_path / "o.png"
    cv2.imwrite(str(p), np.zeros((1080, 1920, 3), np.uint8))
    assert decoded_width(C.overlay_jpeg(str(p), 960)) == 960
    assert C.overlay_jpeg(str(tmp_path / "missing.png"), 960) is None


def test_parse_camera_off_keeps_known_names_in_camera_order():
    from dual_cut_ui.camera import DEFAULT_CAMERAS_OFF, parse_camera_off
    assert parse_camera_off(DEFAULT_CAMERAS_OFF, DEFAULT_CAMERAS) == ["knife"]
    assert parse_camera_off(" right_gripper, scene ,nope", DEFAULT_CAMERAS) == ["scene", "right_gripper"]
    assert parse_camera_off("none", DEFAULT_CAMERAS) == [] and parse_camera_off("", DEFAULT_CAMERAS) == []
    assert parse_camera_off("knife", {"scene": "/a"}) == []                    # custom list without a knife camera


def test_hub_default_off_is_only_a_hint():
    clock = FakeClock(100.0)
    h = CameraHub(FakeBridge(clock), DEFAULT_CAMERAS, clock=clock, default_off=["knife", "ghost"])
    assert h.default_off == ["knife"] and "knife" in h.status()                # still served when the page asks for it
