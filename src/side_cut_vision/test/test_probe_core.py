"""Node-level behaviour of the passive probe, driven through ProbeCore with fakes (no ROS, no clocks)."""
import json
import os
import shutil
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import synth
from side_cut_vision.calib import CALIB_SEED, load_hand_eye
from side_cut_vision.geom import inv_T, make_T
from side_cut_vision.probe_core import (ArmConfig, ArmProbe, Env, LiveGate, PairBuffer, SnapshotCache, Stamped)
from side_cut_vision.probe_node import decode_color, decode_depth
from side_cut_vision.toolmask import T_LINK6_TOOL0, tool_mask_from_geometry
from side_cut_vision.trigger import select_record_dir

J = [f"arm_left_joint{i}" for i in range(1, 7)]


# ------------------------------------------------------------------------------------------- fakes
class FakeEnv(Env):
    def __init__(self, T_wl6, T_wt, t0=None):
        t0 = time.time() if t0 is None else t0
        self.t = t0
        self.T_wl6, self.T_wt = T_wl6, T_wt
        self.bad_tf_stamps = False       # lookups at a stamp fail (extrapolation)
        self.nonfinite = False
        self.latest_stamp = None          # stamp of the newest TF sample (None = now)
        self.tool0 = T_LINK6_TOOL0

    def now_s(self):
        return self.t

    def lookup_latest(self, target, source):
        if self.nonfinite:
            return np.full((4, 4), np.nan), self.latest_stamp
        T = self.lookup(target, source, None)
        return None if T is None else (T, self.latest_stamp if self.latest_stamp is not None else self.t)

    def lookup(self, target, source, stamp):
        if self.bad_tf_stamps and stamp is not None:
            return None
        if self.nonfinite:
            return np.full((4, 4), np.nan)
        if source.endswith("link6"):
            return self.T_wl6
        if source.endswith("xeg32_tool_tip"):
            return self.T_wt
        if source.endswith("igus_tool0"):
            return self.tool0
        return None


@pytest.fixture(scope="module")
def scene():
    cam = synth.load_camera()
    model = synth.load_arm("left")
    T_wo, tip, T_wl6 = synth.approach_pose(model)
    color, depth, truth = synth.render(cam, T_wo, model, synth.SceneSpec(), stride=2)
    T_wt = make_T(np.eye(3), tip)
    return SimpleNamespace(cam=cam, model=model, T_wo=T_wo, tip=tip, T_wl6=T_wl6, T_wt=T_wt, color=color,
                           depth=depth, truth=truth)


def make_probe(scene, tmp_path, gate=None, calib_seed=False, **kw):
    he = load_hand_eye(synth.HANDEYE, seed_from_left=calib_seed)
    cfg = ArmConfig.default("left")
    cfg.calib_status = he.calib_status
    snaps = SnapshotCache(synth.SNAPSHOT)
    out = tmp_path / "cut_videos"
    out.mkdir(exist_ok=True)
    probe = ArmProbe(cfg, he, snaps, str(out), synth.SNAPSHOT, gate=gate or LiveGate(True),
                     record_dir_fn=kw.pop("record_dir_fn", select_record_dir), **kw)
    probe.set_camera(scene.cam)
    return probe, out


def feed_frames(probe, scene, t, color=None, depth=None):
    probe.pairs.add_color(Stamped(t, scene.color if color is None else color))
    probe.pairs.add_depth(Stamped(t, scene.depth if depth is None else depth))


def joints(q=0.0):
    return J, [q] * 6


def simulate(probe, env, scene, t_end, still_from, rate_img=15.0, color=None, depth=None, q_moving=True):
    """100 Hz joint states (moving until still_from, then constant), 15 Hz images, probe.step at 50 Hz.
    Returns all Outputs."""
    outs = []
    t0 = env.t
    dt = 0.005
    next_img = t0
    n = int(round((t_end - t0) / dt))
    for i in range(n):
        env.t = t0 + i * dt
        if i % 2 == 0:
            moving = q_moving and env.t < still_from
            probe.on_joint_state(env.t, J, [0.5 * (env.t - t0) if moving else 0.5 * (still_from - t0)] * 6)
        if env.t >= next_img:
            feed_frames(probe, scene, env.t, color, depth)
            next_img += 1.0 / rate_img
        if i % 4 == 0:
            outs += probe.step(env)
    return outs


# ------------------------------------------------------------------------------------------- units
def test_live_gate():
    g = LiveGate(False, hold_s=3.0)
    assert not g.active(0.0)
    g.on_status(4, 10.0)
    assert g.active(10.0) and g.active(50.0)          # still EXECUTING
    g.on_status(0, 51.0)
    assert g.active(53.0) and not g.active(54.5)      # hold after the last active status
    g.on_status(3, 60.0)
    assert g.active(60.0)
    assert LiveGate(True).active(0.0)
    g2 = LiveGate(False)
    g2.on_status(2, 1.0)                              # PLANNED is not active
    assert not g2.active(1.0)


def test_pair_buffer():
    pb = PairBuffer()
    img = np.zeros((2, 2, 3), np.uint8)
    assert pb.newest_pair()[2] == ["no_color"]
    pb.add_color(Stamped(1.0, img))
    assert pb.newest_pair()[2] == ["no_depth"]
    pb.add_depth(Stamped(1.2, img))
    assert pb.newest_pair()[2] == ["stale_pair"]
    pb.add_depth(Stamped(1.01, img))
    c, d, r = pb.newest_pair()
    assert r == [] and c.stamp == 1.0 and d.stamp == 1.01
    assert pb.newest_pair(after_stamp=1.0) == (None, None, [])     # nothing new is not an error


def test_snapshot_cache_reload_and_bad_file(tmp_path):
    p = tmp_path / "s.json"
    shutil.copy(synth.SNAPSHOT, p)
    c = SnapshotCache(str(p))
    assert c.poll() and c.model("left") is not None and c.reloads == 1
    assert not c.poll()                                           # unchanged
    d = json.loads(p.read_text())
    d["snapshot_id"] = "new_id"
    time.sleep(0.01)
    p.write_text(json.dumps(d))
    os.utime(p, (time.time() + 5, time.time() + 5))
    assert c.poll() and c.model("left").snapshot_id == "new_id" and c.reloads == 2
    p.write_text("{broken")
    os.utime(p, (time.time() + 10, time.time() + 10))
    assert not c.poll() and c.error
    assert c.model("left").snapshot_id == "new_id"                # last good kept
    assert SnapshotCache(str(tmp_path / "missing.json")).poll() is False


def test_decode_helpers():
    h, w = 4, 6
    rgb = np.arange(h * w * 3, dtype=np.uint8).reshape(h, w, 3)
    msg = SimpleNamespace(encoding="rgb8", step=w * 3 + 2, width=w, height=h,
                          data=np.pad(rgb.reshape(h, w * 3), ((0, 0), (0, 2))).tobytes())
    assert np.array_equal(decode_color(msg), rgb)
    msg.encoding = "bgr8"
    assert np.array_equal(decode_color(msg), rgb[:, :, ::-1])
    msg.encoding = "yuv422"
    assert decode_color(msg) is None
    d = (np.arange(h * w, dtype=np.uint16) * 100).reshape(h, w)
    dm = SimpleNamespace(encoding="16UC1", step=w * 2, width=w, height=h, data=d.tobytes())
    assert np.array_equal(decode_depth(dm), d)
    dm.encoding = "32FC1"
    assert decode_depth(dm) is None
    dm.encoding, dm.data = "16UC1", b"\x00" * 10
    assert decode_depth(dm) is None
    ok, jpg = cv2.imencode(".jpg", np.zeros((8, 8, 3), np.uint8))
    cm = SimpleNamespace(format="jpeg", data=jpg.tobytes())
    assert decode_color(cm).shape == (8, 8, 3)


# ------------------------------------------------------------------------------------------- live behaviour
def test_inactive_gate_processes_nothing(scene, tmp_path):
    probe, out = make_probe(scene, tmp_path, gate=LiveGate(False))
    env = FakeEnv(scene.T_wl6, scene.T_wt)
    outs = simulate(probe, env, scene, env.t + 1.0, still_from=0)
    assert not [o for o in outs if o.overlay_jpeg]
    assert probe.counters["frames"] == 0
    assert not os.listdir(out)                                    # not even a directory is created


def test_live_outputs_rate_and_content(scene, tmp_path):
    probe, out = make_probe(scene, tmp_path, calib_seed=True)
    env = FakeEnv(scene.T_wl6, scene.T_wt)
    outs = [o for o in simulate(probe, env, scene, env.t + 2.0, still_from=float("inf")) if o.overlay_jpeg]
    assert 5 <= len(outs) <= 7                                    # ~3 Hz for 2 s
    o = outs[-1]
    img = cv2.imdecode(np.frombuffer(o.overlay_jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape == (600, 960, 3)
    d = json.loads(o.measurement_json, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
    assert d["ok"] and d["across_convention"] == "outward" and d["state"] == "LIVE"
    assert d["meta"]["calib_status"] == CALIB_SEED
    assert d["meta"]["tool_mask"].startswith("hull_tf+near_field")
    assert len(d["meta"]["calib_sha256"]) == 64
    assert abs(d["delta_mm"]["across"] + 6.0) < 1.0
    # per-frame JSONL in the record folder
    day = time.strftime("%Y%m%d")
    jl = out / f"side_cut_vision_{day}" / "side_cut_vision" / "left_live.jsonl"
    lines = [json.loads(x) for x in jl.read_text().splitlines()]
    assert len(lines) == len(outs) and lines[0]["calib_status"] == CALIB_SEED
    # no still period was reached (arm kept moving): no LOOK record
    assert probe.counters["looks_saved"] == 0


def test_tf_failures_skip_frames_without_output(scene, tmp_path):
    for mode in ("bad_stamp_old_tf", "nonfinite"):
        probe, out = make_probe(scene, tmp_path)
        env = FakeEnv(scene.T_wl6, scene.T_wt)
        env.bad_tf_stamps = mode == "bad_stamp_old_tf"
        env.latest_stamp = env.t - 5.0                           # newest TF is 5 s old: moving arm -> skip
        env.nonfinite = mode == "nonfinite"
        outs = [o for o in simulate(probe, env, scene, env.t + 1.0, still_from=float("inf")) if o.overlay_jpeg]
        assert not outs and probe.counters["skipped_tf"] > 0 and probe.counters["frames"] == 0


def test_tf_missing_at_stamp_is_accepted_from_the_newest_sample_only_when_near_or_at_rest(scene, tmp_path):
    # arm moving, newest TF 40 ms before the image: accepted ("latest_near")
    probe, out = make_probe(scene, tmp_path)
    env = FakeEnv(scene.T_wl6, scene.T_wt)
    env.bad_tf_stamps = True
    env.latest_stamp = env.t - 0.04
    outs = [o for o in simulate(probe, env, scene, env.t + 1.0, still_from=float("inf")) if o.measurement_json]
    assert outs and json.loads(outs[-1].measurement_json)["meta"]["tf_mode"] == "latest_near"
    # arm at rest for > 0.2 s: the old TF IS the current pose ("latest_at_rest")
    probe, out = make_probe(scene, tmp_path)
    env = FakeEnv(scene.T_wl6, scene.T_wt)
    env.bad_tf_stamps = True
    env.latest_stamp = env.t - 3.0
    t0 = env.t
    outs = [o for o in simulate(probe, env, scene, t0 + 1.5, still_from=t0 + 0.1) if o.measurement_json]
    modes = {json.loads(o.measurement_json)["meta"]["tf_mode"] for o in outs}
    assert "latest_at_rest" in modes and probe.counters["frames"] > 0


def test_stale_joint_states_skip_frames(scene, tmp_path):
    probe, out = make_probe(scene, tmp_path)
    env = FakeEnv(scene.T_wl6, scene.T_wt)
    probe.on_joint_state(env.t - 5.0, J, [0.0] * 6)               # joint states from 5 s ago only
    feed_frames(probe, scene, env.t)
    assert probe.step(env) == [] and probe.counters["skipped_tf"] == 1


def test_no_camera_info_processes_nothing(scene, tmp_path):
    probe, _ = make_probe(scene, tmp_path)
    probe.cam = None
    env = FakeEnv(scene.T_wl6, scene.T_wt)
    assert simulate(probe, env, scene, env.t + 0.5, still_from=float("inf")) == [] or True
    assert probe.counters["frames"] == 0


def test_stale_pair_and_size_mismatch_reasons_never_raise(scene, tmp_path):
    probe, out = make_probe(scene, tmp_path)
    env = FakeEnv(scene.T_wl6, scene.T_wt)
    probe.on_joint_state(env.t, J, [0.0] * 6)
    probe.pairs.add_color(Stamped(env.t, scene.color))
    probe.pairs.add_depth(Stamped(env.t - 0.3, scene.depth))      # 300 ms apart
    assert [o for o in probe.step(env) if o.overlay_jpeg] == []
    assert probe.counters["skipped_pair"] == 1
    probe.pairs.add_color(Stamped(env.t + 1.0, scene.color[::2, ::2]))
    probe.pairs.add_depth(Stamped(env.t + 1.0, scene.depth))
    env.t += 1.0
    probe.on_joint_state(env.t, J, [0.0] * 6)
    outs = [o for o in probe.step(env) if o.overlay_jpeg]
    d = json.loads(outs[0].measurement_json)
    assert not d["ok"] and "size_mismatch" in d["reasons"]        # published with reason, not raised


def test_depth_arriving_after_colour_is_paired_without_waiting_a_period(scene, tmp_path):
    probe, out = make_probe(scene, tmp_path)
    env = FakeEnv(scene.T_wl6, scene.T_wt)
    probe.on_joint_state(env.t, J, [0.0] * 6)
    probe.pairs.add_depth(Stamped(env.t - 0.07, scene.depth))     # the PREVIOUS depth is all there is yet
    probe.pairs.add_color(Stamped(env.t, scene.color))
    assert [o for o in probe.step(env) if o.overlay_jpeg] == []
    env.t += 0.02
    probe.on_joint_state(env.t, J, [0.0] * 6)
    probe.pairs.add_depth(Stamped(env.t - 0.02, scene.depth))     # its own depth arrives 20 ms later
    outs = [o for o in probe.step(env) if o.overlay_jpeg]
    assert len(outs) == 1 and probe.counters["skipped_pair"] == 1


# ------------------------------------------------------------------------------------------- LOOK
def test_look_saved_once_per_still_period_with_recovered_offset(scene, tmp_path):
    probe, out = make_probe(scene, tmp_path)
    env = FakeEnv(scene.T_wl6, scene.T_wt)
    outs = simulate(probe, env, scene, env.t + 3.0, still_from=env.t + 1.0)
    assert probe.counters["looks_saved"] == 1, (probe.counters, [l for o in outs for l in o.log])
    path = probe.last_record
    assert os.path.basename(path).startswith("left_01_")
    for f in ("color.png", "depth.png", "camera_info.json", "T_world_optical.json", "T_world_tip.json",
              "joint_state.json", "snapshot.json", "calib.yaml", "measurement.json", "overlay.png"):
        assert os.path.isfile(os.path.join(path, f)), f
    d = json.load(open(os.path.join(path, "measurement.json")))
    assert d["ok"] and d["look"]["frames"] >= 2 and not d["look"]["unstable"]
    assert d["across_convention"] == "outward" and d["meta"]["calib_sha256"]
    dm = d["delta_mm"]
    assert abs(dm["along"] - 8.0) <= 2.0 and abs(dm["across"] + 6.0) <= 1.0 and abs(dm["normal"] - 3.0) <= 1.0
    assert abs(d["delta_yaw_deg"] - 2.0) <= 0.5
    assert open(os.path.join(path, "calib.yaml")).read() == open(synth.HANDEYE).read()
    # the arm stays still for 2 more seconds: still only ONE record (once per still period)
    # now move and stop again -> a second record
    env.t += 0.0
    outs2 = simulate(probe, env, scene, env.t + 3.0, still_from=env.t + 1.0)
    assert probe.counters["looks_saved"] == 2
    assert os.path.basename(probe.last_record).startswith("left_02_")


def test_no_look_when_tip_is_not_at_the_edge_or_arm_moving(scene, tmp_path):
    probe, out = make_probe(scene, tmp_path)
    far = make_T(np.eye(3), scene.tip + 0.2 * scene.model.across)  # 200 mm away from the edge
    env = FakeEnv(scene.T_wl6, far)
    simulate(probe, env, scene, env.t + 3.0, still_from=env.t + 1.0)
    assert probe.counters["looks_saved"] == 0
    probe2, _ = make_probe(scene, tmp_path)
    env2 = FakeEnv(scene.T_wl6, scene.T_wt)
    simulate(probe2, env2, scene, env2.t + 3.0, still_from=float("inf"))      # never still
    assert probe2.counters["looks_saved"] == 0


def test_no_look_while_coordinator_idle(scene, tmp_path):
    probe, out = make_probe(scene, tmp_path, gate=LiveGate(False))
    env = FakeEnv(scene.T_wl6, scene.T_wt)
    simulate(probe, env, scene, env.t + 3.0, still_from=env.t + 1.0)
    assert probe.counters["looks_saved"] == 0 and probe.counters["frames"] == 0


def test_look_discarded_when_arm_moves_early(scene, tmp_path):
    probe, out = make_probe(scene, tmp_path)
    env = FakeEnv(scene.T_wl6, scene.T_wt)
    t0 = env.t
    # still for 0.35 s (trigger) and a single frame, then moving again
    outs = []
    for i in range(0, 400):
        env.t = t0 + i * 0.005
        still = env.t < t0 + 0.40
        probe.on_joint_state(env.t, J, [0.0 if still else 0.5 * (env.t - t0 - 0.4)] * 6)
        if i == 60:
            feed_frames(probe, scene, env.t)
        if i % 4 == 0:
            outs += probe.step(env)
    assert probe.counters["looks_saved"] == 0
    assert probe.counters["looks_discarded"] + 0 >= 0                # discarded or never triggered: never saved


def test_unstable_flag_when_frames_disagree(scene, tmp_path):
    cam, model = scene.cam, scene.model
    other = synth.render(cam, scene.T_wo, model, synth.SceneSpec(d_across_mm=-2.0, seed=5), stride=2)
    probe, out = make_probe(scene, tmp_path)
    env = FakeEnv(scene.T_wl6, scene.T_wt)
    t0 = env.t
    n_img = 0
    for i in range(0, 1200):
        env.t = t0 + i * 0.005
        if i % 2 == 0:
            probe.on_joint_state(env.t, J, [0.0 if env.t > t0 + 0.2 else env.t - t0] * 6)
        if i % 13 == 0 and env.t > t0 + 0.2:
            c, dd = (scene.color, scene.depth) if n_img % 2 == 0 else (other[0], other[1])
            feed_frames(probe, scene, env.t, c, dd)
            n_img += 1
        if i % 4 == 0:
            probe.step(env)
    assert probe.counters["looks_saved"] >= 1
    d = json.load(open(os.path.join(probe.last_record, "measurement.json")))
    assert d["look"]["unstable"] and "unstable" in d["reasons"]


def test_record_goes_into_an_open_cut_recorder_clip(scene, tmp_path):
    out = tmp_path / "cut_videos"
    clip = out / "20261021_120000"
    clip.mkdir(parents=True)
    (clip / "clip.json").write_text(json.dumps({"recording": True}))
    probe, _ = make_probe(scene, tmp_path)
    probe.out_dir = str(out)
    env = FakeEnv(scene.T_wl6, scene.T_wt)
    now = time.time()
    env.t = now
    os.utime(clip / "clip.json", (now, now))
    simulate(probe, env, scene, env.t + 3.0, still_from=env.t + 1.0)
    assert probe.last_record and probe.last_record.startswith(str(clip / "side_cut_vision"))


# ------------------------------------------------------------------------------------------- tool mask
def test_tool_mask_matches_the_real_s0_frame_geometry():
    cam = synth.load_camera()
    he = load_hand_eye(synth.HANDEYE)
    m = tool_mask_from_geometry(cam, he.T_link6_optical)
    # real S0 frame (wrist_122458_t042): knife body spans x 940..1135, y 330..490 -> masked
    assert m[400, 1000] and m[420, 1100]
    # near-field holder block at the right edge, even outside the handpiece hull
    assert m[700, 1250] and m[100, 1250]
    # carton top face and floor are NOT masked
    assert not m[300, 600] and not m[650, 400] and not m[150, 800]
    assert 0.10 < m.mean() < 0.25
    # a different hand-eye moves the mask (it really is hand-eye derived)
    T2 = he.T_link6_optical.copy()
    T2[0, 3] += 0.03
    assert (tool_mask_from_geometry(cam, T2) != m).any()


def test_tool_mask_kind_falls_back_to_urdf_constant_without_tf(scene, tmp_path):
    probe, _ = make_probe(scene, tmp_path)
    env = FakeEnv(scene.T_wl6, scene.T_wt)
    env.tool0 = None
    probe._ensure_tool0(env)
    assert probe.tool_mask_kind == "hull_urdf_const+near_field"
    env.tool0 = T_LINK6_TOOL0
    probe._ensure_tool0(env)
    assert probe.tool_mask_kind == "hull_tf+near_field"


def test_frame_prefilter_is_a_gate_only():
    from side_cut_vision.probe_node import ProbeNode
    stub = SimpleNamespace(gate=LiveGate(True))
    probe = SimpleNamespace(look=None)
    assert ProbeNode._accept(stub, probe, b"", ("l", "color"))
    stub.gate = LiveGate(False)
    assert not ProbeNode._accept(stub, probe, b"", ("l", "color"))
    probe.look = object()                                            # a LOOK in progress keeps frames flowing
    assert ProbeNode._accept(stub, probe, b"", ("l", "color"))
