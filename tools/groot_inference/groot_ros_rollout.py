#!/usr/bin/env python3
"""
groot_ros_rollout.py -- closed-loop policy client for the real single igus ReBeL arm (ROS 2 Humble).  Two backends (config key
policy.backend): groot (GR00T N1.7, default, rollout_config.yaml) and turbovla (TurboVLA, rollout_config_turbovla.yaml, see the
"TurboVLA backend" section below and README.md).

Every cycle: grab the head camera + the gripper camera of the controlled arm + the 6 joints of that (single) arm, preprocess the images
exactly like the mocap recorder did for the training data (preprocess.mode recorder_pil), ask the GR00T policy server (ZeroMQ + msgpack-numpy, same protocol as Isaac-GR00T) for a 16-step chunk of
ABSOLUTE joint targets, run the safety gates on it, send the first `execution_horizon` steps through the
FollowJointTrajectory action, wait for the result, query again.

Modes
  observe  (default)  queries the policy in a loop and logs chunks + gate verdicts. Never commands motion.
  execute             --execute AND a typed `yes` (the hardware type is printed first). Moves the arm.

    ROS_DOMAIN_ID=<yours> ROS_LOCALHOST_ONLY=1 source /opt/ros/humble/setup.bash && source ~/igus_rebel_ros2/install/setup.bash
    ~/venvs/gr00t_client/bin/python groot_ros_rollout.py --check                    # inputs, hardware, limits; no server
    ~/venvs/gr00t_client/bin/python groot_ros_rollout.py --max-queries 5            # observe
    ~/venvs/gr00t_client/bin/python groot_ros_rollout.py --execute                  # asks for `yes`

Stop: Ctrl+C, Enter / q+Enter, or any exception -> cancel the active goal and command a hold at the current position.
This is NOT a substitute for the e-stop.

TurboVLA backend: each cycle = snapshot -> 224x224 preprocessing -> get_action (30x14 absolute chunk) -> ACT temporal ensemble ->
gates -> ONE FollowJointTrajectory goal to the ensembled 6-joint target of the controlled arm -> wait until it ran -> next cycle
(step-synchronous, stop-and-go).  The arm is chosen by the config (model.arm_channels [0, 6] = left, [7, 13] = right) and cross-checked
against the server's announced arm_channels.

    ~/venvs/gr00t_client/bin/python groot_ros_rollout.py --config rollout_config_turbovla.yaml --max-queries 5     # observe, LEFT arm
    ~/venvs/gr00t_client/bin/python groot_ros_rollout.py --config rollout_config_turbovla.yaml --execute --max-steps 60
    ~/venvs/gr00t_client/bin/python groot_ros_rollout.py --config rollout_config_turbovla_right.yaml --max-queries 5   # observe, RIGHT arm

Needs rclpy (system), numpy, opencv, pyyaml, pyzmq, msgpack, msgpack-numpy (see README.md: ~/venvs/gr00t_client).
"""
import argparse
import collections
import copy
import datetime
import io
import json
import math
import numbers
import os
import select
import signal
import sys
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "rollout_config.yaml"

EXIT_OK, EXIT_ERROR, EXIT_USAGE, EXIT_REFUSED, EXIT_INPUTS, EXIT_EXEC, EXIT_STOPPED = 0, 1, 2, 3, 4, 5, 130
MOCK_PLUGIN = "mock_components/GenericSystem"


class StopRun(Exception):
    """End the run abnormally (in execute mode: cancel the goal and hold)."""

    def __init__(self, code, reason):
        super().__init__(reason)
        self.code, self.reason = code, reason


class ServerError(Exception):
    pass


class ServerTimeout(ServerError):
    pass


class Interrupted(Exception):
    pass


def say(msg=""):
    print(f"[groot] {msg}", flush=True)


# ---------------------------------------------------------------------------------------------------------------- config
def deep_set(cfg, dotted, value):
    keys = dotted.split(".")
    d = cfg
    for k in keys[:-1]:
        if k not in d or not isinstance(d[k], dict):
            raise KeyError(f"--set {dotted}: no such section {k!r}")
        d = d[k]
    if keys[-1] not in d:
        raise KeyError(f"--set {dotted}: unknown key (all keys must exist in the config)")
    d[keys[-1]] = value


def load_config(path, sets=()):
    cfg = yaml.safe_load(Path(path).read_text())
    for s in sets:
        if "=" not in s:
            raise ValueError(f"--set expects key=value, got {s!r}")
        k, v = s.split("=", 1)
        deep_set(cfg, k.strip(), yaml.safe_load(v))
    cfg["_path"] = str(path)
    return cfg


def backend_of(cfg):
    """policy.backend: groot (default when the key is absent) | turbovla."""
    return str((cfg.get("policy") or {}).get("backend", "groot"))


def validate_turbovla_config(cfg, args):
    """Raise ValueError/KeyError for a config that cannot drive the TurboVLA backend (all keys are required, like the GR00T config)."""
    need = {"model": ["recorder_size", "image_size", "state_dim", "action_dim", "chunk_len", "control_rate_hz", "server_cameras", "arm_channels"],
            "ensemble": ["horizon", "decay"],
            "gates": ["first_step_max_rad", "row_step_max_rad", "max_target_step_rad", "urdf_inset_rad", "training_margin_rad",
                      "state_in_training_range", "check_all_rows"],
            "execution": ["max_joint_vel", "max_steps", "max_time_s", "confirm_timeout_s", "hold_time_s", "result_margin_s", "settle_tol_rad",
                          "settle_timeout_s", "tracking_error_rad", "inflight_joint_stale_s", "max_measured_vel_factor"],
            "logging": ["run_root", "fallback_root", "save_images", "image_quality", "log_full_chunk", "print_every"],
            "preprocess": ["mode"], "observe": ["period_s", "max_queries"], "training_range": ["min", "max"],
            "freshness": ["max_image_age_s", "max_joint_age_s", "max_image_header_age_s", "check_image_size"],
            "ros": ["joint_states_topic", "robot_description_topic", "joints", "action", "controller_manager", "wait_for_inputs_s"]}
    for sect, keys in need.items():
        if not isinstance(cfg.get(sect), dict):
            raise ValueError(f"turbovla config: section {sect!r} missing")
        for k in keys:
            if k not in cfg[sect]:
                raise ValueError(f"turbovla config: {sect}.{k} missing")
    for k in ("cameras", "server", "instruction"):
        if k not in cfg:
            raise ValueError(f"turbovla config: {k!r} missing")
    if sorted(cfg["cameras"]) != sorted(TV_CAMERAS):
        raise ValueError(f"turbovla config: cameras must be exactly {list(TV_CAMERAS)}")
    m, e = cfg["model"], cfg["execution"]
    if args.horizon is not None:
        raise ValueError("--horizon is a GR00T option (the TurboVLA backend always executes one ensembled target per cycle)")
    if (m["chunk_len"], m["action_dim"], m["state_dim"]) != (30, 14, 14) or cfg["ensemble"]["horizon"] != m["chunk_len"]:
        raise ValueError("turbovla contract: chunk_len 30, action_dim 14, state_dim 14, ensemble.horizon == chunk_len")
    if not (isinstance(cfg["ensemble"]["decay"], (int, float)) and cfg["ensemble"]["decay"] >= 0):
        raise ValueError("ensemble.decay must be a number >= 0")
    if not 0 < e["max_joint_vel"] <= 0.5:
        raise ValueError("execution.max_joint_vel must be in (0, 0.5] rad/s")
    if not m["control_rate_hz"] > 0:
        raise ValueError("model.control_rate_hz must be > 0")
    if len(cfg["training_range"]["min"]) != 6 or len(cfg["training_range"]["max"]) != 6 or len(cfg["ros"]["joints"]) != 6:
        raise ValueError("6 joints / training_range entries expected")
    tv_arm_channels(m)  # raises ValueError for a malformed model.arm_channels


def jsonable(o):
    """JSON-safe copy: numpy -> lists, non-finite floats -> strings."""
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.floating, float)):
        o = float(o)
        return o if math.isfinite(o) else str(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


# ------------------------------------------------------------------------------------------------- image preprocessing
def decode_compressed(data):
    img = cv2.imdecode(np.asarray(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("cannot decode CompressedImage as a colour image")
    return img  # BGR


def to_model_rgb(bgr, pp, video_size):
    """LEGACY OpenCV path (preprocess.mode: opencv_legacy; recipe of the old lid_open recordings, NOT the mocap checkpoint):
    resize(scale, INTER_AREA) -> JPEG round trip -> resize to video_size INTER_AREA -> RGB. Aspect ratio not preserved."""
    if pp["scale"] != 1.0:
        bgr = cv2.resize(bgr, None, fx=pp["scale"], fy=pp["scale"], interpolation=cv2.INTER_AREA)
    if pp["jpeg_roundtrip"]:
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(pp["jpeg_quality"])])
        if not ok:
            raise ValueError("JPEG round trip failed")
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    w, h = video_size
    bgr = cv2.resize(bgr, (int(w), int(h)), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(bgr[:, :, ::-1])


def recorder_rgb(data, video_size):
    """Exact copy of the mocap recorder's path (~/robot_remote/recording/record_dataset.py image_msg_to_rgb + resize_rgb):
    PIL.Image.open(CompressedImage bytes) -> img.draft("RGB", (W, H)) (JPEG DCT downscale, never below W x H) -> convert("RGB")
    -> if not W x H: PIL resize((W, H), BILINEAR). No JPEG round trip; aspect ratio NOT preserved (16:9 is squashed to 4:3).
    Returns (rgb uint8 (H, W, 3), (full_w, full_h) of the image as published, before the draft)."""
    from PIL import Image
    w, h = int(video_size[0]), int(video_size[1])
    try:
        img = Image.open(io.BytesIO(data.tobytes() if isinstance(data, np.ndarray) else bytes(data)))
        full = img.size
        img.draft("RGB", (w, h))
        a = np.asarray(img.convert("RGB"))
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"cannot decode CompressedImage with PIL: {e}") from e
    if a.shape[1] != w or a.shape[0] != h:
        a = np.asarray(Image.fromarray(a).resize((w, h), Image.BILINEAR))
    return np.ascontiguousarray(a), (int(full[0]), int(full[1]))


def preprocess_image(data, pp, video_size):
    """CompressedImage bytes -> (model RGB uint8 (H, W, 3), (full_w, full_h)) according to preprocess.mode."""
    mode = pp["mode"]
    if mode == "recorder_pil":
        return recorder_rgb(data, video_size)
    if mode == "opencv_legacy":
        bgr = decode_compressed(data)
        return to_model_rgb(bgr, pp, video_size), (bgr.shape[1], bgr.shape[0])
    raise ValueError(f"preprocess.mode {mode!r} unknown (recorder_pil | opencv_legacy)")


def build_observation(videos, q, instruction, model):
    """videos: {video_key: HxWx3 uint8 RGB}; layout identical to the friend's gr00t_rollout.build_observation."""
    return {
        "video": {k: v[None, None] for k, v in videos.items()},                          # (1,1,H,W,3) uint8
        "state": {model["state_key"]: np.asarray(q, dtype=np.float32)[None, None]},     # (1,1,6) float32
        "language": {model["language_key"]: [[instruction]]},
    }


# ---------------------------------------------------------------------------------------------------- chunk handling
def normalize_chunk(raw, chunk_len, dof):
    """-> (float64 array (chunk_len, dof) or None, problem string or None). Accepts (16,6) / (1,16,6) / (1,1,16,6)."""
    try:
        a = np.asarray(raw)
    except Exception as e:  # noqa: BLE001
        return None, f"not an array: {e}"
    if a.dtype.kind not in "fiu":
        return None, f"dtype {a.dtype} is not numeric"
    while a.ndim > 2 and a.shape[0] == 1:
        a = a[0]
    if a.ndim != 2 or a.shape != (chunk_len, dof):
        return None, f"shape {tuple(np.asarray(raw).shape)} is not (…,{chunk_len},{dof})"
    return a.astype(np.float64), None


def evaluate_gates(chunk, q_now, limits, gates, train_min, train_max, horizon):
    """All gates, never short-circuited.  chunk (N,6) float64 or None; limits (6,2).  -> list of {name, ok, detail}."""
    out = []

    def add(name, ok, detail):
        out.append({"name": name, "ok": bool(ok), "detail": detail})

    q_now = np.asarray(q_now, dtype=np.float64)
    train_lo, train_hi = np.asarray(train_min) - gates["training_margin_rad"], np.asarray(train_max) + gates["training_margin_rad"]
    # the state the model starts from
    if gates["state_in_training_range"]:
        bad = (q_now < train_lo) | (q_now > train_hi)
        add("state_in_training_range", not bad.any(),
            "ok" if not bad.any() else f"state joint(s) {[int(i) + 1 for i in np.flatnonzero(bad)]} outside training range +-{gates['training_margin_rad']}: "
            + ", ".join(f"j{i + 1}={q_now[i]:.3f} not in [{train_lo[i]:.3f},{train_hi[i]:.3f}]" for i in np.flatnonzero(bad)))
    if chunk is None:
        return out
    full = chunk if gates["validate_full_chunk"] else chunk[:horizon]
    exe = chunk[:horizon]
    finite = np.isfinite(chunk).all()
    add("finite", finite, "ok" if finite else f"{int((~np.isfinite(chunk)).sum())} non-finite value(s)")
    if not finite:
        return out
    d0 = np.abs(chunk[0] - q_now)
    add("first_step", d0.max() <= gates["first_step_max_rad"],
        f"max |chunk[0]-state| = {d0.max():.4f} rad (joint {int(d0.argmax()) + 1}), limit {gates['first_step_max_rad']}")
    steps = np.abs(np.diff(full, axis=0)) if len(full) > 1 else np.zeros((1, chunk.shape[1]))
    add("step_jump", steps.max() <= gates["max_step_rad"],
        f"max step {steps.max():.4f} rad (step {int(np.unravel_index(steps.argmax(), steps.shape)[0]) + 1}->{int(np.unravel_index(steps.argmax(), steps.shape)[0]) + 2}, "
        f"joint {int(np.unravel_index(steps.argmax(), steps.shape)[1]) + 1}), limit {gates['max_step_rad']}")
    u_lo, u_hi = limits[:, 0] + gates["urdf_inset_rad"], limits[:, 1] - gates["urdf_inset_rad"]
    viol = (full < u_lo) | (full > u_hi)
    add("urdf_limits", not viol.any(),
        "ok" if not viol.any() else "; ".join(f"step {r + 1} j{c + 1}={full[r, c]:.3f} not in [{u_lo[c]:.3f},{u_hi[c]:.3f}]"
                                              for r, c in list(zip(*np.nonzero(viol)))[:3]))
    viol = (full < train_lo) | (full > train_hi)
    add("training_range", not viol.any(),
        "ok" if not viol.any() else "; ".join(f"step {r + 1} j{c + 1}={full[r, c]:.3f} not in [{train_lo[c]:.3f},{train_hi[c]:.3f}]"
                                              for r, c in list(zip(*np.nonzero(viol)))[:3]))
    disp = np.abs(exe - q_now)
    add("chunk_displacement", disp.max() <= gates["max_chunk_disp_rad"],
        f"max |step-state| over executed horizon = {disp.max():.4f} rad, limit {gates['max_chunk_disp_rad']}")
    return out


def trajectory_times(q_now, qs, dt, vmax):
    """Cumulative time_from_start per point: nominal dt per step, stretched per segment so that
    max_j |dq_j| / segment_time <= vmax.  Never shorter than dt."""
    prev = np.asarray(q_now, dtype=np.float64)
    t, times = 0.0, []
    for q in qs:
        need = float(np.max(np.abs(q - prev))) / vmax if vmax > 0 else 0.0
        t += max(dt, need)
        times.append(t)
        prev = q
    return np.array(times)


def peak_speed(ts, qs, window):
    """Largest per-joint speed (rad/s) measured over >= `window` seconds, from samples ts (n,), qs (n,6)."""
    ts, qs = np.asarray(ts), np.asarray(qs)
    if len(ts) < 2:
        return 0.0
    j = np.searchsorted(ts, ts + window)
    ok = j < len(ts)
    if not ok.any():
        return 0.0
    i = np.flatnonzero(ok)
    v = np.abs(qs[j[i]] - qs[i]) / (ts[j[i]] - ts[i])[:, None]
    return float(v.max())


def parse_robot_description(xml_text, joints):
    """-> (limits (6,2) in `joints` order, [ {name, type, plugin, params{}} ] for ros2_control hardware)."""
    root = ET.fromstring(xml_text)
    lim = {}
    for j in root.iter("joint"):
        lj = j.find("limit")
        if j.get("name") in joints and lj is not None and lj.get("lower") is not None:
            lim[j.get("name")] = (float(lj.get("lower")), float(lj.get("upper")))
    missing = [n for n in joints if n not in lim]
    if missing:
        raise RuntimeError(f"/robot_description has no limits for {missing}")
    hw = []
    for rc in root.iter("ros2_control"):
        h = rc.find("hardware")
        if h is None:
            continue
        plugin = h.findtext("plugin", default="").strip()
        params = {p.get("name"): (p.text or "").strip() for p in h.findall("param")}
        hw.append({"name": rc.get("name"), "type": rc.get("type"), "plugin": plugin, "params": params,
                   "joints": [j.get("name") for j in rc.findall("joint")]})
    return np.array([lim[n] for n in joints]), hw


def hardware_is_mock(components):
    """True only if there is at least one component and every one is mock_components/GenericSystem."""
    return bool(components) and all(c.get("class_type") == MOCK_PLUGIN for c in components)


def parse_modality_reply(reply):
    """get_modality_config reply -> {modality: {"keys": [...], "n": len(delta_indices)}} or None if not understood.
    Isaac-GR00T encodes each ModalityConfig as {"__ModalityConfig__": True, "as_json": <json str or dict>}."""
    if not isinstance(reply, dict) or not reply:
        return None
    out = {}
    for mod, v in reply.items():
        payload = v
        if isinstance(v, dict) and ("as_json" in v or b"as_json" in v):
            payload = v.get("as_json", v.get(b"as_json"))
        if isinstance(payload, (bytes, bytearray)):
            payload = payload.decode()
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except ValueError:
                continue
        if isinstance(payload, dict) and payload.get("modality_keys") is not None:
            d = payload.get("delta_indices")
            out[str(mod)] = {"keys": [str(k) for k in payload["modality_keys"]], "n": None if d is None else len(d)}
    return out or None


def compare_modality(info, cfg):
    """Problems between what the server's checkpoint expects and what this client sends/reads."""
    m, probs = cfg["model"], []
    want = {"video": sorted(cfg["cameras"]), "state": [m["state_key"]], "action": [m["action_key"]], "language": [m["language_key"]]}
    for mod, keys in want.items():
        if mod in info and sorted(info[mod]["keys"]) != sorted(keys):
            probs.append(f"{mod} keys: server expects {sorted(info[mod]['keys'])}, client uses {sorted(keys)}")
    if "action" in info and info["action"]["n"] is not None and info["action"]["n"] != m["chunk_len"]:
        probs.append(f"action horizon: server {info['action']['n']} steps, client model.chunk_len {m['chunk_len']}")
    return probs


# ============================================================================================================ TurboVLA backend
# Pure functions of the TurboVLA backend (unit-tested by run_mock_test.sh).  The wire protocol and the model contract are fixed:
# 14-D state [left j1..6, left gripper, right j1..6, right gripper] with everything but the controlled arm's 6 joints ZERO (training
# contract, normalisation std 0.01 on those channels), two 224x224 RGB cameras (head, wrist_left; the knife slot is filled black by the
# server), a (30, 14) chunk of absolute radians of which only the controlled arm's 6 channels are used.  Which arm that is comes from
# the REQUIRED config key model.arm_channels ([0, 6] = left joints, [7, 13] = right joints; half-open channel range) and is
# cross-checked against the server's announced arm_channels.  The camera SLOT names (head, wrist_left) are the model's, not the arm's:
# the right-arm checkpoint was trained with the right gripper camera in the wrist_left slot.
TV_CAMERAS = ("head", "wrist_left")


def tv_arm_channels(model):
    """model.arm_channels [first, end) -> slice of the 14 model channels holding the controlled arm's 6 joints.  ValueError if malformed."""
    ac = model.get("arm_channels") if isinstance(model, dict) else None
    ok = (isinstance(ac, (list, tuple)) and len(ac) == 2 and all(isinstance(v, int) and not isinstance(v, bool) for v in ac)
          and 0 <= ac[0] and ac[1] - ac[0] == 6 and ac[1] <= int(model.get("state_dim", 14)))
    if not ok:
        raise ValueError(f"model.arm_channels must be [first, first+6] inside the {model.get('state_dim', 14)} channels (left [0, 6], right [7, 13]), got {ac!r}")
    return slice(int(ac[0]), int(ac[1]))


class ACTTemporalEnsembler:
    """Causal ACT temporal aggregation for absolute-position action chunks.

    Verbatim copy (docstring shortened) of ACTTemporalEnsembler in
    compal-turbovla-handoff-v1/source/experiments/compal/act_temporal.py (TurboVLA handoff bundle), copied because this client runs on the
    system Python 3.10 without the bundle.  Reference: https://github.com/tonyzhaozh/act/blob/main/imitate_episodes.py

    One fresh [horizon, action_dim] chunk per logical control step.  At step t the chunk issued at query s contributes its element t-s
    (never element 0 of an earlier chunk); contributing queries are ordered oldest first and weighted exp(-decay * i), i = 0, 1, ...
    (positive decay = older queries weigh more, the original ACT convention).  Steps must be consecutive from 0; reset at episode start.
    """

    def __init__(self, horizon=30, action_dim=14, decay=.01):
        if not isinstance(horizon, numbers.Integral) or horizon < 1:
            raise ValueError('horizon must be a positive integer')
        if not isinstance(action_dim, numbers.Integral) or action_dim < 1:
            raise ValueError('action_dim must be a positive integer')
        if not np.isfinite(decay) or decay < 0:
            raise ValueError('decay must be finite and nonnegative')
        self.horizon, self.action_dim, self.decay = horizon, action_dim, float(decay)
        self.reset()

    def reset(self):
        self._history = collections.deque(maxlen=self.horizon)
        self._next_step = 0

    def step(self, chunk, *, step):
        if not isinstance(step, numbers.Integral) or step != self._next_step:
            raise ValueError(f'Expected fresh chunk for step {self._next_step}, got {step}; reset at episode start')
        values = np.asarray(chunk, dtype=np.float64)
        if values.shape != (self.horizon, self.action_dim) or not np.isfinite(values).all():
            raise ValueError('Expected finite [horizon, action_dim] chunk')
        self._history.append((int(step), values.copy()))
        candidates = np.stack([prediction[step - query] for query, prediction in self._history])
        weights = np.exp(-self.decay * np.arange(len(candidates), dtype=np.float64))
        result = np.sum(candidates * (weights / weights.sum())[:, None], axis=0)
        self._next_step += 1
        return result

    @property
    def history_size(self):
        return len(self._history)

    @property
    def next_step(self):
        return self._next_step


def turbovla_resize(rgb, size):
    """The bundle's per-frame resize, called the same way: Image.fromarray(rgb).resize((224, 224)) with Pillow's DEFAULT resample
    (BICUBIC for RGB since Pillow 7.0; Pillow 9.0.1 here and 12.3.0 in the bundle give bit-identical output)."""
    from PIL import Image
    return np.ascontiguousarray(np.asarray(Image.fromarray(rgb).resize((int(size[0]), int(size[1])))))


def turbovla_rgb(data, model, pp):
    """CompressedImage bytes -> (224x224 RGB uint8, (full_w, full_h)): the recorder path to model.recorder_size (recorder_rgb; 640x480 for
    the left-arm checkpoint = the GR00T recorder_pil path, 1152x648 for the right-arm checkpoint = its source videos), then ONE PIL resize
    to 224x224 (what TurboVLA's training data went through)."""
    if pp["mode"] != "recorder_pil":
        raise ValueError(f"preprocess.mode {pp['mode']!r} unsupported for turbovla (recorder_pil only)")
    rgb640, full = recorder_rgb(data, model["recorder_size"])
    return turbovla_resize(rgb640, model["image_size"]), full


def build_observation_turbovla(images, q, step, arm, state_dim=14):
    """Request data["observation"]: state float64 (14,) = the 6 measured joints of the controlled arm in channels `arm` (a slice, see
    tv_arm_channels), every other channel (other arm, both grippers) exactly ZERO; uint8 (224,224,3) images."""
    state = np.zeros(int(state_dim), dtype=np.float64)
    state[arm] = np.asarray(q, dtype=np.float64)
    return {"state": state, "head": images["head"], "wrist_left": images["wrist_left"], "step": int(step)}


def tv_normalize_chunk(raw, horizon, dim):
    """-> (float64 (horizon, dim) or None, problem).  Strict: exactly a numeric 2-D array of that shape."""
    try:
        a = np.asarray(raw)
    except Exception as e:  # noqa: BLE001
        return None, f"not an array: {e}"
    if a.dtype.kind not in "fiu":
        return None, f"dtype {a.dtype} is not numeric"
    if a.shape != (horizon, dim):
        return None, f"shape {tuple(a.shape)} is not ({horizon},{dim})"
    return a.astype(np.float64), None


def tv_state_gate(q_now, gates, train_min, train_max):
    q_now = np.asarray(q_now, dtype=np.float64)
    lo, hi = np.asarray(train_min) - gates["training_margin_rad"], np.asarray(train_max) + gates["training_margin_rad"]
    bad = (q_now < lo) | (q_now > hi)
    return {"name": "state_in_training_range", "ok": bool(not bad.any()),
            "detail": "ok" if not bad.any() else f"state joint(s) {[int(i) + 1 for i in np.flatnonzero(bad)]} outside training range +-{gates['training_margin_rad']}: "
            + ", ".join(f"j{i + 1}={q_now[i]:.3f} not in [{lo[i]:.3f},{hi[i]:.3f}]" for i in np.flatnonzero(bad))}


def _limit_gates(vals, limits, gates, train_min, train_max, prefix, where):
    """urdf_limits + training_range gates for vals (n,6); `where` names the thing checked ('step'/'row')."""
    out = []
    u_lo, u_hi = limits[:, 0] + gates["urdf_inset_rad"], limits[:, 1] - gates["urdf_inset_rad"]
    t_lo, t_hi = np.asarray(train_min) - gates["training_margin_rad"], np.asarray(train_max) + gates["training_margin_rad"]
    for name, lo, hi in ((prefix + "urdf_limits", u_lo, u_hi), (prefix + "training_range", t_lo, t_hi)):
        viol = (vals < lo) | (vals > hi)
        out.append({"name": name, "ok": bool(not viol.any()),
                    "detail": "ok" if not viol.any() else "; ".join(f"{where} {r + 1} j{c + 1}={vals[r, c]:.3f} not in [{lo[c]:.3f},{hi[c]:.3f}]"
                                                                    for r, c in list(zip(*np.nonzero(viol)))[:3])})
    return out


def evaluate_gates_turbovla(chunk, q_now, limits, gates, train_min, train_max, arm):
    """Raw-chunk gates (never short-circuited).  chunk (30,14) float64 or None, limits (6,2), arm = slice of the controlled arm's 6
    channels.  -> [{name, ok, detail}].  Only those channels are judged; all others are ignored (but must be finite)."""
    out = []
    if gates["state_in_training_range"]:
        out.append(tv_state_gate(q_now, gates, train_min, train_max))
    if chunk is None:
        return out
    finite = bool(np.isfinite(chunk).all())
    out.append({"name": "finite", "ok": finite, "detail": "ok" if finite else f"{int((~np.isfinite(chunk)).sum())} non-finite value(s)"})
    if not finite:
        return out
    q_now = np.asarray(q_now, dtype=np.float64)
    rows = chunk[:, arm] if gates["check_all_rows"] else chunk[:1, arm]
    d0 = np.abs(chunk[0, arm] - q_now)
    out.append({"name": "first_step", "ok": bool(d0.max() <= gates["first_step_max_rad"]),
                "detail": f"max |chunk[0]-state| = {d0.max():.4f} rad (joint {int(d0.argmax()) + 1}), limit {gates['first_step_max_rad']}"})
    if len(rows) > 1:
        st = np.abs(np.diff(rows, axis=0))
        r, c = np.unravel_index(st.argmax(), st.shape)
        out.append({"name": "row_step", "ok": bool(st.max() <= gates["row_step_max_rad"]),
                    "detail": f"max row-to-row step {st.max():.4f} rad (row {r + 1}->{r + 2}, joint {c + 1}), limit {gates['row_step_max_rad']}"})
    else:
        out.append({"name": "row_step", "ok": True, "detail": "single row checked (gates.check_all_rows=false)"})
    out += _limit_gates(rows, limits, gates, train_min, train_max, "", "row")
    return out


def evaluate_target_gates(target, q_now, limits, gates, train_min, train_max):
    """Gates on the ENSEMBLED 6-joint target of the controlled arm that is about to be commanded."""
    target = np.asarray(target, dtype=np.float64)
    out = []
    finite = bool(np.isfinite(target).all())
    out.append({"name": "target_finite", "ok": finite, "detail": "ok" if finite else "non-finite ensembled target"})
    if not finite:
        return out
    d = np.abs(target - np.asarray(q_now, dtype=np.float64))
    out.append({"name": "target_step", "ok": bool(d.max() <= gates["max_target_step_rad"]),
                "detail": f"max |ensembled target - state| = {d.max():.4f} rad (joint {int(d.argmax()) + 1}), limit {gates['max_target_step_rad']}"})
    out += _limit_gates(target[None], limits, gates, train_min, train_max, "target_", "step")
    return out


def chunk_summary(chunk, q_now, arm):
    """Compact log record of a raw (30,14) chunk (the controlled arm's channels `arm`; the other channels are reduced to their abs max)."""
    c6 = chunk[:, arm]
    ignored = np.delete(chunk, np.arange(chunk.shape[1])[arm], axis=1)
    return {"shape": list(chunk.shape), "row0": c6[0], "row_last": c6[-1], "min": c6.min(axis=0), "max": c6.max(axis=0),
            "row0_minus_state_max": float(np.abs(c6[0] - q_now).max()),
            "max_row_step": float(np.abs(np.diff(c6, axis=0)).max()) if len(c6) > 1 else 0.0,
            "ignored_channels_absmax": float(np.abs(ignored).max()) if ignored.size else 0.0}


def compare_modality_turbovla(reply, cfg):
    """-> (problems, info).  reply = get_modality_config result.  info None = not understood (empty / not a dict): the caller only warns.
    A reply without backend == 'turbovla' (e.g. a GR00T server) is a mismatch."""
    if not isinstance(reply, dict) or not reply:
        return [], None
    m, probs = cfg["model"], []
    if reply.get("backend") != "turbovla":
        probs.append(f"backend: server announces {reply.get('backend')!r} (keys {sorted(reply)[:6]}), this config needs 'turbovla'")
        return probs, reply

    def same(a, b):
        if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
            return isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and list(a) == list(b)
        if isinstance(a, bool) or isinstance(b, bool) or not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            return a == b
        return float(a) == float(b)

    def expect(key, want, got=None):
        got = reply.get(key) if got is None else got
        if not same(got, want):
            probs.append(f"{key}: server {reply.get(key)!r}, client expects {want!r}")
    expect("client_cameras", sorted(cfg["cameras"]), sorted(reply["client_cameras"]) if isinstance(reply.get("client_cameras"), (list, tuple)) else None)
    expect("cameras", list(m["server_cameras"]))
    expect("image_size", list(m["image_size"]))
    expect("state_dim", int(m["state_dim"]))
    expect("action_dim", int(m["action_dim"]))
    expect("horizon", int(m["chunk_len"]))
    expect("control_rate_hz", m["control_rate_hz"])
    # a server that does not announce arm_channels is a pre-right-arm server: it controls the LEFT arm, channels [0, 6]
    announced = reply.get("arm_channels", [0, 6])
    if not same(list(announced) if isinstance(announced, (list, tuple)) else announced, list(m["arm_channels"])):
        probs.append(f"arm_channels: server controls channels {announced!r}" + ("" if "arm_channels" in reply else " (not announced: pre-right-arm server = left arm)")
                     + f", client config model.arm_channels is {list(m['arm_channels'])!r}")
    if m.get("language") is not None:
        expect("language", m["language"])
    want_sha = cfg["server"].get("expect_checkpoint_sha256")
    if want_sha and reply.get("checkpoint_sha256") != want_sha:
        probs.append(f"checkpoint_sha256: server {reply.get('checkpoint_sha256')!r}, config expects {want_sha!r}")
    return probs, reply


# --------------------------------------------------------------------------------------------------------- server client
class Gr00tClient:
    """Isaac-GR00T PolicyClient protocol (ZeroMQ REQ, msgpack-numpy), without importing gr00t.  Interruptible."""

    def __init__(self, host, port, timeout_s, api_token=None, label="GR00T"):
        import zmq
        self.label = label
        self.zmq = zmq
        self.ctx = zmq.Context()
        self.addr = f"tcp://{host}:{port}"
        self.timeout_s = float(timeout_s)
        self.api_token = api_token
        self.sock = None
        self._connect()

    def _connect(self):
        if self.sock is not None:
            self.sock.close(linger=0)
        self.sock = self.ctx.socket(self.zmq.REQ)
        self.sock.setsockopt(self.zmq.LINGER, 0)
        self.sock.setsockopt(self.zmq.SNDTIMEO, int(self.timeout_s * 1000))
        self.mon = None
        if os.environ.get("GROOT_ZMQ_DEBUG"):  # debugging aid: remember the socket's connect/disconnect events
            self.mon = self.sock.get_monitor_socket()
        self.sock.connect(self.addr)

    def monitor_events(self):
        out = []
        if self.mon is not None:
            from zmq.utils.monitor import recv_monitor_message
            while self.mon.poll(0):
                m = recv_monitor_message(self.mon)
                out.append(f"{m['event'].name if hasattr(m['event'], 'name') else m['event']}:{m['value']}")
        return out

    @staticmethod
    def _decode_hook(obj):
        import msgpack_numpy as mnp
        if isinstance(obj, dict):
            marker = obj.get("__ndarray_class__", obj.get(b"__ndarray_class__"))
            if marker:  # Isaac-GR00T's np.save form (allow_pickle=False)
                return np.load(io.BytesIO(obj.get("as_npy", obj.get(b"as_npy"))), allow_pickle=False)
            if obj.get(b"nd", obj.get("nd")) and obj.get(b"kind", obj.get("kind")) in (b"O", "O"):
                raise ValueError("refusing object-dtype array from server")
        return mnp.decode(obj)

    def call(self, endpoint, data=None, timeout_s=None, stop_event=None, retries=0):
        """One request/reply.  On a timeout the REQ socket is recreated and the request re-sent up to `retries` times
        (lazy-pirate pattern: a request or reply was occasionally lost right after a connect, see README).  Only used for
        idempotent endpoints.  `self.last_retries` = how many re-sends the last call needed."""
        import msgpack
        import msgpack_numpy as mnp
        timeout_s = self.timeout_s if timeout_s is None else timeout_s
        req = {"endpoint": endpoint}
        if data is not None:
            req["data"] = data
        if self.api_token:
            req["api_token"] = self.api_token
        payload = msgpack.packb(req, default=mnp.encode)
        self.last_retries = 0
        self._last_events = None
        for attempt in range(retries + 1):
            self.last_retries = attempt
            got = False
            try:
                self.sock.send(payload)
                deadline = time.monotonic() + timeout_s
                while True:
                    if self.sock.poll(50, self.zmq.POLLIN):
                        got = True
                        break
                    if stop_event is not None and stop_event.is_set():
                        self._connect()
                        raise Interrupted()
                    if time.monotonic() > deadline:
                        self._last_events = self.monitor_events() if self.mon is not None else None
                        self._connect()  # a REQ socket that never got its reply is unusable
                        break
            except self.zmq.error.Again:
                self._connect()
            if got:
                break
            if attempt == retries:
                raise ServerTimeout(f"{self.label} server {self.addr}: no reply to {endpoint!r} within {timeout_s:g} s"
                                    + (f" (after {attempt} re-send)" if attempt else "")
                                    + (f" zmq events: {self._last_events}" if self.mon is not None else ""))
            say(f"no reply to {endpoint!r} within {timeout_s:g} s: re-sending on a fresh socket ({attempt + 1}/{retries})"
                + (f" zmq events: {self._last_events}" if self.mon is not None else ""))
        msg = self.sock.recv()
        if msg == b"ERROR":
            raise ServerError("server replied ERROR (see its console)")
        res = msgpack.unpackb(msg, object_hook=self._decode_hook, raw=False)
        if isinstance(res, dict) and "error" in res:
            raise ServerError(f"server error: {res['error']}")
        return res

    def get_action(self, observation, stop_event=None, retries=0, timeout_s=None):
        res = self.call("get_action", {"observation": observation, "options": None}, timeout_s=timeout_s, stop_event=stop_event, retries=retries)
        action = res[0] if isinstance(res, (list, tuple)) else res
        return action

    def get_action_full(self, observation, stop_event=None, retries=0, timeout_s=None):
        """Like get_action but returns (action, info): the TurboVLA server answers [{"actions": ...}, {"inference_ms", "step"}]."""
        res = self.call("get_action", {"observation": observation, "options": None}, timeout_s=timeout_s, stop_event=stop_event, retries=retries)
        if isinstance(res, (list, tuple)):
            return (res[0] if len(res) > 0 else None), (res[1] if len(res) > 1 else {})
        return res, {}

    def close(self):
        try:
            self.ctx.destroy(linger=0)  # closes every socket (also the debug monitor), never blocks like term()
        except Exception:  # noqa: BLE001
            pass


# --------------------------------------------------------------------------------------------------------------- ROS IO
class RosIO:
    """rclpy node + background executor: latest joints/images, robot_description, hardware list, trajectory action."""

    def __init__(self, cfg, need_action):
        import rclpy
        from control_msgs.action import FollowJointTrajectory
        from rclpy.action import ActionClient
        from rclpy.executors import MultiThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
        from rclpy.signals import SignalHandlerOptions
        from sensor_msgs.msg import CompressedImage, JointState
        from std_msgs.msg import String
        self.cfg, self.rclpy, self.FJT = cfg, rclpy, FollowJointTrajectory
        self.joints = list(cfg["ros"]["joints"])
        self.cams = list(cfg["cameras"])
        rclpy.init(args=None, signal_handler_options=SignalHandlerOptions.NO)  # our own SIGINT handling
        self.node = Node("groot_ros_rollout")
        self.lock = threading.Lock()
        self._q, self._q_t, self._q_stamp = None, None, None
        self.hist = collections.deque(maxlen=5000)  # (monotonic, q)
        self._img = {}
        self._urdf = None
        self.node.create_subscription(JointState, cfg["ros"]["joint_states_topic"], self._on_js, qos_profile_sensor_data)
        for k, c in cfg["cameras"].items():
            self.node.create_subscription(CompressedImage, c["topic"], lambda m, k=k: self._on_img(k, m), qos_profile_sensor_data)
        self.node.create_subscription(
            String, cfg["ros"]["robot_description_topic"], self._on_urdf,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE))
        from controller_manager_msgs.srv import ListHardwareComponents
        self._lhc_type = ListHardwareComponents
        self.lhc = self.node.create_client(ListHardwareComponents, cfg["ros"]["controller_manager"].rstrip("/") + "/list_hardware_components")
        self.ac = ActionClient(self.node, FollowJointTrajectory, cfg["ros"]["action"]) if need_action else None
        self.active = None  # goal handle of the running goal
        self.active_res_fut = None
        self.active_is_hold = False
        self.exec = MultiThreadedExecutor(num_threads=3)
        self.exec.add_node(self.node)
        self._thread = threading.Thread(target=self.exec.spin, daemon=True)
        self._thread.start()

    # --- callbacks
    def _on_js(self, msg):
        d = dict(zip(msg.name, msg.position))
        if not all(j in d for j in self.joints):
            return
        q = np.array([d[j] for j in self.joints], dtype=np.float64)
        t = time.monotonic()
        with self.lock:
            self._q, self._q_t = q, t
            self._q_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self.hist.append((t, q))

    def _on_img(self, key, msg):
        t = time.monotonic()
        try:
            buf = np.frombuffer(msg.data, dtype=np.uint8)
        except (TypeError, ValueError):
            buf = np.asarray(msg.data, dtype=np.uint8)
        with self.lock:
            prev = self._img.get(key)
            self._img[key] = {"data": buf, "t": t, "stamp": msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                              "n": (prev["n"] + 1) if prev else 1, "format": msg.format}

    def _on_urdf(self, msg):
        self._urdf = msg.data

    # --- state access
    def now_ros_s(self):
        return self.node.get_clock().now().nanoseconds * 1e-9

    def snapshot(self):
        """Consistent copy of the latest inputs with ages (s) -> dict."""
        with self.lock:
            now = time.monotonic()
            snap = {"q": None if self._q is None else self._q.copy(), "q_age": None if self._q_t is None else now - self._q_t,
                    "q_stamp": self._q_stamp, "img": {}}
            for k in self.cams:
                im = self._img.get(k)
                snap["img"][k] = None if im is None else dict(im, age=now - im["t"])
        snap["ros_now"] = self.now_ros_s()
        return snap

    def recent_speed(self, window=0.25):
        with self.lock:
            h = list(self.hist)[-400:]
        if len(h) < 2:
            return 0.0
        ts = np.array([x[0] for x in h])
        qs = np.array([x[1] for x in h])
        sel = ts >= ts[-1] - 2 * window
        return peak_speed(ts[sel], qs[sel], window)

    def history_between(self, t0, t1):
        with self.lock:
            h = [x for x in self.hist if t0 <= x[0] <= t1]
        return np.array([x[0] for x in h]), (np.array([x[1] for x in h]) if h else np.zeros((0, len(self.joints))))

    def wait(self, pred, timeout, stop_event=None, period=0.02):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if pred():
                return True
            if stop_event is not None and stop_event.wait(period):
                return False
            elif stop_event is None:
                time.sleep(period)
        return pred()

    def wait_inputs(self, timeout, stop_event):
        def ready():
            s = self.snapshot()
            return s["q"] is not None and all(s["img"][k] is not None for k in self.cams)
        return self.wait(ready, timeout, stop_event)

    def robot_description(self, timeout=5.0, stop_event=None):
        self.wait(lambda: self._urdf is not None, timeout, stop_event)
        return self._urdf

    def list_hardware(self, timeout=5.0):
        """-> list of {name, type, class_type, state} or None when the service is unavailable."""
        if not self.lhc.wait_for_service(timeout_sec=timeout):
            return None
        fut = self.lhc.call_async(self._lhc_type.Request())
        if not self.wait(fut.done, timeout):
            return None
        return [{"name": c.name, "type": c.type, "class_type": c.class_type, "state": c.state.label} for c in fut.result().component]

    # --- trajectory action
    def server_ready(self, timeout):
        return self.ac.wait_for_server(timeout_sec=timeout)

    def _goal(self, q_list, times):
        from trajectory_msgs.msg import JointTrajectoryPoint
        from rclpy.duration import Duration
        g = self.FJT.Goal()
        g.trajectory.joint_names = list(self.joints)
        for q, t in zip(q_list, times):
            p = JointTrajectoryPoint()
            p.positions = [float(v) for v in q]
            p.time_from_start = Duration(seconds=float(t)).to_msg()
            g.trajectory.points.append(p)
        return g

    def send_goal(self, q_list, times, stop_event, accept_timeout=5.0):
        """Send a trajectory and wait for acceptance.  Returns the result future."""
        fut = self.ac.send_goal_async(self._goal(q_list, times))
        if not self.wait(fut.done, accept_timeout):
            raise StopRun(EXIT_EXEC, "trajectory goal was not accepted/rejected within %.0f s" % accept_timeout)
        handle = fut.result()
        if not handle.accepted:
            raise StopRun(EXIT_EXEC, "trajectory goal rejected by the controller")
        self.active = handle
        self.active_is_hold = False
        res_fut = self.active_res_fut = handle.get_result_async()
        if stop_event.is_set():  # a stop arrived while we were sending: do not let the goal run
            self.cancel_nowait()
        return res_fut

    def cancel_nowait(self):
        h = self.active
        if h is not None and not self.active_is_hold:
            try:
                h.cancel_goal_async()
            except Exception:  # noqa: BLE001
                pass

    def cancel_active(self, timeout=2.0):
        """Cancel and wait until the goal has really ended.  -> True if nothing is running afterwards."""
        h = self.active
        if h is None:
            return True
        try:
            cf = h.cancel_goal_async()
            self.wait(cf.done, timeout)
            rf = self.active_res_fut or h.get_result_async()
            self.wait(rf.done, timeout)
            ended = rf.done()
        except Exception as e:  # noqa: BLE001
            say(f"cancel failed: {e}")
            ended = False
        if ended:
            self.active = None
        return ended

    def hold(self, hold_time_s, timeout=5.0):
        """Command the current measured position (single point)."""
        s = self.snapshot()
        if s["q"] is None:
            say("no joint state: cannot command hold")
            return False
        fut = self.ac.send_goal_async(self._goal([s["q"]], [hold_time_s]))
        if not self.wait(fut.done, timeout):
            say("hold goal not answered")
            return False
        handle = fut.result()
        if not handle.accepted:
            say("hold goal rejected")
            return False
        self.active, self.active_is_hold = handle, True
        rf = self.active_res_fut = handle.get_result_async()
        self.wait(rf.done, hold_time_s + timeout)
        if rf.done():
            self.active = None
        return rf.done()

    def shutdown(self):
        try:
            self.exec.shutdown(timeout_sec=1.0)
            self._thread.join(2.0)
            self.node.destroy_node()
            self.rclpy.shutdown()
            # rclpy only finalises the DDS participant when the context is destroyed; we leave through os._exit, so do it
            # here, otherwise the participant (and its shared-memory ports) lingers until the peers' lease times out
            self.rclpy.get_default_context().destroy()
        except Exception:  # noqa: BLE001
            pass


# ------------------------------------------------------------------------------------------------------------ run dir/log
class RunLog:
    def __init__(self, cfg, override):
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.warning = None
        if override:
            root, name = Path(override).expanduser(), None
            self.dir = root
        else:
            root = Path(cfg["logging"]["run_root"]).expanduser()
            self.dir = root / ts
            try:
                if not root.parent.is_dir():
                    raise OSError(f"{root.parent} missing")
                self.dir.mkdir(parents=True, exist_ok=False)
            except OSError as e:
                fb = Path(cfg["logging"]["fallback_root"]).expanduser() / ts
                self.warning = f"cannot use {root} ({e}); logging to {fb} instead (NOT on the external drive)"
                self.dir = fb
        self.dir.mkdir(parents=True, exist_ok=True)
        self.f = open(self.dir / "queries.jsonl", "a", buffering=1)
        self.save_images = bool(cfg["logging"]["save_images"])
        self.image_quality = int(cfg["logging"]["image_quality"])

    def write(self, rec):
        self.f.write(json.dumps(jsonable(rec)) + "\n")

    def save_image(self, idx, key, rgb):
        d = self.dir / "images"
        d.mkdir(exist_ok=True)
        cv2.imwrite(str(d / f"q{idx:05d}_{key}.jpg"), np.ascontiguousarray(rgb[:, :, ::-1]), [cv2.IMWRITE_JPEG_QUALITY, self.image_quality])

    def close(self):
        self.f.close()


# ------------------------------------------------------------------------------------------------------------- the rollout
class Rollout:
    def __init__(self, cfg, args):
        self.cfg, self.args = cfg, args
        self.backend = backend_of(cfg)
        self.mode = "execute" if args.execute else "observe"
        self.joints = list(cfg["ros"]["joints"])
        self.stop_event = threading.Event()
        self.stop_reason = None
        self.signal_count = 0
        self.force_event = threading.Event()
        self._guardian_done = False
        self.motion_enabled = False  # set only after the execute confirmation
        self._stop_info = None
        self.io = None
        self.client = None
        self.log = None
        self.limits = None
        self.hw_components = None
        self.hw_params = []
        self.stdin_leftover = b""
        self.counts = {"queries": 0, "pass": 0, "refuse": 0, "stale": 0, "server_error": 0, "input_error": 0}

    # --- stop machinery
    def request_stop(self, reason):
        if not self.stop_event.is_set():
            self.stop_reason = reason
            self.stop_event.set()

    def _on_signal(self, signum, frame):
        self.signal_count += 1
        if self.signal_count == 1:
            sys.stderr.write(f"\n[groot] signal {signum}: stopping (cancel + hold). Ctrl+C is NOT an e-stop.\n")
            self.request_stop(f"signal {signum}")
        else:
            sys.stderr.write(f"\n[groot] signal {signum} again: already stopping, cancel re-sent (the process will not exit with a goal running)\n")
            self.force_event.set()

    def _guardian(self):
        """Reacts within ms to a stop request, whatever the main thread is doing: cancels the running goal.  Every further
        signal re-sends the cancel (a hold goal is never cancelled by it)."""
        self.stop_event.wait()
        if self.io is not None:
            self.io.cancel_nowait()
        while not self._guardian_done:
            if self.force_event.wait(0.5):
                self.force_event.clear()
                if self.io is not None:
                    self.io.cancel_nowait()

    def _key_thread(self):
        fd = sys.stdin.fileno()
        while not self.stop_event.is_set():
            try:
                r, _, _ = select.select([fd], [], [], 0.2)
            except (OSError, ValueError):
                return
            if r:
                data = os.read(fd, 4096)
                if not data:  # EOF: no operator keyboard, never a stop
                    return
                self.request_stop("operator key")
                return

    def _read_line(self, timeout_s):
        """One line from stdin (raw fd, so select works), stop-aware.  None on EOF/timeout/stop."""
        try:
            fd = sys.stdin.fileno()
        except (OSError, ValueError):
            return None
        buf = b""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and not self.stop_event.is_set():
            r, _, _ = select.select([fd], [], [], 0.2)
            if r:
                chunk = os.read(fd, 4096)
                if not chunk:
                    return None
                buf += chunk
                if b"\n" in buf:
                    line, _, rest = buf.partition(b"\n")
                    self.stdin_leftover = rest
                    return line.decode(errors="replace").strip()
        return None

    # --- inputs
    def check_freshness(self, snap):
        fr = self.cfg["freshness"]
        problems = []
        if snap["q"] is None:
            problems.append("joint_states: never received")
        elif snap["q_age"] > fr["max_joint_age_s"]:
            problems.append(f"joint_states: {snap['q_age']:.3f} s old (> {fr['max_joint_age_s']})")
        for k, im in snap["img"].items():
            if im is None:
                problems.append(f"image {k}: never received")
                continue
            if im["age"] > fr["max_image_age_s"]:
                problems.append(f"image {k} ({self.cfg['cameras'][k]['topic']}): {im['age']:.3f} s old (> {fr['max_image_age_s']})")
            hmax = fr["max_image_header_age_s"]
            if hmax is not None and im["stamp"] > 0 and abs(snap["ros_now"] - im["stamp"]) > hmax:
                problems.append(f"image {k}: header stamp {snap['ros_now'] - im['stamp']:.3f} s behind the clock (> {hmax})")
        return problems

    def prep(self, data):
        """CompressedImage bytes -> (model RGB uint8, (full_w, full_h)) for the active backend."""
        if self.backend == "turbovla":
            return turbovla_rgb(data, self.cfg["model"], self.cfg["preprocess"])
        return preprocess_image(data, self.cfg["preprocess"], self.cfg["model"]["video_size"])

    def preprocess(self, snap, rec):
        videos, shapes, crcs = {}, {}, {}
        import zlib
        for k, im in snap["img"].items():
            video, (w, h) = self.prep(im["data"])
            shapes[k] = [w, h]
            crcs[k] = zlib.crc32(im["data"].tobytes())
            exp = self.cfg["cameras"][k]["expected_size"]
            if self.cfg["freshness"]["check_image_size"] and [w, h] != list(exp):
                raise ValueError(f"image {k} is {w}x{h}, training used {exp[0]}x{exp[1]}")
            videos[k] = video
        rec["image_size_wh"], rec["image_crc32"] = shapes, crcs
        rec["image_age_s"] = {k: round(im["age"], 4) for k, im in snap["img"].items()}
        return videos

    # --- one query
    def query_once(self, idx):
        c = self.cfg
        t_wall = time.time()
        rec = {"type": "query", "idx": idx, "t_wall": t_wall, "t_iso": datetime.datetime.now().isoformat(timespec="milliseconds"),
               "mode": self.mode}
        res = {"verdict": None, "chunk": None, "q_now": None, "rec": rec, "problems": []}
        snap = self.io.snapshot()
        rec["state"] = snap["q"]
        rec["joint_age_s"] = snap["q_age"]
        problems = self.check_freshness(snap)
        if problems:
            rec["verdict"], rec["problems"] = "stale", problems
            res.update(verdict="stale", problems=problems)
            return res
        try:
            videos = self.preprocess(snap, rec)
        except Exception as e:  # noqa: BLE001
            rec["verdict"], rec["problems"] = "input_error", [str(e)]
            res.update(verdict="input_error", problems=[str(e)])
            return res
        if self.log.save_images:
            for k, v in videos.items():
                self.log.save_image(idx, k, v)
        obs = build_observation(videos, snap["q"], c["instruction"], c["model"])
        t0 = time.monotonic()
        try:
            tmo = c["server"]["first_query_timeout_s"] if self.counts["queries"] == 0 else c["server"]["timeout_s"]
            action = self.client.get_action(obs, self.stop_event, retries=int(c["server"]["get_action_retries"]), timeout_s=tmo)
        except Interrupted:
            raise StopRun(EXIT_STOPPED, self.stop_reason or "stopped")
        except ServerError as e:
            rec["verdict"], rec["problems"] = "server_error", [str(e)]
            rec["latency_ms"] = (time.monotonic() - t0) * 1000
            res.update(verdict="server_error", problems=[str(e)])
            return res
        rec["latency_ms"] = round((time.monotonic() - t0) * 1000, 1)
        rec["server_retries"] = self.client.last_retries
        raw = action.get(c["model"]["action_key"]) if isinstance(action, dict) else None
        if raw is None:
            msg = f"reply has no action key {c['model']['action_key']!r} (keys: {list(action) if isinstance(action, dict) else type(action).__name__})"
            rec["verdict"], rec["problems"] = "refuse", [msg]
            rec["gates"] = [{"name": "shape", "ok": False, "detail": msg}]
            res.update(verdict="refuse", problems=[msg])
            return res
        rec["chunk_raw_shape"] = list(np.asarray(raw).shape)
        chunk, prob = normalize_chunk(raw, c["model"]["chunk_len"], len(self.joints))
        rec["chunk"] = chunk if chunk is not None else (np.asarray(raw).astype(float).tolist() if np.asarray(raw).dtype.kind in "fiu" else str(raw)[:200])
        # the state used for the gates is read AFTER the reply (the arm may have moved while waiting)
        snap2 = self.io.snapshot()
        gates = []
        if prob is not None:
            gates.append({"name": "shape", "ok": False, "detail": prob})
        else:
            gates.append({"name": "shape", "ok": True, "detail": f"{chunk.shape}"})
        fresh_msg = None
        if snap2["q"] is None or snap2["q_age"] > c["freshness"]["max_joint_age_s"]:
            fresh_msg = f"joint_states {None if snap2['q'] is None else round(snap2['q_age'], 3)} s old after the reply (> {c['freshness']['max_joint_age_s']})"
            gates.append({"name": "state_fresh_at_send", "ok": False, "detail": fresh_msg})
        q_now = snap2["q"] if snap2["q"] is not None else snap["q"]
        # urdf_limit_mode clamp: targets beyond the URDF limits shrunk by urdf_inset_rad are clamped onto them (the clamped
        # chunk is what the gates check and what is executed); a target beyond them by more than urdf_clamp_max_rad is refused.
        if chunk is not None and c["gates"].get("urdf_limit_mode", "refuse") == "clamp" and np.isfinite(chunk).all():
            ins = c["gates"]["urdf_inset_rad"]
            lo, hi = self.limits[:, 0] + ins, self.limits[:, 1] - ins
            over = np.maximum(chunk - hi, lo - chunk).clip(min=0.0)
            if over.max() > 0.0:
                lim = float(c["gates"]["urdf_clamp_max_rad"])
                rec["urdf_clamp"] = {f"j{j + 1}": round(float(over[:, j].max()), 4) for j in np.flatnonzero(over.max(0) > 0)}
                gates.append({"name": "urdf_clamp", "ok": bool(over.max() <= lim),
                              "detail": f"clamped {rec['urdf_clamp']} rad onto URDF limits -{ins} (max allowed {lim})"})
                chunk = np.clip(chunk, lo, hi)
                rec["chunk"] = chunk
        gates += evaluate_gates(chunk, q_now, self.limits, c["gates"], c["training_range"]["min"], c["training_range"]["max"],
                                c["execution"]["horizon"])
        rec["state_at_send"] = q_now
        rec["gates"] = gates
        ok = all(g["ok"] for g in gates)
        rec["verdict"] = "pass" if ok else "refuse"
        res.update(verdict=rec["verdict"], chunk=chunk, q_now=q_now,
                   problems=[f"{g['name']}: {g['detail']}" for g in gates if not g["ok"]])
        return res

    # --- TurboVLA: one query (observation -> raw chunk + raw-chunk gates)
    def query_once_turbovla(self, idx, step):
        c = self.cfg
        m = c["model"]
        t_cycle = time.monotonic()
        rec = {"type": "query", "idx": idx, "step": step, "t_wall": time.time(), "t_iso": datetime.datetime.now().isoformat(timespec="milliseconds"),
               "mode": self.mode, "backend": "turbovla"}
        res = {"verdict": None, "chunk": None, "q_now": None, "rec": rec, "problems": [], "t_cycle": t_cycle, "target": None}
        snap = self.io.snapshot()
        rec["state"], rec["joint_age_s"] = snap["q"], snap["q_age"]
        problems = self.check_freshness(snap)
        if problems:
            rec["verdict"], rec["problems"] = "stale", problems
            res.update(verdict="stale", problems=problems)
            return res
        try:
            videos = self.preprocess(snap, rec)
        except Exception as e:  # noqa: BLE001
            rec["verdict"], rec["problems"] = "input_error", [str(e)]
            res.update(verdict="input_error", problems=[str(e)])
            return res
        if self.log.save_images:
            for k, v in videos.items():
                self.log.save_image(idx, k, v)
        t0 = time.monotonic()
        try:
            tmo = c["server"]["first_query_timeout_s"] if self.counts["queries"] == 0 else c["server"]["timeout_s"]
            action, info = self.client.get_action_full(build_observation_turbovla(videos, snap["q"], step, tv_arm_channels(m), m["state_dim"]), self.stop_event,
                                                       retries=int(c["server"]["get_action_retries"]), timeout_s=tmo)
        except Interrupted:
            raise StopRun(EXIT_STOPPED, self.stop_reason or "stopped")
        except ServerError as e:
            rec["verdict"], rec["problems"] = "server_error", [str(e)]
            rec["latency_ms"] = (time.monotonic() - t0) * 1000
            res.update(verdict="server_error", problems=[str(e)])
            return res
        rec["latency_ms"] = round((time.monotonic() - t0) * 1000, 1)
        rec["server_retries"] = self.client.last_retries
        info = info if isinstance(info, dict) else {}
        rec["server_info"] = {k: info[k] for k in ("inference_ms", "step") if k in info}
        raw = action.get("actions") if isinstance(action, dict) else None
        gates = []
        if raw is None:
            msg = f"reply has no 'actions' (reply: {list(action) if isinstance(action, dict) else type(action).__name__})"
            rec["verdict"], rec["problems"] = "refuse", [msg]
            rec["gates"] = [{"name": "shape", "ok": False, "detail": msg}]
            res.update(verdict="refuse", problems=[msg])
            return res
        if "step" in info and info["step"] != step:  # a stale / mismatched answer must never enter the ensembler
            msg = f"reply is for step {info['step']!r}, requested step {step}"
            rec["verdict"], rec["problems"] = "server_error", [msg]
            res.update(verdict="server_error", problems=[msg])
            return res
        chunk, prob = tv_normalize_chunk(raw, m["chunk_len"], m["action_dim"])
        gates.append({"name": "shape", "ok": prob is None, "detail": f"{chunk.shape}" if prob is None else prob})
        snap2 = self.io.snapshot()  # the state for the gates is read AFTER the reply (the arm may have moved while waiting)
        if snap2["q"] is None or snap2["q_age"] > c["freshness"]["max_joint_age_s"]:
            gates.append({"name": "state_fresh_at_send", "ok": False,
                          "detail": f"joint_states {None if snap2['q'] is None else round(snap2['q_age'], 3)} s old after the reply (> {c['freshness']['max_joint_age_s']})"})
        q_now = snap2["q"] if snap2["q"] is not None else snap["q"]
        gates += evaluate_gates_turbovla(chunk, q_now, self.limits, c["gates"], c["training_range"]["min"], c["training_range"]["max"], tv_arm_channels(m))
        rec["state_at_send"], rec["gates"] = q_now, gates
        if chunk is not None and np.isfinite(chunk).all():
            rec["chunk_summary"] = chunk_summary(chunk, q_now, tv_arm_channels(m))
        else:
            rec["chunk_summary"] = {"shape": list(np.asarray(raw).shape), "note": prob or "non-finite values"}
        if c["logging"]["log_full_chunk"] or chunk is None or not np.isfinite(chunk).all():
            rec["chunk"] = chunk if chunk is not None else str(raw)[:200]
        ok = all(g["ok"] for g in gates)
        rec["verdict"] = "pass" if ok else "refuse"
        res.update(verdict=rec["verdict"], chunk=chunk, q_now=q_now, problems=[f"{g['name']}: {g['detail']}" for g in gates if not g["ok"]])
        return res

    def tv_add_target(self, res, ens):
        """Push a passed raw chunk into the ensembler (consecutive steps), gate the ensembled target and update the verdict."""
        c, rec = self.cfg, res["rec"]
        step = ens.next_step
        full = ens.step(res["chunk"], step=step)
        target = full[tv_arm_channels(c["model"])].copy()
        tg = evaluate_target_gates(target, res["q_now"], self.limits, c["gates"], c["training_range"]["min"], c["training_range"]["max"])
        rec["gates"] += tg
        rec["ensemble"] = {"step": step, "history_size": ens.history_size, "target": target,
                           "target_minus_state_max": float(np.abs(target - res["q_now"]).max()) if np.isfinite(target).all() else None}
        ok = all(g["ok"] for g in rec["gates"])
        res["verdict"] = rec["verdict"] = "pass" if ok else "refuse"
        res["problems"] = [f"{g['name']}: {g['detail']}" for g in rec["gates"] if not g["ok"]]
        res["target"] = target

    def execute_target(self, res, rec):
        """Command ONE ensembled 6-joint target as a single-point FollowJointTrajectory goal and wait for it (step-synchronous)."""
        c = self.cfg["execution"]
        dt = 1.0 / self.cfg["model"]["control_rate_hz"]
        q_now, target = res["q_now"], res["target"]
        times = trajectory_times(q_now, [target], dt, c["max_joint_vel"])
        T = float(times[0])
        ex = {"n_points": 1, "duration_s": T, "nominal_s": dt, "stretch": T / dt, "cmd_peak_vel": float(np.max(np.abs(target - q_now)) / T)}
        rec["exec"] = ex
        self._run_goal(q_now, np.asarray([target]), times, ex)

    def tv_add_segment(self, res, n):
        """ensemble.mode chunk: the first n raw rows (controlled arm's 6 joints) are the segment to execute; gate its displacement from the state."""
        c, rec = self.cfg, res["rec"]
        qs = res["chunk"][:n][:, tv_arm_channels(c["model"])].copy()
        disp = np.abs(qs - res["q_now"]).max()
        cap = float(c["gates"].get("max_segment_disp_rad", 0.3))
        rec["gates"].append({"name": "segment_displacement", "ok": bool(disp <= cap),
                             "detail": f"max |segment - state| over {n} steps = {disp:.4f} rad, limit {cap}"})
        rec["segment"] = {"n": n, "last": qs[-1], "last_minus_state": qs[-1] - res["q_now"]}
        ok = all(g["ok"] for g in rec["gates"])
        res["verdict"] = rec["verdict"] = "pass" if ok else "refuse"
        res["problems"] = [f"{g['name']}: {g['detail']}" for g in rec["gates"] if not g["ok"]]
        res["segment"] = qs

    def execute_segment(self, res, rec):
        """ensemble.mode chunk: send the raw segment as one trajectory (1/control_rate per step, stretched to max_joint_vel)."""
        c = self.cfg["execution"]
        dt = 1.0 / self.cfg["model"]["control_rate_hz"]
        q_now, qs = res["q_now"], res["segment"]
        times = trajectory_times(q_now, qs, dt, c["max_joint_vel"])
        nominal = len(qs) * dt
        ex = {"n_points": len(qs), "duration_s": float(times[-1]), "nominal_s": nominal, "stretch": float(times[-1] / nominal),
              "cmd_peak_vel": float(np.max(np.abs(np.diff(np.vstack([q_now, qs]), axis=0)) / np.diff(np.r_[0.0, times])[:, None]))}
        rec["exec"] = ex
        self._run_goal(q_now, qs, times, ex)

    def loop_turbovla_chunk(self):
        """ensemble.mode chunk (diagnostic): no temporal ensemble.  Query -> raw-chunk gates -> execute the first
        ensemble.chunk_exec_steps rows as one trajectory -> wait -> query again (like the GR00T backend)."""
        c = self.cfg
        e = c["execution"]
        n = int(c["ensemble"].get("chunk_exec_steps", 10))
        if not 1 <= n <= int(c["model"]["chunk_len"]):
            raise StopRun(EXIT_USAGE, f"ensemble.chunk_exec_steps {n} not in 1..{c['model']['chunk_len']}")
        t_start = time.monotonic()
        steps_done, idx = 0, 0
        while True:
            if self.stop_event.is_set():
                raise StopRun(EXIT_STOPPED, self.stop_reason or "stopped")
            if time.monotonic() - t_start > e["max_time_s"]:
                return EXIT_OK, f"max_time_s {e['max_time_s']} reached"
            if self.mode == "execute" and steps_done >= e["max_steps"]:
                return EXIT_OK, f"max_steps {e['max_steps']} reached"
            if self.mode == "observe" and c["observe"]["max_queries"] and idx >= c["observe"]["max_queries"]:
                return EXIT_OK, f"max_queries {c['observe']['max_queries']} reached"
            idx += 1
            res = self.query_once_turbovla(idx, idx - 1)  # server is stateless; step only has to be echoed back
            rec = res["rec"]
            if res["verdict"] == "pass":
                self.tv_add_segment(res, n)
            self.counts["queries"] += 1
            self.counts[res["verdict"]] = self.counts.get(res["verdict"], 0) + 1
            if res["verdict"] == "pass" and self.mode == "execute":
                rec["sent"] = True
                try:
                    self.execute_segment(res, rec)
                finally:
                    rec["cycle_ms"] = round((time.monotonic() - res["t_cycle"]) * 1000, 1)
                    self.log.write(rec)
                steps_done += n
                self.print_query_turbovla(res, steps_done)
                continue
            rec["sent"] = False
            rec["cycle_ms"] = round((time.monotonic() - res["t_cycle"]) * 1000, 1)
            self.log.write(rec)
            self.print_query_turbovla(res, steps_done)
            if res["verdict"] != "pass" and self.mode == "execute":
                code = {"refuse": EXIT_REFUSED}.get(res["verdict"], EXIT_INPUTS)
                raise StopRun(code, f"{res['verdict']}: " + " | ".join(res["problems"]))
            if self.stop_event.wait(c["observe"]["period_s"]):
                raise StopRun(EXIT_STOPPED, self.stop_reason or "stopped")

    def loop_turbovla(self):
        c = self.cfg
        e = c["execution"]
        if c["ensemble"].get("mode", "act") == "chunk":
            return self.loop_turbovla_chunk()
        ens = ACTTemporalEnsembler(c["ensemble"]["horizon"], c["model"]["action_dim"], c["ensemble"]["decay"])
        t_start = time.monotonic()
        steps_done, idx = 0, 0
        while True:
            if self.stop_event.is_set():
                raise StopRun(EXIT_STOPPED, self.stop_reason or "stopped")
            if time.monotonic() - t_start > e["max_time_s"]:
                return EXIT_OK, f"max_time_s {e['max_time_s']} reached"
            if self.mode == "execute" and steps_done >= e["max_steps"]:
                return EXIT_OK, f"max_steps {e['max_steps']} reached"
            if self.mode == "observe" and c["observe"]["max_queries"] and idx >= c["observe"]["max_queries"]:
                return EXIT_OK, f"max_queries {c['observe']['max_queries']} reached"
            idx += 1
            res = self.query_once_turbovla(idx, ens.next_step)
            rec = res["rec"]
            if res["verdict"] == "pass":
                self.tv_add_target(res, ens)
            self.counts["queries"] += 1
            self.counts[res["verdict"]] = self.counts.get(res["verdict"], 0) + 1
            if res["verdict"] == "pass" and self.mode == "execute":
                rec["sent"] = True
                try:
                    self.execute_target(res, rec)
                finally:
                    rec["cycle_ms"] = round((time.monotonic() - res["t_cycle"]) * 1000, 1)
                    self.log.write(rec)
                steps_done += 1
                self.print_query_turbovla(res, steps_done)
                continue
            rec["sent"] = False
            if res["target"] is not None:
                rec["would_command"] = res["target"]  # observe: what execute mode would have sent
            rec["cycle_ms"] = round((time.monotonic() - res["t_cycle"]) * 1000, 1)
            self.log.write(rec)
            self.print_query_turbovla(res, steps_done)
            if res["verdict"] != "pass" and self.mode == "execute":
                code = {"refuse": EXIT_REFUSED}.get(res["verdict"], EXIT_INPUTS)
                raise StopRun(code, f"{res['verdict']}: " + " | ".join(res["problems"]))
            if self.stop_event.wait(c["observe"]["period_s"]):
                raise StopRun(EXIT_STOPPED, self.stop_reason or "stopped")

    def print_query_turbovla(self, res, steps_done):
        rec, v = res["rec"], res["verdict"]
        every = int(self.cfg["logging"]["print_every"])
        if v == "pass" and rec["idx"] != 1 and every > 0 and rec["idx"] % every:
            return
        head = f"#{rec['idx']:04d} step {rec['step']:4d} {v.upper():12s}"
        if rec.get("latency_ms") is not None:
            head += f" latency {rec['latency_ms']:5.0f} ms"
        si = rec.get("server_info") or {}
        if "inference_ms" in si:
            head += f" (server {si['inference_ms']:.0f} ms)"
        ens = rec.get("ensemble")
        if ens and ens.get("target_minus_state_max") is not None:
            head += f"  |target-state| {ens['target_minus_state_max']:.4f} rad  hist {ens['history_size']}"
        seg = rec.get("segment")
        if seg is not None:
            head += f"  segment {seg['n']} steps, |last-state| {float(np.abs(seg['last_minus_state']).max()):.4f} rad"
        if rec.get("cycle_ms") is not None:
            head += f"  cycle {rec['cycle_ms']:.0f} ms"
        say(head)
        for p in res["problems"]:
            say(f"      refused/failed: {p}")

    # --- execute one validated chunk
    def execute_chunk(self, res, rec):
        c = self.cfg["execution"]
        H = c["horizon"]
        q_now, qs = res["q_now"], res["chunk"][:H]
        dt = 1.0 / self.cfg["model"]["fps"]
        times = trajectory_times(q_now, qs, dt, c["max_joint_vel"])
        nominal = H * dt
        ex = {"n_points": H, "duration_s": float(times[-1]), "nominal_s": nominal, "stretch": float(times[-1] / nominal),
              "cmd_peak_vel": float(np.max(np.abs(np.diff(np.vstack([q_now, qs]), axis=0)) / np.diff(np.r_[0.0, times])[:, None]))}
        rec["exec"] = ex
        self._run_goal(q_now, qs, times, ex)

    def _run_goal(self, q_now, qs, times, ex):
        """Send one trajectory (points qs at cumulative times), watch it (tracking error, limits, speed, staleness), check the result,
        let the arm settle on the last point.  Shared by both backends.  Fills `ex`."""
        c = self.cfg["execution"]
        grid_t, grid_q = np.r_[0.0, times], np.vstack([q_now, qs])
        t_send = time.monotonic()
        fut = self.io.send_goal(qs, times, self.stop_event)
        t_acc = time.monotonic()
        deadline = t_acc + times[-1] + c["result_margin_s"]
        max_track = 0.0
        while not fut.done():
            if self.stop_event.is_set():
                raise StopRun(EXIT_STOPPED, self.stop_reason or "stopped")
            if time.monotonic() > deadline:
                raise StopRun(EXIT_EXEC, f"goal did not finish within {times[-1] + c['result_margin_s']:.1f} s")
            snap = self.io.snapshot()
            if snap["q"] is None or snap["q_age"] > c["inflight_joint_stale_s"]:
                raise StopRun(EXIT_INPUTS, f"joint_states stale during execution ({snap['q_age']})")
            el = time.monotonic() - t_acc
            want = np.array([np.interp(el, grid_t, grid_q[:, j]) for j in range(len(self.joints))])
            err = float(np.max(np.abs(snap["q"] - want)))
            max_track = max(max_track, err)
            if err > c["tracking_error_rad"]:
                raise StopRun(EXIT_EXEC, f"tracking error {err:.3f} rad > {c['tracking_error_rad']}")
            if (snap["q"] < self.limits[:, 0]).any() or (snap["q"] > self.limits[:, 1]).any():
                raise StopRun(EXIT_EXEC, f"measured joints outside URDF limits: {np.round(snap['q'], 3).tolist()}")
            sp = self.io.recent_speed(0.25)
            if sp > c["max_measured_vel_factor"] * c["max_joint_vel"]:
                raise StopRun(EXIT_EXEC, f"measured joint speed {sp:.3f} rad/s > {c['max_measured_vel_factor']} x {c['max_joint_vel']}")
            time.sleep(0.01)
        result = fut.result().result
        self.io.active = None
        t_done = time.monotonic()
        ex["error_code"], ex["error_string"] = int(result.error_code), result.error_string
        ex["wall_s"], ex["max_tracking_err"] = t_done - t_acc, max_track
        ts, Q = self.io.history_between(t_send, t_done)
        ex["measured_peak_vel"] = peak_speed(ts, Q, 0.2) if len(ts) > 2 else None
        if result.error_code != 0:
            raise StopRun(EXIT_EXEC, f"trajectory action failed: error_code={result.error_code} {result.error_string}")
        # let the arm settle on the last point (logged, not a gate)
        last = qs[-1]
        self.io.wait(lambda: (lambda s: s["q"] is not None and np.max(np.abs(s["q"] - last)) <= c["settle_tol_rad"])(self.io.snapshot()),
                     c["settle_timeout_s"], self.stop_event)
        after = self.io.snapshot()
        ex["state_after"] = after["q"]
        ex["residual_to_last_point"] = None if after["q"] is None else float(np.max(np.abs(after["q"] - last)))

    # --- hardware report / confirmation
    def hardware_report(self, require):
        comps = self.io.list_hardware()
        self.hw_components = comps
        urdf = self.io.robot_description(5.0, self.stop_event)
        if urdf:
            _, self.hw_params = parse_robot_description(urdf, self.joints)
        lines = []
        if comps is None:
            lines.append(f"  controller_manager {self.cfg['ros']['controller_manager']}: list_hardware_components NOT available")
        for h in comps or []:
            lines.append(f"  component {h['name']}: type={h['type']} plugin={h['class_type']} state={h['state']}")
        # dual-arm URDFs carry one ros2_control block per arm: show the one(s) that own the controlled joints (all when unknown)
        mine = [h for h in self.hw_params if set(h.get("joints") or ()) & set(self.joints)]
        for h in (mine or self.hw_params):
            ip = h["params"].get("robot_ip")
            lines.append(f"  /robot_description ros2_control '{h['name']}': plugin={h['plugin']}" + (f" robot_ip={ip}" if ip else " (robot_ip not in the URDF)"))
        return "\n".join(lines), comps

    def confirm_execute(self):
        text, comps = self.hardware_report(True)
        mock = hardware_is_mock(comps)
        print("=" * 78)
        say("EXECUTE MODE: this client will MOVE THE ARM")
        print(text)
        if comps is None:
            say("cannot read the hardware type -> refusing to execute (fail closed)")
            return False
        e = self.cfg["execution"]
        say(f"hardware is {'MOCK (mock_components/GenericSystem only)' if mock else '*** NOT MOCK: REAL HARDWARE (or unknown) ***'}")
        if self.backend == "turbovla":
            g = self.cfg["gates"]
            if self.cfg["ensemble"].get("mode", "act") == "chunk":
                say(f"TurboVLA CHUNK mode (no ensemble, diagnostic): first {self.cfg['ensemble'].get('chunk_exec_steps', 10)} raw steps of each "
                    f"chunk per query, segment limit {g.get('max_segment_disp_rad', 0.3)} rad, action {self.cfg['ros']['action']}")
            else:
                say(f"TurboVLA step loop (stop-and-go): one ensembled target per cycle (ACT horizon {self.cfg['ensemble']['horizon']}, decay {self.cfg['ensemble']['decay']}), "
                    f"per-step limit {g['max_target_step_rad']} rad, action {self.cfg['ros']['action']}")
            say(f"controlled joints {self.joints} = model channels {self.cfg['model']['arm_channels']}")
            say(f"max joint speed {e['max_joint_vel']} rad/s, max_steps {e['max_steps']}, max_time {e['max_time_s']} s")
        else:
            say(f"action {self.cfg['ros']['action']}, horizon {e['horizon']} steps, max joint speed {e['max_joint_vel']} rad/s, "
                f"max_steps {e['max_steps']}, max_time {e['max_time_s']} s")
        say(f"instruction: {self.cfg['instruction']!r}   server {self.cfg['server']['host']}:{self.cfg['server']['port']}")
        say("stop: Ctrl+C or Enter.  This is NOT an e-stop; keep the e-stop in reach.")
        print("=" * 78)
        if mock and self.args.mock_yes:
            say("--mock-yes: all hardware components are mock_components/GenericSystem -> confirmation skipped")
            return True
        if self.args.mock_yes and not mock:
            say("--mock-yes ignored: hardware is not (provably) mock, the typed confirmation is required")
        print("[groot] type  yes  and Enter to allow motion: ", end="", flush=True)
        ans = self._read_line(self.cfg["execution"]["confirm_timeout_s"])
        print(flush=True)
        if ans != "yes":
            say(f"confirmation not given ({ans!r}) -> staying out of execute mode")
            return False
        return True

    # --- main
    def run(self):
        c, a = self.cfg, self.args
        self.log = RunLog(c, a.run_dir)
        if self.log.warning:
            say("WARNING: " + self.log.warning)
        say(f"mode {self.mode}; run dir {self.log.dir}")
        signal.signal(signal.SIGINT, self._on_signal)
        signal.signal(signal.SIGTERM, self._on_signal)
        signal.signal(signal.SIGHUP, self._on_signal)
        threading.Thread(target=self._guardian, daemon=True).start()
        code, reason, abnormal = EXIT_OK, "completed", False
        try:
            self.io = RosIO(c, need_action=(self.mode == "execute"))
            say("waiting for joint states and both camera topics ...")
            if not self.io.wait_inputs(c["ros"]["wait_for_inputs_s"], self.stop_event):
                s = self.io.snapshot()
                raise StopRun(EXIT_INPUTS, "no inputs within %.0f s: joints=%s %s" % (
                    c["ros"]["wait_for_inputs_s"], s["q"] is not None, {k: v is not None for k, v in s["img"].items()}))
            urdf = self.io.robot_description(5.0, self.stop_event)
            if not urdf:
                raise StopRun(EXIT_INPUTS, f"no {c['ros']['robot_description_topic']}: cannot read the URDF joint limits")
            self.limits, _ = parse_robot_description(urdf, self.joints)
            say("URDF joint limits: " + " ".join(f"j{i + 1}[{lo:.2f},{hi:.2f}]" for i, (lo, hi) in enumerate(self.limits)))
            self.log.write({"type": "meta", "t_iso": datetime.datetime.now().isoformat(), "mode": self.mode, "argv": sys.argv,
                            "config": {k: v for k, v in c.items()}, "urdf_limits": self.limits})
            if a.check:
                code, reason = self.check(), "check"
                return code
            if self.mode == "execute":
                if not self.confirm_execute():
                    raise StopRun(EXIT_USAGE, "execute not confirmed")
                if self.stdin_leftover.strip():
                    raise StopRun(EXIT_STOPPED, "extra input after the confirmation = stop request")
                self.motion_enabled = True
                if not self.io.server_ready(5.0):
                    raise StopRun(EXIT_INPUTS, f"action server {c['ros']['action']} not available")
            else:
                text, comps = self.hardware_report(False)
                say("hardware (informational):\n" + text)
            threading.Thread(target=self._key_thread, daemon=True).start()
            self.client = Gr00tClient(c["server"]["host"], c["server"]["port"], c["server"]["timeout_s"], c["server"]["api_token"],
                                      label="TurboVLA" if self.backend == "turbovla" else "GR00T")
            attempts = int(c["server"]["ping_attempts"])
            for attempt in range(1, attempts + 1):
                try:
                    pong = self.client.call("ping", timeout_s=c["server"]["ping_timeout_s"], stop_event=self.stop_event)
                    say(f"server {self.client.addr} answers ping" + (f" (attempt {attempt})" if attempt > 1 else ""))
                    if self.backend == "turbovla" and not (isinstance(pong, dict) and pong.get("status") == "ok"):
                        say(f"WARNING: ping reply is not {{'status': 'ok'}}: {str(pong)[:120]}")
                    break
                except Interrupted:
                    raise StopRun(EXIT_STOPPED, self.stop_reason or "stopped")
                except ServerError as e:
                    say(f"ping attempt {attempt}/{attempts} failed: {e}")
                    if attempt == attempts:
                        raise StopRun(EXIT_INPUTS, f"policy server: {e}")
            self.verify_modality()
            try:
                # TurboVLA: `reset` without data (the server is stateless; the ensembler lives here and is reset by construction)
                self.client.call("reset", None if self.backend == "turbovla" else {"options": None},
                                 timeout_s=c["server"]["ping_timeout_s"], stop_event=self.stop_event)
            except ServerError as e:
                say(f"reset endpoint failed ({e}); continuing")
            code, reason = self.loop_turbovla() if self.backend == "turbovla" else self.loop()
        except StopRun as e:
            code, reason, abnormal = e.code, e.reason, True
        except Interrupted:
            code, reason, abnormal = EXIT_STOPPED, self.stop_reason or "stopped", True
        except BaseException as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            code, reason, abnormal = EXIT_ERROR, f"exception {type(e).__name__}: {e}", True
        finally:
            if self.stop_event.is_set() and code == EXIT_OK:
                code, reason, abnormal = EXIT_STOPPED, self.stop_reason or "stopped", True
            self.finish(code, reason, abnormal)
        return code

    def finish(self, code, reason, abnormal):
        if self.motion_enabled and self.io is not None and self.io.ac is not None and (abnormal or self.io.active is not None):
            say(f"stopping ({reason}): cancel active goal, hold current position")
            ended = self.io.cancel_active()
            held = self.io.hold(self.cfg["execution"]["hold_time_s"])
            say(f"cancel confirmed={ended}, hold goal finished={held}")
            self._stop_info = {"cancelled": ended, "held": held}
        else:
            self._stop_info = None
        try:
            self.log.write({"type": "end", "code": code, "reason": reason, "abnormal": abnormal, "counts": self.counts,
                            "stop": self._stop_info, "t_iso": datetime.datetime.now().isoformat()})
        except Exception:  # noqa: BLE001
            pass
        say(f"END code={code} reason={reason!r} counts={self.counts}  log: {self.log.dir}/queries.jsonl")
        self._guardian_done = True
        self.force_event.set()
        self.stop_event.set()
        if self.client:
            self.client.close()
        if self.io:
            self.io.shutdown()
        self.log.close()

    def verify_modality(self):
        """Cross-check the checkpoint's video/state/action/language keys and action horizon with this client (e.g. wrist_left vs
        wrist_right).  Mismatch: execute refuses, observe warns.  An unreadable reply only warns."""
        c = self.cfg
        try:
            reply = self.client.call("get_modality_config", timeout_s=c["server"]["ping_timeout_s"], stop_event=self.stop_event)
        except Interrupted:
            raise StopRun(EXIT_STOPPED, self.stop_reason or "stopped")
        except ServerError as e:
            say(f"WARNING: get_modality_config failed ({e}): cannot cross-check the model keys")
            return
        if self.backend == "turbovla":
            return self.verify_modality_turbovla(reply)
        if isinstance(reply, dict) and reply.get("backend") not in (None, "groot"):  # e.g. a TurboVLA server on the GR00T config
            probs = [f"backend: server announces {reply.get('backend')!r}, this config needs the GR00T server"]
            self.log.write({"type": "server_modality", "info": {"backend": reply.get("backend")}, "problems": probs})
            for p in probs:
                say("MODALITY MISMATCH: " + p)
            if self.mode == "execute":
                raise StopRun(EXIT_INPUTS, "server modality mismatch: " + " | ".join(probs))
            return
        info = parse_modality_reply(reply)
        if info is None:
            say("WARNING: get_modality_config reply not understood: cannot cross-check the model keys")
            self.log.write({"type": "server_modality", "info": None})
            return
        probs = compare_modality(info, c)
        self.log.write({"type": "server_modality", "info": info, "problems": probs})
        say("server modality: " + "; ".join(f"{k} {v['keys']}" + (f"x{v['n']}" if v["n"] is not None else "") for k, v in info.items()))
        for p in probs:
            say("MODALITY MISMATCH: " + p)
        if probs and self.mode == "execute":
            raise StopRun(EXIT_INPUTS, "server modality mismatch: " + " | ".join(probs))

    def verify_modality_turbovla(self, reply):
        """TurboVLA get_modality_config cross-check (backend, cameras, 224 size, 14-D state/action, horizon 30, 30 Hz, language,
        optional checkpoint sha).  Mismatch: execute refuses, observe warns.  An empty / non-dict reply only warns."""
        probs, info = compare_modality_turbovla(reply, self.cfg)
        if info is None:
            say("WARNING: get_modality_config reply not understood: cannot cross-check the model")
            self.log.write({"type": "server_modality", "info": None})
            return
        self.log.write({"type": "server_modality", "info": info, "problems": probs})
        say("server modality: backend " + str(info.get("backend")) + " " + " ".join(
            f"{k}={info[k]}" for k in ("cameras", "client_cameras", "image_size", "state_dim", "action_dim", "horizon", "control_rate_hz", "arm_channels") if k in info)
            + (f" language={info['language']!r}" if "language" in info else "") + (f" checkpoint_sha256={str(info['checkpoint_sha256'])[:16]}" if "checkpoint_sha256" in info else ""))
        for p in probs:
            say("MODALITY MISMATCH: " + p)
        if probs and self.mode == "execute":
            raise StopRun(EXIT_INPUTS, "server modality mismatch: " + " | ".join(probs))

    def check(self):
        s = self.io.snapshot()
        say(f"preprocess.mode {self.cfg['preprocess']['mode']} -> model input "
            + (f"{self.cfg['model']['recorder_size']} (recorder) then {self.cfg['model']['image_size']}" if self.backend == "turbovla" else f"{self.cfg['model']['video_size']}") + " (W, H)")
        say(f"joints {self.joints}: {None if s['q'] is None else np.round(s['q'], 4).tolist()} age {s['q_age']:.3f} s")
        if self.backend == "turbovla":
            say(f"model.arm_channels {self.cfg['model']['arm_channels']} (model state/action channels of the controlled arm; others sent as 0), "
                f"action {self.cfg['ros']['action']}")
        for k, im in s["img"].items():
            try:
                shp = list(self.prep(im["data"])[1])
            except Exception as e:  # noqa: BLE001
                shp = f"undecodable ({e})"
            say(f"camera {k}: {self.cfg['cameras'][k]['topic']} age {im['age']:.3f} s size {shp} expected {self.cfg['cameras'][k]['expected_size']}")
        probs = self.check_freshness(s)
        text, comps = self.hardware_report(False)
        print(text)
        say(f"hardware_is_mock={hardware_is_mock(comps)}")
        if self.io.ac is None:
            from rclpy.action import ActionClient
            ac = ActionClient(self.io.node, self.io.FJT, self.cfg["ros"]["action"])
            say(f"action server {self.cfg['ros']['action']}: {'available' if ac.wait_for_server(timeout_sec=3.0) else 'NOT available'}")
        say("freshness: " + ("OK" if not probs else "; ".join(probs)))
        return EXIT_OK if not probs else EXIT_INPUTS

    def loop(self):
        c = self.cfg
        e = c["execution"]
        t_start = time.monotonic()
        steps_done, idx = 0, 0
        while True:
            if self.stop_event.is_set():
                raise StopRun(EXIT_STOPPED, self.stop_reason or "stopped")
            if time.monotonic() - t_start > e["max_time_s"]:
                return EXIT_OK, f"max_time_s {e['max_time_s']} reached"
            if self.mode == "execute" and steps_done >= e["max_steps"]:
                return EXIT_OK, f"max_steps {e['max_steps']} reached"
            if self.mode == "observe" and c["observe"]["max_queries"] and idx >= c["observe"]["max_queries"]:
                return EXIT_OK, f"max_queries {c['observe']['max_queries']} reached"
            idx += 1
            res = self.query_once(idx)
            rec = res["rec"]
            self.counts["queries"] += 1
            self.counts[res["verdict"]] = self.counts.get(res["verdict"], 0) + 1
            self.print_query(res)
            if res["verdict"] == "pass" and self.mode == "execute":
                rec["sent"] = True
                try:
                    self.execute_chunk(res, rec)
                finally:
                    self.log.write(rec)
                steps_done += e["horizon"]
                continue
            rec["sent"] = False
            self.log.write(rec)
            if res["verdict"] != "pass" and self.mode == "execute":
                code = {"refuse": EXIT_REFUSED}.get(res["verdict"], EXIT_INPUTS)
                raise StopRun(code, f"{res['verdict']}: " + " | ".join(res["problems"]))
            if self.stop_event.wait(c["observe"]["period_s"]):
                raise StopRun(EXIT_STOPPED, self.stop_reason or "stopped")

    def print_query(self, res):
        rec = res["rec"]
        v = res["verdict"]
        head = f"#{rec['idx']:04d} {v.upper():12s}"
        if rec.get("latency_ms") is not None:
            head += f" latency {rec['latency_ms']:6.0f} ms"
        if res["chunk"] is not None and res["q_now"] is not None:
            ch = res["chunk"]
            head += (f"  step0-state {np.max(np.abs(ch[0] - res['q_now'])):.4f}  step{len(ch)}-state "
                     f"{np.max(np.abs(ch[-1] - res['q_now'])):.4f} rad")
        say(head)
        if rec.get("urdf_clamp"):
            say(f"      clamped onto URDF limits (rad beyond): {rec['urdf_clamp']}")
        for p in res["problems"]:
            say(f"      refused/failed: {p}")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config value (dotted key, YAML value)")
    ap.add_argument("--execute", action="store_true", help="send the chunks to the arm (needs the typed `yes`)")
    ap.add_argument("--mock-yes", action="store_true",
                    help="skip the typed confirmation ONLY if every hardware component is mock_components/GenericSystem (automated mock tests)")
    ap.add_argument("--check", action="store_true", help="print input ages/sizes, hardware type, limits and exit (no server, no motion)")
    ap.add_argument("--host"), ap.add_argument("--port", type=int)
    ap.add_argument("--instruction")
    ap.add_argument("--max-queries", type=int, help="observe: number of queries")
    ap.add_argument("--max-steps", type=int, help="execute: executed model steps")
    ap.add_argument("--max-time", type=float, help="seconds")
    ap.add_argument("--horizon", type=int)
    ap.add_argument("--max-joint-vel", type=float)
    ap.add_argument("--run-dir", help="log directory (default ~/groot/runs/<timestamp>)")
    ap.add_argument("--save-images", action="store_true", help="save the model input images of every query")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        cfg = load_config(args.config, args.set)
        if backend_of(cfg) == "turbovla" and args.horizon is not None:
            raise ValueError("--horizon is a GR00T option (the TurboVLA backend always executes one ensembled target per cycle)")
        for flag, dotted in [("host", "server.host"), ("port", "server.port"), ("instruction", "instruction"),
                             ("max_queries", "observe.max_queries"), ("max_steps", "execution.max_steps"),
                             ("max_time", "execution.max_time_s"), ("horizon", "execution.horizon"),
                             ("max_joint_vel", "execution.max_joint_vel")]:
            if getattr(args, flag) is not None:
                deep_set(cfg, dotted, getattr(args, flag))
        if args.save_images:
            cfg["logging"]["save_images"] = True
        backend = backend_of(cfg)
        if backend == "turbovla":
            validate_turbovla_config(cfg, args)
        elif backend == "groot":
            e = cfg["execution"]
            if not 1 <= e["horizon"] <= cfg["model"]["chunk_len"]:
                raise ValueError(f"execution.horizon must be in 1..{cfg['model']['chunk_len']}")
            if not 0 < e["max_joint_vel"] <= 0.5:
                raise ValueError("execution.max_joint_vel must be in (0, 0.5] rad/s")
            if len(cfg["training_range"]["min"]) != 6 or len(cfg["ros"]["joints"]) != 6:
                raise ValueError("6 joints / training_range entries expected")
        else:
            raise ValueError(f"policy.backend {backend!r} unknown (groot | turbovla)")
    except (KeyError, ValueError, OSError, yaml.YAMLError) as ex:
        print(f"[groot] config error: {ex}", file=sys.stderr)
        return EXIT_USAGE
    return Rollout(cfg, args).run()


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)  # no interpreter finalization with live rclpy/zmq threads (was an intermittent SIGABRT at exit)
