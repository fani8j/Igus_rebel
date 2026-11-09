#!/usr/bin/env python3
"""
stub_policy_server.py -- stand-in for Isaac-GR00T's run_gr00t_server.py, for testing groot_ros_rollout.py WITHOUT a model.

Same wire protocol as gr00t/policy/server_client.py: ZeroMQ REP (implemented with a ROUTER socket, see main()), msgpack + msgpack-numpy, request
{"endpoint": ..., "data": {...}}, reply = result, or {"error": "..."} on any exception.  Endpoints: ping, reset, kill,
get_modality_config (announces --video-keys / --action-len, same encoding as the real server), get_action (data = {"observation", "options"} -> reply [action_dict, info]).

get_action validates the observation like Gr00tPolicy.check_observation PLUS the exact shapes the rollout must send
(video head/wrist_left uint8 (1,1,480,640,3); state single_arm float32 (1,1,6); language [[str]]) and answers with a
(1,16,6) float32 chunk under "single_arm".  Modes (--mode):

  good      absolute joints moving from the received state toward --target at --speed rad/s per joint (15 Hz steps)
  fast      same as good but at --fast-speed rad/s (default 0.3): passes the step gates; the client must stretch timing
  static    16 x the current state (no motion)
  jump      good, but from step 6 on joint1 is offset by +0.4 rad (one big step)
  nan       good with a NaN in step 4
  limit     joint --limit-joint (default 5 = joint5) ramps +0.03 rad/step from the state: leaves URDF / training limits
  relative  DELTAS (chunk - state) instead of absolute joints
  badshape  (1,16,5)
  slow      good, but sleeps --slow-s seconds first (longer than the client timeout)

  ~/venvs/gr00t_client/bin/python stub_policy_server.py --mode good --port 5555

--backend turbovla speaks the TurboVLA protocol instead (tools/turbovla_inference server, same ZeroMQ ROUTER + msgpack-numpy framing):
  ping -> {"status": "ok"};  reset -> {"status": "ok"};  get_modality_config -> {"backend": "turbovla", "cameras": [head, knife, wrist_left],
  "client_cameras": [head, wrist_left], "image_size": [224, 224], "state_dim": 14, "action_dim": 14, "horizon": 30, "control_rate_hz": 30,
  "checkpoint_sha256": ..., "language": "Open the box flaps.", "arm_channels": [0, 6]};
  get_action data {"observation": {"state": float (14,), "head": uint8 (224,224,3), "wrist_left": uint8 (224,224,3), "step": int}, "options": None}
  -> [{"actions": float32 (30, 14)}, {"inference_ms": float, "step": int}].  The observation is validated EXACTLY (key set, dtypes, shapes,
  finite state, every channel outside the announced arm_channels == 0, consecutive step numbers); a violation comes back as {"error": ...}.
  Modes: good (smooth ABSOLUTE chunk from the received state toward --target at --speed rad/s, 30 rows at 30 Hz), fast (--fast-speed),
  static (30 x state), jump (joint1 +0.4 from row 5), nan, limit (--limit-joint ramps 0.03 rad/row out of range), relative (deltas),
  badshape ((30,13)), slow (sleeps --slow-s).  --error-at N: the N-th get_action answers {"error"}.  Channels 6..13 of the chunk are
  filled with 0.7 (the client must ignore them).
  Emulating the RIGHT-arm checkpoint (right19-10k): --tv-arm-channels 7,13 --tv-language "open the front flap of the cardboard box"
  --tv-sha <64 hex> and a --target inside the right joint ranges (the arm joints are state[7:13] / chunk[:, 7:13]).  --tv-arm-channels none
  = do not announce arm_channels at all (a pre-right-arm server; the stub then controls the left arm [0, 6]).
  ~/venvs/gr00t_client/bin/python stub_policy_server.py --backend turbovla --mode good --port 5556
"""
import argparse
import io
import json
import time
import traceback

import msgpack
import msgpack_numpy as mnp
import numpy as np
import zmq

VIDEO_KEYS = ("head", "wrist_left")
STATE_KEY = ACTION_KEY = "single_arm"
LANGUAGE_KEY = "annotation.human.task_description"
VIDEO_SHAPE = (1, 1, 480, 640, 3)
CHUNK = 16
FPS = 15.0
# TurboVLA flavour
TV_CHUNK, TV_DIM, TV_FPS, TV_IMG = 30, 14, 30.0, (224, 224, 3)
TV_LANGUAGE = "Open the box flaps."


def decode_hook(obj):
    if isinstance(obj, dict):
        marker = obj.get("__ndarray_class__", obj.get(b"__ndarray_class__"))
        if marker:
            return np.load(io.BytesIO(obj.get("as_npy", obj.get(b"as_npy"))), allow_pickle=False)
        if obj.get(b"nd", obj.get("nd")) and obj.get(b"kind", obj.get("kind")) in (b"O", "O"):
            raise ValueError("refusing object-dtype array")
    return mnp.decode(obj)


def to_bytes(data):
    return msgpack.packb(data, default=mnp.encode)


def from_bytes(data):
    return msgpack.unpackb(data, object_hook=decode_hook, raw=False)


def check_observation(obs, expect_instruction):
    """Mirror of Gr00tPolicy.check_observation with the shapes this checkpoint needs."""
    for m in ("video", "state", "language"):
        assert m in obs and isinstance(obs[m], dict), f"observation must contain dict {m!r}"
    for k in VIDEO_KEYS:
        assert k in obs["video"], f"video key {k!r} missing (have {list(obs['video'])})"
        v = obs["video"][k]
        assert isinstance(v, np.ndarray), f"video {k}: not a numpy array ({type(v)})"
        assert v.dtype == np.uint8, f"video {k}: dtype {v.dtype}, need uint8"
        assert v.shape == VIDEO_SHAPE, f"video {k}: shape {v.shape}, need {VIDEO_SHAPE}"
    extra = set(obs["video"]) - set(VIDEO_KEYS)
    assert not extra, f"unexpected video keys {sorted(extra)}"
    assert STATE_KEY in obs["state"], f"state key {STATE_KEY!r} missing"
    s = obs["state"][STATE_KEY]
    assert isinstance(s, np.ndarray) and s.dtype == np.float32, f"state dtype {getattr(s, 'dtype', type(s))}, need float32"
    assert s.shape == (1, 1, 6), f"state shape {s.shape}, need (1,1,6)"
    assert np.isfinite(s).all(), "state not finite"
    lang = obs["language"].get(LANGUAGE_KEY)
    assert isinstance(lang, list) and len(lang) == 1 and isinstance(lang[0], list) and len(lang[0]) == 1 \
        and isinstance(lang[0][0], str), f"language must be [[str]], got {lang!r}"
    if expect_instruction is not None:
        assert lang[0][0] == expect_instruction, f"instruction {lang[0][0]!r} != expected {expect_instruction!r}"


def toward(state, target, speed):
    """16 absolute steps from `state` toward `target`, each joint at most speed/FPS rad per step (clamped at the target)."""
    step = speed / FPS
    k = np.arange(1, CHUNK + 1)[:, None]
    delta = np.clip(target - state, -step * k, step * k)
    return state + delta


def tv_check_observation(obs, last_step, arm=(0, 6)):
    """Exact TurboVLA observation contract.  Returns the step.  `last_step` = previous accepted step (-1 after reset): a repeat (re-send,
    refused chunk) or +1 is fine, anything else is a protocol error."""
    assert isinstance(obs, dict), f"observation must be a dict, got {type(obs).__name__}"
    assert set(obs) == {"state", "head", "wrist_left", "step"}, f"observation keys {sorted(obs)}, need ['head', 'state', 'step', 'wrist_left']"
    s = obs["state"]
    assert isinstance(s, np.ndarray) and s.dtype.kind == "f", f"state must be a float ndarray, got {getattr(s, 'dtype', type(s))}"
    assert s.shape == (TV_DIM,), f"state shape {s.shape}, need ({TV_DIM},)"
    assert np.isfinite(s).all(), "state not finite"
    rest = np.delete(s, np.arange(*arm))
    assert (rest == 0).all(), f"state channels outside the arm channels {list(arm)} (other arm, grippers) must be exactly 0 (training contract), got {s.tolist()}"
    for k in ("head", "wrist_left"):
        v = obs[k]
        assert isinstance(v, np.ndarray), f"{k}: not a numpy array ({type(v)})"
        assert v.dtype == np.uint8, f"{k}: dtype {v.dtype}, need uint8"
        assert v.shape == TV_IMG, f"{k}: shape {v.shape}, need {TV_IMG}"
    st = obs["step"]
    assert isinstance(st, (int, np.integer)) and not isinstance(st, bool), f"step must be an int, got {type(st).__name__}"
    assert st >= 0 and st in (last_step, last_step + 1), f"step {st} after step {last_step}: must repeat or advance by 1 (reset starts at 0)"
    return int(st)


def tv_toward(state, target, speed, arm=(0, 6)):
    """30 absolute rows from state[arm] toward target, each joint at most speed/30 rad per row (row k = k+1 steps ahead); other channels 0.7."""
    step = speed / TV_FPS
    k = np.arange(1, TV_CHUNK + 1)[:, None]
    out = np.full((TV_CHUNK, TV_DIM), 0.7)
    cur = state[arm[0]:arm[1]]
    out[:, arm[0]:arm[1]] = cur + np.clip(target - cur, -step * k, step * k)
    return out


class Stub:
    def __init__(self, args):
        self.args = args
        self.target = np.array([float(v) for v in args.target.split(",")])
        self.n = 0
        self.tv_last_step = -1
        # --tv-arm-channels "7,13" = right arm; "none" = not announced, the stub then controls the left arm (pre-right-arm server)
        self.tv_arm = (0, 6) if args.tv_arm_channels == "none" else tuple(int(v) for v in args.tv_arm_channels.split(","))
        assert len(self.tv_arm) == 2 and self.tv_arm[1] - self.tv_arm[0] == 6, "--tv-arm-channels must be first,first+6 or none"
        self.logf = open(args.log, "a", buffering=1) if args.log else None

    def get_action_turbovla(self, observation, options=None):
        a = self.args
        t0 = time.monotonic()
        assert options is None, f"options must be None, got {options!r}"
        arm = self.tv_arm
        step = tv_check_observation(observation, self.tv_last_step, arm)
        self.tv_last_step = step
        state = observation["state"].astype(np.float64)
        a0, a1 = arm
        self.n += 1
        mode = a.mode
        if a.error_at and self.n == a.error_at:
            raise RuntimeError(f"stub error injected at get_action #{self.n}")
        if mode == "slow" and (a.slow_first == 0 or self.n <= a.slow_first):
            time.sleep(a.slow_s)
        if mode == "fast":
            chunk = tv_toward(state, self.target, a.fast_speed, arm)
        elif mode == "static":
            chunk = tv_toward(state, state[a0:a1], 0.0, arm)
        else:
            chunk = tv_toward(state, self.target, a.speed, arm)
        if a.first_step_offset:  # constant offset of the whole chunk from the state on the arm's joint1 (first_step refusal tests)
            chunk[:, a0] += a.first_step_offset
        if mode == "jump":
            chunk[5:, a0] += 0.4
        elif mode == "nan":
            chunk[3, a0 + 2] = np.nan
        elif mode == "limit":
            j = a0 + a.limit_joint - 1
            chunk = tv_toward(state, state[a0:a1], 0.0, arm)
            chunk[:, j] = state[j] + 0.03 * np.arange(1, TV_CHUNK + 1)
        elif mode == "relative":
            chunk[:, a0:a1] = chunk[:, a0:a1] - state[a0:a1]
        elif mode == "badshape":
            chunk = chunk[:, :13]
        out = chunk.astype(np.float32)
        elapsed_ms = (time.monotonic() - t0) * 1000
        if self.logf:
            rec = {"n": self.n, "t": time.time(), "mode": mode, "step": step, "state": state.tolist(), "chunk0": out[0].tolist(),
                   "chunk_last": out[-1].tolist(), "img_mean": {k: float(observation[k].mean()) for k in ("head", "wrist_left")}, "handler_s": elapsed_ms / 1000}
            self.logf.write(json.dumps(rec) + "\n")
        if a.dump_obs:
            for k in ("head", "wrist_left"):
                np.save(f"{a.dump_obs}/{k}.npy", observation[k])
            np.save(f"{a.dump_obs}/state.npy", state)
        return {"actions": out}, {"inference_ms": float(elapsed_ms), "step": step, "stub": True, "mode": mode, "n": self.n}

    def reset_turbovla(self):
        self.tv_last_step = -1

    def get_action(self, observation, options=None):
        a = self.args
        t0 = time.monotonic()
        check_observation(observation, a.expect_instruction)
        state = observation["state"][STATE_KEY][0, 0].astype(np.float64)
        self.n += 1
        mode = a.mode
        if mode == "slow" and (a.slow_first == 0 or self.n <= a.slow_first):
            time.sleep(a.slow_s)
        if mode == "fast":
            chunk = toward(state, self.target, a.fast_speed)
        elif mode == "static":
            chunk = np.tile(state, (CHUNK, 1))
        else:
            chunk = toward(state, self.target, a.speed)
        if mode == "jump":
            chunk[5:, 0] += 0.4
        elif mode == "nan":
            chunk[3, 2] = np.nan
        elif mode == "limit":
            j = a.limit_joint - 1
            chunk = np.tile(state, (CHUNK, 1))
            chunk[:, j] = state[j] + 0.03 * np.arange(1, CHUNK + 1)
        elif mode == "relative":
            chunk = chunk - state
        elif mode == "badshape":
            chunk = chunk[:, :5]
        out = chunk.astype(np.float32)[None]
        if self.logf:
            rec = {"n": self.n, "t": time.time(), "mode": mode, "state": state.tolist(), "chunk0": out[0, 0].tolist(),
                   "chunk_last": out[0, -1].tolist(), "instruction": observation["language"][LANGUAGE_KEY][0][0],
                   "img_mean": {k: float(v.mean()) for k, v in observation["video"].items()}, "handler_s": time.monotonic() - t0}
            self.logf.write(json.dumps(rec) + "\n")
        if a.dump_obs:
            for k, v in observation["video"].items():
                np.save(f"{a.dump_obs}/{k}.npy", v[0, 0])
            np.save(f"{a.dump_obs}/state.npy", state)
        return {ACTION_KEY: out}, {"stub": True, "mode": mode, "n": self.n}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="groot", choices=["groot", "turbovla"], help="which server protocol to speak")
    ap.add_argument("--error-at", type=int, default=0, help="turbovla: the N-th get_action raises (answers {'error'})")
    ap.add_argument("--tv-horizon", type=int, default=TV_CHUNK, help="turbovla: horizon announced by get_modality_config (mismatch tests)")
    ap.add_argument("--tv-cameras", default="head,knife,wrist_left", help="turbovla: server cameras announced by get_modality_config (mismatch tests)")
    ap.add_argument("--tv-language", default=TV_LANGUAGE)
    ap.add_argument("--tv-arm-channels", default="0,6", help="turbovla: controlled arm channels announced as arm_channels and used for the state check / chunk: "
                    "'0,6' (left, default), '7,13' (right) or 'none' (not announced, left behaviour)")
    ap.add_argument("--tv-sha", default="stub" + "0" * 60, help="turbovla: checkpoint_sha256 announced by get_modality_config")
    ap.add_argument("--first-step-offset", type=float, default=0.0, help="turbovla: add this many rad to the controlled arm's joint1 in every chunk row (first_step gate tests)")
    ap.add_argument("--mode", default="good", choices=["good", "fast", "static", "jump", "nan", "limit", "relative", "badshape", "slow"])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--target", default="0.45,0.85,1.35,0.15,1.4,0.15", help="good/fast: absolute joint1..6 target (rad)")
    ap.add_argument("--speed", type=float, default=0.08, help="good: rad/s per joint")
    ap.add_argument("--fast-speed", type=float, default=0.3, help="fast: rad/s per joint")
    ap.add_argument("--limit-joint", type=int, default=5, help="limit mode: 1-based joint that ramps out of range")
    ap.add_argument("--slow-s", type=float, default=20.0)
    ap.add_argument("--slow-first", type=int, default=0, help="slow mode: only the first N get_action calls are slow (0 = all)")
    ap.add_argument("--video-keys", default="head,wrist_left", help="what get_modality_config claims the checkpoint uses (to test the client cross-check)")
    ap.add_argument("--action-len", type=int, default=16, help="action horizon announced by get_modality_config")
    ap.add_argument("--expect-instruction", default=None, help="reject observations whose language differs")
    ap.add_argument("--log", default=None, help="append one JSON line per get_action")
    ap.add_argument("--dump-obs", default=None, help="directory: save the last received images/state as .npy")
    args = ap.parse_args()

    stub = Stub(args)
    ctx = zmq.Context()
    # A ROUTER socket speaks the same wire protocol to a REQ client as Isaac-GR00T's REP socket (envelope = identity + empty
    # delimiter + body), but a stray connection cannot desynchronise it.  Reason: on this machine some tool probes every
    # localhost listener every ~16 s with an HTTP GET and a TLS ClientHello; libzmq reads the first byte of such a
    # connection as a ZMTP 1.0 frame length, so a REP socket takes the HTTP text for a request, answers it and loses/garbles a
    # real exchange (seen as "unpack(b) received extra data" and replies that never arrive).
    sock = ctx.socket(zmq.ROUTER)
    sock.bind(f"tcp://{args.host}:{args.port}")
    print(f"stub policy server (backend={args.backend}, mode={args.mode}) listening on tcp://{args.host}:{args.port}", flush=True)
    running = True
    while running:
        frames = sock.recv_multipart()
        if len(frames) < 3 or frames[-2] != b"":  # not a REQ message (port probe etc.): ignore
            print(f"ignored a non-REQ message ({len(frames)} frames, {sum(map(len, frames))} bytes)", flush=True)
            continue
        envelope = frames[:-1]
        try:
            req = from_bytes(frames[-1])
            ep = req.get("endpoint", "get_action")
            tv = args.backend == "turbovla"
            if ep == "ping":
                res = {"status": "ok", "message": "stub server"}
            elif ep == "reset":
                stub.reset_turbovla() if tv else None
                res = {"status": "ok"} if tv else {}
            elif ep == "kill":
                running, res = False, {}
            elif ep == "get_modality_config" and tv:
                res = {"backend": "turbovla", "cameras": args.tv_cameras.split(","), "client_cameras": ["head", "wrist_left"], "image_size": [224, 224],
                       "state_dim": TV_DIM, "action_dim": TV_DIM, "horizon": args.tv_horizon, "control_rate_hz": 30,
                       "checkpoint_sha256": args.tv_sha, "language": args.tv_language}
                if args.tv_arm_channels != "none":
                    res["arm_channels"] = list(stub.tv_arm)
            elif ep == "get_modality_config":
                mc = lambda keys, n: {"__ModalityConfig__": True, "as_json": json.dumps({"delta_indices": list(range(n)), "modality_keys": keys})}
                res = {"video": mc(args.video_keys.split(","), 1), "state": mc([STATE_KEY], 1), "action": mc([ACTION_KEY], args.action_len),
                       "language": mc([LANGUAGE_KEY], 1)}
            elif ep == "get_action":
                res = list((stub.get_action_turbovla if tv else stub.get_action)(**req.get("data", {})))
            else:
                raise ValueError(f"Unknown endpoint: {ep}")
            sock.send_multipart(envelope + [to_bytes(res)])
        except Exception as e:  # noqa: BLE001
            print(f"error in stub: {e}", flush=True)
            traceback.print_exc()
            sock.send_multipart(envelope + [to_bytes({"error": str(e)})])
    sock.close(linger=1000)
    ctx.term()


if __name__ == "__main__":
    main()
