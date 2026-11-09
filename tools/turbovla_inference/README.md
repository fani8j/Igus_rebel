# TurboVLA policy server (Compal) - setup, profiles (left mocap10k / right right19), server protocol, offline check

Offline only: no robot, no ROS. Everything bulky lives on the external drive, reached through the symlink
`~/turbovla -> "/media/farhan/My Passport/igus_nvidia/turbovla"` (NTFS via ntfs-3g).

Model: TurboVLA, 3 views (`head`, `knife`, `wrist_left`, 224x224 RGB), 14-D state, 30x14 absolute-radian action chunk,
30 Hz. The `knife` view is fed fixed black at inference; the client only supplies `head` and `wrist_left`. State/action layout:
left joint1..6, left gripper, right joint1..6, right gripper. **Each checkpoint knows exactly one arm** (all other channels are 0 in its
training data): the bundle's `mocap10k` the LEFT arm (channels 0..5, language `Open the box flaps.`), the run `right19-10k` the RIGHT arm
(channels 7..12, language `open the front flap of the cardboard box`, trained with the right gripper camera in the `wrist_left` slot).
ACT temporal ensembling (horizon 30, k=0.01) is done by the CLIENT; the server is stateless.

| profile (`--profile`) | checkpoint | arm / `arm_channels` | language | pin |
|---|---|---|---|---|
| `bundle` (default) | `compal-turbovla-handoff-v1/models/mocap10k/.../steps_10000_ema_pytorch_model.pt` | left `[0, 6]` | `Open the box flaps.` | bundle `MANIFEST.json` `checkpoint_sha256` |
| `right19` | `right19/runs/right19-10k/checkpoints/steps_4000_ema_pytorch_model.pt` | right `[7, 13]` | `open the front flap of the cardboard box` | `02f163bef316fcc1894ec3b614a8d9cb7640a7d1771c48bc8f23607fa5598068` |
| `mocap31` | `mocap31/runs/mocap31-10k/checkpoints/steps_6000_ema_pytorch_model.pt` | right `[7, 13]` | `open the front flap of the cardboard box` | `82eab1f745064fbabbdea12f5a8d14f54a5e5195e5f03ac61e713b0e574b3ec6` |
| `real3nov` | `real3nov/real3nov/runs/real3nov-10k/checkpoints/steps_10000_ema_pytorch_model.pt` | right `[7, 13]` | `open the next closed flap of the cardboard box` | `53d1f024930295be142cba3aad14550b6f7e1568fead7653b42b24a8b86a83d1` |

Profiles are `profiles/<name>.profile` next to the start script (`KEY=value`: `CHECKPOINT`, `LANGUAGE`, `EXPECT_SHA256`; parsed, never sourced).
Rules enforced by the server (`turbovla_server.py`): the bundle's own checkpoint is pinned by the MANIFEST; **any other checkpoint needs
`--language` and `--expect-sha256`** (file hashed before any weight is loaded; mismatch or missing pin = refuses to start). The active arm is derived from the run's
`normalization.json` + `embodiment.json` (the arm whose six joint channels have an observed action span; `retained_channels` lists all 14 kept channels and says nothing about
the arm; gripper channels carry a fixed 0..1 placeholder range and are skipped) and the server refuses to start unless exactly one arm is active. A `get_action` whose state is non-zero
outside the active arm's channels is rejected.

## Layout (`~/turbovla`)

| path | content |
|---|---|
| `compal-turbovla-handoff-v1/` | the colleague's handoff bundle (untrusted third-party code, verified against its MANIFEST), `.venv` = policy venv (python 3.12, torch 2.13.0+cu130, ~4.5 GB) |
| `server_deps/` | extra packages for the server/client, on `PYTHONPATH` only (the bundle venv and lock are untouched): pyzmq, msgpack, msgpack-numpy, pyarrow 21 (needs numpy 1.x), av |
| `uv_cache/`, `tmp/` | `UV_CACHE_DIR`, `TMPDIR` |
| `logs/` | `server.log`, `setup.log`, `isaac-smoke.log`, `offline_check/` |

## Install (what worked)

```bash
cd ~/turbovla/compal-turbovla-handoff-v1
export TMPDIR=~/turbovla/tmp UV_CACHE_DIR=~/turbovla/uv_cache UV_LINK_MODE=copy
python3 bundle.py verify
nice -n 19 python3 bundle.py setup --python ~/.local/bin/python3.12      # uv; ~6 min
python3 bundle.py check                                                  # runs/policy-check.json passed:true
cd ~/turbovla && nice -n 19 ~/.local/bin/uv pip install --python compal-turbovla-handoff-v1/.venv/bin/python \
    --target server_deps --no-deps pyzmq msgpack msgpack-numpy av "pyarrow==21.0.0"
# server_deps is put on PYTHONPATH by start_turbovla_server.sh; do NOT install numpy/Pillow there (the venv has numpy 1.26.4, pyarrow 26 needs numpy 2)
```

## Server

```bash
tools/turbovla_inference/start_turbovla_server.sh start                      # default profile bundle (LEFT arm); transient user unit turbovla-server, blocks until ping answers
tools/turbovla_inference/start_turbovla_server.sh start --profile right19    # RIGHT arm checkpoint right19-10k
tools/turbovla_inference/start_turbovla_server.sh start --profile mocap31    # RIGHT arm checkpoint mocap31-10k (step 6000 EMA, 31 Oct set)
tools/turbovla_inference/start_turbovla_server.sh start --profile real3nov   # RIGHT arm checkpoint real3nov-10k (R1, step 10000 EMA, 3 Nov playback set)
tools/turbovla_inference/start_turbovla_server.sh start --checkpoint F --language "..." --expect-sha256 HEX   # ad-hoc checkpoint (all three required)
tools/turbovla_inference/start_turbovla_server.sh status | ping | info | logs | stop | restart   # info = get_modality_config (arm_channels, language, sha)
tools/turbovla_inference/start_turbovla_server.sh start --port 5557     # or TURBOVLA_PORT / TURBOVLA_HOST / TURBOVLA_BUNDLE / TURBOVLA_UNIT
```

One server at a time (same unit `turbovla-server`, same port 5556; `start` while it runs does nothing: `stop` first to switch profile). GPU memory about 2 GB per server.

Stop it with `systemctl --user stop turbovla-server` (what `stop` does); never kill by pattern. Log: `~/turbovla/logs/server.log`.
The GR00T server uses port 5555; this one uses 5556 and can run alongside only if GPU memory allows (check `nvidia-smi`).

### Protocol (fixed)

ZeroMQ **ROUTER** on `tcp://127.0.0.1:5556` (strays such as HTTP/TLS probes are ignored; a plain `zmq.REQ` client works).
Wire format identical to Isaac-GR00T `gr00t/policy/server_client.py` `MsgSerializer`: a request is a msgpack map
`{"endpoint": str, "data": {...}}` (`data` omitted for `ping` / `get_modality_config`), numpy arrays are msgpack-numpy encoded
(`allow_pickle=False`). A bad request gets `{"error": "<message>"}`, never a bare `b"ERROR"`.

| endpoint | request `data` | reply |
|---|---|---|
| `ping` | - | `{"status": "ok", "message": "..."}` |
| `get_modality_config` | - | `{"backend": "turbovla", "cameras": ["head","knife","wrist_left"], "client_cameras": ["head","wrist_left"], "image_size": [224,224], "state_dim": 14, "action_dim": 14, "horizon": 30, "control_rate_hz": 30, "checkpoint_sha256": "...", "language": "Open the box flaps.", "arm": "left", "arm_channels": [0, 6]}` (right19: `"arm": "right", "arm_channels": [7, 13]`; `arm_channels` is `[first, end)`) |
| `reset` | anything (e.g. `{"options": None}`) | `{"status": "ok"}` (stateless) |
| `get_action` | `{"observation": {"state": float (14,), "head": uint8 (224,224,3), "wrist_left": uint8 (224,224,3), "step": int}, "options": None}` | `[{"actions": float32 (30,14)}, {"inference_ms": float, "step": int}]` |

`actions` are absolute joint targets in radians for the model's 14 channels (left joints 0-5, left gripper 6, right 7-12, right 13); only the announced `arm_channels` are meaningful.
Validation (via the bundle's `live_observation`): `state` finite float, shape (14,), exactly 0 outside `arm_channels`; images uint8 RGB (224,224,3); missing
keys / wrong dtype / NaN -> `{"error": ...}`. Extra observation keys (e.g. `knife`) are ignored.

Minimal client:

```python
import zmq, numpy as np
from turbovla_server import MsgSerializer          # tools/turbovla_inference/turbovla_server.py (needs only msgpack, msgpack_numpy, numpy at import)
s = zmq.Context().socket(zmq.REQ); s.connect("tcp://127.0.0.1:5556")
s.send(MsgSerializer.to_bytes({"endpoint": "get_action", "data": {"observation": {
    "state": np.zeros(14), "head": head_u8, "wrist_left": wrist_u8, "step": 0}, "options": None}}))
(actions, info) = MsgSerializer.from_bytes(s.recv()); actions["actions"].shape   # (30, 14)
```

## Offline check

```bash
cd ~/turbovla/compal-turbovla-handoff-v1
# LEFT (server profile bundle): the 12 raw mocap episodes
PYTHONPATH=~/turbovla/server_deps:source nice -n 19 .venv/bin/python ~/igus_rebel_ros2/tools/turbovla_inference/offline_check_turbovla.py \
    --out ~/turbovla/logs/offline_check            # server must be running
# RIGHT (server profile right19): the 2 held-out episodes real_0005, real_0014 of the right19 validation dataset, through the real server
PYTHONPATH=~/turbovla/server_deps:source nice -n 19 .venv/bin/python ~/igus_rebel_ros2/tools/turbovla_inference/offline_check_turbovla.py \
    --arm right --black-frames -1                  # --arm auto (default) takes the arm from the server's arm_channels
```

See the docstring of `offline_check_turbovla.py` for the metric definitions (training target = recorded `action[t+k]` for chunk step k). The right-arm run uses the
data of the training evaluation (`$T/right19/data/training-mp4/Compal/rebel_right_frontflap_validation_v1`, 1152x648 h264 videos decoded and resized to 224 like the training loader).

### Results (2026-10-29, RTX 5070 Ti, whole raw episodes, 12 x ~1500-2660 frames; report in `~/turbovla/logs/offline_check/report.json`)

Server latency (model only, warm): median 21 ms, p95 31 ms; client round trip about 32 ms. GPU memory of the server about 2.1 GB.
`bundle.py check`: passed, max_reference_error_rad 0.0. Server vs `smoke-reference.npz`: 5.6e-8 rad (float32 cast).

| left 6 joints, MAE rad | raw chunk vs action[t+k] | ACT ensemble vs action[t] | hold-still raw / ensemble |
|---|---|---|---|
| 10 train, real wrist | 0.075 | 0.074 | 0.057 / 0.022 |
| 2 held-out, real wrist | 0.320 | 0.312 | 0.058 / 0.023 |
| 10 train, wrist black | 0.106 | 0.105 | 0.057 / 0.022 |
| 2 held-out, wrist black | 0.356 | 0.350 | 0.058 / 0.023 |

Held-out error is dominated by joints 4 and 6 (about 0.5 rad): all 10 train episodes except `mocap_left_13` run joint4 negative
(0 -> -3) with joint6 positive, both held-out episodes run joint4 positive with joint6 negative (the mirrored wrist branch).

### Results right19-10k (2026-10-30, whole held-out episodes real_0005 + real_0014 = 2855 frames, through the real server, `--arm right`)

The server path reproduces the training evaluation (`$T/right19/runs/right19-10k/evaluations/steps_4000.json`, `eval/right19-10k-heldout/steps_4000_summary.json`):

| right 6 joints, MAE rad | this check (server) | training eval (step 4000 EMA) |
|---|---|---|
| raw chunk vs action[t+k], valid targets only (`raw_mae`) | 0.037742 (real_0005 0.0347, real_0014 0.0411) | 0.037721 (0.03461, 0.04112) |
| raw chunk, repeat-last padded targets (`raw_mae_padded`) | 0.037496 | 0.037476 (`right6_mae_rad`) |
| hold-still (state[t]) raw | 0.022607 | 0.022607 |
| ACT ensemble (k=0.01) vs action[t] / hold-still | 0.03701 / 0.00145 | - |

The 2e-5 rad difference is bf16 autocast plus the mp4 decode path (here PyAV in the server env, training: its own loader). Server latency (model only): median 19.8 ms, p95 22.1 ms (n=2845).
The live `first_step` gate looks at `max_j |chunk[0] - state|`: median 0.0526 rad (3.0 deg), p95 0.175 rad, 42 % (real_0005) / 49 % (real_0014) of the frames are at or under 0.05 rad.
