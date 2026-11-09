# GR00T N1.7 / TurboVLA closed-loop client for the real igus ReBeL arm (GR00T: LEFT arm; TurboVLA: LEFT or RIGHT arm by config)

`groot_ros_rollout.py` is a ROS 2 Humble node + CLI with two policy backends (`policy.backend` in the config): **groot** (default,
`rollout_config.yaml`, described first) and **turbovla** (`rollout_config_turbovla.yaml` left arm, `rollout_config_turbovla_right.yaml` right arm; see [TurboVLA backend](#turbovla-backend) at the end).
Both share the freshness gates, the hardware confirmation, stop handling, logging and the mock test.

The GR00T description: Every cycle it takes the head camera, the LEFT arm's gripper camera and the 6 joints of the left arm,
preprocesses the images exactly like the mocap recorder did for the training data, asks the GR00T policy server for a 16-step chunk of absolute joint targets, runs
safety gates on the chunk and (execute mode only) sends the first 8 steps to the arm through the FollowJointTrajectory action.

| File | What |
|---|---|
| `groot_ros_rollout.py` | the client (observe / execute / `--check`) |
| `rollout_config.yaml` | GR00T: topics, joints, controller, instruction, server, gates (every key required; override with `--set a.b=v`) |
| `rollout_config_turbovla.yaml` | TurboVLA LEFT arm: same layout, port 5556, 224 preprocessing, ACT ensemble, 30-row gates (`--config rollout_config_turbovla.yaml`) |
| `rollout_config_turbovla_right.yaml` | TurboVLA RIGHT arm (checkpoint right19-10k): right joints/controller/camera, 1152x648 recorder size, `arm_channels [7, 13]`, same gates and speed limits as the left config |
| `rollout_config_turbovla_mocap31.yaml` | TurboVLA RIGHT arm, checkpoint mocap31-10k (step 6000 EMA, 31 Oct mocap set): copy of the right config, differs only in the pinned sha256, `recorder_size [640, 480]` and `training_range` |
| `rollout_config_turbovla_real3nov.yaml` | TurboVLA RIGHT arm, checkpoint real3nov-10k (R1, step 10000 EMA, 3 Nov robot-playback set, front/back/right flap): copy of the right config, differs only in the pinned sha256, the language/instruction `open the next closed flap of the cardboard box` and `training_range` (recorder 1152x648 like right19) |
| `stub_policy_server.py` | fake server (same ZeroMQ + msgpack-numpy protocol), `--backend groot` (default) or `--backend turbovla` (`--tv-arm-channels 7,13 --tv-language ... --tv-sha ...` emulates the right model); modes good / fast / static / jump / nan / limit / relative / badshape / slow |
| `run_mock_test.sh` | brings up the MOCK stack + fake cameras on an isolated domain and runs the test scenarios: `--suite left` (default, single arm: GR00T and TurboVLA left) or `--suite right [--model right19\|mocap31\|real3nov]` (dual-table mock, TurboVLA right arm, optionally against the real server of that model) |

The real server and checkpoint are set up separately (`start_server.sh`, `README_server.md`).

## Safety summary (read before the first real run)

- **Default is `observe`: it never commands motion** (no goal, no hold). Use it first against the real server to look at chunks and gate verdicts.
- `execute` needs `--execute` **and** a typed `yes`. Before the prompt it prints the hardware components from
  `/controller_manager/list_hardware_components` (plugin class, e.g. `Igus::Rebel`) and the `robot_ip` from `/robot_description` if present.
  If the hardware type cannot be read, execute is refused. `--mock-yes` skips the prompt **only** if every component is
  `mock_components/GenericSystem`; on any other hardware it is ignored and the prompt stays. There is no other bypass.
- Gates (all evaluated, logged; any failure refuses the whole chunk, stops the run and holds the arm): finite values; shape (16,6) /
  (1,16,6); first step within 0.05 rad of the current joints (also catches relative-vs-absolute mix-ups); per-step jump <= 0.1 rad;
  all steps inside URDF limits minus 0.05 rad AND inside the training range +-0.1 rad; displacement from the current state over the executed
  horizon <= 0.3 rad; the current state itself inside the training range +-0.1 (the model is out of distribution otherwise). Inputs: any image
  older than 0.5 s or joint_states older than 0.2 s is refused (the message names which); decoded image size must equal `cameras.*.expected_size` (the live camera size).
- At start (after ping) the client calls the server's `get_modality_config` and compares the checkpoint's video keys (`head`, `wrist_left` vs `wrist_right`),
  state/action/language keys and the action horizon (16) with its own: a mismatch refuses `--execute` and is a loud warning in observe. If the server's reply is
  empty or not understood it only warns.
- Timing: nominal 1/15 s per step, **stretched** so no joint exceeds `max_joint_vel` (0.1 rad/s = training playback speed); never compressed.
- While a goal runs the client watches: joint_states staleness, tracking error vs the commanded interpolation (0.2 rad), measured joints
  outside URDF limits, measured speed > 2x the cap. Any violation cancels the goal and holds.
- **Stop:** Ctrl+C (SIGINT/SIGTERM/SIGHUP), Enter (or q+Enter) in the terminal, or any exception -> cancel the active goal, then command a hold at the
  current measured position. A second/third Ctrl+C never aborts the stop sequence; it only re-sends the cancel.
  **Ctrl+C / goal cancel is NOT a substitute for the e-stop.** Keep the e-stop within reach. A `kill -9` leaves whatever goal is running.
- ZeroMQ and a port prober (found while testing): some tool on this machine (the probe carries `b3:` trace headers, so an Electron app) connects to **every**
  localhost listening port about every 16 s and sends an HTTP `GET /` and a TLS ClientHello. libzmq reads the first byte of such a connection as the length of a
  legacy ZMTP 1.0 frame, so a plain **REP** server (Isaac-GR00T's `PolicyServer`, on 5555) can take the HTTP text for a request, answer it, and lose or garble a real
  exchange ("unpack(b) received extra data", or a reply that never arrives). It hit roughly 1 in 10 short runs against the first stub version. The stub therefore uses a ROUTER
  socket (same wire protocol, ignores strays). The client cannot fix the real server, so it pings up to 3 times and re-sends a timed-out `get_action` once on a fresh
  socket (`server.get_action_retries`; the arm is idle while waiting; `server_retries` is logged per query). A lost reply costs `server.timeout_s` (6 s) of waiting,
  the first query of a run gets `first_query_timeout_s` (60 s, model warm-up). If the real server misbehaves this way (an SSH tunnel port is probed too), stop the prober
  or make the server's socket a ROUTER as in `stub_policy_server.py`.
- Server timeout (`timeout_s`, twice), server error, a stale input or a refused chunk in execute mode all end the run with a hold. Exit codes:
  0 completed, 1 exception, 2 usage / not confirmed, 3 chunk refused by a gate, 4 inputs or server unavailable/stale, 5 execution failure, 130 stopped by operator/signal.

## Install (done on this machine)

System python3 (3.10) already has rclpy, numpy, opencv, pyyaml, pillow (PIL 9.0.1, needed by `preprocess.mode: recorder_pil`). Missing were pyzmq, msgpack, msgpack-numpy:

```bash
python3 -m venv --system-site-packages ~/venvs/gr00t_client
~/venvs/gr00t_client/bin/pip install pyzmq msgpack msgpack-numpy      # installed: pyzmq 27.2.0, msgpack 1.2.4, msgpack-numpy 0.4.8
```

Always run the client with `~/venvs/gr00t_client/bin/python` after sourcing ROS (the venv sees the system rclpy through `PYTHONPATH`).

## Run

```bash
export ROS_DOMAIN_ID=11 ROS_LOCALHOST_ONLY=1        # the domain of the real stack (single-arm driver of the LEFT arm: rebel.launch.py hardware_protocol:=rebel robot_ip:=192.168.3.11, cameras)
source /opt/ros/humble/setup.bash && source ~/igus_rebel_ros2/install/setup.bash
cd ~/igus_rebel_ros2/tools/groot_inference
PY=~/venvs/gr00t_client/bin/python

$PY groot_ros_rollout.py --check                                  # joints, camera ages/sizes, hardware type, URDF limits, action server. No server needed.
$PY groot_ros_rollout.py --max-queries 5 --save-images            # observe: needs the policy server (default 127.0.0.1:5555)
$PY groot_ros_rollout.py --execute --max-steps 64                 # execute (typed `yes`), first run: short
```

Useful options: `--host/--port`, `--instruction`, `--horizon N` (<=16), `--max-joint-vel`, `--max-steps` (executed model steps; training episodes are 1112-2660 frames at 30 fps = 556-1330 model steps),
`--max-time`, `--run-dir`, `--save-images`, `--set gates.first_step_max_rad=0.03`. The server may be remote through an SSH tunnel.

Suggested order on the real cell: `--check` -> observe with the real server (compare chunk[0] with the state; look at `queries.jsonl`) ->
`--execute --max-steps 32 --horizon 4` at the training start pose with someone at the e-stop.

## Model input (matches the training pipeline)

The checkpoint `rebel_open_frontlid_real_12ep/checkpoint-2000` was trained on 12 mocap-teleop episodes of the LEFT arm
(`MOCAP_DATASET_FLAP_TOP_OPEN/mocap_left_*`, LeRobot v3.0, 30 fps, task text `open the box flaps` (9 episodes) or `open flap 1` (3), AV1 videos 640x480 for head / knife / wrist_left; 14-D state/action of which the
left arm joints 0..5 are used; every 2nd frame = 15 Hz, action = next subsampled state, 16 steps). The recorder (`~/robot_remote/recording/record_dataset.py`,
`image_msg_to_rgb` + `resize_rgb`) turned each ROS CompressedImage into the video frame like this, and `preprocess.mode: recorder_pil` (default) reproduces it exactly:
`PIL.Image.open` -> `img.draft("RGB", (640, 480))` (JPEG DCT downscale, only to 1/2 for the 1080p head camera, a no-op for 720p) -> `convert("RGB")` -> if not 640x480:
PIL `resize((640, 480), BILINEAR)`. No JPEG round trip. The aspect ratio is NOT preserved (16:9 is squashed to 4:3). Result: RGB uint8 `(1,1,480,640,3)`.
Not reproduced: the AV1 compression artifacts of the training videos. `preprocess.mode: opencv_legacy` keeps the old OpenCV recipe (0.5 INTER_AREA, JPEG q90, INTER_AREA) for experiments.

- video keys `head` = `/camera/color/image_raw/compressed` (1920x1080 live), `wrist_left` = `/arm_left/gripper_camera/image_raw/compressed` (1280x720 live; the LEFT arm's own gripper camera, user-confirmed "cam 2 = gripper camera").
  The knife camera is not an input. Side-by-side of a training frame and the old lid_open frames through this preprocessing: `/tmp/groot_cam_compare.png` (see README_server.md).
- state `single_arm` float32 `(1,1,6)` rad, joints picked **by name** (`joint1..6`) from `/joint_states` (the single-arm driver `rebel.launch.py ... robot_ip:=192.168.3.11`; same igus_rebel driver package and joint conventions as the dual-arm driver used for the recordings).
  Mocap episodes (11 of 12) start at `[0, 0, 1.047, 0, 0.873, 0]`; the training range in `rollout_config.yaml` equals the state min/max of the 12 episodes (= checkpoint statistics).
- language `annotation.human.task_description` = `[["open the box flaps"]]` (the text of 9 of the 12 training episodes; the other 3 say "open flap 1"; the processor lowercases and strips punctuation, so only the words matter). The offline check against the training data compares it with a free-form sentence (README_server.md).
- The reply `single_arm` is read as `(16,6)` / `(1,16,6)` absolute joint targets.

## Logs

`~/groot/runs/<timestamp>/` (falls back to `/tmp/groot_runs/<timestamp>` with a warning if `~/groot` is missing):
`queries.jsonl` = one `meta` record, one `query` record per query (time, state, ages, image sizes/crc, latency, raw chunk, every gate with detail,
verdict, `sent`, and for executed chunks: planned/stretched duration, measured peak speed, tracking error, state after), one `end` record.
`images/qNNNNN_<key>.jpg` with `--save-images` (the exact 640x480 model inputs).

## Mock test

```bash
./run_mock_test.sh [--suite left|right] [--model right19|mocap31|real3nov] [--domain 197] [--port 5597] [--only unit,execute_good] [--out DIR] [--real-port 5556] [--udp-only]
```

`--udp-only` runs every process of the test without Fast DDS's shared-memory transport. Needed on this machine since 2026-10-30: the SHM transport intermittently delivers no camera frame to a
freshly started subscriber (a bare rclpy subscriber got nothing in 10 s in 14 of 25 launches, UDP only 0 of 15), which shows up as random `no inputs within 10 s ... head False` (exit 4) failures in any scenario.

`--suite right` starts the dual-table mock instead (`ros2 launch rebel_dual_table_description dual_table.launch.py hardware_protocol:=mock_hardware mock_move_joint:=false`) with
fake cameras on `/camera/color/image_raw/compressed` and `/arm_right/gripper_camera/image_raw/compressed`, and runs the `tvr_*` scenarios (config identical to the left one except the
documented keys, right-arm functions, recorder-size preprocessing (1152x648 right19, 640x480 mocap31), check mode, observe, good execute, `first_step` and other refusals, `arm_channels` / language / sha mismatch, stale camera).
`--model` (right suite only, default `right19`) selects which right-arm checkpoint is tested: config (`rollout_config_turbovla_right.yaml` / `rollout_config_turbovla_mocap31.yaml`), pinned sha256, `normalization.json` for the `training_range` cross-check,
the stub identity and the start pose of the real scenarios. For `mocap31` the config-diff test additionally asserts that the config differs from the right19 config only in `model.recorder_size`, `server.expect_checkpoint_sha256` and `training_range.min/max`; for `real3nov` only in `instruction`, `model.language`, `server.expect_checkpoint_sha256` and `training_range.min/max`.
With `--real-port 5556` (server started separately with `start_turbovla_server.sh start --profile right19|mocap31|real3nov`, matching `--model`: the sha256 the server announces must be the model's, else the real scenarios report a failure; the script never starts or stops it) it also runs the `tvr_real_*` scenarios
against the REAL model with fake images: modality check, 30 observe queries (latency, gate verdicts) and a short execute on the mock arm. The outputs are meaningless (fake images);
those scenarios only test plumbing, latency and that the gates hold. Use a different `--domain` / `--port` per concurrent run.

Starts the mock single-arm driver (`ros2 launch igus_rebel rebel.launch.py hardware_protocol:=mock_hardware`) and fake cameras as `systemd --user` units
(`groot-mock-stack-<domain>`, `groot-mock-cams-<domain>`) on an empty isolated domain with `ROS_LOCALHOST_ONLY=1`, refuses to run if the hardware is not
`mock_components/GenericSystem`, runs the scenarios (gate unit tests, preprocessing vs an independent copy of the recorder's PIL code, wire compatibility against the real Isaac-GR00T `MsgSerializer`,
observe, execute, every bad stub mode, slow/erroring/dying server, lost reply + retry, modality mismatch, stale camera, Ctrl+C / repeated Ctrl+C / Enter during a running goal, confirmation prompt) and stops
only those two units plus the processes it spawned. Results: `<out>/results.json`, per-scenario run dirs with `console.log` and `queries.jsonl`.
Stub only: `stub_policy_server.py --mode good --port 5555` can also stand in for the server during dry runs on the real stack in observe mode.

## Known limits

- Each chunk is a separate goal that ends at zero velocity: the arm stops and starts every 8 steps (about 0.5 s). Same as the friend's Isaac client.
- Image freshness is receive time on this machine; a driver that keeps re-publishing one frozen frame is not detected (the CRC of each frame is logged).
- `joint_states` freshness cannot tell a frozen real STATUS from a live one (the broadcaster keeps publishing).
- The mock arm has no dynamics (position follows the command exactly); tracking behaviour of the real controller is untested here.

## TurboVLA backend

Same task, same safety machinery, another model: TurboVLA. Two checkpoints, one client, selected by the config (`model.arm_channels`):

| | config | server profile | arm | joints / action |
|---|---|---|---|---|
| left | `rollout_config_turbovla.yaml` | `bundle` (mocap10k step 10000 EMA, the 12 LEFT-arm mocap episodes) | `arm_channels [0, 6]` | `joint1..6`, `/rebel_arm_trajectory_controller/follow_joint_trajectory` |
| right | `rollout_config_turbovla_right.yaml` | `right19` (right19-10k step 4000 EMA, 17 RIGHT-arm episodes, 2 held out) | `arm_channels [7, 13]` | `arm_right_joint1..6`, `/arm_right_trajectory_controller/follow_joint_trajectory` (dual-table bringup, one controller_manager) |
| right (31 Oct) | `rollout_config_turbovla_mocap31.yaml` | `mocap31` (mocap31-10k step 6000 EMA, 20 RIGHT-arm episodes, 3 held out; same language, joints, controller and camera mapping as right19) | `arm_channels [7, 13]` | same as right19; **`recorder_size [640, 480]`** (its training videos are 640x480, all three slots) |

The server (built separately) lives in `tools/turbovla_inference/`; this client only speaks its ZeroMQ protocol. **Real-arm status: mock only, never run on the real arm. First real run needs the operator at the e-stop.**

### Start the server, run the client

```bash
~/igus_rebel_ros2/tools/turbovla_inference/start_turbovla_server.sh start                    # LEFT: ZeroMQ ROUTER on tcp://127.0.0.1:5556 (not groot-server, port 5555)
~/igus_rebel_ros2/tools/turbovla_inference/start_turbovla_server.sh start --profile right19  # RIGHT (stop the other one first; one server at a time)
~/igus_rebel_ros2/tools/turbovla_inference/start_turbovla_server.sh start --profile mocap31  # RIGHT, 31 Oct checkpoint (config rollout_config_turbovla_mocap31.yaml)

export ROS_DOMAIN_ID=11 ROS_LOCALHOST_ONLY=1      # the real stack (or an isolated mock domain)
source /opt/ros/humble/setup.bash && source ~/igus_rebel_ros2/install/setup.bash
cd ~/igus_rebel_ros2/tools/groot_inference
PY=~/venvs/gr00t_client/bin/python
CFG="--config rollout_config_turbovla.yaml"           # right arm: CFG="--config rollout_config_turbovla_right.yaml" (right19) or rollout_config_turbovla_mocap31.yaml (mocap31)

$PY groot_ros_rollout.py $CFG --check                                # inputs, recorder size (640x480 left / 1152x648 right) -> 224 preprocessing, hardware, limits; no server
$PY groot_ros_rollout.py $CFG --max-queries 20 --save-images         # observe: never moves the arm
$PY groot_ros_rollout.py $CFG --execute --max-steps 60               # execute (typed `yes`), first run: short
```

Options that still apply: `--host/--port`, `--max-steps` (executed steps; one episode is 1112-2660 steps at 30 Hz), `--max-time`, `--max-joint-vel`, `--run-dir`,
`--save-images` (the exact 224x224 inputs), `--set gates.max_target_step_rad=0.03`. `--horizon` is GR00T-only (exit 2). `--set policy.backend=...` on the wrong config is refused (exit 2).

### What is sent, what is used

- **Cameras:** `head` = `/camera/color/image_raw/compressed`; the model's `wrist_left` SLOT = `/arm_left/gripper_camera/image_raw/compressed` (left config) or
  **`/arm_right/gripper_camera/image_raw/compressed` (right config: right19 was trained with the right gripper camera in the `wrist_left` slot)**. The knife slot is filled black by the server.
- **Preprocessing** (`preprocess.mode: recorder_pil`, the only mode): the recorder path to `model.recorder_size` (PIL open -> `draft("RGB", size)` -> `convert("RGB")` ->
  BILINEAR resize to that size), then ONE `Image.fromarray(rgb).resize((224, 224))` called without a `resample` argument, exactly as in the
  bundle's data pipeline. Pillow's default is BICUBIC for RGB (since Pillow 7.0); the client's Pillow 9.0.1 and the bundle's Pillow 12.3.0 were compared on a 640x480 -> 224x224
  resize and are bit-identical (also asserted by the mock test when the bundle is on this machine). Result `uint8 (224,224,3)`.
  Left: `recorder_size [640, 480]` (aspect not preserved, 16:9 squashed to 4:3, as the left training videos). **Right: `recorder_size [1152, 648]`**: the right19 training videos are 1152x648 (16:9,
  h264 crf 16) and were stretched to 224x224 with one PIL resize, no crop, so the live path is 1920x1080 (head) / 1280x720 (gripper) JPEG -> BILINEAR 1152x648 -> 224. PIL's JPEG `draft` never
  decodes below the requested size, so for both live resolutions it is a no-op (full decode, then the antialiased BILINEAR resize). The filter that made the 1152x648 videos from the recorder frames is
  not recorded anywhere in the dataset metadata (info.json / modality.json only give 1152x648, h264, yuv420p, 30 fps); BILINEAR (the left recorder's filter) is an assumption.
- **State** `float64 (14,)` = the 6 measured joints of the controlled arm in `model.arm_channels` (left `[0, 6]`, right `[7, 13]`), **every other channel (other arm, both grippers) ALWAYS exactly 0**
  (training contract: their normalisation std is 0.01, any real value is far off-distribution; the server rejects non-zero values there).
- **Request** `get_action` data `{"observation": {"state", "head", "wrist_left", "step"}, "options": None}`; `step` = number of chunks already ensembled (consecutive from 0).
  `reset` is sent without data. **Reply** `[{"actions": float32 (30,14)}, {"inference_ms", "step"}]`: absolute radians; only the `arm_channels` are used, the others are ignored
  (must be finite). A reply whose `step` differs from the request is rejected as a server error.
- **Modality check** at start (`get_modality_config`): `backend == "turbovla"`, server cameras `[head, knife, wrist_left]`, client cameras `[head, wrist_left]`, `image_size [224,224]`,
  `state_dim`/`action_dim` 14, `horizon` 30, `control_rate_hz` 30, `language` (left "Open the box flaps.", right "open the front flap of the cardboard box"), **`arm_channels` equal to
  `model.arm_channels`** (a server that announces none is a pre-right-arm server and counts as `[0, 6]`), and `server.expect_checkpoint_sha256` if set (pinned in the right config). A mismatch refuses `--execute`
  and is a loud warning in observe; an empty/unparseable reply only warns. A GR00T server behind this config is a mismatch (no `backend`), and a TurboVLA server behind the GR00T config
  is refused the same way. `checkpoint_sha256` is logged.

### Control loop (step-synchronous, stop-and-go)

ACT temporal ensemble (`ACTTemporalEnsembler`, a copy with attribution of the bundle's `act_temporal.py`): horizon 30, decay k = 0.01, a fresh chunk EVERY control step, the chunk
queried at step s contributes its element t-s at step t, contributors oldest first with weights exp(-k*i), consecutive step numbers, reset at the start of the run.

Every cycle: snapshot (joints + both images, freshness gates) -> preprocess -> `get_action` -> raw-chunk gates -> ensembler -> ensembled-target gates -> **one** single-point
`FollowJointTrajectory` goal on `/rebel_arm_trajectory_controller/follow_joint_trajectory` with `time_from_start = max(1/30 s, max|dq| / max_joint_vel)` (`max_joint_vel` default **0.1 rad/s**,
timing is only stretched, never compressed) -> wait until that goal has run and the arm settled (in-flight monitor: tracking error, URDF limits, measured speed, joint staleness) -> next cycle.
This mirrors the Isaac runner (simulation waits for inference). **It is stop-and-go: the arm comes to rest at the end of every step, and at 0.1 rad/s the policy's 30 Hz logical time is
not met in wall-clock time (the arm lags the policy; the ensemble is indexed by step, not time).** Smoother streaming (overlapping goals, a higher cap, time-based indexing) is a later tuning
step for the real arm. Mock numbers (stub, mock arm): about one step per 70-100 ms, see "Mock test".

Gates (all evaluated every step and logged; any failure in execute refuses the step, ends the run with exit 3 and holds the arm; in observe they are logged and the loop continues):

| gate | meaning (defaults) |
|---|---|
| `state_in_training_range` | measured joints of the controlled arm inside the training min/max +- 0.1 rad (normalization.json action bounds) |
| `shape`, `finite` | reply is exactly (30,14) numeric and every value finite |
| `first_step` | `max abs(raw chunk[0, arm] - state) <= 0.05` rad (right config: 0.15) |
| `row_step` | consecutive raw rows differ by <= 0.1 rad (extra gate, catches a discontinuity inside the chunk) |
| `urdf_limits`, `training_range` | all 30 raw rows (`gates.check_all_rows: true`; false = row 0 only) inside URDF limits - 0.05 and training range +- 0.1 |
| `target_finite`, `target_step` | the ENSEMBLED target is finite and `max abs(target - state) <= 0.05` rad per step (right config: 0.10) |
| `target_urdf_limits`, `target_training_range` | the ensembled target inside both ranges |
| velocity cap | by timing (above); the in-flight monitor aborts above 2x the cap |

Exit codes as for GR00T (0 completed, 2 usage / not confirmed, 3 refused by a gate, 4 inputs or server unavailable/stale, 5 execution failure, 130 stopped). Stop (Ctrl+C, repeated Ctrl+C, Enter)
cancels the running step goal and commands a hold, exactly as for GR00T. Ctrl+C is NOT an e-stop.

**Observe mode** queries at a fixed period (`observe.period_s`, 0.2 s) and never moves the arm. It still runs the ensembler with consecutive steps on the (static) measured state and
logs what execute would command (`would_command`, `ensemble.target`). Note: ensembling over a static arm is not representative of a closed loop - in execute each chunk is computed from
a state that has followed the previous targets. Use observe to check the plumbing (images, gates, latency, plausibility of chunk[0] vs state), not the quality of the ensemble.

### Logs

Same run directory as GR00T. One `query` record per cycle: step, state, ages, image sizes/crc, `latency_ms` (client round trip), `server_info` (`inference_ms`, `step`), `chunk_summary`
(row 0 / last row / per-joint min,max of the left joints, `row0_minus_state_max`, `max_row_step`, `ignored_channels_absmax`; the full 30x14 chunk with `logging.log_full_chunk: true`),
`ensemble` (step, `history_size`, ensembled `target`, `target_minus_state_max`), every gate with detail, verdict, `sent`, `cycle_ms`, and for executed steps `exec` (planned/stretched duration, command and
measured peak speed, tracking error, state after). `images/qNNNNN_<key>.jpg` (224x224) with `--save-images`.

### Differences to the GR00T backend

| | GR00T | TurboVLA |
|---|---|---|
| server port / stub | 5555 / `stub_policy_server.py` | 5556 / `stub_policy_server.py --backend turbovla` |
| model images | 640x480 | `recorder_size` (640x480 left, 1152x648 right) then 224x224 |
| state | `single_arm` (1,1,6) | 14-D, only the controlled arm's 6 joints non-zero |
| chunk | (16,6), 8 executed per goal | (30,14), the 6 `arm_channels` used, ACT ensemble, 1 target per goal |
| rate | 15 Hz steps | 30 Hz logical, step-synchronous stop-and-go |


### Right arm (right19-10k): what is different, and the expected finding

- Config `rollout_config_turbovla_right.yaml`: joints `arm_right_joint1..6`, action `/arm_right_trajectory_controller/follow_joint_trajectory`, `/joint_states` (dual-table bringup:
  `ros2 launch rebel_dual_table_description dual_table.launch.py hardware_protocol:=mock_hardware|rebel`, right controller 192.168.3.12), `arm_channels [7, 13]`, language
  `open the front flap of the cardboard box`, `recorder_size [1152, 648]`, `server.expect_checkpoint_sha256` pinned to the right19 checkpoint, `training_range` = the right joints of its `normalization.json`.
- **Two gates are raised for this model** (operator request 2026-10-30): `first_step_max_rad 0.15` and `max_target_step_rad 0.10` (left: 0.05 / 0.05). Every other gate, the speed
  limit (`max_joint_vel 0.1`), freshness and execution limits are identical to the left config (asserted by `run_mock_test.sh --suite right`).
- Why: on the held-out episodes `max_j |chunk[0] - state|` has median 0.053 and p95 0.16-0.22 rad, and the ensembled target is about as far from the state (median 0.05, p95 0.15-0.24).
  Offline (open loop) at 0.05/0.05 only 30-39 % of frames pass and a run would stop within ~1 s; at 0.15/0.10 80-81 % pass, first refusal after 542 / 955 frames. A refusal still ends
  the run (exit 3, hold). One step moves at most 0.10 rad per joint at <= 0.1 rad/s.
- **mocap31 (checkpoint mocap31-10k, `rollout_config_turbovla_mocap31.yaml`)**: same joints, controller, cameras, language, gates and limits as above; differs only in the pinned sha256
  (`82eab1f7...`), `recorder_size [640, 480]` and `training_range`. Its training videos (all of head / knife / wrist_left) are 640x480 (info.json), so the live path is the same recorder path as the
  left config (1920x1080 / 1280x720 JPEG -> PIL draft -> BILINEAR 640x480 -> 224; a 16:9 frame squashed to 4:3). Camera mapping (bundle manifest `camera_mapping`): head <- head, knife <- black (server side),
  wrist_left <- wrist_right (the right USB gripper camera), as right19. Its `training_range` is much wider than right19 on joint4 (-3.03..0.17) and joint6 (-0.13..3.02) and narrower on joint1; all
  23 mocap31 episodes start at the home pose 0 / 0 / 60 / 0 / 50 / 0 deg, which `run_mock_test.sh --model mocap31` uses as the start pose of the real-server scenarios. The raised gates are taken over from right19 unchanged (not re-measured).
- Stub for the right model: `stub_policy_server.py --backend turbovla --tv-arm-channels 7,13 --tv-language "open the front flap of the cardboard box" --tv-sha 02f163be... --target <right joints>`.
