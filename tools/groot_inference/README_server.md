# GR00T N1.7 policy server (rebel flap-open checkpoint, trained on the LEFT-arm mocap episodes) - setup and offline check

Offline only: no robot, no ROS. Everything bulky lives on the external drive, reached through the symlink
`~/groot -> "/media/farhan/My Passport/igus_nvidia/groot"`.

## Layout (`~/groot`)

| path | content |
|---|---|
| `Isaac-GR00T/` | repo clone, commit `944fe70c4fd54e3fcb80a7d7258a328f9c061176` (main, 2026-10-22, "Highlight N1.7 edge inference performance"), unmodified (no patches), venv in `Isaac-GR00T/.venv` |
| `checkpoints/rebel_open_frontlid_real_12ep/checkpoint-2000/` | extracted from `~/Downloads/rebel_frontlid_real_12ep.tar.gz` (`training_args.bin` is never loaded) |
| `hf_home/` | `HF_HOME`; holds `nvidia/Cosmos-Reason2-2B` (backbone, 4.6 GB) and a copy of the HF token (`hf_home/token`, mode 600) |
| `uv_cache/` | `UV_CACHE_DIR` |
| `datasets/real_front_5ep/` | the old 5 lid_open (right arm, NOT the training data of this checkpoint) episodes converted with `to_lerobot_gr00t.py ... --arm arm_right --fill-stale 3`, plus `meta/stats.json` + `relative_stats.json` (generated for Isaac's open-loop eval) |
| `configs/rebel_lid_wrist_left_config.py` | modality config (head + wrist_left); only needed by dataset tools, the server uses the config embedded in the checkpoint |
| `logs/` | `server.log`, offline-check outputs (`offline_check*.txt/json`), `open_loop/` |

## Install (what worked)

```bash
mkdir -p ~/groot/{checkpoints,configs,patches,datasets,logs,uv_cache,hf_home,tmp}
git clone https://github.com/NVIDIA/Isaac-GR00T ~/groot/Isaac-GR00T            # needs git-lfs (was present)
cd ~/groot/Isaac-GR00T
export UV_CACHE_DIR=~/groot/uv_cache UV_LINK_MODE=copy HF_HOME=~/groot/hf_home TMPDIR=~/groot/tmp
nice -n 19 ~/.local/bin/uv sync --python ~/.local/bin/python3.12                # torch 2.9.0+cu128, flash-attn 2.8.3 wheel, ~15 GB venv, ~25 min on the NTFS drive
mkdir -p ~/groot/checkpoints && nice -n 19 tar -xzf ~/Downloads/rebel_frontlid_real_12ep.tar.gz -C ~/groot/checkpoints --no-same-owner --no-same-permissions
cp ~/.cache/huggingface/token ~/groot/hf_home/token && chmod 600 ~/groot/hf_home/token
nice -n 19 .venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('nvidia/Cosmos-Reason2-2B')"
```

Run the venv python directly (`.venv/bin/python`), not `uv run`: uv re-installs the URL-pinned flash-attn wheel on every run.
System ffmpeg 4.4.2 is compatible with torchcodec 0.8.0.

## Server

```bash
tools/groot_inference/start_server.sh start     # ~3.5 min cold (weights on NTFS), blocks until ping answers
tools/groot_inference/start_server.sh status | ping | logs | stop | restart
tools/groot_inference/start_server.sh start --port 5556        # port / host / checkpoint are arguments (or GROOT_PORT, GROOT_HOST, GROOT_CHECKPOINT, GROOT_UNIT)
```

It is a transient user unit `groot-server.service` (stop it with `systemctl --user stop groot-server`, which is what `stop`
does; never kill by pattern). The command it runs:

```
~/groot/Isaac-GR00T/.venv/bin/python gr00t/eval/run_gr00t_server.py \
    --model-path ~/groot/checkpoints/rebel_open_frontlid_real_12ep/checkpoint-2000 \
    --embodiment-tag new_embodiment --host 127.0.0.1 --port 5555
```
env: `HF_HOME=~/groot/hf_home TMPDIR=~/groot/tmp PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false`; log `~/groot/logs/server.log`.

* No `--modality-config-path` is needed: `processor_config.json` embeds the config for `new_embodiment`
  (video `head`,`wrist_left`; state `single_arm`; action `single_arm` delta 0..15 RELATIVE/NON_EEF; language `annotation.human.task_description`).
  `get_modality_config` on the server returns exactly that.
* Do not set `HF_HUB_OFFLINE=1`: transformers 4.57.3 then raises `OfflineModeIsEnabled` while loading the Qwen3-VL tokenizer
  (it calls `model_info`). The server therefore needs network (a few small HEAD requests) at start, though nothing is downloaded.
* Attention: `flash_attention_2` is active on the RTX 5070 Ti (sm_120) and works (verified at runtime: backbone config reports
  `flash_attention_2` and inference runs; no SDPA fallback and no patch were needed). The "Flash Attention 2 only supports
  float16/bfloat16 ... dtype is torch.float32" warnings at load are emitted before `.to(bfloat16)`; final params are all bf16.
* Model: Gr00tN1d7, 3.14 B params, all bfloat16 (`model.to(dtype=bfloat16)` in `Gr00tPolicy`), 4 denoising steps.
  GPU memory: 6.3-6.5 GiB for the server process (about 7.2 GiB total GPU used incl. other desktop use).
* Loading printed no missing/unexpected-weights warnings (only albumentations / DiT `FutureWarning`s).

## Wire format (verified)

Request (ZeroMQ REQ, msgpack-numpy): `{"endpoint": "get_action", "data": {"observation": obs, "options": None}}`, with
`obs = {"video": {"head": uint8 (1,1,480,640,3) RGB, "wrist_left": uint8 (1,1,480,640,3) RGB}, "state": {"single_arm": float32 (1,1,6)}, "language": {"annotation.human.task_description": [[text]]}}`.
Reply: `(action, info)` with `action["single_arm"]` = float32 **(1, 16, 6)** (batch, step, joint).
**The values are ABSOLUTE joint targets in rad**, not deltas: the processor decodes the RELATIVE head output by adding the
current state. Evidence: on the 5 episodes the MAE against the recorded absolute `actions[t..t+15]` is 0.054 rad, against
`actions - state` it is 0.71 rad; the first step is within 0.02 rad of the sent state.

## Offline check

`offline_check.py` (runs in the GR00T venv; needs numpy pandas pyarrow pyzmq msgpack msgpack-numpy and a system `ffmpeg` with libdav1d). Two modes:

**1. Against the checkpoint's TRAINING data** (`--mocap-dataset`): the 12 single-episode LeRobot v3.0 datasets `mocap_left_*` of
`MOCAP_DATASET_FLAP_TOP_OPEN` (left arm, 30 fps, AV1 640x480, task text `open the box flaps` (9 episodes) or `open flap 1` (3)). Frames at even indices t (stride 40) go to the server
(head + wrist_left, already 640x480 RGB, left joints `state[t, :6]`), the 16-step chunk is compared with the demo targets `state[t + 2*(k+1)]`, k = 0..15
(the training samples are every 2nd frame). Treat the dataset as untrusted data: read-only, `python -I`.

```bash
R=<dir containing mocap_left_*>
nice -n 19 ~/groot/Isaac-GR00T/.venv/bin/python -I tools/groot_inference/offline_check.py --mocap-dataset "$R" --stride 40 --quiet \
    --instruction "open flap 1" --json-out /tmp/oc_flap1.json       # --instruction "open the front flap of the cardboard box" ; --wrist-source black
```
These are IN-SAMPLE frames (the checkpoint saw them), so this is a pipeline/format sanity check, not a generalisation measure.

**2. Old 5 lid_open episodes** (`--dataset`, 640x400, right arm; NOT this checkpoint's training data, kept for reference):
```bash
~/groot/Isaac-GR00T/.venv/bin/python -I tools/groot_inference/offline_check.py \
    --dataset ~/groot/datasets/real_front_5ep --stride 20 --json-out ~/groot/logs/offline_check.json
```
Isaac's own open-loop eval needs a dataset in the training format (`meta/stats.json` made with `gr00t/data/stats.py --embodiment-tag NEW_EMBODIMENT --modality-config-path ~/groot/configs/rebel_lid_wrist_left_config.py`);
it was run only on the lid_open episodes (MAE avg 0.058, i.e. a wrong-data result, see below).

## Results (2026-10-29, server running the checkpoint above)

### Training data (mocap_left_*, 12 episodes, 542 samples at stride 40, language "open flap 1", wrist_left = real wrist_left)

Pipeline correct: the reply is ABSOLUTE joint targets; MAE 0.0157 rad vs 0.0413 for "hold still" (the model beats doing nothing by 2.6x).
Per episode MAE (rad): 13 0.0130, 15 0.0185, new_1 0.0170, new_10 0.0148, new_2 0.0141, new_3 0.0160, new_4 0.0160, new_5 0.0146, new_6 0.0164, new_7 0.0176, new_8 0.0165, new_9 0.0149.
Per-joint MAE j1..j6: [0.0072, 0.0151, 0.0210, 0.0162, 0.0191, 0.0156]. Speed ratio |pred[15]-state|/|demo[15]-state| median 1.07 (IQR 1.00-1.23).
First-step deviation 0.0176 rad mean (demo's own first step 0.0157), max 0.0455. Direction of the 16-step displacement: cosine mean 0.81,
per-joint sign agreement 0.88; per-joint Pearson correlation j1..j6 `[0.84, 0.94, 0.93, 0.97, 0.93, 0.97]`. Sampling noise (same input x10) 0.008 rad per element;
latency median 66 ms. A repeat of the same run gives MAE 0.0158 (run-to-run noise of the mean about 0.0001-0.0003).

| variant | MAE | speed ratio | cosine | sign agree | per-joint MAE j1..j6 |
|---|---|---|---|---|---|
| "open flap 1" | 0.0157 | 1.07 | 0.81 | 0.88 | 0.0072 0.0151 0.0210 0.0162 0.0191 0.0156 |
| "open the front flap of the cardboard box" | 0.0154 | 1.07 | 0.82 | 0.89 | 0.0070 0.0150 0.0202 0.0165 0.0182 0.0155 |
| "open flap 1", wrist_left = black | 0.0418 | 0.51 | 0.27 | 0.61 | 0.0116 0.0366 0.0427 0.0608 0.0391 0.0601 |
| front-flap text, wrist_left = black | 0.0410 | 0.52 | 0.28 | 0.59 | 0.0109 0.0369 0.0422 0.0595 0.0385 0.0581 |

* **The language text does not matter measurably**: paired difference (same samples) front-flap text minus "open flap 1" = -0.0003 +- 0.0003 rad (1 se), the same size as the
  run-to-run difference of two identical runs (+0.0001 +- 0.0003). The processor lowercases and strips punctuation (`formalize_language`), so "Open flap 1." == "open flap 1".
  The config uses the training text "open flap 1" (in-distribution, no downside).
* **The model depends heavily on the wrist_left image**: with a black wrist image the MAE is 0.042 (= hold-still 0.041), the motion is half as large and the direction is lost (cosine 0.27).
  The live gripper camera must therefore look like the recorded one (same camera, mounting, exposure); see `/tmp/groot_cam_compare.png` for what the live pipeline produces.

### Old lid_open episodes (wrong data for this checkpoint; kept for the record)

Mean MAE vs demo 0.054 rad, hold-still 0.027 (the model was NOT better than doing nothing), speed ratio median 1.93, direction cosine 0.02, joints 4-6 often opposite.
That was expected in hindsight: those 5 episodes are right-arm recordings with a different wrist branch / joint convention and are not in the training distribution of this checkpoint
(state j3 min 0.26 vs 0.0, j4 range -3.02..2.94 vs 0..1.42, j6 -3.02..3.03 vs -0.91..0.31; training start pose `[0, 0, 1.047, 0, 0.873, 0]`, not zero).

### Pipeline checks with the real model (mock arm, 2026-10-29)

* `/tmp/groot_cam_compare.png`: row 1 head, row 2 wrist_left; left = frame 0 of `mocap_left_new_1` (training video), right = the old lid_open `real_0001` frame 0
  (recorded at half resolution, upscaled to 1920x1080 / 1280x720, then through `preprocess.mode recorder_pil`). Same aspect handling; the scenes differ (other arm / camera pose),
  so only the framing/aspect/colour pipeline is comparable, not the content.
* Mock e2e (isolated domain, `rebel.launch.py hardware_protocol:=mock_hardware`, fake cameras publishing mocap_left_new_1 frame 0 at live size): observe 4 queries all PASS
  (chunk[15]-state ~ +0.10..0.13 rad on j3, -0.09 on j5, same direction as the demo's first motion); `--execute --mock-yes --execution.max_steps=64`: 8 chunks, all gates PASS, 5.6 s at the 0.1 rad/s cap,
  executed displacement `[-0.011 -0.143 0.401 0.082 -0.348 -0.076]` vs demo frames 0..128 `[0 -0.129 0.339 -0.031 -0.305 0.015]` (cosine 0.96).
  The demo arm is still for its first ~2 s, the model starts moving at once from the identical frame-0 input (the idle time is not predictable from a static image).
