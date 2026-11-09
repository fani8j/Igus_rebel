# side_cut_vision: passive side-cut camera probe (Stage A, measure only)

Measures, with the knife-holder Gemini 305 (RGB-D), where the carton top plane, the cut-side edge and the tape
seam really are, and compares them with the perception snapshot. **Nothing is corrected and nothing moves**: the
probe only subscribes (TF, `/joint_states`, coordinator status, camera topics) and publishes its own overlay image
and a JSON measurement. Plan: `scan_results/side_cut_camera_align_20261021/PLAN.md`, spec: `STAGE_A_SPEC.md`.

## What it produces

* Live, while the coordinator is `WAITING_BARRIER`/`EXECUTING` (or `--always-live`): ~3 Hz per arm,
  `/side_cut_vision/arm_<left|right>/overlay/compressed` (JPEG, 960x600, model in red, measurement in green, knife tip in
  yellow, text block) and `/side_cut_vision/arm_<arm>/measurement` (`std_msgs/String`, JSON), plus a per-frame
  `side_cut_vision/<arm>_live.jsonl` next to the cut_recorder clip.
* One **LOOK record** per still period: the arm's joints are still (max |dq/dt| < 0.01 rad/s over 0.1 s) for >= 0.3 s
  **and** the tool tip is within 40 mm of the model cut-side edge line and outside the carton. That is the
  `stabilize blade before ... side cut` dwell. Three frames of the still window are measured (two-frame agreement
  <= 1 mm, else flag `unstable`) and saved to `<clip>/side_cut_vision/<arm>_<NN>_<HHMMSS>/`:
  `color.png depth.png(16 bit) camera_info.json T_world_optical.json T_world_tip.json joint_state.json snapshot.json
  calib.yaml measurement.json overlay.png`. `<clip>` is the newest `cut_recorder` clip folder whose `clip.json` has
  `"recording": true` (and was modified in the last 30 min, so a crashed recorder does not capture it); otherwise
  `<out_dir>/side_cut_vision_<YYYYmmdd>/`.

### Convention (read this before comparing with the executor)

World = the snapshot frame = the left arm's base. Cut basis at the crossing S: `along` = edge direction back -> front,
`normal` = snapshot surface normal (up), **+`across` points OUT of the carton, for both arms**. This is the **opposite
of the executor's left-cut `cutDebugBasis` across** (`normal x along`, which points INTO the carton for the left face).
`measurement.json` carries `"across_convention": "outward"`. delta = measured - model, in mm `[along, across, normal]`,
plus `delta_yaw_deg` (edge direction in the top plane, + = turning outward as it advances) and
`tip_vs_seam` = knife tip relative to the measured crossing (`along`, `inward_from_edge` (+ inside the carton),
`below_top`).

## Real cell: how to run

Facts of the cell: the cameras run on **ROS domain 11** (`ROS_LOCALHOST_ONLY=1`); the dual robot stack and the
cut_recorder run on **domain 12** (`ROS_LOCALHOST_ONLY=0`). The probe runs in the **robot environment** (12) and reads
the cameras through a second rclpy context in `camera_domain_id` (11), the same pattern as `dual_cut_ui`.

1. The cameras must be up with both wrist cameras: `ros2 launch rebel_cameras cameras.launch.py wrist:=true right_wrist:=true`
   (domain 11). Topics used per arm: `/arm_<arm>/wrist_camera/color/image_raw`, `/depth/image_raw` (aligned to colour,
   `depth_registration: true`, 16UC1 mm), `/color/camera_info` (8 distortion coefficients, treated as rational).
2. Start the probe (adds ONE process, `nice -n 10`, ~15 % of a core idle, <= 1 core live; it subscribes to the 5 MB
   colour/depth frames only while a run is live and releases them 5 s after):

   ```bash
   source /opt/ros/humble/setup.bash && source /home/farhan/igus_rebel_ros2/install/setup.bash && source /home/farhan/igus_rebel_ros2/install_vision/setup.bash && ROS_DOMAIN_ID=12 ROS_LOCALHOST_ONLY=0 ros2 launch side_cut_vision side_cut_probe.launch.py camera_domain_id:=11
   ```
   Launch args: `camera_domain_id` (required), `arms` (`left,right`), `left_calib`, `right_calib`, `snapshot_file`,
   `out_dir`, `live_rate_hz`, `overlay_context` (`robot`: the recorder's domain). The snapshot defaults to the executor's
   `/home/farhan/Igus_rebel/outputs/seam_3d_base_link.json` and is reloaded when its mtime changes.
3. Optional, the operator's choice: restart the recorder so that it also records the overlays and reads the cameras in
   domain 11 (the running recorder is NOT touched by this package):

   ```bash
   ROS_DOMAIN_ID=12 ROS_LOCALHOST_ONLY=0 python3 /home/farhan/igus_rebel_ros2/tools/cut_recorder/cut_recorder.py --coordinator-status-topic /dual_cut_coordinator/status --post-roll-s 5.0 --camera-domain 11
   ```
   (`--cameras` defaults now include `left_vision,right_vision`: the overlay topics in the robot domain, recorded into
   15 fps mp4s by repeating each ~3 Hz overlay until the next, so they stay in time sync with `wrist.mp4`.)

Stop the probe with Ctrl-C. It is purely additive: it has no service/action clients, publishes nothing but its two
topics per arm, and never sets a parameter (`test_passivity.py` enforces it on the source).

### Known open points for the real cell

* **`ROS_LOCALHOST_ONLY` 0/1 mix is unverified.** The cameras (domain 11) are localhost-only, the probe runs with
  `ROS_LOCALHOST_ONLY=0` (robot environment). One process makes both contexts with the same environment variable, so
  whether the probe's domain-11 context discovers the localhost-only camera publishers has not been tried on the real
  cell. If the log shows no frames (`frames: 0` in the 30 s status line, or the arm lines stay at "subscribed" with no
  overlay), fall back to `camera_domain_id:=12` only if the cameras are moved to 12, or run the probe with
  `ROS_LOCALHOST_ONLY=1` and check that it still sees `/tf` of the robot stack. The mock runs use one domain, with the
  second-context code exercised only in a separate two-domain smoke test.
* **Right arm calibration is a SEED.** There is no right hand-eye. The right arm uses the left result
  (`calib_status: seed_from_left_UNCALIBRATED`, shown as a banner on every overlay and in every JSON/record). Do not read
  right-arm numbers as measurements. Pass `right_calib:=<wrist_handeye/v1 yaml>` once it exists.
* **Left hand-eye quality.** The 2026-10-13 `result.yaml` says LOO position rms 5.2 mm (max 8.75 mm), reprojection rms
  6.6 px, not the <= 1.6 mm assumed in PLAN.md. Expect Delta along/across/normal to carry several mm of hand-eye error
  until the calibration is redone (S3). The sha256 of the calibration is recorded in every measurement.
* **Depth topic.** `depth/image_raw` is the colour-aligned depth only with `depth_registration: true`
  (`config/wrist_camera.yaml` sets it, `align_target_stream: COLOR`). A size mismatch with the colour image is reported
  as `size_mismatch`, not guessed.
* **TF.** robot_state_publisher publishes TF only when joints move (<= 20 Hz), so at rest there is no TF sample at the
  image stamp. The probe then uses the newest sample if it is within 60 ms of the image or the arm has been still for
  >= 0.2 s (`meta.tf_mode` = `at_stamp` / `latest_near` / `latest_at_rest`). It always requires a `/joint_states` sample
  within 50 ms of the image.

## Package layout

Library (pure numpy/OpenCV, no rclpy): `camera.py` (K + 8-coefficient rational distortion), `model.py` (snapshot, per-arm
carton model, cut basis), `detect.py` (top plane RANSAC, edge, seam, side plane), `measure.py`, `overlay.py`,
`toolmask.py` (tool region from the handpiece hull projected through the hand-eye; static polygon as fallback),
`coloronly.py` (qualitative colour-only detector for recorded video), `trigger.py` (still tracker, LOOK trigger,
record-dir selection), `record.py`, `probe_core.py` (everything the node does, with an injectable TF/clock),
`synthcam.py` (synthetic RGB-D renderer, tests and the mock camera).
ROS: `probe_node.py` (`side_cut_probe`), `mock_camera.py` (`mock_side_cut_camera`, MOCK ONLY), `launch/side_cut_probe.launch.py`.

### Tool mask

The knife body occludes the right ~25 % of the image. The mask is the convex hull of the collision geometry of the
handpiece (`igus_rebel_description/meshes/knife_c/collision`, in `igus_tool0`) clipped at 70 mm and projected through the
hand-eye and the `link6 -> igus_tool0` transform (read from TF at start, URDF constant as fallback), dilated 12 px,
plus the near-field holder block at the right image edge (x >= 1130 of 1280, no depth there). The holder and
bracket parts wrap around the camera itself (within ~70 mm) and are not projected. Every measurement records which
mask was used (`meta.tool_mask`: `hull_tf+near_field`, `hull_urdf_const+near_field`, or `static_polygon`).

## Tools

* `side_cut_vision_replay record <look_dir>...`: re-run the detector on saved LOOK records and re-draw the overlay.
* `side_cut_vision_replay color-only <jpg|mp4> --out-dir DIR [--at-s T ...] [--orientation horizontal|vertical]`:
  qualitative colour-only edge/seam candidates on recorded frames (the 2026-10-20 clips have no depth). No metric output.

## Tests

`cd src/side_cut_vision && python3 -m pytest -p no:anyio test -q` (also via `colcon test`). The synthetic renderer
(`test/synth.py`) uses the real camera intrinsics (8 distortion coefficients), the real left hand-eye and a real snapshot
copied into `test/data/`. Mock end to end (domain 203, one domain for robot and cameras):
`scan_results/side_cut_camera_align_20261021/mock/` (`start_mock.sh`, evidence).
