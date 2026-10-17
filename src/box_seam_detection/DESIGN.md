# box_seam_detection — design and integration contract

The RGB-D box top-face + tape-seam detector, moved into igus_rebel_ros2 (2026-10-12) from two external repos:
`~/Downloads/rgbd_captures` (library `scripts/autolabel_v2/*`, helpers `scripts/autolabel/*`, modules of
`precision_box_detector`) and `~/Downloads/seam_handoff_20260908/scripts` (live detector `autolabel_v2_detect.py`,
orchestrator `capture_and_query_depth.py`, `query_depth_at_pixel.py`, `camera_point_to_base_link.py`,
`side_cutting_point.py`, `side_face_seed.py`, `rgbd_recorder.py`). This package is the source of truth from now on;
the external copies are frozen. Detection results must stay identical (431-capture replay + 8-capture full
orchestrator bench + blind sets).

The H-cut executor contract is unchanged: it reads `box_h_cut_snapshot/v1` from `external_snapshot_file`
(`igus_rebel_moveit_config/config/box_h_cut_executor.yaml`), checks mtime + `acquisition_stamp_ns`.

## Layout

```
box_seam_detection/
  core/            library (was rgbd_captures scripts/autolabel_v2): pipeline, config, fold_edge, seam_estimator,
                   corner_names, corner_checks, coarse, overlay, box_profiles, tape_strip, sam3_worker, sam3_daemon,
                   sam3_session, sam3_live, run (offline autolabel CLI)
    autolabel/     helpers (was rgbd_captures scripts/autolabel): depth_top, pcd, ... (only what core needs)
    pbd/           vendored precision_box_detector modules core needs (sam3_depth_fusion, plane_fit, camera_model,
                   subpixel_edge, depth_geometry, tf_transform + transitive deps), each with a provenance header
                   (source repo, path, commit). The side-tape session keeps using its own precision_box_detector.
  live/            detector.py (was autolabel_v2_detect.py; keeps --serve for the replay harness),
                   query_depth.py, base_link.py (calib ∘ driver extrinsic; no rclpy init inside library functions),
                   side_points.py, side_seed.py, snapshot.py, recorder.py (fallback capture), orchestrator.py
  ros/             detect_node.py (DetectBox action server), capture.py (depth image + CameraInfo -> cloud,
                   freshness), markers.py
launch/box_seam_detection.launch.py, config/box_seam_detection.yaml, test/
```
No `sys.path` manipulation: package-relative imports only. The SAM3 interpreter (conda `book_grasping`) is a
parameter (`sam3_python`, env `BOX_SEAM_SAM3_PYTHON`, default `/home/farhan/miniconda3/envs/book_grasping/bin/python`);
the worker/daemon are launched as files (`Path(__file__)`), so the package is installed unzipped.

## live.orchestrator API (shared by the CLI and the ROS node)

```python
@dataclass
class CaptureFiles:
    capture_id: str
    image_path: Path          # rgb png
    pcd_path: Path            # camera-frame XYZ(RGB) cloud registered to the colour camera
    manifest_path: Path       # same schema as today's captures/<id>.json (camera info, stamps, frames, extrinsic)
    stamp_ns: int             # acquisition stamp of the colour image

@dataclass
class DetectConfig:           # every option of today's capture_and_query_depth.py --autolabel-v2 that matters
    max_attempts: int = 4; retry_policy: str = "smart"; allow_needs_review: bool = False
    allow_cut_blocked: bool = False; dry_run: bool = False; rect_tol_mm: float | None = None
    sam3_python: str = ...; sam3_daemon: str = "auto"; downstream_inprocess: bool = True
    calib_file: str = "~/.ros2/easy_handeye2/calibrations/rebel_handeye_h_v2.calib"
    driver_extrinsic: str = "recorded"      # node records it in the manifest at capture time
    target_frame: str = "base_link"; side_seed: bool = True
    output_root: Path = ...                 # pipeline/<capture_id>/<run>/ outputs (as today)
    snapshot_path: Path = Path("/home/farhan/Igus_rebel/outputs/seam_3d_base_link.json")

@dataclass
class DetectResult:
    success: bool; status: str            # accepted | rejected | blocked | error
    reason: str; reasons: list[str]; cut_permission: dict | None
    snapshot: dict | None; snapshot_path: str | None; capture_id: str | None
    seam_status: str | None; attempts: list[dict]; elapsed_s: float; pipeline_dir: str | None

def run_detection(next_capture: Callable[[int], CaptureFiles], cfg: DetectConfig,
                  progress: Callable[[str, int, str], None] | None = None) -> DetectResult
def main(argv=None) -> int                # CLI; keeps the offline flags of capture_and_query_depth.py
```

Safety rules (fail closed):
1. At the start of every run (not dry_run) the existing snapshot file at `snapshot_path` is removed (as the
   orchestrator does today), so a failed / blocked detection can never leave an older fresh snapshot behind.
2. The snapshot is written (atomic same-directory replace, as `_write_json_atomically`) only when the detection is
   accepted AND (`cut_permission.status == "allowed"` OR `allow_cut_blocked`); the snapshot JSON carries the
   `cut_permission` block (incl. any override) — the executor ignores unknown fields.
3. `acquisition_stamp_ns` = the colour image stamp; `side_tape_seed.source` matches snapshot_id + stamp; a seed
   failure quarantines as today (`.side_seed_rejected`).
4. base_link = calib ∘ driver extrinsic exactly as `camera_point_to_base_link.py` (do not switch to plain tf2).

## ROS node (ros.detect_node)

- Action server `~/detect_box` (`compal_box_msgs/action/DetectBox`), MultiThreadedExecutor; subscriptions in their
  own callback group, the action in a reentrant group, detection in one worker (one goal at a time; a second goal is
  rejected while busy). Feedback per stage/attempt.
- Capture (`ros.capture`): subscribe to colour image, aligned depth image and colour CameraInfo of the scene camera
  (topics as parameters; defaults from `rebel_cameras/config/scene_camera.yaml`). Do NOT subscribe to the coloured
  point cloud (34 MB, driver CPU load). For each attempt: wait for a synced pair whose stamps are AFTER the attempt's
  request time (+ margin, timeout), build the camera-frame cloud from the aligned depth + colour K, write rgb png +
  pcd + manifest (same schema as today's recorder, including the driver extrinsic camera_link -> colour optical
  frame looked up from tf2 at the image stamp) under `output_root/{rgb,pointcloud,captures}`, return CaptureFiles.
  Parameter `capture_source: node | recorder` (recorder = today's rgbd_recorder path, fallback until node capture is
  validated on the live camera).
- RAM-first capture staging (`live.capture_staging`; node parameter `capture_staging_dir`, CLI `--capture-staging-dir`,
  env `BOX_SEAM_CAPTURE_STAGING`, default `/dev/shm/box_seam_detection/staging`, `off` = write straight to the archive):
  captures are written to a per-process RAM session dir and the detection runs from there. After the result is known
  (the snapshot is already published; CLI: synchronously before exit, node: one background archiver thread, bounded
  wait in `destroy_node`) the accepted run's capture is archived to `<output_root>/{rgb,pointcloud,captures}` and every
  other capture to `<output_root>/failed_recent/{rgb,pointcloud,captures}` (3 most recent; only files its journal
  `.ring.json` lists are ever deleted). Archived manifests get their paths rewritten and re-validated; staged paths in
  the `pipeline/<id>/<run>/` JSON records are text-replaced (never the planner snapshot). Leftovers of dead processes are
  archived to `failed_recent` at the next start. `--from-image` never stages.
- SAM3 daemon warm-up at node start (`ensure_daemon`, shared singleton, never stopped by the node); readiness in
  the status.
- Publishes: latched status (std_msgs/String JSON or a small msg), MarkerArray of the seam/corners in base_link,
  overlay image. Nothing that commands the robot.
- Never touches ROS domain 11 in tests: tests use ROS_DOMAIN_ID=87, ROS_LOCALHOST_ONLY=1.

## Validation gates

- core: offline autolabel runner outputs identical to rgbd_captures run.py on the blind sets and a capture sample.
- live: 431-capture replay (rgbd_captures `scripts/eval/live_reliability.py` pointed at `box_seam_detection.live.detector`)
  vs `eval/multibox_final_20261006` with `replay_diff.py --float-tol 1e-9` and exactly 431 ids; 8-capture full
  orchestrator bench (`bench_speed.py` → `box_seam_detect`) vs the old orchestrator: planner snapshots identical.
- ros: node tests with fake publishers built from saved captures (PCD re-projected to an aligned depth image), the
  produced snapshot identical to the CLI on the same capture (modulo stamps/ids), fail-closed behaviour, mock H-cut
  planning with the produced snapshot (isolated domain, mock_hardware, plan only).

## How to run the e2e check

Node + replayed saved captures + real detection (SAM3), no robot, no camera driver. Tests live in `test/e2e/` and are
skipped unless `BOX_SEAM_E2E=1` (they need the GPU/SAM3 conda env and the saved captures under
`/home/farhan/Igus_rebel/outputs/rgbd_captures/captures`, override with `BOX_SEAM_E2E_CAPTURES`).

```bash
cd <workspace>; source /opt/ros/humble/setup.bash
colcon build --packages-select compal_box_msgs box_seam_detection --symlink-install && source install/setup.bash
BOX_SEAM_E2E=1 BOX_SEAM_E2E_SCRATCH=/some/scratch/dir \
  python3 -m pytest src/box_seam_detection/test/e2e -v -s        # ~3 min, 11 tests
```

- Isolation: every spawned process gets `ROS_DOMAIN_ID=87` (override `BOX_SEAM_E2E_DOMAIN`, never the robot's domain 11),
  `ROS_LOCALHOST_ONLY=1`, and the SHM-only FastDDS profile `test/e2e/fastdds_shm_only.xml` (no network sockets).
  Without that profile loopback UDP (`ROS_LOCALHOST_ONLY=1`, `net.core.rmem_max` 208 KB) drops most 4-6 MB image
  messages and the node reports `CaptureTimeout`; it is a test-environment effect, not a node defect.
- The node writes `snap.json` and `out/` only inside the scratch dir (`snapshot_path` / `output_root` parameters);
  the real executor snapshot `/home/farhan/Igus_rebel/outputs/seam_3d_base_link.json` is never touched.
- What is checked: accepted captures (small box x2, high-yaw R2) -> success, `cut_permission` allowed, executor-parser
  port (`e2e_lib.executor_snapshot_problems`, mirrors `box_h_cut_executor.cpp` parse + validate + side seed),
  `acquisition_stamp_ns` is a replayed colour-image stamp, feedback stages, status topic busy -> idle, markers, overlay;
  node snapshot vs `box_seam_detect --from-image ... --driver-extrinsic recorded` (tolerance `BOX_SEAM_E2E_TOL_MM`,
  default 1 mm; seen 0 / 0.06 / 0.44 mm, caused only by the node rebuilding the cloud from a 16UC1 depth image,
  <= 5e-7 m per point, amplified by discrete fit decisions; the CLI run on the node's own capture is identical);
  rejected / blocked captures remove a previously written snapshot; a second goal while busy is rejected; a stopped
  camera gives `status=error` (CaptureTimeout, goal ABORTED), no snapshot, and the node recovers afterwards.
- Manual: `python3 -m box_seam_detection.ros.fake_camera <manifest.json>` + `ros2 run box_seam_detection
  box_seam_detector_node --ros-args -p snapshot_path:=... -p output_root:=...` (same env as above), then
  `ros2 action send_goal /box_seam_detector/detect_box compal_box_msgs/action/DetectBox "{max_attempts: 1}" --feedback`.
- Mock H-cut planning with the produced snapshot (plan only): launch per the workspace CLAUDE.md "Running in mock"
  (`hardware_protocol:=mock_hardware`, `run_box_h_cut_executor:=true`, camera static TFs from HANDOFF.md), then
  `ros2 param set /box_h_cut_executor external_snapshot_file <scratch>/snap.json` (fresh stamp, max age 240 s) and
  call `/box_h_cut_executor/plan_h_cut`. Keep `execution_enabled` false.

## Scene configs (`config/scenes/*.yaml`, `--scene-config`, node parameter `scene_config`)

A scene bundles the camera/cell-specific constants (core/scene.py): depth-top workspace + support mode, guided-plane
(wrong-face) gate, reliability overrides, default calib file, driver-extrinsic reference/fallback, seam-direction prior,
rectangle-gate tolerance. `table_cell_v1` = today's values and the default (built in, no file read: bit-identical);
`dual_arm_v1` = dual-arm cell (box on a stand smaller than the box, floor visible, 1.0-1.3 m, ~2.7 mm depth noise,
left-arm calib `dualcal_20261009_left_single`, NOT verified by touches).

- `support: hidden_stand`: the support (floor) is the FARTHEST large gravity-aligned plane family, NOT the largest plane
  (2026-10-13): `depth_top.pick_hidden_support` (one helper for `identify_table_top`, `pipeline.estimate_table_plane`,
  hence the height / independent check / trim plane choice). The detector passes `depth_top.gravity_cam` (base_link +z in
  the camera frame, from the hand-eye calibration); candidate planes face the camera with their normal within
  `support_gravity_tol_deg` (20) of it (a back wall or a ceiling never qualifies), are grouped (parallel within
  `parallel_tol_deg`, offset within `support_merge_m` = 0.08 m: a noisy far floor splits into fragments 20-30 mm apart),
  inliers summed, groups >= `table_min_inliers` compete, the farthest wins; the support plane is the group's largest member
  (no refit). Fail closed: calibration present but no such family -> no support (depth check unconfirmed). Without a
  calibration the old rule (largest plane) applies and is tagged `support_gravity_unchecked`. The floor footprint merge uses
  `support_merge_m` too. `box_profile.support_clearance_mm` records the top-to-floor distance along the floor normal
  (box + stand: an UPPER bound on the box height; recorded only). visible_table is untouched.
  Old text: the dominant plane is the floor; the top is a large plane parallel to it, `offset_min..offset_max`
  above, inside the floor's footprint. A desk at the same height is indistinguishable by depth: `depth_ambiguous_top` is
  resolved by the SAM3 candidate's ROI (floor passed as `support_hint`). The floor distance is never a box height:
  `box_profile.height_unavailable = "support_hidden"`; a KNOWN profile (large_392) then uses its own height (fold, side
  seed) and is not cut-blocked for it; generic still is, unless the scene sets `live.generic_cut_without_height: true`
  (dual_arm_v1): then a generic box with `height_unavailable == support_hidden` is not blocked for the height iff
  `checks.independent_top_check.confirmed` (not a circular rectangle-fitted skip), `box_profile.support_clearance_mm` > 0
  (and no `support_gravity_unchecked`), and `top_face.depth_diag.edge_support` has all 4 edges `ok` with `valid_frac >= 0.8`
  and none unsupported (missing diagnostics / insufficient depth stay blocked). The permission then carries
  `flags: ["height_waived(support_hidden)"]`. The side-tape seed still needs a height (stays blocked). The snapshot then carries the
  optional additive keys `support_depth_below_top_m` (= support_clearance_mm / 1000; for a table-cell scene the measured
  box height) and `support_depth_below_top_source`; omitted when the floor was not gravity-checked
  (`support_gravity_unchecked`) or nothing was measured (schema stays box_h_cut_snapshot/v1: the Python validator and the
  executor's ptree parser ignore unknown keys). The executor sizes the carton collision box with it: height = snapshot
  value if `carton_height_from_snapshot` and `carton_height_m` <= value <= `carton_height_max_m` (2.0), else
  `carton_height_m` (0.20 = the minimum; WARN if the value is invalid or above the maximum).
- `v2.noise_adaptive`: guided-plane band and p95 limit grow with the measured scene noise (median RMS of the fitted planes)
  above `rms_floor_mm`, plus a bounded lid-bulge allowance; never below today's values.
- `live.driver_extrinsic.reference: null` = no verified reference for the camera (manifest block used with a warning);
  `fallback` = value composed for a `recorded` run without a manifest block (tagged `recorded_constant_assumed`).
- `large_392` profile (39 x 32 x 48 cm carton): see core/box_profiles.py.
- `live.top_size_range_mm: {long: [min, max], short: [min, max]}` (2026-10-17; default null = gate off; `dual_arm_v1`:
  long 330-450, short 230-380): HARD top-size gate in `live/detector.py::top_size_check`, run on the FINAL top quad (after
  truncation recovery, depth-edge fix and R9; the measured RGB quad when the published one is the gap-anchored K1 quad),
  for EVERY profile (generic included, which has no profile-dims check). long/short = mean of the longer/shorter opposite
  side pair of the metric 3-D quad (rotation-invariant, no position constraint). Outside -> exit 1, stage `top_size`,
  `top_size_out_of_range(long=..mm,short=..mm,range=..)`, `retryable=True` (a fresh capture may segment the whole top);
  range configured but quad not measurable -> `top_size_unmeasurable(..)` (fail closed, not retryable). NOT bypassable by
  `--allow-needs-review` / `--allow-cut-blocked` (it is a reject, not a cut_permission reason). Motivation: a carton with
  already-cut tape whose near flap droops was detected as only its flat far flap (372 x 144 mm, 391 x 117 mm) and
  published. The measurement is recorded in `autolabel_v2.top_size_check` (`enabled`, `range_mm`, `long_mm`, `short_mm`,
  `sides_mm`, `pass`). The smart retry policy compares consecutive `top_size_out_of_range` rejects (long and short each
  within 8 mm) and stops early like other deterministic rejects. The scene value is validated at load (two `[min, max]`
  pairs, finite, `0 < min < max`).
- Orchestrator contract: a detector seam JSON whose `cut_permission` is missing, malformed or has a status outside
  `{allowed, blocked}` aborts the run (status `error`) regardless of `--allow-cut-blocked`; only a real `blocked` can be
  overridden.

## Work area (2026-10-17; scene option `live.work_area`, `core/work_area.py`; off when absent)

The robot can only cut a box inside its reach, and the geometry stage (`post_sam3`: full-cloud RANSAC planes, guided
plane, table plane, pre-refine, 2.6-5.6 s) works on all 1.7 M points. The scene may declare a work area in the detector's
target `base_link`: `polygon_xy_m` (simple polygon, >= 3 finite vertices), `top_z_range_m` (allowed height of the box TOP),
`outside_keep_every` (int >= 1, or 0 = drop all other outside points), optional `margin_m` (default 0.10) and
`floor_keep_below_z_m` (default null). `dual_arm_v1`: TCP reach of both arms at box-top height in front of the robot table
(x >= 0.11), an UPPER bound (no collision or tool-orientation check). Validated at scene load (`SceneError`).

- **Cloud crop** (`detector.run_request`, before SAM3 is touched; `work_area_crop` in `timing_s`, ~0.2 s): the cloud is
  moved to base_link with `_camera_to_base_matrix` (the SAME hand-eye calib + driver-extrinsic composition as
  `seam_direction.base_link_*`; no new transform). Kept: every point inside the polygon grown by `margin_m` at ALL heights
  (box side walls, stand and floor under the box feed the support / hidden-stand / edge-support / depth-edge / truncation
  checks; the margin covers the 80 mm edge-support probes), every point below `floor_keep_below_z_m` (the floor) and every
  `outside_keep_every`-th remaining point. The kept points are written as an unorganized binary PCD plus a manifest copy
  (`<work_dir>/<capture_id>_work_area{.pcd,_capture.json}`, removed after the run unless `--debug`); the pipeline and the
  depth-edge fix read those, the originals and the manifest binding checks are untouched. No consumer assumes an organized
  cloud or pixel-indexed points (the PCD reader and the z-buffer projection treat it as unorganized; the cropped manifest
  gets the new `point_count/width/row_step`). Stats (`points_total/inside/outside/outside_kept/floor_kept/kept`, `time_s`)
  go to `autolabel_v2.work_area_check.crop` (and the reject JSON).
- **The floor must stay dense** (measured, 10 offline frames of the good box, 2026-10-16): thinning the floor with the rest
  (`outside_keep_every` 2..5, 10) changed accept/reject on 2-4 of 10 frames (no top found `depth_no_top`, `height_unavailable`,
  r2/gross-rect trims): the hidden-stand support is the farthest large gravity-aligned plane family with
  `table_min_inliers` (absolute count) and the sequential RANSAC (`max_planes` 6) ranks planes by size, so a floor thinned x10
  drops behind the clutter planes inside the area and the support / box height are lost or change. `outside_keep_every: 1`
  (no thinning, only the path through the cropped files) reproduced the uncropped result exactly. With the floor dense
  (`floor_keep_below_z_m: -0.75`, floor at base z ~ -0.83) and the rest thinned x10 all 10 frames are accepted. Re-measure
  when the camera pose or the floor level changes.
- **SAM3 candidate filter** (`pipeline._drop_candidates_outside_work_area`): the work-area prism (polygon at `zmin` and
  `zmax`; union of the projected top / bottom polygons and side quads, edges densified) is rasterized in the image with the
  same pinhole the pipeline projects the cloud with. A `sam3_live` / `sam3_archived` candidate (primary or alternate) whose
  quad has < 80 % of its area inside is dropped before ranking: `coarse` None, reason
  `sam3_candidate_outside_work_area(inside=0.12<0.80)` in `provenance.attempts` / `reasons` and in
  `work_area_check.candidates_dropped`. The worker does not return masks, so the candidate quad (the mask's min-area / poly
  quad) stands in for the mask; the image is never cropped. If the prism cannot be projected (behind the camera) the filter
  is skipped (`prov.work_area.applied=false`); the hard check below still guards the output.
- **Hard reject** (`detector.work_area_check`, right after the top-size gate): all four final top corners (and the measured
  RGB quad of a gap-anchored K1 record) in base_link must lie inside the polygon (no margin; boundary counts as inside) with z
  in `top_z_range_m`, else exit 1, stage `work_area`, `box_outside_work_area(xy: back_left=(x,y); z: ...; polygon=<hash>)`,
  `retryable=false`, NOT bypassable by `--allow-needs-review` / `--allow-cut-blocked`. A configured work area with no
  base_link transform (calib unreadable) -> `work_area_transform_unavailable(..)` before SAM3 is queried; corners that cannot
  be measured -> `work_area_unmeasurable(..)` (both fail closed, not retryable). Recorded in `autolabel_v2.work_area_check`
  (`enabled`, `pass`, `polygon_hash`, `corners_base_link`, `xy_outside`, `z_outside`, `crop`, `candidates_dropped`;
  `{"enabled": false}` when the scene has no work area) and in the record.
- Tests: `test/core/test_work_area.py` (validation, polygon, crop, prism, candidate filter, corner check),
  `test/live/test_work_area_gate.py` (CLI: crop wiring, hard reject, fail closed, flags do not bypass, feature off).

## Fixed floor (2026-10-17; scene option `live.fixed_floor`, `core/fixed_floor.py`, `depth_top.top_refit_iters`; off when absent)

**Problem** (`dual_arm_v1`, fixed camera, hidden stand): a static, well-taped box got a different verdict from frame to
frame (and when the work area was switched on/off): `needs_review` with `no_independent_depth_check` + `depth_no_top` /
`depth_top_ambiguous_unresolved`, or `cut_permission` `height_unavailable` because the `height_waived(support_hidden)`
waiver failed. Same input = same answer, a tiny data change = another answer.

**Root cause (measured on 15 frames of the unmoved box, 2 placements, work area on and off)**: the hidden-stand support
was re-estimated from every cloud. The far floor (1.5-4.4 m) is a *thick* noisy slab (depth MAD 7.3 mm, p95 18 mm), not a
thin plane; the seeded sequential RANSAC (7 mm band, 300 iterations, `max_planes` 6) cut it into 2-6 facets tilted 0-5 deg
(base z of the picked "floor" -0.77..-0.95 m, true -0.832) and `pick_hidden_support` took the largest member of the
farthest group without a refit. Everything keyed on that facet moved: the top-above-floor offset / `support_clearance_mm`
(815-956 mm for a box that did not move; std 31 mm), the floor-footprint test (`no_floor_support` -> `depth_no_top`, margins
-496..+91 mm), the 10-deg parallel test (`depth_ambiguous_top` when a clutter plane fell either side of it) and the trim-plane
/ independent top check (`estimate_table_plane`). A robust plane fit of the floor per frame, instead, agrees across all 15
frames to ~1 mm at the box (tilt 0.33-0.44 deg, top 857.3-858.8 mm above it). Second, independent source: the seeded RANSAC
start of the TOP plane is 0.3-2.4 deg off the true top; the 7 mm band then cuts the far corners out of the inliers and the
depth quad IoU against the (unchanged) final quad moved 0.87-0.97 with `ransac_seed` alone; below the waiver's 0.9 the
independent R5 check is not run, so the waiver failed.

**Fix**
- `live.fixed_floor` `{frame: base_link, normal, offset_m, tolerance_m, search_m, min_inliers, min_inlier_frac,
  max_median_offset_m}`: the floor is one measured plane in base_link (only with `depth_top.support: hidden_stand`,
  `SceneError` otherwise). Measure it with `python3 -m box_seam_detection.core.fixed_floor --scene-config S --calib-file C
  --z-hint -0.83 <capture manifests>` (robust trimmed fit per capture + pooled; prints the YAML). The detector converts it
  to the camera frame with the hand-eye transform (`depth_top.fixed_floor`) before SAM3 is touched. Used by
  `identify_table_top` (support plane + footprint from ALL floor points; the top candidate's offset is its centroid height
  above the plane, not a plane-offset difference), `pipeline.estimate_table_plane` (so `support_clearance_mm` /
  `support_depth_below_top_m`, trim-plane choice, R5, ROI hint). No second RANSAC for the table plane (-0.28 s).
- **Fail closed**: every cloud is checked against the plane (`fixed_floor.evaluate`): floor points within `tolerance_m`
  >= `min_inliers`, >= `min_inlier_frac` of the +-`search_m` band, |median signed distance| <= `max_median_offset_m`. A
  moved camera / another calibration fails it -> `depth_fixed_floor_mismatch(...)` (needs_review, no top), the support plane is
  unavailable (no `support_clearance_mm`, `box_profile.support_unavailable: fixed_floor_mismatch`, so the height waiver
  does not apply). No transform / camera below the floor -> exit 1, stage `fixed_floor` before SAM3. The plane is never
  trusted blindly; re-measure after any camera or calibration change. `search_m` stays inside the work area's dense floor band.
- `depth_top.top_refit_iters` (default 0 = off; `dual_arm_v1` 12): after the LO polish the top plane is refit on its own 7 mm
  band until the normal moves < `top_refit_tol_deg` (0.01 deg) (`_converge_plane_top`); same threshold, same `top_rms_max`
  gate afterwards. Start-independent: the IoU spread over 5 RANSAC seeds drops from 0.874-0.968 to 0.962-0.968.
- No gate threshold was changed (IoU 0.9, R5 0.85, edge limits, `parallel_tol_deg` 10, offset 0.30-1.50, footprint +-0.30 m).
- Tests: `test/core/test_fixed_floor.py` (validation, camera plane, cloud check, top check, support plane, mismatch, absent,
  refit), `test/live/test_fixed_floor_gate.py` (CLI wiring, fail closed before SAM3, off when absent).

## Top-face depth diagnostics and the gap-anchored top (2026-10-12; plan: docs/friend_detector_plan.md)

Recorded for EVERY detection under `record.top_face.depth_diag` (shadow; `AUTOLABEL_V2_TOP_DIAG_OFF=1` switches it off,
~0.2 s): per-edge depth support (`core/edge_support.py`: outward walk from each quad edge over the height above the top
plane, depth "cliff" offset, share of stations where the coplanar top continues, `unsupported` edges), multiple-box evidence
(`core/top_diag.py`: other box-sized coplanar components / an off-plane blob inside the top), a known-size fit score and a
height-band component summary (`core/known_size_fit.py`, `autolabel/depth_top.height_band_components`; never used).
Profile flags (`core/box_profiles.py`, only `large_392` has them on; small/big stay shadow = bit-identical):

- `edge_support_cut_rule`: cut rule `top_edge_unsupported(<edge>)` (retryable).
- `multi_box_reject`: reject `multiple_boxes_suspect(...)` with an operator hint (stage `multi_box`, before cut permission).
- `top_from_gap` (K1, `core/top_from_gap.py`): the published top is the gap-anchored known-size quad: edges parallel to the
  seam at observed gap +- across_seam/2 in a plane re-fitted on the central 60 % of the quad; seam-end edges keep their RGB
  position unless the length along the seam deviates > 5 mm from along_seam (then re-centred); declines (RGB quad kept,
  reason recorded) when the RGB quad does not already agree with the known size. `record.top_face.top_from_gap` holds BOTH
  quads (`rgb_quad`, `gap_anchored_quad`), per-edge `deltas_mm` (outward displacement, + = further out) and the re-fitted
  plane; `top_face.corners_*` / `plane` / `seam.endpoints_px` carry the published (gap-anchored) values. The detector's
  rectangle gate and the profile size check evaluate the MEASURED (RGB) quad of a K1 record, so K1 can never launder a
  quad those gates would reject. `AUTOLABEL_V2_TOP_FROM_GAP=shadow|off` computes-and-records / disables it.

## Truncation recovery (2026-10-13; scene key `truncation_recovery.mode: off | shadow | on`)

Failure: SAM3 outlines only ONE FLAP of the lid (back edge -> flap gap, ~399 x 162 mm). The half-top is coplanar, so the
depth cross-check passes; the profile resolves `generic` and the seam axis is not published (`seam_status_ambiguous`).
D1 (`edge_support.decide`, same thresholds as the `top_edge_unsupported` rule) sees it: the coplanar top continues past the front edge.

`pipeline._truncation_recovery` runs right after the top quad is accepted (before fold / profile / seam). Only when D1 flags an
edge: (a) the first SAM3 alternate of the same response whose quad contains the current one and passes the normal evaluation
(refine + depth check); (b) otherwise a depth-grown quad (`top_diag.grow_coplanar_region`: pixels within the depth-check band
of the top plane, closed over the gap, 8-connected component under the seed, opened, hole-filled, convex hull -> `_mask_to_quad_ex`),
which goes through the normal refine + depth check + fold path; (c) otherwise the old behaviour (reject). A candidate replaces the
detection only if: it contains >= 90 % of the original quad, D1 flags no edge, its dims resolve to a known profile (generic:
rectangle gate `generic_rect_tol_mm`), and the downstream stages (re-run through `process_capture(_forced=...)`) give seam status ok
(observed gap for `top_from_gap` profiles) with a clean D1 on the final quad. Provenance: `record.truncation_recovery`
(trigger edge / extent / continue share, before and after quads + dims, per-method `tried` results, `applied`).
`table_cell_v1` = `off` (the code path never runs: records are bit-identical); `AUTOLABEL_V2_TRUNC_RECOVERY=on|shadow|off` overrides.
