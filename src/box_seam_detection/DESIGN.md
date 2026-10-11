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
