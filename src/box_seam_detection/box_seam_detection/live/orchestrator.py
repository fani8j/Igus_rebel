# Provenance: moved into box_seam_detection (2026-10-12) from seam_handoff_20260908/scripts/capture_and_query_depth.py
# at commit 08d1041 (working tree identical to HEAD, unmodified). Only the --autolabel-v2 flow is kept; the legacy
# detection branches (predict_seam ResNet, depth-first, learned-depth-first, hybrid, guided, pcd-box, --seam-json,
# --pixel/--pixel-only) are dropped. The flow is exposed as run_detection(next_capture, cfg, progress); main(argv)
# is the CLI around it. Safety rules: DESIGN.md "live.orchestrator API".
"""box_seam_detection.live.orchestrator -- capture -> autolabel_v2 detector -> camera geometry -> base_link
planner snapshot (box_h_cut_snapshot/v1) -> side cutting midpoints -> side-tape seed.

``run_detection(next_capture, cfg, progress)`` is shared by the CLI (``main``, offline --from-image or the live
rgbd recorder) and the ROS node (``ros.detect_node``). Fail closed:

1. At the start of every non-dry run the snapshot at ``cfg.snapshot_path`` (and its side-seed quarantine file)
   is removed, so a failed / blocked detection never leaves an older fresh snapshot behind.
2. The snapshot is built in the pipeline directory and published (atomic same-directory replace) only when the
   detection is accepted AND (cut_permission allowed OR ``cfg.allow_cut_blocked``) AND the side seed passed;
   it carries the ``cut_permission`` block. A seed failure quarantines it to ``<snapshot>.side_seed_rejected``.
3. ``acquisition_stamp_ns`` = colour image stamp; ``side_tape_seed.source`` must match snapshot_id/stamp.
4. Exceptions raised by ``next_capture`` (e.g. a capture timeout) propagate to the caller (after the detector
   server is stopped); the snapshot was already removed at the start.
Everything else (rejections, prerequisite failures) is returned as a DetectResult, never raised or sys.exit()ed.
"""

import os as _blas_os  # noqa: E402  (BLAS pin must precede the first numpy import)
if _blas_os.environ.get("AUTOLABEL_V2_BLAS_THREADS", "").strip().lower() != "default":
    if "OPENBLAS_NUM_THREADS" not in _blas_os.environ:   # RANSAC gemv: 7-8x faster
        _blas_os.environ["OPENBLAS_NUM_THREADS"] = "1"
        _blas_os.environ["AUTOLABEL_V2_BLAS_PINNED"] = "1"  # ours, not the user's

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from .capture_staging import (link_or_copy as _link_or_copy, open_staging, resolve_staging_setting,
                              run_archive_kwargs, write_manifest_atomically as _write_manifest_atomically)
from .snapshot import (SIDE_CUTTING_POINT_NAMES, SIDE_SEED_REJECTED_SUFFIX,  # noqa: F401
                       _quarantine_planner_snapshot, _snapshot_xyz, _write_json_atomically,
                       add_side_cutting_points, remove_snapshot, side_seed_rejected_path,
                       stamp_detector_method, support_depth_below_top,
                       validate_planner_snapshot)

PACKAGE = __package__ or "box_seam_detection.live"
PACKAGE_ROOT = Path(__file__).resolve().parents[2]      # directory that holds the box_seam_detection package
DEFAULT_SAM3_PYTHON = os.environ.get("BOX_SEAM_SAM3_PYTHON",
                                     "/home/farhan/miniconda3/envs/book_grasping/bin/python")
DEFAULT_CALIB_FILE = "~/.ros2/easy_handeye2/calibrations/rebel_handeye_h_v2.calib"
DEFAULT_OUTPUT_ROOT = Path("/home/farhan/Igus_rebel/outputs/rgbd_captures")
DEFAULT_SNAPSHOT_PATH = Path("/home/farhan/Igus_rebel/outputs/seam_3d_base_link.json")
DETECTOR_METHOD = "autolabel_v2_sam3_fold_midline"
DEFAULT_AUTOLABEL_V2_ATTEMPTS = 4  # 1 capture + 3 retries
MAX_AUTOLABEL_V2_ATTEMPTS = 10
# R4 smart retry: a rectangle_inconsistent reject repeated "the same way" means a
# deterministic placement problem (live data 2026-10-03), so further identical
# recaptures only waste time.
AUTOLABEL_V2_RETRY_POLICIES = ("smart", "all")
DEFAULT_AUTOLABEL_V2_RETRY_POLICY = "smart"
RETRY_SAME_DIFF_TOL_MM = 3.0     # failing-pair diff must agree within this
RETRY_SAME_CORNER_PX = 5.0       # every top-face corner must agree within this
RETRY_SAME_SKEW_DEG = 1.0     # rectangle_skewed: same max corner deviation (deg)
RETRY_DEFAULT_RECT_TOL_MM = 3.0  # used when the reject JSON has no tol_mm
RETRY_SAME_TOP_SIZE_MM = 8.0     # top_size_out_of_range: long and short each agree within this
HINT_HIGH_YAW_DEG = 20.0         # yaw above this -> "rotate the box" hint
HINT_IMAGE_EDGE_MARGIN_PX = 40.0 # corner closer than this to the image border
DEFAULT_AUTOLABEL_V2_AVERAGE = 1  # live burst frames averaged per attempt (1 = single capture; averaging showed no gain live 2026-09-28)
MAX_AUTOLABEL_V2_AVERAGE = 15
DEFAULT_AVERAGE_MAX_DIFF = 6.0  # gray levels: per-frame mean |frame - median frame|
DEFAULT_AVERAGE_RATE_HZ = 5.0  # rgbd_recorder.py save_rate_hz during a burst
MAX_AVERAGE_RATE_HZ = 30.0
MAX_RECT_TOL_MM = 20.0
AUTOLABEL_V2_TITLE = "2/5 autolabel_v2 偵測（autolabel_v2_detect.py：SAM3 live + fold + 接縫估計）"
# (module-level so the server/step helpers below can use it)


# ----------------------------------------------------------------------------- API
@dataclass
class CaptureFiles:
    capture_id: str
    image_path: Path          # rgb png
    pcd_path: Path            # camera-frame XYZ(RGB) cloud registered to the colour camera
    manifest_path: Path       # same schema as captures/<id>.json (camera info, stamps, frames, extrinsic)
    stamp_ns: int             # acquisition stamp of the colour image


@dataclass
class DetectConfig:
    """Options of today's capture_and_query_depth.py --autolabel-v2 (CLI flag names in brackets)."""
    max_attempts: int = DEFAULT_AUTOLABEL_V2_ATTEMPTS   # [--autolabel-v2-attempts]; forced to 1 when not live
    retry_policy: str = "smart"                         # smart | all [--autolabel-v2-retry-policy]
    allow_needs_review: bool = False
    allow_cut_blocked: bool = False
    dry_run: bool = False       # run everything, never touch snapshot_path (result.snapshot still filled)
    rect_tol_mm: Optional[float] = None
    sam3_python: str = DEFAULT_SAM3_PYTHON              # [--autolabel-v2-sam3-python]
    sam3_daemon: str = "auto"                           # auto | on | off
    downstream_inprocess: bool = True
    calib_file: str = DEFAULT_CALIB_FILE
    driver_extrinsic: str = "recorded"                  # recorded | tf   (tf needs ROS: base_link step is a subprocess)
    target_frame: str = "base_link"
    side_seed: bool = True                              # [--no-side-seed]
    output_root: Path = DEFAULT_OUTPUT_ROOT             # detector-server workdir + fallback pipeline root
    snapshot_path: Optional[Path] = DEFAULT_SNAPSHOT_PATH   # None -> <pipeline_dir>/planner_snapshot.json only
    # --- further options of the CLI flow
    live: bool = True                  # False = offline replay of one existing capture (single attempt)
    autolabel_v2_server: str = "auto"  # auto (= live only) | on | off   [--autolabel-v2-server]
    no_rect_check: bool = False
    no_depth_edge: bool = False
    debug: bool = False
    tf_timeout: float = 5.0
    pipeline_root: Optional[Path] = None      # [--predict-out-dir] pipeline/<id>/<run>/ root override
    camera_geometry_path: Optional[Path] = None
    base_points_path: Optional[Path] = None
    detector_script: Optional[str] = None     # run this script instead of `-m <package>.live.detector` (tests)
    extra_query_args: tuple = ()
    average_frames: int = 1                   # provenance only: burst averaging happens inside next_capture
    average_max_diff: float = 6.0
    average_rate_hz: float = 5.0
    rgb_topic: str = "/camera/color/image_raw"
    pointcloud_topic: str = "/camera/depth_registered/points"
    # a manifest driver_extrinsic block (node captures) that differs from the recorded constant by
    # > 1 mm / 0.1 deg: refuse (default, fail closed) | warn (use the manifest, log) | use_manifest
    driver_extrinsic_policy: str = "refuse"
    # scene config (core/scene.py): YAML path or shipped name; None = table_cell_v1 (today's values).
    # It supplies the default calib file, the driver extrinsic reference, the seam-direction prior and
    # the depth-top / gate overrides of the detector.
    scene_config: Optional[str] = None
    # RAM staging session dir (live.capture_staging) the captures were written to, or None = captures live
    # directly in ``output_root``. pipeline/<capture>/<run>/ stays under output_root either way.
    capture_staging_dir: Optional[Path] = None

    def __post_init__(self):
        # "" / "." (Path("")) from a caller means "no snapshot path": never publish to Path('.')
        if self.snapshot_path is not None and str(self.snapshot_path).strip() in ("", "."):
            self.snapshot_path = None


@dataclass
class DetectResult:
    success: bool
    status: str                     # accepted | rejected | blocked | error | cancelled
    reason: str = ""
    reasons: list = field(default_factory=list)
    cut_permission: Optional[dict] = None
    snapshot: Optional[dict] = None
    snapshot_path: Optional[str] = None   # ONLY the file published at cfg.snapshot_path, else None
    capture_id: Optional[str] = None
    seam_status: Optional[str] = None
    attempts: list = field(default_factory=list)
    elapsed_s: float = 0.0
    pipeline_dir: Optional[str] = None
    # additive (not in DESIGN.md): CLI exit code and the contract overlay png of the accepted/last attempt
    exit_code: int = 0
    overlay_path: Optional[str] = None
    staged_snapshot_path: Optional[str] = None   # the finished snapshot in the pipeline dir (always, if built)
    # "allowed" | "overridden" (blocked cut accepted via allow_cut_blocked; cut_permission then has
    # status "blocked" + override "allow_cut_blocked") | "blocked" | "" (not evaluated)
    cut_summary: str = ""


class DetectionAbort(Exception):
    """Internal control flow of the run (replaces the old sys.exit): becomes a DetectResult."""

    def __init__(self, exit_code, reason="", status="error", reasons=None, cut_permission=None):
        super().__init__(reason or f"exit {exit_code}")
        self.exit_code = int(exit_code)
        self.reason = str(reason)
        self.status = status
        self.reasons = list(reasons or [])
        self.cut_permission = cut_permission


class CaptureRejected(Exception):
    """Raised by ``next_capture`` when the capture itself must be dropped before the detector runs
    (e.g. a non-static averaging burst). Recorded as a rejected attempt; retried like a retryable reject."""

    def __init__(self, capture_id, reasons, retryable=True, extra=None):
        super().__init__("; ".join(reasons))
        self.capture_id = capture_id
        self.reasons = list(reasons)
        self.retryable = bool(retryable)
        self.extra = dict(extra or {})


# _SAVED_*_RE: output of the recorder subprocess
# Paths may contain spaces (e.g. outputs on "/media/<user>/My Passport/..."): match up to
# the extension at the end of the log line (the PCD line continues with " (N 點...)").
_SAVED_IMAGE_RE = re.compile(r"已儲存影像:\s*(.+?\.png)\s*$", re.M)
_SAVED_PCD_RE = re.compile(r"已儲存點雲:\s*(.+?\.pcd)(?=\s+\(|\s*$)", re.M)
_SAVED_MANIFEST_RE = re.compile(r"已儲存 capture manifest:\s*(.+?\.json)\s*$", re.M)



def find_matching_pcd(image_path):
    """依 outputs/rgb/<stem>.png ↔ outputs/pointcloud/<stem>.pcd 的既有命名慣例，
    幫 --from-image 離線模式自動配對同一次拍攝的點雲。找不到回傳 None。

    跟 run_pick_pipeline.py 的 find_matching_pcd() 完全相同的邏輯，故意重複
    而不是 import 對方，維持這支腳本能單獨執行、不依賴 run_pick_pipeline.py。
    """
    img = Path(image_path).resolve()
    stem = img.stem
    for suffix in ("_tape_seam", "_auto_roi", "_seam"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    candidates = [
        img.parent.parent / "pointcloud" / f"{stem}.pcd",  # outputs/rgb/../pointcloud/
        img.parent / f"{stem}.pcd",                        # 跟影像同一個資料夾
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def find_capture_manifest(image_path, explicit_manifest=None):
    if explicit_manifest:
        manifest_path = Path(explicit_manifest).resolve()
    else:
        image = Path(image_path).resolve()
        manifest_path = image.parent.parent / "captures" / f"{image.stem}.json"
    if not manifest_path.is_file():
        raise ValueError(
            f"找不到影像的 capture manifest: {manifest_path}。"
            "請使用 rgbd_recorder.py 重新拍攝，或以 --capture-manifest 指定同一次 capture 的 manifest。")
    return manifest_path


def load_capture_manifest(manifest_path, image_path, pcd_path):
    try:
        with open(manifest_path, encoding="utf-8") as input_file:
            manifest = json.load(input_file)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"無法讀取 capture manifest {manifest_path}: {error}") from error
    if manifest.get("schema") != "rgbd_capture/v1":
        raise ValueError(f"不支援的 capture manifest schema: {manifest.get('schema')!r}")
    capture_id = manifest.get("capture_id")
    if not isinstance(capture_id, str) or not capture_id:
        raise ValueError("capture manifest 缺少 capture_id")
    image = manifest.get("image", {})
    pointcloud = manifest.get("pointcloud", {})
    camera_info = manifest.get("camera_info", {})
    if Path(image.get("path", "")).resolve() != Path(image_path).resolve():
        raise ValueError("指定影像不屬於 capture manifest")
    if Path(pointcloud.get("path", "")).resolve() != Path(pcd_path).resolve():
        raise ValueError("指定點雲不屬於 capture manifest")
    image_frame = image.get("header", {}).get("frame_id")
    cloud_frame = pointcloud.get("header", {}).get("frame_id")
    camera_frame = camera_info.get("header", {}).get("frame_id")
    if not image_frame or image_frame != cloud_frame or image_frame != camera_frame:
        raise ValueError("capture manifest 的 RGB、PointCloud2 與 CameraInfo frame 不一致")
    if image.get("width") != camera_info.get("width") or image.get("height") != camera_info.get("height"):
        raise ValueError("CameraInfo 尺寸與 RGB 影像不一致")
    if not isinstance(image.get("header", {}).get("stamp_ns"), int) or not isinstance(
            pointcloud.get("header", {}).get("stamp_ns"), int):
        raise ValueError("capture manifest 缺少原始 RGB 或 PointCloud2 timestamp")
    return manifest


def _read_manifest_paths(manifest_path):
    """不做完整驗證，只從 capture manifest 讀出 RGB/PCD 路徑（temporal 逐幀解析用）。"""
    try:
        with open(manifest_path, encoding="utf-8") as input_file:
            raw = json.load(input_file)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"無法讀取 capture manifest {manifest_path}: {error}") from error
    try:
        image_path = raw["image"]["path"]
        pcd_path = raw["pointcloud"]["path"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"capture manifest {manifest_path} 缺少影像或點雲路徑") from error
    return str(image_path), str(pcd_path)


def _load_pcd_sequence_record(manifest_path):
    """載入並驗證一組 temporal capture，回傳 manifest/路徑 record（anchor 選擇用）。"""
    manifest_path = Path(manifest_path)
    image_path, pcd_path = _read_manifest_paths(manifest_path)
    manifest = load_capture_manifest(manifest_path, image_path, pcd_path)
    return {
        "manifest_path": manifest_path,
        "manifest": manifest,
        "image_path": str(Path(image_path).resolve()),
        "pcd_path": str(Path(pcd_path).resolve()),
    }


def _parse_pcd_sequence_manifests(recorder_out, expected_frames):
    """收集 rgbd_recorder.py 本次輸出的全部 capture manifest，要求恰好 N 組同步 capture。"""
    manifest_paths = _SAVED_MANIFEST_RE.findall(recorder_out)
    if len(manifest_paths) != expected_frames:
        raise ValueError(
            f"rgbd_recorder.py 提交 {len(manifest_paths)} 組 capture manifest，"
            f"與要求的 {expected_frames} 組不一致，拒絕使用不完整 temporal capture")
    return [Path(path) for path in manifest_paths]

def detection_output_directory(output_dir, capture_id):
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + f"_{os.getpid()}"
    directory = Path(output_dir) / "pipeline" / capture_id / run_id
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def validate_detection(seam_json_path, image_path, manifest):
    try:
        with open(seam_json_path, encoding="utf-8") as input_file:
            detection = json.load(input_file)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"無法讀取偵測 JSON {seam_json_path}: {error}") from error
    required_points = {"back_left", "back_right", "front_left", "front_right", "left_mid", "right_mid"}
    missing = sorted(required_points.difference(detection))
    if missing or detection.get("incomplete_groups"):
        raise ValueError(f"偵測 JSON 不完整，缺少 {missing}，incomplete_groups={detection.get('incomplete_groups')}")
    if detection.get("lr_ambiguous"):
        raise ValueError("偵測 JSON 的 left/right 命名不明確，拒絕規劃")
    if Path(detection.get("source_image", "")).resolve() != Path(image_path).resolve():
        raise ValueError("偵測 JSON 的 source_image 與本次 capture 不一致")
    if detection.get("capture_id") is not None and detection["capture_id"] != manifest.get("capture_id"):
        raise ValueError("偵測 JSON 的 capture_id 與本次 capture 不一致（拒用舊檔）")
    expected_size = [manifest["image"]["width"], manifest["image"]["height"]]
    if detection.get("image_size") != expected_size:
        raise ValueError(f"偵測 JSON 尺寸 {detection.get('image_size')} 不符 capture {expected_size}")
    return detection

def side_seed_profile_args(seam_json_path):
    """Extra side_face_seed.py flags from the detector's canonical seam JSON
    (autolabel_v2.box_profile.name when its source is the record / measured_height_mm). [] when absent/unreadable (the seed then
    classifies by today's rule)."""
    if seam_json_path is None:
        return []
    try:
        with open(seam_json_path, encoding="utf-8") as handle:
            block = (json.load(handle).get("autolabel_v2") or {})
    except (OSError, json.JSONDecodeError, AttributeError):
        return []
    extra = []
    profile = block.get("box_profile")
    name = profile.get("name") if isinstance(profile, dict) else None
    # Only the pipeline's own record profile is handed over; a profile the
    # detector merely guessed from the corners (source corners_legacy) is left
    # to the seed's identical legacy classification.
    from ..core import box_profiles  # noqa: PLC0415
    if name in box_profiles.PROFILES and profile.get("source") == "record":
        extra += ["--box-profile", name]
    height = block.get("measured_height_mm")
    if isinstance(height, (int, float)) and not isinstance(height, bool) \
            and math.isfinite(height) and height > 0:
        extra += ["--measured-height-mm", repr(float(height))]
    return extra


# ----------------------------------------------------------------------------- steps
def _subprocess_env():
    """Environment of the package's own `python -m` subprocesses: the package root first on PYTHONPATH
    (no sys.path manipulation anywhere)."""
    env = dict(os.environ)
    paths = [str(PACKAGE_ROOT)] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(paths))
    return env


def module_command(name, argv):
    """[python, -m, <package>.live.<name>, *argv] -- one of this package's CLIs in its own process."""
    return [sys.executable, "-m", f"{PACKAGE}.{name}", *[str(a) for a in argv]]


def preflight_calibration(cfg):
    """Prerequisite check (before acquisition): the calib that the base_link step will use.

    Verifies that PyYAML imports and that the effective calib file exists and parses as an eye_on_base
    calibration with robot_base_frame == target frame (and tracking frame camera_link when the driver
    extrinsic is ``recorded``). Raises ValueError with a clear message; returns the parsed calibration.
    """
    try:
        import yaml  # noqa: F401,PLC0415 — needed by live.base_link
    except ImportError as error:
        raise ValueError(f"{sys.executable} 無法 import PyYAML（base_link 讀取 .calib 需要）: {error}") from error
    from . import base_link as base_tf  # noqa: PLC0415
    calib_file = Path(cfg.calib_file or DEFAULT_CALIB_FILE).expanduser()
    if not calib_file.is_file():
        raise ValueError(f"找不到手眼標定檔: {calib_file}")
    try:
        calibration = base_tf.load_handeye_calibration(calib_file)
    except (OSError, ValueError) as error:
        raise ValueError(f"手眼標定檔無效: {error}") from error
    if calibration["robot_base_frame"] != cfg.target_frame:
        raise ValueError(f"手眼標定檔 {calib_file} robot_base_frame "
                         f"{calibration['robot_base_frame']!r} != --target-frame {cfg.target_frame!r}")
    expected_tracking = base_tf.RECORDED_DRIVER_EXTRINSIC["parent_frame"]
    if getattr(cfg, "scene_config", None):
        from ..core import scene as _scene  # noqa: PLC0415
        try:
            _scene.load_scene(cfg.scene_config)
        except _scene.SceneError as error:
            raise ValueError(f"scene config: {error}") from error
    if (getattr(cfg, "driver_extrinsic", None) == "recorded"
            and calibration["tracking_base_frame"] != expected_tracking):
        raise ValueError(f"手眼標定檔 {calib_file} tracking_base_frame "
                         f"{calibration['tracking_base_frame']!r} != {expected_tracking!r}"
                         "（--driver-extrinsic recorded 需要）")
    return calibration


def run_step(title, cmd, capture=False, check=True):
    print(f"\n{'=' * 60}\n[{title}]\n{'=' * 60}")
    print(f"$ {' '.join(cmd)}\n")
    env = _subprocess_env()
    if capture:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        print(result.stdout)
    else:
        result = subprocess.run(cmd, env=env)
    if check and result.returncode != 0:
        print(f"[{title}] 失敗（exit code {result.returncode}），中止", file=sys.stderr)
        raise DetectionAbort(result.returncode, f"{title}: exit code {result.returncode}")
    if capture:
        return result.stdout
    # check=False: the caller decides what the exit code means.
    return None if check else result.returncode


def _downstream_inprocess_enabled(cfg):
    """DetectConfig.downstream_inprocess (bool). A missing attribute (old callers/tests) = off."""
    value = getattr(cfg, "downstream_inprocess", False)
    return value is True or value == "on"


def _run_script_main_inprocess(script_name, argv):
    """Run ``<package>.live.<script_name>.main(argv)`` in THIS interpreter and return the exit
    code the same module would have produced as a ``python -m`` subprocess.

    SystemExit(code) -> code (None = 0; a non-int is printed to stderr, exit 1,
    exactly like the interpreter does); any other exception -> traceback on
    stderr and exit 1 (the interpreter's uncaught-exception code); a returned
    int is the exit code (side_seed.main), None = 0. KeyboardInterrupt is
    not swallowed.
    """
    import importlib  # noqa: PLC0415
    import traceback  # noqa: PLC0415
    try:
        module = importlib.import_module(f"{PACKAGE}.{script_name}")
        returned = module.main(list(argv))
        code = 0 if returned is None else returned
    except SystemExit as error:
        code = error.code
        if code is None:
            code = 0
        elif not isinstance(code, int):
            print(code, file=sys.stderr)
            code = 1
    except Exception:  # noqa: BLE001 — a subprocess would die with a traceback, exit 1
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    return code


def run_script_step(cfg, title, script_name, argv, check=True, inprocess_ok=True):
    """run_step() for one of the downstream modules (query_depth, base_link, side_seed), in-process
    when enabled. The in-process path runs the same argv through the module's main(); output files,
    stdout/stderr and exit codes are those of the subprocess path."""
    cmd = module_command(script_name, argv)
    if not (inprocess_ok and _downstream_inprocess_enabled(cfg)):
        return run_step(title, cmd, check=check)
    print(f"\n{'=' * 60}\n[{title}]\n{'=' * 60}")
    print(f"$ {' '.join(cmd)}  [in-process]\n")
    returncode = _run_script_main_inprocess(script_name, argv)
    if check and returncode != 0:
        print(f"[{title}] 失敗（exit code {returncode}），中止", file=sys.stderr)
        raise DetectionAbort(returncode, f"{title}: exit code {returncode}")
    return None if check else returncode


def run_side_face_seed(snapshot_path, cfg, seam_json_path=None, quarantine_path=None):
    """Step 6: add side_tape_seed/v1.1 to the planner snapshot (``snapshot_path``, the staged copy).

    side_face_seed validates first and only then rewrites the snapshot atomically. FAIL CLOSED: if
    the step fails (non-zero exit, cannot be launched, or the seed's source block does not match the
    snapshot id / acquisition stamp) the staged snapshot is renamed to ``quarantine_path`` (default
    <snapshot>.side_seed_rejected; kept for diagnosis, replacing any older one) so that no valid-looking
    box_h_cut_snapshot/v1 remains at the planner path, and the run aborts with the step's code
    (1 = geometry/input rejected, 2 = could not run).
    """
    argv = ["--planner-json", str(snapshot_path), *side_seed_profile_args(seam_json_path)]
    title = "6/6 側面膠帶種子：側面平面 + 膠帶中心線（side_face_seed.py）"
    print(f"\n{'=' * 60}\n[{title}]\n{'=' * 60}")
    cmd = module_command("side_seed", argv)
    print(f"$ {' '.join(cmd)}\n")
    try:
        if _downstream_inprocess_enabled(cfg):
            print("[in-process]")
            returncode = _run_script_main_inprocess("side_seed", argv)
        else:
            result = subprocess.run(cmd, env=_subprocess_env())
            returncode = result.returncode
        reason = f"exit code {returncode}"
    except OSError as error:
        returncode = 2
        reason = f"無法執行: {error}"
    if returncode == 0:
        try:
            with open(snapshot_path, encoding="utf-8") as handle:
                snapshot = json.load(handle)
            source = (snapshot.get("side_tape_seed") or {}).get("source") or {}
            if (source.get("snapshot_id") != snapshot.get("snapshot_id")
                    or source.get("acquisition_stamp_ns") != snapshot.get("acquisition_stamp_ns")
                    or source.get("capture_id") != snapshot.get("capture_id")):
                returncode, reason = 1, "side_tape_seed.source does not match the snapshot (id/stamp)"
        except (OSError, ValueError, AttributeError) as error:
            returncode, reason = 1, f"cannot re-read the seeded snapshot: {error}"
    if returncode != 0:
        rejected = _quarantine_planner_snapshot(snapshot_path, quarantine_path)
        where = f"已移至 {rejected}" if rejected is not None else "已移除"
        print(f"[{title}] 失敗（{reason}），中止；planner snapshot 不可用於切割，"
              f"{where}（{snapshot_path} 不存在）", file=sys.stderr)
        raise DetectionAbort(returncode, f"side seed failed ({reason}); snapshot quarantined")


RECORDER_EXIT_TIMEOUT = 3  # rgbd_recorder: no synced RGB-D pair within --timeout-sec


def _run_recorder_step(title, cmd):
    """run_step(capture=True) for a recorder acquisition; a recorder timeout (exit 3: camera not
    publishing, silent topics listed in its output) becomes an exit 2 (prerequisite failure, no retry)."""
    try:
        return run_step(title, cmd, capture=True)
    except DetectionAbort as stop:
        if stop.exit_code == RECORDER_EXIT_TIMEOUT:
            print(f"[{title}] rgbd_recorder.py 逾時（exit {RECORDER_EXIT_TIMEOUT}）：相機沒有發布同步的 "
                  f"RGB-D 影像/點雲（沒有訊息的 topic 見上方 recorder 輸出）。前置條件失敗，"
                  f"不重試，中止（exit 2）", file=sys.stderr)
            raise DetectionAbort(2, "recorder timeout: camera not publishing synced RGB-D") from stop
        raise


def recorder_command(output_dir, cfg, extra_params=(), once=True):
    cmd = [sys.executable, "-m", f"{PACKAGE}.recorder", str(output_dir)]
    if once:
        cmd.append("--once")
    cmd += ["--ros-args", *extra_params,
            "-p", f"rgb_topic:={cfg.rgb_topic}", "-p", f"pointcloud_topic:={cfg.pointcloud_topic}"]
    return cmd


def acquire_live_capture(cfg, output_dir, title="1/5 拍照（rgbd_recorder.py --once）"):
    """One live RGB-D capture via the recorder (--once).

    Returns (image_path, pcd_path, manifest_path, manifest) after the
    manifest/image/pcd binding checks; aborts (exit 1) on an incomplete capture.
    """
    recorder_out = _run_recorder_step(title, recorder_command(output_dir, cfg))
    image_match = _SAVED_IMAGE_RE.search(recorder_out)
    pcd_match = _SAVED_PCD_RE.search(recorder_out)
    manifest_match = _SAVED_MANIFEST_RE.search(recorder_out)
    if not image_match or not pcd_match or not manifest_match:
        print("rgbd_recorder.py 沒有提交完整 capture manifest，拒絕使用不完整 capture", file=sys.stderr)
        raise DetectionAbort(1, "recorder committed no complete capture manifest")
    image_path = str(Path(image_match.group(1)).resolve())
    pcd_path = str(Path(pcd_match.group(1)).resolve())
    manifest_path = Path(manifest_match.group(1)).resolve()
    try:
        manifest = load_capture_manifest(manifest_path, image_path, pcd_path)
    except ValueError as error:
        print(f"capture manifest 驗證失敗: {error}", file=sys.stderr)
        raise DetectionAbort(1, f"capture manifest validation failed: {error}") from error
    return image_path, pcd_path, manifest_path, manifest


AVERAGING_METHOD = "rgb_mean_pcd_middle"


def _camera_info_signature(camera_info):
    """CameraInfo fields that must be identical across a burst (stamp excluded)."""
    camera_info = camera_info or {}
    signature = {key: camera_info.get(key) for key in
                 ("width", "height", "distortion_model", "d", "k", "r", "p")}
    signature["frame_id"] = (camera_info.get("header") or {}).get("frame_id")
    return signature


def check_burst_consistency(records):
    """All frames of a burst: same RGB size/encoding/frame_id and identical CameraInfo.

    Raises ValueError (driver/hardware inconsistency, not a scene problem).
    """
    if not records:
        raise ValueError("burst 沒有任何 capture")
    reference = records[0]["manifest"]
    ref_image = reference["image"]
    ref_cloud = reference["pointcloud"]
    ref_camera = _camera_info_signature(reference.get("camera_info"))
    for record in records[1:]:
        manifest = record["manifest"]
        image = manifest["image"]
        cloud = manifest["pointcloud"]
        cid = manifest["capture_id"]
        if (image.get("width"), image.get("height"), image.get("encoding")) != (
                ref_image.get("width"), ref_image.get("height"), ref_image.get("encoding")):
            raise ValueError(f"burst capture {cid} 的 RGB 尺寸/encoding 與第一幀不一致")
        if (image["header"]["frame_id"] != ref_image["header"]["frame_id"]
                or cloud["header"]["frame_id"] != ref_cloud["header"]["frame_id"]):
            raise ValueError(f"burst capture {cid} 的 frame_id 與第一幀不一致")
        if _camera_info_signature(manifest.get("camera_info")) != ref_camera:
            raise ValueError(f"burst capture {cid} 的 CameraInfo 與第一幀不一致")
    ids = [record["manifest"]["capture_id"] for record in records]
    if len(set(ids)) != len(ids):
        raise ValueError(f"burst 內有重複的 capture_id: {ids}")


def average_rgb_frames(frames):
    """Per-pixel mean of N same-shape uint8 frames + a static-scene measure.

    Returns (mean_uint8, frame_diffs): the mean is accumulated in float64 and
    rounded with np.rint to uint8; frame_diffs[i] = mean |frame_i - median
    frame| over all pixels and channels (gray levels).
    """
    import numpy as np  # noqa: PLC0415 — only needed for live averaging

    if not frames:
        raise ValueError("沒有影像可平均")
    shape = frames[0].shape
    for frame in frames:
        if frame.shape != shape or frame.dtype != np.uint8:
            raise ValueError(f"影像尺寸/型別不一致: {frame.shape}/{frame.dtype} vs {shape}/uint8")
    total = np.zeros(shape, dtype=np.float64)
    for frame in frames:
        total += frame
    mean = np.clip(np.rint(total / len(frames)), 0, 255).astype(np.uint8)
    median = np.median(np.stack(frames, axis=0), axis=0).astype(np.float32)
    frame_diffs = [float(np.mean(np.abs(frame.astype(np.float32) - median))) for frame in frames]
    return mean, frame_diffs


def _atomic_write_png(path, image):
    import cv2  # noqa: PLC0415

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.stem}.tmp-{os.getpid()}.png")
    try:
        if not cv2.imwrite(str(temporary_path), image):
            raise OSError(f"cv2.imwrite failed for {path}")
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def build_averaged_capture(records, max_diff, rgb_dir=None, pcd_dir=None, manifest_dir=None):
    """Average a burst of N synced captures into ONE new rgbd_capture/v1 capture.

    RGB = per-pixel mean of the N frames (new PNG); point cloud = the MIDDLE
    frame's PCD (unorganized clouds cannot be averaged point-wise; the plane
    fit already averages depth noise), hard-linked/copied under the new id;
    manifest = the middle frame's manifest (camera_info, stamps, cloud block)
    with the new capture id/paths plus an additive "averaging" block. The new
    id is ``<middle capture id>_avg<N>``. Directories default to the middle
    frame's rgb/, pointcloud/ and captures/ directories.

    Returns {"status": "ok", "image_path", "pcd_path", "manifest_path",
    "manifest", "averaging"}, or {"status": "not_static", "averaging"} (nothing
    written) when some frame differs from the median frame by more than
    ``max_diff`` gray levels. Raises ValueError on inconsistent or unreadable
    frames.
    """
    import copy  # noqa: PLC0415
    import time  # noqa: PLC0415

    import cv2  # noqa: PLC0415

    check_burst_consistency(records)
    n_frames = len(records)
    middle = records[n_frames // 2]
    middle_manifest = middle["manifest"]
    width = int(middle_manifest["image"]["width"])
    height = int(middle_manifest["image"]["height"])
    frames = []
    for record in records:
        frame = cv2.imread(str(record["image_path"]), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"無法讀取 burst 影像 {record['image_path']}")
        if [frame.shape[1], frame.shape[0]] != [width, height]:
            raise ValueError(f"burst 影像 {record['image_path']} 實際尺寸 "
                             f"{[frame.shape[1], frame.shape[0]]} 與 manifest {[width, height]} 不一致")
        frames.append(frame)
    mean, frame_diffs = average_rgb_frames(frames)
    averaging = {
        "n_frames": n_frames,
        "source_capture_ids": [record["manifest"]["capture_id"] for record in records],
        "middle_capture_id": middle_manifest["capture_id"],
        "method": AVERAGING_METHOD,
        "max_frame_diff": round(max(frame_diffs), 4),
        "frame_diffs": [round(value, 4) for value in frame_diffs],
        "max_diff_threshold": float(max_diff),
    }
    if max(frame_diffs) > max_diff:
        return {"status": "not_static", "averaging": averaging}

    capture_id = f"{middle_manifest['capture_id']}_avg{n_frames}"
    rgb_dir = Path(rgb_dir) if rgb_dir else Path(middle["image_path"]).parent
    pcd_dir = Path(pcd_dir) if pcd_dir else Path(middle["pcd_path"]).parent
    manifest_dir = Path(manifest_dir) if manifest_dir else Path(middle["manifest_path"]).parent
    image_path = (rgb_dir / f"{capture_id}.png").resolve()
    pcd_path = (pcd_dir / f"{capture_id}.pcd").resolve()
    manifest_path = (manifest_dir / f"{capture_id}.json").resolve()
    _atomic_write_png(image_path, mean)
    _link_or_copy(middle["pcd_path"], pcd_path)
    manifest = copy.deepcopy(middle_manifest)
    manifest["capture_id"] = capture_id
    manifest["created_at_ns"] = time.time_ns()
    manifest["image"]["path"] = str(image_path)
    manifest["pointcloud"]["path"] = str(pcd_path)
    manifest["averaging"] = dict(averaging, source_manifests=[
        str(record["manifest_path"]) for record in records])
    _write_manifest_atomically(manifest_path, manifest)   # tmp + fsync + replace, 0o644 like rgbd_recorder.py
    manifest = load_capture_manifest(manifest_path, str(image_path), str(pcd_path))
    return {"status": "ok", "image_path": str(image_path), "pcd_path": str(pcd_path),
            "manifest_path": manifest_path, "manifest": manifest, "averaging": averaging}


def acquire_averaged_live_capture(cfg, output_dir, n_frames, max_diff, title):
    """Live burst: recorder max_frames:=N -> exactly N manifests -> averaged capture.

    Returns build_averaged_capture()'s dict (status "ok" or "not_static");
    aborts (exit 1) on an incomplete or inconsistent burst (fail closed, like an
    incomplete single capture).
    """
    recorder_out = _run_recorder_step(
        title,
        recorder_command(output_dir, cfg, once=False, extra_params=(
            "-p", f"max_frames:={int(n_frames)}", "-p", f"save_rate_hz:={float(cfg.average_rate_hz)}")))
    try:
        manifest_paths = _parse_pcd_sequence_manifests(recorder_out, int(n_frames))
        records = [_load_pcd_sequence_record(path) for path in manifest_paths]
        result = build_averaged_capture(records, max_diff)
    except (OSError, ValueError) as error:
        print(f"burst capture 平均失敗: {error}", file=sys.stderr)
        raise DetectionAbort(1, f"burst capture averaging failed: {error}") from error
    averaging = result["averaging"]
    print(f"→ burst {averaging['n_frames']} 幀: {', '.join(averaging['source_capture_ids'])}")
    print(f"→ 各幀與中位數影像的平均絕對差（灰階）: {averaging['frame_diffs']}"
          f"（上限 {max_diff:g}）")
    if result["status"] == "ok":
        print(f"→ 平均 capture: {result['manifest']['capture_id']}（RGB 平均 {n_frames} 幀，"
              f"點雲取中間幀 {averaging['middle_capture_id']}）\n"
              f"→ 影像: {result['image_path']}\n→ 點雲: {result['pcd_path']}\n"
              f"→ capture manifest: {result['manifest_path']}")
    return result


def pipeline_output_paths(cfg, manifest_path, manifest, output_dir):
    """Fresh per-run pipeline directory for this capture + downstream output paths.

    Returns (pipeline_dir, camera_geometry_path, base_points_path, staged_planner_snapshot_path).
    The planner snapshot is built at <pipeline_dir>/planner_snapshot.json; run_detection publishes it
    to ``cfg.snapshot_path`` only when everything passed.
    """
    staging_dir = getattr(cfg, "capture_staging_dir", None)
    staged = staging_dir is not None and Path(staging_dir).resolve() in Path(manifest_path).resolve().parents
    pipeline_root = Path(cfg.pipeline_root).resolve() if cfg.pipeline_root else (
        manifest_path.parent.parent if manifest_path.parent.name == "captures" and not staged else output_dir)
    pipeline_dir = detection_output_directory(pipeline_root, manifest["capture_id"])
    camera_geometry_path = Path(cfg.camera_geometry_path).resolve() if cfg.camera_geometry_path else (
        pipeline_dir / "camera_geometry.json")
    base_points_path = Path(cfg.base_points_path).resolve() if cfg.base_points_path else (
        pipeline_dir / "base_geometry.json")
    return pipeline_dir, camera_geometry_path, base_points_path, pipeline_dir / "planner_snapshot.json"


# ------------------------------------------------------------------ autolabel_v2 detector server
# Live --autolabel-v2: autolabel_v2_detect.py --serve is started BEFORE the
# first acquisition so the SAM3 worker loads while the camera captures, and
# every attempt (retries included) is sent to that one server. Each request
# runs the exact one-shot detector code (same files, same exit code). If the
# server cannot be spawned, dies or times out, the attempt falls back to the
# one-shot subprocess (logged). A server that reports startup_error (SAM3 /
# autolabel_v2 unavailable) stops the run with exit 2 before detection.
AUTOLABEL_V2_SERVER_READY_TIMEOUT_S = 900.0     # SAM3 load (Sam3Worker handshake limit)
AUTOLABEL_V2_SERVER_REQUEST_TIMEOUT_S = 900.0   # one request after ready (SAM3 query <= 300 s)
AUTOLABEL_V2_SERVER_SHUTDOWN_TIMEOUT_S = 75.0   # Sam3Worker.close waits up to 60 s
AUTOLABEL_V2_SERVER_ABORT_GRACE_S = 2.0         # abort path: SIGTERM -> SIGKILL delay
_AUTOLABEL_V2_SERVER = None


class AutolabelV2ServerError(RuntimeError):
    """The detector server is unusable (spawn failure, died, timed out)."""


class AutolabelV2ServerStartupError(RuntimeError):
    """The detector server reported startup_error (prerequisite failure)."""


class AutolabelV2Server:
    """Client of ``<package>.live.detector --serve`` (JSON lines on stdin/stdout).

    The server runs in its own process group so a forced kill also takes its
    SAM3 worker down; its stderr is inherited (detector messages land in this
    run's log). A reader thread queues the stdout lines.
    """

    def __init__(self, cmd, popen=None):
        import queue  # noqa: PLC0415
        import threading  # noqa: PLC0415

        self.cmd = [str(c) for c in cmd]
        self._lines = queue.Queue()
        self.status = None          # None | "ready" | "startup_error"
        self.startup = None         # the status message
        self.dead_reason = None
        self._next_id = 0
        self._closed = False
        try:
            self.proc = (popen or subprocess.Popen)(
                self.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                text=True, bufsize=1, start_new_session=True, env=_subprocess_env())
        except (OSError, ValueError) as error:
            raise AutolabelV2ServerError(f"cannot start detector server: {error}") from error

        def _pump(stream, lines):
            try:
                for line in stream:
                    lines.put(line)
            except (OSError, ValueError):
                pass
            lines.put(None)  # EOF

        self._reader = threading.Thread(target=_pump, args=(self.proc.stdout, self._lines),
                                        daemon=True)
        self._reader.start()

    @property
    def usable(self):
        return (not self._closed and self.dead_reason is None
                and self.status != "startup_error")

    def _next_message(self, deadline):
        """Next JSON object from the server; raises AutolabelV2ServerError on
        EOF (server died) or when ``deadline`` (time.monotonic) passes."""
        import queue  # noqa: PLC0415
        import time  # noqa: PLC0415

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._mark_dead("timed out waiting for the detector server")
                self.kill()
                raise AutolabelV2ServerError(self.dead_reason)
            try:
                line = self._lines.get(timeout=min(remaining, 1.0))
            except queue.Empty:
                continue
            if line is None:
                code = self.proc.poll()
                if code is None:
                    try:
                        code = self.proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        code = "unknown"
                self._mark_dead(f"detector server exited (code {code})")
                raise AutolabelV2ServerError(self.dead_reason)
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                print(f"[autolabel_v2 server] {line}", file=sys.stderr)
                continue
            if isinstance(message, dict):
                return message

    def _mark_dead(self, reason):
        if self.dead_reason is None:
            self.dead_reason = reason

    def wait_ready(self, timeout=None):
        """Block until the status line; returns it. Raises
        AutolabelV2ServerStartupError on startup_error, AutolabelV2ServerError
        when the server died / timed out first."""
        import time  # noqa: PLC0415

        if self.status is None:
            if self.dead_reason is not None:
                raise AutolabelV2ServerError(self.dead_reason)
            if timeout is None:
                timeout = AUTOLABEL_V2_SERVER_READY_TIMEOUT_S
            deadline = time.monotonic() + float(timeout)
            while self.status is None:
                message = self._next_message(deadline)
                if message.get("type") in ("ready", "startup_error"):
                    self.status = message["type"]
                    self.startup = message
        if self.status == "startup_error":
            raise AutolabelV2ServerStartupError(
                str(self.startup.get("reason") or "unknown startup failure"))
        return self.startup

    def request(self, argv, timeout=None, ready_timeout=None):
        """Run one detector request (one-shot CLI ``argv``); returns the response."""
        import time  # noqa: PLC0415

        if self._closed:
            raise AutolabelV2ServerError("detector server already shut down")
        self.wait_ready(ready_timeout)
        self._next_id += 1
        request_id = self._next_id
        try:
            self.proc.stdin.write(json.dumps({"id": request_id, "argv": [str(a) for a in argv]})
                                  + "\n")
            self.proc.stdin.flush()
        except (OSError, ValueError) as error:
            self._mark_dead(f"cannot send request to the detector server: {error}")
            raise AutolabelV2ServerError(self.dead_reason) from error
        if timeout is None:
            timeout = AUTOLABEL_V2_SERVER_REQUEST_TIMEOUT_S
        deadline = time.monotonic() + float(timeout)
        while True:
            message = self._next_message(deadline)
            if message.get("type") == "response" and message.get("id") == request_id:
                if not isinstance(message.get("exit_code"), int):
                    self._mark_dead(f"detector server response without exit code: {message}")
                    self.kill()
                    raise AutolabelV2ServerError(self.dead_reason)
                return message

    def begin_shutdown(self):
        """Ask the server to exit (closes its stdin); does not wait."""
        if self._closed:
            return
        self._closed = True
        try:
            if self.proc.stdin is not None:
                self.proc.stdin.close()
        except (OSError, ValueError):
            pass

    def kill(self):
        """Kill the server's whole process group (server + SAM3 worker)."""
        import signal  # noqa: PLC0415

        self._closed = True
        try:
            os.killpg(self.proc.pid, signal.SIGKILL)
        except (OSError, AttributeError):
            try:
                self.proc.kill()
            except OSError:
                pass
        try:
            self.proc.wait(timeout=10)
        except (subprocess.TimeoutExpired, OSError):
            pass

    def terminate_now(self, grace=None):
        """Abort path (Ctrl-C / crash): SIGTERM the server's whole process
        group, wait ``grace`` s, then SIGKILL it. Does not wait for an
        in-flight request. SIGINT is ignored while it runs (main thread) so a
        second Ctrl-C cannot interrupt the kill and leave the server behind."""
        import signal  # noqa: PLC0415

        if grace is None:
            grace = AUTOLABEL_V2_SERVER_ABORT_GRACE_S
        previous = None
        try:
            previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
        except (ValueError, OSError, TypeError):   # not the main thread
            previous = None
        try:
            self._closed = True
            if self.proc.poll() is None:
                try:
                    os.killpg(self.proc.pid, signal.SIGTERM)
                except (OSError, AttributeError):
                    try:
                        self.proc.terminate()
                    except OSError:
                        pass
                try:
                    self.proc.wait(timeout=grace)
                except (subprocess.TimeoutExpired, OSError):
                    pass
            # SIGKILL the group even when the leader already exited: a SAM3
            # worker that ignored SIGTERM must not survive it.
            self.kill()
            for stream in (self.proc.stdin, self.proc.stdout):
                try:
                    if stream is not None:
                        stream.close()
                except (OSError, ValueError):
                    pass
        finally:
            if previous is not None:
                try:
                    signal.signal(signal.SIGINT, previous)
                except (ValueError, OSError, TypeError):
                    pass

    def close(self, timeout=None):
        """Graceful stop (stdin EOF -> server closes its SAM3 worker), then a
        process-group kill if it has not exited within ``timeout``. A Ctrl-C
        while waiting switches to ``terminate_now`` and re-raises."""
        if timeout is None:
            timeout = AUTOLABEL_V2_SERVER_SHUTDOWN_TIMEOUT_S
        self.begin_shutdown()
        try:
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            print(f"autolabel_v2 偵測 server 未在 {timeout:g} s 內結束，強制終止", file=sys.stderr)
            self.kill()
        except OSError:
            self.kill()
        except KeyboardInterrupt:
            self.terminate_now()
            raise
        try:
            if self.proc.stdout is not None:
                self.proc.stdout.close()
        except (OSError, ValueError):
            pass


def autolabel_v2_server_wanted(args):
    """Server mode for this run: auto = live only (offline stays one-shot)."""
    mode = getattr(args, "autolabel_v2_server", "auto") or "auto"
    if mode == "auto":
        return bool(getattr(args, "live", True))
    return mode == "on"


def sam3_daemon_mode(args, environ=None):
    """AUTOLABEL_V2_SAM3_DAEMON value for the detector: explicit --sam3-daemon
    on/off wins; auto = an already-set environment value, else auto for live
    runs and off offline (like the detector server)."""
    environ = os.environ if environ is None else environ
    mode = getattr(args, "sam3_daemon", "auto") or "auto"
    if mode != "auto":
        return mode
    if environ.get("AUTOLABEL_V2_SAM3_DAEMON"):
        return environ["AUTOLABEL_V2_SAM3_DAEMON"]
    return "auto" if getattr(args, "live", True) else "off"


def detector_base_cmd(cfg):
    """[python, -m, <package>.live.detector] (or [python, script] for cfg.detector_script)."""
    script = getattr(cfg, "detector_script", None)
    if script:
        return [sys.executable, str(Path(script).resolve())]
    return [sys.executable, "-m", f"{PACKAGE}.detector"]


def autolabel_v2_server_cmd(args, server_dir):
    return [*detector_base_cmd(args), "--serve", "--work-dir", str(server_dir),
            "--sam3-python", str(args.sam3_python)]


def start_autolabel_v2_server(args, output_dir):
    """Spawn the detector server (SAM3 starts loading at once). Returns the
    client, or None (logged) when it cannot be spawned."""
    global _AUTOLABEL_V2_SERVER
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + f"_{os.getpid()}"
    server_dir = Path(output_dir) / "autolabel_v2_server" / run_id
    cmd = autolabel_v2_server_cmd(args, server_dir)
    print(f"→ autolabel_v2 偵測 server 啟動（SAM3 於拍照期間載入）: {' '.join(cmd)}")
    try:
        _AUTOLABEL_V2_SERVER = AutolabelV2Server(cmd)
    except AutolabelV2ServerError as error:
        print(f"autolabel_v2 偵測 server 無法啟動，改用單次子行程: {error}", file=sys.stderr)
        _AUTOLABEL_V2_SERVER = None
    return _AUTOLABEL_V2_SERVER


def stop_autolabel_v2_server(wait=True):
    """Stop the detector server (idempotent). ``wait=False`` only signals it
    to exit (frees the GPU while later steps run); the final call waits and
    kills on timeout."""
    global _AUTOLABEL_V2_SERVER
    server = _AUTOLABEL_V2_SERVER
    if server is None:
        return
    if not wait:
        server.begin_shutdown()
        return
    _AUTOLABEL_V2_SERVER = None
    server.close()


def abort_autolabel_v2_server():
    """Abort-path stop (Ctrl-C / unexpected exception): kill the detector
    server's process group at once instead of waiting for its in-flight
    request (which would otherwise keep writing into the aborted run)."""
    global _AUTOLABEL_V2_SERVER
    server = _AUTOLABEL_V2_SERVER
    if server is None:
        return
    # Cleared only after the kill: if a second Ctrl-C lands before
    # terminate_now ignores SIGINT, main()'s finally still stops the server.
    server.terminate_now()
    _AUTOLABEL_V2_SERVER = None
    print("autolabel_v2 偵測 server：執行中止，已立即終止其行程群組", file=sys.stderr)


def _autolabel_v2_via_server(title, cmd, argv=None):
    """Run one detector command on the live detector server.

    Returns its exit code, or None when there is no usable server (the caller
    then runs the one-shot subprocess ``cmd``). A server that dies / times out
    is killed and dropped (logged; this and later attempts go one-shot). A
    server startup_error (SAM3 / autolabel_v2 unavailable) exits 2 here,
    before any downstream step.
    """
    global _AUTOLABEL_V2_SERVER
    server = _AUTOLABEL_V2_SERVER
    if server is None or not server.usable:
        return None
    print(f"\n{'=' * 60}\n[{title}]\n{'=' * 60}")
    print(f"$ [autolabel_v2 server pid {server.proc.pid}] {' '.join(cmd)}\n", flush=True)
    try:
        response = server.request(argv if argv is not None else cmd[2:])
    except AutolabelV2ServerStartupError as error:
        print(f"[{title}] autolabel_v2 偵測 server 啟動失敗（SAM3/autolabel_v2 不可用）: {error}；"
              f"中止（exit 2）", file=sys.stderr)
        stop_autolabel_v2_server()
        raise DetectionAbort(2, f"detector server startup failed: {error}") from error
    except AutolabelV2ServerError as error:
        print(f"autolabel_v2 偵測 server 無法使用（{error}），改用單次子行程重跑此次偵測",
              file=sys.stderr)
        _AUTOLABEL_V2_SERVER = None
        server.kill()
        return None
    timing = response.get("timing") or {}
    print(f"→ autolabel_v2 server: exit {response['exit_code']}，"
          f"耗時 {response.get('elapsed_s')} s（SAM3 等待 {timing.get('sam3_load')} s，"
          f"推論 {timing.get('sam3_inference')} s）")
    return int(response["exit_code"])

# ------------------------------------------------------------ R4 smart retry helpers
_RING_NAMES = ("back_left", "back_right", "front_right", "front_left")


def _finite_vec(value, size):
    try:
        vec = [float(c) for c in value]
    except (TypeError, ValueError):
        return None
    if len(vec) != size or not all(math.isfinite(c) for c in vec):
        return None
    return vec


def _corner_set(top, key, size):
    block = top.get(key) if isinstance(top, dict) else None
    if not isinstance(block, dict):
        return None
    points = {}
    for name in _RING_NAMES:
        vec = _finite_vec(block.get(name), size)
        if vec is None:
            return None
        points[name] = vec
    return points


def load_autolabel_v2_record_top_face(pipeline_dir, capture_id):
    """top_face block of <pipeline_dir>/<id>_autolabel_v2_record.json, or None."""
    path = Path(pipeline_dir) / f"{capture_id}_autolabel_v2_record.json"
    try:
        with open(path, encoding="utf-8") as handle:
            record = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    top = record.get("top_face") if isinstance(record, dict) else None
    return top if isinstance(top, dict) else None


def rectangle_failure_signature(reject, top=None):
    """Describe a rectangle_inconsistent / rectangle_skewed reject, or None for any other reject.

    {"pairs": sorted failing pair names, "diffs": {pair: diff_mm},
     "signs": {pair: +1/-1 (first side longer)}, "corners_px": {name: [x, y]}|None}
    """
    if not isinstance(reject, dict):
        return None
    rect = reject.get("rectangle")
    if not isinstance(rect, dict) or not rect.get("diffs_mm"):
        return None
    reasons = [str(r) for r in reject.get("reasons") or []]
    if any(r.startswith("rectangle_skewed") for r in reasons):
        try:
            dev = float(rect["max_corner_dev_deg"])
        except (KeyError, TypeError, ValueError):
            return None
        return {"pairs": ["skew"], "diffs": {"skew": dev}, "signs": {"skew": 0},
                "corners_px": _corner_set(top, "corners_px", 2)}
    if not any(r.startswith("rectangle_inconsistent") for r in reasons):
        return None
    try:
        tol = float(rect.get("tol_mm", RETRY_DEFAULT_RECT_TOL_MM))
        diffs = {k: float(rect["diffs_mm"][k]) for k in ("A", "B")}
    except (KeyError, TypeError, ValueError):
        return None
    lengths = rect.get("lengths_mm") or {}
    sides = {"A": ("back", "front"), "B": ("left", "right")}
    failing = sorted(k for k, v in diffs.items() if v > tol)
    if not failing:
        return None
    signs = {}
    for key in failing:
        try:
            first, second = (float(lengths[n]) for n in sides[key])
            signs[key] = 1 if first >= second else -1
        except (KeyError, TypeError, ValueError):
            signs[key] = 0
    return {"pairs": failing, "diffs": {k: diffs[k] for k in failing}, "signs": signs,
            "corners_px": _corner_set(top, "corners_px", 2)}


def cut_permission_signature(reject):
    """Sorted cut_permission reason codes of a stage-cut_permission reject (None otherwise)."""
    if not isinstance(reject, dict) or reject.get("stage") != "cut_permission":
        return None
    block = reject.get("cut_permission")
    codes = sorted({str(r.get("code")) for r in (block or {}).get("reasons") or []
                    if isinstance(r, dict)})
    return codes or None


def top_size_signature(reject):
    """{"long_mm", "short_mm"} of a stage-top_size ``top_size_out_of_range`` reject, else None."""
    if not isinstance(reject, dict) or reject.get("stage") != "top_size":
        return None
    if not any(str(r).startswith("top_size_out_of_range") for r in reject.get("reasons") or []):
        return None
    check = (reject.get("details") or {}).get("top_size_check")
    try:
        return {"long_mm": float(check["long_mm"]), "short_mm": float(check["short_mm"])}
    except (KeyError, TypeError, ValueError):
        return None


def same_top_size_failure(first, second, tol_mm=RETRY_SAME_TOP_SIZE_MM):
    """True when two top-size signatures describe the same (deterministic) wrong-size top."""
    if not first or not second:
        return False
    return (abs(first["long_mm"] - second["long_mm"]) <= tol_mm
            and abs(first["short_mm"] - second["short_mm"]) <= tol_mm)


def same_rectangle_failure(first, second, diff_tol_mm=RETRY_SAME_DIFF_TOL_MM,
                           corner_tol_px=RETRY_SAME_CORNER_PX):
    """True when two failure signatures describe the same deterministic failure."""
    if not first or not second or first["pairs"] != second["pairs"]:
        return False
    for key in first["pairs"]:
        tol = RETRY_SAME_SKEW_DEG if key == "skew" else diff_tol_mm
        if abs(first["diffs"][key] - second["diffs"][key]) > tol:
            return False
        if first["signs"][key] and second["signs"][key] and first["signs"][key] != second["signs"][key]:
            return False
    c1, c2 = first["corners_px"], second["corners_px"]
    if c1 is not None and c2 is not None:
        if any(math.dist(c1[n], c2[n]) > corner_tol_px for n in _RING_NAMES):
            return False
    return True


def estimate_top_face_yaw_deg(top):
    """Box yaw (deg, folded to 0..45) of the long edge vs the camera x axis.

    3-D: corners_xyz_camera_m, long edge and camera x both projected onto the
    top plane (record plane, else the corner-fitted normal). 2-D fallback:
    corners_px. Returns None when no usable corners.
    """
    def fold(angle):
        angle = math.degrees(angle) % 90.0
        return min(angle, 90.0 - angle)

    xyz = _corner_set(top, "corners_xyz_camera_m", 3)
    if xyz is not None:
        def sub(a, b):
            return [a[i] - b[i] for i in range(3)]

        def dot(a, b):
            return sum(a[i] * b[i] for i in range(3))

        def cross(a, b):
            return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
                    a[0] * b[1] - a[1] * b[0]]

        def unit(v):
            n = math.sqrt(dot(v, v))
            return [c / n for c in v] if n > 1e-9 else None

        u = sub(xyz["back_right"], xyz["back_left"])          # A direction
        u2 = sub(xyz["front_right"], xyz["front_left"])
        v = sub(xyz["front_left"], xyz["back_left"])          # B direction
        v2 = sub(xyz["front_right"], xyz["back_right"])
        a_dir = [u[i] + u2[i] for i in range(3)]
        b_dir = [v[i] + v2[i] for i in range(3)]
        normal = None
        plane = top.get("plane")
        if isinstance(plane, dict):
            vec = _finite_vec(plane.get("normal"), 3)
            normal = unit(vec) if vec else None
        if normal is None:
            normal = unit(cross(a_dir, b_dir))
        long_dir = a_dir if dot(a_dir, a_dir) >= dot(b_dir, b_dir) else b_dir
        if normal is not None:
            def project(vec):
                k = dot(vec, normal)
                return [vec[i] - k * normal[i] for i in range(3)]
            long_p, cam_x = unit(project(long_dir)), unit(project([1.0, 0.0, 0.0]))
            if long_p is not None and cam_x is not None:
                cos_a = max(-1.0, min(1.0, dot(long_p, cam_x)))
                return fold(math.acos(abs(cos_a)))
    px = _corner_set(top, "corners_px", 2)
    if px is not None:
        a_dir = [px["back_right"][i] - px["back_left"][i] + px["front_right"][i] - px["front_left"][i]
                 for i in range(2)]
        b_dir = [px["front_left"][i] - px["back_left"][i] + px["front_right"][i] - px["back_right"][i]
                 for i in range(2)]
        long_dir = a_dir if math.hypot(*a_dir) >= math.hypot(*b_dir) else b_dir
        if math.hypot(*long_dir) > 1e-9:
            return fold(math.atan2(long_dir[1], long_dir[0]))
    return None


def near_image_edge(top, width, height, margin_px=HINT_IMAGE_EDGE_MARGIN_PX):
    px = _corner_set(top, "corners_px", 2)
    if px is None or not width or not height:
        return False
    return any(x < margin_px or y < margin_px or x > width - margin_px or y > height - margin_px
               for x, y in px.values())


def build_operator_hint(reject, top, consistent, image_size=None):
    """Plain-language operator advice from the final rejected attempt's evidence.

    ``consistent``: the rectangle failure repeated identically (deterministic).
    Returns a list of hint strings (possibly empty).
    """
    hints = []
    sig = rectangle_failure_signature(reject, top)
    yaw = estimate_top_face_yaw_deg(top) if top else None
    high_yaw = yaw is not None and yaw > HINT_HIGH_YAW_DEG
    edge = bool(top) and image_size is not None and near_image_edge(top, *image_size)
    if sig is not None and high_yaw:
        hints.append(f"box rotated ~{yaw:.0f} deg relative to the camera - rotate it closer to "
                     f"square (edges parallel to the image), or move it toward the table centre")
    if edge:
        hints.append("box near the image/table edge - move it toward the table centre")
    if sig is not None and consistent and not high_yaw:
        worst = max(sig["diffs"].values())
        if sig["pairs"] == ["skew"]:
            hints.append(f"box damaged/irregular top (corners {worst:.0f} deg off square "
                         f"consistently)")
        else:
            hints.append(f"box damaged/irregular top (opposite sides differ {worst:.0f} mm "
                         f"consistently)")
    size_sig = top_size_signature(reject)
    if size_sig is not None:
        hints.append(f"detected top is {size_sig['long_mm']:.0f} x {size_sig['short_mm']:.0f} mm, outside "
                     f"the expected box size: only part of the top (e.g. one flap of a cut/drooping "
                     f"carton) was found, or the box is not one of the cell's boxes")
    detector_hint = (reject or {}).get("operator_hint")
    if isinstance(detector_hint, dict) and detector_hint.get("suggestion"):
        hints.append(f"detector hint: {detector_hint['suggestion']}")
    return hints


def run_autolabel_v2_attempt(cfg, image_path, pcd_path, manifest_path, manifest,
                             pipeline_dir, attempt=1, max_attempts=1):
    """Run the autolabel_v2 detector once on one capture (outputs in its own pipeline_dir).

    Returns {"returncode", "title", "seam_json", "overlay_dir", "overlay_png",
    "reject"} where "reject" is the detector's reject JSON for THIS capture
    (capture_id checked) or None.
    """
    seam_json = pipeline_dir / f"{Path(image_path).stem}_autolabel_v2_seam.json"
    # 完整 v5 record、SAM3 worker log 寫入本次 pipeline 目錄；疊圖（contract
    # 名稱 + rule 名稱）與 reject JSON 寫入 autolabel_v2_overlays/。
    overlay_dir = pipeline_dir / "autolabel_v2_overlays"
    overlay_png = overlay_dir / f"{manifest['capture_id']}_autolabel_v2.png"
    print(f"→ autolabel_v2 疊圖目錄: {overlay_dir}")
    argv = [
        "--image", image_path, "--pcd", pcd_path,
        "--capture-manifest", str(manifest_path),
        "--out-seam-json", str(seam_json),
        "--work-dir", str(pipeline_dir),
        "--overlay-dir", str(overlay_dir),
        "--sam3-python", str(cfg.sam3_python),
        "--calib-file", str(Path(cfg.calib_file or DEFAULT_CALIB_FILE).expanduser()),
    ]
    if getattr(cfg, "scene_config", None):
        argv += ["--scene-config", str(cfg.scene_config)]
    if cfg.allow_needs_review:
        argv.append("--allow-needs-review")
    if getattr(cfg, "allow_cut_blocked", False):
        argv.append("--allow-cut-blocked")
    if getattr(cfg, "rect_tol_mm", None) is not None:
        argv += ["--rect-tol-mm", str(cfg.rect_tol_mm)]
    if getattr(cfg, "no_rect_check", False):
        argv.append("--no-rect-check")
    if getattr(cfg, "no_depth_edge", False):
        argv.append("--no-depth-edge")
    if cfg.debug:
        argv.append("--debug")
    cmd = [*detector_base_cmd(cfg), *argv]
    title = AUTOLABEL_V2_TITLE
    if max_attempts > 1:
        title += f"（第 {attempt}/{max_attempts} 次）"
    # 1 = 場景拒絕（無檔，reject JSON 在疊圖目錄），2 = 前置條件失敗。
    returncode = _autolabel_v2_via_server(title, cmd, argv)
    if returncode is None:
        returncode = run_step(title, cmd, check=False)
    if returncode is None:
        # A run_step that reports no code only returns on success; the seam
        # JSON existence/validation checks below still fail closed.
        returncode = 0
    reject = None
    reject_path = overlay_dir / f"{manifest['capture_id']}_autolabel_v2_reject.json"
    if returncode != 0 and reject_path.is_file():
        try:
            with open(reject_path, encoding="utf-8") as handle:
                candidate = json.load(handle)
            if isinstance(candidate, dict) and candidate.get("capture_id") == manifest["capture_id"]:
                reject = candidate
        except (OSError, json.JSONDecodeError) as error:
            print(f"無法讀取 autolabel_v2 reject JSON {reject_path}: {error}", file=sys.stderr)
    return {"returncode": int(returncode), "title": title, "seam_json": seam_json,
            "overlay_dir": overlay_dir, "overlay_png": overlay_png, "reject": reject}


# ----------------------------------------------------------------------------- run_detection
def _validate_cfg(cfg):
    if not 1 <= int(cfg.max_attempts) <= MAX_AUTOLABEL_V2_ATTEMPTS:
        raise DetectionAbort(2, f"max_attempts must be in 1..{MAX_AUTOLABEL_V2_ATTEMPTS}")
    if cfg.retry_policy not in AUTOLABEL_V2_RETRY_POLICIES:
        raise DetectionAbort(2, f"retry_policy must be one of {AUTOLABEL_V2_RETRY_POLICIES}")
    if cfg.rect_tol_mm is not None and not (
            math.isfinite(cfg.rect_tol_mm) and 0.0 < cfg.rect_tol_mm <= MAX_RECT_TOL_MM):
        raise DetectionAbort(2, f"rect_tol_mm must be in (0, {MAX_RECT_TOL_MM:g}]")
    if cfg.rect_tol_mm is not None and cfg.no_rect_check:
        raise DetectionAbort(2, "rect_tol_mm and no_rect_check cannot be combined")
    if cfg.sam3_daemon not in ("auto", "on", "off") or cfg.autolabel_v2_server not in ("auto", "on", "off"):
        raise DetectionAbort(2, "sam3_daemon / autolabel_v2_server must be auto|on|off")
    if cfg.driver_extrinsic not in ("recorded", "tf"):
        raise DetectionAbort(2, "driver_extrinsic must be recorded|tf")
    if cfg.driver_extrinsic_policy not in ("refuse", "warn", "use_manifest"):
        raise DetectionAbort(2, "driver_extrinsic_policy must be refuse|warn|use_manifest")


def _manifest_averaging(manifest):
    block = manifest.get("averaging") if isinstance(manifest, dict) else None
    if not isinstance(block, dict):
        return None
    return {k: v for k, v in block.items() if k != "source_manifests"}


def _load_json_object(path):
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def run_detection(next_capture: Callable[[int], CaptureFiles], cfg: DetectConfig,
                  progress: Optional[Callable[[str, int, str], None]] = None,
                  *, abort_check: Optional[Callable[[], bool]] = None) -> DetectResult:
    """Capture (via ``next_capture(attempt)``) -> detect -> base_link snapshot (see the module docstring).

    Returns a DetectResult; never sys.exit()s. Exceptions raised by ``next_capture`` propagate (the
    detector server is stopped first; the snapshot was already removed unless dry_run).

    ``abort_check`` (cancel / shutdown state of the caller): polled before every attempt and immediately
    before the snapshot is published; True -> status "cancelled", nothing is published (the snapshot
    removed at the start stays removed).
    """
    t0 = time.monotonic()
    ctx = {"attempts": [], "pipeline_dir": None, "capture_id": None, "overlay": None,
           "capture_exc": None, "last_reject": None}
    try:
        result = _run_detection(next_capture, cfg, progress, ctx, abort_check)
    except DetectionAbort as abort:
        reject = ctx.get("last_reject") or {}
        result = DetectResult(
            success=False, status=abort.status, reason=abort.reason,
            reasons=abort.reasons or ([abort.reason] if abort.reason else []),
            cut_permission=abort.cut_permission, capture_id=ctx["capture_id"],
            attempts=list(ctx["attempts"]), pipeline_dir=ctx["pipeline_dir"],
            exit_code=abort.exit_code,
            cut_summary=("blocked" if abort.status == "blocked" else ""),
            overlay_path=(str(ctx["overlay"]) if ctx["overlay"] is not None
                          and Path(ctx["overlay"]).is_file() else None))
        if result.cut_permission is None and isinstance(reject.get("cut_permission"), dict):
            result.cut_permission = reject["cut_permission"]
    except BaseException as error:
        abort_autolabel_v2_server()
        if error is ctx["capture_exc"] or not isinstance(error, Exception):
            raise                      # next_capture's own exception (or Ctrl-C): the caller decides
        import traceback  # noqa: PLC0415
        traceback.print_exc()
        result = DetectResult(
            success=False, status="error", reason=f"internal error: {type(error).__name__}: {error}",
            reasons=[f"internal error: {type(error).__name__}: {error}"], capture_id=ctx["capture_id"],
            attempts=list(ctx["attempts"]), pipeline_dir=ctx["pipeline_dir"], exit_code=1)
    finally:
        stop_autolabel_v2_server()
    result.elapsed_s = round(time.monotonic() - t0, 3)
    return result


def _run_detection(next_capture, cfg, progress, ctx, abort_check=None):
    def say(stage, attempt, message=""):
        if progress is not None:
            try:
                progress(stage, attempt, message)
            except Exception:  # noqa: BLE001 — a broken progress sink must not break the run
                pass

    _validate_cfg(cfg)
    def cancelled():
        if abort_check is None:
            return False
        try:
            return bool(abort_check())
        except Exception:  # noqa: BLE001 — a broken hook must fail closed
            return True

    final_target = (Path(cfg.snapshot_path).expanduser().resolve()
                    if cfg.snapshot_path is not None and str(cfg.snapshot_path).strip() not in ("", ".")
                    else None)
    publish = final_target is not None and not cfg.dry_run
    if publish:
        # Safety rule 1: nothing older than this run may stay at the planner path.
        error = remove_snapshot(final_target)
        if error is not None:
            raise DetectionAbort(2, f"cannot remove the stale snapshot {final_target}: {error}")

    # SAM3 daemon mode is inherited by every detector process (server, one-shot, probes) through the env.
    os.environ["AUTOLABEL_V2_SAM3_DAEMON"] = sam3_daemon_mode(cfg)
    autolabel_v2_env = dict(os.environ)
    autolabel_v2_env["HF_HUB_OFFLINE"] = "1"
    autolabel_v2_env["TRANSFORMERS_OFFLINE"] = "1"
    if cfg.detector_script and not Path(cfg.detector_script).is_file():
        raise DetectionAbort(2, f"找不到 autolabel_v2_detect.py: {cfg.detector_script}")
    if not Path(cfg.sam3_python).is_file():
        raise DetectionAbort(2, f"找不到 SAM3 直譯器: {cfg.sam3_python}")
    sam3_python = Path(cfg.sam3_python)
    if not sam3_python.is_file():
        raise DetectionAbort(2, f"找不到 SAM3 直譯器: {sam3_python}")

    def _autolabel_v2_probe(label, probe_cmd):
        try:
            probe = subprocess.run(probe_cmd, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True,
                                   timeout=180.0, env=autolabel_v2_env)
        except (OSError, subprocess.SubprocessError) as error:
            raise DetectionAbort(2, f"{label} 探測失敗: {error}") from error
        if probe.returncode != 0:
            raise DetectionAbort(2, f"{label} 缺少可用環境: {(probe.stdout or '')[-2000:]}")

    # 拍照前先驗證依賴（exit 2，不浪費 live acquisition）：本直譯器可匯入 cv2/numpy/核心函式庫；
    # SAM3 直譯器可匯入 torch/transformers Sam3Model（只 import，不載入權重；HF 離線）。
    try:
        import cv2  # noqa: F401,PLC0415
        import numpy  # noqa: F401,PLC0415
        from ..core import pipeline as _core_pipeline, sam3_session as _core_sam3  # noqa: F401,PLC0415
    except Exception as error:  # noqa: BLE001
        raise DetectionAbort(2, f"autolabel_v2 偵測直譯器缺少可用環境（cv2/numpy/core）: "
                                f"{type(error).__name__}: {error}") from error
    sam3_probe = ("SAM3 直譯器（torch/transformers Sam3Model）",
                  [str(cfg.sam3_python), "-c", "import torch; from transformers import Sam3Model, Sam3Processor"])
    output_dir = Path(cfg.output_root).expanduser().resolve()
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        print(f"無法建立輸出資料夾 {output_dir}: {error}", file=sys.stderr)
        raise DetectionAbort(2, f"cannot create the output dir {output_dir}: {error}") from error
    # 偵測 server（live 預設）：拍照前啟動，SAM3 在拍照期間載入；server 自己載入 SAM3 並回報
    # ready/startup_error，所以省略 SAM3 import 探測。server 無法 spawn 時照舊探測並用單次子行程。
    server_wanted = autolabel_v2_server_wanted(cfg)
    if not server_wanted:
        _autolabel_v2_probe(*sam3_probe)
    else:
        if start_autolabel_v2_server(cfg, output_dir) is None:
            _autolabel_v2_probe(*sam3_probe)
    try:
        calibration = preflight_calibration(cfg)
    except ValueError as error:
        raise DetectionAbort(2, str(error)) from error
    print(f"→ 手眼標定預檢通過: {calibration['path']} "
          f"({calibration['robot_base_frame']} <- {calibration['tracking_base_frame']})")

    # Live: a deterministic retryable reject (e.g. rectangle_inconsistent) drops the capture (its files stay
    # in its own pipeline dir for debugging) and takes a FRESH one, up to N attempts in total. Offline: one run.
    live_mode = bool(cfg.live)
    max_attempts = int(cfg.max_attempts) if live_mode else 1
    attempts = ctx["attempts"]
    used_capture_ids = set()
    used_images = set()
    attempt = 0
    retry_policy = cfg.retry_policy or DEFAULT_AUTOLABEL_V2_RETRY_POLICY
    failure_sigs = {}   # attempt number -> rectangle failure signature
    cut_sigs = {}       # attempt number -> sorted cut_permission reason codes
    size_sigs = {}      # attempt number -> top_size_out_of_range signature
    last_evidence = {"reject": None, "top": None, "size": None, "consistent": False}

    def _stop_all_rejected(early=False):
        summary = "; ".join(f"#{e['attempt']} {e['capture_id']}: {', '.join(e['reasons'])}"
                            f" [{e.get('decision', '-')}]" for e in attempts)
        if early:
            head = (f"autolabel_v2 第 {attempt}/{max_attempts} 次以相同方式失敗（確定性的擺放問題，"
                    f"提早停止，retry-policy=smart）")
        else:
            head = f"autolabel_v2 {attempt}/{max_attempts} 次嘗試全部被拒絕"
        print(f"{head}，安全中止，不查深度、不沿用舊檔。各次原因: {summary}", file=sys.stderr)
        hints = build_operator_hint(last_evidence["reject"], last_evidence["top"],
                                    last_evidence["consistent"], last_evidence["size"]) \
            if last_evidence["reject"] else []
        for hint_text in hints:
            print(f"操作員提示 / operator hint: {hint_text}", file=sys.stderr)
        reject = last_evidence["reject"] or {}
        blocked = reject.get("stage") == "cut_permission"
        raise DetectionAbort(1, f"{head}: {summary}", status="blocked" if blocked else "rejected",
                             reasons=list(attempts[-1]["reasons"]) if attempts else [],
                             cut_permission=reject.get("cut_permission"))

    def _capture(n):
        try:
            return next_capture(n)
        except CaptureRejected:
            raise
        except BaseException as error:
            ctx["capture_exc"] = error
            raise

    while True:
        attempt += 1
        if cancelled():
            raise DetectionAbort(1, f"cancelled before attempt {attempt}", status="cancelled")
        say("capture", attempt, "")
        try:
            files = _capture(attempt)
        except CaptureRejected as rejected:
            used_capture_ids.update(rejected.extra.get("source_capture_ids") or [])
            entry = {"attempt": attempt, "capture_id": rejected.capture_id, "exit_code": None,
                     "pipeline_dir": None, "outcome": "rejected", "reasons": rejected.reasons,
                     "retryable": rejected.retryable,
                     **{k: v for k, v in rejected.extra.items() if k != "source_capture_ids"}}
            attempts.append(entry)
            ctx["capture_id"] = rejected.capture_id
            if rejected.retryable and live_mode and attempt < max_attempts:
                print(f"autolabel_v2 第 {attempt}/{max_attempts} 次 capture 被拒絕（可重試）: "
                      f"{'; '.join(rejected.reasons)}；不跑偵測器，重新拍照", file=sys.stderr)
                continue
            _stop_all_rejected()
        image_path = str(Path(files.image_path).resolve())
        pcd_path = str(Path(files.pcd_path).resolve())
        manifest_path = Path(files.manifest_path).resolve()
        try:
            manifest = load_capture_manifest(manifest_path, image_path, pcd_path)
        except ValueError as error:
            print(f"capture manifest 驗證失敗: {error}", file=sys.stderr)
            raise DetectionAbort(1, f"capture manifest validation failed: {error}") from error
        if int(files.stamp_ns) != int(manifest["image"]["header"]["stamp_ns"]):
            raise DetectionAbort(1, f"CaptureFiles.stamp_ns {files.stamp_ns} != manifest colour image stamp "
                                    f"{manifest['image']['header']['stamp_ns']} (acquisition stamp must be the "
                                    f"colour image stamp)")
        if manifest["capture_id"] != files.capture_id:
            raise DetectionAbort(1, f"CaptureFiles.capture_id {files.capture_id!r} != manifest "
                                    f"{manifest['capture_id']!r}")
        ctx["capture_id"] = manifest["capture_id"]
        if attempt > 1 or live_mode:
            if manifest["capture_id"] in used_capture_ids or image_path in used_images:
                print(f"rgbd_recorder.py 沒有產生新的 capture（{manifest['capture_id']} "
                      f"已在先前嘗試使用），安全中止", file=sys.stderr)
                raise DetectionAbort(1, f"capture {manifest['capture_id']} was already used by an earlier attempt")
        averaging = _manifest_averaging(manifest)
        (pipeline_dir, camera_geometry_path, base_points_path,
         staged_snapshot_path) = pipeline_output_paths(cfg, manifest_path, manifest, output_dir)
        ctx["pipeline_dir"] = str(pipeline_dir)
        used_capture_ids.add(manifest["capture_id"])
        used_images.add(image_path)
        say("detect", attempt, f"running the detector on {manifest['capture_id']}")
        result = run_autolabel_v2_attempt(cfg, image_path, pcd_path, manifest_path,
                                          manifest, pipeline_dir, attempt, max_attempts)
        ctx["overlay"] = result["overlay_png"]
        returncode = result["returncode"]
        entry = {"attempt": attempt, "capture_id": manifest["capture_id"],
                 "exit_code": returncode, "pipeline_dir": str(pipeline_dir)}
        if averaging is not None:
            entry["averaging"] = averaging
        if returncode == 0:
            entry.update({"outcome": "accepted", "reasons": [], "retryable": False})
            attempts.append(entry)
            break
        if returncode != 1:
            entry.update({"outcome": "failed", "reasons": [f"exit code {returncode}"],
                          "retryable": False})
            attempts.append(entry)
            print(f"[{result['title']}] 失敗（exit code {returncode}），中止", file=sys.stderr)
            raise DetectionAbort(returncode, f"detector failed (exit code {returncode})")
        reject = result["reject"] or {}
        ctx["last_reject"] = reject
        reasons = [str(r) for r in (reject.get("reasons") or [])] or [
            "no reject JSON for this capture"]
        retryable = reject.get("retryable") is True
        hint = reject.get("operator_hint")
        if isinstance(hint, dict) and hint.get("suggestion"):
            print(f"操作員提示: {hint['suggestion']}", file=sys.stderr)
        entry.update({"outcome": "rejected", "reasons": reasons, "retryable": retryable})
        top_face = load_autolabel_v2_record_top_face(pipeline_dir, manifest["capture_id"])
        sig = rectangle_failure_signature(reject, top_face)
        if sig is not None:
            failure_sigs[attempt] = sig
            entry["failure_signature"] = {
                "pairs": sig["pairs"], "diffs_mm": sig["diffs"],
                "has_corners_px": sig["corners_px"] is not None}
        same_as_prev = (sig is not None and failure_sigs.get(attempt - 1) is not None
                        and same_rectangle_failure(failure_sigs[attempt - 1], sig))
        cut_sig = cut_permission_signature(reject)
        if cut_sig is not None:
            cut_sigs[attempt] = cut_sig
            entry["failure_signature"] = {"cut_permission_codes": cut_sig}
            same_as_prev = same_as_prev or cut_sigs.get(attempt - 1) == cut_sig
        size_sig = top_size_signature(reject)
        if size_sig is not None:
            size_sigs[attempt] = size_sig
            entry["failure_signature"] = {"top_size_mm": size_sig}
            same_as_prev = same_as_prev or same_top_size_failure(size_sigs.get(attempt - 1), size_sig)
        image_size = None
        try:
            image_size = (int(manifest["image"]["width"]), int(manifest["image"]["height"]))
        except (KeyError, TypeError, ValueError):
            pass
        last_evidence.update({"reject": reject, "top": top_face, "size": image_size,
                              "consistent": same_as_prev})
        entry["same_as_previous"] = bool(same_as_prev)
        stop_early = (retryable and live_mode and retry_policy == "smart" and same_as_prev)
        if stop_early:
            entry["decision"] = "stop_early_same_failure"
        elif retryable and live_mode and attempt < max_attempts:
            entry["decision"] = "retry"
        else:
            entry["decision"] = "stop"
        attempts.append(entry)
        if stop_early:
            _stop_all_rejected(early=True)
        if retryable and live_mode and attempt < max_attempts:
            print(f"autolabel_v2 第 {attempt}/{max_attempts} 次拒絕 capture "
                  f"{manifest['capture_id']}（可重試）: {'; '.join(reasons)}；"
                  f"此 capture 的輸出保留在 {pipeline_dir} 供除錯、不往下使用，重新拍照",
                  file=sys.stderr)
            continue
        if retryable and live_mode:
            _stop_all_rejected()
        print(f"autolabel_v2 拒絕此 frame（{result['seam_json']} 未產生，疊圖目錄: "
              f"{result['overlay_dir']}；原因: {'; '.join(reasons)}），安全中止，"
              f"不查深度、不沿用舊檔", file=sys.stderr)
        blocked = reject.get("stage") == "cut_permission"
        raise DetectionAbort(1, "; ".join(reasons), status="blocked" if blocked else "rejected",
                             reasons=reasons, cut_permission=reject.get("cut_permission"))
    # 偵測完成：請 server 結束並釋放 GPU（不等待；最後的 stop 確認/強制終止）。
    stop_autolabel_v2_server(wait=False)
    if max_attempts > 1:
        print(f"→ autolabel_v2 第 {attempt}/{max_attempts} 次成功（capture {manifest['capture_id']}）")
    seam_json_path = result["seam_json"]
    if not seam_json_path.is_file():
        print(f"autolabel_v2 拒絕此 frame（{seam_json_path} 未產生，疊圖目錄: "
              f"{result['overlay_dir']}），安全中止，不查深度、不沿用舊檔", file=sys.stderr)
        raise DetectionAbort(1, f"detector produced no seam JSON ({seam_json_path})")
    try:
        validate_detection(seam_json_path, image_path, manifest)
    except ValueError as error:
        print(f"偵測 JSON 驗證失敗: {error}", file=sys.stderr)
        raise DetectionAbort(1, f"detection JSON validation failed: {error}") from error
    print(f"→ 偵測結果: {seam_json_path}")
    print(f"→ autolabel_v2 疊圖: {result['overlay_png']}"
          + ("" if result["overlay_png"].is_file() else "（未產生，請手動檢查）"))

    # Safety rule 2: only an accepted AND (cut-allowed OR operator-overridden) detection may become a snapshot.
    try:
        seam_doc = _load_json_object(seam_json_path)
    except (OSError, ValueError) as error:
        raise DetectionAbort(1, f"cannot read the detector seam JSON: {error}") from error
    cut_permission = seam_doc.get("cut_permission")
    status = cut_permission.get("status") if isinstance(cut_permission, dict) else None
    if status not in ("allowed", "blocked"):
        # A missing / malformed / unknown cut_permission is a contract violation of the detector,
        # never a real "blocked": --allow-cut-blocked must not promote it to an override.
        raise DetectionAbort(
            1, f"detector seam JSON has a missing or invalid cut_permission "
               f"(status {status!r}; expected 'allowed' or 'blocked'): contract violation, "
               f"aborting regardless of allow_cut_blocked",
            status="error", reasons=["cut_permission missing or invalid"])
    if status != "allowed":
        if not cfg.allow_cut_blocked:
            raise DetectionAbort(
                1, f"cut permission is {status}, not allowed (allow_cut_blocked not set)",
                status="blocked",
                reasons=[f"cut_blocked({r.get('code')}: {r.get('detail')})"
                         for r in cut_permission.get("reasons", []) if isinstance(r, dict)]
                or ["cut_permission blocked"],
                cut_permission=cut_permission)
        cut_permission = {**cut_permission, "override": "allow_cut_blocked"}

    # ---- downstream (3-6), built at the pipeline-dir staging path
    say("query_depth", attempt, "camera-frame geometry")
    query_argv = ["--pcd", pcd_path, "--capture-manifest", str(manifest_path),
                  "--output-json", str(camera_geometry_path), "--seam-json", str(seam_json_path)]
    if cfg.extra_query_args:
        print(f"[透傳給 query_depth 的額外參數] {' '.join(cfg.extra_query_args)}")
        query_argv += list(cfg.extra_query_args)
    run_script_step(cfg, "3/5 查量測紙箱頂面深度（query_depth_at_pixel.py）", "query_depth", query_argv)
    if not camera_geometry_path.is_file():
        print(f"query_depth 沒有提交 {camera_geometry_path}", file=sys.stderr)
        raise DetectionAbort(1, f"query_depth produced no {camera_geometry_path}")
    say("base_link", attempt, "camera -> base_link")
    if cfg.driver_extrinsic == "recorded":
        # clear fail-closed reason BEFORE the transform step (the step itself enforces it too)
        from . import base_link as _base_link  # noqa: PLC0415
        try:
            scene = None
            if getattr(cfg, "scene_config", None):
                from ..core import scene as _scene  # noqa: PLC0415
                scene = _scene.load_scene(cfg.scene_config)
            block = _base_link.driver_extrinsic_from_manifest(
                manifest_path, _base_link.RECORDED_DRIVER_EXTRINSIC["parent_frame"],
                manifest["pointcloud"]["header"]["frame_id"])
            if block is not None:
                _base_link.driver_extrinsic_check(
                    block, cfg.driver_extrinsic_policy,
                    reference=_base_link.scene_driver_blocks(scene)[0])
        except ValueError as error:
            print(f"driver extrinsic check failed: {error}", file=sys.stderr)
            raise DetectionAbort(1, str(error), reasons=[str(error)]) from error
    convert_argv = ["--json", str(camera_geometry_path), "--target-frame", cfg.target_frame,
                    "--timeout", str(cfg.tf_timeout), "--output-json", str(base_points_path),
                    "--planner-output-json", str(staged_snapshot_path),
                    "--calib-file", str(Path(cfg.calib_file).expanduser()),
                    "--driver-extrinsic", cfg.driver_extrinsic,
                    "--driver-extrinsic-policy", cfg.driver_extrinsic_policy]
    if getattr(cfg, "scene_config", None):
        convert_argv += ["--scene-config", str(cfg.scene_config)]
    run_script_step(cfg, "4/5 轉換 base_link 並提交 canonical planner snapshot",
                    "base_link", convert_argv, inprocess_ok=cfg.driver_extrinsic == "recorded")
    say("snapshot", attempt, "side cutting points + provenance")
    try:
        print(f"\n{'=' * 60}\n[5/5 加入側邊切割中點（side_cutting_point.py）]\n{'=' * 60}")
        add_side_cutting_points(staged_snapshot_path)
        validate_planner_snapshot(staged_snapshot_path, manifest, cfg.target_frame)
        stamp_extra = {
            "autolabel_v2_attempts": {
                "max_attempts": int(cfg.max_attempts) if live_mode else 1,
                "used": len(attempts),
                "retry_policy": retry_policy,
                "average_frames": int(cfg.average_frames) if live_mode else 1,
                "attempts": attempts},
            "cut_permission": cut_permission}
        if isinstance(seam_doc.get("seam_curvature_mm"), (int, float)):
            stamp_extra["seam_curvature_mm"] = float(seam_doc["seam_curvature_mm"])
        stamp_extra.update(support_depth_below_top(seam_doc))
        stamp_detector_method(staged_snapshot_path, DETECTOR_METHOD, stamp_extra)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"planner snapshot 側邊切割點整合失敗: {error}", file=sys.stderr)
        Path(staged_snapshot_path).unlink(missing_ok=True)
        raise DetectionAbort(1, f"planner snapshot post-processing failed: {error}") from error
    height_waived = ("height_waived(support_hidden)" in ((cut_permission or {}).get("flags") or [])
                     or any(isinstance(r, dict) and r.get("code") == "height_unavailable"
                            for r in ((cut_permission or {}).get("reasons") or [])))
    if cfg.side_seed and height_waived:
        # generic box without a height (dual_arm_v1 waiver): the side-tape seed needs a height and
        # would fail; the top cut does not need it (the executor fails closed on a missing seed block
        # only when side_tape_handoff_enabled)
        print("→ side-tape seed skipped: generic box without a measured height (top cut only)")
    elif cfg.side_seed:
        # 步驟 6：其 contract 角點命名保證 seam_start/seam_end 落在 left/right 邊上；失敗時 fail closed：
        # 暫存 snapshot 移到 <snapshot>.side_seed_rejected（planner 路徑不再有檔案），非零退出。
        say("side_seed", attempt, "side-tape seed")
        # dry run: never touch <snapshot>.side_seed_rejected next to the production path
        quarantine_for = final_target if publish else Path(staged_snapshot_path)
        run_side_face_seed(staged_snapshot_path, cfg, seam_json_path,
                           quarantine_path=side_seed_rejected_path(quarantine_for))
        print(f"→ side_tape_seed/v1.1 已加入 planner snapshot: {staged_snapshot_path}")
    try:
        snapshot = _load_json_object(staged_snapshot_path)
    except (OSError, ValueError) as error:
        raise DetectionAbort(1, f"cannot re-read the finished snapshot: {error}") from error
    published = Path(staged_snapshot_path)
    if cancelled():
        raise DetectionAbort(1, "cancelled before publish", status="cancelled")
    if publish:
        try:
            _write_json_atomically(final_target, snapshot)      # atomic same-directory replace
        except (OSError, TypeError, ValueError) as error:
            remove_snapshot(final_target)
            raise DetectionAbort(1, f"cannot write the snapshot {final_target}: {error}") from error
        published = final_target
    print(f"\n完成，capture={manifest['capture_id']}\n→ camera geometry: {camera_geometry_path}"
          f"\n→ base geometry: {base_points_path}\n→ planner snapshot: {published}")
    say("done", attempt, f"snapshot {published}")
    overlay = result["overlay_png"]
    return DetectResult(
        success=True, status="accepted", reason="", reasons=[], cut_permission=cut_permission,
        snapshot=snapshot, snapshot_path=str(final_target) if publish else None,
        staged_snapshot_path=str(staged_snapshot_path),
        cut_summary="overridden" if cut_permission.get("override") else "allowed",
        capture_id=manifest["capture_id"],
        seam_status=seam_doc.get("seam_status"), attempts=list(attempts),
        pipeline_dir=str(pipeline_dir), exit_code=0,
        overlay_path=str(overlay) if Path(overlay).is_file() else None)


# ----------------------------------------------------------------------------- CLI
def _build_parser():
    parser = argparse.ArgumentParser(
        prog="box_seam_detect",
        description="Capture (rgbd recorder) or replay (--from-image) -> autolabel_v2 detector -> camera "
                    "geometry -> base_link planner snapshot -> side cutting points -> side-tape seed. "
                    "Exit 0 accepted, 1 scene rejection / blocked / failed step, 2 prerequisite failure.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default=str(Path.home() / "rgbd_captures"),
                        help="recorder output root (rgb/pointcloud/captures), detector-server workdir and "
                             "fallback pipeline root (default ~/rgbd_captures)")
    parser.add_argument("--rgb-topic", default="/camera/color/image_raw")
    parser.add_argument("--pointcloud-topic", default="/camera/depth_registered/points")
    parser.add_argument("--from-image", default=None, metavar="IMAGE_PATH",
                        help="offline mode: use this existing RGB image (its .pcd / manifest are found by name)")
    parser.add_argument("--pcd", default=None, metavar="PCD_PATH", help="with --from-image: explicit cloud")
    parser.add_argument("--capture-manifest", default=None, metavar="MANIFEST_PATH",
                        help="with --from-image: the rgbd_capture/v1 manifest of that capture")
    parser.add_argument("--camera-geometry-json", "--output-json", dest="camera_geometry_json", default=None,
                        metavar="PATH", help="camera-frame geometry JSON (default: in the pipeline dir)")
    parser.add_argument("--base-points-json", default=None, metavar="PATH",
                        help="base_link geometry JSON (default: in the pipeline dir)")
    parser.add_argument("--planner-output-json", default=None, metavar="PATH",
                        help="atomically published box_h_cut_snapshot/v1 (default: only <pipeline dir>/"
                             "planner_snapshot.json). Removed at the start of every run.")
    parser.add_argument("--predict-out-dir", default=None, metavar="DIR",
                        help="root of the pipeline/<capture_id>/<run>/ output dirs (default: next to the "
                             "capture's captures/ dir, else --output-dir)")
    parser.add_argument("--target-frame", default="base_link")
    parser.add_argument("--calib-file", default=None,
                        help=f"hand-eye calibration (default: the scene's calib_file, else {DEFAULT_CALIB_FILE})")
    parser.add_argument("--scene-config", default=None, metavar="PATH",
                        help="scene config (YAML path or shipped name such as dual_arm_v1; default "
                             "table_cell_v1 = today's values): depth-top workspace + support mode, "
                             "guided-plane gate, seam-direction prior, driver extrinsic reference, default "
                             "calib file")
    parser.add_argument("--driver-extrinsic", choices=("tf", "recorded"), default=None,
                        help="camera_link->optical source: tf (default; needs ROS, runs the base_link step in "
                             "its own process) or recorded (manifest block, else the recorded constant)")
    parser.add_argument("--driver-extrinsic-policy", choices=("refuse", "warn", "use_manifest"),
                        default="refuse",
                        help="manifest driver_extrinsic block differing from the recorded constant by > 1 mm / "
                             "0.1 deg: refuse (default), warn (use + log) or use_manifest")
    parser.add_argument("--tf-timeout", type=float, default=5.0)
    parser.add_argument("--downstream-inprocess", choices=("on", "off"), default="on",
                        help="on (default): depth query / base_link (recorded only) / side seed run in this "
                             "process; off = one subprocess per step. Same files and exit codes.")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--autolabel-v2", action="store_true",
                        help="accepted for compatibility with capture_and_query_depth.py (always on)")
    parser.add_argument("--autolabel-v2-script", default=None, metavar="PATH",
                        help="run this detector script instead of `python -m box_seam_detection.live.detector`")
    parser.add_argument("--autolabel-v2-sam3-python", default=DEFAULT_SAM3_PYTHON, metavar="PYTHON_PATH",
                        help="SAM3 worker interpreter (book_grasping env)")
    parser.add_argument("--allow-needs-review", action="store_true")
    parser.add_argument("--allow-cut-blocked", action="store_true",
                        help="accept cut_permission=blocked detections (recorded as an override)")
    parser.add_argument("--autolabel-v2-attempts", type=int, default=None, metavar="N",
                        help=f"live: retry retryable rejects with a fresh capture, at most N captures in total "
                             f"(1..{MAX_AUTOLABEL_V2_ATTEMPTS}, default {DEFAULT_AUTOLABEL_V2_ATTEMPTS}); "
                             "offline always 1")
    parser.add_argument("--autolabel-v2-retry-policy", choices=AUTOLABEL_V2_RETRY_POLICIES, default=None,
                        help=f"smart (default {DEFAULT_AUTOLABEL_V2_RETRY_POLICY}): stop early when the "
                             f"second capture fails the same way; all: use every attempt")
    parser.add_argument("--autolabel-v2-average", type=int, default=None, metavar="N",
                        help=f"live: record a burst of N frames per attempt and average the RGB "
                             f"(1..{MAX_AUTOLABEL_V2_AVERAGE}, default {DEFAULT_AUTOLABEL_V2_AVERAGE} = single)")
    parser.add_argument("--average-max-diff", type=float, default=None, metavar="GRAY")
    parser.add_argument("--average-rate-hz", type=float, default=None, metavar="HZ")
    parser.add_argument("--rect-tol-mm", type=float, default=None, metavar="MM",
                        help=f"detector --rect-tol-mm (0 < tol <= {MAX_RECT_TOL_MM:g}, detector default 3.0)")
    parser.add_argument("--no-rect-check", action="store_true")
    parser.add_argument("--no-depth-edge", action="store_true")
    parser.add_argument("--autolabel-v2-server", choices=("auto", "on", "off"), default="auto",
                        help="detector server: auto = live only, on = also offline, off = one-shot subprocesses")
    parser.add_argument("--sam3-daemon", choices=("auto", "on", "off"), default="auto",
                        help="persistent SAM3 daemon: auto = live (an already set "
                             "AUTOLABEL_V2_SAM3_DAEMON wins), on = also offline, off = per-run worker")
    parser.add_argument("--no-side-seed", action="store_true")
    parser.add_argument("--capture-staging-dir", default=None, metavar="PATH|off",
                        help="live: write the captures to this RAM dir first and archive them to --output-dir "
                             "after the snapshot is published (the accepted capture -> rgb/pointcloud/captures, "
                             "every other one -> failed_recent/, last 3). 'off' = write straight into "
                             "--output-dir. Default: env BOX_SEAM_CAPTURE_STAGING, else "
                             "/dev/shm/box_seam_detection/staging. Ignored with --from-image.")
    parser.add_argument("--dry-run", action="store_true",
                        help="run everything but never touch the --planner-output-json path")
    return parser


def _scene_calib(scene_config):
    """Default calib file: the scene's, else the table-cell default (a bad scene
    config is reported by preflight_calibration, not here)."""
    if not scene_config:
        return DEFAULT_CALIB_FILE
    try:
        from ..core import scene as _scene  # noqa: PLC0415
        return _scene.load_scene(scene_config).calib_file
    except Exception:  # noqa: BLE001
        return DEFAULT_CALIB_FILE


def config_from_args(args):
    """DetectConfig of the parsed CLI args (``args.from_image`` decides live vs offline)."""
    live = not args.from_image
    return DetectConfig(
        max_attempts=(args.autolabel_v2_attempts if args.autolabel_v2_attempts is not None
                      else DEFAULT_AUTOLABEL_V2_ATTEMPTS),
        retry_policy=args.autolabel_v2_retry_policy or DEFAULT_AUTOLABEL_V2_RETRY_POLICY,
        allow_needs_review=args.allow_needs_review, allow_cut_blocked=args.allow_cut_blocked,
        dry_run=args.dry_run, rect_tol_mm=args.rect_tol_mm, sam3_python=args.autolabel_v2_sam3_python,
        sam3_daemon=args.sam3_daemon, downstream_inprocess=args.downstream_inprocess == "on",
        calib_file=args.calib_file or _scene_calib(args.scene_config), driver_extrinsic=args.driver_extrinsic or "tf",
        scene_config=args.scene_config,
        target_frame=args.target_frame, side_seed=not args.no_side_seed,
        output_root=Path(args.output_dir).resolve(),
        snapshot_path=Path(args.planner_output_json).resolve() if args.planner_output_json else None,
        live=live, autolabel_v2_server=args.autolabel_v2_server, no_rect_check=args.no_rect_check,
        no_depth_edge=args.no_depth_edge, debug=args.debug, tf_timeout=args.tf_timeout,
        pipeline_root=Path(args.predict_out_dir).resolve() if args.predict_out_dir else None,
        camera_geometry_path=(Path(args.camera_geometry_json).resolve()
                              if args.camera_geometry_json else None),
        base_points_path=Path(args.base_points_json).resolve() if args.base_points_json else None,
        detector_script=args.autolabel_v2_script,
        average_frames=(args.autolabel_v2_average if args.autolabel_v2_average is not None
                        else DEFAULT_AUTOLABEL_V2_AVERAGE) if live else 1,
        average_max_diff=(args.average_max_diff if args.average_max_diff is not None
                          else DEFAULT_AVERAGE_MAX_DIFF),
        average_rate_hz=(args.average_rate_hz if args.average_rate_hz is not None
                         else DEFAULT_AVERAGE_RATE_HZ),
        rgb_topic=args.rgb_topic, pointcloud_topic=args.pointcloud_topic,
        driver_extrinsic_policy=args.driver_extrinsic_policy)


def _check_cli_args(parser, args):
    if args.pcd and not args.from_image:
        parser.error("--pcd 必須搭配 --from-image 使用")
    if args.capture_manifest and not args.from_image:
        parser.error("--capture-manifest 只適用於 --from-image 離線模式")
    if args.autolabel_v2_attempts is not None and not (
            1 <= args.autolabel_v2_attempts <= MAX_AUTOLABEL_V2_ATTEMPTS):
        parser.error(f"--autolabel-v2-attempts 必須在 1..{MAX_AUTOLABEL_V2_ATTEMPTS} 之間")
    if args.autolabel_v2_average is not None and not (
            1 <= args.autolabel_v2_average <= MAX_AUTOLABEL_V2_AVERAGE):
        parser.error(f"--autolabel-v2-average 必須在 1..{MAX_AUTOLABEL_V2_AVERAGE} 之間")
    if args.average_max_diff is not None and not (
            math.isfinite(args.average_max_diff) and 0.0 < args.average_max_diff <= 255.0):
        parser.error("--average-max-diff 必須在 (0, 255] 之間")
    if args.average_rate_hz is not None and not (
            math.isfinite(args.average_rate_hz) and 0.0 < args.average_rate_hz <= MAX_AVERAGE_RATE_HZ):
        parser.error(f"--average-rate-hz 必須在 (0, {MAX_AVERAGE_RATE_HZ:g}] 之間")
    if args.rect_tol_mm is not None and not (
            math.isfinite(args.rect_tol_mm) and 0.0 < args.rect_tol_mm <= MAX_RECT_TOL_MM):
        parser.error(f"--rect-tol-mm 必須在 (0, {MAX_RECT_TOL_MM:g}] 之間")
    if args.rect_tol_mm is not None and args.no_rect_check:
        parser.error("--rect-tol-mm 跟 --no-rect-check 不能一起用")


def _offline_capture(parser, args):
    image_path = str(Path(args.from_image).resolve())
    if not Path(image_path).is_file():
        parser.error(f"找不到指定的影像: {image_path}")
    if args.pcd:
        pcd_path = str(Path(args.pcd).resolve())
        if not Path(pcd_path).is_file():
            parser.error(f"找不到指定的點雲: {pcd_path}")
    else:
        pcd_path = find_matching_pcd(image_path)
        if pcd_path is None:
            parser.error("依檔名自動配對點雲失敗；請改用 --pcd 手動指定")
    try:
        manifest_path = find_capture_manifest(image_path, args.capture_manifest)
        manifest = load_capture_manifest(manifest_path, image_path, pcd_path)
    except ValueError as error:
        parser.error(str(error))
    print(f"\n{'=' * 60}\n[1/5 離線模式：驗證既有 RGB-D capture]\n{'=' * 60}")
    print(f"→ 影像: {image_path}\n→ 點雲: {pcd_path}\n→ capture manifest: {manifest_path}")
    return CaptureFiles(capture_id=manifest["capture_id"], image_path=Path(image_path),
                        pcd_path=Path(pcd_path), manifest_path=Path(manifest_path),
                        stamp_ns=int(manifest["image"]["header"]["stamp_ns"]))


def _live_capture_source(cfg):
    """next_capture of the CLI's live mode: the recorder subprocess (single frame or averaged burst)."""
    output_dir = Path(getattr(cfg, "capture_staging_dir", None) or cfg.output_root)   # RAM staging when on
    seen_sources = set()
    average_frames = int(cfg.average_frames)

    def next_capture(attempt):
        verb = "拍照" if attempt == 1 else "重新拍照"
        if average_frames > 1:
            burst = acquire_averaged_live_capture(
                cfg, output_dir, average_frames, float(cfg.average_max_diff),
                title=f"1/5 {verb}（rgbd_recorder.py 連拍 {average_frames} 幀平均，"
                      f"第 {attempt}/{cfg.max_attempts} 次）")
            averaging = burst["averaging"]
            source_ids = averaging["source_capture_ids"]
            if seen_sources.intersection(source_ids):
                print(f"rgbd_recorder.py 沒有產生新的 capture（burst {source_ids} "
                      f"有先前嘗試已使用的 capture），安全中止", file=sys.stderr)
                raise DetectionAbort(1, f"burst {source_ids} reuses captures of an earlier attempt")
            seen_sources.update(source_ids)
            if burst["status"] != "ok":
                reason = (f"burst_not_static(max_diff={averaging['max_frame_diff']:.2f}"
                          f">{float(cfg.average_max_diff):g})")
                raise CaptureRejected(averaging["middle_capture_id"], [reason], True,
                                      {"averaging": averaging, "source_capture_ids": source_ids})
            image_path, pcd_path = burst["image_path"], burst["pcd_path"]
            manifest_path, manifest = burst["manifest_path"], burst["manifest"]
        else:
            title = ("1/5 拍照（rgbd_recorder.py --once）" if attempt == 1 else
                     f"1/5 重新拍照（rgbd_recorder.py --once，第 {attempt}/{cfg.max_attempts} 次）")
            image_path, pcd_path, manifest_path, manifest = acquire_live_capture(cfg, output_dir, title=title)
        return CaptureFiles(capture_id=manifest["capture_id"], image_path=Path(image_path),
                            pcd_path=Path(pcd_path), manifest_path=Path(manifest_path),
                            stamp_ns=int(manifest["image"]["header"]["stamp_ns"]))
    return next_capture


def _archive_staged_captures(staging, cfg, result):
    """CLI: archive the staged captures synchronously, right before exit (prints one line); never raises."""
    try:
        report = staging.archive_run(result, **run_archive_kwargs(cfg, result))
        if report.success_ids or report.failed_ids or report.discarded_ids or report.errors:
            print(report.summary())
        staging.close()
    except Exception as error:  # noqa: BLE001 — archiving must not change the run's exit code
        print(f"capture 歸檔失敗: {type(error).__name__}: {error}（staging 檔案留給下次啟動歸檔）", file=sys.stderr)


def main(argv=None) -> int:
    parser = _build_parser()
    args, extra_query_args = parser.parse_known_args(sys.argv[1:] if argv is None else list(argv))
    _check_cli_args(parser, args)
    cfg = config_from_args(args)
    cfg.extra_query_args = tuple(extra_query_args)
    staging = None      # RAM capture staging (live only; offline --from-image never moves or deletes anything)
    if not cfg.live:
        if args.autolabel_v2_attempts not in (None, 1, DEFAULT_AUTOLABEL_V2_ATTEMPTS):
            print("→ 離線 --from-image 模式：--autolabel-v2-attempts 不適用，只跑一次")
        if args.autolabel_v2_average is not None:
            print("→ 離線 --from-image 模式：--autolabel-v2-average 不適用，使用單張影像")
        offline_files = _offline_capture(parser, args)
        next_capture = lambda attempt: offline_files  # noqa: E731
    else:
        staging = open_staging(resolve_staging_setting(args.capture_staging_dir), cfg.output_root)
        if staging is not None:
            cfg.capture_staging_dir = staging.root
            print(f"→ capture staging（RAM）: {staging.root} → 歸檔到 {cfg.output_root}")
        next_capture = _live_capture_source(cfg)
    result = None
    try:
        result = run_detection(next_capture, cfg)
    except Exception as error:  # noqa: BLE001 — e.g. a recorder failure raised by next_capture
        print(f"detection failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    finally:
        # The snapshot (if any) was published inside run_detection; archiving only starts now.
        if staging is not None:
            _archive_staged_captures(staging, cfg, result)
    if not result.success:
        print(f"[{result.status}] {result.reason}", file=sys.stderr)
        return result.exit_code or 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
