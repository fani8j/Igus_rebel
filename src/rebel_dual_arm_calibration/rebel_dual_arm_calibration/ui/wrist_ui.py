"""Wrist-camera (eye-in-hand) calibration panel: state machine and the automatic view sequence.

    WristManager   no ROS imports.  All collaborators are injected (``ros`` = ui.wrist_ros.WristRos-like, ``moveit`` =
                   suggest.MoveItBridge-like, ``jog`` = JogController-like, ``detector_fn``), so the tests use fakes.

Flow (buttons in order): Locate board (scene camera) -> Add safety scene -> jog the arm by hand until the board is fully
visible in the wrist image -> Use as seed -> Generate views -> [Enable motion + confirm] Start auto sequence -> (Solve,
Write result, Publish TF).  The sequence is the ONLY place this module asks the jog controller to move, and it does so only
through ``jog.plan_move`` / ``jog.execute_move`` ("Move there"): same speed caps (0.04 rad/s, 0.25 rad/s^2, 40 mm/s), same
MotionSupervisor, collision re-check of the executed curve, STOP epoch and allow_real_move gating.  This module adds:

    * on REAL hardware the sequence is refused unless the UI was launched with allow_real_move:=true, AND this page holds the
      jog's real-hardware enable, AND the start request carries the explicit per-run confirmation;
    * the safety scene (calib_table_slab / calib_board_stand / calib_pole_pad / calib_head_pad) must be present in the
      planning scene before the views are generated and before EVERY move (fail closed);
    * a motion that is not a clean "done" (supervision fault, STOP, refusal) ends the run; so do 3 consecutive failures, a
      STOP epoch change, and a page that stopped polling (heartbeat);
    * after every move: measured standstill, >= 1.5 s settle, several frames, board detected (>= 16 corners), corner stillness
      across the frames and an unchanged TF at the frame stamps, otherwise the view is skipped (not an error of the run).
"""
from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .. import eye_in_hand as eih
from .. import safety_scene as ss
from .. import wrist_poses as wp
from ..capture_quality import corner_motion_stats, judge_motion
from ..transforms import R_to_quat, inv_T, make_T, rotation_angle_deg, T_to_xyz_rpy
from .suggest import SuggestRefused, clean

SETTLE_S = 1.5
N_FRAMES = 3
FRAME_GAP_S = 0.2
FRAME_WAIT_S = 12.0
STILL_COHERENT_PX = 0.4
STILL_SINGLE_PX = 1.5
TF_STILL_POS_M = 3e-4
TF_STILL_ROT_DEG = 0.05
TF_MAX_AGE_S = 1.0
MAX_CONSECUTIVE_FAILURES = 3
HEARTBEAT_TIMEOUT_S = 20.0
BOARD_STILL_POS_MM = 1.5
BOARD_STILL_ROT_DEG = 0.3
LOCATE_FRAMES = 5
LOCATE_SPAN_S = 1.0
SEED_MIN_CORNERS = 16
SEED_IMAGE_MARGIN = 0.03
SCENE_STALE_MM = 2.0
SCENE_STALE_DEG = 0.3
MOCK_FALLBACK_FOV_DEG = 90.0
MOVE_WAIT_EXTRA_S = 45.0
LOG_KEEP = 40
FRAME_FRESH_S = 2.0


class WristRefused(SuggestRefused):
    pass


class _Abort(Exception):
    """End the run now (message is the reason)."""


class _Skip(Exception):
    """Skip this view and go on."""


def _T_list(T) -> Optional[list]:
    return None if T is None else np.asarray(T, dtype=float).tolist()


def _pose_summary(T: np.ndarray) -> dict:
    xyz, rpy = T_to_xyz_rpy(T)
    return {"xyz_m": [float(v) for v in xyz], "rpy_deg": [float(math.degrees(v)) for v in rpy]}


def _mean_pose(Ts: Sequence[np.ndarray]) -> np.ndarray:
    t = np.mean([T[:3, 3] for T in Ts], axis=0)
    M = sum(T[:3, :3] for T in Ts) / len(Ts)
    U, _, Vt = np.linalg.svd(M)
    return make_T(U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt, t)


class WristManager:
    def __init__(self, mode: str, cfgs: Dict[str, Any], jog, ros, moveit, out_dir_fn: Callable[[], Optional[str]],
                 detector_fn: Callable[[], Tuple[Any, Any]], *, allow_real_move: bool = False,
                 camera_serial: str = "", options: Optional[wp.PoseOptions] = None, cad_roll_deg: float = 0.0,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep,
                 start_thread: bool = True):
        self.mode, self.cfgs, self.jog, self.ros, self.moveit = mode, cfgs, jog, ros, moveit
        self.out_dir_fn, self.detector_fn = out_dir_fn, detector_fn
        self.allow_real_move = bool(allow_real_move)
        self.camera_serial = camera_serial
        self.opt = options or wp.PoseOptions()
        self.cad_roll_deg = float(cad_roll_deg)
        self.clock, self.sleep = clock, sleep
        self._start_thread = start_thread
        self.log = logging.getLogger("calibration_ui.wrist")
        self.lock = threading.RLock()
        self.arm: Optional[str] = next(iter(cfgs)) if cfgs and mode in ("single_left", "single_right") else None
        self._board: Optional[dict] = None
        self._scene: Optional[dict] = None
        self._seed: Optional[dict] = None
        self._cands: List[wp.Candidate] = []
        self._cand_state = {"state": "idle", "message": "", "stats": {}}
        self._gen = 0
        self._samples: List[eih.WristSample] = []
        self._est: Optional[eih.HandEyeResult] = None
        self._result: Optional[eih.HandEyeResult] = None
        self._result_files: Dict[str, str] = {}
        self._tf_published = False
        self._run: dict = {"state": "idle", "message": "", "index": 0, "total": 0, "skipped": 0, "failures": 0, "log": []}
        self._run_id = 0
        self._stop_evt = threading.Event()
        self._run_thread: Optional[threading.Thread] = None
        self._run_page: Optional[str] = None
        self._hb: Optional[float] = None
        self._archived = False
        self._preview_cache: Tuple[int, Any] = (-1, None)
        self._next_outline: Optional[list] = None

    # ---- small helpers ---------------------------------------------------------------------------
    @property
    def supported(self) -> bool:
        return self.arm is not None

    def _need_arm(self) -> str:
        if self.arm is None:
            raise WristRefused("wrist-camera calibration: single-arm modes (single_left / single_right) only", 400)
        return self.arm

    def _is_real(self) -> bool:
        """True unless controller_manager says the arm is mock_components/GenericSystem (fail closed)."""
        arm = self._need_arm()
        for obj in (getattr(self.jog, "core", None), self.jog):
            fn = getattr(obj, "is_real", None)
            if fn is not None:
                try:
                    return bool(fn(arm))
                except Exception:  # noqa: BLE001
                    return True
        return True

    def _standstill(self) -> bool:
        arm = self._need_arm()
        for obj in (getattr(self.jog, "core", None), self.jog):
            fn = getattr(obj, "standstill", None)
            if fn is not None:
                try:
                    return bool(fn(arm))
                except Exception:  # noqa: BLE001
                    return False
        return True       # no direct probe (fakes): the "done" phase of the move already needed a measured standstill

    def _stop_epoch(self) -> Optional[int]:
        core = getattr(self.jog, "core", None)
        return getattr(core, "stop_epoch", None)

    def _jog_blocked(self) -> str:
        try:
            st = self.jog.status()
        except Exception as e:  # noqa: BLE001
            return f"jog state unknown ({e})"
        if st.get("available") is False:
            return "jog unavailable: " + str(st.get("error", ""))
        if st.get("busy") or st.get("stopping") or st.get("fault"):
            return "the arm is busy (a jog step / move is running, stopping or in fault)"
        return ""

    def _current_q(self, spread_max: Optional[float] = None) -> Optional[Dict[str, float]]:
        arm = self._need_arm()
        cfg = self.cfgs[arm]
        r = self.jog.joint_window([arm], 0.5)
        if r is None:
            return None
        q, spread = r[0], r[1]
        if spread_max is not None and not spread <= spread_max:
            return None
        if not all(n in q for n in cfg.joint_names):
            return None
        return {n: float(q[n]) for n in cfg.joint_names}

    def _out_dir(self) -> str:
        d = self.out_dir_fn()
        if not d:
            raise WristRefused("output directory unknown (set wrist_handeye_output_dir or wait for the recorder status)", 503)
        os.makedirs(os.path.join(d, "images"), exist_ok=True)
        return d

    def _rlog(self, msg: str) -> None:
        self.log.info("wrist run: %s", msg)
        with self.lock:
            self._run["log"].append(f"{time.strftime('%H:%M:%S')} {msg}")
            del self._run["log"][:-LOG_KEEP]
            self._run["message"] = msg

    def heartbeat(self, page_id: Optional[str]) -> None:
        with self.lock:
            if page_id is not None and page_id == self._run_page:
                self._hb = self.clock()

    # ---- geometry sources --------------------------------------------------------------------------
    def _table(self):
        det, spec = self.detector_fn()
        return det, spec

    def _wrist_camera(self) -> dict:
        """K / dist / size of the wrist camera: camera_info; in MOCK only a nominal 90 deg pinhole when there is none."""
        info = self.ros.wrist_info()
        if info is not None:
            return info
        if self._is_real():
            raise WristRefused("no wrist camera_info received (is the wrist camera running?)", 503)
        w, h = 1280, 800
        f = (w / 2.0) / math.tan(math.radians(MOCK_FALLBACK_FOV_DEG / 2.0))
        return {"K": np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]]), "dist": np.zeros(5), "size": (w, h),
                "frame_id": "mock_wrist_camera_optical"}

    def _current_X(self) -> Tuple[np.ndarray, str]:
        with self.lock:
            est, seed, board = self._est, self._seed, self._board
        if est is not None:
            return est.X.copy(), f"interim hand-eye solve ({est.stats['n_used']} samples)"
        if seed is not None and seed.get("T_opt_board") is not None and board is not None:
            X = inv_T(seed["A"]) @ board["T"] @ inv_T(seed["T_opt_board"])
            return X, "seed view + scene-camera board (one-shot estimate)"
        return wp.nominal_T_link6_optical(self.cad_roll_deg), "CAD nominal (rough)"

    # ---- 1. locate the board (scene camera) ------------------------------------------------------------
    def locate_board(self) -> dict:
        from ..hover import _table_frame_pose        # private helper reused on purpose (same PnP as the hover assist)
        arm = self._need_arm()
        info = self.ros.scene_info()
        if info is None:
            raise WristRefused("no scene-camera camera_info yet", 503)
        det, spec = self._table()
        poses, seen, t0 = [], set(), self.clock()
        n_corners = []
        while len(poses) < LOCATE_FRAMES and self.clock() - t0 < LOCATE_SPAN_S * 6 + 4:
            item = self.ros.scene_frame()
            if item is not None:
                bgr, seq, rx = item
                if seq not in seen and self.clock() - rx < FRAME_FRESH_S:
                    seen.add(seq)
                    if (bgr.shape[1], bgr.shape[0]) != tuple(info["size"]):
                        raise WristRefused(f"scene image {bgr.shape[1]}x{bgr.shape[0]} != camera_info {info['size']}", 409)
                    r = _table_frame_pose(bgr, info["K"], info["dist"], spec, det)
                    if isinstance(r, str):
                        raise WristRefused("table board not usable in the scene image: " + r, 409)
                    poses.append(r[0])
                    n_corners.append(r[2])
                    if len(poses) < LOCATE_FRAMES:
                        self.sleep(LOCATE_SPAN_S / max(LOCATE_FRAMES - 1, 1))
                    continue
            self.sleep(0.1)
        if len(poses) < 3:
            raise WristRefused(f"only {len(poses)} fresh scene-camera frames in time: is the scene camera running?", 503)
        mean = _mean_pose(poses)
        dp = max(float(np.linalg.norm(p[:3, 3] - mean[:3, 3])) for p in poses) * 1e3
        dr = max(rotation_angle_deg(p[:3, :3], mean[:3, :3]) for p in poses)
        if dp > BOARD_STILL_POS_MM or dr > BOARD_STILL_ROT_DEG:
            raise WristRefused(f"the board is not still in the scene camera ({dp:.2f} mm / {dr:.2f} deg spread over "
                               f"{len(poses)} frames; limits {BOARD_STILL_POS_MM:g} mm / {BOARD_STILL_ROT_DEG:g} deg)", 409)
        T_base_opt, why = self.ros.tf_base_scene_optical()
        if T_base_opt is None:
            raise WristRefused(why, 503)
        B = T_base_opt @ mean
        self._set_board(B, "scene_camera", {"n_frames": len(poses), "n_corners": int(np.min(n_corners)),
                                            "spread_mm": dp, "spread_deg": dr})
        return self.board_state()

    def mock_board(self, body: dict) -> dict:
        """TEST / MOCK ONLY: set T_base_board without a camera.  Refused unless the arm's hardware is
        mock_components/GenericSystem.  Body: {"xyz": [3], "rpy_deg": [3]} or {"distance_m": d} (board straight ahead of the
        seed camera at distance d, facing it)."""
        self._need_arm()
        if self._is_real():
            raise WristRefused("synthetic board pose is only available on MOCK hardware (mock_components/GenericSystem); "
                               "this arm is not (or not confirmed to be) mock", 403)
        from ..transforms import xyz_rpy_to_T
        if "xyz" in body:
            xyz, rpy = body.get("xyz"), body.get("rpy_deg", [180.0, 0.0, 0.0])
            try:
                B = xyz_rpy_to_T([float(v) for v in xyz], np.radians([float(v) for v in rpy]))
            except (TypeError, ValueError):
                raise WristRefused("xyz / rpy_deg must be 3 numbers each", 400)
        else:
            with self.lock:
                seed = self._seed
            if seed is None:
                raise WristRefused("use 'Use as seed' first (mock board is placed relative to the seed camera)", 409)
            d = float(body.get("distance_m", 0.40))
            X, _ = self._current_X()
            C = seed["A"] @ X
            # board frame: z INTO the paper = along the optical axis (away from the camera), x = camera x, y = camera y
            B = make_T(C[:3, :3], C[:3, 3] + d * C[:3, 2])
        if not np.all(np.isfinite(B)):
            raise WristRefused("non-finite board pose", 400)
        self._set_board(B, "mock_synthetic", {"warning": "SYNTHETIC board pose (mock only)"})
        return self.board_state()

    def _set_board(self, B: np.ndarray, source: str, extra: dict) -> None:
        with self.lock:
            self._board = {"T": B.copy(), "source": source, "time": time.time(), **extra}
            self._cands, self._cand_state = [], {"state": "idle", "message": "board changed: generate the views again",
                                                 "stats": {}}
            self._gen += 1
            self._est = None if not self._samples else self._est
        try:
            d = self._out_dir()
            with open(os.path.join(d, eih.SCENE_BOARD_FILE), "w") as f:
                json.dump({"T_base_board": B.tolist(), "source": source, "time": time.time(), **clean(extra)}, f, indent=1)
        except Exception as e:  # noqa: BLE001 - the output directory may not exist yet; recorded again at the next write
            self.log.warning("scene_board.json not written: %s", e)

    def board_state(self) -> Optional[dict]:
        with self.lock:
            b = self._board
        if b is None:
            return None
        B = b["T"]
        _, spec = self._table()
        n = -B[:3, 2]
        corners = ss.board_corners(B, spec.board_size())
        return clean({"source": b["source"], **{k: v for k, v in b.items() if k not in ("T", "source")},
                      "T_base_board": B, "centre_m": B[:3, 3], "normal_out": n,
                      "tilt_from_vertical_deg": math.degrees(math.acos(max(-1.0, min(1.0, float(n[2]))))),
                      "corners_z_m": corners[:, 2], "dist_from_base_axis_m": float(np.hypot(B[0, 3], B[1, 3]))})

    # ---- 2. safety scene --------------------------------------------------------------------------------
    def scene_add(self) -> dict:
        arm = self._need_arm()
        with self.lock:
            board = self._board
        if board is None:
            raise WristRefused("locate the board first (the slab is built from it)", 409)
        _, spec = self._table()
        try:
            objs = self.ros.scene_boxes((ss.POLE_ID, ss.HEAD_ID))
            specs = ss.build_scene(board["T"], spec.board_size(), objs.get(ss.POLE_ID), objs.get(ss.HEAD_ID))
        except ss.SceneRefused as e:
            raise WristRefused(str(e), 409)
        try:
            self.ros.apply_boxes(specs)
        except ss.SceneRefused as e:
            raise WristRefused(str(e), 503)
        # the current robot state must not touch the new boxes (e.g. a slab that reaches the base / link1)
        q = self._current_q()
        bad = ""
        if q is not None and self.moveit is not None:
            ok, why = self.moveit.valid(self.cfgs[arm], q)
            if not ok and "calib_" in why:
                bad = why
            elif not ok:
                self.log.warning("current state invalid after adding the safety scene: %s", why)
        if bad:
            try:
                self.ros.remove_boxes()
            except ss.SceneRefused:
                pass
            raise WristRefused(f"the safety scene intersects the robot in its current state ({bad}): removed, refusing", 409)
        with self.lock:
            self._scene = {"applied": True, "time": time.time(), "boxes": [s.to_dict() for s in specs],
                           "board_T": board["T"].copy()}
        return self.scene_state()

    def scene_remove(self) -> dict:
        try:
            self.ros.remove_boxes()
        except ss.SceneRefused as e:
            raise WristRefused(str(e), 503)
        with self.lock:
            self._scene = None
        return self.scene_state()

    def scene_state(self) -> dict:
        with self.lock:
            sc, b = self._scene, self._board
        if sc is None:
            return {"applied": False, "boxes": [], "stale": False}
        stale = b is not None and (np.linalg.norm(b["T"][:3, 3] - sc["board_T"][:3, 3]) * 1e3 > SCENE_STALE_MM or
                                   rotation_angle_deg(b["T"][:3, :3], sc["board_T"][:3, :3]) > SCENE_STALE_DEG)
        return clean({"applied": True, "boxes": sc["boxes"], "stale": bool(stale), "time": sc["time"]})

    def _require_scene(self) -> None:
        """Fail closed: the four boxes must be in the live planning scene, built from the current board pose."""
        st = self.scene_state()
        if not st["applied"]:
            raise WristRefused("add the safety scene first", 409)
        if st["stale"]:
            raise WristRefused("the board moved since the safety scene was built: remove and add the scene again", 409)
        try:
            ok, missing = self.ros.scene_has(ss.OUR_IDS)
        except ss.SceneRefused as e:
            raise WristRefused(f"cannot verify the safety scene: {e}", 503)
        if not ok:
            raise WristRefused(f"safety scene objects missing in the planning scene ({', '.join(missing)}): add it again", 409)

    # ---- 3. seed ----------------------------------------------------------------------------------------------
    def use_seed(self) -> dict:
        arm = self._need_arm()
        why = self._jog_blocked()
        if why:
            raise WristRefused("cannot take the seed now: " + why, 409)
        q = self._current_q(spread_max=5e-4)
        if q is None:
            raise WristRefused("the arm is not at standstill (or joint states unavailable): hold still", 409)
        A, why = self.ros.tf_base_link6(None)
        if A is None:
            raise WristRefused(why, 503)
        age = self.ros.tf_age_s()
        if age is not None and age > TF_MAX_AGE_S:
            raise WristRefused(f"TF base_link->link6 is {age:.1f} s old", 503)
        det_info: Optional[dict] = None
        T_ob = None
        frames = self.ros.wrist_frames(self.clock() - FRAME_FRESH_S)
        if frames:
            cam = self._wrist_camera()
            T_ob, det_info = self._detect_pose(frames[-1], cam)       # raises WristRefused with the reason
        elif self._is_real():
            raise WristRefused("no fresh wrist image: is the wrist camera running?", 503)
        seed = {"q": q, "A": A, "T_opt_board": T_ob, "time": time.time(), "detection": det_info,
                "mock_no_image": not frames}
        with self.lock:
            self._seed = seed
            self._cands, self._cand_state = [], {"state": "idle", "message": "new seed: generate the views", "stats": {}}
            self._gen += 1
        return self.seed_state()

    def _detect_pose(self, frame: dict, cam: dict) -> Tuple[np.ndarray, dict]:
        """Detect the table board in a wrist frame; require it FULLY visible.  Returns (T_optical_board, info)."""
        from ..hover import _table_frame_pose
        det, spec = self._table()
        bgr = frame["bgr"]
        if (bgr.shape[1], bgr.shape[0]) != tuple(cam["size"]):
            raise WristRefused(f"wrist image {bgr.shape[1]}x{bgr.shape[0]} != camera_info {cam['size']}", 409)
        r = _table_frame_pose(bgr, cam["K"], cam["dist"], spec, det)
        if isinstance(r, str):
            raise WristRefused("board not usable in the wrist image: " + r, 409)
        T, rms, n = r
        if n < SEED_MIN_CORNERS:
            raise WristRefused(f"only {n} board corners in the wrist image (need >= {SEED_MIN_CORNERS})", 409)
        uv = eih.project(T, wp.outline_points(spec.board_size()), cam["K"], cam["dist"])
        w, h = cam["size"]
        m = SEED_IMAGE_MARGIN
        if uv[:, 0].min() < m * w or uv[:, 0].max() > (1 - m) * w or uv[:, 1].min() < m * h or uv[:, 1].max() > (1 - m) * h:
            raise WristRefused("the board is not fully inside the wrist image: jog until the whole board is visible", 409)
        return T, {"n_corners": int(n), "pnp_rms_px": float(rms), "distance_m": float(np.linalg.norm(T[:3, 3]))}

    def seed_state(self) -> Optional[dict]:
        with self.lock:
            s = self._seed
        if s is None:
            return None
        return clean({"q_deg": [math.degrees(v) for v in s["q"].values()], "time": s["time"],
                      "detection": s["detection"], "mock_no_image": s["mock_no_image"],
                      "link6": _pose_summary(s["A"])})

    # ---- 4. views ------------------------------------------------------------------------------------------------
    def generate(self) -> str:
        arm = self._need_arm()
        with self.lock:
            seed, board = self._seed, self._board
            if self._cand_state["state"] == "computing":
                raise WristRefused("already generating", 409)
        if seed is None:
            raise WristRefused("take the seed first", 409)
        if board is None:
            raise WristRefused("locate the board first", 409)
        why = self._jog_blocked()
        if why:
            raise WristRefused("cannot generate now: " + why, 409)
        self._require_scene()
        if self.moveit is None:
            raise WristRefused("MoveIt services are not available in this UI", 503)
        ok, msg = self.moveit.ready()
        if not ok:
            raise WristRefused(msg, 503)
        cam = self._wrist_camera()
        _, spec = self._table()
        X, src = self._current_X()
        with self.lock:
            self._gen += 1
            gen = self._gen
            self._cands = []
            self._cand_state = {"state": "computing", "message": f"generating views (estimate: {src}) ...", "stats": {}}
        args = (gen, arm, seed, board["T"].copy(), cam, spec.board_size(), X, src)
        if self._start_thread:
            threading.Thread(target=self._gen_worker, args=args, name="wrist-gen", daemon=True).start()
        else:
            self._gen_worker(*args)
        return f"generating views (camera mount estimate: {src})"

    def _gen_worker(self, gen, arm, seed, B, cam, board_size, X, src) -> None:
        cfg = self.cfgs[arm]
        names = list(cfg.joint_names)
        stats: Dict[str, Any] = {"estimate": src, "rejected": {}}
        try:
            cands, rej = wp.generate_poses(seed["A"], X, B, cam["K"], cam["dist"], cam["size"], board_size, self.opt)
            stats["rejected"] = dict(rej)
            stats["view_ok"] = len(cands)
            q_seed = seed["q"]
            accepted: List[wp.Candidate] = []
            for c in cands:
                if len(accepted) >= self.opt.target_count:
                    break
                with self.lock:
                    if gen != self._gen:
                        return
                tilt0 = c.tilt_deg == 0 and c.roll_deg == 0 and abs(c.dist_factor - 1.0) < 1e-9
                if tilt0:                       # the seed view itself: captured without any motion
                    c.q, c.info["seed_view"] = dict(q_seed), True
                    accepted.append(c)
                    continue
                try:
                    q = self.moveit.ik(self.cfgs[arm], c.T_base_link6, q_seed)
                except Exception as e:  # noqa: BLE001
                    self._count(stats, "ik_error")
                    self.log.warning("IK call failed: %s", e)
                    continue
                if q is None:
                    self._count(stats, "no collision-free IK")
                    continue
                jump = wp.max_joint_change_deg([q[n] for n in names], [q_seed[n] for n in names])
                if jump > self.opt.max_joint_jump_deg:
                    self._count(stats, f"joint jump > {self.opt.max_joint_jump_deg:g} deg")
                    continue
                ok, why = self.moveit.valid(self.cfgs[arm], q)
                if not ok:
                    self._count(stats, "invalid state (collision/limits)")
                    continue
                c.q, c.info["max_joint_change_deg"] = dict(q), jump
                accepted.append(c)
            # nearest-neighbour order in joint space, the seed view first
            keyed = [(i, [c.q[n] for n in names]) for i, c in enumerate(accepted) if not c.info.get("seed_view")]
            order = wp.order_by_joint_distance([q_seed[n] for n in names], keyed)
            final = [c for c in accepted if c.info.get("seed_view")] + [accepted[i] for i in order]
            R = [(c.T_base_link6 @ X)[:3, :3] for c in final]
            span = max([rotation_angle_deg(a, b) for a in R for b in R] or [0.0])
            stats.update(accepted=len(final), rotation_span_deg=span,
                         rejected_total=sum(stats["rejected"].values()))
            with self.lock:
                if gen != self._gen:
                    return
                self._cands = final
                n_move = sum(1 for c in final if not c.info.get("seed_view"))
                self._cand_state = {"state": "ready" if final else "error",
                                    "message": f"{len(final)} views accepted ({n_move} need a move), "
                                               f"rotation span {span:.0f} deg" if final else
                                               "no reachable collision-free view found (see the rejection reasons)",
                                    "stats": stats}
        except Exception as e:  # noqa: BLE001
            self.log.exception("view generation failed")
            with self.lock:
                if gen == self._gen:
                    self._cands = []
                    self._cand_state = {"state": "error", "message": f"generation failed: {e}", "stats": stats}

    @staticmethod
    def _count(stats: dict, key: str) -> None:
        stats["rejected"][key] = stats["rejected"].get(key, 0) + 1

    # ---- 5. the automatic sequence ---------------------------------------------------------------------------
    def start(self, page_id: str, confirm: bool, epoch=None) -> dict:
        arm = self._need_arm()
        if not isinstance(page_id, str) or not page_id:
            raise WristRefused("X-Page-Id header missing", 400)
        if confirm is not True:
            raise WristRefused("the explicit per-run confirmation is missing: tick 'I confirm ...' and press Start", 409)
        real = self._is_real()
        if real and not self.allow_real_move:
            raise WristRefused("the automatic sequence on REAL hardware needs the UI to be launched with "
                               "allow_real_move:=true (this launch does not allow it)", 403)
        try:
            st = self.jog.status()
        except Exception as e:  # noqa: BLE001
            raise WristRefused(f"jog state unknown ({e})", 503)
        if real:
            if not st.get("real_enabled") or st.get("real_enabled_page") != page_id:
                raise WristRefused("press 'Enable motion' on this page first (real-hardware enable of THIS page)", 409)
        why = self._jog_blocked()
        if why:
            raise WristRefused("cannot start now: " + why, 409)
        with self.lock:
            if self._run["state"] in ("running", "stopping"):
                raise WristRefused("a sequence is already running", 409)
            cands = list(self._cands) if self._cand_state["state"] == "ready" else []
            if not cands:
                raise WristRefused("no generated views: press 'Generate views'", 409)
            if self._seed is None or self._board is None:
                raise WristRefused("seed or board missing", 409)
        self._require_scene()
        ok, msg = self.moveit.ready() if self.moveit is not None else (False, "MoveIt services are not available")
        if not ok:
            raise WristRefused(msg, 503)
        epoch_now = self._stop_epoch()
        if epoch is not None and epoch_now is not None and epoch != epoch_now:
            raise WristRefused("STOP was pressed after this request was sent: request void", 409)
        with self.lock:
            self._run_id += 1
            rid = self._run_id
            self._stop_evt.clear()
            self._run = {"state": "running", "message": "starting", "index": 0, "total": len(cands), "skipped": 0,
                         "failures": 0, "log": [], "started": time.time(), "real": real, "id": rid}
            self._run_page, self._hb = page_id, self.clock()
        args = (rid, page_id, cands, epoch_now)
        if self._start_thread:
            t = threading.Thread(target=self._run_loop, args=args, name="wrist-run", daemon=True)
            self._run_thread = t
            t.start()
        else:
            self._run_loop(*args)
        return {"ok": True, "message": f"sequence started: {len(cands)} views" + (" [REAL hardware]" if real else " [mock]")}

    def stop(self) -> dict:
        """Ends the run loop; the caller also presses the jog STOP (HTTP layer) so a move in flight is cancelled."""
        with self.lock:
            running = self._run["state"] == "running"
            if running:
                self._run["state"] = "stopping"
        self._stop_evt.set()
        try:
            self.jog.stop()
        except Exception:  # noqa: BLE001
            self.log.exception("jog.stop failed")
        return {"ok": True, "message": "stop requested: sequence ends, the arm is stopped" if running else
                "no sequence running (jog stop sent)"}

    def _alive(self, rid: int, page_id: str, epoch0: Optional[int]) -> None:
        if self._stop_evt.is_set():
            raise _Abort("stopped by the operator")
        with self.lock:
            hb = self._hb
        if hb is not None and self.clock() - hb > HEARTBEAT_TIMEOUT_S:
            raise _Abort("the page stopped polling (browser closed?)")
        e = self._stop_epoch()
        if epoch0 is not None and e is not None and e != epoch0:
            raise _Abort("a STOP / fault was registered by the jog controller")
        try:
            st = self.jog.status()
        except Exception as ex:  # noqa: BLE001
            raise _Abort(f"jog state unknown ({ex})")
        if st.get("fault") or st.get("stopping"):
            raise _Abort("the jog controller is stopping / in fault")

    def _run_loop(self, rid: int, page_id: str, cands: List[wp.Candidate], epoch0: Optional[int]) -> None:
        arm = self._need_arm()
        names = list(self.cfgs[arm].joint_names)
        cam = self._wrist_camera()
        _, spec = self._table()
        end_state, end_msg = "finished", "sequence finished"
        queue = list(cands)
        fails = 0
        done = 0
        try:
            with self.lock:
                B = self._board["T"].copy()
            while queue:
                self._alive(rid, page_id, epoch0)
                self._require_scene_abort()
                X, src = self._current_X()
                keep = []
                for c in queue:
                    ok, why, _ = wp.view_check(c.T_base_link6, X, B, cam["K"], cam["dist"], cam["size"],
                                               spec.board_size(), self.opt)
                    if ok or c.info.get("seed_view"):
                        keep.append(c)
                    else:
                        self._rlog(f"view '{c.desc}' dropped (prediction with {src}): {why}")
                        with self.lock:
                            self._run["skipped"] += 1
                queue = keep
                if not queue:
                    break
                q_now = self._current_q()
                if q_now is None:
                    raise _Abort("joint states unavailable")
                qv = [q_now[n] for n in names]
                nxt = min(queue, key=lambda c: 0.0 if c.info.get("seed_view") else
                          float(np.linalg.norm(np.array([c.q[n] for n in names]) - np.array(qv))) + 1e-6)
                queue.remove(nxt)
                with self.lock:
                    self._run["index"] = done + 1
                    self._next_outline = nxt.info.get("outline_px")
                self._rlog(f"view {done + 1}/{self._run['total']}: {nxt.desc}")
                try:
                    if not nxt.info.get("seed_view"):
                        self._move_to(nxt, page_id, epoch0, rid)
                    ok = self._capture_step(rid, page_id, epoch0)
                except _Skip as e:
                    ok = False
                    self._rlog(f"view skipped: {e}")
                done += 1
                if ok:
                    fails = 0
                else:
                    fails += 1
                    with self.lock:
                        self._run["skipped"] += 1
                        self._run["failures"] = fails
                    if fails >= MAX_CONSECUTIVE_FAILURES:
                        raise _Abort(f"{fails} consecutive failures: stopping")
        except _Abort as e:
            end_state = "stopped" if self._stop_evt.is_set() else "failed"
            end_msg = str(e)
        except Exception as e:  # noqa: BLE001
            self.log.exception("wrist run failed")
            end_state, end_msg = "failed", f"internal error: {type(e).__name__}: {e}"
        with self.lock:
            self._next_outline = None
        self._rlog(f"{end_state}: {end_msg}")
        try:
            self._final_solve()
        except Exception as e:  # noqa: BLE001
            self._rlog(f"final solve not possible: {e}")
        with self.lock:
            self._run["state"], self._run["message"] = end_state, end_msg
            self._run_page = None

    def _require_scene_abort(self) -> None:
        try:
            self._require_scene()
        except WristRefused as e:
            raise _Abort(f"safety scene check failed: {e.message}")

    def _move_to(self, cand: wp.Candidate, page_id: str, epoch0, rid: int) -> None:
        arm = self._need_arm()
        plan = self._reserved(lambda ep: self.jog.plan_move(page_id, arm, dict(cand.q), epoch=ep), epoch0)
        if not plan.get("ok") or not plan.get("plan_id"):
            msg = str(plan.get("message"))
            low = msg.lower()
            if any(k in low for k in ("stop was pressed", "real-hardware", "disabled", "fault", "busy", "enable", "unavailable",
                                      "not active", "stale")):
                raise _Abort("planning refused: " + msg)
            raise _Skip("no plan: " + msg)
        self._alive(rid, page_id, epoch0)
        self._require_scene_abort()
        ex = self._reserved(lambda ep: self.jog.execute_move(page_id, plan["plan_id"], epoch=ep), epoch0)
        if not ex.get("ok"):
            raise _Abort("execute refused: " + str(ex.get("message")))
        sid = ex.get("id")
        deadline = self.clock() + float(plan.get("duration_s") or 60.0) + MOVE_WAIT_EXTRA_S
        while True:
            if self._stop_evt.is_set():
                raise _Abort("stopped by the operator")
            try:
                st = self.jog.status()
            except Exception as e:  # noqa: BLE001
                self._emergency_stop()
                raise _Abort(f"jog state unknown ({e})")
            last = st.get("last") or {}
            if not (st.get("busy") or st.get("stopping")) and (sid is None or last.get("id") == sid):
                if last.get("phase") == "done" and last.get("ok"):
                    return
                raise _Abort(f"the move did not finish cleanly ({last.get('phase')}: {last.get('message')})")
            if st.get("fault"):
                raise _Abort("the jog controller went into FAULT")
            with self.lock:
                hb = self._hb
            if hb is not None and self.clock() - hb > HEARTBEAT_TIMEOUT_S:
                self._emergency_stop()
                raise _Abort("the page stopped polling (browser closed?): move stopped")
            if self.clock() > deadline:
                self._emergency_stop()
                raise _Abort("the move took too long: stopped")
            self.sleep(0.1)

    def _reserved(self, fn, epoch0) -> dict:
        """Run a motion request under the jog's ingress reservation (one queue slot, STOP epoch)."""
        res = {"ok": True}
        reserve = getattr(self.jog, "reserve_motion", None)
        token = None
        epoch = epoch0
        if reserve is not None:
            for _ in range(20):
                res = reserve()
                if res.get("ok"):
                    break
                self.sleep(0.25)
            if not res.get("ok"):
                raise _Abort("could not reserve the motion slot: " + str(res.get("message")))
            token, epoch = res.get("token"), res.get("epoch")
            if epoch0 is not None and epoch != epoch0:
                self._release(token)
                raise _Abort("a STOP was registered since the run started")
        try:
            out = fn(epoch)
        finally:
            self._release(token)
        return dict(out) if isinstance(out, dict) else {"ok": False, "message": "bad answer from the jog module"}

    def _release(self, token) -> None:
        rel = getattr(self.jog, "release_motion", None)
        if rel is not None and token is not None:
            try:
                rel(token)
            except Exception:  # noqa: BLE001
                pass

    def _emergency_stop(self) -> None:
        try:
            self.jog.stop()
        except Exception:  # noqa: BLE001
            self.log.exception("jog.stop failed")

    # ---- capture ----------------------------------------------------------------------------------------------
    def _capture_step(self, rid, page_id, epoch0) -> bool:
        """Settle, capture, record.  Returns True if a sample was stored; a refused capture is logged and returns False."""
        t_end = self.clock() + SETTLE_S
        while self.clock() < t_end:
            self._alive(rid, page_id, epoch0)
            self.sleep(0.1)
        for _ in range(100):             # up to ~10 s for the measured standstill (the jog's own gate used 1.2 s windows)
            self._alive(rid, page_id, epoch0)
            if self._standstill():
                break
            self.sleep(0.1)
        else:
            self._rlog("capture skipped: arm not at standstill")
            return False
        try:
            sample = self._capture_sample(lambda: self._alive(rid, page_id, epoch0))
        except WristRefused as e:
            self._rlog(f"capture skipped: {e.message}")
            return False
        self._store_sample(sample)
        return True

    def _capture_sample(self, check: Callable[[], None] = lambda: None) -> eih.WristSample:
        arm = self._need_arm()
        cam = self._wrist_camera()
        det, spec = self._table()
        t0 = self.clock()
        # Frames must have been CAPTURED (header stamp) after this point: a frame that was queued or processed late (callback
        # delay, slow driver) can show the arm in an earlier, moving state although it arrives after the standstill.
        now_fn = getattr(self.ros, "now_s", None)
        t0_ros = now_fn() if now_fn is not None else None
        got: List[dict] = []
        seen = set()
        while len(got) < N_FRAMES:
            check()
            if self.clock() - t0 > FRAME_WAIT_S:
                raise WristRefused(f"only {len(got)} fresh wrist frame(s) in {FRAME_WAIT_S:g} s: no wrist image"
                                   + (" (mock hardware has no wrist camera)" if not self._is_real() else ""), 503)
            fr = [f for f in self.ros.wrist_frames(t0) if f["seq"] not in seen and (t0_ros is None or f["stamp"] >= t0_ros)]
            for f in fr:
                if got and f["stamp"] - got[-1]["stamp"] < FRAME_GAP_S:
                    continue
                seen.add(f["seq"])
                got.append(f)
                if len(got) >= N_FRAMES:
                    break
            if len(got) < N_FRAMES:
                self.sleep(0.05)
        dets = []
        for f in got:
            if (f["bgr"].shape[1], f["bgr"].shape[0]) != tuple(cam["size"]):
                raise WristRefused("wrist image size != camera_info", 409)
            d = det.detect(f["bgr"]).get("table")
            if d is None or len(d.points) < eih.MIN_CORNERS:
                raise WristRefused(f"board not detected with >= {eih.MIN_CORNERS} corners "
                                   f"({0 if d is None else len(d.points)} found)", 409)
            dets.append(d)
        ok, txt = self._still_across(dets)
        if not ok:
            cen = [np.round(np.mean([p[2] for p in d.points], axis=0), 2).tolist() for d in dets]
            self.log.info("capture refused (%s); frames: %s", txt,
                          [(round(f["stamp"] - got[0]["stamp"], 3), round(f["rx"] - t0, 3), f.get("lag_s")) for f in got] + cen)
            raise WristRefused(txt, 409)
        # arm unmoved: TF at the first / middle / last frame stamp identical, and fresh
        Ts = []
        for f in (got[0], got[len(got) // 2], got[-1]):
            T, why = self.ros.tf_base_link6(f["stamp"])
            if T is None:
                raise WristRefused(why, 409)
            Ts.append(T)
        for T in Ts[1:]:
            dp = float(np.linalg.norm(T[:3, 3] - Ts[0][:3, 3]))
            dr = rotation_angle_deg(T[:3, :3], Ts[0][:3, :3])
            if dp > TF_STILL_POS_M or dr > TF_STILL_ROT_DEG:
                raise WristRefused(f"the arm moved during the capture ({dp * 1e3:.2f} mm / {dr:.3f} deg)", 409)
        age = self.ros.tf_age_s()
        if age is not None and age > TF_MAX_AGE_S:
            raise WristRefused(f"TF base_link->link6 is {age:.1f} s old", 409)
        # average the corners over the frames (ids present in all)
        common = sorted(set.intersection(*[{p[0] for p in d.points} for d in dets]))
        if len(common) < eih.MIN_CORNERS:
            raise WristRefused(f"only {len(common)} corners common to all frames", 409)
        by = [{p[0]: p for p in d.points} for d in dets]
        ids = np.array(common, dtype=int)
        obj = np.array([by[0][i][1] for i in common], dtype=float)
        pix = np.mean([[by[k][i][2] for i in common] for k in range(len(dets))], axis=0)
        s = eih.WristSample(index=len(self._samples), stamp=float(got[len(got) // 2]["stamp"]), T_base_link6=Ts[1], ids=ids,
                            obj=obj, pix=pix, K=cam["K"], dist=cam["dist"], image_size=cam["size"],
                            camera_frame_id=str(got[0].get("frame_id") or cam.get("frame_id", "")),
                            meta={"n_frames": len(got), "tf_age_s": age, "still": txt})
        s._bgr = got[len(got) // 2]["bgr"]          # noqa: SLF001  (kept only until stored)
        # reject a sample the board model cannot explain at all (mis-detection): PnP reprojection
        cands = eih.sample_pnp(s)
        if not cands or cands[0][1] > 2.0:
            raise WristRefused(f"PnP of the sample failed or reprojection {cands[0][1] if cands else float('nan'):.2f} px > 2", 409)
        s.meta["pnp_rms_px"] = float(cands[0][1])
        return s

    @staticmethod
    def _still_across(dets) -> Tuple[bool, str]:
        worst_c, worst_s = 0.0, 0.0
        for a, b in zip(dets[:-1], dets[1:]):
            c, sg, n = corner_motion_stats(a.points, b.points)
            ok, txt = judge_motion(c, sg, n, STILL_COHERENT_PX, STILL_SINGLE_PX, eih.MIN_CORNERS)
            if not ok:
                return False, txt
            worst_c, worst_s = max(worst_c, c), max(worst_s, sg)
        return True, f"still (common {worst_c:.2f} px, single {worst_s:.2f} px)"

    def _archive_old_samples(self, d: str) -> None:
        """A new UI run starts with an empty sample set: an existing wrist_samples.jsonl of an earlier run is renamed (kept)."""
        if self._archived:
            return
        self._archived = True
        path = os.path.join(d, eih.SAMPLES_FILE)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            dst = os.path.join(d, f"wrist_samples.{time.strftime('%Y%m%d_%H%M%S', time.localtime(os.path.getmtime(path)))}.jsonl")
            n = 0
            while os.path.exists(dst):
                n += 1
                dst = dst[:-6] + f"_{n}.jsonl"
            os.replace(path, dst)
            self.log.info("earlier samples archived as %s", dst)

    def _store_sample(self, s: eih.WristSample) -> None:
        d = self._out_dir()
        self._archive_old_samples(d)
        bgr = getattr(s, "_bgr", None)
        with self.lock:
            s.index = len(self._samples)
            s.image = f"images/wrist_{s.index:03d}.png"
        if bgr is not None:
            cv2.imwrite(os.path.join(d, s.image), bgr)
            del s._bgr
        eih.append_sample(d, s)
        with self.lock:
            self._samples.append(s)
            samples = list(self._samples)
        self._rlog(f"sample #{len(samples)} stored: {len(s.ids)} corners, PnP {s.meta.get('pnp_rms_px', float('nan')):.2f} px")
        if len(samples) >= eih.MIN_SOLVE_SAMPLES:
            est = eih.quick_estimate(samples)
            with self.lock:
                self._est = est
            if est is not None:
                self._rlog(f"interim solve: RMS {est.stats['rms_px']:.2f} px, span {est.stats['diversity']['span_deg']:.0f} deg")

    def capture_now(self) -> dict:
        """Manual capture at the current pose (no motion).  Refused while a sequence runs."""
        self._need_arm()
        with self.lock:
            if self._run["state"] in ("running", "stopping"):
                raise WristRefused("a sequence is running", 409)
        why = self._jog_blocked()
        if why:
            raise WristRefused("cannot capture now: " + why, 409)
        if not self._standstill():
            raise WristRefused("the arm is not at measured standstill", 409)
        self.sleep(SETTLE_S)
        s = self._capture_sample()
        self._store_sample(s)
        return {"ok": True, "message": f"sample #{len(self._samples)} stored ({len(s.ids)} corners)"}

    # ---- solve / outputs -------------------------------------------------------------------------------------
    def _scene_T(self) -> Optional[np.ndarray]:
        with self.lock:
            b = self._board
        return None if b is None else b["T"]

    def _final_solve(self) -> None:
        with self.lock:
            samples = list(self._samples)
        if len(samples) < eih.MIN_SOLVE_SAMPLES:
            self._rlog(f"{len(samples)} samples: too few to solve")
            return
        res = eih.solve(samples, self._scene_T())
        with self.lock:
            self._result, self._est = res, res
        self._rlog("result: " + res.gate["verdict"] + f" (RMS {res.stats['rms_px']:.2f} px)")

    def solve(self) -> dict:
        self._need_arm()
        with self.lock:
            n = len(self._samples)
        if n < eih.MIN_SOLVE_SAMPLES:
            raise WristRefused(f"{n} samples, need >= {eih.MIN_SOLVE_SAMPLES}", 409)
        try:
            self._final_solve()
        except eih.HandEyeError as e:
            raise WristRefused(f"solve failed: {e}", 409)
        return self.result_state()

    def _camera_link_optical(self, samples) -> Tuple[Optional[np.ndarray], str]:
        """T_wrist_camera_link_optical from the camera driver's TF (frame of the images); (None, reason) if unavailable."""
        frame = (samples[0].camera_frame_id if samples else "") or eih.OPTICAL_FRAME
        try:
            T, why = self.ros.tf_camera_link_optical(frame)
        except Exception as e:  # noqa: BLE001
            return None, f"TF {eih.CHILD_FRAME} -> {frame} lookup failed: {e}"
        return T, why

    def write(self) -> dict:
        arm = self._need_arm()
        with self.lock:
            res, samples = self._result, list(self._samples)
        if res is None:
            raise WristRefused("solve first", 409)
        cam = self._wrist_camera_or_none()
        camera = {"frame_id": samples[0].camera_frame_id, "serial": self.camera_serial or None,
                  "image_size": list(samples[0].image_size), "K": samples[0].K.tolist(), "dist": samples[0].dist.tolist()}
        if cam:
            camera["topic_frame_id"] = cam.get("frame_id")
        T_cl, why = self._camera_link_optical(samples)
        X_nom = wp.nominal_T_link6_optical(self.cad_roll_deg)
        files = eih.write_outputs(self._out_dir(), res, arm, samples, camera=camera, T_base_board_scene=self._scene_T(),
                                  T_camera_link_optical=T_cl,
                                  nominal={"T_link6_optical": X_nom.tolist(), "source": "CAD gemini_305_body.stl front face",
                                           "delta_to_result_mm_deg": list(eih.pose_delta(X_nom, res.X))})
        with self.lock:
            self._result_files = files
        msg = "written: " + ", ".join(files.values())
        if T_cl is None:
            msg += (f". The link6 -> {eih.CHILD_FRAME} joint / xacro snippet were NOT written (fail closed): {why}")
        return {"ok": True, "message": msg, "files": files}

    def _wrist_camera_or_none(self):
        try:
            return self.ros.wrist_info()
        except Exception:  # noqa: BLE001
            return None

    def publish_tf(self) -> dict:
        """Static TF link6 -> wrist_camera_link (live check only).  X is the optical-frame mount, so it is composed with the
        driver's wrist_camera_link -> optical transform; fail closed if that cannot be looked up.  The optical frame itself is
        never given a second parent."""
        self._need_arm()
        with self.lock:
            res, samples = self._result, list(self._samples)
        if res is None:
            raise WristRefused("solve first", 409)
        if res.gate["verdict"] == "FAIL":
            raise WristRefused("the result FAILED its gate: not publishing it as a TF", 409)
        T_cl, why = self._camera_link_optical(samples)
        T_att = eih.attach_transform(res.X, T_cl)
        if T_att is None:
            raise WristRefused(f"cannot publish: TF {eih.CHILD_FRAME} -> optical frame is not available ({why}); "
                               f"is the wrist camera driver running?", 503)
        self.ros.publish_static_tf(T_att)
        with self.lock:
            self._tf_published = True
        return {"ok": True, "message": f"static TF {eih.PARENT_FRAME} -> {eih.CHILD_FRAME} published (live check only; "
                                       f"restart the TF publisher/UI to remove it)"}

    def delete_sample(self, index) -> dict:
        self._need_arm()
        with self.lock:
            if self._run["state"] in ("running", "stopping"):
                raise WristRefused("a sequence is running", 409)
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(self._samples):
                raise WristRefused("index out of range", 400)
            del self._samples[index]
            for k, s in enumerate(self._samples):
                s.index = k
            samples = list(self._samples)
            self._result = self._est = None
        eih.rewrite_samples(self._out_dir(), samples)
        return {"ok": True, "message": f"sample {index} removed ({len(samples)} left; images stay on disk)"}

    # ---- views ---------------------------------------------------------------------------------------------------
    def result_state(self) -> Optional[dict]:
        with self.lock:
            r, files = self._result, dict(self._result_files)
        if r is None:
            return None
        h = r.stats.get("heldout") or {}
        return clean({"verdict": r.gate["verdict"], "reasons": r.gate["reasons"], "xyz_m": T_to_xyz_rpy(r.X)[0],
                      "rpy_deg": np.degrees(T_to_xyz_rpy(r.X)[1]), "rms_px": r.stats["rms_px"], "max_px": r.stats["max_px"],
                      "n_used": r.stats["n_used"], "n_rejected": r.stats["n_rejected"],
                      "heldout_position_mm": h.get("position_rms_mm"), "heldout_reproj_px": h.get("reproj_rms_px"),
                      "board_consistency_mm": (r.stats.get("board_consistency") or {}).get("position_rms_mm"),
                      "board_consistency_deg": (r.stats.get("board_consistency") or {}).get("rotation_rms_deg"),
                      "diversity": r.stats["diversity"], "scene_cross_check": r.stats.get("scene_cross_check"),
                      "init_method": r.method, "files": files})

    def _coverage(self) -> list:
        with self.lock:
            samples = list(self._samples)
        grid = np.zeros((4, 4), dtype=int)
        for s in samples:
            w, h = s.image_size
            x0, y0 = s.pix.min(axis=0)
            x1, y1 = s.pix.max(axis=0)
            c0, c1 = int(np.clip(x0 / w * 4, 0, 3.999)), int(np.clip(x1 / w * 4, 0, 3.999))
            r0, r1 = int(np.clip(y0 / h * 4, 0, 3.999)), int(np.clip(y1 / h * 4, 0, 3.999))
            grid[r0:r1 + 1, c0:c1 + 1] += 1
        return grid.tolist()

    def state(self) -> dict:
        if not self.supported:
            return {"supported": False, "message": "single-arm modes (single_left / single_right) only"}
        try:
            real = self._is_real()
        except WristRefused:
            real = True
        with self.lock:
            cs = dict(self._cand_state)
            cands = list(self._cands)
            run = dict(self._run, log=list(self._run["log"]))
            samples = list(self._samples)
            est = self._est
            tfp = self._tf_published
        items = [{"i": i, "desc": c.desc, "tilt_deg": c.tilt_deg, "az_deg": c.az_deg, "roll_deg": c.roll_deg,
                  "range_factor": c.dist_factor, "distance_m": c.info.get("distance_m"),
                  "max_joint_change_deg": c.info.get("max_joint_change_deg"), "seed_view": bool(c.info.get("seed_view"))}
                 for i, c in enumerate(cands)]
        try:
            X, src = self._current_X()
            xs = {"source": src, **_pose_summary(X)}
        except Exception:  # noqa: BLE001
            xs = None
        try:
            health = self.ros.wrist_health()
        except Exception:  # noqa: BLE001
            health = {}
        return clean({
            "supported": True, "arm": self.arm, "hardware_real": real, "allow_real_move": self.allow_real_move,
            "wrist": health, "board": self.board_state(), "scene": self.scene_state(), "seed": self.seed_state(),
            "estimate": xs,
            "views": {**cs, "items": items},
            "run": run,
            "samples": {"n": len(samples), "coverage": self._coverage(),
                        "items": [{"index": s.index, "n_corners": int(len(s.ids)), "stamp": s.stamp,
                                   "pnp_rms_px": s.meta.get("pnp_rms_px")} for s in samples],
                        "interim": None if est is None else est.summary()},
            "result": self.result_state(), "tf_published": tfp, "output_dir": self.out_dir_fn(),
        })

    # ---- live image ---------------------------------------------------------------------------------------------
    def render_jpeg(self, width: int = 960, quality: int = 80) -> Optional[bytes]:
        fr = self.ros.latest_wrist_frame()
        if fr is None:
            return None
        bgr = fr["bgr"]
        h0, w0 = bgr.shape[:2]
        dets = None
        seq = fr["seq"]
        if self._preview_cache[0] == seq:
            dets = self._preview_cache[1]
        else:
            try:
                det, _ = self._table()
                d = det.detect(bgr).get("table")
                dets = None if d is None else [(float(p[2][0]), float(p[2][1])) for p in d.points]
            except Exception:  # noqa: BLE001
                dets = None
            self._preview_cache = (seq, dets)
        s = min(width, w0) / w0
        img = cv2.resize(bgr, (int(w0 * s), int(h0 * s)), interpolation=cv2.INTER_AREA) if s != 1 else bgr.copy()
        n = 0
        for u, v in dets or []:
            cv2.circle(img, (int(round(u * s)), int(round(v * s))), 3, (0, 200, 0), -1, cv2.LINE_AA)
            n += 1
        cv2.putText(img, f"board corners: {n}" + (" (>=16 ok)" if n >= 16 else ""), (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 200, 0) if n >= 16 else (0, 140, 255), 2, cv2.LINE_AA)
        with self.lock:
            outline = self._next_outline
        if outline:
            pts = (np.asarray(outline, dtype=float).reshape(-1, 2) * s).round().astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], True, (255, 0, 255), 2, cv2.LINE_AA)
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        return buf.tobytes() if ok else None

    # ---- shutdown ----------------------------------------------------------------------------------------------------
    def shutdown(self) -> None:
        self._stop_evt.set()
        with self.lock:
            applied = self._scene is not None
        t = self._run_thread
        if t is not None and t.is_alive():
            t.join(timeout=3.0)
        if applied:
            try:
                self.ros.remove_boxes()
            except Exception as e:  # noqa: BLE001
                self.log.warning("safety scene not removed at shutdown: %s", e)
