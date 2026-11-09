#!/usr/bin/env python3
"""Small TEST of box-pose retargeting (joint-1 rotation) for the right-arm mocap VLA dataset, with a head-camera
synthesis prototype.  Prototype quality: look at the outputs, read retarget_test.md, do not scale up blindly.

Idea (EXACT retarget mode): the right arm's joint 1 axis is the base z axis.  Rotating the box about that axis by
d_theta and adding d_theta to joint 1 (everything else identical: same wrist branch, same timing) reproduces the
demonstration with a box that is rotated about the base.  The new joint-1 trajectory is

    j1_new(t) = j1(t) + d_theta * w(t),   w: 0 -> 1 (min-jerk) during the free-space approach, 1 afterwards,

so the arm is untouched while it sits at home and the whole demonstration after contact is rigidly rotated.
Because j1 is the base z axis, adding d to j1 pre-multiplies EVERY link pose by Rz(d): FK(q_new) = Rz(d) FK(q) exactly.

Sub commands (see ``--help``)::

    plate      per-pixel median head-camera background plate (arm and box excluded) -> scratch dir
    run        trajectories, wrist + head synthesis, videos, plots, contact sheet for the test episodes
    validate   synthesise one real episode from another real episode whose box differs by a rotation about base z
    dataset    write the tiny LeRobot v3.0 test dataset (2 source episodes x 2 d_theta = 4 episodes)

Heavy work runs under ``nice -n 19`` with at most 4 worker processes, one BLAS/OMP thread each, and no GPU.
"""
from __future__ import annotations

import argparse
import json
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))

SRC = "/media/farhan/My Passport/igus_nvidia/datasets/mocap_right_31oct_clean_v1"
OUT_DS = "/media/farhan/My Passport/igus_nvidia/datasets/mocap_right_31oct_retarget_test_v1"
SCRATCH = Path("/media/farhan/My Passport/igus_nvidia/tmp_retarget")
REPORT = REPO / "scan_results" / "dataset_augment_20261031" / "retarget_test"
URDF = REPO / "scan_results" / "jtc_settings_dual_20261031" / "dual_mock_hardware.urdf"
MESH_ROOT = REPO / "src" / "igus_rebel_description"
ARM = "arm_right"
FPS = 30.0
VMAX = 0.785                        # rad/s (joint_limits.yaml)
STEP_MAX = np.radians(5.7)          # rad / frame (= VMAX/FPS*... plus margin, the task's bound)
LIMIT_INSET = np.radians(2.0)
TEST_EPISODES = ("mocap_right_2_ep001", "mocap_right_3_ep002")
DELTAS_DEG = (-4.0, 4.0)

# head camera: camera_link pose in world (src/rebel_dual_table_description/config/dual_arm_calibration.yaml:8-10) and
# right base in world (same file); intrinsics ASSUMED (1080p K of ~/dual_arm_calibration/intrinsics_report.json
# centre-cropped to 4:3 and scaled by 1/2.25 to 640x480).
CAM_XYZ = np.array([-0.048397427117980334, -0.20788777061552297, 0.8681737492514855])
CAM_RPY = np.array([0.03437519478050081, 0.8025743383544222, 0.010918217808845202])
BASE_XYZ = np.array([-8.357992070818208e-05, -0.4397165297206802, 0.0046762267300018028])
BASE_RPY = np.array([-0.01822495461019838, 0.008429085689667826, 0.04908320092978558])
_K = 640 / 1440.0
HEAD_K = np.array([[1051.84 * _K, 0, (979.89 - 240) * _K], [0, 1048.51 * _K, 541.49 * _K], [0, 0, 1.0]])
OPT_FROM_LINK = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], float).T   # camera_link (x fwd) -> optical (z fwd): columns = optical axes in link
H, W = 480, 640


def rotz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def rpy_matrix(rpy) -> np.ndarray:
    from scipy.spatial.transform import Rotation as Rot
    return Rot.from_euler("xyz", rpy).as_matrix()


def T_world_base() -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rpy_matrix(BASE_RPY)
    T[:3, 3] = BASE_XYZ
    return T


def T_world_cam_optical() -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rpy_matrix(CAM_RPY) @ OPT_FROM_LINK
    T[:3, 3] = CAM_XYZ
    return T


def T_base_cam_optical() -> np.ndarray:
    """Pose of the head-camera optical frame in the right-arm base frame (z of the base = joint-1 axis)."""
    return np.linalg.inv(T_world_base()) @ T_world_cam_optical()


# ------------------------------------------------------------------------------------------ kinematics (numpy chain)
class Chain:
    """Revolute chain base_link -> link6 of the right arm from the URDF (own numpy FK, no yourdfpy per-frame cost)."""

    def __init__(self, urdf: Path = URDF):
        import yourdfpy
        u = yourdfpy.URDF.load(str(urdf), load_meshes=False, build_collision_scene_graph=False)
        chain, lnk = [], f"{ARM}_link6"
        while lnk != f"{ARM}_base_link":
            j = [j for j in u.robot.joints if j.child == lnk][0]
            chain.append(j)
            lnk = j.parent
        self.joints = chain[::-1]
        rev = [j for j in self.joints if j.type == "revolute"]
        self.lo = np.array([j.limit.lower for j in rev])
        self.hi = np.array([j.limit.upper for j in rev])
        self.vel = np.array([j.limit.velocity for j in rev])
        self.origins = [np.eye(4) if j.origin is None else np.array(j.origin, float) for j in self.joints]
        self.axes = [None if j.type != "revolute" else np.array(j.axis, float) for j in self.joints]

    @staticmethod
    def _rot(ax, a):
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K

    def fk(self, q: np.ndarray) -> np.ndarray:
        """4x4 pose of link6 in the arm base frame for one 6-vector."""
        T, k = np.eye(4), 0
        for o, ax in zip(self.origins, self.axes):
            T = T @ o
            if ax is not None:
                A = np.eye(4)
                A[:3, :3] = self._rot(ax, q[k])
                T = T @ A
                k += 1
        return T

    def fk_series(self, Q: np.ndarray) -> np.ndarray:
        return np.stack([self.fk(q) for q in Q])


# ------------------------------------------------------------------------------------------ trajectory retarget
def minjerk(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return 10 * x ** 3 - 15 * x ** 4 + 6 * x ** 5


def ramp_weights(n: int, start: int, end: int) -> np.ndarray:
    """w(t): 0 before ``start``, min-jerk 0 -> 1 on [start, end], 1 afterwards."""
    t = np.arange(n, dtype=float)
    if end <= start:
        return (t >= end).astype(float)
    return minjerk((t - start) / (end - start))


def detect_phases(q: np.ndarray, chain: Chain, fps: float = FPS) -> dict:
    """Free-space approach window [ramp_start, ramp_end] of an (N, 6) filtered right-arm joint series.

    Heuristic (documented in retarget_test.md):
      * first_move / last_move from ``episode_metrics.motion_segments`` (the home idle ends at first_move).
      * tcp = link6 position in the arm base frame (FK).  z_lo = 5th percentile of z over the moving part; the
        *contact onset* is the first frame after first_move where z <= z_lo + 0.05 m (the tool has come down to the
        working height of the front-flap cut; everything before is the free-space approach).
      * ramp_end   = onset - 30 frames (1 s margin: the arm must not touch the box while it is still blending).
      * ramp_start = first frame (>= first_move) after the home wrist counter-rotation has settled, i.e. the first
        frame where |dj4|+|dj6| speed has fallen below 0.15 rad/s after having peaked, or first_move + 60 if
        that never happens; at least 90 frames of ramp are enforced.
    """
    import episode_metrics as em
    n = len(q)
    seg = em.motion_segments(q, fps)
    fm, lm = int(seg["first_move"]), int(seg["last_move"])
    P = chain.fk_series(q)[:, :3, 3]
    z = P[:, 2]
    zlo = float(np.percentile(z[fm:lm + 1], 5))
    below = np.flatnonzero((z <= zlo + 0.05) & (np.arange(n) >= fm))
    onset = int(below[0]) if len(below) else lm
    ramp_end = max(onset - 30, fm + 90)
    v = np.abs(np.gradient(q[:, 3], 1 / fps)) + np.abs(np.gradient(q[:, 5], 1 / fps))
    vs = np.convolve(v, np.ones(15) / 15, mode="same")
    pk = fm + int(np.argmax(vs[fm:fm + 240]))
    after = np.flatnonzero((vs[pk:] < 0.15))
    ramp_start = pk + int(after[0]) if len(after) else fm + 60
    ramp_start = int(min(max(ramp_start, fm), ramp_end - 90))
    return {"first_move": fm, "last_move": lm, "contact_onset": onset, "z_lo": zlo,
            "ramp_start": ramp_start, "ramp_end": int(ramp_end)}


def retarget_q(q: np.ndarray, dtheta: float, w: np.ndarray) -> np.ndarray:
    """q: (N, 6) -> q with joint 1 += dtheta * w (all other joints identical)."""
    out = np.array(q, float, copy=True)
    out[:, 0] += dtheta * w
    return out


def next_state_action(s: np.ndarray) -> np.ndarray:
    """action[t] = state[t+1] (last row repeats), like the source dataset (action_offset_frames = 1)."""
    a = np.empty_like(s)
    a[:-1] = s[1:]
    a[-1] = s[-1]
    return a


def check_q(q: np.ndarray, chain: Chain, fps: float = FPS) -> dict:
    """Limits (URDF with 2 deg inset), speed and per-frame step of an (N, 6) series."""
    dq = np.diff(q, axis=0)
    viol_lo = (q < chain.lo + LIMIT_INSET)
    viol_hi = (q > chain.hi - LIMIT_INSET)
    return {"min_margin_deg": float(np.degrees(np.minimum(q - chain.lo, chain.hi - q).min())),
            "limit_violations": int(viol_lo.sum() + viol_hi.sum()),
            "max_speed_rad_s": float(np.abs(dq).max() * fps), "max_step_deg": float(np.degrees(np.abs(dq).max())),
            "speed_ok": bool(np.abs(dq).max() * fps <= VMAX + 1e-9), "step_ok": bool(np.abs(dq).max() <= STEP_MAX),
            "limits_ok": bool(not (viol_lo.any() or viol_hi.any()))}


# ------------------------------------------------------------------------------------------ video / dataset reading
def read_source(root: str = SRC):
    """(info, per-episode dict list) of the LeRobot v3.0 source: state (N,14), action, episode rows, source names."""
    import pandas as pd
    import pyarrow.parquet as pq
    root_p = Path(root)
    info = json.loads((root_p / "meta" / "info.json").read_text())
    df = pd.concat([pd.read_parquet(f) for f in sorted((root_p / "data").rglob("*.parquet"))], ignore_index=True)
    rows = [r for f in sorted((root_p / "meta" / "episodes").rglob("*.parquet")) for r in pq.read_table(f).to_pylist()]
    rows.sort(key=lambda r: r["episode_index"])
    smap = json.loads((root_p / "meta" / "source_episodes.json").read_text())
    return info, df, rows, smap


def episode_index_of(name: str, smap) -> int:
    for r in smap:
        if r["source"] == name:
            return int(r["episode_index"])
    raise KeyError(name)


def episode_state(df, ep: int) -> dict:
    d = df[df["episode_index"] == ep].sort_values("frame_index")
    return {k: np.stack(d[k].to_numpy()).astype(np.float64) for k in ("observation.state", "action", "observation.state_raw", "action_teleop")
            if k in d.columns} | {"n": len(d)}


def video_path(root: str, cam: str, row: dict) -> tuple[str, float]:
    key = f"observation.images.{cam}"
    return (f"{root}/videos/{key}/chunk-{row[f'videos/{key}/chunk_index']:03d}/file-{row[f'videos/{key}/file_index']:03d}.mp4",
            float(row[f"videos/{key}/from_timestamp"]))


def read_video(root: str, cam: str, row: dict, indices: list[int]) -> dict[int, np.ndarray]:
    import video_frames as vf
    p, t0 = video_path(root, cam, row)
    return vf.read_frames_by_index(p, indices, FPS, t0, threads=1)


# ------------------------------------------------------------------------------------------ arm mesh renderer (embree)
class ArmRenderer:
    """Raycast depth + link id of the URDF right-arm meshes seen from the head camera (half resolution, step 2)."""

    def __init__(self, step: int = 2):
        import trimesh
        import yourdfpy
        fh = lambda f: f.replace("package://igus_rebel_description", str(MESH_ROOT))  # noqa: E731
        self.u = yourdfpy.URDF.load(str(URDF), filename_handler=fh, load_meshes=False, build_collision_scene_graph=False)
        self.parts = []
        for ln, L in self.u.link_map.items():
            if not ln.startswith(ARM):
                continue
            for v in L.visuals:
                if v.geometry.mesh is None:
                    continue
                m = trimesh.load(fh(v.geometry.mesh.filename), force="mesh")
                if v.geometry.mesh.scale is not None:
                    m.apply_scale(v.geometry.mesh.scale)
                if v.origin is not None:
                    m.apply_transform(v.origin)
                self.parts.append((ln, m))
        self.links = sorted({p[0] for p in self.parts})
        self.step = step
        K, Tbc = HEAD_K, T_base_cam_optical()
        self.Tbc = Tbc
        vv, uu = np.mgrid[0:H:step, 0:W:step]
        d = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones_like(uu, float)], -1).reshape(-1, 3)
        self.dirs_cam = d                      # not normalised: z = 1, so the hit distance along it IS the optical depth
        self.dirs = d @ Tbc[:3, :3].T
        self.shape = uu.shape
        # links that do not move when joint 1 turns (so keep their pixels when the rest rotates)
        self.u.update_cfg({f"{ARM}_joint{i}": 0.0 for i in range(1, 7)})
        T0 = {ln: self.u.get_transform(ln, f"{ARM}_base_link") for ln in self.links}
        self.u.update_cfg({f"{ARM}_joint1": 0.4, **{f"{ARM}_joint{i}": 0.0 for i in range(2, 7)}})
        self.static = {i for i, ln in enumerate(self.links) if np.abs(self.u.get_transform(ln, f"{ARM}_base_link") - T0[ln]).max() < 1e-9}

    def render(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(depth float32 (h,w) optical z in m, inf = none; link id int8 (h,w), -1 = none) at 1/step resolution."""
        import trimesh
        self.u.update_cfg({f"{ARM}_joint{i + 1}": float(q[i]) for i in range(6)})
        ms, ids = [], []
        for ln, m in self.parts:
            mm = m.copy()
            mm.apply_transform(self.u.get_transform(ln, f"{ARM}_base_link"))
            ms.append(mm)
            ids += [self.links.index(ln)] * len(mm.faces)
        sc = trimesh.util.concatenate(ms)
        ids = np.array(ids, np.int8)
        org = np.repeat(self.Tbc[None, :3, 3], len(self.dirs), 0)
        tri, ray, loc = sc.ray.intersects_id(org, self.dirs, multiple_hits=False, return_locations=True)
        depth = np.full(len(self.dirs), np.inf, np.float32)
        lid = np.full(len(self.dirs), -1, np.int8)
        dist = np.linalg.norm(loc - self.Tbc[:3, 3], axis=1)                      # distance along the unit-normalised ray
        zc = (loc - self.Tbc[:3, 3]) @ self.Tbc[:3, :3]                             # points in the camera frame
        depth[ray] = zc[:, 2]
        lid[ray] = ids[tri]
        del dist
        return depth.reshape(self.shape), lid.reshape(self.shape)


def up2(a: np.ndarray, step: int, interp=None) -> np.ndarray:
    import cv2
    return cv2.resize(a, (W, H), interpolation=cv2.INTER_NEAREST if interp is None else interp)


# ------------------------------------------------------------------------------------------ simple image segmentation
BOX_ROI = (110, 250, 500, 480)       # x0, y0, x1, y1 where the carton (and lifted flaps) can appear in the head image


def cardboard_mask(rgb: np.ndarray) -> np.ndarray:
    """Tan cardboard pixels (HSV threshold) inside BOX_ROI, bool (H, W)."""
    import cv2
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    m = (hsv[..., 0] >= 5) & (hsv[..., 0] <= 28) & (hsv[..., 1] >= 45) & (hsv[..., 1] <= 165) & (hsv[..., 2] >= 90)
    roi = np.zeros_like(m)
    x0, y0, x1, y1 = BOX_ROI
    roi[y0:y1, x0:x1] = True
    return m & roi


def box_region(rgb: np.ndarray, margin: int = 22) -> np.ndarray:
    """Generous rectangle around the carton (bounding box of the large cardboard blobs in BOX_ROI, + margin), bool (H, W)."""
    import cv2
    out = np.zeros((H, W), bool)
    m = cv2.morphologyEx(cardboard_mask(rgb).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8)) > 0
    m = keep_large(m, 500)
    ys, xs = np.nonzero(m)
    if len(ys):
        out[max(ys.min() - margin, 0):ys.max() + margin, max(xs.min() - margin, 0):xs.max() + margin] = True
    return out


def dilate(m: np.ndarray, r: int) -> np.ndarray:
    import cv2
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(m.astype(np.uint8), k) > 0


def erode(m: np.ndarray, r: int) -> np.ndarray:
    import cv2
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.erode(m.astype(np.uint8), k) > 0


# ------------------------------------------------------------------------------------------ background plate
PLATE_STEP = 25
PLATE_EPISODES = tuple(range(21))     # episodes 21, 22 have a person in the background


def _worker_init():
    os.nice(19)


def _plate_worker(args):
    ep, scratch = args
    info, df, rows, smap = read_source()
    st = episode_state(df, ep)
    q = st["observation.state"][:, 7:13]
    idx = list(range(0, st["n"], PLATE_STEP))
    fr = read_video(SRC, "head", rows[ep], idx)
    rend = ArmRenderer()
    frames = np.stack([fr[i] for i in idx])
    excl = np.zeros((len(idx), H, W), bool)
    for k, i in enumerate(idx):
        depth, lid = rend.render(q[i])
        arm = up2((lid >= 0).astype(np.uint8), rend.step) > 0
        excl[k] = dilate(arm, 14)
    np.savez(Path(scratch) / f"plate_src_ep{ep:02d}.npz", frames=frames, excl=excl, idx=np.array(idx))
    return ep, len(idx)


def build_plate(workers: int = 4) -> Path:
    import cv2
    from concurrent.futures import ProcessPoolExecutor
    SCRATCH.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=min(workers, 4), initializer=_worker_init) as ex:
        for ep, n in ex.map(_plate_worker, [(e, str(SCRATCH)) for e in PLATE_EPISODES]):
            print(f"plate source ep{ep}: {n} frames", flush=True)
    frames, excl = [], []
    for ep in PLATE_EPISODES:
        z = np.load(SCRATCH / f"plate_src_ep{ep:02d}.npz")
        frames.append(z["frames"])
        excl.append(z["excl"] | np.stack([box_region(f) for f in z["frames"]]))
    frames = np.concatenate(frames)
    excl = np.concatenate(excl)
    plate = np.zeros((H, W, 3), np.uint8)
    count = np.zeros((H, W), np.int32)
    for y0 in range(0, H, 40):
        f = frames[:, y0:y0 + 40].astype(np.float32)
        ok = ~excl[:, y0:y0 + 40]
        f[~ok] = np.nan
        with np.errstate(all="ignore"):
            med = np.nanmedian(f, axis=0)
        plate[y0:y0 + 40] = np.nan_to_num(med, nan=0).astype(np.uint8)
        count[y0:y0 + 40] = ok.sum(0)
    return refill_plate(plate, count)


def refill_plate(raw: np.ndarray, count: np.ndarray, min_count: int = 40) -> Path:
    """Inpaint the pixels seen (without arm / carton) in fewer than ``min_count`` of the ~1100 sampled frames."""
    import cv2
    hole = dilate(count < min_count, 2)
    bgr = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
    filled = cv2.inpaint(bgr, hole.astype(np.uint8) * 255, 15, cv2.INPAINT_TELEA)
    plate_f = cv2.cvtColor(filled, cv2.COLOR_BGR2RGB)
    np.savez(SCRATCH / "plate.npz", plate=plate_f, raw=raw, count=count, hole=hole)
    cv2.imwrite(str(SCRATCH / "plate.png"), filled)
    cv2.imwrite(str(SCRATCH / "plate_holes.png"), hole.astype(np.uint8) * 255)
    print("plate holes (rarely observed):", int(hole.sum()), "px")
    return SCRATCH / "plate.npz"


# ------------------------------------------------------------------------------------------ head-camera synthesis
DIFF_THR = 30          # |frame - plate| (max over RGB, after a 5x5 blur) that counts as foreground
BOX_DIFF_THR = 22


def fill_holes(m: np.ndarray) -> np.ndarray:
    import cv2
    inv = (~m).astype(np.uint8) * 255
    ff = inv.copy()
    msk = np.zeros((H + 2, W + 2), np.uint8)
    cv2.floodFill(ff, msk, (0, 0), 0)
    return m | (ff > 0)


def keep_large(m: np.ndarray, min_area: int) -> np.ndarray:
    import cv2
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[lab]


def nearest_fill(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Replace values of ~valid pixels by the value of the nearest valid pixel."""
    from scipy import ndimage
    if not valid.any():
        return values
    _, (iy, ix) = ndimage.distance_transform_edt(~valid, return_indices=True)
    return values[iy, ix]


class HeadSynth:
    """Layered rigid-rotation synthesis of the head image for a joint-1 retarget.

    Layers: static scene (real frame with the rotating foreground replaced by the background plate), arm (links that
    turn with joint 1, mesh-render depth, rotated by ``d_arm`` about the base z axis) and carton (plane depth, rotated
    by ``d_box``).  Foreground pixels are lifted to 3-D, rotated, re-projected with a z-buffer, cracks are filled from
    the neighbouring pixel and the result is sampled bilinearly from the source frame."""

    def __init__(self, plate: np.ndarray, rend: ArmRenderer, box_z: float):
        self.plate = plate
        self.rend = rend
        self.K = HEAD_K.copy()
        self.Tbc = rend.Tbc
        self.box_z = box_z                      # height of the carton-top plane in the arm base frame [m]
        self.tool_ids = [i for i, l in enumerate(rend.links) if "knife" in l or l.endswith("link6") or l.endswith("link5")]
        self.static_ids = rend.static
        self.dy, self.dx = np.mgrid[0:H, 0:W]
        self.ray_cam = np.stack([(self.dx - self.K[0, 2]) / self.K[0, 0], (self.dy - self.K[1, 2]) / self.K[1, 1],
                                 np.ones((H, W))], -1).astype(np.float32)

    # ---- segmentation
    def box_depth(self) -> np.ndarray:
        """Optical depth of the camera ray hitting the horizontal plane z = box_z of the base frame (inf if none)."""
        R, t = self.Tbc[:3, :3], self.Tbc[:3, 3]
        dirb = self.ray_cam @ R.T.astype(np.float32)
        with np.errstate(all="ignore"):
            s = (self.box_z - t[2]) / dirb[..., 2]
        return np.where(s > 0, s, np.inf).astype(np.float32)

    def segment(self, rgb: np.ndarray, q: np.ndarray, gain=None) -> dict:
        import cv2
        d_h, l_h = self.rend.render(q)
        lid = cv2.resize((l_h + 1).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(np.int16) - 1
        depth = cv2.resize(np.where(np.isfinite(d_h), d_h, 0).astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
        render = lid >= 0
        tool = np.isin(lid, self.tool_ids)
        plate = self.plate_for(rgb, render | cardboard_mask(rgb)) if gain is None else self.plate_gain(gain)
        a = cv2.GaussianBlur(rgb, (5, 5), 0).astype(np.int16)
        b = cv2.GaussianBlur(plate, (5, 5), 0).astype(np.int16)
        diff = np.abs(a - b).max(-1)
        cb = cardboard_mask(rgb)
        core = erode(render, 3)
        band = dilate(render & ~tool, 8) | dilate(tool, 26)
        arm = core | (band & (diff > DIFF_THR) & ~cb)
        arm = cv2.morphologyEx(arm.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
        arm = cv2.morphologyEx(arm.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0
        arm = keep_large(arm, 200)
        lid_f = nearest_fill(lid, render)
        depth_f = nearest_fill(depth, render & (depth > 0))
        box = cb & (diff > BOX_DIFF_THR) & ~dilate(arm, 2)
        box = cv2.morphologyEx(box.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))) > 0
        box = keep_large(fill_holes(keep_large(box, 300)), 600) & ~arm
        rot = arm & ~np.isin(lid_f, list(self.static_ids))
        return {"arm": arm, "arm_rot": rot, "arm_depth": depth_f, "lid": lid_f, "box": box, "render": render, "diff": diff,
                "plate": plate}

    def plate_for(self, rgb: np.ndarray, fg: np.ndarray) -> np.ndarray:
        """Background plate with a per-channel gain/offset fitted to the visible background of this frame."""
        m = ~dilate(fg, 12)
        m[250:] = False                                    # upper image only: static scene, no carton
        m[:, 500:] = False
        out = self.plate.astype(np.float32)
        sel = m & (np.abs(rgb.astype(np.int16) - self.plate.astype(np.int16)).max(-1) < 60)
        if sel.sum() > 2000:
            for c in range(3):
                x, y = self.plate[..., c][sel].astype(np.float32), rgb[..., c][sel].astype(np.float32)
                g = float(np.clip((y * x).mean() / max((x * x).mean(), 1.0), 0.8, 1.25))
                out[..., c] *= g
        return np.clip(out, 0, 255).astype(np.uint8)

    # ---- warp
    def lift(self, mask: np.ndarray, depth: np.ndarray) -> np.ndarray:
        ys, xs = np.nonzero(mask)
        z = depth[ys, xs]
        ok = np.isfinite(z) & (z > 0.05)
        ys, xs, z = ys[ok], xs[ok], z[ok]
        Xc = np.stack([(xs - self.K[0, 2]) / self.K[0, 0] * z, (ys - self.K[1, 2]) / self.K[1, 1] * z, z], 1)
        return np.concatenate([xs[:, None], ys[:, None], Xc @ self.Tbc[:3, :3].T + self.Tbc[:3, 3]], 1)   # (n, 5): u, v, X_base

    def project(self, Xb: np.ndarray) -> np.ndarray:
        Xc = (Xb - self.Tbc[:3, 3]) @ self.Tbc[:3, :3]
        z = Xc[:, 2]
        return np.stack([self.K[0, 0] * Xc[:, 0] / z + self.K[0, 2], self.K[1, 1] * Xc[:, 1] / z + self.K[1, 2], z], 1)

    def warp_layers(self, layers: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        """layers: dicts(img, mask, depth, angle).  Returns (warped RGB float32 (H,W,3), alpha float32 (H,W))."""
        import cv2
        from scipy import ndimage
        pts, lay = [], []
        for li, L in enumerate(layers):
            P = self.lift(L["mask"], L["depth"])
            c, s = np.cos(L["angle"]), np.sin(L["angle"])
            Xb = P[:, 2:5].copy()
            Xr = np.stack([c * Xb[:, 0] - s * Xb[:, 1], s * Xb[:, 0] + c * Xb[:, 1], Xb[:, 2]], 1)
            uvz = self.project(Xr)
            pts.append(np.concatenate([P[:, :2], uvz], 1))
            lay.append(np.full(len(P), li, np.int16))
        pts = np.concatenate(pts)
        lay = np.concatenate(lay)
        xd, yd = np.rint(pts[:, 2]).astype(int), np.rint(pts[:, 3]).astype(int)
        ok = (xd >= 0) & (xd < W) & (yd >= 0) & (yd < H) & (pts[:, 4] > 0)
        pts, lay, xd, yd = pts[ok], lay[ok], xd[ok], yd[ok]
        order = np.argsort(-pts[:, 4], kind="stable")        # far first, near overwrites
        pts, lay, xd, yd = pts[order], lay[order], xd[order], yd[order]
        valid = np.zeros((H, W), bool)
        sx = np.zeros((H, W), np.float32)
        sy = np.zeros((H, W), np.float32)
        lm = np.full((H, W), -1, np.int16)
        valid[yd, xd] = True
        sx[yd, xd] = pts[:, 0] + (xd - pts[:, 2])
        sy[yd, xd] = pts[:, 1] + (yd - pts[:, 3])
        lm[yd, xd] = lay
        # z-buffer correctness for duplicate destinations: numpy fancy assignment keeps the last (= nearest) write
        closed = cv2.morphologyEx(valid.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
        crack = closed & ~valid
        if crack.any():
            _, (iy, ix) = ndimage.distance_transform_edt(~valid, return_indices=True)
            cy, cx = np.nonzero(crack)
            sx[cy, cx] = sx[iy[cy, cx], ix[cy, cx]] + (cx - ix[cy, cx])
            sy[cy, cx] = sy[iy[cy, cx], ix[cy, cx]] + (cy - iy[cy, cx])
            lm[cy, cx] = lm[iy[cy, cx], ix[cy, cx]]
        cover = valid | crack
        out = np.zeros((H, W, 3), np.float32)
        for li, L in enumerate(layers):
            sel = cover & (lm == li)
            if not sel.any():
                continue
            smp = cv2.remap(L["img"], sx, sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
            out[sel] = smp[sel]
        alpha = cv2.GaussianBlur(erode(cover, 1).astype(np.float32), (0, 0), 0.8)
        return out, alpha

    def synth(self, rgb: np.ndarray, q: np.ndarray, d_arm: float, d_box: float, box_mem: dict | None = None,
              seg: dict | None = None) -> tuple[np.ndarray, dict]:
        """Synthetic head frame for a source frame ``rgb`` with right-arm joints ``q``.

        d_arm / d_box: rotation of the arm / of the carton about the base z axis [rad].  ``box_mem`` (dict img, mask)
        replaces the per-frame carton layer by the clean carton of an early frame (used while the arm is still blending)."""
        seg = seg or self.segment(rgb, q)
        if abs(d_arm) < 1e-12 and abs(d_box) < 1e-12:
            return rgb.copy(), seg
        box_depth = self.box_depth()
        arm_only = seg["arm_rot"]
        layers = [{"img": rgb, "mask": arm_only, "depth": seg["arm_depth"], "angle": d_arm}]
        if box_mem is not None:
            layers.append({"img": box_mem["img"], "mask": box_mem["mask"] & ~arm_only, "depth": box_depth, "angle": d_box})
        else:
            layers.append({"img": rgb, "mask": seg["box"] & ~arm_only, "depth": box_depth, "angle": d_box})
        warped, alpha = self.warp_layers(layers)
        old = dilate(arm_only | seg["box"], 3)
        base = rgb.astype(np.float32).copy()
        base[old] = seg["plate"][old]
        a = alpha[..., None]
        out = base * (1 - a) + warped * a
        return np.clip(out + 0.5, 0, 255).astype(np.uint8), seg


# ------------------------------------------------------------------------------------------ wrist camera
# ASSUMPTIONS (no wrist-camera calibration exists for the gripper camera): pinhole, hfov ~70 deg (fx = fy = 460 px,
# centre of the 640x480 image), optical axis = link6 +z, image x/y = link6 x/y.
WRIST_K = np.array([[460.0, 0, 320.0], [0, 460.0, 240.0], [0, 0, 1.0]])


def wrist_homography(T_link6: np.ndarray, eps: float) -> np.ndarray:
    """H = K R K^-1 for the apparent rotation of the scene about the base z axis by ``eps`` [rad], seen from a camera whose
    orientation (in the base frame) is that of link6.

    Retarget: the carton is rotated by d_theta from the start but the arm only by d_theta * w(t), so relative to the
    camera the scene is rotated by eps = d_theta * (1 - w) about base z.  X_new = C^T Rz(eps) C X_old.  Translation of
    the camera (the camera sits ~0.4 m from the z axis) is NOT modelled: only exact at eps = 0 (w = 1)."""
    R = T_link6[:3, :3]
    Rc = R.T @ rotz(eps) @ R
    return WRIST_K @ Rc @ np.linalg.inv(WRIST_K)


def warp_wrist(img: np.ndarray, T_link6: np.ndarray, eps: float) -> np.ndarray:
    import cv2
    if abs(eps) < 1e-9:
        return img
    return cv2.warpPerspective(img, wrist_homography(T_link6, eps), (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


# ------------------------------------------------------------------------------------------ metrics + validation pair
def to_gray(rgb: np.ndarray) -> np.ndarray:
    import cv2
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float64)


def ssim_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Gaussian-window SSIM map (sigma 1.5, 11x11, K1=.01, K2=.03) of two uint8 RGB images (computed on grey)."""
    import cv2
    x, y = to_gray(a), to_gray(b)
    g = lambda z: cv2.GaussianBlur(z, (11, 11), 1.5)  # noqa: E731
    mx, my = g(x), g(y)
    sxx, syy, sxy = g(x * x) - mx * mx, g(y * y) - my * my, g(x * y) - mx * my
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    return ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))


def masked_metrics(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> dict:
    """PSNR [dB] (RGB) and mean SSIM over ``mask`` (SSIM mask eroded by 5 px so the window sees real content)."""
    if mask.sum() == 0:
        return {"psnr": float("nan"), "ssim": float("nan"), "px": 0}
    mse = float(np.mean((a[mask].astype(np.float64) - b[mask].astype(np.float64)) ** 2))
    sm = ssim_map(a, b)
    me = erode(mask, 5)
    return {"psnr": float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse),
            "ssim": float(sm[me].mean()) if me.any() else float("nan"), "px": int(mask.sum())}


def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else float("nan")


def rotate_mask(synth: "HeadSynth", mask: np.ndarray, ang: float) -> np.ndarray:
    """Mask of the box-plane pixels ``mask`` after rotating the plane about base z by ``ang``."""
    import cv2
    P = synth.lift(mask, synth.box_depth())
    c, s = np.cos(ang), np.sin(ang)
    Xb = P[:, 2:5]
    Xr = np.stack([c * Xb[:, 0] - s * Xb[:, 1], s * Xb[:, 0] + c * Xb[:, 1], Xb[:, 2]], 1)
    uv = synth.project(Xr)
    out = np.zeros((H, W), np.uint8)
    xi, yi = np.rint(uv[:, 0]).astype(int), np.rint(uv[:, 1]).astype(int)
    ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
    out[yi[ok], xi[ok]] = 1
    return cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0


def find_box_rotation_pairs(synth: "HeadSynth", first_frames: dict, angles_deg=np.arange(-8.0, 8.01, 0.5)) -> list[dict]:
    """For every ordered pair of episodes, the rotation about base z that best maps the carton mask of A onto that of B.

    first_frames: {ep: rgb at frame 0 (arm at home, carton untouched)}.  Score = IoU of the rotated plane mask vs B's mask."""
    import cv2
    masks = {}
    for ep, f in first_frames.items():
        m = cv2.morphologyEx(cardboard_mask(f).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8)) > 0
        masks[ep] = fill_holes(keep_large(m, 800))
    res = []
    for a in masks:
        rot = {float(d): rotate_mask(synth, masks[a], np.radians(d)) for d in angles_deg}
        for b in masks:
            if a == b:
                continue
            sc = {d: iou(m, masks[b]) for d, m in rot.items()}
            d_best = max(sc, key=sc.get)
            res.append({"a": a, "b": b, "deg": d_best, "iou_best": sc[d_best], "iou_zero": sc[0.0]})
    return res


# ------------------------------------------------------------------------------------------ episode driver
BOX_Z = 0.10      # carton-top plane height in the arm base frame [m]; see retarget_test.md (assumption)


def load_plate() -> np.ndarray:
    return np.load(SCRATCH / "plate.npz")["plate"]


def episode_plan(name: str, chain: "Chain", df, smap, deltas_deg=DELTAS_DEG, min_ramp: int = 120) -> dict:
    ep = episode_index_of(name, smap)
    st = episode_state(df, ep)
    q = st["observation.state"][:, 7:13]
    ph = detect_phases(q, chain)
    if ph["ramp_end"] - ph["ramp_start"] < min_ramp:
        ph["ramp_start"] = max(ph["first_move"], ph["ramp_end"] - min_ramp)
    w = ramp_weights(len(q), ph["ramp_start"], ph["ramp_end"])
    return {"name": name, "ep": ep, "state": st, "q": q, "phases": ph, "w": w}


def _head_worker(task: dict):
    """Synthesise head frames [lo, hi) of one episode for every d_theta; write into the memmaps created by the caller."""
    os.nice(19)
    import cv2
    chain_plan = task["plan"]
    q, w = chain_plan["q"], chain_plan["w"]
    ph = chain_plan["phases"]
    rend = ArmRenderer()
    syn = HeadSynth(load_plate(), rend, BOX_Z)
    src = np.load(task["src_npy"], mmap_mode="r")
    outs = {d: np.load(p, mmap_mode="r+") for d, p in task["outs"].items()}
    f0 = np.array(src[0])
    seg0 = syn.segment(f0, q[0])
    import cv2
    cs, _ = cv2.findContours(seg0["box"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = np.zeros((H, W), np.uint8)
    if cs:
        cv2.fillPoly(hull, [cv2.convexHull(np.vstack(cs))], 1)       # the closed carton is convex: fixes mask notches
    mem = {"img": f0, "mask": hull > 0}
    for t in range(task["lo"], task["hi"]):
        rgb = np.array(src[t])
        seg = syn.segment(rgb, q[t])
        for d, o in outs.items():
            dth = np.radians(float(d))
            m = mem if t < ph["ramp_end"] else None
            img, _ = syn.synth(rgb, q[t], dth * w[t], dth, box_mem=m, seg=seg)
            o[t] = img
    for o in outs.values():
        o.flush()
    return task["lo"], task["hi"]


# ------------------------------------------------------------------------------------------ outputs
def encode_frames(frames, dst: str, fps: float, crf: int, gop: int = 2, preset: str = "medium", threads: int = 1,
                  tb_scale: int = 512, sample_every: int | None = None) -> dict:
    """libx264 yuv420p, no B-frames, like video_reencode.reencode_episode.  ``frames``: iterable of RGB uint8 (H, W, 3)."""
    import av
    from fractions import Fraction
    dst_p = Path(dst)
    dst_p.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst_p.with_suffix(".tmp.mp4")
    out = av.open(str(tmp), "w", options={"movflags": "faststart"})
    n = 0
    st = None
    sampled = []
    try:
        for img in frames:
            if st is None:
                st = out.add_stream("libx264", rate=int(round(fps)))
                st.height, st.width = img.shape[:2]
                st.pix_fmt = "yuv420p"
                st.time_base = Fraction(1, int(round(fps)) * tb_scale)
                st.codec_context.time_base = st.time_base
                st.options = {"crf": str(crf), "g": str(gop), "bf": "0", "preset": preset, "threads": str(threads)}
            f = av.VideoFrame.from_ndarray(np.ascontiguousarray(img), format="rgb24").reformat(format="yuv420p")
            f.pts = n * tb_scale
            f.time_base = st.time_base
            for p in st.encode(f):
                out.mux(p)
            n += 1
        for p in st.encode():
            out.mux(p)
        out.close()
        out = None
        tmp.replace(dst_p)
    finally:
        if out is not None:
            out.close()
        tmp.unlink(missing_ok=True)
    return {"frames": n, "bytes": dst_p.stat().st_size}


def label(img: np.ndarray, text: str, y: int = 22) -> np.ndarray:
    import cv2
    out = img.copy()
    cv2.putText(out, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def side_by_side_video(orig, synth, w: np.ndarray, dst: str, dtheta_deg: float, cam: str, step: int = 3, play_fps: int = 20,
                       crf: int = 27) -> dict:
    n = len(orig)

    def gen():
        for t in range(0, n, step):
            a = label(np.array(orig[t]), f"original {cam}  f{t}")
            b = label(np.array(synth[t]), f"retarget {dtheta_deg:+.0f} deg  w={w[t]:.2f}")
            yield np.hstack([a, b])
    return encode_frames(gen(), dst, play_fps, crf, gop=30, preset="medium")


def plot_joints(plans: dict, deltas_deg, dst: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(len(plans), 6, figsize=(24, 3.6 * len(plans)), squeeze=False)
    for r, (name, pl) in enumerate(plans.items()):
        t = np.arange(len(pl["q"])) / FPS
        ph = pl["phases"]
        for j in range(6):
            a = ax[r, j]
            a.plot(t, np.degrees(pl["q"][:, j]), "k", lw=1.6, label="original")
            for d, c in zip(deltas_deg, ("tab:blue", "tab:red")):
                a.plot(t, np.degrees(retarget_q(pl["q"], np.radians(d), pl["w"])[:, j]), c, lw=1.0, label=f"{d:+.0f} deg")
            a.axvspan(ph["ramp_start"] / FPS, ph["ramp_end"] / FPS, color="orange", alpha=0.2)
            a.axvline(ph["contact_onset"] / FPS, color="g", ls=":")
            a.set_title(f"{name} j{j + 1}" + ("  (orange = ramp, green = contact onset)" if j == 0 else ""), fontsize=9)
            a.set_xlabel("s")
            a.set_ylabel("deg")
            a.grid(alpha=0.3)
            if j == 0:
                a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(dst, dpi=70)
    plt.close(fig)


def contact_sheet(rows: list[list[np.ndarray]], heads: list[str], dst: Path, scale: float = 0.5) -> None:
    import cv2
    out = []
    for r in rows:
        out.append(np.hstack([cv2.resize(c, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) for c in r]))
    top = np.hstack([label(np.zeros_like(cv2.resize(rows[0][i], None, fx=scale, fy=scale))[:30], h, 20) for i, h in enumerate(heads)])
    cv2.imwrite(str(dst), cv2.cvtColor(np.vstack([top] + out), cv2.COLOR_RGB2BGR))



# ------------------------------------------------------------------------------------------ test dataset (LeRobot v3.0)
MARKER, MARKER_TOOL = "AUGMENT.json", "retarget_test.py"


def write_dataset(out_final: Path, runs: list[dict], source_root: str = SRC, crf: int = 16, log=print) -> dict:
    """runs: [{name, ep, d_deg, q_new (N,6), w (N,), head (path .npy), wrist (path .npy), plan}] -> LeRobot v3.0 dataset."""
    import shutil
    import time
    import datetime as _dt
    import datasets
    import pyarrow as pa
    import pyarrow.parquet as pq
    from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats, sample_indices, auto_downsample_height_width
    from lerobot.datasets.utils import get_hf_features_from_features, write_info, write_stats
    import preprocess as pp
    import video_reencode as vr

    t_start = time.time()
    info, df, rows, smap = read_source(source_root)
    names = list(info["features"]["observation.state"]["names"])
    slots = [names.index(f"{ARM}_joint{i}") for i in range(1, 7)]
    if out_final.exists():
        if not (out_final / MARKER).is_file() or json.loads((out_final / MARKER).read_text()).get("tool") != MARKER_TOOL:
            raise RuntimeError(f"{out_final} exists and was not created by {MARKER_TOOL}; not removing it")
        shutil.rmtree(out_final)
    stage = out_final.parent / f".{out_final.name}.partial-{os.getpid()}"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    video_keys = [k for k, f in info["features"].items() if f["dtype"] == "video"]
    cams = {"observation.images.head": "head", "observation.images.wrist_right": "wrist"}
    vcols = ("observation.state", "action", pp.RAW_STATE, "action_teleop")
    cols = {k: [] for k in (*vcols, *pp.SCALAR_KEYS, pp.SRC_INDEX, pp.SRC_TIME)}
    features = {k: dict(v) for k, v in info["features"].items()}
    ep_rows, ep_stats, smap_out, per_cam = [], [], [], {k: [] for k in video_keys}
    lengths = [len(r["q_new"]) for r in runs]
    starts = np.concatenate([[0], np.cumsum(lengths)]).astype(int)
    tmpdir = SCRATCH / "dataset_tmp"
    tmpdir.mkdir(parents=True, exist_ok=True)
    maxdev = {}
    for oe, r in enumerate(runs):
        sel = df[df["episode_index"] == r["ep"]].sort_values("frame_index")
        n = len(sel)
        S = {c: np.stack(sel[c].to_numpy()).astype(np.float64) for c in vcols}
        q_src = S["observation.state"][:, slots]
        d = np.radians(r["d_deg"])
        dq = r["q_new"] - q_src
        if np.abs(dq[:, 1:]).max() > 1e-9:
            raise RuntimeError("retarget changed a joint other than joint 1")
        a_src = S["action"][:, slots]
        a_new = a_src + (next_state_action(r["q_new"]) - next_state_action(q_src))
        w_next = np.r_[r["w"][1:], r["w"][-1:]]
        data = {}
        for c in vcols:
            data[c] = S[c].copy()
        data["observation.state"][:, slots[0]] = r["q_new"][:, 0]
        data["action"][:, slots] = a_new
        data[pp.RAW_STATE][:, slots[0]] += d * r["w"]
        data["action_teleop"][:, slots[0]] += d * w_next
        maxdev[oe] = {"state_vs_next_state_action": float(np.abs(data["action"][:-1, slots] - data["observation.state"][1:, slots]).max())}
        for c in vcols:
            cols[c].append(data[c].astype(np.float32))
        for c in (*pp.SCALAR_KEYS, pp.SRC_INDEX, pp.SRC_TIME):
            v = sel[c].to_numpy()
            if c == "episode_index":
                v = np.full(n, oe, np.int64)
            elif c == "index":
                v = np.arange(starts[oe], starts[oe] + n, dtype=np.int64)
            cols[c].append(v)
        data_e = {**{c: data[c].astype(np.float32) for c in vcols}, **{c: cols[c][-1] for c in (*pp.SCALAR_KEYS, pp.SRC_INDEX, pp.SRC_TIME)}}
        st = compute_episode_stats({k: pp.as_stats_array(v) for k, v in data_e.items()}, {k: features[k] for k in data_e})
        srow = rows[r["ep"]]
        row = {"episode_index": oe, "tasks": srow["tasks"], "length": n, "data/chunk_index": 0, "data/file_index": 0,
               "dataset_from_index": int(starts[oe]), "dataset_to_index": int(starts[oe] + n)}
        vstats = {}
        for key, tag in cams.items():
            arr = np.load(r[tag], mmap_mode="r")
            if len(arr) != n:
                raise RuntimeError(f"{tag} frames {len(arr)} != {n}")
            mp4 = tmpdir / f"ep{oe:03d}_{tag}.mp4"
            res = encode_frames((np.array(arr[i]) for i in range(n)), str(mp4), FPS, crf, gop=2)
            sm = [np.ascontiguousarray(auto_downsample_height_width(np.array(arr[i]).transpose(2, 0, 1))) for i in sample_indices(n)]
            vstats[key] = vr.image_stats(np.stack(sm))
            per_cam[key].append((mp4, n))
            log(f"  ep{oe} {tag}: {res['bytes'] / 2**20:.1f} MiB", flush=True)
        st.update(vstats)
        ep_stats.append(st)
        row["_stats"] = st
        ep_rows.append(row)
        m = smap[r["ep"]]
        smap_out.append({"episode_index": oe, "source": m["source"], "source_episode_index": r["ep"], "source_dataset": str(source_root),
                         "d_theta_deg": r["d_deg"], "ramp_start": r["plan"]["phases"]["ramp_start"], "ramp_end": r["plan"]["phases"]["ramp_end"],
                         "augmented": True, "frames": n, "clean_dataset_record": m})
    # pack videos
    place = {}
    for key in video_keys:
        place[key] = vr.pack_videos(per_cam[key], lambda ch, fi, key=key: stage / pp.V30_VIDEO_PATH.format(video_key=key, chunk_index=ch, file_index=fi),
                                    info.get("video_files_size_in_mb", 200), FPS, info.get("chunks_size", 1000))
    for oe, row in enumerate(ep_rows):
        for key in video_keys:
            ch, fi, t0, t1 = place[key][oe]
            row.update({f"videos/{key}/chunk_index": ch, f"videos/{key}/file_index": fi,
                        f"videos/{key}/from_timestamp": t0, f"videos/{key}/to_timestamp": t1})
        row.update(pp.flat_stats(row.pop("_stats")))
        row["meta/episodes/chunk_index"] = 0
        row["meta/episodes/file_index"] = 0
    full = {k: np.concatenate(v) for k, v in cols.items()}
    hf_features = get_hf_features_from_features({k: {**v, "shape": tuple(v["shape"])} for k, v in features.items() if v["dtype"] != "video"})
    ds = datasets.Dataset.from_dict({k: (full[k].tolist() if full[k].ndim == 2 else full[k]) for k in hf_features},
                                    features=hf_features, split="train")
    dp = stage / pp.V30_DATA_PATH.format(chunk_index=0, file_index=0)
    dp.parent.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(str(dp))
    ep_path = stage / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    ep_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(ep_rows), ep_path)
    shutil.copyfile(Path(source_root) / "meta" / "tasks.parquet", stage / "meta" / "tasks.parquet")
    gstats = compute_episode_stats({k: pp.as_stats_array(v) for k, v in full.items()}, {k: features[k] for k in full})
    gstats.update(aggregate_stats([{k: s[k] for k in video_keys} for s in ep_stats]))
    write_stats(gstats, stage)
    new_info = dict(info)
    new_info.update({"total_episodes": len(runs), "total_frames": int(starts[-1]), "splits": {"train": f"0:{len(runs)}"}})
    write_info(new_info, stage)
    (stage / "meta" / "source_episodes.json").write_text(json.dumps(smap_out, indent=2))
    aug = {"tool": MARKER_TOOL, "command": " ".join(sys.argv), "created": _dt.datetime.now().isoformat(timespec="seconds"),
           "source_root": str(source_root), "output": str(out_final), "kind": "TEST (2 source episodes x 2 d_theta); not for training",
           "mode": "exact joint-1 retarget: j1_new = j1 + d_theta * w(t), w min-jerk 0->1 during the free-space approach, 1 afterwards",
           "head_camera": "synthetic (layered 3-D rigid rotation about base z over a median background plate); see retarget_test.md",
           "wrist_camera": "source frames; homography K R K^-1 (rotation only, assumed intrinsics) while w < 1",
           "action_label": "action[right arm] = source action + (next_state(q_new) - next_state(q_src)); raw state += d*w; action_teleop += d*w(t+1)",
           "lerobot_version": __import__("lerobot").__version__, "box_plane_z_m": BOX_Z, "episodes": smap_out,
           "per_episode_checks": maxdev, "build_seconds": round(time.time() - t_start, 1)}
    (stage / MARKER).write_text(json.dumps(aug, indent=2))
    stage.rename(out_final)
    shutil.rmtree(tmpdir, ignore_errors=True)
    return aug



def run_all(workers: int = 4, chunk: int = 200, skip_head: bool = False) -> dict:
    """Trajectories, wrist + head synthesis for the test episodes; arrays in SCRATCH, figures/videos in REPORT."""
    from concurrent.futures import ProcessPoolExecutor
    info, df, rows, smap = read_source()
    chain = Chain()
    REPORT.mkdir(parents=True, exist_ok=True)
    plans = {n: episode_plan(n, chain, df, smap) for n in TEST_EPISODES}
    result = {"episodes": {}}
    tasks, wrist_done = [], {}
    for name, pl in plans.items():
        ep, n = pl["ep"], len(pl["q"])
        outs = {}
        for d in DELTAS_DEG:
            pth = SCRATCH / f"syn_head_ep{ep}_{d:+.0f}.npy"
            if not skip_head:
                np.lib.format.open_memmap(pth, mode="w+", dtype=np.uint8, shape=(n, H, W, 3)).flush()
            outs[f"{d}"] = str(pth)
        for lo in range(0, n, chunk):
            tasks.append({"plan": {k: pl[k] for k in ("q", "w", "phases")}, "src_npy": str(SCRATCH / f"ep{ep}_head.npy"),
                          "outs": outs, "lo": lo, "hi": min(lo + chunk, n)})
        # wrist
        wsrc = np.load(SCRATCH / f"ep{ep}_wrist_right.npy", mmap_mode="r")
        T = chain.fk_series(pl["q"])
        for d in DELTAS_DEG:
            pth = SCRATCH / f"syn_wrist_ep{ep}_{d:+.0f}.npy"
            o = np.lib.format.open_memmap(pth, mode="w+", dtype=np.uint8, shape=(n, H, W, 3))
            for t in range(n):
                o[t] = warp_wrist(np.array(wsrc[t]), T[t], np.radians(d) * (1 - pl["w"][t]))
            o.flush()
        res = {"phases": pl["phases"], "source": check_q(pl["q"], chain), "frames": n}
        for d in DELTAS_DEG:
            res[f"{d:+.0f}"] = check_q(retarget_q(pl["q"], np.radians(d), pl["w"]), chain)
        result["episodes"][name] = res
    (REPORT / "trajectory_checks.json").write_text(json.dumps(result, indent=2))
    if not skip_head:
        with ProcessPoolExecutor(max_workers=min(workers, 4), initializer=_worker_init) as ex:
            for lo, hi in ex.map(_head_worker, tasks):
                print(f"head frames {lo}-{hi} done", flush=True)
    plot_joints(plans, DELTAS_DEG, REPORT / "joints_original_vs_retargeted.png")
    # videos + contact sheets
    for name, pl in plans.items():
        ep = pl["ep"]
        for cam, pre, src in (("head", "syn_head", "ep{}_head"), ("wrist", "syn_wrist", "ep{}_wrist_right")):
            o = np.load(SCRATCH / (src.format(ep) + ".npy"), mmap_mode="r")
            for d in DELTAS_DEG:
                sy = np.load(SCRATCH / f"{pre}_ep{ep}_{d:+.0f}.npy", mmap_mode="r")
                r = side_by_side_video(o, sy, pl["w"], str(REPORT / f"{name}_{cam}_{d:+.0f}deg.mp4"), d, cam)
                print(name, cam, d, f"{r['bytes'] / 2**20:.1f} MiB", flush=True)
        ph = pl["phases"]
        ts = np.linspace(0, len(pl["q"]) - 1, 9)[1:8].astype(int)
        ts = sorted({int(t) for t in ts} | {ph["ramp_end"] - 10, ph["contact_onset"] + 40})[:8]
        oh, ow = np.load(SCRATCH / f"ep{ep}_head.npy", mmap_mode="r"), np.load(SCRATCH / f"ep{ep}_wrist_right.npy", mmap_mode="r")
        sh = {d: np.load(SCRATCH / f"syn_head_ep{ep}_{d:+.0f}.npy", mmap_mode="r") for d in DELTAS_DEG}
        sw = {d: np.load(SCRATCH / f"syn_wrist_ep{ep}_{d:+.0f}.npy", mmap_mode="r") for d in DELTAS_DEG}
        sheet = [[label(np.array(oh[t]), f"f{t} w={pl['w'][t]:.2f}"), np.array(sh[4.0][t]), np.array(sh[-4.0][t]),
                  np.array(ow[t]), np.array(sw[4.0][t])] for t in ts]
        contact_sheet(sheet, ["head original", "head +4 deg", "head -4 deg", "wrist original", "wrist +4 deg"], REPORT / f"contact_sheet_{name}.png")
    return {"plans": plans, "result": result}


def dataset_cmd() -> dict:
    info, df, rows, smap = read_source()
    chain = Chain()
    runs = []
    for name in TEST_EPISODES:
        pl = episode_plan(name, chain, df, smap)
        for d in DELTAS_DEG:
            runs.append({"name": name, "ep": pl["ep"], "d_deg": d, "q_new": retarget_q(pl["q"], np.radians(d), pl["w"]), "w": pl["w"],
                         "head": str(SCRATCH / f"syn_head_ep{pl['ep']}_{d:+.0f}.npy"),
                         "wrist": str(SCRATCH / f"syn_wrist_ep{pl['ep']}_{d:+.0f}.npy"), "plan": pl})
    for r in runs:
        r["plan"] = {"phases": r["plan"]["phases"]}
    return write_dataset(Path(OUT_DS), runs)


def validate_cmd(n_arm_cases: int = 3, n_box_cases: int = 3) -> dict:
    """Synthesise one REAL frame from another REAL episode and compare (box-at-home pairs and arm pairs differing in j1)."""
    import cv2
    from scipy.spatial.distance import cdist
    info, df, rows, smap = read_source()
    rend = ArmRenderer()
    syn = HeadSynth(load_plate(), rend, BOX_Z)
    REPORT.mkdir(parents=True, exist_ok=True)
    eps = list(PLATE_EPISODES)
    states = {e: episode_state(df, e)["observation.state"][:, 7:13] for e in eps}
    first = {e: read_video(SRC, "head", rows[e], [0])[0] for e in eps}
    out = {"box_pairs": [], "arm_pairs": []}
    # ---- A. carton rotated about base z (arm at home in both first frames)
    prs = find_box_rotation_pairs(syn, first)
    cand = [p for p in prs if abs(p["deg"]) >= 2.0 and p["iou_best"] - p["iou_zero"] >= 0.05]
    cand.sort(key=lambda p: -p["iou_best"])
    out["box_pair_candidates"] = len(cand)
    out["box_pairs_all_best_iou_quantiles"] = [float(x) for x in np.quantile([p["iou_best"] for p in prs], [0.5, 0.9, 1.0])]
    panels = []
    used = set()
    for p in cand:
        if len(out["box_pairs"]) >= n_box_cases:
            break
        if (p["a"], p["b"]) in used or (p["b"], p["a"]) in used:
            continue
        used.add((p["a"], p["b"]))
        A, B = first[p["a"]], first[p["b"]]
        segA, segB = syn.segment(A, states[p["a"]][0]), syn.segment(B, states[p["b"]][0])
        img, _ = syn.synth(A, states[p["a"]][0], 0.0, np.radians(p["deg"]), seg=segA)
        mA = rotate_mask(syn, fill_holes(segA["box"]), np.radians(p["deg"]))
        region = dilate(mA | segB["box"], 3)
        m = masked_metrics(img, B, region)
        m0 = masked_metrics(A, B, region)
        out["box_pairs"].append({**p, "a_source": smap[p["a"]]["source"], "b_source": smap[p["b"]]["source"],
                                 "box_iou_synth": iou(mA, fill_holes(segB["box"])), "box_iou_unrotated": iou(fill_holes(segA["box"]), fill_holes(segB["box"])),
                                 "synth": m, "no_retarget_baseline": m0})
        dif = np.abs(img.astype(int) - B.astype(int)).max(-1).clip(0, 255).astype(np.uint8)
        panels.append(np.hstack([label(A, f"real A ep{p['a']}"), label(img, f"A rotated {p['deg']:+.1f} deg"), label(B, f"real B ep{p['b']}"),
                                 cv2.cvtColor(dif * 3, cv2.COLOR_GRAY2RGB)]))
    if panels:
        cv2.imwrite(str(REPORT / "validation_box_pairs.jpg"), cv2.cvtColor(np.vstack(panels), cv2.COLOR_RGB2BGR))
    # ---- B. arm frames with equal joints 2..6 but joint 1 differing by 3-9 deg
    pool = []
    for e in eps:
        q = states[e]
        mv = np.abs(np.diff(q, axis=0)).sum(1) > 1e-4
        for t in range(150, len(q) - 50, 6):
            if mv[t]:
                pool.append((e, t))
    Q = np.array([states[e][t] for e, t in pool])
    D = np.abs(Q[:, None, 1:] - Q[None, :, 1:]).max(-1) if len(Q) < 3000 else None
    if D is None:
        D = cdist(Q[:, 1:], Q[:, 1:], "chebyshev")
    dj1 = Q[None, :, 0] - Q[:, None, 0]
    ok = (D < np.radians(1.5)) & (np.abs(dj1) > np.radians(3)) & (np.abs(dj1) < np.radians(9))
    ii, jj = np.nonzero(ok)
    order = np.argsort(D[ii, jj])
    panels, seen = [], set()
    for k in order:
        a, b = pool[ii[k]], pool[jj[k]]
        if a[0] == b[0] or (a[0], b[0]) in seen or len(out["arm_pairs"]) >= n_arm_cases:
            continue
        seen.add((a[0], b[0]))
        A = read_video(SRC, "head", rows[a[0]], [a[1]])[a[1]]
        B = read_video(SRC, "head", rows[b[0]], [b[1]])[b[1]]
        qa, qb = states[a[0]][a[1]], states[b[0]][b[1]]
        dj = float(qb[0] - qa[0])
        segA, segB = syn.segment(A, qa), syn.segment(B, qb)
        img, _ = syn.synth(A, qa, dj, 0.0, seg=segA)
        # synthesised arm mask = lifted-and-rotated mask of A
        Pm = np.zeros((H, W), np.uint8)
        P = syn.lift(segA["arm_rot"], segA["arm_depth"])
        Xb = P[:, 2:5]
        c, s_ = np.cos(dj), np.sin(dj)
        uv = syn.project(np.stack([c * Xb[:, 0] - s_ * Xb[:, 1], s_ * Xb[:, 0] + c * Xb[:, 1], Xb[:, 2]], 1))
        xi, yi = np.rint(uv[:, 0]).astype(int), np.rint(uv[:, 1]).astype(int)
        v = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        Pm[yi[v], xi[v]] = 1
        mA = cv2.morphologyEx(Pm, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
        mB = segB["arm_rot"]
        region = erode(mA & mB, 2) | (mA & mB)
        region = mB.copy()
        out["arm_pairs"].append({"a": list(map(int, a)), "b": list(map(int, b)), "dj1_deg": float(np.degrees(dj)),
                                 "max_other_joint_diff_deg": float(np.degrees(D[ii[k], jj[k]])),
                                 "arm_iou_synth": iou(mA, mB), "arm_iou_unrotated": iou(segA["arm_rot"], mB),
                                 "synth_on_realB_arm": masked_metrics(img, B, region), "unrotated_on_realB_arm": masked_metrics(A, B, region)})
        dif = np.abs(img.astype(int) - B.astype(int)).max(-1).clip(0, 255).astype(np.uint8)
        panels.append(np.hstack([label(A, f"real A ep{a[0]} f{a[1]}"), label(img, f"A rotated j1 {np.degrees(dj):+.1f} deg"),
                                 label(B, f"real B ep{b[0]} f{b[1]}"), cv2.cvtColor(dif * 3, cv2.COLOR_GRAY2RGB)]))
    if panels:
        cv2.imwrite(str(REPORT / "validation_arm_pairs.jpg"), cv2.cvtColor(np.vstack(panels), cv2.COLOR_RGB2BGR))
    (REPORT / "validation.json").write_text(json.dumps(out, indent=2, default=float))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plate")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--refill", action="store_true", help="only redo the inpainting from the saved median (--min-count)")
    p.add_argument("--min-count", type=int, default=40)
    r = sub.add_parser("run")
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--skip-head", action="store_true")
    sub.add_parser("dataset")
    sub.add_parser("validate")
    args = ap.parse_args(argv)
    if args.cmd == "run":
        run_all(args.workers, skip_head=args.skip_head)
    elif args.cmd == "dataset":
        dataset_cmd()
    elif args.cmd == "validate":
        validate_cmd()
    if args.cmd == "plate":
        if args.refill:
            z = np.load(SCRATCH / "plate.npz")
            refill_plate(z["raw"], z["count"], args.min_count)
        else:
            build_plate(args.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
