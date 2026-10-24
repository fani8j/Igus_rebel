"""Analytic table + box scenes as camera-frame XYZ clouds (depth tests).

Scene (table frame: origin on the table, +z = table normal toward the camera):
the table is the plane z = 0, the box is an upright cuboid (``width`` x
``length`` footprint, yaw about +z, height ``h``) with a flat top at z = h and
four vertical walls. The camera sits at (0, 0, cam_dist) in the table frame,
tilted like the real rig. Every pixel of a REAL manifest camera (size + P from
``camera_info``, same loader the pipeline uses) is ray-cast against the planes,
so visibility is exact (nearest hit wins, i.e. an analytic z-buffer): the
cloud contains only what the camera can see, and the cloud is the organised
(H, W, 3) grid (NaN = no hit) or the unorganised valid-point list.

Default pose = the real rig: camera ~0.454 m above the table along the table
normal ``(0.0309, -0.7088, -0.7047)`` (camera frame; estimated from capture
``frame_000001_20260928_040422_447556``: top plane n.p + d = 0, d = 0.3678 at
box height 86.3 mm). For tall boxes pass a larger ``cam_dist`` and a
``center_px`` that keeps the box in view.

Camera distortion is ignored on purpose (rectified RGB-D, as the pipeline).

Typical use::

    sc = Scene(h=0.0863, width=0.19, length=0.10, yaw_deg=10)
    g = sc.cast(step=2, noise_sigma_m=0.0005, seed=1)   # dict of arrays
    xyz = g["xyz"].reshape(-1, 3)                       # organised -> flat
    write_scene(tmp_dir, "synth_000001", sc)            # PCD + manifest + RGB
"""
from __future__ import annotations

import copy
import json
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from box_seam_detection.core.autolabel import depth_top as _depth  # noqa: E402

DEFAULT_MANIFEST = Path("/home/farhan/Igus_rebel/outputs/rgbd_captures/captures/"
                        "frame_000001_20260928_040422_447556.json")
# Built-in copy of that manifest's camera_info (used when the file is absent).
_BUILTIN_CAMERA_INFO = {
    "width": 1920, "height": 1080, "distortion_model": "plumb_bob",
    "d": [0.007051904685795307, -0.049205388873815536, 0.00026131237973459065,
          9.177037281915545e-05, 0.03320758789777756, 0.0, 0.0, 0.0],
    "k": [1037.8170166015625, 0.0, 977.0050048828125, 0.0, 1037.6590576171875,
          548.9818115234375, 0.0, 0.0, 1.0],
    "r": [1, 0, 0, 0, 1, 0, 0, 0, 1],
    "p": [1037.8170166015625, 0.0, 977.0050048828125, 0.0, 0.0,
          1037.6590576171875, 548.9818115234375, 0.0, 0.0, 0.0, 1.0, 0.0],
    "header": {"frame_id": "camera_color_optical_frame", "stamp_ns": 0},
}
REAL_TABLE_NORMAL = (0.0309, -0.7088, -0.7047)   # camera frame, toward camera
REAL_TABLE_DIST = 0.3678 + 0.0863                 # d of n.p + d = 0 for the table

LABEL_NONE, LABEL_TABLE, LABEL_TOP = 0, 1, 2
LABEL_WALLS = (3, 4, 5, 6)                        # +x1, -x1, +x2, -x2 (box axes)

# RGB colours of the synthetic image (table grey, top brown, walls darker).
RGB_TABLE = (128, 128, 130)
RGB_TOP = (176, 124, 78)
RGB_WALLS = ((120, 84, 52), (96, 66, 40), (110, 77, 48), (86, 60, 36))


def load_manifest(path=None) -> dict:
    p = Path(path) if path else DEFAULT_MANIFEST
    if p.is_file():
        with open(p, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {"schema": "rgbd_capture/v1", "capture_id": "builtin",
            "image": {"width": 1920, "height": 1080, "encoding": "rgb8"},
            "pointcloud": {}, "camera_info": copy.deepcopy(_BUILTIN_CAMERA_INFO)}


def load_camera(manifest: dict | None = None, path=None) -> dict:
    """``{fx, fy, cx, cy, width, height}`` via the pipeline's own loader
    (``autolabel.depth_top.camera_from_capture_info``; P preferred, no D)."""
    m = manifest if manifest is not None else load_manifest(path)
    cam, _src = _depth.camera_from_capture_info(m["camera_info"])
    return cam


@dataclass
class Scene:
    h: float = 0.0863                 # box (top) height above the table, m
    width: float = 0.19               # footprint size along box axis 1, m
    length: float = 0.10              # footprint size along box axis 2, m
    yaw_deg: float = 0.0              # rotation about the table normal
    center_px: tuple = (960.0, 720.0)  # pixel whose table ray = box base centre
    cam_dist: float = REAL_TABLE_DIST  # camera distance to the table plane, m
    tilt_deg: float | None = None     # None = real normal; else n = (0,-sin,-cos)
    manifest: dict = field(default_factory=load_manifest)

    # ---- geometry -----------------------------------------------------
    def __post_init__(self):
        self.cam = load_camera(self.manifest)
        if self.tilt_deg is None:
            n = np.asarray(REAL_TABLE_NORMAL, dtype=np.float64)
        else:
            t = np.radians(self.tilt_deg)
            n = np.array([0.0, -np.sin(t), -np.cos(t)])
        self.n = n / np.linalg.norm(n)
        ex = np.array([1.0, 0.0, 0.0])
        e1 = ex - (ex @ self.n) * self.n
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(self.n, e1)
        # columns: table axes in the camera frame (right-handed: e1 x e2 = n)
        self.Rm = np.stack([e1, e2, self.n], axis=1)
        self.O = -self.cam_dist * self.n          # table origin, camera frame
        # box base centre: table intersection of the centre_px ray
        c = self._table_point(self.center_px)
        self.c12 = c[:2]
        a = np.radians(self.yaw_deg)
        self.a1 = np.array([np.cos(a), np.sin(a)])
        self.a2 = np.array([-np.sin(a), np.cos(a)])

    def _ray_dirs_cam(self, u, v):
        cam = self.cam
        d = np.stack([(np.asarray(u, float) - cam["cx"]) / cam["fx"],
                      (np.asarray(v, float) - cam["cy"]) / cam["fy"],
                      np.ones(np.broadcast(u, v).shape)], axis=-1)
        return d

    def _table_point(self, px) -> np.ndarray:
        """Table-frame (x, y, 0) hit of the pixel ray (must hit the table)."""
        D = self._ray_dirs_cam(px[0], px[1]) @ self.Rm        # table frame
        if D[2] >= 0:
            raise ValueError("center_px ray does not hit the table")
        t = -self.cam_dist / D[2]
        return np.array([t * D[0], t * D[1], 0.0])

    def top_corners_table(self) -> np.ndarray:
        """4 top corners (table frame, z = h), order +x1+x2, -x1+x2, -x1-x2, +x1-x2."""
        hw, hl = self.width / 2, self.length / 2
        out = []
        for s1, s2 in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
            p = self.c12 + s1 * hw * self.a1 + s2 * hl * self.a2
            out.append([p[0], p[1], self.h])
        return np.asarray(out)

    def to_cam(self, p_table) -> np.ndarray:
        return np.asarray(p_table, float) @ self.Rm.T + self.O

    def project(self, p_cam) -> np.ndarray:
        p = np.asarray(p_cam, float)
        return np.stack([self.cam["fx"] * p[..., 0] / p[..., 2] + self.cam["cx"],
                         self.cam["fy"] * p[..., 1] / p[..., 2] + self.cam["cy"]], -1)

    def truth(self) -> dict:
        """Ground truth in the camera frame."""
        top = self.to_cam(self.top_corners_table())
        base = top.copy()
        tb = self.top_corners_table()
        tb[:, 2] = 0.0
        # plane n.p + d = 0 (n toward the camera, d > 0)
        return {"table_n": self.n.copy(), "table_d": float(self.cam_dist),
                "top_n": self.n.copy(), "top_d": float(self.cam_dist - self.h),
                "height_m": float(self.h),
                "top_corners_xyz": top, "top_corners_px": self.project(top),
                "base_corners_xyz": self.to_cam(tb)}

    # ---- ray casting --------------------------------------------------
    def cast(self, step: int = 1, noise_sigma_m: float = 0.0, seed: int = 0,
             max_range: float = 5.0) -> dict:
        """Cast every ``step``-th pixel. Returns dict: ``xyz`` (h, w, 3) camera
        frame (NaN where nothing is hit), ``label`` (h, w) int8, ``u``/``v``
        pixel grids, ``valid`` mask. Gaussian noise (sigma, m) is applied
        along each ray (depth noise)."""
        W, H = int(self.cam["width"]), int(self.cam["height"])
        us = np.arange(0, W, step, dtype=np.float64)
        vs = np.arange(0, H, step, dtype=np.float64)
        uu, vv = np.meshgrid(us, vs)
        Dc = self._ray_dirs_cam(uu, vv)                       # (h, w, 3) camera
        D = Dc @ self.Rm                                      # table frame
        C = np.array([0.0, 0.0, self.cam_dist])
        hw, hl = self.width / 2, self.length / 2
        # camera + ray in box-local coordinates (x1, x2)
        c1 = (C[:2] - self.c12) @ self.a1
        c2 = (C[:2] - self.c12) @ self.a2
        D1 = D[..., :2] @ self.a1
        D2 = D[..., :2] @ self.a2
        Dz = D[..., 2]
        inf = np.inf
        best = np.full(Dz.shape, inf)
        label = np.zeros(Dz.shape, dtype=np.int8)

        def consider(t, ok, lab):
            nonlocal best, label
            ok = ok & np.isfinite(t) & (t > 0) & (t < best)
            best = np.where(ok, t, best)
            label = np.where(ok, np.int8(lab), label)

        with np.errstate(divide="ignore", invalid="ignore"):
            consider(-C[2] / Dz, Dz < 0, LABEL_TABLE)
            t = (self.h - C[2]) / Dz
            x1, x2 = c1 + t * D1, c2 + t * D2
            consider(t, (np.abs(x1) <= hw) & (np.abs(x2) <= hl), LABEL_TOP)
            for (sgn, axis), lab in zip(((1, 1), (-1, 1), (1, 2), (-1, 2)), LABEL_WALLS):
                if axis == 1:
                    t = (sgn * hw - c1) / D1
                    z = C[2] + t * Dz
                    x2 = c2 + t * D2
                    ok = (np.abs(x2) <= hl)
                else:
                    t = (sgn * hl - c2) / D2
                    z = C[2] + t * Dz
                    x1 = c1 + t * D1
                    ok = (np.abs(x1) <= hw)
                consider(t, ok & (z >= 0) & (z <= self.h), lab)
        hit = np.isfinite(best) & (best < max_range)
        t = np.where(hit, best, np.nan)
        if noise_sigma_m > 0:
            rng = np.random.default_rng(seed)
            t = t + rng.normal(0.0, noise_sigma_m, t.shape)
        # camera-frame point: Dc is the camera-frame direction with unit z,
        # but table-frame t is for the table-frame direction D = Rm^T Dc
        # (norm preserved), so the camera point is t * Dc.
        xyz = t[..., None] * Dc
        label = np.where(hit, label, np.int8(LABEL_NONE))
        return {"xyz": xyz, "label": label, "u": uu, "v": vv, "valid": hit,
                "step": step}

    def cloud(self, organised: bool = False, **kw):
        g = self.cast(**kw)
        if organised:
            return g["xyz"].astype(np.float32)
        return g["xyz"][g["valid"]].astype(np.float32)

    # ---- RGB ------------------------------------------------------------
    def render_rgb(self, noise: float = 0.0, seed: int = 0) -> np.ndarray:
        """(H, W, 3) uint8 RGB: grey table, flat brown top, darker walls."""
        g = self.cast(step=1)
        lab = g["label"]
        img = np.empty(lab.shape + (3,), dtype=np.float64)
        img[:] = RGB_TABLE
        img[lab == LABEL_TOP] = RGB_TOP
        for l, c in zip(LABEL_WALLS, RGB_WALLS):
            img[lab == l] = c
        if noise > 0:
            img += np.random.default_rng(seed).normal(0, noise, img.shape)
        return np.clip(np.rint(img), 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ files
def write_pcd(path, xyz, rgb=None, binary: bool = True):
    """Write a PCD v0.7 readable by ``autolabel.pcd.read_pcd_xyz``.

    ``xyz`` is (N, 3) (unorganised, HEIGHT 1) or (H, W, 3) organised with NaN
    holes (kept in the file; the reader drops them). ``rgb`` optional uint8
    (N, 3) or (H, W, 3), packed like the real captures (float32 field)."""
    xyz = np.asarray(xyz, dtype=np.float32)
    if xyz.ndim == 3:
        height, width = xyz.shape[:2]
        flat = xyz.reshape(-1, 3)
        rgb_flat = None if rgb is None else np.asarray(rgb, np.uint8).reshape(-1, 3)
    else:
        flat = xyz.reshape(-1, 3)
        width, height = flat.shape[0], 1
        rgb_flat = None if rgb is None else np.asarray(rgb, np.uint8).reshape(-1, 3)
    n = flat.shape[0]
    packed = np.zeros(n, dtype=np.uint32)
    if rgb_flat is not None:
        packed = (rgb_flat[:, 0].astype(np.uint32) << 16
                  | rgb_flat[:, 1].astype(np.uint32) << 8
                  | rgb_flat[:, 2].astype(np.uint32))
    rgbf = packed.view(np.float32)
    head = ("# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\n"
            "FIELDS x y z rgb\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\n"
            f"WIDTH {width}\nHEIGHT {height}\nVIEWPOINT 0 0 0 1 0 0 0\n"
            f"POINTS {n}\nDATA {'binary' if binary else 'ascii'}\n")
    with open(path, "wb") as fh:
        fh.write(head.encode("ascii"))
        if binary:
            rec = np.empty((n, 4), dtype="<f4")
            rec[:, :3] = flat
            rec[:, 3] = rgbf
            fh.write(rec.tobytes())
        else:
            for i in range(n):
                fh.write(f"{flat[i,0]!r} {flat[i,1]!r} {flat[i,2]!r} {float(rgbf[i])!r}\n".encode())


def write_png(path, rgb: np.ndarray):
    import cv2
    ok = cv2.imwrite(str(path), np.ascontiguousarray(rgb[..., ::-1]))
    if not ok:
        raise OSError(f"cannot write {path}")


def write_scene(out_dir, capture_id: str, scene: Scene, step: int = 1,
                noise_sigma_m: float = 0.0, seed: int = 0, organised: bool = False,
                image_noise: float = 0.0) -> dict:
    """Write ``captures/<id>.json`` (manifest copy), ``pointcloud/<id>.pcd`` and
    ``rgb/<id>.png`` under ``out_dir`` (same layout as a real capture dir).
    The manifest keeps the real camera_info; image/pointcloud paths and counts
    are rewritten. Returns the paths plus the ground truth."""
    out = Path(out_dir)
    for sub in ("captures", "pointcloud", "rgb"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    g = scene.cast(step=step, noise_sigma_m=noise_sigma_m, seed=seed)
    rgb_full = scene.render_rgb(noise=image_noise, seed=seed)
    if organised:
        xyz = g["xyz"]
        col = rgb_full[::step, ::step]
        n_valid = int(g["valid"].sum())
    else:
        xyz = g["xyz"][g["valid"]]
        col = rgb_full[::step, ::step][g["valid"]]
        n_valid = xyz.shape[0]
    pcd_path = out / "pointcloud" / f"{capture_id}.pcd"
    png_path = out / "rgb" / f"{capture_id}.png"
    man_path = out / "captures" / f"{capture_id}.json"
    write_pcd(pcd_path, xyz, col)
    write_png(png_path, rgb_full)
    m = copy.deepcopy(scene.manifest)
    m["capture_id"] = capture_id
    m.setdefault("image", {}).update(
        {"path": str(png_path), "width": int(scene.cam["width"]),
         "height": int(scene.cam["height"]), "encoding": "rgb8"})
    pc = m.setdefault("pointcloud", {})
    pc.update({"path": str(pcd_path), "point_count": n_valid,
               "width": int(xyz.shape[1]) if organised else n_valid,
               "height": int(xyz.shape[0]) if organised else 1,
               "is_dense": not organised})
    man_path.write_text(json.dumps(m, indent=1))
    return {"manifest": man_path, "pcd": pcd_path, "image": png_path,
            "truth": scene.truth(), "n_points": n_valid}
