"""Bridge-aware support estimate (voxel, layer by layer) used by verify.py - one function, no tool state.

Model: part voxelised at `pitch` (= layer height = 1 mm), +z of the rotated frame is the build direction.  A solid voxel in layer k is
  * self-supported if one of the 3 x 3 voxels below it (8-neighbourhood = 45 deg overhang rule) is solid,
  * bridged (printable in the air) if it is within `bridge_mm / 2` of a self-supported voxel of the same layer (gaps up to bridge_mm between two supports),
  * otherwise needs support: the empty voxels below it down to the bed or the first solid voxel become support material.
Returns the support volume (cm3), the overhang (non-self-supported) area and the part height.  The raw 'solid upper bound' used by v3/v4 (every down-facing
face x its free drop, no bridging) is kept in verify.py for comparison."""
import numpy as np
import trimesh
from scipy import ndimage as ndi


def rotate_up(mesh, up):
    z = np.asarray(up, float); z /= np.linalg.norm(z)
    a = np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1.0, 0])
    x = a - (a @ z) * z; x /= np.linalg.norm(x); y = np.cross(z, x)
    R = np.stack([x, y, z])
    m = mesh.copy(); m.vertices = m.vertices @ R.T
    return m


def support_voxel(mesh, up, pitch=1.0, bridge_mm=12.0):
    m = rotate_up(mesh, up)
    vg = m.voxelized(pitch).fill()
    solid = vg.matrix.astype(bool)
    nx, ny, nz = solid.shape
    st = ndi.generate_binary_structure(2, 2)
    need = np.zeros_like(solid)
    over_area = 0
    prev = np.ones((nx, ny), bool)            # the bed
    for k in range(nz):
        S = solid[:, :, k]
        sup_ok = ndi.binary_dilation(prev, st)
        ok = S & sup_ok
        over = S & ~sup_ok
        if over.any():
            over_area += int(over.sum())
            d = ndi.distance_transform_edt(~ok) * pitch if ok.any() else np.full(S.shape, 1e9)
            need[:, :, k] = over & ~(d <= bridge_mm / 2.0)
        prev = S
    support = np.zeros_like(solid)
    active = np.zeros((nx, ny), bool)
    for k in range(nz - 1, -1, -1):
        s = active & ~solid[:, :, k]
        support[:, :, k] = s
        active = active & ~solid[:, :, k]
        active |= need[:, :, k]
    return dict(support_cm3=float(support.sum() * pitch ** 3 / 1000.0), overhang_area_mm2=float(over_area * pitch ** 2), part_cm3=float(solid.sum() * pitch ** 3 / 1000.0),
                height_mm=float(nz * pitch), need_voxels=int(need.sum()))


def best_axis(mesh, pitch=1.0, bridge_mm=12.0):
    dirs = {'+z': [0, 0, 1], '-z': [0, 0, -1], '+x': [1, 0, 0], '-x': [-1, 0, 0], '+y': [0, 1, 0], '-y': [0, -1, 0]}
    res = {k: support_voxel(mesh, np.array(v, float), pitch, bridge_mm) for k, v in dirs.items()}
    b = min(res, key=lambda k: res[k]['support_cm3'])
    return b, res
