"""Nominal handoff geometry: both tools on one carton, in a shelf frame W. Units mm, 4x4 homogeneous matrices.

Shelf frame W: origin on the shelf floor at the shelf front edge, laterally centred on the target carton.
  +X_w lateral along the shelf (across the carton thickness), +Y_w into the shelf, +Z_w up.
Handoff state (operator-confirmed sequence): the target carton is pulled PULL_OUT out of the shelf and still
rests on the floor; the suction pads are pressed (compressed TCP on the spine); the gripper has come down from
above, its fixed top-stop GRIPPER_STOP_GAP above the carton top edge (v3: position-based, no contact), pads closed on the two big faces at s = 15..70 mm
behind the spine. Neighbours keep their spines at y = 0 (optionally standing `protrusion` proud).

Tool frames below are copied from the URDFs (suction_tool/urdf, gripper_tool/urdf) and must be re-checked
against them by any script that uses this file (assert |delta| < 0.1 mm).
"""
import numpy as np

from params import BOXES, PULL_OUT

# link6 -> tool frames (translation only, identity rotation), from the expanded URDFs 2026-10-05 (suction v2)
SUCTION_TCP_Z = 159.0             # middle pad lip, uncompressed (suction v2)
SUCTION_TCP_COMPRESSED_Z = 154.0
GRIPPER_TCP_Z = 193.2             # midpoint between the pad centres
GRIPPER_STOP_Z = 167.5            # top-stop face (= carton top edge at handoff)
GRIPPER_PAD_S_CENTRE = 42.5       # pad centre behind the spine (pads span s = 15..70)
GRIPPER_STOP_GAP = 4.0            # gripper v3: stop face sits this far above the carton top edge at the final pose (G_NOM)


def _T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def box_pose(box_id, pulled=True):
    """Carton centre pose in W. Carton local axes: x = thickness, y = length (into the shelf), z = height."""
    b = BOXES[box_id]
    y_spine = -PULL_OUT if pulled else 0.0
    return _T(np.eye(3), [0.0, y_spine + b['l'] / 2.0, b['h'] / 2.0])


def neighbour_poses(box_id, protrusion=0.0, neighbour_t=None):
    """Two neighbours (same carton size unless neighbour_t given), spines at y = -protrusion."""
    b = BOXES[box_id]
    tn = b['t'] if neighbour_t is None else neighbour_t
    off = b['t'] / 2.0 + tn / 2.0
    return [_T(np.eye(3), [s * off, -protrusion + b['l'] / 2.0, b['h'] / 2.0]) for s in (-1.0, 1.0)]


def suction_link6_pose(box_id, pad_row_centre_z=None, pulled=True):
    """Left arm link6 in W. Tool z = +Y_w (approach into the spine), tool x = +Z_w (pad row vertical),
    tool y = +X_w (camera side). The compressed TCP sits on the spine centre line."""
    b = BOXES[box_id]
    zc = b['h'] / 2.0 if pad_row_centre_z is None else pad_row_centre_z
    R = np.column_stack([[0, 0, 1], [1, 0, 0], [0, 1, 0]]).astype(float)
    spine_pt = np.array([0.0, -PULL_OUT if pulled else 0.0, zc])
    return _T(R, spine_pt - R[:, 2] * SUCTION_TCP_COMPRESSED_Z)


def gripper_link6_pose(box_id, lift=0.0):
    """Right arm link6 in W at the handoff (lift > 0 = that far above the final pose, for approach sweeps).
    Tool z = -Z_w (down), tool x = +X_w (stroke across the thickness), tool y = -Y_w."""
    b = BOXES[box_id]
    R = np.column_stack([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).astype(float)
    tcp = np.array([0.0, -PULL_OUT + GRIPPER_PAD_S_CENTRE, b['h'] + GRIPPER_STOP_GAP - (GRIPPER_TCP_Z - GRIPPER_STOP_Z) + lift])
    return _T(R, tcp - R[:, 2] * GRIPPER_TCP_Z)


if __name__ == '__main__':
    np.set_printoptions(precision=2, suppress=True)
    for k in BOXES:
        print(k, BOXES[k]['name'])
        print(' suction link6\n', suction_link6_pose(k))
        print(' gripper link6\n', gripper_link6_pose(k))
