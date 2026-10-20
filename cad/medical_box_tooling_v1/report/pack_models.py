#!/usr/bin/env python3
"""Packs the suction tool and gripper visual STLs (metres, link6 frame) into media/models.js for the report's 3D viewer.

Usage: python3 pack_models.py [--suction-dir ../suction_tool_v5]
Vertices are de-duplicated and stored as base64 Float32 positions + Uint32 indices; the page builds BufferGeometries from them.
"""
import argparse
import base64
import json
import struct
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

SUCTION_COLORS = {          # part-name prefix -> (colour, label)
    'tool_body': ('#3a3f47', 'Tool body (PA12-CF, printed)'),
    'camera_gemini305': ('#1b1d21', 'Gemini 305 camera'),
    'pad_': ('#2f6fd6', 'ZP2 suction pads'),
    'nut_': ('#b8bec7', 'Pad nuts'),
    'washer_': ('#b8bec7', 'Pad nuts'),
    'elbow_': ('#d9a21b', 'KQ2L04-M5 elbows'),
    'tube_': ('#e8e8e8', 'Ø4 tubes'),
    'trunk_tube': ('#e8e8e8', 'Ø4 tubes'),
    'collector': ('#d9a21b', 'Collector'),
}
GRIPPER_BASE = {
    'base_flange_disc': ('#7d8590', 'Flange'),
    'base_gripper_plate': ('#7d8590', 'Adapter plate'),
    'base_smc_body': ('#d7dbe0', 'SMC LEHF20K2-48 body'),
    'base_top_stop': ('#e0663a', 'Top stop'),
}
FINGER = {
    'finger_': ('#9aa3ad', 'Fingers'),
    'finger_pad_': ('#2b2f36', 'Finger pads'),
    'finger_jaw_': ('#e0663a', 'Jaws'),
}


def read_stl(p):
    b = p.read_bytes()
    n = struct.unpack('<I', b[80:84])[0]
    assert 84 + 50 * n == len(b), f'{p} is not a binary STL'
    a = np.frombuffer(b[84:84 + 50 * n], dtype=np.dtype([('n', '<3f4'), ('v', '<9f4'), ('a', '<u2')]))
    return a['v'].reshape(-1, 3)


def pack(p, color, label):
    v = read_stl(p)
    q = np.round(v / 1e-6).astype(np.int64)           # 1 micron weld tolerance
    _, first, inv = np.unique(q, axis=0, return_index=True, return_inverse=True)
    pos = v[first].astype('<f4')
    idx = inv.reshape(-1).astype('<u4')
    return {'name': p.stem, 'color': color, 'label': label,
            'pos': base64.b64encode(pos.tobytes()).decode(), 'idx': base64.b64encode(idx.tobytes()).decode()}


def pick(name, table):
    best = max((k for k in table if name.startswith(k)), key=len, default=None)
    return table[best] if best else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--suction-dir', default=str(HERE.parent / 'suction_tool_v5'))
    ap.add_argument('--gripper-dir', default=str(HERE.parent / 'gripper_tool'))
    ap.add_argument('--out', default=str(HERE / 'media' / 'models.js'))
    a = ap.parse_args()
    sdir, gdir = Path(a.suction_dir) / 'meshes' / 'visual', Path(a.gripper_dir) / 'meshes' / 'visual'
    models = {'suction': [], 'gripper_base': [], 'finger_left': [], 'finger_right': [],
              'suction_source': Path(a.suction_dir).name,
              # from rebel_medicine_grasp_description/urdf/right_gripper_tool.urdf.xacro
              'finger_joint_z': 0.15, 'finger_stroke_m': 0.024}
    for p in sorted(sdir.glob('*.stl')):
        c = pick(p.stem, SUCTION_COLORS)
        if c:
            models['suction'].append(pack(p, *c))
    for p in sorted(gdir.glob('*.stl')):
        if p.stem.startswith('base_'):
            models['gripper_base'].append(pack(p, *pick(p.stem, GRIPPER_BASE)))
        else:
            side = 'finger_left' if p.stem.endswith('left') else 'finger_right'
            models[side].append(pack(p, *pick(p.stem, FINGER)))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('window.MODELS = ' + json.dumps(models) + ';\n')
    print(out, round(out.stat().st_size / 1e6, 2), 'MB')


if __name__ == '__main__':
    main()
