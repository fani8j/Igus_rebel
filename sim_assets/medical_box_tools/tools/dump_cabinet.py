#!/usr/bin/env python3
"""Dump the geometry of the stocked cabinet (as used by the cabinet test scenes: shelf board = one plate 0.05 .. 0.40, lips dropped) to data/cabinet_geometry.json
(used by tools/robot_kin.py, which has no pxr).  Run with the USD venv python."""
import json
from pathlib import Path
import numpy as np
from pxr import Usd, UsdGeom

OUT = Path(__file__).resolve().parent.parent
src = OUT.parent / 'medical_boxes' / 'usd' / 'medicine_cabinet_filled.usda'
s = Usd.Stage.Open(str(src))
bb = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_, UsdGeom.Tokens.proxy, UsdGeom.Tokens.render, UsdGeom.Tokens.guide])
boxes = {}
for ch in s.GetPrimAtPath('/World/Cabinet/Cabinet').GetChildren():
    r = bb.ComputeWorldBound(ch).ComputeAlignedRange()
    if r.IsEmpty() or ch.GetName().startswith('ShelfLip'):
        continue
    lo, hi = np.array(r.GetMin()), np.array(r.GetMax())
    if ch.GetName().startswith('Shelf_'):
        lo[1] = 0.05                                    # scene layer: one plate 0.05 .. 0.40
    boxes[ch.GetName()] = [lo.tolist(), hi.tolist()]
BOX = {'agent_monorail': (0.230, 0.250, 0.015), 'angiolite': (0.240, 0.250, 0.016), 'accuforce': (0.220, 0.250, 0.016),
       'ultimaster_nagomi': (0.250, 0.295, 0.016), 'conqueror_nc_pro': (0.230, 0.250, 0.012)}
cartons = {}
xc = UsdGeom.XformCache()
for k in range(1, 7):
    items = []
    for c in s.GetPrimAtPath(f'/World/Cartons/Shelf_0{k}').GetChildren():
        M = np.array(xc.GetLocalToWorldTransform(c)).T
        items.append(dict(name=c.GetName(), path=str(c.GetPath()), box=c.GetName().rsplit('_', 1)[0], c=M[:3, 3].tolist(), R=M[:3, :3].tolist()))
    cartons[f'Shelf_0{k}'] = items
json.dump(dict(boxes=boxes, carton_dims_m_h_l_t=BOX, cartons=cartons), open(OUT / 'data' / 'cabinet_geometry.json', 'w'))
print('boxes', len(boxes), 'cartons', sum(len(v) for v in cartons.values()))
