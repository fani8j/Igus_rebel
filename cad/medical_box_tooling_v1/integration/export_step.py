#!/usr/bin/env python3
"""step/handoff_box3.step (default suction tool v5; `--tool suction_tool` -> step/handoff_box3_v3.step): suction assembly + gripper (closed on 16 mm) + target box 3 + 2 flush neighbours, all in the shelf frame W."""
import os
import sys

import numpy as np
import cadquery as cq
from OCP.gp import gp_Trsf, gp_Mat, gp_XYZ

import handoff_check as hc

HERE = os.path.dirname(os.path.abspath(__file__))
BOX = 3


def loc(T):
    t = gp_Trsf()
    R = T[:3, :3]
    t.SetValues(R[0, 0], R[0, 1], R[0, 2], T[0, 3], R[1, 0], R[1, 1], R[1, 2], T[1, 3], R[2, 0], R[2, 1], R[2, 2], T[2, 3])
    return cq.Location(t)


def main():
    b = hc.P.BOXES[BOX]
    Ts = hc.H.suction_link6_pose(BOX)
    Tg = hc.H.gripper_link6_pose(BOX)
    asm = cq.Assembly(name='handoff_box%d' % BOX)
    for name, f, T, col in (('suction_tool', os.path.join(hc.SUCTION, 'step', 'assembly.step'), Ts, (0.15, 0.6, 0.67)),
                            ('gripper_tool_closed16', os.path.join(hc.GRIPPER, 'step', 'assembly_closed16.step'), Tg, (0.92, 0.55, 0.12))):
        shp = cq.importers.importStep(f)
        for i, s in enumerate(shp.solids().vals()):
            asm.add(s, name='%s_%02d' % (name, i), loc=loc(T), color=cq.Color(*col))
    def box(c, size):
        return cq.Workplane('XY').box(*size).val().moved(loc(hc.T_of(t=c)))
    asm.add(box(hc.H.box_pose(BOX)[:3, 3], (b['t'], b['l'], b['h'])), name='target_box', color=cq.Color(0.8, 0.66, 0.43))
    for i, Tn in enumerate(hc.H.neighbour_poses(BOX)):
        asm.add(box(Tn[:3, 3], (b['t'], b['l'], b['h'])), name='neighbour_%d' % (i + 1), color=cq.Color(0.9, 0.86, 0.74))
    p = os.path.join(HERE, 'step', 'handoff_box%d%s.step' % (BOX, hc.OUT_SUFFIX))
    asm.save(p)
    print(p, os.path.getsize(p) / 1e6, 'MB')
    r = cq.importers.importStep(p)
    bb = r.val().BoundingBox()
    print('solids', len(r.solids().vals()), 'bbox', bb.xmin, bb.xmax, bb.ymin, bb.ymax, bb.zmin, bb.zmax)


if __name__ == '__main__':
    main()
