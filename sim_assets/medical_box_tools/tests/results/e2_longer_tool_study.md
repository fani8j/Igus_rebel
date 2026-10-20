# E2 (shelf 6, first carton next to the left wall): can a longer suction tool pick it?  Answer: NO (study only, nothing built)

Geometry (cabinet_geometry.json): wall inner face x = -0.49; FrontTrim_Left x -0.49 .. -0.472, y 0 .. 0.012 (full height); E2 carton angiolite_001 centre x = -0.480, 16 mm thick -> x -0.488 .. -0.472, spine y = 0.070 (front plane y = 0).
The carton's whole thickness lies inside the lateral band of the front trim, 58 mm behind it: no object that is laterally inside -0.49 .. -0.472 can move straight in +Y to the spine.

## 1. Diagnosis (v5, straight approach, pads centred on the carton, point-sampled collision meshes vs cabinet boxes; `tools/e2_study/e2_first.py`)
First contact as the tool advances (distance before the pressed pose, compressed TCP at the spine):
| part | hits | first contact |
|---|---|---|
| **pads (bare, 9.8 mm wide)** | front trim | **60 mm before the pressed pose** (already when the lip enters the opening) |
| front nuts, tool body (pad bar, stem) | wall and front trim | 49 mm |
| washers phi20 | wall and trim | 43 mm |
| rear nuts | trim | 42 mm |
| elbows KQ2L04-M5 (z 83.5..97.5, 61.5..75.5 mm behind the lip) | trim | 13 mm |
Other v5 facts: the collector bay reaches y = -52.7 mm on the wall side (z 76..98, only 0..14 mm inside the front plane when pressed), stem and bar are +-11.6 mm wide (wall gap 10 mm), washers +-10, elbow barrels -8.3 / +4.5, camera +53.7 mm on the stock side (harmless at a left wall, it is in front of the spine plane over the stock).
Wall-side half width of v5 along the pad axis (tool z, mm): hub 28 (52.7..57.7), stem 28..11.6 (57..130), collector bay 52.7 (74..98), 38.9 (98..104), bar/stem 13..11.6 (104..134), pads 4.9 -> 2.4 (134..159). Stock side: camera 53.7 (92..122), else as above.

## 2. Options (what a longer / narrower / asymmetric tool can and cannot do)
* Pad geometry is the limit, not tool width. A straight approach needs the pad on the stock side of the trim face: pad axis x >= -0.472 + 4.9 + 2 = -0.465 -> carton centre >= 25 mm from the wall face (28 mm with the 4.5 mm elbow, 33 mm with the 11.6 mm stem); E2 is at 10 mm. No tool length changes that: the pad body, its inlet fitting (elbow 63..75 mm behind the lip = exactly the trim depth band) and the bar nuts sit at fixed distances behind the lip because of the pad itself (lip -> inlet face 61.5 mm, spine 70 mm inside the plane, trim 12 mm deep).
* a) longer narrow nose: moves the stem / collector out of the opening but the pads, nuts, washers and elbows cannot move: still collides at 60 / 49 / 43 / 13 mm before the press. FAILS.
* b) asymmetric tool (collector on the stock side, stem shaved to 8 mm on the wall side): removes the wall hits, trim hits remain. FAILS straight.
* c) only a YAWED approach (tool z axis turned about the cabinet vertical, tip toward the wall, rear toward the stock side) lets the pads pass the trim. Minimum yaw with clearance >= 2 mm to wall / trim / stock (`tools/e2_study/e2_yaw.py`, pads laterally offset 0 / 2 mm toward the stock):
  - bare pads only (ideal limit): 13 deg / 12 deg
  - v5 as built: none up to 40 deg (collector and stem keep hitting wall and trim)
  - v5 with collector + bay moved to the stock side (4 mm lower, under the camera): 30 deg / 28 deg
  - plus stem / bar / washers / nuts shaved to 8 mm on the wall side: 21 deg / 17 deg (remaining blocker: the stem inside the trim depth band)
  - elbows removed does not help further (the stem is the limit).
  A nose lengthening / narrowing of the stem so that only a thin neck crosses the trim band would bring it towards the 12-13 deg of the pads alone.
  Cost: the pads press the spine 12..21 deg off normal (ZP2-T4030 angle tolerance not in my data: needs a bench test), the pull must be along -Y with the tool yawed, and the 2 mm lateral offset of the pad axes is already used.

## Conclusion / recommendation
A longer tool does NOT make E2 pickable; v6 was therefore NOT built (v5 untouched, nothing under cad/ changed). Options that would work, in order of effort: (1) cabinet side: leave the first carton >= 25-33 mm away from the wall (or a narrower / chamfered trim) -> straight approach with v5; (2) yawed approach >= 17-21 deg with a v6 that moves the collector to the stock side and shaves the wall side (needs a pad tilt test); (3) treat E2 as manual / not pickable by suction.
