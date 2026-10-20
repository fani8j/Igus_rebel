# integration - left suction tool + right LEHF gripper around one carton at the shelf handoff

> **Re-run 2026-10-14 with the left tool switched to suction tool v5** (`../suction_tool_v5`, canonical; 3 pads, one Gemini 305, one printed PA12-CF body, 0.204 kg). `handoff_check.py`, `render_handoff.py` and `export_step.py` take the suction folder from a constant `DEFAULT_SUCTION = 'suction_tool_v5'`; `--tool suction_tool` (or env `SUCTION_TOOL=suction_tool`) runs v3 again. Outputs: `data/handoff_check.json` (v5), `data/handoff_check_v3.json` (v3 re-run the same day with the same code), `data/handoff_check_v3_2026-10-05.json` (the original v3 run, unchanged); renders `renders/handoff_box{3,4}_*.png` (v5) and `renders/v3/` (v3); `step/handoff_box3.step` (v5; the v3 one, 5.8 MB, was moved to `/media/farhan/My Passport/igus_nvidia/scratch_integration/handoff_box3_v3.step`).
> Result v5 vs v3 (all 5 boxes x flush / 15 mm proud / 12 mm-neighbour cases, yaw 0 and yaw 180, 2 mm gripper descent sweep): **every clearance is identical to v3** because the tightest pairs are set by the unchanged pad geometry and the gripper: worst unexpected pair 8.062 mm (finger column vs box, yaw 0 and 180), suction<->gripper 10.805 / 11.068 / 11.068 / 11.746 / 10.124 mm for boxes 1-5, suction<->floor 39.6 / 44.6 / 34.6 / 49.6 / 39.6 mm, suction tool vs neighbours 80.1-80.2 mm (15 mm proud), nearest non-pad tool part to the target: front nut 20.5 mm, descent sweep minimum 11.0 mm with no contact below 3 mm, suction vs gripper wrist >= 154 mm. The new camera, collector bay and trunk (all at tool z <= 118, 21 mm below the neighbours' proud faces when pressed) never become the closest pair: the smallest suction-involved distance other than suction<->gripper and the pads' own contacts is the 34.6 mm floor clearance (box 3). Asserted constants (suction_tcp 159, compressed 154, gripper_tcp 193.2, top stop 167.5): all deltas 0.000. **What changed is the mass budget** (URDF inertials, carton 0.15 kg): tool 0.2782 -> 0.2038 kg, moment at the handoff pose (tool + carton) 0.460 -> 0.441 N m, torque about link6 z 0.054 -> 0.014 N m (CoG 19.8 -> 7.1 mm off axis in y), payload margin 1.57 -> 1.65 kg.
> The v3 run below the line (2026-10-05) and everything under "Method" still describes the method; the tables further down are the older v1 numbers.

> **Re-run 2026-10-05 with suction tool v3 and gripper tool v3** (suction lip z 159 / compressed 154, remote ejector so no envelope; gripper stop face `GRIPPER_STOP_GAP` = 4 mm above the top edge, position-based registration). `handoff_check.py` reads the collision-piece count from the manifest and adds an ejector envelope only if `../suction_tool/step/ejector_envelope.step` exists. Results, `data/handoff_check.json`:
> * yaw 0 and yaw 180: worst unexpected 8.1 mm (finger column vs box, all boxes and neighbour cases); suction<->gripper 10.8 / 11.07 / 11.07 / 11.75 / 10.12 mm (boxes 1-5); suction<->floor 39.6 / 44.6 / 34.6 / 49.6 / 39.6 mm.
> * open-finger descent sweep: min 11.0 mm for every box/case. The v3 procedure pre-closes the fingers to t + 4 mm for the last part of the descent; that state has 2 mm per side to the box by design and is not part of this sweep (the suction<->gripper gap is along the box length and does not change with the jaw opening).
> * Asserted constants: suction_tcp 159 / compressed 154, gripper_tcp 193.2, top_stop_frame 167.5 (all deltas 0.000).
>
> The tables and text below are the **v1** run (lip 165, stop on the edge) and are kept for reference; v1-specific numbers no longer apply.


Read-only check of `../suction_tool` and `../gripper_tool` against `../common/{params,handoff}.py`. Nothing in those folders was changed. No ROS node was started.

## Method
* Frame W (common/handoff.py): origin on the shelf floor at the front edge, +X lateral, +Y into the shelf, +Z up. Target pulled out `PULL_OUT = 100` mm, resting on the floor z = 0 (modelled as a slab, top face z = 0). Poses from `suction_link6_pose / gripper_link6_pose / box_pose / neighbour_poses`.
* **Constants asserted against the URDFs** (< 0.1 mm, all deltas 0.000): suction_tcp 165 / compressed 160, gripper_tcp 193.2, top_stop_frame 167.5, finger joint z 150, TCP x,y = 0. Also against the meshes: pad lip z 165.0, stop face z 167.5, finger pad centre z 193.2, pad crest |x| 4.0, pad width 50. (`data/handoff_check.json -> assertions`)
* Geometry: suction = the 72 URDF collision pieces (convex, coacd hulls, ~1.1x volume: slightly conservative) + the reserved ejector envelope (box x 38..117.8, y +-39.5, z 71.2..86.2). Gripper = URDF collision pieces; fingers placed with the prismatic joints (origin z 150, left -X, right +X, gap = 8 + 2 d; closed on thickness t: d = (t-8)/2, so pad crests touch the faces). `finger_*_ledge.stl` in `gripper_tool/meshes/collision` is a stale file (not in the URDF) and is ignored.
* **Wrists: real meshes, not capsules.** `link6.dae` and `link5.dae` as in `igus_rebel2.urdf` (visual origin (1.345, 0, -0.52) mm; joint6 origin z = 76.8 mm, joint6 = 0), exact triangles. Note the link6 mesh flange face is at z = 52.2, the tool CAD starts at 52.7: 0.5 mm gap tool-wrist (min distance S-SW and G-GW = 0.498), the tools themselves are not checked against their own wrist.
* Distances: python-fcl (exact convex/triangle distance; overlap -> negative penetration depth). python-fcl is not in the cad venv; it was installed with `pip --target` into `/media/farhan/My Passport/igus_nvidia/scratch_integration/pylib` (the script puts it on `sys.path`; reinstall with `pip install --target ... python-fcl` if lost; it needs the venv's numpy 2.2, do not keep the numpy it pulls in).
* Groups: S (suction tool), SW (its wrist), G (gripper tool, closed on t), GW (its wrist), B (target), N1/N2 (neighbours, same h and l as the target), F (floor). Tool vs its own wrist is not checked.
* **Expected contacts (reported, not failures)**: S pads (`pad_*`) <-> B spine; G `finger_pad_*` and `finger_*_block` (block sits 0.4 mm behind the rib crests) <-> B faces; G `base_top_stop_bar` <-> B top; B/N/F touching each other. Everything else must be >= 3 mm (`CLEARANCE_MIN`).
* Neighbour cases: (a) flush, (b) 15 mm proud, (c) 15 mm proud with thickness 12 (target 15/16) or 16 (target 12).

Run: `"<cad venv>/bin/python" handoff_check.py` (~40 s, writes `data/handoff_check.json`); `render_handoff.py` (renders, ~30 s, software z-buffer on the visual meshes), `export_step.py` (STEP).

## Result 1 - handoff state, min distance (mm), yaw 0 (camera on +X_w)
Identical for neighbour cases a/b/c wherever neighbours are not involved (the suction tool and gripper pads are 80+ / 14+ mm from any neighbour because the target is pulled out 100 mm and the neighbour spines are at y >= -15).

| box | worst unexpected pair | S-G (suction vs gripper) | S-B nearest non-pad (nut) | S-floor | G-B (non-pad) | G-N a / b / c | S-N (b) | wrists (min of SW-*, GW-*) |
|---|---|---|---|---|---|---|---|---|
| 1 (15x230) | **7.0** finger column - B | **10.8** pad_R - finger block | 20.5 | 20.0 | 7.0 | 29.3 / 14.3 / 14.3 | 80.2 | 107.8 (SW-B), 116.3 |
| 2 (16x240) | **7.0** | **11.07** | 20.5 | 25.0 | 7.0 | 29.3 / 14.3 / 14.3 | 80.2 | 107.8, 116.3 |
| 3 (16x220) | **7.0** | **11.07** | 20.5 | 15.0 | 7.0 | 29.3 / 14.3 / 14.3 | 80.2 | 107.8, 116.3 |
| 4 (16x250) | **7.0** | **11.08** | 20.5 | 30.0 | 7.0 | 29.3 / 14.3 / 14.3 | 80.2 | 107.8, 116.3 |
| 5 (12x230) | **7.0** | **10.12** | 20.5 | 20.0 | 7.0 | 29.3 / 14.3 / 14.3 | 80.1 | 107.8, 116.3 |

Expected contacts: pads press 5.0 mm into the spine (= the buffer compression, TCP 160 vs lip 165); the three pads' lips: 5.0 (centre of each pad), outer parts 3-5.5; gripper pads touch the faces (0.0), finger blocks 0.4 mm off, top-stop bar on the top edge (0.0). Wrists are far away (>= 107 mm). **No failure: every non-expected pair is >= 7.0 mm in all 5 boxes x 3 neighbour cases (15 configurations).** The suction-vs-gripper gap (10.1-11.1 mm) is the tightest cross-tool value: it is a y-gap, the suction pad (y <= -95) against the gripper finger block (starts ~ y = -85.7, i.e. s ~ 14 mm behind the spine); it does not depend on the box height.

## Result 2 - gripper descent sweep (fingers open 56 mm, lift 150 -> 0, 2 mm steps, suction at handoff)
No unexpected contact and nothing < 3 mm at any lift, for any box and any neighbour case (`first_lift_below_3mm = None`, 15 sweeps x 76 steps). Min distance over the sweep **8.5 mm** (at lift 0: yoke riser `base_top_stop_neg` vs the box top corner, same for every box). Other minima over the sweep: S-G 27.6, G-N 29.6 (flush) / 15.0-16.2 (proud, the finger blocks vs the neighbour), G-floor >= 163, S-GW >= 133.8. The stop bar touching the top edge at lift 0 is the intended contact. Note the S-G gap shrinks from 27.6 (open) to 10-11 mm when the fingers close, so the closed state at the handoff is the critical one, not the descent.

## Result 3 - pad-row height (default `pad_row_centre_z = h/2`)
Suction top end (pads, bracket, camera arm; excluding the reserved ejector envelope, which is +3.8 mm higher) vs gripper: the suction head tops at z = h/2 + 114 (box 3: 224.0; box 4: 239.0; incl. ejector 227.8 / 242.8), the gripper's lowest parts are the finger blocks at z = h - 56.4 (box 3: 163.6). So **in height they overlap by ~50-65 mm** (z-gap suction top -> gripper lowest = -64 / -54 / -59 / -49 for boxes 3 / 2 / 1,5 / 4), and they are separated only in y (10-11 mm, above). Gripper body / wrist are far above: gripper wrist lowest z = h + 115, 107-122 mm above the suction top; suction top to gripper wrist 3-D >= 133.8 mm; suction wrist to gripper >= 125 mm.
3-D min suction <-> gripper (S/SW vs G/GW) at the default height: 10.8 / 11.07 / 11.07 / 11.08 / 10.12 mm (boxes 1-5): **all >= 10, so no search was required**; it was run anyway (1 mm steps over zc in [100.4, h - 100.4]):

| box | default zc | clearance at default | best zc (range edge) | clearance at best |
|---|---|---|---|---|
| 1 | 115 | 10.8 | 101 | 11.5 |
| 2 | 120 | 11.07 | 101 | 16.7 |
| 3 | 110 | 11.07 | any (flat) | 11.07 |
| 4 | 125 | 11.08 | 101 | 23.3 |
| 5 | 115 | 10.12 | 101 | 10.8 |

The y-gap (11.07) is a ceiling for the boxes <= 230; lowering the pad row only helps the tall boxes 2 and 4 (by lifting the suction pads below the finger blocks) and costs floor clearance (S-floor drops from 15 to ~5.6 mm for box 3 at 101, and the spine-end margin to 25.0 mm). **Recommendation: keep `pad_row_centre_z = h/2`.** For 240/250 mm boxes `zc = 105` would give 14-20 mm if a larger margin is ever wanted.

## Result 4 - suction rotated 180 deg about its z (camera on -X_w), min distance (mm)
| box | worst pair | value | without ejector envelope |
|---|---|---|---|
| 1 | ejector envelope - floor | **-2.8 (collides)** | 1.0 (carrier_00 - floor) |
| 2 | ejector - floor | 2.2 | 6.0 |
| 3 | carrier_00 - floor | **-4.0 (collides)** | -4.0 |
| 4 | finger column - B | 7.0 | 7.0 |
| 5 | ejector - floor | **-2.8 (collides)** | 1.0 |

With the camera on -X_w the tool's long side (tool x up to +114 -> -Z_w) reaches below the pad row and hits the shelf floor for h <= 230 (S-G is unchanged, 10.1-11.1). **Yaw 0 is better** (nothing below 7.0 mm, floor clearance 15-30 mm). Yaw 180 would need `pad_row_centre_z >= ~121`.

## Result 5 - mass / moment (URDF inertials, carton 0.15 kg; moment about the flange face centre (0,0,52.7); `data/handoff_check.json -> check5_mass`)
| | tool mass | tool CoG link6 (mm) | + carton: mass / CoG | moment tool / tool + carton at handoff pose (N m) | worst orientation (z horizontal) tool + carton | payload margin (2 kg) |
|---|---|---|---|---|---|---|
| suction (L), box 3 | 0.437 kg (incl. ejector envelope 0.081) | (11.9, 10.5, 93.8) | 0.587 kg / (8.9, 7.8, 142.7)* | 0.18 / 0.52* | 0.52 | 1.41 kg |
| gripper (R), closed on 16 | 0.946 kg | (2.9, -0.1, 110.1) | 1.096 kg / (2.5, -11.3, 133.0)* | 0.03 / 0.12* | 0.87 | 0.90 kg |
(* box 3; per-box values in the JSON; box 4: suction 0.55, gripper 0.157 N m at handoff.) Suction arm has z horizontal at the handoff, so the pose is the worst case; the carton rests on the floor there, so the carton share is an upper bound. Gripper arm (z down) has only the lateral CoG offset in its handoff moment. Torque about link6 z: suction 0.045, gripper 0.0 N m. Both are within the 2 kg payload and the sub-1 N m flange loads of the individual READMEs.

## Findings
1. **All five boxes fit together at the handoff with the specified default geometry**: no unexpected contact, >= 7.0 mm worst pair, >= 10.1 mm between the two tools, descent sweep clear. Nothing needs to change in either tool.
2. Tightest cross-tool value is 10.1 mm (box 5, 12 mm) and 10.8-11.1 mm for the others. It is a y-gap between the suction pad bodies (y <= -95) and the gripper finger blocks (start ~ y = -85.7). It is not margin-rich: spine tilt (+-3 deg assumed in common) and shelf lean were NOT modelled and would eat into it; re-run with a tilted `box_pose` if those matter.
3. **Keep suction yaw 0 (camera on +X_w).** Yaw 180 puts the carrier/ejector envelope through the shelf floor for h <= 230 mm (-4.0 mm on box 3).
4. The finger block sits 0.4 mm behind the TPU rib crests (expected contact set); with a carton thicker than the closing width the ribs compress by up to 0.4 mm per side before the rigid block touches.
5. `gripper_tool/meshes/collision/finger_{left,right}_ledge.stl` are stale files (replaced by the fixed top-stop); not referenced by the URDF.
6. The link6.dae flange face is at z = 52.2 in the link6 frame vs 52.7 in the tool CAD (common FLANGE_Z): a 0.5 mm gap between each tool and its wrist mesh, harmless but it means the real stack-up depends on the true flange plane.

## Not checked (scope)
Shelf lean +-10 deg (roll of the tool about z), spine tilt +-3 deg, neighbours taller or deeper than the target, more than one neighbour per side (a third carton beyond the open finger pads is not modelled; in y the pads stay >= 14 mm in front of any neighbour spine because of the 100 mm pull-out), the rest of the arm (links 1-4), the vacuum tubing beyond the straight segments in the CAD, overlap volumes by boolean (penetrations are given as fcl max penetration depth; the 5.0 mm pad penetration is the designed compression).

## Files
`handoff_check.py` (re-runnable), `render_handoff.py`, `export_step.py`, `data/handoff_check.json`, `renders/handoff_box{3,4}_{iso,side,front}.png` (flush neighbours drawn stippled/ghosted, link5/link6 in grey), `step/handoff_box3.step` (80 solids: suction 33 + gripper closed16 44 + target + 2 neighbours, shelf frame W; wrists not included).
