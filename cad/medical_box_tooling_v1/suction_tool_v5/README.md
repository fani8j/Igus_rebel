# suction_tool v5 - left-arm suction tool, ONE printed body, ONE camera (canonical)

Final left-arm tool for the medical-box shelf pick (decisions of 2026-10-14: no Zebra SE4710, no PL3307, one Orbbec Gemini 305, fewest and lightest parts, vacuum source still open and to be simulated in Isaac Sim first).
3 x SMC ZP2-T4030WNK10 pads at x = -60 / 0 / +60 (lip z 159, compressed 154, unchanged hand-off contract), a push-in 3-to-1 collector with one phi6 trunk, and the Gemini 305 bolted straight onto a single printed PA12-CF body (flange hub + C-section stem + pad bar + collector bay).
Parametric CadQuery, link6 frame, millimetres (flange face z = 52.7, pad axes +z, pad row along +x). `../common/params.py` is imported read-only and was not changed. `../suction_tool` (v3) and `../suction_tool_v4_concept` are untouched.
Nothing was run on a robot or with ROS nodes; `xacro` + `check_urdf` were run with `ROS_DOMAIN_ID=169 ROS_LOCALHOST_ONLY=1`.

## Rebuild
```
PY="/media/farhan/My Passport/igus_nvidia/venvs/cad/bin/python"      # cadquery 2.7, OCP 7.8, trimesh, coacd, shapely
export TMPDIR="/media/farhan/My Passport/igus_nvidia/scratch_suction_tool/tmp"
nice -n 19 $PY build.py      # ~1 min: step/, stl/, meshes/visual + collision (coacd), urdf/, data/mass.json, data/printed_parts.json
nice -n 19 $PY verify.py     # ~2 min: data/verify.json (every number below)
nice -n 19 $PY render.py     # renders/*.png   (render.py <assembly|exploded|section|pressed|camera|compare> for one set)
```
Files: `params.py`, `parts.py` (geometry), `build.py`, `verify.py`, `render.py`, `support_model.py` (bridge-aware support estimate), `lib.py` / `meshlib.py` / `massprops.py` (helpers copied from v3, mass model now PA12-CF).

## Final numbers (vs targets)
| item | v5 | target / v3 / v4 |
|---|---|---|
| total mass (CAD parts) | **0.2038 kg** (printed body 56.8 g, camera 68 g, 3 pads 36 g + 3 elbows 9 g, 6 nuts 15 g, 3 washers 3 g, collector 12 g, tubes 4 g); **0.2088 kg with screws** (6 x M3 CSK, 2 x M3 pan: 5 g) | target <= 0.22 kg **met**; v3 0.278, v4 0.332 (Al bar) / 0.265 (PA12-CF bar) |
| CoG (link6) / inertia | (0.08, 7.13, 106.5) mm; Ixx 2.26e-4, Iyy 3.17e-4, Izz 3.14e-4 kg m2 in the URDF | v3 CoG (-0.5, 19.8, 98.2) |
| printed parts | **1** (`tool_body`) | target 1 (max 2); v3 3, v4 2 + Al bar |
| solids in the CAD | 22 = 1 printed + 21 purchased (3 pads, 3 elbows, 6 nuts, 3 washers, 3 tubes, collector, trunk stub, camera) | v3 23, v4 20 |
| envelope x y z | **150.8 x 106.4 x 106.3 mm** (y -52.7 .. +53.7), hull 716 cm3 | v3 150.8 x 148.6 x 106.3 (hull 1151), v4 150.8 x 95.7 x 106.3 (hull 723) |
| camera y reach | **53.7 mm** (camera body; the whole tool +y side) | target <= 60 **met**; v4 65.7 |
| camera blocked by bar / pads / body | **0.000 %** of the 88 x 65 deg depth FOV at the approach pose (3 origins, 45 x 33 rays each, to the spine plane; also 0.000 % within 300 mm) | target <= 10 % **met**; v4 29 % |
| aim point depth at the approach pose | **91.0 mm** (spine centre, lip + 50 mm) | target >= 90 mm good depth **met** |
| spine +-1 neighbour in the FOV | y = -25.3 .. +90.6 mm visible at x = 0 on the spine plane (needs +-24 mm for 3 x 16 mm); +-60 mm along x inside the 88 deg with 10.6 deg margin | **met** nominally, margin only 0.6 deg on the -y side (see Camera) |
| pad bar | (b) pull 3 x 9.28 N: outer pad deflection **0.031 mm**; safety factor >= **6.9** in every case (3.4 with a x2 stress-concentration factor) | targets <= 0.3 mm and SF >= 3 **met** |
| min wall | **3.00 mm** (ray thickness, 60 000 samples, no sample below 3.0) | target >= 3 mm **met** |
| supports | build bar-face-down: **11.5 cm3** support (bridge-aware voxel model, 22 % of the part); raw v3/v4-style solid bound 130.5 cm3 (87 cm3 best axis) | v3 67.6 / v4 1.5 cm3 with the same voxel model (114 / 33 cm3 raw bound) |
| pressed on boxes 1-5 | pads vs 15 mm proud neighbours 3.5 / 4.0 / 4.0 / 4.0 / **2.0** mm; camera >= **21.0 mm**; bar front 5.5 mm from the neighbours' proud plane; nothing else closer | same pad numbers as v3 / v4 |

## What changed vs v3 / v4
* **One camera** (Gemini 305). Scanner, PL3307 decoder, cradle, flex slot, `scanner_optical_frame` removed. Barcodes are decoded in software from the colour stream (see Barcode pixels).
* **One printed body** instead of v3's carrier + camera arm + scanner arm and v4's hub/stem + pod + milled Al bar. No aluminium, no heat-set inserts (no screw goes into plastic: the 6 flange screws go into the robot flange, the 2 camera screws into the camera's own M3 holes), 3 steel washers under the rear nuts. The bar is a 3 mm skin with rear bosses and edge ribs, so the printed bar is very stiff for its mass (below).
* **Pad variant**: ZP2-T4030WNK10-**B5** (M5 female inlet) + **KQ2L04-M5 swivel elbow** (common/params.py `PAD_FITTING`, 14 mm tall, ASSUMED) on every pad instead of the v3/v4 one-touch `-04` straight fitting. The pad body, thread, lip z, pitch and nut faces are the same as in v3/v4; only the inlet fitting differs (BOM: +3 elbows). The elbow lets every tube leave horizontally at z = 88 instead of dropping to the hub (v3/v4 loops).
* **Tubes**: no loop below the bar. Outer pad tubes leave their elbows horizontally (z = 88.0), run in one plane to the collector ports (single R10 bend, 54 deg); the middle tube is a straight 27.5 mm. Everything is at z 86-90, 37 mm behind the bar's rear face, and cannot become taut: the elbow is on the clamped body, the 10 mm stroke only moves the sliding head in front of the bar.
* **Collector in a bay on the stem's open -y side** (not inside the stem: the middle pad's elbow and rear nut already fill it), at the tube axis height z = 88, and the **trunk leaves straight down (-z)** at y = -44.5: outside the hub rim (r 28) and outside the ReBeL wrist housing (link6 / link5 meshes reach r = 40.2 / 40.0; the trunk's outer surface is 1.3 mm clear of it), so it can follow the arm with no bend inside the CAD. phi6 PU bend radius assumed 15 mm when it is dressed.
* **Camera lies flat on the stem +y wall** (tilt 0, no seat/wedge/pod): bottom face on the wall, 2 x M3 through the 3 mm wall from inside the open stem; USB-C relief lug on the **+x** side. v3's lug was on -x, the wrong side (see Camera).

## Design (link6 frame, mm)
| item | value |
|---|---|
| hub | disc phi56 x 5 (z 52.7 .. 57.7), phi14 centre bore, 6 x M3 CSK on PCD 43 (angles 45 + 60k), screws M3x10 (5 mm in the flange); 12 hub gussets (3 mm, 45 deg, every 30 deg, staggered clear of the screw columns) and a lower -y wall (z to 74.1) make the hub plate self-supporting when printed bar-face-down |
| stem | C-section x +-15.5 (31 wide) x y -11.6 .. +11.6, 3 mm walls (+y wall and +-x walls, open toward -y), z 57.7 .. 130.5; the collector bay hangs off the open side; `y <= 11.6` everywhere above z = 119 (the 24 mm keep-out slab) |
| pad bar | 150.8 x 23.2 stadium skin 3 mm (z 130.5 .. 133.5, front face 133.5 = nut seat), phi22 rear bosses (z 127 .. 130.5; pocket floor 3.3 mm), 14.25 AF x 3.2 hex pockets in the front (keying the front nut), phi10.5 holes, edge ribs 3 x 11.5 mm at y = +-[8.6, 11.6], x 15.5 .. 54 (merging into the outer bosses); all inside the keep-out slab |
| nuts | front nut captive in the pocket (z 130.5 .. 133.5); rear nut z 123 .. 126 + steel washer 126 .. 127 on the boss. Assembly torque on the printed bar **0.3 N m** with Loctite 243 (SMC's 2.5-3.5 N m is for metal brackets and would crush 14 AF of PA12-CF: 32 MPa at 3 N m) |
| elbows / tubes | KQ2L04-M5, axis z = 88.0 (9.5 below the inlet face 97.5, ASSUMED), 7 mm barrel; outer tubes (+-60 -> collector ports at x = +-17, y = -44.5): bend R10 54 deg, straight 42.7 / 5.9 mm, length 58.0 mm; middle tube straight 27.5 mm to the collector +y port |
| collector bay | two cheeks (3 mm, x +-[12.5, 15.5], z 80 .. 98) with open-topped slots for the two phi9 side ports, two ledges (3 x 6 mm) under the body, collector 24 x 10 x 16 proxy (same ASSUMED 12 g block as v3: phi4 ports on -x, +x, +y, phi6 trunk on the bottom face) centred (0, -44.5, 88); drops in from +z |
| camera | Gemini 305 front-face centre (0, 32.65, 118.0), optical axis +z (tilt 0), bottom face on the stem +y wall (0.05 mm gap), highest point z 118.0 (keep-out 119), y max 53.7; 2 x M3 at x = +-10, 12 mm behind the front face, M3x8 + 0.5 mm washer (4.5 mm thread, max 4.8); cable tie lug on the +x stem wall below the USB-C plug |
| frames (URDF) | `suction_tcp` (0, 0, 159), `suction_tcp_compressed` (0, 0, 154), `wrist_camera_mount` (0, 32.65, 118) = front-face centre (camera x = tool x, z = optical axis), **`wrist_camera_color_optical_frame`** (same origin, rpy 0 0 pi = ROS optical: x right, y down, z forward; the +-9 mm lens offsets are not applied), `pad_{1,2,3}_suction_frame` (x -60 / 0 / +60, z 159, +Z out of the lip, Isaac Sim Surface Gripper attachment points). Macro `left_suction_tool(prefix, parent, mesh_dir)` and link `suction_tool` as in v3; no scanner frame |
| vacuum | unchanged circuit: collector -> one phi6 trunk -> pump enclosure (source undecided; see `../common/params.py` `VACUUM_PUMP`). Tool-side volume about 4 cm3 as v3. Use the v3 README's vacuum-gate model only as a starting point |

## Camera (Gemini 305)
**Placement**: lens face on the +z side at z 118, y 32.65, tilt 0. The constraints: the whole camera must stay at z <= 119 (keep-out) and reach y <= 60; at that height the optical axis along +z passes the pad row at y = 32.6 mm, so the lip outer edge (y = 2.4, z = 159) is seen under 36.4 deg from the lens centre, outside the 32.5 deg half FOV in the tilt plane. Result: **no ray hits a pad, the bar, the stem, the collector or the tubes** before the spine plane (0.000 %). A tilt toward -y would gain FOV margin on the neighbour side but needs a wedge between the camera and the wall that collides with the flange-screw access (a 5 deg trial ended with 0.5 mm webs next to the (15.2, 15.2) screw column); the flat mount also saves the wedge's mass.
**Approach pose (document)**: pad lip plane **50 mm in front of the spine** (spine at link6 z = 209, tool 55 mm backed off from the pressed pose z = 154); depth of the spine centre along the optical axis 91.0 mm (>= 90 mm good depth), range 96.7 mm. At that pose y = -25.3 .. +90.6 mm of the spine plane (x = 0) is inside the depth FOV: target + both neighbours (12-16 mm boxes: up to 48 mm wide) fit. Along x the +-60 mm samples have 10.6 deg of horizontal margin.
**Limits (honest)**: (1) the -y margin is only 0.6 deg; with the datasheet FOV tolerance (V = 65 +- 3 deg) the 48 mm width needs lip + 59 mm (depth 100 mm); nominal needs lip + 48 mm. (2) A 15 mm proud neighbour casts a shadow on the target spine because the camera sits 33 mm to the +y side: at the approach pose the +y neighbour hides the target spine's +y part, visible width at x = 0 (box 5, 12 mm) **6.7 mm of 12**, box 4 (16 mm) 11.1 mm of 16; the shadow shrinks slowly with standoff (box 5: 40 % of the width visible at lip + 30, 53 % at lip + 45, 56 % at lip + 50, 61 % at lip + 60, 71 % at lip + 90). With flush neighbours the whole width and both neighbour fronts are visible (100 %). Whether a code on a recessed spine can be read therefore depends on where it sits across the spine; no mitigation exists with a camera beside the pad row except a different yaw.
**USB-C side**: vendor STEP (recess vertices on the +x face at camera t = -22.6 .. -12.7, |u| < 7.6) and the datasheet right-hand view agree: the receptacle is on the **+x face** (camera x = tool x). v3's relief lug was on -x (wrong side; v4 README had noted it). v5: tie lug on the +x stem wall below the plug (x 15.4 .. 23.4, y 3.4 .. 11.6, z 85 .. 95, tie slot 1.6 x 3.6 through it). The plug and its cable bend are not modelled (right-angle plug about 16 x 9 x 15 mm, ASSUMED, fits beside the camera: nothing within x 21 .. 35 of the +x face except the lug).
Overview image: `renders/camera_view.png` (side view with FOV wedge, rays, spines and neighbours; occlusion map).

## Barcode pixels per module (software decode on the colour stream)
Colour 1280 x 800, H 94 deg: f = 596.8 px, pixel pitch = 2 d tan 47 deg / 1280 = 0.001676 d (d = depth along the optical axis, spine plane). px/module = module / pitch; DataMatrix 10 / 15 / 20 mil = 0.254 / 0.381 / 0.508 mm, EAN-13 0.33 mm. Targets 2.5 px/module.
| lip -> spine | depth d | pitch | DM 10 mil | DM 15 mil | DM 20 mil | EAN-13 | visible y at x = 0 |
|---|---|---|---|---|---|---|---|
| 20 mm | 61.0 | 0.102 | 2.49 | 3.73 | 4.97 | 3.23 | -6.2 .. +71.5 |
| 30 | 71.0 | 0.119 | 2.14 | 3.20 | 4.27 | 2.77 | -12.6 .. +77.9 |
| 33 | 74.0 | 0.124 | 2.05 | 3.07 | 4.10 | 2.66 | -14.5 .. +79.8 |
| 36 | 77.0 | 0.129 | 1.97 | 2.95 | 3.94 | 2.56 | -16.4 .. +81.7 |
| 45 | 86.0 | 0.144 | 1.76 | 2.64 | 3.53 | 2.29 | -22.1 .. +87.4 |
| **50 (approach)** | **91.0** | 0.153 | 1.67 | **2.50** | 3.33 | 2.16 | -25.3 .. +90.6 |
Smallest of the two image axes (the tilt axis is the same pitch at tilt 0; the exact projection is in `data/verify.json`) differs by < 1 %.
**Standoff for >= 2.5 px/module** (spine centre): DataMatrix **10 mil: d <= 60.6 mm (lip + 19.6 mm)**; **15 mil: d <= 91.0 mm (lip + 50 mm)**; **20 mil: d <= 121.3 mm (lip + 80 mm)**; **EAN-13: d <= 78.8 mm (lip + 37.8 mm)**. All are >= the 50 mm minimum depth of `common/params.py` (the colour stream does not need depth; the datasheet's own depth numbers are operating range 4-100 cm, ideal 7-50 cm); only 15 mil and 20 mil are also >= the 90 mm "good depth". The Gemini colour camera is fixed focus: its sharp range is not in the datasheet (ASSUMED sharp from about 60 mm); **measure it before relying on the 10 mil / EAN numbers.** In practice: approach at lip + 50 (sees everything, decodes 15 / 20 mil codes), then creep to lip + 20-37 mm for 10 mil and EAN-13 (the spine's +-6 mm width stays in the FOV down to about lip + 20; the neighbours leave the FOV below lip + 48 mm).
The pressed pose (spine at z 154) is d = 36 mm: closer than the 50 mm minimum depth and with the spine outside the FOV, so reading is only done in the backed-off poses above.

## Pad bar strength (PA12-CF printed, E = 3.5 GPa, allowable 25 MPa: the brief's conservative values; no layer-adhesion knock-down)
First-order beam model, no FEA: the bar is a cantilever from the stem face (x = 15.5) to the pad axis (x = 60) with EI(x) taken from the CAD cross-sections (skin + ribs + bosses, pad hole included: I = 2634 mm4 at the root, 231 mm4 at the pad), plus the rigid rotation of the stem top (C-section I = 29 234 mm4) under the net moment; bending stress M c / I along the bar and in the stem walls (+ axial).
| case | loads | outer-pad deflection | max stress | safety factor (x2 Kt) |
|---|---|---|---|---|
| (a) push-on | buffers compressed 3 N each + 20 N arm push on one outer pad (23 N) | 0.128 mm | 3.65 MPa (bar root) | 6.9 (3.4) |
| (b) pull | 3 pads x 80 kPa x 1.16 cm2 = 9.28 N each, symmetric | **0.031 mm** | 1.47 MPa | 17.0 (8.5) |
| (c) worst case, one outer pad only | 3 N + 20 N = 23 N on one outer pad (a single pad cannot hold more than 9.28 N in pull) | 0.136 mm | 3.65 MPa | 6.9 (3.4) |
| (b1) one outer pad holds | 9.28 N | 0.055 mm | 1.47 MPa | 17.0 (8.5) |
Nut bearing at the 0.3 N m assembly torque (preload 150 N + 23 N): front nut on the pocket floor 3.7 MPa (SF 6.7), rear washer on the boss 0.8 MPa (SF 33). At SMC's 3 N m the front nut would load the pocket floor with 32.5 MPa (> 25), so the torque limit is part of the design. The rib end at x = 54 merges with the boss circle, so there is no skin-only section next to the pad (with ribs to x = 49 the same calculation gave SF 3.4).
**Not covered**: layer adhesion (the printed bar is loaded in the layer plane when built bar-face-down), creep under the nut preload, the boss plate bending around the hole, vibration, the +-10 deg lean.

## Tubes, collector, buffer stroke
Elbow tube axis z = 88.0, 37 mm behind the bar's rear face (127.0), nothing below it. Outer tubes: min bend radius 10.0 mm (required 10), one 54.3 deg bend, straight 42.7 mm from the elbow barrel and 5.9 mm into the collector port; 3.1 mm of slack over the chord. Middle: straight 27.5 mm. Grooves/clips: the tubes are held by the elbow barrel and the push-in ports only; one zip tie around the cheeks over the collector holds the collector (the cheek outer faces are flat; no clip features were modelled). If the elbows moved with the head (not expected; ASSUMED from the catalog drawing: inlet on the rear of the clamped body) a 10 mm shift would lengthen the outer tubes' chord by 0.9 mm (3.1 mm of slack: not taut) but the straight middle tube (27.5 mm, chord +1.8 mm) would be taut: re-check on the real part. Buffer sweep 0..10 mm in 1 mm steps against every other part: 0 collisions, 1.5 mm gap adapter-to-bar at full stroke. Collision with the neighbour boxes in the pressed pose: pad heads touch the target only.

## Printability and process
`tool_body`: 50.5 cm3, 150.0 x 80.7 x 80.8 mm in the build orientation (fits a 220 x 220 bed). **Build with the bar front face (z = 133.5, the side with the hex pockets and pad holes) on the bed**, stem growing up (-z of link6 up), flange face on top: the pad holes and nut pockets are bed-side features (the pocket roofs are 14 mm bridges with the phi10.5 hole in them), ribs / bosses / stem walls are vertical or 45 deg, the 12 hub gussets and the lower -y wall carry the hub plate, the two collector-bay cheeks (they run from the stem to the bay end) carry the ledges, and the only supports left are under the cheek ends and a few small ledges: **11.5 cm3 support** in the bridge-aware voxel model (`support_model.py`: 45 deg rule + bridges up to 12 mm, 1 mm voxels), 22 % of the part. Other axes: 56 (-y), 91 (+z) .. 118 (+x) cm3. The simpler "every down-facing face x its free drop" bound used for v3 and v4 (no bridging) gives 130.5 cm3 for this orientation (87 cm3 for the best axis, -y) and 114 / 33 cm3 for v3 / v4; with the voxel model v3 = 67.6 and v4 = 1.5 cm3, so v5 is better than v3 but worse than the small v4 pod (v4 needed a separate milled aluminium bar). With SLS PA12 / PA12-CF powder no supports are needed at all. The flange face is the top surface of the print: check its flatness (mill or sand 0.2 mm) before mounting.
Wall check: inward-normal ray thickness on 60 000 surface samples, minimum **3.00 mm** (ribs, skin, cheeks, ledges, stem walls 3 mm; hex-pocket floor 3.3 mm; the flange access columns cut through ledges / the left cheek without leaving walls < 3 mm). Mass model: PA12-CF 1.20 g/cm3 solid x 0.90 effective fill (ASSUMED; 3 mm walls are essentially all perimeters).

## Assembly order (all steps swept in `verify.py`, section `access`: 0 mm3 blocked)
1. Body alone on the flange: 6 x M3x10 CSK from +z (phi6 access columns are cut through the skin for the two screws under the bar at (-20.8, 5.6) and (20.8, -5.6); plug them afterwards). The camera is fitted last (step 4); with it on, the camera body covers the screws at (15.2, 15.2) and (-5.6, 20.8), so mount the tool before the camera (the sweep in `verify.py` is done with the camera removed). Re-teach the TCP afterwards (no dowel / pilot on the ReBeL).
2. Pads: front nut (pre-threaded on the pad) in the hex pocket, pad through the bar from the front, steel washer + rear nut from behind (outer pads: deep 14 mm socket, OD 19, from -z before the elbows go on; middle pad: thin 14 mm open-end spanner from +y through the window in the +y wall), set the oval index along x, tighten to 0.3 N m, Loctite 243.
3. Elbows (hand tight, swivel to the tube direction), collector dropped into the bay from +z (ports slide into the cheek slots), tubes: pad elbows -> collector ports, trunk into the bottom port, one zip tie around the cheeks.
4. Camera: 2 x M3x8 + 0.5 mm washer, driven from the open -y side through the stem (45 mm of driver shank clears the collector), USB-C cable tied at the +x lug.
5. Trunk straight down along the arm (>= 40 mm to the first tie on the wrist); keep a service loop for joint 6.

## Verification (`data/verify.json`, run on this build)
| check | result |
|---|---|
| interference, all 231 pairs of 22 solids | **0 mm3** everywhere; 36 pairs within 3 mm are designed contacts (tool_body - camera 0.055 mm, - pads 0.3, - front nuts 0.125, - rear nuts 0.8-1.0, washers / nuts / tubes touching) |
| keep-out (nothing but pads and the bar above z = 119, bar within y +-12) | **pass**: only tool_body, front / rear nuts, washers above 119, all inside the 150.8 x 24 stadium and below 133.5; camera top 118.0, everything else <= 119 |
| pressed pose boxes 1-5 (neighbours 15 mm proud, exact OCC distance, pad heads retracted 5 mm) | min clearance 3.5 / 4.0 / 4.0 / 4.0 / 2.0 mm = the pad bodies against the proud neighbours (as v3 / v4); next closest tool_body / front nut 5.5 mm; camera 21.0 mm |
| camera | see above; 0 % blocked, depth 91.0, y max 53.7, +-24 mm in the FOV |
| strength, supports, wall | see above |
| tubes / trunk | bend R 10.0 (>= 10), loops none, trunk 1.3 mm clear of the wrist housing |
| access sweeps | 6 flange screws, 2 camera screws, 3 rear nuts, collector drop-in, 3 pads from the front: all free |
| STEP / STL | 22 STEP parts re-import (volume error < 1e-3); assembly.step 22 solids; `stl/tool_body.stl` watertight |
| lips / pad span | lip z 159.00 on all three; span 150.8 mm; margin to each spine end 34.6 (220 mm) .. 49.6 mm (>= 25 mm for boxes 1-5) |
| URDF | `xacro` + `check_urdf` (`ROS_DOMAIN_ID=169 ROS_LOCALHOST_ONLY=1`): root link6, 7 frames; 22 visual meshes, 55 convex collision pieces (hull volume 1.30 x part volume; tubes excluded), inertial from the CAD |

## Assumptions / unconfirmed (every ASSUMED value)
1. Collector part number and block size (24 x 10 x 16, 12 g, ports on three faces + bottom) are the v3 proxy, not a catalog part; the elbow height 14 mm (from `common/params.py`), tube axis 9.5 mm below the inlet face, 7 mm barrel, 3 g elbow mass; pad 12 g. The pad body is a proxy from the catalog drawing (no SMC STEP).
2. The elbow inlet is on the clamped (rear) body, the buffer moves only the head. If not, see Tubes.
3. PA12-CF 1.2 g/cm3, fill factor 0.90, E 3.5 GPa, allowable 25 MPa (brief), assembly torque 0.3 N m, nut friction K 0.2, min bend radius 10 mm (phi4) / 15 mm (phi6), screws 0.6 / 0.7 g.
4. Camera: FOV H 94 x V 68 colour, H 88 x V 65 depth (datasheet, +-3 deg), fixed focus (range not published), USB-C plug size, the colour sensor is taken at the front-face centre (lens offset +-9 mm not applied), the camera hull is the convex hull of the vendor STEP main solid.
5. Barcodes: module sizes as in the brief; the position of a code across the spine is unknown; recessed spines behind 15 mm proud neighbours are shadowed (above).
6. Screws, washers (camera), Loctite, zip ties, silicone plugs for the two skin access holes are not modelled.
7. The hub plate and the stem were not checked for the robot's flange loads or for vibration (the tool is 0.2 kg; moment at the handoff 0.44 N m with the carton).

## Files
`params.py parts.py build.py verify.py render.py support_model.py lib.py meshlib.py massprops.py README.md`
`step/assembly.step` (22 solids), `step/parts/*.step`, `stl/tool_body.stl` (mm), `meshes/visual/*.stl` (m, link6 frame, 22 parts), `meshes/collision/*.stl` + `manifest.json` (55 convex pieces, m, coacd; tubes excluded), `urdf/left_suction_tool.urdf.xacro`, `data/{verify,mass,printed_parts}.json`.
`renders/`: `assembly_{iso,front,side,top}.png`, `exploded_iso.png`, `section_x0.png` (through the middle pad, stem, collector bay, camera), `section_y0.png` (long section through the bar, the three pads and the stem), `pressed_on_box5_{side,iso}.png` (12 x 230 x 250 spine, heads retracted, neighbours 15 mm proud), `pressed_on_box4_{side,iso}.png` (16 x 250 x 295), `camera_view.png` (frustum, rays, occlusion map), `compare_v3_v4_v5_{iso,front,side}.png`.
