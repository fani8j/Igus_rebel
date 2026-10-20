# Robot stand CAD analysis (pump-box placement) - analysis only, nothing designed

Source: `/home/farhan/Downloads/2 arms step(軌).step` (107 MB, SolidWorks 2017 AP214, units mm). The source file and the rest of the repo are untouched.
Large intermediates (BREP dumps, tessellated meshes, voxel grids, raw search results, logs) are in
`/media/farhan/My Passport/igus_nvidia/shelf_pick_research/robot_stand/`. Machine-readable results: `data/stand_summary.json` (everything), `data/components.json`, `data/leaves.csv` (315 leaf solids), `data/candidates_targetA.json`, `data/regions.json`.

## 1. Short answer

* The "stand" is **not** a floor stand with a base plate. It is a **tall triangular column (100 mm plate, 1151 mm tall) with a 1250 mm belt lift (igus ZLW-20120S) on its front face**. The lift carriage carries a gusseted bracket, a 660 mm X rail (drylin W) and **both ReBeL arms standing upright side by side on two sliding carriages**. There is no AMR, floor plate, controller box, electronics or T-slot profile in the file.
* The column footprint on the AMR is only **100 x 397 mm (foot) + two 149 x 70 mm wing feet (398 mm wide overall)**. The operator's "no room on the AMR because of the stand" is not supported by the CAD: the stand itself takes very little deck. Whether there is deck room depends on the AMR geometry, which we do not have.
* **Fixed placements (AMR deck, column side, column foot wings) all force the tube to follow the 1250 mm lift stroke: 2.1 m (best, but inside the arm workspace) to 3.0+ m worst case** with the arm run included. They do not meet "<= 1.5 m along the arm".
* **Best place: the open pocket in the rail bracket on the lift carriage, between its two triangular gussets, under the X-rail shelf.** The box travels with the arms, so no drag chain. Box fits with >= 24.9 mm exact clearance (>= 40 mm uniform clearance is feasible in the pocket), no collision over the whole lift stroke and arm X slide, no part outside the stand envelope. Estimated tube to the left arm: **1.43 m (shortest) to 1.58 m** (0.32 to 0.47 m box-to-arm-base leg + 0.96 m along the arm + 0.15 m slack).
* Left/right is **inferred**, not stated (section 3). Arm A (x=-424) is the left arm if the robot faces +Z. The structure is mirror-symmetric about x=-654, so every candidate mirrors for the other arm.

## 2. What was loaded

XCAF read 4 s, transfer 43 s, tree walk with bounding boxes, volumes and BREP dump about 7 min (nice 19, 1 process); RAM was never a problem, no fallback to bbox-only was needed. The tree is flat: 16 top-level instances (NAUO1..16; NAUO1 holds two solids = the ZLW body and carriage) and 315 leaf solid instances (65 distinct solids). The two arms are 150 `D00xxxxxx.ipt` parts each. Names are Big5 mojibake in the STEP; decoded: 三角支撐 = triangular support, 三角形板側 = triangular side plate, 支撐架for滑軌 = support frame for slide rail, 載板for雙臂 = carrier plate for dual arms, 上版 = upper plate (the file name 軌 = rail).

## 3. Orientation: CAD +Y is UP (inferred, please confirm)

Evidence: both ReBeL arms are in their upright zero pose along +Y; the ZLW (1250 mm stroke) is a 1746 mm long axis along Y mounted on a triangular column that is wide (405 mm) at y=-1351 and narrow (95 mm) at y=-200, with a flat foot carrying 4 bolt holes at y=-1351. That is a lift column, with SolidWorks' Y-up convention. In the renders +Y is drawn up. I take **+Z as the forward/working direction** (the arms stand in front of the lift face z=739, the column foot extends back to z=278) and **+X lateral**. Then robot-left = +X, so **arm A (x=-424) is the left arm, arm B (x=-884) is the right arm**. If the customer's forward direction is different, left and right swap; nothing else changes. "Height above deck" below is `y + 1351` mm, with the deck assumed at the column's lowest face y=-1351.

## 4. Components (world frame, mm)

| id | name | solids | world bbox min -> max (x, y, z mm) | size mm | volume cm3 | what it is |
|---|---|---|---|---|---|---|
| NAUO1 (carriage) | 2026_06_12_05_56_45_0290_2 | 1 | (-738, -610, 695) -> (-566, -310, 739) | 172 x 300 x 44 | 675 | ZLW-20120S carriage (300 mm, lift carriage, travels in Y) |
| NAUO1 (body) | 2026_06_12_05_56_45_0322_2 | 1 | (-738, -1334, 675) -> (-566, 412, 738) | 172 x 1746 x 63 | 3203 | ZLW_20120S_K0BW0_D0A4B_0A0A0_1250: toothed-belt axis body, 1746 mm along Y, 1250 mm stroke, drive end at y=-1333 |
| NAUO2 | 支撐架for滑軌 (support frame for slide rail) | 1 | (-757, -632, 739) -> (-547, -284, 927) | 210 x 348 x 188 | 1107 | rail bracket: U/channel with two triangular gussets, bolted on the ZLW carriage; carries the X rail |
| NAUO3 | WW-16-120-20 | 1 | (-524, -280, 786) -> (-324, -247, 952) | 200 x 34 x 166 | 336 | WW-16-120-20 rail carriage (arm A) |
| NAUO4 | WHKD-1620 | 1 | (-524, -299, 918) -> (-324, -255, 1000) | 200 x 44 x 82 | 112 | WHKD-1620 clamp/lock (arm A) |
| NAUO5 | WS-16-120-660 | 1 | (-984, -282, 802) -> (-324, -260, 938) | 660 x 22 x 136 | 767 | WS-16-120-660 drylin W rail, 660 mm, along X |
| NAUO6 | WW-16-120-20 | 1 | (-984, -280, 786) -> (-784, -247, 952) | 200 x 34 x 166 | 336 | WW-16-120-20 rail carriage (arm B) |
| NAUO7 | WHKD-1620 | 1 | (-984, -299, 918) -> (-784, -255, 1000) | 200 x 44 x 82 | 112 | WHKD-1620 clamp/lock (arm B) |
| NAUO8 | 上版for... (upper plate) | 1 | (-549, -247, 764) -> (-299, -242, 964) | 250 x 5 x 200 | 174 | upper (mounting) plate on carriage A, arm base bolts to it |
| NAUO9 | 上版for... (upper plate) | 1 | (-1009, -247, 764) -> (-759, -242, 964) | 250 x 5 x 200 | 174 | upper (mounting) plate on carriage B |
| NAUO10 | D00391412.ipt | 150 | (-515, -242, 742) -> (-333, 681, 946) | 182 x 923 x 204 | 2890 | ReBeL arm A: 150 igus D00xxxxxx parts, straight pose along +Y |
| NAUO11 | D00391412.ipt | 150 | (-975, -242, 742) -> (-793, 681, 946) | 182 x 923 x 204 | 2890 | ReBeL arm B: identical copy |
| NAUO12 | 三角支撐 (triangular support) | 1 | (-702, -1351, 278) -> (-602, -200, 675) | 100 x 1150 x 397 | 21189 | gusset/column: 100 mm thick plate, y -1351..-200, slanted underside; the only fixed structure |
| NAUO13 | 三角形板側 (triangular side plate) | 1 | (-602, -1351, 605) -> (-453, -751, 675) | 149 x 600 x 70 | 3125 | side wing plate (70 mm thick) on the gusset top at the rear end |
| NAUO14 | 三角形板側 (triangular side plate) | 1 | (-851, -1351, 605) -> (-702, -751, 675) | 149 x 600 x 70 | 3125 | side wing plate, mirror |
| NAUO15 | 載板for雙臂 (carrier plate for dual arms) | 1 | (-323, -242, 722) -> (2, -239, 1022) | 325 x 3 x 300 | 278 | 3 mm plate 325x300 at the +X end of the rail (not attached to anything; probably stray/for later) |
| NAUO16 | 載板for雙臂 (carrier plate for dual arms) | 1 | (-1310, -242, 722) -> (-985, -239, 1022) | 325 x 3 x 300 | 278 | 3 mm plate at the -X end of the rail |

(`NAUO10`/`NAUO11` each contain 150 parts: base block D00431510, six joint housings, flange D00617758 etc.; see `data/leaves.csv`.)

## 5. Structure

* **Overall envelope** x -1310..2, y -1351..681, z 278..1022 (1312 x 2032 x 744). Without the two loose carrier plates: x -1009..-299 (710), y -1351..681, z 278..1000. Drawn pose, top of arms = 2032 mm above the deck.
* **Mounting plane to the AMR (presumed):** deck plane y=-1351, the lowest face of column and wings. Column foot face (100 x 377 mm) has **4 x d6.8 holes (d12 counterbore) on a 50 x 100 mm pattern** at x=-677/-627, z=466/566; the wings have d8.6/d10 holes. Nothing in the CAD defines the AMR top, so this is an assumption.
* **Column:** `NAUO12`, x -702..-602 (100 thick), y -1351..-200.5, z 278..675 (depth 405 at the foot, 95 at the top, three lightening cut-outs). Front face z=675 (100 x 1136 mm, 20 x d6.8 holes) carries the ZLW. Side faces x=-702/-602 are big free 1150 x 400 mm plate faces (plate, not slot profile; 6 holes each, d6.8/d40 cross bores).
* **Wings:** `NAUO13/14`, 149 x 600 x 70 mm each on the column front face, x -851..-453, y -1351..-751.
* **Linear axis (ZLW-20120S, 1250 stroke):** body x -738..-566, y -1333..412 (172 wide, 63 thick, drive end at -y), carriage 300 long, drawn at y -610..-310 = **mid-stroke** (body centre = carriage centre = y -460). Travel direction = Y (vertical). Carriage centre range y -1085..165, i.e. the whole carriage group moves **+/-625 mm** from the drawn pose, bracket bottom reaches y=-1256 = 95 mm above the deck at the lowest position. Stroke taken from the name; the end blocks leave exactly that stroke.
* **Carriage group (moves with the lift):** bracket `NAUO2` (x -757..-547, y -631..-283, z 739..927, 8 mm plates: vertical floor plate on the carriage, shelf at y=-283 under the rail, two triangular gussets, **open pocket 194 mm wide between the gussets**), X rail WS-16-120-660 (x -984..-324, 660 long, centre x=-654), two WW-16-120-20 carriages with WHKD-1620 clamps (A at the +X end, B at the -X end; each slides inward up to 260 mm, they cannot pass each other), two 250 x 200 x 5 mm upper plates on which the arms stand.
* **Arms:** A base centre (-424, -242, 855), B (-884, -242, 855), 460 mm apart; each 923 mm base-to-flange in the drawn upright pose (tip at y=681), J1 axis along Y, no gripper/tool/cables in the model.
* **Loose parts:** two 325 x 300 x 3 mm "carrier plates for dual arms" at the rail ends (x -323..2 and -1310..-985, y=-242) are not attached to anything. Treated as obstacles (conservative); they may be unfinished design.
* **Not in the CAD:** AMR, floor, controller, electronics, cable chain, T-slot profiles (no 40x40 / 45x90 found).

## 6. Free-space method (voxel and box sweep, not by eye)

* All 315 solids tessellated and rasterised on a **5 mm voxel grid** (conservative surface raster + cavity fill; volumes come out 5-15 % high, so clearances are safe-side).
* Box 200 x 110 x 100 mm in **all 6 axis-aligned orientations**, **>= 10 mm clearance** (box inflated by 10 mm, 3-D summed-area table over the grid). Yaw/tilted placements were not searched.
* **Motion sweeps:** ZLW carriage group +/-625 mm in Y; arm A slides -260..0 and arm B 0..+260 in X on the rail (relative sweeps respect the 200 mm carriage separation). Four mounting frames: `FIXED` (rigid with column/wings), `DECK` (resting on an assumed deck at y=-1351), `CARRIAGE` (rigid with the lift carriage: must clear the column over the stroke and the arms sliding in X), `ARM_A` (rigid with arm A's X carriage: moves in X and Y).
* "Supported" = a mount solid within 40 mm of a box face (standoff/bracket); mount solids: column+wings (FIXED), the deck plane (DECK), bracket+carriage (CARRIAGE), carriage/clamp/plate of arm A (ARM_A).
* **Verification:** every listed finalist was re-checked with ~80 000 mesh surface sample points per part group (4 mm spacing) over the full sweep (Y step 12.5 mm, X step 20 mm): the exact clearance column below. This is independent of the voxel grid.
* **Tube estimate** = worst-case L1 (routing) distance from box centre to the left arm's base centre over the family's motion + **960 mm along the arm** (923 mm straight, plus routing) + **150 mm slack**. The 0.96 m arm run is the same for every candidate; only the first term differs. These are estimates, no routing was designed.
* **Arm reach flag:** box inside a 700 mm sphere above the arm base plane for any base position (arms can reach it, the box would have to be a keep-out).
* **Bump flag:** box outside x -984..-324 and z 278..1000 (the stand's own envelope). The AMR footprint is unknown, so "sticks out of the stand envelope" is the proxy; every top candidate is inside it.

## 7. Candidates, ranked (target arm A = presumed left; mirror x -> -1308.2 - x for arm B)

Box sizes are x, y(vertical), z(forward) in mm. Heights are above the assumed deck. Tube = box to arm A base leg + 0.96 + 0.15 m.

| # | where | frame | box x,y,z / centre | height | leg | tube est. | exact clearance | on moving part? | mount surface | travel collision | in arm reach | sticks out of envelope |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **P1 bracket pocket, box upright** (best) | lift carriage | 110x200x100 at (-635,-505,855) | 746..946 (moves +/-625) | 0.47 m | **1.58 m** | 24.9 mm (>=40 mm uniform feasible in the pocket) | yes, with the arms (not with the arm's X slide) | 8 mm bracket floor plate (5 holes) or gusset plates; no slot profile | none over the stroke and X slide | no (below the arm base plane) | no |
| 2 | **P2 same pocket, upper part under the rail shelf, box lying forward** | lift carriage | 110x100x200 at (-630,-355,860) | 946..1046 | 0.32 m | **1.43 m** | 12.8 mm | yes | same plates, shelf underside | none | no | no (front edge z 960, 33 mm past the shelf) |
| 3 | Beside the bracket on the carriage, outboard of A | lift carriage | 110x100x200 at (-480,-345,855) | 956..1056 | 0.31 m | 1.42 m | 12.1 mm | yes | needs an L-bracket off the bracket wall/carriage | none | no | no |
| 4 | Below the rail on arm A's X carriage | ARM_A frame | 200x110x100 at (-425,-375,990) | 921..1031 | 0.27 m | 1.38 m | 13.0 mm | yes, moves with arm A in X and Y | clamp/carriage only, cantilevered 100 mm | none | no | 40 mm past z=1000 |
| 5 | Column side face, under the lift (fixed) | FIXED | 110x200x100 at (-530,-390,630) | 861..1061 | worst 1.15 m | 2.26 m worst, 1.59 m at drawn pose | 14.7 mm | no (fixed) | column plate face (x=-602) | none, but the carriage sweeps right past it | **yes** (arms can reach it) | no |
| 6 | Column wing front face, base (fixed) | FIXED | 100x110x200 at (-485,-975,815) | 321..431 | 1.60 m | 2.71 m worst, 1.94 m at drawn pose | 12.1 mm | no | 70 mm wing plate, front face | none | no | no |
| 7 | AMR deck beside the column foot (assumed deck) | DECK | 200x110x100 at (-555,-1270,860) | 26..136 | 1.79 m | 2.90 m worst, 2.27 m at drawn pose | 14.9 mm | no | AMR deck (unknown) | none | no | no |

Notes on the ranking:
1. **Items 1-3 are the same idea: put the pump box on the lift carriage.** Item 1 has by far the largest margin (the whole pocket allows a 110 x 200 x 100 mm box with 40 mm clearance: box centre x -690..-610, y -530..-395 for the upright box, any depth). Item 2 gives the shortest tube. The 200 mm orientations along X do not fit the 194 mm pocket.
2. The carriage payload (ZLW-20120S rating at this lever arm) and heat of two Thomas pumps in a semi-closed pocket were **not** checked. The pocket is open at the front and bottom, so ventilation is plausible, but that is not analysed.
3. Items 5-7 need a drag chain or a coiled tube absorbing the 1250 mm stroke; with the 0.96 m arm run, any fixed box gives a tube of at least about 2.1 m worst case. Item 5 is inside the arm workspace.
4. Placements on the arm X carriage (item 4) have the shortest leg but nothing solid to mount to, hang near the arm and fall in the 700 mm reach sphere or past the envelope; not recommended.
5. The full ranked lists (10 strict + 10 relaxed per family, with exact clearances and nearby solids) are in `data/candidates_targetA.json`.

## 8. Renders (`renders/`, 2000 x 1300 px, light background, colour by group, axes triad and scale bar)

Standard views: `iso_front_left.png`, `iso_back_right.png`, `front.png`, `side.png`, `top.png`.
Close-ups at each arm base: `closeup_base_A_front_iso.png`, `closeup_base_A_back_iso.png`, `closeup_base_B_front_iso.png`, `closeup_base_B_back_iso.png`.
Candidate overlays (parts washed out, boxes coloured, ids P1-P6): `candidates_iso_front_left.png`, `candidates_iso_back_right.png`, `candidates_side.png`, `candidates_front.png`, `candidates_top.png`; per candidate `closeup_P<n>_{iso,iso_back,side}.png` (translucent copies show the box at the +/-625 mm lift ends).
Id key: P1 pocket upright, P2 pocket under shelf, P3 arm-A carriage, P4 column side, P5 wing front face, P6 deck.

## 9. Uncertain or not determined

* **Up/forward/left/right** are inferred (section 3). The deck plane (y=-1351) and the 4 foot holes as AMR interface are assumptions.
* The 1250 mm stroke comes from the product name; the CAD only shows the carriage at mid-stroke.
* Arm pose is the CAD's upright pose; the arms' real workspace is not modelled, only the 700 mm reach-sphere flag. The tool and any gripper are absent. The 0.96 m arm run and 0.15 m slack are estimates.
* The ZLW carriage's real position limits (end-block stop) are taken as symmetric +/-625 mm.
* Loose carrier plates, WHKD clamp function, bracket plate thickness (8 mm read from geometry) not confirmed. No standard igus article was matched to the `D00xxxxxx` parts.
* Only axis-aligned orientations; no yaw/tilt, no cable/hose bend radius, no service-access check, no payload or thermal check.

## 10. Reproduce

`PY="/media/farhan/My Passport/igus_nvidia/venvs/cad/bin/python"`; run with `nice -n 19`. Scripts (in order): `01_load_tree.py <step> <leaves.json> <brep_dir>`, `02_tessellate.py <work_dir>`, `05_voxelize.py`, `06_freespace.py`, `TARGET=A 07_candidates.py` (about 4 min), `make_picks.py`, `08_renders.py renders`, `09_candidate_renders.py renders`, `10_components.py`, `11_regions.py`, `12_summary.py`. `03_overview.py` and `04_faces.py` are exploratory helpers. Work dir is `/media/farhan/My Passport/igus_nvidia/shelf_pick_research/robot_stand/` (`common.py` has the path).
