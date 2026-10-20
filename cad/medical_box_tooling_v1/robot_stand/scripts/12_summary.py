"""Assemble data/stand_summary.json from the analysis outputs (components, candidates, regions) + measured structure facts."""
import json, numpy as np
comp = json.load(open('data/components.json')); cand = json.load(open('data/candidates_targetA.json')); reg = json.load(open('data/regions.json')); picks = json.load(open('data/picks.json'))
S = dict(
 source=dict(file="/home/farhan/Downloads/2 arms step(軌).step", size_bytes=106970424, writer="SolidWorks 2017, STEP AP214 (SwSTEP 2.0)", length_unit="mm (SI_UNIT .MILLI. .METRE.)",
   assembly="1 root, 16 top-level instances (NAUO1..16; NAUO1 holds 2 solids), 315 leaf solid instances, 65 distinct solids; no sub-assembly nesting beyond one level for the arms (150 parts each)",
   load="STEPCAFControl/XCAF: read 4 s, transfer 43 s, full tree walk + bbox/volume/BREP dump ~7 min at nice 19, single process; RAM never an issue (<10 GB machine-wide)"),
 axes=dict(up="+Y (inferred: arms are in the upright ReBeL pose, ZLW is a 1250 mm lift, column is triangular with a flat foot at y=-1351)", forward="+Z (arms stand in front of the lift face z=739; column foot extends rearwards to z=278)",
   lateral="X; with forward=+Z and up=+Y, robot-left = +X, so arm A (x=-424) is the LEFT arm and arm B (x=-884) the right arm. NOT confirmed by the customer.",
   deck_plane="y=-1351 (lowest face of column and wings). The CAD contains no AMR, no floor plate; height above deck h = y + 1351 mm"),
 envelope_mm=dict(all=dict(x=[-1310,2], y=[-1351,681], z=[278,1022]), without_carrier_plates=dict(x=[-1009,-299], y=[-1351,681], z=[278,1000]), height_above_deck_drawn_pose=[0,2032]),
 structure=dict(
  column_gusset=dict(x=[-702.1,-602.1], y=[-1351.0,-200.5], z=[278.3,675.2], note="100 mm thick triangular plate, 1151 tall, depth 405 at the foot tapering to 95 at the top, 3 lightening cut-outs; front face z=675 carries the ZLW (20 x M6.8 holes). Foot face y=-1351 has 4 x d6.8 (counterbore d12) holes on a 50 x 100 mm pattern (x=-677/-627, z=466/566): presumed AMR interface"),
  side_wings=dict(x=[-851.0,-453.2], y=[-1351.0,-751.0], z=[605.2,675.2], note="two 149 x 600 x 70 mm triangular rib plates on the column front face, bottom at the deck plane"),
  zlw_lift=dict(body=dict(x=[-738.1,-566.1], y=[-1333.5,412.5], z=[675.2,738.2]), carriage_drawn=dict(x=[-738.1,-566.1], y=[-610.5,-310.5], z=[694.7,739.2]),
     stroke_mm=1250, travel_dir="Y (vertical)", carriage_centre_range_y=[-1085.5,164.5], drawn_pose="carriage is at mid-stroke (body centre y=-460.5 = carriage centre)", assumption="stroke 1250 mm taken from the product name; end-block geometry (carriage y -1235..315 at the ends) agrees"),
  rail_bracket=dict(x=[-757.1,-547.1], y=[-631.5,-283.5], z=[739.2,927.0], note="8 mm plate bracket: vertical floor plate on the carriage (z 739-747, 5 holes), shelf at y=-283 carrying the X rail (10 holes), two triangular side gussets with an open pocket between them (inner width 194 mm)"),
  x_rail=dict(name="WS-16-120-660", x=[-983.6,-323.6], y=[-281.7,-259.7], z=[801.5,937.5], centre_x=-653.6, carriages="2 x WW-16-120-20 (200 long) + WHKD-1620 clamps; A at the +X end, B at the -X end; each can slide inwards 260 mm (combined free length 660-2x200)"),
  upper_plates="2 x 250x200x5 mm plates at y=-246.7..-241.7 on the carriages = arm base mounting",
  carrier_plates="2 x 325x300x3 mm plates at y=-241.7..-238.7 at x=-323..2 and -1310..-985: not attached to anything, treated as obstacles",
  arms=dict(A=dict(x=[-515.1,-332.7], y=[-241.7,681.0], z=[741.6,945.7], base_centre=[-424,-242,855], flange_y=681.0), B=dict(x=[-975.0,-792.6], y=[-241.7,681.0], z=[741.6,945.7], base_centre=[-884,-242,855], flange_y=681.0),
     pose="upright (all joints near zero), J1 axis along Y, base flange bolted to the upper plate; 923 mm base-to-flange; centre spacing 460 mm; tube path along the arm assumed 960 mm", note="no gripper/tool, no cabling, no controller in the CAD"),
  electronics="none in the CAD", t_slot_profiles="none; the structure is plate-built, the only extrusions are the ZLW body (172 x 63 mm) and WS rail"),
 components=comp,
 search=dict(box_mm=[200,110,100], orientations="6 axis-aligned permutations (no yaw/tilt)", clearance_mm=10, voxel_mm=5, voxel_note="conservative surface raster + cavity fill (volumes overestimate by 5-15 %); finalists re-checked against ~80k mesh sample points per group (4 mm spacing) over the full sweep",
   sweep=dict(lift_stroke_mm=[-625,625], arm_x_slide_mm=260, families={"FIXED":"rigid with column/wings, clears all moving parts","DECK":"as FIXED but resting on an assumed deck at y=-1351","CARRIAGE":"rigid with the ZLW carriage (moves in Y), clears the column over the stroke and the arms sliding in X","ARM_A":"rigid with arm A's X carriage (moves in X and Y), clears everything else"}),
   tube_model="tube_total = max L1 distance(box centre -> arm base centre over the family's motion) + 960 mm (arm) + 150 mm slack; arm hemisphere test R=700 mm above the base plane",
   target_arm="A (left if robot faces +Z)"),
 candidates=cand['results'], picks=picks, regions=reg)
json.dump(S, open('data/stand_summary.json','w'), indent=1, ensure_ascii=False)
print('ok', len(json.dumps(S))//1024, 'KB')
