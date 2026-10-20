# Bench-test plan: medical carton pick tooling

These tests check the assumptions the CAD cannot settle. Run them on real cartons **before** printing the final tools or picking with the robot.
Tests 0–11 need no robot arm: everything is done by hand, with the LEHF powered from its own driver on the bench and the vacuum from the electric pump module (`vacuum_module/`, KNF N 838 KN.29DC-B on 24 V). No compressed air is used: the robot runs on an AMR in the hallway to the operating room.
**Run test 0 first: it decides which pump to buy.**
Any test that moves a ReBeL needs explicit operator approval at that time.

Record the results in `bench_results/<date>.md` (create it), one row per trial, and update the design values listed under "Feeds".

## Equipment
- **Cartons:** cartons 1–5 (at least two of each if possible), plus a packed group of ≥ 10 cartons on a shelf board with the real cabinet's spine setback (10 mm).
- **Gauges:** a 0–50 N push/pull force gauge or spring scale, and a kitchen scale (0.1 g).
- **Gripper:** LEHF20K2-48 with its driver, the printed fingers and pads (TPU 95A ribbed), plus one pair of smooth 30–40 Shore A silicone pad samples for comparison.
- **Vacuum:** 3 × ZP2-T4030WNK10-04 in the printed carrier, the collector and a 2 m ⌀6 trunk (the real AMR-to-tool length), the vacuum module (pump, dust filter, ZSE20B switch, vent valve, HEPA exhaust filter) on a 24 V bench supply with a current display, a vacuum gauge at the pads, and a 0–20 L/min flowmeter (rotameter or thermal mass flow) for test 0.
- **Noise:** a sound level meter (A-weighted) for test 11.
- **Scanner:** SE4710 + PL3307 on the printed scanner arm, plus a ruler or jig for the scan distance.
- **Cleaning:** the site-approved disinfectant wipes.

## Tests

| # | Test | Assumption it checks | Procedure | Pass | Feeds |
|---|---|---|---|---|---|
| 0 | **Pad leakage on carton (decides the pump)** | Design leak 1 L/min per sealed pad, 4 L/min on rough carton | Hold one ZP2 pad on a carton spine, connect it through the flowmeter to any vacuum source and set −50 kPa at the pad. Read the flow for each carton type: smooth/coated face, printed spine, across the spine fold, and on a tuck flap. 5 cartons each. | — (measurement) | Pump choice: > 2 L/min per pad → KNF N 838; ≤ 1 L/min per pad → Schmalz ECBPi is enough. Also the leak values in `suction_tool/params.py` |
| 1 | **Carton mass** | 105–120 g estimated, 150 g design value | Weigh 2 of each carton type, sealed if available. If only empty cartons exist, weigh them and note it. | max ≤ 150 g | `common/params.py BOX_MASS_MAX`, `sim_assets/medical_boxes/boxes.yaml` |
| 2 | **Pad friction μ** | μ = 0.5 on the carton face | Clamp a carton face between the two pads at a known normal force N (use the gripper at 40 / 70 / 100 % force). Pull the carton out of the grip along its length with the force gauge and record the peak F. Then μ = F / (2N). Do it for the TPU ribs and for the silicone samples, dry and again after 20 disinfectant wipes. | μ ≥ 0.5 (TPU) | Hold margins in `gripper_tool` verify; TPU or silicone decision |
| 3 | **Grip hold, hanging** | Hold margin 1.54× (box 4, worst force) with the top stop | Grip carton 4 at s = 15–70 mm, 25 mm below the top edge, 100 % force. Hang it level for 60 s. Then shake it by hand (about 2 g) in each direction and tilt it 45° to the hang position. | No slip > 1 mm; no dent > 0.5 mm after release | Force setting; rotate-to-hang procedure |
| 4 | **Two-stage close capture** | ±2 mm centring under the stop | Pre-close to t + 4 mm above the top edge, offset the carton laterally by 0, ±2, ±4, ±6 mm, lower the gripper by hand (or on a jig) and clamp. Note whether the bar ever lands on the edge. | Centres up to the capture range in the gripper README; bar never lands on the edge | Registration procedure |
| 5 | **Suction hold on the spine** | 3 pads hold the pull; the spine flap doesn't peel | At −60 kPa, pull the spine straight out with the force gauge: one pad, then all three. Record the detach force and whether the tuck flap or tamper seal lifts. Repeat with the pads 3 mm off-centre across the spine. | Detach ≥ 2 × the extraction force (test 6); no flap or seal damage | Pad count, threshold, pull speed |
| 6 | **Extraction force from a packed shelf** | Partial pull of 100 mm is possible; neighbours stay put | In a packed group (neighbours leaning up to 10°), pull one carton out by 100 mm with the force gauge on a pad. Record the peak force and whether neighbours fall into the gap afterwards. | Peak ≤ 50 % of test 5 | Pull speed; "neighbour lean after pull" camera check |
| 7 | **Vacuum-OK threshold** | The gate (at the switch, pump side) separates 3 sealed pads from 1 lost pad, with the 2 m trunk | With the full vacuum module and the 2 m trunk: seal all 3 pads and record the vacuum at the switch and at the pads, the time to reach the gate and the pump current. Lift one pad off and record again. Repeat on a rough carton. | 3 pads stronger than the gate + 5 kPa; 1 pad off weaker than the gate − 5 kPa; gate reached ≤ 0.5 s | Switch setting (gate value from `suction_tool/data/verify.json` → `vacuum_gate`) |
| 8 | **Barcode read** | SE4710 reads the spine codes at 115–150 mm | Read each carton's GS1 DataMatrix and 1D code at 100, 115 and 150 mm, at 0° and 10° lean, under the cabinet lighting, 20 tries each. | ≥ 19/20 at the chosen distance | Scan distance and scanner tilt |
| 9 | **Pad orientation drift** | Nut + Loctite in the PETG pocket keeps the oval aligned | Torque the nuts to 2.5 N·m, press and pull 200 times on a carton, then measure the pad angle. | ≤ 3° rotation | Whether the pad bracket must be aluminium |
| 10 | **Cleaning** | Printed PETG/PA12 and TPU survive site cleaning | Wipe all printed parts and pads 20 times with the site disinfectant. Check for surface damage, insert pull-out and changed pad friction (repeat test 2). | No cracks, no loose inserts, μ drop < 20 % | Materials; MJF PA12 or sealed pockets |
| 11 | **Noise and heat of the vacuum module** | The closed module is quiet enough for the OR hallway and stays cool | Run the pump at full speed and at the holding speed inside the closed module: sound level (dB(A)) at 1 m on 4 sides, and the air temperature inside after 30 min of 8 s on / 22 s off. Then 30 picks with the HEPA filter fitted: check the exhaust back-pressure. | Agree the dB(A) limit with the hospital (record it here); inside temperature ≤ 40 °C (pump limit); outlet back-pressure < 0.2 bar | Pump speed setting, fan yes/no, foam |

## After the bench tests
1. Update the design values (leakage, mass, μ, threshold, scan distance) and re-run `gripper_tool/verify.py`, `suction_tool/verify.py` and `integration/handoff_check.py`.
2. Any failure in tests 2, 3 or 5 changes the design, so tell the design owner before printing again:
   - **Test 2 or 3:** use silicone pads.
   - **Test 5:** reconsider the pad count, add ZP2V check valves, or pull more slowly.
   - **Test 0 or 7:** too much leakage for the KNF N 838 → second pump in parallel (2 × Thomas 1520VDP is the listed alternative), or larger/softer pads.
   - **Test 11:** too loud → lower the holding speed, thicker foam, or move the module lower on the AMR.
3. Only then test on the robot, in mock first (`ROS_DOMAIN_ID` unused, `ROS_LOCALHOST_ONLY=1`), then with operator approval on hardware.
