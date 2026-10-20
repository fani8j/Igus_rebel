# vacuum_module - electric vacuum source enclosure on the AMR (left-arm suction tool)

Parametric, printable CAD for the box that stands on the AMR next to the robot and feeds the left arm's 3 x SMC ZP2 suction tool through **one tube** (operator decision 2026-10-13: no compressed air, 24 V battery, quiet, filtered exhaust, hospital hallway).
It holds a KNF N 838 KN.29DC-B pump on 4 rubber mounts, the SMC ZSE20B vacuum switch, the 24 V vent valve with its own intake filter, the dust filter, a manifold, a muffler and a HEPA exhaust filter, behind a 10 mm foam-lined PETG shell with a baffled cooling labyrinth.
Frame: module frame, **+x = service door**, y = width, z up, **z = 0 = underside of the base plate** (the 4 AMR spacers reach z = -12). Shared inputs come read-only from `../common/params.py` (`VACUUM_PUMP`, `VACUUM_PARTS`, `VACUUM_PUMP_ALTERNATIVES`); everything specific is in `params.py`. No ROS node and no robot was used.

Rebuild: `./run_all.sh` (about 3 minutes: `build.py` -> `verify.py` -> `render.py`; shared cadquery venv, `nice -n 19`, single process).

## Key numbers
| item | value |
|---|---|
| outer size (housing + door + base plate) | **240 x 186 mm footprint, 220 mm tall** (264 mm to the top of the arm-tube stub, +12 mm spacers under the plate). Brief target was about 220 x 170: **does not fit** (area x1.19), see "Mismatches" |
| mass (estimate) | **4.5 kg**: pump 1.9, printed PETG 1.37 (4 parts), foam + film 0.13, other hardware/fittings/cables about 1.1; CoG (-12, 1, 100) mm in the module frame |
| cabin (inside the film) | 213.4 x 159.4 x 200.7 mm (x, y, z) |
| wall stack | 3 mm printed shell + **10 mm foam** + 0.3 mm PET film holding it; door has the same stack (13.3 mm deep tray) |
| AMR mount | 4 x **M6 slots** in the base plate, pattern **180 x 100 mm** (`AMR_SLOT_PITCH`), slot 10.6 x 6.6 (+-2 mm), 12 mm spacers (exhaust nozzle exits downwards into that gap), M6 plate washers 26 x 14 x 2 close the slots acoustically. Bolts are reached from inside through the open door (heads sit on the cabin floor) |
| arm tube | bulkhead push-in fitting for the d8 tube on the roof at (88, -10) |
| electrical | one M12 panel connector on the rear wall (-x) at y = 58, z = 90 |
| exhaust | HEPA -> bulkhead nozzle through the base plate at (58, 20), pointing **down** into the 12 mm gap above the AMR deck (no jet toward people) |
| cooling | baffled labyrinth: intake slots on the -y wall (low, near the door) -> 160 mm duct -> fan at the pump side; exhaust slots on the +y wall (high, near the door) fed from the same pump-side end; 24 V 40 x 10 mm fan, see thermal |
| service | one-sided: door at +x held by **4 captive M4 screws** (M4 heat-set inserts in the housing corner blocks); dust filter, HEPA, valve, switch, manifold, muffler and terminal block are reachable without moving the pump |

## Print plan (PETG, 250 x 250 bed, 4 parts, no aluminium needed)
| part | print orientation | extents (mm) | notes |
|---|---|---|---|
| `base_plate` | flat, legs up | 240 x 186 x 143 | plate 6 mm, 4 x M6 slots, 4 pump-mount bosses (M6 inserts), HEPA cradle, 4 standing deck legs 12 x 10 x 137 (swap for 12 mm aluminium square tube if they wobble), 10 countersunk M4 holes for the housing, exhaust hole |
| `housing` | roof down, open side up | 226.7 x 186 x 214 | roof + 3 walls, bottom ledge with 10 M4 heat-set inserts, 4 corner blocks (M4 inserts for the door), roof and M12 foam sleeves, 2 vent ducts (partitions 3 mm, slots 3 x 32 mm bridged by design) |
| `door` | outer plate down, rim up | 214 x 186 x 13.3 | tray for the 10 mm foam, 4 corner towers with d4.5 holes |
| `deck` | flat, pads up | 116 x 148 x 16 | filter/valve deck at z = 146 above the pump motor, 4 M4 screws into the leg inserts |

Not printed: foam (5 pieces, die-cut 10 mm, adhesive back, hospital-grade), PET film (5 pieces, 0.3 mm, bonded over the inner foam faces so nothing can shed), rubber mounts, all purchased parts. STL files in `stl/` are already in print orientation. Fasteners: M4 heat-set inserts (hole 5.6) for the base screws, door and deck; M6 inserts (hole 8.0) for the 4 pump mounts; M3 self-tapping for the fan (4 pilot holes in the duct wall).

## BOM
| # | item | qty | part number / model | note |
|---|---|---|---|---|
| 1 | vacuum pump | 1 | **KNF N 838 KN.29DC-B (MI)** | 24 V, 55 W max, 1.9 kg, 0.1-5 V / PWM speed, IP20; alternatives in `common/params.py` |
| 2 | anti-vibration mounts | 4 | to choose (d20 x 15, M6 stud + M6 female) | |
| 3 | vacuum switch | 1 | **SMC ZSE20B(F)** PNP (variant to confirm) | |
| 4 | vent valve 2/2 N.C. 24 V | 1 | to choose (about 22 x 22 x 50) | |
| 5 | vent valve intake filter | 1 | to choose (d12 x 12) | |
| 6 | inline dust filter | 1 | SMC ZFC series - to confirm | |
| 7 | HEPA exhaust filter | 1 | Whatman HEPA-VENT 50 mm - catalogue number to confirm | 99.97 % at 0.3 um |
| 8 | exhaust muffler | 1 | to choose (G1/8, d20 x 40) | |
| 9 | 4-port manifold (arm line, switch, valve, filter) | 1 | to choose | |
| 10 | G1/8 male -> d8 push-in elbow | 2 | to choose (SMC KQ2L08-01 style) | pump ports |
| 11 | bulkhead push-in union d8 (roof) | 1 | to choose | arm tube |
| 12 | bulkhead push-in union d8 (base) | 1 | to choose | exhaust nozzle |
| 13 | PU tube d8/d6 | about 0.5 m | to choose | internal; d6 for the arm tube is possible, the fitting then changes |
| 14 | M12 panel connector (A-coded, M16 rear mount) | 1 | to choose | pin count and current to choose (see wiring) |
| 15 | cooling fan 24 V 40 x 40 x 10 | 1 | to choose (>= 7 m3/h, <= 22 dB(A)) | |
| 16 | terminal strip | 1 | to choose | |
| 17 | M4 captive panel screws (M4 x 20) | 4 | to choose | door |
| 18 | M4 heat-set inserts / M4 countersunk screws / pan screws | 18 / 10 / 4+ | to choose | |
| 19 | M6 heat-set inserts + M6 screws, plate washers, spacers d18 x 12 | 4 + 4 + 4 + 4 | to choose | AMR slots |
| 20 | acoustic foam 10 mm, PU, adhesive back | about 0.1 m2 | to choose (fire class B1/V0) | |
| 21 | PET film 0.3 mm, self-adhesive | about 0.12 m2 | to choose | |

## Pneumatic schematic
```
 arm tube d8 (up, roof bulkhead R)
        |
   [4-port manifold M1] --(d8)-- [vent valve 2/2 N.C. 24 V] --- [intake filter] --- room air
        |        \
        |         `--(d8)-- [SMC ZSE20B vacuum switch, PNP out]
        |
     (d8)
        |
   [inline dust filter] --(d8)-- pump INLET --[ KNF N 838 ]-- pump OUTLET --(d8)-- [muffler]
                                                                                       |
                                                                                    (d8, 1 bend r25)
                                                                                       |
                                                           room air <-- bulkhead nozzle (base, down) <-- [HEPA-VENT]
```
Pump runs during the pick only (about 8 s), vent valve opens to release the carton. Pump inlet/outlet assignment (inlet at y = -20) is **not readable** in the datasheet drawing; check the KNF CAD before cutting plastic.

## Wiring (M12 connector on the rear wall; pin numbers to assign)
| signal | nominal | goes to | robot / AMR IO |
|---|---|---|---|
| 24 V | 24 V DC, pump 2.3 A max + valve + fan + switch (about 3 A) | terminal strip -> pump, valve, fan, switch | AMR 24 V aux output (fused 4 A) |
| GND | 0 V | same | AMR 0 V, common with the robot IO ground |
| pump on | 24 V on = run | pump enable (via a relay or the pump's control input) | DO on **either** ReBeL controller |
| pump speed | 0.1-5 V analog or PWM | pump control input | fixed: trimmer or resistor divider in the module (ReBeL analog out not confirmed); optional 2nd speed = 1 more DO switching a second divider |
| vent valve | 24 V on = release | valve coil | DO on **either** ReBeL controller |
| switch output | PNP, vacuum OK | ZSE20B OUT (switch supplied 12-24 V from the 24 V rail) | DI on the **same** controller as the two DOs (so one robot program owns the pick) |
| fan | 24 V (with the pump or thermostat) | fan | runs with pump-on (no IO) |

Operator decision 2026-10-14: the IO can come from either robot. Put all three signals (pump on, vent valve, vacuum OK) on one controller. The ROS driver sets outputs through `set_digital_output` (`igus_rebel_msgs/SetDigitalOutput`) and parses DIN from the CRI STATUS. Check whether DIN is published as a topic before using the vacuum-OK signal.
Pump motor cable is a 300 mm pigtail out of the motor end (drawing). A single M12 A-coded pin carries about 4 A: use two pins in parallel for 24 V and GND or a separate power connector (open question).

## Verification (`data/verify.json`)
* **Clearances >= 3 mm**: pass, 337 pairs checked with exact solid distance, smallest gap 3.7 mm (M6 plate washers to the housing/door); everything touching is a declared mount (see `allowed_contact()` in `verify.py`).
* **Foam**: 5 pieces, thickness 10.00 mm each, film area covers 100 % of every inner foam face.
* **Tubes**: all internal tubes are d8, minimum bend radius 25 mm; only the exhaust tube to the HEPA bends (r = 25, 7 mm of margin), the other tubes are straight.
* **Print fit**: all four parts fit 250 x 250 (largest 240 x 186 and 226.7 x 186 x 214), 4 printed parts.
* **No line of sight**: ray check, 240 000 rays (from the pump surface, from the two duct openings, from random cabin points), **0 escape** through slots/ducts/penetrations; positive control with the door removed: 8974 of 90 000 rays escape (the test detects open paths). Foam/film/deck/fan are treated as transparent (conservative).
* **Thermal** (single lumped node, all 48 W treated as heat, duty 8 s per 30 s = 12.8 W mean, ambient 25 C, pump limit +40 C ambient): wall UA = 0.49 W/K (foam is a good insulator). **No fan: about +26 K steady (+24 K after 4 h of picks) = 49 C, above the pump limit -> a fan is needed.** With one 24 V 40 x 10 mm fan moving about 3 m3/h through the labyrinth (UA 1.5 W/K): about +9 K (34 C). Continuous 48 W would still be +32 K even with the fan: the pump must stay duty-limited. Measure the real temperature in the bench test (pump head runs hotter than the cabin air).

## ASSUMED values (replace with datasheet / CAD / measurement)
Pump drawing readings (datasheet p.3 rendered at 400 dpi): foot hole pitch 100 (width) x 50 (length) and 134 x 115 x 129 envelope are from the datasheet; the **absolute foot position along the length (63 / 113 mm from the head end), foot plate 56.5-119.5 x 5 mm, head block 79 x 83 x 99 + cap, motor d56 at z 45, bridge block 60-115 x 60 x 12 mm, port x = 71 mm (+-4), port y = +-20 mm and which port is the inlet** are read off a small drawing (+-2..4 mm) or invented. Also assumed: mount 15 x d20, elbow envelope d14 rising 12 mm, switch 25 x 30 x 30 with its port on the -y face, valve 22 x 22 x 50 with its port 17 mm above the bottom, valve intake filter d12 x 12, dust filter d25 x 60, HEPA d56 x 30 with ports on its rim, muffler d20 x 40, manifold 16 x 28 x 16, fan 40 x 40 x 10, terminal block 50 x 14 x 12, M12 flange 20 x 20 x 3 with d16 body, M6 plate washer 26 x 14 x 2, all small masses (switch 100 g, valve 120 g, dust filter 80 g, HEPA 50 g, muffler 30 g, manifold 60 g, terminal 120 g, fan 25 g, M12 40 g, fittings/screws/cables 250 g), foam density 30 kg/m3, film 0.3 mm, 250 mm printer height, ambient 25 C, duty 8 s per 30 s, 48 W typical = all heat, surface coefficients 5 / 7.5 W/m2K, thermal capacity 3 kJ/K, fan flow through the ducts 3 m3/h. Full list with reasons: `data/verify.json` -> `assumed`.

## Mismatches against the brief
* Footprint is 240 x 186 mm, not 220 x 170: the pump (134 x 115) plus the 13.3 mm wall stack on every side plus a deck level for the filters, the strip for the HEPA (d56) and the 25 mm tube bends did not fit smaller. The bed limit (250) is still respected. The next shrink would be a taller deck/second level, or an external HEPA.
* The exhaust outlet points down into a 12 mm gap, so the AMR deck under the module must be free and flat there.
* Foam is held by film (bonded) plus printed duct walls and sleeves, not by a printed inner liner (keeps the part count at 4).
* Tubes are modelled as solid rods; the 6-12 mm tube stubs between neighbouring fittings are short but modelled as touching their ports.

## Open questions
1. Real AMR top plate: hole pattern, free height, can the exhaust nozzle blow into a 12 mm gap? Is the 4 x M6 slot pattern 180 x 100 right?
2. Pump inlet/outlet positions and a KNF CAD file (the drawing does not show which G1/8 port is which).
3. Connector choice: one M12 for 24 V + signals (3 A) or a separate power connector; pin assignment; which robot or AMR IO supplies the speed signal and the valve.
4. Fan: model, noise at 24 V, and whether it runs with the pump or on a thermostat. Measured noise of the pump (KNF does not publish it) with this enclosure.
5. Foam grade and hospital cleaning agents the PETG shell must survive; is a silicone/PU paint needed.
6. Does the switch need to sit behind the dust filter (clean side) or on the arm-side manifold as modelled?
7. Leak-down/response at the pads: the arm tube is d8 here, the tool side collector is d4/d6 - tube sizing to confirm in the bench test (`../BENCH_TEST_PLAN.md`).

## Files
`params.py` (all inputs, ASSUMED list), `parts.py` (geometry), `build.py` (STEP/STL/meshes/mass), `verify.py` (checks -> `data/verify.json`), `render.py` (PNG), `run_all.sh`.
Outputs: `step/assembly.step`, `step/parts/*.step` (45 solids), `stl/*.stl` (4 printed parts, print orientation), `meshes/visual/*.stl` (metres, module frame, + `meshes/manifest.json`, for a later sim model), `renders/{iso,exploded,section,service_open}.png`, `data/{verify,mass,part_summary}.json`.
