# suction_tool v4 CONCEPT - "pad blade + one sensor pod"

Slimmer replacement idea for the v3 left-arm suction/scan tool (operator: v3 is "chunky"). **Concept level**: no URDF/USD, no collision meshes. Nothing in `../suction_tool` or `../common` was modified; v3 helpers (pad/nut/tube/Gemini loading, mass, render style) are imported by path in `bridge.py`.
Frame link6, z out of the flange, flange face z = 52.7, 6 x M3 on PCD 43 (angles 45 + 60k). Hand-off contract kept: LIP_Z = 159 (compressed 154), pads at x = -60 / 0 / +60, bar front face z = 133.5, nothing but pads + pad bar above z = 119.

## Rebuild
```
PY="/media/farhan/My Passport/igus_nvidia/venvs/cad/bin/python"
nice -n 19 $PY build.py     # step/, stl/, data/mass.json, data/printed_parts.json   (~10 s)
nice -n 19 $PY verify.py    # data/verify.json (~15 s)
nice -n 19 $PY render.py    # renders/*.png
```
Files: `bridge.py` (v3/common access), `v4_params.py`, `v4_parts.py`, `build.py`, `verify.py`, `render.py`.

## Concept
- **Hub + stem (1 printed part, `hub_stem`)**: disc dia 56 x 8 (z 52.7-60.7), 6 x M3 CSK; hollow column 31 x 22 outside, 3 mm walls (inside 25 x 16) up to the bar. Inside: the middle pad's buffer + one-touch fitting (nothing moves inside: the 10 mm stroke is the sliding head in front of the bar), the rear nut of the middle pad, the 3-to-1 collector (v3 proxy block, 12 g, turned: 24 x 15.6 x 10, ports re-assigned: 2 x phi4 on +-x, 1 x phi4 on +z, phi6 trunk on -y), the straight 10 mm middle tube. Side windows (phi 10.4) let the two phi4 port bosses of the collector out through the +-x walls; the phi6 trunk leaves through the -y wall at z = 62.8 (groove in the hub rim). A collar at the top (z 111.5-118.5, 45 deg gussets, x +-25.7) carries 2 M3 heat-set inserts for the bar.
- **Pad bar**: 6061-T6 flat bar 20 x 15 x 150 (stadium ends), z 118.5-133.5, 3 x phi10.5, 14.25 AF x 3.2 hex pocket in the front face for each front M10x1 nut (this is the orientation key, same as v3), 2 CSK M3 from the front into the collar inserts. Rear nuts sit inside the stem / behind the bar. Washers dropped (hard aluminium face).
- **Sensor pod (1 printed part, `sensor_pod`)** on the +y face of the stem: Gemini 305 in a front-open pocket (front flush with the pod face, held by 2 M3 CSK through the 3 mm back wall into the camera's back M3 holes at x = +-10, found in the vendor STEP), Zebra SE4710 in a pocket beside it at -x (front-load, 2 x M1.4 through the outside wall as in v3), PL3307 board in a slot under the camera back wall, open at the +x end so its cable leaves with the camera USB-C cable (USB-C receptacle is on the +x face of the camera, STEP: plug pocket + cable exit in a small +x lobe). A keel carries a flex channel from the engine down to the decoder slot. The pod is bolted to the stem by 4 M3 (x +-10, z 76/84) driven from inside the open stem before the pads/bar go in (heads in the cavity, inserts in the pod wedge). The pod is removable, so all 6 flange screws are reachable.

## Part list
Printed (2): `hub_stem` (PETG, PA12-CF optional), `sensor_pod` (PETG). Machined (1): `bar_al6061`.
Purchased in the CAD (17): 3 x ZP2-T4030WNK10-04, 6 x M10x1 nut, 3 x phi4 tube, collector (proxy, part number to confirm), phi6 trunk stub, SE4710, PL3307-B, Gemini 305.
Not modelled: 6 x M3x12 CSK (flange), 4 x M3x8 + 4 inserts (pod-stem), 2 x M3 CSK + 2 inserts (bar), 2 x M3x7 CSK (camera), 2 x M1.4 (engine), zip ties. Total CAD solids 20 (v3: 23); printed parts 2 (v3: 3).

## Check results (`data/verify.json`)
| check | v4 | v3 |
|---|---|---|
| bbox x y z (all solids) | **150.8 x 95.7 x 106.3** (y -30.0 .. +65.7) | 150.8 x 148.6 x 106.3 (y -43 .. +105.6) |
| bbox volume / convex hull | 1535 / **723 cm3** | 2382 / 1150 cm3 |
| mass (printed PETG x 0.55 fill, purchased as v3) | **0.332 kg** (printed 73 g, Al bar 104 g), CoG (-1.1, 14.6, 104.7) | 0.278 kg |
| parts | 20 solids = 2 printed + 1 machined + 17 purchased | 23 = 3 printed + 20 purchased |
| interference, 190 pairs, boolean | 0 mm3 everywhere; closest designed contacts (nut/bar 0.1, pad/bar 0.3, pod/camera 0.2, pod/scanner 0.25, pod/stem 0.1, stem/rear nut of M pad 1.0 mm) | 0 |
| keep-out z > 119 | pass: only pads, front nuts and the bar (z 118.5-133.5); next highest: sensor_pod 116.9, scanner 114.7, camera 112.5 | pass |
| pressed pose (compressed TCP 154 on the spine, pad heads retracted 5 mm, neighbours 15 mm proud, `handoff.suction_link6_pose`, boxes 1-5) | min clearance to carton + neighbours **2.0 mm** (box 5, 12 mm thick; pad L lower body vs neighbour); 3.5 / 4.0 / 4.0 / 4.0 / 2.0 mm for boxes 1-5 (same as v3); bar front to the neighbours' front plane **5.5 mm**; pad heads touch the target only | same numbers |
| camera | axis through (0,0,199): miss **0.000 mm**; front face (0, 33.9, 105.9), tilt 20 deg; range to the aim point **99.1 mm** (>= 90); range to the middle lip only 63.0 mm; camera max y 60.7, pod max y 65.7; camera top z 112.5 | range to lip 93.6, tilt 39 deg |
| scanner | window (-40, 47, 110), axis (0.31, -0.36, 0.88); glare tilt to the spine normal **28.3 deg** (>= 12); aimed at the spine centre; top z 114.7; FOV rays clipped by the tool: **2 of 441 (0.45 %)** at the camera-pod shoulder | glare 12 deg, 0 clipped |
| scan pose (window -> spine along the axis; tool backed off from the pressed pose, spine tool-z 154) | 115 mm: tool **53 mm** back; **130 mm (nominal): 70 mm back** (spine at tool z 224.4, 65 mm in front of the lip plane); 150 mm: 93 mm back. Camera to the spine centre at 130 mm: 123 mm | 150 mm, ~104 mm back |
| decoder flex | farthest board corner 79.2 mm from the engine rear, reach 89.4 -> slack **10.2 mm** (needs >= 10, thin) | 57.4 mm |
| buffer stroke 10 mm | 0..10 mm in 1 mm steps: 0 collisions; 1.5 mm gap adapter-to-bar at full stroke; nothing in the stem moves | same |
| tubes | L / R: 1 bend R = 12.0 (min), 8.0 mm straight out of the fitting, 31 mm straight run, 0.15 mm above the hub; M: straight 10 mm, no bend; trunk phi6 through the -y wall | R 12 |
| STEP | all 20 parts re-import (vol. error 3e-9) | |
| printability (overhang > 45 deg, solid support upper bound, best of 6 build directions, same method on v3) | hub_stem 1.7 cm3 (flange face down), sensor_pod 31 cm3 (46 % of the part; z up) = **33 cm3** | carrier 88 + camera arm 16 + scanner arm 11 = **115 cm3** |

## Bar material decision
Aluminium 6061-T6 flat bar 20 x 15 x 150 (38.7 cm3): **104 g**; PA12-CF printed: 37 g (1.2 g/cm3 x 0.8 fill) and the tool would weigh 0.265 kg instead of 0.332. Both deflect < 0.02 mm at 10 N per outer pad (Al 0.001 mm, PA12-CF 0.011 mm). **Chosen: aluminium**: the SMC nut torque (2.5-3.5 N m on a 14 AF nut) would crush/creep a printed bar, the hex pocket and the thread seat stay exact, and the stock bar is used without facing it. Cost: +66 g; the tool is still 0.33 kg (budget 0.6 kg).

## What changed vs v3
- One printed stem + one printed pod instead of the carrier, the camera arm and the scanner arm; flange-to-lip cross-section is a 31 x 22 column, the 144 x 24 bracket is a clean 150 x 20 blade.
- Everything sensor-related is on +y within y <= 65.7 (v3: -43 .. +105.6): y extent 149 -> 96 mm, hull 1150 -> 723 cm3 (-37 %), printed support estimate 115 -> 33 cm3. Mass +54 g (aluminium bar).
- Collector hidden in the stem, one phi6 trunk through the -y wall; the flap, groove and 14 zip-tie slits are gone.
- Camera tilt 39 -> 20 deg (pod compact); scanner moved from -y to the -x side of the camera pod, glare tilt 12 -> 28 deg.

## Mismatches with the brief / assumptions
1. **Bar 15 mm thick, rear face z = 118.5** (brief 14.5 / 119): stock 20 x 15 flat bar used unmachined; front face unchanged at 133.5.
2. **y <= 60 not met**: camera body max y 60.7, pod 65.7 (3 mm walls; the 42 mm camera lies mostly along y at a 20 deg tilt).
3. **Tubes do not run along the bar's rear**: the built-in fitting points down (-z), the shortest R12 route drops 20 mm to z = 62.8 and runs horizontally over the hub (as in v3), entering the stem through side windows low in the column. No rear cover was added (optional).
4. **Collector re-posed**: the v3 proxy turned 24 x 15.6 x 10 and its three phi4 ports moved to the +-x and +z faces, the trunk on -y (part number still unconfirmed, same 12 g). Its +-x port bosses stick 1.5 mm out of the stem walls.
5. **Camera range**: >= 90 mm holds to the aim point (99 mm) as briefed, not to the middle lip (63 mm, below the good-depth minimum), and the outer pad lips (x = +-75) are outside the 88 deg depth FOV; the bar and pads also block 29 % of the camera's depth FOV rays (rays within 300 mm, from the front face). v3 saw all pad lips. The wrist camera is a spine camera here.
6. **Scanner**: 2 of 441 edge rays clipped (0.45 %) by the camera-pod shoulder, so `scanner.pass_` is false in `verify.json` strictly; flex slack 10.2 mm is just above the 10 mm limit. FOV orientation (42 deg along tool x) and the decoder connector position are assumed, as in v3.
7. Printing: pod prints with supports in the best of 6 directions (31 cm3 upper bound), so the "no unsupported overhang > 45 deg" rule is NOT fully met for the pod (the camera back face is 20 deg off horizontal). The hub_stem is nearly support-free (collar gussets 45 deg). Hub floor under the collector is 4.9 mm; pod walls 3 mm; min walls not otherwise checked by ray.
8. Decoder: slid in from the +x end and held by a snug fit plus tape/foam (no screw, hole pattern unknown); camera/decoder cables are not routed or tied in the CAD.
9. Pod is bolted from inside the stem: maintenance needs the bar and middle pad out. Camera retention by the 2 back M3 holes (M3 x 7: 3 mm wall + 4 mm of the 4.8 deep thread).
10. Mass: pad+fitting 15 g, collector 12 g, PETG fill 0.55 (conservative for the bulky pod) and Al density are assumptions; screws, inserts, cables not included.
11. Not checked: lean +-10 deg, vibration/stiffness of the printed stem under the pod moment, USB-C plug size (pocket 16 x 9 x 15 assumed, right-angle plug), thermal/vacuum aspects (unchanged from v3).
12. Observation about v3 (not changed): its USB-C relief lug is on the -x side of the camera cradle while the vendor STEP has the receptacle on the +x face with the same frame mapping; worth checking before v3 is printed.

## Renders (`renders/`)
`assembly_{iso,front,side,top}.png`, `pressed_on_box_{iso,side}.png` (16 x 230 x 250 spine at the lip plane, neighbours 15 mm proud, as v3), `compare_v3_v4_{iso,front}.png` (same scale, labelled), `exploded_iso.png`.
Exports: `step/assembly.step`, `step/parts/*.step` (20), `stl/hub_stem.stl`, `stl/sensor_pod.stl`, `data/{verify,mass,printed_parts}.json`.
