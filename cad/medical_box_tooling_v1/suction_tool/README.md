# suction_tool v3 - left-arm suction / scan end effector (medical-box shelf pick)

3-pad suction head (3 x SMC ZP2-T4030WNK10-**04**, one-touch phi4 built in) + on-tool push-in collector (3 x phi4 in, 1 x phi6 trunk out) + Zebra SE4710 barcode engine + Orbbec Gemini 305 wrist camera, bolted to the igus ReBeL flange.
The vacuum source is **remote** (KNF N 838 pump, vacuum switch and vent valve in an enclosure on the AMR, connected by a ~2 m phi6 trunk; not part of this CAD).
Parametric CadQuery source, everything in the URDF `link6` frame, millimetres (flange face z = 52.7, pad axes = +z, pad row along +x).
`../common/params.py` is imported read-only (through `params.py`, which only holds tool-specific values). Nothing was run on a robot or with ROS nodes
(only `xacro` + `check_urdf` with `ROS_DOMAIN_ID=173 ROS_LOCALHOST_ONLY=1`).
v3 = v2 with the reserved ZK2 envelope removed and an on-tool collector + single phi6 trunk instead of three tubes leaving the tool. v1/v2 numbers come from the earlier builds (kept outside this folder).

## Rebuild
```
PY="/media/farhan/My Passport/igus_nvidia/venvs/cad/bin/python"      # cadquery 2.7, OCP 7.8, trimesh, coacd (installed in this venv only)
export TMPDIR="/media/farhan/My Passport/igus_nvidia/scratch_suction_tool/tmp"   # / is nearly full: keep temp files off it
nice -n 19 $PY build.py      # ~3.5 min: step/, stl/, meshes/, urdf/, data/mass.json, data/printed_parts.json (deletes the old outputs first)
nice -n 19 $PY verify.py     # ~1 min: data/verify.json (all numbers below; also reads scratch v1/v2_baseline.json for the comparison)
nice -n 19 $PY render.py     # renders/*.png
```
Head height is ONE parameter (`LIP_Z`, default 159); camera pose, bracket, nuts, tubes follow from it. Other knobs in `params.py`: `PAD_PITCH` (60), `CAM_TILT_DEG` (39), `FIT_H` (14.7), `TUBE_*`, `COLLECTOR`, `COLL_Y`, `VAC`.
Files: `lib.py` (helpers adapted from `dual_arm_tooling_v1/tooling_lib.py`), `meshlib.py`, `massprops.py`, `parts.py` (geometry), `build.py`, `verify.py`, `render.py`.

## Comparison v1 -> v2 -> v3
| item | v1 | v2 | v3 | v2 -> v3 |
|---|---|---|---|---|
| bounding box x y z (all parts) | 209.0 x 162.7 x 112.3 | 150.8 x 147.6 x 106.3 | **150.8 x 148.6 x 106.3** | +1.05 mm in y (flap + trunk stub reach y = -43) |
| bbox volume | 3819 cm3 | 2366 cm3 | 2382 cm3 | +0.7 % |
| convex hull | 1694 cm3 | 1178 cm3 | **1151 cm3** | -2.3 % |
| (v2: hull with the ZK2 envelope fitted) | 1863 | 1396 | n/a (removed) | |
| printed parts only: bbox / hull | 209 x 144 x 86.8 / 1247 cm3 | 144 x 127.7 x 80.8 / 845 cm3 | 144 x 129.6 x 80.8 / 795 cm3 | hull -6 % |
| flange -> lip z (lip z) | 112.3 (165) | 106.3 (159) | 106.3 (159) | same (tube-limited, see below) |
| mass (default tool) | 0.356 kg | 0.264 kg | **0.278 kg** | **+14 g** (collector 12 g, trunk 1 g, longer tubes, flap, 6 mm cradle) |
| CoG | -3.2, 13.0, 97.3 | -0.4, 21.9, 100.1 | -0.5, 19.8, 98.2 | |
| printed parts | 6 | 3 | 3 | |
| purchased parts in the CAD | 27 | 18 | **20** | +2 (collector, trunk stub) |
| camera range / horizontal FOV margin | 96.2 mm / 4.4 deg | 93.6 / 3.6 | 93.6 / 3.6 | same (poses unchanged) |
| decoder flex slack | 4.5 mm | 57.4 mm | 57.4 mm | same |
| support estimate (solid upper bound) | 208 cm3 | 223 cm3 | 226 cm3 | +1 % (carrier 158, camera arm 54, scanner arm 14) |
| lip deflection (E 1.8 / 1.2 GPa / x2) | 0.105 / 0.157 / 0.31 | 0.104 / 0.156 / 0.31 | 0.104 / 0.156 / 0.31 | same |
| tube length on the tool | 3 x ~50 mm + manifold | 3 tubes leave the tool | 35 + 83 + 83 mm pad tubes, **1 trunk leaves** | |

What the ejector removal gave back: the 2 M4 rib inserts, the envelope bookkeeping, the thinned camera cradle (back to 6 mm plate and the 3 mm rear rail, +1.5 g). The floor disc stays phi56 although the ejector no longer forces it: 56 mm is the minimum that still holds the 6 flange screws on PCD 43 with their head and 3 mm wall (r 28). The camera/scanner poses were **not** moved: nothing in the freed space helped (camera arm length is fixed by the range >= 90 mm and the axis through (0,0,lip+40); the head height is fixed by the tube run, not by the ejector).
Everything that got worse in v3 vs v2: +14 g mass; +1.05 mm bbox in y (the flap carries the collector pocket, the trunk groove and the trunk strain relief out to y = -43; the scanner arm alone reached -41.95); +2 purchased parts and an unconfirmed collector part number; the carrier flap is a bigger overhang-free but bulkier print (+1 % support); one more tube (trunk) must be dressed along the wrist so that joint 6 twists only that tube; the trunk exits 39 mm off the joint-6 axis (not on the axis).

## Design (link6 frame, mm)
| item | value |
|---|---|
| pad lips (uncompressed), coplanar | z = **159.0** (spread 0.0), x = -60 / 0 / +60 (pitch 60, outer lip edges +-75.4 = span 150.8), y = 0; compressed TCP z = 154 |
| pad bracket | stadium 144 x 24, z = 119.0 .. 133.5 (lip-40 .. lip-25.5), 14.5 thick, 3 x phi10.5 holes, 14.25 AF hex pocket in the front face for the front M10x1 nut |
| carrier (one printed part) | floor phi56 x 8 (z 52.7 .. 60.7) with 6 x M3 CSK on PCD 43 (angles 45 + 60k), phi20 bore; two ribs 10.4 x 24 at x = +-[24.8, 35.2] up to the bracket; a 2.5 mm web between the ribs that closes the decoder pocket; **collector flap** at floor level (x +-46, y -43..-12) with the collector pocket, the phi6 trunk groove and 14 zip-tie slits; 8 M3 inserts (4 scanner arm on -y, 4 camera arm on +y); two phi6.2 access holes through the bracket for the flange screws under it |
| collector | 24 x 10 x 16 block, 3 x phi4 push-in ports (-x, +x, +y faces, bosses phi9 x 5) and 1 x phi6 push-in port (-y face, boss phi11 x 4), port axis z = 62.8, centre (0, -30), **base z = 54.8 (2.1 mm above the flange plane) sunk in the flap pocket**, retained by 2 zip ties; ASSUMED mass 12 g |
| pad tubes | phi4: pad fitting end (z 82.8) -> straight 8 mm -> R12 bend(s) -> collector port. M: 35 mm, 1 bend, into the +y port. L / R: 83 mm, 2 bends (down, toward -y, then toward the centre along y = -30), into the -x / +x ports; every run lies 0.15 mm above the flap top; minimum bend radius 12, straight >= 8 mm out of the fitting, 6-31 mm elsewhere |
| trunk | phi6 leaves the collector along -y at z = 62.8 (groove in the flap, zip ties at y = -41.4), stub ends at y = -43 in the CAD, then runs down the arm into the AMR enclosure (about 2.0 m in total) |
| camera | Gemini 305 at +Y: front-face centre (0, 76.29, 104.78), optical axis (0, -0.629, 0.777) = 39.0 deg tilt about X, passes (0, 0, 199) exactly; mounted by its bottom face (2 x M3, +-10 mm); cradle plate 6 mm; USB-C (-x side) 12 mm clear zone + tie slot lug |
| scanner | SE4710 at -Y: window centre (0, -31.19, 111.5), axis (0, 0.208, 0.978) = 12 deg about X, crosses y = 0 at exactly 150 mm from the window (z = 258.2, 99.2 mm in front of the lip plane), window 47.5 mm behind the lip plane; cradle open to the front (no cover); M1.4 screws through the +w wall, driven through a window in the arm's attach plate |
| decoder | PL3307-B board in a pocket inside the scanner arm directly under the engine (x +-19.95, y -12..-39.4, z 84.8..91.6), opening toward the carrier, closed by the carrier web when the arm is mounted; flex slot to the engine rear |
| vacuum source (not in the CAD) | KNF N 838 KN.29DC-B 24 V pump + SMC ZSE20B switch + 24 V N.C. vent valve + dust and HEPA filters in an enclosure on the AMR; phi6 trunk of about 2.0 m (phi8 x 6 ID section optional) |

Pad keying: the buffer body has **no flat or hex** on the thread (catalog drawing), so the 30 mm axis is fixed by two M10x1 14 AF nuts: the front nut is captured in the printed hex pocket (3 mm deep), the rear nut + 1 mm steel washer is tightened on the rear face (z = lip-40). Rotating the pad would have to move it axially against the clamp; add Loctite 243 on the rear nut. Align the oval with the printed index line before the final tightening.
Head height (159) is set by the tubes: fitting 14.7 + 8 mm straight + R12 bend over the 8 mm floor/flap top, minimum 158.85. Not shortened further.

### Collector part (check of the research catalogs)
The research folder (`zk.txt`, `zp.txt`, `zp2f.txt`) contains only the ZK2 and ZP2 catalogs; **neither the KQ2 nor the KM family is in it** (KQ2 only appears as fitting codes: KQ2H04-M5N / KQ2H06-M5N pad fittings, KQ2S23-M3G, KQ2P). So no SMC push-in manifold/branch block could be confirmed or dimensioned from the files. The CAD uses a proxy: 24 x 10 x 16 mm body, phi4 ports 9 mm boss diameter x 5 long (as the KQ2H04 body OD 9, release-ring height), phi6 boss 11 x 4, port axis at mid-height. **Order code: to confirm** with SMC (a 1 x phi6 in / 3 x phi4 out push-in branch block, ports on three faces + trunk on the fourth, or one custom milled block with M5 threads and KQ2H04-M5N / KQ2H06-M5N fittings, which would give exact dimensions from the catalog drawing p.183). Off-the-shelf fallback from standard union fittings (also to confirm): 2 x KQ2T04-00A (union tee, phi4) + KQ2R06-04 (phi6 -> phi4 reducer) in a chain; the pocket in the flap would need to be reshaped. `COLLECTOR` in `params.py` is the single place to change the block dimensions.
Routing for joint-6 rotation: all three pad tubes end at the collector on the tool, so a joint-6 rotation never twists three tubes; it acts on the one phi6 trunk, which leaves along -y 39 mm off the axis. Dress it with a service loop of >= 100 mm and clamp it to the wrist at one point; if the flange turns out to be hollow (the phi20 floor bore is kept), take the trunk through the bore along the axis so that only torsion remains (unconfirmed).

## Vacuum gate (vacuum-OK check) - estimate, not a measurement
Source (operator decision 2026-10-13): the robot stands on an AMR in the hallway, no compressed air. Vacuum comes from a **KNF N 838 KN.29DC-B** brushless 24 V diaphragm pump (data in `../common/params.py` `VACUUM_PUMP`) in an enclosure on the AMR. Off-tool circuit: tool collector -> ~2.0 m phi6 trunk down the arm into the AMR -> inline dust filter -> tee with the **SMC ZSE20B** PNP vacuum switch and a 24 V N.C. vent valve (release) -> pump inlet; pump outlet -> HEPA filter. The tool CAD is unchanged (3 x ZP2, phi4 tubes, 3-to-1 collector, one phi6 trunk). The switch sits **at the pump end**, so it reads a stronger vacuum than the pads whenever air flows; the model reports both.
Trunk options (`VAC` in `params.py`, lengths are parameters): **A** all phi6 x 4 ID, 2000 mm; **B** phi6 x 4 ID 1200 mm on the arm + phi8 x 6 ID 800 mm arm base -> AMR.
Model (`verify.py`, section `vacuum_gate`): pump curve (`curve_mbar_lpm`, standard L/min vs absolute inlet mbar, linear interpolation; vacuum kPa = (1000 - p_abs mbar)/10); pad leaks as before (ISO 6358 style, open pad line passes at most 25.7 L/min, sealed pad 1 L/min, rough carton 4 L/min per pad - assumptions to be measured); trunk friction by Darcy-Weisbach, stepped along the tube at the actual (rarefied) density and volumetric flow, laminar / Blasius friction factor (Re ~ 100-1700, laminar everywhere); fittings, bends, filter and valve losses not included. Volumes: tool side 4.1 cm3 + trunk 25.1 cm3 (A) / 37.7 cm3 (B) + filter/switch/tee 10 cm3 (ASSUMED) = **39.3 / 51.8 cm3**.
| case | A pads / switch [kPa] | B pads / switch [kPa] |
|---|---|---|
| 3 pads sealed (1 L/min each) | -78.6 / -80.0 | -79.0 / -80.0 |
| 3 pads at 2 L/min each | -66.8 / -68.6 | -67.3 / -68.6 |
| 3 pads on a rough carton (4 L/min each) | -46.6 / -52.6 | -48.5 / -52.5 |
| **one pad lost (open line), 2 sealed** | **-19.0 / -30.3** | **-20.7 / -28.6** |
| one pad half lost (3 mm2 leak = 28.6 L/min) | -16.4 / -28.2 | -18.0 / -26.2 |
| one pad 1 mm2 leak | -48.0 / -53.7 | -49.9 / -53.7 |
Gate margins at the switch (rough carton passes / lost pad detected), A and B:
| threshold | rough carton margin | lost pad margin |
|---|---|---|
| -35 kPa | 17.6 / 17.5 | 4.7 / 6.4 |
| -40 kPa | 12.6 / 12.5 | 9.7 / 11.4 |
| -45 kPa | 7.6 / 7.5 | 14.7 / 16.4 |
| -50 kPa | 2.6 / 2.5 | 19.7 / 21.4 |
**Recommended gate: -40 kPa at the switch** (hold 150 ms): midpoint of the lost-pad (-30 kPa) and rough-carton (-52.5 kPa) readings, rounded to 5 kPa; both margins (>= 9.7 kPa) exceed twice the switch accuracy (+-2 kPa) for either trunk. -35 leaves only 4.7 kPa on the lost pad; -50 leaves 2.5 kPa on the rough carton (within the switch accuracy) so a rough-carton pick could be refused. The rough carton reads stronger than the gate and the lost pad weaker (`pass_` true), but note the rough-carton pads themselves only reach -47 / -49 kPa. A half-lost pad (3 mm2, -26..-28 kPa) is caught; a single pad with a 1 mm2 leak (-54 kPa) is not (the gate checks pad loss, not marginal seal quality).
Evacuation (lumped volume, 3 pads sealed): to **-40 kPa 0.05 s (A) / 0.06 s (B)**, to -50 kPa 0.08 s / 0.09 s at the pads; on a rough carton -40 kPa takes 0.11 s / 0.12 s and **-50 kPa is never reached at the pads** (steady -47 / -49 kPa). So a 150-300 ms evaluation delay is ample.
Energy per pick (pump only, 8 s pump-on ASSUMED, 24 V): **384 J = 0.107 Wh** at 2.0 A (chart value at 500 mbar), 442 J = 0.123 Wh at I_max 2.3 A; the current at deep vacuum and low flow is probably lower, so these are upper estimates. Valves, switch and AMR converter losses not included.
**Assumption risk**: leak per pad, open-line limit and the datasheet curve (read off a chart) dominate; measure (a) 3 pads sealed on the real carton and (b) one pad lifted at the switch and set the gate at the midpoint. Pad-tube drop is neglected for the sealed cases.

## Hardware / BOM
| qty | item | order code | note |
|---|---|---|---|
| 3 | suction pad + 10 mm non-rotating buffer with built-in one-touch phi4 | SMC `ZP2-T4030WNK10-04` | catalog p.183: fitting KQ2H04-M5N, 8 AF, 14.7 high; **assumed fully outside the 61.5 mm overall length**; 15 g each (ASSUMED) |
| 6 | M10x1 nut, 14 AF x 3 | SMC ZP2 nut or DIN 439 M10x1 | 2 per pad |
| 3 | steel washer 10.5 / 20 / 1 | | under the rear nut |
| 1 | push-in collector, 1 x phi6 in / 3 x phi4 out | **to confirm** (see Collector part) | proxy 24 x 10 x 16, 12 g ASSUMED |
| 3 | tube phi4 x 2.5, 35 / 83 / 83 mm (+ trim) | SMC `TU0425` | pad fitting -> collector, R >= 12 bends |
| 1 | tube phi6 x 4, ~1 m (to the ejector) | SMC `TU0604` | trunk |
| 1 | **off-tool (AMR enclosure):** KNF N 838 KN.29DC-B 24 V brushless diaphragm pump (1.9 kg) + SMC ZSE20B PNP vacuum switch + 24 V N.C. vent valve + inline dust filter + HEPA filter on the outlet | see `../common/params.py` `VACUUM_PUMP`, `VACUUM_PARTS`; valve and dust filter model **to choose** | trunk phi6 x 4 ID about 2.0 m (optionally phi8 x 6 ID for ~0.8 m into the AMR) |
| 1 | Zebra SE4710 imaging engine (89.4 mm flex) | `SE4710` (confirm the exact -xxx SKU) | 4.8 g |
| 1 | PL3307-B decoder board | Zebra PL3307 B | 26.91 x 39.04 x 6.37 |
| 1 | Orbbec Gemini 305 | Orbbec 100100614 / 100100615 (drawing P/N) | 68 g |
| 6 | ISO 10642 M3 x 12 countersunk | | carrier -> flange (4 mm engagement) |
| 4 + 4 | M3 x 10 DIN 912 | | scanner arm / camera arm -> rib inserts |
| 2 | M3 x 8 DIN 912 | | camera -> cradle (6 mm plate with counterbore) |
| 2 | M1.4 screws (2.1 mm max thread engagement) | | engine in the cradle (Fig. 2-4: holes C1/C2) |
| 8 M3 | heat-set inserts (hole 4.2 from `common`) | | ribs |
| 2 | silicone plugs phi6 | | close the two flange-screw access holes in the bracket after assembly |
| ~12 | zip ties 2.5 x 100 | | collector (2), trunk (2), L/R tubes (2) |

## Assembly order (all screws reachable: free-cylinder / sweep tests in `verify.py`, section `screw_access`)
1. **Carrier alone** onto the flange: 6 x M3x12 CSK from +z (two of them, at (-20.8, +5.6) and (+20.8, -5.6), through the phi6.2 holes in the bracket). Plug the two holes afterwards. Re-teach the TCP afterwards (no dowel/pilot on the ReBeL).
2. Pads: front nut into the hex pocket, pad through the bracket from the front, washer + rear nut from below (flat spanner from +y or -y), set the oval index, tighten (SMC limit 2.5-3.5 N m, use the low end), Loctite.
3. Collector: drop it into the flap pocket from +z (nothing above it yet), 2 zip ties over the block; push the phi6 trunk into its port and tie it at the flap end.
4. Tubes: push the three phi4 tubes into the pad fittings (from below) and the collector ports (bends R >= 12), tie L and R on the flap.
5. **Scanner arm, alone on the bench**: decoder board slides into the pocket along -y, engine into the cradle from the front along the scan axis, 2 x M1.4 driven from the +y side through the window in the attach plate; flex through the slot. Then the arm on the ribs: 4 x M3x10 from -y (the carrier web closes the decoder pocket).
6. Camera arm: 4 x M3x10 from +y; camera on the cradle: 2 x M3x8 from below along -u; USB-C cable through the tie slot.
7. Run the trunk along the wrist and arm into the AMR enclosure (service loop, clamps).

## Verification (`data/verify.json`, from the exported CAD, run on this build)
| check | result |
|---|---|
| interference, all 253 pairs of 23 solids (3 printed + 20 purchased), boolean common volume | **0 mm3 for every pair** (max 0.0). 42 pairs are < 3 mm apart (designed contacts: pad-bracket 0.3, nut-pocket 0.125, rear nut-bracket 1.0 washer gap; tubes lie 0.15 mm above the flap); screws, inserts and zip ties not modelled |
| keep-out req. 3 | z > lip-25: only `pad_*`. lip-40 < z <= lip-25: carrier slice is exactly the bracket (volume outside the 144 x 24 stadium = 0, y = +-12.0), the rest are pads and captured front nuts. Everything else is at or below lip-40 = 119 (collector top 70.8, camera 117.0, washers 119.0) -> **pass** |
| neighbours (zero gap, 15 mm proud) | tool geometry within 15 mm of the lip plane has |y| <= 4.0 mm -> clearance to a neighbour face 3.5 / 4.0 / 4.0 / 4.0 / **2.0 mm** for boxes 1-5 |
| camera | range **93.6 mm**; lip outlines inside the 88 x 65 depth FOV with **3.6 deg** (horizontal, needs 3) and 16.1 deg (vertical); axis misses (0,0,lip+40) by 0.0 mm; 100 % ray visibility excluding the z = lip samples under a pad footprint (85.7 % overall) |
| scanner | crossing y = 0 at **150.0 mm** (z = 258.2, 99.2 mm in front of the lip plane); window 47.5 mm behind the lip plane; 441 rays: **0 clipped** within 300 mm |
| decoder | farthest board corner 32.0 mm from the engine rear; flex reach 89.4 mm -> **57.4 mm slack** (>= 10) |
| buffer stroke | 0..10 mm in 1 mm steps against every other part: 0 collisions; 1.5 mm gap at full stroke |
| tubes / collector | 3 pad tubes: bend R12 (min), 8 mm straight out of each fitting, 6..31 mm elsewhere, 0.15 mm above the flap; collector z 54.8..70.8 (below the keep-out, 2.1 mm above the flange plane); trunk stub to y = -43 |
| vacuum gate | see section above (pump on the AMR, 2 m trunk): at the switch lost pad -29..-30 kPa, sealed -80 kPa, rough carton -52.5 kPa; 39.3 / 51.8 cm3; recommended gate -40 kPa at the switch |
| STEP / STL / stale files | 23 STEP parts re-import valid (volume error < 1e-3); assembly.step re-imports with 23 solids; 3 printable STL watertight; **`step/ejector_envelope.step` and the manifold/elbow/ejector outputs no longer exist** |
| mass / CoG | **0.278 kg** (CoG -0.5, 19.8, 98.2); printed 0.120 kg; screws, inserts, ties not included (~12 g); budget 0.6 kg met |
| deflection | 0.104 mm (E 1.8 GPa), 0.156 mm (E 1.2 GPa), 0.31 mm with a x2 factor; target 0.5 mm -> pass |
| lip coplanarity / pad span | z = 159.000 on all three lips; span 150.8 mm: margin to each spine end 34.6 (220 mm box) .. 49.6 mm (250) -> >= 25 mm for all five boxes |
| screw access | all 27 checks pass: 6 flange screws (carrier alone), spanner on the 3 rear nuts, 4 + 4 arm screws, 2 camera screws, decoder board and engine slide-in, 2 x M1.4, collector drop-in, 3 tube-in-fitting clearances |
| printability | carrier: flange face down, 144 x 71 x 81 mm, support estimate 158 cm3; camera arm 54 cm3; scanner arm 14 cm3; all fit a 220 x 220 bed; 1 % wall percentile >= 1.9 mm |
| URDF | `xacro` + `check_urdf`: root link6, 4 frames; 23 visual meshes, 63 convex collision pieces (hull volume 1.15 x part volume; tubes excluded), inertial from the CAD (0.278 kg) |

### URDF
Macro `left_suction_tool(prefix, parent, mesh_dir:=file://<abs path of this checkout>/meshes)`, same macro and frame names as v1/v2. **Frame values are unchanged from v2** (poses did not move): `suction_tcp` (0, 0, 159), `suction_tcp_compressed` (0, 0, 154), `scanner_optical_frame` (0, -31.187, 111.5) rpy -12 deg, `wrist_camera_mount` (0, 76.294, 104.784) rpy 39 deg. Only the inertial changed (0.278 kg, CoG -0.49, 19.79, 98.24 mm). Mesh paths are absolute `file://` URIs.

## Files
`params.py parts.py build.py verify.py render.py lib.py meshlib.py massprops.py README.md`
`step/assembly.step` (23 solids), `step/parts/*.step` (23; **no `step/ejector_envelope.step` any more**), `stl/*.stl` (3 printable parts, mm), `meshes/visual/*.stl` (metres, link6 frame, 23 parts), `meshes/collision/*.stl` + `manifest.json` (63 convex pieces, metres, coacd; tubes excluded),
`urdf/left_suction_tool.urdf.xacro`, `data/{verify,mass,printed_parts}.json`, `renders/assembly_{iso,front,side,top}.png`, `renders/pressed_on_box_{side,iso,front}.png` (16 x 230 x 250 spine at the lip plane, two neighbours 15 mm proud).

## Mismatches / assumptions / unconfirmed
1. **Collector part number and dimensions are unconfirmed**: no KQ2/KM push-in manifold/branch block is in the research catalogs (see Collector part); the CAD block is a proxy (24 x 10 x 16, 12 g assumed, ports on three faces + trunk on the fourth). If the real part has all phi4 ports on one face, the L/R tube routing and the flap pocket change.
2. **One-touch fitting height**: 14.7 mm read from catalog p.183 and assumed fully outside the 61.5 mm overall length; if part is inside A, the head can drop up to ~5 mm (limit formula in `verify.json -> tubes`). Fitting body modelled as phi9. Pad body is a proxy from the drawing dimensions (no SMC STEP).
3. **Vacuum gate numbers are a model, not a measurement** (pump curve read off a datasheet chart, 25.7 L/min open-line limit, 1 L/min seal leak, 2 m trunk friction by Darcy-Weisbach without fittings, 10 cm3 filter/switch/tee volume and 8 s pump-on ASSUMED): recommended gate -40 kPa at the switch leaves 12.5 kPa (rough carton) / 9.7 kPa (lost pad); measure and set the threshold on the real tool. The switch reads the pump end, 2-11 kPa stronger than the pads under flow.
4. **Joint-6 twist**: the trunk exits 39 mm off the axis, so rotation of joint 6 bends/twists that one tube; the through-bore route needs a hollow flange (unconfirmed). The trunk dressing (service loop, clamp) is not modelled.
5. **Gemini 305 body = 177-face convex envelope** of the vendor STEP main solid, posed from the drawing. Only the 2 x M3 are used. **SE4710**: block 22.3 x 8.1 x 13.7, hole pattern from Fig. 2-4 (signs not verifiable), FOV orientation (42 deg along tool X) assumed; M1.4 engagement 2.1 mm is thin, the engine is held by the pocket and 2 screws only (no cover, no pegs).
6. **Decoder**: slack uses the farthest board corner; the real connector position on the PL3307 is not in the research files.
7. **Scan geometry**: the 12 deg tilt / 150 mm standoff puts the scan plane 99 mm in front of the lip plane (the arm scans about 100 mm away from the spine, not at the pressing position).
8. **Mass**: effective PETG fill 0.55, pad+fitting 15 g, collector 12 g and the purchased-part masses are assumptions; screws, inserts, flex, cables and ties not included. Deflection is a first-order beam estimate (0.31 mm with a x2 factor).
9. **Printing**: carrier support estimate 158 cm3 (solid upper bound; tree supports 30-50 %); if unacceptable add 45 deg gussets under the bracket ends (adds mass) or split the bracket.
10. **Integration impact**: `../common/handoff.py` and `../integration` use the TCP/lip z (now 159.0 / 154.0 since v2) and, until now, `step/ejector_envelope.step` of this folder. **That file no longer exists in v3** (the reserved ZK2 envelope is removed); integration scripts that read it must be adapted. No other file used by integration was renamed: `step/assembly.step`, `step/parts/*.step`, `meshes/`, `urdf/` keep their names (the part list changed: manifold/elbow/plate/lid/cover gone, `collector` and `trunk_tube` new).
11. Neighbour lean of +-10 deg is handled by rolling the tool about z, not checked here. The pad lip is modelled as a 1.5 mm oval slab (the NBR lip is a bellows).
