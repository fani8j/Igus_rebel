# rebel_dual_arm_calibration

Eye-to-hand calibration of the dual-arm cell with one static scene camera
(Orbbec Gemini 336, 1920x1080 colour). One solve gives

- `arm_right_base_link` pose in the world frame (world = `arm_left_base_link`),
- `camera_link` pose in the world frame,
- the rigid mount of each target on its arm (`link6 -> target`, a by-product).

The result is written as `dual_arm_calibration.yaml` in the structure read by
`rebel_dual_table_description` (URDF arm_right pose, static camera TF), plus a full JSON report.

The recorder never moves a robot, and nothing in this package does. You (or MoveIt/teleop) move the arms, one at a
time, and call a service when the arm is still.

## What the arm targets can and cannot give you (read this first)

With **one static camera** the "joint" fit of the two arm targets is mathematically the same as calibrating each
arm separately and chaining the results: the left data fix `T_world_camera` (world = left base) and the right data
fix `T_camera_rightbase`; nothing in the data couples the two. `solve` reports both ("composed vs joint") and they
agree to a fraction of a millimetre by construction, so that agreement is **not** a check of the result. Any
error of the left arm's camera pose is passed on 1:1 to the right base.

Two things limit the accuracy and neither is visible in the pixel residual alone:

- the arms are low-cost: **joint zero offsets / kinematic error** (a few tenths of a degree, i.e. 2-4 mm at the
  flange) are the dominant unmodelled error. The std devs below are detection noise only;
- the arms are not coupled by the camera data.

**What a PASS means.** The trust gate (below) only demonstrates that the two arms agree **at the held-out check points**
(free stickers that are never used in any fit) within a limit, with the written base and the nominal forward kinematics
the robot runs. It does **not** certify the accuracy elsewhere in the workspace. Residuals of points that were used in a
fit (touch mismatch, "fit residual") are FRE (fiducial registration error, in-sample, optimistic); only points that were
not used (check points) give TRE (target registration error), see Fitzpatrick/West. Use the recommended padding
(`max(20, 2 x max check-point error + 10)` mm) for collision avoidance between the arms.

Therefore the package offers (all optional, all off the critical path of the basic procedure):

1. a **fixed table board + TCP touches**: the knife tip of each arm is put on the same printed points of a board
   that the camera also sees. This closes the loop left base - board - right base - camera - left base and gives
   the touch mismatch in mm (a fit residual); free **fit stickers** widen the touch geometry and free **check
   stickers** (never used in a fit) give the held-out check-point error (TRE);
2. **joint zero offsets** (`--joint-offsets`) as regularised nuisance parameters (6 per arm, Gaussian prior
   0.3 deg), computed with forward kinematics from the URDF saved in the session;
3. **held-out validation**, a **correlation matrix** and an optional **intrinsics refinement**.

## Which number to trust (the deployed robot runs NOMINAL forward kinematics)

The controller and MoveIt use the nominal URDF/TF forward kinematics; **fitted joint offsets are never applied**.
What matters for collision avoidance between the arms is therefore the error of the *nominal-model prediction*
`T_world_rightbase * FK_nominal(q)` against where the tip physically is, not the base error in the gauge the data
identify. The two differ:

- The headline synthetic numbers (right base 13.8 -> 0.22 mm with `--joint-offsets`) are errors in the *identifiable
  gauge*, where the joint-1 zero offsets are absorbed into the base yaw (which is exactly right for nominal FK).
  Against the unrotated physical base the same result is off by about 3.4 mm (0.45 m x 0.5 deg of joint-1 offset),
  see `scan_results/dual_arm_calibration_20261006/`.
- **That synthetic result is circular**: the injected error is exactly the modelled one (joint zero offsets), so the
  fit recovers it almost perfectly. Real arms also have link-length, axis, backlash and compliance errors that the
  offsets only partly absorb; real gains will be smaller, and the held-out validation and the touch checks are
  there to find out how much smaller.
- Even a *perfect* base leaves a nominal-model prediction error of several mm over the workspace when the arms
  have 0.4-0.5 deg joint offsets (about 7 mm RMS, up to 15 mm, in the study, `nominal_model_study.json`): no base pose can remove it.
  Use it for the safety padding below.
- **Which base is written.** Without `--joint-offsets` the (only) nominal fit. With `--joint-offsets` the offset-fit
  base (its yaw absorbs the joint-1 offset; the study shows it predicts the nominal tip positions about as well as the
  physical base and clearly better than a plain refit with the offsets forced to 0) is written **only if the held-out
  validation recommends KEEP**; otherwise (validation says no, was not run, or could not evaluate the offsets) the plain
  nominal fit is written (`nominal_refit` in the report). The report (`written_base`), the notes of the YAML and the gate
  say which fit was written and why. All touch mismatch numbers, the check-point error and the trust gate use the
  **nominal FK** with the written base; the same numbers with the fitted offsets applied are only given as
  `offset_corrected` diagnostics.
- **The fitted joint offsets are nuisance parameters of the solve only and must NEVER be applied to the robot or the
  URDF**: re-referencing the joints invalidates them, and they absorb backlash, compliance and link errors, not only
  zero offsets.

## Targets

Each arm carries a small rigid target on its flange or gripper. The pose of the target
relative to `link6` does not need to be known; it is solved for.

| Type | Config | Points per view |
|---|---|---|
| ChArUco board, default 3x3 squares of 35 mm (105 mm board, 26 mm markers) | `config/targets.yaml` | 4 markers x 4 corners + 4 inner corners |
| ChArUco 4x4 squares of 25 mm (100 mm board) | `config/targets_charuco_4x4_25mm.yaml` | 8 markers x 4 corners + 9 inner corners |
| Single ArUco marker (fallback), 59 mm, ids 13 and 14 | `config/targets_aruco.yaml` | 4 |

The two boards use the same dictionary (`DICT_4X4_50`) and **disjoint marker IDs** (left 0..3,
right 20..23), so one image can contain both and the detector knows which arm is which. Overlapping
or out-of-range IDs are rejected when the config is loaded.

The ChArUco layout is implemented in `targets.py` and does not use `cv2.aruco.CharucoBoard`,
because OpenCV 4.5.4 (system, Humble) cannot set custom marker ids. It detects the ArUco markers,
fits a homography, and refines the inner chessboard saddles with `cornerSubPix`. The same code gives
identical results on OpenCV 4.5.4 and 4.12 (checked). Marker corners are also used, with lower weight.

The **table board** (`table_board` entry of `config/targets.yaml`, 5x4 squares of 40 mm = 200x160 mm, markers
30 mm, ids 40..49 in the same dictionary, so disjoint from both arm boards) is described below.

Which size to print: see `scan_results/dual_arm_calibration_20261006/README.md`. In short, the markers
must show a shortest projected side of about 20-25 px in the 1080p image, so the 4x4/25 mm board only works
close to the camera (about 0.8 m); the 3x3/35 mm default works to about 1.0 m, and the 59/80 mm single
marker is the fallback for greater distances or poor light.

### Print the targets

```bash
ros2 run rebel_dual_arm_calibration generate_boards --out-dir ~/dual_arm_targets   # uses config/targets.yaml
# or: python3 -m rebel_dual_arm_calibration.targets --targets <yaml> --out-dir <dir> --dpi 600 --paper A4
```

This writes `target_left.png/pdf`, `target_right.png/pdf` (600 dpi, A4, exact physical scale, with
a 50 mm scale bar and labels for IDs and sizes) and `target_table.png/pdf` (the table board on **A4 landscape**, with the
numbered touch crosshairs). Print at 100 % (no "fit to page"), verify the 50 mm
bar with calipers, and mount the print on a stiff flat plate (aluminium or acrylic, matt, no glossy
laminate). Fix it rigidly to the flange or gripper; any wobble ruins the calibration.

**Measure the printed square and marker size with calipers** (several squares across the board, divide)
and enter the measured values (metres) in `square_length` / `marker_length` of your targets file. A 1 %
scale error shifts the camera position by about 1 % of the camera distance (about 8 mm at 0.8 m).
**Printers scale x and y differently** (paper feed vs. print head): measure both directions and enter the y values in the
optional `square_length_y` / `marker_length_y` (default: the x values); `square_length` / `marker_length` are the sizes
along the board x axis, the `*_y` ones along the board y axis (the direction "down" as you read the print). Board points,
detection, the solver, the generated print files and the touch crosshair coordinates all use the two scales.
The target frame origin is the board centre; the mount offset is relative to this frame.

### Hover tiles (approach-assist targets: ids 13-15, touch ids 101-113)

The approach assist (below) finds printed crosshairs from the pose of an ArUco marker. They are printed as three small **tiles**
A, B, C on ONE A4 page (the old single hover sheet is replaced; a `hover_sheet:` section in a targets file still works as one tile
named "sheet").

```bash
ros2 run rebel_dual_arm_calibration generate_tiles --out-dir ~/dual_arm_targets    # reads config/targets.yaml ('hover_tiles')
# or: ros2 run rebel_dual_arm_calibration generate_tiles --targets <your targets.yaml> --out-dir <dir> --dpi 600
```

It writes `hover_tiles.pdf` and `hover_tiles.png` (A4 portrait, 600 dpi, exact scale: one pixel = 25.4/600 mm). Every tile is a
100 x 100 mm square with a thin cut outline, its label (A / B / C) and marker id, a 50 mm DICT_4X4_50 marker placed **off-centre**
(designed 50 mm edge) and three touch crosshairs (fine cross with a 1 mm centre ring and the touch id beside it) at irregular
positions in the free area, as far apart as the tile allows. A 50 mm scale bar and the line "print at 100 % / Actual size; measure
the 50 mm bar in x and y; marker edge 50 mm" are at the bottom.

1. **Print at 100 %** ("Actual size", no "fit to page"), matt paper.
2. **Measure** the printed marker edge (the outer edge of the black border) with calipers in **x and y** (and check the scale bar):
   printers scale x and y differently. Enter the values in `hover_tiles: marker_length` (x) and `marker_length_y` (y): ONE pair for the
   whole print (all three tiles come from the same page). The crosshair positions are design millimetres in the marker frame and are
   scaled by `marker_length / design_marker_length` (x) and `marker_length_y / design_marker_length` (y) at load time, exactly like the
   old sheet; do not edit `design_marker_length` or the point coordinates unless you reprint.
3. **Cut** along the outlines and **glue** each tile flat (no bubbles, no curl) on its own rigid 100 x 100 mm plate.
4. **Place** A and B spread around the table board where both arms reach (not on top of the board, a free hand width from it); put
   **C on a ~40 mm block** (flat top, at least 150 mm from the table-board centre) so it also gives a check point at another
   height. A tile lying exactly face-on to the camera can give an ambiguous marker pose (the gate says "ambiguous IPPE pose ... tilt the sheet"): tilt it a little, never edge-on.
   They do not have to be in view all the time: each tile is detected, gated and checked on its own, a hidden tile only greys its own rows
   ("tile B: marker 14 not found").
5. The touch ids: **A: 101, 102, 103; B: 104, 105, 106; C: 111, 112, 113**. 101-104 are FIT and 111, 112 CHECK stickers of
   `touch_stickers` (so they can be captured as touches); 105, 106, 113 are hover-only points (the Touches tab lists them under their
   tile, greyed, not capturable) unless you add them to `touch_stickers`.

`hover_tiles` in the targets file (the default layout is the one in `config/targets.yaml`; `targets_aruco.yaml` has it commented out
because its arm markers already use ids 13 and 14):

```yaml
hover_tiles:
  dictionary: DICT_4X4_50
  tile_size: 0.100              # m, printed tile edge (cut line)
  design_marker_length: 0.050   # m, designed marker edge
  marker_length: 0.050          # m, MEASURED x      (one value for all tiles of one print)
  marker_length_y: 0.050        # m, MEASURED y
  tiles:
    - marker_id: 13
      label: A
      tile_center: [20, 20]     # mm, tile centre in the marker frame (default [0, 0] = marker centred)
      points: {101: [59, -21], 102: [-17, 59], 103: [62, 52]}      # mm, marker frame
    - {marker_id: 14, label: B, tile_center: [-18, -21], points: {104: [-60, 19], 105: [-52, -63], 106: [21, -58]}}
    - {marker_id: 15, label: C, tile_center: [-20, 20], points: {111: [-60, -20], 112: [-44, 60], 113: [21, 48]}}
```

Marker frame: origin = marker centre, x to the right, y towards the tile's TOP as printed (as the page is read). Checked at load time: marker ids unique and disjoint from every arm board and the table board; point ids positive,
unique across the tiles and disjoint from the table board's touch ids (1..4); the marker lies >= 3 mm inside the tile; every crosshair is
on the tile >= 6 mm from its edge, >= 8 mm from the marker's outer edge, and >= 15 mm from the other crosshairs of its tile. Statuses and
UI labels use "tile A", "tile B", "tile C" (touch point names A1 ... C3).

## Table board and touches (loop closure between the arms)

**Board.** Print `target_table.pdf` on A4 (landscape) at 100 %, verify the 50 mm bar and the 40 mm squares with calipers,
and enter the measured `square_length` / `marker_length` in the `table_board` entry (a 1 % scale error is a
10 mm depth error of the board at 1 m and shows up as a large touch mismatch). Glue it **flat and rigid** (3 mm
aluminium, acrylic or MDF; no wrinkles) on the work area, lying on the table, where the camera sees it and
**both arms can reach all four touch points**. It must not move during the whole calibration (all sessions of one
merged solve share it). The board is seen by the camera in every `capture` image in which it is detected; keep it
unoccluded in some of them (the arms parked away from it).

**Touch points.** Four fine diagonal crosshairs are printed on the inner chessboard corners near the board
corners, numbered 1..4 (`touch_points` in the YAML; coordinates in the board frame, origin = board centre, x right,
y down as read). The corner detections under the crosshairs are not used as camera observations.

**How to touch (operator decides how to move; nothing here moves the arm).**

1. Use the knife tip (the kinematic TCP `xeg32_tool_tip`). Move the arm yourself, at **low speed** (jog/teleop at the
   lowest setting or by hand-guiding if available), first above the point, then down in small steps; with the blade
   cover/guard fitted as for any manual knife work. Stop at first contact with the paper: the tip touches, it does
   not press (a sheet of paper under the tip that just starts to drag is a good criterion).
2. Hold the arm still. Tell the recorder which arm and point: `ros2 param set /dual_arm_calibration_recorder
   touch_arm left` (dual mode; in `single_*` modes the arm is implicit) and `... touch_point_id 2`.
3. `ros2 service call /dual_arm_calibration_recorder/capture_touch std_srvs/srv/Trigger`. The recorder refuses
   unless the active arms have been stationary for 2.0 s (same check as `capture`; and, if the table board is visible,
   unless two images >= 0.4 s apart show it still, see "Capture quality"), and stores `T_base_tcp` from TF plus the
   joint values in `touches.jsonl`, together with the contemporaneous `T_base_link6` and the `T_link6_tcp` transform
   in use (so a later TCP change can be re-based exactly, see "TCP consistency and changing the TCP"). Touches
   recorded before these keys existed still load (the session URDF then supplies `T_link6_tcp`).
4. Touch **every point with both arms**, at least 4 points each (>= 3 non-collinear are required), ideally 2-3
   times each with the **wrist in a different orientation** (rotate joints 4/6 so the tool tilts differently;
   this separates joint offsets and a TCP offset error from the base pose). Lift away between touches.

**TCP check.** Touch **one** board point with the same arm at **three or more** clearly different wrist orientations
(tilt the tool by 30-40 deg about **two different axes**; a pure roll about one axis, e.g. rotating only about the
tool axis, leaves the TCP component along that axis unobservable) and record each with `capture_touch` using the same
`touch_point_id`. A point qualifies only with enough tilt diversity, measured like the pivot method: with
`B_i = R_i - mean(R)` over that point's touches, `B` (3N x 3) must have rank 3 and `sigma_min(B)/sqrt(N) >= 0.2`
(`GateOptions.min_tilt_sigma`; roughly 4 touches at 25 deg or 3 at 40 deg about two axes; two touches never qualify).
A point that fails is **excluded** from the number and reported as "insufficient tilt diversity for a TCP check at
point N" with the weakest direction (unit vector in the tool frame, the one the tilts do not test); the report also
gives the weakest direction of the qualifying points. For the qualifying points the spread of the tip positions is
reported per arm as **TCP consistency** (mm RMS over those points, max pair). It is a **consistency indicator, not an
upper bound on the TCP error**: it contains the TCP offset error along the tested directions plus joint zero offsets,
backlash and touch repeatability (a Monte Carlo with 0.4 deg joint zero errors gives about 1-2.5 mm on its own), and
it is blind along the weakest tilt direction. A warning is issued above 1.5 mm. To *measure* the TCP, use the TCP
calibration, not this number.

**Touch-only right base.** Because both arms touch the same points (same point ids), the right base also follows
directly from the touches: a rigid Kabsch/Umeyama fit (no scale) between the left TCP positions (world) and the
right TCP positions (right base frame), per point averaged over repeats. It uses the **board crosshairs and the fit
stickers** (below) and never a check point. It needs >= 3 non-collinear common points (the solver refuses collinear
sets), >= 4 for the gate. It does not use the camera or the board pose, so it is the independent evidence for the right
base. The report prints it next to the written estimate and the difference (mm at the base origin, mm at a
representative reach point 0.4 m from the base towards the touched points, and degrees, also per world axis).
The quality of this base depends on the spread of the points. Its **1-sigma is predicted** (small Monte Carlo of the
Kabsch fit, Fitzpatrick/West) from the fit-point geometry and the **measured repeat scatter** of both arms (pooled per
axis over all points touched >= 2 times; `*` in the report marks an assumed 0.5 mm when an arm has no repeats). With the
four crosshairs alone (about 70 mm RMS from their centre) 1 mm of touch error becomes about 2.8 mm at 0.4 m reach
(`reach amplification`), with fit stickers at +-140 mm about 0.9 mm; a warning says so.

**Free stickers: fit points and check points.** Print or draw small crosshairs on stickers and stick them on the table
(or on a block) **anywhere both arms reach**; their positions are **not** known in any board frame and are never
entered anywhere. Declare their ids in `config/targets.yaml` (section `touch_stickers`, ids must be positive and
different from the board's 1..4; the defaults are 101..104 and 111, 112) and touch them with `capture_touch` exactly like
board points (`touch_point_id`, both arms, 2-3 wrist orientations or more):

| kind | ids (default) | use |
|---|---|---|
| `fit_points` | 101-104 | touched by both arms; used with the board crosshairs **only in the touch-only Kabsch base** (never in the joint solve). Put 3-4 of them about +-150 mm around the board, on the table plane: the wider spread is what makes the touch-only base (and the agreement test) meaningful |
| `check_points` | 111, 112 | touched by both arms and **never used in any fit** (not in the joint solve, not in Kabsch, not in the validation folds). Stick one on a ~40 mm block (>= 30 mm off the table plane) and one >= 150 mm from the board centre |

The recorder stores the kind with every touch (`touches.jsonl`, `kind`); `solve --fit-point-ids` / `--check-point-ids`
re-assign kinds afterwards. **Check-point error (TRE, held-out)** per check point is
`||T_world_left p_left - T_world_right(base) p_right||` with the **written** base and **nominal FK** (repeats averaged
per arm); the report prints each point, the RMS and the maximum, and the 'off the plane' / 'from the centre' distances
measured from the touched positions (plane and centre of the board crosshairs). What it demonstrates is the agreement
of the two arms at those points only.

**Trust gate for using arm_right's base in collision avoidance.** After the solve a PASS/FAIL block is printed and
written to the report JSON (`trust_gate`) and the YAML `notes`. Every check evaluates **values**; running a step is never
enough, and an in-sample number (fit residual, FRE) can never satisfy the gate on its own:

| check | default | option |
|---|---|---|
| `common_touch_points`: fit points (board crosshairs + fit stickers) touched by both arms | >= 4 | `--gate-min-points` |
| `touch_mismatch_rms`: worst of the fit residual (FRE, in-sample) and the **out-of-fit** loop closure (camera-only / cross-arm), nominal FK; needs an out-of-fit number | <= 1.5 mm | `--gate-touch-rms-mm` |
| `check_point_coverage`: check points touched by both arms; one >= 30 mm off the board plane; one >= 150 mm from the board centre (measured from the touched positions) | >= 2; 30 mm; 150 mm | `--gate-min-check-points`, `--gate-check-height-mm`, `--gate-check-dist-mm` |
| `check_point_error`: max check-point error (TRE, held-out) | <= 3.0 mm | `--gate-check-error-mm` |
| `touch_geometry`: predicted 1-sigma of the touch-only base: yaw / reach point (3D RMS) | <= 0.15 deg / <= 1.0 mm, else "touch spread insufficient: add fit stickers further apart" | `--gate-yaw-sigma-deg`, `--gate-reach-sigma-mm` |
| `joint_vs_touch`: written vs touch-only base, origin / reach point / each world rotation axis; limit = max(floor, 3 sigma) where sigma combines the predicted touch-only noise and the fit std of the written base | floors 1.0 mm / 0.1 deg, k = 3 | `--gate-joint-touch-mm`, `--gate-joint-touch-deg`, `--gate-agree-sigma-k` |
| `heldout_validation`: held-out reprojection RMS and (if there are held-out touches) held-out touch error, nominal FK, of the **written** variant (with / without offsets) | <= 1.0 px, <= 1.5 mm; fails if not run | `--gate-heldout-px`, `--gate-touch-rms-mm`, `--no-validate` makes it fail |
| `intrinsics_validated`: an `--intrinsics-file` was supplied, **or** the data would pull f by <= 0.3 % and c by <= 3 px | file, or 0.3 % / 3 px, else "calibrate the camera intrinsics" | `--intrinsics-file`, `--gate-intrinsics-f-pct`, `--gate-intrinsics-c-px` |

The exit code stays 0. The YAML `source` is `calibrated` only on PASS and `calibrated_unverified` otherwise (the
description package currently distinguishes only `measured` and `calibrated`; treat `calibrated_unverified` as not
usable for collision avoidance between the arms). The gate text always ends with the scope ("accuracy demonstrated at the
check points only") and `recommended_padding_mm`. `--base-from {joint,touches}` (default `joint`) selects which estimate is written
to the YAML (`arm_right_base` only; the camera pose always comes from the camera chain; `joint` means the written fit,
see "Which base is written"). **When the agreement check fails, the TCP consistency is good (below 1.5 mm) and the
touch geometry is adequate, the touch-only base is the one to prefer**: the arms agree with themselves, so the camera
chain is the suspect. Otherwise the arms' kinematics (joint offsets, TCP) are suspect and the touch-only base inherits
their error (it cannot model offsets): in the synthetic study with 0.4-0.56 deg joint offsets the touch-only base was
clearly *worse* (12.7 mm RMS nominal prediction error against 6.8 mm for the joint fit with offsets, TCP consistency
1-11 mm). The gate output states which case applies; if neither base is trustworthy, fix the arms/setup first.

**Safety padding.** `recommended_padding_mm = max(20, 2 x max check-point error + 10)` is printed by `solve` and
`check_points` and written into the YAML `notes`; set the MoveIt padding between the two arms to at least that. Without
check points only the 20 mm minimum is known. Do not reduce it because the synthetic numbers look better (circularity
note above), and a larger value is right if the cell has not been validated with a caliper near-miss check.

Safety: touching is the only phase that brings the tool into contact with something; keep one hand on the
stop, keep the other arm parked clear, and never let `capture_touch` stand in for a motion command (it only
reads TF and joints).

**What it gives.** The solver adds the board pose `T_world_board` (6 unknowns), the table-board corner
reprojection residuals and, per touch, 3 position residuals `T_world_base(arm) * T_base_tcp(origin) -
T_world_board * p_point` with sigma 1 mm (`--touch-sigma-mm`; converted to the pixel scale of the fit by the
nominal 0.3 px so that the two kinds of residual are weighted by their sigmas). Left data, right data and
touches now close a loop. The report states the touch mismatch **per arm and per point** and, **without using
the touches to determine what they test** (see Reading the report), the right base with and without touches.
Only the printed board crosshairs enter this joint solve; fit and check stickers do not.

### Capture quality (stillness, two-frame check, session info)

- The arms must have been **stationary for 2.0 s** (`stationary_window_s`, was 0.5 s) before `capture` / `capture_touch`.
- **Encoder stillness does not prove that the printed structure is still**: a board on a flexible bracket vibrates after
  the joints stopped. `capture` therefore also takes two images >= 0.4 s apart (`min_image_gap_s`), both after the
  stationary start, detects the targets in both and rejects the capture if the common corners moved coherently by more than
  0.2 px (`max_corner_motion_px`) or any single corner moved more than 1.0 px (`max_corner_single_px`). Detection noise
  (about 0.14 px per frame at 1920x1080) makes the *largest single-corner* displacement of a perfectly still board about 0.5 px
  over 20 corners, so the 0.2 px limit is applied to the **coherent motion** instead: a least-squares 2D similarity transform
  (rotation, uniform scale, translation; closed form, no RANSAC) is fitted from the earlier to the newer corners, and the
  largest displacement that transform produces at the corners is judged. Real sway moves all corners together, noise does not.
  The single-corner limit catches gross shaking or a mis-detected corner. The reply says `still (common 0.06 px, single max
  0.48 px over 20 corners)` or `board still moving (common 0.31 px > 0.20 px, single max 0.62 px over 20 corners)` /
  `board still moving (single corner 1.30 px > 1.00 px, common 0.12 px over 20 corners)` (call again after it settled); a target that cannot be
  compared (not detected in the earlier frame, < 4 common corners) is rejected too; `capture_touch` does the same for the
  table board **if it is visible** (and skips the image check otherwise: the touch itself needs no camera).
  `corner_check:=false` switches the check off (only for debugging). The accepted samples store `corner_motion_px` (the coherent value) and `corner_motion_max_px` (the single-corner maximum).
- **Session metadata**: the optional recorder parameters `ambient_temp_c`, `ambient_rh_pct`, `warmup_min` (minutes since the
  camera/stack were switched on) and `operator_note` are stored in `session.json` (`session_info`), shown by `status` and
  by `solve`: `ros2 param set /dual_arm_calibration_recorder ambient_temp_c 21.5` (or as launch/node parameters). Camera and arms
  drift while warming up: record after the warm-up and note the time.

## Camera intrinsics (measure them, do not trust the factory camera_info)

`solve` always computes what `--refine-intrinsics` **would** move (f and the principal point, with the priors) without
changing the result (`joint.intrinsics`: `delta_px`, `pull_f_frac`, `pull_c_px`). The gate check `intrinsics_validated` is
PASS only if an `--intrinsics-file` was supplied, or if the pull is <= 0.3 % in f and <= 3 px in c; otherwise it FAILs with
"calibrate the camera intrinsics". To calibrate:

1. A board of the table-board spec (default; any target of the targets file via parameter `intrinsics_target`): the operator
   holds a **second, rigidly mounted print of the table board** (flat plate, matt) in the camera view, or the arm holds it.
   Start the recorder (any mode; no arm is needed) and for every view call
   `ros2 service call /dual_arm_calibration_recorder/capture_intrinsics std_srvs/srv/Trigger`. It needs two sharp images
   >= 0.4 s apart (a moving board is refused: "board still moving ... (blurred / moving)"), >= 16 corners, a PnP below 2 px
   and a view that differs from the previous one (>= 15 px); the full image goes to `<session>/intrinsics/intr_NNNN.png`,
   the detections (same custom detector as everything else; the crosshair corners excluded) to `intrinsics.jsonl`.
2. Take **>= 25 views** (40 is better): tilt the board +-45 deg about both axes, vary the distance over the working range
   (e.g. 0.5-1.2 m), and put it into **all four image corners and along the image borders** (distortion is largest
   there). A small board gives a principal point good to only about 1.5-2 px (1 sigma) with 30 views: take more views
   and use the table board close to the camera.
3. `ros2 run rebel_dual_arm_calibration calibrate_intrinsics <session_dir> [...] --out intrinsics.yaml`: OpenCV plumb_bob
   calibration (k1 k2 p1 p2 k3; `--fix-k3`, `--chess-only` available) with the same detections. It prints the RMS (must be
   **< 0.5 px**), the standard deviations, a coverage map (corners per image cell, the four quadrants, the four border
   bands), the board depth range, and the difference to camera_info (delta f in %, delta c in px). With fewer than 25 views or
   RMS >= 0.5 px the file is **not** written (exit 2; `--force` overrides). Views with a per-view RMS above
   max(1 px, 3 x median) are dropped (listed).
4. Use it: `solve ... --intrinsics-file intrinsics.yaml` replaces camera_info (K and distortion) in **every** sample and
   table-board image (the image size must match), and the recorder parameter `intrinsics_file:=...` does so while recording
   (PnP check and stored K). The Monte Carlo budget then uses f +-0.1 % / c +-1 px instead of +-0.5 % / +-3 px.

## Data collection protocol

Precondition: camera intrinsics are good. The gate's `intrinsics_validated` check passes only with an
`--intrinsics-file` (see "Camera intrinsics" below) or when the data would pull f by <= 0.3 % and c by <= 3 px; the factory
`/camera/color/camera_info` is not assumed to be right. Camera must not move during the session; let the stack and camera
warm up first and note it (`warmup_min`). Mount the targets, then:

1. Start the robot description, drivers and the camera so that `/joint_states` (both arms), TF
   `arm_left_base_link -> arm_left_link6`, `arm_right_base_link -> arm_right_link6` and
   `camera_link -> camera_color_optical_frame`, plus `/camera/color/image_raw` and `camera_info` exist.
2. Start the recorder (below). Use a **new `output_dir` for every session**.
3. Move **one arm at a time**; park the other arm out of the way of the camera view of the moving
   arm. (A parked target that stays visible is recorded once and then skipped as a duplicate pose.)
4. For each arm record **30-40 distinct poses (the solver warns below 20 distinct poses)**:
   - spread positions over the whole carton/work area the camera will be used in, at different
     heights (e.g. 4x4 grid of positions at 2 heights);
   - tilt the target up to about 60 degrees away from facing the camera (not edge-on, not always
     face-on) and **rotate the wrist about different axes** (joints 4, 5 and 6, not only joint 6). The solver
     warns when the second principal rotation spread of the link6 orientations is below **30 degrees** (was 10),
     because a pure translation or a single rotation axis cannot separate the camera pose from the target mount
     offset, and joint offsets need diverse configurations to be observable at all;
   - keep the whole board in the image, in focus, without motion blur or glare on the print;
   - hold still (the recorder demands 2 s of encoder stillness and checks the corners of two images, see "Capture quality").
5. Call `capture` after each pose, read the reply (corners, PnP error). A PnP RMS above about
   0.7 px or fewer corners than usual indicates blur, glare, or partial occlusion; re-take it.
6. With the table board in place, touch the four board points, the **fit stickers** (spread +-150 mm) and the **check
   stickers** with both arms (previous section), then put the arms back to normal and run `solve`. Make sure `robot_description.urdf` is in the session directory (the recorder saves
   it once, see below); it is required for `--joint-offsets`.

## Record with today's single-arm driver (one arm at a time)

The dual-arm bring-up is not needed to calibrate: `launch/record.launch.py` has a `mode` argument.

| mode | stack it records | frames | joint prefix |
|---|---|---|---|
| `dual` (default) | `rebel_dual_table_description dual_table.launch.py` | `arm_left_base_link -> arm_left_link6`, `arm_right_base_link -> arm_right_link6` | `arm_left_` / `arm_right_` |
| `single_left` | single-arm stack, arm in front of the camera is the LEFT arm | `base_link -> link6` | none |
| `single_right` | single-arm stack, the arm in front of the camera is the RIGHT arm | `base_link -> link6` | none |

In a `single_*` mode only that arm is required in TF and `/joint_states` and only its target is recorded
(the other arm's board, if visible, is ignored). Record the two arms in two sessions, even with two different
bring-ups, **as long as the camera, its driver calibration and its mounting do not change in between**, and
solve them together:

```bash
ros2 launch rebel_dual_arm_calibration record.launch.py mode:=single_left  output_dir:=$HOME/dual_arm_calibration/left
# ... move the arm, call capture >= 20 times, stop; connect the other arm; the camera stays where it is
ros2 launch rebel_dual_arm_calibration record.launch.py mode:=single_right output_dir:=$HOME/dual_arm_calibration/right
ros2 run rebel_dual_arm_calibration solve $HOME/dual_arm_calibration/left $HOME/dual_arm_calibration/right \
  --out $HOME/dual_arm_calibration/dual_arm_calibration.yaml
```

Launch arguments: `mode`, `output_dir`, `targets_file`, `image_topic`, `camera_info_topic`, `joint_states_topic`,
`robot_description_file`, `intrinsics_file`, `intrinsics_target`, `ambient_temp_c`, `ambient_rh_pct`, `warmup_min`, `operator_note`. In `single_*` sessions the arm's touches are recorded the same way (`touch_point_id`; the arm is implicit);
both sessions then need the same table board and camera, and each saves its own URDF (identical kinematics required).
The launch file only starts the recorder; it never starts or moves a robot. The `capture`/`status` services
are the same as below (`status` lists the active arms).

## Operator command sequence (verified end to end in MOCK on 2026-10-08)

Evidence, logs and numbers: `scan_results/dual_arm_calibration_metrology_20261008/` (the previous run without the
metrology additions: `scan_results/dual_arm_calibration_e2e_20261007/`). Everything below ran in mock
(`hardware_protocol:=mock_hardware`, simulated scene camera, hidden ground truth) except where marked
**REAL (not executed)**. Use one terminal per long-lived command; source `install/setup.bash` in each.
Prepare once: print the table board (A4 landscape) and **stick 4 fit stickers (ids 101-104) about +-150 mm around it and 2 check
stickers (ids 111, 112: one on a ~40 mm block, one >= 150 mm from the board centre) anywhere both arms reach** (positions are
not measured); measure the printed boards in x and y (`square_length_y`, `marker_length_y`); let the camera and stack warm up.

```bash
# 0. scene camera only (REAL, not executed here; needs the Gemini 336). It USB-resets the camera first
#    (reset_scene_usb:=true by default); skip this step if cameras.launch.py is already running.
ros2 launch rebel_cameras cameras.launch.py scene:=true wrist:=false knife:=false
#    check: ros2 topic hz /camera/color/image_raw ; the recorder status line "image: ... age" must be < 1 s

# 0b. intrinsics (once per camera / lens setting; REAL, not executed): recorder in any mode, a second print of the table board held in
#     the camera view, 25-40 views over the whole image (corners, borders, tilt, distance), then
ros2 service call /dual_arm_calibration_recorder/capture_intrinsics std_srvs/srv/Trigger      # once per view
ros2 run rebel_dual_arm_calibration calibrate_intrinsics $HOME/dual_arm_calibration/intrinsics_session \
  --out $HOME/dual_arm_calibration/intrinsics.yaml                     # RMS must be < 0.5 px, >= 25 views, read the coverage map

# 1. ARM 1 (the one in front of the camera = LEFT), existing single-arm stack. REAL: hardware_protocol:=rebel
#    (mock run: hardware_protocol:=mock_hardware). robot_ip = the controller of the arm that is wired.
ros2 launch igus_rebel rebel.launch.py hardware_protocol:=rebel robot_ip:=192.168.3.11
ros2 launch igus_rebel_moveit_config move_group.launch.py hardware_protocol:=rebel use_gui:=true run_camera_pole:=false
#    (run_camera_pole:=false: the pole needs a hand-eye camera TF that does not exist yet; run_teleop:=true adds servo jog)

# 2. recorder for arm 1 (a NEW output_dir per session; targets_file:=<measured targets.yaml> if you measured the print;
#    intrinsics_file:=<intrinsics.yaml> if you calibrated the intrinsics, see "Camera intrinsics"; warmup_min:=30 ambient_temp_c:=22.5 ...)
ros2 launch rebel_dual_arm_calibration record.launch.py mode:=single_left output_dir:=$HOME/dual_arm_calibration/left
ros2 service call /dual_arm_calibration_recorder/status std_srvs/srv/Trigger     # all lines must be ok / "stationary: yes"
ros2 param set /dual_arm_calibration_recorder ambient_temp_c 22.5                 # session metadata (also ambient_rh_pct, warmup_min, operator_note)

# 3. move the arm to a pose (RViz / teleop), hold still >= 2 s, then:
ros2 service call /dual_arm_calibration_recorder/capture std_srvs/srv/Trigger    # repeat for 30-40 poses
#    READ THE REPLY. Good:  success=True, "cap N: left: 20 corners (4 markers, 4 chess), PnP 0.2 px, sample #K; table board: ..."
#    No arm sample even though success=True: message starts "NO ARM SAMPLE RECORDED (table-board image only ...)", e.g.
#    "left: board still moving (common 1.41 px > 0.20 px, single max 1.80 px over 20 corners), rejected": wait and call again.

# 4. touches (TCP xeg32_tool_tip on the printed crosshair 1..4 AND on the stickers 101..104 (fit) and 111, 112 (check),
#    arm still, low speed). Per touch:
ros2 param set /dual_arm_calibration_recorder touch_point_id 2
ros2 service call /dual_arm_calibration_recorder/capture_touch std_srvs/srv/Trigger
#    in single_* modes the arm is implicit (touch_arm not needed). Touch every point 2-3 times, wrist tilted >= 25 deg between repeats
#    (and one point 5x at the SAME orientation if you want a repeatability number).

# 5. stop recorder + stack (Ctrl-C). Connect ARM 2 (RIGHT) the same way, SAME camera, SAME table board and stickers, new output_dir:
ros2 launch igus_rebel rebel.launch.py hardware_protocol:=rebel robot_ip:=<controller of arm 2>
ros2 launch igus_rebel_moveit_config move_group.launch.py hardware_protocol:=rebel use_gui:=true run_camera_pole:=false
ros2 launch rebel_dual_arm_calibration record.launch.py mode:=single_right output_dir:=$HOME/dual_arm_calibration/right
#    then steps 3 and 4 again with the same touch point ids (board 1-4, fit 101-104, check 111-112).

# 6. solve (CPU heavy, ~20 s for 54 samples with validation + about 0.3 s per Monte Carlo draw; nothing is moved). The YAML and <yaml basename>_report.json
#    land next to --out.
ros2 run rebel_dual_arm_calibration solve $HOME/dual_arm_calibration/left $HOME/dual_arm_calibration/right \
  --out $HOME/dual_arm_calibration/dual_arm_calibration.yaml \
  --intrinsics-file $HOME/dual_arm_calibration/intrinsics.yaml \
  --uncertainty-mc 200 [--joint-offsets] [--compare $HOME/dual_arm_calibration/previous.yaml]
#    inspect: the printed report; "TRUST GATE ... PASS|FAIL" block with its 8 checks; "Check points"; "Uncertainty budget";
#             "Wrote ... (source: calibrated|calibrated_unverified)";
#             jq -r .trust_gate.summary $HOME/dual_arm_calibration/dual_arm_calibration_report.json
#    the padding to use between the arms: recommended_padding_mm in the gate block / YAML notes.

# 6b. any later time (after a power cycle / re-homing): both arms touch the check stickers again (short sessions), then
ros2 run rebel_dual_arm_calibration check_points $HOME/dual_arm_calibration/left_recheck $HOME/dual_arm_calibration/right_recheck \
  --calibration $HOME/dual_arm_calibration/dual_arm_calibration.yaml     # check-point error of the deployed calibration, no fitting

# 7. use it in the dual cell (the SAME file in both launches). REAL: hardware_protocol:=rebel (not executed; needs the rebuilt driver, see HANDOFF.md)
ros2 launch rebel_dual_table_description dual_table.launch.py hardware_protocol:=rebel \
  left_robot_ip:=192.168.3.11 right_robot_ip:=192.168.3.12 calibration_file:=$HOME/dual_arm_calibration/dual_arm_calibration.yaml
ros2 launch rebel_dual_table_moveit_config move_group.launch.py hardware_protocol:=rebel \
  calibration_file:=$HOME/dual_arm_calibration/dual_arm_calibration.yaml
ros2 run tf2_ros tf2_echo world arm_right_base_link        # equals arm_right_base of the YAML
ros2 run tf2_ros tf2_echo world camera_link                # equals camera_link of the YAML
```

Things to know when it does not work:

- `capture` replies are the only feedback; a refusal records nothing. Reasons seen: `arms not stationary yet (still for X s,
  need 2.00 s)` (wait, call again), `board still moving (common x px ...)` / `no earlier image ...` (the two-frame check: hold still,
  call again), `newest image is X s old (max 1.0 s)` (the camera is not delivering: check the camera
  launch / `ros2 topic hz`), `newest image predates the stationary window` (call again), `no image / camera_info received yet`,
  `TF base->tip is stale/unavailable`, `left: not detected`, `only N corners`, `PnP reprojection ... rejected`,
  `same pose as an existing sample, skipped`. Check counts with `status` (`samples left N right M`).
- The solver needs at least 4 fit points (board crosshairs + fit stickers) that BOTH arms touched (same ids) and >= 2 check
  points touched by both arms, else the gate FAILs. A gate FAIL still writes the
  YAML with `source: calibrated_unverified`; the dual launch only warns (rebel) and the description package must not be trusted
  for arm-to-arm collision avoidance with it.
- If the recorder restarts with the same `output_dir` it resumes (existing samples/touches kept). To discard a wrong touch,
  stop the recorder, delete that line of `touches.jsonl`, start it again (the touch counters are read at start).

## MOCK ONLY tools (no robot, no camera)

`scripts/mock_scene_camera.py`, `scripts/mock_pose_sequence.py`, `scripts/mock_eval_truth.py`, `rebel_dual_arm_calibration/mock_common.py`
and `config/mock_truth.yaml` run the procedure above without hardware. **They must never be used for a real calibration.**

- `mock_scene_camera.py` publishes `/camera/color/image_raw` (1920x1080 rgb8, rendered boards with distortion, blur, noise),
  `/camera/color/camera_info` and the static TF `camera_link -> camera_color_optical_frame`, from a hidden ground truth (camera pose,
  arm_right base, target mounts on link6, table board pose) and the live mock arm TF: `ros2 run rebel_dual_arm_calibration
  mock_scene_camera.py --ros-args -p arm:=left|right` (single-arm stack) or `-p frame_mode:=dual`. It publishes the image RELIABLE
  and needs the SHM-capable Fast DDS profile of `scan_results/dual_arm_calibration_e2e_20261007/tools/fastdds_shm.xml`
  (`FASTRTPS_DEFAULT_PROFILES_FILE`, camera and recorder): with default best-effort UDP about 2 of 5 6-MB frames are lost on this host.
- `mock_pose_sequence.py poses|touch --arm left|right` drives the mock JTC through collision-checked, board-visible poses and the touch
  points computed from the hidden truth, and calls the recorder services (touches through the real `ros2 param set` / `ros2 service call`
  CLI). `touch --points 1,2,3,4,101,102,103,104,111,112` touches the board crosshairs and the free stickers of the truth file
  (`config/mock_truth.yaml`, `touch_stickers`: fit 101-104 at about +-140 mm around the board, check 111 on a 40 mm block and 112
  ~215 mm out); `--touch-offset-mm 5,0,0` displaces the touched points (negative test); `--touch-noise-mm 0.4` adds a random contact error
  to every touch (the mock arm itself is exact, so without it the repeat scatter is ~0 and the predicted touch geometry error vanishes);
  `--repeat-same K` touches each point K times at one wrist orientation (repeatability). `poses` retries a capture that the two-frame
  check rejected ("board still moving"; the arm is still settling within the 2 mrad stationary tolerance).
- `mock_eval_truth.py <yaml>` compares a solved YAML with the hidden truth.

## Calibration web UI

```bash
ros2 launch rebel_dual_arm_calibration calibration_ui.launch.py mode:=dual output_dir:=~/dual_arm_calibration/s1
# open the URL the log prints: http://127.0.0.1:8095/?token=...   (new token every start; localhost only)
```
It starts the recorder too (`start_recorder:=false` attaches to a running one). Tabs: **0. TCP** (knife TCP pivot calibration,
the FIRST step of the suite: see "TCP calibration" below), **Setup** (session info, health
checklist), **Intrinsics** (optional, hand-held table board, coverage grid, `calibrate_intrinsics` job), **Record arm**
(live image with detected corners, pose counters, Capture), **Touches** (board points 1-4, FIT 101-104, CHECK 111/112 with
done marks per arm, Capture touch, approach assist), **Hover test** (approach assist accuracy test, below). Solve and the re-checks stay on the command line.

**Jog panel** (`ui/jog.py`): one bounded step per mouse click (Cartesian 0.1/0.5/1/5/10 mm, rotation 0.5/1/5 deg about the
TCP, joint 0.5/1/5 deg), planned with MoveIt `/compute_cartesian_path` (fraction must be 1.0) or interpolated and checked
with `/check_state_validity`, re-timed to <= 10 mm/s, 5 deg/s, 0.1 rad/s and 0.5 rad/s^2 (checked on the cubic Hermite
interpolation the JTC executes) and sent to the arm's trajectory controller. On REAL hardware the panel stays locked until
"Enable motion" is pressed and confirmed on that page (expires 10 min after the last step; a reload locks it). STOP / Esc
cancels; the arm stays busy until the controller result and a measured standstill, otherwise the jog goes to FAULT and
"Clear fault" needs a measured standstill. Every step reports the achieved vs requested motion. The knife tip has no
collision geometry: collision checking does not protect the blade. At a stretched (singular) pose Cartesian steps are
refused; bend the elbow with a joint step first.

*Motion safety rules of the jog and of "Move there"* (all in `ui/jog.py` / `ui/server.py`):
- **Invalid joint states fail closed**: a NaN / inf position or velocity, a missing joint or a state older than 0.5 s makes the
  arm's joint state invalid: no step / plan / move starts, the pre-dispatch check fails, standstill is false, the auto capture
  sees no joints.
- **Standstill** = all six joints fresh, max-min <= 0.5 mrad over **1.2 s**, drift |last - first| / window <= 0.5 mrad/s and
  (if `/joint_states` carries a velocity) |velocity| <= 0.03 rad/s (the real driver's velocity is a finite difference of
  0.01 deg encoder steps, so one step of flicker reads ~0.01 rad/s). The window is longer than the driver's 1.0 s STATUS-age
  ERROR cutoff on purpose: while the driver coasts / holds on stale STATUS it keeps publishing the FROZEN position with fresh
  stamps, and a window of <= 1.0 s could be filled by such a freeze; with 1.2 s the motion before the freeze (or the catch-up
  jump after it) is always inside the window. Settling after a finished goal waits up to 3 s for it, a STOP up to 9 s.
- **Confirmations are event-anchored** (stop confirmation, settling after a goal, Clear fault): `standstill(arm, since=anchor)`
  counts only joint_states samples received at or after the event (the begin of the STOP, the goal result, the fault entry -
  whichever is latest) and the whole 1.2 s window must consist of such samples without gaps (> 0.5 s) in between. A stationary
  history from before the event proves nothing: a feedback that freezes exactly when the motion starts would otherwise "confirm"
  a STOP at once. A confirmation also needs the hardware component and the controller to be `active` in a controller_manager
  answer that was received at least 1.0 s (the driver's STATUS-age ERROR cutoff) after the event, and fresh joint_states;
  otherwise the stop stays "stopping" (fault at the 9 s deadline) and Clear fault is refused with the reason.
- **Pre-dispatch standstill gate**: a step or move is only accepted (and again right before its goal is sent) when the arm is
  measured at standstill over the full 1.2 s window (not anchored: the arm must simply be still); otherwise it is refused with
  "the arm is moving (teleop active?): release the keys and wait - it is not measured at standstill". After a finished step the arm has just been confirmed still for 1.2 s, so the
  next step is accepted as soon as the 0.3 s minimum gap has passed; after any other motion (hand guiding, a fault, a reconnect)
  the effective minimum wait is the standstill window.
- **Clear fault** needs a measured standstill AND the controller's confirmation that the goal ended: a result that arrived as a
  normal response with status SUCCEEDED / CANCELED / ABORTED, a goal the controller explicitly rejected, or a goal that was
  never sent. A result future that completed with an exception, a missing response or an unknown status is NOT a
  confirmation (fail closed): the fault stays and says "the controller's answer could not be confirmed: press the e-stop if the
  arm moves; restart the trajectory controller or the driver (not just the UI) before moving again". Restarting only the UI would
  drop the goal / fault ownership without stopping anything. A goal that is accepted late (after STOP or a fault) is cancelled the
  moment it exists, and a STOP that finds the controller's acceptance already in but not yet recorded by the worker cancels it at once.
- **STOP is atomic against the send**: the final pre-dispatch check, the `send_goal_async` and the registration of the pending
  send happen under one lock that `request_stop` also takes. A STOP therefore either completes before the check (nothing is
  sent) or comes after the registered pending send and hooks "cancel when accepted"; the arm stays busy until a validated
  terminal result AND a standstill. STOP, cancel, fault handling and the supervision never depend on the real-hardware enable:
  an enable that expires (10 min) mid-motion stops nothing and never blocks a STOP. A move that would take longer than 300 s is
  refused at planning, so the enable that was valid when the move started cannot expire before the move ends.
- **Queued motion does not survive STOP**: plan / step / move requests are refused with 409 at HTTP ingress while the jog is
  busy / stopping / in fault or another request is queued; every request carries the STOP epoch it was ingressed under and is
  void if STOP (or a fault) happened since ("STOP was pressed after this request was sent").
- **Executed curve = validated path**: the dense cubic-Hermite samples of the final trajectory must stay within 0.002 rad
  (joint-space, about 1.8 mm TCP for one joint at a 0.9 m lever, at most 4.4 mm for all six) of the polyline that was
  collision-checked, and the re-timed points must lie on it; the joint-speed peak is computed analytically (no interior peak
  can hide between subsamples). In addition, for steps AND moves, `/check_state_validity` is called on dense samples of that
  final Hermite curve itself (consecutive samples <= 0.25 deg apart in every joint, at most 6000 calls, more -> refused), so a
  state that is only reachable on the curve between the re-timed points is checked too.
- **Supervision while a goal runs** (steps and moves): joint_states stale > 0.5 s or non-finite, measured joint speed (finite
  difference over >= 0.5 s: a 300 ms feedback freeze followed by a catch-up read 3x the true speed over 0.1 s) > 2 x the cap of
  the motion, or tracking error against the reference trajectory > 0.05 rad (moves and steps; with the current gains the steady
  lag is ~0.47 x velocity, ~0.045 rad at the 0.1 rad/s step cap), or **frozen feedback**: the reference commands some joint
  faster than 0.005 rad/s (one 0.01 deg count per ~35 ms) but no joint position changed for 0.25 s ("joint feedback frozen
  (driver STATUS stall?)": the driver repeats the last positions with fresh stamps while its STATUS stalls) -> the same stop path
  as the STOP button ("stopped by supervision: ...").
- **Start tolerance**: the arm may differ from the state a step / plan was made from by at most 4e-4 rad per joint (about 2.3
  encoder counts of 0.01 deg, ~0.36 mm at 0.9 m) at accept, at the pre-dispatch check and when a stored plan is executed;
  more means "the arm moved": plan again.
- **Plan vs scene**: EVERY move is re-checked with `/check_state_validity` at execute time (no age grace) on a coarse subset of
  the stored final Hermite curve samples (the ones `curve_check_states` produced at planning: about every 1 deg of joint travel,
  at most 300 calls), phase "checking" (the arm is busy, STOP works); an invalid state refuses the move ("the planning scene
  changed since planning").

**Auto capture** (Record arm tab, `ui/autocapture.py`): the toggle next to Capture takes a sample by itself whenever the
arm stands still at a pose that differs from the poses already handled, so you jog (or move by hand) and watch the counters
instead of pressing Capture each time. It never moves the robot and needs no "Enable motion". It runs in the UI backend
(about 2 Hz) and calls the same `capture` service as the button; the recorder still applies all its own checks (2 s
stationary, two-frame corner check, PnP, same-pose test).
- **When**: every joint of all active arms moved less than 2 mrad over the last 2.0 s (same limits as the recorder; the window
  restarts after every jog step, and the check is paused while a jog step moves, stops or is in FAULT), and the pose differs
  by at least the selected **min change** (1 / 2 / 5 deg, default 2) in some joint from **every** pose already handled in
  this run (captured, skipped, given up or unusable ones all count). A pose is handled once; returning to it does nothing.
- **Result per pose** (status line under the buttons, and the result line below it like a manual capture):
  `captured (n this run)` (an arm sample was stored); `same pose, skipped` (the recorder already has this pose);
  `retrying (n): ...` for timing refusals (board still moving, not stationary, no earlier image, image predates the
  stationary window or too old, cannot verify stillness), retried every 0.5 s and given up after 15 s from the first attempt
  (`gave up: <last reason>`, no more tries until the arm goes to a new pose); `board not usable here: <reason>` when the
  board is not detected / too few corners / PnP fails / no arm sample (one retry after 0.5 s); any other failure (service
  unavailable, TF stale, ...) is shown and retried every 2 s without marking the pose handled.
- **Ownership**: the run belongs to the page that started it. A second page cannot start or stop it ("auto capture is running
  on another page"). It stops when the owner stops it, leaves the Record arm tab or closes the page, when the owner page stops
  polling `/api/state` for 5 s, or when the recorder disappears (the status line says why). Manual Capture is disabled on the
  owner page while it runs, and the backend serialises the calls (a manual capture during an automatic one gets 409).
- API (same token / Origin checks as the other POST endpoints, `X-Page-Id` required): `POST /api/auto_capture/start
  {"min_change_deg": 1|2|5}`, `POST /api/auto_capture/stop`; `/api/state` carries `auto_capture: {on, owner_is_you, state,
  message, attempts, captured, skipped, gave_up, min_change_deg, events}` (`events`: the recent per-pose results for the
  result line).

**Suggested pose** (Record arm tab, `ui/suggest.py` + `pose_suggest.py`; single-arm modes only, the panel says so in dual mode):
after at least 6 samples of the arm, *Suggest pose* fits the arm's samples and searches for the next best poses (up to 5: a
different board tilt/spin, an empty image area, a different wrist configuration). IK and collision checks go through
MoveIt `/compute_ik` and `/check_state_validity` in a worker thread (the page keeps working; state `computing`). The live
image shows the predicted board outline of the selected suggestion in magenta ("suggested #k"); the card lists why it helps
(rotation spread before -> after, information gain, new image area, max joint change, distance, view angle).
- *Next* cycles the suggestions. *Send to RViz* publishes the joint goal (one `moveit_msgs/RobotState` on
  `/rviz/moveit/update_custom_goal_state`, nothing else) so the goal appears in the MotionPlanning panel: tick **Allow External
  Comm.** in the Context tab if nothing changes, then Plan and Execute there yourself.
- *Plan preview* asks the jog controller for a collision-checked joint-space plan (`plan_move`; it stays stored, see
  `move_plan` in the jog status). *Move there* executes exactly that stored plan (`execute_move`), only for the selected
  suggestion and while the plan is fresh. On REAL hardware it needs the jog's "Enable motion" (same as a jog step) and an
  extra confirm dialog ("Move the REAL arm to the suggested pose? Plan <id>, duration ~N s", with "The knife has no collision
  model: keep the blade covered and the path clear."). OK sends exactly the plan id that was shown when the dialog opened (the
  server refuses any other id). STOP / Esc stops a move like a jog step.
  Auto capture (if ON) takes the pose when the arm stands still after the move.
- A new sample makes the suggestions stale ("based on N samples, session now has M"); press *Suggest pose* again. Selecting
  another suggestion drops the stored plan.
- The session directory comes from the recorder status, else from the UI node parameter `session_dir` (the launch passes
  `output_dir`). API (token / Origin checks as above; plan and move need `X-Page-Id`): `POST /api/suggest/compute {arm?}`,
  `/select {index}`, `/send_rviz {index}`, `/plan {index}`, `/move {plan_id}`; `/api/state` carries `suggest: {state:
  idle|computing|ready|error, message, based_on_samples, session_samples, stale, items: [{rank, description, gains, centre_px,
  outline_px, q_deg}], selected, stats, plan}`; `/api/image.jpg?...&suggest=1` draws the outline. Only `/move` can move the arm.

**Approach assist / hover** (Touches tab card "Approach assist" and tab **5. Hover test**; `ui/hover_ui.py` + `hover.py`; single-arm
modes only, the card says so in dual mode). It predicts where the printed touch marks (table board crosshairs 1-4, the crosshairs of the
hover tiles A / B / C: 101-103, 104-106, 111-113; see "Hover tiles") are in the arm base frame from the arm's own calibration fit, so the
knife tip can be brought to a mark by teleop instead of by blind jogging.
- *Detect points*: the newest ~5 camera frames go through `hover.detect_targets`, the session samples of the arm through
  `pose_suggest.fit_arm` (needs >= 6 samples), `hover.to_base` gives each mark's position / normal in the base frame. Marks that
  were not seen are greyed with the reason. The live image (Touches and Hover test tabs, `/api/image.jpg?...&hover=1&sel=<id>`) draws the
  predicted crosshairs with id and name; **click within 25 displayed pixels of a crosshair (or on a row) to select a point**
  (selected = yellow in the image and in the list). Motions are confirmed by a fresh observation (below), not by the prediction's age.
- *Plan hover*: for the selected mark, `hover.hover_tcp_poses` gives TCP poses 60 mm above the paper (tool z = -normal, spun in
  30 deg steps). The IK is asked for the **flange** (`link6`): `T_base_link6 = T_base_tcp * inv(T_link6_tcp)` with the fixed
  `link6 -> xeg32_tool_tip` transform of the robot model (`JogController.link6_to_tcp`), through the existing `/compute_ik` bridge
  of the suggested-pose panel, seeded with the current joints (MoveIt's `ik_link_name` of the rigidly attached TCP link was not
  used: not verified without a ROS graph). Every solution is checked with the model's own FK (TCP within 0.5 mm / 0.5 deg),
  the one with the smallest max joint change wins, more than 120 deg is refused ("jog the arm closer first"), and the best
  collision-free one (`/check_state_validity`) is planned with `jog.plan_move` (preview in RViz). *Move* executes exactly that plan
  (`execute_move`; on REAL hardware it needs `allow_real_move:=true`, the page's Enable motion and a confirm dialog bound to the plan id,
  like "Move there"). *Send to RViz* publishes the same joint goal as one `moveit_msgs/RobotState` and nothing else (also the way to
  go on REAL hardware without `allow_real_move`: the prepare reply says so).
- *Descend to N mm* (N = 10 / 20 / 30, default 20, selector "Hover height"): `JogController.descend` moves the TCP in a
  **straight line along -n** from where it is to N mm above the predicted point at constant orientation. Refused (409) if the TCP is more
  than 5 mm off the line through the mark along the normal, its z axis is more than 5 deg off -n, the remaining distance is < 2 mm
  or > 70 mm ("go to the pre-hover pose first (Plan hover -> Move)"). The target is never closer than 10 mm to the predicted surface
  (choices 10 / 20 / 30 mm, checked in code). It is exactly a Cartesian jog step with a target pose: `/compute_cartesian_path`
  (fraction must be 1.0, 0.5 mm waypoints), END / PATH deviation checks, 10 mm/s, 5 deg/s, 0.1 rad/s, 0.5 rad/s^2 on the Hermite curve,
  executed-curve validity check, MotionSupervisor, atomic dispatch against STOP, event-anchored settle, STOP / fault semantics, the
  standstill gate and the ingress / epoch gate. **Gating (safety review): on REAL hardware a descend needs `allow_real_move:=true`
  exactly like "Move there"** (`REAL_MOVE_DISABLED_MSG` otherwise; the page's Enable motion is needed as for a step; mock is unaffected).
  The target comes from a camera prediction, not from a bounded step. Without the switch the Descend button is disabled with the reason
  and the operator fine-tunes with teleop from the pre-hover pose instead (`jog.status.allow_real_move` tells the page).
- **Prediction correction (opt-in stopgap)**: `hover_correction_left_mm` / `hover_correction_right_mm` (launch args, `"x,y,z"` or
  `"[x, y, z]"`, mm in that arm's BASE frame, default off) are added to every predicted mark position at the one place it is produced
  (`HoverManager._detect`; stored and fresh detections alike, so the 2 mm moved check is unaffected; the normal is unchanged). Prepare,
  Descend, Record offset, the Touches log and the row positions all use the corrected point. Startup refuses non-finite components or
  |v| > 20 mm and logs a WARNING per non-zero arm; the page shows "Prediction correction ACTIVE (left): ... mm (base frame)" while
  non-zero. Every `hover_log.jsonl` / `hover_test.jsonl` record carries `correction_mm` and `predicted_raw_p_base_m` next to
  `predicted_p_base_m` (corrected); `offset_mm` stays touched - corrected prediction. The image overlay still draws the detected mark.
- **Pre-hover interlock**: a descend is only accepted if the LAST motion of the arm was the successfully finished hover move
  (Plan hover -> Move) to this same point, the CURRENT measured TCP is within 2 mm / 2 deg of that plan's pre-hover TCP pose, and no
  STOP / fault happened since. Any other motion (a jog step, another descend ...), a stopped / failed move, STOP, a fault, a new
  detection or a new plan clears it. So after a descend the next one needs Plan hover -> Move again.
- **Fresh observation**: *Plan hover* re-detects the point on fresh camera frames (only frames received within the last second, >= 3 of
  the newest 5) and refuses unless it agrees with the stored prediction within 2.0 mm and 2 deg of the surface normal ("the board
  moved or the detection changed: Detect points again"); the plan is aimed at the fresh prediction. **At the pre-hover pose the arm
  hides part of the table board (and often a whole hover tile), so the descend does NOT demand a full re-detection**: it runs
  `hover.unmoved` instead, which projects the VISIBLE corners of >= 3 fresh frames with the STORED board / marker pose (rebuilt from the
  compute's TargetPoints): table board >= 8 visible corners over >= 2 rows and columns, median reprojection residual <= 1.0 px and 90th
  percentile <= 2.0 px in every frame; hover tile (each tile on its own, against its own stored marker pose): all 4 marker corners, median <= 1.5 px. Pass: the paper has not moved and the descend
  is aimed at the STORED prediction. Fail: "the board moved or the detection changed: Detect points again"; too few visible corners / the
  tile hidden: "the board is hidden by the arm: cannot confirm it has not moved - use teleop from here, or Detect points with the arm
  away". *Plan hover* falls back to the same test if its full detection fails (not if it finds the point moved). One pixel at the board
  is distance / fx (0.9 m at fx 1400 px = 0.65 mm; 1.3 m at 1300 px = 1.0 mm), so the limits mean "no in-plane motion beyond about
  1-2 mm". Limits of the test: motion along the optical axis (2 mm of depth at a corner 0.3 m off the axis at 1.3 m is ~0.5 px) and a
  small out-of-plane tilt are barely visible to it; a corner next to the arm's edge can be mis-detected (a blank-rectangle occluder over
  70 % of the board gave a 90th percentile of 2.4 px with a median of 0.5 px), which refuses with the "moved" text on the safe side.
  Predictions older than 120 s are not refused by age any more (a 60 s move would expire them all): this fresh observation is the proof.
  Record offset is only a comparison and has no check.
- **Fine-tuning is done with MoveIt Servo teleop** (`rebel_servo_teleop_keyboard`, which publishes to the same
  `rebel_arm_trajectory_controller/joint_trajectory` the UI's action goes to) or the jog panel. After the hover pose the UI says:
  "Fine-tune with teleop (or the jog panel), release the keys, then Capture touch / Record offset." The UI never moves while
  Servo is commanding: every step / descend / move is only accepted and dispatched when the arm is measured at standstill and the
  refusal says "the arm is moving (teleop active?): release the keys and wait". What happens if teleop starts while one of our goals
  runs (read from the JTC 2.54 source): a command on the controller's topic does NOT cancel the active action goal, it silently
  replaces the trajectory; our goal then ends by the controller's tolerances (ABORTED "Aborted due to path tolerance violation",
  shown with the controller's own text) or, usually first, by our own supervision (tracking error / speed -> STOP, message ends
  "teleop active? ...: release the keys"). Only a NEW ACTION GOAL makes the controller cancel ours (CANCELED, INVALID_GOAL,
  "Current goal cancelled due to new incoming action."): only that exact controller text is labelled "stopped: the trajectory was
  replaced by another command". All of them go through the same stop confirmation (result + measured standstill; a teleop that keeps
  driving the arm still ends in the usual "stop not confirmed" fault after 9 s). Planning (`plan_move`) moves nothing and has no standstill gate; the plan can only be
  executed from the pose it was made at.
- **Predicted vs touched**: after *Capture touch* of a mark that has a prediction the reply and the row show
  `predicted vs touched: 2.3 mm lateral (x +1.9, y -1.2, above paper +0.1; 3D ...)` (`hover.compare`: x / y in the paper plane, "above paper"
  along the normal) and one line goes to `<session>/hover_log.jsonl` (`time, time_iso, kind, arm, point_id, name, source,
  predicted_p_base_m, touched_p_base_m, offset_mm, calibration {n_samples, arm}, quality, prediction_age_s`); the TCP is the one the
  recorder stored in `touches.jsonl`. **Hover test** tab: jog / teleop the tip onto a mark, arm still, press *Record offset* (same
  comparison with the CURRENT TCP, no recorder touch) -> `<session>/hover_test.jsonl`; the card keeps a table of the last offset, mean and
  max lateral per point and overall (read back from the file after a restart).
- API (token / Origin checks as above; prepare / move / descend need `X-Page-Id` and pass the motion ingress gate + STOP epoch):
  `POST /api/hover/compute {arm?}`, `/prepare {point_id, stage: "prehover"}`, `/send_rviz {point_id}`, `/move {plan_id}`,
  `/descend {point_id, height_mm: 10|20|30}`, `/record_offset {point_id}`; `/api/state` carries `hover: {supported, state:
  idle|computing|ready|error, message, arm, calib_samples, points: [{id, name, source, detected, reason, quality, pred_base_mm, uv,
  last_offset}], selected, prepared, plan, reached_point, interlock, descended, test_summary, heights_mm}`; the `/api/touch` reply gains
  `hover_offset`. After a failed detection every row's `reason` is the failure text and the old predictions are gone; rows show only
  the offsets of the current detection (older ones stay in the Hover test table). The Touches grid also lists hover tile points the
  targets file defines but the recorder's touch list lacks (kind `hover`, one group per tile "tile B: 105, 106", not capturable);
  every list item that is a tile point carries `tile` ("tile A") and `tile_name` ("A1"). Both hover cards group their rows by source
  ("table: 1, 2, 3, 4", "tile A: 101, 102, 103") and show that source's detection status under the header (`hover.status[source]`, e.g.
  "tile B: marker 14 not found (0 of 5 frames)"); each tile has its own IPPE pose and gates, so one hidden tile does not affect the others.

## Wrist camera hand-eye (eye-in-hand, "6. Wrist camera" tab)

Calibrates `T_link6_optical` (link6 -> the colour optical frame of the Gemini 305 on the knife holder) from ~15-20
**automatic** views of the fixed A4 table board. Single-arm modes only (`mode:=single_left|single_right`). Nothing is
written to the URDF; the result is a YAML (`wrist_handeye/v1`), a xacro snippet (text) and an optional static TF for a live check.

```bash
ros2 launch rebel_dual_arm_calibration calibration_ui.launch.py mode:=single_left start_recorder:=false \
  output_dir:=~/dual_arm_calibration/wrist_$(date +%Y%m%d) targets_file:=~/dual_arm_calibration/targets_measured.yaml \
  wrist_image_topic:=/arm_left/wrist_camera/color/image_raw wrist_camera_info_topic:=/arm_left/wrist_camera/color/camera_info \
  wrist_camera_serial:=CV27561000GA allow_real_move:=true       # allow_real_move only for the supervised real run
# result files: <output_dir>/wrist_handeye/ (wrist_samples.jsonl, images/, scene_board.json, wrist_handeye_result.yaml, ...)
```
The scene camera (`image_topic`, default `/camera/color/image_raw`) and `move_group` (with `run_camera_pole:=true`) must run.
After a source change `colcon build --packages-select rebel_dual_arm_calibration` is needed (the installed copy is what runs).

**Buttons in order.** (1) *Locate board*: table board in the scene camera -> `T_base_board` (still check over 5 frames,
works with the board near the image border; needs >= 12 corners). (2) *Add safety scene* (below). (3) Jog by hand (jog panel) until the
whole board is visible in the wrist image, *Use as seed* (needs >= 16 corners, board fully inside the image, measured standstill).
(4) *Generate views*. (5) tick the confirmation, *Enable motion* (real hardware), *Start auto sequence*; *STOP sequence* / the global
STOP / Esc stop it. (6) *Solve*, *Write result*, *Publish TF (live check)*. *Capture here* records one sample at the current pose.

**Safety scene** (`calib_*` boxes in `base_link`, via `/apply_planning_scene`; idempotent; removed with the button and on UI shutdown):
`calib_table_slab` is a HORIZONTAL box (the real table is horizontal even if the board stands tilted): top = highest of the 4 board
corners + 50 mm, 1.2 x 1.2 m centred on the board centre in x/y, 0.10 m thick; if its top is within 50 mm of the base bottom
the footprint is clipped to stay 50 mm (horizontally) away from the robot base footprint (radius 0.12 m), and the request is
refused if the board would no longer fit, if the board normal is > 50 deg off vertical or the top is above z = 0.30 m. `calib_board_stand`
= the 4 board corners + 0.05 m in plane + 0.05 m padding, from the slab top up to (highest corner + 0.03 m) + 0.05 m.
`calib_pole_pad` / `calib_head_pad` = `camera_pole` / `camera_head` from `/get_planning_scene` inflated by 50 mm on every side; the scene is
REFUSED if either is missing. After adding, the current robot state is checked with `/check_state_validity`; contact with a `calib_*`
object removes the scene again. All IK, planning and the executed-curve re-check run against this scene; the four objects must
be present in the live planning scene before the views are generated and before EVERY move, otherwise the run stops.

**Views.** Rigid rotations of the seed link6 pose: orbit about axes through the board centre IN the board plane (tilt 10/20/30 deg, 8
azimuths; the board's real normal is used), +-15..45 deg about the camera->board-centre line, range 0.8/1.0/1.2 x the seed
distance (camera-board >= 0.25 m, <= 0.65 m, board seen <= 65 deg from its normal). The board centre keeps its place in the camera
image whatever the mount is, so the orbit is robust to a wrong mount guess. Each candidate must project all board corners/edge
mid-points inside the image with a 10 % margin using the current mount estimate: after the first samples the interim solve; before
that the seed view + scene-camera board (one-shot estimate); else the rough CAD guess (front face of `gemini_305_body.stl`, optical +Z
along its normal, roll set with `wrist_cad_roll_deg`; ASSUMPTION, expect cm / several deg). Then IK (`avoid_collisions`, seeded from the
seed joints), joint jump <= 90 deg, `/check_state_validity`. 18 views max, diversity order, executed nearest-first. The panel shows
accepted / rejected counts by reason.

**Sequence safety rules.** Every move is `jog.plan_move` + `jog.execute_move` ("Move there": 0.04 rad/s, 0.25 rad/s^2, 40 mm/s,
MotionSupervisor, executed-curve collision re-check, STOP epoch, atomic dispatch). On REAL hardware the sequence is refused unless
the UI was launched with `allow_real_move:=true` AND this page holds the jog's real-hardware "Enable motion" AND the start request
carries the per-run confirmation (the page clears the tick after every start). Never a move without a valid plan. The run ends on: STOP /
epoch change / fault, a move that is not a clean "done", a lost scene object, 3 consecutive failed views (plan or capture), or a page
that stopped polling for 20 s. After a move: measured standstill, 1.5 s settle, 3 frames captured after that instant (header stamps), board >= 16
common corners, corner stillness (0.4 px coherent / 1.5 px single), identical TF at the three frame stamps (0.3 mm / 0.05 deg),
sample PnP < 2 px; otherwise the view is skipped. After every sample from #4 on, the interim solve updates the in-view prediction
and the remaining views are re-checked.

**Solver** (`eye_in_hand.py`, pure): per-sample PnP (gross corner outliers trimmed) -> `cv2.calibrateHandEye` (Tsai, Park, Daniilidis; the one with
the smallest board scatter, IPPE flip resolved against the model) -> joint nonlinear least squares over X and `T_base_board` (12 parameters,
intrinsics fixed, soft-L1, marker corners weight 1/2). **The refinement is only trusted if it stays within 5 mm / 2 deg of the closed-form
(AX = XB) result**: the ReBeL's absolute positioning error (a few mm, ~4.5 mm board scatter on the real cell) cannot be explained by X, and
on the real data of 2026-10-13 the refinement moved X by 37 mm / 49 deg for -1 px. Otherwise it is REJECTED, the closed-form X/B are kept and a
warning is shown (same in every leave-one-out fold and in the interim solve). Reports: reprojection RMS / max (informational on this arm),
board pose scatter across the samples, leave-one-out held-out board position, rotation span / distinct axes / planarity, agreement of the three
closed-form methods and the board pose of the solve vs the scene camera. **Gate:**

| check | PASS | FAIL |
|---|---|---|
| absolute reprojection RMS | informational | > 12 px |
| board consistency (RMS over samples) | <= 6 mm and <= 1 deg | > 10 mm or > 3 deg |
| held-out (leave-one-out) board position | <= 6 mm | > 12 mm |
| closed-form methods agree | <= 5 mm / 2 deg (else WARN) | - |
| scene-camera cross-check | <= 10 mm / 5 deg | > 25 mm or > 15 deg |
| samples / rotation span / distinct axes | >= 10 / >= 30 deg / >= 3 | < 6 / < 15 deg / < 2 |

Anything between PASS and FAIL is WARN. **Output frame:** the camera driver publishes `wrist_camera_link -> ... -> wrist_camera_color_optical_frame`
(identity on the real camera), so the result is attached as the joint **link6 -> wrist_camera_link** = `X inv(T_wrist_camera_link_optical)` (the
transform is looked up from TF; if that fails nothing is attached: no xacro snippet, no TF). "Publish TF" broadcasts only that joint; the optical
frame never gets a second parent. The YAML also keeps `T_link6_optical`. Offline: `python3 -m rebel_dual_arm_calibration.eye_in_hand_cli <wrist_handeye dir>
[--camera-link-optical x y z r p y]` (without the option the joint/xacro are not written). Real-data regression fixture:
`test/data/wrist_real_20261013/`.

**Mock.** `scripts/mock_wrist_camera.py` (MOCK ONLY; refuses unless every hardware component is `mock_components/GenericSystem`) renders the
table board into `/arm_left/wrist_camera/...` from the live mock arm TF and a hidden mount. `POST /api/wrist/mock_board` (button "Mock board") sets a
synthetic `T_base_board` and is refused on anything that is not mock hardware. Evidence: `scan_results/wrist_handeye_ui_20261013/`.

## TCP calibration

Tab **0. TCP** of the calibration UI finds the knife tip `r` (the origin xyz of the URDF joint `igus_tool0_to_xeg32_tool_tip`)
by a pivot calibration, so that every later step (touches, hover, cutting) uses the right TCP. It replaces the Tkinter tool
`rebel_tcp_gui` (same maths, now in `tcp_pivot.py`, with held-out validation and a verdict) and is the first thing to do after the
knife was remounted or after a crash. Single-arm modes only (`mode:=single_left|single_right`); in dual mode the tab says so.

**The buttons of the tab never move the robot and never write the xacro.** They only read the joint states (through the jog
controller's history), TF and the xacro source file. The jog panel is also shown on this tab (right column) for positioning,
with its usual enable rules; the Servo teleop keyboard is the recommended way.
- **Method**: you position the arm with the MoveIt Servo teleop keyboard (`ros2 run igus_rebel_moveit_config rebel_servo_teleop_keyboard`),
  **lift the knife tip out of a fixed shallow countersink and RE-SEAT it** between poses (do not rotate about the tip), release the
  keys and capture. Ultrasonic OFF. Fixture: hard, shallow (~2 mm deep) 90 deg countersink, clamped at working height in the
  touch region.
- **Checklist** (server side; every capture is refused until all four are ticked, and the refusal lists ALL unticked items):
  ultrasonic power off; fixture clamped; only this arm connected and the other parked; hand on the e-stop. The ticks belong to the
  PAGE (`X-Page-Id`): a reload or another browser tab starts with nothing ticked, a UI restart clears all of them, and a page's
  ticks expire after 30 min without a tick or a capture (a capture renews them). `/checklist` and `/capture` need `X-Page-Id`.
- **Capture FIT / VALIDATE pose**: each capture stores `T_base_igus_tool0` and `T_base_link6` from TF (both must be <= 0.5 s old;
  prefixed frames `arm_<side>_...` in dual stacks), the joints and the time. Refused with a text if the arm is not stationary
  (every joint within 2 mrad over the last 2.0 s: the jog's joint history, `joint_window`), the jog is busy / in fault, TF is
  stale or missing, or the tool rotation is less than 15 deg from an existing pose ("too similar to pose #k ... re-seat").
  The Capture buttons are disabled with the reason written next to them (checklist, arm moving, TF not fresh, jog busy, too similar
  to pose #k, no session directory). FIT poses go into the fit, VALIDATE poses are held out and only measure the error (`tcp_pivot.solve_pivot`).
- **Storage**: `<session_dir>/tcp_pivot.jsonl`, one `PivotSample.to_json_dict()` per line, loaded again when the UI restarts
  (the session directory is the recorder's own, else the `session_dir` parameter, as for the other panels). **Delete** removes one pose
  (the file is rewritten; the previous content is kept as `tcp_pivot.jsonl.bak`); lines that cannot be read stay in the file and are
  reported as a warning. Poses are never deleted automatically.
- **Live guidance**: arm still / TF fresh, the knife axis (direction of `r`) from vertical, tilt and roll about the knife axis relative
  to the reference pose (the fit pose closest to vertical), the nearest stored pose (and a warning that a capture here would be
  refused), the `tcp_pivot.coverage()` suggestion ("Next: ...") and the `POSE_PLAN` as a checklist (11 FIT poses: vertical, +-30 about
  tool x and y, four diagonals, two rolls of +-90 deg; 4 VALIDATE poses) with a tick for every direction already covered.
- **Solve** (`tcp_pivot.solve_pivot(samples, current_r=<URDF r>)`): verdict GOOD / PASS / FAIL / INSUFFICIENT with the reasons, `r` in mm
  and the delta to the current URDF (`TF igus_tool0 -> xeg32_tool_tip`, i.e. the live `robot_description`; the xacro source is the fallback
  and a mismatch between the two is reported), 1-sigma of `r` along tool x, y, z and along its principal directions with the weakest
  direction, fit residual and leave-one-out error per pose (flagged outliers are highlighted with "Delete this pose and re-capture"),
  held-out errors of the VALIDATE poses. Poses are numbered "#N" everywhere, 1-based by their position in the pose list (fit and
  validate together), in the table and in the solver's reasons (`tcp_pivot.solve_pivot`). The result goes stale (and cannot be
  written) as soon as the poses change: the page then shows only a "Stale" line, with no per-pose numbers, outliers, reasons or patch.
- **Mechanical sanity check**: enter the calipered distance flange face -> blade tip along the flange axis. It is compared with
  `52.7 mm + r_z` of the solved `r` (before a solve: of the current URDF `r`) through `tcp_pivot.mechanical_check`. The current URDF
  implies 183.4 mm; an old pivot data set suggested about 168 mm, so a ~15 mm z question is open: do not apply a new `r` that the
  caliper contradicts.
- **Output**: *Write result* (GOOD / PASS only) writes `<session_dir>/tcp_result_<arm>.yaml` (`tcp_pivot.write_result_yaml`; an earlier
  file is kept as `.bak`; `urdf_hash` = first 16 hex of the sha256 of the xacro source text) and the page shows
  `tcp_pivot.xacro_patch(<xacro text>, r)` as a diff. The xacro path is the UI node parameter `tcp_xacro_path` (launch argument
  `tcp_xacro_path:=`, default `/home/farhan/igus_rebel_ros2/src/igus_rebel_description/urdf/igus_rebel_xeg32_wrapper.urdf.xacro`); it is
  only READ. **You apply the diff by hand**: review it, edit the xacro (or `patch -p1` from the workspace root), commit, rebuild
  `colcon build --symlink-install --packages-select igus_rebel_description igus_rebel_moveit_config`, restart the robot stack and the UI.
  After a TCP change, touches recorded with the old TCP must be re-based (`solve --tcp-file ...`) or re-taken.
- API (same token / Origin checks as above; none needs `X-Page-Id`; none moves anything): `POST /api/tcp/checklist {items: {key: bool}}`
  (keys `ultrasonic_off`, `fixture`, `single_arm`, `estop`), `/capture {role: "fit"|"validate", note?}`, `/delete {index}`, `/solve`,
  `/mechanical {length_mm}`, `/write`. Refusals are HTTP 400 (bad input, dual mode) / 409 (checklist, stationary, TF, duplicate, no solve,
  verdict) / 503 (no session directory). `/api/state` carries `tcp: {supported, arm, checklist, samples, n_fit, n_validate, live,
  coverage: {suggestion, plan: [{id, role, label, text, covered}]}, current_r_mm, result, mechanical, written, diff, apply_steps,
  rebase_warning, warnings, thresholds}`; every action reply also carries the fresh `tcp` block.
- Tests: `test/test_ui_tcp.py` (manager with a fake jog / TF and the real `tcp_pivot` on synthetic poses, plus the endpoints).

## Run the recorder

```bash
colcon build --packages-select rebel_dual_arm_calibration && source install/setup.bash
export ROS_DOMAIN_ID=<yours>             # never the real robot's domain unless you intend to
ros2 run rebel_dual_arm_calibration recorder_node --ros-args \
  -p output_dir:=$HOME/dual_arm_calibration/session_01 \
  -p targets_file:=<path to your measured targets.yaml>

ros2 service call /dual_arm_calibration_recorder/status  std_srvs/srv/Trigger
ros2 service call /dual_arm_calibration_recorder/capture std_srvs/srv/Trigger
ros2 param set /dual_arm_calibration_recorder touch_point_id 2      # then, with the tip on point 2 (or sticker 101/111) and the arm still:
ros2 service call /dual_arm_calibration_recorder/capture_touch std_srvs/srv/Trigger
ros2 service call /dual_arm_calibration_recorder/capture_intrinsics std_srvs/srv/Trigger   # intrinsics board view, see "Camera intrinsics"
```

**Robot description.** For joint offsets the solver needs forward kinematics. The recorder saves the URDF **once**
as `robot_description.urdf` in the session directory, from (first that works) the parameter
`robot_description_file`, the transient-local topic `robot_description` or the `robot_description` parameter of
`/robot_state_publisher` (parameter `robot_description_node`). `status` says whether it was saved. The frames
`<arm>_base_frame`, `<arm>_tip_frame` and the TCP frame (`left_tcp_frame` / `right_tcp_frame`; default derived from
the tip frame: `arm_left_xeg32_tool_tip`, `arm_right_xeg32_tool_tip`, in `single_*` modes `xeg32_tool_tip`) are
stored in `session.json`; the FK reproduces the TF `base -> link6` and `base -> tool tip` to better than 1e-9
(checked against PyKDL and against a running robot_state_publisher).

`capture` also stores the corners of the table board (as a separate observation, not tied to an arm, in
`table_obs.jsonl`; at most 60 per session) when it is detected with at least 16 points. `capture` refuses with a
message if: no image or camera_info, the newest image is older than 1 s,
joint states are missing or stale, any arm joint moved more than 2 mrad / 0.01 rad/s in the last
2.0 s, the image predates the stationary window, there is no earlier image for the two-frame check, or a TF lookup is
missing or older than 1 s. It
records one sample per arm whose target is detected with enough corners (8 for ChArUco, 4 for ArUco) and a PnP
reprojection RMS below 2 px, and skips a sample whose joint configuration is within 0.01 rad of an
existing sample of that arm. Parameters (frames, topics, thresholds) are declared in
`recorder_node.py`. The session directory contains `images/cap_NNNN.png`, `samples.jsonl` (one
sample per line: link6 pose from TF, joints, point ids, 3D target points, pixels, K, distortion,
`camera_link -> optical` TF) and `session.json` (target config used). The other files of a session: `table_obs.jsonl`, `touches.jsonl`,
`robot_description.urdf`.

## Solve

```bash
ros2 run rebel_dual_arm_calibration solve <session_dir> [<session_dir> ...] \
  [--out calib.yaml] [--report calib_report.json] [--easy-handeye-dir DIR] [--exclude-image cap_0007] \
  [--joint-offsets [--offset-prior-deg 0.3]] [--refine-intrinsics [--intrinsics-prior-frac 0.01 --intrinsics-prior-px 5]] \
  [--touch-sigma-mm 1.0] [--no-touches] [--tcp-file left=PATH[,right=PATH]] \
  [--kfold 5 | --holdout 0.25] [--validate | --no-validate] [--corr-warn 0.8] \
  [--base-from joint|touches] [--keep-touch-margin 0.10] \
  [--intrinsics-file intrinsics.yaml] [--fit-point-ids 101,102] [--check-point-ids 111,112] \
  [--uncertainty-mc N [--mc-seed S]] [--compare previous_calibration.yaml] \
  [--gate-min-points 4] [--gate-touch-rms-mm 1.5] [--gate-joint-touch-mm 1.0] [--gate-joint-touch-deg 0.1] \
  [--gate-agree-sigma-k 3] [--gate-heldout-px 1.0] [--gate-min-check-points 2] [--gate-check-error-mm 3.0] \
  [--gate-check-height-mm 30] [--gate-check-dist-mm 150] [--gate-yaw-sigma-deg 0.15] [--gate-reach-sigma-mm 1.0] \
  [--gate-intrinsics-f-pct 0.3] [--gate-intrinsics-c-px 3]
```

Several session directories are merged (samples, table-board images and touches of the same arm from different
sessions are pooled; the table board and the camera must not have moved between them); `--out` is then required.
Before merging, `solve` checks that all samples and table-board images share image size, camera matrix
(fx, fy, cx, cy within 0.5 px), distortion (1e-3) and the `camera_link -> optical` TF (1 mm, 0.1 deg) and
**refuses with exit code 2 and a message naming both sessions** if they differ (different camera, driver
calibration, resolution or mounting); the URDF kinematics of an arm must also be identical in all sessions that
contain it. It cannot see that the camera was physically moved between sessions
with identical intrinsics: if it was, the joint fit shows a large RMS / warnings; do not use such a result.
`--exclude-image` matches the image name, which is `<session dir name>/images/cap_NNNN.png` when several
sessions are merged. Different `targets` definitions in `session.json` only produce a warning (the target
mount on each arm must not change between sessions).

Options:

- `--joint-offsets`: 6 joint zero offsets per arm as nuisance parameters with a Gaussian prior (sigma
  `--offset-prior-deg`, default 0.3 deg), using FK from `robot_description.urdf`. **Joint 1 of each arm is fixed to 0**:
  its offset is exactly a rotation of the base about z and cannot be told from the base yaw (for the right arm) or
  the world definition (for the left arm). Consequence: the reported right base is the base *as the nominal FK
  sees it*; the data cannot say how much of its yaw is joint-1 zero offset. Joint 6 is separable from the target
  mount rotation only through the touches.
- `--refine-intrinsics`: fx, fy, cx, cy with priors (sigma 1 % of f, 5 px for the centre). **Off by default**. Whether
  on or off, the report states how far the data pull the intrinsics (with the priors, from a separate fit when off)
  and how much the right base and camera would move. A pull of more than a few px means the `camera_info` should be
  recalibrated; do not accept a refined result without understanding why.
- `--kfold K` / `--holdout F`: held-out evaluation (default 5-fold; on by default because the trust gate needs it,
  `--no-validate` skips it). **Every fold is solved from scratch** (PnP, closed-form start values, per-arm fits,
  outlier rejection and joint fit) from its own training samples only; the held-out samples take part neither in
  the initialisation nor in the outlier filtering. Held-out samples are scored with the fold's training thresholds.
- `--tcp-file left=PATH[,right=PATH]`: see "TCP consistency and changing the TCP".
- `--base-from`, `--gate-*`: see the trust gate. `--keep-touch-margin`: see the keep rule.
- `--intrinsics-file`: see "Camera intrinsics". `--fit-point-ids` / `--check-point-ids`: re-assign the kind of touches by id
  (normally the kind is recorded with each touch).
- `--uncertainty-mc N`, `--compare`: see "Uncertainty budget" and "Reproducibility".
- `--no-touches`: ignore touches and table-board images.

Writes `dual_arm_calibration.yaml` (default in the session directory), `<name>_report.json`, and
optionally one easy_handeye2-style `.calib` per arm (camera pose in that arm's base frame). The YAML structure is
unchanged (`notes` additionally carries the trust gate result, the base source, the joint-vs-touch difference, the touch
mismatch and the offsets; `source` is `calibrated` only when the gate passed, else `calibrated_unverified`).

```yaml
dual_arm_calibration:
  source: calibrated          # calibrated_unverified when the trust gate fails
  date: "YYYY-MM-DD"
  world_frame: world          # coincident with arm_left_base_link
  arm_right_base: {xyz: [...], rpy: [...]}   # pose of arm_right_base_link in world, URDF rpy [rad]
  camera_link:    {xyz: [...], rpy: [...]}   # pose of camera_link in world
  notes: "samples, rms px, std devs, dataset path"
```

### TCP consistency and changing the TCP

The solver anchors the forward kinematics of the joint-offset path on the recorded TCP poses, and the session URDF is
saved once, so **replacing the TCP in the URDF/launch does not migrate touches that were already recorded** (and a
change would silently cancel in the fit). Each touch therefore carries the `link6 -> TCP` transform it was recorded
with (`T_link6_tcp`; older touches: taken from the session `robot_description.urdf`), and `solve` makes the TCP
explicit:

- **Without `--tcp-file`**: behaviour is unchanged, the TCP used per arm is printed
  (`TCP left: as recorded (...): link6->TCP xyz [...] mm, igus_tool0->tip [...] mm; N touches`) and stored in the report
  JSON (`tcp`). Touches of **one arm recorded with different TCPs** (different sessions, or touches before/after a TCP
  change) are **refused** (exit 2) because averaging them mixes two TCPs. If the stored TCP differs from the session
  URDF (URDF saved before a TCP change), the stored one is used for the FK of the joint-offset path and a warning is
  printed.
- **`--tcp-file left=PATH[,right=PATH]`** (repeatable): PATH is a TCP result YAML; read are `r_tool0_m: [x, y, z]`
  (the new `igus_tool0 -> xeg32_tool_tip` translation in m), `arm` (must match the key if present) and `verdict`
  (reported; a failing verdict is refused). Every touch of that arm (all sessions, all kinds, boards and stickers) is
  re-based: `X_new` = `X_old` with the `igus_tool0 -> tip` translation replaced by `r_tool0_m` (the
  `link6 -> igus_tool0` transform and all rotations are kept) and
  `T_base_tcp_new = T_base_tcp_old * inv(X_old) * X_new`. Touches of different sessions with different `X_old` are
  therefore allowed together, as long as all of that arm's touches are re-based. The FK of the joint-offset path uses
  the same `X_new`. It is refused (exit 2, nothing solved) if `X_old` of a touch is unknown (neither stored with the
  touch nor in a session URDF) or if the URDF chain is not `link6 -> igus_tool0 -> TCP`.
  The solve output and the report JSON (`tcp.<arm>`: `tcp_file`, `verdict`, `r_tool0_tip_m` old and new,
  `shift_mm` mean/max and per touch) list the TCP used and the base-frame shift of each touch. Re-basing happens in
  memory; the session files are never modified.

```bash
ros2 run rebel_dual_arm_calibration solve $HOME/dual_arm_calibration/left $HOME/dual_arm_calibration/right \
  --out calib.yaml --tcp-file left=tcp_left.yaml,right=tcp_right.yaml
```

### Uncertainty budget (`--uncertainty-mc N`)

The standard deviations of the fit contain detection noise only. `--uncertainty-mc N` (default 0 = off) re-solves the
**written** solution N times (warm start from the written fit, fixed inlier sets, no validation, no loop-closure variants,
no covariance) with the data perturbed by draws of the Type-B terms (1 sigma, Gaussian): print scale in x and y of **each**
of the three boards independently (+-0.15 %; the table-board scale also scales the board touch coordinates), focal length
(fx and fy together) +-0.5 % and principal point +-3 px (+-0.1 % / +-1 px with `--intrinsics-file`), TCP offset +-0.5 mm per
axis per arm (systematic, in the tool frame), touch noise from the **measured** repeat scatter (per arm and axis), and joint zero
offsets +-0.3 deg per joint and arm (only if the written fit does not estimate them; needs the URDF in the sessions).
It reports the 95 % intervals (2.5 / 97.5 percentiles of the draws) of `arm_right_base` (x y z, roll pitch yaw) and `camera_link`
and the predicted arm-to-arm TCP disagreement from the base uncertainty at three workspace points (the carton centre ~0.35 m in
front of the bases midway, and the 0.4 m reach point of each arm towards the board): mean, 95th percentile and maximum of
the distance in mm. **Runtime: about 0.3 s per draw for 54 samples + 40 table images + 53 touches on this workstation (20 draws 5 s,
200 draws 59 s, measured in the mock run); N = 20 only shows the order of magnitude, use N >= 200 for a value you quote.**
The intervals are in the report (`uncertainty_mc`) and summarised in the YAML notes. Not covered: link-length / axis errors,
board flatness, camera mount flexure.

### Repeatability (ISO 9283-inspired)

Touches of the **same id by the same arm at nearly the same wrist orientation** (every pair < 5 deg apart) with **>= 5 repeats**
give `RP = mean distance to the barycentre + 3 sigma` per arm and point (report `repeatability`, section "Repeatability").
To get it, touch a point 5 or more times with the same tool orientation, lifting the tool between touches. This is only
ISO 9283-*inspired*: the standard prescribes programmed repeated approaches, not operator touches, so the number includes
the operator's hand and the jog resolution.

### Reproducibility (`--compare`) and re-reference check (`check_points`)

- `solve ... --compare previous_dual_arm_calibration.yaml` prints the difference of the written `arm_right_base` (mm / deg) and
  `camera_link` to an earlier calibration and stores it in the report and the notes. **Repeat the calibration on another day and
  after a power cycle of both arms and the camera**; accept a difference below **2 mm / 0.2 deg**, otherwise find out what moved
  (camera, base, re-referencing of the joints) before trusting either.
- `ros2 run rebel_dual_arm_calibration check_points <session_dir> [...] --calibration dual_arm_calibration.yaml`
  computes only the check-point error (TRE, held-out) of the touches of the given sessions against a given calibration, **no
  fitting**: use it after a power cycle / re-homing (record a new short session: both arms touch the same check stickers, no
  camera needed) and as an independent verification of a deployed calibration. It prints per point and RMS / max, the
  recommended padding and `RESULT: PASS|FAIL` (exit code 0 / 1; 2 for bad input); `--check-point-ids` assigns ids,
  `--min-check-points`, `--max-error-mm` set the limits.

### Recovery of results

Everything is in the session directories and is deterministic: `solve` can be re-run any time, with other options,
with `--exclude-image` or `--no-touches`; nothing is modified in the sessions. If the machine stopped during
recording, start the recorder again with the **same** `output_dir` (existing samples, touches and the saved URDF are
kept and counted). A crashed or interrupted `solve` writes nothing (the YAML is written last). If a session is
suspect (target moved, camera bumped), drop that directory from the command; sessions are only merged, never
edited. A touch that was recorded at the wrong point id cannot be deleted by a tool: remove the line from
`touches.jsonl` (one JSON object per touch, `point_id`, `arm`) with a text editor, or leave it: a wrong touch shows
up as a large per-touch mismatch.

## Reading the report

- **samples used / rejected**: gross-outlier samples (median residual > max(1.5 px, 5x the typical
  sample)) are removed automatically; individual corners above max(2.5 px, 5 sigma) are masked. A
  few corner outliers are normal. Many rejected samples mean motion blur, a loose target, or joint
  states not in sync with the image.
- **RMS reprojection px** (per arm and overall, Euclidean 2D, inliers only): good is 0.3-0.7 px. Above
  1 px the solver warns; check intrinsics, the measured target dimensions and a loose target. **A high RMS with
  unmodelled joint offsets is typical for these arms (1.5-2 px for 0.4 deg offsets)**; compare the held-out error with
  `--joint-offsets`. `sigma0` is the unit-weight standard deviation in the units of the weighted residuals.
- **std xyz [mm], std rpy [deg]** are 1-sigma values propagated from the Jacobian. **They are detection-noise only
  unless `--joint-offsets` is on** (then the offsets' uncertainty enters through the prior); `joint.std_basis`
  in the JSON and the line `STD DEVS` in the report say which. They never cover the TCP offset, print scale or
  flatness, camera shift. **Only the held-out check-point error (TRE) says something about points that were not
  fitted, and only at those points**; the touch residuals of fitted points are FRE.
- **Loop closure** (only with touches). Three views of the same touches:
  - `in-fit`: the mismatch after the fit (touches were allowed to bend the solution);
  - `camera-only`: touches compared with a solution that did not use them (board pose from the table-board images,
    both bases from the camera chain). Needs table-board images. A large number = the arms/camera/board are
    inconsistent;
  - `cross-arm` (`left touches vs other arm's`): one arm's touches compared with the board pose defined by the other
    arm's touches (works without table-board images).
  All three are evaluated with **nominal FK** (as recorded); with `--joint-offsets` the same numbers with the
  fitted offsets applied are printed as a diagnostic line only.
  The **out-of-fit touch mismatch** is the camera-only number if available, else the cross-arm number: the RMS touch
  mismatch of two arms and the camera against one physical board, in mm, at the board points. If neither exists (for
  example only one arm touched and there are no table-board images) there is **no** out-of-fit number
  (`out_of_fit_rms_mm: null`): the in-fit residual is never substituted and cannot satisfy the gate. The numbers include
  the TCP offset error (the touched point is `xeg32_tool_tip`, touch with different wrist orientations to expose it),
  board flatness and print scale, and the contact error of your hand. A warning is issued at 2.5 mm (wrong board scale,
  TCP offset, moved base, unmodelled joint offsets); the in-fit number is labelled "fit residual (FRE, in-sample)".
  `right base with vs without touches` is how much the touches move the right base; the touches mostly *check*
  the result, they do not outweigh the camera data (1 mm sigma against sub-pixel corners).
- **Joint zero offsets** (with `--joint-offsets`): estimated offsets in degrees with 1-sigma, joint 1 fixed.
  They are nuisance parameters for this solve only and must **never** be applied to the robot or the URDF.
- **Touch-only right base, TCP consistency, trust gate**: see the touch section; the gate block is the last
  thing printed before the correlation list.
- **Observability** (`observability` in the JSON): singular values of the **data-only** Jacobian (pixel and touch
  rows, prior rows removed; centred parametrisation, unit-norm columns, relative to the largest), the six weakest
  directions with their dominant parameters, and a warning when the smallest is below 0.03. Priors make every
  covariance finite, so this is the check that they do not hide a direction the data do not determine. Expected
  with `--joint-offsets`: the joint-6 offset vs the target mount (`structural`, only determined through touches:
  exactly 0 without touches) and, when touches/board are weakly coupled, base y/z vs board y/z; joint-2/3 offsets vs
  base/camera show up here when the poses are not diverse enough. A value near 0 means whatever the report says
  for these parameters comes from the prior.
- **Held-out validation**: every fold is solved on the other samples and scored on the held-out ones: reprojection RMS
  in px, the same as lateral error in mm at the target depth, and the touch mismatch of held-out touches. Evaluated
  without and with joint offsets; **the recommendation says KEEP only if the held-out reprojection AND the held-out
  touch mismatch both improve** (reprojection by at least 3 %, touch mismatch by at least 10 %, `--keep-touch-margin`).
  Without touches KEEP needs only the reprojection improvement and the report says that the decision is weak. KEEP
  means "keep the offsets in the solve as nuisance parameters", never "apply them to the robot". If they do not help,
  the offsets are fitting noise.
- **Parameter correlation**: pairs with |r| > 0.8 are warnings (the matrix is in the JSON under
  `correlation`). The camera and right-base translations are expressed at the data centroid, otherwise a rotation
  and the translation it causes at 1 m distance would always be correlated. Pairs of the target mount parameters
  among themselves and the joint-6 offset with its target mount are expected (`structural`) and listed only as a
  note. Anything else (camera vs target mount, base vs mount, offset vs offset) means the poses do not separate
  these parameters: take more varied poses.
- **3D consistency vs per-sample PnP**: position disagreement between the calibrated model and the
  independent per-image PnP target pose. It is dominated by the PnP depth noise of a small board,
  so it is an upper bound on the position error of one view, not on the calibration.
- **Separate per-arm solutions** (nominal kinematics, no touches, no offsets): each arm alone gives the camera in its
  own base frame. With this model the two arms constrain different parameters (see the first section), so the joint
  and composed-separate results agree to a fraction of a millimetre **without** touches/offsets; with them the
  difference is the effect of the extras.
- **Intrinsics**: how far the data would pull fx, fy, cx, cy (see Options).
- **cond(J)** and **WARNINGS**: fewer than 20 distinct poses per arm (30-40 recommended, noted), rotation spread below
  30 degrees, link6 position spread below 50 mm, grazing views, condition number above 1e5, correlations, touch
  loop closure. A warning means collect more diverse poses or fix the setup; do not use the result for cutting on
  warnings you do not understand.

The solver: per-sample IPPE PnP -> `cv2.calibrateRobotWorldHandEye` per arm (with IPPE flip
resolution and outlier gating) -> per-arm reprojection refinement -> joint refinement of 24 unknowns
(camera optical pose in world, right base in world, two target mounts) with a soft-L1 loss on pixel
residuals and fixed intrinsics from `camera_info` -> if touches / offsets / intrinsics are requested, a second joint fit
adds the table board pose (6), the touch residuals, the joint offsets (5 per arm) and intrinsics, all with their
priors; the variants for the loop-closure numbers and the intrinsics pull are re-fits from that solution with the same
inlier set.

## Tests

```bash
cd src/rebel_dual_arm_calibration && OMP_NUM_THREADS=3 nice -n 19 python3 -m pytest -p no:anyio test      # ~5 min (132 tests; the validation folds re-run the whole pipeline)
python3 -s -m pytest -p no:anyio test     # same tests with the system OpenCV 4.5.4 (if a pip OpenCV shadows it)
```

Source the ROS environment first if you want the recorder tests (they skip without `rclpy`; they use an isolated
DDS domain, never the robot's) and the PyKDL cross-check of the FK. Synthetic ground-truth recovery (3x3 and 4x4
ChArUco, single marker, outliers), rendered-board detection with correct per-arm IDs, print-file scale,
conditioning warning, YAML writer, CLI; table board (config, detection without the touch corners, crosshairs on the
print); FK against PyKDL (1e-9) and against the recorded poses; joint offsets, touches (wrong right base, wrong
board scale), held-out validation, correlation warning, intrinsics, multi-session merge with touches, recorder
`capture_touch` / table observations / URDF saving; touch-only Kabsch base (known base with noise, collinear/too few points, spread warning),
TCP consistency, the trust gate (PASS on consistent data; FAIL on a displaced base / large mismatch / displaced check point / no or one-sided check
points / insufficient touch geometry / in-fit-only numbers / bad held-out values / intrinsics pull; every check evaluated on values) and the YAML
`source`, `--base-from`, the written-base choice (offset fit only on KEEP), the keep rule, leak-free validation folds, observability of the
data-only Jacobian; check-point error (TRE) and its independence of all fits, Monte Carlo noise prediction of the touch-only base against the
empirical spread, repeatability, `--compare`, the `check_points` and `calibrate_intrinsics` commands (synthetic views, known intrinsics recovered),
`--intrinsics-file`, print-scale anisotropy (x/y squares and markers, detection, solver, print), the 2 s stationary window, the two-frame corner
check, free stickers, `capture_intrinsics`, session metadata, the Monte Carlo uncertainty budget. `rebel_dual_arm_calibration/study.py` regenerates the
target-size study, `study_loop.py` the offsets/touches study, `study_nominal.py` the nominal-model prediction error and touch-only vs joint base study
(numbers in `scan_results/dual_arm_calibration_20261006/README.md`).

## Limitations

- Intrinsics are fixed to `camera_info` unless `--refine-intrinsics` (priors; the pull is always reported).
- Without `--joint-offsets` the arm links are assumed to follow the joint angles exactly. With it, only joint zero
  offsets are modelled, not link-length / axis-misalignment errors (the full kinematic calibration of
  industrial_calibration's DH chain is not implemented). The offsets must not be applied to the robot (nuisance
  parameters; they absorb backlash, compliance and link errors and are invalidated by re-referencing); the base
  pose obtained *with* offsets is the one written, and tool positions computed with the nominal FK stay off by the
  offsets' effect (about 7 mm RMS in the synthetic study) - hence the padding guidance.
- The TCP (`xeg32_tool_tip`) is taken from the URDF; its error enters the touch mismatch (see above) but is not
  estimated.
- No prior is used that both bases stand on the same table plane. If the CAD says the bases are
  coplanar and 0.4474 m apart, compare the result with that as a sanity check.
- Joint states and TF are read when the service is called; the stationary check makes timing offsets irrelevant.
- Touches give a loop closure at the table board and the fit stickers; the held-out check stickers demonstrate the agreement of
  the two arms **at the check points only** (typically 2 points: a PASS is a statement about those points, not a certificate
  for the workspace). Accuracy elsewhere is extrapolated; the padding recommendation is the protection against that.
- The check-point and agreement statistics assume the touches are independent and Gaussian; the repeat scatter that feeds the noise
  prediction contains the kinematic scatter between wrist orientations but not the systematic error that all repeats share.
- The two-frame corner check (coherent motion 0.2 px, single corner 1.0 px) also rejects captures with unstable corner detection
  (a marker corner that jumps by more than 1 px), not only a moving board; tune `max_corner_motion_px` /
  `max_corner_single_px` only if you understand why the corners jitter.
- The Monte Carlo budget covers the listed Type-B terms; unmodelled link-length / axis errors, board flatness and camera-mount
  flexure are not in it.
