# AMR parking study (static AMR placeholder; only a recommendation where to park)

Frame: d = AMR front bumper to cabinet front face (m); lateral = AMR centre line relative to the target carton centre (+ = robot y); lift / sliders = joint values (fixed during one pick).
Window = contiguous range of d (10 mm steps) with the full sequence feasible at the best lateral/lift/sliders; best d = centre of that window; `box` = the 8 corners of +-20 mm x/y and +-1.5 deg yaw all pass.

## Front mode (recommended default)

| target | shelf / carton | feasible | best d [mm] | d window [mm] | lateral [mm] (window) | lift [m] (window) | sliders L/R [m] | joint margin [deg] | min sigma_min L/R | min clearance [mm] | +-20 mm / +-1.5 deg box | MoveIt mock plan |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T1 | 1 / ultimaster_nagomi (mid) | NO | - | - | - | - | - | - | - | - | - | - |
| T2 | 2 / accuforce (mid) | NO | - | - | - | - | - | - | - | - | - | - |
| T3 | 3 / agent_monorail (mid) | yes | 410 | 350..480 | 5 (-65..75) | -0.600 (-0.62..-0.58) | -0.0060 / +0.0060 | 4.6 | 0.100 / 0.144 | 6 | pass | all OK |
| T4 | 3 / conqueror_nc_pro (mid) | yes | 410 | 350..480 | 5 (-65..75) | -0.600 (-0.62..-0.58) | -0.0060 / +0.0060 | 4.6 | 0.100 / 0.144 | 4 | pass | not run |
| T5 | 4 / angiolite (mid) | yes | 440 | 360..530 | 55 (-35..145) | -0.460 (-0.62..-0.30) | -0.0060 / +0.0060 | 6.2 | 0.110 / 0.131 | 6 | pass | not run |
| T6 | 5 / agent_monorail (mid) | yes | 440 | 360..520 | 55 (-35..145) | -0.210 (-0.45..+0.03) | -0.0060 / +0.0060 | 7.2 | 0.094 / 0.125 | 6 | pass | not run |
| T7 | 6 / accuforce (mid) | yes | 440 | 360..520 | 55 (-35..145) | +0.090 (-0.15..+0.33) | -0.0060 / +0.0060 | 7.5 | 0.093 / 0.125 | 6 | pass | not run |
| T8 | 6 / conqueror_nc_pro (mid) | yes | 440 | 360..520 | 55 (-35..145) | +0.100 (-0.14..+0.34) | -0.0060 / +0.0060 | 7.2 | 0.094 / 0.125 | 4 | pass | all OK |
| E1 | 3 / agent_monorail (last) | yes | 410 | 350..480 | 5 (-65..75) | -0.600 (-0.62..-0.58) | -0.0060 / +0.0060 | 4.6 | 0.100 / 0.144 | 6 | pass | not run |
| E2 | 6 / angiolite (first) | NO | - | - | - | - | - | - | - | - | - | - |

Reasons for NO: T1: no (d, lat, lift, slider) puts all key poses of both arms inside the IK reach map (same lift for both arms) ; T2: every map-feasible candidate fails the full path / collision evaluation (see full_eval_failures) {'left: L0 home->stage': 23}; E2: every map-feasible candidate fails the full path / collision evaluation (see full_eval_failures) {'left: L0 home->stage': 91, 'left: L0b stage->pre': 59}

**Common front standoff: d = 420 mm** passes for 7 of 7 reachable targets (each at its own lateral/lift); corner box: {'T3': True, 'T4': True, 'T5': True, 'T6': True, 'T7': False, 'T8': True, 'E1': True}. Targets needing something else: T1, T2, E2 (not reachable, see reasons).

## Top-down mode

No target has a parking pose where the right-arm top-down grasp is collision free (>= 15 mm) with the real arm; pose-level diagnostic (necessary condition, independent of the transit model), 1500 sampled IK-reachable (d, lateral, lift, slider) per target:

| target | IK-reachable map cells | collision-free pre/grasp poses | clearance violations (count of samples) |
|---|---|---|---|
| T1 | 72 | 0 | {'arm_blocks': 72, 'arm_cabinet': 72, 'tool_cabinet': 72} |
| T2 | 10516 | 0 | {'arm_blocks': 1499, 'arm_cabinet': 1500, 'tool_cabinet': 1500} |
| T3 | 11256 | 0 | {'arm_blocks': 1500, 'arm_cabinet': 1500, 'tool_cabinet': 1500} |
| T4 | 11256 | 0 | {'arm_blocks': 1500, 'arm_cabinet': 1500, 'tool_cabinet': 1500} |
| T5 | 10271 | 0 | {'arm_blocks': 1500, 'tool_cabinet': 1500, 'arm_cabinet': 154} |
| T6 | 10271 | 0 | {'arm_blocks': 1500, 'arm_cabinet': 1500, 'tool_cabinet': 1500, 'robot_stand': 104} |
| T7 | 5856 | 0 | {'arm_cabinet': 1500, 'tool_cabinet': 1500, 'robot_stand': 1} |
| T8 | 5844 | 0 | {'arm_cabinet': 1500, 'tool_cabinet': 1500, 'robot_stand': 1} |
| E1 | 11256 | 0 | {'arm_blocks': 1500, 'arm_cabinet': 1500, 'tool_cabinet': 1500} |
| E2 | 4786 | 0 | {'arm_cabinet': 1500, 'tool_cabinet': 1500} |

## E2 with the suction tool turned 180 deg about the pad axis (extra variant)

{
 "E2": {
  "feasible": false,
  "best_d": null,
  "lateral_offset_m": null,
  "lift": null,
  "feasible_d_window_m": null,
  "corners_box_ok": null,
  "reason": "every map-feasible candidate fails the full path / collision evaluation (see full_eval_failures)"
 }
}
