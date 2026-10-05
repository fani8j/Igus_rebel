# Calibrating the carton physics against the real boxes

The USD physics values are reviewed defaults (two independent advisor reviews, 2026-10-05),
not measurements. Each test below replaces one of them. Do the tests in this order: friction
and pack pressure decide whether the suction pull-out works in sim, and mass and bounce matter
much less.

Equipment: the five empty cartons, a kitchen scale (1 g), a phone with an inclinometer app
and slow-motion video, and a small force gauge with **0.01-0.05 N resolution**. A luggage
scale is too coarse for these forces, because an unpacked carton slides at about 0.3 N. You
also need a flat painted/powder-coated steel panel like the shelf, and one SMC ZP2 pad, plus
a gripper finger or a sample of its pad material if available.

Repeat every measurement 5 times and record the median and the spread.

| # | Test | How | Result → where it goes |
|---|---|---|---|
| 1 | **Empty carton mass** | Weigh each empty carton. | `boxes.yaml` → `mass_estimate_g.carton`; then set `mass_kg` = carton + contents (likely) and narrow `low`/`high`. |
| 2 | **Static friction, board on board** | Lay one carton flat on another (big faces touching) on a board you can tilt. Raise it slowly until it slips; read the angle θ. | μs = tan θ → `boxes.yaml` `physics_material.static_friction` (now 0.35, interim). |
| 3 | **Static friction, board on shelf steel** | As in 2, on the steel panel. | μs = tan θ → `shelf_layout.yaml` `shelf_material.static_friction` (now 0.30). |
| 4 | **Dynamic friction** | Tilt a few degrees past θ, film the slide in slow motion, measure the acceleration a from frame positions; or drag at steady speed with the gauge and read F. | μd = (g sin θ − a)/(g cos θ), or μd = F/(m g) → `dynamic_friction` of the same pair (now 0.30 board, 0.25 shelf). |
| 5 | **Pad on board, gripper finger on board** | Tilt test with the carton on the pad / finger material. | Rubber pair goes on the gripper's material with `frictionCombineMode = max` (target about 0.8/0.6-0.7). A bare metal finger is about 0.25-0.3; fit rubber pads if the clamp slips. |
| 6 | **Pack-pressure pull-out (most important)** | Stock a real shelf row as in `shelf_layout.yaml`, packed as tight as the hospital does. Tape a loop to one spine and pull straight out slowly with the gauge. Record the breakaway peak and the steady force, for a loose row and a tight row. | The force the sim must reproduce. The current sim has 1 mm gaps and no squeeze, so it gives only shelf friction (~0.3-0.4 N). If the real force is clearly higher, the scene needs gap/tilt jitter, a settled start state and a side preload. |
| 7 | **Suction limits** | Pad on the carton face at the planned vacuum. Pull straight off (normal), then sideways (shear), until it lets go. | Surface Gripper `isaac:coaxialForceLimit` and `isaac:shearForceLimit` (use about half the measured value as a margin); `isaac:maxGripDistance` about 1 mm. |
| 8 | **Bounce (low priority)** | Drop a carton flat from 0.3 m onto the steel panel and film it; read the rebound height h′. | e = √(h′/h) → `restitution` (now 0.05; expected 0.1-0.3, and it barely matters for the pick). |
| 9 | **Centre of mass (needs a full carton)** | Balance a sealed carton on a pencil along its height. | Centre-of-mass offset along carton X. Until measured, randomize it 10-25 mm towards the lower spine end. |

## Randomization ranges to use until the measurements are in

- **Mass:** uniform over `mass_estimate_g.low..high` per carton, plus the two endpoints.
- **Friction:** board 0.2-0.6, shelf 0.2-0.5, rubber 0.5-1.2.
- **Centre of mass:** along carton X, 10-25 mm towards the lower end (sign from `up_end`); ±3 mm through the thickness.

Mass and centre of mass can be set per carton instance (an override on the instance root).
Friction lives inside the shared asset, so a per-carton friction needs a material bound on the
instance root with `strongerThanDescendants`.

## Solver check

After any change, run the pull-out at 240 Hz and at 480 Hz with 32/2 solver iterations. Keep
240 Hz / 16-2 only if the pull forces differ by less than 5 %.
