# igus_fk_check

Does our URDF match the igus ReBeL controller's own kinematic model?

The controller reports its computed Cartesian pose in every CRI STATUS (`POSCARTROBOT x y z a b c`).
The `igus_rebel` driver publishes it unconverted on `~/robot_cart_pose`
(`igus_rebel_msgs/msg/RobotCartPose`, best-effort sensor QoS) together with the joint angles
of the same STATUS. `capture` records still poses; `analyze` compares them with our forward
kinematics.

The units of `pos_cart_robot_raw` are as reported by the controller, believed mm and deg; the
A/B/C Euler convention is unverified (that is one of the things `analyze` tests).

## Real-robot procedure

1. In iRC (or the pendant) check which **TCP** is set. POSCARTROBOT refers to the TCP configured
   in the controller: with no TCP it is the flange (compare against `link6`), with a tool TCP it is
   the tool (compare against `xeg32_tool_tip` or fit the tool offset). Write the TCP down.
2. Start the driver from the worktree install on an **unused** ROS domain, never domain 11:
   ```bash
   source /opt/ros/humble/setup.bash && source <worktree>/install/setup.bash
   export ROS_DOMAIN_ID=<unused> ROS_LOCALHOST_ONLY=1
   ros2 node list          # must be empty
   ros2 launch igus_rebel rebel.launch.py hardware_protocol:=rebel robot_ip:=<robot ip>
   ```
   Only enable the robot (no controller needs to move it, no commands are sent by the tool).
   The tool itself only subscribes.
3. In another terminal (same domain, same sourcing):
   ```bash
   python3 tools/igus_fk_check/igus_fk_check.py capture --out fk_capture.csv
   ```
   Default node is `/igus_rebel` (`--node` for another name; the topic is `<node>/robot_cart_pose`).
4. Jog the arm by hand or with the pendant to **8-12 well spread poses** (each of joints 2..6
   should cover more than 20 deg across the set, with different wrist orientations; j1 can stay
   put). Hold still at each pose and press Enter. Each capture averages 1 s and is rejected if any
   joint std exceeds 0.02 deg, if POSCARTROBOT moves, or if KINSTATE is non-zero; just press Enter
   again. `q` + Enter finishes. For a scripted run: `--auto N --auto-interval S`.
5. Analyze (expand the URDF first; the same xacro the launch uses):
   ```bash
   xacro src/igus_rebel_description/urdf/igus_rebel_xeg32_wrapper.urdf.xacro \
         hardware_protocol:=mock_hardware > /tmp/robot.urdf
   python3 tools/igus_fk_check/igus_fk_check.py analyze fk_capture.csv --urdf /tmp/robot.urdf \
         --tip-link link6          # or xeg32_tool_tip
   ```
   Dual-arm: `--prefix arm_left_` (applied to `--base-link` / `--tip-link`), with the dual-arm URDF.
   `--pos-units m` / `--angle-units rad` override the mm / deg assumption.

## What analyze reports

- (a) raw position difference per pose (igus - ours, assuming mm).
- (b) `p_igus = p_ours + R_ours t + b`: constant base offset `b` and tool offset `t` (in the tip
  frame), linear least squares, residual per pose and RMS.
- (c) the same with a rigid base rotation: `p_igus = Rb (p_ours + R_ours t) + b`.
- (d) orientation: for all 24 Euler conventions (12 axis orders, intrinsic/extrinsic, degrees)
  fit constant `Rb`, `Rt` in `R_igus = Rb R_ours Rt` and report the angular RMS. Note: with `Rb`
  and `Rt` both free, some conventions are mathematically indistinguishable (cyclic relabellings of
  Tait-Bryan orders such as xyz = yzx = zxy, and all proper-Euler orders such as zyz = zxz = xyx
  of the same flavour). They are reported as one class of equal residual; the member with the
  smallest base rotation is named as most plausible. If you know the controller's base frame
  equals ours, that member is the answer; otherwise check the fitted `Rb` is sensible.
- Flags: values that look like metres or radians, fewer than 6 poses, a joint (except j1) spanning
  less than 20 deg, ill-conditioned fit (tool orientation barely changes), no clear best convention.
- `VERDICT` line: RMS position residual after (b) and (c), and the best orientation convention.
  Heuristic labels: <= 1 mm consistent, <= 3 mm marginal, else mismatch. A good RMS with a large
  `b` or `t` means a frame/TCP offset, not wrong link lengths; wrong link lengths or joint
  zero/sign errors show up as a large residual that does not go away in (b) or (c).

Tests: `cd tools/igus_fk_check && source /opt/ros/humble/setup.bash && python3 -m pytest -p no:anyio test_igus_fk_check.py`
(builds synthetic captures from the real URDF, expanded from the source tree; needs `xacro`).
