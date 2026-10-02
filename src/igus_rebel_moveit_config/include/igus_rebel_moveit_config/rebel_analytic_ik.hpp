// Closed-form (Pieper-style) inverse kinematics for the igus ReBeL arm + xeg32_tool_tip TCP.
//
// Kinematic structure (derived from igus_rebel_description/urdf/igus_rebel2.urdf, read
// 2026-09-28): joint1 (z), joint2/joint3 (parallel, axis -y, planar elbow with small offsets),
// joint4 (z), joint5 (-y), joint6 (z). Axes 4, 5 and 6 intersect at joint5's origin, so the
// wrist is spherical and the classic Pieper decoupling applies: the wrist-center position is a
// function of joint1..3 only, and joint4..6 are a ZY'Z Euler-like orientation solve once the
// wrist center orientation is known. Every constant below is taken directly from the URDF
// joint origins (composing consecutive fixed joints into the following revolute joint's
// frame); the tiny link3->motor4 x-offset (1.3449814287804 mm) and the link1->motor2/
// motor2->link2 and motor3->link3 y-offsets are carried through exactly rather than rounded
// away (the y-offsets cancel algebraically; see rebel_analytic_ik.cpp).
//
// This header has no ROS or MoveIt dependency (Eigen only) so it can be unit-tested in
// isolation. The caller (box_h_cut_executor) is responsible for feeding it joint limits taken
// from the live RobotModel/joint_limits.yaml and for running selfCheck() at startup to decide
// whether to trust this solver at all (see box_h_cut_executor.cpp's analytic-IK startup gate).
#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <vector>

#include <Eigen/Geometry>

namespace rebel_analytic_ik
{

// One IK solution: joint1..joint6, radians, in URDF joint order.
using JointVector = std::array<double, 6>;

struct JointLimits
{
  double lower{-3.106686};
  double upper{3.106686};
};

// URDF joint limits (igus_rebel2.urdf, 2026-09-28: A1 +-179, A2 -80..140, A3 -80..140,
// A4 +-179, A5 +-95, A6 +-179 deg, 1 deg inside the controller's reported limits).
// Callers integrating with MoveIt should prefer limits read from the live JointModelGroup
// bounds; these defaults exist for standalone use and tests.
std::array<JointLimits, 6> defaultJointLimits();

// Forward kinematics of the xeg32_tool_tip TCP in base_link, matching the URDF chain
// exactly (base_link -> ... -> link6 -> igus_tool0 -> xeg32_tool_tip). Does not apply or
// check joint limits.
Eigen::Isometry3d forwardKinematics(const JointVector& q);

// Returns every IK solution for the given TCP pose (base_link frame) that lies within
// `limits` (checking the raw solution and, where the joint's range allows, the +-2*pi
// wrapped variants). Solutions are not sorted or deduplicated beyond the up-to-8 Pieper
// branches (2 shoulder x 2 elbow x 2 wrist); callers that need a specific order (e.g. by
// distance to a seed) must sort the result themselves. Returns an empty vector fast
// (no iterative search) for unreachable poses: wrist center out of the arm's reach
// envelope, or an orientation solve with no valid wrist decomposition.
std::vector<JointVector> solve(
  const Eigen::Isometry3d& target, const std::array<JointLimits, 6>& limits = defaultJointLimits());

struct SelfCheckResult
{
  bool ok{false};
  std::size_t samples{0};
  // For every sampled q: FK(q) -> solve() -> every returned solution's FK must match the
  // target within tolerance, and q itself (mod the solver's own wrap/branch choice) must be
  // among the returned solutions within tolerance.
  std::size_t samples_with_original_recovered{0};
  double max_position_residual_m{0.0};
  double max_orientation_residual_rad{0.0};
  std::string failure_detail;
};

// Startup self-check: samples `num_samples` random configurations within `limits`, and for
// each one checks FK -> solve() -> (a) every returned solution's FK reproduces the target
// pose within (position_tol_m, orientation_tol_rad) and (b) the original sampled q is among
// the returned solutions within the same tolerance. `ok` is false (and ik should not be
// trusted / fall back to KDL) if any residual exceeds tolerance or if any sample's original
// configuration is not recovered.
SelfCheckResult selfCheck(
  std::size_t num_samples, double position_tol_m, double orientation_tol_rad,
  unsigned seed, const std::array<JointLimits, 6>& limits = defaultJointLimits());

}  // namespace rebel_analytic_ik
