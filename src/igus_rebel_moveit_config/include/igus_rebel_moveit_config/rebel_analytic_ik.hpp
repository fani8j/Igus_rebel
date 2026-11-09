// Closed-form (Pieper-style) inverse kinematics for the igus ReBeL arm + a caller-supplied
// fixed link6 -> TCP transform (xeg32_tool_tip for either tool variant).
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

// The fixed link6 -> TCP transform of the OLD tool (tool_variant:=xeg32_knife_v1):
// link6 -> igus_tool0 (0,0,0.0527; rpy 0,0,pi) -> xeg32_tool_tip (0.005937,0.094121,0.130699;
// rpy 0), composed into one transform (rotation Rz(pi)). Only for tests/back-compat: the
// executor derives the offset from the live RobotModel instead (the current default tool,
// knife_c, includes a -30 deg pitch rotation in its offset).
Eigen::Isometry3d legacyXeg32KnifeV1ToolOffset();

// Forward kinematics of the TCP in base_link: base_link -> ... -> link6, then the caller's
// fixed `tool_offset` (link6 -> TCP, any rigid transform). Does not apply or check joint
// limits.
Eigen::Isometry3d forwardKinematics(const JointVector& q, const Eigen::Isometry3d& tool_offset);

// Returns every IK solution for the given TCP pose (base_link frame) that lies within
// `limits` (checking the raw solution and, where the joint's range allows, the +-2*pi
// wrapped variants). Solutions are not sorted or deduplicated beyond the up-to-8 Pieper
// branches (2 shoulder x 2 elbow x 2 wrist); callers that need a specific order (e.g. by
// distance to a seed) must sort the result themselves. Returns an empty vector fast
// (no iterative search) for unreachable poses: wrist center out of the arm's reach
// envelope, or an orientation solve with no valid wrist decomposition.
std::vector<JointVector> solve(
  const Eigen::Isometry3d& target, const Eigen::Isometry3d& tool_offset,
  const std::array<JointLimits, 6>& limits = defaultJointLimits());

// Pieper branch of a joint vector, exactly the (shoulder x elbow x wrist) partition solve()
// enumerates (2 x 2 x 2). Each field is +1 or -1 (a value of exactly 0 counts as +1):
//  - shoulder: sign of the wrist centre's radial coordinate X in the joint1-rotated frame
//    (+1 = the wrist centre lies in front of joint1's x axis, solve()'s theta1 = psi - base yaw
//    branch; -1 = behind the axis, the theta1 + pi branch). Boundary: wrist centre on joint1's axis.
//  - elbow: sign of (theta3 - beta), solve()'s signed gamma (+1 = gamma >= 0). Boundary: the
//    straight-arm singularity theta3 = beta.
//  - wrist: sign of theta5 (+1 = solve()'s sign5 = +1). Boundary: theta5 = 0 (wrist singularity).
struct BranchLabel
{
  int shoulder{1};
  int elbow{1};
  int wrist{1};
  bool operator==(const BranchLabel& other) const
  {
    return shoulder == other.shoulder && elbow == other.elbow && wrist == other.wrist;
  }
  bool operator!=(const BranchLabel& other) const {return !(*this == other);}
};

// Distance of a configuration from each branch boundary (all >= 0): shoulder in metres (|X| of
// the wrist centre), elbow and wrist in radians (|theta3 - beta|, |theta5|).
struct BranchMargins
{
  double shoulder_m{0.0};
  double elbow_rad{0.0};
  double wrist_rad{0.0};
};

// Pure functions of the six joint angles (no tool offset: the branch does not depend on the TCP).
BranchLabel classifyBranch(const JointVector& q);
BranchMargins branchMargins(const JointVector& q);

// The shoulder coordinate behind the shoulder label: X(q2, q3), the signed radial coordinate of
// the wrist centre in the joint1-rotated frame (m). Nonlinear in (q2, q3).
double wristRadialX(double q2, double q3);
// A bound on |X(q2 + s*dq2, q3 + s*dq3) - X(q2, q3)| valid for every s in [0, 1] and every
// (q2, q3): the Lipschitz constants of X are (L1 + L2) in q2 and L2 in q3, with L1 = 0.2415 m
// (upper arm) and L2 = hypot(kJ4X, kJ4Z + kJ5Z) = 0.2995 m (effective forearm).
double wristRadialXVariation(double dq2, double dq3);
// beta: the elbow's straight-arm angle, the boundary of the elbow label (q3 = beta).
double elbowStraightArmAngle();

// Diagnostic twin of solve() (read-only; solve() is untouched): enumerates the same 2 shoulder x
// 2 elbow x 2 wrist Pieper branches but does NOT drop the ones outside the joint limits; it
// reports why a pose has no solution. Used only by the opt-in left_varying_diagnostics log.
//  - reach: the wrist centre distance from the shoulder-elbow plane origin versus the planar
//    2-link reach envelope [|L1 - L2|, L1 + L2] (identical for both shoulder branches).
//  - branches (only when the wrist centre is reachable): per branch the joint values (wrapped to
//    +-pi, the representative closest to the limits), the violation of each joint limit in rad
//    (0 = inside; after considering the +-2*pi representatives exactly like solve()) and the
//    worst joint. `within_limits` is exactly "solve() returns this branch".
struct ReachDiagnosis
{
  bool reachable{false};
  double radius_m{0.0};      // sqrt(X^2 + Z'^2) of the wrist centre in the elbow plane
  double reach_max_m{0.0};   // L1 + L2
  double reach_min_m{0.0};   // |L1 - L2|
  double excess_m{0.0};      // > 0: this much too far; < 0: this much too close; 0 = reachable
};
struct BranchDiagnosis
{
  int shoulder{1}, elbow{1}, wrist{1};  // +1 / -1, solve()'s order (see classifyBranch)
  JointVector q{};
  std::array<double, 6> violation{};    // rad, >= 0
  double max_violation{0.0};
  std::size_t worst_joint{0};           // 0-based
  bool within_limits{false};
};
struct IkDiagnosis
{
  ReachDiagnosis reach;
  std::vector<BranchDiagnosis> branches;  // 8 when reachable, else empty
};
IkDiagnosis diagnose(
  const Eigen::Isometry3d& target, const Eigen::Isometry3d& tool_offset,
  const std::array<JointLimits, 6>& limits = defaultJointLimits());

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
  unsigned seed, const Eigen::Isometry3d& tool_offset,
  const std::array<JointLimits, 6>& limits = defaultJointLimits());

}  // namespace rebel_analytic_ik
