// Pure helpers for splitting a planned MTC solution into JTC runs and robot-side CRI
// "Move Joint" transits (box_h_cut_executor, opt-in cri_transit_enabled). No ROS or MoveIt
// dependency, so everything here is unit-testable in isolation (test/test_transit_split.cpp).
//
// Safety model: a transit leaf may be sent to the controller's own joint move only if it is
// PROVEN to be a collision-checked joint-space straight line. The proof is made of
//   (1) provenance: the leaf was created by a recorded transit stage and its trajectory is the
//       output of the JointInterpolationPlanner (which collision-checks every interpolated
//       state) -- decided by the caller, passed in as booleans;
//   (2) geometry: IsJointLinear() re-verifies from the waypoints that the path really is the
//       straight chord first -> last, monotone, within `tol` per joint;
//   (3) the leaf moves exactly the arm joints, by at least a minimum length, and its
//       scene_diff carries nothing beyond robot state (so skipping it loses nothing).
// Any doubt means "not eligible": the leaf then runs on the JTC exactly as before.
#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace rebel_transit_split
{

using Waypoints = std::vector<std::vector<double>>;

struct JointLinearResult
{
  bool linear{false};
  double max_chord_deviation{0.0};  // rad, max over waypoints and joints
  std::string reason;               // empty when linear
};

// True iff `waypoints` (>= 2 points, equal non-zero width, all finite) is the straight joint
// space line first -> last: with s = projection of (q - q_first) on (q_last - q_first) / |.|^2,
// every s lies in [-eps, 1+eps], s never decreases by more than eps from one waypoint to the
// next, and every joint of every waypoint is within `tol` rad of q_first + s (q_last - q_first).
// A degenerate (zero-length) path is rejected: it cannot be parameterised.
JointLinearResult IsJointLinear(const Waypoints& waypoints, double tol = 1e-3);

// max_j |last_j - first_j|; NaN when the sizes differ or a value is not finite.
double MaxJointChange(const std::vector<double>& first, const std::vector<double>& last);

// Reorders `values` (named by `names`) into the order of `order`. Returns an empty vector
// unless `names` and `order` contain exactly the same set of distinct names.
std::vector<double> ReorderToJointOrder(
  const std::vector<std::string>& names, const std::vector<double>& values,
  const std::vector<std::string>& order);

// Same reordering applied to every waypoint; empty if any waypoint fails.
Waypoints ReorderWaypoints(
  const std::vector<std::string>& names, const Waypoints& waypoints,
  const std::vector<std::string>& order);

struct LeafCandidate
{
  bool recorded_transit{false};         // created by a stage recorded as a free-space transit
  bool provenance_registered{false};    // trajectory is a JointInterpolationPlanner output
  bool scene_diff_robot_state_only{false};
  std::vector<std::string> joint_names;  // of the leaf's trajectory message
  Waypoints waypoints;                   // positions in joint_names order
};

struct LeafVerdict
{
  bool cri{false};
  std::string reason;  // why not (empty when cri); a short human string for the log
};

// Applies the full eligibility rule. `arm_joints` is the planning group's joint order.
LeafVerdict EvaluateLeaf(
  const LeafCandidate& leaf, const std::vector<std::string>& arm_joints, double min_length_rad,
  double linear_tolerance = 1e-3);

// One contiguous slice [begin, end) of the leaf list. cri == true runs are always exactly one
// leaf; consecutive non-CRI leaves are merged into one JTC run (so a task with no CRI leaf
// yields a single run covering everything).
struct Run
{
  bool cri{false};
  std::size_t begin{0};
  std::size_t end{0};
};
std::vector<Run> BuildDispatchPlan(const std::vector<bool>& cri_leaf);

// Hand-off parameters for the robot-side move.
// MoveIt-style scaling fraction (0,1] -> CRI percent, clamped to [1, cap_percent].
double PercentFromScaling(double scaling, double cap_percent);
// max(0.03, 0.03 * largest joint delta): allowed deviation from the start->goal line.
double PathToleranceRad(double largest_joint_delta_rad);
// Trapezoidal duration estimate of a synchronized joint move whose leading joint travels
// `max_delta_rad`, at velocity/acceleration percentages of the reference limits. Used only
// for a client-side wait bound and logging, never for safety decisions. Returns +inf for
// invalid input.
double EstimateDurationS(
  double max_delta_rad, double velocity_percent, double acceleration_percent,
  double reference_velocity_rad_s, double reference_acceleration_rad_s2);

}  // namespace rebel_transit_split
