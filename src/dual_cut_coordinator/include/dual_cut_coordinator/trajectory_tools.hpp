// Trajectory sampling, densifying, concatenation and joint-name mapping for the dual-arm cut
// coordinator (WP5). Pure logic: no ROS node, no MoveIt.
//
// SAMPLING RULES (match the Humble joint_trajectory_controller, ros-humble-joint-trajectory-controller
// 2.54.0, src/trajectory.cpp, Trajectory::sample / interpolate_between_points / deduce_from_derivatives):
//   1. A trajectory is a list of points with time_from_start. A sample at time t lies in the segment
//      [t_i, t_{i+1}) with t_i <= t < t_{i+1} (a sample exactly on a waypoint belongs to the segment
//      that starts there; the value is continuous so the choice only matters for the degree below).
//   2. The interpolation degree is chosen PER SEGMENT from the two end points:
//        has_velocity = both points have velocities, has_accel = both points have accelerations
//        has_velocity && has_accel   -> quintic spline (p, v, a at both ends)
//        has_velocity && !has_accel  -> cubic spline (p, v at both ends)
//        !has_velocity               -> linear in position (accelerations are ignored)
//      The JTC leaves the output at zero if accelerations are given without velocities; this library
//      rejects such trajectories (validate()) instead of reproducing that.
//   3. Quintic coefficients, with T the segment duration and tau = t - t_i:
//        c0 = p0, c1 = v0, c2 = a0/2
//        c3 = (-20 p0 + 20 p1 - 3 a0 T^2 + a1 T^2 - 12 v0 T - 8 v1 T) / (2 T^3)
//        c4 = ( 30 p0 - 30 p1 + 3 a0 T^2 - 2 a1 T^2 + 16 v0 T + 14 v1 T) / (2 T^4)
//        c5 = (-12 p0 + 12 p1 - a0 T^2 + a1 T^2 - 6 v0 T - 6 v1 T) / (2 T^5)
//        p(tau) = sum c_k tau^k   (velocity and acceleration are the analytic derivatives)
//      Cubic: c2 = (-3 p0 + 3 p1 - 2 v0 T - v1 T) / T^2, c3 = (2 p0 - 2 p1 + v0 T + v1 T) / T^3.
//   4. After the last point the output is the last point; missing velocities/accelerations are zero.
//      At t == t_0 the first segment is evaluated (so velocity/acceleration are the spline's, e.g. the
//      segment's constant velocity for linear segments, not the waypoint's own value).
//   5. Before the first point (t < t_0) this library returns the first point. The real controller
//      instead blends from the measured state to the first point; that is not modelled here. The
//      coordinator starts goals with the arm at rest at the first point, so the two coincide.
//   6. Not modelled: angle wrap-around (all ReBeL joints are non-continuous), position deduction from
//      velocity-only points (MTC always fills positions: positions are required), controller
//      tolerances and the controller's update-rate quantisation.
#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "trajectory_msgs/msg/joint_trajectory.hpp"

namespace dual_cut
{

/// Result of a fallible operation; fail-closed code paths return !ok with an explanation.
struct Status
{
  bool ok = true;
  std::string message;

  static Status Ok() { return Status{}; }
  static Status Error(std::string msg)
  {
    Status s;
    s.ok = false;
    s.message = std::move(msg);
    return s;
  }
  explicit operator bool() const { return ok; }
};

struct TrajPoint
{
  double time_s = 0.0;
  std::vector<double> positions;
  std::vector<double> velocities;      // empty or same size as positions
  std::vector<double> accelerations;   // empty, or same size as positions and velocities non-empty
};

struct JointTrajectory
{
  std::vector<std::string> joint_names;
  std::vector<TrajPoint> points;

  /// Time of the last point (0 if empty).
  double duration() const { return points.empty() ? 0.0 : points.back().time_s; }
};

JointTrajectory fromMsg(const trajectory_msgs::msg::JointTrajectory & msg);
/// Header stamp is left zero; the caller sets the common start stamp of a goal.
trajectory_msgs::msg::JointTrajectory toMsg(const JointTrajectory & traj);

/// Non-empty, strictly increasing finite times, finite values, consistent sizes, accelerations only
/// together with velocities.
Status validate(const JointTrajectory & traj);

enum class Degree
{
  Linear,
  Cubic,
  Quintic
};
/// Interpolation degree the JTC uses between two points (rule 2).
Degree segmentDegree(const TrajPoint & a, const TrajPoint & b);

/// Sample like the JTC (rules 1-5). Output has positions, velocities and accelerations filled.
Status sampleAt(const JointTrajectory & traj, double t, TrajPoint & out);

struct SampleOptions
{
  double dt_s = 0.05;                  // regular sample period
  bool include_waypoints = true;       // keep every original waypoint time
  double max_joint_step_rad = 0.01;    // densify until no joint steps more than this; <= 0 disables
  int max_bisect_depth = 24;           // refinement depth limit (fail closed beyond it)
  size_t max_samples = 2000000;        // fail closed above this many samples
};

/// Regular samples (k*dt_s from the first point, plus the end and optionally all waypoints),
/// then densified by bisecting the spline until no joint moves more than max_joint_step_rad between
/// consecutive samples. The refinement tests both halves and the span, so a reversal inside a span
/// is caught as long as it moves a joint more than the limit relative to the mid sample.
Status sampleTrajectory(
  const JointTrajectory & traj, const SampleOptions & options, std::vector<TrajPoint> & out);

/// Pure joint-space densify of an already sampled sequence of configurations: linear sub-steps so
/// that no joint changes more than max_step between consecutive outputs. Keeps all inputs.
Status densifyLinear(
  const std::vector<std::vector<double>> & path, double max_step_rad,
  std::vector<std::vector<double>> & out, size_t max_points = 2000000);

struct ConcatOptions
{
  double position_tol = 1e-6;       // boundary positions must match within this (rad)
  double velocity_tol = 1e-3;       // checked only if both boundary points have velocities
  // Checked only if both boundary points have accelerations. MTC segments end at rest (v = 0) but their
  // time parameterisation leaves NONZERO accelerations at the ends (measured 0.5 rad/s^2 on real exports,
  // up to the 5.24 rad/s^2 limit), so consecutive segments disagree here. The concatenated trajectory keeps
  // the earlier segment's value at the boundary; the sampler reproduces exactly that, and this tolerance
  // only catches garbage.
  double acceleration_tol = 6.0;
};

/// Concatenate segments (each starting at its own time 0, and whose first point equals the previous
/// segment's last point). The duplicate first point of every later segment is dropped, the later
/// times are shifted by the cumulative duration. All segments must have identical joint_names.
/// segment_first_point (optional out) gets, per input segment, the index of its first point in the
/// result; the final entry is the index of the last point.
Status concatenate(
  const std::vector<JointTrajectory> & segments, const ConcatOptions & options,
  JointTrajectory & out, std::vector<size_t> * segment_first_point = nullptr);

/// One densely sampled configuration along a phase for one arm.
struct PathSample
{
  double time_s = 0.0;               // time since the start of the first segment
  std::vector<double> q;             // same order as SampledPath::joint_names
  bool contact_allowed = false;      // tool/carton contact allowed at this sample
  size_t segment_index = 0;          // input segment this sample belongs to
};

struct SampledPath
{
  std::vector<std::string> joint_names;
  std::vector<PathSample> samples;

  double duration() const { return samples.empty() ? 0.0 : samples.back().time_s; }
};

struct SegmentInput
{
  JointTrajectory trajectory;
  bool contact_allowed = false;
};

/// Sample a chain of segments (each from its own time 0) into one path. Each segment is sampled with
/// its own spline (sampleTrajectory), the duplicate boundary sample is dropped, boundaries are
/// checked with concat.position_tol / velocity_tol / acceleration_tol. A boundary sample is the same
/// physical state for both neighbours, so it is contact_allowed if EITHER neighbour allows contact
/// (and belongs to the earlier segment).
Status buildSampledPath(
  const std::vector<SegmentInput> & segments, const SampleOptions & options,
  const ConcatOptions & concat, SampledPath & out);

// ---- joint-name prefix mapping:  jointN -> arm_X_jointN ----------------------------------------

/// "left" -> "arm_left_", "right" -> "arm_right_"; empty string for any other side.
std::string armPrefix(const std::string & side);
/// Only plain executor names `joint<digits>` are accepted (fail closed on anything else, e.g. a
/// name that is already prefixed).
Status prefixJointName(const std::string & prefix, const std::string & name, std::string & out);
/// All-or-nothing in-place mapping.
Status prefixJointNames(const std::string & prefix, std::vector<std::string> & names);
Status prefixTrajectory(const std::string & prefix, JointTrajectory & traj);
/// Inverse: names must start with the prefix followed by `joint<digits>`.
Status stripJointPrefix(const std::string & prefix, const std::string & name, std::string & out);

}  // namespace dual_cut
