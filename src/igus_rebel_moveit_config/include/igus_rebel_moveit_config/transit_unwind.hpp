// Safe "unwind" detour of a free-space transit (2026-10-22, operator): when a transit to a goal pose has no
// acceptable DIRECT solution (every IK branch is either in collision, e.g. the blade clipping the carton right
// after the previous cut, or would wind joint6 through more than pi), the NearestBranchTransitPlanner may take
// ONE detour through an intermediate waypoint:
//
//   leg 1  straight Cartesian lift of the TCP along base +Z by a candidate height (collision checked),
//   leg 2  at the lifted pose the wrist joints are rotated IN JOINT SPACE to the goal branch's values
//          (joint6; joint4 + joint6; or the whole wrist joint4/5/6),
//   leg 3  the remaining transit from that waypoint to the goal branch (joint interpolation).
//
// The three legs are concatenated into ONE trajectory of the ONE MoveTo stage, so the stage is still a
// recorded free-space transit and validateFreeSpaceTransitWrist() examines the legs TOGETHER (the sum of the
// joint6 travel of all legs must stay <= pi, splitting a transit into legs cannot bypass the gate). This header
// holds only the pure parts (no ROS / MoveIt): candidate lists, the total-travel rule, the selection, the
// clearance floor, the contact-allowance refusal, the stop-at-join rule, the dense (JTC-style quintic) sampler, the
// lift-line check and the budget rule. Review 2026-10-22: see CONTEXT.md "Safe unwind detour".
#pragma once

#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace transit_unwind
{
constexpr double kPi = 3.14159265358979323846;
/// Same limit as box_h_cut_executor's validateFreeSpaceTransitWrist (kMaxWristTravelRad).
constexpr double kMaxWristTravelRad = kPi;
constexpr double kLiftStepM = 0.05;
constexpr double kMinMaxLiftM = 0.05;
constexpr double kMaxMaxLiftM = 0.30;
constexpr double kDefaultMaxLiftM = 0.20;

using Path = std::vector<std::vector<double>>;   // joint waypoints, one row each

/// transit_unwind_max_lift_m must be finite and within [0.05, 0.30] m.
inline bool maxLiftValid(double max_lift_m)
{
  return std::isfinite(max_lift_m) && max_lift_m >= kMinMaxLiftM - 1e-12 && max_lift_m <= kMaxMaxLiftM + 1e-12;
}

/// Lift heights tried, ascending: 0.05, 0.10, ... up to max_lift_m. Empty for an invalid maximum.
inline std::vector<double> liftCandidates(double max_lift_m)
{
  std::vector<double> lifts;
  if (!maxLiftValid(max_lift_m)) return lifts;
  for (int k = 1; k * kLiftStepM <= max_lift_m + 1e-9; ++k) lifts.push_back(k * kLiftStepM);
  return lifts;
}

/// Sum over consecutive waypoints of |dq| of one joint: exactly the formula of the wrist gate.
inline double jointTravel(const Path& path, std::size_t joint)
{
  double travel = 0.0;
  for (std::size_t i = 1; i < path.size(); ++i) {
    if (joint >= path[i].size() || joint >= path[i - 1].size()) return std::numeric_limits<double>::infinity();
    travel += std::abs(path[i][joint] - path[i - 1][joint]);
  }
  return travel;
}

/// Sum over consecutive waypoints of the L1 joint distance (all joints): the "total joint travel" score.
inline double totalJointTravel(const Path& path)
{
  double travel = 0.0;
  for (std::size_t i = 1; i < path.size(); ++i) {
    if (path[i].size() != path[i - 1].size()) return std::numeric_limits<double>::infinity();
    for (std::size_t j = 0; j < path[i].size(); ++j) travel += std::abs(path[i][j] - path[i - 1][j]);
  }
  return travel;
}

/// Concatenate legs into one path; the first waypoint of every later leg repeats the previous leg's last waypoint
/// and is dropped. A leg with fewer than 2 waypoints is ignored (a zero-length leg).
inline Path concatenateLegs(const std::vector<Path>& legs)
{
  Path out;
  for (const Path& leg : legs) {
    if (leg.size() < 2U) continue;
    if (out.empty()) {
      out = leg;
    } else {
      out.insert(out.end(), leg.begin() + 1, leg.end());
    }
  }
  return out;
}

/// A goal branch can only ever be reached by a legal transit when its NET joint6 change is within the gate (the
/// travel of any path is at least the net change), detour or not.
inline bool goalBranchReachable(double start_joint6, double goal_joint6)
{
  return std::isfinite(start_joint6) && std::isfinite(goal_joint6) &&
    std::abs(goal_joint6 - start_joint6) <= kMaxWristTravelRad;
}

/// Which wrist joints are set to the goal branch's values at the lifted pose.
enum class WristOption : int { kJoint6 = 0, kJoint4And6 = 1, kWholeWrist = 2 };
constexpr int kWristOptionCount = 3;

/// The waypoint at the lifted pose: `lifted` with the option's wrist joints taken from `goal`. j4/j5 may be
/// absent (the option then falls back to what is available).
inline std::vector<double> wristWaypoint(
  const std::vector<double>& lifted, const std::vector<double>& goal, std::optional<std::size_t> j4,
  std::optional<std::size_t> j5, std::size_t j6, WristOption option)
{
  std::vector<double> w = lifted;
  w[j6] = goal[j6];
  if (option == WristOption::kJoint4And6 || option == WristOption::kWholeWrist) {
    if (j4) w[*j4] = goal[*j4];
  }
  if (option == WristOption::kWholeWrist) {
    if (j5) w[*j5] = goal[*j5];
  }
  return w;
}

/// One fully planned and validated detour.
struct Candidate
{
  double lift_m{0.0};
  std::size_t goal_index{0};
  int option{0};
  double joint_travel{0.0};    // total L1 joint travel over all legs
  double joint6_travel{0.0};   // total joint6 travel over all legs (the gate quantity)
};

/// The acceptance rule: finite numbers and the TOTAL joint6 travel (all legs together) within the wrist gate.
inline bool acceptable(const Candidate& c)
{
  return std::isfinite(c.joint_travel) && std::isfinite(c.joint6_travel) && c.joint6_travel >= 0.0 &&
    c.joint6_travel <= kMaxWristTravelRad;
}

/// Strict ordering of acceptable candidates: smallest total joint travel, ties: lower lift, then goal index, then option.
inline bool betterCandidate(const Candidate& a, const Candidate& b)
{
  constexpr double kTie = 1e-9;
  if (a.joint_travel < b.joint_travel - kTie) return true;
  if (a.joint_travel > b.joint_travel + kTie) return false;
  if (a.lift_m < b.lift_m - kTie) return true;
  if (a.lift_m > b.lift_m + kTie) return false;
  if (a.goal_index != b.goal_index) return a.goal_index < b.goal_index;
  return a.option < b.option;
}

/// Indices of the acceptable candidates, best first (the order in which they are validated and the first one that
/// passes the full validation is used).
inline std::vector<std::size_t> rankCandidates(const std::vector<Candidate>& candidates)
{
  std::vector<std::size_t> order;
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    if (acceptable(candidates[i])) order.push_back(i);
  }
  std::stable_sort(order.begin(), order.end(), [&](std::size_t x, std::size_t y) {
    return betterCandidate(candidates[x], candidates[y]);
  });
  return order;
}

/// Index of the best acceptable candidate (front of rankCandidates), or nullopt when none is acceptable.
inline std::optional<std::size_t> selectBest(const std::vector<Candidate>& candidates)
{
  const auto order = rankCandidates(candidates);
  if (order.empty()) return std::nullopt;
  return order.front();
}

// ---- clearance floor (review finding 1) ----------------------------------------------------------------------------
/// Fixed minimum clearance, robot vs world AND robot vs robot, of the lifted pose, the wrist waypoint and every dense
/// sample of the wrist-rotation leg. Never lowered to suit a candidate.
constexpr double kClearanceFloorM = 0.005;
inline bool clearanceOk(double clearance_m)
{
  return std::isfinite(clearance_m) && clearance_m >= kClearanceFloorM;
}

// ---- contact allowance (finding 2) ---------------------------------------------------------------------------------
/// `allowed` lists (link, is a tool/carton contact allowed for that link) from the transit's START scene. Returns ""
/// when no link has an allowance, otherwise the refusal text naming the links. The detour is refused in that case.
inline std::string contactAllowanceRefusal(
  const std::vector<std::pair<std::string, bool>>& allowed, const std::string& carton_id)
{
  std::string links;
  for (const auto& entry : allowed) {
    if (!entry.second) continue;
    links += (links.empty() ? "" : ", ") + entry.first;
  }
  if (links.empty()) return std::string();
  return "tool/carton contact allowance is active in the transit's start scene (" + links + " vs " + carton_id +
    "): a detour would inherit it, refused";
}

// ---- stops at the joins (finding 3) --------------------------------------------------------------------------------
/// Velocity and acceleration of a join waypoint must be exactly zero (a full stop) for the legs to be independent.
inline bool isFullStop(const std::vector<double>& velocity, const std::vector<double>& acceleration, double tol = 0.0)
{
  for (const double v : velocity) {
    if (!std::isfinite(v) || std::abs(v) > tol) return false;
  }
  for (const double a : acceleration) {
    if (!std::isfinite(a) || std::abs(a) > tol) return false;
  }
  return true;
}

// ---- dense sampling of the controller's spline (finding 3) ---------------------------------------------------------
/// One waypoint as the controller sees it (joint_trajectory_controller, "splines" interpolation).
struct SplinePoint
{
  double time_s{0.0};
  std::vector<double> q;
  std::vector<double> v;  // empty = absent
  std::vector<double> a;  // empty = absent
};

/// Position at fraction tau in [0,1] of the segment between p0 and p1 (duration T = p1.time_s - p0.time_s), the way
/// the JTC does: quintic when both ends have velocity and acceleration, cubic Hermite with velocities only, linear
/// otherwise.
inline std::vector<double> sampleSegment(const SplinePoint& p0, const SplinePoint& p1, double tau)
{
  const double T = p1.time_s - p0.time_s;
  const double t = tau * T;
  const std::size_t n = p0.q.size();
  std::vector<double> out(n);
  const bool has_v = p0.v.size() == n && p1.v.size() == n;
  const bool has_a = has_v && p0.a.size() == n && p1.a.size() == n;
  for (std::size_t j = 0; j < n; ++j) {
    const double dq = p1.q[j] - p0.q[j];
    if (!(T > 0.0) || !has_v) {
      out[j] = p0.q[j] + tau * dq;
    } else if (!has_a) {
      const double c2 = (3.0 * dq - (2.0 * p0.v[j] + p1.v[j]) * T) / (T * T);
      const double c3 = (-2.0 * dq + (p0.v[j] + p1.v[j]) * T) / (T * T * T);
      out[j] = p0.q[j] + p0.v[j] * t + c2 * t * t + c3 * t * t * t;
    } else {
      const double T2 = T * T, T3 = T2 * T, T4 = T3 * T, T5 = T4 * T;
      const double c3 = (20.0 * dq - (8.0 * p1.v[j] + 12.0 * p0.v[j]) * T - (3.0 * p0.a[j] - p1.a[j]) * T2) / (2.0 * T3);
      const double c4 = (-30.0 * dq + (14.0 * p1.v[j] + 16.0 * p0.v[j]) * T + (3.0 * p0.a[j] - 2.0 * p1.a[j]) * T2) / (2.0 * T4);
      const double c5 = (12.0 * dq - 6.0 * (p1.v[j] + p0.v[j]) * T - (p0.a[j] - p1.a[j]) * T2) / (2.0 * T5);
      out[j] = p0.q[j] + p0.v[j] * t + 0.5 * p0.a[j] * t * t + c3 * t * t * t + c4 * t * t * t * t + c5 * t * t * t * t * t;
    }
  }
  return out;
}

constexpr double kDenseTimeStepS = 0.01;
constexpr double kDenseJointStepRad = 0.003;
/// Number of sub-intervals of one segment: at most kDenseTimeStepS and at most kDenseJointStepRad of joint change per
/// sample (estimated from the end-point difference and the end velocities).
inline std::size_t denseIntervals(const SplinePoint& p0, const SplinePoint& p1)
{
  const double T = p1.time_s - p0.time_s;
  double dq = 0.0;
  for (std::size_t j = 0; j < p0.q.size(); ++j) {
    dq = std::max(dq, std::abs(p1.q[j] - p0.q[j]));
    if (p0.v.size() == p0.q.size()) dq = std::max(dq, std::abs(p0.v[j]) * T);
    if (p1.v.size() == p1.q.size()) dq = std::max(dq, std::abs(p1.v[j]) * T);
  }
  const double by_time = T > 0.0 ? std::ceil(T / kDenseTimeStepS) : 1.0;
  const double by_joint = std::ceil(dq / kDenseJointStepRad);
  return static_cast<std::size_t>(std::max(1.0, std::max(by_time, by_joint)));
}

/// Dense samples of the whole trajectory along the spline. segment_of[i] is the index of the waypoint interval a
/// sample belongs to (the sample at waypoint k belongs to interval k, 0 for the first).
struct DenseSamples
{
  Path q;
  std::vector<std::size_t> interval_end;  // index of the waypoint that ends the interval the sample is in
};
inline DenseSamples sampleDense(const std::vector<SplinePoint>& points)
{
  DenseSamples out;
  if (points.empty()) return out;
  out.q.push_back(points.front().q);
  out.interval_end.push_back(0U);
  for (std::size_t k = 1; k < points.size(); ++k) {
    const std::size_t n = denseIntervals(points[k - 1], points[k]);
    for (std::size_t i = 1; i <= n; ++i) {
      out.q.push_back(i == n ? points[k].q : sampleSegment(points[k - 1], points[k], static_cast<double>(i) / static_cast<double>(n)));
      out.interval_end.push_back(k);
    }
  }
  return out;
}

/// Largest |dq| of any joint between consecutive samples.
inline double maxJointJump(const Path& path)
{
  double m = 0.0;
  for (std::size_t i = 1; i < path.size(); ++i) {
    for (std::size_t j = 0; j < path[i].size() && j < path[i - 1].size(); ++j) {
      m = std::max(m, std::abs(path[i][j] - path[i - 1][j]));
    }
  }
  return m;
}
constexpr double kMaxJointJumpRad = 0.05;

// ---- lift line (finding 5) -----------------------------------------------------------------------------------------
constexpr double kLiftLineTolM = 0.001;
constexpr double kLiftOrientTolRad = 1.0 * kPi / 180.0;
struct LiftLineCheck
{
  bool ok{false};
  double max_line_deviation_m{0.0};
  double max_orientation_rad{0.0};
  double end_error_m{0.0};
  std::string message;
};
/// Leg 1 must be a straight line from poses.front() along +direction (unit) of length lift_m with the orientation
/// held: TCP deviation from the line <= 1 mm, progress along the line within [-1 mm, lift + 1 mm], the last pose
/// within 1 mm of the lift target, orientation within 1 degree of the first pose.
inline LiftLineCheck checkLiftLine(const std::vector<Eigen::Isometry3d>& poses, const Eigen::Vector3d& direction, double lift_m)
{
  LiftLineCheck r;
  if (poses.size() < 2U || !(lift_m > 0.0) || !direction.allFinite() || std::abs(direction.norm() - 1.0) > 1e-6) {
    r.message = "lift line check: degenerate input";
    return r;
  }
  const Eigen::Vector3d origin = poses.front().translation();
  for (const auto& pose : poses) {
    const Eigen::Vector3d d = pose.translation() - origin;
    const double along = d.dot(direction);
    const double off = (d - along * direction).norm();
    r.max_line_deviation_m = std::max(r.max_line_deviation_m, off);
    const double angle = std::abs(Eigen::AngleAxisd(poses.front().linear().transpose() * pose.linear()).angle());
    r.max_orientation_rad = std::max(r.max_orientation_rad, angle);
    if (along < -kLiftLineTolM || along > lift_m + kLiftLineTolM) {
      r.message = "lift leaves the segment [0, lift] along +Z";
      return r;
    }
  }
  r.end_error_m = (poses.back().translation() - (origin + lift_m * direction)).norm();
  if (r.max_line_deviation_m > kLiftLineTolM) r.message = "lift deviates from the straight line";
  else if (r.max_orientation_rad > kLiftOrientTolRad) r.message = "lift does not hold the orientation";
  else if (r.end_error_m > kLiftLineTolM) r.message = "lift does not end at the lift target";
  else r.ok = true;
  return r;
}

// ---- time budget (finding 6) ---------------------------------------------------------------------------------------
/// Remaining part of the caller's budget (never negative).
inline double remainingBudget(double budget_s, double elapsed_s)
{
  return std::isfinite(budget_s) && std::isfinite(elapsed_s) ? std::max(0.0, budget_s - elapsed_s) : 0.0;
}
/// True when the deadline has been reached; a detour found or validated after this is never returned.
inline bool deadlineExpired(double budget_s, double elapsed_s)
{
  return remainingBudget(budget_s, elapsed_s) <= 0.0;
}

/// The WARN / plan-response text of an accepted detour.
inline std::string describe(
  double lift_m, double start_joint6, double waypoint_joint6, double goal_joint6, double total_joint6_travel,
  const std::string& direct_reason)
{
  std::ostringstream o;
  o << std::fixed << std::setprecision(1) << "transit detour via unwind waypoint: lift " << lift_m * 1000.0 << " mm"
    << std::setprecision(4) << ", joint6 " << start_joint6 << "->" << goal_joint6 << " rad (waypoint joint6 " << waypoint_joint6
    << "), total joint6 travel " << total_joint6_travel << " rad (direct refused: " << direct_reason << ")";
  return o.str();
}

}  // namespace transit_unwind
