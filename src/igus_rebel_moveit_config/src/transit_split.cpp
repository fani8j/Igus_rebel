#include "igus_rebel_moveit_config/transit_split.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>

namespace rebel_transit_split
{

namespace
{
constexpr double kSEps = 1e-3;  // slack on the chord parameter
}  // namespace

JointLinearResult IsJointLinear(const Waypoints& waypoints, double tol)
{
  JointLinearResult result;
  if (!std::isfinite(tol) || tol < 0.0) {
    result.reason = "invalid tolerance";
    return result;
  }
  if (waypoints.size() < 2U) {
    result.reason = "fewer than 2 waypoints";
    return result;
  }
  const std::size_t width = waypoints.front().size();
  if (width == 0U) {
    result.reason = "no joints";
    return result;
  }
  for (const auto & point : waypoints) {
    if (point.size() != width) {
      result.reason = "waypoints have different joint counts";
      return result;
    }
    for (const double value : point) {
      if (!std::isfinite(value)) {
        result.reason = "non-finite waypoint value";
        return result;
      }
    }
  }
  const auto & first = waypoints.front();
  const auto & last = waypoints.back();
  std::vector<double> chord(width);
  double length_sq = 0.0;
  for (std::size_t j = 0; j < width; ++j) {
    chord[j] = last[j] - first[j];
    length_sq += chord[j] * chord[j];
  }
  if (!(length_sq > 1e-12)) {
    result.reason = "zero-length path";
    return result;
  }
  double previous_s = 0.0;
  for (std::size_t i = 0; i < waypoints.size(); ++i) {
    double dot = 0.0;
    for (std::size_t j = 0; j < width; ++j) dot += (waypoints[i][j] - first[j]) * chord[j];
    const double s = dot / length_sq;
    if (s < -kSEps || s > 1.0 + kSEps) {
      result.reason = "waypoint " + std::to_string(i) + " lies outside the first-last segment";
      return result;
    }
    if (i > 0U && s < previous_s - kSEps) {
      result.reason = "path is not monotone at waypoint " + std::to_string(i);
      return result;
    }
    previous_s = std::max(previous_s, s);
    for (std::size_t j = 0; j < width; ++j) {
      const double deviation = std::abs(waypoints[i][j] - (first[j] + s * chord[j]));
      result.max_chord_deviation = std::max(result.max_chord_deviation, deviation);
    }
  }
  if (result.max_chord_deviation > tol) {
    result.reason = "path deviates " + std::to_string(result.max_chord_deviation) +
      " rad from the straight joint line (tolerance " + std::to_string(tol) + ")";
    return result;
  }
  result.linear = true;
  return result;
}

double MaxJointChange(const std::vector<double> & first, const std::vector<double> & last)
{
  if (first.size() != last.size() || first.empty()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  double largest = 0.0;
  for (std::size_t j = 0; j < first.size(); ++j) {
    if (!std::isfinite(first[j]) || !std::isfinite(last[j])) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    largest = std::max(largest, std::abs(last[j] - first[j]));
  }
  return largest;
}

std::vector<double> ReorderToJointOrder(
  const std::vector<std::string> & names, const std::vector<double> & values,
  const std::vector<std::string> & order)
{
  if (names.size() != values.size() || names.size() != order.size() || names.empty()) return {};
  if (std::set<std::string>(names.begin(), names.end()).size() != names.size()) return {};
  std::vector<double> reordered;
  reordered.reserve(order.size());
  for (const auto & wanted : order) {
    const auto it = std::find(names.begin(), names.end(), wanted);
    if (it == names.end()) return {};
    reordered.push_back(values[static_cast<std::size_t>(std::distance(names.begin(), it))]);
  }
  return reordered;
}

Waypoints ReorderWaypoints(
  const std::vector<std::string> & names, const Waypoints & waypoints,
  const std::vector<std::string> & order)
{
  Waypoints reordered;
  reordered.reserve(waypoints.size());
  for (const auto & point : waypoints) {
    auto row = ReorderToJointOrder(names, point, order);
    if (row.empty()) return {};
    reordered.push_back(std::move(row));
  }
  return reordered;
}

LeafVerdict EvaluateLeaf(
  const LeafCandidate & leaf, const std::vector<std::string> & arm_joints, double min_length_rad,
  double linear_tolerance)
{
  LeafVerdict verdict;
  if (!leaf.recorded_transit) {
    verdict.reason = "not a recorded free-space transit stage";
    return verdict;
  }
  if (!leaf.provenance_registered) {
    verdict.reason = "trajectory was not produced by the joint-interpolation planner";
    return verdict;
  }
  if (!leaf.scene_diff_robot_state_only) {
    verdict.reason = "scene_diff changes more than the robot state";
    return verdict;
  }
  if (arm_joints.size() != 6U || leaf.joint_names.size() != 6U) {
    verdict.reason = "trajectory does not have exactly the 6 arm joints";
    return verdict;
  }
  const Waypoints ordered = ReorderWaypoints(leaf.joint_names, leaf.waypoints, arm_joints);
  if (ordered.empty()) {
    verdict.reason = "trajectory joints differ from the arm joints";
    return verdict;
  }
  const auto linear = IsJointLinear(ordered, linear_tolerance);
  if (!linear.linear) {
    verdict.reason = "not joint-linear: " + linear.reason;
    return verdict;
  }
  const double change = MaxJointChange(ordered.front(), ordered.back());
  if (!std::isfinite(change) || change < min_length_rad) {
    verdict.reason = "shorter than the minimum CRI transit length (" +
      std::to_string(std::isfinite(change) ? change : 0.0) + " rad)";
    return verdict;
  }
  verdict.cri = true;
  return verdict;
}

std::vector<Run> BuildDispatchPlan(const std::vector<bool> & cri_leaf)
{
  std::vector<Run> runs;
  for (std::size_t i = 0; i < cri_leaf.size(); ++i) {
    if (cri_leaf[i]) {
      runs.push_back(Run{true, i, i + 1U});
    } else if (!runs.empty() && !runs.back().cri) {
      runs.back().end = i + 1U;
    } else {
      runs.push_back(Run{false, i, i + 1U});
    }
  }
  return runs;
}

double PercentFromScaling(double scaling, double cap_percent)
{
  if (!std::isfinite(scaling) || !std::isfinite(cap_percent) || cap_percent < 1.0) return 1.0;
  return std::clamp(scaling * 100.0, 1.0, cap_percent);
}

double PathToleranceRad(double largest_joint_delta_rad)
{
  const double delta = std::isfinite(largest_joint_delta_rad) ? std::abs(largest_joint_delta_rad) : 0.0;
  return std::max(0.03, 0.03 * delta);
}

double EstimateDurationS(
  double max_delta_rad, double velocity_percent, double acceleration_percent,
  double reference_velocity_rad_s, double reference_acceleration_rad_s2)
{
  const double inf = std::numeric_limits<double>::infinity();
  if (!std::isfinite(max_delta_rad) || max_delta_rad < 0.0 || !(velocity_percent > 0.0) ||
      !(acceleration_percent > 0.0) || !(reference_velocity_rad_s > 0.0) ||
      !(reference_acceleration_rad_s2 > 0.0) || !std::isfinite(velocity_percent) ||
      !std::isfinite(acceleration_percent)) {
    return inf;
  }
  const double v = reference_velocity_rad_s * velocity_percent / 100.0;
  const double a = reference_acceleration_rad_s2 * acceleration_percent / 100.0;
  const double ramp_distance = v * v / a;
  if (max_delta_rad >= ramp_distance) return max_delta_rad / v + v / a;
  return 2.0 * std::sqrt(max_delta_rad / a);
}

}  // namespace rebel_transit_split
