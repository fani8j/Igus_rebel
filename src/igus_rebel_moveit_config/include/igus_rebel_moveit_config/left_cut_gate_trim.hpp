// Pure (ROS/MoveIt-free, Eigen only) decision maths of the "left cut gate trim" of
// box_h_cut_executor: when the final trajectory collision gate finds a contact inside the forward
// left cut stroke of the varying corner-to-corner search, the cut is shortened at the END NEARER to
// the contact and planned again (planning parameters left_cut_gate_trim_max_retries and
// left_cut_gate_trim_clearance_m).
//
// Everything is in TRAVEL coordinates of the left varying search: s = 0 at the first point the blade
// reaches (LeftVaryingPlan::span_origin), growing along LeftVaryingPlan::travel_dir. "Start" is the
// travel start, "end" the travel end; the physical back/front name of an end depends on the travel
// direction (forward: start = back, end = front; reverse: start = front, end = back).
//
// The limits only REMOVE stations from the search (a station that is not allowed has no nodes), so
// every requirement of the search (seam band coverage, minimum length, blade-angle caps, travel
// gates ...) and every validator after it stays in force on the shortened plan.
#pragma once

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

namespace left_cut_gate_trim
{

constexpr double kStationToleranceM = 1e-9;  // a station exactly on a limit is allowed
constexpr double kContactToleranceM = 2e-3;  // a contact this far outside the kept interval is clamped
constexpr double kProgressM = 1e-4;          // every trim removes at least the current end station

enum class End { kStart, kEnd };

// Extra limits on the kept interval (travel coordinates, metres). Infinite = no limit at that end.
struct Limits
{
  double min_start_m{-std::numeric_limits<double>::infinity()};
  double max_end_m{std::numeric_limits<double>::infinity()};
  bool hasStart() const { return std::isfinite(min_start_m); }
  bool hasEnd() const { return std::isfinite(max_end_m); }
  bool any() const { return hasStart() || hasEnd(); }
  // May a chain keep the station at travel coordinate s_m?
  bool stationAllowed(double s_m) const
  {
    return s_m >= min_start_m - kStationToleranceM && s_m <= max_end_m + kStationToleranceM;
  }
};

inline std::string validateParams(std::int64_t max_retries, double clearance_m)
{
  if (max_retries < 0 || max_retries > 5) {
    return "left_cut_gate_trim_max_retries must be within [0,5] (0 = refuse the plan as before)";
  }
  if (!std::isfinite(clearance_m) || clearance_m < 0.0 || clearance_m > 0.05) {
    return "left_cut_gate_trim_clearance_m must be finite and within [0,0.05] m";
  }
  return std::string();
}

// Travel coordinate of a point: projection onto the travel direction from the span origin.
inline double travelCoordinate(
  const Eigen::Vector3d& point, const Eigen::Vector3d& span_origin, const Eigen::Vector3d& travel_dir)
{
  return (point - span_origin).dot(travel_dir);
}

// The stroke stage the trim reacts to: the forward contact stroke "cut left side cut [..]", not the
// reverse patch stroke ("cut left side cut patch [..]"), not entry / finish / transit / withdraw.
inline bool isForwardLeftStrokeStage(const std::string& stage_name)
{
  const std::string stroke = "cut left side cut";
  const std::string patch = "cut left side cut patch";
  return stage_name.rfind(stroke, 0) == 0U && stage_name.rfind(patch, 0) != 0U;
}

// Physical name of a travel end.
inline const char* physicalEndName(End end, bool reverse_travel)
{
  const bool front = (end == End::kStart) == reverse_travel;
  return front ? "front" : "back";
}

struct Request
{
  Limits current;              // limits of the attempt that failed (infinite on the first failure)
  int retries_done{0};         // trims already applied
  int max_retries{0};          // left_cut_gate_trim_max_retries
  double contact_s_m{0.0};     // travel coordinate of the colliding tool_link position
  double kept_lo_m{0.0};       // kept interval of the failed plan (travel coordinates)
  double kept_hi_m{0.0};
  double clearance_m{0.0};     // left_cut_gate_trim_clearance_m
  double min_kept_length_m{0.0};  // the search's minimum kept length (a cheap early give-up)
  double span_length_m{0.0};      // extended span length (travel coordinates 0 .. span_length_m)
};

struct Decision
{
  bool trim{false};              // false: give up (reason), the plan stays refused
  End end{End::kEnd};            // the end that is shortened
  Limits limits;                 // the accumulated limits for the next attempt
  double contact_s_m{0.0};       // the contact, clamped into the kept interval
  double removed_at_least_m{0.0};  // how much of the failed kept interval the new limit removes
  std::string reason;            // give-up reason, or empty
};

inline Decision decide(const Request& request)
{
  Decision decision;
  decision.limits = request.current;
  const auto give_up = [&](const std::string& reason) {
      decision.trim = false;
      decision.reason = reason;
      return decision;
    };
  if (request.max_retries <= 0) return give_up("left_cut_gate_trim_max_retries is 0");
  if (request.retries_done >= request.max_retries) {
    return give_up(
      "retry budget exhausted (" + std::to_string(request.retries_done) + " of " +
      std::to_string(request.max_retries) + ")");
  }
  if (!std::isfinite(request.contact_s_m) || !std::isfinite(request.kept_lo_m) ||
      !std::isfinite(request.kept_hi_m) || !(request.kept_hi_m >= request.kept_lo_m) ||
      !std::isfinite(request.clearance_m) || request.clearance_m < 0.0) {
    return give_up("the contact position or the kept interval is not finite");
  }
  if (request.contact_s_m < request.kept_lo_m - kContactToleranceM ||
      request.contact_s_m > request.kept_hi_m + kContactToleranceM) {
    return give_up("the contact lies outside the kept stroke interval (not a stroke contact)");
  }
  const double contact = std::min(request.kept_hi_m, std::max(request.kept_lo_m, request.contact_s_m));
  decision.contact_s_m = contact;
  // The end nearer to the contact; a tie shortens the travel end.
  decision.end = (contact - request.kept_lo_m) < (request.kept_hi_m - contact) ?
    End::kStart : End::kEnd;
  if (decision.end == End::kEnd) {
    const double limit = std::min(contact - request.clearance_m, request.kept_hi_m - kProgressM);
    decision.limits.max_end_m = std::min(request.current.max_end_m, limit);
    decision.removed_at_least_m = request.kept_hi_m - decision.limits.max_end_m;
  } else {
    const double limit = std::max(contact + request.clearance_m, request.kept_lo_m + kProgressM);
    decision.limits.min_start_m = std::max(request.current.min_start_m, limit);
    decision.removed_at_least_m = decision.limits.min_start_m - request.kept_lo_m;
  }
  const double window_lo = decision.limits.hasStart() ? decision.limits.min_start_m : 0.0;
  const double window_hi = decision.limits.hasEnd() ? decision.limits.max_end_m : request.span_length_m;
  if (!(window_hi - window_lo >= request.min_kept_length_m - kStationToleranceM)) {
    decision.trim = false;
    decision.reason = "the allowed window would be only " +
      std::to_string(std::max(0.0, window_hi - window_lo) * 1000.0) +
      " mm, below the minimum kept length " + std::to_string(request.min_kept_length_m * 1000.0) + " mm";
    return decision;
  }
  decision.trim = true;
  return decision;
}

}  // namespace left_cut_gate_trim
