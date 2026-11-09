// Pure (ROS/MoveIt-free; Eigen only through rebel_analytic_ik) joint-travel check for ONE left_parallel contact
// stroke of box_h_cut_executor (entry, cut, finish or lift).
//
// The Cartesian contact strokes run with jump detection off, so a wrist or elbow flip inside a
// stroke would otherwise be accepted. The check sums |dq| of every joint over the stroke's
// waypoint list and refuses a stroke whose aggregate travel of any joint exceeds
// kMaxJointTravelRad. The SAME function is used by the varying-orientation search replay
// (searchLeftVaryingOrientationPass) and by validateLeftContactJointTravel, so the search can
// never emit a stroke the validator refuses.
//
// The 1.0 rad limit is a wrist-flip guard sized for 25-60 mm constant-orientation strokes; a
// ~270 mm corner-to-corner stroke can move joint2/joint3 further legitimately (2026-10-16
// measurements). checkJointTravel / kMaxJointTravelRad are the default `aggregate` mode and stay
// unchanged.
//
// Opt-in `split` mode (left_contact_travel_mode = split, checkSplitTravel below). It replaces the
// single aggregate bound by one rule per failure cause. What is and is NOT guaranteed:
//   - joint4, joint5 and joint6 keep the aggregate travel limit (1.0 rad); joints 1-3 are capped
//     by arm_travel_max_rad (a coarse bound, NOT a flip detector);
//   - the Pieper IK branch label (shoulder x elbow x wrist, rebel_analytic_ik::classifyBranch) is
//     evaluated at the stroke's SAMPLED waypoints and must not change between them, and every
//     sampled waypoint keeps a margin to each branch boundary (|joint5|, |joint3 - beta|, |X|);
//   - between two waypoints: joint3 and joint5 are the boundary coordinates of the elbow and wrist
//     labels and the executor interpolates them linearly, so equal signs and margins at both
//     waypoints give the same margins for the whole segment; the shoulder coordinate
//     X(q2, q3) is NOT linear, so the margin between waypoints is proven with a Lipschitz bound
//     and subdivision (shoulderSegmentClear), and a segment that cannot be proven is refused;
//   - a branch flip that happens entirely between two waypoints of a jump larger than the
//     margins is caught by the label comparison (labels differ) or by the travel rules, but
//     nothing here inspects a joint-space path that is not linear between the waypoints.
// The SAME function is used by the search replay and by validateLeftContactJointTravel.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "igus_rebel_moveit_config/rebel_analytic_ik.hpp"

namespace left_contact_travel
{

// Existing aggregate joint-travel bound for one left contact stroke.
constexpr double kMaxJointTravelRad = 1.0;

// Sum over consecutive waypoints of |q[j] - q_previous[j]|, one entry per joint (the size of the
// first waypoint). Empty for an empty waypoint list.
inline std::vector<double> aggregateJointTravel(const std::vector<std::vector<double> >& waypoints)
{
  std::vector<double> travel;
  for (std::size_t w = 0; w < waypoints.size(); ++w) {
    const std::vector<double>& values = waypoints[w];
    if (w == 0U) travel.assign(values.size(), 0.0);
    for (std::size_t j = 0; j < values.size() && j < travel.size() && w > 0U; ++j) {
      if (j < waypoints[w - 1U].size()) travel[j] += std::abs(values[j] - waypoints[w - 1U][j]);
    }
  }
  return travel;
}

struct Result
{
  bool ok{true};
  std::size_t joint{0};                 // 0-based index of the FIRST joint above the limit
  double travel_rad{0.0};               // its aggregate travel
  std::vector<double> per_joint_travel; // every joint's aggregate travel
};

// ok = false (first offending joint reported) when any joint's aggregate travel is above
// `limit_rad` (strictly greater: exactly the limit passes).
inline Result checkJointTravel(
  const std::vector<std::vector<double> >& waypoints, double limit_rad = kMaxJointTravelRad)
{
  Result result;
  result.per_joint_travel = aggregateJointTravel(waypoints);
  for (std::size_t j = 0; j < result.per_joint_travel.size(); ++j) {
    if (result.per_joint_travel[j] > limit_rad) {
      result.ok = false;
      result.joint = j;
      result.travel_rad = result.per_joint_travel[j];
      return result;
    }
  }
  return result;
}

// "joint<N> <travel, 3 decimals>" (N is 1-based): the part of the refusal text both callers share;
// each caller adds its own wording around it.
inline std::string jointTravelText(const Result& result)
{
  std::ostringstream text;
  text << "joint" << (result.joint + 1U) << " " << std::fixed << std::setprecision(3)
       << result.travel_rad;
  return text.str();
}

// Search attribution: which end of the stroke to trim when joint `joint` exceeds the limit. The
// "head" is the travel of that joint over the first `station_count`-th share of the waypoints
// (k = max(1, n / station_count) waypoint steps) and the "tail" the travel over the last k steps;
// the end that moves the joint more is the one to shorten.
struct HeadTail
{
  double head{0.0}, tail{0.0};
  bool trimStart() const {return head > tail;}
};
inline HeadTail headTailTravel(
  const std::vector<std::vector<double> >& waypoints, std::size_t joint, std::size_t station_count)
{
  HeadTail result;
  const std::size_t n = waypoints.size();
  const std::size_t k = std::max<std::size_t>(1U, n / std::max<std::size_t>(station_count, 1U));
  for (std::size_t w = 1; w < n && w <= k; ++w) {
    result.head += std::abs(waypoints[w][joint] - waypoints[w - 1U][joint]);
  }
  for (std::size_t w = n > k ? n - k : 1U; w < n; ++w) {
    result.tail += std::abs(waypoints[w][joint] - waypoints[w - 1U][joint]);
  }
  return result;
}

// ---------------------------------------------------------------------------------------------
// Split mode
// ---------------------------------------------------------------------------------------------

enum class Mode { kAggregate, kSplit };

inline std::optional<Mode> modeFromString(const std::string& text)
{
  if (text == "aggregate") return Mode::kAggregate;
  if (text == "split") return Mode::kSplit;
  return std::nullopt;
}

struct SplitLimits
{
  double wrist_travel_max_rad{kMaxJointTravelRad};  // joint4, joint5 and joint6 aggregate travel
  double arm_travel_max_rad{2.0};                   // joints 1-3 aggregate travel
  double min_abs_joint5_rad{0.2617993877991494};    // 15 deg; wrist singularity margin
  double min_elbow_margin_rad{0.0872664625997165};  // 5 deg; |joint3 - beta| (straight arm)
  double min_shoulder_margin_m{0.05};               // |X| of the wrist centre (joint1 axis)
};

enum class Rule
{
  kNone,
  kInvalidInput,       // wrong joint count or non-finite value: fail closed
  kBranchChange,       // shoulder or elbow Pieper branch label changed along the stroke
  kJoint5SignChange,   // joint5 (= wrist label) changed sign
  kJoint5Margin,       // |joint5| below the margin
  kElbowMargin,        // |joint3 - beta| below the margin
  kShoulderMargin,     // |X| below the margin (at a waypoint or between two waypoints)
  kWristTravel,        // joint4, joint5 or joint6 aggregate travel
  kArmTravel           // joint1, 2 or 3 aggregate travel
};

inline const char* ruleName(Rule rule)
{
  switch (rule) {
    case Rule::kNone: return "none";
    case Rule::kInvalidInput: return "invalid_input";
    case Rule::kBranchChange: return "branch_change";
    case Rule::kJoint5SignChange: return "joint5_sign_change";
    case Rule::kJoint5Margin: return "joint5_margin";
    case Rule::kElbowMargin: return "elbow_margin";
    case Rule::kShoulderMargin: return "shoulder_margin";
    case Rule::kWristTravel: return "wrist_travel";
    case Rule::kArmTravel: return "arm_travel";
  }
  return "unknown";
}

inline bool isTravelRule(Rule rule)
{
  return rule == Rule::kWristTravel || rule == Rule::kArmTravel;
}

// Measurements over the whole stroke, independent of any limit (also filled when a rule fails).
// They are taken at the sampled waypoints (the shoulder margin also between waypoints in
// checkSplitTravel's verdict, not here).
struct StrokeStats
{
  std::vector<double> per_joint_travel;
  double min_abs_joint5_rad{std::numeric_limits<double>::infinity()};
  std::size_t min_abs_joint5_waypoint{0};
  rebel_analytic_ik::BranchLabel branch;       // of the first waypoint
  bool branch_constant{true};                  // shoulder, elbow and wrist labels never change
  // Smallest margin to each branch boundary over the waypoints (shoulder in m, elbow/wrist in rad).
  double min_shoulder_margin_m{std::numeric_limits<double>::infinity()};
  double min_elbow_margin_rad{std::numeric_limits<double>::infinity()};
};

struct SplitResult
{
  bool ok{true};
  Rule rule{Rule::kNone};
  std::size_t joint{0};          // 0-based offending joint (branch_change: shoulder -> 0, elbow -> 2)
  std::size_t waypoint{0};       // waypoint index where the rule first fails (the later one of a segment)
  bool between_waypoints{false}; // failure proven only on the segment (waypoint - 1, waypoint)
  std::size_t waypoint_count{0};
  double value{0.0};             // travel / margin value / signed boundary coordinate at the failure
  double reference{0.0};         // the limit (travel, margin) or the first waypoint's value
  const char* what{""};          // "shoulder" / "elbow" for branch_change
  StrokeStats stats;
};

namespace detail
{
inline bool toJointVector(const std::vector<double>& values, rebel_analytic_ik::JointVector& out)
{
  if (values.size() != out.size()) return false;
  for (std::size_t j = 0; j < out.size(); ++j) {
    if (!std::isfinite(values[j])) return false;
    out[j] = values[j];
  }
  return true;
}

inline std::string fixed(double value, int digits)
{
  std::ostringstream text;
  text << std::fixed << std::setprecision(digits) << value;
  return text.str();
}

constexpr std::size_t kNpos = static_cast<std::size_t>(-1);
}  // namespace detail

// Position of joint1..joint6 in a joint-variable list (e.g. a MoveIt JointModelGroup's
// getVariableNames()). Fails closed (nullopt, `error` set) when a name is missing or duplicated.
// The pure functions below take waypoints in joint1..joint6 order; reorder with orderWaypoints.
inline std::optional<std::array<std::size_t, 6> > jointIndexByName(
  const std::vector<std::string>& variable_names, std::string& error)
{
  static const std::array<const char*, 6> kNames{
    "joint1", "joint2", "joint3", "joint4", "joint5", "joint6"};
  std::array<std::size_t, 6> index{};
  for (std::size_t j = 0; j < 6U; ++j) {
    std::size_t found = detail::kNpos, count = 0;
    for (std::size_t v = 0; v < variable_names.size(); ++v) {
      if (variable_names[v] == kNames[j]) {
        found = v;
        ++count;
      }
    }
    if (count != 1U) {
      error = std::string("planning group has ") + (count == 0U ? "no " : "more than one ") +
        kNames[j] + "; the split joint-travel gate needs joint1..joint6 by name";
      return std::nullopt;
    }
    index[j] = found;
  }
  return index;
}

// Waypoints in group-variable order -> joint1..joint6 order. nullopt (error set) when a waypoint
// does not have exactly one value per variable.
inline std::optional<std::vector<std::vector<double> > > orderWaypoints(
  const std::vector<std::vector<double> >& waypoints, const std::array<std::size_t, 6>& index,
  std::size_t variable_count, std::string& error)
{
  std::vector<std::vector<double> > ordered;
  ordered.reserve(waypoints.size());
  for (std::size_t w = 0; w < waypoints.size(); ++w) {
    if (waypoints[w].size() != variable_count) {
      error = "waypoint " + std::to_string(w) + " has " + std::to_string(waypoints[w].size()) +
        " values for " + std::to_string(variable_count) + " group variables";
      return std::nullopt;
    }
    std::vector<double> q(6);
    for (std::size_t j = 0; j < 6U; ++j) q[j] = waypoints[w][index[j]];
    ordered.push_back(std::move(q));
  }
  return ordered;
}

// Per-segment facts the stroke-coverage expectation depends on (mirrors CutSegment).
struct SegmentFacts
{
  bool is_left{false};              // cut == CutKind::kLeft
  bool hover_only{false};           // segment has no contact stroke at all
  bool force_fixed_orientation{false};  // fixed-orientation left (not the parallel contact family)
  bool stop_after_start{false};     // entry stroke only (no cut stroke)
};

struct StrokeExpectation
{
  bool contact{false};  // at least one left contact stroke must have been examined
  bool cut{false};      // the left CUT stroke must have been examined
};

// Which left contact strokes a plan must contain. An approach-only plan (`stop_at_hover_start`,
// left_debug_stop_at_approach) ends at the hover pose after the pre-cut dwell and by construction
// has NO contact stroke (no entry/cut/finish/lift), so it expects none; the full plan it was
// derived from is validated with the normal expectation before it is derived. Every stroke that
// IS present in an approach-only plan is still checked individually by the caller. Normal plans
// are unchanged: a left parallel segment expects its entry stroke, and its cut stroke unless it is
// entry-only.
inline StrokeExpectation expectedLeftStrokes(
  const std::vector<SegmentFacts>& segments, bool stop_at_hover_start)
{
  StrokeExpectation expectation;
  if (stop_at_hover_start) return expectation;
  for (const auto& segment : segments) {
    if (!segment.is_left || segment.hover_only || segment.force_fixed_orientation) continue;
    expectation.contact = true;
    if (!segment.stop_after_start) expectation.cut = true;
  }
  return expectation;
}

// Stroke coverage: with split mode on, a plan that contains a left contact stroke must have had
// strokes examined, and a plan that contains the left CUT must have examined its cut stroke.
// Returns an error text or an empty string.
inline std::string coverageError(
  bool expect_left_contact, bool expect_left_cut, std::size_t examined, bool cut_seen)
{
  if (expect_left_contact && examined == 0U) {
    return "the plan contains a left side cut but no left contact stroke was found to check; "
      "refusing to guess";
  }
  if (expect_left_cut && !cut_seen) {
    return "the plan contains a left side cut but its cut stroke was not found among the "
      "examined strokes; refusing to guess";
  }
  return std::string();
}

// Stroke measurements (6-joint waypoints; other sizes are skipped here and reported by
// checkSplitTravel as invalid input).
inline StrokeStats strokeStats(const std::vector<std::vector<double> >& waypoints)
{
  StrokeStats stats;
  stats.per_joint_travel = aggregateJointTravel(waypoints);
  bool have_first = false;
  for (std::size_t w = 0; w < waypoints.size(); ++w) {
    rebel_analytic_ik::JointVector q{};
    if (!detail::toJointVector(waypoints[w], q)) continue;
    const auto label = rebel_analytic_ik::classifyBranch(q);
    const auto margins = rebel_analytic_ik::branchMargins(q);
    if (!have_first) {
      stats.branch = label;
      have_first = true;
    } else if (label != stats.branch) {
      stats.branch_constant = false;
    }
    if (std::abs(q[4]) < stats.min_abs_joint5_rad) {
      stats.min_abs_joint5_rad = std::abs(q[4]);
      stats.min_abs_joint5_waypoint = w;
    }
    stats.min_shoulder_margin_m = std::min(stats.min_shoulder_margin_m, margins.shoulder_m);
    stats.min_elbow_margin_rad = std::min(stats.min_elbow_margin_rad, margins.elbow_rad);
  }
  return stats;
}

// First waypoint index at which the cumulative |dq| of joint `j` exceeds `limit` (kNpos if never).
inline std::size_t firstTravelExceedance(
  const std::vector<std::vector<double> >& waypoints, std::size_t j, double limit)
{
  double travel = 0.0;
  for (std::size_t w = 1; w < waypoints.size(); ++w) {
    travel += std::abs(waypoints[w][j] - waypoints[w - 1U][j]);
    if (travel > limit) return w;
  }
  return detail::kNpos;
}

// True when |X(q2, q3)| >= margin (and X keeps its sign, so no zero crossing) along the straight
// joint-space segment a -> b. X is nonlinear, so this is proven, not sampled: with Xa, Xb at the
// ends and B = wristRadialXVariation(dq2, dq3), |X| on the segment is at least
// (|Xa| + |Xb| - B) / 2; when that is below the margin the segment is split in half and both
// halves are checked (depth limit 24 -> false: unproven means refused). Requires the signs at the
// two ends to be equal and both ends to satisfy the margin.
inline bool shoulderSegmentClear(
  double q2a, double q3a, double q2b, double q3b, double margin, int depth = 0)
{
  const double xa = rebel_analytic_ik::wristRadialX(q2a, q3a);
  const double xb = rebel_analytic_ik::wristRadialX(q2b, q3b);
  if ((xa >= 0.0) != (xb >= 0.0)) return false;
  if (std::abs(xa) < margin || std::abs(xb) < margin) return false;
  const double bound = rebel_analytic_ik::wristRadialXVariation(q2b - q2a, q3b - q3a);
  if ((std::abs(xa) + std::abs(xb) - bound) / 2.0 >= std::max(margin, 1e-9)) return true;
  if (depth >= 24) return false;
  const double q2m = 0.5 * (q2a + q2b), q3m = 0.5 * (q3a + q3b);
  return shoulderSegmentClear(q2a, q3a, q2m, q3m, margin, depth + 1) &&
         shoulderSegmentClear(q2m, q3m, q2b, q3b, margin, depth + 1);
}

// The split gate. Every rule is evaluated; the rule whose violation occurs at the EARLIEST waypoint
// is reported (ties: branch_change, joint5_sign_change, joint5_margin, elbow_margin,
// shoulder_margin, wrist_travel, arm_travel). Strictly-greater semantics for the travel limits
// (exactly the limit passes), strictly-below for the margins (exactly the margin passes). Boundary
// values: a coordinate of exactly 0 (j5 = 0, j3 = beta, X = 0) takes the + label and has margin 0,
// so it is refused by its margin rule whenever that margin is > 0. Waypoints are joint1..joint6.
inline SplitResult checkSplitTravel(
  const std::vector<std::vector<double> >& waypoints, const SplitLimits& limits)
{
  using detail::kNpos;
  SplitResult result;
  result.waypoint_count = waypoints.size();
  for (std::size_t w = 0; w < waypoints.size(); ++w) {
    rebel_analytic_ik::JointVector q{};
    if (!detail::toJointVector(waypoints[w], q)) {
      result.ok = false;
      result.rule = Rule::kInvalidInput;
      result.waypoint = w;
      return result;
    }
  }
  result.stats = strokeStats(waypoints);
  if (waypoints.empty()) return result;

  std::size_t best = kNpos;
  const auto offer = [&](Rule rule, std::size_t waypoint, std::size_t joint, double value,
                         double reference, const char* what, bool between = false) {
      if (waypoint == kNpos || waypoint >= best) return;
      best = waypoint;
      result.ok = false;
      result.rule = rule;
      result.waypoint = waypoint;
      result.between_waypoints = between;
      result.joint = joint;
      result.value = value;
      result.reference = reference;
      result.what = what;
    };

  rebel_analytic_ik::JointVector first{};
  detail::toJointVector(waypoints.front(), first);
  const auto first_label = rebel_analytic_ik::classifyBranch(first);
  const auto first_margins = rebel_analytic_ik::branchMargins(first);
  const double first_shoulder = first_label.shoulder * first_margins.shoulder_m;
  const double first_elbow = first_label.elbow * first_margins.elbow_rad;
  std::size_t branch_w = kNpos, sign_w = kNpos, j5_w = kNpos, elbow_w = kNpos, shoulder_w = kNpos;
  std::size_t branch_joint = 0;
  bool shoulder_between = false;
  const char* branch_what = "";
  double branch_value = 0.0, branch_reference = 0.0, sign_value = 0.0, elbow_value = 0.0,
    shoulder_value = 0.0;
  for (std::size_t w = 0; w < waypoints.size(); ++w) {
    rebel_analytic_ik::JointVector q{};
    detail::toJointVector(waypoints[w], q);
    const auto label = rebel_analytic_ik::classifyBranch(q);
    const auto margins = rebel_analytic_ik::branchMargins(q);
    if (branch_w == kNpos) {
      if (label.shoulder != first_label.shoulder) {
        branch_w = w;
        branch_what = "shoulder";
        branch_joint = 0U;
        branch_value = label.shoulder * margins.shoulder_m;
        branch_reference = first_shoulder;
      } else if (label.elbow != first_label.elbow) {
        branch_w = w;
        branch_what = "elbow";
        branch_joint = 2U;
        branch_value = label.elbow * margins.elbow_rad;
        branch_reference = first_elbow;
      }
    }
    if (sign_w == kNpos && label.wrist != first_label.wrist) {
      sign_w = w;
      sign_value = q[4];
    }
    if (j5_w == kNpos && std::abs(q[4]) < limits.min_abs_joint5_rad) j5_w = w;
    if (elbow_w == kNpos && margins.elbow_rad < limits.min_elbow_margin_rad) {
      elbow_w = w;
      elbow_value = margins.elbow_rad;
    }
    if (shoulder_w == kNpos) {
      if (margins.shoulder_m < limits.min_shoulder_margin_m) {
        shoulder_w = w;
        shoulder_value = margins.shoulder_m;
      } else if (w > 0U) {
        // Both ends are fine; X is nonlinear in (q2, q3) so prove the segment (joint3 and joint5,
        // the elbow/wrist coordinates, are linear: equal sign and margin at both ends hold on it).
        rebel_analytic_ik::JointVector previous{};
        detail::toJointVector(waypoints[w - 1U], previous);
        if (rebel_analytic_ik::classifyBranch(previous).shoulder == label.shoulder &&
            !shoulderSegmentClear(
              previous[1], previous[2], q[1], q[2], limits.min_shoulder_margin_m)) {
          shoulder_w = w;
          shoulder_between = true;
          shoulder_value = std::min(margins.shoulder_m,
            rebel_analytic_ik::branchMargins(previous).shoulder_m);
        }
      }
    }
  }
  offer(Rule::kBranchChange, branch_w, branch_joint, branch_value, branch_reference, branch_what);
  offer(Rule::kJoint5SignChange, sign_w, 4U, sign_value, first[4], "");
  if (j5_w != kNpos) {
    offer(Rule::kJoint5Margin, j5_w, 4U, std::abs(waypoints[j5_w][4]), limits.min_abs_joint5_rad, "");
  }
  offer(Rule::kElbowMargin, elbow_w, 2U, elbow_value, limits.min_elbow_margin_rad, "");
  offer(Rule::kShoulderMargin, shoulder_w, 0U, shoulder_value, limits.min_shoulder_margin_m, "",
    shoulder_between);

  // Travel rules. The reported joint is the one that exceeds its limit first (ties: lower index);
  // the value is that joint's TOTAL aggregate travel (as in aggregate mode).
  const auto travel_rule = [&](Rule rule, std::initializer_list<std::size_t> joints, double limit) {
      std::size_t at = kNpos, joint = 0;
      for (const std::size_t j : joints) {
        const std::size_t w = firstTravelExceedance(waypoints, j, limit);
        if (w != kNpos && w < at) {
          at = w;
          joint = j;
        }
      }
      if (at != kNpos) offer(rule, at, joint, result.stats.per_joint_travel[joint], limit, "");
    };
  travel_rule(Rule::kWristTravel, {3U, 4U, 5U}, limits.wrist_travel_max_rad);
  travel_rule(Rule::kArmTravel, {0U, 1U, 2U}, limits.arm_travel_max_rad);
  return result;
}

// Search attribution for the non-travel rules: the end of the stroke the first failing waypoint
// lies on (true = the start). Travel rules use headTailTravel instead.
inline bool trimStartFor(const SplitResult& result)
{
  return 2U * result.waypoint < result.waypoint_count;
}

// Refusal text naming the rule, the joint and the value (shared by search and validator).
inline std::string describe(const SplitResult& result, const SplitLimits& limits)
{
  using detail::fixed;
  constexpr double kDeg = 57.29577951308232;
  const std::string where = " at waypoint " + std::to_string(result.waypoint) + " of " +
    std::to_string(result.waypoint_count);
  const std::string joint = "joint" + std::to_string(result.joint + 1U);
  const std::string text =
    std::string("split joint-travel gate, rule ") + ruleName(result.rule) + ": ";
  switch (result.rule) {
    case Rule::kNone:
      return "split joint-travel gate: ok";
    case Rule::kInvalidInput:
      return text + "waypoint " + std::to_string(result.waypoint) +
        " does not have 6 finite joint values; refusing to guess";
    case Rule::kBranchChange:
      return text + result.what + " IK branch changes (" + joint + ")" + where +
        ": signed boundary coordinate " + fixed(result.reference, 4) + " -> " +
        fixed(result.value, 4) + (std::string(result.what) == "shoulder" ? " m" : " rad");
    case Rule::kJoint5SignChange:
      return text + "joint5 changes sign" + where + " (" + fixed(result.value, 3) +
        " rad, started at " + fixed(result.reference, 3) + " rad)";
    case Rule::kJoint5Margin:
      return text + "|joint5| " + fixed(result.value, 3) + " rad (" + fixed(result.value * kDeg, 1) +
        " deg)" + where + " is below the minimum " + fixed(limits.min_abs_joint5_rad, 3) +
        " rad (" + fixed(limits.min_abs_joint5_rad * kDeg, 1) + " deg)";
    case Rule::kElbowMargin:
      return text + "|joint3 - beta| " + fixed(result.value, 3) + " rad (" +
        fixed(result.value * kDeg, 1) + " deg, distance to the straight arm)" + where +
        " is below the minimum " + fixed(limits.min_elbow_margin_rad, 3) + " rad (" +
        fixed(limits.min_elbow_margin_rad * kDeg, 1) + " deg)";
    case Rule::kShoulderMargin:
      return text + "wrist-centre radial distance |X| " + fixed(result.value * 1000.0, 1) + " mm" +
        (result.between_waypoints ? std::string(" (not provable between waypoints ") +
          std::to_string(result.waypoint - 1U) + " and " + std::to_string(result.waypoint) + ")" :
          where) +
        " is below the minimum " + fixed(limits.min_shoulder_margin_m * 1000.0, 1) + " mm";
    case Rule::kWristTravel:
    case Rule::kArmTravel:
      return text + joint + " moves " + fixed(result.value, 3) + " rad (> " +
        fixed(result.reference, 3) + " rad " +
        (result.rule == Rule::kWristTravel ? "wrist" : "arm") + " limit), first exceeded" + where;
  }
  return text;
}

// One-line measurements for logs / plan summaries.
inline std::string statsText(const StrokeStats& stats)
{
  using detail::fixed;
  constexpr double kDeg = 57.29577951308232;
  std::string text = "travel rad j1..j6 [";
  for (std::size_t j = 0; j < stats.per_joint_travel.size(); ++j) {
    text += (j ? "," : "") + fixed(stats.per_joint_travel[j], 3);
  }
  text += "], min|joint5| " + fixed(stats.min_abs_joint5_rad * kDeg, 1) +
    " deg, IK branch (shoulder " + (stats.branch.shoulder > 0 ? "+" : "-") + ", elbow " +
    (stats.branch.elbow > 0 ? "+" : "-") + ", wrist " + (stats.branch.wrist > 0 ? "+" : "-") +
    ")" + (stats.branch_constant ? " constant" : " CHANGES") +
    ", min boundary margin shoulder " + fixed(stats.min_shoulder_margin_m * 1000.0, 1) +
    " mm, elbow " + fixed(stats.min_elbow_margin_rad * kDeg, 1) + " deg, wrist " +
    fixed(stats.min_abs_joint5_rad * kDeg, 1) + " deg";
  return text;
}

}  // namespace left_contact_travel
