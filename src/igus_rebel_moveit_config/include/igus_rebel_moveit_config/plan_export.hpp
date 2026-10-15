// Pure (ROS/MoveIt-free) helpers for exporting the executor's retained MTC solution
// (service ~/get_planned_solution, message dual_cut_msgs/PlannedSegment).
//
// The segment kind of a leaf stage is derived from its MTC stage name. Names are built in
// BoxHCutExecutor::planCutSegments (and carry a " [<motion_profile>]" suffix and, for lifts, a
// " [lift tilt X deg]" suffix; only prefixes are matched). The classifier is a closed table: a
// stage that has a trajectory but an unknown name is NOT guessed, the export fails.
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace plan_export
{

// Values equal dual_cut_msgs/PlannedSegment.msg constants.
enum class Kind : std::uint8_t
{
  kTransit = 0,
  kAlign = 1,
  kDwell = 2,
  kEntry = 3,
  kCut = 4,
  kFinish = 5,
  kLift = 6,
};

inline const char * kindName(Kind kind)
{
  switch (kind) {
    case Kind::kTransit: return "TRANSIT";
    case Kind::kAlign: return "ALIGN";
    case Kind::kDwell: return "DWELL";
    case Kind::kEntry: return "ENTRY";
    case Kind::kCut: return "CUT";
    case Kind::kFinish: return "FINISH";
    case Kind::kLift: return "LIFT";
  }
  return "UNKNOWN";
}

inline bool startsWith(const std::string & text, const char * prefix)
{
  return text.rfind(prefix, 0) == 0U;
}

/// Stage name -> segment kind; nullopt for an unknown name.
///
///   TRANSIT  "approach pose above/outside", "transition from", "transit to supplied"
///   ALIGN    "align above "                       (align above middle seam start)
///   DWELL    "stabilize blade before", "hold at penetration before"
///   ENTRY    "enter laterally into", "descend from aligned"
///   CUT      "cut "
///   FINISH   "finish pose outside/above"
///   LIFT     "lift clear of", "lift straight up", "raise to safe transit height after"
///            (the last one is the Cartesian raise after the left cut, planCutSegments)
inline std::optional<Kind> classifyStageName(const std::string & name)
{
  if (startsWith(name, "approach pose above ") || startsWith(name, "approach pose outside ") ||
      startsWith(name, "transition from ") || startsWith(name, "transit to supplied")) {
    return Kind::kTransit;
  }
  if (startsWith(name, "align above ")) return Kind::kAlign;
  if (startsWith(name, "stabilize blade before ") || startsWith(name, "hold at penetration before ")) {
    return Kind::kDwell;
  }
  if (startsWith(name, "enter laterally into ") || startsWith(name, "descend from aligned ")) {
    return Kind::kEntry;
  }
  if (startsWith(name, "cut ")) return Kind::kCut;
  if (startsWith(name, "finish pose outside ") || startsWith(name, "finish pose above ")) {
    return Kind::kFinish;
  }
  if (startsWith(name, "lift clear of ") || startsWith(name, "lift straight up") ||
      startsWith(name, "raise to safe transit height after ")) {
    return Kind::kLift;
  }
  return std::nullopt;
}

/// Index of the executor segment a stage belongs to: the segment whose name occurs in the stage
/// name, the one occurring LAST when several do ("transition from A finish pose to B approach
/// pose" belongs to B). nullopt when none occurs.
inline std::optional<std::size_t> segmentIndexForStage(
  const std::string & stage_name, const std::vector<std::string> & segment_names)
{
  std::optional<std::size_t> best;
  std::size_t best_pos = 0;
  for (std::size_t i = 0; i < segment_names.size(); ++i) {
    if (segment_names[i].empty()) continue;
    const std::size_t pos = stage_name.rfind(segment_names[i]);
    if (pos == std::string::npos) continue;
    if (!best || pos > best_pos) {
      best = i;
      best_pos = pos;
    }
  }
  return best;
}

/// The physical cut on the carton for an executor cut label. With the rotate_180 relabel the
/// executor's "left" cut is the carton's right face and vice versa; "middle" stays "middle".
inline std::string physicalCut(const std::string & cut_id, bool relabelled_rotate_180)
{
  if (!relabelled_rotate_180) return cut_id;
  if (cut_id == "left") return "right";
  if (cut_id == "right") return "left";
  return cut_id;
}

}  // namespace plan_export
