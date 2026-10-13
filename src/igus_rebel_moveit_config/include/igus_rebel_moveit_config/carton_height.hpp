// Collision-box height choice for the H-cut executor's carton collision object.
// Pure (no ROS/MoveIt), so it can be unit-tested in isolation.
#pragma once

#include <cmath>
#include <limits>
#include <string>

namespace carton_height
{
struct Choice {
  double height_m{0.0};
  std::string source;   // "parameter", "snapshot", "parameter_min(snapshot_below_min)", "parameter(...)"
  bool warn{false};     // the snapshot value was unusable (invalid or above the maximum)
};

// snapshot_depth_m: snapshot `support_depth_below_top_m` (NaN when the key is absent).
// The configured minimum min_m is never shrunk; a value above max_m or non-finite/non-positive
// falls back to min_m (warn). An absent value is not a warning.
inline Choice choose(
  bool from_snapshot, double snapshot_depth_m, bool snapshot_has_depth, double min_m, double max_m)
{
  Choice choice;
  choice.height_m = min_m;
  choice.source = "parameter";
  if (!from_snapshot || !snapshot_has_depth) return choice;
  if (!std::isfinite(snapshot_depth_m) || snapshot_depth_m <= 0.0) {
    choice.source = "parameter(snapshot value invalid)";
    choice.warn = true;
  } else if (snapshot_depth_m > max_m) {
    choice.source = "parameter(snapshot value above carton_height_max_m)";
    choice.warn = true;
  } else if (snapshot_depth_m < min_m) {
    choice.source = "parameter(snapshot value below carton_height_m)";
  } else {
    choice.height_m = snapshot_depth_m;
    choice.source = "snapshot";
  }
  return choice;
}
}  // namespace carton_height
