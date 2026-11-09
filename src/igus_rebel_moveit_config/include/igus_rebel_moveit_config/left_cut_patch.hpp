// Pure (ROS/MoveIt-free) maths of the LEFT side cut "reverse patch" of box_h_cut_executor
// (planning parameters left_cut_patch_enabled, left_cut_patch_min_gap_m, left_cut_patch_overlap_m).
//
// Background: the forward (back -> front) varying corner-to-corner left cut regularly loses
// 10-117 mm at the BACK end of the extended span (blade-angle cap at the back corner). The patch is a
// second, short REVERSE stroke (front -> back travel) planned in the same executor plan that covers
// the uncut back part and overlaps the forward stroke by left_cut_patch_overlap_m.
//
// All arc coordinates here are FORWARD span coordinates in metres: 0 = back end of the extended span
// (back corner + corner margin), span_length_m = front end. The forward kept interval is
// [fwd_start, fwd_end]; the gap at the back end is fwd_start (the forward "back trim").
//
// The patch line runs in TRAVEL order from `start_arc` (front side, at or beyond the overlap point)
// to arc 0 (the back end of the span). Travel coordinate t = start_arc - arc.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>

namespace left_cut_patch
{

constexpr double kMinGapMax = 0.10;     // left_cut_patch_min_gap_m range [0, kMinGapMax]
constexpr double kOverlapMin = 0.005;   // left_cut_patch_overlap_m range [kOverlapMin, kOverlapMax]
constexpr double kOverlapMax = 0.05;

// Empty string when the parameters are valid.
inline std::string validateParams(double min_gap_m, double overlap_m)
{
  if (!std::isfinite(min_gap_m) || !(min_gap_m >= 0.0 && min_gap_m <= kMinGapMax)) {
    return "left_cut_patch_min_gap_m must be finite and within [0, 0.10] m";
  }
  if (!std::isfinite(overlap_m) || !(overlap_m >= kOverlapMin && overlap_m <= kOverlapMax)) {
    return "left_cut_patch_overlap_m must be finite and within [0.005, 0.05] m";
  }
  return std::string();
}

// A patch is planned only when the forward back trim is LARGER than the minimum gap.
inline bool needsPatch(double forward_back_trim_m, double min_gap_m)
{
  return std::isfinite(forward_back_trim_m) && std::isfinite(min_gap_m) &&
         forward_back_trim_m > min_gap_m + 1e-9;
}

struct Line
{
  bool ok{false};
  std::string error;
  double length_m{0.0};            // patch search line length (travel order), start_arc - 0
  double start_arc_m{0.0};         // arc of the travel start (front side)
  double overlap_point_arc_m{0.0}; // fwd_start + overlap: the kept interval must contain it
  double forward_start_arc_m{0.0};
};

// Search line of the patch. The line begins `slack_m` in front of the overlap point (so a patch
// whose very first station cannot be entered may start a little further front) and ends at arc 0.
// Fails closed on non-finite input, a forward stroke that does not start behind the span end, or a
// forward stroke too short to hold the overlap.
inline Line patchLine(double span_length_m, double forward_start_arc_m, double forward_end_arc_m,
                      double overlap_m, double slack_m)
{
  Line line;
  for (const double v : {span_length_m, forward_start_arc_m, forward_end_arc_m, overlap_m, slack_m}) {
    if (!std::isfinite(v)) {
      line.error = "non-finite patch geometry input";
      return line;
    }
  }
  if (!(span_length_m > 0.0) || !(forward_start_arc_m >= -1e-9) || !(slack_m >= 0.0) ||
      !(forward_end_arc_m > forward_start_arc_m)) {
    line.error = "degenerate patch geometry (span/forward interval/slack)";
    return line;
  }
  line.forward_start_arc_m = std::max(0.0, forward_start_arc_m);
  line.overlap_point_arc_m = line.forward_start_arc_m + overlap_m;
  if (line.overlap_point_arc_m > forward_end_arc_m + 1e-9) {
    line.error = "the forward stroke is shorter than the patch overlap";
    return line;
  }
  line.start_arc_m = std::min(span_length_m, line.overlap_point_arc_m + slack_m);
  line.length_m = line.start_arc_m;
  line.ok = line.length_m > 0.0;
  if (!line.ok) line.error = "empty patch line";
  return line;
}

// Station bounds (0..intervals, spaced length_m / intervals from the travel start) for the patch
// chain: it must START at or before the overlap point (first_max) and END at least one station
// beyond the forward stroke's start (last_min), so the patch really covers the gap's front edge and
// gains something. last_min is clamped to `intervals`.
struct Required
{
  std::size_t first_max{0}, last_min{0};
};
inline Required requiredStations(const Line& line, std::size_t intervals)
{
  Required required;
  if (!line.ok || intervals == 0U) return required;
  const double spacing = line.length_m / static_cast<double>(intervals);
  const double t_overlap = line.start_arc_m - line.overlap_point_arc_m;
  const double t_forward_start = line.start_arc_m - line.forward_start_arc_m;
  const double first = std::floor(t_overlap / spacing + 1e-9);
  const double last = std::ceil(t_forward_start / spacing - 1e-9) + 1.0;
  required.first_max = first < 0.0 ? 0U : std::min(static_cast<std::size_t>(first), intervals);
  required.last_min = last < 1.0 ? 1U : std::min(static_cast<std::size_t>(last), intervals);
  return required;
}

struct Coverage
{
  double forward_kept_mm{0.0};
  double gap_mm{0.0};          // forward back trim
  bool patched{false};
  double patch_kept_mm{0.0};
  double patch_from_arc_mm{0.0}, patch_to_arc_mm{0.0};  // patch kept interval, forward span arc, from < to
  double overlap_mm{0.0};      // actual overlap of the two kept intervals
  double total_covered_mm{0.0};
  double edge_mm{0.0};         // corner-to-corner edge length
  double percent_of_edge{0.0};
};

// Union coverage of the forward kept interval [fwd_start, fwd_end] and (when patched) the patch
// kept interval [patch_from, patch_to] (all forward span arc metres). edge_m is the
// corner-to-corner edge length used for the percentage.
inline Coverage coverage(
  double fwd_start_m, double fwd_end_m, bool patched, double patch_from_m, double patch_to_m,
  double edge_m)
{
  Coverage c;
  c.forward_kept_mm = std::max(0.0, fwd_end_m - fwd_start_m) * 1000.0;
  c.gap_mm = std::max(0.0, fwd_start_m) * 1000.0;
  c.edge_mm = edge_m * 1000.0;
  double total = std::max(0.0, fwd_end_m - fwd_start_m);
  if (patched && patch_to_m > patch_from_m) {
    c.patched = true;
    c.patch_from_arc_mm = patch_from_m * 1000.0;
    c.patch_to_arc_mm = patch_to_m * 1000.0;
    c.patch_kept_mm = (patch_to_m - patch_from_m) * 1000.0;
    const double overlap = std::max(0.0, std::min(patch_to_m, fwd_end_m) - std::max(patch_from_m, fwd_start_m));
    c.overlap_mm = overlap * 1000.0;
    total += (patch_to_m - patch_from_m) - overlap;
  }
  c.total_covered_mm = total * 1000.0;
  c.percent_of_edge = edge_m > 0.0 ? 100.0 * total / edge_m : 0.0;
  return c;
}

}  // namespace left_cut_patch
