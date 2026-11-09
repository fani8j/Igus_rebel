// Top-seam split between the two arms (pure logic, no ROS): the "left_max" mode derived snapshot for
// the right arm, the fraction arithmetic and the fail-closed coverage check.
//
// Operator rule (2026-10-20): the LEFT arm cuts the top seam as far as it can from the left end; the
// RIGHT arm cuts only the remainder, starting a little before where the left stopped and running to
// the right end in the LEFT arm's travel direction (same knife orientation). The right executor
// relabels the snapshot by 180 degrees, so a snapshot whose seam_start (the world LEFT end) is moved
// to fraction f of the seam, planned with middle_cut_reverse_travel=true, gives the stroke
// world f -> right end.
#pragma once

#include <cstdint>
#include <string>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "dual_cut_coordinator/gates.hpp"
#include "dual_cut_coordinator/trajectory_tools.hpp"

namespace dual_cut
{

enum class TopSplitMode
{
  LeftMax,   // left cuts as far as it reaches, right cuts the remainder (operator rule, default)
  Half       // today's behaviour: half / half around the seam centre
};

const char * toString(TopSplitMode mode);
Status topSplitModeFromString(const std::string & text, TopSplitMode & out);

struct TopSplitParams
{
  TopSplitMode mode = TopSplitMode::LeftMax;
  double overlap_m = 0.012;            // right start = left reach minus this           [0.003, 0.05]
  double right_min_tail = 0.10;        // the right cuts at least the last this fraction [0.05, 0.5]
  double left_min_fraction = 0.05;     // top_cut_min_cut_fraction of the left (phase A) [0.05, 1.0]
  double min_overlap_m = 0.005;        // coverage check: required overlap               [0, 0.05]
  double end_tolerance_m = 0.005;      // coverage check: left starts at 0, right ends at L (0, 0.05]
  /// SEQUENCE schedule only: if the left top's exported middle_cut_fraction is >= this, the right arm does
  /// not cut the top at all (its timeline is the right face only). [0.9, 1.0]; 1.0 = never skip.
  double skip_right_top_fraction = 0.995;
  /// left_max only: the derived right snapshot's seam_end is moved this far OUTWARD along the ORIGINAL seam
  /// direction (past the carton end), so the right top cut finishes this much beyond the seam end. [0, 0.02]; 0 = off.
  double right_end_extension_m = 0.0;
  std::string snapshot_file = "/home/farhan/.ros/dual_cut_coordinator/right_top_snapshot.json";
};

/// The extension that actually applies: right_end_extension_m in left_max (the only mode with a derived right
/// snapshot), 0 in half mode.
double effectiveRightEndExtension(const TopSplitParams & p);

/// Range and format checks (finite numbers, absolute .json path). Fails closed.
Status validateTopSplitParams(const TopSplitParams & p);

/// f = clamp(a - overlap_m / L, 0.05, 1 - right_min_tail): the fraction of the seam (from the world
/// left end) where the right arm's stroke starts. a is the left arm's exported middle_cut_fraction.
/// Fails if a or L is not finite / L <= 0 or the parameters are invalid.
Status topSplitStartFraction(double left_reach_fraction, double seam_length_m, const TopSplitParams & p, double & out);

/// Sequence schedule: skip the right top? True iff skip_fraction < 1.0 and left_reach_fraction >= skip_fraction
/// (1.0 means never skip, even for a left reach of exactly 1.0).
bool skipRightTop(double left_reach_fraction, double skip_fraction);

/// SHA-256 as 64 lowercase hex characters.
std::string sha256Hex(const std::string & data);

/// Whole file as bytes.
Status readTextFile(const std::string & path, std::string & out);

/// seam_start / seam_end of a box_h_cut snapshot document (strict JSON text), as written in the file.
Status seamFromSnapshotText(const std::string & json_text, Eigen::Vector3d & seam_start, Eigen::Vector3d & seam_end);

/// The snapshot entries the top-cut sag needs (all in the snapshot frame).
struct SnapshotSagGeometry
{
  Eigen::Vector3d seam_start, seam_end, seam_center, surface_normal;
  Eigen::Vector3d back_left, front_left, back_right, front_right;
};
Status sagGeometryFromSnapshotText(const std::string & json_text, SnapshotSagGeometry & out);
/// Expected-tip-curve parameters in the WORLD: u = unit(seam_end - seam_start), down = -surface_normal,
/// h = |mean proj(back_left, front_left) - mean proj(back_right, front_right)| / 2 along u. sag_m == 0 gives an
/// INACTIVE CutSag. Fails closed on sag < 0 / non-finite, a degenerate seam axis or h <= 0.
Status makeCutSag(const SnapshotSagGeometry & g, const Eigen::Isometry3d & snapshot_to_world, double sag_m, CutSag & out);

/// What the derived snapshot records about its origin.
struct TopSplitMarkerInfo
{
  std::string source_file;
  std::string source_sha256;        // empty -> source_size / source_mtime_ns are recorded instead
  uint64_t source_size = 0;
  int64_t source_mtime_ns = 0;
  double left_reach_fraction = 0.0;
  double right_end_extension_m = 0.0;   // [0, 0.02]; > 0 is also recorded in the marker
};

/// Pure transformation of a snapshot document: copies everything, moves seam_start to
/// seam_start + f * (seam_end - seam_start), recomputes back_middle_left and front_middle_left as
/// midpoint(back_left | front_left, seam_start'), and adds the object "dual_cut_coordinator_top_split".
/// surface_normal, acquisition_stamp_ns, every other key and the key order are untouched. f must be in
/// (0, 1). If info.right_end_extension_m > 0, seam_end is also moved by that distance along the unit vector
/// of the ORIGINAL seam (seam_start -> seam_end of the source) and back_middle_right / front_middle_right are
/// recomputed as midpoint(back_right | front_right, seam_end') (the executor validates these formulas to 1 um);
/// the marker then records right_end_extension_m. With 0 the output is byte-identical to before.
/// Refuses a source that already carries the marker (a derived file is never a source).
Status deriveTopSnapshotText(
  const std::string & source_json, double start_fraction, const TopSplitMarkerInfo & info, std::string & out_json);

/// Reads source_path, derives, and writes derived_path atomically (temporary file in the same directory
/// + rename; the directory is created), then sets its mtime to the SOURCE file's mtime (the derived
/// file must never look fresher than the capture). The marker records the source SHA-256. If
/// expected_source_sha256 is not empty the source must still hash to it (the seam length the fraction was
/// computed from must be the one that is split), otherwise nothing is written.
Status writeDerivedTopSnapshot(
  const std::string & source_path, const std::string & derived_path, double start_fraction,
  double left_reach_fraction, const std::string & expected_source_sha256 = std::string(), double right_end_extension_m = 0.0);

// ---- coverage check ---------------------------------------------------------------------------

/// A cut line's extent along the seam axis, in metres from the world left end (s = 0) to the right end (s = L).
struct SeamInterval
{
  double s0 = 0.0;   // smaller end
  double s1 = 0.0;   // larger end
};

/// Projection of a world cut line onto the seam axis (start -> end of the source snapshot).
Status projectCutLineOnSeam(
  const CutLine & line, const Eigen::Vector3d & seam_start, const Eigen::Vector3d & seam_end, SeamInterval & out);

struct TopCoverage
{
  double seam_length_m = 0.0;
  SeamInterval left, right;
  double overlap_m = 0.0;
  double right_end_extension_m = 0.0;   // the right cut is expected to end at L + this
};

/// left must start within end_tolerance of 0, right must end within end_tolerance of L + right_end_extension_m
/// (0 = L, as always), and the two intervals overlap by at least min_overlap. The failure message names both intervals in mm and %.
/// `out` is filled even when the check fails.
Status checkTopCoverage(
  const SeamInterval & left, const SeamInterval & right, double seam_length_m, double end_tolerance_m,
  double min_overlap_m, TopCoverage * out = nullptr, double right_end_extension_m = 0.0);

/// Left-only coverage (the right top is skipped): the left cut must start within end_tolerance of 0 and end within
/// end_tolerance of L. `out` gets the left interval (right = the empty interval at L) even when the check fails.
Status checkLeftOnlyCoverage(const SeamInterval & left, double seam_length_m, double end_tolerance_m, TopCoverage * out = nullptr);

/// "left 0..80.0 %, right 77.0..100.0 %, overlap 11.8 mm"
std::string describeTopCoverage(const TopCoverage & c);
/// The same as JSON (mm and fractions).
std::string topCoverageJson(const TopCoverage & c);

}  // namespace dual_cut
