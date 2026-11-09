// Pure (ROS/MoveIt-free, Eigen only) true blade-angle gate for the varying-orientation LEFT side
// cut of box_h_cut_executor (planning parameters left_varying_max_theta_p_deg,
// left_varying_end_max_theta_p_deg, left_varying_start_max_theta_p_deg,
// left_varying_start_ramp_m, read-only left_varying_theta_p_tolerance_deg).
//
// theta_p = acos(|tool X . surface_normal|) is the TRUE angle between the blade plane and the
// detected carton top face (0 = blade flat on the top plane; CONTEXT.md "Blade parallelism").
//
// The cap is a function of the arc position s of the TCP along the TRAVEL direction on the
// EXTENDED span (fixed geometry, independent of how the kept interval is trimmed):
//   s from the stroke start through the end of the seam band  -> band cap
//   after the band exit                                       -> linear ramp to the end cap at
//                                                                the end of the span
// Optional START cap (left_varying_start_max_theta_p_deg > band cap, default 0 = off): a steep
// blade at the start is allowed (the arm needs it to reach the back corner) but it must be back
// down at the band cap BEFORE the tape band:
//   s <= band_entry - ramp          -> start cap
//   band_entry - ramp < s < entry   -> linear start cap -> band cap (a ramp that would begin
//                                      before s = 0 is clipped, it still ends at the band entry)
//   band_entry <= s <= band_exit    -> band cap (never more, also not with a ramp of 0 m)
// Inconsistent or non-finite start geometry falls back to the band cap (never more permissive
// than the band cap). Operator rule: a steep blade at the TAPE did not cut it, so the tape band
// always stays at the band cap; only the start before the band and the END of the travel may be
// steeper. The profile is defined in travel coordinates (s = 0 at the first point the blade
// reaches), so reverse travel (left_cut_direction reverse) only needs the mapped band: the end
// ramp then lies toward the BACK corner.
//
// The same functions are used for (a) the per-station candidate filter of the search, (b) the
// search replay of the Cartesian stroke and (c) validateLeftVaryingRetainedStroke on the final
// retained trajectory, so the search never emits a stroke the validator refuses.
#pragma once

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace left_blade_cap
{

constexpr double kPi = 3.14159265358979323846;

// theta_p (deg) of a tool rotation: acos(|tool X . normal|), normal a unit vector.
inline double thetaPDeg(const Eigen::Matrix3d& tool_rotation, const Eigen::Vector3d& normal)
{
  return std::acos(std::min(1.0, std::abs(tool_rotation.col(0).dot(normal)))) * 180.0 / kPi;
}

// Angle (deg) between the blade length axis (tool Y) and the cut line direction `-along` (the
// existing blade-deviation definition of the varying search).
inline double bladeDeviationDeg(const Eigen::Matrix3d& tool_rotation, const Eigen::Vector3d& along)
{
  return std::acos(std::clamp(tool_rotation.col(1).dot(-along), -1.0, 1.0)) * 180.0 / kPi;
}

// Travel coordinate at which the seam band ends (where the ramp starts). band_lo/band_hi are the
// band limits in FORWARD span coordinates (from the span start, as LeftExtension), span_length_m
// the extended span length. Forward travel exits the band at band_hi; reverse travel enters at the
// other end of the span and exits at span_length - band_lo. The executor maps the whole span
// geometry to TRAVEL coordinates (left_cut_direction::toTravel) and then calls this with
// reverse = false and the mapped band; the two ways agree (test_left_cut_direction).
inline double bandExitTravel(
  double span_length_m, double band_lo_forward_m, double band_hi_forward_m, bool reverse)
{
  return reverse ? span_length_m - band_lo_forward_m : band_hi_forward_m;
}

// Pure validation of the start-cap parameters (shared by the executor's validateSettings).
// Returns an empty string when valid. start_cap_deg == 0 is OFF (ramp is then only range-checked).
inline std::string validateStartCap(double start_cap_deg, double band_cap_deg, double start_ramp_m)
{
  if (!std::isfinite(start_cap_deg) || !std::isfinite(band_cap_deg) || !std::isfinite(start_ramp_m)) {
    return "left_varying_start_max_theta_p_deg and left_varying_start_ramp_m must be finite";
  }
  if (!(start_ramp_m >= 0.0 && start_ramp_m <= 0.2)) {
    return "left_varying_start_ramp_m must be within [0, 0.2] m";
  }
  if (start_cap_deg != 0.0 && !(start_cap_deg >= band_cap_deg && start_cap_deg <= 45.0)) {
    return "left_varying_start_max_theta_p_deg must be 0 (off) or within "
           "[left_varying_max_theta_p_deg, 45] deg (the start cap may not be below the band cap)";
  }
  return std::string();
}

struct Profile
{
  double band_cap_deg{15.0};   // left_varying_max_theta_p_deg
  double end_cap_deg{15.0};    // left_varying_end_max_theta_p_deg (>= band cap)
  double span_length_m{0.0};   // extended span, travel coordinates 0..span_length_m
  double band_exit_m{0.0};     // travel coordinate where the band ends and the ramp starts
  double start_cap_deg{0.0};   // left_varying_start_max_theta_p_deg (0 = off, else >= band cap)
  double start_ramp_m{0.0};    // left_varying_start_ramp_m: length of the start-cap ramp
  double band_entry_m{0.0};    // travel coordinate where the band starts (the start ramp ends)
  // Reverse-patch search only (left_cut_patch.hpp): the search line is a sub-span of the full
  // reverse travel. All profile coordinates above stay in FULL-span travel coordinates and a
  // LOCAL coordinate s_local (metres from the sub-span start) maps to s_local + origin_offset_m.
  // 0 (default) = the profile is used in its own coordinates, nothing changes.
  double origin_offset_m{0.0};

  bool graded() const {return end_cap_deg > band_cap_deg + 1e-12;}
  // True when a start cap above the band cap is requested (the geometry may still be invalid,
  // capAt then falls back to the band cap).
  bool startGraded() const {return start_cap_deg > band_cap_deg + 1e-12;}

  // Report text of the cap range: "25.0..40.0" (band..end) while the start cap is off,
  // "start 40.0 -> band 25.0 -> end 40.0" when it is active.
  std::string rangeText() const
  {
    std::ostringstream out;
    out << std::fixed << std::setprecision(1);
    if (startGraded()) {
      out << "start " << start_cap_deg << " (ramp " << std::setprecision(0) << start_ramp_m * 1000.0 <<
        " mm) -> band " << std::setprecision(1) << band_cap_deg << " -> end " << end_cap_deg;
    } else {
      out << band_cap_deg << ".." << end_cap_deg;
    }
    return out.str();
  }

  // Waypoints this far (m) before s = 0 still count as the stroke start (numerical noise of the
  // first waypoint); anything further out is outside the span and gets the band cap.
  static constexpr double kStartSlackM = 1e-3;

  // capAt for a coordinate measured from the (sub-)span start the search line begins at.
  double capAtLocal(double s_local) const {return capAt(s_local + origin_offset_m);}

  // Cap (deg) at travel coordinate s (metres from the stroke start along the travel direction).
  double capAt(double s) const
  {
    if (startGraded() && std::isfinite(s)) {
      const bool geometry_ok =
        std::isfinite(start_cap_deg) && std::isfinite(start_ramp_m) && std::isfinite(band_entry_m) &&
        std::isfinite(band_exit_m) && std::isfinite(span_length_m) && start_ramp_m >= 0.0 &&
        band_entry_m >= 0.0 && band_entry_m <= band_exit_m + 1e-9 &&
        band_entry_m <= span_length_m + 1e-9;
      // The band (and everything after it) is never touched by the start cap.
      if (geometry_ok && s < band_entry_m && s >= -kStartSlackM) {
        const double ramp_begin = band_entry_m - start_ramp_m;
        if (s <= ramp_begin) return start_cap_deg;
        // ramp_begin < s < band_entry_m, so start_ramp_m > 0 here
        return start_cap_deg + (band_cap_deg - start_cap_deg) * (s - ramp_begin) / start_ramp_m;
      }
    }
    if (!graded() || !(span_length_m > band_exit_m + 1e-9) || !(s > band_exit_m)) {
      return band_cap_deg;
    }
    if (s >= span_length_m) return end_cap_deg;
    return band_cap_deg + (end_cap_deg - band_cap_deg) * (s - band_exit_m) /
           (span_length_m - band_exit_m);
  }
};

struct StrokeReport
{
  bool ok{true};
  std::size_t waypoints{0};
  double max_theta_p_deg{0.0};
  double max_theta_p_band_deg{0.0};     // inside the seam band (valid when has_band)
  bool has_band{false};
  double max_blade_deviation_deg{0.0};
  double max_blade_deviation_band_deg{0.0};  // inside the seam band (valid when has_band); statistics only
  double max_overshoot_deg{0.0};        // max(theta_p - station cap), floored at 0
  // First waypoint above cap + tolerance (valid when !ok).
  std::size_t violation_index{0};
  double violation_s_m{0.0}, violation_theta_p_deg{0.0}, violation_cap_deg{0.0};
};

// Evaluates the true blade angle of every waypoint of a stroke. `tool_poses` are the tool-frame
// poses of the stroke's waypoints, `origin` the span start (travel coordinate 0), `travel_dir` the
// unit travel direction, `normal` the unit surface normal and `along` the nominal cut direction.
// A waypoint violates when theta_p > cap(s) + tolerance_deg. When `has_band` the maximum inside
// [band_lo_m, band_hi_m] (travel coordinates) is reported separately. Non-finite input fails
// closed (ok = false).
inline StrokeReport evaluateStroke(
  const Profile& profile, bool has_band, double band_lo_m, double band_hi_m,
  const std::vector<Eigen::Isometry3d>& tool_poses, const Eigen::Vector3d& origin,
  const Eigen::Vector3d& travel_dir, const Eigen::Vector3d& normal, const Eigen::Vector3d& along,
  double tolerance_deg)
{
  StrokeReport report;
  report.has_band = has_band;
  report.waypoints = tool_poses.size();
  bool violated = false;
  for (std::size_t w = 0; w < tool_poses.size(); ++w) {
    const Eigen::Isometry3d& pose = tool_poses[w];
    const double s = (pose.translation() - origin).dot(travel_dir);
    const double theta = thetaPDeg(pose.linear(), normal);
    const double deviation = bladeDeviationDeg(pose.linear(), along);
    if (!std::isfinite(s) || !std::isfinite(theta) || !std::isfinite(deviation)) {
      report.ok = false;
      if (!violated) {
        violated = true;
        report.violation_index = w;
        report.violation_s_m = s;
        report.violation_theta_p_deg = theta;
        report.violation_cap_deg = std::numeric_limits<double>::quiet_NaN();
      }
      continue;
    }
    const double s_full = s + profile.origin_offset_m;  // == s unless a patch sub-span
    double cap = profile.capAt(s_full);
    // Defence in depth: inside the reported seam band the cap is never above the band cap, and a
    // start cap without a known band is not applied (the band cannot be protected then).
    if ((has_band && s_full >= band_lo_m - 1e-9 && s_full <= band_hi_m + 1e-9) ||
      (!has_band && profile.startGraded()))
    {
      cap = std::min(cap, profile.band_cap_deg);
    }
    report.max_theta_p_deg = std::max(report.max_theta_p_deg, theta);
    report.max_blade_deviation_deg = std::max(report.max_blade_deviation_deg, deviation);
    report.max_overshoot_deg = std::max(report.max_overshoot_deg, theta - cap);
    if (has_band && s_full >= band_lo_m - 1e-9 && s_full <= band_hi_m + 1e-9) {
      report.max_theta_p_band_deg = std::max(report.max_theta_p_band_deg, theta);
      report.max_blade_deviation_band_deg =
        std::max(report.max_blade_deviation_band_deg, deviation);
    }
    if (theta > cap + tolerance_deg && !violated) {
      report.ok = false;
      violated = true;
      report.violation_index = w;
      report.violation_s_m = s;
      report.violation_theta_p_deg = theta;
      report.violation_cap_deg = cap;
    }
  }
  return report;
}

}  // namespace left_blade_cap
