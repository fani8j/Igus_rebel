// Top (middle seam) cut "sag" profile (2026-10-20, operator). The carton lid sags in the middle, so
// a straight stroke at middle_cut_clearance_m passes above the tape mid-seam. The profile adds a
// parabolic extra depth (along -surface_normal) that is `sag` at the seam centre and 0 at the two
// carton edges the seam ends on:
//
//   d = |(p - seam_center) . u|,   s(p) = sag * max(0, 1 - (d / h)^2)
//
// u is the seam axis (seam_end - seam_start, normalised), h the half carton length along u taken
// from the four CORNERS. Only seam_center, the corners and the seam DIRECTION enter, never the
// stroke's own start/end, so a shortened stroke, a reversed stroke and the right arm's derived
// snapshot (seam_start moved, corners and seam_center unchanged) all lie on the SAME profile.
// Pure Eigen/std header (no ROS) so it is unit-testable.
#pragma once

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>

namespace middle_cut_sag
{
constexpr double kMaxSagM = 0.006;
constexpr double kMinHalfLengthM = 0.05;
constexpr double kMinSeamLengthM = 0.01;

struct Profile {
  Eigen::Vector3d center{Eigen::Vector3d::Zero()};  // seam_center
  Eigen::Vector3d axis{Eigen::Vector3d::UnitX()};   // unit seam axis u
  double half_length_m{0.0};                        // h
  double sag_m{0.0};
};

// Half carton length along `axis`: |mean(left corners . u) - mean(right corners . u)| / 2.
inline double halfLengthM(
  const Eigen::Vector3d& axis, const Eigen::Vector3d& back_left, const Eigen::Vector3d& front_left,
  const Eigen::Vector3d& back_right, const Eigen::Vector3d& front_right)
{
  const double left = 0.5 * (back_left.dot(axis) + front_left.dot(axis));
  const double right = 0.5 * (back_right.dot(axis) + front_right.dot(axis));
  return 0.5 * std::abs(left - right);
}

// Fails closed (false + error) on a non-finite/out-of-range sag, a degenerate seam or an
// implausibly short carton.
inline bool makeProfile(
  const Eigen::Vector3d& seam_start, const Eigen::Vector3d& seam_end,
  const Eigen::Vector3d& seam_center, const Eigen::Vector3d& back_left,
  const Eigen::Vector3d& front_left, const Eigen::Vector3d& back_right,
  const Eigen::Vector3d& front_right, double sag_m, Profile& profile, std::string& error)
{
  if (!std::isfinite(sag_m) || sag_m < 0.0 || sag_m > kMaxSagM) {
    error = "middle_cut_sag_m must be finite and within [0, 0.006] m";
    return false;
  }
  const Eigen::Vector3d seam = seam_end - seam_start;
  const double length = seam.norm();
  if (!std::isfinite(length) || length < kMinSeamLengthM || !seam_center.allFinite() ||
      !back_left.allFinite() || !front_left.allFinite() || !back_right.allFinite() ||
      !front_right.allFinite()) {
    error = "middle_cut_sag: degenerate or non-finite seam/corner geometry";
    return false;
  }
  profile.axis = seam / length;
  profile.center = seam_center;
  profile.sag_m = sag_m;
  profile.half_length_m = halfLengthM(profile.axis, back_left, front_left, back_right, front_right);
  if (!std::isfinite(profile.half_length_m) || profile.half_length_m < kMinHalfLengthM) {
    error = "middle_cut_sag: half carton length along the seam is " +
      std::to_string(profile.half_length_m * 1000.0) + " mm (< " +
      std::to_string(kMinHalfLengthM * 1000.0) + " mm); refusing";
    return false;
  }
  return true;
}

// Extra depth (m, >= 0, along -surface_normal) at stroke point `p`; 0 outside the carton.
inline double extraDepthM(const Profile& profile, const Eigen::Vector3d& p)
{
  const double d = std::abs((p - profile.center).dot(profile.axis));
  const double ratio = d / profile.half_length_m;
  return profile.sag_m * std::max(0.0, 1.0 - ratio * ratio);
}

// Number of polyline legs for a stroke of `length_m` at `step_m` spacing (at least 1).
inline std::size_t legCount(double length_m, double step_m)
{
  if (!(step_m > 0.0) || !(length_m > 0.0)) return 1U;
  return std::max<std::size_t>(1U, static_cast<std::size_t>(std::ceil(length_m / step_m - 1e-9)));
}
}  // namespace middle_cut_sag
