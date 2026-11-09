// "Seam and side cutting points lie on the detected carton top face" check of the H-cut executor's
// validateSnapshot, as a pure Eigen/std helper (no ROS) so it is unit-testable.
//
// Normal rule (unchanged): every checked point lies within kFaceToleranceM (10 mm) of the top-face
// rectangle in both in-plane directions and within kNormalToleranceM (10 mm) of the face plane.
//
// Coordinator-derived top-split snapshot (2026-10-22, operator: the right top cut finishes past the seam end):
// src/dual_cut_coordinator/src/top_split.cpp moves seam_end OUTWARD along the seam by right_end_extension_m
// and marks the file with the object "dual_cut_coordinator_top_split" (key right_end_extension_m, only when > 0).
// For such a snapshot ONLY the extended seam end may lie beyond the 10 mm tolerance, and only ALONG the seam:
// it must be reachable from inside the 10 mm tolerance rectangle by walking back along the seam direction at
// most right_end_extension_m (so at most 10 mm + extension past the edge when the seam is perpendicular to it).
// A point that is off to the side (the walk back along the seam never enters the rectangle, or needs more than
// the extension), every other point, and a marker whose extension is not finite or outside [0, 0.02] m all keep
// the normal rule or are refused.
//
// The two side cutting points that belong to the extended end (back/front_middle_right in the file's labels,
// back/front_middle_left after a rotate_180 relabel) are canonical midpoints 0.5 * (corner + end), so the
// extension moves them by extension / 2 along the seam. They are checked at their ORIGINAL position, i.e. exactly
// as the face check would see them with no extension: shifted back by extension / 2 along the seam axis. This only
// happens when the marker carries a valid extension in (0, kMaxEndExtensionM] and the point in the file really is
// that canonical midpoint (within kMidpointMatchM, the tolerance of the executor's midpoint-formula check);
// otherwise the point is checked as written (a tampered midpoint is not relaxed).
#pragma once

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <limits>

namespace snapshot_top_face
{
constexpr double kFaceToleranceM = 0.01;
constexpr double kNormalToleranceM = 0.01;
constexpr double kMaxEndExtensionM = 0.02;  // same bound as the coordinator's top_split_right_end_extension_m
constexpr double kMidpointMatchM = 1e-6;    // same tolerance as validateSnapshot's canonical midpoint check

// Which labelled seam point the coordinator extended (after any relabel). kNone = ordinary snapshot.
enum class ExtendedEnd { kNone, kSeamStart, kSeamEnd };

struct Marker {
  bool present{false};
  // NaN = the marker carries an unparsable right_end_extension_m. 0 = marker without the key (no extension).
  double extension_m{0.0};
  ExtendedEnd extended{ExtendedEnd::kNone};
};

enum class Result { kOk, kOutsideTopFace, kInvalidExtension };

struct Face {
  Eigen::Vector3d back_left, back_right, front_left;
  Eigen::Vector3d normal;  // as given in the snapshot (the executor checks it is ~unit)
};

inline bool extensionValid(double extension_m)
{
  return std::isfinite(extension_m) && extension_m >= 0.0 && extension_m <= kMaxEndExtensionM;
}

// The executor's existing rule, bit for bit.
inline bool onTopFace(const Face & face, const Eigen::Vector3d & point)
{
  const Eigen::Vector3d back = face.back_right - face.back_left;
  const Eigen::Vector3d left = face.front_left - face.back_left;
  const double width = back.norm(), depth = left.norm();
  const Eigen::Vector3d relative = point - face.back_left;
  const double x = relative.dot(back) / width;
  const double y = relative.dot(left) / depth;
  return x >= -kFaceToleranceM && x <= width + kFaceToleranceM && y >= -kFaceToleranceM &&
         y <= depth + kFaceToleranceM && std::abs(relative.dot(face.normal)) <= kNormalToleranceM;
}

// `end` is the extended seam end, `other_end` the opposite seam end; `extension_m` must already be valid.
inline bool extendedEndOnTopFace(
  const Face & face, const Eigen::Vector3d & end, const Eigen::Vector3d & other_end, double extension_m)
{
  const Eigen::Vector3d back = face.back_right - face.back_left;
  const Eigen::Vector3d left = face.front_left - face.back_left;
  const double width = back.norm(), depth = left.norm();
  const Eigen::Vector3d relative = end - face.back_left;
  if (!(std::abs(relative.dot(face.normal)) <= kNormalToleranceM)) return false;
  // outward seam direction, in the face plane
  Eigen::Vector3d u = end - other_end;
  const Eigen::Vector3d n = face.normal.normalized();
  u -= u.dot(n) * n;
  const double u_norm = u.norm();
  if (!std::isfinite(u_norm) || !(u_norm > 1e-6)) return false;
  u /= u_norm;
  // in-plane coordinates of the point and of the direction
  const double coords[2] = {relative.dot(back) / width, relative.dot(left) / depth};
  const double dir[2] = {u.dot(back) / width, u.dot(left) / depth};
  const double lo[2] = {-kFaceToleranceM, -kFaceToleranceM};
  const double hi[2] = {width + kFaceToleranceM, depth + kFaceToleranceM};
  // walk back: p - t*u inside the tolerance rectangle for some t in [0, extension_m]
  double t_lo = 0.0, t_hi = extension_m + 1e-9;
  for (int i = 0; i < 2; ++i) {
    if (std::abs(dir[i]) < 1e-12) {
      if (coords[i] < lo[i] || coords[i] > hi[i]) return false;
      continue;
    }
    const double a = (coords[i] - hi[i]) / dir[i];  // coords - t*dir <= hi  <=> t >= a (dir > 0)
    const double b = (coords[i] - lo[i]) / dir[i];
    t_lo = std::max(t_lo, std::min(a, b));
    t_hi = std::min(t_hi, std::max(a, b));
  }
  return t_lo <= t_hi;
}

// Side cutting point `mid` of the extended end: `corner` is the carton corner it was built from, `end` the extended
// seam end, `other_end` the opposite seam end. Returns the position it had before the extension (mid shifted back by
// extension_m / 2 along the seam axis) when mid is the canonical midpoint of corner and the extended end; otherwise
// returns mid unchanged. `extension_m` must already be valid and > 0.
inline Eigen::Vector3d sideMidpointBeforeExtension(
  const Eigen::Vector3d & mid, const Eigen::Vector3d & corner, const Eigen::Vector3d & end,
  const Eigen::Vector3d & other_end, double extension_m)
{
  const Eigen::Vector3d canonical = 0.5 * (corner + end);
  if (!((mid - canonical).cwiseAbs().maxCoeff() <= kMidpointMatchM)) return mid;
  const Eigen::Vector3d axis = end - other_end;
  const double axis_norm = axis.norm();
  if (!std::isfinite(axis_norm) || !(axis_norm > 1e-6)) return mid;
  return mid - 0.5 * extension_m * axis / axis_norm;
}

inline Result checkSeamAndSidePoints(
  const Face & face, const Eigen::Vector3d & seam_start, const Eigen::Vector3d & seam_end,
  const Eigen::Vector3d & seam_center, const Eigen::Vector3d & back_middle_left,
  const Eigen::Vector3d & front_middle_left, const Eigen::Vector3d & back_middle_right,
  const Eigen::Vector3d & front_middle_right, const Eigen::Vector3d & front_right, const Marker & marker)
{
  if (marker.present && !extensionValid(marker.extension_m)) return Result::kInvalidExtension;
  const bool extended_start = marker.present && marker.extension_m > 0.0 && marker.extended == ExtendedEnd::kSeamStart;
  const bool extended_end = marker.present && marker.extension_m > 0.0 && marker.extended == ExtendedEnd::kSeamEnd;
  const bool start_ok = extended_start ? extendedEndOnTopFace(face, seam_start, seam_end, marker.extension_m)
                                       : onTopFace(face, seam_start);
  const bool end_ok = extended_end ? extendedEndOnTopFace(face, seam_end, seam_start, marker.extension_m)
                                   : onTopFace(face, seam_end);
  // Side points of the extended end are judged where they were before the extension (see the file comment).
  Eigen::Vector3d check_back_left = back_middle_left, check_front_left = front_middle_left;
  Eigen::Vector3d check_back_right = back_middle_right, check_front_right = front_middle_right;
  if (extended_start) {
    check_back_left = sideMidpointBeforeExtension(back_middle_left, face.back_left, seam_start, seam_end, marker.extension_m);
    check_front_left = sideMidpointBeforeExtension(front_middle_left, face.front_left, seam_start, seam_end, marker.extension_m);
  }
  if (extended_end) {
    check_back_right = sideMidpointBeforeExtension(back_middle_right, face.back_right, seam_end, seam_start, marker.extension_m);
    check_front_right = sideMidpointBeforeExtension(front_middle_right, front_right, seam_end, seam_start, marker.extension_m);
  }
  if (!start_ok || !end_ok || !onTopFace(face, seam_center) || !onTopFace(face, check_back_left) ||
      !onTopFace(face, check_front_left) || !onTopFace(face, check_back_right) ||
      !onTopFace(face, check_front_right)) {
    return Result::kOutsideTopFace;
  }
  return Result::kOk;
}
}  // namespace snapshot_top_face
