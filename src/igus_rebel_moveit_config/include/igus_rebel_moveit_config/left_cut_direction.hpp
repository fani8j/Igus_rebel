// Pure (ROS/MoveIt-free, Eigen only) travel-direction maths of the varying-orientation, corner-to-
// corner LEFT side cut of box_h_cut_executor (planning parameters left_cut_direction = forward |
// reverse and left_reverse_frame = about_normal | about_edge).
//
// Everything the search computes is expressed in TRAVEL coordinates: station 0 is the first point
// the blade reaches, s grows along the travel direction. Forward travel is the nominal
// back_middle_left -> front_middle_left direction; reverse travel is front -> back. To reuse the
// search unchanged for reverse, the executor
//   (1) passes the extended span END points swapped (cut_start = front end),
//   (2) maps the span geometry (nominal segment, seam, seam band) with toTravel(), and
//   (3) builds the candidate orientations in a TRAVEL basis (along = travel direction), not in the
//       nominal back->front basis (cutDebugBasis) that corrections and the hand-off use.
//
// Tool frame (CONTEXT.md "Blade parallelism"): X = blade-plane normal, Y = blade length / edge
// direction (the blade travels towards tool -Y, sharp edge leading), Z = knife axis towards the
// blade point. Forward reference frame (zero tilt/pitch/yaw): X = -normal, Y = -along,
// Z = X x Y = normal x along (towards the carton interior on the left face).
//
// Reverse needs Y = -along_travel, i.e. Y reverses. That leaves one free choice:
//   about_normal : keep X = -normal. The frame is the forward frame rotated 180 deg about the
//                  surface normal. Z (the knife axis) flips to the OTHER side of the cut line.
//   about_edge   : keep Z. The frame is the forward frame rotated 180 deg about tool Z. The knife
//                  axis keeps pointing to the same physical side, X flips to +normal (the blade
//                  plate is upside down). The penetration gate (X or Z within 0.7 of "into the
//                  carton") then has to read the sign-mapped X axis.
#pragma once

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <optional>
#include <string>

namespace left_cut_direction
{

enum class Direction { kForward, kReverse };
enum class ReverseFrame { kAboutNormal, kAboutEdge };
// How the zero-rotation reference tool frame is built from the travel basis.
enum class Convention { kStandard, kEdgeFlipped };

inline std::optional<Direction> directionFromString(const std::string& name)
{
  if (name == "forward") return Direction::kForward;
  if (name == "reverse") return Direction::kReverse;
  return std::nullopt;
}

inline std::optional<ReverseFrame> reverseFrameFromString(const std::string& name)
{
  if (name == "about_normal") return ReverseFrame::kAboutNormal;
  if (name == "about_edge") return ReverseFrame::kAboutEdge;
  return std::nullopt;
}

inline Convention conventionFor(Direction direction, ReverseFrame frame)
{
  return direction == Direction::kReverse && frame == ReverseFrame::kAboutEdge ?
         Convention::kEdgeFlipped : Convention::kStandard;
}

struct Basis
{
  Eigen::Vector3d along{Eigen::Vector3d::UnitX()};
  Eigen::Vector3d across{Eigen::Vector3d::UnitY()};
  Eigen::Vector3d normal{Eigen::Vector3d::UnitZ()};
};

// Nominal (back -> front) cut basis -> basis of the travel direction. Reverse negates along and
// across (normal x along_travel = -across), keeps the normal: still right-handed
// (along x across = normal).
inline Basis travelBasis(const Basis& nominal, bool reverse)
{
  Basis basis = nominal;
  if (reverse) {
    basis.along = -nominal.along;
    basis.across = -nominal.across;
  }
  return basis;
}

// Zero-rotation tool frame (columns X, Y, Z) for a travel basis.
//   kStandard    : X = -normal, Y = -along_travel, Z = X x Y   (forward and reverse about_normal)
//   kEdgeFlipped : X = +normal, Y = -along_travel, Z = X x Y   (reverse about_edge; Z equals the
//                  forward Z)
inline Eigen::Matrix3d referenceFrame(const Basis& travel, Convention convention)
{
  Eigen::Matrix3d reference;
  reference.col(0) = convention == Convention::kEdgeFlipped ? travel.normal : -travel.normal;
  reference.col(1) = -travel.along;
  reference.col(2) = reference.col(0).cross(reference.col(1));
  return reference;
}

// Sign applied to "into the carton" for the X axis of the penetration gate. The gate accepts a
// pose when tool X (blade plate normal, flat blade) OR tool Z (knife axis, perpendicular blade)
// points at least 0.7 into the carton. For the edge-flipped frame the flat-blade X axis points
// along +normal by construction, so the gate reads the sign-mapped axis (equivalent to the
// forward test on the frame rotated back by 180 deg about Z); Z is unchanged by that rotation.
inline double penetrationXSign(Convention convention)
{
  return convention == Convention::kEdgeFlipped ? -1.0 : 1.0;
}

inline double penetrationDotX(
  const Eigen::Matrix3d& rotation, const Eigen::Vector3d& normal, Convention convention)
{
  return rotation.col(0).dot(-normal) * penetrationXSign(convention);
}

inline double penetrationDotZ(const Eigen::Matrix3d& rotation, const Eigen::Vector3d& normal)
{
  return rotation.col(2).dot(-normal);
}

// The existing left penetration gate: skip when dot_x < 0.7 and dot_z < 0.7.
inline bool penetrationGateOk(
  const Eigen::Matrix3d& rotation, const Eigen::Vector3d& normal, Convention convention)
{
  return penetrationDotX(rotation, normal, convention) >= 0.7 ||
         penetrationDotZ(rotation, normal) >= 0.7;
}

// Extended-span geometry in one set of coordinates (metres along the span from its first point).
struct SpanGeometry
{
  double span_length_m{0.0};
  double nominal_start_m{0.0};   // arc position of the nominal segment's first point
  double nominal_length_m{0.0};
  double seam_m{0.0};
  double band_lo_m{0.0}, band_hi_m{0.0};
};

// Forward span coordinates (s from the BACK end of the extended span; nominal segment starts at
// back_middle_left) -> TRAVEL coordinates (s from the first point the blade reaches). Reverse:
// s' = L - s, the band limits swap. Self-inverse: toTravel(toTravel(g, true), true) == g.
inline SpanGeometry toTravel(const SpanGeometry& forward, bool reverse)
{
  if (!reverse) return forward;
  const double length = forward.span_length_m;
  SpanGeometry travel;
  travel.span_length_m = length;
  travel.nominal_length_m = forward.nominal_length_m;
  travel.nominal_start_m = length - (forward.nominal_start_m + forward.nominal_length_m);
  travel.seam_m = length - forward.seam_m;
  travel.band_lo_m = length - forward.band_hi_m;
  travel.band_hi_m = length - forward.band_lo_m;
  return travel;
}

// Position (metres) measured from back_middle_left along back -> front (negative = beyond the
// nominal start toward the back corner) of a TRAVEL coordinate. `travel` is the geometry in
// travel coordinates.
inline double physicalFromBackMiddle(const SpanGeometry& travel, bool reverse, double s_travel_m)
{
  return reverse ? travel.nominal_start_m + travel.nominal_length_m - s_travel_m :
         s_travel_m - travel.nominal_start_m;
}

struct PhysicalTrims
{
  double back_mm{0.0}, front_mm{0.0};
};

// Trim at the first / last end of the travel -> trim at the physical back / front corner.
inline PhysicalTrims physicalTrims(double trim_at_travel_start_mm, double trim_at_travel_end_mm,
                                   bool reverse)
{
  return reverse ? PhysicalTrims{trim_at_travel_end_mm, trim_at_travel_start_mm} :
         PhysicalTrims{trim_at_travel_start_mm, trim_at_travel_end_mm};
}

}  // namespace left_cut_direction
