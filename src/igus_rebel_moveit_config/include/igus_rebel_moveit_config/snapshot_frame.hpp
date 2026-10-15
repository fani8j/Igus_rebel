// Pure (Eigen-only) snapshot frame handling for box_h_cut_executor.
//
// The perception snapshot (box_h_cut_snapshot/v1) is written in ONE frame (the left arm base,
// called `base_link`, = world in the dual cell). An executor that plans for another arm has to
// (a) express the 11 snapshot points and the surface normal in its own planning frame
// (snapshot_frame_mode = fixed_transform: p_plan = B^-1 * p_snapshot, B = pose of this arm's base
// in the snapshot frame, URDF fixed-axis rpy, R = Rz(yaw) Ry(pitch) Rx(roll)) and
// (b) optionally relabel the carton corners (snapshot_relabel = rotate_180) so that the face the
// OTHER arm cuts becomes the executor's "left" face with the same geometry/handedness the left
// arm's code expects.
//
// rotate_180 swaps back_left <-> front_right, back_right <-> front_left, seam_start <-> seam_end,
// back_middle_left <-> front_middle_right, front_middle_left <-> back_middle_right and keeps
// seam_center and the normal. It is a relabelling by a 180 degree rotation about the carton normal:
// the canonical midpoint formulas and the rectangle check still hold and the label basis keeps its
// handedness. A mirror relabel (left <-> right with back/front kept) flips the handedness and is
// detected by sameHandedness().
#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <string>

namespace snapshot_frame
{

struct Points
{
  Eigen::Vector3d seam_start{Eigen::Vector3d::Zero()};
  Eigen::Vector3d seam_end{Eigen::Vector3d::Zero()};
  Eigen::Vector3d seam_center{Eigen::Vector3d::Zero()};
  Eigen::Vector3d back_left{Eigen::Vector3d::Zero()};
  Eigen::Vector3d back_right{Eigen::Vector3d::Zero()};
  Eigen::Vector3d front_left{Eigen::Vector3d::Zero()};
  Eigen::Vector3d front_right{Eigen::Vector3d::Zero()};
  Eigen::Vector3d back_middle_left{Eigen::Vector3d::Zero()};
  Eigen::Vector3d front_middle_left{Eigen::Vector3d::Zero()};
  Eigen::Vector3d back_middle_right{Eigen::Vector3d::Zero()};
  Eigen::Vector3d front_middle_right{Eigen::Vector3d::Zero()};
  Eigen::Vector3d surface_normal{Eigen::Vector3d::UnitZ()};
};

enum class Mode { kPlanningFrame, kFixedTransform };
enum class Relabel { kNone, kRotate180 };

inline std::optional<Mode> modeFromString(const std::string & text)
{
  if (text == "planning_frame") return Mode::kPlanningFrame;
  if (text == "fixed_transform") return Mode::kFixedTransform;
  return std::nullopt;
}

inline std::optional<Relabel> relabelFromString(const std::string & text)
{
  if (text == "none") return Relabel::kNone;
  if (text == "rotate_180") return Relabel::kRotate180;
  return std::nullopt;
}

/// URDF fixed-axis rpy (R = Rz(yaw) Ry(pitch) Rx(roll)) plus translation.
inline Eigen::Isometry3d poseFromXyzRpy(
  const std::array<double, 3> & xyz, const std::array<double, 3> & rpy)
{
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  pose.linear() = (Eigen::AngleAxisd(rpy[2], Eigen::Vector3d::UnitZ()) *
                   Eigen::AngleAxisd(rpy[1], Eigen::Vector3d::UnitY()) *
                   Eigen::AngleAxisd(rpy[0], Eigen::Vector3d::UnitX()))
                    .toRotationMatrix();
  pose.translation() = Eigen::Vector3d(xyz[0], xyz[1], xyz[2]);
  return pose;
}

/// Maps `f` over the 11 points (full transform) and the normal (rotation only).
template<typename PointFn, typename NormalFn>
inline Points mapPoints(const Points & in, PointFn point_fn, NormalFn normal_fn)
{
  Points out;
  out.seam_start = point_fn(in.seam_start);
  out.seam_end = point_fn(in.seam_end);
  out.seam_center = point_fn(in.seam_center);
  out.back_left = point_fn(in.back_left);
  out.back_right = point_fn(in.back_right);
  out.front_left = point_fn(in.front_left);
  out.front_right = point_fn(in.front_right);
  out.back_middle_left = point_fn(in.back_middle_left);
  out.front_middle_left = point_fn(in.front_middle_left);
  out.back_middle_right = point_fn(in.back_middle_right);
  out.front_middle_right = point_fn(in.front_middle_right);
  out.surface_normal = normal_fn(in.surface_normal);
  return out;
}

/// Snapshot frame -> planning frame. `planning_base_in_snapshot` is the pose of the planning
/// frame (this arm's base) expressed in the snapshot frame.
inline Points toPlanningFrame(const Points & in, const Eigen::Isometry3d & planning_base_in_snapshot)
{
  const Eigen::Isometry3d inverse = planning_base_in_snapshot.inverse();
  return mapPoints(
    in, [&inverse](const Eigen::Vector3d & p) -> Eigen::Vector3d { return inverse * p; },
    [&inverse](const Eigen::Vector3d & n) -> Eigen::Vector3d { return inverse.linear() * n; });
}

/// Inverse of toPlanningFrame (planning frame -> snapshot frame).
inline Points toSnapshotFrame(const Points & in, const Eigen::Isometry3d & planning_base_in_snapshot)
{
  return mapPoints(
    in,
    [&planning_base_in_snapshot](const Eigen::Vector3d & p) -> Eigen::Vector3d {
      return planning_base_in_snapshot * p;
    },
    [&planning_base_in_snapshot](const Eigen::Vector3d & n) -> Eigen::Vector3d {
      return planning_base_in_snapshot.linear() * n;
    });
}

/// 180 degree rotation about the carton normal, as a relabelling (see the file comment).
inline Points relabelRotate180(const Points & in)
{
  Points out = in;
  out.back_left = in.front_right;
  out.front_right = in.back_left;
  out.back_right = in.front_left;
  out.front_left = in.back_right;
  out.seam_start = in.seam_end;
  out.seam_end = in.seam_start;
  out.back_middle_left = in.front_middle_right;
  out.front_middle_right = in.back_middle_left;
  out.front_middle_left = in.back_middle_right;
  out.back_middle_right = in.front_middle_left;
  return out;
}

/// Mirror relabel (left <-> right, back/front kept). NOT a valid relabel: it flips the handedness
/// of the label basis. Provided so that the detection can be tested and used to refuse it.
inline Points relabelMirrorLeftRight(const Points & in)
{
  Points out = in;
  out.back_left = in.back_right;
  out.back_right = in.back_left;
  out.front_left = in.front_right;
  out.front_right = in.front_left;
  out.seam_start = in.seam_end;
  out.seam_end = in.seam_start;
  out.back_middle_left = in.back_middle_right;
  out.back_middle_right = in.back_middle_left;
  out.front_middle_left = in.front_middle_right;
  out.front_middle_right = in.front_middle_left;
  return out;
}

/// Sign (+1 / -1) of the label basis ((back_right - back_left) x (front_left - back_left)) . normal,
/// 0 when degenerate or not finite.
inline int labelHandedness(const Points & p)
{
  const Eigen::Vector3d across = p.back_right - p.back_left;
  const Eigen::Vector3d along = p.front_left - p.back_left;
  const double value = across.cross(along).dot(p.surface_normal);
  if (!std::isfinite(value) || std::abs(value) < 1e-12) return 0;
  return value > 0.0 ? 1 : -1;
}

/// True when both labellings are non-degenerate and have the same handedness.
inline bool sameHandedness(const Points & before, const Points & after)
{
  const int a = labelHandedness(before);
  const int b = labelHandedness(after);
  return a != 0 && a == b;
}

/// Largest deviation (m) of the four side cutting points from their canonical midpoint formulas
/// (back_middle_left = (back_left + seam_start)/2, front_middle_left = (front_left + seam_start)/2,
/// back_middle_right = (back_right + seam_end)/2, front_middle_right = (front_right + seam_end)/2).
inline double midpointResidual(const Points & p)
{
  const auto dev = [](const Eigen::Vector3d & value, const Eigen::Vector3d & a, const Eigen::Vector3d & b) {
    return (value - 0.5 * (a + b)).cwiseAbs().maxCoeff();
  };
  return std::max(
    std::max(
      dev(p.back_middle_left, p.back_left, p.seam_start),
      dev(p.front_middle_left, p.front_left, p.seam_start)),
    std::max(
      dev(p.back_middle_right, p.back_right, p.seam_end),
      dev(p.front_middle_right, p.front_right, p.seam_end)));
}

}  // namespace snapshot_frame
