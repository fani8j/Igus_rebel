// Tests for the pure snapshot frame helpers (snapshot_frame.hpp): fixed transform round trip,
// rotate_180 relabel (midpoint formulas, rectangle, handedness) and mirror relabel detection.
#include <gtest/gtest.h>

#include <array>
#include <cmath>

#include "igus_rebel_moveit_config/snapshot_frame.hpp"

namespace
{
using snapshot_frame::Points;
using Vec = Eigen::Vector3d;

// 2026-10-13 09:20 capture (left base frame), only the 11 points and the normal.
Points capture0920()
{
  Points p;
  p.seam_start = Vec(0.4306676236910675, 0.058624765785719124, 0.03068237973238641);
  p.seam_end = Vec(0.431363487388733, -0.3328658228192014, 0.034335138337843074);
  p.seam_center = 0.5 * (p.seam_start + p.seam_end);
  p.back_left = Vec(0.57068035886023, 0.05900201784926623, 0.03091421093142166);
  p.back_right = Vec(0.573049389868955, -0.33491206871560925, 0.03459238692694988);
  p.front_left = Vec(0.2920537275298444, 0.05850348016305057, 0.03045051236642715);
  p.front_right = Vec(0.2912207025716767, -0.3309646553650467, 0.03408183680808752);
  p.back_middle_left = 0.5 * (p.back_left + p.seam_start);
  p.front_middle_left = 0.5 * (p.front_left + p.seam_start);
  p.back_middle_right = 0.5 * (p.back_right + p.seam_end);
  p.front_middle_right = 0.5 * (p.front_right + p.seam_end);
  p.surface_normal = Vec(-0.001680842887023748, 0.009326980083697839, 0.9999550900964042).normalized();
  return p;
}

// dual_arm_calibration.yaml arm_right_base (pose of the right base in the left-base/world frame).
Eigen::Isometry3d rightBase()
{
  return snapshot_frame::poseFromXyzRpy(
    {-0.0013202340013850325, -0.4542609819306861, 0.004780591203829147},
    {-0.01822495461019838, 0.008429085689667826, 0.04908320092978558});
}

double maxDiff(const Points & a, const Points & b)
{
  double worst = 0.0;
  const auto upd = [&worst](const Vec & x, const Vec & y) { worst = std::max(worst, (x - y).cwiseAbs().maxCoeff()); };
  upd(a.seam_start, b.seam_start); upd(a.seam_end, b.seam_end); upd(a.seam_center, b.seam_center);
  upd(a.back_left, b.back_left); upd(a.back_right, b.back_right);
  upd(a.front_left, b.front_left); upd(a.front_right, b.front_right);
  upd(a.back_middle_left, b.back_middle_left); upd(a.front_middle_left, b.front_middle_left);
  upd(a.back_middle_right, b.back_middle_right); upd(a.front_middle_right, b.front_middle_right);
  upd(a.surface_normal, b.surface_normal);
  return worst;
}
}  // namespace

TEST(SnapshotFrame, PoseFromXyzRpyIsFixedAxisZyx)
{
  // yaw 90 deg: x axis -> y axis; rotation order R = Rz Ry Rx (roll applied first).
  const auto yaw = snapshot_frame::poseFromXyzRpy({1.0, 2.0, 3.0}, {0.0, 0.0, M_PI / 2});
  EXPECT_NEAR((yaw * Vec(1, 0, 0) - Vec(1.0, 3.0, 3.0)).norm(), 0.0, 1e-12);
  const auto both = snapshot_frame::poseFromXyzRpy({0, 0, 0}, {M_PI / 2, 0, M_PI / 2});
  // Rx(90): y -> z ; then Rz(90): z stays z. x: Rx leaves x, Rz maps x -> y.
  EXPECT_NEAR((both.linear() * Vec(1, 0, 0) - Vec(0, 1, 0)).norm(), 0.0, 1e-12);
  EXPECT_NEAR((both.linear() * Vec(0, 1, 0) - Vec(0, 0, 1)).norm(), 0.0, 1e-12);
}

TEST(SnapshotFrame, RoundTripAndMidpointFormulas)
{
  const Points original = capture0920();
  ASSERT_LT(snapshot_frame::midpointResidual(original), 1e-12);
  const Eigen::Isometry3d base = rightBase();
  const Points planned = snapshot_frame::toPlanningFrame(original, base);
  // a real change happened (the right base is 454 mm away) ...
  EXPECT_GT((planned.seam_start - original.seam_start).norm(), 0.3);
  // ... the normal is rotated, not translated, and stays unit length ...
  EXPECT_NEAR(planned.surface_normal.norm(), 1.0, 1e-14);
  EXPECT_LT((planned.surface_normal - original.surface_normal).norm(), 0.1);
  // ... and the canonical midpoint formulas survive the affine map far inside the 1e-6 gate.
  EXPECT_LT(snapshot_frame::midpointResidual(planned), 1e-14);
  // round trip
  EXPECT_LT(maxDiff(snapshot_frame::toSnapshotFrame(planned, base), original), 1e-14);
  // identity transform changes nothing
  EXPECT_EQ(maxDiff(snapshot_frame::toPlanningFrame(original, Eigen::Isometry3d::Identity()), original), 0.0);
}

TEST(SnapshotFrame, PlanningFramePointIsBaseToPointTransform)
{
  const Points original = capture0920();
  const Eigen::Isometry3d base = rightBase();
  const Points planned = snapshot_frame::toPlanningFrame(original, base);
  // base * p_plan == p_snapshot for every point; the base origin maps to the planning origin.
  EXPECT_LT((base * planned.back_left - original.back_left).norm(), 1e-14);
  EXPECT_LT((base * planned.seam_end - original.seam_end).norm(), 1e-14);
  Points at_base = original;
  at_base.seam_start = base.translation();
  EXPECT_LT(snapshot_frame::toPlanningFrame(at_base, base).seam_start.norm(), 1e-14);
}

TEST(SnapshotFrame, Rotate180KeepsMidpointsRectangleAndHandedness)
{
  const Points original = capture0920();
  const Points relabelled = snapshot_frame::relabelRotate180(original);
  EXPECT_LT(snapshot_frame::midpointResidual(relabelled), 1e-14);
  EXPECT_TRUE(snapshot_frame::sameHandedness(original, relabelled));
  EXPECT_EQ(snapshot_frame::labelHandedness(original), snapshot_frame::labelHandedness(relabelled));
  // seam centre and the normal are kept; the seam runs the other way
  EXPECT_EQ((relabelled.seam_center - original.seam_center).norm(), 0.0);
  EXPECT_EQ((relabelled.surface_normal - original.surface_normal).norm(), 0.0);
  EXPECT_EQ((relabelled.seam_start - original.seam_end).norm(), 0.0);
  EXPECT_EQ((relabelled.seam_end - original.seam_start).norm(), 0.0);
  // the right face (back_middle_right -> front_middle_right) is now the "left" cut, travelling
  // from the old front to the old back
  EXPECT_EQ((relabelled.back_middle_left - original.front_middle_right).norm(), 0.0);
  EXPECT_EQ((relabelled.front_middle_left - original.back_middle_right).norm(), 0.0);
  // rectangle: opposite edges stay parallel and equal, edges stay perpendicular
  const Vec back = relabelled.back_right - relabelled.back_left;
  const Vec left = relabelled.front_left - relabelled.back_left;
  const Vec right = relabelled.front_right - relabelled.back_right;
  const Vec front = relabelled.front_right - relabelled.front_left;
  EXPECT_NEAR((back - front).norm(), 0.0, 1e-2);   // the measured carton is not an exact rectangle
  EXPECT_NEAR((left - right).norm(), 0.0, 1e-2);
  EXPECT_NEAR(std::abs(back.normalized().dot(left.normalized())), 0.0, 0.02);
  // involution
  EXPECT_EQ(maxDiff(snapshot_frame::relabelRotate180(relabelled), original), 0.0);
}

TEST(SnapshotFrame, Rotate180IsA180DegreeRotationAboutTheNormalOfTheLabelBasis)
{
  // Axis-aligned carton: corners on a rectangle, normal +z. rotate_180 about the centre must map
  // every old corner to the position of the label it takes over.
  Points p;
  p.back_left = Vec(1, 1, 0); p.back_right = Vec(1, -1, 0);
  p.front_left = Vec(-1, 1, 0); p.front_right = Vec(-1, -1, 0);
  p.seam_start = Vec(0, 1, 0); p.seam_end = Vec(0, -1, 0); p.seam_center = Vec(0, 0, 0);
  p.back_middle_left = 0.5 * (p.back_left + p.seam_start);
  p.front_middle_left = 0.5 * (p.front_left + p.seam_start);
  p.back_middle_right = 0.5 * (p.back_right + p.seam_end);
  p.front_middle_right = 0.5 * (p.front_right + p.seam_end);
  const Points r = snapshot_frame::relabelRotate180(p);
  const Eigen::Matrix3d rot180 = Eigen::AngleAxisd(M_PI, Vec::UnitZ()).toRotationMatrix();
  // new label position == rotated old label position
  EXPECT_LT((r.back_left - rot180 * p.back_left).norm(), 1e-12);
  EXPECT_LT((r.back_right - rot180 * p.back_right).norm(), 1e-12);
  EXPECT_LT((r.front_left - rot180 * p.front_left).norm(), 1e-12);
  EXPECT_LT((r.front_right - rot180 * p.front_right).norm(), 1e-12);
  EXPECT_LT((r.seam_start - rot180 * p.seam_start).norm(), 1e-12);
  EXPECT_LT((r.seam_end - rot180 * p.seam_end).norm(), 1e-12);
  EXPECT_LT((r.back_middle_left - rot180 * p.back_middle_left).norm(), 1e-12);
  EXPECT_LT((r.front_middle_left - rot180 * p.front_middle_left).norm(), 1e-12);
  EXPECT_LT((r.back_middle_right - rot180 * p.back_middle_right).norm(), 1e-12);
  EXPECT_LT((r.front_middle_right - rot180 * p.front_middle_right).norm(), 1e-12);
}

TEST(SnapshotFrame, MirrorRelabelIsDetected)
{
  const Points original = capture0920();
  const Points mirrored = snapshot_frame::relabelMirrorLeftRight(original);
  // the canonical midpoint formulas still hold after a mirror, so they cannot detect it ...
  EXPECT_LT(snapshot_frame::midpointResidual(mirrored), 1e-14);
  // ... the label handedness does.
  EXPECT_EQ(snapshot_frame::labelHandedness(mirrored), -snapshot_frame::labelHandedness(original));
  EXPECT_FALSE(snapshot_frame::sameHandedness(original, mirrored));
  EXPECT_TRUE(snapshot_frame::sameHandedness(original, snapshot_frame::relabelRotate180(original)));
}

TEST(SnapshotFrame, HandednessIsInvariantUnderTheFixedTransform)
{
  const Points original = capture0920();
  const Points planned = snapshot_frame::toPlanningFrame(original, rightBase());
  EXPECT_EQ(snapshot_frame::labelHandedness(planned), snapshot_frame::labelHandedness(original));
  const Points both = snapshot_frame::relabelRotate180(planned);
  EXPECT_TRUE(snapshot_frame::sameHandedness(planned, both));
  // transform and relabel commute
  const Points other = snapshot_frame::toPlanningFrame(snapshot_frame::relabelRotate180(original), rightBase());
  EXPECT_LT(maxDiff(both, other), 1e-14);
}

TEST(SnapshotFrame, DegenerateLabelsHaveNoHandedness)
{
  Points p;  // all zero
  EXPECT_EQ(snapshot_frame::labelHandedness(p), 0);
  EXPECT_FALSE(snapshot_frame::sameHandedness(p, p));
  Points nan_points = capture0920();
  nan_points.back_left.x() = std::nan("");
  EXPECT_EQ(snapshot_frame::labelHandedness(nan_points), 0);
}

TEST(SnapshotFrame, ModeAndRelabelNames)
{
  EXPECT_EQ(*snapshot_frame::modeFromString("planning_frame"), snapshot_frame::Mode::kPlanningFrame);
  EXPECT_EQ(*snapshot_frame::modeFromString("fixed_transform"), snapshot_frame::Mode::kFixedTransform);
  EXPECT_FALSE(snapshot_frame::modeFromString("tf").has_value());
  EXPECT_EQ(*snapshot_frame::relabelFromString("none"), snapshot_frame::Relabel::kNone);
  EXPECT_EQ(*snapshot_frame::relabelFromString("rotate_180"), snapshot_frame::Relabel::kRotate180);
  EXPECT_FALSE(snapshot_frame::relabelFromString("mirror").has_value());
  EXPECT_FALSE(snapshot_frame::relabelFromString("").has_value());
}
