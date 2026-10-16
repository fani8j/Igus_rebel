// Unit tests for the pure CRI transit-split helpers (see transit_split.hpp).
#include <gtest/gtest.h>

#include <cmath>
#include <limits>

#include "igus_rebel_moveit_config/transit_split.hpp"

namespace ts = rebel_transit_split;

namespace
{
const std::vector<std::string> kArm{"joint1", "joint2", "joint3", "joint4", "joint5", "joint6"};

ts::Waypoints line(const std::vector<double> & a, const std::vector<double> & b, int n)
{
  ts::Waypoints points;
  for (int i = 0; i <= n; ++i) {
    const double s = static_cast<double>(i) / n;
    std::vector<double> q(a.size());
    for (std::size_t j = 0; j < a.size(); ++j) q[j] = a[j] + s * (b[j] - a[j]);
    points.push_back(q);
  }
  return points;
}

ts::LeafCandidate goodLeaf()
{
  ts::LeafCandidate leaf;
  leaf.recorded_transit = true;
  leaf.provenance_registered = true;
  leaf.scene_diff_robot_state_only = true;
  leaf.joint_names = kArm;
  leaf.waypoints = line({0, 0, 0, 0, 0, 0}, {0.3, -0.2, 0.1, 0.0, 0.4, -0.5}, 8);
  return leaf;
}
}  // namespace

TEST(IsJointLinear, AcceptsStraightLine)
{
  const auto r = ts::IsJointLinear(line({0, 0, 0, 0, 0, 0}, {0.3, -0.2, 0.1, 0.0, 0.4, -0.5}, 10));
  EXPECT_TRUE(r.linear) << r.reason;
  EXPECT_LT(r.max_chord_deviation, 1e-9);
}

TEST(IsJointLinear, AcceptsTwoPointPath)
{
  EXPECT_TRUE(ts::IsJointLinear({{0, 0}, {1, 1}}).linear);
}

TEST(IsJointLinear, AcceptsNonUniformButMonotoneSpacing)
{
  ts::Waypoints p{{0, 0}, {0.1, 0.2}, {0.5, 1.0}, {0.9, 1.8}, {1.0, 2.0}};
  EXPECT_TRUE(ts::IsJointLinear(p).linear);
}

TEST(IsJointLinear, RejectsDeviationAboveTolerance)
{
  auto p = line({0, 0, 0}, {1, 1, 1}, 10);
  p[5][1] += 0.002;
  EXPECT_FALSE(ts::IsJointLinear(p, 1e-3).linear);
  p[5][1] -= 0.0015;  // 0.0005 off: inside
  EXPECT_TRUE(ts::IsJointLinear(p, 1e-3).linear);
}

TEST(IsJointLinear, RejectsDetourThroughOffLinePoint)
{
  ts::Waypoints p{{0, 0}, {0.5, 0.4}, {1, 0}};
  EXPECT_FALSE(ts::IsJointLinear(p).linear);
}

TEST(IsJointLinear, RejectsOvershootAndBacktrack)
{
  // Overshoots past the end then returns: s > 1.
  EXPECT_FALSE(ts::IsJointLinear({{0, 0}, {1.2, 1.2}, {1, 1}}).linear);
  // Starts behind the first point.
  EXPECT_FALSE(ts::IsJointLinear({{0, 0}, {-0.2, -0.2}, {1, 1}}).linear);
  // On the line but goes back and forth.
  EXPECT_FALSE(ts::IsJointLinear({{0, 0}, {0.6, 0.6}, {0.3, 0.3}, {1, 1}}).linear);
}

TEST(IsJointLinear, RejectsDegenerateAndMalformed)
{
  EXPECT_FALSE(ts::IsJointLinear({}).linear);
  EXPECT_FALSE(ts::IsJointLinear({{0, 0}}).linear);
  EXPECT_FALSE(ts::IsJointLinear({{0, 0}, {0, 0}}).linear);
  EXPECT_FALSE(ts::IsJointLinear({{0, 0}, {1}}).linear);
  EXPECT_FALSE(ts::IsJointLinear({{}, {}}).linear);
  EXPECT_FALSE(ts::IsJointLinear({{0, 0}, {1, 1}}, std::nan("")).linear);
}

TEST(IsJointLinear, RejectsNonFinite)
{
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  EXPECT_FALSE(ts::IsJointLinear({{0, 0}, {nan, 1}, {1, 1}}).linear);
  EXPECT_FALSE(ts::IsJointLinear({{0, 0}, {0.5, 0.5}, {1, inf}}).linear);
  EXPECT_FALSE(ts::IsJointLinear({{nan, 0}, {1, 1}}).linear);
}

TEST(MaxJointChange, Basics)
{
  EXPECT_DOUBLE_EQ(ts::MaxJointChange({0, 1, 2}, {0.1, -1, 2}), 2.0);
  EXPECT_TRUE(std::isnan(ts::MaxJointChange({0}, {0, 1})));
  EXPECT_TRUE(std::isnan(ts::MaxJointChange({}, {})));
  EXPECT_TRUE(std::isnan(ts::MaxJointChange({std::nan("")}, {1.0})));
}

TEST(Reorder, ReordersAndRejectsMismatch)
{
  const auto r = ts::ReorderToJointOrder({"b", "a", "c"}, {2, 1, 3}, {"a", "b", "c"});
  ASSERT_EQ(r.size(), 3U);
  EXPECT_DOUBLE_EQ(r[0], 1);
  EXPECT_DOUBLE_EQ(r[1], 2);
  EXPECT_DOUBLE_EQ(r[2], 3);
  EXPECT_TRUE(ts::ReorderToJointOrder({"a", "a", "c"}, {1, 2, 3}, {"a", "b", "c"}).empty());
  EXPECT_TRUE(ts::ReorderToJointOrder({"a", "b", "d"}, {1, 2, 3}, {"a", "b", "c"}).empty());
  EXPECT_TRUE(ts::ReorderToJointOrder({"a", "b"}, {1, 2}, {"a", "b", "c"}).empty());
  EXPECT_TRUE(ts::ReorderToJointOrder({"a", "b"}, {1}, {"a", "b"}).empty());
}

TEST(EvaluateLeaf, AcceptsGoodLeaf)
{
  const auto v = ts::EvaluateLeaf(goodLeaf(), kArm, 0.05);
  EXPECT_TRUE(v.cri) << v.reason;
}

TEST(EvaluateLeaf, AcceptsShuffledJointOrder)
{
  auto leaf = goodLeaf();
  std::swap(leaf.joint_names[0], leaf.joint_names[5]);
  for (auto & point : leaf.waypoints) std::swap(point[0], point[5]);
  EXPECT_TRUE(ts::EvaluateLeaf(leaf, kArm, 0.05).cri);
}

TEST(EvaluateLeaf, RejectsEachCondition)
{
  auto leaf = goodLeaf();
  leaf.recorded_transit = false;
  EXPECT_FALSE(ts::EvaluateLeaf(leaf, kArm, 0.05).cri);

  leaf = goodLeaf();
  leaf.provenance_registered = false;
  EXPECT_FALSE(ts::EvaluateLeaf(leaf, kArm, 0.05).cri);

  leaf = goodLeaf();
  leaf.scene_diff_robot_state_only = false;
  EXPECT_FALSE(ts::EvaluateLeaf(leaf, kArm, 0.05).cri);

  leaf = goodLeaf();  // 7 joints
  leaf.joint_names.push_back("gripper");
  for (auto & point : leaf.waypoints) point.push_back(0.0);
  EXPECT_FALSE(ts::EvaluateLeaf(leaf, kArm, 0.05).cri);

  leaf = goodLeaf();  // wrong joint name
  leaf.joint_names[2] = "joint9";
  EXPECT_FALSE(ts::EvaluateLeaf(leaf, kArm, 0.05).cri);

  leaf = goodLeaf();  // not linear
  leaf.waypoints[3][1] += 0.05;
  EXPECT_FALSE(ts::EvaluateLeaf(leaf, kArm, 0.05).cri);

  leaf = goodLeaf();  // too short
  leaf.waypoints = line({0, 0, 0, 0, 0, 0}, {0.01, 0, 0, 0, 0, 0}, 4);
  EXPECT_FALSE(ts::EvaluateLeaf(leaf, kArm, 0.05).cri);

  leaf = goodLeaf();  // NaN
  leaf.waypoints[2][4] = std::nan("");
  EXPECT_FALSE(ts::EvaluateLeaf(leaf, kArm, 0.05).cri);

  leaf = goodLeaf();  // no waypoints
  leaf.waypoints.clear();
  EXPECT_FALSE(ts::EvaluateLeaf(leaf, kArm, 0.05).cri);
}

TEST(EvaluateLeaf, MinLengthBoundary)
{
  auto leaf = goodLeaf();
  leaf.waypoints = line({0, 0, 0, 0, 0, 0}, {0.05, 0, 0, 0, 0, 0}, 4);
  EXPECT_TRUE(ts::EvaluateLeaf(leaf, kArm, 0.05).cri);
  EXPECT_FALSE(ts::EvaluateLeaf(leaf, kArm, 0.0501).cri);
}

TEST(BuildDispatchPlan, MergesJtcLeavesAndIsolatesCri)
{
  const auto runs = ts::BuildDispatchPlan({false, false, true, false, true, true, false});
  ASSERT_EQ(runs.size(), 6U);
  EXPECT_FALSE(runs[0].cri);
  EXPECT_EQ(runs[0].begin, 0U);
  EXPECT_EQ(runs[0].end, 2U);
  EXPECT_TRUE(runs[1].cri);
  EXPECT_EQ(runs[1].begin, 2U);
  EXPECT_EQ(runs[1].end, 3U);
  EXPECT_FALSE(runs[2].cri);
  EXPECT_EQ(runs[2].begin, 3U);
  EXPECT_EQ(runs[2].end, 4U);
  EXPECT_TRUE(runs[3].cri);
  EXPECT_TRUE(runs[4].cri);
  EXPECT_EQ(runs[4].begin, 5U);
  EXPECT_FALSE(runs[5].cri);
  EXPECT_EQ(runs[5].begin, 6U);
  EXPECT_EQ(runs[5].end, 7U);
}

TEST(BuildDispatchPlan, NoCriLeafIsOneRunAndEmptyIsEmpty)
{
  const auto runs = ts::BuildDispatchPlan({false, false, false});
  ASSERT_EQ(runs.size(), 1U);
  EXPECT_FALSE(runs[0].cri);
  EXPECT_EQ(runs[0].begin, 0U);
  EXPECT_EQ(runs[0].end, 3U);
  EXPECT_TRUE(ts::BuildDispatchPlan({}).empty());
}

TEST(BuildDispatchPlan, CoversEveryLeafOnceInOrder)
{
  const std::vector<bool> flags{true, false, true, true, false, false, true};
  const auto runs = ts::BuildDispatchPlan(flags);
  std::size_t next = 0;
  for (const auto & run : runs) {
    EXPECT_EQ(run.begin, next);
    EXPECT_GT(run.end, run.begin);
    for (std::size_t i = run.begin; i < run.end; ++i) EXPECT_EQ(flags[i], run.cri);
    next = run.end;
  }
  EXPECT_EQ(next, flags.size());
}

TEST(Helpers, PercentAndTolerance)
{
  EXPECT_DOUBLE_EQ(ts::PercentFromScaling(0.1, 50.0), 10.0);
  EXPECT_DOUBLE_EQ(ts::PercentFromScaling(0.9, 50.0), 50.0);
  EXPECT_DOUBLE_EQ(ts::PercentFromScaling(0.0, 50.0), 1.0);
  EXPECT_DOUBLE_EQ(ts::PercentFromScaling(std::nan(""), 50.0), 1.0);
  EXPECT_DOUBLE_EQ(ts::PathToleranceRad(0.2), 0.03);
  EXPECT_DOUBLE_EQ(ts::PathToleranceRad(2.0), 0.06);
  EXPECT_DOUBLE_EQ(ts::PathToleranceRad(std::nan("")), 0.03);
}

TEST(Helpers, DurationModel)
{
  // Long move: cruise + ramp. v = 0.785*0.5, a = 5.24*0.5.
  const double v = 0.3925, a = 2.62;
  EXPECT_NEAR(ts::EstimateDurationS(2.0, 50, 50, 0.785, 5.24), 2.0 / v + v / a, 1e-9);
  // Short move: triangular profile.
  EXPECT_NEAR(ts::EstimateDurationS(0.01, 50, 50, 0.785, 5.24), 2.0 * std::sqrt(0.01 / a), 1e-9);
  EXPECT_TRUE(std::isinf(ts::EstimateDurationS(1.0, 0, 50, 0.785, 5.24)));
  EXPECT_TRUE(std::isinf(ts::EstimateDurationS(-1.0, 10, 50, 0.785, 5.24)));
  EXPECT_TRUE(std::isinf(ts::EstimateDurationS(std::nan(""), 10, 50, 0.785, 5.24)));
  // Monotone in distance.
  EXPECT_LT(ts::EstimateDurationS(0.5, 10, 30, 0.785, 5.24), ts::EstimateDurationS(1.0, 10, 30, 0.785, 5.24));
}
