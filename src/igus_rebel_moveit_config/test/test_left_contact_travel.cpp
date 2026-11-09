// Tests for the pure left contact-stroke joint-travel check (left_contact_travel.hpp), used by
// both the varying-orientation search replay and validateLeftContactJointTravel.
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "igus_rebel_moveit_config/left_contact_travel.hpp"
#include "igus_rebel_moveit_config/rebel_analytic_ik.hpp"

namespace
{
constexpr double kPi = 3.14159265358979323846;
using Waypoints = std::vector<std::vector<double> >;

Waypoints interpolate(const std::vector<double>& a, const std::vector<double>& b, int steps)
{
  Waypoints out;
  for (int k = 0; k <= steps; ++k) {
    const double u = static_cast<double>(k) / steps;
    std::vector<double> q(a.size());
    for (std::size_t j = 0; j < a.size(); ++j) q[j] = a[j] + u * (b[j] - a[j]);
    out.push_back(q);
  }
  return out;
}

const std::vector<double> kStart{0.30, 0.50, 1.00, 0.20, 0.80, -0.40};
}  // namespace

TEST(LeftContactTravel, LimitMatchesTheExecutorConstant)
{
  EXPECT_DOUBLE_EQ(left_contact_travel::kMaxJointTravelRad, 1.0);
}

TEST(LeftContactTravel, LegitSmallStrokePasses)
{
  // ~25-60 mm constant-orientation stroke: a few hundredths of a radian per joint.
  std::vector<double> end = kStart;
  end[0] += 0.05;
  end[1] -= 0.08;
  end[2] += 0.12;
  end[3] += 0.02;
  end[4] -= 0.03;
  end[5] += 0.02;
  const auto result = left_contact_travel::checkJointTravel(interpolate(kStart, end, 40));
  EXPECT_TRUE(result.ok);
  ASSERT_EQ(result.per_joint_travel.size(), 6U);
  EXPECT_NEAR(result.per_joint_travel[2], 0.12, 1e-9);
}

TEST(LeftContactTravel, SyntheticWristFlipIsRejectedAndNamesTheJoint)
{
  // The wrist variant of the same tool pose: q4 + pi, q5 -> -q5, q6 + pi. Only joints 4-6 move.
  std::vector<double> flipped = kStart;
  flipped[3] += kPi;
  flipped[4] = -flipped[4];
  flipped[5] += kPi;
  const auto result = left_contact_travel::checkJointTravel(interpolate(kStart, flipped, 40));
  ASSERT_FALSE(result.ok);
  EXPECT_EQ(result.joint, 3U);  // the first offending joint is joint4
  EXPECT_NEAR(result.travel_rad, kPi, 1e-9);
  const std::string text = left_contact_travel::jointTravelText(result);
  EXPECT_NE(text.find("joint4"), std::string::npos) << text;
  EXPECT_EQ(text, "joint4 3.142");
  // joints 1-3 do not move in a wrist flip
  EXPECT_NEAR(result.per_joint_travel[0], 0.0, 1e-12);
  EXPECT_NEAR(result.per_joint_travel[1], 0.0, 1e-12);
  EXPECT_NEAR(result.per_joint_travel[2], 0.0, 1e-12);
}

TEST(LeftContactTravel, Joint5SignFlipAloneIsRejectedOnJoint5)
{
  // q5 swinging +0.8 -> -0.8 (through zero) is 1.6 rad of aggregate travel.
  std::vector<double> end = kStart;
  end[4] = -kStart[4];
  const auto result = left_contact_travel::checkJointTravel(interpolate(kStart, end, 40));
  ASSERT_FALSE(result.ok);
  EXPECT_EQ(result.joint, 4U);
  EXPECT_EQ(left_contact_travel::jointTravelText(result), "joint5 1.600");
}

TEST(LeftContactTravel, MonotonicJoint3PassOfOnePointFiveRadIsCurrentlyRejected)
{
  // A legitimate ~270 mm corner-to-corner stroke can move the elbow this far monotonically; the
  // existing gate still refuses it (this documents today's behaviour, 2026-10-16 measurements).
  std::vector<double> end = kStart;
  end[2] += 1.5;
  const auto result = left_contact_travel::checkJointTravel(interpolate(kStart, end, 40));
  ASSERT_FALSE(result.ok);
  EXPECT_EQ(result.joint, 2U);
  EXPECT_NEAR(result.travel_rad, 1.5, 1e-9);
  const std::string text = left_contact_travel::jointTravelText(result);
  EXPECT_NE(text.find("joint3"), std::string::npos) << text;
}

TEST(LeftContactTravel, LimitIsStrictlyGreaterThan)
{
  std::vector<double> end = kStart;
  end[1] += 1.0;
  EXPECT_TRUE(left_contact_travel::checkJointTravel(interpolate(kStart, end, 40)).ok);
  end[1] += 1e-6;
  EXPECT_FALSE(left_contact_travel::checkJointTravel(interpolate(kStart, end, 40)).ok);
}

TEST(LeftContactTravel, BackAndForthTravelAccumulates)
{
  // Aggregate |dq|, not net displacement: out 0.4 and back 0.4 per joint = 0.8; twice = 1.6.
  Waypoints w;
  for (int rep = 0; rep < 2; ++rep) {
    w.push_back({0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
    w.push_back({0.0, 0.0, 0.4, 0.0, 0.0, 0.0});
  }
  w.push_back({0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
  const auto result = left_contact_travel::checkJointTravel(w);
  ASSERT_FALSE(result.ok);
  EXPECT_EQ(result.joint, 2U);
  EXPECT_NEAR(result.travel_rad, 1.6, 1e-12);
}

TEST(LeftContactTravel, EmptyAndSingleWaypointPass)
{
  EXPECT_TRUE(left_contact_travel::checkJointTravel({}).ok);
  const auto single = left_contact_travel::checkJointTravel({kStart});
  EXPECT_TRUE(single.ok);
  EXPECT_NEAR(single.per_joint_travel[0], 0.0, 1e-15);
}

TEST(LeftContactTravel, HeadTailAttributionTrimsTheEndThatMovesTheJointMore)
{
  // 100 waypoints over 10 stations -> k = 10 waypoint steps at each end.
  Waypoints late_move;   // joint3 moves 1.2 rad in the last 10 steps only
  for (int w = 0; w < 100; ++w) {
    const double q3 = w < 90 ? 0.0 : 0.12 * (w - 89);
    late_move.push_back({0.0, 0.0, q3, 0.0, 0.0, 0.0});
  }
  auto ht = left_contact_travel::headTailTravel(late_move, 2U, 10U);
  EXPECT_NEAR(ht.head, 0.0, 1e-12);
  EXPECT_NEAR(ht.tail, 1.2, 1e-9);
  EXPECT_FALSE(ht.trimStart());  // trim the END
  Waypoints early_move;
  for (int w = 0; w < 100; ++w) {
    const double q3 = w < 10 ? 0.12 * w : 1.08;
    early_move.push_back({0.0, 0.0, q3, 0.0, 0.0, 0.0});
  }
  ht = left_contact_travel::headTailTravel(early_move, 2U, 10U);
  EXPECT_GT(ht.head, ht.tail);
  EXPECT_TRUE(ht.trimStart());
  // equal head and tail -> the end (search convention: head > tail trims the start)
  Waypoints symmetric = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.5}, {0.0, 0.0, 0.5}, {0.0, 0.0, 1.0}};
  EXPECT_FALSE(left_contact_travel::headTailTravel(symmetric, 2U, 4U).trimStart());
}

// ---------------------------------------------------------------------------------------------
// Split mode (left_contact_travel_mode = split)
// ---------------------------------------------------------------------------------------------
namespace
{
using left_contact_travel::Rule;
namespace ik = rebel_analytic_ik;

constexpr double kDeg = 3.14159265358979323846 / 180.0;

// Only the named rule can fire: everything else effectively off.
left_contact_travel::SplitLimits onlyBranchRules()
{
  left_contact_travel::SplitLimits limits;
  limits.wrist_travel_max_rad = 100.0;
  limits.arm_travel_max_rad = 100.0;
  limits.min_abs_joint5_rad = 0.0;
  limits.min_elbow_margin_rad = 0.0;
  limits.min_shoulder_margin_m = 0.0;
  return limits;
}

std::array<ik::JointLimits, 6> wideLimits()
{
  std::array<ik::JointLimits, 6> limits;
  for (auto& l : limits) l = {-kPi, kPi};
  return limits;
}

// A Cartesian pose with its (up to) 8 Pieper solutions, in solve()'s enumeration order
// (shoulder, elbow, wrist), each index 0 = + and 1 = -. Empty when not all 8 exist.
std::vector<ik::JointVector> allEightBranches(const ik::JointVector& q, const Eigen::Isometry3d& offset)
{
  auto solutions = ik::solve(ik::forwardKinematics(q, offset), offset, wideLimits());
  if (solutions.size() != 8U) solutions.clear();
  return solutions;
}

// Eight branches of a pose whose wrist is well away from its singularity.
std::vector<ik::JointVector> sampleBranches()
{
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> d2(-0.5, 1.2), d3(0.4, 2.0), d5(0.4, 1.3), dw(-2.0, 2.0);
  for (int i = 0; i < 200; ++i) {
    const ik::JointVector q{dw(rng) * 0.5, d2(rng), d3(rng), dw(rng), d5(rng), dw(rng)};
    auto solutions = allEightBranches(q, ik::legacyXeg32KnifeV1ToolOffset());
    if (solutions.size() == 8U) return solutions;
  }
  return {};
}

std::vector<double> toVector(const ik::JointVector& q)
{
  return std::vector<double>(q.begin(), q.end());
}
}  // namespace

TEST(LeftContactSplit, ModeParsing)
{
  using left_contact_travel::Mode;
  EXPECT_EQ(left_contact_travel::modeFromString("aggregate"), Mode::kAggregate);
  EXPECT_EQ(left_contact_travel::modeFromString("split"), Mode::kSplit);
  EXPECT_FALSE(left_contact_travel::modeFromString("").has_value());
  EXPECT_FALSE(left_contact_travel::modeFromString("Split").has_value());
}

TEST(LeftContactSplit, SyntheticWristFlipInAFewWaypointsIsRefusedAsJoint5SignChange)
{
  // q4 + pi, q5 -> -q5, q6 + pi in one step after five unchanged waypoints.
  std::vector<double> flipped = kStart;
  flipped[3] += kPi;
  flipped[4] = -flipped[4];
  flipped[5] += kPi;
  Waypoints w(5, kStart);
  w.push_back(flipped);
  w.push_back(flipped);
  const left_contact_travel::SplitLimits limits;
  const auto result = left_contact_travel::checkSplitTravel(w, limits);
  ASSERT_FALSE(result.ok);
  EXPECT_EQ(result.rule, Rule::kJoint5SignChange);  // wins the tie with wrist_travel at waypoint 5
  EXPECT_EQ(result.waypoint, 5U);
  EXPECT_EQ(result.joint, 4U);
  const std::string text = left_contact_travel::describe(result, limits);
  EXPECT_NE(text.find("joint5_sign_change"), std::string::npos) << text;
  EXPECT_NE(text.find("joint5"), std::string::npos) << text;
  EXPECT_NE(text.find("-0.800"), std::string::npos) << text;
  EXPECT_FALSE(left_contact_travel::trimStartFor(result));  // waypoint 5 of 7 lies on the end half
}

TEST(LeftContactSplit, SmoothSyntheticWristFlipIsRefusedByTheEarliestRule)
{
  // 40 interpolated steps: joint4 accumulates > 1.0 rad (13th step) before joint5 gets within
  // 15 deg of zero (the 14th step), so the wrist travel rule reports first.
  std::vector<double> flipped = kStart;
  flipped[3] += kPi;
  flipped[4] = -flipped[4];
  flipped[5] += kPi;
  const left_contact_travel::SplitLimits limits;
  const auto result = left_contact_travel::checkSplitTravel(interpolate(kStart, flipped, 40), limits);
  ASSERT_FALSE(result.ok);
  EXPECT_EQ(result.rule, Rule::kWristTravel);
  EXPECT_EQ(result.joint, 3U);
  EXPECT_LT(result.waypoint, 20U);
  // with the travel rules off the same stroke is caught by joint5 (margin before the sign change)
  auto only_j5 = limits;
  only_j5.wrist_travel_max_rad = 100.0;
  const auto j5 = left_contact_travel::checkSplitTravel(interpolate(kStart, flipped, 40), only_j5);
  ASSERT_FALSE(j5.ok);
  EXPECT_EQ(j5.rule, Rule::kJoint5Margin);
  EXPECT_NE(left_contact_travel::describe(j5, only_j5).find("joint5_margin"), std::string::npos);
}

TEST(LeftContactSplit, Joint5ZeroCrossingWithSmallJ4J6TravelIsRefused)
{
  // j4/j6 do not move at all; only j5 goes +0.3 -> -0.3 (both outside the margin, so the
  // sign rule, not the margin, is what catches it).
  std::vector<double> a = kStart, b = kStart;
  a[4] = 0.3;
  b[4] = -0.3;
  const left_contact_travel::SplitLimits limits;
  const auto result = left_contact_travel::checkSplitTravel({a, a, b, b}, limits);
  ASSERT_FALSE(result.ok);
  EXPECT_EQ(result.rule, Rule::kJoint5SignChange);
  EXPECT_EQ(result.waypoint, 2U);
  EXPECT_NEAR(result.stats.per_joint_travel[3], 0.0, 1e-12);
  EXPECT_NEAR(result.stats.per_joint_travel[5], 0.0, 1e-12);
  EXPECT_TRUE(left_contact_travel::trimStartFor(result) == false);  // 2*2 < 4 is false
}

TEST(LeftContactSplit, Joint5BelowTheMarginIsRefusedAndNamesValueAndMargin)
{
  left_contact_travel::SplitLimits limits;
  limits.min_abs_joint5_rad = 5.0 * kDeg;
  std::vector<double> q = kStart;
  q[4] = 0.05;  // 2.9 deg
  const auto result = left_contact_travel::checkSplitTravel({q, q, q}, limits);
  ASSERT_FALSE(result.ok);
  EXPECT_EQ(result.rule, Rule::kJoint5Margin);
  EXPECT_EQ(result.waypoint, 0U);
  const std::string text = left_contact_travel::describe(result, limits);
  EXPECT_NE(text.find("0.050"), std::string::npos) << text;
  EXPECT_NE(text.find("5.0 deg"), std::string::npos) << text;
  EXPECT_TRUE(left_contact_travel::trimStartFor(result));
  // exactly at the margin passes; just above too
  q[4] = limits.min_abs_joint5_rad;
  EXPECT_TRUE(left_contact_travel::checkSplitTravel({q, q, q}, limits).ok);
  // the margin is on |j5|: a negative j5 below the margin is refused as well
  q[4] = -0.05;
  EXPECT_EQ(left_contact_travel::checkSplitTravel({q, q, q}, limits).rule, Rule::kJoint5Margin);
  // margin 0 disables the margin rule (the sign rule stays)
  limits.min_abs_joint5_rad = 0.0;
  q[4] = 0.001;
  EXPECT_TRUE(left_contact_travel::checkSplitTravel({q, q, q}, limits).ok);
}

TEST(LeftContactSplit, ElbowMirroredBranchInsertedMidStrokeIsRefused)
{
  const auto branches = sampleBranches();
  ASSERT_EQ(branches.size(), 8U);
  // index = shoulder*4 + elbow*2 + wrist; branch 0 and branch 2 differ only in the elbow.
  Waypoints w(6, toVector(branches[0]));
  w[3] = toVector(branches[2]);
  const auto limits = onlyBranchRules();  // travel and margin rules off: only the branch rule can fire
  const auto result = left_contact_travel::checkSplitTravel(w, limits);
  ASSERT_FALSE(result.ok);
  EXPECT_EQ(result.rule, Rule::kBranchChange);
  EXPECT_STREQ(result.what, "elbow");
  EXPECT_EQ(result.joint, 2U);
  EXPECT_EQ(result.waypoint, 3U);
  EXPECT_FALSE(result.stats.branch_constant);
  const std::string text = left_contact_travel::describe(result, limits);
  EXPECT_NE(text.find("branch_change"), std::string::npos) << text;
  EXPECT_NE(text.find("elbow"), std::string::npos) << text;
  EXPECT_NE(text.find("joint3"), std::string::npos) << text;
}

TEST(LeftContactSplit, ShoulderBranchChangeIsRefused)
{
  const auto branches = sampleBranches();
  ASSERT_EQ(branches.size(), 8U);
  Waypoints w(6, toVector(branches[0]));
  w[4] = toVector(branches[4]);  // differs only in the shoulder (and its dependent joints)
  w[5] = toVector(branches[4]);
  const auto limits = onlyBranchRules();
  const auto result = left_contact_travel::checkSplitTravel(w, limits);
  ASSERT_FALSE(result.ok);
  EXPECT_EQ(result.rule, Rule::kBranchChange);
  EXPECT_STREQ(result.what, "shoulder");
  EXPECT_EQ(result.joint, 0U);
  EXPECT_EQ(result.waypoint, 4U);
  EXPECT_FALSE(left_contact_travel::trimStartFor(result));
}

TEST(LeftContactSplit, WristBranchOfTheSamePoseIsRefusedAsJoint5SignChange)
{
  const auto branches = sampleBranches();
  ASSERT_EQ(branches.size(), 8U);
  Waypoints w(6, toVector(branches[0]));
  w[2] = w[3] = w[4] = w[5] = toVector(branches[1]);  // wrist mirrored (q4 + pi, -q5, q6 + pi)
  const auto result = left_contact_travel::checkSplitTravel(w, onlyBranchRules());
  ASSERT_FALSE(result.ok);
  EXPECT_EQ(result.rule, Rule::kJoint5SignChange);
  EXPECT_EQ(result.waypoint, 2U);
}

TEST(LeftContactSplit, LegitMonotonicLongElbowStrokeIsAcceptedInSplitAndRefusedInAggregate)
{
  // 1.5 rad of joint3 and 0.6 rad of joint2, monotonic, one IK branch, joint5 far from zero,
  // joints 4/6 nearly still: a long corner-to-corner stroke.
  const std::vector<double> a{0.20, 0.10, 0.40, 0.20, 0.80, -0.40};
  const std::vector<double> b{0.45, 0.70, 1.90, 0.25, 0.85, -0.45};
  const auto w = interpolate(a, b, 80);
  const left_contact_travel::SplitLimits limits;  // wrist 1.0, arm 2.0, j5 15 deg, elbow 5 deg, shoulder 50 mm
  const auto split = left_contact_travel::checkSplitTravel(w, limits);
  EXPECT_TRUE(split.ok) << left_contact_travel::describe(split, limits);
  EXPECT_TRUE(split.stats.branch_constant);
  EXPECT_NEAR(split.stats.per_joint_travel[2], 1.5, 1e-9);
  EXPECT_NEAR(split.stats.per_joint_travel[1], 0.6, 1e-9);
  EXPECT_NEAR(split.stats.min_abs_joint5_rad, 0.8, 1e-12);
  EXPECT_GT(split.stats.min_shoulder_margin_m, 0.01);
  EXPECT_GT(split.stats.min_elbow_margin_rad, 0.35);
  const auto aggregate = left_contact_travel::checkJointTravel(w);
  ASSERT_FALSE(aggregate.ok);
  EXPECT_EQ(aggregate.joint, 2U);
  EXPECT_EQ(left_contact_travel::jointTravelText(aggregate), "joint3 1.500");
}

TEST(LeftContactSplit, ArmTravelCapAndWristTravelLimitAreStrictAndNameTheJoint)
{
  const std::vector<double> a{0.20, 0.10, 0.40, 0.20, 0.80, -0.40};
  auto b = a;
  b[2] += 1.8 - 1e-9;  // just inside the (tightened) arm cap (the sum of 80 steps rounds)
  left_contact_travel::SplitLimits limits;
  limits.arm_travel_max_rad = 1.8;
  EXPECT_TRUE(left_contact_travel::checkSplitTravel(interpolate(a, b, 80), limits).ok);
  b[2] += 1e-6 + 1e-9;
  auto result = left_contact_travel::checkSplitTravel(interpolate(a, b, 80), limits);
  ASSERT_FALSE(result.ok);
  EXPECT_EQ(result.rule, Rule::kArmTravel);
  EXPECT_EQ(result.joint, 2U);
  EXPECT_NE(left_contact_travel::describe(result, limits).find("joint3 moves 1.800"), std::string::npos)
    << left_contact_travel::describe(result, limits);
  // joint4 / joint6 keep the 1.0 rad aggregate travel
  auto c = a;
  c[3] += 1.0;
  EXPECT_TRUE(left_contact_travel::checkSplitTravel(interpolate(a, c, 80), limits).ok);
  c[3] += 1e-6;
  result = left_contact_travel::checkSplitTravel(interpolate(a, c, 80), limits);
  ASSERT_FALSE(result.ok);
  EXPECT_EQ(result.rule, Rule::kWristTravel);
  EXPECT_EQ(result.joint, 3U);
  auto d = a;
  d[5] -= 1.2;
  result = left_contact_travel::checkSplitTravel(interpolate(a, d, 80), limits);
  ASSERT_FALSE(result.ok);
  EXPECT_EQ(result.rule, Rule::kWristTravel);
  EXPECT_EQ(result.joint, 5U);
  EXPECT_NE(left_contact_travel::describe(result, limits).find("joint6 moves 1.200"), std::string::npos);
  // the head/tail attribution is unchanged for travel rules (applies to result.joint)
  EXPECT_TRUE(left_contact_travel::isTravelRule(result.rule));
}

TEST(LeftContactSplit, EarliestFailingWaypointDecidesTheReportedRule)
{
  // j5 margin fails at waypoint 3, joint3 arm travel only exceeds at the end: the margin is reported.
  Waypoints w;
  for (int k = 0; k < 20; ++k) {
    std::vector<double> q{0.2, 0.1, 0.4 + 0.1 * k, 0.2, 0.8, -0.4};
    if (k == 3) q[4] = 0.01;
    w.push_back(q);
  }
  const auto result = left_contact_travel::checkSplitTravel(w, left_contact_travel::SplitLimits());
  ASSERT_FALSE(result.ok);
  EXPECT_EQ(result.rule, Rule::kJoint5Margin);
  EXPECT_EQ(result.waypoint, 3U);
}

TEST(LeftContactSplit, InvalidInputFailsClosed)
{
  const left_contact_travel::SplitLimits limits;
  auto result = left_contact_travel::checkSplitTravel({kStart, {0.0, 0.0, 0.0}}, limits);
  ASSERT_FALSE(result.ok);
  EXPECT_EQ(result.rule, Rule::kInvalidInput);
  EXPECT_EQ(result.waypoint, 1U);
  std::vector<double> nan = kStart;
  nan[1] = std::numeric_limits<double>::quiet_NaN();
  result = left_contact_travel::checkSplitTravel({kStart, nan}, limits);
  ASSERT_FALSE(result.ok);
  EXPECT_EQ(result.rule, Rule::kInvalidInput);
  EXPECT_TRUE(left_contact_travel::checkSplitTravel({}, limits).ok);
  EXPECT_TRUE(left_contact_travel::checkSplitTravel({kStart}, limits).ok);
}

TEST(LeftContactSplit, StatsReportBranchAndMargins)
{
  const std::vector<double> a{0.20, -0.10, 0.30, 0.20, 0.80, -0.40};
  const auto stats = left_contact_travel::strokeStats(interpolate(a, a, 3));
  EXPECT_TRUE(stats.branch_constant);
  EXPECT_EQ(stats.branch.elbow, 1);
  EXPECT_EQ(stats.branch.wrist, 1);
  EXPECT_NEAR(stats.min_elbow_margin_rad, 0.3 - 0.0045, 1e-3);
  EXPECT_NEAR(stats.min_abs_joint5_rad, 0.8, 1e-12);
  const std::string text = left_contact_travel::statsText(stats);
  EXPECT_NE(text.find("min|joint5| 45.8 deg"), std::string::npos) << text;
  EXPECT_NE(text.find("constant"), std::string::npos) << text;
}

// The pure classifier must agree with rebel_analytic_ik::solve's own branch enumeration: with
// limits wide enough to keep all branches, solve() returns up to 8 solutions in the order
// shoulder (theta1 branch), elbow (+gamma, -gamma), wrist (+t5, -t5), and the label of each must
// be exactly that index; the label of the sampled q must be its own index.
TEST(LeftContactSplit, ClassifierAgreesWithSolveBranchesOverRandomSamples)
{
  const std::array<Eigen::Isometry3d, 2> offsets = []() {
      Eigen::Isometry3d knife_c = Eigen::Isometry3d::Identity();  // -30 deg pitch + translation
      knife_c.linear() = Eigen::AngleAxisd(-30.0 * kDeg, Eigen::Vector3d::UnitY()).toRotationMatrix();
      knife_c.translation() = Eigen::Vector3d(0.02, -0.09, 0.19);
      return std::array<Eigen::Isometry3d, 2>{ik::legacyXeg32KnifeV1ToolOffset(), knife_c};
    }();
  const auto limits = ik::defaultJointLimits();
  std::mt19937 rng(20261016);
  std::size_t full = 0, checked_original = 0;
  for (const auto& offset : offsets) {
    for (int i = 0; i < 4000; ++i) {
      ik::JointVector q{};
      for (std::size_t j = 0; j < 6; ++j) {
        std::uniform_real_distribution<double> d(limits[j].lower, limits[j].upper);
        q[j] = d(rng);
      }
      const auto solutions = allEightBranches(q, offset);
      if (solutions.empty()) continue;
      ++full;
      std::array<int, 8> seen{};
      for (std::size_t index = 0; index < 8U; ++index) {
        const ik::BranchLabel expected{
          index / 4U == 0U ? 1 : -1, (index / 2U) % 2U == 0U ? 1 : -1, index % 2U == 0U ? 1 : -1};
        const auto label = ik::classifyBranch(solutions[index]);
        ASSERT_TRUE(label == expected)
          << "sample " << i << " solution " << index << ": shoulder " << label.shoulder << " elbow "
          << label.elbow << " wrist " << label.wrist;
        ++seen[index];
        // margins are non-negative and consistent with the label signs
        const auto margins = ik::branchMargins(solutions[index]);
        EXPECT_GE(margins.shoulder_m, 0.0);
        EXPECT_NEAR(margins.wrist_rad, std::abs(solutions[index][4]), 1e-15);
      }
      // the sampled configuration is one of the solutions and carries its index label
      for (std::size_t index = 0; index < 8U; ++index) {
        bool same = true;
        for (std::size_t j = 0; j < 6; ++j) {
          double diff = std::fmod(solutions[index][j] - q[j], 2.0 * kPi);
          if (diff > kPi) diff -= 2.0 * kPi;
          if (diff < -kPi) diff += 2.0 * kPi;
          if (std::abs(diff) > 1e-6) same = false;
        }
        if (same) {
          ++checked_original;
          EXPECT_TRUE(ik::classifyBranch(q) == ik::classifyBranch(solutions[index]));
        }
      }
    }
  }
  EXPECT_GT(full, 2000U);
  EXPECT_GT(checked_original, 2000U);
}

TEST(LeftContactSplit, ClassifierBoundariesAreTheSingularities)
{
  // elbow boundary at theta3 = beta (~4.5 mrad), wrist boundary at theta5 = 0
  ik::JointVector q{0.1, 0.3, 0.5, 0.2, 0.6, 0.1};
  EXPECT_EQ(ik::classifyBranch(q).elbow, 1);
  q[2] = 0.0;
  EXPECT_EQ(ik::classifyBranch(q).elbow, -1);
  EXPECT_NEAR(ik::branchMargins(q).elbow_rad, 0.0044904, 1e-4);
  q[4] = -1e-3;
  EXPECT_EQ(ik::classifyBranch(q).wrist, -1);
  q[4] = 0.0;
  EXPECT_EQ(ik::classifyBranch(q).wrist, 1);
}

// ---------------------------------------------------------------------------------------------
// Fix round: joint5 travel, margins, boundary values, wraps, between-waypoint shoulder, order
// ---------------------------------------------------------------------------------------------
TEST(LeftContactSplit, Joint5OscillationIsRefusedAsWristTravel)
{
  // j5 swings 0.8 -> 1.4 -> 0.8 rad: 1.2 rad of aggregate travel, no sign change, never near
  // zero, small steps (the per-waypoint step gate would pass it), but the aggregate gate refuses it.
  std::vector<double> peak = kStart;
  peak[4] = 1.4;
  Waypoints w = interpolate(kStart, peak, 30);
  const Waypoints back = interpolate(peak, kStart, 30);
  w.insert(w.end(), back.begin() + 1, back.end());
  const left_contact_travel::SplitLimits limits;
  const auto split = left_contact_travel::checkSplitTravel(w, limits);
  ASSERT_FALSE(split.ok);
  EXPECT_EQ(split.rule, Rule::kWristTravel);
  EXPECT_EQ(split.joint, 4U);
  EXPECT_NEAR(split.value, 1.2, 1e-9);
  EXPECT_NE(left_contact_travel::describe(split, limits).find("joint5 moves 1.200"), std::string::npos);
  EXPECT_FALSE(left_contact_travel::checkJointTravel(w).ok);  // aggregate refuses it too
  // exactly 1.0 passes
  peak[4] = 1.3;
  Waypoints w2 = interpolate(kStart, peak, 30);
  const Waypoints back2 = interpolate(peak, kStart, 30);
  w2.insert(w2.end(), back2.begin() + 1, back2.end());
  EXPECT_NEAR(left_contact_travel::strokeStats(w2).per_joint_travel[4], 1.0, 1e-9);
  EXPECT_TRUE(left_contact_travel::checkSplitTravel(w2, limits).ok);
}

TEST(LeftContactSplit, ExactBoundaryValues)
{
  left_contact_travel::SplitLimits limits;
  // j5 exactly at the margin passes (strictly-below refuses), at either sign.
  std::vector<double> q = kStart;
  q[4] = limits.min_abs_joint5_rad;
  EXPECT_TRUE(left_contact_travel::checkSplitTravel({q, q}, limits).ok);
  q[4] = -limits.min_abs_joint5_rad;
  EXPECT_TRUE(left_contact_travel::checkSplitTravel({q, q}, limits).ok);
  q[4] = limits.min_abs_joint5_rad - 1e-9;
  EXPECT_EQ(left_contact_travel::checkSplitTravel({q, q}, limits).rule, Rule::kJoint5Margin);
  // j5 = 0 exactly: + label, margin 0
  q[4] = 0.0;
  EXPECT_EQ(ik::classifyBranch({q[0], q[1], q[2], q[3], q[4], q[5]}).wrist, 1);
  EXPECT_EQ(left_contact_travel::checkSplitTravel({q, q}, limits).rule, Rule::kJoint5Margin);

  // j3 = beta exactly: + label, margin 0 -> refused by the elbow margin; passes with margin 0.
  std::vector<double> e = kStart;
  e[2] = ik::elbowStraightArmAngle();
  EXPECT_EQ(ik::classifyBranch({e[0], e[1], e[2], e[3], e[4], e[5]}).elbow, 1);
  EXPECT_DOUBLE_EQ(ik::branchMargins({e[0], e[1], e[2], e[3], e[4], e[5]}).elbow_rad, 0.0);
  EXPECT_EQ(left_contact_travel::checkSplitTravel({e, e}, limits).rule, Rule::kElbowMargin);
  left_contact_travel::SplitLimits no_elbow = limits;
  no_elbow.min_elbow_margin_rad = 0.0;
  EXPECT_TRUE(left_contact_travel::checkSplitTravel({e, e}, no_elbow).ok);
  // just on the margin passes, just inside is refused
  e[2] = ik::elbowStraightArmAngle() + limits.min_elbow_margin_rad;
  EXPECT_TRUE(left_contact_travel::checkSplitTravel({e, e}, limits).ok);
  e[2] = ik::elbowStraightArmAngle() + limits.min_elbow_margin_rad - 1e-9;
  EXPECT_EQ(left_contact_travel::checkSplitTravel({e, e}, limits).rule, Rule::kElbowMargin);
  e[2] = ik::elbowStraightArmAngle() - limits.min_elbow_margin_rad - 1e-9;  // other side of the singularity
  EXPECT_TRUE(left_contact_travel::checkSplitTravel({e, e}, limits).ok);

  // X = 0 (wrist centre on joint1's axis): bisect q3 for a fixed q2.
  const double q2 = -0.5;
  double lo = 0.0, hi = 2.0;
  ASSERT_GT(ik::wristRadialX(q2, lo), 0.0);
  ASSERT_LT(ik::wristRadialX(q2, hi), 0.0);
  for (int i = 0; i < 200; ++i) {
    const double mid = 0.5 * (lo + hi);
    (ik::wristRadialX(q2, mid) > 0.0 ? lo : hi) = mid;
  }
  std::vector<double> x = kStart;
  x[1] = q2;
  x[2] = lo;  // X ~ 0+ (positive side)
  EXPECT_NEAR(ik::wristRadialX(x[1], x[2]), 0.0, 1e-12);
  EXPECT_EQ(ik::classifyBranch({x[0], x[1], x[2], x[3], x[4], x[5]}).shoulder, 1);
  EXPECT_EQ(left_contact_travel::checkSplitTravel({x, x}, limits).rule, Rule::kShoulderMargin);
  x[2] = hi;  // X ~ 0- (negative side, same margin)
  EXPECT_EQ(ik::classifyBranch({x[0], x[1], x[2], x[3], x[4], x[5]}).shoulder,
    ik::wristRadialX(x[1], x[2]) >= 0.0 ? 1 : -1);
  EXPECT_EQ(left_contact_travel::checkSplitTravel({x, x}, limits).rule, Rule::kShoulderMargin);
  // labels flip across X = 0 and the label comparison reports it as a branch change
  std::vector<double> pos = kStart, neg = kStart;
  pos[1] = neg[1] = q2;
  neg[2] = lo + 0.3;
  left_contact_travel::SplitLimits no_shoulder = limits;
  no_shoulder.min_shoulder_margin_m = 0.0;
  pos[2] = lo - 0.3;
  const auto flip = left_contact_travel::checkSplitTravel({pos, neg}, no_shoulder);
  ASSERT_FALSE(flip.ok);
  EXPECT_EQ(flip.rule, Rule::kBranchChange);
  EXPECT_STREQ(flip.what, "shoulder");
}

TEST(LeftContactSplit, TwoPiWrapsOfJoints146LeaveLabelsAndMarginsUnchanged)
{
  std::mt19937 rng(11);
  const auto limits = ik::defaultJointLimits();
  for (int i = 0; i < 500; ++i) {
    ik::JointVector q{};
    for (std::size_t j = 0; j < 6; ++j) {
      std::uniform_real_distribution<double> d(limits[j].lower, limits[j].upper);
      q[j] = d(rng);
    }
    ik::JointVector wrapped = q;
    wrapped[0] += 2.0 * kPi;
    wrapped[3] -= 2.0 * kPi;
    wrapped[5] += 4.0 * kPi;
    EXPECT_TRUE(ik::classifyBranch(q) == ik::classifyBranch(wrapped));
    const auto a = ik::branchMargins(q), b = ik::branchMargins(wrapped);
    EXPECT_DOUBLE_EQ(a.shoulder_m, b.shoulder_m);
    EXPECT_DOUBLE_EQ(a.elbow_rad, b.elbow_rad);
    EXPECT_DOUBLE_EQ(a.wrist_rad, b.wrist_rad);
  }
}

TEST(LeftContactSplit, WristCentreLipschitzBoundHolds)
{
  std::mt19937 rng(3);
  std::uniform_real_distribution<double> d(-3.0, 3.0), step(-0.8, 0.8);
  for (int i = 0; i < 20000; ++i) {
    const double q2 = d(rng), q3 = d(rng), d2 = step(rng), d3 = step(rng);
    const double s = std::uniform_real_distribution<double>(0.0, 1.0)(rng);
    const double change = std::abs(ik::wristRadialX(q2 + s * d2, q3 + s * d3) - ik::wristRadialX(q2, q3));
    EXPECT_LE(change, ik::wristRadialXVariation(d2, d3) + 1e-12);
  }
}

namespace
{
// Pair found by a seeded scan: both ends have X > 60 mm (the same shoulder label, margin satisfied
// at the ends) but the straight joint-space segment between them crosses X = 0 (twice).
const std::vector<double> kCrossA{0.30, 2.025924567460116, 2.3582656521594245, 0.20, 0.80, -0.40};
const std::vector<double> kCrossB{0.30, -0.3175068167152626, 0.33855530413832446, 0.20, 0.80, -0.40};
}  // namespace

TEST(LeftContactSplit, ShoulderCrossingBetweenWaypointsIsCaughtBySubdivision)
{
  const double xa = ik::wristRadialX(kCrossA[1], kCrossA[2]);
  const double xb = ik::wristRadialX(kCrossB[1], kCrossB[2]);
  ASSERT_GT(xa, 0.06);
  ASSERT_GT(xb, 0.06);
  bool crosses = false;
  for (int k = 0; k <= 1000; ++k) {
    const double t = k / 1000.0;
    if (ik::wristRadialX(
        kCrossA[1] + t * (kCrossB[1] - kCrossA[1]), kCrossA[2] + t * (kCrossB[2] - kCrossA[2])) < 0.0) {
      crosses = true;
    }
  }
  ASSERT_TRUE(crosses) << "the fixture must really cross X = 0";
  left_contact_travel::SplitLimits limits;  // margin 50 mm
  limits.wrist_travel_max_rad = 100.0;
  limits.arm_travel_max_rad = 100.0;
  limits.min_elbow_margin_rad = 0.0;
  // both ends alone are fine: same labels, margins satisfied
  EXPECT_TRUE(left_contact_travel::checkSplitTravel({kCrossA}, limits).ok);
  EXPECT_TRUE(left_contact_travel::checkSplitTravel({kCrossB}, limits).ok);
  EXPECT_TRUE(ik::classifyBranch({kCrossA[0], kCrossA[1], kCrossA[2], kCrossA[3], kCrossA[4], kCrossA[5]}) ==
    ik::classifyBranch({kCrossB[0], kCrossB[1], kCrossB[2], kCrossB[3], kCrossB[4], kCrossB[5]}));
  EXPECT_FALSE(left_contact_travel::shoulderSegmentClear(
    kCrossA[1], kCrossA[2], kCrossB[1], kCrossB[2], limits.min_shoulder_margin_m));
  const auto result = left_contact_travel::checkSplitTravel({kCrossA, kCrossB}, limits);
  ASSERT_FALSE(result.ok);
  EXPECT_EQ(result.rule, Rule::kShoulderMargin);
  EXPECT_TRUE(result.between_waypoints);
  EXPECT_EQ(result.waypoint, 1U);
  const std::string text = left_contact_travel::describe(result, limits);
  EXPECT_NE(text.find("shoulder_margin"), std::string::npos) << text;
  EXPECT_NE(text.find("between waypoints 0 and 1"), std::string::npos) << text;
  // The same pair subdivided finely (what a real stroke looks like) is also refused, at the
  // waypoint where X first gets within the margin.
  const auto fine = left_contact_travel::checkSplitTravel(interpolate(kCrossA, kCrossB, 400), limits);
  ASSERT_FALSE(fine.ok);
  EXPECT_EQ(fine.rule, Rule::kShoulderMargin);
}

TEST(LeftContactSplit, BenignShoulderSegmentsAreProvenClear)
{
  // typical stroke step: ~0.01 rad per waypoint at |X| ~ 0.2-0.4 m
  EXPECT_TRUE(left_contact_travel::shoulderSegmentClear(0.5, 1.0, 0.505, 1.012, 0.05));
  // a long but clearly separated move (X stays near -0.3 .. -0.45 m)
  EXPECT_TRUE(left_contact_travel::shoulderSegmentClear(0.5, 1.0, 0.7, 1.4, 0.05));
  // an endpoint inside the margin
  EXPECT_FALSE(left_contact_travel::shoulderSegmentClear(0.5, 1.0, 0.0, 0.0, 0.05));
  // the subdivision is bounded: a segment hugging the margin is refused rather than looping
  const double q2 = -0.5;
  double lo = 0.0, hi = 2.0;
  for (int i = 0; i < 200; ++i) {
    const double mid = 0.5 * (lo + hi);
    (ik::wristRadialX(q2, mid) > 0.0 ? lo : hi) = mid;
  }
  EXPECT_FALSE(left_contact_travel::shoulderSegmentClear(q2, lo, q2, lo, 0.05));
}

TEST(LeftContactSplit, JointOrderMappingAndMissingJointsFailClosed)
{
  std::string error;
  const std::vector<std::string> natural{"joint1", "joint2", "joint3", "joint4", "joint5", "joint6"};
  const auto natural_index = left_contact_travel::jointIndexByName(natural, error);
  ASSERT_TRUE(natural_index.has_value()) << error;
  for (std::size_t j = 0; j < 6; ++j) EXPECT_EQ((*natural_index)[j], j);

  // A group that lists the joints in another order: the pure check must see joint1..joint6.
  const std::vector<std::string> shuffled{"joint3", "joint1", "joint6", "joint2", "joint5", "joint4"};
  const auto index = left_contact_travel::jointIndexByName(shuffled, error);
  ASSERT_TRUE(index.has_value()) << error;
  // group-order waypoint holding joint1..6 = 0.1, 0.2, 0.3, 0.4, 0.5, 0.6
  const std::vector<double> group_values{0.3, 0.1, 0.6, 0.2, 0.5, 0.4};
  const auto ordered = left_contact_travel::orderWaypoints({group_values}, *index, 6U, error);
  ASSERT_TRUE(ordered.has_value()) << error;
  const std::vector<double> expected{0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
  EXPECT_EQ((*ordered)[0], expected);
  // Reading the shuffled vector as if it were already joint1..joint6 would mis-classify the
  // branch (joint5 would read 0.2, joint3 would read 0.6): the reordering matters.
  EXPECT_NE(ik::branchMargins({0.3, 0.1, 0.6, 0.2, 0.5, 0.4}).elbow_rad,
    ik::branchMargins({0.1, 0.2, 0.3, 0.4, 0.5, 0.6}).elbow_rad);

  // missing / duplicated / extra-named joints fail closed
  EXPECT_FALSE(left_contact_travel::jointIndexByName(
    {"joint1", "joint2", "joint3", "joint4", "joint5"}, error).has_value());
  EXPECT_NE(error.find("joint6"), std::string::npos) << error;
  EXPECT_FALSE(left_contact_travel::jointIndexByName(
    {"arm_joint1", "joint2", "joint3", "joint4", "joint5", "joint6"}, error).has_value());
  EXPECT_NE(error.find("joint1"), std::string::npos) << error;
  EXPECT_FALSE(left_contact_travel::jointIndexByName(
    {"joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint3"}, error).has_value());
  EXPECT_NE(error.find("more than one"), std::string::npos) << error;
  // waypoint with the wrong number of values
  EXPECT_FALSE(left_contact_travel::orderWaypoints({{0.0, 0.0, 0.0}}, *natural_index, 6U, error).has_value());
  EXPECT_NE(error.find("waypoint 0"), std::string::npos) << error;
}

TEST(LeftContactSplit, CoverageFailsClosedWhenTheLeftStrokesAreMissing)
{
  using left_contact_travel::coverageError;
  EXPECT_TRUE(coverageError(false, false, 0U, false).empty());   // plan without a left cut
  EXPECT_TRUE(coverageError(true, true, 4U, true).empty());      // entry, cut, finish, lift
  EXPECT_TRUE(coverageError(true, false, 1U, false).empty());    // entry only (stop_after_start)
  EXPECT_FALSE(coverageError(true, true, 0U, false).empty());    // left cut, nothing examined
  EXPECT_NE(coverageError(true, true, 0U, false).find("no left contact stroke"), std::string::npos);
  EXPECT_FALSE(coverageError(true, true, 3U, false).empty());    // strokes but the cut stroke is missing
  EXPECT_NE(coverageError(true, true, 3U, false).find("cut stroke"), std::string::npos);
}

TEST(LeftContactSplit, ApproachOnlyPlanExpectsNoLeftContactStroke)
{
  using left_contact_travel::SegmentFacts;
  using left_contact_travel::expectedLeftStrokes;
  using left_contact_travel::coverageError;
  const SegmentFacts parallel_left{true, false, false, false};
  const SegmentFacts entry_only_left{true, false, false, true};
  const SegmentFacts fixed_left{true, false, true, false};
  const SegmentFacts hover_only_left{true, true, false, false};
  const SegmentFacts middle{false, false, false, false};

  // normal plans keep expecting their strokes (the gate still fails closed without them)
  auto normal = expectedLeftStrokes({parallel_left}, false);
  EXPECT_TRUE(normal.contact);
  EXPECT_TRUE(normal.cut);
  EXPECT_FALSE(coverageError(normal.contact, normal.cut, 0U, false).empty());
  auto entry = expectedLeftStrokes({entry_only_left}, false);
  EXPECT_TRUE(entry.contact);
  EXPECT_FALSE(entry.cut);
  auto mixed = expectedLeftStrokes({middle, parallel_left}, false);
  EXPECT_TRUE(mixed.contact);
  EXPECT_TRUE(mixed.cut);
  EXPECT_FALSE(expectedLeftStrokes({fixed_left}, false).contact);
  EXPECT_FALSE(expectedLeftStrokes({hover_only_left}, false).contact);
  EXPECT_FALSE(expectedLeftStrokes({middle}, false).contact);
  EXPECT_FALSE(expectedLeftStrokes({}, false).contact);

  // approach-only plan: the same segments, no stroke expected, zero strokes examined is fine
  for (const auto& segment : {parallel_left, entry_only_left, fixed_left, hover_only_left, middle}) {
    auto approach = expectedLeftStrokes({segment}, true);
    EXPECT_FALSE(approach.contact);
    EXPECT_FALSE(approach.cut);
    EXPECT_TRUE(coverageError(approach.contact, approach.cut, 0U, false).empty());
  }
}

TEST(LeftContactSplit, MarginRulesNameTheirValues)
{
  const left_contact_travel::SplitLimits limits;  // elbow 5 deg, shoulder 50 mm
  std::vector<double> e = kStart;
  e[2] = ik::elbowStraightArmAngle() + 0.05;  // 2.9 deg
  auto result = left_contact_travel::checkSplitTravel({kStart, e}, limits);
  ASSERT_FALSE(result.ok);
  EXPECT_EQ(result.rule, Rule::kElbowMargin);
  EXPECT_EQ(result.joint, 2U);
  EXPECT_EQ(result.waypoint, 1U);
  std::string text = left_contact_travel::describe(result, limits);
  EXPECT_NE(text.find("elbow_margin"), std::string::npos) << text;
  EXPECT_NE(text.find("2.9 deg"), std::string::npos) << text;
  EXPECT_NE(text.find("5.0 deg"), std::string::npos) << text;
  std::vector<double> s = kStart;
  s[1] = -0.5;
  double lo = 0.0, hi = 2.0;  // bisect q3 so that X = +20 mm (inside the 50 mm margin)
  for (int i = 0; i < 100; ++i) {
    const double mid = 0.5 * (lo + hi);
    (ik::wristRadialX(s[1], mid) > 0.02 ? lo : hi) = mid;
  }
  s[2] = lo;
  EXPECT_NEAR(ik::wristRadialX(s[1], s[2]), 0.02, 1e-9);
  result = left_contact_travel::checkSplitTravel({s, s}, limits);
  ASSERT_FALSE(result.ok);
  EXPECT_EQ(result.rule, Rule::kShoulderMargin);
  text = left_contact_travel::describe(result, limits);
  EXPECT_NE(text.find("shoulder_margin"), std::string::npos) << text;
  EXPECT_NE(text.find("50.0 mm"), std::string::npos) << text;
}
