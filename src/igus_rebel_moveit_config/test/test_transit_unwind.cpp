// Tests for the pure parts of the safe unwind detour of a free-space transit (transit_unwind.hpp).
#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <vector>

#include "igus_rebel_moveit_config/transit_unwind.hpp"

namespace
{
using transit_unwind::Candidate;
using transit_unwind::Path;
using transit_unwind::WristOption;
constexpr double kPi = transit_unwind::kPi;

// 6-joint path whose joint6 (index 5) goes through the given values; other joints constant.
Path pathJoint6(const std::vector<double>& j6)
{
  Path p;
  for (const double v : j6) p.push_back({0.1, 0.2, 0.3, 0.4, 0.5, v});
  return p;
}
Candidate cand(double lift, std::size_t goal, int option, double travel, double j6)
{
  Candidate c;
  c.lift_m = lift;
  c.goal_index = goal;
  c.option = option;
  c.joint_travel = travel;
  c.joint6_travel = j6;
  return c;
}
}  // namespace

TEST(TransitUnwind, LiftCandidatesAreFiveCentimetreSteps)
{
  const auto def = transit_unwind::liftCandidates(0.20);
  ASSERT_EQ(def.size(), 4U);
  EXPECT_NEAR(def.front(), 0.05, 1e-12);
  EXPECT_NEAR(def.back(), 0.20, 1e-12);
  EXPECT_EQ(transit_unwind::liftCandidates(0.05).size(), 1U);
  EXPECT_EQ(transit_unwind::liftCandidates(0.30).size(), 6U);
  EXPECT_EQ(transit_unwind::liftCandidates(0.12).size(), 2U);  // 0.05, 0.10
  // out of scope: no candidates at all (fail closed)
  EXPECT_TRUE(transit_unwind::liftCandidates(0.04).empty());
  EXPECT_TRUE(transit_unwind::liftCandidates(0.31).empty());
  EXPECT_TRUE(transit_unwind::liftCandidates(std::nan("")).empty());
  EXPECT_TRUE(transit_unwind::maxLiftValid(0.05));
  EXPECT_TRUE(transit_unwind::maxLiftValid(0.30));
  EXPECT_FALSE(transit_unwind::maxLiftValid(0.0));
  EXPECT_FALSE(transit_unwind::maxLiftValid(-0.1));
}

TEST(TransitUnwind, GoalBranchReachableByNetJoint6Change)
{
  // left face end joint6 ~143 deg: the 15 deg branch is a legal (128 deg) transit, the -165 deg one needs 308 deg
  const double start = 143.0 * kPi / 180.0;
  EXPECT_TRUE(transit_unwind::goalBranchReachable(start, 15.0 * kPi / 180.0));
  EXPECT_FALSE(transit_unwind::goalBranchReachable(start, -165.0 * kPi / 180.0));
  EXPECT_TRUE(transit_unwind::goalBranchReachable(0.0, kPi));
  EXPECT_FALSE(transit_unwind::goalBranchReachable(0.0, kPi + 1e-3));
  EXPECT_FALSE(transit_unwind::goalBranchReachable(std::nan(""), 0.0));
}

TEST(TransitUnwind, JointTravelIsTheGateFormula)
{
  // the wrist gate sums |dq6| over consecutive waypoints, so a back-and-forth counts every pass
  EXPECT_NEAR(transit_unwind::jointTravel(pathJoint6({0.0, 1.0, 0.0, 1.0}), 5), 3.0, 1e-12);
  EXPECT_NEAR(transit_unwind::jointTravel(pathJoint6({2.5, 0.27}), 5), 2.23, 1e-12);
  EXPECT_DOUBLE_EQ(transit_unwind::jointTravel(pathJoint6({1.0}), 5), 0.0);
  EXPECT_TRUE(std::isinf(transit_unwind::jointTravel(pathJoint6({0.0, 1.0}), 6)));  // joint index out of range
  EXPECT_NEAR(transit_unwind::totalJointTravel({{0.0, 0.0}, {1.0, -2.0}, {1.0, 0.0}}), 5.0, 1e-12);
}

TEST(TransitUnwind, ConcatenationKeepsTheSumAndSplittingCannotBypassTheGate)
{
  // three legs of 1.2 rad of joint6 each: every leg is far below pi, the TOTAL (3.6 rad) is not
  const Path leg1 = pathJoint6({0.0, 0.6, 1.2});
  const Path leg2 = pathJoint6({1.2, 1.8, 2.4});
  const Path leg3 = pathJoint6({2.4, 3.0, 3.6});
  const Path all = transit_unwind::concatenateLegs({leg1, leg2, leg3});
  ASSERT_EQ(all.size(), 7U);  // 3 + 2 + 2: the repeated first waypoint of a later leg is dropped
  const double sum = transit_unwind::jointTravel(leg1, 5) + transit_unwind::jointTravel(leg2, 5) +
    transit_unwind::jointTravel(leg3, 5);
  EXPECT_NEAR(transit_unwind::jointTravel(all, 5), sum, 1e-12);
  EXPECT_NEAR(transit_unwind::jointTravel(all, 5), 3.6, 1e-12);
  EXPECT_LT(transit_unwind::jointTravel(leg1, 5), kPi);
  Candidate c;
  c.joint_travel = transit_unwind::totalJointTravel(all);
  c.joint6_travel = transit_unwind::jointTravel(all, 5);
  EXPECT_FALSE(transit_unwind::acceptable(c));  // total > pi: refused although each leg alone passes
  // empty and one-point legs are ignored
  EXPECT_EQ(transit_unwind::concatenateLegs({Path{}, leg1, pathJoint6({9.0}), leg2}).size(), 5U);
  EXPECT_TRUE(transit_unwind::concatenateLegs({}).empty());
}

TEST(TransitUnwind, AcceptanceIsTheTotalJoint6TravelRule)
{
  EXPECT_TRUE(transit_unwind::acceptable(cand(0.1, 0, 0, 5.0, kPi)));          // exactly pi is allowed (gate: > pi refuses)
  EXPECT_FALSE(transit_unwind::acceptable(cand(0.1, 0, 0, 5.0, kPi + 1e-6)));
  EXPECT_FALSE(transit_unwind::acceptable(cand(0.1, 0, 0, 5.0, std::nan(""))));
  EXPECT_FALSE(transit_unwind::acceptable(cand(0.1, 0, 0, std::nan(""), 1.0)));
  EXPECT_FALSE(transit_unwind::acceptable(cand(0.1, 0, 0, 5.0, -0.1)));
}

TEST(TransitUnwind, SelectionPicksTheSmallestTotalJointTravelAmongAcceptable)
{
  std::vector<Candidate> v;
  EXPECT_FALSE(transit_unwind::selectBest(v).has_value());
  v.push_back(cand(0.05, 0, 0, 4.0, 3.5));   // cheapest but winds the wrist: never selected
  v.push_back(cand(0.05, 0, 1, 6.0, 2.0));
  v.push_back(cand(0.10, 1, 0, 5.5, 2.5));
  v.push_back(cand(0.15, 0, 2, 7.0, 1.0));
  auto best = transit_unwind::selectBest(v);
  ASSERT_TRUE(best.has_value());
  EXPECT_EQ(*best, 2U);
  // only unacceptable candidates: none
  EXPECT_FALSE(transit_unwind::selectBest({cand(0.05, 0, 0, 1.0, 4.0), cand(0.1, 0, 0, 2.0, 3.2)}).has_value());
}

TEST(TransitUnwind, SelectionTiesPreferLowerLiftThenGoalThenOption)
{
  std::vector<Candidate> v{
    cand(0.15, 0, 0, 5.0, 2.0), cand(0.10, 1, 0, 5.0, 2.0), cand(0.10, 0, 2, 5.0, 2.0), cand(0.10, 0, 1, 5.0, 2.0)};
  const auto best = transit_unwind::selectBest(v);
  ASSERT_TRUE(best.has_value());
  EXPECT_EQ(*best, 3U);  // lift 0.10, goal 0, option 1
  // the order of the list does not change the winner
  std::vector<Candidate> r(v.rbegin(), v.rend());
  const auto best_r = transit_unwind::selectBest(r);
  ASSERT_TRUE(best_r.has_value());
  EXPECT_EQ(r[*best_r].option, 1);
  EXPECT_NEAR(r[*best_r].lift_m, 0.10, 1e-12);
}

TEST(TransitUnwind, WristWaypointOptions)
{
  const std::vector<double> lifted{0.1, 0.2, 0.3, 0.4, 0.5, 2.5};
  const std::vector<double> goal{0.9, 0.8, 0.7, 1.4, -0.5, 0.27};
  const auto a = transit_unwind::wristWaypoint(lifted, goal, 3U, 4U, 5U, WristOption::kJoint6);
  EXPECT_DOUBLE_EQ(a[5], 0.27);
  EXPECT_DOUBLE_EQ(a[3], 0.4);
  EXPECT_DOUBLE_EQ(a[4], 0.5);
  const auto b = transit_unwind::wristWaypoint(lifted, goal, 3U, 4U, 5U, WristOption::kJoint4And6);
  EXPECT_DOUBLE_EQ(b[3], 1.4);
  EXPECT_DOUBLE_EQ(b[4], 0.5);
  EXPECT_DOUBLE_EQ(b[5], 0.27);
  const auto c = transit_unwind::wristWaypoint(lifted, goal, 3U, 4U, 5U, WristOption::kWholeWrist);
  EXPECT_DOUBLE_EQ(c[4], -0.5);
  EXPECT_DOUBLE_EQ(c[0], 0.1);  // arm joints stay at the lifted pose
  EXPECT_DOUBLE_EQ(c[1], 0.2);
  EXPECT_DOUBLE_EQ(c[2], 0.3);
  // missing joint4/5 indices: only joint6 is available
  const auto d = transit_unwind::wristWaypoint(lifted, goal, std::nullopt, std::nullopt, 5U, WristOption::kWholeWrist);
  EXPECT_DOUBLE_EQ(d[3], 0.4);
  EXPECT_DOUBLE_EQ(d[5], 0.27);
}

TEST(TransitUnwind, DescriptionNamesLiftWristAndReason)
{
  const std::string text = transit_unwind::describe(0.10, 2.4958, 0.2653, 0.2653, 2.2305, "every branch collides");
  EXPECT_NE(text.find("transit detour via unwind waypoint"), std::string::npos);
  EXPECT_NE(text.find("lift 100.0 mm"), std::string::npos);
  EXPECT_NE(text.find("joint6 2.4958->0.2653 rad"), std::string::npos);
  EXPECT_NE(text.find("total joint6 travel 2.2305 rad"), std::string::npos);
  EXPECT_NE(text.find("(direct refused: every branch collides)"), std::string::npos);
}

// ---- review 2026-10-22 rules ----------------------------------------------------------------------------------------
namespace
{
using transit_unwind::SplinePoint;
SplinePoint pt(double t, std::vector<double> q, std::vector<double> v = {}, std::vector<double> a = {})
{
  SplinePoint p;
  p.time_s = t;
  p.q = std::move(q);
  p.v = std::move(v);
  p.a = std::move(a);
  return p;
}
}  // namespace

TEST(TransitUnwindReview, ClearanceFloorIsFixedFiveMillimetres)
{
  EXPECT_DOUBLE_EQ(transit_unwind::kClearanceFloorM, 0.005);
  EXPECT_TRUE(transit_unwind::clearanceOk(0.005));
  EXPECT_TRUE(transit_unwind::clearanceOk(0.04));
  EXPECT_FALSE(transit_unwind::clearanceOk(0.00499));
  EXPECT_FALSE(transit_unwind::clearanceOk(0.0));
  EXPECT_FALSE(transit_unwind::clearanceOk(-0.001));          // penetration
  EXPECT_FALSE(transit_unwind::clearanceOk(std::nan("")));
  // a lifted pose with 4 mm is rejected; there is no relative threshold that could lower the floor
  EXPECT_FALSE(transit_unwind::clearanceOk(0.5 * 0.004));
}

TEST(TransitUnwindReview, ContactAllowanceRefusal)
{
  EXPECT_TRUE(transit_unwind::contactAllowanceRefusal({}, "detected_carton").empty());
  EXPECT_TRUE(transit_unwind::contactAllowanceRefusal({{"link6", false}, {"xeg32_tool_tip", false}}, "detected_carton").empty());
  const std::string text = transit_unwind::contactAllowanceRefusal(
    {{"link6", false}, {"xeg32_tool_tip", true}, {"knife_c_blade", true}}, "detected_carton");
  EXPECT_NE(text.find("xeg32_tool_tip"), std::string::npos);
  EXPECT_NE(text.find("knife_c_blade"), std::string::npos);
  EXPECT_EQ(text.find("link6"), std::string::npos);
  EXPECT_NE(text.find("detected_carton"), std::string::npos);
  EXPECT_NE(text.find("refused"), std::string::npos);
}

TEST(TransitUnwindReview, JoinStopConditions)
{
  EXPECT_TRUE(transit_unwind::isFullStop({0.0, 0.0}, {0.0, 0.0}));
  EXPECT_FALSE(transit_unwind::isFullStop({0.0, 1e-9}, {0.0, 0.0}));         // tolerance 0: exact
  EXPECT_FALSE(transit_unwind::isFullStop({0.0, 0.0}, {0.0, -0.2}));         // a leg end that keeps its acceleration
  EXPECT_TRUE(transit_unwind::isFullStop({0.0, 1e-9}, {0.0, 0.0}, 1e-6));
  EXPECT_FALSE(transit_unwind::isFullStop({std::nan("")}, {0.0}));
}

TEST(TransitUnwindReview, KeptAccelerationAtAJoinMovesAJointThatShouldStayStillButAFullStopDoesNot)
{
  // joint 1 must stay at 0.5 across the join; the previous leg ended with a = -0.5 rad/s^2 and the next leg starts
  // with a = +0.5: the stitched waypoint keeps the first one (v = 0)
  const SplinePoint kept = pt(0.0, {0.0, 0.5}, {0.0, 0.0}, {0.0, -0.5});
  const SplinePoint next = pt(0.2, {0.1, 0.5}, {0.0, 0.0}, {0.0, 0.0});
  double worst = 0.0;
  for (int k = 0; k <= 20; ++k) worst = std::max(worst, std::abs(transit_unwind::sampleSegment(kept, next, k / 20.0)[1] - 0.5));
  EXPECT_GT(worst, 1e-4);  // the still joint wanders
  const SplinePoint stop = pt(0.0, {0.0, 0.5}, {0.0, 0.0}, {0.0, 0.0});
  worst = 0.0;
  for (int k = 0; k <= 20; ++k) worst = std::max(worst, std::abs(transit_unwind::sampleSegment(stop, next, k / 20.0)[1] - 0.5));
  EXPECT_LT(worst, 1e-12);  // explicit stop: it stays exactly still
}

TEST(TransitUnwindReview, SplineSamplerFollowsTheControllerInterpolation)
{
  // rest-to-rest quintic: ends exact, midpoint exactly half, monotone
  const SplinePoint a = pt(0.0, {0.0}, {0.0}, {0.0});
  const SplinePoint b = pt(2.0, {1.0}, {0.0}, {0.0});
  EXPECT_NEAR(transit_unwind::sampleSegment(a, b, 0.0)[0], 0.0, 1e-12);
  EXPECT_NEAR(transit_unwind::sampleSegment(a, b, 1.0)[0], 1.0, 1e-9);
  EXPECT_NEAR(transit_unwind::sampleSegment(a, b, 0.5)[0], 0.5, 1e-9);
  double previous = -1.0;
  for (int k = 0; k <= 50; ++k) {
    const double v = transit_unwind::sampleSegment(a, b, k / 50.0)[0];
    EXPECT_GE(v, previous - 1e-12);
    previous = v;
  }
  // velocities only: cubic Hermite; nothing: linear
  const SplinePoint c = pt(0.0, {0.0}, {0.0}), d = pt(2.0, {1.0}, {0.0});
  EXPECT_NEAR(transit_unwind::sampleSegment(c, d, 0.5)[0], 0.5, 1e-12);
  EXPECT_NEAR(transit_unwind::sampleSegment(pt(0.0, {0.0}), pt(2.0, {1.0}), 0.25)[0], 0.25, 1e-12);
  // matches a hand-computed quintic with non-zero end velocity: q(t) with v1 = 0.5 overshoots less than a cubic
  const SplinePoint e = pt(0.0, {0.0}, {0.0}, {0.0}), f = pt(1.0, {1.0}, {2.0}, {0.0});
  EXPECT_NEAR(transit_unwind::sampleSegment(e, f, 1.0)[0], 1.0, 1e-9);
}

TEST(TransitUnwindReview, DenseSamplingBoundsStepsAndMeasuresTheTrueJoint6Travel)
{
  // joint 1 (index 0) goes 0 -> 1 -> 0.9 with a fast end velocity at the middle waypoint: the spline overshoots 1.0
  std::vector<SplinePoint> pts{
    pt(0.0, {0.0}, {0.0}, {0.0}), pt(1.0, {1.0}, {3.0}, {0.0}), pt(2.0, {0.9}, {0.0}, {0.0})};
  const auto dense = transit_unwind::sampleDense(pts);
  ASSERT_GE(dense.q.size(), 100U);
  EXPECT_DOUBLE_EQ(dense.q.front()[0], 0.0);
  EXPECT_DOUBLE_EQ(dense.q.back()[0], 0.9);
  EXPECT_EQ(dense.q.size(), dense.interval_end.size());
  EXPECT_LE(transit_unwind::maxJointJump(dense.q), transit_unwind::kDenseJointStepRad * 1.5);
  const Path waypoints{{0.0}, {1.0}, {0.9}};
  EXPECT_NEAR(transit_unwind::jointTravel(waypoints, 0), 1.1, 1e-12);
  EXPECT_GT(transit_unwind::jointTravel(dense.q, 0), 1.1 + 1e-4);  // the dense measure sees the overshoot
  // the time and joint step bounds
  EXPECT_GE(transit_unwind::denseIntervals(pt(0.0, {0.0}), pt(1.0, {0.0})), 100U);               // 1 s at 0.01 s
  EXPECT_GE(transit_unwind::denseIntervals(pt(0.0, {0.0}), pt(0.01, {0.3})), 100U);              // 0.3 rad at 0.003 rad
  EXPECT_EQ(transit_unwind::denseIntervals(pt(0.0, {0.0}), pt(0.0, {0.0})), 1U);
  EXPECT_NEAR(transit_unwind::maxJointJump({{0.0, 0.0}, {0.02, -0.07}, {0.02, -0.07}}), 0.07, 1e-12);
  EXPECT_GT(transit_unwind::maxJointJump({{0.0}, {0.06}}), transit_unwind::kMaxJointJumpRad);
}

TEST(TransitUnwindReview, LiftLineCheck)
{
  auto at = [](double z, double x = 0.0, double yaw = 0.0) {
    Eigen::Isometry3d p = Eigen::Isometry3d::Identity();
    p.translation() = Eigen::Vector3d(0.3 + x, 0.1, 0.05 + z);
    p.linear() = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    return p;
  };
  const Eigen::Vector3d up = Eigen::Vector3d::UnitZ();
  std::vector<Eigen::Isometry3d> good;
  for (int k = 0; k <= 10; ++k) good.push_back(at(0.1 * k / 10.0));
  EXPECT_TRUE(transit_unwind::checkLiftLine(good, up, 0.1).ok);
  // 2 mm sideways bulge
  auto bulge = good;
  bulge[5] = at(0.05, 0.002);
  const auto r1 = transit_unwind::checkLiftLine(bulge, up, 0.1);
  EXPECT_FALSE(r1.ok);
  EXPECT_NEAR(r1.max_line_deviation_m, 0.002, 1e-9);
  // 2 degrees of yaw
  auto turned = good;
  turned[7] = at(0.07, 0.0, 2.0 * kPi / 180.0);
  EXPECT_FALSE(transit_unwind::checkLiftLine(turned, up, 0.1).ok);
  // 0.5 degrees is fine
  auto slight = good;
  slight[7] = at(0.07, 0.0, 0.5 * kPi / 180.0);
  EXPECT_TRUE(transit_unwind::checkLiftLine(slight, up, 0.1).ok);
  // overshoots the lift, stops short, goes down
  auto over = good;
  over.push_back(at(0.103));
  EXPECT_FALSE(transit_unwind::checkLiftLine(over, up, 0.1).ok);
  std::vector<Eigen::Isometry3d> shortlift(good.begin(), good.begin() + 8);
  EXPECT_FALSE(transit_unwind::checkLiftLine(shortlift, up, 0.1).ok);
  auto down = good;
  down.insert(down.begin() + 1, at(-0.01));
  EXPECT_FALSE(transit_unwind::checkLiftLine(down, up, 0.1).ok);
  // degenerate input fails closed
  EXPECT_FALSE(transit_unwind::checkLiftLine({at(0.0)}, up, 0.1).ok);
  EXPECT_FALSE(transit_unwind::checkLiftLine(good, Eigen::Vector3d(0.0, 0.0, 2.0), 0.1).ok);
}

TEST(TransitUnwindReview, BudgetAndDeadline)
{
  EXPECT_DOUBLE_EQ(transit_unwind::remainingBudget(15.0, 4.0), 11.0);
  EXPECT_DOUBLE_EQ(transit_unwind::remainingBudget(15.0, 15.0), 0.0);
  EXPECT_DOUBLE_EQ(transit_unwind::remainingBudget(15.0, 20.0), 0.0);  // never negative: no fresh budget
  EXPECT_DOUBLE_EQ(transit_unwind::remainingBudget(std::nan(""), 1.0), 0.0);
  EXPECT_FALSE(transit_unwind::deadlineExpired(15.0, 14.9));
  EXPECT_TRUE(transit_unwind::deadlineExpired(15.0, 15.0));
  EXPECT_TRUE(transit_unwind::deadlineExpired(0.0, 0.0));
}

TEST(TransitUnwindReview, RankingIsTheValidationOrder)
{
  std::vector<Candidate> v{
    cand(0.10, 0, 0, 6.0, 2.0), cand(0.05, 0, 0, 5.0, 3.5), cand(0.05, 0, 1, 5.5, 2.0), cand(0.05, 1, 0, 5.5, 2.0)};
  const auto order = transit_unwind::rankCandidates(v);
  ASSERT_EQ(order.size(), 3U);  // the 3.5 rad candidate is not acceptable
  EXPECT_EQ(order[0], 2U);      // 5.5 rad, goal 0, option 1
  EXPECT_EQ(order[1], 3U);      // 5.5 rad, goal 1
  EXPECT_EQ(order[2], 0U);
  EXPECT_EQ(*transit_unwind::selectBest(v), 2U);
}
