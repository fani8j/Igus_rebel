// Tests for the pure decision maths of the left cut gate trim (left_cut_gate_trim.hpp).
#include <gtest/gtest.h>

#include <cmath>
#include <limits>

#include "igus_rebel_moveit_config/left_cut_gate_trim.hpp"

namespace
{
using left_cut_gate_trim::Decision;
using left_cut_gate_trim::End;
using left_cut_gate_trim::Limits;
using left_cut_gate_trim::Request;

// Extended span 270 mm, kept 0 .. 270 mm, minimum kept length 100 mm, clearance 10 mm.
Request baseRequest()
{
  Request request;
  request.max_retries = 3;
  request.kept_lo_m = 0.0;
  request.kept_hi_m = 0.270;
  request.clearance_m = 0.010;
  request.min_kept_length_m = 0.100;
  request.span_length_m = 0.270;
  return request;
}
}  // namespace

TEST(LeftCutGateTrim, ParameterRanges)
{
  EXPECT_TRUE(left_cut_gate_trim::validateParams(3, 0.010).empty());
  EXPECT_TRUE(left_cut_gate_trim::validateParams(0, 0.0).empty());
  EXPECT_TRUE(left_cut_gate_trim::validateParams(5, 0.05).empty());
  EXPECT_FALSE(left_cut_gate_trim::validateParams(-1, 0.010).empty());
  EXPECT_FALSE(left_cut_gate_trim::validateParams(6, 0.010).empty());
  EXPECT_FALSE(left_cut_gate_trim::validateParams(3, -0.001).empty());
  EXPECT_FALSE(left_cut_gate_trim::validateParams(3, 0.051).empty());
  EXPECT_FALSE(left_cut_gate_trim::validateParams(3, std::nan("")).empty());
}

TEST(LeftCutGateTrim, LimitsAllowStationsInsideOnly)
{
  Limits limits;
  EXPECT_FALSE(limits.any());
  EXPECT_TRUE(limits.stationAllowed(-1.0));
  EXPECT_TRUE(limits.stationAllowed(10.0));
  limits.max_end_m = 0.257;
  EXPECT_TRUE(limits.any());
  EXPECT_TRUE(limits.stationAllowed(0.257));  // on the limit
  EXPECT_FALSE(limits.stationAllowed(0.2571));
  limits.min_start_m = 0.020;
  EXPECT_FALSE(limits.stationAllowed(0.0199));
  EXPECT_TRUE(limits.stationAllowed(0.020));
}

TEST(LeftCutGateTrim, TravelCoordinateProjectsOntoTheStroke)
{
  const Eigen::Vector3d origin(0.5, 0.2, 0.1);
  const Eigen::Vector3d dir = Eigen::Vector3d(-1.0, 0.0, 0.0);
  // a point 0.120 m along the travel and 3 mm off the line sideways / up
  const Eigen::Vector3d point = origin + 0.120 * dir + Eigen::Vector3d(0.0, 0.003, 0.002);
  EXPECT_NEAR(left_cut_gate_trim::travelCoordinate(point, origin, dir), 0.120, 1e-12);
}

TEST(LeftCutGateTrim, ContactNearTheEndShortensTheEnd)
{
  // the real case: contact 13 mm before the finish end of a 297 mm stroke
  Request request = baseRequest();
  request.kept_hi_m = request.span_length_m = 0.2972;
  request.contact_s_m = 0.2972 - 0.013;
  const Decision decision = left_cut_gate_trim::decide(request);
  ASSERT_TRUE(decision.trim) << decision.reason;
  EXPECT_EQ(decision.end, End::kEnd);
  EXPECT_NEAR(decision.limits.max_end_m, 0.2972 - 0.013 - 0.010, 1e-12);
  EXPECT_FALSE(decision.limits.hasStart());
  EXPECT_NEAR(decision.removed_at_least_m, 0.023, 1e-12);
}

TEST(LeftCutGateTrim, ContactNearTheStartShortensTheStart)
{
  Request request = baseRequest();
  request.contact_s_m = 0.030;
  const Decision decision = left_cut_gate_trim::decide(request);
  ASSERT_TRUE(decision.trim) << decision.reason;
  EXPECT_EQ(decision.end, End::kStart);
  EXPECT_NEAR(decision.limits.min_start_m, 0.040, 1e-12);
  EXPECT_FALSE(decision.limits.hasEnd());
  EXPECT_NEAR(decision.removed_at_least_m, 0.040, 1e-12);
}

TEST(LeftCutGateTrim, TieShortensTheTravelEnd)
{
  Request request = baseRequest();
  request.contact_s_m = 0.135;
  const Decision decision = left_cut_gate_trim::decide(request);
  ASSERT_TRUE(decision.trim);
  EXPECT_EQ(decision.end, End::kEnd);
}

TEST(LeftCutGateTrim, ZeroClearanceStillRemovesTheEndStation)
{
  Request request = baseRequest();
  request.clearance_m = 0.0;
  request.contact_s_m = request.kept_hi_m;  // exactly on the last station
  Decision decision = left_cut_gate_trim::decide(request);
  ASSERT_TRUE(decision.trim);
  EXPECT_LT(decision.limits.max_end_m, request.kept_hi_m);
  EXPECT_FALSE(decision.limits.stationAllowed(request.kept_hi_m));
  request.contact_s_m = request.kept_lo_m;
  decision = left_cut_gate_trim::decide(request);
  ASSERT_TRUE(decision.trim);
  EXPECT_EQ(decision.end, End::kStart);
  EXPECT_FALSE(decision.limits.stationAllowed(request.kept_lo_m));
}

TEST(LeftCutGateTrim, LimitsAccumulateAndOnlyTighten)
{
  Request request = baseRequest();
  request.contact_s_m = 0.250;
  Decision first = left_cut_gate_trim::decide(request);
  ASSERT_TRUE(first.trim);
  EXPECT_NEAR(first.limits.max_end_m, 0.240, 1e-12);

  // second contact at the same end of the shorter plan tightens further
  request.current = first.limits;
  request.retries_done = 1;
  request.kept_hi_m = 0.240;
  request.contact_s_m = 0.200;
  Decision second = left_cut_gate_trim::decide(request);
  ASSERT_TRUE(second.trim) << second.reason;
  EXPECT_EQ(second.end, End::kEnd);
  EXPECT_NEAR(second.limits.max_end_m, 0.190, 1e-12);

  // a contact at the OTHER end adds a start limit and keeps the end limit
  request.current = second.limits;
  request.retries_done = 2;
  request.kept_hi_m = 0.190;
  request.contact_s_m = 0.015;
  Decision third = left_cut_gate_trim::decide(request);
  ASSERT_TRUE(third.trim) << third.reason;
  EXPECT_EQ(third.end, End::kStart);
  EXPECT_NEAR(third.limits.min_start_m, 0.025, 1e-12);
  EXPECT_NEAR(third.limits.max_end_m, 0.190, 1e-12);
}

TEST(LeftCutGateTrim, GivesUpWhenTheBudgetIsSpent)
{
  Request request = baseRequest();
  request.contact_s_m = 0.250;
  request.retries_done = 3;
  const Decision decision = left_cut_gate_trim::decide(request);
  EXPECT_FALSE(decision.trim);
  EXPECT_NE(decision.reason.find("budget"), std::string::npos);
  request.retries_done = 0;
  request.max_retries = 0;
  EXPECT_FALSE(left_cut_gate_trim::decide(request).trim);
}

TEST(LeftCutGateTrim, GivesUpWhenTheWindowGetsTooShort)
{
  Request request = baseRequest();
  request.min_kept_length_m = 0.250;
  request.contact_s_m = 0.200;  // limit 0.190 < 0.250 minimum
  const Decision decision = left_cut_gate_trim::decide(request);
  EXPECT_FALSE(decision.trim);
  EXPECT_NE(decision.reason.find("minimum kept length"), std::string::npos);
}

TEST(LeftCutGateTrim, GivesUpOnAContactOutsideTheStroke)
{
  Request request = baseRequest();
  request.contact_s_m = 0.290;  // 20 mm beyond the kept end
  EXPECT_FALSE(left_cut_gate_trim::decide(request).trim);
  request.contact_s_m = -0.020;
  EXPECT_FALSE(left_cut_gate_trim::decide(request).trim);
  request.contact_s_m = std::nan("");
  EXPECT_FALSE(left_cut_gate_trim::decide(request).trim);
  // a hair outside (interpolation / lateral offset) is clamped, not refused
  request.contact_s_m = request.kept_hi_m + 0.001;
  const Decision decision = left_cut_gate_trim::decide(request);
  ASSERT_TRUE(decision.trim);
  EXPECT_NEAR(decision.contact_s_m, request.kept_hi_m, 1e-12);
}

TEST(LeftCutGateTrim, OnlyTheForwardStrokeStageQualifies)
{
  using left_cut_gate_trim::isForwardLeftStrokeStage;
  EXPECT_TRUE(isForwardLeftStrokeStage("cut left side cut [ompl_cartesian]"));
  EXPECT_FALSE(isForwardLeftStrokeStage("cut left side cut patch [reverse] [ompl_cartesian]"));
  EXPECT_FALSE(isForwardLeftStrokeStage("finish pose outside left side cut [ompl_cartesian]"));
  EXPECT_FALSE(isForwardLeftStrokeStage("enter laterally into left side cut [ompl_cartesian]"));
  EXPECT_FALSE(isForwardLeftStrokeStage("approach pose outside left side cut [ompl_cartesian]"));
  EXPECT_FALSE(isForwardLeftStrokeStage("cut middle seam cut [ompl_cartesian]"));
  EXPECT_FALSE(isForwardLeftStrokeStage("cut right side cut [ompl_cartesian]"));
  EXPECT_FALSE(isForwardLeftStrokeStage(""));
}

TEST(LeftCutGateTrim, PhysicalEndNames)
{
  using left_cut_gate_trim::physicalEndName;
  EXPECT_STREQ(physicalEndName(End::kStart, false), "back");
  EXPECT_STREQ(physicalEndName(End::kEnd, false), "front");
  EXPECT_STREQ(physicalEndName(End::kStart, true), "front");
  EXPECT_STREQ(physicalEndName(End::kEnd, true), "back");
}
