#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <string>

#include "igus_rebel/MoveSupervisor.hpp"

namespace
{
using namespace Igus::Move;  // NOLINT
constexpr double kDeg = 0.0174532925199432957;

const std::array<std::string,
  6> kNames = {"joint1", "joint2", "joint3", "joint4", "joint5", "joint6"};

GoalRequest ValidRequest()
{
  GoalRequest request;
  request.jointNames = {kNames.begin(), kNames.end()};
  request.startPositions = {0.1, 0.2, 0.3, 0.0, 0.0, 0.0};
  request.goalPositions = {0.2, 0.2, 0.3, 0.0, 0.0, 0.5};
  request.velocityPercent = 25.0;
  request.accelerationPercent = 40.0;
  request.startTolerance = 0.05;
  request.pathTolerance = 0.03;
  request.goalTolerance = 0.02;
  request.timeoutS = 0.0;
  return request;
}

bool Validate(const GoalRequest & request, std::string & reason, const Caps & caps = Caps{})
{
  Joints start{};
  Joints goal{};
  return ValidateGoalRequest(request, kNames, caps, start, goal, reason);
}
}  // namespace

TEST(MoveValidate, AcceptsAWellFormedRequestAndFillsStartAndGoal)
{
  Joints start{};
  Joints goal{};
  std::string reason;
  const auto request = ValidRequest();
  ASSERT_TRUE(ValidateGoalRequest(request, kNames, Caps{}, start, goal, reason)) << reason;
  EXPECT_DOUBLE_EQ(start[0], 0.1);
  EXPECT_DOUBLE_EQ(goal[5], 0.5);
}

TEST(MoveValidate, RejectsWrongNamesAndSizes)
{
  std::string reason;
  auto request = ValidRequest();
  request.jointNames[2] = "joint9";
  EXPECT_FALSE(Validate(request, reason));
  EXPECT_NE(reason.find("joint9"), std::string::npos);

  request = ValidRequest();
  std::swap(request.jointNames[0], request.jointNames[1]);
  EXPECT_FALSE(Validate(request, reason));

  request = ValidRequest();
  request.jointNames.pop_back();
  EXPECT_FALSE(Validate(request, reason));

  request = ValidRequest();
  request.goalPositions.push_back(0.0);
  EXPECT_FALSE(Validate(request, reason));
}

TEST(MoveValidate, RejectsNonFiniteValues)
{
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  std::string reason;
  for (double bad : {nan, inf, -inf}) {
    auto request = ValidRequest();
    request.goalPositions[3] = bad;
    EXPECT_FALSE(Validate(request, reason)) << bad;
    request = ValidRequest();
    request.startPositions[1] = bad;
    EXPECT_FALSE(Validate(request, reason)) << bad;
    request = ValidRequest();
    request.velocityPercent = bad;
    EXPECT_FALSE(Validate(request, reason)) << bad;
    request = ValidRequest();
    request.accelerationPercent = bad;
    EXPECT_FALSE(Validate(request, reason)) << bad;
    request = ValidRequest();
    request.pathTolerance = bad;
    EXPECT_FALSE(Validate(request, reason)) << bad;
    request = ValidRequest();
    request.timeoutS = bad;
    EXPECT_FALSE(Validate(request, reason)) << bad;
  }
}

TEST(MoveValidate, VelocityAndAccelerationAboveTheCapAreRejectedNotClamped)
{
  std::string reason;
  auto request = ValidRequest();
  request.velocityPercent = 25.5;
  EXPECT_FALSE(Validate(request, reason));
  EXPECT_NE(reason.find("not clamped"), std::string::npos);
  request = ValidRequest();
  request.accelerationPercent = 40.5;
  EXPECT_FALSE(Validate(request, reason));
  request = ValidRequest();
  request.velocityPercent = 0.0;
  EXPECT_FALSE(Validate(request, reason));
  request = ValidRequest();
  request.accelerationPercent = -1.0;
  EXPECT_FALSE(Validate(request, reason));

  Caps caps;
  caps.maxVelocityPercent = kHardMaxVelocityPercent;
  request = ValidRequest();
  request.velocityPercent = 50.0;
  EXPECT_TRUE(Validate(request, reason, caps)) << reason;
  request.velocityPercent = 50.1;
  EXPECT_FALSE(Validate(request, reason, caps));
}

TEST(MoveValidate, TolerancesMustBePositiveAndBounded)
{
  std::string reason;
  for (int which = 0; which < 3; ++which) {
    for (double value : {0.0, -0.01, 1.0}) {
      auto request = ValidRequest();
      (which == 0 ? request.startTolerance : which == 1 ? request.pathTolerance :
      request.goalTolerance) = value;
      EXPECT_FALSE(Validate(request, reason)) << which << " " << value;
    }
  }
  auto request = ValidRequest();
  request.timeoutS = -1.0;
  EXPECT_FALSE(Validate(request, reason));
  request.timeoutS = 1e6;
  EXPECT_FALSE(Validate(request, reason));
  request.timeoutS = 5.0;
  EXPECT_TRUE(Validate(request, reason)) << reason;
}

TEST(MoveValidate, GoalMustBeInsideTheJointLimits)
{
  std::string reason;
  const auto & limits = HardwareJointLimits();
  for (std::size_t joint = 0; joint < 6; ++joint) {
    for (int side = 0; side < 2; ++side) {
      auto request = ValidRequest();
      request.startPositions = {0, 0, 0, 0, 0, 0};
      request.goalPositions = {0.1, 0, 0, 0, 0, 0};
      request.goalPositions[joint] = limits[joint][side] + (side == 0 ? -1.0 : 1.0) * 0.01;
      EXPECT_FALSE(Validate(request, reason)) << joint << " " << side;
      request.goalPositions[joint] = limits[joint][side];  // exactly at the limit is allowed
      EXPECT_TRUE(Validate(request, reason)) << joint << " " << side << ": " << reason;
    }
  }
  // An arm that is far outside its limits (bad calibration) must not be moved.
  auto request = ValidRequest();
  request.startPositions[4] = 2.5;
  request.goalPositions[4] = 1.0;
  EXPECT_FALSE(Validate(request, reason));
  // Caller-provided limits win.
  Caps caps;
  caps.jointLimits[0] = {-0.05, 0.15};
  request = ValidRequest();
  request.goalPositions[0] = 0.3;
  EXPECT_FALSE(Validate(request, reason, caps));
}

TEST(MoveValidate, ZeroLengthMoveIsRefused)
{
  std::string reason;
  auto request = ValidRequest();
  request.goalPositions = request.startPositions;
  EXPECT_FALSE(Validate(request, reason));
  request.goalPositions[0] += 0.0005;
  EXPECT_FALSE(Validate(request, reason));
  request.goalPositions[0] += 0.01;
  EXPECT_TRUE(Validate(request, reason)) << reason;
}

TEST(MoveModel, DurationIsAConservativeTrapezoid)
{
  // Long move at 25 % / 40 %: cruise 0.6 deg/s per % = 15 deg/s (measured 17.6 deg/s).
  const double distance = 60.0 * kDeg;
  const double duration = ModelDurationS(distance, 25.0, 40.0);
  const double measuredLike = 60.0 / 17.6 + 0.13 + 0.3;  // cruise + latency + ramps
  EXPECT_GT(duration, measuredLike);
  EXPECT_LT(duration, 3.0 * measuredLike);
  // Monotonic in distance, speed and acceleration.
  EXPECT_GT(ModelDurationS(2 * distance, 25.0, 40.0), duration);
  EXPECT_LT(ModelDurationS(distance, 50.0, 40.0), duration);
  EXPECT_LT(ModelDurationS(distance, 25.0, 40.0), ModelDurationS(distance, 25.0, 10.0));
  // Short move that never reaches cruise: triangle profile.
  const double tiny = ModelDurationS(0.01, 25.0, 40.0);
  EXPECT_GT(tiny, kModelStartLatencyS);
  EXPECT_LT(tiny, kModelStartLatencyS + 0.5);
  // The measured 30 deg move at vel 25 took 2.25 s including latency.
  EXPECT_GT(ModelDurationS(30.0 * kDeg, 25.0, 40.0), 2.25);
  EXPECT_GT(ModelDurationS(30.0 * kDeg, 100.0, 40.0), 0.94);
  EXPECT_GT(ModelDurationS(30.0 * kDeg, 50.0, 40.0), 1.3);
}

TEST(MoveSupervisorTest, TimeoutIsTwiceTheModelPlusOneSecondUnlessGiven)
{
  const Joints start{0, 0, 0, 0, 0, 0};
  const Joints goal{0.5, 0, 0, 0, 0, 0};
  MoveSupervisor model(start, goal, 25.0, 40.0, 0.03, 0.02);
  EXPECT_NEAR(model.TimeoutS(), 2.0 * model.ModelDurationS() + 1.0, 1e-9);
  MoveSupervisor given(start, goal, 25.0, 40.0, 0.03, 0.02, 7.5);
  EXPECT_DOUBLE_EQ(given.TimeoutS(), 7.5);
}

TEST(MoveSupervisorTest, ProgressAndDeviationAlongTheLine)
{
  const Joints start{0, 0, 0, 0, 0, 0};
  const Joints goal{1.0, 0.5, 0, 0, 0, -0.5};
  MoveSupervisor supervisor(start, goal, 25.0, 40.0, 0.03, 0.02);

  Sample sample = supervisor.Evaluate(start);
  EXPECT_NEAR(sample.progress, 0.0, 1e-12);
  EXPECT_NEAR(sample.deviation, 0.0, 1e-12);

  sample = supervisor.Evaluate({0.5, 0.25, 0, 0, 0, -0.25});
  EXPECT_NEAR(sample.progress, 0.5, 1e-12);
  EXPECT_NEAR(sample.deviation, 0.0, 1e-12);

  sample = supervisor.Evaluate(goal);
  EXPECT_NEAR(sample.progress, 1.0, 1e-12);

  // A joint that is not part of the move drifts off the line.
  sample = supervisor.Evaluate({0.5, 0.25, 0, 0, 0.04, -0.25});
  EXPECT_NEAR(sample.deviation, 0.04, 1e-12);
  // The synchronized joint spread of the real robot (progress differs by 0.015) is fine.
  sample = supervisor.Evaluate({0.5, 0.25 + 0.0075, 0, 0, 0, -0.25});
  EXPECT_LT(sample.deviation, 0.03);
}

TEST(MoveSupervisorTest, CheckFlagsDeviationOvershootBackwardsAndTimeout)
{
  const Joints start{0, 0, 0, 0, 0, 0};
  const Joints goal{1.0, 0, 0, 0, 0, 0};
  MoveSupervisor supervisor(start, goal, 25.0, 40.0, 0.03, 0.02, 5.0);

  EXPECT_EQ(supervisor.Check({0.4, 0, 0, 0, 0, 0}, 1.0), Verdict::Ok);
  EXPECT_EQ(supervisor.Check({0.4, 0, 0, 0.031, 0, 0}, 1.0), Verdict::PathDeviation);
  EXPECT_EQ(supervisor.Check({0.4, -0.031, 0, 0, 0, 0}, 1.0), Verdict::PathDeviation);
  EXPECT_EQ(supervisor.Check({0.4, 0, 0, 0, 0, 0}, 5.01), Verdict::Timeout);
  // Overshoot past the goal.
  EXPECT_EQ(supervisor.Check({1.0, 0, 0, 0, 0, 0}, 1.0), Verdict::Ok);
  EXPECT_EQ(supervisor.Check({1.01, 0, 0, 0, 0, 0}, 1.0), Verdict::Ok);
  EXPECT_EQ(supervisor.Check({1.03, 0, 0, 0, 0, 0}, 1.0), Verdict::ProgressOverrun);
  // Travelling backwards past the start.
  EXPECT_EQ(supervisor.Check({-0.03, 0, 0, 0, 0, 0}, 1.0), Verdict::ProgressOverrun);
  // Non-finite measurement fails closed.
  EXPECT_EQ(
    supervisor.Check({std::nan(""), 0, 0, 0, 0, 0}, 1.0), Verdict::PathDeviation);
  // Deviation wins over the timeout when both apply.
  EXPECT_EQ(supervisor.Check({0.4, 0, 0, 0.5, 0, 0}, 9.0), Verdict::PathDeviation);
}

TEST(MoveSupervisorTest, ShortMovesHaveANoiseFloorOnTheProgressMargin)
{
  const Joints start{0, 0, 0, 0, 0, 0};
  const Joints goal{0.01, 0, 0, 0, 0, 0};  // 10 mrad
  MoveSupervisor supervisor(start, goal, 25.0, 40.0, 0.03, 0.02);
  // 2 % of 10 mrad is 0.2 mrad; encoder noise of 1 mrad must not stop the move.
  EXPECT_EQ(supervisor.Check({0.011, 0, 0, 0, 0, 0}, 0.1), Verdict::Ok);
  EXPECT_EQ(supervisor.Check({-0.002, 0, 0, 0, 0, 0}, 0.1), Verdict::Ok);
  EXPECT_EQ(supervisor.Check({0.0135, 0, 0, 0, 0, 0}, 0.1), Verdict::ProgressOverrun);
}

TEST(MoveSupervisorTest, GoalReachedUsesTheGoalTolerance)
{
  const Joints start{0, 0, 0, 0, 0, 0};
  const Joints goal{1.0, 0, 0, 0, 0, 0.3};
  MoveSupervisor supervisor(start, goal, 25.0, 40.0, 0.03, 0.02);
  EXPECT_TRUE(supervisor.AtGoal(goal));
  EXPECT_TRUE(supervisor.AtGoal({1.019, 0, 0, 0, 0, 0.3}));
  EXPECT_FALSE(supervisor.AtGoal({1.021, 0, 0, 0, 0, 0.3}));
  EXPECT_FALSE(supervisor.AtGoal({1.0, 0, 0, 0, 0.03, 0.3}));
}
