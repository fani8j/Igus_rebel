#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <cstdint>
#include <string>
#include <vector>

#include "igus_rebel/RebelSafety.hpp"

namespace
{
using Igus::Safety::SteadyClock;
using std::chrono::milliseconds;
}  // namespace

TEST(JogClamp, ScalesVelocityLikeTheOriginalDriver)
{
  // jog % = 2.0 * deg/s
  const auto jog = Igus::Safety::JogPercentFromVelocity(10.0 * Igus::Safety::kDegToRad);
  EXPECT_NEAR(jog.percent, 20.0f, 1e-4);
  EXPECT_FALSE(jog.nonFinite);
  EXPECT_FALSE(jog.saturated);

  const auto negative = Igus::Safety::JogPercentFromVelocity(-0.1);
  EXPECT_NEAR(negative.percent, -2.0 * 0.1 / Igus::Safety::kDegToRad, 1e-3);

  EXPECT_EQ(Igus::Safety::JogPercentFromVelocity(0.0).percent, 0.0f);
}

TEST(JogScale, DefaultKeepsTodaysJogValues)
{
  // The optional jog_velocity_scale defaults to the historic 2.0: passing nothing, passing the
  // constant and the 2-argument call must give bit-identical results for any velocity.
  EXPECT_EQ(Igus::JOINT_VELOCITY_SCALE, 2.0);
  for (double v : {-0.785, -0.3054, -0.05, -1e-4, 0.0, 1e-4, 0.0611, 0.3054, 0.6, 0.785, 1.2}) {
    const auto implicit = Igus::Safety::JogPercentFromVelocity(v);
    const auto explicitScale = Igus::Safety::JogPercentFromVelocity(v, Igus::JOINT_VELOCITY_SCALE);
    const auto literal = Igus::Safety::JogPercentFromVelocity(v, 2.0);
    EXPECT_EQ(implicit.percent, explicitScale.percent) << v;
    EXPECT_EQ(implicit.percent, literal.percent) << v;
    EXPECT_EQ(implicit.saturated, literal.saturated) << v;
    const double expected = std::clamp(2.0 * v / Igus::Safety::kDegToRad, -100.0, 100.0);
    EXPECT_NEAR(implicit.percent, expected, 1e-3) << v;
  }
}

TEST(JogScale, CustomScaleChangesJogProportionally)
{
  const double v = 0.3054;  // rad/s
  const auto base = Igus::Safety::JogPercentFromVelocity(v);
  const auto fitted = Igus::Safety::JogPercentFromVelocity(v, 1.65);
  EXPECT_NEAR(fitted.percent, base.percent * 1.65 / 2.0, 1e-3);
  // The +-100 % clamp still applies with a larger scale.
  const auto big = Igus::Safety::JogPercentFromVelocity(0.7, Igus::kMaxJogVelocityScale);
  EXPECT_EQ(big.percent, 100.0f);
  EXPECT_TRUE(big.saturated);
}

TEST(JogScale, ParseAcceptsOnlySaneNumbers)
{
  double scale = 2.0;
  EXPECT_TRUE(Igus::Safety::ParseJogVelocityScale("1.65", scale));
  EXPECT_DOUBLE_EQ(scale, 1.65);
  EXPECT_TRUE(Igus::Safety::ParseJogVelocityScale("2", scale));
  EXPECT_DOUBLE_EQ(scale, 2.0);
  scale = 2.0;
  for (const char * bad : {"", "abc", "1.5x", "nan", "inf", "-1", "0", "0.4", "4.01", "100"}) {
    EXPECT_FALSE(Igus::Safety::ParseJogVelocityScale(bad, scale)) << bad;
    EXPECT_DOUBLE_EQ(scale, 2.0) << bad;  // untouched on failure
  }
}

TEST(JogClamp, ClampsToPlusMinusHundredPercent)
{
  // 50 deg/s is exactly 100 %.
  EXPECT_NEAR(
    Igus::Safety::JogPercentFromVelocity(50.0 * Igus::Safety::kDegToRad).percent, 100.0f, 1e-3);

  const auto high = Igus::Safety::JogPercentFromVelocity(10.0);
  EXPECT_EQ(high.percent, 100.0f);
  EXPECT_TRUE(high.saturated);

  const auto low = Igus::Safety::JogPercentFromVelocity(-10.0);
  EXPECT_EQ(low.percent, -100.0f);
  EXPECT_TRUE(low.saturated);

  const auto huge = Igus::Safety::JogPercentFromVelocity(std::numeric_limits<double>::max());
  EXPECT_EQ(huge.percent, 0.0f);  // scaled value overflows to inf -> treated as non-finite
  EXPECT_TRUE(huge.nonFinite);
}

TEST(JogClamp, NonFiniteBecomesZero)
{
  for (double value : {std::numeric_limits<double>::quiet_NaN(),
      std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()})
  {
    const auto jog = Igus::Safety::JogPercentFromVelocity(value);
    EXPECT_EQ(jog.percent, 0.0f);
    EXPECT_TRUE(jog.nonFinite);
    EXPECT_FALSE(jog.saturated);
  }
}

TEST(Watchdog, StaleWhenNeverWrittenOrTooOld)
{
  const auto now = SteadyClock::now();
  const auto timeout = Igus::Safety::kCommandWatchdogTimeout;
  EXPECT_EQ(timeout, milliseconds(50));

  EXPECT_TRUE(Igus::Safety::IsCommandStale(SteadyClock::time_point{}, now, timeout));
  EXPECT_FALSE(Igus::Safety::IsCommandStale(now, now, timeout));
  EXPECT_FALSE(Igus::Safety::IsCommandStale(now - milliseconds(10), now, timeout));
  EXPECT_FALSE(Igus::Safety::IsCommandStale(now - milliseconds(50), now, timeout));
  EXPECT_TRUE(Igus::Safety::IsCommandStale(now - milliseconds(51), now, timeout));
  EXPECT_TRUE(Igus::Safety::IsCommandStale(now - std::chrono::seconds(10), now, timeout));
  // A write that raced ahead of the sampled time is fresh, not stale.
  EXPECT_FALSE(Igus::Safety::IsCommandStale(now + milliseconds(1), now, timeout));
}

TEST(Watchdog, StatusStaleness)
{
  const auto now = SteadyClock::now();
  const auto timeout = Igus::Safety::kStatusStaleTimeout;
  EXPECT_EQ(timeout, milliseconds(200));
  EXPECT_TRUE(Igus::Safety::IsStatusStale(SteadyClock::time_point{}, now, timeout));
  EXPECT_FALSE(Igus::Safety::IsStatusStale(now - milliseconds(30), now, timeout));
  EXPECT_TRUE(Igus::Safety::IsStatusStale(now - milliseconds(201), now, timeout));
}

TEST(Watchdog, StatusHealthClassification)
{
  using Igus::Safety::ClassifyStatusAge;
  using Igus::Safety::StatusHealth;
  const auto stale = Igus::Safety::kStatusStaleTimeout;
  const auto coast = Igus::Safety::kStatusCoastTimeout;
  const auto latch = Igus::Safety::kStatusLatchTimeout;
  EXPECT_EQ(coast, milliseconds(400));
  EXPECT_EQ(latch, milliseconds(1000));
  const auto classify = [&](bool have, int ms) {
      return ClassifyStatusAge(have, milliseconds(ms), stale, coast, latch);
    };
  EXPECT_EQ(classify(true, 30), StatusHealth::Fresh);
  EXPECT_EQ(classify(true, 200), StatusHealth::Fresh);
  EXPECT_EQ(classify(true, 201), StatusHealth::Coast);
  EXPECT_EQ(classify(true, 400), StatusHealth::Coast);
  EXPECT_EQ(classify(true, 401), StatusHealth::Hold);
  EXPECT_EQ(classify(true, 1000), StatusHealth::Hold);
  EXPECT_EQ(classify(true, 1001), StatusHealth::Latch);
  EXPECT_EQ(classify(false, 0), StatusHealth::Latch);
}

TEST(Watchdog, GapResumeDecision)
{
  using Igus::Safety::GapResumeAllowed;
  using Igus::Safety::StatusHealth;
  // Coast-tier gaps resume regardless of motion; hold-tier only when idle; latch never.
  EXPECT_TRUE(GapResumeAllowed(StatusHealth::Fresh, true));
  EXPECT_TRUE(GapResumeAllowed(StatusHealth::Coast, true));
  EXPECT_TRUE(GapResumeAllowed(StatusHealth::Coast, false));
  EXPECT_FALSE(GapResumeAllowed(StatusHealth::Hold, true));
  EXPECT_TRUE(GapResumeAllowed(StatusHealth::Hold, false));
  EXPECT_FALSE(GapResumeAllowed(StatusHealth::Latch, false));
}

TEST(Watchdog, HoldMayOnlyResumeWhenIdle)
{
  using Igus::Safety::IsCommandMoving;
  using Igus::Safety::MotionDuringGap;
  EXPECT_FALSE(IsCommandMoving(0.0));
  EXPECT_FALSE(IsCommandMoving(0.02));
  EXPECT_TRUE(IsCommandMoving(0.021));
  EXPECT_TRUE(IsCommandMoving(std::numeric_limits<double>::quiet_NaN()));
  const std::chrono::nanoseconds lookback = Igus::Safety::kHoldMotionLookback;
  const std::int64_t status = 10'000'000'000;
  EXPECT_FALSE(MotionDuringGap(0, status, lookback));                       // never moved
  EXPECT_FALSE(MotionDuringGap(status - 101'000'000, status, lookback));    // long before
  EXPECT_TRUE(MotionDuringGap(status - 99'000'000, status, lookback));      // just before
  EXPECT_TRUE(MotionDuringGap(status + 500'000'000, status, lookback));     // during the gap
}

TEST(Velocity, DividesByActualStatusInterval)
{
  const auto t0 = SteadyClock::now();
  const std::array<double, 6> p0{0.0, 1.0, -1.0, 0.0, 0.0, 0.0};
  const std::array<double, 6> p1{0.01, 1.0, -1.02, 0.0, 0.0, 0.5};
  std::array<double, 6> velocity{};

  ASSERT_TRUE(
    Igus::Safety::EstimateVelocity(
      p0, t0, p1, t0 + milliseconds(25), Igus::Safety::kMinVelocityInterval, velocity));
  EXPECT_NEAR(velocity[0], 0.4, 1e-9);
  EXPECT_NEAR(velocity[1], 0.0, 1e-9);
  EXPECT_NEAR(velocity[2], -0.8, 1e-9);
  EXPECT_NEAR(velocity[5], 20.0, 1e-9);
}

TEST(Velocity, HoldsEstimateForTooShortIntervals)
{
  const auto t0 = SteadyClock::now();
  const std::array<double, 6> p0{};
  const std::array<double, 6> p1{1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  std::array<double, 6> velocity{7.0, 7.0, 7.0, 7.0, 7.0, 7.0};

  EXPECT_FALSE(
    Igus::Safety::EstimateVelocity(
      p0, t0, p1, t0 + milliseconds(1), Igus::Safety::kMinVelocityInterval, velocity));
  EXPECT_FALSE(
    Igus::Safety::EstimateVelocity(
      p0, t0, p1, t0, Igus::Safety::kMinVelocityInterval, velocity));
  EXPECT_FALSE(
    Igus::Safety::EstimateVelocity(
      p0, t0, p1, t0 - milliseconds(10), std::chrono::nanoseconds(0), velocity));
  for (double v : velocity) {
    EXPECT_EQ(v, 7.0);
  }
}

TEST(JointPositions, ConvertsDegreesAndRejectsNonFinite)
{
  Igus::CriMessages::Status status;
  status.posJointCurrent[0] = 90.0f;
  status.posJointCurrent[5] = -45.0f;
  status.posJointCurrent[6] = std::numeric_limits<float>::quiet_NaN();  // not a robot joint
  std::array<double, 6> positions{};
  ASSERT_TRUE(Igus::Safety::JointPositionsRad(status, positions));
  EXPECT_NEAR(positions[0], M_PI / 2.0, 1e-9);
  EXPECT_NEAR(positions[5], -M_PI / 4.0, 1e-9);

  std::array<double, 6> untouched{1, 2, 3, 4, 5, 6};
  status.posJointCurrent[3] = std::numeric_limits<float>::infinity();
  EXPECT_FALSE(Igus::Safety::JointPositionsRad(status, untouched));
  EXPECT_EQ(untouched[0], 1.0);
}

TEST(Override, ExpectedIsHundredPercent)
{
  EXPECT_TRUE(Igus::Safety::IsOverrideAsExpected(100.0f));
  EXPECT_TRUE(Igus::Safety::IsOverrideAsExpected(99.6f));
  EXPECT_FALSE(Igus::Safety::IsOverrideAsExpected(50.0f));
  EXPECT_FALSE(Igus::Safety::IsOverrideAsExpected(0.0f));
  EXPECT_FALSE(Igus::Safety::IsOverrideAsExpected(std::numeric_limits<float>::quiet_NaN()));
}

TEST(Formatting, AliveJogMatchesOriginalWireFormat)
{
  EXPECT_EQ(
    Igus::Safety::FormatAliveJog(12, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}),
    "CRISTART 12 ALIVEJOG 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 "
    "0.00000000 0.00000000 0.00000000 0.00000000 CRIEND\n");
  EXPECT_EQ(
    Igus::Safety::FormatAliveJog(
      9999, {100.0f, -100.0f, 12.5f, -0.25f, 0.0f, std::numeric_limits<float>::quiet_NaN()}),
    "CRISTART 9999 ALIVEJOG 100.00000000 -100.00000000 12.50000000 -0.25000000 0.00000000 "
    "0.00000000 0.00000000 0.00000000 0.00000000 CRIEND\n");
}

TEST(Formatting, CommandMatchesOriginalWireFormat)
{
  EXPECT_EQ(
    Igus::Safety::FormatCommand(7, "Override 100.0"),
    "CRISTART 7 CMD Override 100.0 CRIEND\n");
  EXPECT_EQ(
    Igus::Safety::FormatCommand(
      1,
      "SetActive true"),
    "CRISTART 1 CMD SetActive true CRIEND\n");
}

TEST(FaultGroupKinstate, OnlyJointLimitsAreLocalEverythingElseTripsTheGroup)
{
  using Igus::CriMessages::Kinstate;
  using Igus::Safety::IsGroupTrippingKinstate;
  using Igus::Safety::IsJointLimitKinstate;

  EXPECT_FALSE(IsGroupTrippingKinstate(Kinstate::NO_ERROR));
  EXPECT_FALSE(IsGroupTrippingKinstate(Kinstate::JOINT_LIMIT_MIN));
  EXPECT_FALSE(IsGroupTrippingKinstate(Kinstate::JOINT_LIMIT_MAX));
  EXPECT_TRUE(IsJointLimitKinstate(Kinstate::JOINT_LIMIT_MIN));
  EXPECT_TRUE(IsJointLimitKinstate(Kinstate::JOINT_LIMIT_MAX));
  EXPECT_FALSE(IsJointLimitKinstate(Kinstate::NO_ERROR));

  for (const Kinstate tripping : {
    Kinstate::CARTESIAN_SINGULARITY_CENTER, Kinstate::CARTESIAN_SINGULARITY_REACH,
    Kinstate::CARTESIAN_SINGULARITY_WRIST, Kinstate::TOOL_AT_VIRTUAL_BOX_LIMIT_1,
    Kinstate::TOOL_AT_VIRTUAL_BOX_LIMIT_2, Kinstate::TOOL_AT_VIRTUAL_BOX_LIMIT_3,
    Kinstate::TOOL_AT_VIRTUAL_BOX_LIMIT_4, Kinstate::TOOL_AT_VIRTUAL_BOX_LIMIT_5,
    Kinstate::TOOL_AT_VIRTUAL_BOX_LIMIT_6, Kinstate::MOTION_NOT_ALLOWED, Kinstate::UNKNOWN})
  {
    EXPECT_TRUE(IsGroupTrippingKinstate(tripping)) << static_cast<int>(tripping);
    EXPECT_FALSE(IsJointLimitKinstate(tripping)) << static_cast<int>(tripping);
  }
}

TEST(StartsAllCommandInterfaces, RequiresEverySixVelocityInterfaceOfThisComponent)
{
  using Igus::Safety::StartsAllCommandInterfaces;
  const std::array<std::string, 6> left = {
    "arm_left_joint1", "arm_left_joint2", "arm_left_joint3",
    "arm_left_joint4", "arm_left_joint5", "arm_left_joint6"};
  std::vector<std::string> all;
  for (const auto & joint : left) {
    all.push_back(joint + "/velocity");
  }
  EXPECT_TRUE(StartsAllCommandInterfaces(all, left, "velocity"));

  // The full list of a dual-arm switch contains both arms' interfaces.
  std::vector<std::string> both = {"arm_right_joint1/velocity", "arm_right_joint6/velocity"};
  both.insert(both.end(), all.begin(), all.end());
  EXPECT_TRUE(StartsAllCommandInterfaces(both, left, "velocity"));

  EXPECT_FALSE(StartsAllCommandInterfaces({}, left, "velocity"));
  // Another component's interfaces.
  EXPECT_FALSE(
    StartsAllCommandInterfaces(
      {"arm_right_joint1/velocity", "arm_right_joint2/velocity", "arm_right_joint3/velocity",
        "arm_right_joint4/velocity", "arm_right_joint5/velocity", "arm_right_joint6/velocity"},
      left, "velocity"));
  // A partial start.
  std::vector<std::string> partial(all.begin(), all.end() - 1);
  EXPECT_FALSE(StartsAllCommandInterfaces(partial, left, "velocity"));
  // The right joints but another interface type; and a prefix that merely starts with the name.
  EXPECT_FALSE(StartsAllCommandInterfaces(all, left, "position"));
  std::vector<std::string> lookalike;
  for (const auto & joint : left) {
    lookalike.push_back(joint + "x/velocity");
  }
  EXPECT_FALSE(StartsAllCommandInterfaces(lookalike, left, "velocity"));
  // Unset joint names never match.
  EXPECT_FALSE(StartsAllCommandInterfaces(all, std::array<std::string, 6>{}, "velocity"));
}

TEST(Formatting, MoveJointMatchesTheRecordedRealRobotCommand)
{
  // From raw_messages.log of the real-robot characterization (degrees, 3 decimals, e1..e3 = 0).
  EXPECT_EQ(
    Igus::Safety::FormatMoveJoint(82, {5.0, 0.01, 0.0, 0.02, 0.0, 0.0}, 10, 20),
    "CRISTART 82 CMD Move Joint 5.000 0.010 0.000 0.020 0.000 0.000 0 0 0 10 20 CRIEND\n");
  EXPECT_EQ(
    Igus::Safety::MoveJointCommandText({-0.0004, 0.01, 0.0, 0.02, 0.0, 0.0}, 30, 40),
    "Move Joint -0.000 0.010 0.000 0.020 0.000 0.000 0 0 0 30 40");
  EXPECT_EQ(
    Igus::Safety::MoveJointCommandText({1.23456, -179.9996, 90.0, 0.0, 0.0, 12.0005}, 25, 40),
    "Move Joint 1.235 -180.000 90.000 0.000 0.000 12.001 0 0 0 25 40");
  // Non-integral percentages keep 3 decimals.
  EXPECT_EQ(
    Igus::Safety::MoveJointCommandText({0, 0, 0, 0, 0, 0}, 12.5, 7.25),
    "Move Joint 0.000 0.000 0.000 0.000 0.000 0.000 0 0 0 12.500 7.250");
}

TEST(Formatting, MoveJointRejectsNonFiniteValues)
{
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  EXPECT_TRUE(Igus::Safety::FormatMoveJoint(1, {nan, 0, 0, 0, 0, 0}, 10, 20).empty());
  EXPECT_TRUE(Igus::Safety::FormatMoveJoint(1, {0, 0, 0, 0, 0, inf}, 10, 20).empty());
  EXPECT_TRUE(Igus::Safety::FormatMoveJoint(1, {0, 0, 0, 0, 0, 0}, nan, 20).empty());
  EXPECT_TRUE(Igus::Safety::FormatMoveJoint(1, {0, 0, 0, 0, 0, 0}, 10, -inf).empty());
  EXPECT_TRUE(Igus::Safety::MoveJointCommandText({0, 0, 0, 0, 0, nan}, 10, 20).empty());
}

// A robot-owned joint move uses the tighter STATUS-gap tier than the jog hold tier.
TEST(MoveStatusGap, AbortThresholdIsTheStaleLimitNotTheHoldTier)
{
  EXPECT_EQ(Igus::Safety::kMoveStatusAbortTimeout, Igus::Safety::kStatusStaleTimeout);
  EXPECT_EQ(Igus::Safety::kMoveStatusAbortTimeout.count(), 200);
  EXPECT_LT(Igus::Safety::kMoveStatusAbortTimeout, Igus::Safety::kStatusCoastTimeout);
}

TEST(MoveStop, ResendPeriodIsBoundedAndStillWindowCoversTheStopResends)
{
  EXPECT_EQ(Igus::Safety::kMoveStopResendPeriod.count(), 100);
  EXPECT_GT(Igus::Safety::kMoveStopStillWindow, Igus::Safety::kMoveStopResendPeriod);
}

// ---- ResendTimer: a new request ("send now") is never overwritten by a slower sender ----
TEST(ResendTimer, StartsDueAndDefersToTheDeadline)
{
  Igus::Safety::ResendTimer timer;
  EXPECT_TRUE(timer.Due(1000));
  const auto token = timer.Observe();
  EXPECT_TRUE(timer.Defer(token, 5000));
  EXPECT_FALSE(timer.Due(4999));
  EXPECT_TRUE(timer.Due(5000));
}

TEST(ResendTimer, ArmNowWhileASendIsInFlightSurvivesTheDeferral)
{
  Igus::Safety::ResendTimer timer;
  ASSERT_TRUE(timer.Defer(timer.Observe(), 9000000000LL));  // previous deadline far in the future
  EXPECT_FALSE(timer.Due(1000));

  // The sender observes, starts sending ...
  const auto token = timer.Observe();
  // ... meanwhile a new request arrives (false -> true transition) ...
  timer.ArmNow();
  EXPECT_TRUE(timer.Due(1000));
  // ... and the sender's deferral must NOT overwrite the "send now".
  EXPECT_FALSE(timer.Defer(token, 9000000000LL));
  EXPECT_TRUE(timer.Due(1000));
}

TEST(ResendTimer, ArmNowTwiceInARowIsStillDetectedByTheSender)
{
  Igus::Safety::ResendTimer timer;
  timer.ArmNow();
  const auto token = timer.Observe();  // "send now" #1 observed
  timer.ArmNow();                      // "send now" #2 while sending (same logical state, new value)
  EXPECT_FALSE(timer.Defer(token, 9000000000LL));
  EXPECT_TRUE(timer.Due(0));
}

TEST(DigitalIo, MaskDecodesToSortedIrcNumbers)
{
  using Igus::Safety::MaskToIrcNumbers;
  EXPECT_TRUE(MaskToIrcNumbers(0, 1).empty());
  EXPECT_EQ(MaskToIrcNumbers(0x5, 1), (std::vector<std::uint8_t>{1, 3}));
  EXPECT_EQ(MaskToIrcNumbers(1ull << 30, 1), (std::vector<std::uint8_t>{31}));
  EXPECT_EQ(MaskToIrcNumbers((1ull << 20) | (1ull << 31), 1), (std::vector<std::uint8_t>{21, 32}));
  EXPECT_EQ(MaskToIrcNumbers(1ull << 30, 0), (std::vector<std::uint8_t>{30}));
  EXPECT_EQ(MaskToIrcNumbers(1ull << 63, 1), (std::vector<std::uint8_t>{64}));
  EXPECT_EQ(MaskToIrcNumbers(~0ull, 1).size(), 64u);
}

TEST(DigitalIo, ParseNumberRanges)
{
  using Igus::Safety::ParseNumberRanges;
  std::vector<unsigned> n;
  ASSERT_TRUE(ParseNumberRanges("21-27,31-32", n));
  EXPECT_EQ(n, (std::vector<unsigned>{21, 22, 23, 24, 25, 26, 27, 31, 32}));
  ASSERT_TRUE(ParseNumberRanges(" 5 , 3-4,5 ", n));
  EXPECT_EQ(n, (std::vector<unsigned>{3, 4, 5}));
  const auto before = n;
  for (const char * bad : {"", "a", "3-", "-3", "5-3", "1,,2", "1 2", "300", "1-2-3", "1,"}) {
    EXPECT_FALSE(ParseNumberRanges(bad, n)) << bad;
    EXPECT_EQ(n, before);
  }
}

TEST(DigitalIo, ValidateDigitalOutputs)
{
  using Igus::Safety::ValidateDigitalOutputs;
  const std::vector<unsigned> allowed{21, 22, 23, 24, 25, 26, 27, 31, 32};
  std::vector<int> cri;
  std::string error;
  ASSERT_TRUE(ValidateDigitalOutputs({31, 22}, {true, false}, allowed, 1, cri, error)) << error;
  EXPECT_EQ(cri, (std::vector<int>{30, 21}));
  EXPECT_FALSE(ValidateDigitalOutputs({}, {}, allowed, 1, cri, error));
  EXPECT_TRUE(cri.empty());
  EXPECT_FALSE(ValidateDigitalOutputs({21, 22}, {true}, allowed, 1, cri, error));
  EXPECT_NE(error.find("same length"), std::string::npos);
  EXPECT_FALSE(ValidateDigitalOutputs({21, 28}, {true, true}, allowed, 1, cri, error));
  EXPECT_NE(error.find("28"), std::string::npos);
  EXPECT_TRUE(cri.empty());
  EXPECT_FALSE(ValidateDigitalOutputs({21, 22, 21}, {true, true, true}, allowed, 1, cri, error));
  EXPECT_NE(error.find("twice"), std::string::npos);
  // Offset 0 sends the iRC number unchanged.
  ASSERT_TRUE(ValidateDigitalOutputs({31}, {true}, allowed, 0, cri, error));
  EXPECT_EQ(cri, (std::vector<int>{31}));
}
