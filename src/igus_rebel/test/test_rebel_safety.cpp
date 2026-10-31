#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <string>

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
    // 2.0 * deg/s, the value the real-arm tests were run with.
    const double expected = std::clamp(2.0 * v / Igus::Safety::kDegToRad, -100.0, 100.0);
    EXPECT_NEAR(implicit.percent, expected, 1e-3) << v;
  }
}

TEST(JogScale, CustomScaleChangesJogProportionally)
{
  const double v = 0.3054;  // rad/s
  const auto base = Igus::Safety::JogPercentFromVelocity(v);
  const auto fitted = Igus::Safety::JogPercentFromVelocity(v, 1.652);
  EXPECT_NEAR(fitted.percent, base.percent * 1.652 / 2.0, 1e-3);
  // The +-100 % clamp still applies with a larger scale.
  const auto big = Igus::Safety::JogPercentFromVelocity(0.7, Igus::kMaxJogVelocityScale);
  EXPECT_EQ(big.percent, 100.0f);
  EXPECT_TRUE(big.saturated);
}

TEST(JogScale, ParseAcceptsOnlySaneNumbers)
{
  double scale = 2.0;
  EXPECT_TRUE(Igus::Safety::ParseJogVelocityScale("1.652", scale));
  EXPECT_DOUBLE_EQ(scale, 1.652);
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

// ---- Cartesian jog conversion (CRI route) ----
TEST(CartJog, DefaultsAreConservativeAndRotationIsOff)
{
  const Igus::Safety::CartJogConfig config;
  EXPECT_DOUBLE_EQ(config.percentPerMps, 200.0);
  EXPECT_DOUBLE_EQ(config.percentPerRadps, 100.0);
  EXPECT_DOUBLE_EQ(config.maxPercent, 10.0);
  EXPECT_FALSE(config.allowRotation);
  EXPECT_DOUBLE_EQ(config.timeoutS, 0.15);
  EXPECT_EQ(config.baseFrame, "base_link");
  for (const double sign : config.sign) {
    EXPECT_EQ(sign, 1.0);
  }
}

TEST(CartJog, ScalesLinearAxesAndClampsEachAxisToTheMaximum)
{
  Igus::Safety::CartJogConfig config;
  config.maxPercent = 8.0;
  const auto r = Igus::Safety::CartJogFromTwist(config, {0.02, -0.03, 0.5, 0, 0, 0});
  EXPECT_NEAR(r.percent[0], 4.0f, 1e-5);   // 0.02 m/s * 200
  EXPECT_NEAR(r.percent[1], -6.0f, 1e-5);
  EXPECT_NEAR(r.percent[2], 8.0f, 1e-5);   // 100 % -> clamped
  EXPECT_TRUE(r.saturated);
  EXPECT_FALSE(r.nonFinite);
  const auto negative = Igus::Safety::CartJogFromTwist(config, {-5.0, 0, 0, 0, 0, 0});
  EXPECT_NEAR(negative.percent[0], -8.0f, 1e-5);
}

TEST(CartJog, MaximumNeverExceedsTheHundredPercentJogLimit)
{
  Igus::Safety::CartJogConfig config;
  config.maxPercent = 500.0;  // not reachable through the parser, but the converter still clamps
  config.percentPerMps = 2000.0;
  const auto r = Igus::Safety::CartJogFromTwist(config, {1.0, 0, 0, 0, 0, 0});
  EXPECT_EQ(r.percent[0], 100.0f);
}

TEST(CartJog, NonFiniteAxesBecomeZeroAndAreReported)
{
  const Igus::Safety::CartJogConfig config;
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  const auto r = Igus::Safety::CartJogFromTwist(config, {nan, 0.01, inf, 0, 0, nan});
  EXPECT_TRUE(r.nonFinite);
  EXPECT_EQ(r.percent[0], 0.0f);
  EXPECT_NEAR(r.percent[1], 2.0f, 1e-5);
  EXPECT_EQ(r.percent[2], 0.0f);
}

TEST(CartJog, AngularInputIsZeroedUnlessRotationIsAllowed)
{
  Igus::Safety::CartJogConfig config;
  const auto dropped = Igus::Safety::CartJogFromTwist(config, {0.01, 0, 0, 0.05, -0.05, 0.05});
  EXPECT_TRUE(dropped.rotationDropped);
  EXPECT_NEAR(dropped.percent[0], 2.0f, 1e-5);
  for (std::size_t i = 3; i < 6; ++i) {
    EXPECT_EQ(dropped.percent[i], 0.0f);
  }
  // A zero angular input is not "dropped".
  EXPECT_FALSE(Igus::Safety::CartJogFromTwist(config, {0.01, 0, 0, 0, 0, 0}).rotationDropped);

  config.allowRotation = true;
  const auto allowed = Igus::Safety::CartJogFromTwist(config, {0, 0, 0, 0.05, -0.05, 0.5});
  EXPECT_FALSE(allowed.rotationDropped);
  EXPECT_NEAR(allowed.percent[3], 5.0f, 1e-5);   // 0.05 rad/s * 100
  EXPECT_NEAR(allowed.percent[4], -5.0f, 1e-5);
  EXPECT_NEAR(allowed.percent[5], 10.0f, 1e-5);  // 50 % -> clamped
}

TEST(CartJog, PerAxisSignIsAppliedBeforeTheClamp)
{
  Igus::Safety::CartJogConfig config;
  config.allowRotation = true;
  config.sign = {-1, 1, -1, 1, -1, 1};
  const auto r = Igus::Safety::CartJogFromTwist(config, {0.01, 0.01, 1.0, 0.05, 0.05, 0.05});
  EXPECT_NEAR(r.percent[0], -2.0f, 1e-5);
  EXPECT_NEAR(r.percent[1], 2.0f, 1e-5);
  EXPECT_NEAR(r.percent[2], -10.0f, 1e-5);
  EXPECT_NEAR(r.percent[3], 5.0f, 1e-5);
  EXPECT_NEAR(r.percent[4], -5.0f, 1e-5);
  EXPECT_NEAR(r.percent[5], 5.0f, 1e-5);
}

TEST(CartJog, ParametersParseStrictly)
{
  double value = 0.0;
  EXPECT_TRUE(Igus::Safety::ParseBoundedNumber("10.5", 0.01, 100.0, value));
  EXPECT_DOUBLE_EQ(value, 10.5);
  for (const char * bad : {"", "abc", "10x", "nan", "inf", "-1", "0", "101", "1e9"}) {
    value = 42.0;
    EXPECT_FALSE(Igus::Safety::ParseBoundedNumber(bad, 0.01, 100.0, value)) << bad;
    EXPECT_EQ(value, 42.0) << "a failed parse must not change the value";
  }
  bool flag = false;
  EXPECT_TRUE(Igus::Safety::ParseBoolText("true", flag));
  EXPECT_TRUE(flag);
  EXPECT_TRUE(Igus::Safety::ParseBoolText("False", flag));
  EXPECT_FALSE(flag);
  EXPECT_FALSE(Igus::Safety::ParseBoolText("yes", flag));
  EXPECT_FALSE(flag);

  std::array<double, 6> signs{};
  EXPECT_TRUE(Igus::Safety::ParseCartJogSigns("1,-1,1,1,-1,+1", signs));
  EXPECT_EQ(signs, (std::array<double, 6>{1, -1, 1, 1, -1, 1}));
  EXPECT_TRUE(Igus::Safety::ParseCartJogSigns("-1 -1 1 1 1 1", signs));
  EXPECT_EQ(signs[0], -1.0);
  const std::array<double, 6> before = signs;
  for (const char * bad : {"", "1,1,1", "1,1,1,1,1,1,1", "1,1,2,1,1,1", "1,1,0,1,1,1",
      "1,1,x,1,1,1", "1,1,1,1,1,1.5"})
  {
    EXPECT_FALSE(Igus::Safety::ParseCartJogSigns(bad, signs)) << bad;
    EXPECT_EQ(signs, before);
  }
  EXPECT_TRUE(Igus::Safety::IsValidFrameName("base_link"));
  EXPECT_TRUE(Igus::Safety::IsValidFrameName("arm_left/base_link"));
  EXPECT_FALSE(Igus::Safety::IsValidFrameName(""));
  EXPECT_FALSE(Igus::Safety::IsValidFrameName("bad frame"));
}

// ---- Direct joint jog conversion (CRI route) ----
TEST(JointJog, DefaultsAreConservative)
{
  const Igus::Safety::JointJogConfig config;
  EXPECT_DOUBLE_EQ(config.maxPercent, 10.0);
  EXPECT_DOUBLE_EQ(config.timeoutS, 0.15);
  EXPECT_DOUBLE_EQ(config.idleExitS, 10.0);
}

TEST(JointJog, ScalesWithTheJogVelocityScaleAndClampsEachJointToTheMaximum)
{
  const double perRad = 180.0 / M_PI;
  const auto r = Igus::Safety::JointJogFromVelocities(2.0, 10.0, {0.05, -0.05, 0.0, 0.5, -0.5, 0.01});
  EXPECT_NEAR(r.percent[0], 2.0 * 0.05 * perRad, 1e-4);
  EXPECT_NEAR(r.percent[1], -2.0 * 0.05 * perRad, 1e-4);
  EXPECT_EQ(r.percent[2], 0.0f);
  EXPECT_EQ(r.percent[3], 10.0f);
  EXPECT_EQ(r.percent[4], -10.0f);
  EXPECT_NEAR(r.percent[5], 2.0 * 0.01 * perRad, 1e-4);
  EXPECT_TRUE(r.saturated);
  EXPECT_FALSE(r.nonFinite);
  const auto slow = Igus::Safety::JointJogFromVelocities(1.0, 10.0, {0.05, 0, 0, 0, 0, 0});
  EXPECT_NEAR(slow.percent[0], 0.05 * perRad, 1e-4);
  EXPECT_FALSE(slow.saturated);
}

TEST(JointJog, MaximumNeverExceedsTheHundredPercentJogLimit)
{
  const auto r = Igus::Safety::JointJogFromVelocities(4.0, 500.0, {10.0, -10.0, 0, 0, 0, 0});
  EXPECT_EQ(r.percent[0], 100.0f);
  EXPECT_EQ(r.percent[1], -100.0f);
}

TEST(JointJog, NonFiniteVelocitiesBecomeZeroAndAreReported)
{
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  const auto r = Igus::Safety::JointJogFromVelocities(2.0, 10.0, {nan, 0.01, inf, -inf, 0, 0});
  EXPECT_TRUE(r.nonFinite);
  EXPECT_EQ(r.percent[0], 0.0f);
  EXPECT_GT(r.percent[1], 0.0f);
  EXPECT_EQ(r.percent[2], 0.0f);
  EXPECT_EQ(r.percent[3], 0.0f);
}

TEST(JointJogMessage, MapsSubsetsInAnyOrderAndLeavesTheRestZero)
{
  const std::array<std::string, 6> hardware = {"j1", "j2", "j3", "j4", "j5", "j6"};
  std::array<double, 6> out{};
  std::string error;
  ASSERT_TRUE(
    Igus::Safety::MapJointJogMessage(hardware, {"j6", "j2"}, {-0.03, 0.04}, 0, out, error))
    << error;
  EXPECT_EQ(out, (std::array<double, 6>{0, 0.04, 0, 0, 0, -0.03}));
  ASSERT_TRUE(Igus::Safety::MapJointJogMessage(hardware, {}, {}, 0, out, error));
  EXPECT_EQ(out, (std::array<double, 6>{}));
  ASSERT_TRUE(
    Igus::Safety::MapJointJogMessage(
      hardware, {"j6", "j5", "j4", "j3", "j2", "j1"}, {6, 5, 4, 3, 2, 1}, 0, out, error));
  EXPECT_EQ(out, (std::array<double, 6>{1, 2, 3, 4, 5, 6}));
}

TEST(JointJogMessage, RejectsTheWholeMessageAndZeroesTheOutput)
{
  const std::array<std::string, 6> hardware = {"j1", "j2", "j3", "j4", "j5", "j6"};
  std::array<double, 6> out{9, 9, 9, 9, 9, 9};
  std::string error;
  EXPECT_FALSE(Igus::Safety::MapJointJogMessage(hardware, {"j1", "x"}, {1, 1}, 0, out, error));
  EXPECT_NE(error.find("unknown"), std::string::npos) << error;
  EXPECT_EQ(out, (std::array<double, 6>{}));
  out.fill(9);
  EXPECT_FALSE(Igus::Safety::MapJointJogMessage(hardware, {"j1", "j1"}, {1, 1}, 0, out, error));
  EXPECT_NE(error.find("duplicate"), std::string::npos) << error;
  EXPECT_EQ(out, (std::array<double, 6>{}));
  out.fill(9);
  EXPECT_FALSE(Igus::Safety::MapJointJogMessage(hardware, {"j1", "j2"}, {1}, 0, out, error));
  EXPECT_NE(error.find("length"), std::string::npos) << error;
  EXPECT_FALSE(Igus::Safety::MapJointJogMessage(hardware, {"j1"}, {1, 2}, 0, out, error));
  EXPECT_FALSE(Igus::Safety::MapJointJogMessage(hardware, {"j1"}, {1}, 1, out, error));
  EXPECT_NE(error.find("displacements"), std::string::npos) << error;
  EXPECT_EQ(out, (std::array<double, 6>{}));
  // Names are exact: a prefixed name is not this hardware's joint.
  EXPECT_FALSE(Igus::Safety::MapJointJogMessage(hardware, {"arm_left_j1"}, {1}, 0, out, error));
  EXPECT_FALSE(Igus::Safety::MapJointJogMessage(hardware, {"J1"}, {1}, 0, out, error));
}

TEST(JogCommand, LifetimeCountsFromTheStampNotTheReceipt)
{
  using Igus::Safety::JogCommandLifetimeNs;
  EXPECT_EQ(JogCommandLifetimeNs(0.15, 0.0), 150000000);
  EXPECT_NEAR(static_cast<double>(JogCommandLifetimeNs(0.15, 0.12)), 30e6, 10.0);
  EXPECT_EQ(JogCommandLifetimeNs(0.15, 0.15), 0);
  EXPECT_EQ(JogCommandLifetimeNs(0.15, 0.5), 0);
  // A stamp in the future (negative age) never extends the lifetime beyond the timeout.
  EXPECT_EQ(JogCommandLifetimeNs(0.15, -0.3), 150000000);
  // Garbage fails closed.
  EXPECT_EQ(JogCommandLifetimeNs(0.15, std::nan("")), 0);
  EXPECT_EQ(JogCommandLifetimeNs(0.0, 0.0), 0);
  EXPECT_EQ(JogCommandLifetimeNs(std::nan(""), 0.0), 0);
}

TEST(JogCommand, StampMustBeStrictlyNewerThanTheLastAcceptedOne)
{
  using Igus::Safety::JogStampIsNewer;
  EXPECT_TRUE(JogStampIsNewer(100, 0));  // first command of a session
  EXPECT_TRUE(JogStampIsNewer(101, 100));
  EXPECT_FALSE(JogStampIsNewer(100, 100));  // duplicate
  EXPECT_FALSE(JogStampIsNewer(99, 100));   // out of order
}

TEST(JogDeadband, ZeroDeadbandChangesNothing)
{
  const std::array<float, 6> jog{0.0f, 0.001f, -0.3f, 0.8f, -100.0f, 100.0f};
  EXPECT_EQ(Igus::Safety::ApplyJogDeadband(jog, 0.0), jog);
  // An invalid deadband never zeroes anything either.
  EXPECT_EQ(Igus::Safety::ApplyJogDeadband(jog, -1.0), jog);
  EXPECT_EQ(Igus::Safety::ApplyJogDeadband(jog, std::nan("")), jog);
  EXPECT_EQ(
    Igus::Safety::ApplyJogDeadband(jog, std::numeric_limits<double>::infinity()), jog);
}

TEST(JogDeadband, ZeroesOnlyValuesStrictlyBelowTheDeadbandAndKeepsTheSign)
{
  const std::array<float, 6> jog{0.3f, -0.3f, 0.8f, -0.8f, 0.0f, 50.0f};
  const auto out = Igus::Safety::ApplyJogDeadband(jog, 0.5);
  EXPECT_EQ(out[0], 0.0f);
  EXPECT_EQ(out[1], 0.0f);
  EXPECT_EQ(out[2], 0.8f);
  EXPECT_EQ(out[3], -0.8f);  // sign preserved
  EXPECT_EQ(out[4], 0.0f);
  EXPECT_EQ(out[5], 50.0f);
}

TEST(JogDeadband, ValueExactlyOnTheBoundaryIsKept)
{
  const std::array<float, 6> jog{0.5f, -0.5f, 0.7f, -0.7f, 0.499f, -0.499f};
  const auto out = Igus::Safety::ApplyJogDeadband(jog, 0.5);
  EXPECT_EQ(out[0], 0.5f);
  EXPECT_EQ(out[1], -0.5f);
  EXPECT_EQ(out[4], 0.0f);
  EXPECT_EQ(out[5], 0.0f);
  // 0.7f is not exactly 0.7: the comparison is made in float so it still counts as the boundary.
  const auto at07 = Igus::Safety::ApplyJogDeadband(jog, 0.7);
  EXPECT_EQ(at07[2], 0.7f);
  EXPECT_EQ(at07[3], -0.7f);
  EXPECT_EQ(at07[0], 0.0f);
}

TEST(JogDeadband, NonFiniteInputIsPassedThroughLikeBefore)
{
  // Not changed here: FormatAliveJog already turns a non-finite value into 0.
  const float nan = std::nanf("");
  const float inf = std::numeric_limits<float>::infinity();
  const std::array<float, 6> jog{nan, inf, -inf, 0.1f, 2.0f, 0.0f};
  const auto out = Igus::Safety::ApplyJogDeadband(jog, 1.0);
  EXPECT_TRUE(std::isnan(out[0]));
  EXPECT_EQ(out[1], inf);
  EXPECT_EQ(out[2], -inf);
  EXPECT_EQ(out[3], 0.0f);
  EXPECT_EQ(out[4], 2.0f);
  EXPECT_NE(Igus::Safety::FormatAliveJog(1, out).find(" 0.00000000 "), std::string::npos);
}

TEST(JogDeadband, ValidRangeIsZeroToTwo)
{
  using Igus::Safety::IsValidJogDeadband;
  EXPECT_TRUE(IsValidJogDeadband(0.0));
  EXPECT_TRUE(IsValidJogDeadband(0.5));
  EXPECT_TRUE(IsValidJogDeadband(2.0));
  EXPECT_FALSE(IsValidJogDeadband(2.0001));
  EXPECT_FALSE(IsValidJogDeadband(-0.001));
  EXPECT_FALSE(IsValidJogDeadband(std::nan("")));
  EXPECT_FALSE(IsValidJogDeadband(std::numeric_limits<double>::infinity()));
}
