#include <gtest/gtest.h>

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
