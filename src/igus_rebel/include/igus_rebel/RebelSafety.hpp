#ifndef IGUS_REBEL__REBEL_SAFETY_HPP_
#define IGUS_REBEL__REBEL_SAFETY_HPP_

#include <array>
#include <chrono>
#include <string>

#include "igus_rebel/CriMessages.hpp"

// Pure helpers used by the Rebel hardware interface. They have no I/O and no ROS
// dependency so that the safety-relevant arithmetic can be unit tested without a robot.
namespace Igus
{
// jog [%] = JOINT_VELOCITY_SCALE * velocity [deg/s]. Empirical scale of the original driver.
constexpr double JOINT_VELOCITY_SCALE = 2.0;

namespace Safety
{
using SteadyClock = std::chrono::steady_clock;

// pi / 180
constexpr double kDegToRad = 0.0174532925199432957692369076848861271344287188854172545609719144;

// CRI jog values are percentages; anything outside [-100, 100] is clamped.
constexpr float kMaxJogPercent = 100.0f;

// ALIVEJOG period.
constexpr std::chrono::milliseconds kAlivePeriod{10};

// The ALIVEJOG thread sends a zero jog if write() has not refreshed the command for this
// long (5 missed cycles of the 100 Hz controller manager).
constexpr std::chrono::milliseconds kCommandWatchdogTimeout{50};

// read() reports ERROR if no valid STATUS arrived for this long. The robot sends STATUS
// every 20-30 ms (measured on the real arm), so this is ~8 missed messages.
constexpr std::chrono::milliseconds kStatusStaleTimeout{200};

// on_init fails if no valid STATUS arrives within this time after start-up.
constexpr std::chrono::milliseconds kFirstStatusTimeout{5000};

// Statuses closer together than this (bursts after a delayed STATUS, two in one TCP
// segment) are not used as a velocity baseline; the velocity is computed over the longer
// interval instead. Below the nominal 20-30 ms STATUS period, so normally every STATUS
// updates the velocity; limits the error from arrival jitter to roughly 0.6x-1.5x.
constexpr std::chrono::milliseconds kMinVelocityInterval{15};

// Velocity override requested at start-up and the tolerance for the STATUS check.
constexpr float kExpectedOverridePercent = 100.0f;
constexpr float kOverrideTolerancePercent = 0.5f;

struct JogCommand
{
  float percent = 0.0f;
  bool nonFinite = false;  // the input (or the scaled value) was NaN/inf; percent is 0
  bool saturated = false;  // the scaled value exceeded +-kMaxJogPercent and was clamped
};

// Converts a joint velocity command [rad/s] to a CRI jog percentage:
// JOINT_VELOCITY_SCALE * velocity / degToRad, non-finite -> 0, clamped to [-100, 100].
JogCommand JogPercentFromVelocity(double velocityRadPerSec);

// True if the last command is older than timeout, or if no command was ever written
// (lastWrite == time_point{}).
bool IsCommandStale(
  SteadyClock::time_point lastWrite, SteadyClock::time_point now,
  std::chrono::nanoseconds timeout);

// True if no valid STATUS was ever received or the last one is older than timeout.
bool IsStatusStale(
  SteadyClock::time_point lastStatus, SteadyClock::time_point now,
  std::chrono::nanoseconds timeout);

// Velocity between two STATUS samples, divided by the actual time between them.
// Returns false (and leaves velocity untouched) if the interval is shorter than
// minInterval, so the caller keeps the previous estimate and the old baseline.
bool EstimateVelocity(
  const std::array<double, 6> & previousPosition, SteadyClock::time_point previousStamp,
  const std::array<double, 6> & position, SteadyClock::time_point stamp,
  std::chrono::nanoseconds minInterval, std::array<double, 6> & velocity);

// Joint 1-6 positions of a STATUS in radians. False if any of them is not finite.
bool JointPositionsRad(const CriMessages::Status & status, std::array<double, 6> & positions);

// True if the STATUS override is within kOverrideTolerancePercent of the expected value.
bool IsOverrideAsExpected(float overridePercent);

// "CRISTART <counter> ALIVEJOG j1 .. j6 0 0 0 CRIEND\n", 8 decimals, non-finite -> 0.
std::string FormatAliveJog(int counter, const std::array<float, 6> & jogPercent);

// "CRISTART <counter> CMD <command> CRIEND\n"
std::string FormatCommand(int counter, const std::string & command);
}  // namespace Safety
}  // namespace Igus

#endif  // IGUS_REBEL__REBEL_SAFETY_HPP_
