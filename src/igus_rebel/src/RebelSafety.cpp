#include "igus_rebel/RebelSafety.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

#include "igus_rebel/CriKeywords.hpp"

namespace Igus
{
namespace Safety
{
JogCommand JogPercentFromVelocity(double velocityRadPerSec)
{
  JogCommand jog;
  const double percent = JOINT_VELOCITY_SCALE * velocityRadPerSec / kDegToRad;
  if (!std::isfinite(percent)) {
    jog.nonFinite = true;
    jog.percent = 0.0f;
    return jog;
  }
  const double limit = static_cast<double>(kMaxJogPercent);
  if (percent > limit || percent < -limit) {
    jog.saturated = true;
  }
  jog.percent = static_cast<float>(std::clamp(percent, -limit, limit));
  return jog;
}

bool IsCommandStale(
  SteadyClock::time_point lastWrite, SteadyClock::time_point now,
  std::chrono::nanoseconds timeout)
{
  if (lastWrite == SteadyClock::time_point{}) {
    return true;
  }
  return now - lastWrite > timeout;
}

bool IsStatusStale(
  SteadyClock::time_point lastStatus, SteadyClock::time_point now,
  std::chrono::nanoseconds timeout)
{
  return IsCommandStale(lastStatus, now, timeout);
}

bool EstimateVelocity(
  const std::array<double, 6> & previousPosition, SteadyClock::time_point previousStamp,
  const std::array<double, 6> & position, SteadyClock::time_point stamp,
  std::chrono::nanoseconds minInterval, std::array<double, 6> & velocity)
{
  const std::chrono::nanoseconds interval = stamp - previousStamp;
  if (interval < minInterval || interval <= std::chrono::nanoseconds::zero()) {
    return false;
  }
  const double seconds = std::chrono::duration<double>(interval).count();
  for (std::size_t i = 0; i < velocity.size(); ++i) {
    velocity[i] = (position[i] - previousPosition[i]) / seconds;
  }
  return true;
}

bool JointPositionsRad(const CriMessages::Status & status, std::array<double, 6> & positions)
{
  std::array<double, 6> result{};
  for (std::size_t i = 0; i < result.size(); ++i) {
    const double deg = static_cast<double>(status.posJointCurrent.at(i));
    if (!std::isfinite(deg)) {
      return false;
    }
    result[i] = deg * kDegToRad;
  }
  positions = result;
  return true;
}

bool IsOverrideAsExpected(float overridePercent)
{
  return std::isfinite(overridePercent) &&
         std::fabs(overridePercent - kExpectedOverridePercent) <= kOverrideTolerancePercent;
}

std::string FormatAliveJog(int counter, const std::array<float, 6> & jogPercent)
{
  std::array<double, 6> jog{};
  for (std::size_t i = 0; i < jog.size(); ++i) {
    jog[i] = std::isfinite(jogPercent[i]) ? static_cast<double>(jogPercent[i]) : 0.0;
  }
  char buffer[256];
  const int written = std::snprintf(
    buffer, sizeof(buffer),
    "%s %d %s %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %s\n",
    CriKeywords::START.c_str(), counter, CriKeywords::TYPE_ALIVEJOG.c_str(),
    jog[0], jog[1], jog[2], jog[3], jog[4], jog[5], 0.0, 0.0, 0.0,
    CriKeywords::END.c_str());
  if (written <= 0) {
    return std::string();
  }
  return std::string(buffer, std::min<std::size_t>(written, sizeof(buffer) - 1));
}

std::string FormatCommand(int counter, const std::string & command)
{
  return CriKeywords::START + " " + std::to_string(counter) + " " + CriKeywords::TYPE_CMD + " " +
         command + " " + CriKeywords::END + "\n";
}
}  // namespace Safety
}  // namespace Igus
