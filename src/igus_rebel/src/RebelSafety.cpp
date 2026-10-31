#include "igus_rebel/RebelSafety.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <exception>
#include <sstream>
#include <string>

#include "igus_rebel/CriKeywords.hpp"

namespace Igus
{
namespace Safety
{
bool ParseJogVelocityScale(const std::string & text, double & scale)
{
  double parsed = 0.0;
  try {
    std::size_t used = 0;
    parsed = std::stod(text, &used);
    if (used != text.size()) {
      return false;
    }
  } catch (const std::exception &) {
    return false;
  }
  if (!std::isfinite(parsed) || parsed < kMinJogVelocityScale || parsed > kMaxJogVelocityScale) {
    return false;
  }
  scale = parsed;
  return true;
}

JogCommand JogPercentFromVelocity(double velocityRadPerSec, double scale)
{
  JogCommand jog;
  const double percent = scale * velocityRadPerSec / kDegToRad;
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

bool IsValidJogDeadband(double deadbandPercent)
{
  return std::isfinite(deadbandPercent) && deadbandPercent >= 0.0 &&
         deadbandPercent <= kMaxJtcJogDeadbandPercent;
}

std::array<float, 6> ApplyJogDeadband(
  const std::array<float, 6> & jogPercent, double deadbandPercent)
{
  std::array<float, 6> result = jogPercent;
  if (!std::isfinite(deadbandPercent) || deadbandPercent <= 0.0) {
    return result;
  }
  // Compared in float, like the jog values themselves: a value that was computed as exactly the
  // deadband (e.g. 0.7f vs 0.7) is on the boundary and kept.
  const float limit = static_cast<float>(deadbandPercent);
  for (float & value : result) {
    if (std::isfinite(value) && std::fabs(value) < limit) {
      value = 0.0f;
    }
  }
  return result;
}

bool ParseBoundedNumber(const std::string & text, double minimum, double maximum, double & value)
{
  double parsed = 0.0;
  try {
    std::size_t used = 0;
    parsed = std::stod(text, &used);
    if (used != text.size()) {
      return false;
    }
  } catch (const std::exception &) {
    return false;
  }
  if (!std::isfinite(parsed) || parsed < minimum || parsed > maximum) {
    return false;
  }
  value = parsed;
  return true;
}

bool ParseBoolText(const std::string & text, bool & value)
{
  if (text == "true" || text == "True" || text == "1") {
    value = true;
    return true;
  }
  if (text == "false" || text == "False" || text == "0") {
    value = false;
    return true;
  }
  return false;
}

bool ParseCartJogSigns(const std::string & text, std::array<double, 6> & signs)
{
  std::string normalized = text;
  std::replace(normalized.begin(), normalized.end(), ',', ' ');
  std::istringstream stream(normalized);
  std::array<double, 6> parsed{};
  for (double & value : parsed) {
    std::string token;
    if (!(stream >> token)) {
      return false;
    }
    if (token == "1" || token == "+1" || token == "1.0" || token == "+1.0") {
      value = 1.0;
    } else if (token == "-1" || token == "-1.0") {
      value = -1.0;
    } else {
      return false;
    }
  }
  std::string rest;
  if (stream >> rest) {
    return false;
  }
  signs = parsed;
  return true;
}

bool IsValidFrameName(const std::string & name)
{
  return !name.empty() && std::all_of(
    name.begin(), name.end(), [](unsigned char c) {
      return std::isalnum(c) != 0 || c == '_' || c == '/' || c == '.' || c == '-';
    });
}

CartJogResult CartJogFromTwist(const CartJogConfig & config, const std::array<double, 6> & twist)
{
  CartJogResult result;
  const double limit = std::min(config.maxPercent, static_cast<double>(kMaxJogPercent));
  for (std::size_t i = 0; i < 6; ++i) {
    const bool angular = i >= 3;
    const double input = twist[i];
    if (!std::isfinite(input)) {
      result.nonFinite = true;
      result.percent[i] = 0.0f;
      continue;
    }
    if (angular && !config.allowRotation) {
      if (input != 0.0) {
        result.rotationDropped = true;
      }
      result.percent[i] = 0.0f;
      continue;
    }
    const double scale = angular ? config.percentPerRadps : config.percentPerMps;
    double percent = config.sign[i] * scale * input;
    if (!std::isfinite(percent)) {
      result.nonFinite = true;
      percent = 0.0;
    }
    if (percent > limit || percent < -limit) {
      result.saturated = true;
      percent = std::clamp(percent, -limit, limit);
    }
    result.percent[i] = static_cast<float>(percent);
  }
  return result;
}

JointJogResult JointJogFromVelocities(
  double scale, double maxPercent, const std::array<double, 6> & velocities)
{
  JointJogResult result;
  const double limit = std::min(maxPercent, static_cast<double>(kMaxJogPercent));
  for (std::size_t i = 0; i < 6; ++i) {
    const JogCommand command = JogPercentFromVelocity(velocities[i], scale);
    if (command.nonFinite) {
      result.nonFinite = true;
      result.percent[i] = 0.0f;
      continue;
    }
    double percent = static_cast<double>(command.percent);
    if (percent > limit || percent < -limit) {
      result.saturated = true;
      percent = std::clamp(percent, -limit, limit);
    }
    result.percent[i] = static_cast<float>(percent);
  }
  return result;
}

bool MapJointJogMessage(
  const std::array<std::string, 6> & hardwareJoints, const std::vector<std::string> & names,
  const std::vector<double> & velocities, std::size_t displacementCount,
  std::array<double, 6> & out, std::string & error)
{
  out.fill(0.0);
  if (displacementCount != 0) {
    error = "displacements are not supported (velocities only)";
    return false;
  }
  if (names.size() != velocities.size()) {
    error = "joint_names and velocities differ in length";
    return false;
  }
  std::array<double, 6> mapped{};
  std::array<bool, 6> seen{};
  for (std::size_t k = 0; k < names.size(); ++k) {
    std::size_t index = hardwareJoints.size();
    for (std::size_t i = 0; i < hardwareJoints.size(); ++i) {
      if (hardwareJoints[i] == names[k]) {
        index = i;
        break;
      }
    }
    if (index == hardwareJoints.size()) {
      error = "unknown joint name '" + names[k] + "'";
      return false;
    }
    if (seen[index]) {
      error = "duplicate joint name '" + names[k] + "'";
      return false;
    }
    seen[index] = true;
    mapped[index] = velocities[k];
  }
  out = mapped;
  return true;
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

StatusHealth ClassifyStatusAge(
  bool haveStatus, std::chrono::nanoseconds age, std::chrono::nanoseconds staleTimeout,
  std::chrono::nanoseconds coastTimeout, std::chrono::nanoseconds latchTimeout)
{
  if (!haveStatus || age > latchTimeout) {
    return StatusHealth::Latch;
  }
  if (age > coastTimeout) {
    return StatusHealth::Hold;
  }
  return age > staleTimeout ? StatusHealth::Coast : StatusHealth::Fresh;
}

bool GapResumeAllowed(StatusHealth gapTier, bool motionDuringGap)
{
  switch (gapTier) {
    case StatusHealth::Fresh:
    case StatusHealth::Coast:
      return true;
    case StatusHealth::Hold:
      return !motionDuringGap;
    case StatusHealth::Latch:
      return false;
  }
  return false;
}

const char * StatusHealthName(StatusHealth tier)
{
  switch (tier) {
    case StatusHealth::Fresh: return "fresh";
    case StatusHealth::Coast: return "coast";
    case StatusHealth::Hold: return "hold";
    case StatusHealth::Latch: return "latch";
  }
  return "?";
}

bool IsCommandMoving(double maxAbsVelocityRadS)
{
  return !std::isfinite(maxAbsVelocityRadS) || maxAbsVelocityRadS > kHoldIdleVelocityRadS;
}

bool MotionDuringGap(
  std::int64_t lastMovingNs, std::int64_t lastStatusBeforeGapNs, std::chrono::nanoseconds lookback)
{
  return lastMovingNs != 0 && lastMovingNs >= lastStatusBeforeGapNs - lookback.count();
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

std::string MoveJointCommandText(
  const std::array<double, 6> & jointsDeg, double velocityPercent, double accelerationPercent)
{
  const auto finite = [](double v) {return std::isfinite(v);};
  if (!std::all_of(jointsDeg.begin(), jointsDeg.end(), finite) || !finite(velocityPercent) ||
    !finite(accelerationPercent))
  {
    return std::string();
  }
  const auto percent = [](double v) {
      char buffer[40];
      if (v == std::floor(v) && std::fabs(v) < 1e6) {
        std::snprintf(buffer, sizeof(buffer), "%d", static_cast<int>(v));
      } else {
        std::snprintf(buffer, sizeof(buffer), "%.3f", v);
      }
      return std::string(buffer);
    };
  char buffer[256];
  const int written = std::snprintf(
    buffer, sizeof(buffer), "%s %.3f %.3f %.3f %.3f %.3f %.3f 0 0 0 ",
    CriKeywords::COMMAND_MOVE_JOINT.c_str(), jointsDeg[0], jointsDeg[1], jointsDeg[2],
    jointsDeg[3], jointsDeg[4], jointsDeg[5]);
  if (written <= 0 || written >= static_cast<int>(sizeof(buffer))) {
    return std::string();
  }
  return std::string(buffer, static_cast<std::size_t>(written)) + percent(velocityPercent) + " " +
         percent(accelerationPercent);
}

std::int64_t JogCommandLifetimeNs(double timeoutS, double ageS)
{
  if (!std::isfinite(timeoutS) || timeoutS <= 0.0) {
    return 0;
  }
  const double age = std::isfinite(ageS) ? std::max(ageS, 0.0) : timeoutS;
  const double remaining = std::clamp(timeoutS - age, 0.0, timeoutS);
  return static_cast<std::int64_t>(remaining * 1e9);
}

bool JogStampIsNewer(std::int64_t stampNs, std::int64_t lastAcceptedStampNs)
{
  return lastAcceptedStampNs == 0 || stampNs > lastAcceptedStampNs;
}

std::string FormatMoveJoint(
  int counter, const std::array<double, 6> & jointsDeg, double velocityPercent,
  double accelerationPercent)
{
  const std::string text = MoveJointCommandText(jointsDeg, velocityPercent, accelerationPercent);
  return text.empty() ? std::string() : FormatCommand(counter, text);
}
}  // namespace Safety
}  // namespace Igus
