#include "igus_rebel/RebelSafety.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
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

bool StartsAllCommandInterfaces(
  const std::vector<std::string> & startInterfaces, const std::array<std::string, 6> & jointNames,
  const std::string & interfaceType)
{
  for (const std::string & joint : jointNames) {
    if (joint.empty()) {
      return false;
    }
    const std::string wanted = joint + "/" + interfaceType;
    bool found = false;
    for (const std::string & started : startInterfaces) {
      if (started == wanted) {
        found = true;
        break;
      }
    }
    if (!found) {
      return false;
    }
  }
  return true;
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

std::string FormatMoveJoint(
  int counter, const std::array<double, 6> & jointsDeg, double velocityPercent,
  double accelerationPercent)
{
  const std::string text = MoveJointCommandText(jointsDeg, velocityPercent, accelerationPercent);
  return text.empty() ? std::string() : FormatCommand(counter, text);
}

bool ParseNumberRanges(const std::string & text, std::vector<unsigned> & numbers)
{
  std::vector<unsigned> parsed;
  std::size_t pos = 0;
  const auto skipSpace = [&]() {
      while (pos < text.size() && (text[pos] == ' ' || text[pos] == '\t')) {
        ++pos;
      }
    };
  const auto readNumber = [&](unsigned & value) {
      skipSpace();
      if (pos >= text.size() || text[pos] < '0' || text[pos] > '9') {
        return false;
      }
      unsigned long v = 0;
      while (pos < text.size() && text[pos] >= '0' && text[pos] <= '9') {
        v = v * 10 + static_cast<unsigned long>(text[pos] - '0');
        if (v > 255) {
          return false;
        }
        ++pos;
      }
      value = static_cast<unsigned>(v);
      return true;
    };
  while (true) {
    unsigned first = 0;
    if (!readNumber(first)) {
      return false;
    }
    unsigned last = first;
    skipSpace();
    if (pos < text.size() && text[pos] == '-') {
      ++pos;
      if (!readNumber(last) || last < first) {
        return false;
      }
    }
    for (unsigned n = first; n <= last; ++n) {
      parsed.push_back(n);
    }
    skipSpace();
    if (pos >= text.size()) {
      break;
    }
    if (text[pos] != ',') {
      return false;
    }
    ++pos;
  }
  std::sort(parsed.begin(), parsed.end());
  parsed.erase(std::unique(parsed.begin(), parsed.end()), parsed.end());
  numbers = parsed;
  return true;
}

std::vector<std::uint8_t> MaskToIrcNumbers(std::uint64_t mask, unsigned ircOffset)
{
  std::vector<std::uint8_t> result;
  for (unsigned bit = 0; bit < 64; ++bit) {
    if ((mask >> bit) & 1u) {
      result.push_back(static_cast<std::uint8_t>(bit + ircOffset));
    }
  }
  return result;
}

bool ValidateDigitalOutputs(
  const std::vector<std::uint8_t> & ircOutputs, const std::vector<bool> & states,
  const std::vector<unsigned> & allowed, unsigned ircOffset, std::vector<int> & criNumbers,
  std::string & error)
{
  criNumbers.clear();
  if (ircOutputs.empty()) {
    error = "irc_outputs is empty";
    return false;
  }
  if (ircOutputs.size() != states.size()) {
    error = "irc_outputs and states must have the same length";
    return false;
  }
  std::vector<int> result;
  for (std::size_t i = 0; i < ircOutputs.size(); ++i) {
    const unsigned irc = ircOutputs[i];
    if (std::find(allowed.begin(), allowed.end(), irc) == allowed.end()) {
      error = "iRC output " + std::to_string(irc) + " is not in the allowed set";
      return false;
    }
    for (std::size_t j = 0; j < i; ++j) {
      if (ircOutputs[j] == irc) {
        error = "iRC output " + std::to_string(irc) + " is listed twice";
        return false;
      }
    }
    if (irc < ircOffset) {
      error = "iRC output " + std::to_string(irc) + " is below the iRC offset";
      return false;
    }
    result.push_back(static_cast<int>(irc - ircOffset));
  }
  criNumbers = result;
  error.clear();
  return true;
}
}  // namespace Safety
}  // namespace Igus
