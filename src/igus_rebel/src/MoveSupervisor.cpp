#include "igus_rebel/MoveSupervisor.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace Igus
{
namespace Move
{
namespace
{
constexpr double kDegToRad = 0.0174532925199432957692369076848861271344287188854172545609719144;

std::string Num(double value)
{
  char buffer[40];
  std::snprintf(buffer, sizeof(buffer), "%.4g", value);
  return buffer;
}
}  // namespace

const std::array<std::array<double, 2>, 6> & HardwareJointLimits()
{
  static const std::array<std::array<double, 2>, 6> limits = {{
    {{-3.106686, 3.106686}},
    {{-1.378810, 2.426008}},
    {{-1.378810, 2.426008}},
    {{-3.106686, 3.106686}},
    {{-1.640609, 1.640609}},
    {{-3.106686, 3.106686}}}};
  return limits;
}

double ModelDurationS(double maxDeltaRad, double velocityPercent, double accelerationPercent)
{
  const double distance = std::fabs(maxDeltaRad);
  const double cruise = kModelDegPerSecPerVelPercent * kDegToRad * std::max(velocityPercent, 1e-6);
  const double accel =
    kModelAccelRadS2AtFullAcc * std::max(accelerationPercent, 1e-6) / kHardMaxAccelerationPercent;
  const double rampDistance = cruise * cruise / accel;  // accelerate + decelerate
  double motion;
  if (distance >= rampDistance) {
    motion = distance / cruise + cruise / accel;
  } else {
    motion = 2.0 * std::sqrt(distance / accel);
  }
  return kModelStartLatencyS + motion;
}

double MaxAbsDifference(const Joints & a, const Joints & b)
{
  double result = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    result = std::max(result, std::fabs(a[i] - b[i]));
  }
  return result;
}

bool ValidateGoalRequest(
  const GoalRequest & request, const std::array<std::string, 6> & hardwareJointNames,
  const Caps & caps, Joints & start, Joints & goal, std::string & reason)
{
  if (request.jointNames.size() != 6) {
    reason = "joint_names must list exactly the 6 joints of this arm";
    return false;
  }
  for (std::size_t i = 0; i < 6; ++i) {
    if (request.jointNames[i] != hardwareJointNames[i]) {
      reason = "joint_names[" + std::to_string(i) + "] is '" + request.jointNames[i] +
        "' but this hardware's joint " + std::to_string(i + 1) + " is '" +
        hardwareJointNames[i] + "'";
      return false;
    }
  }
  if (request.startPositions.size() != 6 || request.goalPositions.size() != 6) {
    reason = "start_positions and goal_positions must have 6 values";
    return false;
  }
  for (std::size_t i = 0; i < 6; ++i) {
    if (!std::isfinite(request.startPositions[i]) || !std::isfinite(request.goalPositions[i])) {
      reason = "start_positions/goal_positions must be finite";
      return false;
    }
    start[i] = request.startPositions[i];
    goal[i] = request.goalPositions[i];
  }
  if (!std::isfinite(request.velocityPercent) || request.velocityPercent <= 0.0) {
    reason = "velocity_percent must be finite and > 0";
    return false;
  }
  if (request.velocityPercent > caps.maxVelocityPercent) {
    reason = "velocity_percent " + Num(request.velocityPercent) + " exceeds the driver cap " +
      Num(caps.maxVelocityPercent) + " (move_joint_max_velocity_percent); not clamped";
    return false;
  }
  if (!std::isfinite(request.accelerationPercent) || request.accelerationPercent <= 0.0) {
    reason = "acceleration_percent must be finite and > 0";
    return false;
  }
  if (request.accelerationPercent > caps.maxAccelerationPercent) {
    reason = "acceleration_percent " + Num(request.accelerationPercent) +
      " exceeds the driver cap " + Num(caps.maxAccelerationPercent) +
      " (move_joint_max_acceleration_percent); not clamped";
    return false;
  }
  const struct
  {
    const char * name;
    double value;
    double max;
  } tolerances[] = {
    {"start_tolerance", request.startTolerance, kMaxStartTolerance},
    {"path_tolerance", request.pathTolerance, kMaxPathTolerance},
    {"goal_tolerance", request.goalTolerance, kMaxGoalTolerance}};
  for (const auto & tolerance : tolerances) {
    if (!std::isfinite(tolerance.value) || tolerance.value <= 0.0 ||
      tolerance.value > tolerance.max)
    {
      reason = std::string(tolerance.name) + " must be in (0, " + Num(tolerance.max) + "] rad";
      return false;
    }
  }
  if (!std::isfinite(request.timeoutS) || request.timeoutS < 0.0 ||
    request.timeoutS > kMaxTimeoutS)
  {
    reason = "timeout_s must be 0 (driver model) or in (0, " + Num(kMaxTimeoutS) + "] s";
    return false;
  }
  for (std::size_t i = 0; i < 6; ++i) {
    const double lower = caps.jointLimits[i][0];
    const double upper = caps.jointLimits[i][1];
    if (goal[i] < lower || goal[i] > upper) {
      reason = "goal for joint " + std::to_string(i + 1) + " (" + Num(goal[i]) +
        " rad) is outside the joint limits [" + Num(lower) + ", " + Num(upper) + "]";
      return false;
    }
    if (start[i] < lower - request.startTolerance || start[i] > upper + request.startTolerance) {
      reason = "start for joint " + std::to_string(i + 1) + " (" + Num(start[i]) +
        " rad) is outside the joint limits [" + Num(lower) + ", " + Num(upper) + "]";
      return false;
    }
  }
  if (MaxAbsDifference(start, goal) < kMinMoveDistance) {
    reason = "start and goal are closer than " + Num(kMinMoveDistance) + " rad: nothing to move";
    return false;
  }
  return true;
}

const char * VerdictName(Verdict verdict)
{
  switch (verdict) {
    case Verdict::Ok:
      return "ok";
    case Verdict::PathDeviation:
      return "path deviation";
    case Verdict::ProgressOverrun:
      return "progress beyond the goal";
    case Verdict::Timeout:
      return "timeout";
  }
  return "unknown";
}

MoveSupervisor::MoveSupervisor(
  const Joints & start, const Joints & goal, double velocityPercent, double accelerationPercent,
  double pathTolerance, double goalTolerance, double timeoutOverrideS)
: start_(start), goal_(goal), pathTolerance_(pathTolerance), goalTolerance_(goalTolerance)
{
  for (std::size_t i = 0; i < 6; ++i) {
    delta_[i] = goal_[i] - start_[i];
    deltaNormSquared_ += delta_[i] * delta_[i];
  }
  modelDurationS_ = Igus::Move::ModelDurationS(
    MaxAbsDifference(start_, goal_), velocityPercent, accelerationPercent);
  timeoutS_ = timeoutOverrideS > 0.0 ? timeoutOverrideS :
    kTimeoutFactor * modelDurationS_ + kTimeoutMarginS;
}

Sample MoveSupervisor::Evaluate(const Joints & q) const
{
  Sample sample;
  if (deltaNormSquared_ <= 0.0) {
    sample.progress = 0.0;
    sample.deviation = MaxAbsDifference(q, start_);
    return sample;
  }
  double dot = 0.0;
  for (std::size_t i = 0; i < 6; ++i) {
    dot += (q[i] - start_[i]) * delta_[i];
  }
  sample.progress = dot / deltaNormSquared_;
  for (std::size_t i = 0; i < 6; ++i) {
    const double onLine = start_[i] + sample.progress * delta_[i];
    sample.deviation = std::max(sample.deviation, std::fabs(q[i] - onLine));
  }
  return sample;
}

Verdict MoveSupervisor::Check(const Joints & q, double elapsedS) const
{
  for (const double value : q) {
    if (!std::isfinite(value)) {
      return Verdict::PathDeviation;
    }
  }
  const Sample sample = Evaluate(q);
  if (sample.deviation > pathTolerance_) {
    return Verdict::PathDeviation;
  }
  // Overshoot of the goal or travel backwards past the start. The margin is at least
  // kProgressEpsilon of the move and never less than kProgressMinMarginRad of joint-space
  // distance, so STATUS noise on a very short move cannot trip it.
  const double norm = std::sqrt(deltaNormSquared_);
  const double margin = std::max(kProgressEpsilon, norm > 0.0 ? kProgressMinMarginRad / norm : 0.0);
  if (sample.progress > 1.0 + margin || sample.progress < -margin) {
    return Verdict::ProgressOverrun;
  }
  if (elapsedS > timeoutS_) {
    return Verdict::Timeout;
  }
  return Verdict::Ok;
}

bool MoveSupervisor::AtGoal(const Joints & q) const
{
  return MaxAbsDifference(q, goal_) <= goalTolerance_;
}
}  // namespace Move
}  // namespace Igus
