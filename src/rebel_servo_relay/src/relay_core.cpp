#include "rebel_servo_relay/relay_core.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace rebel_servo_relay
{
namespace
{
bool finitePositive(double x)
{
  return std::isfinite(x) && x > 0.0;
}

// Cubic Hermite on [0, T], exactly as the JTC samples a segment given positions and velocities.
double hermite(double p0, double v0, double p1, double v1, double T, double t)
{
  const double s = t / T, s2 = s * s, s3 = s2 * s;
  return (2 * s3 - 3 * s2 + 1) * p0 + (s3 - 2 * s2 + s) * T * v0 + (-2 * s3 + 3 * s2) * p1 + (s3 - s2) * T * v1;
}
}  // namespace

void RelayParams::resolve()
{
  if (!finitePositive(publish_period_s))
    throw std::invalid_argument("publish_period_s must be a positive number");
  if (!std::isfinite(horizon_s) || horizon_s <= 0.0)
    horizon_s = 3.0 * publish_period_s;
  if (!std::isfinite(reset_timeout_s) || reset_timeout_s <= 0.0)
    reset_timeout_s = 3.0 * publish_period_s;
  if (!std::isfinite(stop_time_s) || stop_time_s == 0.0)
    throw std::invalid_argument("stop_time_s must be a positive finite number (or negative for horizon_s)");
  if (stop_time_s < 0.0)
    stop_time_s = horizon_s;
  if (!std::isfinite(silence_hold_s) || silence_hold_s <= 0.0)
    silence_hold_s = 2.0 * publish_period_s;
  if (!finitePositive(max_lead_rad))
    throw std::invalid_argument("max_lead_rad must be a positive number");
  if (!std::isfinite(lead_time_s) || lead_time_s < 0.0)
    throw std::invalid_argument("lead_time_s must be >= 0");
  if (!std::isfinite(lead_floor_rad) || lead_floor_rad < 0.0)
    throw std::invalid_argument("lead_floor_rad must be >= 0");
  if (!std::isfinite(joint_limit_margin_rad) || joint_limit_margin_rad < 0.0)
    throw std::invalid_argument("joint_limit_margin_rad must be >= 0");
  if (!finitePositive(joint_state_timeout_s))
    throw std::invalid_argument("joint_state_timeout_s must be a positive number");
  if (!std::isfinite(zero_velocity_eps) || zero_velocity_eps < 0.0)
    throw std::invalid_argument("zero_velocity_eps must be >= 0");
}

RelayCore::RelayCore(RelayParams params) : params_(std::move(params))
{
  params_.resolve();
}

void RelayCore::reset()
{
  active_ = false;
  reference_.clear();
  names_.clear();
  traj_ = Trajectory{};
}

double RelayCore::evaluate(std::size_t j, double elapsed) const
{
  const double H = params_.horizon_s, Ts = params_.stop_time_s;
  if (!(elapsed > 0.0))
    return traj_.p0[j];
  if (elapsed < H)
    return hermite(traj_.p0[j], traj_.v0[j], traj_.p1[j], traj_.v1[j], H, elapsed);
  if (elapsed < H + Ts)
    return hermite(traj_.p1[j], traj_.v1[j], traj_.p2[j], 0.0, Ts, elapsed - H);
  return traj_.p2[j];
}

RelayOutput RelayCore::reject(const std::string& reason)
{
  reset();
  RelayOutput out;
  out.action = Action::kReject;
  out.reason = reason;
  return out;
}

RelayOutput RelayCore::holdNow(double now_s, const Measured& measured, const std::string& reason)
{
  if (!active_ || reference_.empty() || traj_.p0.size() != reference_.size())
    return reject(reason);
  const std::size_t n = reference_.size();
  std::vector<double> pos(n);
  bool fresh = !measured.unstamped && measured.position.size() == n && std::isfinite(measured.age_s) &&
               measured.age_s <= params_.joint_state_timeout_s;
  for (std::size_t i = 0; fresh && i < n; ++i)
    fresh = std::isfinite(measured.position[i]);
  const double elapsed = now_s - last_time_;
  for (std::size_t i = 0; i < n; ++i)
    pos[i] = fresh ? measured.position[i] : evaluate(i, elapsed);
  reset();
  RelayOutput out;
  out.action = Action::kHold;
  out.fault = true;
  out.reason = reason;
  out.start = pos;
  out.end = pos;
  out.mid = pos;
  out.velocity.assign(n, 0.0);
  out.velocity_mid.assign(n, 0.0);
  return out;
}

RelayOutput RelayCore::fault(double now_s, const std::vector<std::string>& names, const Measured& measured,
                             const std::string& reason)
{
  // The measurement in a message with a different joint list does not belong to the active reference's joints.
  const bool same_joints = names == names_;
  return holdNow(now_s, same_joints ? measured : Measured{}, reason);  // rejects when there is no reference
}

RelayOutput RelayCore::checkSilence(double now_s, const Measured& measured)
{
  if (!active_ || !(now_s - last_time_ > params_.silence_hold_s))
    return RelayOutput{};
  return holdNow(now_s, measured, "no valid Servo message for " + std::to_string(now_s - last_time_) + " s");
}

RelayOutput RelayCore::process(double now_s, const std::vector<std::string>& names, const std::vector<double>& velocity,
                               const Measured& measured, const std::vector<JointLimit>& limits)
{
  const std::size_t n = velocity.size();
  // Validate EVERYTHING before touching the reference: a rejected message must hold the reference that is active
  // (with its own joint names), not a fresh one built from a message that is not trustworthy.
  if (n == 0)
    return fault(now_s, names, measured, "Servo message has no velocities (publish_joint_velocities must be true)");
  if (names.size() != n)
    return fault(now_s, names, measured, "joint_names and velocities differ in size");
  if (!std::isfinite(now_s))
    return fault(now_s, names, measured, "non-finite time");
  for (double v : velocity)
    if (!std::isfinite(v))
      return fault(now_s, names, measured, "non-finite velocity in Servo message");
  if (limits.size() != n)
    return fault(now_s, names, measured, "joint limits missing or not matching the Servo message");
  for (const JointLimit& limit : limits)
  {
    if (!limit.has_limits)
      continue;
    if (!std::isfinite(limit.lower) || !std::isfinite(limit.upper) ||
        limit.lower + params_.joint_limit_margin_rad >= limit.upper - params_.joint_limit_margin_rad)
      return fault(now_s, names, measured, "invalid joint limits (empty range after the margin)");
  }
  if (measured.position.size() != n)
    return fault(now_s, names, measured, "no joint_states for the joints of the Servo message");
  if (!std::isfinite(measured.age_s) || measured.age_s > params_.joint_state_timeout_s)
    return fault(now_s, names, measured, "joint_states stale");
  for (double p : measured.position)
    if (!std::isfinite(p))
      return fault(now_s, names, measured, "non-finite measured position");
  if (measured.unstamped)
    return fault(now_s, names, measured, "joint_states unstamped (header.stamp 0)");

  // Valid message. A different joint list (order included) is a different reference: drop the old one and reseed.
  if (active_ && names != names_)
    reset();

  bool moving = false;
  for (double v : velocity)
    moving = moving || std::fabs(v) >= params_.zero_velocity_eps;

  if (!moving)
  {
    // Servo halted (collision stop, timeout, zero command): drop the reference and hold where the arm is now.
    reset();
    last_time_ = now_s;
    RelayOutput out;
    out.action = Action::kHold;
    out.start = measured.position;
    out.mid = measured.position;
    out.end = measured.position;
    out.velocity.assign(n, 0.0);
    out.velocity_mid.assign(n, 0.0);
    return out;
  }

  const double gap = now_s - last_time_;
  const bool reseed = !active_ || reference_.size() != n || traj_.p0.size() != n || !(gap >= 0.0) ||
                      gap > params_.reset_timeout_s;
  std::vector<double> r(n);
  for (std::size_t i = 0; i < n; ++i)
    r[i] = reseed ? measured.position[i] : evaluate(i, gap);  // what the JTC is executing now

  const double H = params_.horizon_s;
  const double Ts = params_.stop_time_s;
  const double margin = params_.joint_limit_margin_rad;
  RelayOutput out;
  out.action = Action::kForward;
  Trajectory t;
  t.p0.resize(n);
  t.p1.resize(n);
  t.p2.resize(n);
  t.v0.resize(n);
  t.v1.resize(n);
  for (std::size_t i = 0; i < n; ++i)
  {
    const double v = velocity[i];
    // Lead bound shrinks with the (Servo-scaled) velocity: near an obstacle no stale lead is kept.
    const double lead = std::min(params_.max_lead_rad, std::fabs(v) * params_.lead_time_s + params_.lead_floor_rad);
    double ri = std::clamp(r[i], measured.position[i] - lead, measured.position[i] + lead);
    double dist = INFINITY;  // distance to the limit bound in the direction of v
    if (limits[i].has_limits)
    {
      const double lo = limits[i].lower + margin, hi = limits[i].upper - margin;
      ri = std::clamp(ri, lo, hi);  // the limit wins over the lead bound
      dist = std::max(0.0, v > 0.0 ? hi - ri : ri - lo);
    }
    // Scale the velocity so that the whole profile (distance v' (H + T_stop/2)) ends at or before the bound.
    const double vmax = dist / (H + 0.5 * Ts);
    const double vs = std::copysign(std::min(std::fabs(v), vmax), v);
    t.p0[i] = ri;
    t.v0[i] = vs;
    t.p1[i] = ri + vs * H;
    t.v1[i] = vs;
    t.p2[i] = ri + vs * (H + 0.5 * Ts);
  }
  active_ = true;
  names_ = names;
  last_time_ = now_s;
  reference_ = t.p0;
  traj_ = t;
  out.start = t.p0;
  out.mid = t.p1;
  out.end = t.p2;
  out.velocity = t.v0;
  out.velocity_mid = t.v1;
  return out;
}

}  // namespace rebel_servo_relay
