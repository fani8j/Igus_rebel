#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "rebel_servo_relay/relay_core.hpp"

using namespace rebel_servo_relay;

namespace
{
constexpr double kPeriod = 0.034;

RelayParams params()
{
  RelayParams p;
  p.publish_period_s = kPeriod;
  return p;
}

// No lead bound in the way (tests of the integration itself).
RelayParams freeLead()
{
  RelayParams p = params();
  p.max_lead_rad = 10.0;
  p.lead_time_s = 1000.0;
  return p;
}

std::vector<JointLimit> limits(std::size_t n, double lo = -2.0, double hi = 2.0)
{
  JointLimit l;
  l.has_limits = true;
  l.lower = lo;
  l.upper = hi;
  return std::vector<JointLimit>(n, l);
}

Measured measured(std::vector<double> pos, double age = 0.001)
{
  Measured m;
  m.position = std::move(pos);
  m.age_s = age;
  return m;
}

std::vector<std::string> names(std::size_t n)
{
  std::vector<std::string> v;
  for (std::size_t i = 0; i < n; ++i)
    v.push_back("joint" + std::to_string(i + 1));
  return v;
}

RelayOutput run(RelayCore& core, double now, const std::vector<double>& vel, const Measured& m,
                const std::vector<JointLimit>& lim)
{
  return core.process(now, names(vel.size()), vel, m, lim);
}

// Cubic Hermite exactly as the JTC samples a segment (positions + velocities given).
double hermite(double t0, double p0, double v0, double t1, double p1, double v1, double t)
{
  const double T = t1 - t0, s = (t - t0) / T;
  const double s2 = s * s, s3 = s2 * s;
  return (2 * s3 - 3 * s2 + 1) * p0 + (s3 - 2 * s2 + s) * T * v0 + (-2 * s3 + 3 * s2) * p1 + (s3 - s2) * T * v1;
}
double hermiteVel(double t0, double p0, double v0, double t1, double p1, double v1, double t)
{
  const double T = t1 - t0, s = (t - t0) / T;
  return ((6 * s * s - 6 * s) * p0 + (3 * s * s - 4 * s + 1) * T * v0 + (-6 * s * s + 6 * s) * p1 +
          (3 * s * s - 2 * s) * T * v1) /
         T;
}
double hermiteAcc(double t0, double p0, double v0, double t1, double p1, double v1, double t)
{
  const double T = t1 - t0, s = (t - t0) / T;
  return ((12 * s - 6) * p0 + (6 * s - 4) * T * v0 + (-12 * s + 6) * p1 + (6 * s - 2) * T * v1) / (T * T);
}
double posAt(const RelayOutput& o, std::size_t i, const RelayParams& p, double t)
{
  const double t1 = p.horizon_s, t2 = p.horizon_s + p.stop_time_s;
  if (t < t1)
    return hermite(0, o.start[i], o.velocity[i], t1, o.mid[i], o.velocity_mid[i], t);
  if (t < t2)
    return hermite(t1, o.mid[i], o.velocity_mid[i], t2, o.end[i], 0.0, t);
  return o.end[i];
}

struct Sampled
{
  double min_p, max_p, max_abs_acc, max_abs_vel, final_p;
  bool monotone;
};
// Samples all (two) segments of joint i densely.
Sampled sample(const RelayOutput& o, std::size_t i, const RelayParams& p)
{
  const double t1 = p.horizon_s, t2 = p.horizon_s + p.stop_time_s;
  Sampled r{1e9, -1e9, 0.0, 0.0, o.end[i], true};
  double prev = o.start[i];
  const double dir = (o.end[i] >= o.start[i]) ? 1.0 : -1.0;
  for (int k = 0; k <= 4000; ++k)
  {
    const double t = t2 * k / 4000.0;
    double pos, vel, acc;
    if (t < t1)
    {
      pos = hermite(0, o.start[i], o.velocity[i], t1, o.mid[i], o.velocity_mid[i], t);
      vel = hermiteVel(0, o.start[i], o.velocity[i], t1, o.mid[i], o.velocity_mid[i], t);
      acc = hermiteAcc(0, o.start[i], o.velocity[i], t1, o.mid[i], o.velocity_mid[i], t);
    }
    else
    {
      pos = hermite(t1, o.mid[i], o.velocity_mid[i], t2, o.end[i], 0.0, t);
      vel = hermiteVel(t1, o.mid[i], o.velocity_mid[i], t2, o.end[i], 0.0, t);
      acc = hermiteAcc(t1, o.mid[i], o.velocity_mid[i], t2, o.end[i], 0.0, t);
    }
    r.min_p = std::min(r.min_p, pos);
    r.max_p = std::max(r.max_p, pos);
    r.max_abs_acc = std::max(r.max_abs_acc, std::fabs(acc));
    r.max_abs_vel = std::max(r.max_abs_vel, std::fabs(vel));
    if (dir * (pos - prev) < -1e-12)
      r.monotone = false;
    prev = pos;
  }
  return r;
}
}  // namespace

TEST(RelayParams, DerivedDefaults)
{
  RelayCore core(params());
  EXPECT_NEAR(core.params().horizon_s, 3 * kPeriod, 1e-12);
  EXPECT_NEAR(core.params().reset_timeout_s, 3 * kPeriod, 1e-12);
  EXPECT_NEAR(core.params().stop_time_s, core.params().horizon_s, 1e-12);
  EXPECT_NEAR(core.params().silence_hold_s, 2 * kPeriod, 1e-12);
  EXPECT_DOUBLE_EQ(core.params().lead_time_s, 0.2);
  EXPECT_DOUBLE_EQ(core.params().lead_floor_rad, 0.005);
}

TEST(RelayParams, InvalidThrows)
{
  RelayParams p = params();
  p.max_lead_rad = 0.0;
  EXPECT_THROW(RelayCore{p}, std::invalid_argument);
  p = params();
  p.publish_period_s = std::nan("");
  EXPECT_THROW(RelayCore{p}, std::invalid_argument);
  p = params();
  p.joint_state_timeout_s = -1.0;
  EXPECT_THROW(RelayCore{p}, std::invalid_argument);
  p = params();
  p.lead_time_s = -0.1;
  EXPECT_THROW(RelayCore{p}, std::invalid_argument);
  p = params();
  p.lead_floor_rad = std::nan("");
  EXPECT_THROW(RelayCore{p}, std::invalid_argument);
}

TEST(RelayParams, StopTimeValidation)
{
  RelayParams p = params();
  p.stop_time_s = 0.2;
  EXPECT_NEAR(RelayCore(p).params().stop_time_s, 0.2, 1e-12);
  p.stop_time_s = 0.0;
  EXPECT_THROW(RelayCore{p}, std::invalid_argument);
  p.stop_time_s = std::nan("");
  EXPECT_THROW(RelayCore{p}, std::invalid_argument);
  p.stop_time_s = std::numeric_limits<double>::infinity();
  EXPECT_THROW(RelayCore{p}, std::invalid_argument);
}

TEST(RelayCore, FirstMessageSeedsFromMeasuredAndPublishesThreePoints)
{
  RelayCore core(params());
  const auto out = run(core, 10.0, { 0.3, -0.2 }, measured({ 0.5, 1.0 }), limits(2));
  ASSERT_EQ(out.action, Action::kForward);
  EXPECT_DOUBLE_EQ(out.start[0], 0.5);
  EXPECT_DOUBLE_EQ(out.start[1], 1.0);
  const double h = core.params().horizon_s;
  const double ts = core.params().stop_time_s;
  EXPECT_NEAR(out.mid[0], 0.5 + 0.3 * h, 1e-12);
  EXPECT_NEAR(out.mid[1], 1.0 - 0.2 * h, 1e-12);
  EXPECT_NEAR(out.end[0], 0.5 + 0.3 * (h + ts / 2), 1e-12);
  EXPECT_NEAR(out.end[1], 1.0 - 0.2 * (h + ts / 2), 1e-12);
  EXPECT_DOUBLE_EQ(out.velocity[0], 0.3);
  EXPECT_DOUBLE_EQ(out.velocity[1], -0.2);
  EXPECT_DOUBLE_EQ(out.velocity_mid[0], 0.3);
  EXPECT_DOUBLE_EQ(out.velocity_mid[1], -0.2);
  EXPECT_TRUE(core.active());
}

TEST(RelayCore, ConstantVelocityIntegratesExactly)
{
  RelayCore c(freeLead());
  run(c, 0.0, { 0.3 }, measured({ 0.0 }), limits(1, -5, 5));
  RelayOutput out;
  for (int k = 1; k <= 10; ++k)
    out = run(c, k * kPeriod, { 0.3 }, measured({ 0.0 }), limits(1, -5, 5));
  ASSERT_EQ(out.action, Action::kForward);
  EXPECT_NEAR(out.start[0], 0.3 * 10 * kPeriod, 1e-9);
}

TEST(RelayCore, ReferenceFollowsLastTrajectoryEvenWhenMessagesAreLate)
{
  RelayParams p = freeLead();
  p.reset_timeout_s = 1.0;  // allow a long gap without reseed
  RelayCore c(p);
  const auto first = run(c, 0.0, { 1.0 }, measured({ 0.0 }), limits(1, -5, 5));
  const double H = c.params().horizon_s, Ts = c.params().stop_time_s;
  // 0.5 s later the previous trajectory has played out: the reference is its final point (it ended at rest).
  const auto out = run(c, 0.5, { 1.0 }, measured({ 0.0 }), limits(1, -5, 5));
  ASSERT_EQ(out.action, Action::kForward);
  EXPECT_NEAR(out.start[0], first.end[0], 1e-12);
  EXPECT_NEAR(out.start[0], 1.0 * (H + Ts / 2), 1e-12);
  // 0.150 s: inside the deceleration segment, the reference is the Hermite point there.
  RelayCore c2(p);
  const auto f2 = run(c2, 0.0, { 1.0 }, measured({ 0.0 }), limits(1, -5, 5));
  const auto o2 = run(c2, 0.150, { 1.0 }, measured({ 0.0 }), limits(1, -5, 5));
  EXPECT_NEAR(o2.start[0], posAt(f2, 0, c2.params(), 0.150), 1e-12);
}

TEST(RelayCore, ReversalHasNoReferenceJump)
{
  RelayCore c(freeLead());
  const auto a = run(c, 0.0, { 0.3 }, measured({ 0.0 }), limits(1, -5, 5));
  const auto b = run(c, kPeriod, { -0.3 }, measured({ 0.0 }), limits(1, -5, 5));
  ASSERT_EQ(b.action, Action::kForward);
  // The new segment starts exactly where the previous one is after 34 ms (no jump), then reverses.
  EXPECT_NEAR(b.start[0], posAt(a, 0, c.params(), kPeriod), 1e-12);
  EXPECT_NEAR(b.start[0], 0.3 * kPeriod, 1e-12);
  EXPECT_DOUBLE_EQ(b.velocity[0], -0.3);
  EXPECT_LT(b.end[0], b.start[0]);
}

TEST(RelayCore, NegativeDtReseeds)
{
  RelayCore c(freeLead());
  run(c, 5.0, { 1.0 }, measured({ 0.0 }), limits(1, -5, 5));
  const auto out = run(c, 4.0, { 1.0 }, measured({ 0.7 }), limits(1, -5, 5));
  ASSERT_EQ(out.action, Action::kForward);
  EXPECT_DOUBLE_EQ(out.start[0], 0.7);
}

TEST(RelayCore, LeadClampBoundsReferenceAroundMeasured)
{
  RelayCore core(params());  // lead min(0.05, 0.2 |v| + 0.005): 0.05 at 0.6 rad/s
  run(core, 0.0, { 0.6 }, measured({ 0.0 }), limits(1));
  RelayOutput out;
  for (int k = 1; k <= 50; ++k)  // arm does not move (blocked)
    out = run(core, k * kPeriod, { 0.6 }, measured({ 0.0 }), limits(1));
  ASSERT_EQ(out.action, Action::kForward);
  EXPECT_DOUBLE_EQ(out.start[0], 0.05);
  RelayCore c2(params());
  run(c2, 0.0, { -0.6 }, measured({ 0.0 }), limits(1));
  for (int k = 1; k <= 50; ++k)
    out = run(c2, k * kPeriod, { -0.6 }, measured({ 0.0 }), limits(1));
  EXPECT_DOUBLE_EQ(out.start[0], -0.05);
  EXPECT_DOUBLE_EQ(c2.reference()[0], -0.05);
}

TEST(RelayCore, LeadBoundShrinksWithVelocity)
{
  // Blocked arm: the retained lead is |v| * lead_time + floor (below max_lead), so a Servo slow-down near an obstacle
  // takes the stored lead away with it.
  for (double v : { 0.2, 0.1, 0.02 })
  {
    RelayCore core(params());
    RelayOutput out;
    for (int k = 0; k < 60; ++k)
      out = run(core, k * kPeriod, { v }, measured({ 0.0 }), limits(1));
    ASSERT_EQ(out.action, Action::kForward);
    EXPECT_NEAR(out.start[0], std::min(0.05, v * 0.2 + 0.005), 1e-9) << "v " << v;
  }
  // A big drop of the velocity pulls an already built-up lead back at once.
  RelayCore core(params());
  RelayOutput out;
  for (int k = 0; k < 60; ++k)
    out = run(core, k * kPeriod, { 0.3 }, measured({ 0.0 }), limits(1));
  EXPECT_NEAR(out.start[0], 0.05, 1e-9);
  out = run(core, 60 * kPeriod, { 0.001 }, measured({ 0.0 }), limits(1));
  EXPECT_NEAR(out.start[0], 0.001 * 0.2 + 0.005, 1e-9);
}

TEST(RelayCore, LimitScalesVelocityAndEndsAtOrBeforeTheBound)
{
  RelayCore core(params());
  const double margin = core.params().joint_limit_margin_rad;
  const double hi = 1.0 - margin;
  const double span = core.params().horizon_s + core.params().stop_time_s / 2;
  const auto out = run(core, 0.0, { 0.5 }, measured({ hi - 0.01 }), limits(1, -1.0, 1.0));
  ASSERT_EQ(out.action, Action::kForward);
  EXPECT_NEAR(out.velocity[0], 0.01 / span, 1e-12);        // v' = d / (H + T/2)
  EXPECT_DOUBLE_EQ(out.velocity_mid[0], out.velocity[0]);  // same profile, just slower
  EXPECT_NEAR(out.end[0], hi, 1e-12);                      // ends exactly at the bound
  // At the bound and pushing out: stands still; pointing inwards: unaffected.
  RelayCore c2(params());
  const auto at = run(c2, 0.0, { 0.6 }, measured({ hi }), limits(1, -1.0, 1.0));
  EXPECT_DOUBLE_EQ(at.velocity[0], 0.0);
  EXPECT_DOUBLE_EQ(at.end[0], hi);
  RelayCore c3(params());
  const auto in = run(c3, 0.0, { -0.3 }, measured({ hi }), limits(1, -1.0, 1.0));
  EXPECT_DOUBLE_EQ(in.velocity[0], -0.3);
}

TEST(RelayCore, MeasuredOutsideMarginIsPulledInside)
{
  RelayCore core(params());
  const double margin = core.params().joint_limit_margin_rad;
  const auto out = run(core, 0.0, { 0.1 }, measured({ 1.0 - margin / 2 }), limits(1, -1.0, 1.0));
  ASSERT_EQ(out.action, Action::kForward);
  EXPECT_NEAR(out.start[0], 1.0 - margin, 1e-12);
}

TEST(RelayCore, JointWithoutLimitsIsOnlyLeadClamped)
{
  RelayCore core(params());
  std::vector<JointLimit> lim(1);  // continuous joint
  const auto out = run(core, 0.0, { 0.3 }, measured({ 100.0 }), lim);
  ASSERT_EQ(out.action, Action::kForward);
  EXPECT_DOUBLE_EQ(out.start[0], 100.0);
  EXPECT_NEAR(out.mid[0], 100.0 + 0.3 * core.params().horizon_s, 1e-12);
  EXPECT_NEAR(out.end[0], 100.0 + 0.3 * (core.params().horizon_s + core.params().stop_time_s / 2), 1e-12);
}

TEST(RelayCore, ResetsOnGap)
{
  RelayCore core(params());
  run(core, 0.0, { 0.3 }, measured({ 0.0 }), limits(1));
  run(core, kPeriod, { 0.3 }, measured({ 0.0 }), limits(1));
  const auto out = run(core, kPeriod + 0.2, { 0.3 }, measured({ 0.01 }), limits(1));
  ASSERT_EQ(out.action, Action::kForward);
  EXPECT_DOUBLE_EQ(out.start[0], 0.01);
}

TEST(RelayCore, ShortGapKeepsReference)
{
  RelayCore core(freeLead());
  run(core, 0.0, { 0.3 }, measured({ 0.0 }), limits(1));
  const auto out = run(core, 2.5 * kPeriod, { 0.3 }, measured({ 0.0 }), limits(1));  // < 3 periods
  ASSERT_EQ(out.action, Action::kForward);
  EXPECT_NEAR(out.start[0], 0.3 * 2.5 * kPeriod, 1e-12);  // the previous trajectory at 85 ms, not a reseed
}

TEST(RelayCore, ZeroCommandHoldsAtMeasuredAndDropsReference)
{
  RelayCore core(params());
  run(core, 0.0, { 0.3 }, measured({ 0.0 }), limits(1));
  run(core, kPeriod, { 0.3 }, measured({ 0.0 }), limits(1));
  ASSERT_TRUE(core.active());
  const auto hold = run(core, 2 * kPeriod, { 0.0 }, measured({ 0.012 }), limits(1));
  ASSERT_EQ(hold.action, Action::kHold);
  EXPECT_FALSE(hold.fault);
  EXPECT_DOUBLE_EQ(hold.start[0], 0.012);
  EXPECT_DOUBLE_EQ(hold.velocity[0], 0.0);
  EXPECT_FALSE(core.active());
  const auto out = run(core, 3 * kPeriod, { 0.3 }, measured({ 0.012 }), limits(1));
  ASSERT_EQ(out.action, Action::kForward);
  EXPECT_DOUBLE_EQ(out.start[0], 0.012);
}

TEST(RelayCore, TinyVelocitiesCountAsHalted)
{
  RelayCore core(params());
  const auto out = run(core, 0.0, { 1e-6, -1e-6 }, measured({ 0.0, 0.0 }), limits(2));
  EXPECT_EQ(out.action, Action::kHold);
}

TEST(RelayCore, FaultWhileActiveHoldsAtMeasuredOrReference)
{
  RelayCore core(params());
  run(core, 0.0, { 0.3 }, measured({ 0.2 }), limits(1));
  ASSERT_TRUE(core.active());
  // Stale joint_states: hold at the reference evaluated from the last trajectory (0.2 + 0.3 * 0.034).
  auto out = run(core, kPeriod, { 0.3 }, measured({ 0.0 }, 0.5), limits(1));
  ASSERT_EQ(out.action, Action::kHold);
  EXPECT_TRUE(out.fault);
  EXPECT_NEAR(out.start[0], 0.2 + 0.3 * kPeriod, 1e-12);
  EXPECT_DOUBLE_EQ(out.velocity[0], 0.0);
  EXPECT_FALSE(core.active());
  // Not active any more: nothing to hold, plain reject.
  out = run(core, 2 * kPeriod, { 0.3 }, Measured{}, limits(1));
  EXPECT_EQ(out.action, Action::kReject);
  // A fault with fresh measurements while active holds at the measured position.
  RelayCore c2(params());
  run(c2, 0.0, { 0.3 }, measured({ 0.2 }), limits(1));
  out = c2.process(kPeriod, names(1), { std::nan("") }, measured({ 0.21 }), limits(1));
  ASSERT_EQ(out.action, Action::kHold);
  EXPECT_DOUBLE_EQ(out.start[0], 0.21);
}

TEST(RelayCore, BadMessagesAreRejected)
{
  RelayCore core(params());
  EXPECT_EQ(core.process(0.0, {}, {}, measured({}), {}).action, Action::kReject);
  EXPECT_EQ(run(core, 0.0, { std::nan("") }, measured({ 0.0 }), limits(1)).action, Action::kReject);
  EXPECT_EQ(run(core, 0.0, { std::numeric_limits<double>::infinity() }, measured({ 0.0 }), limits(1)).action,
            Action::kReject);
  EXPECT_EQ(run(core, 0.0, { 0.1, 0.1 }, measured({ 0.0 }), limits(2)).action, Action::kReject);
  EXPECT_EQ(run(core, 0.0, { 0.1, 0.1 }, measured({ 0.0, 0.0 }), limits(1)).action, Action::kReject);
  EXPECT_EQ(run(core, 0.0, { 0.1 }, measured({ std::nan("") }), limits(1)).action, Action::kReject);
  EXPECT_EQ(core.process(0.0, names(2), { 0.1 }, measured({ 0.0 }), limits(1)).action, Action::kReject);  // names/vel
  EXPECT_FALSE(core.active());
  EXPECT_EQ(run(core, 0.0, { 0.1 }, measured({ 0.0 }), limits(1, -0.01, 0.01)).action, Action::kReject);
}

TEST(RelayCore, JointNameChangeDropsReference)
{
  RelayCore c(freeLead());
  c.process(0.0, { "a", "b" }, { 0.3, 0.3 }, measured({ 0.0, 0.0 }), limits(2));
  // Same count, different order: the reference of "a" must not be used for "b".
  auto out = c.process(kPeriod, { "b", "a" }, { 0.3, 0.3 }, measured({ 0.5, 0.7 }), limits(2));
  ASSERT_EQ(out.action, Action::kForward);
  EXPECT_DOUBLE_EQ(out.start[0], 0.5);
  EXPECT_DOUBLE_EQ(out.start[1], 0.7);
  // Different name, same order.
  c.process(2 * kPeriod, { "b", "a" }, { 0.3, 0.3 }, measured({ 0.5, 0.7 }), limits(2));
  out = c.process(3 * kPeriod, { "b", "c" }, { 0.3, 0.3 }, measured({ 0.9, 0.1 }), limits(2));
  EXPECT_DOUBLE_EQ(out.start[0], 0.9);
  EXPECT_DOUBLE_EQ(out.start[1], 0.1);
  // Different count.
  out = c.process(4 * kPeriod, { "b" }, { 0.3 }, measured({ 0.4 }), limits(1));
  EXPECT_DOUBLE_EQ(out.start[0], 0.4);
}

TEST(RelayCore, SilenceWatchdogHoldsOnceAndReseeds)
{
  RelayCore core(params());  // silence_hold 2 x 34 ms = 68 ms
  EXPECT_EQ(core.checkSilence(0.0, measured({ 0.0 })).action, Action::kNone);  // not active
  run(core, 0.0, { 0.3 }, measured({ 0.0 }), limits(1));
  EXPECT_EQ(core.checkSilence(0.060, measured({ 0.02 })).action, Action::kNone);
  auto out = core.checkSilence(0.070, measured({ 0.021 }));
  ASSERT_EQ(out.action, Action::kHold);
  EXPECT_TRUE(out.fault);
  EXPECT_DOUBLE_EQ(out.start[0], 0.021);  // measured, fresh
  EXPECT_FALSE(core.active());
  EXPECT_EQ(core.checkSilence(0.200, measured({ 0.021 })).action, Action::kNone);  // only ONE hold
  // The next Servo message reseeds from the measured state.
  out = run(core, 0.3, { 0.3 }, measured({ 0.021 }), limits(1));
  EXPECT_DOUBLE_EQ(out.start[0], 0.021);
  // Stale measurement: hold at the reference evaluated from the last trajectory.
  RelayCore c2(params());
  run(c2, 0.0, { 0.3 }, measured({ 0.0 }), limits(1));
  out = c2.checkSilence(0.080, measured({ 0.0 }, 5.0));
  ASSERT_EQ(out.action, Action::kHold);
  EXPECT_NEAR(out.start[0], 0.3 * 0.080, 1e-12);
}

TEST(RelayCore, SteadyFollowingTracksCommandedSpeed)
{
  RelayParams p = freeLead();
  RelayCore core(p);
  double measured_pos = 0.0;
  const double v = 0.3;
  RelayOutput out = run(core, 0.0, { v }, measured({ measured_pos }), limits(1, -5.0, 5.0));
  double ref0 = out.start[0];
  double t = 0.0;
  for (int k = 1; k <= 200; ++k)
  {
    t = k * kPeriod;
    measured_pos = out.start[0] + v * 0.02;  // arm is ~20 ms behind the reference
    out = run(core, t, { v }, measured({ measured_pos }), limits(1, -5.0, 5.0));
    ASSERT_EQ(out.action, Action::kForward);
  }
  EXPECT_NEAR((out.start[0] - ref0) / t, v, 1e-9);
}

TEST(RelayCore, UnclampedPathIsConstantVelocityThenConstantDeceleration)
{
  RelayCore core(params());
  const double v = 0.3;
  const auto out = run(core, 0.0, { v }, measured({ 0.0 }), limits(1, -5, 5));
  ASSERT_EQ(out.action, Action::kForward);
  const auto s = sample(out, 0, core.params());
  EXPECT_NEAR(s.final_p, v * (core.params().horizon_s + core.params().stop_time_s / 2), 1e-12);
  EXPECT_NEAR(s.max_p, s.final_p, 1e-9);
  EXPECT_TRUE(s.monotone);
  // Peak deceleration = v / T_stop (2.94 rad/s^2 at 0.3 rad/s, 0.102 s) and the segment ends at rest.
  EXPECT_NEAR(s.max_abs_acc, v / core.params().stop_time_s, 1e-6);
  EXPECT_LT(s.max_abs_acc, 5.24);
  EXPECT_NEAR(s.max_abs_vel, v, 1e-9);
}

TEST(RelayCore, SampledPathNeverPassesLimitMinusMarginOrSpeedsUp)
{
  const double margin = RelayParams{}.joint_limit_margin_rad;
  const double lo_lim = -1.0, hi_lim = 1.0;
  const double lo = lo_lim + margin, hi = hi_lim - margin;
  int cases = 0;
  for (double dist : { 0.0, 0.001, 0.005, 0.02, 0.0306, 0.0459, 0.05, 0.12 })
    for (double v : { 0.02, 0.1, 0.3, 0.6, -0.02, -0.1, -0.3, -0.6 })
      for (int upper = 0; upper < 2; ++upper)
      {
        RelayParams p = params();
        p.max_lead_rad = 1.0;  // keep the lead clamp out of the way: test the limit handling
        p.lead_time_s = 1000.0;
        RelayCore core(p);
        const double m = upper ? hi - dist : lo + dist;
        const auto out = run(core, 0.0, { v }, measured({ m }), limits(1, lo_lim, hi_lim));
        ASSERT_EQ(out.action, Action::kForward);
        const auto s = sample(out, 0, core.params());
        const std::string ctx = "dist " + std::to_string(dist) + " v " + std::to_string(v) + " upper " + std::to_string(upper);
        EXPECT_LE(s.max_p, hi + 1e-9) << ctx;
        EXPECT_GE(s.min_p, lo - 1e-9) << ctx;
        EXPECT_TRUE(s.monotone) << ctx;
        // Never faster than the commanded velocity, and never decelerating harder than the profile's own bound.
        EXPECT_LE(s.max_abs_vel, std::fabs(v) + 1e-9) << ctx;
        EXPECT_LE(s.max_abs_acc, std::fabs(v) / core.params().stop_time_s + 1e-6) << ctx;
        const double Ts = core.params().stop_time_s, H = core.params().horizon_s;
        EXPECT_NEAR(hermiteVel(H, out.mid[0], out.velocity_mid[0], H + Ts, out.end[0], 0.0, H + Ts), 0.0, 1e-9) << ctx;
        // Moving away from the near bound is not touched.
        ++cases;
      }
  EXPECT_EQ(cases, 128);
}

TEST(RelayCore, ReversalAfterLimitScalingStaysInsideBounds)
{
  // Approach the upper bound, then reverse: the second segment starts where the first one is (inside the bounds).
  RelayParams p = params();
  p.max_lead_rad = 1.0;
  p.lead_time_s = 1000.0;
  RelayCore core(p);
  const double hi = 1.0 - p.joint_limit_margin_rad;
  const auto a = run(core, 0.0, { 0.3 }, measured({ hi - 0.03 }), limits(1, -1.0, 1.0));
  const auto b = run(core, kPeriod, { -0.3 }, measured({ hi - 0.03 }), limits(1, -1.0, 1.0));
  EXPECT_NEAR(b.start[0], posAt(a, 0, core.params(), kPeriod), 1e-12);
  EXPECT_LE(b.start[0], hi);
  const auto s = sample(b, 0, core.params());
  EXPECT_LE(s.max_p, hi + 1e-9);
}

TEST(RelayCore, RejectedMessageWithChangedNamesHoldsTheOldReference)
{
  RelayCore c(params());
  c.process(0.0, { "a", "b" }, { 0.3, -0.3 }, measured({ 0.2, 0.4 }), limits(2));
  ASSERT_TRUE(c.active());
  const std::vector<std::string> old_names = c.names();
  ASSERT_EQ(old_names, (std::vector<std::string>{ "a", "b" }));
  // Changed names AND an invalid message (NaN velocity): the hold belongs to the old joints, it must not use the
  // new message's measurement (9.0 / 9.0) and must not reseed from it.
  auto out = c.process(kPeriod, { "c", "d" }, { std::nan(""), 0.1 }, measured({ 9.0, 9.0 }), limits(2));
  ASSERT_EQ(out.action, Action::kHold);
  EXPECT_TRUE(out.fault);
  ASSERT_EQ(out.start.size(), 2u);
  EXPECT_NEAR(out.start[0], 0.2 + 0.3 * kPeriod, 1e-12);  // old reference evaluated from the last trajectory
  EXPECT_NEAR(out.start[1], 0.4 - 0.3 * kPeriod, 1e-12);
  EXPECT_FALSE(c.active());
  // Same with stale joint_states and changed names.
  RelayCore c2(params());
  c2.process(0.0, { "a" }, { 0.3 }, measured({ 0.2 }), limits(1));
  out = c2.process(kPeriod, { "z" }, { 0.3 }, measured({ 5.0 }, 3.0), limits(1));
  ASSERT_EQ(out.action, Action::kHold);
  EXPECT_NEAR(out.start[0], 0.2 + 0.3 * kPeriod, 1e-12);
  // A fault with the SAME names and fresh measurement still holds at the measurement.
  RelayCore c3(params());
  c3.process(0.0, { "a" }, { 0.3 }, measured({ 0.2 }), limits(1));
  out = c3.process(kPeriod, { "a" }, { std::nan("") }, measured({ 0.25 }), limits(1));
  ASSERT_EQ(out.action, Action::kHold);
  EXPECT_DOUBLE_EQ(out.start[0], 0.25);
}

TEST(RelayCore, ValidMessageWithChangedNamesReseeds)
{
  RelayCore c(freeLead());
  c.process(0.0, { "a", "b" }, { 0.3, 0.3 }, measured({ 0.0, 0.0 }), limits(2));
  const auto out = c.process(kPeriod, { "c", "d" }, { 0.3, 0.3 }, measured({ 0.7, 0.8 }), limits(2));
  ASSERT_EQ(out.action, Action::kForward);
  EXPECT_DOUBLE_EQ(out.start[0], 0.7);
  EXPECT_DOUBLE_EQ(out.start[1], 0.8);
  EXPECT_EQ(c.names(), (std::vector<std::string>{ "c", "d" }));
}

TEST(RelayCore, UnstampedJointStatesAreRejected)
{
  RelayCore c(params());
  Measured m = measured({ 0.1 });
  m.unstamped = true;
  // Nothing active: plain reject, nothing is forwarded.
  EXPECT_EQ(run(c, 0.0, { 0.3 }, m, limits(1)).action, Action::kReject);
  EXPECT_FALSE(c.active());
  // Active, then an unstamped sample: hold at the reference, reset.
  run(c, 0.1, { 0.3 }, measured({ 0.1 }), limits(1));
  ASSERT_TRUE(c.active());
  const auto out = run(c, 0.1 + kPeriod, { 0.3 }, m, limits(1));
  ASSERT_EQ(out.action, Action::kHold);
  EXPECT_TRUE(out.fault);
  EXPECT_NEAR(out.start[0], 0.1 + 0.3 * kPeriod, 1e-12);
  // Also a zero command is not held at an unstamped measurement.
  RelayCore c2(params());
  EXPECT_EQ(run(c2, 0.0, { 0.0 }, m, limits(1)).action, Action::kReject);
}
