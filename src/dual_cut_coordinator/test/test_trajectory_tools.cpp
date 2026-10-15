#include <gtest/gtest.h>

#include <cmath>
#include <random>

#include <Eigen/Dense>

#include "dual_cut_coordinator/trajectory_tools.hpp"

#ifdef DUAL_CUT_HAVE_JTC
#include "joint_trajectory_controller/trajectory.hpp"
#include "rclcpp/time.hpp"
#endif

using dual_cut::JointTrajectory;
using dual_cut::TrajPoint;

namespace
{

TrajPoint pt(double t, std::vector<double> p, std::vector<double> v = {}, std::vector<double> a = {})
{
  TrajPoint q;
  q.time_s = t;
  q.positions = std::move(p);
  q.velocities = std::move(v);
  q.accelerations = std::move(a);
  return q;
}

JointTrajectory traj(std::vector<TrajPoint> pts, size_t n = 2)
{
  JointTrajectory t;
  for (size_t i = 1; i <= n; ++i) t.joint_names.push_back("joint" + std::to_string(i));
  t.points = std::move(pts);
  return t;
}

// Independent reference: solve the 6x6 boundary value problem for a quintic with Eigen.
double quinticRef(
  double p0, double v0, double a0, double p1, double v1, double a1, double T, double tau, int deriv)
{
  Eigen::Matrix<double, 6, 6> A;
  Eigen::Matrix<double, 6, 1> b;
  A.setZero();
  // rows: p(0), v(0), a(0), p(T), v(T), a(T)
  A(0, 0) = 1.0;
  A(1, 1) = 1.0;
  A(2, 2) = 2.0;
  for (int k = 0; k < 6; ++k)
  {
    A(3, k) = std::pow(T, k);
    A(4, k) = k >= 1 ? k * std::pow(T, k - 1) : 0.0;
    A(5, k) = k >= 2 ? k * (k - 1) * std::pow(T, k - 2) : 0.0;
  }
  b << p0, v0, a0, p1, v1, a1;
  const Eigen::Matrix<double, 6, 1> c = A.fullPivLu().solve(b);
  double r = 0.0;
  for (int k = deriv; k < 6; ++k)
  {
    double f = 1.0;
    for (int m = 0; m < deriv; ++m) f *= (k - m);
    r += c[k] * f * std::pow(tau, k - deriv);
  }
  return r;
}

}  // namespace

TEST(Sampling, QuinticEqualsAnalyticPolynomial)
{
  // Two quintic segments with non-trivial v and a, including negative accelerations.
  const JointTrajectory t = traj({
    pt(0.0, {0.10, -0.4}, {0.0, 0.0}, {0.0, 0.0}),
    pt(0.8, {0.55, 0.3}, {0.9, -0.2}, {0.3, 0.1}),
    pt(1.5, {0.40, 0.9}, {0.0, 0.0}, {0.0, 0.0}),
  });
  ASSERT_TRUE(dual_cut::validate(t));
  for (double ts = 0.0; ts <= 1.5 + 1e-12; ts += 0.0137)
  {
    TrajPoint out;
    ASSERT_TRUE(dual_cut::sampleAt(t, ts, out));
    const bool seg0 = ts < 0.8;
    const TrajPoint & a = seg0 ? t.points[0] : t.points[1];
    const TrajPoint & b = seg0 ? t.points[1] : t.points[2];
    const double T = b.time_s - a.time_s;
    const double tau = std::min(ts, 1.5) - a.time_s;
    for (size_t j = 0; j < 2; ++j)
    {
      EXPECT_NEAR(
        out.positions[j],
        quinticRef(a.positions[j], a.velocities[j], a.accelerations[j], b.positions[j],
                   b.velocities[j], b.accelerations[j], T, tau, 0),
        1e-9)
        << "t=" << ts << " j=" << j;
      EXPECT_NEAR(
        out.velocities[j],
        quinticRef(a.positions[j], a.velocities[j], a.accelerations[j], b.positions[j],
                   b.velocities[j], b.accelerations[j], T, tau, 1),
        1e-8);
      EXPECT_NEAR(
        out.accelerations[j],
        quinticRef(a.positions[j], a.velocities[j], a.accelerations[j], b.positions[j],
                   b.velocities[j], b.accelerations[j], T, tau, 2),
        1e-7);
    }
  }
}

TEST(Sampling, WaypointsAreHitExactlyAndContinuous)
{
  const JointTrajectory t = traj({
    pt(0.0, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}),
    pt(0.5, {0.2, -0.1}, {0.8, 0.1}, {0.2, 0.0}),
    pt(1.2, {0.0, 0.3}, {0.0, 0.0}, {0.0, 0.0}),
  });
  TrajPoint at;
  ASSERT_TRUE(dual_cut::sampleAt(t, 0.5, at));
  EXPECT_NEAR(at.positions[0], 0.2, 1e-12);
  EXPECT_NEAR(at.velocities[0], 0.8, 1e-12);
  TrajPoint before;
  ASSERT_TRUE(dual_cut::sampleAt(t, 0.5 - 1e-9, before));
  EXPECT_NEAR(before.positions[0], at.positions[0], 1e-7);
  EXPECT_NEAR(before.velocities[0], at.velocities[0], 1e-6);
}

TEST(Sampling, DegreeFollowsThePerSegmentRule)
{
  const TrajPoint pva = pt(0.0, {0.0}, {0.0}, {0.0});
  const TrajPoint pv = pt(1.0, {1.0}, {0.0});
  const TrajPoint p = pt(1.0, {1.0});
  EXPECT_EQ(dual_cut::segmentDegree(pva, pva), dual_cut::Degree::Quintic);
  EXPECT_EQ(dual_cut::segmentDegree(pva, pv), dual_cut::Degree::Cubic);
  EXPECT_EQ(dual_cut::segmentDegree(pva, p), dual_cut::Degree::Linear);
  EXPECT_EQ(dual_cut::segmentDegree(pv, pv), dual_cut::Degree::Cubic);
}

TEST(Sampling, CubicHermiteAndLinear)
{
  // Cubic: p0=0, v0=1, p1=2, v1=-1, T=2 -> Hermite
  const JointTrajectory c = traj({pt(0.0, {0.0}, {1.0}), pt(2.0, {2.0}, {-1.0})}, 1);
  TrajPoint o;
  ASSERT_TRUE(dual_cut::sampleAt(c, 0.7, o));
  const double T = 2.0, s = 0.7 / T;
  const double h00 = 2 * s * s * s - 3 * s * s + 1, h10 = s * s * s - 2 * s * s + s,
               h01 = -2 * s * s * s + 3 * s * s, h11 = s * s * s - s * s;
  EXPECT_NEAR(o.positions[0], h00 * 0.0 + h10 * T * 1.0 + h01 * 2.0 + h11 * T * -1.0, 1e-12);

  // Linear when velocities are missing at one end.
  const JointTrajectory l = traj({pt(0.0, {1.0}, {0.0}), pt(2.0, {3.0})}, 1);
  ASSERT_TRUE(dual_cut::sampleAt(l, 0.5, o));
  EXPECT_NEAR(o.positions[0], 1.5, 1e-12);
  EXPECT_NEAR(o.velocities[0], 1.0, 1e-12);
}

TEST(Sampling, BeforeFirstAndAfterLast)
{
  const JointTrajectory t = traj({pt(0.2, {0.1, 0.2}), pt(1.0, {0.5, 0.6})});
  TrajPoint o;
  ASSERT_TRUE(dual_cut::sampleAt(t, 0.0, o));
  EXPECT_DOUBLE_EQ(o.positions[0], 0.1);  // hold the first point (documented rule 5)
  ASSERT_TRUE(dual_cut::sampleAt(t, 9.0, o));
  EXPECT_DOUBLE_EQ(o.positions[1], 0.6);
  ASSERT_EQ(o.velocities.size(), 2u);  // zero filled
  EXPECT_DOUBLE_EQ(o.velocities[0], 0.0);
  ASSERT_EQ(o.accelerations.size(), 2u);
}

TEST(Sampling, ValidateFailsClosed)
{
  EXPECT_FALSE(dual_cut::validate(JointTrajectory{}));
  JointTrajectory t = traj({pt(0.0, {0.0, 0.0}), pt(0.0, {1.0, 1.0})});  // not strictly increasing
  EXPECT_FALSE(dual_cut::validate(t));
  t = traj({pt(0.0, {0.0, 0.0}), pt(1.0, {1.0})});  // size mismatch
  EXPECT_FALSE(dual_cut::validate(t));
  t = traj({pt(0.0, {0.0, 0.0}), pt(1.0, {1.0, NAN})});
  EXPECT_FALSE(dual_cut::validate(t));
  t = traj({pt(0.0, {0.0, 0.0}, {}, {0.0, 0.0}), pt(1.0, {1.0, 1.0}, {}, {0.0, 0.0})});
  EXPECT_FALSE(dual_cut::validate(t));  // accelerations without velocities
  TrajPoint o;
  EXPECT_FALSE(dual_cut::sampleAt(t, 0.5, o));
}

#ifdef DUAL_CUT_HAVE_JTC
// Direct comparison with the installed Humble joint_trajectory_controller::Trajectory::sample.
TEST(Sampling, MatchesTheRealJointTrajectoryControllerSampler)
{
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> U(-1.0, 1.0);
  for (int variant = 0; variant < 3; ++variant)  // quintic, cubic, linear
  {
    JointTrajectory t;
    t.joint_names = {"joint1", "joint2", "joint3"};
    double tt = 0.0;
    for (int i = 0; i < 6; ++i)
    {
      TrajPoint p;
      p.time_s = tt;
      tt += 0.2 + 0.3 * std::abs(U(rng));
      for (int j = 0; j < 3; ++j)
      {
        p.positions.push_back(U(rng));
        if (variant < 2) p.velocities.push_back(U(rng));
        if (variant < 1) p.accelerations.push_back(U(rng));
      }
      t.points.push_back(p);
    }
    auto msg = std::make_shared<trajectory_msgs::msg::JointTrajectory>(dual_cut::toMsg(t));
    // Times are exact ns after toMsg; use the rounded values on our side too.
    const JointTrajectory tr = dual_cut::fromMsg(*msg);
    const int64_t start_ns = 100000000000LL;
    msg->header.stamp = rclcpp::Time(start_ns, RCL_ROS_TIME);
    trajectory_msgs::msg::JointTrajectoryPoint current;
    current.positions = tr.points.front().positions;
    joint_trajectory_controller::Trajectory jtc(rclcpp::Time(start_ns - 1000000000LL, RCL_ROS_TIME), current, msg);

    joint_trajectory_controller::TrajectoryPointConstIter s, e;
    for (int64_t k = 0; k < static_cast<int64_t>(tr.duration() * 1e3) + 50; k += 13)
    {
      const double ts = std::min(1e-3 * static_cast<double>(k), tr.duration() + 0.01);
      trajectory_msgs::msg::JointTrajectoryPoint ref;
      ASSERT_TRUE(jtc.sample(
        rclcpp::Time(start_ns + static_cast<int64_t>(std::llround(ts * 1e9)), RCL_ROS_TIME),
        joint_trajectory_controller::interpolation_methods::InterpolationMethod::
          VARIABLE_DEGREE_SPLINE,
        ref, s, e));
      TrajPoint mine;
      ASSERT_TRUE(dual_cut::sampleAt(tr, ts, mine));
      for (size_t j = 0; j < 3; ++j)
      {
        EXPECT_NEAR(mine.positions[j], ref.positions[j], 1e-9) << "variant " << variant << " t=" << ts;
        if (!ref.velocities.empty())
        {
          EXPECT_NEAR(mine.velocities[j], ref.velocities[j], 1e-8) << "variant " << variant;
        }
        if (!ref.accelerations.empty())
        {
          EXPECT_NEAR(mine.accelerations[j], ref.accelerations[j], 1e-7) << "variant " << variant;
        }
      }
    }
  }
}
#endif

TEST(Sampling, SampleTrajectoryKeepsWaypointsAndEndsAtTheEnd)
{
  const JointTrajectory t = traj({
    pt(0.0, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}),
    pt(0.37, {0.1, 0.1}, {0.5, 0.2}, {0.0, 0.0}),
    pt(1.0, {0.3, -0.1}, {0.0, 0.0}, {0.0, 0.0}),
  });
  dual_cut::SampleOptions o;
  o.dt_s = 0.05;
  o.max_joint_step_rad = 0.0;  // no densify
  std::vector<TrajPoint> s;
  ASSERT_TRUE(dual_cut::sampleTrajectory(t, o, s));
  EXPECT_DOUBLE_EQ(s.front().time_s, 0.0);
  EXPECT_DOUBLE_EQ(s.back().time_s, 1.0);
  bool has_wp = false;
  for (const auto & p : s) has_wp = has_wp || std::abs(p.time_s - 0.37) < 1e-12;
  EXPECT_TRUE(has_wp);
  for (size_t i = 1; i < s.size(); ++i) EXPECT_GT(s[i].time_s, s[i - 1].time_s);
  for (const auto & p : s)
  {
    TrajPoint ref;
    ASSERT_TRUE(dual_cut::sampleAt(t, p.time_s, ref));
    EXPECT_NEAR(p.positions[0], ref.positions[0], 1e-12);
  }
}

TEST(Densify, NoJointStepsMoreThanTheLimit)
{
  const JointTrajectory t = traj({
    pt(0.0, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}),
    pt(0.5, {1.2, -0.7}, {0.0, 0.0}, {0.0, 0.0}),
  });
  dual_cut::SampleOptions o;
  o.dt_s = 0.05;
  o.max_joint_step_rad = 0.01;
  std::vector<TrajPoint> s;
  ASSERT_TRUE(dual_cut::sampleTrajectory(t, o, s));
  EXPECT_GT(s.size(), 120u);
  for (size_t i = 1; i < s.size(); ++i)
    for (size_t j = 0; j < 2; ++j)
      EXPECT_LE(std::abs(s[i].positions[j] - s[i - 1].positions[j]), 0.01 + 1e-12) << i;
}

TEST(Densify, CatchesASwingInsideASpanWithEqualEndpoints)
{
  // Cubic with p0 = p1 = 0 and v0 = 1, v1 = -1, T = 0.2: p(tau) = tau - 5 tau^2, peak 0.05 at 0.1.
  const JointTrajectory t = traj({pt(0.0, {0.0}, {1.0}), pt(0.2, {0.0}, {-1.0})}, 1);
  dual_cut::SampleOptions o;
  o.dt_s = 0.2;
  o.max_joint_step_rad = 0.01;
  std::vector<TrajPoint> s;
  ASSERT_TRUE(dual_cut::sampleTrajectory(t, o, s));
  double peak = 0.0;
  for (const auto & p : s) peak = std::max(peak, p.positions[0]);
  EXPECT_NEAR(peak, 0.05, 0.01);
  for (size_t i = 1; i < s.size(); ++i)
    EXPECT_LE(std::abs(s[i].positions[0] - s[i - 1].positions[0]), 0.01 + 1e-12);
}

TEST(Densify, FailsClosedOnSampleCap)
{
  const JointTrajectory t = traj({pt(0.0, {0.0}), pt(1.0, {5.0})}, 1);
  dual_cut::SampleOptions o;
  o.max_samples = 50;
  std::vector<TrajPoint> s;
  EXPECT_FALSE(dual_cut::sampleTrajectory(t, o, s));
  EXPECT_TRUE(s.empty());
}

TEST(Densify, LinearPathHelper)
{
  std::vector<std::vector<double>> out;
  ASSERT_TRUE(dual_cut::densifyLinear({{0.0, 0.0}, {0.1, 0.0}, {0.1, -0.034}}, 0.01, out));
  EXPECT_EQ(out.size(), 1u + 10u + 4u);
  EXPECT_EQ(out[10], (std::vector<double>{0.1, 0.0}));  // an input point is kept exactly
  EXPECT_EQ(out.back(), (std::vector<double>{0.1, -0.034}));
  for (size_t i = 1; i < out.size(); ++i)
    for (size_t j = 0; j < 2; ++j) EXPECT_LE(std::abs(out[i][j] - out[i - 1][j]), 0.01 + 1e-12);
  EXPECT_FALSE(dual_cut::densifyLinear({{0.0}, {100.0}}, 0.01, out, 100));
  EXPECT_FALSE(dual_cut::densifyLinear({{0.0}, {1.0}}, 0.0, out));
}

TEST(Concatenate, DropsDuplicateBoundaryAndShiftsTimes)
{
  const JointTrajectory a = traj({
    pt(0.0, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}), pt(1.0, {0.5, 0.5}, {0.0, 0.0}, {0.0, 0.0})});
  const JointTrajectory b = traj({
    pt(0.0, {0.5, 0.5}, {0.0, 0.0}, {0.0, 0.0}), pt(0.4, {0.7, 0.6}, {0.1, 0.0}, {0.0, 0.0}),
    pt(2.0, {0.9, 0.9}, {0.0, 0.0}, {0.0, 0.0})});
  const JointTrajectory c = traj({
    pt(0.0, {0.9, 0.9}, {0.0, 0.0}, {0.0, 0.0}), pt(0.5, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0})});
  JointTrajectory out;
  std::vector<size_t> first;
  ASSERT_TRUE(dual_cut::concatenate({a, b, c}, dual_cut::ConcatOptions{}, out, &first));
  ASSERT_EQ(out.points.size(), 2u + 2u + 1u);  // 7 - 2 duplicates
  EXPECT_DOUBLE_EQ(out.points[0].time_s, 0.0);
  EXPECT_DOUBLE_EQ(out.points[1].time_s, 1.0);
  EXPECT_DOUBLE_EQ(out.points[2].time_s, 1.4);
  EXPECT_DOUBLE_EQ(out.points[3].time_s, 3.0);
  EXPECT_DOUBLE_EQ(out.points[4].time_s, 3.5);
  EXPECT_DOUBLE_EQ(out.duration(), a.duration() + b.duration() + c.duration());
  ASSERT_EQ(first.size(), 4u);
  EXPECT_EQ(first[0], 0u);
  EXPECT_EQ(first[1], 1u);
  EXPECT_EQ(first[2], 3u);
  EXPECT_EQ(first[3], 4u);
  // A concatenated trajectory samples like the pieces.
  TrajPoint x, y;
  ASSERT_TRUE(dual_cut::sampleAt(out, 1.2, x));
  ASSERT_TRUE(dual_cut::sampleAt(b, 0.2, y));
  EXPECT_NEAR(x.positions[0], y.positions[0], 1e-12);
}

TEST(Concatenate, BoundaryMismatchAndNameMismatchFailClosed)
{
  const JointTrajectory a = traj({pt(0.0, {0.0, 0.0}), pt(1.0, {0.5, 0.5})});
  JointTrajectory b = traj({pt(0.0, {0.5, 0.5}), pt(1.0, {0.9, 0.9})});
  JointTrajectory out;
  EXPECT_TRUE(dual_cut::concatenate({a, b}, dual_cut::ConcatOptions{}, out));

  b.points[0].positions[1] = 0.5 + 2e-6;  // beyond the 1e-6 default
  EXPECT_FALSE(dual_cut::concatenate({a, b}, dual_cut::ConcatOptions{}, out));
  dual_cut::ConcatOptions loose;
  loose.position_tol = 1e-5;
  EXPECT_TRUE(dual_cut::concatenate({a, b}, loose, out));

  b.points[0].positions[1] = 0.5;
  b.joint_names[0] = "joint9";
  EXPECT_FALSE(dual_cut::concatenate({a, b}, dual_cut::ConcatOptions{}, out));

  // velocity continuity is checked when both sides have velocities
  const JointTrajectory av = traj({pt(0.0, {0.0, 0.0}, {0.0, 0.0}), pt(1.0, {0.5, 0.5}, {0.0, 0.0})});
  const JointTrajectory bv = traj({pt(0.0, {0.5, 0.5}, {0.3, 0.0}), pt(1.0, {0.9, 0.9}, {0.0, 0.0})});
  EXPECT_FALSE(dual_cut::concatenate({av, bv}, dual_cut::ConcatOptions{}, out));
  EXPECT_FALSE(dual_cut::concatenate({}, dual_cut::ConcatOptions{}, out));
}

TEST(SampledPath, ContactFlagMergesAtBoundaryAndTimesAreMonotonic)
{
  const JointTrajectory a = traj({pt(0.0, {0.0, 0.0}), pt(0.5, {0.1, 0.0})});
  const JointTrajectory b = traj({pt(0.0, {0.1, 0.0}), pt(0.5, {0.2, 0.0})});
  const JointTrajectory c = traj({pt(0.0, {0.2, 0.0}), pt(0.5, {0.3, 0.0})});
  dual_cut::SampleOptions o;
  o.dt_s = 0.1;
  o.max_joint_step_rad = 0.0;
  dual_cut::SampledPath p;
  ASSERT_TRUE(dual_cut::buildSampledPath(
    {{a, false}, {b, true}, {c, false}}, o, dual_cut::ConcatOptions{}, p));
  // 6 samples per segment (0, .1, .2, .3, .4, .5), 2 duplicates dropped
  ASSERT_EQ(p.samples.size(), 16u);
  for (size_t i = 1; i < p.samples.size(); ++i)
    EXPECT_GT(p.samples[i].time_s, p.samples[i - 1].time_s);
  EXPECT_DOUBLE_EQ(p.samples.back().time_s, 1.5);
  // boundary a|b (index 5) and b|c (index 10) both touch the contact segment b
  EXPECT_TRUE(p.samples[5].contact_allowed);
  EXPECT_TRUE(p.samples[10].contact_allowed);
  EXPECT_FALSE(p.samples[4].contact_allowed);
  EXPECT_TRUE(p.samples[6].contact_allowed);
  EXPECT_FALSE(p.samples[11].contact_allowed);
  EXPECT_FALSE(p.samples[0].contact_allowed);
  EXPECT_EQ(p.samples[6].segment_index, 1u);
  // boundary mismatch fails
  JointTrajectory bad = b;
  bad.points[0].positions[0] = 0.5;
  EXPECT_FALSE(dual_cut::buildSampledPath(
    {{a, false}, {bad, true}}, o, dual_cut::ConcatOptions{}, p));
}

TEST(Prefix, MapsJointNToArmPrefix)
{
  EXPECT_EQ(dual_cut::armPrefix("left"), "arm_left_");
  EXPECT_EQ(dual_cut::armPrefix("right"), "arm_right_");
  EXPECT_EQ(dual_cut::armPrefix("middle"), "");

  std::vector<std::string> n{"joint1", "joint2", "joint6"};
  ASSERT_TRUE(dual_cut::prefixJointNames("arm_right_", n));
  EXPECT_EQ(n, (std::vector<std::string>{"arm_right_joint1", "arm_right_joint2", "arm_right_joint6"}));

  std::string s;
  ASSERT_TRUE(dual_cut::stripJointPrefix("arm_right_", "arm_right_joint3", s));
  EXPECT_EQ(s, "joint3");
  EXPECT_FALSE(dual_cut::stripJointPrefix("arm_left_", "arm_right_joint3", s));
  EXPECT_FALSE(dual_cut::stripJointPrefix("arm_left_", "arm_left_elbow", s));
}

TEST(Prefix, FailsClosedAndIsAllOrNothing)
{
  std::string out;
  EXPECT_FALSE(dual_cut::prefixJointName("arm_left_", "arm_left_joint1", out));  // already prefixed
  EXPECT_FALSE(dual_cut::prefixJointName("arm_left_", "joint", out));
  EXPECT_FALSE(dual_cut::prefixJointName("arm_left_", "joint1x", out));
  EXPECT_FALSE(dual_cut::prefixJointName("arm_left_", "shoulder", out));
  EXPECT_FALSE(dual_cut::prefixJointName("", "joint1", out));

  std::vector<std::string> n{"joint1", "oops", "joint3"};
  EXPECT_FALSE(dual_cut::prefixJointNames("arm_left_", n));
  EXPECT_EQ(n, (std::vector<std::string>{"joint1", "oops", "joint3"}));  // unchanged

  JointTrajectory t = traj({pt(0.0, {0.0, 0.0}), pt(1.0, {1.0, 1.0})});
  ASSERT_TRUE(dual_cut::prefixTrajectory("arm_left_", t));
  EXPECT_EQ(t.joint_names[1], "arm_left_joint2");
}

TEST(Message, RoundTrip)
{
  const JointTrajectory t = traj({
    pt(0.0, {0.0, 0.1}, {0.0, 0.0}, {0.0, 0.0}), pt(1.234567891, {0.5, 0.6}, {0.1, 0.2}, {0.0, 0.0})});
  const auto msg = dual_cut::toMsg(t);
  EXPECT_EQ(msg.points[1].time_from_start.sec, 1);
  EXPECT_EQ(msg.points[1].time_from_start.nanosec, 234567891u);
  const JointTrajectory back = dual_cut::fromMsg(msg);
  EXPECT_NEAR(back.points[1].time_s, 1.234567891, 1e-12);
  EXPECT_EQ(back.points[1].positions, t.points[1].positions);
  EXPECT_EQ(back.joint_names, t.joint_names);
}
