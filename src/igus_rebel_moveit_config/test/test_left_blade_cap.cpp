// Tests for the pure true blade-angle gate of the varying left cut (left_blade_cap.hpp).
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

#include "igus_rebel_moveit_config/left_blade_cap.hpp"

namespace
{
constexpr double kDeg = 3.14159265358979323846 / 180.0;
using left_blade_cap::Profile;

// Tool frame as in the left cut: X = blade-plane normal, Y = blade length (along -cut direction),
// Z = edge. normal = +Z world, along = +X world (travel direction).
const Eigen::Vector3d kNormal(0, 0, 1);
const Eigen::Vector3d kAlong(1, 0, 0);

// Tool pose at world x = s whose blade plane is theta_deg from the top plane: tool Y stays on the
// cut line, tool X is the surface normal rotated by theta_deg about the cut line (theta_p = theta).
Eigen::Isometry3d poseAtTheta(double s, double theta_deg)
{
  Eigen::Matrix3d r;
  const Eigen::Vector3d y = -kAlong;
  const Eigen::Vector3d x0 = kNormal;
  const Eigen::AngleAxisd tilt(theta_deg * kDeg, kAlong);
  const Eigen::Vector3d x = tilt * x0;
  const Eigen::Vector3d z = x.cross(y);
  r.col(0) = x;
  r.col(1) = y;
  r.col(2) = z;
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  pose.linear() = r;
  pose.translation() = Eigen::Vector3d(s, 0.0, 0.0);
  return pose;
}

Profile profile(double band, double end, double length = 0.300, double band_exit = 0.200)
{
  Profile p;
  p.band_cap_deg = band;
  p.end_cap_deg = end;
  p.span_length_m = length;
  p.band_exit_m = band_exit;
  return p;
}
}  // namespace

TEST(LeftBladeCap, ThetaPAndDeviationDefinitions)
{
  EXPECT_NEAR(left_blade_cap::thetaPDeg(poseAtTheta(0, 0).linear(), kNormal), 0.0, 1e-9);
  EXPECT_NEAR(left_blade_cap::thetaPDeg(poseAtTheta(0, 12.5).linear(), kNormal), 12.5, 1e-9);
  EXPECT_NEAR(left_blade_cap::thetaPDeg(poseAtTheta(0, 90).linear(), kNormal), 90.0, 1e-9);
  // |X . n|: a blade tilted the other way has the same angle
  EXPECT_NEAR(left_blade_cap::thetaPDeg(poseAtTheta(0, -20).linear(), kNormal), 20.0, 1e-9);
  EXPECT_NEAR(left_blade_cap::bladeDeviationDeg(poseAtTheta(0, 30).linear(), kAlong), 0.0, 1e-6);
}

TEST(LeftBladeCap, DefaultProfileIsFlatAtTheBandCap)
{
  const Profile p = profile(15.0, 15.0);
  EXPECT_FALSE(p.graded());
  for (const double s : {-0.01, 0.0, 0.1, 0.2, 0.25, 0.3, 0.4}) EXPECT_DOUBLE_EQ(p.capAt(s), 15.0);
}

TEST(LeftBladeCap, RampStartsOnlyAfterTheBandAndEndsAtTheEndCap)
{
  const Profile p = profile(15.0, 30.0);
  EXPECT_TRUE(p.graded());
  EXPECT_DOUBLE_EQ(p.capAt(0.0), 15.0);    // start of the stroke stays at the band cap
  EXPECT_DOUBLE_EQ(p.capAt(0.1), 15.0);
  EXPECT_DOUBLE_EQ(p.capAt(0.2), 15.0);    // band exit
  EXPECT_NEAR(p.capAt(0.25), 22.5, 1e-12);  // half way along the ramp
  EXPECT_DOUBLE_EQ(p.capAt(0.3), 30.0);
  EXPECT_DOUBLE_EQ(p.capAt(0.35), 30.0);   // beyond the span: clamped to the end cap
  // monotone non-decreasing
  double previous = 0.0;
  for (int k = 0; k <= 60; ++k) {
    const double cap = p.capAt(0.005 * k);
    EXPECT_GE(cap, previous - 1e-12);
    previous = cap;
  }
}

TEST(LeftBladeCap, BandReachingTheEndLeavesNoRamp)
{
  const Profile p = profile(15.0, 30.0, 0.300, 0.300);
  EXPECT_DOUBLE_EQ(p.capAt(0.3), 15.0);
  EXPECT_DOUBLE_EQ(p.capAt(0.2), 15.0);
}

TEST(LeftBladeCap, BandExitIsDefinedByTravelOrder)
{
  EXPECT_DOUBLE_EQ(left_blade_cap::bandExitTravel(0.3, 0.12, 0.20, false), 0.20);
  EXPECT_NEAR(left_blade_cap::bandExitTravel(0.3, 0.12, 0.20, true), 0.18, 1e-12);
}

TEST(LeftBladeCap, StrokeWithinTheCapPassesAndReportsBandAndOverallMax)
{
  const Profile p = profile(15.0, 30.0);
  std::vector<Eigen::Isometry3d> poses;
  for (int k = 0; k <= 60; ++k) {
    const double s = 0.005 * k;
    // 10 deg in the band, rising along the ramp but staying under the cap
    poses.push_back(poseAtTheta(s, s <= 0.2 ? 10.0 : 10.0 + (s - 0.2) * 150.0));
  }
  const auto r = left_blade_cap::evaluateStroke(
    p, true, 0.08, 0.20, poses, Eigen::Vector3d::Zero(), Eigen::Vector3d(1, 0, 0), kNormal, kAlong, 1.0);
  EXPECT_TRUE(r.ok);
  EXPECT_NEAR(r.max_theta_p_band_deg, 10.0, 1e-6);
  EXPECT_NEAR(r.max_theta_p_deg, 25.0, 1e-6);
  EXPECT_NEAR(r.max_blade_deviation_deg, 0.0, 1e-5);
  EXPECT_NEAR(r.max_overshoot_deg, 0.0, 1e-9);
  EXPECT_EQ(r.waypoints, 61U);
}

TEST(LeftBladeCap, SteepBladeInTheBandIsRefusedEvenWhereTheRampWouldAllowIt)
{
  const Profile p = profile(15.0, 30.0);
  std::vector<Eigen::Isometry3d> poses;
  for (int k = 0; k <= 60; ++k) poses.push_back(poseAtTheta(0.005 * k, k == 20 ? 20.0 : 5.0));
  // k = 20 -> s = 0.1 m, inside the band: 20 deg > 15 + 1
  const auto r = left_blade_cap::evaluateStroke(
    p, true, 0.08, 0.20, poses, Eigen::Vector3d::Zero(), Eigen::Vector3d(1, 0, 0), kNormal, kAlong, 1.0);
  EXPECT_FALSE(r.ok);
  EXPECT_EQ(r.violation_index, 20U);
  EXPECT_NEAR(r.violation_theta_p_deg, 20.0, 1e-6);
  EXPECT_NEAR(r.violation_cap_deg, 15.0, 1e-12);
  EXPECT_NEAR(r.max_overshoot_deg, 5.0, 1e-6);
  // the same angle at the start of the stroke is refused too (start stays at the band cap)
  poses.assign(1, poseAtTheta(0.0, 20.0));
  EXPECT_FALSE(left_blade_cap::evaluateStroke(
      p, true, 0.08, 0.20, poses, Eigen::Vector3d::Zero(), Eigen::Vector3d(1, 0, 0), kNormal, kAlong, 1.0).ok);
}

TEST(LeftBladeCap, ToleranceIsInclusive)
{
  const Profile p = profile(15.0, 15.0);
  const std::vector<Eigen::Isometry3d> within{poseAtTheta(0.1, 15.9)};
  const std::vector<Eigen::Isometry3d> beyond{poseAtTheta(0.1, 16.1)};
  const Eigen::Vector3d o = Eigen::Vector3d::Zero();
  const Eigen::Vector3d d(1, 0, 0);
  EXPECT_TRUE(left_blade_cap::evaluateStroke(p, false, 0, 0, within, o, d, kNormal, kAlong, 1.0).ok);
  EXPECT_FALSE(left_blade_cap::evaluateStroke(p, false, 0, 0, beyond, o, d, kNormal, kAlong, 1.0).ok);
  EXPECT_TRUE(left_blade_cap::evaluateStroke(p, false, 0, 0, beyond, o, d, kNormal, kAlong, 1.2).ok);
  const auto report = left_blade_cap::evaluateStroke(p, false, 0, 0, within, o, d, kNormal, kAlong, 1.0);
  EXPECT_FALSE(report.has_band);
  EXPECT_NEAR(report.max_overshoot_deg, 0.9, 1e-6);
}

TEST(LeftBladeCap, NonFiniteInputFailsClosed)
{
  const Profile p = profile(15.0, 15.0);
  Eigen::Isometry3d bad = poseAtTheta(0.1, 5.0);
  bad.translation().x() = std::numeric_limits<double>::quiet_NaN();
  const auto r = left_blade_cap::evaluateStroke(
    p, false, 0, 0, {bad}, Eigen::Vector3d::Zero(), Eigen::Vector3d(1, 0, 0), kNormal, kAlong, 1.0);
  EXPECT_FALSE(r.ok);
}

// ---------------------------------------------------------------------------------------------
// Start cap (left_varying_start_max_theta_p_deg / left_varying_start_ramp_m)
namespace
{
// Band [entry, exit] inside a 0.300 m span; band cap `band`, start cap `start` ramping over `ramp`.
Profile startProfile(
  double band, double start, double ramp, double end = -1.0, double entry = 0.100,
  double exit = 0.200, double length = 0.300)
{
  Profile p = profile(band, end < 0.0 ? band : end, length, exit);
  p.start_cap_deg = start;
  p.start_ramp_m = ramp;
  p.band_entry_m = entry;
  return p;
}
}  // namespace

TEST(LeftBladeCapStart, OffIsIdenticalToTheOldProfile)
{
  for (const double end : {25.0, 40.0}) {
    const Profile old_profile = profile(25.0, end);
    for (const double start : {0.0, 25.0}) {   // 0 = off, equal to the band cap = nothing to add
      Profile p = startProfile(25.0, start, 0.05, end, 0.100, 0.200, 0.300);
      EXPECT_FALSE(p.startGraded());
      for (int k = -10; k <= 400; ++k) {
        const double s = 0.001 * k;
        EXPECT_DOUBLE_EQ(p.capAt(s), old_profile.capAt(s)) << "s=" << s << " end=" << end;
      }
    }
  }
  EXPECT_EQ(startProfile(25.0, 0.0, 0.05).rangeText(), "25.0..25.0");
}

TEST(LeftBladeCapStart, RampValues)
{
  const Profile p = startProfile(25.0, 40.0, 0.05);   // ramp over [0.05, 0.10]
  EXPECT_TRUE(p.startGraded());
  EXPECT_DOUBLE_EQ(p.capAt(0.0), 40.0);
  EXPECT_DOUBLE_EQ(p.capAt(0.03), 40.0);
  EXPECT_DOUBLE_EQ(p.capAt(0.05), 40.0);              // ramp begin
  EXPECT_NEAR(p.capAt(0.075), 32.5, 1e-9);            // mid-ramp
  EXPECT_NEAR(p.capAt(0.10), 25.0, 1e-9);             // band entry: exactly the band cap
  EXPECT_DOUBLE_EQ(p.capAt(0.15), 25.0);              // inside the band
  EXPECT_DOUBLE_EQ(p.capAt(0.20), 25.0);              // band exit
  EXPECT_DOUBLE_EQ(p.capAt(0.25), 25.0);              // after the band, no end ramp
  EXPECT_DOUBLE_EQ(p.capAt(0.30), 25.0);
  EXPECT_EQ(p.rangeText(), "start 40.0 (ramp 50 mm) -> band 25.0 -> end 25.0");
  // monotone non-increasing up to the band entry
  double previous = 100.0;
  for (int k = 0; k <= 100; ++k) {
    const double cap = p.capAt(0.001 * k);
    EXPECT_LE(cap, previous + 1e-12);
    previous = cap;
  }
}

TEST(LeftBladeCapStart, StartCapCombinesWithTheEndRamp)
{
  const Profile p = startProfile(25.0, 40.0, 0.05, 40.0);
  EXPECT_DOUBLE_EQ(p.capAt(0.0), 40.0);
  EXPECT_DOUBLE_EQ(p.capAt(0.15), 25.0);
  EXPECT_DOUBLE_EQ(p.capAt(0.20), 25.0);
  EXPECT_NEAR(p.capAt(0.25), 32.5, 1e-9);             // today's end ramp, unchanged
  EXPECT_DOUBLE_EQ(p.capAt(0.30), 40.0);
  // identical to the old profile from the band entry on
  const Profile old_profile = profile(25.0, 40.0);
  for (int k = 100; k <= 350; ++k) {
    EXPECT_DOUBLE_EQ(p.capAt(0.001 * k), old_profile.capAt(0.001 * k));
  }
}

TEST(LeftBladeCapStart, ClippedRampStillEndsAtTheBandEntry)
{
  const Profile p = startProfile(25.0, 40.0, 0.05, -1.0, /*entry=*/0.030);  // ramp would begin at -0.02
  EXPECT_NEAR(p.capAt(0.0), 40.0 + (25.0 - 40.0) * 0.02 / 0.05, 1e-9);      // 34: interpolated, not 40
  EXPECT_NEAR(p.capAt(0.015), 40.0 + (25.0 - 40.0) * 0.035 / 0.05, 1e-9);
  EXPECT_NEAR(p.capAt(0.030), 25.0, 1e-9);
  EXPECT_DOUBLE_EQ(p.capAt(0.05), 25.0);
  // waypoint noise just before s = 0 is the start; far before the span it is band cap
  EXPECT_NEAR(p.capAt(-1e-6), p.capAt(0.0), 1e-3);
  EXPECT_DOUBLE_EQ(p.capAt(-0.05), 25.0);
}

TEST(LeftBladeCapStart, ZeroRampIsAStepAtTheBandEntry)
{
  const Profile p = startProfile(25.0, 40.0, 0.0);
  EXPECT_DOUBLE_EQ(p.capAt(0.0), 40.0);
  EXPECT_DOUBLE_EQ(p.capAt(0.0999), 40.0);
  EXPECT_DOUBLE_EQ(p.capAt(0.10), 25.0);   // the band wins at the entry itself (fail closed)
  EXPECT_DOUBLE_EQ(p.capAt(0.1001), 25.0);
  EXPECT_DOUBLE_EQ(p.capAt(0.15), 25.0);
}

TEST(LeftBladeCapStart, BandEntryAtTheStartLeavesNoStartRegion)
{
  const Profile p = startProfile(25.0, 40.0, 0.05, -1.0, /*entry=*/0.0);
  for (int k = 0; k <= 300; ++k) EXPECT_DOUBLE_EQ(p.capAt(0.001 * k), 25.0);
}

TEST(LeftBladeCapStart, BandNeverAboveBandCapForAnyParameters)
{
  std::srand(12345);
  const auto uniform = [](double lo, double hi) {
      return lo + (hi - lo) * (static_cast<double>(std::rand()) / RAND_MAX);
    };
  for (int trial = 0; trial < 3000; ++trial) {
    const double length = uniform(0.05, 0.40);
    const double entry = uniform(0.0, length);
    const double exit = uniform(entry, length);
    const double band = uniform(5.0, 45.0);
    const double start = trial % 5 == 0 ? 0.0 : uniform(band, 45.0);
    const double end = uniform(band, 45.0);
    const double ramp = trial % 7 == 0 ? 0.0 : uniform(0.0, 0.2);
    Profile p = profile(band, end, length, exit);
    p.start_cap_deg = start;
    p.start_ramp_m = ramp;
    p.band_entry_m = entry;
    for (int k = 0; k <= 200; ++k) {
      const double s = entry + (exit - entry) * k / 200.0;
      ASSERT_LE(p.capAt(s), band + 1e-9) << "trial " << trial << " s=" << s;
      ASSERT_GE(p.capAt(s), band - 1e-9);
    }
    // the cap never exceeds max(start, end) and never drops below the band cap anywhere
    for (int k = -20; k <= 440; ++k) {
      const double s = 0.001 * k;
      const double cap = p.capAt(s);
      ASSERT_GE(cap, band - 1e-9);
      ASSERT_LE(cap, std::max({start, end, band}) + 1e-9);
    }
  }
}

TEST(LeftBladeCapStart, InconsistentGeometryFallsBackToTheBandCap)
{
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  const auto expect_band = [](const Profile& p, const char* what) {
      for (const double s : {0.0, 0.02, 0.05, 0.09, 0.15}) {
        EXPECT_DOUBLE_EQ(p.capAt(s), 25.0) << what << " s=" << s;
      }
    };
  Profile p = startProfile(25.0, 40.0, 0.05);
  p.band_entry_m = 0.25;  p.band_exit_m = 0.20;   // entry after exit
  expect_band(p, "entry > exit");
  p = startProfile(25.0, 40.0, 0.05, -1.0, -0.01);
  expect_band(p, "entry < 0");
  p = startProfile(25.0, 40.0, 0.05, -1.0, 0.35, 0.40, 0.300);
  expect_band(p, "entry > span");
  p = startProfile(25.0, 40.0, -0.05);
  expect_band(p, "negative ramp");
  p = startProfile(25.0, 40.0, nan);
  expect_band(p, "NaN ramp");
  p = startProfile(25.0, 40.0, 0.05, -1.0, nan);
  expect_band(p, "NaN entry");
  p = startProfile(25.0, 40.0, 0.05, -1.0, 0.1, inf);
  expect_band(p, "inf exit");
  p = startProfile(25.0, nan, 0.05);
  expect_band(p, "NaN start cap");
  p = startProfile(25.0, 40.0, 0.05);
  EXPECT_DOUBLE_EQ(p.capAt(nan), 25.0);
}

TEST(LeftBladeCapStart, EvaluateStrokeRefusesAboveBandCapInsideTheBandEvenWithStartCap45)
{
  const Profile p = startProfile(25.0, 45.0, 0.05);   // band [0.10, 0.20]
  const Eigen::Vector3d o = Eigen::Vector3d::Zero();
  const Eigen::Vector3d d(1, 0, 0);
  const auto eval = [&](double s, double theta, bool has_band = true) {
      return left_blade_cap::evaluateStroke(
        p, has_band, 0.10, 0.20, {poseAtTheta(s, theta)}, o, d, kNormal, kAlong, 1.0);
    };
  EXPECT_TRUE(eval(0.0, 45.0).ok);                  // steep at the start is allowed
  EXPECT_TRUE(eval(0.02, 44.0).ok);
  EXPECT_FALSE(eval(0.10, 40.0).ok);                // at the band entry the cap is the band cap
  const auto inside = eval(0.15, 40.0);
  EXPECT_FALSE(inside.ok);
  EXPECT_NEAR(inside.violation_cap_deg, 25.0, 1e-12);
  EXPECT_FALSE(eval(0.20, 26.5).ok);
  EXPECT_TRUE(eval(0.15, 25.5).ok);                 // within the 1 deg tolerance
  EXPECT_TRUE(eval(0.075, 32.0).ok);                // mid-ramp: cap 35 + 1
  EXPECT_FALSE(eval(0.075, 36.5).ok);
  // a Profile whose band_entry disagrees with the reported band is still held by the band args
  Profile wrong = p;
  wrong.band_entry_m = 0.19;                        // claims the band starts late
  const auto r = left_blade_cap::evaluateStroke(
    wrong, true, 0.10, 0.20, {poseAtTheta(0.15, 40.0)}, o, d, kNormal, kAlong, 1.0);
  EXPECT_FALSE(r.ok);
  // no band known: the start cap is not applied at all
  EXPECT_FALSE(eval(0.0, 45.0, /*has_band=*/false).ok);
}

TEST(LeftBladeCapStart, ParameterValidation)
{
  using left_blade_cap::validateStartCap;
  EXPECT_TRUE(validateStartCap(0.0, 25.0, 0.05).empty());     // off
  EXPECT_TRUE(validateStartCap(25.0, 25.0, 0.05).empty());    // equal to the band cap
  EXPECT_TRUE(validateStartCap(40.0, 25.0, 0.0).empty());
  EXPECT_TRUE(validateStartCap(45.0, 25.0, 0.2).empty());
  EXPECT_FALSE(validateStartCap(24.9, 25.0, 0.05).empty());   // below the band cap
  EXPECT_FALSE(validateStartCap(45.1, 25.0, 0.05).empty());
  EXPECT_FALSE(validateStartCap(-5.0, 25.0, 0.05).empty());
  EXPECT_FALSE(validateStartCap(40.0, 25.0, -0.01).empty());
  EXPECT_FALSE(validateStartCap(40.0, 25.0, 0.21).empty());
  EXPECT_FALSE(validateStartCap(0.0, 25.0, 0.21).empty());    // ramp is range-checked when off too
  const double nan = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(validateStartCap(nan, 25.0, 0.05).empty());
  EXPECT_FALSE(validateStartCap(40.0, 25.0, nan).empty());
  EXPECT_NE(validateStartCap(24.9, 25.0, 0.05).find("left_varying_start_max_theta_p_deg"),
    std::string::npos);
}
