// Tests for the pure top-cut sag profile (middle_cut_sag.hpp).
#include <gtest/gtest.h>

#include <cmath>
#include <string>

#include "igus_rebel_moveit_config/middle_cut_sag.hpp"

namespace
{
using Eigen::Vector3d;
using middle_cut_sag::Profile;

// capture_0920-like carton: seam along -y from the left edge (y=+0.0586) to the right edge
// (y=-0.3329), corners 0.14 m either side of the seam in x.
struct Carton {
  Vector3d seam_start{0.4307, 0.0586, 0.0307};
  Vector3d seam_end{0.4314, -0.3329, 0.0343};
  Vector3d seam_center{0.4310, -0.1371, 0.0325};
  Vector3d back_left{0.5707, 0.0590, 0.0309};
  Vector3d front_left{0.2921, 0.0585, 0.0305};
  Vector3d back_right{0.5730, -0.3349, 0.0346};
  Vector3d front_right{0.2912, -0.3310, 0.0341};
};

Profile make(const Carton& c, double sag, const Vector3d& start, const Vector3d& end)
{
  Profile profile;
  std::string error;
  EXPECT_TRUE(middle_cut_sag::makeProfile(
    start, end, c.seam_center, c.back_left, c.front_left, c.back_right, c.front_right, sag, profile,
    error)) << error;
  return profile;
}
}  // namespace

TEST(MiddleCutSag, HalfLengthIsHalfTheCartonAlongTheSeam)
{
  const Carton c;
  const Profile profile = make(c, 0.002, c.seam_start, c.seam_end);
  EXPECT_NEAR(profile.half_length_m, 0.1968, 0.001);
  EXPECT_NEAR(profile.axis.norm(), 1.0, 1e-12);
}

TEST(MiddleCutSag, CentreIsSagAndEndsAreZero)
{
  const Carton c;
  const Profile profile = make(c, 0.002, c.seam_start, c.seam_end);
  EXPECT_NEAR(middle_cut_sag::extraDepthM(profile, c.seam_center), 0.002, 1e-12);
  // The seam ends sit on the carton edges (d == h up to the corner noise): ~0.
  EXPECT_NEAR(middle_cut_sag::extraDepthM(profile, c.seam_start), 0.0, 5e-5);
  EXPECT_NEAR(middle_cut_sag::extraDepthM(profile, c.seam_end), 0.0, 5e-5);
}

TEST(MiddleCutSag, ParabolaShapeAndSymmetry)
{
  const Carton c;
  const Profile profile = make(c, 0.004, c.seam_start, c.seam_end);
  const Vector3d u = profile.axis;
  const double h = profile.half_length_m;
  for (const double f : {0.1, 0.25, 0.5, 0.75, 0.9}) {
    const double expected = 0.004 * (1.0 - f * f);
    EXPECT_NEAR(middle_cut_sag::extraDepthM(profile, c.seam_center + f * h * u), expected, 1e-12);
    EXPECT_NEAR(middle_cut_sag::extraDepthM(profile, c.seam_center - f * h * u), expected, 1e-12);
  }
  // Independent of the offset across the seam and of the height above it.
  const Vector3d across(0.05, 0.0, 0.02);
  EXPECT_NEAR(
    middle_cut_sag::extraDepthM(profile, c.seam_center + 0.5 * h * u + across),
    middle_cut_sag::extraDepthM(profile, c.seam_center + 0.5 * h * u), 1e-4);
}

TEST(MiddleCutSag, ClampsToZeroOutsideTheCarton)
{
  const Carton c;
  const Profile profile = make(c, 0.006, c.seam_start, c.seam_end);
  const Vector3d u = profile.axis;
  const double h = profile.half_length_m;
  EXPECT_DOUBLE_EQ(middle_cut_sag::extraDepthM(profile, c.seam_center + 1.2 * h * u), 0.0);
  EXPECT_DOUBLE_EQ(middle_cut_sag::extraDepthM(profile, c.seam_center - 3.0 * h * u), 0.0);
  EXPECT_GE(middle_cut_sag::extraDepthM(profile, c.seam_center + 0.999 * h * u), 0.0);
}

TEST(MiddleCutSag, InvariantToMovingSeamStartAlongTheAxis)
{
  const Carton c;
  const Profile full = make(c, 0.002, c.seam_start, c.seam_end);
  // Right-arm derived snapshot: seam_start at 80 % of the seam, corners and centre unchanged.
  const Vector3d trimmed_start = c.seam_start + 0.8 * (c.seam_end - c.seam_start);
  const Profile trimmed = make(c, 0.002, trimmed_start, c.seam_end);
  // Shortened end (left_max style) and a reversed seam.
  const Vector3d short_end = c.seam_start + 0.6 * (c.seam_end - c.seam_start);
  const Profile shortened = make(c, 0.002, c.seam_start, short_end);
  const Profile reversed = make(c, 0.002, c.seam_end, c.seam_start);
  EXPECT_NEAR(trimmed.half_length_m, full.half_length_m, 1e-9);
  for (const double t : {0.0, 0.2, 0.5, 0.8, 0.95, 1.0}) {
    const Vector3d p = c.seam_start + t * (c.seam_end - c.seam_start);
    const double s = middle_cut_sag::extraDepthM(full, p);
    EXPECT_NEAR(middle_cut_sag::extraDepthM(trimmed, p), s, 1e-9);
    EXPECT_NEAR(middle_cut_sag::extraDepthM(shortened, p), s, 1e-9);
    EXPECT_NEAR(middle_cut_sag::extraDepthM(reversed, p), s, 1e-9);
  }
}

TEST(MiddleCutSag, ZeroSagIsExactlyZeroEverywhere)
{
  const Carton c;
  const Profile profile = make(c, 0.0, c.seam_start, c.seam_end);
  for (const double t : {0.0, 0.3, 0.5, 1.0}) {
    EXPECT_EQ(middle_cut_sag::extraDepthM(profile, c.seam_start + t * (c.seam_end - c.seam_start)), 0.0);
  }
}

TEST(MiddleCutSag, FailsClosedOnBadInput)
{
  const Carton c;
  Profile profile;
  std::string error;
  const auto build = [&](double sag, const Vector3d& start, const Vector3d& end) {
    return middle_cut_sag::makeProfile(
      start, end, c.seam_center, c.back_left, c.front_left, c.back_right, c.front_right, sag,
      profile, error);
  };
  EXPECT_FALSE(build(-0.001, c.seam_start, c.seam_end));
  EXPECT_FALSE(build(0.0061, c.seam_start, c.seam_end));
  EXPECT_FALSE(build(std::nan(""), c.seam_start, c.seam_end));
  EXPECT_FALSE(build(0.002, c.seam_start, c.seam_start));
  // Corners collapsed onto one line across the seam: half length 0.
  EXPECT_FALSE(middle_cut_sag::makeProfile(
    c.seam_start, c.seam_end, c.seam_center, c.back_left, c.front_left, c.back_left, c.front_left,
    0.002, profile, error));
  EXPECT_TRUE(build(0.006, c.seam_start, c.seam_end));
}

TEST(MiddleCutSag, LegCount)
{
  EXPECT_EQ(middle_cut_sag::legCount(0.39, 0.01), 39U);
  EXPECT_EQ(middle_cut_sag::legCount(0.395, 0.01), 40U);
  EXPECT_EQ(middle_cut_sag::legCount(0.001, 0.01), 1U);
  EXPECT_EQ(middle_cut_sag::legCount(0.0, 0.01), 1U);
}
