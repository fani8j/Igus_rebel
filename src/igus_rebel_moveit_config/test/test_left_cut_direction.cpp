// Tests for the pure travel-direction maths of the varying left cut (left_cut_direction.hpp):
// extension / seam-band mapping, travel basis, reference frames of both reverse variants,
// penetration gate sign mapping, physical trims and the blade-cap profile in reverse.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include "igus_rebel_moveit_config/left_blade_cap.hpp"
#include "igus_rebel_moveit_config/left_cut_direction.hpp"
#include "igus_rebel_moveit_config/left_varying_chain.hpp"

namespace
{
using left_cut_direction::Basis;
using left_cut_direction::Convention;
using left_cut_direction::Direction;
using left_cut_direction::ReverseFrame;
using left_cut_direction::SpanGeometry;
constexpr double kDeg = 3.14159265358979323846 / 180.0;

// A tilted carton top: normal tilted 3 deg off +Z, nominal back -> front along -X (like the real
// left face: back at larger x, front at smaller x), across = normal x along.
Basis nominalBasis()
{
  Basis basis;
  basis.normal = Eigen::Vector3d(std::sin(3 * kDeg), 0.0, std::cos(3 * kDeg)).normalized();
  basis.along = Eigen::Vector3d(-1.0, 0.0, 0.0);
  basis.along -= basis.along.dot(basis.normal) * basis.normal;
  basis.along.normalize();
  basis.across = basis.normal.cross(basis.along);
  return basis;
}

// Real-ish numbers: nominal 141 mm, extended span 270 mm (back corner margin gives 50 mm in front
// of back_middle_left), seam band 30 mm either side of the seam at 100 mm.
SpanGeometry forwardGeometry()
{
  SpanGeometry g;
  g.span_length_m = 0.270;
  g.nominal_start_m = 0.050;
  g.nominal_length_m = 0.141;
  g.seam_m = 0.100;
  g.band_lo_m = 0.070;
  g.band_hi_m = 0.130;
  return g;
}

Eigen::Matrix3d rotationAbout(const Eigen::Vector3d& axis, double angle)
{
  return Eigen::AngleAxisd(angle, axis.normalized()).toRotationMatrix();
}
}  // namespace

TEST(LeftCutDirection, ParsesTheTwoParameters)
{
  EXPECT_EQ(left_cut_direction::directionFromString("forward"), Direction::kForward);
  EXPECT_EQ(left_cut_direction::directionFromString("reverse"), Direction::kReverse);
  EXPECT_FALSE(left_cut_direction::directionFromString("auto"));
  EXPECT_FALSE(left_cut_direction::directionFromString(""));
  EXPECT_EQ(left_cut_direction::reverseFrameFromString("about_normal"), ReverseFrame::kAboutNormal);
  EXPECT_EQ(left_cut_direction::reverseFrameFromString("about_edge"), ReverseFrame::kAboutEdge);
  EXPECT_FALSE(left_cut_direction::reverseFrameFromString("about_x"));
  // The frame variant only matters for reverse travel.
  EXPECT_EQ(left_cut_direction::conventionFor(Direction::kForward, ReverseFrame::kAboutEdge),
    Convention::kStandard);
  EXPECT_EQ(left_cut_direction::conventionFor(Direction::kReverse, ReverseFrame::kAboutNormal),
    Convention::kStandard);
  EXPECT_EQ(left_cut_direction::conventionFor(Direction::kReverse, ReverseFrame::kAboutEdge),
    Convention::kEdgeFlipped);
}

TEST(LeftCutDirection, ExtensionMappingIsTheMirrorAndSelfInverse)
{
  const SpanGeometry f = forwardGeometry();
  const SpanGeometry same = left_cut_direction::toTravel(f, false);
  EXPECT_DOUBLE_EQ(same.nominal_start_m, f.nominal_start_m);
  EXPECT_DOUBLE_EQ(same.band_lo_m, f.band_lo_m);
  EXPECT_DOUBLE_EQ(same.band_hi_m, f.band_hi_m);

  const SpanGeometry r = left_cut_direction::toTravel(f, true);
  EXPECT_DOUBLE_EQ(r.span_length_m, 0.270);
  EXPECT_DOUBLE_EQ(r.nominal_length_m, 0.141);
  // nominal segment: back_middle_left at forward 0.050, front_middle_left at 0.191 -> in travel
  // coordinates the segment starts at 0.270 - 0.191 = 0.079 (that is front_middle_left).
  EXPECT_NEAR(r.nominal_start_m, 0.079, 1e-12);
  EXPECT_NEAR(r.seam_m, 0.170, 1e-12);
  EXPECT_NEAR(r.band_lo_m, 0.140, 1e-12);   // L - band_hi
  EXPECT_NEAR(r.band_hi_m, 0.200, 1e-12);   // L - band_lo
  EXPECT_NEAR(r.band_hi_m - r.band_lo_m, f.band_hi_m - f.band_lo_m, 1e-12);

  const SpanGeometry back = left_cut_direction::toTravel(r, true);
  EXPECT_NEAR(back.nominal_start_m, f.nominal_start_m, 1e-12);
  EXPECT_NEAR(back.seam_m, f.seam_m, 1e-12);
  EXPECT_NEAR(back.band_lo_m, f.band_lo_m, 1e-12);
  EXPECT_NEAR(back.band_hi_m, f.band_hi_m, 1e-12);
}

TEST(LeftCutDirection, PhysicalPositionsAreIndependentOfTheDirection)
{
  const SpanGeometry f = forwardGeometry();
  const SpanGeometry r = left_cut_direction::toTravel(f, true);
  // The seam sits 50 mm beyond back_middle_left toward the front, whichever way we travel.
  EXPECT_NEAR(left_cut_direction::physicalFromBackMiddle(f, false, f.seam_m), 0.050, 1e-12);
  EXPECT_NEAR(left_cut_direction::physicalFromBackMiddle(r, true, r.seam_m), 0.050, 1e-12);
  // The back corner end of the span is -50 mm; the front end 220 mm from back_middle_left.
  EXPECT_NEAR(left_cut_direction::physicalFromBackMiddle(f, false, 0.0), -0.050, 1e-12);
  EXPECT_NEAR(left_cut_direction::physicalFromBackMiddle(r, true, r.span_length_m), -0.050, 1e-12);
  EXPECT_NEAR(left_cut_direction::physicalFromBackMiddle(r, true, 0.0), 0.220, 1e-12);
  // The band in physical coordinates is the same (20..80 mm) for both directions.
  const double lo_f = left_cut_direction::physicalFromBackMiddle(f, false, f.band_lo_m);
  const double hi_f = left_cut_direction::physicalFromBackMiddle(f, false, f.band_hi_m);
  const double a = left_cut_direction::physicalFromBackMiddle(r, true, r.band_lo_m);
  const double b = left_cut_direction::physicalFromBackMiddle(r, true, r.band_hi_m);
  EXPECT_NEAR(std::min(a, b), lo_f, 1e-12);
  EXPECT_NEAR(std::max(a, b), hi_f, 1e-12);
}

TEST(LeftCutDirection, TrimsAreReportedAsPhysicalBackAndFront)
{
  // 30 mm cut off the first end of the travel, 10 mm off the last.
  const auto fwd = left_cut_direction::physicalTrims(30.0, 10.0, false);
  EXPECT_DOUBLE_EQ(fwd.back_mm, 30.0);
  EXPECT_DOUBLE_EQ(fwd.front_mm, 10.0);
  const auto rev = left_cut_direction::physicalTrims(30.0, 10.0, true);
  EXPECT_DOUBLE_EQ(rev.back_mm, 10.0);    // the last end of the reverse travel is the back corner
  EXPECT_DOUBLE_EQ(rev.front_mm, 30.0);
}

TEST(LeftCutDirection, RequiredStationRangeOnTheMappedBandCoversTheSameSeam)
{
  const SpanGeometry f = forwardGeometry();
  const SpanGeometry r = left_cut_direction::toTravel(f, true);
  const std::size_t intervals = 27;  // 10 mm spacing
  const auto fwd = left_varying::requiredStationRange(f.band_lo_m, f.band_hi_m, f.span_length_m, intervals);
  const auto rev = left_varying::requiredStationRange(r.band_lo_m, r.band_hi_m, r.span_length_m, intervals);
  // Mirror image: station i of the forward span is station intervals - i of the reverse span.
  EXPECT_EQ(rev.first_max, intervals - fwd.last_min);
  EXPECT_EQ(rev.last_min, intervals - fwd.first_max);
}

TEST(LeftCutDirection, TravelBasisIsRightHandedAndFlipsAlongAcross)
{
  const Basis nominal = nominalBasis();
  const Basis fwd = left_cut_direction::travelBasis(nominal, false);
  const Basis rev = left_cut_direction::travelBasis(nominal, true);
  EXPECT_LT((fwd.along - nominal.along).norm(), 1e-15);
  EXPECT_LT((rev.along + nominal.along).norm(), 1e-15);
  EXPECT_LT((rev.across + nominal.across).norm(), 1e-15);
  EXPECT_LT((rev.normal - nominal.normal).norm(), 1e-15);
  EXPECT_NEAR(rev.along.cross(rev.across).dot(rev.normal), 1.0, 1e-12);
  EXPECT_NEAR(rev.normal.cross(rev.along).dot(rev.across), 1.0, 1e-12);
}

TEST(LeftCutDirection, ForwardReferenceFrameIsTheExistingConvention)
{
  const Basis nominal = nominalBasis();
  const Eigen::Matrix3d f =
    left_cut_direction::referenceFrame(left_cut_direction::travelBasis(nominal, false),
      Convention::kStandard);
  EXPECT_LT((f.col(0) + nominal.normal).norm(), 1e-15);   // X = -normal
  EXPECT_LT((f.col(1) + nominal.along).norm(), 1e-15);    // Y = -along
  EXPECT_LT((f.col(2) - nominal.normal.cross(nominal.along)).norm(), 1e-12);  // Z = X x Y
  EXPECT_LT((f.col(2) - nominal.across).norm(), 1e-12);   // = across on the left face
  EXPECT_NEAR(f.determinant(), 1.0, 1e-12);
}

TEST(LeftCutDirection, ReverseAboutNormalIsTheForwardFrameRotated180AboutTheNormal)
{
  const Basis nominal = nominalBasis();
  const Eigen::Matrix3d f = left_cut_direction::referenceFrame(
    left_cut_direction::travelBasis(nominal, false), Convention::kStandard);
  const Eigen::Matrix3d r = left_cut_direction::referenceFrame(
    left_cut_direction::travelBasis(nominal, true), Convention::kStandard);
  EXPECT_NEAR(r.determinant(), 1.0, 1e-12);
  // Y = -along_travel = +along_nominal: the blade length axis reverses with the travel.
  EXPECT_LT((r.col(1) - nominal.along).norm(), 1e-15);
  EXPECT_LT((r.col(1) + f.col(1)).norm(), 1e-15);
  // X = -normal is unchanged, Z flips: across the cut line the knife axis points the other way.
  EXPECT_LT((r.col(0) - f.col(0)).norm(), 1e-15);
  EXPECT_LT((r.col(2) + f.col(2)).norm(), 1e-12);
  const Eigen::Matrix3d rotation_n = rotationAbout(nominal.normal, 180.0 * kDeg);
  EXPECT_LT((rotation_n * f - r).norm(), 1e-12);
}

TEST(LeftCutDirection, ReverseAboutEdgeIsTheForwardFrameRotated180AboutToolZ)
{
  const Basis nominal = nominalBasis();
  const Eigen::Matrix3d f = left_cut_direction::referenceFrame(
    left_cut_direction::travelBasis(nominal, false), Convention::kStandard);
  const Eigen::Matrix3d e = left_cut_direction::referenceFrame(
    left_cut_direction::travelBasis(nominal, true), Convention::kEdgeFlipped);
  EXPECT_NEAR(e.determinant(), 1.0, 1e-12);
  EXPECT_LT((e.col(1) - nominal.along).norm(), 1e-15);       // Y = -along_travel
  EXPECT_LT((e.col(2) - f.col(2)).norm(), 1e-12);            // the knife axis keeps its side
  EXPECT_LT((e.col(0) + f.col(0)).norm(), 1e-15);            // X = +normal: blade upside down
  EXPECT_LT((e.col(0) - nominal.normal).norm(), 1e-15);
  const Eigen::Matrix3d rotation_z = rotationAbout(f.col(2), 180.0 * kDeg);
  EXPECT_LT((rotation_z * f - e).norm(), 1e-12);
}

TEST(LeftCutDirection, ThetaPIsUnchangedByTheFlip)
{
  const Basis nominal = nominalBasis();
  // Same physical tilt about the cut line applied to every variant's reference frame.
  for (const double tilt_deg : {0.0, 5.0, 12.5, -9.0, 30.0}) {
    std::vector<double> theta;
    for (const auto& variant : {std::make_pair(false, Convention::kStandard),
        std::make_pair(true, Convention::kStandard), std::make_pair(true, Convention::kEdgeFlipped)}) {
      const Basis travel = left_cut_direction::travelBasis(nominal, variant.first);
      const Eigen::Matrix3d reference = left_cut_direction::referenceFrame(travel, variant.second);
      const Eigen::Matrix3d rotation = rotationAbout(travel.along, tilt_deg * kDeg) * reference;
      theta.push_back(left_blade_cap::thetaPDeg(rotation, nominal.normal));
      // blade deviation from the cut line (tool Y vs -travel) is also unchanged
      EXPECT_NEAR(left_blade_cap::bladeDeviationDeg(rotation, travel.along), 0.0, 1e-6);
    }
    EXPECT_NEAR(theta[0], std::abs(tilt_deg), 1e-9);
    EXPECT_NEAR(theta[1], theta[0], 1e-9);
    EXPECT_NEAR(theta[2], theta[0], 1e-9);
  }
}

TEST(LeftCutDirection, PenetrationGateUsesTheSignMappedXForTheEdgeFlippedFrame)
{
  const Basis nominal = nominalBasis();
  const Basis rev = left_cut_direction::travelBasis(nominal, true);
  const Eigen::Matrix3d standard = left_cut_direction::referenceFrame(rev, Convention::kStandard);
  const Eigen::Matrix3d flipped = left_cut_direction::referenceFrame(rev, Convention::kEdgeFlipped);
  // The flat blade of each variant is accepted by its own gate.
  EXPECT_TRUE(left_cut_direction::penetrationGateOk(standard, nominal.normal, Convention::kStandard));
  EXPECT_TRUE(left_cut_direction::penetrationGateOk(flipped, nominal.normal, Convention::kEdgeFlipped));
  // The UNMAPPED gate would reject the flipped frame (X = +normal, Z horizontal): the sign map is
  // what keeps the flat upside-down blade a valid candidate.
  EXPECT_FALSE(left_cut_direction::penetrationGateOk(flipped, nominal.normal, Convention::kStandard));
  // The mapping is only the X sign: a perpendicular blade (knife axis Z down) is accepted by both,
  // a blade whose X and Z both lie far from "into the carton" by neither.
  Eigen::Matrix3d perpendicular;
  perpendicular.col(2) = -nominal.normal;
  perpendicular.col(1) = rev.along;
  perpendicular.col(0) = perpendicular.col(1).cross(perpendicular.col(2));
  EXPECT_TRUE(left_cut_direction::penetrationGateOk(perpendicular, nominal.normal, Convention::kStandard));
  EXPECT_TRUE(left_cut_direction::penetrationGateOk(perpendicular, nominal.normal, Convention::kEdgeFlipped));
  const Eigen::Matrix3d tilted_out = rotationAbout(rev.along, 60.0 * kDeg) * standard;
  EXPECT_FALSE(left_cut_direction::penetrationGateOk(tilted_out, nominal.normal, Convention::kStandard));
  // (the sense of the rotation matters: about +along_nominal the flat upside-down blade's knife axis
  // swings UP, away from the carton; about -along it would swing down and be a valid steep blade)
  const Eigen::Matrix3d flipped_out = rotationAbout(nominal.along, 60.0 * kDeg) * flipped;
  EXPECT_FALSE(left_cut_direction::penetrationGateOk(flipped_out, nominal.normal, Convention::kEdgeFlipped));
  // The forward gate reads exactly the old expression.
  const Eigen::Matrix3d forward = left_cut_direction::referenceFrame(
    left_cut_direction::travelBasis(nominal, false), Convention::kStandard);
  const Eigen::Vector3d into = -nominal.normal;
  EXPECT_DOUBLE_EQ(left_cut_direction::penetrationDotX(forward, nominal.normal, Convention::kStandard),
    forward.col(0).dot(into));
  EXPECT_DOUBLE_EQ(left_cut_direction::penetrationDotZ(forward, nominal.normal), forward.col(2).dot(into));
}

TEST(LeftCutDirection, ReverseRampLiesTowardTheBackCornerAndTheBandKeepsTheBandCap)
{
  const SpanGeometry f = forwardGeometry();
  const SpanGeometry r = left_cut_direction::toTravel(f, true);
  left_blade_cap::Profile forward;
  forward.band_cap_deg = 15.0;
  forward.end_cap_deg = 30.0;
  forward.span_length_m = f.span_length_m;
  forward.band_exit_m = left_blade_cap::bandExitTravel(f.span_length_m, f.band_lo_m, f.band_hi_m, false);
  left_blade_cap::Profile reverse = forward;
  // In reverse the profile is built from the MAPPED geometry (travel coordinates, forward formula)
  // and that equals the generic reverse = true formula applied to the forward geometry.
  reverse.band_exit_m = left_blade_cap::bandExitTravel(r.span_length_m, r.band_lo_m, r.band_hi_m, false);
  EXPECT_NEAR(reverse.band_exit_m,
    left_blade_cap::bandExitTravel(f.span_length_m, f.band_lo_m, f.band_hi_m, true), 1e-12);
  // Physical positions (mm from back_middle_left): cap at the front corner (reverse START) and in
  // the band stays 15; cap at the back corner (reverse END) is the end cap.
  const auto physical_cap = [&](const left_blade_cap::Profile& p, const SpanGeometry& g, bool rev,
      double from_back_m) {
      const double s = rev ? g.nominal_start_m + g.nominal_length_m - from_back_m :
        from_back_m + g.nominal_start_m;
      return p.capAt(s);
    };
  EXPECT_NEAR(physical_cap(reverse, r, true, 0.220), 15.0, 1e-9);  // front corner = travel start
  EXPECT_NEAR(physical_cap(reverse, r, true, 0.080), 15.0, 1e-9);  // band upper edge (front side)
  EXPECT_NEAR(physical_cap(reverse, r, true, 0.020), 15.0, 1e-9);  // band lower edge (back side)
  EXPECT_NEAR(physical_cap(reverse, r, true, -0.050), 30.0, 1e-9); // back corner = travel end
  EXPECT_GT(physical_cap(reverse, r, true, -0.020), 15.0);
  EXPECT_LT(physical_cap(reverse, r, true, -0.020), 30.0);
  // Forward is the mirror image: the ramp is toward the FRONT corner.
  EXPECT_NEAR(physical_cap(forward, f, false, -0.050), 15.0, 1e-9);
  EXPECT_NEAR(physical_cap(forward, f, false, 0.220), 30.0, 1e-9);
}
