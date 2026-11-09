// Tests for the pure interval / overlap / coverage maths of the left side cut reverse patch
// (left_cut_patch.hpp) and the blade-cap profile origin offset the patch search uses.
#include <gtest/gtest.h>

#include <cmath>

#include "igus_rebel_moveit_config/left_blade_cap.hpp"
#include "igus_rebel_moveit_config/left_cut_patch.hpp"

namespace
{
using namespace left_cut_patch;

TEST(LeftCutPatchParams, Ranges)
{
  EXPECT_TRUE(validateParams(0.015, 0.015).empty());
  EXPECT_TRUE(validateParams(0.0, 0.005).empty());
  EXPECT_TRUE(validateParams(0.10, 0.05).empty());
  EXPECT_FALSE(validateParams(-0.001, 0.015).empty());
  EXPECT_FALSE(validateParams(0.101, 0.015).empty());
  EXPECT_FALSE(validateParams(0.015, 0.0049).empty());
  EXPECT_FALSE(validateParams(0.015, 0.0501).empty());
  EXPECT_FALSE(validateParams(std::nan(""), 0.015).empty());
  EXPECT_FALSE(validateParams(0.015, std::nan("")).empty());
  EXPECT_FALSE(validateParams(INFINITY, 0.015).empty());
}

TEST(LeftCutPatchGap, NeedsPatchOnlyAboveMinGap)
{
  EXPECT_FALSE(needsPatch(0.0, 0.015));
  EXPECT_FALSE(needsPatch(0.015, 0.015));  // equal: skip
  EXPECT_FALSE(needsPatch(0.0149, 0.015));
  EXPECT_TRUE(needsPatch(0.0151, 0.015));
  EXPECT_TRUE(needsPatch(0.039, 0.015));
  EXPECT_TRUE(needsPatch(0.001, 0.0));
  EXPECT_FALSE(needsPatch(std::nan(""), 0.015));
}

TEST(LeftCutPatchLine, GeometryAndSlack)
{
  // span 263.5 mm, forward kept 39..263.5, overlap 15, slack 10
  const Line line = patchLine(0.2635, 0.039, 0.2635, 0.015, 0.010);
  ASSERT_TRUE(line.ok) << line.error;
  EXPECT_NEAR(line.overlap_point_arc_m, 0.054, 1e-12);
  EXPECT_NEAR(line.start_arc_m, 0.064, 1e-12);
  EXPECT_NEAR(line.length_m, 0.064, 1e-12);
  // 64 mm line at 10 mm spacing -> 7 intervals of 9.142857 mm
  const Required req = requiredStations(line, 7U);
  const double spacing = 0.064 / 7.0;
  EXPECT_EQ(req.first_max, static_cast<std::size_t>(std::floor(0.010 / spacing + 1e-9)));  // 1
  // forward start at t = 25 mm -> ceil(25/9.14)=3, +1 = 4
  EXPECT_EQ(req.last_min, 4U);
  EXPECT_LE(req.last_min, 7U);
}

TEST(LeftCutPatchLine, ClipsAtSpanEndAndRejectsDegenerate)
{
  // overlap point + slack beyond the span end -> clipped at the span length
  const Line clipped = patchLine(0.100, 0.080, 0.100, 0.015, 0.010);
  ASSERT_TRUE(clipped.ok) << clipped.error;
  EXPECT_NEAR(clipped.start_arc_m, 0.100, 1e-12);  // 0.095 + 0.010 clipped to the span length
}

TEST(LeftCutPatchLine, ForwardStrokeShorterThanOverlapFails)
{
  const Line line = patchLine(0.2, 0.190, 0.195, 0.015, 0.010);
  EXPECT_FALSE(line.ok);
  EXPECT_FALSE(line.error.empty());
  EXPECT_FALSE(patchLine(0.2, std::nan(""), 0.2, 0.015, 0.01).ok);
  EXPECT_FALSE(patchLine(0.0, 0.0, 0.2, 0.015, 0.01).ok);
  EXPECT_FALSE(patchLine(0.2, 0.05, 0.04, 0.015, 0.01).ok);
}

TEST(LeftCutPatchRequired, ClampedToIntervals)
{
  const Line line = patchLine(0.060, 0.040, 0.060, 0.015, 0.010);
  ASSERT_TRUE(line.ok) << line.error;
  const Required req = requiredStations(line, 2U);
  EXPECT_LE(req.first_max, 2U);
  EXPECT_LE(req.last_min, 2U);
  EXPECT_GE(req.last_min, 1U);
}

TEST(LeftCutPatchCoverage, UnionOfForwardAndPatch)
{
  // forward 39..263.5 (edge 273.5), patch 0..54: overlap 15 mm
  const Coverage c = coverage(0.039, 0.2635, true, 0.0, 0.054, 0.2735);
  EXPECT_NEAR(c.forward_kept_mm, 224.5, 1e-9);
  EXPECT_NEAR(c.gap_mm, 39.0, 1e-9);
  EXPECT_NEAR(c.patch_kept_mm, 54.0, 1e-9);
  EXPECT_NEAR(c.overlap_mm, 15.0, 1e-9);
  EXPECT_NEAR(c.total_covered_mm, 263.5, 1e-9);
  EXPECT_NEAR(c.percent_of_edge, 100.0 * 0.2635 / 0.2735, 1e-9);
  EXPECT_TRUE(c.patched);
}

TEST(LeftCutPatchCoverage, PartialPatchLeavesRemainingGap)
{
  // patch only reaches 10..54: still 10 mm uncovered at the back end
  const Coverage c = coverage(0.039, 0.2635, true, 0.010, 0.054, 0.2735);
  EXPECT_NEAR(c.total_covered_mm, 253.5, 1e-9);
}

TEST(LeftCutPatchCoverage, NoPatchIsForwardOnly)
{
  const Coverage c = coverage(0.039, 0.2635, false, 0.0, 0.0, 0.2735);
  EXPECT_FALSE(c.patched);
  EXPECT_NEAR(c.total_covered_mm, 224.5, 1e-9);
  EXPECT_NEAR(c.overlap_mm, 0.0, 1e-12);
}

TEST(LeftBladeCapOffset, CapAndBandUseTheShiftedCoordinate)
{
  left_blade_cap::Profile p;
  p.band_cap_deg = 15.0;
  p.end_cap_deg = 25.0;
  p.span_length_m = 0.300;
  p.band_exit_m = 0.100;
  EXPECT_NEAR(p.capAt(0.150), 15.0 + 10.0 * 0.25, 1e-9);
  // a patch whose line starts 0.050 into the full travel: local s = 0.100 is full s = 0.150
  p.origin_offset_m = 0.050;
  EXPECT_NEAR(p.capAtLocal(0.100), 15.0 + 10.0 * 0.25, 1e-9);
  EXPECT_NEAR(p.capAtLocal(0.0), 15.0, 1e-9);
}
}  // namespace
