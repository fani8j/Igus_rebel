// Tests for the pure helpers of the varying-orientation left cut (left_varying_chain.hpp).
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <set>
#include <tuple>
#include <vector>

#include "igus_rebel_moveit_config/left_varying_chain.hpp"

namespace
{
using left_varying::ChainDp;
using left_varying::Node;

constexpr double kPi = 3.14159265358979323846;

// Orientations: rotation about Z by k * 5 degrees, k = 0..count-1 (adjacent ones are 5 deg apart).
std::vector<std::array<double, 4> > yawLadder(std::size_t count)
{
  std::vector<std::array<double, 4> > q;
  for (std::size_t k = 0; k < count; ++k) {
    const double half = 0.5 * 5.0 * static_cast<double>(k) * kPi / 180.0;
    q.push_back({0.0, 0.0, std::sin(half), std::cos(half)});
  }
  return q;
}

Node node(std::size_t candidate, double joint, double score)
{
  return Node{candidate, {joint, 0.0, 0.0, 0.0, 0.0, 0.0}, score};
}
}  // namespace

TEST(LeftVaryingChain, StationIntervals)
{
  EXPECT_EQ(left_varying::stationIntervals(0.160, 0.010), 16U);
  EXPECT_EQ(left_varying::stationIntervals(0.165, 0.010), 17U);
  EXPECT_EQ(left_varying::stationIntervals(0.005, 0.010), 1U);
  EXPECT_EQ(left_varying::stationIntervals(0.0, 0.010), 1U);
  EXPECT_EQ(left_varying::stationIntervals(0.16, 0.0), 1U);
  EXPECT_EQ(left_varying::stationIntervals(std::nan(""), 0.01), 1U);
}

TEST(LeftVaryingChain, NeighborTableUsesRotationAngle)
{
  const auto table = left_varying::buildNeighborTable(yawLadder(4), 5.0);
  ASSERT_EQ(table.size(), 4U);
  // candidate 0: itself and candidate 1 (5 deg); candidate 2 is 10 deg away.
  ASSERT_EQ(table[0].size(), 2U);
  EXPECT_EQ(table[0][0].first, 0U);
  EXPECT_NEAR(table[0][0].second, 0.0, 1e-9);
  EXPECT_EQ(table[0][1].first, 1U);
  EXPECT_NEAR(table[0][1].second, 5.0, 1e-6);
  EXPECT_EQ(table[1].size(), 3U);
  const auto wide = left_varying::buildNeighborTable(yawLadder(4), 10.0);
  EXPECT_EQ(wide[0].size(), 3U);
  // q and -q are the same rotation.
  const std::vector<std::array<double, 4> > flipped{{0, 0, 0, 1}, {0, 0, 0, -1}};
  EXPECT_EQ(left_varying::buildNeighborTable(flipped, 1.0)[0].size(), 2U);
}

TEST(LeftVaryingChain, LongestContiguousChainWinsOverScore)
{
  const auto table = left_varying::buildNeighborTable(yawLadder(6), 5.0);
  ChainDp dp({0.2, 0.0}, table, {}, {});
  // Station 0: only candidate 0. Station 1: candidates 0,1. Station 2: candidate 2 only
  // (10 deg from candidate 0, 5 deg from candidate 1).
  // Station 3: nothing feasible (gap). Station 4: candidate 4, cheap. Station 5: candidate 4.
  dp.addStation({node(0, 0.0, 1.0)}, true);
  dp.addStation({node(0, 0.01, 1.0), node(1, 0.01, 5.0)}, true);
  dp.addStation({node(2, 0.02, 5.0)}, true);
  dp.addStation({}, true);
  dp.addStation({node(4, 0.0, 0.0)}, true);
  dp.addStation({node(4, 0.0, 0.0)}, true);
  EXPECT_EQ(dp.bestLength(), 3U);
  const auto ends = dp.rankedEnds();
  ASSERT_FALSE(ends.empty());
  EXPECT_EQ(ends.front().length, 3U);
  EXPECT_EQ(ends.front().station, 2U);
  const auto chain = dp.extract(ends.front());
  EXPECT_EQ(chain.first_station, 0U);
  ASSERT_EQ(chain.nodes.size(), 3U);
  // The chain must route through candidate 1 at station 1 (candidate 0 is not a neighbour of 2).
  EXPECT_EQ(dp.node(1, chain.nodes[1]).candidate, 1U);
}

TEST(LeftVaryingChain, EqualLengthPicksLowerScoreAndSmoothness)
{
  const auto table = left_varying::buildNeighborTable(yawLadder(3), 5.0);
  ChainDp dp({0.2, 0.01}, table, {}, {});
  dp.addStation({node(0, 0.0, 3.0), node(1, 0.0, 1.0)}, true);
  dp.addStation({node(0, 0.0, 3.0), node(1, 0.0, 1.0)}, true);
  const auto ends = dp.rankedEnds();
  ASSERT_FALSE(ends.empty());
  const auto chain = dp.extract(ends.front());
  ASSERT_EQ(chain.nodes.size(), 2U);
  EXPECT_EQ(dp.node(0, chain.nodes[0]).candidate, 1U);
  EXPECT_EQ(dp.node(1, chain.nodes[1]).candidate, 1U);
  EXPECT_NEAR(chain.score, 2.0, 1e-9);  // no orientation step, no smoothness penalty
}

TEST(LeftVaryingChain, JointBranchJumpBreaksChain)
{
  const auto table = left_varying::buildNeighborTable(yawLadder(2), 5.0);
  ChainDp dp({0.2, 0.0}, table, {}, {});
  dp.addStation({node(0, 0.0, 1.0)}, true);
  // Same orientation but the only IK branch is 3.0 rad away (a wrap): no continuation.
  dp.addStation({node(0, 3.0, 1.0)}, true);
  EXPECT_EQ(dp.bestLength(), 1U);
  ChainDp wrap({0.2, 0.0}, table, {}, {});
  wrap.addStation({node(0, 0.0, 1.0)}, true);
  wrap.addStation({node(0, 0.15, 1.0)}, true);
  EXPECT_EQ(wrap.bestLength(), 2U);
}

TEST(LeftVaryingChain, StartAndEdgeGatesAreHonoured)
{
  const auto table = left_varying::buildNeighborTable(yawLadder(2), 5.0);
  // Starting at station 0 is not allowed: the chain can only begin at station 1.
  ChainDp gated({0.2, 0.0}, table,
    [](std::size_t station, std::size_t, const Node&) { return station >= 1U; }, {});
  gated.addStation({node(0, 0.0, 1.0)}, true);
  gated.addStation({node(0, 0.0, 1.0)}, true);
  gated.addStation({node(0, 0.0, 1.0)}, true);
  EXPECT_EQ(gated.bestLength(), 2U);
  EXPECT_EQ(gated.rankedEnds().front().station, 2U);
  // A banned edge splits the chain; allow_new_starts=false never starts a chain.
  ChainDp banned({0.2, 0.0}, table, {},
    [](std::size_t station, std::size_t, std::size_t) { return station != 1U; });
  banned.addStation({node(0, 0.0, 1.0)}, true);
  banned.addStation({node(0, 0.0, 1.0)}, true);
  banned.addStation({node(0, 0.0, 1.0)}, true);
  EXPECT_EQ(banned.bestLength(), 2U);
  ChainDp nostart({0.2, 0.0}, table, {}, {});
  nostart.addStation({node(0, 0.0, 1.0)}, false);
  EXPECT_EQ(nostart.bestLength(), 0U);
  EXPECT_TRUE(nostart.aliveCandidates().empty());
}

TEST(LeftVaryingChain, AliveCandidatesAndBranchChoice)
{
  const auto table = left_varying::buildNeighborTable(yawLadder(3), 5.0);
  ChainDp dp({0.2, 0.0}, table, {}, {});
  // Two IK branches of one candidate; only the second continues to station 1.
  dp.addStation({node(1, 0.0, 1.0), node(1, 1.0, 1.0)}, true);
  dp.addStation({node(1, 1.05, 1.0)}, true);
  const auto ends = dp.rankedEnds();
  const auto chain = dp.extract(ends.front());
  ASSERT_EQ(chain.nodes.size(), 2U);
  EXPECT_NEAR(dp.node(0, chain.nodes[0]).q[0], 1.0, 1e-12);
  const auto alive = dp.aliveCandidates();
  ASSERT_EQ(alive.size(), 1U);
  EXPECT_EQ(alive[0], 1U);
}

// Counterexample to the removed "skip new starts once the best chain so far is longer than the
// stations that remain" pruning: 7 stations; chain A (stations 0-3, candidate 0) is enterable but
// has no valid exit; chain B (stations 4-6, candidate 3, 15 deg away so it cannot be reached from
// A) is enterable and leavable. With pruning, A (length 4) made station 4 unable to start a chain
// (7 - 4 = 3 < 4) and B was never found. All stations are fed with new starts allowed here, so
// the verified end of B (length 3) must be selected.
TEST(LeftVaryingChain, DisconnectedVerifiedChainIsNotSuppressedByUnleavableLongerChain)
{
  const auto table = left_varying::buildNeighborTable(yawLadder(4), 5.0);
  ChainDp dp({0.2, 0.0}, table, {}, {});
  for (std::size_t station = 0; station < 7U; ++station) {
    dp.addStation({node(station < 4U ? 0U : 3U, 0.0, 1.0)}, true);
  }
  EXPECT_EQ(dp.bestLength(), 4U);
  ChainDp::End chosen;
  const auto end_ok = [](std::size_t station, std::size_t) { return station >= 4U; };
  ASSERT_TRUE(left_varying::bestVerifiedEnd(dp, end_ok, chosen));
  EXPECT_EQ(chosen.station, 6U);
  EXPECT_EQ(chosen.length, 3U);
  const auto chain = dp.extract(chosen);
  EXPECT_EQ(chain.first_station, 4U);
  // With a minimum length above B's, nothing qualifies.
  EXPECT_FALSE(left_varying::bestVerifiedEnd(dp, end_ok, chosen, 4U));
  // And if A's end were fine, the longer chain wins as before.
  ASSERT_TRUE(left_varying::bestVerifiedEnd(dp, {}, chosen));
  EXPECT_EQ(chosen.length, 4U);
}

TEST(LeftVaryingChain, NeighborBuildHonoursAbort)
{
  const auto orientations = yawLadder(6);
  left_varying::NeighborTable table;
  std::size_t polls = 0;
  EXPECT_FALSE(left_varying::buildNeighborTableBounded(
    orientations, 5.0, table, [&polls]() { return ++polls > 2U; }));
  EXPECT_LE(polls, 3U);
  EXPECT_TRUE(left_varying::buildNeighborTableBounded(orientations, 5.0, table, {}));
  EXPECT_EQ(table.size(), 6U);
}

// ---- left_cut_extend_to_corners ----------------------------------------------------------

TEST(LeftVaryingExtend, SpanFromCornersMinusMargin)
{
  // Nominal line 0..0.160 m; corners at -0.080 and +0.240 m (edge 0.320 m, the nominal segment is
  // the middle half), seam crossing at the nominal midpoint, margin 20 mm, cover 30 mm.
  const auto span = left_varying::computeExtendedSpan(0.160, -0.080, 0.240, 0.080, 0.020, 0.030);
  ASSERT_TRUE(span.ok) << span.error;
  EXPECT_NEAR(span.s_start, -0.060, 1e-12);
  EXPECT_NEAR(span.s_end, 0.220, 1e-12);
  EXPECT_NEAR(span.length_m, 0.280, 1e-12);
  EXPECT_NEAR(span.band_lo, 0.050, 1e-12);
  EXPECT_NEAR(span.band_hi, 0.110, 1e-12);
}

TEST(LeftVaryingExtend, SeamBandIsClampedToSpan)
{
  // Seam 10 mm from the span start: the band is clamped, not extended past the span.
  const auto span = left_varying::computeExtendedSpan(0.160, -0.080, 0.240, -0.050, 0.020, 0.030);
  ASSERT_TRUE(span.ok) << span.error;
  EXPECT_NEAR(span.band_lo, -0.060, 1e-12);
  EXPECT_NEAR(span.band_hi, -0.020, 1e-12);
  const auto zero_cover = left_varying::computeExtendedSpan(0.160, -0.080, 0.240, 0.080, 0.020, 0.0);
  ASSERT_TRUE(zero_cover.ok);
  EXPECT_NEAR(zero_cover.band_lo, 0.080, 1e-12);
  EXPECT_NEAR(zero_cover.band_hi, 0.080, 1e-12);
}

TEST(LeftVaryingExtend, DegenerateOrShortSpansFailClosed)
{
  // Margin eats the whole edge.
  EXPECT_FALSE(left_varying::computeExtendedSpan(0.160, -0.080, 0.240, 0.080, 0.200, 0.03).ok);
  // Corners reversed.
  EXPECT_FALSE(left_varying::computeExtendedSpan(0.160, 0.240, -0.080, 0.080, 0.02, 0.03).ok);
  // Extended span shorter than the nominal segment (edge 0.2 m, margin 0.05 -> 0.1 m < 0.16 m).
  EXPECT_FALSE(left_varying::computeExtendedSpan(0.160, -0.020, 0.180, 0.080, 0.05, 0.03).ok);
  // Does not contain the nominal segment (shifted past its start) although long enough.
  EXPECT_FALSE(left_varying::computeExtendedSpan(0.160, 0.010, 0.400, 0.200, 0.0, 0.03).ok);
  // Seam outside the span.
  EXPECT_FALSE(left_varying::computeExtendedSpan(0.160, -0.080, 0.240, 0.300, 0.02, 0.03).ok);
  // Non-finite / non-positive inputs.
  EXPECT_FALSE(left_varying::computeExtendedSpan(std::nan(""), -0.08, 0.24, 0.08, 0.02, 0.03).ok);
  EXPECT_FALSE(left_varying::computeExtendedSpan(0.0, -0.08, 0.24, 0.08, 0.02, 0.03).ok);
  EXPECT_FALSE(left_varying::computeExtendedSpan(0.16, -0.08, 0.24, 0.08, -0.01, 0.03).ok);
}

TEST(LeftVaryingExtend, RequiredStationRange)
{
  // Span 0.28 m, 28 intervals of 10 mm; band [0.110, 0.170] -> first station <= 11, last >= 17.
  auto range = left_varying::requiredStationRange(0.110, 0.170, 0.28, 28U);
  EXPECT_EQ(range.first_max, 11U);
  EXPECT_EQ(range.last_min, 17U);
  // Off-grid: lo floors, hi ceils, so the band is always covered.
  range = left_varying::requiredStationRange(0.1149, 0.1651, 0.28, 28U);
  EXPECT_EQ(range.first_max, 11U);
  EXPECT_EQ(range.last_min, 17U);
  // Band at the span ends.
  range = left_varying::requiredStationRange(0.0, 0.020, 0.28, 28U);
  EXPECT_EQ(range.first_max, 0U);
  EXPECT_EQ(range.last_min, 2U);
  range = left_varying::requiredStationRange(0.250, 0.280, 0.28, 28U);
  EXPECT_EQ(range.first_max, 25U);
  EXPECT_EQ(range.last_min, 28U);
  // Degenerate input is harmless.
  range = left_varying::requiredStationRange(0.1, 0.2, 0.0, 0U);
  EXPECT_EQ(range.first_max, 0U);
  EXPECT_EQ(range.last_min, 0U);
}

namespace
{
// One orientation, one joint branch: feasibility is the per-station set only.
ChainDp feasibleStations(const left_varying::NeighborTable& table, const std::vector<bool>& ok)
{
  ChainDp dp({0.2, 0.0}, table, {}, {});
  for (const bool feasible : ok) {
    dp.addStation(feasible ? std::vector<Node>{node(0, 0.0, 1.0)} : std::vector<Node>{}, true);
  }
  return dp;
}
}  // namespace

TEST(LeftVaryingExtend, ShorterIntervalContainingSeamBeatsLongerOneWithout)
{
  const auto table = left_varying::buildNeighborTable(yawLadder(1), 5.0);
  // 12 stations: 0-6 feasible (7 long, no seam), 7 infeasible, 8-11 feasible (4 long).
  const auto dp = feasibleStations(
    table, {true, true, true, true, true, true, true, false, true, true, true, true});
  ChainDp::End chosen;
  ASSERT_TRUE(left_varying::bestVerifiedEnd(dp, {}, chosen));  // unconstrained: the long one
  EXPECT_EQ(chosen.length, 7U);
  EXPECT_EQ(chosen.station, 6U);
  // Seam band needs stations 9..10 (first <= 9, last >= 10): only the short interval covers it.
  ASSERT_TRUE(left_varying::bestVerifiedEndCovering(dp, {}, chosen, 1U, 9U, 10U));
  EXPECT_EQ(chosen.length, 4U);
  EXPECT_EQ(chosen.station, 11U);
  EXPECT_EQ(dp.extract(chosen).first_station, 8U);
  // A minimum length above the covering interval's rejects it.
  EXPECT_FALSE(left_varying::bestVerifiedEndCovering(dp, {}, chosen, 5U, 9U, 10U));
}

TEST(LeftVaryingExtend, NoIntervalContainingSeamFails)
{
  const auto table = left_varying::buildNeighborTable(yawLadder(1), 5.0);
  // The seam station 5 is infeasible: intervals 0-4 and 6-9 each miss a band edge.
  const auto dp = feasibleStations(
    table, {true, true, true, true, true, false, true, true, true, true});
  ChainDp::End chosen;
  ASSERT_TRUE(left_varying::bestVerifiedEnd(dp, {}, chosen));
  EXPECT_GE(chosen.length, 4U);
  EXPECT_FALSE(left_varying::bestVerifiedEndCovering(dp, {}, chosen, 1U, 4U, 6U));
  // A band that fits on one side is still found.
  ASSERT_TRUE(left_varying::bestVerifiedEndCovering(dp, {}, chosen, 1U, 2U, 3U));
  EXPECT_EQ(chosen.station, 4U);
}

TEST(LeftVaryingExtend, RequiredBandAtSpanEnds)
{
  const auto table = left_varying::buildNeighborTable(yawLadder(1), 5.0);
  // Feasible 3..9 of 0..9: a band at the back end (first <= 0) is uncoverable, at the front end
  // (last >= 9) it is coverable.
  const auto dp = feasibleStations(
    table, {false, false, false, true, true, true, true, true, true, true});
  ChainDp::End chosen;
  EXPECT_FALSE(left_varying::bestVerifiedEndCovering(dp, {}, chosen, 1U, 0U, 1U));
  ASSERT_TRUE(left_varying::bestVerifiedEndCovering(dp, {}, chosen, 1U, 8U, 9U));
  EXPECT_EQ(chosen.station, 9U);
  EXPECT_EQ(chosen.length, 7U);
  // Fully feasible: the band at the back end is satisfied by the whole span.
  const auto all = feasibleStations(table, std::vector<bool>(10U, true));
  ASSERT_TRUE(left_varying::bestVerifiedEndCovering(all, {}, chosen, 1U, 0U, 1U));
  EXPECT_EQ(chosen.length, 10U);
}

TEST(LeftVaryingExtend, CoveringHonoursEndGate)
{
  const auto table = left_varying::buildNeighborTable(yawLadder(1), 5.0);
  const auto dp = feasibleStations(table, std::vector<bool>(8U, true));
  ChainDp::End chosen;
  // The full-length end (station 7) is un-leavable: the next covering end is station 6 (length 7).
  const auto end_ok = [](std::size_t station, std::size_t) { return station != 7U; };
  ASSERT_TRUE(left_varying::bestVerifiedEndCovering(dp, end_ok, chosen, 1U, 3U, 4U));
  EXPECT_EQ(chosen.station, 6U);
  EXPECT_EQ(chosen.length, 7U);
}

TEST(LeftVaryingInward, TaperIsLinearFromExtraToZero)
{
  const double extra = 0.003;
  EXPECT_DOUBLE_EQ(left_varying::inwardTaperM(extra, 0.0, 0.0, 0.2), extra);
  EXPECT_NEAR(left_varying::inwardTaperM(extra, 0.1, 0.0, 0.2), 0.0015, 1e-12);
  EXPECT_NEAR(left_varying::inwardTaperM(extra, 0.05, 0.0, 0.2), 0.00225, 1e-12);
  EXPECT_DOUBLE_EQ(left_varying::inwardTaperM(extra, 0.2, 0.0, 0.2), 0.0);
  // A window that does not start at the origin (kept interval).
  EXPECT_DOUBLE_EQ(left_varying::inwardTaperM(extra, 0.05, 0.05, 0.15), extra);
  EXPECT_NEAR(left_varying::inwardTaperM(extra, 0.10, 0.05, 0.15), 0.0015, 1e-12);
  EXPECT_DOUBLE_EQ(left_varying::inwardTaperM(extra, 0.15, 0.05, 0.15), 0.0);
}

TEST(LeftVaryingInward, ClampedOutsideWindowAndZeroWhenDisabled)
{
  EXPECT_DOUBLE_EQ(left_varying::inwardTaperM(0.003, -0.05, 0.0, 0.2), 0.003);
  EXPECT_DOUBLE_EQ(left_varying::inwardTaperM(0.003, 0.3, 0.0, 0.2), 0.0);
  EXPECT_DOUBLE_EQ(left_varying::inwardTaperM(0.0, 0.1, 0.0, 0.2), 0.0);
  EXPECT_DOUBLE_EQ(left_varying::inwardTaperM(-0.001, 0.1, 0.0, 0.2), 0.0);
  EXPECT_DOUBLE_EQ(left_varying::inwardTaperM(0.003, 0.1, 0.2, 0.2), 0.0);
  EXPECT_DOUBLE_EQ(left_varying::inwardTaperM(std::nan(""), 0.1, 0.0, 0.2), 0.0);
}
