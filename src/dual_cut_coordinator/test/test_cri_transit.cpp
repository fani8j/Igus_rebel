// Unit tests of the robot-side joint move logic (cri_transit.hpp): parameter ranges, constants, eligibility, run split,
// gates, result classification, doStop order, and the padded certificate on the synthetic dual cell.
#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <limits>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <geometric_shapes/shapes.h>

#include "dual_cut_coordinator/cri_transit.hpp"
#include "dual_cut_coordinator/phase_validator.hpp"
#include "test_common.hpp"

using namespace dual_cut;

namespace
{
const std::vector<std::string> k6 = {"joint1", "joint2", "joint3", "joint4", "joint5", "joint6"};
constexpr double kDeg = kCriPi / 180.0;

TrajPoint pt(double t, std::vector<double> q)
{
  TrajPoint p;
  p.time_s = t;
  p.positions = std::move(q);
  return p;
}

// straight (linear in position, no velocities) segment from a to b with n+1 waypoints on the chord
PlannedSegmentData lineSeg(SegmentKind kind, const std::string & name, std::vector<double> a, std::vector<double> b, int n = 4, std::vector<std::string> names = k6)
{
  PlannedSegmentData s;
  s.kind = kind;
  s.stage_name = name;
  s.trajectory.joint_names = std::move(names);
  for (int k = 0; k <= n; ++k)
  {
    std::vector<double> q(a.size());
    for (size_t j = 0; j < a.size(); ++j) q[j] = a[j] + (b[j] - a[j]) * k / n;
    s.trajectory.points.push_back(pt(1.0 * k, q));
  }
  return s;
}

using V = std::vector<double>;
const V kHome{0, 0, 0, 0, 0, 0};

// Align (home -> A) | Transit (A -> B) | Cut (B -> C)
ArmPhase threeSegments(SegmentKind middle = SegmentKind::Transit)
{
  ArmPhase ph;
  ph.side = "left";
  ph.joint_names = k6;
  const V A{0.1, 0.0, 0.0, 0.0, 0.0, 0.0};
  const V B{0.9, 0.4, -0.2, 0.1, 0.0, 0.3};
  const V C{0.9, 0.5, -0.2, 0.1, 0.0, 0.3};
  ph.segments = {lineSeg(SegmentKind::Align, "align", kHome, A), lineSeg(middle, "transit", A, B), lineSeg(SegmentKind::Cut, "cut", B, C)};
  ph.start_q = kHome;
  ph.end_q = C;
  return ph;
}

CriGeomLimits limits()
{
  CriGeomLimits l;
  l.min_length_rad = 0.05;
  l.goal_tolerance_rad = 0.005;
  return l;
}
}  // namespace

// ---- parameters -------------------------------------------------------------------------------

TEST(CriParams, DefaultsAreValidAndOff)
{
  const CriParams p;
  EXPECT_FALSE(p.enabled);
  EXPECT_FALSE(p.assume_driver_cap_ok);
  EXPECT_EQ(p.velocity_percent, 20.0);
  EXPECT_EQ(p.acceleration_percent, 30.0);
  EXPECT_EQ(p.min_length_rad, 0.05);
  EXPECT_EQ(p.path_tolerance_rad, 0.005);
  EXPECT_EQ(p.reach_m, 0.0);   // derive the rigorous bound
  EXPECT_NEAR(p.subchord_velocity_rad_s, 0.20 * 0.785, 1e-9);
  EXPECT_NEAR(p.subchord_acceleration_rad_s2, 0.400, 1e-9);
  EXPECT_EQ(p.goal_tolerance_rad, 0.005);
  EXPECT_EQ(p.start_tolerance_rad, 0.01);
  EXPECT_EQ(p.max_runs_per_arm_phase, 1);
  EXPECT_EQ(p.driver_max_velocity_cap_percent, 25.0);
  EXPECT_TRUE(validateCriParams(p));
}

TEST(CriParams, OutOfRangeIsRejectedNotClamped)
{
  const double nan = std::numeric_limits<double>::quiet_NaN();
  EXPECT_TRUE(validateCriParameter("cri_transit_velocity_percent", 25.0));
  EXPECT_FALSE(validateCriParameter("cri_transit_velocity_percent", 25.01));
  EXPECT_FALSE(validateCriParameter("cri_transit_velocity_percent", 0.0));
  EXPECT_FALSE(validateCriParameter("cri_transit_velocity_percent", -5.0));
  EXPECT_FALSE(validateCriParameter("cri_transit_velocity_percent", nan));
  EXPECT_TRUE(validateCriParameter("cri_transit_acceleration_percent", 40.0));
  EXPECT_FALSE(validateCriParameter("cri_transit_acceleration_percent", 40.5));
  EXPECT_FALSE(validateCriParameter("cri_transit_acceleration_percent", 0.0));
  EXPECT_FALSE(validateCriParameter("cri_transit_path_tolerance_rad", 0.0));
  EXPECT_FALSE(validateCriParameter("cri_transit_path_tolerance_rad", 0.5));
  EXPECT_FALSE(validateCriParameter("cri_transit_goal_tolerance_rad", -0.001));
  EXPECT_FALSE(validateCriParameter("cri_transit_start_tolerance_rad", 0.2));
  EXPECT_FALSE(validateCriParameter("cri_transit_min_length_rad", 0.0));
  EXPECT_FALSE(validateCriParameter("cri_driver_max_velocity_cap_percent", 120.0));
  EXPECT_TRUE(validateCriParameter("cri_transit_reach_m", 0.0));      // 0 = derive
  EXPECT_FALSE(validateCriParameter("cri_transit_reach_m", 0.05));
  EXPECT_TRUE(validateCriParameter("cri_subchord_velocity_rad_s", 0.157));
  EXPECT_FALSE(validateCriParameter("cri_subchord_velocity_rad_s", 0.9));
  EXPECT_FALSE(validateCriParameter("cri_subchord_acceleration_rad_s2", 0.0));
  EXPECT_TRUE(validateCriParameter("cri_transit_max_runs_per_arm_phase", int64_t{1}));
  EXPECT_TRUE(validateCriParameter("cri_transit_max_runs_per_arm_phase", int64_t{4}));
  EXPECT_FALSE(validateCriParameter("cri_transit_max_runs_per_arm_phase", int64_t{0}));
  EXPECT_FALSE(validateCriParameter("cri_transit_max_runs_per_arm_phase", int64_t{5}));
  CriParams p;
  p.velocity_percent = 30.0;
  EXPECT_FALSE(validateCriParams(p));
}

// ---- constants ----------------------------------------------------------------------------------

TEST(CriConstants, StopTravelPaddingAndModel)
{
  // v = 0.66 deg/s per % * 20 % = 13.2 deg/s; * 0.06 s = 0.792 deg; + 0.2 deg = 0.992 deg
  EXPECT_NEAR(criStopTravelRad(20.0), 0.992 * kDeg, 1e-12);
  EXPECT_NEAR(criStopTravelRad(25.0), (0.66 * 25 * 0.06 + 0.2) * kDeg, 1e-12);
  EXPECT_GT(criStopTravelRad(25.0), criStopTravelRad(10.0));
  // the stop travel is information only: the padding is path tolerance * reach
  EXPECT_NEAR(criPaddingM(0.005, 3.6), 0.018, 1e-12);
  EXPECT_NEAR(criPaddingM(0.01, 1.0), 0.01, 1e-12);
  // 1 rad at 20 % / 30 %: cruise 12 deg/s = 0.20944 rad/s, accel 0.75 rad/s^2: 1/0.20944 + 0.20944/0.75 + 0.2
  EXPECT_NEAR(criModelDurationS(1.0, 20.0, 30.0), 1.0 / (12.0 * kDeg) + 12.0 * kDeg / 0.75 + 0.2, 1e-9);
  EXPECT_GT(criModelDurationS(2.0, 20.0, 30.0), criModelDurationS(1.0, 20.0, 30.0));
  EXPECT_LT(criModelDurationS(0.001, 20.0, 30.0), 1.0);   // a very short move is still the ramp + latency
  EXPECT_NEAR(criTimeoutS(10.0), 30.0, 1e-12);
}

TEST(CriConstants, ReachBoundOfTheSyntheticArm)
{
  const auto model = dual_cut_test::syntheticModel();
  ASSERT_NE(model, nullptr);
  ArmSpec left;
  ASSERT_TRUE(makeArmSpec(*model, "left", left));
  double sum = 0.0;
  std::vector<double> per;
  const Status s = armReachBound(*model, left, sum, &per);
  ASSERT_TRUE(s) << s.message;
  ASSERT_EQ(per.size(), 2u);
  // joint1: link2 origin 0.4, tool joint 0.2, tool sphere centre 0.03 + radius 0.03 -> 0.66; joint2: 0.2 + 0.06 -> 0.26
  EXPECT_NEAR(per[0], 0.66, 2e-3);
  EXPECT_NEAR(per[1], 0.26, 2e-3);
  EXPECT_NEAR(sum, 0.92, 4e-3);
}

// ---- eligibility --------------------------------------------------------------------------------

TEST(CriEligibility, StraightTransitIsEligible)
{
  const ArmPhase ph = threeSegments();
  const CriLegGeometry g = checkTransitGeometry(ph, 1, limits());
  EXPECT_TRUE(g.eligible) << g.reason;
  EXPECT_NEAR(g.length_rad, 0.8, 1e-12);
  EXPECT_EQ(g.joint_names, k6);
  EXPECT_NEAR(g.start[0], 0.1, 1e-12);
  EXPECT_NEAR(g.goal[0], 0.9, 1e-12);
  EXPECT_LT(g.max_chord_deviation_rad, 1e-9);
}

TEST(CriEligibility, OnlySegmentKindTransitCountsNeverTheStageName)
{
  ArmPhase ph = threeSegments(SegmentKind::Align);   // middle one is an Align called "transit"
  const CriLegGeometry g = checkTransitGeometry(ph, 1, limits());
  EXPECT_FALSE(g.eligible);
  EXPECT_NE(g.reason.find("not Transit"), std::string::npos) << g.reason;
  // a Transit with a misleading stage name is still eligible
  ph = threeSegments();
  ph.segments[1].stage_name = "cut left";
  EXPECT_TRUE(checkTransitGeometry(ph, 1, limits()).eligible);
  // scanTransitLegs lists Transit segments only
  const auto legs = scanTransitLegs(threeSegments(), limits());
  ASSERT_EQ(legs.size(), 1u);
  EXPECT_EQ(legs[0].segment_index, 1u);
  EXPECT_TRUE(scanTransitLegs(threeSegments(SegmentKind::Lift), limits()).empty());
}

TEST(CriEligibility, ExactlySixJoints)
{
  ArmPhase ph;
  ph.side = "left";
  ph.joint_names = {"joint1", "joint2", "joint3", "joint4", "joint5"};
  const std::vector<std::string> five = ph.joint_names;
  ph.segments = {lineSeg(SegmentKind::Transit, "t", {0, 0, 0, 0, 0}, {1, 0, 0, 0, 0}, 4, five)};
  ph.start_q = {0, 0, 0, 0, 0};
  ph.end_q = {1, 0, 0, 0, 0};
  const CriLegGeometry g = checkTransitGeometry(ph, 0, limits());
  EXPECT_FALSE(g.eligible);
  EXPECT_NE(g.reason.find("not exactly 6"), std::string::npos) << g.reason;
  // the 2-joint synthetic cell can use another count explicitly
  CriGeomLimits l = limits();
  l.joint_count = 5;
  EXPECT_TRUE(checkTransitGeometry(ph, 0, l).eligible);
}

TEST(CriEligibility, WaypointOffTheChordWithinAndBeyondOneMilliradian)
{
  ArmPhase ph = threeSegments();
  ph.segments[1].trajectory.points[2].positions[3] += 0.0006;   // 0.6 mrad: still on the chord
  EXPECT_TRUE(checkTransitGeometry(ph, 1, limits()).eligible);
  ph = threeSegments();
  ph.segments[1].trajectory.points[2].positions[3] += 0.0025;   // 2.5 mrad: off
  const CriLegGeometry g = checkTransitGeometry(ph, 1, limits());
  EXPECT_FALSE(g.eligible);
  EXPECT_NE(g.reason.find("off the straight chord"), std::string::npos) << g.reason;
}

TEST(CriEligibility, SplineBetweenChordWaypointsMustStayOnTheChordToo)
{
  // all waypoints ON the chord, but the cubic spline (velocities not along the chord) bulges away between them
  ArmPhase ph = threeSegments();
  auto & tr = ph.segments[1].trajectory;
  tr.points.assign({tr.points.front(), tr.points.back()});
  tr.points[0].time_s = 0.0;
  tr.points[1].time_s = 2.0;
  tr.points[0].velocities = V{0, 0.5, 0, 0, 0, 0};   // sideways on joint 2 only: chord moves joint 2 by 0.4 over 2 s
  tr.points[1].velocities = V{0, 0.0, 0, 0, 0, 0};
  const CriLegGeometry g = checkTransitGeometry(ph, 1, limits());
  EXPECT_FALSE(g.eligible);
  EXPECT_NE(g.reason.find("sampled point"), std::string::npos) << g.reason;
}

TEST(CriEligibility, MonotoneProgressAndNoOvershoot)
{
  ArmPhase ph = threeSegments();
  // a waypoint that goes back along the chord (progress 0.2 after 0.5) but stays on the line
  auto & pts = ph.segments[1].trajectory.points;
  pts[2].positions = pts[1].positions;   // waypoint 2 repeats waypoint 1: progress 0.25 after 0.25 is fine ...
  pts[1].positions = {0.1 + 0.8 * 0.5, 0.4 * 0.5, -0.2 * 0.5, 0.1 * 0.5, 0.0, 0.3 * 0.5};   // ... but now 0.5 then 0.25: backwards
  const CriLegGeometry g = checkTransitGeometry(ph, 1, limits());
  EXPECT_FALSE(g.eligible);
  EXPECT_NE(g.reason.find("goes back"), std::string::npos) << g.reason;
  // beyond the chord end
  ph = threeSegments();
  ph.segments[1].trajectory.points[3].positions = {0.1 + 0.8 * 1.1, 0.4 * 1.1, -0.2 * 1.1, 0.1 * 1.1, 0.0, 0.3 * 1.1};
  EXPECT_FALSE(checkTransitGeometry(ph, 1, limits()).eligible);
}

TEST(CriEligibility, MinimumLength)
{
  ArmPhase ph;
  ph.side = "left";
  ph.joint_names = k6;
  const V a{0, 0, 0, 0, 0, 0};
  const V b{0.04, 0, 0, 0, 0, 0};
  ph.segments = {lineSeg(SegmentKind::Transit, "t", a, b)};
  ph.start_q = a;
  ph.end_q = b;
  const CriLegGeometry g = checkTransitGeometry(ph, 0, limits());
  EXPECT_FALSE(g.eligible);
  EXPECT_NE(g.reason.find("minimum"), std::string::npos) << g.reason;
  CriGeomLimits l = limits();
  l.min_length_rad = 0.03;
  EXPECT_TRUE(checkTransitGeometry(ph, 0, l).eligible);
}

TEST(CriEligibility, ChordEndsMustMeetTheNeighbours)
{
  ArmPhase ph = threeSegments();
  ph.segments[2].trajectory.points[0].positions[0] += 0.01;   // next segment starts 10 mrad away from the chord end
  CriLegGeometry g = checkTransitGeometry(ph, 1, limits());
  EXPECT_FALSE(g.eligible);
  EXPECT_NE(g.reason.find("next segment"), std::string::npos) << g.reason;
  ph = threeSegments();
  ph.segments[0].trajectory.points.back().positions[1] += 0.01;   // previous segment ends 10 mrad away from the chord start
  g = checkTransitGeometry(ph, 1, limits());
  EXPECT_FALSE(g.eligible);
  EXPECT_NE(g.reason.find("previous segment"), std::string::npos) << g.reason;
  ph = threeSegments();
  ph.segments[2].trajectory.points[0].positions[0] += 0.004;   // within the 5 mrad goal tolerance
  EXPECT_TRUE(checkTransitGeometry(ph, 1, limits()).eligible);
}

TEST(CriEligibility, FirstAndLastSegmentUseThePhaseEnds)
{
  ArmPhase ph;
  ph.side = "left";
  ph.joint_names = k6;
  const V a{0.0, 0, 0, 0, 0, 0}, b{0.5, 0.2, 0, 0, 0, 0};
  ph.segments = {lineSeg(SegmentKind::Transit, "t", a, b)};
  ph.start_q = a;
  ph.end_q = b;
  EXPECT_TRUE(checkTransitGeometry(ph, 0, limits()).eligible);
  ph.start_q = V{0.02, 0, 0, 0, 0, 0};
  EXPECT_FALSE(checkTransitGeometry(ph, 0, limits()).eligible);
  ph.start_q = a;
  ph.end_q = V{0.5, 0.3, 0, 0, 0, 0};
  EXPECT_FALSE(checkTransitGeometry(ph, 0, limits()).eligible);
}

TEST(CriChord, ResampleStepAndEnds)
{
  const V a{0, 0, 0, 0, 0, 0}, b{0.0, 0.0, 0.5, -0.03, 0.0, 0.0};
  const auto c = resampleChord(a, b, 0.01);
  ASSERT_GE(c.size(), 51u);
  EXPECT_EQ(c.size(), 51u);   // 0.5 / 0.01 = 50 steps
  EXPECT_EQ(c.front(), a);
  EXPECT_EQ(c.back(), b);
  for (size_t k = 1; k < c.size(); ++k) EXPECT_LE(maxAbsDiff(c[k], c[k - 1]), 0.01 + 1e-12);
  // a non-multiple length still ends exactly on the goal with steps below the limit
  const auto d = resampleChord(a, V{0.123, 0, 0, 0, 0, 0}, 0.01);
  EXPECT_EQ(d.size(), 14u);
  EXPECT_EQ(d.back()[0], 0.123);
  for (size_t k = 1; k < d.size(); ++k) EXPECT_LE(maxAbsDiff(d[k], d[k - 1]), 0.01 + 1e-12);
  // zero length: two identical points
  EXPECT_EQ(resampleChord(a, a, 0.01).size(), 2u);
  EXPECT_TRUE(resampleChord(a, V{1.0}, 0.01).empty());
}

// ---- selection and runs ---------------------------------------------------------------------

namespace
{
CriLegRecord leg(size_t seg, double length, bool certified)
{
  CriLegRecord r;
  r.geometry.segment_index = seg;
  r.geometry.eligible = true;
  r.geometry.length_rad = length;
  r.cri_length_rad = length;
  r.certified = certified;
  return r;
}
}  // namespace

TEST(CriSelection, LongestCertifiedLegsUpToTheMaximum)
{
  CriArmPhasePlan ap;
  ap.legs = {leg(1, 0.4, true), leg(3, 0.9, true), leg(5, 0.7, true), leg(7, 1.5, false)};
  selectCriLegs(ap, 1);
  EXPECT_EQ(ap.selected_segments, (std::vector<size_t>{3}));
  EXPECT_TRUE(ap.legs[1].selected);
  EXPECT_FALSE(ap.legs[0].selected);
  EXPECT_NE(ap.legs[0].reason.find("not selected"), std::string::npos);
  EXPECT_TRUE(ap.legs[3].reason.empty() || ap.legs[3].reason.find("not selected") == std::string::npos);   // uncertified keeps its own reason
  selectCriLegs(ap, 2);
  EXPECT_EQ(ap.selected_segments, (std::vector<size_t>{3, 5}));   // sorted by segment
  selectCriLegs(ap, 4);
  EXPECT_EQ(ap.selected_segments, (std::vector<size_t>{1, 3, 5}));
  CriArmPhasePlan none;
  none.legs = {leg(2, 0.5, false)};
  selectCriLegs(none, 1);
  EXPECT_TRUE(none.selected_segments.empty());
}

TEST(CriRuns, SplitIntoJtcCriJtc)
{
  using R = CriRun;
  using T = CriRunType;
  EXPECT_EQ(splitArmPhaseRuns(5, {2}), (std::vector<R>{{T::Jtc, 0, 2}, {T::Cri, 2, 3}, {T::Jtc, 3, 5}}));
  EXPECT_EQ(splitArmPhaseRuns(5, {0}), (std::vector<R>{{T::Cri, 0, 1}, {T::Jtc, 1, 5}}));
  EXPECT_EQ(splitArmPhaseRuns(5, {4}), (std::vector<R>{{T::Jtc, 0, 4}, {T::Cri, 4, 5}}));
  EXPECT_EQ(splitArmPhaseRuns(3, {0, 1, 2}), (std::vector<R>{{T::Cri, 0, 1}, {T::Cri, 1, 2}, {T::Cri, 2, 3}}));
  EXPECT_EQ(splitArmPhaseRuns(6, {1, 4}), (std::vector<R>{{T::Jtc, 0, 1}, {T::Cri, 1, 2}, {T::Jtc, 2, 4}, {T::Cri, 4, 5}, {T::Jtc, 5, 6}}));
  EXPECT_TRUE(splitArmPhaseRuns(0, {}).empty());
  // unsorted / duplicate / out-of-range indices never create a CRI run out of order (they stay on the JTC)
  EXPECT_EQ(splitArmPhaseRuns(4, {9}), (std::vector<R>{{T::Jtc, 0, 4}}));
  EXPECT_EQ(splitArmPhaseRuns(4, {2, 2}), (std::vector<R>{{T::Jtc, 0, 2}, {T::Cri, 2, 3}, {T::Jtc, 3, 4}}));
}

TEST(CriRuns, CriDisabledGivesTheSingleJtcRunOfToday)
{
  CriArmPhasePlan ap;
  ap.legs = {leg(2, 0.9, true)};
  selectCriLegs(ap, 1);
  ASSERT_EQ(ap.selected_segments.size(), 1u);
  const std::vector<CriRun> today{{CriRunType::Jtc, 0, 5}};
  EXPECT_EQ(criRunsForArmPhase(false, true, &ap, 5), today);   // disabled
  EXPECT_EQ(criRunsForArmPhase(true, false, &ap, 5), today);   // driver cap not ok
  EXPECT_EQ(criRunsForArmPhase(true, true, nullptr, 5), today);   // no plan-time analysis
  CriArmPhasePlan empty;
  EXPECT_EQ(criRunsForArmPhase(true, true, &empty, 5), today);   // nothing selected
  EXPECT_EQ(criRunsForArmPhase(true, true, &ap, 5), splitArmPhaseRuns(5, {2}));
  EXPECT_EQ(splitArmPhaseRuns(5, {}), today);
}

// ---- live helpers -------------------------------------------------------------------------------

TEST(CriLive, WindowVelocityIgnoresJitterAndSeesMotion)
{
  WindowVelocity w(0.15, 0.5);
  EXPECT_TRUE(std::isinf(w.maxAbsVelocity()));   // no baseline: never "at rest"
  const V q0{0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
  for (int i = 0; i < 12; ++i)
  {
    V q = q0;
    q[2] += (i % 2 ? 0.001 : -0.001);   // +-1 mrad STATUS noise at 30 Hz
    w.add(0.033 * i, q);
  }
  EXPECT_LT(w.maxAbsVelocity(), 0.03);   // would be 0.06 rad/s as a single-step difference
  WindowVelocity m(0.15, 0.5);
  for (int i = 0; i < 12; ++i)
  {
    V q = q0;
    q[0] += 0.1 * 0.033 * i;   // 0.1 rad/s
    m.add(0.033 * i, q);
  }
  EXPECT_NEAR(m.maxAbsVelocity(), 0.1, 0.01);
  WindowVelocity few(0.15, 0.5);
  few.add(0.0, q0);
  few.add(0.05, q0);
  EXPECT_TRUE(std::isinf(few.maxAbsVelocity()));   // span below min_span
}

TEST(CriLive, ChordStateIsGeometricNotTimeIndexed)
{
  const V s{0, 0, 0, 0, 0, 0}, g{1.0, 0.5, 0, 0, 0, 0};
  EXPECT_NEAR(chordState(s, s, g).deviation_rad, 0.0, 1e-12);
  EXPECT_NEAR(chordState(g, s, g).deviation_rad, 0.0, 1e-12);
  const ChordState mid = chordState(V{0.5, 0.25, 0, 0, 0, 0}, s, g);
  EXPECT_NEAR(mid.progress, 0.5, 1e-12);
  EXPECT_NEAR(mid.deviation_rad, 0.0, 1e-12);
  // 15 mrad sideways on joint 6 at mid chord
  const ChordState off = chordState(V{0.5, 0.25, 0, 0, 0, 0.015}, s, g);
  EXPECT_NEAR(off.deviation_rad, 0.015, 1e-12);
  // overshoot past the goal is a deviation from the clamped end point
  EXPECT_GT(chordState(V{1.05, 0.525, 0, 0, 0, 0}, s, g).deviation_rad, 0.04);
  // before the start
  EXPECT_GT(chordState(V{-0.05, -0.025, 0, 0, 0, 0}, s, g).deviation_rad, 0.04);
  // fail closed
  EXPECT_TRUE(std::isinf(chordState(V{0.5}, s, g).deviation_rad));
  EXPECT_TRUE(std::isinf(chordState(V{0.5, std::nan(""), 0, 0, 0, 0}, s, g).deviation_rad));
}

// ---- hardware, cap, gates ---------------------------------------------------------------------

namespace
{
CriHwView goodHw()
{
  CriHwView h;
  h.have = true;
  h.age_s = 0.05;
  h.connected = true;
  h.status_fresh = true;
  h.latched = false;
  h.move_stop_pending = false;
  h.move_active = false;
  h.activated = true;
  return h;
}
}  // namespace

TEST(CriHardware, DefaultViewFailsClosed)
{
  EXPECT_FALSE(evaluateCriHardware(CriHwView{}, 0.5, true));
  EXPECT_FALSE(evaluateCriHardware(CriHwView{}, 0.5, false));
}

TEST(CriHardware, EachConditionAlone)
{
  EXPECT_TRUE(evaluateCriHardware(goodHw(), 0.5, true));
  auto expectFail = [](CriHwView h, const char * what, bool idle = true)
  {
    std::vector<std::string> f;
    EXPECT_FALSE(evaluateCriHardware(h, 0.5, idle, &f)) << what;
    EXPECT_EQ(f.size(), 1u) << what;
  };
  CriHwView h = goodHw();
  h.have = false;
  expectFail(h, "missing");
  h = goodHw(); h.age_s = 0.6; expectFail(h, "stale");
  h = goodHw(); h.connected = false; expectFail(h, "disconnected");
  h = goodHw(); h.status_fresh = false; expectFail(h, "status stale");
  h = goodHw(); h.latched = true; expectFail(h, "latched");
  h = goodHw(); h.move_stop_pending = true; expectFail(h, "stop pending");
  h = goodHw(); h.activated = false; expectFail(h, "not activated");
  h = goodHw(); h.move_active = true; expectFail(h, "move active");
  // during a run move_active is expected
  h = goodHw(); h.move_active = true;
  EXPECT_TRUE(evaluateCriHardware(h, 0.5, false));
  // ... but never latched, whatever else is true
  h.latched = true;
  EXPECT_FALSE(evaluateCriHardware(h, 0.5, false));
}

TEST(CriHardware, CapDecision)
{
  CriParams p;
  CriHwView h = goodHw();
  std::string why;
  EXPECT_FALSE(criCapOk(h, p, &why));   // unknown cap, no assumption: refuse
  EXPECT_NE(why.find("does not report"), std::string::npos) << why;
  p.assume_driver_cap_ok = true;
  EXPECT_TRUE(criCapOk(h, p));
  h.cap_known = true;
  h.cap_velocity_percent = 25.0;
  h.cap_acceleration_percent = 40.0;
  p.assume_driver_cap_ok = false;   // a known cap decides, the assumption is not needed
  EXPECT_TRUE(criCapOk(h, p));
  h.cap_velocity_percent = 50.0;
  EXPECT_FALSE(criCapOk(h, p, &why));   // the driver allows more than the 25 % this coordinator is built for
  h.cap_velocity_percent = 15.0;
  EXPECT_FALSE(criCapOk(h, p, &why));   // 20 % requested > cap 15 %
  EXPECT_NE(why.find("exceeds"), std::string::npos) << why;
  h.cap_velocity_percent = 25.0;
  h.cap_acceleration_percent = 20.0;
  EXPECT_FALSE(criCapOk(h, p, &why));   // 30 % acc requested > 20 %
  h.cap_acceleration_percent = std::nan("");
  EXPECT_FALSE(criCapOk(h, p));
  p.assume_driver_cap_ok = true;
  EXPECT_FALSE(criCapOk(h, p));   // a reported but invalid cap is never overridden by the assumption
}

namespace
{
CriPreDispatchInputs goodPre()
{
  CriPreDispatchInputs in;
  in.hw = goodHw();
  in.live.have = true;
  in.live.age_s = 0.03;
  in.live.q = V{0.1, 0, 0, 0, 0, 0};
  in.live.rest_s = 1.0;
  in.chord_start = V{0.1, 0, 0, 0, 0, 0};
  in.start_tolerance_rad = 0.01;
  in.stop_latched = false;
  in.controller_active = true;
  in.cap_ok = true;
  return in;
}
}  // namespace

TEST(CriPreDispatch, PassesWhenEverythingHolds)
{
  EXPECT_TRUE(evaluateCriPreDispatch(goodPre(), CriPreDispatchLimits{}));
}

TEST(CriPreDispatch, EachGateAlone)
{
  const CriPreDispatchLimits lim;
  auto expectFail = [&](CriPreDispatchInputs in, const char * what)
  {
    std::vector<std::string> f;
    EXPECT_FALSE(evaluateCriPreDispatch(in, lim, &f)) << what;
    EXPECT_EQ(f.size(), 1u) << what << ": " << (f.empty() ? "" : f[0]);
  };
  CriPreDispatchInputs in = goodPre();
  in.live.age_s = 0.5; expectFail(in, "live stale");
  in = goodPre(); in.live.have = false; expectFail(in, "no live");
  in = goodPre(); in.live.rest_s = 0.1; expectFail(in, "not at rest");
  in = goodPre(); in.live.q[0] = 0.12; expectFail(in, "start deviation");
  in = goodPre(); in.stop_latched = true; expectFail(in, "stop latched");
  in = goodPre(); in.cap_ok = false; expectFail(in, "cap");
  in = goodPre(); in.controller_active = false; expectFail(in, "controller");
  in = goodPre(); in.hw.latched = true; expectFail(in, "latched");
  in = goodPre(); in.hw.move_active = true; expectFail(in, "move_active");
  in = goodPre(); in.hw.move_stop_pending = true; expectFail(in, "stop pending");
  in = goodPre(); in.hw.age_s = 0.5; expectFail(in, "hw stale");
  in = goodPre(); in.hw.activated = false; expectFail(in, "not activated");
  in = goodPre(); in.hw.status_fresh = false; expectFail(in, "status not fresh");
  in = goodPre(); in.hw.have = false; expectFail(in, "no hardware_state");
}

TEST(CriPreDispatch, ArmGateNeedsHardwareAndLive)
{
  CriLiveView live;
  live.have = true;
  live.age_s = 0.02;
  live.q = V(6, 0.0);
  const CriArmGateLimits lim;
  EXPECT_TRUE(evaluateCriArmGate(goodHw(), live, lim));
  EXPECT_FALSE(evaluateCriArmGate(CriHwView{}, live, lim));
  EXPECT_FALSE(evaluateCriArmGate(goodHw(), CriLiveView{}, lim));
  CriLiveView stale = live;
  stale.age_s = 0.3;
  EXPECT_FALSE(evaluateCriArmGate(goodHw(), stale, lim));
  CriHwView lat = goodHw();
  lat.latched = true;
  EXPECT_FALSE(evaluateCriArmGate(lat, live, lim));
}

namespace
{
CriRunMonitorInputs goodRun()
{
  CriRunMonitorInputs in;
  in.hw = goodHw();
  in.hw.move_active = true;
  in.live.have = true;
  in.live.age_s = 0.03;
  in.start = V{0, 0, 0, 0, 0, 0};
  in.goal = V{1.0, 0, 0, 0, 0, 0};
  in.live.q = V{0.4, 0.002, 0, 0, 0, 0};
  in.path_tolerance_rad = 0.01;
  in.elapsed_s = 3.0;
  in.timeout_s = 30.0;
  return in;
}
}  // namespace

TEST(CriRunMonitor, HealthyRunPasses)
{
  EXPECT_FALSE(evaluateCriRunMonitor(goodRun()).stop);
}

TEST(CriRunMonitor, StopConditions)
{
  auto expectStop = [](CriRunMonitorInputs in, const char * text)
  {
    const MonitorDecision d = evaluateCriRunMonitor(in);
    EXPECT_TRUE(d.stop) << text;
    EXPECT_NE(d.reason.find(text), std::string::npos) << d.reason;
  };
  CriRunMonitorInputs in = goodRun();
  in.hw.latched = true; expectStop(in, "LATCHED");
  in = goodRun(); in.hw.age_s = 0.9; expectStop(in, "hardware_state");
  in = goodRun(); in.hw.have = false; expectStop(in, "hardware_state");
  in = goodRun(); in.hw.activated = false; expectStop(in, "activated");
  in = goodRun(); in.live.age_s = 0.6; expectStop(in, "stale");
  in = goodRun(); in.live.have = false; expectStop(in, "no robot_joint_states");
  in = goodRun(); in.live.q[1] = 0.012; expectStop(in, "off the chord");      // 12 mrad > 10 mrad
  in = goodRun(); in.live.q[0] = 1.2; expectStop(in, "off the chord");        // ran past the goal
  in = goodRun(); in.elapsed_s = 31.0; expectStop(in, "no result");
  in = goodRun(); in.hw.move_stop_pending = true; expectStop(in, "Move Stop");
  // the timeout only applies while the move is outstanding
  in = goodRun(); in.elapsed_s = 31.0; in.result_received = true;
  EXPECT_FALSE(evaluateCriRunMonitor(in).stop);
  // a hardware latch is a stop even after the result
  in.hw.latched = true;
  EXPECT_TRUE(evaluateCriRunMonitor(in).stop);
}

// ---- controller gate ---------------------------------------------------------------------------

TEST(CriControllerGate, ExpectedStatePerMode)
{
  // fresh poll (sent after the mode began)
  EXPECT_TRUE(controllerStateAllowed(CtrlGateMode::Normal, "active", true));
  EXPECT_FALSE(controllerStateAllowed(CtrlGateMode::Normal, "inactive", true));
  EXPECT_FALSE(controllerStateAllowed(CtrlGateMode::Normal, "unconfigured", true));
  EXPECT_FALSE(controllerStateAllowed(CtrlGateMode::Normal, "", true));
  EXPECT_TRUE(controllerStateAllowed(CtrlGateMode::CriRun, "inactive", true));
  EXPECT_FALSE(controllerStateAllowed(CtrlGateMode::CriRun, "active", true));   // somebody re-activated the JTC during the move: STOP
  EXPECT_FALSE(controllerStateAllowed(CtrlGateMode::CriRun, "finalized", true));
  EXPECT_TRUE(controllerStateAllowed(CtrlGateMode::Switching, "active", true));
  EXPECT_TRUE(controllerStateAllowed(CtrlGateMode::Switching, "inactive", true));
  EXPECT_FALSE(controllerStateAllowed(CtrlGateMode::Switching, "unconfigured", true));
  // an answer to a poll sent BEFORE the mode began may still show the previous state: both accepted
  EXPECT_TRUE(controllerStateAllowed(CtrlGateMode::CriRun, "active", false));
  EXPECT_TRUE(controllerStateAllowed(CtrlGateMode::Normal, "inactive", false));
  EXPECT_FALSE(controllerStateAllowed(CtrlGateMode::CriRun, "unconfigured", false));
}

// ---- MoveJoint result ---------------------------------------------------------------------------

namespace
{
MoveJointResultView goodResult(const V & goal)
{
  MoveJointResultView r;
  r.outcome = ActionOutcome::Succeeded;
  r.error_code = kMjSuccess;
  r.end_reason = "PLAN";
  r.stop_sent = false;
  r.stop_confirmed = false;
  r.hardware_latched = false;
  r.live_valid = true;
  r.live_q = goal;
  return r;
}
}  // namespace

TEST(CriResult, SuccessOnlyWhenEverythingIsRight)
{
  const V goal{1.0, 0.5, 0, 0, 0, 0};
  EXPECT_TRUE(classifyMoveJointResult(goodResult(goal), goal, 0.005).success);
  MoveJointResultView r = goodResult(goal);
  r.live_q[0] += 0.004;
  EXPECT_TRUE(classifyMoveJointResult(r, goal, 0.005).success);
  r.live_q[0] += 0.003;   // 7 mrad from the goal
  EXPECT_FALSE(classifyMoveJointResult(r, goal, 0.005).success);
}

TEST(CriResult, RobotErrorIsAFaultWhateverTheErrorCodeSays)
{
  const V goal{1.0, 0.5, 0, 0, 0, 0};
  MoveJointResultView r = goodResult(goal);
  r.end_reason = "ERROR";
  r.error_code = kMjCanceled;     // the driver reports CANCELED ...
  r.outcome = ActionOutcome::Canceled;
  r.stop_sent = true;
  r.stop_confirmed = true;        // ... and the stop is confirmed: still a FAULT (the robot reported an error)
  const CriVerdict v = classifyMoveJointResult(r, goal, 0.005);
  EXPECT_FALSE(v.success);
  EXPECT_NE(v.reason.find("robot ERROR"), std::string::npos) << v.reason;
  // even an action that "succeeded" with end_reason ERROR and error_code SUCCESS
  r = goodResult(goal);
  r.end_reason = "ERROR";
  EXPECT_FALSE(classifyMoveJointResult(r, goal, 0.005).success);
  // ERROR with CANCELED code and a live state exactly at the goal
  r = goodResult(goal);
  r.end_reason = "ERROR";
  r.error_code = kMjCanceled;
  EXPECT_FALSE(classifyMoveJointResult(r, goal, 0.005).success);
}

TEST(CriResult, LatchedIsAFaultAnywhere)
{
  const V goal{1.0, 0.5, 0, 0, 0, 0};
  MoveJointResultView r = goodResult(goal);
  r.hardware_latched = true;
  const CriVerdict v = classifyMoveJointResult(r, goal, 0.005);
  EXPECT_FALSE(v.success);
  EXPECT_NE(v.reason.find("latched"), std::string::npos) << v.reason;
  // latched beats a perfect result of every other kind
  r.error_code = kMjSuccess;
  r.end_reason = "PLAN";
  EXPECT_FALSE(classifyMoveJointResult(r, goal, 0.005).success);
}

TEST(CriResult, EveryOtherOutcomeIsAFault)
{
  const V goal{1.0, 0.5, 0, 0, 0, 0};
  for (uint8_t code : {kMjRejected, kMjControllerActive, kMjStartDeviation, kMjCmdError, kMjAckTimeout, kMjMotionTimeout, kMjPathDeviation, kMjStatusGap,
                       kMjLatched, kMjFault, kMjCanceled, kMjRobotEnded, kMjGoalDeviation})
  {
    MoveJointResultView r = goodResult(goal);
    r.error_code = code;
    EXPECT_FALSE(classifyMoveJointResult(r, goal, 0.005).success) << static_cast<int>(code);
  }
  for (ActionOutcome o : {ActionOutcome::Aborted, ActionOutcome::Canceled, ActionOutcome::Unknown})
  {
    MoveJointResultView r = goodResult(goal);
    r.outcome = o;
    EXPECT_FALSE(classifyMoveJointResult(r, goal, 0.005).success);
  }
  MoveJointResultView r = goodResult(goal);
  r.end_reason = "USER";
  EXPECT_FALSE(classifyMoveJointResult(r, goal, 0.005).success);
  r = goodResult(goal);
  r.end_reason.clear();
  EXPECT_FALSE(classifyMoveJointResult(r, goal, 0.005).success);
  r = goodResult(goal);
  r.live_valid = false;
  EXPECT_FALSE(classifyMoveJointResult(r, goal, 0.005).success);
}

// ---- doStop ---------------------------------------------------------------------------------------

TEST(CriStop, MoveJointCancelsComeFirstThenHoldsThenFjtCancels)
{
  const auto steps = planStopSteps(true, {true, true});
  using S = StopStep;
  using A = StopAction;
  ASSERT_EQ(steps.size(), 6u);
  EXPECT_EQ(steps, (std::vector<S>{{A::CancelMoveJoint, 0}, {A::CancelMoveJoint, 1}, {A::HoldJtc, 0}, {A::HoldJtc, 1}, {A::CancelFjt, 0}, {A::CancelFjt, 1}}));
  // every MoveJoint cancel precedes every other step
  size_t last_mj = 0, first_other = steps.size();
  for (size_t i = 0; i < steps.size(); ++i)
  {
    if (steps[i].action == A::CancelMoveJoint) last_mj = i;
    else first_other = std::min(first_other, i);
  }
  EXPECT_LT(last_mj, first_other);
}

TEST(CriStop, NoJointStateDependentHoldForAnArmInACriRun)
{
  const auto steps = planStopSteps(true, {false, true});   // left JTC is not in Normal mode
  using S = StopStep;
  using A = StopAction;
  EXPECT_EQ(steps, (std::vector<S>{{A::CancelMoveJoint, 0}, {A::CancelMoveJoint, 1}, {A::HoldJtc, 1}, {A::CancelFjt, 0}, {A::CancelFjt, 1}}));
  for (const S & s : steps) EXPECT_FALSE(s.action == A::HoldJtc && s.arm == 0);
  EXPECT_EQ(planStopSteps(true, {false, false}),
            (std::vector<S>{{A::CancelMoveJoint, 0}, {A::CancelMoveJoint, 1}, {A::CancelFjt, 0}, {A::CancelFjt, 1}}));
}

TEST(CriStop, WithoutCriItIsTodaysOrder)
{
  using S = StopStep;
  using A = StopAction;
  const auto steps = planStopSteps(false, {true, true});
  EXPECT_EQ(steps, (std::vector<S>{{A::HoldJtc, 0}, {A::HoldJtc, 1}, {A::CancelFjt, 0}, {A::CancelFjt, 1}}));
  for (const S & s : steps) EXPECT_NE(s.action, A::CancelMoveJoint);
}

// ---- barrier with the arm gate --------------------------------------------------------------------

TEST(CriBarrier, ArmHealthIsOnlyRequiredWhenAsked)
{
  BarrierInputs in;
  in.joint_state_age_s = 0.01;
  in.max_joint_error_to_start_rad = 0.0;
  in.rest_duration_s = 1.0;
  in.controllers_active = true;
  in.hardware_active = true;
  in.snapshot_age_s = 1.0;
  in.stop_latched = false;
  in.monitor_age_s = 0.05;
  in.monitor_healthy = true;
  const BarrierLimits lim;
  EXPECT_TRUE(evaluateBarriers(in, lim));
  in.arm_health_ok = false;                      // not required: ignored (CRI off behaves exactly as before)
  EXPECT_TRUE(evaluateBarriers(in, lim));
  in.arm_health_required = true;
  in.arm_health_text = "left: driver is LATCHED";
  std::vector<std::string> f;
  EXPECT_FALSE(evaluateBarriers(in, lim, &f));
  ASSERT_EQ(f.size(), 1u);
  EXPECT_NE(f[0].find("LATCHED"), std::string::npos);
  in.arm_health_ok = true;
  EXPECT_TRUE(evaluateBarriers(in, lim));
}

// ---- the padded certificate on the synthetic cell ---------------------------------------------

namespace
{
class CriCertificate : public ::testing::Test
{
protected:
  void SetUp() override
  {
    model = dual_cut_test::syntheticModel();
    ASSERT_NE(model, nullptr);
    scene = std::make_shared<planning_scene::PlanningScene>(model);
    base.tip_link_suffix = "tool";
    base.swept.padding_m = 0.05;
    base.contact_links_left = {"arm_left_tool"};
    base.contact_links_right = {"arm_right_tool"};
    base.table_link_name = "table";
    base.table_min_clearance_m = 0.01;
    base.table_exempt_links_left = {"arm_left_base_link"};
    base.table_exempt_links_right = {"arm_right_base_link"};
    base.sample.dt_s = 0.05;
    base.sample.max_joint_step_rad = 0.01;
  }
  // table = false: the table gate is off (so a test of another gate is not decided by the 8 cm the synthetic arm hovers above it)
  std::unique_ptr<PhaseValidator> make(double extra, bool table = true)
  {
    PhaseValidatorConfig cfg = base;
    cfg.swept.padding_m += extra;
    cfg.table_min_clearance_m = table ? cfg.table_min_clearance_m + extra : 0.0;
    cfg.world_object_clearance_m = extra;
    std::unique_ptr<PhaseValidator> v;
    const Status s = PhaseValidator::create(scene, cfg, v);
    EXPECT_TRUE(s) << s.message;
    return v;
  }
  const std::vector<std::string> names{"joint1", "joint2"};
  moveit::core::RobotModelPtr model;
  std::shared_ptr<planning_scene::PlanningScene> scene;
  PhaseValidatorConfig base;
};
}  // namespace

TEST_F(CriCertificate, ClearChordPassesAndReportsClearances)
{
  auto v = make(0.0);
  ASSERT_NE(v, nullptr);
  // left swings away from the parked right arm (to -y); right parked at home
  const ChordCheck c = v->checkChord("left", names, {0, 0}, {-0.8, 0.4}, names, {0, 0}, 0.01);
  EXPECT_TRUE(c.ok) << c.reason;
  EXPECT_GT(c.samples, 80u);
  EXPECT_TRUE(c.cross.safe());
  EXPECT_NEAR(c.cross_required_m, 0.05, 1e-12);
  EXPECT_GT(c.cross_effective_min_m, 0.05);
  ASSERT_TRUE(std::isfinite(c.table_min_m));
  EXPECT_GT(c.table_min_m, 0.01);
  EXPECT_NEAR(c.table_required_m, 0.01, 1e-12);
  // the right arm too (swings to +y, left parked)
  EXPECT_TRUE(v->checkChord("right", names, {0, 0}, {0.8, -0.4}, names, {0, 0}, 0.01).ok);
}

TEST_F(CriCertificate, ExtraPaddingTurnsMarginalChordsIntoRefusals)
{
  // left swings toward the parked right arm: find the farthest angle that still passes with the plain 0.05 m padding
  auto tight = make(0.0, false);
  double goal = 0.0;
  ChordCheck c0;
  for (double a = 0.1; a <= 1.2; a += 0.025)
  {
    const ChordCheck c = tight->checkChord("left", names, {0, 0}, {a, 0.0}, names, {0, 0}, 0.01);
    if (!c.ok) break;
    goal = a;
    c0 = c;
  }
  ASSERT_GT(goal, 0.3) << "the scan found no chord that passes";
  ASSERT_TRUE(std::isfinite(c0.cross_effective_min_m));
  const double margin = c0.cross_effective_min_m - 0.05;
  ASSERT_GT(margin, 0.0);
  ASSERT_LT(margin, 0.3) << "the scan never got close to the parked arm";
  // padding just below the margin still passes, just above it fails (the padding is exactly what is added to the 0.05)
  auto below = make(margin * 0.8, false);
  EXPECT_TRUE(below->checkChord("left", names, {0, 0}, {goal, 0.0}, names, {0, 0}, 0.01).ok);
  auto above = make(margin * 1.25 + 0.001, false);
  const ChordCheck c1 = above->checkChord("left", names, {0, 0}, {goal, 0.0}, names, {0, 0}, 0.01);
  EXPECT_FALSE(c1.ok);
  EXPECT_NE(c1.reason.find("cross-arm"), std::string::npos) << c1.reason;
  EXPECT_NEAR(c1.cross_required_m, 0.05 + margin * 1.25 + 0.001, 1e-12);
}

TEST_F(CriCertificate, ParkedArmPositionDecides)
{
  auto v = make(0.0);
  // the right arm parked reaching toward the left chord: refused; parked home: fine
  EXPECT_TRUE(v->checkChord("left", names, {0, 0}, {0.5, 0.0}, names, {0, 0}, 0.01).ok);
  const ChordCheck blocked = v->checkChord("left", names, {0, 0}, {1.2, 0.0}, names, {-1.2, 0.0}, 0.01);
  EXPECT_FALSE(blocked.ok);
}

TEST_F(CriCertificate, ObjectClearanceUsesTheExactDistance)
{
  scene->getWorldNonConst()->addToObject(
    "detected_carton", std::make_shared<shapes::Box>(0.1, 0.1, 0.1), Eigen::Isometry3d(Eigen::Translation3d(0.2, -0.2, 0.1)));   // y in [-0.25, -0.15]
  auto plain = make(0.0, false);
  // joint1 0 -> 0.2 rotates link1 away from the box (to +y); closest approach is at the start: link1 face y=-0.02 to the box face y=-0.15
  const ChordCheck c = plain->checkChord("left", names, {0, 0}, {0.2, 0.0}, names, {0, 0}, 0.01);
  EXPECT_TRUE(c.ok) << c.reason;
  EXPECT_FALSE(c.world.object_measured);   // gate off: nothing measured, nothing changed
  auto pad = make(0.1, false);
  const ChordCheck ok = pad->checkChord("left", names, {0, 0}, {0.2, 0.0}, names, {0, 0}, 0.01);
  EXPECT_TRUE(ok.ok) << ok.reason;
  ASSERT_TRUE(ok.world.object_measured);
  EXPECT_NEAR(ok.object_min_m, 0.13, 0.01);
  EXPECT_NEAR(ok.object_required_m, 0.1, 1e-12);
  auto big = make(0.2, false);
  const ChordCheck bad = big->checkChord("left", names, {0, 0}, {0.2, 0.0}, names, {0, 0}, 0.01);
  EXPECT_FALSE(bad.ok);
  EXPECT_EQ(bad.world.code, WorldCode::ObjectClearance);
  EXPECT_NE(bad.reason.find("object clearance"), std::string::npos) << bad.reason;
  EXPECT_NE(bad.world.object.find("detected_carton"), std::string::npos) << bad.world.object;
  // a chord INTO the box is a plain collision whatever the padding
  const ChordCheck hit = plain->checkChord("left", names, {0, 0}, {-0.8, 0.0}, names, {0, 0}, 0.01);
  EXPECT_FALSE(hit.ok);
  EXPECT_EQ(hit.world.code, WorldCode::Collision);
}

TEST_F(CriCertificate, TableClearanceGrowsWithThePadding)
{
  auto v = make(0.0);
  const ChordCheck c = v->checkChord("left", names, {0, 0}, {0.5, 0.0}, names, {0, 0}, 0.01);
  ASSERT_TRUE(c.ok) << c.reason;
  ASSERT_TRUE(std::isfinite(c.table_min_m));
  EXPECT_NEAR(c.table_min_m, 0.071, 0.005);   // tool sphere lowest point z = 0.07, table top z = -0.001
  auto pad = make(0.1);
  const ChordCheck bad = pad->checkChord("left", names, {0, 0}, {0.5, 0.0}, names, {0, 0}, 0.01);
  EXPECT_FALSE(bad.ok);
  EXPECT_EQ(bad.world.code, WorldCode::TableClearance);
}

TEST_F(CriCertificate, BadInputsFailClosed)
{
  auto v = make(0.0);
  EXPECT_FALSE(v->checkChord("middle", names, {0, 0}, {0.1, 0}, names, {0, 0}, 0.01).ok);
  EXPECT_FALSE(v->checkChord("left", {"joint1"}, {0}, {0.1}, names, {0, 0}, 0.01).ok);               // a joint missing
  EXPECT_FALSE(v->checkChord("left", names, {0, 0}, {0.1, std::nan("")}, names, {0, 0}, 0.01).ok);   // NaN goal
  EXPECT_FALSE(v->checkChord("left", names, {0, 0}, {0.1, 0.0}, names, {0}, 0.01).ok);               // wrong parked size
}

// ---- JTC sub-chords, partial legs, executed phase --------------------------------------------------------

TEST(CriSubChord, RestToRestQuinticOnTheChordWithinTheLimits)
{
  const V from{0.1, 0.0, -0.2, 0.0, 0.3, 0.0}, to{0.5, 0.2, -0.2, 0.1, 0.3, -0.1};   // leading joint 0.4 rad
  const double vmax = 0.20 * 0.785, amax = 0.10 * 5.24;
  const PlannedSegmentData seg = makeChordSegment(k6, from, to, vmax, amax, "head");
  ASSERT_EQ(seg.trajectory.points.size(), 2u);
  EXPECT_EQ(seg.kind, SegmentKind::Transit);
  EXPECT_TRUE(validate(seg.trajectory));
  const double d = 0.4;
  const double T = std::max(1.875 * d / vmax, std::sqrt(5.7735 * d / amax));
  EXPECT_NEAR(seg.trajectory.duration(), T, 1e-9);
  double max_v = 0.0, max_a = 0.0, max_off = 0.0;
  for (int k = 0; k <= 200; ++k)
  {
    TrajPoint p;
    ASSERT_TRUE(sampleAt(seg.trajectory, T * k / 200.0, p));
    max_v = std::max(max_v, std::abs(p.velocities[0]));
    max_a = std::max(max_a, std::abs(p.accelerations[0]));
    const double s = (p.positions[0] - from[0]) / (to[0] - from[0]);   // progress by the leading joint
    for (size_t j = 0; j < 6; ++j) max_off = std::max(max_off, std::abs(p.positions[j] - (from[j] + s * (to[j] - from[j]))));
  }
  EXPECT_LE(max_v, vmax + 1e-6);
  EXPECT_LE(max_a, amax + 1e-6);
  EXPECT_GT(max_v, 0.5 * vmax);   // not absurdly slow either
  EXPECT_LT(max_off, 1e-9);
  TrajPoint end;
  ASSERT_TRUE(sampleAt(seg.trajectory, T, end));
  for (size_t j = 0; j < 6; ++j)
  {
    EXPECT_NEAR(end.positions[j], to[j], 1e-12);
    EXPECT_EQ(end.velocities[j], 0.0);
  }
  // a very short piece takes at least 0.2 s; bad input gives an empty trajectory
  EXPECT_GE(makeChordSegment(k6, from, V{0.1005, 0, -0.2, 0, 0.3, 0}, vmax, amax, "x").trajectory.duration(), 0.2);
  EXPECT_TRUE(makeChordSegment(k6, from, from, vmax, amax, "x").trajectory.points.empty());
  EXPECT_TRUE(makeChordSegment(k6, from, V{1, 1, 1, 1, 1, std::nan("")}, vmax, amax, "x").trajectory.points.empty());
  EXPECT_TRUE(makeChordSegment(k6, from, to, 0.0, amax, "x").trajectory.points.empty());
}

TEST(CriRuns, PartialLegIsExecutedAsHeadCriTail)
{
  ArmPhase ph = threeSegments();   // Align | Transit | Cut
  CriArmPhasePlan ap;
  CriLegRecord l;
  l.geometry = checkTransitGeometry(ph, 1, limits());
  ASSERT_TRUE(l.geometry.eligible);
  l.certified = true;
  l.partial = true;
  l.cri_start = V{0.3, 0.1, -0.05, 0.025, 0.0, 0.075};
  l.cri_goal = V{0.7, 0.3, -0.15, 0.075, 0.0, 0.225};
  l.cri_length_rad = 0.4;
  l.head = makeChordSegment(k6, l.geometry.start, l.cri_start, 0.157, 0.524, "head");
  l.tail = makeChordSegment(k6, l.cri_goal, l.geometry.goal, 0.157, 0.524, "tail");
  ap.legs = {l};
  selectCriLegs(ap, 1);
  ASSERT_EQ(ap.selected_segments, (std::vector<size_t>{1}));
  const CriArmRuns r = buildCriArmRuns(true, true, &ap, ph);
  using T = CriRunType;
  ASSERT_EQ(r.phase.segments.size(), 5u);   // align, head, [leg], tail, cut
  EXPECT_EQ(r.phase.segments[1].stage_name, "head");
  EXPECT_EQ(r.phase.segments[3].stage_name, "tail");
  EXPECT_EQ(r.runs, (std::vector<CriRun>{{T::Jtc, 0, 2}, {T::Cri, 2, 3}, {T::Jtc, 3, 5}}));
  ASSERT_EQ(r.leg.size(), 3u);
  EXPECT_EQ(r.leg[0], nullptr);
  ASSERT_NE(r.leg[1], nullptr);
  EXPECT_EQ(r.leg[1]->cri_goal, l.cri_goal);
  EXPECT_EQ(r.leg[2], nullptr);
  // the pieces chain: head ends where the leg starts, the tail starts where it ends
  EXPECT_EQ(r.phase.segments[1].trajectory.points.back().positions, l.cri_start);
  EXPECT_EQ(r.phase.segments[3].trajectory.points.front().positions, l.cri_goal);
  // no head (the stretch starts at the chord start)
  CriArmPhasePlan ap2 = ap;
  ap2.legs[0].head = PlannedSegmentData{};
  const CriArmRuns r2 = buildCriArmRuns(true, true, &ap2, ph);
  EXPECT_EQ(r2.runs, (std::vector<CriRun>{{T::Jtc, 0, 1}, {T::Cri, 1, 2}, {T::Jtc, 2, 4}}));
  // CRI off / cap not ok: the arm phase untouched, one JTC run
  for (bool on : {false, true})
  {
    const CriArmRuns off = buildCriArmRuns(on, false, &ap, ph);
    EXPECT_EQ(off.phase.segments.size(), ph.segments.size());
    EXPECT_EQ(off.runs, (std::vector<CriRun>{{T::Jtc, 0, 3}}));
  }
  EXPECT_EQ(buildCriArmRuns(false, true, &ap, ph).runs, (std::vector<CriRun>{{T::Jtc, 0, 3}}));
  EXPECT_EQ(buildCriArmRuns(true, true, nullptr, ph).runs, (std::vector<CriRun>{{T::Jtc, 0, 3}}));
}

TEST(CriSelection, LongestCriStretchWins)
{
  CriArmPhasePlan ap;
  ap.legs = {leg(1, 0.9, true), leg(3, 1.2, true)};
  ap.legs[0].cri_length_rad = 0.9;
  ap.legs[1].cri_length_rad = 0.3;   // a long leg of which only a short stretch passes
  selectCriLegs(ap, 1);
  EXPECT_EQ(ap.selected_segments, (std::vector<size_t>{1}));
}

TEST_F(CriCertificate, PartialChordKeepsTheMiddleStretchThatPassesThePadding)
{
  // a carton at the left arm's start side and the parked right arm near the far end of the swing: with 0.14 m of padding
  // neither end passes, the middle does
  scene->getWorldNonConst()->addToObject(
    "detected_carton", std::make_shared<shapes::Box>(0.1, 0.1, 0.1), Eigen::Isometry3d(Eigen::Translation3d(0.2, -0.2, 0.1)));
  auto pad = make(0.14, false);
  const V home{0, 0};
  const ChordCheck whole = pad->checkChord("left", names, {0, 0}, {0.9, 0.0}, names, home, 0.01);
  ASSERT_FALSE(whole.ok) << "the full chord must fail for this test";
  const ChordStretch st = pad->certifyChord("left", names, {0, 0}, {0.9, 0.0}, names, home, 0.01, 0.05);
  ASSERT_TRUE(st.found) << st.reason;
  EXPECT_FALSE(st.full);
  EXPECT_GT(st.s0, 0.0);
  EXPECT_LT(st.s1, 1.0);
  EXPECT_GE(st.length_rad, 0.05);
  EXPECT_NEAR(st.start[0], st.s0 * 0.9, 1e-9);
  EXPECT_NEAR(st.goal[0], st.s1 * 0.9, 1e-9);
  EXPECT_TRUE(st.check.ok) << st.check.reason;   // the final check is of exactly this stretch, with the padding
  EXPECT_NEAR(st.check.cross_required_m, 0.05 + 0.14, 1e-12);
  EXPECT_GE(st.check.object_min_m, 0.14 - 1e-9);
  // the stretch re-checked on its own passes, and one more sample at either end would not be the maximal stretch
  EXPECT_TRUE(pad->checkChord("left", names, st.start, st.goal, names, home, 0.01).ok);
  // a minimum length the stretch cannot reach: nothing is found, with the reason
  const ChordStretch none = pad->certifyChord("left", names, {0, 0}, {0.9, 0.0}, names, home, 0.01, 0.8);
  EXPECT_FALSE(none.found);
  EXPECT_NE(none.reason.find("less than the minimum"), std::string::npos) << none.reason;
}

TEST_F(CriCertificate, FullChordIsReportedAsFullAndBadInputsNeverCertify)
{
  auto v = make(0.0, false);
  const ChordStretch st = v->certifyChord("left", names, {0, 0}, {-0.8, 0.4}, names, {0, 0}, 0.01, 0.05);
  ASSERT_TRUE(st.found) << st.reason;
  EXPECT_TRUE(st.full);
  EXPECT_EQ(st.s0, 0.0);
  EXPECT_EQ(st.s1, 1.0);
  // a chord whose every sample fails (inside the parked arm's volume) finds nothing
  EXPECT_FALSE(v->certifyChord("left", names, {1.2, 0.0}, {1.3, 0.0}, names, {-1.2, 0.0}, 0.01, 0.05).found);
  EXPECT_FALSE(v->certifyChord("left", names, {0, 0}, {0.5, std::nan("")}, names, {0, 0}, 0.01, 0.05).found);
  EXPECT_FALSE(v->certifyChord("both", names, {0, 0}, {0.5, 0.0}, names, {0, 0}, 0.01, 0.05).found);
}

// ---- the real dual model: how large the lever arm really is ------------------------------------------

namespace
{
std::string runCommand(const std::string & cmd)
{
  std::string out;
  FILE * p = popen(cmd.c_str(), "r");
  if (!p) return out;
  char buf[4096];
  size_t n;
  while ((n = fread(buf, 1, sizeof(buf), p)) > 0) out.append(buf, n);
  if (pclose(p) != 0) return std::string();
  return out;
}
}  // namespace

TEST(CriRealModel, RigorousReachBoundAndAPaddedChord)
{
  std::string desc, moveit;
  try
  {
    desc = ament_index_cpp::get_package_share_directory("rebel_dual_table_description");
    moveit = ament_index_cpp::get_package_share_directory("rebel_dual_table_moveit_config");
  }
  catch (const std::exception & e)
  {
    GTEST_SKIP() << "real dual model packages not installed: " << e.what();
  }
  const std::string urdf = runCommand("xacro " + desc + "/urdf/rebel_dual_table.urdf.xacro include_ros2_control:=false 2>/dev/null");
  if (urdf.empty()) GTEST_SKIP() << "xacro failed";
  const std::string srdf = dual_cut_test::readFile(moveit + "/config/rebel_dual_table.srdf");
  ASSERT_FALSE(srdf.empty());
  auto model = dual_cut_test::buildModel(urdf, srdf);
  ASSERT_NE(model, nullptr);
  ArmSpec left, right;
  ASSERT_TRUE(makeArmSpec(*model, "left", left));
  ASSERT_TRUE(makeArmSpec(*model, "right", right));
  for (const ArmSpec * a : {&left, &right})
  {
    double sum = 0.0;
    std::vector<double> per;
    const Status s = armReachBound(*model, *a, sum, &per);
    ASSERT_TRUE(s) << s.message;
    ASSERT_EQ(per.size(), 6u);
    std::printf("[real model] %s rigorous reach bound: sum %.3f m, per joint", a->side.c_str(), sum);
    for (double r : per) std::printf(" %.3f", r);
    std::printf("\n");
    EXPECT_GT(sum, 1.0);   // more than the 1.0 m constant: the constant is NOT a rigorous bound for a deviation in every joint
    EXPECT_LT(sum, 10.0);
    for (size_t i = 1; i < per.size(); ++i) EXPECT_LE(per[i], per[i - 1] + 1e-9);   // each later joint carries a subset of the geometry
    const double padding_const = criPaddingM(0.005, 1.0);
    const double padding_rigorous = criPaddingM(0.005, sum);
    std::printf("[real model] %s padding at 5 mrad: constant 1.0 m -> %.4f m, rigorous -> %.4f m\n", a->side.c_str(), padding_const, padding_rigorous);
  }
  // a chord of the real left arm with the constant-reach padding, other arm parked at home
  auto scene = std::make_shared<planning_scene::PlanningScene>(model);
  PhaseValidatorConfig cfg;
  cfg.tip_link_suffix = "xeg32_tool_tip";
  cfg.table_link_name = "table";
  cfg.table_min_clearance_m = 0.010;
  for (const ArmSpec * a : {&left, &right}) (a == &left ? cfg.table_exempt_links_left : cfg.table_exempt_links_right) = {a->prefix + "base_link"};
  const double pad = criPaddingM(0.005, 1.0);
  cfg.swept.padding_m = 0.05 + pad;
  cfg.table_min_clearance_m += pad;
  cfg.world_object_clearance_m = pad;
  std::unique_ptr<PhaseValidator> v;
  ASSERT_TRUE(PhaseValidator::create(scene, cfg, v));
  const std::vector<std::string> names{"joint1", "joint2", "joint3", "joint4", "joint5", "joint6"};
  const std::vector<double> home(6, 0.0);
  const ChordCheck c = v->checkChord("left", names, home, {0.4, 0.3, 0.0, 0.0, 0.0, 0.0}, names, home, 0.01);
  std::printf("[real model] left chord home->(0.4,0.3,..): ok %d, samples %zu, table min %.4f m (required %.4f), cross %.4f m (required %.4f) %s\n",
              c.ok ? 1 : 0, c.samples, c.table_min_m, c.table_required_m, c.cross_effective_min_m, c.cross_required_m, c.reason.c_str());
  EXPECT_GT(c.samples, 30u);
  EXPECT_NEAR(c.table_required_m, 0.010 + pad, 1e-12);
}
