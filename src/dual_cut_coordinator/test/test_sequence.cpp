// Sequence schedule: job joining / delay hold, segment windows, end clamping (tail hold), skew gate.
#include <gtest/gtest.h>

#include <cmath>

#include "dual_cut_coordinator/phase_validator.hpp"
#include "dual_cut_coordinator/sequence_plan.hpp"
#include "dual_cut_coordinator/top_split.hpp"
#include "test_common.hpp"

using namespace dual_cut;

namespace
{

JointTrajectory rest2rest(const std::vector<double> & from, const std::vector<double> & to, double T)
{
  JointTrajectory t;
  t.joint_names = {"joint1", "joint2"};
  const int n = 8;
  for (int k = 0; k <= n; ++k)
  {
    const double s = static_cast<double>(k) / n;
    const double b = 10 * s * s * s - 15 * s * s * s * s + 6 * s * s * s * s * s;
    const double db = (30 * s * s - 60 * s * s * s + 30 * s * s * s * s) / T;
    const double ddb = (60 * s - 180 * s * s + 120 * s * s * s) / (T * T);
    TrajPoint p;
    p.time_s = s * T;
    for (size_t j = 0; j < 2; ++j)
    {
      p.positions.push_back(from[j] + (to[j] - from[j]) * b);
      p.velocities.push_back((to[j] - from[j]) * db);
      p.accelerations.push_back((to[j] - from[j]) * ddb);
    }
    t.points.push_back(p);
  }
  return t;
}

PlannedSegmentData seg(SegmentKind k, const std::string & name, std::vector<double> from, std::vector<double> to, double T)
{
  PlannedSegmentData s;
  s.kind = k;
  s.stage_name = name;
  s.trajectory = rest2rest(from, to, T);
  return s;
}

SnapshotIdentity snap()
{
  SnapshotIdentity s;
  s.snapshot_id = "snap";
  s.capture_id = "cap";
  s.acquisition_stamp_ns = 42;
  return s;
}

ArmPlan plan(const std::string & side, std::vector<PlannedSegmentData> segs)
{
  ArmPlan p;
  p.side = side;
  p.snapshot = snap();
  p.planning_frame = "base_link";
  p.joint_names = {"joint1", "joint2"};
  p.start_q = segs.front().trajectory.points.front().positions;
  p.end_q = segs.back().trajectory.points.back().positions;
  p.segments = std::move(segs);
  return p;
}

ArmPhase phaseOf(const std::string & side, std::vector<PlannedSegmentData> segs)
{
  ArmPhase ph;
  ph.side = side;
  ph.joint_names = {"joint1", "joint2"};
  ph.start_q = segs.front().trajectory.points.front().positions;
  ph.end_q = segs.back().trajectory.points.back().positions;
  ph.segments = std::move(segs);
  return ph;
}

}  // namespace

TEST(SequencePlan, DelayHoldGoesJustBeforeEntry)
{
  const ArmPlan j1 = plan("left", {seg(SegmentKind::Transit, "a", {0, 0}, {1, 0.5}, 3.0), seg(SegmentKind::Cut, "b", {1, 0.5}, {1.2, 0.5}, 2.0)});
  const ArmPlan j2 = plan("left", {seg(SegmentKind::Transit, "c", {1.2, 0.5}, {0.2, 0.1}, 4.0), seg(SegmentKind::Dwell, "d", {0.2, 0.1}, {0.2, 0.1}, 1.0),
                                   seg(SegmentKind::Entry, "e", {0.2, 0.1}, {0.2, 0.2}, 0.5), seg(SegmentKind::Cut, "f", {0.2, 0.2}, {0.0, 0.2}, 3.0)});
  ArmPhase none, held;
  double d1 = 0;
  ASSERT_TRUE(joinJobs(j1, j2, PlanTolerances{}, 0.0, none, &d1));
  ASSERT_TRUE(joinJobs(j1, j2, PlanTolerances{}, 2.5, held));
  EXPECT_NEAR(d1, 5.0, 1e-9);
  ASSERT_EQ(none.segments.size(), 6u);   // hold 0 inserts nothing
  ASSERT_EQ(held.segments.size(), 7u);
  EXPECT_NEAR(held.duration() - none.duration(), 2.5, 1e-9);
  EXPECT_EQ(held.start_q, none.start_q);
  EXPECT_EQ(held.end_q, none.end_q);
  // the hold sits between the dwell 'd' and the ENTRY 'e' (job 2's pre-entry pose), standing still
  EXPECT_EQ(held.segments[3].stage_name, "d");
  const PlannedSegmentData & h = held.segments[4];
  EXPECT_EQ(h.stage_name, kSequenceHoldName);
  EXPECT_EQ(h.kind, SegmentKind::Dwell);
  EXPECT_FALSE(h.tool_carton_contact_allowed);
  const std::vector<double> pre_entry{0.2, 0.1};
  EXPECT_EQ(h.trajectory.points.front().positions, pre_entry);
  EXPECT_EQ(h.trajectory.points.back().positions, pre_entry);
  EXPECT_EQ(held.segments[3].trajectory.points.back().positions, pre_entry);   // neighbours unchanged
  EXPECT_EQ(held.segments[5].stage_name, "e");
  EXPECT_EQ(held.segments[5].trajectory.points.front().positions, pre_entry);
  EXPECT_NEAR(h.trajectory.duration(), 2.5, 1e-12);
  // segments before the hold keep their times, later segments are shifted by exactly D
  const auto wn = segmentWindows(none, 2.0), wh = segmentWindows(held, 2.0);
  for (size_t i = 0; i < 4; ++i) EXPECT_NEAR(wh[i].start_s, wn[i].start_s, 1e-12);
  EXPECT_NEAR(wh[5].start_s - wn[4].start_s, 2.5, 1e-9);   // ENTRY
  EXPECT_NEAR(wh[6].start_s - wn[5].start_s, 2.5, 1e-9);   // CUT
  // the joined segments still chain (the hold would break a concatenation otherwise)
  std::vector<SegmentInput> in;
  ASSERT_TRUE(held.prefixedSegments(in));
  SampledPath path;
  EXPECT_TRUE(buildSampledPath(in, SampleOptions{}, ConcatOptions{}, path));
}

TEST(SequencePlan, DelayHoldFallbacks)
{
  const ArmPlan j1 = plan("left", {seg(SegmentKind::Transit, "a", {0, 0}, {1, 0.5}, 3.0)});
  // no ENTRY: before the first CUT
  {
    const ArmPlan j2 = plan("left", {seg(SegmentKind::Transit, "c", {1, 0.5}, {0.2, 0.1}, 4.0), seg(SegmentKind::Cut, "f", {0.2, 0.1}, {0.0, 0.2}, 3.0)});
    ArmPhase held;
    ASSERT_TRUE(joinJobs(j1, j2, PlanTolerances{}, 1.0, held));
    ASSERT_EQ(held.segments.size(), 4u);
    EXPECT_EQ(held.segments[2].stage_name, kSequenceHoldName);
    EXPECT_EQ(held.segments[3].stage_name, "f");
  }
  // neither: at the start of job 2
  {
    const ArmPlan j2 = plan("left", {seg(SegmentKind::Transit, "c", {1, 0.5}, {0.2, 0.1}, 4.0)});
    ArmPhase held;
    ASSERT_TRUE(joinJobs(j1, j2, PlanTolerances{}, 1.0, held));
    ASSERT_EQ(held.segments.size(), 3u);
    EXPECT_EQ(held.segments[1].stage_name, kSequenceHoldName);
    EXPECT_EQ(held.segments[1].trajectory.points.front().positions, j1.end_q);
  }
}

TEST(SequencePlan, LeftTopFirstOrderChainsTopThenFace)
{
  // sequence_left_top_first: job 1 = the top cut planned from HOME, job 2 = the face cut planned from the top job's end pose.
  // Top: transit from home, entry, cut (ends above the seam end); face: transit from there, entry, cut, return home.
  const std::vector<double> home{0, 0}, top_end{1.2, 0.6};
  ArmPlan top = plan("left", {seg(SegmentKind::Transit, "approach pose above middle seam cut", home, {1.0, 0.5}, 3.0),
                              seg(SegmentKind::Entry, "enter middle seam", {1.0, 0.5}, {1.0, 0.6}, 1.0),
                              seg(SegmentKind::Cut, "cut middle seam", {1.0, 0.6}, top_end, 2.0)});
  ArmPlan face = plan("left", {seg(SegmentKind::Transit, "approach pose outside left side cut", top_end, {-0.4, 0.3}, 4.0),
                               seg(SegmentKind::Entry, "enter laterally into left side cut start", {-0.4, 0.3}, {-0.4, 0.4}, 1.0),
                               seg(SegmentKind::Cut, "cut left side", {-0.4, 0.4}, {-0.2, 0.4}, 3.0),
                               seg(SegmentKind::Transit, "return home", {-0.2, 0.4}, home, 3.0)});
  geometry_msgs::msg::PoseStamped t0, t1, f0, f1;
  t0.pose.position.x = 1;
  t1.pose.position.x = 2;
  f0.pose.position.x = 3;
  f1.pose.position.x = 4;
  top.cut_targets = {t0, t1};
  face.cut_targets = {f0, f1};
  ArmPhase timeline;
  double d1 = 0;
  ASSERT_TRUE(joinJobs(top, face, PlanTolerances{}, 0.0, timeline, &d1));
  EXPECT_NEAR(d1, top.duration(), 1e-9);
  ASSERT_EQ(timeline.segments.size(), 7u);
  EXPECT_EQ(timeline.segments[0].stage_name, "approach pose above middle seam cut");   // top cut first ...
  EXPECT_EQ(timeline.segments[3].stage_name, "approach pose outside left side cut");   // ... then the face
  EXPECT_EQ(timeline.start_q, home);
  EXPECT_EQ(timeline.end_q, home);   // the arm ends at home only after its LAST cut
  // cut targets follow the segment order (tip-on-line gate pairs the i-th CUT segment with targets 2i, 2i+1)
  ASSERT_EQ(timeline.cut_targets.size(), 4u);
  EXPECT_EQ(timeline.cut_targets[0].pose.position.x, 1);
  EXPECT_EQ(timeline.cut_targets[2].pose.position.x, 3);
  // a delay hold goes just before the FACE job's entry (job 2), not the top's
  ArmPhase held;
  ASSERT_TRUE(joinJobs(top, face, PlanTolerances{}, 2.0, held));
  ASSERT_EQ(held.segments.size(), 8u);
  EXPECT_EQ(held.segments[4].stage_name, kSequenceHoldName);
  EXPECT_EQ(held.segments[5].stage_name, "enter laterally into left side cut start");
  const std::vector<double> pre_entry{-0.4, 0.3};
  EXPECT_EQ(held.segments[4].trajectory.points.front().positions, pre_entry);
  EXPECT_NEAR(held.duration() - timeline.duration(), 2.0, 1e-9);
  // the windows of the swapped timeline: the face job starts after the top job
  const auto w = segmentWindows(timeline, 2.0);
  EXPECT_NEAR(w[3].start_s, top.duration(), 1e-9);
  // the wrong pairing is refused: a face job that was planned from HOME (the old order) does not start where the top job ends
  ArmPlan face_from_home = plan("left", {seg(SegmentKind::Transit, "approach pose outside left side cut", home, {-0.4, 0.3}, 4.0),
                                         seg(SegmentKind::Cut, "cut left side", {-0.4, 0.3}, {-0.2, 0.3}, 3.0)});
  ArmPhase bad;
  EXPECT_FALSE(joinJobs(top, face_from_home, PlanTolerances{}, 0.0, bad));
}

TEST(SequencePlan, JobsMustChain)
{
  const ArmPlan j1 = plan("left", {seg(SegmentKind::Transit, "a", {0, 0}, {1, 0.5}, 3.0)});
  const ArmPlan j2 = plan("left", {seg(SegmentKind::Transit, "c", {1.0, 0.6}, {0.2, 0.1}, 4.0)});
  ArmPhase out;
  EXPECT_FALSE(joinJobs(j1, j2, PlanTolerances{}, 0.0, out));
  ArmPlan other = plan("right", {seg(SegmentKind::Transit, "c", {1.0, 0.5}, {0.2, 0.1}, 4.0)});
  EXPECT_FALSE(joinJobs(j1, other, PlanTolerances{}, 0.0, out));
}

TEST(SequencePlan, SegmentWindowOverlapFilter)
{
  // left: one segment [0, 10]; right: [0, 10) then [10 + gap, ...]; margin m: windows [start-m, end+m]
  const double m = 2.0;
  const ArmPhase left = phaseOf("left", {seg(SegmentKind::Transit, "l", {0, 0}, {1, 0}, 10.0)});
  auto windowsFor = [&](double right_start)
  {
    // right arm: a stationary 'wait' segment of right_start seconds, then a 3 s segment
    return phaseOf("right", {seg(SegmentKind::Transit, "wait", {0, 0}, {0, 0}, right_start), seg(SegmentKind::Transit, "r", {0, 0}, {1, 0}, 3.0)});
  };
  // right segment 'r' starts at 14.0: its window starts at 12.0 = left's end + m -> touching windows overlap
  {
    const auto wl = segmentWindows(left, m), wr = segmentWindows(windowsFor(14.0), m);
    EXPECT_TRUE(windowsOverlap(wl[0], wr[1]));
  }
  // 14.01 s: just no overlap
  {
    const auto wl = segmentWindows(left, m), wr = segmentWindows(windowsFor(14.01), m);
    EXPECT_FALSE(windowsOverlap(wl[0], wr[1]));
    // ... but the tail (end clamping) of the left arm overlaps everything from 8 s on
    ASSERT_TRUE(wl.back().tail);
    EXPECT_TRUE(windowsOverlap(wl.back(), wr[1]));
    EXPECT_NEAR(wl.back().lo_s, 8.0, 1e-12);
  }
  // a smaller margin shrinks the windows
  {
    const auto wl = segmentWindows(left, 0.5), wr = segmentWindows(windowsFor(11.01), 0.5);
    EXPECT_FALSE(windowsOverlap(wl[0], wr[1]));   // window of r starts at 10.51 > 10.5
  }
}

TEST(SequencePlan, SkewGateDecision)
{
  // planned [10, 14], margin 2: early -> wait (hold), due -> dispatch, late -> stop
  EXPECT_EQ(skewDecision(9.0, 10.0, 14.0, 2.0, false, false), SkewAction::Wait);
  EXPECT_EQ(skewDecision(10.0, 10.0, 14.0, 2.0, false, false), SkewAction::Dispatch);
  EXPECT_EQ(skewDecision(12.0, 10.0, 14.0, 2.0, false, false), SkewAction::Dispatch);
  EXPECT_EQ(skewDecision(12.0, 10.0, 14.0, 2.0, true, false), SkewAction::Wait);    // running, on time
  EXPECT_EQ(skewDecision(16.0, 10.0, 14.0, 2.0, true, false), SkewAction::Wait);    // exactly planned end + margin: still ok
  EXPECT_EQ(skewDecision(16.01, 10.0, 14.0, 2.0, true, false), SkewAction::Stop);   // not complete in time
  EXPECT_EQ(skewDecision(16.01, 10.0, 14.0, 2.0, false, false), SkewAction::Stop);  // never even started
  EXPECT_EQ(skewDecision(20.0, 10.0, 14.0, 2.0, true, true), SkewAction::Wait);     // completed (early or late): nothing to do
}

TEST(SequencePlan, DelayCandidatesOrder)
{
  const auto c = delayCandidates(1.0, 3.0);
  ASSERT_EQ(c.size(), 7u);
  EXPECT_EQ(c[0].arm, "");
  EXPECT_EQ(c[1].arm, "left");
  EXPECT_NEAR(c[3].delay_s, 3.0, 1e-12);
  EXPECT_EQ(c[4].arm, "right");
  EXPECT_NEAR(c[4].delay_s, 1.0, 1e-12);
  EXPECT_EQ(delayCandidates(1.0, 0.0).size(), 1u);
}

TEST(PlanRetry, RetriesOnlyKnownRandomExecutorFailures)
{
  const std::vector<std::string> sigs{"winds the wrist", "MTC candidate failed", "no planner solution"};
  // executor plan-stage failures with a known signature: retry
  EXPECT_TRUE(shouldRetryPlan("right phase A: plan_h_cut_from_state failed: planned free-space transit winds the wrist: joint6 travel 3.5 rad exceeds pi", sigs));
  EXPECT_TRUE(shouldRetryPlan("left job 2 (left top): plan_h_cut_from_state failed: MTC candidate failed", sigs));
  EXPECT_TRUE(shouldRetryPlan("right phase B: plan_h_cut_from_state failed: no planner solution for the approach", sigs));
  // an executor failure that is not on the list: no retry
  EXPECT_FALSE(shouldRetryPlan("left phase A: plan_h_cut_from_state failed: snapshot is stale", sigs));
  // validator / gate refusals never retry, even when the text happens to contain a signature
  EXPECT_FALSE(shouldRetryPlan("phase A refused: left world check: collision with detected_carton", sigs));
  EXPECT_FALSE(shouldRetryPlan("phase A refused: cross-arm check failed in parallel and in both serial orders: effective distance 0.04 m < 0.05 m", sigs));
  EXPECT_FALSE(shouldRetryPlan("top seam coverage refused: GAP between the cuts", sigs));
  EXPECT_FALSE(shouldRetryPlan("phase A refused: ... MTC candidate failed ...", sigs));
  EXPECT_FALSE(shouldRetryPlan("", sigs));
  EXPECT_FALSE(shouldRetryPlan("left: plan_h_cut_from_state failed: winds the wrist", {}));   // empty list: never
  uint64_t ns = 0;
  EXPECT_TRUE(snapshotStampFromText("{\n \"a\": 1,\n \"acquisition_stamp_ns\": 1792455243549349273,\n \"b\": 2}", ns));
  EXPECT_EQ(ns, 1792455243549349273ull);
  EXPECT_FALSE(snapshotStampFromText("{\"x\": 1}", ns));
}

TEST(TopSkip, SkipDecisionAndLeftOnlyCoverage)
{
  EXPECT_TRUE(skipRightTop(1.0, 0.995));
  EXPECT_TRUE(skipRightTop(0.995, 0.995));
  EXPECT_FALSE(skipRightTop(0.9949, 0.995));
  EXPECT_FALSE(skipRightTop(0.95, 0.995));
  EXPECT_FALSE(skipRightTop(1.0, 1.0));    // 1.0 = never skip
  EXPECT_TRUE(skipRightTop(0.9, 0.9));
  EXPECT_FALSE(skipRightTop(std::nan(""), 0.995));
  TopSplitParams p;
  EXPECT_TRUE(validateTopSplitParams(p));
  p.skip_right_top_fraction = 0.89;
  EXPECT_FALSE(validateTopSplitParams(p));
  p.skip_right_top_fraction = 1.01;
  EXPECT_FALSE(validateTopSplitParams(p));
  // left-only coverage: start within the tolerance of 0 and end within the tolerance of L
  const double L = 0.39350477, tol = 0.005;
  SeamInterval full{0.0, L};
  EXPECT_TRUE(checkLeftOnlyCoverage(full, L, tol));
  EXPECT_TRUE(checkLeftOnlyCoverage(SeamInterval{0.002, 0.995 * L}, L, tol));   // 1.97 mm short of the end
  const Status short_end = checkLeftOnlyCoverage(SeamInterval{0.0, 0.98 * L}, L, tol);
  EXPECT_FALSE(short_end);
  EXPECT_TRUE(short_end.message.find("right top skipped") != std::string::npos) << short_end.message;
  EXPECT_FALSE(checkLeftOnlyCoverage(SeamInterval{0.02, L}, L, tol));           // does not start at the left end
  EXPECT_FALSE(checkLeftOnlyCoverage(full, std::nan(""), tol));
}

TEST(SequencePlan, SingleJobHoldBeforeEntry)
{
  const ArmPlan j = plan("right", {seg(SegmentKind::Transit, "t", {0, 0}, {0.5, 0.5}, 3.0), seg(SegmentKind::Entry, "e", {0.5, 0.5}, {0.5, 0.6}, 1.0),
                                   seg(SegmentKind::Cut, "c", {0.5, 0.6}, {0.1, 0.6}, 4.0)});
  ArmPhase none, held;
  ASSERT_TRUE(singleJob(j, PlanTolerances{}, 0.0, none));
  ASSERT_TRUE(singleJob(j, PlanTolerances{}, 2.0, held));
  EXPECT_EQ(none.segments.size(), 3u);
  ASSERT_EQ(held.segments.size(), 4u);
  EXPECT_EQ(held.segments[1].stage_name, kSequenceHoldName);
  EXPECT_EQ(held.start_q, j.start_q);
  EXPECT_EQ(held.end_q, j.end_q);
  EXPECT_NEAR(held.duration() - none.duration(), 2.0, 1e-9);
}

class SequenceValidatorTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    model = dual_cut_test::syntheticModel();
    ASSERT_NE(model, nullptr);
    scene = std::make_shared<planning_scene::PlanningScene>(model);
    PhaseValidatorConfig cfg;
    cfg.tip_link_suffix = "tool";
    cfg.swept.padding_m = 0.05;
    cfg.contact_links_left = {"arm_left_tool"};
    cfg.contact_links_right = {"arm_right_tool"};
    const Status s = PhaseValidator::create(scene, cfg, validator);
    ASSERT_TRUE(s) << s.message;
  }
  moveit::core::RobotModelPtr model;
  std::shared_ptr<planning_scene::PlanningScene> scene;
  std::unique_ptr<PhaseValidator> validator;
};

TEST_F(SequenceValidatorTest, EndClampingCatchesAnArmReachingAStandingArm)
{
  // The left arm moves {0,0} -> P = {0.3, 0} in 10 s and then STANDS at P (its timeline is over). The right
  // arm waits far away (pointing -x) for 18 s and then swings to R = {-0.5, 0} in 2 s, i.e. it reaches the
  // neighbourhood of P at t = 20 s. The left SEGMENT's window [-2, 12] does not overlap the right swing's
  // window [16, 22]: only the left arm's tail hold (window [8, inf)) catches it.
  const ArmPhase left = phaseOf("left", {seg(SegmentKind::Transit, "left to P", {0, 0}, {0.3, 0}, 10.0)});
  const ArmPhase right = phaseOf("right", {seg(SegmentKind::Transit, "right waits", {M_PI, 0}, {M_PI, 0}, 18.0),
                                           seg(SegmentKind::Transit, "right swings to R", {M_PI, 0}, {-0.5, 0}, 2.0)});
  const double m = 2.0;
  const auto wl = segmentWindows(left, m), wr = segmentWindows(right, m);
  ASSERT_FALSE(windowsOverlap(wl[0], wr[2 - 1]));   // segment vs swing: no overlap
  ASSERT_TRUE(windowsOverlap(wl.back(), wr[1]));    // tail vs swing: overlap
  SegmentPairCache cache;
  const WindowCrossResult r = validator->crossWindows(left, right, m, &cache);
  EXPECT_FALSE(r.safe()) << r.worst.message;
  EXPECT_TRUE(r.worst_left.find("tail") != std::string::npos) << r.worst_left;
  EXPECT_EQ(r.worst_right, "right swings to R");
  EXPECT_GE(r.overlapping_pairs, 3u);
  // the standing left pose and the waiting right arm alone are fine
  bool waits_ok = false;
  for (const auto & p : r.pairs)
    if (p.right_name == "right waits" && p.left_name.find("tail") != std::string::npos) waits_ok = p.safe;
  EXPECT_TRUE(waits_ok);
  // the same result from the cache
  const WindowCrossResult again = validator->crossWindows(left, right, m, &cache);
  EXPECT_FALSE(again.safe());
}

TEST_F(SequenceValidatorTest, SeparatedTimelinesAreSafeAndCached)
{
  // both arms swing away from each other
  const ArmPhase left = phaseOf("left", {seg(SegmentKind::Transit, "l", {0, 0}, {-0.6, 0.3}, 3.0)});
  const ArmPhase right = phaseOf("right", {seg(SegmentKind::Transit, "r", {0, 0}, {0.6, -0.3}, 3.0)});
  SegmentPairCache cache;
  const WindowCrossResult r = validator->crossWindows(left, right, 2.0, &cache);
  EXPECT_TRUE(r.safe()) << r.worst.message;
  EXPECT_EQ(r.overlapping_pairs, 4u);   // (seg, tail) x (seg, tail)
  EXPECT_EQ(cache.pairs.size(), 4u);
  EXPECT_FALSE(validator->crossWindows(left, right, -1.0, &cache).safe());   // invalid margin fails closed
}
