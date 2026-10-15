#include <gtest/gtest.h>

#include <cmath>

#include "dual_cut_coordinator/phase_plan.hpp"

using namespace dual_cut;

namespace
{

JointTrajectory line(const std::vector<double> & from, const std::vector<double> & to, double T = 1.0)
{
  JointTrajectory t;
  for (size_t i = 1; i <= from.size(); ++i) t.joint_names.push_back("joint" + std::to_string(i));
  TrajPoint a, b;
  a.time_s = 0.0;
  a.positions = from;
  a.velocities.assign(from.size(), 0.0);
  a.accelerations.assign(from.size(), 0.0);
  b = a;
  b.time_s = T;
  b.positions = to;
  t.points = {a, b};
  return t;
}

PlannedSegmentData seg(
  SegmentKind kind, const std::string & name, const std::vector<double> & from,
  const std::vector<double> & to, bool contact = false)
{
  PlannedSegmentData s;
  s.kind = kind;
  s.stage_name = name;
  s.cut_id = "left";
  s.physical_cut = "left";
  s.tool_carton_contact_allowed = contact;
  s.trajectory = line(from, to);
  return s;
}

SnapshotIdentity snap()
{
  SnapshotIdentity s;
  s.snapshot_id = "snap-1";
  s.capture_id = "cap-1";
  s.acquisition_stamp_ns = 1234567890ull;
  return s;
}

ArmPlan plan(
  const std::string & side, std::vector<PlannedSegmentData> segs)
{
  ArmPlan p;
  p.side = side;
  p.snapshot = snap();
  p.joint_names = {"joint1", "joint2"};
  p.start_q = segs.front().trajectory.points.front().positions;
  p.end_q = segs.back().trajectory.points.back().positions;
  p.segments = std::move(segs);
  return p;
}

// Left: A = [approach 0->1, cut 1->2, lift 2->3]; B = [transit 3->4, transit 4->5, align 5->6, cut 6->7]
ArmPlan leftA()
{
  return plan("left", {
    seg(SegmentKind::Transit, "approach pose above x", {0, 0}, {1, 0}),
    seg(SegmentKind::Cut, "cut middle", {1, 0}, {2, 0}, true),
    seg(SegmentKind::Lift, "lift straight up", {2, 0}, {3, 0})});
}
ArmPlan leftB()
{
  return plan("left", {
    seg(SegmentKind::Transit, "transition from a", {3, 0}, {4, 0}),
    seg(SegmentKind::Transit, "transit to supplied", {4, 0}, {5, 0}),
    seg(SegmentKind::Align, "align above middle seam start", {5, 0}, {6, 0}),
    seg(SegmentKind::Cut, "cut left", {6, 0}, {7, 0}, true)});
}
ArmPlan rightA()
{
  return plan("right", {
    seg(SegmentKind::Transit, "approach pose above y", {0, 1}, {0, 2}),
    seg(SegmentKind::Cut, "cut right", {0, 2}, {0, 3}, true)});
}
ArmPlan rightB()
{
  return plan("right", {
    seg(SegmentKind::Transit, "transition from b", {0, 3}, {0, 4}),
    seg(SegmentKind::Finish, "finish pose above", {0, 4}, {0, 5})});
}

}  // namespace

TEST(PhasePlan, KindMapping)
{
  SegmentKind k;
  ASSERT_TRUE(segmentKindFromUint8(dual_cut_msgs::msg::PlannedSegment::LIFT, k));
  EXPECT_EQ(k, SegmentKind::Lift);
  EXPECT_FALSE(segmentKindFromUint8(7, k));
  EXPECT_STREQ(toString(SegmentKind::Dwell), "DWELL");
  // The enum values mirror the message constants.
  EXPECT_EQ(static_cast<uint8_t>(SegmentKind::Transit), dual_cut_msgs::msg::PlannedSegment::TRANSIT);
  EXPECT_EQ(static_cast<uint8_t>(SegmentKind::Cut), dual_cut_msgs::msg::PlannedSegment::CUT);
  EXPECT_EQ(static_cast<uint8_t>(SegmentKind::Finish), dual_cut_msgs::msg::PlannedSegment::FINISH);
}

TEST(PhasePlan, ValidateArmPlan)
{
  EXPECT_TRUE(validateArmPlan(leftA(), PlanTolerances{}));
  ArmPlan p = leftA();
  p.segments[1].trajectory.points.front().positions[0] += 1e-3;  // chain gap
  EXPECT_FALSE(validateArmPlan(p, PlanTolerances{}));
  p = leftA();
  p.end_q[0] += 1e-3;
  EXPECT_FALSE(validateArmPlan(p, PlanTolerances{}));
  p = leftA();
  p.segments[2].trajectory.joint_names[0] = "joint9";
  EXPECT_FALSE(validateArmPlan(p, PlanTolerances{}));
  p = leftA();
  p.segments.clear();
  EXPECT_FALSE(validateArmPlan(p, PlanTolerances{}));
  p = leftA();
  p.side = "middle";
  EXPECT_FALSE(validateArmPlan(p, PlanTolerances{}));
}

TEST(PhasePlan, ShiftMovesOnlyTheLeadingTransits)
{
  ArmPlan a = leftA(), b = leftB();
  size_t moved = 99;
  ASSERT_TRUE(shiftLeadingTransits(a, b, PlanTolerances{}, &moved));
  EXPECT_EQ(moved, 2u);
  ASSERT_EQ(a.segments.size(), 5u);
  ASSERT_EQ(b.segments.size(), 2u);
  EXPECT_EQ(a.segments[3].stage_name, "transition from a");
  EXPECT_EQ(a.segments[4].stage_name, "transit to supplied");
  EXPECT_EQ(b.segments[0].kind, SegmentKind::Align);
  EXPECT_EQ(a.end_q, (std::vector<double>{5, 0}));
  EXPECT_EQ(b.start_q, a.end_q);  // handoff exact after the move
  EXPECT_TRUE(validateArmPlan(a, PlanTolerances{}));
  EXPECT_TRUE(validateArmPlan(b, PlanTolerances{}));
  // Total duration is conserved.
  EXPECT_DOUBLE_EQ(a.duration() + b.duration(), leftA().duration() + leftB().duration());
}

TEST(PhasePlan, ShiftWithoutLeadingTransitsChangesNothing)
{
  ArmPlan a = rightA();
  ArmPlan b = plan("right", {seg(SegmentKind::Align, "align above middle seam start", {0, 3}, {0, 4})});
  size_t moved = 99;
  ASSERT_TRUE(shiftLeadingTransits(a, b, PlanTolerances{}, &moved));
  EXPECT_EQ(moved, 0u);
  EXPECT_EQ(a.segments.size(), 2u);
  EXPECT_EQ(b.segments.size(), 1u);
}

TEST(PhasePlan, ShiftFailsClosedAndLeavesPlansUnchanged)
{
  {  // handoff gap
    ArmPlan a = leftA(), b = leftB();
    b.start_q[0] += 1e-6;
    b.segments[0].trajectory.points.front().positions[0] += 1e-6;
    EXPECT_FALSE(shiftLeadingTransits(a, b, PlanTolerances{}));
    EXPECT_EQ(a.segments.size(), 3u);
    EXPECT_EQ(b.segments.size(), 4u);
  }
  {  // different snapshots
    ArmPlan a = leftA(), b = leftB();
    b.snapshot.acquisition_stamp_ns += 1;
    EXPECT_FALSE(shiftLeadingTransits(a, b, PlanTolerances{}));
  }
  {  // different arms
    ArmPlan a = leftA(), b = rightB();
    EXPECT_FALSE(shiftLeadingTransits(a, b, PlanTolerances{}));
  }
  {  // B only transits: nothing would be left for B
    ArmPlan a = leftA();
    ArmPlan b = plan("left", {seg(SegmentKind::Transit, "transition from a", {3, 0}, {4, 0})});
    EXPECT_FALSE(shiftLeadingTransits(a, b, PlanTolerances{}));
    EXPECT_EQ(a.segments.size(), 3u);
    EXPECT_EQ(b.segments.size(), 1u);
  }
}

TEST(PhasePlan, BuildPhasesAndValidationBookkeeping)
{
  PhaseSet set;
  ASSERT_TRUE(buildPhases(leftA(), leftB(), rightA(), rightB(), PlanTolerances{}, set));
  EXPECT_EQ(set.moved_left, 2u);
  EXPECT_EQ(set.moved_right, 1u);
  EXPECT_EQ(set.a.name(), "A");
  EXPECT_EQ(set.a.left().segments.size(), 5u);
  EXPECT_EQ(set.a.right().segments.size(), 3u);
  EXPECT_EQ(set.b.left().segments.size(), 2u);
  EXPECT_EQ(set.b.right().segments.size(), 1u);
  EXPECT_EQ(set.a.left().end_q, set.b.left().start_q);
  EXPECT_EQ(set.a.right().end_q, set.b.right().start_q);
  EXPECT_DOUBLE_EQ(set.a.duration(), 5.0);  // left 5 segments of 1 s
  EXPECT_DOUBLE_EQ(set.b.duration(), 2.0);

  Phase & a = set.a;
  EXPECT_FALSE(a.validated());
  EXPECT_FALSE(a.rejected());
  const uint64_t rev = a.revision();
  ValidationRecord rec;
  rec.cross_arm_ok = true;
  rec.min_cross_distance_m = 0.07;
  a.recordValidation(rec, true);
  EXPECT_TRUE(a.validated());
  EXPECT_EQ(a.validation().revision, rev);

  // Any mutable access drops the validation.
  a.mutableLeft();
  EXPECT_FALSE(a.validated());
  EXPECT_GT(a.revision(), rev);

  rec.cross_arm_ok = false;
  a.recordValidation(rec, false);
  EXPECT_FALSE(a.validated());
  EXPECT_TRUE(a.rejected());
  a.mutableRight().segments.pop_back();
  EXPECT_FALSE(a.rejected());
}

TEST(PhasePlan, BuildPhasesFailClosed)
{
  PhaseSet set;
  // mixed snapshot between the arms
  ArmPlan r = rightA();
  r.snapshot.capture_id = "other";
  EXPECT_FALSE(buildPhases(leftA(), leftB(), r, rightB(), PlanTolerances{}, set));
  // wrong order of sides
  EXPECT_FALSE(buildPhases(rightA(), leftB(), leftA(), rightB(), PlanTolerances{}, set));
  // invalid plan
  ArmPlan bad = leftB();
  bad.segments.clear();
  EXPECT_FALSE(buildPhases(leftA(), bad, rightA(), rightB(), PlanTolerances{}, set));
  // A/B handoff gap
  ArmPlan b2 = leftB();
  b2.start_q[0] += 1e-5;
  b2.segments[0].trajectory.points.front().positions[0] += 1e-5;
  EXPECT_FALSE(buildPhases(leftA(), b2, rightA(), rightB(), PlanTolerances{}, set));
}

TEST(PhasePlan, PrefixedSegmentsFeedTheSampler)
{
  PhaseSet set;
  ASSERT_TRUE(buildPhases(leftA(), leftB(), rightA(), rightB(), PlanTolerances{}, set));
  std::vector<SegmentInput> in;
  ASSERT_TRUE(set.a.right().prefixedSegments(in));
  ASSERT_EQ(in.size(), 3u);
  EXPECT_EQ(in[0].trajectory.joint_names[0], "arm_right_joint1");
  EXPECT_TRUE(in[1].contact_allowed);  // the cut segment
  EXPECT_FALSE(in[0].contact_allowed);

  SampleOptions so;
  so.dt_s = 0.1;
  so.max_joint_step_rad = 0.0;
  SampledPath path;
  ASSERT_TRUE(buildSampledPath(in, so, ConcatOptions{}, path));
  EXPECT_DOUBLE_EQ(path.duration(), 3.0);
  EXPECT_EQ(path.joint_names[1], "arm_right_joint2");

  // The unprefixed original is untouched.
  EXPECT_EQ(set.a.right().segments[0].trajectory.joint_names[0], "joint1");
}

TEST(PhasePlan, FromResponseReordersAndChecks)
{
  dual_cut_msgs::srv::GetPlannedSolution::Response resp;
  resp.success = true;
  resp.plan_seq = 4;
  resp.snapshot_id = "snap-1";
  resp.capture_id = "cap-1";
  resp.acquisition_stamp_ns = 1234567890ull;
  // start/end states list the joints in reverse order; segments use the natural order.
  resp.start_state.name = {"joint2", "joint1"};
  resp.start_state.position = {0.0, 0.0};
  resp.end_state.name = {"joint2", "joint1"};
  resp.end_state.position = {0.0, 2.0};
  for (int k = 0; k < 2; ++k)
  {
    dual_cut_msgs::msg::PlannedSegment s;
    s.kind = k == 0 ? dual_cut_msgs::msg::PlannedSegment::TRANSIT : dual_cut_msgs::msg::PlannedSegment::CUT;
    s.stage_name = k == 0 ? "approach pose above x" : "cut middle";
    s.tool_carton_contact_allowed = k == 1;
    s.trajectory.joint_trajectory = toMsg(line({1.0 * k, 0.0}, {1.0 * (k + 1), 0.0}));
    resp.segments.push_back(s);
  }
  ArmPlan p;
  Status st = fromResponse("left", resp, p);
  ASSERT_TRUE(st) << st.message;
  EXPECT_EQ(p.joint_names, (std::vector<std::string>{"joint2", "joint1"}));  // plan order = start_state
  EXPECT_EQ(p.segments[1].trajectory.joint_names, p.joint_names);
  EXPECT_DOUBLE_EQ(p.segments[1].trajectory.points.back().positions[1], 2.0);  // joint1 moved
  EXPECT_EQ(p.end_q, (std::vector<double>{0.0, 2.0}));
  EXPECT_EQ(p.snapshot, snap());
  EXPECT_TRUE(p.segments[1].tool_carton_contact_allowed);

  // fail closed
  auto bad = resp;
  bad.success = false;
  EXPECT_FALSE(fromResponse("left", bad, p));
  bad = resp;
  bad.segments[0].kind = 42;
  EXPECT_FALSE(fromResponse("left", bad, p));
  bad = resp;
  bad.end_state.position[1] = 5.0;  // end_state does not match the last segment
  EXPECT_FALSE(fromResponse("left", bad, p));
  bad = resp;
  bad.segments[0].trajectory.joint_trajectory.points.clear();
  EXPECT_FALSE(fromResponse("left", bad, p));
  bad = resp;
  bad.segments[0].trajectory.multi_dof_joint_trajectory.points.resize(1);
  EXPECT_FALSE(fromResponse("left", bad, p));
  EXPECT_FALSE(fromResponse("middle", resp, p));
}
