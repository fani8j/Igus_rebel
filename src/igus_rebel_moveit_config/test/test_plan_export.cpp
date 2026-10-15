// Tests for the stage-name -> segment-kind classifier of the plan export (plan_export.hpp).
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "igus_rebel_moveit_config/plan_export.hpp"

using plan_export::Kind;
using plan_export::classifyStageName;

namespace
{
// Every stage name planCutSegments / addTiltableLift / StationaryDwell can emit with a trajectory,
// as seen in a t_cut plan (scan_results/dual_cut_p3_regression_20261015) plus the variants of the
// other patterns.
struct Case
{
  const char * name;
  Kind kind;
};
const Case kCases[] = {
  {"approach pose outside left side cut [ompl_cartesian]", Kind::kTransit},
  {"approach pose above middle seam cut [ompl_cartesian]", Kind::kTransit},
  {"approach pose above right side cut [ompl_pilz]", Kind::kTransit},
  {"transition from left side cut finish pose to middle seam cut approach pose [ompl_cartesian]", Kind::kTransit},
  {"transit to supplied link6 approach [ompl_cartesian]", Kind::kTransit},
  {"align above middle seam start [ompl_cartesian]", Kind::kAlign},
  {"stabilize blade before left side cut", Kind::kDwell},
  {"hold at penetration before middle seam cut", Kind::kDwell},
  {"enter laterally into left side cut start [ompl_cartesian]", Kind::kEntry},
  {"descend from aligned middle seam cut start [ompl_cartesian]", Kind::kEntry},
  {"descend from aligned right side cut start [ompl_cartesian]", Kind::kEntry},
  {"cut left side cut [ompl_cartesian]", Kind::kCut},
  {"cut middle seam cut [ompl_cartesian]", Kind::kCut},
  {"cut right side cut [ompl_pilz]", Kind::kCut},
  {"cut middle debug short move [ompl_cartesian]", Kind::kCut},
  {"finish pose outside left side cut [ompl_cartesian]", Kind::kFinish},
  {"finish pose above middle seam cut [ompl_cartesian] [lift tilt 0.0 deg]", Kind::kFinish},
  {"finish pose above middle seam cut [ompl_cartesian] [lift tilt -4.0 deg]", Kind::kFinish},
  {"lift clear of left side cut to hover [cartesian]", Kind::kLift},
  {"lift straight up before transit (table clearance)", Kind::kLift},
  {"raise to safe transit height after left side cut [cartesian]", Kind::kLift},
};
}  // namespace

TEST(PlanExport, ClassifiesEveryKnownStageName)
{
  for (const auto & c : kCases) {
    const auto kind = classifyStageName(c.name);
    ASSERT_TRUE(kind.has_value()) << c.name;
    EXPECT_EQ(*kind, c.kind) << c.name << " -> " << plan_export::kindName(*kind);
  }
}

TEST(PlanExport, UnknownStageNamesAreNotGuessed)
{
  for (const char * name : {
         "", "current robot state", "add detected carton collision object",
         "allow knife-tip contact with carton for left side cut",
         "restore knife-tip collision checking after left side cut", "start state (override)",
         "Approach pose above middle seam cut", "approach pose", "approach pose above",
         "cutting stage", "uncut left side cut", "lifting something", "finish pose left side cut",
         "transition to left side cut", "stabilise blade before left side cut",
         " cut left side cut"}) {
    EXPECT_FALSE(classifyStageName(name).has_value()) << "'" << name << "'";
  }
}

TEST(PlanExport, KindValuesMatchTheMessageConstants)
{
  // dual_cut_msgs/PlannedSegment: TRANSIT=0 ALIGN=1 DWELL=2 ENTRY=3 CUT=4 FINISH=5 LIFT=6
  EXPECT_EQ(static_cast<int>(Kind::kTransit), 0);
  EXPECT_EQ(static_cast<int>(Kind::kAlign), 1);
  EXPECT_EQ(static_cast<int>(Kind::kDwell), 2);
  EXPECT_EQ(static_cast<int>(Kind::kEntry), 3);
  EXPECT_EQ(static_cast<int>(Kind::kCut), 4);
  EXPECT_EQ(static_cast<int>(Kind::kFinish), 5);
  EXPECT_EQ(static_cast<int>(Kind::kLift), 6);
  EXPECT_STREQ(plan_export::kindName(Kind::kCut), "CUT");
  EXPECT_STREQ(plan_export::kindName(Kind::kLift), "LIFT");
}

TEST(PlanExport, SegmentIndexTakesTheLastNameOccurrence)
{
  const std::vector<std::string> names{"left side cut", "middle seam cut"};
  EXPECT_EQ(plan_export::segmentIndexForStage("approach pose outside left side cut [ompl_cartesian]", names), 0U);
  EXPECT_EQ(plan_export::segmentIndexForStage("cut middle seam cut [ompl_cartesian]", names), 1U);
  EXPECT_EQ(
    plan_export::segmentIndexForStage(
      "transition from left side cut finish pose to middle seam cut approach pose [ompl_cartesian]", names),
    1U);
  EXPECT_FALSE(plan_export::segmentIndexForStage("lift straight up before transit (table clearance)", names));
  EXPECT_FALSE(plan_export::segmentIndexForStage("cut something else", {}));
  EXPECT_FALSE(plan_export::segmentIndexForStage("cut left side cut", {""}));
}

TEST(PlanExport, PhysicalCutSwapsSidesOnlyWithTheRotate180Relabel)
{
  EXPECT_EQ(plan_export::physicalCut("left", false), "left");
  EXPECT_EQ(plan_export::physicalCut("right", false), "right");
  EXPECT_EQ(plan_export::physicalCut("left", true), "right");
  EXPECT_EQ(plan_export::physicalCut("right", true), "left");
  EXPECT_EQ(plan_export::physicalCut("middle", true), "middle");
  EXPECT_EQ(plan_export::physicalCut("", true), "");
}
