// Tests for the early stop of the top-cut fraction descent (top_cut_descent.hpp). The failure texts are real
// explainFailure / wrist-gate lines from scan_results/final_planner_fixes_20261022/cells/t133743_a.log.
#include <gtest/gtest.h>

#include <set>
#include <string>

#include "igus_rebel_moveit_config/top_cut_descent.hpp"

namespace
{
using namespace top_cut_descent;

const std::string kApproach = "approach pose above middle seam cut [ompl_cartesian]";
const std::string kAlign = "align above middle seam start [ompl_cartesian]";
const std::set<std::string> kPre{kApproach, kAlign};

const std::string kTree =
  "  0  - <-   0 ->   -  0 / Snapshot-relative middle_only dry run\n"
  "    1  - <-   1 ->   -  0 / start state (override)\n"
  "    -  0 ->   1 ->   -  0 / approach pose above middle seam cut [ompl_cartesian]\n";

std::string failing(const std::string & lines, int blocks = 2)
{
  std::string out = kTree;
  for (int i = 0; i < blocks; ++i) out += "Failing stage(s):\n" + lines;
  return out;
}
}  // namespace

TEST(TopCutDescentParse, ApproachFailureIsBeforeTheCutStroke)
{
  const auto text = failing(kApproach + " (0/1): INVALID_MOTION_PLAN; original pose goal retry: INVALID_MOTION_PLAN\n");
  const auto stages = parseFailingStages(text);
  ASSERT_EQ(stages.size(), 1U);
  EXPECT_EQ(stages[0].name, kApproach);
  EXPECT_EQ(stages[0].solutions, 0);
  EXPECT_EQ(stages[0].failures, 1);
  const auto c = classifyMtcFailure(text, kPre);
  EXPECT_TRUE(c.before_cut_stroke);
  EXPECT_EQ(c.stage, kApproach);
  EXPECT_NE(c.reason.find("INVALID_MOTION_PLAN"), std::string::npos);
}

TEST(TopCutDescentParse, AlignFailureIsBeforeTheCutStroke)
{
  const auto c = classifyMtcFailure(failing(kAlign + " (0/1): no IK\n"), kPre);
  EXPECT_TRUE(c.before_cut_stroke);
}

TEST(TopCutDescentParse, CutStrokeFailureIsNot)
{
  // the dominant failure of the data (1510 of 1946 failed steps): the stroke itself
  for (const char * comment :
       {"waypoint stroke failed closed: leg 6 moves joint4 3.140044 rad between two waypoints",
        "waypoint stroke failed closed: leg 33 reached Cartesian fraction 0.500000 < 1.0"})
  {
    const auto c = classifyMtcFailure(failing("cut middle seam cut [ompl_cartesian] (0/1): " + std::string(comment) + "\n"), kPre);
    EXPECT_FALSE(c.before_cut_stroke) << comment;
    EXPECT_EQ(c.stage, "cut middle seam cut [ompl_cartesian]");
  }
}

TEST(TopCutDescentParse, EntryAndLiftAreNotInTheSet)
{
  // the entry depends on the sag profile of the shortened seam: not proven independent
  EXPECT_FALSE(classifyMtcFailure(failing("descend from aligned middle seam cut start [ompl_cartesian] (0/1): x\n"), kPre).before_cut_stroke);
  // a stage name with parentheses of its own is parsed by the LAST counter
  const auto s = parseFailingStages(failing("finish pose above middle seam cut [ompl_cartesian] (straight, then tilted) (0/1): none\n"));
  ASSERT_EQ(s.size(), 1U);
  EXPECT_EQ(s[0].name, "finish pose above middle seam cut [ompl_cartesian] (straight, then tilted)");
  EXPECT_FALSE(classifyMtcFailure(failing("finish pose above middle seam cut [ompl_cartesian] (straight, then tilted) (0/1): none\n"), kPre).before_cut_stroke);
}

TEST(TopCutDescentParse, MixedAndUnknownAreNotClassified)
{
  // one fraction-dependent stage in the list is enough to keep descending
  EXPECT_FALSE(classifyMtcFailure(
    failing(kApproach + " (0/1): INVALID_MOTION_PLAN\ncut middle seam cut [ompl_cartesian] (0/1): leg 5\n"), kPre).before_cut_stroke);
  // a stage with solutions is not "failed before"
  EXPECT_FALSE(classifyMtcFailure(failing(kApproach + " (1/2): x\n"), kPre).before_cut_stroke);
  // empty block (MTC listed no stage) and no block at all: fail open = keep descending
  EXPECT_FALSE(classifyMtcFailure(failing(""), kPre).before_cut_stroke);
  EXPECT_FALSE(classifyMtcFailure(kTree, kPre).before_cut_stroke);
  EXPECT_FALSE(classifyMtcFailure("", kPre).before_cut_stroke);
  // a stage that is not in the set (e.g. a transition of a multi-segment plan)
  EXPECT_FALSE(classifyMtcFailure(failing("transition from left finish pose to middle seam cut approach pose [ompl_cartesian] (0/1): x\n"), kPre).before_cut_stroke);
}

TEST(TopCutDescentParse, OnlyTheFirstBlockCounts)
{
  const std::string text = kTree + "Failing stage(s):\n" + kApproach + " (0/1): a\nFailing stage(s):\ncut middle seam cut [ompl_cartesian] (0/1): b\n";
  const auto s = parseFailingStages(text);
  ASSERT_EQ(s.size(), 1U);
  EXPECT_EQ(s[0].name, kApproach);
  // block followed by other log text stops at the first non-stage line
  const auto t = parseFailingStages(
    "Failing stage(s):\n" + kApproach + " (0/1): a\n[WARN] [123.4] [arm_left.box_h_cut_executor]: middle_only: x (0/1): y\n");
  EXPECT_EQ(t.size(), 1U);
}

TEST(TopCutDescentWristGate, TransitInSetIsBeforeTheCutStroke)
{
  const std::string err =
    "planned free-space transit winds the wrist: transit 'approach pose above middle seam cut [ompl_cartesian]' joint6 travel 5.40266 rad exceeds 3.14159 rad";
  const auto c = classifyWristGateFailure(err, {kApproach});
  EXPECT_TRUE(c.before_cut_stroke);
  EXPECT_EQ(c.stage, kApproach);
}

TEST(TopCutDescentWristGate, OtherGateTextsAndTransitsAreNot)
{
  // a transit that is not the recorded fraction-independent one
  EXPECT_FALSE(classifyWristGateFailure(
    "planned free-space transit winds the wrist: transit 'transition from a to b' joint6 travel 5.4 rad exceeds 3.14159 rad", {kApproach}).before_cut_stroke);
  // bookkeeping / fail-closed texts of the gate are never classified
  for (const char * text :
       {"planned free-space transit winds the wrist: unexpected solution type; refusing to guess transit stages",
        "planned free-space transit winds the wrist: transit bookkeeping is inconsistent; refusing to guess",
        "planned free-space transit winds the wrist: examined 0 of 1 planned free-space transits; refusing to accept an unchecked transit",
        "planned free-space transit winds the wrist: transit 'approach pose above middle seam cut [ompl_cartesian]' has no joint6 waypoints",
        "planned trajectory has invalid pre-cut stabilization: x", ""})
    EXPECT_FALSE(classifyWristGateFailure(text, {kApproach}).before_cut_stroke) << text;
}

TEST(TopCutDescentSteps, LastStepFollowsTheMinimumFraction)
{
  EXPECT_EQ(lastStep(0.05), 19);   // 100 ... 5 %
  EXPECT_EQ(lastStep(0.5), 10);
  EXPECT_EQ(lastStep(0.52), 9);    // 100 ... 55 %
  EXPECT_EQ(lastStep(1.0), 0);
  EXPECT_NEAR(fractionOfStep(19), 0.05, 1e-12);
}

TEST(TopCutDescentSteps, ZeroDifferenceToTheOldLoopWithoutAStop)
{
  // flag off or a fraction-dependent failure: the old loop step by step, ending exactly where it ended
  for (const double min_fraction : {0.05, 0.5, 0.52, 0.8})
  {
    for (const bool before : {false, true})
    {
      for (const bool stop_flag : {false, true})
      {
        int visited = 0;
        int step = 0;
        while (true)
        {
          const auto d = decideAfterFailure(step, min_fraction, true, stop_flag && before, before && stop_flag);
          const int old_next = step + 1;
          const bool old_stop = 1.0 - 0.05 * old_next < min_fraction - 1e-9;
          if (!(stop_flag && before))
          {
            ASSERT_EQ(d.action == Action::kStop, old_stop) << min_fraction << " " << step;
            if (d.action != Action::kStop) { ASSERT_EQ(d.action, Action::kTryNext); ASSERT_EQ(d.next_step, old_next); }
          }
          if (d.action == Action::kStop) break;
          step = d.next_step;
          ++visited;
          ASSERT_LT(visited, 100);
        }
        if (!(stop_flag && before)) EXPECT_EQ(step, lastStep(min_fraction));
      }
    }
  }
}

TEST(TopCutDescentSteps, IndependentFailureJumpsToTheMinimumOnce)
{
  // min 5 %: failure at step 2 (90 %) before the stroke -> next try is step 19 (5 %)
  auto d = decideAfterFailure(2, 0.05, true, true, true);
  EXPECT_EQ(d.action, Action::kJumpToMinimum);
  EXPECT_EQ(d.next_step, 19);
  // the minimum try fails too (any reason): the descent ends
  d = decideAfterFailure(19, 0.05, true, true, true);
  EXPECT_EQ(d.action, Action::kStop);
  // one step before the minimum there is nothing to skip
  d = decideAfterFailure(18, 0.05, true, true, true);
  EXPECT_EQ(d.action, Action::kTryNext);
  EXPECT_EQ(d.next_step, 19);
  // a cut-stroke failure keeps the plain descent
  d = decideAfterFailure(2, 0.05, true, true, false);
  EXPECT_EQ(d.action, Action::kTryNext);
  EXPECT_EQ(d.next_step, 3);
  // the planning parameter off: the old behaviour for every failure
  d = decideAfterFailure(2, 0.05, true, false, true);
  EXPECT_EQ(d.action, Action::kTryNext);
  // shortening disabled: stop (as before)
  d = decideAfterFailure(0, 0.05, false, true, true);
  EXPECT_EQ(d.action, Action::kStop);
  // min fraction between the steps (52 %): the jump target is 55 %
  d = decideAfterFailure(1, 0.52, true, true, true);
  EXPECT_EQ(d.action, Action::kJumpToMinimum);
  EXPECT_EQ(d.next_step, 9);
}

TEST(TopCutDescentSteps, WholeDescentOfTheFallbackDetections)
{
  // t133743-like: cut-stroke failures at 100..., an approach failure at 75 % (step 5), the minimum try fails
  int step = 0, tried = 1;
  const int approach_fail_step = 5;
  while (true)
  {
    const bool before = step == approach_fail_step;
    const auto d = decideAfterFailure(step, 0.05, true, true, before);
    if (d.action == Action::kStop) break;
    step = d.next_step;
    ++tried;
  }
  EXPECT_EQ(step, 19);
  EXPECT_EQ(tried, 7);   // steps 0..5 plus the minimum try instead of 20
}
