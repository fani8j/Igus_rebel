// Plan order fallback chain (chain building, refusal classification, left job order) and detection permission
// (cut_permission parsing, execute gate).
#include <gtest/gtest.h>

#include "dual_cut_coordinator/plan_policy.hpp"
#include "dual_cut_coordinator/sequence_plan.hpp"   // shouldRetryPlan

using namespace dual_cut;

namespace
{
const std::vector<std::string> kDefaultChain = {"sequence", "sequence_left_top_first", "phases_top_first", "phases_side_first"};

std::vector<std::string> names(const std::vector<PlanVariant> & v)
{
  std::vector<std::string> out;
  for (const auto & x : v) out.push_back(x.name);
  return out;
}
}  // namespace

// ---- chain ------------------------------------------------------------------------------------

TEST(PlanChain, ParsesEveryKnownEntry)
{
  PlanVariant v;
  ASSERT_TRUE(parsePlanVariant("sequence", v));
  EXPECT_TRUE(v.sequence);
  EXPECT_FALSE(v.left_top_first);
  ASSERT_TRUE(parsePlanVariant("sequence_left_top_first", v));
  EXPECT_TRUE(v.sequence);
  EXPECT_TRUE(v.left_top_first);
  ASSERT_TRUE(parsePlanVariant("phases_top_first", v));
  EXPECT_FALSE(v.sequence);
  EXPECT_EQ(v.phase_order, "top_first");
  ASSERT_TRUE(parsePlanVariant("phases_side_first", v));
  EXPECT_EQ(v.phase_order, "side_first");
  EXPECT_FALSE(parsePlanVariant("sequence_right_first", v));
  EXPECT_FALSE(parsePlanVariant("", v));
  EXPECT_EQ(knownPlanVariantNames().size(), 4u);
}

TEST(PlanChain, ConfiguredModeNames)
{
  EXPECT_EQ(configuredVariantName("sequence", "side_first"), "sequence");   // phase_order is irrelevant in sequence mode
  EXPECT_EQ(configuredVariantName("phases", "side_first"), "phases_side_first");
  EXPECT_EQ(configuredVariantName("phases", "top_first"), "phases_top_first");
}

TEST(PlanChain, StartsAtTheConfiguredEntry)
{
  std::vector<PlanVariant> chain;
  ASSERT_TRUE(buildPlanChain(kDefaultChain, "sequence", chain));
  EXPECT_EQ(names(chain), kDefaultChain);
  ASSERT_TRUE(buildPlanChain(kDefaultChain, "phases_top_first", chain));
  EXPECT_EQ(names(chain), (std::vector<std::string>{"phases_top_first", "phases_side_first"}));
  ASSERT_TRUE(buildPlanChain(kDefaultChain, "phases_side_first", chain));
  EXPECT_EQ(names(chain), (std::vector<std::string>{"phases_side_first"}));
}

TEST(PlanChain, ConfiguredModeOutsideTheListGoesFirst)
{
  std::vector<PlanVariant> chain;
  ASSERT_TRUE(buildPlanChain({"sequence_left_top_first", "phases_top_first"}, "sequence", chain));
  EXPECT_EQ(names(chain), (std::vector<std::string>{"sequence", "sequence_left_top_first", "phases_top_first"}));
}

TEST(PlanChain, CustomOrderIsKept)
{
  std::vector<PlanVariant> chain;
  ASSERT_TRUE(buildPlanChain({"sequence", "phases_side_first", "sequence_left_top_first"}, "sequence", chain));
  EXPECT_EQ(names(chain), (std::vector<std::string>{"sequence", "phases_side_first", "sequence_left_top_first"}));
  EXPECT_TRUE(chain[2].left_top_first);
}

TEST(PlanChain, InvalidConfigurationFailsClosed)
{
  std::vector<PlanVariant> chain;
  EXPECT_FALSE(buildPlanChain({}, "sequence", chain));
  EXPECT_TRUE(chain.empty());
  EXPECT_FALSE(buildPlanChain({"sequence", "bogus"}, "sequence", chain));
  EXPECT_TRUE(chain.empty());
  EXPECT_FALSE(buildPlanChain({"sequence", "sequence"}, "sequence", chain));
  EXPECT_FALSE(buildPlanChain(kDefaultChain, "nonsense", chain));
}

// ---- classification --------------------------------------------------------------------------

TEST(PlanClassify, TodaysLeftTopWristRefusalIsOrderFixable)
{
  const std::string m =
    "refused after 2 retries: left job 2 (left top): plan_h_cut_from_state failed: external snapshot snap_t074923_a.json: middle_only planning "
    "failed down to 5 % of the seam: planned free-space transit winds the wrist: transit 'approach pose above middle seam cut [ompl_cartesian]' "
    "joint6 travel 5.2807 rad exceeds 3.14159 rad";
  EXPECT_EQ(classifyRefusal(m), RefusalClass::OrderFixable);
  EXPECT_EQ(classifyRefusal("left job 2 (left top): plan_h_cut_from_state failed: MTC candidate failed: task=x"), RefusalClass::OrderFixable);
}

TEST(PlanClassify, ValidatorRefusalsAreOrderFixable)
{
  EXPECT_EQ(classifyRefusal("phase A refused: right world check: collision at sample 608 between arm_right_knife_c_holder and table"),
            RefusalClass::OrderFixable);
  EXPECT_EQ(classifyRefusal("sequence schedule not possible: world / tip-on-line check of the sequence timelines failed (a delay cannot fix it)"),
            RefusalClass::OrderFixable);
  EXPECT_EQ(classifyRefusal("top split coverage: left top (job 1) has 2 CUT segments, expected exactly one"), RefusalClass::OrderFixable);
}

TEST(PlanClassify, SnapshotAndPreconditionRefusalsAreNot)
{
  EXPECT_EQ(classifyRefusal("left: snapshot is stale (301.2 s old, limit 240 s)"), RefusalClass::NotOrderFixable);
  EXPECT_EQ(classifyRefusal("left: snapshot has no valid acquisition stamp"), RefusalClass::NotOrderFixable);
  EXPECT_EQ(classifyRefusal("left: plan was made with a stale-snapshot override"), RefusalClass::NotOrderFixable);
  EXPECT_EQ(classifyRefusal("precondition: the arms are not at rest (0.1 s)"), RefusalClass::NotOrderFixable);
  EXPECT_EQ(classifyRefusal("precondition: no /joint_states for both arms"), RefusalClass::NotOrderFixable);
  EXPECT_EQ(classifyRefusal("busy: another plan / rearm call holds the plan lock"), RefusalClass::NotOrderFixable);
  EXPECT_EQ(classifyRefusal("stop latched during planning"), RefusalClass::NotOrderFixable);
  EXPECT_EQ(classifyRefusal("top split source snapshot: cannot read x"), RefusalClass::NotOrderFixable);
  EXPECT_EQ(classifyRefusal("left: frame-consistency gate: max |dt| 0.1"), RefusalClass::NotOrderFixable);
  EXPECT_EQ(classifyRefusal("left: planning_frame 'world' != 'base_link'"), RefusalClass::NotOrderFixable);
  EXPECT_EQ(classifyRefusal("planning scene: service /get_planning_scene is not available"), RefusalClass::NotOrderFixable);
  EXPECT_EQ(classifyRefusal("carton 'detected_carton' missing in the right arm's scene"), RefusalClass::NotOrderFixable);
  EXPECT_EQ(classifyRefusal("left executor plan mismatch: the executor holds plan_seq 3, the chosen attempt's last job is 4"),
            RefusalClass::NotOrderFixable);
}

TEST(PlanClassify, SnapshotRejectedByTheExecutorWinsOverThePlanStagePrefix)
{
  // an executor plan-stage failure, but a geometric snapshot problem no order can fix
  EXPECT_EQ(classifyRefusal("right phase A: plan_h_cut_from_state failed: external snapshot right_top_snapshot.json: snapshot rejected: snapshot seam or "
                            "side cutting point is outside the detected carton top face"),
            RefusalClass::NotOrderFixable);
  EXPECT_EQ(classifyRefusal("left phase A: plan_h_cut_from_state failed: left varying orientation search failed closed (left_cut_extend_to_corners)"),
            RefusalClass::NotOrderFixable);
}

TEST(PlanClassify, UnknownTextIsNotGuessed)
{
  EXPECT_EQ(classifyRefusal("something nobody has seen before"), RefusalClass::NotOrderFixable);
  EXPECT_EQ(classifyRefusal(""), RefusalClass::NotOrderFixable);
}

TEST(PlanChain, ReportJsonAndDescription)
{
  std::vector<PlanVariant> chain;
  ASSERT_TRUE(buildPlanChain(kDefaultChain, "sequence", chain));
  PlanAttemptRecord a;
  a.variant = "sequence";
  a.reason = "left job 2 \"quoted\" winds the wrist";
  a.cls = RefusalClass::OrderFixable;
  a.wall_s = 12.5;
  a.retries = 2;
  const std::string j = planFallbackJson(chain, 1, {a}, 280.0, 12.5);
  EXPECT_NE(j.find("\"attempt\":2"), std::string::npos);
  EXPECT_NE(j.find("\"variant\":\"sequence_left_top_first\""), std::string::npos);
  EXPECT_NE(j.find("\"class\":\"order_fixable\""), std::string::npos);
  EXPECT_NE(j.find("\\\"quoted\\\""), std::string::npos);   // escaped
  EXPECT_EQ(describeAttempts({a, a}, 10).find("[2] sequence"), describeAttempts({a, a}, 10).rfind("[2] sequence"));
  EXPECT_NE(describeAttempts({a}, 10).find("..."), std::string::npos);
}

// ---- left job order --------------------------------------------------------------------------

TEST(LeftJobOrder, FaceFirstIsTodaysOrder)
{
  const std::vector<std::string> face = {"cut_pattern=left_only", "middle_seam_stop=end"};
  const std::vector<std::string> top = {"cut_pattern=middle_only", "top_cut_min_cut_fraction=0.05"};
  const LeftJobOrder o = orderLeftJobs(false, face, top, "home=off", "home=on");
  EXPECT_FALSE(o.top_first);
  EXPECT_EQ(o.job1, (std::vector<std::string>{"cut_pattern=left_only", "middle_seam_stop=end", "home=off"}));
  EXPECT_EQ(o.job2, (std::vector<std::string>{"cut_pattern=middle_only", "top_cut_min_cut_fraction=0.05", "home=on"}));
  EXPECT_EQ(o.job1_label, "left face");
  EXPECT_EQ(o.job2_label, "left top");
}

TEST(LeftJobOrder, TopFirstSwapsTheJobsAndTheHomeFlags)
{
  const std::vector<std::string> face = {"cut_pattern=left_only"};
  const std::vector<std::string> top = {"cut_pattern=middle_only"};
  const LeftJobOrder o = orderLeftJobs(true, face, top, "home=off", "home=on");
  EXPECT_TRUE(o.top_first);
  // the first job never returns home, the last one does (the arm ends at home after its last cut)
  EXPECT_EQ(o.job1, (std::vector<std::string>{"cut_pattern=middle_only", "home=off"}));
  EXPECT_EQ(o.job2, (std::vector<std::string>{"cut_pattern=left_only", "home=on"}));
  EXPECT_EQ(o.job1_label, "left top");
  EXPECT_EQ(o.job2_label, "left face");
}

// ---- detection permission --------------------------------------------------------------------

TEST(DetectionPermission, AllowedAndBlockedAsTheDetectorWritesThem)
{
  const auto allowed = parseDetectionPermission(R"J({"seam_start":[0,0,0],"cut_permission":{"status":"allowed","reasons":[],"flags":["height_waived(support_hidden)"]}})J");
  EXPECT_EQ(allowed.state, DetectionPermission::State::Allowed);
  EXPECT_FALSE(allowed.malformed);
  EXPECT_FALSE(allowed.has_override);
  EXPECT_TRUE(allowed.reasons.empty());
  EXPECT_STREQ(allowed.stateName(), "allowed");

  const auto blocked = parseDetectionPermission(
    R"J({"cut_permission":{"status":"blocked","reasons":[{"code":"generic_seam_unconfirmed","retryable":false,"detail":"seam status estimated"}]}})J");
  EXPECT_EQ(blocked.state, DetectionPermission::State::Blocked);
  EXPECT_FALSE(blocked.malformed);
  ASSERT_EQ(blocked.reasons.size(), 1u);
  EXPECT_EQ(blocked.reasons[0], "generic_seam_unconfirmed: seam status estimated");
  EXPECT_FALSE(blocked.has_override);
  EXPECT_STREQ(blocked.stateName(), "blocked");
}

TEST(DetectionPermission, MissingFieldIsUnknownAndAllowed)
{
  for (const char * text : {R"J({"seam_start":[0,0,0]})J", R"J({"cut_permission":null})J"})
  {
    const auto p = parseDetectionPermission(text);
    EXPECT_EQ(p.state, DetectionPermission::State::Unknown);
    EXPECT_STREQ(p.stateName(), "unknown");
    const auto g = detectionExecuteGate(p, false, true);
    EXPECT_TRUE(g.allowed);
    EXPECT_EQ(g.by, "unknown_permission");
  }
}

TEST(DetectionPermission, UnusableFieldFailsClosed)
{
  for (const char * text : {R"J({"cut_permission":"allowed"})J", R"J({"cut_permission":{"status":"maybe"}})J", R"J({"cut_permission":{}})J",
                            R"J({"cut_permission":{"reasons":[]}})J", R"J({"cut_permission":[]})J", "not json at all", R"J({"cut_permission":{"status":"ALLOWED"}})J"})
  {
    const auto p = parseDetectionPermission(text);
    EXPECT_EQ(p.state, DetectionPermission::State::Blocked) << text;
    EXPECT_TRUE(p.malformed) << text;
    EXPECT_FALSE(p.note.empty()) << text;
    EXPECT_FALSE(detectionExecuteGate(p, false, true).allowed) << text;
  }
}

TEST(DetectionPermission, OverrideTruthiness)
{
  EXPECT_TRUE(overrideTruthy("true"));
  EXPECT_TRUE(overrideTruthy("\"allow_cut_blocked\""));
  EXPECT_TRUE(overrideTruthy("1"));
  EXPECT_TRUE(overrideTruthy("[\"x\"]"));
  EXPECT_TRUE(overrideTruthy("{\"by\":\"operator\"}"));
  EXPECT_FALSE(overrideTruthy("false"));
  EXPECT_FALSE(overrideTruthy("null"));
  EXPECT_FALSE(overrideTruthy("0"));
  EXPECT_FALSE(overrideTruthy("\"\""));
  EXPECT_FALSE(overrideTruthy("\"None\""));
  EXPECT_FALSE(overrideTruthy("\"false\""));
  EXPECT_FALSE(overrideTruthy("[]"));
  EXPECT_FALSE(overrideTruthy("{}"));
  EXPECT_FALSE(overrideTruthy("garbage"));
}

TEST(DetectionPermission, ExecuteGateTable)
{
  const auto blocked = parseDetectionPermission(R"J({"cut_permission":{"status":"blocked","reasons":[{"code":"c","detail":"d"}]}})J");
  const auto overridden =
    parseDetectionPermission(R"J({"cut_permission":{"status":"blocked","reasons":[{"code":"c"}],"override":"allow_cut_blocked"}})J");
  const auto allowed = parseDetectionPermission(R"J({"cut_permission":{"status":"allowed"}})J");

  // allowed: always
  EXPECT_TRUE(detectionExecuteGate(allowed, false, true).allowed);
  EXPECT_EQ(detectionExecuteGate(allowed, false, true).by, "allowed");
  // blocked, nothing else: refused, with the reason in the message and the way out
  const auto g = detectionExecuteGate(blocked, false, true);
  EXPECT_FALSE(g.allowed);
  EXPECT_EQ(g.by, "refused");
  EXPECT_NE(g.message.find("c: d"), std::string::npos);
  EXPECT_NE(g.message.find("allow_blocked_detection"), std::string::npos);
  // blocked + the runtime parameter: allowed (said loudly)
  const auto ga = detectionExecuteGate(blocked, true, true);
  EXPECT_TRUE(ga.allowed);
  EXPECT_EQ(ga.by, "allow_blocked_detection");
  EXPECT_NE(ga.message.find("BLOCKED"), std::string::npos);
  // blocked + the detector's own override: allowed only when honored
  const auto go = detectionExecuteGate(overridden, false, true);
  EXPECT_TRUE(go.allowed);
  EXPECT_EQ(go.by, "detector_override");
  EXPECT_NE(go.message.find("allow_cut_blocked"), std::string::npos);
  const auto gn = detectionExecuteGate(overridden, false, false);
  EXPECT_FALSE(gn.allowed);
  // the parameter wins over the override when both are set (reported as the parameter)
  EXPECT_EQ(detectionExecuteGate(overridden, true, true).by, "allow_blocked_detection");
  // an unusable cut_permission is never rescued by an override field, only by the explicit parameter
  const auto bad = parseDetectionPermission(R"J({"cut_permission":{"status":"maybe","override":true}})J");
  EXPECT_TRUE(bad.malformed);
  EXPECT_FALSE(detectionExecuteGate(bad, false, true).allowed);
  EXPECT_TRUE(detectionExecuteGate(bad, true, true).allowed);
}

TEST(DetectionPermission, ReportJson)
{
  const auto blocked =
    parseDetectionPermission(R"J({"cut_permission":{"status":"blocked","reasons":[{"code":"c","detail":"d \"q\""}],"override":"allow_cut_blocked"}})J");
  const std::string j = detectionPermissionJson(blocked, true);
  EXPECT_NE(j.find("\"permission\":\"blocked\""), std::string::npos);
  EXPECT_NE(j.find("\"blocked\":true"), std::string::npos);
  EXPECT_NE(j.find("\"override\":\"allow_cut_blocked\""), std::string::npos);
  EXPECT_NE(j.find("\"override_honored\":true"), std::string::npos);
  EXPECT_NE(detectionPermissionJson(blocked, false).find("\"override_honored\":false"), std::string::npos);
  const std::string u = detectionPermissionJson(parseDetectionPermission("{}"), true);
  EXPECT_NE(u.find("\"permission\":\"unknown\""), std::string::npos);
  EXPECT_NE(u.find("\"blocked\":false"), std::string::npos);
  EXPECT_NE(u.find("\"override\":null"), std::string::npos);
}

TEST(DetectionPermission, MustBelongToThePlannedSnapshot)
{
  const auto p = parseDetectionPermission(
    R"J({"snapshot_id":"snapA","acquisition_stamp_ns":1792000000000000001,"cut_permission":{"status":"blocked","reasons":[]}})J");
  EXPECT_EQ(p.snapshot_id, "snapA");
  EXPECT_EQ(p.stamp_ns, 1792000000000000001ull);
  EXPECT_TRUE(checkPermissionMatchesPlan(p, "snapA", 1792000000000000001ull));
  const Status other_id = checkPermissionMatchesPlan(p, "snapB", 1792000000000000001ull);
  EXPECT_FALSE(other_id);
  EXPECT_NE(other_id.message.find("source snapshot changed"), std::string::npos);
  EXPECT_FALSE(checkPermissionMatchesPlan(p, "snapA", 1792000000000000002ull));
  // an absent value in the document is not compared (old snapshots)
  const auto bare = parseDetectionPermission(R"J({"cut_permission":{"status":"allowed"}})J");
  EXPECT_TRUE(checkPermissionMatchesPlan(bare, "whatever", 5));
  // classified as not fixable by an order change
  EXPECT_EQ(classifyRefusal(other_id.message), RefusalClass::NotOrderFixable);
}

// ---- retry order (breadth first, 2026-10-24) ----------------------------------------------------
// The refusal texts are the real ones of scan_results/final_planner_fixes_20261022 (cells/*.log).

namespace
{
const char * const kLeftTopDescent =
  "left job 2 (left top): plan_h_cut_from_state failed: external snapshot s.json: middle_only planning failed down to 5 % of the seam: "
  "planned free-space transit winds the wrist: transit 'approach pose above middle seam cut [ompl_cartesian]' joint6 travel 5.40262 rad exceeds 3.14159 rad";
const char * const kLeftFaceAfterTop =
  "left job 2 (left face): plan_h_cut_from_state failed: external snapshot s.json: MTC candidate failed: task=Snapshot-relative left_only dry run profile=ompl_cartesian segments=1";
const char * const kRightFace =
  "right job 2 (right face): plan_h_cut_from_state failed: external snapshot r.json: MTC candidate failed: task=Snapshot-relative left_only dry run profile=ompl_cartesian segments=1";
const char * const kRightTop =
  "right job 1 (right top): plan_h_cut_from_state failed: external snapshot r.json: MTC candidate failed: task=Snapshot-relative middle_only dry run profile=ompl_cartesian segments=1";
const char * const kCrossArm = "phase A refused: cross-arm check failed in parallel and in both serial orders: effective distance 0.04 m < 0.05 m";
const char * const kStale = "right job 1 (right top): plan_h_cut_from_state failed: snapshot is stale";

PlanRetryPolicy policy(int retries = 2, bool breadth = true)
{
  PlanRetryPolicy p;
  p.max_retries = retries;
  p.signatures = {"winds the wrist", "MTC candidate failed", "no planner solution"};
  p.inplace_labels = {"right job 2 (right face)"};
  p.breadth_first = breadth;
  return p;
}

struct Run
{
  std::vector<std::pair<size_t, bool>> order;   // (entry, retry)
  bool ok = false;
  size_t used = 0;
  bool second = false;
  int retries = 0;
};
// outcome(entry, execution number of that entry) -> "" = success, else the refusal text
template <typename F>
Run drive(size_t entries, const PlanRetryPolicy & p, F outcome, size_t cap = 64)
{
  PlanChainDriver d(entries, p);
  Run r;
  std::vector<int> n(entries, 0);
  while (r.order.size() < cap)
  {
    const auto nx = d.next();
    if (nx.done) break;
    r.order.push_back({nx.entry, nx.retry});
    const std::string refusal = outcome(nx.entry, n[nx.entry]++);
    if (refusal.empty())
    {
      d.succeeded();
      break;
    }
    d.failed(refusal, classifyRefusal(refusal));
  }
  r.ok = d.ok();
  r.used = d.used();
  r.second = d.usedInSecondPass();
  r.retries = d.totalRetries();
  return r;
}
using Order = std::vector<std::pair<size_t, bool>>;
}  // namespace

TEST(PlanRetryOrder, InPlaceLabelIsOnlyTheRightFaceOfTheData)
{
  const auto p = policy();
  EXPECT_TRUE(retryInPlaceNow(kRightFace, p));
  EXPECT_FALSE(retryInPlaceNow(kLeftTopDescent, p));
  EXPECT_FALSE(retryInPlaceNow(kLeftFaceAfterTop, p));
  EXPECT_FALSE(retryInPlaceNow(kRightTop, p));
  EXPECT_FALSE(retryInPlaceNow(kCrossArm, p));   // not an executor plan-stage failure
  EXPECT_FALSE(retryInPlaceNow("right job 2 (right face): plan_h_cut_from_state failed: snapshot is stale", p));
  // the former behaviour: every random-looking refusal
  const auto legacy = policy(2, false);
  EXPECT_TRUE(retryInPlaceNow(kLeftTopDescent, legacy));
  EXPECT_TRUE(retryInPlaceNow(kRightTop, legacy));
  EXPECT_FALSE(retryInPlaceNow(kCrossArm, legacy));
}

TEST(PlanRetryOrder, EveryEntryOnceBeforeAnyRetry)
{
  // the six fallback detections: sequence and sequence_left_top_first refuse, phases_top_first passes
  const auto r = drive(4, policy(), [](size_t entry, int) -> std::string {
    if (entry == 0) return kLeftTopDescent;
    if (entry == 1) return kLeftFaceAfterTop;
    return "";
  });
  EXPECT_EQ(r.order, (Order{{0, false}, {1, false}, {2, false}}));   // before: 0, 0r, 0r, 1, 1r, 1r, 2
  EXPECT_TRUE(r.ok);
  EXPECT_EQ(r.used, 2u);
  EXPECT_FALSE(r.second);
  EXPECT_EQ(r.retries, 0);
}

TEST(PlanRetryOrder, RightFaceIsRetriedAtOnceAndKeepsItsVariant)
{
  const auto r = drive(4, policy(), [](size_t, int exec) -> std::string { return exec == 0 ? std::string(kRightFace) : std::string(); });
  EXPECT_EQ(r.order, (Order{{0, false}, {0, true}}));
  EXPECT_TRUE(r.ok);
  EXPECT_EQ(r.used, 0u);
  EXPECT_FALSE(r.second);
  EXPECT_EQ(r.retries, 1);
}

TEST(PlanRetryOrder, InPlaceRetriesAreBoundedByPlanRetries)
{
  // refused every time: 1 + 2 in place, then the next entry (which passes)
  const auto r = drive(4, policy(2), [](size_t entry, int) -> std::string { return entry == 0 ? std::string(kRightFace) : std::string(); });
  EXPECT_EQ(r.order, (Order{{0, false}, {0, true}, {0, true}, {1, false}}));
  EXPECT_EQ(r.used, 1u);
  EXPECT_EQ(r.retries, 2);
  // plan_retries 0: no retry at all
  const auto z = drive(4, policy(0), [](size_t entry, int) -> std::string { return entry == 0 ? std::string(kRightFace) : std::string(); });
  EXPECT_EQ(z.order, (Order{{0, false}, {1, false}}));
}

TEST(PlanRetryOrder, NothingPassedRetriesInChainOrderRoundRobin)
{
  const auto r = drive(4, policy(2), [](size_t, int) -> std::string { return kLeftTopDescent; });
  EXPECT_EQ(r.order, (Order{{0, false}, {1, false}, {2, false}, {3, false},
                            {0, true}, {1, true}, {2, true}, {3, true},
                            {0, true}, {1, true}, {2, true}, {3, true}}));
  EXPECT_FALSE(r.ok);
  EXPECT_EQ(r.retries, 8);
}

TEST(PlanRetryOrder, SecondPassSuccessIsReportedAsSuch)
{
  // entry 2 passes on its first retry only
  const auto r = drive(4, policy(2), [](size_t entry, int exec) -> std::string { return (entry == 2 && exec == 1) ? std::string() : std::string(kLeftTopDescent); });
  EXPECT_EQ(r.order, (Order{{0, false}, {1, false}, {2, false}, {3, false}, {0, true}, {1, true}, {2, true}}));
  EXPECT_TRUE(r.ok);
  EXPECT_EQ(r.used, 2u);
  EXPECT_TRUE(r.second);
  EXPECT_EQ(r.retries, 3);
}

TEST(PlanRetryOrder, OnlyRandomLookingRefusalsAreRetriedInThePass)
{
  // a validator refusal (cross-arm) is order-fixable but not random: the pass leaves it alone
  const auto r = drive(2, policy(2), [](size_t entry, int) -> std::string { return entry == 0 ? std::string(kCrossArm) : std::string(kLeftTopDescent); });
  EXPECT_EQ(r.order, (Order{{0, false}, {1, false}, {1, true}, {1, true}}));
}

TEST(PlanRetryOrder, RefusalNoOrderCanFixEndsEverything)
{
  // stale snapshot in entry 1: chain ends, no second pass
  const auto r = drive(4, policy(2), [](size_t entry, int) -> std::string { return entry == 0 ? std::string(kLeftTopDescent) : std::string(kStale); });
  EXPECT_EQ(r.order, (Order{{0, false}, {1, false}}));
  EXPECT_FALSE(r.ok);
  // a not-order-fixable refusal on the in-place list is still retried at once (as before), then ends
  const std::string rf = "right job 2 (right face): plan_h_cut_from_state failed: MTC candidate failed: varying orientation search failed closed";
  ASSERT_EQ(classifyRefusal(rf), RefusalClass::NotOrderFixable);
  const auto q = drive(4, policy(1), [&](size_t, int) -> std::string { return rf; });
  EXPECT_EQ(q.order, (Order{{0, false}, {0, true}}));
}

TEST(PlanRetryOrder, VetoesFromTheNode)
{
  // half budget used before an in-place retry: the chain goes on with the next entry
  {
    PlanChainDriver d(3, policy());
    auto nx = d.next();
    ASSERT_FALSE(nx.done);
    d.failed(kRightFace, classifyRefusal(kRightFace));
    nx = d.next();
    EXPECT_TRUE(nx.retry);
    EXPECT_EQ(nx.entry, 0u);
    d.declineRetry();
    EXPECT_EQ(d.retriesUsed(0), 0);
    nx = d.next();
    EXPECT_FALSE(nx.done);
    EXPECT_EQ(nx.entry, 1u);
    EXPECT_FALSE(nx.retry);
  }
  // second pass declined: everything ends
  {
    PlanChainDriver d(2, policy());
    for (int i = 0; i < 2; ++i)
    {
      const auto nx = d.next();
      ASSERT_FALSE(nx.done);
      d.failed(kLeftTopDescent, classifyRefusal(kLeftTopDescent));
    }
    auto nx = d.next();
    EXPECT_TRUE(nx.second_pass);
    d.declineRetry();
    EXPECT_TRUE(d.next().done);
  }
  // stop() (budget / stale snapshot / stop latch)
  {
    PlanChainDriver d(4, policy());
    ASSERT_FALSE(d.next().done);
    d.failed(kLeftTopDescent, classifyRefusal(kLeftTopDescent));
    d.stop();
    EXPECT_TRUE(d.next().done);
    EXPECT_FALSE(d.ok());
  }
}

TEST(PlanRetryOrder, FormerBehaviourWhenBreadthFirstIsOff)
{
  const auto r = drive(4, policy(2, false), [](size_t entry, int) -> std::string { return entry < 2 ? std::string(kLeftTopDescent) : std::string(); });
  EXPECT_EQ(r.order, (Order{{0, false}, {0, true}, {0, true}, {1, false}, {1, true}, {1, true}, {2, false}}));
  EXPECT_EQ(r.used, 2u);
  // no second pass: all refused -> done after the last entry
  const auto n = drive(2, policy(1, false), [](size_t, int) -> std::string { return kLeftTopDescent; });
  EXPECT_EQ(n.order, (Order{{0, false}, {0, true}, {1, false}, {1, true}}));
  EXPECT_FALSE(n.ok);
}

TEST(PlanRetryOrder, ReasonsOfAnEntryAreKeptForTheReport)
{
  PlanChainDriver d(2, policy());
  ASSERT_FALSE(d.next().done);
  d.failed(kRightFace, classifyRefusal(kRightFace));
  ASSERT_FALSE(d.next().done);
  d.failed("right job 2 (right face): plan_h_cut_from_state failed: no planner solution", RefusalClass::OrderFixable);
  ASSERT_EQ(d.reasons(0).size(), 2u);
  EXPECT_EQ(d.reasons(0).front(), kRightFace);
  EXPECT_EQ(d.retriesUsed(0), 1);
  EXPECT_EQ(d.executions(1), 0);
}

TEST(PlanChain, FallbackJsonCarriesThePass)
{
  std::vector<PlanVariant> chain;
  ASSERT_TRUE(buildPlanChain(kDefaultChain, "sequence", chain));
  EXPECT_NE(planFallbackJson(chain, 0, {}, 150.0, 1.0).find("\"pass\":1"), std::string::npos);
  EXPECT_NE(planFallbackJson(chain, 2, {}, 150.0, 1.0, 2).find("\"pass\":2"), std::string::npos);
}

// ---- collision-gate retries (plan_retry_signatures with the three collision-gate entries, 2026-10-24) -----------------
// The refusal texts of fallback_speed_20261024/cells/t074923_a.log (paths shortened); the signature list is the default of
// coordinator_node.cpp / config/coordinator.yaml (a drift test below reads both files).

#include <fstream>
#include <sstream>

namespace
{
const std::vector<std::string> kSignaturesWithGate = {
  "winds the wrist", "MTC candidate failed", "no planner solution",
  "collision gate: stage 'approach pose", "collision gate: stage 'transition from", "collision gate: stage 'transit to"};

// t074923_a, attempt 3/4 phases_top_first: the right arm's random OMPL approach grazed the camera head
const char * const kGateApproachPhasesTopFirst =
  "right phase B: plan_h_cut_from_state failed: external snapshot run/right_top_snapshot.json: collision gate: stage "
  "'approach pose above middle seam cut [ompl_cartesian]' collides (camera_head / knife_c_wrist_camera) between waypoints 107 and 108/362; "
  "refusing the plan";
// t074923_a, attempt 1/4 sequence: the minimum-fraction try of the left top, approach into the carton
const char * const kGateApproachSequence =
  "left job 2 (left top): plan_h_cut_from_state failed: external snapshot run/snap_t074923_a.json: middle_only planning failed down to 5 % of the "
  "seam: collision gate: stage 'approach pose above middle seam cut [ompl_cartesian]' collides (detected_carton / knife_c_blade) between "
  "waypoints 4 and 5/332; refusing the plan [top cut descent: 100 % failed before the cut stroke in 'approach pose above middle seam cut "
  "[ompl_cartesian]', independent of the cut length; only the minimum fraction 5 % was tried after it]";
// strokes and other Cartesian / contact stages (formats of validateTrajectoriesCollisionFree: "between waypoints" or "at waypoint")
std::string gate(const std::string & stage, const std::string & who = "left job 2 (left top)")
{
  return who + ": plan_h_cut_from_state failed: external snapshot run/snap.json: collision gate: stage '" + stage +
         "' collides (detected_carton / knife_c_blade) at waypoint 12/100; refusing the plan";
}

PlanRetryPolicy gatePolicy(int retries = 2)
{
  PlanRetryPolicy p = policy(retries);
  p.signatures = kSignaturesWithGate;
  return p;
}

std::string readFile(const std::string & path)
{
  std::ifstream in(path);
  std::stringstream ss;
  ss << in.rdbuf();
  return ss.str();
}
}  // namespace

TEST(PlanRetryGate, ApproachAndTransitCollisionGateRefusalsLookRandom)
{
  EXPECT_TRUE(shouldRetryPlan(kGateApproachPhasesTopFirst, kSignaturesWithGate));
  EXPECT_TRUE(shouldRetryPlan(kGateApproachSequence, kSignaturesWithGate));
  EXPECT_TRUE(shouldRetryPlan(gate("approach pose outside left side cut [ompl_cartesian]", "right job 2 (right face)"), kSignaturesWithGate));
  EXPECT_TRUE(shouldRetryPlan(gate("transition from left side cut finish pose to middle seam cut approach pose [ompl_cartesian]"), kSignaturesWithGate));
  EXPECT_TRUE(shouldRetryPlan(gate("transit to supplied link6 approach [ompl_cartesian]"), kSignaturesWithGate));
  EXPECT_TRUE(shouldRetryPlan(gate("transit to supplied home pose after the cut [ompl_cartesian]"), kSignaturesWithGate));
  // without the entries (the former list) the t074923_a refusal is not retried
  const std::vector<std::string> former{"winds the wrist", "MTC candidate failed", "no planner solution"};
  EXPECT_FALSE(shouldRetryPlan(kGateApproachPhasesTopFirst, former));
  // still an order-fixable executor plan-stage failure (it enters the chain as before)
  EXPECT_EQ(classifyRefusal(kGateApproachPhasesTopFirst), RefusalClass::OrderFixable);
}

TEST(PlanRetryGate, StrokeAndOtherCartesianCollisionGateRefusalsAreNotRetried)
{
  // every stage name of planCutSegments / the staged planners that is NOT a free-space transit
  for (const char * stage :
       {"cut middle seam cut [ompl_cartesian]", "cut left side cut [ompl_cartesian]", "cut right side cut [ompl_cartesian]",
        "enter laterally into left side cut start [ompl_cartesian]", "descend from aligned middle seam cut start [ompl_cartesian]",
        "align above middle seam start [ompl_cartesian]", "finish pose above middle seam cut [ompl_cartesian]",
        "finish pose above middle seam cut [ompl_cartesian] [lift tilt 4.0 deg]", "finish pose outside left side cut [ompl_cartesian]",
        "lift clear of left side cut to hover [cartesian]", "raise to safe transit height after left side cut [cartesian]",
        "lift straight up before transit (table clearance)", "top cut: cut stroke to planned end plane [ompl_cartesian]",
        "top cut: penetrate to planned cut start [ompl_cartesian]", "side cut: penetrate to planned start [ompl_cartesian]",
        "side-tape continue: cut stroke parallel to planned line [ompl_cartesian]",
        "side-tape retract: transit to left approach pose [ompl_cartesian]"})   // not a plan_h_cut stage; does not start with "transit to"
    EXPECT_FALSE(shouldRetryPlan(gate(stage), kSignaturesWithGate)) << stage;
  // the signature needs the stage-name prefix right after "stage '": a stroke whose BODIES mention a transit is not matched
  EXPECT_FALSE(shouldRetryPlan(
    "left job 2 (left top): plan_h_cut_from_state failed: collision gate: stage 'cut left side cut [ompl_cartesian]' collides (approach pose / x) at waypoint 3/9; refusing the plan",
    kSignaturesWithGate));
  // a validator-level text with the words is not an executor plan-stage failure
  EXPECT_FALSE(shouldRetryPlan("phase A refused: collision gate: stage 'approach pose above middle seam cut' collides", kSignaturesWithGate));
  // a stroke refusal is still order-fixable, i.e. the chain goes on, it is just not retried
  EXPECT_EQ(classifyRefusal(gate("cut middle seam cut [ompl_cartesian]")), RefusalClass::OrderFixable);
}

TEST(PlanRetryGate, T074923SecondPassRetriesPhasesTopFirstAfterItsCollisionGateRefusal)
{
  // t074923_a: sequence (left top wrist) and sequence_left_top_first (left face MTC) refuse as always, phases_top_first refuses once with the
  // approach collision gate, phases_side_first (left phase B wrist wind) refuses; in the second pass phases_top_first passes on its retry.
  const char * const kPhasesSideFirst =
    "left phase B: plan_h_cut_from_state failed: external snapshot run/snap_t074923_a.json: middle_only planning failed down to 5 % of the seam: "
    "planned free-space transit winds the wrist: transit 'approach pose above middle seam cut [ompl_cartesian]' joint6 travel 5.28 rad exceeds 3.14159 rad";
  const auto outcome = [&](size_t entry, int exec) -> std::string {
    if (entry == 0) return kLeftTopDescent;
    if (entry == 1) return kLeftFaceAfterTop;
    if (entry == 2) return exec == 0 ? std::string(kGateApproachPhasesTopFirst) : std::string();
    return kPhasesSideFirst;
  };
  const auto r = drive(4, gatePolicy(), outcome);
  EXPECT_TRUE(r.ok);
  EXPECT_EQ(r.used, 2u);
  EXPECT_TRUE(r.second);                       // reported as a second-pass success
  EXPECT_EQ(r.order, (Order{{0, false}, {1, false}, {2, false}, {3, false}, {0, true}, {1, true}, {2, true}}));
  EXPECT_EQ(r.retries, 3);
  // with the former signature list phases_top_first is never retried and the plan is lost (the t074923_a outcome)
  PlanRetryPolicy former = policy();
  const auto lost = drive(4, former, outcome);
  EXPECT_FALSE(lost.ok);
  for (const auto & step : lost.order) EXPECT_FALSE(step.first == 2 && step.second);
}

TEST(PlanRetryGate, StrokeGateRefusalOfPhasesTopFirstIsNotRetriedInThePass)
{
  const std::string stroke = gate("cut middle seam cut [ompl_cartesian]", "right phase B");
  const auto r = drive(4, gatePolicy(), [&](size_t entry, int) -> std::string { return entry == 2 ? stroke : std::string(kLeftTopDescent); });
  EXPECT_FALSE(r.ok);
  for (const auto & step : r.order) EXPECT_FALSE(step.first == 2 && step.second) << "the stroke refusal must not be retried";
  // entries 0, 1, 3 are retried twice each as before (random-looking), entry 2 once
  EXPECT_EQ(r.order.size(), 4u + 3u * 2u);
}

TEST(PlanRetryGate, CollisionGateRefusalOnTheInPlaceListIsRetriedAtOnce)
{
  // the in-place label rule works on the same signature list: a right-face approach gate hit is retried in place
  const std::string right_face = gate("approach pose outside left side cut [ompl_cartesian]", "right job 2 (right face)");
  const auto r = drive(4, gatePolicy(), [&](size_t, int exec) -> std::string { return exec == 0 ? right_face : std::string(); });
  EXPECT_EQ(r.order, (Order{{0, false}, {0, true}}));
  EXPECT_TRUE(r.ok);
  EXPECT_FALSE(r.second);
  // ... but the right TOP approach gate hit is not on the list: next entry first
  const std::string right_top = gate("approach pose above middle seam cut [ompl_cartesian]", "right job 1 (right top)");
  const auto q = drive(4, gatePolicy(), [&](size_t entry, int) -> std::string { return entry == 0 ? right_top : std::string(); });
  EXPECT_EQ(q.order, (Order{{0, false}, {1, false}}));
}

TEST(PlanRetryGate, DefaultSignatureListIsTheSameInNodeAndYaml)
{
  // drift guard: coordinator_node.cpp (code default) and config/coordinator.yaml must carry the entries used above
  const std::string dir = DUAL_CUT_TEST_DATA_DIR;
  const std::string yaml = readFile(dir + "/../../config/coordinator.yaml");
  const std::string node = readFile(dir + "/../../src/coordinator_node.cpp");
  ASSERT_FALSE(yaml.empty()) << dir;
  ASSERT_FALSE(node.empty()) << dir;
  for (const auto & sig : kSignaturesWithGate)
  {
    EXPECT_NE(yaml.find("\"" + sig + "\""), std::string::npos) << "coordinator.yaml lacks: " << sig;
    EXPECT_NE(node.find("\"" + sig + "\""), std::string::npos) << "coordinator_node.cpp lacks: " << sig;
  }
}
