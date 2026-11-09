// Pure planning / execution policy of the coordinator (no ROS, plain gtests):
//
//   1. PLAN ORDER FALLBACK CHAIN. When the requested plan is refused, ~/plan tries the next entry of a configurable
//      chain and keeps the first plan that passes every existing check:
//        sequence                  schedule_mode sequence, left arm face cut first, then its top cut (today's order)
//        sequence_left_top_first   schedule_mode sequence, the LEFT arm's job order swapped: top cut first (from home),
//                                  then the face cut; the right arm keeps its order (top remainder, then face)
//        phases_top_first          schedule_mode phases, phase_order top_first
//        phases_side_first         schedule_mode phases, phase_order side_first
//      Only refusals an order change can plausibly fix are retried (classifyRefusal); everything else ends the chain.
//
//   2. DETECTION PERMISSION. The detector writes cut_permission into the snapshot ({"status": "allowed"|"blocked",
//      "reasons": [...], "override": ...}). The plan still runs; the report says so and ~/execute is refused when the
//      plan's snapshot is blocked unless the runtime parameter allow_blocked_detection is true or the detector's own
//      override is set (and honored).
#pragma once

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "dual_cut_coordinator/trajectory_tools.hpp"   // Status

namespace dual_cut
{

// ---------------------------------------------------------------------------------------------
// plan order fallback chain
// ---------------------------------------------------------------------------------------------

struct PlanVariant
{
  std::string name;            // chain entry name
  bool sequence = false;       // schedule_mode sequence (else phases)
  bool left_top_first = false; // sequence only: the left arm cuts its top first, then its face
  std::string phase_order;     // phases only: "top_first" | "side_first"
};

/// The names a chain may contain.
const std::vector<std::string> & knownPlanVariantNames();
/// Parses one chain entry name ("sequence", "sequence_left_top_first", "phases_top_first", "phases_side_first").
Status parsePlanVariant(const std::string & name, PlanVariant & out);
/// Chain entry of the configured schedule_mode / phase_order: "sequence" or "phases_<phase_order>".
std::string configuredVariantName(const std::string & schedule_mode, const std::string & phase_order);
/// Effective chain: the entries of `configured_chain` from the one that equals `configured_name` to the end. If the
/// configured mode is not in the list it is put first and followed by the whole list (duplicates removed). An unknown or
/// duplicate name, or an empty list, is an error (fail closed, nothing is planned).
Status buildPlanChain(const std::vector<std::string> & configured_chain, const std::string & configured_name, std::vector<PlanVariant> & out);

enum class RefusalClass
{
  OrderFixable,      // a different job / phase order or start state can change the outcome: try the next chain entry
  NotOrderFixable    // snapshot, scene, frame, precondition, busy ... or pure geometry: a retry would only repeat it
};
const char * toString(RefusalClass c);
/// Classifies a refusal message of one planning attempt.
///  OrderFixable (an order change changes the start state of a left / right job, the phase structure or the pairing
///  that the cross-arm check sees):
///    - an executor plan-stage failure ("plan_h_cut_from_state failed"), e.g. the planned free-space transit winds the wrist
///      (the same goal needs 0.28 rad of joint6 from home but 5.3 rad after the face cut), "MTC candidate failed", "no planner solution"
///    - a validator refusal of the trajectories: world check, tip-on-line, cross-arm (phase / segment-window), top-seam coverage
///    - the sequence schedule found no safe delay ("sequence schedule not possible")
///  NotOrderFixable (checked first, wins over the above):
///    - preconditions (joint states, at rest), stop latch, busy / another call, execution running
///    - snapshot problems: stale / unreadable / no stamp / "snapshot rejected" / outside the detected top face / seam too short /
///      the source changed during planning
///    - frame / planning_frame / tool_link / frame-consistency gate, executor parameter rejected, bad configuration
///    - planning scene / carton missing or inconsistent, robot model / service not available / timeouts
///    - left face reach ("varying orientation search failed closed"): pure carton / IK geometry, identical in every order
///    - a retained executor plan that does not match the chosen attempt
RefusalClass classifyRefusal(const std::string & refusal);

/// One finished (failed) attempt for the report.
struct PlanAttemptRecord
{
  std::string variant;
  std::string reason;
  RefusalClass cls = RefusalClass::NotOrderFixable;
  double wall_s = 0.0;
  int retries = 0;
};
/// JSON object for ~/plan_report. `current` is the attempt that is running / produced the report.
std::string planFallbackJson(
  const std::vector<PlanVariant> & chain, size_t current, const std::vector<PlanAttemptRecord> & previous, double budget_s,
  double elapsed_before_s, int pass = 1);
/// Short human text: "[1] sequence: <reason, cut to max_len>; [2] ...".
std::string describeAttempts(const std::vector<PlanAttemptRecord> & attempts, size_t max_len = 260);

// ---------------------------------------------------------------------------------------------
// retry order (breadth first, 2026-10-24)
// ---------------------------------------------------------------------------------------------

/// How ~/plan spends plan_retries over the chain.
///   BREADTH FIRST (default): every chain entry is tried ONCE before any entry is retried. The only exception is a
///   refusal whose job text is on `inplace_labels` (data 2026-10-22: a retry of "right job 2 (right face)" - a plain
///   OMPL approach failure of the right face cut - succeeded in 4 of 4 cases, while 0 of 70 retries of the left top
///   descent, the right top and the left face after the top succeeded): that one is retried at once, in place.
///   If no entry passed, the second pass retries the entries in chain order (only refusals that look random, see
///   shouldRetryPlan), up to plan_retries retries per entry in total.
///   breadth_first = false: the former behaviour (every random-looking refusal is retried in place first).
struct PlanRetryPolicy
{
  int max_retries = 2;                         // per chain entry, in place and second pass together
  std::vector<std::string> signatures;         // shouldRetryPlan(): refusal texts that look random
  std::vector<std::string> inplace_labels;     // substrings of a refusal that is retried at once (breadth first only)
  bool breadth_first = true;
};
/// true: this refusal is retried immediately in place (executor plan-stage failure with a known random signature AND
/// one of the labels in the text).
bool retryInPlaceNow(const std::string & refusal, const PlanRetryPolicy & policy);

/// The order of attempts of ~/plan as a state machine without clock or ROS (the node adds the budget / snapshot checks and the
/// planning itself). Usage: loop { n = next(); if done break; (veto: stop() / declineRetry()); run; succeeded() | failed() }.
class PlanChainDriver
{
public:
  PlanChainDriver(size_t entries, PlanRetryPolicy policy);

  struct Next
  {
    bool done = true;
    size_t entry = 0;        // chain index to run
    bool retry = false;      // a retry of an entry that was already run
    bool second_pass = false;
  };
  /// The next attempt (done = nothing left). Must be followed by succeeded() / failed() (or stop() / declineRetry()).
  Next next();
  void succeeded();
  /// The attempt returned by next() was refused. `cls` = classifyRefusal(reason).
  void failed(const std::string & reason, RefusalClass cls);
  /// The caller does not allow the retry that next() returned (half budget used, snapshot too old): in the first pass the
  /// chain goes on with the next entry (or ends if the refusal is not order-fixable), in the second pass it ends.
  void declineRetry();
  /// End of everything (budget used up, stale snapshot, stop latch).
  void stop();

  bool finished() const { return phase_ == Phase::Done; }
  /// failed() scheduled an immediate in-place retry of the same entry.
  bool inplacePending() const { return inplace_pending_; }
  bool ok() const { return ok_; }
  size_t used() const { return used_; }
  bool usedInSecondPass() const { return ok_ && used_second_pass_; }
  /// Executions of `entry` so far (first run + retries).
  int executions(size_t entry) const { return entry < executions_.size() ? executions_[entry] : 0; }
  int retriesUsed(size_t entry) const { return std::max(0, executions(entry) - 1); }
  int totalRetries() const;
  /// Refusals of the earlier executions of `entry`, oldest first.
  const std::vector<std::string> & reasons(size_t entry) const { return reasons_.at(entry); }

private:
  enum class Phase { First, Second, Done };
  bool eligibleForSecondPass(size_t entry) const;

  size_t n_;
  PlanRetryPolicy policy_;
  Phase phase_ = Phase::First;
  size_t first_next_ = 0;       // next entry to start in the first pass
  size_t cur_ = 0;
  bool awaiting_ = false;       // next() returned an attempt that is not reported yet
  bool inplace_pending_ = false;
  bool in_second_ = false;      // the attempt in flight belongs to the second pass
  bool ok_ = false;
  size_t used_ = 0;
  bool used_second_pass_ = false;
  std::vector<int> executions_;
  std::vector<std::vector<std::string>> reasons_;
  std::vector<RefusalClass> cls_;
};

/// Parameter lists of the LEFT arm's two jobs of a sequence plan. face / top are the arm's lists WITHOUT the
/// return_home_after_cut entry. The first job never returns home (return_home_after_cut=false), the last does.
struct LeftJobOrder
{
  std::vector<std::string> job1, job2;
  std::string job1_label, job2_label;   // "left face" / "left top"
  bool top_first = false;
};
LeftJobOrder orderLeftJobs(
  bool top_first, const std::vector<std::string> & face, const std::vector<std::string> & top, const std::string & home_off,
  const std::string & home_on);

// ---------------------------------------------------------------------------------------------
// detection permission
// ---------------------------------------------------------------------------------------------

struct DetectionPermission
{
  enum class State
  {
    Allowed,    // cut_permission.status == "allowed"
    Blocked,    // cut_permission.status == "blocked" (or present but unusable: fail closed, see `malformed`)
    Unknown     // no cut_permission field (old snapshot): treated as allowed
  };
  State state = State::Unknown;
  std::string raw_status;                 // as written by the detector
  std::vector<std::string> reasons;       // "code: detail" / "code"
  bool has_override = false;              // detector's own override field is truthy
  std::string override_text;              // its JSON text
  bool malformed = false;                 // cut_permission present but not an object / no usable status
  std::string note;                       // why malformed
  std::string snapshot_id;                // of the document the permission was read from ("" if absent)
  uint64_t stamp_ns = 0;                  // its acquisition_stamp_ns (0 if absent)
  const char * stateName() const;         // "allowed" | "blocked" | "unknown"
};
/// Reads cut_permission from a snapshot document. Never throws; unparsable text -> Blocked + malformed.
DetectionPermission parseDetectionPermission(const std::string & snapshot_text);
/// Fail closed: the permission must belong to the snapshot the executors really planned with. Compares snapshot_id and
/// acquisition_stamp_ns of the document read at the start of planning with those of the exported plan (an absent value in the
/// document is not compared). A mismatch means the source file was replaced while planning ("source snapshot changed").
Status checkPermissionMatchesPlan(const DetectionPermission & p, const std::string & plan_snapshot_id, uint64_t plan_stamp_ns);
/// Truthiness of the detector's override value: true, a non-zero number, a non-empty string other than
/// false/none/no/off/0/null, a non-empty array / object.
bool overrideTruthy(const std::string & json_text);

struct ExecuteGateResult
{
  bool allowed = true;
  std::string by;        // "allowed" | "unknown_permission" | "detector_override" | "allow_blocked_detection" | "refused"
  std::string message;
};
/// Execute-time decision for the plan's snapshot permission.
///  allowed / unknown            -> allowed
///  blocked + allow_blocked_detection (runtime parameter)        -> allowed (said loudly)
///  blocked + detector override + honor_override                -> allowed (said loudly)
///  blocked otherwise                                           -> refused
ExecuteGateResult detectionExecuteGate(const DetectionPermission & p, bool allow_blocked_detection, bool honor_override);
/// JSON object for the plan report (members: permission, blocked, reasons, override, ...).
std::string detectionPermissionJson(const DetectionPermission & p, bool honor_override);

}  // namespace dual_cut
