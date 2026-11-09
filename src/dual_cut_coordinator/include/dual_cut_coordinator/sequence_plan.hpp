// "sequence" schedule mode (pure logic): each arm runs its own jobs back to back on one timeline, the arms
// run in parallel, the execution dispatches every segment as its own goal at its planned start (with a skew
// gate) and the cross-arm check is SEGMENT-WINDOW ALL-PAIRS (PhaseValidator::crossWindows): every left
// segment against every right segment whose time windows overlap, no time assumption inside a segment.
//
//   joinJobs         job1 + [hold] + job2 of one arm -> one ArmPhase (the hold is a stationary DWELL
//                    segment at job1's end pose, i.e. the "delay" of that arm's second job)
//   segmentWindows   per segment [planned start - margin, planned end + margin] + the tail hold (END CLAMPING)
//   skewDecision     the execution's late/early rule for one segment
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "dual_cut_coordinator/phase_plan.hpp"
#include "dual_cut_coordinator/trajectory_tools.hpp"

namespace dual_cut
{

/// Stationary DWELL segment of hold_s seconds at q (zero velocities and accelerations at both ends).
Status makeHoldSegment(
  const std::vector<std::string> & joint_names, const std::vector<double> & q, double hold_s,
  PlannedSegmentData & out);

/// job1 followed by job2 (the hold, if any, sits just before job 2's first ENTRY segment, else its first CUT, else
/// at job 2's start) (same arm, same snapshot, same joints, job1.end_q == job2.start_q within
/// tol.handoff_tol; both pass validateArmPlan). hold_s > 0 inserts a stationary hold segment (DWELL, kSequenceHoldName) at that point;
/// hold_s == 0 inserts nothing. job1_duration (optional) receives the duration of job1.
/// The result has start_q = job1.start_q, end_q = job2.end_q, cut_targets = job1's then job2's.
Status joinJobs(
  const ArmPlan & job1, const ArmPlan & job2, const PlanTolerances & tol, double hold_s, ArmPhase & out,
  double * job1_duration = nullptr);

/// Same for a timeline of ONE job (the right top is skipped): the hold goes before the first ENTRY / CUT of the job,
/// else at its start.
Status singleJob(const ArmPlan & job, const PlanTolerances & tol, double hold_s, ArmPhase & out);

/// Automatic re-plan decision (all schedule modes). A refusal is retried only if it is an EXECUTOR plan-stage failure
/// (the message carries "plan_h_cut_from_state failed") that contains one of the known random signatures. Validator
/// refusals (world, cross-arm, coverage, frame, snapshot gates ...) and every other message are never retried.
bool shouldRetryPlan(const std::string & refusal, const std::vector<std::string> & signatures);

/// acquisition_stamp_ns of a snapshot document (text scan, no JSON library); false if absent.
bool snapshotStampFromText(const std::string & text, uint64_t & stamp_ns);

/// Name of the stationary DWELL segment inserted by joinJobs (the execution does not dispatch it).
extern const char * const kSequenceHoldName;

/// Time window of one segment of an arm timeline for the SEGMENT-WINDOW cross-arm check.
/// A segment planned at [start, end] may really be running anywhere in [start - margin, end + margin]
/// (the execution stops both arms if a segment is later than that). The extra TAIL entry (last, index ==
/// number of segments, tail == true) is the arm standing still at its final pose after its timeline is
/// over (END CLAMPING): window [end - margin, +inf).
struct SegWindow
{
  size_t index = 0;
  std::string name;
  bool tail = false;
  double start_s = 0.0, end_s = 0.0;   // planned, on the arm's own timeline
  double lo_s = 0.0, hi_s = 0.0;       // window
};
std::vector<SegWindow> segmentWindows(const ArmPhase & arm, double margin_s);
/// Closed intervals: touching windows overlap.
bool windowsOverlap(const SegWindow & a, const SegWindow & b);

enum class SkewAction
{
  Wait,       // not yet time to start (early): the arm holds its pose
  Dispatch,   // start the segment now
  Stop        // fail closed: STOP both arms
};
/// Pure rule of the execution's skew gate. now_s/planned_*: seconds on the common clock (t0 = 0).
///   a segment that has not been started waits until its planned start, then is dispatched;
///   a segment that is not complete by planned_end + margin (running or not yet started) -> Stop.
SkewAction skewDecision(double now_s, double planned_start_s, double planned_end_s, double margin_s, bool started, bool completed);

/// Candidate delays in search order: 0, then (left, k*step) for k = 1..N, then (right, k*step), N =
/// floor(max/step). arm is "" / "left" / "right".
struct DelayCandidate
{
  std::string arm;
  double delay_s = 0.0;
};
std::vector<DelayCandidate> delayCandidates(double step_s, double max_s);

}  // namespace dual_cut
