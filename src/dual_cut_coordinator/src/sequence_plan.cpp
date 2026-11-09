#include "dual_cut_coordinator/sequence_plan.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace dual_cut
{
namespace
{
double maxAbsDiff(const std::vector<double> & a, const std::vector<double> & b)
{
  double m = 0.0;
  for (size_t i = 0; i < a.size() && i < b.size(); ++i) m = std::max(m, std::abs(a[i] - b[i]));
  return m;
}
}  // namespace

Status makeHoldSegment(
  const std::vector<std::string> & joint_names, const std::vector<double> & q, double hold_s, PlannedSegmentData & out)
{
  if (!(hold_s > 0.0) || !std::isfinite(hold_s)) return Status::Error("hold must be a positive finite time");
  if (joint_names.size() != q.size() || q.empty()) return Status::Error("hold: joint names / positions mismatch");
  out = PlannedSegmentData{};
  out.kind = SegmentKind::Dwell;
  out.stage_name = kSequenceHoldName;
  out.tool_carton_contact_allowed = false;
  out.trajectory.joint_names = joint_names;
  TrajPoint a;
  a.time_s = 0.0;
  a.positions = q;
  a.velocities.assign(q.size(), 0.0);
  a.accelerations.assign(q.size(), 0.0);
  TrajPoint b = a;
  b.time_s = hold_s;
  out.trajectory.points = {a, b};
  return Status::Ok();
}

namespace
{
// job1 may be null (single-job timeline).
Status join(const ArmPlan * job1p, const ArmPlan & job2, const PlanTolerances & tol, double hold_s, ArmPhase & out, double * job1_duration)
{
  out = ArmPhase{};
  if (!(hold_s >= 0.0) || !std::isfinite(hold_s)) return Status::Error("hold must be finite and >= 0");
  Status s = validateArmPlan(job2, tol);
  if (!s) return Status::Error(job2.side + " job: " + s.message);
  if (job1p)
  {
    const ArmPlan & job1 = *job1p;
    if (job1.side != job2.side) return Status::Error("jobs are for different arms");
    if (!(job1.snapshot == job2.snapshot)) return Status::Error("jobs use different snapshots");
    if (job1.joint_names != job2.joint_names) return Status::Error("jobs use different joints");
    s = validateArmPlan(job1, tol);
    if (!s) return Status::Error(job1.side + " job: " + s.message);
    const double gap = maxAbsDiff(job1.end_q, job2.start_q);
    if (!(gap <= tol.handoff_tol) || job1.end_q.size() != job2.start_q.size())
      return Status::Error(
        "job 2 does not start where job 1 ends (max |dq| = " + std::to_string(gap) + " rad, tol " +
        std::to_string(tol.handoff_tol) + ")");
    if (job2.planning_frame != job1.planning_frame) return Status::Error("jobs use different planning frames");
  }
  if (job1_duration) *job1_duration = job1p ? job1p->duration() : 0.0;
  out.side = job2.side;
  out.joint_names = job2.joint_names;
  out.planning_frame = job2.planning_frame;
  if (job1p) out.cut_targets = job1p->cut_targets;
  out.cut_targets.insert(out.cut_targets.end(), job2.cut_targets.begin(), job2.cut_targets.end());
  out.start_q = job1p ? job1p->start_q : job2.start_q;
  out.end_q = job2.end_q;
  if (job1p) out.segments = job1p->segments;
  // The hold goes JUST BEFORE PENETRATION: before job 2's first ENTRY segment, else before its first CUT
  // segment, else (nothing to enter) at the start of job 2. The hold pose is the start pose of that segment.
  size_t at = 0;
  if (hold_s > 0.0)
  {
    bool found = false;
    for (SegmentKind kind : {SegmentKind::Entry, SegmentKind::Cut})
    {
      for (size_t i = 0; i < job2.segments.size() && !found; ++i)
        if (job2.segments[i].kind == kind)
        {
          at = i;
          found = true;
        }
      if (found) break;
    }
  }
  out.segments.insert(out.segments.end(), job2.segments.begin(), job2.segments.begin() + static_cast<std::ptrdiff_t>(at));
  if (hold_s > 0.0)
  {
    PlannedSegmentData hold;
    const std::vector<double> & pose =
      (at == 0 && job1p) ? job1p->end_q : job2.segments[at].trajectory.points.front().positions;
    s = makeHoldSegment(job2.joint_names, pose, hold_s, hold);
    if (!s) return s;
    out.segments.push_back(std::move(hold));
  }
  out.segments.insert(out.segments.end(), job2.segments.begin() + static_cast<std::ptrdiff_t>(at), job2.segments.end());
  return Status::Ok();
}
}  // namespace

Status joinJobs(
  const ArmPlan & job1, const ArmPlan & job2, const PlanTolerances & tol, double hold_s, ArmPhase & out, double * job1_duration)
{
  return join(&job1, job2, tol, hold_s, out, job1_duration);
}

Status singleJob(const ArmPlan & job, const PlanTolerances & tol, double hold_s, ArmPhase & out)
{
  return join(nullptr, job, tol, hold_s, out, nullptr);
}

bool shouldRetryPlan(const std::string & refusal, const std::vector<std::string> & signatures)
{
  if (refusal.find("plan_h_cut_from_state failed") == std::string::npos) return false;
  for (const auto & sig : signatures)
    if (!sig.empty() && refusal.find(sig) != std::string::npos) return true;
  return false;
}

bool snapshotStampFromText(const std::string & text, uint64_t & stamp_ns)
{
  const std::string key = "\"acquisition_stamp_ns\"";
  size_t p = text.find(key);
  if (p == std::string::npos) return false;
  p = text.find(':', p + key.size());
  if (p == std::string::npos) return false;
  ++p;
  while (p < text.size() && (text[p] == ' ' || text[p] == '\t' || text[p] == '\n' || text[p] == '\r')) ++p;
  if (p >= text.size() || text[p] < '0' || text[p] > '9') return false;
  uint64_t v = 0;
  while (p < text.size() && text[p] >= '0' && text[p] <= '9') v = v * 10 + static_cast<uint64_t>(text[p++] - '0');
  stamp_ns = v;
  return true;
}

const char * const kSequenceHoldName = "sequence delay hold";

std::vector<SegWindow> segmentWindows(const ArmPhase & arm, double margin_s)
{
  std::vector<SegWindow> out;
  double t = 0.0;
  for (size_t i = 0; i < arm.segments.size(); ++i)
  {
    const auto & pts = arm.segments[i].trajectory.points;
    const double d = pts.empty() ? 0.0 : pts.back().time_s - pts.front().time_s;
    SegWindow w;
    w.index = i;
    w.name = arm.segments[i].stage_name;
    w.start_s = t;
    w.end_s = t + d;
    w.lo_s = w.start_s - margin_s;
    w.hi_s = w.end_s + margin_s;
    out.push_back(std::move(w));
    t += d;
  }
  SegWindow tail;
  tail.index = arm.segments.size();
  tail.name = "tail hold (timeline over)";
  tail.tail = true;
  tail.start_s = tail.end_s = t;
  tail.lo_s = t - margin_s;
  tail.hi_s = std::numeric_limits<double>::infinity();
  out.push_back(std::move(tail));
  return out;
}

bool windowsOverlap(const SegWindow & a, const SegWindow & b) { return a.lo_s <= b.hi_s && b.lo_s <= a.hi_s; }

SkewAction skewDecision(double now_s, double planned_start_s, double planned_end_s, double margin_s, bool started, bool completed)
{
  if (completed) return SkewAction::Wait;   // nothing to do for this segment any more
  if (now_s > planned_end_s + margin_s) return SkewAction::Stop;
  if (!started) return now_s >= planned_start_s ? SkewAction::Dispatch : SkewAction::Wait;
  return SkewAction::Wait;                  // running: waiting for its result
}

std::vector<DelayCandidate> delayCandidates(double step_s, double max_s)
{
  std::vector<DelayCandidate> out;
  out.push_back(DelayCandidate{"", 0.0});
  if (!(step_s > 0.0) || !(max_s >= step_s)) return out;
  const int n = static_cast<int>(std::floor(max_s / step_s + 1e-9));
  for (const char * arm : {"left", "right"})
    for (int k = 1; k <= n; ++k) out.push_back(DelayCandidate{arm, k * step_s});
  return out;
}

}  // namespace dual_cut
