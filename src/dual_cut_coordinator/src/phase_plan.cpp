#include "dual_cut_coordinator/phase_plan.hpp"

#include <algorithm>
#include <cmath>
#include <map>

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

// Reorder the columns of a trajectory to `names` (same set required).
Status reorderColumns(JointTrajectory & traj, const std::vector<std::string> & names)
{
  if (traj.joint_names == names) return Status::Ok();
  if (traj.joint_names.size() != names.size())
    return Status::Error("joint count differs from the plan's joints");
  std::vector<size_t> src(names.size());
  for (size_t i = 0; i < names.size(); ++i)
  {
    auto it = std::find(traj.joint_names.begin(), traj.joint_names.end(), names[i]);
    if (it == traj.joint_names.end()) return Status::Error("joint '" + names[i] + "' missing");
    src[i] = static_cast<size_t>(it - traj.joint_names.begin());
  }
  auto perm = [&](std::vector<double> & v)
  {
    if (v.empty()) return;
    std::vector<double> w(v.size());
    for (size_t i = 0; i < src.size(); ++i) w[i] = v[src[i]];
    v = std::move(w);
  };
  for (auto & p : traj.points)
  {
    perm(p.positions);
    perm(p.velocities);
    perm(p.accelerations);
  }
  traj.joint_names = names;
  return Status::Ok();
}

Status stateInOrder(
  const sensor_msgs::msg::JointState & js, const std::vector<std::string> & names,
  std::vector<double> & out)
{
  if (js.name.size() != js.position.size()) return Status::Error("JointState name/position size");
  out.assign(names.size(), 0.0);
  for (size_t i = 0; i < names.size(); ++i)
  {
    auto it = std::find(js.name.begin(), js.name.end(), names[i]);
    if (it == js.name.end()) return Status::Error("JointState lacks joint '" + names[i] + "'");
    out[i] = js.position[static_cast<size_t>(it - js.name.begin())];
    if (!std::isfinite(out[i])) return Status::Error("JointState has a non-finite position");
  }
  if (js.name.size() != names.size()) return Status::Error("JointState has extra joints");
  return Status::Ok();
}

}  // namespace

const char * toString(SegmentKind kind)
{
  switch (kind)
  {
    case SegmentKind::Transit: return "TRANSIT";
    case SegmentKind::Align: return "ALIGN";
    case SegmentKind::Dwell: return "DWELL";
    case SegmentKind::Entry: return "ENTRY";
    case SegmentKind::Cut: return "CUT";
    case SegmentKind::Finish: return "FINISH";
    case SegmentKind::Lift: return "LIFT";
  }
  return "?";
}

Status segmentKindFromUint8(uint8_t value, SegmentKind & out)
{
  if (value > static_cast<uint8_t>(SegmentKind::Lift))
    return Status::Error("unknown segment kind " + std::to_string(value));
  out = static_cast<SegmentKind>(value);
  return Status::Ok();
}

double ArmPlan::duration() const
{
  double d = 0.0;
  for (const auto & s : segments)
    if (!s.trajectory.points.empty()) d += s.trajectory.points.back().time_s - s.trajectory.points.front().time_s;
  return d;
}

Status fromMsg(const dual_cut_msgs::msg::PlannedSegment & msg, PlannedSegmentData & out)
{
  out = PlannedSegmentData{};
  Status s = segmentKindFromUint8(msg.kind, out.kind);
  if (!s) return s;
  if (!msg.trajectory.multi_dof_joint_trajectory.points.empty())
    return Status::Error("segment '" + msg.stage_name + "' has a multi-DOF trajectory");
  out.stage_name = msg.stage_name;
  out.cut_id = msg.cut_id;
  out.physical_cut = msg.physical_cut;
  out.tool_carton_contact_allowed = msg.tool_carton_contact_allowed;
  out.trajectory = fromMsg(msg.trajectory.joint_trajectory);
  s = validate(out.trajectory);
  if (!s) return Status::Error("segment '" + msg.stage_name + "': " + s.message);
  return Status::Ok();
}

Status fromResponse(
  const std::string & side, const dual_cut_msgs::srv::GetPlannedSolution::Response & resp,
  ArmPlan & out)
{
  out = ArmPlan{};
  if (armPrefix(side).empty()) return Status::Error("side must be 'left' or 'right'");
  if (!resp.success) return Status::Error("executor reported failure: " + resp.message);
  out.side = side;
  out.plan_seq = resp.plan_seq;
  out.snapshot.snapshot_id = resp.snapshot_id;
  out.snapshot.capture_id = resp.capture_id;
  out.snapshot.acquisition_stamp_ns = resp.acquisition_stamp_ns;
  out.snapshot.stale_override = resp.stale_override;
  out.planning_frame = resp.planning_frame;
  out.tool_link = resp.tool_link;
  out.cut_pattern = resp.cut_pattern;
  out.snapshot_frame_mode = resp.snapshot_frame_mode;
  out.snapshot_frame_id = resp.snapshot_frame_id;
  out.snapshot_relabel = resp.snapshot_relabel;
  for (size_t i = 0; i < 3; ++i)
  {
    out.base_xyz[i] = resp.snapshot_planning_base_xyz[i];
    out.base_rpy[i] = resp.snapshot_planning_base_rpy[i];
  }
  out.middle_cut_fraction = resp.middle_cut_fraction;
  out.cut_targets = resp.cut_targets;
  out.joint_names = resp.start_state.name;
  Status s = stateInOrder(resp.start_state, out.joint_names, out.start_q);
  if (!s) return Status::Error("start_state: " + s.message);
  s = stateInOrder(resp.end_state, out.joint_names, out.end_q);
  if (!s) return Status::Error("end_state: " + s.message);
  for (const auto & seg : resp.segments)
  {
    PlannedSegmentData d;
    s = fromMsg(seg, d);
    if (!s) return s;
    s = reorderColumns(d.trajectory, out.joint_names);
    if (!s) return Status::Error("segment '" + seg.stage_name + "': " + s.message);
    out.segments.push_back(std::move(d));
  }
  return validateArmPlan(out, PlanTolerances{});
}

Status validateArmPlan(const ArmPlan & plan, const PlanTolerances & tol)
{
  if (armPrefix(plan.side).empty()) return Status::Error("plan side must be 'left' or 'right'");
  if (plan.joint_names.empty()) return Status::Error("plan has no joint names");
  if (plan.segments.empty()) return Status::Error("plan has no segments");
  if (plan.start_q.size() != plan.joint_names.size() || plan.end_q.size() != plan.joint_names.size())
    return Status::Error("start_q / end_q size does not match joint_names");
  for (size_t i = 0; i < plan.segments.size(); ++i)
  {
    const auto & seg = plan.segments[i];
    const std::string where = "segment " + std::to_string(i) + " '" + seg.stage_name + "': ";
    Status s = validate(seg.trajectory);
    if (!s) return Status::Error(where + s.message);
    if (seg.trajectory.joint_names != plan.joint_names)
      return Status::Error(where + "joint names differ from the plan's");
    if (i > 0)
    {
      const double d = maxAbsDiff(
        plan.segments[i - 1].trajectory.points.back().positions,
        seg.trajectory.points.front().positions);
      if (!(d <= tol.chain_tol))
        return Status::Error(where + "does not start where the previous segment ends");
    }
  }
  if (!(maxAbsDiff(plan.start_q, plan.segments.front().trajectory.points.front().positions) <=
        tol.chain_tol))
    return Status::Error("start_state differs from the first segment's start");
  if (!(maxAbsDiff(plan.end_q, plan.segments.back().trajectory.points.back().positions) <=
        tol.chain_tol))
    return Status::Error("end_state differs from the last segment's end");
  return Status::Ok();
}

Status shiftLeadingTransits(ArmPlan & a, ArmPlan & b, const PlanTolerances & tol, size_t * moved)
{
  if (moved) *moved = 0;
  if (a.side != b.side) return Status::Error("plans are for different arms");
  if (!(a.snapshot == b.snapshot)) return Status::Error("plans A and B use different snapshots");
  if (a.joint_names != b.joint_names) return Status::Error("plans A and B use different joints");
  if (a.end_q.size() != b.start_q.size())
    return Status::Error("A.end_q / B.start_q size mismatch");
  const double gap = maxAbsDiff(a.end_q, b.start_q);
  if (!(gap <= tol.handoff_tol))
    return Status::Error(
      "B does not start where A ends (max |dq| = " + std::to_string(gap) + " rad, tol " +
      std::to_string(tol.handoff_tol) + ")");

  size_t n = 0;
  while (n < b.segments.size() && b.segments[n].kind == SegmentKind::Transit) ++n;
  if (n == b.segments.size())
    return Status::Error("plan B consists only of transit segments: nothing would be left for B");

  if (n > 0)
  {
    a.segments.insert(
      a.segments.end(), std::make_move_iterator(b.segments.begin()),
      std::make_move_iterator(b.segments.begin() + static_cast<std::ptrdiff_t>(n)));
    b.segments.erase(b.segments.begin(), b.segments.begin() + static_cast<std::ptrdiff_t>(n));
    a.end_q = a.segments.back().trajectory.points.back().positions;
    b.start_q = b.segments.front().trajectory.points.front().positions;
  }
  if (moved) *moved = n;
  return Status::Ok();
}

double ArmPhase::duration() const
{
  double d = 0.0;
  for (const auto & s : segments)
    if (!s.trajectory.points.empty()) d += s.trajectory.points.back().time_s - s.trajectory.points.front().time_s;
  return d;
}

Status ArmPhase::prefixedSegments(std::vector<SegmentInput> & out) const
{
  out.clear();
  const std::string prefix = armPrefix(side);
  if (prefix.empty()) return Status::Error("side must be 'left' or 'right'");
  for (const auto & seg : segments)
  {
    SegmentInput in;
    in.trajectory = seg.trajectory;
    in.contact_allowed = seg.tool_carton_contact_allowed;
    Status s = prefixTrajectory(prefix, in.trajectory);
    if (!s) return s;
    out.push_back(std::move(in));
  }
  return Status::Ok();
}

Phase::Phase(std::string name, ArmPhase left, ArmPhase right)
: name_(std::move(name)), left_(std::move(left)), right_(std::move(right))
{
}

const ArmPhase & Phase::arm(const std::string & side) const { return side == "left" ? left_ : right_; }

ArmPhase & Phase::mutableLeft()
{
  ++revision_;
  validated_ = false;
  recorded_ = false;
  return left_;
}

ArmPhase & Phase::mutableRight()
{
  ++revision_;
  validated_ = false;
  recorded_ = false;
  return right_;
}

double Phase::duration() const { return std::max(left_.duration(), right_.duration()); }

void Phase::recordValidation(const ValidationRecord & rec, bool all_ok)
{
  validation_ = rec;
  validation_.revision = revision_;
  validated_ = all_ok;
  recorded_ = true;
}

Status bothArmsTrajectory(const Phase & phase, double dt_s, trajectory_msgs::msg::JointTrajectory & out)
{
  out = trajectory_msgs::msg::JointTrajectory();
  if (!(dt_s > 0.0)) return Status::Error("dt must be positive");
  JointTrajectory traj[2];
  const ArmPhase * arms[2] = {&phase.left(), &phase.right()};
  for (int k = 0; k < 2; ++k)
  {
    std::vector<JointTrajectory> segs;
    const std::string prefix = armPrefix(arms[k]->side);
    for (const auto & seg : arms[k]->segments)
    {
      JointTrajectory t = seg.trajectory;
      Status s = prefixTrajectory(prefix, t);
      if (!s) return s;
      segs.push_back(std::move(t));
    }
    if (segs.empty()) return Status::Error("phase has an arm without segments");
    Status s = concatenate(segs, ConcatOptions{}, traj[k]);
    if (!s) return s;
  }
  const bool serial = phase.validation().serial_fallback;
  const bool left_first = phase.validation().serial_first != "right";
  double offset[2] = {0.0, 0.0};
  double total = std::max(traj[0].duration(), traj[1].duration());
  if (serial)
  {
    const int first = left_first ? 0 : 1;
    offset[1 - first] = traj[first].duration();
    total = traj[0].duration() + traj[1].duration();
  }
  out.joint_names = traj[0].joint_names;
  out.joint_names.insert(out.joint_names.end(), traj[1].joint_names.begin(), traj[1].joint_names.end());
  const size_t n = static_cast<size_t>(std::floor(total / dt_s));
  for (size_t i = 0; i <= n + 1; ++i)
  {
    const double t = std::min(static_cast<double>(i) * dt_s, total);
    trajectory_msgs::msg::JointTrajectoryPoint pt;
    for (int k = 0; k < 2; ++k)
    {
      TrajPoint tp;
      Status s = sampleAt(traj[k], std::clamp(t - offset[k], 0.0, traj[k].duration()), tp);
      if (!s) return s;
      pt.positions.insert(pt.positions.end(), tp.positions.begin(), tp.positions.end());
    }
    const long long ns = std::llround(t * 1e9);
    pt.time_from_start.sec = static_cast<int32_t>(ns / 1000000000LL);
    pt.time_from_start.nanosec = static_cast<uint32_t>(ns % 1000000000LL);
    out.points.push_back(std::move(pt));
    if (t >= total) break;
  }
  return Status::Ok();
}

Status buildPhases(
  ArmPlan left_a, ArmPlan left_b, ArmPlan right_a, ArmPlan right_b, const PlanTolerances & tol,
  PhaseSet & out)
{
  out = PhaseSet{};
  if (left_a.side != "left" || left_b.side != "left" || right_a.side != "right" ||
      right_b.side != "right")
    return Status::Error("plan sides are not (left, left, right, right)");
  for (const ArmPlan * p : {&left_a, &left_b, &right_a, &right_b})
  {
    Status s = validateArmPlan(*p, tol);
    if (!s) return Status::Error(p->side + " plan: " + s.message);
  }
  if (!(left_a.snapshot == right_a.snapshot) || !(left_a.snapshot == left_b.snapshot) ||
      !(left_a.snapshot == right_b.snapshot))
    return Status::Error("the four plans do not all come from the same snapshot");

  size_t ml = 0;
  size_t mr = 0;
  Status s = shiftLeadingTransits(left_a, left_b, tol, &ml);
  if (!s) return Status::Error("left: " + s.message);
  s = shiftLeadingTransits(right_a, right_b, tol, &mr);
  if (!s) return Status::Error("right: " + s.message);

  auto armPhase = [](const ArmPlan & p)
  {
    ArmPhase ph;
    ph.side = p.side;
    ph.joint_names = p.joint_names;
    ph.planning_frame = p.planning_frame;
    ph.cut_targets = p.cut_targets;
    ph.start_q = p.start_q;
    ph.end_q = p.end_q;
    ph.segments = p.segments;
    return ph;
  };
  out.a = Phase("A", armPhase(left_a), armPhase(right_a));
  out.b = Phase("B", armPhase(left_b), armPhase(right_b));
  out.moved_left = ml;
  out.moved_right = mr;
  out.snapshot = left_a.snapshot;
  return Status::Ok();
}

}  // namespace dual_cut
