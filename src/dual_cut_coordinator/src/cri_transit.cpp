#include "dual_cut_coordinator/cri_transit.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace dual_cut
{

namespace
{
constexpr double kInf = std::numeric_limits<double>::infinity();

std::string num(double v)
{
  std::ostringstream o;
  o << v;
  return o.str();
}

bool allFinite(const std::vector<double> & v)
{
  for (double x : v)
    if (!std::isfinite(x)) return false;
  return true;
}
}  // namespace

// ---- parameters -------------------------------------------------------------------------------

Status validateCriParameter(const std::string & name, double v)
{
  auto bad = [&](const std::string & range) { return Status::Error(name + " must be finite and within " + range + " (rejected, not clamped)"); };
  if (!std::isfinite(v)) return bad("its range");
  if (name == "cri_transit_velocity_percent") { if (!(v > 0.0 && v <= 25.0)) return bad("(0, 25]"); }
  else if (name == "cri_transit_acceleration_percent") { if (!(v > 0.0 && v <= 40.0)) return bad("(0, 40]"); }
  else if (name == "cri_transit_min_length_rad") { if (!(v >= 0.01 && v <= 3.2)) return bad("[0.01, 3.2]"); }
  else if (name == "cri_transit_path_tolerance_rad") { if (!(v > 0.0 && v <= 0.05)) return bad("(0, 0.05]"); }
  else if (name == "cri_transit_goal_tolerance_rad") { if (!(v > 0.0 && v <= 0.05)) return bad("(0, 0.05]"); }
  else if (name == "cri_transit_start_tolerance_rad") { if (!(v > 0.0 && v <= 0.05)) return bad("(0, 0.05]"); }
  else if (name == "cri_driver_max_velocity_cap_percent") { if (!(v > 0.0 && v <= 100.0)) return bad("(0, 100]"); }
  else if (name == "cri_driver_max_acceleration_cap_percent") { if (!(v > 0.0 && v <= 40.0)) return bad("(0, 40]"); }
  else if (name == "cri_subchord_velocity_rad_s") { if (!(v > 0.0 && v <= 0.785)) return bad("(0, 0.785] rad/s"); }
  else if (name == "cri_subchord_acceleration_rad_s2") { if (!(v > 0.0 && v <= 5.24)) return bad("(0, 5.24] rad/s^2"); }
  else if (name == "cri_transit_reach_m") { if (!(v == 0.0 || (v >= 0.1 && v <= 5.0))) return bad("{0 (derive)} or [0.1, 5] m"); }
  return Status::Ok();
}

Status validateCriParameter(const std::string & name, int64_t v)
{
  if (name == "cri_transit_max_runs_per_arm_phase" && !(v >= 1 && v <= 4))
    return Status::Error(name + " must be within [1, 4] (rejected, not clamped)");
  return Status::Ok();
}

Status validateCriParams(const CriParams & p)
{
  struct D
  {
    const char * name;
    double v;
  };
  for (const D & d : {D{"cri_transit_velocity_percent", p.velocity_percent}, D{"cri_transit_acceleration_percent", p.acceleration_percent},
                      D{"cri_transit_min_length_rad", p.min_length_rad}, D{"cri_transit_path_tolerance_rad", p.path_tolerance_rad},
                      D{"cri_transit_goal_tolerance_rad", p.goal_tolerance_rad}, D{"cri_transit_start_tolerance_rad", p.start_tolerance_rad},
                      D{"cri_driver_max_velocity_cap_percent", p.driver_max_velocity_cap_percent},
                      D{"cri_driver_max_acceleration_cap_percent", p.driver_max_acceleration_cap_percent}, D{"cri_transit_reach_m", p.reach_m},
                      D{"cri_subchord_velocity_rad_s", p.subchord_velocity_rad_s}, D{"cri_subchord_acceleration_rad_s2", p.subchord_acceleration_rad_s2}})
  {
    const Status s = validateCriParameter(d.name, d.v);
    if (!s) return s;
  }
  return validateCriParameter("cri_transit_max_runs_per_arm_phase", static_cast<int64_t>(p.max_runs_per_arm_phase));
}

// ---- stop travel, padding, duration ----------------------------------------------------------

double criStopTravelRad(double velocity_percent)
{
  return velocity_percent * kCriMeasuredRadPerSecPerVelPercent * kCriStopDecelS + kCriStopExtraRad;
}

double criPaddingM(double path_tolerance_rad, double reach_m) { return path_tolerance_rad * reach_m; }

PlannedSegmentData makeChordSegment(
  const std::vector<std::string> & joint_names, const std::vector<double> & from, const std::vector<double> & to, double v_lim, double a_lim,
  const std::string & stage_name)
{
  PlannedSegmentData seg;
  seg.kind = SegmentKind::Transit;
  seg.stage_name = stage_name;
  if (from.size() != to.size() || from.size() != joint_names.size() || from.empty() || !allFinite(from) || !allFinite(to) || !(v_lim > 0.0) ||
      !(a_lim > 0.0) || !std::isfinite(v_lim) || !std::isfinite(a_lim))
    return seg;
  const double d = maxAbsDiff(from, to);
  if (!(d > 0.0)) return seg;
  // s(tau) = 10 tau^3 - 15 tau^4 + 6 tau^5: peak s' = 1.875, peak s'' = 5.7735 (per unit of D / T and D / T^2)
  const double t = std::max({1.875 * d / v_lim, std::sqrt(5.7735 * d / a_lim), 0.2});
  seg.trajectory.joint_names = joint_names;
  for (int end = 0; end < 2; ++end)
  {
    TrajPoint p;
    p.time_s = end ? t : 0.0;
    p.positions = end ? to : from;
    p.velocities.assign(from.size(), 0.0);
    p.accelerations.assign(from.size(), 0.0);
    seg.trajectory.points.push_back(std::move(p));
  }
  return seg;
}

double criModelDurationS(double max_delta_rad, double velocity_percent, double acceleration_percent)
{
  const double distance = std::fabs(max_delta_rad);
  const double cruise = kCriModelRadPerSecPerVelPercent * std::max(velocity_percent, 1e-6);
  const double accel = kCriModelAccelRadS2AtAcc40 * std::max(acceleration_percent, 1e-6) / 40.0;
  const double ramp = cruise * cruise / accel;
  const double motion = distance >= ramp ? distance / cruise + cruise / accel : 2.0 * std::sqrt(distance / accel);
  return kCriModelStartLatencyS + motion;
}

double criTimeoutS(double model_s) { return kCriTimeoutFactor * model_s + kCriTimeoutMarginS; }

namespace
{
Status reachOfLink(const moveit::core::LinkModel & link, double & out)
{
  double best = 0.0;
  Sphere s;
  bool has = false;
  Status st = linkBoundingSphere(link, s, has);
  if (!st) return st;
  if (has) best = s.center.norm() + s.radius;
  for (const moveit::core::JointModel * cj : link.getChildJointModels())
  {
    const moveit::core::LinkModel * child = cj->getChildLinkModel();
    double sub = 0.0;
    st = reachOfLink(*child, sub);
    if (!st) return st;
    best = std::max(best, child->getJointOriginTransform().translation().norm() + sub);
  }
  out = best;
  return Status::Ok();
}
}  // namespace

Status armReachBound(const moveit::core::RobotModel & model, const ArmSpec & arm, double & sum_m, std::vector<double> * per_joint)
{
  sum_m = 0.0;
  if (per_joint) per_joint->clear();
  if (arm.joint_names.empty()) return Status::Error("arm has no joints");
  for (const auto & name : arm.joint_names)
  {
    if (!model.hasJointModel(name)) return Status::Error("reach bound: unknown joint " + name);
    const moveit::core::LinkModel * child = model.getJointModel(name)->getChildLinkModel();
    double r = 0.0;
    const Status st = reachOfLink(*child, r);
    if (!st) return Status::Error("reach bound: " + st.message);
    sum_m += r;
    if (per_joint) per_joint->push_back(r);
  }
  return Status::Ok();
}

// ---- eligibility ------------------------------------------------------------------------------

namespace
{
struct ChordTest
{
  bool ok = true;
  std::string reason;
  double max_dev = 0.0;
};

// Every point on the chord within tol, projection monotone (never goes back by more than tol_progress) and not beyond the
// ends by more than tol.
ChordTest testChordPoints(
  const std::vector<std::vector<double>> & pts, const std::vector<double> & first, const std::vector<double> & delta, double length, double tol,
  const char * what)
{
  ChordTest r;
  double dd = 0.0;
  for (double d : delta) dd += d * d;
  double prev_t = -kInf;
  const double progress_slack = tol / length;
  for (size_t k = 0; k < pts.size(); ++k)
  {
    const auto & p = pts[k];
    double dot = 0.0;
    for (size_t j = 0; j < p.size(); ++j) dot += (p[j] - first[j]) * delta[j];
    const double t = dot / dd;
    double dev = 0.0;
    for (size_t j = 0; j < p.size(); ++j) dev = std::max(dev, std::abs(p[j] - (first[j] + t * delta[j])));
    r.max_dev = std::max(r.max_dev, dev);
    if (!(dev <= tol))
    {
      r.ok = false;
      r.reason = std::string(what) + " " + std::to_string(k) + " is " + num(dev) + " rad off the straight chord (limit " + num(tol) + ")";
      return r;
    }
    if (t < -progress_slack || t > 1.0 + progress_slack)
    {
      r.ok = false;
      r.reason = std::string(what) + " " + std::to_string(k) + " lies beyond the chord ends (progress " + num(t) + ")";
      return r;
    }
    if (t < prev_t - progress_slack)
    {
      r.ok = false;
      r.reason = std::string(what) + " " + std::to_string(k) + " goes back along the chord (progress " + num(t) + " after " + num(prev_t) + ")";
      return r;
    }
    prev_t = std::max(prev_t, t);
  }
  return r;
}

// last point of `seg` (positions in `names` order) vs `ref` (in ref_names order)
bool jointsMatch(
  const std::vector<std::string> & names, const std::vector<double> & q, const std::vector<std::string> & ref_names, const std::vector<double> & ref,
  double tol, double & err)
{
  if (names.size() != q.size() || ref_names.size() != ref.size()) return false;
  err = 0.0;
  for (size_t i = 0; i < names.size(); ++i)
  {
    const auto it = std::find(ref_names.begin(), ref_names.end(), names[i]);
    if (it == ref_names.end()) return false;
    err = std::max(err, std::abs(q[i] - ref[static_cast<size_t>(it - ref_names.begin())]));
  }
  return err <= tol;
}
}  // namespace

CriLegGeometry checkTransitGeometry(const ArmPhase & arm, size_t index, const CriGeomLimits & lim)
{
  CriLegGeometry g;
  g.segment_index = index;
  auto no = [&](const std::string & why)
  {
    g.eligible = false;
    g.reason = why;
    return g;
  };
  if (index >= arm.segments.size()) return no("segment index out of range");
  const PlannedSegmentData & seg = arm.segments[index];
  g.stage_name = seg.stage_name;
  if (seg.kind != SegmentKind::Transit) return no(std::string("segment kind is ") + toString(seg.kind) + ", not Transit");
  const JointTrajectory & tr = seg.trajectory;
  if (tr.joint_names.size() != lim.joint_count)
    return no("moves " + std::to_string(tr.joint_names.size()) + " joints, not exactly " + std::to_string(lim.joint_count));
  if (tr.joint_names != arm.joint_names) return no("joint order differs from the arm phase's");
  const Status vs = validate(tr);
  if (!vs) return no("trajectory invalid: " + vs.message);
  if (tr.points.size() < 2) return no("fewer than two waypoints");
  const std::vector<double> & first = tr.points.front().positions;
  const std::vector<double> & last = tr.points.back().positions;
  if (first.size() != lim.joint_count || last.size() != lim.joint_count || !allFinite(first) || !allFinite(last)) return no("end points are not finite 6-joint positions");
  std::vector<double> delta(first.size());
  double length = 0.0;
  for (size_t j = 0; j < first.size(); ++j)
  {
    delta[j] = last[j] - first[j];
    length = std::max(length, std::abs(delta[j]));
  }
  g.start = first;
  g.goal = last;
  g.joint_names = tr.joint_names;
  g.length_rad = length;
  if (!(length >= lim.min_length_rad)) return no("chord length " + num(length) + " rad < minimum " + num(lim.min_length_rad) + " rad");

  // exported waypoints
  std::vector<std::vector<double>> wp;
  for (const auto & p : tr.points) wp.push_back(p.positions);
  ChordTest ct = testChordPoints(wp, first, delta, length, lim.chord_tolerance_rad, "waypoint");
  g.max_chord_deviation_rad = ct.max_dev;
  if (!ct.ok) return no(ct.reason);
  // the JTC interpolates between the waypoints with its own splines: the sampled path must be on the chord too
  std::vector<TrajPoint> samples;
  SampleOptions so;
  so.dt_s = 0.05;
  so.max_joint_step_rad = kCriResampleStepRad;
  const Status ss = sampleTrajectory(tr, so, samples);
  if (!ss) return no("cannot sample the trajectory: " + ss.message);
  std::vector<std::vector<double>> sp;
  for (const auto & p : samples) sp.push_back(p.positions);
  ct = testChordPoints(sp, first, delta, length, lim.chord_tolerance_rad, "sampled point");
  g.max_chord_deviation_rad = std::max(g.max_chord_deviation_rad, ct.max_dev);
  if (!ct.ok) return no(ct.reason);

  // continuity with the neighbours (the previous end / next start are the chord ends)
  double err = 0.0;
  if (index == 0)
  {
    if (!jointsMatch(tr.joint_names, first, arm.joint_names, arm.start_q, lim.goal_tolerance_rad, err))
      return no("chord start differs from the phase start by " + num(err) + " rad (limit " + num(lim.goal_tolerance_rad) + ")");
  }
  else
  {
    const JointTrajectory & pt = arm.segments[index - 1].trajectory;
    if (pt.points.empty() || !jointsMatch(tr.joint_names, first, pt.joint_names, pt.points.back().positions, lim.goal_tolerance_rad, err))
      return no("chord start differs from the previous segment's end by " + num(err) + " rad (limit " + num(lim.goal_tolerance_rad) + ")");
  }
  if (index + 1 == arm.segments.size())
  {
    if (!jointsMatch(tr.joint_names, last, arm.joint_names, arm.end_q, lim.goal_tolerance_rad, err))
      return no("chord end differs from the phase end by " + num(err) + " rad (limit " + num(lim.goal_tolerance_rad) + ")");
  }
  else
  {
    const JointTrajectory & nt = arm.segments[index + 1].trajectory;
    if (nt.points.empty() || !jointsMatch(tr.joint_names, last, nt.joint_names, nt.points.front().positions, lim.goal_tolerance_rad, err))
      return no("chord end differs from the next segment's start by " + num(err) + " rad (limit " + num(lim.goal_tolerance_rad) + ")");
  }
  g.eligible = true;
  g.reason.clear();
  return g;
}

std::vector<CriLegGeometry> scanTransitLegs(const ArmPhase & arm, const CriGeomLimits & limits)
{
  std::vector<CriLegGeometry> out;
  for (size_t i = 0; i < arm.segments.size(); ++i)
    if (arm.segments[i].kind == SegmentKind::Transit) out.push_back(checkTransitGeometry(arm, i, limits));
  return out;
}

std::vector<std::vector<double>> resampleChord(const std::vector<double> & start, const std::vector<double> & goal, double max_step_rad)
{
  std::vector<std::vector<double>> out;
  if (start.size() != goal.size() || start.empty() || !allFinite(start) || !allFinite(goal)) return out;
  const double step = max_step_rad > 0.0 ? max_step_rad : kCriResampleStepRad;
  const double len = maxAbsDiff(start, goal);
  if (!(len / step < 1e6)) return out;   // absurd length / step: fail closed
  size_t n = len > 0.0 ? static_cast<size_t>(std::ceil(len / step - 1e-9)) : 1;
  if (n < 1) n = 1;
  out.reserve(n + 1);
  for (size_t k = 0; k <= n; ++k)
  {
    const double t = static_cast<double>(k) / static_cast<double>(n);
    std::vector<double> q(start.size());
    for (size_t j = 0; j < q.size(); ++j) q[j] = k == n ? goal[j] : start[j] + t * (goal[j] - start[j]);
    out.push_back(std::move(q));
  }
  return out;
}

// ---- plan -------------------------------------------------------------------------------------

void selectCriLegs(CriArmPhasePlan & plan, int max_runs)
{
  plan.selected_segments.clear();
  std::vector<size_t> cand;
  for (size_t i = 0; i < plan.legs.size(); ++i)
  {
    plan.legs[i].selected = false;
    if (plan.legs[i].certified) cand.push_back(i);
  }
  std::stable_sort(cand.begin(), cand.end(), [&](size_t a, size_t b) { return plan.legs[a].cri_length_rad > plan.legs[b].cri_length_rad; });
  const size_t limit = max_runs > 0 ? static_cast<size_t>(max_runs) : 0;
  for (size_t k = 0; k < cand.size(); ++k)
  {
    CriLegRecord & leg = plan.legs[cand[k]];
    if (k < limit)
    {
      leg.selected = true;
      leg.reason.clear();
      plan.selected_segments.push_back(leg.geometry.segment_index);
    }
    else
    {
      leg.reason = "not selected: cri_transit_max_runs_per_arm_phase=" + std::to_string(max_runs) + " reached (a longer leg was chosen)";
    }
  }
  std::sort(plan.selected_segments.begin(), plan.selected_segments.end());
}

const CriArmPhasePlan * CriPlan::find(const std::string & phase, const std::string & side) const
{
  for (const auto & a : arm_phases)
    if (a.phase == phase && a.side == side) return &a;
  return nullptr;
}

// ---- runs -------------------------------------------------------------------------------------

const char * toString(CriRunType t) { return t == CriRunType::Cri ? "cri" : "jtc"; }

std::vector<CriRun> splitArmPhaseRuns(size_t segment_count, const std::vector<size_t> & cri_segments)
{
  std::vector<CriRun> runs;
  if (segment_count == 0) return runs;
  size_t pos = 0;
  size_t last_cri = 0;
  bool have_last = false;
  for (size_t c : cri_segments)
  {
    if (c >= segment_count || (have_last && c <= last_cri)) continue;   // out of range / unsorted / duplicate: ignored (JTC)
    if (c > pos) runs.push_back(CriRun{CriRunType::Jtc, pos, c});
    runs.push_back(CriRun{CriRunType::Cri, c, c + 1});
    pos = c + 1;
    last_cri = c;
    have_last = true;
  }
  if (pos < segment_count) runs.push_back(CriRun{CriRunType::Jtc, pos, segment_count});
  return runs;
}

std::vector<CriRun> criRunsForArmPhase(bool cri_enabled, bool cap_ok, const CriArmPhasePlan * plan, size_t segment_count)
{
  if (!cri_enabled || !cap_ok || plan == nullptr || plan->selected_segments.empty()) return splitArmPhaseRuns(segment_count, {});
  return splitArmPhaseRuns(segment_count, plan->selected_segments);
}

CriArmRuns buildCriArmRuns(bool cri_enabled, bool cap_ok, const CriArmPhasePlan * plan, const ArmPhase & arm)
{
  CriArmRuns out;
  out.phase = arm;
  if (!cri_enabled || !cap_ok || plan == nullptr || plan->selected_segments.empty())
  {
    out.runs = splitArmPhaseRuns(arm.segments.size(), {});
    out.leg.assign(out.runs.size(), nullptr);
    return out;
  }
  // replace every selected leg's segment by [head] + the segment itself (the placeholder of the CRI run) + [tail]
  out.phase.segments.clear();
  std::vector<size_t> cri_index;
  std::vector<const CriLegRecord *> by_index;
  for (size_t i = 0; i < arm.segments.size(); ++i)
  {
    const CriLegRecord * leg = nullptr;
    for (const CriLegRecord & l : plan->legs)
      if (l.selected && l.geometry.segment_index == i) leg = &l;
    if (leg && !leg->head.trajectory.points.empty()) out.phase.segments.push_back(leg->head);
    if (leg)
    {
      cri_index.push_back(out.phase.segments.size());
      by_index.push_back(leg);
    }
    out.phase.segments.push_back(arm.segments[i]);
    if (leg && !leg->tail.trajectory.points.empty()) out.phase.segments.push_back(leg->tail);
  }
  out.runs = splitArmPhaseRuns(out.phase.segments.size(), cri_index);
  for (const CriRun & r : out.runs)
  {
    const CriLegRecord * leg = nullptr;
    if (r.type == CriRunType::Cri)
      for (size_t k = 0; k < cri_index.size(); ++k)
        if (cri_index[k] == r.first_segment) leg = by_index[k];
    out.leg.push_back(leg);
  }
  return out;
}

// ---- live helpers -----------------------------------------------------------------------------

void WindowVelocity::add(double t_s, const std::vector<double> & q)
{
  if (!samples_.empty() && (t_s <= samples_.back().t || q.size() != samples_.back().q.size())) samples_.clear();   // time went back / joint set changed
  samples_.push_back(S{t_s, q});
  while (samples_.size() > 2 && samples_[1].t < t_s - keep_) samples_.pop_front();
}

double WindowVelocity::maxAbsVelocity() const
{
  if (samples_.size() < 2) return kInf;
  const S & now = samples_.back();
  // newest sample that is at least min_span_ older than `now`
  const S * base = nullptr;
  for (auto it = samples_.rbegin() + 1; it != samples_.rend(); ++it)
    if (now.t - it->t >= min_span_)
    {
      base = &*it;
      break;
    }
  if (!base) return kInf;
  const double dt = now.t - base->t;
  double v = 0.0;
  for (size_t j = 0; j < now.q.size(); ++j)
  {
    const double d = std::abs(now.q[j] - base->q[j]);
    if (!std::isfinite(d)) return kInf;
    v = std::max(v, d / dt);
  }
  return v;
}

// ---- chord supervision ------------------------------------------------------------------------

double maxAbsDiff(const std::vector<double> & a, const std::vector<double> & b)
{
  if (a.size() != b.size() || a.empty()) return kInf;
  double m = 0.0;
  for (size_t i = 0; i < a.size(); ++i)
  {
    const double d = std::abs(a[i] - b[i]);
    if (!std::isfinite(d)) return kInf;
    m = std::max(m, d);
  }
  return m;
}

ChordState chordState(const std::vector<double> & q, const std::vector<double> & start, const std::vector<double> & goal)
{
  ChordState c;
  if (q.size() != start.size() || q.size() != goal.size() || q.empty() || !allFinite(q) || !allFinite(start) || !allFinite(goal)) return c;
  double dd = 0.0, dot = 0.0;
  for (size_t j = 0; j < q.size(); ++j)
  {
    const double d = goal[j] - start[j];
    dd += d * d;
    dot += (q[j] - start[j]) * d;
  }
  if (!(dd > 0.0))
  {
    c.progress = 0.0;
    c.deviation_rad = maxAbsDiff(q, start);
    return c;
  }
  c.progress = dot / dd;
  const double t = std::min(1.0, std::max(0.0, c.progress));
  double dev = 0.0;
  for (size_t j = 0; j < q.size(); ++j) dev = std::max(dev, std::abs(q[j] - (start[j] + t * (goal[j] - start[j]))));
  c.deviation_rad = dev;
  return c;
}

// ---- gates ------------------------------------------------------------------------------------

namespace
{
Status joinFailures(const std::string & head, const std::vector<std::string> & f, std::vector<std::string> * out)
{
  if (out) *out = f;
  if (f.empty()) return Status::Ok();
  std::string msg = head;
  for (size_t i = 0; i < f.size(); ++i) msg += (i ? "; " : "") + f[i];
  return Status::Error(msg);
}
}  // namespace

Status evaluateCriHardware(const CriHwView & hw, double max_age_s, bool require_idle, std::vector<std::string> * failures)
{
  std::vector<std::string> f;
  if (!hw.have)
    f.push_back("no hardware_state received");
  else
  {
    if (!(hw.age_s < max_age_s)) f.push_back("hardware_state is " + num(hw.age_s) + " s old (limit " + num(max_age_s) + " s)");
    if (!hw.connected) f.push_back("driver not connected to the robot");
    if (!hw.status_fresh) f.push_back("robot STATUS is stale");
    if (hw.latched) f.push_back("driver is LATCHED");
    if (hw.move_stop_pending) f.push_back("a Move Stop is pending");
    if (!hw.activated) f.push_back("hardware component is not activated");
    if (require_idle && hw.move_active) f.push_back("a joint move is still active");
  }
  return joinFailures("hardware not healthy: ", f, failures);
}

bool criCapOk(const CriHwView & hw, const CriParams & p, std::string * why)
{
  auto no = [&](const std::string & s)
  {
    if (why) *why = s;
    return false;
  };
  if (!hw.cap_known)
  {
    if (p.assume_driver_cap_ok) return true;
    return no("the driver does not report its effective MoveJoint caps and cri_assume_driver_cap_ok is false");
  }
  if (!(std::isfinite(hw.cap_velocity_percent) && std::isfinite(hw.cap_acceleration_percent) && hw.cap_velocity_percent > 0.0 && hw.cap_acceleration_percent > 0.0))
    return no("the driver's reported caps are not valid numbers");
  if (hw.cap_velocity_percent > p.driver_max_velocity_cap_percent)
    return no("driver velocity cap " + num(hw.cap_velocity_percent) + " % > cri_driver_max_velocity_cap_percent " + num(p.driver_max_velocity_cap_percent));
  if (hw.cap_acceleration_percent > p.driver_max_acceleration_cap_percent)
    return no("driver acceleration cap " + num(hw.cap_acceleration_percent) + " % > cri_driver_max_acceleration_cap_percent " + num(p.driver_max_acceleration_cap_percent));
  if (p.velocity_percent > hw.cap_velocity_percent) return no("requested velocity " + num(p.velocity_percent) + " % exceeds the driver cap " + num(hw.cap_velocity_percent));
  if (p.acceleration_percent > hw.cap_acceleration_percent) return no("requested acceleration " + num(p.acceleration_percent) + " % exceeds the driver cap " + num(hw.cap_acceleration_percent));
  return true;
}

Status evaluateCriPreDispatch(const CriPreDispatchInputs & in, const CriPreDispatchLimits & lim, std::vector<std::string> * failures)
{
  std::vector<std::string> f;
  std::vector<std::string> hwf;
  evaluateCriHardware(in.hw, lim.hardware_max_age_s, true, &hwf);
  f.insert(f.end(), hwf.begin(), hwf.end());
  if (in.stop_latched) f.push_back("coordinator stop is latched");
  if (!in.cap_ok) f.push_back("driver caps not ok");
  if (!in.controller_active) f.push_back("the arm's trajectory controller is not active");
  if (!in.live.have)
    f.push_back("no robot_joint_states received");
  else
  {
    if (!(in.live.age_s < lim.live_max_age_s)) f.push_back("robot_joint_states is " + num(in.live.age_s) + " s old (limit " + num(lim.live_max_age_s) + " s)");
    if (!(in.live.rest_s >= lim.rest_duration_s)) f.push_back("arm at rest for only " + num(in.live.rest_s) + " s (need " + num(lim.rest_duration_s) + " s)");
    const double e = maxAbsDiff(in.live.q, in.chord_start);
    if (!(e <= in.start_tolerance_rad)) f.push_back("measured start is " + num(e) + " rad from the chord start (limit " + num(in.start_tolerance_rad) + ")");
  }
  return joinFailures("CRI pre-dispatch gate: ", f, failures);
}

Status evaluateCriArmGate(const CriHwView & hw, const CriLiveView & live, const CriArmGateLimits & lim, std::vector<std::string> * failures)
{
  std::vector<std::string> f;
  std::vector<std::string> hwf;
  evaluateCriHardware(hw, lim.hardware_max_age_s, true, &hwf);
  f.insert(f.end(), hwf.begin(), hwf.end());
  if (!live.have)
    f.push_back("no robot_joint_states received");
  else if (!(live.age_s < lim.live_max_age_s))
    f.push_back("robot_joint_states is " + num(live.age_s) + " s old (limit " + num(lim.live_max_age_s) + " s)");
  return joinFailures("arm gate: ", f, failures);
}

MonitorDecision evaluateCriRunMonitor(const CriRunMonitorInputs & in)
{
  MonitorDecision d;
  auto stop = [&](const std::string & why)
  {
    d.stop = true;
    d.reason = why;
    return d;
  };
  std::vector<std::string> hwf;
  if (!evaluateCriHardware(in.hw, in.hardware_max_age_s, false, &hwf)) return stop("CRI FAULT: " + hwf.front());
  if (!in.live.have) return stop("CRI run: no robot_joint_states");
  if (!(in.live.age_s <= in.live_stale_s)) return stop("CRI run: robot_joint_states stale for " + num(in.live.age_s) + " s (limit " + num(in.live_stale_s) + " s)");
  const ChordState cs = chordState(in.live.q, in.start, in.goal);
  if (!(cs.deviation_rad <= in.path_tolerance_rad))
    return stop("CRI run: " + num(cs.deviation_rad) + " rad off the chord (limit " + num(in.path_tolerance_rad) + " rad)");
  if (!in.result_received && !(in.elapsed_s <= in.timeout_s))
    return stop("CRI run: no result after " + num(in.elapsed_s) + " s (timeout " + num(in.timeout_s) + " s)");
  return d;
}

// ---- controller gate -----------------------------------------------------------------------------

const char * toString(CtrlGateMode m)
{
  switch (m)
  {
    case CtrlGateMode::Normal: return "normal";
    case CtrlGateMode::Switching: return "switching";
    case CtrlGateMode::CriRun: return "cri_run";
  }
  return "?";
}

bool controllerStateAllowed(CtrlGateMode mode, const std::string & state, bool poll_after_mode_start)
{
  const bool active = state == "active";
  const bool inactive = state == "inactive";
  switch (mode)
  {
    case CtrlGateMode::Normal: return poll_after_mode_start ? active : (active || inactive);
    case CtrlGateMode::Switching: return active || inactive;
    case CtrlGateMode::CriRun: return poll_after_mode_start ? inactive : (active || inactive);
  }
  return false;
}

// ---- MoveJoint result ----------------------------------------------------------------------------

CriVerdict classifyMoveJointResult(const MoveJointResultView & r, const std::vector<double> & goal, double goal_tolerance_rad)
{
  CriVerdict v;
  auto fault = [&](const std::string & why)
  {
    v.success = false;
    v.reason = why;
    return v;
  };
  const std::string detail = " (error_code " + std::to_string(static_cast<int>(r.error_code)) + ", end_reason '" + r.end_reason + "', stop_sent " +
                             (r.stop_sent ? "true" : "false") + ", stop_confirmed " + (r.stop_confirmed ? "true" : "false") +
                             (r.message.empty() ? "" : ", " + r.message) + ")";
  // a robot ERROR or a latched driver is a FAULT whatever the error code says (CANCELED with stop_confirmed included)
  if (r.end_reason == "ERROR") return fault("robot ERROR" + detail);
  if (r.hardware_latched) return fault("driver is latched" + detail);
  if (r.outcome != ActionOutcome::Succeeded) return fault("MoveJoint did not succeed" + detail);
  if (r.error_code != kMjSuccess) return fault("MoveJoint error code is not SUCCESS" + detail);
  if (r.end_reason != "PLAN") return fault("MoveJoint ended with a reason other than PLAN" + detail);
  if (!r.live_valid) return fault("no valid live state to confirm the goal" + detail);
  const double e = maxAbsDiff(r.live_q, goal);
  if (!(e <= goal_tolerance_rad)) return fault("live state is " + num(e) + " rad from the goal (limit " + num(goal_tolerance_rad) + ")" + detail);
  v.success = true;
  return v;
}

// ---- doStop ----------------------------------------------------------------------------------------

const char * toString(StopAction a)
{
  switch (a)
  {
    case StopAction::CancelMoveJoint: return "cancel_move_joint";
    case StopAction::HoldJtc: return "hold_jtc";
    case StopAction::CancelFjt: return "cancel_fjt";
  }
  return "?";
}

std::vector<StopStep> planStopSteps(bool cri_enabled, const std::array<bool, 2> & hold_allowed)
{
  std::vector<StopStep> steps;
  if (cri_enabled)
    for (int arm : {0, 1}) steps.push_back(StopStep{StopAction::CancelMoveJoint, arm});
  for (int arm : {0, 1})
    if (hold_allowed[static_cast<size_t>(arm)]) steps.push_back(StopStep{StopAction::HoldJtc, arm});
  for (int arm : {0, 1}) steps.push_back(StopStep{StopAction::CancelFjt, arm});
  return steps;
}

}  // namespace dual_cut
