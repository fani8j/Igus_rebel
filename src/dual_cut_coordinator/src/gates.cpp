#include "dual_cut_coordinator/gates.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace dual_cut
{

Eigen::Isometry3d poseFromXyzRpy(const std::array<double, 3> & xyz, const std::array<double, 3> & rpy)
{
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.linear() = (Eigen::AngleAxisd(rpy[2], Eigen::Vector3d::UnitZ()) *
                Eigen::AngleAxisd(rpy[1], Eigen::Vector3d::UnitY()) *
                Eigen::AngleAxisd(rpy[0], Eigen::Vector3d::UnitX()))
                 .toRotationMatrix();
  T.translation() = Eigen::Vector3d(xyz[0], xyz[1], xyz[2]);
  return T;
}

Status checkFrameConsistency(const Eigen::Isometry3d & exported, const Eigen::Isometry3d & expected, double tol)
{
  if (!exported.matrix().allFinite() || !expected.matrix().allFinite())
    return Status::Error("non-finite frame transform");
  const double dt = (exported.translation() - expected.translation()).cwiseAbs().maxCoeff();
  const double dr = std::abs(Eigen::AngleAxisd(exported.linear().transpose() * expected.linear()).angle());
  if (!(dt <= tol) || !(dr <= tol))
  {
    std::ostringstream os;
    os << "exported base transform differs from the dual URDF: max |dt| = " << dt
       << " m, rotation angle = " << dr << " rad (tolerance " << tol << ")";
    return Status::Error(os.str());
  }
  return Status::Ok();
}

Status checkBoxConsistency(
  const Eigen::Isometry3d & a, const Eigen::Vector3d & a_size, const Eigen::Isometry3d & b,
  const Eigen::Vector3d & b_size, double tol_m, double tol_rad, double * center_error_m,
  double * rotation_error_rad)
{
  if (!a.matrix().allFinite() || !b.matrix().allFinite() || !a_size.allFinite() || !b_size.allFinite())
    return Status::Error("non-finite box");
  const double dsize = (a_size - b_size).cwiseAbs().maxCoeff();
  const double dc = (a.translation() - b.translation()).norm();
  // relative rotation in a's frame; the box is symmetric under 180 degrees about its z axis
  const Eigen::Matrix3d rel = a.linear().transpose() * b.linear();
  const Eigen::Matrix3d flip = Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  const double r0 = std::abs(Eigen::AngleAxisd(rel).angle());
  const double r1 = std::abs(Eigen::AngleAxisd(flip.transpose() * rel).angle());
  const double dr = std::min(r0, r1);
  if (center_error_m) *center_error_m = dc;
  if (rotation_error_rad) *rotation_error_rad = dr;
  if (!(dsize <= tol_m) || !(dc <= tol_m) || !(dr <= tol_rad))
  {
    std::ostringstream os;
    os << "the two arms see different cartons: size diff " << dsize * 1e3 << " mm, centre diff " << dc * 1e3
       << " mm, rotation diff " << dr << " rad (tolerances " << tol_m * 1e3 << " mm, " << tol_rad << " rad)";
    return Status::Error(os.str());
  }
  return Status::Ok();
}

double distancePointToSegment(const Eigen::Vector3d & p, const Eigen::Vector3d & a, const Eigen::Vector3d & b)
{
  const Eigen::Vector3d ab = b - a;
  const double len2 = ab.squaredNorm();
  if (len2 <= 1e-18) return (p - a).norm();
  const double t = std::clamp((p - a).dot(ab) / len2, 0.0, 1.0);
  return (p - (a + t * ab)).norm();
}

Status cutLinesFromTargets(
  const std::vector<geometry_msgs::msg::PoseStamped> & targets, size_t expected_cut_segments,
  const std::string & expected_frame, const Eigen::Isometry3d & base_to_world, std::vector<CutLine> & out)
{
  out.clear();
  if (targets.size() != 2 * expected_cut_segments)
    return Status::Error(
      "cut_targets has " + std::to_string(targets.size()) + " poses but " +
      std::to_string(expected_cut_segments) + " CUT segments need " + std::to_string(2 * expected_cut_segments));
  std::vector<Eigen::Vector3d> pts;
  for (const auto & t : targets)
  {
    if (t.header.frame_id != expected_frame)
      return Status::Error(
        "cut target frame '" + t.header.frame_id + "' != planning frame '" + expected_frame + "'");
    const Eigen::Vector3d p(t.pose.position.x, t.pose.position.y, t.pose.position.z);
    if (!p.allFinite()) return Status::Error("non-finite cut target");
    pts.push_back(base_to_world * p);
  }
  for (size_t i = 0; i < expected_cut_segments; ++i) out.push_back(CutLine{pts[2 * i], pts[2 * i + 1]});
  return Status::Ok();
}

TipGateResult checkTipOnLine(
  const ArmPhase & arm, const std::vector<CutLine> & lines, const TipFk & fk, double tol_m,
  const SampleOptions & sample)
{
  TipGateResult r;
  size_t cuts = 0;
  for (const auto & s : arm.segments) cuts += s.kind == SegmentKind::Cut;
  r.cut_segments = cuts;
  if (cuts != lines.size())
  {
    r.message = "tip-on-line: " + std::to_string(cuts) + " CUT segments but " + std::to_string(lines.size()) +
                " cut lines (fail closed)";
    return r;
  }
  size_t ci = 0;
  for (const auto & seg : arm.segments)
  {
    if (seg.kind != SegmentKind::Cut) continue;
    const CutLine & line = lines[ci];
    std::vector<TrajPoint> samples;
    Status s = sampleTrajectory(seg.trajectory, sample, samples);
    if (!s)
    {
      r.message = "tip-on-line: cannot sample CUT segment '" + seg.stage_name + "': " + s.message;
      return r;
    }
    for (size_t k = 0; k < samples.size(); ++k)
    {
      Eigen::Vector3d tip;
      s = fk(seg.trajectory.joint_names, samples[k].positions, tip);
      if (!s)
      {
        r.message = "tip-on-line: FK failed: " + s.message;
        return r;
      }
      ++r.samples;
      const double dev = distancePointToSegment(tip, line.start, line.end);
      r.max_deviation_m = std::max(r.max_deviation_m, dev);
      if (k == 0) r.max_endpoint_error_m = std::max(r.max_endpoint_error_m, (tip - line.start).norm());
      if (k + 1 == samples.size()) r.max_endpoint_error_m = std::max(r.max_endpoint_error_m, (tip - line.end).norm());
      if (!(dev <= tol_m))
      {
        std::ostringstream os;
        os << "tip-on-line: CUT '" << seg.stage_name << "' sample " << k << " is " << dev * 1e3
           << " mm from the exported cut line (limit " << tol_m * 1e3 << " mm)";
        r.message = os.str();
        return r;
      }
    }
    if (r.max_endpoint_error_m > tol_m)
    {
      std::ostringstream os;
      os << "tip-on-line: CUT '" << seg.stage_name << "' end points are " << r.max_endpoint_error_m * 1e3
         << " mm from the exported line ends (limit " << tol_m * 1e3 << " mm)";
      r.message = os.str();
      return r;
    }
    ++ci;
  }
  r.ok = true;
  r.message = "ok";
  return r;
}

void RestTracker::reset()
{
  have_ = false;
  at_rest_ = false;
  rest_since_ = 0.0;
  last_t_ = 0.0;
}

void RestTracker::update(double t_s, double max_abs_velocity)
{
  if (!std::isfinite(max_abs_velocity) || max_abs_velocity >= threshold_)
  {
    at_rest_ = false;
  }
  else if (!at_rest_)
  {
    at_rest_ = true;
    rest_since_ = t_s;
  }
  have_ = true;
  last_t_ = t_s;
}

double RestTracker::restDuration(double now_s) const
{
  if (!have_ || !at_rest_) return 0.0;
  return std::max(0.0, now_s - rest_since_);
}

Status evaluateBarriers(const BarrierInputs & in, const BarrierLimits & l, std::vector<std::string> * failures)
{
  std::vector<std::string> f;
  if (in.stop_latched) f.push_back("stop is latched (rearm first)");
  if (!(in.joint_state_age_s < l.joint_state_max_age_s))
    f.push_back("/joint_states age " + std::to_string(in.joint_state_age_s) + " s >= " + std::to_string(l.joint_state_max_age_s) + " s");
  if (!(in.max_joint_error_to_start_rad <= l.start_tolerance_rad))
    f.push_back("a joint is " + std::to_string(in.max_joint_error_to_start_rad) + " rad from the phase start (limit " + std::to_string(l.start_tolerance_rad) + ")");
  if (!(in.rest_duration_s >= l.rest_duration_s))
    f.push_back("arms at rest for only " + std::to_string(in.rest_duration_s) + " s (need " + std::to_string(l.rest_duration_s) + ")");
  if (!in.controllers_active) f.push_back("a trajectory controller is not active");
  if (!in.hardware_active) f.push_back("a hardware component is not active");
  if (!(in.snapshot_age_s <= l.snapshot_max_age_s))
    f.push_back("snapshot is " + std::to_string(in.snapshot_age_s) + " s old (limit " + std::to_string(l.snapshot_max_age_s) + ")");
  if (!(in.monitor_age_s <= l.monitor_max_age_s)) f.push_back("monitor is not running (last tick " + std::to_string(in.monitor_age_s) + " s ago)");
  if (!in.monitor_healthy) f.push_back("monitor reports an unhealthy state");
  if (failures) *failures = f;
  if (f.empty()) return Status::Ok();
  std::string msg = "barrier failed: ";
  for (size_t i = 0; i < f.size(); ++i) msg += (i ? "; " : "") + f[i];
  return Status::Error(msg);
}

MonitorDecision evaluateMonitor(const MonitorInputs & in, const MonitorLimits & l)
{
  MonitorDecision d;
  auto stop = [&](const std::string & why)
  {
    d.stop = true;
    d.reason = why;
    return d;
  };
  if (in.unexpected_goal_end) return stop("goal ended unexpectedly: " + in.goal_end_reason);
  if (!in.have_joint_state) return stop("no /joint_states received");
  if (!(in.joint_state_age_s <= l.joint_state_stale_s))
    return stop("/joint_states stale for " + std::to_string(in.joint_state_age_s) + " s (limit " + std::to_string(l.joint_state_stale_s) + ")");
  if (!in.distance_valid || !std::isfinite(in.cross_distance_m)) return stop("cross-arm distance unavailable (fail closed)");
  if (in.cross_distance_m < l.stop_distance_m)
    return stop("cross-arm distance " + std::to_string(in.cross_distance_m) + " m < " + std::to_string(l.stop_distance_m) + " m");
  if (!in.poll_valid || !(in.poll_age_s <= l.status_poll_stale_s))
    return stop("controller / hardware state poll is not answering (fail closed)");
  if (!in.controllers_active) return stop("a trajectory controller is no longer active");
  if (!in.hardware_active) return stop("a hardware component is no longer active");
  if (!(in.tracking_error_rad <= l.tracking_stop_rad))
    return stop("tracking error " + std::to_string(in.tracking_error_rad) + " rad > " + std::to_string(l.tracking_stop_rad) + " rad");
  if (in.tracking_error_rad > l.tracking_warn_rad)
  {
    d.warn = true;
    d.reason = "tracking error " + std::to_string(in.tracking_error_rad) + " rad > " + std::to_string(l.tracking_warn_rad) + " rad";
  }
  return d;
}

}  // namespace dual_cut
