#include "dual_cut_coordinator/trajectory_tools.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <sstream>

namespace dual_cut
{
namespace
{

std::string fmt(const char * what, size_t idx, double a, double b)
{
  std::ostringstream os;
  os << what << " (index " << idx << ": " << a << " vs " << b << ")";
  return os.str();
}

double maxAbsDiff(const std::vector<double> & a, const std::vector<double> & b)
{
  double m = 0.0;
  for (size_t i = 0; i < a.size() && i < b.size(); ++i) m = std::max(m, std::abs(a[i] - b[i]));
  return m;
}

// Evaluate the segment [a, b] at local time tau in [0, T], exactly like the JTC (rules 2 and 3).
void evalSegment(const TrajPoint & a, const TrajPoint & b, double T, double tau, TrajPoint & out)
{
  const size_t dim = a.positions.size();
  out.positions.assign(dim, 0.0);
  out.velocities.assign(dim, 0.0);
  out.accelerations.assign(dim, 0.0);

  const Degree deg = segmentDegree(a, b);
  double t[6];
  t[0] = 1.0;
  for (int i = 1; i <= 5; ++i) t[i] = t[i - 1] * tau;
  double Tp[6];
  Tp[0] = 1.0;
  for (int i = 1; i <= 5; ++i) Tp[i] = Tp[i - 1] * T;

  for (size_t i = 0; i < dim; ++i)
  {
    const double p0 = a.positions[i];
    const double p1 = b.positions[i];
    if (deg == Degree::Linear)
    {
      const double c1 = T != 0.0 ? (p1 - p0) / T : 0.0;
      out.positions[i] = p0 + t[1] * c1;
      out.velocities[i] = c1;
    }
    else if (deg == Degree::Cubic)
    {
      const double v0 = a.velocities[i];
      const double v1 = b.velocities[i];
      double c2 = 0.0;
      double c3 = 0.0;
      if (T != 0.0)
      {
        c2 = (-3.0 * p0 + 3.0 * p1 - 2.0 * v0 * Tp[1] - v1 * Tp[1]) / Tp[2];
        c3 = (2.0 * p0 - 2.0 * p1 + v0 * Tp[1] + v1 * Tp[1]) / Tp[3];
      }
      out.positions[i] = p0 + t[1] * v0 + t[2] * c2 + t[3] * c3;
      out.velocities[i] = v0 + t[1] * 2.0 * c2 + t[2] * 3.0 * c3;
      out.accelerations[i] = 2.0 * c2 + t[1] * 6.0 * c3;
    }
    else
    {
      const double v0 = a.velocities[i];
      const double v1 = b.velocities[i];
      const double a0 = a.accelerations[i];
      const double a1 = b.accelerations[i];
      double c3 = 0.0;
      double c4 = 0.0;
      double c5 = 0.0;
      if (T != 0.0)
      {
        c3 = (-20.0 * p0 + 20.0 * p1 - 3.0 * a0 * Tp[2] + a1 * Tp[2] - 12.0 * v0 * Tp[1] -
              8.0 * v1 * Tp[1]) /
             (2.0 * Tp[3]);
        c4 = (30.0 * p0 - 30.0 * p1 + 3.0 * a0 * Tp[2] - 2.0 * a1 * Tp[2] + 16.0 * v0 * Tp[1] +
              14.0 * v1 * Tp[1]) /
             (2.0 * Tp[4]);
        c5 = (-12.0 * p0 + 12.0 * p1 - a0 * Tp[2] + a1 * Tp[2] - 6.0 * v0 * Tp[1] -
              6.0 * v1 * Tp[1]) /
             (2.0 * Tp[5]);
      }
      const double c2 = 0.5 * a0;
      out.positions[i] = p0 + t[1] * v0 + t[2] * c2 + t[3] * c3 + t[4] * c4 + t[5] * c5;
      out.velocities[i] =
        v0 + t[1] * 2.0 * c2 + t[2] * 3.0 * c3 + t[3] * 4.0 * c4 + t[4] * 5.0 * c5;
      out.accelerations[i] = 2.0 * c2 + t[1] * 6.0 * c3 + t[2] * 12.0 * c4 + t[3] * 20.0 * c5;
    }
  }
}

// Sampling without re-validating (callers validate once).
void sampleUnchecked(const JointTrajectory & traj, double t, TrajPoint & out)
{
  const auto & pts = traj.points;
  out.time_s = t;
  if (t < pts.front().time_s)
  {
    out = pts.front();
    out.time_s = t;
    const size_t n = out.positions.size();
    if (out.velocities.empty()) out.velocities.assign(n, 0.0);
    if (out.accelerations.empty()) out.accelerations.assign(n, 0.0);
    return;
  }
  if (t >= pts.back().time_s)
  {
    out = pts.back();
    out.time_s = t;
    const size_t n = out.positions.size();
    if (out.velocities.empty()) out.velocities.assign(n, 0.0);
    if (out.accelerations.empty()) out.accelerations.assign(n, 0.0);
    return;
  }
  // first point with time > t, the segment starts one before it
  auto it = std::upper_bound(
    pts.begin(), pts.end(), t, [](double v, const TrajPoint & p) { return v < p.time_s; });
  const size_t i1 = static_cast<size_t>(it - pts.begin());
  const TrajPoint & a = pts[i1 - 1];
  const TrajPoint & b = pts[i1];
  evalSegment(a, b, b.time_s - a.time_s, t - a.time_s, out);
  out.time_s = t;
}

Status checkBoundary(
  const TrajPoint & prev_last, const TrajPoint & next_first, const ConcatOptions & o, size_t seg)
{
  if (prev_last.positions.size() != next_first.positions.size())
    return Status::Error("segment " + std::to_string(seg) + ": joint count differs at boundary");
  const double dp = maxAbsDiff(prev_last.positions, next_first.positions);
  if (!(dp <= o.position_tol))
    return Status::Error(
      fmt("segment boundary position mismatch", seg, dp, o.position_tol) + " [max |dq| vs tol]");
  if (!prev_last.velocities.empty() && !next_first.velocities.empty())
  {
    const double dv = maxAbsDiff(prev_last.velocities, next_first.velocities);
    if (!(dv <= o.velocity_tol))
      return Status::Error(fmt("segment boundary velocity mismatch", seg, dv, o.velocity_tol));
  }
  if (!prev_last.accelerations.empty() && !next_first.accelerations.empty())
  {
    const double da = maxAbsDiff(prev_last.accelerations, next_first.accelerations);
    if (!(da <= o.acceleration_tol))
      return Status::Error(
        fmt("segment boundary acceleration mismatch", seg, da, o.acceleration_tol));
  }
  return Status::Ok();
}

bool isJointDigits(const std::string & s, size_t start)
{
  if (s.compare(start, 5, "joint") != 0) return false;
  const size_t d = start + 5;
  if (d >= s.size()) return false;
  for (size_t i = d; i < s.size(); ++i)
    if (s[i] < '0' || s[i] > '9') return false;
  return true;
}

}  // namespace

JointTrajectory fromMsg(const trajectory_msgs::msg::JointTrajectory & msg)
{
  JointTrajectory out;
  out.joint_names = msg.joint_names;
  out.points.reserve(msg.points.size());
  for (const auto & p : msg.points)
  {
    TrajPoint q;
    q.time_s = static_cast<double>(p.time_from_start.sec) +
               1e-9 * static_cast<double>(p.time_from_start.nanosec);
    q.positions = p.positions;
    q.velocities = p.velocities;
    q.accelerations = p.accelerations;
    out.points.push_back(std::move(q));
  }
  return out;
}

trajectory_msgs::msg::JointTrajectory toMsg(const JointTrajectory & traj)
{
  trajectory_msgs::msg::JointTrajectory msg;
  msg.joint_names = traj.joint_names;
  msg.points.reserve(traj.points.size());
  for (const auto & p : traj.points)
  {
    trajectory_msgs::msg::JointTrajectoryPoint q;
    // Round to ns exactly like rclcpp::Duration::from_seconds would.
    const double whole = std::floor(p.time_s);
    long long ns = std::llround((p.time_s - whole) * 1e9);
    long long sec = static_cast<long long>(whole);
    if (ns >= 1000000000LL)
    {
      ns -= 1000000000LL;
      ++sec;
    }
    q.time_from_start.sec = static_cast<int32_t>(sec);
    q.time_from_start.nanosec = static_cast<uint32_t>(ns);
    q.positions = p.positions;
    q.velocities = p.velocities;
    q.accelerations = p.accelerations;
    msg.points.push_back(std::move(q));
  }
  return msg;
}

Status validate(const JointTrajectory & traj)
{
  if (traj.points.empty()) return Status::Error("trajectory has no points");
  const size_t n = traj.joint_names.size();
  if (n == 0) return Status::Error("trajectory has no joint names");
  for (size_t i = 0; i < traj.points.size(); ++i)
  {
    const auto & p = traj.points[i];
    const std::string where = "point " + std::to_string(i) + ": ";
    if (!std::isfinite(p.time_s)) return Status::Error(where + "non-finite time");
    if (i > 0 && !(p.time_s > traj.points[i - 1].time_s))
      return Status::Error(where + "time is not strictly increasing");
    if (p.positions.size() != n)
      return Status::Error(where + "positions size does not match joint_names");
    if (!p.velocities.empty() && p.velocities.size() != n)
      return Status::Error(where + "velocities size does not match joint_names");
    if (!p.accelerations.empty() && p.accelerations.size() != n)
      return Status::Error(where + "accelerations size does not match joint_names");
    if (!p.accelerations.empty() && p.velocities.empty())
      return Status::Error(where + "accelerations without velocities (JTC would output zeros)");
    for (double v : p.positions)
      if (!std::isfinite(v)) return Status::Error(where + "non-finite position");
    for (double v : p.velocities)
      if (!std::isfinite(v)) return Status::Error(where + "non-finite velocity");
    for (double v : p.accelerations)
      if (!std::isfinite(v)) return Status::Error(where + "non-finite acceleration");
  }
  return Status::Ok();
}

Degree segmentDegree(const TrajPoint & a, const TrajPoint & b)
{
  const bool has_velocity = !a.velocities.empty() && !b.velocities.empty();
  const bool has_accel = !a.accelerations.empty() && !b.accelerations.empty();
  if (has_velocity && has_accel) return Degree::Quintic;
  if (has_velocity) return Degree::Cubic;
  return Degree::Linear;
}

Status sampleAt(const JointTrajectory & traj, double t, TrajPoint & out)
{
  Status s = validate(traj);
  if (!s) return s;
  if (!std::isfinite(t)) return Status::Error("non-finite sample time");
  sampleUnchecked(traj, t, out);
  return Status::Ok();
}

Status sampleTrajectory(
  const JointTrajectory & traj, const SampleOptions & o, std::vector<TrajPoint> & out)
{
  out.clear();
  Status s = validate(traj);
  if (!s) return s;
  if (!(o.dt_s > 0.0) || !std::isfinite(o.dt_s)) return Status::Error("dt_s must be positive");

  const double t0 = traj.points.front().time_s;
  const double t1 = traj.points.back().time_s;

  std::vector<double> times;
  const double n_reg_d = std::floor((t1 - t0) / o.dt_s);
  if (!(n_reg_d < static_cast<double>(o.max_samples)))
    return Status::Error("too many regular samples for the given dt");
  for (size_t k = 0; k <= static_cast<size_t>(n_reg_d); ++k)
    times.push_back(t0 + static_cast<double>(k) * o.dt_s);
  if (o.include_waypoints)
    for (const auto & p : traj.points) times.push_back(p.time_s);
  times.push_back(t1);
  std::sort(times.begin(), times.end());
  // Merge times closer than 1e-9 s (regular grid point landing on a waypoint).
  std::vector<double> uniq;
  for (double t : times)
    if (uniq.empty() || t - uniq.back() > 1e-9) uniq.push_back(t);
  // The last entry must be exactly t1.
  uniq.back() = t1;
  if (uniq.size() > 1 && uniq.back() - uniq[uniq.size() - 2] <= 1e-9) uniq.erase(uniq.end() - 2);

  out.reserve(uniq.size());
  {
    TrajPoint p;
    sampleUnchecked(traj, uniq.front(), p);
    out.push_back(std::move(p));
  }

  const bool densify = o.max_joint_step_rad > 0.0;
  Status err = Status::Ok();

  std::function<void(const TrajPoint &, double, int)> refine;
  refine = [&](const TrajPoint & a, double tb, int depth)
  {
    // 'a' is the already emitted left end; emit samples after it up to and including tb.
    TrajPoint b;
    sampleUnchecked(traj, tb, b);
    bool split = false;
    if (densify)
    {
      const double step = maxAbsDiff(a.positions, b.positions);
      split = step > o.max_joint_step_rad;
      if (!split)
      {
        // Endpoints are close but the spline may have swung out and back: look at the middle too.
        TrajPoint m;
        sampleUnchecked(traj, 0.5 * (a.time_s + tb), m);
        split = maxAbsDiff(a.positions, m.positions) > o.max_joint_step_rad ||
                maxAbsDiff(m.positions, b.positions) > o.max_joint_step_rad;
      }
    }
    if (split)
    {
      if (depth >= o.max_bisect_depth || tb - a.time_s < 1e-9)
      {
        if (err.ok) err = Status::Error("densify: bisection depth exceeded (discontinuous?)");
        return;
      }
      const double tm = 0.5 * (a.time_s + tb);
      refine(a, tm, depth + 1);
      if (!err.ok) return;
      const TrajPoint left = out.back();
      refine(left, tb, depth + 1);
      return;
    }
    if (out.size() >= o.max_samples)
    {
      if (err.ok) err = Status::Error("densify: too many samples");
      return;
    }
    out.push_back(std::move(b));
  };

  for (size_t i = 1; i < uniq.size() && err.ok; ++i)
  {
    TrajPoint left = out.back();
    refine(left, uniq[i], 0);
  }
  if (!err.ok)
  {
    out.clear();
    return err;
  }
  return Status::Ok();
}

Status densifyLinear(
  const std::vector<std::vector<double>> & path, double max_step_rad,
  std::vector<std::vector<double>> & out, size_t max_points)
{
  out.clear();
  if (!(max_step_rad > 0.0) || !std::isfinite(max_step_rad))
    return Status::Error("max_step_rad must be positive");
  if (path.empty()) return Status::Ok();
  out.push_back(path.front());
  for (size_t i = 1; i < path.size(); ++i)
  {
    const auto & a = path[i - 1];
    const auto & b = path[i];
    if (a.size() != b.size()) return Status::Error("densifyLinear: configuration size differs");
    const double step = maxAbsDiff(a, b);
    if (!std::isfinite(step)) return Status::Error("densifyLinear: non-finite configuration");
    const double nd = std::ceil(step / max_step_rad);
    if (nd + static_cast<double>(out.size()) > static_cast<double>(max_points))
    {
      out.clear();
      return Status::Error("densifyLinear: too many points");
    }
    const size_t n = std::max<size_t>(1, static_cast<size_t>(nd));
    for (size_t k = 1; k <= n; ++k)
    {
      const double f = static_cast<double>(k) / static_cast<double>(n);
      std::vector<double> q(a.size());
      for (size_t j = 0; j < a.size(); ++j) q[j] = (k == n) ? b[j] : a[j] + f * (b[j] - a[j]);
      out.push_back(std::move(q));
    }
  }
  return Status::Ok();
}

Status concatenate(
  const std::vector<JointTrajectory> & segments, const ConcatOptions & options,
  JointTrajectory & out, std::vector<size_t> * segment_first_point)
{
  out = JointTrajectory{};
  if (segment_first_point) segment_first_point->clear();
  if (segments.empty()) return Status::Error("no segments to concatenate");
  for (size_t i = 0; i < segments.size(); ++i)
  {
    Status s = validate(segments[i]);
    if (!s) return Status::Error("segment " + std::to_string(i) + ": " + s.message);
    if (segments[i].joint_names != segments.front().joint_names)
      return Status::Error("segment " + std::to_string(i) + ": joint_names differ from segment 0");
  }
  out.joint_names = segments.front().joint_names;
  double offset = 0.0;  // output time of the current segment's first point
  for (size_t i = 0; i < segments.size(); ++i)
  {
    const auto & seg = segments[i];
    const double first_t = seg.points.front().time_s;
    size_t start = 0;
    if (i > 0)
    {
      Status s = checkBoundary(out.points.back(), seg.points.front(), options, i);
      if (!s) return s;
      start = 1;  // drop the duplicate boundary point (the previous segment's last point is kept)
    }
    if (segment_first_point) segment_first_point->push_back(i == 0 ? 0 : out.points.size() - 1);
    for (size_t k = start; k < seg.points.size(); ++k)
    {
      TrajPoint p = seg.points[k];
      p.time_s = offset + (p.time_s - first_t);
      out.points.push_back(std::move(p));
    }
    offset += seg.points.back().time_s - first_t;
  }
  if (segment_first_point) segment_first_point->push_back(out.points.size() - 1);
  return Status::Ok();
}

Status buildSampledPath(
  const std::vector<SegmentInput> & segments, const SampleOptions & options,
  const ConcatOptions & concat, SampledPath & out)
{
  out = SampledPath{};
  if (segments.empty()) return Status::Error("no segments");
  out.joint_names = segments.front().trajectory.joint_names;
  double offset = 0.0;

  for (size_t i = 0; i < segments.size(); ++i)
  {
    const auto & seg = segments[i];
    Status s = validate(seg.trajectory);
    if (!s) return Status::Error("segment " + std::to_string(i) + ": " + s.message);
    if (seg.trajectory.joint_names != out.joint_names)
      return Status::Error("segment " + std::to_string(i) + ": joint_names differ from segment 0");

    if (i > 0)
    {
      Status b = checkBoundary(
        segments[i - 1].trajectory.points.back(), seg.trajectory.points.front(), concat, i);
      if (!b) return b;
    }

    std::vector<TrajPoint> samples;
    s = sampleTrajectory(seg.trajectory, options, samples);
    if (!s) return Status::Error("segment " + std::to_string(i) + ": " + s.message);

    const double first_t = seg.trajectory.points.front().time_s;
    size_t start = 0;
    if (i > 0)
    {
      start = 1;  // drop the duplicate boundary sample; merge the contact flag into the kept one
      if (seg.contact_allowed) out.samples.back().contact_allowed = true;
    }
    for (size_t k = start; k < samples.size(); ++k)
    {
      PathSample ps;
      ps.time_s = offset + (samples[k].time_s - first_t);
      ps.q = std::move(samples[k].positions);
      ps.contact_allowed = seg.contact_allowed;
      ps.segment_index = i;
      out.samples.push_back(std::move(ps));
      if (out.samples.size() > options.max_samples)
      {
        out = SampledPath{};
        return Status::Error("too many samples in the sampled path");
      }
    }
    // The first sample of a segment (kept for i == 0 only) is the boundary; the last sample of
    // this segment is the boundary to the next one and carries this segment's flag already.
    offset += seg.trajectory.points.back().time_s - first_t;
  }
  return Status::Ok();
}

std::string armPrefix(const std::string & side)
{
  if (side == "left" || side == "right") return "arm_" + side + "_";
  return std::string();
}

Status prefixJointName(const std::string & prefix, const std::string & name, std::string & out)
{
  if (prefix.empty()) return Status::Error("empty joint prefix");
  if (!isJointDigits(name, 0))
    return Status::Error("joint name '" + name + "' is not of the form joint<N>");
  out = prefix + name;
  return Status::Ok();
}

Status prefixJointNames(const std::string & prefix, std::vector<std::string> & names)
{
  std::vector<std::string> mapped;
  mapped.reserve(names.size());
  for (const auto & n : names)
  {
    std::string m;
    Status s = prefixJointName(prefix, n, m);
    if (!s) return s;
    mapped.push_back(std::move(m));
  }
  names = std::move(mapped);
  return Status::Ok();
}

Status prefixTrajectory(const std::string & prefix, JointTrajectory & traj)
{
  return prefixJointNames(prefix, traj.joint_names);
}

Status stripJointPrefix(const std::string & prefix, const std::string & name, std::string & out)
{
  if (prefix.empty() || name.compare(0, prefix.size(), prefix) != 0 ||
      !isJointDigits(name, prefix.size()))
    return Status::Error("joint name '" + name + "' is not " + prefix + "joint<N>");
  out = name.substr(prefix.size());
  return Status::Ok();
}

}  // namespace dual_cut
