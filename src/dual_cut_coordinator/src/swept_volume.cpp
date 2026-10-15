#include "dual_cut_coordinator/swept_volume.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <set>
#include <unordered_map>

#include <geometric_shapes/shapes.h>
#include <moveit/collision_detection/collision_common.h>
#include <moveit/collision_detection/collision_env.h>

namespace dual_cut
{
namespace
{

bool startsWith(const std::string & s, const std::string & prefix)
{
  return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
}

bool finite3(const Eigen::Vector3d & v)
{
  return std::isfinite(v.x()) && std::isfinite(v.y()) && std::isfinite(v.z());
}

// Variable indices of the given joints (single-DOF active joints only).
Status variableIndices(
  const moveit::core::RobotModel & model, const std::vector<std::string> & joints,
  std::vector<int> & idx)
{
  idx.clear();
  for (const auto & j : joints)
  {
    if (!model.hasJointModel(j)) return Status::Error("joint '" + j + "' is not in the robot model");
    const auto * jm = model.getJointModel(j);
    if (jm->getVariableCount() != 1)
      return Status::Error("joint '" + j + "' is not a single-variable joint");
    idx.push_back(static_cast<int>(model.getVariableIndex(j)));
  }
  return Status::Ok();
}

// Index mapping from the columns of a path to variable indices, requiring that the path joints are
// exactly the arm's joints (as a set).
Status columnIndices(
  const moveit::core::RobotModel & model, const std::vector<std::string> & path_joints,
  const ArmSpec & arm, std::vector<int> & idx)
{
  if (path_joints.size() != arm.joint_names.size())
    return Status::Error(
      "path has " + std::to_string(path_joints.size()) + " joints, arm " + arm.side + " has " +
      std::to_string(arm.joint_names.size()));
  const std::set<std::string> a(arm.joint_names.begin(), arm.joint_names.end());
  const std::set<std::string> p(path_joints.begin(), path_joints.end());
  if (a != p || p.size() != path_joints.size())
    return Status::Error("path joint names do not match the joints of arm " + arm.side);
  return variableIndices(model, path_joints, idx);
}

std::vector<std::string> armLinkNames(
  const moveit::core::RobotModel & model, const std::string & prefix, bool geometry_only)
{
  std::vector<std::string> out;
  for (const auto * l : model.getLinkModels())
  {
    if (!startsWith(l->getName(), prefix)) continue;
    if (geometry_only && l->getShapes().empty()) continue;
    out.push_back(l->getName());
  }
  return out;
}

}  // namespace

// ---- bounding spheres ------------------------------------------------------------------------

Status boundingSphereOfShape(const shapes::Shape & shape, Sphere & out)
{
  out = Sphere{};
  switch (shape.type)
  {
    case shapes::SPHERE:
    {
      const auto & s = static_cast<const shapes::Sphere &>(shape);
      out.radius = s.radius;
      break;
    }
    case shapes::CYLINDER:
    {
      const auto & s = static_cast<const shapes::Cylinder &>(shape);
      out.radius = std::hypot(s.radius, 0.5 * s.length);
      break;
    }
    case shapes::CONE:
    {
      // Origin at mid-height (FCL convention), so the circumscribing sphere of the cylinder
      // [radius, length] contains the cone.
      const auto & s = static_cast<const shapes::Cone &>(shape);
      out.radius = std::hypot(s.radius, 0.5 * s.length);
      break;
    }
    case shapes::BOX:
    {
      const auto & s = static_cast<const shapes::Box &>(shape);
      out.radius = 0.5 * std::sqrt(s.size[0] * s.size[0] + s.size[1] * s.size[1] + s.size[2] * s.size[2]);
      break;
    }
    case shapes::MESH:
    {
      const auto & m = static_cast<const shapes::Mesh &>(shape);
      if (m.vertex_count == 0 || m.vertices == nullptr)
        return Status::Error("mesh without vertices");
      Eigen::Vector3d lo(m.vertices[0], m.vertices[1], m.vertices[2]);
      Eigen::Vector3d hi = lo;
      for (unsigned int i = 0; i < m.vertex_count; ++i)
      {
        const Eigen::Vector3d v(m.vertices[3 * i], m.vertices[3 * i + 1], m.vertices[3 * i + 2]);
        if (!finite3(v)) return Status::Error("mesh with non-finite vertex");
        lo = lo.cwiseMin(v);
        hi = hi.cwiseMax(v);
      }
      out.center = 0.5 * (lo + hi);
      double r2 = 0.0;
      for (unsigned int i = 0; i < m.vertex_count; ++i)
      {
        const Eigen::Vector3d v(m.vertices[3 * i], m.vertices[3 * i + 1], m.vertices[3 * i + 2]);
        r2 = std::max(r2, (v - out.center).squaredNorm());
      }
      out.radius = std::sqrt(r2);
      break;
    }
    default:
      return Status::Error("unsupported collision geometry type (plane, octomap or unknown)");
  }
  if (!std::isfinite(out.radius) || out.radius < 0.0 || !finite3(out.center))
    return Status::Error("non-finite bounding sphere");
  return Status::Ok();
}

Sphere mergeSpheres(const Sphere & a, const Sphere & b)
{
  const Eigen::Vector3d d = b.center - a.center;
  const double dist = d.norm();
  if (dist + b.radius <= a.radius) return a;
  if (dist + a.radius <= b.radius) return b;
  Sphere out;
  out.radius = 0.5 * (dist + a.radius + b.radius);
  out.center = a.center + d * ((out.radius - a.radius) / dist);
  return out;
}

Status linkBoundingSphere(const moveit::core::LinkModel & link, Sphere & out, bool & has_geometry)
{
  out = Sphere{};
  has_geometry = false;
  const auto & shapes = link.getShapes();
  const auto & origins = link.getCollisionOriginTransforms();
  if (shapes.size() != origins.size())
    return Status::Error("link " + link.getName() + ": shape / origin count mismatch");
  for (size_t i = 0; i < shapes.size(); ++i)
  {
    if (!shapes[i]) return Status::Error("link " + link.getName() + ": null collision shape");
    Sphere s;
    Status st = boundingSphereOfShape(*shapes[i], s);
    if (!st) return Status::Error("link " + link.getName() + ": " + st.message);
    s.center = origins[i] * s.center;
    out = has_geometry ? mergeSpheres(out, s) : s;
    has_geometry = true;
  }
  return Status::Ok();
}

Status makeArmSpec(const moveit::core::RobotModel & model, const std::string & side, ArmSpec & out)
{
  out = ArmSpec{};
  const std::string prefix = armPrefix(side);
  if (prefix.empty()) return Status::Error("side must be 'left' or 'right'");
  out.side = side;
  out.prefix = prefix;
  for (const auto * jm : model.getActiveJointModels())
    if (startsWith(jm->getName(), prefix)) out.joint_names.push_back(jm->getName());
  if (out.joint_names.empty())
    return Status::Error("no active joint with prefix " + prefix + " in the robot model");
  return Status::Ok();
}

// ---- SweptVolumeChecker ----------------------------------------------------------------------

double effectiveMinDistance(double min_distance, double max_step_a_m, double max_step_b_m)
{
  return min_distance - 0.5 * (max_step_a_m + max_step_b_m);
}

Status SweptVolumeChecker::create(
  const planning_scene::PlanningSceneConstPtr & scene, const ArmSpec & arm_a, const ArmSpec & arm_b,
  const SweptVolumeConfig & config, std::unique_ptr<SweptVolumeChecker> & out)
{
  out.reset();
  if (!scene) return Status::Error("null planning scene");
  if (!(config.padding_m >= 0.0) || !std::isfinite(config.padding_m))
    return Status::Error("padding must be finite and >= 0");
  if (!(config.exact_distance_tolerance_m > 0.0) || !std::isfinite(config.exact_distance_tolerance_m))
    return Status::Error("exact_distance_tolerance_m must be finite and > 0 (never zero)");
  if (arm_a.prefix.empty() || arm_b.prefix.empty() || arm_a.prefix == arm_b.prefix ||
      startsWith(arm_a.prefix, arm_b.prefix) || startsWith(arm_b.prefix, arm_a.prefix))
    return Status::Error("the two arms need distinct, non-nested prefixes");

  std::unique_ptr<SweptVolumeChecker> c(new SweptVolumeChecker());
  c->scene_ = scene;
  c->model_ = scene->getRobotModel();
  c->arm_a_ = arm_a;
  c->arm_b_ = arm_b;
  c->config_ = config;

  Status s = variableIndices(*c->model_, arm_a.joint_names, c->vars_a_);
  if (!s) return s;
  s = variableIndices(*c->model_, arm_b.joint_names, c->vars_b_);
  if (!s) return s;

  auto build = [&](const ArmSpec & arm, std::vector<LinkSphere> & spheres,
                   std::vector<std::string> & names) -> Status
  {
    for (const auto * l : c->model_->getLinkModels())
    {
      if (!startsWith(l->getName(), arm.prefix)) continue;
      Sphere sp;
      bool has = false;
      Status st = linkBoundingSphere(*l, sp, has);
      if (!st) return st;  // unsupported geometry: fail closed
      if (!has) continue;
      LinkSphere ls;
      ls.link = l;
      ls.sphere = sp;
      spheres.push_back(ls);
      names.push_back(l->getName());
    }
    return Status::Ok();
  };
  s = build(arm_a, c->spheres_a_, c->names_a_);
  if (!s) return s;
  s = build(arm_b, c->spheres_b_, c->names_b_);
  if (!s) return s;
  if (c->spheres_a_.empty() || c->spheres_b_.empty())
    return Status::Error("an arm has no link with collision geometry");

  // ACM: every pair allowed, except the cross-arm link pairs.
  std::vector<std::string> all;
  for (const auto * l : c->model_->getLinkModels())
    if (!l->getShapes().empty()) all.push_back(l->getName());
  c->cross_only_acm_.setEntry(all, all, true);
  c->cross_only_acm_.setEntry(c->names_a_, c->names_b_, false);

  out = std::move(c);
  return Status::Ok();
}

void SweptVolumeChecker::centersFor(
  moveit::core::RobotState & state, const std::vector<LinkSphere> & spheres,
  std::vector<Eigen::Vector3d> & centers) const
{
  state.updateLinkTransforms();
  centers.resize(spheres.size());
  for (size_t l = 0; l < spheres.size(); ++l)
    centers[l] = state.getGlobalLinkTransform(spheres[l].link) * spheres[l].sphere.center;
}

Status SweptVolumeChecker::prepare(
  const SampledPath & path, const std::vector<int> & var_index,
  const std::vector<LinkSphere> & spheres, Prepared & out) const
{
  const size_t n = path.samples.size();
  const size_t L = spheres.size();
  out.centers.resize(n * L);
  out.arm_center.resize(n);
  out.arm_radius.resize(n);
  out.cum_disp.assign(n, 0.0);
  out.max_step = 0.0;
  moveit::core::RobotState state(model_);
  state.setToDefaultValues();
  std::vector<Eigen::Vector3d> c;
  std::vector<Eigen::Isometry3d> prev_tf(L), tf(L);
  for (size_t i = 0; i < n; ++i)
  {
    const auto & q = path.samples[i].q;
    if (q.size() != var_index.size())
      return Status::Error("sample " + std::to_string(i) + " has the wrong joint count");
    for (size_t k = 0; k < q.size(); ++k)
    {
      if (!std::isfinite(q[k])) return Status::Error("sample " + std::to_string(i) + " is not finite");
      state.setVariablePosition(var_index[k], q[k]);
    }
    centersFor(state, spheres, c);
    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    for (size_t l = 0; l < L; ++l)
    {
      out.centers[i * L + l] = c[l];
      mean += c[l];
      tf[l] = state.getGlobalLinkTransform(spheres[l].link);
    }
    if (i > 0)
    {
      // Displacement bound of any point of link l between samples i-1 and i: the origin translation
      // plus the rotation angle times the largest distance of its geometry from the link origin.
      double step = 0.0;
      for (size_t l = 0; l < L; ++l)
      {
        const double extent = spheres[l].sphere.center.norm() + spheres[l].sphere.radius;
        const double theta = Eigen::AngleAxisd(prev_tf[l].linear().transpose() * tf[l].linear()).angle();
        step = std::max(step, (tf[l].translation() - prev_tf[l].translation()).norm() + std::abs(theta) * extent);
      }
      out.cum_disp[i] = out.cum_disp[i - 1] + step;
      out.max_step = std::max(out.max_step, step);
    }
    prev_tf = tf;
    mean /= static_cast<double>(L);
    double r = 0.0;
    for (size_t l = 0; l < L; ++l) r = std::max(r, (c[l] - mean).norm() + spheres[l].sphere.radius);
    out.arm_center[i] = mean;
    out.arm_radius[i] = r;
  }
  return Status::Ok();
}

Status SweptVolumeChecker::exactAt(
  moveit::core::RobotState & state, double & distance, std::string & la, std::string & lb) const
{
  state.update();
  collision_detection::DistanceRequest req;
  req.enable_nearest_points = true;
  req.enable_signed_distance = true;
  req.type = collision_detection::DistanceRequestType::GLOBAL;
  req.acm = &cross_only_acm_;
  collision_detection::DistanceResult res;
  scene_->getCollisionEnvUnpadded()->distanceSelf(req, res, state);
  const double d = res.minimum_distance.distance;
  if (!std::isfinite(d) || d >= std::numeric_limits<double>::max() ||
      res.minimum_distance.link_names[0].empty() || res.minimum_distance.link_names[1].empty())
    return Status::Error("distance query returned no cross-arm pair");
  distance = d;
  la = res.minimum_distance.link_names[0];
  lb = res.minimum_distance.link_names[1];
  if (startsWith(la, arm_b_.prefix)) std::swap(la, lb);
  if (!startsWith(la, arm_a_.prefix) || !startsWith(lb, arm_b_.prefix))
    return Status::Error("distance query returned a non-cross pair: " + la + " / " + lb);
  return Status::Ok();
}

Status SweptVolumeChecker::exactDistance(
  const std::vector<double> & q_a, const std::vector<double> & q_b, double & distance,
  std::string & link_a, std::string & link_b) const
{
  if (q_a.size() != vars_a_.size() || q_b.size() != vars_b_.size())
    return Status::Error("configuration size does not match the arm joints");
  moveit::core::RobotState state(model_);
  state.setToDefaultValues();
  for (size_t k = 0; k < q_a.size(); ++k)
  {
    if (!std::isfinite(q_a[k])) return Status::Error("non-finite joint value");
    state.setVariablePosition(vars_a_[k], q_a[k]);
  }
  for (size_t k = 0; k < q_b.size(); ++k)
  {
    if (!std::isfinite(q_b[k])) return Status::Error("non-finite joint value");
    state.setVariablePosition(vars_b_[k], q_b[k]);
  }
  return exactAt(state, distance, link_a, link_b);
}

Status SweptVolumeChecker::sphereLowerBound(
  const std::vector<double> & q_a, const std::vector<double> & q_b, double & lower_bound,
  std::string & link_a, std::string & link_b) const
{
  if (q_a.size() != vars_a_.size() || q_b.size() != vars_b_.size())
    return Status::Error("configuration size does not match the arm joints");
  moveit::core::RobotState state(model_);
  state.setToDefaultValues();
  for (size_t k = 0; k < q_a.size(); ++k) state.setVariablePosition(vars_a_[k], q_a[k]);
  for (size_t k = 0; k < q_b.size(); ++k) state.setVariablePosition(vars_b_[k], q_b[k]);
  std::vector<Eigen::Vector3d> ca, cb;
  centersFor(state, spheres_a_, ca);
  centersFor(state, spheres_b_, cb);
  lower_bound = std::numeric_limits<double>::infinity();
  for (size_t a = 0; a < ca.size(); ++a)
    for (size_t b = 0; b < cb.size(); ++b)
    {
      const double lb = (ca[a] - cb[b]).norm() - spheres_a_[a].sphere.radius - spheres_b_[b].sphere.radius;
      if (lb < lower_bound)
      {
        lower_bound = lb;
        link_a = names_a_[a];
        link_b = names_b_[b];
      }
    }
  return Status::Ok();
}

Status SweptVolumeChecker::pairReport(
  const std::vector<double> & q_a, const std::vector<double> & q_b,
  std::vector<PairReport> & out) const
{
  out.clear();
  if (q_a.size() != vars_a_.size() || q_b.size() != vars_b_.size())
    return Status::Error("configuration size does not match the arm joints");
  moveit::core::RobotState state(model_);
  state.setToDefaultValues();
  for (size_t k = 0; k < q_a.size(); ++k) state.setVariablePosition(vars_a_[k], q_a[k]);
  for (size_t k = 0; k < q_b.size(); ++k) state.setVariablePosition(vars_b_[k], q_b[k]);
  state.update();
  std::vector<Eigen::Vector3d> ca, cb;
  centersFor(state, spheres_a_, ca);
  centersFor(state, spheres_b_, cb);

  collision_detection::DistanceRequest req;
  req.enable_nearest_points = true;
  req.enable_signed_distance = true;
  req.type = collision_detection::DistanceRequestType::SINGLE;
  req.acm = &cross_only_acm_;
  collision_detection::DistanceResult res;
  scene_->getCollisionEnvUnpadded()->distanceSelf(req, res, state);

  std::unordered_map<std::string, double> exact;  // "a|b" -> min distance
  for (const auto & kv : res.distances)
    for (const auto & d : kv.second)
    {
      std::string la = d.link_names[0];
      std::string lb = d.link_names[1];
      if (startsWith(la, arm_b_.prefix)) std::swap(la, lb);
      const std::string key = la + "|" + lb;
      auto it = exact.find(key);
      if (it == exact.end() || d.distance < it->second) exact[key] = d.distance;
    }

  for (size_t a = 0; a < ca.size(); ++a)
    for (size_t b = 0; b < cb.size(); ++b)
    {
      PairReport r;
      r.link_a = names_a_[a];
      r.link_b = names_b_[b];
      r.sphere_lower_bound =
        (ca[a] - cb[b]).norm() - spheres_a_[a].sphere.radius - spheres_b_[b].sphere.radius;
      auto it = exact.find(r.link_a + "|" + r.link_b);
      r.exact_distance = it == exact.end() ? std::numeric_limits<double>::quiet_NaN() : it->second;
      out.push_back(std::move(r));
    }
  return Status::Ok();
}

SweptVolumeResult SweptVolumeChecker::check(const SampledPath & path_a, const SampledPath & path_b) const
{
  SweptVolumeResult r;
  auto fail = [&](SweptCode code, const std::string & msg)
  {
    r.code = code;
    r.message = msg;
    return r;
  };

  if (path_a.samples.empty() || path_b.samples.empty())
    return fail(SweptCode::InvalidInput, "empty path");
  std::vector<int> idx_a, idx_b;
  Status s = columnIndices(*model_, path_a.joint_names, arm_a_, idx_a);
  if (!s) return fail(SweptCode::InvalidInput, "path A: " + s.message);
  s = columnIndices(*model_, path_b.joint_names, arm_b_, idx_b);
  if (!s) return fail(SweptCode::InvalidInput, "path B: " + s.message);

  const size_t na = path_a.samples.size();
  const size_t nb = path_b.samples.size();
  r.sample_pairs = na * nb;
  if (na > config_.max_sample_pairs || nb > config_.max_sample_pairs ||
      r.sample_pairs > config_.max_sample_pairs)
    return fail(
      SweptCode::PairCapExceeded,
      "sample pair count " + std::to_string(na) + " x " + std::to_string(nb) + " exceeds cap " +
        std::to_string(config_.max_sample_pairs));

  Prepared pa, pb;
  s = prepare(path_a, idx_a, spheres_a_, pa);
  if (!s) return fail(SweptCode::InvalidInput, "path A: " + s.message);
  s = prepare(path_b, idx_b, spheres_b_, pb);
  if (!s) return fail(SweptCode::InvalidInput, "path B: " + s.message);

  const size_t La = spheres_a_.size();
  const size_t Lb = spheres_b_.size();
  const double padding = config_.padding_m;
  const double tol = config_.exact_distance_tolerance_m;
  // The continuous motion between two samples can come closer than any sampled pair by up to half
  // the sum of the per-arm step displacement bounds: verdict on effective_min = min - slack, i.e. a
  // sample-level threshold of padding + slack everywhere below.
  const double slack = 0.5 * (pa.max_step + pb.max_step);
  const double thr = padding + slack;

  struct Cand
  {
    double lb;
    uint32_t i, j;
  };
  std::vector<Cand> cands;

  // Lower bound over every sample pair that is NOT a candidate (all of its bounds >= padding).
  double clear_lb = std::numeric_limits<double>::infinity();
  std::string clear_la, clear_lb_name;
  size_t clear_i = 0, clear_j = 0;

  for (size_t i = 0; i < na; ++i)
  {
    for (size_t j = 0; j < nb; ++j)
    {
      const double coarse = (pa.arm_center[i] - pb.arm_center[j]).norm() - pa.arm_radius[i] - pb.arm_radius[j];
      if (coarse >= thr)
      {
        if (coarse < clear_lb)
        {
          clear_lb = coarse;
          clear_la.clear();
          clear_lb_name.clear();
          clear_i = i;
          clear_j = j;
        }
        continue;
      }
      double best = std::numeric_limits<double>::infinity();
      size_t ba = 0, bb = 0;
      for (size_t a = 0; a < La; ++a)
      {
        const Eigen::Vector3d & ca = pa.centers[i * La + a];
        const double ra = spheres_a_[a].sphere.radius;
        for (size_t b = 0; b < Lb; ++b)
        {
          const double lb = (ca - pb.centers[j * Lb + b]).norm() - ra - spheres_b_[b].sphere.radius;
          if (lb < best)
          {
            best = lb;
            ba = a;
            bb = b;
          }
        }
      }
      if (best < thr)
      {
        if (cands.size() >= config_.max_candidate_pairs)
        {
          r.candidate_pairs = cands.size() + 1;
          return fail(
            SweptCode::PairCapExceeded,
            "more than " + std::to_string(config_.max_candidate_pairs) +
              " candidate sample pairs closer than the padding (fail closed)");
        }
        cands.push_back(Cand{best, static_cast<uint32_t>(i), static_cast<uint32_t>(j)});
      }
      else if (best < clear_lb)
      {
        clear_lb = best;
        clear_la = names_a_[ba];
        clear_lb_name = names_b_[bb];
        clear_i = i;
        clear_j = j;
      }
    }
  }
  r.candidate_pairs = cands.size();
  r.max_step_a_m = pa.max_step;
  r.max_step_b_m = pb.max_step;

  std::sort(cands.begin(), cands.end(), [](const Cand & x, const Cand & y) { return x.lb < y.lb; });

  double exact_min = std::numeric_limits<double>::infinity();
  std::string ex_la, ex_lb;
  size_t ex_i = 0, ex_j = 0;
  if (!cands.empty())
  {
    struct Cert
    {
      double ua, ub, d;
      uint32_t i, j;
      std::string la, lb;
    };
    std::vector<Cert> certs;
    moveit::core::RobotState state(model_);
    state.setToDefaultValues();
    for (const Cand & c : cands)
    {
      if (config_.use_certificates)
      {
        // Best lower bound on this pair from any earlier exact distance >= threshold (tolerance
        // already subtracted in the stored value).
        double bound = -std::numeric_limits<double>::infinity();
        const Cert * from = nullptr;
        const double ua = pa.cum_disp[c.i];
        const double ub = pb.cum_disp[c.j];
        for (const Cert & k : certs)
        {
          const double b = k.d - std::abs(ua - k.ua) - std::abs(ub - k.ub);
          if (b > bound)
          {
            bound = b;
            from = &k;
            if (bound >= thr) break;
          }
        }
        if (from && bound >= thr)
        {
          ++r.certified_pairs;
          if (bound < clear_lb)
          {
            clear_lb = bound;
            clear_la = from->la;
            clear_lb_name = from->lb;
            clear_i = c.i;
            clear_j = c.j;
          }
          continue;
        }
      }
      if (r.exact_checks >= config_.max_exact_checks)
        return fail(
          SweptCode::PairCapExceeded,
          "more than " + std::to_string(config_.max_exact_checks) +
            " exact distance queries needed (fail closed)");
      const auto & qa = path_a.samples[c.i].q;
      const auto & qb = path_b.samples[c.j].q;
      for (size_t k = 0; k < qa.size(); ++k) state.setVariablePosition(idx_a[k], qa[k]);
      for (size_t k = 0; k < qb.size(); ++k) state.setVariablePosition(idx_b[k], qb[k]);
      double d = 0.0;
      std::string la, lb;
      s = exactAt(state, d, la, lb);
      ++r.exact_checks;
      if (!s) return fail(SweptCode::QueryFailed, s.message);
      d -= tol;  // FCL's distance is not trusted to be exact: subtract the tolerance EVERYWHERE
      if (d < exact_min)
      {
        exact_min = d;
        ex_la = la;
        ex_lb = lb;
        ex_i = c.i;
        ex_j = c.j;
        if (config_.stop_at_first_violation && d < thr) break;
      }
      if (config_.use_certificates && d >= thr)
        certs.push_back(Cert{pa.cum_disp[c.i], pb.cum_disp[c.j], d, c.i, c.j, la, lb});
    }
  }

  if (exact_min <= clear_lb)
  {
    r.min_distance = exact_min;
    r.min_distance_exact = true;
    r.link_a = ex_la;
    r.link_b = ex_lb;
    r.sample_index_a = ex_i;
    r.sample_index_b = ex_j;
  }
  else
  {
    r.min_distance = clear_lb;
    r.min_distance_exact = false;
    r.link_a = clear_la;
    r.link_b = clear_lb_name;
    r.sample_index_a = clear_i;
    r.sample_index_b = clear_j;
  }
  r.sampling_gap_m = slack;
  r.effective_min_distance = effectiveMinDistance(r.min_distance, pa.max_step, pb.max_step);
  if (r.effective_min_distance < padding)
  {
    r.code = SweptCode::Violation;
    r.message = "effective cross-arm distance " + std::to_string(r.effective_min_distance) +
                " m (sampled min " + std::to_string(r.min_distance) + " m - sampling gap " +
                std::to_string(slack) + " m) < padding " + std::to_string(padding) + " m";
  }
  else
  {
    r.code = SweptCode::Ok;
    r.message = "ok";
  }
  return r;
}

// ---- WorldChecker ----------------------------------------------------------------------------

Status WorldChecker::create(
  const planning_scene::PlanningSceneConstPtr & scene, const ArmSpec & arm, const ArmSpec & other_arm,
  const WorldCheckConfig & config, std::unique_ptr<WorldChecker> & out)
{
  out.reset();
  if (!scene) return Status::Error("null planning scene");
  if (arm.prefix.empty() || other_arm.prefix.empty() || arm.prefix == other_arm.prefix ||
      startsWith(arm.prefix, other_arm.prefix) || startsWith(other_arm.prefix, arm.prefix))
    return Status::Error("the two arms need distinct, non-nested prefixes");
  std::unique_ptr<WorldChecker> c(new WorldChecker());
  c->scene_ = scene;
  c->model_ = scene->getRobotModel();
  c->arm_ = arm;
  c->other_ = other_arm;
  c->config_ = config;
  c->arm_links_ = armLinkNames(*c->model_, arm.prefix, false);
  c->other_links_ = armLinkNames(*c->model_, other_arm.prefix, false);
  if (c->arm_links_.empty()) return Status::Error("no links with prefix " + arm.prefix);
  for (const auto & l : config.contact_link_names)
    if (std::find(c->arm_links_.begin(), c->arm_links_.end(), l) == c->arm_links_.end())
      return Status::Error("contact link '" + l + "' is not a link of arm " + arm.side);
  out = std::move(c);
  return Status::Ok();
}

WorldCheckResult WorldChecker::check(const SampledPath & path, const std::vector<double> & other_q) const
{
  WorldCheckResult r;
  auto fail = [&](WorldCode code, const std::string & msg)
  {
    r.code = code;
    r.message = msg;
    return r;
  };
  if (path.samples.empty()) return fail(WorldCode::InvalidInput, "empty path");
  if (path.samples.size() > config_.max_samples)
    return fail(
      WorldCode::TooManySamples,
      "path has " + std::to_string(path.samples.size()) + " samples, cap " +
        std::to_string(config_.max_samples) + " (fail closed)");

  std::vector<int> idx, other_idx;
  Status s = columnIndices(*model_, path.joint_names, arm_, idx);
  if (!s) return fail(WorldCode::InvalidInput, s.message);
  s = variableIndices(*model_, other_.joint_names, other_idx);
  if (!s) return fail(WorldCode::InvalidInput, s.message);
  if (other_q.size() != other_idx.size())
    return fail(WorldCode::InvalidInput, "other arm configuration has the wrong size");
  for (double v : other_q)
    if (!std::isfinite(v)) return fail(WorldCode::InvalidInput, "other arm configuration not finite");

  bool any_contact = false;
  for (const auto & smp : path.samples) any_contact = any_contact || smp.contact_allowed;
  if (any_contact && config_.contact_link_names.empty())
    return fail(
      WorldCode::InvalidInput, "contact_allowed samples but no contact_link_names configured");

  // Everything involving the other arm is allowed (it is held fixed; the cross-arm check owns it).
  collision_detection::AllowedCollisionMatrix acm(scene_->getAllowedCollisionMatrix());
  for (const auto & l : other_links_)
  {
    acm.setDefaultEntry(l, true);
    acm.setEntry(l, true);
  }
  // Static scene geometry is not this arm's business: pairs of a non-arm robot link (e.g. the table)
  // with a world object (e.g. the camera pole standing on it) are allowed. Arm links against these
  // static links still follow the scene's ACM.
  for (const auto * l : model_->getLinkModels())
  {
    const std::string & n = l->getName();
    if (startsWith(n, arm_.prefix) || startsWith(n, other_.prefix) || l->getShapes().empty()) continue;
    for (const auto & id : scene_->getWorld()->getObjectIds()) acm.setEntry(n, id, true);
  }
  collision_detection::AllowedCollisionMatrix acm_contact(acm);
  for (const auto & l : config_.contact_link_names)
    for (const auto & o : config_.contact_object_ids) acm_contact.setEntry(l, o, true);

  moveit::core::RobotState state(model_);
  state.setToDefaultValues();
  for (size_t k = 0; k < other_idx.size(); ++k) state.setVariablePosition(other_idx[k], other_q[k]);

  const auto env = scene_->getCollisionEnvUnpadded();
  for (size_t i = 0; i < path.samples.size(); ++i)
  {
    const auto & smp = path.samples[i];
    if (smp.q.size() != idx.size()) return fail(WorldCode::InvalidInput, "sample has the wrong joint count");
    for (size_t k = 0; k < idx.size(); ++k)
    {
      if (!std::isfinite(smp.q[k])) return fail(WorldCode::InvalidInput, "sample is not finite");
      state.setVariablePosition(idx[k], smp.q[k]);
    }
    state.update();
    collision_detection::CollisionRequest req;
    const bool tolerant = config_.penetration_tolerance_m > 0.0;
    req.contacts = true;
    req.max_contacts = tolerant ? 200 : 1;   // with a tolerance every contact of the sample is needed
    req.max_contacts_per_pair = tolerant ? 4 : 1;
    collision_detection::CollisionResult res;
    env->checkCollision(req, res, state, smp.contact_allowed ? acm_contact : acm);
    ++r.samples_checked;
    bool collision = res.collision;
    if (collision && tolerant)
    {
      collision = false;
      double sample_max = 0.0;
      for (const auto & kv : res.contacts)
        for (const auto & c : kv.second)
        {
          sample_max = std::max(sample_max, c.depth);
          if (c.depth > config_.penetration_tolerance_m) collision = true;
        }
      if (!collision && !res.contacts.empty())
      {
        ++r.tolerated_samples;
        if (sample_max > r.max_tolerated_depth_m)
        {
          r.max_tolerated_depth_m = sample_max;
          r.max_tolerated_sample = i;
        }
      }
    }
    if (collision)
    {
      r.code = WorldCode::Collision;
      r.sample_index = i;
      r.time_s = smp.time_s;
      r.sample_contact_allowed = smp.contact_allowed;
      r.segment_index = smp.segment_index;
      if (!res.contacts.empty())
      {
        // the deepest contact (with a tolerance the first one may be a harmless graze)
        auto best = res.contacts.begin();
        double best_depth = -1.0;
        for (auto it = res.contacts.begin(); it != res.contacts.end(); ++it)
          for (const auto & c : it->second)
            if (c.depth > best_depth)
            {
              best_depth = c.depth;
              best = it;
            }
        r.link = best->first.first;
        r.object = best->first.second;
        if (!best->second.empty()) {
          auto deepest = std::max_element(best->second.begin(), best->second.end(),
                                          [](const auto & x, const auto & y) { return x.depth < y.depth; });
          r.depth_m = deepest->depth;
          r.contact_pos = deepest->pos;
          r.contact_normal = deepest->normal;
        }
      }
      r.message = "collision at sample " + std::to_string(i) + " between " + r.link + " and " + r.object;
      return r;
    }
  }
  r.code = WorldCode::Ok;
  r.message = "ok";
  return r;
}

}  // namespace dual_cut
