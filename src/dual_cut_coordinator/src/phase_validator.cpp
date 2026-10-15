#include "dual_cut_coordinator/phase_validator.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace dual_cut
{

Status PhaseValidator::create(
  const planning_scene::PlanningSceneConstPtr & scene, const PhaseValidatorConfig & config,
  std::unique_ptr<PhaseValidator> & out)
{
  out.reset();
  if (!scene) return Status::Error("null planning scene");
  std::unique_ptr<PhaseValidator> v(new PhaseValidator());
  v->scene_ = scene;
  v->config_ = config;
  const auto model = scene->getRobotModel();
  Status s = makeArmSpec(*model, "left", v->left_);
  if (!s) return s;
  s = makeArmSpec(*model, "right", v->right_);
  if (!s) return s;
  for (const ArmSpec * a : {&v->left_, &v->right_})
    if (!model->hasLinkModel(a->prefix + config.tip_link_suffix))
      return Status::Error("model has no tip link " + a->prefix + config.tip_link_suffix);
  s = SweptVolumeChecker::create(scene, v->left_, v->right_, config.swept, v->swept_);
  if (!s) return s;
  WorldCheckConfig wl;
  wl.contact_object_ids = config.contact_object_ids;
  wl.contact_link_names = config.contact_links_left;
  wl.max_samples = config.world_max_samples;
  wl.penetration_tolerance_m = config.world_penetration_tolerance_m;
  s = WorldChecker::create(scene, v->left_, v->right_, wl, v->world_left_);
  if (!s) return s;
  WorldCheckConfig wr = wl;
  wr.contact_link_names = config.contact_links_right;
  s = WorldChecker::create(scene, v->right_, v->left_, wr, v->world_right_);
  if (!s) return s;
  out = std::move(v);
  return Status::Ok();
}

Status PhaseValidator::orderedQ(
  const ArmSpec & arm, const std::vector<std::string> & names, const std::vector<double> & q,
  std::vector<double> & out) const
{
  if (names.size() != q.size()) return Status::Error("joint name / value count mismatch");
  out.assign(arm.joint_names.size(), 0.0);
  for (size_t i = 0; i < arm.joint_names.size(); ++i)
  {
    std::string plain;
    Status s = stripJointPrefix(arm.prefix, arm.joint_names[i], plain);
    if (!s) return s;
    auto it = std::find(names.begin(), names.end(), plain);
    if (it == names.end()) return Status::Error("joint '" + plain + "' missing in the plan state");
    out[i] = q[static_cast<size_t>(it - names.begin())];
  }
  return Status::Ok();
}

Status PhaseValidator::armPath(const ArmPhase & arm, SampledPath & out) const
{
  std::vector<SegmentInput> in;
  Status s = arm.prefixedSegments(in);
  if (!s) return s;
  return buildSampledPath(in, config_.sample, config_.concat, out);
}

Status PhaseValidator::parkedPath(
  const ArmSpec & spec, const ArmPhase & arm, const std::vector<double> & q, SampledPath & out) const
{
  std::vector<double> ordered;
  Status s = orderedQ(spec, arm.joint_names, q, ordered);
  if (!s) return s;
  out = SampledPath{};
  out.joint_names = spec.joint_names;
  PathSample ps;
  ps.q = ordered;
  out.samples.push_back(std::move(ps));
  return Status::Ok();
}

TipFk PhaseValidator::tipFk(const ArmSpec & spec) const
{
  const auto model = scene_->getRobotModel();
  const std::string link = spec.prefix + config_.tip_link_suffix;
  const std::string prefix = spec.prefix;
  return [model, link, prefix](const std::vector<std::string> & names, const std::vector<double> & q,
                               Eigen::Vector3d & tip) -> Status
  {
    if (names.size() != q.size()) return Status::Error("FK: name / value size mismatch");
    moveit::core::RobotState state(model);
    state.setToDefaultValues();
    for (size_t i = 0; i < names.size(); ++i)
    {
      std::string full = names[i];
      if (full.compare(0, prefix.size(), prefix) != 0) full = prefix + names[i];
      if (!model->hasJointModel(full)) return Status::Error("FK: unknown joint " + full);
      state.setVariablePosition(full, q[i]);
    }
    state.update();
    tip = state.getGlobalLinkTransform(link).translation();
    return Status::Ok();
  };
}

PhaseValidation PhaseValidator::validate(
  const ArmPhase & left, const ArmPhase & right, const std::vector<CutLine> & left_lines,
  const std::vector<CutLine> & right_lines) const
{
  PhaseValidation v;
  auto fail = [&](const std::string & msg)
  {
    v.ok = false;
    v.message = msg;
    return v;
  };
  if (left.side != "left" || right.side != "right") return fail("arms are not (left, right)");

  SampledPath pl, pr;
  Status s = armPath(left, pl);
  if (!s) return fail("left path: " + s.message);
  s = armPath(right, pr);
  if (!s) return fail("right path: " + s.message);
  v.samples_left = pl.samples.size();
  v.samples_right = pr.samples.size();

  std::vector<double> left_start, right_start;
  if (!(s = orderedQ(left_, left.joint_names, left.start_q, left_start))) return fail(s.message);
  if (!(s = orderedQ(right_, right.joint_names, right.start_q, right_start))) return fail(s.message);

  // 1. world check per arm, the other arm held at its phase start
  v.world_left = world_left_->check(pl, right_start);
  v.world_right = world_right_->check(pr, left_start);
  // 2. tip-on-line gate
  v.tip_left = checkTipOnLine(left, left_lines, tipFk(left_), config_.tip_tolerance_m, config_.sample);
  v.tip_right = checkTipOnLine(right, right_lines, tipFk(right_), config_.tip_tolerance_m, config_.sample);
  // 3. cross-arm swept volume in parallel
  v.cross = swept_->check(pl, pr);

  // Name the segment of a world collision (the sample's segment_index indexes the arm phase's segments).
  auto describe = [](const WorldCheckResult & w, const ArmPhase & arm)
  {
    std::ostringstream o;
    o << w.message;
    if (w.code == WorldCode::Collision && w.segment_index < arm.segments.size())
    {
      const auto & seg = arm.segments[w.segment_index];
      o << " [segment " << w.segment_index << " " << toString(seg.kind) << " '" << seg.stage_name << "', t="
        << w.time_s << " s, contact " << (w.sample_contact_allowed ? "allowed" : "not allowed") << ", depth "
        << w.depth_m * 1e3 << " mm at (" << w.contact_pos.x() << ", " << w.contact_pos.y() << ", " << w.contact_pos.z()
        << "), normal (" << w.contact_normal.x() << ", " << w.contact_normal.y() << ", " << w.contact_normal.z() << ")]";
    }
    return o.str();
  };
  std::ostringstream why;
  bool ok = true;
  if (!v.world_left.safe()) { ok = false; why << "left world check: " << describe(v.world_left, left) << "; "; }
  if (!v.world_right.safe()) { ok = false; why << "right world check: " << describe(v.world_right, right) << "; "; }
  if (!v.tip_left.safe()) { ok = false; why << "left " << v.tip_left.message << "; "; }
  if (!v.tip_right.safe()) { ok = false; why << "right " << v.tip_right.message << "; "; }

  bool cross_ok = v.cross.safe();
  if (cross_ok)
  {
    v.min_cross_distance_m = v.cross.effective_min_distance;
  }
  else if (config_.allow_serial_fallback)
  {
    // One arm after the other. The mover is checked against the other arm parked where it is during
    // that stretch: first mover vs the other's start, second mover vs the first's end.
    SampledPath park_l_start, park_r_start, park_l_end, park_r_end;
    if (parkedPath(left_, left, left.start_q, park_l_start) && parkedPath(right_, right, right.start_q, park_r_start) &&
        parkedPath(left_, left, left.end_q, park_l_end) && parkedPath(right_, right, right.end_q, park_r_end))
    {
      const SweptVolumeResult l1 = swept_->check(pl, park_r_start);
      const SweptVolumeResult r2 = swept_->check(park_l_end, pr);
      if (l1.safe() && r2.safe())
      {
        v.serial = true;
        v.serial_first = "left";
        v.serial_first_result = l1;
        v.serial_second_result = r2;
      }
      else
      {
        const SweptVolumeResult r1 = swept_->check(park_l_start, pr);
        const SweptVolumeResult l2 = swept_->check(pl, park_r_end);
        if (r1.safe() && l2.safe())
        {
          v.serial = true;
          v.serial_first = "right";
          v.serial_first_result = r1;
          v.serial_second_result = l2;
        }
      }
      if (v.serial)
      {
        cross_ok = true;
        v.min_cross_distance_m =
          std::min(v.serial_first_result.effective_min_distance, v.serial_second_result.effective_min_distance);
      }
    }
  }
  if (!cross_ok)
  {
    ok = false;
    why << "cross-arm check failed in parallel" << (config_.allow_serial_fallback ? " and in both serial orders" : "")
        << ": " << v.cross.message << "; ";
  }
  v.ok = ok;
  if (ok)
  {
    v.message = v.serial ? "ok (serial: " + v.serial_first + " arm first)" : "ok (parallel)";
  }
  else
  {
    v.message = why.str();
  }
  return v;
}

ValidationRecord PhaseValidator::toRecord(const PhaseValidation & v)
{
  ValidationRecord r;
  r.cross_arm_ok = v.cross.safe() || v.serial;
  r.world_ok_left = v.world_left.safe();
  r.world_ok_right = v.world_right.safe();
  r.tip_on_line_ok = v.tip_left.safe() && v.tip_right.safe();
  r.serial_fallback = v.serial;
  r.serial_first = v.serial ? v.serial_first : std::string();
  r.min_cross_distance_m = std::isfinite(v.min_cross_distance_m) ? v.min_cross_distance_m : v.cross.effective_min_distance;
  r.note = v.message;
  return r;
}

}  // namespace dual_cut
