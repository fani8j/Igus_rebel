#include "dual_cut_coordinator/phase_validator.hpp"

#include <cstring>

#include "dual_cut_coordinator/cri_transit.hpp"
#include "dual_cut_coordinator/sequence_plan.hpp"

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
  wl.table_link_name = config.table_link_name;
  wl.table_min_clearance_m = config.table_min_clearance_m;
  wl.table_exempt_links = config.table_exempt_links_left;
  wl.object_clearance_m = config.world_object_clearance_m;
  s = WorldChecker::create(scene, v->left_, v->right_, wl, v->world_left_);
  if (!s) return s;
  WorldCheckConfig wr = wl;
  wr.contact_link_names = config.contact_links_right;
  wr.table_exempt_links = config.table_exempt_links_right;
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

namespace
{
uint64_t fnv(uint64_t h, const void * data, size_t n)
{
  const unsigned char * p = static_cast<const unsigned char *>(data);
  for (size_t i = 0; i < n; ++i)
  {
    h ^= p[i];
    h *= 1099511628211ull;
  }
  return h;
}
// Content key of a segment's geometry (positions of every point; times do not matter for the all-pairs check).
uint64_t segmentKey(const PlannedSegmentData & seg, uint64_t salt)
{
  uint64_t h = 14695981039346656037ull;
  h = fnv(h, &salt, sizeof(salt));
  const size_t nj = seg.trajectory.joint_names.size();
  h = fnv(h, &nj, sizeof(nj));
  for (const auto & pt : seg.trajectory.points)
    for (double v : pt.positions) h = fnv(h, &v, sizeof(v));
  return h;
}
}  // namespace

WindowCrossResult PhaseValidator::crossWindows(
  const ArmPhase & left, const ArmPhase & right, double margin_s, SegmentPairCache * cache) const
{
  WindowCrossResult out;
  SegmentPairCache local;
  SegmentPairCache & c = cache ? *cache : local;
  auto error = [&](const std::string & msg)
  {
    out.worst = SweptVolumeResult{};
    out.worst.code = SweptCode::InvalidInput;
    out.worst.message = msg;
    return out;
  };
  if (!(margin_s >= 0.0) || !std::isfinite(margin_s)) return error("segment window margin must be finite and >= 0");
  const std::vector<SegWindow> wl = segmentWindows(left, margin_s), wr = segmentWindows(right, margin_s);

  // key and path of segment (or tail) i of an arm
  struct Seg
  {
    uint64_t key;
    const SampledPath * path;
  };
  auto get = [&](const ArmPhase & arm, const ArmSpec & spec, const SegWindow & w, Seg & seg) -> Status
  {
    uint64_t key;
    if (w.tail)
    {
      PlannedSegmentData pose;
      pose.trajectory.joint_names = arm.joint_names;
      TrajPoint tp;
      tp.positions = arm.end_q;
      pose.trajectory.points = {tp};
      key = segmentKey(pose, 0x7a11ull + (spec.side == "left" ? 1 : 2));
    }
    else
    {
      key = segmentKey(arm.segments[w.index], spec.side == "left" ? 1 : 2);
    }
    auto it = c.paths.find(key);
    if (it == c.paths.end())
    {
      SampledPath p;
      Status s;
      if (w.tail) s = parkedPath(spec, arm, arm.end_q, p);
      else
      {
        ArmPhase one;
        one.side = arm.side;
        one.joint_names = arm.joint_names;
        one.segments = {arm.segments[w.index]};
        s = armPath(one, p);
      }
      if (!s) return s;
      it = c.paths.emplace(key, std::move(p)).first;
    }
    seg.key = key;
    seg.path = &it->second;
    return Status::Ok();
  };

  bool have_worst = false;
  for (const SegWindow & a : wl)
    for (const SegWindow & b : wr)
    {
      if (!windowsOverlap(a, b)) continue;
      ++out.overlapping_pairs;
      Seg sa, sb;
      Status s = get(left, left_, a, sa);
      if (!s) return error("left segment '" + a.name + "': " + s.message);
      s = get(right, right_, b, sb);
      if (!s) return error("right segment '" + b.name + "': " + s.message);
      const auto pk = std::make_pair(sa.key, sb.key);
      auto it = c.pairs.find(pk);
      if (it == c.pairs.end()) it = c.pairs.emplace(pk, swept_->check(*sa.path, *sb.path)).first;
      const SweptVolumeResult & r = it->second;
      WindowPair wp;
      wp.left_index = a.index;
      wp.right_index = b.index;
      wp.left_name = a.name;
      wp.right_name = b.name;
      wp.safe = r.safe();
      wp.effective_min_m = r.effective_min_distance;
      out.pairs.push_back(wp);
      // worst: an error beats a violation beats the smallest distance
      auto rank = [](const SweptVolumeResult & x) { return x.code == SweptCode::Ok ? 0 : x.code == SweptCode::Violation ? 1 : 2; };
      const bool better =
        !have_worst || rank(r) > rank(out.worst) ||
        (rank(r) == rank(out.worst) && r.effective_min_distance < out.worst.effective_min_distance);
      if (better)
      {
        out.worst = r;
        out.worst_left = a.name;
        out.worst_right = b.name;
        have_worst = true;
      }
    }
  if (!have_worst)
  {
    out.worst = SweptVolumeResult{};
    out.worst.code = SweptCode::Ok;
    out.worst.message = "no overlapping segment windows";
    out.worst.min_distance = out.worst.effective_min_distance = std::numeric_limits<double>::infinity();
  }
  return out;
}

PhaseValidation PhaseValidator::validateSequence(
  const ArmPhase & left, const ArmPhase & right, const std::vector<CutLine> & left_lines,
  const std::vector<CutLine> & right_lines, double margin_s, SegmentPairCache * cache, WindowCrossResult * windows) const
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
  v.world_left = world_left_->check(pl, right_start);
  v.world_right = world_right_->check(pr, left_start);
  v.tip_left = checkTipOnLine(left, left_lines, tipFk(left_), config_.tip_tolerance_m, config_.sample);
  v.tip_right = checkTipOnLine(right, right_lines, tipFk(right_), config_.tip_tolerance_m, config_.sample);
  WindowCrossResult wc = crossWindows(left, right, margin_s, cache);
  v.cross = wc.worst;
  std::ostringstream why;
  bool ok = true;
  if (!v.world_left.safe()) { ok = false; why << "left world check: " << v.world_left.message << "; "; }
  if (!v.world_right.safe()) { ok = false; why << "right world check: " << v.world_right.message << "; "; }
  if (!v.tip_left.safe()) { ok = false; why << "left " << v.tip_left.message << "; "; }
  if (!v.tip_right.safe()) { ok = false; why << "right " << v.tip_right.message << "; "; }
  if (v.cross.safe()) v.min_cross_distance_m = v.cross.effective_min_distance;
  else
  {
    ok = false;
    why << "segment-window cross-arm check failed (left '" << wc.worst_left << "' vs right '" << wc.worst_right << "'): " << v.cross.message << "; ";
  }
  v.ok = ok;
  v.message = ok ? "ok (sequence, segment windows)" : why.str();
  if (windows) *windows = std::move(wc);
  return v;
}

ChordCheck PhaseValidator::checkChord(
  const std::string & side, const std::vector<std::string> & joint_names, const std::vector<double> & start,
  const std::vector<double> & goal, const std::vector<std::string> & other_joint_names,
  const std::vector<double> & other_parked_q, double max_step_rad) const
{
  ChordCheck c;
  auto fail = [&](const std::string & why)
  {
    c.ok = false;
    c.reason = why;
    return c;
  };
  if (side != "left" && side != "right") return fail("side must be left or right");
  const bool is_left = side == "left";
  const ArmSpec & arm = is_left ? left_ : right_;
  const ArmSpec & other = is_left ? right_ : left_;
  std::vector<double> a, b, parked;
  Status s = orderedQ(arm, joint_names, start, a);
  if (s) s = orderedQ(arm, joint_names, goal, b);
  if (s) s = orderedQ(other, other_joint_names, other_parked_q, parked);
  if (!s) return fail(s.message);
  const auto chord = resampleChord(a, b, max_step_rad);
  if (chord.size() < 2) return fail("chord could not be sampled");
  SampledPath path;
  path.joint_names = arm.joint_names;
  for (size_t k = 0; k < chord.size(); ++k)
  {
    PathSample ps;
    ps.time_s = static_cast<double>(k);
    ps.q = chord[k];
    path.samples.push_back(std::move(ps));
  }
  c.samples = path.samples.size();
  SampledPath park;
  park.joint_names = other.joint_names;
  {
    PathSample ps;
    ps.q = parked;
    park.samples.push_back(std::move(ps));
  }
  c.world = (is_left ? world_left_ : world_right_)->check(path, parked);
  c.cross = is_left ? swept_->check(path, park) : swept_->check(park, path);
  c.cross_required_m = config_.swept.padding_m;
  c.cross_effective_min_m = c.cross.effective_min_distance;
  if (config_.table_min_clearance_m > 0.0)
  {
    c.table_required_m = config_.table_min_clearance_m;
    if (c.world.table_measured) c.table_min_m = c.world.table_min_distance_m;
  }
  if (config_.world_object_clearance_m > 0.0)
  {
    c.object_required_m = config_.world_object_clearance_m;
    if (c.world.object_measured) c.object_min_m = c.world.object_min_distance_m;
  }
  if (!c.world.safe()) return fail("world: " + c.world.message);
  if (!c.cross.safe()) return fail("cross-arm (other arm parked): " + c.cross.message);
  c.ok = true;
  return c;
}

ChordStretch PhaseValidator::certifyChord(
  const std::string & side, const std::vector<std::string> & joint_names, const std::vector<double> & start,
  const std::vector<double> & goal, const std::vector<std::string> & other_joint_names,
  const std::vector<double> & other_parked_q, double max_step_rad, double min_length_rad) const
{
  ChordStretch out;
  out.check = checkChord(side, joint_names, start, goal, other_joint_names, other_parked_q, max_step_rad);
  const auto chord = resampleChord(start, goal, max_step_rad);   // plan order, as given
  if (chord.size() < 2)
  {
    out.reason = out.check.reason.empty() ? "chord could not be sampled" : out.check.reason;
    return out;
  }
  out.total_steps = chord.size() - 1;
  if (out.check.ok)
  {
    out.found = out.full = true;
    out.first_sample = 0;
    out.last_sample = out.total_steps;
    out.s0 = 0.0;
    out.s1 = 1.0;
    out.start = start;
    out.goal = goal;
    out.length_rad = maxAbsDiff(start, goal);
    return out;
  }
  const std::string full_reason = out.check.reason;
  // Only a clean failure can be refined: invalid input or a cap hit fail closed.
  if (side != "left" && side != "right") { out.reason = full_reason; return out; }
  // sample by sample (a one-sample chord is the same two checks without a swept gap)
  std::vector<char> pass(chord.size(), 0);
  for (size_t k = 0; k < chord.size(); ++k)
  {
    const ChordCheck c = checkChord(side, joint_names, chord[k], chord[k], other_joint_names, other_parked_q, max_step_rad);
    pass[k] = c.ok ? 1 : 0;
  }
  // longest contiguous run
  size_t best_a = 0, best_b = 0;
  bool have = false;
  for (size_t k = 0; k < pass.size();)
  {
    if (!pass[k]) { ++k; continue; }
    size_t e = k;
    while (e + 1 < pass.size() && pass[e + 1]) ++e;
    if (!have || e - k > best_b - best_a) { best_a = k; best_b = e; have = true; }
    k = e + 1;
  }
  const double chord_len = maxAbsDiff(start, goal);
  auto stretchLen = [&](size_t a, size_t b) { return chord_len * static_cast<double>(b - a) / static_cast<double>(out.total_steps); };
  if (!have)
  {
    out.reason = "no sample of the chord passes the padded checks (" + full_reason + ")";
    return out;
  }
  // re-check the stretch as a chord of its own; the swept check adds its sampling gap, so trim until it passes
  for (int trim = 0; trim < 30 && best_a + 1 <= best_b; ++trim)
  {
    if (!(stretchLen(best_a, best_b) >= min_length_rad))
    {
      out.reason = "longest passing stretch is " + std::to_string(stretchLen(best_a, best_b)) + " rad, less than the minimum " +
                   std::to_string(min_length_rad) + " rad (full chord: " + full_reason + ")";
      return out;
    }
    const ChordCheck c = checkChord(side, joint_names, chord[best_a], chord[best_b], other_joint_names, other_parked_q, max_step_rad);
    if (c.ok)
    {
      out.found = true;
      out.full = false;
      out.first_sample = best_a;
      out.last_sample = best_b;
      out.s0 = static_cast<double>(best_a) / static_cast<double>(out.total_steps);
      out.s1 = static_cast<double>(best_b) / static_cast<double>(out.total_steps);
      out.start = chord[best_a];
      out.goal = chord[best_b];
      out.length_rad = maxAbsDiff(out.start, out.goal);
      out.check = c;
      return out;
    }
    ++best_a;
    --best_b;
  }
  out.reason = "the passing stretch did not survive the re-check as a chord (full chord: " + full_reason + ")";
  return out;
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
