// schedule_mode "sequence": planning (4 jobs, delay search, segment-window validation) and the
// per-segment execution with the skew gate. See sequence_plan.hpp and coordinator.yaml.
#include <algorithm>
#include <cmath>
#include <future>
#include <iomanip>
#include <sstream>

#include "dual_cut_coordinator/coordinator.hpp"

namespace dual_cut
{
using Clock = std::chrono::steady_clock;
using namespace std::chrono_literals;

namespace
{
std::string esc(const std::string & in)
{
  std::string out;
  for (char c : in)
  {
    if (c == '"' || c == '\\') { out += '\\'; out += c; }
    else if (c == '\n') out += "\\n";
    else if (static_cast<unsigned char>(c) < 0x20) out += ' ';
    else out += c;
  }
  return out;
}

std::string segsJson(const ArmPlan & p)
{
  std::ostringstream o;
  o << std::setprecision(9) << "{\"pattern\":\"" << p.cut_pattern << "\",\"plan_seq\":" << p.plan_seq << ",\"duration_s\":" << p.duration()
    << ",\"cut_targets\":" << p.cut_targets.size() << ",\"segments\":[";
  for (size_t i = 0; i < p.segments.size(); ++i)
  {
    const auto & g = p.segments[i];
    o << (i ? "," : "") << "{\"kind\":\"" << toString(g.kind) << "\",\"name\":\"" << esc(g.stage_name) << "\",\"s\":"
      << g.trajectory.duration() << ",\"contact\":" << (g.tool_carton_contact_allowed ? "true" : "false") << "}";
  }
  o << "]}";
  return o.str();
}

std::string windowsJson(const ArmPhase & arm, double margin)
{
  std::ostringstream o;
  o << std::setprecision(6) << "[";
  const auto w = segmentWindows(arm, margin);
  for (size_t i = 0; i < w.size(); ++i)
    o << (i ? "," : "") << "{\"i\":" << w[i].index << ",\"name\":\"" << esc(w[i].name) << "\",\"start_s\":" << w[i].start_s
      << ",\"end_s\":" << w[i].end_s << ",\"window_lo_s\":" << w[i].lo_s << ",\"window_hi_s\":"
      << (std::isfinite(w[i].hi_s) ? w[i].hi_s : -1.0) << "}";
  o << "]";
  return o.str();
}

struct Cand
{
  std::string arm;
  double delay_s = 0.0;
  double makespan_s = 0.0;
  bool safe = false;
  size_t overlapping_pairs = 0;
  double effective_min_m = 0.0;
  std::string message;
};

std::string candJson(const Cand & c)
{
  std::ostringstream o;
  o << std::setprecision(6) << "{\"arm\":\"" << c.arm << "\",\"delay_s\":" << c.delay_s << ",\"makespan_s\":" << c.makespan_s
    << ",\"safe\":" << (c.safe ? "true" : "false") << ",\"overlapping_pairs\":" << c.overlapping_pairs << ",\"effective_min_m\":"
    << (std::isfinite(c.effective_min_m) ? c.effective_min_m : -1.0) << ",\"message\":\"" << esc(c.message) << "\"}";
  return o.str();
}
}  // namespace

// ---------------------------------------------------------------------------------------------
// planning
// ---------------------------------------------------------------------------------------------

CoordinatorNode::SeqOutcome CoordinatorNode::planSequence(const SeqInputs & in, std::string & summary)
{
  SeqOutcome out;
  const double timeout = pd("executor_call_timeout_s");
  const std::string return_home_on = "return_home_after_cut=true";

  // --- 1. plan the four jobs (left: face then top, or top then face when in.left_top_first; right: top remainder then face) ---
  auto planJob = [&](ArmIo & io, const char * label, const std::vector<std::string> & params,
                     const sensor_msgs::msg::JointState & from, ArmPlan & plan) -> Status
  {
    if (stop_latched_) return Status::Error("stop latched during planning");
    Status s = setExecutorParams(io, params);
    if (!s) return s;
    auto req = std::make_shared<dual_cut_msgs::srv::PlanFromState::Request>();
    req->start_state = from;
    std::string err;
    auto r1 = callService<dual_cut_msgs::srv::PlanFromState>(io.plan_from_state, req, timeout, err);
    if (!r1) return Status::Error(io.side + " job " + label + ": " + err);
    if (!(*r1)->success) return Status::Error(io.side + " job " + label + ": plan_h_cut_from_state failed: " + (*r1)->message);
    auto r2 = callService<dual_cut_msgs::srv::GetPlannedSolution>(
      io.get_plan, std::make_shared<dual_cut_msgs::srv::GetPlannedSolution::Request>(), 30.0, err);
    if (!r2) return Status::Error(io.side + " job " + label + ": " + err);
    if (!(*r2)->success) return Status::Error(io.side + " job " + label + ": get_planned_solution failed: " + (*r2)->message);
    if ((*r2)->plan_seq != (*r1)->plan_seq)
      return Status::Error(io.side + " job " + label + ": exported plan_seq " + std::to_string((*r2)->plan_seq) + " != planned " +
                           std::to_string((*r1)->plan_seq) + " (the executor plan changed underneath us)");
    s = fromResponse(io.side, **r2, plan);
    if (!s) return Status::Error(io.side + " job " + label + ": " + s.message);
    return Status::Ok();
  };
  auto after = [](const ArmPlan & p)
  {
    sensor_msgs::msg::JointState js;
    js.name = p.joint_names;
    js.position = p.end_q;
    return js;
  };

  ArmPlan lj1, lj2, rj1, rj2;   // lj1 / lj2: the left arm's first / second job in TIME (top or face, see in.left_top_first)
  double t_l1 = 0, t_l2 = 0, t_r1 = 0, t_r2 = 0;
  const std::string left_label1 = "1 (" + in.left_job1_label + ")", left_label2 = "2 (" + in.left_job2_label + ")";
  const auto t0 = Clock::now();
  auto lap = [&]() { return std::chrono::duration<double>(Clock::now() - t0).count(); };
  auto fl = std::async(std::launch::async, [&]() -> Status
  {
    Status s = planJob(left_, left_label1.c_str(), in.left_job1, in.left_start, lj1);
    t_l1 = lap();
    // top first: the right arm's top remainder can be planned as soon as the left top is done (always called: the right may be waiting)
    if (in.left_top_first && in.left_after_top) in.left_after_top(s, lj1);
    if (s) s = planJob(left_, left_label2.c_str(), in.left_job2, after(lj1), lj2);
    t_l2 = lap();
    if (!in.left_top_first && in.left_after_top) in.left_after_top(s, lj2);   // always: the right arm may be waiting for it
    return s;
  });
  bool right_skipped = false;
  auto fr = std::async(std::launch::async, [&]() -> Status
  {
    std::vector<std::string> p1;
    bool skip = false;
    Status s = in.right_job1(p1, skip);
    right_skipped = skip;
    if (s && !skip) s = planJob(right_, "1 (right top)", p1, in.right_start, rj1);
    t_r1 = lap();
    // right top skipped: the right timeline is the face only, planned from the current state
    if (s) s = planJob(right_, "2 (right face)", in.right_job2, skip ? in.right_start : after(rj1), rj2);
    t_r2 = lap();
    return s;
  });
  const Status ls = fl.get();
  const Status rs = fr.get();
  const double plan_wall_s = lap();
  // Executors back to the normal state whatever happened (best effort): source snapshot on the right arm,
  // return_home_after_cut=true on both (the phases mode does not set it).
  // (Only after a failed planning: on success both second jobs already set these, and changing a planning
  // parameter after the plan clears the executor's plan and its detected_carton collision object.)
  if (!ls || !rs)
  {
    const Status a = setExecutorParams(right_, {in.src_param, return_home_on});
    const Status b = setExecutorParams(left_, {return_home_on});
    if (!a) RCLCPP_WARN(get_logger(), "sequence: could not restore the right executor: %s", a.message.c_str());
    if (!b) RCLCPP_WARN(get_logger(), "sequence: could not restore the left executor: %s", b.message.c_str());
  }

  std::ostringstream rep;
  rep << std::setprecision(9);
  rep << "{\"schedule_mode\":\"sequence\",\"left_order\":\"" << (in.left_top_first ? "top_first" : "face_first") << "\",\"executor_calls\":{\"wall_s\":" << plan_wall_s << ",\"left_job1_done_s\":" << t_l1
      << ",\"left_job2_done_s\":" << t_l2 << ",\"right_job1_done_s\":" << t_r1 << ",\"right_job2_done_s\":" << t_r2 << "}";
  rep << ",\"plan_retry\":" << (retry_json_.empty() ? std::string("null") : retry_json_);
  rep << ",\"plan_fallback\":" << (fallback_json_.empty() ? std::string("null") : fallback_json_);
  rep << ",\"detection\":" << (detection_json_.empty() ? std::string("null") : detection_json_);
  rep << ",\"right_top_skipped\":" << (right_skipped ? "true" : "false");
  rep << ",\"top_split\":{\"mode\":\"" << toString(in.tsp.mode) << "\",\"source_file\":\"" << esc(in.source_path)
      << "\",\"source_sha256\":\"" << in.source_sha << "\",\"seam_length_m\":" << in.seam_length_m;
  const std::string split = in.split_json ? in.split_json() : std::string();
  if (!split.empty()) rep << "," << split;
  rep << "}";
  bool seq_open = false;   // the "sequence" member of the report is an object that is still open
  auto finishReport = [&](bool ok, const std::string & msg)
  {
    if (seq_open) rep << "}";
    seq_open = false;
    rep << ",\"ok\":" << (ok ? "true" : "false") << ",\"message\":\"" << esc(msg) << "\"}";
    last_plan_report_ = rep.str();
    std_msgs::msg::String m;
    m.data = last_plan_report_;
    report_pub_->publish(m);
  };
  auto fail = [&](const Status & st)
  {
    finishReport(false, st.message);
    out.status = st;
    return out;
  };
  if (!ls) return fail(ls);
  if (!rs) return fail(rs);
  if (stop_latched_) return fail(Status::Error("stop latched during planning"));

  const ArmPlan & ltop = in.left_top_first ? lj1 : lj2;
  const ArmPlan & lface = in.left_top_first ? lj2 : lj1;

  // --- 2. gates on every exported plan (as in phases mode) ---
  double age = 0.0, age_max = 0.0;
  for (auto * item : {&lj1, &lj2, &rj1, &rj2})
  {
    if (right_skipped && item == &rj1) continue;
    Status s = gatePlan(item->side == "left" ? left_ : right_, *item, age);
    if (!s) return fail(s);
    age_max = std::max(age_max, age);
  }
  {
    const Status st = checkPermissionMatchesPlan(detection_, lj1.snapshot.snapshot_id, lj1.snapshot.acquisition_stamp_ns);
    if (!st) return fail(st);
  }
  rep << ",\"plans\":{\"left_job1_is\":\"" << (in.left_top_first ? "top" : "face") << "\",\"left\":{\"job1\":" << segsJson(lj1) << ",\"job2\":" << segsJson(lj2) << "},\"right\":{\"job1\":" << (right_skipped ? std::string("null") : segsJson(rj1))
      << ",\"job2\":" << segsJson(rj2) << "}}";
  rep << ",\"snapshot\":{\"id\":\"" << esc(lj1.snapshot.snapshot_id) << "\",\"capture\":\"" << esc(lj1.snapshot.capture_id)
      << "\",\"acquisition_stamp_ns\":" << lj1.snapshot.acquisition_stamp_ns << ",\"age_at_plan_end_s\":" << age_max << "}";

  // --- 3. top-seam coverage (fail closed): left job 2 and right job 1 ---
  std::string split_summary;
  {
    const Eigen::Isometry3d snap_to_world = ltop.snapshot_frame_mode == "planning_frame" ? urdfBaseToWorld(left_) : Eigen::Isometry3d::Identity();
    const Eigen::Vector3d w0 = snap_to_world * in.seam_a, w1 = snap_to_world * in.seam_b;
    auto topInterval = [&](const ArmPlan & p, ArmIo & io, const char * what, SeamInterval & iv) -> Status
    {
      size_t cuts = 0;
      for (const auto & seg : p.segments) cuts += seg.kind == SegmentKind::Cut;
      if (cuts != 1)
        return Status::Error(std::string("top split coverage: ") + what + " has " + std::to_string(cuts) + " CUT segments, expected exactly one (middle_only)");
      std::vector<CutLine> lines;
      Status st = cutLinesFromTargets(p.cut_targets, cuts, p.planning_frame, urdfBaseToWorld(io), lines);
      if (!st) return Status::Error(std::string("top split coverage: ") + what + ": " + st.message);
      return projectCutLineOnSeam(lines[0], w0, w1, iv);
    };
    SeamInterval li, ri;
    Status st = topInterval(ltop, left_, in.left_top_first ? "left top (job 1)" : "left top (job 2)", li);
    if (st && !right_skipped) st = topInterval(rj1, right_, "right top (job 1)", ri);
    if (!st) return fail(st);
    TopCoverage cov;
    if (right_skipped)
    {
      // the left top alone must cover the whole seam
      st = checkLeftOnlyCoverage(li, (w1 - w0).norm(), in.tsp.end_tolerance_m, &cov);
      rep << ",\"top_coverage\":{\"left_only\":true,\"seam_length_mm\":" << cov.seam_length_m * 1e3 << ",\"left_mm\":[" << cov.left.s0 * 1e3 << ","
          << cov.left.s1 * 1e3 << "],\"left_reach_fraction\":" << ltop.middle_cut_fraction << "},\"top_coverage_ok\":" << (st.ok ? "true" : "false");
      if (!st) return fail(st);
      std::ostringstream o;
      o << std::fixed << std::setprecision(1) << "top split " << toString(in.tsp.mode) << ": right top skipped (left reach "
        << 100.0 * ltop.middle_cut_fraction << " %), left " << cov.left.s0 * 1e3 << ".." << cov.left.s1 * 1e3 << " mm of the " << cov.seam_length_m * 1e3
        << " mm seam; ";
      split_summary = o.str();
    }
    else
    {
      st = checkTopCoverage(li, ri, (w1 - w0).norm(), in.tsp.end_tolerance_m, in.tsp.min_overlap_m, &cov, effectiveRightEndExtension(in.tsp));
      rep << ",\"top_coverage\":" << topCoverageJson(cov) << ",\"top_coverage_ok\":" << (st.ok ? "true" : "false");
      if (!st) return fail(st);
      split_summary = std::string("top split ") + toString(in.tsp.mode) + ": " + describeTopCoverage(cov) + "; ";
    }
  }

  // --- 4. scene and validator ---
  planning_scene::PlanningScenePtr scene;
  Status s = refreshScene(scene);
  if (!s) return fail(s);
  double carton_dc = 0.0, carton_dr = 0.0;
  s = addCartonFromArmScenes(scene, carton_dc, carton_dr);
  rep << ",\"carton\":{\"center_diff_m\":" << carton_dc << ",\"rotation_diff_rad\":" << carton_dr << "}";
  if (!s) return fail(s);
  std::unique_ptr<PhaseValidator> validator;
  s = makeValidator(scene, validator);
  if (!s) return fail(Status::Error("validator: " + s.message));

  // --- 5. arm timelines and cut lines ---
  const PlanTolerances tol;
  auto build = [&](const std::string & delayed_arm, double d, ArmPhase & L, ArmPhase & R) -> Status
  {
    double j1 = 0;
    Status st = joinJobs(lj1, lj2, tol, delayed_arm == "left" ? d : 0.0, L, &j1);
    if (!st) return Status::Error("left: " + st.message);
    st = right_skipped ? singleJob(rj2, tol, delayed_arm == "right" ? d : 0.0, R)
                       : joinJobs(rj1, rj2, tol, delayed_arm == "right" ? d : 0.0, R, &j1);
    if (!st) return Status::Error("right: " + st.message);
    return Status::Ok();
  };
  ArmPhase L0, R0;
  s = build("", 0.0, L0, R0);
  if (!s) return fail(s);
  CutSag sag_left, sag_right;
  {
    const Eigen::Isometry3d snap_to_world = ltop.snapshot_frame_mode == "planning_frame" ? urdfBaseToWorld(left_) : Eigen::Isometry3d::Identity();
    s = topSagFor(left_, in.source_text, snap_to_world, sag_left);
    if (s) s = topSagFor(right_, in.source_text, snap_to_world, sag_right);
    if (!s) return fail(s);
    rep << ",\"top_sag_m\":{\"left\":" << sag_left.sag_m << ",\"right\":" << sag_right.sag_m << "}";
  }
  std::vector<CutLine> ll, rl;
  for (auto [arm, lines] : {std::pair<const ArmPhase *, std::vector<CutLine> *>{&L0, &ll}, std::pair<const ArmPhase *, std::vector<CutLine> *>{&R0, &rl}})
  {
    size_t cuts = 0;
    for (const auto & seg : arm->segments) cuts += seg.kind == SegmentKind::Cut;
    s = cutLinesFromTargets(arm->cut_targets, cuts, arm->planning_frame, urdfBaseToWorld(this->arm(arm->side)), *lines);
    if (!s) return fail(Status::Error("sequence " + arm->side + ": " + s.message));
    applyTopSag(*arm, *lines, arm->side == "left" ? sag_left : sag_right);
  }
  const double dur_l = L0.duration(), dur_r = R0.duration();
  const double est_phases = std::max(ltop.duration(), rj2.duration()) + std::max(lface.duration(), right_skipped ? 0.0 : rj1.duration());
  seq_open = true;
  rep << ",\"sequence\":{\"skew_margin_s\":" << in.margin_s << ",\"delay_step_s\":" << in.step_s << ",\"max_delay_s\":" << in.max_delay_s
      << ",\"duration_left_s\":" << dur_l << ",\"duration_right_s\":" << dur_r << ",\"phases_estimate_lower_bound_s\":" << est_phases;

  // --- 6. world / tip once (a hold changes neither), cross-arm by schedule search ---
  SegmentPairCache cache;
  WindowCrossResult wc0;
  const auto t_val0 = Clock::now();
  PhaseValidation v0 = validator->validateSequence(L0, R0, ll, rl, in.margin_s, &cache, &wc0);
  const bool world_tip_ok = v0.world_left.safe() && v0.world_right.safe() && v0.tip_left.safe() && v0.tip_right.safe();
  std::vector<Cand> log;
  bool have_best = false;
  Cand best;
  auto record = [&](const std::string & arm, double d, const WindowCrossResult & wc, double makespan)
  {
    Cand c;
    c.arm = arm;
    c.delay_s = d;
    c.makespan_s = makespan;
    c.safe = wc.safe();
    c.overlapping_pairs = wc.overlapping_pairs;
    c.effective_min_m = wc.worst.effective_min_distance;
    c.message = wc.safe() ? "ok" : ("left '" + wc.worst_left + "' vs right '" + wc.worst_right + "': " + wc.worst.message);
    log.push_back(c);
    return c;
  };
  std::string no_schedule_reason;
  if (!world_tip_ok)
  {
    no_schedule_reason = "world / tip-on-line check of the sequence timelines failed (a delay cannot fix it): " + v0.message;
  }
  else
  {
    Cand c0 = record("", 0.0, wc0, std::max(dur_l, dur_r));
    if (c0.safe)
    {
      best = c0;
      have_best = true;
    }
    else
    {
      for (const char * arm : {"left", "right"})
      {
        for (int k = 1; k <= static_cast<int>(std::floor(in.max_delay_s / in.step_s + 1e-9)); ++k)
        {
          const double d = k * in.step_s;
          const double ms = std::max(dur_l + (std::string(arm) == "left" ? d : 0.0), dur_r + (std::string(arm) == "right" ? d : 0.0));
          if (have_best && ms >= best.makespan_s) break;   // cannot beat the best so far
          if (stop_latched_) return fail(Status::Error("stop latched during planning"));
          ArmPhase L, R;
          s = build(arm, d, L, R);
          if (!s) return fail(s);
          const WindowCrossResult wc = validator->crossWindows(L, R, in.margin_s, &cache);
          Cand c = record(arm, d, wc, ms);
          if (c.safe)
          {
            best = c;
            have_best = true;
            break;
          }
        }
      }
      if (!have_best)
        no_schedule_reason = "no delay of the left or right second job up to " + std::to_string(in.max_delay_s) +
                             " s gives a clear segment-window cross-arm check; delay 0: " + log.front().message;
    }
  }

  // --- 7. the chosen schedule, validated once more as it will run ---
  ArmPhase L, R;
  PhaseValidation vf;
  WindowCrossResult wcf;
  if (have_best)
  {
    s = build(best.arm, best.delay_s, L, R);
    if (!s) return fail(s);
    vf = validator->validateSequence(L, R, ll, rl, in.margin_s, &cache, &wcf);
    if (!vf.ok)
    {
      have_best = false;
      no_schedule_reason = "the chosen schedule failed its final validation: " + vf.message;
    }
  }
  const double val_s = std::chrono::duration<double>(Clock::now() - t_val0).count();
  rep << ",\"validation_s\":" << val_s << ",\"cache_pairs\":" << cache.pairs.size() << ",\"candidates\":[";
  for (size_t i = 0; i < log.size(); ++i) rep << (i ? "," : "") << candJson(log[i]);
  rep << "]";
  if (!have_best)
  {
    // no safe schedule: the caller (the plan fallback chain) tries the next order / the phases plan and says why
    std::ostringstream fj;
    fj << std::setprecision(9) << "{\"requested\":true,\"fallback_to_phases\":true,\"reason\":\"" << esc(no_schedule_reason)
       << "\",\"skew_margin_s\":" << in.margin_s << ",\"duration_left_s\":" << dur_l << ",\"duration_right_s\":" << dur_r
       << ",\"phases_estimate_lower_bound_s\":" << est_phases << ",\"candidates\":[";
    for (size_t i = 0; i < log.size(); ++i) fj << (i ? "," : "") << candJson(log[i]);
    fj << "]}";
    out.fallback = true;
    out.fallback_reason = no_schedule_reason;
    out.fallback_json = fj.str();
    rep << ",\"sequence_search\":" << out.fallback_json;
    return fail(Status::Error("sequence schedule not possible: " + no_schedule_reason));
  }

  const double ms = std::max(L.duration(), R.duration());
  rep << ",\"fallback_to_phases\":false,\"chosen\":{\"delayed_arm\":\"" << best.arm << "\",\"delay_s\":" << best.delay_s
      << ",\"makespan_s\":" << ms << ",\"min_cross_window_m\":" << vf.min_cross_distance_m << ",\"overlapping_pairs\":" << wcf.overlapping_pairs
      << ",\"worst_left\":\"" << esc(wcf.worst_left) << "\",\"worst_right\":\"" << esc(wcf.worst_right) << "\"}";
  rep << ",\"windows\":{\"left\":" << windowsJson(L, in.margin_s) << ",\"right\":" << windowsJson(R, in.margin_s) << "}";
  rep << ",\"window_pairs\":[";
  for (size_t i = 0; i < wcf.pairs.size(); ++i)
    rep << (i ? "," : "") << "{\"l\":" << wcf.pairs[i].left_index << ",\"r\":" << wcf.pairs[i].right_index << ",\"safe\":"
        << (wcf.pairs[i].safe ? "true" : "false") << ",\"eff_min_m\":" << wcf.pairs[i].effective_min_m << "}";
  rep << "]";
  rep << ",\"world_left_ok\":" << (vf.world_left.safe() ? "true" : "false") << ",\"world_right_ok\":" << (vf.world_right.safe() ? "true" : "false")
      << ",\"world_left_table_min_m\":" << (vf.world_left.table_measured ? vf.world_left.table_min_distance_m : -1.0)
      << ",\"world_left_table_min_link\":\"" << vf.world_left.table_min_link << "\""
      << ",\"world_right_table_min_m\":" << (vf.world_right.table_measured ? vf.world_right.table_min_distance_m : -1.0)
      << ",\"world_right_table_min_link\":\"" << vf.world_right.table_min_link << "\""
      << ",\"tip_left_ok\":" << (vf.tip_left.safe() ? "true" : "false") << ",\"tip_right_ok\":" << (vf.tip_right.safe() ? "true" : "false")
      << ",\"tip_left_max_dev_m\":" << vf.tip_left.max_deviation_m << ",\"tip_right_max_dev_m\":" << vf.tip_right.max_deviation_m;

  // --- 7b. the executors must hold exactly the plans of THIS attempt's last jobs (the coordinator executes its own copies;
  // a mismatch means a failed earlier attempt's plan is what an operator would see / re-export: fail closed) ---
  out.executor_seq_left = lj2.plan_seq;
  out.executor_seq_right = rj2.plan_seq;
  rep << ",\"executor_plans\":{\"left_plan_seq\":" << lj2.plan_seq << ",\"right_plan_seq\":" << rj2.plan_seq << "}";
  s = verifyExecutorPlans(lj2.plan_seq, rj2.plan_seq);
  if (!s) return fail(s);

  // --- 8. retain and display ---
  PhaseSet set;
  set.a = Phase("A", L, R);
  set.b = Phase("B", ArmPhase{}, ArmPhase{});
  set.snapshot = lj1.snapshot;
  set.a.recordValidation(PhaseValidator::toRecord(vf), vf.ok);
  std::ostringstream sum;
  sum << std::fixed << std::setprecision(3) << split_summary << "SEQUENCE schedule: makespan " << ms << " s (left " << L.duration() << ", right "
      << R.duration() << "), delay " << (best.arm.empty() ? std::string("none") : best.arm + " job 2 by " + std::to_string(best.delay_s) + " s")
      << ", min cross (segment windows, margin " << in.margin_s << " s) " << vf.min_cross_distance_m << " m over " << wcf.overlapping_pairs
      << " segment pairs; phases-mode lower-bound estimate " << est_phases << " s";
  {
    std::lock_guard<std::mutex> lk(data_mutex_);
    retained_.valid = true;
    retained_.id = next_plan_id_++;
    retained_.set = set;
    retained_.sequence = true;
    retained_.sequence_margin_s = in.margin_s;
    retained_.variant = current_variant_;
    retained_.detection = detection_;
    retained_.executor_plan_seq_left = lj2.plan_seq;
    retained_.executor_plan_seq_right = rj2.plan_seq;
    retained_.measured_left = in.ql;
    retained_.measured_right = in.qr;
    retained_.snapshot_age_at_plan_s = age_max;
    retained_.summary = sum.str();
    phase_validated_ = true;
  }
  publishDisplay(set, in.ql, in.qr);
  summary = sum.str();
  finishReport(true, summary);
  out.status = Status::Ok();
  return out;
}

// ---------------------------------------------------------------------------------------------
// execution: one FollowJointTrajectory goal per segment at its planned start, skew gate
// ---------------------------------------------------------------------------------------------

void CoordinatorNode::runSequence(const Retained & plan)
{
  const Phase & phase = plan.set.a;
  current_phase_ = "A";
  state_ = Status_::STATE_WAITING_BARRIER;
  publishStatus();
  Status s = waitBarriers(phase, 0.0);
  if (!s)
  {
    RCLCPP_ERROR(get_logger(), "sequence not started: %s", s.message.c_str());
    finishExecution(Status_::STATE_PLANNED);
    return;
  }
  if (stop_latched_) { executing_ = false; publishStatus(); return; }

  const double margin = plan.sequence_margin_s;
  const double lead = pd("sequence_dispatch_lead_s");
  struct Seg
  {
    size_t index;
    std::string name;
    bool hold;
    double ps = 0, pe = 0;                  // planned start / end, seconds after t0
    ArmPhase one;                           // single-segment view for buildGoal
    double as = -1, ae = -1;                // actual start / end after t0
  };
  struct Run
  {
    ArmIo * io;
    bool left;
    std::vector<Seg> segs;
    size_t next = 0;
    bool active = false;
  } runs[2] = {{&left_, true, {}, 0, false}, {&right_, false, {}, 0, false}};
  for (Run & r : runs)
  {
    const ArmPhase & arm = r.left ? phase.left() : phase.right();
    double t = 0.0;
    for (size_t i = 0; i < arm.segments.size(); ++i)
    {
      Seg g;
      g.index = i;
      g.name = arm.segments[i].stage_name;
      g.hold = arm.segments[i].kind == SegmentKind::Dwell && g.name == kSequenceHoldName;
      g.ps = t;
      t += arm.segments[i].trajectory.duration();
      g.pe = t;
      g.one.side = arm.side;
      g.one.joint_names = arm.joint_names;
      g.one.planning_frame = arm.planning_frame;
      g.one.segments = {arm.segments[i]};
      g.one.start_q = arm.segments[i].trajectory.points.front().positions;
      g.one.end_q = arm.segments[i].trajectory.points.back().positions;
      r.segs.push_back(std::move(g));
    }
  }
  const double total = std::max(runs[0].segs.empty() ? 0.0 : runs[0].segs.back().pe, runs[1].segs.empty() ? 0.0 : runs[1].segs.back().pe);

  state_ = Status_::STATE_EXECUTING;
  resetExecStats();
  const double snapshot_age_at_start = snapshotAgeS();
  const double start_delay = pd("start_delay_s");
  const auto t_exec0 = Clock::now();
  const Clock::time_point t0 = t_exec0 + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(start_delay));
  const rclcpp::Time stamp0 = now() + rclcpp::Duration::from_seconds(start_delay);
  left_.goal.reset();
  right_.goal.reset();
  {
    std::lock_guard<std::mutex> lk(expected_mutex_);
    expected_left_ = Expected{};
    expected_right_ = Expected{};
  }
  auto since0 = [&]() { return std::chrono::duration<double>(Clock::now() - t0).count(); };

  bool ok = true;
  std::string why;
  double max_skew_end = 0.0, max_skew_start = 0.0;
  const double hard_deadline = total + margin + start_delay + 20.0;
  while (true)
  {
    if (stop_latched_) { ok = false; why = "stop latched"; break; }
    const double tn = since0();
    bool all_done = true;
    for (Run & r : runs)
    {
      if (r.active)
      {
        int st;
        std::string txt;
        {
          std::lock_guard<std::mutex> lk(r.io->goal.m);
          st = r.io->goal.state;
          txt = r.io->goal.text;
        }
        Seg & g = r.segs[r.next];
        if (st == kSucceeded)
        {
          g.ae = tn;
          max_skew_end = std::max(max_skew_end, std::abs(tn - g.pe));
          {
            std::lock_guard<std::mutex> lk(expected_mutex_);
            (r.left ? expected_left_ : expected_right_).finished = true;
          }
          r.active = false;
          ++r.next;
        }
        else if (st >= kAborted || st == kRejected)
        {
          ok = false;
          why = r.io->side + " segment " + std::to_string(g.index) + " '" + g.name + "' goal " + txt;
          break;
        }
        else if (skewDecision(tn, g.ps, g.pe, margin, true, false) == SkewAction::Stop)
        {
          ok = false;
          why = "skew gate: " + r.io->side + " segment " + std::to_string(g.index) + " '" + g.name + "' not complete by planned end " +
                std::to_string(g.pe) + " s + margin " + std::to_string(margin) + " s (now " + std::to_string(tn) + " s)";
          break;
        }
      }
      // holds (inserted delays) are not dispatched: the arm stands still, the next segment waits for its planned start
      while (!r.active && r.next < r.segs.size() && r.segs[r.next].hold) ++r.next;
      if (!r.active && r.next < r.segs.size())
      {
        Seg & g = r.segs[r.next];
        // the first segments are sent right away with the common start stamp t0 (= now + start_delay_s)
        const double tn_eff = g.ps <= 1e-9 ? tn + start_delay : tn;
        const SkewAction act = skewDecision(tn_eff, g.ps, g.pe, margin, false, false);
        if (act == SkewAction::Stop)
        {
          ok = false;
          why = "skew gate: " + r.io->side + " segment " + std::to_string(g.index) + " '" + g.name + "' not started before planned end " +
                std::to_string(g.pe) + " s + margin (now " + std::to_string(tn) + " s)";
          break;
        }
        if (act == SkewAction::Dispatch)
        {
          Expected ex;
          FJT::Goal goal;
          s = buildGoal(g.one, ex, goal);
          if (!s) { ok = false; why = "cannot build goal: " + s.message; break; }
          // the first segment of both arms starts at the common stamp; later ones as soon as due (never in the past)
          const bool at_t0 = g.ps <= 1e-9;
          Clock::time_point start_steady;
          if (at_t0)
          {
            goal.trajectory.header.stamp = stamp0;
            start_steady = t0;
          }
          else
          {
            goal.trajectory.header.stamp = now() + rclcpp::Duration::from_seconds(lead);
            start_steady = Clock::now() + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(lead));
          }
          ex.t_start = start_steady;
          ex.started = true;
          {
            std::lock_guard<std::mutex> lk(expected_mutex_);
            (r.left ? expected_left_ : expected_right_) = ex;
          }
          sendGoal(*r.io, goal);
          g.as = std::chrono::duration<double>(start_steady - t0).count();
          max_skew_start = std::max(max_skew_start, g.as - g.ps);
          r.active = true;
        }
      }
      if (r.active || r.next < r.segs.size()) all_done = false;
    }
    if (!ok) break;
    if (all_done) break;
    if (tn > hard_deadline) { ok = false; why = "sequence execution timeout"; break; }
    std::this_thread::sleep_for(10ms);
  }
  if (!ok && !stop_latched_) doStop(why);

  const double wall = std::chrono::duration<double>(Clock::now() - t_exec0).count();
  {
    double min_d, max_t;
    size_t ticks;
    {
      std::lock_guard<std::mutex> lk(mon_mutex_);
      min_d = exec_min_distance_m_;
      max_t = exec_max_tracking_rad_;
      ticks = exec_monitor_ticks_;
    }
    std::ostringstream o;
    o << std::setprecision(6) << "{\"plan_variant\":\"" << plan.variant << "\",\"phase\":\"A\",\"schedule\":\"sequence\",\"completed\":" << (ok ? "true" : "false") << ",\"mode\":\"sequence\""
      << ",\"planned_duration_left_s\":" << phase.left().duration() << ",\"planned_duration_right_s\":" << phase.right().duration()
      << ",\"wall_s\":" << wall << ",\"skew_margin_s\":" << margin << ",\"max_start_skew_s\":" << max_skew_start
      << ",\"max_end_skew_s\":" << max_skew_end << ",\"snapshot_age_at_phase_start_s\":" << snapshot_age_at_start
      << ",\"snapshot_age_limit_s\":" << pd("external_snapshot_max_age_s") << ",\"monitor_ticks\":" << ticks
      << ",\"min_cross_distance_during_execution_m\":" << (std::isfinite(min_d) ? min_d : -1.0) << ",\"max_tracking_error_rad\":" << max_t
      << ",\"stop_latched\":" << (stop_latched_.load() ? "true" : "false") << ",\"stop_reason\":\"" << esc(stopReason()) << "\",\"segments\":[";
    bool firstj = true;
    for (Run & r : runs)
      for (const Seg & g : r.segs)
      {
        o << (firstj ? "" : ",") << "{\"arm\":\"" << (r.left ? "left" : "right") << "\",\"i\":" << g.index << ",\"name\":\"" << esc(g.name)
          << "\",\"hold\":" << (g.hold ? "true" : "false") << ",\"planned_start_s\":" << g.ps << ",\"planned_end_s\":" << g.pe
          << ",\"actual_start_s\":" << g.as << ",\"actual_end_s\":" << g.ae << "}";
        firstj = false;
      }
    o << "]}";
    publishExecReport(o.str());
  }
  if (stop_latched_)
  {
    executing_ = false;
    publishStatus();
    return;
  }
  RCLCPP_INFO(get_logger(), "sequence complete (wall %.1f s, max end skew %.2f s)", wall, max_skew_end);
  current_phase_.clear();
  finishExecution(Status_::STATE_PLANNED);
}

}  // namespace dual_cut
