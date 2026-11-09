// ~/execute, start barriers, direct FollowJointTrajectory goals, ~/stop and ~/rearm.
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>

#include "dual_cut_coordinator/coordinator.hpp"

namespace dual_cut
{
using namespace std::chrono_literals;
using Clock = std::chrono::steady_clock;

double CoordinatorNode::snapshotAgeS() const
{
  uint64_t stamp = 0;
  {
    std::lock_guard<std::mutex> lk(const_cast<std::mutex &>(data_mutex_));
    if (!retained_.valid) return std::numeric_limits<double>::infinity();
    stamp = retained_.set.snapshot.acquisition_stamp_ns;
  }
  const auto now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
  return 1e-9 * (static_cast<double>(now_ns) - static_cast<double>(stamp));
}

// ---------------------------------------------------------------------------------------------
// ~/stop and ~/rearm
// ---------------------------------------------------------------------------------------------

void CoordinatorNode::doStop(const std::string & reason)
{
  std::lock_guard<std::mutex> lk(stop_mutex_);
  stop_latched_.store(true);
  const bool first = stop_reason_.empty();
  if (first) stop_reason_ = reason;
  state_ = Status_::STATE_STOPPED;
  // The order comes from planStopSteps() (unit tested). With a robot-side joint move in play: FIRST every MoveJoint cancel
  // (-> CMD Move Stop; the robot keeps moving if nothing tells it to stop, even after a driver death), THEN the JTC hold
  // (not for an arm whose JTC is not in Normal mode: it is inactive and its joint states may be frozen, so nothing here
  // reads /joint_states for an arm in a CRI run), THEN the FJT cancels. Without CRI it is today's hold-then-cancel.
  const bool cri = cri_used_.load();
  std::array<bool, 2> hold_ok{true, true};
  if (cri)
  {
    std::lock_guard<std::mutex> g(cri_mutex_);
    for (size_t i = 0; i < 2; ++i) hold_ok[i] = cri_[i].mode == CtrlGateMode::Normal;
  }
  for (const StopStep & step : planStopSteps(cri, hold_ok))
  {
    ArmIo * io = step.arm == 0 ? &left_ : &right_;
    if (cri) RCLCPP_WARN(get_logger(), "doStop step: %s %s (t=%.3f)", toString(step.action), io->side.c_str(), now().seconds());
    switch (step.action)
    {
      case StopAction::CancelMoveJoint:
      {
        MoveJointHandle::SharedPtr h;
        {
          std::lock_guard<std::mutex> g(cri_mutex_);
          h = cri_[static_cast<size_t>(step.arm)].handle;
        }
        try
        {
          if (h) io->cri_action->async_cancel_goal(h);
        }
        catch (const std::exception & e)
        {
          RCLCPP_DEBUG(get_logger(), "cancel MoveJoint %s: %s", io->side.c_str(), e.what());
        }
        try
        {
          io->cri_action->async_cancel_all_goals();   // also a goal whose acceptance has not come back yet
        }
        catch (const std::exception & e)
        {
          RCLCPP_DEBUG(get_logger(), "cancel-all MoveJoint %s: %s", io->side.c_str(), e.what());
        }
        break;
      }
      case StopAction::HoldJtc:
      {
        // a one-point trajectory at the measured position replaces the controller's trajectory (an EMPTY trajectory is
        // rejected by the Humble JTC and does NOT hold; see wp6 evidence).
        std::vector<double> q;
        if (!(cri ? armQEff(*io, q) : armQ(*io, q))) break;
        trajectory_msgs::msg::JointTrajectory hold;
        hold.joint_names = io->spec.joint_names;
        trajectory_msgs::msg::JointTrajectoryPoint p;
        p.positions = q;
        p.velocities.assign(q.size(), 0.0);
        p.time_from_start = rclcpp::Duration(0, 0);
        hold.points.push_back(p);
        io->hold_pub->publish(hold);
        break;
      }
      case StopAction::CancelFjt:
      {
        // cancel every goal (the controller also holds on cancel)
        GoalHandle::SharedPtr h;
        {
          std::lock_guard<std::mutex> g(io->goal.m);
          h = io->goal.handle;
        }
        try
        {
          if (h) io->action->async_cancel_goal(h);   // throws if the goal already finished and was forgotten
        }
        catch (const std::exception & e)
        {
          RCLCPP_DEBUG(get_logger(), "cancel %s: %s", io->side.c_str(), e.what());
        }
        try
        {
          io->action->async_cancel_all_goals();
        }
        catch (const std::exception & e)
        {
          RCLCPP_DEBUG(get_logger(), "cancel-all %s: %s", io->side.c_str(), e.what());
        }
        break;
      }
    }
  }
  if (first) RCLCPP_ERROR(get_logger(), "STOP latched: %s", reason.c_str());
}

void CoordinatorNode::requestStopAsync(const std::string & reason)
{
  stop_latched_.store(true);   // immediately: nothing new may start, expected-cancel logic sees it
  {
    std::lock_guard<std::mutex> lk(stop_req_mutex_);
    if (!stop_req_pending_)
    {
      stop_req_reason_ = reason;
      stop_req_pending_ = true;
    }
  }
  stop_req_cv_.notify_one();
}

void CoordinatorNode::stopWorkerLoop()
{
  std::unique_lock<std::mutex> lk(stop_req_mutex_);
  while (!stop_worker_quit_)
  {
    stop_req_cv_.wait(lk, [this]() { return stop_req_pending_ || stop_worker_quit_; });
    if (stop_worker_quit_) return;
    const std::string reason = stop_req_reason_;
    stop_req_pending_ = false;
    lk.unlock();
    doStop(reason);
    lk.lock();
  }
}

void CoordinatorNode::handleStop(const std::shared_ptr<Trigger::Request>, std::shared_ptr<Trigger::Response> res)
{
  doStop("operator ~/stop");   // no plan lock: stop must work while planning or executing
  res->success = true;
  res->message = "stop latched; hold sent, goals cancelled";
  publishStatus();
}

void CoordinatorNode::handleRearm(const std::shared_ptr<Trigger::Request>, std::shared_ptr<Trigger::Response> res)
{
  std::unique_lock<std::mutex> plan_lock(plan_mutex_, std::try_to_lock);
  if (!plan_lock.owns_lock())
  {
    res->success = false;
    res->message = "busy: planning in progress";
    return;
  }
  if (executing_)
  {
    // the execution thread exits promptly once stop is latched; wait briefly
    for (int i = 0; i < 40 && executing_; ++i) std::this_thread::sleep_for(50ms);
    if (executing_)
    {
      res->success = false;
      res->message = "refused: an execution is still running";
      return;
    }
  }
  if (exec_thread_.joinable()) exec_thread_.join();
  // at rest. With a robot-side joint move in play (enabled, or used since the last rearm) the live per-arm measurement and
  // BOTH drivers' hardware_state (fresh, not latched, no stop pending, idle) decide, not /joint_states.
  const bool cri_rearm = cri_used_.load() || pb("cri_transit_enabled");
  const double rest = restDurationS();
  if (rest < get_parameter("barrier_rest_duration_s").as_double() || jointStateAgeS() > 0.5)
  {
    res->success = false;
    res->message = "refused: the arms are not at rest (rest " + std::to_string(rest) + " s)";
    return;
  }
  if (cri_rearm)
  {
    const Status g = criBothArmsGate(0.5);
    if (!g)
    {
      res->success = false;
      res->message = "refused: " + g.message;
      return;
    }
    std::lock_guard<std::mutex> lk(cri_mutex_);
    for (size_t i = 0; i < 2; ++i)
      if (cri_[i].sending || (cri_[i].accepted && !cri_[i].have_result))
      {
        res->success = false;
        res->message = std::string("refused: a MoveJoint goal is still active on ") + (i == 0 ? "left" : "right");
        return;
      }
  }
  for (ArmIo * io : {&left_, &right_})
  {
    std::lock_guard<std::mutex> g(io->goal.m);
    if (io->goal.state == kSent || io->goal.state == kAccepted)
    {
      res->success = false;
      res->message = "refused: a goal is still active on " + io->side;
      return;
    }
  }
  if (cri_rearm)
  {
    // a stop in (or around) a robot-side move leaves that arm's JTC deactivated: strict reactivation, hold check
    const Status rs = criRecoverControllers();
    if (!rs)
    {
      res->success = false;
      res->message = "refused: " + rs.message;
      return;
    }
  }
  {
    std::lock_guard<std::mutex> lk(data_mutex_);
    retained_ = Retained{};
    phase_validated_ = false;
  }
  for (ArmIo * io : {&left_, &right_}) io->goal.reset();
  {
    std::lock_guard<std::mutex> lk(expected_mutex_);
    expected_left_ = Expected{};
    expected_right_ = Expected{};
  }
  {
    std::lock_guard<std::mutex> lk(stop_mutex_);
    stop_reason_.clear();
  }
  current_phase_.clear();
  if (cri_rearm)
  {
    std::lock_guard<std::mutex> lk(cri_mutex_);
    for (CriArmState & a : cri_)
    {
      a.run_active = false;
      a.sending = a.accepted = a.rejected = a.have_result = false;
      a.handle.reset();
      a.mode = CtrlGateMode::Normal;
      a.mode_since = Clock::now();
    }
  }
  cri_used_ = false;
  stop_latched_ = false;
  state_ = Status_::STATE_IDLE;
  res->success = true;
  res->message = "rearmed; the retained plan was cleared";
  RCLCPP_WARN(get_logger(), "rearmed (plan cleared)");
  publishStatus();
}

// ---------------------------------------------------------------------------------------------
// ~/execute
// ---------------------------------------------------------------------------------------------

void CoordinatorNode::handleExecute(const std::shared_ptr<Trigger::Request>, std::shared_ptr<Trigger::Response> res)
{
  auto refuse = [&](const std::string & why)
  {
    res->success = false;
    res->message = "refused: " + why;
    RCLCPP_WARN(get_logger(), "execute %s", res->message.c_str());
  };
  if (!pb("execution_enabled")) return refuse("execution_enabled is false");
  if (stop_latched_) return refuse("stop is latched (" + stopReason() + "); call ~/rearm");
  if (planning_) return refuse("planning in progress");
  std::vector<char> phases;
  if (pb("execute_phase_a_enabled")) phases.push_back('A');
  if (pb("execute_phase_b_enabled")) phases.push_back('B');
  if (phases.empty()) return refuse("no phase is enabled (execute_phase_a_enabled / execute_phase_b_enabled)");
  if (pb("cri_transit_enabled"))
  {
    // fail closed: both drivers' hardware_state and robot_joint_states must be there, fresh and clean
    const Status g = criBothArmsGate(0.5);
    if (!g) return refuse("cri_transit_enabled: " + g.message);
  }
  bool expected = false;
  if (!executing_.compare_exchange_strong(expected, true)) return refuse("an execution is already running");

  {
    std::lock_guard<std::mutex> lk(data_mutex_);
    if (!retained_.valid || !phase_validated_)
    {
      executing_ = false;
      return refuse("no validated plan is retained");
    }
    // detection permission of the plan's snapshot (cut_permission written by the detector); evaluated at execute time
    const ExecuteGateResult dg = detectionExecuteGate(retained_.detection, pb("allow_blocked_detection"), pb("honor_detection_override"));
    if (!dg.allowed)
    {
      executing_ = false;
      return refuse("detection blocked: " + dg.message);
    }
    if (dg.by == "allow_blocked_detection" || dg.by == "detector_override")
      RCLCPP_WARN(get_logger(), "execute: %s", dg.message.c_str());
    else if (dg.by == "unknown_permission")
      RCLCPP_WARN(get_logger(), "execute: the plan's snapshot has no cut_permission field (old snapshot): treated as allowed");
    if (retained_.sequence)
    {
      // sequence schedule: one execution of the whole timelines, gated by execute_phase_a_enabled only
      // (execute_phase_b_enabled is not used)
      if (!pb("execute_phase_a_enabled"))
      {
        executing_ = false;
        return refuse("the retained plan is a sequence schedule: it needs execute_phase_a_enabled (phase B enable is not used)");
      }
      phases = {'A'};
    }
    for (char p : phases)
      if (!(p == 'A' ? retained_.set.a : retained_.set.b).validated())
      {
        executing_ = false;
        return refuse(std::string("phase ") + p + " is not validated");
      }
  }
  const double age = snapshotAgeS();
  if (!(age <= pd("external_snapshot_max_age_s")))
  {
    executing_ = false;
    return refuse("snapshot is stale (" + std::to_string(age) + " s old, limit " + std::to_string(pd("external_snapshot_max_age_s")) + " s)");
  }
  if (exec_thread_.joinable()) exec_thread_.join();
  state_ = Status_::STATE_WAITING_BARRIER;
  exec_thread_ = std::thread([this, phases]() { executionThread(phases); });
  res->success = true;
  res->message = "dispatch accepted (barriers are checked next; progress on ~/status)";
  RCLCPP_INFO(get_logger(), "execute: %s", res->message.c_str());
}

Status CoordinatorNode::waitBarriers(const Phase & phase, double)
{
  BarrierLimits lim;
  lim.joint_state_max_age_s = pd("barrier_joint_state_age_s");
  lim.start_tolerance_rad = pd("barrier_start_tolerance_rad");
  lim.rest_duration_s = pd("barrier_rest_duration_s");
  lim.snapshot_max_age_s = pd("external_snapshot_max_age_s");
  const auto deadline = Clock::now() + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(pd("barrier_timeout_s")));
  Status last = Status::Error("barrier not evaluated");
  while (true)
  {
    BarrierInputs in;
    in.stop_latched = stop_latched_.load();
    std::vector<double> ql, qr;
    double a1 = 0, a2 = 0;
    const bool cri_on = pb("cri_transit_enabled");   // live robot_joint_states per arm (the driver's own telemetry), not /joint_states
    const bool have = cri_on ? (criQ(left_, ql, &a1) && criQ(right_, qr, &a2)) : (armQ(left_, ql, &a1) && armQ(right_, qr, &a2));
    in.joint_state_age_s = have ? std::max({a1, a2, jointStateAgeS()}) : std::numeric_limits<double>::infinity();
    if (have)
    {
      double err = 0.0;
      for (auto [io, q, arm] : {std::tuple<ArmIo *, std::vector<double> *, const ArmPhase *>{&left_, &ql, &phase.left()},
                                std::tuple<ArmIo *, std::vector<double> *, const ArmPhase *>{&right_, &qr, &phase.right()}})
      {
        for (size_t i = 0; i < io->spec.joint_names.size(); ++i)
        {
          std::string plain;
          stripJointPrefix(io->prefix, io->spec.joint_names[i], plain);
          auto it = std::find(arm->joint_names.begin(), arm->joint_names.end(), plain);
          if (it == arm->joint_names.end())
          {
            err = std::numeric_limits<double>::infinity();
            continue;
          }
          err = std::max(err, std::abs((*q)[i] - arm->start_q[static_cast<size_t>(it - arm->joint_names.begin())]));
        }
      }
      in.max_joint_error_to_start_rad = err;
    }
    in.rest_duration_s = restDurationS();
    if (cri_on)
    {
      // BOTH arms: hardware_state fresh, not latched, no stop pending, idle; live measurement fresh. Missing = refused.
      const Status g = criBothArmsGate(pd("barrier_joint_state_age_s"));
      in.arm_health_required = true;
      in.arm_health_ok = g.ok;
      in.arm_health_text = g.message;
    }
    {
      std::lock_guard<std::mutex> lk(poll_mutex_);
      const auto now = Clock::now();
      const bool fresh = poll_ctrl_valid_ && poll_hw_valid_ && std::chrono::duration<double>(now - poll_ctrl_time_).count() < 1.0 &&
                         std::chrono::duration<double>(now - poll_hw_time_).count() < 1.0;
      in.controllers_active = fresh && ctrl_active_;
      in.hardware_active = fresh && hw_active_;
    }
    in.snapshot_age_s = snapshotAgeS();
    {
      std::lock_guard<std::mutex> lk(mon_mutex_);
      in.monitor_age_s = std::chrono::duration<double>(Clock::now() - mon_last_tick_).count();
      in.monitor_healthy = mon_healthy_;
    }
    last = evaluateBarriers(in, lim);
    {
      std::lock_guard<std::mutex> lk(mon_mutex_);
      barrier_ok_ = last.ok;
      barrier_message_ = last.ok ? "ok" : last.message;
    }
    if (last.ok) return last;
    if (stop_latched_) return Status::Error("stop latched while waiting for the barriers");
    if (Clock::now() > deadline) return last;
    std::this_thread::sleep_for(50ms);
  }
}

Status CoordinatorNode::buildGoal(const ArmPhase & arm, Expected & expected, FJT::Goal & goal)
{
  std::vector<SegmentInput> in;
  Status s = arm.prefixedSegments(in);
  if (!s) return s;
  std::vector<JointTrajectory> segs;
  for (auto & i : in) segs.push_back(i.trajectory);
  JointTrajectory joined;
  s = concatenate(segs, ConcatOptions{}, joined);
  if (!s) return s;
  expected = Expected{};
  expected.valid = true;
  expected.traj = joined;
  ArmIo & io = this->arm(arm.side);
  // spec order of start / end
  for (const auto & n : io.spec.joint_names)
  {
    std::string plain;
    stripJointPrefix(io.prefix, n, plain);
    auto it = std::find(arm.joint_names.begin(), arm.joint_names.end(), plain);
    if (it == arm.joint_names.end()) return Status::Error("joint missing in the phase: " + plain);
    const size_t k = static_cast<size_t>(it - arm.joint_names.begin());
    expected.q_start.push_back(arm.start_q[k]);
    expected.q_end.push_back(arm.end_q[k]);
  }
  // expected.traj uses the arm.joint_names order; the tracking compares in spec order, so reorder it.
  std::vector<size_t> order;
  for (const auto & n : io.spec.joint_names)
    order.push_back(static_cast<size_t>(std::find(joined.joint_names.begin(), joined.joint_names.end(), n) - joined.joint_names.begin()));
  JointTrajectory reordered = joined;
  reordered.joint_names = io.spec.joint_names;
  for (auto & p : reordered.points)
  {
    for (auto * vec : {&p.positions, &p.velocities, &p.accelerations})
    {
      if (vec->empty()) continue;
      std::vector<double> w(vec->size());
      for (size_t i = 0; i < order.size(); ++i) w[i] = (*vec)[order[i]];
      *vec = w;
    }
  }
  // the controller goal keeps the planned order (joint names are explicit)
  goal.trajectory = toMsg(joined);
  expected.traj = reordered;
  return Status::Ok();
}

void CoordinatorNode::sendGoal(ArmIo & io, const FJT::Goal & goal)
{
  io.goal.reset();
  {
    std::lock_guard<std::mutex> lk(io.goal.m);
    io.goal.state = kSent;
  }
  rclcpp_action::Client<FJT>::SendGoalOptions opts;
  opts.goal_response_callback = [this, &io](GoalHandle::SharedPtr h)
  {
    bool stop_now = false;
    std::string why;
    {
      std::lock_guard<std::mutex> lk(io.goal.m);
      if (!h)
      {
        io.goal.state = kRejected;
        io.goal.text = "REJECTED by " + io.controller;
        io.goal.unexpected = !stop_latched_.load();
        stop_now = io.goal.unexpected;
        why = "goal rejected on " + io.side;
      }
      else
      {
        io.goal.handle = h;
        if (io.goal.state < kSucceeded) io.goal.state = kAccepted;   // the result may already be in
      }
    }
    if (stop_now) requestStopAsync(why);
  };
  opts.result_callback = [this, &io](const GoalHandle::WrappedResult & r)
  {
    bool stop_now = false;
    std::string why;
    {
      std::lock_guard<std::mutex> lk(io.goal.m);
      switch (r.code)
      {
        case rclcpp_action::ResultCode::SUCCEEDED: io.goal.state = kSucceeded; io.goal.text = "SUCCEEDED"; break;
        case rclcpp_action::ResultCode::ABORTED: io.goal.state = kAborted; io.goal.text = "ABORTED"; break;
        case rclcpp_action::ResultCode::CANCELED: io.goal.state = kCanceled; io.goal.text = "CANCELED"; break;
        default: io.goal.state = kUnknown; io.goal.text = "UNKNOWN"; break;
      }
      RCLCPP_INFO(get_logger(), "%s goal result: %s", io.side.c_str(), io.goal.text.c_str());
      if (r.result)
        io.goal.text += " (error_code " + std::to_string(r.result->error_code) + ": " + r.result->error_string + ")";
      if (io.goal.state != kSucceeded && !stop_latched_.load())
      {
        io.goal.unexpected = true;
        stop_now = true;
        why = io.side + " goal " + io.goal.text;
      }
    }
    if (stop_now) requestStopAsync(why);   // abort / cancel we did not ask for: stop the other arm too
  };
  io.action->async_send_goal(goal, opts);
}

bool CoordinatorNode::waitGoalsDone(const std::vector<ArmIo *> & arms, double timeout_s)
{
  const auto deadline = Clock::now() + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(timeout_s));
  while (Clock::now() < deadline)
  {
    if (stop_latched_) return false;
    bool all = true;
    for (ArmIo * io : arms)
    {
      std::lock_guard<std::mutex> lk(io->goal.m);
      if (io->goal.state < kSucceeded) all = false;
      if (io->goal.state >= kAborted || io->goal.state == kRejected) return false;
    }
    if (all) return true;
    std::this_thread::sleep_for(20ms);
  }
  doStop("execution timeout");
  return false;
}

void CoordinatorNode::finishExecution(int new_state)
{
  if (!stop_latched_) state_ = new_state;
  executing_ = false;
  publishStatus();
}

void CoordinatorNode::executionThread(std::vector<char> phases)
{
  Retained plan;
  {
    std::lock_guard<std::mutex> lk(data_mutex_);
    plan = retained_;
  }
  if (plan.sequence)
  {
    runSequence(plan);
    return;
  }
  // The certificate parameters are the plan's own; whether the driver caps may be assumed is the operator's switch NOW
  // (a planning-scope parameter, so it cannot change while this execution runs).
  plan.cri.params.assume_driver_cap_ok = pb("cri_assume_driver_cap_ok");
  for (char label : phases)
  {
    if (stop_latched_) break;
    const Phase & phase = label == 'A' ? plan.set.a : plan.set.b;
    current_phase_ = std::string(1, label);
    state_ = Status_::STATE_WAITING_BARRIER;
    publishStatus();
    Status s = waitBarriers(phase, 0.0);
    if (!s)
    {
      RCLCPP_ERROR(get_logger(), "phase %c not started: %s", label, s.message.c_str());
      finishExecution(Status_::STATE_PLANNED);
      return;
    }
    if (stop_latched_) break;

    // goals
    FJT::Goal gl, gr;
    Expected el, er;
    s = buildGoal(phase.left(), el, gl);
    if (s) s = buildGoal(phase.right(), er, gr);
    if (!s)
    {
      RCLCPP_ERROR(get_logger(), "phase %c: cannot build goals: %s", label, s.message.c_str());
      finishExecution(Status_::STATE_PLANNED);
      return;
    }
    {
      std::lock_guard<std::mutex> lk(expected_mutex_);
      expected_left_ = el;
      expected_right_ = er;
    }
    state_ = Status_::STATE_EXECUTING;
    resetExecStats();
    const double snapshot_age_at_start = snapshotAgeS();
    const auto t_phase0 = Clock::now();
    left_.goal.reset();
    right_.goal.reset();

    const bool serial = phase.validation().serial_fallback;
    const bool cri_on = pb("cri_transit_enabled");
    std::string cri_note = cri_on ? "parallel phase: every segment stays on the JTC (stage 1: serial phases only)" : "CRI off";
    std::vector<CriRunReport> cri_runs;
    auto launch = [&](ArmIo & io, FJT::Goal & goal, Expected & ex, bool left_side)
    {
      const double delay = pd("start_delay_s");
      goal.trajectory.header.stamp = now() + rclcpp::Duration::from_seconds(delay);
      {
        std::lock_guard<std::mutex> lk(expected_mutex_);
        Expected & dst = left_side ? expected_left_ : expected_right_;
        dst.t_start = Clock::now() + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(delay));
        dst.started = true;
      }
      sendGoal(io, goal);
      (void)ex;
    };
    auto markFinished = [&](bool left_side)
    {
      std::lock_guard<std::mutex> lk(expected_mutex_);
      (left_side ? expected_left_ : expected_right_).finished = true;
    };

    bool ok = true;
    if (!serial)
    {
      // one common start stamp for both goals
      const double delay = pd("start_delay_s");
      const rclcpp::Time stamp = now() + rclcpp::Duration::from_seconds(delay);
      gl.trajectory.header.stamp = stamp;
      gr.trajectory.header.stamp = stamp;
      {
        std::lock_guard<std::mutex> lk(expected_mutex_);
        const auto t0 = Clock::now() + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(delay));
        expected_left_.t_start = expected_right_.t_start = t0;
        expected_left_.started = expected_right_.started = true;
      }
      sendGoal(left_, gl);
      sendGoal(right_, gr);
      const double dur = std::max(phase.left().duration(), phase.right().duration());
      ok = waitGoalsDone({&left_, &right_}, dur + delay + 10.0);
      if (ok)
      {
        markFinished(true);
        markFinished(false);
      }
    }
    else
    {
      const bool left_first = phase.validation().serial_first != "right";
      ArmIo & first = left_first ? left_ : right_;
      ArmIo & second = left_first ? right_ : left_;
      FJT::Goal & g1 = left_first ? gl : gr;
      FJT::Goal & g2 = left_first ? gr : gl;
      Expected & e1 = left_first ? el : er;
      Expected & e2 = left_first ? er : el;
      RCLCPP_INFO(get_logger(), "phase %c runs serially: %s arm first", label, first.side.c_str());

      // Robot-side joint move legs of this serial phase (opt-in, certified at plan time with the plan's own parameters). The
      // run split is [JTC run | CRI run | JTC run]; without a selected leg (CRI off, parallel phase, sequence plan, driver cap
      // unknown ...) every arm has ONE JTC run and the code below is today's code.
      std::array<CriArmRuns, 2> runs;
      cri_note = "CRI off";
      for (const ArmIo * io : {&left_, &right_})
      {
        const CriArmPhasePlan * ap = cri_on && plan.cri.computed ? plan.cri.find(std::string(1, label), io->side) : nullptr;
        std::string why;
        const bool cap_ok = criCapOk(criHwView(armIndex(*io)), plan.cri.params, &why);
        runs[armIndex(*io)] = buildCriArmRuns(cri_on, cap_ok, ap, phase.arm(io->side));
        if (!cri_on) continue;
        if (!plan.cri.computed)
          cri_note = "the retained plan carries no CRI analysis (it was made with cri_transit_enabled=false)";
        else if (ap && !ap->selected_segments.empty() && !cap_ok)
        {
          cri_note = io->side + ": CRI leg stays on the JTC, " + why;
          RCLCPP_WARN(get_logger(), "phase %c: %s", label, cri_note.c_str());
        }
        else if (cri_note == "CRI off" || cri_note == "no leg selected for this phase")
          cri_note = ap && !ap->selected_segments.empty() ? "CRI legs dispatched" : "no leg selected for this phase";
      }
      auto hasCri = [](const CriArmRuns & r)
      {
        for (const CriRun & x : r.runs)
          if (x.type == CriRunType::Cri) return true;
        return false;
      };
      const bool use_runs = hasCri(runs[0]) || hasCri(runs[1]);

      if (!use_runs)
      {
        launch(first, g1, e1, left_first);
        ok = waitGoalsDone({&first}, (left_first ? phase.left() : phase.right()).duration() + pd("start_delay_s") + 10.0);
        if (ok)
        {
          markFinished(left_first);
          // the second arm starts only once everything is at rest again
          const auto deadline = Clock::now() + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(pd("barrier_timeout_s")));
          while (!stop_latched_ && Clock::now() < deadline)
          {
            if (restDurationS() >= pd("barrier_rest_duration_s")) break;
            std::this_thread::sleep_for(50ms);
          }
          if (stop_latched_) ok = false;
          else if (const Status g = cri_on ? criBothArmsGate(pd("barrier_joint_state_age_s")) : Status::Ok(); !g)
          {
            doStop("second arm of the serial phase not started: " + g.message);
            ok = false;
          }
          else
          {
            launch(second, g2, e2, !left_first);
            ok = waitGoalsDone({&second}, (left_first ? phase.right() : phase.left()).duration() + pd("start_delay_s") + 10.0);
            if (ok) markFinished(!left_first);
          }
        }
      }
      else
      {
        // One arm after the other, each as its list of runs. Between runs the arm must be at rest on LIVE state with healthy
        // drivers (both arms); the second arm starts only after ALL runs of the first are done and at rest.
        auto parkExpected = [&](bool left_side, const std::vector<double> & q_spec, bool finished)
        {
          std::lock_guard<std::mutex> lk(expected_mutex_);
          Expected & dst = left_side ? expected_left_ : expected_right_;
          dst = Expected{};
          dst.valid = true;
          dst.q_start = q_spec;
          dst.q_end = q_spec;
          dst.finished = finished;
        };
        auto settle = [&]() -> bool
        {
          if (!waitAtRest({&left_, &right_}, pd("barrier_timeout_s")))
          {
            if (!stop_latched_) doStop("arms did not come to rest between runs");
            return false;
          }
          const Status g = criBothArmsGate(pd("barrier_joint_state_age_s"));
          if (!g)
          {
            if (!stop_latched_) doStop("between runs: " + g.message);
            return false;
          }
          return true;
        };
        auto runArm = [&](ArmIo & io, bool left_side) -> bool
        {
          const CriArmRuns & arm_runs = runs[armIndex(io)];
          const ArmPhase & ap = arm_runs.phase;   // a partial CRI leg is [JTC head, leg, JTC tail] here
          for (size_t run_i = 0; run_i < arm_runs.runs.size(); ++run_i)
          {
            const CriRun & run = arm_runs.runs[run_i];
            if (stop_latched_) return false;
            CriRunReport rep;
            rep.arm = io.side;
            rep.type = toString(run.type);
            rep.first_segment = run.first_segment;
            rep.end_segment = run.end_segment;
            const auto t_run0 = Clock::now();
            bool run_ok = false;
            if (run.type == CriRunType::Jtc)
            {
              ArmPhase sub = ap;
              sub.segments.assign(ap.segments.begin() + static_cast<std::ptrdiff_t>(run.first_segment), ap.segments.begin() + static_cast<std::ptrdiff_t>(run.end_segment));
              sub.start_q = sub.segments.front().trajectory.points.front().positions;
              sub.end_q = sub.segments.back().trajectory.points.back().positions;
              FJT::Goal g;
              Expected ex;
              const Status bs = buildGoal(sub, ex, g);
              if (!bs)
              {
                if (!stop_latched_) doStop("cannot build the goal of a JTC run: " + bs.message);
                rep.note = bs.message;
              }
              else
              {
                {
                  std::lock_guard<std::mutex> lk(expected_mutex_);
                  (left_side ? expected_left_ : expected_right_) = ex;
                }
                rep.planned_s = sub.duration();
                launch(io, g, ex, left_side);
                run_ok = waitGoalsDone({&io}, sub.duration() + pd("start_delay_s") + 10.0);
                if (run_ok) markFinished(left_side);
              }
              rep.ok = run_ok;
            }
            else
            {
              const CriLegRecord * leg = arm_runs.leg[run_i];
              if (!leg)
              {
                if (!stop_latched_) doStop("CRI run without a selected leg record (internal)");
                rep.note = "no leg record";
              }
              else
              {
                rep.planned_s = ap.segments[run.first_segment].trajectory.duration();
                run_ok = runCriLeg(io, ap, *leg, plan.cri.params, rep);
                if (run_ok)
                {
                  // the arm now stands at the chord end (spec order): the tracking reference of the parked arm
                  std::vector<double> goal_spec;
                  for (const auto & n : io.spec.joint_names)
                  {
                    std::string plain;
                    stripJointPrefix(io.prefix, n, plain);
                    const auto it = std::find(leg->geometry.joint_names.begin(), leg->geometry.joint_names.end(), plain);
                    goal_spec.push_back(leg->cri_goal[static_cast<size_t>(it - leg->geometry.joint_names.begin())]);
                  }
                  parkExpected(left_side, goal_spec, false);
                }
              }
            }
            rep.wall_s = std::chrono::duration<double>(Clock::now() - t_run0).count();
            rep.ok = run_ok;
            cri_runs.push_back(rep);
            if (!run_ok) return false;
            if (!settle()) return false;
          }
          return true;
        };
        // the arm that has not started yet is parked at its phase start: its tracking reference
        ok = runArm(first, left_first);
        if (ok)
        {
          markFinished(left_first);
          ok = !stop_latched_;
        }
        if (ok) ok = runArm(second, !left_first);
        if (ok) markFinished(!left_first);
      }
    }
    const double phase_wall = std::chrono::duration<double>(Clock::now() - t_phase0).count();
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
      o << std::setprecision(9) << "{\"plan_variant\":\"" << plan.variant << "\",\"phase\":\"" << label << "\",\"completed\":" << (ok ? "true" : "false") << ",\"mode\":\""
        << (serial ? "serial" : "parallel") << "\",\"serial_first\":\"" << phase.validation().serial_first << "\",\"planned_duration_left_s\":"
        << phase.left().duration() << ",\"planned_duration_right_s\":" << phase.right().duration() << ",\"wall_s\":" << phase_wall
        << ",\"snapshot_age_at_phase_start_s\":" << snapshot_age_at_start << ",\"snapshot_age_limit_s\":"
        << pd("external_snapshot_max_age_s") << ",\"monitor_ticks\":" << ticks << ",\"min_cross_distance_during_execution_m\":"
        << (std::isfinite(min_d) ? min_d : -1.0) << ",\"max_tracking_error_rad\":" << max_t << ",\"stop_latched\":"
        << (stop_latched_.load() ? "true" : "false") << ",\"stop_reason\":\"" << stopReason() << "\"";
      if (cri_on) o << ",\"cri\":" << criRunReportJson(cri_runs, cri_note);
      o << "}";
      publishExecReport(o.str());
    }
    if (!ok)
    {
      if (!stop_latched_) doStop("phase " + std::string(1, label) + " did not complete");
      break;
    }
    RCLCPP_INFO(get_logger(), "phase %c complete%s", label, serial ? " (serial)" : "");
  }
  if (stop_latched_)
  {
    executing_ = false;
    publishStatus();
    return;
  }
  current_phase_.clear();
  finishExecution(Status_::STATE_PLANNED);
}

}  // namespace dual_cut
