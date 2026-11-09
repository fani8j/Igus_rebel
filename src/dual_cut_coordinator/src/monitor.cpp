// Joint-state cache, 20 Hz monitor, controller / hardware poll and the status topic.
#include <algorithm>
#include <cmath>

#include "dual_cut_coordinator/coordinator.hpp"

namespace dual_cut
{
using Clock = std::chrono::steady_clock;

namespace
{
double secs(Clock::time_point t) { return std::chrono::duration<double>(t.time_since_epoch()).count(); }
}

void CoordinatorNode::onJointState(const sensor_msgs::msg::JointState::SharedPtr msg)
{
  const auto now = Clock::now();
  std::lock_guard<std::mutex> lk(js_mutex_);
  const bool has_vel = msg->velocity.size() == msg->name.size();
  double max_v = 0.0;
  for (size_t i = 0; i < msg->name.size() && i < msg->position.size(); ++i)
  {
    JointSample & j = joints_[msg->name[i]];
    // Reported velocity (mock hardware reports none: it is always 0) AND the finite difference of the
    // positions; the larger magnitude counts, so a missing velocity can never look like "at rest".
    double v = has_vel ? msg->velocity[i] : 0.0;
    if (j.have_prev)
    {
      const double dt = std::chrono::duration<double>(now - j.t).count();
      if (dt > 2e-3 && dt < 0.5)
      {
        const double fd = (msg->position[i] - j.pos) / dt;
        if (std::abs(fd) > std::abs(v)) v = fd;
      }
    }
    j.pos = msg->position[i];
    j.vel = v;
    j.t = now;
    j.have_prev = true;
    max_v = std::max(max_v, std::isfinite(v) ? std::abs(v) : std::numeric_limits<double>::infinity());
  }
  js_last_rx_ = now;
  js_have_ = true;
  rest_tracker_.update(secs(now), max_v);
}

bool CoordinatorNode::armQ(const ArmIo & io, std::vector<double> & q, double * age_s) const
{
  std::lock_guard<std::mutex> lk(js_mutex_);
  if (io.spec.joint_names.empty()) return false;
  q.clear();
  double age = 0.0;
  const auto now = Clock::now();
  for (const auto & n : io.spec.joint_names)
  {
    auto it = joints_.find(n);
    if (it == joints_.end()) return false;
    q.push_back(it->second.pos);
    age = std::max(age, std::chrono::duration<double>(now - it->second.t).count());
  }
  if (age_s) *age_s = age;
  return true;
}

double CoordinatorNode::jointStateAgeS() const
{
  std::lock_guard<std::mutex> lk(js_mutex_);
  if (!js_have_) return std::numeric_limits<double>::infinity();
  return std::chrono::duration<double>(Clock::now() - js_last_rx_).count();
}

void CoordinatorNode::pollTick()
{
  if (list_controllers_->service_is_ready())
  {
    const auto sent = Clock::now();   // the answer describes the controller manager at (or after) this instant
    list_controllers_->async_send_request(
      std::make_shared<controller_manager_msgs::srv::ListControllers::Request>(),
      [this, sent](rclcpp::Client<controller_manager_msgs::srv::ListControllers>::SharedFuture f)
      {
        const auto r = f.get();
        bool l = false, rr = false;
        std::string sl, sr;
        for (const auto & c : r->controller)
        {
          if (c.name == left_.controller) { l = c.state == "active"; sl = c.state; }
          if (c.name == right_.controller) { rr = c.state == "active"; sr = c.state; }
        }
        const double latency = std::chrono::duration<double>(Clock::now() - sent).count();
        if (latency > 0.4) RCLCPP_WARN(get_logger(), "list_controllers answered after %.2f s", latency);
        std::lock_guard<std::mutex> lk(poll_mutex_);
        ctrl_active_ = l && rr;
        ctrl_state_ = {sl, sr};
        poll_ctrl_sent_ = sent;
        poll_ctrl_valid_ = true;
        poll_ctrl_time_ = Clock::now();
      });
  }
  if (list_hardware_->service_is_ready())
  {
    const auto sent_hw = Clock::now();
    list_hardware_->async_send_request(
      std::make_shared<controller_manager_msgs::srv::ListHardwareComponents::Request>(),
      [this, sent_hw](rclcpp::Client<controller_manager_msgs::srv::ListHardwareComponents>::SharedFuture f)
      {
        const auto r = f.get();
        const double latency = std::chrono::duration<double>(Clock::now() - sent_hw).count();
        if (latency > 0.4) RCLCPP_WARN(get_logger(), "list_hardware_components answered after %.2f s", latency);
        int active = 0;
        bool all = !r->component.empty();
        for (const auto & c : r->component)
        {
          if (c.state.id == lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE) ++active;
          else all = false;
        }
        std::lock_guard<std::mutex> lk(poll_mutex_);
        hw_active_ = all && active >= get_parameter("expected_hardware_components").as_int();
        poll_hw_valid_ = true;
        poll_hw_time_ = Clock::now();
      });
  }
}

void CoordinatorNode::monitorTick()
{
  if (!model_ready_)
  {
    // Model not loaded yet (the first ~/plan loads it): still publish the status.
    static thread_local int n0 = 0;
    if (++n0 % 2 == 0) publishStatus();
    return;
  }
  const auto now = Clock::now();
  MonitorInputs in;
  MonitorLimits lim;
  lim.joint_state_stale_s = get_parameter("joint_state_stale_s").as_double();
  lim.stop_distance_m = get_parameter("monitor_stop_distance_m").as_double();
  lim.tracking_stop_rad = get_parameter("tracking_stop_rad").as_double();
  lim.tracking_warn_rad = get_parameter("tracking_warn_rad").as_double();

  // joint states (CRI on: the driver's live robot_joint_states; /joint_states is frozen while a driver is latched)
  const bool cri_on = pb("cri_transit_enabled");
  std::vector<double> ql, qr;
  double age_l = 0, age_r = 0;
  const bool have = cri_on ? (criQ(left_, ql, &age_l) && criQ(right_, qr, &age_r)) : (armQ(left_, ql, &age_l) && armQ(right_, qr, &age_r));
  in.have_joint_state = have;
  in.joint_state_age_s = std::max(jointStateAgeS(), have ? std::max(age_l, age_r) : 0.0);

  // cross-arm distance (conservative: tolerance subtracted)
  double dist = std::numeric_limits<double>::quiet_NaN();
  std::string la, lb;
  if (have && monitor_checker_)
  {
    double raw;
    if (monitor_checker_->exactDistance(ql, qr, raw, la, lb))
    {
      dist = raw - monitor_checker_->config().exact_distance_tolerance_m;
      in.distance_valid = true;
    }
  }
  in.cross_distance_m = dist;

  // controller / hardware poll
  {
    std::lock_guard<std::mutex> lk(poll_mutex_);
    const double a_ctrl = std::chrono::duration<double>(now - poll_ctrl_time_).count(), a_hw = std::chrono::duration<double>(now - poll_hw_time_).count();
    if (poll_ctrl_valid_ && poll_hw_valid_ && std::max(a_ctrl, a_hw) > 0.8 && executing_.load())
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 500, "controller poll answer is %.2f s old, hardware poll answer %.2f s old", a_ctrl, a_hw);
  }
  std::array<std::string, 2> ctrl_state;
  Clock::time_point ctrl_sent;
  {
    std::lock_guard<std::mutex> lk(poll_mutex_);
    in.poll_valid = poll_ctrl_valid_ && poll_hw_valid_;
    if (in.poll_valid)
      in.poll_age_s = std::max(
        std::chrono::duration<double>(now - poll_ctrl_time_).count(), std::chrono::duration<double>(now - poll_hw_time_).count());
    in.controllers_active = ctrl_active_;
    in.hardware_active = hw_active_;
    ctrl_state = ctrl_state_;
    ctrl_sent = poll_ctrl_sent_;
  }
  // CRI on: the JTC gate is PER ARM. A JTC is expected inactive during its own arm's MoveJoint run and active otherwise
  // (a STRICT switch in between accepts both); any other state is a stop. The hardware-component ACTIVE check is not touched.
  std::array<CtrlGateMode, 2> arm_mode{CtrlGateMode::Normal, CtrlGateMode::Normal};
  if (cri_on)
  {
    std::lock_guard<std::mutex> lk(cri_mutex_);
    bool all = true;
    for (size_t i = 0; i < 2; ++i)
    {
      arm_mode[i] = cri_[i].mode;
      all = all && controllerStateAllowed(cri_[i].mode, ctrl_state[i], ctrl_sent >= cri_[i].mode_since);
    }
    in.controllers_active = all;
  }

  // unexpected goal ends and tracking error (only meaningful while executing)
  const bool armed = executing_.load() && !stop_latched_.load();
  if (armed)
  {
    for (ArmIo * io : {&left_, &right_})
    {
      std::lock_guard<std::mutex> lk(io->goal.m);
      if (io->goal.unexpected)
      {
        in.unexpected_goal_end = true;
        in.goal_end_reason = io->side + " goal " + io->goal.text;
      }
    }
    double err = 0.0;
    std::lock_guard<std::mutex> lk(expected_mutex_);
    size_t arm_i = 0;
    for (auto [e, q] : {std::pair<Expected *, std::vector<double> *>{&expected_left_, &ql},
                        std::pair<Expected *, std::vector<double> *>{&expected_right_, &qr}})
    {
      const size_t ai = arm_i++;
      if (arm_mode[ai] != CtrlGateMode::Normal) continue;   // robot-side move: supervised geometrically (evaluateCriRunMonitor)
      if (!e->valid || !have) continue;
      std::vector<double> want = e->q_start;
      if (e->finished)
      {
        want = e->q_end;
      }
      else if (e->started)
      {
        const double t = std::chrono::duration<double>(now - e->t_start).count();
        TrajPoint p;
        if (t > 0.0 && sampleAt(e->traj, std::min(t, e->traj.duration()), p)) want = p.positions;
      }
      if (want.size() == q->size())
        for (size_t i = 0; i < want.size(); ++i) err = std::max(err, std::abs((*q)[i] - want[i]));
    }
    in.tracking_error_rad = err;
  }

  MonitorDecision d = evaluateMonitor(in, lim);
  if (cri_on && armed)
  {
    // Driver health and the robot-side move of each arm. A latched / stale driver of EITHER arm is a FAULT STOP, a MoveJoint
    // run is also supervised against its chord (geometric, not time-indexed) and its timeout.
    // (a CRI stop reason replaces a generic one: a stale live state must say "robot_joint_states", not "/joint_states")
    bool cri_stop = false;
    for (size_t i = 0; i < 2 && !cri_stop; ++i)
    {
      CriRunMonitorInputs ri;
      ri.hw = criHwView(i);
      ri.live = criLiveView(i);
      ri.hardware_max_age_s = 0.5;
      ri.live_stale_s = lim.joint_state_stale_s;
      bool run = false;
      {
        std::lock_guard<std::mutex> lk(cri_mutex_);
        const CriArmState & a = cri_[i];
        run = a.run_active;
        ri.start = a.run_start;
        ri.goal = a.run_goal;
        ri.path_tolerance_rad = a.run_path_tol;
        ri.elapsed_s = std::chrono::duration<double>(now - a.run_sent).count();
        ri.timeout_s = a.run_timeout_s;
        ri.result_received = a.run_result_received;
      }
      if (run)
      {
        const MonitorDecision rd = evaluateCriRunMonitor(ri);
        if (rd.stop)
        {
          d = rd;
          cri_stop = true;
        }
      }
      else
      {
        std::vector<std::string> hwf;
        if (!evaluateCriHardware(ri.hw, ri.hardware_max_age_s, false, &hwf))
        {
          d.stop = true;
          d.reason = std::string("CRI FAULT (") + (i == 0 ? "left" : "right") + "): " + hwf.front();
          cri_stop = true;
        }
        else if (!ri.live.have || !(ri.live.age_s <= ri.live_stale_s))
        {
          d.stop = true;
          d.reason = std::string("CRI FAULT (") + (i == 0 ? "left" : "right") + "): robot_joint_states stale for " + std::to_string(ri.live.age_s) + " s";
          cri_stop = true;
        }
      }
    }
  }
  if (!armed)
  {
    // Idle: only report. A stop condition while idle makes the monitor "unhealthy" (barriers fail).
  }
  else if (d.stop)
  {
    doStop("monitor: " + d.reason);
  }
  else if (d.warn)
  {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "monitor: %s", d.reason.c_str());
  }
  {
    std::lock_guard<std::mutex> lk(mon_mutex_);
    mon_last_tick_ = now;
    mon_healthy_ = !d.stop;
    mon_distance_valid_ = in.distance_valid;
    mon_distance_m_ = dist;
    mon_link_a_ = la;
    mon_link_b_ = lb;
    mon_tracking_err_ = in.tracking_error_rad;
    if (executing_.load() && state_.load() == Status_::STATE_EXECUTING)
    {
      ++exec_monitor_ticks_;
      if (in.distance_valid) exec_min_distance_m_ = std::min(exec_min_distance_m_, dist);
      exec_max_tracking_rad_ = std::max(exec_max_tracking_rad_, in.tracking_error_rad);
    }
  }
  static thread_local int n = 0;
  if (++n % 2 == 0) publishStatus();
}

void CoordinatorNode::resetExecStats()
{
  std::lock_guard<std::mutex> lk(mon_mutex_);
  exec_min_distance_m_ = std::numeric_limits<double>::infinity();
  exec_max_tracking_rad_ = 0.0;
  exec_monitor_ticks_ = 0;
}

void CoordinatorNode::publishExecReport(const std::string & json)
{
  std_msgs::msg::String m;
  m.data = json;
  exec_report_pub_->publish(m);
}

void CoordinatorNode::publishStatus()
{
  Status_ msg;
  msg.header.stamp = now();
  msg.state = static_cast<uint8_t>(state_.load());
  msg.phase = current_phase_;
  {
    std::lock_guard<std::mutex> lk(data_mutex_);
    msg.plan_retained = retained_.valid;
    msg.plan_seq = retained_.valid ? retained_.id : 0;
  }
  msg.phase_validated = phase_validated_.load();
  {
    std::lock_guard<std::mutex> lk(mon_mutex_);
    msg.min_cross_distance_m = mon_distance_valid_ ? mon_distance_m_ : std::numeric_limits<double>::quiet_NaN();
    msg.closest_link_a = mon_link_a_;
    msg.closest_link_b = mon_link_b_;
    msg.barrier_ok = barrier_ok_;
    msg.barrier_message = barrier_message_;
  }
  msg.joint_states_age_s = jointStateAgeS();
  if (!std::isfinite(msg.joint_states_age_s)) msg.joint_states_age_s = std::numeric_limits<double>::quiet_NaN();
  msg.stop_latched = stop_latched_.load();
  {
    std::lock_guard<std::mutex> lk(stop_mutex_);
    msg.stop_reason = stop_reason_;
  }
  status_pub_->publish(msg);
}

}  // namespace dual_cut
