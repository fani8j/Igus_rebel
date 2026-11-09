// Robot-side joint move (CRI MoveJoint) for eligible free-space transit legs: node side. The pure decisions are in
// cri_transit.cpp (unit tested); this file is the I/O around them: the two telemetry callbacks of each driver, the plan-time
// certificate, the STRICT controller switches and runCriLeg() (one run: gate, deactivate, MoveJoint, supervise, classify,
// idle, reactivate, hold check).
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <type_traits>

#include "dual_cut_coordinator/coordinator.hpp"

namespace dual_cut
{
using namespace std::chrono_literals;
using Clock = std::chrono::steady_clock;

namespace
{
double secs(Clock::time_point t) { return std::chrono::duration<double>(t.time_since_epoch()).count(); }

int64_t systemNowNs()
{
  return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

std::string jsonEscape(const std::string & in)
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

std::string jnum(double v)
{
  if (!std::isfinite(v)) return "null";
  std::ostringstream o;
  o << std::setprecision(9) << v;
  return o.str();
}

std::string jbool(bool b) { return b ? "true" : "false"; }

std::string jarr(const std::vector<double> & v)
{
  std::string out = "[";
  for (size_t i = 0; i < v.size(); ++i) out += (i ? "," : "") + jnum(v[i]);
  return out + "]";
}

// The effective MoveJoint caps (move_joint_max_velocity_percent / move_joint_max_acceleration_percent) are in HardwareState
// since driver commit c2e8811. Without the fields the cap is "unknown" and dispatch is refused unless
// cri_assume_driver_cap_ok; the static_assert below keeps a field rename from silently turning the cap check off.
template<class T, class = void>
struct HasMoveJointCaps : std::false_type
{
};
template<class T>
struct HasMoveJointCaps<
  T, std::void_t<decltype(std::declval<const T &>().move_joint_max_velocity_percent), decltype(std::declval<const T &>().move_joint_max_acceleration_percent)>>
: std::true_type
{
};
static_assert(HasMoveJointCaps<igus_rebel_msgs::msg::HardwareState>::value,
              "igus_rebel_msgs/HardwareState lost the move_joint_max_*_percent cap fields");
template<class T>
void readDriverCaps(const T & m, CriHwView & v)
{
  if constexpr (HasMoveJointCaps<T>::value)
  {
    v.cap_known = true;
    v.cap_velocity_percent = static_cast<double>(m.move_joint_max_velocity_percent);
    v.cap_acceleration_percent = static_cast<double>(m.move_joint_max_acceleration_percent);
  }
  else
  {
    (void)m;
    v.cap_known = false;
    v.cap_velocity_percent = v.cap_acceleration_percent = std::numeric_limits<double>::quiet_NaN();
  }
}
}  // namespace

CriParams CoordinatorNode::criParams() const
{
  CriParams p;
  p.enabled = get_parameter("cri_transit_enabled").as_bool();
  p.velocity_percent = get_parameter("cri_transit_velocity_percent").as_double();
  p.acceleration_percent = get_parameter("cri_transit_acceleration_percent").as_double();
  p.min_length_rad = get_parameter("cri_transit_min_length_rad").as_double();
  p.path_tolerance_rad = get_parameter("cri_transit_path_tolerance_rad").as_double();
  p.goal_tolerance_rad = get_parameter("cri_transit_goal_tolerance_rad").as_double();
  p.start_tolerance_rad = get_parameter("cri_transit_start_tolerance_rad").as_double();
  p.max_runs_per_arm_phase = static_cast<int>(get_parameter("cri_transit_max_runs_per_arm_phase").as_int());
  p.driver_max_velocity_cap_percent = get_parameter("cri_driver_max_velocity_cap_percent").as_double();
  p.driver_max_acceleration_cap_percent = get_parameter("cri_driver_max_acceleration_cap_percent").as_double();
  p.assume_driver_cap_ok = get_parameter("cri_assume_driver_cap_ok").as_bool();
  p.reach_m = get_parameter("cri_transit_reach_m").as_double();
  p.subchord_velocity_rad_s = get_parameter("cri_subchord_velocity_rad_s").as_double();
  p.subchord_acceleration_rad_s2 = get_parameter("cri_subchord_acceleration_rad_s2").as_double();
  return p;
}

// ---------------------------------------------------------------------------------------------
// telemetry of the two drivers
// ---------------------------------------------------------------------------------------------

void CoordinatorNode::onHardwareState(size_t arm, const HardwareState::SharedPtr msg)
{
  const auto now = Clock::now();
  std::lock_guard<std::mutex> lk(cri_mutex_);
  CriArmState & a = cri_[arm];
  a.hw.have = true;
  a.hw.connected = msg->connected;
  a.hw.status_fresh = msg->status_fresh;
  a.hw.latched = msg->latched;
  a.hw.move_stop_pending = msg->move_stop_pending;
  a.hw.move_active = msg->move_active;
  a.hw.activated = msg->activated;
  readDriverCaps(*msg, a.hw);
  a.latch_reason = msg->latch_reason;
  a.hw_rx = now;
  a.hw_stamp_ns = rclcpp::Time(msg->header.stamp).nanoseconds();
}

void CoordinatorNode::onRobotJointStates(size_t arm, const sensor_msgs::msg::JointState::SharedPtr msg)
{
  if (!model_ready_.load()) return;
  const ArmIo & io = arm == 0 ? left_ : right_;
  if (io.spec.joint_names.empty() || msg->position.size() != msg->name.size()) return;
  std::vector<double> q;
  double reported_v = 0.0;
  const bool has_vel = msg->velocity.size() == msg->name.size();
  for (const auto & n : io.spec.joint_names)
  {
    const auto it = std::find(msg->name.begin(), msg->name.end(), n);
    if (it == msg->name.end())
    {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "robot_joint_states of %s lacks joint %s", io.side.c_str(), n.c_str());
      std::lock_guard<std::mutex> lk(cri_mutex_);
      cri_[arm].live_have = false;
      return;
    }
    const size_t k = static_cast<size_t>(it - msg->name.begin());
    q.push_back(msg->position[k]);
    if (has_vel) reported_v = std::max(reported_v, std::isfinite(msg->velocity[k]) ? std::abs(msg->velocity[k]) : std::numeric_limits<double>::infinity());
  }
  const auto now = Clock::now();
  std::lock_guard<std::mutex> lk(cri_mutex_);
  CriArmState & a = cri_[arm];
  a.live_q = q;
  a.live_have = true;
  a.live_rx = now;
  a.live_stamp_ns = rclcpp::Time(msg->header.stamp).nanoseconds();
  a.window.add(secs(now), q);
  // the larger of the windowed position speed and the reported speed: a missing velocity can never look like "at rest"
  a.rest.update(secs(now), std::max(a.window.maxAbsVelocity(), reported_v));
}

CriHwView CoordinatorNode::criHwView(size_t arm) const
{
  std::lock_guard<std::mutex> lk(cri_mutex_);
  const CriArmState & a = cri_[arm];
  CriHwView v = a.hw;
  if (!v.have) return v;
  const auto now = Clock::now();
  double age = std::chrono::duration<double>(now - a.hw_rx).count();
  if (a.hw_stamp_ns != 0) age = std::max(age, 1e-9 * static_cast<double>(systemNowNs() - a.hw_stamp_ns));
  v.age_s = age;
  return v;
}

CriLiveView CoordinatorNode::criLiveView(size_t arm) const
{
  std::lock_guard<std::mutex> lk(cri_mutex_);
  const CriArmState & a = cri_[arm];
  CriLiveView v;
  if (!a.live_have) return v;
  const auto now = Clock::now();
  double age = std::chrono::duration<double>(now - a.live_rx).count();
  if (a.live_stamp_ns != 0) age = std::max(age, 1e-9 * static_cast<double>(systemNowNs() - a.live_stamp_ns));
  v.have = true;
  v.age_s = age;
  v.q = a.live_q;
  v.rest_s = a.rest.restDuration(secs(now));
  return v;
}

bool CoordinatorNode::criQ(const ArmIo & io, std::vector<double> & q, double * age_s) const
{
  const CriLiveView v = criLiveView(armIndex(io));
  if (!v.have) return false;
  q = v.q;
  if (age_s) *age_s = v.age_s;
  return true;
}

bool CoordinatorNode::armQEff(const ArmIo & io, std::vector<double> & q, double * age_s) const
{
  if (pb("cri_transit_enabled")) return criQ(io, q, age_s);
  return armQ(io, q, age_s);
}

double CoordinatorNode::restDurationS()
{
  if (!pb("cri_transit_enabled"))
  {
    std::lock_guard<std::mutex> lk(js_mutex_);
    return rest_tracker_.restDuration(secs(Clock::now()));
  }
  return std::min(criLiveView(0).rest_s, criLiveView(1).rest_s);
}

Status CoordinatorNode::criBothArmsGate(double live_max_age_s)
{
  CriArmGateLimits lim;
  lim.hardware_max_age_s = 0.5;
  lim.live_max_age_s = live_max_age_s;
  std::string bad;
  for (size_t i = 0; i < 2; ++i)
  {
    const Status s = evaluateCriArmGate(criHwView(i), criLiveView(i), lim);
    if (!s) bad += std::string(bad.empty() ? "" : " | ") + (i == 0 ? "left: " : "right: ") + s.message;
  }
  return bad.empty() ? Status::Ok() : Status::Error(bad);
}

void CoordinatorNode::criSetMode(size_t arm, CtrlGateMode mode)
{
  std::lock_guard<std::mutex> lk(cri_mutex_);
  cri_[arm].mode = mode;
  cri_[arm].mode_since = Clock::now();
}

bool CoordinatorNode::waitAtRest(const std::vector<ArmIo *> &, double timeout_s)
{
  const auto deadline = Clock::now() + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(timeout_s));
  while (!stop_latched_ && Clock::now() < deadline)
  {
    if (restDurationS() >= pd("barrier_rest_duration_s")) return true;
    std::this_thread::sleep_for(50ms);
  }
  return false;
}

// ---------------------------------------------------------------------------------------------
// plan time: eligibility + certificate + selection
// ---------------------------------------------------------------------------------------------

CriPlan CoordinatorNode::planCri(const PhaseSet & set, const planning_scene::PlanningScenePtr & scene, std::string & json)
{
  CriPlan plan;
  plan.computed = true;
  plan.params = criParams();
  const CriParams & cp = plan.params;
  std::ostringstream o;
  o << std::setprecision(9);
  o << "{\"scope\":\"phases mode, SERIAL phases only (stage 1); parallel phases and sequence mode stay 100 % on the JTC\"";
  o << ",\"params\":{\"velocity_percent\":" << cp.velocity_percent << ",\"acceleration_percent\":" << cp.acceleration_percent << ",\"min_length_rad\":"
    << cp.min_length_rad << ",\"path_tolerance_rad\":" << cp.path_tolerance_rad << ",\"goal_tolerance_rad\":" << cp.goal_tolerance_rad
    << ",\"start_tolerance_rad\":" << cp.start_tolerance_rad << ",\"max_runs_per_arm_phase\":" << cp.max_runs_per_arm_phase << ",\"reach_m_param\":" << cp.reach_m
    << ",\"subchord_velocity_rad_s\":" << cp.subchord_velocity_rad_s << ",\"subchord_acceleration_rad_s2\":" << cp.subchord_acceleration_rad_s2
    << ",\"padding_rule\":\"padding = path_tolerance * reach (no stop travel: a Move Stop decelerates along the certified chord)\"}";
  const Status ps_ok = validateCriParams(cp);
  if (!ps_ok)
  {
    plan.skipped = "invalid parameters: " + ps_ok.message;
    o << ",\"skipped\":\"" << jsonEscape(plan.skipped) << "\"}";
    json = o.str();
    return plan;
  }

  // lever arm per arm: the constant, or the rigorous bound from the model; the bound is reported either way
  std::array<double, 2> reach_rigorous{{0.0, 0.0}}, reach{{0.0, 0.0}};
  std::array<std::string, 2> reach_err;
  for (size_t i = 0; i < 2; ++i)
  {
    const ArmIo & io = i == 0 ? left_ : right_;
    const Status rs = armReachBound(*model_, io.spec, reach_rigorous[i]);
    if (!rs) reach_err[i] = rs.message;
    reach[i] = cp.reach_m > 0.0 ? cp.reach_m : reach_rigorous[i];
  }
  o << ",\"reach_m\":{\"left\":" << reach[0] << ",\"right\":" << reach[1] << ",\"left_rigorous_sum_m\":" << reach_rigorous[0] << ",\"right_rigorous_sum_m\":"
    << reach_rigorous[1] << ",\"constant_below_rigorous\":" << jbool(cp.reach_m > 0.0 && (cp.reach_m < reach_rigorous[0] || cp.reach_m < reach_rigorous[1])) << "}";

  CriGeomLimits gl;
  gl.joint_count = kCriJointCount;
  gl.min_length_rad = cp.min_length_rad;
  gl.goal_tolerance_rad = cp.goal_tolerance_rad;

  std::array<std::unique_ptr<PhaseValidator>, 2> padded;      // validators with the extra padding of each arm
  std::array<std::string, 2> padded_err;
  std::array<bool, 2> padded_tried{{false, false}};
  auto paddedFor = [&](size_t i) -> PhaseValidator *
  {
    if (!padded_tried[i])
    {
      padded_tried[i] = true;
      if (!(reach[i] > 0.0))
        padded_err[i] = "no lever arm (" + (reach_err[i].empty() ? std::string("reach is 0") : reach_err[i]) + ")";
      else
      {
        const double pad = criPaddingM(cp.path_tolerance_rad, reach[i]);
        const Status vs = makeValidator(scene, padded[i], pad);
        if (!vs) padded_err[i] = "padded validator: " + vs.message;
      }
    }
    return padded[i].get();
  };

  o << ",\"arm_phases\":[";
  bool first_ap = true;
  for (const Phase * ph : {&set.a, &set.b})
    for (const ArmPhase * arm : {&ph->left(), &ph->right()})
    {
      const size_t i = arm->side == "left" ? 0 : 1;
      CriArmPhasePlan ap;
      ap.phase = ph->name();
      ap.side = arm->side;
      const ArmPhase & other = arm->side == "left" ? ph->right() : ph->left();
      const bool serial = ph->validation().serial_fallback;
      if (!serial)
        ap.skipped = "phase runs in parallel: stage 1 sends serial phases only, every segment stays on the JTC";
      else
      {
        const bool mover_first = (ph->validation().serial_first == "right" ? "right" : "left") == arm->side;
        // serial rule of the validator: the first mover is checked against the other arm at ITS phase start, the second
        // mover against the other arm where the first one ended
        const std::vector<double> & parked = mover_first ? other.start_q : other.end_q;
        for (const CriLegGeometry & g : scanTransitLegs(*arm, gl))
        {
          CriLegRecord rec;
          rec.geometry = g;
          rec.reach_m = reach[i];
          rec.padding_m = criPaddingM(cp.path_tolerance_rad, reach[i]);
          if (!g.eligible)
            rec.reason = g.reason;
          else
          {
            PhaseValidator * v = paddedFor(i);
            if (!v)
              rec.reason = "certificate unavailable: " + padded_err[i];
            else
            {
              const ChordStretch st = v->certifyChord(arm->side, g.joint_names, g.start, g.goal, other.joint_names, parked, kCriResampleStepRad, cp.min_length_rad);
              const ChordCheck & cc = st.check;
              rec.samples = cc.samples;
              rec.cross_effective_min_m = cc.cross_effective_min_m;
              rec.cross_required_m = cc.cross_required_m;
              rec.table_min_m = cc.table_min_m;
              rec.table_required_m = cc.table_required_m;
              rec.object_min_m = cc.object_min_m;
              rec.object_required_m = cc.object_required_m;
              if (st.found)
              {
                rec.certified = true;
                rec.partial = !st.full;
                rec.s0 = st.s0;
                rec.s1 = st.s1;
                rec.cri_start = st.start;
                rec.cri_goal = st.goal;
                rec.cri_length_rad = st.length_rad;
                if (rec.partial)
                {
                  rec.reason = "partial chord: the full chord fails the padded checks near its ends";
                  if (st.first_sample > 0)
                  {
                    rec.head = makeChordSegment(g.joint_names, g.start, st.start, cp.subchord_velocity_rad_s, cp.subchord_acceleration_rad_s2, g.stage_name + " [JTC head]");
                    rec.head_s = rec.head.trajectory.duration();
                  }
                  if (st.last_sample < st.total_steps)
                  {
                    rec.tail = makeChordSegment(g.joint_names, st.goal, g.goal, cp.subchord_velocity_rad_s, cp.subchord_acceleration_rad_s2, g.stage_name + " [JTC tail]");
                    rec.tail_s = rec.tail.trajectory.duration();
                  }
                }
              }
              else
                rec.reason = "certificate failed: " + (st.reason.empty() ? cc.reason : st.reason);
            }
          }
          ap.legs.push_back(std::move(rec));
        }
        selectCriLegs(ap, cp.max_runs_per_arm_phase);
        // tip-on-line: the gate checks the CUT segments only; a Transit leg has no CUT sample, so there is nothing to re-run
      }
      o << (first_ap ? "" : ",") << "{\"phase\":\"" << ap.phase << "\",\"side\":\"" << ap.side << "\",\"serial\":" << jbool(serial) << ",\"skipped\":\""
        << jsonEscape(ap.skipped) << "\",\"tip_on_line\":\"not applicable: a Transit leg has no CUT sample\",\"selected_segments\":[";
      for (size_t k = 0; k < ap.selected_segments.size(); ++k) o << (k ? "," : "") << ap.selected_segments[k];
      o << "],\"legs\":[";
      for (size_t k = 0; k < ap.legs.size(); ++k)
      {
        const CriLegRecord & l = ap.legs[k];
        o << (k ? "," : "") << "{\"segment\":" << l.geometry.segment_index << ",\"name\":\"" << jsonEscape(l.geometry.stage_name) << "\",\"eligible\":"
          << jbool(l.geometry.eligible) << ",\"certified\":" << jbool(l.certified) << ",\"selected\":" << jbool(l.selected) << ",\"reason\":\""
          << jsonEscape(l.reason) << "\",\"length_rad\":" << jnum(l.geometry.length_rad) << ",\"max_chord_deviation_rad\":" << jnum(l.geometry.max_chord_deviation_rad)
          << ",\"reach_m\":" << jnum(l.reach_m) << ",\"padding_m\":" << jnum(l.padding_m) << ",\"partial\":" << jbool(l.partial)
          << ",\"s0\":" << jnum(l.s0) << ",\"s1\":" << jnum(l.s1) << ",\"cri_length_rad\":" << jnum(l.cri_length_rad) << ",\"head_s\":" << jnum(l.head_s)
          << ",\"tail_s\":" << jnum(l.tail_s) << ",\"cri_start\":" << jarr(l.cri_start) << ",\"cri_goal\":" << jarr(l.cri_goal)
          << ",\"samples\":" << l.samples << ",\"cross_effective_min_m\":" << jnum(l.cross_effective_min_m) << ",\"cross_required_m\":" << jnum(l.cross_required_m)
          << ",\"table_min_m\":" << jnum(l.table_min_m) << ",\"table_required_m\":" << jnum(l.table_required_m) << ",\"object_min_m\":" << jnum(l.object_min_m)
          << ",\"object_required_m\":" << jnum(l.object_required_m) << "}";
      }
      o << "]}";
      first_ap = false;
      plan.arm_phases.push_back(std::move(ap));
    }
  o << "]}";
  json = o.str();
  size_t selected = 0;
  for (const auto & ap : plan.arm_phases) selected += ap.selected_segments.size();
  RCLCPP_INFO(get_logger(), "CRI transit: %zu leg(s) selected for the robot-side joint move (serial phases only)", selected);
  return plan;
}

// ---------------------------------------------------------------------------------------------
// controller switching
// ---------------------------------------------------------------------------------------------

Status CoordinatorNode::criControllerState(ArmIo & io, std::string & state)
{
  if (!list_controllers_->service_is_ready()) return Status::Error("list_controllers is not available");
  auto pending = list_controllers_->async_send_request(std::make_shared<controller_manager_msgs::srv::ListControllers::Request>());
  if (pending.future.wait_for(3s) != std::future_status::ready)
  {
    list_controllers_->remove_pending_request(pending.request_id);
    return Status::Error("timeout calling list_controllers");
  }
  const auto r = pending.future.get();
  for (const auto & c : r->controller)
    if (c.name == io.controller)
    {
      state = c.state;
      return Status::Ok();
    }
  return Status::Error("controller " + io.controller + " is not listed");
}

Status CoordinatorNode::criSwitchController(ArmIo & io, bool activate)
{
  if (!switch_controller_->service_is_ready()) return Status::Error("switch_controller is not available");
  auto req = std::make_shared<controller_manager_msgs::srv::SwitchController::Request>();
  (activate ? req->activate_controllers : req->deactivate_controllers) = {io.controller};
  req->strictness = controller_manager_msgs::srv::SwitchController::Request::STRICT;
  req->activate_asap = false;
  req->timeout = rclcpp::Duration::from_seconds(5.0);
  auto pending = switch_controller_->async_send_request(req);
  if (pending.future.wait_for(10s) != std::future_status::ready)
  {
    switch_controller_->remove_pending_request(pending.request_id);
    return Status::Error("timeout calling switch_controller");
  }
  const auto resp = pending.future.get();
  if (!resp->ok) return Status::Error(std::string("STRICT switch_controller ") + (activate ? "activate " : "deactivate ") + io.controller + " refused");
  // confirm with list_controllers: the state must really be the wanted one before anything else happens
  const char * want = activate ? "active" : "inactive";
  std::string state, last_err;
  const auto deadline = Clock::now() + 3s;
  while (Clock::now() < deadline)
  {
    const Status s = criControllerState(io, state);
    if (s && state == want) return Status::Ok();
    last_err = s ? "state is '" + state + "'" : s.message;
    std::this_thread::sleep_for(50ms);
  }
  return Status::Error("controller " + io.controller + " is not " + want + " after the switch (" + last_err + ")");
}

Status CoordinatorNode::criRecoverControllers()
{
  // Both drivers are idle, not latched, no stop pending (the caller checked). An arm whose JTC is inactive gets a STRICT
  // activation and a hold check; an arm whose JTC is active just goes back to Normal.
  for (ArmIo * io : {&left_, &right_})
  {
    const size_t ai = armIndex(*io);
    std::string state;
    Status s = criControllerState(*io, state);
    if (!s) return s;
    if (state == "inactive")
    {
      criSetMode(ai, CtrlGateMode::Switching);
      s = criSwitchController(*io, true);
      if (!s) return s;
      // hold check: the reactivated JTC must hold where the robot is
      std::this_thread::sleep_for(400ms);
      std::vector<double> js, live;
      double js_age = 0.0;
      if (!armQ(*io, js, &js_age) || !criQ(*io, live) || js_age > 0.2)
        return Status::Error(io->side + ": no fresh joint states after the controller reactivation");
      const double e = maxAbsDiff(js, live);
      if (!(e <= pd("barrier_start_tolerance_rad")))
        return Status::Error(io->side + ": after the reactivation /joint_states differs from the live robot state by " + std::to_string(e) + " rad");
    }
    else if (state != "active")
      return Status::Error(io->side + ": trajectory controller is '" + state + "'; recover it by hand (dual_arm_recover.py)");
    criSetMode(ai, CtrlGateMode::Normal);
  }
  return Status::Ok();
}

// ---------------------------------------------------------------------------------------------
// one robot-side run
// ---------------------------------------------------------------------------------------------

bool CoordinatorNode::runCriLeg(ArmIo & io, const ArmPhase &, const CriLegRecord & leg, const CriParams & cp, CriRunReport & rep)
{
  const size_t ai = armIndex(io);
  const auto t_start_run = Clock::now();
  cri_used_ = true;
  auto fault = [&](const std::string & why) -> bool
  {
    rep.ok = false;
    rep.note = why;
    RCLCPP_ERROR(get_logger(), "CRI run on %s: FAULT: %s", io.side.c_str(), why.c_str());
    if (!stop_latched_.load()) doStop("CRI " + io.side + ": " + why);
    rep.wall_s = std::chrono::duration<double>(Clock::now() - t_start_run).count();
    return false;
  };

  // chord in the driver's joint order (spec order)
  std::vector<double> start, goal;
  for (const auto & n : io.spec.joint_names)
  {
    std::string plain;
    stripJointPrefix(io.prefix, n, plain);
    const auto it = std::find(leg.geometry.joint_names.begin(), leg.geometry.joint_names.end(), plain);
    if (it == leg.geometry.joint_names.end()) return fault("joint " + plain + " missing in the leg");
    const size_t k = static_cast<size_t>(it - leg.geometry.joint_names.begin());
    start.push_back(leg.cri_start[k]);
    goal.push_back(leg.cri_goal[k]);
  }

  // (a) pre-dispatch gate, polled until barrier_timeout_s (the arm may still be settling after the previous run)
  const auto gate_deadline = Clock::now() + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(pd("barrier_timeout_s")));
  CriHwView hw;
  while (true)
  {
    if (stop_latched_.load()) return fault("stop latched before the dispatch");
    hw = criHwView(ai);
    CriPreDispatchInputs pi;
    pi.hw = hw;
    pi.live = criLiveView(ai);
    pi.chord_start = start;
    pi.start_tolerance_rad = cp.start_tolerance_rad;
    pi.stop_latched = stop_latched_.load();
    {
      std::lock_guard<std::mutex> lk(poll_mutex_);
      const bool fresh = poll_ctrl_valid_ && std::chrono::duration<double>(Clock::now() - poll_ctrl_time_).count() < 1.0;
      pi.controller_active = fresh && ctrl_state_[ai] == "active";
    }
    std::string cap_why;
    pi.cap_ok = criCapOk(hw, cp, &cap_why);
    CriPreDispatchLimits lim;
    lim.hardware_max_age_s = 0.5;
    lim.live_max_age_s = std::max(0.2, pd("barrier_joint_state_age_s"));
    lim.rest_duration_s = pd("barrier_rest_duration_s");
    Status g = evaluateCriPreDispatch(pi, lim);
    std::string why = g.message;
    if (g)
    {
      const Status og = criBothArmsGate(std::max(0.2, pd("barrier_joint_state_age_s")));
      if (!og)
      {
        g = og;
        why = "arm gate: " + og.message;
      }
    }
    if (g && !io.cri_action->action_server_is_ready())
    {
      g = Status::Error("MoveJoint action server is not available");
      why = g.message;
    }
    if (g) break;
    if (Clock::now() > gate_deadline) return fault(why + (cap_why.empty() ? "" : " [" + cap_why + "]"));
    std::this_thread::sleep_for(50ms);
  }
  const double eff_vel = hw.cap_known ? std::min(cp.velocity_percent, hw.cap_velocity_percent) : cp.velocity_percent;
  const double model_s = criModelDurationS(leg.cri_length_rad, eff_vel, cp.acceleration_percent);
  const double timeout_s = criTimeoutS(model_s);
  rep.model_s = model_s;

  // (b) STRICT deactivation of this arm's JTC, confirmed by list_controllers BEFORE anything is sent
  criSetMode(ai, CtrlGateMode::Switching);
  Status sw = criSwitchController(io, false);
  if (!sw) return fault("deactivation: " + sw.message);
  RCLCPP_INFO(get_logger(), "CRI run on %s: %s confirmed INACTIVE by list_controllers (t=%.3f)", io.side.c_str(), io.controller.c_str(), now().seconds());
  if (stop_latched_.load()) return fault("stop latched after the controller deactivation");
  // re-check right before the send: the driver may have changed during the switch
  {
    std::vector<std::string> f;
    const CriHwView hw2 = criHwView(ai);
    if (!evaluateCriHardware(hw2, 0.5, true, &f)) return fault("before the dispatch: " + f.front());
    const CriLiveView lv = criLiveView(ai);
    if (!lv.have || !(lv.age_s < 0.5) || !(maxAbsDiff(lv.q, start) <= cp.start_tolerance_rad)) return fault("before the dispatch: the live state left the chord start");
  }

  // (c) the run state for the monitor, then the goal
  const auto t_sent = Clock::now();
  {
    std::lock_guard<std::mutex> lk(cri_mutex_);
    CriArmState & a = cri_[ai];
    a.mode = CtrlGateMode::CriRun;
    a.mode_since = t_sent;
    a.run_active = true;
    a.run_start = start;
    a.run_goal = goal;
    a.run_path_tol = cp.path_tolerance_rad;
    a.run_sent = t_sent;
    a.run_timeout_s = timeout_s;
    a.run_result_received = false;
    a.sending = true;
    a.handle.reset();
    a.accepted = a.rejected = a.have_result = false;
    a.result = MoveJointResultView{};
    a.max_path_deviation = 0.0;
    a.progress = 0.0;
  }
  MoveJoint::Goal g;
  g.joint_names = io.spec.joint_names;
  g.start_positions = start;
  g.goal_positions = goal;
  g.velocity_percent = cp.velocity_percent;
  g.acceleration_percent = cp.acceleration_percent;
  g.start_tolerance = cp.start_tolerance_rad;
  g.path_tolerance = cp.path_tolerance_rad;
  g.goal_tolerance = cp.goal_tolerance_rad;
  g.timeout_s = 0.0;   // the driver computes 2 x its model + 1 s; the monitor's own timeout is 2 x model + 10 s
  rclcpp_action::Client<MoveJoint>::SendGoalOptions opts;
  opts.goal_response_callback = [this, ai](MoveJointHandle::SharedPtr h)
  {
    bool late_stop = false;
    {
      std::lock_guard<std::mutex> lk(cri_mutex_);
      CriArmState & a = cri_[ai];
      a.sending = false;
      if (h)
      {
        a.handle = h;
        a.accepted = true;
      }
      else
        a.rejected = true;
      late_stop = h && stop_latched_.load();
    }
    // accepted after a stop was latched: doStop() may have missed this goal. Never cancel from inside the callback.
    if (late_stop) requestStopAsync("MoveJoint goal accepted after the stop was latched");
  };
  opts.feedback_callback = [this, ai](MoveJointHandle::SharedPtr, const std::shared_ptr<const MoveJoint::Feedback> fb)
  {
    std::lock_guard<std::mutex> lk(cri_mutex_);
    cri_[ai].progress = fb->progress;
    cri_[ai].max_path_deviation = std::max(cri_[ai].max_path_deviation, fb->max_path_deviation);
  };
  opts.result_callback = [this, ai](const MoveJointHandle::WrappedResult & r)
  {
    std::lock_guard<std::mutex> lk(cri_mutex_);
    CriArmState & a = cri_[ai];
    switch (r.code)
    {
      case rclcpp_action::ResultCode::SUCCEEDED: a.result.outcome = ActionOutcome::Succeeded; break;
      case rclcpp_action::ResultCode::ABORTED: a.result.outcome = ActionOutcome::Aborted; break;
      case rclcpp_action::ResultCode::CANCELED: a.result.outcome = ActionOutcome::Canceled; break;
      default: a.result.outcome = ActionOutcome::Unknown; break;
    }
    if (r.result)
    {
      a.result.error_code = r.result->error_code;
      a.result.end_reason = r.result->end_reason;
      a.result.stop_sent = r.result->stop_sent;
      a.result.stop_confirmed = r.result->stop_confirmed;
      a.result.message = r.result->message;
    }
    a.have_result = true;
    a.run_result_received = true;
  };
  if (stop_latched_.load()) return fault("stop latched before the dispatch");
  io.cri_action->async_send_goal(g, opts);
  RCLCPP_INFO(
    get_logger(), "CRI run on %s: MoveJoint sent (%.3f rad, %.0f %% / %.0f %%, model %.1f s, timeout %.1f s)", io.side.c_str(), leg.cri_length_rad,
    cp.velocity_percent, cp.acceleration_percent, model_s, timeout_s);

  // (d) wait; the monitor supervises the chord, the driver health, the cross-arm distance and the timeout
  const auto response_deadline = t_sent + 5s;
  Clock::time_point t_result;
  while (true)
  {
    if (stop_latched_.load()) return fault("stop latched during the run");
    bool rejected = false, have_result = false, sending = false;
    {
      std::lock_guard<std::mutex> lk(cri_mutex_);
      rejected = cri_[ai].rejected;
      have_result = cri_[ai].have_result;
      sending = cri_[ai].sending;
    }
    if (rejected) return fault("the MoveJoint goal was rejected");
    if (have_result)
    {
      t_result = Clock::now();
      break;
    }
    if (sending && Clock::now() > response_deadline) return fault("no answer to the MoveJoint goal within 5 s");
    if (std::chrono::duration<double>(Clock::now() - t_sent).count() > timeout_s + 1.0)
    {
      doStop("CRI timeout");
      return fault("timeout " + std::to_string(timeout_s) + " s");
    }
    std::this_thread::sleep_for(20ms);
  }

  // (e) result: wait for one live sample after the result, then classify. ERROR / latched is FAULT whatever error_code says.
  {
    const auto until = Clock::now() + 500ms;
    while (Clock::now() < until)
    {
      {
        std::lock_guard<std::mutex> lk(cri_mutex_);
        if (cri_[ai].live_have && cri_[ai].live_rx > t_result) break;
      }
      std::this_thread::sleep_for(10ms);
    }
  }
  MoveJointResultView rv;
  {
    std::lock_guard<std::mutex> lk(cri_mutex_);
    rv = cri_[ai].result;
    rep.max_path_deviation_rad = cri_[ai].max_path_deviation;
  }
  const CriHwView hw_after = criHwView(ai);
  const CriLiveView live_after = criLiveView(ai);
  rv.hardware_latched = !hw_after.have || hw_after.latched;
  rv.live_valid = live_after.have && live_after.age_s < 0.5;
  rv.live_q = live_after.q;
  rep.action_outcome = static_cast<int>(rv.outcome);
  rep.error_code = rv.error_code;
  rep.end_reason = rv.end_reason;
  rep.message = rv.message;
  rep.stop_sent = rv.stop_sent;
  rep.stop_confirmed = rv.stop_confirmed;
  const CriVerdict verdict = classifyMoveJointResult(rv, goal, cp.goal_tolerance_rad);
  if (!verdict.success) return fault(verdict.reason);
  {
    std::lock_guard<std::mutex> lk(cri_mutex_);
    cri_[ai].run_active = false;   // the chord supervision ends with the move
  }
  {
    // The tracking reference of the parked arm is the chord end from now on. It must be set BEFORE the JTC is reactivated
    // (the monitor tracks this arm against it again as soon as the mode is Normal); found in the first mock run, where the
    // old reference (the chord start) tripped "tracking error 2.09 rad" during the hold check.
    std::lock_guard<std::mutex> lk(expected_mutex_);
    Expected & dst = &io == &left_ ? expected_left_ : expected_right_;
    dst = Expected{};
    dst.valid = true;
    dst.q_start = goal;
    dst.q_end = goal;
  }

  // (f) the driver must report the move finished and no stop pending, THEN the strict reactivation
  {
    const auto until = Clock::now() + 3s;
    bool idle = false;
    while (Clock::now() < until && !stop_latched_.load())
    {
      CriHwView h;
      Clock::time_point rx;
      {
        std::lock_guard<std::mutex> lk(cri_mutex_);
        h = cri_[ai].hw;
        rx = cri_[ai].hw_rx;
      }
      if (h.have && rx > t_result && !h.latched && !h.move_active && !h.move_stop_pending && h.activated)
      {
        idle = true;
        break;
      }
      if (h.have && h.latched) return fault("the driver latched after the move");
      std::this_thread::sleep_for(20ms);
    }
    if (!idle) return fault(stop_latched_.load() ? "stop latched while waiting for the driver to go idle" : "hardware_state did not report move_active=false within 3 s");
  }
  criSetMode(ai, CtrlGateMode::Switching);
  sw = criSwitchController(io, true);
  if (!sw) return fault("reactivation: " + sw.message);
  criSetMode(ai, CtrlGateMode::Normal);
  RCLCPP_INFO(get_logger(), "CRI run on %s: %s REACTIVATED, confirmed active (t=%.3f)", io.side.c_str(), io.controller.c_str(), now().seconds());
  // the reactivated JTC must hold where the robot is (no jump back to an old trajectory)
  {
    const auto until = Clock::now() + 400ms;
    while (Clock::now() < until)
    {
      if (stop_latched_.load()) return fault("stop latched during the hold check");
      std::this_thread::sleep_for(20ms);
    }
    std::vector<double> js, live;
    double js_age = 0.0, live_age = 0.0;
    if (!armQ(io, js, &js_age) || !criQ(io, live, &live_age) || js_age > 0.3 || live_age > 0.3) return fault("hold check: no fresh joint states after the reactivation");
    const double ej = maxAbsDiff(js, goal), el = maxAbsDiff(live, goal);
    const double tol = std::max(cp.goal_tolerance_rad, 0.002) * 2.0;
    if (!(ej <= tol && el <= tol))
      return fault("hold check: after the reactivation /joint_states is " + std::to_string(ej) + " rad and the live state " + std::to_string(el) + " rad from the goal");
  }
  rep.ok = true;
  rep.wall_s = std::chrono::duration<double>(Clock::now() - t_start_run).count();
  RCLCPP_INFO(
    get_logger(), "CRI run on %s done in %.2f s (model %.2f s, planned JTC %.2f s), max path deviation %.4f rad", io.side.c_str(), rep.wall_s, model_s, rep.planned_s,
    rep.max_path_deviation_rad);
  return true;
}

std::string CoordinatorNode::criRunReportJson(const std::vector<CriRunReport> & runs, const std::string & note) const
{
  std::ostringstream o;
  o << std::setprecision(9) << "{\"note\":\"" << jsonEscape(note) << "\",\"runs\":[";
  for (size_t i = 0; i < runs.size(); ++i)
  {
    const CriRunReport & r = runs[i];
    o << (i ? "," : "") << "{\"arm\":\"" << r.arm << "\",\"type\":\"" << r.type << "\",\"first_segment\":" << r.first_segment << ",\"end_segment\":"
      << r.end_segment << ",\"ok\":" << jbool(r.ok) << ",\"planned_s\":" << jnum(r.planned_s) << ",\"wall_s\":" << jnum(r.wall_s);
    if (r.type == "cri")
      o << ",\"model_s\":" << jnum(r.model_s) << ",\"action_outcome\":" << r.action_outcome << ",\"error_code\":" << r.error_code << ",\"end_reason\":\""
        << jsonEscape(r.end_reason) << "\",\"stop_sent\":" << jbool(r.stop_sent) << ",\"stop_confirmed\":" << jbool(r.stop_confirmed)
        << ",\"max_path_deviation_rad\":" << jnum(r.max_path_deviation_rad) << ",\"message\":\"" << jsonEscape(r.message) << "\"";
    o << ",\"note\":\"" << jsonEscape(r.note) << "\"}";
  }
  o << "]}";
  return o.str();
}

}  // namespace dual_cut
