// Relay between MoveIt Servo's JointTrajectory output and a joint_trajectory_controller (topic interface).
// Logic lives in relay_core.hpp (unit-tested); this file is the ROS glue:
//   input_topic         <- Servo (its command_out_topic is remapped here); only points[0].velocities is used
//   joint_states_topic  <- measured positions of the joints named in the Servo message
//   output_topic        -> the JTC's joint_trajectory topic
// Joint limits come from robot_description (parameter, or the get_parameters service of robot_description_node).
// Until limits are known nothing is forwarded (fail closed).
#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <unordered_map>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include <rcl_interfaces/msg/parameter_type.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <urdf/model.h>

#include "rebel_servo_relay/relay_core.hpp"

using rebel_servo_relay::Action;
using rebel_servo_relay::JointLimit;
using rebel_servo_relay::Measured;
using rebel_servo_relay::RelayCore;
using rebel_servo_relay::RelayParams;

class ServoTrajectoryRelay : public rclcpp::Node
{
public:
  ServoTrajectoryRelay() : Node("servo_trajectory_relay"), steady_clock_(RCL_STEADY_TIME)
  {
    const auto input_topic = declare_parameter<std::string>("input_topic", "servo_trajectory_out");
    const auto output_topic = declare_parameter<std::string>("output_topic", "joint_trajectory");
    const auto joint_states_topic = declare_parameter<std::string>("joint_states_topic", "/joint_states");
    rsp_node_ = declare_parameter<std::string>("robot_description_node", "/robot_state_publisher");
    const auto urdf_param = declare_parameter<std::string>("robot_description", "");

    RelayParams p;
    p.publish_period_s = declare_parameter<double>("publish_period_s", p.publish_period_s);
    p.reset_timeout_s = declare_parameter<double>("reset_timeout_s", p.reset_timeout_s);
    p.horizon_s = declare_parameter<double>("horizon_s", p.horizon_s);
    p.stop_time_s = declare_parameter<double>("stop_time_s", p.stop_time_s);
    p.max_lead_rad = declare_parameter<double>("max_lead_rad", p.max_lead_rad);
    p.lead_time_s = declare_parameter<double>("lead_time_s", p.lead_time_s);
    p.lead_floor_rad = declare_parameter<double>("lead_floor_rad", p.lead_floor_rad);
    p.joint_limit_margin_rad = declare_parameter<double>("joint_limit_margin", p.joint_limit_margin_rad);
    p.joint_state_timeout_s = declare_parameter<double>("joint_state_timeout_s", p.joint_state_timeout_s);
    p.silence_hold_s = declare_parameter<double>("silence_hold_s", p.silence_hold_s);
    p.zero_velocity_eps = declare_parameter<double>("zero_velocity_eps", p.zero_velocity_eps);
    // Caps on the forwarded commands (defaults = igus_rebel_moveit_config joint_limits.yaml).
    p.max_joint_velocity_rad_s = declare_parameter<double>("max_joint_velocity_rad_s", p.max_joint_velocity_rad_s);
    p.max_joint_accel_rad_s2 = declare_parameter<double>("max_joint_accel_rad_s2", p.max_joint_accel_rad_s2);
    // Used only when the middleware gives no message receipt time (see receivedBeforeLastTrip).
    post_trip_quarantine_s_ = declare_parameter<double>("post_trip_quarantine_s", post_trip_quarantine_s_);
    if (!std::isfinite(post_trip_quarantine_s_) || post_trip_quarantine_s_ < 0.0)
      throw std::invalid_argument("post_trip_quarantine_s must be >= 0");
    core_ = std::make_unique<RelayCore>(p);  // throws on invalid parameters: refuse to start
    horizon_ = core_->params().horizon_s;
    stop_time_ = core_->params().stop_time_s;

    require_stamped_ = declare_parameter<bool>("require_stamped_joint_states", true);

    pub_ = create_publisher<trajectory_msgs::msg::JointTrajectory>(output_topic, rclcpp::SystemDefaultsQoS());
    // Depth 1 on both inputs: the executor can run the watchdog timer before it takes the messages that are already
    // queued, and a deep queue would then replay OLD Servo messages (and re-start the arm) right after the watchdog
    // hold. With KeepLast(1) at most the newest message survives; onServo additionally drops a message that was
    // received before the last watchdog hold (see below). Servo stamps its trajectories 0, so the stamp cannot be used.
    js_sub_ = create_subscription<sensor_msgs::msg::JointState>(
        joint_states_topic, rclcpp::QoS(rclcpp::KeepLast(1)).best_effort(),
        [this](sensor_msgs::msg::JointState::ConstSharedPtr m) { onJointState(*m); });

    if (!urdf_param.empty())
    {
      if (!parseUrdf(urdf_param))
        throw std::runtime_error("robot_description parameter could not be parsed");
    }
    else
    {
      params_client_ = std::make_shared<rclcpp::AsyncParametersClient>(this, rsp_node_);
      limits_timer_ = create_wall_timer(std::chrono::seconds(1), [this]() { requestLimits(); });
    }
    sub_ = create_subscription<trajectory_msgs::msg::JointTrajectory>(
        input_topic, rclcpp::QoS(rclcpp::KeepLast(1)).reliable(),
        [this](trajectory_msgs::msg::JointTrajectory::ConstSharedPtr m, const rclcpp::MessageInfo& info) {
          onServo(*m, info);
        });
    // Watchdog: Servo goes silent (stale commands, crash) without a halt message; hold instead of playing out the
    // previous segment. 10 ms resolution on a steady clock.
    watchdog_timer_ = create_wall_timer(std::chrono::milliseconds(10), [this]() { onWatchdog(); });

    RCLCPP_INFO(get_logger(),
                "relay %s -> %s (joint_states %s): publish_period %.3f s, horizon %.3f s, stop_time %.3f s, "
                "reset_timeout %.3f s, silence_hold %.3f s, max_lead %.3f rad, lead %.2f s x |v| + %.4f rad, "
                "limit margin %.4f rad, caps %.3f rad/s (effective %.3f) and %.2f rad/s^2",
                sub_->get_topic_name(), pub_->get_topic_name(), joint_states_topic.c_str(), p.publish_period_s,
                horizon_, stop_time_, core_->params().reset_timeout_s, core_->params().silence_hold_s, p.max_lead_rad,
                p.lead_time_s, p.lead_floor_rad, p.joint_limit_margin_rad, p.max_joint_velocity_rad_s,
                std::min(p.max_joint_velocity_rad_s, p.max_joint_accel_rad_s2 * stop_time_),
                p.max_joint_accel_rad_s2);
  }

private:
  struct Sample
  {
    double position = 0.0;
    double receipt_s = 0.0;  // steady clock
    rclcpp::Time stamp{ 0, 0, RCL_ROS_TIME };
    bool has_stamp = false;
  };

  static int64_t systemNs()
  {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch())
        .count();
  }

  double nowS() { return steady_clock_.now().seconds(); }

  void onJointState(const sensor_msgs::msg::JointState& msg)
  {
    const double now = nowS();
    const bool has_stamp = msg.header.stamp.sec != 0 || msg.header.stamp.nanosec != 0;
    const rclcpp::Time stamp(msg.header.stamp.sec, msg.header.stamp.nanosec, RCL_ROS_TIME);
    const std::size_t n = std::min(msg.name.size(), msg.position.size());
    for (std::size_t i = 0; i < n; ++i)
    {
      auto& s = state_[msg.name[i]];
      s.position = msg.position[i];
      s.receipt_s = now;
      s.stamp = stamp;
      s.has_stamp = has_stamp;
    }
  }

  // Measured positions of `names`; age = oldest of (receipt age, |ROS clock - header.stamp|); empty when a joint is missing.
  Measured buildMeasured(const std::vector<std::string>& names)
  {
    Measured m;
    m.position.resize(names.size());
    m.age_s = 0.0;
    m.unstamped = false;
    const double now = nowS();
    const rclcpp::Time ros_now = get_clock()->now();
    for (std::size_t i = 0; i < names.size(); ++i)
    {
      const auto st = state_.find(names[i]);
      if (st == state_.end())
      {
        m.position.clear();
        m.age_s = 1e9;
        return m;
      }
      m.position[i] = st->second.position;
      if (require_stamped_ && !st->second.has_stamp)
        m.unstamped = true;  // the broadcaster always stamps: an unstamped sample cannot be checked for freshness
      m.age_s = std::max(m.age_s, now - st->second.receipt_s);
      if (st->second.has_stamp)
        m.age_s = std::max(m.age_s, std::fabs((ros_now - st->second.stamp).seconds()));
    }
    return m;
  }

  void requestLimits()
  {
    if (have_limits_ || request_pending_ || !params_client_->service_is_ready())
      return;
    request_pending_ = true;
    params_client_->get_parameters({ "robot_description" },
                                   [this](std::shared_future<std::vector<rclcpp::Parameter>> fut) {
                                     request_pending_ = false;
                                     try
                                     {
                                       const auto params = fut.get();
                                       if (params.size() == 1 &&
                                           params[0].get_type() == rclcpp::ParameterType::PARAMETER_STRING &&
                                           parseUrdf(params[0].as_string()))
                                         limits_timer_->cancel();
                                     }
                                     catch (const std::exception& e)
                                     {
                                       RCLCPP_WARN(get_logger(), "robot_description request failed: %s", e.what());
                                     }
                                   });
  }

  bool parseUrdf(const std::string& xml)
  {
    urdf::Model model;
    if (xml.empty() || !model.initString(xml))
    {
      RCLCPP_ERROR(get_logger(), "could not parse robot_description");
      return false;
    }
    limits_.clear();
    for (const auto& [name, joint] : model.joints_)
    {
      JointLimit l;
      if (joint->type == urdf::Joint::REVOLUTE || joint->type == urdf::Joint::PRISMATIC)
      {
        if (!joint->limits)
          continue;  // revolute without limits is invalid URDF; leave it unknown -> rejected when used
        l.has_limits = true;
        l.lower = joint->limits->lower;
        l.upper = joint->limits->upper;
      }
      else if (joint->type != urdf::Joint::CONTINUOUS)
      {
        continue;  // fixed/floating/planar: not an actuated joint
      }
      limits_[name] = l;
    }
    have_limits_ = true;
    RCLCPP_INFO(get_logger(), "joint limits loaded (%zu joints)", limits_.size());
    return true;
  }

  void publishHold(const std::vector<std::string>& names, const std::vector<double>& position)
  {
    trajectory_msgs::msg::JointTrajectory out;
    out.joint_names = names;
    trajectory_msgs::msg::JointTrajectoryPoint pt;
    pt.positions = position;
    pt.velocities.assign(position.size(), 0.0);
    out.points.push_back(pt);
    pub_->publish(out);
  }

  void onWatchdog()
  {
    if (!core_->active())
      return;
    const auto names = core_->names();
    const auto out = core_->checkSilence(nowS(), buildMeasured(names));
    if (out.action == Action::kHold)
    {
      ++watchdog_trips_;
      last_trip_system_ns_ = systemNs();
      last_trip_steady_s_ = nowS();
      publishHold(names, out.start);
      RCLCPP_INFO(get_logger(), "watchdog hold #%zu: %s", watchdog_trips_, out.reason.c_str());
    }
  }

  // A message that cannot be forwarded: while a reference is active, hold (measured position if fresh, else the
  // reference); never leave the JTC playing out the previous segment.
  void fault(const std::string& why)
  {
    auto& clock = *get_clock();
    const auto names = core_->names();
    const auto out = core_->holdNow(nowS(), buildMeasured(names), why);
    if (out.action == Action::kHold)
      publishHold(names, out.start);
    RCLCPP_ERROR_THROTTLE(get_logger(), clock, 2000, "not forwarding Servo message (%s)%s", why.c_str(),
                          out.action == Action::kHold ? ", holding" : "");
  }

  // True if this message may have been received before the last watchdog hold (it sat in a queue while the timer ran)
  // and must be dropped. Fail closed: with a usable middleware receipt time the check is exact; without one (0 =
  // unsupported rmw, or a clock we cannot identify) every message dispatched within post_trip_quarantine_s of the hold
  // is dropped, because a stale queued message is dispatched right behind the timer callback and no callback-time stamp
  // can tell it from a fresh one. Before the first hold nothing can be stale.
  bool receivedBeforeLastTrip(const rclcpp::MessageInfo& info)
  {
    rebel_servo_relay::TripClocks trip;
    trip.seen = last_trip_system_ns_ != 0;
    trip.system_ns = last_trip_system_ns_;
    trip.steady_s = last_trip_steady_s_;
    bool used_rmw = false;
    double delay_s = 0.0;
    const bool stale = rebel_servo_relay::staleAfterTrip(trip, info.get_rmw_message_info().received_timestamp,
                                                         systemNs(), nowS(), post_trip_quarantine_s_, &used_rmw,
                                                         &delay_s);
    if (used_rmw && !rx_timestamp_logged_)
    {
      rx_timestamp_logged_ = true;
      RCLCPP_INFO(get_logger(), "message receipt timestamps available (delay of the first message %.1f ms)",
                  delay_s * 1e3);
    }
    else if (!used_rmw && trip.seen && !no_rx_timestamp_logged_)
    {
      no_rx_timestamp_logged_ = true;
      RCLCPP_WARN(get_logger(),
                  "no usable message receipt timestamps from the middleware: dropping Servo messages for %.0f ms "
                  "after each watchdog hold instead",
                  post_trip_quarantine_s_ * 1e3);
    }
    return stale;
  }

  void onServo(const trajectory_msgs::msg::JointTrajectory& msg, const rclcpp::MessageInfo& info)
  {
    auto& clock = *get_clock();
    if (receivedBeforeLastTrip(info))
    {
      ++dropped_stale_;
      RCLCPP_WARN_THROTTLE(get_logger(), clock, 2000, "dropped a Servo message received before the watchdog hold (%zu so far)",
                           dropped_stale_);
      return;
    }
    const std::size_t n = msg.joint_names.size();
    if (n == 0 || msg.points.empty())
      return fault("empty joint_names or points");
    const auto& pt = msg.points[0];
    if (pt.velocities.size() != n)
      return fault("point has " + std::to_string(pt.velocities.size()) + " velocities for " + std::to_string(n) +
                   " joints (Servo needs publish_joint_velocities: true)");
    std::unordered_set<std::string> seen(msg.joint_names.begin(), msg.joint_names.end());
    if (seen.size() != n)
      return fault("duplicate joint names");
    if (!have_limits_)
      return fault("joint limits not loaded yet (waiting for " + rsp_node_ + " robot_description)");

    std::vector<JointLimit> limits(n);
    for (std::size_t i = 0; i < n; ++i)
    {
      const auto lim = limits_.find(msg.joint_names[i]);
      if (lim == limits_.end())
        return fault("joint " + msg.joint_names[i] + " is not an actuated joint of robot_description");
      limits[i] = lim->second;
    }

    const auto prev_names = core_->names();
    const auto out = core_->process(nowS(), msg.joint_names, pt.velocities, buildMeasured(msg.joint_names), limits);
    switch (out.action)
    {
      case Action::kNone:
        return;
      case Action::kReject:
        RCLCPP_ERROR_THROTTLE(get_logger(), clock, 2000, "not forwarding Servo message: %s", out.reason.c_str());
        return;
      case Action::kHold:
        if (out.fault)
        {
          RCLCPP_ERROR_THROTTLE(get_logger(), clock, 2000, "not forwarding Servo message (%s), holding",
                                out.reason.c_str());
          publishHold(prev_names, out.start);  // the hold belongs to the reference that was active
        }
        else
        {
          publishHold(msg.joint_names, out.start);
        }
        return;
      case Action::kForward:
      {
        if (out.velocity_clamped || out.accel_clamped)
          RCLCPP_WARN_THROTTLE(get_logger(), clock, 2000, "limiting forwarded command: %s%s%s",
                               out.velocity_clamped ? "velocity above the cap" : "",
                               out.velocity_clamped && out.accel_clamped ? " and " : "",
                               out.accel_clamped ? "acceleration above the cap" : "");
        trajectory_msgs::msg::JointTrajectory traj;  // header.stamp stays 0: the JTC starts it at its next update
        traj.joint_names = msg.joint_names;
        trajectory_msgs::msg::JointTrajectoryPoint a, b, c;
        a.positions = out.start;
        a.velocities = out.velocity;
        b.positions = out.mid;
        b.velocities = out.velocity_mid;
        b.time_from_start = rclcpp::Duration::from_seconds(horizon_);
        c.positions = out.end;
        c.velocities.assign(n, 0.0);  // every trajectory ends at rest: a dead relay/Servo leaves the JTC holding
        c.time_from_start = rclcpp::Duration::from_seconds(horizon_ + stop_time_);
        traj.points = { a, b, c };
        pub_->publish(traj);
        return;
      }
    }
  }

  rclcpp::Clock steady_clock_;
  std::unique_ptr<RelayCore> core_;
  double horizon_ = 0.0;
  double stop_time_ = 0.0;
  std::size_t watchdog_trips_ = 0;
  std::size_t dropped_stale_ = 0;
  int64_t last_trip_system_ns_ = 0;
  double last_trip_steady_s_ = 0.0;
  bool rx_timestamp_logged_ = false;
  bool no_rx_timestamp_logged_ = false;
  double post_trip_quarantine_s_ = 0.1;
  bool require_stamped_ = true;
  std::string rsp_node_;
  std::unordered_map<std::string, Sample> state_;
  std::unordered_map<std::string, JointLimit> limits_;
  bool have_limits_ = false;
  bool request_pending_ = false;
  std::shared_ptr<rclcpp::AsyncParametersClient> params_client_;
  rclcpp::TimerBase::SharedPtr limits_timer_;
  rclcpp::TimerBase::SharedPtr watchdog_timer_;
  rclcpp::Subscription<trajectory_msgs::msg::JointTrajectory>::SharedPtr sub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr js_sub_;
  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr pub_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  try
  {
    rclcpp::spin(std::make_shared<ServoTrajectoryRelay>());
  }
  catch (const std::exception& e)
  {
    RCLCPP_FATAL(rclcpp::get_logger("servo_trajectory_relay"), "refusing to start: %s", e.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
