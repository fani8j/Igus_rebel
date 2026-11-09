// Pass-through online_signal_smoothing plugin for MoveIt Servo 2.5.10 (select with the Servo parameter
// smoothing_filter_plugin_name: "rebel_servo_relay::PassThroughFilterPlugin").
//
// Servo applies the plugin to (measured position + v * publish_period) and then publishes
// velocity = (smoothed - measured) / publish_period. The default Butterworth filter lags that ramp by
// 0.75 samples, which costs a quarter or more of the commanded speed; doing nothing keeps velocity == v.
#include <moveit/online_signal_smoothing/smoothing_base_class.h>
#include <pluginlib/class_list_macros.hpp>

namespace rebel_servo_relay
{
class PassThroughFilterPlugin : public online_signal_smoothing::SmoothingBaseClass
{
public:
  bool initialize(rclcpp::Node::SharedPtr /*node*/, moveit::core::RobotModelConstPtr /*robot_model*/,
                  size_t /*num_joints*/) override
  {
    return true;
  }

  bool doSmoothing(std::vector<double>& /*position_vector*/) override
  {
    return true;
  }

  bool reset(const std::vector<double>& /*joint_positions*/) override
  {
    return true;
  }
};
}  // namespace rebel_servo_relay

PLUGINLIB_EXPORT_CLASS(rebel_servo_relay::PassThroughFilterPlugin, online_signal_smoothing::SmoothingBaseClass)
