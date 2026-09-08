#include <rclcpp/rclcpp.hpp>

#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <moveit/move_group_interface/move_group_interface.h>

#include <chrono>
#include <memory>
#include <string>

using namespace std::chrono_literals;

class PointTargetMoveIt : public rclcpp::Node
{
public:
  PointTargetMoveIt()
  : Node("point_target_moveit"),
    tf_buffer_(this->get_clock()),
    tf_listener_(tf_buffer_)
  {
    declare_parameter<std::string>("planning_group", "rebel_arm");
    declare_parameter<std::string>("planning_frame", "base_link");
    declare_parameter<std::string>("tcp_link", "xeg32_tool_tip");
    declare_parameter<double>("approach_offset_z", 0.05);
    declare_parameter<double>("velocity_scaling", 0.05);
    declare_parameter<double>("acceleration_scaling", 0.10);
    declare_parameter<bool>("execute", false);

    planning_group_ = get_parameter("planning_group").as_string();
    planning_frame_ = get_parameter("planning_frame").as_string();
    tcp_link_ = get_parameter("tcp_link").as_string();
    approach_offset_z_ = get_parameter("approach_offset_z").as_double();
    velocity_scaling_ = get_parameter("velocity_scaling").as_double();
    acceleration_scaling_ = get_parameter("acceleration_scaling").as_double();
    execute_ = get_parameter("execute").as_bool();

    target_sub_ =
      create_subscription<geometry_msgs::msg::PointStamped>(
        "/cut_target",
        10,
        std::bind(
          &PointTargetMoveIt::targetCallback,
          this,
          std::placeholders::_1));

    // Delay MoveIt initialization until the node is already owned by shared_ptr.
    init_timer_ = create_wall_timer(
      500ms,
      std::bind(&PointTargetMoveIt::initializeMoveIt, this));

    RCLCPP_INFO(get_logger(), "point_target_moveit started.");
    RCLCPP_INFO(get_logger(), "Planning group parameter: %s", planning_group_.c_str());
    RCLCPP_INFO(get_logger(), "Target planning frame: %s", planning_frame_.c_str());
    RCLCPP_INFO(get_logger(), "TCP link: %s", tcp_link_.c_str());
    RCLCPP_INFO(
      get_logger(),
      "Execution: %s",
      execute_ ? "ENABLED" : "DISABLED (planning only)");
  }

private:
  void initializeMoveIt()
  {
    if (move_group_) {
      return;
    }

    init_timer_->cancel();

    try {
      move_group_ =
        std::make_shared<moveit::planning_interface::MoveGroupInterface>(
          shared_from_this(),
          planning_group_);

      move_group_->setMaxVelocityScalingFactor(velocity_scaling_);
      move_group_->setMaxAccelerationScalingFactor(acceleration_scaling_);

      RCLCPP_INFO(
        get_logger(),
        "MoveIt initialized. Planning frame reported by MoveIt: %s",
        move_group_->getPlanningFrame().c_str());

      RCLCPP_INFO(
        get_logger(),
        "MoveIt default end-effector link: %s",
        move_group_->getEndEffectorLink().c_str());
    }
    catch (const std::exception & e) {
      RCLCPP_FATAL(
        get_logger(),
        "Failed to initialize MoveGroupInterface: %s",
        e.what());
      rclcpp::shutdown();
    }
  }

  void targetCallback(
    const geometry_msgs::msg::PointStamped::SharedPtr msg)
  {
    if (!move_group_) {
      RCLCPP_WARN(get_logger(), "MoveIt is not initialized yet.");
      return;
    }

    RCLCPP_INFO(
      get_logger(),
      "Received target in '%s': [%.4f, %.4f, %.4f]",
      msg->header.frame_id.c_str(),
      msg->point.x,
      msg->point.y,
      msg->point.z);

    geometry_msgs::msg::PointStamped point_base;

    try {
      point_base =
        tf_buffer_.transform(
          *msg,
          planning_frame_,
          tf2::durationFromSec(1.0));
    }
    catch (const tf2::TransformException & ex) {
      RCLCPP_ERROR(
        get_logger(),
        "TF transform to '%s' failed: %s",
        planning_frame_.c_str(),
        ex.what());
      return;
    }

    RCLCPP_INFO(
      get_logger(),
      "Target in '%s': [%.4f, %.4f, %.4f]",
      planning_frame_.c_str(),
      point_base.point.x,
      point_base.point.y,
      point_base.point.z);

    geometry_msgs::msg::PoseStamped target_pose;
    target_pose.header.frame_id = planning_frame_;
    target_pose.header.stamp = now();

    target_pose.pose.position.x = point_base.point.x;
    target_pose.pose.position.y = point_base.point.y;
    target_pose.pose.position.z =
      point_base.point.z + approach_offset_z_;

    // Temporary fixed orientation for initial testing.
    // Replace this quaternion with the correct knife/tool orientation
    // for your cutting task.
    target_pose.pose.orientation.x = 0.0;
    target_pose.pose.orientation.y = 1.0;
    target_pose.pose.orientation.z = 0.0;
    target_pose.pose.orientation.w = 0.0;

    move_group_->setStartStateToCurrentState();
    move_group_->clearPoseTargets();

    const bool target_set =
      move_group_->setPoseTarget(target_pose, tcp_link_);

    if (!target_set) {
      RCLCPP_ERROR(
        get_logger(),
        "MoveIt rejected the target pose for TCP link '%s'.",
        tcp_link_.c_str());
      return;
    }

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    const bool success =
      static_cast<bool>(move_group_->plan(plan));

    if (!success) {
      RCLCPP_ERROR(get_logger(), "MoveIt planning failed.");
      move_group_->clearPoseTargets();
      return;
    }

    RCLCPP_INFO(
      get_logger(),
      "Planning succeeded. Approach offset Z = %.3f m",
      approach_offset_z_);

    if (execute_) {
      RCLCPP_WARN(
        get_logger(),
        "EXECUTION ENABLED: sending planned trajectory to the robot/controllers.");

      const bool executed =
        static_cast<bool>(move_group_->execute(plan));

      if (!executed) {
        RCLCPP_ERROR(get_logger(), "Trajectory execution failed.");
      } else {
        RCLCPP_INFO(get_logger(), "Trajectory execution completed.");
      }
    } else {
      RCLCPP_INFO(
        get_logger(),
        "Execution disabled. Set parameter execute:=true only after verifying the plan.");
    }

    move_group_->clearPoseTargets();
  }

  rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr target_sub_;
  rclcpp::TimerBase::SharedPtr init_timer_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  std::shared_ptr<
    moveit::planning_interface::MoveGroupInterface> move_group_;

  std::string planning_group_;
  std::string planning_frame_;
  std::string tcp_link_;

  double approach_offset_z_;
  double velocity_scaling_;
  double acceleration_scaling_;
  bool execute_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<PointTargetMoveIt>();

  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  executor.spin();

  rclcpp::shutdown();
  return 0;
}
