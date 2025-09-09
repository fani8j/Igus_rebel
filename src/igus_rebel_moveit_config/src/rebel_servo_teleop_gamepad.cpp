#include <chrono>
#include <control_msgs/msg/joint_jog.hpp>
#include <sensor_msgs/msg/joy.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <moveit_msgs/srv/servo_command_type.hpp>
#include <rclcpp/rclcpp.hpp>
#include <signal.h>
#include <stdio.h>

#define MAX_VEL 0.02

// Some constants used in the Servo Teleop demo
namespace
{
    const std::string TWIST_TOPIC = "delta_twist_cmds";
    const std::string JOINT_TOPIC = "delta_joint_cmds";
    const size_t ROS_QUEUE_SIZE = 10;
    const std::string PLANNING_FRAME_ID = "base_link";
    const std::string EE_FRAME_ID = "link6";
} // namespace

// Converts key-presses to Twist or Jog commands for Servo, in lieu of a controller
class JoyServo
{
public:
    JoyServo();
    void spin();

private:
    void initializeParams();
    void joyCallback(const sensor_msgs::msg::Joy::SharedPtr msg);

    rclcpp::Node::SharedPtr nh_;

    rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr twist_pub_;
    rclcpp::Publisher<control_msgs::msg::JointJog>::SharedPtr joint_pub_;
    rclcpp::Client<moveit_msgs::srv::ServoCommandType>::SharedPtr switch_input_;
    rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr joy_sub_;

    std::shared_ptr<moveit_msgs::srv::ServoCommandType::Request> request_;
    double joint_vel_cmd_;
    std::string command_frame_id_;
    bool status_enable_ = false;

    // Axis and button mappings
    int axis_linear_x_, axis_linear_y_, axis_linear_z_, button_linear_z_down_;
    int axis_angular_x_, axis_angular_y_, axis_angular_z_, button_angular_z_right_;
    int button_enable_;
};

JoyServo::JoyServo() : joint_vel_cmd_(1.0), command_frame_id_{"base_link"}
{
    nh_ = rclcpp::Node::make_shared("servo_keyboard_input");

    twist_pub_ = nh_->create_publisher<geometry_msgs::msg::TwistStamped>(TWIST_TOPIC, ROS_QUEUE_SIZE);
    joint_pub_ = nh_->create_publisher<control_msgs::msg::JointJog>(JOINT_TOPIC, ROS_QUEUE_SIZE);
    joy_sub_ = nh_->create_subscription<sensor_msgs::msg::Joy>(
        "joy", ROS_QUEUE_SIZE, std::bind(&JoyServo::joyCallback, this, std::placeholders::_1));

    // Client for switching input types
    switch_input_ = nh_->create_client<moveit_msgs::srv::ServoCommandType>("servo_node/switch_command_type");
    initializeParams();

    // Wait to enable twist mode
    RCLCPP_INFO_STREAM(nh_->get_logger(), "Waiting for servoing...");
    request_ = std::make_shared<moveit_msgs::srv::ServoCommandType::Request>();
    request_->command_type = moveit_msgs::srv::ServoCommandType::Request::TWIST;
    switch_input_->wait_for_service();
    switch_input_->async_send_request(request_);

    RCLCPP_INFO_STREAM(nh_->get_logger(), "Servoing enabled. Ready to send commands.");
}

void JoyServo::initializeParams()
{
    // Initialize axis and button mappings defaults
    axis_linear_x_ = 1;
    axis_linear_y_ = 0;
    axis_linear_z_ = 7;

    axis_angular_x_ = 2;
    axis_angular_y_ = 3;
    axis_angular_z_ = 6;

    button_enable_ = 7;

    // Get parameters from the parameter server, if they exist
    nh_->declare_parameter("axis_linear_x", axis_linear_x_);
    nh_->declare_parameter("axis_linear_y", axis_linear_y_);
    nh_->declare_parameter("axis_linear_z", axis_linear_z_);
    nh_->declare_parameter("axis_angular_x", axis_angular_x_);
    nh_->declare_parameter("axis_angular_y", axis_angular_y_);
    nh_->declare_parameter("axis_angular_z", axis_angular_z_);
    nh_->declare_parameter("button_enable", button_enable_);
    nh_->get_parameter("axis_linear_x", axis_linear_x_);
    nh_->get_parameter("axis_linear_y", axis_linear_y_);
    nh_->get_parameter("axis_linear_z", axis_linear_z_);
    nh_->get_parameter("axis_angular_x", axis_angular_x_);
    nh_->get_parameter("axis_angular_y", axis_angular_y_);
    nh_->get_parameter("axis_angular_z", axis_angular_z_);
    nh_->get_parameter("button_enable", button_enable_);
}

void JoyServo::joyCallback(const sensor_msgs::msg::Joy::SharedPtr msg)
{
    auto twist_msg = std::make_shared<geometry_msgs::msg::TwistStamped>();
    twist_msg->header.frame_id = command_frame_id_;
    twist_msg->header.stamp = nh_->now();
    twist_msg->twist.linear.x = 0.0;
    twist_msg->twist.linear.y = 0.0;
    twist_msg->twist.linear.z = 0.0;
    twist_msg->twist.angular.x = 0.0;
    twist_msg->twist.angular.y = 0.0;
    twist_msg->twist.angular.z = 0.0;

    try
    {
        // Enable button must be held to send commands
        if (msg->buttons[button_enable_])
        {
            twist_msg->twist.linear.x = MAX_VEL * msg->axes[axis_linear_x_];
            twist_msg->twist.linear.y = MAX_VEL * msg->axes[axis_linear_y_];
            twist_msg->twist.linear.z = MAX_VEL * msg->axes[axis_linear_z_];
            twist_msg->twist.angular.x = MAX_VEL * msg->axes[axis_angular_x_];
            twist_msg->twist.angular.y = MAX_VEL * msg->axes[axis_angular_y_];
            twist_msg->twist.angular.z = MAX_VEL * msg->axes[axis_angular_z_];
        }
        else
        {
            RCLCPP_DEBUG_STREAM(nh_->get_logger(), "Joystick disabled");
        }

        twist_pub_->publish(*twist_msg);
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
    }
}

void JoyServo::spin()
{
    while (rclcpp::ok())
    {
        rclcpp::spin_some(nh_);
    }
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    JoyServo joy_servo;
    joy_servo.spin();
    rclcpp::shutdown();

    return 0;
}
