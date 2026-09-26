#include <functional>
#include <iostream>
#include <memory>
#include <string>

#include <geometry_msgs/msg/twist_stamped.hpp>
#include <sensor_msgs/msg/joy.hpp>
#if __has_include(<moveit_msgs/srv/servo_command_type.hpp>)
#include <moveit_msgs/srv/servo_command_type.hpp>
#define IGUS_REBEL_HAS_SERVO_COMMAND_TYPE 1
#else
#define IGUS_REBEL_HAS_SERVO_COMMAND_TYPE 0
#endif
#include <rclcpp/rclcpp.hpp>

#define MAX_LINEAR_VEL 0.10
#define MAX_ANGULAR_VEL 0.5

namespace
{
    const std::string TWIST_TOPIC = "delta_twist_cmds";
    const size_t ROS_QUEUE_SIZE = 10;
} // namespace

// Converts joystick axes to Twist commands for MoveIt Servo while the enable button is held.
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
#if IGUS_REBEL_HAS_SERVO_COMMAND_TYPE
    rclcpp::Client<moveit_msgs::srv::ServoCommandType>::SharedPtr switch_input_;
    std::shared_ptr<moveit_msgs::srv::ServoCommandType::Request> request_;
#endif
    rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr joy_sub_;
    std::string command_frame_id_{"base_link"};

    // Axis and button mappings (overridden by config/gamepad.yaml)
    int axis_linear_x_ = 1;
    int axis_linear_y_ = 0;
    int axis_linear_z_ = 7;
    int axis_angular_x_ = 2;
    int axis_angular_y_ = 3;
    int axis_angular_z_ = 6;
    int button_enable_ = 7;
};

JoyServo::JoyServo()
{
    nh_ = rclcpp::Node::make_shared("servo_gamepad_input");
    initializeParams();

    twist_pub_ = nh_->create_publisher<geometry_msgs::msg::TwistStamped>(TWIST_TOPIC, ROS_QUEUE_SIZE);
    joy_sub_ = nh_->create_subscription<sensor_msgs::msg::Joy>(
        "joy", ROS_QUEUE_SIZE, std::bind(&JoyServo::joyCallback, this, std::placeholders::_1));

#if IGUS_REBEL_HAS_SERVO_COMMAND_TYPE
    // MoveIt 2.10 and newer require explicitly selecting the active command type.
    switch_input_ = nh_->create_client<moveit_msgs::srv::ServoCommandType>("servo_node/switch_command_type");

    RCLCPP_INFO_STREAM(nh_->get_logger(), "Waiting for servoing...");
    request_ = std::make_shared<moveit_msgs::srv::ServoCommandType::Request>();
    request_->command_type = moveit_msgs::srv::ServoCommandType::Request::TWIST;
    switch_input_->wait_for_service();
    switch_input_->async_send_request(request_);
#endif

    RCLCPP_INFO_STREAM(nh_->get_logger(), "Servoing enabled. Ready to send commands.");
}

void JoyServo::initializeParams()
{
    command_frame_id_ = nh_->declare_parameter("command_frame_id", command_frame_id_);
    axis_linear_x_ = nh_->declare_parameter("axis_linear_x", axis_linear_x_);
    axis_linear_y_ = nh_->declare_parameter("axis_linear_y", axis_linear_y_);
    axis_linear_z_ = nh_->declare_parameter("axis_linear_z", axis_linear_z_);
    axis_angular_x_ = nh_->declare_parameter("axis_angular_x", axis_angular_x_);
    axis_angular_y_ = nh_->declare_parameter("axis_angular_y", axis_angular_y_);
    axis_angular_z_ = nh_->declare_parameter("axis_angular_z", axis_angular_z_);
    button_enable_ = nh_->declare_parameter("button_enable", button_enable_);
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
        if (msg->buttons.at(button_enable_))
        {
            twist_msg->twist.linear.x = MAX_LINEAR_VEL * msg->axes.at(axis_linear_x_);
            twist_msg->twist.linear.y = MAX_LINEAR_VEL * msg->axes.at(axis_linear_y_);
            twist_msg->twist.linear.z = MAX_LINEAR_VEL * msg->axes.at(axis_linear_z_);
            twist_msg->twist.angular.x = MAX_ANGULAR_VEL * msg->axes.at(axis_angular_x_);
            twist_msg->twist.angular.y = MAX_ANGULAR_VEL * msg->axes.at(axis_angular_y_);
            twist_msg->twist.angular.z = MAX_ANGULAR_VEL * msg->axes.at(axis_angular_z_);
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
    rclcpp::spin(nh_);
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    JoyServo joy_servo;
    joy_servo.spin();
    rclcpp::shutdown();

    return 0;
}
