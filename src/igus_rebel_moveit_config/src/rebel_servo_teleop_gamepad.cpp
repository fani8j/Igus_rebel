#include <chrono>
#include <exception>
#include <functional>
#include <memory>
#include <string>

#include <geometry_msgs/msg/twist_stamped.hpp>
#include <sensor_msgs/msg/joy.hpp>
#if __has_include(<moveit_msgs/srv/servo_command_type.hpp>)
#include <moveit_msgs/srv/servo_command_type.hpp>
#define IGUS_REBEL_HAS_SERVO_COMMAND_TYPE 1
#else
#define IGUS_REBEL_HAS_SERVO_COMMAND_TYPE 0
#include <std_srvs/srv/trigger.hpp>
#endif
#include <rclcpp/rclcpp.hpp>

#define MAX_LINEAR_VEL 0.10
#define MAX_ANGULAR_VEL 0.5

namespace
{
    const std::string TWIST_TOPIC = "delta_twist_cmds";
    const size_t ROS_QUEUE_SIZE = 10;
    constexpr auto SERVICE_WAIT_TIMEOUT = std::chrono::seconds(10);
    constexpr auto SERVICE_RESPONSE_TIMEOUT = std::chrono::seconds(2);
} // namespace

// Converts joystick axes to Twist commands for MoveIt Servo while the enable button is held.
class JoyServo
{
public:
    JoyServo();
    void spin();

private:
    void initializeParams();
    void joyCallback(const sensor_msgs::msg::Joy::ConstSharedPtr msg);
    template <typename ServiceT>
    bool callStartupService(const typename rclcpp::Client<ServiceT>::SharedPtr& client,
                            const typename ServiceT::Request::SharedPtr& request, const char* description);

    rclcpp::Node::SharedPtr nh_;

    rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr twist_pub_;
#if IGUS_REBEL_HAS_SERVO_COMMAND_TYPE
    rclcpp::Client<moveit_msgs::srv::ServoCommandType>::SharedPtr switch_input_;
#else
    rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr start_servo_;
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

    RCLCPP_INFO_STREAM(nh_->get_logger(), "Waiting for servoing...");
    bool servo_ready = false;
#if IGUS_REBEL_HAS_SERVO_COMMAND_TYPE
    // MoveIt 2.10 and newer require explicitly selecting the active command type.
    switch_input_ = nh_->create_client<moveit_msgs::srv::ServoCommandType>("servo_node/switch_command_type");
    auto request = std::make_shared<moveit_msgs::srv::ServoCommandType::Request>();
    request->command_type = moveit_msgs::srv::ServoCommandType::Request::TWIST;
    servo_ready = callStartupService<moveit_msgs::srv::ServoCommandType>(
        switch_input_, request, "servo_node/switch_command_type (TWIST)");
#else
    // Humble's servo_node stays idle until ~/start_servo is called.
    start_servo_ = nh_->create_client<std_srvs::srv::Trigger>("servo_node/start_servo");
    servo_ready = callStartupService<std_srvs::srv::Trigger>(
        start_servo_, std::make_shared<std_srvs::srv::Trigger::Request>(), "servo_node/start_servo");
#endif

    if (servo_ready)
    {
        RCLCPP_INFO_STREAM(nh_->get_logger(), "Servoing enabled. Ready to send commands.");
    }
    else
    {
        RCLCPP_WARN_STREAM(nh_->get_logger(),
                           "Servo was not confirmed ready; commands are published but Servo may ignore them");
    }
}

template <typename ServiceT>
bool JoyServo::callStartupService(const typename rclcpp::Client<ServiceT>::SharedPtr& client,
                                  const typename ServiceT::Request::SharedPtr& request, const char* description)
{
    if (!client->wait_for_service(SERVICE_WAIT_TIMEOUT))
    {
        RCLCPP_WARN(nh_->get_logger(), "%s not available after %ld s", description,
                    static_cast<long>(SERVICE_WAIT_TIMEOUT.count()));
        return false;
    }
    auto future = client->async_send_request(request);
    // The node is not spinning yet, so spin it until the response arrives.
    if (rclcpp::spin_until_future_complete(nh_, future, SERVICE_RESPONSE_TIMEOUT) !=
        rclcpp::FutureReturnCode::SUCCESS)
    {
        client->remove_pending_request(future);
        RCLCPP_WARN(nh_->get_logger(), "Timed out waiting for %s response", description);
        return false;
    }
    const auto response = future.get();
    if (!response->success)
    {
        RCLCPP_WARN(nh_->get_logger(), "%s reported failure", description);
        return false;
    }
    RCLCPP_INFO(nh_->get_logger(), "%s succeeded", description);
    return true;
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

void JoyServo::joyCallback(const sensor_msgs::msg::Joy::ConstSharedPtr msg)
{
    // Value-initialized on the stack (all twist components zero); publish() copies it.
    geometry_msgs::msg::TwistStamped twist_msg;
    twist_msg.header.frame_id = command_frame_id_;
    twist_msg.header.stamp = nh_->now();

    try
    {
        // Enable button must be held to send commands
        if (msg->buttons.at(button_enable_))
        {
            twist_msg.twist.linear.x = MAX_LINEAR_VEL * msg->axes.at(axis_linear_x_);
            twist_msg.twist.linear.y = MAX_LINEAR_VEL * msg->axes.at(axis_linear_y_);
            twist_msg.twist.linear.z = MAX_LINEAR_VEL * msg->axes.at(axis_linear_z_);
            twist_msg.twist.angular.x = MAX_ANGULAR_VEL * msg->axes.at(axis_angular_x_);
            twist_msg.twist.angular.y = MAX_ANGULAR_VEL * msg->axes.at(axis_angular_y_);
            twist_msg.twist.angular.z = MAX_ANGULAR_VEL * msg->axes.at(axis_angular_z_);
        }
        else
        {
            RCLCPP_DEBUG_STREAM(nh_->get_logger(), "Joystick disabled");
        }

        twist_pub_->publish(twist_msg);
    }
    catch (const std::exception &e)
    {
        RCLCPP_ERROR_THROTTLE(nh_->get_logger(), *nh_->get_clock(), 5000,
                              "Joy message does not match the configured axis/button mapping: %s", e.what());
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
