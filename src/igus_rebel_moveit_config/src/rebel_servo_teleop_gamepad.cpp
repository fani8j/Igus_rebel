#include <atomic>
#include <chrono>
#include <csignal>
#include <cmath>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

#include <geometry_msgs/msg/twist_stamped.hpp>
#include <igus_rebel_msgs/msg/hardware_state.hpp>
#include <sensor_msgs/msg/joy.hpp>
#if __has_include(<moveit_msgs/srv/servo_command_type.hpp>)
#include <moveit_msgs/srv/servo_command_type.hpp>
#define IGUS_REBEL_HAS_SERVO_COMMAND_TYPE 1
#else
#define IGUS_REBEL_HAS_SERVO_COMMAND_TYPE 0
#include <std_srvs/srv/trigger.hpp>
#endif
#include <std_msgs/msg/int8.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <rclcpp/rclcpp.hpp>

#define MAX_LINEAR_VEL 0.10
#define MAX_ANGULAR_VEL 0.5

namespace
{
    const std::string TWIST_TOPIC = "delta_twist_cmds";
    const size_t ROS_QUEUE_SIZE = 10;
    constexpr auto SERVICE_WAIT_TIMEOUT = std::chrono::seconds(10);
    constexpr auto SERVICE_RESPONSE_TIMEOUT = std::chrono::seconds(2);
    // cart_jog_enable can take ~3 s in the driver (waits for the controller to confirm the mode).
    constexpr auto CRI_SERVICE_RESPONSE_TIMEOUT = std::chrono::seconds(6);
    // Zero twists published after the enable button is released (the driver's own 0.15 s command
    // watchdog zeroes the jog as well; the extra zeros cover a lost best-effort sample).
    constexpr int CRI_ZERO_TICKS_AFTER_RELEASE = 5;
    constexpr double CRI_MIN_PUBLISH_RATE_HZ = 20.0;

    std::atomic<bool> g_cri_stop{ false };
    void criSignalHandler(int)
    {
        g_cri_stop = true;
    }
} // namespace

// Converts joystick axes to Twist commands while the enable button is held.
//   output:=servo (default)  TwistStamped on delta_twist_cmds for MoveIt Servo (unchanged).
//   output:=cri              TwistStamped on the driver's cart_jog_cmd (controller-side Cartesian
//                            jog, no MoveIt/Servo): see Igus::Rebel cart_jog_enable.
class JoyServo
{
public:
    JoyServo();
    void spin();

private:
    void initializeParams();
    void joyCallback(const sensor_msgs::msg::Joy::ConstSharedPtr msg);
    void joyCallbackCri(const sensor_msgs::msg::Joy::ConstSharedPtr msg);
    // CRI route
    void startCri();
    void criTimerCallback();
    void criStateCallback(const igus_rebel_msgs::msg::HardwareState::ConstSharedPtr msg);
    void publishCri(const geometry_msgs::msg::Twist& twist);
    // Blocking enable/disable of the driver's Cartesian jog mode (spins nh_ while waiting).
    bool callCriEnable(bool enable, bool wait_for_service);
    void requestCriToggle();
    void shutdownCri();
    template <typename ServiceT>
    bool callStartupService(const typename rclcpp::Client<ServiceT>::SharedPtr& client,
                            const typename ServiceT::Request::SharedPtr& request, const char* description);

    rclcpp::Node::SharedPtr nh_;

    rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr twist_pub_;
#if IGUS_REBEL_HAS_SERVO_COMMAND_TYPE
    rclcpp::Client<moveit_msgs::srv::ServoCommandType>::SharedPtr switch_input_;
#else
    rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr start_servo_;
    rclcpp::Subscription<std_msgs::msg::Int8>::SharedPtr servo_status_;
    bool servo_active_{ false };
#endif
    rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr joy_sub_;
    std::string command_frame_id_{"base_link"};

    // ---- CRI route (output:=cri) ----
    bool cri_ = false;
    std::string output_{ "servo" };
    std::string cri_cmd_topic_{ "/igus_rebel/cart_jog_cmd" };
    std::string cri_enable_service_{ "/igus_rebel/cart_jog_enable" };
    double cri_publish_rate_hz_ = 50.0;
    bool cri_auto_enable_ = true;
    int button_cri_toggle_ = 6;
    double cri_max_linear_vel_ = 0.05;
    double cri_max_angular_vel_ = 0.25;
    // Joy gap that counts as "joy lost" while the enable button is held. Must stay below the driver's
    // 0.15 s cart_jog_cmd watchdog's order of magnitude: the timer re-publishes the last twist until
    // then. joy_node autorepeat 20 Hz gives a normal gap of 50 ms.
    double joy_timeout_s_ = 0.15;
    std::string cri_state_topic_{ "/igus_rebel/hardware_state" };
    rclcpp::Subscription<igus_rebel_msgs::msg::HardwareState>::SharedPtr cri_state_sub_;
    rclcpp::Time cri_on_set_time_;
    rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr cri_pub_;
    rclcpp::Client<std_srvs::srv::SetBool>::SharedPtr cri_enable_client_;
    rclcpp::TimerBase::SharedPtr cri_timer_;
    std::mutex cri_mutex_;
    geometry_msgs::msg::Twist cri_twist_;       // latest command while the enable button is held
    bool cri_held_ = false;                     // enable button currently held
    bool cri_was_held_ = false;
    int cri_zero_ticks_ = 0;                    // zero twists still to publish after a release
    bool cri_on_ = false;                       // the driver confirmed the mode
    bool cri_toggle_was_pressed_ = false;
    bool cri_call_pending_ = false;
    rclcpp::Time cri_last_joy_;
    bool cri_joy_timed_out_ = false;

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

    if (cri_)
    {
        startCri();
        return;
    }

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
    servo_status_ = nh_->create_subscription<std_msgs::msg::Int8>(
        "/status", 10,
        [this](std_msgs::msg::Int8::ConstSharedPtr) { servo_active_ = true; });
    rclcpp::executors::SingleThreadedExecutor status_executor;
    status_executor.add_node(nh_);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!servo_active_ && std::chrono::steady_clock::now() < deadline)
        status_executor.spin_once(std::chrono::milliseconds(20));
    status_executor.remove_node(nh_);
    if (servo_active_)
        RCLCPP_INFO(nh_->get_logger(), "Servo already active; skipping start_servo");
    servo_ready = servo_active_ || callStartupService<std_srvs::srv::Trigger>(
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

    // Output route: servo (default, MoveIt Servo) or cri (controller-side Cartesian jog).
    output_ = nh_->declare_parameter("output", output_);
    if (output_ != "servo" && output_ != "cri")
    {
        throw std::runtime_error("parameter output must be 'servo' or 'cri', got '" + output_ + "'");
    }
    cri_ = output_ == "cri";
    cri_cmd_topic_ = nh_->declare_parameter("cri_cmd_topic", cri_cmd_topic_);
    cri_enable_service_ = nh_->declare_parameter("cri_enable_service", cri_enable_service_);
    cri_publish_rate_hz_ = nh_->declare_parameter("cri_publish_rate_hz", cri_publish_rate_hz_);
    cri_auto_enable_ = nh_->declare_parameter("cri_auto_enable", cri_auto_enable_);
    button_cri_toggle_ = nh_->declare_parameter("button_cri_toggle", button_cri_toggle_);
    cri_max_linear_vel_ = nh_->declare_parameter("cri_max_linear_vel", cri_max_linear_vel_);
    cri_max_angular_vel_ = nh_->declare_parameter("cri_max_angular_vel", cri_max_angular_vel_);
    joy_timeout_s_ = nh_->declare_parameter("joy_timeout_s", joy_timeout_s_);
    cri_state_topic_ = nh_->declare_parameter("cri_state_topic", cri_state_topic_);
    if (cri_ && !(cri_publish_rate_hz_ >= CRI_MIN_PUBLISH_RATE_HZ))
    {
        RCLCPP_WARN(nh_->get_logger(), "cri_publish_rate_hz %.1f is below %.0f Hz (the driver's 0.15 s command "
                                       "watchdog would stutter): using %.0f Hz",
                    cri_publish_rate_hz_, CRI_MIN_PUBLISH_RATE_HZ, CRI_MIN_PUBLISH_RATE_HZ);
        cri_publish_rate_hz_ = CRI_MIN_PUBLISH_RATE_HZ;
    }
}

// ---------------------------------------------------------------------------------------------
// CRI route: TwistStamped to the driver's cart_jog_cmd, Cartesian jog done by the controller.
// ---------------------------------------------------------------------------------------------
void JoyServo::startCri()
{
    // Own signal handling in this mode only (the servo route keeps rclcpp's): Ctrl+C must leave
    // the context valid long enough to turn the driver's Cartesian jog mode off again.
    std::signal(SIGINT, criSignalHandler);
    std::signal(SIGTERM, criSignalHandler);

    cri_last_joy_ = nh_->now();
    cri_on_set_time_ = nh_->now();
    // The driver is the truth about the mode (it can leave it on its own: latch, idle timeout).
    cri_state_sub_ = nh_->create_subscription<igus_rebel_msgs::msg::HardwareState>(
        cri_state_topic_, rclcpp::QoS(rclcpp::KeepLast(10)).reliable(),
        std::bind(&JoyServo::criStateCallback, this, std::placeholders::_1));
    cri_pub_ = nh_->create_publisher<geometry_msgs::msg::TwistStamped>(cri_cmd_topic_, 2);
    cri_enable_client_ = nh_->create_client<std_srvs::srv::SetBool>(cri_enable_service_);
    joy_sub_ = nh_->create_subscription<sensor_msgs::msg::Joy>(
        "joy", ROS_QUEUE_SIZE, std::bind(&JoyServo::joyCallback, this, std::placeholders::_1));
    cri_timer_ = nh_->create_wall_timer(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(1.0 / cri_publish_rate_hz_)),
        std::bind(&JoyServo::criTimerCallback, this));

    RCLCPP_WARN(nh_->get_logger(),
                "CRI route: publishing %s at %.0f Hz while button %d is held (no MoveIt/Servo). The "
                "driver refuses the mode while a trajectory controller is active.",
                cri_cmd_topic_.c_str(), cri_publish_rate_hz_, button_enable_);
    if (cri_auto_enable_)
    {
        if (!callCriEnable(true, true))
        {
            RCLCPP_ERROR(nh_->get_logger(),
                         "Cartesian jog mode was NOT enabled. Fix the cause (see the driver message above; the "
                         "trajectory controller must be deactivated) and press button %d to retry.",
                         button_cri_toggle_);
        }
    }
    else
    {
        RCLCPP_INFO(nh_->get_logger(), "cri_auto_enable is false: press button %d to enable the mode",
                    button_cri_toggle_);
    }
}

bool JoyServo::callCriEnable(bool enable, bool wait_for_service)
{
    if (wait_for_service && !cri_enable_client_->wait_for_service(SERVICE_WAIT_TIMEOUT))
    {
        RCLCPP_ERROR(nh_->get_logger(), "%s not available after %ld s", cri_enable_service_.c_str(),
                     static_cast<long>(SERVICE_WAIT_TIMEOUT.count()));
        return false;
    }
    if (!cri_enable_client_->service_is_ready())
    {
        RCLCPP_ERROR(nh_->get_logger(), "%s is not available", cri_enable_service_.c_str());
        return false;
    }
    auto request = std::make_shared<std_srvs::srv::SetBool::Request>();
    request->data = enable;
    auto future = cri_enable_client_->async_send_request(request);
    // Only called while the node is not being spun by another thread (start-up, shutdown) or from
    // the executor thread that runs the timer/subscription: spin_until_future_complete on a node
    // already owned by an executor is not allowed, so the in-run toggle uses the async path below.
    if (rclcpp::spin_until_future_complete(nh_, future, CRI_SERVICE_RESPONSE_TIMEOUT) !=
        rclcpp::FutureReturnCode::SUCCESS)
    {
        cri_enable_client_->remove_pending_request(future);
        RCLCPP_ERROR(nh_->get_logger(), "Timed out waiting for %s", cri_enable_service_.c_str());
        return false;
    }
    const auto response = future.get();
    if (!response->success)
    {
        RCLCPP_ERROR(nh_->get_logger(), "%s(%s) refused: %s", cri_enable_service_.c_str(), enable ? "true" : "false",
                     response->message.c_str());
        if (!enable)
        {
            std::lock_guard<std::mutex> lock(cri_mutex_);
            cri_on_ = false;
        }
        return false;
    }
    RCLCPP_WARN(nh_->get_logger(), "Cartesian jog mode %s: %s", enable ? "ON" : "OFF", response->message.c_str());
    std::lock_guard<std::mutex> lock(cri_mutex_);
    cri_on_ = enable;
    cri_on_set_time_ = nh_->now();
    return true;
}

void JoyServo::criStateCallback(const igus_rebel_msgs::msg::HardwareState::ConstSharedPtr msg)
{
    const bool active = msg->cart_jog_state == "active";
    std::lock_guard<std::mutex> lock(cri_mutex_);
    if (cri_call_pending_)
    {
        return;  // the answer of our own call decides
    }
    if (active)
    {
        cri_on_ = true;
    }
    else if (cri_on_ && (nh_->now() - cri_on_set_time_).seconds() > 0.5)
    {
        // A state sample older than the answer of the last call must not flip the belief.
        cri_on_ = false;
        RCLCPP_ERROR(nh_->get_logger(), "The driver left the Cartesian jog mode (state '%s', latched %d): "
                                        "nothing is published; press button %d to re-enable",
                     msg->cart_jog_state.c_str(), static_cast<int>(msg->latched), button_cri_toggle_);
    }
}

void JoyServo::requestCriToggle()
{
    // Called from the Joy callback (executor thread): asynchronous, never blocks the callbacks.
    bool enable;
    {
        std::lock_guard<std::mutex> lock(cri_mutex_);
        if (cri_call_pending_)
        {
            return;
        }
        cri_call_pending_ = true;
        enable = !cri_on_;
        cri_zero_ticks_ = CRI_ZERO_TICKS_AFTER_RELEASE;
    }
    if (!cri_enable_client_->service_is_ready())
    {
        RCLCPP_ERROR(nh_->get_logger(), "%s is not available", cri_enable_service_.c_str());
        std::lock_guard<std::mutex> lock(cri_mutex_);
        cri_call_pending_ = false;
        return;
    }
    auto request = std::make_shared<std_srvs::srv::SetBool::Request>();
    request->data = enable;
    RCLCPP_WARN(nh_->get_logger(), "Requesting Cartesian jog mode %s ...", enable ? "ON" : "OFF");
    cri_enable_client_->async_send_request(
        request, [this, enable](rclcpp::Client<std_srvs::srv::SetBool>::SharedFuture future) {
            const auto response = future.get();
            std::lock_guard<std::mutex> lock(cri_mutex_);
            cri_call_pending_ = false;
            if (response->success)
            {
                cri_on_ = enable;
                cri_on_set_time_ = nh_->now();
                RCLCPP_WARN(nh_->get_logger(), "Cartesian jog mode %s: %s", enable ? "ON" : "OFF",
                            response->message.c_str());
            }
            else
            {
                if (!enable)
                {
                    cri_on_ = false;
                }
                RCLCPP_ERROR(nh_->get_logger(), "cart_jog_enable(%s) refused: %s", enable ? "true" : "false",
                             response->message.c_str());
            }
        });
}

void JoyServo::publishCri(const geometry_msgs::msg::Twist& twist)
{
    geometry_msgs::msg::TwistStamped msg;
    msg.header.frame_id = command_frame_id_;
    msg.header.stamp = nh_->now();
    msg.twist = twist;
    cri_pub_->publish(msg);
}

void JoyServo::joyCallbackCri(const sensor_msgs::msg::Joy::ConstSharedPtr msg)
{
    try
    {
        const bool held = msg->buttons.at(button_enable_) != 0;
        geometry_msgs::msg::Twist twist;  // zero unless held
        if (held)
        {
            twist.linear.x = cri_max_linear_vel_ * msg->axes.at(axis_linear_x_);
            twist.linear.y = cri_max_linear_vel_ * msg->axes.at(axis_linear_y_);
            twist.linear.z = cri_max_linear_vel_ * msg->axes.at(axis_linear_z_);
            twist.angular.x = cri_max_angular_vel_ * msg->axes.at(axis_angular_x_);
            twist.angular.y = cri_max_angular_vel_ * msg->axes.at(axis_angular_y_);
            twist.angular.z = cri_max_angular_vel_ * msg->axes.at(axis_angular_z_);
        }
        bool toggle = false;
        if (button_cri_toggle_ >= 0)
        {
            if (static_cast<size_t>(button_cri_toggle_) < msg->buttons.size())
            {
                const bool pressed = msg->buttons[button_cri_toggle_] != 0;
                toggle = pressed && !cri_toggle_was_pressed_;  // rising edge only
                cri_toggle_was_pressed_ = pressed;
            }
            else
            {
                RCLCPP_WARN_THROTTLE(nh_->get_logger(), *nh_->get_clock(), 10000,
                                     "button_cri_toggle %d is outside the Joy message (%zu buttons)",
                                     button_cri_toggle_, msg->buttons.size());
            }
        }
        bool released = false;
        {
            std::lock_guard<std::mutex> lock(cri_mutex_);
            cri_last_joy_ = nh_->now();
            cri_joy_timed_out_ = false;
            cri_twist_ = twist;
            released = cri_was_held_ && !held;
            cri_held_ = held;
            cri_was_held_ = held;
            if (released)
            {
                cri_zero_ticks_ = CRI_ZERO_TICKS_AFTER_RELEASE;
            }
        }
        if (released)
        {
            // Zero at once, not at the next timer tick.
            publishCri(geometry_msgs::msg::Twist());
        }
        if (toggle)
        {
            requestCriToggle();
        }
    }
    catch (const std::exception& e)
    {
        RCLCPP_ERROR_THROTTLE(nh_->get_logger(), *nh_->get_clock(), 5000,
                              "Joy message does not match the configured axis/button mapping: %s", e.what());
    }
}

void JoyServo::criTimerCallback()
{
    geometry_msgs::msg::Twist twist;
    bool publish = false;
    {
        std::lock_guard<std::mutex> lock(cri_mutex_);
        // A dead joy_node must not keep the last stick position alive.
        if (cri_held_ && joy_timeout_s_ > 0.0 && (nh_->now() - cri_last_joy_).seconds() > joy_timeout_s_)
        {
            cri_held_ = false;
            cri_was_held_ = false;
            cri_twist_ = geometry_msgs::msg::Twist();
            cri_zero_ticks_ = CRI_ZERO_TICKS_AFTER_RELEASE;
            if (!cri_joy_timed_out_)
            {
                cri_joy_timed_out_ = true;
                RCLCPP_ERROR(nh_->get_logger(), "No Joy message for %.2f s while the enable button was held: "
                                                "sending zero twist", joy_timeout_s_);
            }
        }
        if (cri_held_)
        {
            twist = cri_twist_;
            publish = cri_on_;
            if (!cri_on_)
            {
                RCLCPP_WARN_THROTTLE(nh_->get_logger(), *nh_->get_clock(), 5000,
                                     "Cartesian jog mode is not on: nothing is published (press button %d)",
                                     button_cri_toggle_);
            }
        }
        else if (cri_zero_ticks_ > 0)
        {
            --cri_zero_ticks_;
            publish = true;  // zero twist
        }
    }
    if (publish)
    {
        publishCri(twist);
    }
}

void JoyServo::shutdownCri()
{
    // Zero twist first (a few, the topic is best effort), then leave the mode.
    if (cri_timer_)
    {
        cri_timer_->cancel();
    }
    for (int i = 0; i < 3; ++i)
    {
        publishCri(geometry_msgs::msg::Twist());
    }
    bool on;
    {
        std::lock_guard<std::mutex> lock(cri_mutex_);
        on = cri_on_;
        cri_held_ = false;
    }
    if (on || cri_auto_enable_)
    {
        // Idempotent in the driver; also covers a lost "enable" answer.
        callCriEnable(false, false);
    }
}

void JoyServo::joyCallback(const sensor_msgs::msg::Joy::ConstSharedPtr msg)
{
    if (cri_)
    {
        joyCallbackCri(msg);
        return;
    }
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
    if (!cri_)
    {
        rclcpp::spin(nh_);
        return;
    }
    // CRI route: spin until SIGINT/SIGTERM (handled here, so the context is still valid for the
    // final zero twist and cart_jog_enable(false)).
    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(nh_);
    while (rclcpp::ok() && !g_cri_stop)
    {
        executor.spin_once(std::chrono::milliseconds(20));
    }
    executor.remove_node(nh_);
    if (rclcpp::ok())
    {
        shutdownCri();
    }
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    try
    {
        JoyServo joy_servo;
        joy_servo.spin();
    }
    catch (const std::exception& e)
    {
        RCLCPP_FATAL(rclcpp::get_logger("servo_gamepad_input"), "%s", e.what());
        rclcpp::shutdown();
        return 1;
    }
    rclcpp::shutdown();

    return 0;
}
