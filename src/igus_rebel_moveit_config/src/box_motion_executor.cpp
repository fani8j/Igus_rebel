#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/vector3_stamped.hpp>
#include <moveit/task_constructor/solvers/pipeline_planner.h>
#include <moveit/task_constructor/stages/current_state.h>
#include <moveit/task_constructor/stages/move_to.h>
#include <moveit/task_constructor/task.h>
#include <rclcpp/rclcpp.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <array>
#include <cmath>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

namespace mtc = moveit::task_constructor;

namespace
{
bool sameStamp(const builtin_interfaces::msg::Time& first, const builtin_interfaces::msg::Time& second)
{
  return first.sec == second.sec && first.nanosec == second.nanosec;
}

bool finitePoint(const geometry_msgs::msg::Point& point)
{
  return std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z);
}
}  // namespace

class BoxMotionExecutor : public rclcpp::Node
{
public:
  BoxMotionExecutor() : Node("box_motion_executor")
  {
    planning_group_ = declare_parameter<std::string>("planning_group", "igus_rebel_arm");
    tool_link_ = declare_parameter<std::string>("tool_link", "link6");
    planning_pipeline_ = declare_parameter<std::string>("planning_pipeline", "ompl");
    planner_id_ = declare_parameter<std::string>("planner_id", "");
    approach_offset_m_ = declare_parameter<double>("approach_offset_m", 0.10);
    max_detection_age_s_ = declare_parameter<double>("max_detection_age_s", 0.50);
    max_plane_rms_m_ = declare_parameter<double>("max_plane_rms_m", 0.005);
    max_solutions_ = declare_parameter<int>("max_solutions", 1);
    velocity_scaling_ = declare_parameter<double>("velocity_scaling", 0.05);
    acceleration_scaling_ = declare_parameter<double>("acceleration_scaling", 0.05);
    align_tool_z_to_surface_ = declare_parameter<bool>("align_tool_z_to_surface", true);
    execute_ = declare_parameter<bool>("execute", false);

    if (approach_offset_m_ <= 0.0 || max_detection_age_s_ <= 0.0 || max_plane_rms_m_ <= 0.0) {
      throw std::invalid_argument("approach offset, maximum age, and plane RMS must be positive");
    }
    if (max_solutions_ < 1) {
      throw std::invalid_argument("max_solutions must be positive");
    }
    if (velocity_scaling_ <= 0.0 || velocity_scaling_ > 1.0 ||
        acceleration_scaling_ <= 0.0 || acceleration_scaling_ > 1.0) {
      throw std::invalid_argument("velocity and acceleration scaling must be in (0, 1]");
    }
    if (execute_) {
      throw std::invalid_argument(
        "box_motion_executor is plan-only; physical execution is intentionally unavailable");
    }

    const auto qos = rclcpp::QoS(10);
    start_sub_ = create_subscription<geometry_msgs::msg::PointStamped>(
      "/box_seam/start", qos,
      [this](geometry_msgs::msg::PointStamped::SharedPtr message) {
        std::lock_guard<std::mutex> lock(data_mutex_);
        start_ = *message;
      });
    end_sub_ = create_subscription<geometry_msgs::msg::PointStamped>(
      "/box_seam/end", qos,
      [this](geometry_msgs::msg::PointStamped::SharedPtr message) {
        std::lock_guard<std::mutex> lock(data_mutex_);
        end_ = *message;
      });
    center_sub_ = create_subscription<geometry_msgs::msg::PointStamped>(
      "/box_seam/center", qos,
      [this](geometry_msgs::msg::PointStamped::SharedPtr message) {
        std::lock_guard<std::mutex> lock(data_mutex_);
        center_ = *message;
      });
    normal_sub_ = create_subscription<geometry_msgs::msg::Vector3Stamped>(
      "/box_seam/surface_normal", rclcpp::QoS(1).reliable().transient_local(),
      [this](geometry_msgs::msg::Vector3Stamped::SharedPtr message) {
        std::lock_guard<std::mutex> lock(data_mutex_);
        normal_ = *message;
      });
    diagnostics_sub_ = create_subscription<diagnostic_msgs::msg::DiagnosticArray>(
      "/box_detection/diagnostics", qos,
      [this](diagnostic_msgs::msg::DiagnosticArray::SharedPtr message) {
        std::lock_guard<std::mutex> lock(data_mutex_);
        tracked_ = false;
        plane_rms_m_ = std::nullopt;
        for (const auto& status : message->status) {
          if (status.name != "compal_box_perception/seam_detection") {
            continue;
          }
          tracked_ = status.level == diagnostic_msgs::msg::DiagnosticStatus::OK &&
                     status.message == "tracked seam";
          for (const auto& value : status.values) {
            if (value.key == "plane_rms_m") {
              try {
                plane_rms_m_ = std::stod(value.value);
              } catch (const std::exception&) {
                tracked_ = false;
              }
            }
          }
        }
      });

    marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
      "/box_motion/markers", rclcpp::QoS(1).reliable().transient_local());
    plan_service_ = create_service<std_srvs::srv::Trigger>(
      "~/plan_approach",
      [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
             std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
        std::unique_lock<std::mutex> plan_lock(plan_mutex_, std::try_to_lock);
        if (!plan_lock.owns_lock()) {
          response->success = false;
          response->message = "planning is already in progress";
          return;
        }
        response->success = planApproach(response->message);
      });
    clear_service_ = create_service<std_srvs::srv::Trigger>(
      "~/clear_plan",
      [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
             std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
        std::lock_guard<std::mutex> lock(plan_mutex_);
        planned_task_.reset();
        planned_stamp_.reset();
        response->success = true;
        response->message = "stored plan cleared";
      });

    RCLCPP_INFO(
      get_logger(),
      "Plan-only MTC executor ready; call ~/plan_approach after perception is tracked");
  }

private:
  struct Detection
  {
    geometry_msgs::msg::PointStamped start;
    geometry_msgs::msg::PointStamped end;
    geometry_msgs::msg::PointStamped center;
    geometry_msgs::msg::Vector3Stamped normal;
    double plane_rms_m;
  };

  std::optional<Detection> validDetection(std::string& error)
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    if (!start_ || !end_ || !center_ || !normal_) {
      error = "waiting for start, end, center, and surface normal";
      return std::nullopt;
    }
    if (!tracked_ || !plane_rms_m_) {
      error = "perception diagnostics are not tracked";
      return std::nullopt;
    }
    if (*plane_rms_m_ > max_plane_rms_m_) {
      error = "plane RMS exceeds configured limit";
      return std::nullopt;
    }
    if (start_->header.frame_id.empty() || start_->header.frame_id != end_->header.frame_id ||
        start_->header.frame_id != center_->header.frame_id ||
        start_->header.frame_id != normal_->header.frame_id) {
      error = "perception result frames do not match";
      return std::nullopt;
    }
    if (!sameStamp(start_->header.stamp, end_->header.stamp) ||
        !sameStamp(start_->header.stamp, center_->header.stamp) ||
        !sameStamp(start_->header.stamp, normal_->header.stamp)) {
      error = "perception result timestamps do not match";
      return std::nullopt;
    }
    if (!finitePoint(start_->point) || !finitePoint(end_->point) || !finitePoint(center_->point)) {
      error = "perception result contains non-finite points";
      return std::nullopt;
    }
    const rclcpp::Time detection_time(start_->header.stamp);
    const double age = (now() - detection_time).seconds();
    if (age < -0.1 || age > max_detection_age_s_) {
      error = "perception result is stale";
      return std::nullopt;
    }
    const auto& vector = normal_->vector;
    const double norm = std::sqrt(vector.x * vector.x + vector.y * vector.y + vector.z * vector.z);
    if (!std::isfinite(norm) || std::abs(norm - 1.0) > 0.05) {
      error = "surface normal is not a unit vector";
      return std::nullopt;
    }
    return Detection{*start_, *end_, *center_, *normal_, *plane_rms_m_};
  }

  bool planApproach(std::string& message)
  {
    const auto detection = validDetection(message);
    if (!detection) {
      RCLCPP_WARN(get_logger(), "Plan request rejected: %s", message.c_str());
      return false;
    }

    geometry_msgs::msg::PointStamped target;
    target.header = detection->center.header;
    target.point.x = detection->center.point.x + approach_offset_m_ * detection->normal.vector.x;
    target.point.y = detection->center.point.y + approach_offset_m_ * detection->normal.vector.y;
    target.point.z = detection->center.point.z + approach_offset_m_ * detection->normal.vector.z;
    publishMarkers(*detection, target);

    try {
      auto task = std::make_unique<mtc::Task>();
      task->stages()->setName("Box seam safe hover plan");
      task->loadRobotModel(shared_from_this());
      task->add(std::make_unique<mtc::stages::CurrentState>("current robot state"));

      auto planner = std::make_shared<mtc::solvers::PipelinePlanner>(
        shared_from_this(), planning_pipeline_);
      if (!planner_id_.empty()) {
        planner->setPlannerId(planner_id_);
      }
      planner->setMaxVelocityScalingFactor(velocity_scaling_);
      planner->setMaxAccelerationScalingFactor(acceleration_scaling_);

      auto move = std::make_unique<mtc::stages::MoveTo>("plan to safe hover", planner);
      move->setGroup(planning_group_);
      move->setIKFrame(tool_link_);
      if (align_tool_z_to_surface_) {
        move->setGoal(alignedTargetPose(*detection, target));
      } else {
        move->setGoal(target);
      }
      task->add(std::move(move));

      task->init();
      const auto result = task->plan(static_cast<std::size_t>(max_solutions_));
      if (result != moveit::core::MoveItErrorCode::SUCCESS || task->solutions().empty()) {
        task->explainFailure();
        message = "MTC failed to find a hover trajectory";
        RCLCPP_ERROR(get_logger(), "%s", message.c_str());
        return false;
      }
      task->introspection().publishSolution(*task->solutions().front());
      planned_stamp_ = detection->center.header.stamp;
      planned_task_ = std::move(task);
      message = "MTC hover trajectory planned and published to RViz; execution is disabled";
      RCLCPP_INFO(get_logger(), "%s", message.c_str());
      return true;
    } catch (const mtc::InitStageException& exception) {
      message = std::string("MTC initialization failed: ") + exception.what();
    } catch (const std::exception& exception) {
      message = std::string("MTC planning failed: ") + exception.what();
    }
    RCLCPP_ERROR(get_logger(), "%s", message.c_str());
    return false;
  }

  geometry_msgs::msg::PoseStamped alignedTargetPose(
    const Detection& detection, const geometry_msgs::msg::PointStamped& target) const
  {
    const auto normalize = [](std::array<double, 3> vector) {
      const double norm = std::sqrt(
        vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]);
      if (norm < 1e-9) {
        throw std::runtime_error("cannot normalize a zero direction");
      }
      for (auto& value : vector) {
        value /= norm;
      }
      return vector;
    };
    const auto cross = [](
      const std::array<double, 3>& first, const std::array<double, 3>& second) {
      return std::array<double, 3>{
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0]};
    };
    const auto dot = [](
      const std::array<double, 3>& first, const std::array<double, 3>& second) {
      return first[0] * second[0] + first[1] * second[1] + first[2] * second[2];
    };

    const auto z_axis = normalize(
      {-detection.normal.vector.x, -detection.normal.vector.y,
       -detection.normal.vector.z});
    std::array<double, 3> x_axis{
      detection.end.point.x - detection.start.point.x,
      detection.end.point.y - detection.start.point.y,
      detection.end.point.z - detection.start.point.z};
    const double projection = dot(x_axis, z_axis);
    for (std::size_t index = 0; index < 3; ++index) {
      x_axis[index] -= projection * z_axis[index];
    }
    x_axis = normalize(x_axis);
    const auto y_axis = normalize(cross(z_axis, x_axis));

    const double m00 = x_axis[0], m01 = y_axis[0], m02 = z_axis[0];
    const double m10 = x_axis[1], m11 = y_axis[1], m12 = z_axis[1];
    const double m20 = x_axis[2], m21 = y_axis[2], m22 = z_axis[2];
    geometry_msgs::msg::Quaternion quaternion;
    const double trace = m00 + m11 + m22;
    if (trace > 0.0) {
      const double scale = std::sqrt(trace + 1.0) * 2.0;
      quaternion.w = 0.25 * scale;
      quaternion.x = (m21 - m12) / scale;
      quaternion.y = (m02 - m20) / scale;
      quaternion.z = (m10 - m01) / scale;
    } else if (m00 > m11 && m00 > m22) {
      const double scale = std::sqrt(1.0 + m00 - m11 - m22) * 2.0;
      quaternion.w = (m21 - m12) / scale;
      quaternion.x = 0.25 * scale;
      quaternion.y = (m01 + m10) / scale;
      quaternion.z = (m02 + m20) / scale;
    } else if (m11 > m22) {
      const double scale = std::sqrt(1.0 + m11 - m00 - m22) * 2.0;
      quaternion.w = (m02 - m20) / scale;
      quaternion.x = (m01 + m10) / scale;
      quaternion.y = 0.25 * scale;
      quaternion.z = (m12 + m21) / scale;
    } else {
      const double scale = std::sqrt(1.0 + m22 - m00 - m11) * 2.0;
      quaternion.w = (m10 - m01) / scale;
      quaternion.x = (m02 + m20) / scale;
      quaternion.y = (m12 + m21) / scale;
      quaternion.z = 0.25 * scale;
    }

    geometry_msgs::msg::PoseStamped pose;
    pose.header = target.header;
    pose.pose.position = target.point;
    pose.pose.orientation = quaternion;
    return pose;
  }

  void publishMarkers(const Detection& detection, const geometry_msgs::msg::PointStamped& target)
  {
    visualization_msgs::msg::MarkerArray array;
    visualization_msgs::msg::Marker normal;
    normal.header = detection.center.header;
    normal.ns = "box_motion";
    normal.id = 0;
    normal.type = visualization_msgs::msg::Marker::ARROW;
    normal.action = visualization_msgs::msg::Marker::ADD;
    normal.points = {detection.center.point, target.point};
    normal.scale.x = 0.012;
    normal.scale.y = 0.024;
    normal.scale.z = 0.030;
    normal.color.r = 1.0;
    normal.color.g = 0.8;
    normal.color.a = 1.0;
    array.markers.push_back(normal);

    visualization_msgs::msg::Marker hover;
    hover.header = target.header;
    hover.ns = "box_motion";
    hover.id = 1;
    hover.type = visualization_msgs::msg::Marker::SPHERE;
    hover.action = visualization_msgs::msg::Marker::ADD;
    hover.pose.position = target.point;
    hover.pose.orientation.w = 1.0;
    hover.scale.x = hover.scale.y = hover.scale.z = 0.04;
    hover.color.b = 1.0;
    hover.color.a = 1.0;
    array.markers.push_back(hover);
    marker_pub_->publish(array);
  }

  std::mutex data_mutex_;
  std::mutex plan_mutex_;
  std::optional<geometry_msgs::msg::PointStamped> start_;
  std::optional<geometry_msgs::msg::PointStamped> end_;
  std::optional<geometry_msgs::msg::PointStamped> center_;
  std::optional<geometry_msgs::msg::Vector3Stamped> normal_;
  bool tracked_{false};
  std::optional<double> plane_rms_m_;
  std::optional<builtin_interfaces::msg::Time> planned_stamp_;
  std::unique_ptr<mtc::Task> planned_task_;

  std::string planning_group_;
  std::string tool_link_;
  std::string planning_pipeline_;
  std::string planner_id_;
  double approach_offset_m_;
  double max_detection_age_s_;
  double max_plane_rms_m_;
  int max_solutions_;
  double velocity_scaling_;
  double acceleration_scaling_;
  bool execute_;
  bool align_tool_z_to_surface_;

  rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr start_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr end_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr center_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Vector3Stamped>::SharedPtr normal_sub_;
  rclcpp::Subscription<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diagnostics_sub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr plan_service_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr clear_service_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<BoxMotionExecutor>();
  rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 2);
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
