#include "igus_rebel/Rebel.hpp"

#include <arpa/inet.h>

#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>
#include <utility>
#include <variant>

#include "rclcpp/rclcpp.hpp"
#include "igus_rebel/CriKeywords.hpp"
#include "hardware_interface/types/hardware_interface_type_values.hpp"

namespace Igus
{
namespace
{
constexpr std::size_t kMaxUnacknowledgedCommands = 100;
constexpr std::chrono::milliseconds kMessageWaitTimeout{100};

std::int64_t ToNs(Safety::SteadyClock::time_point time)
{
  return std::chrono::duration_cast<std::chrono::nanoseconds>(time.time_since_epoch()).count();
}

Safety::SteadyClock::time_point FromNs(std::int64_t ns)
{
  return Safety::SteadyClock::time_point(std::chrono::nanoseconds(ns));
}

std::string Truncate(const std::string & text, std::size_t maxLength = 200)
{
  return text.size() <= maxLength ? text : text.substr(0, maxLength) + "...";
}
}  // namespace

//
// Constructor(s) / Destructor(s)
//
Rebel::Rebel()
: logger_(rclcpp::get_logger("igus_rebel"))
{
  ZeroJog();
}

Rebel::Rebel(const std::string & testIp, int testPort)
: logger_(rclcpp::get_logger("igus_rebel")),
  ip(testIp),
  port(testPort)
{
  ZeroJog();
}

Rebel::~Rebel()
{
  Stop();
}

//
// private functions
//
void Rebel::ZeroJog()
{
  for (auto & value : jog) {
    value.store(0.0f);
  }
}

void Rebel::AliveThreadFunction()
{
  bool announced = false;
  // Until the controller manager calls write() the command counts as stale (zero jog).
  bool wasStale = true;
  auto next = Safety::SteadyClock::now();

  while (continueAlive) {
    if (aliveTransmit) {
      if (!announced) {
        RCLCPP_INFO(logger_, "Starting to send ALIVEJOG");
        announced = true;
      }

      const auto now = Safety::SteadyClock::now();
      const bool stale = Safety::IsCommandStale(
        FromNs(lastWriteNs.load()), now, Safety::kCommandWatchdogTimeout);

      std::array<float, 6> currentJog{};
      if (!stale) {
        for (std::size_t i = 0; i < currentJog.size(); ++i) {
          currentJog[i] = jog[i].load();
        }
      }

      if (stale != wasStale) {
        if (stale) {
          RCLCPP_WARN(
            logger_,
            "No fresh write() command (watchdog %lld ms, or hardware stopped): sending zero jog",
            static_cast<long long>(Safety::kCommandWatchdogTimeout.count()));
        } else {
          RCLCPP_INFO(logger_, "Receiving write() commands: forwarding jog commands");
        }
        wasStale = stale;
      }

      rebelSocket->SendMessage(Safety::FormatAliveJog(Ccnt(), currentJog));
    }

    next += Safety::kAlivePeriod;
    const auto now = Safety::SteadyClock::now();
    if (now > next + 5 * Safety::kAlivePeriod) {
      // Overrun (e.g. a blocked send); resynchronise instead of sending a burst.
      next = now + Safety::kAlivePeriod;
    }
    std::unique_lock<std::mutex> lock(aliveWakeLock);
    aliveWake.wait_until(lock, next, [this]() {return !continueAlive;});
  }

  RCLCPP_WARN(logger_, "Stopped to send ALIVEJOG");
}

void Rebel::MessageThreadFunction()
{
  RCLCPP_INFO(logger_, "Starting to process robot messages");

  while (continueMessage) {
    std::string msg;
    if (!rebelSocket->WaitForMessage(msg, kMessageWaitTimeout)) {
      continue;
    }

    try {
      HandleMessage(msg);
    } catch (const std::exception & e) {
      RCLCPP_ERROR_THROTTLE(
        logger_, throttleClock_, 5000, "Failed to process robot message '%s': %s",
        Truncate(msg).c_str(), e.what());
    } catch (...) {
      RCLCPP_ERROR_THROTTLE(
        logger_, throttleClock_, 5000, "Failed to process robot message '%s'",
        Truncate(msg).c_str());
    }
  }

  RCLCPP_WARN(logger_, "Stopped to process robot messages");
}

void Rebel::HandleMessage(const std::string & msg)
{
  {
    // Track skipped CRI counter values (every message type shares the robot's counter).
    std::istringstream counterStream(msg);
    int counter = 0;
    if (counterStream >> counter) {
      if (lastCriCounter_ >= 0 && counter > lastCriCounter_ + 1) {
        counterSkipped_ += static_cast<std::uint64_t>(counter - lastCriCounter_ - 1);
      }
      lastCriCounter_ = counter;
    }
  }
  CriMessages::ParsedMessage parsed = CriMessages::ParseMessage(msg);

  if (!parsed.ok) {
    if (parsed.type == CriMessages::MessageType::STATUS) {
      RCLCPP_WARN_THROTTLE(
        logger_, throttleClock_, 5000, "Ignoring malformed CRI status message");
    } else if (parsed.type == CriMessages::MessageType::UNKNOWN) {
      RCLCPP_ERROR_THROTTLE(
        logger_, throttleClock_, 5000, "UNKNOWN MESSAGE: %s", Truncate(msg).c_str());
    } else {
      RCLCPP_WARN_THROTTLE(
        logger_, throttleClock_, 5000, "Ignoring malformed CRI message (%s): %s",
        parsed.error.c_str(), Truncate(msg).c_str());
    }
    return;
  }

  switch (parsed.type) {
    case CriMessages::MessageType::STATUS:
      {
        const auto & status = std::get<CriMessages::Status>(parsed.payload);
        std::array<double, 6> positionRad{};
        if (!Safety::JointPositionsRad(status, positionRad)) {
          RCLCPP_WARN_THROTTLE(
            logger_, throttleClock_, 5000,
            "Ignoring CRI status message with non-finite joint positions");
          break;
        }
        {
          std::lock_guard<std::mutex> lock(snapshotLock);
          snapshot.positionRad = positionRad;
          snapshot.stamp = Safety::SteadyClock::now();
          snapshot.overridePercent = status.overrideValue;
          snapshot.counterSkipped = counterSkipped_;
          ++snapshot.sequence;
        }
        snapshotCondition.notify_all();
        ProcessStatus(status);
        break;
      }

    case CriMessages::MessageType::MESSAGE:
      {
        const auto & message = std::get<CriMessages::Message>(parsed.payload);
        RCLCPP_INFO(logger_, "Rebel MESSAGE: %s", message.message.c_str());
        break;
      }

    case CriMessages::MessageType::CMD:
      {
        const auto & command = std::get<CriMessages::Command>(parsed.payload);
        RCLCPP_INFO(logger_, "CMD: %s", command.command.c_str());
        break;
      }

    case CriMessages::MessageType::INFO:
      {
        const auto & info = std::get<CriMessages::Info>(parsed.payload);
        RCLCPP_INFO(logger_, "INFO: %s", info.info.c_str());
        break;
      }

    case CriMessages::MessageType::LOGMSG:
      {
        const auto & log = std::get<CriMessages::LogMsg>(parsed.payload);
        switch (log.logLevel) {
          case CriMessages::LogLevel::DEBUG:
            RCLCPP_DEBUG(logger_, "REBEL LOG: %s (%ld ms)", log.logMsg.c_str(), log.timestamp);
            break;
          case CriMessages::LogLevel::APP_INFO:
            RCLCPP_INFO(
              logger_, "REBEL LOG (APP_INFO): %s (%ld ms)", log.logMsg.c_str(), log.timestamp);
            break;
          case CriMessages::LogLevel::APP_ERROR:
            RCLCPP_ERROR(
              logger_, "REBEL LOG (APP_ERROR): %s (%ld ms)", log.logMsg.c_str(), log.timestamp);
            break;
          case CriMessages::LogLevel::INFO:
            // The Rebel is chatty with INFO log messages; they are forwarded at ROS INFO level.
            RCLCPP_INFO(logger_, "REBEL LOG: %s (%ld ms)", log.logMsg.c_str(), log.timestamp);
            break;
          case CriMessages::LogLevel::WARN:
            RCLCPP_WARN(logger_, "REBEL LOG: %s (%ld ms)", log.logMsg.c_str(), log.timestamp);
            break;
          case CriMessages::LogLevel::ERROR:
            RCLCPP_ERROR(logger_, "REBEL LOG: %s (%ld ms)", log.logMsg.c_str(), log.timestamp);
            break;
          case CriMessages::LogLevel::FATAL:
            RCLCPP_FATAL(logger_, "REBEL LOG: %s (%ld ms)", log.logMsg.c_str(), log.timestamp);
            break;
          case CriMessages::LogLevel::UNKNOWN:
            RCLCPP_ERROR(
              logger_, "REBEL LOG (UNKNOWN LOG LEVEL): %s (%ld ms)", log.logMsg.c_str(),
              log.timestamp);
            break;
        }
        break;
      }

    case CriMessages::MessageType::CMDERROR:
      {
        const auto & error = std::get<CriMessages::CmdError>(parsed.payload);
        std::string command;
        {
          std::lock_guard<std::mutex> lock(commandsLock);
          const auto it = unacknowledgedCommands.find(error.recjectedCmd);
          if (it != unacknowledgedCommands.end()) {
            command = it->second;
            unacknowledgedCommands.erase(it);
          }
        }
        if (!command.empty()) {
          RCLCPP_ERROR(
            logger_, "Rebel did not accept command: %s. Error message: %s", command.c_str(),
            error.error.c_str());
        } else {
          RCLCPP_ERROR(
            logger_, "Rebel did not accept unknown command. Error message: %s (%d)",
            error.error.c_str(), error.recjectedCmd);
        }
        break;
      }

    case CriMessages::MessageType::CMDACK:
      {
        const auto & ack = std::get<CriMessages::CmdAck>(parsed.payload);
        std::string command;
        {
          std::lock_guard<std::mutex> lock(commandsLock);
          const auto it = unacknowledgedCommands.find(ack.acceptedCmd);
          if (it != unacknowledgedCommands.end()) {
            command = it->second;
            unacknowledgedCommands.erase(it);
          }
        }
        if (!command.empty()) {
          RCLCPP_INFO(logger_, "Rebel accepted command: %s", command.c_str());
        } else {
          RCLCPP_WARN(logger_, "Rebel accepted unknown command: %d", ack.acceptedCmd);
        }
        break;
      }

    case CriMessages::MessageType::CYCLESTAT:
      {
        const auto & cyclestat = std::get<CriMessages::Cyclestat>(parsed.payload);
        RCLCPP_INFO_THROTTLE(
          logger_, throttleClock_, 120000,
          "Rebel cycle statistics -- Cycletime: %.1f ms -- Workload: %.2f", cyclestat.cycletime,
          cyclestat.workload);
        break;
      }

    case CriMessages::MessageType::CONFIG:
      RCLCPP_DEBUG(logger_, "Ignoring CONFIG message: %s", Truncate(msg).c_str());
      break;

    case CriMessages::MessageType::UNKNOWN:
    case CriMessages::MessageType::RUNSTATE:
    case CriMessages::MessageType::VARIABLES:
    case CriMessages::MessageType::OPINFO:
    case CriMessages::MessageType::GSIG:
    case CriMessages::MessageType::GRIPPERSTATE:
      break;
  }
}

int Rebel::Ccnt()
{
  std::lock_guard<std::mutex> lockGuard(cntLock);
  int current = current_ccnt;
  current_ccnt = (current_ccnt % 9999) + 1;
  return current;
}

bool Rebel::SetDigitalOut(const int & output, const bool & is_on)
{
  std::ostringstream cmd;
  cmd << CriKeywords::COMMAND_DOUT << " " << output << " " << (is_on ? "true" : "false");
  return Command(cmd.str());
}

bool Rebel::Command(const std::string & command)
{
  if (!rebelSocket) {
    return false;
  }

  const int commandCount = Ccnt();
  {
    std::lock_guard<std::mutex> lock(commandsLock);
    if (unacknowledgedCommands.size() >= kMaxUnacknowledgedCommands) {
      unacknowledgedCommands.clear();
    }
    unacknowledgedCommands[commandCount] = command;
  }

  if (!rebelSocket->SendMessage(Safety::FormatCommand(commandCount, command))) {
    std::lock_guard<std::mutex> lock(commandsLock);
    unacknowledgedCommands.erase(commandCount);
    RCLCPP_WARN(logger_, "Could not send command '%s': not connected", command.c_str());
    return false;
  }
  return true;
}

void Rebel::OnConnected(bool reconnect)
{
  if (stopping) {
    return;
  }
  std::lock_guard<std::mutex> startupGuard(startupLock_);

  // Start-up sequence, re-sent on every (re)connect. ALIVEJOG (initially zero) starts
  // after Enable, as in the original driver.
  aliveTransmit = false;
  ZeroJog();
  {
    std::lock_guard<std::mutex> lock(commandsLock);
    unacknowledgedCommands.clear();
  }

  // Stop() may run concurrently (it disables reconnects first); do not send anything then.
  const auto send = [this](const std::string & command) {
      return !stopping && Command(command);
    };

  // The CRI "Connect" command is intentionally not sent: CRI_V17 answers it with CMDERROR.
  send(CriKeywords::COMMAND_SETACTIVE + " true");
  send(CriKeywords::COMMAND_RESET);
  send(CriKeywords::COMMAND_ENABLE);

  aliveTransmit = !stopping;

  // Joint jog (velocity) mode; the only control mode this driver uses.
  send(CriKeywords::COMMAND_MOTIONTYPEJOINT);

  // Global velocity override 100 % so the robot moves at the commanded (planned) speed.
  std::ostringstream overrideCommand;
  overrideCommand << CriKeywords::COMMAND_OVERRIDE << " " << std::fixed;
  overrideCommand.precision(1);
  overrideCommand << Safety::kExpectedOverridePercent;
  if (send(overrideCommand.str())) {
    RCLCPP_INFO(
      logger_, "Sent CRI '%s' (velocity override %.1f %%)", overrideCommand.str().c_str(),
      Safety::kExpectedOverridePercent);
  }

  if (reconnect) {
    RCLCPP_WARN(
      logger_,
      "Reconnected to ReBeL: re-sent SetActive, Reset, Enable, MotionTypeJoint and Override");
  }
  RCLCPP_INFO(logger_, "Rebel now controlled by velocity control.");
}

void Rebel::ProcessStatus(const CriMessages::Status & status)
{
  const CriMessages::Kinstate currentKinstate = status.kinstate;
  const std::array<int, 16> & currentErrorJoints = status.errorJoints;

  if (lastKinstate != currentKinstate) {
    if (lastKinstate != CriMessages::Kinstate::NO_ERROR) {
      RCLCPP_INFO(logger_, "Kinematics error resolved [%s]", kinstateMessage.c_str());
    }

    kinstateMessage = CriMessages::KinstateToString(currentKinstate);
    if (currentKinstate != CriMessages::Kinstate::NO_ERROR) {
      RCLCPP_ERROR(logger_, "Kinematics error [%s]", kinstateMessage.c_str());
    }
  }

  if (currentErrorJoints != lastErrorJoints) {
    // Loop through the 6 joint errors.
    for (unsigned int i = 0; i < 6; i++) {
      const int errorJoint = currentErrorJoints.at(i);
      if (errorJoint == lastErrorJoints.at(i)) {
        continue;
      }
      const std::string errorMsg = CriMessages::DescribeJointErrorBits(errorJoint);
      if (!errorMsg.empty()) {
        RCLCPP_ERROR(logger_, "Joint %u Error: [%s]", i, errorMsg.c_str());
      } else {
        RCLCPP_INFO(logger_, "Joint %u Error: Cleared", i);
      }
    }
  }

  // Velocity override: log the first value and every change, warn while not ~100 %.
  const float overrideValue = status.overrideValue;
  if (!haveOverride || std::fabs(overrideValue - lastOverride) > 0.05f) {
    RCLCPP_INFO(logger_, "Robot velocity override is %.1f %%", overrideValue);
    haveOverride = true;
    lastOverride = overrideValue;
  }
  if (!Safety::IsOverrideAsExpected(overrideValue)) {
    RCLCPP_WARN_THROTTLE(
      logger_, throttleClock_, 10000,
      "Robot velocity override is %.1f %%, expected %.1f %%: the arm will not move at the "
      "commanded speed", overrideValue, Safety::kExpectedOverridePercent);
  }

  unsigned disabledMask = 0;
  for (unsigned i = 0; i < 6; ++i) {
    if ((currentErrorJoints.at(i) & static_cast<int>(CriMessages::ErrorJoint::MNE)) != 0) {
      disabledMask |= (1u << i);
    }
  }
  drivesDisabledMask_.store(disabledMask);

  lastKinstate = currentKinstate;
  lastErrorJoints = currentErrorJoints;
}

bool Rebel::ValidateJoints(const HardwareInfo & hardware_info)
{
  const auto & joints = hardware_info.joints;
  if (joints.size() != 6) {
    RCLCPP_FATAL(
      logger_,
      "Hardware component '%s' must declare exactly 6 joints (joint1..joint6 in CRI order, "
      "optionally prefixed), got %zu",
      componentName_.c_str(), joints.size());
    return false;
  }
  for (std::size_t i = 0; i < joints.size(); ++i) {
    const auto & joint = joints[i];
    const std::string suffix = "joint" + std::to_string(i + 1);
    const bool orderOk = joint.name.size() >= suffix.size() &&
      joint.name.compare(joint.name.size() - suffix.size(), suffix.size(), suffix) == 0;
    if (!orderOk) {
      RCLCPP_FATAL(
        logger_,
        "Hardware component '%s': joint #%zu is '%s' but must be '%s' (optionally prefixed): "
        "the joints must be listed in CRI order 1..6 or commands would reach the wrong joint",
        componentName_.c_str(), i + 1, joint.name.c_str(), suffix.c_str());
      return false;
    }
    for (std::size_t j = 0; j < i; ++j) {
      if (joints[j].name == joint.name) {
        RCLCPP_FATAL(
          logger_, "Hardware component '%s': duplicate joint name '%s'",
          componentName_.c_str(), joint.name.c_str());
        return false;
      }
    }
    if (joint.command_interfaces.size() != 1 ||
      joint.command_interfaces[0].name != hardware_interface::HW_IF_VELOCITY)
    {
      RCLCPP_FATAL(
        logger_,
        "Hardware component '%s': joint '%s' must have exactly one command interface "
        "('velocity')",
        componentName_.c_str(), joint.name.c_str());
      return false;
    }
    bool havePosition = false;
    bool haveVelocity = false;
    for (const auto & state : joint.state_interfaces) {
      havePosition = havePosition || state.name == hardware_interface::HW_IF_POSITION;
      haveVelocity = haveVelocity || state.name == hardware_interface::HW_IF_VELOCITY;
    }
    if (joint.state_interfaces.size() != 2 || !havePosition || !haveVelocity) {
      RCLCPP_FATAL(
        logger_,
        "Hardware component '%s': joint '%s' must have exactly the state interfaces "
        "'position' and 'velocity'",
        componentName_.c_str(), joint.name.c_str());
      return false;
    }
  }
  return true;
}

//
// public functions
//
CallbackReturn Rebel::on_init(const HardwareInfo & hardware_info)
{
  if (SystemInterface::on_init(hardware_info) != CallbackReturn::SUCCESS) {
    return CallbackReturn::ERROR;
  }

  // Per-component logger so that several arms in one controller_manager can be told apart.
  componentName_ = hardware_info.name;
  if (!componentName_.empty()) {
    logger_ = rclcpp::get_logger("igus_rebel").get_child(componentName_);
  }

  const auto & params = hardware_info.hardware_parameters;
  const auto ip_it = params.find("robot_ip");
  if (ip_it != params.end() && !ip_it->second.empty()) {
    in_addr parsed{};
    if (inet_pton(AF_INET, ip_it->second.c_str(), &parsed) != 1) {
      RCLCPP_FATAL(logger_, "Invalid robot_ip '%s' (expected IPv4 address)", ip_it->second.c_str());
      return CallbackReturn::ERROR;
    }
    ip = ip_it->second;
  }
  const auto port_it = params.find("robot_port");
  if (port_it != params.end() && !port_it->second.empty()) {
    int parsed_port = 0;
    try {
      std::size_t used = 0;
      parsed_port = std::stoi(port_it->second, &used);
      if (used != port_it->second.size()) {
        parsed_port = 0;
      }
    } catch (const std::exception &) {
      parsed_port = 0;
    }
    if (parsed_port < 1 || parsed_port > 65535) {
      RCLCPP_FATAL(logger_, "Invalid robot_port '%s'", port_it->second.c_str());
      return CallbackReturn::ERROR;
    }
    port = parsed_port;
  }
  RCLCPP_INFO(logger_, "CRI controller target %s:%d", ip.c_str(), port);

  // Optional naming parameters; the defaults are the single-arm names.
  const auto readName = [&params](const char * key, std::string & target) {
      const auto it = params.find(key);
      if (it != params.end() && !it->second.empty()) {
        target = it->second;
      }
    };
  nodeName_ = kDefaultNodeName;
  digitalOutputService_ = kDefaultDigitalOutputService;
  trajectoryControllerName_ = kDefaultTrajectoryControllerName;
  readName("node_name", nodeName_);
  // A renamed node (one of several arms) defaults to a node-private service so two arms never
  // both advertise the global /set_digital_output.
  if (nodeName_ != kDefaultNodeName) {
    digitalOutputService_ = std::string("~/") + kDefaultDigitalOutputService;
  }
  readName("digital_output_service", digitalOutputService_);
  readName("trajectory_controller_name", trajectoryControllerName_);

  // Validate the ros2_control description before anything is started: the interfaces are
  // exported by joint position (info_.joints[i] <-> CRI joint i + 1), so a wrong count or
  // order would send commands to the wrong joint.
  if (!ValidateJoints(hardware_info)) {
    return CallbackReturn::ERROR;
  }
  for (std::size_t i = 0; i < 6; ++i) {
    jointNames_[i] = hardware_info.joints[i].name;
  }

  ZeroJog();
  lastWriteNs = 0;
  current_ccnt = 1;
  lastKinstate = CriMessages::Kinstate::NO_ERROR;
  lastErrorJoints.fill(0);
  kinstateMessage = "";
  haveOverride = false;
  stopping = false;
  stopped = false;
  hardwareErrorLatched_ = false;

  drivesDisabledMask_ = 0;
  holdJogZero_ = false;
  statusHold_ = false;
  statusCoast_ = false;
  lastCriCounter_ = -1;
  counterSkipped_ = 0;
  lastFreshCounterSkipped_ = 0;
  lastMovingNs_ = 0;
  lastMovingCmd_ = 0.0;
  lastFreshStamp_ = {};
  gapCount_ = 0;
  gapMaxMs_ = 0.0;

  try {
    node_ = std::make_shared<rclcpp::Node>(nodeName_);
    reset_enable_srv_ = node_->create_service<std_srvs::srv::Trigger>(
      "~/reset_and_enable",
      [this](
        const std::shared_ptr<std_srvs::srv::Trigger::Request>,
        std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
        response->success = ResetAndEnable(response->message);
      });
    digital_output_srv_ = node_->create_service<igus_rebel_msgs::srv::SetDigitalOutput>(
      digitalOutputService_,
      std::bind(&Rebel::dio_callback, this, std::placeholders::_1, std::placeholders::_2));
  } catch (const std::exception & error) {
    RCLCPP_FATAL(
      logger_, "Cannot create ROS node '%s' or its services ('%s'): %s", nodeName_.c_str(),
      digitalOutputService_.c_str(), error.what());
    digital_output_srv_.reset();
    reset_enable_srv_.reset();
    node_.reset();
    return CallbackReturn::ERROR;
  }

  // The socket exists before the executor starts, so service callbacks never see it unset.
  rebelSocket = std::make_shared<RebelSocket>(ip, port, logger_);
  rebelSocket->SetConnectedCallback([this](bool reconnect) {OnConnected(reconnect);});

  // Service callbacks run on their own executor thread instead of inside write().
  executor_ = std::make_shared<rclcpp::executors::SingleThreadedExecutor>();
  executor_->add_node(node_);
  executorThread = std::thread([this]() {executor_->spin();});

  Start();

  // Do not report success before the robot state is known: seed the state from the first
  // valid STATUS so read() never publishes the all-zero default pose.
  JointSnapshot first;
  {
    std::unique_lock<std::mutex> lock(snapshotLock);
    snapshotCondition.wait_for(
      lock, Safety::kFirstStatusTimeout, [this]() {return snapshot.sequence > 0;});
    first = snapshot;
  }

  if (first.sequence == 0) {
    RCLCPP_ERROR(
      logger_, "No valid STATUS received from ReBeL at %s:%d within %lld ms; aborting init",
      ip.c_str(), port, static_cast<long long>(Safety::kFirstStatusTimeout.count()));
    Stop();
    return CallbackReturn::ERROR;
  }

  for (std::size_t i = 0; i < 6; ++i) {
    pos[i] = first.positionRad[i];
    vel[i] = 0.0;
    vel_cmd[i] = 0.0;
  }
  initialConnectionCount = rebelSocket->ConnectionCount();
  last_pos = first.positionRad;
  lastPosStamp = first.stamp;
  lastFreshStamp_ = first.stamp;
  lastFreshCounterSkipped_ = first.counterSkipped;
  lastConsumedSequence = first.sequence;

  RCLCPP_INFO(
    logger_,
    "First STATUS: joints [deg] %.2f %.2f %.2f %.2f %.2f %.2f, override %.1f %%",
    first.positionRad[0] / degToRad, first.positionRad[1] / degToRad,
    first.positionRad[2] / degToRad, first.positionRad[3] / degToRad,
    first.positionRad[4] / degToRad, first.positionRad[5] / degToRad, first.overridePercent);

  return CallbackReturn::SUCCESS;
}

CallbackReturn Rebel::on_configure(const rclcpp_lifecycle::State &)
{
  // Reached from UNCONFIGURED: at normal start-up (before the first on_activate()) and,
  // after a latched hardware error, via
  // `ros2 control set_hardware_component_state <component> inactive`. Either way jog stays
  // zero here; on_activate() is the only place the latch can clear, and only if STATUS is
  // fresh.
  for (double & command : vel_cmd) {
    command = 0.0;
  }
  ZeroJog();
  if (hardwareErrorLatched_) {
    RCLCPP_WARN(
      logger_,
      "Hardware component '%s' configured (still latched from a prior error); jog held at "
      "zero. Recover with 'ros2 control set_hardware_component_state %s active' -- refused "
      "unless STATUS is fresh.",
      componentName_.c_str(), componentName_.c_str());
  }
  return CallbackReturn::SUCCESS;
}

CallbackReturn Rebel::on_activate(const rclcpp_lifecycle::State &)
{
  for (double & command : vel_cmd) {
    command = 0.0;
  }
  ZeroJog();
  lastWriteNs = 0;

  if (hardwareErrorLatched_) {
    JointSnapshot latest;
    {
      std::lock_guard<std::mutex> lock(snapshotLock);
      latest = snapshot;
    }
    const auto now = Safety::SteadyClock::now();
    const bool stale = latest.sequence == 0 ||
      Safety::IsStatusStale(latest.stamp, now, Safety::kStatusStaleTimeout);
    if (stale) {
      const double ageMs = latest.sequence == 0 ?
        -1.0 : std::chrono::duration<double, std::milli>(now - latest.stamp).count();
      RCLCPP_ERROR(
        logger_,
        "Refusing to activate hardware component '%s': STATUS is still stale (%.0f ms, limit "
        "%lld ms). Check the robot and CRI connection, then retry "
        "'ros2 control set_hardware_component_state %s active'.",
        componentName_.c_str(), ageMs,
        static_cast<long long>(Safety::kStatusStaleTimeout.count()), componentName_.c_str());
      return CallbackReturn::FAILURE;
    }

    // Fresh STATUS: accept the current connection and STATUS sequence as the new baseline
    // (so read() does not immediately re-trip on the reconnect-count check or compute a
    // velocity spike across the outage), then re-send the same start-up command sequence
    // used on every (re)connect -- SetActive, Reset, Enable, MotionTypeJoint, Override --
    // with jog held at zero throughout (OnConnected() zeroes it before sending anything).
    initialConnectionCount = rebelSocket->ConnectionCount();
    for (std::size_t i = 0; i < 6; ++i) {
      pos[i] = latest.positionRad[i];
      vel[i] = 0.0;
    }
    last_pos = latest.positionRad;
    lastPosStamp = latest.stamp;
    lastConsumedSequence = latest.sequence;
    lastFreshStamp_ = latest.stamp;
    statusHold_ = false;
    statusCoast_ = false;
    lastFreshCounterSkipped_ = latest.counterSkipped;
    lastMovingNs_ = 0;

    OnConnected(true);
    hardwareErrorLatched_ = false;
    RCLCPP_WARN(
      logger_,
      "Hardware component '%s' recovered from a latched error: re-sent SetActive, Reset, "
      "Enable, MotionTypeJoint and Override with jog held at zero. This driver does NOT "
      "restart the controllers claiming this hardware (e.g. '%s' "
      "in this workspace) -- switch them too, e.g. "
      "'ros2 control switch_controllers --deactivate <controller> --activate <controller>', "
      "so they resample the current joint state instead of continuing a stale trajectory.",
      componentName_.c_str(), trajectoryControllerName_.c_str());
  }

  return CallbackReturn::SUCCESS;
}

CallbackReturn Rebel::on_deactivate(const rclcpp_lifecycle::State &)
{
  for (double & command : vel_cmd) {
    command = 0.0;
  }
  ZeroJog();
  lastWriteNs = 0;
  RCLCPP_INFO(logger_, "Hardware deactivated: jog set to zero");
  return CallbackReturn::SUCCESS;
}

CallbackReturn Rebel::on_error(const rclcpp_lifecycle::State &)
{
  // ros2_control stops calling read()/write() after an error; make sure the ALIVEJOG
  // thread sends zero from now on (the stale-command watchdog would do so after 50 ms).
  // The latch is fail-closed: it is never cleared here or by any automatic retry, only by
  // on_activate() after an operator-driven lifecycle reset confirms STATUS is fresh again.
  hardwareErrorLatched_ = true;
  ZeroJog();
  lastWriteNs = 0;
  RCLCPP_ERROR(
    logger_,
    "Hardware error on component '%s': jog forced to zero and latched. Recover with "
    "'ros2 control set_hardware_component_state %s inactive' then "
    "'ros2 control set_hardware_component_state %s active' (the latter is refused unless "
    "STATUS is fresh); also cycle the controllers claiming this hardware (e.g. "
    "'ros2 control switch_controllers --deactivate %s --activate "
    "%s') so they do not resume from a stale setpoint.",
    componentName_.c_str(), componentName_.c_str(), componentName_.c_str(),
    trajectoryControllerName_.c_str(), trajectoryControllerName_.c_str());
  return CallbackReturn::SUCCESS;
}

std::vector<StateInterface> Rebel::export_state_interfaces()
{
  std::vector<StateInterface> state_interfaces;

  for (int i = 0; i < 6; ++i) {
    state_interfaces.emplace_back(
      StateInterface(
        jointNames_[i], hardware_interface::HW_IF_POSITION, &pos[i]));
    state_interfaces.emplace_back(
      StateInterface(
        jointNames_[i], hardware_interface::HW_IF_VELOCITY, &vel[i]));
  }

  return state_interfaces;
}

std::vector<CommandInterface> Rebel::export_command_interfaces()
{
  std::vector<CommandInterface> command_interfaces;

  for (int i = 0; i < 6; ++i) {
    command_interfaces.emplace_back(
      CommandInterface(
        jointNames_[i], hardware_interface::HW_IF_VELOCITY, &vel_cmd[i]));
  }

  return command_interfaces;
}

return_type Rebel::read(const rclcpp::Time &, const rclcpp::Duration &)
{
  if (hardwareErrorLatched_) {
    // Fail-closed: once latched, read() keeps returning ERROR regardless of STATUS
    // freshness. Only on_activate() (after an operator-driven lifecycle reset) can clear
    // this, and only once it confirms STATUS is fresh again.
    ZeroJog();
    RCLCPP_ERROR_THROTTLE(
      logger_, throttleClock_, 1000,
      "Hardware error latched on component '%s': read() returns ERROR until an operator "
      "recovers it ('ros2 control set_hardware_component_state %s inactive' then "
      "'... active'; the latter is refused unless STATUS is fresh).",
      componentName_.c_str(), componentName_.c_str());
    return return_type::ERROR;
  }

  const unsigned disabledMask = drivesDisabledMask_.load();
  if (disabledMask != 0 && !holdJogZero_) {
    RCLCPP_WARN_THROTTLE(
      logger_, throttleClock_, 10000,
      "Robot drives disabled (STATUS reports 'Motor not enabled', joint mask 0x%02x) while the "
      "connection is healthy: the arm will not move. Call "
      "'ros2 service call /%s/reset_and_enable std_srvs/srv/Trigger' to re-enable.",
      disabledMask, nodeName_.c_str());
  }

  JointSnapshot latest;
  {
    std::lock_guard<std::mutex> lock(snapshotLock);
    latest = snapshot;
  }

  const auto now = Safety::SteadyClock::now();
  const std::string recovery =
    "Recover with 'ros2 control set_hardware_component_state " + componentName_ +
    " inactive' then '... active' (refused unless STATUS is fresh).";

  if (rebelSocket->ConnectionCount() != initialConnectionCount) {
    // The connection dropped and was re-established (OnConnected re-enabled the robot).
    // Do not let the controllers resume motion on their own: fail closed. An operator
    // lifecycle reset resumes control after checking the robot.
    ZeroJog();
    hardwareErrorLatched_ = true;
    statusHold_ = false;
    statusCoast_ = false;
    RCLCPP_ERROR_THROTTLE(
      logger_, throttleClock_, 1000,
      "Connection to ReBeL was re-established after a drop: read() returns ERROR and "
      "latches so motion does not resume automatically. %s", recovery.c_str());
    return return_type::ERROR;
  }

  const auto age = latest.sequence == 0 ?
    std::chrono::nanoseconds::zero() : std::chrono::duration_cast<std::chrono::nanoseconds>(
    now - latest.stamp);
  const auto health = Safety::ClassifyStatusAge(
    latest.sequence != 0, age, Safety::kStatusStaleTimeout, Safety::kStatusCoastTimeout,
    Safety::kStatusLatchTimeout);

  if (health == Safety::StatusHealth::Latch) {
    // Returning ERROR makes ros2_control call on_error() and stop calling read()/write()
    // for this component. Latch here too so fail-closed does not depend on on_error().
    ZeroJog();
    hardwareErrorLatched_ = true;
    statusHold_ = false;
    statusCoast_ = false;
    const double ageMs = latest.sequence == 0 ?
      -1.0 : std::chrono::duration<double, std::milli>(age).count();
    RCLCPP_ERROR_THROTTLE(
      logger_, throttleClock_, 1000,
      "No valid STATUS from ReBeL for %.0f ms (latch limit %lld ms): read() returns ERROR and "
      "latches. %s",
      ageMs, static_cast<long long>(Safety::kStatusLatchTimeout.count()), recovery.c_str());
    return return_type::ERROR;
  }

  if (health == Safety::StatusHealth::Coast) {
    // Soft stale, tier 1: keep sending the last jog (frozen; write() ignores new commands,
    // the controller computes them on frozen state). Position and velocity are held at the
    // last estimate: the arm really is still moving at about that velocity, and reporting 0
    // would make the controller see a stopped arm and add a P-term surge on resume.
    if (!statusCoast_.exchange(true)) {
      RCLCPP_WARN(
        logger_,
        "STATUS gap: coasting the last jog (no STATUS for %.0f ms, limit %lld ms; jog is zeroed "
        "after %lld ms, latches after %lld ms)",
        std::chrono::duration<double, std::milli>(age).count(),
        static_cast<long long>(Safety::kStatusStaleTimeout.count()),
        static_cast<long long>(Safety::kStatusCoastTimeout.count()),
        static_cast<long long>(Safety::kStatusLatchTimeout.count()));
    }
    return return_type::OK;
  }

  if (health == Safety::StatusHealth::Hold) {
    // Soft stale, tier 2: hold at zero jog, keep last-known position with zero velocity.
    ZeroJog();
    if (!statusHold_.exchange(true)) {
      statusCoast_ = false;
      for (double & v : vel) {
        v = 0.0;
      }
      RCLCPP_WARN(
        logger_,
        "STATUS gap: holding jog at zero (no STATUS for %.0f ms, limit %lld ms; latches after "
        "%lld ms)",
        std::chrono::duration<double, std::milli>(age).count(),
        static_cast<long long>(Safety::kStatusCoastTimeout.count()),
        static_cast<long long>(Safety::kStatusLatchTimeout.count()));
    }
    return return_type::OK;
  }

  if (statusHold_ || statusCoast_) {
    // Fresh STATUS again after a coast/hold. Measure the actual gap between the last STATUS
    // before it and the first one after, and the tier that gap reached.
    const double gapMs = lastFreshStamp_ == Safety::SteadyClock::time_point{} ?
    0.0 : std::chrono::duration<double, std::milli>(latest.stamp - lastFreshStamp_).count();
    const auto gapTier = Safety::ClassifyStatusAge(
      true, std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double, std::milli>(gapMs)),
      Safety::kStatusStaleTimeout, Safety::kStatusCoastTimeout, Safety::kStatusLatchTimeout);
    const bool motion = Safety::MotionDuringGap(
      lastMovingNs_.load(), ToNs(lastFreshStamp_),
      std::chrono::duration_cast<std::chrono::nanoseconds>(Safety::kHoldMotionLookback));
    if (!Safety::GapResumeAllowed(gapTier, motion)) {
      ZeroJog();
      hardwareErrorLatched_ = true;
      statusHold_ = false;
      statusCoast_ = false;
      if (gapTier == Safety::StatusHealth::Latch) {
        RCLCPP_ERROR(
          logger_, "STATUS resumed after %.0f ms (latch limit %lld ms): latched. %s", gapMs,
          static_cast<long long>(Safety::kStatusLatchTimeout.count()), recovery.c_str());
      } else {
        // The controller kept commanding motion through a gap long enough that jog was zeroed:
        // its trajectory advanced while the arm stopped, so resuming would cause a catch-up
        // burst. Fail closed.
        RCLCPP_ERROR(
          logger_,
          "STATUS gap during motion (%.0f ms, max cmd %.3f rad/s): latched; lifecycle reset "
          "needed. %s", gapMs, lastMovingCmd_.load(), recovery.c_str());
      }
      return return_type::ERROR;
    }
    ++gapCount_;
    gapMaxMs_ = std::max(gapMaxMs_, gapMs);
    // Re-baseline the velocity estimate so the outage does not produce a velocity spike.
    // After a coast the last estimate is still the best guess of the arm's velocity; after a
    // hold the arm was stopped.
    const bool wasHold = statusHold_;
    for (std::size_t i = 0; i < 6; ++i) {
      pos[i] = latest.positionRad[i];
      if (wasHold) {
        vel[i] = 0.0;
      }
    }
    last_pos = latest.positionRad;
    lastPosStamp = latest.stamp;
    lastConsumedSequence = latest.sequence;
    statusHold_ = false;
    statusCoast_ = false;
    const std::uint64_t skipped = latest.counterSkipped - lastFreshCounterSkipped_;
    RCLCPP_INFO(
      logger_,
      "STATUS resumed after a gap of %.0f ms (tier reached: %s, CRI counter values skipped: "
      "%llu%s): control resumes (gaps so far: %llu, max %.0f ms)",
      gapMs, Safety::StatusHealthName(gapTier), static_cast<unsigned long long>(skipped),
      skipped == 0 ? ", i.e. delayed/retransmitted, nothing lost" : "",
      static_cast<unsigned long long>(gapCount_), gapMaxMs_);
  }
  lastFreshCounterSkipped_ = latest.counterSkipped;
  lastFreshStamp_ = latest.stamp;

  if (latest.sequence != lastConsumedSequence) {
    for (std::size_t i = 0; i < 6; ++i) {
      pos[i] = latest.positionRad[i];
    }

    std::array<double, 6> velocity{};
    if (
      Safety::EstimateVelocity(
        last_pos, lastPosStamp, latest.positionRad, latest.stamp, Safety::kMinVelocityInterval,
        velocity))
    {
      for (std::size_t i = 0; i < 6; ++i) {
        vel[i] = velocity[i];
      }
      last_pos = latest.positionRad;
      lastPosStamp = latest.stamp;
    }
    lastConsumedSequence = latest.sequence;
  }
  // Without a new STATUS, position and velocity are held.

  return return_type::OK;
}

return_type Rebel::write(const rclcpp::Time &, const rclcpp::Duration &)
{
  bool nonFinite = false;
  bool saturated = false;
  {
    double maxAbs = 0.0;
    for (const double command : vel_cmd) {
      maxAbs = std::max(maxAbs, std::isfinite(command) ? std::fabs(command) : INFINITY);
    }
    if (Safety::IsCommandMoving(maxAbs)) {
      lastMovingCmd_.store(maxAbs);
      lastMovingNs_.store(ToNs(Safety::SteadyClock::now()));
    }
  }
  for (std::size_t i = 0; i < 6; ++i) {
    const Safety::JogCommand command = Safety::JogPercentFromVelocity(vel_cmd[i]);
    nonFinite = nonFinite || command.nonFinite;
    saturated = saturated || command.saturated;
    if (!statusCoast_) {
      jog[i].store(command.percent);  // frozen during a STATUS coast (last jog keeps going)
    }
  }
  if (holdJogZero_ || statusHold_) {
    // Checked after the store so a concurrent ResetAndEnable() can never be overtaken.
    ZeroJog();
  }
  lastWriteNs = ToNs(Safety::SteadyClock::now());

  if (nonFinite) {
    RCLCPP_ERROR_THROTTLE(
      logger_, throttleClock_, 1000,
      "Non-finite joint velocity command; sending zero jog for the affected joint(s)");
  }
  if (saturated) {
    RCLCPP_WARN_THROTTLE(
      logger_, throttleClock_, 1000, "Joint velocity command saturated at +-%.0f %% jog",
      Safety::kMaxJogPercent);
  }
  return return_type::OK;
}

bool Rebel::ResetAndEnable(std::string & message)
{
  if (stopping || !rebelSocket) {
    message = "driver is stopping";
    return false;
  }
  if (hardwareErrorLatched_) {
    message = "refused: hardware error is latched; recover with 'ros2 control "
      "set_hardware_component_state " + componentName_ + " inactive' then 'active'";
    return false;
  }
  JointSnapshot latest;
  {
    std::lock_guard<std::mutex> lock(snapshotLock);
    latest = snapshot;
  }
  const auto now = Safety::SteadyClock::now();
  if (latest.sequence == 0 ||
    Safety::IsStatusStale(latest.stamp, now, Safety::kStatusStaleTimeout))
  {
    message = "refused: STATUS is stale or missing";
    return false;
  }
  if (rebelSocket->ConnectionCount() != initialConnectionCount) {
    message = "refused: connection was re-established; recover via the hardware lifecycle reset";
    return false;
  }
  if (holdJogZero_.exchange(true)) {
    message = "refused: another reset_and_enable is in progress";
    return false;
  }

  RCLCPP_WARN(
    logger_, "reset_and_enable requested (drive mask 0x%02x): re-sending start-up sequence with "
    "jog held at zero", drivesDisabledMask_.load());
  ZeroJog();
  OnConnected(false);

  // Keep jog at zero until the drives report enabled (Enable takes ~1 s on the real robot).
  const auto deadline = Safety::SteadyClock::now() + std::chrono::seconds(3);
  while (drivesDisabledMask_.load() != 0 && Safety::SteadyClock::now() < deadline && !stopping) {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  const unsigned mask = drivesDisabledMask_.load();
  ZeroJog();
  holdJogZero_ = false;

  if (mask != 0) {
    message = "start-up sequence re-sent but drives still report 'Motor not enabled' (mask 0x" +
      std::to_string(mask) + "); check the robot";
    RCLCPP_ERROR(logger_, "%s", message.c_str());
    return false;
  }
  message = "start-up sequence re-sent; drives enabled. Jog was held at zero. Cycle the "
    "trajectory controller if it holds a stale goal.";
  RCLCPP_INFO(logger_, "%s", message.c_str());
  return true;
}

void Rebel::dio_callback(
  const std::shared_ptr<igus_rebel_msgs::srv::SetDigitalOutput::Request> request,
  std::shared_ptr<igus_rebel_msgs::srv::SetDigitalOutput::Response> response)
{
  response->success = SetDigitalOut(request->output.output, request->output.is_on);
  if (!response->success) {
    response->message = "not connected to the robot";
  }
}

void Rebel::Start()
{
  stopping = false;

  continueMessage = true;
  messageThread = std::thread(&Rebel::MessageThreadFunction, this);

  // The ALIVEJOG thread only transmits once OnConnected() has sent SetActive/Reset/Enable.
  aliveTransmit = false;
  continueAlive = true;
  aliveThread = std::thread(&Rebel::AliveThreadFunction, this);

  // Connects in the background; OnConnected() sends the start-up command sequence.
  rebelSocket->Start();
}

void Rebel::Stop()
{
  if (stopped.exchange(true)) {
    return;
  }
  stopping = true;

  // Service callbacks must not send commands while shutting down.
  if (executor_) {
    executor_->cancel();
  }
  if (executorThread.joinable()) {
    executorThread.join();
  }

  ZeroJog();
  lastWriteNs = 0;

  if (rebelSocket) {
    rebelSocket->DisableReconnect();
  }

  const bool connected = rebelSocket && rebelSocket->IsConnected();
  if (connected && aliveTransmit) {
    // Let at least one zero ALIVEJOG go out.
    std::this_thread::sleep_for(2 * Safety::kAlivePeriod);
  }

  continueAlive = false;
  aliveWake.notify_all();
  if (aliveThread.joinable()) {
    aliveThread.join();
  }

  if (rebelSocket && rebelSocket->IsConnected()) {
    RCLCPP_INFO(logger_, "Stopping: sending Disable and Quit to the ReBeL");
    Command(CriKeywords::COMMAND_DISABLE);
    Command(CriKeywords::COMMAND_QUIT);
  } else {
    RCLCPP_WARN(logger_, "Stopping while disconnected: Disable and Quit not sent");
  }

  if (rebelSocket) {
    rebelSocket->Stop();
  }

  continueMessage = false;
  if (messageThread.joinable()) {
    messageThread.join();
  }
}
}  // namespace Igus

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(
  Igus::Rebel, SystemInterface);
