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

  lastKinstate = currentKinstate;
  lastErrorJoints = currentErrorJoints;
}

//
// public functions
//
CallbackReturn Rebel::on_init(const HardwareInfo & hardware_info)
{
  if (SystemInterface::on_init(hardware_info) != CallbackReturn::SUCCESS) {
    return CallbackReturn::ERROR;
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

  ZeroJog();
  lastWriteNs = 0;
  current_ccnt = 1;
  lastKinstate = CriMessages::Kinstate::NO_ERROR;
  lastErrorJoints.fill(0);
  kinstateMessage = "";
  haveOverride = false;
  stopping = false;
  stopped = false;

  node_ = std::make_shared<rclcpp::Node>("igus_rebel");
  digital_output_srv_ = node_->create_service<igus_rebel_msgs::srv::SetDigitalOutput>(
    "set_digital_output",
    std::bind(&Rebel::dio_callback, this, std::placeholders::_1, std::placeholders::_2));
  // Service callbacks run on their own executor thread instead of inside write().
  executor_ = std::make_shared<rclcpp::executors::SingleThreadedExecutor>();
  executor_->add_node(node_);
  executorThread = std::thread([this]() {executor_->spin();});

  rebelSocket = std::make_shared<RebelSocket>(ip, port);
  rebelSocket->SetConnectedCallback([this](bool reconnect) {OnConnected(reconnect);});

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
  lastConsumedSequence = first.sequence;

  RCLCPP_INFO(
    logger_,
    "First STATUS: joints [deg] %.2f %.2f %.2f %.2f %.2f %.2f, override %.1f %%",
    first.positionRad[0] / degToRad, first.positionRad[1] / degToRad,
    first.positionRad[2] / degToRad, first.positionRad[3] / degToRad,
    first.positionRad[4] / degToRad, first.positionRad[5] / degToRad, first.overridePercent);

  return CallbackReturn::SUCCESS;
}

CallbackReturn Rebel::on_activate(const rclcpp_lifecycle::State &)
{
  for (double & command : vel_cmd) {
    command = 0.0;
  }
  ZeroJog();
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
  ZeroJog();
  lastWriteNs = 0;
  RCLCPP_ERROR(
    logger_, "Hardware error: jog forced to zero. Restart the driver to resume control.");
  return CallbackReturn::SUCCESS;
}

std::vector<StateInterface> Rebel::export_state_interfaces()
{
  std::vector<StateInterface> state_interfaces;

  for (int i = 0; i < 6; ++i) {
    state_interfaces.emplace_back(
      StateInterface(
        JOINT_NAME[i], hardware_interface::HW_IF_POSITION, &pos[i]));
    state_interfaces.emplace_back(
      StateInterface(
        JOINT_NAME[i], hardware_interface::HW_IF_VELOCITY, &vel[i]));
  }

  return state_interfaces;
}

std::vector<CommandInterface> Rebel::export_command_interfaces()
{
  std::vector<CommandInterface> command_interfaces;

  for (int i = 0; i < 6; ++i) {
    command_interfaces.emplace_back(
      CommandInterface(
        JOINT_NAME[i], hardware_interface::HW_IF_VELOCITY, &vel_cmd[i]));
  }

  return command_interfaces;
}

return_type Rebel::read(const rclcpp::Time &, const rclcpp::Duration &)
{
  JointSnapshot latest;
  {
    std::lock_guard<std::mutex> lock(snapshotLock);
    latest = snapshot;
  }

  const auto now = Safety::SteadyClock::now();
  if (
    latest.sequence == 0 ||
    Safety::IsStatusStale(latest.stamp, now, Safety::kStatusStaleTimeout))
  {
    // Returning ERROR makes ros2_control call on_error() and stop calling read()/write()
    // for this component (Humble 2.54: the component goes to UNCONFIGURED; controllers
    // are not deactivated). on_error() and the watchdog then keep the jog at zero.
    ZeroJog();
    const double ageMs = latest.sequence == 0 ?
      -1.0 : std::chrono::duration<double, std::milli>(now - latest.stamp).count();
    RCLCPP_ERROR_THROTTLE(
      logger_, throttleClock_, 1000,
      "No valid STATUS from ReBeL for %.0f ms (limit %lld ms): read() returns ERROR", ageMs,
      static_cast<long long>(Safety::kStatusStaleTimeout.count()));
    return return_type::ERROR;
  }

  if (rebelSocket->ConnectionCount() != initialConnectionCount) {
    // The connection dropped and was re-established (OnConnected re-enabled the robot).
    // Do not let the controllers resume motion on their own: fail closed like a stale
    // STATUS. The operator restarts the driver after checking the robot.
    ZeroJog();
    RCLCPP_ERROR_THROTTLE(
      logger_, throttleClock_, 1000,
      "Connection to ReBeL was re-established after a drop: read() returns ERROR so motion "
      "does not resume automatically. Restart the driver to continue.");
    return return_type::ERROR;
  }

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
  for (std::size_t i = 0; i < 6; ++i) {
    const Safety::JogCommand command = Safety::JogPercentFromVelocity(vel_cmd[i]);
    nonFinite = nonFinite || command.nonFinite;
    saturated = saturated || command.saturated;
    jog[i].store(command.percent);
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
