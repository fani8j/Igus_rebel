#include "igus_rebel/Rebel.hpp"

#include <arpa/inet.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
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
// ~/hardware_state: the timer polls for flag changes this often and publishes at least every
// kHardwareStatePeriod (10 Hz).
constexpr std::chrono::milliseconds kTelemetryPollPeriod{10};
constexpr std::chrono::milliseconds kHardwareStatePeriod{100};

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

void Rebel::LatchHardwareError(const std::string & reason)
{
  // Latch first, then zero: write() re-checks the latch after storing a command.
  const bool wasLatched = hardwareErrorLatched_.exchange(true);
  if (!wasLatched) {
    SetLatchReason(reason);  // telemetry only (~/hardware_state); the first cause is kept
  }
  ZeroJog();
  // Zero jog does not stop a robot-side joint move; only CMD Move Stop does.
  RequestMoveStop();
  if (wasLatched || !faultMember_) {
    // Already latched (by an earlier root cause, or by a peer): not a new root-cause fault.
    return;
  }
  const std::size_t stopped = ReportRootFault(faultMember_, reason);
  RCLCPP_ERROR(
    logger_,
    "Fault group '%s': component '%s' faulted (%s); stopped %zu other group member(s). This "
    "component is latched as well (jog zero, read() returns ERROR). %s",
    faultGroup_.c_str(), componentName_.c_str(), reason.c_str(), stopped,
    FaultGroupRecoveryHint().c_str());
}

void Rebel::OnPeerFault(const std::string & originName, const std::string & reason)
{
  if (stopping) {
    return;
  }
  // Latch first, then zero (see LatchHardwareError). Takes no lock: it may run on any thread
  // that reports a fault, including the message thread of the peer.
  peerFaulted_ = true;
  const bool wasLatched = hardwareErrorLatched_.exchange(true);
  if (!wasLatched) {
    // Leaf lock held only for a string copy by the telemetry timer: cannot deadlock the caller.
    SetLatchReason("fault-group member '" + originName + "' faulted: " + reason);
  }
  ZeroJog();
  lastWriteNs = 0;
  RequestMoveStop();
  if (wasLatched) {
    return;
  }
  RCLCPP_ERROR(
    logger_,
    "Hardware component '%s' stopped because fault-group member '%s' faulted: %s. Jog forced to "
    "zero and latched; recover '%s' FIRST, then this component: for each, "
    "'ros2 control set_hardware_component_state <component> inactive' then '... active' "
    "(refused unless STATUS is fresh, and refused here while '%s' is still faulted), then cycle "
    "the controllers ('%s').",
    componentName_.c_str(), originName.c_str(), reason.c_str(), originName.c_str(),
    originName.c_str(), trajectoryControllerName_.c_str());
}

void Rebel::SetLatchReason(const std::string & reason)
{
  {
    std::lock_guard<std::mutex> lock(latchReasonLock_);
    latchReason_ = reason;
  }
  latchVersion_.fetch_add(1);
}

void Rebel::ClearLatchReason()
{
  {
    std::lock_guard<std::mutex> lock(latchReasonLock_);
    latchReason_.clear();
  }
  latchVersion_.fetch_add(1);
}

std::string Rebel::FaultGroupRecoveryHint() const
{
  if (!faultMember_) {
    return "";
  }
  return "Fault group '" + faultGroup_ + "' recovery order: recover the member that faulted "
         "first (the root cause) before the others. A member's activation is refused while "
         "another member holds an un-recovered root-cause fault. Every latched member "
         "(including the root cause) recovers the same way: 'ros2 control "
         "set_hardware_component_state <component> inactive' then 'active', then cycle its "
         "controllers; the reset_and_enable service alone is NOT the recovery path in a fault "
         "group (it is refused while latched). The root cause counts as recovered after "
         "500 ms of continuous enabled, clean STATUS. A root-cause member that cannot recover "
         "(for example a persistent kinematics error) blocks its peers until the stack is "
         "relaunched without fault_group.";
}

void Rebel::CheckFaultGroupStatus(
  CriMessages::Kinstate kinstate, const std::array<int, 16> & errorJoints)
{
  if (!faultMember_ || stopping) {
    return;
  }
  // Any known per-joint error bit of joints 1-6 (over-temperature, supply/e-stop, motor not
  // enabled, communication watchdog, position lag, encoder, over-current, driver) or any CRI
  // kinematics error state means the controller stopped this arm.
  constexpr int kKnownJointErrorBits = 0xFF;
  std::string jointText;
  for (std::size_t i = 0; i < 6; ++i) {
    const int bits = errorJoints.at(i) & kKnownJointErrorBits;
    if (bits != 0) {
      char buffer[40];
      std::snprintf(buffer, sizeof(buffer), "joint %zu error bits 0x%02x", i + 1, bits);
      jointText += (jointText.empty() ? "" : ", ") + std::string(buffer);
      const std::string described = CriMessages::DescribeJointErrorBits(bits);
      if (!described.empty()) {
        jointText += " [" + described + "]";
      }
    }
  }
  const bool kinError = Safety::IsGroupTrippingKinstate(kinstate);
  const auto nowTime = Safety::SteadyClock::now();
  const std::int64_t nowNs = ToNs(nowTime);
  const bool gap = lastCheckedStatusNs_ != 0 &&
    nowTime - FromNs(lastCheckedStatusNs_) > Safety::kStatusStaleTimeout;
  lastCheckedStatusNs_ = nowNs;

  if (jointText.empty() && !kinError) {
    // Track the start of the current clean run: restarted by any non-clean STATUS (below), by
    // OnConnected and by a STATUS gap (a run with missing STATUS is not continuous).
    std::int64_t cleanSince = cleanSinceNs_.load();
    if (cleanSince == 0 || gap) {
      cleanSince = nowNs;
      cleanSinceNs_ = cleanSince;
    }
    const bool cleanLongEnough = nowTime - FromNs(cleanSince) >= Safety::kFaultArmSettleTime;
    if (!faultArmed_) {
      const std::int64_t connectedAt = connectedAtNs_.load();
      if (cleanLongEnough && connectedAt != 0 &&
        nowTime - FromNs(connectedAt) >= Safety::kFaultArmSettleTime)
      {
        faultArmed_ = true;
        RCLCPP_INFO(
          logger_, "Fault group '%s': drives enabled and STATUS clean; monitoring armed",
          faultGroup_.c_str());
      }
    }
    if (faultArmed_ && cleanLongEnough && !hardwareErrorLatched_ &&
      faultMember_->HasRootFault() && faultMember_->ClearRootFault())
    {
      RCLCPP_WARN(
        logger_,
        "Fault group '%s': '%s' is enabled with clean STATUS again; its root-cause fault is "
        "cleared, other members may now be recovered.",
        faultGroup_.c_str(), componentName_.c_str());
    }
    return;
  }

  cleanSinceNs_ = 0;  // a non-clean STATUS ends the clean run
  // Not armed (normal disabled state before the first enable, or a start-up sequence still
  // being processed) or already latched (then this is not a new root cause): ignore.
  if (!faultArmed_ || hardwareErrorLatched_) {
    return;
  }
  faultArmed_ = false;
  std::string reason = "robot controller reports a stop after the arm was enabled: ";
  if (!jointText.empty()) {
    reason += jointText;
  }
  if (kinError) {
    reason += (jointText.empty() ? "" : "; ") +
      std::string("kinematics error [") + CriMessages::KinstateToString(kinstate) + "]";
  }
  // Fail closed: this arm is latched too (jog zero, read() ERROR), so its own controller
  // cannot resume an old trajectory after a reset_and_enable while the peers are down. It
  // recovers like every other latch (inactive -> active, then cycle the controllers).
  LatchHardwareError(reason);
}

void Rebel::AliveThreadFunction()
{
  bool announced = false;
  // Until the controller manager calls write() the command counts as stale (zero jog).
  bool wasStale = true;
  auto next = Safety::SteadyClock::now();

  while (continueAlive) {
    ConsumeMoveStopRequest();
    if (aliveTransmit) {
      if (!announced) {
        RCLCPP_INFO(logger_, "Starting to send ALIVEJOG");
        announced = true;
      }

      const auto now = Safety::SteadyClock::now();
      const bool stale = Safety::IsCommandStale(
        FromNs(lastWriteNs.load()), now, Safety::kCommandWatchdogTimeout);

      std::array<float, 6> currentJog{};
      // Transmit-boundary inhibit: never send a non-zero jog while latched or held, even if a
      // write() stored one just before the latch/hold took effect. A STATUS coast (not latched,
      // not held) keeps sending the frozen jog.
      // While a joint move is active jog is always zero (the robot owns the motion).
      const bool inhibited = hardwareErrorLatched_.load() || holdJogZero_.load() ||
        statusHold_.load() || moveState_.load() != kMoveIdle || moveStopRequested_.load();
      if (!stale && !inhibited) {
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
    // A Move Stop request ends the wait early so it goes out within a few ms; the ALIVEJOG
    // period itself is not shifted (the wait resumes until `next`).
    while (continueAlive) {
      const bool woke = aliveWake.wait_until(
        lock, next, [this]() {return !continueAlive || MoveStopDue();});
      if (!woke || !continueAlive) {
        break;
      }
      lock.unlock();
      ConsumeMoveStopRequest();
      lock.lock();
      if (Safety::SteadyClock::now() >= next) {
        break;
      }
    }
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
        JointSnapshot published;
        {
          std::lock_guard<std::mutex> lock(snapshotLock);
          snapshot.positionRad = positionRad;
          snapshot.stamp = Safety::SteadyClock::now();
          snapshot.systemStampNs = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
          snapshot.overridePercent = status.overrideValue;
          snapshot.counterSkipped = counterSkipped_;
          ++snapshot.sequence;
          published = snapshot;
        }
        snapshotCondition.notify_all();
        // Once per NEW sample, stamped with its receive time; independent of read() and of the
        // latch (see hardwareErrorLatched_).
        PublishRobotJointStates(published);
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

    case CriMessages::MessageType::MOVETOEXECACK:
    case CriMessages::MessageType::MOVETOEXECEND:
    case CriMessages::MessageType::MOVETOEXECERROR:
    case CriMessages::MessageType::EXECERROR:
      {
        const auto & event = std::get<CriMessages::ExecEvent>(parsed.payload);
        RecordMoveEvent(event, parsed.type);
        break;
      }

    case CriMessages::MessageType::EXECACK:
    case CriMessages::MessageType::EXECEND:
      RCLCPP_DEBUG(logger_, "Ignoring program execution message: %s", Truncate(msg).c_str());
      break;

    case CriMessages::MessageType::CMDERROR:
      {
        const auto & error = std::get<CriMessages::CmdError>(parsed.payload);
        if (error.recjectedCmd == moveCommandCounter_.load() && moveGoalBusy_.load()) {
          std::lock_guard<std::mutex> eventLock(moveEventLock_);
          moveEvents_.cmdError = true;
          moveEvents_.cmdErrorText = error.error;
        }
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
        if (ack.acceptedCmd == moveCommandCounter_.load() && moveGoalBusy_.load()) {
          std::lock_guard<std::mutex> eventLock(moveEventLock_);
          moveEvents_.cmdAck = true;
        }
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

bool Rebel::Command(const std::string & command, bool quiet)
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
    if (!quiet) {
      RCLCPP_WARN(logger_, "Could not send command '%s': not connected", command.c_str());
    }
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
  // Drive state is in flux until the sequence below took effect: do not monitor it meanwhile.
  faultArmed_ = false;
  connectedAtNs_ = 0;
  cleanSinceNs_ = 0;
  {
    std::lock_guard<std::mutex> lock(commandsLock);
    unacknowledgedCommands.clear();
  }

  // Stop() may run concurrently (it disables reconnects first); do not send anything then.
  const auto send = [this](const std::string & command) {
      return !stopping && Command(command);
    };

  // A robot-side joint move survives a lost link, a closed socket and zero jog, and the
  // controller only accepts SetActive/Reset/Enable while idle: stop whatever may still be
  // running, unconditionally, before anything else. A pending stop request is NOT cleared by
  // this (a written command is not a confirmation): the ALIVEJOG thread keeps re-sending until
  // STATUS shows standstill. It cannot interleave with the sequence below (it skips its send
  // while startupLock_ is held).
  send(CriKeywords::COMMAND_MOVE_STOP);

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
      "Reconnected to ReBeL: sent Move Stop, then re-sent SetActive, Reset, Enable, "
      "MotionTypeJoint and Override");
  }
  RCLCPP_INFO(logger_, "Rebel now controlled by velocity control.");
  connectedAtNs_ = ToNs(Safety::SteadyClock::now());
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
  kinstateValue_.store(static_cast<int>(currentKinstate));
  CheckFaultGroupStatus(currentKinstate, currentErrorJoints);

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
  // Joint move caps: lowering is allowed, exceeding the hard caps is not (clamped, with a warning).
  const auto readPercent = [&](const char * key, double hardCap, double & target) {
      const auto it = params.find(key);
      if (it == params.end() || it->second.empty()) {
        return true;
      }
      double parsed = 0.0;
      try {
        std::size_t used = 0;
        parsed = std::stod(it->second, &used);
        if (used != it->second.size()) {
          parsed = std::nan("");
        }
      } catch (const std::exception &) {
        parsed = std::nan("");
      }
      if (!std::isfinite(parsed) || parsed <= 0.0) {
        RCLCPP_FATAL(logger_, "Invalid %s '%s' (expected a number > 0)", key, it->second.c_str());
        return false;
      }
      if (parsed > hardCap) {
        RCLCPP_WARN(
          logger_, "%s %.1f exceeds the hard cap %.1f: using %.1f", key, parsed, hardCap, hardCap);
        parsed = hardCap;
      }
      target = parsed;
      return true;
    };
  moveMaxVelocityPercent_ = Move::kDefaultMaxVelocityPercent;
  moveMaxAccelerationPercent_ = Move::kDefaultMaxAccelerationPercent;
  if (!readPercent(
      "move_joint_max_velocity_percent", Move::kHardMaxVelocityPercent, moveMaxVelocityPercent_) ||
    !readPercent(
      "move_joint_max_acceleration_percent", Move::kHardMaxAccelerationPercent,
      moveMaxAccelerationPercent_))
  {
    return CallbackReturn::ERROR;
  }
  std::string faultGroup;
  readName("fault_group", faultGroup);
  const bool validGroup = std::all_of(
    faultGroup.begin(), faultGroup.end(), [](unsigned char c) {
      return std::isalnum(c) != 0 || c == '_' || c == '.' || c == '-';
    });
  if (!validGroup) {
    RCLCPP_FATAL(
      logger_, "Invalid fault_group '%s' (allowed: letters, digits, '_', '.', '-')",
      faultGroup.c_str());
    return CallbackReturn::ERROR;
  }

  // Validate the ros2_control description before anything is started: the interfaces are
  // exported by joint position (info_.joints[i] <-> CRI joint i + 1), so a wrong count or
  // order would send commands to the wrong joint.
  if (!ValidateJoints(hardware_info)) {
    return CallbackReturn::ERROR;
  }
  jointLimits_ = Move::HardwareJointLimits();
  for (std::size_t i = 0; i < 6; ++i) {
    jointNames_[i] = hardware_info.joints[i].name;
    // The ros2_control tag normally carries no position limits (the URDF <limit> is not passed
    // to the plugin); if the position state interface declares min/max, those win.
    for (const auto & state : hardware_info.joints[i].state_interfaces) {
      if (state.name == hardware_interface::HW_IF_POSITION && !state.min.empty() &&
        !state.max.empty())
      {
        try {
          const double lower = std::stod(state.min);
          const double upper = std::stod(state.max);
          if (std::isfinite(lower) && std::isfinite(upper) && lower < upper) {
            jointLimits_[i] = {lower, upper};
          }
        } catch (const std::exception &) {
        }
      }
    }
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
  ClearLatchReason();

  // The member exists before the message thread starts; it joins the group at the end of a
  // successful on_init (before that no peer can see or call it).
  if (faultMember_) {
    LeaveFaultGroup(faultMember_);
    faultMember_.reset();
  }
  faultGroup_ = faultGroup;
  faultArmed_ = false;
  connectedAtNs_ = 0;
  cleanSinceNs_ = 0;
  lastCheckedStatusNs_ = 0;
  if (!faultGroup_.empty()) {
    faultMember_ = std::make_shared<FaultGroupMember>(
      faultGroup_, componentName_.empty() ? nodeName_ : componentName_,
      [this](const std::string & origin, const std::string & reason) {
        OnPeerFault(origin, reason);
      });
  }

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
  moveState_ = kMoveIdle;
  moveGoalBusy_ = false;
  moveCancelRequested_ = false;
  moveStopRequested_ = false;
  peerFaulted_ = false;
  activated_ = false;
  kinstateValue_ = 0;
  moveCommandCounter_ = -1;
  velocityClaimMask_ = 0;
  claimPendingUntilNs_ = 0;
  claimDuringMove_ = false;
  {
    std::lock_guard<std::mutex> lock(moveEventLock_);
    moveEvents_ = MoveEvents{};
  }

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
    // Own callback group: a cancel or goal request is never blocked by reset_and_enable.
    moveGroup_ = node_->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    // Telemetry that keeps running while the component is latched/UNCONFIGURED/INACTIVE (read()
    // is not called then). robot_joint_states: sensor-data QoS (best effort) -- subscribe with
    // rclcpp::SensorDataQoS(); hardware_state: reliable.
    robotJointStatePub_ = node_->create_publisher<sensor_msgs::msg::JointState>(
      "~/robot_joint_states", rclcpp::SensorDataQoS().keep_last(10));
    hardwareStatePub_ = node_->create_publisher<igus_rebel_msgs::msg::HardwareState>(
      "~/hardware_state", rclcpp::QoS(rclcpp::KeepLast(10)).reliable());
    // Own callback group: a 3 s reset_and_enable never stalls the hardware_state timer.
    telemetryGroup_ = node_->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    hardwareStateTimer_ = node_->create_wall_timer(
      kTelemetryPollPeriod, [this]() {PublishHardwareState();}, telemetryGroup_);
    moveServer_ = rclcpp_action::create_server<MoveJointAction>(
      node_, "~/move_joint",
      [this](const rclcpp_action::GoalUUID & uuid,
      std::shared_ptr<const MoveJointAction::Goal> goal) {
        return OnMoveGoal(uuid, std::move(goal));
      },
      [this](const std::shared_ptr<MoveGoalHandle> goalHandle) {return OnMoveCancel(goalHandle);},
      [this](const std::shared_ptr<MoveGoalHandle> goalHandle) {OnMoveAccepted(goalHandle);},
      rcl_action_server_get_default_options(), moveGroup_);
  } catch (const std::exception & error) {
    RCLCPP_FATAL(
      logger_, "Cannot create ROS node '%s' or its services ('%s'): %s", nodeName_.c_str(),
      digitalOutputService_.c_str(), error.what());
    moveServer_.reset();
    moveGroup_.reset();
    hardwareStateTimer_.reset();
    telemetryGroup_.reset();
    hardwareStatePub_.reset();
    robotJointStatePub_.reset();
    digital_output_srv_.reset();
    reset_enable_srv_.reset();
    node_.reset();
    return CallbackReturn::ERROR;
  }

  // The socket exists before the executor starts, so service callbacks never see it unset.
  rebelSocket = std::make_shared<RebelSocket>(ip, port, logger_);
  rebelSocket->SetConnectedCallback([this](bool reconnect) {OnConnected(reconnect);});

  // Service/action callbacks run on their own executor threads instead of inside write().
  // Three threads: reset_and_enable (up to 3 s) occupies one, the move_joint callbacks and the
  // hardware_state timer (separate callback groups) always find a free one.
  executor_ = std::make_shared<rclcpp::executors::MultiThreadedExecutor>(
    rclcpp::ExecutorOptions(), 3);
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

  if (faultMember_ && JoinFaultGroup(faultMember_)) {
    RCLCPP_INFO(
      logger_, "Joined fault group '%s' (%zu member(s)): a fault of any member stops the others",
      faultGroup_.c_str(), FaultGroupSize(faultGroup_));
    BlockingFault blocking;
    if (FindBlockingRootFault(faultMember_, blocking)) {
      // A peer is already faulted: start latched; activation follows the recovery rules.
      OnPeerFault(blocking.member, blocking.reason);
    }
  }

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
      "unless STATUS is fresh.%s%s",
      componentName_.c_str(), componentName_.c_str(), faultMember_ ? " " : "",
      FaultGroupRecoveryHint().c_str());
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
  activated_ = false;

  if (hardwareErrorLatched_) {
    BlockingFault blocking;
    if (faultMember_ && FindBlockingRootFault(faultMember_, blocking)) {
      RCLCPP_ERROR(
        logger_,
        "Refusing to activate hardware component '%s': fault-group '%s' member '%s' still "
        "holds an un-recovered root-cause fault (%s). Recover '%s' first ('ros2 control "
        "set_hardware_component_state %s inactive' then 'active'; reset_and_enable alone is not "
        "enough); a recovered member counts as healthy "
        "after %lld ms of enabled, clean STATUS, so retry shortly after. Then retry this "
        "component.",
        componentName_.c_str(), faultGroup_.c_str(), blocking.member.c_str(),
        blocking.reason.c_str(), blocking.member.c_str(), blocking.member.c_str(),
        static_cast<long long>(Safety::kFaultArmSettleTime.count()));
      return CallbackReturn::FAILURE;
    }
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
    peerFaulted_ = false;
    hardwareErrorLatched_ = false;
    ClearLatchReason();
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

  activated_ = true;
  return CallbackReturn::SUCCESS;
}

CallbackReturn Rebel::on_deactivate(const rclcpp_lifecycle::State &)
{
  activated_ = false;
  for (double & command : vel_cmd) {
    command = 0.0;
  }
  ZeroJog();
  lastWriteNs = 0;
  RequestMoveStop();
  RCLCPP_INFO(logger_, "Hardware deactivated: jog set to zero");
  return CallbackReturn::SUCCESS;
}

CallbackReturn Rebel::on_error(const rclcpp_lifecycle::State &)
{
  // ros2_control stops calling read()/write() after an error; make sure the ALIVEJOG
  // thread sends zero from now on (the stale-command watchdog would do so after 50 ms).
  // The latch is fail-closed: it is never cleared here or by any automatic retry, only by
  // on_activate() after an operator-driven lifecycle reset confirms STATUS is fresh again.
  LatchHardwareError("on_error() was called for this component (read()/write() reported ERROR)");
  activated_ = false;
  ZeroJog();
  lastWriteNs = 0;
  RequestMoveStop();
  RCLCPP_ERROR(
    logger_,
    "Hardware error on component '%s': jog forced to zero and latched. Recover with "
    "'ros2 control set_hardware_component_state %s inactive' then "
    "'ros2 control set_hardware_component_state %s active' (the latter is refused unless "
    "STATUS is fresh); also cycle the controllers claiming this hardware (e.g. "
    "'ros2 control switch_controllers --deactivate %s --activate "
    "%s') so they do not resume from a stale setpoint.%s%s",
    componentName_.c_str(), componentName_.c_str(), componentName_.c_str(),
    trajectoryControllerName_.c_str(), trajectoryControllerName_.c_str(),
    faultMember_ ? " " : "", FaultGroupRecoveryHint().c_str());
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
    RequestMoveStop();
    RCLCPP_ERROR_THROTTLE(
      logger_, throttleClock_, 1000,
      "Hardware error latched on component '%s': read() returns ERROR until an operator "
      "recovers it ('ros2 control set_hardware_component_state %s inactive' then "
      "'... active'; the latter is refused unless STATUS is fresh).%s%s",
      componentName_.c_str(), componentName_.c_str(), faultMember_ ? " " : "",
      FaultGroupRecoveryHint().c_str());
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
    " inactive' then '... active' (refused unless STATUS is fresh)." +
    (faultMember_ ? " " + FaultGroupRecoveryHint() : std::string());

  if (rebelSocket->ConnectionCount() != initialConnectionCount) {
    // The connection dropped and was re-established (OnConnected re-enabled the robot).
    // Do not let the controllers resume motion on their own: fail closed. An operator
    // lifecycle reset resumes control after checking the robot.
    ZeroJog();
    RequestMoveStop();
    LatchHardwareError("connection to the ReBeL was re-established after a drop");
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
    const double ageMs = latest.sequence == 0 ?
      -1.0 : std::chrono::duration<double, std::milli>(age).count();
    ZeroJog();
    RequestMoveStop();
    {
      char buffer[96];
      std::snprintf(
        buffer, sizeof(buffer), "no valid STATUS for %.0f ms (latch limit %lld ms)", ageMs,
        static_cast<long long>(Safety::kStatusLatchTimeout.count()));
      LatchHardwareError(buffer);
    }
    statusHold_ = false;
    statusCoast_ = false;
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
      // The 400 ms hold tier and above must stop a robot-side move (zero jog does not).
      RequestMoveStop();
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
      {
        char buffer[112];
        std::snprintf(
          buffer, sizeof(buffer), "STATUS resumed after a %.0f ms gap (%s)", gapMs,
          gapTier == Safety::StatusHealth::Latch ? "beyond the latch limit" :
          "motion was commanded during the hold");
        LatchHardwareError(buffer);
      }
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
  // While a joint move is active the robot owns the motion: jog is zero whatever the
  // controllers command, and the cycle counts as "moving" so that any STATUS gap during the
  // move takes the stop + latch path (never an auto-resume).
  const bool moveActive = moveState_.load() != kMoveIdle;
  if (moveActive) {
    lastMovingCmd_.store(0.0);
    lastMovingNs_.store(ToNs(Safety::SteadyClock::now()));
  }
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
  // Decide the inhibit BEFORE storing so a non-zero jog is never published while latched/held.
  // A pending (unconfirmed) Move Stop also inhibits: after a lifecycle recovery the robot may
  // still be running the old move, and controller jog on top of it must wait for the standstill
  // confirmation (ConsumeMoveStopRequest).
  const bool stopPending = moveStopRequested_.load();
  const bool inhibited =
    holdJogZero_ || statusHold_ || hardwareErrorLatched_ || moveActive || stopPending;
  if (stopPending && !moveActive) {
    RCLCPP_WARN_THROTTLE(
      logger_, throttleClock_, 2000,
      "Holding jog at zero: a CMD Move Stop is not confirmed yet (waiting for standstill)");
  }
  for (std::size_t i = 0; i < 6; ++i) {
    const Safety::JogCommand command = Safety::JogPercentFromVelocity(vel_cmd[i]);
    nonFinite = nonFinite || command.nonFinite;
    saturated = saturated || command.saturated;
    if (inhibited) {
      jog[i].store(0.0f);
    } else if (!statusCoast_) {
      jog[i].store(command.percent);  // frozen during a STATUS coast (last jog keeps going)
    }
  }
  if (holdJogZero_ || statusHold_ || hardwareErrorLatched_ ||
    moveState_.load() != kMoveIdle || moveStopRequested_.load())
  {
    // Re-checked after the store so a concurrent ResetAndEnable() or a latch set by another thread
    // (fault-group stop, message thread) can never be overtaken.
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
  if (moveState_.load() != kMoveIdle || moveGoalBusy_.load()) {
    message = "refused: a joint move (move_joint action) is active";
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

//
// Robot-side joint move (CRI "CMD Move Joint")
//
// The controller plans and executes a synchronized joint-space line from the current joints to
// the goal. Once the command is sent the robot owns the motion: zero jog, a paused ALIVEJOG and a
// closed socket do NOT stop it, only "CMD Move Stop" (or an e-stop) does. Every fault path of this
// driver therefore ends in RequestMoveStop().
//
void Rebel::RequestMoveStop()
{
  // Lock-free and allocation-free: called from read()/write() threads, the fault-group callback
  // of another arm's message thread and the lifecycle callbacks. A stop while no move is active
  // is not needed (zero jog does the job) and must not change the idle command stream.
  if (moveState_.load() == kMoveIdle) {
    return;
  }
  moveStopSeq_.fetch_add(1);
  if (!moveStopRequested_.exchange(true)) {
    moveStopTimer_.ArmNow();  // a new request goes out at once
  }
  aliveWake.notify_all();
}

bool Rebel::SendMoveStopNow()
{
  return Command(CriKeywords::COMMAND_MOVE_STOP, true);
}

bool Rebel::MoveStopDue() const
{
  return moveStopRequested_.load() && moveStopTimer_.Due(ToNs(Safety::SteadyClock::now()));
}

// Runs on the ALIVEJOG thread only. The request is NOT consumed by a successful send: the
// controller may have missed or ignored the command, so it is re-sent every
// kMoveStopResendPeriod until the robot is known to stand still.
void Rebel::ConsumeMoveStopRequest()
{
  if (!moveStopRequested_.load()) {
    stopStillRefValid_ = false;
    stopMotionSeen_ = false;
    stopDrivesOff_ = false;
    return;
  }
  const auto now = Safety::SteadyClock::now();

  if (moveState_.load() == kMoveIdle) {
    // No move thread to confirm the stop (e.g. an unconfirmed stop that latched the driver):
    // fresh STATUS showing no joint motion for kMoveStopStillWindow confirms it.
    const std::uint64_t seqBefore = moveStopSeq_.load();
    JointSnapshot latest;
    {
      std::lock_guard<std::mutex> lock(snapshotLock);
      latest = snapshot;
    }
    const bool fresh = latest.sequence != 0 &&
      !Safety::IsStatusStale(latest.stamp, now, Safety::kStatusStaleTimeout);
    // The way out for a stop that never saw motion or an END: every drive disabled (e-stop,
    // drives off) on fresh STATUS for the still window.
    bool drivesOffConfirmed = false;
    if (fresh && drivesDisabledMask_.load() == Safety::kAllDrivesDisabledMask) {
      if (!stopDrivesOff_) {
        stopDrivesOff_ = true;
        stopDrivesOffSince_ = now;
      }
      drivesOffConfirmed = now - stopDrivesOffSince_ >= Safety::kMoveStopStillWindow;
    } else {
      stopDrivesOff_ = false;
    }
    if (drivesOffConfirmed) {
      moveStopRequested_.store(false);
      if (moveStopSeq_.load() != seqBefore) {
        moveStopRequested_.store(true);  // a request raced the check: keep it
      } else {
        stopStillRefValid_ = false;
        stopMotionSeen_ = false;
        stopDrivesOff_ = false;
        RCLCPP_WARN(
          logger_, "Pending CMD Move Stop confirmed: drives disabled (e-stop) on %s",
          componentName_.c_str());
        return;
      }
    } else if (!fresh) {
      stopStillRefValid_ = false;
    } else if (!stopStillRefValid_ ||
      Move::MaxAbsDifference(latest.positionRad, stopStillRef_) > Safety::kMoveStillEpsilonRad)
    {
      if (stopStillRefValid_) {
        stopMotionSeen_ = true;  // the robot is (still) moving: standstill later is meaningful
      }
      stopStillRef_ = latest.positionRad;
      stopStillRefValid_ = true;
      stopStillSince_ = latest.stamp;
    } else if (latest.stamp - stopStillSince_ >= Safety::kMoveStopStillWindow &&
      (stopMotionSeen_ || moveMotionSeen_.load()))
    {
      // Standstill confirms only a robot that was seen moving (during the move, or now): a robot
      // that has not started yet, or never got the stop, would look the same.
      moveStopRequested_.store(false);
      if (moveStopSeq_.load() != seqBefore) {
        moveStopRequested_.store(true);  // a request raced the check: keep it
      } else {
        stopStillRefValid_ = false;
        stopMotionSeen_ = false;
        RCLCPP_INFO(logger_, "Pending CMD Move Stop confirmed by standstill");
        return;
      }
    }
  } else {
    stopStillRefValid_ = false;
  }

  if (!moveStopTimer_.Due(ToNs(now))) {
    return;
  }
  // Observe BEFORE sending: a request that arrives while this send runs re-arms "send now" and
  // the Defer below then fails instead of overwriting it.
  const auto token = moveStopTimer_.Observe();
  if (const auto hook = std::atomic_load(&beforeStopSendHook_)) {
    (*hook)();  // tests only: pauses here to widen the check-to-send window
  }
  bool sent = false;
  {
    // Serialized with the CMD Move Joint send: the request is re-checked under the lock, so a
    // stop that was confirmed (and its goal finished) while this thread was paused is not sent
    // behind the next goal's Move Joint.
    std::lock_guard<std::mutex> sendGuard(moveSendLock_);
    if (!moveStopRequested_.load()) {
      return;
    }
    // OnConnected() sends its own Move Stop first and then the start-up sequence; do not
    // interleave with it (it holds startupLock_). Retry shortly.
    std::unique_lock<std::mutex> startupGuard(startupLock_, std::try_to_lock);
    sent = startupGuard.owns_lock() && SendMoveStopNow();
  }
  // Not connected / start-up in progress: retry soon.
  moveStopTimer_.Defer(
    token, ToNs(now + (sent ? Safety::kMoveStopResendPeriod : std::chrono::milliseconds(5))));
}

void Rebel::RecordMoveEvent(const CriMessages::ExecEvent & event, CriMessages::MessageType type)
{
  const bool isError = type == CriMessages::MessageType::MOVETOEXECERROR ||
    type == CriMessages::MessageType::EXECERROR;
  const bool isEnd = type == CriMessages::MessageType::MOVETOEXECEND;
  const char * name = isError ?
    (type == CriMessages::MessageType::MOVETOEXECERROR ? "MOVETOEXECERROR" : "EXECERROR") :
    (isEnd ? "MOVETOEXECEND" : "MOVETOEXECACK");
  if (isError) {
    // Always logged at ERROR. The numeric code of an ERROR (0 for an e-stop) is not "success".
    RCLCPP_ERROR(
      logger_, "Robot reported %s: code %d sub %d program '%s' reason '%s'", name, event.code,
      event.sub, event.program.c_str(), event.reason.c_str());
    if (moveState_.load() == kMoveIdle) {
      // An unsolicited error with no joint move of ours: log only. It must never latch and
      // never leak into a later goal (nothing is recorded while idle).
      RCLCPP_WARN(logger_, "%s ignored: no joint move active", name);
      return;
    }
    // Recorded in ANY non-idle state (Sending/Moving/Stopping), without the "END before ACK"
    // gate: an error always concerns the move in flight (or a robot fault that makes it unsafe).
    std::lock_guard<std::mutex> lock(moveEventLock_);
    if (!moveEvents_.errorSeen) {
      moveEvents_.errorSeen = true;
      moveEvents_.errorType = name;
      moveEvents_.errorReason = event.reason;
    }
    return;
  }
  if (moveState_.load() == kMoveIdle) {
    RCLCPP_INFO(
      logger_, "Ignoring %s %d %d %s %s (no joint move active)", name, event.code, event.sub,
      event.program.c_str(), event.reason.c_str());
    return;
  }
  std::lock_guard<std::mutex> lock(moveEventLock_);
  if (isEnd) {
    if (!moveEvents_.moveAck) {
      // The ACK always precedes the END of the same move on the stream: this END belongs to an
      // earlier move.
      RCLCPP_WARN(logger_, "Ignoring MOVETOEXECEND before MOVETOEXECACK of the current move");
      return;
    }
    moveEvents_.endSeen = true;
    moveEvents_.endCode = event.code;
    moveEvents_.endReason = event.reason;
    RCLCPP_INFO(
      logger_, "Robot move ended: code %d reason '%s'", event.code, event.reason.c_str());
  } else {
    moveEvents_.moveAck = true;
  }
}

void Rebel::PublishRobotJointStates(const JointSnapshot & sample)
{
  if (!robotJointStatePub_) {
    return;
  }
  sensor_msgs::msg::JointState msg;
  // The acquisition time of THIS STATUS sample, never "now": a dead link shows as an ageing stamp.
  msg.header.stamp = rclcpp::Time(sample.systemStampNs, RCL_SYSTEM_TIME);
  msg.name.assign(jointNames_.begin(), jointNames_.end());
  msg.position.assign(sample.positionRad.begin(), sample.positionRad.end());

  // Velocity: finite difference of consecutive samples, as read() does. A gap longer than the
  // stale tier is not a velocity (the baseline restarts): such a sample carries no velocity.
  const std::int64_t stampNs = ToNs(sample.stamp);
  if (pubLastStampNs_ != 0) {
    if (sample.stamp - FromNs(pubLastStampNs_) > Safety::kStatusStaleTimeout) {
      pubVelValid_ = false;
      pubLastPos_ = sample.positionRad;
      pubLastStampNs_ = stampNs;
    } else {
      std::array<double, 6> velocity{};
      if (
        Safety::EstimateVelocity(
          pubLastPos_, FromNs(pubLastStampNs_), sample.positionRad, sample.stamp,
          Safety::kMinVelocityInterval, velocity))
      {
        pubVel_ = velocity;
        pubVelValid_ = true;
        pubLastPos_ = sample.positionRad;
        pubLastStampNs_ = stampNs;
      }
      // Too close to the previous sample: keep the previous estimate and baseline.
    }
  } else {
    pubLastPos_ = sample.positionRad;
    pubLastStampNs_ = stampNs;
    pubVelValid_ = false;
  }
  if (pubVelValid_) {
    msg.velocity.assign(pubVel_.begin(), pubVel_.end());
  }
  try {
    robotJointStatePub_->publish(msg);
  } catch (const std::exception & error) {
    RCLCPP_WARN_THROTTLE(
      logger_, throttleClock_, 5000, "Could not publish robot_joint_states: %s", error.what());
  }
}

void Rebel::PublishHardwareState()
{
  if (!hardwareStatePub_ || !rebelSocket) {
    return;
  }
  const bool latched = hardwareErrorLatched_.load();
  const bool stopPending = moveStopRequested_.load();
  const bool moveActive = moveState_.load() != kMoveIdle;
  const bool activated = activated_.load();
  const bool connected = rebelSocket->IsConnected();
  const std::uint64_t connectionCount = rebelSocket->ConnectionCount();
  const unsigned disabledMask = drivesDisabledMask_.load();
  const int kinstate = kinstateValue_.load();
  JointSnapshot latest;
  {
    std::lock_guard<std::mutex> lock(snapshotLock);
    latest = snapshot;
  }
  const auto steadyNow = Safety::SteadyClock::now();
  const bool haveStatus = latest.sequence != 0;
  const double ageS = haveStatus ? std::chrono::duration<double>(steadyNow - latest.stamp).count() :
    -1.0;
  const bool fresh = haveStatus && (steadyNow - latest.stamp) <= Safety::kStatusStaleTimeout;
  const std::uint64_t version = latchVersion_.load();

  // Everything that must be published immediately when it changes (not the continuous age).
  const std::uint64_t key = (latched ? 1u : 0u) | (stopPending ? 2u : 0u) | (moveActive ? 4u : 0u) |
    (activated ? 8u : 0u) | (connected ? 16u : 0u) | (fresh ? 32u : 0u) |
    (static_cast<std::uint64_t>(disabledMask & 0x3Fu) << 8) |
    (static_cast<std::uint64_t>(static_cast<std::uint32_t>(kinstate) & 0xFFFFu) << 16) |
    (connectionCount << 32);
  const std::int64_t nowNs = ToNs(steadyNow);
  const bool changed = key != hwPubLastKey_ || version != hwPubLastVersion_;
  if (!changed &&
    nowNs - hwPubLastNs_ <
    std::chrono::duration_cast<std::chrono::nanoseconds>(kHardwareStatePeriod).count())
  {
    return;
  }
  hwPubLastKey_ = key;
  hwPubLastVersion_ = version;
  hwPubLastNs_ = nowNs;

  igus_rebel_msgs::msg::HardwareState msg;
  msg.header.stamp = node_->now();
  msg.component = componentName_.empty() ? nodeName_ : componentName_;
  msg.connected = connected;
  msg.connection_count = static_cast<std::uint32_t>(connectionCount);
  msg.status_sequence = latest.sequence;
  if (haveStatus) {
    msg.status_stamp = rclcpp::Time(latest.systemStampNs, RCL_SYSTEM_TIME);
  }
  msg.status_age_s = ageS;
  msg.status_fresh = fresh;
  msg.latched = latched;
  if (latched) {
    std::lock_guard<std::mutex> lock(latchReasonLock_);
    msg.latch_reason = latchReason_;
  }
  msg.move_stop_pending = stopPending;
  msg.move_active = moveActive;
  msg.drives_disabled_mask = static_cast<std::uint8_t>(disabledMask & 0x3Fu);
  msg.kinstate = kinstate;
  msg.activated = activated;
  try {
    hardwareStatePub_->publish(msg);
  } catch (const std::exception & error) {
    RCLCPP_WARN_THROTTLE(
      logger_, throttleClock_, 5000, "Could not publish hardware_state: %s", error.what());
  }
}

return_type Rebel::prepare_command_mode_switch(
  const std::vector<std::string> & start_interfaces, const std::vector<std::string> &)
{
  bool claimsOurs = false;
  for (const auto & key : start_interfaces) {
    for (const auto & joint : jointNames_) {
      if (key == joint + "/" + hardware_interface::HW_IF_VELOCITY) {
        claimsOurs = true;
      }
    }
  }
  if (!claimsOurs) {
    return return_type::OK;
  }
  std::lock_guard<std::mutex> lock(claimLock_);
  if (moveState_.load() != kMoveIdle) {
    RCLCPP_ERROR(
      logger_,
      "Refusing to let a controller claim the velocity command interfaces of '%s' while a "
      "joint move (move_joint action) is active",
      componentName_.c_str());
    return return_type::ERROR;
  }
  // Until perform_command_mode_switch() records the claim, a goal must not slip in.
  claimPendingUntilNs_.store(
    ToNs(Safety::SteadyClock::now() + std::chrono::seconds(2)));
  return return_type::OK;
}

return_type Rebel::perform_command_mode_switch(
  const std::vector<std::string> & start_interfaces,
  const std::vector<std::string> & stop_interfaces)
{
  const auto indexOf = [this](const std::string & key) {
      for (std::size_t i = 0; i < jointNames_.size(); ++i) {
        if (key == jointNames_[i] + "/" + hardware_interface::HW_IF_VELOCITY) {
          return static_cast<int>(i);
        }
      }
      return -1;
    };
  // Same lock as move acceptance (CheckMoveAcceptable/ExecuteMoveGoal): a claim is recorded
  // either before a goal is checked (the goal is then refused) or after the move state was set
  // (handled below), never in between.
  std::unique_lock<std::mutex> claimGuard(claimLock_);
  unsigned mask = velocityClaimMask_.load();
  for (const auto & key : stop_interfaces) {
    const int index = indexOf(key);
    if (index >= 0) {
      mask &= ~(1u << index);
      // A released interface keeps its last value: zero it so a later claim (or a stale reader of
      // vel_cmd) never replays it before the new controller has written.
      vel_cmd[static_cast<std::size_t>(index)] = 0.0;
    }
  }
  bool claimsOurs = false;
  for (const auto & key : start_interfaces) {
    const int index = indexOf(key);
    if (index >= 0) {
      mask |= (1u << index);
      claimsOurs = true;
    }
  }
  velocityClaimMask_.store(mask);
  claimPendingUntilNs_.store(0);
  // prepare_command_mode_switch() refuses a claim during a move, but its 2 s reservation can
  // have expired, and Humble's manage_switch() activates the controllers even when this returns
  // ERROR. So do not refuse: record the claim truthfully, stop the robot-owned move and let the
  // move thread abort and latch (a controller that holds the joints would fight it, or drive the
  // arm back to its activation pose afterwards).
  const bool claimDuring = claimsOurs && moveState_.load() != kMoveIdle;
  if (claimDuring) {
    claimDuringMove_.store(true);
  }
  claimGuard.unlock();
  if (claimDuring) {
    RequestMoveStop();
    RCLCPP_ERROR(
      logger_,
      "A controller claimed the velocity command interfaces of '%s' during a joint move: "
      "stopping the move", componentName_.c_str());
  }
  return return_type::OK;
}

rclcpp_action::GoalResponse Rebel::OnMoveGoal(
  const rclcpp_action::GoalUUID &, std::shared_ptr<const MoveJointAction::Goal>)
{
  if (stopping) {
    return rclcpp_action::GoalResponse::REJECT;
  }
  // One goal at a time. Everything else is validated by the move thread, which answers with a
  // result carrying the specific error code.
  if (moveGoalBusy_.exchange(true)) {
    RCLCPP_WARN(logger_, "move_joint goal rejected: another joint move is active");
    return rclcpp_action::GoalResponse::REJECT;
  }
  moveCancelRequested_ = false;
  return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

rclcpp_action::CancelResponse Rebel::OnMoveCancel(const std::shared_ptr<MoveGoalHandle>)
{
  RCLCPP_WARN(logger_, "move_joint cancel requested: sending CMD Move Stop");
  moveCancelRequested_ = true;
  RequestMoveStop();
  return rclcpp_action::CancelResponse::ACCEPT;
}

void Rebel::OnMoveAccepted(const std::shared_ptr<MoveGoalHandle> goalHandle)
{
  std::lock_guard<std::mutex> lock(moveThreadLock_);
  if (moveThread_.joinable()) {
    moveThread_.join();
  }
  if (stopping) {
    auto result = std::make_shared<MoveJointAction::Result>();
    result->error_code = MoveJointAction::Result::REJECTED;
    result->message = "driver is stopping";
    try {
      goalHandle->abort(result);
    } catch (...) {
    }
    moveGoalBusy_ = false;
    return;
  }
  moveThread_ = std::thread([this, goalHandle]() {ExecuteMoveGoal(goalHandle);});
}

std::uint8_t Rebel::CheckMoveAcceptable(
  const MoveJointAction::Goal & goal, Move::Joints & measured, Move::Joints & start,
  Move::Joints & target, std::string & reason)
{
  using Result = MoveJointAction::Result;
  if (stopping) {
    reason = "driver is stopping";
    return Result::REJECTED;
  }
  if (hardwareErrorLatched_) {
    reason = "hardware error is latched; recover via the hardware lifecycle reset";
    return Result::LATCHED;
  }
  if (!activated_) {
    reason = "hardware component is not active";
    return Result::REJECTED;
  }
  if (moveStopRequested_.load()) {
    // A stop that STATUS has not confirmed yet (e.g. after an unconfirmed abort and a lifecycle
    // recovery): the robot may still be running the old move.
    reason = "a previous Move Stop is not confirmed yet";
    return Result::REJECTED;
  }
  BlockingFault blocking;
  if (faultMember_ &&
    (faultMember_->HasRootFault() || FindBlockingRootFault(faultMember_, blocking)))
  {
    reason = "fault group '" + faultGroup_ + "' holds an un-recovered root-cause fault";
    return Result::FAULT;
  }

  Move::GoalRequest request;
  request.jointNames = goal.joint_names;
  request.startPositions = goal.start_positions;
  request.goalPositions = goal.goal_positions;
  request.velocityPercent = goal.velocity_percent;
  request.accelerationPercent = goal.acceleration_percent;
  request.startTolerance = goal.start_tolerance;
  request.pathTolerance = goal.path_tolerance;
  request.goalTolerance = goal.goal_tolerance;
  request.timeoutS = goal.timeout_s;
  Move::Caps caps;
  caps.maxVelocityPercent = moveMaxVelocityPercent_;
  caps.maxAccelerationPercent = moveMaxAccelerationPercent_;
  caps.jointLimits = jointLimits_;
  if (!Move::ValidateGoalRequest(request, jointNames_, caps, start, target, reason)) {
    return Result::REJECTED;
  }

  // The JTC (or any controller) holding the velocity interfaces would fight the robot-side move.
  const auto nowNs = ToNs(Safety::SteadyClock::now());
  if (velocityClaimMask_.load() != 0 || claimPendingUntilNs_.load() > nowNs) {
    reason = "a controller claims (or is about to claim) the velocity command interfaces of this "
      "arm; deactivate it (e.g. '" + trajectoryControllerName_ + "') first";
    return Result::CONTROLLER_ACTIVE;
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
    reason = "STATUS is stale or missing";
    return Result::STATUS_GAP;
  }
  if (statusHold_ || statusCoast_) {
    reason = "STATUS gap in progress (coast/hold)";
    return Result::STATUS_GAP;
  }
  if (rebelSocket->ConnectionCount() != initialConnectionCount) {
    reason = "connection was re-established; recover via the hardware lifecycle reset";
    return Result::STATUS_GAP;
  }
  if (connectedAtNs_.load() == 0 || holdJogZero_) {
    reason = "start-up sequence or reset_and_enable in progress";
    return Result::REJECTED;
  }
  if (drivesDisabledMask_.load() != 0) {
    reason = "drives are not enabled (call reset_and_enable)";
    return Result::REJECTED;
  }
  if (!Safety::IsOverrideAsExpected(latest.overridePercent)) {
    reason = "velocity override is " + std::to_string(latest.overridePercent) +
      " %, expected 100 %";
    return Result::REJECTED;
  }
  if (kinstateValue_.load() != static_cast<int>(CriMessages::Kinstate::NO_ERROR)) {
    reason = std::string("kinematics state is not NO_ERROR: ") +
      CriMessages::KinstateToString(CriMessages::Status::KinstateFromInt(kinstateValue_.load()));
    return Result::FAULT;
  }
  measured = latest.positionRad;
  const double startError = Move::MaxAbsDifference(start, measured);
  if (startError > goal.start_tolerance) {
    reason = "measured joints differ from start_positions by " + std::to_string(startError) +
      " rad (start_tolerance " + std::to_string(goal.start_tolerance) + ")";
    return Result::START_DEVIATION;
  }
  if (Move::MaxAbsDifference(measured, target) < Move::kMinMoveDistance) {
    reason = "the arm is already at the goal: nothing to move";
    return Result::REJECTED;
  }
  return Result::SUCCESS;
}

void Rebel::ExecuteMoveGoal(const std::shared_ptr<MoveGoalHandle> goalHandle)
{
  using Result = MoveJointAction::Result;
  using Clock = Safety::SteadyClock;
  const auto goal = goalHandle->get_goal();
  auto result = std::make_shared<Result>();
  bool stopSent = false;
  bool stopConfirmed = false;
  std::string endReason;

  const auto latestSnapshot = [this]() {
      std::lock_guard<std::mutex> lock(snapshotLock);
      return snapshot;
    };

  // Ends the goal. State goes back to Idle BEFORE the client sees the result, so a controller
  // can be activated as soon as it has it.
  const auto finish = [&](std::uint8_t code, const std::string & message) {
      result->error_code = code;
      result->message = message;
      result->end_reason = endReason;
      result->stop_sent = stopSent;
      result->stop_confirmed = stopConfirmed;
      const JointSnapshot last = latestSnapshot();
      result->final_positions.assign(last.positionRad.begin(), last.positionRad.end());
      if (moveState_.load() != kMoveIdle) {
        // A move was started by this goal (a goal refused up front must not touch a stop that an
        // earlier, unconfirmed move left pending).
        if (velocityClaimMask_.load() != 0 || claimDuringMove_.load()) {
          // A controller took the joints while the move ran (the mask was 0 when it was
          // accepted), also during the stopping phase where the abort checks do not run: it
          // would drive the arm back to its activation pose. Latch BEFORE the stop bookkeeping
          // below (the latch itself requests a stop while the state is not idle).
          LatchHardwareError("a controller claimed the joints during a joint move");
        }
        if (!stopSent || stopConfirmed) {
          moveStopRequested_ = false;  // nothing to stop, or the robot is known to stand still
        } else {
          // Unconfirmed: keep asking. The ALIVEJOG thread re-sends CMD Move Stop every
          // kMoveStopResendPeriod until STATUS shows standstill (ConsumeMoveStopRequest).
          RequestMoveStop();
        }
      }
      moveState_ = kMoveIdle;
      // Also free BEFORE the result: a client may send its next goal the moment it has the
      // result (OnMoveAccepted joins this thread before starting the next one).
      moveGoalBusy_ = false;
      if (code == Result::SUCCESS) {
        RCLCPP_INFO(logger_, "move_joint finished: %s", message.c_str());
      } else {
        RCLCPP_WARN(
          logger_, "move_joint ended with error %u: %s (stop sent %d, confirmed %d)", code,
          message.c_str(), stopSent, stopConfirmed);
      }
      try {
        if (code == Result::SUCCESS) {
          goalHandle->succeed(result);
        } else if (code == Result::CANCELED && goalHandle->is_canceling()) {
          goalHandle->canceled(result);
        } else {
          goalHandle->abort(result);
        }
      } catch (const std::exception & error) {
        RCLCPP_WARN(logger_, "Could not report the move_joint result: %s", error.what());
      }
    };

  //
  // 1. Validate and claim: under claimLock_ so a controller cannot be activated in between.
  //
  Move::Joints measured{};
  Move::Joints start{};
  Move::Joints target{};
  std::string reason;
  std::uint8_t code;
  std::uint64_t stopSeqAtStart = 0;
  {
    std::lock_guard<std::mutex> lock(claimLock_);
    code = CheckMoveAcceptable(*goal, measured, start, target, reason);
    if (code == Result::SUCCESS && moveCancelRequested_) {
      code = Result::CANCELED;
      reason = "canceled before the move was sent";
    }
    if (code == Result::SUCCESS) {
      // (moveStopRequested_ is false here: CheckMoveAcceptable refused a pending stop.)
      claimDuringMove_ = false;
      moveMotionSeen_ = false;
      stopSeqAtStart = moveStopSeq_.load();
      {
        // Nothing of an earlier goal (an old ACK/END, above all an old MOVETOEXECERROR) may
        // survive into this one. While idle RecordMoveEvent records nothing, so from here on
        // everything in moveEvents_ arrived after this point.
        std::lock_guard<std::mutex> eventLock(moveEventLock_);
        moveEvents_ = MoveEvents{};
      }
      moveState_ = kMoveSending;
    }
  }
  if (code != Result::SUCCESS) {
    finish(code, reason);
    return;
  }

  // A latch/fault/cancel that raced the checks above (their stop request was ignored while idle).
  if (stopping || !activated_ || hardwareErrorLatched_ || moveCancelRequested_ ||
    statusHold_ || statusCoast_)
  {
    finish(
      moveCancelRequested_ ? Result::CANCELED : Result::LATCHED,
      "aborted before sending: driver stopped, latched, deactivated or canceled");
    return;
  }

  //
  // 2. Send. The line is supervised from the MEASURED start (the robot moves from there).
  //
  Move::MoveSupervisor supervisor(
    measured, target, goal->velocity_percent, goal->acceleration_percent, goal->path_tolerance,
    goal->goal_tolerance, goal->timeout_s);
  Move::Joints targetDeg{};
  for (std::size_t i = 0; i < 6; ++i) {
    targetDeg[i] = target[i] / Safety::kDegToRad;
  }
  const std::string commandText = Safety::MoveJointCommandText(
    targetDeg, goal->velocity_percent, goal->acceleration_percent);
  {
    // Fresh events for this send (a stale ACK/END that arrived since the state left Idle must
    // not count) -- but a MOVETOEXECERROR recorded in that window is KEPT: it is never a stale
    // ACK/END, and dropping it would lose the robot's report (fail closed: the goal faults).
    std::lock_guard<std::mutex> lock(moveEventLock_);
    const MoveEvents previous = moveEvents_;
    moveEvents_ = MoveEvents{};
    if (previous.errorSeen) {
      moveEvents_.errorSeen = true;
      moveEvents_.errorType = previous.errorType;
      moveEvents_.errorReason = previous.errorReason;
    }
  }
  const int counter = Ccnt();
  moveCommandCounter_ = counter;
  {
    std::lock_guard<std::mutex> lock(commandsLock);
    if (unacknowledgedCommands.size() >= kMaxUnacknowledgedCommands) {
      unacknowledgedCommands.clear();
    }
    unacknowledgedCommands[counter] = commandText;
  }
  RCLCPP_WARN(
    logger_, "move_joint: sending '%s' (model %.2f s, timeout %.2f s)", commandText.c_str(),
    supervisor.ModelDurationS(), supervisor.TimeoutS());
  if (commandText.empty()) {
    stopSent = true;
    RequestMoveStop();
    finish(Result::CMD_ERROR, "could not format CMD Move Joint");
    return;
  }
  bool sendOk;
  {
    // Serialized with the ALIVEJOG thread's CMD Move Stop send (see ConsumeMoveStopRequest): an
    // old stop request that passed its check earlier can never land behind this Move Joint.
    std::lock_guard<std::mutex> sendGuard(moveSendLock_);
    if (const auto hook = std::atomic_load(&beforeMoveSendHook_)) {
      (*hook)();  // tests only: a slow send / a wait behind the stop send
    }
    sendOk = !failNextMoveSend_.exchange(false) &&
      rebelSocket->SendMessage(Safety::FormatCommand(counter, commandText));
  }
  if (!sendOk) {
    // A failed send can follow a partial transmission, so the robot may have received (and may
    // run) the move: like every post-send abort, stop it and latch (stops the peer arm at once)
    // instead of waiting for the reconnect check.
    stopSent = true;
    RequestMoveStop();
    LatchHardwareError("CMD Move Joint could not be sent completely");
    finish(Result::CMD_ERROR, "could not send CMD Move Joint (not connected)");
    return;
  }
  // Everything timed from the send (the 0.5 s MOVETOEXECACK window, the supervisor's progress and
  // timeout clock, the standstill reference) starts only now: a slow send or a wait for
  // moveSendLock_ behind a Move Stop send must not eat the robot's acknowledgement window.
  const auto sentAt = Clock::now();

  //
  // 3. Supervise: Sending -> Moving -> (Stopping) -> Idle.
  //
  constexpr auto kPollPeriod = std::chrono::milliseconds(5);
  constexpr auto kAckTimeout = std::chrono::milliseconds(500);
  constexpr auto kStillWindow = std::chrono::milliseconds(200);
  constexpr auto kStopStillWindow = std::chrono::milliseconds(300);
  constexpr auto kStopRepeat = Safety::kMoveStopResendPeriod;
  constexpr auto kStopLimit = std::chrono::milliseconds(1000);
  constexpr auto kStopLimitShutdown = std::chrono::milliseconds(300);
  constexpr double kStillEpsilonRad = Safety::kMoveStillEpsilonRad;

  bool stopPhase = false;
  // Motion seen since the Move Joint was sent (STATUS differs from the start). Standstill only
  // means "stopped" for a robot that was moving: before that it may simply not have started yet.
  bool motionSeen = false;
  // Reference for "motion": the first STATUS received at/after the send (not `measured`, the
  // validation snapshot, which can differ from the arm by a settle that happened before the send).
  Move::Joints motionRef{};
  bool motionRefValid = false;
  // All six drives report 'Motor not enabled' on fresh STATUS since (e-stop / drives off): the
  // robot cannot execute a move, which confirms a stop even if no motion was ever seen.
  bool drivesOff = false;
  Clock::time_point drivesOffSince{};
  std::uint8_t abortCode = Result::SUCCESS;
  std::string abortMessage;
  Clock::time_point stopFirstAt{};
  Clock::time_point stopLastAt{};
  double maxDeviation = 0.0;
  std::uint64_t lastSequence = 0;
  Move::Joints stillRef = measured;
  auto stillSince = sentAt;
  auto lastFeedbackAt = sentAt - std::chrono::seconds(1);

  // A robot-reported error was handled (latched, reported) by this goal.
  bool errorNoted = false;
  const auto beginStop = [&](std::uint8_t abortWith, const std::string & message) {
      stopPhase = true;
      abortCode = abortWith;
      abortMessage = message;
      stopSent = true;
      stopFirstAt = Clock::now();
      stopLastAt = stopFirstAt;
      moveState_ = kMoveStopping;
      RCLCPP_ERROR(logger_, "move_joint: stopping the move (%s)", message.c_str());
      RequestMoveStop();
    };

  while (true) {
    std::this_thread::sleep_for(kPollPeriod);
    const auto now = Clock::now();
    const JointSnapshot latest = latestSnapshot();
    MoveEvents events;
    {
      std::lock_guard<std::mutex> lock(moveEventLock_);
      events = moveEvents_;
    }
    if (events.errorSeen) {
      endReason = "ERROR";  // the robot failed the move; takes precedence over any END reason
    } else if (events.endSeen) {
      endReason = events.endReason;
    }
    const auto age = now - latest.stamp;
    const double elapsed = std::chrono::duration<double>(now - sentAt).count();

    if (latest.sequence != lastSequence) {
      lastSequence = latest.sequence;
      const Move::Sample sample = supervisor.Evaluate(latest.positionRad);
      maxDeviation = std::max(maxDeviation, sample.deviation);
      if (!motionSeen && latest.stamp >= sentAt) {
        if (!motionRefValid) {
          motionRef = latest.positionRad;
          motionRefValid = true;
        } else if (Move::MaxAbsDifference(latest.positionRad, motionRef) > kStillEpsilonRad) {
          motionSeen = true;
          moveMotionSeen_.store(true);  // for the ALIVEJOG thread's pending-stop check
        }
      }
      if (Move::MaxAbsDifference(latest.positionRad, stillRef) > kStillEpsilonRad) {
        stillRef = latest.positionRad;
        stillSince = latest.stamp;
      }
      if (now - lastFeedbackAt >= std::chrono::milliseconds(50)) {
        lastFeedbackAt = now;
        auto feedback = std::make_shared<MoveJointAction::Feedback>();
        feedback->state =
          stopPhase ? "STOPPING" : (moveState_.load() == kMoveSending ? "SENDING" : "MOVING");
        feedback->progress = sample.progress;
        feedback->max_path_deviation = maxDeviation;
        feedback->actual_positions.assign(latest.positionRad.begin(), latest.positionRad.end());
        try {
          goalHandle->publish_feedback(feedback);
        } catch (...) {
        }
      }
    }
    const bool statusFresh = latest.sequence != 0 && age <= Safety::kStatusStaleTimeout;
    if (statusFresh && drivesDisabledMask_.load() == Safety::kAllDrivesDisabledMask) {
      if (!drivesOff) {
        drivesOff = true;
        drivesOffSince = now;
      }
    } else {
      drivesOff = false;
    }
    const bool still = statusFresh && (now - stillSince) >= kStillWindow;

    if (!stopPhase) {
      if (events.moveAck && moveState_.load() == kMoveSending) {
        moveState_ = kMoveMoving;
      }

      // Aborts, in priority order. A robot-reported error comes first: it must be reported (and
      // latched) with the robot's own reason whatever else is going on, and it overrides END.
      if (events.errorSeen) {
        errorNoted = true;
        const std::string text = "robot reported " + events.errorType + ": " + events.errorReason;
        beginStop(Result::FAULT, text);
        LatchHardwareError(text);
      } else if (stopping) {
        beginStop(Result::FAULT, "driver is shutting down");
      } else if (moveCancelRequested_) {
        beginStop(Result::CANCELED, "canceled by the client");
      } else if (hardwareErrorLatched_) {
        beginStop(
          peerFaulted_ ? Result::FAULT : Result::LATCHED,
          peerFaulted_ ? "a fault-group peer faulted" : "hardware error latched");
      } else if (!activated_) {
        beginStop(Result::FAULT, "hardware component deactivated");
      } else if (velocityClaimMask_.load() != 0 || claimDuringMove_.load()) {
        beginStop(Result::CONTROLLER_ACTIVE, "a controller claimed the joints during a joint move");
        LatchHardwareError("a controller claimed the joints during a joint move");
      } else if (events.cmdError) {
        beginStop(Result::CMD_ERROR, "robot rejected CMD Move Joint: " + events.cmdErrorText);
        LatchHardwareError("robot rejected CMD Move Joint: " + events.cmdErrorText);
      } else if (rebelSocket->ConnectionCount() != initialConnectionCount) {
        beginStop(Result::STATUS_GAP, "connection to the robot was re-established");
        LatchHardwareError("connection lost during a joint move");
      } else if (latest.sequence == 0 || age > Safety::kMoveStatusAbortTimeout) {
        // 200 ms, not the 400 ms jog hold tier: see kMoveStatusAbortTimeout.
        beginStop(Result::STATUS_GAP, "no STATUS during the move (200 ms)");
        LatchHardwareError("STATUS gap during a joint move");
      } else if (moveStopSeq_.load() != stopSeqAtStart) {
        beginStop(Result::FAULT, "a stop was requested (hold tier, deactivate or shutdown)");
      } else if (!events.moveAck && !events.endSeen && now - sentAt > kAckTimeout) {
        beginStop(Result::ACK_TIMEOUT, "no MOVETOEXECACK within 0.5 s");
        LatchHardwareError("no MOVETOEXECACK within 0.5 s of CMD Move Joint");
      } else if (drivesDisabledMask_.load() != 0) {
        beginStop(Result::FAULT, "drives reported 'Motor not enabled' during the move");
        LatchHardwareError("drives reported 'Motor not enabled' during a joint move");
      } else if (kinstateValue_.load() != static_cast<int>(CriMessages::Kinstate::NO_ERROR)) {
        beginStop(Result::FAULT, "kinematics error during the move");
        LatchHardwareError("kinematics error during a joint move");
      } else {
        const Move::Verdict verdict = supervisor.Check(latest.positionRad, elapsed);
        if (verdict == Move::Verdict::PathDeviation || verdict == Move::Verdict::ProgressOverrun) {
          char text[160];
          const Move::Sample sample = supervisor.Evaluate(latest.positionRad);
          std::snprintf(
            text, sizeof(text), "%s (deviation %.4f rad, progress %.3f)",
            Move::VerdictName(verdict), sample.deviation, sample.progress);
          beginStop(Result::PATH_DEVIATION, text);
          // The robot left the supervised line: not a clean stop. Latch (and so stop the other
          // arms of the fault group); recovery is inactive -> active like any other latch.
          LatchHardwareError(std::string("joint move aborted by the supervisor: ") + text);
        } else if (verdict == Move::Verdict::Timeout) {
          char text[96];
          std::snprintf(text, sizeof(text), "no completion within %.2f s", supervisor.TimeoutS());
          beginStop(Result::MOTION_TIMEOUT, text);
          LatchHardwareError(std::string("joint move aborted by the supervisor: ") + text);
        }
      }
      if (stopPhase) {
        continue;
      }

      if (events.endSeen) {
        // The robot says the move is over. PLAN = it reached its goal.
        if (events.endReason != "PLAN") {
          finish(
            Result::ROBOT_ENDED, "the robot ended the move on its own (reason '" +
            events.endReason + "')");
          return;
        }
        if (still) {
          if (supervisor.AtGoal(latest.positionRad)) {
            finish(Result::SUCCESS, "goal reached");
          } else {
            finish(
              Result::GOAL_DEVIATION,
              "the robot reported the move done but the joints are " +
              std::to_string(Move::MaxAbsDifference(latest.positionRad, target)) +
              " rad from the goal (goal_tolerance " + std::to_string(goal->goal_tolerance) + ")");
          }
          return;
        }
      }
      continue;
    }

    // Stopping.
    // A robot-reported error that arrives while the stop is already in progress: latch (if not
    // yet) and say so in the result, but do NOT restart the stop deadline. It is NOT a stop
    // confirmation (see below).
    if (events.errorSeen && !errorNoted) {
      errorNoted = true;
      const std::string text = "robot reported " + events.errorType + ": " + events.errorReason;
      abortMessage += " [" + text + "]";
      if (!hardwareErrorLatched_.load()) {
        LatchHardwareError(text);
      }
    }
    // Confirmed by (a) the robot's MOVETOEXECEND of this move, or (b) standstill AFTER motion
    // was seen. Not by standstill alone (the robot may not have started yet, or never received
    // the stop and start later), and not by a CMDACK of the Move Stop: the controller
    // acknowledges commands it did not act on, and not by a MOVETOEXECERROR (the robot reports
    // that it failed, not that it stands still). A stop that is not confirmed within the limit
    // below latches the driver and stays pending.
    bool confirmed = events.endSeen;
    if (!confirmed && motionSeen && statusFresh && (now - stillSince) >= kStopStillWindow &&
      now - stopFirstAt >= kStopStillWindow)
    {
      confirmed = true;
    }
    if (!confirmed && drivesOff && now - drivesOffSince >= kStopStillWindow) {
      // The way out for a stop that never saw motion or an END: with every drive disabled
      // (e-stop, drives off) the robot cannot execute a move.
      RCLCPP_WARN(
        logger_, "Pending CMD Move Stop confirmed: drives disabled (e-stop) on %s",
        componentName_.c_str());
      confirmed = true;
    }
    if (confirmed) {
      stopConfirmed = true;
      finish(abortCode, abortMessage);
      return;
    }
    if (now - stopFirstAt >= (stopping ? kStopLimitShutdown : kStopLimit)) {
      LatchHardwareError("CMD Move Stop was not confirmed during a joint move");
      stopConfirmed = false;
      finish(
        abortCode, abortMessage + " [CMD Move Stop NOT confirmed within 1 s: hardware latched, "
        "the stop stays pending and jog is held at zero. E-stop the robot if it may still move. "
        "Recovery: e-stop (or switch the drives off) so STATUS reports all drives disabled, wait "
        "for 'Pending CMD Move Stop confirmed', then 'ros2 control set_hardware_component_state " +
        componentName_ + " inactive' then 'active', then cycle the controllers]");
      return;
    }
    if (now - stopLastAt >= kStopRepeat) {
      // Re-assert the (persistent) request: OnConnected() may have cleared it after a reconnect.
      stopLastAt = now;
      RequestMoveStop();
    }
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

  // Robot-side joint move first: zero jog / Disable do not stop it, only Move Stop does.
  RequestMoveStop();
  if (moveState_.load() != kMoveIdle && rebelSocket && rebelSocket->IsConnected()) {
    SendMoveStopNow();
  }

  // Leave the fault group before anything is shut down: waits for a peer callback that is
  // running, after which this instance is never called back, and the Disable below (drives
  // reporting 'Motor not enabled') cannot count as a fault.
  if (faultMember_) {
    LeaveFaultGroup(faultMember_);
  }

  // The move thread sees `stopping`, confirms the stop (bounded) and ends its goal.
  {
    std::lock_guard<std::mutex> lock(moveThreadLock_);
    if (moveThread_.joinable()) {
      moveThread_.join();
    }
  }

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
