// Exercises RebelSocket and the Rebel hardware interface against a fake CRI server on
// 127.0.0.1. Mock hardware never runs Rebel.cpp, so this is the only automated coverage
// of the real driver code paths. The real controller address is never used here.

#include <gtest/gtest.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdarg>
#include <cstdio>
#include <functional>
#include <future>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "control_msgs/msg/joint_jog.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "igus_rebel_msgs/action/move_joint.hpp"
#include "igus_rebel_msgs/msg/cri_diagnostics.hpp"
#include "igus_rebel_msgs/msg/hardware_state.hpp"
#include "igus_rebel_msgs/msg/robot_cart_pose.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "std_srvs/srv/set_bool.hpp"
#include "std_srvs/srv/trigger.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rcutils/logging.h"

#include "igus_rebel/CriFraming.hpp"
#include "igus_rebel/MoveSupervisor.hpp"
#include "igus_rebel/Rebel.hpp"
#include "igus_rebel/RebelSocket.hpp"

namespace
{
using namespace std::chrono_literals;
using Clock = std::chrono::steady_clock;

struct ReceivedFrame
{
  std::size_t connection;  // 1-based
  std::string body;        // without the leading counter
  Clock::time_point stamp;
};

std::string StripCounter(const std::string & body)
{
  const auto space = body.find(' ');
  return space == std::string::npos ? body : body.substr(space + 1);
}

// Minimal single-client CRI robot: sends STATUS every 20 ms, acknowledges CMDs and records
// every frame it receives.
class FakeCriServer
{
public:
  explicit FakeCriServer(bool listen = true)
  {
    currentDeg_ = baseJointsDeg;
    listenFd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    const int enable = 1;
    ::setsockopt(listenFd_, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(enable));
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = 0;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (::bind(listenFd_, reinterpret_cast<sockaddr *>(&address), sizeof(address)) != 0) {
      throw std::runtime_error("bind failed");
    }
    socklen_t length = sizeof(address);
    ::getsockname(listenFd_, reinterpret_cast<sockaddr *>(&address), &length);
    port_ = ntohs(address.sin_port);
    if (!listen) {
      // Nothing listens on this port: connect() is refused.
      ::close(listenFd_);
      listenFd_ = -1;
      return;
    }
    ::listen(listenFd_, 1);
    running_ = true;
    thread_ = std::thread(&FakeCriServer::Run, this);
  }

  ~FakeCriServer()
  {
    Shutdown();
  }

  void Shutdown()
  {
    running_ = false;
    if (thread_.joinable()) {
      thread_.join();
    }
    if (listenFd_ >= 0) {
      ::close(listenFd_);
      listenFd_ = -1;
    }
  }

  int port() const {return port_;}

  void SetSendStatus(bool send) {sendStatus_ = send;}
  void SetJointStepDeg(float step) {jointStepDeg_ = step;}
  void SetOverride(float value) {override_ = value;}
  // CURRENTJOINTS (joints 1-6) / CURRENTALL reported in every STATUS, and an offset added to
  // POSJOINTSETPOINT (joints 1-6) so that it differs from POSJOINTCURRENT.
  void SetCurrents(const std::array<int, 6> & joints, int all)
  {
    for (std::size_t i = 0; i < 6; ++i) {
      currents_[i] = joints[i];
    }
    currentAll_ = all;
  }
  void SetSetpointOffsetDeg(const std::array<float, 6> & offset)
  {
    for (std::size_t i = 0; i < 6; ++i) {
      setpointOffsetDeg_[i] = offset[i];
    }
  }
  void DropClient() {dropClient_ = true;}
  // Joints (bit i) reported with ErrorJoint MNE (motor not enabled); cleared by an Enable command.
  void SetDisabledMask(unsigned mask) {disabledMask_ = mask;}
  // ErrorJoint bits reported for the joints of the mask above (default 4 = MNE; 2 = ESTOP_LOWV).
  void SetJointErrorBits(int bits) {jointErrorBits_ = bits;}
  // While set, an Enable command does not clear the mask (drives stay disabled).
  void SetIgnoreEnable(bool ignore) {ignoreEnable_ = ignore;}
  // CRI KINSTATE value reported in every STATUS (0 = no error, 99 = motion not allowed).
  void SetKinstate(int kinstate) {kinstate_ = kinstate;}
  std::size_t Connections() const {return connections_.load();}

  // ---- Cartesian jog (CMD MotionTypeCartBase / MotionTypeJoint) emulation ----
  // The STATUS MODE follows the last accepted MotionType command and, like the real controller,
  // survives a reconnect of the client.
  bool ModeIsCartBase() const {return mode_.load() == 1;}
  void SetCartBaseError(bool error) {cartBaseError_ = error;}            // CMDERROR instead of ACK
  void SetSuppressCartBaseAck(bool suppress) {suppressCartBaseAck_ = suppress;}  // mode changes
  void SetIgnoreCartBase(bool ignore) {ignoreCartBase_ = ignore;}        // ACK, mode stays
  void SetSuppressJointAck(bool suppress) {suppressJointAck_ = suppress;}  // mode changes, no ACK
  void SetIgnoreMotionTypeJoint(bool ignore) {ignoreJoint_ = ignore;}   // ACK, mode stays
  // Changes the reported mode unprompted: 0 joint, 1 cartbase, 2 fsm, 3 an unknown string.
  void ForceMode(int mode) {mode_ = mode;}

  // ---- Robot-side "CMD Move Joint" emulation (measured behaviour of the real controller) ----
  // Joint-space linear move at a constant speed after a start latency; MOVETOEXECACK + CMDACK on
  // receipt, MOVETOEXECEND -1 0 MoveTo PLAN on arrival, END 0 0 MoveTo USER on Move Stop. The move
  // belongs to the robot: it continues when the client drops.
  void SetMoveSpeedDegS(double speed) {moveSpeedDegS_ = speed;}
  void SetMoveCmdError(bool error) {moveCmdError_ = error;}       // CMDERROR instead of ACK
  void SetSuppressMoveAck(bool suppress) {suppressMoveAck_ = suppress;}
  void SetCmdAckFirst(bool first) {cmdAckFirst_ = first;}         // CMDACK before MOVETOEXECACK
  void SetMoveStall(bool stall) {moveStall_ = stall;}             // position freezes, no END
  void SetMoveDeviationDeg(double deg) {moveDeviationDeg_ = deg;} // joint 5 drifts off the line
  void SetIgnoreMoveStop(bool ignore) {ignoreMoveStop_ = ignore;}
  // Delay between receiving a Move Joint and the first motion (default 120 ms, measured).
  void SetMoveStartLatencyMs(int ms) {startLatencyMs_ = ms;}
  // A Move Stop that arrives before the motion started is acknowledged but has no effect and no
  // END (a controller that only stops a running motion).
  void SetStopIgnoredBeforeStart(bool ignore) {stopIgnoredBeforeStart_ = ignore;}
  // The robot ends the move by itself (pendant stop) this long after it started: END ... USER.
  void SetMoveAbortAfterS(double seconds) {moveAbortAfterS_ = seconds;}
  // The robot reports PLAN but joint 1 stops this many degrees short of the target.
  void SetMoveEndShortDeg(double deg) {moveEndShortDeg_ = deg;}
  // Sends "<counter> <body>" to the connected client on the next loop iteration (about 2 ms),
  // e.g. "MOVETOEXECERROR 0 0 MoveTo Hardware error _ESTOP". Messages go out in call order.
  void InjectMessage(const std::string & body)
  {
    std::lock_guard<std::mutex> lock(injectMutex_);
    injected_.push_back(body);
  }
  // Shifts the standing arm (a settle of a few mrad) without any move.
  void NudgeJointDeg(int joint, float deg)
  {
    std::lock_guard<std::mutex> lock(moveMutex_);
    currentDeg_[joint] += deg;
  }
  bool MoveActive()
  {
    std::lock_guard<std::mutex> lock(moveMutex_);
    return moveActive_;
  }
  std::array<float, 6> CurrentDeg()
  {
    std::lock_guard<std::mutex> lock(moveMutex_);
    AdvanceMoveLocked();
    return CurrentLocked();
  }
  struct MoveCommand
  {
    std::array<double, 6> targetDeg;
    double velocity;
    double acceleration;
    Clock::time_point stamp;
  };
  std::vector<MoveCommand> MoveCommands()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return moveCommands_;
  }
  std::size_t CountCommands(const std::string & command, Clock::time_point after = {})
  {
    std::size_t count = 0;
    for (const auto & frame : Frames()) {
      if (frame.body == "CMD " + command && frame.stamp >= after) {
        ++count;
      }
    }
    return count;
  }
  // Time of the first "CMD <command>" frame at or after `after`; time_point{} if none.
  Clock::time_point FirstCommandAfter(const std::string & command, Clock::time_point after)
  {
    for (const auto & frame : Frames()) {
      if (frame.body == "CMD " + command && frame.stamp >= after) {
        return frame.stamp;
      }
    }
    return {};
  }

  std::vector<ReceivedFrame> Frames()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return frames_;
  }

  std::vector<std::string> Commands(std::size_t connection = 0)
  {
    std::vector<std::string> commands;
    for (const auto & frame : Frames()) {
      if ((connection == 0 || frame.connection == connection) &&
        frame.body.rfind("CMD ", 0) == 0)
      {
        commands.push_back(frame.body.substr(4));
      }
    }
    return commands;
  }

  std::vector<ReceivedFrame> AliveJogs(Clock::time_point after = Clock::time_point{})
  {
    std::vector<ReceivedFrame> jogs;
    for (const auto & frame : Frames()) {
      if (frame.body.rfind("ALIVEJOG ", 0) == 0 && frame.stamp >= after) {
        jogs.push_back(frame);
      }
    }
    return jogs;
  }

  std::array<float, 6> baseJointsDeg{10.0f, 20.0f, -30.0f, 40.0f, 50.0f, -60.0f};
  // POSCARTROBOT sent in every STATUS: x = kCartPoseBase[0] + joint 1 in deg, the rest constant.
  static constexpr std::array<float, 6> kCartPoseBase{300.0f, -45.25f, 410.75f, 12.5f, -170.25f, 33.125f};

private:
  std::string MakeStatus(int counter, const std::array<float, 6> & joints)
  {
    std::ostringstream s;
    s << "CRISTART " << counter << " STATUS MODE " << (mode_.load() == 1 ? "cartbase" : mode_.load() == 2 ? "fsm" :
      mode_.load() == 3 ? "weirdmode" : "joint")
      << " POSJOINTSETPOINT";
    for (int i = 0; i < 16; ++i) {
      s << " " << (i < 6 ? joints[i] + setpointOffsetDeg_[i].load() : 0.0f);
    }
    s << " POSJOINTCURRENT";
    for (int i = 0; i < 16; ++i) {
      s << " " << (i < 6 ? joints[i] : 0.0f);
    }
    // Non-zero POSCARTROBOT so tests can check it is passed through unconverted. x follows
    // joint 1, so the pair (cart pose, joints) of one STATUS is checkable.
    s << " POSCARTROBOT";
    s << " " << (kCartPoseBase[0] + joints[0]);
    for (int i = 1; i < 6; ++i) {
      s << " " << kCartPoseBase[i];
    }
    s << " POSCARTPLATTFORM 0 0 0 OVERRIDE " << override_.load() <<
      " DIN 0 DOUT 0 ESTOP 0 SUPPLY 48 CURRENTALL " << currentAll_.load() << " CURRENTJOINTS";
    for (int i = 0; i < 16; ++i) {
      s << " " << (i < 6 ? currents_[i].load() : 0);
    }
    s << " ERROR 0";
    for (int i = 0; i < 16; ++i) {
      s << " " << ((i < 6 && ((disabledMask_.load() >> i) & 1u)) ? jointErrorBits_.load() : 0);
    }
    s << " KINSTATE " << kinstate_.load() << " OPMODE -1 CRIEND\n";
    return s.str();
  }

  void SendAll(int fd, const std::string & data)
  {
    ::send(fd, data.data(), data.size(), MSG_NOSIGNAL);
  }

  // ---- move emulation; *Locked functions need moveMutex_ ----
  std::array<float, 6> CurrentLocked() const
  {
    return currentDeg_;
  }

  void AdvanceMoveLocked()
  {
    if (!moveActive_) {
      return;
    }
    const auto now = Clock::now();
    if (now < moveT0_) {
      return;  // start latency
    }
    if (moveStall_) {
      return;
    }
    double maxDelta = 0.0;
    for (int i = 0; i < 6; ++i) {
      maxDelta =
        std::max(maxDelta, std::fabs(static_cast<double>(moveTargetDeg_[i] - moveStartDeg_[i])));
    }
    const double sinceStart = std::chrono::duration<double>(now - moveT0_).count();
    if (moveAbortAfterS_.load() > 0.0 && sinceStart > moveAbortAfterS_.load()) {
      moveActive_ = false;
      pendingEnd_ = "0 0 MoveTo USER";
      return;
    }
    const double travelled = sinceStart * moveSpeedDegS_.load();
    const double s = maxDelta <= 0.0 ? 1.0 : std::min(1.0, travelled / maxDelta);
    for (int i = 0; i < 6; ++i) {
      currentDeg_[i] = moveStartDeg_[i] + static_cast<float>(s) *
        (moveTargetDeg_[i] - moveStartDeg_[i]);
    }
    currentDeg_[4] += static_cast<float>(moveDeviationDeg_.load() * s);
    if (s >= 1.0) {
      const float direction = moveTargetDeg_[0] >= moveStartDeg_[0] ? 1.0f : -1.0f;
      currentDeg_[0] -= direction * static_cast<float>(moveEndShortDeg_.load());
      moveActive_ = false;
      pendingEnd_ = "-1 0 MoveTo PLAN";
    }
  }

  void StartMove(const std::array<double, 6> & targetDeg, bool replace)
  {
    std::lock_guard<std::mutex> lock(moveMutex_);
    AdvanceMoveLocked();
    moveStartDeg_ = currentDeg_;
    // Without the deviation offset the line starts where the arm is.
    moveStartDeg_[4] = currentDeg_[4] - static_cast<float>(0.0);
    for (int i = 0; i < 6; ++i) {
      moveTargetDeg_[i] = static_cast<float>(targetDeg[i]);
    }
    moveT0_ = Clock::now() + (replace ? 0ms : std::chrono::milliseconds(startLatencyMs_.load()));
    moveActive_ = true;
    pendingEnd_.clear();
  }

  // Returns true if a move was running (END USER is then sent by the caller).
  bool StopMove()
  {
    std::lock_guard<std::mutex> lock(moveMutex_);
    AdvanceMoveLocked();
    if (stopIgnoredBeforeStart_ && moveActive_ && Clock::now() < moveT0_) {
      return false;  // not started yet: nothing is stopped, the move still begins later
    }
    const bool wasActive = moveActive_;
    moveActive_ = false;
    return wasActive;
  }

  void HandleMoveJoint(
    int client, const std::string & stripped, const std::string & counterText,
    int & counter)
  {
    std::istringstream stream(stripped);
    std::string cmd;
    std::string move;
    std::string joint;
    stream >> cmd >> move >> joint;
    MoveCommand command{};
    for (double & value : command.targetDeg) {
      stream >> value;
    }
    double external = 0.0;
    stream >> external >> external >> external >> command.velocity >> command.acceleration;
    command.stamp = Clock::now();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      moveCommands_.push_back(command);
    }
    if (moveCmdError_) {
      SendAll(
        client, "CRISTART " + std::to_string(counter++) + " CMDERROR " + counterText +
        " Move not possible CRIEND\n");
      return;
    }
    bool replace;
    {
      std::lock_guard<std::mutex> lock(moveMutex_);
      replace = moveActive_;
    }
    StartMove(command.targetDeg, replace);
    const std::string moveAck =
      "CRISTART " + std::to_string(counter++) + " MOVETOEXECACK 0 0 MoveTo CRIEND\n";
    const std::string cmdAck =
      "CRISTART " + std::to_string(counter++) + " CMDACK " + counterText + " CRIEND\n";
    if (cmdAckFirst_) {
      SendAll(client, cmdAck);
    }
    if (!suppressMoveAck_) {
      SendAll(client, moveAck);
    }
    if (!cmdAckFirst_) {
      SendAll(client, cmdAck);
    }
  }

  void Run()
  {
    while (running_) {
      pollfd listenPoll{listenFd_, POLLIN, 0};
      if (::poll(&listenPoll, 1, 20) <= 0) {
        std::lock_guard<std::mutex> lock(moveMutex_);
        AdvanceMoveLocked();
        pendingEnd_.clear();  // nobody is connected to hear it
        continue;
      }
      const int client = ::accept(listenFd_, nullptr, nullptr);
      if (client < 0) {
        continue;
      }
      const std::size_t connection = ++connections_;
      dropClient_ = false;
      {
        std::lock_guard<std::mutex> lock(moveMutex_);
        pendingEnd_.clear();
      }
      Igus::CriFrameParser parser;
      int counter = 1;
      int statusCount = 0;
      auto nextStatus = Clock::now();
      char buffer[4096];

      while (running_ && !dropClient_) {
        pollfd clientPoll{client, POLLIN, 0};
        if (::poll(&clientPoll, 1, 2) > 0) {
          const ssize_t n = ::recv(client, buffer, sizeof(buffer), 0);
          if (n <= 0) {
            break;
          }
          for (const auto & body : parser.Feed(buffer, static_cast<std::size_t>(n))) {
            const std::string stripped = StripCounter(body);
            {
              std::lock_guard<std::mutex> lock(mutex_);
              frames_.push_back({connection, stripped, Clock::now()});
            }
            if (stripped == "CMD Enable" && !ignoreEnable_) {
              disabledMask_ = 0;
            }
            if (stripped.rfind("CMD Move Joint", 0) == 0) {
              HandleMoveJoint(client, stripped, body.substr(0, body.find(' ')), counter);
            } else if (stripped == "CMD Move Stop") {
              const bool wasMoving = !ignoreMoveStop_ && StopMove();
              if (wasMoving) {
                SendAll(
                  client, "CRISTART " + std::to_string(counter++) +
                  " MOVETOEXECEND 0 0 MoveTo USER CRIEND\n");
              }
              SendAll(
                client, "CRISTART " + std::to_string(counter++) + " CMDACK " +
                body.substr(0, body.find(' ')) + " CRIEND\n");
            } else if (stripped == "CMD MotionTypeCartBase" || stripped == "CMD MotionTypeJoint") {
              const std::string counterText = body.substr(0, body.find(' '));
              const bool cart = stripped == "CMD MotionTypeCartBase";
              if (cart && cartBaseError_) {
                SendAll(
                  client, "CRISTART " + std::to_string(counter++) + " CMDERROR " + counterText +
                  " Motion type not possible CRIEND\n");
              } else {
                if (!(cart ? ignoreCartBase_.load() : ignoreJoint_.load())) {
                  mode_ = cart ? 1 : 0;
                }
                if (!(cart ? suppressCartBaseAck_.load() : suppressJointAck_.load())) {
                  SendAll(
                    client, "CRISTART " + std::to_string(counter++) + " CMDACK " + counterText +
                    " CRIEND\n");
                }
              }
            } else if (stripped.rfind("CMD ", 0) == 0) {
              const std::string counterText = body.substr(0, body.find(' '));
              SendAll(
                client, "CRISTART " + std::to_string(counter++) + " CMDACK " + counterText +
                " CRIEND\n");
            }
          }
        }
        {
          std::vector<std::string> injected;
          {
            std::lock_guard<std::mutex> lock(injectMutex_);
            injected.swap(injected_);
          }
          for (const auto & body : injected) {
            SendAll(client, "CRISTART " + std::to_string(counter++) + " " + body + " CRIEND\n");
          }
        }
        {
          std::string end;
          {
            std::lock_guard<std::mutex> lock(moveMutex_);
            AdvanceMoveLocked();
            end.swap(pendingEnd_);
          }
          if (!end.empty()) {
            SendAll(
              client, "CRISTART " + std::to_string(
                counter++) + " MOVETOEXECEND " + end + " CRIEND\n");
          }
        }
        if (sendStatus_ && Clock::now() >= nextStatus) {
          nextStatus += 20ms;
          std::array<float, 6> joints;
          {
            std::lock_guard<std::mutex> lock(moveMutex_);
            AdvanceMoveLocked();
            joints = currentDeg_;
          }
          joints[0] += jointStepDeg_.load() * static_cast<float>(statusCount++);
          const std::string status = MakeStatus(counter++, joints);
          // Split every STATUS across two writes to exercise the framing.
          const std::size_t half = status.size() / 2;
          SendAll(client, status.substr(0, half));
          SendAll(client, status.substr(half));
        }
      }
      ::close(client);
    }
  }

  int listenFd_ = -1;
  int port_ = 0;
  std::atomic<bool> running_{false};
  std::atomic<bool> sendStatus_{true};
  std::atomic<bool> dropClient_{false};
  std::atomic<unsigned> disabledMask_{0};
  std::atomic<int> jointErrorBits_{4};
  std::atomic<bool> ignoreEnable_{false};
  std::atomic<int> kinstate_{0};
  std::atomic<float> jointStepDeg_{0.0f};
  std::atomic<float> override_{40.0f};
  std::array<std::atomic<int>, 6> currents_{};
  std::atomic<int> currentAll_{2};
  std::array<std::atomic<float>, 6> setpointOffsetDeg_{};
  std::atomic<std::size_t> connections_{0};
  std::atomic<int> mode_{0};  // 0 joint, 1 cartbase
  std::atomic<bool> cartBaseError_{false};
  std::atomic<bool> suppressCartBaseAck_{false};
  std::atomic<bool> ignoreCartBase_{false};
  std::atomic<bool> suppressJointAck_{false};
  std::atomic<bool> ignoreJoint_{false};
  std::thread thread_;
  std::mutex mutex_;
  std::vector<ReceivedFrame> frames_;
  std::vector<MoveCommand> moveCommands_;  // guarded by mutex_

  std::mutex injectMutex_;
  std::vector<std::string> injected_;  // guarded by injectMutex_

  std::mutex moveMutex_;
  std::array<float, 6> currentDeg_{};  // set from baseJointsDeg when the server starts
  std::array<float, 6> moveStartDeg_{};
  std::array<float, 6> moveTargetDeg_{};
  Clock::time_point moveT0_{};
  bool moveActive_ = false;
  std::string pendingEnd_;
  std::atomic<double> moveSpeedDegS_{40.0};
  std::atomic<bool> moveCmdError_{false};
  std::atomic<bool> suppressMoveAck_{false};
  std::atomic<bool> cmdAckFirst_{false};
  std::atomic<bool> moveStall_{false};
  std::atomic<double> moveDeviationDeg_{0.0};
  std::atomic<bool> ignoreMoveStop_{false};
  std::atomic<int> startLatencyMs_{120};
  std::atomic<bool> stopIgnoredBeforeStart_{false};
  std::atomic<double> moveAbortAfterS_{0.0};
  std::atomic<double> moveEndShortDeg_{0.0};
};

template<class Predicate>
bool WaitFor(Predicate predicate, std::chrono::milliseconds timeout)
{
  const auto deadline = Clock::now() + timeout;
  while (Clock::now() < deadline) {
    if (predicate()) {
      return true;
    }
    std::this_thread::sleep_for(5ms);
  }
  return predicate();
}

std::array<double, 6> ParseJog(const std::string & aliveJogBody)
{
  std::istringstream stream(aliveJogBody);
  std::string keyword;
  stream >> keyword;
  std::array<double, 6> jog{};
  for (double & value : jog) {
    stream >> value;
  }
  return jog;
}

// HardwareInfo as the URDF's ros2_control tag for the "rebel" protocol yields it: six joints with
// a velocity command interface and position + velocity state interfaces. `prefix` is prepended
// to joint1..joint6 (dual-arm cell: arm_left_ / arm_right_).
hardware_interface::HardwareInfo MakeInfo(
  const std::string & componentName = "rebel_arm_controller", const std::string & prefix = "")
{
  hardware_interface::HardwareInfo info;
  info.name = componentName;
  for (int i = 1; i <= 6; ++i) {
    hardware_interface::ComponentInfo joint;
    joint.name = prefix + "joint" + std::to_string(i);
    hardware_interface::InterfaceInfo velocity;
    velocity.name = hardware_interface::HW_IF_VELOCITY;
    hardware_interface::InterfaceInfo position;
    position.name = hardware_interface::HW_IF_POSITION;
    joint.command_interfaces.push_back(velocity);
    joint.state_interfaces.push_back(position);
    joint.state_interfaces.push_back(velocity);
    info.joints.push_back(joint);
  }
  return info;
}

const std::vector<std::string> kStartupSequence = {
  "Move Stop", "SetActive true", "Reset", "Enable", "MotionTypeJoint", "Override 100.0"};

std::int64_t SystemNowNs()
{
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
    std::chrono::system_clock::now().time_since_epoch()).count();
}

// Subscribes to the driver's own telemetry (~/robot_joint_states, ~/hardware_state) from a
// separate node spun on its own thread, and keeps every message with its arrival time.
class TelemetryListener
{
public:
  using JointState = sensor_msgs::msg::JointState;
  using HardwareState = igus_rebel_msgs::msg::HardwareState;
  using RobotCartPose = igus_rebel_msgs::msg::RobotCartPose;
  struct JointSample
  {
    JointState msg;
    Clock::time_point received{};
    std::int64_t receivedSystemNs = 0;
  };
  struct StateSample
  {
    HardwareState msg;
    Clock::time_point received{};
  };

  explicit TelemetryListener(const std::string & driverNode = "igus_rebel")
  {
    node_ = std::make_shared<rclcpp::Node>("telemetry_listener_" + std::to_string(counter_++));
    jointSub_ = node_->create_subscription<JointState>(
      "/" + driverNode + "/robot_joint_states", rclcpp::SensorDataQoS().keep_last(10),
      [this](JointState::ConstSharedPtr msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        joints_.push_back({*msg, Clock::now(), SystemNowNs()});
      });
    stateSub_ = node_->create_subscription<HardwareState>(
      "/" + driverNode + "/hardware_state", rclcpp::QoS(rclcpp::KeepLast(10)).reliable(),
      [this](HardwareState::ConstSharedPtr msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        states_.push_back({*msg, Clock::now()});
      });
    cartSub_ = node_->create_subscription<RobotCartPose>(
      "/" + driverNode + "/robot_cart_pose", rclcpp::SensorDataQoS().keep_last(10),
      [this](RobotCartPose::ConstSharedPtr msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        carts_.push_back(*msg);
      });
    executor_.add_node(node_);
    thread_ = std::thread([this]() {executor_.spin(); finished_ = true;});
  }
  std::vector<RobotCartPose> Carts()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return carts_;
  }
  ~TelemetryListener()
  {
    while (!finished_) {
      executor_.cancel();
      std::this_thread::sleep_for(2ms);
    }
    if (thread_.joinable()) {
      thread_.join();
    }
  }

  std::size_t JointCount()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return joints_.size();
  }
  std::size_t StateCount()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return states_.size();
  }
  std::vector<JointSample> Joints()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return joints_;
  }
  std::vector<StateSample> States()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return states_;
  }
  JointSample LastJoint()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return joints_.empty() ? JointSample() : joints_.back();
  }
  HardwareState LastState()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return states_.empty() ? HardwareState() : states_.back().msg;
  }
  // Waits until the newest hardware_state satisfies `predicate`.
  template<class Predicate>
  bool WaitState(Predicate predicate, std::chrono::milliseconds timeout)
  {
    return WaitFor(
      [&]() {
        std::lock_guard<std::mutex> lock(mutex_);
        return !states_.empty() && predicate(states_.back().msg);
      }, timeout);
  }
  // Waits until the newest robot_joint_states satisfies `predicate`.
  template<class Predicate>
  bool WaitJoint(Predicate predicate, std::chrono::milliseconds timeout)
  {
    return WaitFor(
      [&]() {
        std::lock_guard<std::mutex> lock(mutex_);
        return !joints_.empty() && predicate(joints_.back());
      }, timeout);
  }

private:
  static inline std::atomic<int> counter_{0};
  std::atomic<bool> finished_{false};
  rclcpp::Node::SharedPtr node_;
  rclcpp::Subscription<JointState>::SharedPtr jointSub_;
  rclcpp::Subscription<HardwareState>::SharedPtr stateSub_;
  rclcpp::Subscription<RobotCartPose>::SharedPtr cartSub_;
  std::vector<RobotCartPose> carts_;
  rclcpp::executors::SingleThreadedExecutor executor_;
  std::thread thread_;
  std::mutex mutex_;
  std::vector<JointSample> joints_;
  std::vector<StateSample> states_;
};

constexpr double kDegToRad = Igus::Safety::kDegToRad;

class RebelDriverTest : public ::testing::Test
{
protected:
  void StartDriver(FakeCriServer & server)
  {
    rebel = std::make_unique<Igus::Rebel>("127.0.0.1", server.port());
    // component name matches the ros2_control tag name in the URDF
    const auto info = MakeInfo("rebel_arm_controller");
    ASSERT_EQ(rebel->on_init(info), CallbackReturn::SUCCESS);
    states = rebel->export_state_interfaces();
    commands = rebel->export_command_interfaces();
  }

  double Position(std::size_t joint) {return states.at(2 * joint).get_value();}
  double Velocity(std::size_t joint) {return states.at(2 * joint + 1).get_value();}

  rclcpp::Time now{0, 0, RCL_STEADY_TIME};
  rclcpp::Duration period{0, 10000000};
  std::unique_ptr<Igus::Rebel> rebel;
  std::vector<hardware_interface::StateInterface> states;
  std::vector<hardware_interface::CommandInterface> commands;
};
}  // namespace

TEST_F(RebelDriverTest, StartupSequenceSeedsStateAndStopDisables)
{
  FakeCriServer server;
  StartDriver(server);

  // State is seeded from the first STATUS before on_init returns.
  for (std::size_t i = 0; i < 6; ++i) {
    EXPECT_NEAR(Position(i), server.baseJointsDeg[i] * Igus::Safety::kDegToRad, 1e-6);
    EXPECT_EQ(Velocity(i), 0.0);
  }

  ASSERT_TRUE(WaitFor([&]() {return server.Commands().size() >= 6;}, 2000ms));
  EXPECT_EQ(server.Commands(), kStartupSequence);

  // ALIVEJOG starts after Enable and is zero while write() was never called.
  ASSERT_TRUE(WaitFor([&]() {return server.AliveJogs().size() >= 5;}, 2000ms));
  const auto frames = server.Frames();
  std::size_t enableIndex = 0;
  std::size_t firstJogIndex = 0;
  for (std::size_t i = 0; i < frames.size(); ++i) {
    if (frames[i].body == "CMD Enable") {
      enableIndex = i;
    }
    if (firstJogIndex == 0 && frames[i].body.rfind("ALIVEJOG", 0) == 0) {
      firstJogIndex = i;
    }
  }
  EXPECT_GT(firstJogIndex, enableIndex);
  for (const auto & jogFrame : server.AliveJogs()) {
    for (double value : ParseJog(jogFrame.body)) {
      EXPECT_EQ(value, 0.0);
    }
  }

  EXPECT_EQ(rebel->read(now, period), return_type::OK);

  rebel->Stop();
  ASSERT_TRUE(WaitFor([&]() {return server.Commands().size() >= 8;}, 1000ms));
  const auto commandsAfterStop = server.Commands();
  ASSERT_GE(commandsAfterStop.size(), 8u);
  EXPECT_EQ(commandsAfterStop[commandsAfterStop.size() - 2], "Disable");
  EXPECT_EQ(commandsAfterStop.back(), "QUIT");
}

TEST_F(RebelDriverTest, WriteForwardsClampedJogAndWatchdogZeroesIt)
{
  FakeCriServer server;
  StartDriver(server);
  ASSERT_TRUE(WaitFor([&]() {return server.Commands().size() >= 6;}, 2000ms));

  commands.at(0).set_value(0.1);  // 11.459 %
  commands.at(1).set_value(std::numeric_limits<double>::quiet_NaN());
  commands.at(2).set_value(10.0);  // saturates
  commands.at(3).set_value(-10.0);

  for (int i = 0; i < 20; ++i) {
    ASSERT_EQ(rebel->write(now, period), return_type::OK);
    std::this_thread::sleep_for(10ms);
  }
  const auto whileWriting = server.AliveJogs(Clock::now() - 60ms);
  ASSERT_FALSE(whileWriting.empty());
  const auto jog = ParseJog(whileWriting.back().body);
  EXPECT_NEAR(jog[0], 2.0 * 0.1 / Igus::Safety::kDegToRad, 1e-4);
  EXPECT_EQ(jog[1], 0.0);
  EXPECT_EQ(jog[2], 100.0);
  EXPECT_EQ(jog[3], -100.0);

  // Stop calling write(): within the 50 ms watchdog (+ one period and delivery margin)
  // every ALIVEJOG must be zero.
  const auto stoppedWriting = Clock::now();
  std::this_thread::sleep_for(200ms);
  const auto afterWatchdog = server.AliveJogs(stoppedWriting + 80ms);
  ASSERT_GE(afterWatchdog.size(), 5u);
  for (const auto & frame : afterWatchdog) {
    for (double value : ParseJog(frame.body)) {
      EXPECT_EQ(value, 0.0);
    }
  }
}

TEST_F(RebelDriverTest, OnErrorAndOnDeactivateZeroTheJogImmediately)
{
  FakeCriServer server;
  StartDriver(server);
  ASSERT_TRUE(WaitFor([&]() {return server.Commands().size() >= 6;}, 2000ms));

  commands.at(0).set_value(0.2);
  for (int i = 0; i < 10; ++i) {
    rebel->write(now, period);
    std::this_thread::sleep_for(10ms);
  }
  rebel->write(now, period);
  const auto errorTime = Clock::now();
  rebel->on_error(rclcpp_lifecycle::State());
  // Well before the 50 ms watchdog would trigger.
  std::this_thread::sleep_for(40ms);
  const auto afterError = server.AliveJogs(errorTime + 15ms);
  ASSERT_FALSE(afterError.empty());
  for (const auto & frame : afterError) {
    EXPECT_EQ(ParseJog(frame.body)[0], 0.0);
  }

  EXPECT_EQ(rebel->on_deactivate(rclcpp_lifecycle::State()), CallbackReturn::SUCCESS);
  EXPECT_EQ(commands.at(0).get_value(), 0.0);
}

TEST_F(RebelDriverTest, ReadUsesStatusIntervalsAndHoldsBetweenStatuses)
{
  FakeCriServer server;
  server.SetJointStepDeg(0.5f);  // 0.5 deg per 20 ms = 25 deg/s
  StartDriver(server);

  const double expected = 25.0 * Igus::Safety::kDegToRad;
  double sum = 0.0;
  int samples = 0;
  double previousPosition = Position(0);
  int unchangedReads = 0;
  std::this_thread::sleep_for(100ms);
  for (int i = 0; i < 60; ++i) {
    ASSERT_EQ(rebel->read(now, period), return_type::OK);
    if (Position(0) == previousPosition) {
      ++unchangedReads;
    }
    previousPosition = Position(0);
    sum += Velocity(0);
    ++samples;
    EXPECT_NEAR(Velocity(0), expected, 0.5 * expected);
    EXPECT_NEAR(Velocity(1), 0.0, 1e-9);
    std::this_thread::sleep_for(10ms);
  }
  // Reads outnumber statuses, so some reads hold the previous value, yet the velocity never
  // collapses to zero between statuses (the old driver alternated 0 / 2x).
  EXPECT_GT(unchangedReads, 0);
  EXPECT_NEAR(sum / samples, expected, 0.15 * expected);
}

TEST_F(RebelDriverTest, ReadReturnsErrorWhenStatusStops)
{
  FakeCriServer server;
  StartDriver(server);
  EXPECT_EQ(rebel->read(now, period), return_type::OK);

  server.SetSendStatus(false);
  const auto stopped = Clock::now();
  std::this_thread::sleep_for(100ms);
  // 100 ms into the gap: still fresh. 300 ms: coast (OK, no latch). 500 ms: hold (OK).
  EXPECT_EQ(rebel->read(now, period), return_type::OK);
  std::this_thread::sleep_for(200ms);
  EXPECT_EQ(rebel->read(now, period), return_type::OK);
  EXPECT_TRUE(rebel->StatusCoastActive());
  EXPECT_FALSE(rebel->StatusHoldActive());
  std::this_thread::sleep_for(200ms);
  EXPECT_EQ(rebel->read(now, period), return_type::OK);
  EXPECT_TRUE(rebel->StatusHoldActive());
  EXPECT_FALSE(rebel->StatusCoastActive());
  EXPECT_FALSE(rebel->HardwareErrorLatched());
  // Beyond kStatusLatchTimeout (1 s) read() returns ERROR.
  ASSERT_TRUE(
    WaitFor([&]() {return rebel->read(now, period) == return_type::ERROR;}, 2500ms));
  const auto elapsed = Clock::now() - stopped;
  EXPECT_GE(elapsed, 990ms);
  EXPECT_LT(elapsed, 1400ms);
}

TEST_F(RebelDriverTest, ReconnectResendsStartupSequence)
{
  FakeCriServer server;
  StartDriver(server);
  ASSERT_TRUE(WaitFor([&]() {return server.Commands(1).size() >= 6;}, 2000ms));

  server.DropClient();
  ASSERT_TRUE(WaitFor([&]() {return server.Connections() >= 2;}, 3000ms));
  ASSERT_TRUE(WaitFor([&]() {return server.Commands(2).size() >= 6;}, 3000ms));
  EXPECT_EQ(server.Commands(2), kStartupSequence);
  // Motion must not resume on its own after a reconnect: read() fails closed. ros2_control
  // then calls on_error() and stops calling read()/write(), so the jog stays zero.
  EXPECT_EQ(rebel->read(now, period), return_type::ERROR);
  EXPECT_EQ(rebel->on_error(rclcpp_lifecycle::State()), CallbackReturn::SUCCESS);
  const auto afterError = Clock::now();
  std::this_thread::sleep_for(100ms);
  const auto jogs = server.AliveJogs(afterError + 15ms);
  ASSERT_FALSE(jogs.empty());
  for (const auto & frame : jogs) {
    EXPECT_EQ(frame.connection, 2u);
    for (double value : ParseJog(frame.body)) {
      EXPECT_EQ(value, 0.0);
    }
  }
}

TEST_F(RebelDriverTest, LatchedErrorRecoversOnlyViaLifecycleResetWithFreshStatus)
{
  FakeCriServer server;
  StartDriver(server);
  ASSERT_TRUE(WaitFor([&]() {return server.Commands(1).size() >= 6;}, 2000ms));
  EXPECT_EQ(server.Commands(1), kStartupSequence);

  // STATUS stops: read() eventually returns ERROR and on_error() (as ros2_control would
  // call) latches it.
  server.SetSendStatus(false);
  const auto stopped = Clock::now();
  ASSERT_TRUE(
    WaitFor([&]() {return rebel->read(now, period) == return_type::ERROR;}, 2500ms));
  EXPECT_EQ(rebel->on_error(rclcpp_lifecycle::State()), CallbackReturn::SUCCESS);

  // STATUS resumes, but the latch is not cleared by fresh STATUS alone.
  server.SetSendStatus(true);
  std::this_thread::sleep_for(100ms);
  EXPECT_EQ(rebel->read(now, period), return_type::ERROR);

  // `ros2 control set_hardware_component_state <component> inactive`: still latched, jog
  // held at zero, read() still refuses.
  EXPECT_EQ(rebel->on_configure(rclcpp_lifecycle::State()), CallbackReturn::SUCCESS);
  EXPECT_EQ(rebel->read(now, period), return_type::ERROR);

  // Jog was zero for the entire outage and recovery so far.
  for (const auto & frame : server.AliveJogs(stopped)) {
    for (double value : ParseJog(frame.body)) {
      EXPECT_EQ(value, 0.0);
    }
  }

  // `ros2 control set_hardware_component_state <component> active`: STATUS is fresh, so the
  // latch clears and the connect sequence is re-sent on the same (still-connected) socket.
  EXPECT_EQ(rebel->on_activate(rclcpp_lifecycle::State()), CallbackReturn::SUCCESS);
  ASSERT_TRUE(WaitFor([&]() {return server.Commands(1).size() >= 12;}, 2000ms));
  const auto allCommandsConn1 = server.Commands(1);
  const std::vector<std::string> resent(
    allCommandsConn1.end() - 6, allCommandsConn1.end());
  EXPECT_EQ(resent, kStartupSequence);
  EXPECT_EQ(server.Connections(), 1u);  // no socket reconnect happened

  EXPECT_EQ(rebel->read(now, period), return_type::OK);
}

TEST_F(RebelDriverTest, ActivateRefusesWhileStatusStillStale)
{
  FakeCriServer server;
  StartDriver(server);
  ASSERT_TRUE(WaitFor([&]() {return server.Commands(1).size() >= 6;}, 2000ms));

  server.SetSendStatus(false);
  ASSERT_TRUE(
    WaitFor([&]() {return rebel->read(now, period) == return_type::ERROR;}, 2500ms));
  EXPECT_EQ(rebel->on_error(rclcpp_lifecycle::State()), CallbackReturn::SUCCESS);
  EXPECT_EQ(rebel->on_configure(rclcpp_lifecycle::State()), CallbackReturn::SUCCESS);

  // STATUS is still stale (never resumed): activation must be refused, and the latch must
  // stay set.
  EXPECT_EQ(rebel->on_activate(rclcpp_lifecycle::State()), CallbackReturn::FAILURE);
  EXPECT_EQ(rebel->read(now, period), return_type::ERROR);
  // No extra commands were sent by the refused activation attempt.
  EXPECT_EQ(server.Commands(1), kStartupSequence);
}

TEST_F(RebelDriverTest, ResetAndEnableResendsSequenceWithJogHeldAtZero)
{
  FakeCriServer server;
  StartDriver(server);
  ASSERT_TRUE(WaitFor([&]() {return server.Commands(1).size() >= 6;}, 2000ms));
  ASSERT_EQ(rebel->on_activate(rclcpp_lifecycle::State()), CallbackReturn::SUCCESS);

  // Robot-side disable while STATUS stays fresh and NoError: read() keeps returning OK.
  server.SetDisabledMask(0x3f);
  ASSERT_TRUE(WaitFor([&]() {return rebel->DrivesDisabledMask() == 0x3f;}, 1000ms));
  EXPECT_EQ(rebel->read(now, period), return_type::OK);
  EXPECT_FALSE(rebel->HardwareErrorLatched());

  // A controller keeps commanding a non-zero velocity the whole time.
  std::atomic<bool> run{true};
  std::thread writer([&]() {
      while (run) {
        for (auto & command : commands) {
          command.set_value(0.05);
        }
        rebel->write(now, period);
        std::this_thread::sleep_for(5ms);
      }
    });
  ASSERT_TRUE(
    WaitFor(
      [&]() {
        for (const auto & frame : server.AliveJogs(Clock::now() - 100ms)) {
          for (double value : ParseJog(frame.body)) {
            if (value != 0.0) {return true;}
          }
        }
        return false;
      }, 1000ms));

  std::string message;
  EXPECT_TRUE(rebel->ResetAndEnable(message)) << message;
  const auto returned = Clock::now();
  run = false;
  writer.join();

  ASSERT_TRUE(WaitFor([&]() {return server.Commands(1).size() >= 12;}, 2000ms));
  const auto all = server.Commands(1);
  const std::vector<std::string> resent(all.end() - 6, all.end());
  EXPECT_EQ(resent, kStartupSequence);
  EXPECT_EQ(server.Connections(), 1u);
  EXPECT_EQ(rebel->DrivesDisabledMask(), 0u);

  // From the moment SetActive went out until ResetAndEnable() returned, every ALIVEJOG was zero.
  Clock::time_point setActiveStamp{};
  int index = 0;
  for (const auto & frame : server.Frames()) {
    if (frame.body == "CMD SetActive true" && ++index == 2) {
      setActiveStamp = frame.stamp;
    }
  }
  ASSERT_NE(setActiveStamp, Clock::time_point{});
  std::size_t checked = 0;
  for (const auto & frame : server.AliveJogs(setActiveStamp)) {
    if (frame.stamp > returned) {
      continue;
    }
    ++checked;
    for (double value : ParseJog(frame.body)) {
      EXPECT_EQ(value, 0.0);
    }
  }
  EXPECT_GT(checked, 0u);
}

namespace
{
// Emulates the controller manager: read()/write() every 5 ms with a constant velocity command
// (the command can be changed on the fly). Stops calling both after read() returns ERROR, as
// ros2_control does.
struct ControllerLoop
{
  ControllerLoop(
    Igus::Rebel & driver, std::vector<hardware_interface::CommandInterface> & commandInterfaces,
    double velocity)
  : rebel(driver), commands(commandInterfaces), velocity_(velocity)
  {
    thread = std::thread(
      [this]() {
        while (run) {
          if (rebel.read(rclcpp::Time(0, 0, RCL_STEADY_TIME), rclcpp::Duration(0, 10000000)) ==
          return_type::ERROR)
          {
            ++errors;
            break;
          }
          for (auto & command : commands) {
            command.set_value(velocity_.load());
          }
          rebel.write(rclcpp::Time(0, 0, RCL_STEADY_TIME), rclcpp::Duration(0, 10000000));
          std::this_thread::sleep_for(5ms);
        }
      });
  }
  ~ControllerLoop() {Stop();}
  void Stop()
  {
    run = false;
    if (thread.joinable()) {
      thread.join();
    }
  }
  void SetVelocity(double velocity) {velocity_ = velocity;}

  Igus::Rebel & rebel;
  std::vector<hardware_interface::CommandInterface> & commands;
  std::atomic<double> velocity_;
  std::atomic<bool> run{true};
  std::atomic<int> errors{0};
  std::thread thread;
};

bool AnyNonZeroJog(FakeCriServer & server, Clock::time_point after)
{
  for (const auto & frame : server.AliveJogs(after)) {
    for (double value : ParseJog(frame.body)) {
      if (value != 0.0) {return true;}
    }
  }
  return false;
}
}  // namespace

TEST_F(RebelDriverTest, CoastKeepsSendingFrozenJogThroughShortGapAndResumes)
{
  FakeCriServer server;
  StartDriver(server);
  ASSERT_TRUE(WaitFor([&]() {return server.Commands(1).size() >= 6;}, 2000ms));
  ASSERT_EQ(rebel->on_activate(rclcpp_lifecycle::State()), CallbackReturn::SUCCESS);

  ControllerLoop loop(*rebel, commands, 0.1);  // 0.1 rad/s -> ~11.46 % jog
  ASSERT_TRUE(WaitFor([&]() {return AnyNonZeroJog(server, Clock::now() - 100ms);}, 1000ms));
  const double moving = ParseJog(server.AliveJogs(Clock::now() - 50ms).back().body)[0];
  ASSERT_GT(moving, 5.0);

  server.SetSendStatus(false);
  const auto pauseStart = Clock::now();
  ASSERT_TRUE(WaitFor([&]() {return rebel->StatusCoastActive();}, 1000ms));
  const auto coastStart = Clock::now();
  loop.SetVelocity(0.2);  // new commands during the coast must be ignored (frozen jog)
  std::this_thread::sleep_for(pauseStart + 300ms - Clock::now());
  server.SetSendStatus(true);
  ASSERT_TRUE(WaitFor([&]() {return !rebel->StatusCoastActive();}, 1000ms));
  const auto resumed = Clock::now();

  EXPECT_FALSE(rebel->StatusHoldActive());
  EXPECT_FALSE(rebel->HardwareErrorLatched());
  EXPECT_EQ(loop.errors.load(), 0);
  // The robot kept receiving the same non-zero ALIVEJOG for the whole pause.
  std::size_t checked = 0;
  for (const auto & frame : server.AliveJogs(pauseStart + 20ms)) {
    if (frame.stamp > resumed - 20ms) {continue;}
    ++checked;
    for (double value : ParseJog(frame.body)) {
      EXPECT_NEAR(value, moving, 1e-3);
    }
  }
  EXPECT_GT(checked, 15u);
  (void)coastStart;
  // After the resume the new command goes out.
  EXPECT_TRUE(
    WaitFor(
      [&]() {
        for (const auto & frame : server.AliveJogs(resumed + 30ms)) {
          if (ParseJog(frame.body)[0] > 1.5 * moving) {return true;}
        }
        return false;
      }, 1000ms));
  loop.Stop();
  EXPECT_EQ(server.Connections(), 1u);
}

TEST_F(RebelDriverTest, HoldTierWhileMovingZeroesJogAndLatchesOnResume)
{
  FakeCriServer server;
  StartDriver(server);
  ASSERT_TRUE(WaitFor([&]() {return server.Commands(1).size() >= 6;}, 2000ms));
  ASSERT_EQ(rebel->on_activate(rclcpp_lifecycle::State()), CallbackReturn::SUCCESS);

  ControllerLoop loop(*rebel, commands, 0.1);
  ASSERT_TRUE(WaitFor([&]() {return AnyNonZeroJog(server, Clock::now() - 100ms);}, 1000ms));

  server.SetSendStatus(false);
  const auto pauseStart = Clock::now();
  ASSERT_TRUE(WaitFor([&]() {return rebel->StatusHoldActive();}, 1000ms));
  const auto holdStart = Clock::now();
  EXPECT_GE(holdStart - pauseStart, 380ms);  // coast lasts until ~400 ms
  std::this_thread::sleep_for(pauseStart + 600ms - Clock::now());
  server.SetSendStatus(true);
  ASSERT_TRUE(WaitFor([&]() {return !rebel->StatusHoldActive();}, 1000ms));
  loop.Stop();

  EXPECT_EQ(loop.errors.load(), 1);
  EXPECT_TRUE(rebel->HardwareErrorLatched());
  // Jog is zero from the start of the hold onwards (small margin for frames in flight).
  for (const auto & frame : server.AliveJogs(holdStart + 30ms)) {
    for (double value : ParseJog(frame.body)) {
      EXPECT_EQ(value, 0.0);
    }
  }
  EXPECT_EQ(rebel->read(now, period), return_type::ERROR);
  EXPECT_EQ(rebel->on_configure(rclcpp_lifecycle::State()), CallbackReturn::SUCCESS);
  EXPECT_EQ(rebel->on_activate(rclcpp_lifecycle::State()), CallbackReturn::SUCCESS);
  EXPECT_EQ(rebel->read(now, period), return_type::OK);
}

TEST_F(RebelDriverTest, HoldTierWhileIdleZeroesJogAndResumes)
{
  FakeCriServer server;
  StartDriver(server);
  ASSERT_TRUE(WaitFor([&]() {return server.Commands(1).size() >= 6;}, 2000ms));
  ASSERT_EQ(rebel->on_activate(rclcpp_lifecycle::State()), CallbackReturn::SUCCESS);

  ControllerLoop loop(*rebel, commands, 0.005);  // idle JTC hold level (< 0.02 rad/s)
  ASSERT_TRUE(WaitFor([&]() {return AnyNonZeroJog(server, Clock::now() - 100ms);}, 1000ms));

  server.SetSendStatus(false);
  const auto pauseStart = Clock::now();
  ASSERT_TRUE(WaitFor([&]() {return rebel->StatusHoldActive();}, 1000ms));
  const auto holdStart = Clock::now();
  std::this_thread::sleep_for(pauseStart + 600ms - Clock::now());
  server.SetSendStatus(true);
  ASSERT_TRUE(WaitFor([&]() {return !rebel->StatusHoldActive();}, 1000ms));
  const auto resumed = Clock::now();

  EXPECT_FALSE(rebel->HardwareErrorLatched());
  EXPECT_EQ(loop.errors.load(), 0);
  for (const auto & frame : server.AliveJogs(holdStart + 30ms)) {
    if (frame.stamp > resumed - 30ms) {continue;}
    for (double value : ParseJog(frame.body)) {
      EXPECT_EQ(value, 0.0);
    }
  }
  EXPECT_TRUE(WaitFor([&]() {return AnyNonZeroJog(server, resumed + 30ms);}, 1000ms));
  loop.Stop();
}

TEST_F(RebelDriverTest, LongStatusGapLatches)
{
  FakeCriServer server;
  StartDriver(server);
  ASSERT_TRUE(WaitFor([&]() {return server.Commands(1).size() >= 6;}, 2000ms));

  server.SetSendStatus(false);
  bool sawHold = false;
  const auto start = Clock::now();
  while (Clock::now() - start < 1200ms) {
    const auto result = rebel->read(now, period);
    sawHold = sawHold || rebel->StatusHoldActive();
    if (result == return_type::ERROR) {break;}
    std::this_thread::sleep_for(5ms);
  }
  EXPECT_TRUE(sawHold);
  EXPECT_TRUE(rebel->HardwareErrorLatched());
  EXPECT_GE(Clock::now() - start, 900ms);

  // STATUS resuming does not clear it; only the lifecycle reset does.
  server.SetSendStatus(true);
  std::this_thread::sleep_for(100ms);
  EXPECT_EQ(rebel->read(now, period), return_type::ERROR);
  EXPECT_EQ(rebel->on_configure(rclcpp_lifecycle::State()), CallbackReturn::SUCCESS);
  EXPECT_EQ(rebel->on_activate(rclcpp_lifecycle::State()), CallbackReturn::SUCCESS);
  EXPECT_EQ(rebel->read(now, period), return_type::OK);
}

// ---- Telemetry that survives a latch: ~/robot_joint_states and ~/hardware_state ----
// These tests drive the plugin directly (no controller_manager), as the tests above do: a
// ControllerLoop emulates read()/write(), and on_error()/on_configure()/on_activate() are called
// the way ros2_control calls them. They therefore verify what the driver publishes, not what
// joint_state_broadcaster does with the frozen state interfaces.

TEST_F(RebelDriverTest, TelemetryFlowsWhileInactiveAndAtTenHertz)
{
  FakeCriServer server;
  StartDriver(server);
  TelemetryListener telemetry;
  // Never activated: the component is INACTIVE, read() is not called, telemetry still flows.
  ASSERT_TRUE(WaitFor([&]() {return telemetry.JointCount() >= 5 && telemetry.StateCount() >= 3;}, 4000ms));

  const auto joint = telemetry.LastJoint();
  ASSERT_EQ(joint.msg.name.size(), 6u);
  ASSERT_EQ(joint.msg.position.size(), 6u);
  for (std::size_t i = 0; i < 6; ++i) {
    EXPECT_EQ(joint.msg.name[i], rebel->JointNames()[i]);
    EXPECT_NEAR(joint.msg.position[i], server.baseJointsDeg[i] * kDegToRad, 1e-5);
  }
  // Stamp = acquisition time of the sample: shortly before it arrived, never after.
  const std::int64_t stampNs = rclcpp::Time(joint.msg.header.stamp).nanoseconds();
  EXPECT_LE(stampNs, joint.receivedSystemNs);
  EXPECT_LT(joint.receivedSystemNs - stampNs, 100'000'000);

  const auto hw = telemetry.LastState();
  EXPECT_EQ(hw.component, "rebel_arm_controller");
  EXPECT_TRUE(hw.connected);
  EXPECT_GE(hw.connection_count, 1u);
  EXPECT_GT(hw.status_sequence, 0u);
  EXPECT_TRUE(hw.status_fresh);
  EXPECT_GE(hw.status_age_s, 0.0);
  EXPECT_LT(hw.status_age_s, 0.2);
  EXPECT_FALSE(hw.latched);
  EXPECT_TRUE(hw.latch_reason.empty());
  EXPECT_FALSE(hw.activated);
  EXPECT_FALSE(hw.move_active);
  EXPECT_FALSE(hw.move_stop_pending);
  EXPECT_EQ(hw.drives_disabled_mask, 0u);
  EXPECT_EQ(hw.kinstate, 0);
  EXPECT_GT(rclcpp::Time(hw.status_stamp).nanoseconds(), 0);

  // hardware_state at 10 Hz while nothing changes; robot_joint_states once per STATUS (20 ms).
  const std::size_t states0 = telemetry.StateCount();
  const std::size_t joints0 = telemetry.JointCount();
  std::this_thread::sleep_for(1000ms);
  const std::size_t states = telemetry.StateCount() - states0;
  const std::size_t joints = telemetry.JointCount() - joints0;
  EXPECT_GE(states, 8u);
  EXPECT_LE(states, 13u);
  EXPECT_GE(joints, 35u);
  EXPECT_LE(joints, 55u);

  // Activation shows up (immediately, not only at the next 10 Hz tick).
  ASSERT_EQ(rebel->on_activate(rclcpp_lifecycle::State()), CallbackReturn::SUCCESS);
  EXPECT_TRUE(telemetry.WaitState([](const auto & msg) {return msg.activated;}, 60ms));
}

TEST_F(RebelDriverTest, RobotCartPoseCarriesRawPoscartrobotWithTheJointsOfTheSameStatus)
{
  FakeCriServer server;
  StartDriver(server);
  TelemetryListener telemetry;
  // Never activated and never read(): the topic is independent of read()/write().
  ASSERT_TRUE(WaitFor([&]() {return telemetry.Carts().size() >= 5;}, 4000ms));

  // Move joint 1 mid-stream: the cart x (which follows joint 1 in the fake) must change with the
  // joints in every single sample, i.e. the pair really comes from one STATUS.
  server.NudgeJointDeg(0, 7.0f);
  const float expectedJ1Deg = server.baseJointsDeg[0] + 7.0f;
  ASSERT_TRUE(WaitFor([&]() {
        const auto carts = telemetry.Carts();
        return !carts.empty() &&
        std::fabs(carts.back().joint_position_rad[0] - expectedJ1Deg * kDegToRad) < 1e-4;
      }, 2000ms));

  const auto carts = telemetry.Carts();
  bool sawBefore = false;
  bool sawAfter = false;
  for (const auto & cart : carts) {
    // Raw, unconverted values exactly as sent (y..c constant; x = 300 + joint 1 [deg]).
    constexpr auto base = FakeCriServer::kCartPoseBase;
    for (std::size_t i = 1; i < 6; ++i) {
      EXPECT_NEAR(cart.pos_cart_robot_raw[i], base[i], 1e-4) << "index " << i;
    }
    const double j1Deg = cart.joint_position_rad[0] / kDegToRad;
    EXPECT_NEAR(cart.pos_cart_robot_raw[0], base[0] + j1Deg, 1e-3);
    sawBefore = sawBefore || std::fabs(j1Deg - server.baseJointsDeg[0]) < 1e-3;
    sawAfter = sawAfter || std::fabs(j1Deg - expectedJ1Deg) < 1e-3;
    for (std::size_t i = 1; i < 6; ++i) {
      EXPECT_NEAR(cart.joint_position_rad[i], server.baseJointsDeg[i] * kDegToRad, 1e-5);
    }
    EXPECT_EQ(cart.mode, "joint");
    EXPECT_EQ(cart.kinstate, 0);
    EXPECT_GT(rclcpp::Time(cart.header.stamp).nanoseconds(), 0);
  }
  EXPECT_TRUE(sawBefore);
  EXPECT_TRUE(sawAfter);
  for (std::size_t i = 1; i < carts.size(); ++i) {
    EXPECT_GT(
      rclcpp::Time(carts[i].header.stamp).nanoseconds(),
      rclcpp::Time(carts[i - 1].header.stamp).nanoseconds());
  }

  // The stamp equals that of the robot_joint_states sample of the same STATUS.
  const auto joints = telemetry.Joints();
  std::size_t matched = 0;
  for (const auto & cart : carts) {
    for (const auto & joint : joints) {
      if (rclcpp::Time(joint.msg.header.stamp) == rclcpp::Time(cart.header.stamp)) {
        ++matched;
        for (std::size_t i = 0; i < 6; ++i) {
          EXPECT_DOUBLE_EQ(joint.msg.position[i], cart.joint_position_rad[i]);
        }
        break;
      }
    }
  }
  EXPECT_GT(matched, 0u);

  // A reported KINSTATE comes through.
  server.SetKinstate(99);
  EXPECT_TRUE(WaitFor([&]() {
        const auto latest = telemetry.Carts();
        return !latest.empty() && latest.back().kinstate == 99;
      }, 1000ms));
}

TEST_F(RebelDriverTest, RobotJointStatesTrackTheRobotWhileLatchedAndInterfacesStayFrozen)
{
  FakeCriServer server;
  StartDriver(server);
  ASSERT_EQ(rebel->on_activate(rclcpp_lifecycle::State()), CallbackReturn::SUCCESS);
  TelemetryListener telemetry;
  ControllerLoop loop(*rebel, commands, 0.0);
  ASSERT_TRUE(WaitFor([&]() {return telemetry.JointCount() >= 5 && telemetry.StateCount() >= 2;}, 4000ms));
  EXPECT_TRUE(telemetry.WaitState([](const auto & msg) {return msg.activated;}, 500ms));
  EXPECT_FALSE(telemetry.LastState().latched);

  // Latch through a connection drop: read() returns ERROR, the loop stops (as ros2_control
  // stops calling read()) and on_error() follows.
  server.DropClient();
  ASSERT_TRUE(WaitFor([&]() {return loop.errors.load() > 0;}, 4000ms));
  ASSERT_TRUE(rebel->HardwareErrorLatched());
  const auto latchedAt = Clock::now();
  EXPECT_EQ(rebel->on_error(rclcpp_lifecycle::State()), CallbackReturn::SUCCESS);

  // hardware_state reports the latch immediately (poll period 10 ms), with the reason.
  ASSERT_TRUE(telemetry.WaitState([](const auto & msg) {return msg.latched;}, 200ms));
  const auto hw = telemetry.LastState();
  EXPECT_TRUE(hw.latched);
  EXPECT_NE(hw.latch_reason.find("re-established"), std::string::npos) << hw.latch_reason;
  EXPECT_TRUE(hw.connected);
  EXPECT_LT(Clock::now() - latchedAt, 400ms);
  // on_error() deactivates the component: also reported at once.
  EXPECT_TRUE(telemetry.WaitState([](const auto & msg) {return !msg.activated;}, 100ms));

  // The state interfaces are frozen at their last values ...
  const double frozen = Position(0);
  EXPECT_NEAR(frozen, server.baseJointsDeg[0] * kDegToRad, 1e-5);
  // ... while the robot moves on (the fake server keeps sending STATUS): robot_joint_states
  // tracks it, stamped with the samples' receive time.
  for (const float delta : {5.0f, 3.0f}) {
    server.NudgeJointDeg(0, delta);
    const double expected = (server.CurrentDeg()[0]) * kDegToRad;
    ASSERT_TRUE(
      telemetry.WaitJoint(
        [&](const auto & sample) {
          return std::fabs(sample.msg.position[0] - expected) < 1e-4;
        }, 500ms)) << "robot_joint_states did not follow the robot while latched";
    const auto sample = telemetry.LastJoint();
    const std::int64_t stampNs = rclcpp::Time(sample.msg.header.stamp).nanoseconds();
    EXPECT_LE(stampNs, sample.receivedSystemNs);
    EXPECT_LT(sample.receivedSystemNs - stampNs, 100'000'000);
  }
  EXPECT_EQ(Position(0), frozen);  // nothing wrote the state interfaces from another thread
  EXPECT_NEAR((server.CurrentDeg()[0] - server.baseJointsDeg[0]) * kDegToRad, 8.0 * kDegToRad, 1e-5);

  // Samples keep coming at the STATUS rate with strictly increasing stamps, and every one
  // carries a velocity estimate once a baseline exists.
  const std::size_t from = telemetry.JointCount();
  std::this_thread::sleep_for(500ms);
  const auto joints = telemetry.Joints();
  ASSERT_GT(joints.size(), from + 10);
  for (std::size_t i = from + 1; i < joints.size(); ++i) {
    EXPECT_GT(
      rclcpp::Time(joints[i].msg.header.stamp).nanoseconds(),
      rclcpp::Time(joints[i - 1].msg.header.stamp).nanoseconds());
    EXPECT_EQ(joints[i].msg.velocity.size(), 6u);
  }
  EXPECT_TRUE(telemetry.LastState().latched);
}

TEST_F(RebelDriverTest, DeadLinkShowsAsAgeingStampsAndRecoveryClearsTheLatchReport)
{
  FakeCriServer server;
  StartDriver(server);
  ASSERT_EQ(rebel->on_activate(rclcpp_lifecycle::State()), CallbackReturn::SUCCESS);
  TelemetryListener telemetry;
  ControllerLoop loop(*rebel, commands, 0.0);
  ASSERT_TRUE(WaitFor([&]() {return telemetry.JointCount() >= 5 && telemetry.StateCount() >= 2;}, 4000ms));

  // STATUS stops: read() goes through hold and latches after about 1 s.
  server.SetSendStatus(false);
  const auto silentAt = Clock::now();
  bool sawStale = false;
  ASSERT_TRUE(
    WaitFor(
      [&]() {
        sawStale = sawStale || !telemetry.LastState().status_fresh;
        return rebel->HardwareErrorLatched();
      }, 2500ms));
  EXPECT_TRUE(sawStale) << "status_fresh never went false before the latch";
  EXPECT_EQ(rebel->on_error(rclcpp_lifecycle::State()), CallbackReturn::SUCCESS);
  ASSERT_TRUE(telemetry.WaitState([](const auto & msg) {return msg.latched;}, 200ms));
  EXPECT_NE(telemetry.LastState().latch_reason.find("no valid STATUS"), std::string::npos)
    << telemetry.LastState().latch_reason;

  // No stale republish: robot_joint_states stops (at most the in-flight sample), its last stamp
  // ages, and hardware_state keeps reporting a growing status age.
  std::this_thread::sleep_for(100ms);
  const std::size_t joints0 = telemetry.JointCount();
  const auto lastJoint = telemetry.LastJoint();
  const std::int64_t lastStampNs = rclcpp::Time(lastJoint.msg.header.stamp).nanoseconds();
  const std::size_t states0 = telemetry.StateCount();
  const double age0 = telemetry.LastState().status_age_s;
  std::this_thread::sleep_for(600ms);
  EXPECT_EQ(telemetry.JointCount(), joints0) << "an old sample was republished";
  EXPECT_EQ(rclcpp::Time(telemetry.LastJoint().msg.header.stamp).nanoseconds(), lastStampNs);
  EXPECT_GE(SystemNowNs() - lastStampNs, 600'000'000);
  EXPECT_GE(telemetry.StateCount() - states0, 4u);
  const auto hw = telemetry.LastState();
  EXPECT_GE(hw.status_age_s - age0, 0.5);
  EXPECT_GE(hw.status_age_s, 1.0);  // the latch itself needs > 1 s without STATUS
  EXPECT_GE(Clock::now() - silentAt, 1600ms);
  EXPECT_FALSE(hw.status_fresh);
  EXPECT_TRUE(hw.latched);
  EXPECT_GT(hw.status_sequence, 0u);
  // status_stamp is the receive time of the latest STATUS: that of the last published sample
  // (a best-effort sample may have been dropped, never a newer one).
  EXPECT_GE(rclcpp::Time(hw.status_stamp).nanoseconds(), lastStampNs);
  EXPECT_LT(rclcpp::Time(hw.status_stamp).nanoseconds() - lastStampNs, 100'000'000);

  // Recovery: STATUS back, inactive -> active. The latch report clears, samples resume.
  server.SetSendStatus(true);
  ASSERT_TRUE(telemetry.WaitJoint([&](const auto & sample) {
        return sample.received > Clock::now() - 100ms;
      }, 1000ms));
  EXPECT_EQ(rebel->on_configure(rclcpp_lifecycle::State()), CallbackReturn::SUCCESS);
  EXPECT_TRUE(telemetry.LastState().latched);  // still latched until on_activate()
  EXPECT_EQ(rebel->on_activate(rclcpp_lifecycle::State()), CallbackReturn::SUCCESS);
  ASSERT_TRUE(telemetry.WaitState([](const auto & msg) {return !msg.latched;}, 100ms));
  const auto recovered = telemetry.LastState();
  EXPECT_TRUE(recovered.latch_reason.empty());
  EXPECT_TRUE(recovered.activated);
  EXPECT_TRUE(recovered.status_fresh);
  EXPECT_LT(recovered.status_age_s, 0.2);
}

TEST_F(RebelDriverTest, ResetAndEnableRefusesWhenStatusStale)
{
  FakeCriServer server;
  StartDriver(server);
  ASSERT_TRUE(WaitFor([&]() {return server.Commands(1).size() >= 6;}, 2000ms));

  server.SetSendStatus(false);
  std::this_thread::sleep_for(300ms);
  std::string message;
  EXPECT_FALSE(rebel->ResetAndEnable(message));
  EXPECT_NE(message.find("stale"), std::string::npos) << message;
  EXPECT_EQ(server.Commands(1), kStartupSequence);  // nothing extra was sent
}

TEST_F(RebelDriverTest, ResetAndEnableRefusesWhenLatched)
{
  FakeCriServer server;
  StartDriver(server);
  ASSERT_TRUE(WaitFor([&]() {return server.Commands(1).size() >= 6;}, 2000ms));

  server.SetSendStatus(false);
  ASSERT_TRUE(
    WaitFor([&]() {return rebel->read(now, period) == return_type::ERROR;}, 2500ms));
  server.SetSendStatus(true);  // STATUS fresh again, but the latch stays
  std::this_thread::sleep_for(100ms);
  std::string message;
  EXPECT_FALSE(rebel->ResetAndEnable(message));
  EXPECT_NE(message.find("latched"), std::string::npos) << message;
  EXPECT_EQ(server.Commands(1), kStartupSequence);
}

TEST_F(RebelDriverTest, DigitalOutputReportsConnectionState)
{
  FakeCriServer server;
  StartDriver(server);
  ASSERT_TRUE(WaitFor([&]() {return server.Commands().size() >= 6;}, 2000ms));

  auto request = std::make_shared<igus_rebel_msgs::srv::SetDigitalOutput::Request>();
  auto response = std::make_shared<igus_rebel_msgs::srv::SetDigitalOutput::Response>();
  request->output.output = 3;
  request->output.is_on = true;
  rebel->dio_callback(request, response);
  EXPECT_TRUE(response->success);
  ASSERT_TRUE(WaitFor([&]() {return server.Commands().back() == "DOUT 3 true";}, 1000ms));

  server.Shutdown();
  // Wait until the driver notices the closed connection.
  std::this_thread::sleep_for(300ms);
  rebel->dio_callback(request, response);
  EXPECT_FALSE(response->success);
}

TEST_F(RebelDriverTest, StopIsBoundedWhileDisconnected)
{
  FakeCriServer server;
  StartDriver(server);
  server.Shutdown();
  std::this_thread::sleep_for(300ms);

  const auto start = Clock::now();
  rebel->Stop();
  EXPECT_LT(Clock::now() - start, 2500ms);
}

TEST(RebelDriverInit, FailsWhenNoStatusArrives)
{
  FakeCriServer server;
  server.SetSendStatus(false);
  Igus::Rebel rebel("127.0.0.1", server.port());

  const auto start = Clock::now();
  EXPECT_EQ(rebel.on_init(MakeInfo()), CallbackReturn::ERROR);
  const auto elapsed = Clock::now() - start;
  EXPECT_GE(elapsed, Igus::Safety::kFirstStatusTimeout - 100ms);
  EXPECT_LT(elapsed, Igus::Safety::kFirstStatusTimeout + 2500ms);
  // on_init already stopped the driver: nothing is sent any more.
  // Stop() legitimately sends Disable and Quit just before on_init returns; the fake server
  // may only read them from the loopback socket after on_init returned, so let those drain,
  // then require that nothing more (in particular no ALIVEJOG) arrives.
  std::this_thread::sleep_for(150ms);
  const auto framesAfterInit = server.Frames().size();
  std::this_thread::sleep_for(150ms);
  EXPECT_EQ(server.Frames().size(), framesAfterInit);
  const auto frames = server.Frames();
  ASSERT_GE(frames.size(), 2u);
  EXPECT_EQ(frames[frames.size() - 2].body, "CMD Disable");
  EXPECT_EQ(frames[frames.size() - 1].body, "CMD QUIT");
}

TEST(RebelDriverInit, DefaultTargetIsWiredLanAddress)
{
  Igus::Rebel rebel;
  EXPECT_EQ(rebel.ip, "192.168.3.11");
  EXPECT_EQ(rebel.port, 3920);
}

TEST(RebelDriverInit, HardwareParametersOverrideTarget)
{
  FakeCriServer server;
  // Constructed with an unused port; the hardware parameters must redirect it.
  Igus::Rebel rebel("10.255.255.1", 1);
  auto info = MakeInfo();
  info.hardware_parameters["robot_ip"] = "127.0.0.1";
  info.hardware_parameters["robot_port"] = std::to_string(server.port());
  ASSERT_EQ(rebel.on_init(info), CallbackReturn::SUCCESS);
  EXPECT_EQ(rebel.ip, "127.0.0.1");
  EXPECT_EQ(rebel.port, server.port());
  rebel.Stop();
}

TEST(RebelDriverInit, RejectsInvalidTargetParameters)
{
  for (const auto & bad : std::vector<std::pair<std::string, std::string>>{
    {"robot_ip", "192.168.3"}, {"robot_ip", "robot.local"}, {"robot_port", "0"},
    {"robot_port", "70000"}, {"robot_port", "39x"}})
  {
    Igus::Rebel rebel("127.0.0.1", 1);
    auto info = MakeInfo();
    info.hardware_parameters[bad.first] = bad.second;
    const auto start = Clock::now();
    EXPECT_EQ(rebel.on_init(info), CallbackReturn::ERROR) << bad.first << "=" << bad.second;
    EXPECT_LT(Clock::now() - start, 500ms) << "must fail before connecting";
  }
}

TEST(RebelDriverInit, FailsWhenConnectionIsRefused)
{
  FakeCriServer closedPort(false);
  Igus::Rebel rebel("127.0.0.1", closedPort.port());

  const auto start = Clock::now();
  EXPECT_EQ(rebel.on_init(MakeInfo()), CallbackReturn::ERROR);
  EXPECT_LT(Clock::now() - start, Igus::Safety::kFirstStatusTimeout + 2500ms);
}

TEST(RebelDriverInit, DefaultsKeepSingleArmNames)
{
  FakeCriServer server;
  Igus::Rebel rebel("127.0.0.1", server.port());
  ASSERT_EQ(rebel.on_init(MakeInfo()), CallbackReturn::SUCCESS);
  const auto states = rebel.export_state_interfaces();
  const auto commands = rebel.export_command_interfaces();
  ASSERT_EQ(states.size(), 12u);
  ASSERT_EQ(commands.size(), 6u);
  for (std::size_t i = 0; i < 6; ++i) {
    const std::string joint = "joint" + std::to_string(i + 1);
    EXPECT_EQ(states[2 * i].get_name(), joint + "/position");
    EXPECT_EQ(states[2 * i + 1].get_name(), joint + "/velocity");
    EXPECT_EQ(commands[i].get_name(), joint + "/velocity");
  }
  rebel.Stop();
}

TEST(RebelDriverInit, RejectsWrongJointDescriptionBeforeConnecting)
{
  const auto expectRejected = [](
    const std::string & what, const std::function<void(hardware_interface::HardwareInfo &)> & edit)
    {
      FakeCriServer server;
      Igus::Rebel rebel("127.0.0.1", server.port());
      auto info = MakeInfo();
      edit(info);
      const auto start = Clock::now();
      EXPECT_EQ(rebel.on_init(info), CallbackReturn::ERROR) << what;
      EXPECT_LT(Clock::now() - start, 500ms) << what << ": must fail before connecting";
      std::this_thread::sleep_for(100ms);
      EXPECT_EQ(server.Connections(), 0u) << what << ": must not have connected";
    };
  expectRejected("no joints", [](auto & info) {info.joints.clear();});
  expectRejected("five joints", [](auto & info) {info.joints.pop_back();});
  expectRejected(
    "seven joints", [](auto & info) {
      auto extra = info.joints.back();
      extra.name = "joint7";
      info.joints.push_back(extra);
    });
  expectRejected(
    "joints out of CRI order", [](auto & info) {std::swap(info.joints[0], info.joints[1]);});
  expectRejected(
    "duplicate joint name", [](auto & info) {info.joints[5].name = info.joints[4].name;});
  expectRejected(
    "position command interface",
    [](auto & info) {info.joints[2].command_interfaces[0].name = "position";});
  expectRejected(
    "missing command interface", [](auto & info) {info.joints[2].command_interfaces.clear();});
  expectRejected(
    "missing velocity state", [](auto & info) {info.joints[3].state_interfaces.pop_back();});
  expectRejected(
    "extra state interface",
    [](auto & info) {
      hardware_interface::InterfaceInfo effort;
      effort.name = "effort";
      info.joints[3].state_interfaces.push_back(effort);
    });
  expectRejected(
    "invalid node name", [](auto & info) {info.hardware_parameters["node_name"] = "bad name";});
  expectRejected(
    "invalid digital output service",
    [](auto & info) {info.hardware_parameters["digital_output_service"] = "bad service";});
  expectRejected(
    "non-numeric move velocity cap",
    [](auto & info) {info.hardware_parameters["move_joint_max_velocity_percent"] = "fast";});
  expectRejected(
    "zero move acceleration cap",
    [](auto & info) {info.hardware_parameters["move_joint_max_acceleration_percent"] = "0";});
  expectRejected(
    "negative move velocity cap",
    [](auto & info) {info.hardware_parameters["move_joint_max_velocity_percent"] = "-5";});
}

// Two Igus/Rebel instances in one controller_manager (dual-arm cell): prefixed joints, one fake
// controller each, distinct node names.
class RebelDualArmTest : public ::testing::Test
{
protected:
  struct Arm
  {
    std::unique_ptr<FakeCriServer> server;
    std::unique_ptr<Igus::Rebel> rebel;
    std::vector<hardware_interface::StateInterface> states;
    std::vector<hardware_interface::CommandInterface> commands;
  };

  void StartArm(Arm & arm, const std::string & side, bool explicitDigitalOutput = true)
  {
    arm.server = std::make_unique<FakeCriServer>();
    arm.rebel = std::make_unique<Igus::Rebel>("10.255.255.1", 1);
    auto info = MakeInfo("arm_" + side + "_controller", "arm_" + side + "_");
    info.hardware_parameters["robot_ip"] = "127.0.0.1";
    info.hardware_parameters["robot_port"] = std::to_string(arm.server->port());
    info.hardware_parameters["node_name"] = "igus_rebel_" + side;
    if (explicitDigitalOutput) {
      info.hardware_parameters["digital_output_service"] = "set_digital_output_" + side;
    }
    info.hardware_parameters["trajectory_controller_name"] = "arm_" + side + "_jtc";
    ASSERT_EQ(arm.rebel->on_init(info), CallbackReturn::SUCCESS);
    arm.states = arm.rebel->export_state_interfaces();
    arm.commands = arm.rebel->export_command_interfaces();
  }

  rclcpp::Time now{0, 0, RCL_STEADY_TIME};
  rclcpp::Duration period{0, 10000000};
};

TEST_F(RebelDualArmTest, TwoPrefixedInstancesRunIndependently)
{
  Arm left;
  Arm right;
  StartArm(left, "left");
  ASSERT_NE(left.rebel, nullptr);
  // The two arms report different poses so that mixed-up state would be visible.
  StartArm(right, "right");
  ASSERT_NE(right.rebel, nullptr);
  ASSERT_NE(left.server->port(), right.server->port());

  // Interface names are the prefixed joint names.
  for (const std::string side : {"left", "right"}) {
    Arm & arm = side == "left" ? left : right;
    ASSERT_EQ(arm.states.size(), 12u);
    ASSERT_EQ(arm.commands.size(), 6u);
    for (std::size_t i = 0; i < 6; ++i) {
      const std::string joint = "arm_" + side + "_joint" + std::to_string(i + 1);
      EXPECT_EQ(arm.states[2 * i].get_name(), joint + "/position");
      EXPECT_EQ(arm.states[2 * i + 1].get_name(), joint + "/velocity");
      EXPECT_EQ(arm.commands[i].get_name(), joint + "/velocity");
    }
  }

  // Both ran their own start-up sequence against their own server.
  ASSERT_TRUE(WaitFor([&]() {return left.server->Commands().size() >= 6;}, 2000ms));
  ASSERT_TRUE(WaitFor([&]() {return right.server->Commands().size() >= 6;}, 2000ms));
  EXPECT_EQ(left.server->Commands(), kStartupSequence);
  EXPECT_EQ(right.server->Commands(), kStartupSequence);
  EXPECT_EQ(left.rebel->read(now, period), return_type::OK);
  EXPECT_EQ(right.rebel->read(now, period), return_type::OK);

  // A jog command on the left arm reaches only the left server.
  const auto jogStart = Clock::now();
  for (int i = 0; i < 20; ++i) {
    left.commands.at(0).set_value(0.1);
    ASSERT_EQ(left.rebel->write(now, period), return_type::OK);
    ASSERT_EQ(right.rebel->write(now, period), return_type::OK);
    std::this_thread::sleep_for(10ms);
  }
  const auto leftJogs = left.server->AliveJogs(jogStart + 100ms);
  ASSERT_FALSE(leftJogs.empty());
  EXPECT_GT(ParseJog(leftJogs.back().body)[0], 5.0);
  EXPECT_TRUE(AnyNonZeroJog(*left.server, jogStart + 100ms));
  EXPECT_FALSE(AnyNonZeroJog(*right.server, jogStart));

  // Independent stale-STATUS handling: the left STATUS stream stops, only left latches.
  left.server->SetSendStatus(false);
  ASSERT_TRUE(
    WaitFor(
      [&]() {
        left.rebel->write(now, period);
        return left.rebel->read(now, period) == return_type::ERROR;
      }, 2500ms));
  EXPECT_EQ(left.rebel->on_error(rclcpp_lifecycle::State()), CallbackReturn::SUCCESS);
  EXPECT_TRUE(left.rebel->HardwareErrorLatched());
  EXPECT_FALSE(right.rebel->HardwareErrorLatched());
  EXPECT_EQ(right.rebel->read(now, period), return_type::OK);
  EXPECT_FALSE(right.rebel->StatusHoldActive());
  EXPECT_FALSE(right.rebel->StatusCoastActive());

  // The right arm keeps accepting jog while the left one is zeroed.
  right.commands.at(1).set_value(0.1);
  const auto rightStart = Clock::now();
  for (int i = 0; i < 20; ++i) {
    right.commands.at(1).set_value(0.1);
    ASSERT_EQ(right.rebel->write(now, period), return_type::OK);
    std::this_thread::sleep_for(10ms);
  }
  EXPECT_TRUE(AnyNonZeroJog(*right.server, rightStart + 100ms));
  EXPECT_FALSE(AnyNonZeroJog(*left.server, Clock::now() - 100ms));

  // Digital output goes to the right server only.
  auto request = std::make_shared<igus_rebel_msgs::srv::SetDigitalOutput::Request>();
  auto response = std::make_shared<igus_rebel_msgs::srv::SetDigitalOutput::Response>();
  request->output.output = 2;
  request->output.is_on = true;
  right.rebel->dio_callback(request, response);
  EXPECT_TRUE(response->success);
  ASSERT_TRUE(WaitFor([&]() {return right.server->Commands().back() == "DOUT 2 true";}, 1000ms));
  const auto leftCommands = left.server->Commands();
  EXPECT_EQ(
    std::count(leftCommands.begin(), leftCommands.end(), std::string("DOUT 2 true")), 0);

  // Stopping one instance does not disturb the other.
  left.rebel->Stop();
  EXPECT_EQ(right.rebel->read(now, period), return_type::OK);
  right.rebel->Stop();
}

TEST_F(RebelDualArmTest, ServiceAndNodeNamesFollowParameters)
{
  Arm left;
  Arm right;
  StartArm(left, "left");
  ASSERT_NE(left.rebel, nullptr);
  StartArm(right, "right");
  ASSERT_NE(right.rebel, nullptr);

  // Two nodes with distinct names exist in this process's context, each with its own services.
  auto probe = std::make_shared<rclcpp::Node>("dual_arm_probe");
  EXPECT_TRUE(
    WaitFor(
      [&]() {
        const auto names = probe->get_service_names_and_types();
        return names.count("/igus_rebel_left/reset_and_enable") == 1 &&
        names.count("/igus_rebel_right/reset_and_enable") == 1 &&
        names.count("/set_digital_output_left") == 1 &&
        names.count("/set_digital_output_right") == 1;
      }, 3000ms));
  const auto names = probe->get_service_names_and_types();
  EXPECT_EQ(names.count("/igus_rebel/reset_and_enable"), 0u);
  EXPECT_EQ(names.count("/set_digital_output"), 0u);
  left.rebel->Stop();
  right.rebel->Stop();
}

// Each arm publishes its own telemetry in its own namespace, with its prefixed joint names; one
// arm's latch is not reported by the other.
TEST_F(RebelDualArmTest, TelemetryIsPerArmWithPrefixedJointNames)
{
  Arm left;
  Arm right;
  StartArm(left, "left");
  ASSERT_NE(left.rebel, nullptr);
  StartArm(right, "right");
  ASSERT_NE(right.rebel, nullptr);
  right.server->NudgeJointDeg(0, 7.0f);  // distinguishable from the left arm
  TelemetryListener leftTelemetry("igus_rebel_left");
  TelemetryListener rightTelemetry("igus_rebel_right");
  ASSERT_TRUE(
    WaitFor(
      [&]() {
        return leftTelemetry.JointCount() >= 3 && rightTelemetry.JointCount() >= 3 &&
        leftTelemetry.StateCount() >= 1 && rightTelemetry.StateCount() >= 1;
      }, 4000ms));
  const auto leftJoint = leftTelemetry.LastJoint();
  const auto rightJoint = rightTelemetry.LastJoint();
  ASSERT_EQ(leftJoint.msg.name.size(), 6u);
  ASSERT_EQ(rightJoint.msg.name.size(), 6u);
  for (std::size_t i = 0; i < 6; ++i) {
    EXPECT_EQ(leftJoint.msg.name[i], "arm_left_joint" + std::to_string(i + 1));
    EXPECT_EQ(rightJoint.msg.name[i], "arm_right_joint" + std::to_string(i + 1));
  }
  EXPECT_NEAR(leftJoint.msg.position[0], left.server->baseJointsDeg[0] * kDegToRad, 1e-5);
  EXPECT_TRUE(
    rightTelemetry.WaitJoint(
      [&](const auto & sample) {
        return std::fabs(
          sample.msg.position[0] - (right.server->baseJointsDeg[0] + 7.0f) * kDegToRad) < 1e-4;
      }, 500ms));
  EXPECT_EQ(leftTelemetry.LastState().component, "arm_left_controller");
  EXPECT_EQ(rightTelemetry.LastState().component, "arm_right_controller");

  // Latch only the left arm (no fault group here): only its report changes.
  left.rebel->on_error(rclcpp_lifecycle::State());
  ASSERT_TRUE(leftTelemetry.WaitState([](const auto & msg) {return msg.latched;}, 200ms));
  std::this_thread::sleep_for(150ms);
  EXPECT_FALSE(rightTelemetry.LastState().latched);
  left.rebel->Stop();
  right.rebel->Stop();
}

TEST_F(RebelDualArmTest, RenamedNodesDefaultToPrivateDigitalOutputService)
{
  Arm left;
  Arm right;
  StartArm(left, "left", false);
  ASSERT_NE(left.rebel, nullptr);
  StartArm(right, "right", false);
  ASSERT_NE(right.rebel, nullptr);

  // Without digital_output_service, a renamed node must not advertise the global service.
  auto probe = std::make_shared<rclcpp::Node>("dual_arm_private_probe");
  EXPECT_TRUE(
    WaitFor(
      [&]() {
        const auto names = probe->get_service_names_and_types();
        return names.count("/igus_rebel_left/set_digital_output") == 1 &&
        names.count("/igus_rebel_right/set_digital_output") == 1;
      }, 3000ms));
  EXPECT_EQ(probe->get_service_names_and_types().count("/set_digital_output"), 0u);
  left.rebel->Stop();
  right.rebel->Stop();
}

TEST(RebelSocketTest, SendFailsFastWhileDisconnectedAndStopIsBounded)
{
  FakeCriServer closedPort(false);
  Igus::RebelSocket socket("127.0.0.1", closedPort.port());
  socket.Start();
  std::this_thread::sleep_for(200ms);
  EXPECT_FALSE(socket.IsConnected());

  auto start = Clock::now();
  EXPECT_FALSE(socket.SendMessage("CRISTART 1 CMD Reset CRIEND\n"));
  EXPECT_LT(Clock::now() - start, 50ms);

  start = Clock::now();
  socket.Stop();
  EXPECT_LT(Clock::now() - start, 1500ms);

  std::string message;
  EXPECT_FALSE(socket.WaitForMessage(message, 10ms));
}

TEST(RebelSocketTest, ReceivesSplitFramesInOrderAndCallsBackOnConnect)
{
  FakeCriServer server;
  Igus::RebelSocket socket("127.0.0.1", server.port());
  std::atomic<int> callbacks{0};
  socket.SetConnectedCallback([&](bool) {++callbacks;});
  socket.Start();

  std::string message;
  int statuses = 0;
  const auto deadline = Clock::now() + 2s;
  while (statuses < 5 && Clock::now() < deadline) {
    if (socket.WaitForMessage(message, 100ms)) {
      if (message.find(" STATUS MODE joint") != std::string::npos) {
        ++statuses;
        EXPECT_NE(message.find("KINSTATE 0 OPMODE -1"), std::string::npos);
      }
    }
  }
  EXPECT_EQ(statuses, 5);
  EXPECT_EQ(callbacks.load(), 1);
  EXPECT_TRUE(socket.SendMessage("CRISTART 1 CMD Reset CRIEND\n"));
  EXPECT_TRUE(WaitFor([&]() {return !server.Commands().empty();}, 1000ms));

  server.DropClient();
  EXPECT_TRUE(WaitFor([&]() {return callbacks.load() == 2;}, 3000ms));
  socket.Stop();
}

TEST(RebelSocketTest, StampsTheReceiveTimeAndCountsDroppedMessages)
{
  FakeCriServer server;
  Igus::RebelSocket socket("127.0.0.1", server.port());
  socket.Start();
  ASSERT_TRUE(WaitFor([&]() {return socket.IsConnected();}, 2000ms));
  EXPECT_EQ(socket.DroppedMessageCount(), 0u);
  // Nobody consumes: the 20 ms STATUS stream overflows the 25-message queue.
  ASSERT_TRUE(WaitFor([&]() {return socket.DroppedMessageCount() > 0;}, 3000ms));
  server.SetSendStatus(false);
  std::this_thread::sleep_for(400ms);
  const auto popAt = Clock::now();
  const std::int64_t popSystemNs = SystemNowNs();

  Igus::RebelSocket::ReceivedMessage message;
  std::size_t count = 0;
  auto previous = Clock::time_point{};
  while (socket.WaitForMessage(message, 10ms)) {
    ++count;
    EXPECT_GE(message.received, previous);
    previous = message.received;
    // Stamped by the receive thread, not when it is popped: all were read >= ~400 ms ago.
    EXPECT_LT(message.received, popAt - 300ms);
    EXPECT_LT(message.systemNs, popSystemNs - 300'000'000);
  }
  EXPECT_EQ(count, Igus::RebelSocket::kMaxUnprocessedMessages);
  socket.Stop();
}

//
// Fault groups: several Igus/Rebel instances in one process that stop together.
//
namespace
{
// Collects ROS log lines of severity ERROR or above while it exists.
class LogCapture
{
public:
  LogCapture()
  {
    Previous() = rcutils_logging_get_output_handler();
    rcutils_logging_set_output_handler(&LogCapture::Handler);
  }
  ~LogCapture() {rcutils_logging_set_output_handler(Previous());}

  bool Contains(const std::string & text)
  {
    std::lock_guard<std::mutex> lock(Mutex());
    for (const auto & line : Lines()) {
      if (line.find(text) != std::string::npos) {
        return true;
      }
    }
    return false;
  }

private:
  static rcutils_logging_output_handler_t & Previous()
  {
    static rcutils_logging_output_handler_t previous = nullptr;
    return previous;
  }
  static std::mutex & Mutex()
  {
    static std::mutex mutex;
    return mutex;
  }
  static std::vector<std::string> & Lines()
  {
    static std::vector<std::string> lines;
    return lines;
  }
  static void Handler(
    const rcutils_log_location_t * location, int severity, const char * name,
    rcutils_time_point_value_t timestamp, const char * format, va_list * args)
  {
    if (severity >= RCUTILS_LOG_SEVERITY_ERROR) {
      va_list copy;
      va_copy(copy, *args);
      char buffer[2048];
      std::vsnprintf(buffer, sizeof(buffer), format, copy);
      va_end(copy);
      std::lock_guard<std::mutex> lock(Mutex());
      Lines().push_back(std::string(name != nullptr ? name : "") + ": " + buffer);
    }
    if (Previous() != nullptr) {
      Previous()(location, severity, name, timestamp, format, args);
    }
  }
};

bool AllJogsZero(FakeCriServer & server, Clock::time_point after, std::size_t * count = nullptr)
{
  std::size_t frames = 0;
  for (const auto & frame : server.AliveJogs(after)) {
    ++frames;
    for (double value : ParseJog(frame.body)) {
      if (value != 0.0) {
        return false;
      }
    }
  }
  if (count != nullptr) {
    *count = frames;
  }
  return true;
}
}  // namespace

TEST_F(RebelDriverTest, WriteKeepsJogZeroWhileLatchedWithoutFaultGroup)
{
  FakeCriServer server;
  StartDriver(server);
  ASSERT_TRUE(WaitFor([&]() {return server.Commands().size() >= 6;}, 2000ms));

  commands.at(0).set_value(0.2);
  for (int i = 0; i < 10; ++i) {
    ASSERT_EQ(rebel->write(now, period), return_type::OK);
    std::this_thread::sleep_for(10ms);
  }
  ASSERT_TRUE(AnyNonZeroJog(server, Clock::now() - 60ms));

  // Latched (no fault_group): a controller that keeps calling write() with a non-zero command
  // must not get a non-zero jog to the robot, even though each write() refreshes the watchdog.
  rebel->on_error(rclcpp_lifecycle::State());
  ASSERT_TRUE(rebel->HardwareErrorLatched());
  const auto latchedTime = Clock::now();
  for (int i = 0; i < 25; ++i) {
    commands.at(0).set_value(0.2);
    rebel->write(now, period);
    std::this_thread::sleep_for(10ms);
  }
  std::size_t frames = 0;
  EXPECT_TRUE(AllJogsZero(server, latchedTime + 15ms, &frames));
  EXPECT_GE(frames, 5u);
}

TEST_F(RebelDriverTest, NonZeroWriteWhileLatchingNeverReachesTheRobot)
{
  FakeCriServer server;
  StartDriver(server);
  ASSERT_TRUE(WaitFor([&]() {return server.Commands().size() >= 6;}, 2000ms));

  std::atomic<bool> run{true};
  std::thread hammer([&]() {
      while (run) {
        for (auto & command : commands) {
          command.set_value(0.2);
        }
        rebel->write(now, period);
        std::this_thread::sleep_for(500us);
      }
    });
  ASSERT_TRUE(WaitFor([&]() {return AnyNonZeroJog(server, Clock::now() - 60ms);}, 2000ms));
  std::this_thread::sleep_for(50ms);

  // Latch from this thread while write() keeps being called with a non-zero command.
  rebel->on_error(rclcpp_lifecycle::State());
  const auto latchedAt = Clock::now();
  ASSERT_TRUE(rebel->HardwareErrorLatched());
  std::this_thread::sleep_for(300ms);
  run = false;
  hammer.join();

  std::size_t frames = 0;
  EXPECT_TRUE(AllJogsZero(server, latchedAt + 15ms, &frames));
  EXPECT_GE(frames, 10u);
}

class RebelFaultGroupTest : public ::testing::Test
{
protected:
  struct Arm
  {
    std::unique_ptr<FakeCriServer> server;
    std::unique_ptr<Igus::Rebel> rebel;
    std::vector<hardware_interface::StateInterface> states;
    std::vector<hardware_interface::CommandInterface> commands;
    bool errored = false;  // the emulated controller manager stopped calling read()/write()
  };

  // `disabledAtStart`: the fake robot starts with all drives disabled and ignores Enable until
  // told otherwise (the normal state of a real arm before its first enable).
  void StartArm(
    Arm & arm, const std::string & side, const std::string & group, bool disabledAtStart = false)
  {
    arm.server = std::make_unique<FakeCriServer>();
    if (disabledAtStart) {
      arm.server->SetDisabledMask(0x3F);
      arm.server->SetIgnoreEnable(true);
    }
    arm.rebel = std::make_unique<Igus::Rebel>("10.255.255.1", 1);
    auto info = MakeInfo("arm_" + side + "_controller", "arm_" + side + "_");
    info.hardware_parameters["robot_ip"] = "127.0.0.1";
    info.hardware_parameters["robot_port"] = std::to_string(arm.server->port());
    info.hardware_parameters["node_name"] = "igus_rebel_" + side;
    info.hardware_parameters["trajectory_controller_name"] = "arm_" + side + "_jtc";
    if (!group.empty()) {
      info.hardware_parameters["fault_group"] = group;
    }
    ASSERT_EQ(arm.rebel->on_init(info), CallbackReturn::SUCCESS);
    arm.states = arm.rebel->export_state_interfaces();
    arm.commands = arm.rebel->export_command_interfaces();
  }

  // One controller-manager cycle: read all, then write all; a component whose read() returns
  // ERROR gets on_error() and is not called again.
  void Cycle(Arm & left, Arm & right, double leftVelocity, double rightVelocity)
  {
    for (Arm * arm : {&left, &right}) {
      if (!arm->errored && arm->rebel->read(now, period) == return_type::ERROR) {
        arm->rebel->on_error(rclcpp_lifecycle::State());
        arm->errored = true;
      }
    }
    for (Arm * arm : {&left, &right}) {
      if (!arm->errored) {
        for (auto & command : arm->commands) {
          command.set_value(arm == &left ? leftVelocity : rightVelocity);
        }
        arm->rebel->write(now, period);
      }
    }
  }

  void RunCycles(Arm & left, Arm & right, int count, double leftVelocity, double rightVelocity)
  {
    for (int i = 0; i < count; ++i) {
      Cycle(left, right, leftVelocity, rightVelocity);
      std::this_thread::sleep_for(10ms);
    }
  }

  // Both arms enabled, monitored and driving the right arm.
  void StartArmedPair(Arm & left, Arm & right, const std::string & group)
  {
    StartArm(left, "left", group);
    ASSERT_NE(left.rebel, nullptr);
    StartArm(right, "right", group);
    ASSERT_NE(right.rebel, nullptr);
    if (group.empty()) {
      // Without a group nothing is monitored (never "armed"): just let the arms settle.
      std::this_thread::sleep_for(
        Igus::Safety::kFaultArmSettleTime + 200ms);
      return;
    }
    ASSERT_TRUE(
      WaitFor(
        [&]() {
          return left.rebel->FaultMonitorArmed() && right.rebel->FaultMonitorArmed();
        }, 3000ms));
  }

  // The right arm is commanded to move; after a stop of the group every ALIVEJOG it sends must
  // be zero.
  void ExpectRightStoppedAfter(Arm & right, Clock::time_point stopTime)
  {
    // The right arm's controller keeps (wrongly) calling write(): jog must stay zero.
    for (int i = 0; i < 15; ++i) {
      for (auto & command : right.commands) {
        command.set_value(0.1);
      }
      right.rebel->write(now, period);
      std::this_thread::sleep_for(10ms);
    }
    std::size_t frames = 0;
    EXPECT_TRUE(AllJogsZero(*right.server, stopTime + 15ms, &frames));
    EXPECT_GE(frames, 5u);
  }

  rclcpp::Time now{0, 0, RCL_STEADY_TIME};
  rclcpp::Duration period{0, 10000000};
};

TEST_F(RebelFaultGroupTest, WithoutGroupAFaultDoesNotStopTheOtherArm)
{
  Arm left;
  Arm right;
  StartArmedPair(left, right, "");  // no fault_group
  EXPECT_FALSE(left.rebel->FaultGroupRootCause());

  left.server->SetSendStatus(false);
  ASSERT_TRUE(
    WaitFor(
      [&]() {
        Cycle(left, right, 0.0, 0.1);
        return left.errored;
      }, 3000ms));
  EXPECT_TRUE(left.rebel->HardwareErrorLatched());
  EXPECT_FALSE(right.errored);
  EXPECT_FALSE(right.rebel->HardwareErrorLatched());
  EXPECT_EQ(right.rebel->read(now, period), return_type::OK);

  // Drive faults and kinematics errors are not monitored without a group either.
  right.server->SetDisabledMask(0x01);
  std::this_thread::sleep_for(200ms);
  EXPECT_EQ(right.rebel->read(now, period), return_type::OK);
  EXPECT_FALSE(right.rebel->HardwareErrorLatched());  // no self-latch on drive-disabled
  EXPECT_FALSE(right.rebel->FaultGroupRootCause());
  EXPECT_FALSE(right.rebel->FaultMonitorArmed());
}

TEST_F(RebelFaultGroupTest, StatusLossLatchesTheOtherArmInTheSameCycle)
{
  LogCapture log;
  Arm left;
  Arm right;
  StartArmedPair(left, right, "cell_status");

  // The right arm is moving; the left one is idle.
  RunCycles(left, right, 20, 0.0, 0.1);
  ASSERT_TRUE(AnyNonZeroJog(*right.server, Clock::now() - 100ms));
  EXPECT_FALSE(left.rebel->HardwareErrorLatched());
  EXPECT_FALSE(right.rebel->HardwareErrorLatched());

  left.server->SetSendStatus(false);
  ASSERT_TRUE(
    WaitFor(
      [&]() {
        // The right arm stays healthy for as long as the left one has not latched.
        EXPECT_FALSE(right.rebel->HardwareErrorLatched());
        Cycle(left, right, 0.0, 0.1);
        return left.errored;
      }, 3000ms));
  const auto stopTime = Clock::now();

  // Same cycle: the left read() latched and stopped the right arm, whose read() followed it.
  EXPECT_TRUE(left.rebel->HardwareErrorLatched());
  EXPECT_TRUE(right.rebel->HardwareErrorLatched());
  EXPECT_TRUE(right.errored);
  EXPECT_TRUE(left.rebel->FaultGroupRootCause());
  EXPECT_FALSE(right.rebel->FaultGroupRootCause());
  EXPECT_EQ(right.rebel->read(now, period), return_type::ERROR);
  ExpectRightStoppedAfter(right, stopTime);

  EXPECT_TRUE(log.Contains("stopped because fault-group member 'arm_left_controller' faulted"));
}

TEST_F(RebelFaultGroupTest, OnErrorOfOneMemberStopsTheOther)
{
  LogCapture log;
  Arm left;
  Arm right;
  StartArmedPair(left, right, "cell_on_error");
  RunCycles(left, right, 20, 0.0, 0.1);
  ASSERT_TRUE(AnyNonZeroJog(*right.server, Clock::now() - 100ms));

  EXPECT_EQ(left.rebel->on_error(rclcpp_lifecycle::State()), CallbackReturn::SUCCESS);
  const auto stopTime = Clock::now();
  EXPECT_TRUE(right.rebel->HardwareErrorLatched());
  EXPECT_TRUE(left.rebel->FaultGroupRootCause());
  EXPECT_FALSE(right.rebel->FaultGroupRootCause());
  EXPECT_EQ(right.rebel->read(now, period), return_type::ERROR);
  ExpectRightStoppedAfter(right, stopTime);
  EXPECT_TRUE(log.Contains("stopped because fault-group member 'arm_left_controller' faulted"));

  // ros2_control then calls on_error() on the right arm too: not a new root cause, no ping-pong.
  EXPECT_EQ(right.rebel->on_error(rclcpp_lifecycle::State()), CallbackReturn::SUCCESS);
  EXPECT_FALSE(right.rebel->FaultGroupRootCause());
  EXPECT_TRUE(left.rebel->FaultGroupRootCause());
}

class RebelFaultGroupDriveTest : public RebelFaultGroupTest
{
protected:
  // `fault` makes the fake left robot report a stop after it was enabled.
  void RunCase(const std::string & group, const std::function<void(FakeCriServer &)> & fault)
  {
    LogCapture log;
    Arm left;
    Arm right;
    StartArmedPair(left, right, group);
    RunCycles(left, right, 20, 0.0, 0.1);
    ASSERT_TRUE(AnyNonZeroJog(*right.server, Clock::now() - 100ms));

    fault(*left.server);
    const auto faultTime = Clock::now();
    Clock::time_point latchedAt;
    ASSERT_TRUE(
      WaitFor(
        [&]() {
          Cycle(left, right, 0.0, 0.1);
          latchedAt = Clock::now();
          return right.rebel->HardwareErrorLatched();
        }, 1000ms));
    // Within a few STATUS periods.
    EXPECT_LT(latchedAt - faultTime, 500ms);
    Cycle(left, right, 0.0, 0.1);  // the next controller-manager read() sees the latch
    EXPECT_TRUE(right.errored);
    EXPECT_TRUE(left.rebel->FaultGroupRootCause());
    // The faulting (root-cause) arm is latched too: fail closed, same recovery path as any latch.
    EXPECT_TRUE(left.rebel->HardwareErrorLatched());
    EXPECT_EQ(left.rebel->read(now, period), return_type::ERROR);
    ExpectRightStoppedAfter(left, latchedAt);
    ExpectRightStoppedAfter(right, latchedAt);
    EXPECT_TRUE(log.Contains("stopped because fault-group member 'arm_left_controller' faulted"));
  }
};

TEST_F(RebelFaultGroupDriveTest, MotorNotEnabledAfterEnableStopsTheOtherArm)
{
  RunCase("cell_mne", [](FakeCriServer & server) {server.SetDisabledMask(0x04);});
}

TEST_F(RebelFaultGroupDriveTest, SupplyLowEmergencyStopStopsTheOtherArm)
{
  RunCase(
    "cell_estop", [](FakeCriServer & server) {
      server.SetJointErrorBits(2);  // ESTOP_LOWV
      server.SetDisabledMask(0x3F);
    });
}

TEST_F(RebelFaultGroupDriveTest, KinematicsErrorAfterEnableStopsTheOtherArm)
{
  RunCase("cell_kin", [](FakeCriServer & server) {server.SetKinstate(99);});
}

TEST_F(RebelFaultGroupTest, DisabledDrivesBeforeTheFirstEnableDoNotTrigger)
{
  Arm left;
  Arm right;
  StartArm(left, "left", "cell_startup", /*disabledAtStart=*/ true);
  StartArm(right, "right", "cell_startup");
  ASSERT_TRUE(WaitFor([&]() {return right.rebel->FaultMonitorArmed();}, 3000ms));

  // The drives of the left arm stay disabled well past the settle time (Enable not effective).
  RunCycles(left, right, 120, 0.0, 0.0);
  EXPECT_EQ(left.rebel->DrivesDisabledMask(), 0x3Fu);
  EXPECT_FALSE(left.rebel->FaultMonitorArmed());
  EXPECT_FALSE(left.rebel->FaultGroupRootCause());
  EXPECT_FALSE(right.rebel->HardwareErrorLatched());
  EXPECT_FALSE(right.errored);
  EXPECT_FALSE(left.errored);

  // The operator's reset_and_enable then enables the drives (still no trigger while the sequence
  // is processed), and only after a clean settle time is the arm monitored.
  left.server->SetIgnoreEnable(false);
  std::string message;
  ASSERT_TRUE(left.rebel->ResetAndEnable(message)) << message;
  RunCycles(left, right, 10, 0.0, 0.0);
  EXPECT_FALSE(right.rebel->HardwareErrorLatched());
  ASSERT_TRUE(WaitFor([&]() {return left.rebel->FaultMonitorArmed();}, 3000ms));
  EXPECT_FALSE(right.rebel->HardwareErrorLatched());

  // Now a stop of the enabled left arm does count.
  left.server->SetDisabledMask(0x01);
  ASSERT_TRUE(WaitFor([&]() {return right.rebel->HardwareErrorLatched();}, 1000ms));
  EXPECT_TRUE(left.rebel->FaultGroupRootCause());
}

TEST_F(RebelFaultGroupTest, RecoveryIsRefusedUntilTheRootCauseMemberRecovered)
{
  Arm left;
  Arm right;
  StartArmedPair(left, right, "cell_recovery");
  RunCycles(left, right, 10, 0.0, 0.1);

  left.server->SetSendStatus(false);
  ASSERT_TRUE(
    WaitFor(
      [&]() {
        Cycle(left, right, 0.0, 0.1);
        return left.errored;
      }, 3000ms));
  ASSERT_TRUE(right.errored);
  EXPECT_TRUE(left.rebel->FaultGroupRootCause());
  EXPECT_FALSE(right.rebel->FaultGroupRootCause());

  left.server->SetSendStatus(true);
  std::this_thread::sleep_for(100ms);  // STATUS of the left arm is fresh again

  // Right (propagated only) may not resume while the left root cause is un-recovered, even
  // though its own STATUS is fresh. Nothing is re-sent to its robot.
  const std::size_t rightCommands = right.server->Commands(1).size();
  EXPECT_EQ(right.rebel->on_configure(rclcpp_lifecycle::State()), CallbackReturn::SUCCESS);
  EXPECT_EQ(right.rebel->on_activate(rclcpp_lifecycle::State()), CallbackReturn::FAILURE);
  EXPECT_TRUE(right.rebel->HardwareErrorLatched());
  EXPECT_EQ(right.rebel->read(now, period), return_type::ERROR);
  EXPECT_EQ(right.server->Commands(1).size(), rightCommands);

  // The root-cause member recovers first (fresh STATUS required: it is, so it succeeds).
  EXPECT_EQ(left.rebel->on_configure(rclcpp_lifecycle::State()), CallbackReturn::SUCCESS);
  EXPECT_EQ(left.rebel->on_activate(rclcpp_lifecycle::State()), CallbackReturn::SUCCESS);
  EXPECT_FALSE(left.rebel->HardwareErrorLatched());
  EXPECT_EQ(left.rebel->read(now, period), return_type::OK);
  // ... but it counts as healthy only once its drives were seen enabled and clean again.
  EXPECT_TRUE(left.rebel->FaultGroupRootCause());
  EXPECT_EQ(right.rebel->on_activate(rclcpp_lifecycle::State()), CallbackReturn::FAILURE);

  ASSERT_TRUE(WaitFor([&]() {return !left.rebel->FaultGroupRootCause();}, 3000ms));
  EXPECT_EQ(right.rebel->on_activate(rclcpp_lifecycle::State()), CallbackReturn::SUCCESS);
  EXPECT_FALSE(right.rebel->HardwareErrorLatched());
  ASSERT_TRUE(
    WaitFor(
      [&]() {
        return right.server->Commands(1).size() >= rightCommands + 6;
      }, 2000ms));
  EXPECT_EQ(right.rebel->read(now, period), return_type::OK);
  EXPECT_EQ(left.rebel->read(now, period), return_type::OK);

  // Both run again and nobody re-latches.
  left.errored = false;
  right.errored = false;
  RunCycles(left, right, 30, 0.0, 0.1);
  EXPECT_FALSE(left.errored);
  EXPECT_FALSE(right.errored);
  EXPECT_TRUE(AnyNonZeroJog(*right.server, Clock::now() - 100ms));
}

TEST_F(RebelFaultGroupTest, KinstateRootCauseClearsOnlyAfterContinuousCleanStatus)
{
  Arm left;
  Arm right;
  StartArmedPair(left, right, "cell_kin_clean");
  RunCycles(left, right, 10, 0.0, 0.1);

  // Strict bring-up trigger: any Kinstate other than NO_ERROR (here 99; JOINT_LIMIT_MIN/MAX
  // behave the same) stops the group, and latches the faulting arm too.
  left.server->SetKinstate(99);
  ASSERT_TRUE(
    WaitFor(
      [&]() {
        Cycle(left, right, 0.0, 0.1);
        return right.rebel->HardwareErrorLatched();
      }, 1000ms));
  Cycle(left, right, 0.0, 0.1);
  ASSERT_TRUE(right.errored);
  ASSERT_TRUE(left.errored);
  ASSERT_TRUE(left.rebel->HardwareErrorLatched());
  ASSERT_TRUE(left.rebel->FaultGroupRootCause());
  ASSERT_FALSE(left.rebel->FaultMonitorArmed());

  auto expectPeerRefused = [&]() {
      EXPECT_EQ(right.rebel->on_configure(rclcpp_lifecycle::State()), CallbackReturn::SUCCESS);
      EXPECT_EQ(right.rebel->on_activate(rclcpp_lifecycle::State()), CallbackReturn::FAILURE);
      EXPECT_TRUE(right.rebel->HardwareErrorLatched());
    };
  expectPeerRefused();

  // The fault goes away; clean STATUS arrives while the root-cause arm is still latched. The
  // root cause stays (recovery needs the deliberate inactive -> active path) and so does the
  // block on the peer.
  left.server->SetKinstate(0);
  std::this_thread::sleep_for(100ms);
  EXPECT_TRUE(left.rebel->FaultGroupRootCause());
  EXPECT_TRUE(left.rebel->HardwareErrorLatched());
  expectPeerRefused();

  // The root-cause arm recovers (fresh STATUS): unlatched, but not yet healthy for the group.
  EXPECT_EQ(left.rebel->on_configure(rclcpp_lifecycle::State()), CallbackReturn::SUCCESS);
  EXPECT_EQ(left.rebel->on_activate(rclcpp_lifecycle::State()), CallbackReturn::SUCCESS);
  const auto activatedAt = Clock::now();
  EXPECT_FALSE(left.rebel->HardwareErrorLatched());
  left.errored = false;
  ASSERT_FALSE(left.rebel->FaultMonitorArmed());
  // One (or a few) clean STATUS must not clear the root cause.
  std::this_thread::sleep_for(50ms);
  if (Clock::now() - activatedAt < 400ms) {
    EXPECT_TRUE(left.rebel->FaultGroupRootCause());
    EXPECT_FALSE(left.rebel->FaultMonitorArmed());
    expectPeerRefused();
  }

  // An intermittent non-clean STATUS restarts the continuous 500 ms. (The arm is not armed yet,
  // so this does not re-trigger; it only restarts the clean run.)
  ASSERT_FALSE(left.rebel->FaultMonitorArmed());
  left.server->SetKinstate(99);
  std::this_thread::sleep_for(100ms);
  left.server->SetKinstate(0);
  const auto cleanStart = Clock::now();
  std::this_thread::sleep_for(250ms);
  if (Clock::now() - cleanStart < 400ms) {
    EXPECT_TRUE(left.rebel->FaultGroupRootCause());
    EXPECT_FALSE(left.rebel->FaultMonitorArmed());
    expectPeerRefused();
  }

  // After >= 500 ms of continuous clean STATUS it clears (and not before).
  ASSERT_TRUE(WaitFor([&]() {return !left.rebel->FaultGroupRootCause();}, 5000ms));
  const auto clearedAfter = Clock::now() - cleanStart;
  EXPECT_GE(clearedAfter, 450ms);  // one STATUS period of slack for the first clean STATUS
  EXPECT_TRUE(left.rebel->FaultMonitorArmed());
  EXPECT_FALSE(left.rebel->HardwareErrorLatched());
  EXPECT_EQ(right.rebel->on_activate(rclcpp_lifecycle::State()), CallbackReturn::SUCCESS);
  EXPECT_FALSE(right.rebel->HardwareErrorLatched());
}

TEST_F(RebelFaultGroupTest, DriveDisabledRootCauseLatchesBothAndRecoversInOrder)
{
  Arm left;
  Arm right;
  StartArmedPair(left, right, "cell_drive_order");
  RunCycles(left, right, 10, 0.0, 0.1);

  left.server->SetDisabledMask(0x04);  // cleared again by the Enable of the recovery
  ASSERT_TRUE(
    WaitFor(
      [&]() {
        Cycle(left, right, 0.0, 0.1);
        return right.rebel->HardwareErrorLatched() && left.rebel->HardwareErrorLatched();
      }, 1000ms));
  EXPECT_TRUE(left.rebel->FaultGroupRootCause());
  EXPECT_FALSE(right.rebel->FaultGroupRootCause());

  // reset_and_enable alone is not the recovery path of a latched group member.
  std::string message;
  EXPECT_FALSE(left.rebel->ResetAndEnable(message));
  EXPECT_NE(message.find("latched"), std::string::npos) << message;

  // Order enforced: the peer is refused first, the root cause recovers, the peer follows once
  // the root cause was enabled and clean for the settle time.
  EXPECT_EQ(right.rebel->on_configure(rclcpp_lifecycle::State()), CallbackReturn::SUCCESS);
  EXPECT_EQ(right.rebel->on_activate(rclcpp_lifecycle::State()), CallbackReturn::FAILURE);
  EXPECT_EQ(left.rebel->on_configure(rclcpp_lifecycle::State()), CallbackReturn::SUCCESS);
  EXPECT_EQ(left.rebel->on_activate(rclcpp_lifecycle::State()), CallbackReturn::SUCCESS);
  EXPECT_FALSE(left.rebel->HardwareErrorLatched());
  EXPECT_EQ(right.rebel->on_activate(rclcpp_lifecycle::State()), CallbackReturn::FAILURE);
  ASSERT_TRUE(WaitFor([&]() {return !left.rebel->FaultGroupRootCause();}, 5000ms));
  EXPECT_EQ(right.rebel->on_activate(rclcpp_lifecycle::State()), CallbackReturn::SUCCESS);
  EXPECT_FALSE(right.rebel->HardwareErrorLatched());
}

TEST_F(RebelFaultGroupTest, DifferentGroupsDoNotInteract)
{
  Arm left;
  Arm right;
  StartArm(left, "left", "cell_a");
  StartArm(right, "right", "cell_b");
  ASSERT_TRUE(
    WaitFor(
      [&]() {return left.rebel->FaultMonitorArmed() && right.rebel->FaultMonitorArmed();},
      3000ms));

  EXPECT_EQ(left.rebel->on_error(rclcpp_lifecycle::State()), CallbackReturn::SUCCESS);
  left.server->SetDisabledMask(0x01);
  left.server->SetKinstate(99);
  std::this_thread::sleep_for(300ms);
  EXPECT_FALSE(right.rebel->HardwareErrorLatched());
  EXPECT_FALSE(right.rebel->FaultGroupRootCause());
  EXPECT_EQ(right.rebel->read(now, period), return_type::OK);
}

TEST_F(RebelFaultGroupTest, MemberLeavesTheGroupWhenStopped)
{
  Arm left;
  Arm right;
  StartArmedPair(left, right, "cell_leave");
  EXPECT_EQ(Igus::FaultGroupSize("cell_leave"), 2u);
  left.rebel->Stop();  // sends Disable: its drives report disabled, which must not count
  EXPECT_EQ(Igus::FaultGroupSize("cell_leave"), 1u);
  std::this_thread::sleep_for(200ms);
  EXPECT_FALSE(right.rebel->HardwareErrorLatched());
  EXPECT_EQ(right.rebel->read(now, period), return_type::OK);
  left.rebel.reset();
  right.rebel.reset();
  EXPECT_EQ(Igus::FaultGroupSize("cell_leave"), 0u);
}

TEST_F(RebelFaultGroupTest, DestroyingAMemberWhileAnotherFaultsIsSafe)
{
  for (int iteration = 0; iteration < 6; ++iteration) {
    Arm left;
    Arm right;
    const std::string group = "cell_destroy_" + std::to_string(iteration);
    StartArm(left, "left", group);
    StartArm(right, "right", group);
    ASSERT_NE(left.rebel, nullptr);
    ASSERT_NE(right.rebel, nullptr);
    std::atomic<bool> go{false};
    // Even iterations: the healthy arm is destroyed while the other faults; odd iterations: the
    // faulting arm is destroyed while it propagates.
    Arm & victim = iteration % 2 == 0 ? right : left;
    Arm & faulter = iteration % 2 == 0 ? left : right;
    std::thread destroyer(
      [&]() {
        while (!go) {}
        victim.rebel.reset();
      });
    go = true;
    faulter.rebel->on_error(rclcpp_lifecycle::State());
    destroyer.join();
    EXPECT_EQ(Igus::FaultGroupSize(group), 1u);
    EXPECT_TRUE(faulter.rebel->HardwareErrorLatched());
    EXPECT_EQ(faulter.rebel->read(now, period), return_type::ERROR);
  }
}

TEST(RebelFaultGroupInit, RejectsInvalidGroupName)
{
  auto info = MakeInfo();
  info.hardware_parameters["robot_ip"] = "127.0.0.1";
  info.hardware_parameters["robot_port"] = "1";
  info.hardware_parameters["fault_group"] = "bad group";
  Igus::Rebel rebel("10.255.255.1", 1);
  EXPECT_EQ(rebel.on_init(info), CallbackReturn::ERROR);
}

// The registry on its own (no robots): ordering, lifetime and concurrency.
TEST(FaultGroupRegistry, PropagatesToOtherMembersOfTheSameGroupOnly)
{
  std::atomic<int> a{0};
  std::atomic<int> b{0};
  std::atomic<int> c{0};
  std::atomic<int> d{0};
  const auto make = [](const std::string & group, const std::string & name, std::atomic<int> & n) {
      return std::make_shared<Igus::FaultGroupMember>(
        group, name, [&n](const std::string &, const std::string &) {++n;});
    };
  auto ma = make("reg_x", "a", a);
  auto mb = make("reg_x", "b", b);
  auto mc = make("reg_x", "c", c);
  auto md = make("reg_y", "d", d);
  for (const auto & member : {ma, mb, mc, md}) {
    ASSERT_TRUE(Igus::JoinFaultGroup(member));
  }
  EXPECT_FALSE(Igus::JoinFaultGroup(ma));  // already joined
  EXPECT_EQ(Igus::FaultGroupSize("reg_x"), 3u);

  EXPECT_EQ(Igus::ReportRootFault(ma, "boom"), 2u);
  EXPECT_EQ(a.load(), 0);
  EXPECT_EQ(b.load(), 1);
  EXPECT_EQ(c.load(), 1);
  EXPECT_EQ(d.load(), 0);
  EXPECT_TRUE(ma->HasRootFault());
  EXPECT_EQ(ma->RootReason(), "boom");

  Igus::LeaveFaultGroup(mb);
  EXPECT_EQ(Igus::ReportRootFault(ma, "again"), 1u);
  EXPECT_EQ(b.load(), 1);  // detached: never called again
  EXPECT_EQ(c.load(), 2);
  for (const auto & member : {ma, mb, mc, md}) {
    Igus::LeaveFaultGroup(member);
  }
  EXPECT_EQ(Igus::FaultGroupSize("reg_x"), 0u);
  EXPECT_EQ(Igus::ReportRootFault(ma, "alone"), 0u);

  // A member that already faulted before it joined stops the existing members on joining.
  auto mx = make("reg_z", "x", a);
  auto my = make("reg_z", "y", b);
  ASSERT_TRUE(Igus::JoinFaultGroup(mx));
  my->MarkRootFault("early");
  const int before = a.load();
  ASSERT_TRUE(Igus::JoinFaultGroup(my));
  EXPECT_EQ(a.load(), before + 1);
  Igus::LeaveFaultGroup(mx);
  Igus::LeaveFaultGroup(my);
}

TEST(FaultGroupRegistry, RecoveryOrderFollowsTheOrderOfRootCauseFaults)
{
  const auto noop = [](const std::string &, const std::string &) {};
  auto a = std::make_shared<Igus::FaultGroupMember>("order", "a", noop);
  auto b = std::make_shared<Igus::FaultGroupMember>("order", "b", noop);
  auto c = std::make_shared<Igus::FaultGroupMember>("order", "c", noop);
  for (const auto & member : {a, b, c}) {
    ASSERT_TRUE(Igus::JoinFaultGroup(member));
  }
  Igus::BlockingFault blocking;
  EXPECT_FALSE(Igus::FindBlockingRootFault(a, blocking));

  // a and b both faulted on their own (a first); c only by propagation.
  a->MarkRootFault("a failed");
  b->MarkRootFault("b failed");
  a->MarkRootFault("ignored: first root cause wins");
  EXPECT_EQ(a->RootReason(), "a failed");

  EXPECT_FALSE(Igus::FindBlockingRootFault(a, blocking));  // the root cause recovers first
  ASSERT_TRUE(Igus::FindBlockingRootFault(b, blocking));   // no mutual blocking: a before b
  EXPECT_EQ(blocking.member, "a");
  ASSERT_TRUE(Igus::FindBlockingRootFault(c, blocking));
  EXPECT_EQ(blocking.member, "a");

  EXPECT_TRUE(a->ClearRootFault());
  EXPECT_FALSE(a->ClearRootFault());
  EXPECT_FALSE(Igus::FindBlockingRootFault(b, blocking));
  ASSERT_TRUE(Igus::FindBlockingRootFault(c, blocking));
  EXPECT_EQ(blocking.member, "b");
  EXPECT_TRUE(b->ClearRootFault());
  EXPECT_FALSE(Igus::FindBlockingRootFault(c, blocking));

  // A member that left no longer blocks anybody.
  a->MarkRootFault("again");
  ASSERT_TRUE(Igus::FindBlockingRootFault(c, blocking));
  Igus::LeaveFaultGroup(a);
  EXPECT_FALSE(Igus::FindBlockingRootFault(c, blocking));
  Igus::LeaveFaultGroup(b);
  Igus::LeaveFaultGroup(c);
}

TEST(FaultGroupRegistry, LeaveWaitsForARunningCallback)
{
  std::atomic<bool> entered{false};
  std::atomic<bool> finished{false};
  auto reporter = std::make_shared<Igus::FaultGroupMember>(
    "wait", "reporter", [](const std::string &, const std::string &) {});
  auto victim = std::make_shared<Igus::FaultGroupMember>(
    "wait", "victim", [&](const std::string &, const std::string &) {
      entered = true;
      std::this_thread::sleep_for(150ms);
      finished = true;
    });
  ASSERT_TRUE(Igus::JoinFaultGroup(reporter));
  ASSERT_TRUE(Igus::JoinFaultGroup(victim));
  std::thread thread([&]() {Igus::ReportRootFault(reporter, "x");});
  ASSERT_TRUE(WaitFor([&]() {return entered.load();}, 1000ms));
  Igus::LeaveFaultGroup(victim);
  // The owner may be destroyed right after Leave returns: the callback must be over.
  EXPECT_TRUE(finished.load());
  thread.join();
  Igus::LeaveFaultGroup(reporter);
}

TEST(FaultGroupRegistry, ConcurrentFaultsAndOwnerDestructionDoNotTouchFreedOwners)
{
  struct Owner
  {
    std::atomic<std::uint32_t> magic{0xC0FFEEu};
    std::atomic<int> * badCalls;
    Igus::FaultGroupMemberPtr member;
    explicit Owner(std::atomic<int> * bad)
    : badCalls(bad)
    {
      member = std::make_shared<Igus::FaultGroupMember>(
        "stress", "owner", [this](const std::string &, const std::string &) {
          if (magic.load() != 0xC0FFEEu) {
            ++*badCalls;
          }
        });
    }
    // Same discipline as Igus::Rebel: leave first, then let the members die.
    ~Owner()
    {
      Igus::LeaveFaultGroup(member);
      magic = 0;
    }
  };
  std::atomic<int> bad{0};
  std::atomic<bool> run{true};
  auto root = std::make_shared<Igus::FaultGroupMember>(
    "stress", "root", [](const std::string &, const std::string &) {});
  ASSERT_TRUE(Igus::JoinFaultGroup(root));

  std::vector<std::thread> reporters;
  for (int i = 0; i < 3; ++i) {
    reporters.emplace_back(
      [&]() {
        while (run) {
          Igus::ReportRootFault(root, "stress");
          root->ClearRootFault();
          Igus::BlockingFault blocking;
          Igus::FindBlockingRootFault(root, blocking);
        }
      });
  }
  const auto end = Clock::now() + 600ms;
  int created = 0;
  while (Clock::now() < end) {
    auto owner = std::make_unique<Owner>(&bad);
    Igus::JoinFaultGroup(owner->member);
    std::this_thread::yield();
    owner.reset();
    ++created;
  }
  run = false;
  for (auto & thread : reporters) {
    thread.join();
  }
  EXPECT_GT(created, 50);
  EXPECT_EQ(bad.load(), 0);
  Igus::LeaveFaultGroup(root);
  EXPECT_EQ(Igus::FaultGroupSize("stress"), 0u);
}

// ===========================================================================================
// Robot-side joint move (move_joint action, CRI "CMD Move Joint")
// ===========================================================================================
namespace
{
using MoveJoint = igus_rebel_msgs::action::MoveJoint;
using MoveResult = MoveJoint::Result;
using MoveGoalHandle = rclcpp_action::ClientGoalHandle<MoveJoint>;

// Action client on its own node, spun on its own thread.
class MoveClient
{
public:
  explicit MoveClient(const std::string & actionName = "/igus_rebel/move_joint")
  {
    node_ = std::make_shared<rclcpp::Node>(
      "move_test_client_" + std::to_string(counter_++));
    client_ = rclcpp_action::create_client<MoveJoint>(node_, actionName);
    executor_.add_node(node_);
    thread_ = std::thread([this]() {executor_.spin(); finished_ = true;});
  }
  ~MoveClient()
  {
    // cancel() before the thread entered spin() is lost (spin() then never returns): repeat it.
    while (!finished_) {
      executor_.cancel();
      std::this_thread::sleep_for(2ms);
    }
    if (thread_.joinable()) {
      thread_.join();
    }
  }

  bool WaitForServer() {return client_->wait_for_action_server(5s);}

  // Returns nullptr if the goal was rejected (or not answered).
  MoveGoalHandle::SharedPtr Send(const MoveJoint::Goal & goal)
  {
    feedbackCount_ = 0;
    rclcpp_action::Client<MoveJoint>::SendGoalOptions options;
    options.feedback_callback =
      [this](MoveGoalHandle::SharedPtr, const std::shared_ptr<const MoveJoint::Feedback> feedback) {
        std::lock_guard<std::mutex> lock(mutex_);
        ++feedbackCount_;
        lastFeedback_ = *feedback;
      };
    auto future = client_->async_send_goal(goal, options);
    if (future.wait_for(5s) != std::future_status::ready) {
      return nullptr;
    }
    return future.get();
  }

  // Waits for the result; code UNKNOWN + empty result if it did not arrive in time.
  rclcpp_action::ClientGoalHandle<MoveJoint>::WrappedResult Result(
    const MoveGoalHandle::SharedPtr & handle, std::chrono::milliseconds timeout = 15000ms)
  {
    auto future = client_->async_get_result(handle);
    rclcpp_action::ClientGoalHandle<MoveJoint>::WrappedResult wrapped;
    wrapped.code = rclcpp_action::ResultCode::UNKNOWN;
    wrapped.result = std::make_shared<MoveResult>();
    wrapped.result->error_code = 255;
    if (future.wait_for(timeout) == std::future_status::ready) {
      wrapped = future.get();
    }
    return wrapped;
  }

  void Cancel(const MoveGoalHandle::SharedPtr & handle) {client_->async_cancel_goal(handle);}

  std::size_t FeedbackCount()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return feedbackCount_;
  }
  MoveJoint::Feedback LastFeedback()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return lastFeedback_;
  }

private:
  static inline std::atomic<int> counter_{0};
  std::atomic<bool> finished_{false};
  rclcpp::Node::SharedPtr node_;
  rclcpp_action::Client<MoveJoint>::SharedPtr client_;
  rclcpp::executors::SingleThreadedExecutor executor_;
  std::thread thread_;
  std::mutex mutex_;
  std::size_t feedbackCount_ = 0;
  MoveJoint::Feedback lastFeedback_;
};

MoveJoint::Goal MakeMoveGoal(
  const std::array<std::string, 6> & names, const std::array<float, 6> & currentDeg,
  const std::array<double, 6> & deltaDeg, double velocity = 25.0, double acceleration = 40.0)
{
  MoveJoint::Goal goal;
  for (std::size_t i = 0; i < 6; ++i) {
    goal.joint_names.push_back(names[i]);
    goal.start_positions.push_back(currentDeg[i] * Igus::Safety::kDegToRad);
    goal.goal_positions.push_back((currentDeg[i] + deltaDeg[i]) * Igus::Safety::kDegToRad);
  }
  goal.velocity_percent = velocity;
  goal.acceleration_percent = acceleration;
  goal.start_tolerance = 0.05;
  goal.path_tolerance = 0.03;
  goal.goal_tolerance = 0.02;
  goal.timeout_s = 0.0;
  return goal;
}

std::size_t MoveStopCount(FakeCriServer & server, Clock::time_point after = {})
{
  return server.CountCommands("Move Stop", after);
}
}  // namespace

class RebelMoveTest : public ::testing::Test
{
protected:
  // Starts the driver against `server`, activates it (as the controller manager does) and
  // connects the action client.
  void Begin(
    const std::map<std::string, std::string> & hardwareParameters = {}, bool activate = true)
  {
    server.SetOverride(100.0f);
    rebel = std::make_unique<Igus::Rebel>("127.0.0.1", server.port());
    auto info = MakeInfo("rebel_arm_controller");
    for (const auto & entry : hardwareParameters) {
      info.hardware_parameters[entry.first] = entry.second;
    }
    ASSERT_EQ(rebel->on_init(info), CallbackReturn::SUCCESS);
    states = rebel->export_state_interfaces();
    commands = rebel->export_command_interfaces();
    if (activate) {
      ASSERT_EQ(rebel->on_activate(rclcpp_lifecycle::State()), CallbackReturn::SUCCESS);
    }
    ASSERT_TRUE(WaitFor([&]() {return server.Commands().size() >= 6;}, 3000ms));
    // The override STATUS must have reached the driver.
    std::this_thread::sleep_for(100ms);
    client = std::make_unique<MoveClient>();
    ASSERT_TRUE(client->WaitForServer());
  }

  void TearDown() override
  {
    loop.reset();
    client.reset();
    rebel.reset();
  }

  void StartLoop(double velocity = 0.0)
  {
    loop = std::make_unique<ControllerLoop>(*rebel, commands, velocity);
  }

  MoveJoint::Goal Goal(
    const std::array<double, 6> & deltaDeg, double velocity = 25.0, double acceleration = 40.0)
  {
    return MakeMoveGoal(rebel->JointNames(), server.CurrentDeg(), deltaDeg, velocity, acceleration);
  }

  // Sends the goal and expects it to be accepted.
  MoveGoalHandle::SharedPtr SendAccepted(const MoveJoint::Goal & goal)
  {
    auto handle = client->Send(goal);
    EXPECT_NE(handle, nullptr);
    return handle;
  }

  bool WaitMoving(double minDegTravelled = 1.0)
  {
    const auto start = server.CurrentDeg();
    return WaitFor(
      [&]() {
        const auto now = server.CurrentDeg();
        double travelled = 0.0;
        for (int i = 0; i < 6; ++i) {
          travelled = std::max(travelled, static_cast<double>(std::fabs(now[i] - start[i])));
        }
        return travelled >= minDegTravelled;
      }, 5000ms);
  }

  void ExpectNoMoveJointSent()
  {
    EXPECT_TRUE(server.MoveCommands().empty());
  }

  FakeCriServer server;
  std::unique_ptr<Igus::Rebel> rebel;
  std::vector<hardware_interface::StateInterface> states;
  std::vector<hardware_interface::CommandInterface> commands;
  std::unique_ptr<ControllerLoop> loop;
  std::unique_ptr<MoveClient> client;
  rclcpp::Time now{0, 0, RCL_STEADY_TIME};
  rclcpp::Duration period{0, 10000000};
};

TEST_F(RebelMoveTest, HappyPathSendsOneMoveJointAndKeepsEveryAliveJogZero)
{
  Begin();
  server.SetMoveSpeedDegS(60.0);
  // A stale controller command (what an active JTC would hold) must not reach the robot.
  StartLoop(0.2);
  ASSERT_TRUE(WaitFor([&]() {return AnyNonZeroJog(server, Clock::now() - 60ms);}, 2000ms));

  const auto startDeg = server.CurrentDeg();
  const auto goal = Goal({8.0, 0.0, 0.0, -5.0, 0.0, 6.0});
  const auto handle = SendAccepted(goal);
  ASSERT_NE(handle, nullptr);
  const auto result = client->Result(handle);

  EXPECT_EQ(result.code, rclcpp_action::ResultCode::SUCCEEDED);
  EXPECT_EQ(result.result->error_code, MoveResult::SUCCESS) << result.result->message;
  EXPECT_EQ(result.result->end_reason, "PLAN");
  EXPECT_FALSE(result.result->stop_sent);
  ASSERT_EQ(result.result->final_positions.size(), 6u);
  for (std::size_t i = 0; i < 6; ++i) {
    EXPECT_NEAR(result.result->final_positions[i], goal.goal_positions[i], 0.002);
  }
  EXPECT_EQ(rebel->MoveStateValue(), 0);

  // Exactly one Move Joint in the documented format: degrees, vel/acc.
  const auto moves = server.MoveCommands();
  ASSERT_EQ(moves.size(), 1u);
  EXPECT_NEAR(moves[0].targetDeg[0], startDeg[0] + 8.0, 1e-2);
  EXPECT_NEAR(moves[0].targetDeg[3], startDeg[3] - 5.0, 1e-2);
  EXPECT_NEAR(moves[0].targetDeg[5], startDeg[5] + 6.0, 1e-2);
  EXPECT_EQ(moves[0].velocity, 25.0);
  EXPECT_EQ(moves[0].acceleration, 40.0);

  // Every ALIVEJOG from the moment the command was sent is zero although write() kept
  // delivering a non-zero velocity command the whole time.
  std::size_t frames = 0;
  EXPECT_TRUE(AllJogsZero(server, moves[0].stamp, &frames));
  EXPECT_GE(frames, 5u);
  EXPECT_EQ(MoveStopCount(server), 1u);  // only the one at connect
  EXPECT_GT(client->FeedbackCount(), 0u);
  EXPECT_FALSE(rebel->HardwareErrorLatched());
  EXPECT_EQ(loop->errors.load(), 0);

  // After the move the jog path is live again.
  std::this_thread::sleep_for(100ms);
  EXPECT_TRUE(AnyNonZeroJog(server, Clock::now() - 60ms));
}

TEST_F(RebelMoveTest, AcceptsCmdAckBeforeMoveAck)
{
  Begin();
  server.SetMoveSpeedDegS(60.0);
  server.SetCmdAckFirst(true);
  StartLoop();
  const auto handle = SendAccepted(Goal({6.0, 0.0, 0.0, 0.0, 0.0, 0.0}));
  ASSERT_NE(handle, nullptr);
  EXPECT_EQ(client->Result(handle).result->error_code, MoveResult::SUCCESS);
}

TEST_F(RebelMoveTest, ParametersLowerTheCapsAndTheHardCapsCannotBeExceeded)
{
  Begin({{"move_joint_max_velocity_percent", "150"}, {"move_joint_max_acceleration_percent", "10"}});
  server.SetMoveSpeedDegS(80.0);
  StartLoop();
  // 150 % is clamped to the 100 % hard cap: 101 is refused (never clamped), 100 is fine.
  auto handle = SendAccepted(Goal({4.0, 0, 0, 0, 0, 0}, 101.0, 10.0));
  ASSERT_NE(handle, nullptr);
  EXPECT_EQ(client->Result(handle).result->error_code, MoveResult::REJECTED);
  // Acceleration cap lowered to 10 %.
  handle = SendAccepted(Goal({4.0, 0, 0, 0, 0, 0}, 20.0, 11.0));
  ASSERT_NE(handle, nullptr);
  EXPECT_EQ(client->Result(handle).result->error_code, MoveResult::REJECTED);
  ExpectNoMoveJointSent();
  handle = SendAccepted(Goal({4.0, 0, 0, 0, 0, 0}, 100.0, 10.0));
  ASSERT_NE(handle, nullptr);
  EXPECT_EQ(client->Result(handle).result->error_code, MoveResult::SUCCESS);
  ASSERT_EQ(server.MoveCommands().size(), 1u);
  EXPECT_EQ(server.MoveCommands()[0].velocity, 100.0);
}

TEST_F(RebelMoveTest, DefaultCapsRejectAboveTwentyFivePercentVelocity)
{
  Begin();
  StartLoop();
  auto handle = SendAccepted(Goal({4.0, 0, 0, 0, 0, 0}, 26.0, 40.0));
  ASSERT_NE(handle, nullptr);
  EXPECT_EQ(client->Result(handle).result->error_code, MoveResult::REJECTED);
  handle = SendAccepted(Goal({4.0, 0, 0, 0, 0, 0}, 25.0, 41.0));
  ASSERT_NE(handle, nullptr);
  EXPECT_EQ(client->Result(handle).result->error_code, MoveResult::REJECTED);
  ExpectNoMoveJointSent();
}

TEST_F(RebelMoveTest, InvalidGoalsAreRefusedWithoutSendingAnything)
{
  Begin();
  StartLoop();
  const auto expectCode = [&](MoveJoint::Goal goal, std::uint8_t code, const char * what) {
      const auto handle = client->Send(goal);
      ASSERT_NE(handle, nullptr) << what;
      const auto result = client->Result(handle);
      EXPECT_EQ(result.result->error_code, code) << what << ": " << result.result->message;
      EXPECT_FALSE(result.result->stop_sent) << what;
    };
  auto base = Goal({4.0, 0, 0, 0, 0, 0});

  auto wrongNames = base;
  std::swap(wrongNames.joint_names[0], wrongNames.joint_names[1]);
  expectCode(wrongNames, MoveResult::REJECTED, "joint names swapped");

  auto prefixed = base;
  for (auto & name : prefixed.joint_names) {
    name = "arm_left_" + name;
  }
  expectCode(prefixed, MoveResult::REJECTED, "joint names of another arm");

  auto nan = base;
  nan.goal_positions[2] = std::numeric_limits<double>::quiet_NaN();
  expectCode(nan, MoveResult::REJECTED, "NaN goal");

  auto beyondLimit = base;
  beyondLimit.goal_positions[4] = 2.0;  // joint 5 limit is +-1.64 rad
  expectCode(beyondLimit, MoveResult::REJECTED, "beyond joint limit");

  auto sizes = base;
  sizes.goal_positions.pop_back();
  expectCode(sizes, MoveResult::REJECTED, "short goal vector");

  auto startOff = base;
  startOff.start_positions[0] += 0.3;
  expectCode(startOff, MoveResult::START_DEVIATION, "start differs from the measured joints");

  auto zeroTolerance = base;
  zeroTolerance.path_tolerance = 0.0;
  expectCode(zeroTolerance, MoveResult::REJECTED, "zero path tolerance");

  auto huge = base;
  huge.goal_tolerance = 1.0;
  expectCode(huge, MoveResult::REJECTED, "oversized goal tolerance");

  auto nothing = Goal({0.0, 0, 0, 0, 0, 0});
  expectCode(nothing, MoveResult::REJECTED, "zero-length move");

  ExpectNoMoveJointSent();
  EXPECT_EQ(MoveStopCount(server), 1u);
}

TEST_F(RebelMoveTest, RefusedWhileOverrideDrivesOrKinematicsAreNotOk)
{
  Begin();
  StartLoop();
  const auto goal = Goal({4.0, 0, 0, 0, 0, 0});

  server.SetOverride(40.0f);
  std::this_thread::sleep_for(150ms);
  auto handle = SendAccepted(goal);
  ASSERT_NE(handle, nullptr);
  EXPECT_EQ(client->Result(handle).result->error_code, MoveResult::REJECTED);
  server.SetOverride(100.0f);
  std::this_thread::sleep_for(150ms);

  server.SetKinstate(99);
  std::this_thread::sleep_for(150ms);
  handle = SendAccepted(goal);
  ASSERT_NE(handle, nullptr);
  const auto kin = client->Result(handle);
  // kinstate != NO_ERROR trips the move check (the driver is not in a fault group here).
  EXPECT_EQ(kin.result->error_code, MoveResult::FAULT) << kin.result->message;
  server.SetKinstate(0);
  std::this_thread::sleep_for(150ms);

  ExpectNoMoveJointSent();
}

TEST_F(RebelMoveTest, RefusedWhenDrivesAreDisabledOrStatusIsStaleOrNotActivated)
{
  Begin();
  StartLoop();
  const auto goal = Goal({4.0, 0, 0, 0, 0, 0});
  server.SetDisabledMask(0x3F);
  server.SetIgnoreEnable(true);
  std::this_thread::sleep_for(150ms);
  auto handle = SendAccepted(goal);
  ASSERT_NE(handle, nullptr);
  EXPECT_EQ(client->Result(handle).result->error_code, MoveResult::REJECTED);
  server.SetDisabledMask(0);
  std::this_thread::sleep_for(150ms);
  ExpectNoMoveJointSent();

  // Deactivated (what the controller manager does before on_deactivate): refused.
  EXPECT_EQ(rebel->on_deactivate(rclcpp_lifecycle::State()), CallbackReturn::SUCCESS);
  handle = SendAccepted(goal);
  ASSERT_NE(handle, nullptr);
  EXPECT_EQ(client->Result(handle).result->error_code, MoveResult::REJECTED);
  ExpectNoMoveJointSent();
}

TEST_F(RebelMoveTest, RefusedWhenNeverActivated)
{
  Begin({}, false);
  const auto handle = SendAccepted(Goal({4.0, 0, 0, 0, 0, 0}));
  ASSERT_NE(handle, nullptr);
  EXPECT_EQ(client->Result(handle).result->error_code, MoveResult::REJECTED);
  ExpectNoMoveJointSent();
}

TEST_F(RebelMoveTest, RefusedWhileLatched)
{
  Begin();
  StartLoop();
  const auto goal = Goal({4.0, 0, 0, 0, 0, 0});
  rebel->on_error(rclcpp_lifecycle::State());
  const auto handle = SendAccepted(goal);
  ASSERT_NE(handle, nullptr);
  EXPECT_EQ(client->Result(handle).result->error_code, MoveResult::LATCHED);
  ExpectNoMoveJointSent();
  // No stray stop is sent for a latch while idle: idle behaviour is unchanged.
  EXPECT_EQ(MoveStopCount(server), 1u);
}

TEST_F(RebelMoveTest, RefusedWhileAControllerClaimsTheVelocityInterfaces)
{
  Begin();
  server.SetMoveSpeedDegS(60.0);
  StartLoop();
  const auto joint = [&](int i) {return rebel->JointNames()[i] + "/velocity";};
  const std::vector<std::string> all = {joint(0), joint(1), joint(2), joint(3), joint(4), joint(5)};

  // A controller (the JTC) claims them.
  EXPECT_EQ(rebel->prepare_command_mode_switch(all, {}), return_type::OK);
  EXPECT_EQ(rebel->perform_command_mode_switch(all, {}), return_type::OK);
  EXPECT_EQ(rebel->VelocityClaimMask(), 0x3Fu);
  auto handle = SendAccepted(Goal({4.0, 0, 0, 0, 0, 0}));
  ASSERT_NE(handle, nullptr);
  const auto refused = client->Result(handle);
  EXPECT_EQ(refused.result->error_code, MoveResult::CONTROLLER_ACTIVE) << refused.result->message;
  ExpectNoMoveJointSent();

  // One interface still claimed is enough to refuse.
  EXPECT_EQ(
    rebel->perform_command_mode_switch({}, {joint(0), joint(1), joint(2), joint(3), joint(4)}),
    return_type::OK);
  EXPECT_EQ(rebel->VelocityClaimMask(), 0x20u);
  handle = SendAccepted(Goal({4.0, 0, 0, 0, 0, 0}));
  ASSERT_NE(handle, nullptr);
  EXPECT_EQ(client->Result(handle).result->error_code, MoveResult::CONTROLLER_ACTIVE);
  ExpectNoMoveJointSent();

  // Released: accepted.
  EXPECT_EQ(rebel->perform_command_mode_switch({}, {joint(5)}), return_type::OK);
  EXPECT_EQ(rebel->VelocityClaimMask(), 0u);
  handle = SendAccepted(Goal({4.0, 0, 0, 0, 0, 0}));
  ASSERT_NE(handle, nullptr);
  EXPECT_EQ(client->Result(handle).result->error_code, MoveResult::SUCCESS);
}

TEST_F(RebelMoveTest, ClaimPreparedButNotYetPerformedAlsoBlocksAGoal)
{
  Begin();
  StartLoop();
  const std::vector<std::string> one = {rebel->JointNames()[2] + "/velocity"};
  EXPECT_EQ(rebel->prepare_command_mode_switch(one, {}), return_type::OK);
  const auto handle = SendAccepted(Goal({4.0, 0, 0, 0, 0, 0}));
  ASSERT_NE(handle, nullptr);
  EXPECT_EQ(client->Result(handle).result->error_code, MoveResult::CONTROLLER_ACTIVE);
  ExpectNoMoveJointSent();
}

TEST_F(RebelMoveTest, PrepareRejectsAClaimMidMoveButNotOtherInterfaces)
{
  Begin();
  server.SetMoveSpeedDegS(10.0);
  StartLoop();
  const auto handle = SendAccepted(Goal({20.0, 0, 0, 0, 0, 0}));
  ASSERT_NE(handle, nullptr);
  ASSERT_TRUE(WaitMoving());
  ASSERT_NE(rebel->MoveStateValue(), 0);

  const std::vector<std::string> ours = {rebel->JointNames()[0] + "/velocity"};
  EXPECT_EQ(rebel->prepare_command_mode_switch(ours, {}), return_type::ERROR);
  EXPECT_EQ(rebel->VelocityClaimMask(), 0u);
  // Releasing and interfaces of other hardware are not affected.
  EXPECT_EQ(
    rebel->prepare_command_mode_switch({"other_joint1/velocity", "joint1/position"}, ours),
    return_type::OK);
  EXPECT_EQ(rebel->prepare_command_mode_switch({}, {}), return_type::OK);

  client->Cancel(handle);
  const auto result = client->Result(handle);
  EXPECT_EQ(result.result->error_code, MoveResult::CANCELED);
  // Idle again: the claim is allowed.
  EXPECT_EQ(rebel->prepare_command_mode_switch(ours, {}), return_type::OK);
}

TEST_F(RebelMoveTest, SecondGoalIsRejectedWhileOneIsActive)
{
  Begin();
  server.SetMoveSpeedDegS(10.0);
  StartLoop();
  const auto first = SendAccepted(Goal({20.0, 0, 0, 0, 0, 0}));
  ASSERT_NE(first, nullptr);
  ASSERT_TRUE(WaitMoving());
  EXPECT_EQ(client->Send(Goal({0.0, 5.0, 0, 0, 0, 0})), nullptr);
  EXPECT_EQ(server.MoveCommands().size(), 1u);
  client->Cancel(first);
  EXPECT_EQ(client->Result(first).result->error_code, MoveResult::CANCELED);
}

TEST_F(RebelMoveTest, CancelSendsMoveStopPromptlyAndReportsIt)
{
  Begin();
  server.SetMoveSpeedDegS(10.0);
  StartLoop();
  const auto handle = SendAccepted(Goal({30.0, 0, 0, 0, 0, 0}));
  ASSERT_NE(handle, nullptr);
  ASSERT_TRUE(WaitMoving(1.0));

  const auto cancelAt = Clock::now();
  client->Cancel(handle);
  const auto result = client->Result(handle);
  EXPECT_EQ(result.code, rclcpp_action::ResultCode::CANCELED);
  EXPECT_EQ(result.result->error_code, MoveResult::CANCELED);
  EXPECT_TRUE(result.result->stop_sent);
  EXPECT_TRUE(result.result->stop_confirmed);
  EXPECT_EQ(result.result->end_reason, "USER");

  const auto stopAt = server.FirstCommandAfter("Move Stop", cancelAt);
  ASSERT_NE(stopAt, Clock::time_point{});
  EXPECT_LT(stopAt - cancelAt, 150ms);
  EXPECT_FALSE(server.MoveActive());
  EXPECT_LT(server.CurrentDeg()[0], server.baseJointsDeg[0] + 29.0f);
  EXPECT_EQ(rebel->MoveStateValue(), 0);
  // Cancel is an operator action, not a hardware fault.
  EXPECT_FALSE(rebel->HardwareErrorLatched());
  // Zero jog all the time.
  EXPECT_TRUE(AllJogsZero(server, server.MoveCommands()[0].stamp));
}

TEST_F(RebelMoveTest, CancelIsNotBlockedBehindResetAndEnable)
{
  Begin();
  server.SetMoveSpeedDegS(10.0);
  StartLoop();
  const auto handle = SendAccepted(Goal({30.0, 0, 0, 0, 0, 0}));
  ASSERT_NE(handle, nullptr);
  ASSERT_TRUE(WaitMoving(1.0));
  // Occupy the service thread: reset_and_enable via ROS (refused quickly during a move, but the
  // important part is that the action keeps working while services are served).
  auto probe = std::make_shared<rclcpp::Node>("reset_probe");
  auto reset = probe->create_client<std_srvs::srv::Trigger>("/igus_rebel/reset_and_enable");
  ASSERT_TRUE(reset->wait_for_service(3s));
  auto future = reset->async_send_request(std::make_shared<std_srvs::srv::Trigger::Request>());
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(probe);
  ASSERT_EQ(executor.spin_until_future_complete(future, 3s), rclcpp::FutureReturnCode::SUCCESS);
  EXPECT_FALSE(future.get()->success);  // refused: a move is active
  EXPECT_EQ(MoveStopCount(server), 1u);  // and nothing was re-sent

  std::string message;
  EXPECT_FALSE(rebel->ResetAndEnable(message));
  EXPECT_NE(message.find("joint move"), std::string::npos) << message;

  client->Cancel(handle);
  EXPECT_EQ(client->Result(handle).result->error_code, MoveResult::CANCELED);
}

TEST_F(RebelMoveTest, CmdErrorFailsTheGoalAndSendsMoveStop)
{
  Begin();
  server.SetMoveCmdError(true);
  StartLoop();
  const auto handle = SendAccepted(Goal({10.0, 0, 0, 0, 0, 0}));
  ASSERT_NE(handle, nullptr);
  const auto result = client->Result(handle);
  EXPECT_EQ(result.code, rclcpp_action::ResultCode::ABORTED);
  EXPECT_EQ(result.result->error_code, MoveResult::CMD_ERROR) << result.result->message;
  EXPECT_TRUE(result.result->stop_sent);
  EXPECT_GE(MoveStopCount(server), 2u);
  EXPECT_EQ(rebel->MoveStateValue(), 0);
}

TEST_F(RebelMoveTest, MissingMoveAckStopsAfterHalfASecond)
{
  Begin();
  server.SetSuppressMoveAck(true);
  server.SetMoveSpeedDegS(10.0);
  StartLoop();
  const auto sent = Clock::now();
  const auto handle = SendAccepted(Goal({30.0, 0, 0, 0, 0, 0}));
  ASSERT_NE(handle, nullptr);
  const auto result = client->Result(handle);
  EXPECT_EQ(result.result->error_code, MoveResult::ACK_TIMEOUT) << result.result->message;
  EXPECT_TRUE(result.result->stop_sent);
  EXPECT_TRUE(result.result->stop_confirmed);  // by standstill: the robot's END has no ACK to match
  const auto stopAt = server.FirstCommandAfter("Move Stop", sent);
  ASSERT_NE(stopAt, Clock::time_point{});
  EXPECT_GE(stopAt - sent, 450ms);
  EXPECT_LT(stopAt - sent, 900ms);
  EXPECT_FALSE(server.MoveActive());
}

TEST_F(RebelMoveTest, MotionTimeoutStopsTheMove)
{
  Begin();
  server.SetMoveStall(true);  // the robot acknowledges but never moves and never ends
  StartLoop();
  auto goal = Goal({10.0, 0, 0, 0, 0, 0});
  goal.timeout_s = 0.8;
  const auto sent = Clock::now();
  const auto handle = SendAccepted(goal);
  ASSERT_NE(handle, nullptr);
  const auto result = client->Result(handle);
  EXPECT_EQ(result.result->error_code, MoveResult::MOTION_TIMEOUT) << result.result->message;
  EXPECT_TRUE(result.result->stop_sent);
  EXPECT_TRUE(result.result->stop_confirmed);
  const auto stopAt = server.FirstCommandAfter("Move Stop", sent);
  ASSERT_NE(stopAt, Clock::time_point{});
  EXPECT_GE(stopAt - sent, 750ms);
  EXPECT_LT(stopAt - sent, 1300ms);
}

TEST_F(RebelMoveTest, DefaultTimeoutComesFromTheDurationModel)
{
  Begin();
  server.SetMoveStall(true);
  StartLoop();
  // timeout_s = 0: 2 x model + 1 s.
  const double expected =
    2.0 * Igus::Move::ModelDurationS(3.0 * Igus::Safety::kDegToRad, 25.0, 40.0) + 1.0;
  const auto sent = Clock::now();
  const auto handle = SendAccepted(Goal({3.0, 0, 0, 0, 0, 0}));
  ASSERT_NE(handle, nullptr);
  const auto result = client->Result(handle);
  EXPECT_EQ(result.result->error_code, MoveResult::MOTION_TIMEOUT);
  const auto stopAt = server.FirstCommandAfter("Move Stop", sent);
  ASSERT_NE(stopAt, Clock::time_point{});
  const double seen = std::chrono::duration<double>(stopAt - sent).count();
  EXPECT_GT(seen, expected - 0.1);
  EXPECT_LT(seen, expected + 0.6);
}

TEST_F(RebelMoveTest, PathDeviationStopsTheMove)
{
  Begin();
  server.SetMoveSpeedDegS(20.0);
  server.SetMoveDeviationDeg(5.0);  // joint 5 drifts 5 deg off the line; tolerance is 0.03 rad
  StartLoop();
  const auto handle = SendAccepted(Goal({30.0, 0, 0, 0, 0, 0}));
  ASSERT_NE(handle, nullptr);
  const auto result = client->Result(handle);
  EXPECT_EQ(result.result->error_code, MoveResult::PATH_DEVIATION) << result.result->message;
  EXPECT_TRUE(result.result->stop_sent);
  EXPECT_TRUE(result.result->stop_confirmed);
  EXPECT_GE(MoveStopCount(server), 2u);
  EXPECT_FALSE(server.MoveActive());
  EXPECT_LT(server.CurrentDeg()[0], server.baseJointsDeg[0] + 29.0f);
}

TEST_F(RebelMoveTest, RobotEndingTheMoveItselfIsReportedAndNeedsNoStop)
{
  Begin();
  server.SetMoveSpeedDegS(10.0);
  server.SetMoveAbortAfterS(0.4);  // e.g. the operator pressed stop on the pendant
  StartLoop();
  const auto handle = SendAccepted(Goal({30.0, 0, 0, 0, 0, 0}));
  ASSERT_NE(handle, nullptr);
  const auto result = client->Result(handle);
  EXPECT_EQ(result.result->error_code, MoveResult::ROBOT_ENDED) << result.result->message;
  EXPECT_EQ(result.result->end_reason, "USER");
  EXPECT_FALSE(result.result->stop_sent);
  EXPECT_EQ(MoveStopCount(server), 1u);
  EXPECT_EQ(rebel->MoveStateValue(), 0);
}

TEST_F(RebelMoveTest, EndingShortOfTheGoalIsAGoalDeviation)
{
  Begin();
  server.SetMoveSpeedDegS(60.0);
  server.SetMoveEndShortDeg(1.0);  // reports PLAN, but joint 1 is 1 deg (0.017 rad) short
  StartLoop();
  auto goal = Goal({10.0, 0, 0, 0, 0, 0});
  goal.goal_tolerance = 0.005;
  const auto handle = SendAccepted(goal);
  ASSERT_NE(handle, nullptr);
  const auto result = client->Result(handle);
  EXPECT_EQ(result.result->error_code, MoveResult::GOAL_DEVIATION) << result.result->message;
  EXPECT_EQ(result.result->end_reason, "PLAN");
  EXPECT_FALSE(result.result->stop_sent);
}

TEST_F(RebelMoveTest, HoldTierDuringAMoveStopsAndLatchesWithoutAutoResume)
{
  Begin();
  server.SetMoveSpeedDegS(10.0);
  StartLoop(0.2);
  const auto handle = SendAccepted(Goal({30.0, 0, 0, 0, 0, 0}));
  ASSERT_NE(handle, nullptr);
  ASSERT_TRUE(WaitMoving(1.0));
  const auto movesAt = server.MoveCommands()[0].stamp;

  // STATUS stops for 700 ms: a robot-owned move is aborted as soon as STATUS is stale (200 ms),
  // before the jog hold tier (400 ms); STATUS then comes back.
  const auto gapStart = Clock::now();
  server.SetSendStatus(false);
  ASSERT_TRUE(
    WaitFor([&]() {return MoveStopCount(server, gapStart) >= 1;}, 1500ms));
  const auto stopAt = server.FirstCommandAfter("Move Stop", gapStart);
  EXPECT_GE(stopAt - gapStart, 150ms);  // not before STATUS was stale
  EXPECT_LT(stopAt - gapStart, 380ms);  // and well before the 400 ms jog hold tier
  std::this_thread::sleep_for(
    std::max<std::chrono::milliseconds>(
      0ms, 700ms - std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - gapStart)));
  server.SetSendStatus(true);

  const auto result = client->Result(handle);
  EXPECT_NE(result.result->error_code, MoveResult::SUCCESS);
  EXPECT_TRUE(result.result->stop_sent);
  EXPECT_TRUE(
    result.result->error_code == MoveResult::STATUS_GAP ||
    result.result->error_code == MoveResult::FAULT) << result.result->message;
  EXPECT_TRUE(WaitFor([&]() {return rebel->HardwareErrorLatched();}, 2000ms));
  EXPECT_TRUE(WaitFor([&]() {return loop->errors.load() > 0;}, 2000ms));  // read() reports ERROR
  EXPECT_FALSE(server.MoveActive());

  // Never auto-resumed: with the (stale) velocity command still non-zero, every ALIVEJOG since
  // the move was sent is zero.
  std::this_thread::sleep_for(200ms);
  EXPECT_TRUE(AllJogsZero(server, movesAt));
}

TEST_F(RebelMoveTest, ShortStatusGapDuringAMoveDoesNotStopIt)
{
  Begin();
  server.SetMoveSpeedDegS(20.0);
  StartLoop();
  const auto handle = SendAccepted(Goal({12.0, 0, 0, 0, 0, 0}));
  ASSERT_NE(handle, nullptr);
  ASSERT_TRUE(WaitMoving(1.0));
  // 120 ms without STATUS is below the 200 ms stale limit: the robot owns the move, it continues.
  server.SetSendStatus(false);
  std::this_thread::sleep_for(120ms);
  server.SetSendStatus(true);
  const auto result = client->Result(handle);
  EXPECT_EQ(result.result->error_code, MoveResult::SUCCESS) << result.result->message;
  EXPECT_FALSE(rebel->HardwareErrorLatched());
  EXPECT_EQ(MoveStopCount(server), 1u);
}

TEST_F(RebelMoveTest, ReconnectDuringAMoveSendsMoveStopBeforeSetActive)
{
  Begin();
  server.SetMoveSpeedDegS(10.0);
  StartLoop();
  const auto handle = SendAccepted(Goal({30.0, 0, 0, 0, 0, 0}));
  ASSERT_NE(handle, nullptr);
  ASSERT_TRUE(WaitMoving(1.0));
  const float before = server.CurrentDeg()[0];

  server.DropClient();
  ASSERT_TRUE(WaitFor([&]() {return server.Commands(2).size() >= 6;}, 3000ms));
  const auto second = server.Commands(2);
  ASSERT_GE(second.size(), 6u);
  EXPECT_EQ(second[0], "Move Stop");
  EXPECT_EQ(second[1], "SetActive true");
  // The robot kept moving while nobody was connected, and stopped on the reconnect.
  EXPECT_GT(server.CurrentDeg()[0], before);
  EXPECT_TRUE(WaitFor([&]() {return !server.MoveActive();}, 1000ms));

  const auto result = client->Result(handle);
  EXPECT_NE(result.result->error_code, MoveResult::SUCCESS) << result.result->message;
  EXPECT_TRUE(WaitFor([&]() {return rebel->HardwareErrorLatched();}, 2000ms));
}

TEST_F(RebelMoveTest, UnconfirmedStopRepeatsAndLatches)
{
  Begin();
  server.SetMoveSpeedDegS(10.0);
  server.SetIgnoreMoveStop(true);  // a robot that does not react
  StartLoop();
  const auto handle = SendAccepted(Goal({30.0, 0, 0, 0, 0, 0}));
  ASSERT_NE(handle, nullptr);
  ASSERT_TRUE(WaitMoving(1.0));
  const auto cancelAt = Clock::now();
  client->Cancel(handle);
  const auto result = client->Result(handle);
  EXPECT_FALSE(result.result->stop_confirmed);
  EXPECT_TRUE(result.result->stop_sent);
  EXPECT_NE(result.result->message.find("NOT confirmed"), std::string::npos)
    << result.result->message;
  EXPECT_TRUE(rebel->HardwareErrorLatched());
  // First stop plus a repeat every 100 ms during the 1 s confirmation window.
  EXPECT_GE(MoveStopCount(server, cancelAt), 6u);
  EXPECT_LE(MoveStopCount(server, cancelAt), 14u);
  // The goal has ended, but the robot is still moving: the stop request stays pending and the
  // alive thread keeps re-sending it (the robot ACKed every one with CMDACK; a successful write
  // or an ACK is not a confirmation).
  const auto afterGoal = Clock::now();
  std::this_thread::sleep_for(500ms);
  EXPECT_GE(MoveStopCount(server, afterGoal), 3u);
  EXPECT_LE(MoveStopCount(server, afterGoal), 7u);
  EXPECT_TRUE(server.MoveActive());
  EXPECT_TRUE(rebel->HardwareErrorLatched());

  server.SetIgnoreMoveStop(false);
  // The next repeat stops the robot ...
  EXPECT_TRUE(WaitFor([&]() {return !server.MoveActive();}, 500ms));
  // ... and once STATUS shows standstill the request is dropped: no more Move Stop.
  std::this_thread::sleep_for(700ms);
  const auto settled = Clock::now();
  const std::size_t before = MoveStopCount(server, settled);
  std::this_thread::sleep_for(500ms);
  EXPECT_EQ(MoveStopCount(server, settled), before);
  EXPECT_LE(before, 0u);
  EXPECT_TRUE(rebel->HardwareErrorLatched());  // still latched: recovery is inactive -> active
}

TEST_F(RebelMoveTest, StatusGapOfTwoHundredFiftyMillisecondsAbortsAndLatches)
{
  Begin();
  server.SetMoveSpeedDegS(10.0);
  StartLoop();
  const auto handle = SendAccepted(Goal({30.0, 0, 0, 0, 0, 0}));
  ASSERT_NE(handle, nullptr);
  ASSERT_TRUE(WaitMoving(1.0));
  const auto gapStart = Clock::now();
  server.SetSendStatus(false);
  ASSERT_TRUE(WaitFor([&]() {return MoveStopCount(server, gapStart) >= 1;}, 1000ms));
  const auto stopAt = server.FirstCommandAfter("Move Stop", gapStart);
  EXPECT_GE(stopAt - gapStart, 150ms);
  EXPECT_LT(stopAt - gapStart, 380ms);  // the old threshold was 400 ms
  std::this_thread::sleep_for(100ms);
  server.SetSendStatus(true);
  const auto result = client->Result(handle);
  EXPECT_EQ(result.result->error_code, MoveResult::STATUS_GAP) << result.result->message;
  EXPECT_TRUE(rebel->HardwareErrorLatched());
}

TEST_F(RebelMoveTest, EverySupervisorAbortLeavesTheDriverLatched)
{
  Begin();
  server.SetMoveSpeedDegS(20.0);
  server.SetMoveDeviationDeg(5.0);
  StartLoop();
  auto handle = SendAccepted(Goal({30.0, 0, 0, 0, 0, 0}));
  ASSERT_NE(handle, nullptr);
  auto result = client->Result(handle);
  EXPECT_EQ(result.result->error_code, MoveResult::PATH_DEVIATION) << result.result->message;
  EXPECT_TRUE(rebel->HardwareErrorLatched());
  // Not silently ready for the next move.
  handle = SendAccepted(Goal({-5.0, 0, 0, 0, 0, 0}));
  ASSERT_NE(handle, nullptr);
  EXPECT_EQ(client->Result(handle).result->error_code, MoveResult::LATCHED);
  ASSERT_TRUE(WaitFor([&]() {return loop->errors.load() > 0;}, 2000ms));  // read() reports ERROR
}

TEST_F(RebelMoveTest, MotionTimeoutLeavesTheDriverLatchedAndRecoversByLifecycleReset)
{
  Begin();
  server.SetMoveStall(true);
  auto goal = Goal({10.0, 0, 0, 0, 0, 0});
  goal.timeout_s = 0.8;
  const auto handle = SendAccepted(goal);
  ASSERT_NE(handle, nullptr);
  const auto result = client->Result(handle);
  EXPECT_EQ(result.result->error_code, MoveResult::MOTION_TIMEOUT) << result.result->message;
  EXPECT_TRUE(result.result->stop_confirmed);
  EXPECT_TRUE(rebel->HardwareErrorLatched());
  // inactive -> active clears it like every other latch.
  server.SetMoveStall(false);
  EXPECT_EQ(rebel->on_deactivate(rclcpp_lifecycle::State()), CallbackReturn::SUCCESS);
  EXPECT_EQ(rebel->on_activate(rclcpp_lifecycle::State()), CallbackReturn::SUCCESS);
  EXPECT_FALSE(rebel->HardwareErrorLatched());
}

TEST_F(RebelMoveTest, ReleasingAVelocityInterfaceZeroesItsCommand)
{
  Begin();
  const auto joint = [&](int i) {return rebel->JointNames()[i] + "/velocity";};
  commands[0].set_value(0.3);
  commands[1].set_value(0.2);
  EXPECT_EQ(rebel->prepare_command_mode_switch({}, {joint(0)}), return_type::OK);
  EXPECT_EQ(rebel->perform_command_mode_switch({}, {joint(0)}), return_type::OK);
  EXPECT_EQ(commands[0].get_value(), 0.0);
  EXPECT_EQ(commands[1].get_value(), 0.2);  // not released: untouched
}

// An unconfirmed Move Stop (the fake robot ignores it) that latched the driver. The goal is long
// enough to outlast the follow-up checks.
#define MAKE_UNCONFIRMED_STOP() \
  do { \
    server.SetMoveSpeedDegS(10.0); \
    server.SetIgnoreMoveStop(true); \
    const auto handle = SendAccepted(Goal({50.0, 0, 0, 0, 0, 0})); \
    ASSERT_NE(handle, nullptr); \
    ASSERT_TRUE(WaitMoving(1.0)); \
    client->Cancel(handle); \
    const auto result = client->Result(handle); \
    ASSERT_FALSE(result.result->stop_confirmed); \
    ASSERT_TRUE(rebel->HardwareErrorLatched()); \
  } while (0)

std::size_t MoveStopsOnConnection(FakeCriServer & server, std::size_t connection)
{
  std::size_t count = 0;
  for (const auto & command : server.Commands(connection)) {
    count += command == "Move Stop" ? 1 : 0;
  }
  return count;
}

TEST_F(RebelMoveTest, PendingStopSurvivesAReconnectUntilStandstill)
{
  Begin();
  MAKE_UNCONFIRMED_STOP();

  server.DropClient();
  ASSERT_TRUE(WaitFor([&]() {return server.Commands(2).size() >= 6;}, 3000ms));
  // The reconnect sends Move Stop first (still ignored by the fake robot) and the request stays
  // pending: the ALIVEJOG thread keeps re-sending on the new connection.
  EXPECT_EQ(server.Commands(2)[0], "Move Stop");
  EXPECT_EQ(server.Commands(2)[1], "SetActive true");  // not interleaved with the start-up
  ASSERT_TRUE(WaitFor([&]() {return MoveStopsOnConnection(server, 2) >= 4;}, 2000ms));
  EXPECT_TRUE(server.MoveActive());
  EXPECT_TRUE(rebel->HardwareErrorLatched());

  server.SetIgnoreMoveStop(false);  // the robot now obeys: the next repeat stops it
  EXPECT_TRUE(WaitFor([&]() {return !server.MoveActive();}, 500ms));
  // Standstill is confirmed by STATUS: the repeats end.
  std::this_thread::sleep_for(800ms);
  const std::size_t settled = MoveStopsOnConnection(server, 2);
  std::this_thread::sleep_for(500ms);
  EXPECT_EQ(MoveStopsOnConnection(server, 2), settled);
}

TEST_F(RebelMoveTest, PendingStopSurvivesLifecycleRecoveryAndBlocksANewMove)
{
  Begin();
  MAKE_UNCONFIRMED_STOP();

  // Operator recovery while the robot is still moving: inactive -> active clears the latch ...
  EXPECT_EQ(rebel->on_deactivate(rclcpp_lifecycle::State()), CallbackReturn::SUCCESS);
  EXPECT_EQ(rebel->on_activate(rclcpp_lifecycle::State()), CallbackReturn::SUCCESS);
  EXPECT_FALSE(rebel->HardwareErrorLatched());
  const auto at = Clock::now();
  // ... the controller keeps commanding velocity (the recovered JTC) ...
  StartLoop(0.2);
  // ... but not the stop request: it is still re-sent, and jog stays zero on the wire while it
  // is pending (the robot is still running the old move).
  ASSERT_TRUE(WaitFor([&]() {return MoveStopCount(server, at) >= 3;}, 2000ms));
  EXPECT_TRUE(server.MoveActive());
  std::size_t frames = 0;
  EXPECT_TRUE(AllJogsZero(server, at, &frames));
  EXPECT_GE(frames, 10u);

  // A new move is refused until the previous stop is confirmed.
  server.SetMoveSpeedDegS(10.0);
  const std::size_t movesBefore = server.MoveCommands().size();
  auto refused = SendAccepted(Goal({-5.0, 0, 0, 0, 0, 0}));
  ASSERT_NE(refused, nullptr);
  const auto refusedResult = client->Result(refused);
  EXPECT_EQ(refusedResult.result->error_code, MoveResult::REJECTED);
  EXPECT_NE(refusedResult.result->message.find("Move Stop is not confirmed"), std::string::npos)
    << refusedResult.result->message;
  EXPECT_EQ(server.MoveCommands().size(), movesBefore);

  server.SetIgnoreMoveStop(false);
  ASSERT_TRUE(WaitFor([&]() {return !server.MoveActive();}, 500ms));
  // Standstill confirmed -> the request is dropped -> jog resumes ...
  EXPECT_TRUE(WaitFor([&]() {return AnyNonZeroJog(server, Clock::now() - 60ms);}, 2000ms));
  std::this_thread::sleep_for(300ms);
  // ... and a move is accepted again.
  auto accepted = SendAccepted(Goal({-4.0, 0, 0, 0, 0, 0}));
  ASSERT_NE(accepted, nullptr);
  server.SetMoveSpeedDegS(60.0);
  EXPECT_EQ(client->Result(accepted).result->error_code, MoveResult::SUCCESS);
}

TEST_F(RebelMoveTest, ControllerClaimAfterTheReservationExpiredAbortsAndLatches)
{
  Begin();
  server.SetMoveSpeedDegS(10.0);
  StartLoop();
  const std::vector<std::string> all = {
    rebel->JointNames()[0] + "/velocity", rebel->JointNames()[1] + "/velocity"};
  // prepare ... the 2 s reservation expires before perform is called ...
  EXPECT_EQ(rebel->prepare_command_mode_switch(all, {}), return_type::OK);
  std::this_thread::sleep_for(2200ms);
  // ... a goal is accepted (nothing claims the joints yet) ...
  const auto handle = SendAccepted(Goal({40.0, 0, 0, 0, 0, 0}));
  ASSERT_NE(handle, nullptr);
  ASSERT_TRUE(WaitMoving(1.0));
  ASSERT_EQ(rebel->VelocityClaimMask(), 0u);
  // ... and then the (late) perform arrives; Humble activates the controller regardless.
  const auto at = Clock::now();
  EXPECT_EQ(rebel->perform_command_mode_switch(all, {}), return_type::OK);
  EXPECT_EQ(rebel->VelocityClaimMask(), 0x3u);  // recorded truthfully

  const auto result = client->Result(handle);
  EXPECT_EQ(result.result->error_code, MoveResult::CONTROLLER_ACTIVE) << result.result->message;
  EXPECT_TRUE(result.result->stop_sent);
  EXPECT_TRUE(result.result->stop_confirmed);
  EXPECT_TRUE(rebel->HardwareErrorLatched());
  EXPECT_GE(MoveStopCount(server, at), 1u);
  EXPECT_LT(server.FirstCommandAfter("Move Stop", at) - at, 100ms);
  EXPECT_FALSE(server.MoveActive());
}

// The arm settles by ~2 mrad between the validation snapshot and the send. That is not motion of
// the commanded move: standstill must still not confirm a stop before the robot started.
TEST_F(RebelMoveTest, SettleBetweenValidationAndSendIsNotMotion)
{
  Begin();
  StartLoop();
  server.SetMoveStartLatencyMs(600);
  server.SetStopIgnoredBeforeStart(true);
  server.SetMoveSpeedDegS(10.0);
  const auto settle = [this]() {
      server.NudgeJointDeg(1, 0.115f);  // ~2 mrad
      std::this_thread::sleep_for(100ms);  // the new STATUS is in before the send
    };
  rebel->SetBeforeMoveSendHookForTest(settle);
  const auto handle = SendAccepted(Goal({30.0, 0, 0, 0, 0, 0}));
  ASSERT_NE(handle, nullptr);
  ASSERT_TRUE(WaitFor([&]() {return server.MoveCommands().size() == 1;}, 2000ms));
  const auto cancelAt = Clock::now();
  client->Cancel(handle);
  const auto result = client->Result(handle);
  const auto took = Clock::now() - cancelAt;
  EXPECT_EQ(result.result->error_code, MoveResult::CANCELED) << result.result->message;
  EXPECT_GT(took, 500ms);  // the settle must not count as motion (~300 ms confirmation)
  EXPECT_TRUE(result.result->stop_confirmed);
  EXPECT_EQ(result.result->end_reason, "USER");
  EXPECT_FALSE(server.MoveActive());
  rebel->SetBeforeMoveSendHookForTest(nullptr);
}

// A stop that never saw motion or an END (the robot ignores it, starts late) has one way out:
// every drive disabled (e-stop). Here inside the stopping phase of the goal.
TEST_F(RebelMoveTest, DrivesDisabledConfirmAStopThatNeverSawMotion)
{
  Begin();
  StartLoop();
  server.SetMoveStartLatencyMs(5000);
  server.SetIgnoreMoveStop(true);
  const auto handle = SendAccepted(Goal({30.0, 0, 0, 0, 0, 0}));
  ASSERT_NE(handle, nullptr);
  ASSERT_TRUE(WaitFor([&]() {return server.MoveCommands().size() == 1;}, 2000ms));
  client->Cancel(handle);
  std::this_thread::sleep_for(100ms);
  server.SetIgnoreEnable(true);
  server.SetDisabledMask(0x3F);  // e-stop: all six drives report 'Motor not enabled'
  const auto result = client->Result(handle);
  EXPECT_EQ(result.result->error_code, MoveResult::CANCELED) << result.result->message;
  EXPECT_TRUE(result.result->stop_confirmed);  // by the drives, well before the 1 s limit
  EXPECT_FALSE(rebel->HardwareErrorLatched());
}

// Only ALL six drives count: one disabled drive is no e-stop.
TEST_F(RebelMoveTest, OneDisabledDriveDoesNotConfirmAStop)
{
  Begin();
  StartLoop();
  server.SetMoveStartLatencyMs(5000);
  server.SetIgnoreMoveStop(true);
  const auto handle = SendAccepted(Goal({30.0, 0, 0, 0, 0, 0}));
  ASSERT_NE(handle, nullptr);
  ASSERT_TRUE(WaitFor([&]() {return server.MoveCommands().size() == 1;}, 2000ms));
  client->Cancel(handle);
  std::this_thread::sleep_for(100ms);
  server.SetIgnoreEnable(true);
  server.SetDisabledMask(0x01);
  const auto result = client->Result(handle);
  EXPECT_FALSE(result.result->stop_confirmed) << result.result->message;
  EXPECT_NE(result.result->message.find("Recovery: e-stop"), std::string::npos);
  EXPECT_TRUE(rebel->HardwareErrorLatched());
}

// The documented recovery of a never-confirmed stop: e-stop -> confirmed -> inactive -> active ->
// jog flows again after the re-enable.
TEST_F(RebelMoveTest, NeverMovedPendingStopIsReleasedByEstopAndRecoveryResumesJog)
{
  Begin();
  server.SetMoveStartLatencyMs(8000);
  server.SetIgnoreMoveStop(true);
  const auto handle = SendAccepted(Goal({30.0, 0, 0, 0, 0, 0}));
  ASSERT_NE(handle, nullptr);
  ASSERT_TRUE(WaitFor([&]() {return server.MoveCommands().size() == 1;}, 2000ms));
  client->Cancel(handle);
  const auto result = client->Result(handle);
  EXPECT_FALSE(result.result->stop_confirmed);
  ASSERT_TRUE(rebel->HardwareErrorLatched());

  // Nothing but the e-stop releases it: neither time nor recovery does.
  std::this_thread::sleep_for(700ms);
  EXPECT_EQ(rebel->on_deactivate(rclcpp_lifecycle::State()), CallbackReturn::SUCCESS);
  EXPECT_EQ(rebel->on_activate(rclcpp_lifecycle::State()), CallbackReturn::SUCCESS);
  StartLoop(0.2);
  const auto pendingFrom = Clock::now();
  std::this_thread::sleep_for(500ms);
  EXPECT_TRUE(AllJogsZero(server, pendingFrom));  // still pending: jog held
  loop.reset();

  server.SetIgnoreEnable(true);
  server.SetDisabledMask(0x3F);  // e-stop
  const auto estopAt = Clock::now();
  ASSERT_TRUE(
    WaitFor(
      [&]() {
        const auto before = MoveStopCount(server, estopAt);
        std::this_thread::sleep_for(450ms);
        return MoveStopCount(server, estopAt) == before;  // the repeats ended: confirmed
      }, 3000ms));
  // Recovery: the re-enable clears the e-stop on the fake robot.
  server.SetIgnoreEnable(false);
  EXPECT_EQ(rebel->on_deactivate(rclcpp_lifecycle::State()), CallbackReturn::SUCCESS);
  EXPECT_EQ(rebel->on_activate(rclcpp_lifecycle::State()), CallbackReturn::SUCCESS);
  EXPECT_FALSE(rebel->HardwareErrorLatched());
  StartLoop(0.2);
  EXPECT_TRUE(WaitFor([&]() {return AnyNonZeroJog(server, Clock::now() - 60ms);}, 2000ms));
}

TEST_F(RebelMoveTest, ClaimWhileStoppingLatchesEvenWhenTheStopIsConfirmed)
{
  Begin();
  server.SetMoveSpeedDegS(10.0);
  server.SetIgnoreMoveStop(true);  // keeps the goal in the stopping phase for a while
  StartLoop();
  const std::vector<std::string> all = {
    rebel->JointNames()[0] + "/velocity", rebel->JointNames()[1] + "/velocity"};
  const auto handle = SendAccepted(Goal({40.0, 0, 0, 0, 0, 0}));
  ASSERT_NE(handle, nullptr);
  ASSERT_TRUE(WaitMoving(1.0));
  client->Cancel(handle);
  ASSERT_TRUE(WaitFor([&]() {return rebel->MoveStateValue() == 3;}, 500ms));  // stopping
  // The late perform arrives while stopping (the abort checks no longer run) ...
  EXPECT_EQ(rebel->perform_command_mode_switch(all, {}), return_type::OK);
  EXPECT_FALSE(rebel->HardwareErrorLatched());  // not yet: nothing has latched in this phase
  // ... then the robot obeys and the stop is confirmed well inside the 1 s limit.
  server.SetIgnoreMoveStop(false);
  const auto result = client->Result(handle);
  EXPECT_EQ(result.result->error_code, MoveResult::CANCELED) << result.result->message;
  EXPECT_TRUE(result.result->stop_confirmed);  // not the unconfirmed-stop latch
  EXPECT_TRUE(rebel->HardwareErrorLatched());
  EXPECT_FALSE(server.MoveActive());
}

TEST_F(RebelMoveTest, SendFailureOfMoveJointStopsAndLatches)
{
  Begin();
  StartLoop();
  rebel->FailNextMoveJointSendForTest();
  const auto handle = SendAccepted(Goal({10.0, 0, 0, 0, 0, 0}));
  ASSERT_NE(handle, nullptr);
  const auto result = client->Result(handle);
  EXPECT_EQ(result.result->error_code, MoveResult::CMD_ERROR) << result.result->message;
  EXPECT_TRUE(result.result->stop_sent);
  EXPECT_TRUE(rebel->HardwareErrorLatched());
  EXPECT_TRUE(server.MoveCommands().empty());
}

// The ALIVEJOG thread passed its check of a pending stop and is paused before the send. Meanwhile
// the goal finishes (here: the robot reaches its target by itself, MOVETOEXECEND confirms), a new
// goal is accepted and its Move Joint is sent. The old stop must not be sent behind it.
TEST_F(RebelMoveTest, OldStopRequestIsNotSentBehindTheNextMoveJoint)
{
  Begin();
  StartLoop();
  server.SetMoveSpeedDegS(20.0);
  const auto first = SendAccepted(Goal({6.0, 0, 0, 0, 0, 0}));
  ASSERT_NE(first, nullptr);
  ASSERT_TRUE(WaitMoving(1.0));

  std::promise<void> entered;
  std::promise<void> release;
  auto released = release.get_future().share();
  std::atomic<bool> once{true};
  const auto pauseOnce = [&]() {
      if (once.exchange(false)) {
        entered.set_value();
        released.wait();
      }
    };
  rebel->SetBeforeStopSendHookForTest(pauseOnce);
  client->Cancel(first);
  ASSERT_EQ(entered.get_future().wait_for(2s), std::future_status::ready);  // paused pre-send
  // The goal ends while the stop is withheld: the robot arrives (END) -> confirmed -> cleared.
  const auto firstResult = client->Result(first);
  EXPECT_TRUE(firstResult.result->stop_confirmed) << firstResult.result->message;
  EXPECT_EQ(firstResult.result->end_reason, "PLAN");
  EXPECT_EQ(MoveStopCount(server), 1u);  // only the one at connect: the paused send never went out

  server.SetMoveSpeedDegS(60.0);
  const auto second = SendAccepted(Goal({6.0, 0, 0, 0, 0, 0}));
  ASSERT_NE(second, nullptr);
  ASSERT_TRUE(WaitFor([&]() {return server.MoveCommands().size() == 2;}, 2000ms));
  release.set_value();  // the paused thread now resumes with its stale request
  const auto secondResult = client->Result(second);
  // It re-checked under the send lock: no Move Stop behind the new Move Joint.
  EXPECT_EQ(secondResult.result->error_code, MoveResult::SUCCESS) << secondResult.result->message;
  EXPECT_EQ(MoveStopCount(server), 1u);
  rebel->SetBeforeStopSendHookForTest(nullptr);
}

// A slow send (here: 700 ms inside the send lock) must not eat the 0.5 s MOVETOEXECACK window.
TEST_F(RebelMoveTest, SlowMoveJointSendDoesNotCauseAnAckTimeout)
{
  Begin();
  StartLoop();
  server.SetMoveSpeedDegS(60.0);
  const auto slow = []() {std::this_thread::sleep_for(700ms);};
  rebel->SetBeforeMoveSendHookForTest(slow);
  const auto handle = SendAccepted(Goal({6.0, 0, 0, 0, 0, 0}));
  ASSERT_NE(handle, nullptr);
  const auto result = client->Result(handle);
  EXPECT_EQ(result.result->error_code, MoveResult::SUCCESS) << result.result->message;
  EXPECT_FALSE(result.result->stop_sent);
  EXPECT_FALSE(rebel->HardwareErrorLatched());
  rebel->SetBeforeMoveSendHookForTest(nullptr);
}

// Cancel right after the send, before the robot started. The robot only acts on a stop while it
// moves (the stop is acknowledged and dropped), and starts the move 600 ms later. Standstill at
// the start is NOT a confirmation: the stop is confirmed only after the robot moved and a repeat
// of the stop stopped it (MOVETOEXECEND).
TEST_F(RebelMoveTest, StopBeforeTheRobotStartedIsNotConfirmedByStandstill)
{
  Begin();
  StartLoop();
  server.SetMoveStartLatencyMs(600);
  server.SetStopIgnoredBeforeStart(true);
  server.SetMoveSpeedDegS(10.0);
  const auto handle = SendAccepted(Goal({30.0, 0, 0, 0, 0, 0}));
  ASSERT_NE(handle, nullptr);
  ASSERT_TRUE(WaitFor([&]() {return server.MoveCommands().size() == 1;}, 2000ms));
  const auto cancelAt = Clock::now();
  client->Cancel(handle);
  const auto result = client->Result(handle);
  const auto took = Clock::now() - cancelAt;
  EXPECT_EQ(result.result->error_code, MoveResult::CANCELED) << result.result->message;
  EXPECT_GT(took, 500ms);  // standstill alone would have "confirmed" after ~300 ms
  EXPECT_TRUE(result.result->stop_confirmed);
  EXPECT_EQ(result.result->end_reason, "USER");
  EXPECT_FALSE(server.MoveActive());
  EXPECT_LT(server.CurrentDeg()[0], server.baseJointsDeg[0] + 3.0f);
  EXPECT_FALSE(rebel->HardwareErrorLatched());
}

// Same, but the robot never acts on the stop: unconfirmed -> latched, the request stays pending
// (jog held, new moves refused) and keeps being sent, also after the robot starts late; it ends
// once the robot obeys and stands still after having moved.
TEST_F(RebelMoveTest, IgnoredStopOfARobotThatStartsLateStaysPending)
{
  Begin();
  StartLoop(0.2);
  server.SetMoveStartLatencyMs(800);
  server.SetIgnoreMoveStop(true);
  server.SetMoveSpeedDegS(10.0);
  const auto handle = SendAccepted(Goal({40.0, 0, 0, 0, 0, 0}));
  ASSERT_NE(handle, nullptr);
  ASSERT_TRUE(WaitFor([&]() {return server.MoveCommands().size() == 1;}, 2000ms));
  const auto cancelAt = Clock::now();
  client->Cancel(handle);
  const auto result = client->Result(handle);
  EXPECT_FALSE(result.result->stop_confirmed) << result.result->message;
  EXPECT_NE(result.result->message.find("NOT confirmed"), std::string::npos);
  EXPECT_TRUE(rebel->HardwareErrorLatched());
  // The robot starts moving after the goal ended; the stop is still being re-sent.
  ASSERT_TRUE(
    WaitFor(
      [&]() {
        return server.CurrentDeg()[0] > server.baseJointsDeg[0] + 0.5f;
      }, 3000ms));
  const auto lateAt = Clock::now();
  std::this_thread::sleep_for(500ms);
  EXPECT_GE(MoveStopCount(server, lateAt), 3u);
  EXPECT_TRUE(server.MoveActive());
  EXPECT_TRUE(AllJogsZero(server, cancelAt));

  server.SetIgnoreMoveStop(false);
  EXPECT_TRUE(WaitFor([&]() {return !server.MoveActive();}, 500ms));
  std::this_thread::sleep_for(800ms);
  const auto settled = Clock::now();
  const std::size_t before = MoveStopCount(server, settled);
  std::this_thread::sleep_for(400ms);
  EXPECT_EQ(MoveStopCount(server, settled), before);  // confirmed: no more repeats
}

TEST_F(RebelMoveTest, ReleaseOnlyAndUnrelatedSwitchesDuringAMoveDoNotAbortIt)
{
  Begin();
  server.SetMoveSpeedDegS(40.0);
  StartLoop();
  const auto handle = SendAccepted(Goal({20.0, 0, 0, 0, 0, 0}));
  ASSERT_NE(handle, nullptr);
  ASSERT_TRUE(WaitMoving(1.0));
  const std::vector<std::string> ours = {rebel->JointNames()[0] + "/velocity"};
  EXPECT_EQ(rebel->perform_command_mode_switch({}, ours), return_type::OK);  // release only
  EXPECT_EQ(
    rebel->perform_command_mode_switch({"other_joint1/velocity", "joint1/position"}, {}),
    return_type::OK);  // other hardware
  const auto result = client->Result(handle);
  EXPECT_EQ(result.result->error_code, MoveResult::SUCCESS) << result.result->message;
  EXPECT_FALSE(rebel->HardwareErrorLatched());
}

TEST_F(RebelMoveTest, OnErrorSendsMoveStopWithinTwentyMilliseconds)
{
  Begin();
  server.SetMoveSpeedDegS(10.0);
  const auto handle = SendAccepted(Goal({30.0, 0, 0, 0, 0, 0}));
  ASSERT_NE(handle, nullptr);
  ASSERT_TRUE(WaitMoving(1.0));
  const auto at = Clock::now();
  rebel->on_error(rclcpp_lifecycle::State());
  ASSERT_TRUE(WaitFor([&]() {return MoveStopCount(server, at) >= 1;}, 500ms));
  EXPECT_LT(server.FirstCommandAfter("Move Stop", at) - at, 20ms);
  const auto result = client->Result(handle);
  EXPECT_EQ(result.result->error_code, MoveResult::LATCHED) << result.result->message;
  EXPECT_FALSE(server.MoveActive());
}

TEST_F(RebelMoveTest, OnDeactivateSendsMoveStopWithinTwentyMilliseconds)
{
  Begin();
  server.SetMoveSpeedDegS(10.0);
  const auto handle = SendAccepted(Goal({30.0, 0, 0, 0, 0, 0}));
  ASSERT_NE(handle, nullptr);
  ASSERT_TRUE(WaitMoving(1.0));
  const auto at = Clock::now();
  rebel->on_deactivate(rclcpp_lifecycle::State());
  ASSERT_TRUE(WaitFor([&]() {return MoveStopCount(server, at) >= 1;}, 500ms));
  EXPECT_LT(server.FirstCommandAfter("Move Stop", at) - at, 20ms);
  const auto result = client->Result(handle);
  EXPECT_EQ(result.result->error_code, MoveResult::FAULT) << result.result->message;
  EXPECT_FALSE(server.MoveActive());
}

TEST_F(RebelMoveTest, StopSendsMoveStopBeforeDisableWithinTwentyMilliseconds)
{
  Begin();
  server.SetMoveSpeedDegS(10.0);
  const auto handle = SendAccepted(Goal({30.0, 0, 0, 0, 0, 0}));
  ASSERT_NE(handle, nullptr);
  ASSERT_TRUE(WaitMoving(1.0));
  // The result has to be requested while the driver's executor still runs.
  auto pendingResult = std::async(
    std::launch::async, [&]() {return client->Result(handle, 5000ms);});
  std::this_thread::sleep_for(100ms);
  const auto at = Clock::now();
  rebel->Stop();
  ASSERT_GE(MoveStopCount(server, at), 1u);
  EXPECT_LT(server.FirstCommandAfter("Move Stop", at) - at, 20ms);
  const auto commandsAfter = server.Commands();
  ASSERT_GE(commandsAfter.size(), 3u);
  EXPECT_EQ(commandsAfter[commandsAfter.size() - 3], "Move Stop");
  EXPECT_EQ(commandsAfter[commandsAfter.size() - 2], "Disable");
  EXPECT_EQ(commandsAfter.back(), "QUIT");
  EXPECT_FALSE(server.MoveActive());
  const auto result = pendingResult.get();
  EXPECT_EQ(result.result->error_code, MoveResult::FAULT) << result.result->message;
  EXPECT_TRUE(result.result->stop_sent);
}

TEST_F(RebelMoveTest, DestructorWithAMoveActiveStopsTheRobot)
{
  Begin();
  server.SetMoveSpeedDegS(10.0);
  const auto handle = SendAccepted(Goal({30.0, 0, 0, 0, 0, 0}));
  ASSERT_NE(handle, nullptr);
  ASSERT_TRUE(WaitMoving(1.0));
  const auto at = Clock::now();
  client.reset();
  rebel.reset();  // ~Rebel -> Stop()
  EXPECT_GE(MoveStopCount(server, at), 1u);
  EXPECT_FALSE(server.MoveActive());
}

TEST_F(RebelMoveTest, ReconnectOfAnIdleDriverStillSendsMoveStopFirst)
{
  Begin();
  StartLoop();
  server.DropClient();
  ASSERT_TRUE(WaitFor([&]() {return server.Commands(2).size() >= 6;}, 3000ms));
  EXPECT_EQ(server.Commands(2), kStartupSequence);
}

// ---- MOVETOEXECERROR / EXECERROR (e-stop during CMD Move Joint, real robot 2026-10-24) ----

constexpr const char * kEstopError = "MOVETOEXECERROR 0 0 MoveTo Hardware error _ESTOP";

namespace
{
bool HasText(const std::string & text, const std::string & part)
{
  return text.find(part) != std::string::npos;
}
}  // namespace

// The robot reports the error while the drives are still enabled and the (ignored) Move Stop does
// not end the motion: the error faults the goal at once and sends Move Stop, but does NOT confirm
// it. Variant: confirmed by standstill after the motion was seen.
TEST_F(RebelMoveTest, RobotErrorFaultsSendsMoveStopAndStopIsConfirmedByStandstill)
{
  Begin();
  server.SetMoveSpeedDegS(10.0);
  server.SetIgnoreMoveStop(true);  // the robot's motion goes on; no END
  TelemetryListener telemetry;
  StartLoop();
  const auto handle = SendAccepted(Goal({30.0, 0, 0, 0, 0, 0}));
  ASSERT_NE(handle, nullptr);
  ASSERT_TRUE(WaitMoving(1.0));
  ASSERT_TRUE(telemetry.WaitState([](const auto & msg) {return msg.move_active;}, 500ms));

  const auto errorAt = Clock::now();
  server.InjectMessage(kEstopError);
  ASSERT_TRUE(WaitFor([&]() {return MoveStopCount(server, errorAt) >= 1;}, 500ms));
  const auto stopAt = server.FirstCommandAfter("Move Stop", errorAt);
  const auto stopDelay = stopAt - errorAt;
  std::printf(
    "[   INFO   ] MOVETOEXECERROR -> CMD Move Stop at the server: %lld ms\n",
    static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(stopDelay).count()));
  EXPECT_LT(stopDelay, 100ms);
  EXPECT_TRUE(WaitFor([&]() {return rebel->HardwareErrorLatched();}, 100ms));
  EXPECT_TRUE(
    telemetry.WaitState(
      [](const auto & msg) {
        return msg.latched && HasText(msg.latch_reason, "Hardware error _ESTOP") &&
        msg.move_stop_pending && msg.move_active;
      }, 300ms));

  // The error alone is no confirmation: 300 ms later the robot still moves and the stop is pending.
  std::this_thread::sleep_for(300ms);
  EXPECT_EQ(rebel->MoveStateValue(), 3);
  EXPECT_TRUE(server.MoveActive());
  EXPECT_TRUE(telemetry.LastState().move_stop_pending);

  server.SetMoveStall(true);  // the arm comes to rest: standstill after motion
  const auto result = client->Result(handle);
  EXPECT_EQ(result.code, rclcpp_action::ResultCode::ABORTED);
  EXPECT_EQ(result.result->error_code, MoveResult::FAULT) << result.result->message;
  EXPECT_TRUE(HasText(result.result->message, "MOVETOEXECERROR")) << result.result->message;
  EXPECT_TRUE(HasText(result.result->message, "Hardware error _ESTOP")) << result.result->message;
  EXPECT_EQ(result.result->end_reason, "ERROR");
  EXPECT_TRUE(result.result->stop_sent);
  EXPECT_TRUE(result.result->stop_confirmed) << result.result->message;
  EXPECT_FALSE(HasText(result.result->message, "NOT confirmed"));
  EXPECT_LT(Clock::now() - errorAt, 1500ms);
  EXPECT_TRUE(rebel->HardwareErrorLatched());
  EXPECT_EQ(rebel->MoveStateValue(), 0);
  EXPECT_TRUE(
    telemetry.WaitState(
      [](const auto & msg) {return msg.latched && !msg.move_active && !msg.move_stop_pending;},
      300ms));
  EXPECT_TRUE(HasText(telemetry.LastState().latch_reason, "Hardware error _ESTOP"));
}

// Variant: confirmed by all six drives disabled (the e-stop itself), not by the error.
TEST_F(RebelMoveTest, RobotErrorStopIsConfirmedByDrivesDisabled)
{
  Begin();
  server.SetMoveSpeedDegS(10.0);
  server.SetIgnoreMoveStop(true);
  StartLoop();
  const auto handle = SendAccepted(Goal({30.0, 0, 0, 0, 0, 0}));
  ASSERT_NE(handle, nullptr);
  ASSERT_TRUE(WaitMoving(1.0));

  const auto errorAt = Clock::now();
  server.InjectMessage(kEstopError);
  ASSERT_TRUE(WaitFor([&]() {return rebel->HardwareErrorLatched();}, 300ms));
  std::this_thread::sleep_for(300ms);
  EXPECT_EQ(rebel->MoveStateValue(), 3);  // still stopping: the error did not confirm
  server.SetIgnoreEnable(true);
  server.SetDisabledMask(0x3F);  // e-stop: all six drives report 'Motor not enabled'
  const auto result = client->Result(handle);
  EXPECT_EQ(result.result->error_code, MoveResult::FAULT) << result.result->message;
  EXPECT_TRUE(HasText(result.result->message, "Hardware error _ESTOP")) << result.result->message;
  EXPECT_EQ(result.result->end_reason, "ERROR");
  EXPECT_TRUE(result.result->stop_confirmed) << result.result->message;
  EXPECT_LT(Clock::now() - errorAt, 1500ms);
  EXPECT_TRUE(rebel->HardwareErrorLatched());
}

// Nothing confirms the stop: the goal ends after the 1 s limit, NOT confirmed, stop stays pending.
TEST_F(RebelMoveTest, RobotErrorAloneNeverConfirmsTheStop)
{
  Begin();
  server.SetMoveSpeedDegS(10.0);
  server.SetIgnoreMoveStop(true);
  TelemetryListener telemetry;
  StartLoop();
  const auto handle = SendAccepted(Goal({30.0, 0, 0, 0, 0, 0}));
  ASSERT_NE(handle, nullptr);
  ASSERT_TRUE(WaitMoving(1.0));
  const auto errorAt = Clock::now();
  server.InjectMessage(kEstopError);
  const auto result = client->Result(handle);
  EXPECT_EQ(result.result->error_code, MoveResult::FAULT) << result.result->message;
  EXPECT_FALSE(result.result->stop_confirmed);
  EXPECT_TRUE(HasText(result.result->message, "NOT confirmed")) << result.result->message;
  EXPECT_TRUE(HasText(result.result->message, "Hardware error _ESTOP")) << result.result->message;
  EXPECT_GE(Clock::now() - errorAt, 950ms);
  EXPECT_TRUE(rebel->HardwareErrorLatched());
  // Still pending after the goal: Move Stop keeps being re-sent.
  const auto afterGoal = Clock::now();
  std::this_thread::sleep_for(400ms);
  EXPECT_GE(MoveStopCount(server, afterGoal), 2u);
  EXPECT_TRUE(
    telemetry.WaitState([](const auto & msg) {return msg.latched && msg.move_stop_pending;}, 200ms));
}

// A robot that stops by itself and ends the move right after the error (the normal e-stop
// sequence on a real robot) confirms through the END, and the error still decides the result.
TEST_F(RebelMoveTest, RobotErrorFollowedByEndIsAFaultNotASuccess)
{
  Begin();
  server.SetMoveSpeedDegS(10.0);
  StartLoop();
  const auto handle = SendAccepted(Goal({30.0, 0, 0, 0, 0, 0}));
  ASSERT_NE(handle, nullptr);
  ASSERT_TRUE(WaitMoving(1.0));
  server.InjectMessage(kEstopError);
  server.InjectMessage("MOVETOEXECEND -1 0 MoveTo PLAN");
  const auto result = client->Result(handle);
  EXPECT_EQ(result.result->error_code, MoveResult::FAULT) << result.result->message;
  EXPECT_NE(result.result->error_code, MoveResult::SUCCESS);
  EXPECT_NE(result.result->error_code, MoveResult::ROBOT_ENDED);
  EXPECT_EQ(result.result->end_reason, "ERROR");
  EXPECT_TRUE(rebel->HardwareErrorLatched());
}

// The robot stands still at (what looks like) the goal when it reports END PLAN after an error:
// without the error taking precedence this would be a SUCCESS.
TEST_F(RebelMoveTest, RobotErrorOverridesEndPlanOfAStandingRobot)
{
  Begin();
  server.SetMoveSpeedDegS(10.0);
  server.SetMoveStall(true);  // never moves: standstill from the start
  StartLoop();
  const auto handle = SendAccepted(Goal({30.0, 0, 0, 0, 0, 0}));
  ASSERT_NE(handle, nullptr);
  ASSERT_TRUE(WaitFor([&]() {return rebel->MoveStateValue() == 2;}, 2000ms));
  std::this_thread::sleep_for(300ms);  // STATUS has shown no motion for the still window
  server.InjectMessage(kEstopError);
  server.InjectMessage("MOVETOEXECEND -1 0 MoveTo PLAN");
  const auto result = client->Result(handle);
  EXPECT_EQ(result.result->error_code, MoveResult::FAULT) << result.result->message;
  EXPECT_TRUE(rebel->HardwareErrorLatched());
}

TEST_F(RebelMoveTest, RobotErrorBeforeTheAckAbortsTheMove)
{
  Begin();
  server.SetMoveSpeedDegS(10.0);
  server.SetSuppressMoveAck(true);
  StartLoop();
  const auto sent = Clock::now();
  const auto handle = SendAccepted(Goal({30.0, 0, 0, 0, 0, 0}));
  ASSERT_NE(handle, nullptr);
  ASSERT_TRUE(WaitFor([&]() {return server.MoveCommands().size() == 1;}, 2000ms));
  server.InjectMessage(kEstopError);
  const auto result = client->Result(handle);
  // Not an ACK timeout (that needs 500 ms of silence): the error decided.
  EXPECT_EQ(result.result->error_code, MoveResult::FAULT) << result.result->message;
  EXPECT_TRUE(HasText(result.result->message, "Hardware error _ESTOP")) << result.result->message;
  EXPECT_TRUE(result.result->stop_sent);
  EXPECT_TRUE(rebel->HardwareErrorLatched());
  const auto stopAt = server.FirstCommandAfter("Move Stop", sent);
  ASSERT_NE(stopAt, Clock::time_point{});
  EXPECT_LT(stopAt - sent, 450ms);
}

// The error of a finished goal is not carried into the next one: after the lifecycle recovery a
// new goal runs to SUCCESS.
TEST_F(RebelMoveTest, RobotErrorOfAnEarlierGoalDoesNotLeakIntoTheNextGoal)
{
  Begin();
  server.SetMoveSpeedDegS(10.0);
  StartLoop();
  const auto handle = SendAccepted(Goal({30.0, 0, 0, 0, 0, 0}));
  ASSERT_NE(handle, nullptr);
  ASSERT_TRUE(WaitMoving(1.0));
  server.InjectMessage(kEstopError);
  const auto result = client->Result(handle);
  ASSERT_EQ(result.result->error_code, MoveResult::FAULT) << result.result->message;
  ASSERT_TRUE(result.result->stop_confirmed) << result.result->message;
  EXPECT_TRUE(rebel->HardwareErrorLatched());

  loop.reset();
  EXPECT_EQ(rebel->on_error(rclcpp_lifecycle::State()), CallbackReturn::SUCCESS);
  EXPECT_EQ(rebel->on_configure(rclcpp_lifecycle::State()), CallbackReturn::SUCCESS);
  EXPECT_EQ(rebel->on_activate(rclcpp_lifecycle::State()), CallbackReturn::SUCCESS);
  ASSERT_FALSE(rebel->HardwareErrorLatched());
  std::this_thread::sleep_for(300ms);  // the start-up sequence and its override STATUS
  server.SetMoveSpeedDegS(60.0);
  StartLoop();
  const auto second = SendAccepted(Goal({6.0, 0, 0, 0, 0, 0}));
  ASSERT_NE(second, nullptr);
  const auto secondResult = client->Result(second);
  EXPECT_EQ(secondResult.result->error_code, MoveResult::SUCCESS) << secondResult.result->message;
  EXPECT_EQ(secondResult.result->end_reason, "PLAN");
  EXPECT_FALSE(rebel->HardwareErrorLatched());
}

// An EXECERROR (program, not MoveTo) while a joint move is active is handled like a
// MOVETOEXECERROR.
TEST_F(RebelMoveTest, ExecErrorWhileAMoveIsActiveFaultsTheGoal)
{
  Begin();
  server.SetMoveSpeedDegS(10.0);
  StartLoop();
  const auto handle = SendAccepted(Goal({30.0, 0, 0, 0, 0, 0}));
  ASSERT_NE(handle, nullptr);
  ASSERT_TRUE(WaitMoving(1.0));
  server.InjectMessage("EXECERROR 0 0 Prog Program aborted by error");
  const auto result = client->Result(handle);
  EXPECT_EQ(result.result->error_code, MoveResult::FAULT) << result.result->message;
  EXPECT_TRUE(HasText(result.result->message, "EXECERROR")) << result.result->message;
  EXPECT_TRUE(HasText(result.result->message, "Program aborted by error"));
  EXPECT_EQ(result.result->end_reason, "ERROR");
  EXPECT_TRUE(rebel->HardwareErrorLatched());
}

// An unsolicited error with no move of ours only logs: no latch, no Move Stop, and it does not
// leak into the next goal.
TEST_F(RebelMoveTest, RobotErrorWhileIdleOnlyLogsAndDoesNotAffectTheNextGoal)
{
  Begin();
  server.SetMoveSpeedDegS(60.0);
  TelemetryListener telemetry;
  StartLoop();
  const auto before = Clock::now();
  server.InjectMessage(kEstopError);
  server.InjectMessage("EXECERROR 0 0 Prog Program aborted by error");
  std::this_thread::sleep_for(300ms);
  EXPECT_FALSE(rebel->HardwareErrorLatched());
  EXPECT_EQ(MoveStopCount(server, before), 0u);
  EXPECT_TRUE(telemetry.WaitState([](const auto & msg) {return msg.activated;}, 300ms));
  EXPECT_FALSE(telemetry.LastState().latched);
  EXPECT_FALSE(telemetry.LastState().move_stop_pending);

  const auto handle = SendAccepted(Goal({6.0, 0, 0, 0, 0, 0}));
  ASSERT_NE(handle, nullptr);
  const auto result = client->Result(handle);
  EXPECT_EQ(result.result->error_code, MoveResult::SUCCESS) << result.result->message;
  EXPECT_EQ(result.result->end_reason, "PLAN");
  EXPECT_FALSE(rebel->HardwareErrorLatched());
}

// An error that arrives while a stop is already in progress latches (the cancel alone does not)
// but does not restart the stop deadline.
TEST_F(RebelMoveTest, RobotErrorWhileStoppingLatchesWithoutRestartingTheStopDeadline)
{
  Begin();
  server.SetMoveSpeedDegS(10.0);
  server.SetIgnoreMoveStop(true);
  StartLoop();
  const auto handle = SendAccepted(Goal({30.0, 0, 0, 0, 0, 0}));
  ASSERT_NE(handle, nullptr);
  ASSERT_TRUE(WaitMoving(1.0));
  const auto cancelAt = Clock::now();
  client->Cancel(handle);
  ASSERT_TRUE(WaitFor([&]() {return rebel->MoveStateValue() == 3;}, 500ms));
  std::this_thread::sleep_for(300ms);
  EXPECT_FALSE(rebel->HardwareErrorLatched());  // a cancel in progress is not a fault
  server.InjectMessage(kEstopError);
  EXPECT_TRUE(WaitFor([&]() {return rebel->HardwareErrorLatched();}, 300ms));
  const auto result = client->Result(handle);
  // The 1 s limit still counts from the start of the stop (a restart would end it at ~1.3 s).
  EXPECT_LT(Clock::now() - cancelAt, 1250ms);
  // A robot error during a client cancel makes the goal a FAULT, never a benign CANCELED.
  EXPECT_EQ(result.code, rclcpp_action::ResultCode::ABORTED);
  EXPECT_EQ(result.result->error_code, MoveResult::FAULT) << result.result->message;
  EXPECT_EQ(result.result->end_reason, "ERROR");
  EXPECT_FALSE(result.result->stop_confirmed);
  EXPECT_TRUE(HasText(result.result->message, "canceled by the client")) << result.result->message;
  EXPECT_TRUE(HasText(result.result->message, "Hardware error _ESTOP")) << result.result->message;
  EXPECT_TRUE(HasText(result.result->message, "NOT confirmed")) << result.result->message;
  EXPECT_TRUE(rebel->HardwareErrorLatched());
}

// Client cancel, then the operator hits the e-stop during the deceleration: the stop IS confirmed
// (all drives disabled) but the goal is a FAULT with end_reason ERROR and the driver is latched.
TEST_F(RebelMoveTest, RobotErrorDuringCancelConfirmedByDrivesOffIsAFaultNotACancel)
{
  Begin();
  server.SetMoveSpeedDegS(10.0);
  server.SetIgnoreMoveStop(true);
  StartLoop();
  const auto handle = SendAccepted(Goal({30.0, 0, 0, 0, 0, 0}));
  ASSERT_NE(handle, nullptr);
  ASSERT_TRUE(WaitMoving(1.0));
  client->Cancel(handle);
  ASSERT_TRUE(WaitFor([&]() {return rebel->MoveStateValue() == 3;}, 500ms));
  EXPECT_FALSE(rebel->HardwareErrorLatched());
  server.InjectMessage(kEstopError);
  ASSERT_TRUE(WaitFor([&]() {return rebel->HardwareErrorLatched();}, 300ms));
  EXPECT_EQ(rebel->MoveStateValue(), 3);  // the error alone confirmed nothing
  server.SetIgnoreEnable(true);
  server.SetDisabledMask(0x3F);  // e-stop: all six drives report 'Motor not enabled'
  const auto result = client->Result(handle);
  EXPECT_EQ(result.code, rclcpp_action::ResultCode::ABORTED);
  EXPECT_EQ(result.result->error_code, MoveResult::FAULT) << result.result->message;
  EXPECT_TRUE(result.result->stop_confirmed) << result.result->message;
  EXPECT_EQ(result.result->end_reason, "ERROR");
  EXPECT_TRUE(HasText(result.result->message, "Hardware error _ESTOP")) << result.result->message;
  EXPECT_FALSE(HasText(result.result->message, "NOT confirmed"));
  EXPECT_TRUE(rebel->HardwareErrorLatched());
}

// Same, confirmed by standstill after the motion was seen.
TEST_F(RebelMoveTest, RobotErrorDuringCancelConfirmedByStandstillIsAFaultNotACancel)
{
  Begin();
  server.SetMoveSpeedDegS(10.0);
  server.SetIgnoreMoveStop(true);
  StartLoop();
  const auto handle = SendAccepted(Goal({30.0, 0, 0, 0, 0, 0}));
  ASSERT_NE(handle, nullptr);
  ASSERT_TRUE(WaitMoving(1.0));
  client->Cancel(handle);
  ASSERT_TRUE(WaitFor([&]() {return rebel->MoveStateValue() == 3;}, 500ms));
  EXPECT_FALSE(rebel->HardwareErrorLatched());
  server.InjectMessage(kEstopError);
  ASSERT_TRUE(WaitFor([&]() {return rebel->HardwareErrorLatched();}, 300ms));
  EXPECT_EQ(rebel->MoveStateValue(), 3);
  server.SetMoveStall(true);  // the arm comes to rest: standstill after motion
  const auto result = client->Result(handle);
  EXPECT_EQ(result.code, rclcpp_action::ResultCode::ABORTED);
  EXPECT_EQ(result.result->error_code, MoveResult::FAULT) << result.result->message;
  EXPECT_TRUE(result.result->stop_confirmed) << result.result->message;
  EXPECT_EQ(result.result->end_reason, "ERROR");
  EXPECT_FALSE(HasText(result.result->message, "NOT confirmed"));
  EXPECT_TRUE(rebel->HardwareErrorLatched());
}

// The move-abort latch is visible in hardware_state while /joint_states would be frozen, and
// robot_joint_states follows the arm that coasts on.
TEST_F(RebelMoveTest, TelemetryTracksTheCoastingArmAfterAMoveAbortLatch)
{
  Begin();
  server.SetMoveSpeedDegS(10.0);
  server.SetIgnoreMoveStop(true);  // the arm keeps moving after the abort
  TelemetryListener telemetry;
  StartLoop();
  const auto handle = SendAccepted(Goal({30.0, 0, 0, 0, 0, 0}));
  ASSERT_NE(handle, nullptr);
  ASSERT_TRUE(WaitMoving(1.0));
  server.InjectMessage(kEstopError);
  ASSERT_TRUE(
    telemetry.WaitState([](const auto & msg) {return msg.latched && msg.move_stop_pending;}, 500ms));
  const double atLatch = telemetry.LastJoint().msg.position[0];
  // While latched and stopping, the arm keeps moving: the published position keeps changing.
  std::this_thread::sleep_for(250ms);
  const double later = telemetry.LastJoint().msg.position[0];
  EXPECT_GT(later - atLatch, 0.5 * kDegToRad);
  EXPECT_NEAR(later, server.CurrentDeg()[0] * kDegToRad, 0.5 * kDegToRad);
  const auto result = client->Result(handle);
  EXPECT_EQ(result.result->error_code, MoveResult::FAULT) << result.result->message;
}

// hardware_state shows the configured MoveJoint caps (after the 100 % hard-cap clamp).
TEST_F(RebelMoveTest, HardwareStateCarriesTheConfiguredMoveJointCaps)
{
  Begin({{"move_joint_max_velocity_percent", "60"}, {"move_joint_max_acceleration_percent", "10"}});
  TelemetryListener telemetry;
  StartLoop();
  ASSERT_TRUE(telemetry.WaitState([](const auto & msg) {return msg.activated;}, 2000ms));
  EXPECT_DOUBLE_EQ(telemetry.LastState().move_joint_max_velocity_percent, 60.0);
  EXPECT_DOUBLE_EQ(telemetry.LastState().move_joint_max_acceleration_percent, 10.0);
}

TEST_F(RebelMoveTest, HardwareStateCarriesTheDefaultMoveJointCaps)
{
  Begin();
  TelemetryListener telemetry;
  StartLoop();
  ASSERT_TRUE(telemetry.WaitState([](const auto & msg) {return msg.activated;}, 2000ms));
  EXPECT_DOUBLE_EQ(telemetry.LastState().move_joint_max_velocity_percent, 25.0);
  EXPECT_DOUBLE_EQ(telemetry.LastState().move_joint_max_acceleration_percent, 40.0);
}

TEST_F(RebelMoveTest, HardwareStateCapsAboveTheHardCapsAreClamped)
{
  Begin({{"move_joint_max_velocity_percent", "150"}, {"move_joint_max_acceleration_percent", "150"}});
  TelemetryListener telemetry;
  StartLoop();
  ASSERT_TRUE(telemetry.WaitState([](const auto & msg) {return msg.activated;}, 2000ms));
  EXPECT_DOUBLE_EQ(telemetry.LastState().move_joint_max_velocity_percent, 100.0);
  EXPECT_DOUBLE_EQ(telemetry.LastState().move_joint_max_acceleration_percent, 40.0);  // hard cap
}

// ---- Dispatch authorization, late errors, receive-time stamps, dropped messages ----

// A robot-reported error recorded after the goal was accepted, with no other flag set, must stop
// the dispatch: nothing is sent, the goal is a FAULT with the robot's reason, the driver latches.
TEST_F(RebelMoveTest, RobotErrorRecordedBeforeTheSendWithNoOtherFlagIsNeverDispatched)
{
  Begin();
  server.SetMoveSpeedDegS(10.0);
  StartLoop();
  const auto errorInTheSendWindow = [this]() {
      server.InjectMessage(kEstopError);
      std::this_thread::sleep_for(200ms);  // the message thread records it (state is Sending)
    };
  rebel->SetBeforeMoveSendHookForTest(errorInTheSendWindow);
  const auto handle = SendAccepted(Goal({30.0, 0, 0, 0, 0, 0}));
  ASSERT_NE(handle, nullptr);
  const auto result = client->Result(handle);
  rebel->SetBeforeMoveSendHookForTest(nullptr);
  EXPECT_EQ(result.code, rclcpp_action::ResultCode::ABORTED);
  EXPECT_EQ(result.result->error_code, MoveResult::FAULT) << result.result->message;
  EXPECT_EQ(result.result->end_reason, "ERROR");
  EXPECT_TRUE(HasText(result.result->message, "Hardware error _ESTOP")) << result.result->message;
  EXPECT_TRUE(HasText(result.result->message, "before sending")) << result.result->message;
  EXPECT_FALSE(result.result->stop_sent);  // no stop "sent" for a move the robot never got
  EXPECT_FALSE(result.result->stop_confirmed);
  ExpectNoMoveJointSent();
  EXPECT_TRUE(rebel->HardwareErrorLatched());
  EXPECT_EQ(rebel->MoveStateValue(), 0);
  EXPECT_FALSE(server.MoveActive());
}

// A latch that arrives inside the send window (after the early abort check, while the send lock
// is held) blocks the dispatch.
TEST_F(RebelMoveTest, LatchInsideTheSendWindowBlocksTheMoveJoint)
{
  Begin();
  server.SetMoveSpeedDegS(10.0);
  StartLoop();
  rebel->SetBeforeMoveSendHookForTest(
    [this]() {rebel->on_error(rclcpp_lifecycle::State());});  // latches, deactivates
  const auto handle = SendAccepted(Goal({30.0, 0, 0, 0, 0, 0}));
  ASSERT_NE(handle, nullptr);
  const auto result = client->Result(handle);
  rebel->SetBeforeMoveSendHookForTest(nullptr);
  EXPECT_EQ(result.result->error_code, MoveResult::LATCHED) << result.result->message;
  EXPECT_FALSE(result.result->stop_sent);
  ExpectNoMoveJointSent();
  EXPECT_TRUE(rebel->HardwareErrorLatched());
  EXPECT_EQ(rebel->MoveStateValue(), 0);
  EXPECT_FALSE(server.MoveActive());
}

// A client cancel that arrives inside the send window blocks the dispatch: CANCELED, nothing sent.
TEST_F(RebelMoveTest, CancelInsideTheSendWindowBlocksTheMoveJoint)
{
  Begin();
  server.SetMoveSpeedDegS(10.0);
  StartLoop();
  std::atomic<bool> entered{false};
  std::atomic<bool> release{false};
  rebel->SetBeforeMoveSendHookForTest(
    [&]() {
      entered = true;
      WaitFor([&]() {return release.load();}, 3000ms);
    });
  const auto handle = SendAccepted(Goal({30.0, 0, 0, 0, 0, 0}));
  ASSERT_NE(handle, nullptr);
  ASSERT_TRUE(WaitFor([&]() {return entered.load();}, 2000ms));
  client->Cancel(handle);
  std::this_thread::sleep_for(200ms);  // the cancel callback has run
  release = true;
  const auto result = client->Result(handle);
  rebel->SetBeforeMoveSendHookForTest(nullptr);
  EXPECT_EQ(result.code, rclcpp_action::ResultCode::CANCELED);
  EXPECT_EQ(result.result->error_code, MoveResult::CANCELED) << result.result->message;
  EXPECT_FALSE(result.result->stop_sent);
  ExpectNoMoveJointSent();
  EXPECT_FALSE(rebel->HardwareErrorLatched());  // a cancel before the send is not a fault
  EXPECT_EQ(rebel->MoveStateValue(), 0);
}

// MOVETOEXECERROR after the supervision loop decided SUCCESS (the robot's END PLAN, standing at
// the goal) but before the result is reported: the goal is a FAULT, never a SUCCESS.
TEST_F(RebelMoveTest, RobotErrorArrivingAfterTheSuccessDecisionBeforeTheResultIsAFault)
{
  Begin();
  server.SetMoveSpeedDegS(60.0);
  StartLoop();
  std::atomic<bool> once{true};
  rebel->SetBeforeMoveFinishHookForTest(
    [&]() {
      if (once.exchange(false)) {
        server.InjectMessage(kEstopError);
        std::this_thread::sleep_for(200ms);  // recorded; the move is not idle yet
      }
    });
  const auto handle = SendAccepted(Goal({6.0, 0, 0, 0, 0, 0}));
  ASSERT_NE(handle, nullptr);
  const auto result = client->Result(handle);
  rebel->SetBeforeMoveFinishHookForTest(nullptr);
  EXPECT_FALSE(once.load());  // the hook ran: the SUCCESS path was taken
  EXPECT_EQ(result.code, rclcpp_action::ResultCode::ABORTED);
  EXPECT_EQ(result.result->error_code, MoveResult::FAULT) << result.result->message;
  EXPECT_NE(result.result->error_code, MoveResult::SUCCESS);
  EXPECT_EQ(result.result->end_reason, "ERROR");
  EXPECT_TRUE(HasText(result.result->message, "Hardware error _ESTOP")) << result.result->message;
  EXPECT_TRUE(rebel->HardwareErrorLatched());
  EXPECT_EQ(rebel->MoveStateValue(), 0);
}

// An error that arrives after the goal ended (idle) is only logged: not latched, not recorded
// into the next goal.
TEST_F(RebelMoveTest, RobotErrorWhileIdleAfterAGoalIsNotRecordedIntoTheNextGoal)
{
  Begin();
  server.SetMoveSpeedDegS(60.0);
  StartLoop();
  auto handle = SendAccepted(Goal({5.0, 0, 0, 0, 0, 0}));
  ASSERT_NE(handle, nullptr);
  ASSERT_EQ(client->Result(handle).result->error_code, MoveResult::SUCCESS);
  const auto idleAt = Clock::now();
  server.InjectMessage(kEstopError);
  std::this_thread::sleep_for(300ms);
  EXPECT_FALSE(rebel->HardwareErrorLatched());
  EXPECT_EQ(MoveStopCount(server, idleAt), 0u);
  handle = SendAccepted(Goal({-5.0, 0, 0, 0, 0, 0}));
  ASSERT_NE(handle, nullptr);
  const auto result = client->Result(handle);
  EXPECT_EQ(result.result->error_code, MoveResult::SUCCESS) << result.result->message;
  EXPECT_EQ(result.result->end_reason, "PLAN");
  EXPECT_FALSE(rebel->HardwareErrorLatched());
}

// The STATUS stamp (snapshot, robot_joint_states, and with it the age tiers) is the time the
// socket read the bytes. A message thread that is 300 ms late must not make those samples look
// 300 ms younger than they are.
TEST_F(RebelMoveTest, StatusStampIsTheReceiveTimeWhenTheMessageThreadIsDelayed)
{
  Begin();
  TelemetryListener telemetry;
  StartLoop();
  ASSERT_TRUE(telemetry.WaitJoint([](const auto &) {return true;}, 1000ms));
  std::atomic<bool> once{true};
  std::atomic<std::int64_t> delayedAtNs{0};
  rebel->SetBeforeHandleMessageHookForTest(
    [&]() {
      if (once.exchange(false)) {
        delayedAtNs = SystemNowNs();
        std::this_thread::sleep_for(300ms);
      }
    });
  ASSERT_TRUE(WaitFor([&]() {return !once.load();}, 1000ms));
  std::this_thread::sleep_for(500ms);  // the backlog is processed
  rebel->SetBeforeHandleMessageHookForTest(nullptr);

  const std::int64_t delayedAt = delayedAtNs.load();
  ASSERT_NE(delayedAt, 0);
  std::int64_t earliestStampOfBacklog = std::numeric_limits<std::int64_t>::max();
  std::size_t backlogSamples = 0;
  for (const auto & sample : telemetry.Joints()) {
    // Published (and received here) only after the delay ended.
    if (sample.receivedSystemNs >= delayedAt + 290'000'000 &&
      sample.receivedSystemNs <= delayedAt + 340'000'000)
    {
      ++backlogSamples;
      earliestStampOfBacklog = std::min(
        earliestStampOfBacklog,
        static_cast<std::int64_t>(rclcpp::Time(sample.msg.header.stamp).nanoseconds()));
    }
  }
  ASSERT_GE(backlogSamples, 3u);
  // Stamped at processing time the backlog would start at ~delayedAt + 300 ms.
  EXPECT_LT(earliestStampOfBacklog - delayedAt, 200'000'000) <<
    "stamp lag " << (earliestStampOfBacklog - delayedAt) / 1'000'000 << " ms";
}

// Messages discarded because the consumer fell behind during a joint move may include the robot's
// MOVETOEXECERROR: the goal faults and the driver latches.
TEST_F(RebelMoveTest, DroppedMessagesDuringAMoveFaultAndLatch)
{
  Begin();
  server.SetMoveSpeedDegS(10.0);
  StartLoop();
  const auto handle = SendAccepted(Goal({30.0, 0, 0, 0, 0, 0}));
  ASSERT_NE(handle, nullptr);
  ASSERT_TRUE(WaitMoving(1.0));
  std::atomic<bool> stalled{false};
  std::atomic<bool> once{true};
  rebel->SetBeforeHandleMessageHookForTest(
    [&]() {
      if (once.exchange(false)) {
        stalled = true;
        std::this_thread::sleep_for(120ms);  // < the 200 ms STATUS abort tier
      }
    });
  ASSERT_TRUE(WaitFor([&]() {return stalled.load();}, 1000ms));
  for (int i = 0; i < 40; ++i) {
    server.InjectMessage("INFO filler " + std::to_string(i));
  }
  const auto result = client->Result(handle);
  rebel->SetBeforeHandleMessageHookForTest(nullptr);
  EXPECT_EQ(result.code, rclcpp_action::ResultCode::ABORTED);
  EXPECT_EQ(result.result->error_code, MoveResult::FAULT) << result.result->message;
  EXPECT_TRUE(HasText(result.result->message, "robot messages were dropped"))
    << result.result->message;
  EXPECT_TRUE(result.result->stop_sent);
  EXPECT_TRUE(rebel->HardwareErrorLatched());
  EXPECT_GE(MoveStopCount(server, Clock::now() - 3s), 1u);
}

// Same before the send: messages dropped after the goal was accepted stop the dispatch.
TEST_F(RebelMoveTest, DroppedMessagesBeforeTheSendBlockTheMoveJoint)
{
  Begin();
  server.SetMoveSpeedDegS(10.0);
  StartLoop();
  const auto dropInTheSendWindow = [this]() {
      rebel->SetBeforeHandleMessageHookForTest([]() {std::this_thread::sleep_for(300ms);});
      std::this_thread::sleep_for(60ms);  // the message thread is inside the delay
      for (int i = 0; i < 40; ++i) {
        server.InjectMessage("INFO filler " + std::to_string(i));
      }
      std::this_thread::sleep_for(100ms);  // the socket queue overflowed
    };
  rebel->SetBeforeMoveSendHookForTest(dropInTheSendWindow);
  const auto handle = SendAccepted(Goal({30.0, 0, 0, 0, 0, 0}));
  ASSERT_NE(handle, nullptr);
  const auto result = client->Result(handle);
  rebel->SetBeforeMoveSendHookForTest(nullptr);
  rebel->SetBeforeHandleMessageHookForTest(nullptr);
  EXPECT_EQ(result.result->error_code, MoveResult::FAULT) << result.result->message;
  EXPECT_TRUE(HasText(result.result->message, "robot messages were dropped"))
    << result.result->message;
  EXPECT_TRUE(HasText(result.result->message, "before sending")) << result.result->message;
  EXPECT_FALSE(result.result->stop_sent);
  ExpectNoMoveJointSent();
  EXPECT_TRUE(rebel->HardwareErrorLatched());
}

// A cancel whose Move Stop is not confirmed within the limit is a FAULT, not a CANCELED (the
// robot may still be moving), with the stop still pending and the driver latched.
TEST_F(RebelMoveTest, UnconfirmedStopAfterAClientCancelIsAFaultNotACancel)
{
  Begin();
  server.SetMoveSpeedDegS(10.0);
  server.SetIgnoreMoveStop(true);
  StartLoop();
  const auto handle = SendAccepted(Goal({30.0, 0, 0, 0, 0, 0}));
  ASSERT_NE(handle, nullptr);
  ASSERT_TRUE(WaitMoving(1.0));
  client->Cancel(handle);
  const auto result = client->Result(handle);
  EXPECT_EQ(result.code, rclcpp_action::ResultCode::ABORTED);
  EXPECT_EQ(result.result->error_code, MoveResult::FAULT) << result.result->message;
  EXPECT_NE(result.result->error_code, MoveResult::CANCELED);
  EXPECT_FALSE(result.result->stop_confirmed);
  EXPECT_TRUE(result.result->stop_sent);
  EXPECT_TRUE(HasText(result.result->message, "canceled by the client")) << result.result->message;
  EXPECT_TRUE(HasText(result.result->message, "NOT confirmed")) << result.result->message;
  EXPECT_TRUE(rebel->HardwareErrorLatched());
}

TEST_F(RebelMoveTest, ServerStaysIdleAndUnusedByDefault)
{
  // Behaviour without any goal is the old behaviour (apart from the Move Stop at connect).
  Begin();
  StartLoop(0.1);
  std::this_thread::sleep_for(300ms);
  EXPECT_EQ(server.Commands(), kStartupSequence);
  EXPECT_TRUE(AnyNonZeroJog(server, Clock::now() - 100ms));
  EXPECT_EQ(rebel->MoveStateValue(), 0);
}

// ---- peer fault: the other arm of a fault group moves ----
TEST_F(RebelFaultGroupTest, PeerFaultSendsMoveStopWithinTwentyMilliseconds)
{
  Arm left;
  Arm right;
  StartArmedPair(left, right, "movegroup");
  left.server->SetOverride(100.0f);
  right.server->SetOverride(100.0f);
  std::this_thread::sleep_for(150ms);
  ASSERT_EQ(left.rebel->on_activate(rclcpp_lifecycle::State()), CallbackReturn::SUCCESS);
  left.server->SetMoveSpeedDegS(10.0);

  MoveClient client("/igus_rebel_left/move_joint");
  ASSERT_TRUE(client.WaitForServer());
  const auto goal = MakeMoveGoal(
    left.rebel->JointNames(), left.server->CurrentDeg(), {30.0, 0, 0, 0, 0, 0});
  const auto handle = client.Send(goal);
  ASSERT_NE(handle, nullptr);
  ASSERT_TRUE(
    WaitFor(
      [&]() {
        return left.server->MoveActive() && left.server->CurrentDeg()[0] > 11.0f;
      }, 3000ms));

  const auto at = Clock::now();
  right.rebel->on_error(rclcpp_lifecycle::State());  // root-cause fault of the other arm
  ASSERT_TRUE(WaitFor([&]() {return MoveStopCount(*left.server, at) >= 1;}, 500ms));
  EXPECT_LT(left.server->FirstCommandAfter("Move Stop", at) - at, 20ms);
  const auto result = client.Result(handle);
  EXPECT_EQ(result.result->error_code, MoveResult::FAULT) << result.result->message;
  EXPECT_TRUE(result.result->stop_sent);
  EXPECT_FALSE(left.server->MoveActive());
  EXPECT_TRUE(left.rebel->HardwareErrorLatched());
}

// A supervisor abort (here: path deviation) is a fault of the group: the other arm is stopped.
TEST_F(RebelFaultGroupTest, SupervisorPathDeviationStopsTheOtherArm)
{
  Arm left;
  Arm right;
  StartArmedPair(left, right, "supgroup");
  left.server->SetOverride(100.0f);
  right.server->SetOverride(100.0f);
  std::this_thread::sleep_for(150ms);
  ASSERT_EQ(left.rebel->on_activate(rclcpp_lifecycle::State()), CallbackReturn::SUCCESS);
  left.server->SetMoveSpeedDegS(20.0);
  left.server->SetMoveDeviationDeg(5.0);

  MoveClient client("/igus_rebel_left/move_joint");
  ASSERT_TRUE(client.WaitForServer());
  const auto goal = MakeMoveGoal(
    left.rebel->JointNames(), left.server->CurrentDeg(), {30.0, 0, 0, 0, 0, 0});
  const auto handle = client.Send(goal);
  ASSERT_NE(handle, nullptr);
  const auto result = client.Result(handle);
  EXPECT_EQ(result.result->error_code, MoveResult::PATH_DEVIATION) << result.result->message;
  EXPECT_TRUE(left.rebel->HardwareErrorLatched());
  EXPECT_TRUE(right.rebel->HardwareErrorLatched());
  EXPECT_FALSE(left.server->MoveActive());
}

// Every abort after CMD Move Joint was sent is a fault of the group, also while the group monitor
// is not armed yet (the first 500 ms after connect): the move thread latches and reports it.
class RebelPostSendAbortTest : public RebelFaultGroupTest
{
protected:
  enum class Abort
  {
    kAckTimeout, kCmdError, kKinematics, kDrivesDisabled, kSendFailure, kRobotError
  };

  void RunAbort(Abort abort, std::uint8_t expectedCode)
  {
    Arm left;
    Arm right;
    StartArm(left, "left", "postsend");
    StartArm(right, "right", "postsend");
    ASSERT_NE(left.rebel, nullptr);
    ASSERT_NE(right.rebel, nullptr);
    left.server->SetOverride(100.0f);
    right.server->SetOverride(100.0f);
    ASSERT_TRUE(WaitFor([&]() {return left.server->Commands().size() >= 6;}, 3000ms));
    std::this_thread::sleep_for(100ms);  // the override STATUS reaches the driver
    ASSERT_EQ(left.rebel->on_activate(rclcpp_lifecycle::State()), CallbackReturn::SUCCESS);
    left.server->SetMoveSpeedDegS(10.0);
    if (abort == Abort::kAckTimeout) {
      left.server->SetSuppressMoveAck(true);
    } else if (abort == Abort::kCmdError) {
      left.server->SetMoveCmdError(true);
    } else if (abort == Abort::kSendFailure) {
      left.rebel->FailNextMoveJointSendForTest();
    }

    MoveClient client("/igus_rebel_left/move_joint");
    ASSERT_TRUE(client.WaitForServer());
    const auto handle = client.Send(
      MakeMoveGoal(left.rebel->JointNames(), left.server->CurrentDeg(), {30.0, 0, 0, 0, 0, 0}));
    ASSERT_NE(handle, nullptr);
    if (abort == Abort::kKinematics || abort == Abort::kDrivesDisabled) {
      ASSERT_TRUE(WaitFor([&]() {return left.server->MoveActive();}, 2000ms));
      // The grace window: the group monitor (CheckFaultGroupStatus) is not armed, so only the
      // move thread can report this.
      ASSERT_FALSE(left.rebel->FaultMonitorArmed());
      if (abort == Abort::kKinematics) {
        left.server->SetKinstate(99);
      } else {
        left.server->SetDisabledMask(0x01);
        left.server->SetIgnoreEnable(true);
      }
    }
    if (abort == Abort::kRobotError) {
      ASSERT_TRUE(WaitFor([&]() {return left.server->MoveActive();}, 2000ms));
      left.server->InjectMessage("MOVETOEXECERROR 0 0 MoveTo Hardware error _ESTOP");
    }
    const auto result = client.Result(handle);
    EXPECT_EQ(result.result->error_code, expectedCode) << result.result->message;
    EXPECT_TRUE(result.result->stop_sent);
    EXPECT_TRUE(left.rebel->HardwareErrorLatched());
    EXPECT_TRUE(WaitFor([&]() {return right.rebel->HardwareErrorLatched();}, 1000ms))
      << "the peer arm was not stopped";
    EXPECT_FALSE(left.server->MoveActive());
  }
};

TEST_F(RebelPostSendAbortTest, AckTimeoutStopsTheOtherArm)
{
  RunAbort(Abort::kAckTimeout, MoveResult::ACK_TIMEOUT);
}

TEST_F(RebelPostSendAbortTest, CmdErrorStopsTheOtherArm)
{
  RunAbort(Abort::kCmdError, MoveResult::CMD_ERROR);
}

TEST_F(RebelPostSendAbortTest, MoveJointSendFailureStopsTheOtherArm)
{
  RunAbort(Abort::kSendFailure, MoveResult::CMD_ERROR);
}

TEST_F(RebelPostSendAbortTest, RobotReportedErrorStopsTheOtherArm)
{
  RunAbort(Abort::kRobotError, MoveResult::FAULT);
}

TEST_F(RebelPostSendAbortTest, KinematicsErrorBeforeTheMonitorIsArmedStopsTheOtherArm)
{
  RunAbort(Abort::kKinematics, MoveResult::FAULT);
}

TEST_F(RebelPostSendAbortTest, DrivesDisabledBeforeTheMonitorIsArmedStopsTheOtherArm)
{
  RunAbort(Abort::kDrivesDisabled, MoveResult::FAULT);
}

// Pre-send refusals and an operator cancel are not faults of the group.
TEST_F(RebelFaultGroupTest, CancelAndRefusalDoNotStopTheOtherArm)
{
  Arm left;
  Arm right;
  StartArmedPair(left, right, "cancelgroup");
  left.server->SetOverride(100.0f);
  right.server->SetOverride(100.0f);
  std::this_thread::sleep_for(150ms);
  ASSERT_EQ(left.rebel->on_activate(rclcpp_lifecycle::State()), CallbackReturn::SUCCESS);
  left.server->SetMoveSpeedDegS(10.0);
  MoveClient client("/igus_rebel_left/move_joint");
  ASSERT_TRUE(client.WaitForServer());
  auto bad = MakeMoveGoal(
    left.rebel->JointNames(), left.server->CurrentDeg(), {30.0, 0, 0, 0, 0, 0});
  bad.start_positions[0] += 0.5;  // START_DEVIATION: refused before anything is sent
  auto refused = client.Send(bad);
  ASSERT_NE(refused, nullptr);
  EXPECT_EQ(client.Result(refused).result->error_code, MoveResult::START_DEVIATION);

  const auto handle = client.Send(
    MakeMoveGoal(left.rebel->JointNames(), left.server->CurrentDeg(), {30.0, 0, 0, 0, 0, 0}));
  ASSERT_NE(handle, nullptr);
  ASSERT_TRUE(WaitFor([&]() {return left.server->MoveActive();}, 2000ms));
  std::this_thread::sleep_for(300ms);
  client.Cancel(handle);
  EXPECT_EQ(client.Result(handle).result->error_code, MoveResult::CANCELED);
  std::this_thread::sleep_for(200ms);
  EXPECT_FALSE(left.rebel->HardwareErrorLatched());
  EXPECT_FALSE(right.rebel->HardwareErrorLatched());
}

// The emulated robot keeps moving when its client goes away (matches the real controller).
TEST(FakeCriServerMove, MoveContinuesWithoutAClientAndStopsOnMoveStop)
{
  FakeCriServer server;
  server.SetMoveSpeedDegS(40.0);
  const auto connect = [&]() {
      const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
      sockaddr_in address{};
      address.sin_family = AF_INET;
      address.sin_port = htons(server.port());
      address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
      EXPECT_EQ(::connect(fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)), 0);
      return fd;
    };
  const auto sendFrame = [](int fd, const std::string & text) {
      ::send(fd, text.data(), text.size(), MSG_NOSIGNAL);
    };
  const auto readAll = [](int fd, std::chrono::milliseconds duration) {
      std::string data;
      const auto end = Clock::now() + duration;
      while (Clock::now() < end) {
        pollfd p{fd, POLLIN, 0};
        if (::poll(&p, 1, 20) > 0) {
          char buffer[8192];
          const ssize_t n = ::recv(fd, buffer, sizeof(buffer), 0);
          if (n <= 0) {break;}
          data.append(buffer, static_cast<std::size_t>(n));
        }
      }
      return data;
    };

  int fd = connect();
  sendFrame(
    fd,
    "CRISTART 5 CMD Move Joint 40.000 20.000 -30.000 40.000 50.000 -60.000 0 0 0 25 40 CRIEND\n");
  const std::string first = readAll(fd, 300ms);
  EXPECT_NE(first.find("MOVETOEXECACK 0 0 MoveTo"), std::string::npos);
  EXPECT_NE(first.find("CMDACK 5"), std::string::npos);
  ::close(fd);  // client gone: the move goes on
  std::this_thread::sleep_for(300ms);
  EXPECT_GT(server.CurrentDeg()[0], 10.0f);
  EXPECT_TRUE(server.MoveActive() || server.CurrentDeg()[0] >= 39.9f);

  fd = connect();
  sendFrame(
    fd,
    "CRISTART 1 CMD Move Joint 10.000 20.000 -30.000 40.000 50.000 -60.000 0 0 0 25 40 CRIEND\n");
  readAll(fd, 100ms);
  // A second Move Joint replaces the target (and the first never reaches 40).
  sendFrame(fd, "CRISTART 2 CMD Move Stop CRIEND\n");
  const std::string stopped = readAll(fd, 200ms);
  EXPECT_NE(stopped.find("MOVETOEXECEND 0 0 MoveTo USER"), std::string::npos) << stopped;
  EXPECT_FALSE(server.MoveActive());
  ::close(fd);
}


// ======================================================================================
// Cartesian jog mode (CRI route): CMD MotionTypeCartBase + ALIVEJOG X Y Z A B C
// ======================================================================================
namespace
{
using TwistStamped = geometry_msgs::msg::TwistStamped;
using Twist6 = std::array<double, 6>;

// Publishes ~/cart_jog_cmd and calls ~/cart_jog_enable of the driver node, from its own node.
class CartClient
{
public:
  explicit CartClient(const std::string & driverNode = "igus_rebel")
  {
    node_ = std::make_shared<rclcpp::Node>("cart_test_client_" + std::to_string(counter_++));
    pub_ = node_->create_publisher<TwistStamped>("/" + driverNode + "/cart_jog_cmd", 10);
    client_ = node_->create_client<std_srvs::srv::SetBool>("/" + driverNode + "/cart_jog_enable");
    executor_.add_node(node_);
    thread_ = std::thread([this]() {executor_.spin(); finished_ = true;});
  }
  ~CartClient()
  {
    StopStream();
    while (!finished_) {
      executor_.cancel();
      std::this_thread::sleep_for(2ms);
    }
    if (thread_.joinable()) {
      thread_.join();
    }
  }
  bool WaitForService() {return client_->wait_for_service(5s);}

  struct Reply
  {
    bool answered = false;
    bool success = false;
    std::string message;
  };
  Reply Enable(bool enable)
  {
    auto request = std::make_shared<std_srvs::srv::SetBool::Request>();
    request->data = enable;
    auto future = client_->async_send_request(request);
    Reply reply;
    if (future.wait_for(8s) == std::future_status::ready) {
      const auto response = future.get();
      reply = {true, response->success, response->message};
    }
    return reply;
  }
  rclcpp::Time Now() {return node_->now();}
  void Publish(const Twist6 & v, const std::string & frame = "base_link")
  {
    PublishStamped(v, node_->now(), frame);
  }
  void PublishStamped(
    const Twist6 & v, const rclcpp::Time & stamp, const std::string & frame = "base_link")
  {
    TwistStamped msg;
    msg.header.frame_id = frame;
    msg.header.stamp = stamp;
    msg.twist.linear.x = v[0];
    msg.twist.linear.y = v[1];
    msg.twist.linear.z = v[2];
    msg.twist.angular.x = v[3];
    msg.twist.angular.y = v[4];
    msg.twist.angular.z = v[5];
    pub_->publish(msg);
  }
  // Re-publishes the (changeable) command at 50 Hz, like a teleop timer.
  void StartStream(const Twist6 & v, const std::string & frame = "base_link")
  {
    StopStream();
    {
      std::lock_guard<std::mutex> lock(streamMutex_);
      streamValue_ = v;
      streamFrame_ = frame;
    }
    streaming_ = true;
    streamThread_ = std::thread(
      [this]() {
        while (streaming_) {
          Twist6 value;
          std::string frame;
          {
            std::lock_guard<std::mutex> lock(streamMutex_);
            value = streamValue_;
            frame = streamFrame_;
          }
          Publish(value, frame);
          std::this_thread::sleep_for(20ms);
        }
      });
  }
  void SetStream(const Twist6 & v)
  {
    std::lock_guard<std::mutex> lock(streamMutex_);
    streamValue_ = v;
  }
  void StopStream()
  {
    streaming_ = false;
    if (streamThread_.joinable()) {
      streamThread_.join();
    }
  }

private:
  static inline std::atomic<int> counter_{0};
  std::atomic<bool> finished_{false};
  std::atomic<bool> streaming_{false};
  rclcpp::Node::SharedPtr node_;
  rclcpp::Publisher<TwistStamped>::SharedPtr pub_;
  rclcpp::Client<std_srvs::srv::SetBool>::SharedPtr client_;
  rclcpp::executors::SingleThreadedExecutor executor_;
  std::thread thread_;
  std::thread streamThread_;
  std::mutex streamMutex_;
  Twist6 streamValue_{};
  std::string streamFrame_;
};

bool JogNear(const std::array<double, 6> & jog, const Twist6 & expected, double tolerance = 1e-3)
{
  for (std::size_t i = 0; i < 6; ++i) {
    if (std::fabs(jog[i] - expected[i]) > tolerance) {
      return false;
    }
  }
  return true;
}

// True once an ALIVEJOG matching `expected` was received after `after`.
bool SawJog(
  FakeCriServer & server, Clock::time_point after, const Twist6 & expected,
  double tolerance = 1e-3)
{
  for (const auto & frame : server.AliveJogs(after)) {
    if (JogNear(ParseJog(frame.body), expected, tolerance)) {
      return true;
    }
  }
  return false;
}

// Index of the first command equal to `command` in the commands of a connection, or -1.
int CommandIndex(
  FakeCriServer & server, const std::string & command, std::size_t connection = 1,
  std::size_t occurrence = 0)
{
  const auto commands = server.Commands(connection);
  std::size_t seen = 0;
  for (std::size_t i = 0; i < commands.size(); ++i) {
    if (commands[i] == command && seen++ == occurrence) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

// Newest ALIVEJOG (all six values) received before the `occurrence`-th "CMD <command>" frame of
// connection 1; empty if there is none. Checks "zero jog first, then the CMD".
std::vector<double> JogBeforeCommand(
  FakeCriServer & server, const std::string & command, std::size_t occurrence = 0)
{
  std::vector<double> last;
  std::size_t seen = 0;
  for (const auto & frame : server.Frames()) {
    if (frame.connection != 1) {continue;}
    if (frame.body.rfind("ALIVEJOG ", 0) == 0) {
      const auto jog = ParseJog(frame.body);
      last.assign(jog.begin(), jog.end());
    } else if (frame.body == "CMD " + command && seen++ == occurrence) {
      return last;
    }
  }
  return {};
}

bool AllZero(const std::vector<double> & values)
{
  return !values.empty() &&
         std::all_of(values.begin(), values.end(), [](double v) {return v == 0.0;});
}

class RebelCartJogTest : public RebelMoveTest
{
protected:
  void BeginCart(
    const std::map<std::string, std::string> & hardwareParameters = {}, bool activate = true)
  {
    Begin(hardwareParameters, activate);
    if (HasFatalFailure()) {
      return;
    }
    StartLoop(0.0);
    cart = std::make_unique<CartClient>();
    ASSERT_TRUE(cart->WaitForService());
  }
  void TearDown() override
  {
    cart.reset();
    RebelMoveTest::TearDown();
  }
  // Enters the mode through the driver method and expects success.
  void EnterMode()
  {
    std::string message;
    ASSERT_TRUE(rebel->SetCartJogEnabled(true, message)) << message;
    ASSERT_TRUE(rebel->CartJogActive());
  }
  bool Refused(const std::string & expectedText)
  {
    std::string message;
    const bool ok = rebel->SetCartJogEnabled(true, message);
    EXPECT_FALSE(ok) << message;
    EXPECT_NE(message.find(expectedText), std::string::npos) << message;
    return !ok;
  }

  std::unique_ptr<CartClient> cart;
};
}  // namespace

TEST_F(RebelCartJogTest, EntryRefusedWhenNotActivated)
{
  BeginCart({}, false);
  Refused("not active");
  EXPECT_EQ(server.CountCommands("MotionTypeCartBase"), 0u);
  EXPECT_EQ(rebel->CartJogStateValue(), 0);
}

TEST_F(RebelCartJogTest, EntryRefusedWhileLatched)
{
  BeginCart();
  rebel->on_error(rclcpp_lifecycle::State());
  Refused("latched");
  EXPECT_EQ(server.CountCommands("MotionTypeCartBase"), 0u);
}

TEST_F(RebelCartJogTest, EntryRefusedWhenStatusIsStale)
{
  BeginCart();
  loop.reset();  // no read(): the 1 s latch is not reached while STATUS is silent
  server.SetSendStatus(false);
  std::this_thread::sleep_for(300ms);
  Refused("STATUS");
  server.SetSendStatus(true);
  EXPECT_EQ(server.CountCommands("MotionTypeCartBase"), 0u);
}

TEST_F(RebelCartJogTest, EntryRefusedWhileAJointMoveIsBusy)
{
  BeginCart();
  server.SetMoveSpeedDegS(10.0);
  const auto handle = SendAccepted(Goal({40.0, 0, 0, 0, 0, 0}));
  ASSERT_NE(handle, nullptr);
  ASSERT_TRUE(WaitMoving(1.0));
  Refused("joint move");
  client->Cancel(handle);
  client->Result(handle);
  EXPECT_EQ(server.CountCommands("MotionTypeCartBase"), 0u);
}

TEST_F(RebelCartJogTest, EntryRefusedWhileAMoveStopIsPending)
{
  Begin();
  MAKE_UNCONFIRMED_STOP();
  // Lifecycle recovery clears the latch; the unconfirmed stop stays pending.
  EXPECT_EQ(rebel->on_deactivate(rclcpp_lifecycle::State()), CallbackReturn::SUCCESS);
  EXPECT_EQ(rebel->on_activate(rclcpp_lifecycle::State()), CallbackReturn::SUCCESS);
  ASSERT_FALSE(rebel->HardwareErrorLatched());
  Refused("Move Stop");
  EXPECT_EQ(server.CountCommands("MotionTypeCartBase"), 0u);
  server.SetIgnoreMoveStop(false);
  ASSERT_TRUE(WaitFor([&]() {return !server.MoveActive();}, 1000ms));
}

TEST_F(RebelCartJogTest, EntryRefusedWhileAControllerClaimsAnInterfaceThenWorksOnceReleased)
{
  BeginCart();
  const auto joint = [&](int i) {return rebel->JointNames()[i] + "/velocity";};
  const std::vector<std::string> all = {joint(0), joint(1), joint(2), joint(3), joint(4), joint(5)};

  // A claim that is prepared but not yet performed already blocks (the controller is about to
  // take the interfaces).
  EXPECT_EQ(rebel->prepare_command_mode_switch(all, {}), return_type::OK);
  Refused("claims");
  EXPECT_EQ(rebel->perform_command_mode_switch(all, {}), return_type::OK);
  Refused("claims");
  // One interface still claimed is enough.
  EXPECT_EQ(rebel->perform_command_mode_switch({}, {joint(0), joint(1), joint(2), joint(3), joint(4)}),
    return_type::OK);
  Refused("claims");
  EXPECT_EQ(server.CountCommands("MotionTypeCartBase"), 0u);

  // The operator deactivates the controller: entering works.
  EXPECT_EQ(rebel->perform_command_mode_switch({}, {joint(5)}), return_type::OK);
  EnterMode();
}

TEST_F(RebelCartJogTest, EntrySequenceKeepsJogZeroAndOnlyThenForwardsTheTopic)
{
  BeginCart();
  const auto t0 = Clock::now();
  cart->StartStream({0.02, 0.0, 0.0, 0.0, 0.0, 0.0});  // arrives before the mode is on: ignored
  std::this_thread::sleep_for(150ms);
  EXPECT_TRUE(AllJogsZero(server, t0));
  EXPECT_EQ(server.CountCommands("MotionTypeCartBase"), 0u);

  const auto cartReply = cart->Enable(true);
  ASSERT_TRUE(cartReply.answered);
  ASSERT_TRUE(cartReply.success) << cartReply.message;
  const auto entered = Clock::now();
  EXPECT_TRUE(server.ModeIsCartBase());
  EXPECT_TRUE(rebel->CartJogActive());
  // Start-up sequence, then MotionTypeCartBase, nothing else.
  auto commands = server.Commands(1);
  ASSERT_EQ(commands.size(), kStartupSequence.size() + 1);
  EXPECT_EQ(commands.back(), "MotionTypeCartBase");
  // Jog was zero up to the MotionTypeCartBase command (nothing Cartesian went out before it).
  EXPECT_TRUE(AllZero(JogBeforeCommand(server, "MotionTypeCartBase")));
  {
    const auto cutoff = server.FirstCommandAfter("MotionTypeCartBase", t0);
    ASSERT_NE(cutoff, Clock::time_point{});
    for (const auto & frame : server.AliveJogs(t0)) {
      if (frame.stamp < cutoff) {
        EXPECT_TRUE(JogNear(ParseJog(frame.body), {0, 0, 0, 0, 0, 0})) << frame.body;
      }
    }
  }
  // The streamed command now goes out as Cartesian jog: 0.02 m/s * 200 %/(m/s) = 4 % in slot 1.
  EXPECT_TRUE(WaitFor([&]() {return SawJog(server, entered, {4.0, 0, 0, 0, 0, 0});}, 1000ms));
  // The mode is reported on the telemetry topic.
  TelemetryListener telemetry;
  EXPECT_TRUE(
    telemetry.WaitState(
      [](const igus_rebel_msgs::msg::HardwareState & m) {
        return m.cart_jog_active && m.cart_jog_state == "active";
      }, 2000ms));
  // Calling it again is a no-op.
  EXPECT_TRUE(cart->Enable(true).success);
  EXPECT_EQ(server.CountCommands("MotionTypeCartBase"), 1u);
}

TEST_F(RebelCartJogTest, DefaultScalingClampRotationDisabledAndFrameRejection)
{
  BeginCart();
  EnterMode();
  // Defaults: 200 %/(m/s), clamp 10 %, rotation off. z = 0.5 m/s -> 100 % -> clamped to 10.
  auto t = Clock::now();
  cart->StartStream({0.02, -0.01, 0.5, 0.3, -0.3, 0.3});
  EXPECT_TRUE(WaitFor([&]() {return SawJog(server, t, {4.0, -2.0, 10.0, 0, 0, 0});}, 1000ms));
  // Non-base frame: rejected, jog goes (and stays) zero.
  cart->StartStream({0.02, 0.0, 0.0, 0.0, 0.0, 0.0}, "tool0");
  t = Clock::now();
  std::this_thread::sleep_for(150ms);
  const auto afterReject = Clock::now();
  std::this_thread::sleep_for(300ms);
  EXPECT_TRUE(AllJogsZero(server, afterReject));
  // An empty frame_id is accepted.
  cart->StartStream({0.01, 0.0, 0.0, 0.0, 0.0, 0.0}, "");
  t = Clock::now();
  EXPECT_TRUE(WaitFor([&]() {return SawJog(server, t, {2.0, 0, 0, 0, 0, 0});}, 1000ms));
  // The joint velocity commands are ignored in the mode (a controller loop would write them).
  loop->SetVelocity(0.3);
  t = Clock::now();
  std::this_thread::sleep_for(300ms);
  for (const auto & frame : server.AliveJogs(t)) {
    EXPECT_TRUE(JogNear(ParseJog(frame.body), {2.0, 0, 0, 0, 0, 0})) << frame.body;
  }
}

TEST_F(RebelCartJogTest, ParametersSetScalesSignsRotationAndClamp)
{
  BeginCart(
    {{"cart_jog_percent_per_mps", "100"}, {"cart_jog_percent_per_radps", "50"},
      {"cart_jog_max_percent", "15"}, {"cart_jog_allow_rotation", "true"},
      {"cart_jog_signs", "-1,1,-1,1,1,-1"}});
  EnterMode();
  auto t = Clock::now();
  cart->StartStream({0.1, -0.1, 0.1, 0.2, -0.2, 0.2});
  EXPECT_TRUE(WaitFor([&]() {return SawJog(server, t, {-10, -10, -10, 10, -10, -10});}, 1000ms));
  cart->SetStream({1.0, 1.0, 0.0, 1.0, 0.0, 0.0});  // clamp at +-15 %, sign applied before it
  t = Clock::now();
  EXPECT_TRUE(WaitFor([&]() {return SawJog(server, t, {-15, 15, 0, 15, 0, 0});}, 1000ms));
}

TEST_F(RebelCartJogTest, InvalidCartJogParametersAreRejectedAtInit)
{
  for (const auto & bad : std::vector<std::pair<std::string, std::string>>{
    {"cart_jog_signs", "1,1,2,1,1,1"}, {"cart_jog_signs", "1,1,1"}, {"cart_jog_max_percent", "0"},
    {"cart_jog_max_percent", "101"}, {"cart_jog_percent_per_mps", "-5"},
    {"cart_jog_percent_per_radps", "nan"}, {"cart_jog_timeout_s", "0.001"},
    {"cart_jog_timeout_s", "5"}, {"cart_jog_allow_rotation", "maybe"},
    {"cart_jog_base_frame", "bad frame"}})
  {
    Igus::Rebel bad_rebel("127.0.0.1", 1);
    auto info = MakeInfo();
    info.hardware_parameters[bad.first] = bad.second;
    const auto start = Clock::now();
    EXPECT_EQ(bad_rebel.on_init(info), CallbackReturn::ERROR) << bad.first << "=" << bad.second;
    EXPECT_LT(Clock::now() - start, 500ms) << "must fail before connecting";
  }
}

TEST_F(RebelCartJogTest, WatchdogZeroesJogWithoutCommandsAndThereIsNoStutterWhileStreaming)
{
  BeginCart();
  EnterMode();
  const auto t = Clock::now();
  cart->StartStream({0.02, 0.0, 0.0, 0.0, 0.0, 0.0});
  ASSERT_TRUE(WaitFor([&]() {return SawJog(server, t, {4.0, 0, 0, 0, 0, 0});}, 1000ms));
  // 50 Hz stream, 0.15 s watchdog: no zero frame while commands keep coming.
  const auto streamingFrom = Clock::now();
  std::this_thread::sleep_for(600ms);
  std::size_t frames = 0;
  for (const auto & frame : server.AliveJogs(streamingFrom)) {
    ++frames;
    EXPECT_TRUE(JogNear(ParseJog(frame.body), {4.0, 0, 0, 0, 0, 0})) << frame.body;
  }
  EXPECT_GT(frames, 40u);
  // The topic goes silent: zero within the watchdog (+ ALIVEJOG period and delivery margin).
  cart->StopStream();
  const auto silentFrom = Clock::now();
  std::this_thread::sleep_for(500ms);
  EXPECT_TRUE(AllJogsZero(server, silentFrom + 230ms));
  EXPECT_TRUE(WaitFor([&]() {return SawJog(server, silentFrom - 100ms, {4.0, 0, 0, 0, 0, 0});}, 10ms));
  // The mode stays on; a new command moves again.
  EXPECT_TRUE(rebel->CartJogActive());
  const auto again = Clock::now();
  cart->StartStream({0.0, 0.01, 0.0, 0.0, 0.0, 0.0});
  EXPECT_TRUE(WaitFor([&]() {return SawJog(server, again, {0, 2.0, 0, 0, 0, 0});}, 1000ms));
  // A zero twist (button released) zeroes immediately.
  cart->SetStream({0, 0, 0, 0, 0, 0});
  const auto zeroAt = Clock::now();
  std::this_thread::sleep_for(150ms);
  EXPECT_TRUE(AllJogsZero(server, zeroAt + 60ms));
}

TEST_F(RebelCartJogTest, LeavingByServiceZeroesJogThenConfirmsMotionTypeJointAndJointJogResumes)
{
  BeginCart();
  EnterMode();
  cart->StartStream({0.02, 0.0, 0.0, 0.0, 0.0, 0.0});
  ASSERT_TRUE(WaitFor([&]() {return SawJog(server, Clock::now() - 60ms, {4.0, 0, 0, 0, 0, 0});}, 1000ms));

  const auto leaving = Clock::now();
  const auto reply = cart->Enable(false);
  const auto answered = Clock::now();
  ASSERT_TRUE(reply.answered);
  EXPECT_TRUE(reply.success) << reply.message;
  EXPECT_FALSE(server.ModeIsCartBase());
  EXPECT_EQ(rebel->CartJogStateValue(), 0);
  // The topic is still being published: it must be ignored now. Jog is zero from the request on.
  const int cartIndex = CommandIndex(server, "MotionTypeCartBase");
  const int jointIndex = CommandIndex(server, "MotionTypeJoint", 1, 1);  // 0 = start-up sequence
  ASSERT_GE(cartIndex, 0);
  ASSERT_GT(jointIndex, cartIndex);
  EXPECT_EQ(server.Commands(1).back(), "MotionTypeJoint");
  // Zero jog first, then MotionTypeJoint: the ALIVEJOG right before the CMD is all zero.
  EXPECT_TRUE(AllZero(JogBeforeCommand(server, "MotionTypeJoint", 1)));
  std::this_thread::sleep_for(200ms);
  EXPECT_TRUE(AllJogsZero(server, answered + 30ms));
  (void)leaving;
  // Joint jog works again.
  loop->SetVelocity(0.1);
  EXPECT_TRUE(WaitFor([&]() {return AnyNonZeroJog(server, Clock::now() - 60ms);}, 1000ms));
  // Leaving twice is fine; re-entering needs a new call.
  std::string message;
  EXPECT_TRUE(rebel->SetCartJogEnabled(false, message)) << message;
}

TEST_F(RebelCartJogTest, ClaimIsRefusedAndMoveJointRejectedWhileInTheMode)
{
  BeginCart();
  EnterMode();
  const auto joint = [&](int i) {return rebel->JointNames()[i] + "/velocity";};
  const std::vector<std::string> all = {joint(0), joint(1), joint(2), joint(3), joint(4), joint(5)};
  EXPECT_EQ(rebel->prepare_command_mode_switch(all, {}), return_type::ERROR);
  EXPECT_EQ(rebel->prepare_command_mode_switch({"unrelated/position"}, {}), return_type::OK);
  EXPECT_TRUE(rebel->CartJogActive());

  const std::size_t movesBefore = server.MoveCommands().size();
  const auto handle = client->Send(Goal({4.0, 0, 0, 0, 0, 0}));
  EXPECT_EQ(handle, nullptr) << "a move_joint goal must be rejected while the mode is on";
  EXPECT_EQ(server.MoveCommands().size(), movesBefore);
  EXPECT_FALSE(rebel->MoveGoalBusy());

  // The mode survives these refusals.
  EXPECT_TRUE(rebel->CartJogActive());
  // reset_and_enable would re-send MotionTypeJoint: refused while the mode is on.
  std::string message;
  EXPECT_FALSE(rebel->ResetAndEnable(message));
  EXPECT_NE(message.find("Cartesian"), std::string::npos) << message;
}

TEST_F(RebelCartJogTest, ClaimThatCannotBeRefusedLeavesTheModeAndLatches)
{
  BeginCart();
  EnterMode();
  cart->StartStream({0.02, 0.0, 0.0, 0.0, 0.0, 0.0});
  ASSERT_TRUE(WaitFor([&]() {return SawJog(server, Clock::now() - 60ms, {4.0, 0, 0, 0, 0, 0});}, 1000ms));
  const auto joint = [&](int i) {return rebel->JointNames()[i] + "/velocity";};
  const auto at = Clock::now();
  // Humble activates the controller even when prepare refused.
  rebel->perform_command_mode_switch(
    {joint(0), joint(1), joint(2), joint(3), joint(4), joint(5)}, {});
  EXPECT_TRUE(rebel->HardwareErrorLatched());
  ASSERT_TRUE(WaitFor([&]() {return rebel->CartJogStateValue() == 0;}, 3000ms));
  EXPECT_FALSE(server.ModeIsCartBase());
  EXPECT_TRUE(AllZero(JogBeforeCommand(server, "MotionTypeJoint", 1)));
  std::this_thread::sleep_for(100ms);
  EXPECT_TRUE(AllJogsZero(server, at + 30ms));
  EXPECT_EQ(server.Commands(1).back(), "MotionTypeJoint");
}

TEST_F(RebelCartJogTest, LeavingByLatchZeroesJogAndRestoresJointMode)
{
  BeginCart();
  EnterMode();
  cart->StartStream({0.02, 0.0, 0.0, 0.0, 0.0, 0.0});
  ASSERT_TRUE(WaitFor([&]() {return SawJog(server, Clock::now() - 60ms, {4.0, 0, 0, 0, 0, 0});}, 1000ms));
  const auto at = Clock::now();
  rebel->on_error(rclcpp_lifecycle::State());  // latches
  ASSERT_TRUE(rebel->HardwareErrorLatched());
  EXPECT_NE(rebel->CartJogStateValue(), 2);
  ASSERT_TRUE(WaitFor([&]() {return rebel->CartJogStateValue() == 0;}, 3000ms));
  EXPECT_FALSE(server.ModeIsCartBase());
  EXPECT_EQ(server.Commands(1).back(), "MotionTypeJoint");
  EXPECT_TRUE(AllZero(JogBeforeCommand(server, "MotionTypeJoint", 1)));
  std::this_thread::sleep_for(100ms);
  // The stream goes on, but nothing is forwarded; jog stayed zero from the latch on.
  EXPECT_TRUE(AllJogsZero(server, at + 30ms));
  // Re-entering is refused while latched.
  Refused("latched");
}

TEST_F(RebelCartJogTest, LeavingByReconnectResendsTheStartupSequenceAndStaysOff)
{
  BeginCart();
  EnterMode();
  cart->StartStream({0.02, 0.0, 0.0, 0.0, 0.0, 0.0});
  ASSERT_TRUE(WaitFor([&]() {return SawJog(server, Clock::now() - 60ms, {4.0, 0, 0, 0, 0, 0});}, 1000ms));
  ASSERT_TRUE(server.ModeIsCartBase());
  const auto at = Clock::now();
  server.DropClient();
  // The fake numbers every accepted TCP connection; use the newest one that carries the start-up
  // sequence (a stray extra accept of the fake must not matter for this test).
  std::size_t second = 0;
  ASSERT_TRUE(
    WaitFor(
      [&]() {
        for (std::size_t c = server.Connections(); c >= 2; --c) {
          if (server.Commands(c).size() >= 6) {
            second = c;
            return true;
          }
        }
        return false;
      }, 3000ms))
    << "connections " << server.Connections() << " all " << server.Commands().size();
  EXPECT_EQ(server.Commands(second), kStartupSequence);  // includes MotionTypeJoint, as today
  ASSERT_TRUE(WaitFor([&]() {return rebel->CartJogStateValue() == 0;}, 1000ms));
  EXPECT_FALSE(server.ModeIsCartBase());
  std::this_thread::sleep_for(200ms);
  // Nothing non-zero on the new connection although the topic keeps streaming.
  for (const auto & frame : server.AliveJogs(at)) {
    if (frame.connection == second) {
      EXPECT_TRUE(JogNear(ParseJog(frame.body), {0, 0, 0, 0, 0, 0})) << frame.body;
    }
  }
  EXPECT_TRUE(WaitFor([&]() {return rebel->HardwareErrorLatched();}, 2000ms));
}

TEST_F(RebelCartJogTest, LeavingByDeactivateZeroesJogAndRestoresJointMode)
{
  BeginCart();
  EnterMode();
  cart->StartStream({0.02, 0.0, 0.0, 0.0, 0.0, 0.0});
  ASSERT_TRUE(WaitFor([&]() {return SawJog(server, Clock::now() - 60ms, {4.0, 0, 0, 0, 0, 0});}, 1000ms));
  const auto at = Clock::now();
  EXPECT_EQ(rebel->on_deactivate(rclcpp_lifecycle::State()), CallbackReturn::SUCCESS);
  ASSERT_TRUE(WaitFor([&]() {return rebel->CartJogStateValue() == 0;}, 3000ms));
  EXPECT_FALSE(server.ModeIsCartBase());
  EXPECT_EQ(server.Commands(1).back(), "MotionTypeJoint");
  EXPECT_TRUE(AllZero(JogBeforeCommand(server, "MotionTypeJoint", 1)));
  std::this_thread::sleep_for(100ms);
  EXPECT_TRUE(AllJogsZero(server, at + 30ms));
  Refused("not active");
}

TEST_F(RebelCartJogTest, EntryFailsCleanlyWhenTheControllerRejectsOrDoesNotFollow)
{
  BeginCart();
  // CMDERROR: fails at once, joint mode untouched (MotionTypeJoint restores it anyway).
  server.SetCartBaseError(true);
  std::string message;
  auto start = Clock::now();
  EXPECT_FALSE(rebel->SetCartJogEnabled(true, message));
  EXPECT_LT(Clock::now() - start, 1500ms);
  EXPECT_NE(message.find("rejected"), std::string::npos) << message;
  EXPECT_EQ(rebel->CartJogStateValue(), 0);
  EXPECT_FALSE(server.ModeIsCartBase());
  EXPECT_FALSE(rebel->HardwareErrorLatched());
  server.SetCartBaseError(false);

  // CMDACK but the STATUS mode never becomes cartbase: not confirmed, joint mode re-asserted.
  server.SetIgnoreCartBase(true);
  EXPECT_FALSE(rebel->SetCartJogEnabled(true, message));
  EXPECT_NE(message.find("not confirmed"), std::string::npos) << message;
  EXPECT_EQ(rebel->CartJogStateValue(), 0);
  EXPECT_EQ(server.Commands(1).back(), "MotionTypeJoint");
  server.SetIgnoreCartBase(false);

  // The mode changes but no CMDACK arrives: the entry is abandoned and the mode undone.
  server.SetSuppressCartBaseAck(true);
  EXPECT_FALSE(rebel->SetCartJogEnabled(true, message));
  EXPECT_NE(message.find("not confirmed"), std::string::npos) << message;
  EXPECT_NE(message.find("joint mode"), std::string::npos) << message;
  EXPECT_FALSE(server.ModeIsCartBase());
  EXPECT_EQ(rebel->CartJogStateValue(), 0);
  EXPECT_FALSE(rebel->HardwareErrorLatched());
  EXPECT_TRUE(AllJogsZero(server, {}));
  // A clean entry still works afterwards.
  server.SetSuppressCartBaseAck(false);
  EnterMode();
}

TEST_F(RebelCartJogTest, UnconfirmedMotionTypeJointLatches)
{
  BeginCart();
  EnterMode();
  server.SetSuppressJointAck(true);
  std::string message;
  const auto start = Clock::now();
  EXPECT_FALSE(rebel->SetCartJogEnabled(false, message));
  EXPECT_GE(Clock::now() - start, 1800ms);
  EXPECT_NE(message.find("NOT confirmed"), std::string::npos) << message;
  EXPECT_TRUE(rebel->HardwareErrorLatched());
  EXPECT_NE(rebel->CartJogStateValue(), 2);
  EXPECT_TRUE(AllJogsZero(server, start));
  // Recovery: the lifecycle reset re-sends the start-up sequence (MotionTypeJoint) and ends the mode.
  server.SetSuppressJointAck(false);
  EXPECT_EQ(rebel->on_deactivate(rclcpp_lifecycle::State()), CallbackReturn::SUCCESS);
  EXPECT_EQ(rebel->on_activate(rclcpp_lifecycle::State()), CallbackReturn::SUCCESS);
  EXPECT_EQ(rebel->CartJogStateValue(), 0);
  EXPECT_FALSE(rebel->HardwareErrorLatched());
}

TEST_F(RebelCartJogTest, StatusCoastSendsZeroCartesianJogAndAHoldDuringMotionLatches)
{
  BeginCart();
  EnterMode();
  cart->StartStream({0.02, 0.0, 0.0, 0.0, 0.0, 0.0});
  ASSERT_TRUE(WaitFor([&]() {return SawJog(server, Clock::now() - 60ms, {4.0, 0, 0, 0, 0, 0});}, 1000ms));

  // Short STATUS gap (coast tier): no coasting in Cartesian mode -- zero, whatever the stream says.
  server.SetSendStatus(false);
  const auto pauseStart = Clock::now();
  ASSERT_TRUE(WaitFor([&]() {return rebel->StatusCoastActive();}, 1000ms));
  const auto coastStart = Clock::now();
  cart->SetStream({0.04, 0.0, 0.0, 0.0, 0.0, 0.0});
  std::this_thread::sleep_for(pauseStart + 300ms - Clock::now());
  server.SetSendStatus(true);
  ASSERT_TRUE(WaitFor([&]() {return !rebel->StatusCoastActive();}, 1000ms));
  const auto resumed = Clock::now();
  EXPECT_FALSE(rebel->HardwareErrorLatched());
  std::size_t checked = 0;
  for (const auto & frame : server.AliveJogs(coastStart + 30ms)) {
    if (frame.stamp > resumed - 20ms) {continue;}
    ++checked;
    EXPECT_TRUE(JogNear(ParseJog(frame.body), {0, 0, 0, 0, 0, 0})) << frame.body;
  }
  EXPECT_GE(checked, 3u);
  EXPECT_TRUE(WaitFor([&]() {return SawJog(server, resumed + 30ms, {8.0, 0, 0, 0, 0, 0});}, 1000ms));

  // Long gap (hold tier) while moving: zero jog, and the resume latches (no auto-resume).
  server.SetSendStatus(false);
  const auto holdPause = Clock::now();
  ASSERT_TRUE(WaitFor([&]() {return rebel->StatusHoldActive();}, 1000ms));
  const auto holdStart = Clock::now();
  std::this_thread::sleep_for(holdPause + 600ms - Clock::now());
  server.SetSendStatus(true);
  ASSERT_TRUE(WaitFor([&]() {return rebel->HardwareErrorLatched();}, 1000ms));
  for (const auto & frame : server.AliveJogs(holdStart + 30ms)) {
    EXPECT_TRUE(JogNear(ParseJog(frame.body), {0, 0, 0, 0, 0, 0})) << frame.body;
  }
  ASSERT_TRUE(WaitFor([&]() {return rebel->CartJogStateValue() == 0;}, 3000ms));
  EXPECT_FALSE(server.ModeIsCartBase());
}

// ---- STATUS mode cross-check, idle auto-exit, enter/deactivate ----
TEST_F(RebelCartJogTest, UnpromptedSwitchToJointWhileActiveZeroesJogAndLeavesTheMode)
{
  BeginCart();
  EnterMode();
  cart->StartStream({0.02, 0.0, 0.0, 0.0, 0.0, 0.0});
  ASSERT_TRUE(WaitFor([&]() {return SawJog(server, Clock::now() - 60ms, {4.0, 0, 0, 0, 0, 0});}, 1000ms));
  const auto at = Clock::now();
  server.ForceMode(0);  // someone else switched the robot to joint mode
  ASSERT_TRUE(WaitFor([&]() {return rebel->CartJogStateValue() != 2;}, 1000ms));
  // Zero jog from the first frame that saw the new mode (STATUS 20 ms + one ALIVEJOG period).
  std::this_thread::sleep_for(150ms);
  EXPECT_TRUE(AllJogsZero(server, at + 80ms));
  ASSERT_TRUE(WaitFor([&]() {return rebel->CartJogStateValue() == 0;}, 3000ms));
  EXPECT_EQ(server.CountCommands("MotionTypeJoint", at), 1u);  // restored through the exit path
  EXPECT_FALSE(rebel->HardwareErrorLatched());
  EXPECT_TRUE(AllJogsZero(server, at + 80ms));
}

TEST_F(RebelCartJogTest, RobotStaysCartbaseAfterRecoveryLatchesAndJointJogStaysZero)
{
  BeginCart();
  EnterMode();
  server.SetIgnoreMotionTypeJoint(true);  // MotionTypeJoint is acknowledged but does not take effect
  rebel->on_error(rclcpp_lifecycle::State());
  ASSERT_TRUE(rebel->HardwareErrorLatched());
  loop.reset();
  // Operator recovery: inactive -> active re-sends the start-up sequence (MotionTypeJoint ignored).
  EXPECT_EQ(rebel->on_deactivate(rclcpp_lifecycle::State()), CallbackReturn::SUCCESS);
  EXPECT_EQ(rebel->on_activate(rclcpp_lifecycle::State()), CallbackReturn::SUCCESS);
  EXPECT_FALSE(rebel->HardwareErrorLatched());
  EXPECT_TRUE(server.ModeIsCartBase());
  const auto at = Clock::now();
  StartLoop(0.1);  // the recovered JTC commands joint velocity: would be read as Cartesian speed
  ASSERT_TRUE(WaitFor([&]() {return rebel->HardwareErrorLatched();}, 2000ms));
  std::this_thread::sleep_for(150ms);
  EXPECT_TRUE(AllJogsZero(server, at + 30ms));
  Refused("latched");
}

TEST_F(RebelCartJogTest, OffWhileRobotReportsCartbaseLatchesAfterTheGraceAndKeepsJogZero)
{
  BeginCart();
  loop->SetVelocity(0.1);
  ASSERT_TRUE(WaitFor([&]() {return AnyNonZeroJog(server, Clock::now() - 60ms);}, 1000ms));
  const auto at = Clock::now();
  server.ForceMode(1);
  ASSERT_TRUE(WaitFor([&]() {return rebel->HardwareErrorLatched();}, 2000ms));
  EXPECT_GE(Clock::now() - at, 250ms);  // not before the grace period
  std::this_thread::sleep_for(100ms);
  EXPECT_TRUE(AllJogsZero(server, at + 60ms));
}

TEST_F(RebelCartJogTest, EntryRefusedWhenTheRobotModeIsNotJoint)
{
  BeginCart();
  server.ForceMode(2);  // fsm: not a mismatch for the guard, but not joint either
  std::this_thread::sleep_for(100ms);
  Refused("not 'joint'");
  EXPECT_EQ(server.CountCommands("MotionTypeCartBase"), 0u);
  EXPECT_FALSE(rebel->HardwareErrorLatched());
  server.ForceMode(0);
  std::this_thread::sleep_for(100ms);
  EnterMode();
}

TEST_F(RebelCartJogTest, IdleAutoExitLeavesTheModeWhenNoCommandArrives)
{
  BeginCart({{"cart_jog_idle_exit_s", "0.8"}});
  EnterMode();
  const auto at = Clock::now();
  cart->StartStream({0.02, 0.0, 0.0, 0.0, 0.0, 0.0});
  std::this_thread::sleep_for(1200ms);
  EXPECT_TRUE(rebel->CartJogActive()) << "commands keep the mode alive";
  cart->StopStream();
  const auto silent = Clock::now();
  ASSERT_TRUE(WaitFor([&]() {return rebel->CartJogStateValue() == 0;}, 4000ms));
  EXPECT_GE(Clock::now() - silent, 600ms);
  EXPECT_FALSE(server.ModeIsCartBase());
  EXPECT_FALSE(rebel->HardwareErrorLatched());
  EXPECT_EQ(server.Commands(1).back(), "MotionTypeJoint");
  EXPECT_TRUE(AllZero(JogBeforeCommand(server, "MotionTypeJoint", 1)));
  (void)at;
}

TEST_F(RebelCartJogTest, IdleAutoExitCountsFromEntryWhenNoCommandEverArrives)
{
  BeginCart({{"cart_jog_idle_exit_s", "0.6"}});
  EnterMode();
  ASSERT_TRUE(WaitFor([&]() {return rebel->CartJogStateValue() == 0;}, 4000ms));
  EXPECT_FALSE(server.ModeIsCartBase());
}

TEST_F(RebelCartJogTest, IdleAutoExitCanBeDisabledAndInvalidValuesAreRejected)
{
  BeginCart({{"cart_jog_idle_exit_s", "0"}});
  EnterMode();
  std::this_thread::sleep_for(1200ms);
  EXPECT_TRUE(rebel->CartJogActive());
  for (const char * bad : {"-1", "abc", "4000"}) {
    Igus::Rebel bad_rebel("127.0.0.1", 1);
    auto info = MakeInfo();
    info.hardware_parameters["cart_jog_idle_exit_s"] = bad;
    EXPECT_EQ(bad_rebel.on_init(info), CallbackReturn::ERROR) << bad;
  }
}

TEST_F(RebelCartJogTest, DeactivateWhileEnteringAbortsAndRestoresJointMode)
{
  BeginCart();
  server.SetSuppressCartBaseAck(true);  // the entry waits for the (missing) CMDACK
  std::string message;
  std::future<bool> entering = std::async(
    std::launch::async, [&]() {return rebel->SetCartJogEnabled(true, message);});
  ASSERT_TRUE(WaitFor([&]() {return rebel->CartJogStateValue() == 1;}, 1000ms));
  const auto start = Clock::now();
  EXPECT_EQ(rebel->on_deactivate(rclcpp_lifecycle::State()), CallbackReturn::SUCCESS);
  ASSERT_EQ(entering.wait_for(5s), std::future_status::ready);
  EXPECT_FALSE(entering.get());
  EXPECT_LT(Clock::now() - start, 1500ms) << "must not wait for the 2 s confirmation timeout";
  EXPECT_NE(message.find("deactivate"), std::string::npos) << message;
  ASSERT_TRUE(WaitFor([&]() {return rebel->CartJogStateValue() == 0;}, 3000ms));
  EXPECT_FALSE(server.ModeIsCartBase());
}

// ---- prepared ALIVEJOG frames cannot cross a reconnect / latch; coast; unknown mode; stamps ----
namespace
{
// Holds the ALIVEJOG thread right before its first non-zero send until Release().
class AliveHold
{
public:
  explicit AliveHold(Igus::Rebel & driver)
  : rebel_(driver)
  {
    rebel_.SetBeforeAliveSendHookForTest(
      [this]() {
        std::unique_lock<std::mutex> lock(mutex_);
        if (done_) {return;}
        entered_ = true;
        condition_.notify_all();
        condition_.wait(lock, [this]() {return release_;});
        done_ = true;
      });
  }
  ~AliveHold()
  {
    Release();
    rebel_.SetBeforeAliveSendHookForTest(nullptr);
  }
  bool WaitEntered(std::chrono::milliseconds timeout)
  {
    std::unique_lock<std::mutex> lock(mutex_);
    return condition_.wait_for(lock, timeout, [this]() {return entered_;});
  }
  void Release()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    release_ = true;
    condition_.notify_all();
  }

private:
  Igus::Rebel & rebel_;
  std::mutex mutex_;
  std::condition_variable condition_;
  bool entered_ = false;
  bool release_ = false;
  bool done_ = false;
};

// Newest connection that carries the start-up sequence (>= 2), waiting up to `timeout`.
std::size_t WaitSecondConnection(FakeCriServer & server, std::chrono::milliseconds timeout)
{
  std::size_t found = 0;
  WaitFor(
    [&]() {
      for (std::size_t c = server.Connections(); c >= 2; --c) {
        if (server.Commands(c).size() >= 6) {
          found = c;
          return true;
        }
      }
      return false;
    }, timeout);
  return found;
}
}  // namespace

TEST_F(RebelCartJogTest, PreparedCartesianFrameCannotCrossAReconnect)
{
  BeginCart();
  EnterMode();
  AliveHold hold(*rebel);
  cart->StartStream({0.02, 0.0, 0.0, 0.0, 0.0, 0.0});
  ASSERT_TRUE(hold.WaitEntered(2000ms));  // a [4,0,0,0,0,0] frame is built and waits
  server.DropClient();
  const std::size_t second = WaitSecondConnection(server, 3000ms);
  ASSERT_GE(second, 2u) << "no reconnect";
  EXPECT_EQ(server.Commands(second), kStartupSequence);
  hold.Release();
  std::this_thread::sleep_for(300ms);
  // Nothing non-zero reaches the new connection, not even the one stale frame.
  std::size_t frames = 0;
  for (const auto & frame : server.AliveJogs()) {
    if (frame.connection >= second) {
      ++frames;
      EXPECT_TRUE(JogNear(ParseJog(frame.body), {0, 0, 0, 0, 0, 0})) << frame.body;
    }
  }
  EXPECT_GT(frames, 5u);
}

TEST_F(RebelCartJogTest, PreparedJointFrameCannotCrossAReconnect)
{
  BeginCart();
  loop->SetVelocity(0.1);
  ASSERT_TRUE(WaitFor([&]() {return AnyNonZeroJog(server, Clock::now() - 60ms);}, 1000ms));
  AliveHold hold(*rebel);
  ASSERT_TRUE(hold.WaitEntered(2000ms));
  // No read() any more (it would latch on the reconnect): only the connection check in the socket
  // send can stop the prepared frame -- state, latch and hold flags are all unchanged.
  loop->Stop();
  server.DropClient();
  const std::size_t second = WaitSecondConnection(server, 3000ms);
  ASSERT_GE(second, 2u);
  EXPECT_FALSE(rebel->HardwareErrorLatched());
  hold.Release();
  std::this_thread::sleep_for(300ms);
  for (const auto & frame : server.AliveJogs()) {
    if (frame.connection >= second) {
      EXPECT_TRUE(JogNear(ParseJog(frame.body), {0, 0, 0, 0, 0, 0})) << frame.body;
    }
  }
}

TEST_F(RebelCartJogTest, PreparedCartesianFrameIsDroppedWhenALatchLandsBeforeTheSend)
{
  BeginCart();
  EnterMode();
  AliveHold hold(*rebel);
  cart->StartStream({0.02, 0.0, 0.0, 0.0, 0.0, 0.0});
  ASSERT_TRUE(hold.WaitEntered(2000ms));
  rebel->on_error(rclcpp_lifecycle::State());  // latch + exit request
  const auto latched = Clock::now();
  hold.Release();
  ASSERT_TRUE(WaitFor([&]() {return rebel->CartJogStateValue() == 0;}, 3000ms));
  EXPECT_TRUE(AllJogsZero(server, latched));
}

TEST_F(RebelCartJogTest, UnknownModeWhileActiveFailsClosedAndLeavesTheMode)
{
  BeginCart();
  EnterMode();
  cart->StartStream({0.02, 0.0, 0.0, 0.0, 0.0, 0.0});
  ASSERT_TRUE(WaitFor([&]() {return SawJog(server, Clock::now() - 60ms, {4.0, 0, 0, 0, 0, 0});}, 1000ms));
  server.SetSendStatus(false);       // freeze STATUS so the seam below is not overwritten
  std::this_thread::sleep_for(40ms);
  rebel->SetStatusModeForTest(static_cast<int>(Igus::CriMessages::Mode::UNKNOWN));
  const auto at = Clock::now();
  ASSERT_TRUE(WaitFor([&]() {return rebel->CartJogStateValue() == 3;}, 120ms)) << "no exit request";
  std::this_thread::sleep_for(60ms);
  EXPECT_TRUE(AllJogsZero(server, at + 40ms));
  server.SetSendStatus(true);        // STATUS is back (joint after MotionTypeJoint): exit confirms
  ASSERT_TRUE(WaitFor([&]() {return rebel->CartJogStateValue() == 0;}, 3000ms));
  EXPECT_FALSE(server.ModeIsCartBase());
}

TEST_F(RebelCartJogTest, UnknownModeStringMakesStatusStaleSoJogIsZeroAndTheDriverLatches)
{
  BeginCart();
  EnterMode();
  cart->StartStream({0.02, 0.0, 0.0, 0.0, 0.0, 0.0});
  ASSERT_TRUE(WaitFor([&]() {return SawJog(server, Clock::now() - 60ms, {4.0, 0, 0, 0, 0, 0});}, 1000ms));
  const auto at = Clock::now();
  server.ForceMode(3);  // "weirdmode": the whole STATUS is rejected as malformed
  std::this_thread::sleep_for(400ms);
  EXPECT_TRUE(AllJogsZero(server, at + 300ms));
  ASSERT_TRUE(WaitFor([&]() {return rebel->HardwareErrorLatched();}, 3000ms));
  EXPECT_TRUE(AllJogsZero(server, at + 300ms));
}

TEST_F(RebelCartJogTest, CommandStampsAreChecked)
{
  BeginCart();
  const rclcpp::Time beforeEntry = cart->Now();
  EnterMode();
  const auto jogAt = [&](const rclcpp::Time & stamp, const std::string & frame = "base_link") {
      cart->PublishStamped({0.02, 0.0, 0.0, 0.0, 0.0, 0.0}, stamp, frame);
    };
  // Hammers one stamped command at 50 Hz for 300 ms and reports whether motion appeared.
  const auto moves = [&](const std::function<rclcpp::Time()> & stamp) {
      const auto from = Clock::now();
      for (int i = 0; i < 15; ++i) {
        jogAt(stamp());
        std::this_thread::sleep_for(20ms);
      }
      std::this_thread::sleep_for(60ms);
      return AnyNonZeroJog(server, from);
    };
  // Stamped before the mode entry but still inside the watchdog window (age < 0.15 s): only the
  // "before the current mode entry" rule can reject it.
  {
    const auto from = Clock::now();
    jogAt(beforeEntry);
    std::this_thread::sleep_for(100ms);
    EXPECT_FALSE(AnyNonZeroJog(server, from));
  }
  // Zero stamp, an old one (a replay after the watchdog expired), a far-future one: rejected.
  EXPECT_FALSE(moves([&]() {return rclcpp::Time(0, 0, RCL_ROS_TIME);}));
  EXPECT_FALSE(moves([&]() {return cart->Now() - rclcpp::Duration(1, 0);}));
  EXPECT_FALSE(moves([&]() {return cart->Now() + rclcpp::Duration(2, 0);}));
  EXPECT_TRUE(rebel->CartJogActive());
  // A fresh stamp is accepted, and the mode kept working.
  EXPECT_TRUE(moves([&]() {return cart->Now();}));
  // After the watchdog expired, one replayed old non-zero command does not restart motion.
  std::this_thread::sleep_for(400ms);
  const auto replay = Clock::now();
  jogAt(cart->Now() - rclcpp::Duration(0, 400000000));
  std::this_thread::sleep_for(150ms);
  EXPECT_FALSE(AnyNonZeroJog(server, replay));
}

TEST_F(RebelCartJogTest, TopicIsIgnoredAndNothingChangesWhileTheModeWasNeverEntered)
{
  BeginCart();
  const auto t = Clock::now();
  cart->StartStream({0.05, 0.05, 0.05, 0.5, 0.5, 0.5});
  std::this_thread::sleep_for(400ms);
  EXPECT_TRUE(AllJogsZero(server, t));
  EXPECT_EQ(server.CountCommands("MotionTypeCartBase"), 0u);
  EXPECT_EQ(server.Commands(1), kStartupSequence);
  // Joint jog behaves as before.
  loop->SetVelocity(0.1);
  EXPECT_TRUE(WaitFor([&]() {return AnyNonZeroJog(server, Clock::now() - 60ms);}, 1000ms));
}


// ======================================================================================
// Direct joint jog mode (CRI route): ALIVEJOG joint % from ~/joint_jog_cmd, robot stays in joint mode
// ======================================================================================
namespace
{
using JointJogMsg = control_msgs::msg::JointJog;

// Publishes ~/joint_jog_cmd and calls ~/joint_jog_enable of the driver node, from its own node.
class JointClient
{
public:
  explicit JointClient(const std::string & driverNode = "igus_rebel")
  {
    node_ = std::make_shared<rclcpp::Node>("joint_test_client_" + std::to_string(counter_++));
    pub_ = node_->create_publisher<JointJogMsg>("/" + driverNode + "/joint_jog_cmd", 10);
    client_ = node_->create_client<std_srvs::srv::SetBool>("/" + driverNode + "/joint_jog_enable");
    executor_.add_node(node_);
    thread_ = std::thread([this]() {executor_.spin(); finished_ = true;});
  }
  ~JointClient()
  {
    StopStream();
    while (!finished_) {
      executor_.cancel();
      std::this_thread::sleep_for(2ms);
    }
    if (thread_.joinable()) {
      thread_.join();
    }
  }
  bool WaitForService() {return client_->wait_for_service(5s);}
  CartClient::Reply Enable(bool enable)
  {
    auto request = std::make_shared<std_srvs::srv::SetBool::Request>();
    request->data = enable;
    auto future = client_->async_send_request(request);
    CartClient::Reply reply;
    if (future.wait_for(8s) == std::future_status::ready) {
      const auto response = future.get();
      reply = {true, response->success, response->message};
    }
    return reply;
  }
  rclcpp::Time Now() {return node_->now();}
  // Names default to joint1..joint6 in order with the given six velocities.
  static JointJogMsg Make(const Twist6 & v, const rclcpp::Time & stamp)
  {
    JointJogMsg msg;
    msg.header.stamp = stamp;
    msg.joint_names = {"joint1", "joint2", "joint3", "joint4", "joint5", "joint6"};
    msg.velocities.assign(v.begin(), v.end());
    return msg;
  }
  void PublishRaw(const JointJogMsg & msg) {pub_->publish(msg);}
  void Publish(const Twist6 & v) {pub_->publish(Make(v, node_->now()));}
  void PublishStamped(const Twist6 & v, const rclcpp::Time & stamp) {pub_->publish(Make(v, stamp));}
  // Re-publishes the (changeable) command at 50 Hz, like the keyboard timer.
  void StartStream(const Twist6 & v)
  {
    StopStream();
    {
      std::lock_guard<std::mutex> lock(streamMutex_);
      streamValue_ = v;
    }
    streaming_ = true;
    streamThread_ = std::thread(
      [this]() {
        while (streaming_) {
          Twist6 value;
          {
            std::lock_guard<std::mutex> lock(streamMutex_);
            value = streamValue_;
          }
          Publish(value);
          std::this_thread::sleep_for(20ms);
        }
      });
  }
  void SetStream(const Twist6 & v)
  {
    std::lock_guard<std::mutex> lock(streamMutex_);
    streamValue_ = v;
  }
  void StopStream()
  {
    streaming_ = false;
    if (streamThread_.joinable()) {
      streamThread_.join();
    }
  }

private:
  static inline std::atomic<int> counter_{0};
  std::atomic<bool> finished_{false};
  std::atomic<bool> streaming_{false};
  rclcpp::Node::SharedPtr node_;
  rclcpp::Publisher<JointJogMsg>::SharedPtr pub_;
  rclcpp::Client<std_srvs::srv::SetBool>::SharedPtr client_;
  rclcpp::executors::SingleThreadedExecutor executor_;
  std::thread thread_;
  std::thread streamThread_;
  std::mutex streamMutex_;
  Twist6 streamValue_{};
};

// jog [%] of the default jog_velocity_scale (2.0) for a joint velocity in rad/s.
double Pct(double radPerS) {return 2.0 * radPerS * 180.0 / M_PI;}

class RebelJointJogTest : public RebelCartJogTest
{
protected:
  void BeginJoint(
    const std::map<std::string, std::string> & hardwareParameters = {}, bool activate = true)
  {
    BeginCart(hardwareParameters, activate);
    if (HasFatalFailure()) {
      return;
    }
    joint = std::make_unique<JointClient>();
    ASSERT_TRUE(joint->WaitForService());
  }
  void TearDown() override
  {
    joint.reset();
    RebelCartJogTest::TearDown();
  }
  void EnterJoint()
  {
    std::string message;
    ASSERT_TRUE(rebel->SetJointJogEnabled(true, message)) << message;
    ASSERT_TRUE(rebel->JointJogActive());
  }
  bool JointRefused(const std::string & expectedText)
  {
    std::string message;
    const bool ok = rebel->SetJointJogEnabled(true, message);
    EXPECT_FALSE(ok) << message;
    EXPECT_NE(message.find(expectedText), std::string::npos) << message;
    return !ok;
  }
  // True if after a single (non-streamed) command the jog is non-zero for a short while.
  bool MovesAfterSingle(const JointJogMsg & msg)
  {
    const auto from = Clock::now();
    joint->PublishRaw(msg);
    std::this_thread::sleep_for(60ms);
    return AnyNonZeroJog(server, from);
  }

  std::unique_ptr<JointClient> joint;
};
}  // namespace

TEST_F(RebelJointJogTest, EntryRefusedWhenNotActivated)
{
  BeginJoint({}, false);
  JointRefused("not active");
  EXPECT_EQ(rebel->JointJogStateValue(), 0);
}

TEST_F(RebelJointJogTest, EntryRefusedWhileLatched)
{
  BeginJoint();
  rebel->on_error(rclcpp_lifecycle::State());
  JointRefused("latched");
}

TEST_F(RebelJointJogTest, EntryRefusedWhenStatusIsStale)
{
  BeginJoint();
  loop.reset();  // no read(): the 1 s latch is not reached while STATUS is silent
  server.SetSendStatus(false);
  std::this_thread::sleep_for(300ms);
  JointRefused("STATUS");
  server.SetSendStatus(true);
}

TEST_F(RebelJointJogTest, EntryRefusedWhileDrivesAreDisabled)
{
  BeginJoint();
  server.SetIgnoreEnable(true);
  server.SetDisabledMask(0x01);
  std::this_thread::sleep_for(150ms);
  JointRefused("drives");
}

TEST_F(RebelJointJogTest, EntryRefusedWhileAJointMoveIsBusy)
{
  BeginJoint();
  server.SetMoveSpeedDegS(10.0);
  const auto handle = SendAccepted(Goal({40.0, 0, 0, 0, 0, 0}));
  ASSERT_NE(handle, nullptr);
  ASSERT_TRUE(WaitMoving(1.0));
  JointRefused("joint move");
  client->Cancel(handle);
  client->Result(handle);
}

TEST_F(RebelJointJogTest, EntryRefusedWhileAMoveStopIsPending)
{
  Begin();
  MAKE_UNCONFIRMED_STOP();
  EXPECT_EQ(rebel->on_deactivate(rclcpp_lifecycle::State()), CallbackReturn::SUCCESS);
  EXPECT_EQ(rebel->on_activate(rclcpp_lifecycle::State()), CallbackReturn::SUCCESS);
  ASSERT_FALSE(rebel->HardwareErrorLatched());
  JointRefused("Move Stop");
  server.SetIgnoreMoveStop(false);
  ASSERT_TRUE(WaitFor([&]() {return !server.MoveActive();}, 1000ms));
}

TEST_F(RebelJointJogTest, EntryRefusedWhileAControllerClaimsAnInterfaceThenWorksOnceReleased)
{
  BeginJoint();
  const auto name = [&](int i) {return rebel->JointNames()[i] + "/velocity";};
  const std::vector<std::string> all = {name(0), name(1), name(2), name(3), name(4), name(5)};
  // A claim that is prepared but not yet performed already blocks.
  EXPECT_EQ(rebel->prepare_command_mode_switch(all, {}), return_type::OK);
  JointRefused("claims");
  EXPECT_EQ(rebel->perform_command_mode_switch(all, {}), return_type::OK);
  JointRefused("claims");
  // One interface still claimed is enough.
  EXPECT_EQ(
    rebel->perform_command_mode_switch({}, {name(0), name(1), name(2), name(3), name(4)}),
    return_type::OK);
  JointRefused("claims");
  EXPECT_EQ(rebel->JointJogStateValue(), 0);
  EXPECT_EQ(rebel->perform_command_mode_switch({}, {name(5)}), return_type::OK);
  EnterJoint();
}

TEST_F(RebelJointJogTest, EntryRefusedWhenTheRobotModeIsNotJoint)
{
  BeginJoint();
  server.ForceMode(2);  // fsm: not a mismatch for the cart guard, but not joint either
  std::this_thread::sleep_for(100ms);
  JointRefused("not 'joint'");
  EXPECT_FALSE(rebel->HardwareErrorLatched());
  server.ForceMode(0);
  std::this_thread::sleep_for(100ms);
  EnterJoint();
}

TEST_F(RebelJointJogTest, EntryRefusedWhenTheRobotReportsCartbase)
{
  BeginJoint();
  server.ForceMode(1);
  std::this_thread::sleep_for(60ms);  // well inside the 0.3 s latch grace of the cart guard
  JointRefused("not 'joint'");
  server.ForceMode(0);
}

TEST_F(RebelJointJogTest, EnteringSendsNothingKeepsJogZeroAndOnlyThenForwardsTheTopic)
{
  BeginJoint();
  const auto t0 = Clock::now();
  joint->StartStream({0.0, 0.05, 0.0, 0.0, 0.0, 0.0});  // arrives before the mode is on: ignored
  std::this_thread::sleep_for(150ms);
  EXPECT_TRUE(AllJogsZero(server, t0));

  const auto reply = joint->Enable(true);
  ASSERT_TRUE(reply.answered);
  ASSERT_TRUE(reply.success) << reply.message;
  const auto entered = Clock::now();
  EXPECT_TRUE(rebel->JointJogActive());
  // No mode switch: the robot stays in joint mode and no CMD went out.
  EXPECT_FALSE(server.ModeIsCartBase());
  EXPECT_EQ(server.Commands(1), kStartupSequence);
  EXPECT_TRUE(WaitFor([&]() {return SawJog(server, entered, {0, Pct(0.05), 0, 0, 0, 0});}, 1000ms));
  TelemetryListener telemetry;
  EXPECT_TRUE(
    telemetry.WaitState(
      [](const igus_rebel_msgs::msg::HardwareState & m) {
        return m.joint_jog_active && m.joint_jog_state == "active" && !m.cart_jog_active &&
        m.cart_jog_state == "off";
      }, 2000ms));
  // Calling it again is a no-op.
  EXPECT_TRUE(joint->Enable(true).success);
  EXPECT_EQ(server.Commands(1), kStartupSequence);
}

TEST_F(RebelJointJogTest, ScalesWithTheJogVelocityScaleClampsToTheMaximumAndKeepsSigns)
{
  BeginJoint();
  EnterJoint();
  // Default clamp 10 %: 0.05 rad/s = 5.73 %, 0.5 rad/s = 57 % -> 10.
  auto t = Clock::now();
  joint->StartStream({0.05, -0.05, 0.0, 0.5, -0.5, 0.01});
  EXPECT_TRUE(
    WaitFor(
      [&]() {return SawJog(server, t, {Pct(0.05), -Pct(0.05), 0, 10.0, -10.0, Pct(0.01)});}, 1000ms));
  const auto settle = Clock::now();
  std::this_thread::sleep_for(200ms);
  for (const auto & frame : server.AliveJogs(settle)) {
    for (const double value : ParseJog(frame.body)) {
      EXPECT_LE(std::fabs(value), 10.0 + 1e-6) << frame.body;
    }
  }
}

TEST_F(RebelJointJogTest, MaxPercentAndVelocityScaleParametersApply)
{
  BeginJoint({{"joint_jog_max_percent", "15"}, {"jog_velocity_scale", "1.0"}});
  EnterJoint();
  EXPECT_DOUBLE_EQ(rebel->JointJogSettings().maxPercent, 15.0);
  const auto t = Clock::now();
  joint->StartStream({0.1, 0.5, -0.5, 0, 0, 0});  // 1.0 * 0.1 rad/s = 5.73 %; the others clamp
  EXPECT_TRUE(
    WaitFor(
      [&]() {return SawJog(server, t, {0.1 * 180.0 / M_PI, 15.0, -15.0, 0, 0, 0});}, 1000ms));
}

TEST_F(RebelJointJogTest, NonFiniteVelocitiesBecomeZeroForThatJointOnly)
{
  BeginJoint();
  EnterJoint();
  const auto nan = std::numeric_limits<double>::quiet_NaN();
  const auto inf = std::numeric_limits<double>::infinity();
  const auto t = Clock::now();
  joint->StartStream({nan, inf, 0.05, -inf, 0.0, 0.02});
  EXPECT_TRUE(WaitFor([&]() {return SawJog(server, t, {0, 0, Pct(0.05), 0, 0, Pct(0.02)});}, 1000ms));
  const auto settle = Clock::now();
  std::this_thread::sleep_for(150ms);
  for (const auto & frame : server.AliveJogs(settle)) {
    EXPECT_TRUE(JogNear(ParseJog(frame.body), {0, 0, Pct(0.05), 0, 0, Pct(0.02)})) << frame.body;
  }
}

TEST_F(RebelJointJogTest, SubsetsAndAnyOrderMapToTheNamedHardwareJoints)
{
  BeginJoint();
  EnterJoint();
  auto msg = JointClient::Make({0, 0, 0, 0, 0, 0}, joint->Now());
  msg.joint_names = {"joint6", "joint2"};
  msg.velocities = {-0.03, 0.04};
  auto t = Clock::now();
  const auto publishLoop = [&](const JointJogMsg & base) {
      for (int i = 0; i < 15; ++i) {
        auto m = base;
        m.header.stamp = joint->Now();
        joint->PublishRaw(m);
        std::this_thread::sleep_for(20ms);
      }
    };
  publishLoop(msg);
  EXPECT_TRUE(SawJog(server, t, {0, Pct(0.04), 0, 0, 0, -Pct(0.03)}));
  // A single joint, an empty message (all zero) and the reverse order.
  msg.joint_names = {"joint3"};
  msg.velocities = {0.02};
  t = Clock::now();
  publishLoop(msg);
  EXPECT_TRUE(SawJog(server, t, {0, 0, Pct(0.02), 0, 0, 0}));
  msg.joint_names = {};
  msg.velocities = {};
  t = Clock::now();
  publishLoop(msg);
  EXPECT_TRUE(rebel->JointJogActive()) << "an empty (all zero) command is valid";
  EXPECT_TRUE(AllJogsZero(server, t + 60ms));
  msg.joint_names = {"joint6", "joint5", "joint4", "joint3", "joint2", "joint1"};
  msg.velocities = {0.06, 0.05, 0.04, 0.03, 0.02, 0.01};
  t = Clock::now();
  publishLoop(msg);
  EXPECT_TRUE(
    SawJog(server, t, {Pct(0.01), Pct(0.02), Pct(0.03), Pct(0.04), Pct(0.05), Pct(0.06)}));
}

TEST_F(RebelJointJogTest, InvalidMessagesAreRejectedAsAWholeAndZeroTheJogAtOnce)
{
  BeginJoint();
  EnterJoint();
  struct Case
  {
    const char * name;
    std::function<void(JointJogMsg &)> corrupt;
  };
  const std::vector<Case> cases = {
    {"unknown name", [](JointJogMsg & m) {m.joint_names[1] = "joint7";}},
    {"name of another arm's joint", [](JointJogMsg & m) {m.joint_names[0] = "arm_left_joint1";}},
    {"duplicate name", [](JointJogMsg & m) {m.joint_names[3] = m.joint_names[0];}},
    {"more names than velocities", [](JointJogMsg & m) {m.velocities.pop_back();}},
    {"more velocities than names", [](JointJogMsg & m) {m.velocities.push_back(0.01);}},
    {"displacements present", [](JointJogMsg & m) {m.displacements = {0.0, 0.0};}},
    {"displacements with names only", [](JointJogMsg & m) {
        m.displacements.assign(m.joint_names.size(), 0.1);
      }},
  };
  for (const auto & testCase : cases) {
    SCOPED_TRACE(testCase.name);
    // A valid command makes the jog non-zero first (and arms a 0.15 s watchdog window) ...
    joint->PublishRaw(JointClient::Make({0.05, 0.05, 0.05, 0.05, 0.05, 0.05}, joint->Now()));
    ASSERT_TRUE(WaitFor([&]() {return AnyNonZeroJog(server, Clock::now() - 30ms);}, 500ms));
    // ... then an invalid one zeroes it at once: well before that watchdog could.
    auto bad = JointClient::Make({0.05, 0.05, 0.05, 0.05, 0.05, 0.05}, joint->Now());
    testCase.corrupt(bad);
    joint->PublishRaw(bad);
    const auto rejected = Clock::now();
    std::this_thread::sleep_for(100ms);
    EXPECT_TRUE(AllJogsZero(server, rejected + 40ms));
    EXPECT_TRUE(rebel->JointJogActive()) << "an invalid message does not leave the mode";
    // Let the watchdog window end before the next case.
    std::this_thread::sleep_for(200ms);
  }
  // A message with one valid and one unknown joint moves neither.
  auto partial = JointClient::Make({0, 0, 0, 0, 0, 0}, joint->Now());
  partial.joint_names = {"joint1", "nope"};
  partial.velocities = {0.05, 0.05};
  EXPECT_FALSE(MovesAfterSingle(partial));
}

TEST_F(RebelJointJogTest, WatchdogZeroesJogWithoutCommandsAndThereIsNoStutterWhileStreaming)
{
  BeginJoint();
  EnterJoint();
  const auto t = Clock::now();
  joint->StartStream({0.05, 0, 0, 0, 0, 0});
  ASSERT_TRUE(WaitFor([&]() {return SawJog(server, t, {Pct(0.05), 0, 0, 0, 0, 0});}, 1000ms));
  const auto streamingFrom = Clock::now();
  std::this_thread::sleep_for(600ms);
  std::size_t frames = 0;
  for (const auto & frame : server.AliveJogs(streamingFrom)) {
    ++frames;
    EXPECT_TRUE(JogNear(ParseJog(frame.body), {Pct(0.05), 0, 0, 0, 0, 0})) << frame.body;
  }
  EXPECT_GT(frames, 40u);
  joint->StopStream();
  const auto silentFrom = Clock::now();
  std::this_thread::sleep_for(500ms);
  EXPECT_TRUE(AllJogsZero(server, silentFrom + 230ms));
  EXPECT_TRUE(rebel->JointJogActive());
  const auto again = Clock::now();
  joint->StartStream({0, 0.03, 0, 0, 0, 0});
  EXPECT_TRUE(WaitFor([&]() {return SawJog(server, again, {0, Pct(0.03), 0, 0, 0, 0});}, 1000ms));
  joint->SetStream({0, 0, 0, 0, 0, 0});
  const auto zeroAt = Clock::now();
  std::this_thread::sleep_for(150ms);
  EXPECT_TRUE(AllJogsZero(server, zeroAt + 60ms));
}

TEST_F(RebelJointJogTest, CommandStampsAreChecked)
{
  BeginJoint();
  const rclcpp::Time beforeEntry = joint->Now();
  EnterJoint();
  const auto moves = [&](const std::function<rclcpp::Time()> & stamp) {
      const auto from = Clock::now();
      for (int i = 0; i < 15; ++i) {
        joint->PublishStamped({0.05, 0, 0, 0, 0, 0}, stamp());
        std::this_thread::sleep_for(20ms);
      }
      std::this_thread::sleep_for(60ms);
      return AnyNonZeroJog(server, from);
    };
  {
    const auto from = Clock::now();
    joint->PublishStamped({0.05, 0, 0, 0, 0, 0}, beforeEntry);
    std::this_thread::sleep_for(100ms);
    EXPECT_FALSE(AnyNonZeroJog(server, from)) << "stamped before the mode entry";
  }
  EXPECT_FALSE(moves([&]() {return rclcpp::Time(0, 0, RCL_ROS_TIME);}));
  EXPECT_FALSE(moves([&]() {return joint->Now() - rclcpp::Duration(1, 0);}));
  EXPECT_FALSE(moves([&]() {return joint->Now() + rclcpp::Duration(2, 0);}));
  EXPECT_TRUE(rebel->JointJogActive());
  EXPECT_TRUE(moves([&]() {return joint->Now();}));
  std::this_thread::sleep_for(400ms);
  const auto replay = Clock::now();
  joint->PublishStamped({0.05, 0, 0, 0, 0, 0}, joint->Now() - rclcpp::Duration(0, 400000000));
  std::this_thread::sleep_for(150ms);
  EXPECT_FALSE(AnyNonZeroJog(server, replay));
}

TEST_F(RebelJointJogTest, TopicIsIgnoredAndNothingChangesWhileTheModeWasNeverEntered)
{
  BeginJoint();
  const auto t = Clock::now();
  joint->StartStream({0.05, 0.05, 0.05, 0.05, 0.05, 0.05});
  std::this_thread::sleep_for(400ms);
  EXPECT_TRUE(AllJogsZero(server, t));
  EXPECT_EQ(server.Commands(1), kStartupSequence);
  loop->SetVelocity(0.1);
  EXPECT_TRUE(WaitFor([&]() {return AnyNonZeroJog(server, Clock::now() - 60ms);}, 1000ms));
}

TEST_F(RebelJointJogTest, ControllerCommandsAreIgnoredWhileTheModeIsOnAndResumeAfterLeaving)
{
  BeginJoint();
  EnterJoint();
  loop->SetVelocity(0.3);  // a controller loop would write this every cycle
  auto t = Clock::now();
  joint->StartStream({0, 0.02, 0, 0, 0, 0});
  ASSERT_TRUE(WaitFor([&]() {return SawJog(server, t, {0, Pct(0.02), 0, 0, 0, 0});}, 1000ms));
  t = Clock::now();
  std::this_thread::sleep_for(300ms);
  for (const auto & frame : server.AliveJogs(t)) {
    EXPECT_TRUE(JogNear(ParseJog(frame.body), {0, Pct(0.02), 0, 0, 0, 0})) << frame.body;
  }
  // Without joint commands the jog is zero, not the controller's 0.3 rad/s.
  joint->StopStream();
  const auto silent = Clock::now();
  std::this_thread::sleep_for(400ms);
  EXPECT_TRUE(AllJogsZero(server, silent + 230ms));
}

TEST_F(RebelJointJogTest, LeavingByServiceZeroesJogAtOnceSendsNoCommandAndTheControllerPathResumes)
{
  BeginJoint();
  EnterJoint();
  joint->StartStream({0.05, 0, 0, 0, 0, 0});
  ASSERT_TRUE(WaitFor([&]() {return SawJog(server, Clock::now() - 60ms, {Pct(0.05), 0, 0, 0, 0, 0});}, 1000ms));
  const auto reply = joint->Enable(false);
  const auto answered = Clock::now();
  ASSERT_TRUE(reply.answered);
  EXPECT_TRUE(reply.success) << reply.message;
  EXPECT_EQ(rebel->JointJogStateValue(), 0);
  EXPECT_EQ(server.Commands(1), kStartupSequence) << "no mode switch in either direction";
  // The stream goes on: nothing is forwarded; zero from the answer on (one ALIVEJOG period margin).
  std::this_thread::sleep_for(200ms);
  EXPECT_TRUE(AllJogsZero(server, answered + 30ms));
  // Leaving twice is fine; the controller path works again.
  std::string message;
  EXPECT_TRUE(rebel->SetJointJogEnabled(false, message)) << message;
  joint->StopStream();
  loop->SetVelocity(0.1);
  EXPECT_TRUE(WaitFor([&]() {return AnyNonZeroJog(server, Clock::now() - 60ms);}, 1000ms));
}

TEST_F(RebelJointJogTest, WhileOnClaimsMoveJointResetAndEnableAreRefused)
{
  BeginJoint();
  EnterJoint();
  const auto name = [&](int i) {return rebel->JointNames()[i] + "/velocity";};
  const std::vector<std::string> all = {name(0), name(1), name(2), name(3), name(4), name(5)};
  EXPECT_EQ(rebel->prepare_command_mode_switch(all, {}), return_type::ERROR);
  EXPECT_EQ(rebel->prepare_command_mode_switch({"unrelated/position"}, {}), return_type::OK);
  EXPECT_TRUE(rebel->JointJogActive());
  const std::size_t movesBefore = server.MoveCommands().size();
  EXPECT_EQ(client->Send(Goal({4.0, 0, 0, 0, 0, 0})), nullptr);
  EXPECT_EQ(server.MoveCommands().size(), movesBefore);
  EXPECT_FALSE(rebel->MoveGoalBusy());
  std::string message;
  EXPECT_FALSE(rebel->ResetAndEnable(message));
  EXPECT_NE(message.find("joint jog"), std::string::npos) << message;
  EXPECT_TRUE(rebel->JointJogActive());
}

TEST_F(RebelJointJogTest, AClaimThatCannotBeRefusedLeavesTheModeAndLatches)
{
  BeginJoint();
  EnterJoint();
  joint->StartStream({0.05, 0, 0, 0, 0, 0});
  ASSERT_TRUE(WaitFor([&]() {return SawJog(server, Clock::now() - 60ms, {Pct(0.05), 0, 0, 0, 0, 0});}, 1000ms));
  const auto name = [&](int i) {return rebel->JointNames()[i] + "/velocity";};
  const auto at = Clock::now();
  // Humble activates the controller even when prepare refused.
  rebel->perform_command_mode_switch({name(0), name(1), name(2), name(3), name(4), name(5)}, {});
  EXPECT_TRUE(rebel->HardwareErrorLatched());
  EXPECT_EQ(rebel->JointJogStateValue(), 0);
  std::this_thread::sleep_for(150ms);
  EXPECT_TRUE(AllJogsZero(server, at + 30ms));
}

TEST_F(RebelJointJogTest, MutualExclusionWithTheCartesianJogModeInBothDirections)
{
  BeginJoint();
  // joint on -> cartesian refused.
  EnterJoint();
  Refused("joint jog mode is on");
  EXPECT_EQ(server.CountCommands("MotionTypeCartBase"), 0u);
  EXPECT_EQ(rebel->CartJogStateValue(), 0);
  EXPECT_TRUE(cart->Enable(true).answered);  // through the service as well
  EXPECT_EQ(rebel->CartJogStateValue(), 0);
  EXPECT_TRUE(rebel->JointJogActive());
  // joint off -> cartesian works.
  std::string message;
  EXPECT_TRUE(rebel->SetJointJogEnabled(false, message)) << message;
  EnterMode();
  // cartesian on -> joint refused, cartesian untouched.
  JointRefused("Cartesian jog mode is on");
  EXPECT_EQ(rebel->JointJogStateValue(), 0);
  EXPECT_TRUE(rebel->CartJogActive());
  EXPECT_FALSE(joint->Enable(true).success);
  // cartesian off -> joint works again, and the CartBase/Joint commands were only the cart's.
  EXPECT_TRUE(rebel->SetCartJogEnabled(false, message)) << message;
  ASSERT_EQ(rebel->CartJogStateValue(), 0);
  EnterJoint();
  EXPECT_EQ(server.CountCommands("MotionTypeCartBase"), 1u);
  EXPECT_EQ(server.CountCommands("MotionTypeJoint"), 2u);  // start-up + the cart exit
}

TEST_F(RebelJointJogTest, EntryIsRefusedWhileTheCartesianModeIsStillBeingLeft)
{
  BeginJoint();
  EnterMode();
  server.SetSuppressJointAck(true);  // MotionTypeJoint is never acknowledged: "leaving" persists
  std::string cartMessage;
  std::future<bool> leaving = std::async(
    std::launch::async, [&]() {return rebel->SetCartJogEnabled(false, cartMessage);});
  ASSERT_TRUE(WaitFor([&]() {return rebel->CartJogStateValue() == 3;}, 1000ms));
  JointRefused("Cartesian jog mode is on");
  EXPECT_EQ(rebel->JointJogStateValue(), 0);
  leaving.wait();  // the unconfirmed exit latches after ~3 s
}

TEST_F(RebelJointJogTest, LeavingByLatchZeroesJogAtOnce)
{
  BeginJoint();
  EnterJoint();
  joint->StartStream({0.05, 0, 0, 0, 0, 0});
  ASSERT_TRUE(WaitFor([&]() {return SawJog(server, Clock::now() - 60ms, {Pct(0.05), 0, 0, 0, 0, 0});}, 1000ms));
  const auto at = Clock::now();
  rebel->on_error(rclcpp_lifecycle::State());
  ASSERT_TRUE(rebel->HardwareErrorLatched());
  EXPECT_EQ(rebel->JointJogStateValue(), 0);
  std::this_thread::sleep_for(150ms);
  EXPECT_TRUE(AllJogsZero(server, at + 30ms));
  JointRefused("latched");
}

TEST_F(RebelJointJogTest, LeavingByReconnectResetsTheStateAndNothingCrossesToTheNewConnection)
{
  BeginJoint();
  EnterJoint();
  joint->StartStream({0.05, 0, 0, 0, 0, 0});
  ASSERT_TRUE(WaitFor([&]() {return SawJog(server, Clock::now() - 60ms, {Pct(0.05), 0, 0, 0, 0, 0});}, 1000ms));
  const auto at = Clock::now();
  server.DropClient();
  const std::size_t second = WaitSecondConnection(server, 3000ms);
  ASSERT_GE(second, 2u) << "no reconnect";
  EXPECT_EQ(server.Commands(second), kStartupSequence);
  ASSERT_TRUE(WaitFor([&]() {return rebel->JointJogStateValue() == 0;}, 1000ms));
  std::this_thread::sleep_for(200ms);
  for (const auto & frame : server.AliveJogs(at)) {
    if (frame.connection == second) {
      EXPECT_TRUE(JogNear(ParseJog(frame.body), {0, 0, 0, 0, 0, 0})) << frame.body;
    }
  }
  EXPECT_TRUE(WaitFor([&]() {return rebel->HardwareErrorLatched();}, 2000ms));
}

TEST_F(RebelJointJogTest, LeavingByDeactivateZeroesJog)
{
  BeginJoint();
  EnterJoint();
  joint->StartStream({0.05, 0, 0, 0, 0, 0});
  ASSERT_TRUE(WaitFor([&]() {return SawJog(server, Clock::now() - 60ms, {Pct(0.05), 0, 0, 0, 0, 0});}, 1000ms));
  const auto at = Clock::now();
  EXPECT_EQ(rebel->on_deactivate(rclcpp_lifecycle::State()), CallbackReturn::SUCCESS);
  EXPECT_EQ(rebel->JointJogStateValue(), 0);
  std::this_thread::sleep_for(150ms);
  EXPECT_TRUE(AllJogsZero(server, at + 30ms));
  JointRefused("not active");
}

TEST_F(RebelJointJogTest, ResetStateAfterLatchRecoveryStartsOff)
{
  BeginJoint();
  EnterJoint();
  rebel->on_error(rclcpp_lifecycle::State());
  EXPECT_EQ(rebel->on_deactivate(rclcpp_lifecycle::State()), CallbackReturn::SUCCESS);
  EXPECT_EQ(rebel->on_activate(rclcpp_lifecycle::State()), CallbackReturn::SUCCESS);
  EXPECT_EQ(rebel->JointJogStateValue(), 0);
  EXPECT_FALSE(rebel->HardwareErrorLatched());
  loop.reset();
  StartLoop(0.0);
  // The stream that never stopped does not start the mode again; only a new call does.
  const auto t = Clock::now();
  joint->StartStream({0.05, 0, 0, 0, 0, 0});
  std::this_thread::sleep_for(200ms);
  EXPECT_TRUE(AllJogsZero(server, t));
}

TEST_F(RebelJointJogTest, UnpromptedSwitchToAnotherModeWhileActiveZeroesJogAndLeavesTheMode)
{
  BeginJoint();
  EnterJoint();
  joint->StartStream({0.05, 0, 0, 0, 0, 0});
  ASSERT_TRUE(WaitFor([&]() {return SawJog(server, Clock::now() - 60ms, {Pct(0.05), 0, 0, 0, 0, 0});}, 1000ms));
  const auto at = Clock::now();
  server.ForceMode(2);  // someone else switched the robot to fsm
  ASSERT_TRUE(WaitFor([&]() {return rebel->JointJogStateValue() == 0;}, 1000ms));
  std::this_thread::sleep_for(150ms);
  EXPECT_TRUE(AllJogsZero(server, at + 80ms));
  EXPECT_FALSE(rebel->HardwareErrorLatched());
  server.ForceMode(0);
  std::this_thread::sleep_for(100ms);
  // The stream is still running but the mode stays off.
  const auto after = Clock::now();
  std::this_thread::sleep_for(150ms);
  EXPECT_TRUE(AllJogsZero(server, after));
}

TEST_F(RebelJointJogTest, CartbaseWhileActiveZeroesJogLeavesTheModeAndThenLatchesLikeJointMode)
{
  BeginJoint();
  EnterJoint();
  joint->StartStream({0.05, 0, 0, 0, 0, 0});
  ASSERT_TRUE(WaitFor([&]() {return SawJog(server, Clock::now() - 60ms, {Pct(0.05), 0, 0, 0, 0, 0});}, 1000ms));
  const auto at = Clock::now();
  server.ForceMode(1);
  ASSERT_TRUE(WaitFor([&]() {return rebel->JointJogStateValue() == 0;}, 1000ms));
  // Joint % would be read as Cartesian speeds: the existing guard latches after its grace period.
  ASSERT_TRUE(WaitFor([&]() {return rebel->HardwareErrorLatched();}, 2000ms));
  std::this_thread::sleep_for(100ms);
  EXPECT_TRUE(AllJogsZero(server, at + 80ms));
}

TEST_F(RebelJointJogTest, UnknownModeWhileActiveFailsClosedAndLeavesTheMode)
{
  BeginJoint();
  EnterJoint();
  joint->StartStream({0.05, 0, 0, 0, 0, 0});
  ASSERT_TRUE(WaitFor([&]() {return SawJog(server, Clock::now() - 60ms, {Pct(0.05), 0, 0, 0, 0, 0});}, 1000ms));
  server.SetSendStatus(false);  // freeze STATUS so the seam below is not overwritten
  std::this_thread::sleep_for(40ms);
  rebel->SetStatusModeForTest(static_cast<int>(Igus::CriMessages::Mode::UNKNOWN));
  const auto at = Clock::now();
  ASSERT_TRUE(WaitFor([&]() {return rebel->JointJogStateValue() == 0;}, 120ms));
  std::this_thread::sleep_for(60ms);
  EXPECT_TRUE(AllJogsZero(server, at + 40ms));
  server.SetSendStatus(true);
}

TEST_F(RebelJointJogTest, IdleAutoExitLeavesTheModeWhenNoCommandArrives)
{
  BeginJoint({{"joint_jog_idle_exit_s", "0.8"}});
  EnterJoint();
  joint->StartStream({0.05, 0, 0, 0, 0, 0});
  std::this_thread::sleep_for(1200ms);
  EXPECT_TRUE(rebel->JointJogActive()) << "commands keep the mode alive";
  joint->StopStream();
  const auto silent = Clock::now();
  ASSERT_TRUE(WaitFor([&]() {return rebel->JointJogStateValue() == 0;}, 4000ms));
  EXPECT_GE(Clock::now() - silent, 600ms);
  EXPECT_FALSE(rebel->HardwareErrorLatched());
  EXPECT_EQ(server.Commands(1), kStartupSequence);
}

TEST_F(RebelJointJogTest, IdleAutoExitCountsFromEntryWhenNoCommandEverArrives)
{
  BeginJoint({{"joint_jog_idle_exit_s", "0.6"}});
  EnterJoint();
  ASSERT_TRUE(WaitFor([&]() {return rebel->JointJogStateValue() == 0;}, 4000ms));
}

TEST_F(RebelJointJogTest, IdleAutoExitCanBeDisabled)
{
  BeginJoint({{"joint_jog_idle_exit_s", "0"}});
  EnterJoint();
  std::this_thread::sleep_for(1200ms);
  EXPECT_TRUE(rebel->JointJogActive());
}

TEST_F(RebelJointJogTest, InvalidJointJogParametersAreRejectedAtInit)
{
  for (const auto & bad : std::vector<std::pair<std::string, std::string>>{
    {"joint_jog_max_percent", "0"}, {"joint_jog_max_percent", "-5"},
    {"joint_jog_max_percent", "101"}, {"joint_jog_max_percent", "nan"},
    {"joint_jog_max_percent", "abc"}, {"joint_jog_timeout_s", "0.001"},
    {"joint_jog_timeout_s", "5"}, {"joint_jog_timeout_s", "inf"},
    {"joint_jog_idle_exit_s", "-1"}, {"joint_jog_idle_exit_s", "abc"},
    {"joint_jog_idle_exit_s", "4000"}})
  {
    Igus::Rebel bad_rebel("127.0.0.1", 1);
    auto info = MakeInfo();
    info.hardware_parameters[bad.first] = bad.second;
    const auto start = Clock::now();
    EXPECT_EQ(bad_rebel.on_init(info), CallbackReturn::ERROR) << bad.first << "=" << bad.second;
    EXPECT_LT(Clock::now() - start, 500ms) << "must fail before connecting";
  }
}

TEST_F(RebelJointJogTest, DefaultsAreTenPercentWatchdogAndIdleExit)
{
  BeginJoint();
  EXPECT_DOUBLE_EQ(rebel->JointJogSettings().maxPercent, 10.0);
  EXPECT_DOUBLE_EQ(rebel->JointJogSettings().timeoutS, 0.15);
  EXPECT_DOUBLE_EQ(rebel->JointJogSettings().idleExitS, 10.0);
}

TEST_F(RebelJointJogTest, StatusCoastSendsZeroJogAndAHoldDuringMotionLatches)
{
  BeginJoint();
  EnterJoint();
  joint->StartStream({0.05, 0, 0, 0, 0, 0});
  ASSERT_TRUE(WaitFor([&]() {return SawJog(server, Clock::now() - 60ms, {Pct(0.05), 0, 0, 0, 0, 0});}, 1000ms));
  // Short STATUS gap (coast tier): the controller path would freeze the jog; the direct mode sends zero.
  server.SetSendStatus(false);
  const auto pauseStart = Clock::now();
  ASSERT_TRUE(WaitFor([&]() {return rebel->StatusCoastActive();}, 1000ms));
  const auto coastStart = Clock::now();
  std::this_thread::sleep_for(pauseStart + 300ms - Clock::now());
  server.SetSendStatus(true);
  ASSERT_TRUE(WaitFor([&]() {return !rebel->StatusCoastActive();}, 1000ms));
  const auto resumed = Clock::now();
  EXPECT_FALSE(rebel->HardwareErrorLatched());
  std::size_t checked = 0;
  for (const auto & frame : server.AliveJogs(coastStart + 30ms)) {
    if (frame.stamp > resumed - 20ms) {continue;}
    ++checked;
    EXPECT_TRUE(JogNear(ParseJog(frame.body), {0, 0, 0, 0, 0, 0})) << frame.body;
  }
  EXPECT_GE(checked, 3u);
  EXPECT_TRUE(WaitFor([&]() {return SawJog(server, resumed + 30ms, {Pct(0.05), 0, 0, 0, 0, 0});}, 1000ms));
  // Long gap (hold tier) while moving: zero jog, and the resume latches (no auto-resume).
  server.SetSendStatus(false);
  const auto holdPause = Clock::now();
  ASSERT_TRUE(WaitFor([&]() {return rebel->StatusHoldActive();}, 1000ms));
  const auto holdStart = Clock::now();
  std::this_thread::sleep_for(holdPause + 600ms - Clock::now());
  server.SetSendStatus(true);
  ASSERT_TRUE(WaitFor([&]() {return rebel->HardwareErrorLatched();}, 1000ms));
  for (const auto & frame : server.AliveJogs(holdStart + 30ms)) {
    EXPECT_TRUE(JogNear(ParseJog(frame.body), {0, 0, 0, 0, 0, 0})) << frame.body;
  }
  EXPECT_EQ(rebel->JointJogStateValue(), 0);
}

TEST_F(RebelJointJogTest, PreparedFrameCannotCrossAReconnect)
{
  BeginJoint();
  EnterJoint();
  AliveHold hold(*rebel);
  joint->StartStream({0.05, 0, 0, 0, 0, 0});
  ASSERT_TRUE(hold.WaitEntered(2000ms));
  server.DropClient();
  const std::size_t second = WaitSecondConnection(server, 3000ms);
  ASSERT_GE(second, 2u) << "no reconnect";
  hold.Release();
  std::this_thread::sleep_for(300ms);
  std::size_t frames = 0;
  for (const auto & frame : server.AliveJogs()) {
    if (frame.connection >= second) {
      ++frames;
      EXPECT_TRUE(JogNear(ParseJog(frame.body), {0, 0, 0, 0, 0, 0})) << frame.body;
    }
  }
  EXPECT_GT(frames, 5u);
}

TEST_F(RebelJointJogTest, PreparedFrameIsDroppedWhenALatchLandsBeforeTheSend)
{
  BeginJoint();
  EnterJoint();
  AliveHold hold(*rebel);
  joint->StartStream({0.05, 0, 0, 0, 0, 0});
  ASSERT_TRUE(hold.WaitEntered(2000ms));
  rebel->on_error(rclcpp_lifecycle::State());
  const auto latched = Clock::now();
  hold.Release();
  std::this_thread::sleep_for(200ms);
  EXPECT_TRUE(AllJogsZero(server, latched));
}

TEST_F(RebelJointJogTest, PreparedFrameIsDroppedWhenTheModeIsLeftBeforeTheSend)
{
  BeginJoint();
  EnterJoint();
  AliveHold hold(*rebel);
  joint->StartStream({0.05, 0, 0, 0, 0, 0});
  ASSERT_TRUE(hold.WaitEntered(2000ms));
  // No latch, no hold, same connection: only the mode itself changes under the prepared frame.
  std::string message;
  EXPECT_TRUE(rebel->SetJointJogEnabled(false, message)) << message;
  const auto left = Clock::now();
  hold.Release();
  std::this_thread::sleep_for(200ms);
  EXPECT_TRUE(AllJogsZero(server, left));
  EXPECT_FALSE(rebel->HardwareErrorLatched());
}

TEST_F(RebelJointJogTest, EntryIsRefusedWhenALatchOrDeactivateRacesIt)
{
  // A fault between the checks and the transition must never leave the mode on: hammer enter
  // against latch + lifecycle recovery and require off whenever latched or deactivated.
  BeginJoint();
  for (int round = 0; round < 20; ++round) {
    std::future<bool> entering = std::async(
      std::launch::async, [&]() {std::string m; return rebel->SetJointJogEnabled(true, m);});
    rebel->on_deactivate(rclcpp_lifecycle::State());
    entering.wait();
    EXPECT_EQ(rebel->JointJogStateValue(), 0) << "round " << round;
    EXPECT_EQ(rebel->on_activate(rclcpp_lifecycle::State()), CallbackReturn::SUCCESS);
  }
}


// ======================================================================================
// Direct jog sessions: generation, commit race, replay / ordering, expiry from the stamp,
// dispatch-time revalidation of a prepared frame (both modes unless noted)
// ======================================================================================
namespace
{
// Holds a cart_jog_cmd / joint_jog_cmd callback between its validation and its commit until
// Release(). One shot: later callbacks pass.
class CommitHold
{
public:
  explicit CommitHold(Igus::Rebel & driver)
  : rebel_(driver)
  {
    rebel_.SetBeforeJogCommitHookForTest(
      [this](bool) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (done_) {return;}
        entered_ = true;
        condition_.notify_all();
        condition_.wait(lock, [this]() {return release_;});
        done_ = true;
      });
  }
  ~CommitHold()
  {
    Release();
    rebel_.SetBeforeJogCommitHookForTest(nullptr);
  }
  bool WaitEntered(std::chrono::milliseconds timeout)
  {
    std::unique_lock<std::mutex> lock(mutex_);
    return condition_.wait_for(lock, timeout, [this]() {return entered_;});
  }
  void Release()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    release_ = true;
    condition_.notify_all();
  }

private:
  Igus::Rebel & rebel_;
  std::mutex mutex_;
  std::condition_variable condition_;
  bool entered_ = false;
  bool release_ = false;
  bool done_ = false;
};

// True if any ALIVEJOG received in [from, to) is non-zero.
bool NonZeroBetween(FakeCriServer & server, Clock::time_point from, Clock::time_point to)
{
  for (const auto & frame : server.AliveJogs(from)) {
    if (frame.stamp >= to) {continue;}
    for (double value : ParseJog(frame.body)) {
      if (value != 0.0) {return true;}
    }
  }
  return false;
}

// Param: true = Cartesian jog, false = direct joint jog.
class RebelJogSessionTest : public RebelJointJogTest, public ::testing::WithParamInterface<bool>
{
protected:
  bool IsCart() const {return GetParam();}
  void EnterMode()
  {
    if (IsCart()) {RebelCartJogTest::EnterMode();} else {EnterJoint();}
  }
  void LeaveMode()
  {
    std::string message;
    if (IsCart()) {
      ASSERT_TRUE(rebel->SetCartJogEnabled(false, message)) << message;
    } else {
      ASSERT_TRUE(rebel->SetJointJogEnabled(false, message)) << message;
    }
    ASSERT_FALSE(ModeOn());
  }
  bool ModeOn() const {return IsCart() ? rebel->CartJogActive() : rebel->JointJogActive();}
  // A command whose frames are non-zero: X 4 % (Cartesian) / joint 1 at Pct(0.05) %.
  Twist6 Nz() const {return IsCart() ? Twist6{0.02, 0, 0, 0, 0, 0} : Twist6{0.05, 0, 0, 0, 0, 0};}
  static Twist6 Zero() {return Twist6{0, 0, 0, 0, 0, 0};}
  rclcpp::Time Now() {return IsCart() ? cart->Now() : joint->Now();}
  void PublishAt(const Twist6 & v, const rclcpp::Time & stamp)
  {
    if (IsCart()) {cart->PublishStamped(v, stamp);} else {joint->PublishStamped(v, stamp);}
  }
  void StartStream(const Twist6 & v)
  {
    if (IsCart()) {cart->StartStream(v);} else {joint->StartStream(v);}
  }
  void StopStream()
  {
    if (IsCart()) {cart->StopStream();} else {joint->StopStream();}
  }
  // The first ALIVEJOG received at or after `after` is all zero.
  bool FirstFrameAfterIsZero(Clock::time_point after)
  {
    WaitFor([&]() {return !server.AliveJogs(after).empty();}, 1000ms);
    const auto frames = server.AliveJogs(after);
    EXPECT_FALSE(frames.empty());
    if (frames.empty()) {return false;}
    return JogNear(ParseJog(frames.front().body), Zero());
  }
};
}  // namespace

TEST_P(RebelJogSessionTest, EntryAndEveryExitAdvanceTheSessionGeneration)
{
  BeginJoint();
  const std::uint64_t g0 = rebel->JogGeneration();
  EnterMode();
  const std::uint64_t g1 = rebel->JogGeneration();
  EXPECT_GT(g1, g0);
  LeaveMode();
  const std::uint64_t g2 = rebel->JogGeneration();
  EXPECT_GT(g2, g1);
  EnterMode();
  const std::uint64_t g3 = rebel->JogGeneration();
  EXPECT_GT(g3, g2);
  rebel->on_deactivate(rclcpp_lifecycle::State());  // exit request
  EXPECT_GT(rebel->JogGeneration(), g3);
}

TEST_P(RebelJogSessionTest, CommandCommitOfAnOlderSessionIsDropped)
{
  BeginJoint();
  EnterMode();
  CommitHold hold(*rebel);
  // Stamped slightly in the future so that it still passes the stamp rules of the next session:
  // only the session generation can reject it at the commit.
  PublishAt(Nz(), Now() + rclcpp::Duration(0, 300000000));
  ASSERT_TRUE(hold.WaitEntered(2000ms));  // validated, not committed
  LeaveMode();
  EnterMode();
  const auto resumed = Clock::now();
  hold.Release();
  std::this_thread::sleep_for(300ms);
  EXPECT_TRUE(AllJogsZero(server, resumed));
  EXPECT_TRUE(ModeOn());
}

TEST_P(RebelJogSessionTest, PreparedFrameIsDroppedWhenAZeroCommandArrivesBeforeTheSend)
{
  BeginJoint();
  EnterMode();
  AliveHold hold(*rebel);
  StartStream(Nz());
  ASSERT_TRUE(hold.WaitEntered(2000ms));
  StopStream();
  PublishAt(Zero(), Now());  // newer than every streamed command, refreshes the deadline
  std::this_thread::sleep_for(60ms);
  const auto released = Clock::now();
  hold.Release();
  std::this_thread::sleep_for(200ms);
  EXPECT_TRUE(AllJogsZero(server, released));
  EXPECT_TRUE(ModeOn());
}

TEST_P(RebelJogSessionTest, PreparedFrameIsDroppedWhenTheCommandExpiresBeforeTheSend)
{
  BeginJoint();
  EnterMode();
  AliveHold hold(*rebel);
  StartStream(Nz());
  ASSERT_TRUE(hold.WaitEntered(2000ms));
  StopStream();
  std::this_thread::sleep_for(260ms);  // > the 0.15 s command timeout, the ALIVEJOG thread is held
  const auto released = Clock::now();
  hold.Release();
  std::this_thread::sleep_for(200ms);
  EXPECT_TRUE(AllJogsZero(server, released));
  EXPECT_TRUE(ModeOn());
}

TEST_P(RebelJogSessionTest, PreparedFrameIsDroppedWhenTheStatusModeChangesBeforeTheSend)
{
  BeginJoint();
  EnterMode();
  AliveHold hold(*rebel);
  StartStream(Nz());
  ASSERT_TRUE(hold.WaitEntered(2000ms));
  server.SetSendStatus(false);  // freeze STATUS so the seam below is not overwritten
  std::this_thread::sleep_for(40ms);
  rebel->SetStatusModeForTest(static_cast<int>(Igus::CriMessages::Mode::UNKNOWN));
  const auto released = Clock::now();
  hold.Release();
  EXPECT_TRUE(FirstFrameAfterIsZero(released));
  server.SetSendStatus(true);
}

TEST_P(RebelJogSessionTest, PreparedFrameIsDroppedWhenAControllerClaimAppearsBeforeTheSend)
{
  BeginJoint();
  EnterMode();
  AliveHold hold(*rebel);
  StartStream(Nz());
  ASSERT_TRUE(hold.WaitEntered(2000ms));
  rebel->SetVelocityClaimMaskForTest(1u);  // no latch: only the dispatch check can see it
  const auto released = Clock::now();
  hold.Release();
  EXPECT_TRUE(FirstFrameAfterIsZero(released));
  rebel->SetVelocityClaimMaskForTest(0u);
}

TEST_P(RebelJogSessionTest, OlderOrDuplicateStampsAreRejectedAndTheTargetStaysZero)
{
  BeginJoint();
  EnterMode();
  const rclcpp::Time older = Now();
  std::this_thread::sleep_for(5ms);
  const rclcpp::Time newer = Now();
  PublishAt(Zero(), newer);  // accepted: first command of the session
  std::this_thread::sleep_for(30ms);
  const auto t = Clock::now();
  PublishAt(Nz(), older);  // age ~35 ms < timeout, but older than the newest accepted stamp
  std::this_thread::sleep_for(200ms);
  EXPECT_FALSE(AnyNonZeroJog(server, t)) << "an older stamp was accepted";
  // Duplicate stamp.
  const rclcpp::Time same = Now();
  PublishAt(Zero(), same);
  std::this_thread::sleep_for(30ms);
  const auto t2 = Clock::now();
  PublishAt(Nz(), same);
  std::this_thread::sleep_for(200ms);
  EXPECT_FALSE(AnyNonZeroJog(server, t2)) << "a duplicate stamp was accepted";
  // A newer stamp is still accepted, the mode kept working.
  const auto t3 = Clock::now();
  PublishAt(Nz(), Now());
  EXPECT_TRUE(WaitFor([&]() {return AnyNonZeroJog(server, t3);}, 1000ms));
  EXPECT_TRUE(ModeOn());
}

TEST_P(RebelJogSessionTest, ACommandExpiresFromItsOwnStampNotFromItsReceipt)
{
  BeginJoint();
  EnterMode();
  // Stamped 0.12 s ago with the default 0.15 s timeout: lives about 0.03 s, not 0.15 s. The stamp
  // must not predate the mode entry, so let the entry age first.
  std::this_thread::sleep_for(200ms);
  const auto t = Clock::now();
  PublishAt(Nz(), Now() - rclcpp::Duration(0, 120000000));
  std::this_thread::sleep_for(450ms);
  EXPECT_TRUE(NonZeroBetween(server, t, t + 60ms)) << "the (still valid) command was not accepted";
  EXPECT_FALSE(NonZeroBetween(server, t + 100ms, t + 450ms)) << "lived a full timeout";
  // A fresh stamp keeps its full lifetime (not shortened).
  const auto t2 = Clock::now();
  PublishAt(Nz(), Now());
  std::this_thread::sleep_for(250ms);
  EXPECT_TRUE(NonZeroBetween(server, t2 + 60ms, t2 + 120ms)) << "fresh command expired early";
  EXPECT_FALSE(NonZeroBetween(server, t2 + 230ms, t2 + 250ms));
}

INSTANTIATE_TEST_SUITE_P(
  Modes, RebelJogSessionTest, ::testing::Bool(),
  [](const ::testing::TestParamInfo<bool> & info) {return info.param ? "Cartesian" : "Joint";});

// The Cartesian exit is completed by the ALIVEJOG thread, so only the joint mode can be left and
// entered again while that thread is held between building and sending a frame.
TEST_F(RebelJointJogTest, PreparedFrameFromAnOlderSessionIsDroppedAfterDisableAndReenable)
{
  BeginJoint();
  EnterJoint();
  AliveHold hold(*rebel);
  joint->StartStream({0.05, 0, 0, 0, 0, 0});
  ASSERT_TRUE(hold.WaitEntered(2000ms));
  joint->StopStream();
  std::string message;
  ASSERT_TRUE(rebel->SetJointJogEnabled(false, message)) << message;
  EnterJoint();
  const auto released = Clock::now();
  hold.Release();
  std::this_thread::sleep_for(250ms);
  EXPECT_TRUE(AllJogsZero(server, released));
  EXPECT_TRUE(rebel->JointJogActive());
}

TEST_F(RebelJointJogTest, PreparedFrameFromAnOlderSessionIsDroppedEvenIfTheNewSessionHasACommand)
{
  BeginJoint();
  EnterJoint();
  AliveHold hold(*rebel);
  joint->StartStream({0.05, 0, 0, 0, 0, 0});
  ASSERT_TRUE(hold.WaitEntered(2000ms));
  joint->StopStream();
  std::string message;
  ASSERT_TRUE(rebel->SetJointJogEnabled(false, message)) << message;
  EnterJoint();
  // A fresh non-zero command of the NEW session: mode state, claims, STATUS mode, deadline,
  // revision and non-zero flag all look fine for the old frame -- only the generation differs.
  joint->Publish({0.05, 0, 0, 0, 0, 0});
  std::this_thread::sleep_for(60ms);
  const auto released = Clock::now();
  hold.Release();
  const auto frames = [&]() {
      std::this_thread::sleep_for(60ms);
      return server.AliveJogs(released);
    }();
  ASSERT_FALSE(frames.empty());
  EXPECT_TRUE(JogNear(ParseJog(frames.front().body), {0, 0, 0, 0, 0, 0}))
    << "the old session's frame was sent: " << frames.front().body;
}

// ======================================================================================
// jtc_jog_deadband_percent (JTC path only) and ~/cri_diagnostics
// ======================================================================================
namespace
{
using CriDiagnostics = igus_rebel_msgs::msg::CriDiagnostics;

// Subscribes to ~/cri_diagnostics from a separate node spun on its own thread.
class DiagnosticsListener
{
public:
  explicit DiagnosticsListener(const std::string & driverNode = "igus_rebel")
  {
    static std::atomic<int> counter{0};
    node_ = std::make_shared<rclcpp::Node>("diag_listener_" + std::to_string(counter++));
    sub_ = node_->create_subscription<CriDiagnostics>(
      "/" + driverNode + "/cri_diagnostics", rclcpp::SensorDataQoS().keep_last(10),
      [this](CriDiagnostics::ConstSharedPtr msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        messages_.push_back(*msg);
        systemReceivedNs_.push_back(SystemNowNs());
      });
    executor_.add_node(node_);
    thread_ = std::thread([this]() {executor_.spin(); finished_ = true;});
  }
  ~DiagnosticsListener()
  {
    while (!finished_) {
      executor_.cancel();
      std::this_thread::sleep_for(2ms);
    }
    if (thread_.joinable()) {
      thread_.join();
    }
  }
  std::vector<CriDiagnostics> Messages()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return messages_;
  }
  std::int64_t LastReceivedSystemNs()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return systemReceivedNs_.empty() ? 0 : systemReceivedNs_.back();
  }
  // Waits for a message received after `fromIndex` messages that satisfies `predicate`.
  template<class Predicate>
  bool WaitMessage(Predicate predicate, std::chrono::milliseconds timeout, std::size_t fromIndex = 0)
  {
    return WaitFor(
      [&]() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (std::size_t i = fromIndex; i < messages_.size(); ++i) {
          if (predicate(messages_[i])) {
            return true;
          }
        }
        return false;
      }, timeout);
  }

private:
  std::atomic<bool> finished_{false};
  rclcpp::Node::SharedPtr node_;
  rclcpp::Subscription<CriDiagnostics>::SharedPtr sub_;
  rclcpp::executors::SingleThreadedExecutor executor_;
  std::thread thread_;
  std::mutex mutex_;
  std::vector<CriDiagnostics> messages_;
  std::vector<std::int64_t> systemReceivedNs_;
};

// Sets / reads parameters of the driver node over the real parameter services.
class DriverParameters
{
public:
  explicit DriverParameters(const std::string & driverNode = "igus_rebel")
  {
    static std::atomic<int> counter{0};
    node_ = std::make_shared<rclcpp::Node>("param_client_" + std::to_string(counter++));
    client_ = std::make_shared<rclcpp::SyncParametersClient>(node_, "/" + driverNode);
  }
  bool WaitForService() {return client_->wait_for_service(5s);}
  // True if the driver accepted the value.
  bool SetDeadband(double value)
  {
    const auto results = client_->set_parameters({rclcpp::Parameter("jtc_jog_deadband_percent", value)});
    return results.size() == 1 && results[0].successful;
  }
  double Deadband()
  {
    return client_->get_parameter<double>("jtc_jog_deadband_percent", -1.0);
  }

private:
  rclcpp::Node::SharedPtr node_;
  std::shared_ptr<rclcpp::SyncParametersClient> client_;
};

// 0.3 % / 0.8 % jog of the default jog_velocity_scale (2.0), as a joint velocity in rad/s.
double VelocityForPercent(double percent) {return percent / 2.0 * Igus::Safety::kDegToRad;}

class RebelJtcDeadbandTest : public RebelJointJogTest
{
protected:
  void BeginDeadband()
  {
    BeginJoint();
    if (HasFatalFailure()) {
      return;
    }
    params = std::make_unique<DriverParameters>();
    ASSERT_TRUE(params->WaitForService());
  }
  void TearDown() override
  {
    params.reset();
    RebelJointJogTest::TearDown();
  }
  // All six jog values of every frame received after `after` are `expected`.
  bool EveryJogIs(Clock::time_point after, double expected, std::size_t minimumFrames)
  {
    const auto frames = server.AliveJogs(after);
    if (frames.size() < minimumFrames) {
      return false;
    }
    for (const auto & frame : frames) {
      for (const double value : ParseJog(frame.body)) {
        if (std::fabs(value - expected) > 1e-3) {
          return false;
        }
      }
    }
    return true;
  }

  std::unique_ptr<DriverParameters> params;
};
}  // namespace

TEST_F(RebelJtcDeadbandTest, DefaultIsOffAndTheParameterRejectsBadValues)
{
  BeginDeadband();
  EXPECT_DOUBLE_EQ(params->Deadband(), 0.0);
  EXPECT_TRUE(params->SetDeadband(0.5));
  EXPECT_DOUBLE_EQ(params->Deadband(), 0.5);
  for (const double bad : {2.5, -0.1, std::numeric_limits<double>::quiet_NaN(),
      std::numeric_limits<double>::infinity()})
  {
    EXPECT_FALSE(params->SetDeadband(bad)) << bad;
    EXPECT_DOUBLE_EQ(params->Deadband(), 0.5) << bad;  // unchanged
  }
  EXPECT_TRUE(params->SetDeadband(2.0));
  EXPECT_TRUE(params->SetDeadband(0.0));
  EXPECT_DOUBLE_EQ(params->Deadband(), 0.0);
}

TEST_F(RebelJtcDeadbandTest, OffByDefaultPassesSmallJtcJogUnchanged)
{
  BeginDeadband();
  const auto t = Clock::now();
  loop->SetVelocity(VelocityForPercent(0.3));
  EXPECT_TRUE(WaitFor([&]() {return SawJog(server, t, {0.3, 0.3, 0.3, 0.3, 0.3, 0.3});}, 1000ms));
}

TEST_F(RebelJtcDeadbandTest, DeadbandZeroesSmallJtcJogOnlyAndPassesLargerJog)
{
  BeginDeadband();
  ASSERT_TRUE(params->SetDeadband(0.5));

  // 0.3 % < 0.5 %: the ros2_control path sends 0 for every joint.
  loop->SetVelocity(VelocityForPercent(0.3));
  std::this_thread::sleep_for(100ms);  // the frames of an older command are gone
  auto t = Clock::now();
  std::this_thread::sleep_for(300ms);
  EXPECT_TRUE(EveryJogIs(t, 0.0, 10)) << "0.3 % was not zeroed";
  EXPECT_FALSE(SawJog(server, t, {0.3, 0.3, 0.3, 0.3, 0.3, 0.3}));

  // 0.8 % >= 0.5 %: passes unchanged.
  loop->SetVelocity(VelocityForPercent(0.8));
  t = Clock::now();
  EXPECT_TRUE(WaitFor([&]() {return SawJog(server, t, {0.8, 0.8, 0.8, 0.8, 0.8, 0.8});}, 1000ms));

  // Back to 0.3 %, deadband removed at runtime: passes again.
  ASSERT_TRUE(params->SetDeadband(0.0));
  loop->SetVelocity(VelocityForPercent(0.3));
  t = Clock::now();
  EXPECT_TRUE(WaitFor([&]() {return SawJog(server, t, {0.3, 0.3, 0.3, 0.3, 0.3, 0.3});}, 1000ms));
}

TEST_F(RebelJtcDeadbandTest, DeadbandDoesNotApplyToTheCriJointJog)
{
  BeginDeadband();
  ASSERT_TRUE(params->SetDeadband(0.5));
  EnterJoint();
  const auto t = Clock::now();
  joint->StartStream({VelocityForPercent(0.3), 0.0, 0.0, 0.0, 0.0, 0.0});
  // 0.3 % on joint 1 reaches the robot although it is below the JTC deadband.
  EXPECT_TRUE(WaitFor([&]() {return SawJog(server, t, {0.3, 0, 0, 0, 0, 0});}, 1000ms));
}

TEST_F(RebelJtcDeadbandTest, CriDiagnosticsCarryStatusCurrentsSetpointAndTheLastSentJog)
{
  BeginDeadband();
  server.SetCurrents({100, -200, 300, -400, 500, -600}, 1234);
  server.SetSetpointOffsetDeg({0.5f, 0.0f, 0.0f, 0.0f, 0.0f, -0.25f});
  DiagnosticsListener listener;

  // 0.8 % from the ros2_control path (deadband 0.5).
  ASSERT_TRUE(params->SetDeadband(0.5));
  loop->SetVelocity(VelocityForPercent(0.8));
  ASSERT_TRUE(
    listener.WaitMessage(
      [](const CriDiagnostics & m) {
        return m.jog_source == "jtc" && std::fabs(m.jog_percent_sent[3] - 0.8f) < 1e-3 &&
        m.current_raw[0] == 100;
      }, 3000ms));
  CriDiagnostics last;
  {
    const auto messages = listener.Messages();
    last = messages.back();
    // The sample of the fake STATUS.
    EXPECT_EQ(last.current_raw[0], 100);
    EXPECT_EQ(last.current_raw[1], -200);
    EXPECT_EQ(last.current_raw[5], -600);
    EXPECT_EQ(last.current_all_raw, 1234);
    for (std::size_t i = 0; i < 6; ++i) {
      EXPECT_NEAR(last.position_deg[i], server.baseJointsDeg[i], 1e-3) << i;
    }
    EXPECT_NEAR(last.setpoint_deg[0], server.baseJointsDeg[0] + 0.5, 1e-3);
    EXPECT_NEAR(last.setpoint_deg[1], server.baseJointsDeg[1], 1e-3);
    EXPECT_NEAR(last.setpoint_deg[5], server.baseJointsDeg[5] - 0.25, 1e-3);
    EXPECT_NEAR(last.jtc_deadband_percent, 0.5f, 1e-6);
    // The stamp is the STATUS receive time (system clock), not far from the delivery time.
    EXPECT_LT(
      std::llabs(
        listener.LastReceivedSystemNs() - rclcpp::Time(last.header.stamp).nanoseconds()),
      500000000);
    // One message per valid STATUS: the sequence is strictly increasing.
    for (std::size_t i = 1; i < messages.size(); ++i) {
      EXPECT_GT(messages[i].status_sequence, messages[i - 1].status_sequence);
    }
  }

  // 0.3 % is below the deadband: what was SENT is 0, still from the jtc path.
  loop->SetVelocity(VelocityForPercent(0.3));
  std::size_t from = listener.Messages().size() + 5;
  ASSERT_TRUE(
    listener.WaitMessage(
      [](const CriDiagnostics & m) {
        return m.jog_source == "jtc" && m.jog_percent_sent[3] == 0.0f;
      }, 3000ms, from));

  // The joint jog mode reports its own source and the percent it sent.
  loop->SetVelocity(0.0);
  EnterJoint();
  joint->StartStream({0.0, VelocityForPercent(0.4), 0.0, 0.0, 0.0, 0.0});
  EXPECT_TRUE(
    listener.WaitMessage(
      [](const CriDiagnostics & m) {
        return m.jog_source == "joint_jog" && std::fabs(m.jog_percent_sent[1] - 0.4f) < 1e-3;
      }, 3000ms));
}

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  ::testing::InitGoogleTest(&argc, argv);
  const int result = RUN_ALL_TESTS();
  rclcpp::shutdown();
  return result;
}
