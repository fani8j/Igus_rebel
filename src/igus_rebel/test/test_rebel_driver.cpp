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
#include <cstdio>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "rclcpp/rclcpp.hpp"

#include "igus_rebel/CriFraming.hpp"
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
  void DropClient() {dropClient_ = true;}
  // Joints (bit i) reported with ErrorJoint MNE (motor not enabled); cleared by an Enable command.
  void SetDisabledMask(unsigned mask) {disabledMask_ = mask;}
  std::size_t Connections() const {return connections_.load();}

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

private:
  std::string MakeStatus(int counter, const std::array<float, 6> & joints)
  {
    std::ostringstream s;
    s << "CRISTART " << counter << " STATUS MODE joint POSJOINTSETPOINT";
    for (int i = 0; i < 16; ++i) {
      s << " " << (i < 6 ? joints[i] : 0.0f);
    }
    s << " POSJOINTCURRENT";
    for (int i = 0; i < 16; ++i) {
      s << " " << (i < 6 ? joints[i] : 0.0f);
    }
    s << " POSCARTROBOT 0 0 0 0 0 0 POSCARTPLATTFORM 0 0 0 OVERRIDE " << override_.load() <<
      " DIN 0 DOUT 0 ESTOP 0 SUPPLY 48 CURRENTALL 2 CURRENTJOINTS";
    for (int i = 0; i < 16; ++i) {
      s << " 0";
    }
    s << " ERROR 0";
    for (int i = 0; i < 16; ++i) {
      s << " " << ((i < 6 && ((disabledMask_.load() >> i) & 1u)) ? 4 : 0);
    }
    s << " KINSTATE 0 OPMODE -1 CRIEND\n";
    return s.str();
  }

  void SendAll(int fd, const std::string & data)
  {
    ::send(fd, data.data(), data.size(), MSG_NOSIGNAL);
  }

  void Run()
  {
    while (running_) {
      pollfd listenPoll{listenFd_, POLLIN, 0};
      if (::poll(&listenPoll, 1, 20) <= 0) {
        continue;
      }
      const int client = ::accept(listenFd_, nullptr, nullptr);
      if (client < 0) {
        continue;
      }
      const std::size_t connection = ++connections_;
      dropClient_ = false;
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
            if (stripped == "CMD Enable") {
              disabledMask_ = 0;
            }
            if (stripped.rfind("CMD ", 0) == 0) {
              const std::string counterText = body.substr(0, body.find(' '));
              SendAll(
                client, "CRISTART " + std::to_string(counter++) + " CMDACK " + counterText +
                " CRIEND\n");
            }
          }
        }
        if (sendStatus_ && Clock::now() >= nextStatus) {
          nextStatus += 20ms;
          std::array<float, 6> joints = baseJointsDeg;
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
  std::atomic<float> jointStepDeg_{0.0f};
  std::atomic<float> override_{40.0f};
  std::atomic<std::size_t> connections_{0};
  std::thread thread_;
  std::mutex mutex_;
  std::vector<ReceivedFrame> frames_;
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
  "SetActive true", "Reset", "Enable", "MotionTypeJoint", "Override 100.0"};

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

  ASSERT_TRUE(WaitFor([&]() {return server.Commands().size() >= 5;}, 2000ms));
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
  ASSERT_TRUE(WaitFor([&]() {return server.Commands().size() >= 7;}, 1000ms));
  const auto commandsAfterStop = server.Commands();
  ASSERT_GE(commandsAfterStop.size(), 7u);
  EXPECT_EQ(commandsAfterStop[commandsAfterStop.size() - 2], "Disable");
  EXPECT_EQ(commandsAfterStop.back(), "QUIT");
}

TEST_F(RebelDriverTest, WriteForwardsClampedJogAndWatchdogZeroesIt)
{
  FakeCriServer server;
  StartDriver(server);
  ASSERT_TRUE(WaitFor([&]() {return server.Commands().size() >= 5;}, 2000ms));

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
  ASSERT_TRUE(WaitFor([&]() {return server.Commands().size() >= 5;}, 2000ms));

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
  ASSERT_TRUE(WaitFor([&]() {return server.Commands(1).size() >= 5;}, 2000ms));

  server.DropClient();
  ASSERT_TRUE(WaitFor([&]() {return server.Connections() >= 2;}, 3000ms));
  ASSERT_TRUE(WaitFor([&]() {return server.Commands(2).size() >= 5;}, 3000ms));
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
  ASSERT_TRUE(WaitFor([&]() {return server.Commands(1).size() >= 5;}, 2000ms));
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
  ASSERT_TRUE(WaitFor([&]() {return server.Commands(1).size() >= 10;}, 2000ms));
  const auto allCommandsConn1 = server.Commands(1);
  const std::vector<std::string> resent(
    allCommandsConn1.end() - 5, allCommandsConn1.end());
  EXPECT_EQ(resent, kStartupSequence);
  EXPECT_EQ(server.Connections(), 1u);  // no socket reconnect happened

  EXPECT_EQ(rebel->read(now, period), return_type::OK);
}

TEST_F(RebelDriverTest, ActivateRefusesWhileStatusStillStale)
{
  FakeCriServer server;
  StartDriver(server);
  ASSERT_TRUE(WaitFor([&]() {return server.Commands(1).size() >= 5;}, 2000ms));

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
  ASSERT_TRUE(WaitFor([&]() {return server.Commands(1).size() >= 5;}, 2000ms));
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

  ASSERT_TRUE(WaitFor([&]() {return server.Commands(1).size() >= 10;}, 2000ms));
  const auto all = server.Commands(1);
  const std::vector<std::string> resent(all.end() - 5, all.end());
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
  ASSERT_TRUE(WaitFor([&]() {return server.Commands(1).size() >= 5;}, 2000ms));
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
  ASSERT_TRUE(WaitFor([&]() {return server.Commands(1).size() >= 5;}, 2000ms));
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
  ASSERT_TRUE(WaitFor([&]() {return server.Commands(1).size() >= 5;}, 2000ms));
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
  ASSERT_TRUE(WaitFor([&]() {return server.Commands(1).size() >= 5;}, 2000ms));

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

TEST_F(RebelDriverTest, ResetAndEnableRefusesWhenStatusStale)
{
  FakeCriServer server;
  StartDriver(server);
  ASSERT_TRUE(WaitFor([&]() {return server.Commands(1).size() >= 5;}, 2000ms));

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
  ASSERT_TRUE(WaitFor([&]() {return server.Commands(1).size() >= 5;}, 2000ms));

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
  ASSERT_TRUE(WaitFor([&]() {return server.Commands().size() >= 5;}, 2000ms));

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
  ASSERT_TRUE(WaitFor([&]() {return left.server->Commands().size() >= 5;}, 2000ms));
  ASSERT_TRUE(WaitFor([&]() {return right.server->Commands().size() >= 5;}, 2000ms));
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

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  ::testing::InitGoogleTest(&argc, argv);
  const int result = RUN_ALL_TESTS();
  rclcpp::shutdown();
  return result;
}
