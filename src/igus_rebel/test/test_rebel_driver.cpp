// Exercises RebelSocket and the Rebel hardware interface against a fake CRI server on
// 127.0.0.1. Mock hardware never runs Rebel.cpp, so this is the only automated coverage
// of the real driver code paths. The real controller address is never used here.

#include <gtest/gtest.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
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
      s << " 0";
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

const std::vector<std::string> kStartupSequence = {
  "SetActive true", "Reset", "Enable", "MotionTypeJoint", "Override 100.0"};

class RebelDriverTest : public ::testing::Test
{
protected:
  void StartDriver(FakeCriServer & server)
  {
    rebel = std::make_unique<Igus::Rebel>("127.0.0.1", server.port());
    ASSERT_EQ(rebel->on_init(hardware_interface::HardwareInfo{}), CallbackReturn::SUCCESS);
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
  EXPECT_EQ(rebel->read(now, period), return_type::OK);
  ASSERT_TRUE(
    WaitFor([&]() {return rebel->read(now, period) == return_type::ERROR;}, 1000ms));
  const auto elapsed = Clock::now() - stopped;
  EXPECT_GE(elapsed, 190ms);
  EXPECT_LT(elapsed, 400ms);
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
  EXPECT_EQ(rebel.on_init(hardware_interface::HardwareInfo{}), CallbackReturn::ERROR);
  const auto elapsed = Clock::now() - start;
  EXPECT_GE(elapsed, Igus::Safety::kFirstStatusTimeout - 100ms);
  EXPECT_LT(elapsed, Igus::Safety::kFirstStatusTimeout + 2500ms);
  // on_init already stopped the driver: nothing is sent any more.
  const auto framesAfterInit = server.Frames().size();
  std::this_thread::sleep_for(100ms);
  EXPECT_EQ(server.Frames().size(), framesAfterInit);
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
  hardware_interface::HardwareInfo info;
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
    hardware_interface::HardwareInfo info;
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
  EXPECT_EQ(rebel.on_init(hardware_interface::HardwareInfo{}), CallbackReturn::ERROR);
  EXPECT_LT(Clock::now() - start, Igus::Safety::kFirstStatusTimeout + 2500ms);
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
