#include "igus_rebel/RebelSocket.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace Igus
{
namespace
{
timeval ToTimeval(std::chrono::milliseconds duration)
{
  timeval tv{};
  tv.tv_sec = static_cast<time_t>(duration.count() / 1000);
  tv.tv_usec = static_cast<suseconds_t>((duration.count() % 1000) * 1000);
  return tv;
}
}  // namespace

RebelSocket::RebelSocket(const std::string & ip, const int & port, const rclcpp::Logger & logger)
: ip(ip),
  port(port),
  logger(logger)
{
}

RebelSocket::~RebelSocket()
{
  Stop();
}

void RebelSocket::SetConnectedCallback(ConnectedCallback callback)
{
  connectedCallback = std::move(callback);
}

bool RebelSocket::TryConnect()
{
  const int fd = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
  if (fd < 0) {
    RCLCPP_ERROR_THROTTLE(
      logger, throttleClock, 5000, "Socket creation error: %s", std::strerror(errno));
    return false;
  }

  const timeval sendTimeout = ToTimeval(kSendTimeout);
  const timeval receiveTimeout = ToTimeval(kReceiveTimeout);
  const int enable = 1;
  const int keepIdleS = 2;
  const int keepIntervalS = 1;
  const int keepCount = 3;
  const bool optionsOk =
    ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &sendTimeout, sizeof(sendTimeout)) == 0 &&
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &receiveTimeout, sizeof(receiveTimeout)) == 0 &&
    ::setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &enable, sizeof(enable)) == 0 &&
    ::setsockopt(fd, IPPROTO_TCP, TCP_KEEPIDLE, &keepIdleS, sizeof(keepIdleS)) == 0 &&
    ::setsockopt(fd, IPPROTO_TCP, TCP_KEEPINTVL, &keepIntervalS, sizeof(keepIntervalS)) == 0 &&
    ::setsockopt(fd, IPPROTO_TCP, TCP_KEEPCNT, &keepCount, sizeof(keepCount)) == 0 &&
    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &enable, sizeof(enable)) == 0;
  if (!optionsOk) {
    RCLCPP_ERROR_THROTTLE(
      logger, throttleClock, 5000, "Failed to set socket options: %s", std::strerror(errno));
    ::close(fd);
    return false;
  }

  sockaddr_in serverAddress{};
  serverAddress.sin_family = AF_INET;
  serverAddress.sin_port = htons(static_cast<uint16_t>(port));
  if (::inet_pton(AF_INET, ip.c_str(), &serverAddress.sin_addr) <= 0) {
    RCLCPP_ERROR_THROTTLE(
      logger, throttleClock, 5000, "Invalid robot IP address / Address not supported: %s",
      ip.c_str());
    ::close(fd);
    return false;
  }

  if (::connect(fd, reinterpret_cast<sockaddr *>(&serverAddress), sizeof(serverAddress)) < 0) {
    RCLCPP_ERROR_THROTTLE(
      logger, throttleClock, 5000, "Connection to ReBeL at %s:%d failed: %s", ip.c_str(), port,
      std::strerror(errno));
    ::close(fd);
    return false;
  }

  {
    std::lock_guard<std::mutex> lockGuard(socketWriteLock);
    sock = fd;
  }
  framer.Reset();
  connected = true;
  ++connectionCount;
  RCLCPP_INFO(logger, "Connected to ReBeL at %s:%d", ip.c_str(), port);
  return true;
}

void RebelSocket::CloseSocket()
{
  std::lock_guard<std::mutex> lockGuard(socketWriteLock);
  connected = false;
  if (sock >= 0) {
    ::shutdown(sock, SHUT_RDWR);
    ::close(sock);
    sock = -1;
  }
}

void RebelSocket::PushMessages(std::vector<std::string> && messages)
{
  if (messages.empty()) {
    return;
  }

  std::size_t dropped = 0;
  {
    std::lock_guard<std::mutex> lockGuard(messageLock);
    for (auto & message : messages) {
      // Keep the newest messages if the message thread falls behind.
      if (unprocessedMessages.size() >= kMaxUnprocessedMessages) {
        unprocessedMessages.pop_front();
        ++dropped;
      }
      unprocessedMessages.push_back(std::move(message));
    }
  }
  messageCondition.notify_one();

  if (dropped > 0) {
    RCLCPP_WARN_THROTTLE(
      logger, throttleClock, 5000,
      "Robot messages are not processed fast enough. Discarded %zu old message(s).", dropped);
  }
}

void RebelSocket::InterruptibleSleep(std::chrono::milliseconds duration)
{
  const auto deadline = std::chrono::steady_clock::now() + duration;
  while (continueReceive && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

void RebelSocket::ReceiveThreadFunction()
{
  RCLCPP_DEBUG(logger, "Starting to receive messages from robot.");

  char buffer[4096];

  while (continueReceive) {
    if (!connected) {
      if (!reconnectAllowed) {
        InterruptibleSleep(kReconnectDelay);
        continue;
      }
      const bool reconnect = connectionCount.load() > 0;
      if (!TryConnect()) {
        InterruptibleSleep(kReconnectDelay);
        continue;
      }
      if (connectedCallback && reconnectAllowed && continueReceive) {
        try {
          connectedCallback(reconnect);
        } catch (const std::exception & e) {
          RCLCPP_ERROR(logger, "Connected callback failed: %s", e.what());
        }
      }
      continue;
    }

    const ssize_t received = ::recv(sock, buffer, sizeof(buffer), 0);
    if (received > 0) {
      PushMessages(framer.Feed(buffer, static_cast<std::size_t>(received)));
      const std::size_t discarded = framer.TakeDiscardedBytes();
      if (discarded > 0) {
        RCLCPP_WARN_THROTTLE(
          logger, throttleClock, 5000, "Discarded %zu byte(s) of unframed CRI data", discarded);
      }
    } else if (received == 0) {
      if (continueReceive) {
        RCLCPP_WARN(logger, "Connection to ReBeL closed; reconnecting");
      }
      CloseSocket();
    } else if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) {
      continue;
    } else {
      if (continueReceive) {
        RCLCPP_ERROR(
          logger, "Receiving from ReBeL failed: %s; reconnecting", std::strerror(errno));
      }
      CloseSocket();
    }
  }

  CloseSocket();
  RCLCPP_DEBUG(logger, "Stopped to receive messages from robot.");
}

void RebelSocket::Start()
{
  if (receiveThread.joinable()) {
    return;
  }
  {
    std::lock_guard<std::mutex> lockGuard(messageLock);
    stopRequested = false;
  }
  continueReceive = true;
  reconnectAllowed = true;
  receiveThread = std::thread(&RebelSocket::ReceiveThreadFunction, this);
}

void RebelSocket::DisableReconnect()
{
  reconnectAllowed = false;
}

void RebelSocket::Stop()
{
  reconnectAllowed = false;
  continueReceive = false;

  {
    // Wake a blocking recv(); the receive thread closes the descriptor.
    std::lock_guard<std::mutex> lockGuard(socketWriteLock);
    if (sock >= 0) {
      ::shutdown(sock, SHUT_RDWR);
    }
  }

  if (receiveThread.joinable()) {
    receiveThread.join();
  }

  {
    std::lock_guard<std::mutex> lockGuard(messageLock);
    stopRequested = true;
  }
  messageCondition.notify_all();
}

bool RebelSocket::HasMessage()
{
  std::lock_guard<std::mutex> lockGuard(messageLock);
  return !unprocessedMessages.empty();
}

std::string RebelSocket::GetMessage()
{
  std::lock_guard<std::mutex> lockGuard(messageLock);
  if (unprocessedMessages.empty()) {
    return "";
  }
  std::string msg = std::move(unprocessedMessages.front());
  unprocessedMessages.pop_front();
  return msg;
}

bool RebelSocket::WaitForMessage(std::string & message, std::chrono::milliseconds timeout)
{
  std::unique_lock<std::mutex> lock(messageLock);
  messageCondition.wait_for(
    lock, timeout, [this]() {return !unprocessedMessages.empty() || stopRequested;});
  if (unprocessedMessages.empty()) {
    return false;
  }
  message = std::move(unprocessedMessages.front());
  unprocessedMessages.pop_front();
  return true;
}

bool RebelSocket::SendMessage(const std::string & msg)
{
  std::lock_guard<std::mutex> lockGuard(socketWriteLock);

  if (sock < 0 || !connected) {
    return false;
  }

  std::size_t sent = 0;
  while (sent < msg.size()) {
    const ssize_t result = ::send(sock, msg.data() + sent, msg.size() - sent, MSG_NOSIGNAL);
    if (result > 0) {
      sent += static_cast<std::size_t>(result);
      continue;
    }
    if (result < 0 && errno == EINTR) {
      continue;
    }
    RCLCPP_ERROR_THROTTLE(
      logger, throttleClock, 1000, "Sending to ReBeL failed: %s; reconnecting",
      result < 0 ? std::strerror(errno) : "no bytes written");
    // Let the receive thread notice, close and reconnect.
    ::shutdown(sock, SHUT_RDWR);
    return false;
  }
  return true;
}
}  // namespace Igus
