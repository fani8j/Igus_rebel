#ifndef REBEL_SOCKET_HPP_
#define REBEL_SOCKET_HPP_

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "rclcpp/rclcpp.hpp"

#include "igus_rebel/CriFraming.hpp"

namespace Igus
{
// TCP client for the CRI interface.
//
// A single receive thread owns (re)connecting, reading and closing the socket. Complete
// message bodies are queued for GetMessage()/WaitForMessage(). SendMessage() never blocks
// on (re)connecting: while disconnected it returns false immediately, so shutdown cannot
// hang on an unreachable robot. After every successful connect the connected callback
// runs on the receive thread (used to re-send the start-up command sequence).
class RebelSocket
{
public:
  using ConnectedCallback = std::function<void (bool reconnect)>;

  // Socket timeouts: connect() and send() are bounded by kSendTimeout (Linux applies
  // SO_SNDTIMEO to connect), recv() wakes up every kReceiveTimeout to check for Stop().
  static constexpr std::chrono::milliseconds kSendTimeout{1000};
  static constexpr std::chrono::milliseconds kReceiveTimeout{100};
  static constexpr std::chrono::milliseconds kReconnectDelay{100};
  static constexpr std::size_t kMaxUnprocessedMessages = 25;

  // `logger` lets several sockets (one per arm) be told apart in the log; the default keeps the
  // historical "igus_rebel" logger.
  RebelSocket(
    const std::string & ip, const int & port,
    const rclcpp::Logger & logger = rclcpp::get_logger("igus_rebel"));
  ~RebelSocket();

  RebelSocket(const RebelSocket &) = delete;
  RebelSocket & operator=(const RebelSocket &) = delete;

  // Must be set before Start().
  void SetConnectedCallback(ConnectedCallback callback);

  void Start();
  // Stops further (re)connect attempts and callbacks; an open connection stays usable.
  void DisableReconnect();
  // Stops the receive thread and closes the socket. Bounded by kSendTimeout + kReceiveTimeout.
  void Stop();

  bool IsConnected() const {return connected.load();}
  std::uint64_t ConnectionCount() const {return connectionCount.load();}

  bool HasMessage();
  std::string GetMessage();
  // Waits up to timeout for a message. Returns false on timeout or after Stop().
  bool WaitForMessage(std::string & message, std::chrono::milliseconds timeout);

  // Returns false if not connected or the send failed (the connection is then shut down
  // and the receive thread reconnects).
  bool SendMessage(const std::string &);

private:
  std::string ip;
  int port;

  // Written only by the receive thread, under socketWriteLock. The receive thread reads it
  // without the lock (it is the only writer); senders read it under the lock.
  int sock = -1;
  std::mutex socketWriteLock;

  std::atomic<bool> continueReceive{false};
  std::atomic<bool> reconnectAllowed{false};
  std::atomic<bool> connected{false};
  std::atomic<std::uint64_t> connectionCount{0};
  std::thread receiveThread;

  ConnectedCallback connectedCallback;

  std::mutex messageLock;
  std::condition_variable messageCondition;
  std::deque<std::string> unprocessedMessages;
  bool stopRequested = false;  // guarded by messageLock

  CriFrameParser framer;  // receive thread only

  rclcpp::Logger logger;
  rclcpp::Clock throttleClock{RCL_STEADY_TIME};

  bool TryConnect();
  void CloseSocket();
  void PushMessages(std::vector<std::string> && messages);
  void ReceiveThreadFunction();
  void InterruptibleSleep(std::chrono::milliseconds duration);
};
}  // namespace Igus

#endif
