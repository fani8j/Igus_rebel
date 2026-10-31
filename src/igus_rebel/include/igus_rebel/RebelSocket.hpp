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

  // A complete message body with the time its bytes were read from the socket (the receive thread
  // stamps it, not the thread that later processes the queue: a backlog must not make old data
  // look fresh). All messages framed from one recv() share the stamp.
  struct ReceivedMessage
  {
    std::string text;
    std::chrono::steady_clock::time_point received{};
    std::int64_t systemNs = 0;  // system_clock ns since the Unix epoch at the same instant
  };

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
  bool WaitForMessage(ReceivedMessage & message, std::chrono::milliseconds timeout);
  // Number of queued messages discarded (oldest first) because the consumer fell behind
  // (kMaxUnprocessedMessages). A robot message (above all a MOVETOEXECERROR) may be among them.
  std::uint64_t DroppedMessageCount() const {return droppedMessages.load();}

  // Returns false if not connected or the send failed (the connection is then shut down
  // and the receive thread reconnects).
  bool SendMessage(const std::string &);

  enum class SendResult {kSent, kNotValid, kFailed};
  // Sends only if the connection is still number `expectedConnectionCount` and `stillValid()`
  // (called with the write lock held, right before the bytes go out) returns true. The check and
  // the send are atomic with respect to every other sender and to a connection change (which
  // swaps the descriptor and the count under the same lock): a frame prepared for an old
  // connection or an old driver state can never reach the wire later. `stillValid` must be
  // short, lock-free and must not call back into the socket.
  SendResult SendMessageIf(
    const std::string & msg, std::uint64_t expectedConnectionCount,
    const std::function<bool()> & stillValid);

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
  std::deque<ReceivedMessage> unprocessedMessages;
  std::atomic<std::uint64_t> droppedMessages{0};
  bool stopRequested = false;  // guarded by messageLock

  CriFrameParser framer;  // receive thread only

  rclcpp::Logger logger;
  rclcpp::Clock throttleClock{RCL_STEADY_TIME};

  bool TryConnect();
  void CloseSocket();
  void PushMessages(std::vector<std::string> && messages);  // stamps the receive time
  void ReceiveThreadFunction();
  void InterruptibleSleep(std::chrono::milliseconds duration);
  SendResult SendLocked(
    const std::string & msg, const std::function<bool()> * stillValid,
    std::uint64_t expectedConnectionCount);
};
}  // namespace Igus

#endif
