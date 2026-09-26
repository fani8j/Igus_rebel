#ifndef CRI_MESSAGE_HPP_
#define CRI_MESSAGE_HPP_

#include <array>
#include <cstdint>
#include <string>
#include <variant>

// Classes to parse the messages the robot may send and to store the parse result.
// Each message type has a constructor that takes a message string as sent from the robot
// (without the CRISTART and CRIEND) which then parses the string and saves what it finds
// in the class members.
//
// There is a base class (CriMessage) which constains a static helper method
// (CriMessage::GetMessageType) to determine a message's type.
//
// ParseMessage() combines both steps and never throws; use it from the message thread.

namespace Igus
{
namespace CriMessages
{
enum class MessageType
{
  STATUS,
  OPINFO,
  GSIG,
  GRIPPERSTATE,
  RUNSTATE,
  MESSAGE,
  CMD,
  CONFIG,
  INFO,
  LOGMSG,
  VARIABLES,
  CMDERROR,
  CMDACK,
  CYCLESTAT,
  UNKNOWN = 1000
};

enum class Mode
{
  JOINT,
  CARTBASE,
  CARTTOOL,
  PLATFORM,
  FSM,
  UNKNOWN = 1000
};

enum class Kinstate
{
  NO_ERROR = 0,
  JOINT_LIMIT_MIN = 13,
  JOINT_LIMIT_MAX = 14,
  CARTESIAN_SINGULARITY_CENTER = 21,
  CARTESIAN_SINGULARITY_REACH = 23,
  CARTESIAN_SINGULARITY_WRIST = 24,
  TOOL_AT_VIRTUAL_BOX_LIMIT_1 = 30,
  TOOL_AT_VIRTUAL_BOX_LIMIT_2 = 31,
  TOOL_AT_VIRTUAL_BOX_LIMIT_3 = 32,
  TOOL_AT_VIRTUAL_BOX_LIMIT_4 = 33,
  TOOL_AT_VIRTUAL_BOX_LIMIT_5 = 34,
  TOOL_AT_VIRTUAL_BOX_LIMIT_6 = 35,
  MOTION_NOT_ALLOWED = 99,
  UNKNOWN = 1000
};

enum class ErrorJoint
{
  TEMP = 1,         // OverTemperature
  ESTOP_LOWV = 2,   // Energy supply too low
  MNE = 4,          // Motor not enabled
  COM = 8,          // Communication watch dog
  POS = 16,         // Position lag
  ENC = 32,         // Encoder Error
  OC = 64,          // Overcurrent
  DRV = 128         // Driver Error
};

enum class LogLevel
{
  DEBUG,
  APP_INFO,
  APP_ERROR,
  INFO,
  WARN,
  ERROR,
  FATAL,
  UNKNOWN = 1000
};

class CriMessage
{
public:
  // Returns UNKNOWN for empty or unrecognised input; never throws.
  static MessageType GetMessageType(const std::string &);

protected:
  static std::string ParseMessageString(
    const std::string &, const std::string::size_type &,
    const std::string::size_type &, const std::string::size_type &);

  template<class T, std::size_t N>
  static bool FillArray(std::array<T, N> &, const std::string &);
};

class Status : public CriMessage
{
public:
  Mode mode;
  std::array<float, 16> posJointSetPoint;
  std::array<float, 16> posJointCurrent;
  std::array<float, 6> posCartRobot;
  std::array<float, 3> posCartPlattform;
  float overrideValue;
  std::uint64_t din;
  std::uint64_t dout;
  int eStop;
  int supply;
  int currentall;
  std::array<int, 16> currentjoints;
  std::string errorSummary;
  std::array<int, 16> errorJoints;
  Kinstate kinstate;

  explicit Status(const std::string &);
  Status();
  bool IsValid() const {return valid;}
  static Kinstate KinstateFromInt(int);

private:
  bool valid;
  static Mode GetMode(const std::string &);
};

class Message : public CriMessage
{
public:
  explicit Message(const std::string &);

  std::string message;
};

class Command : public CriMessage
{
public:
  explicit Command(const std::string &);

  std::string command;
};

class Info : public CriMessage
{
public:
  explicit Info(const std::string &);

  std::string info;
};

class LogMsg : public CriMessage
{
public:
  explicit LogMsg(const std::string &);

  LogLevel logLevel;
  long timestamp;
  std::string logMsg;

private:
  static LogLevel GetLogLevel(const std::string &);
};

class CmdError : public CriMessage
{
public:
  explicit CmdError(const std::string &);
  int recjectedCmd;
  std::string error;
};

class CmdAck : public CriMessage
{
public:
  explicit CmdAck(const std::string &);
  int acceptedCmd;
};

class Cyclestat : public CriMessage
{
public:
  explicit Cyclestat(const std::string &);
  int cycletime;
  int workload;
};

// Result of ParseMessage(). `ok` is false if the message could not be parsed (the
// parser threw or the STATUS was invalid); `error` then holds a short reason.
// `payload` holds the parsed object for the types the driver consumes and
// std::monostate for types that are only recognised (RUNSTATE, OPINFO, ...).
using Payload = std::variant<
  std::monostate, Status, Message, Command, Info, LogMsg, CmdError, CmdAck, Cyclestat>;

struct ParsedMessage
{
  MessageType type = MessageType::UNKNOWN;
  bool ok = false;
  std::string error;
  Payload payload;
};

// Classifies and parses one framed CRI message body. Never throws.
ParsedMessage ParseMessage(const std::string & msg) noexcept;

// Human readable KINSTATE description ("no error" for NO_ERROR).
std::string KinstateToString(Kinstate kinstate);

// Human readable description of the set bits of one joint's ERROR mask
// (e.g. "'Overtemperature' 'Position lag' "). Empty if no known bit is set.
std::string DescribeJointErrorBits(int errorBits);
}  // namespace CriMessages
}  // namespace Igus

#endif
