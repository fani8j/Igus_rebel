#include <cctype>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "igus_rebel/CriMessages.hpp"
#include "igus_rebel/CriKeywords.hpp"

namespace
{
bool ParseFloat(const std::string & text, float & value)
{
  try {
    std::size_t parsed = 0;
    value = std::stof(text, &parsed);
    return parsed == text.size();
  } catch (const std::exception &) {
    return false;
  }
}

bool ParseInt(const std::string & text, int & value)
{
  try {
    std::size_t parsed = 0;
    value = std::stoi(text, &parsed);
    return parsed == text.size();
  } catch (const std::exception &) {
    return false;
  }
}

bool ParseLong(const std::string & text, long & value)
{
  try {
    std::size_t parsed = 0;
    value = std::stol(text, &parsed);
    return parsed == text.size();
  } catch (const std::exception &) {
    return false;
  }
}

bool ParseHexMask(const std::string & text, std::uint64_t & value)
{
  // std::stoull accepts a leading '-' and wraps; a digital mask is never negative.
  if (text.empty() || text[0] == '-' || text[0] == '+') {
    return false;
  }
  try {
    std::size_t parsed = 0;
    value = std::stoull(text, &parsed, 16);
    return parsed == text.size();
  } catch (const std::exception &) {
    return false;
  }
}

bool IsSpace(char c)
{
  return std::isspace(static_cast<unsigned char>(c)) != 0;
}

std::string Trim(const std::string & text)
{
  std::string::size_type begin = 0;
  std::string::size_type end = text.size();
  while (begin < end && IsSpace(text[begin])) {
    ++begin;
  }
  while (end > begin && IsSpace(text[end - 1])) {
    --end;
  }
  return text.substr(begin, end - begin);
}

// Splits a message into whitespace separated tokens, remembering where each token starts.
std::vector<std::pair<std::string, std::string::size_type>> Tokenize(const std::string & text)
{
  std::vector<std::pair<std::string, std::string::size_type>> tokens;
  std::string::size_type i = 0;
  while (i < text.size()) {
    while (i < text.size() && IsSpace(text[i])) {
      ++i;
    }
    if (i >= text.size()) {
      break;
    }
    const std::string::size_type start = i;
    while (i < text.size() && !IsSpace(text[i])) {
      ++i;
    }
    tokens.emplace_back(text.substr(start, i - start), start);
  }
  return tokens;
}

// A CRI message body is "<counter> <TYPE> <payload...>". Returns the payload after the
// type token (left-trimmed) or throws if the message does not have that shape.
std::string PayloadAfterType(const std::string & msg, const std::string & type)
{
  const auto tokens = Tokenize(msg);
  for (std::size_t i = 0; i < tokens.size() && i < 2; ++i) {
    if (tokens[i].first == type) {
      const std::string::size_type payloadStart = tokens[i].second + type.size();
      return Trim(msg.substr(payloadStart));
    }
  }
  throw std::invalid_argument("missing " + type + " type token");
}
}  // namespace

namespace Igus
{
namespace CriMessages
{
MessageType CriMessage::GetMessageType(const std::string & msg)
{
  // A message is "<counter> <TYPE> ...". Sometimes there is an additional leading space.
  const auto tokens = Tokenize(msg);
  if (tokens.size() < 2) {
    return MessageType::UNKNOWN;
  }
  const std::string & typeString = tokens[1].first;

  static const std::pair<const std::string *, MessageType> kTypes[] = {
    {&CriKeywords::TYPE_STATUS, MessageType::STATUS},
    {&CriKeywords::TYPE_OPINFO, MessageType::OPINFO},
    {&CriKeywords::TYPE_GSIG, MessageType::GSIG},
    {&CriKeywords::TYPE_GRIPPERSTATE, MessageType::GRIPPERSTATE},
    {&CriKeywords::TYPE_RUNSTATE, MessageType::RUNSTATE},
    {&CriKeywords::TYPE_MESSAGE, MessageType::MESSAGE},
    {&CriKeywords::TYPE_CMD, MessageType::CMD},
    {&CriKeywords::TYPE_CONFIG, MessageType::CONFIG},
    {&CriKeywords::TYPE_INFO, MessageType::INFO},
    {&CriKeywords::TYPE_LOGMSG, MessageType::LOGMSG},
    {&CriKeywords::TYPE_VARIABLES, MessageType::VARIABLES},
    {&CriKeywords::TYPE_CMDERROR, MessageType::CMDERROR},
    {&CriKeywords::TYPE_CMDACK, MessageType::CMDACK},
    {&CriKeywords::TYPE_CYCLESTAT, MessageType::CYCLESTAT},
  };

  for (const auto & entry : kTypes) {
    if (typeString == *entry.first) {
      return entry.second;
    }
  }
  return MessageType::UNKNOWN;
}

std::string CriMessage::ParseMessageString(
  const std::string & statusString, const std::string::size_type & begin,
  const std::string::size_type & end, const std::string::size_type & keywordLength)
{
  // Value between "<KEYWORD>" (starting at begin) and the next keyword (starting at end).
  const std::string::size_type start = begin + keywordLength;
  if (start >= end || start >= statusString.size()) {
    return std::string();
  }
  return Trim(statusString.substr(start, end - start));
}

template<class T, std::size_t N>
bool CriMessage::FillArray(std::array<T, N> & array, const std::string & spaceSeparatedValues)
{
  std::istringstream stream(spaceSeparatedValues);
  std::string token;
  std::size_t index = 0;

  while (stream >> token) {
    if (index >= N) {
      return false;
    }

    bool parsed = false;
    if constexpr (std::is_same_v<T, float>) {
      parsed = ParseFloat(token, array[index]);
    } else if constexpr (std::is_same_v<T, int>) {
      parsed = ParseInt(token, array[index]);
    }

    if (!parsed) {
      return false;
    }
    ++index;
  }

  return index == N;
}

Status::Status(const std::string & messageString)
: Status()
{
  const auto find = [&messageString](const std::string & keyword) {
      return messageString.find(keyword);
    };

  const std::string::size_type modeStart = find(CriKeywords::STATUS_MODE);
  const std::string::size_type posJointSetPointStart = find(CriKeywords::STATUS_POSJOINTSETPOINT);
  const std::string::size_type posJointCurrentStart = find(CriKeywords::STATUS_POSJOINTCURRENT);
  const std::string::size_type posCartRobotStart = find(CriKeywords::STATUS_POSCARTROBOT);
  const std::string::size_type posCartPlattformStart = find(CriKeywords::STATUS_POSCARTPLATTFORM);
  const std::string::size_type overrideValueStart = find(CriKeywords::STATUS_OVERRIDE);
  const std::string::size_type dinStart = find(CriKeywords::STATUS_DIN);
  const std::string::size_type doutStart = find(CriKeywords::STATUS_DOUT);
  const std::string::size_type eStopStart = find(CriKeywords::STATUS_ESTOP);
  const std::string::size_type supplyStart = find(CriKeywords::STATUS_SUPPLY);
  const std::string::size_type currentallStart = find(CriKeywords::STATUS_CURRENTALL);
  const std::string::size_type currentjointsStart = find(CriKeywords::STATUS_CURRENTJOINTS);
  const std::string::size_type errorStart = find(CriKeywords::STATUS_ERROR);
  const std::string::size_type kinstateStart = find(CriKeywords::STATUS_KINSTATE);

  // npos is the largest value, so a missing keyword breaks the strict ordering below.
  if (
    !(
      (modeStart < posJointSetPointStart) &&
      (posJointSetPointStart < posJointCurrentStart) &&
      (posJointCurrentStart < posCartRobotStart) &&
      (posCartRobotStart < posCartPlattformStart) &&
      (posCartPlattformStart < overrideValueStart) &&
      (overrideValueStart < dinStart) &&
      (dinStart < doutStart) &&
      (doutStart < eStopStart) &&
      (eStopStart < supplyStart) &&
      (supplyStart < currentallStart) &&
      (currentallStart < currentjointsStart) &&
      (currentjointsStart < errorStart) &&
      (errorStart < kinstateStart) &&
      (kinstateStart != std::string::npos)))
  {
    return;
  }

  const std::string modeString = ParseMessageString(
    messageString, modeStart, posJointSetPointStart, CriKeywords::STATUS_MODE.size());
  const std::string posJointSetPointString = ParseMessageString(
    messageString, posJointSetPointStart, posJointCurrentStart,
    CriKeywords::STATUS_POSJOINTSETPOINT.size());
  const std::string posJointCurrentString = ParseMessageString(
    messageString, posJointCurrentStart, posCartRobotStart,
    CriKeywords::STATUS_POSJOINTCURRENT.size());
  const std::string posCartRobotString = ParseMessageString(
    messageString, posCartRobotStart, posCartPlattformStart,
    CriKeywords::STATUS_POSCARTROBOT.size());
  const std::string posCartPlattformString = ParseMessageString(
    messageString, posCartPlattformStart, overrideValueStart,
    CriKeywords::STATUS_POSCARTPLATTFORM.size());
  const std::string overrideValueString = ParseMessageString(
    messageString, overrideValueStart, dinStart, CriKeywords::STATUS_OVERRIDE.size());
  const std::string dinString = ParseMessageString(
    messageString, dinStart, doutStart, CriKeywords::STATUS_DIN.size());
  const std::string doutString = ParseMessageString(
    messageString, doutStart, eStopStart, CriKeywords::STATUS_DOUT.size());
  const std::string eStopString = ParseMessageString(
    messageString, eStopStart, supplyStart, CriKeywords::STATUS_ESTOP.size());
  const std::string supplyString = ParseMessageString(
    messageString, supplyStart, currentallStart, CriKeywords::STATUS_SUPPLY.size());
  const std::string currentallString = ParseMessageString(
    messageString, currentallStart, currentjointsStart, CriKeywords::STATUS_CURRENTALL.size());
  const std::string currentjointsString = ParseMessageString(
    messageString, currentjointsStart, errorStart, CriKeywords::STATUS_CURRENTJOINTS.size());
  const std::string errorString = ParseMessageString(
    messageString, errorStart, kinstateStart, CriKeywords::STATUS_ERROR.size());

  // KINSTATE is followed by firmware-dependent fields (OPMODE, CARTSPEED, ...); its value is
  // the first token after the keyword.
  std::istringstream kinstateStream(
    messageString.substr(kinstateStart + CriKeywords::STATUS_KINSTATE.size()));
  std::string kinstateString;
  kinstateStream >> kinstateString;

  const std::string::size_type errorSummaryEnd = errorString.find(' ');
  if (errorSummaryEnd == std::string::npos) {
    return;
  }
  errorSummary = errorString.substr(0, errorSummaryEnd);
  const std::string errorJointsString = errorString.substr(errorSummaryEnd + 1);

  int kinstateValue = 0;
  mode = GetMode(modeString);
  const bool parsed =
    mode != Mode::UNKNOWN &&
    FillArray(posJointSetPoint, posJointSetPointString) &&
    FillArray(posJointCurrent, posJointCurrentString) &&
    FillArray(posCartRobot, posCartRobotString) &&
    FillArray(posCartPlattform, posCartPlattformString) &&
    ParseFloat(overrideValueString, overrideValue) &&
    ParseHexMask(dinString, din) &&
    ParseHexMask(doutString, dout) &&
    ParseInt(eStopString, eStop) &&
    ParseInt(supplyString, supply) &&
    ParseInt(currentallString, currentall) &&
    FillArray(currentjoints, currentjointsString) &&
    FillArray(errorJoints, errorJointsString) &&
    ParseInt(kinstateString, kinstateValue);

  if (!parsed) {
    return;
  }

  kinstate = KinstateFromInt(kinstateValue);
  valid = true;
}

Status::Status()
{
  mode = Mode::UNKNOWN;
  posJointSetPoint.fill(0.0f);
  posJointCurrent.fill(0.0f);
  posCartRobot.fill(0.0f);
  posCartPlattform.fill(0.0f);
  overrideValue = 0.0f;
  din = 0;
  dout = 0;
  eStop = 0;
  supply = 0;
  currentall = 0;
  currentjoints.fill(0);
  errorSummary = "NotInitialized";
  errorJoints.fill(0);
  valid = false;
  kinstate = Kinstate::UNKNOWN;
}

Mode Status::GetMode(const std::string & modeString)
{
  if (modeString == CriKeywords::MODE_JOINT) {
    return Mode::JOINT;
  }
  if (modeString == CriKeywords::MODE_CARTBASE) {
    return Mode::CARTBASE;
  }
  if (modeString == CriKeywords::MODE_CARTTOOL) {
    return Mode::CARTTOOL;
  }
  if (modeString == CriKeywords::MODE_PLATFORM) {
    return Mode::PLATFORM;
  }
  if (modeString == CriKeywords::MODE_FSM) {
    return Mode::FSM;
  }
  return Mode::UNKNOWN;
}

Kinstate Status::KinstateFromInt(int kinstateInt)
{
  switch (kinstateInt) {
    case static_cast<int>(Kinstate::NO_ERROR):
    case static_cast<int>(Kinstate::JOINT_LIMIT_MIN):
    case static_cast<int>(Kinstate::JOINT_LIMIT_MAX):
    case static_cast<int>(Kinstate::CARTESIAN_SINGULARITY_CENTER):
    case static_cast<int>(Kinstate::CARTESIAN_SINGULARITY_REACH):
    case static_cast<int>(Kinstate::CARTESIAN_SINGULARITY_WRIST):
    case static_cast<int>(Kinstate::TOOL_AT_VIRTUAL_BOX_LIMIT_1):
    case static_cast<int>(Kinstate::TOOL_AT_VIRTUAL_BOX_LIMIT_2):
    case static_cast<int>(Kinstate::TOOL_AT_VIRTUAL_BOX_LIMIT_3):
    case static_cast<int>(Kinstate::TOOL_AT_VIRTUAL_BOX_LIMIT_4):
    case static_cast<int>(Kinstate::TOOL_AT_VIRTUAL_BOX_LIMIT_5):
    case static_cast<int>(Kinstate::TOOL_AT_VIRTUAL_BOX_LIMIT_6):
    case static_cast<int>(Kinstate::MOTION_NOT_ALLOWED):
      return static_cast<Kinstate>(kinstateInt);
    default:
      return Kinstate::UNKNOWN;
  }
}

Message::Message(const std::string & messageString)
{
  message = PayloadAfterType(messageString, CriKeywords::TYPE_MESSAGE);
}

Command::Command(const std::string & messageString)
{
  command = PayloadAfterType(messageString, CriKeywords::TYPE_CMD);
}

Info::Info(const std::string & messageString)
{
  info = PayloadAfterType(messageString, CriKeywords::TYPE_INFO);
}

LogMsg::LogMsg(const std::string & messageString)
{
  // "<counter> LOGMSG <level> <timestamp> <text...>"
  const std::string payload = PayloadAfterType(messageString, CriKeywords::TYPE_LOGMSG);
  const auto tokens = Tokenize(payload);
  if (tokens.size() < 2) {
    throw std::invalid_argument("LOGMSG without level and timestamp");
  }
  logLevel = GetLogLevel(tokens[0].first);
  if (!ParseLong(tokens[1].first, timestamp)) {
    throw std::invalid_argument("LOGMSG timestamp is not an integer");
  }
  logMsg = Trim(payload.substr(tokens[1].second + tokens[1].first.size()));
}

LogLevel LogMsg::GetLogLevel(const std::string & logLvlString)
{
  if (logLvlString == CriKeywords::LOG_LEVEL_DEBUG) {
    return LogLevel::DEBUG;
  }
  if (logLvlString == CriKeywords::LOG_LEVEL_APP_INFO) {
    return LogLevel::APP_INFO;
  }
  if (logLvlString == CriKeywords::LOG_LEVEL_APP_ERROR) {
    return LogLevel::APP_ERROR;
  }
  if (logLvlString == CriKeywords::LOG_LEVEL_INFO) {
    return LogLevel::INFO;
  }
  if (logLvlString == CriKeywords::LOG_LEVEL_WARN) {
    return LogLevel::WARN;
  }
  if (logLvlString == CriKeywords::LOG_LEVEL_ERROR) {
    return LogLevel::ERROR;
  }
  if (logLvlString == CriKeywords::LOG_LEVEL_FATAL) {
    return LogLevel::FATAL;
  }
  return LogLevel::UNKNOWN;
}

CmdError::CmdError(const std::string & messageString)
{
  // "<counter> CMDERROR <rejected counter> <error text...>"
  const std::string payload = PayloadAfterType(messageString, CriKeywords::TYPE_CMDERROR);
  const auto tokens = Tokenize(payload);
  if (tokens.empty() || !ParseInt(tokens[0].first, recjectedCmd)) {
    throw std::invalid_argument("CMDERROR without command counter");
  }
  error = Trim(payload.substr(tokens[0].second + tokens[0].first.size()));
}

CmdAck::CmdAck(const std::string & messageString)
{
  // "<counter> CMDACK <accepted counter>"
  const std::string payload = PayloadAfterType(messageString, CriKeywords::TYPE_CMDACK);
  const auto tokens = Tokenize(payload);
  if (tokens.empty() || !ParseInt(tokens[0].first, acceptedCmd)) {
    throw std::invalid_argument("CMDACK without command counter");
  }
}

Cyclestat::Cyclestat(const std::string & messageString)
{
  // "<counter> CYCLESTAT <cycletime> <workload>"
  const std::string payload = PayloadAfterType(messageString, CriKeywords::TYPE_CYCLESTAT);
  const auto tokens = Tokenize(payload);
  if (
    tokens.size() < 2 || !ParseInt(tokens[0].first, cycletime) ||
    !ParseInt(tokens[1].first, workload))
  {
    throw std::invalid_argument("CYCLESTAT without cycletime and workload");
  }
}

ParsedMessage ParseMessage(const std::string & msg) noexcept
{
  ParsedMessage result;
  try {
    result.type = CriMessage::GetMessageType(msg);
    switch (result.type) {
      case MessageType::STATUS:
        {
          Status status(msg);
          if (!status.IsValid()) {
            result.error = "malformed STATUS";
            return result;
          }
          result.payload = std::move(status);
          break;
        }
      case MessageType::MESSAGE:
        result.payload = Message(msg);
        break;
      case MessageType::CMD:
        result.payload = Command(msg);
        break;
      case MessageType::INFO:
        result.payload = Info(msg);
        break;
      case MessageType::LOGMSG:
        result.payload = LogMsg(msg);
        break;
      case MessageType::CMDERROR:
        result.payload = CmdError(msg);
        break;
      case MessageType::CMDACK:
        result.payload = CmdAck(msg);
        break;
      case MessageType::CYCLESTAT:
        result.payload = Cyclestat(msg);
        break;
      case MessageType::UNKNOWN:
        result.error = "unknown message type";
        return result;
      case MessageType::OPINFO:
      case MessageType::GSIG:
      case MessageType::GRIPPERSTATE:
      case MessageType::RUNSTATE:
      case MessageType::CONFIG:
      case MessageType::VARIABLES:
        // Recognised but not consumed by the driver.
        break;
    }
    result.ok = true;
  } catch (const std::exception & e) {
    result.ok = false;
    result.error = e.what();
  } catch (...) {
    result.ok = false;
    result.error = "unknown exception";
  }
  return result;
}

std::string KinstateToString(Kinstate kinstate)
{
  switch (kinstate) {
    case Kinstate::NO_ERROR:
      return "no error";
    case Kinstate::JOINT_LIMIT_MIN:
      return "joint at minimum limit";
    case Kinstate::JOINT_LIMIT_MAX:
      return "joint at maximum limit";
    case Kinstate::CARTESIAN_SINGULARITY_CENTER:
      return "cartesian singularity (center)";
    case Kinstate::CARTESIAN_SINGULARITY_REACH:
      return "cartesian singularity (reach)";
    case Kinstate::CARTESIAN_SINGULARITY_WRIST:
      return "cartesian singularity (wrist)";
    case Kinstate::TOOL_AT_VIRTUAL_BOX_LIMIT_1:
      return "tool at virtual box limit 1";
    case Kinstate::TOOL_AT_VIRTUAL_BOX_LIMIT_2:
      return "tool at virtual box limit 2";
    case Kinstate::TOOL_AT_VIRTUAL_BOX_LIMIT_3:
      return "tool at virtual box limit 3";
    case Kinstate::TOOL_AT_VIRTUAL_BOX_LIMIT_4:
      return "tool at virtual box limit 4";
    case Kinstate::TOOL_AT_VIRTUAL_BOX_LIMIT_5:
      return "tool at virtual box limit 5";
    case Kinstate::TOOL_AT_VIRTUAL_BOX_LIMIT_6:
      return "tool at virtual box limit 6";
    case Kinstate::MOTION_NOT_ALLOWED:
      return "motion not allowed";
    case Kinstate::UNKNOWN:
      return "unknown error";
  }
  return "unknown error";
}

std::string DescribeJointErrorBits(int errorBits)
{
  static const std::pair<ErrorJoint, const char *> kBits[] = {
    {ErrorJoint::TEMP, "'Overtemperature' "},
    {ErrorJoint::ESTOP_LOWV, "'Supply too low: Is emergency button pressed?' "},
    {ErrorJoint::MNE, "'Motor not enabled' "},
    {ErrorJoint::COM, "'Communication watch dog' "},
    {ErrorJoint::POS, "'Position lag' "},
    {ErrorJoint::ENC, "'Encoder Error' "},
    {ErrorJoint::OC, "'Overcurrent' "},
    {ErrorJoint::DRV, "'DriveError/SVM' "},
  };

  std::string description;
  for (const auto & bit : kBits) {
    if ((errorBits & static_cast<int>(bit.first)) != 0) {
      description += bit.second;
    }
  }
  return description;
}
}  // namespace CriMessages
}  // namespace Igus
