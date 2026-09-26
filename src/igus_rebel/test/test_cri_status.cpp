#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <string>
#include <variant>
#include <vector>

#include "igus_rebel/CriMessages.hpp"

namespace
{
using Igus::CriMessages::Kinstate;
using Igus::CriMessages::MessageType;

std::string MakeStatus(
  const std::string & din, const std::string & dout,
  const std::string & posJointCurrent = "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0",
  const std::string & kinstate = "0", const std::string & errorJoints =
  "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0")
{
  const std::string zeros16 = "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0";
  const std::string zeros6 = "0 0 0 0 0 0";
  const std::string zeros3 = "0 0 0";

  return "STATUS MODE joint POSJOINTSETPOINT " + zeros16 +
         " POSJOINTCURRENT " + posJointCurrent +
         " POSCARTROBOT " + zeros6 +
         " POSCARTPLATTFORM " + zeros3 +
         " OVERRIDE 100 DIN " + din +
         " DOUT " + dout +
         " ESTOP 0 SUPPLY 48 CURRENTALL 2 CURRENTJOINTS " + zeros16 +
         " ERROR 0 " + errorJoints + " KINSTATE " + kinstate;
}
}  // namespace

TEST(CriStatus, ParsesFullWidthHexDigitalMasks)
{
  const Igus::CriMessages::Status status(
    MakeStatus("1000000000000000", "FFFFFFFFFFFFFFFF"));

  ASSERT_TRUE(status.IsValid());
  EXPECT_EQ(status.din, UINT64_C(0x1000000000000000));
  EXPECT_EQ(status.dout, UINT64_MAX);
  EXPECT_FLOAT_EQ(status.overrideValue, 100.0F);
  EXPECT_EQ(status.supply, 48);
  EXPECT_EQ(status.kinstate, Kinstate::NO_ERROR);
}

TEST(CriStatus, ParsesStatusFieldsAfterKinstate)
{
  const Igus::CriMessages::Status status(
    MakeStatus("0", "8000000000000000") +
    " OPMODE -1 CARTSPEED 0.0 GSIG 0 FRAMEROBOT #base 278.0 47.0 358.2 -141.03 44.66 -166.32");

  ASSERT_TRUE(status.IsValid());
  EXPECT_EQ(status.dout, UINT64_C(0x8000000000000000));
  EXPECT_EQ(status.kinstate, Kinstate::NO_ERROR);
}

TEST(CriStatus, ParsesKinstateFollowedByFieldsOtherThanOpmode)
{
  const Igus::CriMessages::Status status(MakeStatus(
      "0", "0", std::string(
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0"), "13") + " CARTSPEED 0.0 GSIG 0");

  ASSERT_TRUE(status.IsValid());
  EXPECT_EQ(status.kinstate, Kinstate::JOINT_LIMIT_MIN);
}

TEST(CriStatus, RejectsMalformedAndOverflowingDigitalMasks)
{
  Igus::CriMessages::Status malformed;
  Igus::CriMessages::Status overflowing;
  Igus::CriMessages::Status negative;

  EXPECT_NO_THROW(malformed = Igus::CriMessages::Status(MakeStatus("not_hex", "0")));
  EXPECT_NO_THROW(
    overflowing = Igus::CriMessages::Status(
      MakeStatus("10000000000000000", "0")));
  EXPECT_NO_THROW(negative = Igus::CriMessages::Status(MakeStatus("-1", "0")));
  EXPECT_FALSE(malformed.IsValid());
  EXPECT_FALSE(overflowing.IsValid());
  EXPECT_FALSE(negative.IsValid());
}

TEST(CriStatus, RejectsIncompleteJointArrays)
{
  std::string message = MakeStatus("0", "0");
  const std::string marker = "POSJOINTSETPOINT 0 0 ";
  const auto position = message.find(marker);
  ASSERT_NE(position, std::string::npos);
  message.erase(position + marker.size() - 2, 2);

  const Igus::CriMessages::Status status(message);
  EXPECT_FALSE(status.IsValid());
}

TEST(CriStatus, ParsesNonZeroNegativeAndExponentJointPositions)
{
  const Igus::CriMessages::Status status(
    MakeStatus(
      "0", "0",
      "12.5 -45.25 1.5e+02 -3E-1 0.001 -179.99 0 0 0 0 0 0 0 0 0 0"));

  ASSERT_TRUE(status.IsValid());
  EXPECT_FLOAT_EQ(status.posJointCurrent[0], 12.5F);
  EXPECT_FLOAT_EQ(status.posJointCurrent[1], -45.25F);
  EXPECT_FLOAT_EQ(status.posJointCurrent[2], 150.0F);
  EXPECT_FLOAT_EQ(status.posJointCurrent[3], -0.3F);
  EXPECT_FLOAT_EQ(status.posJointCurrent[4], 0.001F);
  EXPECT_FLOAT_EQ(status.posJointCurrent[5], -179.99F);
}

TEST(CriStatus, RejectsGarbageJointValues)
{
  EXPECT_FALSE(
    Igus::CriMessages::Status(
      MakeStatus("0", "0", "1 2 x 4 5 6 0 0 0 0 0 0 0 0 0 0")).IsValid());
  EXPECT_FALSE(
    Igus::CriMessages::Status(
      MakeStatus("0", "0", "1 2 3 4 5 6 0 0 0 0 0 0 0 0 0 0 7")).IsValid());
  EXPECT_FALSE(
    Igus::CriMessages::Status(
      MakeStatus("0", "0", "1 2 3 4 5 6 0 0 0 0 0 0 0 0 0 0", "abc")).IsValid());
}

TEST(CriStatus, RejectsMissingOrReorderedKeywords)
{
  std::string missingKinstate = MakeStatus("0", "0");
  missingKinstate.erase(missingKinstate.find(" KINSTATE"));
  EXPECT_FALSE(Igus::CriMessages::Status(missingKinstate).IsValid());

  EXPECT_FALSE(Igus::CriMessages::Status("").IsValid());
  EXPECT_FALSE(Igus::CriMessages::Status("STATUS").IsValid());
  EXPECT_FALSE(Igus::CriMessages::Status("STATUS KINSTATE 0 MODE joint").IsValid());
}

TEST(CriStatus, ParsesJointErrorArray)
{
  const Igus::CriMessages::Status status(
    MakeStatus(
      "0", "0", "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0", "0",
      "4 0 16 0 0 128 0 0 0 0 0 0 0 0 0 0"));

  ASSERT_TRUE(status.IsValid());
  EXPECT_EQ(status.errorJoints[0], 4);
  EXPECT_EQ(status.errorJoints[2], 16);
  EXPECT_EQ(status.errorJoints[5], 128);
}

TEST(CriKinstate, MapsKnownAndUnknownValues)
{
  using Igus::CriMessages::Status;
  EXPECT_EQ(Status::KinstateFromInt(0), Kinstate::NO_ERROR);
  EXPECT_EQ(Status::KinstateFromInt(13), Kinstate::JOINT_LIMIT_MIN);
  EXPECT_EQ(Status::KinstateFromInt(14), Kinstate::JOINT_LIMIT_MAX);
  EXPECT_EQ(Status::KinstateFromInt(21), Kinstate::CARTESIAN_SINGULARITY_CENTER);
  EXPECT_EQ(Status::KinstateFromInt(23), Kinstate::CARTESIAN_SINGULARITY_REACH);
  EXPECT_EQ(Status::KinstateFromInt(24), Kinstate::CARTESIAN_SINGULARITY_WRIST);
  EXPECT_EQ(Status::KinstateFromInt(30), Kinstate::TOOL_AT_VIRTUAL_BOX_LIMIT_1);
  EXPECT_EQ(Status::KinstateFromInt(35), Kinstate::TOOL_AT_VIRTUAL_BOX_LIMIT_6);
  EXPECT_EQ(Status::KinstateFromInt(99), Kinstate::MOTION_NOT_ALLOWED);
  EXPECT_EQ(Status::KinstateFromInt(22), Kinstate::UNKNOWN);
  EXPECT_EQ(Status::KinstateFromInt(-1), Kinstate::UNKNOWN);
  EXPECT_EQ(Status::KinstateFromInt(1000), Kinstate::UNKNOWN);

  EXPECT_EQ(Igus::CriMessages::KinstateToString(Kinstate::NO_ERROR), "no error");
  EXPECT_EQ(
    Igus::CriMessages::KinstateToString(Kinstate::JOINT_LIMIT_MAX), "joint at maximum limit");
  EXPECT_EQ(
    Igus::CriMessages::KinstateToString(Kinstate::MOTION_NOT_ALLOWED),
    "motion not allowed");
  EXPECT_EQ(Igus::CriMessages::KinstateToString(Kinstate::UNKNOWN), "unknown error");
}

TEST(CriJointErrors, DescribesEachBit)
{
  using Igus::CriMessages::DescribeJointErrorBits;
  EXPECT_EQ(DescribeJointErrorBits(0), "");
  EXPECT_EQ(DescribeJointErrorBits(1), "'Overtemperature' ");
  EXPECT_EQ(DescribeJointErrorBits(2), "'Supply too low: Is emergency button pressed?' ");
  EXPECT_EQ(DescribeJointErrorBits(4), "'Motor not enabled' ");
  EXPECT_EQ(DescribeJointErrorBits(8), "'Communication watch dog' ");
  EXPECT_EQ(DescribeJointErrorBits(16), "'Position lag' ");
  EXPECT_EQ(DescribeJointErrorBits(32), "'Encoder Error' ");
  EXPECT_EQ(DescribeJointErrorBits(64), "'Overcurrent' ");
  EXPECT_EQ(DescribeJointErrorBits(128), "'DriveError/SVM' ");
  EXPECT_EQ(DescribeJointErrorBits(1 | 16), "'Overtemperature' 'Position lag' ");
  // Bits above 128 are not documented and are ignored.
  EXPECT_EQ(DescribeJointErrorBits(256), "");
}

TEST(CriParse, ClassifiesMessageTypes)
{
  using Igus::CriMessages::CriMessage;
  EXPECT_EQ(CriMessage::GetMessageType("12 STATUS MODE joint"), MessageType::STATUS);
  EXPECT_EQ(CriMessage::GetMessageType(" 12 CMDACK 3"), MessageType::CMDACK);
  EXPECT_EQ(CriMessage::GetMessageType("12 RUNSTATE x"), MessageType::RUNSTATE);
  EXPECT_EQ(CriMessage::GetMessageType("12 NOPE x"), MessageType::UNKNOWN);
  EXPECT_EQ(CriMessage::GetMessageType(""), MessageType::UNKNOWN);
  EXPECT_EQ(CriMessage::GetMessageType(" "), MessageType::UNKNOWN);
  EXPECT_EQ(CriMessage::GetMessageType("12"), MessageType::UNKNOWN);
}

TEST(CriParse, ParsesWellFormedMessages)
{
  using Igus::CriMessages::ParseMessage;

  auto status = ParseMessage("7 " + MakeStatus("0", "0"));
  ASSERT_TRUE(status.ok);
  EXPECT_EQ(status.type, MessageType::STATUS);
  EXPECT_TRUE(std::get<Igus::CriMessages::Status>(status.payload).IsValid());

  auto ack = ParseMessage("8 CMDACK 42");
  ASSERT_TRUE(ack.ok);
  EXPECT_EQ(std::get<Igus::CriMessages::CmdAck>(ack.payload).acceptedCmd, 42);

  auto error = ParseMessage("9 CMDERROR 43 Command not allowed");
  ASSERT_TRUE(error.ok);
  EXPECT_EQ(std::get<Igus::CriMessages::CmdError>(error.payload).recjectedCmd, 43);
  EXPECT_EQ(std::get<Igus::CriMessages::CmdError>(error.payload).error, "Command not allowed");

  auto log = ParseMessage("10 LOGMSG WARN 123456 Joint 2 near limit");
  ASSERT_TRUE(log.ok);
  const auto & logMsg = std::get<Igus::CriMessages::LogMsg>(log.payload);
  EXPECT_EQ(logMsg.logLevel, Igus::CriMessages::LogLevel::WARN);
  EXPECT_EQ(logMsg.timestamp, 123456);
  EXPECT_EQ(logMsg.logMsg, "Joint 2 near limit");

  auto cycle = ParseMessage("11 CYCLESTAT 10 37");
  ASSERT_TRUE(cycle.ok);
  EXPECT_EQ(std::get<Igus::CriMessages::Cyclestat>(cycle.payload).cycletime, 10);
  EXPECT_EQ(std::get<Igus::CriMessages::Cyclestat>(cycle.payload).workload, 37);

  // iRC V15 (V980-15-000-2) sends decimal values, observed on the real controller.
  auto cycleV15 = ParseMessage("49 CYCLESTAT 12.0 0.2");
  ASSERT_TRUE(cycleV15.ok);
  EXPECT_FLOAT_EQ(std::get<Igus::CriMessages::Cyclestat>(cycleV15.payload).cycletime, 12.0f);
  EXPECT_FLOAT_EQ(std::get<Igus::CriMessages::Cyclestat>(cycleV15.payload).workload, 0.2f);

  auto message = ParseMessage("12 MESSAGE Hello robot");
  ASSERT_TRUE(message.ok);
  EXPECT_EQ(std::get<Igus::CriMessages::Message>(message.payload).message, "Hello robot");

  auto cmd = ParseMessage("13 CMD Override 100.0");
  ASSERT_TRUE(cmd.ok);
  EXPECT_EQ(std::get<Igus::CriMessages::Command>(cmd.payload).command, "Override 100.0");

  auto info = ParseMessage("14 INFO Version V17");
  ASSERT_TRUE(info.ok);
  EXPECT_EQ(std::get<Igus::CriMessages::Info>(info.payload).info, "Version V17");

  auto runstate = ParseMessage("15 RUNSTATE something");
  ASSERT_TRUE(runstate.ok);
  EXPECT_TRUE(std::holds_alternative<std::monostate>(runstate.payload));
}

TEST(CriParse, MalformedInputNeverEscapesParseMessage)
{
  using Igus::CriMessages::ParseMessage;

  const std::vector<std::string> malformed = {
    "",
    " ",
    "1",
    "1 STATUS",
    "1 STATUS MODE",
    "1 STATUS MODE joint POSJOINTSETPOINT POSJOINTCURRENT",
    "1 CMDACK",
    "1 CMDACK x",
    "1 CMDACK 99999999999999999999",
    "1 CMDERROR",
    "1 CMDERROR abc error",
    "1 LOGMSG",
    "1 LOGMSG WARN",
    "1 LOGMSG WARN notanumber text",
    "1 CYCLESTAT",
    "1 CYCLESTAT 10",
    "1 CYCLESTAT a b",
    "1 MESSAGE",
    "1 GARBAGE \x01\x02\xff",
    std::string("1 STATUS \0 MODE", 15),
    std::string(10000, 'X'),
  };

  for (const auto & input : malformed) {
    Igus::CriMessages::ParsedMessage parsed;
    EXPECT_NO_THROW(parsed = ParseMessage(input)) << input;
    if (parsed.type == MessageType::STATUS || parsed.type == MessageType::UNKNOWN) {
      EXPECT_FALSE(parsed.ok) << input;
    }
  }

  EXPECT_FALSE(ParseMessage("1 CMDACK x").ok);
  EXPECT_FALSE(ParseMessage("1 LOGMSG WARN notanumber text").ok);
  EXPECT_FALSE(ParseMessage("1 CYCLESTAT a b").ok);
  EXPECT_FALSE(ParseMessage("1 GARBAGE").ok);
  // An empty MESSAGE body is still a MESSAGE.
  EXPECT_TRUE(ParseMessage("1 MESSAGE").ok);
}
