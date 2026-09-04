#include <cstdint>
#include <string>

#include <gtest/gtest.h>

#include "igus_rebel/CriMessages.hpp"

namespace
{
std::string MakeStatus(const std::string &din, const std::string &dout)
{
    const std::string zeros16 = "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0";
    const std::string zeros6 = "0 0 0 0 0 0";
    const std::string zeros3 = "0 0 0";

    return "STATUS MODE joint POSJOINTSETPOINT " + zeros16 +
           " POSJOINTCURRENT " + zeros16 +
           " POSCARTROBOT " + zeros6 +
           " POSCARTPLATTFORM " + zeros3 +
           " OVERRIDE 100 DIN " + din +
           " DOUT " + dout +
           " ESTOP 0 SUPPLY 48 CURRENTALL 2 CURRENTJOINTS " + zeros16 +
           " ERROR 0 " + zeros16 + " KINSTATE 0";
}
}

TEST(CriStatus, ParsesFullWidthHexDigitalMasks)
{
    const Igus::CriMessages::Status status(
        MakeStatus("1000000000000000", "FFFFFFFFFFFFFFFF"));

    ASSERT_TRUE(status.IsValid());
    EXPECT_EQ(status.din, UINT64_C(0x1000000000000000));
    EXPECT_EQ(status.dout, UINT64_MAX);
    EXPECT_FLOAT_EQ(status.overrideValue, 100.0F);
    EXPECT_EQ(status.supply, 48);
    EXPECT_EQ(status.kinstate, Igus::CriMessages::Kinstate::NO_ERROR);
}
TEST(CriStatus, ParsesStatusFieldsAfterKinstate)
{
    const Igus::CriMessages::Status status(
        MakeStatus("0", "8000000000000000") +
        " OPMODE -1 CARTSPEED 0.0 GSIG 0 FRAMEROBOT #base 278.0 47.0 358.2 -141.03 44.66 -166.32");

    ASSERT_TRUE(status.IsValid());
    EXPECT_EQ(status.dout, UINT64_C(0x8000000000000000));
    EXPECT_EQ(status.kinstate, Igus::CriMessages::Kinstate::NO_ERROR);
}

TEST(CriStatus, RejectsMalformedAndOverflowingDigitalMasks)
{
    Igus::CriMessages::Status malformed;
    Igus::CriMessages::Status overflowing;

    EXPECT_NO_THROW(malformed = Igus::CriMessages::Status(MakeStatus("not_hex", "0")));
    EXPECT_NO_THROW(overflowing = Igus::CriMessages::Status(
        MakeStatus("10000000000000000", "0")));
    EXPECT_FALSE(malformed.IsValid());
    EXPECT_FALSE(overflowing.IsValid());
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
