#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "igus_rebel/CriFraming.hpp"

namespace
{
const char kStatusFrame[] =
  "CRISTART 12 STATUS MODE joint POSJOINTCURRENT 1.0 2.0 CRIEND\n";
const char kAckFrame[] = "CRISTART 13 CMDACK 5 CRIEND\n";

std::vector<std::string> FeedAll(
  Igus::CriFrameParser & parser,
  const std::vector<std::string> & chunks)
{
  std::vector<std::string> bodies;
  for (const auto & chunk : chunks) {
    auto result = parser.Feed(chunk);
    bodies.insert(bodies.end(), result.begin(), result.end());
  }
  return bodies;
}
}  // namespace

TEST(CriFraming, ExtractsSingleMessageBody)
{
  Igus::CriFrameParser parser;
  const auto bodies = parser.Feed(std::string(kStatusFrame));
  ASSERT_EQ(bodies.size(), 1u);
  EXPECT_EQ(bodies[0], "12 STATUS MODE joint POSJOINTCURRENT 1.0 2.0");
  EXPECT_EQ(parser.BufferedBytes(), 0u);
  EXPECT_EQ(parser.TakeDiscardedBytes(), 0u);
}

TEST(CriFraming, ExtractsMultipleMessagesPerRead)
{
  Igus::CriFrameParser parser;
  const auto bodies = parser.Feed(std::string(kStatusFrame) + kAckFrame + kStatusFrame);
  ASSERT_EQ(bodies.size(), 3u);
  EXPECT_EQ(bodies[1], "13 CMDACK 5");
  EXPECT_EQ(bodies[0], bodies[2]);
}

TEST(CriFraming, ReassemblesMessagesSplitAtEveryPosition)
{
  // Regression for the old driver, which dropped the CRISTART marker of a split frame and
  // therefore lost every message that did not arrive in one read.
  const std::string stream = std::string(kStatusFrame) + kAckFrame;
  for (std::size_t split = 1; split < stream.size(); ++split) {
    Igus::CriFrameParser parser;
    const auto bodies = FeedAll(parser, {stream.substr(0, split), stream.substr(split)});
    ASSERT_EQ(bodies.size(), 2u) << "split at " << split;
    EXPECT_EQ(bodies[0], "12 STATUS MODE joint POSJOINTCURRENT 1.0 2.0") << split;
    EXPECT_EQ(bodies[1], "13 CMDACK 5") << split;
    EXPECT_EQ(parser.TakeDiscardedBytes(), 0u) << split;
  }
}

TEST(CriFraming, ReassemblesByteByByte)
{
  const std::string stream = std::string(kStatusFrame) + kAckFrame + kStatusFrame;
  Igus::CriFrameParser parser;
  std::vector<std::string> chunks;
  for (char c : stream) {
    chunks.emplace_back(1, c);
  }
  const auto bodies = FeedAll(parser, chunks);
  ASSERT_EQ(bodies.size(), 3u);
  EXPECT_EQ(bodies[1], "13 CMDACK 5");
}

TEST(CriFraming, SkipsEmptyBodies)
{
  Igus::CriFrameParser parser;
  const auto bodies = parser.Feed(std::string("CRISTART CRIEND\nCRISTARTCRIEND") + kAckFrame);
  ASSERT_EQ(bodies.size(), 1u);
  EXPECT_EQ(bodies[0], "13 CMDACK 5");
}

TEST(CriFraming, DiscardsGarbageOutsideFrames)
{
  Igus::CriFrameParser parser;
  const auto bodies = parser.Feed(std::string("garbage") + kAckFrame + "more junk" + kAckFrame);
  ASSERT_EQ(bodies.size(), 2u);
  EXPECT_EQ(parser.TakeDiscardedBytes(), std::string("garbagemorejunk").size());

  // Pure garbage never accumulates beyond a possible partial start marker.
  for (int i = 0; i < 1000; ++i) {
    EXPECT_TRUE(parser.Feed(std::string("no markers here ")).empty());
  }
  EXPECT_EQ(parser.BufferedBytes(), 0u);
  // Only a genuine partial start marker is kept.
  EXPECT_TRUE(parser.Feed(std::string("junk CRIST")).empty());
  EXPECT_EQ(parser.BufferedBytes(), std::string("CRIST").size());
}

TEST(CriFraming, ResynchronisesOnTruncatedFrame)
{
  Igus::CriFrameParser parser;
  const auto bodies = parser.Feed(std::string("CRISTART 1 STATUS MODE jo") + kAckFrame);
  ASSERT_EQ(bodies.size(), 1u);
  EXPECT_EQ(bodies[0], "13 CMDACK 5");
  EXPECT_GT(parser.TakeDiscardedBytes(), 0u);
}

TEST(CriFraming, BoundsBufferWithoutEndMarker)
{
  Igus::CriFrameParser parser(1024);
  parser.Feed(std::string("CRISTART ") + std::string(2000, 'x'));
  EXPECT_EQ(parser.BufferedBytes(), 0u);
  EXPECT_GT(parser.TakeDiscardedBytes(), 0u);

  const auto bodies = parser.Feed(std::string(kAckFrame));
  ASSERT_EQ(bodies.size(), 1u);
}

TEST(CriFraming, ResetDropsPartialFrame)
{
  Igus::CriFrameParser parser;
  EXPECT_TRUE(parser.Feed(std::string("CRISTART 1 CMDACK")).empty());
  EXPECT_GT(parser.BufferedBytes(), 0u);
  parser.Reset();
  EXPECT_EQ(parser.BufferedBytes(), 0u);
  // The tail of the old frame must not be glued to the next one.
  const auto bodies = parser.Feed(std::string(" 2 CRIEND\n") + kAckFrame);
  ASSERT_EQ(bodies.size(), 1u);
  EXPECT_EQ(bodies[0], "13 CMDACK 5");
}

TEST(CriFraming, HandlesNullAndEmptyInput)
{
  Igus::CriFrameParser parser;
  EXPECT_TRUE(parser.Feed(nullptr, 0).empty());
  EXPECT_TRUE(parser.Feed(std::string()).empty());
  const std::string withNul("CRISTART 1 CMDACK \0 5 CRIEND", 28);
  const auto bodies = parser.Feed(withNul);
  ASSERT_EQ(bodies.size(), 1u);
  EXPECT_EQ(bodies[0].size(), std::string("1 CMDACK \0 5", 12).size());
}
