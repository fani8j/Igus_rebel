#include <gtest/gtest.h>

#include <limits>

#include "igus_rebel_moveit_config/carton_height.hpp"

using carton_height::choose;

TEST(CartonHeight, UsesSnapshotWithinRange)
{
  const auto c = choose(true, 0.967, true, 0.20, 2.0);
  EXPECT_DOUBLE_EQ(c.height_m, 0.967);
  EXPECT_EQ(c.source, "snapshot");
  EXPECT_FALSE(c.warn);
}

TEST(CartonHeight, NeverShrinksBelowMinimum)
{
  const auto c = choose(true, 0.0863, true, 0.20, 2.0);
  EXPECT_DOUBLE_EQ(c.height_m, 0.20);
  EXPECT_FALSE(c.warn);
}

TEST(CartonHeight, AboveMaxOrInvalidFallsBackWithWarning)
{
  for (const double v : {2.5, -1.0, 0.0, std::numeric_limits<double>::quiet_NaN(),
                         std::numeric_limits<double>::infinity()}) {
    const auto c = choose(true, v, true, 0.20, 2.0);
    EXPECT_DOUBLE_EQ(c.height_m, 0.20);
    EXPECT_TRUE(c.warn);
  }
}

TEST(CartonHeight, FlagOffOrAbsentUsesParameter)
{
  EXPECT_DOUBLE_EQ(choose(false, 0.967, true, 0.20, 2.0).height_m, 0.20);
  const auto c = choose(true, std::numeric_limits<double>::quiet_NaN(), false, 0.20, 2.0);
  EXPECT_DOUBLE_EQ(c.height_m, 0.20);
  EXPECT_FALSE(c.warn);
}

TEST(CartonHeight, ExactBoundsAccepted)
{
  EXPECT_DOUBLE_EQ(choose(true, 2.0, true, 0.20, 2.0).height_m, 2.0);
  EXPECT_DOUBLE_EQ(choose(true, 0.20, true, 0.20, 2.0).height_m, 0.20);
}
