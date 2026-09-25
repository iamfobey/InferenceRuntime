#include <gtest/gtest.h>

#include <lowp.hpp>

TEST(SingleHeader, PlaceholderTypesCompile)
{
    const lowp::f16 f16{1.0f};

    EXPECT_EQ(f16.SignValue(), 0);
}
