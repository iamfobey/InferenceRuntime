#include <gtest/gtest.h>

#include <lowp.hpp>

TEST(SingleHeader, PlaceholderTypesCompile)
{
    constexpr lowp::float16 f16;

    static_assert(f16.test() == 0);

    EXPECT_EQ(lowp::version_minor, 1);
}
