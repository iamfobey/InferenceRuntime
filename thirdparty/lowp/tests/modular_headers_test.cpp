#include <gtest/gtest.h>

#include <lowp/lowp.hpp>

TEST(ModularHeaders, PlaceholderTypesCompile)
{
    constexpr lowp::float16 f16;
    
    static_assert(f16.test() == 0);
    
    EXPECT_EQ(lowp::version_major, 0);
}
