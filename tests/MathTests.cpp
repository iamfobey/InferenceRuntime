#include <gtest/gtest.h>

#include "Math/Math.hpp"

#include <array>

TEST(LinearTests, CalculatesExpectedValues)
{
    constexpr std::size_t J = 2;
    constexpr std::size_t K = 3;

    constexpr std::array W{
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };

    constexpr std::array x{
        10.0f,
        20.0f,
        30.0f
    };

    constexpr std::array b{
        1.0f,
        -1.0f
    };

    std::array<float, J> y{};

    Math::Linear(
        W.data(),
        x.data(),
        b.data(),
        y.data(),
        J,
        K
    );

    EXPECT_NEAR(y[0], 141.0f, 1e-5f);
    EXPECT_NEAR(y[1], 319.0f, 1e-5f);
}
