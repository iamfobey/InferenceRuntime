#include <gtest/gtest.h>

#include "Math/Math.hpp"
#include "Utils/Converters.hpp"

#include <array>
#include <cstdint>

TEST(Float16Tests, RoundTripsRepresentativeValues)
{
    constexpr std::array values{
        0.0f,
        -0.0f,
        1.0f,
        -2.0f,
        0.5f,
        65504.0f
    };

    for (const float value : values)
    {
        const auto half = Utils::Converters::Float32ToFloat16(value);
        const auto roundTrip = Utils::Converters::Float16ToFloat32(half);
        EXPECT_EQ(roundTrip, value);
    }
}

TEST(LinearTests, CalculatesExpectedValues)
{
    constexpr std::size_t J = 2;
    constexpr std::size_t K = 3;

    constexpr std::array WFloat{
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };

    std::array<std::uint16_t, WFloat.size()> W{};

    for (std::size_t i{}; i < W.size(); ++i)
        W[i] = Utils::Converters::Float32ToFloat16(WFloat[i]);

    constexpr std::array x{
        10.0f,
        20.0f,
        30.0f
    };

    std::array<float, J> y{};

    Math::Linear(W.data(), x.data(), y.data(), J, K);

    EXPECT_NEAR(y[0], 140.0f, 1e-5f);
    EXPECT_NEAR(y[1], 320.0f, 1e-5f);
}
