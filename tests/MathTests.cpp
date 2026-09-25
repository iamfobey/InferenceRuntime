#include <gtest/gtest.h>

#include "Math/Math.hpp"
#include "lowp/lowp.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <tuple>
#include <type_traits>
#include <vector>

namespace
{
    using PairTypes = testing::Types<
        std::tuple<lowp::f16, lowp::f16>,
        std::tuple<lowp::f16, lowp::f32>,
        std::tuple<lowp::f32, lowp::f16>,
        std::tuple<lowp::f32, lowp::f32>>;

    using TripleTypes = testing::Types<
        std::tuple<lowp::f16, lowp::f16, lowp::f16>,
        std::tuple<lowp::f16, lowp::f16, lowp::f32>,
        std::tuple<lowp::f16, lowp::f32, lowp::f16>,
        std::tuple<lowp::f16, lowp::f32, lowp::f32>,
        std::tuple<lowp::f32, lowp::f16, lowp::f16>,
        std::tuple<lowp::f32, lowp::f16, lowp::f32>,
        std::tuple<lowp::f32, lowp::f32, lowp::f16>,
        std::tuple<lowp::f32, lowp::f32, lowp::f32>>;

    using QuadTypes = testing::Types<
        std::tuple<lowp::f16, lowp::f16, lowp::f16, lowp::f16>,
        std::tuple<lowp::f16, lowp::f16, lowp::f16, lowp::f32>,
        std::tuple<lowp::f16, lowp::f16, lowp::f32, lowp::f16>,
        std::tuple<lowp::f16, lowp::f16, lowp::f32, lowp::f32>,
        std::tuple<lowp::f16, lowp::f32, lowp::f16, lowp::f16>,
        std::tuple<lowp::f16, lowp::f32, lowp::f16, lowp::f32>,
        std::tuple<lowp::f16, lowp::f32, lowp::f32, lowp::f16>,
        std::tuple<lowp::f16, lowp::f32, lowp::f32, lowp::f32>,
        std::tuple<lowp::f32, lowp::f16, lowp::f16, lowp::f16>,
        std::tuple<lowp::f32, lowp::f16, lowp::f16, lowp::f32>,
        std::tuple<lowp::f32, lowp::f16, lowp::f32, lowp::f16>,
        std::tuple<lowp::f32, lowp::f16, lowp::f32, lowp::f32>,
        std::tuple<lowp::f32, lowp::f32, lowp::f16, lowp::f16>,
        std::tuple<lowp::f32, lowp::f32, lowp::f16, lowp::f32>,
        std::tuple<lowp::f32, lowp::f32, lowp::f32, lowp::f16>,
        std::tuple<lowp::f32, lowp::f32, lowp::f32, lowp::f32>>;

    template <class T>
    class MathPairTest : public testing::Test {};

    TYPED_TEST_SUITE(MathPairTest, PairTypes);

    template <class T>
    class MathTripleTest : public testing::Test {};

    TYPED_TEST_SUITE(MathTripleTest, TripleTypes);

    template <class T>
    class MathQuadTest : public testing::Test {};

    TYPED_TEST_SUITE(MathQuadTest, QuadTypes);

    TYPED_TEST(MathPairTest, Embedding)
    {
        using EmbeddingType = std::tuple_element_t<0, TypeParam>;
        using OutputType = std::tuple_element_t<1, TypeParam>;

        constexpr std::size_t vocabularySize = 3;
        constexpr std::size_t hiddenSize = 10;
        constexpr std::size_t tokenCount = 5;

        const std::array<EmbeddingType, vocabularySize * hiddenSize> table{
            EmbeddingType{0.0f}, EmbeddingType{1.0f}, EmbeddingType{2.0f}, EmbeddingType{3.0f}, EmbeddingType{4.0f},
            EmbeddingType{5.0f}, EmbeddingType{6.0f}, EmbeddingType{7.0f}, EmbeddingType{8.0f}, EmbeddingType{9.0f},
            EmbeddingType{10.0f}, EmbeddingType{11.0f}, EmbeddingType{12.0f}, EmbeddingType{13.0f}, EmbeddingType{14.0f},
            EmbeddingType{15.0f}, EmbeddingType{16.0f}, EmbeddingType{17.0f}, EmbeddingType{18.0f}, EmbeddingType{19.0f},
            EmbeddingType{20.0f}, EmbeddingType{21.0f}, EmbeddingType{22.0f}, EmbeddingType{23.0f}, EmbeddingType{24.0f},
            EmbeddingType{25.0f}, EmbeddingType{26.0f}, EmbeddingType{27.0f}, EmbeddingType{28.0f}, EmbeddingType{29.0f}};

        constexpr std::array tokenIds{2, 0, -1, 1, 3};
        std::vector<OutputType> output(tokenCount * hiddenSize, OutputType{-123.0f});

        Math::Embedding(table.data(), tokenIds.data(), output.data(), tokenCount, vocabularySize, hiddenSize);

        constexpr std::array token2Expected{20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f};
        constexpr std::array token0Expected{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
        constexpr std::array token1Expected{10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f};

        for (std::size_t i{}; i < hiddenSize; ++i)
        {
            const auto token2Actual = static_cast<float>(output[i]);
            const auto token0Actual = static_cast<float>(output[hiddenSize + i]);
            const auto invalidNegativeActual = static_cast<float>(output[2 * hiddenSize + i]);
            const auto token1Actual = static_cast<float>(output[3 * hiddenSize + i]);
            const auto invalidUpperActual = static_cast<float>(output[4 * hiddenSize + i]);

            EXPECT_FLOAT_EQ(token2Actual, token2Expected[i]);
            EXPECT_FLOAT_EQ(token0Actual, token0Expected[i]);
            EXPECT_FLOAT_EQ(invalidNegativeActual, -123.0f);
            EXPECT_FLOAT_EQ(token1Actual, token1Expected[i]);
            EXPECT_FLOAT_EQ(invalidUpperActual, -123.0f);
        }
    }

    TYPED_TEST(MathPairTest, SiLU)
    {
        using InputType = std::tuple_element_t<0, TypeParam>;
        using OutputType = std::tuple_element_t<1, TypeParam>;

        const std::array<InputType, 7> input{
            InputType{-4.0f}, InputType{-2.0f}, InputType{-1.0f}, InputType{0.0f},
            InputType{1.0f}, InputType{2.0f}, InputType{4.0f}};

        constexpr std::array expected{
            -0.07194484f,
            -0.23840584f,
            -0.26894143f,
            0.0f,
            0.73105860f,
            1.76159418f,
            3.92805529f};

        std::vector<OutputType> output(input.size(), OutputType{0.0f});

        Math::SiLU(input.data(), output.data(), input.size());

        const auto tolerance = std::is_same_v<OutputType, lowp::f16> ? 0.002f : 0.000001f;

        for (std::size_t i{}; i < output.size(); ++i)
        {
            const auto actual = static_cast<float>(output[i]);
            EXPECT_NEAR(actual, expected[i], tolerance);
        }
    }

    TYPED_TEST(MathPairTest, ComputeRoPECosSin)
    {
        using CosType = std::tuple_element_t<0, TypeParam>;
        using SinType = std::tuple_element_t<1, TypeParam>;

        constexpr std::size_t position = 4;
        constexpr std::size_t headDimension = 4;
        constexpr float theta = 16.0f;

        std::array<CosType, 2> cosine{CosType{0.0f}, CosType{0.0f}};
        std::array<SinType, 2> sine{SinType{0.0f}, SinType{0.0f}};

        Math::ComputeRoPECosSin(cosine.data(), sine.data(), position, headDimension, theta);

        constexpr std::array expectedCosine{-0.65364361f, 0.54030228f};
        constexpr std::array expectedSine{-0.75680250f, 0.84147096f};

        const auto cosineTolerance = std::is_same_v<CosType, lowp::f16> ? 0.001f : 0.000001f;
        const auto sineTolerance = std::is_same_v<SinType, lowp::f16> ? 0.001f : 0.000001f;

        for (std::size_t i{}; i < cosine.size(); ++i)
        {
            const auto cosineActual = static_cast<float>(cosine[i]);
            const auto sineActual = static_cast<float>(sine[i]);

            EXPECT_NEAR(cosineActual, expectedCosine[i], cosineTolerance);
            EXPECT_NEAR(sineActual, expectedSine[i], sineTolerance);
        }
    }

    TYPED_TEST(MathPairTest, ComputeRoPECosSinDoesNotWriteForInvalidHeadDimension)
    {
        using CosType = std::tuple_element_t<0, TypeParam>;
        using SinType = std::tuple_element_t<1, TypeParam>;

        std::array<CosType, 2> cosine{CosType{7.0f}, CosType{8.0f}};
        std::array<SinType, 2> sine{SinType{9.0f}, SinType{10.0f}};

        Math::ComputeRoPECosSin(cosine.data(), sine.data(), 4, 3, 16.0f);

        const auto cosine0 = static_cast<float>(cosine[0]);
        const auto cosine1 = static_cast<float>(cosine[1]);
        const auto sine0 = static_cast<float>(sine[0]);
        const auto sine1 = static_cast<float>(sine[1]);

        EXPECT_FLOAT_EQ(cosine0, 7.0f);
        EXPECT_FLOAT_EQ(cosine1, 8.0f);
        EXPECT_FLOAT_EQ(sine0, 9.0f);
        EXPECT_FLOAT_EQ(sine1, 10.0f);
    }

    TYPED_TEST(MathPairTest, CopyToCache)
    {
        using SourceType = std::tuple_element_t<0, TypeParam>;
        using CacheType = std::tuple_element_t<1, TypeParam>;

        constexpr std::size_t elementCount = 6;
        constexpr std::size_t position = 1;

        const std::array<SourceType, elementCount> source{
            SourceType{-2.0f}, SourceType{-0.5f}, SourceType{0.0f},
            SourceType{0.75f}, SourceType{1.5f}, SourceType{3.0f}};

        std::vector<CacheType> cache(18, CacheType{-9.0f});

        Math::CopyToCache(source.data(), cache.data(), position, elementCount);

        constexpr std::array expected{
            -9.0f, -9.0f, -9.0f, -9.0f, -9.0f, -9.0f,
            -2.0f, -0.5f, 0.0f, 0.75f, 1.5f, 3.0f,
            -9.0f, -9.0f, -9.0f, -9.0f, -9.0f, -9.0f};

        for (std::size_t i{}; i < cache.size(); ++i)
        {
            const auto actual = static_cast<float>(cache[i]);
            EXPECT_FLOAT_EQ(actual, expected[i]);
        }
    }

    TYPED_TEST(MathTripleTest, LinearRange)
    {
        using WeightType = std::tuple_element_t<0, TypeParam>;
        using InputType = std::tuple_element_t<1, TypeParam>;
        using OutputType = std::tuple_element_t<2, TypeParam>;

        constexpr std::size_t rows = 4;
        constexpr std::size_t columns = 40;

        std::vector<WeightType> matrix(rows * columns, WeightType{0.0f});
        std::vector<InputType> input(columns, InputType{1.0f});
        std::vector<OutputType> output(rows, OutputType{123.0f});

        std::fill(matrix.begin() + columns, matrix.begin() + columns + 32, WeightType{0.5f});
        std::fill(matrix.begin() + columns + 32, matrix.begin() + 2 * columns, WeightType{2.0f});

        std::fill(matrix.begin() + 2 * columns, matrix.begin() + 2 * columns + 32, WeightType{1.0f});
        std::fill(matrix.begin() + 2 * columns + 32, matrix.begin() + 3 * columns, WeightType{-1.0f});

        Math::LinearRange(matrix.data(), input.data(), output.data(), 1, 3, columns);

        const auto row0 = static_cast<float>(output[0]);
        const auto row1 = static_cast<float>(output[1]);
        const auto row2 = static_cast<float>(output[2]);
        const auto row3 = static_cast<float>(output[3]);

        EXPECT_FLOAT_EQ(row0, 123.0f);
        EXPECT_FLOAT_EQ(row1, 32.0f);
        EXPECT_FLOAT_EQ(row2, 24.0f);
        EXPECT_FLOAT_EQ(row3, 123.0f);
    }

    TYPED_TEST(MathTripleTest, RMSNorm)
    {
        using InputType = std::tuple_element_t<0, TypeParam>;
        using WeightType = std::tuple_element_t<1, TypeParam>;
        using OutputType = std::tuple_element_t<2, TypeParam>;

        const std::array<InputType, 8> input{
            InputType{1.0f}, InputType{1.0f}, InputType{1.0f}, InputType{1.0f},
            InputType{1.0f}, InputType{1.0f}, InputType{1.0f}, InputType{1.0f}};

        const std::array<WeightType, 8> weight{
            WeightType{1.0f}, WeightType{2.0f}, WeightType{3.0f}, WeightType{4.0f},
            WeightType{-1.0f}, WeightType{-2.0f}, WeightType{0.5f}, WeightType{-0.5f}};

        constexpr std::array expected{0.5f, 1.0f, 1.5f, 2.0f, -0.5f, -1.0f, 0.25f, -0.25f};

        std::vector<OutputType> output(input.size(), OutputType{0.0f});

        Math::RMSNorm(input.data(), weight.data(), 3.0f, output.data(), input.size());

        for (std::size_t i{}; i < output.size(); ++i)
        {
            const auto actual = static_cast<float>(output[i]);
            EXPECT_FLOAT_EQ(actual, expected[i]);
        }
    }

    TYPED_TEST(MathTripleTest, RMSNormDoesNotWriteForZeroElements)
    {
        using InputType = std::tuple_element_t<0, TypeParam>;
        using WeightType = std::tuple_element_t<1, TypeParam>;
        using OutputType = std::tuple_element_t<2, TypeParam>;

        const InputType input{1.0f};
        const WeightType weight{1.0f};
        OutputType output{17.0f};

        Math::RMSNorm(&input, &weight, 1.0e-5f, &output, 0);

        const auto actual = static_cast<float>(output);
        EXPECT_FLOAT_EQ(actual, 17.0f);
    }

    TYPED_TEST(MathTripleTest, Add)
    {
        using InputAType = std::tuple_element_t<0, TypeParam>;
        using InputBType = std::tuple_element_t<1, TypeParam>;
        using OutputType = std::tuple_element_t<2, TypeParam>;

        const std::array<InputAType, 6> inputA{
            InputAType{-4.0f}, InputAType{-2.0f}, InputAType{-0.5f},
            InputAType{0.0f}, InputAType{1.5f}, InputAType{4.0f}};

        const std::array<InputBType, 6> inputB{
            InputBType{1.0f}, InputBType{-1.5f}, InputBType{0.25f},
            InputBType{2.0f}, InputBType{-0.5f}, InputBType{0.5f}};

        constexpr std::array expected{-3.0f, -3.5f, -0.25f, 2.0f, 1.0f, 4.5f};

        std::vector<OutputType> output(inputA.size(), OutputType{0.0f});

        Math::Add(inputA.data(), inputB.data(), output.data(), output.size());

        for (std::size_t i{}; i < output.size(); ++i)
        {
            const auto actual = static_cast<float>(output[i]);
            EXPECT_FLOAT_EQ(actual, expected[i]);
        }
    }

    TYPED_TEST(MathTripleTest, Multiply)
    {
        using InputAType = std::tuple_element_t<0, TypeParam>;
        using InputBType = std::tuple_element_t<1, TypeParam>;
        using OutputType = std::tuple_element_t<2, TypeParam>;

        const std::array<InputAType, 6> inputA{
            InputAType{-4.0f}, InputAType{-2.0f}, InputAType{-0.5f},
            InputAType{0.0f}, InputAType{1.5f}, InputAType{4.0f}};

        const std::array<InputBType, 6> inputB{
            InputBType{1.0f}, InputBType{-1.5f}, InputBType{0.25f},
            InputBType{2.0f}, InputBType{-0.5f}, InputBType{0.5f}};

        constexpr std::array expected{-4.0f, 3.0f, -0.125f, 0.0f, -0.75f, 2.0f};

        std::vector<OutputType> output(inputA.size(), OutputType{0.0f});

        Math::Multiply(inputA.data(), inputB.data(), output.data(), output.size());

        for (std::size_t i{}; i < output.size(); ++i)
        {
            const auto actual = static_cast<float>(output[i]);
            EXPECT_FLOAT_EQ(actual, expected[i]);
        }
    }

    TYPED_TEST(MathTripleTest, RoPE)
    {
        using SourceType = std::tuple_element_t<0, TypeParam>;
        using CosType = std::tuple_element_t<1, TypeParam>;
        using SinType = std::tuple_element_t<2, TypeParam>;

        std::array<SourceType, 8> source{
            SourceType{1.0f}, SourceType{2.0f}, SourceType{3.0f}, SourceType{4.0f},
            SourceType{-1.0f}, SourceType{-2.0f}, SourceType{-3.0f}, SourceType{-4.0f}};

        const std::array<CosType, 2> cosine{CosType{0.0f}, CosType{1.0f}};
        const std::array<SinType, 2> sine{SinType{1.0f}, SinType{0.0f}};
        constexpr std::array expected{-3.0f, 2.0f, 1.0f, 4.0f, 3.0f, -2.0f, -1.0f, -4.0f};

        Math::RoPE(source.data(), cosine.data(), sine.data(), 2, 4);

        for (std::size_t i{}; i < source.size(); ++i)
        {
            const auto actual = static_cast<float>(source[i]);
            EXPECT_FLOAT_EQ(actual, expected[i]);
        }
    }

    TYPED_TEST(MathTripleTest, RoPEDoesNotWriteForInvalidHeadDimension)
    {
        using SourceType = std::tuple_element_t<0, TypeParam>;
        using CosType = std::tuple_element_t<1, TypeParam>;
        using SinType = std::tuple_element_t<2, TypeParam>;

        std::array<SourceType, 3> source{SourceType{1.0f}, SourceType{2.0f}, SourceType{3.0f}};
        const std::array<CosType, 1> cosine{CosType{0.5f}};
        const std::array<SinType, 1> sine{SinType{0.5f}};

        Math::RoPE(source.data(), cosine.data(), sine.data(), 1, 3);

        const auto value0 = static_cast<float>(source[0]);
        const auto value1 = static_cast<float>(source[1]);
        const auto value2 = static_cast<float>(source[2]);

        EXPECT_FLOAT_EQ(value0, 1.0f);
        EXPECT_FLOAT_EQ(value1, 2.0f);
        EXPECT_FLOAT_EQ(value2, 3.0f);
    }

    TYPED_TEST(MathQuadTest, Attention)
    {
        using QType = std::tuple_element_t<0, TypeParam>;
        using KType = std::tuple_element_t<1, TypeParam>;
        using VType = std::tuple_element_t<2, TypeParam>;
        using OutputType = std::tuple_element_t<3, TypeParam>;

        const std::array<QType, 4> q{QType{1.0f}, QType{-1.0f}, QType{1.0f}, QType{-1.0f}};

        const std::array<KType, 4> kCache{
            KType{0.0f}, KType{1.0f},
            KType{1.0f}, KType{0.0f}};

        const std::array<VType, 4> vCache{
            VType{10.0f}, VType{30.0f},
            VType{20.0f}, VType{40.0f}};

        constexpr std::array expected{
            17.31058502f,
            12.68941498f,
            32.68941498f,
            37.31058502f};

        std::vector<OutputType> output(4, OutputType{0.0f});

        Math::Attention(q.data(), kCache.data(), vCache.data(), output.data(), 2, 4, 2, 1);

        const auto tolerance = std::is_same_v<OutputType, lowp::f16> ? 0.02f : 0.00001f;

        for (std::size_t i{}; i < output.size(); ++i)
        {
            const auto actual = static_cast<float>(output[i]);
            EXPECT_NEAR(actual, expected[i], tolerance);
        }
    }

    TYPED_TEST(MathQuadTest, AttentionDoesNotWriteForInvalidHeadGrouping)
    {
        using QType = std::tuple_element_t<0, TypeParam>;
        using KType = std::tuple_element_t<1, TypeParam>;
        using VType = std::tuple_element_t<2, TypeParam>;
        using OutputType = std::tuple_element_t<3, TypeParam>;

        const QType q{1.0f};
        const KType k{1.0f};
        const VType v{1.0f};
        OutputType output{9.0f};

        Math::Attention(&q, &k, &v, &output, 1, 3, 2, 1);

        const auto actual = static_cast<float>(output);
        EXPECT_FLOAT_EQ(actual, 9.0f);
    }

    TEST(MathCheckedMultiplyTest, MultipliesValues)
    {
        EXPECT_EQ(Math::CheckedMultiply(7, 9), 63u);
        EXPECT_EQ(Math::CheckedMultiply(0, std::numeric_limits<std::size_t>::max()), 0u);
    }

    TEST(MathCheckedMultiplyTest, ThrowsOnOverflow)
    {
        EXPECT_THROW(Math::CheckedMultiply(std::numeric_limits<std::size_t>::max(), 2), std::overflow_error);
    }
}
