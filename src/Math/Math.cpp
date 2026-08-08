#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "Math/Math.hpp"

#if HAVE_AVX2_SUPPORT
#include <immintrin.h>
#endif

namespace Math
{
    std::uint16_t Float32ToFloat16(const float value) noexcept
    {
        const std::uint32_t bits = std::bit_cast<std::uint32_t>(value); // NOLINT(*-use-auto)
        const std::uint32_t sign = (bits >> 16) & 0x8000u;
        const std::uint32_t exponent = (bits >> 23) & 0xFFu;
        std::uint32_t mantissa = bits & 0x7FFFFFu;

        if (exponent == 0xFFu)
        {
            if (mantissa == 0)
                return static_cast<std::uint16_t>(sign | 0x7C00u);

            return static_cast<std::uint16_t>(sign | 0x7E00u);
        }

        auto halfExponent = static_cast<std::int32_t>(exponent) - 127 + 15;

        if (halfExponent >= 31)
            return static_cast<std::uint16_t>(sign | 0x7C00u);

        if (halfExponent <= 0)
        {
            if (halfExponent < -10)
                return static_cast<std::uint16_t>(sign);

            mantissa |= 0x800000u;

            const auto shift = static_cast<std::uint32_t>(14 - halfExponent);
            std::uint32_t halfMantissa = mantissa >> shift;
            const std::uint32_t remainderMask = (1u << shift) - 1u;
            const std::uint32_t remainder = mantissa & remainderMask;
            const std::uint32_t halfway = 1u << (shift - 1u);

            if (remainder > halfway || (remainder == halfway && (halfMantissa & 1u) != 0))
                ++halfMantissa;

            return static_cast<std::uint16_t>(sign | halfMantissa);
        }

        std::uint32_t halfMantissa = mantissa >> 13;
        const std::uint32_t remainder = mantissa & 0x1FFFu;

        if (remainder > 0x1000u || (remainder == 0x1000u && (halfMantissa & 1u) != 0))
        {
            ++halfMantissa;

            if (halfMantissa == 0x400u)
            {
                halfMantissa = 0;
                ++halfExponent;

                if (halfExponent >= 31)
                    return static_cast<std::uint16_t>(sign | 0x7C00u);
            }
        }

        return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(halfExponent) << 10) | halfMantissa);
    }

    float Float16ToFloat32(const std::uint16_t value) noexcept
    {
        const std::uint32_t sign = (value & 0x8000u) << 16;
        const std::uint32_t exponent = (value >> 10) & 0x1Fu;
        std::uint32_t mantissa = value & 0x03FFu;
        std::uint32_t bits{};

        if (exponent == 0)
        {
            if (mantissa == 0)
            {
                bits = sign;
            }
            else
            {
                std::int32_t normalizedExponent = -14;

                while ((mantissa & 0x0400u) == 0)
                {
                    mantissa <<= 1;
                    --normalizedExponent;
                }

                mantissa &= 0x03FFu;
                const auto floatExponent = static_cast<std::uint32_t>(normalizedExponent + 127);
                bits = sign | (floatExponent << 23) | (mantissa << 13);
            }
        }
        else if (exponent == 0x1Fu)
        {
            bits = sign | 0x7F800000u | (mantissa << 13);

            if (mantissa != 0)
                bits |= 0x00400000u;
        }
        else
        {
            const std::uint32_t floatExponent = exponent + (127u - 15u);
            bits = sign | (floatExponent << 23) | (mantissa << 13);
        }

        return std::bit_cast<float>(bits);
    }

    void ConvertFloat32ToFloat16(const float* source, std::uint16_t* destination, const std::size_t elementCount)
    {
        std::size_t i{};

#if HAVE_AVX2_SUPPORT
        for (; i + 8 <= elementCount; i += 8)
        {
            const __m256 values = _mm256_loadu_ps(source + i);
            const __m128i halfValues = _mm256_cvtps_ph(values, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(destination + i), halfValues);
        }
#endif

        for (; i < elementCount; ++i)
            destination[i] = Float32ToFloat16(source[i]);
    }

    void ConvertFloat16ToFloat32(const std::uint16_t* source, float* destination, const std::size_t elementCount)
    {
        std::size_t i{};

#if HAVE_AVX2_SUPPORT
        for (; i + 8 <= elementCount; i += 8)
        {
            const __m128i halfValues = _mm_loadu_si128(reinterpret_cast<const __m128i*>(source + i));
            const __m256 values = _mm256_cvtph_ps(halfValues);
            _mm256_storeu_ps(destination + i, values);
        }
#endif

        for (; i < elementCount; ++i)
            destination[i] = Float16ToFloat32(source[i]);
    }

    void Embedding(const std::uint16_t* embeddingTable, const std::int32_t* tokenIds, float* output,
                   const std::size_t tokenCount, const std::size_t vocabularySize, const std::size_t hiddenSize)
    {
        for (std::size_t t{}; t < tokenCount; ++t)
        {
            const auto token = tokenIds[t];

            if (token < 0)
                continue;

            const auto tokenIndex = static_cast<std::size_t>(token);

            if (tokenIndex >= vocabularySize)
                continue;

            const auto sourceOffset = tokenIndex * hiddenSize;
            const auto destinationOffset = t * hiddenSize;

            ConvertFloat16ToFloat32(embeddingTable + sourceOffset, output + destinationOffset, hiddenSize);
        }
    }

    void Linear(const std::uint16_t* matrix, const float* input, const float* bias, float* output,
                const std::size_t rows, const std::size_t columns)
    {
#pragma omp parallel for schedule(static)
        for (std::int64_t row = 0; row < static_cast<std::int64_t>(rows); ++row)
        {
            const auto* matrixRow = matrix + static_cast<std::size_t>(row) * columns;

#if HAVE_AVX2_SUPPORT
            __m256 accumulator = _mm256_setzero_ps();

            std::size_t column{};
            for (; column + 8 <= columns; column += 8)
            {
                const auto inputVec = _mm256_loadu_ps(input + column);
                const auto matrixHalf = _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(matrixRow + column));
                const auto matrixVec = _mm256_cvtph_ps(matrixHalf);

                accumulator = _mm256_fmadd_ps(matrixVec, inputVec, accumulator);
            }

            alignas(32) float values[8];
            _mm256_store_ps(values, accumulator);

            auto sum = bias[row];

            for (const auto value : values)
                sum += value;

            for (; column < columns; ++column)
                sum += Float16ToFloat32(matrixRow[column]) * input[column];

            output[row] = sum;
#else
            auto sum = bias[row];

            for (std::size_t column{}; column < columns; ++column)
                sum += Float16ToFloat32(matrixRow[column]) * input[column];

            output[row] = sum;
#endif
        }
    }

    void RMSNorm(const float* x, const std::uint16_t* weight, const float epsilon, float* y,
                 const std::size_t elementCount)
    {
        if (elementCount == 0)
            return;

        float meanSquare{};

        for (std::size_t i{}; i < elementCount; ++i)
            meanSquare += x[i] * x[i];

        meanSquare /= static_cast<float>(elementCount);
        const float inverseRms = 1.0f / std::sqrt(meanSquare + epsilon);

        std::size_t i{};

#if HAVE_AVX2_SUPPORT
        const __m256 inverseRmsVec = _mm256_set1_ps(inverseRms);

        for (; i + 8 <= elementCount; i += 8)
        {
            const __m256 xVec = _mm256_loadu_ps(x + i);
            const __m128i weightHalf = _mm_loadu_si128(reinterpret_cast<const __m128i*>(weight + i));
            const __m256 weightVec = _mm256_cvtph_ps(weightHalf);
            const __m256 normalized = _mm256_mul_ps(xVec, inverseRmsVec);
            const __m256 result = _mm256_mul_ps(weightVec, normalized);
            _mm256_storeu_ps(y + i, result);
        }
#endif

        for (; i < elementCount; ++i)
            y[i] = Float16ToFloat32(weight[i]) * (x[i] * inverseRms);
    }

    void Add(const float* a, const float* b, float* output, const std::size_t elementCount)
    {
#pragma omp parallel for
        for (std::int64_t i = 0; i < static_cast<std::int64_t>(elementCount); ++i)
            output[i] = a[i] + b[i];
    }

    void Multiply(const float* a, const float* b, float* output, const std::size_t elementCount)
    {
#pragma omp parallel for
        for (std::int64_t i = 0; i < static_cast<std::int64_t>(elementCount); ++i)
            output[i] = a[i] * b[i];
    }

    std::size_t CheckedMultiply(const std::size_t a, const std::size_t b)
    {
        if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a)
            throw std::overflow_error("Tensor size overflow");

        return a * b;
    }

    void SiLU(const float* x, float* output, const std::size_t elementCount)
    {
#pragma omp parallel for
        for (std::int64_t i = 0; i < static_cast<std::int64_t>(elementCount); ++i)
        {
            const auto x_i = x[i];
            output[i] = x_i / (1.0f + std::exp(-x_i));
        }
    }

    void RoPE(float* source, const std::size_t position, const std::size_t headCount,
              const std::size_t headDimension, const float theta)
    {
        if (headDimension == 0 || headDimension % 2 != 0)
            return;

        const auto halfDimension = headDimension / 2;

        for (std::size_t h{}; h < headCount; ++h)
        {
            const auto headOffset = h * headDimension;

            for (std::size_t p{}; p < halfDimension; ++p)
            {
                const auto exponent = static_cast<float>(2 * p) / static_cast<float>(headDimension);
                const auto inverseFrequency = 1.0f / std::pow(theta, exponent);
                const auto angle = static_cast<float>(position) * inverseFrequency;
                const auto cosAngle = std::cos(angle);
                const auto sinAngle = std::sin(angle);
                const auto firstIndex = headOffset + p;
                const auto secondIndex = headOffset + halfDimension + p;
                const auto first = source[firstIndex];
                const auto second = source[secondIndex];

                source[firstIndex] = first * cosAngle - second * sinAngle;
                source[secondIndex] = first * sinAngle + second * cosAngle;
            }
        }
    }

    void Attention(const float* q, const float* kCache, const float* vCache, float* output,
                   const std::size_t validTokenCount, const std::size_t attentionHeadCount,
                   const std::size_t keyValueHeadCount, const std::size_t headDimension)
    {
        if (validTokenCount == 0 || attentionHeadCount == 0 || keyValueHeadCount == 0 || headDimension == 0)
            return;

        if (attentionHeadCount % keyValueHeadCount != 0)
            return;

        const auto groupSize = attentionHeadCount / keyValueHeadCount;
        const auto scale = 1.0f / std::sqrt(static_cast<float>(headDimension));

        std::vector<float> scores(validTokenCount);

        for (std::size_t h{}; h < attentionHeadCount; ++h)
        {
            const auto kvHead = h / groupSize;
            const auto qOffset = h * headDimension;
            auto maximumScore = -std::numeric_limits<float>::infinity();

            for (std::size_t t{}; t < validTokenCount; ++t)
            {
                const auto kOffset = (t * keyValueHeadCount + kvHead) * headDimension;

                float score_h_t{};

                for (std::size_t d{}; d < headDimension; ++d)
                    score_h_t += q[qOffset + d] * kCache[kOffset + d];

                score_h_t *= scale;

                scores[t] = score_h_t;

                maximumScore = std::fmaxf(maximumScore, score_h_t);
            }

            float probabilitySum{};

            for (std::size_t t{}; t < validTokenCount; ++t)
            {
                scores[t] = std::exp(scores[t] - maximumScore);

                probabilitySum += scores[t];
            }

            const auto inverseProbabilitySum = 1.0f / probabilitySum;
            const auto outputOffset = h * headDimension;

            for (std::size_t d{}; d < headDimension; ++d)
                output[outputOffset + d] = 0.0f;

            for (std::size_t t{}; t < validTokenCount; ++t)
            {
                const auto probability_h_t = scores[t] * inverseProbabilitySum;
                const auto vOffset = (t * keyValueHeadCount + kvHead) * headDimension;

                for (std::size_t d{}; d < headDimension; ++d)
                    output[outputOffset + d] += probability_h_t * vCache[vOffset + d];
            }
        }
    }

    void CopyToCache(const float* source, float* cache, const std::size_t position, const std::size_t elementCount)
    {
        const auto cacheOffset = position * elementCount;

        for (std::size_t i{}; i < elementCount; ++i)
            cache[cacheOffset + i] = source[i];
    }
}
